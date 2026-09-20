"""Independent analytical/MC checks and measured variance reduction.

Exit nonzero on failed pricing/martingale checks. Measurements use existing
equity options and bond/spot controls, not a new product or notebook.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from src.market_data import MarketData
from src.engine import MonteCarloEngine
from src.instruments import EuropeanOption, InterestRateSwap
from src.risk import initial_context, independent_samples, mean_se, swap_risk


def sample(md, n=8192, steps=365, seed=17, antithetic=True):
    ctx = MonteCarloEngine(md, {"n_sims":n,"n_steps":steps,"time_horizon":1.,
                                "antithetic":antithetic}).simulate(seed)
    option = EuropeanOption("benchmark","S",md.eq_spot,1.)
    discounted_payoff = ctx["df"][:,-1]*option.payoff(ctx["S"][:,-1])
    control = ctx["df"][:,-1]*ctx["S"][:,-1]
    return ctx, discounted_payoff, control


def run(md):
    checks = []
    def check(name, observed, expected, se=0., atol=1e-8):
        error = abs(observed-expected)
        checks.append({"name":name,"observed":float(observed),"expected":float(expected),
                       "standard_error":float(se),"absolute_error":float(error),
                       "tolerance":float(5*se+atol),"passed":bool(error <= 5*se+atol)})
    model = md.rate_model()
    node_error = max(abs(model.bond(0.,float(t))-md.curve.discount(t)) for t in md.curve.tenors)
    check("Initial curve node fit",node_error,0)
    swap = InterestRateSwap("s",1e6,0,4)
    swap.fixed_rate = swap.par_rate(md.curve)
    check("Par swap NPV",swap.value(0,initial_context(md))[0],0,atol=1e-7)
    option = EuropeanOption("o","S",md.eq_spot,1)
    analytic = float(option.value(0,initial_context(md))[0])
    ctx,payoff,control = sample(md)
    mean,se = mean_se(payoff,True)
    check("European option MC / analytical",mean,analytic,se,atol=md.eq_spot*2e-5)
    for name, values, expected in [
        ("Discount factor martingale",ctx["df"][:,-1],md.curve.discount(1)),
        ("Conditional bond martingale",ctx["df"][:,-1]*model.bond(1,4,ctx["x"][:,-1]),md.curve.discount(4)),
        ("Discounted equity martingale",control,md.eq_spot*np.exp(-md.eq_dividend)),
        ("Discounted FX martingale",ctx["df"][:,-1]*ctx["FX"][:,-1],md.fx_spot*np.exp(-md.foreign_rate))]:
        mean,se = mean_se(values,True)
        check(name,mean,expected,se,atol=abs(expected)*2e-5)
    del ctx
    convergence = []
    for n in [512,2048,8192]:
        _,p,_ = sample(md,n=n)
        mean,se = mean_se(p,True)
        convergence.append({"paths":n,"estimate":mean,"pair_standard_error":se,"analytical":analytic})
    time_steps = []
    for steps in [26,52,365]:
        c,p,_ = sample(md,steps=steps)
        mean,se = mean_se(p,True)
        time_steps.append({"steps":steps,"estimate":mean,"pair_standard_error":se,"analytical":analytic})
        check(f"Time-grid option consistency ({steps} steps)",mean,analytic,se,atol=md.eq_spot*2e-4)
        del c
    # Fit beta on independent pilot draws; evaluate on separate seeds.
    _,pilot,pc = sample(md,n=2048,steps=52,seed=9101)
    pilot,pc = independent_samples(pilot,True), independent_samples(pc,True)
    beta = float(np.cov(pilot,pc,ddof=1)[0,1]/np.var(pc,ddof=1))
    expected_control = md.eq_spot*np.exp(-md.eq_dividend)
    measurements = {"plain":[],"antithetic":[],"antithetic_control":[],"crn_difference":[],"independent_difference":[]}
    bumped = md.shocked(dr=1e-4)
    # Keep the original strike when the market curve is bumped.
    bumped_target = float(option.value(0,initial_context(bumped))[0])-analytic
    for seed in range(20,36):
        _,plain,_ = sample(md,n=2048,steps=52,seed=seed,antithetic=False)
        _,anti,ctrl = sample(md,n=2048,steps=52,seed=seed)
        _,bump,_ = sample(bumped,n=2048,steps=52,seed=seed)
        _,other,_ = sample(bumped,n=2048,steps=52,seed=seed+1000)
        measurements["plain"].append(float(plain.mean()))
        measurements["antithetic"].append(float(anti.mean()))
        measurements["antithetic_control"].append(float((anti-beta*(ctrl-expected_control)).mean()))
        measurements["crn_difference"].append(float((bump-anti).mean()))
        measurements["independent_difference"].append(float(other.mean()-anti.mean()))
    variances = {k:float(np.var(v,ddof=1)) for k,v in measurements.items()}
    cv_mean, cv_se = mean_se(np.array(measurements["antithetic_control"]))
    check("Independent-pilot control variate price consistency",cv_mean,analytic,cv_se,atol=md.eq_spot*2e-4)
    crn_mean, crn_se = mean_se(np.array(measurements["crn_difference"]))
    check("CRN curve bump vs analytical",crn_mean,bumped_target,crn_se,atol=md.eq_spot*2e-7)
    quote_checks = {}
    for label, quote in md.iv_quotes.items():
        idx = 1 if label == "equity" else 2
        sigma = md.eq_vol if label == "equity" else md.fx_vol
        recovered = np.sqrt(model.option_variance(quote["expiry_years"],sigma,md.corr[0,idx])/quote["expiry_years"])
        check(label+" input IV reproduced (FX is a futures proxy)",recovered,quote["iv"],atol=1e-10)
        quote_checks[label] = {"expiry_years":quote["expiry_years"],"iv":quote["iv"],
                               "recovered_iv":float(recovered)}
    return {"input_kind":md.metadata.get("kind"),"asof":md.metadata.get("asof"),
            "passed":all(c["passed"] for c in checks),"checks":checks,
            "swap_risk":swap_risk(swap,md),"path_count_convergence":convergence,
            "time_step_convergence":time_steps,"market_iv_checks":quote_checks,
            "variance_reduction":{"replications":16,"paths_per_replication":2048,
                "pilot_paths":2048,"control_beta":beta,"estimator_variances":variances,
                "plain_over_antithetic":variances["plain"]/variances["antithetic"],
                "antithetic_over_control":variances["antithetic"]/variances["antithetic_control"],
                "independent_over_crn":variances["independent_difference"]/variances["crn_difference"],
                "note":"Equal path counts for main estimators; control pilot is additional cost. Empirical ratios, not guarantees."}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot")
    p.add_argument("--output",default="output/validation.json")
    args = p.parse_args()
    result = run(MarketData(args.snapshot))
    path = Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2))
    print(json.dumps({"passed":result["passed"],"checks":len(result["checks"]),
                      "failures":[c for c in result["checks"] if not c["passed"]],
                      "variance_reduction":result["variance_reduction"],
                      "report":str(path)},indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
