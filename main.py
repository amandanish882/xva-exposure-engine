"""Offline by default; pass --snapshot data/local_market.json for calibrated inputs."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.market_data import MarketData
from src.engine import MonteCarloEngine
from src.instruments import InterestRateSwap, EuropeanOption, FXForward, XCCYSwapSimple
from src.exposure import ExposureEngine, CSA
from src.xva import XVAEngine
from src.risk import swap_risk


def portfolio(md):
    swap = InterestRateSwap("IRS1", 1_000_000, 0., 4.)
    swap.fixed_rate = swap.par_rate(md.curve)
    option_T = md.iv_quotes.get("equity", {}).get("expiry_years", 2.)
    if option_T > 5:
        raise ValueError("Demo option outside five-year horizon")
    forward_strike = md.fx_spot*np.exp(-md.foreign_rate)/md.curve.discount(1.)
    # Identical swap on a second counterparty makes collateral effects visible.
    return [
        (swap, "NS_BANK", CSA(threshold=1e18, mpor_days=0)),
        (FXForward("FXFWD1", 500_000, forward_strike, 1.), "NS_BANK", CSA(threshold=1e18, mpor_days=0)),
        (XCCYSwapSimple("XCCY1", 1_000_000, 1_000_000/md.fx_spot, 0.035, 0.02, 3.),
         "NS_BANK", CSA(threshold=1e18, mpor_days=0)),
        (InterestRateSwap("IRS_CORP", 1_000_000, swap.fixed_rate, 4.), "NS_CORP", CSA(threshold=25_000, mta=5_000)),
        (EuropeanOption("EQOPT1", "S", md.eq_spot, option_T, units=100), "NS_CORP", CSA(threshold=25_000, mta=5_000))
    ]


def run_scenario(md, trades, n_sims=2000, seed=42, hazard=0.02):
    events = np.concatenate([np.asarray(tr.event_times) for tr, _, _ in trades])
    ctx = MonteCarloEngine(md, {"n_sims": n_sims, "time_horizon": 5.,
                                "event_times": events, "antithetic": True}).simulate(seed)
    engine = ExposureEngine(ctx)
    for tr, ns, csa in trades:
        engine.add_trade(tr, ns, csa)
    exposure = engine.run()
    xva = XVAEngine(ctx, exposure).compute(hazard_rate=hazard)
    return ctx, exposure, xva


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot")
    p.add_argument("--n-sims", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="output")
    args = p.parse_args()
    md = MarketData(args.snapshot)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Input kind: {md.metadata.get('kind')} | asof: {md.metadata.get('asof', 'synthetic fixture')}")
    trades = portfolio(md)  # Fixed contracts: never re-strike during a stress or bump.
    ctx, base, xb = run_scenario(md, trades, args.n_sims, args.seed)
    times = ctx["time"].copy()
    del ctx
    _, stress, xs = run_scenario(md.shocked(dr=.01, eq_vol_mult=1.5, fx_vol_mult=1.5),
                                trades, args.n_sims, args.seed, hazard=.03)
    risk = swap_risk(trades[0][0], md)
    summary = {"data_kind": md.metadata.get("kind"), "asof": md.metadata.get("asof"),
               "curve_source":md.metadata.get("curve_source", "Synthetic zero-node fixture"),
               "curve_bootstrap":md.metadata.get("curve_bootstrap"),
               "paths": args.n_sims, "seed": args.seed, "time_points": len(times),
               "swap_risk": risk, "correlation": md.corr.tolist(), "base": xb, "stress": xs}
    if not np.isfinite([v for group in (xb, xs) for d in group.values() for v in d.values()]).all():
        raise ValueError("Nonfinite XVA result")
    with open(out/"results.json", "w") as f:
        json.dump(summary, f, indent=2)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1,2, figsize=(11,4))
    grid = np.linspace(.001, md.curve.tenors[-1], 800)
    axes[0].plot(grid, 100*md.curve.zero(grid), label="Zero rate")
    axes[0].plot(grid, 100*md.curve.forward(grid), label="Forward", alpha=.7)
    title = "SOFR futures curve (zero convexity)" if md.sofr_bootstrap else "Input discount curve"
    axes[0].set(xlabel="Years", ylabel="Percent", title=title)
    axes[0].legend()
    labels = [f"{float(t):.2f}" for t in risk["key_rate_dv01"]]
    axes[1].bar(labels, list(risk["key_rate_dv01"].values()))
    axes[1].tick_params(axis="x", labelrotation=60, labelsize=8)
    axes[1].set(xlabel="Curve node (years)", ylabel="USD / +1bp", title="Payer swap key-rate DV01")
    fig.tight_layout()
    fig.savefig(out/"curve_and_risk.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1,2, figsize=(11,4))
    for ax, ns in zip(axes, ("NS_BANK", "NS_CORP")):
        for label, values in (("Base", base), ("Stress", stress)):
            ax.plot(times, values[ns]["EPE"], label=label+" EE")
            ax.plot(times, values[ns]["PFE"], "--", label=label+" PFE95", alpha=.8)
        ax.set(title=ns, xlabel="Years", ylabel="USD exposure")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/"exposure_profile.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4))
    labels = ["CVA", "FVA", "MVA", "Total"]
    x = np.arange(4)
    ax.bar(x-.2, [xb["PORTFOLIO"][k] for k in labels], .4, label="Base")
    ax.bar(x+.2, [xs["PORTFOLIO"][k] for k in labels], .4, label="Stress")
    ax.set(xticks=x, xticklabels=labels, ylabel="USD", title="XVA (funding and IM are proxies)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"xva_comparison.png", dpi=140)
    plt.close(fig)
    print(json.dumps({"swap": risk, "base": xb["PORTFOLIO"], "stress": xs["PORTFOLIO"]}, indent=2))


if __name__ == "__main__":
    main()
