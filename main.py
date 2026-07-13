import numpy as np
import matplotlib.pyplot as plt

from src.market_data import MarketData
from src.engine import MonteCarloEngine
from src.instruments import InterestRateSwap, EuropeanOption, FXForward, XCCYSwapSimple
from src.exposure import ExposureEngine, CSA
from src.xva import XVAEngine

def build_portfolio(ctx):
    exp = ExposureEngine(ctx)

    csa_bank = CSA(threshold=1e18, mta=0.0, call_frequency_days=1, mpor_days=0)
    exp.add_trade(InterestRateSwap("IRS1", 1_000_000, 0.04, 4.0, pay_fixed=True), "NS_BANK", csa_bank)
    exp.add_trade(FXForward("FXFWD1", 500_000, 1.12, 1.0), "NS_BANK", csa_bank)
    exp.add_trade(XCCYSwapSimple("XCCY1", notional_dom=1_000_000, notional_for=900_000, dom_fixed=0.035, for_fixed=0.02, maturity=3.0), "NS_BANK", csa_bank)

    csa_corp = CSA(threshold=25_000.0, mta=5_000.0, call_frequency_days=1, mpor_days=10)
    exp.add_trade(EuropeanOption("EQOPT1", "S", 4100.0, 2.0, "call"), "NS_CORP", csa_corp)

    return exp

def run_scenario(name, md_shock=None, hazard_rate=0.02):
    print(f"\n=== Scenario: {name} ===")
    md = MarketData()
    md.calibrate_from_fred()
    if md_shock:
        md_shock(md)

    config = {"n_sims": 600, "n_steps": 80, "time_horizon": 5.0, "antithetic": True}
    ctx = MonteCarloEngine(md, config).simulate(seed=42)
    ctx["md"] = md

    exp = build_portfolio(ctx)
    exposure = exp.run()

    xva = XVAEngine(ctx, exposure).compute(
        hazard_rate=hazard_rate,
        recovery=0.4,
        funding_spread=0.01,
        im_funding_spread=0.01
    )

    print("XVA (Portfolio):", {k: round(xva["PORTFOLIO"][k], 2) for k in xva["PORTFOLIO"]})
    return ctx["time"], exposure, xva

def main():
    t, exp_base, xva_base = run_scenario("Base", md_shock=None, hazard_rate=0.02)

    def shock(md):
        md.shock(dr=0.01, eq_vol_mult=1.5, fx_vol_mult=1.5)
        print("Applied shocks: r0 +1%, eq/fx vols x1.5")

    t2, exp_stress, xva_stress = run_scenario("Stress_RatesVol_Credit", md_shock=shock, hazard_rate=0.03)

    ns = "NS_CORP"
    plt.figure(figsize=(10, 6))
    plt.plot(t, exp_base[ns]["EPE"], label="Base EPE", linewidth=2)
    plt.plot(t, exp_base[ns]["PFE"], label="Base PFE(95%)", linestyle="--")
    plt.plot(t2, exp_stress[ns]["EPE"], label="Stress EPE", linewidth=2)
    plt.plot(t2, exp_stress[ns]["PFE"], label="Stress PFE(95%)", linestyle="--")
    plt.title(f"Exposure Profile - {ns} (CSA: Threshold+MTA+MPOR)")
    plt.xlabel("Time (years)")
    plt.ylabel("Exposure")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("exposure_profile.png")
    print("Saved: exposure_profile.png")

    labels = ["CVA", "FVA", "MVA", "Total"]
    base_vals = [xva_base["PORTFOLIO"][k] for k in labels]
    stress_vals = [xva_stress["PORTFOLIO"][k] for k in labels]

    x = np.arange(len(labels))
    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.2, base_vals, width=0.4, label="Base")
    plt.bar(x + 0.2, stress_vals, width=0.4, label="Stress")
    plt.xticks(x, labels)
    plt.title("Portfolio XVA - Base vs Stress")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("xva_comparison.png")
    print("Saved: xva_comparison.png")

if __name__ == "__main__":
    main()
