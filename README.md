<div align="center">

<img src="exposure_profile.png" alt="Counterparty Exposure Profile (EPE / PFE)" width="700"/>

<img src="xva_comparison.png" alt="Portfolio XVA — Base vs Stress" width="700"/>

# Cross-Asset XVA & Exposure Engine

**Runnable Monte Carlo pipeline linking front-office exposure analytics to XVA**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-vectorised-013243?logo=numpy&logoColor=white)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/SciPy-stats-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-plots-11557C?logo=python&logoColor=white)](https://matplotlib.org)
[![Data: FRED](https://img.shields.io/badge/Data-FRED-1f77b4)](https://fred.stlouisfed.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[Quick Start](#quick-start) · [Mathematical Theory](#mathematical-theory) · [Architecture](#architecture) · [Pipeline Walkthrough](#pipeline-walkthrough) · [Stress Testing](#stress-testing) · [Data Sources](#data-sources)

</div>

---

## Overview

An end-to-end counterparty-credit-risk system that simulates a **correlated cross-asset market**, prices a portfolio of OTC derivatives along every path, applies **CSA collateral mechanics** to build netted exposure profiles, and rolls those profiles up into **CVA, FVA and MVA** — all from a single `python main.py` call.

The engine is deliberately compact and dependency-light so the full path from market calibration → simulation → exposure → XVA is readable in one sitting, while still modelling the pieces that matter in production: correlation, antithetic variance reduction, netting sets, thresholds, minimum transfer amounts, and margin-period-of-risk.

### Key Capabilities

| Module | What it does |
|--------|-------------|
| **Market Data** | FRED calibration of short rate, equity vol and FX vol; correlation matrix; scenario shocks |
| **Monte Carlo Engine** | Correlated 3-factor simulation (Hull-White rates, equity GBM, FX GBM) with Cholesky coupling + antithetic paths |
| **Instruments** | Interest-rate swap, European option (Black-Scholes), FX forward, simplified cross-currency swap |
| **Exposure Engine** | Mark-to-market cube, netting sets, CSA collateral (threshold + MTA + call frequency + MPOR lag) |
| **Exposure Metrics** | Expected Positive Exposure (EPE) and 95% Potential Future Exposure (PFE) |
| **XVA Engine** | CVA (hazard-rate default), FVA (funding spread) and MVA (initial-margin proxy funded over time) |
| **Stress Testing** | Base vs. stressed scenario across rates, vols and credit hazard |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (base + stress scenarios)
python main.py
```

`main.py` calibrates the market from FRED, simulates both a base and a stressed scenario, prices the portfolio along every path, and writes two publication-ready figures:

```
exposure_profile.png   # EPE / PFE(95%) for the corporate netting set, base vs stress
xva_comparison.png     # Portfolio CVA / FVA / MVA / Total, base vs stress
```

> If FRED is unreachable the market data falls back to sensible hard-coded defaults, so the pipeline always runs offline.

---

## Mathematical Theory

### 1 · Correlated Cross-Asset Simulation

Three risk factors are evolved jointly on a shared time grid. Correlated Brownian increments are generated from the Cholesky factor $L$ of the correlation matrix $C$ ($C = LL^\top$):

$$d\mathbf{W}_t = L\,\mathbf{z}_t\,\sqrt{\Delta t}, \qquad \mathbf{z}_t \sim \mathcal{N}(\mathbf{0}, I_3)$$

with the factor correlation

$$C = \begin{pmatrix} 1.0 & -0.3 & 0.2 \\ -0.3 & 1.0 & 0.4 \\ 0.2 & 0.4 & 1.0 \end{pmatrix} \quad \text{(order: rates, equity, FX)}$$

**Antithetic variance reduction** — half the paths are drawn as $\mathbf{z}$ and mirrored as $-\mathbf{z}$, halving the standard error of symmetric estimators at no extra path cost.

**Rates — Hull-White / mean-reverting short rate:**

$$dr_t = a\,(r_0 - r_t)\,dt + \sigma_r\,dW_t^{r}$$

**Equity — geometric Brownian motion under the stochastic short rate:**

$$\frac{dS_t}{S_t} = r_t\,dt + \sigma_S\,dW_t^{S} \;\;\Longrightarrow\;\; S_{t+\Delta t} = S_t \exp\!\left[\left(r_t - \tfrac{1}{2}\sigma_S^2\right)\Delta t + \sigma_S\,dW_t^{S}\right]$$

**FX — geometric Brownian motion (single-curve proxy):**

$$X_{t+\Delta t} = X_t \exp\!\left[\left(r_t - \tfrac{1}{2}\sigma_X^2\right)\Delta t + \sigma_X\,dW_t^{X}\right]$$

**Pathwise discount factor** — accumulated from the simulated short rate using the step-midpoint average $\bar r_t$:

$$D(0,t) = \prod_{k=0}^{t-1} e^{-\bar r_k \,\Delta t}, \qquad \bar r_k = \tfrac{1}{2}\left(r_k + r_{k+1}\right)$$

Stochastic discount factors between any two grid points follow from the ratio $P(t,u) = D(0,u)/D(0,t)$, which is what the instruments use for pricing.

### 2 · Instruments

**Interest-rate swap** — valued from pathwise zero-coupon bond ratios. With annuity $A(t) = \alpha \sum_i P(t,u_i)$ over the remaining payment dates:

$$V_{\text{swap}}(t) = N\Big[\underbrace{\big(1 - P(t,T)\big)}_{\text{float leg}} - \underbrace{k\,A(t)}_{\text{fixed leg}}\Big], \qquad V_{\text{payer}} = -V_{\text{receiver}}$$

**European option** — Black-Scholes with pathwise spot $S_t$ and short rate $r_t$, time to expiry $\tau = T - t$:

$$C = S_t\,\Phi(d_1) - K\,e^{-r_t\tau}\,\Phi(d_2), \qquad d_{1,2} = \frac{\ln(S_t/K) + \left(r_t \pm \tfrac{1}{2}\sigma^2\right)\tau}{\sigma\sqrt{\tau}}$$

**FX forward** — foreign-notional forward struck at $K$:

$$V_{\text{fwd}}(t) = N\big(F_t - K\big)e^{-r_t\tau}, \qquad F_t = X_t\,e^{r_t\tau}$$

**Cross-currency swap (simplified)** — domestic and foreign fixed legs plus a final notional exchange, FX-converted along each path.

Each instrument also exposes an **initial-margin proxy** (schedule-style risk weights: DV01 × weight for the swap, delta × spot × weight for the option, notional × spot × weight for FX/XCCY) that feeds the MVA calculation.

### 3 · Exposure & CSA Collateral

For each netting set the netted mark-to-market is $V^{\text{NS}}_t = \sum_{i \in \text{NS}} V_i(t)$. Collateral is posted on a call schedule against the MtM observed one **margin-period-of-risk** ago:

$$\text{Target}_t = \max\!\left(V^{\text{NS}}_{t - \text{MPOR}} - H,\; 0\right)$$

A margin call is only executed when the required move exceeds the **minimum transfer amount**:

$$\text{Collateral}_t = \begin{cases} \text{Target}_t & \text{if } \lvert \text{Target}_t - \text{Collateral}_{t^-}\rvert > \text{MTA} \\ \text{Collateral}_{t^-} & \text{otherwise} \end{cases}$$

Collateralised exposure and its metrics are then

$$E_t = \max\!\left(V^{\text{NS}}_t - \text{Collateral}_t,\; 0\right), \qquad \text{EPE}_t = \mathbb{E}[E_t], \qquad \text{PFE}_t = Q_{0.95}(E_t)$$

where $H$ is the threshold and $Q_{0.95}$ is the 95th path-wise percentile.

### 4 · XVA

With a constant hazard rate $\lambda$, survival $S(t) = e^{-\lambda t}$ and marginal default probability $\text{dPD}_t = S(t) - S(t{+}\Delta t)$, the valuation adjustments integrate the exposure profile against discount factors $\bar D_t$ (mean pathwise DF) on the step midpoints:

$$\text{CVA} = (1 - R)\sum_t \overline{\text{EPE}}_t\,\bar D_t\,\text{dPD}_t$$

$$\text{FVA} = s_f \sum_t \overline{\text{EPE}}_t\,\bar D_t\,\Delta t, \qquad \text{MVA} = s_{\text{im}} \sum_t \overline{\text{IM}}_t\,\bar D_t\,\Delta t$$

where $R$ is recovery, $s_f$ the funding spread and $s_{\text{im}}$ the initial-margin funding spread. The portfolio total is $\text{CVA} + \text{FVA} + \text{MVA}$ summed across netting sets.

---

## Architecture

```
                         python main.py
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ src/market_data.py                                            │
│   FRED calibration (DGS3MO, SP500, DEXUSEU) · corr · shocks   │
└──────────────────────────────┬───────────────────────────────┘
                               │ MarketData
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ src/engine.py — MonteCarloEngine                              │
│   Cholesky-correlated 3-factor paths + antithetic             │
│   Hull-White r · equity GBM S · FX GBM X · discount factors   │
└──────────────────────────────┬───────────────────────────────┘
                               │ ctx = {time, r, S, FX, df}
                               ▼
┌────────────────────────────┐   prices    ┌────────────────────┐
│ src/exposure.py            │◄───────────►│ src/instruments.py │
│   ExposureEngine · CSA     │  along all  │   IRS · EqOption   │
│   MtM cube · netting sets  │    paths    │   FXForward · XCCY │
│   collateral · EPE · PFE   │             │   + IM proxies     │
└──────────────┬─────────────┘             └────────────────────┘
               │ exposure profiles + IM
               ▼
┌──────────────────────────────────────────────────────────────┐
│ src/xva.py — XVAEngine                                        │
│   CVA (hazard) · FVA (funding) · MVA (IM funded over time)    │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
                exposure_profile.png · xva_comparison.png
```

---

## Pipeline Walkthrough

### 1 · Calibrate the market

```python
from src.market_data import MarketData

md = MarketData()
md.calibrate_from_fred()
# Fetching FRED data for simple calibration...
# Calibrated: r0=... hw_sigma=... eq_vol=... fx_vol=...
```

Three FRED series drive the calibration: the 3-month T-bill rate (`DGS3MO`) sets the initial short rate and its annualised vol, the S&P 500 (`SP500`) sets equity spot and realised vol, and the USD/EUR rate (`DEXUSEU`) sets FX spot and vol.

### 2 · Simulate correlated paths

```python
from src.engine import MonteCarloEngine

config = {"n_sims": 600, "n_steps": 80, "time_horizon": 5.0, "antithetic": True}
ctx = MonteCarloEngine(md, config).simulate(seed=42)
# ctx = {time, r, S, FX, df}  — each an (n_sims × n_steps) grid
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `n_sims` | 600 | Monte Carlo paths (antithetic) |
| `n_steps` | 80 | Time steps over the horizon |
| `time_horizon` | 5.0 | Simulation horizon (years) |
| `antithetic` | `True` | Mirror half the normals for variance reduction |

### 3 · Build the portfolio & netting sets

Two counterparties with very different CSAs:

| Netting set | CSA | Trades |
|-------------|-----|--------|
| `NS_BANK` | Near-infinite threshold, no MPOR (≈ uncollateralised) | IRS · FX forward · cross-currency swap |
| `NS_CORP` | Threshold 25,000 · MTA 5,000 · daily calls · 10-day MPOR | European equity call |

```python
from src.exposure import ExposureEngine, CSA
from src.instruments import InterestRateSwap, FXForward, XCCYSwapSimple, EuropeanOption

exp = ExposureEngine(ctx)
csa_corp = CSA(threshold=25_000.0, mta=5_000.0, call_frequency_days=1, mpor_days=10)
exp.add_trade(EuropeanOption("EQOPT1", "S", 4100.0, 2.0, "call"), "NS_CORP", csa_corp)
exposure = exp.run()
# Pricing 4 trades over 81 steps...
```

### 4 · Roll up to XVA

```python
from src.xva import XVAEngine

xva = XVAEngine(ctx, exposure).compute(
    hazard_rate=0.02, recovery=0.4,
    funding_spread=0.01, im_funding_spread=0.01,
)
print(xva["PORTFOLIO"])
# {'CVA': ..., 'FVA': ..., 'MVA': ..., 'Total': ...}
```

The collateralised `NS_CORP` exposure and the raw `NS_BANK` exposure produce visibly different EPE/PFE shapes — the threshold + MTA + MPOR mechanics cap and lag the corporate profile, which is exactly what `exposure_profile.png` illustrates.

---

## Stress Testing

`main.py` reruns the entire pipeline under a combined market-and-credit stress and plots the two scenarios side by side:

| Shock | Base | Stress |
|-------|------|--------|
| Short rate $r_0$ | calibrated | **+100 bps** |
| Equity vol $\sigma_S$ | calibrated | **× 1.5** |
| FX vol $\sigma_X$ | calibrated | **× 1.5** |
| Hazard rate $\lambda$ | 2% | **3%** |

```python
def shock(md):
    md.shock(dr=0.01, eq_vol_mult=1.5, fx_vol_mult=1.5)
```

Higher vols widen the exposure distribution (larger PFE), while the higher hazard rate and wider exposures together lift CVA — the `xva_comparison.png` bar chart makes the base-vs-stress gap explicit across CVA, FVA, MVA and Total.

---

## Data Sources

| Data | Source | Series |
|------|--------|--------|
| Short rate | FRED | `DGS3MO` — 3-Month Treasury Constant Maturity |
| Equity spot & vol | FRED | `SP500` — S&P 500 index |
| FX spot & vol | FRED | `DEXUSEU` — U.S. / Euro exchange rate |

Calibration is intentionally lightweight (last level + annualised realised vol). If the FRED request fails, the engine prints a warning and continues with built-in defaults.

---

## Tech Stack

```
Language        Purpose                        Key libraries
──────────────  ─────────────────────────────  ─────────────────────────
Python 3.10+    Simulation, pricing, XVA       numpy, scipy
                Market data / calibration      pandas, pandas_datareader
                Visualisation                  matplotlib
```

---

## Tests

```bash
python -m unittest discover -s tests -v
```

| Test | Coverage |
|------|----------|
| `test_fx_forward_sign` | FX forward has correct sign when spot > strike |
| `test_option_payoff_nonnegative` | European option payoff is non-negative at expiry |

---

## Project Structure

```
xva-exposure-engine/
├── README.md              ← You are here
├── main.py                Scenario driver: base vs stress, plots
├── requirements.txt       Python dependencies
├── src/
│   ├── market_data.py     Market data, FRED calibration, shocks
│   ├── engine.py          Correlated Monte Carlo path generator
│   ├── instruments.py     IRS, European option, FX forward, XCCY swap + IM proxies
│   ├── exposure.py        Netting sets, CSA collateral, EPE / PFE
│   └── xva.py             CVA / FVA / MVA from the exposure profile
├── tests/
│   └── test_basics.py     Sanity checks
├── exposure_profile.png   Generated: EPE / PFE, base vs stress
└── xva_comparison.png     Generated: portfolio XVA, base vs stress
```

---

## References

- Gregory, J. (2015). *The xVA Challenge: Counterparty Credit Risk, Funding, Collateral, and Capital.* 3rd ed., Wiley.
- Green, A. (2015). *XVA: Credit, Funding and Capital Valuation Adjustments.* Wiley.
- Hull, J. & White, A. (1990). *Pricing Interest-Rate-Derivative Securities.* Review of Financial Studies, 3(4), 573-592.
- Brigo, D. & Mercurio, F. (2006). *Interest Rate Models — Theory and Practice.* 2nd ed., Springer.
- Basel Committee on Banking Supervision (2015). *Margin requirements for non-centrally cleared derivatives.*

---

<div align="center">

*Built with Python, NumPy, and a healthy respect for counterparty risk.*

</div>
