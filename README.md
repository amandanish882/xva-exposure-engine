# Cross-Asset XVA & Exposure Engine

A runnable Monte Carlo engine that demonstrates how front-office exposure analytics connect to XVA.

## What it covers
- Cross-asset joint simulation: Rates (Hull-White short rate), Equity (GBM), FX (GBM) with correlation and antithetic variance reduction
- OTC instruments: interest-rate swap, European equity option, FX forward, simplified cross-currency swap proxy
- Exposure: mark-to-market cube, netting sets, CSA collateral (threshold + minimum transfer amount + call frequency + margin-period-of-risk lag)
- Metrics: expected positive exposure (EPE) and 95% potential future exposure (PFE)
- XVA: CVA, FVA, and an MVA proxy (initial-margin proxy funded over time)
- Stress testing: base versus stressed scenario (rates / vol / credit hazard)

## Install and run
```bash
pip install -r requirements.txt
python main.py
```

Running `main.py` prices the portfolio under a base and a stressed scenario, then writes `exposure_profile.png` and `xva_comparison.png`.

## Layout
```
main.py              # scenario driver and plots
src/market_data.py   # market data, FRED calibration, shocks
src/engine.py        # correlated Monte Carlo path generator
src/instruments.py   # IRS, European option, FX forward, XCCY swap
src/exposure.py      # netting sets, CSA collateral, EPE / PFE
src/xva.py           # CVA / FVA / MVA from exposure
tests/test_basics.py # sanity checks
```
