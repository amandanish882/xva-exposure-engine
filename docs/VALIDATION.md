# Validation record

Run on 21 September 2026, Python 3.12, NumPy 2.5.3, SciPy 1.18.1,
pandas 3.0.6 and QuantLib 1.43. This records checks on the upgraded implementation,
not a claim of production model approval or validation of every market proxy.

## Reproduce the public, offline checks

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python validate.py
python main.py
```

- **49 tests passed.** Coverage includes curve nodes/interpolation/negative
  rates, independent Gaussian bond valuation, par swap/sign/DV01, fixed coupon
  schedules and past resets, no future-path lookahead, settled trades,
  FX and option parity, IV recovery/quote filtering/maturity selection,
  download-budget enforcement, realised shock correlation, martingales,
  pair-aware standard errors, time-discretisation bias, collateral timing,
  pathwise XVA discounting and a full daily-grid portfolio run.
- New SOFR coverage includes front-quarter accrued fixings, IMM dates/leap
  years, ACT/360 versus ACT/365, holiday/weekend day weights, negative rates,
  no use of an unpublished same-day fixing, quote repricing, missing/gapped/
  duplicate/insufficient strips, settlement date/status/deletion filters,
  independent QuantLib bootstrap comparison, an end-to-end bootstrapped-curve
  exposure test and the daily/weekly SOFR correlation diagnostic.
- **12/12 synthetic analytical/Monte Carlo checks passed.** Statistical checks
  use a five-standard-error acceptance band plus a stated absolute tolerance.
  These are conservative regression checks, not narrow pricing accuracy guarantees.
- Full base/stress run: **2,000 paths**, five years, daily steps plus events,
  seed 42. No nonfinite XVA values; all remaining trade exposure is zero after
  final maturity. CI additionally runs the demo with 400 paths.

### Selected synthetic results

| Check | Result |
|---|---:|
| Initial curve node fit error | 0 at the supplied nodes |
| Four-year, USD 1m par swap NPV | USD 0 |
| Signed payer parallel DV01 | USD 371.925 per +1bp |
| One-year equity call, analytical | 457.2996 per unit |
| Same call, 8,192-path MC | 454.2910; pair-aware SE 5.6451 |
| Discounted conditional bond error | 0.000001628 |
| Illustrative base CVA / FVA / MVA | USD 2,688.77 / 2,314.00 / 4,690.15 |
| Illustrative base / stress total | USD 9,692.93 / 15,911.01 |

Increasing path counts from 512 to 2,048 to 8,192 gave call-price standard
errors 19.7742, 11.9764 and 5.6451. Realised price errors need not improve
monotonically. The 26-, 52- and 365-step prices are consistent with the analytical
price within sampling uncertainty; they are not a noise-free proof of convergence.
An additional unit test propagates the Euler/trapezoidal Gaussian covariance
deterministically, verifies decreasing discount-factor bias under refinement,
and bounds the daily-grid five-year relative bias below `1e-6` for the fixture.

### Measured variance reduction

Sixteen independent replications, each using 2,048 paths, compared estimators
of the existing one-year equity call. The control coefficient used a separate
2,048-path pilot; pilot cost is **additional**, not included in equal main path counts.

| Estimator | Variance across replication estimates |
|---|---:|
| Plain Monte Carlo | 342.1412 |
| Antithetic | 133.2196 |
| Antithetic plus control variate | 14.4555 |
| +1bp price difference, independent draws | 143.3985 |
| +1bp price difference, common random numbers | 0.0000028002 |

Antithetic gave a 2.57× variance reduction in this experiment, with a further
9.22× from the control. These are **variance**, not standard-error, ratios.
The very large CRN improvement is for a tiny smooth bump whose errors cancel
strongly; it must not be advertised as general exposure-engine acceleration.
The MC bump of 0.247396 agreed with the analytical 0.247324 within the measured
replication error. The control-adjusted price also passed the analytical check.

## Private market-data validation

The explicit market calibration used a **2026-09-11** common FRED date and
156 weekly changes. This is a historical snapshot, not a 21 September live quote.
The new curve used **21 quarterly SR3 contracts**, extending to 17 September
2031 (5.0192 ACT/365 years), plus realised FRED SOFR fixings for the front quarter.
The additional SR3 definition/statistics download was estimated at **USD 0.00665**.
Existing option downloads were reused from the local cache. Estimates are not invoices.

All selected same-date settlement records in this snapshot carried CME's
**actual, preliminary** status. They are explicitly labelled as such; no final
settlement status or executable far-end liquidity is claimed.

| SOFR bootstrap check | Result |
|---|---:|
| Maximum input futures repricing error | 4.27e-14 index-price points |
| Maximum discount difference versus independent QuantLib bootstrap, 1,001 times | 3.18e-13 |
| Convexity adjustment used by both bootstraps | 0 |
| Four-year USD 1m par swap NPV | Less than USD 1e-7 in absolute value |
| Signed payer parallel DV01 | USD 368.080 per +1bp |

Two European near-ATM quote pairs passed the filters, with expiries approximately
1.76 years for equity and 0.98 years for FX. Their IV/carry inputs were recomputed
using the new discount curve, not copied unchanged from the Treasury snapshot.

**36/36 checks passed** with that snapshot: the previous 14 pricing/MC/input-IV
checks, one saved-versus-rebuilt curve check and 21 futures repricing checks.
Black midpoint repricing errors were below `1e-10` in each quote's native
price units. The full 2,000-path base/stress pipeline also completed. Raw quotes,
the local snapshot and its detailed output are not published; run the documented
calibration with your own access to reproduce the market-data workflow.

IV recovery verifies the inversion/conversion algebra, not the economic validity
of substituting a futures-option IV for spot FX. Zero futures convexity,
the futures-versus-OTC-OIS distinction, preliminary settlement status,
timestamp mismatch, expiry conventions and flat-vol extrapolation remain material
limitations: see [data methodology](DATA.md).

The diagnostic also compares SOFR and DGS3MO correlations on the same dates at
daily and weekly frequencies. The matrices differ, including the sign of the
weekly rates/FX estimate. This curve upgrade leaves the existing weekly DGS3MO
simulation correlation unchanged; comparison alone does not justify a new proxy.

## Automation and change control

`.github/workflows/tests.yml` runs unit tests, deterministic-seed numerical
validation and the portfolio smoke test on pushes and pull requests. These jobs
require no credentials, paid downloads or network access to market-data services.
Dependency installation still requires the package index. Market-data validation
is deliberately a separate, explicit command.
