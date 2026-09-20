# Data and calibration

The repository runs offline with a labelled synthetic fixture. The optional
calibration script creates a private, dated snapshot; it does not silently replace
failed downloads with synthetic values. Do not commit API keys, vendor quotes or
`data/local_market.json`. Exchange/vendor redistribution permissions are separate
from permission to access data.

## Curve and historical dependence

Use FRED `THREEFY1` through `THREEFY10` on their latest common observation date,
at most 14 calendar days before the requested date. Convert annual percentage
zero yields to decimals and build `P(0,T)=exp(-zT)`, interpolating log discounts.
These are Treasury yields fitted by the Fed's Kim–Wright model, not raw OIS
quotes. The **source model** has three factors; this project's rate model has
**one**. See [FRED series notes](https://fred.stlouisfed.org/series/THREEFY1) and
the [Federal Reserve methodology and update schedule](https://www.federalreserve.gov/data/three-factor-nominal-term-structure-model.htm).

Download three years of `DGS3MO`, `SP500` and `DEXUSEU` ending on that date.
Keep actual common-date observations, then select the last common observation
each week. Estimate Pearson correlations of absolute decimal rate changes and
equity/FX log returns. No independent forward-filling of missing closes is used.
Apply `R = 0.9 * sample_R + 0.1 * identity` and require a valid Cholesky factor.
Ten-percent shrinkage is a disclosed stabilisation choice, not an optimised fit.

The standard deviation of weekly rate changes times `sqrt(52)` supplies the
rate diffusion-scale proxy. DGS3MO is not an instantaneous short rate; policy
jumps and maturity effects remain. Historical moves approximate model shock
correlations, not exact OU innovations. No cross-asset implied-correlation
calibration or physical-to-risk-neutral parameter estimation is claimed.

## Option inputs

Request one day's instrument definitions and five minutes of `cbbo-1m` quotes
from Databento: `SPX.OPT` in `OPRA.PILLAR` and European `EUU.OPT` in `GLBX.MDP3`.
The window is 15:00–15:05 UTC on the curve date. Each **uncached** request has
its estimated charge checked against the command's cumulative USD budget before
download. Entitlements and availability may differ by account. Cached results
incur no new request charge; the snapshot cost field records that run's estimate,
not lifetime spend or a final invoice.

Quote checks require positive two-sided prices/sizes, uncrossed markets,
relative bid/ask spread no greater than 25%, and age no greater than 120 seconds.
Select a matched European call/put at the same strike and expiry, with quote
timestamps at most 60 seconds apart. Infer the forward using call-put parity,
`F = K + (call - put)/P(0,T)`, then invert the call midpoint using European Black
pricing. Bid and ask IVs use the same midpoint-implied forward and are indicative,
not a fully propagated uncertainty interval.

Choose near-ATM pairs (`abs(log(K/F)) < 0.025`) near two years for equity and one
year for FX. Reject maturities more than 50% away from the target. These bounds
are broad enough for listed expiries; the selected maturity is always recorded.
The example equity trade uses that selected expiry. All subsequent strikes and
maturities use flat model diffusion sigmas: there is no smile/surface fit.

Because rates are stochastic, quoted Black IV is not identical to the asset's
diffusion sigma. The small quadratic conversion in `HullWhite.asset_sigma`
matches total log-forward variance, including rate variance and rate/asset
covariance, at the selected expiry. This reproduces the chosen IV within the
model; it does not remove data-basis limitations.

## Important approximation boundaries

- SPX options are index options. EUU options deliver into an FX future, not the
  spot currency: see [CME's FX product specification](https://www.cmegroup.com/markets/fx/fx-product-guide.html).
  The EUU Black IV and parity forward are **proxies** for the spot-FX model.
  Futures/forward convexity and the difference between option expiry and futures
  delivery are omitted. Do not call this exact OTC FX-option calibration.
- The carry rates reconcile each selected parity forward with a same-date FRED
  spot. Observations are not synchronous: SP500 is a closing level, while
  [DEXUSEU is a New York noon buying rate](https://fred.stlouisfed.org/series/DEXUSEU).
  These are effective flat carries, not separately calibrated dividend/foreign
  curves. This is a dated end-of-day research approximation, not an intraday
  executable pricing snapshot or point-in-time backtest.
- Vendor expiry timestamps and ACT/365 are used. Index fixing/settlement
  calendars are not separately implemented. FX diffusion volatility is held
  flat beyond the near-one-year calibration expiry for the three-year XCCY trade.
- Curve fit and option IV inversion are reproducible checks; neither implies
  independent economic validation of these input proxies. Production work would
  first require appropriate OIS curves, aligned spot/forward quotes and a rates
  option-volatility source.
