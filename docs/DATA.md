# Data and calibration

The repository runs offline with a labelled synthetic fixture. The optional
calibration script creates a private, dated snapshot; it does not silently replace
failed downloads with synthetic values. Do not commit API keys, vendor quotes or
`data/local_market.json`. Exchange/vendor redistribution permissions are separate
from permission to access data.

## SOFR-futures discount bootstrap

Market mode now replaces the previous FRED Treasury-zero proxy with a genuine
instrument-price bootstrap. Choose the latest common FRED `DGS3MO`, `SP500` and
`DEXUSEU` date no more than 14 calendar days before `--asof`. Request that UTC
day's `definition` and `statistics` data for `SR3.FUT` in Databento `GLBX.MDP3`.
The run must find a consecutive, quoted strip through at least five years.
Failure is an error, not permission to fill a gap or extrapolate the curve.

Select quarterly **outright** SR3 contracts, excluding spreads, packs and serials.
Use definition `maturity_year`/`maturity_month` for the reference quarter: `SR3U6`
starts in September, it does not end in September. The reference period is the
third Wednesday of that month to the third Wednesday three months later.
Cross-check the symbol and vendor expiration. See
[CME's contract and accrual conventions](https://www.cmegroup.com/education/articles-and-reports/understanding-sofr-futures).
SR1 is an arithmetic average and is intentionally not fed into this compounded
bootstrap.

Select settlement statistics (`stat_type=3`) whose **`ts_ref` trading date**
matches the curve date. Keep the latest update, reject deletions, missing prices,
theoretical prices and intraday settlements. Preserve `stat_flags` and the event
timestamp. A record marked preliminary remains labelled preliminary: an EOD
snapshot is not necessarily a final clearing settlement. This is disclosed in
the private audit, not silently upgraded to final. Far-dated exchange settlements
are not proof of executable two-sided liquidity. See
[Databento's CME statistics conventions](https://databento.com/docs/venues-and-datasets/glbx-mdp3).

For price `Q`, set `R=(100-Q)/100` and total growth `G=1+R*D/360`, where `D` is
the quarter's actual calendar-day count. The **futures–forward convexity
adjustment is deliberately zero**. For successive unstarted quarters,
`P(0,end)=P(0,start)/G`. For the active quarter, obtain daily FRED `SOFR` fixings
and calculate `A=product(1+fixing*days/360)` through the curve date, exclusive.
Then `P(0,end)=A/G`. Friday/holiday fixing intervals accrue simply over their
calendar days; business-day intervals compound. QuantLib's SOFR fixing calendar
defines required dates. Missing required fixings are errors. A curve-date fixing
is never used in this realised product because it is published the following
business day. Fixing units are percent in FRED, decimal in the bootstrap.

Store the resulting positive discount factors as continuously compounded zero
nodes on **ACT/365** engine times; interpolate log discounts. The snapshot saves
selected prices, reference dates, realised factors, source status, raw fixing
inputs and repricing residuals privately. `validate.py` rebuilds the curve from
those inputs and reprices every future. An offline test independently compares
the recursion with QuantLib `SofrFutureRateHelper` at zero convexity.

Exact repricing checks the algebra; it does **not** establish that futures and
forwards are economically identical. This simplification becomes more material
farther out. No OTC OIS swap quotes or second projection curve are used. See
[CME's explanation of convexity bias](https://www.cmegroup.com/education/courses/introduction-to-sofr/understanding-convexity-bias).

## Historical dependence (kept separate from curve construction)

Download three years of `DGS3MO`, `SP500` and `DEXUSEU` ending on that date.
Keep actual common-date observations, then select the last common observation
each week. Estimate Pearson correlations of absolute decimal rate changes and
equity/FX log returns. No independent forward-filling of missing closes is used.
Apply `R = 0.9 * sample_R + 0.1 * identity` and require a valid Cholesky factor.
Ten-percent shrinkage is a disclosed stabilisation choice, not an optimised fit.

To compare with overnight rates, the script also reports daily and weekly
`SOFR`-based matrices alongside `DGS3MO` on identical common dates. These are
diagnostics, not an automatic parameter-selection rule. The existing weekly
DGS3MO correlation and rate-volatility proxies are retained for this curve-only
upgrade. Overnight SOFR includes policy steps and funding/period-end effects;
substituting it need not improve the model's Brownian shock estimate. Historical
correlation uses observation dates and is not a publication-time-aligned backtest.

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
  Futures settlements are later in the day than the option window as well.
  These are effective flat carries, not separately calibrated dividend/foreign
  curves. This is a dated end-of-day research approximation, not an intraday
  executable pricing snapshot or point-in-time backtest.
- Vendor expiry timestamps and ACT/365 are used. Index fixing/settlement
  calendars are not separately implemented. FX diffusion volatility is held
  flat beyond the near-one-year calibration expiry for the three-year XCCY trade.
- Curve fit and option IV inversion are reproducible checks; neither implies
  independent economic validation of these input proxies. Production work would
  first require a justified futures convexity treatment and/or OTC OIS quotes,
  aligned spot/forward quotes and a rates
  option-volatility source.
