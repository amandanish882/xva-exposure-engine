# A short, defensible project explanation

“I built a cross-asset exposure engine with one-factor Hull–White rates, equity
and FX. I construct an initial discount curve, use conditional bond prices to
revalue the same trades along correlated scenarios, then net and collateralise
exposure before calculating an independent-default CVA. I validate prices,
discounted martingales, sensitivities and Monte Carlo error.”

## Walk through it in this order

1. **Hull–White:** one mean-reverting random factor plus a deterministic shift.
   The factor supplies uncertainty; the shift makes the model fit today's curve.
   Mean reversion is assumed and rate volatility is historically estimated.
2. **Curve:** turn supplied zero yields into discount factors and interpolate
   their logarithms. This is simple and positive by construction. It is a
   Treasury proxy curve, not an OIS bootstrap from market instrument quotes.
3. **Bond pricing:** at a future time use only the current factor and model
   curve to price remaining cashflows. Never read a path's future discount
   factors to determine its current MtM. QuantLib provides the bond coefficients;
   an independent Gaussian expectation tests them.
4. **Swap:** floating leg minus fixed leg for a payer. Fix the coupon schedule
   at inception, retain past floating resets, and drop cashflows once paid.
   Check zero inception value at par and positive rate sensitivity for a payer.
5. **Correlation:** estimate weekly rate changes against equity and FX log
   returns, lightly shrink the matrix and use Cholesky to generate joint shocks.
   Historical estimation is a transparent proxy, not an implied-correlation fit.
6. **Simulation and exposure:** simulate on daily/event dates, reprice, net by
   counterparty, subtract lagged collateral, then take the positive part.
   Average discount times exposure pathwise before integrating default risk.
7. **Validation:** show the automated tests and analytical/Monte Carlo agreement,
   then explain DV01, grid bias and the variance-reduction experiments.

## Three variance-reduction answers

**Antithetic:** pair each normal shock with its negative. Both paths have the
correct marginal distribution; averaging can cancel part of the noise. Treat
the pair average as one independent observation when estimating standard error.
It does not always improve every nonlinear exposure/PFE estimator.

**Control variate:** for the existing equity call, use discounted terminal equity
as a correlated control with known expectation `S0 * exp(-q*T)`. Subtract a fitted
multiple of its deviation from that expectation. Fit the coefficient on an
independent pilot sample, then test pricing consistency and measured variance.
The coefficient is a regression coefficient, not necessarily the option delta.

**Common random numbers:** reuse the same shocks for original and bumped markets.
The two price errors then largely cancel in the difference. This reduces noise
in a sensitivity or stress comparison, not necessarily in the individual price.
Hold the contract strike/coupon fixed during the bump; otherwise it is a
different trade. The reported DV01 is signed PV change for a +1bp move, estimated
centrally while holding model parameters fixed.

## If asked what changed from the first version

The rate dynamics now fit an initial curve, valuations use conditional bonds
instead of future path information, swaps have fixed schedules and correct
signs, settled options disappear, the FX drift includes a foreign rate, and
collateral lag is resolved by the time grid. Correlation is data-estimated and
equity/FX volatility can come from option quotes. Historical rate volatility is
still a clearly labelled proxy. XVA now preserves discount/exposure dependence.

## Draw a clear boundary

Do not claim SABR/ZABR, multi-factor rates, swaption/cap/CMS pricing, OIS
bootstrapping, SIMM or fully calibrated rates volatility. They are not implemented.
The model has three **cross-asset** shocks but only one **rates** factor.

If asked for the next enhancement, say: “I would obtain consistent OIS and rates
option quotes, bootstrap discount/projection curves, then calibrate Hull–White
parameters to a small liquid swaption set. I would add another rate factor or
a smile model only when the products and validation evidence justified it.”

That is an extension path, not a claim about existing functionality. No separate
SABR or swaption notebook is needed to explain this implementation.
