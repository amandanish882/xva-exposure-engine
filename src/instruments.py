"""Ex-payment-date values. Schedules are fixed; floating coupons use past fixings."""
import numpy as np
from scipy.special import ndtr


def schedule(maturity, freq):
    if maturity <= 0 or freq <= 0 or int(freq) != freq:
        raise ValueError("Positive maturity and integer frequency required")
    dates = np.r_[np.arange(1, int(np.floor(maturity*freq))+1)/freq, maturity]
    dates = np.unique(np.round(dates, 12))
    return dates, np.diff(np.r_[0., dates])


def grid_index(ctx, t):
    idx = int(np.argmin(np.abs(ctx["time"]-t)))
    if abs(ctx["time"][idx]-t) > 1e-9:
        raise ValueError(f"Required reset/payment time {t} missing from simulation grid")
    return idx


def bond(ctx, i, T):
    return ctx["model"].bond(float(ctx["time"][i]), float(T), ctx["x"][:, i])


class Trade:
    def __init__(self, trade_id):
        self.trade_id = trade_id

    def im_proxy(self, t_idx, ctx):
        return np.zeros(ctx["r"].shape[0])


class InterestRateSwap(Trade):
    def __init__(self, trade_id, notional, fixed_rate, maturity, pay_fixed=True, pay_freq_per_year=4):
        super().__init__(trade_id)
        self.notional, self.fixed_rate = float(notional), float(fixed_rate)
        self.maturity, self.pay_fixed = float(maturity), bool(pay_fixed)
        self.pay_times, self.accruals = schedule(self.maturity, pay_freq_per_year)

    @property
    def event_times(self):
        return self.pay_times

    def par_rate(self, curve):
        return float((1-curve.discount(self.maturity))/np.dot(self.accruals, curve.discount(self.pay_times)))

    def value(self, t_idx, ctx):
        t = float(ctx["time"][t_idx])
        remaining = np.flatnonzero(self.pay_times > t+1e-10)
        if not len(remaining):
            return np.zeros(ctx["r"].shape[0])
        j = int(remaining[0])
        reset = 0. if j == 0 else self.pay_times[j-1]
        ri = grid_index(ctx, reset)
        # The next floating coupon was fixed at the previous contractual reset.
        reset_bond = bond(ctx, ri, self.pay_times[j])
        float_leg = bond(ctx, t_idx, self.pay_times[j])/reset_bond - bond(ctx, t_idx, self.maturity)
        annuity = sum(self.accruals[k]*bond(ctx, t_idx, self.pay_times[k]) for k in remaining)
        payer = self.notional*(float_leg-self.fixed_rate*annuity)
        return payer if self.pay_fixed else -payer

    def im_proxy(self, t_idx, ctx):
        tau = max(self.maturity-float(ctx["time"][t_idx]), 0.)
        return np.full(ctx["r"].shape[0], abs(self.notional)*tau*1e-4*30)


class EuropeanOption(Trade):
    def __init__(self, trade_id, underlying, strike, maturity, opt_type="call", units=1.):
        super().__init__(trade_id)
        if underlying not in ("S", "FX") or opt_type not in ("call", "put") or strike <= 0 or maturity <= 0:
            raise ValueError("Invalid European option")
        self.underlying, self.K, self.T = underlying, float(strike), float(maturity)
        self.is_call, self.units = opt_type == "call", float(units)

    @property
    def event_times(self):
        return [self.T]

    def payoff(self, spot):
        return self.units*np.maximum((spot-self.K) if self.is_call else (self.K-spot), 0.)

    def _inputs(self, i, ctx):
        tau = self.T-float(ctx["time"][i])
        md = ctx["md"]
        sigma = md.eq_vol if self.underlying == "S" else md.fx_vol
        carry = md.eq_dividend if self.underlying == "S" else md.foreign_rate
        rho = md.corr[0, 1 if self.underlying == "S" else 2]
        discount = bond(ctx, i, self.T)
        prepaid = ctx[self.underlying][:, i]*np.exp(-carry*tau)
        v = ctx["model"].option_variance(tau, sigma, rho)
        return discount, prepaid, float(v), carry, tau

    def value(self, t_idx, ctx):
        if ctx["time"][t_idx] >= self.T-1e-10:
            return np.zeros(ctx["r"].shape[0])  # cash-settled; payoff is a separate cashflow
        p, s, var, _, _ = self._inputs(t_idx, ctx)
        if var < 1e-20:
            return self.units*np.maximum(s-self.K*p if self.is_call else self.K*p-s, 0.)
        std = np.sqrt(var)
        d1 = np.log(s/(self.K*p))/std + 0.5*std
        d2 = d1-std
        sign = 1 if self.is_call else -1
        return self.units*sign*(s*ndtr(sign*d1)-self.K*p*ndtr(sign*d2))

    def im_proxy(self, t_idx, ctx):
        if ctx["time"][t_idx] >= self.T-1e-10:
            return np.zeros(ctx["r"].shape[0])
        p, s, var, carry, tau = self._inputs(t_idx, ctx)
        d1 = np.log(s/(self.K*p))/np.sqrt(max(var, 1e-20)) + 0.5*np.sqrt(var)
        delta = np.exp(-carry*tau)*(ndtr(d1)-(0 if self.is_call else 1))
        return 0.15*abs(self.units)*np.abs(delta)*ctx[self.underlying][:, t_idx]


class FXForward(Trade):
    def __init__(self, trade_id, notional_foreign, strike, maturity):
        super().__init__(trade_id)
        self.N, self.K, self.T = float(notional_foreign), float(strike), float(maturity)

    @property
    def event_times(self):
        return [self.T]

    def value(self, t_idx, ctx):
        tau = self.T-float(ctx["time"][t_idx])
        if tau <= 1e-10:
            return np.zeros(ctx["r"].shape[0])
        return self.N*(ctx["FX"][:, t_idx]*np.exp(-ctx["md"].foreign_rate*tau)-self.K*bond(ctx, t_idx, self.T))

    def im_proxy(self, t_idx, ctx):
        return (0.08*abs(self.N)*ctx["FX"][:, t_idx] if ctx["time"][t_idx] < self.T-1e-10
                else np.zeros(ctx["r"].shape[0]))


class XCCYSwapSimple(Trade):
    """Fixed-fixed currency swap with terminal principals; deterministic foreign curve."""
    def __init__(self, trade_id, notional_dom, notional_for, dom_fixed, for_fixed, maturity,
                 pay_domestic=True, pay_freq_per_year=2):
        super().__init__(trade_id)
        self.Nd, self.Nf, self.kd, self.kf = map(float, (notional_dom, notional_for, dom_fixed, for_fixed))
        self.T, self.pay_domestic = float(maturity), bool(pay_domestic)
        self.pay_times, self.accruals = schedule(self.T, pay_freq_per_year)

    @property
    def event_times(self):
        return self.pay_times

    def value(self, t_idx, ctx):
        t = float(ctx["time"][t_idx])
        if t >= self.T-1e-10:
            return np.zeros(ctx["r"].shape[0])
        domestic = self.Nd*bond(ctx, t_idx, self.T)
        foreign = self.Nf*np.exp(-ctx["md"].foreign_rate*(self.T-t))
        for u, alpha in zip(self.pay_times, self.accruals):
            if u > t+1e-10:
                domestic += self.Nd*self.kd*alpha*bond(ctx, t_idx, u)
                foreign += self.Nf*self.kf*alpha*np.exp(-ctx["md"].foreign_rate*(u-t))
        v = foreign*ctx["FX"][:, t_idx]-domestic
        return v if self.pay_domestic else -v

    def im_proxy(self, t_idx, ctx):
        return (0.10*abs(self.Nf)*ctx["FX"][:, t_idx] if ctx["time"][t_idx] < self.T-1e-10
                else np.zeros(ctx["r"].shape[0]))
