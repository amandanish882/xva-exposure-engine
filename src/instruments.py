import numpy as np
from scipy.stats import norm

class Trade:
    def __init__(self, trade_id):
        self.trade_id = trade_id

    def value(self, t_idx, ctx):
        raise NotImplementedError

    def im_proxy(self, t_idx, ctx):
        return np.zeros(ctx["r"].shape[0])

class InterestRateSwap(Trade):
    def __init__(self, trade_id, notional, fixed_rate, maturity, pay_fixed=True, pay_freq_per_year=4):
        super().__init__(trade_id)
        self.notional = float(notional)
        self.fixed_rate = float(fixed_rate)
        self.maturity = float(maturity)
        self.pay_fixed = bool(pay_fixed)
        self.freq = int(pay_freq_per_year)

    def value(self, t_idx, ctx):
        t = float(ctx["time"][t_idx])
        if t >= self.maturity:
            return np.zeros(ctx["r"].shape[0])

        tau = self.maturity - t

        n_pay = max(1, int(np.ceil(tau * self.freq)))
        pay_times = t + (np.arange(1, n_pay + 1) / self.freq)
        pay_times = pay_times[pay_times <= self.maturity + 1e-12]

        df_t = ctx["df"][:, t_idx]
        dt = float(ctx["time"][1] - ctx["time"][0])

        p_tu = []
        for u in pay_times:
            u_idx = int(round(u / dt))
            u_idx = min(u_idx, len(ctx["time"]) - 1)
            p_tu.append(ctx["df"][:, u_idx] / df_t)

        if not p_tu:
            return np.zeros(ctx["r"].shape[0])

        p_tu = np.stack(p_tu, axis=1)

        T_idx = int(round(self.maturity / dt))
        T_idx = min(T_idx, len(ctx["time"]) - 1)
        p_tT = ctx["df"][:, T_idx] / df_t

        alpha = 1.0 / self.freq
        annuity = alpha * np.sum(p_tu, axis=1)

        float_leg = 1.0 - p_tT
        fixed_leg = self.fixed_rate * annuity

        pv = self.notional * (float_leg - fixed_leg)
        return -pv if self.pay_fixed else pv

    def im_proxy(self, t_idx, ctx):
        t = float(ctx["time"][t_idx])
        if t >= self.maturity:
            return np.zeros(ctx["r"].shape[0])
        tau = self.maturity - t
        dv01 = self.notional * tau * 1e-4
        rw = 30.0
        return np.full(ctx["r"].shape[0], abs(dv01) * rw)

class EuropeanOption(Trade):
    def __init__(self, trade_id, underlying, strike, maturity, opt_type="call"):
        super().__init__(trade_id)
        self.underlying = underlying  # "S" or "FX"
        self.K = float(strike)
        self.T = float(maturity)
        self.is_call = (opt_type == "call")

    def value(self, t_idx, ctx):
        t = float(ctx["time"][t_idx])
        spot = ctx[self.underlying][:, t_idx]
        if t >= self.T:
            payoff = np.maximum(spot - self.K, 0.0) if self.is_call else np.maximum(self.K - spot, 0.0)
            return payoff

        r = ctx["r"][:, t_idx]
        tau = self.T - t
        vol = ctx["md"].eq_vol if self.underlying == "S" else ctx["md"].fx_vol

        d1 = (np.log(spot / self.K) + (r + 0.5 * vol * vol) * tau) / (vol * np.sqrt(tau))
        d2 = d1 - vol * np.sqrt(tau)

        if self.is_call:
            return spot * norm.cdf(d1) - self.K * np.exp(-r * tau) * norm.cdf(d2)
        return self.K * np.exp(-r * tau) * norm.cdf(-d2) - spot * norm.cdf(-d1)

    def im_proxy(self, t_idx, ctx):
        t = float(ctx["time"][t_idx])
        if t >= self.T:
            return np.zeros(ctx["r"].shape[0])

        spot = ctx[self.underlying][:, t_idx]
        r = ctx["r"][:, t_idx]
        tau = self.T - t
        vol = ctx["md"].eq_vol if self.underlying == "S" else ctx["md"].fx_vol

        d1 = (np.log(spot / self.K) + (r + 0.5 * vol * vol) * tau) / (vol * np.sqrt(tau))
        delta = norm.cdf(d1) if self.is_call else (norm.cdf(d1) - 1.0)

        rw = 0.15 if self.underlying == "S" else 0.10
        return rw * np.abs(delta) * spot

class FXForward(Trade):
    def __init__(self, trade_id, notional_foreign, strike, maturity):
        super().__init__(trade_id)
        self.N = float(notional_foreign)
        self.K = float(strike)
        self.T = float(maturity)

    def value(self, t_idx, ctx):
        t = float(ctx["time"][t_idx])
        if t >= self.T:
            return np.zeros(ctx["r"].shape[0])

        spot = ctx["FX"][:, t_idx]
        r = ctx["r"][:, t_idx]
        tau = self.T - t
        fwd = spot * np.exp(r * tau)
        df = np.exp(-r * tau)
        return self.N * (fwd - self.K) * df

    def im_proxy(self, t_idx, ctx):
        spot = ctx["FX"][:, t_idx]
        rw = 0.08
        return rw * np.abs(self.N) * spot

class XCCYSwapSimple(Trade):
    def __init__(self, trade_id, notional_dom, notional_for, dom_fixed, for_fixed, maturity, pay_domestic=True, pay_freq_per_year=2):
        super().__init__(trade_id)
        self.Nd = float(notional_dom)
        self.Nf = float(notional_for)
        self.kd = float(dom_fixed)
        self.kf = float(for_fixed)
        self.T = float(maturity)
        self.pay_domestic = bool(pay_domestic)
        self.freq = int(pay_freq_per_year)

    def value(self, t_idx, ctx):
        t = float(ctx["time"][t_idx])
        if t >= self.T:
            return np.zeros(ctx["r"].shape[0])

        spot_fx = ctx["FX"][:, t_idx]
        df_t = ctx["df"][:, t_idx]
        dt = float(ctx["time"][1] - ctx["time"][0])

        tau = self.T - t
        n_pay = max(1, int(np.ceil(tau * self.freq)))
        pay_times = t + (np.arange(1, n_pay + 1) / self.freq)
        pay_times = pay_times[pay_times <= self.T + 1e-12]

        alpha = 1.0 / self.freq
        pv_dom = np.zeros(ctx["r"].shape[0])
        pv_for = np.zeros(ctx["r"].shape[0])

        for u in pay_times:
            u_idx = int(round(u / dt))
            u_idx = min(u_idx, len(ctx["time"]) - 1)
            p_tu = ctx["df"][:, u_idx] / df_t
            pv_dom += self.Nd * self.kd * alpha * p_tu
            pv_for += (self.Nf * self.kf * alpha * p_tu) * spot_fx

        T_idx = int(round(self.T / dt))
        T_idx = min(T_idx, len(ctx["time"]) - 1)
        p_tT = ctx["df"][:, T_idx] / df_t
        pv_exchange = (self.Nf * spot_fx - self.Nd) * p_tT

        pv = (pv_for - pv_dom) + pv_exchange
        return -pv if self.pay_domestic else pv

    def im_proxy(self, t_idx, ctx):
        spot = ctx["FX"][:, t_idx]
        rw = 0.10
        return rw * np.abs(self.Nf) * spot
