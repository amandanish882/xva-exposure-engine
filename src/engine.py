"""Correlated HW + equity/FX GBM on a calendar-day/event grid."""
import numpy as np


class MonteCarloEngine:
    def __init__(self, market_data, config):
        self.md = market_data
        self.n_sims = int(config.get("n_sims", 2000))
        self.T = float(config.get("time_horizon", 5.))
        self.n_steps = int(config.get("n_steps", round(365*self.T)))
        self.antithetic = bool(config.get("antithetic", True))
        if self.n_sims < 2 or self.n_steps < 1 or self.T <= 0:
            raise ValueError("Invalid simulation configuration")
        if self.antithetic and self.n_sims % 2:
            raise ValueError("Antithetic simulation requires an even path count")
        events = np.asarray(config.get("event_times", []), dtype=float)
        if np.any(~np.isfinite(events)) or np.any(events < 0) or np.any(events > self.T):
            raise ValueError("Event times must be within the horizon")
        self.time = np.unique(np.round(np.r_[np.linspace(0, self.T, self.n_steps+1), events], 12))
        self.md.curve.discount(self.T)

    def simulate(self, seed=42):
        rng = np.random.default_rng(seed)
        n, times, md = self.n_sims, self.time, self.md
        model = md.rate_model()
        L = np.linalg.cholesky(md.corr)
        x = np.zeros((n, len(times)))
        S = np.full_like(x, md.eq_spot)
        FX = np.full_like(x, md.fx_spot)
        df = np.ones_like(x)
        for i, dt in enumerate(np.diff(times)):
            z = rng.standard_normal((n//2 if self.antithetic else n, 3))
            if self.antithetic:
                z = np.concatenate([z, -z], axis=0)
            dw = z @ L.T * np.sqrt(dt)
            x[:, i+1] = x[:, i]*(1-model.a*dt) + model.sigma*dw[:, 0]
            integral_r = 0.5*(x[:, i]+x[:, i+1])*dt + model.shift_integral(times[i], times[i+1])
            df[:, i+1] = df[:, i]*np.exp(-integral_r)
            S[:, i+1] = S[:, i]*np.exp(integral_r-(md.eq_dividend+0.5*md.eq_vol**2)*dt+md.eq_vol*dw[:, 1])
            FX[:, i+1] = FX[:, i]*np.exp(integral_r-(md.foreign_rate+0.5*md.fx_vol**2)*dt+md.fx_vol*dw[:, 2])
        return {"time": times, "x": x, "r": x+model.phi(times), "S": S, "FX": FX,
                "df": df, "md": md, "model": model, "antithetic": self.antithetic, "seed": seed}
