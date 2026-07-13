import numpy as np

class MonteCarloEngine:
    def __init__(self, market_data, config):
        self.md = market_data
        self.n_sims = int(config.get("n_sims", 1000))
        self.n_steps = int(config.get("n_steps", 80))
        self.T = float(config.get("time_horizon", 5.0))
        self.antithetic = bool(config.get("antithetic", True))

        self.dt = self.T / self.n_steps
        self.time = np.linspace(0.0, self.T, self.n_steps + 1)

    def simulate(self, seed=42):
        np.random.seed(seed)

        n_sims = self.n_sims
        if self.antithetic and (n_sims % 2 == 1):
            n_sims += 1

        L = np.linalg.cholesky(self.md.corr)
        n_factors = 3

        if self.antithetic:
            half = n_sims // 2
            z_half = np.random.normal(0.0, 1.0, size=(n_factors, half, self.n_steps))
            z = np.concatenate([z_half, -z_half], axis=1)
        else:
            z = np.random.normal(0.0, 1.0, size=(n_factors, n_sims, self.n_steps))

        dw = np.einsum("ij,jkl->ikl", L, z) * np.sqrt(self.dt)

        r = np.zeros((n_sims, self.n_steps + 1))
        S = np.zeros((n_sims, self.n_steps + 1))
        FX = np.zeros((n_sims, self.n_steps + 1))

        r[:, 0] = self.md.r0
        S[:, 0] = self.md.eq_spot
        FX[:, 0] = self.md.fx_spot

        a = self.md.hw_a
        sigma_r = self.md.hw_sigma

        for t in range(self.n_steps):
            theta = a * self.md.r0
            r[:, t+1] = r[:, t] + (theta - a * r[:, t]) * self.dt + sigma_r * dw[0, :, t]

            drift_S = (r[:, t] - 0.5 * self.md.eq_vol**2) * self.dt
            diff_S = self.md.eq_vol * dw[1, :, t]
            S[:, t+1] = S[:, t] * np.exp(drift_S + diff_S)

            drift_X = (r[:, t] - 0.5 * self.md.fx_vol**2) * self.dt
            diff_X = self.md.fx_vol * dw[2, :, t]
            FX[:, t+1] = FX[:, t] * np.exp(drift_X + diff_X)

        r_avg = 0.5 * (r[:, :-1] + r[:, 1:])
        df_step = np.exp(-r_avg * self.dt)
        df = np.zeros_like(r)
        df[:, 0] = 1.0
        for t in range(self.n_steps):
            df[:, t+1] = df[:, t] * df_step[:, t]

        return {"time": self.time, "r": r, "S": S, "FX": FX, "df": df}
