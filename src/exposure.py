import numpy as np

class CSA:
    def __init__(self, threshold=0.0, mta=0.0, call_frequency_days=1, mpor_days=10):
        self.threshold = float(threshold)
        self.mta = float(mta)
        self.call_frequency_days = int(call_frequency_days)
        self.mpor_days = int(mpor_days)

class ExposureEngine:
    def __init__(self, ctx):
        self.ctx = ctx
        self.trades = []
        self.netting = {}

    def add_trade(self, trade, netting_set_id, csa=None):
        self.trades.append(trade)
        if netting_set_id not in self.netting:
            if csa is None:
                csa = CSA(threshold=1e18, mta=0.0, call_frequency_days=1, mpor_days=0)
            self.netting[netting_set_id] = {"trades": [], "csa": csa}
        self.netting[netting_set_id]["trades"].append(trade)

    def run(self):
        time = self.ctx["time"]
        n_steps = len(time)
        n_sims = self.ctx["r"].shape[0]
        dt_years = float(time[1] - time[0])
        dt_days = max(1e-9, dt_years * 365.0)

        mtm = np.zeros((n_steps, n_sims, len(self.trades)))
        im_trade = np.zeros((n_steps, n_sims, len(self.trades)))

        print(f"Pricing {len(self.trades)} trades over {n_steps} steps...")
        for t in range(n_steps):
            for i, tr in enumerate(self.trades):
                mtm[t, :, i] = tr.value(t, self.ctx)
                im_trade[t, :, i] = tr.im_proxy(t, self.ctx)

        results = {}

        for ns_id, ns in self.netting.items():
            csa = ns["csa"]
            idx = [self.trades.index(tr) for tr in ns["trades"]]
            ns_mtm = np.sum(mtm[:, :, idx], axis=2)
            ns_im = np.sum(im_trade[:, :, idx], axis=2)

            collateral = np.zeros(n_sims)

            call_every = max(1, int(round(csa.call_frequency_days / dt_days)))
            lag_idx = max(0, int(round(csa.mpor_days / dt_days)))

            exposure = np.zeros_like(ns_mtm)
            im_profile = np.zeros_like(ns_mtm)

            for t in range(n_steps):
                if (t % call_every) == 0:
                    t_lag = max(0, t - lag_idx)
                    mtm_lag = ns_mtm[t_lag, :]

                    target = np.maximum(mtm_lag - csa.threshold, 0.0)

                    change = target - collateral
                    do_change = np.abs(change) > csa.mta
                    collateral = np.where(do_change, target, collateral)

                exposure[t, :] = np.maximum(ns_mtm[t, :] - collateral, 0.0)
                im_profile[t, :] = np.maximum(ns_im[t, :], 0.0)

            results[ns_id] = {
                "EPE": np.mean(exposure, axis=1),
                "PFE": np.percentile(exposure, 95, axis=1),
                "ExposurePaths": exposure,
                "IMPaths": im_profile
            }

        total_exposure = np.zeros((n_steps, n_sims))
        total_im = np.zeros((n_steps, n_sims))
        for ns_id in results:
            total_exposure += results[ns_id]["ExposurePaths"]
            total_im += results[ns_id]["IMPaths"]

        results["PORTFOLIO"] = {
            "EPE": np.mean(total_exposure, axis=1),
            "PFE": np.percentile(total_exposure, 95, axis=1),
            "ExposurePaths": total_exposure,
            "IMPaths": total_im
        }

        return results
