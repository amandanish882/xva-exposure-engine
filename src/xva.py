import numpy as np

class XVAEngine:
    def __init__(self, ctx, exposure_results):
        self.ctx = ctx
        self.res = exposure_results
        self.time = ctx["time"]
        self.dt = np.diff(self.time)
        self.df = np.mean(ctx["df"], axis=0)

    def compute(self, hazard_rate=0.02, recovery=0.4, funding_spread=0.01, im_funding_spread=0.01):
        surv = np.exp(-hazard_rate * self.time)
        dpd = -np.diff(surv)

        out = {}
        for ns_id, data in self.res.items():
            if ns_id == "PORTFOLIO":
                continue

            epe = data["EPE"]
            im = np.mean(data["IMPaths"], axis=1)

            epe_mid = 0.5 * (epe[:-1] + epe[1:])
            im_mid = 0.5 * (im[:-1] + im[1:])
            df_mid = 0.5 * (self.df[:-1] + self.df[1:])

            cva = np.sum((1.0 - recovery) * epe_mid * df_mid * dpd)
            fva = np.sum(funding_spread * epe_mid * df_mid * self.dt)
            mva = np.sum(im_funding_spread * im_mid * df_mid * self.dt)

            out[ns_id] = {"CVA": cva, "FVA": fva, "MVA": mva, "Total": cva + fva + mva}

        total = {"CVA": 0.0, "FVA": 0.0, "MVA": 0.0, "Total": 0.0}
        for ns_id in out:
            for k in total:
                total[k] += out[ns_id][k]
        out["PORTFOLIO"] = total
        return out
