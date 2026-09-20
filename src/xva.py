"""Unilateral independent-default CVA; funding and schedule-IM proxies."""
import numpy as np


class XVAEngine:
    def __init__(self, ctx, exposure_results):
        self.ctx, self.res = ctx, exposure_results

    def compute(self, hazard_rate=0.02, recovery=0.4, funding_spread=0.01, im_funding_spread=0.01):
        if hazard_rate < 0 or not 0 <= recovery <= 1:
            raise ValueError("Invalid credit parameters")
        time = self.ctx["time"]
        dt = np.diff(time)
        dpd = -np.diff(np.exp(-hazard_rate*time))
        df = self.ctx["df"].T
        out = {}
        for ns_id, data in self.res.items():
            if ns_id == "PORTFOLIO":
                continue
            discounted_e = (df*data["ExposurePaths"]).mean(axis=1)
            discounted_im = (df*data["IMPaths"]).mean(axis=1)
            midpoint_e = (discounted_e[:-1]+discounted_e[1:])/2
            midpoint_im = (discounted_im[:-1]+discounted_im[1:])/2
            cva = float((1-recovery)*np.dot(midpoint_e, dpd))
            fva = float(funding_spread*np.dot(midpoint_e, dt))
            mva = float(im_funding_spread*np.dot(midpoint_im, dt))
            out[ns_id] = {"CVA": cva, "FVA": fva, "MVA": mva, "Total": cva+fva+mva}
        out["PORTFOLIO"] = {k: sum(v[k] for v in out.values()) for k in ("CVA", "FVA", "MVA", "Total")}
        return out
