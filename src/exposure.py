"""Simplified unilateral threshold/MTA collateral with an explicit observation lag."""
from dataclasses import dataclass
import numpy as np


@dataclass
class CSA:
    threshold: float = 0.
    mta: float = 0.
    call_frequency_days: float = 1.
    mpor_days: float = 10.

    def __post_init__(self):
        if (not np.isfinite([self.threshold, self.mta, self.mpor_days, self.call_frequency_days]).all()
                or min(self.threshold, self.mta, self.mpor_days) < 0 or self.call_frequency_days <= 0):
            raise ValueError("Invalid collateral settings")


class ExposureEngine:
    def __init__(self, ctx):
        self.ctx, self.trades, self.netting = ctx, [], {}

    def add_trade(self, trade, netting_set_id, csa=None):
        if any(t.trade_id == trade.trade_id for t in self.trades):
            raise ValueError("Duplicate trade id")
        csa = csa or CSA(threshold=1e18, mpor_days=0)
        if netting_set_id in self.netting and self.netting[netting_set_id]["csa"] != csa:
            raise ValueError("Inconsistent CSA within netting set")
        self.trades.append(trade)
        self.netting.setdefault(netting_set_id, {"trades": [], "csa": csa})["trades"].append(trade)

    def run(self):
        time = self.ctx["time"]
        shape = (len(time), self.ctx["r"].shape[0])
        results = {}
        for ns_id, ns in self.netting.items():
            csa = ns["csa"]
            resolution = min(csa.call_frequency_days, csa.mpor_days or csa.call_frequency_days)
            if np.max(np.diff(time))*365 > resolution+1e-7:
                raise ValueError("Simulation grid too coarse for collateral calls/lag")
            mtm, im = np.zeros(shape), np.zeros(shape)
            for tr in ns["trades"]:
                for i in range(len(time)):
                    mtm[i] += tr.value(i, self.ctx)
                    im[i] += tr.im_proxy(i, self.ctx)
            collateral = np.zeros(shape)
            balance = np.zeros(shape[1])
            next_call = 0.
            for i, t in enumerate(time):
                day = t*365
                if day+1e-7 >= next_call:
                    lag_time = t-csa.mpor_days/365
                    j = np.searchsorted(time, lag_time+1e-10, side="right")-1
                    target = np.maximum((mtm[j] if j >= 0 else 0.)-csa.threshold, 0.)
                    balance = np.where(np.abs(target-balance) > csa.mta, target, balance)
                    next_call = (np.floor((day+1e-7)/csa.call_frequency_days)+1)*csa.call_frequency_days
                collateral[i] = balance
            exposure = np.maximum(mtm-collateral, 0.)
            results[ns_id] = self._metrics(exposure, im)
            results[ns_id].update(MtM=mtm, Collateral=collateral)
        if not results:
            raise ValueError("No trades")
        results["PORTFOLIO"] = self._metrics(
            sum(d["ExposurePaths"] for d in results.values()),
            sum(d["IMPaths"] for d in results.values()))
        return results

    @staticmethod
    def _metrics(exposure, im):
        return {"EPE": exposure.mean(axis=1), "PFE": np.percentile(exposure, 95, axis=1),
                "ExposurePaths": exposure, "IMPaths": im}
