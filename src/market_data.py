"""Explicit snapshots: offline fixture or dated, validated market calibration."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .curves import DiscountCurve
from .rates import HullWhite

DEFAULT_SNAPSHOT = Path(__file__).resolve().parents[1] / "data/demo_market.json"


def estimate_correlation(frame, shrinkage=0.1, min_weeks=52):
    if not 0 <= shrinkage <= 1:
        raise ValueError("Shrinkage outside [0,1]")
    data = frame[["DGS3MO", "SP500", "DEXUSEU"]].sort_index().dropna()
    if data.index.has_duplicates or (data[["SP500", "DEXUSEU"]] <= 0).any().any():
        raise ValueError("Invalid historical observations")
    # One actual common-date row per week; no independently forward-filled closes.
    weekly = data.groupby(data.index.to_period("W-FRI")).tail(1)
    moves = pd.DataFrame({"rate": weekly.DGS3MO.diff()/100,
                          "equity": np.log(weekly.SP500).diff(),
                          "fx": np.log(weekly.DEXUSEU).diff()}).dropna()
    if len(moves) < min_weeks or (moves.std() <= 0).any():
        raise ValueError("Insufficient nonconstant common-date observations")
    sample = moves.corr().to_numpy()
    result = (1-shrinkage)*sample + shrinkage*np.eye(3)
    np.linalg.cholesky(result)
    return result, {"weekly_changes": len(moves), "first": str(weekly.index[0].date()),
                    "last": str(weekly.index[-1].date()), "shrinkage": shrinkage,
                    "sample_correlation": sample.tolist()}


class MarketData:
    def __init__(self, snapshot=None):
        with open(snapshot or DEFAULT_SNAPSHOT) as f:
            self.load(json.load(f))

    def load(self, data):
        self.metadata = deepcopy(data.get("metadata", {}))
        self.curve = DiscountCurve(data["curve"]["tenors"], data["curve"]["zero_rates"])
        for name in ("hw_a", "hw_sigma", "eq_spot", "fx_spot", "eq_vol", "fx_vol",
                     "eq_dividend", "foreign_rate"):
            value = float(data[name])
            if not np.isfinite(value):
                raise ValueError(f"Nonfinite {name}")
            setattr(self, name, value)
        if min(self.eq_spot, self.fx_spot) <= 0 or min(self.eq_vol, self.fx_vol) < 0:
            raise ValueError("Invalid spots or volatility")
        self.corr = np.asarray(data["correlation"], dtype=float)
        if (self.corr.shape != (3,3) or not np.allclose(self.corr, self.corr.T)
                or not np.allclose(np.diag(self.corr), 1) or not np.isfinite(self.corr).all()):
            raise ValueError("Invalid correlation matrix")
        np.linalg.cholesky(self.corr)
        self.iv_quotes = deepcopy(data.get("iv_quotes", {}))
        self.rate_model()

    @property
    def r0(self):
        return float(self.curve.forward(0.))

    def rate_model(self):
        return HullWhite(self.curve, self.hw_a, self.hw_sigma)

    def shocked(self, dr=0., eq_vol_mult=1., fx_vol_mult=1., node=None):
        if not np.isfinite([dr, eq_vol_mult, fx_vol_mult]).all() or min(eq_vol_mult, fx_vol_mult) < 0:
            raise ValueError("Invalid market shock")
        md = deepcopy(self)
        md.curve = self.curve.bumped(dr, node)
        md.eq_vol *= eq_vol_mult
        md.fx_vol *= fx_vol_mult
        return md

    def shock(self, dr=0., eq_vol_mult=1., fx_vol_mult=1.):
        updated = self.shocked(dr, eq_vol_mult, fx_vol_mult)
        self.__dict__.update(updated.__dict__)
