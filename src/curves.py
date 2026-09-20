"""Supplied zero nodes with log-linear discount interpolation."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DiscountCurve:
    tenors: np.ndarray
    zero_rates: np.ndarray

    def __post_init__(self):
        t = np.asarray(self.tenors, dtype=float).copy()
        z = np.asarray(self.zero_rates, dtype=float).copy()
        if (t.ndim != 1 or len(t) < 2 or t.shape != z.shape
                or not np.all(np.isfinite(t)) or not np.all(np.isfinite(z))
                or t[0] <= 0 or np.any(np.diff(t) <= 0)):
            raise ValueError("Need finite, strictly increasing positive tenors and matching zero rates")
        t.setflags(write=False)
        z.setflags(write=False)
        object.__setattr__(self, "tenors", t)
        object.__setattr__(self, "zero_rates", z)

    def _check(self, t):
        t = np.asarray(t, dtype=float)
        if np.any(~np.isfinite(t)) or np.any(t < 0) or np.any(t > self.tenors[-1] + 1e-10):
            raise ValueError("Time outside curve range")
        return t

    def log_discount(self, t):
        t = self._check(t)
        return np.interp(t, np.r_[0., self.tenors], np.r_[0., -self.tenors*self.zero_rates])

    def discount(self, t):
        return np.exp(self.log_discount(t))

    def forward(self, t):
        """Right-continuous forward, flat from zero to first node."""
        t = self._check(t)
        knots = np.r_[0., self.tenors]
        slopes = np.diff(np.r_[0., self.tenors*self.zero_rates]) / np.diff(knots)
        idx = np.minimum(np.searchsorted(knots, t, side="right") - 1, len(slopes)-1)
        return slopes[idx]

    def zero(self, t):
        t = self._check(t)
        return np.where(t == 0, self.zero_rates[0], -self.log_discount(t)/np.maximum(t, 1e-30))

    def bumped(self, bump=1e-4, node=None):
        shifts = np.full(len(self.tenors), bump) if node is None else np.eye(len(self.tenors))[node]*bump
        return DiscountCurve(self.tenors, self.zero_rates + shifts)
