"""One-factor Hull-White; QuantLib supplies conditional bond coefficients."""
from functools import lru_cache
import numpy as np
import QuantLib as ql


class HullWhite:
    def __init__(self, curve, a=0.05, sigma=0.01):
        if not np.isfinite(a) or not np.isfinite(sigma) or a < 1e-4 or sigma < 0:
            raise ValueError("Require a >= 1e-4 and sigma >= 0")
        self.curve, self.a, self.sigma = curve, float(a), float(sigma)
        base = ql.Date(1, 1, 2020)
        days = np.rint(curve.tenors*365).astype(int)
        if not np.allclose(days/365, curve.tenors, atol=1e-12, rtol=0):
            raise ValueError("Curve nodes must lie on whole ACT/365 days")
        dates = [base] + [base + int(d) for d in days]
        self.ql_curve = ql.DiscountCurve(dates, [1.] + curve.discount(curve.tenors).tolist(), ql.Actual365Fixed())
        self.ql_model = ql.HullWhite(ql.YieldTermStructureHandle(self.ql_curve), a, max(sigma, 1e-12))

    def B(self, tau):
        return -np.expm1(-self.a*np.asarray(tau))/self.a

    def integral_variance(self, tau):
        tau = np.asarray(tau, dtype=float)
        u = self.a*tau
        small = self.sigma**2*tau**3*(1/3-u/4+7*u*u/60-u**3/24)
        exact = self.sigma**2/self.a**2*(tau-2*self.B(tau)-np.expm1(-2*u)/(2*self.a))
        return np.maximum(np.where(np.abs(u) < 0.01, small, exact), 0.)

    def phi(self, t):
        return self.curve.forward(t) + 0.5*self.sigma**2*self.B(t)**2

    def shift_integral(self, t, u):
        return (self.curve.log_discount(t)-self.curve.log_discount(u)
                + 0.5*(self.integral_variance(u)-self.integral_variance(t)))

    @lru_cache(maxsize=32768)
    def _coefficient(self, t, T):
        # QL forwardRate(t,t) may average a knot; translate from our OU x state.
        f = self.ql_curve.forwardRate(t, t, ql.Continuous).rate()
        phi = f + 0.5*self.sigma**2*self.B(t)**2
        return self.ql_model.discountBond(t, T, phi)

    def bond(self, t, T, x=0.):
        if t < 0 or T < t-1e-10 or T > self.curve.tenors[-1]+1e-10:
            raise ValueError("Invalid bond times")
        if abs(T-t) < 1e-10:
            return np.ones_like(np.asarray(x, dtype=float))
        if t == 0:
            return self.curve.discount(T)*np.exp(-self.B(T)*np.asarray(x))
        return self._coefficient(float(t), float(T))*np.exp(-self.B(T-t)*np.asarray(x))

    def option_variance(self, tau, asset_sigma, rho):
        covariance = 2*rho*asset_sigma*self.sigma/self.a*(tau-self.B(tau))
        return np.maximum(asset_sigma**2*tau+self.integral_variance(tau)+covariance, 0.)

    def asset_sigma(self, black_iv, expiry, rho):
        """Translate quote IV into a diffusion sigma consistent with rate variance."""
        if black_iv <= 0 or expiry <= 0:
            raise ValueError("Positive IV and expiry required")
        b = 2*rho*self.sigma/self.a*(expiry-self.B(expiry))
        c = self.integral_variance(expiry)-black_iv**2*expiry
        disc = b*b-4*expiry*c
        if disc < 0:
            raise ValueError("Quoted IV incompatible with selected rate model")
        sigma = (-b+np.sqrt(disc))/(2*expiry)
        if sigma <= 0:
            raise ValueError("No positive asset diffusion volatility")
        return float(sigma)
