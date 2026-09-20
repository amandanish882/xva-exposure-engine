"""Curve bumps and statistics for independent observations or antithetic pairs."""
import numpy as np


def independent_samples(values, antithetic=False):
    values = np.asarray(values)
    if antithetic:
        if len(values) % 2:
            raise ValueError("Antithetic observations must form pairs")
        values = (values[:len(values)//2]+values[len(values)//2:])/2
    return values


def mean_se(values, antithetic=False):
    samples = independent_samples(values, antithetic)
    return float(samples.mean()), float(samples.std(ddof=1)/np.sqrt(len(samples)))


def initial_context(md):
    model = md.rate_model()
    return {"time": np.array([0.]), "x": np.zeros((2,1)), "r": np.full((2,1), md.r0),
            "S": np.full((2,1), md.eq_spot), "FX": np.full((2,1), md.fx_spot),
            "df": np.ones((2,1)), "md": md, "model": model}


def swap_risk(swap, md):
    def pv(m):
        return float(swap.value(0, initial_context(m))[0])
    base = pv(md)
    # Signed +1bp central response; payer-fixed is generally positive.
    parallel = (pv(md.shocked(dr=1e-4))-pv(md.shocked(dr=-1e-4)))/2
    nodes = [(pv(md.shocked(dr=1e-4, node=i))-pv(md.shocked(dr=-1e-4, node=i)))/2
             for i in range(len(md.curve.tenors))]
    return {"npv": base, "signed_parallel_dv01": parallel,
            "key_rate_dv01": dict(zip(map(str, md.curve.tenors), nodes)),
            "convention": "(PV(curve+1bp)-PV(curve-1bp))/2; fixed a, sigma and contract coupon"}
