"""European Black IV inversion and elementary, explicit quote filters."""
import numpy as np
from scipy.special import ndtr
from scipy.optimize import brentq


def black_price(forward, strike, expiry, discount, vol, is_call=True):
    if min(forward, strike, expiry, discount) <= 0 or vol < 0:
        raise ValueError("Invalid Black inputs")
    sign = 1 if is_call else -1
    if vol == 0:
        return discount*max(sign*(forward-strike), 0.)
    sd = vol*np.sqrt(expiry)
    d1 = np.log(forward/strike)/sd+sd/2
    return discount*sign*(forward*ndtr(sign*d1)-strike*ndtr(sign*(d1-sd)))


def implied_vol(price, forward, strike, expiry, discount, is_call=True):
    lower = black_price(forward, strike, expiry, discount, 0, is_call)
    upper = discount*(forward if is_call else strike)
    if not np.isfinite(price) or not lower < price < upper:
        raise ValueError("Price must be strictly inside European no-arbitrage bounds")
    return float(brentq(lambda v: black_price(forward, strike, expiry, discount, v, is_call)-price,
                        1e-9, 5., xtol=1e-12))


def clean_quotes(frame, asof, max_age_seconds=120, max_relative_spread=0.25):
    q = frame.copy()
    bid, ask = q["bid_px_00"], q["ask_px_00"]
    age = (asof-q["quote_time"]).dt.total_seconds()
    good = (np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask >= bid)
            & (q["bid_sz_00"] > 0) & (q["ask_sz_00"] > 0)
            & (age >= 0) & (age <= max_age_seconds)
            & ((ask-bid)/((ask+bid)/2) <= max_relative_spread))
    q = q.loc[good].copy()
    q["mid"] = (q.bid_px_00+q.ask_px_00)/2
    return q
