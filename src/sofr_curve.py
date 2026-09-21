"""Quarterly SR3 discount bootstrap, with an explicitly ZERO convexity adjustment.

SR3 quotes compounded SOFR over IMM-to-IMM quarters, on ACT/360. The
engine represents the resulting discount factors on ACT/365 time. SR1 (an
arithmetic average), serial contracts and gaps are deliberately not supported.
"""
from dataclasses import dataclass
from datetime import date, timedelta
import numpy as np
import QuantLib as ql
from .curves import DiscountCurve


def qdate(day):
    return ql.Date(day.day, day.month, day.year)


def imm_date(year, month):
    if month not in (3, 6, 9, 12):
        raise ValueError("Only quarterly SR3 contracts are supported")
    first = date(year, month, 1)
    return first + timedelta(days=(2-first.weekday()) % 7 + 14)


@dataclass(frozen=True)
class SofrFuture:
    symbol: str
    year: int
    month: int
    price: float  # IMM index points, e.g. 96.0 -> 4% per annum

    @property
    def start(self):
        return imm_date(self.year, self.month)

    @property
    def end(self):
        return imm_date(self.year + (self.month == 12), self.month % 12 + 3)

    @property
    def accrual(self):
        return (self.end-self.start).days/360


def realised_factor(start, end, fixings):
    """Product of 1 + fixing * calendar_days/360; fixings are DECIMAL rates.

    Both bounds must be SOFR business days. Only fixing dates strictly before
    end are read: a date's fixing is published on the following business day.
    Missing business-day fixings fail, rather than being silently forward-filled.
    """
    calendar = ql.Sofr().fixingCalendar()
    if start > end or not all(calendar.isBusinessDay(qdate(d)) for d in (start, end)):
        raise ValueError("Realised accrual bounds must be ordered SOFR business days")
    factor, day = 1., start
    while day < end:
        if day not in fixings or not np.isfinite(fixings[day]):
            raise ValueError(f"Missing SOFR fixing for {day}")
        next_ql = calendar.advance(qdate(day), 1, ql.Days)
        next_day = date(next_ql.year(), next_ql.month(), next_ql.dayOfMonth())
        accrual_end = min(next_day, end)
        growth = 1 + float(fixings[day])*(accrual_end-day).days/360
        if growth <= 0:
            raise ValueError("Nonpositive realised SOFR accrual factor")
        factor *= growth
        day = accrual_end
    return factor


def reprice_future(curve, asof, future, fixings):
    if future.end <= asof:
        raise ValueError("Cannot bootstrap an expired future")
    end_t = (future.end-asof).days/365
    if future.start < asof:
        growth = realised_factor(future.start, asof, fixings)/curve.discount(end_t)
    else:
        start_t = (future.start-asof).days/365
        growth = curve.discount(start_t)/curve.discount(end_t)
    return float(100 - 100*(growth-1)/future.accrual)


def bootstrap_sofr(asof, futures, fixings, required_horizon=5.):
    """Strip consecutive quarterly futures, beginning with the active quarter.

    Front: P(asof,end) = realised_growth / quoted_total_growth.
    Next:  P(asof,end) = P(asof,start) / quoted_total_growth.
    No root finder is needed. Quotes are fitted under zero futures convexity;
    exact repricing is a construction check, not evidence that the bias is zero.
    """
    if not np.isfinite(required_horizon) or required_horizon <= 0:
        raise ValueError("Required horizon must be positive")
    if not ql.Sofr().fixingCalendar().isBusinessDay(qdate(asof)):
        raise ValueError("Curve date must be a SOFR business day")
    contracts = sorted(futures, key=lambda f: f.start)
    if len({f.start for f in contracts}) != len(contracts):
        raise ValueError("Duplicate SR3 reference quarter")
    contracts = [f for f in contracts if f.end > asof]
    if not contracts or contracts[0].start > asof:
        raise ValueError("Missing active SR3 quarter: no short-end anchor")
    times, discounts, used = [], [], []
    previous_end, discount = None, 1.
    for future in contracts:
        if previous_end is not None and future.start != previous_end:
            raise ValueError("Gap or overlap in quarterly SR3 strip")
        if not np.isfinite(future.price):
            raise ValueError("Nonfinite SR3 price")
        growth = 1+(100-future.price)/100*future.accrual
        if growth <= 0:
            raise ValueError("Nonpositive quoted SR3 accrual factor")
        realised = realised_factor(future.start, asof, fixings) if future.start < asof else 1.
        discount = discount*realised/growth
        times.append((future.end-asof).days/365)
        discounts.append(discount)
        used.append((future, realised))
        previous_end = future.end
        if times[-1] >= required_horizon and len(times) >= 2:
            break
    if len(times) < 2 or times[-1] < required_horizon:
        raise ValueError("SR3 strip does not cover the required horizon; no extrapolation")
    curve = DiscountCurve(times, -np.log(discounts)/np.asarray(times))
    audit = []
    for future, realised in used:
        fitted = reprice_future(curve, asof, future, fixings)
        audit.append({"symbol":future.symbol, "year":future.year, "month":future.month,
                      "start":str(future.start), "end":str(future.end),
                      "accrual_act360":future.accrual, "price":float(future.price),
                      "realised_factor":realised, "model_price":fitted,
                      "price_error":fitted-future.price})
    return curve, {"method":"Quarterly SR3 compounded-SOFR futures bootstrap",
                   "convexity_adjustment":0., "asof":str(asof),
                   "required_horizon":required_horizon, "coverage_years":times[-1],
                   "max_price_error":max(abs(r["price_error"]) for r in audit),
                   "contracts":audit}
