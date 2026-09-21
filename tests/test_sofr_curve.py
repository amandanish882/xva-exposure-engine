from datetime import date, timedelta
import numpy as np
import pandas as pd
import pytest
import QuantLib as ql
from src.sofr_curve import SofrFuture, bootstrap_sofr, realised_factor, reprice_future, qdate
from scripts.calibrate_market import select_sofr_settlements


def fixture(asof=date(2026, 6, 22), zero=.04, years=6):
    """Invented quotes/fixings, NOT redistributed vendor data."""
    calendar = ql.Sofr().fixingCalendar()
    fixings = {}
    day = date(2026, 6, 17)
    while day < asof:
        if calendar.isBusinessDay(qdate(day)):
            fixings[day] = .037
        day += timedelta(days=1)
    futures = []
    for year in range(2026, 2026+years):
        for month in (3, 6, 9, 12):
            f = SofrFuture(f"synthetic-{year}-{month}", year, month, 0.)
            if f.end <= asof or f.start < date(2026, 6, 17):
                continue
            realised = realised_factor(f.start, asof, fixings) if f.start < asof else 1.
            growth = realised*np.exp(zero*(f.end-max(f.start,asof)).days/365)
            futures.append(SofrFuture(f.symbol,year,month,100-100*(growth-1)/f.accrual))
    return asof, futures, fixings


@pytest.mark.parametrize("asof", [date(2026,6,17), date(2026,6,22), date(2026,9,16)])
@pytest.mark.parametrize("zero", [.04, -.01])
def test_bootstrap_flat_curve_and_quote_repricing(asof, zero):
    day, futures, fixings = fixture(asof, zero)
    curve, audit = bootstrap_sofr(day,futures,fixings)
    np.testing.assert_allclose(curve.zero_rates,zero,atol=1e-13)
    assert 5 <= curve.tenors[-1] < 5.3
    assert audit["convexity_adjustment"] == 0
    assert audit["max_price_error"] < 1e-11
    for f in futures[:len(audit["contracts"])]:
        assert reprice_future(curve,day,f,fixings) == pytest.approx(f.price,abs=1e-11)


def test_weekend_holiday_compounding_and_no_unpublished_fixing():
    # Juneteenth Friday: Thursday's SOFR accrues simply over FOUR calendar days.
    start,end = date(2026,6,17),date(2026,6,22)
    fixings = {start:.03, date(2026,6,18):.04, end:99.}
    expected = (1+.03/360)*(1+.04*4/360)
    assert realised_factor(start,end,fixings) == pytest.approx(expected)
    del fixings[end]
    assert realised_factor(start,end,fixings) == pytest.approx(expected)
    del fixings[start]
    with pytest.raises(ValueError,match="Missing SOFR"):
        realised_factor(start,end,fixings)


def test_contract_reference_quarter_uses_start_month_and_leap_year():
    f = SofrFuture("SR3Z7",2027,12,96)
    assert f.start == date(2027,12,15)
    assert f.end == date(2028,3,15)
    assert f.accrual == 91/360


@pytest.mark.parametrize("case,match", [("gap","Gap"),("duplicate","Duplicate"),
    ("front","anchor"),("short","horizon"),("nan","Nonfinite"),("growth","Nonpositive"),
    ("serial","quarterly"),("fixing","Missing SOFR"),("weekend","business day")])
def test_bootstrap_rejects_invalid_inputs(case,match):
    day,futures,fixings = fixture()
    if case == "gap": del futures[2]
    if case == "duplicate": futures.append(futures[1])
    if case == "front": del futures[0]
    if case == "short": futures = futures[:3]
    if case in ("nan","growth","serial"):
        f=futures[0]
        futures[0]=SofrFuture(f.symbol,f.year,5 if case=="serial" else f.month,
                             float("nan") if case=="nan" else 10000.)
    if case == "fixing": fixings = {}
    if case == "weekend": day = date(2026,6,21)
    with pytest.raises(ValueError,match=match):
        bootstrap_sofr(day,futures,fixings)


def test_bootstrap_agrees_with_independent_quantlib_helpers():
    day, futures, fixings = fixture()
    curve,audit = bootstrap_sofr(day,futures,fixings)
    saved = ql.Settings.instance().evaluationDate
    index = ql.Sofr()
    try:
        ql.Settings.instance().evaluationDate = qdate(day)
        for d,r in fixings.items(): index.addFixing(qdate(d),r)
        helpers = [ql.SofrFutureRateHelper(f.price,f.month,f.year,ql.Quarterly,0.)
                   for f in futures[:len(audit["contracts"])]]
        independent = ql.PiecewiseLogLinearDiscount(qdate(day),helpers,ql.Actual365Fixed())
        for t in np.linspace(0,curve.tenors[-1],301):
            assert curve.discount(t) == pytest.approx(independent.discount(float(t)),abs=2e-11)
    finally:
        index.clearFixings()
        ql.Settings.instance().evaluationDate = saved


def settlement_inputs():
    now = pd.Timestamp("2026-09-11T21:40Z")
    definitions = pd.DataFrame({"instrument_id":[1,2,3], "raw_symbol":["SR3M6","SR3U6","SR3M6-SR3U6"],
        "instrument_class":["F","F","S"], "maturity_year":[2026]*3, "maturity_month":[6,9,6],
        "security_update_action":["A"]*3,
        "expiration":pd.to_datetime(["2026-09-15","2026-12-15","2026-09-15"],utc=True)},
        index=[now]*3)
    statistics = pd.DataFrame({"instrument_id":[1,1,2,3],"stat_type":[3]*4,"price":[96.1,96.2,96.3,1.],
        "ts_ref":[pd.Timestamp("2026-09-11",tz="UTC")]*4,"ts_event":[now]*4,
        "stat_flags":[2,3,2,3],"update_action":[1]*4},
        index=[now-pd.Timedelta(minutes=1),now,now,now])
    return definitions,statistics


def test_settlement_selection_date_last_update_and_status():
    defs,stats=settlement_inputs()
    futures,source=select_sofr_settlements(defs,stats,"2026-09-11")
    assert [f.price for f in futures] == [96.2,96.3]
    assert source["SR3M6"]["settlement_status"] == "final"
    assert source["SR3U6"]["settlement_status"] == "preliminary"
    with pytest.raises(ValueError,match="same-date"):
        select_sofr_settlements(defs,stats,"2026-09-10")


@pytest.mark.parametrize("column,value", [("update_action",2),("stat_flags",0),
                                          ("stat_flags",10),("price",float("nan"))])
def test_do_not_resurrect_deleted_or_invalid_latest_settlement(column,value):
    defs,stats=settlement_inputs()
    stats.iloc[1,stats.columns.get_loc(column)]=value
    futures,_=select_sofr_settlements(defs,stats,"2026-09-11")
    assert [f.symbol for f in futures] == ["SR3U6"]


def test_bootstrapped_curve_flows_into_hull_white_swaps_and_exposure():
    from src.market_data import MarketData
    from src.instruments import InterestRateSwap
    from src.risk import initial_context
    from main import portfolio, run_scenario
    day,futures,fixings = fixture()
    md=MarketData()
    md.curve,_=bootstrap_sofr(day,futures,fixings)
    model=md.rate_model()
    for t in md.curve.tenors:
        assert model.bond(0,float(t)) == pytest.approx(md.curve.discount(t))
    swap=InterestRateSwap("par",1e6,0,4)
    swap.fixed_rate=swap.par_rate(md.curve)
    assert abs(swap.value(0,initial_context(md))[0]) < 1e-7
    _,exposure,xva=run_scenario(md,portfolio(md),n_sims=32)
    assert all(np.isfinite(v) for group in xva.values() for v in group.values())
    assert all(np.max(e["EPE"][-1:]) == 0 for e in exposure.values())
