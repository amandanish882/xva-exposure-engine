import numpy as np
import pytest
from scipy.integrate import quad
from src.curves import DiscountCurve
from src.market_data import MarketData
from src.engine import MonteCarloEngine
from src.instruments import InterestRateSwap, EuropeanOption, FXForward, XCCYSwapSimple
from src.risk import initial_context, swap_risk
from src.calibration import black_price, implied_vol


def test_curve_nodes_interpolation_negative_rates_and_domain():
    c = DiscountCurve([1,2,5], [-.01,.02,.04])
    np.testing.assert_allclose(c.discount(c.tenors), np.exp(-c.tenors*c.zero_rates))
    assert c.discount(0) == 1 and c.discount(.5) > 1
    assert np.log(c.discount(1.5)) == pytest.approx((np.log(c.discount(1))+np.log(c.discount(2)))/2)
    with pytest.raises(ValueError):
        c.discount(6)
    with pytest.raises(ValueError):
        DiscountCurve([1,1], [.02,.03])


def test_bond_against_independent_gaussian_integral():
    m = MarketData().rate_model()
    for t, T in [(0.,4.), (.3,2.), (1.,4.), (2.,3.7)]:
        for x in [-.015, 0., .02]:
            # Independent conditional expectation of exp(-integral(r)).
            variance = quad(lambda u: (m.sigma*(1-np.exp(-m.a*(T-u)))/m.a)**2, t, T)[0]
            expected = np.exp(-m.shift_integral(t,T)-x*m.B(T-t)+.5*variance)
            assert m.bond(t,T,x) == pytest.approx(expected, rel=2e-8, abs=1e-10)
    assert m.bond(2,2,.1) == 1


def test_par_swap_signs_and_risk():
    md = MarketData()
    swap = InterestRateSwap("s", 1e6, 0., 4.)
    swap.fixed_rate = swap.par_rate(md.curve)
    ctx = initial_context(md)
    assert abs(swap.value(0,ctx)[0]) < 1e-8
    risk = swap_risk(swap,md)
    assert risk["signed_parallel_dv01"] > 0
    assert sum(risk["key_rate_dv01"].values()) == pytest.approx(risk["signed_parallel_dv01"], rel=1e-5)
    swap.fixed_rate -= .01
    payer = swap.value(0,ctx).copy()
    assert np.all(payer > 0)
    swap.pay_fixed = False
    np.testing.assert_allclose(swap.value(0,ctx), -payer)


def test_fixed_schedule_stub_and_past_reset_without_lookahead():
    md = MarketData()
    tr = InterestRateSwap("s",1e6,.04,1.1)
    np.testing.assert_allclose(tr.pay_times, [.25,.5,.75,1.,1.1])
    ctx = MonteCarloEngine(md, {"n_sims":20,"time_horizon":1.1,"n_steps":110,
                                "event_times":tr.event_times}).simulate()
    i = np.argmin(abs(ctx["time"]-.3))
    original = tr.value(i,ctx).copy()
    ctx["df"][:,i+1:] *= .1
    ctx["x"][:,i+1:] += .5
    np.testing.assert_allclose(tr.value(i,ctx), original)
    # Changing a known past fixing must change the next coupon's value.
    reset = np.argmin(abs(ctx["time"]-.25))
    ctx["x"][:,reset] += .01
    assert not np.allclose(tr.value(i,ctx), original)


def test_missing_reset_rejected():
    md = MarketData()
    ctx = MonteCarloEngine(md, {"n_sims":4,"time_horizon":1,"n_steps":10}).simulate()
    with pytest.raises(ValueError, match="missing"):
        InterestRateSwap("s",1e6,.04,1).value(3,ctx)


def test_fx_forward_parity_and_xccy_sign():
    md = MarketData()
    ctx = initial_context(md)
    K = md.fx_spot*np.exp(-md.foreign_rate)/md.curve.discount(1)
    assert abs(FXForward("f",1e6,K,1).value(0,ctx)[0]) < 1e-8
    a = XCCYSwapSimple("x",1e6,1e6,.04,.03,3)
    v = a.value(0,ctx).copy()
    a.pay_domestic = False
    np.testing.assert_allclose(a.value(0,ctx), -v)


@pytest.mark.parametrize("asset", ["S","FX"])
def test_option_put_call_parity_and_iv_mapping(asset):
    md = MarketData()
    ctx = initial_context(md)
    spot = md.eq_spot if asset == "S" else md.fx_spot
    carry = md.eq_dividend if asset == "S" else md.foreign_rate
    c = EuropeanOption("c",asset,spot,2)
    p = EuropeanOption("p",asset,spot,2,"put")
    np.testing.assert_allclose(c.value(0,ctx)-p.value(0,ctx),
                               spot*np.exp(-carry*2)-spot*md.curve.discount(2), atol=1e-10)
    model = md.rate_model()
    for rho in [-.8,0,.8]:
        sigma = model.asset_sigma(.2,2,rho)
        assert model.option_variance(2,sigma,rho) == pytest.approx(.2**2*2, abs=1e-12)


def test_settled_trades_have_zero_value_and_im():
    md = MarketData()
    ctx = MonteCarloEngine(md, {"n_sims":4,"time_horizon":2,"n_steps":2}).simulate()
    trades = [EuropeanOption("o","S",4000,1), FXForward("f",1000,1.1,1),
              InterestRateSwap("s",1e6,.04,1), XCCYSwapSimple("x",1e6,1e6,.04,.03,1)]
    for tr in trades:
        for i in (1,2):
            assert not tr.value(i,ctx).any()
            assert not tr.im_proxy(i,ctx).any()


def test_black_iv_recovery_and_invalid_quotes():
    for T in [.1,1,3]:
        for K in [80,100,120]:
            for call in [True,False]:
                px = black_price(100,K,T,.95,.25,call)
                assert implied_vol(px,100,K,T,.95,call) == pytest.approx(.25,abs=1e-10)
    with pytest.raises(ValueError):
        implied_vol(101,100,100,1,1)
