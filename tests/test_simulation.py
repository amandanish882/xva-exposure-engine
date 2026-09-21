import numpy as np
import pandas as pd
import pytest
from src.market_data import MarketData, estimate_correlation
from src.engine import MonteCarloEngine
from src.risk import mean_se


def test_seed_reproducibility_antithetic_and_input_validation():
    md = MarketData()
    config = {"n_sims":100,"n_steps":20,"time_horizon":1}
    a = MonteCarloEngine(md,config).simulate(7)
    b = MonteCarloEngine(md,config).simulate(7)
    np.testing.assert_array_equal(a["S"],b["S"])
    np.testing.assert_allclose(a["x"][:50],-a["x"][50:],atol=1e-16)
    with pytest.raises(ValueError):
        MonteCarloEngine(md,dict(config,n_sims=99))


def test_generated_shock_correlation():
    md = MarketData()
    ctx = MonteCarloEngine(md,{"n_sims":40000,"n_steps":1,"time_horizon":.01}).simulate(4)
    dt = .01
    rate_z = ctx["x"][:,1]/(md.hw_sigma*np.sqrt(dt))
    integrated_r = -np.log(ctx["df"][:,1])
    eq_z = (np.log(ctx["S"][:,1]/md.eq_spot)-integrated_r+(md.eq_dividend+.5*md.eq_vol**2)*dt)/(md.eq_vol*np.sqrt(dt))
    fx_z = (np.log(ctx["FX"][:,1]/md.fx_spot)-integrated_r+(md.foreign_rate+.5*md.fx_vol**2)*dt)/(md.fx_vol*np.sqrt(dt))
    np.testing.assert_allclose(np.corrcoef([rate_z,eq_z,fx_z]),md.corr,atol=.018)


def test_discounted_bonds_and_assets_martingales():
    md = MarketData()
    c = MonteCarloEngine(md,{"n_sims":12000,"n_steps":365,"time_horizon":1}).simulate(19)
    bond_payoff = c["df"][:,-1]*c["model"].bond(1.,4.,c["x"][:,-1])
    for values, expected, tolerance in [
        (c["df"][:,-1],md.curve.discount(1),1e-5),
        (bond_payoff,md.curve.discount(4),2e-5),
        (c["df"][:,-1]*c["S"][:,-1],md.eq_spot*np.exp(-md.eq_dividend),.03),
        (c["df"][:,-1]*c["FX"][:,-1],md.fx_spot*np.exp(-md.foreign_rate),1e-5)]:
        mean,se = mean_se(values,True)
        assert abs(mean-expected) < 5*se+tolerance


def test_estimated_correlation_uses_common_actual_weekly_dates():
    rng = np.random.default_rng(23)
    dates = pd.bdate_range("2023-01-01", periods=780)
    z = rng.normal(size=(len(dates),3))
    z[:,1] = .6*z[:,0]+.8*z[:,1]
    df = pd.DataFrame({"DGS3MO":4+np.cumsum(z[:,0]*.01),
                       "SP500":4000*np.exp(np.cumsum(z[:,1]*.005)),
                       "DEXUSEU":np.exp(np.cumsum(z[:,2]*.005))},index=dates)
    df.loc[dates[::13],"SP500"] = np.nan
    R,meta = estimate_correlation(df)
    assert 145 < meta["weekly_changes"] < 158
    assert .35 < R[0,1] < .7
    assert np.linalg.eigvalsh(R).min() > 0
    with pytest.raises(ValueError):
        estimate_correlation(df.iloc[:20])


def test_mean_se_uses_pairs():
    mean,se = mean_se(np.array([1.,2.,3.,4.]),True)
    assert mean == 2.5 and se == pytest.approx(.5)


def test_sofr_correlation_diagnostic_respects_frequency_and_rate_choice():
    dates = pd.bdate_range("2023-01-01", periods=300)
    rng = np.random.default_rng(47)
    moves = rng.normal(size=(len(dates),3))
    frame = pd.DataFrame({"SOFR":4+np.cumsum(moves[:,0]*.01),
                          "SP500":4000*np.exp(np.cumsum(moves[:,1]*.002)),
                          "DEXUSEU":np.exp(np.cumsum(moves[:,2]*.002))}, index=dates)
    matrix, info = estimate_correlation(frame, rate_column="SOFR", frequency="daily")
    expected = pd.DataFrame({"r":frame.SOFR.diff()/100,
                             "s":np.log(frame.SP500).diff(),
                             "fx":np.log(frame.DEXUSEU).diff()}).dropna().corr().to_numpy()
    np.testing.assert_allclose(matrix,.9*expected+.1*np.eye(3))
    assert info["daily_changes"] == 299 and info["rate_column"] == "SOFR"
    assert np.linalg.eigvalsh(matrix).min() > 0
    _, weekly = estimate_correlation(frame, rate_column="SOFR")
    assert weekly["weekly_changes"] < 65
    with pytest.raises(ValueError, match="Frequency"):
        estimate_correlation(frame, frequency="monthly")


def test_rate_integral_discretisation_bias_converges_without_mc_noise():
    m = MarketData().rate_model()
    errors = []
    T = 5.
    for steps in [60, 365, 1825]:
        dt = T/steps
        A = 1-m.a*dt
        transition = np.array([[A, 0.], [.5*dt*(1+A), 1.]])
        noise = m.sigma*np.sqrt(dt)*np.array([1., .5*dt])
        cov = np.zeros((2, 2))
        for _ in range(steps):
            cov = transition @ cov @ transition.T + np.outer(noise, noise)
        # Both integrals are Gaussian: E[exp(-I)] depends on variance only.
        relative_discount_bias = abs(np.expm1(.5*(cov[1,1]-m.integral_variance(T))))
        errors.append(relative_discount_bias)
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 1e-6
