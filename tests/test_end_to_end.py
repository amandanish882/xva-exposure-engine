import numpy as np
from main import portfolio, run_scenario
from src.market_data import MarketData


def test_existing_portfolio_daily_exposure_and_xva():
    md = MarketData()
    trades = portfolio(md)
    ctx, exposure, xva = run_scenario(md, trades, n_sims=32, seed=3)
    assert np.diff(ctx["time"]).max() <= 1/365 + 1e-10
    for values in exposure.values():
        for key in ("ExposurePaths", "IMPaths", "EPE", "PFE"):
            assert np.isfinite(values[key]).all()
            assert (values[key] >= 0).all()
            assert not values[key][-1].any()
    for key in ("CVA", "FVA", "MVA", "Total"):
        assert xva["PORTFOLIO"][key] == xva["NS_BANK"][key] + xva["NS_CORP"][key]
        assert xva["PORTFOLIO"][key] >= 0
