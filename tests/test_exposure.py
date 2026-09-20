import numpy as np
import pytest
from src.exposure import ExposureEngine, CSA
from src.xva import XVAEngine
from src.instruments import Trade


class LinearTrade(Trade):
    def value(self,i,ctx):
        return np.full(ctx["r"].shape[0],100+ctx["time"][i]*365)


def context(days=20,step=1):
    t = np.arange(0,days+1,step)/365
    return {"time": t, "r": np.zeros((2,len(t))), "df": np.ones((2,len(t)))}


def test_actual_ten_day_lag_and_daily_calls():
    ctx = context()
    e = ExposureEngine(ctx)
    e.add_trade(LinearTrade("l"),"n",CSA(threshold=0,mta=0,mpor_days=10))
    r = e.run()["n"]
    assert r["Collateral"][9,0] == 0
    assert r["Collateral"][10,0] == 100
    assert r["Collateral"][11,0] == 101
    assert r["EPE"][20] == pytest.approx(10)


def test_coarse_grid_rejected_and_inconsistent_csa_rejected():
    e = ExposureEngine(context(100,25))
    e.add_trade(LinearTrade("l"),"n",CSA())
    with pytest.raises(ValueError,match="coarse"):
        e.run()
    with pytest.raises(ValueError,match="Inconsistent"):
        e.add_trade(LinearTrade("m"),"n",CSA(threshold=1))


def test_threshold_mta_and_call_frequency():
    e = ExposureEngine(context())
    e.add_trade(LinearTrade("l"),"n",CSA(threshold=10,mta=5,call_frequency_days=2,mpor_days=0))
    r = e.run()["n"]
    assert r["Collateral"][0,0] == 90
    assert r["Collateral"][5,0] == 90
    assert r["Collateral"][6,0] == 96


def test_discounted_exposure_mean_not_product_of_means():
    ctx = {"time":np.array([0.,1.]), "df":np.array([[1.,.5],[1.,1.]])}
    paths = np.array([[0.,0.],[100.,0.]])
    data = {"n":{"ExposurePaths":paths,"IMPaths":np.zeros_like(paths)}}
    result = XVAEngine(ctx,data).compute(hazard_rate=.1,recovery=.4,funding_spread=.01)["n"]
    assert result["CVA"] == pytest.approx(.6*(1-np.exp(-.1))*12.5)
    assert result["FVA"] == pytest.approx(.125)
