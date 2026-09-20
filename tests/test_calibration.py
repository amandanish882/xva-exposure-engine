import pandas as pd
import pytest
from types import SimpleNamespace
from src.calibration import clean_quotes, black_price
from src.curves import DiscountCurve
from scripts.calibrate_market import choose_pair, Downloader


def test_reject_crossed_stale_wide_and_empty_quotes():
    now = pd.Timestamp("2026-09-11T15:05Z")
    quotes = pd.DataFrame({"bid_px_00":[9.,10.,9.,1.,9.],
                          "ask_px_00":[10.,9.,10.,10.,10.],
                          "bid_sz_00":[1,1,1,1,0],"ask_sz_00":[1]*5,
                          "quote_time":[now-pd.Timedelta(seconds=s) for s in [60,60,300,60,60]]})
    result = clean_quotes(quotes,now)
    assert result.index.tolist() == [0]
    assert result.iloc[0]["mid"] == 9.5


def test_matched_european_pair_recovers_forward_and_iv():
    curve = DiscountCurve([1, 2, 3], [.04, .04, .04])
    expiry = pd.Timestamp("2028-01-01", tz="UTC")
    quote_time = pd.Timestamp("2026-01-01", tz="UTC")
    p = curve.discount(2)
    rows = []
    for call in [True, False]:
        price = black_price(101, 100, 2, p, .20, call)
        rows.append({"expiration":expiry, "expiry_years":2., "instrument_class":"C" if call else "P",
                     "strike_price":100., "mid":price, "bid_px_00":price-.05,
                     "ask_px_00":price+.05, "quote_time":quote_time, "raw_symbol":"fixture"})
    quotes = pd.DataFrame(rows)
    result = choose_pair(quotes, 2, 100, curve)
    assert result["forward"] == pytest.approx(101)
    assert result["iv"] == pytest.approx(.20)
    assert result["bid_iv"] < result["iv"] < result["ask_iv"]
    with pytest.raises(ValueError, match="No liquid"):
        choose_pair(quotes, .25, 100, curve)
    quotes.loc[1, "quote_time"] -= pd.Timedelta(minutes=2)
    with pytest.raises(ValueError, match="synchronised"):
        choose_pair(quotes, 2, 100, curve)


def test_vendor_budget_checked_before_download(monkeypatch, tmp_path):
    import scripts.calibrate_market as calibration
    def forbidden(**kwargs):
        pytest.fail("Over-budget download was attempted")
    client = SimpleNamespace(metadata=SimpleNamespace(get_cost=lambda **kwargs: 1.),
                             timeseries=SimpleNamespace(get_range=forbidden))
    monkeypatch.setenv("DATABENTO_API_KEY", "test-placeholder")
    monkeypatch.setattr(calibration.db, "Historical", lambda key: client)
    monkeypatch.setattr(calibration, "PRIVATE", tmp_path)
    with pytest.raises(RuntimeError, match="budget"):
        Downloader(.5).get("fixture")
