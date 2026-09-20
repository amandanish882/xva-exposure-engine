"""Fetch dated FRED/Databento inputs into ignored local files.

Run explicitly with DATABENTO_API_KEY in the environment. Every uncached vendor
request is costed before download; the total must stay below --max-cost-usd.
No credential or vendor error URL is written to the output.
"""
import argparse
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import requests
import databento as db
from src.curves import DiscountCurve
from src.rates import HullWhite
from src.market_data import estimate_correlation
from src.calibration import clean_quotes, implied_vol, black_price

PRIVATE = Path("data/private")


def fred(ids, start, end):
    name = "_".join(ids) + "_" + str(start) + "_" + str(end) + ".csv"
    path = PRIVATE/name
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    key = os.environ.get("FRED_API_KEY")
    if key:
        columns = []
        for series in ids:
            try:
                response = requests.get("https://api.stlouisfed.org/fred/series/observations",
                    params={"series_id": series, "api_key": key, "file_type": "json",
                            "observation_start": str(start), "observation_end": str(end)}, timeout=60)
            except requests.RequestException:
                raise RuntimeError("FRED API connection failed") from None
            if response.status_code != 200:
                raise RuntimeError(f"FRED API {series} failed: HTTP {response.status_code}")
            observations = pd.DataFrame(response.json()["observations"])
            column = pd.Series(pd.to_numeric(observations.value, errors="coerce").to_numpy(),
                               index=pd.to_datetime(observations.date), name=series)
            columns.append(column)
        frame = pd.concat(columns, axis=1)
    else:
        response = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                                params={"id": ",".join(ids), "cosd": str(start), "coed": str(end)}, timeout=60)
        if response.status_code != 200 or not response.text.startswith("observation_date"):
            raise RuntimeError("FRED CSV unavailable; set FRED_API_KEY for the official API")
        frame = pd.read_csv(StringIO(response.text), index_col=0, parse_dates=True, na_values=".")
    frame = frame.apply(pd.to_numeric, errors="coerce").loc[str(start):str(end)]
    if frame.empty or not set(ids).issubset(frame.columns):
        raise ValueError("FRED returned missing/unrecognised series")
    frame.to_csv(path)
    return frame


class Downloader:
    def __init__(self, limit):
        self.client = db.Historical(os.environ["DATABENTO_API_KEY"])
        self.limit, self.spent = limit, 0.

    def get(self, label, **kwargs):
        path = PRIVATE/(label+".parquet")
        if path.exists():
            return pd.read_parquet(path)
        cost = self.client.metadata.get_cost(**kwargs)
        if self.spent+cost > self.limit:
            raise RuntimeError(f"Download estimate exceeds the USD {self.limit:.2f} budget")
        print(f"{label}: estimated USD {cost:.6f}", flush=True)
        frame = self.client.timeseries.get_range(**kwargs).to_df()
        self.spent += cost
        frame.to_parquet(path)
        return frame


def chain(dl, dataset, root, day):
    end_day = str(date.fromisoformat(day)+timedelta(days=1))
    defs = dl.get(f"{day}_{root}_definitions", dataset=dataset, schema="definition", stype_in="parent",
                  symbols=root+".OPT", start=day, end=end_day)
    quotes = dl.get(f"{day}_{root}_quotes", dataset=dataset, schema="cbbo-1m", stype_in="parent",
                    symbols=root+".OPT", start=day+"T15:00:00Z", end=day+"T15:05:00Z")
    defs = defs.drop_duplicates("instrument_id", keep="last").set_index("instrument_id")
    quotes = quotes.copy()
    quotes["quote_time"] = quotes.index
    quotes = quotes.sort_index().drop_duplicates("instrument_id", keep="last").set_index("instrument_id")
    needed = ["raw_symbol", "strike_price", "expiration", "instrument_class"]
    for optional in ("underlying", "underlying_id"):
        if optional in defs:
            needed.append(optional)
    merged = quotes.join(defs[needed], rsuffix="_def")
    merged["expiration"] = pd.to_datetime(merged.expiration, utc=True)
    merged["instrument_class"] = merged.instrument_class.map(lambda x: x.decode() if isinstance(x, bytes) else str(x))
    asof = pd.Timestamp(day+"T15:05:00Z")
    merged["expiry_years"] = (merged.expiration-asof).dt.total_seconds()/(365*86400)
    merged = merged[merged.instrument_class.isin(["C", "P"]) & (merged.expiry_years > 0)]
    return clean_quotes(merged, asof), asof


def choose_pair(q, target_years, spot, curve, max_relative_mismatch=0.5):
    candidates = []
    for expiry, group in q.groupby("expiration"):
        T = float(group.expiry_years.iloc[0])
        if abs(T-target_years) > max_relative_mismatch*target_years:
            continue
        calls = group[group.instrument_class == "C"].set_index("strike_price")
        puts = group[group.instrument_class == "P"].set_index("strike_price")
        joined = calls.join(puts, lsuffix="_c", rsuffix="_p", how="inner")
        for strike, row in joined.iterrows():
            if abs((row.quote_time_c-row.quote_time_p).total_seconds()) > 60:
                continue
            p = float(curve.discount(T))
            forward = float(strike+(row.mid_c-row.mid_p)/p)
            if forward <= 0 or abs(np.log(strike/forward)) > 0.025 or not .5 < forward/spot < 2:
                continue
            try:
                iv = implied_vol(row.mid_c, forward, strike, T, p)
                bid_iv = implied_vol(row.bid_px_00_c, forward, strike, T, p)
                ask_iv = implied_vol(row.ask_px_00_c, forward, strike, T, p)
            except ValueError:
                continue
            score = abs(T-target_years)/target_years + 5*abs(np.log(strike/forward))
            candidates.append((score, {"expiry": str(expiry), "expiry_years": T, "strike": float(strike),
                "forward": forward, "iv": iv, "bid_iv": bid_iv, "ask_iv": ask_iv,
                "mid": float(row.mid_c), "discount": p, "symbol": str(row.raw_symbol_c),
                "quote_time": str(row.quote_time_c), "put_quote_time": str(row.quote_time_p),
                "forward_method": "European call-put parity at matched strike/expiry",
                "repricing_error": abs(black_price(forward, strike, T, p, iv)-row.mid_c)}))
    if not candidates:
        raise ValueError("No liquid, synchronised near-ATM pair in requested maturity range")
    return min(candidates, key=lambda z: z[0])[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asof", default=str(date.today()))
    parser.add_argument("--max-cost-usd", type=float, default=0.50)
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    end = date.fromisoformat(args.asof)
    ids = [f"THREEFY{i}" for i in range(1, 11)]
    nodes = fred(ids, end-timedelta(days=30), end).dropna()
    if nodes.empty:
        raise ValueError("No common-date zero curve")
    day = str(nodes.index[-1].date())
    if (end-nodes.index[-1].date()).days > 14:
        raise ValueError("Zero curve more than 14 calendar days stale")
    curve = DiscountCurve(np.arange(1, 11), nodes.iloc[-1].to_numpy()/100)
    history = fred(["DGS3MO", "SP500", "DEXUSEU"],
                   nodes.index[-1].date()-timedelta(days=3*365), nodes.index[-1].date()).dropna()
    if history.empty or history.index[-1].date() != nodes.index[-1].date():
        raise ValueError("Curve and latest common spot observations must share an as-of date")
    corr, corr_info = estimate_correlation(history)
    # Weekly rate increments: transparent historical diffusion-scale proxy.
    weekly = history.groupby(history.index.to_period("W-FRI")).tail(1)
    rate_sigma = float((weekly.DGS3MO.diff()/100).std(ddof=1)*np.sqrt(52))
    spot, fx = float(history.SP500.iloc[-1]), float(history.DEXUSEU.iloc[-1])
    model = HullWhite(curve, 0.05, rate_sigma)
    dl = Downloader(args.max_cost_usd)
    eq_chain, _ = chain(dl, "OPRA.PILLAR", "SPX", day)
    fx_chain, _ = chain(dl, "GLBX.MDP3", "EUU", day)
    eq = choose_pair(eq_chain, 2., spot, curve)
    fxq = choose_pair(fx_chain, 1., fx, curve)
    eq_sigma = model.asset_sigma(eq["iv"], eq["expiry_years"], corr[0,1])
    fx_sigma = model.asset_sigma(fxq["iv"], fxq["expiry_years"], corr[0,2])
    q = -np.log(eq["forward"]*eq["discount"]/spot)/eq["expiry_years"]
    rf = -np.log(fxq["forward"]*fxq["discount"]/fx)/fxq["expiry_years"]
    if not -0.05 < q < 0.15 or not -0.05 < rf < 0.20:
        raise ValueError("Unreasonable implied carry; check spot/forward synchronisation")
    data = {"metadata": {"kind": "market", "asof": day, "requested_asof": args.asof,
            "curve_source": "FRED THREEFY1..10 fitted continuously compounded zeros",
            "spot_source": "FRED common-date daily observations; equity close and FX noon fixing are not synchronous with 15:05 UTC options",
            "carry": "Effective flat carry from parity forward and same-date FRED spot; timestamp basis approximation",
            "correlation": corr_info, "rate_volatility": "historical weekly DGS3MO change std * sqrt(52)",
            "mean_reversion": "assumed 0.05/year",
            "vendor_cost_estimate_usd": dl.spent,
            "fx_proxy": "European EUU futures-option Black IV used as a spot-FX proxy; futures/forward convexity and delivery-date basis omitted",
            "expiry_convention": "Vendor definition expiry timestamps, ACT/365; index fixing/settlement calendar not separately modelled",
            "flat_vol_extrapolation": "FX volatility is held flat beyond the calibration expiry for XCCY scenarios"},
            "curve": {"tenors": curve.tenors.tolist(), "zero_rates": curve.zero_rates.tolist()},
            "hw_a": 0.05, "hw_sigma": rate_sigma, "eq_spot": spot, "fx_spot": fx,
            "eq_vol": eq_sigma, "fx_vol": fx_sigma, "eq_dividend": float(q),
            "foreign_rate": float(rf), "correlation": corr.tolist(),
            "iv_quotes": {"equity": eq, "fx": fxq}}
    with open("data/local_market.json", "w") as f:
        json.dump(data, f, indent=2)
    print(json.dumps({"asof": day, "weekly_changes": corr_info["weekly_changes"],
                      "equity_iv": eq["iv"], "fx_iv": fxq["iv"],
                      "max_option_repricing_error": max(eq["repricing_error"], fxq["repricing_error"]),
                      "saved": "data/local_market.json"}, indent=2))


if __name__ == "__main__":
    main()
