import numpy as np
import pandas_datareader.data as web
import datetime

class MarketData:
    def __init__(self):
        # Rates (Hull-White short-rate, demo calibration)
        self.r0 = 0.04
        self.hw_a = 0.05
        self.hw_sigma = 0.01

        # Equity (S&P proxy)
        self.eq_spot = 4000.0
        self.eq_vol = 0.20

        # FX (EURUSD proxy)
        self.fx_spot = 1.10
        self.fx_vol = 0.10

        # Correlation order: [Rates, Equity, FX]
        self.corr = np.array([
            [ 1.0, -0.3,  0.2],
            [-0.3,  1.0,  0.4],
            [ 0.2,  0.4,  1.0],
        ])

    def calibrate_from_fred(self):
        print("Fetching FRED data for simple calibration...")
        try:
            start = datetime.datetime(2021, 1, 1)
            end = datetime.datetime.now()
            # DGS3MO = 3-Month Treasury Constant Maturity Rate
            # SP500  = S&P 500
            # DEXUSEU = U.S. / Euro Foreign Exchange Rate
            df = web.DataReader(["DGS3MO", "SP500", "DEXUSEU"], "fred", start, end).ffill().dropna()

            rates = df["DGS3MO"] / 100.0
            self.r0 = float(rates.iloc[-1])
            self.hw_sigma = float(np.std(rates.diff().dropna()) * np.sqrt(252))

            eq = df["SP500"]
            eq_ret = np.log(eq / eq.shift(1)).dropna()
            self.eq_spot = float(eq.iloc[-1])
            self.eq_vol = float(np.std(eq_ret) * np.sqrt(252))

            fx = df["DEXUSEU"]
            fx_ret = np.log(fx / fx.shift(1)).dropna()
            self.fx_spot = float(fx.iloc[-1])
            self.fx_vol = float(np.std(fx_ret) * np.sqrt(252))

            print(f"Calibrated: r0={self.r0:.4f} hw_sigma={self.hw_sigma:.4f} eq_vol={self.eq_vol:.4f} fx_vol={self.fx_vol:.4f}")
        except Exception as e:
            print(f"FRED calibration failed: {e}. Using defaults.")

    def shock(self, dr=0.0, eq_vol_mult=1.0, fx_vol_mult=1.0):
        self.r0 += dr
        self.eq_vol *= eq_vol_mult
        self.fx_vol *= fx_vol_mult
