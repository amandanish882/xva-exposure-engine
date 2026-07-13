import unittest
import numpy as np
from src.instruments import FXForward, EuropeanOption

class TestBasics(unittest.TestCase):
    def test_fx_forward_sign(self):
        class MockMD:
            eq_vol = 0.2
            fx_vol = 0.1

        ctx = {
            "time": np.array([0.0, 1.0]),
            "r": np.array([[0.02, 0.02]]),
            "FX": np.array([[1.20, 1.20]]),
            "S": np.array([[4000.0, 4000.0]]),
            "df": np.array([[1.0, np.exp(-0.02)]]),
            "md": MockMD()
        }

        fwd = FXForward("f", notional_foreign=100000.0, strike=1.10, maturity=1.0)
        v = fwd.value(0, ctx)[0]
        self.assertTrue(v > 0.0)

    def test_option_payoff_nonnegative(self):
        class MockMD:
            eq_vol = 0.2
            fx_vol = 0.1

        ctx = {
            "time": np.array([2.0]),
            "r": np.array([[0.02]]),
            "S": np.array([[4200.0]]),
            "FX": np.array([[1.1]]),
            "df": np.array([[1.0]]),
            "md": MockMD()
        }

        opt = EuropeanOption("o", "S", strike=4100.0, maturity=2.0, opt_type="call")
        payoff = opt.value(0, ctx)[0]
        self.assertTrue(payoff >= 0.0)

if __name__ == "__main__":
    unittest.main()
