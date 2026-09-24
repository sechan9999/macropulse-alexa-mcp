"""
tests/test_macro_live_first.py
─────────────────────────────────────────────────────────────────
load_macro must use live prices first and fall back to the stored mart only when the live download
fails. It used to return the stored mart whenever one existed, which froze the dashboard on the day
the mart was written (the mart is only rewritten by a live refresh).
"""
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src import macro_data
from tests._macro_fakes import fake_fred


def _daily(ticker, *a, **k):
    idx = pd.bdate_range("2005-01-03", pd.Timestamp.today().normalize())
    rng = np.random.default_rng(len(ticker))
    return pd.DataFrame({"Close": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx))))}, index=idx)


def _stored():
    idx = pd.date_range("2010-01-01", periods=150, freq="MS")
    return pd.DataFrame({"sp500": np.linspace(1000, 2000, 150), "realized_vol_12m": 0.15,
                         "credit_spread": 0.01, "_credit_source": "FRED:BAA-AAA",
                         "yc_slope": 0.005, "_slope_source": "FRED:T10Y2Y"}, index=idx)


class TestLiveFirst(unittest.TestCase):
    def setUp(self):
        for target, kwargs in ((macro_data, dict(attribute="_BQ_OK", new=True)),):
            p = mock.patch.object(target, **kwargs)
            p.start()
            self.addCleanup(p.stop)
        self.store = mock.patch.object(macro_data, "load_macro_from_bigquery", return_value=_stored(), create=True)
        self.stored_loader = self.store.start()
        self.addCleanup(self.store.stop)
        for name, value in (("save_macro_to_bigquery", None), ):
            p = mock.patch.object(macro_data, name, return_value=value, create=True)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(macro_data, "_try_load_fred_series", return_value=fake_fred())
        p.start()
        self.addCleanup(p.stop)

    def test_live_prices_win_over_the_stored_copy(self):
        with mock.patch.object(macro_data.yf, "download", side_effect=_daily):
            df = macro_data.load_macro.__wrapped__()
        self.assertEqual(df["_data_source"].iloc[-1], "Live yfinance")
        self.stored_loader.assert_not_called()
        self.assertEqual(df.index[-1], pd.Timestamp.today().normalize().replace(day=1))   # current month

    def test_stored_copy_is_the_fallback_and_is_labelled(self):
        with mock.patch.object(macro_data.yf, "download", side_effect=OSError("yahoo down")):
            df = macro_data.load_macro.__wrapped__()
        self.assertIn("stored copy", df["_data_source"].iloc[-1])
        self.assertFalse(bool(df.get("_is_demo", pd.Series([False])).iloc[-1]))


if __name__ == "__main__":
    unittest.main()
