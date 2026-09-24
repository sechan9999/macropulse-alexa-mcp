"""
tests/test_macro_model.py
─────────────────────────────────────────────────────────────────
Point-in-time guarantees of src/macro_model.py (regime score, FRED loading, expected-return model)
and of the offline regime-conditional model in src/exp_return.py. The core check is the same in
every test: changing data *after* a date must not change anything computed *at* that date.
"""
import io
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src import macro_model as mm
from src.exp_return import train_regime_conditional_alpha


def _macro_frame(n=180, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-01", periods=n, freq="MS")
    d = pd.DataFrame(index=idx)
    d["sp500_ret_m"] = rng.normal(0.006, 0.04, n)
    d["realized_vol_12m"] = d["sp500_ret_m"].rolling(12).std() * np.sqrt(12)
    d["realized_vol_3m"] = d["sp500_ret_m"].rolling(3).std() * np.sqrt(12)
    d["momentum_12_1"] = d["sp500_ret_m"].rolling(11).sum().shift(1)
    d["dgs10"] = 3 + np.cumsum(rng.normal(0, 0.1, n))
    d["credit_spread"] = 0.01 + np.abs(np.cumsum(rng.normal(0, 0.0005, n)))
    d["yc_slope"] = np.cumsum(rng.normal(0, 0.001, n))
    return d


class TestRegimeIsPointInTime(unittest.TestCase):
    def test_expanding_z_ignores_later_values(self):
        s = pd.Series(np.random.default_rng(1).normal(size=120))
        z = mm.expanding_z(s)
        s2 = s.copy()
        s2.iloc[80:] += 50.0                                   # huge shock after row 79
        pd.testing.assert_series_equal(z.iloc[:80], mm.expanding_z(s2).iloc[:80])

    def test_regime_ignores_later_values(self):
        d = _macro_frame()
        base = mm.add_regime(d)
        shocked = d.copy()
        shocked.loc[shocked.index[120]:, ["credit_spread", "realized_vol_12m"]] *= 5.0
        after = mm.add_regime(shocked)
        pd.testing.assert_series_equal(base["regime"].iloc[:120], after["regime"].iloc[:120])
        pd.testing.assert_series_equal(base["regime_score"].iloc[:120], after["regime_score"].iloc[:120])

    def test_warmup_and_missing_data_are_unavailable_not_neutral(self):
        d = _macro_frame()
        out = mm.add_regime(d)
        self.assertEqual(out["regime"].iloc[0], mm.UNAVAILABLE)
        self.assertTrue(np.isnan(out["regime_score"].iloc[0]))
        self.assertIn(out["regime"].iloc[-1], {mm.RISK_ON, mm.NEUTRAL, mm.RISK_OFF})

        no_fred = mm.add_regime(mm.attach_credit_and_slope(d.drop(columns=["credit_spread", "yc_slope"]),
                                                           None, None))
        self.assertTrue((no_fred["regime"] == mm.UNAVAILABLE).all())
        self.assertTrue(no_fred["regime_score"].isna().all())
        self.assertEqual(no_fred["_credit_source"].iloc[-1], "unavailable")

    def test_stored_proxy_spreads_are_discarded(self):
        d = _macro_frame()
        d["_credit_source"] = "proxy"                           # e.g. an old data-lake mart
        d["_slope_source"] = "proxy"
        out = mm.add_regime(d)
        self.assertTrue(out["credit_spread"].isna().all())
        self.assertTrue(out["yc_slope"].isna().all())
        self.assertTrue((out["regime"] == mm.UNAVAILABLE).all())
        self.assertFalse(mm.is_fred_sourced(d, "_credit_source"))


class TestAsOfDate(unittest.TestCase):
    def test_uses_the_last_close_not_the_month_label(self):
        d = _macro_frame(24)
        d["_obs_date"] = d.index + pd.Timedelta(days=23)
        self.assertEqual(mm.as_of_date(d), d.index[-1] + pd.Timedelta(days=23))

    def test_without_observation_dates_it_is_never_in_the_future(self):
        today = pd.Timestamp.today().normalize()
        d = _macro_frame(24)
        d.index = pd.date_range(end=today, periods=24, freq="MS")
        self.assertEqual(mm.as_of_date(d), min(d.index[-1] + pd.offsets.MonthEnd(0), today))


class TestFredLoading(unittest.TestCase):
    CSV = b"observation_date,BAA\n2024-01-01,5.9\n2024-02-01,.\n2024-03-01,6.1\n"

    def test_keyless_csv_parsing_skips_missing_values(self):
        resp = mock.MagicMock()
        resp.__enter__.return_value.read.return_value = self.CSV
        with mock.patch("urllib.request.urlopen", return_value=resp):
            s = mm.fred_series("BAA", "2024-01-01", api_key="")
        self.assertEqual(list(s.values), [5.9, 6.1])
        self.assertEqual(s.index[0], pd.Timestamp("2024-01-01"))

    def test_api_key_uses_the_official_api(self):
        body = (b'{"observations": [{"date": "2024-01-01", "value": "5.9"},'
                b' {"date": "2024-02-01", "value": "."}, {"date": "2024-03-01", "value": "6.1"}]}')
        resp = mock.MagicMock()
        resp.__enter__.return_value.read.return_value = body
        with mock.patch("urllib.request.urlopen", return_value=resp) as op:
            s = mm.fred_series("BAA", "2024-01-01", api_key="SECRETKEY")
        url = op.call_args[0][0].full_url
        self.assertTrue(url.startswith(mm.FRED_API_URL))
        self.assertIn("series_id=BAA", url)
        self.assertEqual(list(s.values), [5.9, 6.1])

    def test_api_failure_falls_back_to_csv_and_never_logs_the_key(self):
        csv = mock.MagicMock()
        csv.__enter__.return_value.read.return_value = self.CSV

        def urlopen(req, timeout=None):
            if req.full_url.startswith(mm.FRED_API_URL):
                raise OSError("Bad Request for key SECRETKEY")
            return csv

        with mock.patch("urllib.request.urlopen", side_effect=urlopen), \
                self.assertLogs("src.macro_model", level="WARNING") as logs:
            s = mm.fred_series("BAA", "2024-01-01", api_key="SECRETKEY")
        self.assertEqual(list(s.values), [5.9, 6.1])
        self.assertNotIn("SECRETKEY", "\n".join(logs.output))

    def test_unreachable_fred_returns_none_instead_of_a_proxy(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("offline")):
            spread, slope = mm.load_fred_credit_and_slope("2020-01-01", "2024-01-01", api_key="")
        self.assertIsNone(spread)
        self.assertIsNone(slope)


class TestFredFailureIsNotCached(unittest.TestCase):
    def test_headless_pipeline_retries_after_the_backoff(self):
        from src import macro_data
        spread = pd.Series([1.0, 1.1], index=pd.date_range("2024-01-01", periods=2, freq="MS"))
        results = [(None, None), (spread, spread)]
        macro_data._fred_failed_at[0] = 0.0
        with mock.patch.object(macro_data, "load_fred_credit_and_slope", side_effect=results) as load:
            self.assertEqual(macro_data._try_load_fred_series("2031-01-01", "2031-02-01"), (None, None))
            self.assertEqual(macro_data._try_load_fred_series("2031-01-01", "2031-02-01"), (None, None))
            self.assertEqual(load.call_count, 1)                    # backing off, not hammering FRED
            macro_data._fred_failed_at[0] = 0.0                     # five minutes later
            got = macro_data._try_load_fred_series("2031-01-01", "2031-02-01")
        self.assertIs(got[0], spread)                               # the failure was not cached for a day
        self.assertEqual(load.call_count, 2)


class TestExpectedReturnModel(unittest.TestCase):
    def test_trains_only_on_realised_targets(self):
        """For the prediction at row i, every training row j satisfies j < i - horizon."""
        d = _macro_frame(120)
        d["y_fwd"] = d["sp500_ret_m"].rolling(12).sum().shift(-12)
        cols = ["dgs10", "credit_spread", "yc_slope"]
        seen = []
        real = mm._ridge_pipeline

        class Recorder:
            def __init__(self, alpha):
                self.m = real(alpha)

            def fit(self, X, y):
                self.last_train_end = X.index.max()
                return self.m.fit(X, y)

            def predict(self, X):
                seen.append((X.index[0], self.last_train_end))
                return self.m.predict(X)

        with mock.patch.object(mm, "_ridge_pipeline", Recorder):
            mm.walk_forward_ridge(d, cols, "y_fwd", horizon=12, min_train=36)
        self.assertTrue(seen)
        pos = {t: k for k, t in enumerate(d.index)}
        for test_t, train_end in seen:
            self.assertLess(pos[train_end], pos[test_t] - 12)

    def test_outputs_ignore_returns_after_each_date(self):
        d = _macro_frame()
        cut = d.index[140]
        base = mm.expected_return_model(d)
        shocked = d.copy()
        shocked.loc[shocked.index > cut, "sp500_ret_m"] += 0.10   # rewrite the future
        after = mm.expected_return_model(shocked)
        for col in ("exp_ann_return", "pred_std"):
            pd.testing.assert_series_equal(base.loc[:cut, col], after.loc[:cut, col])

    def test_last_row_is_todays_forecast(self):
        d = _macro_frame()
        out = mm.expected_return_model(d)
        self.assertEqual(out.index[-1], d.index[-1])           # not 12 months stale
        self.assertTrue(np.isnan(out["realized_fwd_12m"].iloc[-1]))   # outcome not known yet
        self.assertTrue(np.isfinite(out["exp_ann_return"].iloc[-1]))


class TestOfflineRegimeModel(unittest.TestCase):
    def test_regime_conditional_alpha_ignores_future_returns(self):
        rng = np.random.default_rng(3)
        n = 90
        idx = pd.date_range("2010-01-01", periods=n, freq="MS")
        X = pd.DataFrame({"x0": np.cumsum(rng.normal(size=n)), "x1": rng.normal(size=n)}, index=idx)
        p0 = rng.uniform(size=n)
        probs = pd.DataFrame({"pR0": p0, "pR1": 1 - p0}, index=idx)
        rets = pd.Series(rng.normal(0, 0.04, n), index=idx)

        def run(r):
            y = pd.DataFrame({"SPX": r.rolling(12).sum().shift(-12)})
            return train_regime_conditional_alpha(X, y, probs, ["SPX"], ["x0", "x1"], min_train=30)

        cut = idx[70]
        shocked = rets.copy()
        shocked[shocked.index > cut] += 0.2
        pd.testing.assert_series_equal(run(rets).loc[:cut, "SPX"], run(shocked).loc[:cut, "SPX"])


if __name__ == "__main__":
    unittest.main()
