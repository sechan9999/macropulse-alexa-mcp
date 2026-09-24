"""
tests/test_mcp_regime.py
─────────────────────────────────────────────────────────────────
The Alexa/MCP regime uses the dashboard's point-in-time model: FRED BAA-AAA credit spread plus
S&P 500 12m realised vol, expanding z-scores over the history since 2005, no VIX-based proxy.
Offline: prices come from the patched download seam, FRED from tests/_macro_fakes.py.
"""
import time
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import src.alexa_mcp_server as srv
from src import macro_model as mm
from tests._macro_fakes import fake_fred
from tests.test_market_data_layer import DataLayerCase, fake_prices


class TestRegimeModel(DataLayerCase):
    def test_same_model_as_dashboard(self):
        df = srv._fetch_cached_macro_data()
        self.assertGreaterEqual(df.index.min(), pd.Timestamp(srv.MACRO_START))
        self.assertEqual(df.index.min().day, 1)                          # month-start labels, like the app
        spread, slope = fake_fred()
        ref = pd.DataFrame({"sp500": df["sp500"]})
        ref["sp500_ret_m"] = np.log(ref["sp500"]).diff()
        ref["realized_vol_12m"] = ref["sp500_ret_m"].rolling(12).std() * np.sqrt(12)
        ref = mm.add_regime(mm.attach_credit_and_slope(ref, spread, slope))
        pd.testing.assert_series_equal(df["regime_score"], ref["regime_score"])
        self.assertEqual(df["_credit_source"].iloc[-1], "FRED:BAA-AAA")

    def test_vix_no_longer_moves_the_regime(self):
        base = srv._fetch_cached_macro_data()["regime_score"]
        srv._cache.clear()
        self.raw.stop()
        def vix_spike(t, p, timeout):
            px = fake_prices(t, n=5600)
            if t == "^VIX":                     # a spike in the last two years only (a uniform scale
                px.iloc[-500:] *= 3.0           # would leave any z-score unchanged)
            return px

        with mock.patch.object(srv, "_raw_download", side_effect=vix_spike):
            shocked = srv._fetch_cached_macro_data()["regime_score"]
        self.raw.start()
        pd.testing.assert_series_equal(base, shocked)

    def test_past_regime_ignores_later_prices(self):
        base = srv._fetch_cached_macro_data()
        cut = base.index[-24]
        srv._cache.clear()
        self.raw.stop()

        def crash_after_cut(t, p, timeout):
            px = fake_prices(t, n=5600)
            if t == "^GSPC":
                px.loc[px.index > cut + pd.offsets.MonthEnd(1), "Close"] *= 0.5
            return px

        with mock.patch.object(srv, "_raw_download", side_effect=crash_after_cut):
            after = srv._fetch_cached_macro_data()
        self.raw.start()
        pd.testing.assert_series_equal(base.loc[:cut, "regime_score"], after.loc[:cut, "regime_score"])


class TestWithoutFred(DataLayerCase):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(srv, "_cached_fred", return_value=(None, None))
        p.start()
        self.addCleanup(p.stop)

    def test_regime_tool_refuses_to_guess(self):
        with self.assertRaises(srv.MarketDataUnavailable):
            srv.execute_get_macro_regime()

    def test_rates_tool_reports_the_spread_as_unavailable(self):
        r = srv.execute_get_rates_and_spreads()
        self.assertIsNone(r["credit_spread_bps"])
        self.assertEqual(r["credit_spread_source"], "unavailable")
        self.assertIn("temporarily unavailable", r["alexa_spoken_response"])


class TestRatesSpeech(DataLayerCase):
    def test_spread_is_described_against_its_history_not_a_fixed_phrase(self):
        r = srv.execute_get_rates_and_spreads()
        self.assertEqual(r["credit_spread_bps"], round(fake_fred()[0].iloc[-1] * 100, 0))
        self.assertNotIn("stable liquidity", r["alexa_spoken_response"])
        self.assertIn("average since 2005", r["alexa_spoken_response"])


class TestFredCache(unittest.TestCase):
    def setUp(self):
        srv._fred_state.update(fetched_at=None, data=None, failed_at=None, refreshing=False)
        self.addCleanup(srv._fred_state.update, fetched_at=None, data=None, failed_at=None, refreshing=False)

    def test_failure_is_not_retried_on_every_request(self):
        with mock.patch.object(srv, "load_fred_credit_and_slope", return_value=(None, None)) as load:
            self.assertEqual(srv._cached_fred(block=True), (None, None))
            self.assertEqual(srv._cached_fred(), (None, None))
            self.assertEqual(srv._cached_fred(block=True), (None, None))
        self.assertEqual(load.call_count, 1)

    def test_a_request_never_waits_on_fred(self):
        """Empty cache: the request answers at once and the fetch runs in the background."""
        def slow(*a, **k):
            time.sleep(0.5)
            return fake_fred()
        with mock.patch.object(srv, "load_fred_credit_and_slope", side_effect=slow) as load:
            t0 = time.perf_counter()
            self.assertEqual(srv._cached_fred(), (None, None))
            self.assertLess(time.perf_counter() - t0, 0.1)
            for _ in range(100):
                if srv._fred_state["data"] is not None and not srv._fred_state["refreshing"]:
                    break
                time.sleep(0.02)
            self.assertIsNotNone(srv._cached_fred()[0])                       # served once fetched
        self.assertEqual(load.call_count, 1)

    def test_stale_copy_is_served_while_refreshing(self):
        with mock.patch.object(srv, "load_fred_credit_and_slope", return_value=fake_fred()) as load:
            first = srv._cached_fred(block=True)                              # start-up warm-up
            srv._fred_state["fetched_at"] -= srv._FRED_TTL_SECONDS + 60       # make it stale
            t0 = time.perf_counter()
            second = srv._cached_fred()
            self.assertLess(time.perf_counter() - t0, 0.05)                   # no waiting on FRED
            for _ in range(50):                                               # background refresh ran
                if load.call_count == 2 and not srv._fred_state["refreshing"]:
                    break
                time.sleep(0.02)
        self.assertIs(second, first)
        self.assertEqual(load.call_count, 2)


if __name__ == "__main__":
    unittest.main()
