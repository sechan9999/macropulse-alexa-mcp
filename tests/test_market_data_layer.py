"""
tests/test_market_data_layer.py
─────────────────────────────────────────────────────────────────
The market-data layer behind the Alexa/MCP tools: TTL cache with background refresh,
single-flight fetching, warm-up, and the rule that an outage produces an error, never
made-up numbers. No network: the single download seam (_raw_download) is patched.
"""
import os
import threading
import time
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from starlette.testclient import TestClient

import src.alexa_mcp_server as srv
from src.alexa_agent_skill import AlexaMacroSkill

START = {"^VIX": 18.0, "^TNX": 4.2, "^IRX": 4.0}
DEFAULT_TOOLS = (
    srv.execute_get_macro_regime, srv.execute_get_rates_and_spreads,
    srv.execute_simulate_portfolio_risk, srv.execute_check_nvda_danger_zone,
    srv.execute_scan_quant_signals,
)


def fake_prices(ticker="SPY", n=800):
    rng = np.random.default_rng(abs(hash(ticker)) % 2**32)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    close = START.get(ticker, 100.0) * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                         "Volume": rng.integers(1_000_000, 2_000_000, n).astype(float)}, index=idx)


class DataLayerCase(unittest.TestCase):
    def setUp(self):
        for d in (srv._cache, srv._key_locks):
            d.clear()
        srv._refreshing.clear()
        self.calls = []
        self.raw = mock.patch.object(srv, "_raw_download", side_effect=self._fake)
        self.raw.start()
        self.addCleanup(self.raw.stop)
        self.addCleanup(srv._cache.clear)

    def _fake(self, ticker, period, timeout):
        self.calls.append((ticker, period))
        return fake_prices(ticker)

    def age(self, ticker, period, seconds):
        """Make a cached series look `seconds` old."""
        fetched_at, df = srv._cache[(ticker, period)]
        srv._cache[(ticker, period)] = (time.monotonic() - seconds, df)

    def outage(self):
        self.raw.stop()
        p = mock.patch.object(srv, "_raw_download", side_effect=RuntimeError("yahoo down"))
        p.start()
        self.addCleanup(p.stop)
        self.raw = p


class TestCache(DataLayerCase):
    def test_second_call_is_served_from_cache_as_a_private_copy(self):
        first = srv._cached_download("SPY", "1y")
        first["Close"] *= 0
        second = srv._cached_download("spy", "1y")  # same series, different spelling
        self.assertEqual(len(self.calls), 1)
        self.assertGreater(second["Close"].iloc[-1], 0)

    def test_invalid_ticker_rejected(self):
        with self.assertRaises(ValueError):
            srv._cached_download("SPY; DROP TABLE", "1y")
        self.assertEqual(self.calls, [])

    def test_concurrent_callers_share_one_download(self):
        def slow(ticker, period, timeout):
            self.calls.append(ticker)
            time.sleep(0.2)
            return fake_prices(ticker)
        self.raw.stop()
        with mock.patch.object(srv, "_raw_download", side_effect=slow):
            threads = [threading.Thread(target=srv._cached_download, args=("SPY", "1y")) for _ in range(8)]
            [t.start() for t in threads]
            [t.join() for t in threads]
        self.raw.start()
        self.assertEqual(len(self.calls), 1)

    def test_stale_data_is_served_instantly_while_refreshing_in_background(self):
        srv._cached_download("SPY", "1y")
        self.age("SPY", "1y", srv._CACHE_TTL_SECONDS + 10)
        gate = threading.Event()

        def blocked(ticker, period, timeout):
            self.calls.append(ticker)
            gate.wait(5)
            return fake_prices(ticker)
        self.raw.stop()
        with mock.patch.object(srv, "_raw_download", side_effect=blocked):
            t0 = time.perf_counter()
            stale = srv._cached_download("SPY", "1y")
            self.assertLess(time.perf_counter() - t0, 0.5)   # did not wait for the refresh
            self.assertFalse(stale.empty)
            gate.set()
            deadline = time.time() + 5
            while srv._refreshing and time.time() < deadline:
                time.sleep(0.02)
        self.raw.start()
        self.assertFalse(srv._refreshing)
        self.assertLess(time.monotonic() - srv._cache[("SPY", "1y")][0], srv._CACHE_TTL_SECONDS)

    def test_data_past_the_hard_limit_is_never_served(self):
        srv._cached_download("SPY", "1y")
        self.age("SPY", "1y", srv._CACHE_MAX_STALE_SECONDS + 10)
        self.outage()
        with self.assertRaises(srv.MarketDataUnavailable):
            srv._cached_download("SPY", "1y")

    def test_cache_is_bounded(self):
        for i in range(srv._CACHE_MAX_ENTRIES + 10):
            srv._cached_download(f"T{i}", "1y")
        self.assertLessEqual(len(srv._cache), srv._CACHE_MAX_ENTRIES)


class TestNoMadeUpData(DataLayerCase):
    def test_every_data_tool_raises_when_yahoo_is_down(self):
        self.outage()
        for fn in DEFAULT_TOOLS:
            with self.subTest(tool=fn.__name__), self.assertRaises(srv.MarketDataUnavailable):
                fn()

    def test_empty_download_is_an_error(self):
        self.raw.stop()
        with mock.patch.object(srv, "_raw_download", return_value=pd.DataFrame()):
            with self.assertRaises(srv.MarketDataUnavailable):
                srv.execute_scan_quant_signals("ZZZZ")
        self.raw.start()

    def test_serving_stale_is_flagged(self):
        srv.execute_check_nvda_danger_zone()
        self.age("NVDA", "6mo", srv._CACHE_TTL_SECONDS + 30)
        self.outage()  # background refresh will fail, cached copy keeps being served
        result = srv.execute_check_nvda_danger_zone()
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["data_stale"])
        self.assertGreaterEqual(result["data_age_seconds"], srv._CACHE_TTL_SECONDS)

    def test_tools_succeed_on_fresh_data_and_report_freshness(self):
        for fn in DEFAULT_TOOLS:
            with self.subTest(tool=fn.__name__):
                result = fn()
                self.assertEqual(result["status"], "success")
                self.assertFalse(result["data_stale"])

    def test_expected_returns_is_labelled_static_and_downloads_nothing(self):
        result = srv.execute_get_expected_returns()
        self.assertEqual(self.calls, [])
        self.assertIn("Static", result["model"])
        self.assertIn("static reference estimate", result["alexa_spoken_response"])


class TestAnalystIsTimeBoxed(DataLayerCase):
    def test_slow_llm_falls_back_to_the_data_driven_answer(self):
        def slow_llm(query, macro_summary, api_key):
            time.sleep(1.5)
            return "late answer"
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}), \
                mock.patch.object(srv, "_generate_llm_briefing", side_effect=slow_llm), \
                mock.patch.object(srv, "_LLM_TIMEOUT_SECONDS", 0.2):
            t0 = time.perf_counter()
            result = srv.execute_ask_macro_analyst("what is the market regime")
        self.assertLess(time.perf_counter() - t0, 1.2)
        self.assertIn("regime", result["alexa_spoken_response"].lower())
        self.assertNotEqual(result["alexa_spoken_response"], "late answer")

    def test_fast_llm_answer_is_used(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}), \
                mock.patch.object(srv, "_generate_llm_briefing", return_value="LLM says hold"):
            self.assertEqual(srv.execute_ask_macro_analyst("q")["alexa_spoken_response"], "LLM says hold")


class TestWarmUp(DataLayerCase):
    def test_warm_up_prefetches_defaults_so_default_tools_download_nothing(self):
        srv._warm_done.clear()
        self.addCleanup(srv._warm_done.clear)
        srv._warm_up_data()
        self.assertTrue(srv._warm_done.is_set())
        for ticker, period, _ in srv._WARM_SET:
            self.assertIn((ticker, period), srv._cache)
        before = len(self.calls)
        for fn in DEFAULT_TOOLS:
            fn()
        self.assertEqual(len(self.calls), before)


class TestOutageIsHandledHonestlyEverywhere(DataLayerCase):
    def test_voice_path_says_it_cannot_reach_data(self):
        self.outage()
        res = AlexaMacroSkill.execute("Alexa, what is today's market regime?")
        self.assertEqual(res["status"], "error")
        self.assertIn("market data", res["spoken_response"])
        self.assertEqual(res["raw_payload"]["error"], "market_data_unavailable")

    def test_ask_webhook_returns_a_valid_spoken_response_during_an_outage(self):
        self.outage()
        req = {"version": "1.0", "session": {"new": True, "sessionId": "s", "user": {"userId": "u"}},
               "request": {"type": "IntentRequest", "requestId": "r", "locale": "en-US",
                           "intent": {"name": "MacroRegimeIntent", "slots": {}}}}
        resp = AlexaMacroSkill.handle_ask_request(req)
        self.assertIn("market data", resp["response"]["outputSpeech"]["text"])
        self.assertTrue(resp["response"]["directives"])

    def test_rest_endpoint_answers_503_and_health_reports_warm_state(self):
        self.outage()
        client = TestClient(srv.build_starlette_app(warm_up=False))
        r = client.post("/alexa/query", json={"tool_name": "check_nvda_danger_zone"})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["error"], "market_data_unavailable")
        health = client.get("/health").json()
        self.assertIn("warm", health)
        self.assertNotIn("ask_macro_analyst", health["available_tools"])


if __name__ == "__main__":
    unittest.main()
