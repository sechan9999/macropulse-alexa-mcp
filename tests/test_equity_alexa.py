"""
tests/test_equity_alexa.py
─────────────────────────────────────────────────────────────────
Equity report over Alexa+ / MCP, offline: the price cache, the macro regime and SEC EDGAR are
patched with synthetic data, so these run without network access.
"""
import json
import time
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import src.alexa_mcp_server as srv
from src.alexa_agent_skill import AlexaMacroSkill
from src.equity_report.engine import data_us, synthetic


def _fake_download(ticker, period, timeout=8):
    if ticker == "^GSPC":
        return synthetic.price_frame(start=5200, drift=0.0004, vol=0.010, seed=5)
    return synthetic.price_frame(seed=3)


def _fake_macro(series=None):
    if series is srv._DRIFT_SERIES:                  # 10y history for the Ridge S&P 500 view
        return _long_macro()
    idx = pd.date_range(end=pd.Timestamp.today(), periods=3, freq="MS")
    return pd.DataFrame({"regime": ["Neutral 🟡", "Neutral 🟡", "Risk-Off 🔴"], "dgs10": [4.1, 4.2, 4.4]}, index=idx)


def _long_macro(n=120, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="ME")
    ret = rng.normal(0.007, 0.04, n)
    return pd.DataFrame({"sp500": 1000 * np.exp(np.cumsum(ret)), "dgs10": 3 + np.cumsum(rng.normal(0, 0.1, n)),
                         "credit_spread": 0.02 + np.abs(rng.normal(0, 0.005, n))}, index=idx)


def _fake_sec(url):
    if url.endswith("company_tickers.json"):
        return {"0": {"cik_str": 1234567, "ticker": "FIX", "title": "FIXTURE INC"},
                "1": {"cik_str": 1036, "ticker": "TSM", "title": "TAIWAN SEMICONDUCTOR"}}
    if "CIK0000001036" in url:
        return {"facts": {"ifrs-full": {"Revenue": {}}}}          # 20-F filer
    return synthetic.company_facts_fixture()


class TestEquityOverAlexa(unittest.TestCase):
    def setUp(self):
        data_us._mem.clear()
        srv._equity_cache.clear()
        self.patches = [mock.patch.object(srv, "_cached_download", side_effect=_fake_download),
                        mock.patch.object(srv, "_fetch_cached_macro_data", side_effect=_fake_macro),
                        mock.patch.object(srv, "_data_freshness", return_value={"data_as_of": "test"}),
                        mock.patch.object(data_us, "_sec_get_json", side_effect=_fake_sec)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_tool_returns_regime_weighted_summary(self):
        res = srv.execute_get_equity_report("fix")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["ticker"], "FIX")
        self.assertEqual(res["macro_regime"]["regime"], "Risk-Off")
        self.assertEqual(res["scenarios"]["Bear"]["prob"], 0.40)
        for k in ("bullish_count", "bearish_count", "bull_score", "bear_score", "overall"):
            self.assertIn(k, res["pattern_screen"])
        self.assertIn("?ticker=FIX", res["report_url"])
        self.assertIn("Ridge", res["drift"]["source"])               # touch probabilities use the Ridge drift
        self.assertIsNotNone(res["drift"]["stock_expected_return"])
        self.assertEqual(set(res["fomc_overlay"]), {"hawkish_50bps", "dovish_50bps", "stagflation_inversion",
                                                    "liquidity_cascade"})
        self.assertGreater(len(res["alexa_spoken_response"]), 40)
        json.dumps(res)

    def test_second_call_is_cached_and_fast(self):
        srv.execute_get_equity_report("FIX")
        t0 = time.perf_counter()
        srv.execute_get_equity_report("FIX")
        self.assertLess(time.perf_counter() - t0, 0.5)          # Alexa+ 500 ms budget

    def test_short_macro_history_falls_back_to_driftless(self):
        with mock.patch.object(srv, "_fetch_cached_macro_data", side_effect=lambda series=None: _fake_macro()):
            res = srv.execute_get_equity_report("FIX")
        self.assertEqual(res["drift"]["source"], "none (driftless)")
        self.assertEqual(res["touch_prob_6m"], res["touch_prob_6m_driftless"])

    def test_foreign_filer_is_refused_not_guessed(self):
        with self.assertRaises(srv.EquityReportUnsupported):
            srv.execute_get_equity_report("TSM")
        with self.assertRaises(srv.EquityReportUnsupported):
            srv.execute_get_equity_report("^GSPC")

    def test_voice_routing_card_and_apl(self):
        res = AlexaMacroSkill.execute("Alexa, show the FIX equity report on the TV")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["tool_selected"], "get_equity_report")
        self.assertTrue(res["is_tv_request"])
        self.assertIn("Equity Report", res["display_card"]["title"])
        self.assertEqual(len(res["display_card"]["badges"]), 4)
        apl = json.dumps(res["apl_document"])
        self.assertIn("Bear $", apl)                              # scenario footer on the TV card
        self.assertTrue(res["spoken_response"].startswith("Displaying"))

    def test_unsupported_ticker_spoken_error(self):
        res = AlexaMacroSkill.execute("Alexa, give me an equity report on TSM")
        self.assertEqual(res["status"], "error")
        self.assertIn("10-K", res["spoken_response"])

    def test_ask_webhook_equity_intent(self):
        body = {"request": {"type": "IntentRequest",
                            "intent": {"name": "EquityReportIntent", "slots": {"Ticker": {"value": "FIX"}}}}}
        out = AlexaMacroSkill.handle_ask_request(body)
        self.assertIn("FIX", out["response"]["card"]["title"])
        self.assertEqual(out["response"]["directives"][0]["token"], "macropulse-get_equity_report")


class TestIntentRouting(unittest.TestCase):
    CASES = {
        "Alexa, give me an equity report on AAPL": ("get_equity_report", "AAPL"),
        "Alexa, what is the fair value of Apple?": ("get_equity_report", "AAPL"),
        "Alexa, show the Nvidia report on the TV": ("get_equity_report", "NVDA"),
        "Alexa, run a DCF on brk.b": ("get_equity_report", "BRK-B"),
        "Alexa, valuation for Costco": ("get_equity_report", "COST"),
    }
    UNCHANGED = {   # existing presets must keep their routing
        "Alexa, ask MacroPulse for today's market regime": "get_macro_regime",
        "Alexa, show the NVDA Danger Zone on the TV": "check_nvda_danger_zone",
        "Alexa, run an FOMC rate shock test on my portfolio": "simulate_fomc_shock",
        "Alexa, run a 99% risk simulation on SPY over the next 30 days": "simulate_portfolio_risk",
        "Alexa, scan volatility squeeze breakouts on SPY": "scan_quant_signals",
        "Alexa, what is the Value at Risk for QQQ?": "simulate_portfolio_risk",
        "Alexa, check the 10-year Treasury yield and yield curve slope.": "get_rates_and_spreads",
    }

    def test_equity_prompts(self):
        for prompt, (tool, ticker) in self.CASES.items():
            got, args = AlexaMacroSkill.identify_intent(prompt)
            self.assertEqual((got, args.get("ticker")), (tool, ticker), prompt)

    def test_existing_prompts_unchanged(self):
        for prompt, tool in self.UNCHANGED.items():
            self.assertEqual(AlexaMacroSkill.identify_intent(prompt)[0], tool, prompt)


if __name__ == "__main__":
    unittest.main()
