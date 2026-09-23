"""
tests/test_morning_brief.py
─────────────────────────────────────────────────────────────────
Morning brief (Alexa Routine) offline: prices come from a patched cache, the daily scan from a
temporary signals/ folder, so these run without network access.
"""
import json
import os
import tempfile
import unittest
from datetime import date, timedelta
from unittest import mock

import numpy as np
import pandas as pd

import src.alexa_mcp_server as srv
from src.alexa_agent_skill import AlexaMacroSkill

SCAN = """⚡ Macropulse Quant Signals — Daily Recommendation (참고용, 매매 지시 아님)

🟢 BUY 후보 (3)
  · QQQ    $741.07  score +45  [Normal]
      정배열 상승추세
  · NVDA   $225.30  score +40  [Elevated]
  · SPY    $768.82  score +40  [Normal]
  · AAPL   $336.21  score +25  [Normal]

🔴 SELL 후보 (3)
  · GLD    $392.42  score -25  [Normal]
  · TLT    $80.62  score -40  [Elevated]
  · JPM    $337.47  score -45  [Normal]
"""


def _series(n, start, drift, vol, seed, end=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end or pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame({"Close": start * np.exp(np.cumsum(rng.normal(drift, vol, n)))}, index=idx)


def _fake_download(ticker, period, timeout=8):
    base = {"^GSPC": (5000, 0.0004, 0.01), "^VIX": (18, 0, 0.03), "^TNX": (4.2, 0, 0.01),
            "^IRX": (3.9, 0, 0.005), "SPY": (500, 0.0004, 0.01)}[ticker]
    n = 25 if period == "1mo" else 260 if period == "1y" else 760
    return _series(n, *base, seed=len(ticker) + n)


class TestSignalLeaders(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.today = date(2026, 9, 23)

    def _write(self, day, text=SCAN):
        with open(os.path.join(self.dir, f"{day.isoformat()}.md"), "w", encoding="utf-8") as fh:
            fh.write(text)

    def test_parses_newest_scan(self):
        self._write(self.today - timedelta(days=5), SCAN.replace("QQQ", "OLD"))
        self._write(self.today - timedelta(days=1))
        lead = srv._latest_signal_leaders(self.dir, today=self.today)
        self.assertEqual(lead["scan_date"], "2026-09-22")
        self.assertEqual([r["ticker"] for r in lead["buy"]], ["QQQ", "NVDA", "SPY"])      # top 3
        self.assertEqual([r["ticker"] for r in lead["sell"]], ["JPM", "TLT", "GLD"])      # most negative first
        self.assertEqual(lead["sell"][0]["score"], -45)

    def test_stale_or_missing_scan_is_left_out(self):
        self.assertIsNone(srv._latest_signal_leaders(self.dir, today=self.today))
        self._write(self.today - timedelta(days=10))
        self.assertIsNone(srv._latest_signal_leaders(self.dir, today=self.today))
        self.assertIsNone(srv._latest_signal_leaders("/nonexistent/signals", today=self.today))

    def test_repo_signals_folder_parses(self):
        lead = srv._latest_signal_leaders(today=date(2026, 9, 23))
        if lead:                                     # the committed scans follow the same format
            self.assertTrue(lead["buy"] or lead["sell"])


class TestMorningBrief(unittest.TestCase):
    def setUp(self):
        srv._cache.clear()
        self.dir = tempfile.mkdtemp()
        with open(os.path.join(self.dir, f"{date.today().isoformat()}.md"), "w", encoding="utf-8") as fh:
            fh.write(SCAN)
        self.patches = [mock.patch.object(srv, "_cached_download", side_effect=_fake_download),
                        mock.patch.object(srv, "_data_freshness", return_value={"data_age_seconds": 0}),
                        mock.patch.object(srv, "_SIGNALS_DIR", self.dir)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_brief_contents(self):
        b = srv.execute_get_morning_brief()
        self.assertEqual(b["status"], "success")
        self.assertEqual(b["sections_skipped"], [])
        self.assertIsNotNone(b["sp500_change_pct"])
        self.assertIn(b["spy_signal"]["signal"], ("STRONG_BUY", "BUY", "HOLD", "SELL"))
        speech = b["alexa_spoken_response"]
        for part in ("S&P 500 closed at", "market regime is", "10-year Treasury yields", "SPY's quant signal",
                     "QQQ, NVDA and SPY on the buy side", "not investment advice"):
            self.assertIn(part, speech)
        self.assertNotIn("/", speech)                # no "steep / normal" read aloud
        self.assertNotIn("_", speech)                # no "strong_buy" read aloud
        self.assertNotIn("🟢", speech)
        self.assertLess(len(speech.split()), 110)    # ~30-40 s spoken
        json.dumps(b)

    def test_optional_parts_are_skipped_not_invented(self):
        def no_spy(ticker, period, timeout=8):
            if ticker == "SPY":
                raise srv.MarketDataUnavailable("no SPY")
            return _fake_download(ticker, period, timeout)
        with mock.patch.object(srv, "_cached_download", side_effect=no_spy), \
                mock.patch.object(srv, "_SIGNALS_DIR", "/nonexistent/signals"):
            b = srv.execute_get_morning_brief()
        self.assertEqual(b["sections_skipped"], ["spy_signal", "scan_leaders"])
        self.assertIsNone(b["spy_signal"])
        self.assertNotIn("SPY's quant signal", b["alexa_spoken_response"])
        self.assertNotIn("daily scan", b["alexa_spoken_response"])

    def test_required_parts_raise(self):
        with mock.patch.object(srv, "_cached_download", side_effect=srv.MarketDataUnavailable("down")):
            with self.assertRaises(srv.MarketDataUnavailable):
                srv.execute_get_morning_brief()

    def test_voice_card_and_routine_prompt(self):
        res = AlexaMacroSkill.execute("Alexa, ask MacroPulse for my morning brief")
        self.assertEqual(res["tool_selected"], "get_morning_brief")
        self.assertEqual(res["display_card"]["title"], "MacroPulse Morning Brief")
        self.assertEqual(len(res["display_card"]["badges"]), 4)
        self.assertIn("Buy: QQQ, NVDA, SPY", json.dumps(res["apl_document"], ensure_ascii=False))

    def test_ask_webhook_intent_and_launch(self):
        body = {"request": {"type": "IntentRequest", "intent": {"name": "MorningBriefIntent", "slots": {}}}}
        out = AlexaMacroSkill.handle_ask_request(body)
        self.assertEqual(out["response"]["directives"][0]["token"], "macropulse-get_morning_brief")
        self.assertIn("market brief", out["response"]["outputSpeech"]["text"])
        launch = AlexaMacroSkill.handle_ask_request({"request": {"type": "LaunchRequest"}})
        self.assertIn("morning brief", launch["response"]["outputSpeech"]["text"])


class TestRouting(unittest.TestCase):
    def test_brief_prompts(self):
        for p in ("Alexa, ask MacroPulse for my morning brief", "Alexa, good morning", "catch me up on the markets",
                  "give me the pre-market brief", "Alexa, show the morning brief on the TV"):
            self.assertEqual(AlexaMacroSkill.identify_intent(p)[0], "get_morning_brief", p)

    def test_other_intents_unchanged(self):
        for p, tool in (("Alexa, give me an equity report on AAPL", "get_equity_report"),
                        ("Alexa, what is today's market regime?", "get_macro_regime"),
                        ("Alexa, check the 10-year Treasury yield and yield curve slope.", "get_rates_and_spreads"),
                        ("Alexa, show the NVDA Danger Zone on the TV", "check_nvda_danger_zone")):
            self.assertEqual(AlexaMacroSkill.identify_intent(p)[0], tool, p)

    def test_interaction_model_has_intent(self):
        m = json.load(open("skills/macropulse-alexa/interaction_model.json"))
        names = [i["name"] for i in m["interactionModel"]["languageModel"]["intents"]]
        self.assertIn("MorningBriefIntent", names)


if __name__ == "__main__":
    unittest.main()
