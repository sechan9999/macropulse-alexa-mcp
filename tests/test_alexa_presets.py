"""
tests/test_alexa_presets.py
─────────────────────────────────────────────────────────────────
Alexa+ Copilot tab and the voice skill, from two reports on 2026-10-16:
  * clicking a preset (e.g. Morning Brief) did not change "Spoken Voice Input" once the box had been
    used, and the old text was dispatched;
  * "equity report on Johnson & Johnson" returned an Apple report, "J and J" the ticker J (Jacobs).
Offline: no market data, no Streamlit server.
"""
import unittest
from unittest import mock

from src import alexa_agent_skill as skill
from src import alexa_prompt_ui as ui
from src.alexa_agent_skill import AlexaMacroSkill


def _controls_script():
    import streamlit as st
    from src.alexa_prompt_ui import prompt_controls
    prompt, fired = prompt_controls(st)
    if fired:
        st.write("DISPATCH: " + prompt)


class TestPresetButtons(unittest.TestCase):
    def setUp(self):
        from streamlit.testing.v1 import AppTest
        self.at = AppTest.from_function(_controls_script, default_timeout=60)
        self.at.run()

    def box(self):
        return self.at.text_input(key=ui.PROMPT_KEY).value

    def click(self, i):
        self.at.button(key=f"alexa_preset_{i}").click()
        self.at.run()

    def test_starts_with_the_default_prompt(self):
        self.assertEqual(self.box(), ui.DEFAULT_PROMPT)
        self.assertEqual(len(self.at.markdown), 0)

    def test_every_preset_replaces_the_box_and_is_dispatched(self):
        self.at.text_input(key=ui.PROMPT_KEY).set_value("Alexa, give me an equity report on AAPL")
        self.at.run()
        for i, (_, text) in enumerate(ui.PRESETS):
            self.click(i)
            self.assertFalse(self.at.exception)
            self.assertEqual(self.box(), text)
            self.assertEqual(self.at.markdown[-1].value, "DISPATCH: " + text)

    def test_typing_does_not_dispatch_until_asked(self):
        self.click(6)
        self.at.text_input(key=ui.PROMPT_KEY).set_value("Alexa, report on JNJ")
        self.at.run()
        self.assertEqual(self.box(), "Alexa, report on JNJ")
        self.assertEqual(len(self.at.markdown), 0)

    def test_app_uses_the_shared_controls(self):
        src = open("app.py", encoding="utf-8").read()
        self.assertIn("prompt_controls(st)", src)
        self.assertNotIn('value=default_text, key="alexa_prompt_input"', src)


class TestSpokenCompanyNames(unittest.TestCase):
    CASES = {
        "Alexa, ask MacroPulse to give me an equity report on Johnson & Johnson": "JNJ",
        "give me an equity report on johnson and johnson": "JNJ",
        "give me a report on J&J": "JNJ",
        "give me a report on J. and J.": "JNJ",
        "give me a report on j n j": "JNJ",
        "give me a report on jnj": "JNJ",
        "what is the fair value of Procter & Gamble": "PG",
        "run a DCF on ExxonMobil": "XOM",
        "run a DCF on x o m": "XOM",
        "give me an equity report on AAPL": "AAPL",
        "Alexa, run a DCF on brk.b": "BRK-B",
    }

    def test_spoken_names(self):
        for prompt, ticker in self.CASES.items():
            tool, args = AlexaMacroSkill.identify_intent(prompt)
            self.assertEqual((tool, args.get("ticker")), ("get_equity_report", ticker), prompt)

    def test_a_request_without_a_company_keeps_the_default(self):
        self.assertEqual(AlexaMacroSkill.identify_intent("Alexa, give me an equity report")[1]["ticker"], "AAPL")

    def test_an_unknown_name_is_asked_about_not_replaced(self):
        with mock.patch.object(skill, "execute_get_equity_report", side_effect=AssertionError("must not run")):
            res = AlexaMacroSkill.execute("Alexa, give me an equity report on General Motors")
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["raw_payload"]["error"], "unknown_company")
        self.assertIn("General Motors", res["spoken_response"])
        self.assertIn("won't guess", res["spoken_response"])

    def test_ask_slot_with_an_ampersand_reaches_jnj(self):
        seen = []

        def fake(ticker):
            seen.append(ticker)
            raise skill.EquityReportUnsupported(f"{ticker}: test stop")
        req = {"version": "1.0", "session": {"new": True, "sessionId": "s", "user": {"userId": "u"}},
               "request": {"type": "IntentRequest", "requestId": "r", "locale": "en-US",
                           "intent": {"name": "EquityReportIntent",
                                      "slots": {"Ticker": {"name": "Ticker", "value": "johnson & johnson"}}}}}
        with mock.patch.object(skill, "execute_get_equity_report", side_effect=fake):
            AlexaMacroSkill.handle_ask_request(req)
        self.assertEqual(seen, ["JNJ"])


class TestUnexpectedErrors(unittest.TestCase):
    def test_a_bug_is_answered_out_loud(self):
        with mock.patch.object(skill, "execute_get_equity_report", side_effect=TypeError("unhashable type")):
            res = AlexaMacroSkill.execute("Alexa, give me an equity report on JNJ")
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["raw_payload"]["error"], "internal_error")
        self.assertIn("won't guess", res["spoken_response"])
        self.assertTrue(res["apl_document"])


if __name__ == "__main__":
    unittest.main()
