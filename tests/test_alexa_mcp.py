"""
tests/test_alexa_mcp.py
─────────────────────────────────────────────────────────────────
Unit and integration tests for MacroPulse Alexa+ MCP server,
Agent Skills, and tools.
"""
import unittest
from src.alexa_mcp_server import (
    execute_get_macro_regime,
    execute_get_rates_and_spreads,
    execute_simulate_portfolio_risk,
    execute_check_nvda_danger_zone,
    execute_scan_quant_signals,
    execute_get_expected_returns,
)
from src.alexa_agent_skill import AlexaMacroSkill


class TestAlexaMacroMCP(unittest.TestCase):

    def test_macro_regime_tool(self):
        res = execute_get_macro_regime()
        self.assertEqual(res["status"], "success")
        self.assertIn("regime", res)
        self.assertIn("stress_score", res)
        self.assertTrue(len(res["alexa_spoken_response"]) > 20)

    def test_rates_and_spreads_tool(self):
        res = execute_get_rates_and_spreads()
        self.assertEqual(res["status"], "success")
        self.assertIn("treasury_10y", res)
        self.assertIn("curve_slope_bps", res)
        self.assertTrue(len(res["alexa_spoken_response"]) > 20)

    def test_simulate_portfolio_risk(self):
        res = execute_simulate_portfolio_risk(ticker="SPY", confidence=95, days=30, n_paths=1000)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["ticker"], "SPY")
        self.assertIn("var_pct", res)
        self.assertIn("cvar_pct", res)
        self.assertTrue(res["var_pct"] <= 0)  # VaR return should be negative or zero

    def test_nvda_danger_zone(self):
        res = execute_check_nvda_danger_zone()
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["ticker"], "NVDA")
        self.assertTrue(0.0 <= res["danger_index"] <= 1.0)
        self.assertIn(res["danger_level"], ["🔴 DANGER", "🟡 CAUTION", "🟢 SAFE"])

    def test_scan_quant_signals(self):
        res = execute_scan_quant_signals("SPY")
        self.assertEqual(res["status"], "success")
        self.assertIn("signal", res)
        self.assertTrue(-100 <= res["conviction_score"] <= 100)

    def test_expected_returns(self):
        res = execute_get_expected_returns()
        self.assertEqual(res["status"], "success")
        self.assertIn("expected_return_pct", res)
        self.assertIn("lower_1sigma_pct", res)

    def test_agent_skill_intent_resolution(self):
        tool, args = AlexaMacroSkill.identify_intent("Alexa, what is the market regime today?")
        self.assertEqual(tool, "get_macro_regime")

        tool, args = AlexaMacroSkill.identify_intent("Alexa, run a 99% risk simulation on QQQ")
        self.assertEqual(tool, "simulate_portfolio_risk")
        self.assertEqual(args.get("ticker"), "QQQ")
        self.assertEqual(args.get("confidence"), 99)

        tool, args = AlexaMacroSkill.identify_intent("Is Nvidia in the danger zone right now?")
        self.assertEqual(tool, "check_nvda_danger_zone")

    def test_agent_skill_execution(self):
        output = AlexaMacroSkill.execute("Alexa, check today's macro regime")
        self.assertEqual(output["status"], "success")
        self.assertEqual(output["tool_selected"], "get_macro_regime")
        self.assertIn("display_card", output)
        self.assertIn("spoken_response", output)
        self.assertIn("apl_document", output)
        self.assertTrue(output["latency_ms"] >= 0)

    def test_tv_voice_routing(self):
        # "Alexa, show the NVDA Danger Zone on the TV"
        output = AlexaMacroSkill.execute("Alexa, show the NVDA Danger Zone on the TV")
        self.assertEqual(output["status"], "success")
        self.assertEqual(output["tool_selected"], "check_nvda_danger_zone")
        self.assertTrue(output["is_tv_request"])
        self.assertIn("Displaying", output["spoken_response"])
        self.assertEqual(output["apl_document"]["type"], "APL")

    def test_fomc_shock_voice_intent(self):
        # "Alexa, run an FOMC rate shock test"
        tool, args = AlexaMacroSkill.identify_intent("Alexa, run an FOMC rate shock test on my portfolio")
        self.assertEqual(tool, "simulate_fomc_shock")
        output = AlexaMacroSkill.execute("Alexa, run an FOMC rate shock test")
        self.assertEqual(output["status"], "success")
        self.assertIn("total_pnl_pct", output["raw_payload"])

    def test_alexa_ask_webhook_lifecycle(self):
        # LaunchRequest
        launch_req = {"request": {"type": "LaunchRequest"}}
        resp = AlexaMacroSkill.handle_ask_request(launch_req)
        self.assertEqual(resp["version"], "1.0")
        self.assertIn("outputSpeech", resp["response"])

        # IntentRequest: MacroRegimeIntent
        regime_req = {
            "request": {
                "type": "IntentRequest",
                "intent": {"name": "MacroRegimeIntent"}
            }
        }
        resp = AlexaMacroSkill.handle_ask_request(regime_req)
        self.assertEqual(resp["version"], "1.0")
        self.assertIn("directives", resp["response"])
        self.assertEqual(resp["response"]["directives"][0]["type"], "Alexa.Presentation.APL.RenderDocument")

        # IntentRequest: ShowOnTvIntent
        tv_req = {
            "request": {
                "type": "IntentRequest",
                "intent": {
                    "name": "ShowOnTvIntent",
                    "slots": {
                        "TargetView": {"value": "NVDA Danger Zone"}
                    }
                }
            }
        }
        resp = AlexaMacroSkill.handle_ask_request(tv_req)
        self.assertEqual(resp["version"], "1.0")
        self.assertIn("Nvidia Danger Zone", resp["response"]["card"]["title"])


if __name__ == "__main__":
    unittest.main()

