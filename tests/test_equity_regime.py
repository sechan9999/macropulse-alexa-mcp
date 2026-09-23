"""
tests/test_equity_regime.py
─────────────────────────────────────────────────────────────────
Step 4 — the MacroPulse regime drives the equity valuation:
Risk-Off shifts probability to the bear case and raises the ERP; Risk-On does the reverse.
"""
import unittest

from src.equity_report.engine import dcf, regime, synthetic
from src.equity_report.engine.analysis import analyze, compact_summary


class TestRegimeOverlay(unittest.TestCase):
    def test_normalize_dashboard_labels(self):
        self.assertEqual(regime.normalize("Risk-Off 🔴"), "Risk-Off")
        self.assertEqual(regime.normalize("risk on"), "Risk-On")
        self.assertEqual(regime.normalize("Neutral 🟡"), "Neutral")
        self.assertEqual(regime.normalize(None), "Neutral")

    def test_weights_sum_to_one(self):
        for reg in regime.OVERLAY:
            self.assertAlmostEqual(sum(regime.overlay(reg).probs.values()), 1.0)

    def test_missing_regime_is_labelled(self):
        self.assertIn("no regime", regime.overlay(None).source)

    def test_expected_value_ordering(self):
        b = synthetic.stock_bundle()
        vals = {r: dcf.build(b, regime=regime.overlay(r)) for r in ("Risk-On", "Neutral", "Risk-Off")}
        ev = {r: v["expected_per_share"] for r, v in vals.items()}
        self.assertGreater(ev["Risk-On"], ev["Neutral"])
        self.assertGreater(ev["Neutral"], ev["Risk-Off"])
        wacc = {r: v["assumptions"]["wacc"] for r, v in vals.items()}
        self.assertLess(wacc["Risk-On"], wacc["Neutral"])
        self.assertLess(wacc["Neutral"], wacc["Risk-Off"])
        self.assertAlmostEqual(vals["Risk-Off"]["assumptions"]["erp_used"],
                               vals["Risk-Off"]["assumptions"]["erp_base"] + 0.005)

    def test_neutral_reference_is_reported(self):
        v = dcf.build(synthetic.stock_bundle(), regime=regime.overlay("Risk-Off"))
        self.assertNotAlmostEqual(v["neutral_weight_expected"], v["expected_per_share"])

    def test_overlay_flows_into_summary_and_narrative(self):
        rep = analyze(synthetic.stock_bundle(), regime_label="Risk-Off 🔴")
        s = compact_summary(rep)
        self.assertEqual(s["macro_regime"]["regime"], "Risk-Off")
        self.assertEqual(s["scenarios"]["Bear"]["prob"], 0.40)
        self.assertIn("Risk-Off", s["alexa_spoken_response"])
        self.assertIn("valuation", s["basis"])


if __name__ == "__main__":
    unittest.main()
