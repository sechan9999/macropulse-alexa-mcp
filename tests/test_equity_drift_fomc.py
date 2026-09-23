"""
tests/test_equity_drift_fomc.py
─────────────────────────────────────────────────────────────────
Equity report follow-ups, offline:
  * Ridge S&P 500 view (tab 4 model) -> CAPM drift -> touch probabilities with drift
  * tab-13 FOMC shock scenarios overlaid on the WACC x g sensitivity grid
  * the engine stays importable without scikit-learn / matplotlib (slim MCP image)
"""
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

from src.equity_report.engine import analysis, dcf, market_drift, synthetic
from src.equity_report.render import dashboard


def _macro(n=240, seed=4, demo=False):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-31", periods=n, freq="ME")
    ret = rng.normal(0.007, 0.04, n)
    df = pd.DataFrame({"sp500": 1000 * np.exp(np.cumsum(ret)), "dgs10": 3 + np.cumsum(rng.normal(0, 0.1, n)),
                       "credit_spread": 0.02 + np.abs(rng.normal(0, 0.005, n))}, index=idx)
    df["yc_slope"] = df["dgs10"] * 0.003
    if demo:
        df["_is_demo"] = True
    return df


class TestRidgeView(unittest.TestCase):
    def test_numpy_ridge_matches_sklearn_pipeline(self):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        rng = np.random.default_rng(0)
        X, y, x_new = rng.normal(size=(120, 5)) * [1, 5, 0.1, 3, 2], rng.normal(size=120), rng.normal(size=5)
        ref = Pipeline([("sc", StandardScaler()), ("r", Ridge(alpha=5.0))]).fit(X, y).predict(x_new[None])[0]
        self.assertAlmostEqual(market_drift.ridge_fit_predict(X, y, x_new, 5.0), ref, places=10)

    def test_expected_return_uses_latest_row(self):
        m = market_drift.expected_market_return(_macro())
        self.assertIsNotNone(m)
        self.assertEqual(m["as_of"], "2024-12-31")                     # latest month, not 12 months back
        self.assertGreaterEqual(m["n_train"], market_drift.MIN_TRAIN)
        self.assertTrue(market_drift.CLIP[0] <= m["expected_return"] <= market_drift.CLIP[1])
        self.assertIn("momentum_12_1", m["features"])

    def test_short_or_demo_history_gives_none(self):
        self.assertIsNone(market_drift.expected_market_return(_macro(n=50)))
        self.assertIsNone(market_drift.expected_market_return(_macro(demo=True)))
        self.assertIsNone(market_drift.expected_market_return(None))

    def test_capm_drift(self):
        d = market_drift.stock_drift({"expected_return": 0.08, "source": "x", "as_of": "d", "n_train": 99,
                                      "clipped": False}, risk_free=0.04, beta=1.5, sigma_annual=0.3)
        self.assertAlmostEqual(d["stock_expected_return"], 0.04 + 1.5 * 0.04)
        self.assertAlmostEqual(d["log_drift"], np.log1p(0.10) - 0.045)
        self.assertEqual(market_drift.stock_drift(None, 0.04, 1.0, 0.3)["log_drift"], 0.0)


class TestTouchProbability(unittest.TestCase):
    def test_zero_drift_is_reflection_principle(self):
        from scipy.stats import norm
        for k in (80, 120):
            z = abs(np.log(k / 100)) / (0.3 * np.sqrt(0.5))
            self.assertAlmostEqual(dcf.touch_probability(100, k, 0.3, 0.5, 0.0), 2 * (1 - norm.cdf(z)), places=12)

    def test_drift_direction(self):
        up0, up = dcf.touch_probability(100, 120, 0.3, 0.5), dcf.touch_probability(100, 120, 0.3, 0.5, 0.15)
        dn0, dn = dcf.touch_probability(100, 85, 0.3, 0.5), dcf.touch_probability(100, 85, 0.3, 0.5, 0.15)
        self.assertGreater(up, up0)
        self.assertLess(dn, dn0)
        self.assertLess(dcf.touch_probability(100, 120, 0.3, 0.5, -0.15), up0)

    def test_matches_monte_carlo(self):
        rng = np.random.default_rng(1)
        n, steps, sig, nu, T = 10000, 500, 0.3, 0.12, 0.5
        dt = T / steps
        path = np.cumsum(rng.normal(nu * dt, sig * np.sqrt(dt), (n, steps)), axis=1)
        # Broadie-Glasserman: a discretely monitored barrier ~ continuous barrier shifted by 0.5826 sigma sqrt(dt)
        mc = (path.max(axis=1) >= np.log(1.15) - 0.5826 * sig * np.sqrt(dt)).mean()
        self.assertAlmostEqual(dcf.touch_probability(100, 115, sig, T, nu), mc, delta=0.02)

    def test_report_levels_carry_both_versions(self):
        rep = analysis.analyze(synthetic.stock_bundle(), market=market_drift.expected_market_return(_macro()))
        lv = rep.ctx["rule_based"]["levels"]
        self.assertIn("Ridge", rep.ctx["drift"]["source"])
        self.assertNotEqual(lv["touch_prob_6m"]["target1"], lv["touch_prob_6m_driftless"]["target1"])
        text = analysis.rule_based_text(rep.ctx)
        self.assertIn("Driftless for comparison", text)
        s = analysis.compact_summary(rep)
        self.assertIsNotNone(s["drift"]["market_expected_return"])

    def test_without_market_view_stays_driftless(self):
        rep = analysis.analyze(synthetic.stock_bundle())
        lv = rep.ctx["rule_based"]["levels"]
        self.assertEqual(lv["touch_prob_6m"], lv["touch_prob_6m_driftless"])
        self.assertIn("Driftless log-normal", analysis.rule_based_text(rep.ctx))


class TestFomcOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b = synthetic.stock_bundle()
        b["is_synthetic"] = False                # let the tab-13 lookup run for a ticker tab 13 lists
        b["ticker"] = "AAPL"
        cls.val = dcf.build(b)
        cls.ov = {x["scenario"]: x for x in cls.val["fomc_overlay"]}

    def test_four_tab13_scenarios(self):
        self.assertEqual(set(self.ov), set(dcf.FOMC_RATE_SHOCKS))

    def test_direction(self):
        w0 = self.val["assumptions"]["wacc"]
        self.assertGreater(self.ov["hawkish_50bps"]["wacc"], w0)
        self.assertLess(self.ov["dovish_50bps"]["wacc"], w0)
        self.assertGreater(self.ov["dovish_50bps"]["per_share"], self.val["base"]["per_share"])
        self.assertLess(self.ov["stagflation_inversion"]["per_share"], self.ov["hawkish_50bps"]["per_share"])
        self.assertAlmostEqual(self.ov["stagflation_inversion"]["terminal_growth"],
                               self.val["assumptions"]["terminal_growth"] - 0.005)

    def test_wacc_formula(self):
        a, x = self.val["assumptions"], self.ov["hawkish_50bps"]
        ke = a["risk_free_used"] + 0.005 + a["beta_used"] * (a["erp_used"] + 0.0025)
        kd = (a["cost_of_debt_used"] + 0.005 + 0.0025) * (1 - a["tax_rate"])
        self.assertAlmostEqual(x["wacc"], a["weight_equity"] * ke + (1 - a["weight_equity"]) * kd)

    def test_value_on_grid_matches_sensitivity(self):
        x = self.ov["dovish_50bps"]
        inp = dcf.DCFInputs(**self.val["inputs"])
        ref = dcf.sensitivity(inp, self.val["base"]["fcf"], [x["wacc"]], [x["terminal_growth"]]).iloc[0, 0]
        self.assertAlmostEqual(x["per_share"], ref)

    def test_tab13_price_shock_reference(self):
        self.assertAlmostEqual(self.ov["hawkish_50bps"]["tab13_price_shock"], -0.038)   # FOMC_SENSITIVITIES AAPL
        self.assertAlmostEqual(self.ov["liquidity_cascade"]["tab13_price_shock"], -0.090)

    def test_grid_badges(self):
        cells = dashboard.fomc_cells(self.val["sensitivity"], self.val["fomc_overlay"])
        tags = sorted(t for v in cells.values() for t in v)
        self.assertIn("H", tags)
        self.assertIn("D", tags)
        for (i, j), v in cells.items():
            for t in v:
                x = next(o for o in self.val["fomc_overlay"] if dashboard.FOMC_TAGS[o["scenario"]] == t)
                self.assertLessEqual(abs(self.val["sensitivity"].index[i] - x["wacc"]), 0.0025 + 1e-9)


class TestSlimImport(unittest.TestCase):
    def test_engine_imports_without_sklearn_or_matplotlib(self):
        code = ("import sys; import src.equity_report.engine.pipeline, src.equity_report.engine.market_drift; "
                "bad = [m for m in ('sklearn', 'matplotlib', 'openpyxl', 'docx') if m in sys.modules]; "
                "assert not bad, bad")
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, timeout=120)


if __name__ == "__main__":
    unittest.main()
