"""
tests/test_equity_engine.py
─────────────────────────────────────────────────────────────────
Offline tests for src/equity_report/engine: indicators, candle patterns, DCF math,
the SEC EDGAR companyfacts parser, the regime overlay and the end-to-end pipeline.
No network: prices and SEC responses are injected.
"""
import unittest

import numpy as np
import pandas as pd

from src.equity_report.config import DCFAssumptions
from src.equity_report.engine import analysis, data_us, dcf, indicators as I, patterns as P, pipeline, synthetic


def _ohlc(rows, start="2024-01-01"):
    d = pd.bdate_range(start, periods=len(rows))
    return pd.DataFrame(rows, index=d, columns=["open", "high", "low", "close"]).assign(volume=1000)


def _downtrend(n=12, p=100.0):
    rows = []
    for _ in range(n):
        rows.append([p, p + 0.5, p - 2.5, p - 2])
        p -= 2
    return rows, p


class TestIndicators(unittest.TestCase):
    def test_macd_matches_manual_ema(self):
        s = pd.Series(np.linspace(100, 130, 60))
        e12, e26 = s.ewm(span=12, adjust=False).mean(), s.ewm(span=26, adjust=False).mean()
        self.assertAlmostEqual(I.macd(s)["macd"].iloc[-1], (e12 - e26).iloc[-1])

    def test_stochastic_bounds(self):
        kd = I.stochastic(synthetic.stock_bundle()["prices"]).dropna()
        self.assertTrue(kd["k"].between(0, 100).all() and kd["d"].between(0, 100).all())

    def test_rsi_extremes(self):
        self.assertGreater(I.rsi(pd.Series(np.arange(1, 60, dtype=float))).iloc[-1], 95)
        self.assertLess(I.rsi(pd.Series(np.arange(60, 1, -1, dtype=float))).iloc[-1], 5)

    def test_resample_weekly(self):
        df = synthetic.stock_bundle()["prices"]
        w = I.resample(df, "W-FRI")
        wk = df.loc[w.index[5] - pd.Timedelta(days=4): w.index[5]]
        self.assertEqual(w["high"].iloc[5], wk["high"].max())
        self.assertEqual(w["close"].iloc[5], wk["close"].iloc[-1])

    def test_cmf_bounds(self):
        cmf = I.chaikin_money_flow(synthetic.stock_bundle()["prices"]).dropna()
        self.assertTrue(cmf.between(-1, 1).all())

    def test_support_resistance_brackets_price(self):
        df = synthetic.stock_bundle()["prices"].iloc[-126:]
        sr, c = I.support_resistance(df), df["close"].iloc[-1]
        self.assertTrue(all(x["price"] < c for x in sr["support"]))
        self.assertTrue(all(x["price"] > c for x in sr["resistance"]))

    def test_multi_timeframe(self):
        mtf = I.multi_timeframe(synthetic.stock_bundle()["prices"])
        self.assertEqual(set(mtf), {"Daily", "Weekly", "Monthly"})
        self.assertIn(mtf["Weekly"][1]["ma_alignment"], ("bullish stack", "bearish stack", "mixed"))


class TestPatterns(unittest.TestCase):
    def test_hammer_after_downtrend(self):
        rows, p = _downtrend()
        rows.append([p, p + 0.1, p - 6, p + 0.3])
        pats = P.detect(_ohlc(rows))
        self.assertIn("hammer", pats[pats.date == pats.date.max()]["pattern"].tolist())

    def test_bullish_engulfing(self):
        rows, p = _downtrend()
        rows += [[p, p + 0.2, p - 1.2, p - 1], [p - 1.3, p + 1.5, p - 1.4, p + 1.2]]
        self.assertIn("bullish_engulfing", P.detect(_ohlc(rows))["pattern"].tolist())

    def test_screen_keys(self):
        s = P.screen(I.add_all(synthetic.stock_bundle()["prices"]))
        for k in ("bullish_count", "bearish_count", "bull_score", "bear_score", "overall"):
            self.assertIn(k, s)


class TestDCF(unittest.TestCase):
    def test_closed_form(self):
        v = dcf.value(dcf.DCFInputs(100, 0.1, 0.0, 0.0, 0.1, 5, 0, 1, 100))
        self.assertAlmostEqual(v["ev"], 100.0)            # FCF 10 forever at 10%

    def test_sensitivity_center_equals_base(self):
        inp = dcf.DCFInputs(1000, 0.12, 0.08, 0.02, 0.09, 5, 50, 10, 100)
        base = dcf.value(inp)
        self.assertAlmostEqual(dcf.sensitivity(inp, base["fcf"], [0.09], [0.02]).iloc[0, 0], base["per_share"])

    def test_wacc_must_exceed_g(self):
        with self.assertRaises(ValueError):
            dcf.value(dcf.DCFInputs(100, 0.1, 0.0, 0.05, 0.05, 5, 0, 1, 100))

    def test_touch_probability(self):
        self.assertAlmostEqual(dcf.touch_probability(100, 100, 0.3, 1), 1.0)
        self.assertLess(dcf.touch_probability(100, 200, 0.3, 0.5), dcf.touch_probability(100, 110, 0.3, 0.5))

    def test_uses_live_risk_free_and_interest_based_kd(self):
        b = synthetic.stock_bundle()
        v = dcf.build(b)
        a = v["assumptions"]
        self.assertEqual(a["risk_free_source"], "live 10Y Treasury")
        self.assertAlmostEqual(a["risk_free_used"], b["risk_free_live"])
        fin = b["financials"].iloc[-1]
        self.assertAlmostEqual(a["cost_of_debt_used"], np.clip(fin["interest_expense"] / fin["debt"], .03, .10))

    def test_user_override_beats_live(self):
        v = dcf.build(synthetic.stock_bundle(), DCFAssumptions(risk_free=0.05))
        self.assertEqual(v["assumptions"]["risk_free_used"], 0.05)


class TestEdgarParser(unittest.TestCase):
    facts = synthetic.company_facts_fixture()

    def test_annual_table(self):
        t = data_us.annual_table(self.facts)
        self.assertEqual(list(t.index), ["2022", "2023", "2024", "2025"])      # Sept FY ends
        self.assertEqual(t.loc["2022", "revenue"], 100.0)                        # older tag still used
        self.assertEqual(t.loc["2024", "revenue"], 121.0)                        # restatement wins
        self.assertEqual(t.loc["2025", "revenue"], 130.0)                        # quarterly rows ignored
        self.assertEqual(t.loc["2025", "debt"], 57.0)                            # 50 + 5 + CP 2
        self.assertEqual(t.loc["2025", "total_liabilities"], 180.0)              # derived: 300 - 120
        self.assertEqual(t.loc["2025", "fy_end"], "2025-09-27")

    def test_shares_sum_share_classes(self):
        self.assertEqual(data_us.shares_outstanding(self.facts), 1000.0)

    def test_eps_by_filing_date_skips_comparatives(self):
        eps = data_us.eps_by_filing_date(self.facts)
        self.assertEqual(list(eps.values), [2.70, 3.00])
        self.assertEqual(str(eps.index[-1].date()), "2025-11-03")

    def test_latest_annual_filing_ignores_newer_10q(self):
        import copy
        facts = copy.deepcopy(self.facts)
        facts["facts"]["us-gaap"]["RevenueFromContractWithCustomerExcludingAssessedTax"]["units"]["USD"].append(
            {"start": "2025-09-28", "end": "2025-12-27", "val": 40.0, "filed": "2026-01-30", "form": "10-Q", "fp": "Q1"})
        self.assertEqual(data_us.latest_annual_filing(facts), "2025-11-03")

    def test_ifrs_filer_is_unsupported(self):
        with self.assertRaises(data_us.UnsupportedFiler):
            data_us.annual_table({"facts": {"ifrs-full": {}}})

    def test_load_bundle_with_injected_sources(self):
        def http_get(url):
            if url.endswith("company_tickers.json"):
                return {"0": {"cik_str": 1234567, "ticker": "FIX", "title": "FIXTURE INC"}}
            return self.facts
        data_us._mem.clear()
        b = data_us.load_bundle("FIX", price_fetch=lambda t: synthetic.price_frame(seed=3 if t == "FIX" else 5),
                                http_get=http_get, with_positioning=False)
        self.assertEqual(b["cik"], "0001234567")
        self.assertEqual(b["shares"], 1000.0)
        self.assertTrue(b["pe_history"].dropna().size > 0)
        self.assertIn("latest 10-K filed 2025-11-03", b["sources"][1])

    def test_unknown_ticker(self):
        data_us._mem.clear()
        with self.assertRaises(data_us.UnsupportedFiler):
            data_us.cik_for("SPY", lambda url: {"0": {"cik_str": 1, "ticker": "AAPL", "title": "Apple"}})


class TestPipeline(unittest.TestCase):
    def test_demo_report_and_summary(self):
        rep = pipeline.run("DEMO")
        s = analysis.compact_summary(rep)
        self.assertEqual(s["status"], "success")
        self.assertIn(s["rating"], analysis.RATINGS)
        self.assertTrue(s["synthetic_data"])
        self.assertIn("synthetic demo data", s["alexa_spoken_response"].lower())
        self.assertIn("## 6. Verdict", rep.narrative)
        self.assertEqual(rep.narrative_method, "rules")

    def test_summary_is_json_safe(self):
        import json
        json.dumps(analysis.compact_summary(pipeline.run("DEMO")))
        json.dumps(pipeline.run("DEMO").ctx)


if __name__ == "__main__":
    unittest.main()
