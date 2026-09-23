"""
tests/test_data_quality_fixes.py
─────────────────────────────────────────────────────────────────
Regression tests for issues found on the live app (AAPL walk-through), offline:
  * weekly/monthly bars were labelled with the period END (a future date for the current period)
  * NVDA peer returns like MU +1,434% from corrupted/mixed price series
  * Korean reason texts in the English Quant Signals tab
"""
import re
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src import macro_extras, quant_signals
from src.equity_report.engine import indicators as I

HANGUL = re.compile(r"[가-힣]")


def _daily(end="2026-09-23", n=300, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end, periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                         "volume": 1_000_000}, index=idx)


class TestBarLabels(unittest.TestCase):
    def test_current_period_is_labelled_with_last_trading_day(self):
        d = _daily()                                     # Wednesday 2026-09-23 is the last day
        for rule in ("W-FRI", "ME"):
            bars = I.resample(d, rule)
            self.assertEqual(bars.index[-1], pd.Timestamp("2026-09-23"), rule)
            self.assertLessEqual(bars.index.max(), d.index.max(), rule)
            self.assertTrue(set(bars.index) <= set(d.index), rule)   # every label is a real trading day

    def test_values_unchanged(self):
        d = _daily()
        m = I.resample(d, "ME")
        sept = d.loc["2026-09"]
        self.assertEqual(m["close"].iloc[-1], sept["close"].iloc[-1])
        self.assertEqual(m["high"].iloc[-1], sept["high"].max())
        self.assertEqual(m["volume"].iloc[-1], sept["volume"].sum())

    def test_snapshot_cross_dates_never_in_future(self):
        mtf = I.multi_timeframe(_daily(n=900))
        for tf, (_, snap) in mtf.items():
            self.assertLessEqual(snap["date"], "2026-09-23", tf)
            for k in ("macd", "kd"):
                cross = snap[k]["last_cross"]
                if cross:
                    self.assertLessEqual(cross["date"], "2026-09-23", tf)


class _FakeTicker:
    SERIES = {}

    def __init__(self, tkr):
        self.tkr = tkr

    def history(self, period, auto_adjust=True):
        return pd.DataFrame({"Close": self.SERIES[self.tkr]})


class TestPeerContext(unittest.TestCase):
    def setUp(self):
        idx = pd.bdate_range(end="2026-09-22", periods=250, tz="America/New_York")
        smooth = pd.Series(np.linspace(100, 150, 250), index=idx)
        spliced = smooth.copy()
        spliced.iloc[:100] = smooth.iloc[:100] / 15      # another ticker's much lower prices spliced in
        _FakeTicker.SERIES = {"NVDA": smooth, "SOXX": smooth, "AMD": smooth, "TSM": smooth,
                              "AVGO": smooth, "MU": spliced}

    def test_corrupted_series_is_excluded_not_shown(self):
        with mock.patch.object(macro_extras.yf, "Ticker", _FakeTicker):
            ctx = macro_extras.fetch_peer_context(365)
        self.assertNotIn("MU", ctx.columns)
        self.assertEqual(ctx.attrs["excluded"], ["MU"])
        self.assertAlmostEqual(ctx["NVDA"].iloc[0], 100.0)
        self.assertAlmostEqual(ctx["NVDA"].iloc[-1], 150.0)

    def test_guard(self):
        s = pd.Series([100, 101, 102, 103.0])
        self.assertTrue(macro_extras.clean_peer_series(s))
        self.assertFalse(macro_extras.clean_peer_series(pd.Series([10, 10.5, 160.0])))
        self.assertFalse(macro_extras.clean_peer_series(pd.Series([], dtype=float)))


class TestReasonLanguage(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        n = 260
        idx = pd.bdate_range(end="2026-09-22", periods=n)
        c = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.012, n)))
        raw = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c,
                            "Volume": rng.integers(1e6, 3e6, n)}, index=idx)
        self.df = quant_signals.add_indicators(raw)

    def test_english_by_default_korean_on_request(self):
        en = quant_signals.generate_signal("TST", self.df)
        ko = quant_signals.generate_signal("TST", self.df, lang="ko")
        self.assertEqual((en.score, en.signal), (ko.score, ko.signal))
        self.assertTrue(en.reasons)
        self.assertFalse(any(HANGUL.search(r) for r in en.reasons), en.reasons)
        self.assertTrue(all(HANGUL.search(r) for r in ko.reasons), ko.reasons)

    def test_every_reason_has_both_languages(self):
        for key, (en, ko) in quant_signals.REASONS.items():
            self.assertFalse(HANGUL.search(en), key)
            self.assertTrue(HANGUL.search(ko), key)

    def test_daily_korean_report_keeps_korean(self):
        src = open("scripts/daily_signal_alert.py", encoding="utf-8").read()
        self.assertIn('lang="ko"', src)


if __name__ == "__main__":
    unittest.main()
