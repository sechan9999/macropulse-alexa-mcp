"""
tests/test_portfolio.py
─────────────────────────────────────────────────────────────────
src/portfolio.py: tear-sheet metrics in excess of the T-bill rate, the backtest's cash leg,
and the Monte Carlo bootstrap mode. Offline; no market data.
"""
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src import portfolio as pf


def _idx(n):
    return pd.date_range("2010-01-01", periods=n, freq="MS")


def _macro(n=120, tbill=4.0, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(index=_idx(n))
    d["sp500_ret_m"] = rng.normal(0.007, 0.045, n)
    d["sp500"] = 1000 * np.exp(d["sp500_ret_m"].cumsum())
    d["momentum_12_1"] = d["sp500_ret_m"].rolling(11).sum().shift(1)
    d["regime"] = "Neutral 🟡"
    if tbill is not None:
        d[pf.TBILL_COLUMN] = tbill
    return d


class TestMetrics(unittest.TestCase):
    def test_sortino_uses_downside_deviation_over_all_months(self):
        r = pd.Series([0.03, -0.01, 0.02, -0.02, 0.01, 0.04] * 4, index=_idx(24))
        m = pf.compute_hf_metrics(r)
        dd = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(12)
        self.assertAlmostEqual(m["sortino"], r.mean() * 12 / dd)
        # the old formula (std of the negative months only) gives a different number
        self.assertNotAlmostEqual(m["sortino"], r.mean() * 12 / (r[r < 0].std() * np.sqrt(12)))

    def test_one_negative_month_still_has_a_sortino(self):
        r = pd.Series([0.02] * 23 + [-0.05], index=_idx(24))
        self.assertTrue(np.isfinite(pf.compute_hf_metrics(r)["sortino"]))  # std of 1 value is NaN

    def test_sharpe_and_sortino_subtract_the_risk_free_rate(self):
        r = pd.Series(np.random.default_rng(1).normal(0.008, 0.04, 60), index=_idx(60))
        rf = pd.Series(0.004, index=r.index)
        m0, m = pf.compute_hf_metrics(r), pf.compute_hf_metrics(r, rf=rf)
        ex = r - 0.004
        self.assertAlmostEqual(m["sharpe"], ex.mean() / ex.std() * np.sqrt(12))
        self.assertLess(m["sharpe"], m0["sharpe"])
        self.assertLess(m["sortino"], m0["sortino"])
        self.assertAlmostEqual(m["ann_ret"], m0["ann_ret"])           # the return itself is unchanged
        self.assertAlmostEqual(m["rf_ann"], 0.048)

    def test_cash_earning_the_risk_free_rate_has_no_excess_return(self):
        rf = pd.Series(np.linspace(0.001, 0.004, 36), index=_idx(36))
        m = pf.compute_hf_metrics(rf.copy(), rf=rf)
        self.assertAlmostEqual(m["ann_excess"], 0.0)
        self.assertTrue(np.isnan(m["sortino"]))                        # no downside at all

    def test_jensen_alpha_on_excess_returns(self):
        rng = np.random.default_rng(2)
        b = pd.Series(rng.normal(0.007, 0.04, 80), index=_idx(80))
        rf = pd.Series(0.003, index=b.index)
        p = rf + 0.002 + 0.5 * (b - rf)                               # beta 0.5, alpha 0.2%/month
        m = pf.compute_hf_metrics(p, b, rf=rf)
        self.assertAlmostEqual(m["beta"], 0.5)
        self.assertAlmostEqual(m["alpha"], 0.024)


class TestRiskFree(unittest.TestCase):
    def test_rate_earned_in_a_month_is_the_one_quoted_before_it(self):
        d = _macro(24)
        d[pf.TBILL_COLUMN] = np.arange(24, dtype=float)
        rf = pf.rf_monthly(d)
        self.assertTrue(np.isnan(rf.iloc[0]))
        self.assertAlmostEqual(rf.iloc[5], np.log1p(0.04) / 12)       # quoted at the end of month 4

    def test_missing_series_is_reported_not_invented(self):
        self.assertIsNone(pf.rf_monthly(_macro(tbill=None)))
        d = _macro()
        d[pf.TBILL_COLUMN] = np.nan
        self.assertIsNone(pf.rf_monthly(d))


class TestBacktestCashLeg(unittest.TestCase):
    KW = dict(use_regime=False, use_momentum=True, use_trend=False, threshold=0.5, cost_bps=0.0,
              allow_short=False)

    def test_out_of_market_months_earn_the_tbill(self):
        d = _macro()
        d["momentum_12_1"] = -1.0                                      # always out of the market
        bt = pf.run_strategy_backtest(d, **self.KW)
        expect = np.log1p(0.04) / 12
        self.assertTrue(np.allclose(bt["strat_ret"].iloc[1:], expect))
        self.assertEqual(bt.attrs["cash_rate"], "3-month T-bill (^IRX)")

    def test_in_market_months_earn_spy_only(self):
        d = _macro()
        d["momentum_12_1"] = 1.0
        bt = pf.run_strategy_backtest(d, **self.KW)
        pd.testing.assert_series_equal(bt["strat_ret"].iloc[1:], bt["sp500_ret_m"].iloc[1:], check_names=False)

    def test_without_tbill_data_cash_earns_zero_and_says_so(self):
        d = _macro(tbill=None)
        d["momentum_12_1"] = -1.0
        bt = pf.run_strategy_backtest(d, **self.KW)
        self.assertTrue((bt["strat_ret"] == 0).all())
        self.assertTrue(bt.attrs["cash_rate"].startswith("0%"))

    def test_cash_return_ignores_later_rates(self):
        d = _macro()
        d[pf.TBILL_COLUMN] = np.linspace(1, 5, len(d))
        cut = d.index[60]
        shocked = d.copy()
        shocked.loc[shocked.index >= cut, pf.TBILL_COLUMN] = 20.0      # rates quoted from `cut` on
        a = pf.run_strategy_backtest(d, **self.KW)["strat_ret"]
        b = pf.run_strategy_backtest(shocked, **self.KW)["strat_ret"]
        pd.testing.assert_series_equal(a.loc[:cut], b.loc[:cut])       # month `cut` uses the prior quote


class TestMonteCarlo(unittest.TestCase):
    def test_gaussian_is_unchanged(self):
        rng = np.random.default_rng(99)
        expect = np.exp(np.cumsum(rng.normal(0.08 / 12, 0.16 / np.sqrt(12), (500, 12)), axis=1))
        np.testing.assert_allclose(pf.run_monte_carlo(0.08, 0.16, 500), expect)

    def test_bootstrap_only_draws_historical_months(self):
        hist = pd.Series(np.random.default_rng(4).standard_t(3, 120) * 0.03, index=_idx(120))
        paths = pf.run_monte_carlo(0.5, 0.01, 400, 12, method=pf.MC_BOOTSTRAP, hist_rets=hist)
        monthly = np.diff(np.log(np.column_stack([np.ones(400), paths])), axis=1)
        self.assertEqual(paths.shape, (400, 12))
        self.assertTrue(np.isin(np.round(monthly, 12), np.round(hist.values, 12)).all())

    def test_blocks_keep_consecutive_months_together(self):
        hist = pd.Series(np.arange(1, 61) * 1e-3, index=_idx(60))       # each month is its own label
        paths = pf.run_monte_carlo(0, 0, 50, 12, method=pf.MC_BOOTSTRAP, hist_rets=hist, block=3)
        monthly = np.diff(np.log(np.column_stack([np.ones(50), paths])), axis=1)
        steps = np.round(np.diff(monthly, axis=1) * 1e3, 6)[:, [0, 1, 3, 4, 6, 7, 9, 10]]
        self.assertTrue((steps == 1).all())                             # months within a block are consecutive

    def test_bootstrap_has_the_fat_left_tail_the_normal_misses(self):
        rng = np.random.default_rng(5)
        hist = pd.Series(np.where(rng.uniform(size=240) < 0.04, -0.20, rng.normal(0.012, 0.03, 240)),
                         index=pd.date_range("2000-01-01", periods=240, freq="MS"))
        mu, vol = hist.mean() * 12, hist.std() * np.sqrt(12)
        g = pf.run_monte_carlo(mu, vol, 20000, 1)[:, 0]
        b = pf.run_monte_carlo(mu, vol, 20000, 1, method=pf.MC_BOOTSTRAP, hist_rets=hist)[:, 0]
        self.assertLess(np.percentile(b, 1), np.percentile(g, 1))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            pf.run_monte_carlo(0, 0, 10, method=pf.MC_BOOTSTRAP, hist_rets=pd.Series([0.01] * 5))
        with self.assertRaises(ValueError):
            pf.run_monte_carlo(0, 0, 10, method="student-t")


class TestPerformanceLines(unittest.TestCase):
    def test_both_lines_start_at_100_on_the_first_month_of_the_range(self):
        full = pd.Series(1000 * np.exp(np.linspace(0, 1.5, 200)), index=pd.date_range("2005-01-01", periods=200, freq="MS"))
        window = full.loc["2015-01-01":]                              # the dashboard's Start date
        spy = np.log(window * 1.001 ** np.arange(len(window))).diff().dropna()   # + a small dividend drift
        sp5, spy_idx, dd = pf.performance_lines(window, spy)
        self.assertEqual(sp5.iloc[0], 100.0)
        self.assertEqual(spy_idx.index[0], window.index[0])
        self.assertEqual(spy_idx.iloc[0], 100.0)
        self.assertAlmostEqual(sp5.iloc[-1], window.iloc[-1] / window.iloc[0] * 100)
        self.assertGreater(spy_idx.iloc[-1], sp5.iloc[-1])            # total return beats price return

    def test_drawdown_is_measured_inside_the_range(self):
        px = pd.Series([200.0, 100.0, 110.0, 90.0, 120.0], index=_idx(5))
        _, _, dd = pf.performance_lines(px.iloc[1:])                   # the 200 peak is before Start
        self.assertAlmostEqual(dd.min(), 90 / 110 - 1)
        self.assertEqual(dd.iloc[0], 0.0)


class TestMcpSummaries(unittest.TestCase):
    def test_bootstrap_summary_uses_history(self):
        from src import macro_extras
        d = _macro(200)
        d["_is_demo"] = False
        with mock.patch.object(macro_extras, "load_macro", return_value=d):
            r = macro_extras.summarize_monte_carlo(n_paths=1000, method="bootstrap", block_months=2)
        self.assertEqual(r["method"], "bootstrap")
        self.assertEqual(r["block_months"], 2)
        self.assertNotIn("mu_pct", r)
        self.assertIn("error", macro_extras.summarize_monte_carlo(method="nope"))

    def test_backtest_summary_reports_sortino_and_cash_rate(self):
        from src import macro_extras
        r = macro_extras.summarize_backtest(_macro(), pf.compute_hf_metrics, use_regime=False)
        self.assertEqual(r["cash_rate"], "3-month T-bill (^IRX)")
        self.assertIn("sortino", r["strategy"])


if __name__ == "__main__":
    unittest.main()
