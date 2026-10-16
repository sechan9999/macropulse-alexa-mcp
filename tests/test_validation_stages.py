"""
tests/test_validation_stages.py
─────────────────────────────────────────────────────────────────
The validation stages measure the product, not a copy of it: the rebuilt macro frame equals
macro_data.load_macro on the same prices, the point-in-time bundle equals data_us.load_bundle when
nothing is cut, the trial ledger has 32 rules, and the window/vintage helpers line up dates as
pre-registered. Offline: every input is synthetic or injected.
"""
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src.equity_report.engine import data_us, synthetic
from validation import data, frames, stage1, stage2, stage4, stage5, stage6


def _daily(seed, start=100.0, n=5200, vol=0.01, begin="2004-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(begin, periods=n)
    return pd.Series(start * np.exp(np.cumsum(rng.normal(0.0002, vol, n))), index=idx)


PRICES = {"sp500": _daily(1, 1000), "vix": _daily(2, 20, vol=0.03), "dgs10": _daily(3, 4, vol=0.005),
          "tbill_3m": _daily(4, 3, vol=0.005)}


def _fred():
    idx = pd.date_range("2003-01-01", "2026-12-01", freq="MS")
    rng = np.random.default_rng(9)
    return (pd.Series(1 + np.abs(np.cumsum(rng.normal(0, 0.05, len(idx)))), index=idx),
            pd.Series(np.cumsum(rng.normal(0, 0.1, len(idx))), index=idx))


class TestFrameMatchesProduct(unittest.TestCase):
    def test_build_macro_equals_load_macro(self):
        from src import macro_data
        tick = {"^GSPC": "sp500", "^VIX": "vix", "^TNX": "dgs10", "^IRX": "tbill_3m"}

        def fake_download(tkr, **kw):
            if tkr not in tick:
                return pd.DataFrame()
            s = PRICES[tick[tkr]]
            return pd.DataFrame({"Close": s[s.index >= kw.get("start", "1900-01-01")]})
        spread, slope = _fred()
        with mock.patch.object(macro_data.yf, "download", side_effect=fake_download), \
                mock.patch.object(macro_data, "_try_load_fred_series", return_value=(spread, slope)), \
                mock.patch.object(macro_data, "_BQ_OK", False):
            product = macro_data.load_macro.__wrapped__() if hasattr(macro_data.load_macro, "__wrapped__") \
                else macro_data.load_macro()
        ours = frames.build_macro("2005-01-01", prices=PRICES, spread=spread, slope=slope)
        cols = ["sp500", "sp500_ret_m", "realized_vol_12m", "realized_vol_3m", "momentum_12_1", "dgs10", "tbill_3m",
                "credit_spread", "yc_slope", "regime_score", "regime"]
        product = product[product.index >= "2005-01-01"]
        pd.testing.assert_frame_equal(ours[cols], product[cols].loc[ours.index], check_freq=False, check_names=False)


class TestTrialLedger(unittest.TestCase):
    def test_32_rules_11_long_only_and_the_default_is_one(self):
        rules = stage4.distinct_rules()
        self.assertEqual(len(rules), 32)
        self.assertEqual(sum(not r["allow_short"] for r in rules), 11)
        self.assertIn(stage4.default_name(), [stage4.rule_name(r) for r in rules])
        # thresholds 0.34-0.66 on three signals are one rule ("at least two of three")
        self.assertEqual(stage4.rule_name({**stage4.DEFAULT, "threshold": 0.4}), stage4.default_name())


class TestPointInTime(unittest.TestCase):
    def test_facts_asof_keeps_only_filed_rows(self):
        f = synthetic.company_facts_fixture()
        cut = stage6.facts_asof(f, pd.Timestamp("2024-12-31"))
        for concepts in cut["facts"].values():
            for body in concepts.values():
                for rows in body["units"].values():
                    self.assertTrue(all(r["filed"] <= "2024-12-31" for r in rows))
        self.assertEqual(list(data_us.annual_table(cut).index), ["2022", "2023", "2024"])

    def test_uncut_bundle_equals_load_bundle(self):
        facts = synthetic.company_facts_fixture()
        px, ix = synthetic.price_frame(seed=3), synthetic.price_frame(seed=5)
        splits = pd.Series([10.0], index=[pd.Timestamp("2025-06-01")])

        def http_get(url):
            if url.endswith("company_tickers.json"):
                return {"0": {"cik_str": 1234567, "ticker": "FIX", "title": "FIXTURE INC"}}
            return facts
        data_us._mem.clear()
        ref = data_us.load_bundle("FIX", price_fetch=lambda t: px if t == "FIX" else ix, http_get=http_get,
                                  with_positioning=False, splits_fetch=lambda t: splits, risk_free=0.04)
        far = pd.Timestamp("2100-01-01")
        b = stage6.bundle_asof("FIX", ref["name"], ref["cik"], facts, data_us.normalize_ohlcv(px),
                               data_us.normalize_ohlcv(ix)[["close"]], splits, far, 0.04)
        pd.testing.assert_frame_equal(b["financials"], ref["financials"])
        for k in ("shares", "last_price", "eps_latest", "dilution_factor", "shares_diluted", "eps_ttm"):
            self.assertAlmostEqual(b[k], ref[k], msg=k)
        self.assertEqual(b["ttm"], ref["ttm"])

    def test_shares_move_to_todays_split_basis(self):
        facts = synthetic.company_facts_fixture()
        px = data_us.normalize_ohlcv(synthetic.price_frame(seed=3))
        splits = pd.Series([4.0], index=[pd.Timestamp("2099-06-01")])
        b = stage6.bundle_asof("FIX", "F", "1", facts, px, px[["close"]], splits, pd.Timestamp("2098-12-31"), 0.04)
        self.assertEqual(b["shares"], 4000.0)                      # 1,000 shares filed, a 4-for-1 split after


class TestAlfredMonthEnd(unittest.TestCase):
    def test_value_known_at_month_end_and_first_release(self):
        a = pd.DataFrame({
            "realtime_start": pd.to_datetime(["2020-02-03", "2020-03-02", "2020-03-02", "2020-04-01"]),
            "realtime_end": pd.to_datetime(["2020-03-01", "2262-04-11", "2262-04-11", "2262-04-11"]),
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-02-01", "2020-03-01"]),
            "value": [1.0, 1.1, 2.0, 3.0]})
        with mock.patch.object(data, "alfred", return_value=a):
            k = stage1.known_at_month_end("X", with_obs=True)
            fr = stage1.first_release("X")
        self.assertEqual(k.loc["2020-02-01", "value"], 1.0)        # February's end: only January's first print
        self.assertEqual(k.loc["2020-03-01", "value"], 2.0)        # March's end: February, January revised
        self.assertEqual(fr.loc["2020-01-01", "first_value"], 1.0)
        self.assertEqual(fr.loc["2020-01-01", "latest_value"], 1.1)
        self.assertEqual(fr.loc["2020-02-01", "lag_days"], 2)      # published Mar 2, two days after Feb 29


class TestWindows(unittest.TestCase):
    def test_next_month_outcome_starts_after_the_month_end(self):
        c = pd.Series(np.exp(np.arange(80) * 0.01), index=pd.bdate_range("2024-01-01", periods=80))
        o = frames.next_month_outcomes(c, pd.DatetimeIndex([pd.Timestamp("2024-01-01")]))
        self.assertAlmostEqual(o.loc["2024-01-01", "next_ret"], 21 * 0.01)

    def test_prevailing_mean_uses_past_months_only(self):
        r = pd.Series([0.01, 0.03, -0.02], index=pd.date_range("2020-01-01", periods=3, freq="MS"))
        b = stage1.prevailing_mean(r)
        self.assertAlmostEqual(b.iloc[1], 12 * 0.02)

    def test_var_simulation_shapes_and_seed(self):
        w = np.random.default_rng(0).normal(0, 0.01, 252)
        a = stage5.simulate(w, "gaussian", np.random.default_rng(42))
        b = stage5.simulate(w, "gaussian", np.random.default_rng(42))
        self.assertEqual(a.shape, (stage5.PATHS,))
        np.testing.assert_array_equal(a, b)
        boot = stage5.simulate(np.array([0.01, -0.01]), "bootstrap", np.random.default_rng(1))
        sums = np.round(np.log1p(boot) / 0.01)                    # each path is a sum of 21 draws of +-1%
        self.assertTrue(np.all(np.abs(sums) <= 21) and np.all(sums % 2 == 1))


class TestGoyalWelchMechanics(unittest.TestCase):
    def test_a_real_predictor_beats_the_mean_and_noise_does_not(self):
        rng = np.random.default_rng(0)
        yrs = range(1900, 2006)
        x = pd.Series(rng.normal(size=len(yrs)), index=yrs)
        prem = 0.5 * x.shift(1) + rng.normal(scale=0.5, size=len(yrs))
        df = pd.DataFrame({"premium": prem, "good": x, "noise": rng.normal(size=len(yrs))}, index=yrs)
        self.assertGreater(stage2.gw_oos_r2(df, "good", 1920, 2005, 1900), 20)
        self.assertLess(stage2.gw_oos_r2(df, "noise", 1920, 2005, 1900), 1)

    def test_pesaran_timmermann(self):
        a = np.r_[np.ones(60), -np.ones(60)]
        self.assertLess(stage2.pesaran_timmermann(a, a)["p_one_sided"], 1e-6)
        rng = np.random.default_rng(1)
        self.assertGreater(stage2.pesaran_timmermann(rng.normal(size=400), rng.normal(size=400))["p_one_sided"], 0.01)


if __name__ == "__main__":
    unittest.main()
