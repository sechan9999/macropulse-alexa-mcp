"""
tests/test_validation_stats.py
─────────────────────────────────────────────────────────────────
validation/stats.py against independent calculations, published tables and limiting cases.
Offline; no market data.
"""
import unittest

import numpy as np
import pandas as pd
from scipy import stats as sps

from validation import stats as st


class TestForecastEvaluation(unittest.TestCase):
    def test_oos_r2(self):
        y = np.array([1.0, 2.0, 3.0])
        self.assertEqual(st.oos_r2(y, y, np.zeros(3)), 1.0)
        self.assertEqual(st.oos_r2(y, np.zeros(3), np.zeros(3)), 0.0)
        self.assertAlmostEqual(st.oos_r2(y, y + 1, y + 2), 1 - 3 / 12)

    def test_clark_west_is_zero_for_identical_forecasts_and_positive_for_a_good_one(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=600)
        y = 0.5 * x + rng.normal(size=600)
        same = st.clark_west(y, np.zeros(600), np.zeros(600))
        self.assertEqual(same["mean_f"], 0.0)
        good = st.clark_west(y, 0.5 * x, np.zeros(600), lags=12)
        self.assertGreater(good["stat"], 3)
        self.assertLess(good["p_one_sided"], 0.01)

    def test_hac_reduces_to_the_ordinary_t_without_lags(self):
        x = np.random.default_rng(1).normal(0.2, 1, 400)
        r = st.hac_mean_test(x, lags=0)
        self.assertAlmostEqual(r["se"], x.std(ddof=0) / np.sqrt(len(x)), places=10)


class TestVarBacktests(unittest.TestCase):
    def test_kupiec_matches_the_likelihoods(self):
        n, x, p = 250, 7, 0.01
        ll0 = sps.binom.logpmf(x, n, p)
        ll1 = sps.binom.logpmf(x, n, x / n)
        self.assertAlmostEqual(st.kupiec_pof(n, x, p)["lr"], -2 * (ll0 - ll1))
        self.assertGreater(st.kupiec_pof(250, 2, 0.01)["p_value"], 0.5)
        self.assertLess(st.kupiec_pof(250, 12, 0.01)["p_value"], 0.001)
        self.assertTrue(np.isfinite(st.kupiec_pof(100, 0, 0.05)["lr"]))

    def test_basel_traffic_light_table(self):
        zones = [st.basel_zone(250, x)["zone"] for x in range(12)]
        self.assertEqual(zones, ["green"] * 5 + ["yellow"] * 5 + ["red"] * 2)

    def test_christoffersen_flags_clustered_exceedances(self):
        spread = np.zeros(400, bool)
        spread[::20] = True
        clustered = np.zeros(400, bool)
        clustered[100:120] = True
        self.assertGreater(st.christoffersen(spread, 0.05)["p_ind"], 0.05)
        self.assertLess(st.christoffersen(clustered, 0.05)["p_ind"], 0.001)
        c = st.christoffersen(spread, 0.05)
        self.assertEqual(c["n00"] + c["n01"] + c["n10"] + c["n11"], 399)

    def test_acerbi_szekely_is_near_zero_when_es_is_right(self):
        rng = np.random.default_rng(2)
        alpha, n = 0.05, 200000
        L = rng.normal(size=n)
        var = sps.norm.ppf(1 - alpha)
        es = sps.norm.pdf(var) / alpha
        self.assertAlmostEqual(st.acerbi_szekely_z2(L, np.full(n, var), np.full(n, es), alpha), 0.0, places=1)
        self.assertLess(st.acerbi_szekely_z2(2 * L, np.full(n, var), np.full(n, es), alpha), -0.5)


class TestOverfitting(unittest.TestCase):
    def test_psr_closed_form_for_normal_returns(self):
        sr, t = 0.1, 240
        self.assertAlmostEqual(st.probabilistic_sharpe(sr, 0.0, t, 0.0, 3.0),
                               sps.norm.cdf(sr * np.sqrt(t - 1) / np.sqrt(1 + sr ** 2 / 2)))

    def test_one_trial_deflates_to_the_psr(self):
        r = np.random.default_rng(3).normal(0.01, 0.04, 300)
        d = st.deflated_sharpe(r, [r.mean() / r.std(ddof=1)])
        self.assertEqual(d["sr0"], 0.0)
        self.assertAlmostEqual(d["dsr"], d["psr_vs_zero"])

    def test_expected_max_sharpe_matches_simulation(self):
        rng = np.random.default_rng(4)
        sim = rng.normal(size=(20000, 100)).max(axis=1).mean()
        self.assertAlmostEqual(st.expected_max_sharpe(1.0, 100), sim, delta=0.05)

    def test_more_trials_deflate_more(self):
        r = np.random.default_rng(5).normal(0.01, 0.04, 300)
        srs = list(np.random.default_rng(6).normal(0.1, 0.05, 32))
        self.assertLess(st.deflated_sharpe(r, srs)["dsr"], st.deflated_sharpe(r, srs[:2])["dsr"])

    def test_pbo_is_about_one_half_for_noise_and_low_for_a_real_edge(self):
        rng = np.random.default_rng(7)
        # Each split's chance of λ <= 0 is exactly 1/2 for i.i.d. noise; one sample's PBO is noisy because
        # all splits reuse the same data, so average over samples.
        pbos = [st.pbo_cscv(pd.DataFrame(rng.normal(0, 0.04, (160, 10))), 8)["pbo"] for _ in range(40)]
        self.assertAlmostEqual(np.mean(pbos), 0.5, delta=0.1)
        noise = pd.DataFrame(rng.normal(0, 0.04, (320, 10)))
        edge = noise.copy()
        edge[0] = edge[0] + 0.03
        self.assertLess(st.pbo_cscv(edge, 8)["pbo"], 0.05)


class TestHolm(unittest.TestCase):
    def test_holm(self):
        adj = st.holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adj["a"], 0.03)
        self.assertAlmostEqual(adj["c"], 0.06)
        self.assertAlmostEqual(adj["b"], 0.06)                   # monotone: never below the previous step
        self.assertEqual(st.holm({"a": None, "b": 0.2}), {"b": 0.2})


if __name__ == "__main__":
    unittest.main()
