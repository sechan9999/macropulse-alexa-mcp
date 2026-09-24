"""
tests/test_offline_pipeline.py
─────────────────────────────────────────────────────────────────
Offline research pipeline (run_backtest.py): FRED macro values are dated by their release, and the
expanding GMM's regime columns keep the same meaning across refits. Offline; no downloads.
"""
import itertools
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src import regimes
from src.data_fred import to_monthly_point_in_time, to_release_dates


class TestReleaseDates(unittest.TestCase):
    def test_monthly_value_is_used_only_after_its_release(self):
        cpi = pd.Series([300.0, 301.0, 302.0], index=pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]))
        self.assertEqual(to_release_dates(cpi, "cpi").index[-1], pd.Timestamp("2024-04-15"))
        m = to_monthly_point_in_time({"cpi": cpi})
        self.assertEqual(m.loc["2024-03-31", "cpi"], 301.0)       # March CPI is not out on March 31
        self.assertEqual(m.loc["2024-04-30", "cpi"], 302.0)

    def test_daily_and_weekly_series(self):
        dgs10 = pd.Series([4.2, 4.3], index=pd.to_datetime(["2024-03-27", "2024-03-28"]))
        nfci = pd.Series([-0.5, -0.4], index=pd.to_datetime(["2024-03-22", "2024-03-29"]))
        m = to_monthly_point_in_time({"dgs10": dgs10, "nfci": nfci})
        self.assertEqual(m.loc["2024-03-31", "dgs10"], 4.3)        # daily yields: known by month-end
        self.assertEqual(m.loc["2024-03-31", "nfci"], -0.5)       # week of Mar 29 is published Apr 3
        self.assertEqual(m.loc["2024-04-30", "nfci"], -0.4)

    def test_every_release_lag_is_at_least_the_reference_period(self):
        """No monthly series may appear in the month-end row of the month it describes."""
        idx = pd.date_range("2020-01-01", periods=24, freq="MS")
        for col in ("cpi", "unrate", "indpro", "baa", "aaa", "tb3ms"):
            s = pd.Series(np.arange(24.0), index=idx)
            m = to_monthly_point_in_time({col: s})
            months_since_start = (m.index.year - 2020) * 12 + m.index.month - 1
            np.testing.assert_array_equal(m[col].values, months_since_start - 1.0, err_msg=col)


def _three_regimes(n=90, seed=0):
    """Features with three well-separated clusters; cluster c has credit spread ~ c."""
    rng = np.random.default_rng(seed)
    true = rng.integers(0, 3, n)
    true[:9] = [0, 1, 2] * 3                                       # every cluster present early
    df = pd.DataFrame({"credit_spread": true + rng.normal(0, 0.08, n),
                       "yc_slope": -true + rng.normal(0, 0.08, n)},
                      index=pd.date_range("2000-01-31", periods=n, freq="ME"))
    return df, pd.Series(true, index=df.index)


class _PermutingGMM:
    """Stand-in for GaussianMixture that numbers its components differently on every fit."""
    perms = itertools.cycle(itertools.permutations(range(3)))

    def __init__(self, n_components, **kw):
        self.perm = np.array(next(self.perms))

    def fit(self, Z):
        order = np.argsort(Z[:, 0])
        thirds = np.array_split(order, 3)
        centers = np.array([Z[i].mean(axis=0) for i in thirds])    # calm, mid, stressed
        self.means_ = centers[self.perm]
        return self

    def predict_proba(self, z):
        d = ((z[:, None, :] - self.means_[None, :, :]) ** 2).sum(axis=2)
        p = np.zeros_like(d)
        p[np.arange(len(z)), d.argmin(axis=1)] = 1.0
        return p


class TestGmmLabels(unittest.TestCase):
    def test_labels_do_not_switch_when_the_fit_renumbers_components(self):
        df, true = _three_regimes()
        with mock.patch.object(regimes, "GaussianMixture", _PermutingGMM):
            out = regimes.fit_gmm_regimes(df, ["credit_spread", "yc_slope"], n_regimes=3, min_train=30)
        got = out["regime"].dropna().astype(int)
        self.assertEqual(got.nunique(), 3)
        pd.testing.assert_series_equal(got, true.loc[got.index], check_names=False)

    def test_real_gmm_orders_regimes_by_credit_spread(self):
        df, true = _three_regimes()
        out = regimes.fit_gmm_regimes(df, ["yc_slope", "credit_spread"], n_regimes=3, min_train=30)
        got = out["regime"].dropna().astype(int)
        self.assertGreater((got == true.loc[got.index]).mean(), 0.95)   # pR2 = highest-spread regime
        self.assertTrue(np.allclose(out[["pR0", "pR1", "pR2"]].dropna().sum(axis=1), 1.0))

    def test_regimes_ignore_later_data(self):
        df, _ = _three_regimes()
        cut = df.index[60]
        shocked = df.copy()
        shocked.loc[shocked.index > cut, "credit_spread"] += 5.0
        cols = ["credit_spread", "yc_slope"]
        a = regimes.fit_gmm_regimes(df, cols, n_regimes=3, min_train=30)
        b = regimes.fit_gmm_regimes(shocked, cols, n_regimes=3, min_train=30)
        pd.testing.assert_frame_equal(a.loc[:cut], b.loc[:cut])


if __name__ == "__main__":
    unittest.main()
