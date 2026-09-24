"""
tests/test_leakage.py
─────────────────────────────────────────────────────────────────
Noise leakage check for the expected-return model. Returns are pure noise and the features are
persistent AR(1) noise, so nothing can truly forecast them: a point-in-time model must show no
out-of-sample skill. A model that trains on 12m targets that are not yet realised shows a clear
positive correlation (the leaky reference below, which is the model before the fix).

Calibration (6 seeds x 24 runs x 240 months, 3 features): point-in-time -0.14 … +0.04,
leaky reference +0.20 … +0.32. The 0.10 threshold sits between the two ranges.
"""
import unittest

import numpy as np
import pandas as pd

from src.macro_model import _ridge_pipeline, oos_noise_correlation, walk_forward_ridge

THRESHOLD = 0.10
RUNS, MONTHS = 24, 240


def _leaky_reference(d, cols, min_train=48):
    """The pre-fix loop: trains on every earlier row, including the 12 whose targets overlap the future."""
    feats = d.dropna(subset=cols)
    preds = pd.Series(np.nan, index=feats.index)
    model = _ridge_pipeline(5.0)
    for i in range(min_train, len(feats)):
        train = feats.iloc[:i].dropna(subset=["y_fwd"])
        model.fit(train[cols], train["y_fwd"])
        preds.iloc[i] = model.predict(feats[cols].iloc[i: i + 1])[0]
    return preds


class TestNoiseLeakage(unittest.TestCase):
    def test_model_has_no_skill_on_noise(self):
        corr = oos_noise_correlation(lambda d, c: walk_forward_ridge(d, c, "y_fwd"),
                                     n_runs=RUNS, n_months=MONTHS)
        self.assertLess(corr, THRESHOLD, f"out-of-sample correlation on noise = {corr:+.3f}: look-ahead leak")

    def test_check_detects_a_leaky_model(self):
        corr = oos_noise_correlation(_leaky_reference, n_runs=RUNS, n_months=MONTHS)
        self.assertGreater(corr, THRESHOLD, f"leaky reference only reached {corr:+.3f}; the check lost its power")


if __name__ == "__main__":
    unittest.main()
