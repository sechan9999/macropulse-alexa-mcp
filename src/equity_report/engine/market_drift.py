"""Market drift for the target touch probabilities: the S&P 500 12-month expected return from the
same Ridge regression as the Expected Returns tab (tab 4), turned into a stock drift with CAPM.

  E[R_m]  = expm1( Ridge(alpha=5) on standardised macro features -> forward 12m log return )
  E[R_s]  = r_f + beta x (E[R_m] - r_f)
  nu      = ln(1 + E[R_s]) - sigma^2 / 2            (annual log drift used in the touch probability)

The Ridge is solved in NumPy (standardise X, unpenalised intercept), which is numerically the same as
sklearn's Pipeline(StandardScaler, Ridge) used in tab 4, so the slim MCP image needs no scikit-learn.
Unlike tab 4's chart (whose last point needs a realised 12m target), the model is fitted on every month
with a known 12m outcome and applied to the *latest* feature row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ("dgs10", "credit_spread", "yc_slope", "realized_vol_12m", "momentum_12_1", "regime_score",
            "realized_vol_3m")
ALPHA = 5.0
MIN_TRAIN = 60                 # months with a realised 12m outcome
CLIP = (-0.20, 0.25)           # guard against a runaway extrapolation on an unusual feature row


def ridge_fit_predict(X: np.ndarray, y: np.ndarray, x_new: np.ndarray, alpha: float = ALPHA) -> float:
    """Equivalent of Pipeline([StandardScaler(), Ridge(alpha)]).fit(X, y).predict(x_new)."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z, z_new = (X - mu) / sd, (x_new - mu) / sd
    y_mean = y.mean()
    coef = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - y_mean))
    return float(z_new @ coef + y_mean)


def add_features(macro: pd.DataFrame) -> pd.DataFrame:
    """Fill in the tab-4 return features from a monthly S&P 500 level column when they are missing."""
    d = macro.copy()
    if "sp500_ret_m" not in d and "sp500" in d:
        d["sp500_ret_m"] = np.log(d["sp500"]).diff()
    r = d["sp500_ret_m"]
    if "realized_vol_12m" not in d:
        d["realized_vol_12m"] = r.rolling(12).std() * np.sqrt(12)
    if "realized_vol_3m" not in d:
        d["realized_vol_3m"] = r.rolling(3).std() * np.sqrt(12)
    if "momentum_12_1" not in d:
        d["momentum_12_1"] = r.rolling(11).sum().shift(1)
    return d


def expected_market_return(macro: pd.DataFrame | None, source: str = "MacroPulse Ridge (Expected Returns tab)") -> dict | None:
    """Latest S&P 500 12m expected simple return, or None when the history is too short / unusable."""
    if macro is None or len(macro) == 0 or ("sp500" not in macro and "sp500_ret_m" not in macro):
        return None
    if "_is_demo" in macro and bool(macro["_is_demo"].iloc[-1]):
        return None
    d = add_features(macro)
    cols = [c for c in FEATURES if c in d.columns and d[c].notna().any()]
    if not cols:
        return None
    d["_y"] = d["sp500_ret_m"].rolling(12).sum().shift(-12)
    feats = d[cols].replace([np.inf, -np.inf], np.nan)
    train = d[feats.notna().all(axis=1) & d["_y"].notna()]
    live = feats.dropna()
    if len(train) < MIN_TRAIN or live.empty:
        return None
    pred = ridge_fit_predict(train[cols].to_numpy(float), train["_y"].to_numpy(float), live.iloc[-1].to_numpy(float))
    raw = float(np.expm1(pred))
    er = float(np.clip(raw, *CLIP))
    return {"expected_return": er, "raw_expected_return": raw, "clipped": er != raw,
            "as_of": str(pd.Timestamp(live.index[-1]).date()), "n_train": int(len(train)), "features": cols,
            "source": source}


def stock_drift(market: dict | None, risk_free: float, beta: float, sigma_annual: float) -> dict:
    """CAPM stock expected return and the annual log drift for the touch probability (0 without a market view)."""
    if not market:
        return {"source": "none (driftless)", "market_expected_return": None, "stock_expected_return": None,
                "log_drift": 0.0}
    er_s = risk_free + beta * (market["expected_return"] - risk_free)
    return {"source": market["source"], "as_of": market["as_of"], "n_train": market["n_train"],
            "market_expected_return": market["expected_return"], "market_clipped": market["clipped"],
            "stock_expected_return": er_s, "log_drift": float(np.log1p(er_s) - sigma_annual ** 2 / 2),
            "method": "CAPM: r_f + beta x (E[R_m] - r_f); log drift = ln(1 + E[R_s]) - sigma^2/2"}
