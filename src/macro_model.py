"""
src/macro_model.py
─────────────────────────────────────────────────────────────────
Point-in-time macro regime and expected-return model shared by the Streamlit app (app.py) and the
Streamlit-free pipeline (src/macro_data.py). Everything here only uses information available at each
date, so a backtest or chart built on it has no look-ahead:

  * regime score    = expanding z-score of the credit spread + expanding z-score of 12m realised vol
                      (each date is standardised with the mean/std of data up to that date only)
  * credit / slope  = FRED BAA-AAA and T10Y2Y. The keyless fredgraph.csv endpoint is used when no API
                      key is set. If FRED is unreachable the columns are left empty and the regime is
                      reported as unavailable; nothing is synthesised from the 10Y yield.
  * expected return = expanding-window Ridge on the 12m forward log return, refit each month on rows
                      whose 12m outcome is already known at that month, and applied to the latest row.
"""
from __future__ import annotations

import io
import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

RISK_ON, NEUTRAL, RISK_OFF = "Risk-On 🟢", "Neutral 🟡", "Risk-Off 🔴"
UNAVAILABLE = "Unavailable ⚪"
REGIME_THRESHOLD = 0.5
Z_MIN_PERIODS = 36          # months of history before a date's z-score is defined
HORIZON = 12                # forecast horizon (months) of the expected-return model
RIDGE_FEATURES = ("dgs10", "credit_spread", "yc_slope", "realized_vol_12m", "momentum_12_1",
                  "regime_score", "realized_vol_3m")
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}"
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_TIMEOUT = 15.0
logger = logging.getLogger(__name__)


# ── regime ────────────────────────────────────────────────────────────
def expanding_z(s: pd.Series, min_periods: int = Z_MIN_PERIODS) -> pd.Series:
    """z-score of each value against the mean/std of the series up to and including that date."""
    mu = s.expanding(min_periods).mean()
    sd = s.expanding(min_periods).std()
    return (s - mu) / sd.where(sd > 0)


def add_regime(df: pd.DataFrame, min_periods: int = Z_MIN_PERIODS) -> pd.DataFrame:
    """Point-in-time regime_score / regime columns.

    Rows without a defined score (history warm-up, or no real credit-spread data) are labelled
    UNAVAILABLE with a NaN score rather than being guessed. A credit spread or curve slope that was
    produced by the old 10Y-yield proxy (``_credit_source`` / ``_slope_source`` == "proxy", e.g. in a
    stored data-lake mart) is discarded first.
    """
    out = df.copy()
    for col, src in (("credit_spread", "_credit_source"), ("yc_slope", "_slope_source")):
        if src in out.columns and col in out.columns:
            out.loc[out[src].astype(str) == "proxy", col] = np.nan
            out.loc[out[src].astype(str) == "proxy", src] = "unavailable"
    if "credit_spread" not in out or "realized_vol_12m" not in out:
        out["regime_score"] = np.nan
    else:
        out["regime_score"] = (expanding_z(out["credit_spread"].astype(float), min_periods)
                               + expanding_z(out["realized_vol_12m"].astype(float), min_periods))
    score = out["regime_score"]
    out["regime"] = np.select([score < -REGIME_THRESHOLD, score > REGIME_THRESHOLD, score.notna()],
                              [RISK_ON, RISK_OFF, NEUTRAL], default=UNAVAILABLE)
    return out


# ── FRED ──────────────────────────────────────────────────────────────
def _fred_api(series_id: str, start: pd.Timestamp, api_key: str, timeout: float = FRED_TIMEOUT) -> pd.Series:
    """One series from the official FRED API (needs a free API key; works from cloud hosts, where the
    keyless CSV endpoint can hang)."""
    query = urllib.parse.urlencode({"series_id": series_id, "api_key": api_key, "file_type": "json",
                                    "observation_start": pd.Timestamp(start).strftime("%Y-%m-%d")})
    req = urllib.request.Request(f"{FRED_API_URL}?{query}", headers={"User-Agent": "MacroPulse/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obs = json.loads(resp.read().decode("utf-8"))["observations"]
    s = pd.to_numeric(pd.Series([o["value"] for o in obs]), errors="coerce")   # '.' = missing
    s.index = pd.to_datetime([o["date"] for o in obs])
    return s.dropna()


def _fred_csv(series_id: str, start: pd.Timestamp, timeout: float = FRED_TIMEOUT) -> pd.Series:
    """One series from FRED's public fredgraph.csv endpoint (no API key needed)."""
    url = FRED_CSV_URL.format(sid=series_id, start=pd.Timestamp(start).strftime("%Y-%m-%d"))
    req = urllib.request.Request(url, headers={"User-Agent": "MacroPulse/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = pd.read_csv(io.BytesIO(resp.read()))
    s = pd.to_numeric(raw.iloc[:, 1], errors="coerce")       # FRED writes '.' for missing values
    s.index = pd.to_datetime(raw.iloc[:, 0])
    return s.dropna()


def fred_series(series_id: str, start, end=None, api_key: Optional[str] = None) -> pd.Series:
    """A FRED series via the official API when a key is available (FRED_API_KEY), otherwise via the
    keyless CSV endpoint. Failures are logged by type and message only; the key is never logged."""
    key = api_key if api_key is not None else os.environ.get("FRED_API_KEY")
    s = None
    if key:
        try:
            s = _fred_api(series_id, pd.Timestamp(start), key)
        except Exception as e:
            logger.warning("FRED API request for %s failed (%s: %s); trying the keyless CSV endpoint",
                           series_id, type(e).__name__, str(e).replace(key, "***"))
    if s is None:
        s = _fred_csv(series_id, pd.Timestamp(start))
    return s[s.index <= pd.Timestamp(end)] if end is not None else s


def _monthly(s: pd.Series) -> pd.Series:
    """Month-end last value, labelled by the first day of the month (the app's monthly index)."""
    m = s.resample("ME").last()
    m.index = m.index.to_period("M").to_timestamp()
    return m


def load_fred_credit_and_slope(start_ts, end_ts, api_key: Optional[str] = None
                               ) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """(BAA-AAA credit spread %, T10Y2Y slope %) as monthly series, each None when unavailable."""
    start = pd.Timestamp(start_ts) - pd.DateOffset(months=2)
    end = pd.Timestamp(end_ts) + pd.offsets.MonthEnd(1)
    spread = slope = None
    try:
        spread = _monthly((fred_series("BAA", start, end, api_key) - fred_series("AAA", start, end, api_key)).dropna())
        spread = spread if not spread.dropna().empty else None
    except Exception as e:
        logger.warning("FRED credit spread (BAA-AAA) unavailable: %s: %s", type(e).__name__, e)
        spread = None
    try:
        slope = _monthly(fred_series("T10Y2Y", start, end, api_key))
        slope = slope if not slope.dropna().empty else None
    except Exception as e:
        logger.warning("FRED curve slope (T10Y2Y) unavailable: %s: %s", type(e).__name__, e)
        slope = None
    return spread, slope


def attach_credit_and_slope(df: pd.DataFrame, spread: Optional[pd.Series], slope: Optional[pd.Series]
                            ) -> pd.DataFrame:
    """Add credit_spread / yc_slope (decimals) and their sources; NaN + 'unavailable' when missing."""
    out = df.copy()
    for col, src_col, series, label in (("credit_spread", "_credit_source", spread, "FRED:BAA-AAA"),
                                        ("yc_slope", "_slope_source", slope, "FRED:T10Y2Y")):
        if series is not None:
            out[col] = series.reindex(out.index).ffill() / 100.0
            out[src_col] = label
        else:
            out[col] = np.nan
            out[src_col] = "unavailable"
    return out


def is_fred_sourced(df: pd.DataFrame, source_col: str) -> bool:
    """True when the latest row's credit/slope column came from FRED (not a proxy or missing)."""
    return source_col in df.columns and str(df[source_col].iloc[-1]).startswith("FRED")


def missing_macro_fields(row: pd.Series,
                         fields=("regime_score", "credit_spread", "yc_slope", "realized_vol_12m")) -> list:
    """Fields a narrative needs that are missing for this row (so no LLM is asked to explain NaNs)."""
    return [f for f in fields if f not in row.index or pd.isna(row[f])]


# ── expected returns ──────────────────────────────────────────────────
def _ridge_pipeline(alpha: float):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def walk_forward_ridge(d: pd.DataFrame, x_cols: Iterable[str], target: str = "y_fwd",
                       horizon: int = HORIZON, min_train: int = 48, alpha: float = 5.0) -> pd.Series:
    """Out-of-sample prediction for every row, including the latest one.

    ``target`` at row j is the return over rows j+1 … j+horizon, so it is only known from row j+horizon
    on. The model for row i is therefore trained on rows < i - horizon (every one of those targets is
    fully realised by row i) and never on the last ``horizon`` rows before i.
    """
    x_cols = list(x_cols)
    feats = d.dropna(subset=x_cols)
    preds = pd.Series(np.nan, index=feats.index, dtype=float)
    model = _ridge_pipeline(alpha)
    for i in range(min_train + horizon, len(feats)):
        train = feats.iloc[: i - horizon].dropna(subset=[target])
        if len(train) < min_train:
            continue
        model.fit(train[x_cols], train[target])
        preds.iloc[i] = model.predict(feats[x_cols].iloc[i: i + 1])[0]
    return preds


def expected_return_model(macro: pd.DataFrame, horizon: int = HORIZON, min_train: int = 48,
                          alpha: float = 5.0, band_window: int = 24) -> Optional[pd.DataFrame]:
    """Expanding-window Ridge forecast of the S&P 500 12m return.

    Columns:
      exp_ann_return   forecast made at each month for the next 12 months (simple return); the last row
                       is today's forecast
      realized_fwd_12m what actually happened over those 12 months (NaN until it is known)
      pred_std         std of the model's realised out-of-sample errors (log-return units) over the last
                       ``band_window`` forecasts whose outcome was known at that month
    """
    d = macro.copy()
    if "sp500_ret_m" not in d:
        return None
    x_cols = [c for c in RIDGE_FEATURES if c in d.columns and d[c].notna().any()]
    if not x_cols:
        return None
    d = d.dropna(subset=["sp500_ret_m"])
    d["y_fwd"] = d["sp500_ret_m"].rolling(horizon).sum().shift(-horizon)
    if d.dropna(subset=x_cols + ["y_fwd"]).shape[0] < min_train + horizon:
        return None
    preds = walk_forward_ridge(d, x_cols, "y_fwd", horizon, min_train, alpha)
    out = pd.DataFrame(index=preds.index)
    out["exp_log_return"] = preds
    out["exp_ann_return"] = np.expm1(preds)
    out["sp500_ret_m"] = d["sp500_ret_m"].reindex(out.index)
    y = d["y_fwd"].reindex(out.index)
    out["realized_fwd_12m"] = np.expm1(y)
    # Forecast t's error is only known at t + horizon, so the band at t uses errors up to t - horizon.
    err = preds - y
    out["pred_std"] = err.rolling(band_window, min_periods=12).std().shift(horizon)
    return out.dropna(subset=["exp_ann_return"])


def oos_noise_correlation(fit_predict, n_runs: int = 20, n_months: int = 180, n_features: int = 3,
                          horizon: int = HORIZON, persistence: float = 0.95, seed: int = 0) -> float:
    """Leakage check: mean out-of-sample correlation between forecasts and realised forward returns
    when returns are independent noise and the features are persistent AR(1) noise (like real macro
    series: yields, spreads, volatility). Nothing can predict such returns, so a point-in-time model
    scores ~0 (slightly negative in small samples); a model that trains on targets that are not yet
    realised scores clearly above 0, because persistent features let it copy the overlapping target."""
    rng = np.random.default_rng(seed)
    cors = []
    for _ in range(n_runs):
        idx = pd.date_range("2000-01-01", periods=n_months, freq="MS")
        shocks = rng.normal(size=(n_months, n_features))
        x = np.zeros_like(shocks)
        for t in range(1, n_months):
            x[t] = persistence * x[t - 1] + shocks[t]
        d = pd.DataFrame(x, index=idx, columns=[f"x{k}" for k in range(n_features)])
        d["sp500_ret_m"] = rng.normal(0.0, 0.04, n_months)
        d["y_fwd"] = d["sp500_ret_m"].rolling(horizon).sum().shift(-horizon)
        preds = fit_predict(d, [f"x{k}" for k in range(n_features)])
        both = pd.concat([preds.rename("p"), d["y_fwd"]], axis=1).dropna()
        if len(both) > 10:
            cors.append(both["p"].corr(both["y_fwd"]))
    return float(np.mean(cors))
