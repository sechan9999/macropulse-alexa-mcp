"""
src/macro_data.py
─────────────────────────────────────────────────────────────────
Streamlit-free copy of app.py's macro data pipeline (load_macro,
load_spy, compute_hf_metrics) so it can be called from a headless
context — currently mcp_server.py.

app.py keeps its own copies (with @st.cache_data) for the live
dashboard; this module intentionally duplicates rather than imports
from app.py, since app.py executes Streamlit UI code at import time
and can't be safely imported outside a running Streamlit session.

Caching here is a minimal in-process TTL cache (no Streamlit runtime
available), keyed by function name + args.
"""

from __future__ import annotations

import os
import time
import functools
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from fredapi import Fred
    _FREDAPI_OK = True
except Exception:
    Fred = None
    _FREDAPI_OK = False

try:
    import pandas_datareader.data as pdr
    _PDR_OK = True
except Exception:
    pdr = None
    _PDR_OK = False

try:
    from src.bigquery_service import (
        is_bigquery_available,
        load_macro_from_bigquery,
        save_macro_to_bigquery,
    )
    _BQ_OK = is_bigquery_available()
except Exception:
    _BQ_OK = False


def _get_fred_key() -> Optional[str]:
    return os.environ.get("FRED_API_KEY")


def _ttl_cache(ttl_seconds: float):
    """Tiny process-local TTL cache — a headless stand-in for st.cache_data."""
    def decorator(fn):
        cache: dict = {}

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            if key in cache:
                value, expires_at = cache[key]
                if now < expires_at:
                    return value
            value = fn(*args, **kwargs)
            cache[key] = (value, now + ttl_seconds)
            return value

        return wrapper
    return decorator


@_ttl_cache(ttl_seconds=86400)
def _try_load_fred_series(start_ts, end_ts):
    """Return (credit_spread_pct, yc_slope_pct) series from FRED if reachable,
    else (None, None). credit_spread = BAA-AAA, slope = T10Y2Y. Both monthly."""
    try:
        start = pd.Timestamp(start_ts) - pd.DateOffset(months=2)
        end = pd.Timestamp(end_ts)
        key = _get_fred_key()
        if key and _FREDAPI_OK:
            fr = Fred(api_key=key)
            baa = pd.Series(fr.get_series("BAA", observation_start=start, observation_end=end))
            aaa = pd.Series(fr.get_series("AAA", observation_start=start, observation_end=end))
            slope = pd.Series(fr.get_series("T10Y2Y", observation_start=start, observation_end=end))
        elif _PDR_OK:
            baa = pdr.DataReader("BAA", "fred", start, end).iloc[:, 0]
            aaa = pdr.DataReader("AAA", "fred", start, end).iloc[:, 0]
            slope = pdr.DataReader("T10Y2Y", "fred", start, end).iloc[:, 0]
        else:
            return None, None

        spread = (baa - aaa).dropna()
        spread.index = pd.to_datetime(spread.index)
        slope.index = pd.to_datetime(slope.index)
        spread = spread.resample("ME").last()
        slope = slope.resample("ME").last()
        spread.index = spread.index.to_period("M").to_timestamp()
        slope.index = slope.index.to_period("M").to_timestamp()
        if spread.dropna().empty or slope.dropna().empty:
            return None, None
        return spread, slope
    except Exception:
        return None, None


@_ttl_cache(ttl_seconds=3600)
def load_macro() -> pd.DataFrame:
    """Pull S&P500, VIX, 10Y yield via BigQuery or yfinance fallback, with
    regime/regime_score columns. Mirrors app.py's load_macro() exactly."""
    if _BQ_OK:
        try:
            bq_df = load_macro_from_bigquery()
            if bq_df is not None and not bq_df.empty and len(bq_df) > 100:
                if "regime" not in bq_df.columns or "regime_score" not in bq_df.columns:
                    cs = (bq_df["credit_spread"] - bq_df["credit_spread"].mean()) / bq_df["credit_spread"].std()
                    rv = (bq_df["realized_vol_12m"] - bq_df["realized_vol_12m"].mean()) / bq_df["realized_vol_12m"].std()
                    bq_df["regime_score"] = cs.fillna(0) + rv.fillna(0)
                    bq_df["regime"] = np.select([bq_df["regime_score"] < -0.5, bq_df["regime_score"] > 0.5],
                                                 ["Risk-On 🟢", "Risk-Off 🔴"], default="Neutral 🟡")
                bq_df["_data_source"] = "GCP BigQuery"
                return bq_df.dropna(subset=["sp500"])
        except Exception:
            pass

    tmap = {"sp500": "^GSPC", "vix": "^VIX", "dgs10": "^TNX", "gold": "GLD", "oil": "USO"}
    frames = {}
    for col, tkr in tmap.items():
        for attempt in range(3):
            try:
                raw = yf.download(tkr, start="2005-01-01", auto_adjust=True,
                                   progress=False, multi_level_index=False, timeout=20)
                if raw.empty:
                    break
                close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close = close.dropna().resample("ME").last()
                close.index = close.index.to_period("M").to_timestamp()
                frames[col] = close
                break
            except Exception:
                if attempt == 2:
                    pass
    if not frames or "sp500" not in frames:
        idx = pd.date_range("2010-01-01", periods=180, freq="ME")
        frames["sp500"] = pd.Series([2000 + i * 15.0 for i in range(180)], index=idx)
        frames["vix"] = pd.Series(20.0, index=idx)
        frames["dgs10"] = pd.Series(4.0, index=idx)
        frames["gold"] = pd.Series(160.0, index=idx)
        frames["oil"] = pd.Series(60.0, index=idx)
        frames["_is_demo"] = pd.Series(1.0, index=pd.date_range("2010-01-01", periods=len(frames["sp500"]), freq="ME"))
    df = pd.DataFrame({k: v for k, v in frames.items() if k != "_is_demo"}).sort_index()
    df["_is_demo"] = "_is_demo" in frames
    df.index.name = "date"
    df["sp500_ret_m"] = np.log(df["sp500"]).diff()
    df["realized_vol_12m"] = df["sp500_ret_m"].rolling(12).std() * np.sqrt(12)
    df["realized_vol_3m"] = df["sp500_ret_m"].rolling(3).std() * np.sqrt(12)
    df["momentum_12_1"] = df["sp500_ret_m"].rolling(11).sum().shift(1)
    df["cumret"] = np.exp(df["sp500_ret_m"].cumsum()) * 100
    df["drawdown"] = df["cumret"] / df["cumret"].cummax() - 1
    df["dgs10"] = df["dgs10"].ffill()

    fred_credit, fred_slope = _try_load_fred_series(df.index.min(), df.index.max())
    if fred_credit is not None:
        df["credit_spread"] = fred_credit.reindex(df.index).ffill() / 100.0
        df["_credit_source"] = "FRED:BAA-AAA"
    else:
        df["credit_spread"] = (df["dgs10"] * 0.35 + 1.5 - df["dgs10"] * 0.1).abs() / 100
        df["_credit_source"] = "proxy"
    if fred_slope is not None:
        df["yc_slope"] = fred_slope.reindex(df.index).ffill() / 100.0
        df["_slope_source"] = "FRED:T10Y2Y"
    else:
        df["yc_slope"] = (df["dgs10"] - df["dgs10"] * 0.65) / 100
        df["_slope_source"] = "proxy"
    cs = (df["credit_spread"] - df["credit_spread"].mean()) / df["credit_spread"].std()
    rv = (df["realized_vol_12m"] - df["realized_vol_12m"].mean()) / df["realized_vol_12m"].std()
    df["regime_score"] = cs.fillna(0) + rv.fillna(0)
    df["regime"] = np.select([df["regime_score"] < -0.5, df["regime_score"] > 0.5],
                              ["Risk-On 🟢", "Risk-Off 🔴"], default="Neutral 🟡")
    df["_data_source"] = "Live yfinance"
    if _BQ_OK:
        try:
            save_macro_to_bigquery(df)
        except Exception:
            pass
    return df.dropna(subset=["sp500"])


@_ttl_cache(ttl_seconds=3600)
def load_spy(start: str, end: str) -> pd.Series:
    try:
        raw = yf.download("SPY", start=start, end=str(end), auto_adjust=True,
                           progress=False, multi_level_index=False, timeout=20)
        if raw.empty:
            return pd.Series(dtype=float)
        close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.resample("ME").last()
        close.index = close.index.to_period("M").to_timestamp()
        return np.log(close).diff().dropna()
    except Exception:
        return pd.Series(dtype=float)


def compute_hf_metrics(rets: pd.Series, bench_rets: Optional[pd.Series] = None) -> dict:
    ann_ret = rets.mean() * 12
    ann_vol = rets.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    neg = rets[rets < 0]
    sortino = ann_ret / (neg.std() * np.sqrt(12)) if len(neg) > 0 else np.nan
    cum = np.exp(rets.cumsum())
    mdd = (cum / cum.cummax() - 1).min()
    calmar = ann_ret / abs(mdd) if mdd < 0 else np.nan
    win_rate = (rets > 0).mean()
    avg_win = rets[rets > 0].mean() if (rets > 0).any() else 0
    avg_loss = rets[rets < 0].mean() if (rets < 0).any() else 0
    alpha = np.nan
    if bench_rets is not None:
        aligned = rets.align(bench_rets, join="inner")
        if len(aligned[0]) > 12:
            cov = np.cov(aligned[0], aligned[1])
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else np.nan
            alpha = (ann_ret - beta * bench_rets.mean() * 12) if not np.isnan(beta) else np.nan
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe, sortino=sortino,
                mdd=mdd, calmar=calmar, win_rate=win_rate, avg_win=avg_win,
                avg_loss=avg_loss, alpha=alpha)
