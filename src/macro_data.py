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

from src.macro_model import add_regime, attach_credit_and_slope, is_fred_sourced, load_fred_credit_and_slope
from src.portfolio import TBILL_COLUMN, TBILL_TICKER, compute_hf_metrics, rf_monthly  # noqa: F401 (re-export)

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
def _fred_series_cached(start_ts, end_ts):
    """Successful FRED fetches are cached for a day; a failure raises (the TTL cache stores only
    returned values), so it is retried instead of being remembered as 'unavailable' for 24 hours."""
    spread, slope = load_fred_credit_and_slope(start_ts, end_ts, api_key=_get_fred_key())
    if spread is None:
        raise RuntimeError("FRED credit spread unavailable")
    return spread, slope


_fred_failed_at = [0.0]
_FRED_RETRY_SECONDS = 300


def _try_load_fred_series(start_ts, end_ts):
    """(credit_spread_pct, yc_slope_pct) monthly series from FRED (BAA-AAA, T10Y2Y), each None when
    unreachable (retried at most every 5 minutes). Uses the FRED API when FRED_API_KEY is set,
    otherwise the keyless CSV endpoint."""
    if time.time() - _fred_failed_at[0] < _FRED_RETRY_SECONDS:
        return None, None
    try:
        return _fred_series_cached(start_ts, end_ts)
    except Exception:
        _fred_failed_at[0] = time.time()
        return None, None


def _load_macro_stored_copy():
    """The last macro mart written by a live refresh, used only when the live download fails."""
    if _BQ_OK:
        try:
            bq_df = load_macro_from_bigquery()
            if bq_df is not None and not bq_df.empty and len(bq_df) > 100:
                if not (is_fred_sourced(bq_df, "_credit_source") and is_fred_sourced(bq_df, "_slope_source")):
                    spread, slope = _try_load_fred_series(bq_df.index.min(), bq_df.index.max())
                    bq_df = attach_credit_and_slope(bq_df, spread, slope)
                bq_df = add_regime(bq_df)
                bq_df["_data_source"] = "GCP BigQuery (stored copy; live download failed)"
                return bq_df.dropna(subset=["sp500"])
        except Exception:
            pass

    return None


@_ttl_cache(ttl_seconds=3600)
def load_macro() -> pd.DataFrame:
    """Pull S&P500, VIX, 10Y yield via BigQuery or yfinance fallback, with
    regime/regime_score columns. Mirrors app.py's load_macro() exactly."""
    tmap = {"sp500": "^GSPC", "vix": "^VIX", "dgs10": "^TNX", TBILL_COLUMN: TBILL_TICKER, "gold": "GLD", "oil": "USO"}
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
                close = close.dropna()
                if col == "sp500":            # the real last trading day in each month (the as-of date)
                    days = close.index.tz_localize(None) if close.index.tz is not None else close.index
                    obs = pd.Series(days, index=close.index).resample("ME").last()
                    obs.index = obs.index.to_period("M").to_timestamp()
                    frames["_obs_date"] = obs
                close = close.resample("ME").last()
                close.index = close.index.to_period("M").to_timestamp()
                frames[col] = close
                break
            except Exception:
                if attempt == 2:
                    pass
    if not frames or "sp500" not in frames:
        # Live prices unavailable: fall back to the last stored mart (real data, possibly stale)
        # before the synthetic demo frame.
        stored = _load_macro_stored_copy()
        if stored is not None:
            return stored
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
    if TBILL_COLUMN in df:
        df[TBILL_COLUMN] = df[TBILL_COLUMN].ffill()

    fred_credit, fred_slope = _try_load_fred_series(df.index.min(), df.index.max())
    df = attach_credit_and_slope(df, fred_credit, fred_slope)   # no proxy: NaN when FRED is down
    df = add_regime(df)                                          # point-in-time (expanding) z-scores
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


# compute_hf_metrics lives in src/portfolio.py (one copy for the app, MCP tools and briefings);
# it is re-exported here because callers import it from src.macro_data.
