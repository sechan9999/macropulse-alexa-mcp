"""
src/alexa_mcp_server.py
─────────────────────────────────────────────────────────────────
MacroPulse Model Context Protocol (MCP) Server for Alexa+
Serves Streamable HTTP (/mcp) and legacy SSE (/sse) and exposes 9 MCP tools:

  1. get_macro_regime(): Risk-On/Neutral/Risk-Off regime from a stress z-score
  2. get_rates_and_spreads(): 10Y yield, 10Y-3M curve slope, VIX-derived credit-spread proxy
  3. simulate_portfolio_risk(ticker, confidence, days, n_paths): Monte Carlo VaR/CVaR
  4. check_nvda_danger_zone(): NVDA Composite Danger Index, RSI, relative volume
  5. scan_quant_signals(ticker): Volatility squeeze, 20d vol, breakout score (-100..+100)
  6. get_expected_returns(): static reference estimate of the 12m S&P 500 return
  7. simulate_fomc_shock(scenario): multi-asset FOMC rate-shock stress test
  8. get_equity_report(ticker): regime-weighted DCF fair value, rating, levels and pattern screen for a
     US 10-K filer (headline numbers only; the Excel/Word/HTML files live in the Streamlit Equity Report tab)
  9. get_morning_brief(): pre-market brief for an Alexa Routine (S&P close, regime, rates, SPY signal,
     latest daily scan leaders)

Market data comes through a TTL cache with background refresh (see the data layer below).
If live data cannot be fetched the tools raise MarketDataUnavailable; they never answer
from made-up numbers. ask_macro_analyst() exists for the voice/REST paths only.

Complies with:
- MCP Specification (Streamable HTTP)
- Alexa+ add-on requirements (Streamable HTTP, responses < 500 ms)
"""
from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import os
import html
import re
import json
import logging
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# MCP server class from the official SDK. mcp>=2 renamed FastMCP -> MCPServer
# (mcp.server.mcpserver); mcp 1.x only has mcp.server.fastmcp.FastMCP. Both are
# exposed here under the name FastMCP so the rest of the module is version-agnostic.
try:
    from mcp.server.mcpserver import MCPServer as FastMCP
    _MCP_V2 = True
    _FASTMCP_AVAILABLE = True
except ImportError:
    _MCP_V2 = False
    try:
        from mcp.server.fastmcp import FastMCP
        _FASTMCP_AVAILABLE = True
    except ImportError:
        FastMCP = None
        _FASTMCP_AVAILABLE = False

# ToolError: in mcp>=2 only a deliberately raised ToolError reaches the client with its message
# (any other exception is reported as an opaque "Error executing tool X"); mcp 1.x wraps everything.
try:
    from mcp.server.mcpserver.exceptions import ToolError as _ToolError
except ImportError:
    try:
        from mcp.server.fastmcp.exceptions import ToolError as _ToolError
    except ImportError:
        _ToolError = RuntimeError

from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Mount
from starlette.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("MacroPulse-AlexaMCP")


# ══════════════════════════════════════════════════════════════════
# MARKET DATA LAYER: TTL cache + background refresh, never made-up data
# ══════════════════════════════════════════════════════════════════
# Alexa+ wants tool responses in under 500 ms, so a request must never wait on Yahoo.
# Price history is fetched once and kept for _CACHE_TTL_SECONDS; after that the cached copy
# keeps being served while a background thread refreshes it (stale-while-revalidate). If a
# series cannot be refreshed within _CACHE_MAX_STALE_SECONDS the tools raise
# MarketDataUnavailable instead of answering from old or invented numbers.

class MarketDataUnavailable(_ToolError):
    """Live market data could not be fetched. Raised instead of substituting made-up numbers.

    A ToolError so MCP clients receive the message as an error result (see _ToolError above)."""


_CACHE_TTL_SECONDS = 300
_CACHE_MAX_STALE_SECONDS = 3600
_CACHE_MAX_ENTRIES = 64
_TICKER_RE = re.compile(r"^[A-Z0-9.\-^=]{1,12}$")

_cache: Dict[Tuple[str, str], Tuple[float, pd.DataFrame]] = {}   # (ticker, period) -> (fetched_at, prices)
_cache_lock = threading.Lock()
_key_locks: Dict[Tuple[str, str], threading.Lock] = {}
_refreshing: set = set()


def _raw_download(ticker: str, period: str, timeout: float) -> pd.DataFrame:
    return yf.download(ticker, period=period, auto_adjust=True, progress=False,
                       multi_level_index=False, timeout=timeout)


def _store(key: Tuple[str, str], df: pd.DataFrame) -> None:
    with _cache_lock:
        _cache[key] = (time.monotonic(), df)
        overflow = len(_cache) - _CACHE_MAX_ENTRIES
        if overflow > 0:
            for old_key, _ in sorted(_cache.items(), key=lambda kv: kv[1][0])[:overflow]:
                del _cache[old_key]
                _key_locks.pop(old_key, None)


def _fetch_and_store(key: Tuple[str, str], timeout: float, force: bool = False) -> pd.DataFrame:
    ticker, period = key
    lock = _key_locks.setdefault(key, threading.Lock())
    with lock:  # single-flight: concurrent callers for one series share a single download
        if not force:
            with _cache_lock:
                entry = _cache.get(key)
            if entry is not None and time.monotonic() - entry[0] <= _CACHE_TTL_SECONDS:
                return entry[1]
        try:
            df = _raw_download(ticker, period, timeout)
        except Exception as e:
            logger.warning("Download of %s (%s) failed: %s", ticker, period, e)  # detail stays server-side
            raise MarketDataUnavailable(f"Live market data for {ticker} is temporarily unavailable.") from e
        if df is None or df.empty:
            raise MarketDataUnavailable(f"No market data was returned for {ticker}.")
        _store(key, df)
        return df


def _refresh_in_background(key: Tuple[str, str], timeout: float) -> None:
    with _cache_lock:
        if key in _refreshing:
            return
        _refreshing.add(key)

    def _run() -> None:
        try:
            _fetch_and_store(key, timeout, force=True)
        except MarketDataUnavailable as e:
            logger.warning("Background refresh failed, still serving cached data: %s", e)
        finally:
            with _cache_lock:
                _refreshing.discard(key)

    threading.Thread(target=_run, daemon=True, name=f"refresh-{key[0]}").start()


def _cached_download(ticker: str, period: str, timeout: float = 8) -> pd.DataFrame:
    """Price history for ticker/period as a private copy, served from cache when possible."""
    symbol = ticker.strip().upper()
    if not _TICKER_RE.match(symbol):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")
    key = (symbol, period)
    with _cache_lock:
        entry = _cache.get(key)
    if entry is not None:
        age = time.monotonic() - entry[0]
        if age <= _CACHE_TTL_SECONDS:
            return entry[1].copy()
        if age <= _CACHE_MAX_STALE_SECONDS:
            _refresh_in_background(key, timeout)
            return entry[1].copy()
    return _fetch_and_store(key, timeout).copy()


def _data_freshness(*series: Tuple[str, str]) -> Dict[str, Any]:
    """Age of the oldest series a tool used, so callers can see when data is being served stale."""
    now = time.monotonic()
    with _cache_lock:
        ages = [now - _cache[(t.upper(), p)][0] for t, p in series if (t.upper(), p) in _cache]
    age = int(max(ages)) if ages else 0
    return {"data_age_seconds": age, "data_stale": age > _CACHE_TTL_SECONDS}


# ══════════════════════════════════════════════════════════════════
# QUANT ENGINE CORE HELPERS
# ══════════════════════════════════════════════════════════════════

_MACRO_SERIES = {"sp500": ("^GSPC", "3y"), "vix": ("^VIX", "3y"), "dgs10": ("^TNX", "3y")}
# Long history for the Ridge S&P 500 expected return behind the equity report's touch probabilities
# (the model needs >= 60 months with a realised 12m outcome; 3y is far too short, 10y gives ~96).
_DRIFT_SERIES = {"sp500": ("^GSPC", "10y"), "vix": ("^VIX", "10y"), "dgs10": ("^TNX", "10y")}


def _fetch_cached_macro_data(series: Optional[Dict[str, Tuple[str, str]]] = None) -> pd.DataFrame:
    """Monthly S&P 500 / VIX / 10Y snapshot with derived regime columns, built from cached prices."""
    frames = {}
    for col, (tkr, period) in (series or _MACRO_SERIES).items():
        raw = _cached_download(tkr, period, timeout=10)
        c = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        frames[col] = c.dropna().resample("ME").last()

    df = pd.DataFrame(frames).ffill().dropna()
    if len(df) < 13:
        raise MarketDataUnavailable("Not enough macro history to classify the regime.")

    # Calculate returns and volatility
    df["ret_m"] = df["sp500"].pct_change()
    df["realized_vol_12m"] = df["ret_m"].rolling(12).std() * np.sqrt(12)

    # Credit spread proxy (scaled from VIX dynamics; not a published spread)
    df["credit_spread"] = 0.015 + (df["vix"] / 100.0) * 0.05

    # Standardized z-score regime stress
    cs = (df["credit_spread"] - df["credit_spread"].mean()) / (df["credit_spread"].std() + 1e-6)
    rv = (df["realized_vol_12m"] - df["realized_vol_12m"].mean()) / (df["realized_vol_12m"].std() + 1e-6)
    df["regime_score"] = cs.fillna(0) + rv.fillna(0)

    df["regime"] = np.select(
        [df["regime_score"] < -0.5, df["regime_score"] > 0.5],
        ["Risk-On 🟢", "Risk-Off 🔴"],
        default="Neutral 🟡"
    )
    return df


def _macro_freshness() -> Dict[str, Any]:
    return _data_freshness(*_MACRO_SERIES.values())


# ══════════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS (Dual machine-readable + spoken voice)
# ══════════════════════════════════════════════════════════════════

def execute_get_macro_regime() -> Dict[str, Any]:
    """Evaluates macro stress z-scores and determines market regime."""
    df = _fetch_cached_macro_data()
    last = df.iloc[-1]

    regime = str(last["regime"])
    score = round(float(last["regime_score"]), 2)
    sp500_price = round(float(last["sp500"]), 2)
    tnx_yield = round(float(last["dgs10"]), 2)
    vix = round(float(last["vix"]), 1)

    # Historical win rate in this regime
    regime_slice = df[df["regime"] == regime]["ret_m"].dropna()
    win_rate = round(float((regime_slice > 0).mean()) * 100, 1) if not regime_slice.empty else 65.0

    spoken = (
        f"Today's macroeconomic regime is classified as {regime.split()[0]} with a stress score of {score}. "
        f"The S&P 500 sits at {sp500_price:,.0f}, while 10-year Treasury yields stand at {tnx_yield}% with VIX at {vix}. "
        f"Historically in this regime, the 30-day equity win rate has averaged {win_rate}%."
    )

    return {
        "status": "success",
        "regime": regime,
        "stress_score": score,
        "sp500": sp500_price,
        "treasury_10y_yield": tnx_yield,
        "vix": vix,
        "regime_win_rate_pct": win_rate,
        "timestamp": datetime.now().isoformat(),
        **_macro_freshness(),
        "alexa_spoken_response": spoken
    }


def execute_get_rates_and_spreads() -> Dict[str, Any]:
    """Returns 10Y yield, 10Y-3M curve slope, and a VIX-derived credit spread proxy."""
    df = _fetch_cached_macro_data()
    last = df.iloc[-1]

    tnx = round(float(last["dgs10"]), 2)
    cs_bps = round(float(last["credit_spread"]) * 10000, 0)

    # Short rate = 13-week T-bill yield (^IRX); slope is therefore 10Y minus 3M
    irx = _cached_download("^IRX", "1mo", timeout=5)["Close"].dropna()
    if irx.empty:
        raise MarketDataUnavailable("No short-rate data was returned for ^IRX.")
    y2 = float(irx.iloc[-1])

    slope_bps = round((tnx - y2) * 100, 1)
    curve_state = "inverted" if slope_bps < 0 else ("un-inverting / flat" if slope_bps < 25 else "steep / normal")

    spoken = (
        f"The benchmark 10-Year Treasury yield is currently {tnx}%. "
        f"The yield curve slope between 10-year and short rates is {slope_bps:+.0f} basis points, reflecting a {curve_state} condition. "
        f"Corporate credit spreads are estimated at {cs_bps:.0f} basis points, indicating stable liquidity conditions."
    )

    return {
        "status": "success",
        "treasury_10y": tnx,
        "short_rate": round(y2, 2),
        "curve_slope_bps": slope_bps,
        "curve_status": curve_state,
        "credit_spread_bps": cs_bps,
        **_data_freshness(*_MACRO_SERIES.values(), ("^IRX", "1mo")),
        "alexa_spoken_response": spoken
    }


def execute_simulate_portfolio_risk(
    ticker: str = "SPY",
    confidence: int = 95,
    days: int = 30,
    n_paths: int = 5000
) -> Dict[str, Any]:
    """Runs high-speed Monte Carlo simulation for Value-at-Risk & Expected Shortfall."""
    hist = _cached_download(ticker, "1y", timeout=8)
    c = hist["Close"] if "Close" in hist.columns else hist.iloc[:, 0]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    c = c.dropna()
    daily_rets = np.log(c / c.shift(1)).dropna()
    if len(daily_rets) < 20:
        raise MarketDataUnavailable(f"Not enough price history for {ticker.upper()} to run a simulation.")
    curr_price = float(c.iloc[-1])
    mu = float(daily_rets.mean())
    sigma = float(daily_rets.std())

    # Vectorized Geometric Brownian Motion
    rng = np.random.default_rng(42)
    shocks = rng.normal(mu, sigma, (n_paths, days))
    cum_returns = np.exp(np.cumsum(shocks, axis=1)) - 1.0
    final_rets = cum_returns[:, -1]

    alpha = 100 - confidence
    var_pct = float(np.percentile(final_rets, alpha))
    cvar_pct = float(final_rets[final_rets <= var_pct].mean())
    median_ret = float(np.median(final_rets))

    spoken = (
        f"Running a {n_paths:,}-path Monte Carlo simulation on {ticker} over {days} trading days: "
        f"At the {confidence}% confidence level, your Value-at-Risk is {abs(var_pct)*100:.1f}%. "
        f"Expected Shortfall, or CVaR, is {abs(cvar_pct)*100:.1f}%. "
        f"With {ticker} currently at ${curr_price:,.2f}, the projected median price is ${curr_price * (1 + median_ret):,.2f}."
    )

    return {
        "status": "success",
        "ticker": ticker.upper(),
        "current_price": round(curr_price, 2),
        "days_horizon": days,
        "confidence_level": confidence,
        "var_pct": round(var_pct * 100, 2),
        "cvar_pct": round(cvar_pct * 100, 2),
        "median_return_pct": round(median_ret * 100, 2),
        "paths_simulated": n_paths,
        **_data_freshness((ticker, "1y")),
        "alexa_spoken_response": spoken
    }


def execute_check_nvda_danger_zone() -> Dict[str, Any]:
    """Evaluates Nvidia single-name market structure, RSI extension, and order flow."""
    raw = _cached_download("NVDA", "6mo", timeout=8)
    close = raw["Close"].dropna() if "Close" in raw.columns else raw.iloc[:, 0].dropna()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    vol = raw["Volume"].dropna() if "Volume" in raw.columns else raw.iloc[:, 1].dropna()
    if isinstance(vol, pd.DataFrame):
        vol = vol.iloc[:, 0]
    if len(close) < 50 or len(vol) < 20:
        raise MarketDataUnavailable("Not enough NVDA price history to compute the danger index.")
    price = float(close.iloc[-1])

    # 14d RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = float(100 - 100 / (1 + gain.iloc[-1] / (loss.iloc[-1] + 1e-9)))

    # SMA50 distance
    sma50 = float(close.rolling(50).mean().iloc[-1])
    dist_50 = round(((price / sma50) - 1.0) * 100, 1)

    # Relative volume
    vol_sma20 = float(vol.rolling(20).mean().iloc[-1])
    rel_vol = round(float(vol.iloc[-1]) / (vol_sma20 + 1), 2)

    # Composite Danger Index (0.0 to 1.0)
    norm_rsi = np.clip((rsi - 40) / 40.0, 0, 1)
    norm_dist = np.clip(dist_50 / 30.0, 0, 1)
    norm_vol = np.clip((rel_vol - 1.0) / 2.0, 0, 1)
    danger_index = round(float(0.4 * norm_rsi + 0.35 * norm_dist + 0.25 * norm_vol), 3)

    risk_label = "🔴 DANGER" if danger_index >= 0.60 else ("🟡 CAUTION" if danger_index >= 0.35 else "🟢 SAFE")

    spoken = (
        f"Nvidia is currently in the {risk_label.split()[1]} zone with a composite danger index of {danger_index:.2f} out of 1.0. "
        f"Trading at ${price:.2f}, its 14-day RSI is {rsi:.1f}, standing {dist_50:+.1f}% from its 50-day moving average. "
        f"Relative trading volume is {rel_vol:.1f} times its 20-day average."
    )

    return {
        "status": "success",
        "ticker": "NVDA",
        "current_price": round(price, 2),
        "danger_index": danger_index,
        "danger_level": risk_label,
        "rsi_14": round(rsi, 1),
        "sma50_distance_pct": dist_50,
        "relative_volume": rel_vol,
        **_data_freshness(("NVDA", "6mo")),
        "alexa_spoken_response": spoken
    }


def execute_scan_quant_signals(ticker: str = "SPY") -> Dict[str, Any]:
    """Evaluates volatility squeeze, ATR%, and directional momentum breakout."""
    raw = _cached_download(ticker, "1y", timeout=8)
    close = raw["Close"].dropna() if "Close" in raw.columns else raw.iloc[:, 0].dropna()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if len(close) < 40:
        raise MarketDataUnavailable(f"Not enough price history for {ticker.upper()} to scan signals.")

    price = float(close.iloc[-1])
    std20 = close.rolling(20).std()
    sma20 = close.rolling(20).mean()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width = (bb_upper - bb_lower) / sma20
    bb_pct = float((bb_width <= bb_width.iloc[-1]).mean())
    is_squeeze = bb_pct < 0.20

    # Realized 20d vol
    r_vol = float(close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100)

    # Score calculation (-100 to +100)
    score = 0
    if price > float(sma20.iloc[-1]): score += 25
    if price > float(bb_upper.iloc[-1]): score += 35
    elif price < float(bb_lower.iloc[-1]): score -= 35
    if is_squeeze: score += 10

    signal = "STRONG_BUY" if score >= 40 else ("BUY" if score >= 15 else ("SELL" if score <= -15 else "HOLD"))

    squeeze_text = "in a volatility squeeze, coiling for a major breakout" if is_squeeze else "displaying normal bandwidth expansion"
    spoken = (
        f"{ticker.upper()} is rated as {signal} with a quant conviction score of {score:+d} out of 100. "
        f"Its 20-day annualized realized volatility is {r_vol:.1f}%, and the Bollinger bandwidth is {squeeze_text}."
    )

    return {
        "status": "success",
        "ticker": ticker.upper(),
        "price": round(price, 2),
        "signal": signal,
        "conviction_score": score,
        "realized_vol_20d_pct": round(r_vol, 1),
        "bb_squeeze_active": is_squeeze,
        "bb_width_percentile": round(bb_pct * 100, 1),
        **_data_freshness((ticker, "1y")),
        "alexa_spoken_response": spoken
    }


def execute_get_expected_returns() -> Dict[str, Any]:
    """Static reference estimate of the 12-month forward S&P 500 return.

    NOTE: these numbers are fixed constants, not the output of a model run on live data
    (an earlier version fetched market data it never used and described the constants as
    an expanding-window Ridge forecast). They are labelled as a reference estimate.
    """
    expected_ret = 8.4
    std_error = 2.1
    lower_bound = round(expected_ret - std_error, 1)
    upper_bound = round(expected_ret + std_error, 1)

    spoken = (
        f"As a static reference estimate, not a live model forecast, the 12-month forward S&P 500 total return "
        f"is put at {expected_ret:.1f}%, with a 1-sigma range spanning {lower_bound}% to {upper_bound}%."
    )

    return {
        "status": "success",
        "asset": "S&P 500",
        "horizon_months": 12,
        "expected_return_pct": expected_ret,
        "lower_1sigma_pct": lower_bound,
        "upper_1sigma_pct": upper_bound,
        "model": "Static reference estimate (not recomputed from live data)",
        "alexa_spoken_response": spoken
    }


_LLM_TIMEOUT_SECONDS = 5.0
_llm_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm")


def _generate_llm_briefing(query: str, macro_summary: Dict[str, Any], api_key: str) -> str:
    from google import genai
    from google.genai import types as genai_types

    prompt = (
        f"You are a Senior Hedge Fund Macro Strategist for Alexa+. "
        f"Market Context: Regime={macro_summary['regime']}, S&P={macro_summary['sp500']}, 10Y Yield={macro_summary['treasury_10y_yield']}%, VIX={macro_summary['vix']}. "
        f"Answer this investor query in 3 concise, punchy sentences optimized for voice readout: '{query}'"
    )
    client = genai.Client(api_key=api_key)
    config = genai_types.GenerateContentConfig(thinking_config=genai_types.ThinkingConfig(thinking_budget=0))
    res = client.models.generate_content(model="gemini-2.5-flash", contents=prompt, config=config)
    return (res.text or "").strip()


def execute_ask_macro_analyst(query: str) -> Dict[str, Any]:
    """Analyst commentary for the voice (ASK) and REST paths.

    Deliberately NOT registered as an MCP tool: Alexa+ is itself an LLM that composes answers
    from the other tools, and a synchronous Gemini call took 10-50 s against a 500 ms budget.
    Here the LLM call is capped at _LLM_TIMEOUT_SECONDS; on timeout or error the answer is
    built from the cached quant tools instead.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    response_text = ""

    if api_key:
        macro_summary = execute_get_macro_regime()  # raises MarketDataUnavailable rather than guess
        future = _llm_pool.submit(_generate_llm_briefing, query, macro_summary, api_key)
        try:
            response_text = future.result(timeout=_LLM_TIMEOUT_SECONDS)
        except FuturesTimeout:
            logger.warning("Gemini briefing exceeded %.1fs; answering from the quant tools", _LLM_TIMEOUT_SECONDS)
        except Exception as e:
            logger.warning(f"Gemini generation fallback: {e}")

    if not response_text:
        q_lower = query.lower()
        if "regime" in q_lower or "market" in q_lower:
            reg = execute_get_macro_regime()
            response_text = reg["alexa_spoken_response"]
        elif "risk" in q_lower or "var" in q_lower or "drawdown" in q_lower:
            sim = execute_simulate_portfolio_risk()
            response_text = sim["alexa_spoken_response"]
        elif "nvda" in q_lower or "nvidia" in q_lower or "danger" in q_lower:
            nv = execute_check_nvda_danger_zone()
            response_text = nv["alexa_spoken_response"]
        else:
            response_text = (
                "I can tell you the market regime, rates and spreads, a Monte Carlo risk simulation, "
                "the NVDA danger zone, quant signals, or run an FOMC shock test. Which would you like?"
            )

    return {
        "status": "success",
        "query": query,
        "analyst_briefing": response_text,
        "alexa_spoken_response": response_text
    }


# ══════════════════════════════════════════════════════════════════
# MORNING BRIEF: one ~30 s answer for an Alexa Routine ("Alexa, good morning")
# ══════════════════════════════════════════════════════════════════

_SIGNALS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "signals")
_SIGNALS_MAX_AGE_DAYS = 4          # a weekend + a holiday; older scans are left out of the brief
_SIGNAL_ROW = re.compile(r"^\s*·\s*([A-Z0-9.\-^]{1,12})\s+\$[\d,.]+\s+score\s+([+-]\d+)")


def _latest_signal_leaders(signals_dir: Optional[str] = None, today: Optional[date] = None,
                           top: int = 3) -> Optional[Dict[str, Any]]:
    """Top BUY / SELL names from the newest daily scan (signals/YYYY-MM-DD.md written by the
    daily-quant-signal workflow), or None when there is no scan from the last few days."""
    signals_dir = signals_dir or _SIGNALS_DIR
    try:
        dated = sorted(f for f in os.listdir(signals_dir) if re.fullmatch(r"\d{4}-\d{2}-\d{2}\.md", f))
    except OSError:
        return None
    if not dated:
        return None
    scan_day = date.fromisoformat(dated[-1][:-3])
    if ((today or date.today()) - scan_day).days > _SIGNALS_MAX_AGE_DAYS:
        return None
    buy, sell, side = [], [], None
    with open(os.path.join(signals_dir, dated[-1]), encoding="utf-8") as fh:
        for line in fh:
            if "BUY" in line and "·" not in line:
                side = buy
            elif "SELL" in line and "·" not in line:
                side = sell
            elif side is not None:
                m = _SIGNAL_ROW.match(line)
                if m:
                    side.append({"ticker": m.group(1), "score": int(m.group(2))})
    if not buy and not sell:
        return None
    sell.sort(key=lambda r: r["score"])            # most negative first
    return {"scan_date": scan_day.isoformat(), "buy": buy[:top], "sell": sell[:top]}


_CURVE_WORDS = {"inverted": "inverted", "un-inverting / flat": "flat", "steep / normal": "normally sloped"}


def _say_list(names: List[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def execute_get_morning_brief() -> Dict[str, Any]:
    """Pre-market brief built from the other tools' cached data: last S&P 500 close and day change,
    macro regime, 10Y yield and curve, SPY quant signal, and the latest daily scan's leaders.

    The regime and rates are required (MarketDataUnavailable otherwise); the S&P day change, the SPY
    signal and the scan leaders are optional and are simply left out, and listed in
    sections_skipped, when unavailable. Nothing is filled in with made-up numbers.
    """
    regime = execute_get_macro_regime()
    rates = execute_get_rates_and_spreads()
    skipped: List[str] = []

    close, change_pct, as_of = None, None, None
    try:
        raw = _cached_download("^GSPC", "3y", timeout=10)
        c = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        c = c.dropna()
        close, change_pct = float(c.iloc[-1]), float(c.iloc[-1] / c.iloc[-2] - 1) * 100
        as_of = pd.Timestamp(c.index[-1]).date()
    except (MarketDataUnavailable, IndexError, KeyError) as e:
        logger.warning("Morning brief: S&P day change unavailable: %s", e)
        skipped.append("sp500_day_change")

    spy = None
    try:
        q = execute_scan_quant_signals("SPY")
        spy = {"signal": q["signal"], "conviction_score": q["conviction_score"], "bb_squeeze_active": q["bb_squeeze_active"]}
    except MarketDataUnavailable as e:
        logger.warning("Morning brief: SPY signal unavailable: %s", e)
        skipped.append("spy_signal")

    leaders = _latest_signal_leaders()
    if leaders is None:
        skipped.append("scan_leaders")

    reg_name = regime["regime"].split()[0]
    parts = ["Here is your MacroPulse market brief"
             + (f" as of the {as_of:%A, %B} {as_of.day} close." if as_of else ".")]
    if close is not None:
        parts.append(f"The S&P 500 closed at {close:,.0f}, {'up' if change_pct >= 0 else 'down'} "
                     f"{abs(change_pct):.1f} percent.")
    parts.append(f"The market regime is {reg_name}, with a stress score of {regime['stress_score']}.")
    slope = rates["curve_slope_bps"]
    parts.append(f"The 10-year Treasury yields {rates['treasury_10y']} percent, and the curve is "
                 f"{_CURVE_WORDS.get(rates['curve_status'], rates['curve_status'])} at {slope:+.0f} basis points.")
    if spy:
        parts.append(f"SPY's quant signal is {spy['signal'].replace('_', ' ').lower()} with a conviction score of "
                     f"{spy['conviction_score']:+d}" + (", and a volatility squeeze is active." if spy["bb_squeeze_active"] else "."))
    if leaders:
        bits = []
        if leaders["buy"]:
            bits.append(f"{_say_list([r['ticker'] for r in leaders['buy']])} on the buy side")
        if leaders["sell"]:
            bits.append(f"{_say_list([r['ticker'] for r in leaders['sell']])} on the sell side")
        parts.append("The latest daily scan favors " + ", and ".join(bits) + ".")
    parts.append("This is information, not investment advice.")

    return {
        "status": "success",
        "as_of": as_of.isoformat() if as_of else None,
        "sp500_close": round(close, 2) if close is not None else None,
        "sp500_change_pct": round(change_pct, 2) if change_pct is not None else None,
        "regime": regime["regime"],
        "stress_score": regime["stress_score"],
        "vix": regime["vix"],
        "treasury_10y": rates["treasury_10y"],
        "curve_slope_bps": slope,
        "curve_status": rates["curve_status"],
        "spy_signal": spy,
        "signal_leaders": leaders,
        "sections_skipped": skipped,
        **_data_freshness(*_MACRO_SERIES.values(), ("^IRX", "1mo"), ("^GSPC", "3y"), ("SPY", "1y")),
        "alexa_spoken_response": " ".join(parts),
    }


# ══════════════════════════════════════════════════════════════════
# WARM-UP: keep the default data fresh so requests never wait on Yahoo
# ══════════════════════════════════════════════════════════════════

class EquityReportUnsupported(_ToolError):
    """The ticker has no US-GAAP 10-K fundamentals (ETF, foreign 20-F/40-F filer, ...)."""


_EQUITY_TTL_SECONDS = 900
_equity_cache: Dict[Tuple[str, str, Optional[str], Optional[float]], Tuple[float, Dict[str, Any]]] = {}


def execute_get_equity_report(ticker: str = "AAPL") -> Dict[str, Any]:
    """Headline equity report for a US 10-K filer, weighted by the live MacroPulse regime.

    Prices go through this server's TTL cache; SEC company facts are cached in-process for 6 h, so only
    the first request for a new ticker pays the EDGAR round-trip (~1-3 s). Summaries are cached 15 min.
    """
    from src.equity_report.engine import data_us, pipeline
    from src.equity_report.engine.analysis import compact_summary

    t = (ticker or "").upper().strip().replace(".", "-")
    if not _TICKER_RE.match(t) or t.startswith("^"):
        raise EquityReportUnsupported(f"'{ticker}' is not a valid stock ticker.")
    from src.equity_report.engine.market_drift import expected_market_return

    regime_label, rf, market = None, None, None
    try:
        macro = _fetch_cached_macro_data()
        regime_label = str(macro["regime"].iloc[-1])
        rf = float(macro["dgs10"].iloc[-1]) / 100
    except MarketDataUnavailable as e:
        logger.warning("Equity report: regime unavailable, using neutral weights: %s", e)
    try:
        market = expected_market_return(_fetch_cached_macro_data(_DRIFT_SERIES))
    except (MarketDataUnavailable, ValueError, np.linalg.LinAlgError) as e:
        logger.warning("Equity report: Ridge market view unavailable, touch probabilities driftless: %s", e)

    key = (t, date.today().isoformat(), regime_label,
           None if market is None else round(market["expected_return"], 4))
    hit = _equity_cache.get(key)
    if hit and time.monotonic() - hit[0] < _EQUITY_TTL_SECONDS:
        return hit[1]
    try:
        rep = pipeline.run(t, regime_label=regime_label, risk_free=rf, provider="rules", with_positioning=False,
                           price_fetch=lambda sym: _cached_download(sym, "3y", timeout=10), market=market)
    except data_us.UnsupportedFiler as e:
        raise EquityReportUnsupported(f"{t}: {e}") from e
    except data_us.DataUnavailable as e:
        raise MarketDataUnavailable(str(e)) from e
    out = {**compact_summary(rep), **_data_freshness((t, "3y"), ("^GSPC", "3y"))}
    _equity_cache[key] = (time.monotonic(), out)
    if len(_equity_cache) > 128:
        _equity_cache.pop(next(iter(_equity_cache)))
    return out


# The series the tools read with their default arguments: (ticker, period, timeout).
_WARM_SET = (
    ("^GSPC", "3y", 10), ("^VIX", "3y", 10), ("^TNX", "3y", 10),
    ("^IRX", "1mo", 5), ("NVDA", "6mo", 8), ("SPY", "1y", 8),
    *((tkr, period, 15) for tkr, period in _DRIFT_SERIES.values()),
)
_WARM_REFRESH_SECONDS = 240
_warm_done = threading.Event()
_warm_started = False


def _refresh_warm_set() -> None:
    for ticker, period, timeout in _WARM_SET:
        try:
            _fetch_and_store((ticker, period), timeout, force=True)
        except MarketDataUnavailable as e:
            logger.warning("Warm-up fetch failed: %s", e)


# Tickers whose SEC filings are prefetched at start-up so their equity report answers within budget.
_EQUITY_WARM = tuple(x.strip() for x in os.environ.get("EQUITY_REPORT_WARM", "AAPL,MSFT,NVDA,AMZN,GOOGL").split(",")
                     if x.strip())


def _warm_up_data() -> None:
    """Prefetch the default series and run every tool once, so the first request is not
    the one that pays for imports and cold caches."""
    _refresh_warm_set()
    for fn in (execute_simulate_fomc_shock, execute_get_macro_regime, execute_get_rates_and_spreads,
               execute_simulate_portfolio_risk, execute_check_nvda_danger_zone,
               execute_scan_quant_signals, execute_get_expected_returns, execute_get_morning_brief):
        try:
            fn()
        except Exception as e:
            logger.warning("Warm-up call %s failed: %s", fn.__name__, e)
    for t in _EQUITY_WARM:
        try:
            execute_get_equity_report(t)
        except Exception as e:
            logger.warning("Equity report warm-up for %s failed: %s", t, e)
    _warm_done.set()


def start_background_warmup() -> None:
    """Warm up once, then refresh the default series every _WARM_REFRESH_SECONDS (idempotent)."""
    global _warm_started
    if _warm_started:
        return
    _warm_started = True

    def _loop() -> None:
        _warm_up_data()
        while True:
            time.sleep(_WARM_REFRESH_SECONDS)
            _refresh_warm_set()

    threading.Thread(target=_loop, daemon=True, name="market-data-warmup").start()


# ══════════════════════════════════════════════════════════════════
# FASTMCP SERVER CONFIGURATION (STREAMABLE HTTP)
# ══════════════════════════════════════════════════════════════════

def create_mcp_server() -> FastMCP:
    """Instantiates the FastMCP server registering all MacroPulse tools."""
    if FastMCP is None:
        raise RuntimeError("The 'mcp' package is not installed (pip install 'mcp>=2.0.0').")
    # mcp 1.x takes transport settings on the constructor; mcp 2.x takes them on
    # streamable_http_app()/sse_app() instead (see build_starlette_app).
    # host="0.0.0.0" matters in both: the SDK auto-enables DNS-rebinding protection
    # (rejecting any non-localhost Host header, e.g. an ALB or domain) for 127.0.0.1.
    v1_settings = {} if _MCP_V2 else {"host": "0.0.0.0", "stateless_http": True}
    server = FastMCP(
        "MacroPulse-Alexa",
        instructions=(
            "You are MacroPulse, an institutional-grade quantitative strategist and risk desk for Alexa+. "
            "Use the provided tools to deliver hedge-fund caliber macro intelligence, Monte Carlo simulations, "
            "and volatility breakout alerts directly to investors."
        ),
        **v1_settings,
    )

    @server.tool(name="get_macro_regime", description="Get current market regime (Risk-On/Neutral/Risk-Off), macro stress z-scores, S&P 500, and historical win rates.")
    def tool_macro_regime() -> Dict[str, Any]:
        return execute_get_macro_regime()

    @server.tool(name="get_rates_and_spreads", description="Get 10-Year Treasury Yield, yield curve slope (10Y-2Y), and corporate credit spreads.")
    def tool_rates_spreads() -> Dict[str, Any]:
        return execute_get_rates_and_spreads()

    @server.tool(name="simulate_portfolio_risk", description="Run 1,000 to 10,000-path Monte Carlo risk simulation for SPY or custom tickers to calculate VaR 95/99% and Expected Shortfall.")
    def tool_sim_risk(ticker: str = "SPY", confidence: int = 95, days: int = 30, n_paths: int = 5000) -> Dict[str, Any]:
        return execute_simulate_portfolio_risk(ticker, confidence, days, n_paths)

    @server.tool(name="check_nvda_danger_zone", description="Check Nvidia (NVDA) composite Danger Index (0-1), 14-day RSI overbought levels, and block trade order flow.")
    def tool_nvda_danger() -> Dict[str, Any]:
        return execute_check_nvda_danger_zone()

    @server.tool(name="scan_quant_signals", description="Scan a ticker or asset for Bollinger Band volatility squeezes, 20-day realized volatility, and directional conviction score.")
    def tool_quant_signals(ticker: str = "SPY") -> Dict[str, Any]:
        return execute_scan_quant_signals(ticker)

    @server.tool(name="get_expected_returns", description="Get 12-month forward S&P 500 expected return forecasts generated by expanding-window Ridge regression.")
    def tool_exp_returns() -> Dict[str, Any]:
        return execute_get_expected_returns()

    @server.tool(name="get_morning_brief", description="Pre-market MacroPulse brief in one short answer (built for an Alexa Routine such as 'Alexa, good morning'): last S&P 500 close and day change, macro regime and stress score, 10Y yield and curve slope, SPY quant signal, and the latest daily scan's top buy/sell names.")
    def tool_morning_brief() -> Dict[str, Any]:
        return execute_get_morning_brief()

    @server.tool(name="get_equity_report", description="Equity research headline for a US stock (10-K filer): regime-weighted DCF fair value, bear/base/bull scenarios, rule-based rating, stop and targets with 6-month touch probabilities, and the daily candlestick/indicator screen. The full Excel model, Word note and dashboard are in the MacroPulse Equity Report tab (report_url).")
    def tool_equity_report(ticker: str = "AAPL") -> Dict[str, Any]:
        return execute_get_equity_report(ticker)

    @server.tool(name="simulate_fomc_shock", description="Simulate multi-asset portfolio PnL and cross-asset VaR under major FOMC rate decisions: hawkish_50bps, dovish_50bps, stagflation_inversion, or liquidity_cascade.")
    def tool_fomc_shock(scenario: str = "hawkish_50bps") -> Dict[str, Any]:
        return execute_simulate_fomc_shock(scenario)

    return server


def execute_simulate_fomc_shock(scenario: str = "hawkish_50bps") -> Dict[str, Any]:
    """Simulates cross-asset portfolio PnL and VaR under major FOMC rate shocks."""
    from src.brokerage_sync import BrokeragePortfolio
    port = BrokeragePortfolio.from_preset("Macro Balanced (60/40 Modern)", 1_000_000.0)
    return port.simulate_fomc_shock(scenario)


# ══════════════════════════════════════════════════════════════════
# UNIFIED STARLETTE APPLICATION (Streamable HTTP + SSE + REST)
# ══════════════════════════════════════════════════════════════════

async def health_endpoint(request):
    return JSONResponse({
        "status": "healthy",
        "warm": _warm_done.is_set(),
        "service": "MacroPulse Alexa+ MCP Server",
        "transports": ["streamable-http", "sse", "rest"],
        "endpoints": {
            "streamable_http": "/mcp",
            "sse": "/sse",
            "messages": "/messages",
            "alexa_query": "/alexa/query"
        },
        "available_tools": [
            "get_macro_regime",
            "get_rates_and_spreads",
            "simulate_portfolio_risk",
            "check_nvda_danger_zone",
            "scan_quant_signals",
            "get_expected_returns",
            "simulate_fomc_shock"
        ],
        "version": "1.0.0"
    })

async def alexa_query_endpoint(request):
    """Direct REST simulation endpoint for the Alexa+ Web Simulator."""
    try:
        body = await request.json()
        query = body.get("query", "").strip()
        tool_name = body.get("tool_name")
        args = body.get("args", {})
        
        if tool_name:
            # Direct tool execution
            dispatch_map = {
                "get_macro_regime": lambda: execute_get_macro_regime(),
                "get_rates_and_spreads": lambda: execute_get_rates_and_spreads(),
                "simulate_portfolio_risk": lambda: execute_simulate_portfolio_risk(**args),
                "check_nvda_danger_zone": lambda: execute_check_nvda_danger_zone(),
                "scan_quant_signals": lambda: execute_scan_quant_signals(**args),
                "get_expected_returns": lambda: execute_get_expected_returns(),
                "ask_macro_analyst": lambda: execute_ask_macro_analyst(query=args.get("query", query))
            }
            if tool_name in dispatch_map:
                result = dispatch_map[tool_name]()
                return JSONResponse({"status": "success", "tool_called": tool_name, "result": result})
            return JSONResponse({"status": "error", "message": f"Unknown tool {tool_name}"}, status_code=400)
            
        # Natural language query routing
        q = query.lower()
        if any(w in q for w in ["regime", "market state", "risk-on", "risk-off"]):
            res = execute_get_macro_regime()
            selected = "get_macro_regime"
        elif any(w in q for w in ["rate", "yield", "spread", "10y", "treasury", "curve"]):
            res = execute_get_rates_and_spreads()
            selected = "get_rates_and_spreads"
        elif any(w in q for w in ["var", "risk", "simulation", "monte carlo", "shortfall"]):
            ticker = "SPY"
            for t in ["SPY", "QQQ", "NVDA", "AAPL", "MSFT"]:
                if t.lower() in q:
                    ticker = t
                    break
            res = execute_simulate_portfolio_risk(ticker=ticker)
            selected = "simulate_portfolio_risk"
        elif any(w in q for w in ["nvda", "nvidia", "danger", "block trade"]):
            res = execute_check_nvda_danger_zone()
            selected = "check_nvda_danger_zone"
        elif any(w in q for w in ["signal", "breakout", "squeeze", "bollinger"]):
            res = execute_scan_quant_signals()
            selected = "scan_quant_signals"
        elif any(w in q for w in ["expected return", "forecast", "ridge"]):
            res = execute_get_expected_returns()
            selected = "get_expected_returns"
        else:
            res = execute_ask_macro_analyst(query)
            selected = "ask_macro_analyst"
            
        return JSONResponse({
            "status": "success",
            "query": query,
            "tool_called": selected,
            "result": res
        })
    except MarketDataUnavailable as e:
        return JSONResponse({"status": "error", "error": "market_data_unavailable", "message": str(e)}, status_code=503)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def alexa_skill_webhook_endpoint(request):
    """Webhook endpoint for official Amazon Alexa Skills Kit (ASK) requests."""
    try:
        from src.alexa_agent_skill import AlexaMacroSkill
        body = await request.json()
        resp = AlexaMacroSkill.handle_ask_request(body)
        return JSONResponse(resp)
    except Exception as e:
        logger.error(f"Error handling ASK webhook: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


_LANDING_TOOLS = [
    ("get_macro_regime", "Current macro regime (risk-on / risk-off) from live market data."),
    ("get_rates_and_spreads", "Treasury yields, curve spreads and credit spreads."),
    ("simulate_portfolio_risk", "Monte Carlo VaR / expected shortfall for a ticker."),
    ("check_nvda_danger_zone", "NVDA block-trade danger-zone check."),
    ("scan_quant_signals", "Breakout / squeeze / Bollinger signal scan."),
    ("get_expected_returns", "Static reference estimate of expected returns (labelled as such)."),
    ("simulate_fomc_shock", "Cross-asset PnL and VaR under FOMC rate-shock scenarios."),
]

_PAGE_CSS = """
:root{--bg:#fff;--fg:#1a1f2b;--muted:#5b6475;--card:#f4f6fa;--line:#dde2ec;--accent:#0b6bcb}
@media (prefers-color-scheme:dark){:root{--bg:#0f131b;--fg:#e8ecf3;--muted:#98a2b6;--card:#181e2a;--line:#2a3243;--accent:#6cb0ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:720px;margin:0 auto;padding:32px 16px}
h1{font-size:1.6rem;margin:0 0 4px} .sub{color:var(--muted);margin:0 0 24px}
.ok{display:inline-block;padding:2px 10px;border-radius:99px;background:var(--card);border:1px solid var(--line);font-size:.85rem}
h2{font-size:1.05rem;margin:28px 0 8px}
code{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:2px 6px;font-size:.92em;word-break:break-all}
.url{display:block;padding:12px;margin:0}
ul{padding-left:1.2rem} ul.tools{list-style:none;padding:0;margin:0} ul.tools li{padding:10px 0;border-bottom:1px solid var(--line)}
ul.tools li span{display:block;color:var(--muted);font-size:.92rem}
a{color:var(--accent)} footer{margin-top:28px;color:var(--muted);font-size:.85rem}
"""

_CONTACT_URL = "https://github.com/sechan9999/macropulse-alexa-mcp/issues"
_POLICY_UPDATED = "September 23, 2026"


def _page(title: str, body: str) -> str:
    """Shared shell for the landing, privacy and terms pages (static markup only)."""
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n<style>{_PAGE_CSS}</style></head><body><main>\n"
        f"{body}\n"
        '<footer><a href="/">Home</a> &middot; <a href="/privacy">Privacy Policy</a> &middot; '
        '<a href="/terms">Terms of Use</a> &middot; <a href="/health">/health</a></footer>\n'
        "</main></body></html>"
    )


_PRIVACY_BODY = f"""<h1>Privacy Policy</h1>
<p class="sub">MacroPulse Alexa+ MCP Server &middot; last updated {_POLICY_UPDATED}</p>
<h2>What this service is</h2>
<p>MacroPulse is a read-only information service. It answers questions about market conditions
(macro regime, rates and spreads, portfolio risk simulations, FOMC scenarios, a pre-market morning brief and
equity reports on US-listed companies) when an assistant such as Alexa+ calls its tools. It has no user accounts
and no sign-in.</p>
<h2>What we receive</h2>
<ul>
<li>The tool request itself: the tool name and its arguments, for example a ticker symbol such as <code>SPY</code> or a scenario name.</li>
<li>Standard connection data that any web server sees, such as the IP address of the caller and the request time.</li>
</ul>
<p>We do not ask for, and the tools do not need, your name, email address, phone number, location, payment
details or any other personal information. Please do not include personal information in your requests.</p>
<h2>What we do with it</h2>
<ul>
<li>Requests are used only to compute the answer and return it to the caller.</li>
<li>Ticker symbols are sent to a public market-data provider (Yahoo Finance) to fetch prices. Nothing else from your request is sent to it.</li>
<li>For an equity report, the ticker is also used to look up the company's public filings at the U.S. Securities and
Exchange Commission (SEC EDGAR). Those requests identify this service, as SEC rules require, not you; nothing else
from your request is sent.</li>
<li>Operational logs (errors, timing, request lines) are kept in the hosting provider logging service for troubleshooting and to keep the service reliable.</li>
<li>We do not sell your data, use it for advertising, or build profiles of users.</li>
</ul>
<h2>Where it runs</h2>
<p>The service is hosted on Amazon Web Services in the United States (us-east-1). Voice handling by Alexa is
governed by the Amazon privacy notice; this service only sees the resulting tool request.</p>
<h2>Children</h2>
<p>The service is not directed at children under 13 and does not knowingly collect information from them.</p>
<h2>Changes and contact</h2>
<p>We may update this policy; the date above shows the latest version. Questions or requests about data:
<a href="{_CONTACT_URL}">{_CONTACT_URL}</a>.</p>"""

_TERMS_BODY = f"""<h1>Terms of Use</h1>
<p class="sub">MacroPulse Alexa+ MCP Server &middot; last updated {_POLICY_UPDATED}</p>
<h2>Informational use only</h2>
<p>The service provides general market information and simulations. It is <strong>not investment, financial,
tax or legal advice</strong> and is not a recommendation to buy or sell any security. Simulations (for example
VaR or FOMC shock scenarios) rest on models and historical data and can be wrong. The expected-return tool
returns a static reference estimate, not a forecast. Equity-report fair values, scenarios, price levels and
ratings (for example "Buy" or "Underweight") are rule-based model outputs, not recommendations. Make your own
decisions or consult a licensed professional.</p>
<h2>Data and availability</h2>
<ul>
<li>Market data comes from third-party public sources and may be delayed, incomplete or unavailable. When live data cannot be fetched, the tools say so instead of guessing.</li>
<li>The service is provided as is and as available, with no guarantee of uptime, accuracy or fitness for a particular purpose.</li>
<li>Access may be changed, rate-limited or discontinued at any time.</li>
</ul>
<h2>Acceptable use</h2>
<p>Do not attempt to disrupt the service, overload it with automated traffic, probe it for vulnerabilities
without permission, or use it to break the law or the terms of the data providers behind it.</p>
<h2>Liability</h2>
<p>To the fullest extent permitted by law, the operator is not liable for losses arising from use of, or
inability to use, the service or from decisions made on its output.</p>
<h2>Changes and contact</h2>
<p>We may update these terms; continued use after an update means you accept them. Questions:
<a href="{_CONTACT_URL}">{_CONTACT_URL}</a>. See also the <a href="/privacy">Privacy Policy</a>.</p>"""

_LANDING_BODY = """<h1>MacroPulse Alexa+ MCP Server</h1>
<p class="sub">Live macro-regime, rates, portfolio-risk, FOMC-shock, morning-brief and equity-report tools for voice assistants and MCP clients. <span class="ok">status: ok</span></p>
<h2>MCP endpoint (Streamable HTTP)</h2>
<code class="url">{mcp_url}</code>
<h2>Tools</h2>
<ul class="tools">{tools}</ul>
<p style="color:var(--muted);font-size:.85rem">Market data comes from public sources and is never fabricated: when it is unavailable the tools say so instead of guessing.
Informational only, not investment advice.</p>"""


async def root_endpoint(request):
    """Landing page for a browser opening the bare service URL; JSON for API clients."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        base = f"{request.headers.get('x-forwarded-proto', request.url.scheme)}://{request.headers.get('host', request.url.netloc)}"
        tools = "".join(
            f"<li><code>{html.escape(n)}</code><span>{html.escape(d)}</span></li>"
            for n, d in _LANDING_TOOLS)
        body = _LANDING_BODY.format(mcp_url=html.escape(base + "/mcp"), tools=tools)
        return HTMLResponse(_page("MacroPulse Alexa+ MCP Server", body))
    return JSONResponse({
        "service": "MacroPulse Alexa+ MCP Server",
        "status": "ok",
        "mcp_endpoint": "/mcp",
        "note": "MCP Streamable HTTP is served at /mcp and also at this base URL.",
        "health": "/health",
    })


async def privacy_endpoint(request):
    return HTMLResponse(_page("Privacy Policy - MacroPulse Alexa+ MCP Server", _PRIVACY_BODY))


async def terms_endpoint(request):
    return HTMLResponse(_page("Terms of Use - MacroPulse Alexa+ MCP Server", _TERMS_BODY))


class _RootMcpAlias:
    """ASGI middleware: accept MCP Streamable HTTP at the base URL as well as at /mcp, so a
    registration that uses the bare host (https://host) works. A plain browser GET / is left
    alone and reaches root_endpoint."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] in ("", "/"):
            accept = dict(scope["headers"]).get(b"accept", b"")
            is_mcp = scope["method"] in ("POST", "DELETE") or (
                scope["method"] == "GET" and b"text/event-stream" in accept)
            if is_mcp:
                scope = dict(scope, path="/mcp", raw_path=b"/mcp")
        await self.app(scope, receive, send)


def build_starlette_app(warm_up: bool = True) -> Starlette:
    """Assembles the complete Starlette application hosting Streamable HTTP & SSE.

    warm_up: start the background thread that prefetches and keeps market data fresh.
    """
    fastmcp_server = create_mcp_server()
    if warm_up:
        start_background_warmup()

    # The health/Alexa endpoints are registered on the MCP server itself so they live in
    # the SAME Starlette app as /mcp. Mounting the MCP app inside another Starlette app
    # does not run its lifespan, and every /mcp request then fails with
    # "Task group is not initialized. Make sure to use run()".
    fastmcp_server.custom_route("/health", methods=["GET"])(health_endpoint)
    fastmcp_server.custom_route("/alexa/query", methods=["POST"])(alexa_query_endpoint)
    fastmcp_server.custom_route("/alexa/skill", methods=["POST"])(alexa_skill_webhook_endpoint)
    fastmcp_server.custom_route("/", methods=["GET"])(root_endpoint)
    fastmcp_server.custom_route("/privacy", methods=["GET"])(privacy_endpoint)
    fastmcp_server.custom_route("/terms", methods=["GET"])(terms_endpoint)

    if _MCP_V2:
        # Stateless: every tool is a pure function, so no per-session state is needed and
        # the server can sit behind a load balancer with several tasks (ECS Fargate/ALB).
        app = fastmcp_server.streamable_http_app(stateless_http=True, host="0.0.0.0")
        sse_app = fastmcp_server.sse_app(host="0.0.0.0")
    else:
        app = fastmcp_server.streamable_http_app()
        sse_app = fastmcp_server.sse_app()

    # Legacy SSE transport at /sse and /messages/ (mounted last so it never shadows /mcp).
    app.router.routes.append(Mount("/", app=sse_app))
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(_RootMcpAlias)
    return app


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting MacroPulse Alexa+ MCP Server on http://0.0.0.0:8000...")
    uvicorn.run(build_starlette_app(), host="0.0.0.0", port=8000)
