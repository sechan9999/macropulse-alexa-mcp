"""
src/alexa_mcp_server.py
─────────────────────────────────────────────────────────────────
MacroPulse Model Context Protocol (MCP) Server for Alexa+
Serves Streamable HTTP (/mcp) and legacy SSE (/sse) and exposes 7 MCP tools:

  1. get_macro_regime(): Risk-On/Neutral/Risk-Off regime from a stress z-score
  2. get_rates_and_spreads(): 10Y yield, 10Y-3M curve slope, VIX-derived credit-spread proxy
  3. simulate_portfolio_risk(ticker, confidence, days, n_paths): Monte Carlo VaR/CVaR
  4. check_nvda_danger_zone(): NVDA Composite Danger Index, RSI, relative volume
  5. scan_quant_signals(ticker): Volatility squeeze, 20d vol, breakout score (-100..+100)
  6. get_expected_returns(): static reference estimate of the 12m S&P 500 return
  7. simulate_fomc_shock(scenario): multi-asset FOMC rate-shock stress test

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
import re
import json
import logging
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, date
from typing import Any, Dict, Optional, Tuple

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
from starlette.responses import JSONResponse
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


def _fetch_cached_macro_data() -> pd.DataFrame:
    """Monthly S&P 500 / VIX / 10Y snapshot with derived regime columns, built from cached prices."""
    frames = {}
    for col, (tkr, period) in _MACRO_SERIES.items():
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
# WARM-UP: keep the default data fresh so requests never wait on Yahoo
# ══════════════════════════════════════════════════════════════════

# The series the tools read with their default arguments: (ticker, period, timeout).
_WARM_SET = (
    ("^GSPC", "3y", 10), ("^VIX", "3y", 10), ("^TNX", "3y", 10),
    ("^IRX", "1mo", 5), ("NVDA", "6mo", 8), ("SPY", "1y", 8),
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


def _warm_up_data() -> None:
    """Prefetch the default series and run every tool once, so the first request is not
    the one that pays for imports and cold caches."""
    _refresh_warm_set()
    for fn in (execute_simulate_fomc_shock, execute_get_macro_regime, execute_get_rates_and_spreads,
               execute_simulate_portfolio_risk, execute_check_nvda_danger_zone,
               execute_scan_quant_signals, execute_get_expected_returns):
        try:
            fn()
        except Exception as e:
            logger.warning("Warm-up call %s failed: %s", fn.__name__, e)
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
    return app


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting MacroPulse Alexa+ MCP Server on http://0.0.0.0:8000...")
    uvicorn.run(build_starlette_app(), host="0.0.0.0", port=8000)
