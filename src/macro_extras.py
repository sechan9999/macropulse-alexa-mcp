"""
src/macro_extras.py
─────────────────────────────────────────────────────────────────
MCP-tool wrappers around three app.py analysis engines: Monte Carlo risk
simulation (Tab 7), the walk-forward strategy backtest (Tab 10), and the
NVDA danger-zone monitor (Tab 9).

The Monte Carlo and backtest engines live in src/portfolio.py and are
shared with app.py. fetch_nvda_full is still a copy of the app.py version
(app.py runs Streamlit UI code at import time, so it can't be imported).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from src.macro_data import _ttl_cache, load_macro
from src.portfolio import MC_BOOTSTRAP, MC_GAUSSIAN, run_monte_carlo, run_strategy_backtest  # noqa: F401


# ══════════════════════════════════════════════════════════════════
# Monte Carlo risk simulation (Tab 7)
# ══════════════════════════════════════════════════════════════════
@_ttl_cache(ttl_seconds=3600)
def _cached_monte_carlo(mu: float, vol: float, n: int, h: int, method: str, block: int) -> np.ndarray:
    hist = None
    if method == MC_BOOTSTRAP:
        df = load_macro()
        if bool(df["_is_demo"].iloc[-1]) if "_is_demo" in df else False:
            raise ValueError("historical S&P 500 returns are unavailable; use method='gaussian'")
        hist = df["sp500_ret_m"]
    return run_monte_carlo(mu, vol, n, h, method=method, hist_rets=hist, block=block)


def summarize_monte_carlo(mu_pct: float = 8.0, vol_pct: float = 16.0,
                           n_paths: int = 5000, horizon_months: int = 12,
                           method: str = MC_GAUSSIAN, block_months: int = 3) -> dict:
    """Run the simulation and return the same summary stats Tab 7 shows:
    expected/median return, VaR 95%, CVaR 95% (expected shortfall), P10/P90,
    and the probability of a positive / >10% outcome. method="bootstrap" resamples
    historical S&P 500 monthly returns (in blocks of block_months) and ignores mu/vol."""
    if method not in (MC_GAUSSIAN, MC_BOOTSTRAP):
        return {"error": f"method must be '{MC_GAUSSIAN}' or '{MC_BOOTSTRAP}'"}
    block = int(block_months) if method == MC_BOOTSTRAP else 1
    try:
        paths = _cached_monte_carlo(mu_pct / 100, vol_pct / 100, int(n_paths), int(horizon_months),
                                    method, block)
    except ValueError as e:                     # not enough history for a bootstrap
        return {"error": str(e)}
    final_ret = (paths[:, -1] - 1) * 100
    var95 = float(np.percentile(final_ret, 5))
    cvar95 = float(final_ret[final_ret < var95].mean())
    return {
        "method": method,
        **({"block_months": block} if method == MC_BOOTSTRAP else {"mu_pct": mu_pct, "vol_pct": vol_pct}),
        "n_paths": int(n_paths), "horizon_months": int(horizon_months),
        "expected_return_pct": round(float(final_ret.mean()), 2),
        "median_return_pct": round(float(np.median(final_ret)), 2),
        "var_95_pct": round(var95, 2),
        "cvar_95_pct": round(cvar95, 2),
        "p10_pct": round(float(np.percentile(final_ret, 10)), 2),
        "p90_pct": round(float(np.percentile(final_ret, 90)), 2),
        "prob_positive_pct": round(float((final_ret > 0).mean() * 100), 1),
        "prob_above_10pct_pct": round(float((final_ret > 10).mean() * 100), 1),
    }


# ══════════════════════════════════════════════════════════════════
# Walk-forward strategy backtest (Tab 10)
# ══════════════════════════════════════════════════════════════════
def summarize_backtest(macro_df: pd.DataFrame, compute_hf_metrics, *,
                        use_regime: bool = True, use_momentum: bool = True, use_trend: bool = True,
                        threshold: float = 0.5, cost_bps: float = 5.0, allow_short: bool = False) -> dict:
    """Run the backtest and return the same strategy-vs-buy-hold tear sheet
    Tab 10 shows. compute_hf_metrics is passed in (from src.macro_data) to
    avoid a circular import."""
    bt = run_strategy_backtest(macro_df, use_regime=use_regime, use_momentum=use_momentum,
                                use_trend=use_trend, threshold=threshold, cost_bps=cost_bps,
                                allow_short=allow_short)
    if bt["strat_ret"].dropna().empty:
        return {"error": "No data after filters — widen the date range or relax the threshold."}

    m_strat = compute_hf_metrics(bt["strat_ret"].dropna(), bt["sp500_ret_m"].dropna(), rf=bt["rf_m"])
    m_bh = compute_hf_metrics(bt["sp500_ret_m"].dropna(), rf=bt["rf_m"])
    time_in_market_pct = float(bt["position"].abs().mean() * 100)
    n_position_flips = int((bt["position"].diff().abs() > 1e-9).sum())

    def _r(x):
        return None if x is None or x != x else round(float(x), 4)

    return {
        "config": {"use_regime": use_regime, "use_momentum": use_momentum, "use_trend": use_trend,
                   "threshold": threshold, "cost_bps": cost_bps, "allow_short": allow_short},
        "strategy": {
            "annualized_return_pct": _r(m_strat["ann_ret"] * 100),
            "sharpe": _r(m_strat["sharpe"]),
            "sortino": _r(m_strat["sortino"]),
            "max_drawdown_pct": _r(m_strat["mdd"] * 100),
            "calmar": _r(m_strat["calmar"]),
        },
        "buy_and_hold": {
            "annualized_return_pct": _r(m_bh["ann_ret"] * 100),
            "sharpe": _r(m_bh["sharpe"]),
            "sortino": _r(m_bh["sortino"]),
            "max_drawdown_pct": _r(m_bh["mdd"] * 100),
            "calmar": _r(m_bh["calmar"]),
        },
        "cash_rate": bt.attrs.get("cash_rate"),
        "time_in_market_pct": round(time_in_market_pct, 1),
        "n_position_flips": n_position_flips,
    }


# ══════════════════════════════════════════════════════════════════
# NVDA danger zone (Tab 9)
# ══════════════════════════════════════════════════════════════════
def _norm(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)


@_ttl_cache(ttl_seconds=300)
def fetch_nvda_full(period_days: int = 365):
    """Fetch NVDA + SOX + AI peers; compute danger indicators from OHLCV."""
    nvda = yf.Ticker("NVDA")
    df_nvda = nvda.history(period=f"{period_days}d", auto_adjust=True)
    if df_nvda.empty:
        return {}, {}

    df = df_nvda.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["EMA12"] = df["Close"].ewm(span=12).mean()
    df["EMA26"] = df["Close"].ewm(span=26).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal_line"] = df["MACD"].ewm(span=9).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()
    df["ATR_pct"] = df["ATR14"] / df["Close"] * 100

    df["vol_sma20"] = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = df["Volume"] / (df["vol_sma20"] + 1)
    df["block_flag"] = df["vol_ratio"] > 2.0
    df["block_volume"] = np.where(df["block_flag"], df["Volume"], np.nan)

    df["mid"] = (df["High"] + df["Low"]) / 2
    df["ba_imbalance"] = (df["Close"] - df["mid"]) / (df["High"] - df["Low"] + 1e-3) * 100

    vix_raw = yf.download("^VIX", period=f"{period_days}d", auto_adjust=True, progress=False,
                           multi_level_index=False)["Close"]
    if isinstance(vix_raw, pd.DataFrame):
        vix_raw = vix_raw.iloc[:, 0]
    vix = vix_raw.tz_localize(None) if hasattr(vix_raw.index, "tz") and vix_raw.index.tz is not None else vix_raw
    vix.name = "VIX"
    df = df.join(vix.rename("VIX"), how="left")
    df["VIX"] = df["VIX"].ffill()
    df["vol_spike"] = df["ATR_pct"] > df["ATR_pct"].rolling(30).mean() + df["ATR_pct"].rolling(30).std()

    rsi_danger = _norm(df["RSI"].clip(50, 90))
    atr_danger = _norm(df["ATR_pct"])
    vol_danger = _norm(df["vol_ratio"].clip(0, 5))
    vix_danger = _norm(df["VIX"].clip(10, 60))
    close_vs_sma = np.where(df["Close"] > df["SMA50"], (df["Close"] / df["SMA50"] - 1).clip(0, 0.3), 0)
    ext_danger = _norm(pd.Series(close_vs_sma, index=df.index))

    df["danger_index"] = (
        0.30 * rsi_danger + 0.25 * atr_danger + 0.20 * vol_danger + 0.15 * vix_danger + 0.10 * ext_danger
    ).rolling(3).mean()

    df["danger_label"] = pd.cut(
        df["danger_index"], bins=[0, 0.33, 0.60, 1.01],
        labels=["Safe Zone 🟢", "Caution ⚠️", "Danger Zone 🔴"]
    ).astype(str)

    return df, fetch_peer_context(period_days)


NVDA_PEERS = {"NVDA": "NVDA", "SOX": "SOXX", "AMD": "AMD", "TSM": "TSM", "AVGO": "AVGO", "MU": "MU"}
MAX_PLAUSIBLE_DAILY_MOVE = 0.40      # a large cap moving >40% in one session is almost always bad data


def clean_peer_series(close: pd.Series, max_daily_move: float = MAX_PLAUSIBLE_DAILY_MOVE) -> bool:
    """True when a close series looks like one ticker's real history (no single-day jump beyond
    `max_daily_move`). A spliced or mixed-up series (another ticker's prices, unadjusted splits)
    shows up as such a jump and would turn into absurd 1-year returns like +1,400%."""
    r = close.dropna().pct_change().dropna()
    return bool(len(r)) and float(r.abs().max()) <= max_daily_move


def fetch_peer_context(period_days: int = 365, peers: dict | None = None) -> pd.DataFrame:
    """NVDA and AI peers indexed to 100 at the first common date.

    Each ticker is fetched with its own yf.Ticker(...).history() — yf.download() keeps results in a
    module-level dict shared by every thread, so concurrent Streamlit sessions can receive another
    ticker's prices. Series that fail clean_peer_series() are left out; their names are listed in
    df.attrs["excluded"] so the UI can say so instead of showing a made-up return."""
    frames, excluded = {}, []
    for name, tkr in (peers or NVDA_PEERS).items():
        try:
            raw = yf.Ticker(tkr).history(period=f"{period_days}d", auto_adjust=True)["Close"]
            raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
        except Exception:
            excluded.append(name)
            continue
        if clean_peer_series(raw):
            frames[name] = raw
        else:
            excluded.append(name)
    df_ctx = pd.DataFrame(frames).ffill().dropna()
    df_ctx = df_ctx / df_ctx.iloc[0] * 100 if not df_ctx.empty else df_ctx
    df_ctx.attrs["excluded"] = excluded
    return df_ctx


def summarize_nvda_danger_zone(period_days: int = 365) -> dict:
    """Latest NVDA danger-zone reading plus the components behind it, and
    each AI-peer's cumulative return over the same window (indexed to 100)."""
    df, df_ctx = fetch_nvda_full(period_days)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"error": "no_data"}

    last = df.dropna(subset=["danger_index"]).iloc[-1] if df["danger_index"].notna().any() else df.iloc[-1]

    def _r(x):
        return None if x is None or x != x else round(float(x), 2)

    peer_returns = {}
    if isinstance(df_ctx, pd.DataFrame) and not df_ctx.empty:
        last_ctx = df_ctx.iloc[-1]
        peer_returns = {col: _r(last_ctx[col] - 100) for col in df_ctx.columns}

    return {
        "as_of": df.index[-1].strftime("%Y-%m-%d"),
        "price": _r(last["Close"]),
        "danger_index": _r(last.get("danger_index")),
        "danger_label": last.get("danger_label"),
        "rsi": _r(last.get("RSI")),
        "atr_pct": _r(last.get("ATR_pct")),
        "volume_ratio_vs_20d_avg": _r(last.get("vol_ratio")),
        "block_trade_flag": bool(last.get("block_flag")) if last.get("block_flag") == last.get("block_flag") else None,
        "vix": _r(last.get("VIX")),
        "peer_returns_pct_over_window": peer_returns,
    }
