"""
src/macro_extras.py
─────────────────────────────────────────────────────────────────
Streamlit-free copies of three more app.py analysis engines, so they
can be exposed as MCP tools: Monte Carlo risk simulation (Tab 7),
the walk-forward strategy backtest (Tab 10), and the NVDA danger-zone
monitor (Tab 9).

Same rationale as src/macro_data.py / src/macro_briefing.py: app.py
runs Streamlit UI code at import time, so these are duplicated here
rather than imported. run_strategy_backtest and fetch_nvda_full were
already pure functions in app.py (no st. calls inside the function
body) — copied verbatim aside from dropping the @st.cache_data
decorator. The Monte Carlo logic was inline inside `with tab7:` in
app.py; it's extracted into run_monte_carlo()/summarize_monte_carlo()
here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from src.macro_data import _ttl_cache


# ══════════════════════════════════════════════════════════════════
# Monte Carlo risk simulation (Tab 7)
# ══════════════════════════════════════════════════════════════════
@_ttl_cache(ttl_seconds=3600)
def run_monte_carlo(mu: float, vol: float, n: int, h: int = 12) -> np.ndarray:
    """mu/vol are annual, decimal (e.g. 0.08, 0.16). Returns an (n, h) array
    of cumulative return multipliers, one row per simulated path."""
    rng = np.random.default_rng(99)
    shocks = rng.normal(mu / 12, vol / np.sqrt(12), (n, h))
    return np.exp(np.cumsum(shocks, axis=1))


def summarize_monte_carlo(mu_pct: float = 8.0, vol_pct: float = 16.0,
                           n_paths: int = 5000, horizon_months: int = 12) -> dict:
    """Run the simulation and return the same summary stats Tab 7 shows:
    expected/median return, VaR 95%, CVaR 95% (expected shortfall), P10/P90,
    and the probability of a positive / >10% outcome."""
    paths = run_monte_carlo(mu_pct / 100, vol_pct / 100, int(n_paths), int(horizon_months))
    final_ret = (paths[:, -1] - 1) * 100
    var95 = float(np.percentile(final_ret, 5))
    cvar95 = float(final_ret[final_ret < var95].mean())
    return {
        "mu_pct": mu_pct, "vol_pct": vol_pct, "n_paths": int(n_paths), "horizon_months": int(horizon_months),
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
def run_strategy_backtest(macro_df: pd.DataFrame, *, use_regime: bool, use_momentum: bool,
                           use_trend: bool, threshold: float, cost_bps: float,
                           allow_short: bool) -> pd.DataFrame:
    """Walk-forward, no-lookahead backtest of an SPY/cash (or SPY/-SPY) strategy.

    Position is decided from data observable at month T, then applied to the
    realised return from T to T+1 via .shift(1). Net returns include linear
    transaction costs proportional to turnover."""
    out = macro_df.copy()
    out = out.dropna(subset=["sp500_ret_m"])

    out["sig_regime"] = (out["regime"] != "Risk-Off 🔴").astype(int)
    out["sig_momentum"] = (out["momentum_12_1"] > 0).astype(int)
    sma_10m = out["sp500"].rolling(10).mean()
    out["sig_trend"] = (out["sp500"] > sma_10m).astype(int)

    cols = []
    if use_regime:
        cols.append("sig_regime")
    if use_momentum:
        cols.append("sig_momentum")
    if use_trend:
        cols.append("sig_trend")

    if not cols:
        out["sig_score"] = 1.0
    else:
        out["sig_score"] = out[cols].mean(axis=1)

    in_market = (out["sig_score"] >= threshold).astype(float)
    if allow_short:
        out_market = -1.0 * (out["sig_score"] < (1 - threshold)).astype(float)
        target_w = in_market + out_market
    else:
        target_w = in_market

    out["target_w"] = target_w
    out["position"] = out["target_w"].shift(1).fillna(0.0)  # NO-LOOKAHEAD
    out["turnover"] = out["position"].diff().abs().fillna(out["position"].abs())

    cost = (cost_bps / 10000.0) * out["turnover"]
    out["strat_ret_gross"] = out["position"] * out["sp500_ret_m"]
    out["strat_ret"] = out["strat_ret_gross"] - cost
    out["strat_equity"] = np.exp(out["strat_ret"].fillna(0.0).cumsum()) * 100
    out["bh_equity"] = np.exp(out["sp500_ret_m"].fillna(0.0).cumsum()) * 100
    out["strat_dd"] = out["strat_equity"] / out["strat_equity"].cummax() - 1
    out["bh_dd"] = out["bh_equity"] / out["bh_equity"].cummax() - 1
    return out


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

    m_strat = compute_hf_metrics(bt["strat_ret"].dropna(), bt["sp500_ret_m"].dropna())
    m_bh = compute_hf_metrics(bt["sp500_ret_m"].dropna())
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
            "max_drawdown_pct": _r(m_strat["mdd"] * 100),
            "calmar": _r(m_strat["calmar"]),
        },
        "buy_and_hold": {
            "annualized_return_pct": _r(m_bh["ann_ret"] * 100),
            "sharpe": _r(m_bh["sharpe"]),
            "max_drawdown_pct": _r(m_bh["mdd"] * 100),
            "calmar": _r(m_bh["calmar"]),
        },
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
