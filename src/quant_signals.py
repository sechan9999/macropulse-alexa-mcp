"""
src/quant_signals.py
─────────────────────────────────────────────────────────────────
Volatility-aware technical signal engine for Macro Pulse ("Macropulse").

Pure pandas/numpy/yfinance — no Streamlit import — so the exact same
functions power both:
  1) the "🎯 Quant Signals" dashboard tab, and
  2) scripts/daily_signal_alert.py, a headless script for a scheduled
     pre-market push notification.

Scope (by explicit design): this module ONLY produces a signal label,
a numeric score, and human-readable reasoning per ticker. It never
places, sizes, or manages an order — execution stays manual.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

# Same 19-name list Screener defaults to — single source of truth so both
# tabs (and the alert script) stay in sync unless the user overrides it.
DEFAULT_UNIVERSE = (
    "SPY", "QQQ", "IWM", "GLD", "TLT", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "JPM", "BAC", "XOM", "CVX", "JNJ", "UNH", "WMT", "HD", "PG",
)


# ══════════════════════════════════════════════════════════════════
# 1) Indicators — trend, momentum, and volatility columns on one frame
# ══════════════════════════════════════════════════════════════════
def add_indicators(raw: pd.DataFrame) -> pd.DataFrame:
    """Given a daily OHLCV frame, return a copy with every column the
    scorer needs. Volatility columns (ATR%, realized vol, BB width +
    its 1y percentile rank) are the part that's new vs. the existing
    Technical tab — everything else mirrors fetch_stock() in app.py
    so results are directly comparable across tabs."""
    df = raw.copy()
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    # ── Trend ──
    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()

    # ── Momentum ──
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    # ── Bollinger Bands (20, 2σ) + width, the base for "squeeze" detection ──
    std20 = close.rolling(20).std()
    df["BB_mid"] = df["SMA20"]
    df["BB_upper"] = df["BB_mid"] + 2 * std20
    df["BB_lower"] = df["BB_mid"] - 2 * std20
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]
    # Where does *today's* width rank against its own trailing year?
    # Low percentile = bands squeezed tight = energy coiling for a move.
    df["BB_width_pct"] = df["BB_width"].rolling(252, min_periods=60).apply(
        lambda s: float((s <= s.iloc[-1]).mean()), raw=False
    )

    # ── Volatility proper: ATR% and annualized realized vol ──
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    df["ATR14"] = true_range.rolling(14).mean()
    df["ATR_pct"] = df["ATR14"] / close  # % of price — comparable across tickers

    log_ret = np.log(close / close.shift(1))
    df["RVOL_20d"] = log_ret.rolling(20).std() * np.sqrt(252)
    df["RVOL_pct"] = df["RVOL_20d"].rolling(252, min_periods=60).apply(
        lambda s: float((s <= s.iloc[-1]).mean()), raw=False
    )

    # ── Volume confirmation ──
    df["VOL_avg20"] = vol.rolling(20).mean()
    df["VOL_ratio"] = vol / df["VOL_avg20"]

    return df


# ══════════════════════════════════════════════════════════════════
# 2) Volatility regime + breakout detector
# ══════════════════════════════════════════════════════════════════
def classify_vol_regime(rvol_pct: float) -> str:
    """Bucket today's realized vol by where it sits in its own 1y history —
    a fixed 20%-annualized cutoff means something different for SPY than
    for a biotech, so we rank each ticker against itself instead."""
    if pd.isna(rvol_pct):
        return "Unknown"
    if rvol_pct >= 0.85:
        return "Extreme"
    if rvol_pct >= 0.60:
        return "Elevated"
    if rvol_pct <= 0.20:
        return "Low (squeeze)"
    return "Normal"


def detect_vol_breakout(df: pd.DataFrame) -> tuple[bool, str]:
    """'Catching volatility' = trading the expansion right after a squeeze,
    not the squeeze itself. True only when (a) bands were tight ~2 weeks
    ago, (b) width is now clearly expanding, and (c) price has actually
    closed outside the prior band — direction comes from which side."""
    if len(df) < 15 or pd.isna(df["BB_width_pct"].iloc[-11]):
        return False, "—"
    was_squeeze = df["BB_width_pct"].iloc[-11] <= 0.20
    expanding = df["BB_width"].iloc[-1] > df["BB_width"].iloc[-6] * 1.15
    if not (was_squeeze and expanding):
        return False, "—"
    last = df.iloc[-1]
    if last["Close"] > last["BB_upper"]:
        return True, "up"
    if last["Close"] < last["BB_lower"]:
        return True, "down"
    return False, "—"


# ══════════════════════════════════════════════════════════════════
# 3) Scoring — transparent, additive, every point traces to a reason
# ══════════════════════════════════════════════════════════════════
@dataclass
class Signal:
    ticker: str
    price: float
    signal: str          # STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL
    score: int            # -100..+100
    vol_regime: str
    breakout: str          # "up" / "down" / "—"
    reasons: list[str] = field(default_factory=list)
    error: str | None = None


def generate_signal(ticker: str, df: pd.DataFrame) -> Signal:
    if df is None or df.empty or len(df) < 60:
        return Signal(ticker, float("nan"), "HOLD", 0, "Unknown", "—",
                       ["데이터 부족 (60거래일 미만)"], error="insufficient_data")

    last, prev = df.iloc[-1], df.iloc[-2]
    score = 0
    reasons: list[str] = []

    # ── Trend alignment (± 25) ──
    price, sma20, sma50, sma200 = last["Close"], last["SMA20"], last["SMA50"], last["SMA200"]
    if pd.notna(sma200) and price > sma20 > sma50 > sma200:
        score += 25; reasons.append("정배열 상승추세 (가격>20>50>200일선)")
    elif pd.notna(sma200) and price < sma20 < sma50 < sma200:
        score -= 25; reasons.append("역배열 하락추세 (가격<20<50<200일선)")
    elif price > sma50:
        score += 10; reasons.append("50일선 위에서 거래 중 (단기 우위)")
    elif price < sma50:
        score -= 10; reasons.append("50일선 아래에서 거래 중 (단기 열위)")

    # ── MACD momentum (± 15) ──
    hist, prev_hist = last["MACD_hist"], prev["MACD_hist"]
    if hist > 0 and hist > prev_hist:
        score += 15; reasons.append("MACD 히스토그램 확대 (상승 모멘텀 가속)")
    elif hist < 0 and hist < prev_hist:
        score -= 15; reasons.append("MACD 히스토그램 확대 (하락 모멘텀 가속)")

    # ── RSI mean-reversion (± 15) ──
    rsi = last["RSI"]
    if pd.notna(rsi):
        if rsi < 30:
            score += 15; reasons.append(f"RSI {rsi:.0f} 과매도 구간")
        elif rsi > 70:
            score -= 15; reasons.append(f"RSI {rsi:.0f} 과매수 구간")

    # ── Volatility breakout — the "변동성 포착" core signal (± 20) ──
    is_breakout, direction = detect_vol_breakout(df)
    if is_breakout and direction == "up":
        score += 20; reasons.append("변동성 스퀴즈 이후 상단 돌파 (매수 방향 변동성 확장)")
    elif is_breakout and direction == "down":
        score -= 20; reasons.append("변동성 스퀴즈 이후 하단 이탈 (매도 방향 변동성 확장)")

    # ── Volume confirmation (± 10, only when it agrees with the day's move) ──
    vol_ratio = last["VOL_ratio"]
    day_up = last["Close"] >= prev["Close"]
    if pd.notna(vol_ratio) and vol_ratio > 1.5:
        if day_up:
            score += 10; reasons.append(f"평균 대비 {vol_ratio:.1f}배 거래량 동반 상승")
        else:
            score -= 10; reasons.append(f"평균 대비 {vol_ratio:.1f}배 거래량 동반 하락")

    # ── Extreme, non-directional volatility dampens conviction ──
    vol_regime = classify_vol_regime(last["RVOL_pct"])
    if vol_regime == "Extreme" and not is_breakout:
        score = int(score * 0.5)
        reasons.append("변동성 극단적으로 높음 — 방향성 신호 아니면 확신도 하향 조정")

    score = int(np.clip(score, -100, 100))
    if score >= 40:
        label = "STRONG_BUY"
    elif score >= 15:
        label = "BUY"
    elif score <= -40:
        label = "STRONG_SELL"
    elif score <= -15:
        label = "SELL"
    else:
        label = "HOLD"

    if not reasons:
        reasons.append("뚜렷한 방향성 신호 없음 (중립)")

    return Signal(ticker, float(price), label, score, vol_regime, direction, reasons)


# ══════════════════════════════════════════════════════════════════
# 4) Batch scan — parallel fetch + score, mirrors run_screener() in app.py
# ══════════════════════════════════════════════════════════════════
def _scan_one(ticker: str, period: str) -> Signal:
    try:
        raw = yf.Ticker(ticker).history(period=period, auto_adjust=True, timeout=20)
        if raw is None or raw.empty:
            return Signal(ticker, float("nan"), "HOLD", 0, "Unknown", "—",
                          ["데이터 없음"], error="no_data")
        df = add_indicators(raw)
        return generate_signal(ticker, df)
    except Exception as e:
        return Signal(ticker, float("nan"), "HOLD", 0, "Unknown", "—",
                      [f"오류: {e}"], error=str(e))


def run_quant_scan(tickers: tuple[str, ...] = DEFAULT_UNIVERSE,
                    period: str = "1y", max_workers: int = 8) -> pd.DataFrame:
    """Fetch + score every ticker in parallel. Returns one row per ticker,
    ranked by |score| so the highest-conviction names surface first."""
    rows: list[Signal] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_scan_one, t, period): t for t in tickers}
        for f in as_completed(futs):
            rows.append(f.result())

    out = pd.DataFrame([{
        "Ticker": r.ticker, "Price": r.price, "Signal": r.signal, "Score": r.score,
        "Vol Regime": r.vol_regime, "Vol Breakout": r.breakout,
        "Reasons": " · ".join(r.reasons), "_error": r.error,
    } for r in rows])
    if out.empty:
        return out
    signal_order = {"STRONG_BUY": 0, "BUY": 1, "HOLD": 2, "SELL": 3, "STRONG_SELL": 4}
    out["_ord"] = out["Signal"].map(signal_order).fillna(2)
    out = out.sort_values(["_ord", "Score"], ascending=[True, False]).drop(columns="_ord")
    return out.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
# 5) Plain-text summary — used by scripts/daily_signal_alert.py
# ══════════════════════════════════════════════════════════════════
def format_alert_text(scan_df: pd.DataFrame, top_n: int = 5) -> str:
    if scan_df is None or scan_df.empty:
        return "⚠️ Macropulse Quant Signals: no data returned."

    ok = scan_df[scan_df["_error"].isna()] if "_error" in scan_df else scan_df
    buys = ok[ok["Signal"].isin(["STRONG_BUY", "BUY"])].head(top_n)
    sells = ok[ok["Signal"].isin(["STRONG_SELL", "SELL"])].head(top_n)

    lines = ["⚡ Macropulse Quant Signals — Daily Recommendation (참고용, 매매 지시 아님)", ""]
    lines.append(f"🟢 BUY 후보 ({len(buys)})")
    for _, r in buys.iterrows():
        lines.append(f"  · {r['Ticker']:<6} ${r['Price']:.2f}  score {r['Score']:+d}  [{r['Vol Regime']}]")
        lines.append(f"      {r['Reasons']}")
    lines.append("")
    lines.append(f"🔴 SELL 후보 ({len(sells)})")
    for _, r in sells.iterrows():
        lines.append(f"  · {r['Ticker']:<6} ${r['Price']:.2f}  score {r['Score']:+d}  [{r['Vol Regime']}]")
        lines.append(f"      {r['Reasons']}")
    lines.append("")
    lines.append("※ 추천과 근거만 제공하며, 실제 주문은 본인이 직접 판단·실행하세요.")
    return "\n".join(lines)
