"""Candlestick patterns + indicator events -> rule-based screening score (stage 1 of the analysis).

Output keys follow the stock-report-skills convention:
bullish_count, bearish_count, bull_score, bear_score, overall.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind

# key -> (display name, direction, strength 1..3)
PATTERNS = {
    "hammer": ("Hammer", "bull", 2), "hanging_man": ("Hanging Man", "bear", 2),
    "inverted_hammer": ("Inverted Hammer", "bull", 1), "shooting_star": ("Shooting Star", "bear", 2),
    "bullish_engulfing": ("Bullish Engulfing", "bull", 2), "bearish_engulfing": ("Bearish Engulfing", "bear", 2),
    "piercing": ("Piercing Line", "bull", 2), "dark_cloud": ("Dark Cloud Cover", "bear", 2),
    "morning_star": ("Morning Star", "bull", 3), "evening_star": ("Evening Star", "bear", 3),
    "three_white_soldiers": ("Three White Soldiers", "bull", 3), "three_black_crows": ("Three Black Crows", "bear", 3),
    "bullish_harami": ("Bullish Harami", "bull", 1), "bearish_harami": ("Bearish Harami", "bear", 1),
    "bull_marubozu": ("Bullish Marubozu", "bull", 1), "bear_marubozu": ("Bearish Marubozu", "bear", 1),
    "doji": ("Doji", "neutral", 1),
}
COLS = ["date", "pattern", "name", "direction", "strength", "price"]


def detect(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = (df[k] for k in ("open", "high", "low", "close"))
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    bull, bear = c > o, c < o
    avg_body = body.rolling(20, min_periods=5).mean()
    long_body = body > avg_body * 1.2
    sma = c.rolling(10, min_periods=5).mean()
    down = (c.shift() < sma.shift()) & (sma.shift() < sma.shift(5))    # prior downtrend
    up = (c.shift() > sma.shift()) & (sma.shift() > sma.shift(5))      # prior uptrend
    po, pc, pbody = o.shift(), c.shift(), body.shift()
    ppo, ppc, ppbody = o.shift(2), c.shift(2), body.shift(2)
    small = body <= rng * 0.3
    hammer_shape = (lower >= 2 * body) & (upper <= body.clip(lower=rng * 0.1)) & small
    inv_shape = (upper >= 2 * body) & (lower <= body.clip(lower=rng * 0.1)) & small
    m = {
        "hammer": hammer_shape & down,
        "hanging_man": hammer_shape & up,
        "inverted_hammer": inv_shape & down,
        "shooting_star": inv_shape & up,
        "bullish_engulfing": bull & (pc < po) & (c >= po) & (o <= pc) & (body > pbody) & down,
        "bearish_engulfing": bear & (pc > po) & (c <= po) & (o >= pc) & (body > pbody) & up,
        "piercing": bull & (pc < po) & (pbody > avg_body) & (o < pc) & (c > (po + pc) / 2) & (c < po) & down,
        "dark_cloud": bear & (pc > po) & (pbody > avg_body) & (o > pc) & (c < (po + pc) / 2) & (c > po) & up,
        "morning_star": (ppc < ppo) & (ppbody > avg_body) & (pbody < ppbody * 0.4) & bull
                        & (c > (ppo + ppc) / 2) & down.shift(fill_value=False),
        "evening_star": (ppc > ppo) & (ppbody > avg_body) & (pbody < ppbody * 0.4) & bear
                        & (c < (ppo + ppc) / 2) & up.shift(fill_value=False),
        "three_white_soldiers": bull & (pc > po) & (ppc > ppo) & (c > pc) & (pc > ppc)
                                & (o > po) & (po > ppo) & long_body & (upper < body * 0.3),
        "three_black_crows": bear & (pc < po) & (ppc < ppo) & (c < pc) & (pc < ppc)
                             & (o < po) & (po < ppo) & long_body & (lower < body * 0.3),
        "bullish_harami": bull & (pc < po) & (pbody > avg_body) & (o > pc) & (c < po) & (body < pbody * 0.6) & down,
        "bearish_harami": bear & (pc > po) & (pbody > avg_body) & (o < pc) & (c > po) & (body < pbody * 0.6) & up,
        "bull_marubozu": bull & (body >= rng * 0.9) & long_body,
        "bear_marubozu": bear & (body >= rng * 0.9) & long_body,
        "doji": body <= rng * 0.08,
    }
    rows = []
    for key, mask in m.items():
        for d in mask[mask.fillna(False).astype(bool)].index:
            nm, direction, strength = PATTERNS[key]
            rows.append({"date": d, "pattern": key, "name": nm, "direction": direction,
                         "strength": strength, "price": float(c.loc[d])})
    return pd.DataFrame(rows, columns=COLS).sort_values("date").reset_index(drop=True)


def indicator_signals(d: pd.DataFrame) -> pd.DataFrame:
    ev = []

    def add(series, bull_name, bear_name, strength):
        for dt, v in series[series != 0].items():
            ev.append({"date": dt, "pattern": bull_name if v > 0 else bear_name,
                       "name": bull_name if v > 0 else bear_name, "direction": "bull" if v > 0 else "bear",
                       "strength": strength, "price": float(d.loc[dt, "close"])})

    add(ind.cross_events(d["macd"], d["signal"]), "MACD golden cross", "MACD death cross", 2)
    kd = ind.cross_events(d["k"], d["d"])
    kd = kd.where(((kd > 0) & (d["k"] < 30)) | ((kd < 0) & (d["k"] > 70)), 0)
    add(kd, "KD oversold golden cross", "KD overbought death cross", 2)
    if "ma50" in d and "ma200" in d:
        add(ind.cross_events(d["ma50"], d["ma200"]), "50/200 golden cross", "50/200 death cross", 3)
    elif "ma10" in d and "ma20" in d:
        add(ind.cross_events(d["ma10"], d["ma20"]), "10/20 golden cross", "10/20 death cross", 3)
    bb = pd.Series(0, index=d.index)
    bb[(d["close"] < d["bb_lo"]) & (d["close"].shift() >= d["bb_lo"].shift())] = 1
    bb[(d["close"] > d["bb_up"]) & (d["close"].shift() <= d["bb_up"].shift())] = -1
    add(bb, "Close below lower Bollinger (rebound watch)", "Close above upper Bollinger (stretched)", 1)
    return pd.DataFrame(ev, columns=COLS)


def screen(d: pd.DataFrame, recent: int = 20) -> dict:
    candles, sigs = detect(d), indicator_signals(d)
    allsig = pd.concat([candles, sigs], ignore_index=True)
    cutoff = d.index[-min(recent, len(d))]
    rec = allsig[allsig["date"] >= cutoff].sort_values("date")
    bull, bear = rec[rec.direction == "bull"], rec[rec.direction == "bear"]
    bs, brs = int(bull["strength"].sum()), int(bear["strength"].sum())
    ratio = bs / (bs + brs) if bs + brs else 0.5
    return {
        "window_bars": int(recent),
        "bullish_count": int(len(bull)), "bearish_count": int(len(bear)),
        "bull_score": bs, "bear_score": brs, "bull_ratio": round(ratio, 3),
        "overall": "bullish" if ratio >= 0.6 else "bearish" if ratio <= 0.4 else "neutral",
        "recent_signals": [{**r, "date": str(pd.Timestamp(r["date"]).date())} for r in rec.tail(12).to_dict("records")],
        "_all_candles": candles, "_all_signals": sigs,
    }
