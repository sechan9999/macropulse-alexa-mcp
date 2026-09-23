"""Technical indicators in plain pandas (no TA-Lib): MACD, slow stochastic (KD), RSI,
Bollinger, ADX, OBV, Chaikin Money Flow, support/resistance and multi-timeframe snapshots."""
from __future__ import annotations

import numpy as np
import pandas as pd

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
TIMEFRAMES = ("Monthly", "Weekly", "Daily")
WINDOW = {"Daily": 126, "Weekly": 52, "Monthly": 36}      # bars shown: 6 months / 12 months / 3 years


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Daily -> weekly ('W-FRI') or monthly ('ME') OHLCV.

    Each bar is labelled with its LAST ACTUAL trading day, not the period end pandas would use:
    otherwise the unfinished current month shows up as e.g. 2026-09-30 on 2026-09-23, and a cross
    "on 2026-09-30" reads like a signal from the future."""
    bars = df[list(AGG)].resample(rule).agg(AGG)
    bars["_last_day"] = df.index.to_series().resample(rule).last()
    bars = bars.dropna(subset=["close"])
    bars.index = pd.DatetimeIndex(bars.pop("_last_day"), name=df.index.name)
    return bars


def macd(close: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    m = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    s = m.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": m, "signal": s, "hist": m - s})


def stochastic(df: pd.DataFrame, n=14, k_smooth=3, d_smooth=3) -> pd.DataFrame:
    """Slow stochastic (KD): %K = SMA(k_smooth) of fast %K over n bars, %D = SMA(d_smooth) of %K."""
    ll, hh = df["low"].rolling(n).min(), df["high"].rolling(n).max()
    fast_k = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    k = fast_k.rolling(k_smooth).mean()
    return pd.DataFrame({"k": k, "d": k.rolling(d_smooth).mean()})


def rsi(close: pd.Series, n=14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    out = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    return out.mask((dn == 0) & (up > 0), 100.0)


def bollinger(close: pd.Series, n=20, k=2.0) -> pd.DataFrame:
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std()
    up, lo = mid + k * sd, mid - k * sd
    return pd.DataFrame({"bb_mid": mid, "bb_up": up, "bb_lo": lo,
                         "pct_b": (close - lo) / (up - lo), "bandwidth": (up - lo) / mid})


def adx(df: pd.DataFrame, n=14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    up, dn = h.diff(), -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()


def chaikin_money_flow(df: pd.DataFrame, n=20) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng
    return (mfm.fillna(0) * df["volume"]).rolling(n).sum() / df["volume"].rolling(n).sum()


def add_all(df: pd.DataFrame, ma=(5, 20, 50, 200)) -> pd.DataFrame:
    out = df.copy()
    for n in ma:
        out[f"ma{n}"] = out["close"].rolling(n).mean()
    out = out.join(macd(out["close"])).join(stochastic(out)).join(bollinger(out["close"]))
    out["rsi"] = rsi(out["close"])
    out["adx"] = adx(out)
    out["obv"] = obv(out)
    out["cmf"] = chaikin_money_flow(out)
    out["vol_ma20"] = out["volume"].rolling(20).mean()
    return out


def cross_events(a: pd.Series, b: pd.Series) -> pd.Series:
    """+1 = a crosses above b (golden), -1 = crosses below (death), 0 = none."""
    above = (a > b).astype(int)
    return above.diff().fillna(0).astype(int).where(a.notna() & b.notna(), 0)


def support_resistance(df: pd.DataFrame, window: int = 5, n_each: int = 2, tol: float = 0.015) -> dict:
    """Cluster swing highs/lows (±tol) and return the n nearest supports below / resistances above price."""
    h, l, c = df["high"], df["low"], float(df["close"].iloc[-1])
    piv_h = h[(h == h.rolling(2 * window + 1, center=True).max())].dropna()
    piv_l = l[(l == l.rolling(2 * window + 1, center=True).min())].dropna()
    clusters: list[list[float]] = []
    for v in sorted(list(piv_h.values) + list(piv_l.values)):
        if clusters and abs(v / np.mean(clusters[-1]) - 1) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    lv = [(float(np.mean(g)), len(g)) for g in clusters]
    sup = sorted([x for x in lv if x[0] < c * (1 - tol / 3)], key=lambda x: c - x[0])[:n_each]
    res = sorted([x for x in lv if x[0] > c * (1 + tol / 3)], key=lambda x: x[0] - c)[:n_each]
    if not res:                                   # at/near highs: use the window high
        res = [(float(h.max()), 1)]
    if not sup:
        sup = [(float(l.min()), 1)]
    return {"support": [{"price": p, "touches": t} for p, t in sup],
            "resistance": [{"price": p, "touches": t} for p, t in res]}


def _last_cross(ev: pd.Series, lookback: int) -> dict | None:
    hits = ev.iloc[-lookback:]
    hits = hits[hits != 0]
    if hits.empty:
        return None
    return {"type": "golden cross" if hits.iloc[-1] > 0 else "death cross",
            "date": str(hits.index[-1].date()), "bars_ago": int(len(ev) - ev.index.get_loc(hits.index[-1]) - 1)}


def _f(x):
    return float(x) if pd.notna(x) else None


def snapshot(d: pd.DataFrame, label: str, lookback: int = 10) -> dict:
    """Latest state of one timeframe as a JSON-friendly dict."""
    r = d.iloc[-1]
    mas = sorted([c for c in d.columns if c.startswith("ma") and c[2:].isdigit() and pd.notna(r[c])],
                 key=lambda c: int(c[2:]))
    vals = [r[c] for c in mas]
    if len(vals) >= 3 and all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)):
        align = "bullish stack"
    elif len(vals) >= 3 and all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)):
        align = "bearish stack"
    else:
        align = "mixed"
    bw = d["bandwidth"].dropna()
    k, rv, ax = r.get("k"), r.get("rsi"), r.get("adx")
    return {
        "timeframe": label,
        "date": str(d.index[-1].date()),
        "close": float(r["close"]),
        "ma": {c: float(r[c]) for c in mas},
        "ma_alignment": align,
        "macd": {"macd": _f(r["macd"]), "signal": _f(r["signal"]), "hist": _f(r["hist"]),
                 "state": "above signal" if r["macd"] > r["signal"] else "below signal",
                 "hist_trend": "expanding" if abs(r["hist"]) > abs(d["hist"].iloc[-2]) else "contracting",
                 "last_cross": _last_cross(cross_events(d["macd"], d["signal"]), lookback)},
        "kd": {"k": _f(k), "d": _f(r.get("d")),
               "zone": (("overbought (80+)" if k >= 80 else "oversold (20-)" if k <= 20 else "neutral")
                        if pd.notna(k) else None),
               "last_cross": _last_cross(cross_events(d["k"], d["d"]), lookback)},
        "rsi": {"value": _f(rv), "zone": (("overbought (70+)" if rv >= 70 else "oversold (30-)" if rv <= 30
                                           else "neutral") if pd.notna(rv) else None)},
        "bollinger": {"pct_b": _f(r["pct_b"]),
                      "squeeze": bool(len(bw) > 60 and bw.iloc[-1] <= bw.iloc[-120:].quantile(0.1))},
        "adx": {"value": _f(ax), "trend_strength": (("trending (25+)" if ax >= 25 else "weak / range")
                                                   if pd.notna(ax) else None)},
        "cmf_20": _f(r.get("cmf")),
        "volume_ratio_20": float(r["volume"] / r["vol_ma20"]) if pd.notna(r["vol_ma20"]) and r["vol_ma20"] else None,
    }


def multi_timeframe(daily: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, dict]]:
    out = {}
    for label, frame, ma in [("Daily", daily, (5, 20, 50, 200)),
                             ("Weekly", resample(daily, "W-FRI"), (10, 20, 50)),
                             ("Monthly", resample(daily, "ME"), (5, 10, 20))]:
        ind = add_all(frame, ma=ma)
        out[label] = (ind, snapshot(ind, label))
    return out
