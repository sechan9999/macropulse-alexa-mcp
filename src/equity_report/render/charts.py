"""Charts: static PNGs (Monthly / Weekly / Daily, for Word + Excel) and interactive plotly (dashboard).

Conventions: US candles (up green / down red); moving averages use colours that are neither green nor
red, in a fixed order; current price in a gold tag; right-edge labels are de-overlapped; indicators
live in their own panels (never a second y-axis on the price chart).
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..engine.indicators import WINDOW  # noqa: E402

UP, DOWN = "#16a34a", "#dc2626"
MA_COLORS = ["#2a78d6", "#eb6834", "#4a3aa7", "#c98500"]
INK, MUTED, GRID, GOLD = "#0b0b0b", "#52514e", "#e6e5e0", "#b88700"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bdbcb6")
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _dodge(items, gap):
    out, last = [], -np.inf
    for y, t, c in sorted(items, key=lambda x: x[0]):
        ny = max(y, last + gap)
        out.append((y, ny, t, c))
        last = ny
    return out


def price_chart_png(ind: pd.DataFrame, screen: dict, sr: dict, label: str, title: str) -> bytes:
    n = WINDOW.get(label, 126)
    d = ind.iloc[-n:]
    x = np.arange(len(d))
    fig = plt.figure(figsize=(12, 9.5), facecolor="white")
    gs = fig.add_gridspec(4, 1, height_ratios=[5, 1.2, 1.6, 1.6], hspace=0.08)
    ax, axv, axm, axk = (fig.add_subplot(gs[i]) for i in range(4))
    for a in (axv, axm, axk):
        a.sharex(ax)
    up = d["close"] >= d["open"]
    col = np.where(up, UP, DOWN)
    ax.vlines(x, d["low"], d["high"], color=col, lw=0.9)
    body = (d["close"] - d["open"]).abs()
    body = body.where(body > 0, d["close"] * 0.0008)
    ax.bar(x, body, 0.65, bottom=np.minimum(d["open"], d["close"]), color=col, edgecolor=col, lw=0.5)

    labels = []
    mas = sorted([c for c in d.columns if c.startswith("ma") and c[2:].isdigit()], key=lambda c: int(c[2:]))
    unit = {"Daily": "D", "Weekly": "W", "Monthly": "M"}[label]
    for i, c in enumerate(mas):
        ax.plot(x, d[c], color=MA_COLORS[i % 4], lw=1.6)
        if pd.notna(d[c].iloc[-1]):
            labels.append((float(d[c].iloc[-1]), f"MA{c[2:]}{unit} {d[c].iloc[-1]:,.2f}", MA_COLORS[i % 4]))
    last = float(d["close"].iloc[-1])
    ax.axhline(last, color=GOLD, lw=1, ls=":")
    labels.append((last, f"Last {last:,.2f}", GOLD))

    lo, hi = float(d["low"].min()), float(d["high"].max())
    pad = (hi - lo) * 0.08
    for kind, key in (("S", "support"), ("R", "resistance")):
        for j, lv in enumerate(sr.get(key, [])[:2], 1):
            p = lv["price"]
            if lo - pad <= p <= hi + pad:
                ax.axhline(p, color=MUTED, lw=1, ls="--", alpha=0.7)
                ax.text(0.5, p, f" {kind}{j} {p:,.2f}", va="bottom", ha="left", fontsize=10, color=MUTED)

    sig = pd.concat([screen["_all_candles"], screen["_all_signals"]], ignore_index=True)
    sig = sig[(sig["date"] >= d.index[0]) & (sig["strength"] >= 2)]
    pos = {dt: i for i, dt in enumerate(d.index)}
    for _, r in sig.iterrows():
        i = pos.get(r["date"])
        if i is None:
            continue
        if r["direction"] == "bull":
            ax.scatter(i, d["low"].iloc[i] - pad * 0.35, marker="^", s=60, color=UP, zorder=5, edgecolor="white", lw=1)
        elif r["direction"] == "bear":
            ax.scatter(i, d["high"].iloc[i] + pad * 0.35, marker="v", s=60, color=DOWN, zorder=5, edgecolor="white", lw=1)
    for _, r in sig.sort_values("date").tail(2).iterrows():
        i = pos.get(r["date"])
        if i is None:
            continue
        yv = d["low"].iloc[i] - pad * 0.9 if r["direction"] == "bull" else d["high"].iloc[i] + pad * 0.9
        ha = "right" if i > len(d) * 0.8 else "left" if i < len(d) * 0.2 else "center"
        ax.annotate(r["name"], (i, yv), ha=ha, fontsize=10, color=INK,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bdbcb6", lw=0.8))
    ax.set_ylim(lo - pad * 1.3, hi + pad * 1.3)
    for y0, y1, t, c in _dodge(labels, (hi - lo + 2.6 * pad) * 0.06):
        is_px = t.startswith("Last")
        ax.annotate(t, xy=(len(d) - 1, y0), xytext=(len(d) + 1.5, y1), fontsize=10.5, va="center",
                    color="white" if is_px else c, annotation_clip=False, fontweight="bold" if is_px else "normal",
                    bbox=dict(boxstyle="round,pad=0.25", fc=c, ec=c) if is_px else None,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.6))

    axv.bar(x, d["volume"], 0.65, color=col, alpha=0.55)
    axv.set_ylabel("Volume", fontsize=10, color=MUTED)
    axm.bar(x, d["hist"], 0.65, color=np.where(d["hist"] >= 0, UP, DOWN), alpha=0.5)
    axm.plot(x, d["macd"], color=INK, lw=1.3, label="MACD")
    axm.plot(x, d["signal"], color=SERIES[1], lw=1.3, label="Signal")
    axm.axhline(0, color="#9a9994", lw=0.7)
    axm.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)
    axk.plot(x, d["k"], color=INK, lw=1.3, label="%K")
    axk.plot(x, d["d"], color=SERIES[1], lw=1.3, label="%D")
    for lvl in (80, 20):
        axk.axhline(lvl, color="#9a9994", lw=0.7, ls="--")
    axk.set_ylim(0, 100)
    axk.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)
    for a in (ax, axv, axm, axk):
        _style(a)
        a.set_xlim(-1, len(d) + 0.5)
    for a in (ax, axv, axm):
        plt.setp(a.get_xticklabels(), visible=False)
    ticks = np.linspace(0, len(d) - 1, 7).astype(int)
    axk.set_xticks(ticks, [d.index[i].strftime("%Y-%m" if label == "Monthly" else "%y-%m-%d") for i in ticks])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axv.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e6:.0f}M"))
    span = {"Daily": "last 6 months", "Weekly": "last 12 months", "Monthly": "last 3 years"}[label]
    ax.set_title(f"{title} · {label} · {span}", loc="left", fontsize=18, color=INK, pad=10)
    ax.text(1.0, 1.01, f"Pattern screen: {screen['overall']} (bull {screen['bull_score']} : bear {screen['bear_score']})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=11, color=MUTED)
    fig.subplots_adjust(left=0.07, right=0.84, top=0.94, bottom=0.05)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def plotly_chart(ind: pd.DataFrame, screen: dict, label: str, height: int = 720):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    d = ind.iloc[-WINDOW.get(label, 126):]
    fmt = "%Y-%m" if label == "Monthly" else "%Y-%m-%d"
    xs = d.index.strftime(fmt)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.025, row_heights=[0.55, 0.13, 0.16, 0.16])
    fig.add_trace(go.Candlestick(x=xs, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                                 increasing=dict(line=dict(color=UP), fillcolor=UP),
                                 decreasing=dict(line=dict(color=DOWN), fillcolor=DOWN), name="Price",
                                 showlegend=False), 1, 1)
    mas = sorted([c for c in d.columns if c.startswith("ma") and c[2:].isdigit()], key=lambda c: int(c[2:]))
    for i, c in enumerate(mas):
        fig.add_trace(go.Scatter(x=xs, y=d[c], name=f"MA{c[2:]}", line=dict(color=MA_COLORS[i % 4], width=1.6),
                                 hovertemplate="%{y:,.2f}"), 1, 1)
    sig = pd.concat([screen["_all_candles"], screen["_all_signals"]], ignore_index=True)
    sig = sig[(sig["strength"] >= 2) & sig["date"].isin(d.index)]
    for direction, sym, color, ycol, off in (("bull", "triangle-up", UP, "low", 0.985),
                                              ("bear", "triangle-down", DOWN, "high", 1.015)):
        s = sig[sig["direction"] == direction].groupby("date")["name"].apply(", ".join)
        if len(s):
            fig.add_trace(go.Scatter(x=pd.DatetimeIndex(s.index).strftime(fmt), y=d.loc[s.index, ycol] * off,
                                     mode="markers", name="Bullish signal" if direction == "bull" else "Bearish signal",
                                     marker=dict(symbol=sym, size=11, color=color, line=dict(color="white", width=1)),
                                     text=s.values, hovertemplate="%{text}<extra></extra>"), 1, 1)
    fig.add_trace(go.Bar(x=xs, y=d["volume"], marker_color=np.where(d["close"] >= d["open"], UP, DOWN), opacity=0.55,
                         name="Volume", showlegend=False), 2, 1)
    fig.add_trace(go.Bar(x=xs, y=d["hist"], marker_color=np.where(d["hist"] >= 0, UP, DOWN), opacity=0.5,
                         name="MACD hist", showlegend=False), 3, 1)
    fig.add_trace(go.Scatter(x=xs, y=d["macd"], name="MACD", line=dict(color="#8a8984", width=1.4)), 3, 1)
    fig.add_trace(go.Scatter(x=xs, y=d["signal"], name="Signal", line=dict(color=SERIES[1], width=1.4)), 3, 1)
    fig.add_trace(go.Scatter(x=xs, y=d["k"], name="%K", line=dict(color="#8a8984", width=1.4)), 4, 1)
    fig.add_trace(go.Scatter(x=xs, y=d["d"], name="%D", line=dict(color=SERIES[1], width=1.4)), 4, 1)
    for lvl in (80, 20):
        fig.add_hline(y=lvl, line=dict(color="#9a9994", width=1, dash="dash"), row=4, col=1)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.02, x=0, font=dict(size=11)))
    fig.update_xaxes(type="category", nticks=8, showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="KD", row=4, col=1, range=[0, 100])
    return fig
