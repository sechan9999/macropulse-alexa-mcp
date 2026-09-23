"""Single-file HTML DCF dashboard (light/dark, phone-width safe).

Sections: KPI tiles -> macro-regime overlay -> valuation range (football field) -> sensitivity grid ->
scenarios -> FCF actual vs projected -> interactive Daily/Weekly/Monthly charts -> indicator snapshot ->
money flow & positioning -> narrative.
`theme="dark"` pins the dark palette (used when embedded in the dark Streamlit app);
`offline=True` inlines plotly.js so the downloaded file works without internet.
"""
from __future__ import annotations

import html

import markdown as md
import numpy as np
import plotly.graph_objects as go

from ..engine import patterns
from ..engine.analysis import Report
from . import charts

CSS = """
:root{color-scheme:light;--bg:#f7f7f5;--card:#fff;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--line:#e6e5e0;
--up:#15803d;--down:#c0392b;--gold:#b88700;--above:#cde2fb;--below:#fde0cf;--mid:#f0efec;--warn:#b42318;--chip:#eef4fd}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#0b1120;--card:#111827;--ink:#f8fafc;
--ink2:#cbd5e1;--muted:#94a3b8;--line:#1f2a3c;--up:#34d399;--down:#f87171;--gold:#e0b43c;--above:#16345c;--below:#5a2a14;
--mid:#1e293b;--warn:#ff8a80;--chip:#172554}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#0b1120;--card:#111827;--ink:#f8fafc;--ink2:#cbd5e1;--muted:#94a3b8;
--line:#1f2a3c;--up:#34d399;--down:#f87171;--gold:#e0b43c;--above:#16345c;--below:#5a2a14;--mid:#1e293b;--warn:#ff8a80;--chip:#172554}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 Inter,system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
header h1{font-size:26px;margin:0 0 4px}header .sub{color:var(--ink2);font-size:13px}
.banner{background:var(--below);color:var(--warn);font-weight:700;padding:10px 14px;border-radius:10px;margin:14px 0}
.grid{display:grid;gap:14px}.kpis{grid-template-columns:repeat(auto-fit,minmax(170px,1fr));margin:18px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px;min-width:0}
.kpi .lab{color:var(--ink2);font-size:12.5px}.kpi .val{font-size:24px;font-weight:700;margin-top:2px;font-variant-numeric:tabular-nums}
.kpi .d{font-size:13px;color:var(--ink2)}.pos{color:var(--up)}.neg{color:var(--down)}
h2{font-size:18px;margin:30px 0 10px}.two{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}.two>div{min-width:0}
@media(max-width:820px){.two{grid-template-columns:minmax(0,1fr)}}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13.5px}
th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--ink2);font-weight:600}.scroll{overflow-x:auto}
.sens td{text-align:center}.sens td.hl{outline:2px solid var(--gold);outline-offset:-2px;font-weight:700}
.fomc{display:inline-block;margin-left:4px;padding:0 5px;border-radius:6px;background:var(--ink);color:var(--card);
font-size:10.5px;font-weight:700;vertical-align:1px}
.chips{display:flex;gap:8px;flex-wrap:wrap}.chip{background:var(--chip);border-radius:999px;padding:4px 12px;font-size:13px}
.tabs{display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap}.tabs button{background:transparent;border:1px solid var(--line);
color:var(--ink2);padding:6px 14px;border-radius:999px;cursor:pointer;font:inherit}
.tabs button[aria-selected=true]{background:var(--ink);color:var(--card);border-color:var(--ink)}
.tabpane{display:none}.tabpane.on{display:block}
.analysis h2{font-size:17px;border-top:1px solid var(--line);padding-top:18px}.analysis h3{font-size:15px;color:var(--ink2)}
.analysis table{margin:8px 0;display:block;overflow-x:auto}.analysis td{white-space:normal}
.note{color:var(--muted);font-size:12.5px}.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12.5px;color:var(--ink2);margin-top:8px}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-2px;margin-right:5px}
"""
FONT = dict(family="Inter, system-ui, sans-serif", size=12, color="#8a8984")


def _usd(x, d=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"${x:,.{d}f}"


def _pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f'<span class="{"pos" if x > 0 else "neg" if x < 0 else ""}">{x * 100:+.1f}%</span>'


def _div(fig, div_id):
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id,
                       config={"displaylogo": False, "responsive": True})


def _football(ctx) -> go.Figure:
    v, price = ctx["valuation"], ctx["company"]["price"]
    rows = [("52-week range", v["range_52w"]["low"], v["range_52w"]["high"], None)]
    for m, bd in v["relative_bands"].items():
        rows.append((f"{m} band (10–90th pct)", bd["low"], bd["high"], bd["mid"]))
    rows.append(("DCF sensitivity (WACC ±1pp, g ±1pp)", v["sensitivity_min"], v["sensitivity_max"], v["dcf_base_per_share"]))
    sc = v["scenarios"]
    rows.append(("DCF scenarios (bear–bull)", sc["Bear"]["per_share"], sc["Bull"]["per_share"], v["expected_per_share"]))
    fig = go.Figure()
    for lab, lo, hi, mid in rows:
        fig.add_trace(go.Bar(y=[lab], x=[hi - lo], base=[lo], orientation="h", marker_color=charts.SERIES[0],
                             opacity=0.85, width=0.5, showlegend=False,
                             hovertemplate=f"{lab}<br>${lo:,.2f} – ${hi:,.2f}<extra></extra>"))
        if mid:
            fig.add_trace(go.Scatter(y=[lab], x=[mid], mode="markers", showlegend=False,
                                     marker=dict(symbol="line-ns-open", size=22, color="#8a8984", line=dict(width=3)),
                                     hovertemplate=f"mid / expected ${mid:,.2f}<extra></extra>"))
    fig.add_vline(x=price, line=dict(color=charts.GOLD, width=2, dash="dot"), annotation_text=f"Last ${price:,.2f}",
                  annotation_position="top")
    fig.update_layout(height=80 + 58 * len(rows), margin=dict(l=10, r=20, t=30, b=10), barmode="overlay", font=FONT,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor="rgba(128,128,128,0.18)"),
                      yaxis=dict(autorange="reversed"))
    return fig


def _fcf(val) -> go.Figure:
    h, base = val["history"], val["base"]
    y0 = int(h.index[-1])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"FY{y}" for y in h.index], y=h["fcf"] / 1e6, name="FCF actual", marker_color=charts.SERIES[0],
                         hovertemplate="%{x}: $%{y:,.0f}m<extra></extra>"))
    fig.add_trace(go.Bar(x=[f"FY{y0 + i + 1}E" for i in range(len(base["fcf"]))], y=base["fcf"] / 1e6,
                         name="FCF projected (base)", marker_color=charts.SERIES[1],
                         hovertemplate="%{x}: $%{y:,.0f}m<extra></extra>"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), bargap=0.35, font=FONT,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.1, x=0),
                      yaxis=dict(title="USD m", tickformat=",.0f", gridcolor="rgba(128,128,128,0.18)"))
    return fig


FOMC_TAGS = {"hawkish_50bps": "H", "dovish_50bps": "D", "stagflation_inversion": "S", "liquidity_cascade": "L"}


def fomc_cells(sens, overlay) -> dict:
    """{(row, col): [tag, ...]} — nearest grid cell for each FOMC scenario inside the grid (±half a step)."""
    ws, gs = np.array(sens.index, float), np.array(sens.columns, float)
    half_w, half_g = np.min(np.diff(ws)) / 2 + 1e-9, np.min(np.diff(gs)) / 2 + 1e-9
    out: dict = {}
    for x in overlay:
        i, j = int(np.abs(ws - x["wacc"]).argmin()), int(np.abs(gs - x["terminal_growth"]).argmin())
        if abs(ws[i] - x["wacc"]) <= half_w and abs(gs[j] - x["terminal_growth"]) <= half_g:
            out.setdefault((i, j), []).append(FOMC_TAGS[x["scenario"]])
    return out


def _sens(val, price) -> str:
    s = val["sensitivity"]
    vals = s.values
    dev = vals / price - 1
    m = np.nanmax(np.abs(dev)) or 1
    ci, cj = len(s.index) // 2, len(s.columns) // 2
    tags = fomc_cells(s, val.get("fomc_overlay", []))
    rows = ["<tr><th>WACC \\ g</th>" + "".join(f"<th>{g:.1%}</th>" for g in s.columns) + "</tr>"]
    for i, w in enumerate(s.index):
        tds = []
        for j in range(len(s.columns)):
            x, d = vals[i, j], dev[i, j]
            a = int(min(1, abs(d) / m) * 100) if not np.isnan(d) else 0
            bg = "var(--above)" if d > 0 else "var(--below)"
            badge = "".join(f'<span class="fomc">{t}</span>' for t in tags.get((i, j), []))
            tds.append(f'<td class="{"hl" if (i, j) == (ci, cj) else ""}" title="{d * 100:+.1f}% vs price" '
                       f'style="background:color-mix(in srgb,{bg} {a}%,var(--mid))">{_usd(x)}{badge}</td>')
        rows.append(f"<tr><th>{w:.2%}</th>{''.join(tds)}</tr>")
    return (f'<div class="scroll"><table class="sens">{"".join(rows)}</table></div><div class="legend">'
            '<span><i class="sw" style="background:var(--below)"></i>below price</span>'
            '<span><i class="sw" style="background:var(--mid)"></i>near price</span>'
            '<span><i class="sw" style="background:var(--above)"></i>above price</span><span>outlined = base case</span>'
            '<span><span class="fomc">H</span><span class="fomc">D</span><span class="fomc">S</span>'
            '<span class="fomc">L</span> FOMC scenario (table below)</span></div>')


def _fomc(val, price) -> str:
    ov = val.get("fomc_overlay", [])
    if not ov:
        return ""
    body = "".join(
        f"<tr><td><span class='fomc'>{FOMC_TAGS[x['scenario']]}</span> {html.escape(x['title'])}</td>"
        f"<td>{x['d_rf'] * 1e4:+.0f} / {x['d_erp'] * 1e4:+.0f} / {x['d_spread'] * 1e4:+.0f} / {x['d_g'] * 1e4:+.0f}</td>"
        f"<td>{x['wacc']:.2%}</td><td>{x['terminal_growth']:.1%}</td><td>{_usd(x['per_share'])}</td>"
        f"<td>{_pct(x['upside'])}</td><td>{_pct(x['tab13_price_shock'])}</td></tr>" for x in ov)
    return ('<h2>FOMC shock overlay (tab 13 scenarios)</h2><div class="card"><div class="scroll"><table><tr><th>Scenario</th><th>Δ rf / ERP / spread / g (bp)</th><th>WACC</th>'
            f'<th>g</th><th>Value / share</th><th>vs. price</th><th>Tab-13 price shock</th></tr>{body}</table></div>'
            '<p class="note">Illustrative rate / premium / spread shifts, FCF path fixed. Scenarios outside the '
            '±1pp grid have no badge. Tab-13 price shock = the hard-coded single-name shock the FOMC tab applies '
            '(default for tickers it does not list).</p></div>')


def _snapshot(ctx) -> str:
    t, tfs = ctx["technical"], ("Monthly", "Weekly", "Daily")

    def cross(x):
        return f"{x['type']} ({x['date']})" if x else "–"

    rows = [("MA alignment", [t[k]["ma_alignment"] for k in tfs]),
            ("MACD", [f"{t[k]['macd']['state']} · hist {t[k]['macd']['hist_trend']}" for k in tfs]),
            ("MACD last cross", [cross(t[k]["macd"]["last_cross"]) for k in tfs]),
            ("KD %K / %D", [f"{t[k]['kd']['k'] or 0:.0f} / {t[k]['kd']['d'] or 0:.0f} · {t[k]['kd']['zone']}" for k in tfs]),
            ("KD last cross", [cross(t[k]["kd"]["last_cross"]) for k in tfs]),
            ("RSI(14)", [f"{t[k]['rsi']['value'] or 0:.0f} · {t[k]['rsi']['zone']}" for k in tfs]),
            ("ADX", [f"{t[k]['adx']['value'] or 0:.0f} · {t[k]['adx']['trend_strength']}" for k in tfs]),
            ("Bollinger squeeze", ["yes" if t[k]["bollinger"]["squeeze"] else "no" for k in tfs])]
    body = "".join(f"<tr><td>{a}</td>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in xs) + "</tr>" for a, xs in rows)
    return f'<div class="scroll"><table><tr><th>Indicator</th>{"".join(f"<th>{k}</th>" for k in tfs)}</tr>{body}</table></div>'


def _flow(ctx) -> str:
    mf = ctx["money_flow"]
    pos = mf.get("positioning") or {}
    items = [("Chaikin money flow (20d)", f"{mf['cmf_20']:+.2f}" if mf.get("cmf_20") is not None else "–"),
             ("Up-day volume share (20d)", f"{mf['up_volume_share_20d']:.0%}")]
    for k, lab, fmt in (("institutional_pct", "Institutional ownership", "{:.0%}"), ("short_pct_float", "Short % of float", "{:.1%}"),
                        ("short_change_vs_prior_month", "Shares short vs prior month", "{:+.0%}"),
                        ("short_ratio_days", "Days to cover", "{:.1f}"), ("insider_net_shares_6m", "Insider net shares (6m)", "{:+,.0f}")):
        if pos.get(k) is not None:
            items.append((lab, fmt.format(pos[k])))
    body = "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in items)
    return f'<div class="scroll"><table>{body}</table></div><p class="note">{html.escape(mf["note"])}</p>'


def build(rep: Report, *, theme: str | None = None, offline: bool = False) -> str:
    ctx, val = rep.ctx, rep.val
    c, v, rb, rg = ctx["company"], ctx["valuation"], ctx["rule_based"], ctx["macro_regime"]
    lv, price = rb["levels"], c["price"]
    screens = {"Daily": rep.screen, "Weekly": patterns.screen(rep.mtf["Weekly"][0]),
               "Monthly": patterns.screen(rep.mtf["Monthly"][0])}
    tech = {k: _div(charts.plotly_chart(rep.mtf[k][0], screens[k], k), f"tech-{i}")
            for i, k in enumerate(("Daily", "Weekly", "Monthly"))}
    if offline:
        from plotly.offline import get_plotlyjs
        pjs = f"<script>{get_plotlyjs()}</script>"
    else:
        pjs = '<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>'
    kpis = [("Last price", _usd(price), f"as of {c['as_of']}"),
            ("Probability-weighted value", _usd(v["expected_per_share"]), _pct(v["expected_upside"]) + " vs price"),
            ("DCF base case", _usd(v["dcf_base_per_share"]), _pct(v["dcf_base_upside"])),
            ("Rule-based rating", rb["rating"], f"score {rb['score']:+.1f}"),
            ("WACC · terminal g", f"{v['wacc']:.2%}", f"g {v['terminal_growth']:.1%} · β {v['beta']:.2f}"),
            ("Pattern screen (daily)", rep.screen["overall"], f"bull {rep.screen['bull_score']} : bear {rep.screen['bear_score']}"),
            ("Target 1 · 6m touch prob.", _usd(lv["target1"]),
             f"{(lv['touch_prob_6m']['target1'] or 0):.0%} "
             + (f"(Ridge drift, E[R] {ctx['drift']['stock_expected_return']:+.1%})"
                if ctx["drift"].get("stock_expected_return") is not None else "(driftless)")),
            ("Stop", _usd(lv["stop"]), _pct(lv["stop"] / price - 1))]
    kpi_html = "".join(f'<div class="card kpi"><div class="lab">{a}</div><div class="val">{b}</div><div class="d">{d}</div></div>'
                       for a, b, d in kpis)
    scen = "".join(f"<tr><td>{k}</td><td>{_usd(s['per_share'])}</td><td>{_pct(s['upside'])}</td><td>{s['prob']:.0%}</td></tr>"
                   for k, s in v["scenarios"].items())
    notes = "".join(f"<p class='note'>Note: {html.escape(n)}</p>" for n in v["notes"])
    issues = "".join(f"<p class='note'>Data check: {html.escape(n)}</p>" for n in rep.issues)
    banner = ('<div class="banner">SYNTHETIC DEMO DATA — not a real company.</div>' if c["synthetic_data"] else "")
    theme_attr = f' data-theme="{theme}"' if theme in ("light", "dark") else ""
    page = f"""<!doctype html><html lang="en"{theme_attr}><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(c['ticker'])} Equity Report</title>
<style>{CSS}</style>{pjs}</head><body><div class="wrap">
<header><h1>{html.escape(c['name'])} <span style="color:var(--ink2);font-weight:500">{c['ticker']}</span></h1>
<div class="sub">Market cap ${c['market_cap_musd']:,.0f}m · {html.escape('; '.join(ctx['data_sources']))} · narrative: {html.escape(rep.narrative_method)}</div></header>
{banner}
<section class="grid kpis">{kpi_html}</section>
<h2>Macro regime overlay</h2><div class="card"><div class="chips">
<span class="chip">Regime: <b>{rg['regime']}</b></span><span class="chip">Bear {rg['probs']['Bear']:.0%}</span>
<span class="chip">Base {rg['probs']['Base']:.0%}</span><span class="chip">Bull {rg['probs']['Bull']:.0%}</span>
<span class="chip">ERP shift {rg['erp_shift']:+.2%}</span></div>
<p class="note">Source: {html.escape(rg['source'])}. With neutral 25/50/25 weights the expected value would be
{_usd(v['neutral_weight_expected_per_share'])} instead of {_usd(v['expected_per_share'])}.</p></div>
<h2>Valuation range</h2><div class="card">{_div(_football(ctx), 'football')}
<p class="note">Bars = ranges, tick = midpoint / expected value, dotted line = last price.</p></div>
<section class="grid two"><div><h2>DCF sensitivity (value per share)</h2><div class="card">{_sens(val, price)}</div></div>
<div><h2>Scenarios</h2><div class="card"><div class="scroll"><table><tr><th>Scenario</th><th>Value / share</th><th>vs. price</th>
<th>Probability</th></tr>{scen}<tr><td><b>Weighted</b></td><td><b>{_usd(v['expected_per_share'])}</b></td>
<td>{_pct(v['expected_upside'])}</td><td>100%</td></tr></table></div>
<p class="note">Bear: growth −5pp, FCF margin ×0.8, WACC +1pp · Bull: the reverse. Inputs: risk-free {v['risk_free']:.2%}
({html.escape(v['risk_free_source'])}), ERP {v['erp_used']:.2%}, growth {v['growth_high']:.1%} → {v['terminal_growth']:.1%},
FCF margin {v['fcf_margin']:.1%}, terminal value {v['tv_share_of_ev']:.0%} of EV.</p>{notes}{issues}</div></div></section>
{_fomc(val, price)}
<h2>Free cash flow: actual and projected</h2><div class="card">{_div(_fcf(val), 'fcf')}</div>
<h2>Technical charts</h2><div class="card"><div class="tabs" role="tablist">
<button role="tab" aria-selected="true" data-t="0">Daily · 6M</button><button role="tab" aria-selected="false" data-t="1">Weekly · 12M</button>
<button role="tab" aria-selected="false" data-t="2">Monthly · 3Y</button></div>
<div class="tabpane on">{tech['Daily']}</div><div class="tabpane">{tech['Weekly']}</div><div class="tabpane">{tech['Monthly']}</div>
<p class="note">Green = up candle, red = down · ▲ bullish / ▼ bearish signal (strength ≥ 2 patterns, MACD/KD/MA crosses)</p></div>
<section class="grid two"><div><h2>Indicator snapshot</h2><div class="card">{_snapshot(ctx)}</div></div>
<div><h2>Money flow &amp; positioning</h2><div class="card">{_flow(ctx)}</div></div></section>
<h2>Analysis</h2><div class="card analysis">{md.markdown(rep.narrative or '', extensions=['tables'])}</div>
<p class="note">Not investment advice. Price date {c['as_of']}.</p></div>
<script>
document.querySelectorAll('.tabs button').forEach(b=>b.addEventListener('click',()=>{{
 document.querySelectorAll('.tabs button').forEach(x=>x.setAttribute('aria-selected','false'));b.setAttribute('aria-selected','true');
 document.querySelectorAll('.tabpane').forEach((p,i)=>p.classList.toggle('on',String(i)===b.dataset.t));
 const d=document.getElementById('tech-'+b.dataset.t);if(window.Plotly&&d)Plotly.Plots.resize(d);}}));
</script></body></html>"""
    return page
