"""Two-stage analysis.

Stage 1 (rules, reproducible): score valuation, multi-timeframe trend, candle/indicator screen and
money-flow proxies -> rating, stop / targets, touch probabilities.
Stage 2 (LLM, optional): hand the stage-1 JSON to the dashboard's existing LLM providers (Gemini or
Amazon Bedrock via src.macro_briefing) with a prompt that forbids numbers not in the JSON.
If no provider is configured the rule-based narrative is used — the report never blocks on an LLM.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import DCFAssumptions
from ..config import APP_URL
from . import dcf as dcf_mod
from . import indicators as ind
from . import market_drift
from . import patterns as pat
from .regime import overlay as make_overlay

RATINGS = ("Sell", "Underweight", "Neutral", "Overweight", "Buy")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


def fibonacci(prices: pd.DataFrame, lookback: int = 252) -> dict:
    p = prices.iloc[-lookback:]
    hi, lo = float(p["high"].max()), float(p["low"].min())
    up = p["high"].idxmax() > p["low"].idxmin()
    lv = {f"{r:.1%}": (hi - (hi - lo) * r) if up else (lo + (hi - lo) * r) for r in (0.236, 0.382, 0.5, 0.618)}
    return {"swing_high": hi, "swing_low": lo, "direction": "retracing an up-leg" if up else "rebounding from a down-leg",
            "levels": lv}


def money_flow(daily: pd.DataFrame, positioning: dict | None) -> dict:
    """US stand-in for Korea's daily institutional/foreign net buying: volume-based flow proxies
    (every day) + ownership/short-interest snapshots (13F / exchange reports, lagged)."""
    d = daily
    obv = d["obv"]
    out = {
        "cmf_20": float(d["cmf"].iloc[-1]) if pd.notna(d["cmf"].iloc[-1]) else None,
        "obv_20d_change_pct_of_avg_volume": float((obv.iloc[-1] - obv.iloc[-21]) / d["volume"].iloc[-20:].mean())
        if len(d) > 21 else None,
        "up_volume_share_20d": float(d["volume"].iloc[-20:][d["close"].diff().iloc[-20:] > 0].sum()
                                     / d["volume"].iloc[-20:].sum()),
        "positioning": positioning,
        "note": "No US equivalent of daily institutional net-buy data exists; CMF/OBV are price-volume proxies "
                "and ownership/short data are periodic snapshots.",
    }
    return out


def rule_opinion(ctx: dict) -> dict:
    v, t, s, mf = ctx["valuation"], ctx["technical"], ctx["pattern_screen_daily"], ctx["money_flow"]
    score, basis = 0.0, []
    up = v["expected_upside"]
    vs = 3 if up > 0.25 else 2 if up > 0.10 else -3 if up < -0.25 else -2 if up < -0.10 else 0
    score += vs
    basis.append(f"valuation {vs:+d} (prob-weighted upside {up:+.1%})")
    for tf, w in (("Monthly", 1.0), ("Weekly", 1.0), ("Daily", 0.5)):
        al = t[tf]["ma_alignment"]
        sc = (w if al == "bullish stack" else -w if al == "bearish stack" else 0)
        sc += 0.5 * w if t[tf]["macd"]["state"] == "above signal" else -0.5 * w
        score += sc
        basis.append(f"{tf.lower()} {sc:+.1f} ({al}, MACD {t[tf]['macd']['state']})")
    ps = 1 if s["overall"] == "bullish" else -1 if s["overall"] == "bearish" else 0
    score += ps
    basis.append(f"patterns {ps:+d} ({s['overall']})")
    cmf = mf.get("cmf_20")
    if cmf is not None:
        fs = 1 if cmf > 0.05 else -1 if cmf < -0.05 else 0
        score += fs
        basis.append(f"money flow {fs:+d} (CMF20 {cmf:+.2f})")
    pos = mf.get("positioning") or {}
    sc_chg = pos.get("short_change_vs_prior_month")
    if sc_chg is not None and sc_chg > 0.15:
        score -= 0.5
        basis.append(f"short interest -0.5 ({sc_chg:+.0%} m/m)")
    rating = ("Buy" if score >= 3.5 else "Overweight" if score >= 1.5 else "Sell" if score <= -3.5
              else "Underweight" if score <= -1.5 else "Neutral")

    price = ctx["company"]["price"]
    sr = ctx["support_resistance_daily"]
    sup = [x["price"] for x in sr.get("support", [])]
    res = [x["price"] for x in sr.get("resistance", [])]
    stop = sup[1] if len(sup) > 1 else (sup[0] * 0.97 if sup else price * 0.92)
    t1 = res[0] if res else price * 1.08
    t2_tech = res[1] if len(res) > 1 else t1 * 1.05
    t2 = max(t2_tech, v["expected_per_share"]) if v["expected_per_share"] > t1 else t2_tech
    sig = ctx["volatility_annual"]
    nu = ctx["drift"]["log_drift"]
    tp = dcf_mod.touch_probability
    lv = {"stop": stop, "target1": t1, "target2": t2}
    return {"score": round(score, 2), "rating": rating, "basis": " · ".join(basis),
            "levels": {**lv,
                       "touch_prob_6m": {k: tp(price, x, sig, 0.5, nu) for k, x in lv.items()},
                       "touch_prob_6m_driftless": {k: tp(price, x, sig, 0.5) for k, x in lv.items()},
                       "reward_risk": (t1 - price) / (price - stop) if price > stop else None}}


@dataclass
class Report:
    bundle: dict
    mtf: dict
    screen: dict
    sr: dict
    val: dict
    ctx: dict
    narrative: str | None = None
    narrative_method: str = ""
    issues: list = field(default_factory=list)


def analyze(bundle: dict, assumptions: DCFAssumptions | None = None, regime_label: str | None = None,
            regime_source: str = "MacroPulse regime engine", market: dict | None = None) -> Report:
    """`market` is market_drift.expected_market_return(...) (the tab-4 Ridge S&P view) or None (driftless)."""
    mtf = ind.multi_timeframe(bundle["prices"])
    daily = mtf["Daily"][0]
    screen = pat.screen(daily)
    sr = ind.support_resistance(daily.iloc[-ind.WINDOW["Daily"]:])
    ov = make_overlay(regime_label, regime_source)
    val = dcf_mod.build(bundle, assumptions, ov)
    fin = val["history"]
    last, prev = fin.iloc[-1], fin.iloc[-2] if len(fin) > 1 else fin.iloc[-1]

    def yoy(k):
        return float(last[k] / prev[k] - 1) if k in fin and pd.notna(prev[k]) and prev[k] else None

    def ratio(a, b):
        return float(last[a] / last[b]) if a in fin and b in fin and pd.notna(last[a]) and last[b] else None

    fundamentals = {
        "fiscal_year": str(fin.index[-1]), "fiscal_year_end": fin["fy_end"].iloc[-1] if "fy_end" in fin else None,
        "unit": "USD millions",
        "revenue": float(last["revenue"] / 1e6), "revenue_yoy": yoy("revenue"),
        "operating_income": float(last["operating_income"] / 1e6) if pd.notna(last["operating_income"]) else None,
        "op_income_yoy": yoy("operating_income"),
        "net_income": float(last["net_income"] / 1e6), "net_income_yoy": yoy("net_income"),
        "op_margin": ratio("operating_income", "revenue"), "roe": ratio("net_income", "total_equity"),
        "liabilities_to_equity": ratio("total_liabilities", "total_equity"),
        "current_ratio": ratio("current_assets", "current_liabilities"),
        "fcf": float(last["fcf"] / 1e6), "fcf_margin": float(last["fcf_margin"]),
        "eps_diluted": float(last["eps_diluted"]) if "eps_diluted" in fin and pd.notna(last["eps_diluted"]) else None,
        "pe_on_last_fy_eps": (bundle["last_price"] / bundle["eps_latest"]) if bundle.get("eps_latest") else None,
        "history_revenue": {k: float(x / 1e6) for k, x in fin["revenue"].items()},
        "history_fcf": {k: float(x / 1e6) for k, x in fin["fcf"].items() if pd.notna(x)},
    }
    a = val["assumptions"]
    ctx = {
        "company": {"name": bundle["name"], "ticker": bundle["ticker"], "price": bundle["last_price"],
                    "as_of": bundle["as_of"], "market_cap_musd": val["market_cap"] / 1e6,
                    "synthetic_data": bool(bundle.get("is_synthetic"))},
        "fundamentals": fundamentals,
        "money_flow": money_flow(daily, bundle.get("positioning")),
        "technical": {t: mtf[t][1] for t in ind.TIMEFRAMES},
        "pattern_screen_daily": screen,
        "support_resistance_daily": sr,
        "fibonacci_1y": fibonacci(bundle["prices"]),
        "macro_regime": val["regime"],
        "valuation": {
            "dcf_base_per_share": val["base"]["per_share"], "dcf_base_upside": val["base"]["upside"],
            "expected_per_share": val["expected_per_share"], "expected_upside": val["expected_upside"],
            "neutral_weight_expected_per_share": val["neutral_weight_expected"],
            "scenarios": val["scenarios"], "wacc": a["wacc"], "beta": a["beta_used"],
            "risk_free": a["risk_free_used"], "risk_free_source": a["risk_free_source"], "erp_used": a["erp_used"],
            "terminal_growth": a["terminal_growth"], "growth_high": a["growth_high_used"],
            "fcf_margin": a["fcf_margin_used"], "tv_share_of_ev": val["base"]["tv_share_of_ev"],
            "sensitivity_min": float(np.nanmin(val["sensitivity"].values)),
            "sensitivity_max": float(np.nanmax(val["sensitivity"].values)),
            "fomc_overlay": val["fomc_overlay"], "fomc_note": dcf_mod.FOMC_NOTE,
            "relative_bands": val["bands"], "range_52w": val["range_52w"], "notes": val["notes"],
        },
        "volatility_annual": val["sigma_annual"],
        "drift": market_drift.stock_drift(market, a["risk_free_used"], a["beta_used"], val["sigma_annual"]),
        "data_sources": bundle.get("sources", []),
        "not_available": ["news", "analyst consensus estimates", "earnings calendar"],
    }
    ctx["rule_based"] = rule_opinion(ctx)
    return Report(bundle, mtf, screen, sr, val, _clean(ctx))


# ─────────────────────────── stage 2: narrative ───────────────────────────
PROMPT = """You are writing the body of an equity research note for the MacroPulse dashboard.
Use ONLY the JSON context below. Rules:
- Quote numbers only from the context; if you compute something, show the short formula.
- Items listed in context.not_available (news, consensus, earnings calendar) are not connected:
  write "not connected" instead of inventing them.
- Probabilities: use only context.rule_based.levels.touch_prob_6m and state its assumption: log-normal touch
  probability with the drift in context.drift (S&P 500 expected return from the MacroPulse Ridge model via
  CAPM), or driftless when context.drift.source is "none (driftless)". touch_prob_6m_driftless is the
  zero-drift comparison.
- Mention context.valuation.fomc_overlay (tab-13 FOMC shock scenarios on the WACC x g grid) in section 4.
- Explain how the macro regime (context.macro_regime) changed the scenario weights and ERP.
- If company.synthetic_data is true, start with a bold line "SYNTHETIC DEMO DATA — not a real company".
- No hype. End with the line "Not investment advice."

Use exactly these section headings (markdown):
## 1. Fundamentals
## 2. Money Flow & Positioning
## 3. Technical Analysis
### Monthly (long term)
### Weekly (medium term)
### Daily (short term)
## 4. Valuation
## 5. Macro Regime Overlay
## 6. Verdict
(start from the rule-based rating, say whether you agree and why; include a markdown table of
stop / target 1 / target 2 with price, % from current price and 6-month touch probability; give
conditional actions instead of "wait and see")
## 7. Risks & What to Watch

Context:
```json
{context}
```"""


def _pct(x, d=1):
    return "n/a" if x is None else f"{x * 100:+.{d}f}%"


def _usd(x, d=2):
    return "n/a" if x is None else f"${x:,.{d}f}"


def rule_based_text(ctx: dict) -> str:
    c, f, v, t = ctx["company"], ctx["fundamentals"], ctx["valuation"], ctx["technical"]
    rb, s, sr, fib, mf, rg = (ctx["rule_based"], ctx["pattern_screen_daily"], ctx["support_resistance_daily"],
                              ctx["fibonacci_1y"], ctx["money_flow"], ctx["macro_regime"])
    lv = rb["levels"]
    L = ["**SYNTHETIC DEMO DATA — not a real company**", ""] if c["synthetic_data"] else []
    L += ["## 1. Fundamentals", "",
          f"FY{f['fiscal_year']} (ended {f['fiscal_year_end']}): revenue ${f['revenue']:,.0f}M ({_pct(f['revenue_yoy'])} y/y), "
          f"operating income ${(f['operating_income'] or 0):,.0f}M ({_pct(f['op_income_yoy'])}), net income "
          f"${f['net_income']:,.0f}M ({_pct(f['net_income_yoy'])}). Operating margin {(f['op_margin'] or 0):.1%}, "
          f"ROE {(f['roe'] or 0):.1%}, liabilities/equity {(f['liabilities_to_equity'] or 0):.2f}x"
          + (f", current ratio {f['current_ratio']:.2f}x" if f.get("current_ratio") else "")
          + (f", P/E on last fiscal-year EPS {f['pe_on_last_fy_eps']:.1f}x" if f.get("pe_on_last_fy_eps") else "") + ".",
          f"Free cash flow (CFO − CapEx) was ${f['fcf']:,.0f}M, a {f['fcf_margin']:.1%} margin. Revenue path: "
          + " → ".join(f"{k} ${x:,.0f}M" for k, x in f["history_revenue"].items()) + ".",
          "Analyst consensus and the earnings calendar are not connected.", "",
          "## 2. Money Flow & Positioning", ""]
    cmf = mf.get("cmf_20")
    L.append(f"20-day Chaikin Money Flow {cmf:+.2f}; up-day volume share over 20 days {mf['up_volume_share_20d']:.0%}."
             if cmf is not None else "Money-flow proxies unavailable.")
    pos = mf.get("positioning") or {}
    if pos:
        bits = []
        if pos.get("institutional_pct") is not None:
            bits.append(f"institutions hold {pos['institutional_pct']:.0%}")
        if pos.get("short_pct_float") is not None:
            bits.append(f"short interest {pos['short_pct_float']:.1%} of float")
        if pos.get("short_change_vs_prior_month") is not None:
            bits.append(f"shares short {pos['short_change_vs_prior_month']:+.0%} vs prior month")
        if pos.get("insider_net_shares_6m") is not None:
            bits.append(f"insider net shares (6m) {pos['insider_net_shares_6m']:+,.0f}")
        L.append("Positioning snapshot: " + ", ".join(bits) + ".")
    L += [mf["note"], "", "## 3. Technical Analysis", ""]
    for tf in ("Monthly", "Weekly", "Daily"):
        x = t[tf]
        mc, kd = x["macd"], x["kd"]
        cr = mc.get("last_cross")
        L += [f"### {tf}", "",
              f"Moving averages: {x['ma_alignment']}; close {_usd(x['close'])}. MACD {mc['state']} (histogram {mc['hist_trend']})"
              + (f", last {cr['type']} on {cr['date']}" if cr else "") + f". KD %K {(kd['k'] or 0):.0f} / %D {(kd['d'] or 0):.0f} "
              f"({kd['zone']}), RSI {(x['rsi']['value'] or 0):.0f} ({x['rsi']['zone']}), ADX {(x['adx']['value'] or 0):.0f} "
              f"({x['adx']['trend_strength']})" + ("; Bollinger squeeze in place" if x["bollinger"]["squeeze"] else "") + ".", ""]
    L += [f"Daily {s['window_bars']}-bar screen: {s['bullish_count']} bullish signals (score {s['bull_score']}) vs "
          f"{s['bearish_count']} bearish (score {s['bear_score']}) → **{s['overall']}**. Recent: "
          + (", ".join(f"{r['date']} {r['name']}" for r in s["recent_signals"][-4:]) or "none") + ".",
          "Support " + ", ".join(_usd(x["price"]) for x in sr["support"]) + "; resistance "
          + ", ".join(_usd(x["price"]) for x in sr["resistance"]) + ".",
          f"1-year Fibonacci ({fib['direction']}): " + ", ".join(f"{k} {_usd(p)}" for k, p in fib["levels"].items()) + ".", "",
          "## 4. Valuation", "",
          f"DCF base case {_usd(v['dcf_base_per_share'])} ({_pct(v['dcf_base_upside'])}); bear {_usd(v['scenarios']['Bear']['per_share'])}, "
          f"base {_usd(v['scenarios']['Base']['per_share'])}, bull {_usd(v['scenarios']['Bull']['per_share'])}; "
          f"probability-weighted {_usd(v['expected_per_share'])} ({_pct(v['expected_upside'])}).",
          f"Inputs: risk-free {v['risk_free']:.2%} ({v['risk_free_source']}), beta {v['beta']:.2f}, ERP {v['erp_used']:.2%}, "
          f"WACC {v['wacc']:.2%}, year-1 growth {v['growth_high']:.1%} fading to {v['terminal_growth']:.1%}, FCF margin "
          f"{v['fcf_margin']:.1%}. Terminal value is {v['tv_share_of_ev']:.0%} of EV; sensitivity range "
          f"{_usd(v['sensitivity_min'])}–{_usd(v['sensitivity_max'])}.",
          *[f"{m} band (10th/50th/90th pct): {_usd(b['low'])} / {_usd(b['mid'])} / {_usd(b['high'])}."
            for m, b in v["relative_bands"].items()],
          "FOMC shock overlay (tab 13 scenarios): " + "; ".join(
              f"{x['title']} → WACC {x['wacc']:.2%}, g {x['terminal_growth']:.1%}, {_usd(x['per_share'])} "
              f"({_pct(x['upside'])})" for x in v.get("fomc_overlay", [])) + ".",
          *[f"Note: {n}" for n in v["notes"]], "",
          "## 5. Macro Regime Overlay", "",
          f"MacroPulse regime: **{rg['regime']}** ({rg['source']}). Scenario weights bear/base/bull = "
          f"{rg['probs']['Bear']:.0%}/{rg['probs']['Base']:.0%}/{rg['probs']['Bull']:.0%} and ERP shift {rg['erp_shift']:+.2%}. "
          f"With neutral 25/50/25 weights the expected value would be {_usd(v['neutral_weight_expected_per_share'])}.", "",
          "## 6. Verdict", "",
          f"Rule-based rating: **{rb['rating']}** (score {rb['score']:+.1f}) — {rb['basis']}.", "",
          "| Level | Price | vs. current | 6-month touch probability* |", "|---|---|---|---|",
          *[f"| {nm} | {_usd(lv[k])} | {_pct(lv[k] / c['price'] - 1)} | {(lv['touch_prob_6m'][k] or 0):.0%} |"
            for nm, k in (("Stop", "stop"), ("Target 1", "target1"), ("Target 2", "target2"))],
          "", f"Reward/risk to target 1: {(lv['reward_risk'] or 0):.2f}. *{_drift_note(ctx)}", "",
          "## 7. Risks & What to Watch", "",
          "News, consensus estimates and the earnings calendar are not connected. The DCF is most sensitive to the FCF "
          "margin, terminal growth and WACC; the regime overlay can shift quickly when credit spreads move.", "",
          "Not investment advice."]
    return "\n".join(L)


def _drift_note(ctx: dict) -> str:
    dr, vol = ctx["drift"], ctx["volatility_annual"]
    if dr.get("stock_expected_return") is None:
        return (f"Driftless log-normal touch probability at {vol:.0%} annualised volatility — a distance "
                "yardstick, not a forecast (no market expected-return view was available).")
    dl = ctx["rule_based"]["levels"]["touch_prob_6m_driftless"]
    return (f"Log-normal touch probability at {vol:.0%} annualised volatility with drift from the MacroPulse Ridge "
            f"S&P 500 view: E[R_m] {dr['market_expected_return']:+.1%} (as of {dr['as_of']}) → CAPM E[R] "
            f"{dr['stock_expected_return']:+.1%}. Driftless for comparison: stop {(dl['stop'] or 0):.0%}, target 1 "
            f"{(dl['target1'] or 0):.0%}, target 2 {(dl['target2'] or 0):.0%}. A model-conditional estimate, not a forecast.")


def narrate(report: Report, provider: str = "auto") -> Report:
    """provider: 'auto' (Gemini if GEMINI_API_KEY, else rules), 'gemini', 'bedrock', or 'rules'."""
    ctx = report.ctx
    if provider == "auto":
        provider = "gemini" if os.environ.get("GEMINI_API_KEY") else "rules"
    if provider in ("gemini", "bedrock"):
        prompt = PROMPT.replace("{context}", json.dumps(ctx, ensure_ascii=False, indent=1))
        try:
            from src import macro_briefing as mb
            if provider == "gemini":
                text = mb._call_gemini(prompt, os.environ["GEMINI_API_KEY"])
                method = f"Gemini ({mb.GEMINI_MODEL})"
            else:
                text = mb._call_bedrock(prompt)
                method = f"Amazon Bedrock ({os.environ.get('BEDROCK_MODEL_ID', mb.BEDROCK_DEFAULT_MODEL)})"
            report.narrative, report.narrative_method = text, method
            return report
        except Exception as e:  # noqa: BLE001 - any provider failure falls back to rules
            report.narrative = rule_based_text(ctx) + f"\n\n> LLM call failed ({type(e).__name__}); rule-based text shown."
            report.narrative_method = "rules (LLM failed)"
            return report
    report.narrative, report.narrative_method = rule_based_text(ctx), "rules"
    return report


def report_url(ticker: str) -> str:
    return f"{APP_URL.rstrip('/')}/?ticker={ticker}"


def compact_summary(report: Report) -> dict:
    """Small JSON for MCP / Alexa / Fire TV: the headline numbers plus a spoken sentence."""
    c, v, rb, rg = report.ctx["company"], report.ctx["valuation"], report.ctx["rule_based"], report.ctx["macro_regime"]
    wk = report.ctx["technical"]["Weekly"]
    lv = rb["levels"]
    spoken = (f"{c['name']}, ticker {c['ticker']}, trades at {c['price']:,.2f} dollars. "
              f"The probability-weighted fair value is {v['expected_per_share']:,.0f} dollars, "
              f"{abs(v['expected_upside']) * 100:.0f} percent {'above' if v['expected_upside'] >= 0 else 'below'} the price, "
              f"with the {rg['regime']} regime weighting the scenarios. The weekly trend is {wk['ma_alignment']} "
              f"and the rule-based rating is {rb['rating']}. The full Excel, Word and dashboard report is in the "
              f"MacroPulse Equity Report tab.")
    if c["synthetic_data"]:
        spoken = "This is synthetic demo data. " + spoken
    return _clean({
        "status": "success", "ticker": c["ticker"], "name": c["name"], "price": c["price"], "as_of": c["as_of"],
        "rating": rb["rating"], "score": rb["score"], "basis": rb["basis"],
        "fair_value_prob_weighted": v["expected_per_share"], "upside_prob_weighted": v["expected_upside"],
        "dcf_base": v["dcf_base_per_share"],
        "scenarios": {k: {"per_share": s["per_share"], "prob": s["prob"]} for k, s in v["scenarios"].items()},
        "wacc": v["wacc"], "terminal_growth": v["terminal_growth"], "macro_regime": rg,
        "levels": {k: lv[k] for k in ("stop", "target1", "target2")}, "touch_prob_6m": lv["touch_prob_6m"],
        "touch_prob_6m_driftless": lv["touch_prob_6m_driftless"],
        "drift": {k: report.ctx["drift"].get(k) for k in ("source", "market_expected_return", "stock_expected_return")},
        "fomc_overlay": {x["scenario"]: {"per_share": x["per_share"], "wacc": x["wacc"]} for x in v["fomc_overlay"]},
        "pattern_screen": {k: report.screen[k] for k in ("bullish_count", "bearish_count", "bull_score",
                                                          "bear_score", "overall")},
        "weekly_trend": wk["ma_alignment"], "data_sources": report.ctx["data_sources"],
        "synthetic_data": c["synthetic_data"],
        "report_url": report_url(c["ticker"]),
        "alexa_spoken_response": spoken,
    })
