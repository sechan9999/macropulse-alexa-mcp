"""DCF valuation (FCFF proxy: FCF = cash from operations - capital expenditures).

The Excel model (render/excel.py) uses the *same* definitions and formulas, and a test checks that a
LibreOffice recalculation of the workbook reproduces these numbers:

  growth path : year-1 growth g_high fades linearly to the terminal growth g by year N
  FCF_t       = revenue_t x FCF margin,  PV = FCF_t / (1 + WACC)^t     (end-of-year discounting)
  TV          = FCF_N x (1 + g) / (WACC - g),  EV = sum(PV) + TV / (1 + WACC)^N
  per share   = (EV - net debt) / diluted shares
Sensitivity keeps the FCF path fixed and varies only WACC and g (same as the Excel sheet).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..config import FALLBACK_RISK_FREE, DCFAssumptions
from .regime import RegimeOverlay, overlay as make_overlay

SCENARIO_SHIFTS = {  # applied to the base case
    "Bear": dict(d_growth=-0.05, margin_mult=0.8, d_wacc=+0.01),
    "Base": dict(d_growth=0.0, margin_mult=1.0, d_wacc=0.0),
    "Bull": dict(d_growth=+0.05, margin_mult=1.2, d_wacc=-0.01),
}
SENS_STEPS = np.array([-0.01, -0.005, 0.0, 0.005, 0.01])


def estimate_beta(stock_close: pd.Series, index_close: pd.Series, years: int = 2) -> dict:
    """Weekly-return regression beta, Blume-adjusted (0.67 x raw + 0.33)."""
    s = stock_close.resample("W-FRI").last().pct_change()
    m = index_close.resample("W-FRI").last().pct_change()
    df = pd.concat([s, m], axis=1, keys=["s", "m"]).dropna().iloc[-52 * years:]
    if len(df) < 26:
        return {"raw": 1.0, "adjusted": 1.0, "n_weeks": int(len(df)), "r2": None}
    cov = np.cov(df["s"], df["m"])
    raw = cov[0, 1] / cov[1, 1]
    return {"raw": float(raw), "adjusted": float(0.67 * raw + 0.33), "n_weeks": int(len(df)),
            "r2": float(np.corrcoef(df["s"], df["m"])[0, 1] ** 2)}


@dataclass
class DCFInputs:
    revenue0: float
    fcf_margin: float
    growth_high: float
    terminal_growth: float
    wacc: float
    years: int
    net_debt: float
    shares: float
    price: float


def growth_path(g_high: float, g_term: float, n: int) -> np.ndarray:
    return np.array([g_high + (g_term - g_high) * i / (n - 1) for i in range(n)]) if n > 1 else np.array([g_high])


def value(inp: DCFInputs) -> dict:
    if inp.wacc <= inp.terminal_growth:
        raise ValueError("WACC must exceed terminal growth (terminal value diverges)")
    g = growth_path(inp.growth_high, inp.terminal_growth, inp.years)
    rev = inp.revenue0 * np.cumprod(1 + g)
    fcf = rev * inp.fcf_margin
    disc = 1 / (1 + inp.wacc) ** np.arange(1, inp.years + 1)
    pv = fcf * disc
    tv = fcf[-1] * (1 + inp.terminal_growth) / (inp.wacc - inp.terminal_growth)
    pv_tv = tv * disc[-1]
    ev = pv.sum() + pv_tv
    ps = (ev - inp.net_debt) / inp.shares
    return {"growth": g, "revenue": rev, "fcf": fcf, "discount": disc, "pv": pv, "tv": tv, "pv_tv": pv_tv,
            "ev": ev, "equity": ev - inp.net_debt, "per_share": ps, "upside": ps / inp.price - 1,
            "tv_share_of_ev": pv_tv / ev if ev else np.nan}


def sensitivity(inp: DCFInputs, fcf_path: np.ndarray, waccs, gs) -> pd.DataFrame:
    t = np.arange(1, len(fcf_path) + 1)
    vals = np.full((len(waccs), len(gs)), np.nan)
    for i, w in enumerate(waccs):
        disc = 1 / (1 + w) ** t
        for j, g in enumerate(gs):
            if w > g:
                ev = (fcf_path * disc).sum() + fcf_path[-1] * (1 + g) / (w - g) * disc[-1]
                vals[i, j] = (ev - inp.net_debt) / inp.shares
    out = pd.DataFrame(vals, index=[round(w, 4) for w in waccs], columns=[round(g, 4) for g in gs])
    out.index.name, out.columns.name = "WACC", "g"
    return out


def touch_probability(price: float, target: float, sigma_annual: float, years: float) -> float:
    """Driftless log-normal probability that price touches target at least once within `years`
    (reflection principle). A distance-vs-volatility yardstick, not a directional forecast."""
    if price <= 0 or target <= 0 or sigma_annual <= 0:
        return float("nan")
    z = abs(np.log(target / price)) / (sigma_annual * np.sqrt(years))
    return float(min(1.0, 2 * (1 - norm.cdf(z))))


def build(bundle: dict, a: DCFAssumptions | None = None, regime: RegimeOverlay | None = None) -> dict:
    a = a or DCFAssumptions()
    ov = regime or make_overlay(None)
    fin: pd.DataFrame = bundle["financials"].copy()
    px = bundle["prices"]["close"]
    price, shares = float(bundle["last_price"]), float(bundle["shares"])
    notes: list[str] = []

    fin["fcf"] = fin["operating_cf"] - fin["capex"].fillna(0)
    fin["fcf_margin"] = fin["fcf"] / fin["revenue"]
    rev = fin["revenue"].dropna()
    n_yrs = len(rev) - 1
    cagr = (rev.iloc[-1] / rev.iloc[0]) ** (1 / n_yrs) - 1 if n_yrs >= 1 and rev.iloc[0] > 0 else 0.03
    g_high = a.growth_high if a.growth_high is not None else float(np.clip(cagr, -0.05, 0.20))
    margin = a.fcf_margin if a.fcf_margin is not None else float(fin["fcf_margin"].dropna().iloc[-3:].mean())
    if margin <= 0:
        notes.append("3-year average FCF margin is <= 0 (heavy capex cycle or losses): DCF is low-confidence; "
                     "lean on the relative-value bands.")

    rf = a.risk_free if a.risk_free is not None else bundle.get("risk_free_live")
    rf_source = "user input" if a.risk_free is not None else ("live 10Y Treasury" if rf is not None else "fallback")
    if rf is None:
        rf = FALLBACK_RISK_FREE
        notes.append(f"Live 10Y yield unavailable: risk-free rate set to {FALLBACK_RISK_FREE:.2%}.")

    beta_info = estimate_beta(px, bundle["index"]["close"])
    beta = a.beta if a.beta is not None else beta_info["adjusted"]
    erp = a.equity_risk_premium + ov.erp_shift
    ke = rf + beta * erp
    last = fin.iloc[-1]
    debt = float(last.get("debt") or 0) if pd.notna(last.get("debt")) else 0.0
    cash = float(last.get("cash") or 0) if pd.notna(last.get("cash")) else 0.0
    interest = last.get("interest_expense")
    if a.cost_of_debt is not None:
        kd = a.cost_of_debt
    elif interest is not None and pd.notna(interest) and debt > 0:
        kd = float(np.clip(abs(interest) / debt, 0.03, 0.10))
    else:
        kd = rf + 0.015
    kd_after = kd * (1 - a.tax_rate)
    mcap = price * shares
    w_e = mcap / (mcap + debt) if mcap + debt else 1.0
    wacc = w_e * ke + (1 - w_e) * kd_after

    inp = DCFInputs(float(rev.iloc[-1]), margin, g_high, a.terminal_growth, wacc, a.projection_years,
                    debt - cash, shares, price)
    base = value(inp)
    if base["tv_share_of_ev"] > 0.8:
        notes.append(f"Terminal value is {base['tv_share_of_ev']:.0%} of EV: highly sensitive to WACC and g.")

    sens = sensitivity(inp, base["fcf"], wacc + SENS_STEPS, a.terminal_growth + SENS_STEPS)

    scen = {}
    for k, s in SCENARIO_SHIFTS.items():
        v = value(DCFInputs(**{**asdict(inp), "growth_high": g_high + s["d_growth"],
                               "fcf_margin": margin * s["margin_mult"], "wacc": wacc + s["d_wacc"]}))
        scen[k] = {"per_share": float(v["per_share"]), "upside": float(v["upside"]), "prob": ov.probs[k], **s}
    expected = sum(s["per_share"] * s["prob"] for s in scen.values())
    neutral_expected = sum(s["per_share"] * p for s, p in zip(scen.values(), (0.25, 0.5, 0.25)))

    bands = {}
    pe = bundle.get("pe_history")
    if pe is not None and len(pe.dropna()) > 60 and bundle.get("eps_latest", 0) > 0:
        m = pe[(pe > 0) & (pe < pe.quantile(0.99))].dropna()
        q = m.quantile([0.1, 0.5, 0.9])
        eps = bundle["eps_latest"]
        bands["P/E"] = {"low": float(q.iloc[0] * eps), "mid": float(q.iloc[1] * eps), "high": float(q.iloc[2] * eps),
                        "multiple_q10_50_90": [float(x) for x in q]}
    eq_hist = fin["total_equity"].dropna() if "total_equity" in fin else pd.Series(dtype=float)
    if len(eq_hist) and eq_hist.iloc[-1] > 0:
        bvps = eq_hist.iloc[-1] / shares
        pb = px / bvps
        if len(pb) > 60:
            q = pb.iloc[-756:].quantile([0.1, 0.5, 0.9])
            bands["P/B"] = {"low": float(q.iloc[0] * bvps), "mid": float(q.iloc[1] * bvps),
                            "high": float(q.iloc[2] * bvps), "multiple_q10_50_90": [float(x) for x in q],
                            "note": "3y price range / latest book value per share"}
    sigma = float(np.log(px).diff().iloc[-252:].std() * np.sqrt(252))

    return {
        "assumptions": {**asdict(a), "risk_free_used": rf, "risk_free_source": rf_source, "beta_used": beta,
                        "beta_est": beta_info, "erp_base": a.equity_risk_premium, "erp_used": erp,
                        "cost_of_equity": ke, "cost_of_debt_used": kd, "cost_of_debt_after_tax": kd_after,
                        "weight_equity": w_e, "wacc": wacc, "growth_high_used": g_high,
                        "revenue_cagr_hist": float(cagr), "fcf_margin_used": margin},
        "regime": ov.as_dict(),
        "inputs": asdict(inp), "base": base, "sensitivity": sens, "scenarios": scen,
        "expected_per_share": float(expected), "expected_upside": float(expected / price - 1),
        "neutral_weight_expected": float(neutral_expected),
        "bands": bands, "range_52w": {"low": float(px.iloc[-252:].min()), "high": float(px.iloc[-252:].max())},
        "sigma_annual": sigma, "history": fin, "debt": debt, "cash": cash, "market_cap": mcap, "notes": notes,
    }
