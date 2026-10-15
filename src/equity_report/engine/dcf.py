"""DCF valuation on free cash flow to the firm: FCFF = CFO - CapEx + interest expense x (1 - tax).
(CFO is after interest; adding after-tax interest back keeps the cash flow unlevered, consistent with
discounting at WACC and then subtracting net debt.)

The Excel model (render/excel.py) uses the *same* definitions and formulas, and a test checks that a
LibreOffice recalculation of the workbook reproduces these numbers:

  base        : trailing twelve months to the latest 10-Q (last 10-K if no 10-Q since), net debt at that date
  growth path : year-1 growth g_high decays to the terminal growth g by year N (default 10): the excess over g
                halves every `half_life` years, rescaled so year N lands exactly on g
                  g_t = g + (g_high - g) (phi^(t-1) - phi^(N-1)) / (1 - phi^(N-1)),  phi = 0.5^(1/half_life)
                (half_life None -> linear fade)
  FCF_t       = revenue_t x FCFF margin,  PV = FCF_t / (1 + WACC)^(t - s)
                s = years from the base period end to the price date (cash flows already partly elapsed)
  TV          = FCF_N x (1 + g) / (WACC - g),  EV = sum(PV) + TV / (1 + WACC)^(N - s)
  per share   = (EV - net debt) / shares outstanding
Sensitivity keeps the FCF path fixed and varies only WACC and g (same as the Excel sheet).
Reverse DCF solves for the year-1 growth (or the WACC) at which the model value equals the price.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from ..config import FALLBACK_RISK_FREE, GROWTH_CLIP, DCFAssumptions
from .regime import RegimeOverlay, overlay as make_overlay

SCENARIO_SHIFTS = {  # applied to the base case; growth deltas are scaled by scenario_growth_delta()
    "Bear": dict(d_growth=-1.0, margin_mult=0.8, d_wacc=+0.01),
    "Base": dict(d_growth=0.0, margin_mult=1.0, d_wacc=0.0),
    "Bull": dict(d_growth=+1.0, margin_mult=1.2, d_wacc=-0.01),
}


def scenario_growth_delta(g_high: float) -> float:
    """Bear/bull year-1 growth shift: 5pp, or 30% of the base growth when that is larger (a fixed ±5pp
    around 50% growth is far narrower than the real uncertainty), rounded to 0.5pp."""
    return max(0.05, round(0.3 * abs(g_high) / 0.005) * 0.005)
SENS_STEPS = np.array([-0.01, -0.005, 0.0, 0.005, 0.01])


def _reg_beta(stock_close: pd.Series, index_close: pd.Series, rule: str, n: int) -> tuple[float, int, float]:
    s = stock_close.resample(rule).last().pct_change()
    m = index_close.resample(rule).last().pct_change()
    df = pd.concat([s, m], axis=1, keys=["s", "m"]).dropna().iloc[-n:]
    if len(df) < 2:
        return float("nan"), int(len(df)), float("nan")
    cov = np.cov(df["s"], df["m"])
    return float(cov[0, 1] / cov[1, 1]), int(len(df)), float(np.corrcoef(df["s"], df["m"])[0, 1] ** 2)


def estimate_beta(stock_close: pd.Series, index_close: pd.Series, years: int = 2) -> dict:
    """Regression beta vs the S&P 500, Blume-adjusted (0.67 x raw + 0.33).

    Raw beta is the mean of the 2-year weekly regression and the 5-year monthly regression (the usual
    data-vendor definition) when at least 36 months are available; a single 2-year weekly window is noisy
    for high-volatility stocks (one AI-driven year can push it well above the multi-year value)."""
    wk, n_w, r2 = _reg_beta(stock_close, index_close, "W-FRI", 52 * years)
    mo, n_m, _ = _reg_beta(stock_close, index_close, "ME", 60)
    if n_w < 26:
        return {"raw": 1.0, "adjusted": 1.0, "n_weeks": n_w, "r2": None, "raw_weekly": None,
                "raw_monthly": None, "n_months": n_m}
    use_m = n_m >= 36 and np.isfinite(mo)
    raw = (wk + mo) / 2 if use_m else wk
    return {"raw": float(raw), "adjusted": float(0.67 * raw + 0.33), "n_weeks": n_w, "r2": r2,
            "raw_weekly": float(wk), "raw_monthly": float(mo) if use_m else None, "n_months": n_m if use_m else 0}


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
    stub: float = 0.0          # years from the base period end to the price date
    half_life: float | None = None   # growth decay half-life in years; None -> linear fade


def growth_path(g_high: float, g_term: float, n: int, half_life: float | None = None) -> np.ndarray:
    if n <= 1:
        return np.array([g_high])
    t = np.arange(n)
    if not half_life:
        return g_high + (g_term - g_high) * t / (n - 1)
    phi = 0.5 ** (1 / half_life)
    return g_term + (g_high - g_term) * (phi ** t - phi ** (n - 1)) / (1 - phi ** (n - 1))


def value(inp: DCFInputs) -> dict:
    if inp.wacc <= inp.terminal_growth:
        raise ValueError("WACC must exceed terminal growth (terminal value diverges)")
    g = growth_path(inp.growth_high, inp.terminal_growth, inp.years, inp.half_life)
    rev = inp.revenue0 * np.cumprod(1 + g)
    fcf = rev * inp.fcf_margin
    disc = 1 / (1 + inp.wacc) ** (np.arange(1, inp.years + 1) - inp.stub)
    pv = fcf * disc
    tv = fcf[-1] * (1 + inp.terminal_growth) / (inp.wacc - inp.terminal_growth)
    pv_tv = tv * disc[-1]
    ev = pv.sum() + pv_tv
    ps = (ev - inp.net_debt) / inp.shares
    return {"growth": g, "revenue": rev, "fcf": fcf, "discount": disc, "pv": pv, "tv": tv, "pv_tv": pv_tv,
            "ev": ev, "equity": ev - inp.net_debt, "per_share": ps, "upside": ps / inp.price - 1,
            "tv_share_of_ev": pv_tv / ev if ev else np.nan}


def implied_growth(inp: DCFInputs) -> float | None:
    """Year-1 growth (same fade, margin and WACC) at which the DCF value equals the share price."""
    def f(g):
        return value(DCFInputs(**{**asdict(inp), "growth_high": g}))["per_share"] - inp.price
    try:
        return float(brentq(f, -0.5, 3.0)) if inp.fcf_margin > 0 and f(-0.5) < 0 < f(3.0) else None
    except ValueError:
        return None


def implied_wacc(inp: DCFInputs) -> float | None:
    """WACC (same cash flows and terminal growth) at which the DCF value equals the share price."""
    lo = inp.terminal_growth + 0.002

    def f(w):
        return value(DCFInputs(**{**asdict(inp), "wacc": w}))["per_share"] - inp.price
    try:
        return float(brentq(f, lo, 0.40)) if f(lo) > 0 > f(0.40) else None
    except ValueError:
        return None


def sensitivity(inp: DCFInputs, fcf_path: np.ndarray, waccs, gs) -> pd.DataFrame:
    t = np.arange(1, len(fcf_path) + 1) - inp.stub
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


def touch_probability(price: float, target: float, sigma_annual: float, years: float, drift: float = 0.0) -> float:
    """Probability that a log-normal price touches `target` at least once within `years`.

    `drift` is the annual drift of log price (nu). First passage of Brownian motion with drift to the
    barrier b = ln(target / price):
        up   (b > 0): Phi((nu T - b) / s) + exp(2 nu b / sigma^2) Phi((-b - nu T) / s)
        down (b < 0): Phi((b - nu T) / s) + exp(2 nu b / sigma^2) Phi((b + nu T) / s),   s = sigma sqrt(T)
    With nu = 0 this is the reflection-principle value 2 [1 - Phi(|b| / s)]."""
    if price <= 0 or target <= 0 or sigma_annual <= 0:
        return float("nan")
    b, s, nu, var = np.log(target / price), sigma_annual * np.sqrt(years), drift, sigma_annual ** 2
    if b == 0:
        return 1.0
    if b > 0:
        p = norm.cdf((nu * years - b) / s) + np.exp(2 * nu * b / var) * norm.cdf((-b - nu * years) / s)
    else:
        p = norm.cdf((b - nu * years) / s) + np.exp(2 * nu * b / var) * norm.cdf((b + nu * years) / s)
    return float(min(1.0, max(0.0, p)))


# FOMC shocks from the Multi-Asset Sync & FOMC Shock tab (tab 13), translated into discount-rate inputs.
# Illustrative, assumption-based (like tab 13's own shocks): not estimated from data.
FOMC_RATE_SHOCKS = {
    #                          title                                       d_rf     d_erp    d_spread d_g
    "hawkish_50bps":         ("Hawkish Surprise (+50 bps Hike)",          +0.0050, +0.0025, +0.0025, 0.0),
    "dovish_50bps":          ("Dovish Pivot (-50 bps Cut)",               -0.0050, -0.0025, -0.0010, 0.0),
    "stagflation_inversion": ("Stagflation & Curve Inversion Shock",      +0.0025, +0.0075, +0.0120, -0.0050),
    "liquidity_cascade":     ("March 2020 Correlated Liquidity Crunch",   -0.0025, +0.0150, +0.0200, 0.0),
}
FOMC_NOTE = ("Tab 13 FOMC scenarios mapped to risk-free / ERP / credit-spread / terminal-growth shifts "
             "(illustrative assumptions, not estimated). FCF path held fixed, as in the sensitivity grid.")


def _tab13_price_shock(ticker: str, scenario: str) -> float | None:
    """The illustrative single-name price shock tab 13 applies to this ticker (None if unavailable)."""
    try:
        from src.brokerage_sync import BrokeragePortfolio
        return BrokeragePortfolio({ticker: 1.0}).simulate_fomc_shock(scenario)["total_pnl_pct"] / 100
    except Exception:  # noqa: BLE001 - reference column only
        return None


def fomc_overlay(inp: "DCFInputs", fcf_path: np.ndarray, *, risk_free: float, beta: float, erp: float,
                 cost_of_debt: float, tax_rate: float, weight_equity: float, ticker: str | None = None) -> list[dict]:
    """Value per share at each FOMC scenario's (WACC, g) point, to overlay on the WACC x g grid."""
    out = []
    for key, (title, d_rf, d_erp, d_sp, d_g) in FOMC_RATE_SHOCKS.items():
        ke = risk_free + d_rf + beta * (erp + d_erp)
        kd = (cost_of_debt + d_rf + d_sp) * (1 - tax_rate)
        w = weight_equity * ke + (1 - weight_equity) * kd
        g = inp.terminal_growth + d_g
        ps = float(sensitivity(inp, fcf_path, [w], [g]).iloc[0, 0])
        out.append({"scenario": key, "title": title, "d_rf": d_rf, "d_erp": d_erp, "d_spread": d_sp, "d_g": d_g,
                    "wacc": w, "d_wacc": w - inp.wacc, "terminal_growth": g, "per_share": ps,
                    "upside": ps / inp.price - 1 if np.isfinite(ps) else float("nan"),
                    "tab13_price_shock": _tab13_price_shock(ticker, key) if ticker else None})
    return out


def build(bundle: dict, a: DCFAssumptions | None = None, regime: RegimeOverlay | None = None) -> dict:
    a = a or DCFAssumptions()
    ov = regime or make_overlay(None)
    fin: pd.DataFrame = bundle["financials"].copy()
    px = bundle["prices"]["close"]
    price, shares = float(bundle["last_price"]), float(bundle["shares"])
    notes: list[str] = []

    def fcff(cfo, capex, interest):
        return cfo - (0 if pd.isna(capex) else capex) + (0 if pd.isna(interest) else abs(interest)) * (1 - a.tax_rate)

    fin["fcf"] = [fcff(r.operating_cf, r.capex, r.interest_expense) for r in fin.itertuples()]
    fin["fcf_margin"] = fin["fcf"] / fin["revenue"]
    rev = fin["revenue"].dropna()
    n_yrs = len(rev) - 1
    cagr = (rev.iloc[-1] / rev.iloc[0]) ** (1 / n_yrs) - 1 if n_yrs >= 1 and rev.iloc[0] > 0 else 0.03

    # Base period: trailing twelve months to the latest 10-Q when there is one, else the last fiscal year.
    t12 = bundle.get("ttm") or None
    last = fin.iloc[-1]
    if t12 and np.isfinite(t12.get("revenue", np.nan)):
        base_rev, base_end = float(t12["revenue"]), str(t12["period_end"])
        base_label = f"TTM to {base_end}"
        ttm_fcf = (fcff(t12["operating_cf"], t12.get("capex"), t12.get("interest_expense"))
                   if np.isfinite(t12.get("operating_cf", np.nan)) else np.nan)
        ttm_margin = ttm_fcf / base_rev if np.isfinite(ttm_fcf) else np.nan
        recent_growth = t12.get("ytd_growth", np.nan)
        growth_src = (f"revenue growth, {t12['ytd_months']}-month year-to-date vs same period last year (10-Q)"
                      if np.isfinite(recent_growth) else None)
    else:
        base_rev, base_end = float(rev.iloc[-1]), str(last.get("fy_end", fin.index[-1]))
        base_label, ttm_margin, recent_growth, growth_src = f"FY{fin.index[-1]}", np.nan, np.nan, None
    if a.growth_high is not None:
        g_high, growth_src = a.growth_high, "user input"
    else:
        if not growth_src:
            recent_growth, growth_src = cagr, f"{n_yrs}-year historical revenue CAGR"
        g_high = float(np.clip(recent_growth, *GROWTH_CLIP))
        if not np.isclose(g_high, recent_growth):
            notes.append(f"Year-1 growth clipped to {g_high:.0%} from {recent_growth:.1%} ({growth_src}); "
                         "override it if the data supports more.")
    if a.fcf_margin is not None:
        margin, margin_src = a.fcf_margin, "user input"
    elif np.isfinite(ttm_margin):
        margin, margin_src = float(ttm_margin), f"trailing-twelve-month FCFF margin to {base_end}"
    else:
        margin, margin_src = float(fin["fcf_margin"].dropna().iloc[-3:].mean()), "3-year average FCFF margin"
    if margin <= 0:
        notes.append("FCFF margin is <= 0 (heavy capex cycle or losses): DCF is low-confidence; "
                     "lean on the relative-value bands.")
    as_of = pd.Timestamp(bundle.get("as_of") or px.index[-1])
    stub = float(np.clip((as_of - pd.Timestamp(base_end)).days / 365.25, 0.0, 1.0))

    rf = a.risk_free if a.risk_free is not None else bundle.get("risk_free_live")
    rf_source = "user input" if a.risk_free is not None else ("live 10Y Treasury" if rf is not None else "fallback")
    if rf is None:
        rf = FALLBACK_RISK_FREE
        notes.append(f"Live 10Y yield unavailable: risk-free rate set to {FALLBACK_RISK_FREE:.2%}.")

    beta_info = estimate_beta(px, bundle["index"]["close"])
    beta = a.beta if a.beta is not None else beta_info["adjusted"]
    erp = a.equity_risk_premium + ov.erp_shift
    ke = rf + beta * erp
    bs = t12 if t12 and np.isfinite(t12.get("debt", np.nan)) and np.isfinite(t12.get("cash", np.nan)) else last
    bs_date = base_end if bs is t12 else str(last.get("fy_end", fin.index[-1]))
    debt = float(bs.get("debt") or 0) if pd.notna(bs.get("debt")) else 0.0
    cash = float(bs.get("cash") or 0) if pd.notna(bs.get("cash")) else 0.0
    interest = t12.get("interest_expense") if t12 and np.isfinite(t12.get("interest_expense", np.nan)) \
        else last.get("interest_expense")
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

    inp = DCFInputs(base_rev, margin, g_high, a.terminal_growth, wacc, a.projection_years,
                    debt - cash, shares, price, stub, a.growth_half_life)
    base = value(inp)
    if base["tv_share_of_ev"] > 0.8:
        notes.append(f"Terminal value is {base['tv_share_of_ev']:.0%} of EV: highly sensitive to WACC and g.")

    sens = sensitivity(inp, base["fcf"], wacc + SENS_STEPS, a.terminal_growth + SENS_STEPS)
    fomc = fomc_overlay(inp, base["fcf"], risk_free=rf, beta=beta, erp=erp, cost_of_debt=kd, tax_rate=a.tax_rate,
                        weight_equity=w_e, ticker=None if bundle.get("is_synthetic") else bundle.get("ticker"))

    scen = {}
    dg = scenario_growth_delta(g_high)
    for k, s0 in SCENARIO_SHIFTS.items():
        s = {**s0, "d_growth": s0["d_growth"] * dg}
        v = value(DCFInputs(**{**asdict(inp), "growth_high": g_high + s["d_growth"],
                               "fcf_margin": margin * s["margin_mult"], "wacc": wacc + s["d_wacc"]}))
        scen[k] = {"per_share": float(v["per_share"]), "upside": float(v["upside"]), "prob": ov.probs[k], **s}
    expected = sum(s["per_share"] * s["prob"] for s in scen.values())
    neutral_expected = sum(s["per_share"] * p for s, p in zip(scen.values(), (0.25, 0.5, 0.25)))

    # Reverse DCF: what the market price implies, holding everything else.
    g_impl, w_impl = implied_growth(inp), implied_wacc(inp)
    bracket = scen["Bear"]["per_share"] <= price <= scen["Bull"]["per_share"]
    rev_txt = (f"Reverse DCF: the price implies year-1 growth of {g_impl:.1%} (model {g_high:.1%}) at this WACC"
               if g_impl is not None else "Reverse DCF: no year-1 growth in -50%..300% reproduces the price")
    rev_txt += (f", or a WACC of {w_impl:.2%} (model {wacc:.2%}) at this growth." if w_impl is not None else ".")
    notes.append(rev_txt)
    if not bracket:
        notes.append("The bear-bull scenario range does not contain the share price: the model disagrees with the "
                     "market by more than its own scenarios allow, so the rating gives valuation half weight.")

    bands = {}
    pe = bundle.get("pe_history")
    eps = bundle.get("eps_ttm") or bundle.get("eps_latest") or 0
    if pe is not None and len(pe.dropna()) > 60 and eps > 0:
        m = pe[(pe > 0) & (pe < pe.quantile(0.99))].dropna()
        q = m.quantile([0.1, 0.5, 0.9])
        bands["P/E"] = {"low": float(q.iloc[0] * eps), "mid": float(q.iloc[1] * eps), "high": float(q.iloc[2] * eps),
                        "multiple_q10_50_90": [float(x) for x in q],
                        "eps_used": float(eps), "eps_basis": "TTM" if bundle.get("eps_ttm") else "last fiscal year"}
        if q.iloc[2] > 4 * q.iloc[0]:
            notes.append(f"P/E band is very wide ({q.iloc[0]:.0f}x-{q.iloc[2]:.0f}x): GAAP EPS was distorted in "
                         "some years (acquisition amortisation, one-offs or splits); treat it as indicative only.")
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
                        "weight_equity": w_e, "wacc": wacc, "growth_high_used": g_high, "growth_source": growth_src,
                        "revenue_cagr_hist": float(cagr), "fcf_margin_used": margin, "fcf_margin_source": margin_src,
                        "base_period": base_label, "base_period_end": base_end, "base_revenue": base_rev,
                        "stub_years": stub, "balance_sheet_date": bs_date,
                        "implied_growth": g_impl, "implied_wacc": w_impl, "scenario_growth_delta": dg},
        "regime": ov.as_dict(),
        "inputs": asdict(inp), "base": base, "sensitivity": sens, "fomc_overlay": fomc, "scenarios": scen,
        "expected_per_share": float(expected), "expected_upside": float(expected / price - 1),
        "scenarios_bracket_price": bool(bracket), "ttm": t12,
        "neutral_weight_expected": float(neutral_expected),
        "bands": bands, "range_52w": {"low": float(px.iloc[-252:].min()), "high": float(px.iloc[-252:].max())},
        "sigma_annual": sigma, "history": fin, "debt": debt, "cash": cash, "market_cap": mcap, "notes": notes,
    }
