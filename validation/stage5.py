"""
validation/stage5.py — risk model backtest (pre-registration H4)
─────────────────────────────────────────────────────────────────
  The live tool's VaR (src/alexa_mcp_server.execute_simulate_portfolio_risk): GBM fitted to the last
  252 daily SPY log returns, 5,000 paths, seed 42, horizon 21 sessions. Re-computed at every month end
  since 1994 with only the data before it, next to an i.i.d. bootstrap of the same 252 daily returns.

  Primary H4: Kupiec proportion-of-failures test at 95% (H0: exceedance rate = 5%).
  Also: Kupiec at 99%, Christoffersen independence and conditional coverage, the Basel traffic-light
  zone for 99% exceedances (binomial cumulative probability), and the Acerbi-Székely Z2 test of the
  expected shortfall with a p-value simulated from each month's own forecast distribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation import data, stats
from validation.results import save_csv, save_json

STAGE = "stage5"
WINDOW, HORIZON, PATHS, SEED = 252, 21, 5000, 42
CONFIDENCES = (95, 99)


def simulate(window: np.ndarray, method: str, rng: np.random.Generator) -> np.ndarray:
    """Simulated 21-session simple returns. gaussian: the live tool's GBM; bootstrap: i.i.d. resampling."""
    if method == "gaussian":
        shocks = rng.normal(window.mean(), window.std(ddof=1), (PATHS, HORIZON))
    else:
        shocks = window[rng.integers(0, len(window), (PATHS, HORIZON))]
    return np.expm1(shocks.sum(axis=1))


def forecasts(method: str) -> pd.DataFrame:
    px = data.yahoo("SPY").dropna()
    lr = np.log(px).diff().dropna()
    ends = lr.groupby(lr.index.to_period("M")).apply(lambda s: s.index[-1])
    rows = []
    for end in ends:
        hist = lr.loc[:end]
        pos = px.index.get_loc(end)
        if len(hist) < WINDOW or pos + HORIZON >= len(px):
            continue
        sims = simulate(hist.iloc[-WINDOW:].to_numpy(), method, np.random.default_rng(SEED))
        realised = float(px.iloc[pos + HORIZON] / px.iloc[pos] - 1)
        row = {"origin": end, "realised": realised}
        for c in CONFIDENCES:
            q = np.percentile(sims, 100 - c)
            row[f"var{c}"] = q
            row[f"es{c}"] = float(sims[sims <= q].mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("origin")


def z2_pvalue(f: pd.DataFrame, c: int, method: str, sims: int = 2000) -> dict:
    """Z2 on the realised outcomes, and its distribution when each month's outcome is drawn from that
    month's own forecast distribution (H0: the model is right). p = P(Z2_sim <= Z2_obs)."""
    alpha = 1 - c / 100
    L, v, e = -f["realised"], -f[f"var{c}"], -f[f"es{c}"]
    z_obs = stats.acerbi_szekely_z2(L, v, e, alpha)
    px = data.yahoo("SPY").dropna()
    lr = np.log(px).diff().dropna()
    rng = np.random.default_rng(SEED + 1)
    draws = np.empty((sims, len(f)))
    for j, end in enumerate(f.index):
        w = lr.loc[:end].iloc[-WINDOW:].to_numpy()
        if method == "gaussian":
            draws[:, j] = np.expm1(rng.normal(w.mean() * HORIZON, w.std(ddof=1) * np.sqrt(HORIZON), sims))
        else:
            draws[:, j] = np.expm1(w[rng.integers(0, len(w), (sims, HORIZON))].sum(axis=1))
    zs = np.array([stats.acerbi_szekely_z2(-draws[i], v, e, alpha) for i in range(sims)])
    return {"z2": z_obs, "p_value": float(np.mean(zs <= z_obs)), "simulations": sims}


def evaluate(f: pd.DataFrame, method: str) -> dict:
    out = {"months": len(f), "first": f.index.min(), "last": f.index.max()}
    for c in CONFIDENCES:
        p = 1 - c / 100
        hits = (f["realised"] < f[f"var{c}"]).to_numpy()
        out[f"var{c}"] = {"kupiec": stats.kupiec_pof(len(hits), int(hits.sum()), p),
                          "christoffersen": stats.christoffersen(hits, p),
                          "es_test": z2_pvalue(f, c, method)}
        if c == 99:
            out[f"var{c}"]["basel"] = stats.basel_zone(len(hits), int(hits.sum()), p)
    return out


def run() -> dict:
    out = {"method_notes": {"gaussian": "live tool: GBM on 252 daily log returns, 5,000 paths, seed 42",
                            "bootstrap": "i.i.d. resampling of the same 252 daily log returns, 5,000 paths"},
           "horizon_sessions": HORIZON}
    table = []
    for method in ("gaussian", "bootstrap"):
        f = forecasts(method)
        save_csv(STAGE, f"forecasts_{method}", f)
        e = evaluate(f, method)
        out[method] = e
        for c in CONFIDENCES:
            k, ch, es = e[f"var{c}"]["kupiec"], e[f"var{c}"]["christoffersen"], e[f"var{c}"]["es_test"]
            table.append({"method": method, "confidence": c, "months": k["n"], "exceedances": k["exceedances"],
                          "expected": k["expected"], "kupiec_p": k["p_value"], "christoffersen_ind_p": ch["p_ind"],
                          "conditional_coverage_p": ch["p_cc"], "es_z2": es["z2"], "es_p": es["p_value"],
                          "basel_zone": e[f"var{c}"].get("basel", {}).get("zone")})
    t = pd.DataFrame(table).set_index(["method", "confidence"])
    save_csv(STAGE, "comparison", t)
    save_json(STAGE, "summary", out)
    return out
