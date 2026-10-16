"""
validation/stage3.py — regime score (pre-registration H1)
─────────────────────────────────────────────────────────────────
  Primary H1: mean log realised S&P 500 volatility over the 21 sessions after a Risk-Off month end is
  higher than after a Risk-On month end (one-sided, stationary bootstrap, mean block 6, 10,000 draws).

  Also: next-month return by regime; NBER recessions (hit rate, false alarms, lead before each
  recession start); a Gaussian HMM (hmmlearn; states chosen by BIC and out-of-sample log-likelihood);
  regime durations and the month-to-month transition matrix; sensitivity to the z-score warm-up
  (24/36/48/60 months) and to an expanding GMM with 2-4 components (src/regimes.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.macro_model import RISK_OFF, RISK_ON, UNAVAILABLE
from validation import data, frames
from validation.results import save_csv, save_json

STAGE = "stage3"
BLOCK, REPS, SEED = 6, 10000, 0


def boot_diff(values: pd.Series, in_a: pd.Series, in_b: pd.Series, one_sided: bool = True,
              reps: int = REPS) -> dict:
    """mean(values | a) - mean(values | b), with a stationary bootstrap over months (pairs kept together).
    p = (1 + #centred bootstrap differences >= the observed one) / (1 + draws)   (H0: difference <= 0)."""
    from arch.bootstrap import StationaryBootstrap
    v = values.to_numpy(float)
    a, b = in_a.to_numpy(bool), in_b.to_numpy(bool)
    obs = v[a].mean() - v[b].mean()
    bs = StationaryBootstrap(BLOCK, np.arange(len(v)), seed=SEED)
    draws = []
    for (idx,), _ in bs.bootstrap(reps):
        aa, bb = a[idx], b[idx]
        if aa.any() and bb.any():
            draws.append(v[idx][aa].mean() - v[idx][bb].mean())
    d = np.asarray(draws)
    centred = d - d.mean()
    hits = np.sum(centred >= obs) if one_sided else np.sum(np.abs(centred) >= abs(obs))
    p = float((1 + hits) / (1 + len(d)))                   # never 0: the observed sample counts as one draw
    return {"difference": float(obs), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_value": p, "one_sided": one_sided, "n_a": int(a.sum()), "n_b": int(b.sum()), "reps": len(d)}


def regime_panel(min_periods=None, spread_src: str = "latest") -> pd.DataFrame:
    spread, slope = frames.fred_spread_slope(spread_src)
    m = frames.build_macro(frames.FIRST_MONTH["regime"], spread=spread, slope=slope, regime_min_periods=min_periods)
    out = frames.next_month_outcomes(data.yahoo("GSPC"), m.index)
    p = m[["regime", "regime_score", "credit_spread", "realized_vol_12m"]].join(out)
    p["recession"] = data.monthly_last(data.fred_latest("USREC")).reindex(p.index)
    p = p[p["regime"] != UNAVAILABLE].dropna(subset=["next_vol"])
    p["log_vol"] = np.log(p["next_vol"])
    return p


def h1(p: pd.DataFrame) -> dict:
    return boot_diff(p["log_vol"], p["regime"] == RISK_OFF, p["regime"] == RISK_ON)


def by_regime(p: pd.DataFrame) -> pd.DataFrame:
    g = p.groupby("regime").agg(months=("next_ret", "size"), mean_next_return=("next_ret", "mean"),
                                mean_next_vol=("next_vol", "mean"), share_up=("next_ret", lambda s: (s > 0).mean()))
    return g


def nber(p: pd.DataFrame) -> dict:
    off = p["regime"] == RISK_OFF
    rec = p["recession"] == 1
    usrec = data.monthly_last(data.fred_latest("USREC"))
    starts = usrec[(usrec == 1) & (usrec.shift(1) == 0)].index
    starts = [s for s in starts if p.index.min() + pd.DateOffset(months=12) <= s <= p.index.max()]
    leads = []
    for s in starts:
        win = p.loc[s - pd.DateOffset(months=12): s + pd.DateOffset(months=12)]
        first = win.index[win["regime"] == RISK_OFF]
        leads.append({"recession_start": s.date(),
                      "first_risk_off_within_12m": first[0].date() if len(first) else None,
                      # positive = Risk-Off before the recession start, negative = after it
                      "lead_months": int(round((s - first[0]).days / 30.44)) if len(first) else None})
    return {"recession_months": int(rec.sum()), "risk_off_months": int(off.sum()),
            "hit_rate": float(off[rec].mean()), "false_alarm_rate": float((~rec[off]).mean()),
            "risk_off_rate_in_expansions": float(off[~rec].mean()), "recessions": leads,
            "recessions_flagged": int(sum(l["lead_months"] is not None for l in leads)),
            "median_lead_months": float(np.median([l["lead_months"] for l in leads if l["lead_months"] is not None]))
            if any(l["lead_months"] is not None for l in leads) else None}


def durations_and_transitions(p: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lab = p["regime"]
    runs = (lab != lab.shift()).cumsum()
    lengths = lab.groupby(runs).agg(["first", "size"])
    dur = lengths.groupby("first")["size"].describe()[["count", "mean", "50%", "max"]]
    trans = pd.crosstab(lab.shift(), lab, normalize="index")
    return dur, trans


def hmm_comparison(p: pd.DataFrame, split: str = "2005-01-01") -> dict:
    from hmmlearn.hmm import GaussianHMM
    x = p[["credit_spread", "realized_vol_12m"]].dropna()
    train, test = x[x.index < split], x[x.index >= split]
    mu, sd = train.mean(), train.std()
    ztr, zte = ((train - mu) / sd).to_numpy(), ((test - mu) / sd).to_numpy()
    rows, models = [], {}
    for k in (2, 3, 4):
        m = GaussianHMM(n_components=k, covariance_type="full", n_iter=500, random_state=SEED).fit(ztr)
        ll = m.score(ztr)
        d = ztr.shape[1]
        n_params = k * (k - 1) + (k - 1) + k * d + k * d * (d + 1) / 2
        rows.append({"states": k, "train_loglik": ll, "bic": -2 * ll + n_params * np.log(len(ztr)),
                     "test_loglik_per_month": m.score(zte) / len(zte), "converged": bool(m.monitor_.converged)})
        models[k] = m
    t = pd.DataFrame(rows).set_index("states")
    k_bic = int(t["bic"].idxmin())
    m = models[k_bic]
    states = pd.Series(m.predict(((x - mu) / sd).to_numpy()), index=x.index)
    stress = int(np.argmax(m.means_[:, 0]))                       # highest credit-spread state
    hmm_off = states == stress
    lab = p["regime"].reindex(x.index)
    agree = pd.crosstab(lab, hmm_off.map({True: "HMM stress state", False: "other HMM states"}))
    return {"table": t.to_dict(orient="index"), "states_by_bic": k_bic,
            "states_by_test_loglik": int(t["test_loglik_per_month"].idxmax()),
            "risk_off_in_hmm_stress_state": float(hmm_off[lab == RISK_OFF].mean()),
            "hmm_stress_months_labelled_risk_off": float((lab[hmm_off] == RISK_OFF).mean()),
            "crosstab": agree.to_dict(), "train_end": split,
            "note": "The HMM is fitted on 1931-2004 in one pass, so its labels are not point-in-time; it is a "
                    "comparison of how the two methods partition the same months."}


def sensitivity(base: pd.DataFrame) -> dict:
    from src.regimes import fit_gmm_regimes
    out = {"z_warmup": {}, "gmm": {}}
    for mp in (24, 36, 48, 60):
        p = regime_panel(min_periods=mp)
        r = h1(p)
        out["z_warmup"][mp] = {"difference": r["difference"], "p_value": r["p_value"], "n_off": r["n_a"], "n_on": r["n_b"]}
    feat = base[["credit_spread", "realized_vol_12m"]].dropna()
    for k in (2, 3, 4):
        g = fit_gmm_regimes(feat, ["credit_spread", "realized_vol_12m"], n_regimes=k, min_train=120)
        lab = g["regime"].reindex(base.index)
        ok = lab.notna()
        occupied = lab[ok].value_counts().sort_index()
        hi = int(max(c for c, n in occupied.items() if n >= 12))     # highest-spread component with >= 12 months
        r = boot_diff(base["log_vol"][ok], (lab == hi)[ok], (lab == 0)[ok])
        out["gmm"][k] = {"difference_highest_minus_lowest_spread_component": r["difference"], "p_value": r["p_value"],
                         "n_high": r["n_a"], "n_low": r["n_b"], "highest_occupied_component": hi,
                         "months_per_component": {int(c): int(n) for c, n in occupied.items()}}
    return out


def run() -> dict:
    p = regime_panel()
    save_csv(STAGE, "regime_panel", p)
    out = {"sample": {"first": p.index.min(), "last": p.index.max(), "months": len(p)}}
    out["h1"] = h1(p)
    out["h1_product_2005"] = h1(p[p.index >= frames.PRODUCT_START])
    out["h1_alfred_spread"] = h1(regime_panel(spread_src="alfred"))
    ret = boot_diff(p["next_ret"], p["regime"] == RISK_OFF, p["regime"] == RISK_ON, one_sided=False)
    out["next_return_off_minus_on"] = ret
    br = by_regime(p)
    save_csv(STAGE, "by_regime", br)
    out["by_regime"] = br.to_dict(orient="index")
    out["nber"] = nber(p)
    dur, trans = durations_and_transitions(p)
    save_csv(STAGE, "durations", dur)
    save_csv(STAGE, "transition_matrix", trans)
    out["durations"] = dur.to_dict(orient="index")
    out["transition_matrix"] = trans.to_dict(orient="index")
    out["hmm"] = hmm_comparison(p)
    out["sensitivity"] = sensitivity(p)
    save_json(STAGE, "summary", out)
    return out
