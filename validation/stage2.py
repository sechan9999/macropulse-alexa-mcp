"""
validation/stage2.py — expected-return model (pre-registration H2)
─────────────────────────────────────────────────────────────────
  1. Pipeline check: Welch and Goyal (2008), Table 1 (annual log equity premium), re-computed on Amit
     Goyal's current predictor file for d/p, tms and dfy (plus d/y, e/p, tbl for context). Published
     values were transcribed from the paper (RFS 21(4), p. 1470, "Forecasts begin 20 years after sample"
     and "Forecasts begin 1965"); pass = every primary value within 0.5 percentage points.
  2. The product's Ridge model (src/macro_model.expected_return_model) on the frame rebuilt from raw
     data, from the first month all inputs exist (1976-07) and from the product's 2005 start:
     Campbell-Thompson OOS R² against the prevailing mean, Clark-West (Newey-West, 12 lags),
     Pesaran-Timmermann sign test, by decade and NBER recession/expansion.
  3. Leakage checks: the CI noise test, a look-ahead placebo (target windows moved 1-3 months earlier,
     i.e. a model allowed to see k months of its outcome) and a 1,000-draw permutation test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from src.macro_model import HORIZON, RIDGE_FEATURES, expected_return_model, oos_noise_correlation, walk_forward_ridge
from validation import data, frames, stats
from validation.results import save_csv, save_json
from validation.stage1 import prevailing_mean

STAGE = "stage2"

# Welch & Goyal (2008), Table 1, annual OOS R² (percent): (forecasts begin 20 years after sample, begin 1965)
PUBLISHED_OOS_R2 = {"d/p": (-2.06, -3.69), "tms": (-2.42, -2.96), "dfy": (-3.29, -4.15),
                    "d/y": (-1.93, -6.68), "e/p": (-1.78, -1.10), "tbl": (-3.37, -4.90)}
PUBLISHED_DATA_START = {"d/p": 1872, "tms": 1920, "dfy": 1919, "d/y": 1872, "e/p": 1872, "tbl": 1920}
PRIMARY = ("d/p", "tms", "dfy")
TOLERANCE_PP = 0.5
ADJUSTED = True     # the table's OOS R̄² is degrees-of-freedom adjusted; see paper_definition() in the results


# ── 1. Welch-Goyal replication ───────────────────────────────────────────────
def goyal_annual() -> pd.DataFrame:
    g = data.goyal("Annual").rename(columns=str.strip)
    g = g.set_index(g.columns[0])
    g.index = g.index.astype(int)
    num = g.apply(pd.to_numeric, errors="coerce")
    out = pd.DataFrame(index=num.index)
    # The current file has CRSP returns from 1926 only; Welch-Goyal (2008) used S&P 500 returns from 1871.
    # Before 1926 the annual return is rebuilt from the same file's index level and dividends.
    ret = num["ret"].fillna((num["price"] + num["d12"]) / num["price"].shift(1) - 1)
    out["premium"] = np.log1p(ret) - np.log1p(num["Rfree"])
    out["d/p"] = np.log(num["d12"]) - np.log(num["price"])
    out["d/y"] = np.log(num["d12"]) - np.log(num["price"].shift(1))
    out["e/p"] = np.log(num["e12"]) - np.log(num["price"])
    out["tms"] = num["lty"] - num["tbl"]
    out["dfy"] = num["BAA"] - num["AAA"]
    out["tbl"] = num["tbl"]
    return out


def gw_oos_r2(df: pd.DataFrame, var: str, first_forecast: int, end: int, start: int, adjusted: bool = False) -> float:
    """Expanding OLS of premium_t on x_{t-1} vs the expanding mean of premium, both from `start`+1.
    adjusted=True applies the degrees-of-freedom adjustment R̄² = R² - (1 - R²) k / (T - k - 1), k = 1."""
    y = df["premium"]
    x = df[var].shift(1)
    errs_m, errs_b = [], []
    for t in range(first_forecast, end + 1):
        tr = pd.concat([y, x], axis=1).loc[start + 1: t - 1].dropna()
        if len(tr) < 3 or pd.isna(x.get(t)) or pd.isna(y.get(t)):
            continue
        b, a = np.polyfit(tr.iloc[:, 1], tr.iloc[:, 0], 1)
        errs_m.append(y[t] - (a + b * x[t]))
        errs_b.append(y[t] - tr.iloc[:, 0].mean())
    em, eb = np.array(errs_m), np.array(errs_b)
    r2 = 1 - np.sum(em ** 2) / np.sum(eb ** 2)
    if adjusted:
        r2 = r2 - (1 - r2) * 1 / (len(em) - 2)
    return float(r2 * 100)


def paper_definition() -> str:
    """The paper's own definition of its OOS R² (Equation 6), extracted from the cached PDF for the record."""
    p = data.CACHE / "paper" / "welch_goyal_2008_rfs.pdf"
    if not p.exists():
        try:
            data.store("paper:welch_goyal_2008", data.WELCH_GOYAL_PDF, data._get(data.WELCH_GOYAL_PDF),
                       "paper/welch_goyal_2008_rfs.pdf", cache=True)
        except Exception as e:                                   # offline run: say so, never guess
            return f"paper not available in this run: {type(e).__name__}"
    from pypdf import PdfReader
    for i, page in enumerate(PdfReader(str(p)).pages):                 # the sentence that defines Equation (6)
        t = (page.extract_text() or "").replace("\n", " ")
        j = t.find("Our OOS statistics are computed as")
        if j != -1:
            return f"[Welch & Goyal 2008, page {i + 1} of the PDF] " + t[j: j + 420] + " ..."
    return "definition not found in the PDF text"


def replicate_goyal_welch() -> tuple[dict, pd.DataFrame]:
    df = goyal_annual()
    rows = []
    for var, (pub20, pub65) in PUBLISHED_OOS_R2.items():
        s0 = PUBLISHED_DATA_START[var]
        ours20 = gw_oos_r2(df, var, s0 + 20, 2005, s0, adjusted=ADJUSTED)
        ours65 = gw_oos_r2(df, var, 1965, 2005, s0, adjusted=ADJUSTED)
        ext65 = gw_oos_r2(df, var, 1965, int(df.index.max()), s0, adjusted=ADJUSTED)
        rows.append({"variable": var, "primary": var in PRIMARY, "published_20y": pub20, "ours_20y": ours20,
                     "diff_20y_pp": ours20 - pub20, "published_1965": pub65, "ours_1965": ours65,
                     "diff_1965_pp": ours65 - pub65, f"ours_1965_to_{int(df.index.max())}": ext65,
                     "unadjusted_20y": gw_oos_r2(df, var, s0 + 20, 2005, s0),
                     "unadjusted_1965": gw_oos_r2(df, var, 1965, 2005, s0)})
    t = pd.DataFrame(rows).set_index("variable")
    prim = t.loc[list(PRIMARY)]
    worst = float(np.nanmax(np.abs(prim[["diff_20y_pp", "diff_1965_pp"]].to_numpy())))
    return {"tolerance_pp": TOLERANCE_PP, "max_abs_diff_primary_pp": worst, "pass": worst <= TOLERANCE_PP,
            "adjusted_r2": ADJUSTED,
            "fixes_after_first_run": ["pre-1926 returns rebuilt from price and dividends (the file starts CRSP "
                                      "returns in 1926)", "degrees-of-freedom adjusted OOS R² as in the table"],
            "source": "Welch & Goyal (2008), RFS 21(4), Table 1, p. 1470; data: Goyal predictor file (Annual sheet)",
            "sample_end": 2005}, t


# ── 2. the product model ──────────────────────────────────────────────────────
def pesaran_timmermann(actual, forecast) -> dict:
    """Pesaran & Timmermann (1992) test that sign(forecast) predicts sign(actual)."""
    a, f = np.asarray(actual) > 0, np.asarray(forecast) > 0
    n = len(a)
    p_hat = np.mean(a == f)
    py, px = a.mean(), f.mean()
    p_star = py * px + (1 - py) * (1 - px)
    v_hat = p_star * (1 - p_star) / n
    v_star = ((2 * py - 1) ** 2 * px * (1 - px) / n + (2 * px - 1) ** 2 * py * (1 - py) / n
              + 4 * py * px * (1 - py) * (1 - px) / n ** 2)
    z = (p_hat - p_star) / np.sqrt(v_hat - v_star) if v_hat > v_star else np.nan
    return {"hit_rate": float(p_hat), "expected_by_chance": float(p_star), "z": float(z),
            "p_one_sided": float(1 - sps.norm.cdf(z)) if np.isfinite(z) else None, "n": n}


def evaluation_frame(start: str) -> pd.DataFrame:
    m = frames.build_macro(start)
    er = expected_return_model(m)
    d = er.dropna(subset=["realized_fwd_12m"]).copy()
    d["y"] = np.log1p(d["realized_fwd_12m"])
    d["model"] = d["exp_log_return"]
    d["bench"] = prevailing_mean(m["sp500_ret_m"]).reindex(d.index)
    usrec = data.monthly_last(data.fred_latest("USREC"))
    d["recession"] = usrec.reindex(d.index)
    return d.dropna(subset=["bench"])


def evaluate(d: pd.DataFrame) -> dict:
    cw = stats.clark_west(d["y"], d["model"], d["bench"], lags=HORIZON)
    return {"n": len(d), "first_origin": d.index.min(), "last_origin": d.index.max(),
            "oos_r2": stats.oos_r2(d["y"], d["model"], d["bench"]), "clark_west": cw,
            "pesaran_timmermann": pesaran_timmermann(d["y"] - d["bench"], d["model"] - d["bench"])}


def subperiods(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, mask in (("1980s", d.index.year < 1990), ("1990s", (d.index.year >= 1990) & (d.index.year < 2000)),
                       ("2000s", (d.index.year >= 2000) & (d.index.year < 2010)),
                       ("2010s", (d.index.year >= 2010) & (d.index.year < 2020)), ("2020+", d.index.year >= 2020),
                       ("recession", d["recession"] == 1), ("expansion", d["recession"] == 0)):
        s = d[mask]
        if len(s) < 24:
            rows.append({"period": name, "n": len(s), "oos_r2": None, "cw_stat": None, "cw_p": None,
                         "note": "fewer than 24 forecasts"})
            continue
        cw = stats.clark_west(s["y"], s["model"], s["bench"], lags=HORIZON)
        rows.append({"period": name, "n": len(s), "oos_r2": stats.oos_r2(s["y"], s["model"], s["bench"]),
                     "cw_stat": cw["stat"], "cw_p": cw["p_one_sided"], "note": ""})
    return pd.DataFrame(rows).set_index("period")


# ── 3. leakage checks ─────────────────────────────────────────────────────────
def _features(m: pd.DataFrame) -> list:
    return [c for c in RIDGE_FEATURES if c in m.columns and m[c].notna().any()]


def _oos_r2_for_target(m: pd.DataFrame, target: pd.Series) -> float:
    d = m.copy()
    d["y_fwd"] = target
    cols = _features(d)
    preds = walk_forward_ridge(d.dropna(subset=["sp500_ret_m"]), cols, "y_fwd", HORIZON)
    bench = prevailing_mean(d["sp500_ret_m"])
    both = pd.concat([preds.rename("p"), d["y_fwd"].rename("y"), bench.rename("b")], axis=1).dropna()
    return stats.oos_r2(both["y"], both["p"], both["b"])


def lookahead_placebo(m: pd.DataFrame) -> pd.DataFrame:
    """Target window starting k months EARLIER (k of its 12 months already observed at the forecast date).
    k = 0 is the real model; the rise in R² with k is what a k-month leak would add."""
    r = m["sp500_ret_m"]
    rows = []
    for k in range(0, 4):
        target = r.rolling(HORIZON).sum().shift(-(HORIZON - k))
        rows.append({"k_months_seen": k, "oos_r2": _oos_r2_for_target(m, target)})
    return pd.DataFrame(rows).set_index("k_months_seen")


def permutation_test(m: pd.DataFrame, n_perm: int = 1000, seed: int = 0) -> dict:
    """Shuffle the monthly returns (features unchanged), rebuild the 12-month target, refit the walk-forward:
    the share of shuffles scoring at least the real OOS R² is the p-value."""
    from joblib import Parallel, delayed

    actual = _oos_r2_for_target(m, m["sp500_ret_m"].rolling(HORIZON).sum().shift(-HORIZON))
    r = m["sp500_ret_m"].dropna()
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(r.to_numpy()) for _ in range(n_perm)]

    def one(p):
        mm = m.copy()
        mm.loc[r.index, "sp500_ret_m"] = p
        return _oos_r2_for_target(mm, mm["sp500_ret_m"].rolling(HORIZON).sum().shift(-HORIZON))

    null = np.array(Parallel(n_jobs=-1)(delayed(one)(p) for p in perms))
    return {"actual_oos_r2": actual, "n_permutations": n_perm, "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)), "p_value": float((1 + np.sum(null >= actual)) / (1 + n_perm))}


def run(n_perm: int = 1000) -> dict:
    out = {}
    rep, table = replicate_goyal_welch()
    save_csv(STAGE, "goyal_welch_replication", table)
    out["goyal_welch"] = rep
    (save_json(STAGE, "paper_oos_r2_definition", {"text": paper_definition()}))
    for label, start in (("full", frames.FIRST_MONTH["expected_return"]), ("product_2005", frames.PRODUCT_START)):
        d = evaluation_frame(start)
        save_csv(STAGE, f"forecasts_{label}", d[["model", "bench", "y", "recession"]])
        out[label] = evaluate(d)
        sp = subperiods(d)
        save_csv(STAGE, f"subperiods_{label}", sp)
        out[label]["subperiods"] = sp.to_dict(orient="index")
    m = frames.build_macro(frames.FIRST_MONTH["expected_return"])
    out["leakage"] = {"noise_correlation": oos_noise_correlation(lambda d, x: walk_forward_ridge(d, x)),
                      "noise_threshold": 0.10}
    lp = lookahead_placebo(m)
    save_csv(STAGE, "lookahead_placebo", lp)
    out["leakage"]["lookahead_placebo"] = lp["oos_r2"].to_dict()
    out["leakage"]["permutation"] = permutation_test(m, n_perm)
    save_json(STAGE, "summary", out)
    return out
