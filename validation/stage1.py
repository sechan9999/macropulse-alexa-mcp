"""
validation/stage1.py — data integrity (pre-registration, Stage 1)
─────────────────────────────────────────────────────────────────
  1. ALFRED: when each monthly value was first published, how much it was revised, and how often the
     offline pipeline's RELEASE_LAG_DAYS approximation (src/data_fred.py) picks a value that was not yet
     public, or a different vintage, at a month end.
  2. The product model run twice: credit spread from today's FRED values (as the product does: row m
     holds the Moody's average for month m, published in the first days of m+1) and from the values
     known at each month end (ALFRED).
  3. The snapshot manifest: every committed file matches its SHA-256.
  4. Schema and range checks with pandera: unique increasing dates, missing rates, value ranges,
     outliers (|daily log return| z-score > 6, listed, not failed).
  5. Prices from two sources: SPY total return (Yahoo, adjusted) vs the Ken French market return
     (Mkt-RF + RF), and Ken French RF vs the product's T-bill cash rate (^IRX); every month whose
     returns differ by more than 5 bp is listed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_fred import RELEASE_LAG_DAYS, to_monthly_point_in_time
from validation import data
from validation.results import save_csv, save_json

STAGE = "stage1"
SERIES_COL = {"BAA": "baa", "AAA": "aaa", "CPIAUCSL": "cpi", "UNRATE": "unrate", "INDPRO": "indpro",
              "TB3MS": "tb3ms"}


# ── 1. ALFRED ─────────────────────────────────────────────────────────────────
INDEX_SERIES = {"CPIAUCSL", "INDPRO"}      # levels are rebased between vintages; compare growth rates


def _in_vintage(a: pd.DataFrame, date, when) -> float:
    r = a[(a["date"] == date) & (a["realtime_start"] <= when) & (a["realtime_end"] > when)]
    return float(r["value"].iloc[-1]) if len(r) else np.nan


def first_release(sid: str) -> pd.DataFrame:
    """Per observation month: first publication date, first-release value, latest value, and for index
    series the month-over-month growth as first published and as now published (same vintage each)."""
    a = data.alfred(sid).dropna(subset=["value"])
    first = a.sort_values("realtime_start").groupby("date").first()
    latest = a[a["realtime_end"] >= pd.Timestamp("2262-01-01")].set_index("date")["value"]
    out = pd.DataFrame({"first_published": first["realtime_start"], "first_value": first["value"],
                        "latest_value": latest})
    out["lag_days"] = (out["first_published"] - (out.index + pd.offsets.MonthEnd(0))).dt.days
    if sid in INDEX_SERIES:
        prev_first = [_in_vintage(a, d - pd.DateOffset(months=1), v) for d, v in out["first_published"].items()]
        out["first_mom_pct"] = (out["first_value"] / np.array(prev_first) - 1) * 100
        out["latest_mom_pct"] = (out["latest_value"] / out["latest_value"].shift(1) - 1) * 100
    return out


def known_at_month_end(sid: str, with_obs: bool = False, yoy: bool = False):
    """Value of the latest observation in the vintage in force at each month end, labelled by that month
    (row m = what a reader could see at the end of month m). yoy=True gives the 12-month % change of that
    observation within the same vintage. Month ends before ALFRED's first vintage use the first vintage
    with each value published one month after its observation month."""
    a = data.alfred(sid).dropna(subset=["value"])
    first_vintage = a["realtime_start"].min()
    months = pd.date_range(a["date"].min(), a["realtime_start"].max(), freq="MS")
    rows = []
    early = a[a["realtime_start"] == first_vintage].set_index("date")["value"]
    for m in months:
        d = m + pd.offsets.MonthEnd(0)
        if d < first_vintage:
            known = early[early.index <= m - pd.DateOffset(months=1)]
            approx = True
        else:
            v = a[(a["realtime_start"] <= d) & (a["realtime_end"] > d)]
            known = v.set_index("date")["value"].sort_index()
            approx = False
        if len(known):
            val = float(known.iloc[-1])
            if yoy:
                base = known.get(known.index[-1] - pd.DateOffset(months=12))
                val = (val / float(base) - 1) * 100 if base is not None and pd.notna(base) else np.nan
            rows.append((m, val, known.index[-1], approx))
    df = pd.DataFrame(rows, columns=["date", "value", "obs_month", "before_first_vintage"]).set_index("date")
    return df if with_obs else df["value"].rename(sid)


def release_lag_check() -> pd.DataFrame:
    rows = []
    for sid, col in SERIES_COL.items():
        fr = first_release(sid)
        live = fr[fr["first_published"] > pd.Timestamp("1997-01-01")]       # after ALFRED's first vintage
        assumed = RELEASE_LAG_DAYS[col]
        if sid in INDEX_SERIES:
            rev = (fr["latest_mom_pct"] - fr["first_mom_pct"]).abs().dropna()
            measure = "month-over-month growth, pp"
        else:
            rev = (fr["latest_value"] - fr["first_value"]).abs()
            measure = "level, percentage points"
        rows.append({"series": sid, "n_obs_with_real_time_dates": len(live),
                     "median_lag_days": float(live["lag_days"].median()), "max_lag_days": float(live["lag_days"].max()),
                     "assumed_lag_days": assumed,
                     "share_published_later_than_assumed": float((live["lag_days"] > assumed).mean()),
                     "revision_measure": measure, "share_revised": float((rev > 1e-6).mean()),
                     "mean_abs_revision": float(rev.mean()), "max_abs_revision": float(rev.max())})
    return pd.DataFrame(rows).set_index("series")


def approximation_vs_vintage() -> pd.DataFrame:
    """How often RELEASE_LAG_DAYS on today's data gives a different month-end value than ALFRED."""
    raw = {SERIES_COL[s]: data.fred_latest(s) for s in SERIES_COL}
    approx = to_monthly_point_in_time(raw)
    approx.index = approx.index.to_period("M").to_timestamp()
    rows = []
    for sid, col in SERIES_COL.items():
        idx = sid in INDEX_SERIES
        k = known_at_month_end(sid, with_obs=True, yoy=idx)
        k = k[~k["before_first_vintage"]]
        a = (approx[col] / approx[col].shift(12) - 1) * 100 if idx else approx[col]
        both = pd.concat([a.rename("approx"), k["value"].rename("vintage")], axis=1).dropna()
        diff = (both["approx"] - both["vintage"]).abs()
        rows.append({"series": sid, "measure": "12-month % change, pp" if idx else "level",
                     "months_compared": len(both), "share_different": float((diff > 1e-6).mean()),
                     "mean_abs_difference": float(diff.mean()), "max_abs_difference": float(diff.max())})
    return pd.DataFrame(rows).set_index("series")


# ── 2. the product model, latest vs vintage credit spread ─────────────────────
def model_twice() -> dict:
    from src.macro_model import expected_return_model
    from validation import frames, stats

    out = {}
    for start in (frames.FIRST_MONTH["expected_return"], frames.PRODUCT_START):
        runs = {}
        for src in ("latest", "alfred"):
            spread, slope = frames.fred_spread_slope(src)
            m = frames.build_macro(start, spread=spread, slope=slope)
            er = expected_return_model(m)
            runs[src] = (m, er)
        (ml, el), (mv, ev) = runs["latest"], runs["alfred"]
        both = pd.concat([ml["regime"].rename("latest"), mv["regime"].rename("alfred")], axis=1).dropna()
        both = both[(both["latest"] != "Unavailable ⚪") & (both["alfred"] != "Unavailable ⚪")]
        res = {"regime_months": len(both), "regime_label_agreement": float((both["latest"] == both["alfred"]).mean()),
               "regime_score_correlation": float(ml["regime_score"].corr(mv["regime_score"]))}
        for name, (m, er) in (("latest", runs["latest"]), ("alfred", runs["alfred"])):
            d = er.dropna(subset=["realized_fwd_12m"]).copy()
            y = np.log1p(d["realized_fwd_12m"])
            bench = prevailing_mean(m["sp500_ret_m"]).reindex(d.index)
            ok = bench.notna()
            res[f"oos_r2_{name}"] = stats.oos_r2(y[ok], d["exp_log_return"][ok], bench[ok])
            res[f"n_forecasts_{name}"] = int(ok.sum())
        p = pd.concat([el["exp_log_return"].rename("l"), ev["exp_log_return"].rename("v")], axis=1).dropna()
        res["forecast_correlation"] = float(p["l"].corr(p["v"]))
        res["mean_abs_forecast_difference_log"] = float((p["l"] - p["v"]).abs().mean())
        out[f"start_{start[:4]}"] = res
    return out


def prevailing_mean(monthly_log_ret: pd.Series) -> pd.Series:
    """12 x mean of every monthly log return up to and including the forecast month (the benchmark)."""
    r = monthly_log_ret.dropna()
    return (r.expanding().mean() * 12).rename("bench")


# ── 4. schema and range checks ────────────────────────────────────────────────
RANGES = {"BAA": (0, 25), "AAA": (0, 25), "T10Y2Y": (-5, 5), "USREC": (0, 1), "CPIAUCSL": (0, 1000),
          "UNRATE": (0, 30), "INDPRO": (0, 300), "TB3MS": (0, 25), "DGS10": (0, 25), "NFCI": (-5, 10)}
YAHOO_RANGES = {"GSPC": (1, 20000), "SPY": (1, 2000), "VIX": (5, 100), "TNX": (0, 25), "IRX": (-1, 25),
                "VBMFX": (0.1, 50)}


def schema_checks() -> tuple[pd.DataFrame, pd.DataFrame]:
    import pandera.pandas as pa

    rows, outliers = [], []

    def run(name, df, schema, max_missing):
        missing = float(df["value"].isna().mean())
        try:
            schema.validate(df, lazy=True)
            fails = 0
            detail = ""
        except pa.errors.SchemaErrors as e:
            fc = e.failure_cases
            fails = int(len(fc))
            detail = "; ".join(sorted({str(c) for c in fc["check"].astype(str)}))[:300]
        rows.append({"source": name, "rows": len(df), "first": df["date"].min(), "last": df["date"].max(),
                     "missing_rate": missing, "missing_ok": missing <= max_missing, "failed_checks": fails,
                     "failures": detail})

    def schema_for(lo, hi):
        return pa.DataFrameSchema(
            {"date": pa.Column("datetime64[ns]", [pa.Check(lambda s: s.is_monotonic_increasing, element_wise=False,
                                                           name="increasing"),
                                                  pa.Check(lambda s: ~s.duplicated(), name="unique")]),
             "value": pa.Column(float, pa.Check.in_range(lo, hi), nullable=True)})

    for sid, (lo, hi) in RANGES.items():
        s = data.fred_latest(sid).astype(float)
        run(f"fred:{sid}", s.reset_index().rename(columns={sid: "value"}), schema_for(lo, hi), 0.05)
    for name, (lo, hi) in YAHOO_RANGES.items():
        s = data.yahoo(name)
        df = s.reset_index().rename(columns={name: "value"})
        run(f"yahoo:{name}", df, schema_for(lo, hi), 0.0)
        if name in ("GSPC", "SPY"):
            r = np.log(s).diff().dropna()
            z = (r - r.mean()) / r.std()
            for d, v in r[z.abs() > 6].items():
                outliers.append({"source": f"yahoo:{name}", "date": d.date(), "log_return": float(v),
                                 "z": float(z[d])})
    kf = data.ken_french().reset_index()
    for col, (lo, hi) in {"Mkt-RF": (-0.5, 0.5), "RF": (-0.001, 0.02)}.items():
        run(f"kenfrench:{col}", kf[["date", col]].rename(columns={col: "value"}), schema_for(lo, hi), 0.0)
    return pd.DataFrame(rows).set_index("source"), pd.DataFrame(outliers)


# ── 5. two price sources ──────────────────────────────────────────────────────
def price_reconciliation(threshold_bp: float = 5.0) -> tuple[dict, pd.DataFrame]:
    from src.portfolio import rf_monthly

    spy = data.monthly_last(data.yahoo("SPY")).pct_change()
    kf = data.ken_french()
    mkt = (kf["Mkt-RF"] + kf["RF"]).rename("kf_market")
    m = pd.concat([spy.rename("spy_total_return"), mkt, kf["RF"].rename("kf_rf")], axis=1).dropna()
    m = m[m.index < pd.Timestamp.today().normalize().replace(day=1)]          # complete months only
    m["diff_bp"] = (m["spy_total_return"] - m["kf_market"]) * 1e4
    irx = pd.DataFrame({"tbill_3m": data.monthly_last(data.yahoo("IRX"))})
    rf = np.expm1(rf_monthly(irx)).rename("product_cash_rate")
    m = m.join(rf, how="left")
    m["rf_diff_bp"] = (m["product_cash_rate"] - m["kf_rf"]) * 1e4
    over = m[m["diff_bp"].abs() > threshold_bp]
    summary = {"months": len(m), "first": m.index.min(), "last": m.index.max(), "threshold_bp": threshold_bp,
               "months_over_threshold": len(over), "share_over_threshold": len(over) / len(m),
               "mean_diff_bp": float(m["diff_bp"].mean()), "sd_diff_bp": float(m["diff_bp"].std()),
               "max_abs_diff_bp": float(m["diff_bp"].abs().max()),
               "correlation": float(m["spy_total_return"].corr(m["kf_market"])),
               "annualised_tracking_difference_pct": float(m["diff_bp"].mean() * 12 / 100),
               "rf_months": int(m["rf_diff_bp"].notna().sum()),
               "rf_mean_diff_bp": float(m["rf_diff_bp"].mean()), "rf_max_abs_diff_bp": float(m["rf_diff_bp"].abs().max()),
               "note": "SPY tracks the S&P 500 (after fees); Ken French's market is the CRSP value-weighted "
                       "universe of NYSE/AMEX/NASDAQ stocks. They are different portfolios, so monthly gaps "
                       "above 5 bp are expected; the product's cash rate is the 3-month bill, Ken French's RF "
                       "the 1-month bill."}
    return summary, m


def run() -> dict:
    out = {"manifest_problems": data.verify_manifest(), "manifest_files": len(data.read_manifest())}
    lag = release_lag_check()
    save_csv(STAGE, "alfred_release_lags", lag)
    approx = approximation_vs_vintage()
    save_csv(STAGE, "release_lag_approximation_vs_alfred", approx)
    out["alfred"] = {"release_lags": lag.to_dict(orient="index"), "approximation": approx.to_dict(orient="index")}
    out["model_latest_vs_alfred"] = model_twice()
    checks, outliers = schema_checks()
    save_csv(STAGE, "schema_checks", checks)
    save_csv(STAGE, "return_outliers", outliers)
    out["schema"] = {"sources": len(checks), "failed_sources": checks.index[checks["failed_checks"] > 0].tolist(),
                     "missing_rate_exceeded": checks.index[~checks["missing_ok"]].tolist(),
                     "outliers_listed": len(outliers)}
    summary, months = price_reconciliation()
    save_csv(STAGE, "spy_vs_kenfrench_monthly", months)
    save_csv(STAGE, "spy_vs_kenfrench_over_5bp", months[months["diff_bp"].abs() > 5])
    out["prices"] = summary
    save_json(STAGE, "summary", out)
    return out
