"""
validation/score_forward_log.py
─────────────────────────────────────────────────────────────────
Scores every forward-log forecast whose outcome window has closed, using daily closes fetched after the
fact, and writes validation/forward_log/scores.csv (one row per forecast) and summary.md.

Rules, from docs/validation/preregistration.md:
  * Outcome windows start at the record's as-of close. A record written after the next session opened
    (13:30 UTC) is "late" and never scored.
  * A record is "primary" when its as-of close is the last session of a calendar month. Only primary
    records enter the hypothesis tests; others (such as the pilot record) are scored and shown, not tested.
  * Records are grouped by model fingerprint: a code change starts a new sample.
  * No hypothesis test is run below the pre-registered minimum sample (spec.MIN_N); the summary shows
    counts only. The tests themselves are built and checked in Stages 2-5 of the validation plan.

  python -m validation.score_forward_log
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from validation import spec
from validation.forward_log import ROOT, read_log, verify_chain

OPEN_UTC = pd.Timedelta(hours=13, minutes=30)      # 9:30 New York in summer; the earliest the open can be


# ── windows on a daily close series ───────────────────────────────────────────
def close_on(closes: pd.Series, day: pd.Timestamp) -> Optional[float]:
    s = closes.loc[:day]
    return float(s.iloc[-1]) if len(s) else None


def next_session(closes: pd.Series, day: pd.Timestamp) -> Optional[pd.Timestamp]:
    after = closes.index[closes.index > day]
    return after[0] if len(after) else None


def is_primary(closes: pd.Series, as_of: pd.Timestamp) -> Optional[bool]:
    """True when as_of is the last session of its month; None until the next session is known."""
    nxt = next_session(closes, as_of)
    return None if nxt is None else (nxt.year, nxt.month) != (as_of.year, as_of.month)


def is_late(created_at_utc: str, closes: pd.Series, as_of: pd.Timestamp) -> Optional[bool]:
    nxt = next_session(closes, as_of)
    if nxt is None:
        return None
    created = pd.Timestamp(created_at_utc).tz_convert("UTC").tz_localize(None)
    return created >= nxt.normalize() + OPEN_UTC


def days_window(closes: pd.Series, as_of: pd.Timestamp, n: int) -> Optional[pd.Series]:
    """Closes from as_of through the n-th session after it, or None while fewer than n have happened."""
    start = closes.loc[:as_of]
    after = closes.loc[closes.index > as_of]
    if not len(start) or len(after) < n:
        return None
    return pd.concat([start.iloc[-1:], after.iloc[:n]])


def months_window(closes: pd.Series, as_of: pd.Timestamp, months: int) -> Optional[tuple]:
    """(start close, end close, end date) for as_of + months, or None until a session after the end exists."""
    end = as_of + pd.DateOffset(months=months)
    if not len(closes) or closes.index.max() <= end:
        return None
    s0, s1 = close_on(closes, as_of), close_on(closes, end)
    return None if s0 is None or s1 is None else (s0, s1, closes.loc[:end].index[-1])


def month_return(closes: pd.Series, month: str) -> Optional[float]:
    """Log return over the calendar month 'YYYY-MM' (last close of the prior month to its last close)."""
    p = pd.Period(month, "M")
    if not len(closes) or closes.index.max() <= p.end_time:
        return None
    prev, last = close_on(closes, p.start_time - pd.Timedelta(days=1)), close_on(closes, p.end_time)
    return None if prev is None or last is None else float(np.log(last / prev))


# ── one row per forecast ──────────────────────────────────────────────────────
def _row(rec, component, key, forecast, status, outcome=None, horizon_end=None, **detail) -> dict:
    return {"seq": rec["seq"], "as_of": rec["as_of"], "model_fingerprint": rec["model_fingerprint"][:12],
            "component": component, "key": key, "forecast": forecast, "outcome": outcome,
            "horizon_end": None if horizon_end is None else pd.Timestamp(horizon_end).date().isoformat(),
            "status": status, **detail}


def score_record(rec: dict, prices: Callable[[str], pd.Series]) -> List[dict]:
    rows: List[dict] = []
    f = rec.get("forecasts", {})
    as_of = pd.Timestamp(rec["as_of"])
    idx = prices(spec.OUTCOME_INDEX)
    late = is_late(rec["created_at_utc"], idx, as_of)
    primary = is_primary(idx, as_of)
    base = {"primary": primary, "late": late}
    if late:
        return [_row(rec, "record", "", None, "late (written after the next open)", **base)]

    if "regime" in f:
        w = days_window(idx, as_of, spec.REGIME_HORIZON_DAYS)
        if w is None:
            rows.append(_row(rec, "regime", f["regime"]["regime"], f["regime"]["stress_score"], "pending", **base))
        else:
            lr = np.log(w).diff().dropna()
            rows.append(_row(rec, "regime", f["regime"]["regime"], f["regime"]["stress_score"], "scored",
                             outcome=float(lr.std(ddof=1) * np.sqrt(252)), horizon_end=w.index[-1],
                             next_return_log=float(lr.sum()), **base))

    if "expected_return_12m" in f:
        e = f["expected_return_12m"]
        w = months_window(idx, as_of, spec.EXPECTED_RETURN_HORIZON_MONTHS)
        if w is None:
            rows.append(_row(rec, "expected_return", "S&P 500 12m", e["exp_simple_return"], "pending", **base))
        else:
            realized = w[1] / w[0] - 1
            rows.append(_row(rec, "expected_return", "S&P 500 12m", e["exp_simple_return"], "scored",
                             outcome=realized, horizon_end=w[2], benchmark=e["benchmark_prevailing_mean"],
                             sq_err_model=(realized - e["exp_simple_return"]) ** 2,
                             sq_err_benchmark=(realized - e["benchmark_prevailing_mean"]) ** 2, **base))

    if "backtest_position" in f:
        b = f["backtest_position"]
        r = month_return(idx, b["applies_to_month"])
        if r is None:
            rows.append(_row(rec, "backtest_position", b["applies_to_month"], b["target_weight"], "pending", **base))
        else:
            rf = np.log1p((b.get("tbill_3m_pct") or 0.0) / 100) / 12
            rows.append(_row(rec, "backtest_position", b["applies_to_month"], b["target_weight"], "scored",
                             outcome=r, horizon_end=pd.Period(b["applies_to_month"], "M").end_time,
                             strategy_excess_log=b["target_weight"] * (r - rf), **base))

    if "var" in f:
        v = f["var"]
        spy = prices(v["ticker"])
        w = days_window(spy, as_of, v["horizon_days"])
        for c in spec.VAR_CONFIDENCES:
            q = v.get(str(c))
            if q is None:
                continue
            if w is None:
                rows.append(_row(rec, f"var_{c}", v["ticker"], q["var_pct"], "pending", **base))
            else:
                realized = float(w.iloc[-1] / w.iloc[0] - 1)
                rows.append(_row(rec, f"var_{c}", v["ticker"], q["var_pct"], "scored", outcome=realized * 100,
                                 horizon_end=w.index[-1], exceeded=bool(realized * 100 < q["var_pct"]), **base))

    bench = None
    for t, e in (f.get("equity") or {}).items():
        if bench is None:
            bench = prices(spec.TOTAL_RETURN_BENCHMARK)
        start = pd.Timestamp(e.get("as_of") or rec["as_of"]).normalize()
        stock = prices(t)
        w, wb = (months_window(stock, start, spec.EQUITY_HORIZON_MONTHS),
                 months_window(bench, start, spec.EQUITY_HORIZON_MONTHS))
        if w is None or wb is None:
            rows.append(_row(rec, "equity", t, e.get("upside_prob_weighted"), "pending", rating=e.get("rating"), **base))
        else:
            excess = (w[1] / w[0] - 1) - (wb[1] / wb[0] - 1)
            rows.append(_row(rec, "equity", t, e.get("upside_prob_weighted"), "scored", outcome=excess,
                             horizon_end=w[2], rating=e.get("rating"), **base))
    return rows


# ── summary ───────────────────────────────────────────────────────────────────
def summarize(scores: pd.DataFrame, n_records: int) -> str:
    lines = ["# Forward log summary", "",
             f"Records: {n_records}. Rules: `{spec.PREREGISTRATION_DOC}`.", "",
             "Only primary records (as-of close = last session of a month, written before the next open) "
             "count toward the pre-registered minimum sample. No hypothesis test runs below it.", ""]
    if scores.empty:
        return "\n".join(lines + ["No forecasts yet.", ""])
    lines += ["| model | component | primary scored | pending | minimum n | status |", "|---|---|---|---|---|---|"]
    comps = {"regime": "regime", "expected_return": "expected_return", "backtest_position": "backtest_position",
             "var_95": "var_95", "var_99": "var_99", "equity": None}
    for (fp, comp), g in scores.groupby(["model_fingerprint", "component"], sort=True):
        if comp not in comps:
            continue
        n = int(((g["status"] == "scored") & (g["primary"] == True)).sum())   # noqa: E712 (None-aware)
        pend = int((g["status"] == "pending").sum())
        need = spec.MIN_N.get(comps[comp]) if comps[comp] else None
        status = ("descriptive only (5 tickers; cross-sectional test is Stage 6)" if need is None
                  else "test due" if n >= need else "collecting")
        lines.append(f"| `{fp}` | {comp} | {n} | {pend} | {need if need else '-'} | {status} |")
    v = scores[scores["component"].isin(["var_95", "var_99"]) & (scores["status"] == "scored")]
    if len(v):
        lines += ["", "VaR exceedances so far (all scored records, descriptive):"]
        for c, g in v.groupby("component"):
            lines.append(f"- {c}: {int(g['exceeded'].sum())} of {len(g)}")
    return "\n".join(lines + [""])


def yf_prices(start: pd.Timestamp) -> Callable[[str], pd.Series]:
    import yfinance as yf
    cache: Dict[str, pd.Series] = {}

    def get(symbol: str) -> pd.Series:
        if symbol not in cache:
            raw = yf.download(symbol, start=(start - pd.Timedelta(days=10)).date().isoformat(), auto_adjust=True,
                              progress=False, multi_level_index=False, timeout=20)
            c = (raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]).dropna() if len(raw) else pd.Series(dtype=float)
            if getattr(c.index, "tz", None) is not None:
                c.index = c.index.tz_localize(None)
            cache[symbol] = c
        return cache[symbol]
    return get


def main(argv=None, prices: Optional[Callable[[str], pd.Series]] = None, root: Path = ROOT) -> int:
    records = read_log(root / spec.LOG_PATH)
    problems = verify_chain(records)
    if problems:
        print("Forward log chain is broken; not scoring:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    rows: List[dict] = []
    if records:
        prices = prices or yf_prices(min(pd.Timestamp(r["as_of"]) for r in records))
        for r in records:
            rows += score_record(r, prices)
    scores = pd.DataFrame(rows)
    out = root / spec.SCORES_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out, index=False, lineterminator="\n")
    (root / spec.SUMMARY_PATH).write_text(summarize(scores, len(records)), encoding="utf-8")
    print(f"Scored {int((scores['status'] == 'scored').sum()) if len(scores) else 0} forecast(s) "
          f"from {len(records)} record(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
