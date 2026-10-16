"""
validation/forward_log.py
─────────────────────────────────────────────────────────────────
Append-only, hash-chained log of what the live MacroPulse models say each month, recorded before the
outcome is known. It is the only out-of-sample evidence that cannot have been tuned after the fact:
docs/validation/preregistration.md says what is logged, when it is scored and what counts as success.

Each line of validation/forward_log/log.jsonl is one record:
  seq, created_at_utc, trigger, git_commit, code_dirty, model_fingerprint, as_of,
  snapshots   SHA-256 of the input data frames, which are stored next to the log (gzip, mtime 0)
  forecasts   regime, expected_return_12m, backtest_position, var, equity
  errors      component -> message, for anything that could not be produced (never filled in)
  prev_hash   hash of the previous record ("0" * 64 for the first)
  hash        SHA-256 of the canonical JSON of every other field

Editing, removing or reordering a record breaks the chain; CI checks the chain on every run and, on a
pull request, that the base branch's log is an unchanged prefix of the new one.

  python -m validation.forward_log append          # collect live forecasts and append one record
  python -m validation.forward_log verify [--base FILE]
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from validation import spec

ROOT = Path(__file__).resolve().parents[1]
GENESIS = "0" * 64


# ── hashing and the chain ─────────────────────────────────────────────────────
def json_safe(o: Any) -> Any:
    """Plain JSON types; NaN/inf become None so the canonical form never needs allow_nan."""
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (pd.Timestamp, dt.datetime, dt.date)):
        return o.isoformat()
    return o


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def record_hash(rec: dict) -> str:
    return sha256(canonical({k: v for k, v in rec.items() if k != "hash"}))


def read_log(path: Path) -> List[dict]:
    if not Path(path).exists():
        return []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def verify_chain(records: List[dict]) -> List[str]:
    """Problems with the chain; empty when every record is intact and in order."""
    errors, prev = [], GENESIS
    for i, r in enumerate(records):
        if r.get("seq") != i:
            errors.append(f"record {i}: seq is {r.get('seq')!r}, expected {i}")
        if r.get("prev_hash") != prev:
            errors.append(f"record {i}: prev_hash does not match record {i - 1}")
        if r.get("hash") != record_hash(r):
            errors.append(f"record {i}: hash does not match its contents (edited after it was written)")
        prev = r.get("hash")
    return errors


def verify_append_only(base: List[dict], records: List[dict]) -> List[str]:
    """Problems if `records` is not `base` with records added at the end."""
    if len(records) < len(base):
        return [f"{len(base) - len(records)} record(s) removed: the log had {len(base)}, now {len(records)}"]
    return [f"record {i} changed" for i, (a, b) in enumerate(zip(base, records)) if a.get("hash") != b.get("hash")]


def chain(records: List[dict], rec: dict) -> dict:
    """rec with seq, prev_hash and hash set as the next record after `records`."""
    out = json_safe({**rec, "seq": len(records), "prev_hash": records[-1]["hash"] if records else GENESIS})
    out.pop("hash", None)
    out["hash"] = record_hash(out)
    return out


def append_line(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(canonical(rec).decode() + "\n")


# ── provenance ────────────────────────────────────────────────────────────────
def model_fingerprint(root: Path = ROOT) -> str:
    h = hashlib.sha256()
    for f in spec.MODEL_FILES:
        h.update(f.encode() + b"\0" + (root / f).read_bytes() + b"\0")
    return h.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def git_commit() -> Dict[str, Any]:
    commit = os.environ.get("GITHUB_SHA") or _git("rev-parse", "HEAD") or None
    dirty = bool(_git("status", "--porcelain", "--", *spec.MODEL_FILES))
    return {"git_commit": commit, "code_dirty": dirty}


def _versions() -> Dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": platform.python_version()}
    for p in ("pandas", "numpy", "scikit-learn", "scipy", "yfinance"):
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = "not installed"
    return out


def frame_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(lineterminator="\n", float_format="%.10g").encode()


def write_snapshot(folder: Path, name: str, df: pd.DataFrame) -> str:
    """Store df as <name>.csv.gz (byte-identical for identical data) and return the SHA-256 of the CSV."""
    raw = frame_bytes(df)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{name}.csv.gz", "wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as gz:
            gz.write(raw)
    return sha256(raw)


def read_snapshot(path: Path) -> bytes:
    with gzip.open(path, "rb") as gz:
        return gz.read()


# ── what is logged ────────────────────────────────────────────────────────────
EQUITY_FIELDS = ("price", "as_of", "rating", "score", "fair_value_prob_weighted", "upside_prob_weighted",
                 "dcf_base", "scenarios", "wacc", "terminal_growth", "base_period", "growth_year1",
                 "fcf_margin", "implied_growth_year1", "implied_wacc", "scenarios_bracket_price", "synthetic_data")


def regime_forecast(srv) -> dict:
    out = srv.execute_get_macro_regime()
    return {k: out.get(k) for k in ("regime", "stress_score", "sp500", "vix", "credit_spread_source",
                                    "regime_win_rate_pct", "regime_months_in_history")}


def expected_return_forecast(macro: pd.DataFrame) -> dict:
    from src.macro_model import expected_return_model
    er = expected_return_model(macro)
    if er is None or er.empty:
        raise RuntimeError("expected_return_model returned no forecast (history too short or features missing)")
    last = er.iloc[-1]
    r = macro["sp500_ret_m"].dropna()
    return {"model_row": er.index[-1].date().isoformat(), "exp_simple_return": float(last["exp_ann_return"]),
            "exp_log_return": float(last["exp_log_return"]), "pred_std_log": last.get("pred_std"),
            "benchmark_prevailing_mean": float(np.expm1(r.mean() * 12)), "benchmark_n_months": int(len(r)),
            "benchmark": "expm1(12 x mean monthly S&P 500 log return, all months in the snapshot)"}


def backtest_position(macro: pd.DataFrame) -> dict:
    """Default dashboard strategy (regime + momentum + trend, threshold 0.5, long-only): the SPY weight it
    holds next month. target_w at the last row is the position applied to the following month."""
    from src.portfolio import TBILL_COLUMN, run_strategy_backtest
    bt = run_strategy_backtest(macro, use_regime=True, use_momentum=True, use_trend=True, threshold=0.5,
                               cost_bps=5.0, allow_short=False)
    last_row = bt.index[-1]
    tb = macro[TBILL_COLUMN].dropna() if TBILL_COLUMN in macro else pd.Series(dtype=float)
    return {"signal_row": last_row.date().isoformat(),
            "applies_to_month": (last_row + pd.DateOffset(months=1)).strftime("%Y-%m"),
            "target_weight": float(bt["target_w"].iloc[-1]),
            "signals": {k: int(bt[k].iloc[-1]) for k in ("sig_regime", "sig_momentum", "sig_trend")},
            "tbill_3m_pct": float(tb.iloc[-1]) if len(tb) else None,
            "config": "regime+momentum+trend, threshold 0.5, long-only, 5 bp"}


def var_forecast(srv) -> dict:
    out = {"ticker": spec.VAR_TICKER, "horizon_days": spec.VAR_HORIZON_DAYS, "paths": spec.VAR_PATHS,
           "method": "live MCP tool: GBM on 1y of daily log returns, seed 42"}
    for c in spec.VAR_CONFIDENCES:
        r = srv.execute_simulate_portfolio_risk(spec.VAR_TICKER, c, spec.VAR_HORIZON_DAYS, spec.VAR_PATHS)
        out[str(c)] = {"var_pct": r["var_pct"], "cvar_pct": r["cvar_pct"], "price": r["current_price"]}
    return out


def equity_forecast(srv, ticker: str) -> dict:
    out = srv.execute_get_equity_report(ticker)
    rec = {k: out.get(k) for k in EQUITY_FIELDS}
    rec["regime"] = (out.get("macro_regime") or {}).get("regime")
    return rec


def collect(srv, macro: pd.DataFrame, tickers=spec.EQUITY_TICKERS) -> tuple[dict, dict]:
    """(forecasts, errors). A component that fails is reported in errors and left out of forecasts."""
    forecasts: Dict[str, Any] = {"equity": {}}
    errors: Dict[str, str] = {}

    def run(name: str, fn: Callable[[], dict], into: Optional[dict] = None, key: Optional[str] = None):
        try:
            (into if into is not None else forecasts)[key or name] = fn()
        except Exception as e:                                   # recorded, never replaced by a guess
            errors[name] = f"{type(e).__name__}: {e}"[:500]

    run("regime", lambda: regime_forecast(srv))
    run("expected_return_12m", lambda: expected_return_forecast(macro))
    run("backtest_position", lambda: backtest_position(macro))
    run("var", lambda: var_forecast(srv))
    for t in tickers:
        run(f"equity:{t}", lambda t=t: equity_forecast(srv, t), forecasts["equity"], t)
    return forecasts, errors


def build_record(*, now: dt.datetime, as_of: pd.Timestamp, forecasts: dict, errors: dict,
                 snapshots: dict, trigger: str) -> dict:
    return {"schema": spec.SCHEMA_VERSION,
            "created_at_utc": now.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "trigger": trigger, **git_commit(), "model_fingerprint": model_fingerprint(),
            "versions": _versions(), "as_of": pd.Timestamp(as_of).date().isoformat(),
            "snapshots": snapshots, "forecasts": forecasts, "errors": errors,
            "preregistration": spec.PREREGISTRATION_DOC}


# ── commands ──────────────────────────────────────────────────────────────────
def _live_sources():
    """(MCP server module, dashboard macro frame, MCP macro frame, SPY 1y daily close)."""
    for var in ("SEC_USER_AGENT", "FRED_API_KEY"):        # an unset Actions secret arrives as ""
        if not os.environ.get(var):
            os.environ.pop(var, None)
    from src import alexa_mcp_server as srv
    from src.macro_data import load_macro
    srv._cached_fred(block=True)                          # requests never wait on FRED; this run must
    macro = load_macro()
    mcp_macro = srv._fetch_cached_macro_data()
    spy = srv._cached_download(spec.VAR_TICKER, "1y", timeout=15)
    spy = (spy["Close"] if "Close" in spy.columns else spy.iloc[:, 0]).rename("close").to_frame()
    return srv, macro, mcp_macro, spy


def cmd_append(log_path: Path, force: bool = False, now: Optional[dt.datetime] = None, sources=None) -> int:
    records = read_log(log_path)
    problems = verify_chain(records)
    if problems:
        print("Refusing to append to a broken log:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    srv, macro, mcp_macro, spy = sources or _live_sources()
    if bool(macro.get("_is_demo", pd.Series([False])).iloc[-1]):
        print("Live prices unavailable (synthetic demo frame); nothing logged.", file=sys.stderr)
        return 1
    from src.macro_model import as_of_date
    as_of = as_of_date(macro)
    if records and records[-1]["as_of"] == as_of.date().isoformat() and not force:
        print(f"A record for as-of {as_of.date()} already exists (seq {records[-1]['seq']}); nothing logged.")
        return 0
    now = now or dt.datetime.now(dt.timezone.utc)
    forecasts, errors = collect(srv, macro)
    if not any(k in forecasts for k in ("regime", "expected_return_12m", "var")) and not forecasts["equity"]:
        print("Every component failed; nothing logged:\n  " + "\n  ".join(f"{k}: {v}" for k, v in errors.items()),
              file=sys.stderr)
        return 1
    folder = log_path.parent / "snapshots" / f"{len(records):04d}_{as_of.date().isoformat()}"
    snaps = {"folder": folder.relative_to(ROOT).as_posix() if folder.is_relative_to(ROOT) else str(folder),
             "sha256": {name: write_snapshot(folder, name, df)
                        for name, df in (("macro_dashboard", macro), ("macro_mcp", mcp_macro), ("spy_1y_daily", spy))}}
    rec = chain(records, build_record(now=now, as_of=as_of, forecasts=forecasts, errors=errors, snapshots=snaps,
                                      trigger=os.environ.get("GITHUB_EVENT_NAME", "manual")))
    append_line(log_path, rec)
    print(f"Logged record {rec['seq']} (as-of {rec['as_of']}, {len(errors)} error(s)), hash {rec['hash'][:12]}")
    for k, v in errors.items():
        print(f"  error {k}: {v}")
    return 0


def cmd_verify(log_path: Path, base: Optional[Path] = None) -> int:
    records = read_log(log_path)
    problems = verify_chain(records)
    if base is not None:
        problems += verify_append_only(read_log(base), records)
    root = log_path.parent.parent.parent
    for r in records:                                     # stored snapshots must still match their hashes
        folder = root / r["snapshots"]["folder"]
        for name, h in r["snapshots"]["sha256"].items():
            p = folder / f"{name}.csv.gz"
            if not p.exists():
                problems.append(f"record {r['seq']}: snapshot {p} is missing")
            elif sha256(read_snapshot(p)) != h:
                problems.append(f"record {r['seq']}: snapshot {p} does not match its hash")
    if problems:
        print("Forward log check failed:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    print(f"Forward log OK: {len(records)} record(s), chain and snapshots intact.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("append")
    a.add_argument("--force", action="store_true", help="log even if the last record has the same as-of date")
    v = sub.add_parser("verify")
    v.add_argument("--base", type=Path, help="the base branch's log; it must be an unchanged prefix")
    ap.add_argument("--log", type=Path, default=ROOT / spec.LOG_PATH)
    args = ap.parse_args(argv)
    if args.cmd == "append":
        return cmd_append(args.log, force=args.force)
    return cmd_verify(args.log, args.base)


if __name__ == "__main__":
    sys.exit(main())
