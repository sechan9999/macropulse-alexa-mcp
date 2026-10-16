"""
validation/data.py
─────────────────────────────────────────────────────────────────
Raw data for the historical validation (docs/validation/preregistration.md, Stages 1-6).

`python -m validation.run fetch` downloads every source once and records it in
validation/data/raw/manifest.json: name, URL (API keys removed), retrieval time (UTC), SHA-256, size,
and whether the file is committed. Small files are committed under validation/data/raw/ so every later
run reads the same bytes. Large or redistributable-only-by-link files (SEC bulk data, the Welch-Goyal
paper) stay in validation/data/cache/ (git-ignored); the manifest still pins their hashes.

Stages never download: they read through `load_*` below and fail loudly if a file is missing.
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import io
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "validation" / "data" / "raw"
CACHE = ROOT / "validation" / "data" / "cache"
MANIFEST = RAW / "manifest.json"

FRED_API = "https://api.stlouisfed.org/fred/series/observations"
# Series the live model reads (BAA, AAA, T10Y2Y), the offline research pipeline reads (src/data_fred.py),
# and the NBER recession indicator.
FRED_LATEST = ("BAA", "AAA", "T10Y2Y", "USREC", "CPIAUCSL", "UNRATE", "INDPRO", "TB3MS", "DGS10", "NFCI")
# Monthly series with real-time vintages in ALFRED. T10Y2Y and DGS10 are daily market yields (never
# revised; T10Y2Y has more vintages than the API serves in one call) and NFCI is weekly.
ALFRED = ("BAA", "AAA", "CPIAUCSL", "UNRATE", "INDPRO", "TB3MS", "USREC")
YAHOO = {"GSPC": "^GSPC", "SPY": "SPY", "VIX": "^VIX", "TNX": "^TNX", "IRX": "^IRX", "VBMFX": "VBMFX"}
KEN_FRENCH = {
    "ff3_monthly": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip",
    "ff3_daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip",
}
# Amit Goyal's predictor data (linked from https://sites.google.com/view/agoyal145), exported as xlsx.
GOYAL_SHEET = "https://docs.google.com/spreadsheets/d/17mw_IpaiLFDrGnrPRQ2o1ugV5nJsZuD1/export?format=xlsx"
WELCH_GOYAL_PDF = "https://www.ivo-welch.info/research/journalcopy/2008-rfs.pdf"
DAMODARAN = {f: f"https://pages.stern.nyu.edu/~adamodar/pc/datasets/{f}" for f in ("betas.xls", "wacc.xls", "histimpl.xls")}
SP500_HISTORY = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
                 "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")
FSDS = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{q}.zip"


# ── manifest ──────────────────────────────────────────────────────────────────
def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _redact(url: str) -> str:
    key = os.environ.get("FRED_API_KEY")
    return url.replace(key, "***") if key else url


def read_manifest() -> Dict[str, dict]:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def _record(name: str, url: str, path: Path, data: bytes) -> None:
    m = read_manifest()
    m[name] = {"url": _redact(url), "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "sha256": sha256(data), "bytes": len(data), "path": path.relative_to(ROOT).as_posix(),
               "committed": path.is_relative_to(RAW)}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=1, sort_keys=True))


def store(name: str, url: str, data: bytes, rel: str, cache: bool = False) -> Path:
    path = (CACHE if cache else RAW) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    _record(name, url, path, data)
    return path


def gz(b: bytes) -> bytes:
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", mtime=0) as f:
        f.write(b)
    return out.getvalue()


def verify_manifest() -> list:
    """Problems: committed files missing or not matching their recorded hash."""
    problems = []
    for name, e in read_manifest().items():
        p = ROOT / e["path"]
        if not p.exists():
            if e["committed"]:
                problems.append(f"{name}: {e['path']} missing")
            continue
        if sha256(p.read_bytes()) != e["sha256"]:
            problems.append(f"{name}: {e['path']} does not match its SHA-256")
    return problems


# ── fetchers (run in GitHub Actions; this module never downloads at import) ──
def _get(url: str, params: Optional[dict] = None, timeout: float = 120, tries: int = 4, ua: bool = False) -> bytes:
    import requests
    headers = {"User-Agent": os.environ.get("SEC_USER_AGENT") or "MacroPulse validation"} if ua else {}
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=headers)
            r.raise_for_status()
            return r.content
        except Exception as e:                                # retried, then re-raised with the key removed
            last = e
            time.sleep(2 ** i)
    raise RuntimeError(_redact(f"{url}: {type(last).__name__}: {last}"))


def _fred(sid: str, vintages: bool) -> pd.DataFrame:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY is required for FRED/ALFRED downloads")
    params = {"series_id": sid, "file_type": "json", "api_key": key}
    if vintages:
        params.update(realtime_start="1776-07-04", realtime_end="9999-12-31")
    j = json.loads(_get(FRED_API, params))
    df = pd.DataFrame(j["observations"])
    df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")
    cols = ["realtime_start", "realtime_end", "date", "value"] if vintages else ["date", "value"]
    return df[cols]


def fetch_all(include_sec: bool = True, log: Callable[[str], None] = print) -> None:
    from urllib.parse import urlencode
    for sid in FRED_LATEST:
        df = _fred(sid, vintages=False)
        store(f"fred:{sid}", f"{FRED_API}?{urlencode({'series_id': sid})}", gz(df.to_csv(index=False).encode()),
              f"fred/{sid}.csv.gz")
        log(f"FRED {sid}: {len(df)} rows")
    for sid in ALFRED:
        df = _fred(sid, vintages=True)
        store(f"alfred:{sid}", f"{FRED_API}?{urlencode({'series_id': sid, 'realtime_start': '1776-07-04', 'realtime_end': '9999-12-31'})}",
              gz(df.to_csv(index=False).encode()), f"alfred/{sid}.csv.gz")
        log(f"ALFRED {sid}: {len(df)} rows, {df['realtime_start'].nunique()} vintages")
    import yfinance as yf
    for name, tkr in YAHOO.items():
        raw = yf.download(tkr, start="1927-01-01", auto_adjust=True, progress=False, multi_level_index=False, timeout=60)
        if raw.empty:
            raise RuntimeError(f"Yahoo returned no data for {tkr}")
        raw.index = pd.to_datetime(raw.index).tz_localize(None) if raw.index.tz is not None else raw.index
        out = raw[["Close"]].rename(columns={"Close": "close"})
        out.index.name = "date"
        store(f"yahoo:{tkr}", f"yfinance:{tkr}?auto_adjust=True", gz(out.to_csv().encode()), f"yahoo/{name}.csv.gz")
        log(f"Yahoo {tkr}: {len(out)} days from {out.index[0].date()}")
    for name, url in KEN_FRENCH.items():
        store(f"kenfrench:{name}", url, _get(url), f"kenfrench/{name}.zip")
    store("goyal:predictors", GOYAL_SHEET, _get(GOYAL_SHEET), "goyal/PredictorData.xlsx")
    store("paper:welch_goyal_2008", WELCH_GOYAL_PDF, _get(WELCH_GOYAL_PDF), "paper/welch_goyal_2008_rfs.pdf", cache=True)
    for f, url in DAMODARAN.items():
        store(f"damodaran:{f}", url, _get(url), f"damodaran/{f}")
    store("sp500:history", SP500_HISTORY, gz(_get(SP500_HISTORY)), "sp500/history.csv.gz")
    log("static sources done")


# ── loaders ───────────────────────────────────────────────────────────────────
def _need(rel: str) -> Path:
    p = RAW / rel
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing: run `python -m validation.run fetch` (GitHub Actions)")
    return p


def fred_latest(sid: str) -> pd.Series:
    df = pd.read_csv(_need(f"fred/{sid}.csv.gz"), parse_dates=["date"])
    return df.set_index("date")["value"].rename(sid)


def alfred(sid: str) -> pd.DataFrame:
    df = pd.read_csv(_need(f"alfred/{sid}.csv.gz"), parse_dates=["date"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    df["realtime_end"] = pd.to_datetime(df["realtime_end"].replace("9999-12-31", "2262-04-11"))
    return df


def yahoo(name: str) -> pd.Series:
    df = pd.read_csv(_need(f"yahoo/{name}.csv.gz"), parse_dates=["date"])
    return df.set_index("date")["close"].rename(name)


def ken_french(name: str = "ff3_monthly") -> pd.DataFrame:
    """Mkt-RF, SMB, HML, RF in decimals. Monthly rows are labelled by month start; the annual block is dropped."""
    with zipfile.ZipFile(_need(f"kenfrench/{name}.zip")) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5 and parts[0].isdigit() and len(parts[0]) in (6, 8):
            rows.append(parts[:5])
        elif rows and line.strip() == "":
            if name == "ff3_monthly" and len(rows) > 100:   # the annual table follows a blank line
                break
    df = pd.DataFrame(rows, columns=["date", "Mkt-RF", "SMB", "HML", "RF"])
    fmt = "%Y%m" if name == "ff3_monthly" else "%Y%m%d"
    df["date"] = pd.to_datetime(df["date"], format=fmt)
    df = df.set_index("date").astype(float) / 100.0
    return df[~df.index.duplicated()]


def goyal(sheet: str = "Annual") -> pd.DataFrame:
    return pd.read_excel(_need("goyal/PredictorData.xlsx"), sheet_name=sheet)


def damodaran(name: str) -> Path:
    return _need(f"damodaran/{name}")


def sp500_history() -> pd.DataFrame:
    df = pd.read_csv(_need("sp500/history.csv.gz"), parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def sp500_members(on: str | pd.Timestamp) -> list:
    """Index members on a date: the latest snapshot on or before it."""
    h = sp500_history()
    row = h[h["date"] <= pd.Timestamp(on)].iloc[-1]
    return sorted({t.strip() for t in str(row["tickers"]).split(",") if t.strip()})


def monthly_last(s: pd.Series) -> pd.Series:
    """Month-end last value labelled by month start (the app's monthly index)."""
    m = s.dropna().resample("ME").last()
    m.index = m.index.to_period("M").to_timestamp()
    return m


def in_ci_without_data() -> bool:
    return not MANIFEST.exists()


def iter_cached(prefix: str) -> Iterable[Path]:
    d = CACHE / prefix
    return sorted(d.glob("*")) if d.exists() else []
