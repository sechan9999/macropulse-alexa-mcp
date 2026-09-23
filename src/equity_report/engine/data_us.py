"""US data layer: prices from yfinance, fundamentals from SEC EDGAR XBRL "company facts".

EDGAR (free, official): https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
The SEC requires a descriptive User-Agent with a contact e-mail -> set SEC_USER_AGENT.

Scope: companies filing 10-K in USD under us-gaap. Foreign private issuers (20-F/40-F, IFRS,
non-USD such as TSM or ASML) and ETFs raise UnsupportedFiler with a plain-English reason rather
than producing a DCF on mismatched currencies.

Every network call is injectable (price_fetch / http_get) so the engine is testable offline and
the MCP server can route prices through its own TTL cache.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from ..config import SEC_USER_AGENT


class UnsupportedFiler(RuntimeError):
    """The ticker has no usable US-GAAP 10-K fundamentals (ETF, foreign filer, recent IPO...)."""


class DataUnavailable(RuntimeError):
    """A data source could not be reached; never replaced with made-up numbers."""


# ─────────────────────────── HTTP (SEC) ───────────────────────────
_sec_lock = threading.Lock()
_last_call = [0.0]
_mem: dict[str, tuple[float, object]] = {}


def _sec_get_json(url: str, timeout: float = 15) -> dict:
    import requests
    with _sec_lock:                       # SEC fair-access limit is 10 req/s
        wait = 0.12 - (time.monotonic() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()
    r = requests.get(url, timeout=timeout, headers={"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    if r.status_code == 404:
        raise UnsupportedFiler(f"SEC has no data at {url.rsplit('/', 1)[-1]}")
    r.raise_for_status()
    return r.json()


def _cached(key: str, ttl: float, fn: Callable[[], object]):
    hit = _mem.get(key)
    if hit and time.monotonic() - hit[0] < ttl:
        return hit[1]
    val = fn()
    _mem[key] = (time.monotonic(), val)
    return val


def cik_for(ticker: str, http_get: Callable[[str], dict] | None = None) -> tuple[str, str]:
    """Ticker -> (10-digit CIK, company name). SEC lists class shares with '-' (BRK-B)."""
    http_get = http_get or _sec_get_json
    table = _cached("company_tickers", 86400, lambda: http_get("https://www.sec.gov/files/company_tickers.json"))
    t = ticker.upper().replace(".", "-")
    for row in table.values():
        if row["ticker"].upper() == t:
            return f"{int(row['cik_str']):010d}", row["title"]
    raise UnsupportedFiler(f"{ticker.upper()} is not in the SEC company list (ETF, fund, or non-SEC filer?)")


def company_facts(cik: str, http_get: Callable[[str], dict] | None = None) -> dict:
    http_get = http_get or _sec_get_json
    return _cached(f"facts:{cik}", 6 * 3600,
                   lambda: http_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"))


# ─────────────────────────── XBRL -> annual table ───────────────────────────
ANNUAL_FORMS = ("10-K", "10-K/A", "10-KT")
TAGS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "operating_cf": ["NetCashProvidedByUsedInOperatingActivities",
                     "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "depreciation": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "Depreciation"],
    "dividends": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"],
    "eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
    # instants (balance sheet at fiscal year end)
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "total_equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "_ltd_total": ["LongTermDebt"],
    "_ltd_noncurrent": ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"],
    "_ltd_current": ["LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"],
    "_cp": ["CommercialPaper"],
    "_stb": ["ShortTermBorrowings"],
}
INSTANTS = {"total_assets", "total_liabilities", "total_equity", "current_assets", "current_liabilities",
            "cash", "_ltd_total", "_ltd_noncurrent", "_ltd_current", "_cp", "_stb"}


def _rows(usgaap: dict, tag: str, unit: str) -> list[dict]:
    return usgaap.get(tag, {}).get("units", {}).get(unit, [])


def _series(usgaap: dict, tags: list[str], instant: bool, unit: str = "USD") -> pd.Series:
    """Merge tags in priority order into {period end -> value}, keeping the latest-filed figure
    (restatements win) and, for flows, only ~12-month durations from annual filings."""
    out: dict[pd.Timestamp, tuple[str, float]] = {}
    for tag in tags:
        best: dict[pd.Timestamp, tuple[str, float]] = {}
        for u in _rows(usgaap, tag, unit):
            if not str(u.get("form", "")).startswith(ANNUAL_FORMS):
                continue
            end = pd.Timestamp(u["end"])
            if instant:
                if "start" in u:
                    continue
            else:
                if "start" not in u or not 350 <= (end - pd.Timestamp(u["start"])).days <= 380:
                    continue
            filed = u.get("filed", "")
            if end not in best or filed > best[end][0]:
                best[end] = (filed, float(u["val"]))
        for end, v in best.items():
            out.setdefault(end, v)           # higher-priority tag already set -> keep it
    return pd.Series({k: v[1] for k, v in out.items()}, dtype=float).sort_index()


def annual_table(facts: dict, n_years: int = 5) -> pd.DataFrame:
    usgaap = facts.get("facts", {}).get("us-gaap")
    if not usgaap:
        raise UnsupportedFiler("No US-GAAP facts (foreign filer using IFRS, or no XBRL financials).")
    rev = _series(usgaap, TAGS["revenue"], instant=False)
    if rev.empty:
        raise UnsupportedFiler("No annual USD revenue in 10-K filings (20-F/40-F filer or non-USD reporting?).")
    ends = list(rev.index[-n_years:])
    cols = {}
    for key, tags in TAGS.items():
        unit = "USD/shares" if key == "eps_diluted" else "USD"
        s = _series(usgaap, tags, instant=key in INSTANTS, unit=unit)
        cols[key] = [s.get(e, np.nan) for e in ends]
    df = pd.DataFrame(cols, index=ends)
    ltd = df["_ltd_noncurrent"].fillna(0) + df["_ltd_current"].fillna(0)
    ltd = ltd.where((df["_ltd_noncurrent"].notna() | df["_ltd_current"].notna()), df["_ltd_total"])
    df["debt"] = ltd.fillna(0) + df["_cp"].fillna(0) + df["_stb"].fillna(0)
    df["total_liabilities"] = df["total_liabilities"].fillna(
        _series(usgaap, ["LiabilitiesAndStockholdersEquity"], instant=True).reindex(ends).values - df["total_equity"])
    df["capex"] = df["capex"].abs()
    df["dividends"] = df["dividends"].abs()
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    df["fy_end"] = [e.date().isoformat() for e in ends]
    df.index = [str(e.year) for e in ends]
    df.index.name = "fiscal_year"
    return df


def shares_outstanding(facts: dict) -> float | None:
    """Latest dei:EntityCommonStockSharesOutstanding, summed across share classes (e.g. GOOGL + GOOG)."""
    rows = facts.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding", {}) \
        .get("units", {}).get("shares", [])
    if rows:
        latest_filed = max(r.get("filed", "") for r in rows)
        latest = [r for r in rows if r.get("filed", "") == latest_filed]
        end = max(r["end"] for r in latest)
        return float(sum(r["val"] for r in latest if r["end"] == end))
    s = _series(facts.get("facts", {}).get("us-gaap", {}),
                ["WeightedAverageNumberOfDilutedSharesOutstanding"], instant=False, unit="shares")
    return float(s.iloc[-1]) if len(s) else None


def eps_by_filing_date(facts: dict) -> pd.Series:
    """Annual diluted EPS indexed by the date it became public (filing date) -> no look-ahead in P/E bands."""
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    out = {}
    for tag in TAGS["eps_diluted"]:
        for u in _rows(usgaap, tag, "USD/shares"):
            if str(u.get("form", "")).startswith(ANNUAL_FORMS) and "start" in u and \
                    350 <= (pd.Timestamp(u["end"]) - pd.Timestamp(u["start"])).days <= 380:
                end, filed = pd.Timestamp(u["end"]), pd.Timestamp(u["filed"])
                # the period's own 10-K (filed within ~120 days of period end), not later comparatives
                if (filed - end).days <= 120:
                    out.setdefault(filed, float(u["val"]))
        if out:
            break
    return pd.Series(out, dtype=float).sort_index()


# ─────────────────────────── prices (yfinance) ───────────────────────────
def _yf_prices(ticker: str, years: int = 3) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=False,
                      multi_level_index=False, timeout=20)
    return raw


def normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise DataUnavailable("no price rows returned")
    df = raw.rename(columns=str.lower)
    df = df[[c for c in ("open", "high", "low", "close", "volume") if c in df.columns]].dropna(subset=["close"])
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df


def positioning(ticker: str) -> dict | None:
    """US substitute for Korea's daily institutional/foreign flow: ownership & short-interest snapshots
    (13F-based institutional %, insider net activity, short interest trend). Best effort; None if unavailable."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info or {}
        out = {
            "institutional_pct": info.get("heldPercentInstitutions"),
            "insider_pct": info.get("heldPercentInsiders"),
            "short_pct_float": info.get("shortPercentOfFloat"),
            "short_ratio_days": info.get("shortRatio"),
            "short_change_vs_prior_month": (info["sharesShort"] / info["sharesShortPriorMonth"] - 1)
            if info.get("sharesShort") and info.get("sharesShortPriorMonth") else None,
        }
        try:
            ip = t.insider_purchases
            row = ip[ip.iloc[:, 0].astype(str).str.contains("Net Shares", case=False)]
            out["insider_net_shares_6m"] = float(row["Shares"].iloc[0]) if len(row) else None
        except Exception:
            out["insider_net_shares_6m"] = None
        return out
    except Exception:
        return None


# ─────────────────────────── bundle ───────────────────────────
def load_bundle(ticker: str, years: int = 3, *, price_fetch: Callable[[str], pd.DataFrame] | None = None,
                http_get: Callable[[str], dict] | None = None, risk_free: float | None = None,
                with_positioning: bool = True) -> dict:
    """Everything the engine needs for one ticker. Raises UnsupportedFiler / DataUnavailable."""
    ticker = ticker.upper().strip()
    fetch = price_fetch or (lambda t: _yf_prices(t, years))
    try:
        prices = normalize_ohlcv(fetch(ticker))
        index = normalize_ohlcv(fetch("^GSPC"))[["close"]]
    except UnsupportedFiler:
        raise
    except Exception as e:
        raise DataUnavailable(f"price data for {ticker} unavailable: {e}") from e
    if len(prices) < 260:
        raise DataUnavailable(f"only {len(prices)} daily bars for {ticker}; need at least ~1 year")

    try:
        cik, name = cik_for(ticker, http_get)
        facts = company_facts(cik, http_get)
    except UnsupportedFiler:
        raise
    except Exception as e:
        raise DataUnavailable(f"SEC EDGAR unavailable: {e}") from e
    fin = annual_table(facts)
    shares = shares_outstanding(facts)
    if not shares:
        raise UnsupportedFiler("Shares outstanding not found in SEC filings.")

    close = prices["close"]
    eps = eps_by_filing_date(facts)
    pe = None
    if len(eps):
        eps_asof = eps.reindex(close.index.union(eps.index)).ffill().reindex(close.index)
        pe = close / eps_asof.where(eps_asof > 0)
    last_filed = max((r.get("filed", "") for tag in ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                                                     "NetIncomeLoss")
                      for r in _rows(facts["facts"]["us-gaap"], tag, "USD")), default="")
    return {
        "ticker": ticker, "name": name, "cik": cik, "is_synthetic": False,
        "prices": prices, "index": index, "financials": fin, "shares": shares,
        "last_price": float(close.iloc[-1]), "pe_history": pe,
        "eps_latest": float(eps.iloc[-1]) if len(eps) else None,
        "positioning": positioning(ticker) if with_positioning else None,
        "risk_free_live": risk_free,
        "sources": ["Yahoo Finance (prices)", f"SEC EDGAR XBRL 10-K (CIK {cik}, latest filing {last_filed})"],
        "as_of": str(prices.index[-1].date()),
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }


def validate(b: dict) -> list[str]:
    issues = []
    fin = b["financials"]
    for col in ("revenue", "operating_cf", "capex", "total_equity"):
        if col not in fin or fin[col].isna().all():
            issues.append(f"missing fundamental: {col}")
    if len(fin) < 3:
        issues.append(f"only {len(fin)} fiscal years of fundamentals")
    stale = pd.Timestamp(fin["fy_end"].iloc[-1]) < pd.Timestamp(datetime.now() - timedelta(days=500))
    if stale:
        issues.append(f"latest 10-K period ends {fin['fy_end'].iloc[-1]} — fundamentals may be stale")
    return issues
