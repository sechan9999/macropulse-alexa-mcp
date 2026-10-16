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

import copy
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


# A holding-company reorganization moves the ticker to a new registrant before that registrant has filed
# a 10-K, while the annual history stays under the predecessor's CIK. Successor CIK -> predecessor CIK.
PREDECESSOR_CIK = {
    "0002115436": "0000034088",       # XOM: ExxonMobil Holdings Corp <- Exxon Mobil Corp
}


def merge_facts(successor: dict, predecessor: dict) -> dict:
    """The successor's company facts with the predecessor's financial-statement rows added under every
    concept. Where both report the same period, the later filing wins as it does for any restatement.
    Cover-page facts (dei, e.g. shares outstanding) stay the successor's own: they describe the current
    registrant. Rows both filers report identically are kept once."""
    out = copy.deepcopy(successor)
    for tax, concepts in predecessor.get("facts", {}).items():
        if tax == "dei":
            continue
        dst = out.setdefault("facts", {}).setdefault(tax, {})
        for tag, body in concepts.items():
            units = dst.setdefault(tag, {}).setdefault("units", {})
            for unit, rows in body.get("units", {}).items():
                own = units.get(unit, [])
                seen = {tuple(sorted(r.items())) for r in own}
                units[unit] = [r for r in rows if tuple(sorted(r.items())) not in seen] + own
    return out


# ─────────────────────────── XBRL -> annual table ───────────────────────────
ANNUAL_FORMS = ("10-K", "10-K/A", "10-KT")
QUARTERLY_FORMS = ("10-Q", "10-Q/A")
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
        if any(_rows(usgaap, tag, "USD") for tag in TAGS["revenue"]):
            raise UnsupportedFiler("Revenue is only in 10-Q filings: this registrant has not filed a 10-K yet "
                                   "(a new registrant, e.g. after a reorganization).")
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
    df.index = [fiscal_year_label(e) for e in ends]
    df.index.name = "fiscal_year"
    if df.index.duplicated().any():
        raise DataUnavailable(f"Two 10-K fiscal years end in the same year ({', '.join(df['fy_end'])}).")
    return df


def fiscal_year_label(end) -> str:
    """The calendar year a fiscal year ends in, except a 52/53-week year ending in the first week of
    January, which is named for the year before (JNJ: FY2022 ended 2023-01-01, FY2023 on 2023-12-31)."""
    end = pd.Timestamp(end)
    return str(end.year - 1 if end.month == 1 and end.day <= 7 else end.year)


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


def latest_annual_filing(facts: dict) -> str:
    """Filing date of the newest 10-K the fundamentals come from (10-Qs are not used, so not counted)."""
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    return max((r.get("filed", "") for tag in (*TAGS["revenue"], *TAGS["net_income"])
                for r in _rows(usgaap, tag, "USD") if str(r.get("form", "")).startswith(ANNUAL_FORMS)), default="")


def split_factor_after(splits: pd.Series | None, date) -> float:
    """Product of the split ratios that took effect after `date` (EPS filed before a 10-for-1 split is
    divided by 10 so it matches split-adjusted prices)."""
    if splits is None or len(splits) == 0:
        return 1.0
    after = splits[splits.index > pd.Timestamp(date)]
    return float(np.prod(after.values)) if len(after) else 1.0


def eps_by_filing_date(facts: dict, splits: pd.Series | None = None) -> pd.Series:
    """Annual diluted EPS indexed by the date it became public (filing date) -> no look-ahead in P/E bands.
    EPS is restated for stock splits that happened after its filing (prices are split-adjusted)."""
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    out = {}
    for tag in TAGS["eps_diluted"]:
        for u in _rows(usgaap, tag, "USD/shares"):
            if str(u.get("form", "")).startswith(ANNUAL_FORMS) and "start" in u and \
                    350 <= (pd.Timestamp(u["end"]) - pd.Timestamp(u["start"])).days <= 380:
                end, filed = pd.Timestamp(u["end"]), pd.Timestamp(u["filed"])
                # the period's own 10-K (filed within ~120 days of period end), not later comparatives
                if (filed - end).days <= 120:
                    out.setdefault(filed, float(u["val"]) / split_factor_after(splits, filed))
        if out:
            break
    return pd.Series(out, dtype=float).sort_index()


def split_adjust_eps(fin: pd.DataFrame, facts: dict, splits: pd.Series | None) -> pd.DataFrame:
    """The annual table keeps the latest-filed EPS per year, which is pre-split for years no later 10-K
    restated (e.g. FY2021 $15.00 vs FY2022 $2.65 after a 10-for-1 split). Divide each by the splits
    after the filing it came from."""
    if splits is None or len(splits) == 0 or "eps_diluted" not in fin:
        return fin
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    filed = {}
    for tag in TAGS["eps_diluted"]:
        for u in _rows(usgaap, tag, "USD/shares"):
            if str(u.get("form", "")).startswith(ANNUAL_FORMS) and "start" in u:
                k = u["end"]
                if k not in filed or u.get("filed", "") > filed[k][0]:
                    filed[k] = (u.get("filed", ""), float(u["val"]))
        if filed:
            break
    fin = fin.copy()
    for y in fin.index:
        hit = filed.get(fin.loc[y, "fy_end"])
        if hit and np.isclose(hit[1], fin.loc[y, "eps_diluted"]):
            fin.loc[y, "eps_diluted"] = hit[1] / split_factor_after(splits, hit[0])
    return fin


# ─────────────────────────── trailing twelve months (10-Q) ───────────────────────────
TTM_FLOWS = ("revenue", "operating_income", "net_income", "operating_cf", "capex", "depreciation",
             "interest_expense", "dividends")
TTM_INSTANTS = ("cash", "total_equity", "_ltd_total", "_ltd_noncurrent", "_ltd_current", "_cp", "_stb")


def _durations(usgaap: dict, tag: str, unit: str = "USD") -> dict:
    """{(start, end): value} for every duration fact in 10-K / 10-Q filings, latest filing wins."""
    best: dict[tuple, tuple[str, float]] = {}
    for u in _rows(usgaap, tag, unit):
        if "start" not in u or not str(u.get("form", "")).startswith(ANNUAL_FORMS + QUARTERLY_FORMS):
            continue
        k = (pd.Timestamp(u["start"]), pd.Timestamp(u["end"]))
        if k not in best or u.get("filed", "") > best[k][0]:
            best[k] = (u.get("filed", ""), float(u["val"]))
    return {k: v[1] for k, v in best.items()}


def _match(d: dict, start, end, tol: int = 10):
    for (s, e), v in d.items():
        if abs((s - start).days) <= tol and abs((e - end).days) <= tol:
            return v
    return None


def ttm(facts: dict, fin: pd.DataFrame) -> dict | None:
    """Trailing twelve months to the latest 10-Q after the last 10-K:
        TTM = last fiscal year + year-to-date (this year) - same year-to-date last year,
    plus the balance sheet at that quarter end. None when no 10-Q follows the last 10-K.
    `ytd_growth` (this year-to-date vs. last year's) is the most recent revenue growth on record."""
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    fy_end = pd.Timestamp(fin["fy_end"].iloc[-1])
    win = None
    for tag in TAGS["revenue"]:
        d = _durations(usgaap, tag)
        cands = [(s, e) for (s, e) in d if abs((s - fy_end).days - 1) <= 10 and (e - fy_end).days >= 60
                 and (e - s).days < 340]
        if cands:
            s, e = max(cands, key=lambda k: k[1])
            win = (s, e, s - pd.DateOffset(years=1), e - pd.DateOffset(years=1))
            break
    if win is None:
        return None
    cs, ce, ps, pe = win
    out = {"period_end": ce.date().isoformat(), "ytd_start": cs.date().isoformat(),
           "ytd_months": int(round((ce - cs).days / 30.44))}
    for key in TTM_FLOWS:
        out[key] = np.nan
        for tag in TAGS[key]:
            d = _durations(usgaap, tag)
            fy = _match(d, fy_end - pd.DateOffset(years=1) + pd.Timedelta(days=1), fy_end)
            cur, prev = _match(d, cs, ce), _match(d, ps, pe)
            if fy is not None and cur is not None and prev is not None:
                out[key] = fy + cur - prev
                if key == "revenue":
                    out["revenue_ytd"], out["revenue_ytd_prior"] = cur, prev
                break
    if not np.isfinite(out["revenue"]):
        return None
    # Some filers tag interest differently in 10-Qs than in the 10-K (e.g. AVGO), so no tag has all
    # three periods. Fall back to the last fiscal year's figure rather than adding back no interest.
    out["interest_expense_from_fy"] = False
    if not np.isfinite(out["interest_expense"]) and "interest_expense" in fin and pd.notna(fin["interest_expense"].iloc[-1]):
        out["interest_expense"] = float(fin["interest_expense"].iloc[-1])
        out["interest_expense_from_fy"] = True
    for key in TTM_INSTANTS:
        out[key] = np.nan
        for tag in TAGS[key]:
            vals = [(u.get("filed", ""), float(u["val"])) for u in _rows(usgaap, tag, "USD")
                    if "start" not in u and abs((pd.Timestamp(u["end"]) - ce).days) <= 5]
            if vals:
                out[key] = max(vals)[1]
                break
    ltd = (np.nansum([out["_ltd_noncurrent"], out["_ltd_current"]])
           if np.isfinite(out["_ltd_noncurrent"]) or np.isfinite(out["_ltd_current"]) else out["_ltd_total"])
    out["debt"] = float((float(ltd) if np.isfinite(ltd) else np.nan) + np.nansum([out["_cp"], out["_stb"]]))
    for k in [k for k in out if k.startswith("_")]:
        del out[k]
    for k in ("capex", "dividends", "interest_expense"):
        out[k] = abs(out[k]) if np.isfinite(out[k]) else out[k]
    out["ytd_growth"] = (out["revenue_ytd"] / out["revenue_ytd_prior"] - 1
                         if out.get("revenue_ytd_prior") else np.nan)
    return out


def dilution_factor(facts: dict) -> tuple[float, str | None]:
    """Diluted / basic weighted-average shares for the latest reported period (shortest period on the latest
    end date, i.e. the last quarter when a 10-Q exists), clipped to 1.00-1.15. Applied to the current share
    count to put RSUs and options into per-share values. (1.0, None) when either figure is missing."""
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    dil = _durations(usgaap, "WeightedAverageNumberOfDilutedSharesOutstanding", "shares")
    bas = _durations(usgaap, "WeightedAverageNumberOfSharesOutstandingBasic", "shares")
    common = [k for k in dil if k in bas and bas[k] > 0]
    if not common:
        return 1.0, None
    k = max(common, key=lambda k: (k[1], -(k[1] - k[0]).days))
    return float(np.clip(dil[k] / bas[k], 1.0, 1.15)), f"{k[0].date()} to {k[1].date()}"


# ─────────────────────────── prices (yfinance) ───────────────────────────
def _yf_splits(ticker: str) -> pd.Series:
    import yfinance as yf
    s = yf.Ticker(ticker).splits
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s[s > 0]


def _yf_prices(ticker: str, years: int = 5) -> pd.DataFrame:
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
def load_bundle(ticker: str, years: int = 5, *, price_fetch: Callable[[str], pd.DataFrame] | None = None,
                http_get: Callable[[str], dict] | None = None, risk_free: float | None = None,
                with_positioning: bool = True, splits_fetch: Callable[[str], pd.Series] | None = None) -> dict:
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
        predecessor = PREDECESSOR_CIK.get(cik)
        if predecessor:
            facts = merge_facts(facts, company_facts(predecessor, http_get))
    except UnsupportedFiler:
        raise
    except Exception as e:
        raise DataUnavailable(f"SEC EDGAR unavailable: {e}") from e
    fin = annual_table(facts)
    shares = shares_outstanding(facts)
    if not shares:
        raise UnsupportedFiler("Shares outstanding not found in SEC filings.")

    close = prices["close"]
    try:
        splits = (splits_fetch or _yf_splits)(ticker)
    except Exception:  # noqa: BLE001 - EPS then stays as filed; the P/E band note says so
        splits = None
    eps = eps_by_filing_date(facts, splits)
    fin = split_adjust_eps(fin, facts, splits)
    t12 = ttm(facts, fin)
    dil, dil_period = dilution_factor(facts)
    shares_diluted = shares * dil
    pe = None
    if len(eps):
        eps_asof = eps.reindex(close.index.union(eps.index)).ffill().reindex(close.index)
        pe = close / eps_asof.where(eps_asof > 0)
    last_filed = latest_annual_filing(facts)
    return {
        "ticker": ticker, "name": name, "cik": cik, "is_synthetic": False,
        "prices": prices, "index": index, "financials": fin, "shares": shares,
        "last_price": float(close.iloc[-1]), "pe_history": pe,
        "eps_latest": float(eps.iloc[-1]) if len(eps) else None,
        "ttm": t12, "eps_ttm": (t12["net_income"] / shares_diluted) if t12 and np.isfinite(t12["net_income"]) else None,
        "dilution_factor": dil, "dilution_period": dil_period, "shares_diluted": shares_diluted,
        "splits_checked": splits is not None,
        "positioning": positioning(ticker) if with_positioning else None,
        "risk_free_live": risk_free,
        "sources": ["Yahoo Finance (prices, splits)",
                    f"SEC EDGAR XBRL 10-K (CIK {cik}"
                    + (f", 10-K history from predecessor CIK {predecessor}" if predecessor else "")
                    + f", latest 10-K filed {last_filed})"]
        + ([f"SEC EDGAR XBRL 10-Q (trailing twelve months to {t12['period_end']})"] if t12 else []),
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
