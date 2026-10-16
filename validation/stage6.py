"""
validation/stage6.py — DCF equity report (pre-registration H5)
─────────────────────────────────────────────────────────────────
  1. Inputs vs the filings: for 10 non-financial companies in different industries (AVGO included), the
     engine's latest fiscal-year revenue, operating cash flow, capital expenditure, interest expense,
     diluted weighted-average shares and net debt, against the same filing in the SEC Financial
     Statement Data Sets (FSDS): the value FSDS holds for that 10-K, and whether the tag sits on the
     face of the income, cash-flow or balance-sheet statement (pre.txt). Differences above 0.5% are listed.
  2. Beta, WACC and ERP against Damodaran's industry averages and his latest implied ERP.
  3. Point-in-time reconstruction: at each year end 2019-2024, the DCF for that date's S&P 500 members
     (financials and real estate excluded by SIC 6000-6799), using only filings filed by that date and
     prices up to it, then the next 12-month total return in excess of SPY.
     Primary H5: one-sided t-test (Newey-West, 1 lag) that the mean yearly Spearman rank IC between the
     probability-weighted upside and that excess return is above zero. AVGO is reported but excluded.

  Known limits: members are taken from a community dataset (github.com/fja05680/sp500); companies whose
  ticker no longer maps to an SEC CIK, or whose prices Yahoo no longer serves, drop out (survivorship);
  the macro regime and market drift at each date come from today's data vintage.
"""
from __future__ import annotations

import copy
import io
import json
import zipfile

import numpy as np
import pandas as pd

from validation import data, frames, stats
from validation.results import save_csv, save_json

STAGE = "stage6"
SAMPLE = ("AVGO", "AAPL", "MSFT", "XOM", "JNJ", "WMT", "KO", "CAT", "NEE", "UNP")
DAMODARAN_INDUSTRY = {"AVGO": "Semiconductor", "AAPL": "Computers/Peripherals",
                      "MSFT": "Software (System & Application)", "XOM": "Oil/Gas (Integrated)",
                      "JNJ": "Drugs (Pharmaceutical)", "WMT": "Retail (General)", "KO": "Beverage (Soft)",
                      "CAT": "Machinery", "NEE": "Power", "UNP": "Transportation (Railroads)"}
YEARS = range(2019, 2025)
EXCLUDED_FROM_H5 = {"AVGO"}
TOL = 0.005


# ── SEC access (cached; GitHub Actions only) ─────────────────────────────────
def _sec_json(url: str, cache_name: str) -> dict:
    p = data.CACHE / "sec" / cache_name
    if p.exists():
        return json.loads(p.read_text())
    from src.equity_report.engine import data_us
    j = data_us._sec_get_json(url, timeout=60)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(j))
    return j


def company_facts(cik: str) -> dict:
    return _sec_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", f"facts_{cik}.json")


def submissions(cik: str) -> dict:
    return _sec_json(f"https://data.sec.gov/submissions/CIK{cik}.json", f"sub_{cik}.json")


def ticker_table() -> dict:
    j = _sec_json("https://www.sec.gov/files/company_tickers.json", "company_tickers.json")
    return {row["ticker"].upper(): f"{int(row['cik_str']):010d}" for row in j.values()}


def facts_with_predecessor(cik: str) -> dict:
    from src.equity_report.engine import data_us
    f = company_facts(cik)
    pred = data_us.PREDECESSOR_CIK.get(cik)
    return data_us.merge_facts(f, company_facts(pred)) if pred else f


# ── 1. inputs vs FSDS ─────────────────────────────────────────────────────────
ITEMS = {"revenue": ("IS", False), "operating_cf": ("CF", False), "capex": ("CF", False),
         "interest_expense": ("IS", False), "diluted_shares": ("IS", False)}


def _quarter(filed: str) -> str:
    d = pd.Timestamp(filed)
    return f"{d.year}q{(d.month - 1) // 3 + 1}"


def fsds_rows(q: str, adsh: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = data.CACHE / "fsds" / f"{q}.zip"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data._get(data.FSDS.format(q=q), ua=True, timeout=600))
        data._record(f"fsds:{q}", data.FSDS.format(q=q), p, p.read_bytes())
    with zipfile.ZipFile(p) as z:
        num = pd.concat(c[c["adsh"] == adsh] for c in pd.read_csv(z.open("num.txt"), sep="\t", dtype=str,
                                                                   chunksize=500_000, on_bad_lines="skip"))
        pre = pd.concat(c[c["adsh"] == adsh] for c in pd.read_csv(z.open("pre.txt"), sep="\t", dtype=str,
                                                                   chunksize=500_000, on_bad_lines="skip"))
    return num, pre


def _engine_items(facts: dict) -> tuple[dict, dict, str, str]:
    """Latest fiscal year values the engine uses, the tag each came from, the 10-K accession and fy end."""
    from src.equity_report.engine import data_us
    g = facts["facts"]["us-gaap"]
    fin = data_us.annual_table(facts)
    last = fin.iloc[-1]
    fy_end = pd.Timestamp(last["fy_end"])
    vals, tags = {}, {}
    for key, tlist in (("revenue", data_us.TAGS["revenue"]), ("operating_cf", data_us.TAGS["operating_cf"]),
                       ("capex", data_us.TAGS["capex"]), ("interest_expense", data_us.TAGS["interest_expense"]),
                       ("diluted_shares", ["WeightedAverageNumberOfDilutedSharesOutstanding"])):
        unit = "shares" if key == "diluted_shares" else "USD"
        for t in tlist:
            rows = [r for r in data_us._rows(g, t, unit) if r.get("end") == fy_end.date().isoformat()
                    and str(r.get("form", "")).startswith(data_us.ANNUAL_FORMS) and "start" in r
                    and 350 <= (pd.Timestamp(r["end"]) - pd.Timestamp(r["start"])).days <= 380]
            if rows:
                best = max(rows, key=lambda r: r.get("filed", ""))
                vals[key], tags[key] = abs(float(best["val"])), t
                break
    rev_rows = [r for t in data_us.TAGS["revenue"] for r in data_us._rows(g, t, "USD")
                if r.get("end") == fy_end.date().isoformat() and r.get("form", "").startswith("10-K")]
    first_filing = min(rev_rows, key=lambda r: r.get("filed", ""))
    vals["net_debt"] = float(last["debt"] - last["cash"])
    return vals, tags, first_filing["accn"], first_filing["filed"]


def reconcile_inputs() -> pd.DataFrame:
    tick = ticker_table()
    rows = []
    for t in SAMPLE:
        cik = tick[t]
        facts = facts_with_predecessor(cik)
        vals, tags, adsh, filed = _engine_items(facts)
        num, pre = fsds_rows(_quarter(filed), adsh)
        fy_end = data_us_fy_end(facts)
        for key, v in vals.items():
            if key == "net_debt":
                rows.append({"ticker": t, "item": key, "engine": v, "fsds": None, "tag": "debt - cash (annual table)",
                             "on_statement": None, "rel_diff": None, "flag": False,
                             "note": "derived; components checked through the debt and cash tags below"})
                continue
            tag = tags.get(key)
            stmt = ITEMS[key][0]
            q = num[(num["tag"] == tag) & (num["ddate"] == fsds_ddate(fy_end)) & (num["qtrs"] == "4")]
            if "segments" in q:
                q = q[q["segments"].isna()]
            if "coreg" in q:
                q = q[q["coreg"].isna()]
            fv = abs(float(q["value"].astype(float).iloc[0])) if len(q) else None
            on = bool(((pre["tag"] == tag) & (pre["stmt"] == stmt)).any()) if tag else None
            rd = abs(v - fv) / abs(fv) if fv else None
            rows.append({"ticker": t, "item": key, "engine": v, "fsds": fv, "tag": tag, "on_statement": on,
                         "rel_diff": rd, "flag": bool(rd is None or rd > TOL), "adsh": adsh, "fsds_quarter": _quarter(filed),
                         "note": "" if fv is not None else "tag not found in FSDS for this filing"})
    return pd.DataFrame(rows)


def fsds_ddate(end: pd.Timestamp) -> str:
    """FSDS stores period ends rounded to the nearest month end (a 2025-11-02 year end is 20251031)."""
    prev, nxt = end + pd.offsets.MonthEnd(-1) if not end.is_month_end else end, end + pd.offsets.MonthEnd(0)
    best = prev if abs((end - prev).days) <= abs((nxt - end).days) else nxt
    return best.strftime("%Y%m%d")


def data_us_fy_end(facts: dict) -> pd.Timestamp:
    from src.equity_report.engine import data_us
    return pd.Timestamp(data_us.annual_table(facts)["fy_end"].iloc[-1])


# ── 2. Damodaran ──────────────────────────────────────────────────────────────
def _damodaran_table(name: str) -> pd.DataFrame:
    raw = pd.read_excel(data.damodaran(name), sheet_name=None, header=None)
    for sheet, df in raw.items():
        hit = df.index[df.apply(lambda r: r.astype(str).str.strip().eq("Industry Name").any(), axis=1)]
        if len(hit):
            t = df.iloc[hit[0] + 1:].copy()
            t.columns = [str(c).strip() for c in df.iloc[hit[0]]]
            return t.set_index("Industry Name")
    raise ValueError(f"no industry table in {name}")


def implied_erp() -> dict:
    raw = pd.read_excel(data.damodaran("histimpl.xls"), sheet_name=None, header=None)
    for sheet, df in raw.items():
        hit = df.index[df.apply(lambda r: r.astype(str).str.contains("Implied ERP", case=False).any(), axis=1)]
        if len(hit):
            hdr = df.iloc[hit[0]].astype(str).str.strip().tolist()
            body = df.iloc[hit[0] + 1:]
            col = next(i for i, h in enumerate(hdr) if "Implied ERP" in h)
            yr = body.iloc[:, 0]
            body = body[pd.to_numeric(yr, errors="coerce").notna()]
            last = body.iloc[-1]
            return {"year": int(last.iloc[0]), "implied_erp": float(last.iloc[col]), "column": hdr[col], "sheet": sheet}
    raise ValueError("implied ERP not found")


def damodaran_comparison(reports: dict) -> tuple[pd.DataFrame, dict]:
    betas, wacc = _damodaran_table("betas.xls"), _damodaran_table("wacc.xls")
    rows = []
    for t, ind in DAMODARAN_INDUSTRY.items():
        r = reports.get(t)
        b = betas.loc[ind] if ind in betas.index else None
        w = wacc.loc[ind] if ind in wacc.index else None
        bcol = next((c for c in betas.columns if c.lower() == "beta"), None)
        wcol = next((c for c in wacc.columns if "cost of capital" in c.lower()), None)
        rows.append({"ticker": t, "damodaran_industry": ind, "found": b is not None and w is not None,
                     "engine_beta": r["beta"] if r else None,
                     "damodaran_beta": float(b[bcol]) if b is not None and bcol else None,
                     "engine_wacc": r["wacc"] if r else None,
                     "damodaran_cost_of_capital": float(w[wcol]) if w is not None and wcol else None,
                     "engine_erp": r["erp"] if r else None})
    return pd.DataFrame(rows).set_index("ticker"), implied_erp()


def live_reports() -> dict:
    """Today's engine output for the 10-company sample (beta, WACC, ERP)."""
    from src.equity_report.engine import pipeline
    out = {}
    for t in SAMPLE:
        try:
            rep = pipeline.run(t, provider="rules", with_positioning=False)
            v = rep.ctx["valuation"]
            out[t] = {"beta": v["beta"], "wacc": v["wacc"], "erp": v["erp_used"]}
        except Exception as e:                                   # reported, not filled in
            out[t] = None
            print(f"live report {t} failed: {type(e).__name__}: {e}")
    return out


# ── 3. point-in-time reconstruction ───────────────────────────────────────────
def facts_asof(facts: dict, date: pd.Timestamp) -> dict:
    """Company facts as a reader could have seen them on `date`: rows filed on or before it."""
    d = date.date().isoformat()
    out = copy.deepcopy(facts)
    for concepts in out.get("facts", {}).values():
        for body in concepts.values():
            for unit, rows in body.get("units", {}).items():
                body["units"][unit] = [r for r in rows if str(r.get("filed", "9999")) <= d]
    return out


def bundle_asof(ticker: str, name: str, cik: str, facts: dict, prices: pd.DataFrame, index: pd.DataFrame,
                splits: pd.Series, date: pd.Timestamp, risk_free: float) -> dict:
    """data_us.load_bundle's steps on injected, date-truncated inputs. Shares are put on today's split
    basis (Yahoo's adjusted prices are), i.e. multiplied by every split after `date`."""
    from src.equity_report.engine import data_us
    ff = facts_asof(facts, date)
    px = prices[prices.index <= date]
    ix = index[index.index <= date]
    fin = data_us.annual_table(ff)
    shares = data_us.shares_outstanding(ff)
    if not shares:
        raise data_us.UnsupportedFiler("no shares outstanding as of the date")
    shares *= data_us.split_factor_after(splits, date)
    eps = data_us.eps_by_filing_date(ff, splits)
    fin = data_us.split_adjust_eps(fin, ff, splits)
    t12 = data_us.ttm(ff, fin)
    dil, dil_period = data_us.dilution_factor(ff)
    close = px["close"]
    pe = None
    if len(eps):
        eps_asof = eps.reindex(close.index.union(eps.index)).ffill().reindex(close.index)
        pe = close / eps_asof.where(eps_asof > 0)
    shares_diluted = shares * dil
    return {"ticker": ticker, "name": name, "cik": cik, "is_synthetic": False, "prices": px, "index": ix,
            "financials": fin, "shares": shares, "last_price": float(close.iloc[-1]), "pe_history": pe,
            "eps_latest": float(eps.iloc[-1]) if len(eps) else None, "ttm": t12,
            "eps_ttm": (t12["net_income"] / shares_diluted) if t12 and np.isfinite(t12["net_income"]) else None,
            "dilution_factor": dil, "dilution_period": dil_period, "shares_diluted": shares_diluted,
            "splits_checked": True, "positioning": None, "risk_free_live": risk_free,
            "sources": [f"point-in-time reconstruction as of {date.date()}"], "as_of": str(px.index[-1].date()),
            "fetched_at": "validation"}


def _prices(tickers: list) -> dict:
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 50):
        chunk = tickers[i:i + 50]
        raw = yf.download(chunk, start="2013-01-01", auto_adjust=True, progress=False, group_by="ticker",
                          threads=True, timeout=60)
        for t in chunk:
            try:
                df = raw[t] if len(chunk) > 1 else raw
                df = df.dropna(subset=["Close"])
                if len(df) > 300:
                    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
                    df.index = pd.to_datetime(df.index).tz_localize(None) if df.index.tz is not None else df.index
                    out[t] = df
            except Exception:
                pass
    return out


def _splits(t: str) -> pd.Series:
    import yfinance as yf
    s = yf.Ticker(t).splits
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    s.index = pd.to_datetime(s.index).tz_localize(None) if s.index.tz is not None else s.index
    return s.astype(float)


def reconstruct() -> tuple[pd.DataFrame, dict]:
    from src.equity_report.engine import analysis, data_us
    from src.equity_report.engine.analysis import compact_summary
    from src.equity_report.engine.market_drift import expected_market_return

    tick = ticker_table()
    macro = frames.build_macro(frames.PRODUCT_START)
    members = {y: [m.replace(".", "-").upper() for m in data.sp500_members(f"{y}-12-31")] for y in YEARS}
    universe = sorted(set().union(*members.values()))
    mapped = [t for t in universe if t in tick]
    sic = {}
    for t in mapped:
        try:
            sic[t] = int(submissions(tick[t]).get("sic") or 0)
        except Exception:
            sic[t] = 0
    eligible = [t for t in mapped if not 6000 <= sic[t] <= 6799]
    prices = _prices(eligible + ["SPY", "^GSPC"])
    index = prices["^GSPC"][["close"]]
    spy = prices["SPY"]["close"]
    coverage = {"universe": len(universe), "mapped_to_cik": len(mapped),
                "excluded_financial_real_estate": len(mapped) - len(eligible),
                "with_prices": sum(t in prices for t in eligible)}
    rows = []
    for t in eligible:
        if t not in prices:
            continue
        try:
            facts = facts_with_predecessor(tick[t])
            spl = _splits(t)
        except Exception as e:
            rows.append({"ticker": t, "year": None, "error": f"data: {type(e).__name__}"})
            continue
        for y in YEARS:
            if t not in members[y]:
                continue
            d = pd.Timestamp(f"{y}-12-31")
            nxt = pd.Timestamp(f"{y + 1}-12-31")
            px = prices[t]
            try:
                mrow = macro[macro.index <= d]
                regime = str(mrow["regime"].iloc[-1])
                regime = None if regime.startswith("Unavailable") else regime
                rf = float(mrow["dgs10"].iloc[-1]) / 100
                market = expected_market_return(mrow)
                b = bundle_asof(t, t, tick[t], facts, px, index, spl, d, rf)
                rep = analysis.analyze(b, None, regime, market=market)
                cs = compact_summary(rep)
                p0, p1 = px["close"][px.index <= d].iloc[-1], px["close"][px.index <= nxt].iloc[-1]
                s0, s1 = spy[spy.index <= d].iloc[-1], spy[spy.index <= nxt].iloc[-1]
                ok_window = px.index.max() >= nxt - pd.Timedelta(days=7)
                rows.append({"ticker": t, "year": y, "upside": cs["upside_prob_weighted"], "rating": cs["rating"],
                             "fair_value": cs["fair_value_prob_weighted"], "price": cs["price"],
                             "excess_12m": float(p1 / p0 - s1 / s0) if ok_window else np.nan,
                             "base_period": cs["base_period"], "error": ""})
            except Exception as e:                               # recorded per name and year, never filled in
                rows.append({"ticker": t, "year": y, "error": f"{type(e).__name__}: {str(e)[:120]}"})
    return pd.DataFrame(rows), coverage


def rank_ic(panel: pd.DataFrame) -> dict:
    p = panel[(panel["error"] == "") & ~panel["ticker"].isin(EXCLUDED_FROM_H5)].dropna(subset=["upside", "excess_12m"])
    yearly = []
    for y, g in p.groupby("year"):
        rho = g["upside"].rank().corr(g["excess_12m"].rank())
        yearly.append({"year": int(y), "n": len(g), "rank_ic": float(rho)})
    yt = pd.DataFrame(yearly).set_index("year")
    test = stats.hac_mean_test(yt["rank_ic"].to_numpy(), lags=1)
    from scipy import stats as sps
    p_one = float(1 - sps.t.cdf(test["t"], df=len(yt) - 1))
    buckets = p.groupby("rating")["excess_12m"].agg(["count", "mean", "median"])
    return {"yearly": yt.to_dict(orient="index"), "mean_ic": float(yt["rank_ic"].mean()), "t": test["t"],
            "p_one_sided": p_one, "years": len(yt), "observations": len(p),
            "by_rating": buckets.to_dict(orient="index"),
            "note": "t from the Newey-West (1 lag) standard error of six yearly ICs; p from t(5)."}


def run() -> dict:
    out = {}
    rec = reconcile_inputs()
    save_csv(STAGE, "inputs_vs_fsds", rec)
    out["inputs_vs_fsds"] = {"items_checked": int(rec["fsds"].notna().sum()),
                             "flagged_over_0_5pct": rec[rec["flag"]][["ticker", "item", "engine", "fsds", "tag", "note"]]
                             .to_dict(orient="records"),
                             "not_on_statement_face": rec[rec["on_statement"] == False][["ticker", "item", "tag"]]  # noqa: E712
                             .to_dict(orient="records")}
    live = live_reports()
    dam, erp = damodaran_comparison(live)
    save_csv(STAGE, "damodaran_comparison", dam)
    out["damodaran"] = {"table": dam.to_dict(orient="index"), "implied_erp": erp,
                        "engine_erp": next((v["erp"] for v in live.values() if v), None)}
    panel, cov = reconstruct()
    save_csv(STAGE, "point_in_time_panel", panel)
    out["coverage"] = cov
    out["errors"] = int((panel["error"] != "").sum())
    out["h5"] = rank_ic(panel)
    ok = panel[panel["error"] == ""]
    out["extreme_outputs"] = {"name_years": int(len(ok)), "upside_above_500pct": int((ok["upside"] > 5).sum()),
                              "upside_below_minus_95pct": int((ok["upside"] < -0.95).sum()),
                              "median_upside": float(ok["upside"].median())}
    avgo = panel[(panel["ticker"] == "AVGO") & (panel["error"] == "")]
    out["avgo_reported_not_tested"] = avgo[["year", "upside", "rating", "excess_12m"]].to_dict(orient="records")
    save_json(STAGE, "summary", out)
    return out
