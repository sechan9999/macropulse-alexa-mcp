"""Synthetic (fake) inputs for offline tests and the 'DEMO' ticker.

Numbers produced from this module have nothing to do with any real company. Every output built
from it is stamped "SYNTHETIC DEMO DATA".
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def price_frame(n: int = 756, start: float = 180.0, drift: float = 0.0005, vol: float = 0.017,
                seed: int = 11, end: str = "2026-09-22", market: np.ndarray | None = None,
                beta: float = 1.1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    if market is not None:                      # correlate with a market path -> realistic beta
        r = beta * market + rng.normal(drift, vol * 0.7, n)
    r[int(n * .55):int(n * .62)] -= 0.005
    close = start * np.cumprod(1 + r)
    open_ = close / (1 + r) * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.007, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.007, n)))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                         "Volume": rng.lognormal(16, 0.35, n).astype(int)},
                        index=pd.bdate_range(end=end, periods=n))


def stock_bundle(ticker: str = "DEMO") -> dict:
    from .data_us import normalize_ohlcv
    idx_raw = price_frame(start=5200, drift=0.0004, vol=0.010, seed=5)
    mkt = idx_raw["Close"].pct_change().fillna(0).to_numpy()
    prices = normalize_ohlcv(price_frame(market=mkt))
    index = normalize_ohlcv(idx_raw)[["close"]]
    yrs = [str(y) for y in range(2021, 2026)]
    rev = np.array([52.0, 58.1, 56.4, 63.9, 71.2]) * 1e9
    fin = pd.DataFrame(index=yrs)
    fin["revenue"] = rev
    fin["operating_income"] = rev * np.array([.21, .22, .19, .23, .24])
    fin["net_income"] = fin["operating_income"] * 0.81
    fin["operating_cf"] = fin["net_income"] * np.array([1.28, 1.25, 1.33, 1.27, 1.30])
    fin["capex"] = rev * np.array([.055, .058, .061, .057, .056])
    fin["depreciation"] = rev * np.array([.045, .046, .048, .047, .047])
    fin["dividends"] = fin["net_income"] * 0.22
    fin["interest_expense"] = np.array([0.55, 0.52, 0.60, 0.58, 0.56]) * 1e9
    fin["eps_diluted"] = fin["net_income"] / 1.25e9
    fin["total_assets"] = rev * np.array([1.45, 1.40, 1.44, 1.38, 1.35])
    fin["total_equity"] = fin["total_assets"] * np.array([.46, .48, .47, .50, .52])
    fin["total_liabilities"] = fin["total_assets"] - fin["total_equity"]
    fin["current_assets"] = rev * 0.55
    fin["current_liabilities"] = rev * 0.38
    fin["cash"] = rev * np.array([.18, .20, .17, .21, .23])
    fin["debt"] = rev * np.array([.24, .22, .25, .21, .19])
    fin["fy_end"] = [f"{y}-12-31" for y in yrs]
    fin.index.name = "fiscal_year"
    shares = 1.25e9
    eps = pd.Series(fin["eps_diluted"].values, index=pd.to_datetime([f"{int(y) + 1}-02-15" for y in yrs]))
    eps_asof = eps.reindex(prices.index.union(eps.index)).ffill().reindex(prices.index).bfill()
    return {
        "ticker": ticker, "name": "DEMO Corp (synthetic)", "cik": None, "is_synthetic": True,
        "prices": prices, "index": index, "financials": fin, "shares": shares,
        "last_price": float(prices["close"].iloc[-1]), "pe_history": prices["close"] / eps_asof,
        "eps_latest": float(eps.iloc[-1]),
        "positioning": {"institutional_pct": 0.71, "insider_pct": 0.004, "short_pct_float": 0.018,
                        "short_ratio_days": 1.6, "short_change_vs_prior_month": -0.06,
                        "insider_net_shares_6m": -120000.0},
        "risk_free_live": 0.0431,
        "sources": ["SYNTHETIC DEMO DATA"], "as_of": str(prices.index[-1].date()), "fetched_at": "synthetic",
    }


def company_facts_fixture() -> dict:
    """A small hand-written EDGAR companyfacts document exercising the parser's edge cases:
    tag switch over time, restatement (later filing wins), quarterly rows mixed in, a September
    fiscal year end, split debt tags, and two share classes."""
    def dur(start, end, val, filed, form="10-K", fp="FY"):
        return {"start": start, "end": end, "val": val, "filed": filed, "form": form, "fp": fp}

    def ins(end, val, filed, form="10-K"):
        return {"end": end, "val": val, "filed": filed, "form": form}

    return {"cik": 1234567, "entityName": "FIXTURE INC", "facts": {
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            ins("2024-10-15", 900.0, "2024-11-01"),
            ins("2025-10-17", 600.0, "2025-11-03"), ins("2025-10-17", 400.0, "2025-11-03"),   # two classes
        ]}}},
        "us-gaap": {
            "SalesRevenueNet": {"units": {"USD": [dur("2021-09-26", "2022-09-24", 100.0, "2022-10-28")]}},
            "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
                dur("2022-09-25", "2023-09-30", 110.0, "2023-11-03"),
                dur("2023-10-01", "2024-09-28", 120.0, "2024-11-01"),
                dur("2023-10-01", "2024-09-28", 121.0, "2025-11-03"),        # restated in next 10-K
                dur("2024-09-29", "2025-09-27", 130.0, "2025-11-03"),
                dur("2025-06-29", "2025-09-27", 35.0, "2025-11-03"),          # 3-month row: ignored
                dur("2025-03-30", "2025-06-28", 32.0, "2025-08-01", form="10-Q", fp="Q3"),
            ]}},
            "OperatingIncomeLoss": {"units": {"USD": [dur("2024-09-29", "2025-09-27", 40.0, "2025-11-03")]}},
            "NetIncomeLoss": {"units": {"USD": [dur("2024-09-29", "2025-09-27", 30.0, "2025-11-03"),
                                                 dur("2023-10-01", "2024-09-28", 27.0, "2024-11-01")]}},
            "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
                dur("2024-09-29", "2025-09-27", 38.0, "2025-11-03")]}},
            "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [
                dur("2024-09-29", "2025-09-27", 9.0, "2025-11-03")]}},
            "EarningsPerShareDiluted": {"units": {"USD/shares": [
                dur("2023-10-01", "2024-09-28", 2.70, "2024-11-01"),
                dur("2024-09-29", "2025-09-27", 3.00, "2025-11-03"),
                dur("2023-10-01", "2024-09-28", 2.71, "2025-11-03"),          # comparative: not a new EPS event
            ]}},
            "Assets": {"units": {"USD": [ins("2025-09-27", 300.0, "2025-11-03")]}},
            "StockholdersEquity": {"units": {"USD": [ins("2025-09-27", 120.0, "2025-11-03")]}},
            "LiabilitiesAndStockholdersEquity": {"units": {"USD": [ins("2025-09-27", 300.0, "2025-11-03")]}},
            "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [ins("2025-09-27", 25.0, "2025-11-03")]}},
            "LongTermDebtNoncurrent": {"units": {"USD": [ins("2025-09-27", 50.0, "2025-11-03")]}},
            "LongTermDebtCurrent": {"units": {"USD": [ins("2025-09-27", 5.0, "2025-11-03")]}},
            "CommercialPaper": {"units": {"USD": [ins("2025-09-27", 2.0, "2025-11-03")]}},
            "InterestExpense": {"units": {"USD": [dur("2024-09-29", "2025-09-27", 2.5, "2025-11-03")]}},
        }}}
