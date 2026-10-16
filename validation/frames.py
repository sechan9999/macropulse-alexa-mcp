"""
validation/frames.py
─────────────────────────────────────────────────────────────────
The product's monthly macro frame, rebuilt from the committed raw files instead of live downloads.

build_macro() follows src/macro_data.load_macro step for step (month-end closes labelled by month
start, log returns, realised volatility, 12-1 momentum, FRED credit spread and curve slope, the
expanding-z regime); tests/test_validation_stages.py checks it against load_macro on the same prices.
The only differences are the start date and where the inputs come from:

  start        the product starts in 2005; the pre-registration evaluates each component from the
               first month all of its inputs exist (FIRST_MONTH), with 2005 as a robustness check
  spread/slope latest FRED values by default, or the ALFRED values known at each month end (Stage 1)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.macro_model import add_regime, attach_credit_and_slope
from src.portfolio import TBILL_COLUMN
from validation import data

PRODUCT_START = "2005-01-01"
# First month every input of a component exists: T10Y2Y starts 1976-06 (expected return); the regime needs
# only the S&P 500 and BAA/AAA (1928); the backtest adds the ^IRX T-bill cash rate (1960).
FIRST_MONTH = {"expected_return": "1976-07-01", "regime": "1928-01-01", "backtest": "1960-01-01"}


def _month_close(s: pd.Series, start: str) -> pd.Series:
    s = s.dropna()
    s = s[s.index >= pd.Timestamp(start)]
    return data.monthly_last(s)


def obs_dates(s: pd.Series, start: str) -> pd.Series:
    s = s.dropna()
    s = s[s.index >= pd.Timestamp(start)]
    d = pd.Series(s.index, index=s.index).resample("ME").last()
    d.index = d.index.to_period("M").to_timestamp()
    return d


def fred_spread_slope(spread_src: str = "latest") -> tuple[pd.Series, pd.Series]:
    """(BAA-AAA %, T10Y2Y %) monthly, like macro_model.load_fred_credit_and_slope. spread_src="alfred"
    gives, for each month, the spread as known at that month's end (Stage 1)."""
    if spread_src == "alfred":
        from validation.stage1 import known_at_month_end
        baa, aaa = known_at_month_end("BAA"), known_at_month_end("AAA")
        spread = (baa - aaa).dropna()
    else:
        spread = data.monthly_last((data.fred_latest("BAA") - data.fred_latest("AAA")).dropna())
    slope = data.monthly_last(data.fred_latest("T10Y2Y"))
    return spread, slope


def build_macro(start: str = PRODUCT_START, prices: Optional[dict] = None, spread: Optional[pd.Series] = None,
                slope: Optional[pd.Series] = None, regime_min_periods: Optional[int] = None) -> pd.DataFrame:
    """Same columns and steps as macro_data.load_macro. `prices` maps sp500/vix/dgs10/tbill_3m to daily
    closes (defaults: the committed Yahoo files)."""
    prices = prices or {"sp500": data.yahoo("GSPC"), "vix": data.yahoo("VIX"), "dgs10": data.yahoo("TNX"),
                        TBILL_COLUMN: data.yahoo("IRX")}
    frames = {col: _month_close(s, start) for col, s in prices.items()}
    frames["_obs_date"] = obs_dates(prices["sp500"], start)
    df = pd.DataFrame(frames).sort_index()
    df["_is_demo"] = False
    df.index.name = "date"
    df["sp500_ret_m"] = np.log(df["sp500"]).diff()
    df["realized_vol_12m"] = df["sp500_ret_m"].rolling(12).std() * np.sqrt(12)
    df["realized_vol_3m"] = df["sp500_ret_m"].rolling(3).std() * np.sqrt(12)
    df["momentum_12_1"] = df["sp500_ret_m"].rolling(11).sum().shift(1)
    df["cumret"] = np.exp(df["sp500_ret_m"].cumsum()) * 100
    df["drawdown"] = df["cumret"] / df["cumret"].cummax() - 1
    df["dgs10"] = df["dgs10"].ffill()
    if TBILL_COLUMN in df:
        df[TBILL_COLUMN] = df[TBILL_COLUMN].ffill()
    if spread is None and slope is None:
        spread, slope = fred_spread_slope()
    df = attach_credit_and_slope(df, spread, slope)
    df = add_regime(df) if regime_min_periods is None else add_regime(df, regime_min_periods)
    df["_data_source"] = "validation raw snapshot"
    return df.dropna(subset=["sp500"])


def daily_returns(name: str = "GSPC") -> pd.Series:
    return np.log(data.yahoo(name)).diff().dropna()


def next_month_outcomes(daily_close: pd.Series, index: pd.DatetimeIndex, n_days: int = 21) -> pd.DataFrame:
    """For each monthly row (labelled by month start), the 21 sessions after that month's last close:
    log return and annualised realised volatility (the pre-registered H1 outcome window)."""
    c = daily_close.dropna()
    lr = np.log(c).diff()
    out = []
    for m in index:
        end = m + pd.offsets.MonthEnd(0)
        pos = c.index.searchsorted(end, side="right")             # first session after the month end
        win = lr.iloc[pos: pos + n_days]
        if len(win) < n_days:
            out.append((m, np.nan, np.nan))
        else:
            out.append((m, float(win.sum()), float(win.std(ddof=1) * np.sqrt(252))))
    return pd.DataFrame(out, columns=["date", "next_ret", "next_vol"]).set_index("date")
