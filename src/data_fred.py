import os
import pandas as pd

SERIES = {
  "dgs10":"DGS10",
  "tb3ms":"TB3MS",
  "cpi":"CPIAUCSL",
  "unrate":"UNRATE",
  "indpro":"INDPRO",
  "t10y2y":"T10Y2Y",
  "baa":"BAA",
  "aaa":"AAA",
  "nfci":"NFCI"
}

# FRED dates a monthly value by the first day of the month it describes (March CPI = 2024-03-01)
# and a weekly NFCI value by the Friday its week ends, but neither is public on that date: March
# CPI comes out around April 10-15. Each value is moved to the day it is first published —
# end of its observation period plus the release delay below (a conservative upper bound) —
# before resampling, so a month-end row only holds numbers that were public by that date.
MONTHLY = {"tb3ms", "cpi", "unrate", "indpro", "baa", "aaa"}
RELEASE_LAG_DAYS = {
    "cpi": 15,        # BLS CPI: ~10th-15th of the next month
    "unrate": 10,     # BLS Employment Situation: first Friday of the next month
    "indpro": 18,     # Fed G.17: ~15th-17th of the next month
    "baa": 5,         # Moody's monthly averages (Fed H.15): first days of the next month
    "aaa": 5,
    "tb3ms": 5,
    "nfci": 5,        # Chicago Fed: Wednesday after the week ending on the Friday it is dated by
    "dgs10": 1,       # daily H.15 yields: next business day
    "t10y2y": 1,
}


def to_release_dates(s: pd.Series, col: str) -> pd.Series:
    """Re-index a raw FRED series by the date each value was first published."""
    s = s.dropna().copy()
    idx = pd.to_datetime(s.index)
    if col in MONTHLY:
        idx = idx + pd.offsets.MonthEnd(0)                  # end of the observation month
    s.index = idx + pd.Timedelta(days=RELEASE_LAG_DAYS.get(col, 0))
    return s.rename(col)


def to_monthly_point_in_time(raw: dict) -> pd.DataFrame:
    """{col: raw FRED series} → month-end frame holding, for each month, the latest value published
    by that month-end. Revisions are not modelled: FRED serves today's vintage (use ALFRED for
    first-release values)."""
    frames = [to_release_dates(s, col) for col, s in raw.items()]
    df = pd.concat(frames, axis=1).sort_index()
    return df.resample("ME").last().ffill()


def load_fred(start="1990-01-01"):
    key = os.getenv("FRED_API_KEY")
    raw = {}

    if key:
        from fredapi import Fred
        print("FRED_API_KEY found, using official fredapi...")
        fred = Fred(api_key=key)
        for col, sid in SERIES.items():
            print(f"Loading {sid}...")
            raw[col] = pd.Series(fred.get_series(sid), name=col)
    else:
        import pandas_datareader as pdr
        print("FRED_API_KEY not found. Using pandas_datareader fallback (No API key needed)...")
        for col, sid in SERIES.items():
            print(f"Loading {sid}...")
            try:
                raw[col] = pdr.get_data_fred(sid, start=start).iloc[:, 0]
            except Exception as e:
                print(f"Failed to load {sid}: {e}")

    m = to_monthly_point_in_time(raw)
    return m[m.index >= pd.to_datetime(start)]
