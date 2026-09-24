"""Offline stand-ins for FRED, shared by the MCP server tests (no network in CI)."""
import numpy as np
import pandas as pd


def fake_fred(seed: int = 7):
    """(BAA-AAA %, T10Y2Y %) monthly series from 2004-11 to this month, like load_fred_credit_and_slope."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2004-11-01", pd.Timestamp.today().normalize(), freq="MS")
    spread = pd.Series(1.0 + np.abs(np.cumsum(rng.normal(0, 0.05, len(idx)))), index=idx)
    slope = pd.Series(np.cumsum(rng.normal(0, 0.1, len(idx))), index=idx)
    return spread, slope
