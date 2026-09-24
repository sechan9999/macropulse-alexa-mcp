"""
src/firetv_api.py
─────────────────────────────────────────────────────────────────
JSON payloads for the Fire TV companion app (firetv-app/). Shared by two servers:

  * rest_server.py        — standalone FastAPI app for local development
  * src/alexa_mcp_server.py — the public ECS deployment, which serves the same /api/* routes
                            next to /mcp so the Fire TV app needs no separate backend

Each function returns exactly the JSON the app's api/macroPulseApi.ts expects, or raises
FireTvDataUnavailable (served as HTTP 503) — never made-up numbers.
"""
from __future__ import annotations

import pandas as pd

from src.macro_data import _ttl_cache, load_macro
from src.quant_signals import DEFAULT_UNIVERSE, run_quant_scan
from src.macro_extras import summarize_nvda_danger_zone
from src.firestore_service import get_watchlist

MAX_TOP_N = 20


class FireTvDataUnavailable(RuntimeError):
    """Live data could not be loaded; the caller answers 503 instead of guessing."""


def _as_of(df: pd.DataFrame) -> str:
    """Date of the latest S&P 500 close behind the last row. Rows are labelled by month start, so
    the label alone would say 2026-09-01 for data through 2026-09-24."""
    if "_obs_date" in df and pd.notna(df["_obs_date"].iloc[-1]):
        return pd.Timestamp(df["_obs_date"].iloc[-1]).strftime("%Y-%m-%d")
    # A stored mart written before _obs_date existed: the latest the month's data could run to.
    month_end = df.index[-1] + pd.offsets.MonthEnd(0)
    return min(month_end, pd.Timestamp.today().normalize()).strftime("%Y-%m-%d")


def regime_payload() -> dict:
    """Today's macro regime and the readings behind it (RegimeResponse)."""
    try:
        df = load_macro()
    except Exception as e:  # noqa: BLE001 - any data-source failure is a 503
        raise FireTvDataUnavailable(f"Failed to load macro data: {e}") from e
    if df is None or df.empty:
        raise FireTvDataUnavailable("Failed to load macro data: no rows")
    if "_is_demo" in df and bool(df["_is_demo"].iloc[-1]):
        # load_macro() substitutes a synthetic frame when every source fails; never show it as live data.
        raise FireTvDataUnavailable("Live macro data is unavailable (only the offline demo frame loaded).")
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    needed = ["regime_score", "sp500", "dgs10", "yc_slope", "credit_spread", "realized_vol_12m"]
    missing = [c for c in needed if c not in df.columns or pd.isna(last[c])]
    if missing:
        # e.g. FRED unreachable: the regime is unavailable rather than computed from a made-up spread.
        raise FireTvDataUnavailable(f"Live macro data is incomplete ({', '.join(missing)} unavailable).")
    return {
        "as_of": _as_of(df),
        "regime": last["regime"],
        "regime_score": round(float(last["regime_score"]), 2),
        "sp500": round(float(last["sp500"]), 1),
        "sp500_mom_pct": round(float((last["sp500"] / prev["sp500"] - 1) * 100), 2),
        "treasury_10y_pct": round(float(last["dgs10"]), 2),
        "yield_curve_slope_pct": round(float(last["yc_slope"]) * 100, 2),
        "credit_spread_pct": round(float(last["credit_spread"]) * 100, 2),
        "realized_vol_12m_pct": round(float(last["realized_vol_12m"]) * 100, 1),
    }


@_ttl_cache(ttl_seconds=600)
def _scan(universe: tuple) -> list:
    """Scored rows for a universe; cached 10 min (the TV app refreshes every 5). Raises (and so
    caches nothing) when no ticker could be scored, e.g. during a data-source outage."""
    scan = run_quant_scan(universe, period="1y")
    ok = scan[scan["_error"].isna()] if not scan.empty and "_error" in scan else scan
    if ok.empty:
        raise FireTvDataUnavailable(f"No market data for {', '.join(universe)}.")
    return [{"ticker": r["Ticker"], "price": round(float(r["Price"]), 2),
             "signal": r["Signal"], "score": int(r["Score"])} for _, r in ok.iterrows()]


def watchlist_payload(user_id: str = "", tickers: str = "", top_n: int = 5) -> dict:
    """Top buy/sell candidates (WatchlistSignalsResponse). `tickers` (comma-separated) wins over
    `user_id`'s saved Firestore watchlist, which wins over the default universe."""
    top_n = max(1, min(int(top_n), MAX_TOP_N))
    explicit = tuple(t.strip().upper() for t in tickers.split(",") if t.strip())
    if explicit:
        universe = explicit
    elif user_id:
        universe = tuple(get_watchlist(user_id)) or DEFAULT_UNIVERSE
    else:
        universe = DEFAULT_UNIVERSE
    return {"universe": list(universe), "signals": _scan(universe)[:top_n]}


def nvda_payload() -> dict:
    """NVIDIA danger-zone reading (NvdaDangerResponse)."""
    result = summarize_nvda_danger_zone()
    if "error" in result:
        raise FireTvDataUnavailable(f"NVDA data unavailable: {result['error']}")
    return result
