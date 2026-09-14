"""
rest_server.py
─────────────────────────────────────────────────────────────────
Lightweight JSON REST API for the Fire TV companion view.

Why this exists instead of reusing mcp_server.py: MCP (Model Context
Protocol) is an *agent* protocol — the Alexa+ track needs it because an
AI agent is the caller. A Fire TV app is a human-facing remote-control
UI, not an agent; making it speak MCP would be an unnatural fit for a
UI client and adds real client-side complexity for no benefit. Fire
TV's own hackathon track has no MCP requirement, so a plain REST/JSON
API — reusing the exact same src/ engines as mcp_server.py — is the
right-sized interface for it.

Run locally:
    pip install -r requirements.txt
    python rest_server.py
    # -> http://127.0.0.1:8080  (docs at /docs)

CORS is wide open (allow_origins=["*"]) because this is read-only,
unauthenticated-by-design market data (see src/firestore_service.py's
docstring re: user_id trust model) served to a TV app, not a browser
handling secrets.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.macro_data import load_macro
from src.quant_signals import run_quant_scan, DEFAULT_UNIVERSE
from src.macro_extras import summarize_nvda_danger_zone
from src.firestore_service import get_watchlist

app = FastAPI(
    title="Macro Pulse — Fire TV API",
    description="Read-only JSON API powering the Fire TV companion view. "
                 "Same analysis engines as the Alexa+ MCP server (mcp_server.py).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/regime")
def get_regime() -> dict:
    """Today's macro regime + the underlying readings — same shape as
    the Alexa+ MCP tool get_macro_regime."""
    try:
        df = load_macro()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to load macro data: {e}")

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    return {
        "as_of": df.index[-1].strftime("%Y-%m-%d"),
        "regime": last["regime"],
        "regime_score": round(float(last["regime_score"]), 2),
        "sp500": round(float(last["sp500"]), 1),
        "sp500_mom_pct": round(float((last["sp500"] / prev["sp500"] - 1) * 100), 2),
        "treasury_10y_pct": round(float(last["dgs10"]), 2),
        "yield_curve_slope_pct": round(float(last["yc_slope"]) * 100, 2),
        "credit_spread_pct": round(float(last["credit_spread"]) * 100, 2),
        "realized_vol_12m_pct": round(float(last["realized_vol_12m"]) * 100, 1),
    }


@app.get("/api/watchlist-signals")
def get_watchlist_signals(
    user_id: str = Query("", description="If set and tickers is empty, uses this user's saved watchlist"),
    tickers: str = Query("", description="Comma-separated tickers; overrides user_id's saved list"),
    top_n: int = Query(5, ge=1, le=20),
) -> dict:
    """Top buy/sell candidates for a watchlist — same source data as the
    Alexa+ MCP tool get_watchlist_scan, shaped for a glanceable TV list."""
    explicit = tuple(t.strip().upper() for t in tickers.split(",") if t.strip())
    if explicit:
        universe = explicit
    elif user_id:
        universe = tuple(get_watchlist(user_id)) or DEFAULT_UNIVERSE
    else:
        universe = DEFAULT_UNIVERSE

    scan = run_quant_scan(universe, period="1y")
    if scan.empty:
        return {"universe": list(universe), "signals": []}

    ok = scan[scan["_error"].isna()] if "_error" in scan else scan
    ok = ok.head(top_n)
    return {
        "universe": list(universe),
        "signals": [
            {"ticker": r["Ticker"], "price": round(float(r["Price"]), 2),
             "signal": r["Signal"], "score": int(r["Score"])}
            for _, r in ok.iterrows()
        ],
    }


@app.get("/api/nvda-danger")
def get_nvda_danger() -> dict:
    """NVIDIA danger-zone reading — same shape as the Alexa+ MCP tool
    get_nvda_danger_zone."""
    result = summarize_nvda_danger_zone()
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("REST_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
