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

The payloads live in src/firetv_api.py; the deployed MCP server
(src/alexa_mcp_server.py) serves the same /api/* routes, so the Fire TV
app's default backend is that public ECS service and this file is for
local development.

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

from src import firetv_api
from src.firetv_api import FireTvDataUnavailable

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
        return firetv_api.regime_payload()
    except FireTvDataUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/watchlist-signals")
def get_watchlist_signals(
    user_id: str = Query("", description="If set and tickers is empty, uses this user's saved watchlist"),
    tickers: str = Query("", description="Comma-separated tickers; overrides user_id's saved list"),
    top_n: int = Query(5, ge=1, le=firetv_api.MAX_TOP_N),
) -> dict:
    """Top buy/sell candidates for a watchlist — same source data as the
    Alexa+ MCP tool get_watchlist_scan, shaped for a glanceable TV list."""
    try:
        return firetv_api.watchlist_payload(user_id, tickers, top_n)
    except FireTvDataUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/nvda-danger")
def get_nvda_danger() -> dict:
    """NVIDIA danger-zone reading — same shape as the Alexa+ MCP tool
    get_nvda_danger_zone."""
    try:
        return firetv_api.nvda_payload()
    except FireTvDataUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("REST_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
