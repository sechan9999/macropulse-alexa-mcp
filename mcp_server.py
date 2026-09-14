"""
mcp_server.py
─────────────────────────────────────────────────────────────────
Self-hosted MCP server exposing Macro Pulse's analysis engines as
tools, for the Amazon Developer Hackathon's Alexa+ track ("Build a
self-hosted MCP server, spec 2025-11-25+, Streamable HTTP").

Every tool here wraps an existing, Streamlit-free function from src/ —
this file adds no new analysis logic of its own, only the MCP surface.

Run locally:
    pip install -r requirements.txt
    python mcp_server.py
    # -> Streamable HTTP endpoint at http://127.0.0.1:8000/mcp

Environment variables (same as the Streamlit app):
    GEMINI_API_KEY   — required for get_macro_briefing
    FRED_API_KEY     — optional, improves credit-spread/yield-curve accuracy
"""

from __future__ import annotations

import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from mcp.server.mcpserver import MCPServer

from src.macro_data import load_macro, load_spy, compute_hf_metrics
from src.quant_signals import run_quant_scan, format_alert_text, DEFAULT_UNIVERSE
from src.macro_briefing import generate_briefing, ANALYSIS_TYPES

mcp = MCPServer(
    "macropulse",
    instructions=(
        "Macro Pulse hedge-fund-style market intelligence: macro regime "
        "classification, per-ticker technical signals, a watchlist scanner, "
        "and an AI macro-analyst briefing. Prices/macro data refresh at "
        "most hourly; treat all output as informational, not trading advice."
    ),
)


@mcp.tool()
def get_macro_regime() -> dict:
    """Get today's macro risk regime (Risk-On / Neutral / Risk-Off) plus the
    S&P 500 level, 10Y yield, yield-curve slope, credit spread, and realized
    volatility it's derived from."""
    df = load_macro()
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
        "yield_curve_inverted": bool(last["yc_slope"] < 0),
        "credit_spread_pct": round(float(last["credit_spread"]) * 100, 2),
        "realized_vol_12m_pct": round(float(last["realized_vol_12m"]) * 100, 1),
        "realized_vol_3m_pct": round(float(last["realized_vol_3m"]) * 100, 1),
    }


@mcp.tool()
def get_quant_signal(ticker: str) -> dict:
    """Get the current trend/momentum/volatility signal for one ticker
    (e.g. 'NVDA'): STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL with a -100..+100
    score and the specific reasons behind it. Informational only, not
    trading advice."""
    ticker = ticker.strip().upper()
    scan = run_quant_scan((ticker,), period="1y")
    if scan.empty:
        return {"ticker": ticker, "error": "no_data"}
    row = scan.iloc[0]
    if isinstance(row.get("_error"), str):
        return {"ticker": ticker, "error": row["_error"]}
    return {
        "ticker": ticker,
        "price": round(float(row["Price"]), 2),
        "signal": row["Signal"],
        "score": int(row["Score"]),
        "vol_regime": row["Vol Regime"],
        "vol_breakout": row["Vol Breakout"],
        "reasons": row["Reasons"],
    }


@mcp.tool()
def get_watchlist_scan(tickers: str = "", top_n: int = 5) -> dict:
    """Scan a watchlist and return the top buy and sell candidates by
    conviction score. Pass tickers as a comma-separated string
    (e.g. 'AAPL,MSFT,NVDA'); leave empty to scan the default watchlist
    (SPY, QQQ, major mega-caps). Informational only, not trading advice."""
    universe = tuple(t.strip().upper() for t in tickers.split(",") if t.strip()) or DEFAULT_UNIVERSE
    scan = run_quant_scan(universe, period="1y")
    if scan.empty:
        return {"error": "no_data", "universe": list(universe)}

    ok = scan[scan["_error"].isna()] if "_error" in scan else scan
    buys = ok[ok["Signal"].isin(["STRONG_BUY", "BUY"])].head(top_n)
    sells = ok[ok["Signal"].isin(["STRONG_SELL", "SELL"])].head(top_n)

    def _rows(frame):
        return [
            {"ticker": r["Ticker"], "price": round(float(r["Price"]), 2),
             "signal": r["Signal"], "score": int(r["Score"]), "reasons": r["Reasons"]}
            for _, r in frame.iterrows()
        ]

    return {
        "universe": list(universe),
        "buy_candidates": _rows(buys),
        "sell_candidates": _rows(sells),
        "summary_text": format_alert_text(scan, top_n=top_n),
    }


@mcp.tool()
def get_risk_metrics(start: str = "", end: str = "") -> dict:
    """Get hedge-fund-style risk metrics (annualized return, Sharpe,
    Sortino, max drawdown, Calmar ratio, win rate, alpha vs. SPY) for the
    S&P 500 over a date range. Dates are 'YYYY-MM-DD'; defaults to
    2015-01-01 through today."""
    start = start or "2015-01-01"
    end = end or datetime.now().strftime("%Y-%m-%d")

    df = load_macro()
    df = df[(df.index >= start) & (df.index <= end)]
    if df.empty:
        return {"error": f"No data in range {start}..{end}"}

    spy_rets = load_spy(start, end)
    m = compute_hf_metrics(df["sp500_ret_m"].dropna(), spy_rets)

    def _r(x):
        return None if x is None or x != x else round(float(x), 4)  # NaN-safe

    return {
        "start": start, "end": end,
        "annualized_return_pct": _r(m["ann_ret"] * 100),
        "annualized_vol_pct": _r(m["ann_vol"] * 100),
        "sharpe": _r(m["sharpe"]),
        "sortino": _r(m["sortino"]),
        "max_drawdown_pct": _r(m["mdd"] * 100),
        "calmar": _r(m["calmar"]),
        "win_rate_pct": _r(m["win_rate"] * 100),
        "alpha_vs_spy_pct": _r(m["alpha"] * 100) if m["alpha"] == m["alpha"] else None,
    }


@mcp.tool()
def get_macro_briefing(analysis_type: str = "Full Macro Briefing", custom_question: str = "",
                        provider: str = "gemini") -> dict:
    """Get an AI-generated hedge-fund-style macro briefing grounded in the
    live dashboard data. analysis_type must be one of: 'Full Macro
    Briefing', 'Regime Deep-Dive', 'Risk Assessment', 'Investment Outlook',
    or 'Custom Question' (pass your question in custom_question).
    provider is 'gemini' (default, requires GEMINI_API_KEY) or 'bedrock'
    (Amazon Bedrock — Amazon Nova Pro by default, requires AWS credentials
    configured on the server via the standard AWS credential chain)."""
    return generate_briefing(analysis_type=analysis_type, custom_question=custom_question, provider=provider)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    # stateless_http=True: no in-memory session affinity needed, so this
    # runs cleanly behind Cloud Run's auto-scaled, potentially multi-instance
    # deployment (same target as the existing Streamlit app).
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port, stateless_http=True)
