"""Default assumptions for the US equity report.

Every number here is an input the user can override (Streamlit tab, CLI, or the
Excel model's yellow cells). Nothing is hidden inside the math.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class DCFAssumptions:
    risk_free: float | None = None      # None -> live 10Y Treasury (^TNX / dashboard dgs10); 4.25% if unavailable
    equity_risk_premium: float = 0.050  # Damodaran-style implied ERP range for the US is ~4-6%
    beta: float | None = None           # None -> 2y weekly regression vs S&P 500, Blume-adjusted
    cost_of_debt: float | None = None   # None -> interest expense / debt, clipped to [3%, 10%]
    tax_rate: float = 0.25              # 21% federal + ~4% blended state
    growth_high: float | None = None    # None -> historical revenue CAGR clipped to [-5%, 20%]
    terminal_growth: float = 0.025
    projection_years: int = 5
    fcf_margin: float | None = None     # None -> 3-year average (CFO - CapEx) / revenue


FALLBACK_RISK_FREE = 0.0425

# Public URL of the Streamlit app: report links (?ticker=...) from MCP / Alexa and the Buy Zone scanner.
APP_URL = os.getenv("MACROPULSE_APP_URL", "https://hf-macro-dashboard.streamlit.app/")

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "MacroPulse equity-report (set SEC_USER_AGENT=name email)")
