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
    beta: float | None = None           # None -> mean of 2y weekly and 5y monthly raw betas vs S&P 500, Blume-adjusted
    cost_of_debt: float | None = None   # None -> interest expense / debt, clipped to [3%, 10%]
    tax_rate: float = 0.25              # 21% federal + ~4% blended state
    growth_high: float | None = None    # None -> latest year-to-date revenue growth (10-Q vs prior-year 10-Q),
    #                                     else historical CAGR; clipped to [-10%, 50%]
    terminal_growth: float = 0.025
    projection_years: int = 10
    growth_half_life: float | None = 2.0  # years for excess growth (g - terminal) to halve; None -> linear fade.
    #                                     Revenue growth mean-reverts fast (Chan, Karceski & Lakonishok 2003)
    fcf_margin: float | None = None     # None -> trailing-twelve-month FCFF margin, else 3-year average
    #                                     (FCFF = CFO - CapEx + interest x (1 - tax))


FALLBACK_RISK_FREE = 0.0425
GROWTH_CLIP = (-0.10, 0.50)

# Public URL of the Streamlit app: report links (?ticker=...) from MCP / Alexa and the Buy Zone scanner.
APP_URL = os.getenv("MACROPULSE_APP_URL", "https://hf-macro-dashboard.streamlit.app/")

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "MacroPulse equity-report (set SEC_USER_AGENT=name email)")
