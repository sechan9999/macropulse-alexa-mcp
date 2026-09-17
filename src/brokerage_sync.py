"""
src/brokerage_sync.py
─────────────────────────────────────────────────────────────────
Multi-Asset Portfolio Sync & FOMC Cross-Asset VaR Shock Engine.

Supports:
1. Read-only brokerage connectors:
   - Alpaca Read-Only API
   - Interactive Brokers (IBKR) Flex / Client Portal Schema
   - Plaid / SnapTrade JSON schema
   - Preset institutional portfolios (Macro Balanced, Tech Growth, All-Weather, Risk Parity)
2. Multi-asset classes:
   - Equities (SPY, QQQ, AAPL, NVDA, MSFT, IWM)
   - Fixed Income / Treasuries (TLT, IEF, SHY)
   - Commodities (GLD, USO)
   - Digital Assets (BTC-USD, ETH-USD)
3. Cross-asset covariance matrix, portfolio volatility, parametric VaR (95%/99%),
   and Conditional Value-at-Risk (CVaR / Expected Shortfall).
4. Real-time FOMC Rate Decision Shock Tests:
   - Hawkish Surprise (+50 bps hike)
   - Dovish Pivot (-50 bps cut)
   - Stagflation & Curve Inversion (-50 bps slope, +120 bps credit spread)
   - Liquidity Flash Crunch
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger("MacroPulse-BrokerageSync")

# Default asset class mapping
ASSET_CLASSES = {
    "SPY": "Equities",
    "QQQ": "Equities",
    "AAPL": "Equities",
    "NVDA": "Equities",
    "MSFT": "Equities",
    "IWM": "Equities",
    "TLT": "Fixed Income",
    "IEF": "Fixed Income",
    "SHY": "Fixed Income",
    "GLD": "Commodities",
    "USO": "Commodities",
    "BTC-USD": "Digital Assets",
    "ETH-USD": "Digital Assets",
}

# Empirical FOMC sensitivity betas (% move per 25 bps rate surprise)
FOMC_SENSITIVITIES = {
    # Ticker: (hawkish_pct_per_50bps, stagflation_pct, liquidity_pct)
    "SPY": (-0.035, -0.068, -0.095),
    "QQQ": (-0.045, -0.078, -0.115),
    "AAPL": (-0.038, -0.065, -0.090),
    "NVDA": (-0.052, -0.092, -0.135),
    "MSFT": (-0.036, -0.062, -0.088),
    "IWM": (-0.048, -0.085, -0.120),
    "TLT": (-0.048, -0.025, -0.032),  # Duration hit on hike
    "IEF": (-0.022, -0.012, -0.018),
    "SHY": (-0.005, -0.002, -0.004),
    "GLD": (-0.022,  0.052, -0.041),  # Gold rallies in stagflation
    "USO": (-0.028,  0.038, -0.125),
    "BTC-USD": (-0.058, -0.080, -0.142),
    "ETH-USD": (-0.065, -0.095, -0.160),
}

# Preset Portfolios
PRESET_PORTFOLIOS: Dict[str, Dict[str, float]] = {
    "Macro Balanced (60/40 Modern)": {
        "SPY": 0.40,
        "QQQ": 0.20,
        "TLT": 0.25,
        "GLD": 0.10,
        "BTC-USD": 0.05,
    },
    "Tech & Semiconductor Alpha": {
        "NVDA": 0.35,
        "QQQ": 0.25,
        "MSFT": 0.20,
        "TLT": 0.10,
        "BTC-USD": 0.10,
    },
    "Institutional Risk Parity": {
        "SPY": 0.25,
        "TLT": 0.35,
        "IEF": 0.15,
        "GLD": 0.15,
        "USO": 0.10,
    },
    "All-Weather Cross-Asset": {
        "SPY": 0.30,
        "TLT": 0.25,
        "GLD": 0.25,
        "BTC-USD": 0.10,
        "USO": 0.10,
    },
}


class BrokeragePortfolio:
    """Represents a synchronized multi-asset investment portfolio."""

    def __init__(self, holdings: Dict[str, float], total_value_usd: float = 1_000_000.0, source: str = "Preset"):
        self.holdings = holdings  # ticker -> weight (summing to 1.0) or shares
        self.total_value_usd = total_value_usd
        self.source = source
        self._normalize()

    def _normalize(self):
        total_w = sum(self.holdings.values())
        if total_w > 0:
            self.weights = {k: v / total_w for k, v in self.holdings.items()}
        else:
            self.weights = {"SPY": 1.0}

    @classmethod
    def from_alpaca_json(cls, payload: List[Dict[str, Any]]) -> BrokeragePortfolio:
        """Parses read-only Alpaca /v2/positions endpoint JSON."""
        holdings = {}
        total_val = 0.0
        for pos in payload:
            sym = pos.get("symbol", "").upper()
            mkt_val = float(pos.get("market_value", 0.0))
            if sym and mkt_val > 0:
                holdings[sym] = mkt_val
                total_val += mkt_val
        return cls(holdings=holdings, total_value_usd=total_val or 500_000.0, source="Alpaca Read-Only")

    @classmethod
    def from_ibkr_json(cls, payload: List[Dict[str, Any]]) -> BrokeragePortfolio:
        """Parses read-only Interactive Brokers Client Portal positions payload."""
        holdings = {}
        total_val = 0.0
        for pos in payload:
            sym = pos.get("contractDesc", pos.get("ticker", "")).upper()
            mkt_val = float(pos.get("mktVal", pos.get("marketValue", 0.0)))
            if sym and mkt_val > 0:
                holdings[sym] = mkt_val
                total_val += mkt_val
        return cls(holdings=holdings, total_value_usd=total_val or 750_000.0, source="IBKR Read-Only")

    @classmethod
    def from_plaid_json(cls, payload: Dict[str, Any]) -> BrokeragePortfolio:
        """Parses read-only Plaid /investments/holdings/get payload."""
        holdings = {}
        total_val = 0.0
        holdings_list = payload.get("holdings", [])
        securities = {s.get("security_id"): s.get("ticker_symbol") for s in payload.get("securities", [])}
        for h in holdings_list:
            sec_id = h.get("security_id")
            sym = securities.get(sec_id, "").upper()
            val = float(h.get("institution_value", 0.0))
            if sym and val > 0:
                holdings[sym] = val
                total_val += val
        return cls(holdings=holdings, total_value_usd=total_val or 1_000_000.0, source="Plaid Read-Only")

    @classmethod
    def from_preset(cls, preset_name: str, total_value_usd: float = 1_000_000.0) -> BrokeragePortfolio:
        weights = PRESET_PORTFOLIOS.get(preset_name, PRESET_PORTFOLIOS["Macro Balanced (60/40 Modern)"])
        return cls(holdings=weights, total_value_usd=total_value_usd, source=f"Preset: {preset_name}")

    def get_allocation_by_asset_class(self) -> Dict[str, float]:
        """Returns allocation breakdown grouped by asset class."""
        alloc = {}
        for ticker, weight in self.weights.items():
            ac = ASSET_CLASSES.get(ticker, "Equities")
            alloc[ac] = alloc.get(ac, 0.0) + weight
        return alloc

    def compute_cross_asset_var(self, confidence: int = 95, horizon_days: int = 30) -> Dict[str, Any]:
        """
        Computes multi-asset portfolio parametric VaR and CVaR (Expected Shortfall).
        """
        # Historical baseline asset annualized volatilities
        base_vols = {
            "SPY": 0.16, "QQQ": 0.21, "AAPL": 0.22, "NVDA": 0.42, "MSFT": 0.20,
            "IWM": 0.23, "TLT": 0.15, "IEF": 0.08, "SHY": 0.03, "GLD": 0.14,
            "USO": 0.32, "BTC-USD": 0.58, "ETH-USD": 0.68
        }
        
        tickers = list(self.weights.keys())
        w_vec = np.array([self.weights[t] for t in tickers])
        
        # Build correlation matrix proxy
        n = len(tickers)
        corr_matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                t_i, t_j = tickers[i], tickers[j]
                ac_i, ac_j = ASSET_CLASSES.get(t_i, "Equities"), ASSET_CLASSES.get(t_j, "Equities")
                if ac_i == ac_j:
                    corr = 0.75 if ac_i == "Equities" else 0.85
                elif (ac_i == "Equities" and ac_j == "Fixed Income") or (ac_i == "Fixed Income" and ac_j == "Equities"):
                    corr = 0.15  # Slightly positive in current rate regime
                elif (ac_i == "Commodities" and ac_j == "Equities"):
                    corr = 0.25
                elif (ac_i == "Digital Assets" and ac_j == "Equities"):
                    corr = 0.60
                else:
                    corr = 0.20
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        vols = np.array([base_vols.get(t, 0.20) for t in tickers])
        cov_matrix = np.outer(vols, vols) * corr_matrix

        port_ann_vol = float(np.sqrt(np.dot(w_vec, np.dot(cov_matrix, w_vec))))
        t_factor = np.sqrt(horizon_days / 252.0)
        port_horizon_vol = port_ann_vol * t_factor

        z_score = 1.645 if confidence == 95 else 2.326
        var_pct = -1.0 * z_score * port_horizon_vol
        # CVaR (Expected Shortfall for normal distribution)
        pdf = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (z_score ** 2))
        alpha = (100 - confidence) / 100.0
        cvar_pct = -1.0 * (pdf / alpha) * port_horizon_vol

        var_dollar = var_pct * self.total_value_usd
        cvar_dollar = cvar_pct * self.total_value_usd

        # Marginal risk contribution
        marginal_contrib = np.dot(cov_matrix, w_vec) / (port_ann_vol + 1e-8)
        component_var = (w_vec * marginal_contrib) / (port_ann_vol + 1e-8)
        risk_contrib_by_ticker = {tickers[i]: float(component_var[i]) for i in range(n)}

        return {
            "portfolio_value_usd": self.total_value_usd,
            "annualized_volatility_pct": round(port_ann_vol * 100, 2),
            "confidence_level": confidence,
            "horizon_days": horizon_days,
            "var_pct": round(var_pct * 100, 2),
            "var_dollar": round(var_dollar, 2),
            "cvar_pct": round(cvar_pct * 100, 2),
            "cvar_dollar": round(cvar_dollar, 2),
            "risk_contributions": risk_contrib_by_ticker,
            "asset_classes": self.get_allocation_by_asset_class()
        }

    def simulate_fomc_shock(self, scenario: str = "hawkish_50bps") -> Dict[str, Any]:
        """
        Simulates cross-asset portfolio PnL and VaR under major FOMC rate shocks:
        - 'hawkish_50bps': +50 bps rate hike surprise
        - 'dovish_50bps': -50 bps emergency rate cut
        - 'stagflation_inversion': Curve inversion + credit spread blowout
        - 'liquidity_cascade': Dash-for-cash correlated liquidation
        """
        scenario_names = {
            "hawkish_50bps": "Hawkish Surprise (+50 bps Hike)",
            "dovish_50bps": "Dovish Pivot (-50 bps Cut)",
            "stagflation_inversion": "Stagflation & Curve Inversion Shock",
            "liquidity_cascade": "March 2020 Correlated Liquidity Crunch"
        }
        name = scenario_names.get(scenario, "Hawkish Surprise (+50 bps Hike)")

        ticker_shocks = {}
        total_pnl_pct = 0.0

        for ticker, weight in self.weights.items():
            sens = FOMC_SENSITIVITIES.get(ticker, (-0.035, -0.065, -0.090))
            if scenario == "hawkish_50bps":
                s_pct = sens[0]
            elif scenario == "dovish_50bps":
                s_pct = -1.0 * sens[0] * 1.1  # Dovish pivot creates asymmetric upside in bonds/tech
            elif scenario == "stagflation_inversion":
                s_pct = sens[1]
            elif scenario == "liquidity_cascade":
                s_pct = sens[2]
            else:
                s_pct = sens[0]

            dollar_impact = self.total_value_usd * weight * s_pct
            ticker_shocks[ticker] = {
                "weight_pct": round(weight * 100, 1),
                "shock_return_pct": round(s_pct * 100, 2),
                "pnl_dollar": round(dollar_impact, 2),
                "asset_class": ASSET_CLASSES.get(ticker, "Equities")
            }
            total_pnl_pct += weight * s_pct

        total_pnl_dollar = self.total_value_usd * total_pnl_pct
        stressed_value = self.total_value_usd + total_pnl_dollar

        # Spoken Alexa response for TV
        direction = "loss" if total_pnl_dollar < 0 else "gain"
        spoken = (
            f"Under the {name} scenario, your {self.source} portfolio experiences an estimated "
            f"{direction} of {abs(total_pnl_pct)*100:.1f}%, or {abs(total_pnl_dollar):,.0f} dollars. "
            f"Equities and duration face {'contraction' if total_pnl_dollar < 0 else 'expansion'}, "
            f"bringing stressed portfolio equity to {stressed_value:,.0f} dollars."
        )

        return {
            "status": "success",
            "scenario": scenario,
            "scenario_title": name,
            "portfolio_source": self.source,
            "portfolio_value_usd": self.total_value_usd,
            "stressed_portfolio_value": round(stressed_value, 2),
            "total_pnl_pct": round(total_pnl_pct * 100, 2),
            "total_pnl_dollar": round(total_pnl_dollar, 2),
            "breakdown_by_ticker": ticker_shocks,
            "alexa_spoken_response": spoken
        }
