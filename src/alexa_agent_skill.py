"""
src/alexa_agent_skill.py
─────────────────────────────────────────────────────────────────
Alexa+ Agent Skill Orchestrator for Macro Pulse & Fire TV.

Handles:
  1. Natural language query intent classification & Fire TV routing
     (e.g., "Alexa, ask MacroPulse for today's market regime",
            "Alexa, show the NVDA Danger Zone on the TV",
            "Alexa, run an FOMC rate shock test on my portfolio")
  2. Execution of quant tools (Regimes, Rates, Monte Carlo VaR, NVDA Danger Zone,
     Quant Signals, Ridge Expected Returns, Multi-Asset FOMC Shocks)
  3. Alexa Presentation Language (APL 2024.1) full-screen TV render directives
     specifically optimized for 1080p and 4K Fire TV / Echo Show viewports.
  4. Spoken voice response synthesis for Alexa Voice Service.
  5. Standard Alexa Skills Kit (ASK) JSON webhook protocol handling.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from src.alexa_mcp_server import (
    execute_get_macro_regime,
    execute_get_rates_and_spreads,
    execute_simulate_portfolio_risk,
    execute_check_nvda_danger_zone,
    execute_scan_quant_signals,
    execute_get_expected_returns,
    execute_ask_macro_analyst,
    MarketDataUnavailable,
)
from src.brokerage_sync import BrokeragePortfolio


class AlexaMacroSkill:
    """Agent Skill orchestrator that maps voice prompts to MacroPulse MCP tools and Fire TV APL."""

    SKILL_NAME = "MacroPulse Institutional Copilot"
    VERSION = "2.0.0"

    INTENT_KEYWORDS = {
        "regime": ["regime", "market state", "risk-on", "risk-off", "macro status", "stress score"],
        "rates": ["rate", "yield", "spread", "10y", "treasury", "curve", "inversion"],
        "fomc": ["fomc", "shock test", "rate shock", "rate hike", "rate cut", "fed hike", "fed decision", "cross-asset shock"],
        "var": ["var", "risk", "simulation", "monte carlo", "shortfall", "cvar", "drawdown risk"],
        "nvda": ["nvda", "nvidia", "danger", "danger zone", "block trade", "order flow"],
        "signals": ["signal", "breakout", "squeeze", "bollinger", "quant score", "technical signal"],
        "returns": ["expected return", "forecast", "ridge", "projection", "forward return"]
    }

    @classmethod
    def identify_intent(cls, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Identifies target tool and extracts arguments from user voice or text prompt."""
        p_low = prompt.lower()
        args: Dict[str, Any] = {}

        # Detect Fire TV display routing
        is_tv_request = any(k in p_low for k in ["on the tv", "on tv", "on the big screen", "to the tv", "on the television"])
        args["is_tv_request"] = is_tv_request

        # Detect ticker
        tickers = ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "TLT", "GLD"]
        found_ticker = None
        for t in tickers:
            if re.search(rf"\b{t.lower()}\b", p_low):
                found_ticker = t
                break

        # Check explicit TV routing keywords
        if is_tv_request:
            if "nvda" in p_low or "danger" in p_low:
                return "check_nvda_danger_zone", args
            elif "fomc" in p_low or "shock" in p_low or "portfolio" in p_low:
                args["scenario"] = "hawkish_50bps"
                return "simulate_fomc_shock", args
            elif "regime" in p_low or "market" in p_low:
                return "get_macro_regime", args
            elif "signal" in p_low or "squeeze" in p_low:
                args["ticker"] = found_ticker or "SPY"
                return "scan_quant_signals", args

        # Check standard intents
        if any(pat in p_low for pat in cls.INTENT_KEYWORDS["fomc"]):
            scenario = "hawkish_50bps"
            if "cut" in p_low or "dovish" in p_low:
                scenario = "dovish_50bps"
            elif "stagflation" in p_low or "inversion" in p_low:
                scenario = "stagflation_inversion"
            elif any(k in p_low for k in ("liquidity", "crunch", "cascade", "flash crash", "dash for cash")):
                scenario = "liquidity_cascade"
            args["scenario"] = scenario
            return "simulate_fomc_shock", args

        for intent, patterns in cls.INTENT_KEYWORDS.items():
            if any(pat in p_low for pat in patterns):
                if intent == "var":
                    args["ticker"] = found_ticker or "SPY"
                    if "99" in p_low:
                        args["confidence"] = 99
                    elif "95" in p_low:
                        args["confidence"] = 95
                    if "60" in p_low:
                        args["days"] = 60
                    elif "90" in p_low:
                        args["days"] = 90
                    return "simulate_portfolio_risk", args
                elif intent == "signals":
                    args["ticker"] = found_ticker or "SPY"
                    return "scan_quant_signals", args
                elif intent == "regime":
                    return "get_macro_regime", args
                elif intent == "rates":
                    return "get_rates_and_spreads", args
                elif intent == "nvda":
                    return "check_nvda_danger_zone", args
                elif intent == "returns":
                    return "get_expected_returns", args

        return "ask_macro_analyst", {"query": prompt, "is_tv_request": is_tv_request}

    @classmethod
    def generate_apl_document(
        cls,
        title: str,
        subtitle: str,
        sentiment: str,
        badges: List[Dict[str, str]],
        speech_text: str
    ) -> Dict[str, Any]:
        """
        Generates an Alexa Presentation Language (APL) 2024.1 document
        optimized for Fire TV 1080p/4K and Echo Show viewports.
        """
        bg_colors = {
            "bullish": "#064e3b",
            "bearish": "#7f1d1d",
            "caution": "#78350f",
            "neutral": "#0f172a"
        }
        accent_colors = {
            "bullish": "#34d399",
            "bearish": "#f87171",
            "caution": "#fbbf24",
            "neutral": "#38bdf8"
        }
        bg = bg_colors.get(sentiment, "#0f172a")
        accent = accent_colors.get(sentiment, "#38bdf8")

        items = [
            {
                "type": "Container",
                "width": "100vw",
                "height": "100vh",
                "paddingLeft": "5vw",
                "paddingRight": "5vw",
                "paddingTop": "5vh",
                "paddingBottom": "5vh",
                "backgroundColor": bg,
                "items": [
                    {
                        "type": "Text",
                        "text": "⚡ MACROPULSE INSTITUTIONAL TV DESK",
                        "color": accent,
                        "fontSize": "24dp",
                        "fontWeight": "800",
                        "letterSpacing": "2dp"
                    },
                    {
                        "type": "Text",
                        "text": title,
                        "color": "#ffffff",
                        "fontSize": "48dp",
                        "fontWeight": "700",
                        "spacing": "12dp"
                    },
                    {
                        "type": "Text",
                        "text": subtitle,
                        "color": "#94a3b8",
                        "fontSize": "26dp",
                        "spacing": "8dp"
                    },
                    {
                        "type": "Container",
                        "direction": "row",
                        "spacing": "24dp",
                        "paddingTop": "32dp",
                        "items": [
                            {
                                "type": "Container",
                                "backgroundColor": "rgba(255, 255, 255, 0.08)",
                                "paddingLeft": "20dp",
                                "paddingRight": "20dp",
                                "paddingTop": "16dp",
                                "paddingBottom": "16dp",
                                "borderRadius": "12dp",
                                "items": [
                                    {
                                        "type": "Text",
                                        "text": b.get("label", "").upper(),
                                        "color": "#94a3b8",
                                        "fontSize": "18dp",
                                        "fontWeight": "600"
                                    },
                                    {
                                        "type": "Text",
                                        "text": b.get("value", ""),
                                        "color": "#ffffff",
                                        "fontSize": "32dp",
                                        "fontWeight": "700",
                                        "spacing": "6dp"
                                    }
                                ]
                            }
                            for b in badges[:4]
                        ]
                    }
                ]
            }
        ]

        return {
            "type": "APL",
            "version": "2024.1",
            "mainTemplate": {
                "parameters": ["payload"],
                "items": items
            }
        }

    @classmethod
    def execute(cls, prompt: str) -> Dict[str, Any]:
        """Dispatches prompt to a tool. If live market data is unavailable the result is an
        honest spoken error card (status "error"), never an answer from made-up numbers."""
        try:
            return cls._execute(prompt)
        except MarketDataUnavailable as e:
            logging.getLogger("MacroPulse-AlexaSkill").warning("Market data unavailable: %s", e)
            speech = "I can't reach live market data right now, so I won't guess. Please try again in a minute."
            card = {"title": "Market data unavailable", "subtitle": "Live data could not be fetched",
                    "badges": [], "sentiment": "neutral"}
            return {
                "status": "error",
                "prompt": prompt,
                "is_tv_request": False,
                "tool_selected": "market_data_unavailable",
                "tool_args": {},
                "spoken_response": speech,
                "display_card": card,
                "apl_document": cls.generate_apl_document(
                    title=card["title"], subtitle=card["subtitle"], sentiment="neutral",
                    badges=[], speech_text=speech),
                "raw_payload": {"status": "error", "error": "market_data_unavailable", "message": str(e)},
                "latency_ms": 0,
                "timestamp": datetime.now().isoformat(),
            }

    @classmethod
    def _execute(cls, prompt: str) -> Dict[str, Any]:
        tool_name, args = cls.identify_intent(prompt)
        start_time = datetime.now()
        is_tv = args.get("is_tv_request", False)

        if tool_name == "get_macro_regime":
            raw = execute_get_macro_regime()
            card = {
                "title": f"Macro Regime: {raw['regime']}",
                "subtitle": f"Stress Z-Score: {raw['stress_score']:+.2f}",
                "badges": [
                    {"label": "S&P 500", "value": f"${raw['sp500']:,.0f}"},
                    {"label": "10Y Yield", "value": f"{raw['treasury_10y_yield']:.2f}%"},
                    {"label": "VIX", "value": f"{raw['vix']:.1f}"},
                    {"label": "Historical Win Rate", "value": f"{raw['regime_win_rate_pct']:.0f}%"}
                ],
                "sentiment": "bullish" if "Risk-On" in raw['regime'] else ("bearish" if "Risk-Off" in raw['regime'] else "neutral")
            }

        elif tool_name == "simulate_fomc_shock":
            scenario = args.get("scenario", "hawkish_50bps")
            port = BrokeragePortfolio.from_preset("Macro Balanced (60/40 Modern)", 1_000_000.0)
            raw = port.simulate_fomc_shock(scenario)
            card = {
                "title": f"FOMC Shock: {raw['scenario_title']}",
                "subtitle": f"Portfolio Impact: {raw['total_pnl_pct']:+.2f}% (${raw['total_pnl_dollar']:+,.0f})",
                "badges": [
                    {"label": "PnL Return", "value": f"{raw['total_pnl_pct']:+.1f}%"},
                    {"label": "PnL Dollar", "value": f"${raw['total_pnl_dollar']:+,.0f}"},
                    {"label": "Stressed Value", "value": f"${raw['stressed_portfolio_value']:,.0f}"},
                    {"label": "Scenario", "value": scenario.upper().replace('_', ' ')}
                ],
                "sentiment": "bearish" if raw['total_pnl_pct'] < 0 else "bullish"
            }

        elif tool_name == "get_rates_and_spreads":
            raw = execute_get_rates_and_spreads()
            card = {
                "title": f"10Y Yield: {raw['treasury_10y']:.2f}%",
                "subtitle": f"Curve Slope: {raw['curve_slope_bps']:+.0f} bps ({raw['curve_status']})",
                "badges": [
                    {"label": "10Y Treasury", "value": f"{raw['treasury_10y']:.2f}%"},
                    {"label": "Short Rate", "value": f"{raw['short_rate']:.2f}%"},
                    {"label": "Credit Spread", "value": f"{raw['credit_spread_bps']:.0f} bps"},
                    {"label": "Curve Status", "value": raw['curve_status'].upper()}
                ],
                "sentiment": "neutral"
            }

        elif tool_name == "simulate_portfolio_risk":
            t_args = {k: v for k, v in args.items() if k != "is_tv_request"}
            raw = execute_simulate_portfolio_risk(**t_args)
            card = {
                "title": f"Monte Carlo Risk ({raw['ticker']})",
                "subtitle": f"{raw['paths_simulated']:,} Paths over {raw['days_horizon']} Trading Days",
                "badges": [
                    {"label": f"VaR {raw['confidence_level']}%", "value": f"{raw['var_pct']:+.1f}%"},
                    {"label": "Expected Shortfall (CVaR)", "value": f"{raw['cvar_pct']:+.1f}%"},
                    {"label": "Median Return", "value": f"{raw['median_return_pct']:+.1f}%"},
                    {"label": "Current Price", "value": f"${raw['current_price']:,.2f}"}
                ],
                "sentiment": "bearish" if raw['var_pct'] < -7.0 else "neutral"
            }

        elif tool_name == "check_nvda_danger_zone":
            raw = execute_check_nvda_danger_zone()
            card = {
                "title": f"Nvidia Danger Zone: {raw['danger_level']}",
                "subtitle": f"Danger Index: {raw['danger_index']:.3f} / 1.0",
                "badges": [
                    {"label": "NVDA Price", "value": f"${raw['current_price']:,.2f}"},
                    {"label": "RSI (14)", "value": f"{raw['rsi_14']:.1f}"},
                    {"label": "SMA50 Gap", "value": f"{raw['sma50_distance_pct']:+.1f}%"},
                    {"label": "Relative Volume", "value": f"{raw['relative_volume']:.1f}×"}
                ],
                "sentiment": "bearish" if "DANGER" in raw['danger_level'] else ("caution" if "CAUTION" in raw['danger_level'] else "bullish")
            }

        elif tool_name == "scan_quant_signals":
            t_args = {k: v for k, v in args.items() if k != "is_tv_request"}
            raw = execute_scan_quant_signals(**t_args)
            card = {
                "title": f"Quant Signal: {raw['signal']}",
                "subtitle": f"{raw['ticker']} Conviction Score: {raw['conviction_score']:+d}",
                "badges": [
                    {"label": "Signal", "value": raw['signal']},
                    {"label": "20d Realized Vol", "value": f"{raw['realized_vol_20d_pct']:.1f}%"},
                    {"label": "BB Squeeze", "value": "ACTIVE ⚡" if raw['bb_squeeze_active'] else "Normal"},
                    {"label": "Price", "value": f"${raw['price']:,.2f}"}
                ],
                "sentiment": "bullish" if "BUY" in raw['signal'] else ("bearish" if "SELL" in raw['signal'] else "neutral")
            }

        elif tool_name == "get_expected_returns":
            raw = execute_get_expected_returns()
            card = {
                "title": "Reference Expected Return",
                "subtitle": "12-Month S&P 500 (static reference estimate)",
                "badges": [
                    {"label": "Expected Return", "value": f"{raw['expected_return_pct']:+.1f}%"},
                    {"label": "1σ Lower Bound", "value": f"{raw['lower_1sigma_pct']:+.1f}%"},
                    {"label": "1σ Upper Bound", "value": f"{raw['upper_1sigma_pct']:+.1f}%"},
                    {"label": "Horizon", "value": "12 Months"}
                ],
                "sentiment": "bullish" if raw['expected_return_pct'] > 6.0 else "neutral"
            }

        else:
            raw = execute_ask_macro_analyst(args.get("query", prompt))
            card = {
                "title": "Senior Macro Analyst Briefing",
                "subtitle": "MacroPulse AI Intelligence",
                "badges": [
                    {"label": "Analysis Type", "value": "Macro Commentary"},
                    {"label": "Model", "value": "Gemini Copilot"},
                    {"label": "Desk", "value": "Institutional"}
                ],
                "sentiment": "neutral"
            }

        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        spoken_text = raw.get("alexa_spoken_response", "")
        if is_tv:
            spoken_text = f"Displaying {card['title']} on your TV. " + spoken_text

        # Build APL directive
        apl_doc = cls.generate_apl_document(
            title=card["title"],
            subtitle=card["subtitle"],
            sentiment=card.get("sentiment", "neutral"),
            badges=card.get("badges", []),
            speech_text=spoken_text
        )

        return {
            "status": "success",
            "prompt": prompt,
            "is_tv_request": is_tv,
            "tool_selected": tool_name,
            "tool_args": args,
            "spoken_response": spoken_text,
            "display_card": card,
            "apl_document": apl_doc,
            "raw_payload": raw,
            "latency_ms": latency_ms,
            "timestamp": datetime.now().isoformat()
        }

    @classmethod
    def handle_ask_request(cls, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles official Alexa Skills Kit (ASK) JSON webhook requests
        from Amazon Alexa Cloud Services.
        """
        req_type = request_body.get("request", {}).get("type", "LaunchRequest")
        
        if req_type == "LaunchRequest":
            speech = "Welcome to MacroPulse. You can ask for today's market regime, or say show the NVDA Danger Zone on the TV."
            return {
                "version": "1.0",
                "response": {
                    "outputSpeech": {"type": "PlainText", "text": speech},
                    "shouldEndSession": False
                }
            }

        elif req_type == "IntentRequest":
            intent = request_body.get("request", {}).get("intent", {})
            intent_name = intent.get("name", "")

            if intent_name in ["AMAZON.StopIntent", "AMAZON.CancelIntent"]:
                return {
                    "version": "1.0",
                    "response": {
                        "outputSpeech": {"type": "PlainText", "text": "Goodbye from MacroPulse."},
                        "shouldEndSession": True
                    }
                }
            elif intent_name == "AMAZON.HelpIntent":
                return {
                    "version": "1.0",
                    "response": {
                        "outputSpeech": {
                            "type": "PlainText",
                            "text": "You can ask for today's market regime, check the NVDA danger zone on your TV, or simulate an FOMC rate shock test."
                        },
                        "shouldEndSession": False
                    }
                }

            # Map custom intents
            if intent_name == "MacroRegimeIntent":
                prompt = "Alexa, ask MacroPulse for today's market regime"
            elif intent_name == "ShowOnTvIntent":
                target_slot = intent.get("slots", {}).get("TargetView", {}).get("value", "NVDA Danger Zone")
                prompt = f"Alexa, show {target_slot} on the TV"
            elif intent_name == "PortfolioShockTestIntent":
                scenario_slot = intent.get("slots", {}).get("Scenario", {}).get("value", "hawkish_50bps")
                prompt = f"Alexa, run FOMC {scenario_slot} shock test"
            elif intent_name == "NvdaDangerZoneIntent":
                prompt = "Alexa, check if Nvidia is in the Danger Zone"
            elif intent_name == "QuantSignalIntent":
                ticker_slot = intent.get("slots", {}).get("Ticker", {}).get("value", "SPY")
                prompt = f"Alexa, scan quant signals on {ticker_slot}"
            else:
                prompt = "Alexa, what is today's market regime?"

            res = cls.execute(prompt)
            spoken = res["spoken_response"]
            apl = res["apl_document"]

            response_payload = {
                "version": "1.0",
                "response": {
                    "outputSpeech": {"type": "PlainText", "text": spoken},
                    "card": {
                        "type": "Standard",
                        "title": res["display_card"]["title"],
                        "text": res["display_card"]["subtitle"]
                    },
                    "directives": [
                        {
                            "type": "Alexa.Presentation.APL.RenderDocument",
                            "token": f"macropulse-{res['tool_selected']}",
                            "document": apl
                        }
                    ],
                    "shouldEndSession": True
                }
            }
            return response_payload

        return {
            "version": "1.0",
            "response": {
                "outputSpeech": {"type": "PlainText", "text": "MacroPulse session completed."},
                "shouldEndSession": True
            }
        }
