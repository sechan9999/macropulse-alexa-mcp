"""
src/macro_briefing.py
─────────────────────────────────────────────────────────────────
Streamlit-free copy of app.py's Tab 8 "Gemini AI Macro Analyst" logic
(context building + prompt templates + the Gemini call itself), so it
can be invoked headlessly — currently by mcp_server.py.

Duplicated rather than imported from app.py for the same reason as
src/macro_data.py: app.py runs Streamlit UI code at import time.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_OK = True
except Exception:
    genai = None
    genai_types = None
    _GENAI_OK = False

try:
    import boto3
    _BOTO3_OK = True
except Exception:
    boto3 = None
    _BOTO3_OK = False

from src.macro_data import load_macro, load_spy, compute_hf_metrics

# "gemini-2.0-flash" (what app.py's Tab 8 used to hard-code) is no longer
# served — Google's API 404s and points at this model instead (confirmed
# live 2026-09-14). app.py's Tab 8 was updated to match in the same fix.
GEMINI_MODEL = "gemini-3.6-flash"

# Amazon Nova Pro — Amazon's own foundation model on Bedrock, generally
# available without the per-model access request some third-party Bedrock
# models (e.g. Anthropic Claude) need in a fresh AWS account. Override via
# BEDROCK_MODEL_ID if you've enabled a different model in your account.
BEDROCK_DEFAULT_MODEL = "amazon.nova-pro-v1:0"

PROVIDERS = ("gemini", "bedrock")

SYSTEM_INSTRUCTION = (
    "You are a senior quantitative macro analyst at a leading hedge fund. "
    "Your analysis is precise, data-driven, and actionable. "
    "You interpret financial data with institutional rigor."
)

ANALYSIS_TYPES = (
    "Full Macro Briefing",
    "Regime Deep-Dive",
    "Risk Assessment",
    "Investment Outlook",
    "Custom Question",
)


def _get_gemini_key() -> Optional[str]:
    return os.environ.get("GEMINI_API_KEY")


def _call_gemini(prompt: str, api_key: str, model: Optional[str] = None) -> str:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model or GEMINI_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
    )
    return response.text


def _call_bedrock(prompt: str, model_id: Optional[str] = None, region: Optional[str] = None) -> str:
    """Call Amazon Bedrock's unified Converse API. Uses the standard AWS
    credential chain (env vars, ~/.aws/credentials, or an IAM role) — no
    key is threaded through by hand the way the Gemini path does."""
    client = boto3.client("bedrock-runtime", region_name=region or os.environ.get("AWS_REGION", "us-east-1"))
    response = client.converse(
        modelId=model_id or os.environ.get("BEDROCK_MODEL_ID", BEDROCK_DEFAULT_MODEL),
        system=[{"text": SYSTEM_INSTRUCTION}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
    )
    return response["output"]["message"]["content"][0]["text"]


def build_macro_context(df: pd.DataFrame, m: dict, ytd: float, d_start, d_end) -> str:
    """Same context block Tab 8 sends to Gemini — kept byte-for-byte
    compatible so prompts behave identically to the dashboard."""
    latest_data = df.iloc[-1]
    prev_data = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    ytd_pct = ytd if np.isfinite(ytd) else 0.0
    regime_now = df["regime"].iloc[-1]
    regime_dist = df["regime"].value_counts(normalize=True) * 100

    return f"""
## Current Macro Dashboard Data ({df.index[-1].strftime('%B %Y')})

### Market Data
- S&P 500: {latest_data['sp500']:,.0f} ({(latest_data['sp500']/prev_data['sp500']-1)*100:+.2f}% MoM)
- YTD Return: {ytd_pct:.1f}%
- 10Y Treasury Yield: {latest_data['dgs10']:.2f}% ({(latest_data['dgs10']-prev_data['dgs10'])*100:+.0f}bps MoM)
- 12M Realized Vol: {latest_data['realized_vol_12m']*100:.1f}%
- 3M Realized Vol: {latest_data['realized_vol_3m']*100:.1f}%

### Macro Regime
- Current Regime: {regime_now}
- Regime Score: {latest_data['regime_score']:.2f} (positive = risk-off)
- Historical: Risk-On {regime_dist.get('Risk-On 🟢', 0):.0f}% | Neutral {regime_dist.get('Neutral 🟡', 0):.0f}% | Risk-Off {regime_dist.get('Risk-Off 🔴', 0):.0f}%

### Risk Metrics
- Annualized Return ({d_start} to {d_end}): {m['ann_ret']*100:.1f}%
- Sharpe Ratio: {m['sharpe']:.2f}
- Sortino Ratio: {'{:.2f}'.format(m['sortino']) if np.isfinite(m['sortino']) else 'N/A'}
- Max Drawdown: {m['mdd']*100:.1f}%
- Calmar Ratio: {'{:.2f}'.format(m['calmar']) if np.isfinite(m['calmar']) else 'N/A'}
- Win Rate: {m['win_rate']*100:.0f}%

### Yield Curve
- Yield Curve Slope: {latest_data['yc_slope']*100:.2f}% ({'Inverted - recession signal' if latest_data['yc_slope'] < 0 else 'Normal'})
- Credit Spread Proxy: {latest_data['credit_spread']*100:.2f}%

### Momentum
- 12-1 Momentum Signal: {latest_data['momentum_12_1']*100:.2f}%
"""


def build_prompt(analysis_type: str, macro_context: str, latest_data: pd.Series, m: dict, custom_question: str = "") -> str:
    if analysis_type == "Full Macro Briefing":
        return f"""{macro_context}

As a senior hedge fund macro analyst, write a concise but rigorous investment briefing covering:
1. **Macro Environment** — Rate regime, credit conditions, volatility environment
2. **Regime Assessment** — What the current regime implies for asset allocation
3. **Key Risks** — Top 3 tail risks the data is signaling
4. **Actionable View** — Specific positioning recommendations (equities, duration, credit, commodities)
5. **Monitoring Triggers** — What metrics to watch for regime change

Use precise, institutional language. Be direct and opinionated."""

    if analysis_type == "Regime Deep-Dive":
        return f"""{macro_context}

Focus exclusively on the macro regime analysis:
1. What does the current regime score of {latest_data['regime_score']:.2f} imply?
2. How does current vol ({latest_data['realized_vol_12m']*100:.1f}% 12M) and credit spread ({latest_data['credit_spread']*100:.2f}%) compare to historical norms?
3. What typically happens next when transitioning from this regime?
4. How should a long/short equity fund position across regime transitions?"""

    if analysis_type == "Risk Assessment":
        return f"""{macro_context}

Provide a rigorous risk assessment:
1. Is the Sharpe of {m['sharpe']:.2f} and Sortino of {'{:.2f}'.format(m['sortino']) if np.isfinite(m['sortino']) else 'N/A'} adequate for current conditions?
2. Does the {m['mdd']*100:.1f}% max drawdown represent historically elevated risk?
3. What does the {latest_data['yc_slope']*100:.2f}% yield curve slope imply for forward equity returns?
4. Tail risk scenarios: what would trigger a -20%/-30% equity move from here?"""

    if analysis_type == "Investment Outlook":
        return f"""{macro_context}

Give a 3-6 month forward investment outlook:
1. Expected return range for US equities given current macro
2. Fixed income: duration add or reduce?
3. Credit: tighten or widen spreads?
4. Commodities: gold and oil direction
5. Key catalyst calendar to watch"""

    # Custom Question
    return f"""{macro_context}

User Question: {custom_question}

Answer as a senior macro analyst using the data above."""


def _load_dashboard_state(d_start: Optional[str] = None, d_end: Optional[str] = None):
    """Recreate the (df, m, ytd) state app.py builds from sidebar widgets,
    using sensible non-interactive defaults when no range is given."""
    d_start = d_start or "2015-01-01"
    d_end = d_end or datetime.now().strftime("%Y-%m-%d")

    df_raw = load_macro()
    df = df_raw[(df_raw.index >= pd.Timestamp(d_start)) & (df_raw.index <= pd.Timestamp(d_end))].copy()
    if df.empty or len(df) < 2:
        raise ValueError(f"No macro data in range {d_start}..{d_end}")

    spy_rets = load_spy(d_start, d_end)
    m = compute_hf_metrics(df["sp500_ret_m"].dropna(), spy_rets)

    last = df.iloc[-1]
    cur_yr = df[df.index.year == datetime.now().year]
    ytd = (last["sp500"] / float(cur_yr["sp500"].iloc[0]) - 1) * 100 if not cur_yr.empty else float("nan")

    return df, m, ytd, d_start, d_end


def generate_briefing(analysis_type: str, custom_question: str = "",
                       d_start: Optional[str] = None, d_end: Optional[str] = None,
                       api_key: Optional[str] = None, provider: str = "gemini",
                       model_id: Optional[str] = None) -> dict:
    """
    Run the full Tab-8 pipeline headlessly: load data, build context,
    build the prompt, call the chosen LLM provider, return the analysis.

    provider: "gemini" (default, matches the live dashboard's Tab 8 exactly)
              or "bedrock" (Amazon Bedrock via boto3 — AWS Builder mini
              challenge integration; uses the standard AWS credential chain).
    model_id: optional override for the provider's default model — a Gemini
              model name for provider="gemini", or a Bedrock model ID for
              provider="bedrock" (must be enabled for the account/region).

    Returns {"text": str, ...} on success, {"error": str} on failure — never
    raises, so MCP tool callers get a clean structured result either way.
    """
    if analysis_type not in ANALYSIS_TYPES:
        return {"error": f"Unknown analysis_type {analysis_type!r}. Choose one of {ANALYSIS_TYPES}."}
    if analysis_type == "Custom Question" and not custom_question.strip():
        return {"error": "custom_question is required when analysis_type is 'Custom Question'."}
    if provider not in PROVIDERS:
        return {"error": f"Unknown provider {provider!r}. Choose one of {PROVIDERS}."}

    try:
        df, m, ytd, d_start, d_end = _load_dashboard_state(d_start, d_end)
    except Exception as e:
        return {"error": f"Failed to load macro data: {e}"}

    macro_context = build_macro_context(df, m, ytd, d_start, d_end)
    prompt = build_prompt(analysis_type, macro_context, df.iloc[-1], m, custom_question)

    if provider == "bedrock":
        if not _BOTO3_OK:
            return {"error": "boto3 package is not installed."}
        resolved_model = model_id or os.environ.get("BEDROCK_MODEL_ID", BEDROCK_DEFAULT_MODEL)
        try:
            text = _call_bedrock(prompt, model_id=resolved_model)
        except Exception as e:
            return {"error": f"Bedrock error: {e}"}
        model_used = resolved_model
    else:
        if not _GENAI_OK:
            return {"error": "google-genai package is not installed."}
        key = api_key or _get_gemini_key()
        if not key:
            return {"error": "No Gemini API key configured (set GEMINI_API_KEY)."}
        resolved_model = model_id or GEMINI_MODEL
        try:
            text = _call_gemini(prompt, key, model=resolved_model)
        except Exception as e:
            return {"error": f"Gemini API error: {e}"}
        model_used = resolved_model

    return {
        "text": text,
        "provider": provider,
        "model": model_used,
        "as_of": df.index[-1].strftime("%Y-%m-%d"),
        "regime": df["regime"].iloc[-1],
    }
