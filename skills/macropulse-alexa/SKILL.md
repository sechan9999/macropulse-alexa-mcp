---
name: macropulse-alexa
description: Institutional hedge fund macro intelligence, quantitative risk simulations, and volatility breakout alerts over Model Context Protocol (MCP) and Streamable HTTP.
author: MacroPulse Team
version: 1.0.0
category: Financial Intelligence & Productivity
transports:
  - streamable-http
  - sse
---

# MacroPulse Institutional Quant Copilot for Alexa+

Connects Alexa+ to MacroPulse's 11 quantitative analytical modules, bridging retail voice commands with hedge fund-grade macro risk intelligence.

## Available Agent Skills (MCP Tools)

### 1. `get_macro_regime`
* **Description**: Returns the current market regime (`Risk-On 🟢`, `Neutral 🟡`, `Risk-Off 🔴`), standardized macro stress z-scores, S&P 500 index level, 10Y Treasury yield, and historical regime win rates.
* **Sample Voice Invocations**:
  * *"Alexa, what is today's market regime?"*
  * *"Alexa, are we currently in Risk-On or Risk-Off?"*
  * *"Alexa, give me this morning's macro stress score."*

### 2. `get_rates_and_spreads`
* **Description**: Queries benchmark 10-Year Treasury Yields (^TNX), 10Y-2Y yield curve slope, and BAA-AAA corporate credit risk from FRED.
* **Sample Voice Invocations**:
  * *"Alexa, check the 10-year Treasury yield and yield curve slope."*
  * *"Alexa, are corporate credit spreads widening?"*

### 3. `simulate_portfolio_risk`
* **Parameters**:
  * `ticker` (string, default: "SPY"): Ticker symbol to simulate.
  * `confidence` (integer, default: 95): VaR confidence interval (e.g., 95 or 99).
  * `days` (integer, default: 30): Forward forecast horizon.
  * `n_paths` (integer, default: 5000): Number of Monte Carlo simulation paths.
* **Sample Voice Invocations**:
  * *"Alexa, run a 99% risk simulation on SPY for the next 30 days."*
  * *"Alexa, what is the Value at Risk for QQQ?"*

### 4. `check_nvda_danger_zone`
* **Description**: Evaluates Nvidia's single-name market structure risk, combining 14-day RSI overbought levels, 50-day SMA distance, institutional block trade volume, and synthetic order-flow imbalances.
* **Sample Voice Invocations**:
  * *"Alexa, check if Nvidia is in the Danger Zone."*
  * *"Alexa, how extended is NVDA compared to its moving average?"*

### 5. `scan_quant_signals`
* **Parameters**:
  * `ticker` (string, default: "SPY"): Ticker to evaluate.
* **Description**: Computes 20-day annualized realized volatility, 14-day ATR%, Bollinger Band squeezes (lower 20th percentile BBW), and directional breakout convictions.
* **Sample Voice Invocations**:
  * *"Alexa, are there any volatility squeeze breakouts on SPY?"*
  * *"Alexa, what is the quant conviction score on Microsoft?"*

### 6. `get_expected_returns`
* **Description**: Retrieves expanding-window Ridge regression 12-month forward return forecasts for the S&P 500 with ±1σ confidence bands.
* **Sample Voice Invocations**:
  * *"Alexa, what is the 12-month expected return for the S&P 500?"*

## Architecture & Transport
* **Streamable HTTP Endpoint**: `http://localhost:8000/mcp`
* **SSE Endpoint**: `http://localhost:8000/sse`
* **Messages Endpoint**: `http://localhost:8000/messages`
* **Web Simulator & API**: `http://localhost:8000/alexa/query`
