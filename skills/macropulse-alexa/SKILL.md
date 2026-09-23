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

### 7. `get_equity_report`
* **Parameters**:
  * `ticker` (string, default: "AAPL"): US stock ticker of a 10-K filer (spoken company names such as "Apple" are mapped in the voice layer).
* **Description**: Headline equity research for one stock: probability-weighted DCF fair value (bear/base/bull weights set by the live MacroPulse regime), base-case DCF, WACC, rule-based rating, stop / target levels with 6-month touch probabilities, weekly trend and the daily candlestick/indicator screen (`bullish_count`, `bearish_count`, `bull_score`, `bear_score`, `overall`). Fundamentals come from SEC EDGAR 10-K XBRL filings. ETFs and foreign 20-F/40-F filers return a clear "not supported" error instead of a guess. The full Excel model, Word note and dashboard are downloadable from the dashboard's 📑 Equity Report tab (`report_url`).
* **Sample Voice Invocations**:
  * *"Alexa, give me an equity report on Apple."*
  * *"Alexa, what is the fair value of NVDA?"*
  * *"Alexa, show the Microsoft report on the TV."*

### 8. `get_morning_brief`
* **Parameters**: none.
* **Description**: One ~30-second pre-market answer built for an Alexa Routine: last S&P 500 close and day change, macro regime and stress score, 10Y yield and curve slope, SPY quant signal, and the top buy/sell names from the latest daily quant-signal scan (`signals/`, skipped when older than 4 days). The regime and rates are required; the other parts are left out (listed in `sections_skipped`) rather than guessed when unavailable.
* **Alexa Routine setup**: Alexa app → More → Routines → + → When: *Alexa Says* "good morning" (or a schedule, e.g. weekdays 8:00) → Add action → *Custom* → type `ask MacroPulse for my morning brief`. (A *Skills* action opens MacroPulse with a launch request instead, which only greets.)
* **Sample Voice Invocations**:
  * *"Alexa, ask MacroPulse for my morning brief."*
  * *"Alexa, ask MacroPulse to catch me up on the markets."*
  * *"Alexa, show the morning brief on the TV."*

## Architecture & Transport
* **Streamable HTTP Endpoint**: `http://localhost:8000/mcp`
* **SSE Endpoint**: `http://localhost:8000/sse`
* **Messages Endpoint**: `http://localhost:8000/messages`
* **Web Simulator & API**: `http://localhost:8000/alexa/query`
