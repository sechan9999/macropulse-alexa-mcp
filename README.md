# ⚡ Macro Pulse: Hedge Fund Multi-Factor Macro Dashboard & Quant Signals

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hf-macro-dashboard.streamlit.app/)
[![Google Cloud Run](https://img.shields.io/badge/Google%20Cloud%20Run-Live-4285F4?logo=google-cloud&logoColor=white)](https://macro-pulse-652787573242.us-central1.run.app)
[![AWS S3](https://img.shields.io/badge/AWS%20S3-Data%20Lake-569A31?logo=amazons3&logoColor=white)](https://aws.amazon.com/s3/)
[![Daily Quant Signal](https://github.com/sechan9999/hf-macro-dashboard/actions/workflows/daily-quant-signal.yml/badge.svg)](https://github.com/sechan9999/hf-macro-dashboard/actions/workflows/daily-quant-signal.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Macro Pulse** is an institutional-grade financial intelligence and quantitative screening platform. It bridges the gap between retail technical indicators (simple 14-day RSI and moving averages) and hedge-fund macro risk management (credit spreads, yield-curve dynamics, regime-switching models, walk-forward strategy backtesting, and AI-driven macro commentary).

Powered by **Streamlit**, **Plotly**, **yfinance**, **FRED**, **Scikit-learn**, **AWS S3**, and **Google Gemini AI**, Macro Pulse integrates 14 specialized analytical modules into a unified, free, real-time web dashboard.

---

## 🔗 Live Deployments

* **Google Cloud Run (Serverless GCP Deployment)**: [https://macro-pulse-652787573242.us-central1.run.app](https://macro-pulse-652787573242.us-central1.run.app)
* **Streamlit Community Cloud**: [https://hf-macro-dashboard.streamlit.app/](https://hf-macro-dashboard.streamlit.app/)
* **GitHub Repository**: [https://github.com/sechan9999/hf-macro-dashboard](https://github.com/sechan9999/hf-macro-dashboard)


---

## 🎯 14 Analytical Modules

The platform is structured into 14 dedicated analytical tabs, each equipped with its own **"How to Read · Math · HF Terms"** institutional methodology expander:

### 1. 📈 Performance & Benchmark Tear Sheet
* **Cumulative Return (Base = 100)**: Strategy and S&P 500 total return tracked against the SPY ETF benchmark.
* **Rolling Drawdown & Underwater Curve**: Visualizes peak-to-trough decline severity and recovery durations.
* **Monthly Return Distributions**: Log-return histograms displaying skewness and tail-risk behavior.
* **Full Hedge Fund Tear Sheet**: Sharpe Ratio, Sortino Ratio (downside risk only), Calmar Ratio, Maximum Drawdown (MDD), Win Rate, Profit Factor, Alpha, and Beta.

### 2. 🌍 Macro & Rates
* **10Y Treasury Yield (^TNX)**: The global discount rate driving equity duration and valuation multiples.
* **Credit Spreads (BAA - AAA)**: Pulls real corporate credit spreads from FRED (or falls back to a deterministic 10Y proxy).
* **Yield-Curve Slope (10Y - 2Y)**: Classic recession barometer tracking inversion and un-inversion phases.
* **Realized Volatility Structure**: 3-Month vs. 12-Month realized volatility divergence indicating market regime stress.

### 3. 🔍 Macro Regime Classification
* Rule-based quantitative regime engine classifying broader market conditions into **Risk-On 🟢**, **Neutral 🟡**, or **Risk-Off 🔴**.
* Driven by standardized z-scores of credit spreads and volatility dynamics.
* Provides regime-conditional return distributions, Sharpe ratios, and historical win rates.

### 4. 🤖 Expected Returns (Ridge Regression)
* Expanding-window Ridge Regression trained live on historical macro features (interest rates, credit spreads, volatility, and momentum).
* Forecasts 12-month forward equity returns with **±1σ confidence intervals** and realized historical overlay to assess predictive accuracy.

### 5. 📊 High-Throughput Stock Screener
* Multi-threaded screener utilizing Python's `ThreadPoolExecutor` for parallel data acquisition (~70% faster than sequential queries).
* Computes real-time YTD performance, RSI(14), 200-day SMA trend alignment, and exportable CSV tables.

### 6. 📉 Technical Analysis & Weekly Buy Zone Scanner
* **Interactive Candlestick Workbench**: Dual moving averages (SMA 20/50/200), Bollinger Bands, VWAP, RSI, MACD, OBV, and Rate-of-Change.
* **Weekly Buy Zone Scanner**: Multi-ticker institutional panel resampling daily data to weekly Friday closes across large caps (e.g., NVDA, MSFT, TSM, ASML, AMZN, GOOGL, AVGO, LLY, V, COST). Evaluates 20/50-week SMAs, 14-week RSI, and weekly MACD to classify tickers into *Strong Buy*, *Pullback*, *Trend Continuation*, or *Avoid (Extended)*.

### 7. 🎲 Risk Simulation (Monte Carlo Engine)
* Simulates 1,000 to 10,000 forward market paths using geometric Brownian motion and empirical bootstrap modes.
* Interactive probability fan charts with parametric and empirical Value-at-Risk (**VaR 95% / 99%**) and Conditional Value-at-Risk (**CVaR / Expected Shortfall**).

### 8. ✨ AI Macro Analyst
* Ingests real-time dashboard data (macro regime scores, Treasury yields, credit spreads, drawdown metrics, volatility, momentum).
* Powered by **Google Gemini 1.5 / 3 Flash** via the unified Google GenAI SDK.
* Provides institutional briefings across 5 modes:
  1. *Full Macro Briefing*
  2. *Regime Deep-Dive*
  3. *Risk Assessment & Tail-Risk Review*
  4. *Tactical Investment Outlook*
  5. *Custom Financial Analyst Q&A*

### 9. 🔥 NVDA Danger Zone & Micro Footprint
* Specialized single-name market structure risk monitor for NVIDIA (NVDA):
  * **Composite Danger Index (0 to 1)**: Blends RSI extension, ATR volatility, relative volume surges, VIX levels, and 50-day SMA distance.
  * **Micro Footprint Block Trade Detection**: Highlights institutional volume spikes (>2× 20-day average).
  * **Synthetic Order-Flow Imbalance Proxy**: Evaluates buying dominance vs. aggressive supply.
  * **Peer Group Benchmark**: Normalized performance overlay against semiconductor peers (SOXX, AMD, TSM, AVGO, MU).

### 10. 📊 Walk-Forward Strategy Backtest
* No-lookahead, walk-forward backtest of an SPY/Cash/Short allocation engine.
* Configurable with three independent quantitative gates:
  1. *Macro Regime Filter* (Hold long only when regime is not Risk-Off)
  2. *12-1 Cross-Sectional / Time-Series Momentum* (Jegadeesh & Titman)
  3. *Faber 10-Month Moving Average Rule*
* Implements a strict $T+1$ execution lag (`.shift(1)`), linear transaction costs (default 5 bps turnover slippage), equity curves, and side-by-side strategy vs. buy-and-hold metrics.

### 11. 🎯 Quant Signals & Volatility Breakouts
* **Volatility-Aware Multi-Asset Screener**:
  * **14-day ATR%**: Normalizes price volatility as a percentage of current price.
  * **20-day Annualized Realized Volatility**: Historical standard deviation scaled to 252 trading days.
  * **Bollinger Bandwidth Percentile (BBW %)**: Identifies volatility squeeze regimes (BBW in the lower 20th percentile over 252 days).
  * **Volatility Breakout Detector**: Alerts when price pierces upper/lower Bollinger Bands following an expansion from a squeeze.
  * **Additive Scoring Engine (-100 to +100)**: Translates trend, momentum, RSI, and volatility squeeze into actionable signals (*Strong Long*, *Long*, *Neutral*, *Short*, *Strong Short*).

### 12. 🎙️ Alexa+ Institutional Copilot & MCP Terminal
* **Model Context Protocol (MCP)**: Native **Streamable HTTP** (2025-11-25) & SSE transport server.
* **Voice & Multimodal Simulation**: Interactive browser voice input, audio speech synthesis (TTS), and Echo Show / Fire TV multimodal visual cards.
* **8 Quantitative MCP Tools**: Real-time programmatic access to Macro Regimes, 10Y/2Y Yield Curve Slopes, FRED Credit Spreads, 10,000-path Monte Carlo VaR, NVDA Danger Zone index, Volatility Squeezes, Ridge Expected Returns, and FOMC Cross-Asset Shocks.
* **Developer Telemetry**: Live JSON-RPC request/response payload inspector with sub-500ms execution latency.

### 13. 🏦 Multi-Asset Portfolio Sync & FOMC Shock Desk
* **Read-Only Brokerage Connectors**: Synchronize multi-asset holdings via Alpaca Read-Only API, Interactive Brokers (IBKR) Flex Query, Plaid schema, or institutional presets (Macro Balanced, Tech Growth, Risk Parity, All-Weather).
* **Cross-Asset VaR & Expected Shortfall**: Full variance-covariance matrix across Equities, Fixed Income/Duration (TLT, IEF, SHY), Commodities (GLD, USO), and Crypto (BTC-USD, ETH-USD) computing 30-day VaR (95%/99%) and marginal component risk.
* **FOMC Rate Decision Shock Matrix**: Instantaneous scenario stress testing:
  * *Hawkish Surprise (+50 bps Hike)*: Duration hit (-4.8% TLT), equity multiple compression (-3.5%), USD surge.
  * *Dovish Pivot (-50 bps Cut)*: Bond rally (+5.2% TLT), tech equity expansion (+4.1%), Gold rally (+3.4%).
  * *Stagflationary Inversion*: Curve inverts -50 bps, credit spreads widen +120 bps, stocks fall -6.8%, Gold gains +5.2%.
  * *March 2020 Liquidity Crunch*: Correlated liquidation test across all risk assets.
* **Fire TV Big Screen View**: High-contrast, large typography 10-foot UI designed for viewing on living room and trading desk wall displays.
* **Personalized Ambient Audio Chimes**: Procedural harmonic audio cues played on Fire TV / browser when Bollinger Squeeze breakouts, credit spread divergences, or FOMC shocks occur.

### 14. 📑 Equity Report (ticker in, research pack out)
* **Four downloads per ticker**: Excel DCF model with live formulas (edit the yellow assumption cells and everything recalculates), Word research note, offline HTML DCF dashboard, or all of it as a zip. Files are built in memory, nothing is written to disk.
* **Data**: prices from Yahoo Finance, fundamentals from **SEC EDGAR XBRL 10-K** filings (restatements win, multi-class share counts summed, no look-ahead in the P/E band). ETFs and foreign 20-F/40-F filers (e.g. TSM, ASML) are refused with a clear message instead of a currency-mismatched DCF.
* **Valuation**: FCFF-proxy DCF (CFO − CapEx), CAPM with live 10Y yield and Blume-adjusted beta, WACC × g sensitivity, bear/base/bull scenarios, P/E and P/B bands, football-field chart.
* **Macro-regime overlay**: the dashboard's Risk-On / Neutral / Risk-Off call sets the scenario weights (20/50/30 · 25/50/25 · 40/45/15) and shifts the ERP (−0.25pp / 0 / +0.50pp). The report also shows the neutral-weight value for comparison.
* **Technicals**: Monthly (3y) / Weekly (12m) / Daily (6m) candles with MACD, slow stochastic (KD), RSI, Bollinger, ADX, support/resistance and a 17-pattern candlestick + indicator screen (`bullish_count`, `bearish_count`, `bull_score`, `bear_score`, `overall`).
* **Cross-tab links**: target touch probabilities use the drift implied by the Expected Returns tab's Ridge S&P 500 forecast (CAPM; the driftless value is shown alongside), the tab-13 FOMC shock scenarios are overlaid on the WACC × g sensitivity grid (H / D / S / L badges + table, also as live formulas in Excel), and every Buy Zone scanner row links to its report.
* **Narrative**: rule-based rating first, then optional Gemini / Bedrock text that may only quote numbers from the JSON context.
* **Alexa+**: "Alexa, give me an equity report on Apple" → spoken summary + Fire TV APL card (scenario footer); new MCP tool `get_equity_report`; the Alexa+ tab shows the same four downloads. Deep link: `?ticker=AAPL`.
* **Secrets**: set `SEC_USER_AGENT="Your Name you@example.com"` (SEC fair-access policy). Optional: `EQUITY_REPORT_WARM` (tickers prefetched by the MCP server, default `AAPL,MSFT,NVDA,AMZN,GOOGL`), `MACROPULSE_APP_URL` (base URL for `?ticker=` report links, default `https://hf-macro-dashboard.streamlit.app/`).

---

## 📺 Amazon Fire TV Companion & Appstore Release

MacroPulse includes an official companion package for **Amazon Fire TV** (`firetv/`):
* **10-Foot UI Experience**: Overscan-safe layout with high-visibility glowing focus rings (`firetv/remote_nav.js`) fully navigable with standard Fire TV Voice Remote D-pad arrow keys (Up, Down, Left, Right, Select, Back).
* **Hands-Free Alexa Voice Directives**:
  * *"Alexa, ask MacroPulse for today's market regime"*
  * *"Alexa, show the NVDA Danger Zone on the TV"*
  * *"Alexa, run an FOMC rate shock test on my portfolio"*
* **Alexa Presentation Language (APL 2024.1)**: Full-screen responsive templates (`skills/macropulse-alexa/`) dynamically rendered on Fire TV Stick 4K, Fire TV Cube, and Echo Show devices.
* **Appstore Metadata**: Production submission guide and asset specifications (`firetv/appstore_listing.md`).

---

## ☁️ 1-Click AWS Marketplace & CloudFormation Release

Deploy MacroPulse to a dedicated enterprise AWS environment in under 5 minutes:

[![Launch Stack](https://s3.amazonaws.com/cloudformation-examples/cloudformation-launch-stack.png)](aws/README_AWS_MARKETPLACE.md)

* **Architecture**: AWS ECS Fargate serverless container, Application Load Balancer (ALB), multi-AZ VPC, and AWS Secrets Manager for FRED, Gemini, and Brokerage credentials.
* **Template**: [`aws/cloudformation.yaml`](aws/cloudformation.yaml)
* **Automated Deploy Script**: Run `./aws/deploy_cfn.sh macropulse-prod us-east-1`
* **Documentation**: See [`aws/README_AWS_MARKETPLACE.md`](aws/README_AWS_MARKETPLACE.md) for full configuration options.

---

## 🤖 Pre-Market Automation & Scheduled Scans

Macro Pulse features headless alerting that runs outside the UI:

### 1. Headless CLI Alert Runner
Run scans from the terminal or pipe results directly into automated pipelines:
```bash
python scripts/daily_signal_alert.py \
  --watchlist "NVDA,AAPL,MSFT,AMZN,GOOGL,META,TSLA,SPY,QQQ,IWM" \
  --lookback 1y \
  --min-score 25 \
  --slack-webhook "https://hooks.slack.com/services/..."
```

### 2. GitHub Actions Scheduled Workflow
The repository includes [`.github/workflows/daily-quant-signal.yml`](.github/workflows/daily-quant-signal.yml) configured to execute automatically every trading day at **13:30 UTC (9:30 AM ET pre-market)**:
* Fetches the latest market closes.
* Runs `scripts/daily_signal_alert.py` across core index ETFs and mega-cap tech.
* Posts formatted markdown summaries directly to GitHub Workflow Summaries and optionally sends a Slack webhook alert.

---

## 🎙️ Alexa+ Model Context Protocol (MCP) & Streamable HTTP

MacroPulse features an open-standard **Model Context Protocol (MCP)** server conforming to the **2025-11-25 Streamable HTTP (SSE)** specification, built specifically for the **Alexa+ Agent Skills** ecosystem:

### 1. Launch the MCP Server
Run the high-performance server supporting both Streamable HTTP and SSE transports:
```bash
python run_alexa_mcp.py --port 8000
```

### 2. Available Endpoints
* **Streamable HTTP Endpoint**: `http://localhost:8000/mcp`
* **SSE Endpoint**: `http://localhost:8000/sse`
* **Messages Endpoint**: `http://localhost:8000/messages`
* **Alexa Skills Kit Webhook**: `http://localhost:8000/alexa/skill`
* **Health Check**: `http://localhost:8000/health`
* **Alexa Query Simulator API**: `http://localhost:8000/alexa/query`

### 3. Verify Quant MCP Tools Locally
```bash
python run_alexa_mcp.py --test-tools
python -m unittest tests/test_alexa_mcp.py
```

---

## 📐 Mathematical Foundations & Methodology

Every tab in the application includes an embedded **"📐 How to read this tab · Math · HF Terms"** expander that details:
* Chart interpretation guidelines.
* Formal mathematical formulas rendered via LaTeX/MathJax (e.g., Continuous Log Return compounding, Sharpe/Sortino ratios, GMM log-likelihood, Ridge objective function with L2 shrinkage, Bollinger Bandwidth, ATR).
* Institutional hedge fund terminology (e.g., *Tear Sheet*, *High-Water Mark*, *Lookahead Bias*, *Gamma Squeeze*, *Order-Flow Imbalance*, *Capacity*, *PBO*).

The complete documentation is also compiled in [`docs/methodology.md`](docs/methodology.md).

---

## 🛠 Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Cloud Data Lake** | [AWS S3](https://aws.amazon.com/s3/) (Parquet quant marts via boto3 + pyarrow) |
| **Frontend UI** | [Streamlit](https://streamlit.io/) (Dark financial theme, custom CSS layout) |
| **Data Visualization** | [Plotly](https://plotly.com/python/) (Interactive charts, dark theme presets) |
| **Market Data** | [yfinance](https://pypi.org/project/yfinance/) (Live real-time prices & fundamentals) |
| **Macro Data** | [FRED API](https://fred.stlouisfed.org/) via `pandas-datareader` / direct REST |
| **Machine Learning** | [scikit-learn](https://scikit-learn.org/) (Expanding-window Ridge Regression, GMM) |
| **Scientific Computing** | [NumPy](https://numpy.org/) & [Pandas](https://pandas.pydata.org/) |
| **AI Analyst Copilot** | [Google Gemini 1.5 / 3 Flash](https://aistudio.google.com/app/apikey) via Unified `google-genai` SDK |
| **Concurrency** | Python `concurrent.futures.ThreadPoolExecutor` |
| **Automation & CI/CD** | GitHub Actions (`cron: '30 13 * * 1-5'`), Slack Webhooks, Cloud Build |
| **Containerization** | Docker multi-stage build (`python:3.11-slim`) |
| **Cloud Hosting** | Google Cloud Run (`agentichackathon-506620`) & Streamlit Cloud |

---

## ☁️ AWS S3 Quant Data Lake

Macro Pulse decouples live presentation from market-data ingestion using an **AWS S3 data lake**. The access pattern is write-whole-table / read-whole-table (no ad-hoc SQL), so each mart is a single Parquet object rather than a warehouse table:

* **Bucket**: your own (set `MACROPULSE_S3_BUCKET`), prefix `macropulse/` (`MACROPULSE_S3_PREFIX`)
* **Marts** (`src/aws_datalake.py`, written via `boto3` + `pyarrow`):
  * `macropulse/macro_factors.parquet`: S&P 500, VIX, Treasury yields, credit spreads, and regime z-scores, keyed by `date`.
  * `macropulse/quant_signals.parquet`: ATR%, 20d realized vol, Bollinger bandwidth, squeeze detection, and composite scores across the watchlist.

### Enabling the data lake
It is optional: with nothing configured the app runs on live yfinance/local and the sidebar shows **AWS S3: Standby (local/fallback)**. To turn it on, set the bucket and give the app AWS credentials (env vars, a shared profile, or an instance/task role):
```bash
export MACROPULSE_S3_BUCKET=your-bucket
export AWS_REGION=us-east-1
# AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, or an attached role
```
On Streamlit Cloud, put the same values in **App → Settings → Secrets**. Once connected, the marts are written on each live refresh, and **"☁️ Load from S3 Mart (<0.2s)"** reads pre-computed quant signals back instantly.


---

## 📋 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/sechan9999/hf-macro-dashboard.git
cd hf-macro-dashboard
```

### 2. Set up virtual environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys (Optional)
Core market data and technical signals function out of the box without any API keys. To enable optional institutional features, configure your keys:

Create `.streamlit/secrets.toml` or set environment variables:
```toml
# Google Gemini API key (enables the AI Analyst tab):
GEMINI_API_KEY = "your_google_gemini_api_key_here"

# FRED API key (enables live BAA-AAA credit spread & yield curve from St. Louis Fed):
FRED_API_KEY = "your_fred_api_key_here"

# Optional Slack webhook URL for automated daily signals:
SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/..."
```

### 5. Run the application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Google Cloud Run Deployment

The application is containerized and ready for continuous deployment on Google Cloud Run:

```bash
# Deploy directly from source via Google Cloud CLI:
gcloud run deploy macro-pulse \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi
```

Alternatively, use the included PowerShell script for Windows:
```powershell
.\deploy_gcloud.ps1
```

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

*© 2026 HF Research & Antigravity AI — Built for institutional research and quantitative finance education.*
