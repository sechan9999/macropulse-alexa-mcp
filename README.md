# ⚡ Macro Pulse: Macro Regime Dashboard, Alexa+ MCP Server & Fire TV App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hf-macro-dashboard.streamlit.app/)
[![Google Cloud Run](https://img.shields.io/badge/Google%20Cloud%20Run-Live-4285F4?logo=google-cloud&logoColor=white)](https://macro-pulse-652787573242.us-central1.run.app)
[![AWS S3](https://img.shields.io/badge/AWS%20S3-Data%20Lake-569A31?logo=amazons3&logoColor=white)](https://aws.amazon.com/s3/)
[![Tests](https://github.com/sechan9999/macropulse-alexa-mcp/actions/workflows/tests.yml/badge.svg)](https://github.com/sechan9999/macropulse-alexa-mcp/actions/workflows/tests.yml)
[![Daily Quant Signal](https://github.com/sechan9999/macropulse-alexa-mcp/actions/workflows/daily-quant-signal.yml/badge.svg)](https://github.com/sechan9999/macropulse-alexa-mcp/actions/workflows/daily-quant-signal.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Which repository is this?** The source of truth is **[sechan9999/macropulse-alexa-mcp](https://github.com/sechan9999/macropulse-alexa-mcp)**. **[sechan9999/hf-macro-dashboard](https://github.com/sechan9999/hf-macro-dashboard)** is a read-only mirror that Streamlit Community Cloud deploys from; it is overwritten on every merge, so open issues and pull requests in `macropulse-alexa-mcp`.

**Macro Pulse** is a macro research and education project. One Python quant core classifies the market regime (Risk-On / Neutral / Risk-Off from credit spreads and volatility), forecasts the S&P 500's 12-month return, stress-tests portfolios and builds SEC-filing-based equity reports. It serves three front ends:

* a **14-tab Streamlit dashboard** ([hf-macro-dashboard.streamlit.app](https://hf-macro-dashboard.streamlit.app/));
* a self-hosted **MCP server for Alexa+** on Amazon ECS (9 tools over Streamable HTTP);
* a **Fire TV app** (Expo + react-native-tvos) that reads the same AWS endpoint.

**Point-in-time by design.** The regime score and the expected-return model only use data that was available at each date: expanding z-scores, and a 12-month training embargo before each forecast. When FRED or Yahoo data is unavailable, the app and the voice tools say so instead of substituting made-up numbers. CI runs 158 offline tests, including a noise leakage test that fails if the forecast model shows any skill on pure-noise returns.

Built with **Streamlit**, **Plotly**, **yfinance**, **FRED**, **SEC EDGAR**, **scikit-learn**, **AWS (ECS, ECR, S3, Bedrock)** and **Google Gemini**.

---

## 🔗 Live Deployments

* **Google Cloud Run (Serverless GCP Deployment)**: [https://macro-pulse-652787573242.us-central1.run.app](https://macro-pulse-652787573242.us-central1.run.app)
* **Streamlit Community Cloud**: [https://hf-macro-dashboard.streamlit.app/](https://hf-macro-dashboard.streamlit.app/)
* **GitHub Repository (source)**: [https://github.com/sechan9999/macropulse-alexa-mcp](https://github.com/sechan9999/macropulse-alexa-mcp)
* **Deploy mirror for Streamlit Cloud**: [https://github.com/sechan9999/hf-macro-dashboard](https://github.com/sechan9999/hf-macro-dashboard) (read-only; synced from the source repo)

> **Where to make changes:** open PRs in [macropulse-alexa-mcp](https://github.com/sechan9999/macropulse-alexa-mcp). Every push to its `main` is mirrored into `hf-macro-dashboard` (the repo Streamlit Cloud deploys) by `.github/workflows/sync-hf-dashboard.yml` — one-way, `signals/` and `.github/workflows/` excluded (the push token has no `workflow` scope), so code edited directly in `hf-macro-dashboard` is overwritten on the next sync. Needs the Actions secret `HF_DASHBOARD_PUSH_TOKEN` (fine-grained PAT, Contents: read & write on `hf-macro-dashboard`).


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
* **Credit Spreads (BAA - AAA)**: Moody's corporate credit spread from FRED (API key, or FRED's keyless `fredgraph.csv` endpoint). If FRED is unreachable the series is shown as unavailable; no proxy is substituted.
* **Yield-Curve Slope (10Y - 2Y)**: Classic recession barometer tracking inversion and un-inversion phases.
* **Realized Volatility Structure**: 3-Month vs. 12-Month realized volatility divergence indicating market regime stress.

### 3. 🔍 Macro Regime Classification
* Rule-based regime engine classifying market conditions into **Risk-On 🟢**, **Neutral 🟡** or **Risk-Off 🔴** (thresholds ±0.5 on the stress score).
* Stress score = z(credit spread) + z(12-month realised vol), using **expanding** means and standard deviations (data up to each month only, at least 36 months), so a past month's label never depends on later data. Months without enough history or without FRED data are **Unavailable ⚪**.
* The Alexa+ MCP server and the Fire TV API use the same model (`src/macro_model.py`), so voice, TV and dashboard agree.
* Provides regime-conditional return distributions, Sharpe ratios and win rates.

### 4. 🤖 Expected Returns (Ridge Regression)
* Expanding-window Ridge regression on macro features (10Y yield, credit spread, curve slope, volatility, momentum, regime score), refit every month.
* Each month's model is trained only on months whose 12-month outcome was already known then (12-month embargo), and the last point is **today's** forecast.
* The ±1σ band is the spread of the model's **realised out-of-sample errors**, and the realised line is the actual return over the 12 months each forecast was about.

### 5. 📊 High-Throughput Stock Screener
* Multi-threaded screener utilizing Python's `ThreadPoolExecutor` for parallel data acquisition (~70% faster than sequential queries).
* Computes real-time YTD performance, RSI(14), 200-day SMA trend alignment, and exportable CSV tables.

### 6. 📉 Technical Analysis & Weekly Buy Zone Scanner
* **Interactive Candlestick Workbench**: Dual moving averages (SMA 20/50/200), Bollinger Bands, VWAP, RSI, MACD, OBV, and Rate-of-Change.
* **Weekly Buy Zone Scanner**: Multi-ticker institutional panel resampling daily data to weekly Friday closes across large caps (e.g., NVDA, MSFT, TSM, ASML, AMZN, GOOGL, AVGO, LLY, V, COST). Evaluates 20/50-week SMAs, 14-week RSI, and weekly MACD to classify tickers into *Strong Buy*, *Pullback*, *Trend Continuation*, or *Avoid (Extended)*.

### 7. 🎲 Risk Simulation (Monte Carlo Engine)
* Simulates 1,000 to 10,000 forward paths with Gaussian (geometric Brownian motion) returns.
* Probability fan charts with Value-at-Risk (**VaR 95% / 99%**) and Conditional Value-at-Risk (**CVaR / Expected Shortfall**) read from the simulated distribution.

### 8. ✨ AI Macro Analyst
* Ingests real-time dashboard data (macro regime scores, Treasury yields, credit spreads, drawdown metrics, volatility, momentum).
* Powered by **Google Gemini** (`gemini-3.6-flash`, google-genai SDK) or **Amazon Bedrock** (default `amazon.nova-pro-v1:0`). The model may only quote numbers from the dashboard data it is given, and analysis is paused when live macro data is incomplete.
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
  1. *Macro Regime Filter* (hold long only when the point-in-time regime is not Risk-Off)
  2. *12-1 Cross-Sectional / Time-Series Momentum* (Jegadeesh & Titman)
  3. *Faber 10-Month Moving Average Rule*
* Implements a strict $T+1$ execution lag (`.shift(1)`), linear transaction costs (default 5 bps turnover slippage), equity curves, and side-by-side strategy vs. buy-and-hold metrics. The cash leg currently earns 0%.

### 11. 🎯 Quant Signals & Volatility Breakouts
* **Volatility-Aware Multi-Asset Screener**:
  * **14-day ATR%**: Normalizes price volatility as a percentage of current price.
  * **20-day Annualized Realized Volatility**: Historical standard deviation scaled to 252 trading days.
  * **Bollinger Bandwidth Percentile (BBW %)**: Identifies volatility squeeze regimes (BBW in the lower 20th percentile over 252 days).
  * **Volatility Breakout Detector**: Alerts when price pierces upper/lower Bollinger Bands following an expansion from a squeeze.
  * **Additive Scoring Engine (-100 to +100)**: Translates trend, momentum, RSI, and volatility squeeze into actionable signals (*Strong Long*, *Long*, *Neutral*, *Short*, *Strong Short*).

### 12. 🎙️ Alexa+ Institutional Copilot & MCP Terminal
* **Model Context Protocol (MCP)**: Streamable HTTP (spec 2025-11-25) and legacy SSE transports.
* **Voice & Multimodal Simulation**: Interactive browser voice input, audio speech synthesis (TTS), and Echo Show / Fire TV multimodal visual cards.
* **9 MCP tools**: morning brief, equity report, macro regime, rates and spreads (10Y yield, 10Y-3M slope, FRED Baa-Aaa spread), Monte Carlo VaR/CVaR, FOMC shock test, NVDA danger zone, volatility-squeeze scan, and a static 12-month S&P 500 reference estimate (labelled as static, not a live forecast).
* **Developer Telemetry**: Live JSON-RPC request/response payload inspector with sub-500ms execution latency.

### 13. 🏦 Multi-Asset Portfolio Sync & FOMC Shock Desk
* **Read-Only Brokerage Connectors**: Synchronize multi-asset holdings via Alpaca Read-Only API, Interactive Brokers (IBKR) Flex Query, Plaid schema, or institutional presets (Macro Balanced, Tech Growth, Risk Parity, All-Weather).
* **Cross-Asset VaR & Expected Shortfall**: Full variance-covariance matrix across Equities, Fixed Income/Duration (TLT, IEF, SHY), Commodities (GLD, USO), and Crypto (BTC-USD, ETH-USD) computing 30-day VaR (95%/99%) and marginal component risk.
* **FOMC Rate Decision Shock Matrix**: instantaneous scenario stress test. The shock sizes below are hand-set scenario assumptions, not estimates from past FOMC days:
  * *Hawkish Surprise (+50 bps Hike)*: Duration hit (-4.8% TLT), equity multiple compression (-3.5%), USD surge.
  * *Dovish Pivot (-50 bps Cut)*: Bond rally (+5.2% TLT), tech equity expansion (+4.1%), Gold rally (+3.4%).
  * *Stagflationary Inversion*: Curve inverts -50 bps, credit spreads widen +120 bps, stocks fall -6.8%, Gold gains +5.2%.
  * *March 2020 Liquidity Crunch*: Correlated liquidation test across all risk assets.
* **Big-screen view**: high-contrast, large-type layout for wall displays.
* **Audio chimes**: procedural audio cues in the browser when Bollinger squeeze breakouts, credit-spread divergences or FOMC shocks occur.

### 14. 📑 Equity Report (ticker in, research pack out)
* **Four downloads per ticker**: Excel DCF model with live formulas (edit the yellow assumption cells and everything recalculates), Word research note, offline HTML DCF dashboard, or all of it as a zip. Files are built in memory, nothing is written to disk.
* **Data**: prices from Yahoo Finance, fundamentals from **SEC EDGAR XBRL 10-K** filings (restatements win, multi-class share counts summed, no look-ahead in the P/E band). ETFs and foreign 20-F/40-F filers (e.g. TSM, ASML) are refused with a clear message instead of a currency-mismatched DCF.
* **Valuation**: FCFF-proxy DCF (CFO − CapEx), CAPM with live 10Y yield and Blume-adjusted beta, WACC × g sensitivity, bear/base/bull scenarios, P/E and P/B bands, football-field chart.
* **Macro-regime overlay**: the dashboard's Risk-On / Neutral / Risk-Off call sets the scenario weights (20/50/30 · 25/50/25 · 40/45/15) and shifts the ERP (−0.25pp / 0 / +0.50pp). The report also shows the neutral-weight value for comparison.
* **Technicals**: Monthly (3y) / Weekly (12m) / Daily (6m) candles with MACD, slow stochastic (KD), RSI, Bollinger, ADX, support/resistance and a 17-pattern candlestick + indicator screen (`bullish_count`, `bearish_count`, `bull_score`, `bear_score`, `overall`).
* **Cross-tab links**: target touch probabilities use the drift implied by the Expected Returns tab's Ridge S&P 500 forecast (CAPM; the driftless value is shown alongside), the tab-13 FOMC shock scenarios are overlaid on the WACC × g sensitivity grid (H / D / S / L badges + table, also as live formulas in Excel), and every Buy Zone scanner row links to its report.
* **Narrative**: rule-based rating first, then optional Gemini / Bedrock text that may only quote numbers from the JSON context.
* **Alexa+**: "Alexa, give me an equity report on Apple" → spoken summary + Fire TV APL card (scenario footer); new MCP tool `get_equity_report`; the Alexa+ tab shows the same four downloads. Deep link: `?ticker=AAPL` opens the app on the Equity Report tab with the report already generated.
* **Secrets**: set `SEC_USER_AGENT="Your Name you@example.com"` (SEC fair-access policy). Optional: `EQUITY_REPORT_WARM` (tickers prefetched by the MCP server, default `AAPL,MSFT,NVDA,AMZN,GOOGL`), `MACROPULSE_APP_URL` (base URL for `?ticker=` report links, default `https://hf-macro-dashboard.streamlit.app/`).

---

## 📺 Amazon Fire TV App

The Fire TV companion app lives in [`firetv-app/`](firetv-app/) (Expo + react-native-tvos, built as an APK with EAS Build) and has been submitted to the Amazon Appstore for Fire TV.
* **What it shows**: macro regime card, S&P 500, 10-year yield, curve slope, 12-month realised vol, the NVDA danger index and watchlist signals, refreshed every 5 minutes. No sign-in.
* **Data**: the same AWS endpoint as the MCP server (`/api/regime`, `/api/watchlist-signals`, `/api/nvda-danger`). When live data is incomplete the API returns 503 and the app shows an error instead of placeholder numbers.
* **10-foot UI**: D-pad navigation with a collapsible side menu and visible focus outlines.
* **Build and store assets**: see [`firetv-app/README.md`](firetv-app/README.md); listing images are generated by `firetv-app/make_assets.py`.

Voice and screens on Alexa devices come from the Alexa skill in [`skills/macropulse-alexa/`](skills/macropulse-alexa/) (APL 2024.1 cards on Fire TV and Echo Show), for example *"Alexa, ask MacroPulse for today's market regime"* or *"Alexa, show the NVDA Danger Zone on the TV"*. The earlier web prototype is in [`firetv/`](firetv/).

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
python scripts/daily_signal_alert.py --tickers "NVDA,AAPL,MSFT,SPY,QQQ" --period 1y --top 8
```

### 2. GitHub Actions Scheduled Workflow
[`.github/workflows/daily-quant-signal.yml`](.github/workflows/daily-quant-signal.yml) runs every weekday at **12:00 UTC** (about 8:00 ET, before the open):
* Runs `scripts/daily_signal_alert.py --period 1y --top 8` on the default universe.
* Publishes the result to the workflow summary and commits a dated snapshot to `signals/`, which the MCP server's morning brief reads.

---

## 🎙️ Alexa+ MCP Server (Streamable HTTP)

A self-hosted **Model Context Protocol** server (official Python MCP SDK, stateless Streamable HTTP, spec 2025-11-25) exposes 9 tools to Alexa+. The public deployment runs on **Amazon ECS Express Mode**:

* **MCP endpoint**: `https://ma-76bae7709d4340b3ab4733ba644f11e4.ecs.us-east-1.on.aws/mcp` (try it with `npx @modelcontextprotocol/inspector`, transport *Streamable HTTP*)
* The same service serves `/health`, `/privacy`, `/terms` and the Fire TV app's `/api/*`.
* Prices and FRED series are served from caches that refresh in the background, so tools answer well within Alexa+'s latency budget. When live data cannot be fetched, a tool returns an error instead of a guess.

### Run it locally
```bash
pip install -r requirements-mcp.txt
python run_alexa_mcp.py --port 8000
```
Local endpoints: `http://localhost:8000/mcp` (Streamable HTTP), `/sse` and `/messages/` (legacy SSE), `/alexa/skill` (Alexa Skills Kit webhook), `/alexa/query` (voice simulator), `/health`.

### Deploy the container
The MCP server has its own image; build it from **`Dockerfile.mcp`** (the default `Dockerfile` is the Streamlit dashboard):
```bash
docker build -f Dockerfile.mcp -t <account>.dkr.ecr.us-east-1.amazonaws.com/macropulse-alexa-mcp:<tag> .
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/macropulse-alexa-mcp:<tag>
```
Set `SEC_USER_AGENT="Your Name you@example.com"` on the service (SEC fair-access policy); `FRED_API_KEY` is optional.

---

## ✅ Tests & Data Integrity

```bash
python -m pytest -q --ignore=tests/test_alexa_mcp.py   # offline suite (what CI runs)
python -m pytest -q tests/test_alexa_mcp.py            # live-data smoke test of the MCP tools
```
* **Noise leakage test** (`tests/test_leakage.py`): returns are pure noise and features are persistent AR(1) noise, so nothing can be forecast. The model's out-of-sample correlation must stay below 0.10; the pre-fix model scores +0.20 to +0.32, and a second test checks that the check still catches it.
* **Point-in-time tests** (`tests/test_macro_model.py`, `tests/test_mcp_regime.py`): rewriting data after a date must not change the regime, forecast or error band computed at that date. The MCP regime must match the dashboard's model and must not depend on VIX.
* **No made-up numbers**: tests cover Yahoo and FRED outages for the MCP tools, the Fire TV API and the AI briefing.

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
| **Macro Data** | [FRED](https://fred.stlouisfed.org/) (BAA, AAA, T10Y2Y) via `fredapi` or the keyless `fredgraph.csv` endpoint; SEC EDGAR XBRL for fundamentals |
| **Machine Learning** | [scikit-learn](https://scikit-learn.org/) (Expanding-window Ridge Regression, GMM) |
| **Scientific Computing** | [NumPy](https://numpy.org/) & [Pandas](https://pandas.pydata.org/) |
| **AI Analyst Copilot** | [Google Gemini](https://aistudio.google.com/app/apikey) (`google-genai` SDK) or Amazon Bedrock (Converse API) |
| **Concurrency** | Python `concurrent.futures.ThreadPoolExecutor` |
| **Automation & CI/CD** | GitHub Actions: offline test suite on every PR, daily signal scan (`cron: '0 12 * * 1-5'`), one-way sync to the deploy mirror |
| **Containerization** | Docker multi-stage build (`python:3.11-slim`) |
| **Cloud Hosting** | Streamlit Community Cloud (dashboard), Amazon ECS Express Mode + ECR (MCP server), Google Cloud Run |
| **Voice & TV** | MCP Python SDK (FastMCP / Starlette), Alexa Skills Kit + APL, Expo + react-native-tvos (Fire TV) |

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
git clone https://github.com/sechan9999/macropulse-alexa-mcp.git
cd macropulse-alexa-mcp
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

# FRED API key (optional: without it the keyless fredgraph.csv endpoint is used):
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

*© 2026 HF Research & Antigravity AI — Built for macro research and quantitative finance education. Informational only; not investment advice.*
