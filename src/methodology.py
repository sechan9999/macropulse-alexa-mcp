"""Per-tab methodology — single source of truth.

Each entry has three sections:
  • chart  — how to read the charts on this tab
  • math   — formulas the tab actually uses (Streamlit renders $...$ via MathJax)
  • terms  — hedge-fund terminology a reader should recognise

The same content is mirrored into docs/methodology.md so it's readable
outside the app. Update both at the same time.
"""

from typing import Any

METHODOLOGY: dict[str, dict[str, str]] = {
    # ───────────────────────────────────────────────────────────────
    "performance": {
        "title": "📈 Performance — Methodology",
        "chart": """
* **Cumulative Return (Base = 100)** — growth path of a single dollar invested at the start. Compare the blue line (S&P 500 log returns) against the dashed yellow line (SPY actual ETF) — the gap is approximation error from the dataset, not alpha.
* **Rolling Drawdown** — peak-to-trough percentage decline. The deeper and longer the underwater curve, the more painful the strategy is to live through.
* **Monthly Return Distribution** — histogram of monthly log-returns. A right-skewed bell with a tight left tail is the dream; fat left tails are the silent killer.
* **Tear Sheet** — the row of risk-adjusted statistics every quant deck eventually settles on.
""",
        "math": r"""
**Cumulative log return**
$$R_T = \sum_{t=1}^{T} \log(1 + r_t), \qquad V_T = V_0 \cdot e^{R_T}$$

**Annualised return / volatility (monthly inputs)**
$$\mu_{\text{ann}} = 12\,\bar r, \qquad \sigma_{\text{ann}} = \sigma_r \sqrt{12}$$

**Sharpe ratio** (excess return per unit of total risk)
$$\text{SR} = \frac{\mu_{\text{ann}} - r_f}{\sigma_{\text{ann}}}$$

**Sortino ratio** (penalises downside deviation only)
$$\text{Sortino} = \frac{\mu_{\text{ann}} - r_f}{\sigma_d \sqrt{12}}, \quad \sigma_d = \mathrm{std}(r_t \mid r_t < 0)$$

**Maximum drawdown**
$$\text{MDD} = \min_{t}\!\left(\frac{V_t}{\max_{s \le t} V_s} - 1\right)$$

**Calmar ratio**
$$\text{Calmar} = \frac{\mu_{\text{ann}}}{|\text{MDD}|}$$

**Profit factor**
$$\text{PF} = \frac{\sum_t r_t \, \mathbf{1}_{r_t > 0}}{\bigl|\sum_t r_t \, \mathbf{1}_{r_t < 0}\bigr|}$$

**Beta / Alpha vs SPY** (CAPM)
$$\beta = \frac{\mathrm{Cov}(r_p, r_b)}{\mathrm{Var}(r_b)}, \qquad \alpha = \mu_p - \beta\,\mu_b$$
""",
        "terms": """
* **Tear sheet** — the canonical one-pager of strategy stats; what every PM hands to the CIO.
* **Underwater curve** — drawdown plotted as a continuous line; visualises *time spent in pain*.
* **High-water mark (HWM)** — the previous portfolio peak; many fee structures gate performance fees on new HWMs.
* **Information ratio (IR)** — Sharpe of *active* return vs benchmark; a long-only manager's headline number.
""",
    },
    # ───────────────────────────────────────────────────────────────
    "macro_rates": {
        "title": "🌍 Macro & Rates — Methodology",
        "chart": """
* **10Y Treasury Yield** — the global discount rate. Rising yields mechanically compress duration-sensitive equities (long-duration tech, REITs, utilities) and lift the bar for any cash flow far in the future.
* **Credit Spread** — extra yield corporates pay over Treasuries. Spike = stress, compression = complacency. This dashboard pulls real BAA-AAA from FRED when available, otherwise a 10Y-derived proxy (footer indicates source).
* **Yield-Curve Slope (10Y - 2Y)** — recession barometer. Inversion (negative) historically precedes recessions by 6-18 months.
* **Realized Volatility (3M vs 12M)** — when 3M crosses above 12M, regime is unstable.
""",
        "math": r"""
**Realised volatility (annualised, n-month rolling)**
$$\sigma_{rv,n} = \sqrt{\frac{12}{n}\sum_{i=t-n+1}^{t} r_i^{2}}$$

**Yield-curve slope** (treasuries)
$$\text{Slope} = y_{10y} - y_{2y}$$

**Credit spread** (corporate)
$$\text{CS} = y_{\text{BAA}} - y_{\text{AAA}} \quad\text{or}\quad y_{\text{BAA}} - y_{10y}$$

**Real yield** (nominal minus expected inflation)
$$r_{\text{real}} = y_{\text{nom}} - \pi^{e}$$

**Macro Stress Score** (this dashboard)
$$\text{MSS}_t = z(\text{CS})_t + z(\sigma_{rv,12})_t, \qquad z(x)_t = \frac{x_t - \bar x}{s_x}$$

**Bond convexity** (sensitivity-of-sensitivity, why long bonds whip)
$$C = \frac{1}{P}\frac{\partial^{2}P}{\partial y^{2}}, \qquad \Delta P \approx -D\,\Delta y + \tfrac{1}{2} C\,(\Delta y)^{2}$$
""",
        "terms": """
* **Risk-On / Risk-Off (RoRo)** — single dial summarising whether capital is flowing into equities & EM (on) or USTs & USD (off).
* **Inversion** — long minus short rate goes negative; classic late-cycle signal.
* **Carry trade** — borrow at low-yielding leg, lend at high-yielding leg, harvest the slope. Famous for 'picking up nickels in front of a steamroller'.
* **Term premium** — extra yield investors demand for holding duration vs rolling shorter bills.
* **OAS (Option-Adjusted Spread)** — credit spread net of embedded options; bond-trader standard.
* **Duration** — weighted-average time to cash-flow receipt; first-order rate sensitivity.
""",
    },
    # ───────────────────────────────────────────────────────────────
    "regime": {
        "title": "🔍 Regime — Methodology",
        "chart": """
* **Color-coded S&P 500** — each month's price is dotted with its regime colour, so you can see clusters of green (good) and red (bad) periods without reading the score.
* **Macro Stress Score** — the Z-composite that drives the classification. Crossing +0.5 flips toward Risk-Off, crossing -0.5 toward Risk-On.
* **Per-Regime Statistics Table** — the actual edge: annualised return, vol, Sharpe, win rate by regime. Notice how Risk-Off historically delivers <1 Sharpe even when it works at all.
""",
        "math": r"""
**Z-score regime classifier (this dashboard)**
$$\text{regime}_t = \begin{cases}
\text{Risk-On} & \text{MSS}_t < -0.5 \\
\text{Neutral} & |\text{MSS}_t| \le 0.5 \\
\text{Risk-Off} & \text{MSS}_t > +0.5
\end{cases}$$

**Gaussian Mixture Model (probabilistic alternative)**
$$p(x_t) = \sum_{k=1}^{K} \pi_k\,\mathcal{N}(x_t \mid \mu_k, \Sigma_k), \qquad \sum_k \pi_k = 1$$
Soft assignment via responsibilities:
$$\gamma_{tk} = \frac{\pi_k\,\mathcal{N}(x_t \mid \mu_k, \Sigma_k)}{\sum_j \pi_j\,\mathcal{N}(x_t \mid \mu_j, \Sigma_j)}$$

**Hidden Markov Model** (sequential regime persistence)
$$P(R_t \mid R_{t-1}) = A_{R_{t-1}, R_t}, \qquad p(x_t \mid R_t = k) = \mathcal{N}(\mu_k, \Sigma_k)$$
Viterbi gives the most likely regime path.

**Regime-conditional Sharpe**
$$\text{SR}_{R_k} = \frac{E[r_t \mid R_t = k] \cdot 12}{\sigma[r_t \mid R_t = k] \cdot \sqrt{12}}$$
""",
        "terms": """
* **Regime shift** — structural change in the joint distribution of returns/correlations.
* **Risk parity** — Bridgewater's All Weather: scale each asset class so it contributes equal *risk* (not capital) to the book.
* **Macro overlay** — tilt asset weights based on the current regime, on top of a strategic allocation.
* **Vol-targeting** — scale gross exposure so realised volatility hits a constant.
* **Crisis correlation** — the empirical fact that diversification benefits collapse in a real crash; everything goes to one.
""",
    },
    # ───────────────────────────────────────────────────────────────
    "expected_returns": {
        "title": "🤖 Expected Returns — Methodology",
        "chart": """
* **Predicted vs Realised 12M Return** — the model's forecast against what actually happened. Tracking is what matters; level isn't.
* **±1σ Confidence Band** — uncertainty around the prediction (residual standard error). When realised falls outside the band repeatedly, the model is mis-specified.
* **Walk-forward Construction** — at every month *T*, the model is retrained on data ≤ *T*, then asked to predict *T+1...T+12*. No future leakage.
""",
        "math": r"""
**Ridge regression** (L2-regularised least squares; the workhorse of factor models)
$$\hat\beta = \arg\min_\beta \sum_t (y_t - X_t \beta)^2 + \lambda\|\beta\|_2^2$$
Closed form:
$$\hat\beta = (X^{\top}X + \lambda I)^{-1} X^{\top} y$$

**Forward 12-month log return** (the target)
$$y_{t,12} = \log\!\bigl(P_{t+12}/P_t\bigr) = \sum_{i=1}^{12} r_{t+i}$$

**Expanding-window walk-forward** (the protocol)
$$\hat\beta_t = f\bigl(\{(X_s, y_s) : s \le t - 12\}\bigr) \quad\Rightarrow\quad \hat y_{t} = X_t \hat\beta_t$$

**Information Coefficient (IC)** — predictive correlation
$$\text{IC} = \mathrm{corr}(\hat y_t,\, y_t)$$
Quants live and die by IC; an IC of 0.05 over thousands of names beats most anything.

**Out-of-sample R²**
$$R^{2}_{\text{oos}} = 1 - \frac{\sum_t (y_t - \hat y_t)^{2}}{\sum_t (y_t - \bar y)^{2}}$$
""",
        "terms": """
* **Alpha model** — the predictive piece (vs the 'risk model' that estimates covariance).
* **Factor loading** — $\\beta_i$, sensitivity to factor *i*.
* **Cross-validation** — rotate train/test splits to measure generalisation.
* **Overfitting** — in-sample R² explodes, OOS R² collapses; the universal ML failure mode.
* **Shrinkage** — pulling estimates toward a prior (Ridge, James-Stein); lowers variance at the cost of bias.
* **Information ratio** — Sharpe of the residual after hedging out the benchmark.
""",
    },
    # ───────────────────────────────────────────────────────────────
    "screener": {
        "title": "📊 Screener — Methodology",
        "chart": """
* **Multi-ticker grid** — point-in-time snapshot of return, momentum, RSI, and SMA200 status across the universe.
* **YTD column** — calendar-year return; the quintessential cocktail-party number.
* **SMA200 row colour** — a 1-bit trend filter: above means uptrend, below means trend-broken.
""",
        "math": r"""
**Period total return**
$$r_{[t_0, t_1]} = \frac{P_{t_1}}{P_{t_0}} - 1$$

**Wilder's RSI(14)** (the momentum standard since 1978)
$$\text{RS}_t = \frac{\bar U_t}{\bar D_t}, \qquad \text{RSI}_t = 100 - \frac{100}{1 + \text{RS}_t}$$
where Wilder smoothing is
$$\bar U_t = \frac{13\,\bar U_{t-1} + U_t}{14}, \qquad U_t = \max(\Delta P_t, 0), \quad D_t = \max(-\Delta P_t, 0)$$

**Simple moving average**
$$\text{SMA}_n(P_t) = \frac{1}{n}\sum_{i=0}^{n-1} P_{t-i}$$

**Cross-sectional rank / decile**
$$\text{rank}_t(i) = \frac{|\{j : x_{j,t} \le x_{i,t}\}|}{N}, \quad \text{decile}_t(i) = \lceil 10\,\text{rank}_t(i)\rceil$$
""",
        "terms": """
* **Universe** — the set of names a strategy is allowed to consider.
* **Point-in-time (PIT)** — only data observable at moment *T*; the discipline that prevents lookahead.
* **Long/short equity (L/S)** — buy decile 10, sell decile 1, harvest the spread.
* **Beta-neutral** — net dollar long ≈ net dollar short × β-adjustment, so the book is hedged against market direction.
* **Sector neutral** — same idea applied within each GICS sector.
* **Capacity** — maximum AUM a strategy can hold without its own trades moving prices against it.
""",
    },
    # ───────────────────────────────────────────────────────────────
    "technical": {
        "title": "📉 Technical — Methodology",
        "chart": """
* **🎯 Buy Zone Scanner** — weekly bars compressed into a 5-zone label per ticker (Strong Buy / Pullback / Trend / Avoid / Watch) plus a *technical* buy range derived from the 20w/50w MAs. **Reference levels, not advice.**
* **Single-ticker deep dive** — daily candlesticks with SMA20/50/200, Bollinger Bands, volume, RSI, and MACD on stacked sub-panels.
* **OBV / ROC** — confirmation indicators. Price up + OBV up ⇒ healthy trend; price up + OBV flat ⇒ distribution.
""",
        "math": r"""
**Exponential moving average**
$$\text{EMA}_t = \alpha P_t + (1-\alpha)\,\text{EMA}_{t-1}, \qquad \alpha = \frac{2}{N+1}$$

**MACD (12-26-9)**
$$\text{MACD}_t = \text{EMA}_{12}(P_t) - \text{EMA}_{26}(P_t)$$
$$\text{Signal}_t = \text{EMA}_{9}(\text{MACD}_t), \qquad \text{Hist}_t = \text{MACD}_t - \text{Signal}_t$$

**Bollinger Bands (20, 2σ)**
$$\text{BB}_{u/l} = \text{SMA}_{20}(P) \pm 2\,\sigma_{20}(P)$$

**Volume-Weighted Average Price**
$$\text{VWAP}_t = \frac{\sum_{i \le t} P_i V_i}{\sum_{i \le t} V_i}$$

**On-Balance Volume**
$$\text{OBV}_t = \text{OBV}_{t-1} + \mathrm{sign}(\Delta P_t)\,V_t$$

**Average True Range (14)**
$$\text{TR}_t = \max\bigl(H_t - L_t,\;|H_t - C_{t-1}|,\;|L_t - C_{t-1}|\bigr)$$
$$\text{ATR}_{14}(t) = \tfrac{1}{14}\sum_{i=0}^{13} \text{TR}_{t-i}$$

**Faber 10-month rule** (the simplest trend filter that works)
$$\text{Long}_t = \mathbf{1}\bigl[P_t > \text{SMA}_{10\text{m}}(P_t)\bigr]$$
""",
        "terms": """
* **Trend-following** — buy strength, sell weakness; CTAs and managed-futures houses live here.
* **Mean reversion** — bet against extreme moves; stat-arb shops live here.
* **Pullback / dip** — temporary retrace inside an uptrend; canonical entry for trend-followers.
* **Death cross / Golden cross** — SMA50 crossing SMA200 down/up; visually obvious, statistically weak alone but useful as a regime filter.
* **Pivot / Support / Resistance** — price levels where reversals tend to cluster.
* **Liquidity sweep** — fast move that takes out stops above/below an obvious level before reversing.
""",
    },
    # ───────────────────────────────────────────────────────────────
    "risk_sim": {
        "title": "🎲 Risk Sim — Methodology",
        "chart": """
* **Fan chart** — distribution of simulated price paths (5 / 25 / 50 / 75 / 95 percentile envelopes). Width = uncertainty; the 5th-percentile path is the 'realistic worst case' for capital planning.
* **VaR / CVaR table** — single-number summaries of tail loss at common confidence levels.
""",
        "math": r"""
**Geometric Brownian Motion** (the textbook stock-price model)
$$dS_t = \mu\,S_t\,dt + \sigma\,S_t\,dW_t$$
Discretised one period:
$$S_{t+\Delta t} = S_t \exp\!\Bigl(\bigl(\mu - \tfrac{1}{2}\sigma^{2}\bigr)\Delta t + \sigma\sqrt{\Delta t}\,Z\Bigr), \quad Z \sim \mathcal{N}(0,1)$$

**Value at Risk** (the α-quantile of loss)
$$\text{VaR}_\alpha(L) = \inf\{x \in \mathbb{R} : P(L \le x) \ge \alpha\}$$

**Conditional VaR / Expected Shortfall** (the *coherent* tail measure)
$$\text{CVaR}_\alpha = E[L \mid L \ge \text{VaR}_\alpha] = \frac{1}{1-\alpha}\int_\alpha^{1} \text{VaR}_u\,du$$

**Cholesky for correlated normals**
$$\Sigma = LL^{\top} \quad\Rightarrow\quad X = \mu + L\,Z, \quad Z \sim \mathcal{N}(0, I)$$

**Antithetic variance reduction**
$$\hat\theta = \tfrac{1}{2}\bigl[f(Z) + f(-Z)\bigr] \quad\Rightarrow\quad \mathrm{Var}(\hat\theta) \le \tfrac{1}{2}\mathrm{Var}\bigl(f(Z)\bigr)$$
""",
        "terms": """
* **Tail risk** — probability-mass in the loss tail beyond model assumptions.
* **Parametric VaR** — assumes a parametric loss distribution (often normal); fast but blind to fat tails.
* **Historical VaR** — empirical quantile of past returns; honest about what *did* happen, not what *could*.
* **Stress test** — replay specific historical or hypothetical adverse scenarios.
* **Coherent risk measure** — a measure satisfying monotonicity, sub-additivity, homogeneity, translation invariance. CVaR is coherent; VaR is not.
* **Black swan** — extreme tail event that the modeller didn't price in.
""",
    },
    # ───────────────────────────────────────────────────────────────
    "gemini": {
        "title": "✨ Gemini AI Analyst — Methodology",
        "chart": """
* This tab is *generated text*, not a chart. The pipeline is:
  1. Snapshot the live macro state (regime score, yields, vol, drawdown, momentum).
  2. Fill it into a structured 'senior macro analyst' system prompt.
  3. Call Gemini and stream the response.
* Pick one of five framings: Full Briefing, Regime Deep-Dive, Risk Assessment, Investment Outlook, Custom Q&A.
* Treat output as a *draft research note*. Cross-check any specific number it cites against the other tabs.
""",
        "math": r"""
**Softmax sampling** (how the next token is chosen given the model's logits $z$)
$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
*Temperature* $T$ controls sharpness: $T \to 0$ greedy (deterministic), $T = 1$ default, $T > 1$ creative.

**Top-p (nucleus) sampling**
$$\text{Sample from smallest } S : \sum_{i \in S} p_i \ge p_{\text{nucleus}}$$

**Retrieval-Augmented Generation (RAG)** — what this tab does
$$\text{prompt} = \text{template}(\text{context}_{\text{live}}) \;\Rightarrow\; \text{LLM}(\text{prompt}) \;\Rightarrow\; \text{briefing}$$

**Context window** — Gemini 1.5 Flash supports up to ~1 M input tokens; this tab uses < 4 K.
""",
        "terms": """
* **System instruction** — persistent role/voice prompt prepended to every call.
* **Token** — sub-word unit; English averages ~4 chars/token.
* **Hallucination** — confident assertion of a fact the model cannot verify; mitigated here by injecting *real* macro numbers into the prompt.
* **Prompt injection** — adversarial content in retrieved data that hijacks the model; this tab is closed-loop so risk is low.
* **Sell-side note** — a research desk's published view on a stock or theme; the voice this tab tries to imitate.
""",
    },
    # ───────────────────────────────────────────────────────────────
    "danger_zone": {
        "title": "🔥 NVDA Danger Zone — Methodology",
        "chart": """
* **Danger Zone Map** — candlesticks + the Composite Danger Index on a stacked panel. Watch where the index breaks 0.6 (red zone) for clusters of historical mean-reversion.
* **Volume Footprint** — green/red volume bars with block-trade flags (>2× SMA20). Sustained >2× spikes near highs ≠ accumulation; near lows ≠ capitulation.
* **B/A Imbalance Proxy** — synthetic order-flow imbalance; positive = closes near the high (buyer-dominated), negative = closes near the low.
* **Strategy Hint Cards** — three colour-coded cards mirroring the active danger zone. Active card highlight is the canonical 'fix-G1' implementation.
""",
        "math": r"""
**Composite Danger Index** (this dashboard)
$$D_t = 0.30\,\tilde z(\text{RSI}_t) + 0.25\,\tilde z(\text{ATR\%}_t) + 0.20\,\tilde z(\text{volRatio}_t) + 0.15\,\tilde z(\text{VIX}_t) + 0.10\,\tilde z(\text{ext}_t)$$
where $\tilde z(\cdot)$ is min-max scaled to $[0,1]$ over the visible window, then smoothed with a 3-bar rolling mean.

**ATR percent**
$$\text{ATR\%}_t = \frac{\text{ATR}_{14}(t)}{P_t}\times 100$$

**Block-trade flag**
$$B_t = \mathbf{1}\bigl[V_t > 2\,\text{SMA}_{20}(V)\bigr]$$

**Bid-Ask imbalance proxy** (intraday close vs midpoint)
$$\text{BAI}_t = \frac{C_t - (H_t + L_t)/2}{H_t - L_t + \varepsilon}\times 100$$

**Volatility spike**
$$S_t = \mathbf{1}\bigl[\text{ATR\%}_t > \mu_{30}(\text{ATR\%}) + \sigma_{30}(\text{ATR\%})\bigr]$$

**VIX** (CBOE 30-day SPX implied variance, simplified)
$$\text{VIX}^{2} = \frac{2}{T}\sum_{i}\frac{\Delta K_i}{K_i^{2}}\,e^{rT}\,Q(K_i) - \frac{1}{T}\!\left(\frac{F}{K_0}-1\right)^{2}$$
""",
        "terms": """
* **Block trade** — institutional-size order printed on the tape; proxy for absorption/distribution.
* **Order-flow imbalance** — net pressure between aggressive buyers and aggressive sellers in the order book.
* **Volatility surface** — $\\sigma_{\\text{IV}}(K, T)$, the implied-vol function across strikes and maturities.
* **Skew / smile** — non-flat shape of $\\sigma_{\\text{IV}}(K)$; equity skew is asymmetric (puts richer than calls).
* **Gamma squeeze** — dealers short gamma have to buy as price rises, amplifying the move; classic in single names with big options open interest.
* **Crowding** — when too many funds hold the same exposure, the unwind path becomes the risk.
""",
    },
    # ───────────────────────────────────────────────────────────────
    "backtest": {
        "title": "📊 Strategy Backtest — Methodology",
        "chart": """
* **Equity curves** — strategy (net of bps cost) vs SPY buy-and-hold, both on the same Base = 100 scale.
* **Drawdown overlay** — the strategy's drawdown vs B&H's. The cleanest visual proof of regime gating's value: shallower troughs in the strategy line during 2008, 2020, 2022.
* **Position-through-time** — when the strategy was long (1), short (-1), or flat (0). Lagged 1 month so it visually represents what was held *into* each return period.
* **Side-by-side tear sheet** — Sharpe, MDD, Calmar, Win Rate, Alpha — strategy vs B&H, no marketing massage.
""",
        "math": r"""
**No-lookahead position lag**
$$w_t = f\bigl(\text{info}_{\le t-1}\bigr), \qquad r^{p}_t = w_t \cdot r^{\text{SPY}}_t$$

**Aggregate signal score** (this dashboard)
$$s_t = \frac{1}{|\mathcal{S}|}\sum_{i \in \mathcal{S}} \mathbf{1}[\text{gate}_i(t)], \qquad w^{*}_t = \mathbf{1}[s_t \ge \tau]$$

**Turnover & transaction cost**
$$\text{TO}_t = |w_t - w_{t-1}|, \qquad r^{\text{net}}_t = r^{p}_t - c_{\text{bps}} \cdot 10^{-4} \cdot \text{TO}_t$$

**12-1 momentum** (Jegadeesh-Titman; skip the most recent month to dodge short-term reversal)
$$\text{Mom}_{12-1}(t) = \log\!\bigl(P_{t-1}/P_{t-12}\bigr)$$

**Faber rule (10-month SMA filter)**
$$\text{gate}_{\text{trend}}(t) = \mathbf{1}\bigl[P_t > \text{SMA}_{10\text{m}}(P_t)\bigr]$$

**Equity curve compounding (log)**
$$E_t = E_0 \cdot \exp\!\Bigl(\textstyle\sum_{s \le t} r^{\text{net}}_s\Bigr)$$
""",
        "terms": """
* **Walk-forward** — train on $[0, t]$, test on $[t+1, t+h]$, slide and repeat. The honest backtest protocol.
* **Lookahead bias** — using information at time $t$ that wasn't observable until $t + \\varepsilon$. The most common cause of fictitious Sharpe.
* **Survivorship bias** — testing only on currently-existing tickers; biases CAGR upward by ~1 % per year on US large-caps.
* **Slippage** — gap between modelled fill and realised fill. The first thing live trading teaches you.
* **Capacity** — AUM ceiling above which the strategy's own trades move prices against it.
* **Factor crowding** — when a factor becomes consensus (post-publication), live Sharpe deteriorates toward 50 % of historical.
* **Backtest overfitting** — searching enough strategies in-sample produces a great fit by chance; Bailey & López de Prado quantify this with the Probability of Backtest Overfitting (PBO).
""",
    },
    # ───────────────────────────────────────────────────────────────
    "quant_signals": {
        "title": "🎯 Quant Signals — Methodology",
        "chart": """
* **Per-ticker table** — one row per watchlist name: signal label, additive score, volatility regime, and whether a volatility breakout is currently firing.
* **Reasoning detail** — every point in the score is traced back to the indicator that produced it, in plain language.
* **Bollinger chart** — the same price-vs-band view as the Technical tab, so a breakout call can be eyeballed against the bands that triggered it.
""",
        "math": r"""
**Average True Range (14)** — the volatility unit, in price terms
$$TR_t = \max(H_t - L_t,\ |H_t - C_{t-1}|,\ |L_t - C_{t-1}|), \qquad \text{ATR}_{14} = \frac{1}{14}\sum_{i=0}^{13} TR_{t-i}$$

**Annualized realized volatility (20d)**
$$\sigma^{ann}_t = \text{std}_{20}\bigl(\log(P_t/P_{t-1})\bigr) \cdot \sqrt{252}$$

**Bollinger Band width + its own percentile rank** (the squeeze detector)
$$\text{BBW}_t = \frac{\text{BB}_{upper,t} - \text{BB}_{lower,t}}{\text{SMA}_{20,t}}, \qquad \text{pct}_t = \frac{|\{s \le t : \text{BBW}_s \le \text{BBW}_t\}|}{252}$$

**Volatility breakout condition** — squeeze roughly two weeks ago, expanding now, price outside the band
$$\text{pct}_{t-10} \le 0.20 \ \wedge\ \text{BBW}_t > 1.15\cdot\text{BBW}_{t-5} \ \wedge\ (P_t > \text{BB}_{upper,t} \ \vee\ P_t < \text{BB}_{lower,t})$$

**Composite score** — five bounded, signed components summed and clipped
$$\text{score}_t = \text{clip}\Bigl(\textstyle\sum_k w_k \cdot \mathbf{1}[\text{condition}_k(t)],\ -100,\ 100\Bigr)$$
""",
        "terms": """
* **Volatility squeeze** — Bollinger Bands compressed to a low percentile of their own trailing range; energy coiling before a directional move, not a signal by itself.
* **Volatility breakout** — the move that follows a squeeze once price closes outside the bands; this dashboard scores the breakout, not the squeeze.
* **Realized vs. implied volatility** — this tab only uses realized (historical, backward-looking) volatility from price data; it does not use options-implied volatility.
* **Conviction dampening** — reducing a score's magnitude (not flipping its sign) when the regime is noisy enough that the same signal is less trustworthy.
* **Advisory signal** — a recommendation and its reasoning, with no order sizing, routing, or execution attached.
""",
    },
    # ───────────────────────────────────────────────────────────────
    "alexa": {
        "title": "🎙️ Alexa+ & Model Context Protocol (MCP) — Methodology",
        "chart": """
* **Interactive Alexa+ Simulator** — Emulates the voice conversational experience of Alexa+ connected to MacroPulse via Streamable HTTP Transport.
* **Multimodal Card View** — Displays visual tear-sheets and KPI metrics formatted for Echo Show and Fire TV screens.
* **Agent Skill Execution Trace** — Real-time telemetry displaying the selected MCP tool, latency, extracted arguments, and raw JSON-RPC payload.
""",
        "math": r"""
**Streamable HTTP (Server-Sent Events) Transport**
$$\text{Client} \xrightarrow{\text{GET /sse}} \text{MCP Server (assigns SessionID)} \xrightarrow{\text{endpoint event}} \text{Client}$$
$$\text{Client} \xrightarrow{\text{POST /messages?session\_id=...}} \text{JSON-RPC 2.0 Request} \xrightarrow{\text{SSE stream}} \text{Streamed Response}$$

**Parametric Value-at-Risk (Normal Distribution)**
$$\text{VaR}_\alpha = -(\mu \Delta t + z_\alpha \sigma \sqrt{\Delta t})$$

**Empirical Expected Shortfall (CVaR)**
$$\text{CVaR}_\alpha = -\mathbb{E}[R \mid R \le -\text{VaR}_\alpha]$$
""",
        "terms": """
* **Model Context Protocol (MCP)** — An open standard allowing LLMs and AI agents (such as Alexa+) to access live tools and context from external systems.
* **Streamable HTTP Transport** — The 2025-11-25 MCP standard specifying bi-directional communication over HTTP SSE streams and message endpoints.
* **Agent Skills** — Declarative tool manifests defining capability boundaries, argument schemas, and invocation triggers for Alexa+.
* **Multimodal Voice Ergonomics** — Pairing concise, natural conversational audio speech with high-density visual telemetry on smart displays.
""",
    },
    # ───────────────────────────────────────────────────────────────
    "portfolio_fomc": {
        "title": "🏦 Multi-Asset Portfolio Sync & FOMC Shock Testing — Methodology",
        "chart": """
* **Multi-Asset Allocation Breakdown** — Displays capital distribution across Equities, Fixed Income/Duration, Commodities, and Digital Assets.
* **Marginal Risk Contribution (Component VaR)** — Highlights which asset class or single name drives the greatest portion of portfolio volatility.
* **FOMC Rate Shock Matrix** — Real-time PnL simulation across four monetary policy scenarios: Hawkish Surprise (+50 bps), Dovish Pivot (-50 bps), Stagflationary Inversion, and Liquidity Crunch.
* **Big Screen / 10-Foot TV Display** — High-contrast, large-typography mode engineered for Fire TV screens and trading desk wall displays during FOMC rate announcements.
""",
        "math": r"""
**Portfolio Variance & Volatility (Covariance Formulation)**
$$\sigma_p^2 = \mathbf{w}^T \mathbf{\Sigma} \mathbf{w} = \sum_{i=1}^N \sum_{j=1}^N w_i w_j \sigma_{ij}, \qquad \sigma_p = \sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}}$$

**Cross-Asset Parametric Value-at-Risk ($h$-day horizon)**
$$\text{VaR}_\alpha = - z_\alpha \cdot \sigma_p \sqrt{\frac{h}{252}} \cdot V_0$$

**Marginal Contribution to Risk (MCR) & Component VaR**
$$\text{MCR}_i = \frac{\partial \sigma_p}{\partial w_i} = \frac{(\mathbf{\Sigma} \mathbf{w})_i}{\sigma_p}, \qquad \text{CVaR}_i = w_i \cdot \text{MCR}_i$$

**FOMC Duration & Factor Sensitivity Shock**
$$\Delta P_{\text{FI}} \approx -D_{\text{eff}} \cdot \Delta y + \frac{1}{2} C \cdot (\Delta y)^2, \qquad \Delta V_{\text{portfolio}} = \sum_{i=1}^N w_i \cdot \beta_i^{\text{FOMC}}(\Delta r) \cdot V_0$$
""",
        "terms": """
* **Effective Duration ($D_{\text{eff}}$)** — Percentage price change of a bond instrument per 100 bps shift in yields.
* **FOMC Rate Shock** — An instantaneous, unanticipated shift in the target Federal Funds rate and terminal rate expectations.
* **Risk Parity** — Allocating capital such that each asset class contributes equally to the total portfolio variance budget.
* **Cross-Asset Correlation Breakdown** — Tendency of all risk assets to correlate to 1.0 during liquidity flash crunches.
""",
    },
    # ───────────────────────────────────────────────────────────────
    "equity_report": {
        "title": "📑 Equity Report — Methodology",
        "chart": """
* **KPI tiles** — last price, probability-weighted DCF value, base-case DCF, rule-based rating, WACC and the daily pattern screen.
* **Macro regime overlay** — the dashboard's Risk-On / Neutral / Risk-Off call re-weights the bear/base/bull scenarios and nudges the equity risk premium. The note shows what the value would be with neutral 25/50/25 weights.
* **Valuation range (football field)** — 52-week range, historical P/E and P/B bands, DCF sensitivity and scenario range against the dotted last-price line.
* **Sensitivity grid** — value per share for WACC ±1pp × terminal growth ±1pp; blue cells sit above the price, orange below.
* **Technical charts** — Monthly (3y), Weekly (12m), Daily (6m) candles with moving averages, support/resistance, pattern markers (▲ bullish / ▼ bearish), MACD and slow stochastic (KD) panels.
* **FOMC badges (H / D / S / L)** — where each tab-13 FOMC shock lands on the sensitivity grid; the table under the grid gives the exact WACC, g and value per share.
* **Touch probabilities** — use the drift implied by the Expected Returns tab's Ridge S&P 500 forecast (via CAPM); the driftless value is shown alongside.
* **Downloads** — Excel model with live formulas (edit the yellow cells), Word research note, offline HTML dashboard, or everything as a zip. Buy Zone scanner rows link straight to a ticker's report (`?ticker=`).
""",
        "math": r"""
**Free cash flow (FCFF proxy) and projection**
$$\text{FCF}_t = \text{Revenue}_t \times m, \qquad g_t = g_1 + (g_\infty - g_1)\frac{t-1}{N-1}$$

**Discount rate (CAPM + after-tax debt)**
$$k_e = r_f + \beta\,(\text{ERP} + \Delta_{\text{regime}}), \qquad \text{WACC} = \tfrac{E}{D+E}k_e + \tfrac{D}{D+E}k_d(1-\tau)$$

**Enterprise and equity value**
$$\text{EV} = \sum_{t=1}^{N}\frac{\text{FCF}_t}{(1+\text{WACC})^t} + \frac{\text{FCF}_N(1+g_\infty)}{(\text{WACC}-g_\infty)(1+\text{WACC})^N}, \qquad V/\text{share} = \frac{\text{EV} - \text{net debt}}{\text{shares}}$$

**Regime-weighted expected value**
$$\mathbb{E}[V] = \sum_{s\in\{\text{bear},\text{base},\text{bull}\}} p_s(\text{regime})\, V_s$$
Risk-On 20/50/30 (ERP −0.25pp) · Neutral 25/50/25 · Risk-Off 40/45/15 (ERP +0.50pp).

**Blume-adjusted beta** (2y weekly returns vs S&P 500): $\beta_{adj} = 0.67\,\beta_{raw} + 0.33$

**Touch-probability drift** (tab-4 Ridge S&P 500 view → CAPM)
$$\mathbb{E}[R_s] = r_f + \beta\,(\mathbb{E}[R_m]^{\text{Ridge}} - r_f), \qquad \nu = \ln(1+\mathbb{E}[R_s]) - \tfrac{\sigma^2}{2}$$

**6-month touch probability** (first passage of log price with drift $\nu$ to $b = \ln(K/S)$, $s=\sigma\sqrt{T}$)
$$P_{b>0} = \Phi\!\left(\frac{\nu T - b}{s}\right) + e^{2\nu b/\sigma^2}\,\Phi\!\left(\frac{-b - \nu T}{s}\right)$$
With $\nu = 0$ (no macro history) this is the reflection-principle value $2\left[1 - \Phi(|b|/s)\right]$, which the report also shows for comparison.

**FOMC shock overlay** (tab-13 scenarios on the WACC × g grid)
$$\text{WACC}' = \tfrac{E}{D+E}\bigl(r_f + \Delta r_f + \beta(\text{ERP} + \Delta\text{ERP})\bigr) + \tfrac{D}{D+E}(k_d + \Delta r_f + \Delta s)(1-\tau), \qquad g' = g_\infty + \Delta g$$
Hawkish +50/+25/+25bp · Dovish −50/−25/−10bp · Stagflation +25/+75/+120bp, g −50bp · Liquidity crunch −25/+150/+200bp ($\Delta r_f$/$\Delta$ERP/$\Delta s$), FCF path fixed.
""",
        "terms": """
* **FCFF proxy** — cash from operations minus capital expenditures from SEC 10-K XBRL filings; simpler than EBIT(1−t)+D&A−CapEx−ΔNWC and fully traceable to the filing.
* **Terminal value share** — when the Gordon terminal value is >80% of EV, the valuation is mostly a bet on WACC and $g_\\infty$.
* **Football field** — side-by-side valuation ranges from different methods; agreement between methods matters more than any single point.
* **Chaikin Money Flow / OBV** — price-volume proxies for accumulation. The US has no daily institutional net-buy tape like Korea's KRX, so ownership (13F) and short interest are shown as periodic snapshots.
* **Rule-based rating** — a transparent score from valuation, multi-timeframe trend, pattern screen and money flow. The optional LLM narrative may only quote numbers from that JSON.
""",
    },
}


def render(tab_key: str, st: Any) -> None:
    """Drop a 'How to read · Math · HF Terms' expander on the current tab.

    The Streamlit expander is collapsed by default to keep first paint clean.
    Math is rendered via MathJax (st.markdown handles both inline `$x$` and
    block `$$...$$`). Pass the streamlit module as `st` so this stays an
    importable plain module.
    """
    m = METHODOLOGY.get(tab_key)
    if m is None:
        return
    with st.expander("📐 How to read this tab · Math · HF Terms", expanded=False):
        st.markdown(f"### {m['title']}")
        st.markdown("#### How to read the charts")
        st.markdown(m["chart"])
        st.markdown("#### Mathematical formulation")
        st.markdown(m["math"])
        st.markdown("#### Hedge-fund terminology")
        st.markdown(m["terms"])

