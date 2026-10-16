# MacroPulse validation: pre-registration

Written 2026-10-16, before any of the tests below were run. The model code under test is `main` at
commit `dd1c3d4`. Every record of the forward log stores the commit and a fingerprint of the model
source files, so a later code change cannot be scored as if it were this version.

This document fixes what each hypothesis is, how it is tested and what counts as a result. Changing a
hypothesis, a metric, a threshold or an evaluation period after the results are seen is not allowed. A
change needs a new section dated after it, and that section's results are reported separately.

## 1. What is tested

| ID | Component | Code (at `dd1c3d4`) | Claim made in the product |
|---|---|---|---|
| H1 | Macro regime score | `src/macro_model.py` `add_regime` | Risk-Off months are riskier than Risk-On months |
| H2 | 12-month expected return | `src/macro_model.py` `expected_return_model` | The Ridge forecast is informative about the next 12 months |
| H3 | SPY/cash strategy | `src/portfolio.py` `run_strategy_backtest` | The default strategy beats buy-and-hold on a risk-adjusted basis |
| H4 | Monte Carlo VaR | `src/alexa_mcp_server.py` `execute_simulate_portfolio_risk` | The VaR is a calibrated loss quantile |
| H5 | DCF equity report | `src/equity_report/engine` | Probability-weighted upside ranks stocks by later excess return |

Each component ends with one of three conclusions: a statistically significant effect, no effect, or
not decidable with the data available. "No effect" is a valid result. If a component gets it, the
README and the Devpost text stop making that claim.

## 2. Rules for every test

- Significance level 5%. The five primary tests (one per component, marked *primary* below) are one
  family, corrected with the Holm method.
- Overlapping 12-month outcomes: Newey–West standard errors with at least 12 lags.
- Bootstrap: stationary bootstrap, mean block length 6 months, 10,000 resamples, seed 0.
- Historical evaluation period: from the first month every input of the component is defined, through
  2026-09 (the last complete month before this document). It is not shortened or split to find a
  better result. The sub-periods below are reported as robustness checks only.
- Sub-periods: 2005–2009, 2010–2019, 2020–2026-09; and NBER recession months (FRED `USREC`) versus
  expansion months.
- Every number in the results carries its source, its sample period and its sample size. A value that
  could not be computed is reported as not computed, with the reason.

## 3. Trial ledger

The number of configurations tried feeds the Deflated Sharpe Ratio in H3. It is counted from the git
history (142 commits at `dd1c3d4`) and from what the dashboard lets a user change.

| Component | Parameters | Values used in the history | Trials |
|---|---|---|---|
| Regime | threshold, z-score warm-up | 0.5 and 36 months since the first commit (2026-03-31) | 1 |
| Expected return | Ridge alpha, features, min. training months | 5.0; the same 7 features; 48, since the first commit | 1 |
| Strategy | signal subset, threshold 0.34–1.00, short on/off | every rule a user can reach in the Backtest tab | 32 |
| VaR | method, window, seed | GBM on 1 year of daily returns, seed 42 | 1 |
| DCF | growth path, beta, base period | changed in PRs #25–#28 after reviewing one stock (AVGO) | see H5 |

The 32 strategy rules: 11 distinct long-only rules and 21 distinct long/short rules. Thresholds that
produce the same positions count once; for example, with three signals every threshold from 0.34 to
0.66 means "at least two of three". The cost slider changes net returns, not the rule, so it adds no
trials. Earlier changes to the data (point-in-time z-scores, removing the 10Y-yield credit proxy)
were bug fixes, not searches over parameters, and add no trials either.

## 4. Hypotheses

### H1. Regime (*primary*: realized volatility)

- Claim: realized S&P 500 volatility over the 21 trading days after a Risk-Off reading is higher than
  after a Risk-On reading.
- Null hypothesis: the mean log realized volatility is the same after both labels.
- Test: one-sided difference in mean log realized volatility (Risk-Off minus Risk-On), stationary
  bootstrap p-value.
- Secondary: next-month return difference (two-sided); the share of NBER recession months labelled
  Risk-Off, the share of Risk-Off months outside recessions, and the lead in months before each
  recession start.
- Historical data: monthly regime labels from `add_regime` with daily `^GSPC` closes.

### H2. Expected return (*primary*: Clark–West)

- Benchmark: the prevailing mean, expm1(12 × mean monthly S&P 500 log return, using only months up to
  the forecast date), as in Goyal and Welch (2008).
- Metric: Campbell–Thompson out-of-sample R², 1 − Σ(model error)² / Σ(benchmark error)².
- Null hypothesis: the model's mean squared error is not lower than the benchmark's.
- Test: one-sided Clark–West (2007) test, Newey–West with 12 lags.
- Secondary: whether the sign of the forecast minus the benchmark matches the sign of the outcome
  minus the benchmark (Pesaran–Timmermann test).
- Pipeline check first: the same code must reproduce published out-of-sample R² values for the
  dividend yield, term spread and default spread predictors on Amit Goyal's public data, within 0.5
  percentage points. If it does not, H2 is not decidable until the pipeline is fixed.

### H3. Strategy (*primary*: Deflated Sharpe Ratio)

- Strategy under test: the dashboard default, the only one the product presents as its strategy:
  regime + 12-1 momentum + 10-month trend, threshold 0.5, long-only, 5 bp per unit of turnover, cash
  earning the 3-month T-bill.
- Null hypothesis: the strategy's Sharpe ratio, deflated for 32 trials, is not above zero.
- Test: Deflated Sharpe Ratio (Bailey and López de Prado 2014) of at least 0.95.
- Secondary: Probability of Backtest Overfitting by CSCV with 16 blocks over the 32 rules; Hansen's
  SPA test against buy-and-hold, the 10-month trend rule alone, and 60/40; net results at 0, 5, 10
  and 25 bp; one-month execution delay.

### H4. VaR (*primary*: Kupiec, 95%)

- Forecast: the live tool's 21-trading-day VaR on SPY at 95% and 99%, recomputed at each month-end
  with only the year of daily returns before it.
- Null hypothesis: the exceedance rate equals 5% (95% VaR).
- Test: Kupiec proportion-of-failures likelihood-ratio test.
- Secondary: the Kupiec test at 99%; the Christoffersen independence test; the Basel traffic-light zone
  for 99% exceedances; the Acerbi–Székely test of the expected shortfall.

### H5. DCF (*primary*: rank IC)

- Universe: S&P 500 constituents at each year-end from 2019 to 2024, using constituent lists from that
  date, excluding financials and real estate (FCFF is not meaningful for them).
- Inputs: only filings public on the year-end date (SEC filing dates); prices up to that date.
- Null hypothesis: the mean cross-sectional Spearman correlation between upside and the next 12-month
  return in excess of SPY is not above zero.
- Test: one-sided t-test on the six yearly correlations, with Newey–West standard errors.
- Secondary: mean excess return by rating; reconciliation of revenue, operating cash flow, capital
  expenditure, interest expense, diluted shares and net debt against the filing tables for 10 companies,
  flagging any difference over 0.5%.
- Known bias: the DCF settings were changed in October 2026 after reviewing AVGO only. AVGO is
  therefore reported but left out of the H5 sample.

## 5. Forward log

The historical tests above are run on data that already exists, so they can still be overfit by the
choice of what to test. The forward log is the check that cannot be: forecasts are recorded before the
outcome happens and are never changed afterwards.

- Code: `validation/forward_log.py` (writes records), `validation/score_forward_log.py` (scores them),
  `validation/spec.py` (the constants below), `.github/workflows/forward-log.yml` (runs both).
- Schedule: the 1st of every month at 12:17 UTC, before the US open, plus manual runs.
- Content of each record: the live regime, the 12-month expected return and its benchmark, the
  strategy's position for next month, the 21-day SPY VaR at 95% and 99%, and the equity report for
  five stocks. The inputs are stored as snapshots with their SHA-256.
- Integrity: each record holds the hash of the previous one. CI rejects a log whose chain or
  snapshots do not verify, and a pull request that changes or removes an existing record.
- Late records: a record written after the next session's open is not scored.
- Primary records: a record whose as-of close is the last session of a month. Only primary records
  enter the forward tests; others, including the first (pilot) record written in mid-October 2026,
  are scored and shown as descriptive.
- Model versions: records are grouped by the fingerprint of the model source files; a code change
  starts a new sample with its own minimum.
- Forward tests: the same tests as H1–H4, run once a model version reaches the minimum sample below.
  The five-stock equity panel is descriptive only: five names cannot support a cross-sectional test.

### Pre-registered constants

| Constant | Value |
|---|---|
| Equity tickers | AVGO, AAPL, XOM, JNJ, WMT |
| VaR ticker, confidences, horizon | SPY, 95 and 99, 21 trading days |
| VaR paths | 5000 |
| Regime outcome window | 21 trading days |
| Expected-return and equity horizon | 12 months |
| Outcome index; total-return benchmark | ^GSPC; SPY |
| Minimum primary records: regime | 24, with at least 6 Risk-On and 6 Risk-Off |
| Minimum primary records: expected return | 24 |
| Minimum primary records: VaR 95% | 36 |
| Minimum primary records: VaR 99% | 60 |
| Minimum primary records: strategy position | 36 |

At one primary record a month, the first forward test (H1 regime) is due in 2028 at the earliest, and
the first expected-return test in 2029, because each 12-month outcome takes a year to close. Until then
the forward log is evidence that the forecasts existed before the outcomes, not yet evidence that they
are right.
