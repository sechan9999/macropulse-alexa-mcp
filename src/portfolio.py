"""
src/portfolio.py
─────────────────────────────────────────────────────────────────
Streamlit-free portfolio maths shared by the dashboard (app.py), the headless data layer
(src/macro_data.py, src/macro_extras.py) and the MCP tools, so there is one copy of each formula:

  • compute_hf_metrics  — tear-sheet statistics. Sharpe and Sortino are measured on returns in
                          excess of the 3-month T-bill; Sortino uses the downside deviation
                          sqrt(mean(min(excess, 0)^2)) over *all* months, not the std of the
                          negative months.
  • rf_monthly          — monthly log T-bill return from the ^IRX column, lagged one month so the
                          rate earned during month t is the one quoted at the end of month t-1.
  • run_strategy_backtest — regime/momentum/trend-gated SPY/cash backtest; the cash leg earns the
                          T-bill rate.
  • run_monte_carlo     — Gaussian (GBM) paths, or a bootstrap of historical monthly returns.

All returns are monthly log returns, indexed by month-start labels like load_macro().
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

TBILL_TICKER = "^IRX"        # 13-week T-bill yield, annual %
TBILL_COLUMN = "tbill_3m"
SQRT12 = np.sqrt(12)

MC_GAUSSIAN = "gaussian"
MC_BOOTSTRAP = "bootstrap"


# ── Risk-free rate ─────────────────────────────────────────────────────────────
def rf_monthly(macro_df: pd.DataFrame) -> Optional[pd.Series]:
    """Monthly log return of holding 3-month T-bills, or None if the frame has no T-bill data.

    Row t is the return earned *during* month t, i.e. the yield quoted at the end of month t-1
    (known in advance, so using it is not look-ahead)."""
    if TBILL_COLUMN not in macro_df or macro_df[TBILL_COLUMN].notna().sum() == 0:
        return None
    y = macro_df[TBILL_COLUMN].astype(float).ffill() / 100.0
    return (np.log1p(y) / 12.0).shift(1).rename("rf_m")


def _rf_on(index: pd.Index, rf: Optional[pd.Series]) -> pd.Series:
    """rf on the given months; months without a rate (before the series starts) earn 0."""
    if rf is None:
        return pd.Series(0.0, index=index)
    return rf.reindex(index).ffill().fillna(0.0)


# ── Tear-sheet metrics ─────────────────────────────────────────────────────────
def compute_hf_metrics(rets: pd.Series, bench_rets: Optional[pd.Series] = None,
                       rf: Optional[pd.Series] = None) -> dict:
    """Annualised tear sheet from monthly log returns.

    rf is a monthly risk-free series (see rf_monthly); None means a 0% risk-free rate.
    Sharpe = mean excess / std excess; Sortino = mean excess / downside deviation of the excess
    return; alpha is Jensen's alpha against the benchmark's excess return."""
    rets = rets.dropna()
    rf_m = _rf_on(rets.index, rf)
    excess = rets - rf_m

    ann_ret = rets.mean() * 12
    ann_vol = rets.std() * SQRT12
    ann_excess = excess.mean() * 12
    excess_vol = excess.std() * SQRT12
    sharpe = ann_excess / excess_vol if excess_vol > 0 else np.nan
    downside_dev = np.sqrt(np.mean(np.minimum(excess.to_numpy(), 0.0) ** 2)) * SQRT12
    sortino = ann_excess / downside_dev if downside_dev > 0 else np.nan

    cum = np.exp(rets.cumsum())
    mdd = (cum / cum.cummax() - 1).min()
    calmar = ann_ret / abs(mdd) if mdd < 0 else np.nan
    win_rate = (rets > 0).mean()
    avg_win = rets[rets > 0].mean() if (rets > 0).any() else 0
    avg_loss = rets[rets < 0].mean() if (rets < 0).any() else 0

    alpha = beta = np.nan
    if bench_rets is not None:
        b = bench_rets.dropna()
        p_ex, b_ex = excess.align(b - _rf_on(b.index, rf), join="inner")
        if len(p_ex) > 12:
            cov = np.cov(p_ex, b_ex)
            if cov[1, 1] > 0:
                beta = cov[0, 1] / cov[1, 1]
                alpha = (p_ex.mean() - beta * b_ex.mean()) * 12
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe, sortino=sortino,
                mdd=mdd, calmar=calmar, win_rate=win_rate, avg_win=avg_win,
                avg_loss=avg_loss, alpha=alpha, beta=beta,
                ann_excess=ann_excess, downside_dev=downside_dev, rf_ann=rf_m.mean() * 12)


# ── Strategy backtest ──────────────────────────────────────────────────────────
def run_strategy_backtest(macro_df: pd.DataFrame, *, use_regime: bool, use_momentum: bool,
                          use_trend: bool, threshold: float, cost_bps: float,
                          allow_short: bool) -> pd.DataFrame:
    """Walk-forward, no-lookahead backtest of an SPY/cash (or SPY/-SPY) strategy.

    Position is decided from data observable at month T, then applied to the realised return
    from T to T+1 via .shift(1). The part of the book not in SPY (1 - w) earns the T-bill rate
    quoted at T. Net returns include linear transaction costs proportional to turnover.
    out.attrs["cash_rate"] says which cash rate was used."""
    out = macro_df.copy()
    out = out.dropna(subset=["sp500_ret_m"])

    # signals (each ∈ {0,1}, evaluated at month T)
    out["sig_regime"] = (out["regime"] != "Risk-Off 🔴").astype(int)
    out["sig_momentum"] = (out["momentum_12_1"] > 0).astype(int)
    sma_10m = out["sp500"].rolling(10).mean()
    out["sig_trend"] = (out["sp500"] > sma_10m).astype(int)

    cols = [c for c, on in (("sig_regime", use_regime), ("sig_momentum", use_momentum),
                            ("sig_trend", use_trend)) if on]
    out["sig_score"] = out[cols].mean(axis=1) if cols else 1.0

    in_market = (out["sig_score"] >= threshold).astype(float)
    if allow_short:
        target_w = in_market - (out["sig_score"] < (1 - threshold)).astype(float)
    else:
        target_w = in_market

    out["target_w"] = target_w
    out["position"] = out["target_w"].shift(1).fillna(0.0)        # NO-LOOKAHEAD
    out["turnover"] = out["position"].diff().abs().fillna(out["position"].abs())

    rf = rf_monthly(macro_df)
    out["rf_m"] = _rf_on(out.index, rf)
    out.attrs["cash_rate"] = "3-month T-bill (^IRX)" if rf is not None else "0% (T-bill data unavailable)"

    cost = (cost_bps / 10000.0) * out["turnover"]
    out["strat_ret_gross"] = out["position"] * out["sp500_ret_m"] + (1.0 - out["position"]) * out["rf_m"]
    out["strat_ret"] = out["strat_ret_gross"] - cost
    out["strat_equity"] = np.exp(out["strat_ret"].fillna(0.0).cumsum()) * 100
    out["bh_equity"] = np.exp(out["sp500_ret_m"].fillna(0.0).cumsum()) * 100
    out["strat_dd"] = out["strat_equity"] / out["strat_equity"].cummax() - 1
    out["bh_dd"] = out["bh_equity"] / out["bh_equity"].cummax() - 1
    return out


# ── Monte Carlo ────────────────────────────────────────────────────────────────
def run_monte_carlo(mu: float, vol: float, n: int, h: int = 12, *, method: str = MC_GAUSSIAN,
                    hist_rets: Optional[pd.Series] = None, block: int = 1, seed: int = 99) -> np.ndarray:
    """(n, h) array of cumulative return multipliers, one row per simulated path.

    gaussian:  i.i.d. normal monthly log returns with annual mean mu and volatility vol.
    bootstrap: monthly log returns resampled from hist_rets (mu and vol are ignored), in blocks of
               `block` consecutive months so short-range autocorrelation and volatility clustering
               survive; fat tails and skew come from the data instead of a normal assumption."""
    rng = np.random.default_rng(seed)
    if method == MC_GAUSSIAN:
        shocks = rng.normal(mu / 12, vol / SQRT12, (n, h))
    elif method == MC_BOOTSTRAP:
        r = np.asarray(pd.Series(hist_rets).dropna(), dtype=float) if hist_rets is not None else np.array([])
        block = max(1, int(block))
        if len(r) < max(24, block):
            raise ValueError("bootstrap needs at least 24 months of historical returns")
        n_blocks = -(-h // block)
        starts = rng.integers(0, len(r) - block + 1, size=(n, n_blocks))
        shocks = r[starts[:, :, None] + np.arange(block)].reshape(n, n_blocks * block)[:, :h]
    else:
        raise ValueError(f"unknown Monte Carlo method: {method!r}")
    return np.exp(np.cumsum(shocks, axis=1))
