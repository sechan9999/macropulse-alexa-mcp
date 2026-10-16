"""
validation/stats.py
─────────────────────────────────────────────────────────────────
Statistical tests named in docs/validation/preregistration.md. Each function states its formula; the
tests in tests/test_validation_stats.py check them against independent calculations and known tables.

  oos_r2                 Campbell-Thompson (2008):  1 - Σ(y - ŷ_model)² / Σ(y - ŷ_bench)²
  clark_west             Clark-West (2007) MSPE-adjusted test, Newey-West HAC t-statistic (one-sided)
  hac_mean_test          mean of a series with Newey-West standard error
  kupiec_pof             Kupiec (1995) proportion-of-failures likelihood ratio, χ²(1)
  christoffersen         Christoffersen (1998) independence and conditional-coverage LRs
  basel_zone             Basel Committee (1996) traffic light by binomial cumulative probability
  acerbi_szekely_z2      Acerbi-Székely (2014) Z2 expected-shortfall test statistic
  probabilistic_sharpe   Bailey-López de Prado (2012) PSR
  expected_max_sharpe    expected maximum Sharpe ratio of N trials (False Strategy Theorem)
  deflated_sharpe        Bailey-López de Prado (2014) DSR = PSR against the expected maximum
  pbo_cscv               Bailey, Borwein, López de Prado, Zhu (2017) probability of backtest overfitting
  holm                   Holm (1979) step-down adjusted p-values
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


# ── forecast evaluation ───────────────────────────────────────────────────────
def oos_r2(y, model, bench) -> float:
    y, model, bench = (np.asarray(a, float) for a in (y, model, bench))
    return float(1.0 - np.sum((y - model) ** 2) / np.sum((y - bench) ** 2))


def hac_mean_test(x, lags: int) -> dict:
    """Mean of x, its Newey-West (Bartlett) standard error with `lags` lags, t and two-sided p."""
    import statsmodels.api as sm
    x = np.asarray(x, float)
    res = sm.OLS(x, np.ones_like(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    t = float(res.tvalues[0])
    return {"mean": float(res.params[0]), "se": float(res.bse[0]), "t": t, "p_two_sided": float(res.pvalues[0]),
            "n": int(len(x)), "lags": lags}


def clark_west(y, model, bench, lags: int = 12) -> dict:
    """f_t = (y - b)² - [(y - m)² - (b - m)²]; H0: E[f] <= 0 (the model does not beat the benchmark).
    One-sided p from the HAC t-statistic of mean(f) against N(0,1), as in Clark and West (2007)."""
    y, m, b = (np.asarray(a, float) for a in (y, model, bench))
    f = (y - b) ** 2 - ((y - m) ** 2 - (b - m) ** 2)
    r = hac_mean_test(f, lags)
    return {"stat": r["t"], "p_one_sided": float(1 - stats.norm.cdf(r["t"])), "mean_f": r["mean"], "n": r["n"],
            "lags": lags}


# ── VaR backtests ─────────────────────────────────────────────────────────────
def _xlogy(x, y):
    return 0.0 if x == 0 else x * np.log(y)


def kupiec_pof(n: int, x: int, p: float) -> dict:
    """LR = -2 ln[(1-p)^(n-x) p^x / ((1-x/n)^(n-x) (x/n)^x)] ~ χ²(1) under H0: exceedance rate = p."""
    phat = x / n
    ll0 = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    ll1 = _xlogy(n - x, 1 - phat) + _xlogy(x, phat)
    lr = -2.0 * (ll0 - ll1)
    return {"n": n, "exceedances": x, "expected": n * p, "rate": phat, "lr": float(lr),
            "p_value": float(1 - stats.chi2.cdf(lr, 1))}


def christoffersen(hits: Sequence[bool], p: float) -> dict:
    """Independence: first-order Markov chain of exceedances vs i.i.d. Bernoulli(π), LR_ind ~ χ²(1).
    Conditional coverage LR_cc = LR_pof + LR_ind ~ χ²(2)."""
    h = np.asarray(hits, int)
    a, b = h[:-1], h[1:]
    n00, n01 = int(np.sum((a == 0) & (b == 0))), int(np.sum((a == 0) & (b == 1)))
    n10, n11 = int(np.sum((a == 1) & (b == 0))), int(np.sum((a == 1) & (b == 1)))
    pi0 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi1 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_ind = (_xlogy(n00, 1 - pi0) + _xlogy(n01, pi0) + _xlogy(n10, 1 - pi1) + _xlogy(n11, pi1))
    ll_null = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    lr_ind = -2.0 * (ll_null - ll_ind)
    pof = kupiec_pof(len(h), int(h.sum()), p)
    lr_cc = pof["lr"] + lr_ind
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11, "lr_ind": float(lr_ind),
            "p_ind": float(1 - stats.chi2.cdf(lr_ind, 1)), "lr_cc": float(lr_cc),
            "p_cc": float(1 - stats.chi2.cdf(lr_cc, 2))}


def basel_zone(n: int, x: int, p: float = 0.01) -> dict:
    """Green while P(X <= x) < 95%, yellow while < 99.99%, red otherwise (X ~ Binomial(n, p)).
    With n = 250 and p = 1% this gives the Basel table: green 0-4, yellow 5-9, red 10+."""
    c = float(stats.binom.cdf(x, n, p))
    zone = "green" if c < 0.95 else "yellow" if c < 0.9999 else "red"
    return {"n": n, "exceedances": x, "cumulative_probability": c, "zone": zone}


def acerbi_szekely_z2(losses, var, es, alpha: float) -> float:
    """Z2 = 1 - (1 / (T α)) Σ_t L_t 1{L_t > VaR_t} / ES_t, with losses, VaR and ES as positive numbers.
    E[Z2] = 0 when ES is correct; Z2 < 0 means realised tail losses exceed the forecast ES."""
    L, v, e = (np.asarray(a, float) for a in (losses, var, es))
    return float(1.0 - np.sum(L * (L > v) / e) / (len(L) * alpha))


# ── backtest overfitting ──────────────────────────────────────────────────────
def probabilistic_sharpe(sr: float, sr0: float, t: int, skew: float, kurt: float) -> float:
    """PSR = Φ[(SR - SR0) √(T-1) / √(1 - γ3 SR + (γ4 - 1)/4 SR²)], per-period SR, γ4 = raw kurtosis."""
    den = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(t - 1) / den))


def expected_max_sharpe(var_sr: float, n_trials: int) -> float:
    """E[max SR] ≈ √V[SR] ((1-γ) Φ⁻¹(1 - 1/N) + γ Φ⁻¹(1 - 1/(N e))), γ = Euler-Mascheroni; 0 for N = 1."""
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(returns, trial_srs: Sequence[float]) -> dict:
    """DSR of `returns` (per-period) given the per-period Sharpe ratios of every trial, this one included."""
    r = np.asarray(returns, float)
    sr = r.mean() / r.std(ddof=1)
    skew, kurt = float(stats.skew(r)), float(stats.kurtosis(r, fisher=False))
    n = len(trial_srs)
    var_sr = float(np.var(trial_srs, ddof=1)) if n > 1 else 0.0
    sr0 = expected_max_sharpe(var_sr, n)
    return {"sr": float(sr), "sr_annual": float(sr * np.sqrt(12)), "sr0": sr0, "sr0_annual": float(sr0 * np.sqrt(12)),
            "n_trials": n, "var_trial_sr": var_sr, "skew": skew, "kurtosis": kurt, "t": int(len(r)),
            "dsr": probabilistic_sharpe(sr, sr0, len(r), skew, kurt),
            "psr_vs_zero": probabilistic_sharpe(sr, 0.0, len(r), skew, kurt)}


def _sharpe_cols(m: np.ndarray) -> np.ndarray:
    sd = m.std(axis=0, ddof=1)
    return np.where(sd > 0, m.mean(axis=0) / np.where(sd > 0, sd, 1), -np.inf)


def pbo_cscv(returns: pd.DataFrame, n_blocks: int = 16) -> dict:
    """Split T rows into S blocks; for every half/half split pick the in-sample best column by Sharpe,
    find its relative rank ω among the out-of-sample Sharpes, λ = ln(ω / (1 - ω)). PBO = share λ <= 0."""
    m = returns.dropna().to_numpy(float)
    t, n = m.shape
    blocks = np.array_split(np.arange(t - t % n_blocks), n_blocks)
    lambdas, ids = [], list(range(n_blocks))
    for is_ids in combinations(ids, n_blocks // 2):
        is_rows = np.concatenate([blocks[i] for i in is_ids])
        oos_rows = np.concatenate([blocks[i] for i in ids if i not in is_ids])
        best = int(np.argmax(_sharpe_cols(m[is_rows])))
        oos = _sharpe_cols(m[oos_rows])
        rank = stats.rankdata(oos)[best]                       # 1 = worst ... n = best
        w = rank / (n + 1)
        lambdas.append(np.log(w / (1 - w)))
    lam = np.asarray(lambdas)
    return {"pbo": float(np.mean(lam <= 0)), "n_splits": int(len(lam)), "n_blocks": n_blocks, "n_strategies": n,
            "median_logit": float(np.median(lam))}


# ── multiple testing ──────────────────────────────────────────────────────────
def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm step-down: sort ascending, adjusted p_(i) = max_{j<=i} min(1, (m - j + 1) p_(j))."""
    items = sorted(((k, v) for k, v in pvals.items() if v is not None and np.isfinite(v)), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out
