"""
validation/stage4.py — backtest overfitting (pre-registration H3)
─────────────────────────────────────────────────────────────────
  Strategy under test: the dashboard default (regime + 12-1 momentum + 10-month trend, threshold 0.5,
  long-only, 5 bp per unit of turnover, cash at the 3-month T-bill), src/portfolio.run_strategy_backtest.

  Primary H3: Deflated Sharpe Ratio (Bailey & López de Prado 2014) of its monthly excess return, with
  the variance of Sharpe ratios across the 32 distinct rules a user can reach in the Backtest tab
  (pre-registration, trial ledger) and N = 32. Pass: DSR >= 0.95.

  Also: probability of backtest overfitting by CSCV (16 blocks) over the 32 rules; Hansen's SPA test
  of the strategy against buy-and-hold, the 10-month trend rule alone and a monthly-rebalanced 60/40
  (arch.bootstrap.SPA, stationary bootstrap, block 6, 10,000 draws); net results at 0/5/10/25 bp;
  execution one session and one month after the signal.
  Returns are the product's: S&P 500 price index (^GSPC, no dividends) and the T-bill cash leg.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from src.portfolio import compute_hf_metrics, run_strategy_backtest
from validation import data, frames, stats
from validation.results import save_csv, save_json

STAGE = "stage4"
SIGNALS = ("regime", "momentum", "trend")
DEFAULT = dict(use_regime=True, use_momentum=True, use_trend=True, threshold=0.5, allow_short=False)


def _signature(r: dict) -> tuple:
    sub = tuple(sg for sg in SIGNALS if r[f"use_{sg}"])
    levels = [i / len(sub) for i in range(len(sub) + 1)]
    t = r["threshold"]
    return (sub, tuple(l >= t for l in levels), tuple(l < 1 - t for l in levels) if r["allow_short"] else ())


def distinct_rules() -> list:
    """The 32 decision rules reachable from the Backtest tab (signal subset x threshold 0.34-1.00 x short),
    one representative per distinct set of positions; the dashboard default 0.5 is preferred."""
    seen, rules = set(), []
    thr = [0.5] + [float(t) for t in np.round(np.arange(0.34, 1.0001, 0.01), 2) if abs(t - 0.5) > 1e-9]
    for k in (1, 2, 3):
        for sub in combinations(SIGNALS, k):
            for short in (False, True):
                for t in thr:
                    r = {"use_regime": "regime" in sub, "use_momentum": "momentum" in sub,
                         "use_trend": "trend" in sub, "threshold": t, "allow_short": short}
                    sig = _signature(r)
                    if sig not in seen:
                        seen.add(sig)
                        rules.append(r)
    return rules


def rule_name(r: dict) -> str:
    """Name of the representative rule that takes the same positions as r."""
    sig = _signature(r)
    rep = next(x for x in distinct_rules() if _signature(x) == sig)
    sub = "+".join(sg for sg in SIGNALS if rep[f"use_{sg}"])
    return f"{sub}|t={rep['threshold']:.2f}|{'ls' if rep['allow_short'] else 'lo'}"


def excess(bt: pd.DataFrame) -> pd.Series:
    return (bt["strat_ret"] - bt["rf_m"]).iloc[1:]


def rule_returns(m: pd.DataFrame, cost_bps: float = 5.0) -> pd.DataFrame:
    cols = {}
    for r in distinct_rules():
        cols[rule_name(r)] = excess(run_strategy_backtest(m, cost_bps=cost_bps, **r))
    return pd.DataFrame(cols)


def default_name() -> str:
    return rule_name({**DEFAULT})


def sixty_forty(m: pd.DataFrame) -> pd.Series:
    bond = data.monthly_last(data.yahoo("VBMFX")).pct_change()
    eq = np.expm1(m["sp500_ret_m"])
    mix = np.log1p(0.6 * eq + 0.4 * bond).dropna()
    return mix


def spa(strategy: pd.Series, benchmark: pd.Series) -> dict:
    from arch.bootstrap import SPA
    both = pd.concat([strategy.rename("s"), benchmark.rename("b")], axis=1).dropna()
    t = SPA(-both["b"].to_numpy(), -both[["s"]].to_numpy(), block_size=6, reps=10000, bootstrap="stationary",
            studentize=True, seed=0)
    t.compute()
    pv = t.pvalues
    return {"n": len(both), "first": both.index.min(), "p_consistent": float(pv["consistent"]),
            "p_lower": float(pv["lower"]), "p_upper": float(pv["upper"]),
            "mean_excess_vs_benchmark_ann": float((both["s"] - both["b"]).mean() * 12)}


def cost_table(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in (0, 5, 10, 25):
        bt = run_strategy_backtest(m, cost_bps=c, **DEFAULT)
        r = bt["strat_ret"].iloc[1:]
        k = compute_hf_metrics(r, bt["sp500_ret_m"].iloc[1:], rf=bt["rf_m"])
        rows.append({"cost_bps": c, "ann_return": k["ann_ret"], "ann_excess": k["ann_excess"], "sharpe": k["sharpe"],
                     "sortino": k["sortino"], "max_drawdown": k["mdd"],
                     "turnover_per_year": float(bt["turnover"].iloc[1:].mean() * 12)})
    return pd.DataFrame(rows).set_index("cost_bps")


def delayed(m: pd.DataFrame) -> pd.DataFrame:
    """Default rule executed at the close one session after the month end, or one month late."""
    bt = run_strategy_backtest(m, cost_bps=5.0, **DEFAULT)
    px = data.yahoo("GSPC")
    lr = np.log(px).diff()
    first = lr.groupby(lr.index.to_period("M")).apply(lambda s: s.dropna().iloc[0] if len(s.dropna()) else np.nan)
    first.index = first.index.to_timestamp()
    d1 = first.reindex(bt.index)
    pos, prev = bt["position"], bt["position"].shift(1).fillna(0.0)
    r, rf = bt["sp500_ret_m"], bt["rf_m"]
    cost = 5.0 / 1e4 * bt["turnover"]
    one_day = prev * d1 + pos * (r - d1) + (1 - pos) * rf - cost
    one_month = prev * r + (1 - prev) * rf - 5.0 / 1e4 * prev.diff().abs().fillna(prev.abs())
    rows = []
    for name, ret in (("same close (product)", bt["strat_ret"]), ("one session later", one_day),
                      ("one month later", one_month)):
        ret = ret.iloc[1:].dropna()
        k = compute_hf_metrics(ret, rf=rf.reindex(ret.index))
        rows.append({"execution": name, "ann_return": k["ann_ret"], "sharpe": k["sharpe"], "max_drawdown": k["mdd"],
                     "months": len(ret)})
    return pd.DataFrame(rows).set_index("execution")


def run() -> dict:
    out = {}
    for label, start in (("full", frames.FIRST_MONTH["backtest"]), ("product_2005", frames.PRODUCT_START)):
        m = frames.build_macro(start)
        R = rule_returns(m).dropna()
        if label == "full":
            save_csv(STAGE, "rule_excess_returns", R)
        srs = (R.mean() / R.std(ddof=1)).to_numpy()
        dname = default_name()
        res = {"months": len(R), "first": R.index.min(), "last": R.index.max(), "n_rules": R.shape[1],
               "dsr": stats.deflated_sharpe(R[dname], srs), "pbo": stats.pbo_cscv(R, 16),
               "best_rule_in_sample": str(R.columns[int(np.argmax(srs))]),
               "default_rank_of_32_by_sharpe": int((srs > srs[list(R.columns).index(dname)]).sum() + 1)}
        bh = (m["sp500_ret_m"] - run_strategy_backtest(m, cost_bps=5, **DEFAULT)["rf_m"]).iloc[1:]
        trend = R[rule_name({"use_regime": False, "use_momentum": False, "use_trend": True, "threshold": 0.34,
                             "allow_short": False})]
        rf = run_strategy_backtest(m, cost_bps=5, **DEFAULT)["rf_m"]
        res["spa"] = {"vs_buy_and_hold": spa(R[dname], bh), "vs_10m_trend": spa(R[dname], trend),
                      "vs_60_40": spa(R[dname], (sixty_forty(m) - rf).dropna())}
        out[label] = res
        if label == "full":
            ct = cost_table(m)
            save_csv(STAGE, "cost_sensitivity", ct)
            out["cost_sensitivity"] = ct.to_dict(orient="index")
            dl = delayed(m)
            save_csv(STAGE, "execution_delay", dl)
            out["execution_delay"] = dl.to_dict(orient="index")
    out["rules"] = [rule_name(r) for r in distinct_rules()]
    save_json(STAGE, "summary", out)
    return out
