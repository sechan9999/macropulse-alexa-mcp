"""
tests/test_forward_log.py
─────────────────────────────────────────────────────────────────
validation/forward_log.py and validation/score_forward_log.py: the hash chain catches edits, removals
and reordering; a record is built from the live tools' outputs without filling in failures; outcome
windows, late and primary records are scored as pre-registered. Offline; no market data.
"""
import datetime as dt
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from validation import forward_log as fl
from validation import score_forward_log as sc
from validation import spec

ROOT = Path(__file__).resolve().parents[1]


def _rec(i, **forecasts):
    return {"schema": 1, "created_at_utc": "2026-10-01T12:17:00Z", "as_of": "2026-09-30",
            "model_fingerprint": "f" * 64, "forecasts": forecasts or {"x": i}, "errors": {},
            "snapshots": {"folder": "none", "sha256": {}}}


def _chain(n):
    recs = []
    for i in range(n):
        recs.append(fl.chain(recs, _rec(i)))
    return recs


class TestChain(unittest.TestCase):
    def test_intact_chain_verifies(self):
        recs = _chain(4)
        self.assertEqual(fl.verify_chain(recs), [])
        self.assertEqual(recs[0]["prev_hash"], fl.GENESIS)
        self.assertEqual([r["seq"] for r in recs], [0, 1, 2, 3])

    def test_edited_record_is_caught(self):
        recs = _chain(3)
        recs[1]["forecasts"]["x"] = 99
        self.assertTrue(any("record 1: hash" in e for e in fl.verify_chain(recs)))

    def test_edit_with_recomputed_hash_breaks_the_next_link(self):
        recs = _chain(3)
        recs[1]["forecasts"]["x"] = 99
        recs[1]["hash"] = fl.record_hash(recs[1])
        self.assertTrue(any("record 2: prev_hash" in e for e in fl.verify_chain(recs)))

    def test_removed_or_reordered_records_are_caught(self):
        recs = _chain(3)
        self.assertNotEqual(fl.verify_chain([recs[0], recs[2]]), [])
        self.assertNotEqual(fl.verify_chain([recs[1], recs[0], recs[2]]), [])

    def test_append_only_against_the_base_branch(self):
        base = _chain(2)
        grown = list(base)
        grown.append(fl.chain(grown, _rec(2)))
        self.assertEqual(fl.verify_append_only(base, grown), [])
        self.assertNotEqual(fl.verify_append_only(base, base[:1]), [])
        rebuilt = _chain(1) + [fl.chain(_chain(1), _rec(7))]       # same length, different history
        self.assertNotEqual(fl.verify_append_only(base, rebuilt), [])

    def test_nan_is_stored_as_null_not_rejected(self):
        r = fl.chain([], _rec(0, er={"pred_std_log": float("nan"), "w": np.float64(0.5)}))
        self.assertIsNone(r["forecasts"]["er"]["pred_std_log"])
        self.assertEqual(json.loads(fl.canonical(r))["forecasts"]["er"]["w"], 0.5)

    def test_repository_log_is_intact(self):
        self.assertEqual(fl.cmd_verify(ROOT / spec.LOG_PATH), 0)


class _FakeServer:
    """The four MCP tools the forward log calls, returning what the live tools return."""

    def __init__(self, fail=()):
        self.fail = set(fail)

    def _maybe_fail(self, name):
        if name in self.fail:
            raise RuntimeError(f"{name} down")

    def execute_get_macro_regime(self):
        self._maybe_fail("regime")
        return {"status": "success", "regime": "Neutral 🟡", "stress_score": 0.12, "sp500": 6000.0, "vix": 16.0,
                "credit_spread_source": "FRED BAA-AAA", "regime_win_rate_pct": 61.0, "regime_months_in_history": 80}

    def execute_simulate_portfolio_risk(self, ticker, confidence, days, n_paths):
        self._maybe_fail("var")
        return {"var_pct": -6.1 if confidence == 95 else -9.0, "cvar_pct": -8.0, "current_price": 600.0}

    def execute_get_equity_report(self, ticker):
        self._maybe_fail(ticker)
        return {"price": 100.0, "as_of": "2026-10-15", "rating": "Hold", "score": 0.4,
                "fair_value_prob_weighted": 110.0, "upside_prob_weighted": 0.1, "dcf_base": 105.0,
                "scenarios": {"bear": {"per_share": 70.0, "prob": 0.25}}, "wacc": 0.09,
                "macro_regime": {"regime": "Neutral 🟡"}, "alexa_spoken_response": "not logged"}


def _macro(n=240, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2006-01-01", periods=n, freq="MS")
    d = pd.DataFrame(index=idx)
    d["sp500_ret_m"] = rng.normal(0.006, 0.04, n)
    d["sp500"] = 1200 * np.exp(d["sp500_ret_m"].cumsum())
    d["dgs10"] = 3 + np.cumsum(rng.normal(0, 0.1, n))
    d["credit_spread"] = 1 + np.abs(np.cumsum(rng.normal(0, 0.05, n)))
    d["yc_slope"] = np.cumsum(rng.normal(0, 0.1, n))
    d["realized_vol_12m"] = d["sp500_ret_m"].rolling(12).std() * np.sqrt(12)
    d["realized_vol_3m"] = d["sp500_ret_m"].rolling(3).std() * np.sqrt(12)
    d["momentum_12_1"] = d["sp500_ret_m"].rolling(11).sum().shift(1)
    d["tbill_3m"] = 4.0
    d["_obs_date"] = idx + pd.offsets.MonthEnd(0)
    d["_is_demo"] = False
    from src.macro_model import add_regime
    return add_regime(d)


class TestAppend(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.log = self.tmp / "validation" / "forward_log" / "log.jsonl"
        self.now = dt.datetime(2026, 10, 1, 12, 17, tzinfo=dt.timezone.utc)
        macro = _macro()
        self.spy = pd.DataFrame({"close": np.linspace(500, 600, 250)},
                                index=pd.bdate_range("2025-10-01", periods=250))
        self.sources = lambda srv: (srv, macro, macro, self.spy)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_record_holds_every_component_and_verifies(self):
        self.assertEqual(fl.cmd_append(self.log, now=self.now, sources=self.sources(_FakeServer())), 0)
        [r] = fl.read_log(self.log)
        f = r["forecasts"]
        self.assertEqual(f["regime"]["regime"], "Neutral 🟡")
        self.assertEqual(set(f["equity"]), set(spec.EQUITY_TICKERS))
        self.assertNotIn("alexa_spoken_response", f["equity"]["AVGO"])
        self.assertEqual(f["var"]["95"]["var_pct"], -6.1)
        self.assertIn(f["backtest_position"]["target_weight"], (0.0, 1.0))
        self.assertGreater(f["expected_return_12m"]["benchmark_n_months"], 200)
        self.assertEqual(r["as_of"], "2025-12-31")
        self.assertEqual(r["errors"], {})
        self.assertEqual(r["model_fingerprint"], fl.model_fingerprint())
        self.assertEqual(fl.cmd_verify(self.log), 0)

    def test_failed_components_are_reported_not_filled_in(self):
        fl.cmd_append(self.log, now=self.now, sources=self.sources(_FakeServer(fail={"regime", "AVGO"})))
        [r] = fl.read_log(self.log)
        self.assertNotIn("regime", r["forecasts"])
        self.assertNotIn("AVGO", r["forecasts"]["equity"])
        self.assertEqual(set(r["errors"]), {"regime", "equity:AVGO"})

    def test_same_as_of_is_not_logged_twice(self):
        fl.cmd_append(self.log, now=self.now, sources=self.sources(_FakeServer()))
        fl.cmd_append(self.log, now=self.now, sources=self.sources(_FakeServer()))
        self.assertEqual(len(fl.read_log(self.log)), 1)

    def test_demo_data_is_never_logged(self):
        srv, macro, _, spy = self.sources(_FakeServer())
        macro = macro.assign(_is_demo=True)
        self.assertEqual(fl.cmd_append(self.log, now=self.now, sources=(srv, macro, macro, spy)), 1)
        self.assertEqual(fl.read_log(self.log), [])

    def test_snapshot_tampering_is_caught(self):
        fl.cmd_append(self.log, now=self.now, sources=self.sources(_FakeServer()))
        [r] = fl.read_log(self.log)
        snap = Path(r["snapshots"]["folder"]) / "spy_1y_daily.csv.gz"
        fl.write_snapshot(snap.parent, "spy_1y_daily", self.spy * 1.01)
        self.assertEqual(fl.cmd_verify(self.log), 1)

    def test_snapshots_are_byte_identical_for_identical_data(self):
        a, b = self.tmp / "a", self.tmp / "b"
        self.assertEqual(fl.write_snapshot(a, "s", self.spy), fl.write_snapshot(b, "s", self.spy))
        self.assertEqual((a / "s.csv.gz").read_bytes(), (b / "s.csv.gz").read_bytes())


def _closes(start, n, step=0.001, seed=None):
    idx = pd.bdate_range(start, periods=n)
    r = np.full(n, step) if seed is None else np.random.default_rng(seed).normal(0, 0.01, n)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=idx)


def _scored_rec(as_of="2026-09-30", created="2026-10-01T12:17:00Z"):
    f = {"regime": {"regime": "Risk-Off 🔴", "stress_score": 1.2},
         "expected_return_12m": {"exp_simple_return": 0.05, "benchmark_prevailing_mean": 0.08},
         "backtest_position": {"applies_to_month": "2026-10", "target_weight": 1.0, "tbill_3m_pct": 0.0},
         "var": {"ticker": "SPY", "horizon_days": 21, "95": {"var_pct": -5.0}, "99": {"var_pct": -8.0}},
         "equity": {"AVGO": {"as_of": as_of, "upside_prob_weighted": -0.12, "rating": "Underweight"}}}
    r = _rec(0, **f)
    r.update(as_of=as_of, created_at_utc=created)
    return fl.chain([], r)


class TestScoring(unittest.TestCase):
    def test_windows_wait_for_the_outcome(self):
        px = _closes("2026-09-01", 30)
        p = lambda s: px
        rows = {r["component"]: r for r in sc.score_record(_scored_rec(), p)}
        self.assertTrue(all(r["status"] == "pending" for r in rows.values()))

    def test_scored_values(self):
        px = _closes("2026-09-01", 320)                          # +0.1% every session
        rows = {r["component"] + r["key"]: r for r in sc.score_record(_scored_rec(), lambda s: px)}
        self.assertTrue(rows["regimeRisk-Off 🔴"]["primary"])     # 2026-09-30 is the last session of September
        self.assertAlmostEqual(rows["regimeRisk-Off 🔴"]["next_return_log"], 21 * 0.001)
        self.assertAlmostEqual(rows["regimeRisk-Off 🔴"]["outcome"], 0.0, places=10)
        er = rows["expected_returnS&P 500 12m"]
        start, end = px.loc[:"2026-09-30"].iloc[-1], px.loc[:"2027-09-30"].iloc[-1]
        self.assertAlmostEqual(er["outcome"], end / start - 1)
        self.assertAlmostEqual(er["sq_err_model"], (er["outcome"] - 0.05) ** 2)
        self.assertAlmostEqual(er["sq_err_benchmark"], (er["outcome"] - 0.08) ** 2)
        oct_r = np.log(px.loc[:"2026-10-31"].iloc[-1] / px.loc[:"2026-09-30"].iloc[-1])
        self.assertAlmostEqual(rows["backtest_position2026-10"]["outcome"], oct_r)
        self.assertFalse(rows["var_95SPY"]["exceeded"])
        self.assertAlmostEqual(rows["equityAVGO"]["outcome"], 0.0)  # same series as SPY

    def test_var_exceedance(self):
        px = _closes("2026-09-01", 60)
        px.loc[px.index > "2026-09-30"] *= 0.9                    # a 10% drop right after the as-of close
        rows = {r["component"]: r for r in sc.score_record(_scored_rec(), lambda s: px)}
        self.assertTrue(rows["var_95"]["exceeded"])
        self.assertTrue(rows["var_99"]["exceeded"])

    def test_record_written_after_the_open_is_not_scored(self):
        px = _closes("2026-09-01", 320)
        rows = sc.score_record(_scored_rec(created="2026-10-01T14:00:00Z"), lambda s: px)
        self.assertEqual([r["status"] for r in rows], ["late (written after the next open)"])

    def test_mid_month_record_is_not_primary(self):
        px = _closes("2026-09-01", 320)
        rows = sc.score_record(_scored_rec(as_of="2026-09-15", created="2026-09-16T01:00:00Z"), lambda s: px)
        self.assertTrue(all(r["primary"] is False for r in rows))

    def test_summary_runs_no_test_below_the_minimum(self):
        px = _closes("2026-09-01", 320)
        scores = pd.DataFrame(sc.score_record(_scored_rec(), lambda s: px))
        text = sc.summarize(scores, 1)
        self.assertIn("collecting", text)
        self.assertNotIn("test due", text)
        self.assertIn("var_95: 0 of 1", text)

    def test_cli_on_an_empty_log(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            self.assertEqual(sc.main(root=tmp, prices=lambda s: pd.Series(dtype=float)), 0)
            self.assertIn("No forecasts yet", (tmp / spec.SUMMARY_PATH).read_text())
        finally:
            shutil.rmtree(tmp)


class TestPreregistrationPins(unittest.TestCase):
    """The constants in validation/spec.py are the ones the pre-registration document states."""

    doc = (ROOT / spec.PREREGISTRATION_DOC).read_text(encoding="utf-8")

    def test_constants_match_the_document(self):
        rows = {
            "Equity tickers": ", ".join(spec.EQUITY_TICKERS),
            "VaR ticker, confidences, horizon": f"{spec.VAR_TICKER}, 95 and 99, {spec.VAR_HORIZON_DAYS} trading days",
            "VaR paths": str(spec.VAR_PATHS),
            "Regime outcome window": f"{spec.REGIME_HORIZON_DAYS} trading days",
            "Expected-return and equity horizon": f"{spec.EXPECTED_RETURN_HORIZON_MONTHS} months",
            "Outcome index; total-return benchmark": f"{spec.OUTCOME_INDEX}; {spec.TOTAL_RETURN_BENCHMARK}",
            "Minimum primary records: regime": f"{spec.MIN_N['regime']}, with at least {spec.MIN_PER_REGIME_CLASS} "
                                               f"Risk-On and {spec.MIN_PER_REGIME_CLASS} Risk-Off",
            "Minimum primary records: expected return": str(spec.MIN_N["expected_return"]),
            "Minimum primary records: VaR 95%": str(spec.MIN_N["var_95"]),
            "Minimum primary records: VaR 99%": str(spec.MIN_N["var_99"]),
            "Minimum primary records: strategy position": str(spec.MIN_N["backtest_position"]),
        }
        self.assertEqual(spec.VAR_CONFIDENCES, (95, 99))
        self.assertEqual(spec.EQUITY_HORIZON_MONTHS, spec.EXPECTED_RETURN_HORIZON_MONTHS)
        for name, value in rows.items():
            self.assertIn(f"| {name} | {value} |", self.doc)

    def test_model_files_exist(self):
        for f in spec.MODEL_FILES:
            self.assertTrue((ROOT / f).is_file(), f)


if __name__ == "__main__":
    unittest.main()
