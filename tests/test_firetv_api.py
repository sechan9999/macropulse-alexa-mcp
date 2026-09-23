"""
tests/test_firetv_api.py
─────────────────────────────────────────────────────────────────
Fire TV companion API (src/firetv_api.py) and its two hosts — the deployed MCP server's /api/*
routes and rest_server.py — offline, with the data sources patched.
"""
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from starlette.testclient import TestClient

from src import firetv_api
from src.firetv_api import FireTvDataUnavailable

APP_REGIME_KEYS = {"as_of", "regime", "regime_score", "sp500", "sp500_mom_pct", "treasury_10y_pct",
                   "yield_curve_slope_pct", "credit_spread_pct", "realized_vol_12m_pct"}   # api/macroPulseApi.ts


def _macro(demo=False):
    idx = pd.date_range("2025-01-31", periods=24, freq="ME")
    df = pd.DataFrame({"sp500": np.linspace(5000, 6000, 24), "dgs10": 4.2, "yc_slope": 0.005,
                       "credit_spread": 0.012, "realized_vol_12m": 0.15, "regime_score": -0.8,
                       "regime": "Risk-On 🟢"}, index=idx)
    df["_is_demo"] = demo
    return df


def _scan(rows):
    return pd.DataFrame([{"Ticker": t, "Price": p, "Signal": s, "Score": sc, "_error": e}
                         for t, p, s, sc, e in rows])


class TestPayloads(unittest.TestCase):
    def setUp(self):
        self.scan_patch = mock.patch.object(firetv_api, "run_quant_scan")
        self.run_scan = self.scan_patch.start()

    def tearDown(self):
        self.scan_patch.stop()

    def test_regime_matches_app_contract(self):
        with mock.patch.object(firetv_api, "load_macro", return_value=_macro()):
            r = firetv_api.regime_payload()
        self.assertEqual(set(r), APP_REGIME_KEYS)
        self.assertEqual(r["treasury_10y_pct"], 4.2)
        self.assertEqual(r["yield_curve_slope_pct"], 0.5)

    def test_demo_frame_is_never_served_as_live_data(self):
        with mock.patch.object(firetv_api, "load_macro", return_value=_macro(demo=True)):
            with self.assertRaises(FireTvDataUnavailable):
                firetv_api.regime_payload()
        with mock.patch.object(firetv_api, "load_macro", side_effect=OSError("down")):
            with self.assertRaises(FireTvDataUnavailable):
                firetv_api.regime_payload()

    def test_watchlist_explicit_tickers_and_top_n(self):
        self.run_scan.return_value = _scan([("AAA", 10.0, "BUY", 30, None), ("BBB", 5.0, "SELL", -20, None),
                                            ("BAD", 0.0, "HOLD", 0, "no data")])
        r = firetv_api.watchlist_payload(tickers="aaa, bbb,bad", top_n=1)
        self.assertEqual(r["universe"], ["AAA", "BBB", "BAD"])
        self.assertEqual(r["signals"], [{"ticker": "AAA", "price": 10.0, "signal": "BUY", "score": 30}])
        self.assertEqual(len(firetv_api.watchlist_payload(tickers="aaa,bbb,bad", top_n=99)["signals"]), 2)

    def test_failed_scan_is_503_and_not_cached(self):
        self.run_scan.return_value = _scan([("ZZZ", 0.0, "HOLD", 0, "no data")])
        with self.assertRaises(FireTvDataUnavailable):
            firetv_api.watchlist_payload(tickers="ZZZ")
        self.run_scan.return_value = _scan([("ZZZ", 1.0, "BUY", 20, None)])
        self.assertEqual(firetv_api.watchlist_payload(tickers="ZZZ")["signals"][0]["ticker"], "ZZZ")

    def test_default_universe_without_firestore(self):
        self.run_scan.return_value = _scan([("SPY", 500.0, "BUY", 40, None)])
        with mock.patch.object(firetv_api, "get_watchlist", return_value=[]):
            r = firetv_api.watchlist_payload(user_id="someone")
        self.assertEqual(tuple(r["universe"]), firetv_api.DEFAULT_UNIVERSE)

    def test_nvda_error_is_503(self):
        with mock.patch.object(firetv_api, "summarize_nvda_danger_zone", return_value={"error": "no_data"}):
            with self.assertRaises(FireTvDataUnavailable):
                firetv_api.nvda_payload()


class TestMcpServerRoutes(unittest.TestCase):
    """The public ECS deployment is what the Fire TV app talks to by default."""

    @classmethod
    def setUpClass(cls):
        import src.alexa_mcp_server as srv
        cls.client_cm = TestClient(srv.build_starlette_app(warm_up=False))
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)

    def test_routes(self):
        regime = {k: 1 for k in APP_REGIME_KEYS}
        with mock.patch.object(firetv_api, "regime_payload", return_value=regime), \
                mock.patch.object(firetv_api, "watchlist_payload", return_value={"universe": [], "signals": []}) as wl, \
                mock.patch.object(firetv_api, "nvda_payload", side_effect=FireTvDataUnavailable("NVDA down")):
            self.assertEqual(self.client.get("/api/health").json(), {"status": "ok"})
            r = self.client.get("/api/regime", headers={"Origin": "http://tv"})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json(), regime)
            self.assertEqual(r.headers.get("access-control-allow-origin"), "*")
            self.assertEqual(self.client.get("/api/watchlist-signals?top_n=3&user_id=u1").status_code, 200)
            wl.assert_called_once_with("u1", "", 3)
            self.assertEqual(self.client.get("/api/watchlist-signals?top_n=x").status_code, 422)
            n = self.client.get("/api/nvda-danger")
            self.assertEqual((n.status_code, n.json()), (503, {"detail": "NVDA down"}))

    def test_mcp_still_served(self):
        r = self.client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                             headers={"Accept": "application/json, text/event-stream"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("get_macro_regime", r.text)


class TestRestServer(unittest.TestCase):
    def test_rest_server_uses_shared_payloads(self):
        import rest_server
        client = TestClient(rest_server.app)
        with mock.patch.object(firetv_api, "regime_payload", side_effect=FireTvDataUnavailable("down")):
            r = client.get("/api/regime")
        self.assertEqual((r.status_code, r.json()), (503, {"detail": "down"}))
        self.assertEqual(client.get("/api/watchlist-signals?top_n=0").status_code, 422)


if __name__ == "__main__":
    unittest.main()
