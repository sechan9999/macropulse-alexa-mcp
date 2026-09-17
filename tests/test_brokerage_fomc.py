"""
tests/test_brokerage_fomc.py
─────────────────────────────────────────────────────────────────
Unit tests for Brokerage Sync and FOMC Cross-Asset VaR Shock Engine.
"""
import unittest
from src.brokerage_sync import BrokeragePortfolio, PRESET_PORTFOLIOS


class TestBrokerageFOMC(unittest.TestCase):

    def test_preset_portfolio_loading(self):
        port = BrokeragePortfolio.from_preset("Macro Balanced (60/40 Modern)", 1_000_000.0)
        self.assertEqual(port.total_value_usd, 1_000_000.0)
        self.assertAlmostEqual(sum(port.weights.values()), 1.0, places=4)
        alloc = port.get_allocation_by_asset_class()
        self.assertIn("Equities", alloc)
        self.assertIn("Fixed Income", alloc)
        self.assertIn("Commodities", alloc)

    def test_alpaca_read_only_sync(self):
        sample_alpaca = [
            {"symbol": "SPY", "market_value": "400000.0"},
            {"symbol": "TLT", "market_value": "300000.0"},
            {"symbol": "GLD", "market_value": "200000.0"},
            {"symbol": "BTC-USD", "market_value": "100000.0"},
        ]
        port = BrokeragePortfolio.from_alpaca_json(sample_alpaca)
        self.assertEqual(port.total_value_usd, 1_000_000.0)
        self.assertAlmostEqual(port.weights["SPY"], 0.40, places=2)

    def test_cross_asset_var(self):
        port = BrokeragePortfolio.from_preset("Macro Balanced (60/40 Modern)", 1_000_000.0)
        var_res = port.compute_cross_asset_var(confidence=95, horizon_days=30)
        self.assertIn("var_pct", var_res)
        self.assertIn("cvar_pct", var_res)
        self.assertTrue(var_res["var_pct"] < 0)
        self.assertTrue(var_res["cvar_pct"] <= var_res["var_pct"])  # CVaR is worse than VaR
        self.assertTrue(var_res["annualized_volatility_pct"] > 5.0)

    def test_fomc_shock_scenarios(self):
        port = BrokeragePortfolio.from_preset("Macro Balanced (60/40 Modern)", 1_000_000.0)
        
        # Hawkish 50bps shock
        hawk = port.simulate_fomc_shock("hawkish_50bps")
        self.assertEqual(hawk["status"], "success")
        self.assertTrue(hawk["total_pnl_dollar"] < 0)  # Rate hike hurts balanced equity/bond
        self.assertIn("alexa_spoken_response", hawk)
        
        # Dovish 50bps shock
        dov = port.simulate_fomc_shock("dovish_50bps")
        self.assertEqual(dov["status"], "success")
        self.assertTrue(dov["total_pnl_dollar"] > 0)   # Easing lifts equity/bond
        
        # Stagflation shock
        stag = port.simulate_fomc_shock("stagflation_inversion")
        self.assertEqual(stag["status"], "success")
        self.assertIn("breakdown_by_ticker", stag)


if __name__ == "__main__":
    unittest.main()
