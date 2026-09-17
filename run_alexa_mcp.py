"""
run_alexa_mcp.py
─────────────────────────────────────────────────────────────────
Standalone CLI launcher and verification runner for MacroPulse
Alexa+ MCP Server with Streamable HTTP Transport.

Usage:
  python run_alexa_mcp.py              # Launch the MCP server on port 8000
  python run_alexa_mcp.py --test-tools # Test all 6 quantitative tools locally
  python run_alexa_mcp.py --port 8080  # Launch on custom port
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import argparse
import uvicorn
from src.alexa_mcp_server import (
    build_starlette_app,
    execute_get_macro_regime,
    execute_get_rates_and_spreads,
    execute_simulate_portfolio_risk,
    execute_check_nvda_danger_zone,
    execute_scan_quant_signals,
    execute_get_expected_returns,
    execute_simulate_fomc_shock,
)
from src.ambient_audio import generate_chime_wav


def run_tests():
    print("=========================================================")
    print("🔍 Testing MacroPulse Alexa+ Quantitative MCP Tools...")
    print("=========================================================\n")
    
    tools = [
        ("1. Macro Regime Engine", execute_get_macro_regime),
        ("2. Rates & Spreads Engine", execute_get_rates_and_spreads),
        ("3. Monte Carlo VaR Simulation", lambda: execute_simulate_portfolio_risk("SPY", 95, 30, 2000)),
        ("4. NVDA Danger Zone Engine", execute_check_nvda_danger_zone),
        ("5. Quant Volatility Signals", lambda: execute_scan_quant_signals("SPY")),
        ("6. Ridge Expected Returns", execute_get_expected_returns),
        ("7. FOMC Cross-Asset Shock Test", lambda: execute_simulate_fomc_shock("hawkish_50bps")),
    ]
    
    passed = 0
    for name, func in tools:
        try:
            res = func()
            assert res["status"] == "success", f"Status not success: {res}"
            assert "alexa_spoken_response" in res, "Missing alexa_spoken_response"
            print(f"✅ {name}: PASSED")
            print(f"   🗣️ Spoken Voice Output: \"{res['alexa_spoken_response']}\"\n")
            passed += 1
        except Exception as e:
            print(f"❌ {name}: FAILED ({e})\n")

    # Verify ambient audio engine
    try:
        wav = generate_chime_wav("bollinger_breakout", duration_sec=0.5)
        assert len(wav) > 44
        print("✅ 8. Ambient Audio Chime Engine: PASSED (Synthesized WAV successfully)\n")
        passed += 1
    except Exception as e:
        print(f"❌ 8. Ambient Audio Engine: FAILED ({e})\n")
        
    total_checks = len(tools) + 1
    print(f"Summary: {passed}/{total_checks} tools passed successfully.")
    if passed == total_checks:
        print("🎉 All tools verified for Alexa+ MCP & Fire TV deployment!")
    else:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="MacroPulse Alexa+ MCP Server Launcher")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--test-tools", action="store_true", help="Run local tests on all tools and exit")
    args = parser.parse_args()

    if args.test_tools:
        run_tests()
        return

    print("===================================================================")
    print("⚡ MacroPulse Alexa+ Model Context Protocol (MCP) Server")
    print(f"🌐 Streamable HTTP Transport: http://{args.host}:{args.port}/mcp")
    print(f"📡 Server-Sent Events (SSE):   http://{args.host}:{args.port}/sse")
    print(f"🩺 Health Check Endpoint:     http://{args.host}:{args.port}/health")
    print(f"🎙️ Alexa Query Simulator:     http://{args.host}:{args.port}/alexa/query")
    print("===================================================================\n")
    
    app = build_starlette_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
