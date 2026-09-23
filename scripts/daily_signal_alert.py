#!/usr/bin/env python3
"""
scripts/daily_signal_alert.py
─────────────────────────────────────────────────────────────────
Headless (no Streamlit) runner for the Quant Signals engine —
meant to be invoked by a scheduled job (cron / GitHub Actions /
a Claude scheduled task) to push a daily pre-market summary.

Usage:
    python scripts/daily_signal_alert.py
    python scripts/daily_signal_alert.py --tickers NVDA,AAPL,SPY --period 1y --top 5

Exit code is 0 on a successful scan (even if every ticker HOLDs),
non-zero only if the scan itself errored out.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/daily_signal_alert.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.quant_signals import DEFAULT_UNIVERSE, format_alert_text, run_quant_scan


def main() -> int:
    ap = argparse.ArgumentParser(description="Macropulse daily quant-signal alert")
    ap.add_argument("--tickers", type=str, default=None,
                     help="Comma-separated tickers (default: same watchlist as the Screener tab)")
    ap.add_argument("--period", type=str, default="1y", choices=["6mo", "1y", "2y"])
    ap.add_argument("--top", type=int, default=5, help="Max names per BUY/SELL section")
    args = ap.parse_args()

    tickers = (tuple(t.strip().upper() for t in args.tickers.split(",") if t.strip())
               if args.tickers else DEFAULT_UNIVERSE)

    try:
        scan = run_quant_scan(tickers, period=args.period, lang="ko")  # Korean daily report
    except Exception as e:
        print(f"❌ Quant scan failed: {e}", file=sys.stderr)
        return 1

    print(format_alert_text(scan, top_n=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
