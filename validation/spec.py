"""
validation/spec.py
─────────────────────────────────────────────────────────────────
Pre-registered constants for the forward log and its scoring. docs/validation/preregistration.md is the
human-readable version; tests/test_forward_log.py pins these values to that document, so changing one
needs a commit that changes both (and, per the pre-registration, starts a new model version).
"""
from __future__ import annotations

PREREGISTRATION_DOC = "docs/validation/preregistration.md"

# What is logged every month
EQUITY_TICKERS = ("AVGO", "AAPL", "XOM", "JNJ", "WMT")     # semis, consumer tech, energy, health care, staples
VAR_TICKER = "SPY"
VAR_CONFIDENCES = (95, 99)
VAR_HORIZON_DAYS = 21                                       # trading days: one month, so monthly records don't overlap
VAR_PATHS = 5000                                            # the live MCP tool's default

# Outcome windows
REGIME_HORIZON_DAYS = 21                                    # trading days after the as-of close
EXPECTED_RETURN_HORIZON_MONTHS = 12
EQUITY_HORIZON_MONTHS = 12
OUTCOME_INDEX = "^GSPC"                                     # price index the regime and expected-return model use
TOTAL_RETURN_BENCHMARK = "SPY"                              # equity excess returns are measured against SPY total return

# Minimum number of *primary* scored records before any forward hypothesis test is run.
# Below these, the scorer reports counts only.
MIN_N = {
    "regime": 24,              # and at least MIN_PER_REGIME_CLASS Risk-On and Risk-Off records
    "expected_return": 24,
    "var_95": 36,
    "var_99": 60,
    "backtest_position": 36,
}
MIN_PER_REGIME_CLASS = 6
ALPHA = 0.05                                                # before the Holm correction

# Source files that define the logged models. Their SHA-256 is the record's model fingerprint: records
# with different fingerprints are different model versions and are scored separately.
MODEL_FILES = (
    "requirements.txt",
    "src/alexa_mcp_server.py",
    "src/macro_data.py",
    "src/macro_extras.py",
    "src/macro_model.py",
    "src/portfolio.py",
    "src/equity_report/config.py",
    "src/equity_report/engine/analysis.py",
    "src/equity_report/engine/data_us.py",
    "src/equity_report/engine/dcf.py",
    "src/equity_report/engine/indicators.py",
    "src/equity_report/engine/market_drift.py",
    "src/equity_report/engine/patterns.py",
    "src/equity_report/engine/pipeline.py",
    "src/equity_report/engine/regime.py",
)

LOG_PATH = "validation/forward_log/log.jsonl"
SNAPSHOT_DIR = "validation/forward_log/snapshots"
SCORES_PATH = "validation/forward_log/scores.csv"
SUMMARY_PATH = "validation/forward_log/summary.md"
SCHEMA_VERSION = 1
