"""Macro-regime overlay for the valuation (step 4).

MacroPulse already classifies the market as Risk-On / Neutral / Risk-Off from credit-spread and
volatility z-scores. The equity report uses that call in two transparent ways:

1. Scenario probabilities (bear / base / bull) shift toward the regime.
2. The equity risk premium gets a small add-on in Risk-Off and a small discount in Risk-On.

The table below is the whole rule — it is shown in the report, the Excel model and the dashboard,
and every weight can be overridden.
"""
from __future__ import annotations

from dataclasses import dataclass

OVERLAY = {
    #            bear   base   bull   ERP shift
    "Risk-On":  (0.20, 0.50, 0.30, -0.0025),
    "Neutral":  (0.25, 0.50, 0.25, 0.0),
    "Risk-Off": (0.40, 0.45, 0.15, +0.0050),
}


@dataclass(frozen=True)
class RegimeOverlay:
    regime: str
    probs: dict            # {"Bear": p, "Base": p, "Bull": p}
    erp_shift: float
    source: str            # where the regime call came from

    def as_dict(self) -> dict:
        return {"regime": self.regime, "probs": dict(self.probs), "erp_shift": self.erp_shift, "source": self.source}


def normalize(label: str | None) -> str:
    """'Risk-Off 🔴' / 'risk off' / None -> 'Risk-Off' / 'Neutral'."""
    s = (label or "").lower().replace("_", "-").replace(" ", "-")
    if "risk-on" in s:
        return "Risk-On"
    if "risk-off" in s:
        return "Risk-Off"
    return "Neutral"


def overlay(label: str | None, source: str = "MacroPulse regime engine") -> RegimeOverlay:
    reg = normalize(label)
    bear, base, bull, erp = OVERLAY[reg]
    if label is None:
        source = "no regime supplied (neutral weights)"
    return RegimeOverlay(reg, {"Bear": bear, "Base": base, "Bull": bull}, erp, source)
