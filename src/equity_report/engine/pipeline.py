"""One call from ticker to analysed Report (no rendering)."""
from __future__ import annotations

from typing import Callable

import pandas as pd

from ..config import DCFAssumptions
from . import data_us, synthetic
from .analysis import Report, analyze, narrate

DEMO_TICKERS = {"DEMO"}


def run(ticker: str, *, assumptions: DCFAssumptions | None = None, regime_label: str | None = None,
        regime_source: str = "MacroPulse regime engine", risk_free: float | None = None,
        price_fetch: Callable[[str], pd.DataFrame] | None = None, http_get: Callable[[str], dict] | None = None,
        provider: str = "rules", with_positioning: bool = True, market: dict | None = None) -> Report:
    t = ticker.upper().strip()
    if t in DEMO_TICKERS:
        bundle = synthetic.stock_bundle(t)
        if risk_free is not None:
            bundle["risk_free_live"] = risk_free
    else:
        kw = {"price_fetch": price_fetch, "risk_free": risk_free, "with_positioning": with_positioning}
        if http_get is not None:
            kw["http_get"] = http_get
        bundle = data_us.load_bundle(t, **kw)
    rep = analyze(bundle, assumptions, regime_label, regime_source, market=market)
    rep.issues = data_us.validate(bundle)
    return narrate(rep, provider)
