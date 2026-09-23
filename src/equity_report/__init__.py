"""MacroPulse Equity Report — ticker in, Excel model + DCF dashboard + Word note + charts out.

Layout
  engine/  pandas-only analysis (data, indicators, patterns, DCF, regime overlay, rule opinion)
           -> safe to import from the Alexa+ MCP server image
  render/  Excel / Word / HTML dashboard / PNG charts -> Streamlit app only
"""
from .engine.analysis import Report, analyze, compact_summary, narrate  # noqa: F401
