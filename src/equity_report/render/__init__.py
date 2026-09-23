"""Renderers (Streamlit side only): build_all(report) -> in-memory files for st.download_button."""
from __future__ import annotations

import io
import json
import zipfile

from ..engine.analysis import Report, compact_summary
from ..engine.indicators import WINDOW, support_resistance
from ..engine.patterns import screen as pattern_screen


def build_all(rep: Report, *, dashboard_theme: str | None = None) -> dict:
    """Return {'xlsx', 'docx', 'html' (offline, for download), 'html_embed' (CDN, for st.components),
    'png': {tf: bytes}, 'zip', 'filenames': {...}} — nothing is written to disk."""
    from . import charts, dashboard, excel, word

    t, day = rep.ctx["company"]["ticker"], rep.ctx["company"]["as_of"].replace("-", "")
    pngs = {}
    for tf, (ind, _) in rep.mtf.items():
        sc = rep.screen if tf == "Daily" else pattern_screen(ind)
        sr = rep.sr if tf == "Daily" else support_resistance(ind.iloc[-WINDOW[tf]:], window=3 if tf == "Monthly" else 5)
        pngs[tf] = charts.price_chart_png(ind, sc, sr, tf, t)
    names = {"xlsx": f"{t}_model_{day}.xlsx", "docx": f"{t}_equity_report_{day}.docx",
             "html": f"{t}_dcf_dashboard_{day}.html", "zip": f"{t}_equity_report_{day}.zip"}
    out = {
        "xlsx": excel.build(rep, pngs["Daily"]),
        "docx": word.build(rep, pngs),
        "html": dashboard.build(rep, offline=True),
        "html_embed": dashboard.build(rep, theme=dashboard_theme, offline=False),
        "png": pngs, "filenames": names,
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(names["xlsx"], out["xlsx"])
        z.writestr(names["docx"], out["docx"])
        z.writestr(names["html"], out["html"])
        for tf, png in pngs.items():
            z.writestr(f"charts/{t}_{tf.lower()}.png", png)
        z.writestr("analysis.md", rep.narrative or "")
        z.writestr("summary.json", json.dumps(compact_summary(rep), indent=2))
        z.writestr("context.json", json.dumps(rep.ctx, indent=2))
    out["zip"] = buf.getvalue()
    return out
