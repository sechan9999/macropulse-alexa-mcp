"""Streamlit UI for the Equity Report tab (and the download row reused by the Alexa+ tab).

Kept out of app.py so it can be tested with streamlit.testing.AppTest and so app.py only needs
two lines per tab.
"""
from __future__ import annotations

from datetime import datetime

from .config import DCFAssumptions

PROVIDERS = {"Auto (Gemini if key set, else rules)": "auto", "Rules only (no LLM)": "rules",
             "Gemini": "gemini", "Amazon Bedrock": "bedrock"}
REGIME_MAX_AGE_DAYS = 62


def regime_inputs(macro_df) -> tuple[str | None, float | None, str]:
    """Pick the regime label and live 10Y yield from the dashboard's macro frame.

    Returns (regime_label or None, risk_free decimal or None, human-readable source note).
    A stale window (sidebar end date in the past) or the demo fallback frame yields neutral / None.
    """
    try:
        if macro_df is None or len(macro_df) == 0:
            return None, None, "macro data unavailable -> neutral weights"
        if bool(macro_df.get("_is_demo", False).iloc[-1]) if "_is_demo" in macro_df else False:
            return None, None, "macro data is the offline demo frame -> neutral weights, fallback risk-free"
        last_dt = macro_df.index[-1]
        age = (datetime.now() - last_dt.to_pydatetime()).days
        if age > REGIME_MAX_AGE_DAYS:
            return None, None, (f"sidebar end date {last_dt:%Y-%m} is {age} days old -> neutral weights "
                                "(move the end date to today to use the live regime)")
        rf = float(macro_df["dgs10"].iloc[-1]) / 100 if "dgs10" in macro_df else None
        return str(macro_df["regime"].iloc[-1]), rf, f"MacroPulse regime engine, {last_dt:%b %Y}"
    except Exception as e:  # noqa: BLE001
        return None, None, f"regime unavailable ({type(e).__name__}) -> neutral weights"


def _generate_uncached(ticker: str, regime_label, regime_source, risk_free, erp, tg, provider, theme):
    from .engine.analysis import compact_summary
    from .engine.pipeline import run
    from .render import build_all
    rep = run(ticker, assumptions=DCFAssumptions(equity_risk_premium=erp, terminal_growth=tg),
              regime_label=regime_label, regime_source=regime_source, risk_free=risk_free, provider=provider)
    files = build_all(rep, dashboard_theme=theme)
    return {"files": files, "summary": compact_summary(rep), "issues": rep.issues, "method": rep.narrative_method}


def _generate_by_day(ticker, regime_label, regime_source, risk_free, erp, tg, provider, theme, day):
    return _generate_uncached(ticker, regime_label, regime_source, risk_free, erp, tg, provider, theme)


_CACHED = None


def generate(st, ticker, regime_label=None, regime_source="", risk_free=None, erp=0.05, tg=0.025,
             provider="auto", theme="dark"):
    """Cached per (inputs, calendar day) for 6 hours; raises UnsupportedFiler / DataUnavailable."""
    global _CACHED
    if _CACHED is None:
        _CACHED = st.cache_data(ttl=6 * 3600, show_spinner=False, max_entries=24)(_generate_by_day)
    fn = _CACHED
    rf = None if risk_free is None else round(float(risk_free), 5)
    return fn(ticker.upper().strip(), regime_label, regime_source, rf, round(erp, 5), round(tg, 5), provider, theme,
              datetime.now().strftime("%Y-%m-%d"))


def download_row(st, res: dict, key: str) -> None:
    f, n = res["files"], res["files"]["filenames"]
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("⬇️ Excel model", f["xlsx"], n["xlsx"],
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"{key}_xlsx",
                       use_container_width=True)
    c2.download_button("⬇️ Word report", f["docx"], n["docx"],
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document", key=f"{key}_docx",
                       use_container_width=True)
    c3.download_button("⬇️ DCF dashboard (HTML)", f["html"].encode("utf-8"), n["html"], "text/html", key=f"{key}_html",
                       use_container_width=True)
    c4.download_button("⬇️ Everything (.zip)", f["zip"], n["zip"], "application/zip", key=f"{key}_zip",
                       use_container_width=True)


def _error(st, e: Exception, ticker: str) -> None:
    from .engine.data_us import DataUnavailable, UnsupportedFiler
    if isinstance(e, UnsupportedFiler):
        st.warning(f"**{ticker}** isn't supported by the equity report: {e} "
                   "The DCF needs US-GAAP 10-K financials in USD (ETFs and foreign filers such as TSM or ASML "
                   "file 20-F/40-F reports).")
    elif isinstance(e, DataUnavailable):
        st.error(f"Live data for **{ticker}** is unavailable right now: {e}. Nothing was estimated — try again shortly.")
    else:
        st.error(f"Report failed for **{ticker}**: {type(e).__name__}: {e}")


def render_tab(st, components, macro_df=None) -> None:
    st.markdown("### 📑 Equity Report — ticker in, research pack out")
    st.caption("Excel DCF model (live formulas) · Word research note · interactive DCF dashboard · Monthly/Weekly/Daily "
               "KD·MACD·candlestick charts. Fundamentals from SEC EDGAR 10-K filings, prices from Yahoo Finance, "
               "scenario weights from the MacroPulse regime. **Not investment advice.**")
    regime_label, rf_live, regime_src = regime_inputs(macro_df)
    qp = st.query_params.get("ticker", "") if hasattr(st, "query_params") else ""
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        ticker = st.text_input("US ticker (10-K filers; try DEMO for synthetic data)", value=qp or "AAPL",
                               key="er_ticker").upper().strip()
    with c2:
        prov_label = st.selectbox("Narrative", list(PROVIDERS), key="er_provider")
    with c3:
        st.write("")
        go = st.button("📑 Generate report", type="primary", use_container_width=True, key="er_go")
    with st.expander("Assumptions (also editable later in the Excel model)"):
        a1, a2, a3 = st.columns(3)
        erp = a1.number_input("Equity risk premium (base)", 0.02, 0.10, 0.05, 0.0025, format="%.4f", key="er_erp")
        tg = a2.number_input("Terminal growth", 0.0, 0.04, 0.025, 0.0025, format="%.4f", key="er_tg")
        rf_override = a3.number_input("Risk-free override (0 = live 10Y)", 0.0, 0.10, 0.0, 0.0025, format="%.4f",
                                      key="er_rf")
    rf = rf_override if rf_override > 0 else rf_live
    st.caption(f"Regime overlay: **{regime_label or 'Neutral (no live regime)'}** · {regime_src} · risk-free "
               f"{'%.2f%%' % (rf * 100) if rf else 'fallback 4.25%'}")

    if go and ticker:
        with st.spinner(f"Pulling SEC filings and prices for {ticker}, running DCF and rendering files…"):
            try:
                st.session_state["er_result"] = generate(st, ticker, regime_label, regime_src, rf, erp, tg,
                                                         PROVIDERS[prov_label])
                st.session_state["er_error"] = None
            except Exception as e:  # noqa: BLE001
                st.session_state["er_result"], st.session_state["er_error"] = None, (e, ticker)
    if st.session_state.get("er_error"):
        _error(st, *st.session_state["er_error"])
        return
    res = st.session_state.get("er_result")
    if not res:
        st.info("Enter a ticker and press **Generate report**. Alexa+ can also trigger it: "
                "“Alexa, give me a report on AAPL”.")
        return
    s = res["summary"]
    if s.get("synthetic_data"):
        st.warning("SYNTHETIC DEMO DATA — not a real company.")
    m = st.columns(6)
    m[0].metric("Last price", f"${s['price']:,.2f}", s["as_of"])
    m[1].metric("Prob-weighted value", f"${s['fair_value_prob_weighted']:,.2f}", f"{s['upside_prob_weighted'] * 100:+.1f}%")
    m[2].metric("DCF base", f"${s['dcf_base']:,.2f}")
    m[3].metric("Rating (rules)", s["rating"], f"score {s['score']:+.1f}")
    m[4].metric("WACC", f"{s['wacc'] * 100:.2f}%", f"g {s['terminal_growth'] * 100:.1f}%")
    m[5].metric("Regime", s["macro_regime"]["regime"],
                f"bear {s['macro_regime']['probs']['Bear']:.0%} / bull {s['macro_regime']['probs']['Bull']:.0%}")
    for issue in res["issues"]:
        st.warning(f"Data check: {issue}")
    download_row(st, res, "er")
    st.caption(f"Narrative: {res['method']} · Sources: {'; '.join(s['data_sources'])}")
    components.html(res["files"]["html_embed"], height=2900, scrolling=True)
