"""
src/alexa_prompt_ui.py
─────────────────────────────────────────────────────────────────
Preset buttons and the "Spoken Voice Input" box of the Alexa+ Copilot tab.

A Streamlit text_input with a key keeps its own state after the first run and ignores its value=
argument, so a preset that only passes value= never reaches the box (and the old text was dispatched).
Each preset button therefore writes the text into st.session_state in its on_click callback, which
runs before the widget is drawn on the rerun.
"""
from __future__ import annotations

PROMPT_KEY = "alexa_prompt_input"
FIRED_KEY = "alexa_preset_fired"
DEFAULT_PROMPT = "Alexa, what is today's market regime?"

PRESETS = (
    ("🟢 'Market Regime'", "Alexa, ask MacroPulse for today's market regime"),
    ("📺 'Show NVDA on TV'", "Alexa, show the NVDA Danger Zone on the TV"),
    ("⚡ 'FOMC Shock Test'", "Alexa, run an FOMC rate shock test on my portfolio"),
    ("🎲 '99% VaR Sim'", "Alexa, run a 99% risk simulation on SPY over the next 30 days"),
    ("⚡ 'Vol Breakout'", "Alexa, scan volatility squeeze breakouts on SPY"),
    ("📑 'AAPL Report'", "Alexa, give me an equity report on AAPL"),
    ("🌅 'Morning Brief'", "Alexa, ask MacroPulse for my morning brief"),
)


def prompt_controls(st, presets=PRESETS) -> tuple[str, bool]:
    """Draw the preset buttons and the input box. Returns (prompt, a preset was just clicked)."""
    st.session_state.setdefault(PROMPT_KEY, DEFAULT_PROMPT)

    def use(text: str) -> None:
        st.session_state[PROMPT_KEY] = text
        st.session_state[FIRED_KEY] = True

    for i, (col, (label, text)) in enumerate(zip(st.columns(len(presets)), presets)):
        with col:
            st.button(label, use_container_width=True, key=f"alexa_preset_{i}", on_click=use, args=(text,))
    fired = bool(st.session_state.pop(FIRED_KEY, False))
    prompt = st.text_input("Spoken Voice Input:", key=PROMPT_KEY)
    return prompt, fired
