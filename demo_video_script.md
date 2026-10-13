# MacroPulse — 3-Minute Demo Video Script v2 (target 2:45)

**Why v2:** Devpost flagged the first video as "doesn't really show the project functioning."
Every scene below is live footage of the project running on the device it was built for —
a real Echo answering by voice, the live MCP server, the dashboard reacting to clicks,
and the Fire TV app (now live on the Amazon Appstore, v1.0.2) on real hardware.
No slides-only segments; static cards are limited to the open and close.

**Rules checklist (official hackathon rules):**
- Under 3 minutes — target 2:45, judges are not required to watch beyond 3:00.
- Footage shows the project functioning on the device(s) it was built for.
- Uploaded to YouTube as **Public**, link pasted on the Devpost submission form.
- No third-party trademarks and no copyrighted music.

Narration: about 290 words, roughly 2:15 of voice, leaving room for the live
footage beats (Echo answers, chart animations, remote navigation).

Music: none, or only licensed royalty-free tracks (for example the YouTube Audio Library). No copyrighted music.
Logos: show no company logos (Apple, NVIDIA and so on). Ticker symbols appear only as data.

---

## Scene 1 — Hook on the real device (0:00–0:12)

**On screen:** Close-up of an Echo device on a desk. Say: *"Alexa, ask MacroPulse for my morning brief."*
The light ring spins and the real brief plays — let the device audio carry this beat, dip narration under it.

**Narration:**
> Retail investors can ask Alexa for a stock price. MacroPulse answers the question fund managers actually ask — what's the market regime, and what's my risk today?

---

## Scene 2 — The live server (0:12–0:35)

**On screen:** Screen recording — MCP Inspector connected to the live endpoint
`https://ma-76bae7709d4340b3ab4733ba644f11e4.ecs.us-east-1.on.aws/mcp` (Streamable HTTP).
List the tools (nine appear), run `get_morning_brief`, highlight the spoken-response field in the result.

**Narration:**
> One Python quant core runs on Amazon ECS as a self-hosted MCP server — nine tools over Streamable HTTP. Here's the live server: the morning brief returns structured data plus one sentence written for speech — the S&P close, the regime, the ten-year yield and curve, and today's scan leaders.

---

## Scene 3 — Voice plus screen (0:35–1:00)

**On screen:** *"Alexa, ask MacroPulse for the NVDA danger zone"* — the APL card renders on the
Fire TV or Echo Show while Alexa speaks the composite index.

**Narration:**
> On a screen device, the same answer renders as a card — the danger index with its RSI, volume and trend components, built for the remote and the ten-foot view.

---

## Scene 4 — Monte Carlo risk, live (1:00–1:30)

**On screen:** Dashboard Risk Simulation tab — set SPY, 10,000 paths, 30 days, click Run.
The fan chart draws; VaR 95/99 and Expected Shortfall (CVaR) appear.

**Narration:**
> The same engine drives the fourteen-tab dashboard. Ten thousand Monte Carlo paths on SPY — the fan chart draws live, and Value-at-Risk and Expected Shortfall read straight off the simulated distribution, Gaussian or bootstrapped from real market history.

---

## Scene 5 — Equity report (1:30–2:05)

**On screen:**
1. Open `hf-macro-dashboard.streamlit.app/?ticker=AAPL` — it lands directly on the Equity Report tab.
2. Scroll the regime-weighted scenarios and the WACC × g grid with the FOMC shock badges.
3. Download the Excel model; change one yellow assumption cell; show the valuation recalculating.

**Narration:**
> Type any ticker and you get an institutional research pack — fair value from a DCF on SEC ten-K filings, bear, base and bull scenarios weighted by the live macro regime. The Excel model keeps live formulas: change an assumption, and the whole valuation recalculates.

---

## Scene 6 — Fire TV, shipped (2:05–2:30)

**On screen:** A real Fire TV — open the Amazon Appstore, find MacroPulse (live, v1.0.2), open it,
then D-pad through the regime card, rates, the NVDA danger index and the watchlist signals.

**Narration:**
> And it's a real product — the Fire TV companion app is live on the Amazon Appstore. Same AWS endpoint: regime, rates, the danger index, today's signals — all navigable with the remote.

---

## Scene 7 — Trust and close (2:30–2:50)

**On screen:**
1. A terminal showing the offline test suite passing.
2. The honest-error card shown when live data is unavailable ("says so instead of guessing").
3. End card: the GitHub repository URL and "Informational only — not investment advice."

**Narration:**
> Under the hood, a full offline test suite — and one hard rule: if live data isn't available, MacroPulse says so instead of guessing. MacroPulse: an institutional macro desk, one question away.

---

## Narration only (teleprompter copy)

Retail investors can ask Alexa for a stock price. MacroPulse answers the question fund managers actually ask — what's the market regime, and what's my risk today?

One Python quant core runs on Amazon ECS as a self-hosted MCP server — nine tools over Streamable HTTP. Here's the live server: the morning brief returns structured data plus one sentence written for speech — the S&P close, the regime, the ten-year yield and curve, and today's scan leaders.

On a screen device, the same answer renders as a card — the danger index with its RSI, volume and trend components, built for the remote and the ten-foot view.

The same engine drives the fourteen-tab dashboard. Ten thousand Monte Carlo paths on SPY — the fan chart draws live, and Value-at-Risk and Expected Shortfall read straight off the simulated distribution, Gaussian or bootstrapped from real market history.

Type any ticker and you get an institutional research pack — fair value from a DCF on SEC ten-K filings, bear, base and bull scenarios weighted by the live macro regime. The Excel model keeps live formulas: change an assumption, and the whole valuation recalculates.

And it's a real product — the Fire TV companion app is live on the Amazon Appstore. Same AWS endpoint: regime, rates, the danger index, today's signals — all navigable with the remote.

Under the hood, a full offline test suite — and one hard rule: if live data isn't available, MacroPulse says so instead of guessing. MacroPulse: an institutional macro desk, one question away.

---

## Before recording

1. **Real hardware first.** A real Echo for Scenes 1 and 3 and a real Fire TV for Scene 6 are the
   strongest possible answer to "show the project functioning." Fallbacks: the Streamlit
   "Alexa+ Institutional Copilot" tab with "Play Alexa Voice," and MCP Inspector on the live endpoint.
2. **Fire TV capture.** Install from the Amazon Appstore on the device, or sideload the APK
   (`adb connect <ip>`, `adb install MacroPulse.apk`). Record with an HDMI capture card, or
   `adb shell screenrecord /sdcard/demo.mp4` then `adb pull`.
3. **Dashboard capture.** Use a clean browser window; `?ticker=AAPL` opens straight on the Equity Report.
4. **Echo capture.** Phone on a tripod for the device shot; keep room noise down so the spoken
   answer is the star of Scene 1.
5. **Length.** Keep the final cut at or under 2:45; judges may stop at 3:00.
6. **Upload.** YouTube does not allow replacing a video file — upload the new cut as a **new video**,
   set it **Public**, then edit the Devpost submission and swap the video URL. Set the old video
   to Unlisted.
