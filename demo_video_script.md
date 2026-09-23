# MacroPulse — 3-Minute Demo Video Script (target 2:50)

Narration: about 390 words, roughly 2:50 at a relaxed pace.
Music: none, or only licensed royalty-free tracks (for example the YouTube Audio Library). No copyrighted music.
Logos: show no company logos (Apple, NVIDIA and so on). Ticker symbols appear only as data.

---

## Scene 1 — Hook (0:00–0:15)
**On screen:** MacroPulse dashboard home (regime banner, KPI row), then an Echo or Fire TV.

**Narration:**
> Retail investors can ask Alexa for a stock price. Fund managers ask a different question: what regime is the market in, and what is this stock actually worth? MacroPulse brings that desk to Alexa.

---

## Scene 2 — Architecture (0:15–0:35)
**On screen:** One architecture slide.
- **Clients:** Alexa+, the Fire TV app and the Streamlit dashboard.
- **Server:** the MCP server on Amazon ECS, serving `/mcp` and `/api`.
- **Data sources:** yfinance, FRED and SEC EDGAR.

**Narration:**
> One Python quant core runs on Amazon ECS as a self-hosted MCP server: nine tools over Streamable HTTP, MCP spec 2025-11-25. The same service feeds a Fire TV app and a fourteen-tab dashboard.

---

## Scene 3 — Live MCP server (0:35–1:05)
**On screen:** MCP Inspector connected to the live server.
1. Set the transport to Streamable HTTP and the URL to `https://ma-76bae7709d4340b3ab4733ba644f11e4.ecs.us-east-1.on.aws/mcp`, then click Connect.
2. List the tools; nine should appear.
3. Run `get_morning_brief`.
4. Highlight `alexa_spoken_response` in the result.

**Narration:**
> Here's the live server. Nine tools. Morning brief returns structured data plus one sentence written for speech — the S&P close, the regime, the ten-year and curve, and today's scan leaders. It's built for an Alexa Routine: "Alexa, good morning."

---

## Scene 4 — Alexa+ experience (1:05–1:30)
**On screen:** The Streamlit Alexa+ Copilot tab.
1. Click the "Morning Brief" preset. The card and badges appear.
2. Click "Play Alexa Voice" so the audio plays.
3. Click the "AAPL Report" preset. The spoken summary and the four download buttons appear.

**Narration:**
> In our Alexa+ simulator, the same tools answer by voice and render the card a Fire TV or Echo Show would show. Ask for an equity report on Apple, and Alexa gives the headline — fair value, regime weighting, rating — and the full pack is one click away.

---

## Scene 5 — Equity Report (1:30–2:05)
**On screen:**
1. Type `hf-macro-dashboard.streamlit.app/?ticker=AAPL`. The app opens directly on the Equity Report tab.
2. Scroll the dashboard: scenarios, then the WACC × g grid with the H/D/S/L FOMC badges.
3. Download the Excel model.
4. On the Assumptions sheet, change yellow cell B28 (terminal growth) from 2.5% to 3.0%.
5. Show the valuation on the Summary sheet recalculating.

**Narration:**
> A deep link opens straight on the report. Fundamentals come from SEC 10-K filings; bear, base and bull scenarios are weighted by the live macro regime; FOMC shock scenarios sit on the sensitivity grid. The Excel model keeps live formulas — change an assumption and the valuation recalculates.

---

## Scene 6 — Fire TV (2:05–2:35)
**On screen:** The MacroPulse app running on a real Fire TV, or the Fire TV/Vega simulator.
1. The home-screen banner, then open the app.
2. Show the regime card, the S&P 500, rates, the NVDA danger index and the watchlist signals.
3. Use the remote's D-pad to move to Settings.

**Narration:**
> On Fire TV, the companion app reads the same AWS endpoint — regime, rates, the NVDA danger index and today's signals, all navigable with the remote.

---

## Scene 7 — Trust and close (2:35–2:50)
**On screen:**
1. A terminal showing `pytest` with "137 passed".
2. The spoken error card shown when live data is unavailable.
3. End card: the GitHub repository URL and "Informational only — not investment advice."

**Narration:**
> Under the hood: 137 offline tests, and one hard rule — if live data isn't available, MacroPulse says so instead of guessing. MacroPulse: an institutional macro desk, one question away.

---

## Narration only (teleprompter copy)

Retail investors can ask Alexa for a stock price. Fund managers ask a different question: what regime is the market in, and what is this stock actually worth? MacroPulse brings that desk to Alexa.

One Python quant core runs on Amazon ECS as a self-hosted MCP server: nine tools over Streamable HTTP, MCP spec 2025-11-25. The same service feeds a Fire TV app and a fourteen-tab dashboard.

Here's the live server. Nine tools. Morning brief returns structured data plus one sentence written for speech — the S&P close, the regime, the ten-year and curve, and today's scan leaders. It's built for an Alexa Routine: "Alexa, good morning."

In our Alexa+ simulator, the same tools answer by voice and render the card a Fire TV or Echo Show would show. Ask for an equity report on Apple, and Alexa gives the headline — fair value, regime weighting, rating — and the full pack is one click away.

A deep link opens straight on the report. Fundamentals come from SEC 10-K filings; bear, base and bull scenarios are weighted by the live macro regime; FOMC shock scenarios sit on the sensitivity grid. The Excel model keeps live formulas — change an assumption and the valuation recalculates.

On Fire TV, the companion app reads the same AWS endpoint — regime, rates, the NVDA danger index and today's signals, all navigable with the remote.

Under the hood: 137 offline tests, and one hard rule — if live data isn't available, MacroPulse says so instead of guessing. MacroPulse: an institutional macro desk, one question away.

---

## Before recording

1. **Deploy the fixes.** Merge PR #11, redeploy the MCP server, and restart the Streamlit app. That way the footage shows the fixed bar dates and peer returns.
2. **MCP Inspector (Scene 3).** Run `npx @modelcontextprotocol/inspector`, set the transport to Streamable HTTP, paste the `/mcp` URL above and click Connect. If the Alexa+ add-on is live on a device, filming "Alexa, ask MacroPulse for my morning brief" on a real device is even stronger.
3. **Fire TV (Scene 6).** Footage on a real device or simulator is required for the Fire TV track.
   - Install the APK: `adb connect <FireTV-IP>`, then `adb install MacroPulse.apk`.
   - Record the screen with an HDMI capture card, or with `adb shell screenrecord /sdcard/demo.mp4` followed by `adb pull`.
4. **Browser voice (Scene 4).** Make sure the browser's speech synthesis uses an English voice.
5. **Length.** Keep the final cut under 2:50; judges may stop at 3:00.
6. **Upload.** Publish on YouTube as Public and paste the link into the Devpost form.
