# ⚡ Macro Pulse — Fire TV Companion View

A glanceable Fire TV companion app for [Macro Pulse](../README.md): today's macro
regime, key market stats, an NVDA danger-zone reading, and your watchlist
signals — visible on the living-room TV without opening a laptop or phone.

Built for the Amazon Developer Hackathon's **Fire TV** track. Talks to
[`../rest_server.py`](../rest_server.py), a plain JSON API — not the Alexa+
track's MCP server (`../mcp_server.py`). Fire TV apps are human-facing
remote-control UIs, not AI agents, so a REST API is the right-sized interface
here; see `rest_server.py`'s module docstring for the full reasoning.

## Status

Code-complete and **type-checked** (`npx tsc --noEmit` passes clean) and
**linted** (`npm run lint` passes, 2 harmless warnings — one is a lint rule
Amazon's own starter template triggers too). It has **not** been run on an
actual Fire TV device, emulator, or the Expo Android build — this development
environment has Node/npm/Java but no Android SDK (`adb`, `ANDROID_HOME`), so
that verification step is left for you to do locally or via `eas build`.

Scaffolded from and closely follows
[AmazonAppDev/hello-world-fire-tv-react-native](https://github.com/AmazonAppDev/hello-world-fire-tv-react-native)
(MIT-0) — the navigation/drawer/focus-handling code (`navigation/`,
`components/Header.tsx`) is adapted from that starter with minimal changes,
since it's Amazon's own proven pattern for TV remote (d-pad) focus handling
and this environment can't verify a from-scratch implementation. Only the
screens and content components are new.

## Prerequisites

- Node.js 18+ (tested with Node 24)
- A Fire TV device, Fire TV emulator, or Android TV emulator with `adb`
  reachable, **or** an [Expo EAS](https://expo.dev/eas) account for a cloud
  build
- `../rest_server.py` running somewhere reachable from the TV/emulator (see
  the main [README](../README.md))

## Setup

```bash
npm install --legacy-peer-deps
```

`--legacy-peer-deps` is required — `react-native-tvos`'s prerelease-style
version string (`0.74.2-0`) doesn't satisfy some packages' semver peer
ranges even though it's ABI-compatible with the react-native version they
expect. This is a known, common `react-native-tvos` ecosystem quirk, not
specific to this project.

```bash
npm run android   # requires adb + an Android/Fire TV emulator or device
# or
npx eas build --platform android --profile development   # cloud build, no local SDK needed
```

## Configuration

On first launch, open **Settings** (left-hand nav) and set:

- **Macro Pulse API URL** — where `rest_server.py` is reachable from the TV/emulator.
  - From the Android emulator to your host machine: `http://10.0.2.2:8080` (the default)
  - From a real Fire TV device on the same network: `http://<your-machine-LAN-IP>:8080`
  - Deployed: your Cloud Run URL for `rest_server.py`
- **User ID** (optional) — if set, the Watchlist Signals section pulls that
  user's saved Firestore watchlist (see `../src/firestore_service.py`)
  instead of the shared default universe.

Both are persisted locally via `@react-native-async-storage/async-storage`.

## Project Structure

```text
firetv-app/
├── App.tsx                    # Entry point — NavigationContainer + drawer
├── api/
│   └── macroPulseApi.ts       # Fetch wrapper for rest_server.py + settings storage
├── navigation/
│   ├── LeftHandNav.tsx        # Drawer navigator (adapted from the Amazon starter)
│   ├── DrawerContent.tsx      # Collapsible drawer + TV remote focus handling (adapted)
│   └── DrawerItem.tsx         # One drawer menu item (adapted)
├── components/
│   ├── Header.tsx             # Screen title (from the Amazon starter, unchanged)
│   ├── RegimeCard.tsx         # Macro regime badge
│   ├── StatTile.tsx           # One glanceable stat (S&P, yield, vol, ...)
│   └── SignalRow.tsx          # One watchlist signal row, TV-focusable
└── screens/
    ├── HomeScreen.tsx         # The glanceable dashboard — fetches + auto-refreshes every 5 min
    └── SettingsScreen.tsx     # API URL + user_id configuration
```

## Known gaps / next steps

- No app icon / Android TV banner image (`app.json`'s `androidTVBanner` was
  left out rather than pointing at a placeholder) — needed before any real
  store submission.
- No error retry/backoff beyond the 5-minute refresh interval.
- Not tested on a real Fire TV remote's actual d-pad behavior — the focus
  handling follows Amazon's own starter pattern closely, but only that
  starter's authors have verified it on real hardware.
