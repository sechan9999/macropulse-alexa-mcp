# MacroPulse — Friction Log (Amazon App Dev Hackathon 2026)

Issues we hit while building the Alexa+ MCP server and the Fire TV app, in the order we met them.
Each entry lists the task, the steps, expected vs. actual result, severity, the workaround we used and a suggestion.

Severity: **High** = blocked or could not verify a core feature · **Medium** = cost hours or a rebuild · **Low** = confusing but quick to resolve.

| # | Area | Summary | Severity |
|---|---|---|---|
| 1 | Alexa+ add-on / MCP | No way to test how Alexa+ picks among our tools before launch | High |
| 2 | Fire TV testing | No official Fire OS emulator for Android apps | Medium |
| 3 | Fire TV sample app | Permanent side-nav drawer is drawn over screen content | Medium |
| 4 | Fire TV focus | Focused `TouchableOpacity` fades to 20% and looks disabled | Medium |
| 5 | Fire TV build | EAS build fails on react-native-tvos peer dependencies | Medium |
| 6 | Appstore console | "Tablet assets" required for a Fire TV-only app | Low |
| 7 | Appstore console | Featured-logo preview on white hides white-on-transparent logos | Low |
| 8 | Appstore console | DRM and >50 MB warnings without actionable guidance | Low |
| 9 | Appstore console | 91 of 98 Fire TV devices pre-selected with no reason given | Low |

---

## 1. No way to test how Alexa+ picks among our tools — High

- **Task:** Verify that Alexa+ calls the right MCP tool (for example `get_morning_brief` vs. `get_macro_regime`) and reads `alexa_spoken_response` well.
- **Steps:** Deployed the MCP server on Amazon ECS, prepared the add-on manifest (`addon-package/addon.json`), then tested the endpoint with MCP Inspector (`tools/list`, `tools/call`).
- **Expected:** A developer console where we can type an utterance and see which tool Alexa+ selects, the arguments, and the spoken result.
- **Actual:** We could only test the server in isolation. Tool selection and speech rendering could not be observed before the add-on was live.
- **Workaround:** Wrote narrow tool descriptions and typed parameters. Built an "Alexa+ simulator" tab in our dashboard that routes sample utterances to the same tools and plays the spoken sentence with browser TTS.
- **Suggestion:** An Alexa+ add-on test console, like the Alexa Skills Kit simulator, that sends test utterances to a registered MCP endpoint and shows the tool call trace and the spoken response.

## 2. No official Fire OS emulator for Android apps — Medium

- **Task:** Test the Fire TV APK and capture 1920×1080 Appstore screenshots without Fire TV hardware.
- **Steps:** Searched the Fire TV docs for an emulator. The Vega virtual device targets Vega OS apps, not Android APKs.
- **Expected:** An emulator image (or AVD profile) that matches Fire OS 7/8.
- **Actual:** None found for Android-based apps.
- **Workaround:** Installed Android Studio and created an Android TV **API 30** AVD (Android 11, the base of Fire OS 8). Installed the APK with `adb install -r` and captured screenshots with `adb shell screencap`.
- **Suggestion:** Publish a Fire OS system image for the Android emulator, or document the closest AVD per Fire OS version, including how to capture listing screenshots.

## 3. Side-nav drawer drawn over screen content — Medium

- **Task:** Reuse the left-hand navigation from `AmazonAppDev/hello-world-fire-tv-react-native`.
- **Steps:** Adapted `LeftHandNav` / `DrawerContent` (permanent drawer, absolutely positioned so it can expand over the screen) and added our own screens.
- **Expected:** Screen content starts to the right of the collapsed rail.
- **Actual:** The first ~60 px of every screen were hidden under the rail (section titles and the leading `$` of prices were cut off). Found only when testing on the emulator.
- **Workaround:** Reserved the collapsed width with `sceneContainerStyle: {marginLeft: COLLAPSED_WIDTH}` in the drawer's `screenOptions`.
- **Suggestion:** Note in the sample (next to the absolutely positioned drawer) that each screen must reserve the collapsed rail width, or reserve it in the navigator itself, so apps adapted from it don't ship clipped content.

## 4. Focused `TouchableOpacity` looks disabled — Medium

- **Task:** Make the watchlist rows focusable with the D-pad and show which one is selected.
- **Steps:** Used `TouchableOpacity` with an orange border on focus.
- **Expected:** The focused row is highlighted.
- **Actual:** react-native-tvos animates a focused `TouchableOpacity` to `activeOpacity` (0.2 by default), so the selected row faded and looked disabled.
- **Workaround:** Set `activeOpacity={1}` and rely on the border/underline for focus.
- **Suggestion:** Call this out in the Fire TV React Native focus-management docs and use a non-fading focus style in the samples.

## 5. EAS build fails on react-native-tvos peer dependencies — Medium

- **Task:** Build a release APK with Expo EAS (`EXPO_TV=1`, `@react-native-tvos/config-tv`).
- **Steps:** `npx eas-cli build -p android --profile production`.
- **Expected:** The cloud build installs dependencies and produces an APK.
- **Actual:** `npm install --include=dev` failed with `ERESOLVE` because `react-native` is aliased to `react-native-tvos`.
- **Workaround:** Added `firetv-app/.npmrc` with `legacy-peer-deps=true`.
- **Suggestion:** Include this, plus a working `eas.json` production profile (APK build type, `EXPO_TV=1`), in a "Build a Fire TV app with Expo" guide.

## 6. "Tablet assets" required for a Fire TV-only app — Low

- **Task:** Complete the Appstore listing for an app targeting Fire TV only (0 Fire Tablets selected).
- **Expected:** Only Fire TV assets are required.
- **Actual:** The listing stayed "Incomplete" until the Tablet assets section (512/114 icons, 3+ screenshots) was filled.
- **Workaround:** Reused the Fire TV screenshots (1920×1080 is accepted) and the store icons there.
- **Suggestion:** Label that section "Base listing assets (all apps)", or hide the tablet-only fields when no tablets are targeted.

## 7. Featured-logo preview on a white background — Low

- **Task:** Upload the 640×260 featured content logo ("transparency optional").
- **Expected:** A preview that looks like the Fire TV UI.
- **Actual:** The preview is white, so a white wordmark on a transparent background looked broken.
- **Workaround:** Redesigned the logo as an opaque navy badge.
- **Suggestion:** Preview TV assets on a dark Fire TV-style background, and clarify that "PNG (with transparency)" means transparency is allowed, not required.

## 8. DRM and binary-size warnings without guidance — Low

- **Task:** Upload a free APK.
- **Actual:** "Outdated SDK version that does not include DRM integration" (for a free app that needs no DRM) and ">50 MB binary" warnings, with no concrete next step.
- **Workaround:** Selected "No" for DRM. Kept the universal APK.
- **Suggestion:** Hide the DRM warning when DRM is declined. For size, link to ABI-split guidance and state whether AAB uploads are accepted for Fire TV.

## 9. 91 of 98 Fire TV devices pre-selected, no reason given — Low

- **Task:** Review supported devices after upload.
- **Actual:** 7 devices were excluded automatically with no explanation.
- **Workaround:** Inferred it from `minSdkVersion` 23 (older Fire OS 5 devices are API 22).
- **Suggestion:** Show the reason next to excluded devices, e.g. "minSdkVersion 23 > device API 22".

---

What worked well: the Appstore **Testing instructions** field, clear screenshot size rules, fast APK manifest validation, and **Amazon ECS Express Mode**, which gave the MCP server an HTTPS endpoint with no load-balancer setup.
