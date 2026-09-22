# B-5 recorded presentation rehearsal — 2026-09-16

## Result and scope

The prepared laptop starts the offline reference presentation with one command:

```bash
./scripts/start_cramera_offline.sh
```

The browser opens `http://localhost:8713/tour.html`. The recordings live in
`~/.local/share/cramera/offline-demo`, independently of temporary recording folders.
The three scene bundles contain 84 files and 38,408,153 bytes, plus the storyboard
and scene index. Meshes, materials, textures and trajectories pass the recursive
local-asset check.

This rehearsal covers the reference single-robot storyboard: introduction to the
semantic world, handle queries, free placement regions, Tracy Transport, its plan
hierarchy and policy-replacement explanation, and PR2 navigation. Transport and
navigation recordings were made with collision avoidance enabled. They show
separate worlds. No learned policy executes.

The Siemens lab description and external robot models are still missing. This is
therefore not acceptance of the final Siemens storyboard. The optional physical
Stretch segment and multi-robot scene are outside this reference presentation.

## Cold-process timing

The earlier presentation process was stopped cleanly. The launch timestamp was
written immediately before executing the documented launcher with `--no-browser`.
After HTTP readiness, the existing browser tab was reloaded. Responses use
`Cache-Control: no-store`; this was not a laptop reboot or cleared OS file cache.

| Milestone | Time from command start |
| --- | ---: |
| Scene frames and recorded knowledge ready; presentation controls enabled | 21.406 s |
| Transport chapter additionally selected, 3D scene loaded and saved plan tree visible | 33.137 s |

The scene milestone uses the DOM's `#tour-status[data-ready-at]`, set only after
both actual recording frames and the knowledge API are ready. The second milestone
waits for the Transport plan and ready state in the browser. Both measurements
include the tool and manual chapter-selection overhead. Opening the default
desktop browser automatically was covered by unit tests; this measured run used
the already running in-app browser.

Raw epoch times in milliseconds:

- Command start: `1789548862530`
- Scene ready: `1789548883936`
- Transport and plan verified: `1789548895667`

## Offline conditions

The launcher places the viewer in a new Linux network namespace. Its only incoming
connection is an inherited socket bound to host loopback. The verified viewer
process had namespace `net:[4026533323]`; the host had `net:[4026531833]`. The viewer's
route table was empty. An integration test verifies that a new outgoing connection
fails with `ENETUNREACH` while the inherited HTTP listener still serves the host.

The browser receives a same-origin Content Security Policy, including local-only
connections, scripts, fonts and frames. Camera and microphone are disabled. The
tour does not poll a live bridge; saved questions use local HTTP endpoints. The
server rejects new plan executions and recording mutations.

The host's Wi-Fi and airplane-mode switch were not changed. This proves application
operation with external networking denied, rather than a physical whole-laptop
airplane-mode rehearsal. That final hardware check remains explicit:

1. Stop the presentation with Ctrl-C and enable airplane mode on the laptop.
2. Start a stopwatch with the command above.
3. Confirm an interactive scene and the Transport plan within 60 seconds.
4. Return to chapter one, press Start and let every chapter finish automatically.
5. Confirm the final completion message; repeat a query and a recorded movement.

## Browser usability and playback

The complete six-chapter tour was played automatically in the browser, with no
robot, live scaffold or Stretch process. The playback chapters advance only after
their last recorded frame. The final PR2 frame was 1037 of 1038, and the tour showed
“Tour abgeschlossen. Alle Kapitel wurden gezeigt.” The browser reported no warnings
or errors during that run. The complete tour was repeated after the final layout
changes. The recorded placement question was also selected manually and returned
its saved placement region successfully.

The presentation layout was checked at 1280 × 720. Query chapters reserve space
for their answers; execution chapters use the full right column for the plan tree.
The normal viewer's duplicate playback controls and constraint editor are hidden
inside the tour. Pause/resume, chapter navigation and repeat remain available in
the outer presentation controls.

## Automated checks

- Final complete CRAMERA suite: **1,183 passed in 173.30 seconds**, process exit 0.
  The suite reports 1,109 existing NumPy scalar-conversion and generated-string
  escape warnings.
- 35 presentation CLI, browser-launch and isolated-network tests passed unchanged.
- 38 recursive asset validation tests passed.
- 65 frontend and affected viewer regression checks passed, including 14 Node
  cases for the tour and its existing-panel integration.
- Python line coverage: presentation 98%, asset validation 99%, network launcher
  100%.

See [the operating guide](offline_presentation.md) for bundle preparation,
validation, alternate ports and startup options.
