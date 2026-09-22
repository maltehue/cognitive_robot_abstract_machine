# Single-robot validation — 15 September 2026

Validated in the CRAM monorepo worktree on `cramera-port`, using the existing
`.venv`, local robot packages, and a real browser. Generated demos and recordings
were isolated under `/tmp/cramera-validation`; existing generated demos and scene
submodule changes were preserved.

## Automated checks

```bash
.venv/bin/python -m pytest -q test/cramera_test --disable-warnings
```

The initial run passed **1,061 tests**, with 8 warnings, in 168.38 seconds while
coverage was enabled. The suite includes Node-based frontend regressions through
pytest. Added or changed executable Python lines have **98.24% coverage**
(783 of 797), including the new modules and changed Transport/navigation logic.
Docstrings were formatted with the repository script; `git diff --check` passed.

After enabling collision avoidance, the complete suite passed **1,064 tests**,
with 9 warnings, in 188.39 seconds under coverage. The updated semantic Transport
module has 96% executable-line coverage, including the intended-contact rules and
checks that arm, opposite-gripper and tabletop collision checks remain present.

The initial obstacle-navigation implementation passed **1,159 tests** in 343.77
seconds, covering CRAMERA, the existing GCS and Cartesian-controller suites, and
selected executor/template regressions. After refining the apartment footprint,
the final targeted run passed **70 tests** in 56.13 seconds. A further **36 tests**
cover launcher, visualization and ROS marker shutdown behavior. Together with the
actual demonstration run, added or changed executable lines in navigation and the
affected supporting modules have **96.96% coverage** (383 of 395). Docstring
formatting and `git diff --check` passed.

A subsequent broad run collected 1,165 tests and completed the CRAMERA and GCS
portions, but ended with SIGTERM during the native Cartesian-controller suite.
It has no completed test-total result; the targeted run above verifies the final
navigation changes.

Four existing plan tests failed in that separate completed check:

- `test_graph_parsing.py::test_parse_transport_plan`: PR2 Transport raises
  `EmptyUnderspecified` during action grounding. **Resolved on 16 September:** the
  unchanged test passes after the navigation footprint and connector corrections;
  see the [PR2 follow-up](pr2_transport_validation_2026_09_16.md).
- `test_language.py::test_perform_execute_single`: its navigation endpoint is
  occupied under the collision footprint.
- `test_language.py::test_exception_try_in_order` and
  `test_language.py::test_exception_try_all`: their navigation children fail, so
  the combinators correctly report `AllChildrenFailed`.

The old tests' assertions were retained. The three language-test failures remain
outside the subsequent Transport fix; this is not a clean full-CRAM-suite result.
Collision-enabled navigation no longer accepts those blocked goals.

Regressions cover authored poses during an immediate Run after dragging,
namespaced object lookup, stationary and single-arm Transport composition,
semantic placement and occupied surfaces, spatial query geometry, live query
attachment, recording serialization, launcher cleanup, and disabled condition
nodes in a completed plan.

## Rehearsal evidence

| Requirement | Result and evidence | Remaining scope |
| --- | --- | --- |
| A-1: robot/environment selection | Catalog derives 12 robot choices and 7 URDF environments from CRAM. Ten robot descriptions spawned locally; capability tests cover Tracy, PR2 and HSRB. | Garmi's installed description disagrees with its annotation; DAiSy lacks `ur_robot_driver`. The Siemens external package and lab were unavailable. A successful Transport is not established for every robot. |
| A-2: visual authoring | Browser object dragging, pose capture and generated world reuse checked. A regression preserves the released pose when an older bridge snapshot arrives before the final move request finishes. | Furniture editing remains later work. |
| A-3: semantic Transport | Included Tracy demo executes the normal Transport, grasp and motion controller. The cube finishes within 1 mm of its sampled table target. **16 September follow-up:** PR2 now completes pickup, continuous carrying navigation and placement on the named table, with active collision avoidance and a saved browser replay. | Accepted for the supplied annotated simulation scenes. Arbitrary Builder worlds and the Siemens lab still need their own acceptance; see the [PR2 follow-up](pr2_transport_validation_2026_09_16.md). The earlier diagnostic crash was traced to Python's faulthandler watchdog. |
| A-4: continuous navigation | Normal NavigateAction uses SDT's GCS planner and Giskard controllers with collision avoidance. The final PR2 obstacle run records 700 frames and 697 distinct base positions, reaches x=3 m, and detours 1.395 m sideways; the largest recorded step is 9.9 mm. A parked PR2 also drives inside the existing apartment. | Global replanning during an active route and physical base/localization integration remain open. The final browser reload encountered a WebGL context creation failure; an earlier navigation recording played to completion successfully. |
| A-5: spatial answers | Real browser queries highlight two annotated handles and draw free table regions, live and from a saved bundle. Areas account for object footprint and collision obstacles. | Supported areas are horizontal rectangular exterior tabletops. Recorded answers are explicitly finalization snapshots, independent of replay time. |
| A-6: live knowledge | Bridge automatically supplies world queries without per-demo registration. Presets and typed EQL were exercised in the browser; transcript/voice routing has automated coverage. | Microphone capture itself was not rehearsed. |
| A-7: external world updates | A real isolated giskard standalone server drove Stretch. The mirrored world produced 192 snapshots with zero local motion ticks; head pan reached 0.4 rad and lift reached 0.64879 m for a 0.65 m target. | Physical hardware, localization through TF, and a driving circle were not tested. |
| A-9: camera stability | Browser dragging across the environment leaves the camera fixed. Orbit controls are disabled before the drag begins; camera-follow has regression coverage. | — |
| B-5: launch and fallback | Initial endpoint readiness was 10.29 seconds. The [2026-09-16 reference-tour rehearsal](offline_validation_2026_09_16.md) adds a durable six-chapter fallback, application network isolation and browser timing: 21.4 seconds to an interactive scene, 33.1 seconds including the Transport recording and saved plan. | Physical whole-laptop airplane mode and the final Siemens storyboard still require verification; the lab and external robot assets are unavailable. |

## Reproduce the complete Transport

From the monorepo root with the CRAM environment activated and Tracy installed:

```bash
cramera-live --viewer cramera/src/cramera/semantic_transport_demo.py
```

The initial measurements above used the existing Tracy pickup configuration with
collision avoidance disabled. The subsequent collision-avoidance update enables
it in the included demo and every Plan Builder output. CRAM's existing collision
rules permit only cube–gripper and cube–destination contact, with a 5 mm gripper
clearance over the destination table in this simulated scene. Other collision
distances retain the robot's configuration. The complete Transport was repeated
successfully with these rules and collision avoidance enabled.

Open **Plan** to inspect the hierarchy, then **Stop recording → Save episode** to
keep a replay. Previously saved Python demos retain their original settings and
need to be regenerated to pick up the mandatory collision-enabled builder mode.

The local validation viewer was started on port 8712 with
`CRAMERA_DATA=/tmp/cramera-validation/data`. Its saved episode is named
`cramera_semantic_transport`. These temporary validation outputs are not repository
fixtures; the included Python demonstration can recreate the episode.

The subsequent collision-enabled recording is named
`cramera_collision_avoidance_transport`: 5,025 frames over 18.09 seconds, with no
missing assets. Its recorded statecharts include `ExternalCollisionAvoidance`.
The browser showed every executed Transport step as complete, and its replay ran
to the end after the live process was stopped.

## Continuous navigation update

```bash
cramera-live --viewer cramera/src/cramera/navigation_demo.py
```

The demo uses the standard PR2, NavigateAction, MoveMotion and Giskard execution
stack. It moves around a collidable box and records intermediate controller ticks.
Omnidirectional and differential-drive regressions also cover successive relative
goals, final heading, pure rotation, a long detour, interrupted plans and cleanup
after compilation or controller failure. Footprints include attachments and other
agents remain obstacles. Height-aware tests distinguish supporting contact from a
5 mm protrusion.

The saved final episode is `cramera_navigation_verified`, under the local
validation data directory above. Its NavigateAction and MoveMotion both have
`SUCCEEDED` status, its bundle contains no missing assets, and its final base pose
is approximately (3.0, -0.00004, 0). An earlier rotation-safe-footprint recording,
`cramera_collision_aware_navigation`, has 920 frames; the browser showed the live
plan complete and subsequently played its saved episode to frame 919 with the
bridge stopped.

After the apartment footprint refinement, the demonstration completed again.
Opening a fresh 3D view then failed with `THREE.WebGLRenderer: Error creating WebGL
context`. Reloading and a fresh tab reproduced the browser failure; Chrome was
not available through computer control. The final 700-frame recording was saved
through CRAMERA's existing local recording API. Its trajectory and completion
statuses were checked directly; final visual playback remains to be repeated once
the browser can create a graphics context.

The demo registers the existing `RobotDemonstration.stop_visualization` cleanup
before execution, and the marker subscriber treats shared ROS-context shutdown as
a normal end of delivery. A complete PR2 run followed by SIGINT now exits with
code 0, without a native abort or Python traceback. The saved episodes remain
available after the bridge stops.

## Travel-facing navigation — 2026-09-16

The viewer can create its WebGL context again. Both the earlier verified episode
and the new `cramera_navigation_facing` recording played to completion in the
in-app browser. The new PR2 run uses the default travel-facing Navigate action,
the GCS route and native collision avoidance. NavigateAction and MoveMotion both
completed successfully, and there are no missing assets.

The new recording contains 1,038 frames. Base yaw changes continuously between
approximately -86° and +86° while passing the box, then returns to the requested
final orientation at (3, 0, 0). The largest recorded position step is 9.90 mm and
the largest yaw step is 0.575°. The viewer is set to 0.5× playback. Stopping the
live bridge after saving the episode again exited with code 0.

Regression tests cover travel-facing default behavior, explicit sideways driving,
final orientation, no-op and pure-turn requests, the narrow-apartment fallback,
short route segments, tilted bases, and position-based waypoint transitions.
The focused navigation, controller, demo and GCS suite passed all 52 tests. Line
coverage across the three navigation modules is 98%, with the motion module at
100%.

Multi-robot behavior was excluded as requested. Builder execution on physical
robots (A-8), the Siemens lab presentation, and live camera feeds remain outside
this implementation. See the [rehearsal guide](single_robot_rehearsal.md) for the
remaining presentation checks.
