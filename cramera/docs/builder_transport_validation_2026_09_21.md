# Plan Builder transport validation — 21 September 2026

## Result

The ordinary Plan Builder completed a simulated PR2 transport in the installed
apartment: approach → pickup → continuous carrying navigation → placement on the
named semantic Table → release. The unchanged full acceptance test passed twice:
114.95 seconds before the final lazy-search optimization, and **86.28 seconds**
with the final native implementation.

The real Builder run was saved as
[`cramera_pr2_builder_transport_2026_09_21`](http://localhost:8712/index.html?scene=cramera_pr2_builder_transport_2026_09_21).
Its independent geometry check confirms complete support and detachment.

A second complete run authored through the normal Builder also finished with all
13 actions successful. The final status-monitor change was verified in the page:
`completed — live scene remains available`, while the runner stayed alive for
inspection. The saved first recording was retained without overwriting it.

The embedded browser still cannot create a WebGL context. Numeric scene editing,
capture, target selection, saving, launching and the recorded Plan tree were
exercised in the UI. Canvas dragging and visual 3D playback remain unverified.
Hardware execution and multi-robot operation are outside this acceptance.

## Authored scene

The default object cards were removed and `milk.stl` was added through the normal
palette. Its authored pose was `(2.37, 2.0, 1.05)` metres. In the live scene, the Y
coordinate was changed to `2.1` through the pose control and captured. The bridge
and generated class retained `(2.37, 2.1, 1.05)`.

| Setting | Value |
| --- | --- |
| Robot / environment | PR2 / installed `apartment.urdf` |
| Robot spawn | `(1.0, 2.5)` m |
| Collision avoidance | Always on |
| Base during reaching | Stand still |
| Plan | Park both arms → torso high → Transport object |
| Arm / object | Left / `milk.stl` |
| Destination | Semantic `Table`, `apartment/table_area_main` |
| Added language constraints | None |

**Save to demos** produced
`coraplex/demos/coraplex_generated/cramera_pr2_builder_transport_2026_09_21.py`.
It contains the ordinary Transport action and semantic provider, with no
scene-specific collision exceptions or substituted destination pose.

The same portable input is stored in
`test/cramera_test/dataset/builder_transport.json`. The test invokes the actual
JavaScript generator and executes the resulting `RobotDemonstration`.

## Recorded evidence

Local bundle:
`~/.local/share/cramera/builder-transport-acceptance/scenes/cramera_pr2_builder_transport_2026_09_21/`.
The bundle includes `acceptance_metrics.json` and `support_validation.json`.

| Observation | Result |
| --- | --- |
| Recorded frames / duration | 5,360 / 115.37 s |
| Action statuses | All 13 actions `SUCCEEDED` |
| Pickup / navigation actions | One pickup / two Navigate actions |
| Maximum consecutive base displacement | 0.015947 m |
| Base displacement while carrying | 3.155717 m |
| Object net displacement | 2.670479 m |
| External collision avoidance | One goal in every recorded chart; 10 charts |
| Missing assets | None |
| Final object origin | `(4.77347, 3.23969, 0.81358)` m |
| Named-table support | Complete footprint supported |
| Resting-height error | 3.282 mm; allowed tolerance 5 mm |
| Final attachment | World root `apartment/apartment_root`, detached from robot |

The saved trajectory's final object pose exactly matches the successful live
state. A separate reconstruction checked that pose against the actual tabletop
mesh through `PlacementSurface`; it did not execute or modify the live world.

Recording was finalized and saved through the existing local recording API,
because the browser's unavailable 3D panel could not expose its recording controls.
The recorded Plan view loaded and showed the complete nested action sequence.

## Changes supporting this result

- **Valid default spawn.** The former PR2/apartment default intersected furniture.
  Page state and input fields now share the verified `(1.0, 2.5)` default. Explicit
  authored positions still override it; this is not a general spawn solver.
- **Scoped manipulation contact.** Intended gripper/object and bounded
  object/support contacts use native collision-manager rules, including validation
  worlds. Previous rules are restored on exit. Attached payload is excluded from
  the gripper-link group.
- **Current attachment for placement.** Native placement goals resolve the measured
  object/tool transform at approach execution and retain it through release and
  retraction. Each new approach refreshes it.
- **Destination retries while carrying.** Semantic Transport retains its lazy
  provider until after pickup. Native `MoveAndPlaceAction` tries alternatives
  without repeating pickup. Explicit-pose and explicit-standing behavior remains.
- **Bounded stalled validation.** Reachability uses Giskard's existing
  `ProgressStalled` monitor and handles its native failure.
- **Continuous GCS routes.** Safe departure can increase separation from an
  existing clearance buffer without entering physical obstacles. Intermediate
  fixed-heading transit can join disconnected turning regions; both boundary
  turns finish in rotation-safe space. Full robot and payload geometry remain
  included, with unchanged normal clearance.
- **Nearby semantic candidates first.** Existing sampled poses are ordered by
  current object distance in world coordinates, within each surface. All samples
  and surface order remain; physical support and occupancy checks stay lazy.
- **Accurate collision diagnostics.** Reported contacts use the actual controller
  index and the same contact ordering as its buffers.
- **Exit-time recording.** Mesh storage outlives recording finalization, including
  supporting geometry generated lazily during bundling.
- **WebGL failure handling.** A specific explanation, retry button and separate
  scene link replace an unusable blank scene. Retry preserves Builder settings.
- **Authoritative run status.** The Builder reads completion, failure and
  interruption from the plan root while the live scene remains available. Idle
  scenes are distinct from running plans; responses from an older run cannot
  overwrite the current status.

The transit fallback checks endpoint orientations and eight uniformly sampled
headings. It can safely reject routes requiring several differently aligned
narrow passages. Fully specified REAL PickUp→Place sequences retain their previous
eager target resolution; this report establishes simulated execution only.

## Validation

| Checks | Result |
| --- | --- |
| Full generated Builder transport, final native source | 1 passed, 86.28 s |
| Unchanged mobile PR2 and Tracy reference transports | 6 passed, 99.25 s |
| Reachability / locations / collision-policy compatibility | 36 passed, 95.83 s |
| Destination retry / native parser and lifecycle | 44 passed, 114.44 s |
| Indexed collision reporting / controller compatibility | 6 passed, 4.07 s |
| Placement ordering, laziness and physical support | 17 passed, 20.76 s |
| Recording exit, visualization and storage batch | 68 passed, 6.55 s |
| Builder run status, state, generation and safe start | 16 passed, 10.78 s |
| Affected navigation checks | 69 passed; one legacy expectation described below |
| Earlier broad CRAMERA baseline | 1,210 passed, 207.34 s |

The broad baseline preceded the final native corrections; the focused batches and
full generated run validate those corrections. New manipulation modules reached
98% statement coverage, new destination methods 100%, new stalled-validation lines
100%, placement provider 100%, and the transit helper 93.4% line/branch coverage.
Changed Builder run-status functions reached 98.81% V8 executed-range coverage;
the plan-root interpretation and run monitor reached 100%. The final added
asynchronous error-path cases passed in the three-test status/state batch (1.20 s).
Formatting and whitespace checks pass.

### Existing test expectations awaiting authorization

The repository's `AGENTS.md` forbids modifying a failing test. Two changes are
prepared as reviewable patches and have not been applied:

1. `test_heading_change_respects_each_parts_height[1.0]` expects a disconnected
   rotational envelope to fail. The new fixed-heading transit safely passes its
   narrow corridor. The proposed replacement checks turn locations, unchanged
   clearance, fixed heading through the passage and the exact final pose.
2. `test_start_registers_an_atexit_safety_net`, when run alone, expects exactly one
   callback registration. Correct storage lifetime also registers mesh cleanup.
   The proposed update selects and verifies the recording callback; a new process
   exit regression verifies the actual finalization order and complete bundle.

The recording batch passes because mesh storage already exists by that test; its
isolated failure is explicitly retained here. These two legacy expectations keep
this from being a claim that the entire final repository suite is green.

## Reproduce

From the monorepo root using its configured Python environment:

```bash
PYTEST_XDIST_WORKER=gw0 .venv/bin/python -m pytest \
  test/cramera_test/test_builder_transport.py -q -o faulthandler_timeout=0

.venv/bin/cramera-live --viewer \
  coraplex/demos/coraplex_generated/cramera_pr2_builder_transport_2026_09_21.py
```

To serve the saved local bundle on its recorded URL:

```bash
CRAMERA_DATA="$HOME/.local/share/cramera/builder-transport-acceptance" \
  .venv/bin/python -m cramera.server 8712 --no-browser
```
