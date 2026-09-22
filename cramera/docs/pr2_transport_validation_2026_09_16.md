# PR2 semantic Transport validation — 16 September 2026

This rehearsal exercises one simulated PR2 with the existing `TransportAction`,
GCS navigation, Giskard motion controllers and a semantically named destination.
The scene contains a supported object and two annotated tables two metres apart.
It is created by `MobileTransportDemo`, a `RobotDemonstration` subclass using
CRAM's `WorldSpecification` and existing grasp/placement actions.

## Reproduce

From the CRAM checkout with its Python environment and installed PR2 description:

```bash
cramera-live --viewer cramera/src/cramera/mobile_transport_demo.py
```

The transport resolves a free object pose on `delivery_table`. It picks up the
object, parks the arm for driving, navigates to the destination, places and
releases the object, then parks the arm again. Collision avoidance is enabled.
Existing collision rules permit the intended object–left-gripper and
object–table contacts. The left gripper still avoids both tabletops with a 5 mm
buffer. Other robot/environment checks retain the robot's configured distances.
The object–table exceptions apply throughout this small scene's execution.

## Corrections exercised

- Destination approach locations are constructed when navigation executes, using
  the actual attachment and current robot posture after pickup.
- Reachability costmaps transform table-relative target poses into world
  coordinates before sampling base positions. Arm goals retain their reference
  frames. A translated and rotated frame has a dedicated regression.
- Reachability validation uses the active execution environment's collision
  avoidance and the same TCP tolerances as execution. The copied validation world
  retains configured distances and temporary contact rules, and static
  base-screening rules are restored
  before testing the arm trajectory. Infeasible candidates are rejected with
  controller resources released on every exit path.
- GCS uses obstacle heights when calculating turning clearance. Fixed-heading
  departure and approach segments connect manipulation positions to free space
  where the base can turn. Orientation must complete before the final approach.
  Route planning does not change the live world.
- Placement and its reachability check use the measured object–tool transform.
  Existing grasp offsets still determine approach and retreat. Regression cases
  cover residual grasp translation, rotation and a rotated base.
- Rotated navigation footprints transform each original collision shape directly
  into the proposed pose. Repeatedly rotating an already enlarged bounding box
  had incorrectly rejected otherwise free endpoints with an attached payload.
- Collapsed native actions publish their execution lifecycle through the existing
  plan callbacks. Successful transport, pickup and placement appear as completed
  in the live plan tree; failed alternatives remain visible as failed attempts.
- Movable primitive objects remain tracked across grasp attachment and release.
  Recording export excludes their duplicate static geometry and retains each
  shape's local transform.

## Recorded execution and playback

The saved episode is `cramera_pr2_semantic_transport`. With the accompanying
viewer running, open
[the recorded PR2 transport](http://localhost:8712/index.html?scene=cramera_pr2_semantic_transport).
Its local bundle is
`~/.local/share/cramera/mobile-transport-demo/scenes/cramera_pr2_semantic_transport`.

| Observation | Recorded result |
| --- | --- |
| Motion frames | 5,008 |
| Recorded duration | 36.96 seconds |
| Base displacement while carrying | 2.245 m |
| Largest consecutive recorded base step | 11.20 mm |
| Initial object center | (1.50000, 0.00000, 0.89000) m |
| Final object center | (3.49605, −0.06969, 0.88948) m |
| Action statuses | All 11 executed actions succeeded |
| Collision avoidance | Present in all 11 recorded motion statecharts |
| Missing assets | None; all 51 referenced URDF mesh files are local |

These measurements come from the actual episode, not a synthesized animation.
The bundle includes `acceptance_metrics.json`. After stopping the live bridge,
browser playback reached its final frame and showed the released orange object
on the destination table, with no browser console errors. Playback is set to
half speed for inspection.

To reopen that bundle after stopping the viewer:

```bash
CRAMERA_DATA="$HOME/.local/share/cramera/mobile-transport-demo" cramera 8712
```

The saved episode predates the final reachability-policy correction. Its successful
execution demonstrates the complete native chain; the automated acceptance below
also checks the final implementation.

## Automated acceptance

The complete test executes the native controller without replacing its actions.
It requires one successful Transport attempt and exactly two successful navigation
motions. It also observes collision-avoidance goals during every executed motion,
continuous base steps smaller than 2 cm, more than 1 m of motion with the object
attached, final release, the configured orientation tolerance, and the entire
object footprint supported by the named destination table.

```bash
PYTEST_XDIST_WORKER=gw0 python -m pytest \
  test/cramera_test/test_mobile_transport_demo.py \
  -q -o faulthandler_timeout=0
```

| Check | Result |
| --- | --- |
| Final PR2 pickup/transport and Tracy demo/status acceptance | 6 passed in 82.76 seconds |
| CRAMERA regression suite, excluding the separately run PR2 demo tests | 1,206 passed in 167.68 seconds |
| Final shared reachability policy, existing validators/locations and original Transport regression | 36 passed in 129.50 seconds |
| Navigation, connectors, heading and footprint regression | 53 passed |
| Action lifecycle, alternatives, callbacks and interruption | 24 passed |
| Recorded shape geometry and export compatibility | 30 passed |

The complete PR2 test passed on three independently sampled destinations after
collision-aware reachability was introduced. The last run includes full copying
of the active collision policy. The broad CRAMERA suite ran before that final
policy-copy adjustment; dedicated policy tests and the final PR2/Tracy acceptance
cover the adjustment. The last 36-test batch also covers the shared context-copy
helper used by both navigation locations and object-reachability predicates.

Coverage is 94% for `MobileTransportDemo`, 99% combined for the navigation planner
and motion, and 99% for recording-bundle export in the focused geometry suite.
Affected reachability methods have 98% statement coverage; all 86 added/changed
lifecycle statements were exercised. Test groups above
overlap and must not be added together as a unique test count.

## Scope

This is a simulation acceptance scene. It establishes the single-PR2 transport
chain in the supplied annotated world; it does not establish B-2's remaining
Siemens-lab or external-robot acceptance. Hardware localization/navigation,
physical grasp stability and multi-robot operation are separate work. Global
replanning during an active route remains outside this implementation.

The Plan Builder uses the same lazy `PlacementSurface` and `TransportAction`
mechanics. This demo additionally configures the scene's contact rules and skips
looking at the operation site; it is not an end-to-end acceptance of an arbitrary
generated Builder scene.

Action completion is reported from execution. Generic condition nodes that were
not evaluated retain their original status; the acceptance test independently
checks attachment, carried motion, release, orientation and final surface support.

## Native diagnostic crash

Earlier diagnostic runs crashed while the pytest timeout watchdog attempted to
print native threads. GDB inspection of both collected cores places the fault
in `_Py_DumpTracebackThreads` on the faulthandler worker. The earlier attribution
to symbolic evaluation was not supported by those stacks. These checks disable
that watchdog with `-o faulthandler_timeout=0` and use an external process timeout.
