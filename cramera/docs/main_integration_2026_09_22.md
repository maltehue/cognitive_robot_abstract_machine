# Cramera and main integration — 2026-09-22

## Scope

Branch: `codex/cramera-main-integration`.

The integration starts from the complete local `cramera-port` feature work, captured
in commit `8b4eacf406a830a70a670388db815ef763f1ffbd`, and merges upstream
`cram2/main` at `86bca5ddb`. The original `cramera-port` checkout remains separate.
Changes made there after the snapshot are not included. Nothing was pushed.

This branch retains the full feature set: live and recorded scenes, the plan
builder, semantic placement, continuous navigation with collision avoidance,
multi-robot support, offline assets, and knowledge views. It is not the reduced
core PR.

## Integration changes

- Use main's native plan execution, motion state charts, history observations,
  object designators, geometry types, and planar navigation graph.
- Keep viewer controls on each executor tick while publishing progress through
  native history observations. Preserve interruption, reset, and failure reasons.
- Plan a navigation route once, including robot and payload geometry. Retain
  collision avoidance and precise manipulation contact rules.
- Resolve transport placement candidates inside the placement action; avoid
  preflighting an entire transport for each candidate.
- Carry forward recording transaction, asset containment, marker concurrency,
  URDF export, and browser review fixes.
- Build generated ORM interfaces locally with the repository script. They remain
  untracked, as required by main.

## Validation

An independent Python 3.12 environment was installed in the integration worktree.
The following results are separate runs and must not be added together as a
unique-test total:

| Check | Result |
| --- | --- |
| Broad Cramera suite, excluding the native navigation and transport groups listed below | 1,301 passed; 90.89% Cramera coverage; 85% gate passed |
| Native lifecycle, execution, reachability, manipulation contact and visualization/session checks | 115 passed |
| Final live control, history, visualization, world mirroring and plan recording checks | 39 passed |
| Workspace dependency declarations | 22 passed |
| Optional import handling | 8 passed |
| Deferred demo destinations and precise collision-contact defaults | 4 passed |
| Upstream navigation selection | 17 passed, 3 failed; details below |
| Cramera navigation selection | 74 passed, 2 failed; departure and parked-arm apartment motions remain open |
| Browser smoke test | Scene rendering, recording playback, plan builder and generated transport code checked |
| Dependency lock validation and Python syntax | Passed |
| Final ORM regeneration | All five packages built successfully |

The broad run excluded `test_navigation_*.py`, `test_travel_facing_waypoint.py`,
`test_builder_transport.py`, `test_builder_first_found_transport.py`,
`test_mobile_transport_demo.py`, `test_semantic_transport_demo.py`,
`test_multi_robot_transport.py`, `test_transport_capabilities.py`,
`test_transport_destinations.py`, and `test_live_history.py`. Those checks were
handled separately; the coverage result is not full transport acceptance.

Local evidence is saved under
`/home/hassouna/cram-reviews/cramera-main-integration/`.

## Remaining acceptance work

The merge does **not** establish that every physical demo or every upstream test
passes. Collision avoidance remains enabled. The following failures need separate
follow-up:

1. **PR2 apartment navigation:** the parked-arm test stops approximately 2.9 cm
   short of its target. The torso reaches a cabinet handle's native 10 cm safety
   buffer; the base is also near its 20 cm buffer. The path planner's requested
   5 cm clearance and the controller's body-specific buffers are not yet
   consistent. Goal or buffer changes were not used to hide the failure.
   A second apartment-departure test completes parking, torso raising, and route
   discovery but times out in the final Cartesian approach with external
   avoidance active. Its exact convergence cause has not been measured.
2. **Stretch apartment navigation:** collision checking reports an authored wheel
   penetrating the floor by approximately 1.98 mm. This exceeds the 1 micrometre
   geometry tolerance. The floor collision rule remains active. The same upstream
   navigation selection also rejects occupied start or goal configurations for
   Tiago and PR2 with their unchanged authored arm postures; HSRB passes.
3. **Tracy semantic transport:** initial arm parking stalls before transport.
   Native self-collision avoidance reaches the 3 cm buffers between the arms,
   camera pole, fingers, and robot table. The authored parking trajectory needs
   a feasible sequence with avoidance enabled.
4. **Full PR2 and multi-robot transport:** acceptance has not been completed on
   this merged branch. Previous feature-branch validation reports do not certify
   this integration.
5. **Broader upstream plan tests:** an additional exploratory run stopped after
   66 passes and 12 failures: five concern merged motion-chart counts and seven
   concern navigation/control-flow fixtures that previously relied on
   instantaneous movement. These tests were not disabled or made to accept the
   new output solely to obtain a green run.

The branch is suitable for inspecting the integrated code and viewer, with the
above navigation and manipulation acceptance work still open.
