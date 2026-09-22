# Multi-robot validation — 2026-09-21

## Shared-world transport

The ordinary Plan Builder generated and executed a scene containing two PR2s in
the installed apartment. The second instance, `robot_2` (`PR2 Transport`), ran
Park arms → Move torso → Transport milk onto the semantic table
`apartment/table_area_main`. The first instance, `robot_1` (`PR2 Reserve`), stayed
in the world at `(6.5, 1.0, 0.0)`. The Builder reported completion.

The saved demo is
`coraplex/demos/coraplex_generated/cramera_multi_robot_transport.py`.
The recording is [cramera_multi_robot_transport](http://localhost:8712/index.html?scene=cramera_multi_robot_transport),
stored under
`~/.local/share/cramera/builder-transport-acceptance/scenes/cramera_multi_robot_transport`.

| Check | Measured result |
| --- | --- |
| Independent recorded robot models | Two complete articulated PR2 models |
| Recording | 4,815 frames, approximately 166 seconds |
| Passive robot | Every base coordinate and all 45 joint channels exactly constant |
| Active base movement | Maximum frame-to-frame step 12.91 mm |
| Carrying interval | 3.08 m net displacement; 6.45 m travelled |
| Actions | All 13 succeeded; attachment and detachment succeeded |
| Collision avoidance | Present in all 10 recorded controller charts |
| Assets | None missing; all 95 referenced geometry files present |

Multi-robot replay uses each instance's `modelBases` track. The compatibility
`base` track changes identity when the active robot changes; it must not be used
as one robot's trajectory. The actual-page replay regression checks this override.
Carrying intervals in the recording audit use the existing motion-derived object
window helper. Native execution tests independently observe the actual attachment.

## Sequential plan switching through the UI

After the pose-capture correction, the second PR2 completed Park arms, Move torso
and Navigate. Its final root pose was
`[4.754, 2.65782, 0, 0, 0, 0.7257, 0.68801]` (XYZ and quaternion).
The first robot was then selected and its Park arms and Move torso plan executed
successfully. The second robot's complete `/robots` entry, including its base and
all 45 joint positions, was exactly equal before and after that run.

The UI also rejected a navigation goal inside the inflated obstacle region.
Restarting through **Controls → Stop** allowed another run with the previously
validated free delivery pose.

After reloading the backend, the saved recording's **which robots are in this
scene?** preset returned `robot_1` and `robot_2`, each with two arms. **Which arms
belong to each robot?** returned four separately named arms with the correct
robot and gripper ownership. Both results were checked in the normal query UI.
The final Builder scene remains live with both robots.

## Automated checks

- Full generated two-PR2 Transport plus selected-robot navigation and obstacle
  checks: **5 passed** in 235.28 seconds. The transport test checks continuous
  carrying, successful release, full object-footprint support, resting height
  within 5 mm, selected-robot collision goals, and every passive body transform
  at every controller tick.
- Native namespace, scene assembly and captured joint restoration: **43 passed**.
  The new scene module has **98% line and branch coverage**.
- Selected-robot joint resolution and existing joint-motion compatibility:
  **8 passed** across the final focused runs. Native Park arms and torso movement
  with two identical PR2s preserve every passive body transform. The motion module
  has **94% line and branch coverage**.
- Navigation and reachability compatibility: **40 passed**. This includes a
  driven detour around the stationary second robot.
- Builder instance selection, generation, pose capture and startup controls:
  **29 passed**, followed by **8 passed** for the sequential-start regression.
  The latter reproduced a reset to the original position after a robot had
  already driven; accepted launches now acknowledge the applied pose edits.
  All seven changed executable statements in that fix are covered. New state
  methods have **100% coverage**; the focused page check measured **91.26% V8
  line coverage**.
- Bridge, recordings, replay and recorded knowledge: **395 passed** in 48.20
  seconds. The new robot identity and recorded-description modules have **100%
  coverage**; ModelPoses has **90.6% V8 executed-range coverage**.
- Multi-robot presets and existing single-robot query compatibility: **85 passed**.
  Recorded and live scenes return all robots and the arms belonging to each one.

These batches overlap; their counts are not a combined suite total. Existing
tests were not rewritten to accept changed behavior. Formatter and whitespace
checks pass.

## Limits and outstanding acceptance

The in-app browser cannot create a WebGL context. The real Builder workflow,
native execution, saved bundle and replay logic were checked, but the rendered
3D appearance still requires visual acceptance in a working WebGL browser.

An existing ORM round-trip test for semantic annotation bindings fails in the
broader native batch with `NoDAOFoundError`. The prescribed ORM regeneration
completed, and the isolated retry still failed. No generated ORM interface was
read or edited manually. This is not a claim that the entire repository suite
is green.

See the [multi-robot guide](multi_robot.md) for sequential execution boundaries,
attachment continuation limits, and the remaining external lab/Stretch work.

## Local evidence

- `/tmp/cramera-multi-recording-evidence.json`
- `/tmp/cramera-multi-transport-acceptance-numeric.log`
- `/tmp/cramera-multi-robot-final.log`
- `/tmp/cramera-multi-joint-motion-native-final.log`
- `/tmp/cramera-multi-motion-compatibility-final.log`
- `/tmp/cramera-multi-robot-builder-complete-tests.log`
- `/tmp/cramera-multi-restart-green.log`
- `/tmp/cramera-multi-all-final.log`
- `/tmp/cramera-multi-presets-green.log`
- `/tmp/cramera-multi-switch-final-evidence.json`
