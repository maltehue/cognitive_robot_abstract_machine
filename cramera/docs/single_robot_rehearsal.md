# Single-robot rehearsal

See the [validation results](validation_2026_09_15.md) for completed checks and
known integration limits.

Run these commands from the CRAM checkout with its Python environment and ROS
package paths activated. Install the required robot descriptions and environment
meshes before going offline.

## Open a saved demonstration

```bash
cramera-live --viewer coraplex/demos/coraplex_generated/my_demo.py
```

The command starts the existing CRAMERA viewer and runs the saved demonstration
through its live bridge. The browser attaches as the world becomes available.
Ctrl-C stops the wrapper and its viewer. A demo that raises an exception also
stops its accompanying viewer.

To use a different viewer port, add `--viewer-port 8712`. The live bridge still
uses port 8765. With an existing viewer, use `cramera-live path/to/demo.py`.

## Included semantic Transport

With the Tracy description installed, this command builds a small annotated scene
and performs a complete simulated pickup and placement:

```bash
cramera-live --viewer cramera/src/cramera/semantic_transport_demo.py
```

The example subclasses `RobotDemonstration`, builds through `WorldSpecification`,
and uses the existing `TransportAction`, grasp annotations, and motion controller.
Its destination is sampled from the named table when the plan executes. The pickup
configuration follows the existing Tracy demonstration. Collision avoidance is
enabled. Existing CRAM collision rules allow the intended cube–gripper and
cube–destination contacts. The gripper still avoids the destination table, with
a 5 mm clearance configured for this simulated scene; other avoidance distances
come from the robot's existing configuration.

The Plan tab shows the executed hierarchy. After completion, stop recording and
save the episode to keep its motion, plan hierarchy and semantic query snapshots.

## Included mobile Transport

With the PR2 description installed, start the complete mobile transport scenario:

```bash
cramera-live --viewer cramera/src/cramera/mobile_transport_demo.py
```

The scenario uses the existing `TransportAction` to request approach, pickup,
carrying navigation and placement between two annotated tables. Its destination
is resolved from the named `delivery_table` at execution time. It builds on
`RobotDemonstration` and `WorldSpecification`, with native collision avoidance
enabled throughout navigation and manipulation. Existing collision rules permit
the intended grasp and object–table support contacts while retaining table
avoidance for the robot.

See the [PR2 Transport validation report](pr2_transport_validation_2026_09_16.md)
for measured execution results and remaining limits.

## Included obstacle navigation

With the PR2 description installed:

```bash
cramera-live --viewer cramera/src/cramera/navigation_demo.py
```

The normal `NavigateAction` now plans simulated routes through the existing
semantic digital twin's graph of free spaces, then drives the waypoints through
Giskard. Omnidirectional and differential bases use their existing controllers.
Collision avoidance stays active throughout navigation, including when an older
script has not enabled it globally. Stretch and Tiago use the same planner.

Planning includes the whole robot and attached payload. By default, omnidirectional
bases turn toward each travel segment while driving and take the requested final
orientation at the destination. Intermediate waypoints advance when their position
is reached. `NavigateAction` and `MoveMotion` accept `face_travel_direction=False`
to retain sideways driving.

Travel-facing routes and differential driving use a rotation-safe footprint.
Each collision part contributes its turning radius only to obstacles at overlapping
heights, including attached payloads. A wide upper part can therefore pass above
a low obstacle without inflating that obstacle by the upper part's radius.

With joint holding enabled, an omnidirectional base can use fixed-heading paths
at endpoints that fit its orientation but lack room to turn. It first translates
away from a tight starting position, turns in verified free space, and completes
the destination heading before entering a tight final approach. These departure
and approach paths reuse the same free-space graph and account for collision
geometry at each height. Their hypothetical poses never move the live world
during planning. An unchanged requested final orientation also permits a route
with constant orientation throughout.

`MoveMotion.obstacle_clearance` defaults to 5 cm,
with another 5 mm for waypoint tolerance. Native robot collision rules remain active
in the controller. The reusable `NavigationPath` can also be called independently;
its default clearance is 10 cm, with travel-facing orientation available through
`face_travel_direction=True`.

The search extends 2 m beyond the start/goal envelope. Conservative footprints can
reject narrow passages. Occupied endpoints and disconnected routes raise
`NavigationPathUnavailable`.

Each navigation plans after preceding motions finish, using the current world.
Local collision avoidance runs at every controller tick; global replanning during
a running navigation is not yet implemented. Changes that block a route can stop
the controller or exhaust its finite time budget. Failed execution clears pending
commands and releases collision-checking resources.

`keep_joint_states=True` adds the existing joint-position controller alongside
navigation. Collision avoidance has priority over posture preservation. Navigation
goals must remain on the base's current height plane with unchanged roll/pitch.
The footprint is a snapshot at planning time; local avoidance handles deviations
from that posture during execution.

The simulation integration does not require a separate `move_base` or Nav2 server.
Existing real-robot motion mappings remain available; hardware navigation needs
its own controller/localization integration and validation.

## Author and rehearse

1. Open **Plan Builder**. Select a robot and an environment in **Setup**.
2. Start the live scene. Check that the intended robot and environment appear.
3. Add an object and arrange it in the scene. Expand **pose** for precise values.
   Fold **Layers** to make more room for the scene; the choice is remembered.
4. Add the required plan steps. For semantic Transport, choose a surface type
   and optionally a specific surface supplied by the live world.
5. Run the plan. Check its execution tree and inspect the run log if it fails.
6. Generate and save the demonstration. Rehearse the command above using that
   saved file, including startup from a stopped bridge.

Collision avoidance is always enabled for new Plan Builder runs, live scaffolds
and both generated output styles. Old saved scripts retain their existing settings;
regenerate them to apply this choice.

A surface must have a CRAM semantic annotation. A mesh that resembles a table
does not by itself provide placement semantics. A full surface, an object that
does not fit, or a world without matching annotations must produce an actionable
placement error.

## Semantic questions

Every attached world offers bodies, semantic annotations, handles, robots, arms,
and supporting surfaces through the existing EQL panel. Use **show all handles**
to highlight annotated handle bodies. Placement presets appear for published loose
objects and annotated tables/countertops; they draw allowed object-center areas
after accounting for the object footprint and current collision obstacles.

The same named queries work in newly saved recordings. Recorded geometry is labelled
**Recording snapshot at finalization** and stays fixed while the replay cursor moves.
Older recordings need to be captured again to retain these semantic answers.

Placement-area overlays currently support horizontal rectangular exterior tabletops.
Tilted, round, holed and internal storage surfaces return no supported region.

## Offline recording fallback

The [recorded presentation](offline_presentation.md) combines the available semantic
queries, Transport and navigation examples into a chapter player with a single
start command and a self-contained recording bundle.

1. During a successful live run, use **Stop recording**.
2. Name and save the episode locally. Reopen it through the scene's Task selector.
3. Stop the live demo and start only `cramera`.
4. Select the saved episode, press **Play NEEM**, scrub the timeline, and inspect
   the recorded plan. Repeat with networking disabled after all assets are local.

The saved scene bundle contains its recorded motion and copied scene assets.
Check the actual saved bundle before the presentation: older recordings can
contain unresolved package mesh references and may need to be recorded again.

## Presentation acceptance checklist

- Time bring-up from the saved command to a fully visible live world; target less
  than one minute on the presentation laptop.
- Drag an object across the environment; verify the camera stays where it was
  placed and the next run uses the arranged object pose.
- Run a semantic Transport in the annotated environment; verify the final object
  position is on the selected surface.
- Run Navigate; verify intermediate base poses appear throughout the motion.
- Query handles and placement regions; verify the answer's locations in the scene.
- Test the saved recording with the scaffold stopped and networking disabled.
- For hardware mirroring, separately verify that giskard joint-state and TF
  updates reach the semantic digital twin and appear in the viewer.

## Integration boundaries

This rehearsal addresses one robot at a time. The Siemens lab description and
external robot package must be supplied and validated separately. Real-robot
plan execution, multi-robot coordination, and live camera feeds are outside this
single-robot implementation. Simulated obstacle navigation is covered by the
included rehearsal; live sensor mapping and global replanning remain separate work.
