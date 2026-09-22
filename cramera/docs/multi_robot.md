# Multiple robots in one CRAM world

See the [validation report](multi_robot_validation_2026_09_21.md) for execution,
recording and test evidence.

The Plan Builder can place several independently named robots in the same native
semantic digital twin. Instances may use the same model. Each instance has its
own plan; one selected robot executes at a time. Other robots remain part of the
world and of the selected robot's collision geometry.

## Try the prepared demo

The saved demonstration uses two PR2s in the apartment. `PR2 Transport` picks up
the milk, drives it to `apartment/table_area_main`, and releases it on that
semantic surface. `PR2 Reserve` remains stationary.

From the repository root, with the project environment active:

```bash
cramera-live --viewer coraplex/demos/coraplex_generated/cramera_multi_robot_transport.py
```

The local acceptance recording is called `cramera_multi_robot_transport`. On the
running validation server it opens at
[the recorded scene](http://localhost:8712/index.html?scene=cramera_multi_robot_transport).
Its bundle is in
`~/.local/share/cramera/builder-transport-acceptance/scenes/cramera_multi_robot_transport`.
This recording is a local validation artifact, not a checked-in asset.

## Author a scene

1. Open **Plan Builder** and select the environment in **Setup**.
2. Under **Robots in scene**, name and position the first robot.
3. Use **Add robot** to create another instance. Select its model, name, position
   and heading. Choose separated, collision-free starting poses.
4. Use **Active robot** to choose whose plan you are editing. Add that robot's
   supported skills. Switching instances preserves their separate plans.
5. Start the live scene, place objects, then use **Run plan**.
6. After completion, select another instance and run its plan. The Builder
   captures current base poses and scalar joint positions before replacing its
   scene process.

Robot selection is held while the scene starts, and the live bridge rejects a
selection change while a plan is running or paused. Removing an instance leaves
at least one robot in the scene.

## Native CRAM integration

`RobotInstance` and `RobotScene` assemble existing `RobotSpecification` and
`WorldSpecification` objects. Instance namespaces are passed through the native
URDF parser to bodies, joints and localization connections. The selected native
robot annotation becomes the plan's `Context.robot`.

Joint motions resolve names within that robot's connections. Motion execution
and reachability validation pass that robot explicitly to Giskard's external
collision avoidance. Navigation continues through the existing GCS planning and
Giskard controllers.

The live bridge publishes all robots through `/robots`; `/robot` selects one.
Bundles contain separate articulated models, per-instance root trajectories and
robot metadata. Replay, recorded queries and kinematic inspection distinguish
instances even when their model classes and local link names are identical.

## Scope

- Plans execute sequentially. Coordinated simultaneous driving, shared-object
  manipulation and robot-to-robot handover are not implemented.
- A Builder run rebuilds its process using captured poses and scalar joint
  states. An object still attached to a gripper is not a supported continuation
  across separate runs; finish placement before switching plans.
- Collision avoidance applies to the generated native plans. This work does not
  add collision avoidance to the existing direct-IK hand teleoperation.
- The innovation lab, its external robot assets and the real Stretch connection
  still need their own acceptance. This establishes the shared-world foundation
  for requirement B-1; it does not complete B-1 or B-4 with substitute assets.
- The validation browser cannot create a WebGL context. Builder controls,
  execution, scene data and replay code are tested; visual 3D acceptance still
  needs a browser with working WebGL.
