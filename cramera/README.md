# cramera

**Watch a CRAM robot plan run — recorded or live — in the browser, and ask the scene what happened.**

[![Live demo](docs/demo.gif)](https://sunava.github.io/cramera/)

**[Open the live demo](https://sunava.github.io/cramera/)** · [Changelog](CHANGELOG.md) · [Source in the CRAM monorepo](https://github.com/sunava/cognitive_robot_abstract_machine/tree/cramera-port/cramera)

cramera is the visualization front end of the [CRAM](https://github.com/cram2/cognitive_robot_abstract_machine)
cognitive architecture. One tool, three ways to look at a robot:

- **Recorded.** Run a coraplex demo once through the onboarder and get a self-contained
  scene bundle — URDFs, meshes and the real recorded giskardpy trajectory — that replays in
  any browser. No ROS, no simulator, no Python needed to watch it (the demo site above is
  exactly that: static files).
- **Live.** Attach the viewer to a running demo and it renders the executing world as it
  moves: plan steps light up as they run, motion statecharts stream in, and dragging an
  object in the viewer writes its pose back into the demo's world.
- **Asked.** Type a question in English — *which arm picked up the milk?* — and the EQL
  console matches it to a query over the recorded episode, answers it, shows where the
  answer comes from, points an arrow at the object, and can replay the moment.

It also ships a **Plan Builder**: compose a plan by drag and drop, place objects and
targets in the live 3D scene, attach plain-language constraints, and generate a runnable
`RobotDemonstration`.

## Quick start

cramera lives inside the CRAM monorepo; this repository is a read-only mirror of its
`cramera/` directory, kept in sync by a scheduled workflow.

```bash
git clone https://github.com/sunava/cognitive_robot_abstract_machine.git
cd cognitive_robot_abstract_machine
pip install -e cramera
cramera                                              # serves http://localhost:8711
```

Ready-made recordings live in [cram2/cram-scenes](https://github.com/cram2/cram-scenes);
initialize the optional submodule to get them locally:

```bash
git submodule update --init cramera/scenes
```

Record and watch a demo of your own:

```bash
cramera-onboard path/to/demo.py --name my_scene     # record a demo once, get a bundle
cramera-live path/to/demo.py                        # run a demo with the live bridge attached
cramera-live --viewer path/to/demo.py               # start the viewer and demo together
```

For a saved Plan Builder demo, run
`cramera-live --viewer coraplex/demos/coraplex_generated/my_demo.py`.
The viewer opens in the browser and attaches when the world's bridge is ready.
Ctrl-C stops the demo wrapper and the viewer it started. Use
`--viewer-port 8712` when port 8711 is occupied, or omit `--viewer` to use a viewer
you already started. Both commands support `--help`.

See the [single-robot rehearsal guide](docs/single_robot_rehearsal.md) for scene
editing, semantic placement, offline playback and remaining integration checks.

For the prepared single-robot presentation, run `./scripts/start_cramera_offline.sh`
from the monorepo root. The [offline presentation guide](docs/offline_presentation.md)
describes its six recorded chapters, local asset checks and isolated launch.

A small semantic Transport rehearsal is included. With the Tracy model installed,
run this from the monorepo root:

```bash
cramera-live --viewer cramera/src/cramera/semantic_transport_demo.py
```

It uses CRAM's normal grasp and motion execution to place a cube on an annotated
table, resolving the destination at run time.

For the full mobile Transport scenario, with the PR2 model installed:

```bash
cramera-live --viewer cramera/src/cramera/mobile_transport_demo.py
```

The existing `TransportAction` requests pickup, carrying navigation and placement
on the named `delivery_table`, resolved at execution time. Native collision
avoidance remains enabled throughout. See the
[PR2 Transport validation report](docs/pr2_transport_validation_2026_09_16.md)
for measured results and remaining limits.

The ordinary Plan Builder also completes PR2 transport in the installed apartment,
including a moved object, carrying navigation and release on a named semantic
table. See the [Builder transport validation](docs/builder_transport_validation_2026_09_21.md)
for the saved recording, measured results and remaining browser/test boundaries.

The Builder also supports several named robots in one world, including multiple
instances of the same model. Each robot has its own plan and can be selected for
sequential execution with native collision avoidance. See the
[multi-robot guide](docs/multi_robot.md) for the two-PR2 Transport demo, authoring
steps and current limits.

For continuous navigation around an obstacle, with the PR2 model installed:

```bash
cramera-live --viewer cramera/src/cramera/navigation_demo.py
```

The normal Navigate action combines CRAM's existing free-space planning and
Giskard controllers, with collision avoidance throughout the simulated drive.
Height-aware footprints include carried objects. Omnidirectional bases holding
their joint posture can leave or approach tight endpoints with a fixed heading,
completing required turns in free space before the final approach. See the
[navigation rehearsal](docs/single_robot_rehearsal.md#included-obstacle-navigation)
for the supported behavior and planning bounds.

The viewer looks for bundles in `CRAMERA_SCENES=/path`, then the `cramera/scenes`
submodule, then `~/.cramera/scenes` (where the onboarder writes). Pick a scene with
`?scene=<name>`.

## What is in the box

| | |
| --- | --- |
| `robot-scene` panel | three.js scene: environment, robot, draggable objects, TF frame triads, ROS debug markers, playback with key-moment marks and thumbnails, live controls, recording |
| `eql` panel | the question console: English in, EQL query and answer out, spoken if you like |
| `graph` panel | Knowledge · Plan (step tree with live status and constraints) · Statechart · Kinematics · Transforms |
| Models page | the probabilistic-model workbench: query, posterior and mode side by side |
| Plan Builder page | compose a plan, place things in the scene, generate a demo |
| Sandbox page | teleoperation by hand tracking |

## How the UI is composed

The front end (`src/cramera/web/`) is a set of **panels** mounted into layout slots;
`web/config.js` decides which panel appears where. Panels never call each other — they
publish and subscribe on an event bus (`web/core/bus.js`), so any subset works, and a
new visualization is one `Panels.define(...)` plus a line in the config.

```
src/cramera/
  server.py        static front end + JSON API (/api/knowledge, /api/eql, /scenes/)
  knowledge/       the recorded scene as an EQL knowledge base; presets; graph views
  live/            stream a running coraplex demo into the viewer (bridge on :8765)
  onboard/         turn a demo run into a scene bundle
  web/             index.html, config.js, core/, panels/, vendor/ (all local, no CDN)
```

## Tests

```bash
pytest test/cramera_test
```

The JS core is covered by node-based tests invoked from pytest (skipped when node is
unavailable).

## About this mirror

The source of truth is the monorepo branch
[`cramera-port`](https://github.com/sunava/cognitive_robot_abstract_machine/tree/cramera-port);
`.github/workflows/sync.yml` splits its `cramera/` directory into this repository's
`main` every six hours, and `pages.yml` publishes the demo site from it. Please open
issues and pull requests against the monorepo.

cramera is developed by [Vanessa Hassouna](https://github.com/sunava) at the Institute for
Artificial Intelligence, University of Bremen. GPL-3.0.
