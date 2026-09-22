# Changelog

What landed in cramera, by week and area. Entries are features, not commits; the
hashes point into the `cramera-port` branch of the monorepo.

## Week 36 · 31 Aug – 4 Sep 2026

### Plan Builder
- **A page that composes a plan and generates a runnable demo.** Drag steps into a
  plan, choose an environment, place the robot and the transport targets in the
  embedded live 3D scene, then generate a `RobotDemonstration` subclass in the
  output style you pick. The demo's run log is shown so a failure can be traced.
  (`205d79402`, `c47b507e1`, `caaa4bb67`, `89a9637a2`, `6261fb2c5`, `e86a79a04`, `3563737a0`)
- **Navigate steps show and capture the robot pose** as a ring and heading arrow on
  the floor. (`487bb1051`, `932644a4f`)
- **Transport steps know where they start and where they drop off**; a drop-off can
  be a semantic surface or container instead of coordinates.
  (`a18b55ebd`, `d522049af`, `6c86d9406`, `1226f653c`, `1890c9ad6`)
- **Objects are placed with sliders, numbers and "Drop to surface"**; new objects
  stage above the robot under a bobbing arrow. (`e3330c9bd`, `316bc29b1`, `92d74ec26`, `38ba4511c`, `520ee76cd`)
- **Every semantic_digital_twin robot and an all-STL industrial object set**; Garmi
  joins the list. (`40edf54b8`, `1827604c3`, `f55097eb4`, `52a548e19`, `18cd18b93`)
- Constraints in plain language on every plan step. (`d53f0aac0`)
- A big 3D scene with the controls in one dropdown; resizable columns; a reload
  button for the scene; visible feedback on Generate, Save and Download.
  (`2af538b25`, `805d8cbb9`, `2186d4121`, `3623aab48`, `ae75551a8`)

### Plan view
- **Plan steps read as a tree of phases with live status**; the graph view is gone,
  the step list is the Plan tab. (`e32c9ddf4`, `874eb01fc`, `0e8da5a51`, `52b2753b5`, `1ae600f4d`)
- **Constraints you type compile to giskardpy nodes against the live world**, from a
  palette inside the Steps view; the bridge accepts them while the plan runs.
  (`c67f5ce5c`, `1ba0b54a4`, `b0918c4eb`)
- The Plan tab attaches to the running bridge and shows motion-statechart rings.
  (`4dd70f65e`, `3e2d609b8`, `359422308`)
- More constraint goal types and a richer rule-based translation to English.
  (`896ccf0c6`, `8c74f6aea`)

### Questions
- **Ask the EQL console a question in English**: completion, matching to a preset,
  read-back and speech. (`b725dfa46`, `8d666e350`)
- **An answer says where it comes from and points at what it names**; a timestamped
  answer can replay its moment. (`62183bd31`, `241f6939e`, `bf25a199a`, `3714d66cc`)
- A query source can claim the bodies it describes under a prefix. (`8b7945244`)

### Recordings
- **Recordings carry the motion statecharts** (`statecharts.json`). (`c1d8fa054`)
- **Recordings carry what the detectors saw**: pick and place events from segmind,
  answerable from the saved scene without the detectors' package.
  (`932f187a6`, `08ec63b2e`, `1ba75a6b2`, `639acac4c`, `e97c698d8`, `479dd4820`)
- **A recording is saved by what it is** — robot, environment, task — and keeps the
  objects a run carried. (`47b156e63`, `3a2c114d4`, `120891514`)

### 3D scene
- The rectangle on the floor of every scene is gone: the SSAO pass no longer draws
  the scene background into its depth and normal buffers. (`993394afc`)
- A lone panel takes the whole window; overlays and question groups fold away.
  (`2eb61a848`, `c861b8299`, `10d050718`, `240f2f5ca`)
- The Teleoperation tab is now called Sandbox; the blue place target stays out of the
  embedded and live views. (`e63e0f917`, `98b1936a6`, `a141673a2`)

### Platform
- The browser no longer caches static assets; what the branch serves is tested. (`12deaee25`)
- A fresh checkout runs the generated demo without editing a venv path. (`f0741687a`)

## Week 34 · 17 – 18 Aug 2026

- Collada models load with their up axis normalized — contributed by Tigul (PR #38).
  (`d6cef3972`)

## Week 33 · 10 – 15 Aug 2026

### Recordings
- Replay a recorded run, with TF frames and keypoints; trim it before saving.
  (`2db5dd0de`, `1bd3a6fea`, `5438761d5`)
- Onboard demos whose worlds come from Gazebo/SDF or MJCF. (`d6d0ee546`, `cc6559743`)

### 3D scene
- ROS debug markers in the scene, with RViz-style settings. (`32fe7d1ff`, `3849c589f`)
- A View menu chooses which panels show and where; panels are draggable and the
  arrangement persists. (`2207d8c51`, `949829ebc`, `1841d1810`, `463991e6f`, `3a20488ea`)
- Loose objects spawned as primitive boxes are recorded and rendered; mesh rendering
  of bundled environments and multi-material robots is fixed. (`2c546d025`, `72bacdd6d`)

### Models
- The Models tab: the probabilistic-model workbench in the browser — Query,
  Posterior and Mode side by side, sliders for numeric constraints.
  (`de6d10091`, `c3b2c3d05`, `95ba8ef7f`, `aa756483f`, `d83816ac1`)

### Live view
- The live view is rebuilt on `WorldVisualization`; no more monkey patches.
  (`33ca53d4a`, `47700539a`, `6d6f543dd`, `f68daa892`)
- Worlds that are not fully XML can be viewed live. (`021c08f68`)

### Platform
- The cramera server opens the viewer; demos only connect to it. (`c1c07a398`)
- Several scenes per installation, selectable across backend and panels. (`4d2d025bb`, `0891832eb`)
- Graph panel: colors and groups, a plan legend, a four-level architecture scan.
  (`0f07103aa`, `1fa024a56`, `41a12b9f7`, `8696eac41`)
- Variable hints in the EQL query box; panels report their errors. (`7264cd6bc`, `ccdeea54a`)
- The project is named cramera. (`2851475ac`)
