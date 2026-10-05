/* ============================================================================
 * plan_builder.js — the Plan Builder page: compose a plan by drag-and-drop and
 * place objects in a top-down scene, then generate a runnable coraplex demo file.
 * A page script (owns the whole document), like models_page.js.
 * ==========================================================================*/
(function () {
  'use strict';

  // ---- available object meshes (coraplex/resources/objects) ----
  // kitchen items + an industrial/factory set (a robot carries these A->B on a shop floor)
  const MESHES = ['milk.stl', 'bowl.stl', 'spoon.stl', 'breakfast_cereal.stl', 'jeroen_cup.stl',
    'Static_CokeBottle.stl', 'big-knife.stl', 'whisk.stl', 'bread.stl', 'apartment_bowl.stl',
    'wrench.stl', 'axle.stl', 'plate.stl', 'base.stl', 'open_crate.stl',
    // a labelled cardboard box, the size of milk.stl (scripts/make_transport_box_mesh.py)
    'screw_box.obj'];
  const OBJ_COLORS = ['#e6ecff', '#e6c07f', '#9aa1ad', '#8fd6c8', '#c9a0ff', '#ff9db1', '#9ecb6b'];
  // the objects the catalog offers with an annotation class (see cramera.model_catalog):
  // spawned as annotations of that class, which a Detect step looks for
  let offeredObjects = [];

  // ---- skills ----
  const BLOCKS = {
    park_arms: { name: 'Park arms', color: '#b98cff', params: { arm: 'BOTH' } },
    move_torso: { name: 'Move torso', color: '#ff9db1', params: { torso: 'HIGH' } },
    navigate: { name: 'Navigate', color: '#8fd6c8', params: { x: 2.6, y: 1.8, z: 0.0, yaw: 0.0 } },
    look_at: { name: 'Look at', color: '#e6d36b', params: { x: 1.0, y: 0.0, z: 1.0 } },
    detect: { name: 'Detect', color: '#f0a35e', params: { object: '' } },
    transport: { name: 'Transport object', color: '#5b8cff', params: { object: '', x: 5.0, y: 3.3, z: 0.8, yaw: 1.57, arm: 'LEFT', targetMode: 'semantic', surfaceType: 'CounterTop', surfaceName: '' } },
    pick: { name: 'Pick up', color: '#7ec9ff', params: { object: '', arm: 'LEFT', perceive: false } },
    place: { name: 'Place', color: '#ffc46b', params: { object: '', x: 2.4, y: 1.8, z: 0.8, yaw: 0.0, arm: 'LEFT', targetMode: 'pose', surfaceType: 'CounterTop', surfaceName: '' } },
  };
  // which step kinds act on a placed object (see core/plan_steps.js); a Pick or Place is a
  // Transport spelled out, for a world whose floor carries no costmap to search
  const actsOnAnObject = window.PlanSteps.actsOnAnObject;
  const namesAnObject = window.PlanSteps.namesAnObject;
  const placesAnObject = window.PlanSteps.putsAnObjectDown;
  const placesAtASemanticTarget = window.PlanSteps.putsAnObjectDownAtASemanticTarget;
  const TORSO = ['HIGH', 'MID', 'LOW'];
  let builderState = new window.PlanBuilderState([]);
  let lastSpawnRobot = '';
  function showModelStatus() {
    const robot = robotInfo();
    $('pb-model-status').textContent = robot && builderState.failure(robot.name) || 'Robot model assets are checked when the scene starts.';
  }
  function reportSceneFailure(log) {
    const message = builderState.recordFailure(lastSpawnRobot, log || '');
    showModelStatus();
    liveStatus(message, 'err'); showScaffoldLog(log);
    toast(message, 'err');
  }
  function robotInfo() { const instance = builderState.activeRobot(); return builderState.robot(instance ? instance.model : $('pb-robot').value); }
  function robotArms() { const robot = robotInfo(); return robot ? robot.arms : []; }
  function loadCatalog() {
    ['pb-generate', 'pb-run', 'pb-live-start', 'pb-download', 'pb-save'].forEach(function (id) { $(id).disabled = true; });
    fetch('/api/plan/catalog').then(function (response) { return response.json(); }).then(function (catalog) {
      if (!catalog.ok) throw new Error(catalog.error || 'Model catalog unavailable');
      builderState = new window.PlanBuilderState(catalog.robots);
      builderState.offerEnvironments(catalog);
      if (!catalog.robots.length || !builderState.offeredEnvironments.length) throw new Error('No robot or environment descriptions are installed.');
      offeredObjects = catalog.objects || [];
      const robotSelect = $('pb-robot');
      robotSelect.replaceChildren();
      catalog.robots.forEach(function (robot) {
        const option = document.createElement('option'); option.value = robot.name; option.textContent = robot.name;
        robotSelect.appendChild(option);
      });
      const environmentSelect = $('pb-env');
      environmentSelect.replaceChildren();
      builderState.offeredEnvironments.forEach(function (environment) {
        const option = document.createElement('option'); option.value = window.PlanBuilderState.environmentValue(environment); option.textContent = environment.name;
        environmentSelect.appendChild(option);
        if (environment.path && environment.path.endsWith('/apartment.urdf')) environmentSelect.value = option.value;
      });
      builderState.addRobot(robotSelect.value, robotXY);
      renderRobotInstances(); renderBlocks(); showModelStatus();
      addStep('park_arms'); addStep('move_torso');
      ['pb-generate', 'pb-run', 'pb-live-start', 'pb-download', 'pb-save'].forEach(function (id) { $(id).disabled = false; });
      return openFromAddress();
    }).catch(function (error) { status('Cannot load CRAM models: ' + error.message, 'err'); });
  }
  function selectRobot() {
    const count = steps.length;
    const instance = builderState.activeRobot();
    if (instance) {
      instance.steps = steps;
      builderState.updateRobot(instance.id, {model: $('pb-robot').value});
    }
    steps = builderState.adaptSteps(steps, $('pb-robot').value);
    renderRobotInstances(); renderBlocks(); renderSteps(); reshowIfGenerated(); showModelStatus();
    if (count !== steps.length) status('Removed ' + (count - steps.length) + ' steps unavailable for ' + robotInfo().name, 'ok');
  }

  /** Present each independently placed robot and the selected robot's plan. */
  function renderRobotInstances() {
    const instance = builderState.activeRobot();
    if (!instance) return;
    const selector = $('pb-robot-instance');
    selector.replaceChildren();
    builderState.instances.forEach(function (robot) {
      const option = document.createElement('option');
      option.value = robot.id; option.textContent = robot.label + ' · ' + robot.model;
      selector.appendChild(option);
    });
    selector.value = instance.id;
    $('pb-robot-label').value = instance.label;
    $('pb-robot').value = instance.model;
    $('pb-rx').value = instance.x; $('pb-ry').value = instance.y;
    $('pb-ryaw').value = Math.round(instance.yaw * 180 / Math.PI * 100) / 100;
    $('pb-rtopic').value = instance.joint_state_topic || '';
    $('pb-rlocalize').value = instance.localization_topic || '';
    $('pb-rrepeat').checked = !!instance.repeats_plan;
    $('pb-remove-robot').disabled = builderState.instances.length <= 1;
    $('pb-plan-robot').textContent = 'Plan for ' + instance.label;
    robotXY = instance;
  }

  /** @param {string} identifier Stable instance selected by the author. */
  async function selectRobotInstance(identifier) {
    const previous = builderState.activeRobot();
    if (_busyTimer) {
      if (previous) $('pb-robot-instance').value = previous.id;
      status('Wait until the scene has started before selecting another robot.', 'err');
      return;
    }
    try {
      if (liveOn) {
        await synchronizeRobotPoses();
        const response = await fetch(bridgeUrl() + '/robot', {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify({identifier: identifier})});
        if (!response.ok) { const failure = await response.json(); throw new Error(failure.error || 'Robot selection unavailable while a plan runs'); }
      }
      const selected = builderState.selectInstance(identifier, steps);
      steps = selected.steps;
      renderRobotInstances(); renderBlocks(); renderSteps(); showModelStatus(); reshowIfGenerated();
    } catch (error) {
      if (previous) $('pb-robot-instance').value = previous.id;
      status(error.message, 'err');
    }
  }

  /** Add another independently placed instance of the selected installed model. */
  function addRobotInstance() {
    const previous = builderState.activeRobot();
    if (previous) previous.steps = steps;
    const position = window.PlanBuilderState.initialRobotPosition();
    builderState.addRobot($('pb-robot').value, {x: position.x, y: position.y + builderState.instances.length * 1.5, yaw: 0});
    steps = [];
    renderRobotInstances(); renderBlocks(); renderSteps(); showModelStatus();
    status('Robot added. Set its position, then start the live scene to apply scene changes.', 'ok');
    reshowIfGenerated();
  }

  /** Remove the selected robot while keeping at least one available instance. */
  function removeRobotInstance() {
    const selected = builderState.removeRobot(builderState.activeIdentifier);
    if (!selected) return;
    steps = selected.steps;
    renderRobotInstances(); renderBlocks(); renderSteps(); showModelStatus(); reshowIfGenerated();
    status('Robot removed. Start the live scene to apply scene changes.', 'ok');
  }

  /** @returns {Array<string>} Imports for all distinct robot models in the scene. */
  function robotImportLines() {
    const names = builderState.instances.map((instance) => instance.model);
    return Array.from(new Set((names.length ? names : [robotInfo().name]).map((name) => builderState.robot(name).import)));
  }

  /** @returns {Array<string>} Native shared-world configuration for authored instances. */
  function robotSceneLines() {
    if (!builderState.instances.length) return [];
    const lines = ['ROBOT_SCENE = RobotScene(', '    instances=['];
    builderState.instances.forEach(function (instance) {
      lines.push('        RobotInstance(');
      lines.push('            identifier=' + jsonStr(instance.id) + ', label=' + jsonStr(instance.label) + ',');
      lines.push('            robot_type=' + builderState.robot(instance.model).cls + ',');
      lines.push('            pose=HomogeneousTransformationMatrix.from_xyz_rpy(' + instance.x + ', ' + instance.y + ', 0.0, yaw=' + instance.yaw + '),');
      if (Object.keys(instance.joint_positions).length) lines.push('            joint_positions=' + jsonPy(instance.joint_positions) + ',');
      lines.push('        ),');
    });
    lines.push('    ],', '    active_identifier=' + jsonStr(builderState.activeIdentifier) + ',');
    if (Object.keys(builderState.environmentJointPositions).length) {
      lines.push('    environment_joint_positions=' + jsonPy(builderState.environmentJointPositions) + ',');
    }
    lines.push(')', '');
    return lines;
  }
  // semantic place targets: supporting surfaces ("on") and case containers ("in").
  // Both expose HasSupportingSurface.sample_points_from_surface, so resolution is identical.
  const SEMANTIC_SURFACES = ['CounterTop', 'Table', 'ShelfLayer', 'Floor', 'Sofa'];
  const SEMANTIC_CONTAINERS = ['Drawer', 'Fridge', 'Cabinet', 'Cupboard', 'Dresser', 'Dishwasher'];
  const SEMANTIC_TYPES = SEMANTIC_SURFACES.concat(SEMANTIC_CONTAINERS);
  function isContainer(t) { return SEMANTIC_CONTAINERS.indexOf(t) >= 0; }
  function prep(t) { return isContainer(t) ? 'in' : 'on'; }
  const DEFAULT_START = { x: 2.4, y: 2.2, z: 0.95, yaw: 0.0 };   // start pose used when an object was never placed/captured
  const ANGLE_KEYS = { roll: 1, pitch: 1, yaw: 1 };              // stored in radians, edited in degrees
  const CTL_LABEL = { x: 'X', y: 'Y', z: 'Z', roll: 'R', pitch: 'P', yaw: 'Y' };
  let liveSurfaces = [];   // [{type, name}] fetched from the live world when the scene runs

  // ---- constraints: plain sentences, compiled by core/plan_constraints.js ----
  let CONSTRAINTS = [
    { id: 'c1', text: 'Milk must always stay upright' },
    { id: 'c2', text: 'Robot must look where it operates' },
    { id: 'c3', text: 'Keep the bowl above the table' },
  ];
  let conSeq = 4;
  const CON_INFO_ROWS = [
    ['upright, level, flat, tilt, spill, steady, balanced', 'VectorsAligned', "keep the object's up-axis aligned with world up"],
    ['look, watch, observe, "keep in view", gaze, face', 'PointingAt', 'look at the object before picking it up and at the target before placing it — the only one the generated plan performs'],
    ['above, higher, "off the table", "keep high", lift', 'HeightMonitor', 'keep the object at/above a height'],
    ['below, under, "lower than", "keep low"', 'HeightMonitor', 'keep the object below a height'],
    ['"away from", clearance, distance, avoid, "keep clear"', 'DistanceMonitor', 'keep a minimum distance / clearance'],
  ];

  // ---- state ----
  let steps = [];       // [{type, params:{...}}]
  let objects = [];      // [{id, mesh, name, x, y, z, yaw, color}]
  let objSeq = 1, stepSeq = 1;
  let robotXY = window.PlanBuilderState.initialRobotPosition();   // robot spawn (draggable in the scene)
  let liveOn = false;                 // true while the scaffold scene is up (constraints can be pushed live)
  // a real robot's world comes from its world server holding only the robot, and only a map
  // can be spawned into that; a world file is only ever read into a simulated world
  const REAL_ROBOT_NEEDS_A_MAP = 'A real robot takes its world from its world server, so its demo needs a map environment (the real-lab apartment), not a world file.';

  // scene mapping: origin offset so the typical apartment area sits centred
  const SCALE = 40, ORIGIN_X = 2.5, ORIGIN_Y = 2.0;
  const $ = function (id) { return document.getElementById(id); };
  $('pb-rx').value = robotXY.x;
  $('pb-ry').value = robotXY.y;

  // ---------- palette ----------
  function renderBlocks() {
    const el = $('pb-blocks'); el.innerHTML = '';
    const robot = robotInfo();
    (robot ? robot.steps : []).forEach(function (k) {
      const b = BLOCKS[k];
      const d = document.createElement('button');
      d.type = 'button'; d.title = 'Add ' + b.name + ' to the plan';
      d.addEventListener('click', function () { addStep(k); });
      d.className = 'pb-block'; d.draggable = true; d.dataset.block = k;
      d.innerHTML = '<span class="ic" style="background:' + b.color + '"></span>' + b.name;
      d.addEventListener('dragstart', function (e) { e.dataTransfer.setData('text/plain', 'block:' + k); });
      el.appendChild(d);
    });
    const meshSel = $('pb-mesh'); meshSel.innerHTML = MESHES.map(function (m) { return '<option>' + m + '</option>'; })
      .concat(offeredObjects.map(function (t) { return '<option value="' + t.mesh + '">' + t.mesh + ' (' + t.cls + ')</option>'; })).join('');
  }

  // ---------- objects ----------
  // A clear, visible staging pose for a freshly added object: floating directly above the
  // robot, so it can never spawn hidden inside a box/cabinet or behind furniture (the robot
  // spot is collision-free and always in view). Objects stack upward and fan out slightly so
  // several don't overlap; you then drag each onto its real target (the drag snaps it down).
  function stagingPose() {
    const n = objects.length;
    const ang = n * 2.39996;                       // golden-angle spread so they fan out
    return {
      x: robotXY.x + Math.cos(ang) * 0.12,
      y: robotXY.y + Math.sin(ang) * 0.12,
      z: 1.9 + n * 0.16,                           // a little tower above the robot's head
    };
  }
  function addObject(mesh, opts) {
    opts = opts || {};
    const stage = stagingPose();
    const o = { id: 'o' + (objSeq++), mesh: mesh, name: mesh,
      x: opts.x != null ? opts.x : stage.x, y: opts.y != null ? opts.y : stage.y,
      z: opts.z != null ? opts.z : stage.z,
      roll: opts.roll != null ? opts.roll : 0.0, pitch: opts.pitch != null ? opts.pitch : 0.0,
      yaw: opts.yaw != null ? opts.yaw : 0.0,   // roll/pitch/yaw in radians (codegen uses radians)
      size: opts.size || null,                  // [sx, sy, sz] of a box the scene builds; null for a mesh
      poseOpen: false,                          // XYZ/RPY controls collapsed by default
      color: OBJ_COLORS[(objSeq) % OBJ_COLORS.length] };
    // an object the catalog offers with an annotation class carries the class along
    const typed = offeredObjects.find(function (t) { return t.mesh === mesh; });
    if (typed) { o.cls = typed.cls; o.import = typed.import; o.meshUrl = typed.mesh_url; }
    objects.push(o); renderObjects(); renderScene(); refreshObjectSelects();
    return o;
  }
  function renderObjects() {
    const el = $('pb-objects'); el.innerHTML = '';
    objects.forEach(function (o) {
      const d = document.createElement('div'); d.className = 'pb-obj';
      d.innerHTML =
        '<div class="row1"><span class="pb-swatch" style="background:' + o.color + '"></span>' +
        '<span class="oname" title="' + o.mesh + '">' + o.name + '</span></div>' +
        '<div class="pb-object-actions">' +
        '<button type="button" class="ocap" data-cap="' + o.id + '" aria-label="Capture current pose of ' + o.name + '" title="Use the current live pose as this object’s start pose">Capture pose</button>' +
        '<button type="button" class="oreset" data-reset="' + o.id + '" aria-label="Reset pose of ' + o.name + '" title="Move this object back to its authored coordinates">Reset</button>' +
        '<button type="button" class="odel" data-del="' + o.id + '" aria-label="Remove ' + o.name + '" title="Remove this object from the authored scene">Remove</button></div>' +
        '<button class="pb-pose-toggle" data-posetoggle="' + o.id + '">' + (o.poseOpen ? '▾' : '▸') + ' pose (xyz · rpy)</button>' +
        '<div class="pb-pose"' + (o.poseOpen ? '' : ' style="display:none"') + '>' +
        '<div class="pb-pose-grp"><span class="pb-pose-h">position (m)</span>' +
        ctl(o, 'x', -6, 6, 0.05) + ctl(o, 'y', -6, 6, 0.05) + ctl(o, 'z', 0, 3, 0.05) + '</div>' +
        '<div class="pb-pose-grp"><span class="pb-pose-h">rotation (rpy°)</span>' +
        ctl(o, 'roll', -180, 180, 1) + ctl(o, 'pitch', -180, 180, 1) + ctl(o, 'yaw', -180, 180, 1) + '</div>' +
        '</div>';
      el.appendChild(d);
    });
    // slider + number for the same field stay in sync; both write object state
    el.querySelectorAll('.pb-obj [data-oid]').forEach(function (inp) {
      inp.addEventListener('input', function () {
        const o = objects.find(function (x) { return x.id === inp.dataset.oid; }); if (!o) return;
        const k = inp.dataset.k, isAngle = ANGLE_KEYS[k];
        const raw = parseFloat(inp.value) || 0;
        o[k] = isAngle ? raw * Math.PI / 180 : raw;                 // store angles in radians
        // sync the sibling control (the other input for the same field)
        inp.parentNode.querySelectorAll('[data-k="' + k + '"]').forEach(function (other) {
          if (other !== inp) other.value = inp.value;
        });
        renderScene();
        pushObjectPose(o);                                          // move it live so you see it
      });
    });
    el.querySelectorAll('.odel').forEach(function (x) {
      x.addEventListener('click', function () { objects = objects.filter(function (o) { return o.id !== x.dataset.del; }); renderObjects(); renderScene(); refreshObjectSelects(); });
    });
    el.querySelectorAll('.ocap').forEach(function (x) {
      x.addEventListener('click', function () { captureObject(x.dataset.cap); });
    });
    el.querySelectorAll('.oreset').forEach(function (x) {
      x.addEventListener('click', function () { resetObject(x.dataset.reset); });
    });
    el.querySelectorAll('[data-posetoggle]').forEach(function (b) {
      b.addEventListener('click', function () {
        const o = objects.find(function (x) { return x.id === b.dataset.posetoggle; });
        if (o) { o.poseOpen = !o.poseOpen; renderObjects(); }
      });
    });
  }
  // move an object in the live 3D scene back to its builder coordinates (undo a bad snap)
  function resetObject(oid) {
    const o = objects.find(function (x) { return x.id === oid; }); if (!o) return;
    // tell the embedded 3D scene to move the mesh back (the idle sim won't apply a
    // queued /move, so a visual reset must go through the viewer itself)
    const q = rpyToQuat(o.roll, o.pitch, o.yaw);
    builderState.authoredPosition(o.mesh, [o.x, o.y, o.z], q);
    const f = $('pb-3d');
    if (f && f.contentWindow) f.contentWindow.postMessage(
      { type: 'cramera-reset-object', key: o.mesh, position: [o.x, o.y, o.z], quaternion: q }, '*');
    // also update the bridge's last-move overlay so a later capture reads the reset pose
    fetch(bridgeUrl() + '/move', { method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ object: o.mesh, position: [o.x, o.y, o.z], quaternion: q, final: true }) })
      .then(function () { status('reset ' + o.name + ' in the 3D scene → (' + o.x + ', ' + o.y + ', ' + o.z + ')', 'ok'); })
      .catch(function () { status('reset failed — start the live scene first', 'err'); });
  }
  function resetAllObjects() { objects.forEach(function (o) { resetObject(o.id); }); }
  // live-sync an object's position to the 3D scene as the sliders/fields change (rotation
  // is applied in the generated demo / on the next scene start). postMessage moves the mesh
  // smoothly; the /move fetch is throttled so we don't spam the bridge.
  // roll/pitch/yaw (rad, ROS/URDF convention) -> quaternion [x, y, z, w]
  function rpyToQuat(r, p, y) {
    const cr = Math.cos(r / 2), sr = Math.sin(r / 2);
    const cp = Math.cos(p / 2), sp = Math.sin(p / 2);
    const cy = Math.cos(y / 2), sy = Math.sin(y / 2);
    return [sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy];
  }
  let _lastPosePush = 0;
  function pushObjectPose(o) {
    const q = rpyToQuat(o.roll, o.pitch, o.yaw);
    builderState.authoredPosition(o.mesh, [o.x, o.y, o.z], q);
    const f = $('pb-3d');
    if (f && f.contentWindow) f.contentWindow.postMessage(
      { type: 'cramera-reset-object', key: o.mesh, position: [o.x, o.y, o.z], quaternion: q }, '*');
    const now = Date.now();
    if (now - _lastPosePush < 120) return;
    _lastPosePush = now;
    fetch(bridgeUrl() + '/move', { method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ object: o.mesh, position: [o.x, o.y, o.z], quaternion: q, final: true }) }).catch(function () {});
  }
  // ask the embedded 3D view to flag every builder object with a bobbing arrow, so staged
  // objects (which spawn lifted, beside the robot) are easy to find; the arrow clears once
  // the object is grabbed. Applied on each scene load and whenever the object set changes.
  function highlightObjectsInScene() {
    const f = $('pb-3d');
    if (f && f.contentWindow) f.contentWindow.postMessage(
      { type: 'cramera-highlight-objects', keys: objects.map(function (o) { return o.mesh; }) }, '*');
  }
  // show every Navigate step's target as a ground arrow (position + yaw) in the 3D view
  function sendNavigateTargets() {
    const f = $('pb-3d'); if (!f || !f.contentWindow) return;
    const targets = steps.filter(function (s) { return s.type === 'navigate'; }).map(function (s, i) {
      return { id: s.id, label: 'nav ' + (i + 1), x: s.params.x, y: s.params.y, z: s.params.z, yaw: s.params.yaw };
    });
    f.contentWindow.postMessage({ type: 'cramera-navigate-targets', targets: targets }, '*');
  }
  // capture the live robot's current base pose as this Navigate step's goal
  function captureNavigate(sid) {
    const s = steps.find(function (x) { return x.id === sid; }); if (!s) return;
    fetch(bridgeUrl() + '/state').then(function (r) { return r.ok ? r.json() : null; }).then(function (d) {
      const b = d && d.base;
      if (!b || b.length < 7) { status('no live robot pose — start the scene first', 'err'); return; }
      s.params.x = Math.round(b[0] * 100) / 100; s.params.y = Math.round(b[1] * 100) / 100;
      s.params.z = Math.round(b[2] * 100) / 100; s.params.yaw = r3(quatToYaw(b.slice(3)));
      renderSteps();
      toast('captured robot pose → navigate goal (' + s.params.x + ', ' + s.params.y + ')', 'ok');
    }).catch(function () { status('capture failed — start the live scene first', 'err'); });
  }
  // physics-ish "drop": ask the 3D view to let every object fall straight down onto the
  // nearest surface below it (raycast). The viewer reports each settled pose back, which
  // we write into the object cards so the generated demo spawns them resting on the surface.
  function dropObjects() {
    const f = $('pb-3d');
    if (!f || !f.contentWindow) { toast('start the live scene first', 'err'); return; }
    f.contentWindow.postMessage({ type: 'cramera-settle-objects', keys: objects.map(function (o) { return o.mesh; }) }, '*');
    toast('Dropping objects onto the nearest surface…', 'ok');
  }
  // ---------- standing the robot where the floor is clicked ----------
  // The tool arms the 3D view for one pick: press where the robot should stand, drag the
  // way it should face, release. The view answers with the pressed floor point in the map
  // frame and the heading dragged, if any; the active robot's fields take them, and a
  // running scene stands the robot there as a typed position would. Then the tool is off.
  let floorPlacementArmed = false;
  function armFloorPlacement(on) {
    floorPlacementArmed = !!on;
    const button = $('pb-place-robot');
    if (button) button.setAttribute('aria-pressed', floorPlacementArmed ? 'true' : 'false');
    const f = $('pb-3d');
    if (f && f.contentWindow) f.contentWindow.postMessage({ type: 'cramera-pick-floor', on: floorPlacementArmed }, '*');
  }
  function toggleFloorPlacement() {
    const instance = builderState.activeRobot(); if (!instance) return;
    armFloorPlacement(!floorPlacementArmed);
    if (floorPlacementArmed) status('press on the floor of the 3D scene where ' + instance.label + ' should stand, drag the way it should face, release', 'ok');
  }
  function standActiveRobotAt(x, y, yaw) {
    const instance = builderState.activeRobot(); if (!instance) return;
    const changes = {x: Math.round(x * 100) / 100, y: Math.round(y * 100) / 100};
    if (Number.isFinite(yaw)) changes.yaw = yaw;          // a plain click keeps the heading
    builderState.updateRobot(instance.id, changes);
    renderRobotInstances(); reshowIfGenerated();
    placeActiveRobotLive();
  }
  // the 3D view reports poses back (settle / drag-release / floor pick); write them into
  // the object cards, the steps or the active robot
  window.addEventListener('message', function (ev) {
    if (ev && ev.data) handleSceneMessage(ev.data);
  });
  function handleSceneMessage(d) {
    if (d.type === 'cramera-object-settled' && d.key && Array.isArray(d.position)) {
      const o = objects.find(function (x) { return x.mesh === d.key; }); if (!o) return;
      builderState.authoredPosition(d.key, d.position);
      o.x = d.position[0];
      o.y = d.position[1];
      o.z = d.position[2];
      renderObjects();
    } else if (d.type === 'cramera-navigate-moved' && d.id) {
      // a Navigate goal was dragged in the scene -> save into THAT step
      const s = steps.find(function (x) { return x.id === d.id; }); if (!s || s.type !== 'navigate') return;
      s.params.x = Math.round(d.x * 100) / 100; s.params.y = Math.round(d.y * 100) / 100;
      if (d.final) renderSteps();          // persist + re-sync fields + re-emit the marker
      else syncStepNum(s.id);              // live: just update the number fields (don't rebuild the marker mid-drag)
    } else if (d.type === 'cramera-floor-picked' && floorPlacementArmed) {
      armFloorPlacement(false);
      standActiveRobotAt(d.x, d.y, d.yaw);
    }
  }
  // one pose control = a slider + a number input, kept in sync. Angles are shown in
  // degrees (state stores radians); position in metres.
  function ctl(o, k, min, max, step) {
    const isAngle = ANGLE_KEYS[k];
    const v = isAngle ? Math.round(o[k] * 180 / Math.PI) : Math.round(o[k] * 100) / 100;
    return '<label class="pb-ctl"><span class="pb-ctl-k">' + CTL_LABEL[k] + '</span>' +
      '<input class="pb-slider" type="range" data-oid="' + o.id + '" data-k="' + k + '" min="' + min + '" max="' + max + '" step="' + step + '" value="' + v + '">' +
      '<input class="pb-num" data-oid="' + o.id + '" data-k="' + k + '" type="number" step="' + step + '" value="' + v + '"></label>';
  }

  // ---------- constraints palette ----------
  function renderConstraints() {
    const el = $('pb-cons'); if (!el) return;
    el.innerHTML = CONSTRAINTS.map(function (c) {
      const comp = PlanConstraints.compile(c.text, null);
      const badge = comp.goal ? '<span class="pb-con-goal" title="translates to giskardpy ' + comp.goal + '">' + comp.goal + '</span>'
        : '<span class="pb-con-goal nomatch" title="no rule matched — this text will not translate to a goal">no match</span>';
      return '<div class="pb-con" draggable="true" data-cid="' + c.id + '">' +
        '<span class="pb-con-grip">⠿</span><span class="pb-con-txt">' + c.text + '</span>' + badge +
        '<span class="pb-con-del" data-del="' + c.id + '">×</span></div>';
    }).join('');
    el.querySelectorAll('.pb-con').forEach(function (card) {
      card.addEventListener('dragstart', function (e) { e.dataTransfer.setData('text/plain', 'con:' + card.dataset.cid); e.dataTransfer.effectAllowed = 'copy'; });
    });
    el.querySelectorAll('.pb-con-del').forEach(function (x) {
      x.addEventListener('click', function (e) { e.stopPropagation(); CONSTRAINTS = CONSTRAINTS.filter(function (c) { return c.id !== x.dataset.del; }); renderConstraints(); });
    });
  }
  function addConstraintText(txt) {
    const v = String(txt || '').trim(); if (!v) return;
    CONSTRAINTS.push({ id: 'c' + (conSeq++), text: v }); renderConstraints();
  }
  // attach a constraint (by palette id) to a plan step
  function attachConstraint(stepId, cid) {
    const s = steps.find(function (x) { return x.id === stepId; });
    const c = CONSTRAINTS.find(function (x) { return x.id === cid; });
    if (!s || !c) return;
    const comp = PlanConstraints.compile(c.text, s);
    if (!comp.goal) { status('“' + c.text + '” — no rule matched, not attached', 'err'); return; }
    s.constraints = s.constraints || [];
    if (s.constraints.some(function (a) { return a.text === c.text; })) { status('already attached to this step', ''); return; }
    const attached = { text: c.text, goal: comp.goal, params: comp.params, stepArgument: comp.stepArgument };
    s.constraints.push(attached);
    renderSteps();
    if (attached.stepArgument) status('attached “' + c.text + '” → ' + attached.stepArgument + ' on the generated step'
      + (liveOn ? ' — start the scene again to run it' : ''), 'ok');
    else if (liveOn) pushConstraintLive(s, attached);
    else status('attached “' + c.text + '” → ' + comp.goal + ' — only the live scene applies this one', 'ok');
  }
  function detachConstraint(stepId, idx) {
    const s = steps.find(function (x) { return x.id === stepId; }); if (!s || !s.constraints) return;
    s.constraints.splice(idx, 1); renderSteps();
  }
  // push a constraint to the running scaffold's bridge (same endpoint the Plan view uses)
  function pushConstraintLive(s, a) {
    const b = BLOCKS[s.type];
    const body = { op: 'attach_monitor', text: a.text, apply: 'next_activation',
      target_plan_node: { id: s.id, kind: s.type, label: (b ? b.name : s.type) },
      giskard_node: { type: a.goal, params: a.params } };
    fetch(bridgeUrl() + '/constraint', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })
      .then(function (r) { return r.json(); })
      .then(function (j) {
        if (j && j.ok) status('attached “' + a.text + '” → ' + a.goal + ' — queued in the live plan (next activation)', 'ok');
        else status('live attach failed: ' + ((j && j.error) || '?'), 'err');
      })
      .catch(function (e) { status('live attach failed: ' + e, 'err'); });
  }
  function conInfoHtml() {
    const rows = CON_INFO_ROWS.map(function (r) {
      return '<tr><td>' + r[0] + '</td><td class="goal">' + r[1] + '</td><td>' + r[2] + '</td></tr>';
    }).join('');
    return '<div class="ci-h">How constraints are translated <span class="ci-note">(rule-based, not an LLM)</span></div>' +
      '<table class="ci-table"><thead><tr><th>Phrasing</th><th>giskardpy goal</th><th>Effect</th></tr></thead><tbody>' + rows + '</tbody></table>' +
      '<div class="ci-foot">A length in the text (<code>10 cm</code>, <code>0.1 m</code>) sets the thresholds. ' +
      'The object comes from the sentence or, on a Transport step, its transported object. ' +
      'The look-at is generated onto the Transport step itself; every other goal needs the live scene, since no coraplex action enforces it yet.</div>';
  }

  // ---------- scene (top-down) ----------
  function worldToPx(x, y) {
    const sc = $('pb-scene'); const w = sc.clientWidth, h = sc.clientHeight;
    return { px: w / 2 + (x - ORIGIN_X) * SCALE, py: h / 2 - (y - ORIGIN_Y) * SCALE };
  }
  function pxToWorld(px, py) {
    const sc = $('pb-scene'); const w = sc.clientWidth, h = sc.clientHeight;
    return { x: ORIGIN_X + (px - w / 2) / SCALE, y: ORIGIN_Y - (py - h / 2) / SCALE };
  }
  function renderScene() {
    const sc = $('pb-scene');
    if (!sc) return;   // the 2D scene was replaced by the live 3D capture flow
    sc.querySelectorAll('.pb-marker,.pb-tmarker').forEach(function (m) { m.remove(); });
    // robot spawn (draggable)
    const rp = worldToPx(robotXY.x, robotXY.y);
    const rm = document.createElement('div'); rm.className = 'pb-marker pb-rm';
    rm.style.left = rp.px + 'px'; rm.style.top = rp.py + 'px'; rm.style.background = '#38405c'; rm.style.fontSize = '13px';
    rm.innerHTML = '<span class="lbl">robot</span>🤖';
    rm.addEventListener('mousedown', function (e) {
      dragMarker(e, rm, function (x, y) { robotXY.x = x; robotXY.y = y; }, function () {});
    });
    sc.appendChild(rm);
    // objects (draggable)
    objects.forEach(function (o) {
      const p = worldToPx(o.x, o.y);
      const m = document.createElement('div'); m.className = 'pb-marker';
      m.style.left = p.px + 'px'; m.style.top = p.py + 'px'; m.style.background = o.color;
      m.innerHTML = '<span class="lbl">' + o.name.replace(/\.stl$/i, '') + '</span>' + o.name.charAt(0).toUpperCase();
      m.addEventListener('mousedown', function (e) {
        dragMarker(e, m, function (x, y) { o.x = x; o.y = y; syncNum(o.id); }, function () { renderObjects(); });
      });
      sc.appendChild(m);
    });
    // transport destinations (ghost target, draggable) — "where the object should go"
    steps.forEach(function (s, i) {
      if (!placesAnObject(s)) return;
      const p = worldToPx(s.params.x, s.params.y);
      const m = document.createElement('div'); m.className = 'pb-marker pb-tmarker';
      m.style.left = p.px + 'px'; m.style.top = p.py + 'px';
      m.innerHTML = '<span class="lbl">→ ' + (s.params.object ? s.params.object.replace(/\.stl$/i, '') : 'step ' + (i + 1)) + '</span>◎';
      m.addEventListener('mousedown', function (e) {
        dragMarker(e, m, function (x, y) { s.params.x = x; s.params.y = y; syncStepNum(s.id); }, function () { renderSteps(); });
      });
      sc.appendChild(m);
    });
  }
  function dragMarker(e, m, apply, onEnd) {
    e.preventDefault();
    const sc = $('pb-scene');
    function move(ev) {
      const r = sc.getBoundingClientRect();
      const px = Math.max(0, Math.min(r.width, ev.clientX - r.left));
      const py = Math.max(0, Math.min(r.height, ev.clientY - r.top));
      m.style.left = px + 'px'; m.style.top = py + 'px';
      const w = pxToWorld(px, py); apply(Math.round(w.x * 100) / 100, Math.round(w.y * 100) / 100);
    }
    function up() { document.removeEventListener('mousemove', move); document.removeEventListener('mouseup', up); onEnd(); }
    document.addEventListener('mousemove', move); document.addEventListener('mouseup', up);
  }
  function syncNum(oid) {
    document.querySelectorAll('.pb-num[data-oid="' + oid + '"]').forEach(function (inp) {
      const o = objects.find(function (x) { return x.id === oid; }); if (!o) return;
      if (inp.dataset.k === 'x') inp.value = o.x; if (inp.dataset.k === 'y') inp.value = o.y;
    });
  }
  function syncStepNum(sid) {
    document.querySelectorAll('.pb-num[data-sid="' + sid + '"]').forEach(function (inp) {
      const s = steps.find(function (x) { return x.id === sid; }); if (!s) return;
      if (inp.dataset.k === 'x') inp.value = s.params.x; if (inp.dataset.k === 'y') inp.value = s.params.y;
    });
  }

  // ---------- plan steps ----------
  function addStep(type) {
    const b = BLOCKS[type], robot = robotInfo();
    if (!b || !robot || robot.steps.indexOf(type) < 0) return;
    const params = Object.assign({}, b.params);
    if (params.arm && robot.arms.indexOf(params.arm) < 0) params.arm = robot.arms[0];
    const candidates = detectableObjects({ type: type });
    if (namesAnObject({ type: type }) && !params.object && candidates.length) params.object = candidates[0].mesh;
    // a new Navigate starts as a copy of the last one (offset a bit), so its marker appears
    // next to the previous goal and can be dragged from there instead of jumping to a default
    if (type === 'navigate') {
      const prev = steps.filter(function (s) { return s.type === 'navigate'; }).pop();
      if (prev) { params.x = prev.params.x + 0.4; params.y = prev.params.y; params.z = prev.params.z; params.yaw = prev.params.yaw; }
    }
    steps.push({ id: 's' + (stepSeq++), type: type, params: params });
    renderSteps();
  }
  function renderSteps() {
    const el = $('pb-steps');
    $('pb-step-count').textContent = steps.length ? '(' + steps.length + ')' : '';
    if (!steps.length) { el.innerHTML = '<div class="pb-drop-hint">Drop skills here to build the sequence</div>'; renderScene(); return; }
    el.innerHTML = '';
    steps.forEach(function (s, i) {
      const b = BLOCKS[s.type];
      const d = document.createElement('div'); d.className = 'pb-step'; d.style.borderLeftColor = b.color;
      d.dataset.sid = s.id;
      d.innerHTML =
        '<div class="sh"><span class="snum">' + (i + 1) + '</span><span class="sname">' + b.name + '</span>' +
        '<span class="sctl"><button data-up="' + s.id + '" title="Move up">↑</button>' +
        '<button data-down="' + s.id + '" title="Move down">↓</button>' +
        '<button data-del="' + s.id + '" title="Remove">×</button></span></div>' +
        stepChips(s) +
        '<div class="sparams">' + stepParams(s) + '</div>';
      el.appendChild(d);
    });
    wireStepEvents();
    renderScene();
    sendNavigateTargets();
  }
  function stepChips(s) {
    const cs = s.constraints || [];
    if (!cs.length) return '';
    return '<div class="sconstraints">' + cs.map(function (a, idx) {
      return '<span class="scon-chip" title="' + a.text + ' → giskardpy ' + a.goal + '">⛓ ' + a.text +
        '<span class="scon-goal">' + a.goal + '</span>' +
        '<span class="scon-x" data-scon-del="' + s.id + '" data-scon-idx="' + idx + '">×</span></span>';
    }).join('') + '</div>';
  }
  function row(html) { return '<div class="sparam-row">' + html + '</div>'; }
  function stepParams(s) {
    if (s.type === 'park_arms') return row(sel(s, 'arm', robotArms()));
    if (s.type === 'move_torso') return row(sel(s, 'torso', TORSO));
    if (s.type === 'look_at') return row('<span class="pb-group-lbl">look at →</span>' + num(s, 'x') + num(s, 'y') + num(s, 'z'));
    if (s.type === 'navigate') return row('<span class="pb-group-lbl">go to →</span>' + num(s, 'x') + num(s, 'y') + num(s, 'z') + num(s, 'yaw') +
      '<button class="pb-capbtn" data-capnav="' + s.id + '" title="drive/place the robot in the 3D scene, then capture its base pose as this navigate goal">◎ capture robot pose</button>');
    if (s.type === 'transport') {
      return (
        row(objSel(s)) +
        row('<span class="pb-group-lbl start">start (from) →</span>' + startCaptureButton(s)) +
        row('<span class="pb-group-lbl">target →</span>' + modeSel(s)) +
        dropOffRow(s) +
        row(sel(s, 'arm', robotArms()))
      );
    }
    if (s.type === 'detect') {
      return (
        row(objSel(s)) +
        row('<span class="pb-hint3">looks for an object of this one\'s class through perception and takes its pose from there — only an object with a class (shown in brackets) can be detected</span>')
      );
    }
    if (s.type === 'pick') {
      return (
        row(objSel(s)) +
        row('<span class="pb-group-lbl start">start (from) →</span>' + startCaptureButton(s)) +
        row(sel(s, 'arm', robotArms())) +
        row(chk(s, 'perceive', 'perceive before grasping (look at the object and detect it first)')) +
        row('<span class="pb-hint3">the robot grasps from where it stands — put a Navigate step in front of this one</span>')
      );
    }
    if (s.type === 'place') {
      return (
        row(objSel(s)) +
        row('<span class="pb-group-lbl">target →</span>' + modeSel(s)) +
        dropOffRow(s) +
        row(sel(s, 'arm', robotArms())) +
        row('<span class="pb-hint3">places what this arm is holding — put a Pick step in front of this one</span>')
      );
    }
    return '';
  }
  // where a step puts the object down: a semantic location, or an exact pose to capture
  function dropOffRow(s) {
    if ((s.params.targetMode || 'pose') === 'semantic') {
      return row('<span class="pb-group-lbl">place →</span>' + semanticTypeSel(s) + surfaceInstanceSel(s));
    }
    return row('<span class="pb-group-lbl">drop-off (to) →</span>' + num(s, 'x') + num(s, 'y') + num(s, 'z') + num(s, 'yaw') +
      '<button class="pb-capbtn" data-capstep="' + s.id + '" title="drag the object to its drop-off in the 3D scene, then capture that pose as this step\'s target">◎ capture</button>');
  }
  function startCaptureButton(s) {
    return '<button class="pb-capbtn start" data-capstart="' + s.id +
      '" title="drag the object to its START in the 3D scene, then capture that as its start pose (shown on the object card)">◎ capture</button>';
  }
  function num(s, k) { return '<label>' + k.toUpperCase() + '<input class="pb-num xyz" data-sid="' + s.id + '" data-k="' + k + '" type="number" step="0.05" value="' + s.params[k] + '"></label>'; }
  // a yes/no parameter of a step
  function chk(s, k, label) {
    return '<label class="pb-check"><input class="pb-chk" type="checkbox" data-sid="' + s.id + '" data-k="' + k + '"' + (s.params[k] ? ' checked' : '') + '> ' + label + '</label>';
  }
  function sel(s, k, opts) { return '<label>' + k + '<select class="pb-sel" data-sid="' + s.id + '" data-k="' + k + '">' + opts.map(function (o) { return '<option' + (s.params[k] === o ? ' selected' : '') + '>' + o + '</option>'; }).join('') + '</select></label>'; }
  // a select whose option values differ from their labels: pairs = [[value, label], ...]
  function selPairs(s, k, pairs) {
    return '<select class="pb-sel" data-sid="' + s.id + '" data-k="' + k + '">' + pairs.map(function (p) {
      return '<option value="' + p[0] + '"' + ((s.params[k] || '') === p[0] ? ' selected' : '') + '>' + p[1] + '</option>';
    }).join('') + '</select>';
  }
  function modeSel(s) { return selPairs(s, 'targetMode', [['semantic', 'semantic location'], ['pose', 'exact pose (XYZ)']]); }
  // semantic type dropdown, grouped into "on a surface" / "in a container"
  function semanticTypeSel(s) {
    function grp(label, types) {
      return '<optgroup label="' + label + '">' + types.map(function (t) {
        return '<option value="' + t + '"' + ((s.params.surfaceType || '') === t ? ' selected' : '') + '>' + prep(t) + ' ' + t + '</option>';
      }).join('') + '</optgroup>';
    }
    return '<select class="pb-sel" data-sid="' + s.id + '" data-k="surfaceType">' +
      grp('on a surface', SEMANTIC_SURFACES) + grp('in a container', SEMANTIC_CONTAINERS) + '</select>';
  }
  // instance dropdown: automatic selection + live instances of the chosen type
  function surfaceInstanceSel(s) {
    const t = s.params.surfaceType || 'CounterTop';
    const inst = liveSurfaces.filter(function (x) { return x.type === t; });
    const pairs = [['', 'automatic (nearest first)']].concat(inst.map(function (x) { return [x.name, x.name]; }));
    return selPairs(s, 'surfaceName', pairs);
  }
  // the placed objects a step may name: a detect needs one with an annotation class
  function detectableObjects(s) {
    return window.PlanSteps.looksForAnObject(s) ? objects.filter(function (o) { return !!o.cls; }) : objects;
  }
  function objSel(s) {
    const offered = detectableObjects(s);
    // keep the param in sync with the visibly-selected first option, so capture works
    // even for a Transport step whose object dropdown was never touched
    if (!s.params.object && offered.length) s.params.object = offered[0].mesh;
    const opts = offered.map(function (o) { return '<option value="' + o.mesh + '"' + (s.params.object === o.mesh ? ' selected' : '') + '>' + o.name + (o.cls ? ' (' + o.cls + ')' : '') + '</option>'; }).join('');
    return '<label>object<select class="pb-sel" data-sid="' + s.id + '" data-k="object">' + (opts || '<option value="">— add an object' + (window.PlanSteps.looksForAnObject(s) ? ' with a class' : '') + ' —</option>') + '</select></label>';
  }
  function wireStepEvents() {
    const el = $('pb-steps');
    el.querySelectorAll('.pb-num,.pb-sel').forEach(function (inp) {
      inp.addEventListener('input', function () {
        const s = steps.find(function (x) { return x.id === inp.dataset.sid; }); if (!s) return;
        const k = inp.dataset.k;
        const v = inp.classList.contains('pb-num') ? (parseFloat(inp.value) || 0) : inp.value;
        s.params[k] = v;
        // switching target mode / surface type swaps which fields are shown -> re-render
        if (k === 'surfaceType') { s.params.surfaceName = ''; renderSteps(); }
        else if (k === 'targetMode') { renderSteps(); }
        else { renderScene(); if (s.type === 'navigate') sendNavigateTargets(); }
      });
    });
    el.querySelectorAll('.pb-chk').forEach(function (inp) {
      inp.addEventListener('change', function () {
        const s = steps.find(function (x) { return x.id === inp.dataset.sid; }); if (!s) return;
        s.params[inp.dataset.k] = inp.checked;
      });
    });
    el.querySelectorAll('[data-del]').forEach(function (b) { b.addEventListener('click', function () { steps = steps.filter(function (s) { return s.id !== b.dataset.del; }); renderSteps(); }); });
    el.querySelectorAll('[data-up]').forEach(function (b) { b.addEventListener('click', function () { moveStep(b.dataset.up, -1); }); });
    el.querySelectorAll('[data-down]').forEach(function (b) { b.addEventListener('click', function () { moveStep(b.dataset.down, 1); }); });
    el.querySelectorAll('[data-capstep]').forEach(function (b) { b.addEventListener('click', function (e) { e.preventDefault(); captureStepTarget(b.dataset.capstep); }); });
    el.querySelectorAll('[data-capstart]').forEach(function (b) { b.addEventListener('click', function (e) { e.preventDefault(); captureStepStart(b.dataset.capstart); }); });
    el.querySelectorAll('[data-capnav]').forEach(function (b) { b.addEventListener('click', function (e) { e.preventDefault(); captureNavigate(b.dataset.capnav); }); });
    // remove an attached constraint chip
    el.querySelectorAll('[data-scon-del]').forEach(function (x) {
      x.addEventListener('click', function (e) { e.stopPropagation(); detachConstraint(x.dataset.sconDel, parseInt(x.dataset.sconIdx, 10)); });
    });
    // each step is a drop target for a constraint card
    el.querySelectorAll('.pb-step').forEach(function (st) {
      st.addEventListener('dragover', function (e) {
        // the dragged payload isn't readable during dragover, so allow the drop and
        // decide on drop() below (only con: payloads actually attach)
        e.preventDefault(); st.classList.add('con-drop');
      });
      st.addEventListener('dragleave', function () { st.classList.remove('con-drop'); });
      st.addEventListener('drop', function (e) {
        st.classList.remove('con-drop');
        const d = e.dataTransfer.getData('text/plain') || '';
        if (d.indexOf('con:') === 0) { e.preventDefault(); e.stopPropagation(); attachConstraint(st.dataset.sid, d.slice(4)); }
      });
    });
  }
  function moveStep(id, dir) {
    const i = steps.findIndex(function (s) { return s.id === id; }); const j = i + dir;
    if (i < 0 || j < 0 || j >= steps.length) return;
    const t = steps[i]; steps[i] = steps[j]; steps[j] = t; renderSteps();
  }
  function refreshObjectSelects() { renderSteps(); }

  // drop zone
  const stepsEl = $('pb-steps');
  stepsEl.addEventListener('dragover', function (e) { e.preventDefault(); stepsEl.classList.add('drop-ok'); });
  stepsEl.addEventListener('dragleave', function () { stepsEl.classList.remove('drop-ok'); });
  stepsEl.addEventListener('drop', function (e) {
    e.preventDefault(); stepsEl.classList.remove('drop-ok');
    const d = e.dataTransfer.getData('text/plain') || '';
    if (d.indexOf('block:') === 0) addStep(d.slice(6));
  });

  // ---------- code generation ----------
  function py(v) { return (Math.round(v * 1000) / 1000).toString(); }
  function jsonStr(s) { return JSON.stringify(String(s)); }
  // python literal for a constraint param value (list / string / number)
  function jsonPy(v) {
    if (Array.isArray(v)) return '[' + v.map(jsonPy).join(', ') + ']';
    if (typeof v === 'object' && v) return '{' + Object.keys(v).map(function (k) { return jsonStr(k) + ': ' + jsonPy(v[k]); }).join(', ') + '}';
    if (typeof v === 'string') return jsonStr(v);
    return String(v);
  }
  // --- "place on a surface": symbolic target resolution via semantic_digital_twin ---
  function surfaceSteps(useSteps) {
    return useSteps.filter(placesAtASemanticTarget);
  }
  function surfaceTypesUsed(useSteps) {
    const set = {}; surfaceSteps(useSteps).forEach(function (s) { set[s.params.surfaceType || 'CounterTop'] = 1; });
    return Object.keys(set);
  }
  // Candidate locations remain lazy until CRAM grounds each action at execution time.
  function surfaceResolveLines(useSteps, indent) {
    return surfaceSteps(useSteps).map(function (s) {
      const type = s.params.surfaceType || 'CounterTop';
      const name = s.params.surfaceName ? jsonStr(s.params.surfaceName) : 'None';
      const provider = 'PlacementSurface(' +
        'world=world, body=' + body(s.params.object || 'object') +
        ', surface_type=' + type + ', surface_name=' + name + ')';
      return indent + '_target_' + s.id + ' = ' +
        (s.type === 'transport' ? provider : 'variable(Pose, domain=' + provider + ')');
    });
  }
  // objects to spawn: the placed ones, plus any object a transport step references but that
  // was never placed/captured — spawned at DEFAULT_START so the demo still runs.
  // lines that resolve each Pick step's object and the grasp to take it with, given
  // `world` and `context`. The side to approach from is left to the robot's reach rather
  // than spelled out here, so a rotated object is still grasped from a side it can stand on.
  function pickGraspLines(useSteps, indent) {
    const L = [];
    useSteps.forEach(function (s) {
      if (s.type !== 'pick') return;
      const mesh = s.params.object || 'object';
      L.push(indent + '# pick "' + mesh + '" with the ' + s.params.arm.toLowerCase() + ' arm');
      // an object with a class is resolved once as its annotation; a plain one is its body
      const picked = annotatedObject(mesh) ? objectVariable(annotatedObject(mesh)) + '.root' : '_pick_' + s.id;
      if (!annotatedObject(mesh)) L.push(indent + '_pick_' + s.id + ' = ' + body(mesh));
      L.push(indent + '_grasp_' + s.id + ' = GraspDescription.robot_relative_default(');
      L.push(indent + '    ViewManager.get_end_effector_view(Arms.' + s.params.arm + ', context.robot),');
      L.push(indent + '    ' + picked + '.global_pose,');
      L.push(indent + '    ' + picked + ',');
      L.push(indent + ')');
    });
    return L;
  }
  // --- objects with an annotation class: spawned as annotations, found by their class ---
  // the placed object of that mesh, if it carries an annotation class
  function annotatedObject(mesh) {
    return objects.find(function (o) { return o.mesh === mesh && o.cls; }) || null;
  }
  // the python name an annotated object is resolved into, from its mesh: "_cheeze_it"
  function objectVariable(o) {
    return '_' + o.mesh.replace(/\.[^.]+$/, '').replace(/[^a-z0-9_]/gi, '_').toLowerCase();
  }
  // the annotated objects the plan names, each resolved once by its class
  function annotatedObjectLines(useSteps, indent) {
    const seen = {};
    const L = [];
    useSteps.forEach(function (s) {
      const o = namesAnObject(s) ? annotatedObject(s.params.object) : null;
      if (!o || seen[o.mesh]) return;
      seen[o.mesh] = 1;
      L.push(indent + objectVariable(o) + ' = world.get_semantic_annotations_by_type(' + o.cls + ')[0]');
    });
    return L;
  }
  // the class a Detect step looks for; a detect of an object without a class has nothing to look for
  function detectedClass(s) {
    const o = annotatedObject(s.params.object);
    if (!o) throw new RangeError('A Detect step needs an object with an annotation class; "' + (s.params.object || '') + '" has none.');
    return o.cls;
  }
  function detectSteps(useSteps) { return useSteps.filter(window.PlanSteps.looksForAnObject); }
  function annotatedObjects(useSteps) { return effectiveObjects(useSteps).filter(function (o) { return !!o.cls; }); }
  function plainObjects(useSteps) { return effectiveObjects(useSteps).filter(function (o) { return !o.cls; }); }
  // what a demo with annotated objects or a Detect step imports on top of the rest
  function objectImportLines(useSteps) {
    const L = [];
    Array.from(new Set(annotatedObjects(useSteps).map(function (o) { return o.import; }))).forEach(function (line) { L.push(line); });
    if (annotatedObjects(useSteps).length) L.push('from semantic_digital_twin.adapters.package_resolver import CompositePathResolver');
    if (detectSteps(useSteps).length) L.push('from coraplex.robot_plans.actions.core.misc import DetectAction');
    return L;
  }
  // the enum members a demo imports: the arms always, the detection technique for a Detect step
  function enumImportLine(useSteps, members) {
    const names = ['Arms'].concat(detectSteps(useSteps).length ? ['DetectionTechnique'] : []).concat(members);
    return 'from coraplex.datastructures.enums import ' + names.join(', ');
  }
  // the module constant listing the annotated objects: (class, mesh, mesh url, x, y, z, roll, pitch, yaw)
  function annotatedObjectsConstant(useSteps) {
    const annotated = annotatedObjects(useSteps);
    if (!annotated.length) return [];
    const L = ['# objects placed in the Plan Builder with an annotation class: (class, mesh, mesh url, x, y, z, roll, pitch, yaw)', 'ANNOTATED_OBJECTS = ['];
    annotated.forEach(function (o) {
      L.push('    (' + o.cls + ', ' + jsonStr(o.mesh) + ', ' + jsonStr(o.meshUrl) + ', ' + py(o.x) + ', ' + py(o.y) + ', ' + py(o.z) +
        ', ' + py(o.roll) + ', ' + py(o.pitch) + ', ' + py(o.yaw) + '),');
    });
    L.push(']', '');
    return L;
  }
  // spawning each annotated object as an annotation of its class, free to move
  function annotatedObjectSpawnLines(useSteps, indent) {
    if (!annotatedObjects(useSteps).length) return [];
    return [
      indent + '# each object with a class is spawned as its annotation, free to move (Connection6DoF),',
      indent + '# so a detection can write the perceived pose to it',
      indent + 'for annotation, mesh, mesh_url, x, y, z, roll, pitch, yaw in ANNOTATED_OBJECTS:',
      indent + '    annotation.get_annotation_specification(',
      indent + '        mesh,',
      indent + '        BodySpecification.mesh(',
      indent + '            mesh,',
      indent + '            CompositePathResolver().resolve(mesh_url),',
      indent + '            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(',
      indent + '                x, y, z, roll=roll, pitch=pitch, yaw=yaw),',
      indent + '        ),',
      indent + '        parent_connection_specification=Connection6DoFSpecification(),',
      indent + '    ).spawn(world)',
    ];
  }
  // the condition under which every authored object already stands in the world
  function objectsPopulatedTerms(useSteps, environmentTerm) {
    const terms = [];
    if (environmentTerm) terms.push(environmentTerm);
    else if (!annotatedObjects(useSteps).length) terms.push('bool(OBJECTS)');
    terms.push('all(len(world.get_bodies_by_name(spec[0])) == 1 for spec in OBJECTS)');
    if (annotatedObjects(useSteps).length) terms.push('all(len(world.get_bodies_by_name(spec[1])) == 1 for spec in ANNOTATED_OBJECTS)');
    return terms;
  }
  function effectiveObjects(useSteps) {
    const list = objects.slice();
    const have = {}; list.forEach(function (o) { have[o.mesh] = 1; });
    useSteps.forEach(function (s) {
      if (actsOnAnObject(s) && s.params.object && !have[s.params.object]) {
        have[s.params.object] = 1;
        list.push({ mesh: s.params.object, name: s.params.object,
          x: DEFAULT_START.x, y: DEFAULT_START.y, z: DEFAULT_START.z,
          roll: 0.0, pitch: 0.0, yaw: DEFAULT_START.yaw,
          color: '#cccccc', _defaulted: true });
      }
    });
    return list;
  }
  function surfaceImportLine(useSteps) {
    const types = surfaceTypesUsed(useSteps);
    if (!types.length) return null;
    return 'from semantic_digital_twin.semantic_annotations.semantic_annotations import ' + types.sort().join(', ') + '\n' +
      'from cramera.live.placement_surface import PlacementSurface\n' +
      'from semantic_digital_twin.semantic_annotations.mixins import HasRootBody\n' +
      'from krrood.entity_query_language.factories import a, variable';
  }
  // every constraint attached anywhere in the plan
  function attachedConstraints(useSteps) {
    const all = [];
    useSteps.forEach(function (s) { (s.constraints || []).forEach(function (a) { all.push(a); }); });
    return all;
  }
  // the constraints the generated plan cannot enforce on its own, listed as metadata so
  // the demo still records what was asked for and the live bridge can pick them up
  function constraintBlock(useSteps) {
    const liveOnly = attachedConstraints(useSteps).filter(function (a) { return !a.stepArgument; });
    if (!liveOnly.length) return [];
    const L = [];
    L.push('# --- constraints the generated plan does not enforce ---');
    L.push('# These have no coraplex action behind them yet, so they only apply when this');
    L.push('# demo runs under `cramera-live` and the viewer pushes them to the bridge.');
    L.push('CONSTRAINTS = [');
    useSteps.forEach(function (s, i) {
      (s.constraints || []).filter(function (a) { return !a.stepArgument; }).forEach(function (a) {
        L.push('    {"step": ' + (i + 1) + ', "text": ' + jsonStr(a.text) +
          ', "goal": ' + jsonStr(a.goal) + ', "params": ' + jsonPy(a.params) + '},');
      });
    });
    L.push(']');
    L.push('');
    return L;
  }
  // the environment the generated demo is built in
  function selectedEnvironment() {
    const value = ($('pb-env') && $('pb-env').value) || 'apartment.urdf';
    const environment = builderState.environment(value);
    if (!environment) throw new RangeError('The environment ' + value + ' is not offered.');
    return environment;
  }
  // the python both generators write for an environment: what to import, the module
  // constant naming it, how a world is built in it, how it is spawned into a world that
  // came without it (a real robot's world server serves only the robot) and how such a
  // world is recognized. A world file is read by CRAM's parser and reasoned about once
  // read; a map is a class that populates the world itself, annotated as it is spawned,
  // so it has no file, no _WORLDS, no placement annotations and no reasoning.
  function environmentCode(environment) {
    const isMap = window.PlanBuilderState.isMap(environment);
    const worldsPath = function (fileExpression) { return 'os.path.join(_WORLDS, ' + fileExpression + ')'; };
    return {
      isMap: isMap,
      importLines: isMap ? [environment.import] : [
        'from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner',
        'from cramera.live.placement_annotations import PlacementAnnotations',
      ],
      // the module-level paths and constants: where the world files lie, and which one is read
      constantLines: isMap ? [] : ['_WORLDS = os.path.join(_HERE, "..", "..", "resources", "worlds")'],
      fileConstantLines: isMap ? [] : ['ENV_FILE = ' + jsonStr(environment.path)],
      // the world of a scene of authored robots
      sceneWorldLine: function (indent, fileExpression) {
        return indent + 'return ROBOT_SCENE.build_world(' + (isMap ? '' : worldsPath(fileExpression)) + ')';
      },
      // the world of one robot: spawned into the parsed file, or into an empty world the map fills
      specificationLines: function (indent, fileExpression, robotLines) {
        const opening = isMap
          ? [indent + 'return WorldSpecification(']
          : [indent + 'return WorldSpecification.from_urdf(', indent + '    ' + worldsPath(fileExpression) + ','];
        return opening.concat([indent + '    robots=['], robotLines, [indent + '    ],', indent + ').to_domain_object()']);
      },
      // spawning the map into a world that lacks it
      populateLines: function (indent) { return isMap ? [indent + environment.cls + '().populate(world)'] : []; },
      // the condition under which a world already holds the environment, or null for a file, which is always read in
      populatedExpression: isMap ? environment.cls + '.is_populated(world)' : null,
      // reasoning about the world read from a file, and the placement annotations of the
      // bundled apartment file; a real robot's served world is left as it is
      reasoningLines: function (indent, fileExpression, guarded) {
        if (isMap) return [];
        const inner = guarded ? indent + '    ' : indent;
        return (guarded ? [indent + 'if self.execution_type is not ExecutionType.REAL:'] : []).concat([
          inner + 'with world.modify_world():',
          inner + '    WorldReasoner(world).reason()',
          inner + '    PlacementAnnotations(world, ' + worldsPath(fileExpression) + ').apply()',
        ]);
      },
    };
  }
  function pose(p) { return 'Pose.from_xyz_rpy(' + py(p.x) + ', ' + py(p.y) + ', ' + py(p.z) + ', yaw=' + py(p.yaw) + ', reference_frame=world.root)'; }
  function point(p) { return 'Pose.from_xyz_rpy(' + py(p.x) + ', ' + py(p.y) + ', ' + py(p.z) + ', reference_frame=world.root)'; }
  function body(mesh) {
    const o = annotatedObject(mesh);
    return o ? objectVariable(o) + '.root' : 'world.get_body_by_name("' + mesh + '")';
  }
  function generate(stepsOverride) {
    const useSteps = stepsOverride || steps;
    const added = plainObjects(useSteps);
    const environment = selectedEnvironment();
    const code = environmentCode(environment);
    const R = robotInfo();
    const L = [];
    L.push('"""Generated by the cramera Plan Builder."""');
    L.push('import os');
    L.push('from coraplex.datastructures.dataclasses import Context');
    L.push(enumImportLine(useSteps, ['VisualizationBackend']));
    L.push('from coraplex.datastructures.grasp import GraspDescription');
    L.push('from coraplex.execution_environment import ' + executionEnvironment().name);
    L.push('from coraplex.plans.factories import sequential');
    L.push('from coraplex.visualization import WorldVisualization');
    L.push('from coraplex.robot_plans.actions.composite.transporting import TransportAction');
    L.push('from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction');
    L.push('from coraplex.robot_plans.actions.core.pick_up import PickUpAction');
    L.push('from coraplex.robot_plans.actions.core.placing import PlaceAction');
    L.push('from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction');
    L.push('from coraplex.view_manager import ViewManager');
    L.push('from semantic_digital_twin.adapters.mesh import DAEParser, OBJParser, STLParser');
    L.push('from semantic_digital_twin.api import ' + (annotatedObjects(useSteps).length ? 'BodySpecification, Connection6DoFSpecification, ' : '') + 'RobotSpecification, WorldSpecification');
    L.push('from semantic_digital_twin.datastructures.definitions import TorsoState');
    if (!surfaceSteps(useSteps).length) L.push('from semantic_digital_twin.semantic_annotations.mixins import HasRootBody');
    code.importLines.forEach(function (line) { L.push(line); });
    objectImportLines(useSteps).forEach(function (line) { L.push(line); });
    if (window.BaseControl.pinsTheSetting(baseControl())) {
      L.push('from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase');
    }
    robotImportLines().forEach((line) => L.push(line));
    if (builderState.instances.length) L.push('from cramera.multi_robot import RobotInstance, RobotScene');
    L.push('from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix');
    L.push('from semantic_digital_twin.spatial_types.spatial_types import Pose');
    L.push('from semantic_digital_twin.world import World');
    L.push('from semantic_digital_twin.world_description.geometry import Color');
    if (environmentGeometryLines().length) L.push('from cramera.body_geometry import DrawnGeometry');
    const _surfImp = surfaceImportLine(useSteps);
    if (_surfImp) L.push(_surfImp);
    L.push('');
    L.push('_HERE = os.path.dirname(__file__)');
    code.constantLines.forEach(function (line) { L.push(line); });
    L.push('_OBJECTS = os.path.join(_HERE, "..", "..", "resources", "objects")');
    L.push('_MESH_PARSERS = {".stl": STLParser, ".obj": OBJParser, ".dae": DAEParser}');
    L.push('');
    annotatedObjectsConstant(useSteps).forEach(function (ln) { L.push(ln); });
    L.push('');
    L.push('def _parse_mesh(mesh: str) -> World:');
    L.push('    """Parse an object mesh into its own world.');
    L.push('');
    L.push('    :param mesh: Object filename in the installed mesh directory.');
    L.push('    :return: World containing the parsed object.');
    L.push('    """');
    L.push('    ext = os.path.splitext(mesh)[1].lower()');
    L.push('    return _MESH_PARSERS.get(ext, STLParser)(os.path.join(_OBJECTS, mesh)).parse()');
    L.push('');
    L.push('');
    robotSceneLines().forEach((line) => L.push(line));
    L.push('def build_world(' + (code.isMap ? '' : 'env_file: str, ') + 'robot_xy: tuple[float, float]) -> World:');
    L.push('    """Build the environment and spawn its annotated robot.');
    L.push('');
    if (!code.isMap) L.push('    :param env_file: Environment URDF filename or absolute path.');
    L.push('    :param robot_xy: Robot starting x and y position in metres.');
    L.push('    :return: The assembled semantic world' + (code.isMap ? ', still to be populated with the map' : '') + '.');
    L.push('    """');
    if (builderState.instances.length) {
      L.push(code.sceneWorldLine('    ', 'env_file'));
    } else {
      code.specificationLines('    ', 'env_file', [
        '        RobotSpecification(',
        '            semantic_annotation_type=' + R.cls + ',',
        '            world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(',
        '                robot_xy[0], robot_xy[1], 0.0),',
        '        ),',
      ]).forEach(function (ln) { L.push(ln); });
    }
    L.push('');
    L.push('');
    baseControlConstant().forEach(function (ln) { L.push(ln); });
    if (baseControlConstant().length) L.push('');
    L.push('world = build_world(' + (code.isMap ? '' : jsonStr(environment.path) + ', ') + '(' + py(robotXY.x) + ', ' + py(robotXY.y) + '))');
    code.populateLines('').forEach(function (ln) { L.push(ln); });
    L.push('visualization = WorldVisualization.from_environment(');
    L.push('    world, default_backend=VisualizationBackend.CRAMERA).start()');
    environmentGeometryLines().forEach(function (ln) { L.push(ln); });
    L.push('');
    if (added.length) {
      L.push('# --- objects placed in the Plan Builder ---');
      added.forEach(function (o, i) {
        L.push('_obj' + i + ' = _parse_mesh("' + o.mesh + '")');
      });
      L.push('with world.modify_world():');
      added.forEach(function (o, i) {
        L.push('    world.merge_world_at_pose(_obj' + i + ', HomogeneousTransformationMatrix.from_xyz_rpy(');
        L.push('        ' + py(o.x) + ', ' + py(o.y) + ', ' + py(o.z) +
          ', roll=' + py(o.roll) + ', pitch=' + py(o.pitch) + ', yaw=' + py(o.yaw) + ', reference_frame=world.root))');
      });
      added.forEach(function (o) {
        const c = hexToRgb(o.color);
        L.push(body(o.mesh) + '.visual.shapes[0].color = Color(' + c[0] + ', ' + c[1] + ', ' + c[2] + ')');
      });
      L.push('');
    }
    annotatedObjectSpawnLines(useSteps, '').forEach(function (ln) { L.push(ln); });
    if (annotatedObjects(useSteps).length) L.push('');
    if (!useSteps.length) {
      // a plan without steps only shows the world, so no motion has to be feasible in it
      code.reasoningLines('', jsonStr(environment.path), false).forEach(function (ln) { L.push(ln); });
      L.push('');
      return L.join('\n');
    }
    L.push(builderState.instances.length ? 'robot = ROBOT_SCENE.selected_robot(world)' : 'robot = world.get_semantic_annotations_by_type(' + R.cls + ')[0]');
    baseControlLines('').forEach(function (ln) { L.push(ln); });
    L.push('context = Context(world=world, robot=robot, _debug=False, ros_node=visualization.ros_node)');
    code.reasoningLines('', jsonStr(environment.path), false).forEach(function (ln) { L.push(ln); });
    L.push('context.evaluate_conditions = False');
    L.push('');
    annotatedObjectLines(useSteps, '').forEach(function (ln) { L.push(ln); });
    surfaceResolveLines(useSteps, '').forEach(function (ln) { L.push(ln); });
    pickGraspLines(useSteps, '').forEach(function (ln) { L.push(ln); });
    if (annotatedObjectLines(useSteps, '').length || surfaceSteps(useSteps).length || pickGraspLines(useSteps, '').length) L.push('');
    L.push('plan = sequential([');
    useSteps.forEach(function (s) { L.push('    ' + stepCode(s) + ','); });
    L.push('], context=context).plan');
    L.push('visualization.attach_plan(plan)');
    L.push('');
    constraintBlock(useSteps).forEach(function (ln) { L.push(ln); });
    L.push('with ' + executionEnvironment().name + ':');
    L.push('    plan.perform()');
    L.push('');
    return L.join('\n');
  }
  // the lines drawing the environment as the setup says, when not as it looks
  function environmentGeometryLines() {
    const geometry = window.PlanBuilderState.DRAWN_GEOMETRY;
    if (builderState.environmentGeometry !== geometry.COLLISION) return [];
    return [
      'if visualization.cramera_visualization is not None:',
      '    visualization.cramera_visualization.bridge.environment_geometry = DrawnGeometry.COLLISION',
    ];
  }
  function stepCode(s) {
    const p = s.params;
    if (s.type === 'park_arms') return 'ParkArmsAction(Arms.' + p.arm + ')';
    if (s.type === 'move_torso') return 'MoveTorsoAction(TorsoState.' + p.torso + ')';
    if (s.type === 'navigate') return 'NavigateAction(' + pose(p) + ')';
    if (s.type === 'look_at') return 'LookAtAction(' + point(p) + ')';
    if (s.type === 'transport') {
      const given = ['object_designator=HasRootBody(root=' + body(p.object || 'object') + ')',
        'target_location=' + dropOffTarget(s), 'arm=Arms.' + p.arm]
        .concat(PlanConstraints.stepArguments(s.constraints || []));
      return 'TransportAction(' + given.join(', ') + ')';
    }
    if (s.type === 'detect') {
      return 'DetectAction(DetectionTechnique.TYPES, object_sem_annotation=' + detectedClass(s) + ', accept_first_if_multiple=True)';
    }
    if (s.type === 'pick') {
      const o = annotatedObject(p.object);
      const picked = o ? objectVariable(o) : 'HasRootBody(root=_pick_' + s.id + ')';
      return 'PickUpAction(' + picked + ', Arms.' + p.arm + ', _grasp_' + s.id + (p.perceive ? ', perceive_before_grasp=True' : '') + ')';
    }
    if (s.type === 'place') {
      const action = placesAtASemanticTarget(s) ? 'a(PlaceAction)' : 'PlaceAction';
      return action + '(object_designator=' + body(p.object || 'object') +
        ', target_location=' + dropOffTarget(s) + ', arm=Arms.' + p.arm + ')';
    }
    return 'None';
  }
  // the pose expression a step puts the object at: the sampled semantic target, or the pose
  function dropOffTarget(s) {
    return (s.params.targetMode === 'semantic') ? ('_target_' + s.id) : pose(s.params);
  }
  function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255].map(function (v) { return Math.round(v * 100) / 100; }); }

  // PascalCase class name from the demo name (e.g. "my_demo" -> "MyDemoDemonstration")
  function className() {
    const base = ($('pb-name').value || 'my_demo').replace(/[^a-z0-9_\- ]/gi, '').replace(/[_\- ]+/g, ' ').trim();
    const cc = base.split(' ').map(function (w) { return w ? w.charAt(0).toUpperCase() + w.slice(1) : ''; }).join('');
    return (cc || 'MyDemo') + 'Demonstration';
  }
  // ---- output style: a coraplex.demonstrations.RobotDemonstration subclass ----
  function generateClass(stepsOverride) {
    const useSteps = stepsOverride || steps;
    const added = plainObjects(useSteps);
    const environment = selectedEnvironment();
    const code = environmentCode(environment);
    const drives = executionType();
    if (drives.drivesARealRobot && !code.isMap) throw new RangeError(REAL_ROBOT_NEEDS_A_MAP);
    const cls = className();
    const R = robotInfo();
    const L = [];
    L.push('"""Generated by the cramera Plan Builder — a RobotDemonstration subclass."""');
    L.push('import os');
    L.push('from dataclasses import dataclass');
    L.push('');
    L.push('from coraplex.datastructures.dataclasses import Context');
    L.push(enumImportLine(useSteps, ['ExecutionType', 'VisualizationBackend']));
    L.push('from coraplex.datastructures.grasp import GraspDescription');
    L.push('from coraplex.demonstrations import RobotDemonstration');
    L.push('from coraplex.plans.factories import sequential');
    L.push('from coraplex.plans.plan import Plan');
    L.push('from coraplex.robot_plans.actions.composite.transporting import TransportAction');
    L.push('from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction');
    L.push('from coraplex.robot_plans.actions.core.pick_up import PickUpAction');
    L.push('from coraplex.robot_plans.actions.core.placing import PlaceAction');
    L.push('from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction');
    L.push('from coraplex.view_manager import ViewManager');
    L.push('from semantic_digital_twin.api import (');
    L.push('    BodySpecification,');
    L.push('    Connection6DoFSpecification,');
    L.push('    RobotSpecification,');
    L.push('    WorldSpecification,');
    L.push(')');
    L.push('from semantic_digital_twin.datastructures.definitions import TorsoState');
    if (!surfaceSteps(useSteps).length) L.push('from semantic_digital_twin.semantic_annotations.mixins import HasRootBody');
    code.importLines.forEach(function (line) { L.push(line); });
    objectImportLines(useSteps).forEach(function (line) { L.push(line); });
    if (window.BaseControl.pinsTheSetting(baseControl())) {
      L.push('from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase');
    }
    robotImportLines().forEach((line) => L.push(line));
    if (builderState.instances.length) L.push('from cramera.multi_robot import RobotInstance, RobotScene');
    L.push('from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix');
    L.push('from semantic_digital_twin.spatial_types.spatial_types import Pose');
    L.push('from semantic_digital_twin.world import World');
    L.push('from semantic_digital_twin.world_description.geometry import Color');
    const _surfImpC = surfaceImportLine(useSteps);
    if (_surfImpC) L.push(_surfImpC);
    L.push('');
    L.push('_HERE = os.path.dirname(__file__)');
    code.constantLines.forEach(function (line) { L.push(line); });
    L.push('_OBJECTS = os.path.join(_HERE, "..", "..", "resources", "objects")');
    L.push('');
    code.fileConstantLines.forEach(function (line) { L.push(line); });
    L.push('ROBOT_XY = (' + py(robotXY.x) + ', ' + py(robotXY.y) + ')');
    robotSceneLines().forEach((line) => L.push(line));
    baseControlConstant().forEach(function (ln) { L.push(ln); });
    L.push('');
    L.push('# objects placed in the Plan Builder: (mesh, x, y, z, roll, pitch, yaw, (r, g, b))');
    L.push('OBJECTS = [');
    added.forEach(function (o) {
      const c = hexToRgb(o.color);
      L.push('    ("' + o.mesh + '", ' + py(o.x) + ', ' + py(o.y) + ', ' + py(o.z) +
        ', ' + py(o.roll) + ', ' + py(o.pitch) + ', ' + py(o.yaw) +
        ', (' + c[0] + ', ' + c[1] + ', ' + c[2] + ')),');
    });
    L.push(']');
    L.push('');
    annotatedObjectsConstant(useSteps).forEach(function (ln) { L.push(ln); });
    constraintBlock(useSteps).forEach(function (ln) { L.push(ln); });
    L.push('');
    L.push('@dataclass(kw_only=True)');
    L.push('class ' + cls + '(RobotDemonstration):');
    L.push('    """A demonstration composed in the cramera Plan Builder."""');
    L.push('');
    L.push('    def build_simulated_world(self) -> World:');
    L.push('        """Build the selected environment with its annotated robot.');
    L.push('');
    L.push('        :return: The assembled semantic world' + (code.isMap ? ', which populate_scene fills with the map' : '') + '.');
    L.push('        """');
    if (builderState.instances.length) {
      L.push(code.sceneWorldLine('        ', 'ENV_FILE'));
    } else {
      code.specificationLines('        ', 'ENV_FILE', [
        '                RobotSpecification(',
        '                    semantic_annotation_type=self.used_robot,',
        '                    world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(',
        '                        ROBOT_XY[0], ROBOT_XY[1], 0.0),',
        '                ),',
      ]).forEach(function (ln) { L.push(ln); });
    }
    L.push('');
    L.push('    def is_scene_populated(self, world: World) -> bool:');
    L.push('        """Check whether ' + (code.isMap ? 'the map and ' : '') + 'every authored object exists in the world.');
    L.push('');
    L.push('        :param world: World inspected for the authored ' + (code.isMap ? 'furniture and ' : '') + 'objects.');
    L.push('        :return: Whether the scene already contains the complete object set.');
    L.push('        """');
    L.push('        return ' + objectsPopulatedTerms(useSteps, code.populatedExpression).join(' and '));
    L.push('');
    L.push('    def populate_scene(self, world: World) -> None:');
    L.push('        """Spawn ' + (code.isMap ? 'the map and ' : '') + 'the authored objects at their saved poses.');
    L.push('');
    L.push('        :param world: World receiving the ' + (code.isMap ? 'furniture and the ' : '') + 'movable object bodies.');
    L.push('        """');
    code.populateLines('        ').forEach(function (ln) { L.push(ln); });
    L.push('        # each object is free to move (Connection6DoF), so the robot can transport it');
    L.push('        for mesh, x, y, z, roll, pitch, yaw, rgb in OBJECTS:');
    L.push('            BodySpecification.mesh(');
    L.push('                mesh,');
    L.push('                os.path.join(_OBJECTS, mesh),');
    L.push('                color=Color(*rgb),');
    L.push('                parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(');
    L.push('                    x, y, z, roll=roll, pitch=pitch, yaw=yaw),');
    L.push('                connection_specification=Connection6DoFSpecification(),');
    L.push('            ).spawn(world)');
    annotatedObjectSpawnLines(useSteps, '        ').forEach(function (ln) { L.push(ln); });
    L.push('');
    L.push('    def build_context(self, world: World) -> Context:');
    L.push('        """' + (code.isMap ? 'Configure robot execution in the map.' : 'Reason about a simulated world and configure robot execution.'));
    L.push('');
    L.push('        :param world: Semantic world used for planning and execution.');
    L.push('        :return: Context for the selected robot.');
    L.push('        """');
    code.reasoningLines('        ', 'ENV_FILE', true).forEach(function (ln) { L.push(ln); });
    // a real robot's world server names the robot as it likes, not by the scene's instance
    // namespace, so a demo for the real robot finds it by its type
    const byInstance = builderState.instances.length && !drives.drivesARealRobot;
    L.push(byInstance ? '        robot = ROBOT_SCENE.selected_robot(world)' : '        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]');
    baseControlLines('        ').forEach(function (ln) { L.push(ln); });
    L.push('        context = Context(');
    L.push('            world=world,');
    L.push('            robot=robot,');
    L.push('            _debug=False,');
    L.push('            ros_node=self.ros_node,');
    L.push('            alternative_motion_mappings=self.alternative_motion_mappings,');
    L.push('        )');
    L.push('        context.evaluate_conditions = False');
    L.push('        return context');
    L.push('');
    L.push('    def build_plan(self, context: Context) -> Plan:');
    L.push('        """Compose the authored action sequence against the current world.');
    L.push('');
    L.push('        :param context: Robot and world used to resolve the actions.');
    L.push('        :return: Executable plan containing the authored action sequence.');
    L.push('        """');
    L.push('        world = context.world  # bodies/poses below are resolved against it');
    annotatedObjectLines(useSteps, '        ').forEach(function (ln) { L.push(ln); });
    surfaceResolveLines(useSteps, '        ').forEach(function (ln) { L.push(ln); });
    pickGraspLines(useSteps, '        ').forEach(function (ln) { L.push(ln); });
    L.push('        return sequential([');
    useSteps.forEach(function (s) { L.push('            ' + stepCode(s) + ','); });
    L.push('        ], context=context).plan');
    L.push('');
    L.push('');
    L.push('def main(execution_type: ExecutionType = ExecutionType.' + drives.name + ') -> None:');
    L.push('    """Run the demonstration.');
    L.push('');
    L.push('    RobotDemonstration.run() acquires the world, starts the visualization backend,');
    L.push('    attaches the plan and performs it. The backend defaults to CRAMERA (the browser');
    L.push('    viewer); CORAPLEX_VISUALIZATION overrides it, so `cramera-live` works unchanged');
    L.push('    and you can also force RVIZ / NONE from the outside.');
    L.push('');
    L.push('    :param execution_type: Whether to drive the real robot or simulate it.');
    L.push('    """');
    L.push('    ' + cls + '(');
    L.push('        used_robot=' + R.cls + ',');
    L.push('        execution_type=execution_type,');
    L.push('        collision_avoidance=' + (executionEnvironment().collisionAvoidance ? 'True' : 'False') + ',');
    L.push('        default_visualization_backend=VisualizationBackend.CRAMERA,');
    L.push('    ).run()');
    L.push('');
    L.push('');
    L.push('if __name__ == "__main__":');
    L.push('    main()');
    L.push('');
    return L.join('\n');
  }
  // pick the generator by the selected output style
  function outputStyle() { const s = $('pb-style'); return s ? s.value : 'script'; }
  // the execution environment the generated demo performs its plan in
  function executionEnvironment() {
    const s = $('pb-collisions');
    return window.ExecutionEnvironments.byName(s ? s.value : null);
  }
  // whether the generated RobotDemonstration drives the real robot or a simulated one
  function executionType() {
    const s = $('pb-execution');
    return window.ExecutionTypes.byName(s ? s.value : null);
  }
  // whether the generated demo lets the base drive while an arm reaches
  function baseControl() {
    const s = $('pb-base');
    return window.BaseControl.byName(s ? s.value : null);
  }
  // the module-level constant a pinned base-control choice is written as
  function baseControlConstant() {
    const choice = baseControl();
    if (!window.BaseControl.pinsTheSetting(choice)) return [];
    return [
      '# whether the base may drive to help an arm reach (whole-body control). A plan',
      '# built in the Plan Builder says where the robot stands, with its Navigate steps.',
      'BASE_MAY_DRIVE_WHILE_REACHING = ' + (choice.fullBodyControlled ? 'True' : 'False'),
    ];
  }
  // the line applying it to the robot, once the robot is resolved
  function baseControlLines(indent) {
    if (!window.BaseControl.pinsTheSetting(baseControl())) return [];
    return [
      indent + 'if isinstance(robot, HasMobileBase):',
      indent + '    robot.mobile_base.full_body_controlled = BASE_MAY_DRIVE_WHILE_REACHING',
    ];
  }
  function generateSelected() { return outputStyle() === 'class' ? generateClass() : generate(); }
  // the demo in the chosen style, or null with the refusal shown, for a combination no demo can be written for
  function generateOrExplain() {
    try { return generateSelected(); } catch (error) { status(error.message, 'err'); toast(error.message, 'err'); return null; }
  }

  async function showCode() {
    try { await synchronizeObjects(); } catch (error) { status(error.message, 'err'); return; }
    const code = generateOrExplain(); if (code === null) return;
    const pre = $('pb-code');
    pre.textContent = code; pre.style.display = 'block'; status('', '');
    // the preview is collapsed to keep the scene big, so reveal it and bring it into view
    if (pre.scrollIntoView) pre.scrollIntoView({ behavior: 'smooth', block: 'center' });
    toast('Generated ' + fileName() + ' — see preview below', 'ok');
  }
  function status(msg, cls) { const el = $('pb-status'); el.textContent = msg; el.className = 'pb-status ' + (cls || ''); }
  // a short-lived floating confirmation near the top, so button actions are noticed even
  // when the status line / code preview are scrolled out of view
  let _toastTimer = null;
  function toast(msg, cls) {
    let t = $('pb-toast');
    if (!t) { t = document.createElement('div'); t.id = 'pb-toast'; document.body.appendChild(t); }
    t.textContent = msg; t.className = 'pb-toast show ' + (cls || '');
    if (_toastTimer) clearTimeout(_toastTimer);
    _toastTimer = setTimeout(function () { t.className = 'pb-toast ' + (cls || ''); }, 3200);
  }
  function fileName() { return (($('pb-name').value || 'my_demo').replace(/[^a-z0-9_\-]/gi, '_')) + '.py'; }

  async function download() {
    try { await synchronizeObjects(); } catch (error) { status(error.message, 'err'); return; }
    const code = generateOrExplain(); if (code === null) return;
    const blob = new Blob([code], { type: 'text/x-python' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = fileName(); a.click();
    URL.revokeObjectURL(a.href); status('downloaded ' + fileName(), 'ok'); toast('Downloaded ' + fileName(), 'ok');
  }
  async function save() {
    try { await synchronizeObjects(); } catch (error) { status(error.message, 'err'); return; }
    const code = generateOrExplain(); if (code === null) return;
    toast('Saving ' + fileName() + '…', '');
    fetch('/api/plan/save', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ name: fileName(), code: code }) })
      .then(function (r) { return r.json(); })
      .then(function (j) {
        if (j.ok) { status('saved → ' + j.path + '  (run: cramera-live ' + j.path + ')', 'ok'); toast('✓ Saved to ' + j.path.replace(/^.*\/coraplex\//, 'coraplex/'), 'ok'); }
        else { status('save failed: ' + (j.error || '?'), 'err'); toast('Save failed: ' + (j.error || '?'), 'err'); }
      })
      .catch(function (e) { status('save failed: ' + e, 'err'); toast('Save failed: ' + e, 'err'); });
  }

  // ---------- live 3D capture ----------
  function bridgeUrl() { return 'http://' + window.location.hostname + ':8765'; }
  //: the viewer page, opened without a scene so that it attaches to the bridge itself
  const VIEWER_PAGE = 'index.html?scene';
  // Show the running demo in the scene frame. The bridge names the scene it bundles
  // the demo into, and the frame opens that scene directly: opened without one, the
  // viewer would first load whatever recording the index lists, a second world the
  // graphics card has to hold, and only then find the bridge and reload. A frame
  // already on the demo's scene is left alone, since reloading it would load the
  // world over again.
  function showRunningDemo(force) {
    const frame = $('pb-3d');
    if (!frame) return Promise.resolve();
    return fetch(bridgeUrl() + '/live_scene').then(function (r) { return r.ok ? r.json() : null; })
      .catch(function () { return null; })
      .then(function (d) {
        const page = d && d.scene ? VIEWER_PAGE + '=' + encodeURIComponent(d.scene) : VIEWER_PAGE;
        if (!force && frame.src.indexOf(page) >= 0) return;
        frame.src = page;
      });
  }
  function quatToYaw(q) { // q = [qx,qy,qz,qw] -> yaw
    return Math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] * q[1] + q[2] * q[2]));
  }
  function liveStatus(msg, cls) { endBusy(); const el = $('pb-live-status'); el.textContent = msg; el.className = 'pb-live-status ' + (cls || ''); }
  // a spinner + live seconds counter for the long wait while a scene comes up, so it is
  // clear the run is alive and how far in it is; endBusy() runs on any final liveStatus()
  let _busyTimer = 0, _busyStart = 0, _busyBase = '', _busyDetail = '';
  function beginBusy(base) {
    setRobotEditingDisabled(true);
    _busyBase = base; _busyDetail = ''; _busyStart = Date.now();
    if (_busyTimer) clearInterval(_busyTimer);
    renderBusy(); _busyTimer = setInterval(renderBusy, 1000);
  }
  function busyDetail(detail) { if (_busyTimer) { _busyDetail = detail || ''; renderBusy(); } }
  function renderBusy() {
    const el = $('pb-live-status'); if (!el) return;
    const s = Math.round((Date.now() - _busyStart) / 1000);
    const esc = function (t) { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; };
    const detail = _busyDetail ? ' — ' + esc(_busyDetail) : '';
    el.className = 'pb-live-status';
    el.innerHTML = '<span class="cr-busy"><span class="cr-spinner"></span>' + esc(_busyBase) + detail + ' · ' + s + 's</span>';
  }
  function endBusy() { if (_busyTimer) { clearInterval(_busyTimer); _busyTimer = 0; } setRobotEditingDisabled(false); }
  /** @param {boolean} disabled Keep the authored robot stable while its scene starts. */
  function setRobotEditingDisabled(disabled) {
    ['pb-robot-instance', 'pb-add-robot', 'pb-remove-robot', 'pb-robot', 'pb-robot-label', 'pb-rx', 'pb-ry', 'pb-ryaw', 'pb-place-robot'].forEach(function (identifier) { $(identifier).disabled = disabled; });
    if (!disabled) $('pb-remove-robot').disabled = builderState.instances.length <= 1;
  }
  // the last meaningful line of the demo's log, tidied, so the wait shows where it is
  function lastLogLine(text) {
    if (!text) return '';
    const lines = text.split('\n').map(function (l) { return l.trim(); }).filter(Boolean);
    if (!lines.length) return '';
    let line = lines[lines.length - 1].replace(/^(INFO|WARNING|DEBUG|ERROR):[^:]*:/, '').trim();
    return line.length > 72 ? line.slice(0, 71) + '…' : line;
  }
  async function synchronizeObjects() {
    if (!liveOn) return;
    const [captured] = await Promise.all([fetchCaptured(), synchronizeRobotPoses()]);
    builderState.capture(objects, captured);
    renderObjects(); renderSteps();
  }
  /** Capture every robot's current base pose before changing the selected plan or scene. */
  async function synchronizeRobotPoses() {
    if (!liveOn || !builderState.instances.length) return;
    const response = await fetch(bridgeUrl() + '/robots', {cache: 'no-store'});
    if (!response.ok) throw new Error('Could not read robot poses from the live scene');
    builderState.captureRobots(await response.json());
    renderRobotInstances();
  }
  function fetchCaptured() {
    return fetch(bridgeUrl() + '/captured_objects').then(function (r) { if (!r.ok) throw new Error('live scene is unavailable'); return r.json(); }).then(function (d) { return (d && d.objects) || {}; });
  }
  // show the running scene's loose objects as this page's objects where each lies right
  // now: a box the setup lists is moved there, one the scene was given on its command
  // line alone is added, so a step may name either and a saved setup starts it there
  async function adoptLiveObjects() {
    let live;
    try { live = await fetchCaptured(); } catch (e) { return; }
    Object.keys(live).forEach(function (key) {
      const pose = live[key];
      if (!pose || pose.length < 7) return;
      const rpy = quatToRpy(pose.slice(3));
      const at = { x: r3(pose[0]), y: r3(pose[1]), z: r3(pose[2]), roll: r3(rpy[0]), pitch: r3(rpy[1]), yaw: r3(rpy[2]) };
      const listed = objects.find(function (o) { return o.mesh === key; });
      if (listed) Object.assign(listed, at); else addObject(key, at);
    });
  }
  function quatToRpy(q) { // q = [qx,qy,qz,qw] -> [roll, pitch, yaw] (ROS convention)
    const x = q[0], y = q[1], z = q[2], w = q[3];
    const roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
    const sp = 2 * (w * y - z * x);
    const pitch = Math.abs(sp) >= 1 ? Math.sign(sp) * Math.PI / 2 : Math.asin(sp);
    const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
    return [roll, pitch, yaw];
  }
  function r3(v) { return Math.round(v * 1000) / 1000; }
  function poseFromCaptured(objs, mesh) {
    const p = objs[mesh]; if (!p || p.length < 7) return null;
    const rpy = quatToRpy(p.slice(3));
    return { x: Math.round(p[0] * 100) / 100, y: Math.round(p[1] * 100) / 100, z: Math.round(p[2] * 100) / 100,
      roll: r3(rpy[0]), pitch: r3(rpy[1]), yaw: r3(rpy[2]) };
  }
  function captureObject(oid) {
    const o = objects.find(function (x) { return x.id === oid; }); if (!o) return;
    fetchCaptured().then(function (objs) {
      const pz = poseFromCaptured(objs, o.mesh);
      if (!pz) { status('no live pose for ' + o.mesh + ' — is the scene running?', 'err'); return; }
      o.x = pz.x; o.y = pz.y; o.z = pz.z; o.roll = pz.roll; o.pitch = pz.pitch; o.yaw = pz.yaw; renderObjects();
      status('captured ' + o.name + ' → (' + pz.x + ', ' + pz.y + ', ' + pz.z + ')', 'ok');
      toast('Captured ' + o.name + '’s start pose', 'ok');
    }).catch(function () { status('capture failed — start the live scene first', 'err'); });
  }
  function captureStepStart(sid) {
    // capture the transported object's live pose as ITS start pose (the "from")
    const s = steps.find(function (x) { return x.id === sid; }); if (!s) return;
    if (!s.params.object) { status('pick an object for this Transport step first', 'err'); return; }
    const o = objects.find(function (x) { return x.mesh === s.params.object; });
    if (!o) { status('object ' + s.params.object + ' is not in the objects list', 'err'); return; }
    fetchCaptured().then(function (objs) {
      const pz = poseFromCaptured(objs, o.mesh);
      if (!pz) { status('no live pose for ' + o.mesh + ' — is the scene running?', 'err'); return; }
      o.x = pz.x; o.y = pz.y; o.z = pz.z; o.roll = pz.roll; o.pitch = pz.pitch; o.yaw = pz.yaw;
      renderObjects();   // left panel
      renderSteps();     // the start (from) fields on the step read from the object
      status('captured start for ' + o.name + ' → (' + pz.x + ', ' + pz.y + ', ' + pz.z + ')', 'ok');
    }).catch(function () { status('capture failed — start the live scene first', 'err'); });
  }
  function captureStepTarget(sid) {
    const s = steps.find(function (x) { return x.id === sid; }); if (!s) return;
    if (!s.params.object) { status('pick an object for this Transport step first', 'err'); return; }
    fetchCaptured().then(function (objs) {
      const pz = poseFromCaptured(objs, s.params.object);
      if (!pz) { status('no live pose for ' + s.params.object, 'err'); return; }
      s.params.x = pz.x; s.params.y = pz.y; s.params.z = pz.z; s.params.yaw = pz.yaw; renderSteps();
      status('captured target for ' + s.params.object + ' → (' + pz.x + ', ' + pz.y + ', ' + pz.z + ')', 'ok');
    }).catch(function () { status('capture failed — start the live scene first', 'err'); });
  }
  function hideScaffoldLog() { const el = $('pb-scaffold-log'); if (el) { el.style.display = 'none'; el.textContent = ''; } }
  async function startLive() {
    const my = ++_runMonitor;
    beginBusy('Preparing scene — capturing robot poses');
    try { await synchronizeRobotPoses(); }
    catch (error) { if (my === _runMonitor) liveStatus(error.message, 'err'); return; }
    if (my !== _runMonitor) return;
    lastSpawnRobot = robotInfo().name;
    const code = generate([]);   // scaffold: world + objects, idle
    beginBusy('Starting scene — parsing meshes'); hideScaffoldLog();
    return launchRun(code, false, my);
  }
  // run the built plan itself (not the idle scaffold) and watch the robot perform it:
  // the full generated demo ends in `plan.perform()`, launched through the same endpoint
  async function runPlan() {
    if (!steps.length) { liveStatus('add plan steps first', 'err'); return; }
    const my = ++_runMonitor;
    if (attachedToRunningDemo) return runInRunningDemo(my);
    // a demo this page did not start holds the bridge's port, so one started here would die
    if (!liveOn && await runningDemoSetup()) {
      liveStatus('a demo is already running — open “the running demo” under setups to run the plan in it, or stop that demo first', 'err');
      return;
    }
    beginBusy('Preparing plan — capturing the live scene');
    try { await synchronizeObjects(); }
    catch (error) { if (my === _runMonitor) liveStatus('Could not read object poses: ' + error.message, 'err'); return; }
    if (my !== _runMonitor) return;
    const code = generateOrExplain();   // full demo (matches the chosen output style), ends by performing the plan
    if (code === null) { endBusy(); return; }
    lastSpawnRobot = robotInfo().name;
    beginBusy('Running plan — parsing meshes'); hideScaffoldLog();
    return launchRun(code, true, my);
  }
  /** @returns {Promise<object|null>} The running demo's setup answer, or null when none runs. */
  function runningDemoSetup() {
    return fetch(bridgeUrl() + '/setup', {cache: 'no-store'})
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (answer) { return answer && answer.ok ? answer : null; })
      .catch(function () { return null; });
  }
  /**
   * Have the active robot of the running demo perform the steps, in the demo's own
   * process, and follow the plan until it ends.
   * @param {number} my Generation invalidated by a stop or newer launch.
   */
  async function runInRunningDemo(my) {
    const instance = builderState.activeRobot();
    if (!instance) { liveStatus('choose the robot to run the plan on', 'err'); return; }
    beginBusy('Starting the plan in the running demo');
    const previous = window.PlanBuilderState.planRoot(await fetchLivePlan());
    if (my !== _runMonitor) return;
    let answer;
    try {
      answer = await fetch(bridgeUrl() + '/plan/run', {method: 'POST', headers: {'content-type': 'application/json'},
        body: JSON.stringify({identifier: instance.id, model: instance.model,
          steps: steps.map(function (step) { return {type: step.type, params: Object.assign({}, step.params)}; })})})
        .then(function (r) { return r.json(); });
    } catch (error) { if (my === _runMonitor) liveStatus('the running demo cannot be reached: ' + error.message, 'err'); return; }
    if (my !== _runMonitor) return;
    if (!answer.ok) { liveStatus('not run: ' + (answer.error || '?'), 'err'); return; }
    liveStatus(window.PlanBuilderState.RUN_PROGRESS.RUNNING.message, 'ok');
    (function tick() {
      if (my !== _runMonitor) return;
      fetchLivePlan().then(function (snapshot) {
        if (my !== _runMonitor) return;
        const result = window.PlanBuilderState.planResult(snapshot, previous && previous.id);
        if (result) {
          liveStatus(result.message.replace('open the run log', 'see the scene\'s terminal'), result.style);
          return;
        }
        setTimeout(tick, 1000);
      });
    })();
  }
  /**
   * Replace the scene while retaining the previous plan's identity.
   * @param {string} code Generated demo source.
   * @param {boolean} isPlan Whether to display an authored plan's result.
   * @param {number} my Generation invalidated by a stop or newer launch.
   */
  async function launchRun(code, isPlan, my) {
    const authoredRobotPoses = builderState.snapshotRobotPoseEdits();
    const previous = window.PlanBuilderState.planRoot(await fetchLivePlan());
    if (my !== _runMonitor) return;
    liveOn = false;
    return fetch('/api/plan/scaffold', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ code: code }) })
      .then(function (r) { return r.json(); })
      .then(function (j) {
        if (my !== _runMonitor) return;
        if (!j.ok) { liveStatus('failed: ' + (j.error || '?'), 'err'); return; }
        builderState.acknowledgeRobotPoseEdits(authoredRobotPoses);
        pollLive(0, isPlan ? window.PlanBuilderState.RUN_PROGRESS.RUNNING.message : null, my);
        monitorRun(my, previous && previous.id, isPlan);
      })
      .catch(function (e) { if (my === _runMonitor) liveStatus('failed: ' + e, 'err'); });
  }
  // ---- run log: surface the demo subprocess's stdout/stderr (tracebacks) ----
  function fetchScaffoldLog() {
    return fetch('/api/plan/scaffold/log').then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; });
  }
  /** @returns {Promise<object|null>} Current authoritative tree, if the bridge is available. */
  function fetchLivePlan() {
    return fetch(bridgeUrl() + '/plan', {cache: 'no-store'}).then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; });
  }
  function showScaffoldLog(text) {
    const el = $('pb-scaffold-log'); if (!el) return;
    el.textContent = (text && text.trim()) ? text : '(no output yet)';
    el.style.display = 'block';
    el.scrollTop = el.scrollHeight;
    if (el.scrollIntoView) el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
  let _runMonitor = 0;
  /**
   * Follow plan completion independently of the runner's serving lifetime.
   * @param {number} my Current launch generation.
   * @param {string|null} previousRoot Root replaced by this launch.
   * @param {boolean} isPlan Whether this launch performs the authored plan.
   */
  function monitorRun(my, previousRoot, isPlan) {
    let showingProgress = false;
    (function tick() {
      if (my !== _runMonitor) return;                      // superseded by a newer run/stop
      fetchScaffoldLog().then(async function (d) {
        if (my !== _runMonitor || !d) { if (my === _runMonitor) setTimeout(tick, 2500); return; }
        if (d.returncode !== null && d.returncode !== 0) {   // the demo crashed
          liveOn = false;
          stopRunMonitor();
          reportSceneFailure(d.log);
          return;                                            // stop monitoring
        }
        if (isPlan && liveOn) {
          const snapshot = await fetchLivePlan();
          if (my !== _runMonitor) return;
          const result = window.PlanBuilderState.planResult(snapshot, previousRoot);
          if (result) { liveStatus(result.message, result.style); return; }
          const root = window.PlanBuilderState.planRoot(snapshot);
          if (root && root.id !== previousRoot && root.status === 'RUNNING') {
            const progress = window.PlanBuilderState.planProgress(snapshot, previousRoot);
            if (progress || showingProgress) {
              const presentation = progress || window.PlanBuilderState.RUN_PROGRESS.RUNNING;
              liveStatus(presentation.message, presentation.style);
            }
            showingProgress = !!progress;
          }
        }
        setTimeout(tick, 2500);
      });
    })();
  }
  function stopRunMonitor() { _runMonitor++; }
  function pollLive(n, okMsg, my) {
    if (my !== _runMonitor) return;
    fetch(bridgeUrl() + '/captured_objects').then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (my !== _runMonitor) return;
        if (d) { liveOn = true; builderState.clearFailure(lastSpawnRobot); showModelStatus(); liveStatus(okMsg || '● live — drag objects into place, then Run plan', 'ok'); showRunningDemo(false); fetchSurfaces(); return; }
        // bridge not up yet — but if the demo process already died, show why now
        fetchScaffoldLog().then(function (lg) {
          if (my !== _runMonitor) return;
          if (lg && lg.returncode !== null && lg.returncode !== 0) {
            stopRunMonitor();
            reportSceneFailure(lg.log); return;
          }
          if (n < 40) { busyDetail(lastLogLine(lg && lg.log)); setTimeout(function () { pollLive(n + 1, okMsg, my); }, 3000); }
          else { liveStatus('scene did not come up — see the run log below', 'err'); if (lg) showScaffoldLog(lg.log); }
        });
      })
      .catch(function () { if (my !== _runMonitor) return; if (n < 40) setTimeout(function () { pollLive(n + 1, okMsg, my); }, 3000); else { liveStatus('scene did not come up', 'err'); fetchScaffoldLog().then(function (lg) { if (my === _runMonitor && lg) showScaffoldLog(lg.log); }); } });
  }
  // enumerate placement surfaces from the live world (for the "on a surface" target mode)
  function fetchSurfaces() {
    fetch(bridgeUrl() + '/surfaces').then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        const next = (d && d.surfaces) || [];
        const changed = JSON.stringify(next) !== JSON.stringify(liveSurfaces);
        liveSurfaces = next;
        if (changed && steps.some(placesAtASemanticTarget)) renderSteps();
      }).catch(function () {});
  }
  // reload ONLY the embedded 3D view (it sometimes loads partially) without touching the
  // plan/objects/constraints on this page. A running demo's scene is opened again
  // directly; without one the address is varied so a stuck load is force-refreshed.
  function reloadScene() {
    const f = $('pb-3d'); if (!f) return;
    liveStatus('reloading 3D view…', '');
    if (liveOn || attachedToRunningDemo) { showRunningDemo(true); return; }
    f.src = VIEWER_PAGE + '&r=' + Date.now();
  }
  // draggable dividers between the three columns (palette | plan | scene); widths persist
  function wireColumnResizers() {
    const main = document.querySelector('.pb-main'); if (!main) return;
    const c1 = parseInt(localStorage.getItem('cramera.pb.c1') || '', 10);
    const c3 = parseInt(localStorage.getItem('cramera.pb.c3') || '', 10);
    if (c1 >= 160 && c1 <= 520) main.style.setProperty('--pb-c1', c1 + 'px');
    if (c3 >= 300 && c3 <= 1000) main.style.setProperty('--pb-c3', c3 + 'px');
    main.querySelectorAll('.pb-divider').forEach(function (div) {
      div.addEventListener('mousedown', function (e) {
        e.preventDefault();
        const which = div.dataset.div;
        div.classList.add('dragging');
        document.body.style.userSelect = 'none'; document.body.style.cursor = 'col-resize';
        function move(ev) {
          const r = main.getBoundingClientRect();
          if (which === '1') main.style.setProperty('--pb-c1', Math.max(160, Math.min(520, ev.clientX - r.left)) + 'px');
          else main.style.setProperty('--pb-c3', Math.max(300, Math.min(1000, r.right - ev.clientX)) + 'px');
        }
        function up() {
          document.removeEventListener('mousemove', move); document.removeEventListener('mouseup', up);
          div.classList.remove('dragging');
          document.body.style.userSelect = ''; document.body.style.cursor = '';
          const key = which === '1' ? 'cramera.pb.c1' : 'cramera.pb.c3';
          const v = parseInt(main.style.getPropertyValue(which === '1' ? '--pb-c1' : '--pb-c3'), 10);
          if (v) localStorage.setItem(key, String(v));
        }
        document.addEventListener('mousemove', move); document.addEventListener('mouseup', up);
      });
    });
  }
  function stopLive() {
    liveOn = false; liveSurfaces = []; stopRunMonitor();
    const my = _runMonitor;
    fetch('/api/plan/scaffold/stop', { method: 'POST' }).then(function () { if (my !== _runMonitor) return; liveStatus('stopped', ''); const f=$('pb-3d'); if (f) f.src='about:blank'; }).catch(function () {});
  }

  /** Connect authoring controls to the selected robot's independent state. */
  function wireRobotControls() {
    for (const [identifier, coordinate] of [['pb-rx', 'x'], ['pb-ry', 'y'], ['pb-ryaw', 'yaw']]) {
      $(identifier).addEventListener('input', function () {
        const value = Number(this.value) * (coordinate === 'yaw' ? Math.PI / 180 : 1);
        if (!Number.isFinite(value) || !builderState.activeRobot()) return;
        builderState.updateRobot(builderState.activeIdentifier, {[coordinate]: value});
        reshowIfGenerated();
      });
    }
    // a finished edit stands the robot there in a running scene straight away
    for (const identifier of ['pb-rx', 'pb-ry', 'pb-ryaw']) {
      $(identifier).addEventListener('change', placeActiveRobotLive);
    }
    $('pb-rtopic').addEventListener('change', function () {
      builderState.updateRobot(builderState.activeIdentifier, {jointStateTopic: this.value});
    });
    $('pb-rlocalize').addEventListener('change', function () {
      builderState.updateRobot(builderState.activeIdentifier, {localizationTopic: this.value});
    });
    $('pb-rrepeat').addEventListener('change', function () {
      builderState.updateRobot(builderState.activeIdentifier, {repeatsPlan: this.checked});
    });
    $('pb-robot-instance').addEventListener('change', function () { selectRobotInstance(this.value); });
    $('pb-add-robot').addEventListener('click', addRobotInstance);
    $('pb-remove-robot').addEventListener('click', removeRobotInstance);
    $('pb-robot-label').addEventListener('change', function () {
      builderState.updateRobot(builderState.activeIdentifier, {label: this.value});
      renderRobotInstances(); reshowIfGenerated();
    });
  }

  // ---------- demo setups ----------
  // a setup is the environment, where each robot stands, which robots follow a real
  // robot and what the others do; the running demo offers its own through the bridge
  const RUNNING_DEMO_SETUP = '__running__';
  let attachedToRunningDemo = false;   // the 3D view shows a demo this page did not start
  function makeStep(type, params) { return { id: 's' + (stepSeq++), type: type, params: params }; }
  function ensureEnvironmentOption(path) {
    const select = $('pb-env');
    if (!Array.from(select.options).some(function (option) { return option.value === path; })) {
      const option = document.createElement('option');
      option.value = path; option.textContent = path.split('/').filter(Boolean).slice(-2).join('/');
      select.appendChild(option);
    }
    select.value = path;
  }
  // open the setup the address asks for (?open=running for the running demo's, or a
  // saved setup's name), once the catalog is loaded; a demo that is still coming up is
  // waited for, since a browser started with it opens before its scene is built
  const OPEN_POLL_SECONDS = 3;
  const OPEN_WAIT_MINUTES = 15;
  async function openFromAddress() {
    const wanted = new URLSearchParams(window.location.search).get('open');
    if (!wanted) return;
    const chosen = wanted === 'running' ? RUNNING_DEMO_SETUP : wanted;
    const deadline = Date.now() + OPEN_WAIT_MINUTES * 60 * 1000;
    while (Date.now() < deadline) {
      await refreshSetupList();
      const select = $('pb-setup-open');
      if (Array.from(select.options).some(function (option) { return option.value === chosen; })) {
        select.value = chosen;
        await openSetup();
        return;
      }
      if (chosen !== RUNNING_DEMO_SETUP) { status('no setup named ' + wanted, 'err'); return; }
      status('waiting for the running demo to come up ...', 'ok');
      await new Promise(function (resolve) { setTimeout(resolve, OPEN_POLL_SECONDS * 1000); });
    }
    status('the running demo did not come up', 'err');
  }
  async function refreshSetupList() {
    const select = $('pb-setup-open');
    const choices = [['', '— choose a setup —']];
    try {
      const running = await fetch(bridgeUrl() + '/setup', {cache: 'no-store'}).then(function (r) { return r.ok ? r.json() : null; });
      if (running && running.ok) choices.push([RUNNING_DEMO_SETUP, 'the running demo']);
    } catch (e) { /* no demo running */ }
    try {
      const saved = await fetch('/api/setup/list', {cache: 'no-store'}).then(function (r) { return r.json(); });
      (saved.names || []).forEach(function (name) { choices.push([name, name]); });
    } catch (e) { /* the server offers none */ }
    select.innerHTML = choices.map(function (c) { return '<option value="' + c[0] + '">' + c[1] + '</option>'; }).join('');
  }
  async function openSetup() {
    const chosen = $('pb-setup-open').value;
    if (!chosen) { status('choose a setup to open', 'err'); return; }
    try {
      const url = chosen === RUNNING_DEMO_SETUP ? bridgeUrl() + '/setup' : '/api/setup/open?name=' + encodeURIComponent(chosen);
      const answer = await fetch(url, {cache: 'no-store'}).then(function (r) { return r.json(); });
      if (!answer.ok) throw new Error(answer.error || 'the setup could not be opened');
      // a step read back gets the builder's defaults for whatever its form left out
      const active = window.DemoSetupForm.applyTo(builderState, answer.setup, function (type, params) {
        return makeStep(type, Object.assign({}, (BLOCKS[type] || {}).params || {}, params));
      });
      if (answer.setup.environment) ensureEnvironmentOption(answer.setup.environment.path);
      steps = active.steps;
      attachedToRunningDemo = chosen === RUNNING_DEMO_SETUP;
      // the setup's boxes, which Pick, Place and Transport steps may name; in a running
      // demo each is shown where it lies right now rather than where it started
      objects = [];
      (builderState.objects || []).forEach(function (box) {
        addObject(box.name, {x: box.x, y: box.y, z: box.z, yaw: box.yaw, size: box.size});
      });
      if (attachedToRunningDemo) await adoptLiveObjects();
      renderObjects();
      if (attachedToRunningDemo) showRunningDemo(true);
      renderRobotInstances(); renderBlocks(); renderSteps(); showModelStatus(); reshowIfGenerated();
      const name = attachedToRunningDemo ? 'the running demo' : chosen;
      status('opened ' + name + (attachedToRunningDemo ? ' — moving a robot moves it in the scene' : ''), 'ok');
      toast('Opened ' + name, 'ok');
    } catch (error) { status('open failed: ' + error.message, 'err'); toast('Open failed: ' + error.message, 'err'); }
  }
  async function saveSetup() {
    const name = ($('pb-setup-name').value || '').trim();
    if (!name) { status('name the setup first', 'err'); return; }
    // the environment as offered, so a map is refused by the server by its name
    const payload = window.DemoSetupForm.toPayload(builderState, steps, selectedEnvironment(), objects);
    try {
      const answer = await fetch('/api/setup/save', {method: 'POST', headers: {'content-type': 'application/json'},
        body: JSON.stringify({name: name, setup: payload})}).then(function (r) { return r.json(); });
      if (!answer.ok) throw new Error(answer.error || 'the setup could not be saved');
      status('saved setup → ' + answer.path + '  (run it with: --setup ' + answer.path + ')', 'ok');
      toast('✓ Saved setup ' + name, 'ok');
      refreshSetupList();
    } catch (error) { status('save failed: ' + error.message, 'err'); toast('Save failed: ' + error.message, 'err'); }
  }
  // stand the active robot where its fields say, in the scene the 3D view shows
  async function placeActiveRobotLive() {
    const instance = builderState.activeRobot();
    if (!instance || !(liveOn || attachedToRunningDemo)) return;
    try {
      const answer = await fetch(bridgeUrl() + '/robot/place', {method: 'POST', headers: {'content-type': 'application/json'},
        body: JSON.stringify({identifier: instance.id, model: instance.model, x: instance.x, y: instance.y, yaw: instance.yaw})})
        .then(function (r) { return r.json(); });
      if (!answer.ok) throw new Error(answer.error || 'the robot could not be moved');
      builderState.acknowledgeRobotPoseEdits(builderState.snapshotRobotPoseEdits());
      status('moved ' + instance.label + ' in the scene', 'ok');
    } catch (error) { status('move failed: ' + error.message, 'err'); }
  }

  // ---------- boot ----------
  loadCatalog();
  renderConstraints();
  $('pb-add-obj').addEventListener('click', function () {
    const o = addObject($('pb-mesh').value);
    highlightObjectsInScene();
    // objects are spawned when the scaffold is built, so one added mid-session needs a
    // (re)start to appear; it will show at its staging spot beside the robot, in the open.
    if (liveOn) status('added ' + o.name + ' — click “Start live scene” to (re)spawn it in the 3D view (staged beside the robot)', 'ok');
  });
  $('pb-con-add').addEventListener('click', function () { const inp = $('pb-con-in'); addConstraintText(inp.value); inp.value = ''; inp.focus(); });
  $('pb-con-in').addEventListener('keydown', function (e) { if (e.key === 'Enter') { e.preventDefault(); addConstraintText(this.value); this.value = ''; } });
  (function () {
    const box = $('pb-con-info-box'); if (box) box.innerHTML = conInfoHtml();
    const btn = $('pb-con-info'); if (btn && box) btn.addEventListener('click', function () { box.classList.toggle('open'); });
  })();
  $('pb-env').addEventListener('change', renderScene);
  $('pb-run').addEventListener('click', runPlan);
  $('pb-live-start').addEventListener('click', startLive);
  $('pb-live-stop').addEventListener('click', stopLive);
  // a floating dropdown: btn toggles menu; clicking a button/link inside closes it (except
  // `keepOpenId`), as does clicking outside. Selects/inputs inside never close it.
  function wireDropdown(btnId, menuId, keepOpenId) {
    const btn = $(btnId), menu = $(menuId); if (!btn || !menu) return function () {};
    function close() { menu.hidden = true; }
    btn.addEventListener('click', function (e) { e.stopPropagation(); menu.hidden = !menu.hidden; });
    menu.addEventListener('click', function (e) { if (e.target.closest('button, a') && e.target.id !== keepOpenId) close(); });
    document.addEventListener('click', function (e) { if (!menu.hidden && !menu.contains(e.target) && e.target !== btn) close(); });
    return close;
  }
  const closeSceneMenu = wireDropdown('pb-menu-btn', 'pb-menu', 'pb-show-code');
  wireDropdown('pb-setup-btn', 'pb-setup-menu');
  (function () { const sc = $('pb-show-code'); if (sc) sc.addEventListener('click', function () { showCode(); closeSceneMenu(); }); })();
  wireColumnResizers();
  $('pb-reset-all').addEventListener('click', resetAllObjects);
  $('pb-drop').addEventListener('click', dropObjects);
  $('pb-reload-3d').addEventListener('click', reloadScene);
  $('pb-place-robot').addEventListener('click', toggleFloorPlacement);
  $('pb-log').addEventListener('click', function () {
    fetchScaffoldLog().then(function (d) {
      if (!d) { toast('no run log yet — start a scene or run the plan first', 'err'); return; }
      showScaffoldLog(d.log);
      const st = d.returncode === null ? 'running' : ('exited ' + d.returncode);
      toast('Run log (' + st + ')', d.returncode ? 'err' : 'ok');
    });
  });
  // whenever the embedded scene (re)loads, (re)send the objects to flag with arrows
  $('pb-3d').addEventListener('load', function () { setTimeout(function () { highlightObjectsInScene(); sendNavigateTargets(); }, 400); });
  wireRobotControls();
  addObject('milk.stl');   // staged above the robot (never inside furniture); drop/drag to place
  addObject('bowl.stl');
  renderSteps();
  // a friendly starter plan
  $('pb-generate').addEventListener('click', showCode);
  function reshowIfGenerated() {
    const pre = $('pb-code'); if (pre && pre.textContent && pre.textContent.indexOf('Click') !== 0) showCode();
  }
  $('pb-collisions').innerHTML = window.ExecutionEnvironments.all().map(function (e) {
    return '<option value="' + e.name + '">' + e.label + '</option>';
  }).join('');
  $('pb-collisions').addEventListener('change', reshowIfGenerated);
  $('pb-execution').innerHTML = window.ExecutionTypes.all().map(function (t) {
    return '<option value="' + t.name + '">' + t.label + '</option>';
  }).join('');
  $('pb-execution').addEventListener('change', reshowIfGenerated);
  $('pb-base').innerHTML = window.BaseControl.all().map(function (c) {
    return '<option value="' + c.name + '">' + c.label + '</option>';
  }).join('');
  $('pb-base').addEventListener('change', reshowIfGenerated);
  $('pb-style').addEventListener('change', reshowIfGenerated);
  $('pb-robot').addEventListener('change', selectRobot);
  $('pb-env').addEventListener('change', reshowIfGenerated);
  $('pb-download').addEventListener('click', download);
  $('pb-save').addEventListener('click', save);
  $('pb-setup-open-btn').addEventListener('click', openSetup);
  $('pb-setup-save').addEventListener('click', saveSetup);
  $('pb-setup-btn').addEventListener('click', refreshSetupList);
  refreshSetupList();
  window.addEventListener('resize', renderScene);
})();
