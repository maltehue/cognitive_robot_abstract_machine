/* ============================================================================
 * teleop.js — the Teleoperation page: start an idle live scene, then drag a pad
 * to servo the running robot's arm through the live bridge's POST /teleop.
 * A page script (owns the whole document), like plan_builder.js.
 * ==========================================================================*/
(function () {
  'use strict';

  const $ = function (id) { return document.getElementById(id); };
  const bridgeUrl = function () { return 'http://' + window.location.hostname + ':8765'; };

  let liveOn = false;
  let dragging = false;
  let cameraOn = false;
  let handPresent = false;
  const target = { x: 0, y: 0, z: 0, gripper: null };   // normalised [-1,1]; gripper 0 shut..1 open

  // camera → arm mapping (all tunable). Depth comes from apparent hand size (bigger = nearer),
  // the gripper from the pinch (fingers together = shut). Ranges are in MediaPipe's normalised
  // image units, so they hold across camera distances reasonably well.
  const SPAN_NEAR = 0.34;   // hand this big (close to camera) → arm reaches fully forward (x=+1)
  const SPAN_FAR = 0.12;    // hand this small (far) → arm pulled back (x=-1)
  const PINCH_SHUT = 0.18;  // pinch at/below this → gripper fully closed
  const PINCH_OPEN = 0.70;  // pinch at/above this → gripper fully open
  function remap(v, lo, hi) { return Math.max(-1, Math.min(1, ((v - lo) / (hi - lo)) * 2 - 1)); }
  function unit(v, lo, hi) { return Math.max(0, Math.min(1, (v - lo) / (hi - lo))); }

  // ---------- the idle scene the teleop drives (world + PR2, parked, then held) ----------
  function scaffoldCode(env) {
    return [
      '"""Idle scene for the Teleoperation page (cramera)."""',
      'import os',
      'from coraplex.datastructures.dataclasses import Context',
      'from coraplex.datastructures.enums import Arms, VisualizationBackend',
      'from coraplex.execution_environment import simulated_robot',
      'from coraplex.plans.factories import sequential',
      'from coraplex.visualization import WorldVisualization',
      'from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction',
      'from semantic_digital_twin.adapters.urdf import URDFParser',
      'from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner',
      'from semantic_digital_twin.robots.pr2 import PR2',
      'from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix',
      '',
      '_WORLDS = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "worlds")',
      '',
      '',
      'def build_world(env_file, robot_xy):',
      '    robot_world = URDFParser.from_file(PR2.get_ros_file_path()).parse()',
      '    world = URDFParser.from_file(os.path.join(_WORLDS, env_file)).parse()',
      '    with world.modify_world():',
      '        robot_root = robot_world.get_body_by_name(PR2._get_root_body_name())',
      '        drive = PR2.get_drive_connection_type().create_with_dofs(',
      '            parent=world.root, child=robot_root, world=world)',
      '        world.merge_world(robot_world, drive)',
      '        drive.origin = HomogeneousTransformationMatrix.from_xyz_rpy(robot_xy[0], robot_xy[1], 0)',
      '    standing = max(0.0, -world.height_of_lowest_collision_point_of_branch(robot_root))',
      '    with world.modify_world():',
      '        drive.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(',
      '            z=standing, reference_frame=world.root)',
      '    return world',
      '',
      '',
      'world = build_world("' + env + '", (0.0, 0.0))',
      'visualization = WorldVisualization.from_environment(',
      '    world, default_backend=VisualizationBackend.CRAMERA).start()',
      'pr2 = PR2.from_world(world)',
      'context = Context(world=world, robot=pr2, _debug=False, ros_node=visualization.ros_node)',
      'with world.modify_world():',
      '    WorldReasoner(world).reason()',
      'context.evaluate_conditions = False',
      'plan = sequential([ParkArmsAction(Arms.BOTH)], context=context).plan',
      'visualization.attach_plan(plan)',
      'with simulated_robot:',
      '    plan.perform()',
      ''
    ].join('\n');
  }

  function status(msg, kind) {
    endBusy();
    const el = $('te-status'); el.textContent = msg;
    el.className = 'tele-status' + (kind ? ' ' + kind : '');
  }

  // a spinner + live seconds counter (plus the demo's latest log line) for the long wait
  // while the scene comes up, so it is clear the run is alive and where it is
  let _busyTimer = 0, _busyStart = 0, _busyBase = '', _busyDetail = '';
  function beginBusy(base) {
    _busyBase = base; _busyDetail = ''; _busyStart = Date.now();
    if (_busyTimer) clearInterval(_busyTimer);
    renderBusy(); _busyTimer = setInterval(renderBusy, 1000);
  }
  function busyDetail(detail) { if (_busyTimer) { _busyDetail = detail || ''; renderBusy(); } }
  function renderBusy() {
    const el = $('te-status'); if (!el) return;
    const s = Math.round((Date.now() - _busyStart) / 1000);
    const esc = function (t) { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; };
    const detail = _busyDetail ? ' — ' + esc(_busyDetail) : '';
    el.className = 'tele-status';
    el.innerHTML = '<span class="cr-busy"><span class="cr-spinner"></span>' + esc(_busyBase) + detail + ' · ' + s + 's</span>';
  }
  function endBusy() { if (_busyTimer) { clearInterval(_busyTimer); _busyTimer = 0; } }
  function lastLogLine(text) {
    if (!text) return '';
    const lines = text.split('\n').map(function (l) { return l.trim(); }).filter(Boolean);
    if (!lines.length) return '';
    const line = lines[lines.length - 1].replace(/^(INFO|WARNING|DEBUG|ERROR):[^:]*:/, '').trim();
    return line.length > 72 ? line.slice(0, 71) + '…' : line;
  }
  function fetchLog() { return fetch('/api/plan/scaffold/log').then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; }); }

  // ---------- scene lifecycle ----------
  function startScene() {
    beginBusy('Starting scene — parsing meshes');
    fetch('/api/plan/scaffold', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ code: scaffoldCode($('te-env').value) })
    }).then(function (r) { return r.json(); })
      .then(function (j) { if (!j.ok) { status('failed: ' + (j.error || '?'), 'err'); return; } pollScene(0); })
      .catch(function (e) { status('failed: ' + e, 'err'); });
  }

  function pollScene(n) {
    fetch(bridgeUrl() + '/state').then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (d && d.frames) {
          liveOn = true;
          status('● live — drag the pad to move the arm', 'ok');
          const f = $('te-3d'); if (f && f.src.indexOf('index.html') < 0) f.src = 'index.html?scene';
        } else if (n < 40) {
          fetchLog().then(function (lg) {
            if (lg && lg.returncode !== null && lg.returncode !== 0) { status('demo failed to start (exit ' + lg.returncode + ') — check the terminal', 'err'); return; }
            busyDetail(lastLogLine(lg && lg.log));
            setTimeout(function () { pollScene(n + 1); }, 3000);
          });
        } else status('scene did not come up — check the terminal', 'err');
      })
      .catch(function () { if (n < 40) setTimeout(function () { pollScene(n + 1); }, 3000); else status('scene did not come up', 'err'); });
  }

  // ---------- camera ----------
  function camStatus(msg) { $('te-cam-status').textContent = msg; }

  function toggleCamera() {
    if (cameraOn) {
      if (window.teleopHand) window.teleopHand.stop();
      cameraOn = false; handPresent = false; target.gripper = null;
      $('te-cam').textContent = '▶ Start camera'; camStatus('off');
      return;
    }
    if (!window.teleopHand) { camStatus('hand tracker still loading — try again'); return; }
    $('te-cam-status').innerHTML = '<span class="cr-busy"><span class="cr-spinner"></span>starting camera…</span>';
    window.teleopHand.start($('te-video'), $('te-cam-canvas'))
      .then(function () { cameraOn = true; $('te-cam').textContent = '■ Stop camera'; camStatus('● tracking — move your hand'); })
      .catch(function (e) { camStatus('camera failed: ' + (e && e.message ? e.message : e)); });
  }

  function stopScene() {
    liveOn = false;
    if (cameraOn) toggleCamera();
    fetch(bridgeUrl() + '/teleop/stop', { method: 'POST' }).catch(function () {});
    fetch('/api/plan/scaffold/stop', { method: 'POST' })
      .then(function () { status('stopped'); const f = $('te-3d'); if (f) f.src = 'about:blank'; })
      .catch(function () {});
  }

  // ---------- streaming ----------
  function sendTarget() {
    if (!liveOn) return;
    fetch(bridgeUrl() + '/teleop', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ arm: $('te-arm').value, position: [target.x, target.y, target.z], gripper: target.gripper })
    }).then(function (r) { return r.json(); })
      .then(function (j) { if (j && !j.ok) status('teleop: ' + (j.error || '?'), 'err'); })
      .catch(function () {});
  }

  // stream at ~30 Hz: from the camera while it runs, else from the pad (continuously,
  // or only while dragging when the box is checked)
  setInterval(function () {
    if (!liveOn) return;
    if (cameraOn) { if (handPresent) sendTarget(); return; }
    if ($('te-stream').checked && !dragging) return;
    sendTarget();
  }, 33);

  // called by teleop_hand.mjs for every camera frame; the wrist's image position drives
  // the arm sideways (x) and up/down (y), depth held at a fixed forward reach
  window.teleopOnHand = function (hand) {
    if (!cameraOn) return;
    if (!hand.present) { handPresent = false; return; }
    // front camera: move your hand left and the arm goes left (no mirror flip on control)
    target.y = Math.max(-1, Math.min(1, (hand.x - 0.5) * 2));
    target.z = Math.max(-1, Math.min(1, (0.5 - hand.y) * 2));
    // depth from hand size, gripper from pinch
    target.x = (typeof hand.span === 'number') ? remap(hand.span, SPAN_FAR, SPAN_NEAR) : 0;
    target.gripper = (typeof hand.pinch === 'number') ? unit(hand.pinch, PINCH_SHUT, PINCH_OPEN) : null;
    handPresent = true;
    updateReadout();
  };

  // ---------- the pad ----------
  function updateReadout() {
    $('te-read').textContent =
      'x ' + target.x.toFixed(2) + '  y ' + target.y.toFixed(2) + '  z ' + target.z.toFixed(2) +
      (target.gripper === null ? '' : '  grip ' + target.gripper.toFixed(2));
  }

  function padTo(ev) {
    const pad = $('te-pad'); const rect = pad.getBoundingClientRect();
    const px = (ev.clientX - rect.left) / rect.width;    // 0..1 left→right
    const py = (ev.clientY - rect.top) / rect.height;    // 0..1 top→bottom
    // left/right of the pad → y (+y left); up/down → x (+x forward, i.e. top)
    target.y = Math.max(-1, Math.min(1, 1 - 2 * px));
    target.x = Math.max(-1, Math.min(1, 1 - 2 * py));
    const dot = $('te-dot');
    dot.style.left = (Math.max(0, Math.min(1, px)) * 100) + '%';
    dot.style.top = (Math.max(0, Math.min(1, py)) * 100) + '%';
    updateReadout();
  }

  function boot() {
    const pad = $('te-pad');
    pad.addEventListener('pointerdown', function (e) { dragging = true; pad.setPointerCapture(e.pointerId); padTo(e); sendTarget(); });
    pad.addEventListener('pointermove', function (e) { if (dragging) padTo(e); });
    pad.addEventListener('pointerup', function (e) { dragging = false; try { pad.releasePointerCapture(e.pointerId); } catch (_) {} });
    pad.addEventListener('pointercancel', function () { dragging = false; });
    $('te-z').addEventListener('input', function () { target.z = parseFloat(this.value) || 0; updateReadout(); if (!$('te-stream').checked) sendTarget(); });
    $('te-start').addEventListener('click', startScene);
    $('te-stop').addEventListener('click', stopScene);
    $('te-cam').addEventListener('click', toggleCamera);
    updateReadout();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
