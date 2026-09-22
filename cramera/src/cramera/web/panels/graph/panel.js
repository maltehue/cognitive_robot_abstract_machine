/* ============================================================================
 * panels/graph/panel.js — the graph view with five tabs.
 *
 *   Knowledge   the entity graph + CRAM architecture (double-click drills in)
 *   Plan        the executed plan tree, node border = execution status
 *   Statechart  the giskardpy motion statechart of the running motion group
 *   Kinematics  the robot's URDF tree (links as nodes, joints as edges)
 *   Transforms  the executing world's connection graph, node border = freshness
 *
 * The tabs in LIVE_ENDPOINT take their node status from the cramera-live bridge
 * while it is attached: structure changes rebuild the graph, pure status
 * changes only re-colour the rings (no layout jumps). Without a bridge, the
 * Statechart tab follows the playhead through the statecharts the played
 * recording captured.
 *
 * Bus events:
 *   emits    entity:select {id, detail, relations}   node clicked
 *   listens  entity:highlight {ids, focus?}          spotlight matching nodes
 *   listens  scene:step {step}                       highlight the running episode
 *   listens  scene:frame {index}                    follow the replay's statecharts
 *   listens  live:changed {on, url}                  start/stop the status poll
 *
 * Rendering is delegated to graph.js (window.Graph, vis-network wrapper).
 * ==========================================================================*/
Panels.define('graph', function (root, bus) {
  root.innerHTML =
    '<div class="graph-wrap">' +
    '  <div class="graph-tabs" id="graph-tabs">' +
    '    <button data-view="knowledge" class="active" title="the entity graph (EQL / knowledge base)">Knowledge</button>' +
    '    <button data-view="plan" title="the plan tree, with the execution status of every node">Plan</button>' +
    '    <button data-view="chart" title="the giskardpy motion statechart of the running motion group">Statechart</button>' +
    '    <button data-view="kinematics" title="the robot\'s kinematic structure — URDF links &amp; joints">Kinematics</button>' +
    '    <button data-view="transforms" title="the world\'s connection graph — which frame hangs from which, and how recently it moved">Transforms</button>' +
    '    <span class="gt-live" id="gt-live" title="node status is streaming from the running demo">◉ live status</span>' +
    '  </div>' +
    '  <div class="graph-canvas"></div>' +
    '  <div class="graph-steps" id="graph-steps" style="display:none"></div>' +
    '  <div class="graph-zoom">' +
    '    <button id="graph-zoom-in" title="Zoom in — or pinch on a touchpad">+</button>' +
    '    <button id="graph-zoom-out" title="Zoom out — or pinch on a touchpad">−</button>' +
    '    <button id="graph-zoom-fit" title="Fit the whole graph">⤡</button>' +
    '  </div>' +
    '  <div id="graph-empty" class="graph-empty" style="display:none"></div>' +
    '  <div id="graph-nav" class="graph-nav" style="display:none">' +
    '    <button id="gnav-home" title="back to the overview">⌂</button>' +
    '    <button id="gnav-up" title="one level up">↑ back</button>' +
    '    <span id="gnav-path"></span>' +
    '  </div>' +
    '  <div class="legend" id="legend"></div>' +
    '</div>';

  const emptyEl = root.querySelector('#graph-empty');
  const navEl = root.querySelector('#graph-nav');
  const navUp = root.querySelector('#gnav-up');
  const navHome = root.querySelector('#gnav-home');
  const navPath = root.querySelector('#gnav-path');
  const tabsEl = root.querySelector('#graph-tabs');
  const liveBadge = root.querySelector('#gt-live');
  Graph.attach(root.querySelector('.graph-canvas'), root.querySelector('#legend'));
  const canvasEl = root.querySelector('.graph-canvas');

  // vis-network draws onto a canvas of the size it had when it was built, so a canvas
  // that has changed size since leaves the graph in the old one. More than the window
  // changes it: hiding a panel re-columns the layout, dragging a divider re-shares it,
  // and the Plan tab's step list covers it entirely — so what is watched is the canvas
  // itself, whoever resized it. Re-fitted once the change settles rather than per
  // event, of which a drag fires many.
  const REFIT_DELAY_MILLISECONDS = 120;
  let refit = null;
  function refitWhenSettled() {
    if (refit) window.clearTimeout(refit);
    refit = window.setTimeout(function () { refit = null; Graph.resize(); }, REFIT_DELAY_MILLISECONDS);
  }
  if (window.ResizeObserver) {
    new window.ResizeObserver(function (entries) {
      // a canvas with no size is off screen; fitting it there would fit nothing
      const box = entries[entries.length - 1].contentRect;
      if (box.width && box.height) refitWhenSettled();
    }).observe(canvasEl);
  } else {
    window.addEventListener('resize', refitWhenSettled);
  }

  // %% zoom controls
  // one step in and its exact inverse out, so clicking + then − lands where it started
  const ZOOM_STEP = 1.3;
  root.querySelector('#graph-zoom-in').addEventListener('click', function () { Graph.zoomBy(ZOOM_STEP); });
  root.querySelector('#graph-zoom-out').addEventListener('click', function () { Graph.zoomBy(1 / ZOOM_STEP); });
  root.querySelector('#graph-zoom-fit').addEventListener('click', function () { Graph.fit(); });

  // %% tabs
  // in the order the tabs are rendered in
  const TABS = {
    knowledge:  { url: '/api/knowledge' },
    plan:       { url: '/api/knowledge/view?name=plan' },
    chart:      { url: '/api/knowledge/view?name=chart' },
    kinematics: { url: '/api/knowledge/view?name=kinematics' },
    transforms: { url: '/api/knowledge/view?name=transforms' },
  };
  // the tab the panel opens on: the first one, so the highlighted tab is the leftmost
  const DEFAULT_TAB = Object.keys(TABS)[0];
  // the bridge endpoint each live view polls
  const LIVE_ENDPOINT = { plan: '/plan', chart: '/chart', transforms: '/transforms' };
  let tab = DEFAULT_TAB;
  let view = null;            // the currently rendered payload
  const base = {};            // tab -> payload as loaded from the server
  const shown = {};           // tab -> payload currently rendered (drill-downs)
  const stacks = {};          // tab -> parent payloads for the back button
  Object.keys(TABS).forEach(function (t) { stacks[t] = []; });
  let inGraphSet = {};

  // %% Plan tab: readable step-list rendering (an alternative to the vis graph)
  let stepsMode = true;   // Plan tab opens in the readable Steps view; the toggle switches to the graph
  const stepsEl = root.querySelector('#graph-steps');
  const stepsToggle = root.querySelector('#graph-steps-toggle');
  const STEP_KINDS = { ActionNode: 'action', AttachNode: 'attach', DetachNode: 'attach' };
  // conditions are internal checks that never execute — hidden from this view entirely
  const DETAIL_KINDS = { MotionNode: 'motion', MonitorNode: 'monitor' };
  const IGNORE_KINDS = { ConditionNode: 1 };
  const STRUCT_KINDS = { SequentialNode: 1, ParallelNode: 1, UnderspecifiedNode: 1 };
  const STEP_STATUS = { SUCCEEDED: 'done', DONE: 'done', RUNNING: 'running', FAILED: 'failed', CREATED: 'not started', NOT_STARTED: 'not started' };
  function stepWords(x) { return String(x || '').replace(/([a-z0-9])([A-Z])/g, '$1 $2').trim().toLowerCase().replace(/^./, function (c) { return c.toUpperCase(); }); }
  function stepLabel(n) {
    if (n.kind === 'ConditionNode') return 'condition check';
    if (n.kind === 'AttachNode') return 'grasp' + (n.target ? ' ' + String(n.target).replace(/\.(stl|obj|dae)$/i, '') : '');
    if (n.kind === 'DetachNode') return 'release' + (n.target ? ' ' + String(n.target).replace(/\.(stl|obj|dae)$/i, '') : '');
    if (n.kind === 'MotionNode') return stepWords((n.label || 'motion').replace(/Motion$/, '')) || 'motion';
    var l = stepWords((n.label || n.kind).replace(/(Action|Node)$/, ''));
    if (n.target) l += ' ' + String(n.target).replace(/\.(stl|obj|dae)$/i, '');
    return l || 'step';
  }
  function stepTreeFrom(nodes) {
    const by = {}, roots = [];
    nodes.forEach(function (n) { by[n.id] = { n: n, kids: [] }; });
    nodes.forEach(function (n) { const o = by[n.id]; if (n.parent && by[n.parent]) by[n.parent].kids.push(o); else roots.push(o); });
    return roots;
  }
  // a replayed plan carries no statuses (only the live bridge streams them), and a step
  // with nothing to report shows no pill rather than an invented "not started"
  function stepPill(status) {
    if (!status) return '';
    const key = status === 'NOT_STARTED' ? 'CREATED' : status;
    return '<span class="sp sp-' + key + '">' + (STEP_STATUS[status] || String(status).toLowerCase()) + '</span>';
  }
  // flatten structural containers; keep action/attach as numbered steps, details collapsed
  function stepItems(node) {
    const out = [];
    (node.kids || []).forEach(function (c) {
      const k = c.n.kind;
      if (STEP_KINDS[k]) out.push(c);
      else if (STRUCT_KINDS[k]) Array.prototype.push.apply(out, stepItems(c));
    });
    return out;
  }
  // %% constraints palette (in-tab): natural-language -> real giskard goal -> POST /constraint
  let CONSTRAINTS = [
    { id: 'c1', text: 'Milk must always stay upright' },
    { id: 'c2', text: 'Robot must look where it operates' },
    { id: 'c3', text: 'Keep the gripper closed while carrying' },
  ];
  let cSeq = 4;
  function objIn(text, node) {
    const m = String(text).toLowerCase().match(/\b(milk|bowl|spoon|fork|knife|plate|cup|mug|tray|bottle|flask|vial|beaker|tube|rack|sample|cereal|box|jar|glass|can)\b/);
    if (m) return m[1];
    if (node && node.target) return String(node.target).replace(/\.(stl|obj|dae)$/i, '');
    return 'object';
  }
  // read a length in metres from the text ("10 cm", "0.1 m", "5mm"), or null
  function lenIn(text) {
    const m = String(text).toLowerCase().match(/(\d+(?:\.\d+)?)\s*(mm|cm|centimet(?:er|re)s?|m\b|met(?:er|re)s?)/);
    if (!m) return null;
    const v = parseFloat(m[1]), u = m[2];
    if (u.indexOf('mm') === 0) return v / 1000;
    if (u.indexOf('c') === 0) return v / 100;
    return v;
  }
  function compileConstraint(text, node) {
    const t = String(text).toLowerCase();
    const o = objIn(text, node);
    const d = lenIn(t);
    // orientation: keep upright / level / from tipping or spilling
    if (/upright|stand up|stay up|vertical|straight up|tip over|tips?\b|tilt|spill|level|flat|horizontal|steady|balanc|no spill|don.?t (tip|spill|tilt)/.test(t))
      return { goal: 'VectorsAligned', params: { root_link: 'map', tip_link: o, tip_normal: [0, 0, 1], goal_normal: [0, 0, 1], threshold: 0.1 } };
    // gaze: look at / watch / keep in view  (check before generic words)
    if (/look|watch|gaze|point (at|the camera)|face the|observ|keep .*(in view|an eye)|focus on|keep sight|see the|where it (operat|work)/.test(t))
      return { goal: 'PointingAt', params: { tip_link: 'head_camera', root_link: 'map', pointing_axis: [0, 0, 1], goal_point: '@operation_target', goal_point_body: o, threshold: 0.05 } };
    // (gripper open/close is intentionally not mapped yet — the bridge can't resolve the
    //  joint reliably across robots, so such phrasings return "no match" rather than
    //  looking accepted and then failing as "not applicable")
    // height: keep high / above / off the surface
    if (/above|higher|over the|off the (table|ground|surface|bench)|keep .*(high|up high|elevated)|lift(ed)? (up|above)?/.test(t))
      return { goal: 'HeightMonitor', params: { tip_link: o, lower_limit: (d != null ? d : 0.05), upper_limit: 2.0 } };
    // height: keep low / below
    if (/below|under(neath)?|lower than|keep .*(low|down|close to the (table|surface|ground))/.test(t))
      return { goal: 'HeightMonitor', params: { tip_link: o, lower_limit: 0.0, upper_limit: (d != null ? d : 0.1) } };
    // distance / clearance / avoid
    if (/away from|keep .*clear|clearance|distance|avoid|don.?t (hit|touch|collide|bump)|too close|stay .*away|far from|min(imum)? distance/.test(t))
      return { goal: 'DistanceMonitor', params: { tip_link: o, lower_limit: (d != null ? d : 0.05), upper_limit: 5.0 } };
    return { goal: null, params: {} };
  }
  // the ⓘ panel: how the rule-based translation maps phrasings to giskardpy goals
  function constraintInfoHtml() {
    const rows = [
      ['upright, level, flat, tilt, spill, steady, balanced, "don\'t tip"', 'VectorsAligned', "keep the object's up-axis aligned with world up (stays upright / level)"],
      ['look, watch, observe, "keep in view", focus, gaze, face', 'PointingAt', 'aim the head camera at the object'],
      ['above, higher, "off the table", "keep high", lift', 'HeightMonitor', 'keep the object at/above a height over the reference'],
      ['below, under, "lower than", "keep low"', 'HeightMonitor', 'keep the object below a height'],
      ['"away from", clearance, distance, avoid, "too close", "keep clear"', 'DistanceMonitor', 'keep a minimum distance / clearance'],
    ];
    let body = rows.map(function (r) {
      return '<tr><td>' + r[0] + '</td><td class="goal">' + r[1] + '</td><td>' + r[2] + '</td></tr>';
    }).join('');
    return '<div class="cpal-info">' +
      '<div class="ci-h">How constraints are translated <span class="ci-note">(rule-based, not an LLM)</span></div>' +
      '<table class="ci-table"><thead><tr><th>Phrasing (examples)</th><th>giskardpy goal</th><th>Effect</th></tr></thead><tbody>' + body + '</tbody></table>' +
      '<div class="ci-foot">A length in the text — <code>10 cm</code>, <code>0.1 m</code>, <code>5mm</code> — sets the thresholds/limits. ' +
      'The object is taken from the sentence (milk, bowl, spoon, tray, flask, vial, beaker, …) or falls back to the step\'s target. ' +
      'Anything that matches no rule shows <b>no match</b> and is not sent. ' +
      'Gripper open/close is not wired yet (the joint can\'t be resolved reliably across robots).</div>' +
      '</div>';
  }

  const stepNodeById = {};   // id -> raw plan node, filled during renderSteps
  const stepAttached = {};   // id -> [{goal, text}]  attached constraints, shown as persistent chips
  function attachedChips(nodeId) {
    return (stepAttached[nodeId] || []).map(function (a) {
      return '<span class="st-chip" title="' + a.text + '">⛓ ' + a.goal + '</span>';
    }).join('');
  }
  function postConstraint(nodeId, cid) {
    const node = stepNodeById[nodeId];
    const c = CONSTRAINTS.find(function (x) { return x.id === cid; });
    if (!node || !c) return;
    const comp = compileConstraint(c.text, node);
    if (!comp.goal) { flashStep(nodeId, 'no match', false); return; }
    if (!liveState.url) { flashStep(nodeId, 'not live', false); return; }
    const body = { op: 'attach_monitor', text: c.text, apply: 'next_activation',
      target_plan_node: { id: node.id, kind: node.kind, label: node.label },
      giskard_node: { type: comp.goal, params: comp.params } };
    fetch(liveState.url + '/constraint', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })
      .then(function (r) { return r.json(); })
      .then(function (j) {
        if (j.ok) {
          (stepAttached[nodeId] = stepAttached[nodeId] || []).push({ goal: comp.goal, text: c.text });
          const meta = stepsEl.querySelector('.st-row[data-id="' + nodeId + '"] .st-meta');
          if (meta) meta.insertAdjacentHTML('afterbegin', '<span class="st-chip" title="' + c.text + '">⛓ ' + comp.goal + '</span>');
        } else { flashStep(nodeId, '✗ ' + (j.error || 'error'), false); }
      })
      .catch(function (e) { flashStep(nodeId, '✗ ' + e, false); });
  }
  function flashStep(nodeId, msg, ok) {
    const row = stepsEl.querySelector('.st-row[data-id="' + nodeId + '"]');
    if (!row) return;
    let tag = row.querySelector('.st-cflash');
    if (!tag) { tag = document.createElement('span'); tag.className = 'st-cflash'; row.querySelector('.st-meta').prepend(tag); }
    tag.textContent = '⛓ ' + msg; tag.classList.toggle('bad', !ok);
  }
  // status is reported on the leaf motion nodes, not on the action node shown as a step;
  // conditions never execute (stay CREATED), so they are excluded. A node with real
  // children derives purely from them (all done -> done), so a stale own "RUNNING" never
  // keeps a step running once its motions have finished.
  function derivedStatus(item) {
    let anyRunning = false, anyFailed = false, seen = 0, done = 0, reported = false;
    (function scan(it) {
      (it.kids || []).forEach(function (c) {
        if (IGNORE_KINDS[c.n.kind]) return;      // ignore conditions entirely
        seen++;
        const s = c.n.status;
        if (s) reported = true;
        if (s === 'RUNNING') anyRunning = true;
        if (s === 'FAILED') anyFailed = true;
        if (s === 'SUCCEEDED' || s === 'DONE') done++;
        scan(c);
      });
    })(item);
    if (anyFailed) return 'FAILED';
    if (seen > 0) {                              // has real children: derive from them
      if (!reported) return null;                // none of them reports one either
      if (done === seen) return 'SUCCEEDED';
      if (anyRunning || done > 0) return 'RUNNING';
      return 'CREATED';
    }
    return item.n.status;                        // a leaf uses its own status
  }

  const stepsCollapsed = {};   // node id -> true when the user collapsed that step (kept across live refreshes)
  let stepCollapsibleIds = [];  // ids of steps that have children, filled during render (for expand/collapse all)
  let lastStepsPayload = null;  // last plan payload rendered, so the all-buttons can re-render
  function collapseAllSteps(collapse) {
    if (collapse) stepCollapsibleIds.forEach(function (id) { stepsCollapsed[id] = true; });
    else Object.keys(stepsCollapsed).forEach(function (k) { delete stepsCollapsed[k]; });
    if (lastStepsPayload) renderSteps(lastStepsPayload);
  }
  function renderPaletteCards() {
    return CONSTRAINTS.map(function (c) {
      return '<div class="cpal-card" draggable="true" data-cid="' + c.id + '"><span class="cpal-grip">⠿</span>' + c.text + '<span class="cpal-del" data-del="' + c.id + '">×</span></div>';
    }).join('');
  }
  function wirePaletteCards() {
    const pal = stepsEl.querySelector('.cpal'); if (!pal) return;
    pal.querySelectorAll('.cpal-card').forEach(function (card) {
      card.addEventListener('dragstart', function (e) { e.dataTransfer.setData('text/plain', 'c:' + card.dataset.cid); e.dataTransfer.effectAllowed = 'copy'; });
    });
    pal.querySelectorAll('.cpal-del').forEach(function (x) {
      x.addEventListener('click', function (e) { e.stopPropagation(); CONSTRAINTS = CONSTRAINTS.filter(function (c) { return c.id !== x.dataset.del; }); refreshPaletteCards(); });
    });
  }
  function refreshPaletteCards() {
    const list = stepsEl.querySelector('.cpal-list'); if (list) { list.innerHTML = renderPaletteCards(); wirePaletteCards(); }
  }
  // build the palette ONCE (so the live tree refresh never wipes the input or a card)
  function ensurePalette() {
    if (stepsEl.querySelector('.cpal')) return;
    stepsEl.innerHTML =
      '<div class="cpal">' +
      '  <div class="cpal-h">Constraints <button class="cpal-info-btn" title="How natural-language constraints are translated">ⓘ</button></div>' +
      '  <div class="cpal-sub">drag onto a step →</div>' +
      '  <div class="cpal-list">' + renderPaletteCards() + '</div>' +
      '  <div class="cpal-add"><input class="cpal-in" placeholder="e.g. milk must stay upright"><button class="cpal-btn">Add</button></div>' +
      constraintInfoHtml() +
      '</div>' +
      '<div class="cpal-resizer" title="Drag to resize"></div>' +
      '<div class="steps-pane">' +
      '  <div class="steps-toolbar"><button class="steps-tool-btn" data-tree="expand" title="Expand all steps">⊕ Expand all</button>' +
      '     <button class="steps-tool-btn" data-tree="collapse" title="Collapse all steps">⊖ Collapse all</button></div>' +
      '  <div class="steps-tree"></div></div>';
    wirePaletteCards();
    stepsEl.querySelectorAll('.steps-tool-btn').forEach(function (b) {
      b.addEventListener('click', function () { collapseAllSteps(b.dataset.tree === 'collapse'); });
    });
    const infoBtn = stepsEl.querySelector('.cpal-info-btn'), infoBox = stepsEl.querySelector('.cpal-info');
    if (infoBtn && infoBox) infoBtn.addEventListener('click', function () { infoBox.classList.toggle('open'); });
    const inp = stepsEl.querySelector('.cpal-in'), btn = stepsEl.querySelector('.cpal-btn');
    function add() { const v = inp.value.trim(); if (!v) return; CONSTRAINTS.push({ id: 'c' + (cSeq++), text: v }); inp.value = ''; refreshPaletteCards(); inp.focus(); }
    if (btn) btn.addEventListener('click', add);
    if (inp) inp.addEventListener('keydown', function (e) { if (e.key === 'Enter') { e.preventDefault(); add(); } });
    wireResizer();
  }
  // drag the divider between the constraints column and the step list; width persists
  function wireResizer() {
    const cpal = stepsEl.querySelector('.cpal'), rez = stepsEl.querySelector('.cpal-resizer');
    if (!cpal || !rez) return;
    const saved = parseInt(localStorage.getItem('cramera.plan.cpalWidth') || '', 10);
    if (saved >= 140 && saved <= 560) cpal.style.width = saved + 'px';
    rez.addEventListener('mousedown', function (e) {
      e.preventDefault();
      const startX = e.clientX, startW = cpal.getBoundingClientRect().width;
      document.body.style.userSelect = 'none'; rez.classList.add('dragging');
      function move(ev) {
        const w = Math.max(140, Math.min(560, startW + (ev.clientX - startX)));
        cpal.style.width = w + 'px';
      }
      function up() {
        document.removeEventListener('mousemove', move); document.removeEventListener('mouseup', up);
        document.body.style.userSelect = ''; rez.classList.remove('dragging');
        localStorage.setItem('cramera.plan.cpalWidth', String(Math.round(cpal.getBoundingClientRect().width)));
      }
      document.addEventListener('mousemove', move); document.addEventListener('mouseup', up);
    });
  }
  function renderSteps(payload) {
    ensurePalette();
    lastStepsPayload = payload;
    const treeEl = stepsEl.querySelector('.steps-tree'); if (!treeEl) return;
    const roots = stepTreeFrom(payload.nodes || []);
    const top = roots.length === 1 && STRUCT_KINDS[roots[0].n.kind] ? stepItems(roots[0]) : roots;
    for (const k in stepNodeById) delete stepNodeById[k];
    stepCollapsibleIds = [];
    const html = [];
    function walk(item, number, depth) {
      const n = item.n;
      stepNodeById[n.id] = n;
      const sub = stepItems(item);
      const details = (item.kids || []).filter(function (c) { return DETAIL_KINDS[c.n.kind]; });
      const hk = sub.length || details.length;
      if (hk) stepCollapsibleIds.push(n.id);
      const collapsed = !!stepsCollapsed[n.id];
      const lvl = Math.min(depth, 3);
      html.push('<div class="st-row lvl' + lvl + (hk ? ' hk' : '') + '" data-id="' + n.id + '">' +
        '<span class="st-tw">' + (hk ? (collapsed ? '▸' : '▾') : '') + '</span>' +
        '<span class="st-num">' + number + '</span>' +
        '<span class="st-name">' + stepLabel(n) + '</span>' +
        '<span class="st-meta">' + attachedChips(n.id) + (details.length ? '<span class="st-dc">' + details.length + ' detail' + (details.length > 1 ? 's' : '') + '</span>' : '') + stepPill(derivedStatus(item)) + '</span>' +
        '</div>');
      if (hk) {
        html.push('<div class="st-kids"' + (collapsed ? ' style="display:none"' : '') + '>');
        details.forEach(function (d) { html.push('<div class="st-leaf"><span class="st-name detail">' + stepLabel(d.n) + '</span><span class="st-meta">' + stepPill(d.n.status) + '</span></div>'); });
        var i = 1; sub.forEach(function (c) { walk(c, number + '.' + (i++), depth + 1); });
        html.push('</div>');
      }
    }
    var i = 1; top.forEach(function (c) { walk(c, String(i++), 0); });
    treeEl.innerHTML = html.join('');
    // collapse/expand (persist the state so the 700ms live refresh keeps it)
    treeEl.querySelectorAll('.st-row.hk').forEach(function (r) {
      r.addEventListener('click', function () {
        const id = r.dataset.id, kids = r.nextElementSibling;
        if (kids && kids.classList.contains('st-kids')) {
          const nowCollapsed = kids.style.display !== 'none';
          kids.style.display = nowCollapsed ? 'none' : '';
          stepsCollapsed[id] = nowCollapsed;
          r.querySelector('.st-tw').textContent = nowCollapsed ? '▸' : '▾';
        }
      });
    });
    // each step row is a constraint drop target
    treeEl.querySelectorAll('.st-row[data-id]').forEach(function (row) {
      row.addEventListener('dragover', function (e) { e.preventDefault(); row.classList.add('st-drop'); });
      row.addEventListener('dragleave', function () { row.classList.remove('st-drop'); });
      row.addEventListener('drop', function (e) {
        e.preventDefault(); e.stopPropagation(); row.classList.remove('st-drop');
        const d = e.dataTransfer.getData('text/plain') || '';
        if (d.indexOf('c:') === 0) postConstraint(row.dataset.id, d.slice(2));
      });
    });
  }
  function maybeRenderSteps(payload) {
    const on = stepsMode && tab === 'plan';
    const legend = root.querySelector('#legend');
    // the graph's overlay controls (zoom, fullscreen) sit on the left and would cover
    // the constraints column, so hide them while the step list is shown
    ['.graph-zoom', '.graph-max-btn'].forEach(function (sel) {
      const e = root.querySelector(sel); if (e) e.style.display = on ? 'none' : '';
    });
    if (on && payload && (payload.nodes || []).length) {
      canvasEl.style.display = 'none'; if (legend) legend.style.display = 'none';
      stepsEl.style.display = ''; renderSteps(payload);
    } else {
      stepsEl.style.display = 'none'; canvasEl.style.display = ''; if (legend) legend.style.display = '';
    }
  }
  // the toggle label names the view you switch TO, so its state is never ambiguous
  function updateStepsToggle() {
    if (!stepsToggle) return;
    stepsToggle.textContent = stepsMode ? '◫ Graph view' : '☰ Step list';
    stepsToggle.classList.toggle('on', stepsMode);
  }
  if (stepsToggle) stepsToggle.addEventListener('click', function () {
    stepsMode = !stepsMode;
    updateStepsToggle();
    maybeRenderSteps(shown[tab] || base[tab]);
  });

  function setView(payload) {
    view = payload;
    shown[tab] = payload;
    inGraphSet = {};
    payload.nodes.forEach(function (n) { inGraphSet[n.id] = 1; });
    if (emptyEl) {
      const empty = !payload.nodes.length;
      emptyEl.style.display = empty ? '' : 'none';
      emptyEl.textContent = empty ? (payload.empty || 'Nothing to show in this view.') : '';
    }
    // before building: the renderer sizes its canvas from the container it is handed,
    // so a graph built while the step list still covers the canvas comes out empty
    maybeRenderSteps(payload);
    Graph.build({
      nodes: payload.nodes, edges: payload.edges, legend: payload.legend,
      layout: payload.layout, arrows: !!payload.arrows,
      // a view may name the statuses its legend lists, instead of taking the default
      statusLegend: payload.statusLegend || false,
      key: (payload.key || tab) + '#' + stacks[tab].length,
    });
    updateNav();
  }
  function updateNav() {
    const inside = stacks[tab].length > 0;
    navEl.style.display = inside ? '' : 'none';
    if (inside) {
      const path = stacks[tab].slice(1).map(function (v) { return v.breadcrumb; }).concat([view.breadcrumb]);
      navPath.textContent = path.join(' / ');
    }
  }
  async function drill(id) {
    if (!view.details[id]) return;
    try {
      const r = await fetch(SceneContext.withScene('/api/knowledge/expand?node=' + encodeURIComponent(id)));
      const p = await ResponseUtil.parseJson(r);
      if (!p.ok) return;                       // node has no inside view
      stacks[tab].push(view);
      setView(p);
      select(id);
    } catch (err) { /* server unreachable — stay where we are */ }
  }
  function goBack() { if (stacks[tab].length) setView(stacks[tab].pop()); }
  function goHome() {
    if (!stacks[tab].length) return;
    stacks[tab] = [];
    setView(base[tab]);
  }
  navUp.addEventListener('click', goBack);
  navHome.addEventListener('click', goHome);

  // %% loading a tab
  // What one tab's request answered: its view, or why the reader cannot have it.
  async function loadedView(name) {
    try {
      const response = await fetch(SceneContext.withScene(TABS[name].url));
      if (response.status === 404) {
        return { error: 'this build needs the /api/knowledge/view route — restart the server' };
      }
      const payload = await ResponseUtil.parseJson(response);
      if (!payload.ok) return { error: payload.error || 'view unavailable' };
      payload.key = name;
      return { payload: payload };
    } catch (err) {
      return { error: 'Could not load this view: ' + ((err && err.message) || err) };
    }
  }
  async function showTab(name) {
    if (!TABS[name]) return;
    tab = name;
    tabsEl.querySelectorAll('button').forEach(function (b) {
      b.classList.toggle('active', b.dataset.view === name);
    });
    if (!base[name]) {
      emptyEl.style.display = '';
      emptyEl.textContent = 'loading…';
      const loaded = await loadedView(name);
      // a view is fetched, so the reader can be on another tab by the time it arrives:
      // drawing it there would show one tab's graph under another tab's name
      if (tab !== name) return;
      if (loaded.error) { emptyEl.textContent = loaded.error; return; }
      base[name] = loaded.payload;
    }
    if (stepsToggle) { stepsToggle.style.display = (name === 'plan') ? '' : 'none'; updateStepsToggle(); }
    setView(shown[name] || base[name]);
    liveRefresh(true);            // a live tab picks the bridge status up at once
    showRecordedStatechart();     // and a replayed one, the moment it is playing
  }
  tabsEl.querySelectorAll('button').forEach(function (b) {
    b.addEventListener('click', function () { showTab(b.dataset.view); });
  });
  bus.on('graph:view', function (payload) { showTab(payload.name); });

  // %% node click → describe in whatever panel listens
  function select(id) {
    const d = view && view.details && view.details[id];
    if (!d) return;
    const relations = (view.edges || [])
      .filter(function (e) { return e.from === id || e.to === id; })
      .map(function (e) {
        return { s: labelOf(e.from), p: e.label || e.kind, o: labelOf(e.to) };
      });
    bus.emit('entity:select', { id: id, detail: d, relations: relations });
    spotlight({ ids: [id], focus: id });
  }
  function labelOf(id) { return (view.details[id] && view.details[id].label) || id; }
  Graph.onSelect(select);
  Graph.onDoubleSelect(drill);

  // %% highlights (from EQL results or our own selection)
  function spotlight(p) {
    const ids = (p && p.ids) || [];
    let hi = ids.filter(function (id) { return inGraphSet[id]; });
    if (p && p.focus && inGraphSet[p.focus]) {
      const neighbours = (view.edges || [])
        .filter(function (e) { return e.from === p.focus || e.to === p.focus; })
        .map(function (e) { return e.from === p.focus ? e.to : e.from; });
      hi = hi.concat(neighbours.filter(function (id) { return inGraphSet[id]; }));
    }
    if (hi.length) Graph.highlight(hi); else Graph.reset();
  }
  bus.on('entity:highlight', spotlight);
  bus.on('scene:step', function (p) {
    if (p.step === '__done__') { Graph.reset(); return; }
    if (tab === 'knowledge' && !stacks[tab].length && inGraphSet[p.step]) select(p.step);
  });

  // %% live status overlay (Plan / Statechart tabs)
  // The bridge publishes the plan tree and the executing motion statechart with
  // per-node status. Structure changes (the plan grows as actions expand, a new
  // statechart is compiled per motion group) rebuild the graph; a pure status
  // change only re-colours the rings, so the layout never jumps.
  const CHART_LEGEND = [
    { group: 'task', label: 'Task (motion constraint)' },
    { group: 'monitor', label: 'Monitor / observation' },
    { group: 'motion_goal', label: 'Goal (contains nodes)' },
    { group: 'motion_end', label: 'End / cancel motion' },
  ];
  const TRANSFORM_LEGEND = [
    { group: 'world_frame', label: 'World root' },
    { group: 'actuated_frame', label: 'Actuated joint' },
    { group: 'free_frame', label: 'Free-floating' },
    { group: 'fixed_frame', label: 'Fixed to its parent' },
  ];
  const FRESHNESS_LEGEND = ['MOVING', 'SETTLED', 'STALE', 'STATIC'];
  const FRAME_GROUP = { actuated: 'actuated_frame', free: 'free_frame', fixed: 'fixed_frame' };
  const liveSig = { plan: '', chart: '', transforms: '' };
  let liveTimer = null;
  let liveState = { on: false, url: '' };

  function liveSource() {
    if (tab === 'plan') return 'plan';
    const p = shown[tab] || base[tab];
    return (p && p.live) || null;               // 'plan' | 'chart' | null
  }

  // drop the redundant 'Action' suffix only — a label that merely contains the word,
  // such as 'ActionNode', must survive intact. Mirrors
  // PlanViewPayload._shorten_action_label: the bridge sends the raw designator name,
  // so the live path shortens it here.
  function shortenActionLabel(label) {
    const shortened = label.replace(/Action$/, '');
    return shortened || label;
  }

  function planPayload(live) {
    const nodes = [], edges = [], details = {};
    (live.nodes || []).forEach(function (n) {
      const label = shortenActionLabel(n.label || '?');
      const lines = ['a ' + n.kind,
                     'status: ' + n.status + (n.derived ? ' (derived from the motion statechart)' : '')];
      if (n.arm) lines.push('arm: ' + n.arm);
      if (n.target) lines.push('target: ' + n.target);
      nodes.push({ id: n.id, label: n.label, group: n.group,
                   kind: n.kind, parent: n.parent, target: n.target, arm: n.arm,
                   title: [label].concat(lines).join('\n'), status: n.status });
      details[n.id] = { label: label, group: n.group, lines: lines };
      if (n.parent) edges.push({ from: n.parent, to: n.id, kind: 'property', label: 'has step' });
    });
    return { ok: true, breadcrumb: 'live plan', nodes: nodes, edges: edges, details: details,
             legend: live.legend || [], layout: 'hier', arrows: true, statusLegend: true,
             live: 'plan', key: 'plan-live',
             empty: 'The bridge is attached but the demo has not started its plan yet.' };
  }

  function chartPayload(live) {
    const nodes = [], edges = [], details = {}, isParent = {};
    (live.nodes || []).forEach(function (n) { if (n.parent) isParent[n.parent] = 1; });
    (live.nodes || []).forEach(function (n) {
      const group = isParent[n.id] ? 'motion_goal'
        : /EndMotion|CancelMotion/.test(n.class_name) ? 'motion_end'
        : /Monitor|Reached|Observation|Condition/.test(n.class_name + n.name) ? 'monitor' : 'task';
      const lines = ['a ' + n.class_name, 'life cycle: ' + n.life_cycle, 'observation: ' + n.observation];
      nodes.push({ id: n.id, label: n.name, group: group,
                   title: [n.name].concat(lines).join('\n'), status: n.life_cycle });
      details[n.id] = { label: n.name, group: group, lines: lines };
      if (n.parent) edges.push({ from: n.parent, to: n.id, kind: 'type', label: 'contains' });
    });
    (live.edges || []).forEach(function (e) {
      edges.push({ from: e.from, to: e.to, kind: e.kind, label: (e.kind || '').toLowerCase() + ' transition' });
    });
    return { ok: true, breadcrumb: 'statechart' + (live.title ? ' · ' + live.title : ''),
             nodes: nodes, edges: edges, details: details, legend: CHART_LEGEND,
             layout: 'hier', arrows: true, statusLegend: true, live: 'chart', key: 'chart-live',
             empty: 'Attached, but no motion statechart is executing right now.' };
  }

  // The world's connection graph: one node per frame, one edge per connection. A
  // frame's ring is the freshness of the connection that carries it — the root frame,
  // which no connection carries, cannot go stale.
  function transformsPayload(live) {
    const nodes = [], edges = [], details = {}, carried = {};
    const connections = live.connections || [];
    connections.forEach(function (c) { carried[c.child] = c; });
    const frames = {};
    connections.forEach(function (c) { frames[c.parent] = 1; frames[c.child] = 1; });
    Object.keys(frames).forEach(function (frame) {
      const connection = carried[frame];
      const group = connection ? (FRAME_GROUP[connection.kind] || 'fixed_frame') : 'world_frame';
      const status = connection ? connection.freshness : 'STATIC';
      const lines = connection
        ? ['a frame carried by ' + connection.name,
           'connection: ' + connection.kind,
           'last written by: ' + connection.writer,
           'last changed: ' + (connection.ageSeconds === null || connection.ageSeconds === undefined
             ? 'never, since the bridge attached'
             : connection.ageSeconds + ' s ago')]
        : ['the world root frame'];
      nodes.push({ id: frame, label: frame, group: group,
                   title: [frame].concat(lines).join('\n'), status: status });
      details[frame] = { label: frame, group: group, lines: lines };
    });
    connections.forEach(function (c) {
      edges.push({ from: c.parent, to: c.child, kind: c.kind, label: c.name });
    });
    return { ok: true, breadcrumb: 'transform graph', nodes: nodes, edges: edges,
             details: details, legend: TRANSFORM_LEGEND, layout: 'hier', arrows: true,
             statusLegend: FRESHNESS_LEGEND, live: 'transforms', key: 'transforms-live',
             empty: 'Attached, but the demo has not published a world yet.' };
  }

  function livePayload(source, live) {
    if (source === 'plan') return planPayload(live);
    if (source === 'chart') return chartPayload(live);
    return transformsPayload(live);
  }

  async function liveRefresh(force) {
    const src = liveSource();
    const active = !!src && liveState.on;
    liveBadge.classList.toggle('on', active);
    if (!active) return;
    if (stacks[tab].length) return;              // inside a drill-down: leave it alone
    let live;
    try {
      live = await fetch(liveState.url + LIVE_ENDPOINT[src]).then(ResponseUtil.parseJson);
    } catch (err) { return; }                    // bridge gone — the 3D side handles it
    if (!live || !(live.nodes || live.connections)) return;
    const payload = livePayload(src, live);
    if (force || live.signature !== liveSig[src]) {    // structure changed → rebuild
      liveSig[src] = live.signature;
      base[tab] = payload;
      setView(payload);
      return;
    }
    // same structure: only re-colour, and keep the detail lines in sync
    const map = {};
    payload.nodes.forEach(function (n) { map[n.id] = n.status; });
    if (!Graph.setStatuses(map)) { base[tab] = payload; setView(payload); return; }
    base[tab] = payload;
    if (view && view.details) view.details = payload.details;
    maybeRenderSteps(payload);
  }

  bus.on('live:changed', function (p) {
    liveState = { on: !!p.on, url: p.url || '' };
    if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
    if (liveState.on) {
      liveTimer = setInterval(function () { liveRefresh(false); }, 700);
      liveRefresh(true);
    } else {
      liveBadge.classList.remove('on');
      liveSig.plan = liveSig.chart = liveSig.transforms = '';
      // drop the live payloads so the live tabs fall back to the recorded bundle
      const liveTabs = Object.keys(LIVE_ENDPOINT);
      liveTabs.forEach(function (t) { delete base[t]; delete shown[t]; stacks[t] = []; });
      if (liveTabs.indexOf(tab) >= 0) showTab(tab);
    }
  });

  // %% recorded statecharts (Statechart tab of a replay)
  // A recording keeps every statechart its run ticked (see
  // cramera.knowledge.recorded_statecharts); the tab follows the playhead through
  // them, re-colouring rather than rebuilding while the played moments stay inside
  // the same chart.
  const NO_STATECHART = -1;           // mirrors recorded_statecharts.NO_STATECHART
  let recordedFrame = 0;              // the played frame the tab is showing
  let drawnChart = NO_STATECHART;     // index of the recorded chart currently drawn

  function recordedStatecharts() {
    return (base.chart && base.chart.recorded) || null;
  }

  function momentAt(index) {
    const recorded = recordedStatecharts();
    const at = recorded.frames[index];
    return at === undefined || at === NO_STATECHART ? null : recorded.moments[at];
  }

  // one recorded moment in the shape the bridge publishes, so a statechart is drawn
  // by the same renderer whether it is streaming or replayed
  function recordedSnapshot(moment) {
    const chart = recordedStatecharts().charts[moment.chart];
    return {
      signature: chart.signature,
      title: chart.title,
      nodes: chart.nodes.map(function (node, index) {
        return { id: node.id, name: node.name, class_name: node.class_name,
                 parent: node.parent, life_cycle: moment.lifeCycles[index],
                 observation: moment.observations[index] };
      }),
      edges: chart.edges,
    };
  }

  function showRecordedStatechart() {
    if (tab !== 'chart' || liveState.on || stacks.chart.length) return;
    if (!recordedStatecharts()) return;
    const moment = momentAt(recordedFrame);
    if (!moment) { drawnChart = NO_STATECHART; setView(base.chart); return; }
    const payload = chartPayload(recordedSnapshot(moment));
    payload.key = 'chart-recorded-' + moment.chart;
    payload.empty = base.chart.empty;
    if (moment.chart === drawnChart) {
      const statuses = {};
      payload.nodes.forEach(function (node) { statuses[node.id] = node.status; });
      if (Graph.setStatuses(statuses)) {
        shown.chart = payload;
        if (view && view.details) view.details = payload.details;
        return;
      }
    }
    drawnChart = moment.chart;
    setView(payload);
  }

  bus.on('scene:frame', function (p) {
    recordedFrame = p.index;
    showRecordedStatechart();
  });

  // %% boot
  showTab(DEFAULT_TAB);

  return {
    destroy: function () {
      if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
    },
  };
});
