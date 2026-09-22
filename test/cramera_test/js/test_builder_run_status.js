'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// %% page requests and timer callbacks
class RunPage {
  constructor() {
    this.elements = new Map();
    this.timers = [];
    this.requests = [];
    this.plan = {nodes: []};
    this.log = {returncode: null, log: '', alive: true};
    this.pending = new Map();
    this.failures = new Set();
    this.context = vm.createContext({
      window: {addEventListener() {}, location: {hostname: 'localhost'}},
      document: {
        getElementById: (id) => this.element(id),
        createElement: () => ({textContent: '', innerHTML: ''}),
      },
      localStorage: {getItem() { return null; }},
      location: {hostname: 'localhost', protocol: 'http:'},
      setTimeout: (callback) => { this.timers.push(callback); return this.timers.length; },
      setInterval() { return 1; }, clearInterval() {},
      fetch: (url) => this.fetch(url),
    });
    const web = path.join(__dirname, '../../../cramera/src/cramera/web');
    for (const module of ['base_control', 'execution_environment', 'plan_steps', 'plan_constraints', 'builder_state']) {
      const filename = path.join(web, 'core', module + '.js');
      vm.runInContext(fs.readFileSync(filename, 'utf8'), this.context, {filename});
    }
    const source = fs.readFileSync(path.join(web, 'plan_builder.js'), 'utf8');
    const exports = fs.readFileSync(path.join(__dirname, '../dataset/builder_run_exports.js'), 'utf8');
    vm.runInContext(source.slice(0, source.indexOf('  // ---------- boot ----------')) + exports, this.context, {filename: path.join(web, 'plan_builder.js')});
    this.api = this.context.window.builderRunTest;
    this.api.prepare();
  }

  element(id) {
    if (!this.elements.has(id)) this.elements.set(id, {
      value: '', textContent: '', className: '', src: 'index.html?scene',
      style: {}, addEventListener() {},
    });
    return this.elements.get(id);
  }

  async fetch(url) {
    const route = new URL(url, 'http://localhost').pathname;
    this.requests.push(route);
    if (this.failures.has(route)) throw new Error('request unavailable');
    if (this.pending.has(route)) return this.pending.get(route);
    const payload = route === '/plan' ? this.plan : route.endsWith('/log') ? this.log : {ok: true, objects: {}};
    return {ok: true, json: async () => payload};
  }

  defer(route) {
    let resolve;
    this.pending.set(route, new Promise((done) => { resolve = done; }));
    return (payload) => { this.pending.delete(route); resolve({ok: true, json: async () => payload}); };
  }

  root(status, id = 'current') {
    this.plan = {nodes: [{id, parent: null, status}, {id: 'child', parent: id, status: 'FAILED'}]};
  }

  get status() { return this.element('pb-live-status').textContent; }
  async settle() { await new Promise(setImmediate); }
  async tick() { const callbacks = this.timers.splice(0); callbacks.forEach((callback) => callback()); await this.settle(); }
}

// %% authoritative results while the process remains available
async function checkRunStatuses() {
  for (const [status, message, style] of [
    ['SUCCEEDED', 'completed', 'ok'], ['FAILED', 'failed', 'err'], ['INTERRUPTED', 'interrupted', 'err'],
  ]) {
    const page = new RunPage();
    page.root(status);
    page.api.monitorRun(page.api.begin(), null, true);
    await page.settle();
    assert(page.status.includes(message), `${status} must be displayed while the runner is alive; received ${page.status}`);
    assert(page.element('pb-live-status').className.endsWith(style));
    assert.equal(page.api.live, true);
    assert.equal(page.timers.length, 0);
  }

  const stale = new RunPage();
  stale.root('SUCCEEDED', 'previous');
  stale.api.monitorRun(stale.api.begin(), 'previous', true);
  await stale.settle();
  assert.equal(stale.status, '');
  stale.root('RUNNING');
  await stale.tick();
  assert.equal(stale.status, '');
  stale.root('SUCCEEDED');
  await stale.tick();
  assert(stale.status.includes('completed'));

  const unpublished = new RunPage();
  unpublished.api.monitorRun(unpublished.api.begin(), null, true);
  await unpublished.settle();
  assert.equal(unpublished.status, '');
  unpublished.pending.set('/plan', Promise.resolve({ok: false}));
  await unpublished.tick();
  assert.equal(unpublished.status, '');
  unpublished.pending.delete('/plan');
  unpublished.plan = null;
  await unpublished.tick();
  assert.equal(unpublished.status, '');

  const idle = new RunPage();
  idle.root('SUCCEEDED');
  idle.element('pb-live-status').textContent = 'idle scene';
  idle.api.monitorRun(idle.api.begin(), null, false);
  await idle.settle();
  assert.equal(idle.status, 'idle scene');
  assert(!idle.requests.includes('/plan'));
}

// %% placement search remains visible until motion or an authoritative result
async function checkPlacementSearch() {
  const page = new RunPage();
  const runningMessage = 'running before placement';
  page.element('pb-live-status').textContent = runningMessage;
  page.plan = {nodes: [
    {id: 'current', parent: null, kind: 'SequentialNode', label: 'SequentialNode', status: 'RUNNING'},
    {id: 'placement', parent: 'current', kind: 'ActionNode', label: 'MoveAndPlaceAction', status: 'RUNNING'},
    {id: 'approach', parent: 'placement', kind: 'SequentialNode', label: 'SequentialNode', status: 'RUNNING'},
    {id: 'search', parent: 'approach', kind: 'UnderspecifiedNode', label: 'UnderspecifiedNode', status: 'RUNNING'},
    {id: 'old_motion', parent: 'current', kind: 'MotionNode', label: 'MoveJointsMotion', status: 'SUCCEEDED'},
    {id: 'place_motion', parent: 'placement', kind: 'MotionNode', label: 'MovePlacementMotion', status: 'CREATED'},
  ]};
  page.api.monitorRun(page.api.begin(), null, true);
  await page.settle();
  assert(page.status.includes('searching for a reachable placement'), `The active placement search must be visible; received ${page.status}`);
  assert.equal(page.timers.length, 1);
  assert.equal(page.api.live, true);

  const searchMessage = page.status;
  page.pending.set('/plan', Promise.resolve({ok: false}));
  await page.tick();
  assert.equal(page.status, searchMessage);
  assert.equal(page.timers.length, 1);
  page.pending.delete('/plan');

  const motion = {id: 'navigation', parent: 'search', kind: 'MotionNode', label: 'MoveMotion', status: 'RUNNING'};
  page.plan.nodes.push(motion);
  await page.tick();
  assert.equal(page.status, page.context.window.PlanBuilderState.RUN_PROGRESS.RUNNING.message);
  assert.equal(page.timers.length, 1);

  motion.status = 'PAUSED';
  await page.tick();
  assert(!page.status.includes('searching'));

  motion.status = 'SUCCEEDED';
  await page.tick();
  assert.equal(page.status, searchMessage);

  page.plan.nodes[1].status = 'FAILED';
  await page.tick();
  assert(!page.status.includes('searching'), 'A stale running child of a failed placement is not an active search');

  page.plan.nodes[1].label = 'PickUpAction';
  page.plan.nodes[1].status = 'RUNNING';
  await page.tick();
  assert(!page.status.includes('searching'), 'Picking must not be described as placement search');

  page.plan.nodes[1].label = 'MoveAndPlaceAction';
  page.plan.nodes[0].status = 'SUCCEEDED';
  await page.tick();
  assert.equal(page.status, page.context.window.PlanBuilderState.PLAN_RESULTS.SUCCEEDED.message);
  assert.equal(page.timers.length, 0);

  const stale = new RunPage();
  stale.plan = page.plan;
  stale.plan.nodes[0].status = 'RUNNING';
  stale.api.monitorRun(stale.api.begin(), 'current', true);
  await stale.settle();
  assert.equal(stale.status, '');
}

// %% delayed responses cannot overwrite a stop or newer run
async function checkRunRaces() {
  for (const supersede of ['stop', 'new run']) {
    const page = new RunPage();
    const finish = page.defer('/plan');
    page.api.monitorRun(page.api.begin(), null, true);
    await page.settle();
    if (supersede === 'stop') page.api.stopRunMonitor();
    else page.api.begin(false);
    page.element('pb-live-status').textContent = supersede;
    finish({nodes: [{id: 'old', parent: null, status: 'SUCCEEDED'}]});
    await page.settle();
    assert.equal(page.status, supersede);
    assert.equal(page.timers.length, 0);
  }

  const startup = new RunPage();
  const captured = startup.defer('/captured_objects');
  startup.api.pollLive(0, 'running', startup.api.begin(false));
  startup.api.stopRunMonitor();
  startup.element('pb-live-status').textContent = 'stopped';
  captured({objects: {}});
  await startup.settle();
  assert.equal(startup.status, 'stopped');
  assert.equal(startup.api.live, false);

  const starting = new RunPage();
  const previous = starting.defer('/plan');
  const pending = starting.api.runPlan();
  await starting.settle();
  starting.api.stopRunMonitor();
  previous({nodes: []});
  await pending;
  await starting.settle();
  assert(!starting.requests.includes('/api/plan/scaffold'));

  const stopping = new RunPage();
  const stopped = stopping.defer('/api/plan/scaffold/stop');
  stopping.api.stopLive();
  stopping.api.begin(false);
  stopping.element('pb-live-status').textContent = 'new run';
  stopped({ok: true});
  await stopping.settle();
  assert.equal(stopping.status, 'new run');
}

// %% the real Run and Start-live dispatch retain their distinct presentations
async function checkRunStartup() {
  const run = new RunPage();
  run.root('SUCCEEDED', 'previous');
  await run.api.runPlan();
  await run.settle();
  assert(run.status.includes('running'));
  assert.equal(run.api.live, true);
  assert(run.requests.indexOf('/plan') < run.requests.indexOf('/api/plan/scaffold'));
  run.root('SUCCEEDED', 'next');
  await run.tick();
  assert(run.status.includes('completed'));

  const idle = new RunPage();
  idle.root('SUCCEEDED');
  await idle.api.startLive();
  await idle.settle();
  assert(idle.status.includes('live — drag objects'));
  assert.equal(idle.api.live, true);
  await idle.api.stopLive();
  await idle.settle();
  assert.equal(idle.status, 'stopped');
  assert.equal(idle.api.live, false);

  const rejected = new RunPage();
  rejected.pending.set('/api/plan/scaffold', Promise.resolve({ok: true, json: async () => ({ok: false, error: 'launch denied'})}));
  await rejected.api.runPlan();
  await rejected.settle();
  assert.equal(rejected.status, 'failed: launch denied');
  assert.equal(rejected.timers.length, 0);

  const delayed = new RunPage();
  const launched = delayed.defer('/api/plan/scaffold');
  const request = delayed.api.runPlan();
  await delayed.settle();
  delayed.api.stopRunMonitor();
  delayed.element('pb-live-status').textContent = 'stopped';
  launched({ok: true});
  await request;
  await delayed.settle();
  assert.equal(delayed.status, 'stopped');
  assert.equal(delayed.timers.length, 0);

  const crashed = new RunPage();
  const oldCapture = crashed.defer('/captured_objects');
  crashed.log = {returncode: 1, log: 'native process exited'};
  const generation = crashed.api.begin(false);
  crashed.api.pollLive(0, 'running', generation);
  crashed.api.monitorRun(generation, null, true);
  await crashed.settle();
  const failure = crashed.status;
  oldCapture({objects: {}});
  await crashed.settle();
  assert.equal(crashed.status, failure);
  assert.equal(crashed.api.live, false);
}

// %% unavailable bridge responses retain the existing retry and error paths
async function checkRunErrors() {
  for (const offline of [false, true]) {
    const page = new RunPage();
    if (offline) page.failures.add('/captured_objects');
    else page.pending.set('/captured_objects', Promise.resolve({ok: false}));
    const generation = page.api.begin(false);
    page.api.pollLive(0, 'running', generation);
    await page.settle();
    assert.equal(page.timers.length, 1);
    page.api.stopRunMonitor();
    await page.tick();
    assert.equal(page.timers.length, 0);
    page.api.pollLive(40, 'running', page.api.begin(false));
    await page.settle();
    assert(page.status.includes('scene did not come up'));
  }

  const startup = new RunPage();
  startup.pending.set('/captured_objects', Promise.resolve({ok: false}));
  startup.log.returncode = 1;
  startup.api.pollLive(0, 'running', startup.api.begin(false));
  await startup.settle();
  assert(startup.status.includes('Scene process failed'));
  assert.equal(startup.timers.length, 0);

  const staleLog = new RunPage();
  staleLog.pending.set('/captured_objects', Promise.resolve({ok: false}));
  const finishLog = staleLog.defer('/api/plan/scaffold/log');
  staleLog.api.pollLive(0, 'running', staleLog.api.begin(false));
  await staleLog.settle();
  staleLog.api.stopRunMonitor();
  finishLog({returncode: 1, log: 'old failure'});
  await staleLog.settle();
  assert.equal(staleLog.status, '');

  const absentLog = new RunPage();
  absentLog.pending.set('/api/plan/scaffold/log', Promise.resolve({ok: false}));
  absentLog.api.monitorRun(absentLog.api.begin(), null, true);
  await absentLog.settle();
  assert.equal(absentLog.timers.length, 1);
  absentLog.api.stopRunMonitor();
  await absentLog.tick();
  assert.equal(absentLog.timers.length, 0);

  const empty = new RunPage();
  empty.api.clearSteps();
  await empty.api.runPlan();
  assert.equal(empty.status, 'add plan steps first');

  const capture = new RunPage();
  capture.api.rejectCapture();
  await capture.api.runPlan();
  assert.equal(capture.status, 'Could not read object poses: capture unavailable');
  assert(!capture.requests.includes('/api/plan/scaffold'));

  const unavailable = new RunPage();
  unavailable.failures.add('/plan');
  unavailable.failures.add('/api/plan/scaffold');
  await unavailable.api.runPlan();
  assert(unavailable.status.includes('request unavailable'));
  assert.equal(unavailable.timers.length, 0);
}

async function main() { await checkRunStatuses(); await checkPlacementSearch(); await checkRunRaces(); await checkRunStartup(); await checkRunErrors(); }
main().catch((error) => { console.error(error); process.exitCode = 1; });
