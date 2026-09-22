'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// %% page with deterministic live bridge responses
class RobotAuthoringPage {
  constructor() {
    this.elements = new Map();
    this.requests = [];
    this.snapshot = {robots: []};
    this.selectionStatus = 200;
    this.captureStatus = 200;
    this.context = vm.createContext({
      window: {addEventListener() {}, location: {hostname: 'localhost'}},
      document: {getElementById: (id) => this.element(id), createElement: () => this.element('option' + this.elements.size)},
      fetch: async (url, options) => {
        const route = new URL(url, 'http://localhost').pathname;
        this.requests.push({route, options});
        const code = route === '/robot' ? this.selectionStatus : route === '/robots' ? this.captureStatus : 200;
        const payload = route === '/robots' ? this.snapshot : route === '/robot' ? {error: 'Plan is running'} : route === '/plan' ? {nodes: []} : {objects: {}, ok: true};
        return {ok: code === 200, json: async () => payload};
      },
      setInterval() { return 1; }, clearInterval() {}, setTimeout() {},
    });
    const web = path.join(__dirname, '../../../cramera/src/cramera/web');
    for (const module of ['base_control', 'execution_environment', 'plan_steps', 'plan_constraints', 'builder_state']) {
      const filename = path.join(web, 'core', module + '.js');
      vm.runInContext(fs.readFileSync(filename, 'utf8'), this.context, {filename});
    }
    this.context.PlanConstraints = this.context.window.PlanConstraints;
    const source = fs.readFileSync(path.join(web, 'plan_builder.js'), 'utf8');
    const exports = fs.readFileSync(path.join(__dirname, '../dataset/builder_multi_robot_exports.js'), 'utf8');
    vm.runInContext(source.slice(0, source.indexOf('  // ---------- boot ----------')) + exports, this.context, {filename: path.join(web, 'plan_builder.js')});
    this.api = this.context.window.multiRobotPage;
    this.api.initialize([
      {name: 'PR2', cls: 'PR2', import: 'from semantic_digital_twin.robots.pr2 import PR2', arms: ['LEFT', 'RIGHT', 'BOTH'], steps: ['navigate', 'park_arms']},
      {name: 'Tracy', cls: 'Tracy', import: 'from semantic_digital_twin.robots.tracy import Tracy', arms: ['BOTH'], steps: ['park_arms']},
    ]);
    this.element('pb-env').value = 'apartment.urdf';
    this.element('pb-name').value = 'two_robot_demo';
    this.element('pb-style').value = 'class';
  }
  element(identifier) {
    if (!this.elements.has(identifier)) this.elements.set(identifier, {
      value: '', textContent: '', innerHTML: '', className: '', style: {}, children: [], src: 'index.html?scene',
      listeners: {}, addEventListener(event, callback) { this.listeners[event] = callback; }, replaceChildren() { this.children = []; }, appendChild(child) { this.children.push(child); },
    });
    return this.elements.get(identifier);
  }
}

// %% independent plans, presentation, and model capabilities
async function checkAuthoring() {
  const page = new RobotAuthoringPage();
  page.api.wireRobotControls();
  const first = page.api.state.activeRobot();
  assert.equal(page.element('pb-plan-robot').textContent, 'Plan for ' + first.label);
  assert.equal(page.element('pb-remove-robot').disabled, true);
  assert.equal(page.element('pb-rx').value, first.x);
  for (const [identifier, value, coordinate, expected] of [['pb-rx', '2.5', 'x', 2.5], ['pb-ry', '3.5', 'y', 3.5], ['pb-ryaw', '90', 'yaw', Math.PI / 2]]) {
    const element = page.element(identifier); element.value = value; element.listeners.input.call(element);
    assert.equal(first[coordinate], expected);
  }
  const invalid = page.element('pb-rx'); invalid.value = 'invalid'; invalid.listeners.input.call(invalid);
  assert.equal(first.x, 2.5);
  const label = page.element('pb-robot-label'); label.value = 'Courier'; label.listeners.change.call(label);
  assert.equal(first.label, 'Courier');
  const navigate = {type: 'navigate', params: {x: 2, y: 2, z: 0, yaw: 0}};
  page.api.steps = [navigate];
  page.api.addRobotInstance();
  const second = page.api.state.activeRobot();
  assert.equal(page.api.steps.length, 0);
  assert.equal(page.element('pb-remove-robot').disabled, false);
  assert.equal(page.element('pb-robot-instance').children.length, 2);
  assert.notEqual(second.y, first.y);
  await page.api.selectRobotInstance(first.id);
  assert.deepEqual(page.api.steps, [navigate]);
  page.api.state.updateRobot(first.id, {yaw: Math.PI / 2, label: '<Courier>'});
  page.api.renderRobotInstances();
  assert.equal(page.element('pb-ryaw').value, 90);
  assert.equal(page.element('pb-plan-robot').textContent, 'Plan for <Courier>');
  page.element('pb-robot').value = 'Tracy';
  page.api.selectRobot();
  assert.equal(page.api.state.activeRobot().model, 'Tracy');
  assert.equal(page.api.steps.length, 0);
  assert.equal(page.api.robotImportLines().length, 2);
  page.api.removeRobotInstance();
  assert.equal(page.api.state.activeRobot().id, second.id);
  page.api.removeRobotInstance();
  assert.equal(page.api.state.instances.length, 1);
  assert.equal(page.api.robotImportLines().length, 1);
}

// %% current poses survive selection and scene replacement
async function checkLiveSelection() {
  const page = new RobotAuthoringPage();
  const first = page.api.state.activeRobot();
  page.api.addRobotInstance();
  const second = page.api.state.activeRobot();
  page.api.live = true;
  page.snapshot.robots = [
    {identifier: first.id, model: first.model, pose: [3, 4, 0, 0, 0, 0, 1]},
    {identifier: second.id, model: second.model, pose: [6, 7, 0, 0, 0, Math.SQRT1_2, Math.SQRT1_2]},
  ];
  await page.api.selectRobotInstance(first.id);
  assert.equal(page.api.state.activeRobot(), first);
  assert.equal(first.x, 3);
  assert.equal(second.x, 6);
  assert(Math.abs(second.yaw - Math.PI / 2) < 1e-12);
  const selection = page.requests.find((request) => request.route === '/robot');
  assert.deepEqual(JSON.parse(selection.options.body), {identifier: first.id});
  page.selectionStatus = 409;
  await page.api.selectRobotInstance(second.id);
  assert.equal(page.api.state.activeRobot(), first);
  assert.equal(page.element('pb-robot-instance').value, first.id);
  assert.equal(page.element('pb-status').textContent, 'Plan is running');
  page.selectionStatus = 200;
  page.captureStatus = 503;
  await page.api.selectRobotInstance(second.id);
  assert.equal(page.api.state.activeRobot(), first);
  assert(page.element('pb-status').textContent.includes('Could not read robot poses'));
}

async function checkRunPreservesPoses() {
  const page = new RobotAuthoringPage();
  page.api.addRobotInstance();
  const [first, second] = page.api.state.instances;
  page.api.live = true;
  page.snapshot.robots = [
    {identifier: first.id, model: first.model, pose: [3.25, 4, 0, 0, 0, 0, 1], joint_positions: {'robot_1/torso_lift_joint': 0.21}},
    {identifier: second.id, model: second.model, pose: [6.75, 7, 0, 0, 0, 0, 1]},
  ];
  page.api.steps = [{type: 'navigate', params: {x: 7, y: 7, z: 0, yaw: 0}}];
  await page.api.runPlan();
  const launch = page.requests.find((request) => request.route === '/api/plan/scaffold');
  const source = JSON.parse(launch.options.body).code;
  assert(source.includes('HomogeneousTransformationMatrix.from_xyz_rpy(3.25, 4, 0.0, yaw=0)'));
  assert(source.includes('HomogeneousTransformationMatrix.from_xyz_rpy(6.75, 7, 0.0, yaw=0)'));
  assert(source.includes('active_identifier="' + second.id + '"'));
  assert(source.includes('robot = ROBOT_SCENE.selected_robot(world)'));
  assert(source.includes('joint_positions={"robot_1/torso_lift_joint": 0.21}'));
  assert(source.includes('collision_avoidance=True'));
}

(async () => {
  await checkAuthoring(); await checkLiveSelection(); await checkRunPreservesPoses();
  console.log('Actual Builder callbacks preserve independent plans, active selection, current poses, and collision-aware generated runs.');
})().catch((error) => { console.error(error); process.exitCode = 1; });
