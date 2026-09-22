'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {setImmediate: nextTurn} = require('node:timers/promises');

// %% actual authoring callbacks with controlled launch responses
class RobotRestartPage {
  constructor() {
    this.elements = new Map();
    this.requests = [];
    this.snapshot = {robots: []};
    this.launchAccepted = true;
    this.beforeLaunchResponse = null;
    this.context = vm.createContext({
      window: {addEventListener() {}, location: {hostname: 'localhost'}},
      document: {getElementById: (id) => this.element(id), createElement: () => this.element('option' + this.elements.size)},
      fetch: async (url, options) => {
        const route = new URL(url, 'http://localhost').pathname;
        this.requests.push({route, options});
        if (route === '/api/plan/scaffold' && this.beforeLaunchResponse) this.beforeLaunchResponse();
        const payload = route === '/robots' ? this.snapshot : route === '/api/plan/scaffold' ? {ok: this.launchAccepted} : route === '/plan' ? {nodes: []} : {objects: {}, ok: true, returncode: null};
        return {ok: true, json: async () => payload};
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
    const exportsSource = fs.readFileSync(path.join(__dirname, '../dataset/builder_multi_robot_exports.js'), 'utf8');
    vm.runInContext(source.slice(0, source.indexOf('  // ---------- boot ----------')) + exportsSource, this.context, {filename: path.join(web, 'plan_builder.js')});
    this.api = this.context.window.multiRobotPage;
    this.api.initialize([{name: 'PR2', cls: 'PR2', import: 'from semantic_digital_twin.robots.pr2 import PR2', arms: ['BOTH'], steps: ['park_arms']}]);
    this.element('pb-env').value = 'apartment.urdf';
    this.element('pb-name').value = 'sequential_robots';
    this.element('pb-style').value = 'class';
    this.api.addRobotInstance();
    const [first, second] = this.api.state.instances;
    this.api.state.updateRobot(first.id, {x: 6.5, y: 1, yaw: 0});
    this.api.state.updateRobot(second.id, {x: 1, y: 2.5, yaw: 0});
    this.api.steps = [{type: 'park_arms', params: {arm: 'BOTH'}}];
  }
  element(identifier) {
    if (!this.elements.has(identifier)) this.elements.set(identifier, {
      value: '', textContent: '', innerHTML: '', style: {}, children: [], src: 'index.html?scene',
      addEventListener() {}, replaceChildren() { this.children = []; }, appendChild(child) { this.children.push(child); },
    });
    return this.elements.get(identifier);
  }
  completedSnapshot() {
    return JSON.parse(fs.readFileSync(path.join(__dirname, '../dataset/multi_robot_completed_poses.json'), 'utf8'));
  }
  async launch() {
    await this.api.runPlan();
    await nextTurn();
  }
}

// %% robot has moved before the first capture after launching
async function checkCompletedRobotSurvivesAnotherRobotsRun() {
  const page = new RobotRestartPage();
  const [first, second] = page.api.state.instances;
  await page.launch();
  page.snapshot = page.completedSnapshot();
  const completed = page.snapshot.robots.find((robot) => robot.identifier === second.id);
  await page.api.selectRobotInstance(first.id);
  page.api.steps = [{type: 'park_arms', params: {arm: 'BOTH'}}];
  await page.launch();
  assert.equal(second.x, completed.pose[0]);
  assert.equal(second.y, completed.pose[1]);
  assert.deepEqual(JSON.parse(JSON.stringify(second.joint_positions)), completed.joint_positions);
  const launches = page.requests.filter((request) => request.route === '/api/plan/scaffold');
  const source = JSON.parse(launches[1].options.body).code;
  assert(source.includes('HomogeneousTransformationMatrix.from_xyz_rpy(' + completed.pose[0] + ', ' + completed.pose[1] + ', 0.0, yaw=' + second.yaw + ')'));
  assert(source.includes('active_identifier="' + first.id + '"'));
}

// %% only successfully submitted edits have been consumed
async function checkRejectedLaunchRetainsAuthoredPose() {
  const page = new RobotRestartPage();
  page.launchAccepted = false;
  const second = page.api.state.activeRobot();
  const authored = {x: second.x, y: second.y, yaw: second.yaw};
  await page.launch();
  page.api.state.captureRobots(page.completedSnapshot());
  assert.equal(second.x, authored.x);
  assert.equal(second.y, authored.y);
  assert.equal(second.yaw, authored.yaw);
}

async function checkLaterEditSurvivesEarlierLaunchAcknowledgement() {
  const page = new RobotRestartPage();
  const second = page.api.state.activeRobot();
  page.beforeLaunchResponse = () => page.api.state.updateRobot(second.id, {x: 9});
  await page.launch();
  page.api.state.captureRobots(page.completedSnapshot());
  assert.equal(second.x, 9);
}

(async () => {
  await checkCompletedRobotSurvivesAnotherRobotsRun();
  await checkRejectedLaunchRetainsAuthoredPose();
  await checkLaterEditSurvivesEarlierLaunchAcknowledgement();
  console.log('Sequential runs preserve completed robot poses and joints while protecting unsubmitted edits.');
})().catch((error) => { console.error(error); process.exitCode = 1; });
