'use strict';
// A demo setup travels between the Plan Builder and the server in the form
// cramera.demo_setup reads: the builder writes its robots into it and reads them back.
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const test = require('node:test');
const vm = require('vm');

const WEB = path.join(__dirname, '../../../cramera/src/cramera/web/core');
const context = {window: {}};
vm.runInNewContext(fs.readFileSync(path.join(WEB, 'builder_state.js'), 'utf8'), context);
vm.runInNewContext(fs.readFileSync(path.join(WEB, 'demo_setup.js'), 'utf8'), context);
const State = context.window.PlanBuilderState;
const Form = context.window.DemoSetupForm;

const humanoid = {name: 'WalkerS2', steps: ['park_arms', 'look_at'], arms: ['BOTH']};
const arm = {name: 'ContinuumRobot', steps: ['look_at'], arms: ['BOTH']};

function lookAt(x) { return {id: 's' + x, type: 'look_at', params: {x: x, y: 0, z: 1}}; }

function twoRobots() {
  const state = new State([humanoid, arm]);
  const walker = state.addRobot('WalkerS2', {x: 0, y: 2, yaw: Math.PI});
  state.updateRobot(walker.id, {jointStateTopic: ' /walker_s2/joint_states ', localizationTopic: '/walker_s2/odom'});
  const continuum = state.addRobot('ContinuumRobot', {x: 0, y: -0.6, yaw: 0});
  state.updateRobot(continuum.id, {repeatsPlan: true});
  return {state: state, walker: walker, continuum: continuum};
}

test('the builder writes each robot where it stands and what moves it', function () {
  const {state, walker, continuum} = twoRobots();
  const payload = Form.toPayload(state, [lookAt(1), lookAt(2)], '/lab/world.usda');

  assert.deepStrictEqual(JSON.parse(JSON.stringify(payload.environment)), {path: '/lab/world.usda'});
  const [written, looking] = payload.robots;
  assert.strictEqual(written.identifier, walker.id);
  assert.strictEqual(written.model, 'WalkerS2');
  assert.strictEqual(written.yaw, Math.PI);
  assert.strictEqual(written.jointStateTopic, '/walker_s2/joint_states');
  assert.strictEqual(written.localizationTopic, '/walker_s2/odom');
  assert.strictEqual(looking.localizationTopic, '');
  assert.strictEqual(written.repeatsPlan, false);
  assert.strictEqual(looking.identifier, continuum.id);
  assert.strictEqual(looking.repeatsPlan, true);
  // the active robot's plan is the one being edited, not yet stored on the instance
  assert.deepStrictEqual(JSON.parse(JSON.stringify(looking.steps)),
    [{type: 'look_at', params: {x: 1, y: 0, z: 1}}, {type: 'look_at', params: {x: 2, y: 0, z: 1}}]);
});

test('a setup read back gives the builder the robots it was written with', function () {
  const {state} = twoRobots();
  state.environmentJointPositions = {'lab/wall_T_lab/door_0': 1.2};
  state.environmentGeometry = 'collision';
  const payload = Form.toPayload(state, [lookAt(1)], '/lab/world.usda');
  const opened = new State([humanoid, arm]);
  let made = 0;
  const makeStep = function (type, params) { made += 1; return {id: 'n' + made, type: type, params: params}; };

  const active = Form.applyTo(opened, payload, makeStep);

  assert.deepStrictEqual(JSON.parse(JSON.stringify(opened.instances.map((robot) => robot.model))), ['WalkerS2', 'ContinuumRobot']);
  assert.strictEqual(active, opened.instances[0]);
  assert.strictEqual(opened.instances[0].joint_state_topic, '/walker_s2/joint_states');
  assert.strictEqual(opened.instances[0].localization_topic, '/walker_s2/odom');
  assert.strictEqual(opened.instances[1].repeats_plan, true);
  assert.strictEqual(opened.instances[1].steps[0].id, 'n1');
  assert.strictEqual(opened.environmentJointPositions['lab/wall_T_lab/door_0'], 1.2);
  assert.strictEqual(opened.environmentGeometry, 'collision');
  assert.deepStrictEqual(JSON.parse(JSON.stringify(Form.toPayload(opened, opened.instances[0].steps, '/lab/world.usda'))),
    JSON.parse(JSON.stringify(payload)));
  // a robot added afterwards does not take an identifier the setup already uses
  const added = opened.addRobot('WalkerS2');
  assert(!payload.robots.some((robot) => robot.identifier === added.id));
});

test('a setup naming a robot model that is not installed is refused', function () {
  const opened = new State([humanoid]);
  const payload = {environment: null, robots: [{identifier: 'r', label: 'r', model: 'Unheard', x: 0, y: 0, yaw: 0, steps: []}]};

  // thrown from the module's own context, whose RangeError is not this one's
  assert.throws(() => Form.applyTo(opened, payload, function () { return {}; }), (error) => error.name === 'RangeError');
});

// %% a map environment, which a setup file cannot name
test('a map environment is written as a map, for the server to refuse by name', function () {
  const {state} = twoRobots();

  const payload = Form.toPayload(state, [lookAt(1)], {kind: 'map', cls: 'ApartmentEnvironment', name: 'real-lab apartment'});

  assert.deepStrictEqual(JSON.parse(JSON.stringify(payload.environment)), {kind: 'map', cls: 'ApartmentEnvironment'});
});

test('a file environment given as the offered environment is written by its path', function () {
  const {state} = twoRobots();

  const payload = Form.toPayload(state, [lookAt(1)], {kind: 'file', path: '/lab/world.usda', name: 'lab/world.usda'});

  assert.deepStrictEqual(JSON.parse(JSON.stringify(payload.environment)), {path: '/lab/world.usda'});
});

test('a builder that was never told how to draw the environment draws it as it looks', function () {
  const {state} = twoRobots();

  assert.strictEqual(Form.toPayload(state, [], '/lab/world.usda').environmentGeometry, 'visual');
});

// %% how a setup placed its USD scene
function openedWithPlacement() {
  const {state} = twoRobots();
  const payload = Form.toPayload(state, [], '/lab/world.usda');
  payload.environment = {path: '/lab/world.usda', rootPlacement: 'stage_origin'};
  const opened = new State([humanoid, arm]);
  Form.applyTo(opened, payload, function (type, params) { return {id: 'n', type: type, params: params}; });
  return {opened: opened, placement: payload.environment.rootPlacement};
}

test('a setup opened and saved again keeps how it placed its scene', function () {
  const {opened, placement} = openedWithPlacement();

  const saved = Form.toPayload(opened, [], '/lab/world.usda');

  assert.strictEqual(saved.environment.rootPlacement, placement);
});

test('a scene placement is not carried over to another environment', function () {
  const {opened} = openedWithPlacement();

  const saved = Form.toPayload(opened, [], '/lab/other.usda');

  assert.deepStrictEqual(JSON.parse(JSON.stringify(saved.environment)), {path: '/lab/other.usda'});
});

// %% the boxes lying about
test('a box with a size is written as the setup\'s, a mesh for a generated demo is not', function () {
  const {state} = twoRobots();
  const objects = [
    {id: 'o1', mesh: 'block.stl', name: 'block.stl', x: 13.27, y: -2.02, z: 1.02, yaw: 0.4, size: [0.06, 0.06, 0.2]},
    {id: 'o2', mesh: 'milk.stl', name: 'milk.stl', x: 1, y: 2, z: 0.9, yaw: 0},
  ];

  const payload = Form.toPayload(state, [], '/lab/world.usda', objects);

  assert.deepStrictEqual(JSON.parse(JSON.stringify(payload.objects)),
    [{name: 'block.stl', x: 13.27, y: -2.02, z: 1.02, yaw: 0.4, size: [0.06, 0.06, 0.2]}]);
});

test('a setup read back lists its boxes for the page to show', function () {
  const {state} = twoRobots();
  const payload = Form.toPayload(state, [], '/lab/world.usda',
    [{id: 'o1', mesh: 'block.stl', name: 'block.stl', x: 13.27, y: -2.02, z: 1.02, yaw: 0.4, size: [0.06, 0.06, 0.2]}]);
  const opened = new State([humanoid, arm]);

  Form.applyTo(opened, payload, function (type, params) { return {id: 'n', type: type, params: params}; });

  assert.deepStrictEqual(JSON.parse(JSON.stringify(opened.objects)),
    [{name: 'block.stl', x: 13.27, y: -2.02, z: 1.02, yaw: 0.4, size: [0.06, 0.06, 0.2]}]);
  assert.deepStrictEqual(JSON.parse(JSON.stringify(Form.toPayload(state, [], '/lab/world.usda').objects)), []);
  const bare = new State([humanoid, arm]);
  Form.applyTo(bare, {environment: null, robots: payload.robots}, function (type, params) { return {id: 'n', type: type, params: params}; });
  assert.deepStrictEqual(JSON.parse(JSON.stringify(bare.objects)), []);
});
