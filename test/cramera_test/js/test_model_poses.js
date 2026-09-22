'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const context = vm.createContext({window: {}});
const modelSource = path.join(__dirname, '../../../cramera/src/cramera/web/core/model_poses.js');
vm.runInContext(fs.readFileSync(modelSource, 'utf8'), context, {filename: modelSource});
const ModelPoses = context.window.ModelPoses;

test('primary robot is chosen by instance identity rather than load order', () => {
  const first = {robot: true, identifier: 'first', name: 'robot_first'};
  const second = {robot: true, identifier: 'second', name: 'robot_second'};
  assert.equal(ModelPoses.primary([second, first], {identifier: 'first'}), first);
  assert.equal(ModelPoses.primary([first, second], {identifier: 'second'}), second);
  assert.equal(ModelPoses.primary([first], {identifier: 'second'}), null);
  assert.equal(ModelPoses.primary([first], {name: 'robot_first'}), first);
  assert.equal(ModelPoses.primary([first], null), null);
});

test('live and replay robot roots use separate instance tracks', () => {
  const first = {prefix: 'first', obj: {name: 'first'}};
  const second = {prefix: 'second', obj: {name: 'second'}};
  const environment = {prefix: '', obj: {name: 'environment'}};
  const start = {first: [1, 0, 0, 0, 0, 0, 1], second: [3, 0, 0, 0, 0, 0, 1]};
  const end = {first: start.first, second: [4, 0, 0, 0, 0, 0, 1]};
  const applied = [];
  ModelPoses.apply([environment, second, first], start, end, 0.5, (...args) => applied.push(args));
  assert.deepEqual(applied, [[second.obj, start.second, end.second, 0.5], [first.obj, start.first, end.first, 0.5]]);
  const live = [];
  ModelPoses.apply([first, second], end, null, 0, (...args) => live.push(args));
  assert.deepEqual(live, [[first.obj, end.first, end.first, 0], [second.obj, end.second, end.second, 0]]);
  ModelPoses.apply([first], null, null, 0, () => assert.fail('Absent track must not move model'));
});

test('actual viewer applies recorded tracks after the legacy primary base', () => {
  const panel = fs.readFileSync(path.join(__dirname, '../../../cramera/src/cramera/web/panels/robot_scene/panel.js'), 'utf8');
  const models = [{prefix: 'first', obj: {position: {x: 0, y: 0}}}, {prefix: 'second', obj: {position: {x: 0, y: 0}}}];
  const first = [0, 0, 0, 0, 0, 0, 1], second = [3, 0, 0, 0, 0, 0, 1];
  const poses = [];
  const sandbox = vm.createContext({
    traj: {frames: [{}, {}], base: [first, first], modelBases: [{first, second}, {first, second}]},
    models, robotModel: models[1], JointRouting: {}, ModelPoses,
    syncJointControls() {}, setPose: (...args) => poses.push(args), baseOffsetAt: () => ({x: 0, y: 0}),
    objectMeshes: {}, updateArrows() {}, updateFrameAxes() {}, needsRender: false,
  });
  const begin = panel.indexOf('  function applyFrame(f)');
  const end = panel.indexOf('    if (traj.objects)', begin);
  vm.runInContext(panel.slice(begin, end) + '\n}', sandbox);
  sandbox.applyFrame(0.5);
  assert.deepEqual(poses.slice(-2), [[models[0].obj, first, first, 0.5], [models[1].obj, second, second, 0.5]]);
});

test('query highlights distinguish matching parts on repeated robot models', () => {
  const first = {identifier: 'first', prefix: 'first', robot: true};
  const second = {identifier: 'second', prefix: 'second', robot: true};
  const scene = {robots: [
    {identifier: 'first', prefix: 'first', name: 'pr2', label: 'First', parts: {Arm: ['arm_link']}},
    {identifier: 'second', prefix: 'second', name: 'pr2', label: 'Second', parts: {Arm: ['arm_link']}},
  ]};
  assert.equal(ModelPoses.partFor(scene, second, 'second/arm_link'), 'second/Arm');
  assert.equal(ModelPoses.highlighted(scene, first, 'first/arm_link', {'second/Arm': 1}), false);
  assert.equal(ModelPoses.highlighted(scene, second, 'second/arm_link', {'second/Arm': 1}), true);
  assert.equal(ModelPoses.highlighted(scene, second, 'second/base_link', {second: 1}), true);
  assert.equal(ModelPoses.highlighted(scene, second, 'second/base_link', {'second/Second': 1}), true);
  assert.equal(ModelPoses.highlighted(scene, first, 'first/base_link', {'second/Second': 1}), false);
  assert.equal(ModelPoses.highlighted(scene, second, 'second/arm_link', {'urdf:second/arm_link': 1}), true);
  assert.equal(ModelPoses.highlighted({}, second, 'second/arm_link', {}), false);
  const legacy = {robot: {name: 'pr2', prefix: 'pr2', parts: {Arm: ['arm_link']}}};
  assert.equal(ModelPoses.partFor(legacy, {name: 'pr2', prefix: 'pr2'}, 'arm_link'), 'Arm');
});

test('a single named instance retains its legacy robot query identity', () => {
  const robot = {identifier: 'robot_1', prefix: 'robot_1', name: 'pr2', parts: {}};
  const scene = {robot, robots: [robot]};
  assert.equal(ModelPoses.highlighted(scene, {identifier: 'robot_1'}, 'robot_1/base', {pr2: 1}), true);
});
