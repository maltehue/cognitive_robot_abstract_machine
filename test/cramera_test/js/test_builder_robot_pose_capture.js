'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const context = {window: {}};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../../cramera/src/cramera/web/core/builder_state.js'), 'utf8'), context, {filename: path.join(__dirname, '../../../cramera/src/cramera/web/core/builder_state.js')});
const state = new context.window.PlanBuilderState([{name: 'PR2', steps: [], arms: []}]);
const first = state.addRobot('PR2');
const second = state.addRobot('PR2');
state.captureRobots({robots: [
  {identifier: first.id, model: first.model, pose: [3, 4, 0, 0, 0, Math.SQRT1_2, Math.SQRT1_2]},
  {identifier: second.id, model: second.model, pose: [5, 6, 0, 0, 0, 0, 1]},
]});
assert.equal(first.x, 3); assert.equal(first.y, 4);
assert(Math.abs(first.yaw - Math.PI / 2) < 1e-12);
assert.equal(second.x, 5); assert.equal(second.y, 6);
state.updateRobot(first.id, {x: 9, yaw: Math.PI});
state.captureRobots({robots: [{identifier: first.id, model: first.model, pose: [3, 4, 0, 0, 0, 0, 1]}]});
assert.equal(first.x, 9); assert.equal(first.yaw, Math.PI);
state.captureRobots({robots: [{identifier: first.id, model: first.model, pose: [9, 4, 0, 0, 0, 1, 0]}]});
state.captureRobots({robots: [{identifier: first.id, model: first.model, pose: [10, 4, 0, 0, 0, 0, 1]}]});
assert.equal(first.x, 10); assert.equal(first.yaw, 0);
state.captureRobots({robots: [
  {identifier: first.id, model: 'Tracy', pose: [20, 4, 0, 0, 0, 0, 1]},
  {identifier: 'missing', model: 'PR2', pose: [30, 4, 0, 0, 0, 0, 1]},
  {identifier: first.id, model: first.model, pose: [NaN, 4, 0, 0, 0, 0, 1]},
]});
assert.equal(first.x, 10);
state.captureRobots({}); state.captureRobots(null);
console.log('Live captures preserve all robot base poses without overwriting unspawned edits.');
state.captureRobots({robots: [{identifier: first.id, model: first.model, pose: [10, 4, 0, 0, 0, 0, 1], joint_positions: {'robot_1/torso_lift_joint': 0.2}}]});
assert.equal(first.joint_positions['robot_1/torso_lift_joint'], 0.2);
state.captureRobots({robots: [{identifier: first.id, model: first.model, pose: [10, 4, 0, 0, 0, 0, 1], joint_positions: {'robot_1/torso_lift_joint': NaN}}]});
assert.equal(first.joint_positions['robot_1/torso_lift_joint'], 0.2);
state.updateRobot(first.id, {model: first.model});
assert.deepEqual(Object.keys(first.joint_positions), []);
