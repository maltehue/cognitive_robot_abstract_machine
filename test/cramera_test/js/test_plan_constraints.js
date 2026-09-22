// Unit tests for web/core/plan_constraints.js (node:test): which sentence the Plan
// Builder turns into which constraint, and what the plan it generates actually does
// about it.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  const scope = {};
  new Function('window', fs.readFileSync(path.join(WEB, 'core/plan_constraints.js'), 'utf8'))(scope);
  return scope.PlanConstraints;
}

function transportStep(constraints) {
  return { type: constraints.STEP.TRANSPORT, params: { object: 'milk.stl', arm: 'LEFT' } };
}

function navigateStep() {
  return { type: 'navigate', params: { x: 2.6, y: 1.8, z: 0.0, yaw: 0.0 } };
}

// %% sentence -> constraint
test('a sentence about looking compiles to the pointing goal', function () {
  const constraints = load();
  const compiled = constraints.compile('Robot must look where it operates', transportStep(constraints));
  assert.strictEqual(compiled.goal, constraints.GOAL.POINTING_AT);
});

test('a sentence no rule matches compiles to no goal', function () {
  const constraints = load();
  assert.strictEqual(constraints.compile('be nice about it', transportStep(constraints)).goal, null);
});

// %% what the generated plan does about it
test('looking where it operates is enforced by the transport itself', function () {
  const constraints = load();
  const compiled = constraints.compile('Robot must look where it operates', transportStep(constraints));
  assert.strictEqual(compiled.stepArgument, constraints.ARGUMENT.LOOK_AT_OPERATION_SITE);
});

test('the generated transport is switched to look at what it operates on', function () {
  const constraints = load();
  const compiled = constraints.compile('Robot must look where it operates', transportStep(constraints));
  assert.deepStrictEqual(constraints.stepArguments([compiled]), ['look_at_operation_site=True']);
});

test('the same constraint on two steps switches each of them once', function () {
  const constraints = load();
  const compiled = constraints.compile('Robot must look where it operates', transportStep(constraints));
  assert.deepStrictEqual(constraints.stepArguments([compiled, compiled]), ['look_at_operation_site=True']);
});

test('a step that operates on nothing carries no switch it has no argument for', function () {
  const constraints = load();
  const compiled = constraints.compile('Robot must look where it operates', navigateStep());
  assert.strictEqual(compiled.stepArgument, null);
  assert.deepStrictEqual(constraints.stepArguments([compiled]), []);
});

test('a goal no action enforces is left to the live bridge', function () {
  const constraints = load();
  const compiled = constraints.compile('Milk must always stay upright', transportStep(constraints));
  assert.strictEqual(compiled.goal, constraints.GOAL.VECTORS_ALIGNED);
  assert.strictEqual(compiled.stepArgument, null);
});

// %% which object a sentence is about
test('a sentence naming an object is about that one', function () {
  const constraints = load();
  const compiled = constraints.compile('Keep the bowl above the table', transportStep(constraints));
  assert.strictEqual(compiled.params.tip_link, 'bowl');
});

test('a sentence naming none is about what its step carries', function () {
  const constraints = load();
  const compiled = constraints.compile('it must stay upright', transportStep(constraints));
  assert.strictEqual(compiled.params.tip_link, 'milk');
});
