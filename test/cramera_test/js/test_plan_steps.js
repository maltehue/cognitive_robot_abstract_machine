// Unit tests for web/core/plan_steps.js (node:test).
// Whether a step acts on a placed object decides what the generated demo spawns for it and
// which of its targets have to be resolved, so a step kind missing from either list drops
// silently out of the generated file rather than failing loudly.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  global.window = {};
  new Function(fs.readFileSync(path.join(WEB, 'core/plan_steps.js'), 'utf8'))();
  return global.window.PlanSteps;
}

const PICK = { type: 'pick', params: { object: 'milk.stl', arm: 'LEFT' } };
const PLACE_AT_A_POSE = { type: 'place', params: { object: 'milk.stl', targetMode: 'pose' } };
const PLACE_ON_A_SURFACE = { type: 'place', params: { object: 'milk.stl', targetMode: 'semantic' } };
const TRANSPORT = { type: 'transport', params: { object: 'milk.stl', targetMode: 'semantic' } };
const NAVIGATE = { type: 'navigate', params: { x: 1, y: 2 } };
const PARK_ARMS = { type: 'park_arms', params: { arm: 'BOTH' } };

test('transport, pick and place act on an object', function () {
  const steps = load();

  assert.deepStrictEqual(
    [TRANSPORT, PICK, PLACE_AT_A_POSE].map(steps.actsOnAnObject),
    [true, true, true],
  );
});

test('steps that only move the robot act on no object', function () {
  const steps = load();

  assert.deepStrictEqual(
    [NAVIGATE, PARK_ARMS].map(steps.actsOnAnObject),
    [false, false],
  );
});

test('transport and place put an object down, a pick does not', function () {
  const steps = load();

  assert.strictEqual(steps.putsAnObjectDown(TRANSPORT), true);
  assert.strictEqual(steps.putsAnObjectDown(PLACE_AT_A_POSE), true);
  assert.strictEqual(steps.putsAnObjectDown(PICK), false);
});

test('a semantic target is only reported for a step that has one', function () {
  const steps = load();

  assert.strictEqual(steps.putsAnObjectDownAtASemanticTarget(PLACE_ON_A_SURFACE), true);
  assert.strictEqual(steps.putsAnObjectDownAtASemanticTarget(PLACE_AT_A_POSE), false);
  assert.strictEqual(steps.putsAnObjectDownAtASemanticTarget(PICK), false);
  assert.strictEqual(steps.putsAnObjectDownAtASemanticTarget(NAVIGATE), false);
});

test('the kinds acting on an object are listed, and not the module\'s own list', function () {
  const steps = load();

  assert.deepStrictEqual(steps.actingOnAnObject(), ['transport', 'pick', 'place']);

  steps.actingOnAnObject().pop();

  assert.strictEqual(steps.actingOnAnObject().length, 3);
});

test('a missing step or missing params is answered, not thrown at', function () {
  const steps = load();

  assert.strictEqual(steps.actsOnAnObject(undefined), false);
  assert.strictEqual(steps.putsAnObjectDown(null), false);
  assert.strictEqual(steps.putsAnObjectDownAtASemanticTarget({ type: 'place' }), false);
});
