// Unit tests for web/core/execution_environment.js (node:test).
// The Plan Builder offers these as its "collision avoidance" choice, and the generated
// demo either enters the environment by name (flat script) or hands the flag to a
// RobotDemonstration, so a lookup that loses either half produces a demo that plans
// differently from what was picked.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  global.window = {};
  new Function(fs.readFileSync(path.join(WEB, 'core/execution_environment.js'), 'utf8'))();
  return global.window.ExecutionEnvironments;
}

test('every offered environment enables collision avoidance', function () {
  const offered = load().all();

  assert.deepStrictEqual(
    offered.map(function (e) { return [e.name, e.collisionAvoidance]; }),
    [['simulated_robot_advanced', true]],
  );
});

test('every offered environment carries a label to show', function () {
  load().all().forEach(function (environment) {
    assert.ok(environment.label, environment.name);
  });
});

test('byName finds the environment of that name', function () {
  const environments = load();

  assert.strictEqual(environments.byName('simulated_robot_advanced').collisionAvoidance, true);
  assert.strictEqual(environments.byName('simulated_robot').collisionAvoidance, true);
});

test('an unknown name falls back to the first environment offered', function () {
  const environments = load();

  assert.deepStrictEqual(environments.byName(''), environments.all()[0]);
  assert.deepStrictEqual(environments.byName(null), environments.all()[0]);
  assert.strictEqual(environments.byName('real_robot').collisionAvoidance, true);
});

test('the offered environments are not the module\'s own list', function () {
  // all() handing out its array would let one caller's edit change what everyone sees
  const environments = load();

  environments.all().pop();

  assert.strictEqual(environments.all().length, 1);
});
