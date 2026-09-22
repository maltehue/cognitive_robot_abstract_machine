// Unit tests for web/core/base_control.js (node:test).
// The choice decides whether the generated demo writes full_body_controlled at all, and
// with which value -- a wrong mapping either lets the base drive off during a reach or
// pins a setting the robot never asked for.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  global.window = {};
  new Function(fs.readFileSync(path.join(WEB, 'core/base_control.js'), 'utf8'))();
  return global.window.BaseControl;
}

test('standing still is offered first, so it is what a plan gets by default', function () {
  const control = load();

  assert.strictEqual(control.all()[0].name, 'stand_still');
  assert.strictEqual(control.byName(null).name, 'stand_still');
});

test('each choice carries the value it writes', function () {
  const control = load();

  assert.deepStrictEqual(
    control.all().map(function (c) { return [c.name, c.fullBodyControlled]; }),
    [['stand_still', false], ['robot_default', null], ['may_drive', true]],
  );
});

test('only a choice with a value of its own writes the setting', function () {
  const control = load();

  assert.strictEqual(control.pinsTheSetting(control.byName('stand_still')), true);
  assert.strictEqual(control.pinsTheSetting(control.byName('may_drive')), true);
  assert.strictEqual(control.pinsTheSetting(control.byName('robot_default')), false);
});

test('every offered choice carries a label to show', function () {
  load().all().forEach(function (choice) {
    assert.ok(choice.label, choice.name);
  });
});

test('an unknown name falls back to standing still', function () {
  const control = load();

  assert.strictEqual(control.byName('').name, 'stand_still');
  assert.strictEqual(control.byName('drive_through_the_shelf').name, 'stand_still');
});

test('the offered choices are not the module\'s own list', function () {
  const control = load();

  control.all().pop();

  assert.strictEqual(control.all().length, 3);
});
