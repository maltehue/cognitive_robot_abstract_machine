// Unit tests for core/camera-follow.js (node:test): whether the scene camera keeps the
// moving robot in view, and that the choice survives a reload.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function loadCameraFollow() {
  const scope = {};
  new Function('window', fs.readFileSync(path.join(WEB, 'core/camera-follow.js'), 'utf8'))(scope);
  return scope.CameraFollow;
}

function makeStorage(initial) {
  const items = Object.assign({}, initial);
  return {
    items: items,
    getItem(key) { return key in items ? items[key] : null; },
    setItem(key, value) { items[key] = String(value); },
  };
}

// %% the default
test('a viewer that has never touched the switch follows the robot', function () {
  const CameraFollow = loadCameraFollow();

  assert.strictEqual(CameraFollow.on(makeStorage()), true);
});

// %% remembering the choice
test('switching the follow off survives a reload', function () {
  const CameraFollow = loadCameraFollow();
  const storage = makeStorage();

  const stored = CameraFollow.set(storage, false);

  assert.strictEqual(stored, false);
  assert.strictEqual(CameraFollow.on(storage), false);
});

test('switching the follow back on survives a reload', function () {
  const CameraFollow = loadCameraFollow();
  const storage = makeStorage();
  CameraFollow.set(storage, false);

  const stored = CameraFollow.set(storage, true);

  assert.strictEqual(stored, true);
  assert.strictEqual(CameraFollow.on(storage), true);
});

test('the choice is kept under the module\'s own key', function () {
  const CameraFollow = loadCameraFollow();
  const storage = makeStorage();

  CameraFollow.set(storage, false);

  assert.deepStrictEqual(Object.keys(storage.items), [CameraFollow.KEY]);
});

// %% camera ownership during object placement
class OrbitMovement {
  constructor(enabled) {
    this.enabled = enabled;
    this.updates = 0;
    this.target = { value: 0, lerp(point, amount) { this.value += (point.value - this.value) * amount; } };
  }

  update() {
    this.updates += 1;
    return true;
  }
}

test('dragging freezes robot following and residual orbit movement', function () {
  const controls = new OrbitMovement(false);
  const camera = new (loadCameraFollow().Controller)(controls);

  camera.follow({ value: 10 }, 0.5);
  assert.strictEqual(camera.update(), false);
  assert.strictEqual(controls.target.value, 0);
  assert.strictEqual(controls.updates, 0);
});

test('releasing a drag restores normal camera updates', function () {
  const controls = new OrbitMovement(false);
  const camera = new (loadCameraFollow().Controller)(controls);
  camera.follow({ value: 10 }, 0.5);
  controls.enabled = true;

  camera.follow({ value: 10 }, 0.5);
  assert.strictEqual(camera.update(), true);
  assert.strictEqual(controls.target.value, 5);
  assert.strictEqual(controls.updates, 1);
});
