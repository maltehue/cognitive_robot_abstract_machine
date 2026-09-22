'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const WEB = path.join(__dirname, '../../../cramera/src/cramera/web');
const PANEL_PATH = path.join(WEB, 'panels/robot_scene/panel.js');
const SOURCE = fs.readFileSync(PANEL_PATH, 'utf8');

// %% panel DOM and renderer boundary
class PanelElement {
  constructor() {
    this.handlers = new Map();
    this.classList = {add() {}};
  }
  addEventListener(event, handler) { this.handlers.set(event, handler); }
  removeEventListener(event, handler) {
    if (this.handlers.get(event) === handler) this.handlers.delete(event);
  }
  click() {
    const handler = this.handlers.get('click');
    if (handler) handler();
  }
}

class PanelRoot extends PanelElement {
  set innerHTML(markup) {
    this.markup = markup;
    this.elements = new Map(Array.from(markup.matchAll(/id="([^"]+)"/g),
      match => ['#' + match[1], new PanelElement()]));
  }
  get innerHTML() { return this.markup; }
  querySelector(selector) { return this.elements.get(selector) || null; }
}

class UnavailableScene {
  constructor(error) {
    this.error = error;
    this.attempts = 0;
    this.warnings = [];
    this.root = new PanelRoot();
    this.builder = Object.freeze({plan: ['park_arms', 'transport'], object: [2.37, 2, 1.05]});
    this.window = {location: {href: 'http://localhost:8711/index.html?scene=example&offline=1', search: '?scene=example&offline=1'}};
    Object.defineProperty(this.window, 'parent', {get() { throw new Error('The scene must not access Builder state.'); }});
    new Function('window', fs.readFileSync(path.join(WEB, 'core/scene.js'), 'utf8'))(this.window);
    const instance = this;
    const three = {
      Scene: class {},
      PerspectiveCamera: class { constructor() { this.position = {set() {}}; } },
      WebGLRenderer: class {
        constructor() { instance.attempts++; throw instance.error; }
      },
    };
    vm.runInNewContext(SOURCE, {
      Panels: {define(name, factory) { instance.factory = factory; }},
      SceneContext: this.window.SceneContext,
      THREE: three,
      window: this.window,
      console: {log() {}, warn(...args) { instance.warnings.push(args); }},
      Error,
    }, {filename: PANEL_PATH});
  }
  mount() { return this.factory(this.root, {}); }
}

// %% unavailable WebGL and recovery controls
for (const message of ['Error creating WebGL context.', 'Error creating WebGL context with your selected attributes.']) {
  test('WebGL failure provides a usable fallback: ' + message, () => {
    const scene = new UnavailableScene(new Error(message));
    scene.mount();
    assert.match(scene.root.innerHTML, /role="alert"/);
    assert.match(scene.root.innerHTML, /WebGL/);
    assert.match(scene.root.innerHTML, /browser/i);
    assert.match(scene.root.innerHTML, /Builder/);
    assert.ok(scene.root.querySelector('#scene-retry'));
    const link = scene.root.querySelector('#scene-open-browser');
    assert.equal(link.href, scene.window.location.href);
    assert.match(scene.root.innerHTML, /target="_blank"/);
    assert.match(scene.root.innerHTML, /rel="noopener noreferrer"/);
    assert.equal(scene.window.RobotView, undefined);
  });
}

test('retry initializes only the scene and keeps Builder inputs intact', () => {
  const scene = new UnavailableScene(new Error('Error creating WebGL context.'));
  scene.mount();
  const builder = scene.builder;
  scene.root.querySelector('#scene-retry').click();
  assert.equal(scene.attempts, 2);
  assert.equal(scene.builder, builder);
  assert.deepEqual(scene.builder.plan, ['park_arms', 'transport']);
  assert.deepEqual(scene.builder.object, [2.37, 2, 1.05]);
  assert.ok(scene.root.querySelector('#scene-retry'));
});

test('unmount removes recovery handlers after repeated failure', () => {
  const scene = new UnavailableScene(new Error('Error creating WebGL context.'));
  const mounted = scene.mount();
  const firstRetry = scene.root.querySelector('#scene-retry');
  firstRetry.click();
  const nextRetry = scene.root.querySelector('#scene-retry');
  mounted.destroy();
  firstRetry.click();
  nextRetry.click();
  assert.equal(scene.attempts, 2);
});

test('unrelated renderer failures still reach the panel registry', () => {
  const failure = new TypeError('Renderer initialization programming error');
  const scene = new UnavailableScene(failure);
  assert.throws(() => scene.mount(), error => error === failure);
  assert.equal(scene.root.querySelector('#scene-retry'), null);
});
