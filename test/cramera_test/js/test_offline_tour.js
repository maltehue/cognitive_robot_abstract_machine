'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const web = path.join(__dirname, '../../../cramera/src/cramera/web');

function load() {
  const scope = {};
  new Function('window', fs.readFileSync(path.join(web, 'tour.js'), 'utf8'))(scope);
  return scope.OfflineTour;
}

class ViewerPlayback {
  constructor() { this.loads = []; this.plays = []; this.pauses = 0; }
  load(chapter, events) { this.loads.push({chapter, events}); }
  play(speed) { this.plays.push(speed); }
  pause() { this.pauses += 1; }
}

function setup() {
  const api = load();
  const storyboard = new api.Storyboard({title: 'Tour', description: 'Lokal', chapters: [
    {title: 'Fahrt', narration: 'Der Roboter fährt.', scene: 'navigation', view: 'plan', query: null, playback: true, speed: 0.5, durationSeconds: 4},
    {title: 'Wissen', narration: 'Eine gespeicherte Antwort.', scene: 'knowledge', view: 'knowledge', query: 'show all handles', playback: false, speed: 1, durationSeconds: 2},
  ]});
  const viewer = new ViewerPlayback();
  let now = 0;
  const controller = new api.Controller(storyboard, viewer, () => now, () => {});
  return {api, controller, viewer, time(milliseconds) { now = milliseconds; controller.tick(); }};
}

test('loading time does not count and playback waits for the real final frame', () => {
  const state = setup();
  state.controller.start();
  state.time(10000);
  assert.equal(state.controller.index, 0);
  assert.equal(state.viewer.plays.length, 0);
  state.viewer.loads[0].events.ready();
  assert.deepEqual(state.viewer.plays, [0.5]);
  state.time(20000);
  assert.equal(state.controller.index, 0);
  state.viewer.loads[0].events.ended();
  state.time(20001);
  assert.equal(state.controller.index, 1);
});

test('an early playback end still respects the minimum narration duration', () => {
  const state = setup();
  state.viewer.loads[0].events.ready();
  state.controller.start();
  state.viewer.loads[0].events.ended();
  state.time(3999);
  assert.equal(state.controller.index, 0);
  state.time(4000);
  assert.equal(state.controller.index, 1);
});

test('pause freezes narration time and resumes playback without reloading', () => {
  const state = setup();
  state.viewer.loads[0].events.ready();
  state.controller.start();
  state.time(1000);
  state.controller.pause();
  state.time(9000);
  state.viewer.loads[0].events.ended();
  state.controller.start();
  state.time(11999);
  assert.equal(state.controller.index, 0);
  state.time(12000);
  assert.equal(state.controller.index, 1);
  assert.equal(state.viewer.loads.length, 2);
});

test('manual selection discards callbacks from the previous chapter', () => {
  const state = setup();
  const stale = state.viewer.loads[0].events;
  state.controller.select(1);
  stale.ready(); stale.ended(); stale.failed('old request');
  assert.equal(state.controller.index, 1);
  assert.equal(state.controller.ready, false);
  assert.equal(state.controller.error, '');
  state.viewer.loads[1].events.ready();
  state.controller.start();
  state.time(2000);
  assert.equal(state.controller.completed, true);
  assert.equal(state.controller.running, false);
  state.controller.previous();
  assert.equal(state.controller.index, 0);
  assert.equal(state.controller.completed, false);
});

test('a viewer failure stops progression and remains visible', () => {
  const state = setup();
  state.controller.start();
  state.viewer.loads[0].events.failed('Keine Frames');
  state.time(60000);
  assert.equal(state.controller.running, false);
  assert.equal(state.controller.error, 'Keine Frames');
  assert.equal(state.controller.completed, false);
});

test('manifest rejects unsafe scene paths and unsupported chapter settings', () => {
  const api = load();
  const chapter = {title: 'A', narration: 'B', scene: '../remote', view: 'plan', query: null, playback: true, speed: 1, durationSeconds: 2};
  assert.throws(() => new api.Storyboard({title: 'Tour', chapters: [chapter]}), /Szene/);
  assert.throws(() => new api.Storyboard({title: 'Tour', chapters: []}), /Kapitel/);
  assert.throws(() => new api.Storyboard({title: 'Tour', chapters: [{...chapter, scene: 'local', speed: -1}]}), /Geschwindigkeit/);
});

// %% native iframe bridge
function frameFixture() {
  const handlers = {};
  const events = [];
  const bus = {
    on(name, callback) { (handlers[name] ||= []).push(callback); },
    off(name, callback) { handlers[name] = (handlers[name] || []).filter(item => item !== callback); },
    emit(name, payload) { events.push({name, payload}); (handlers[name] || []).slice().forEach(callback => callback(payload)); },
  };
  let ready;
  const view = {count: 4, seeks: [], onReady(callback) { ready = callback; }, frameCount() { return this.count; }, seek(frame) { this.seeks.push(frame); }, stopTrajectory() {}, setPlaybackSpeed() {}, playTrajectory() { return true; }};
  const knowledge = {ready: true, classList: {contains() { return knowledge.ready; }}};
  const frame = {src: '', contentWindow: {Bus: bus, RobotView: view, document: {documentElement: {dataset: {}}, getElementById() { return knowledge; }}}, addEventListener(name, callback) { this[name] = callback; }, removeEventListener(name) { delete this[name]; }};
  return {frame, view, bus, events, knowledge, ready() { ready(); }};
}

test('iframe readiness waits for RobotView and a nonempty trajectory, using native events', () => {
  const api = load();
  const fixture = frameFixture();
  const adapter = new api.Viewer(fixture.frame, () => 1, () => {});
  let ready = 0, ended = 0;
  const chapter = new api.Chapter({title: 'A', narration: 'B', scene: 'saved', view: 'statechart', query: 'show all handles', playback: true, speed: 1, durationSeconds: 0});
  adapter.load(chapter, {ready() { ready += 1; }, ended() { ended += 1; }, failed(error) { assert.fail(error); }});
  assert.equal(fixture.frame.src, 'index.html?scene=saved&offline=1&tour=1');
  fixture.frame.load();
  assert.equal(ready, 0);
  fixture.ready();
  assert.equal(ready, 1);
  assert.deepEqual(fixture.events.find(event => event.name === 'query:ask').payload, {text: chapter.query});
  assert.deepEqual(fixture.events.find(event => event.name === 'graph:view').payload, {name: 'chart'});
  fixture.bus.emit('scene:step', {step: 'navigate'});
  assert.equal(ended, 0);
  fixture.bus.emit('scene:step', {step: '__done__'});
  assert.equal(ended, 1);
});

test('empty recordings fail instead of announcing readiness', () => {
  const api = load();
  const fixture = frameFixture();
  fixture.view.count = 0;
  const adapter = new api.Viewer(fixture.frame, () => 1, () => {});
  let failure = '';
  adapter.load(new api.Chapter({title: 'A', narration: 'B', scene: 'saved', view: 'plan', query: null, playback: false, speed: 1, durationSeconds: 0}), {ready() { assert.fail('empty recording reported ready'); }, ended() {}, failed(error) { failure = error; }});
  fixture.frame.load(); fixture.ready();
  assert.match(failure, /Frames/);
});

test('tour iframe retains knowledge panels and offline mode remains explicit', () => {
  const scope = {location: {search: '?scene=saved&offline=1&tour=1'}};
  new Function('window', fs.readFileSync(path.join(web, 'core/scene.js'), 'utf8'))(scope);
  assert.equal(scope.SceneContext.offline(), true);
  assert.equal(scope.SceneContext.sceneOnly(true), false);
  scope.location.search = '?scene=saved';
  assert.equal(scope.SceneContext.offline(), false);
  assert.equal(scope.SceneContext.sceneOnly(true), true);
});

test('compact presentation layout is confined to explicit tour viewers', () => {
  for (const presentation of [true, false]) {
    const classes = new Set();
    const scope = {location: {search: presentation ? '?scene=saved&offline=1&tour=1' : '?scene=saved'}, self: {}, top: {}};
    new Function('window', fs.readFileSync(path.join(web, 'core/scene.js'), 'utf8'))(scope);
    const document = {documentElement: {classList: {add(name) { classes.add(name); }}}};
    new Function('window', 'document', 'SceneContext', fs.readFileSync(path.join(web, 'config.js'), 'utf8'))(scope, document, scope.SceneContext);
    assert.equal(classes.has(scope.SceneContext.TOUR_CLASS), presentation);
    if (presentation) assert.deepEqual(scope.CRAMERA_CONFIG.layout.right, ['eql', 'graph']);
  }
});

test('readiness waits for the recorded knowledge source as well as the rendered frames', () => {
  const api = load();
  const fixture = frameFixture();
  fixture.knowledge.ready = false;
  const adapter = new api.Viewer(fixture.frame, () => 1, () => {});
  let ready = 0;
  adapter.load(new api.Chapter({title: 'A', narration: 'B', scene: 'saved', view: 'knowledge', query: null, playback: false, speed: 1, durationSeconds: 0}), {ready() { ready += 1; }, ended() {}, failed(error) { assert.fail(error); }});
  fixture.frame.load(); fixture.ready();
  assert.equal(ready, 0);
  fixture.bus.emit('knowledge:ready', {payload: {ok: true}});
  assert.equal(ready, 1);
});

test('a subsequent explanation of the same scene keeps the final recorded pose', () => {
  const api = load();
  const fixture = frameFixture();
  const adapter = new api.Viewer(fixture.frame, () => 1, () => {});
  const entry = {title: 'A', narration: 'B', scene: 'saved', view: 'plan', query: null, playback: true, speed: 1, durationSeconds: 0};
  const events = {ready() {}, ended() {}, failed(error) { assert.fail(error); }};
  adapter.load(new api.Chapter(entry), events);
  fixture.frame.load(); fixture.ready();
  fixture.bus.emit('scene:step', {step: '__done__'});
  const seeks = fixture.view.seeks.length;
  adapter.load(new api.Chapter({...entry, playback: false}), events);
  fixture.ready();
  assert.equal(fixture.view.seeks.length, seeks);
  assert.equal(fixture.frame.load, undefined, 'the already loaded iframe should not await another navigation');
});

test('chapter layout gives queries and plans their respective presentation space', () => {
  const api = load();
  const fixture = frameFixture();
  const adapter = new api.Viewer(fixture.frame, () => 1, () => {});
  for (const view of ['knowledge', 'plan']) {
    adapter.load(new api.Chapter({title: 'A', narration: 'B', scene: view, view, query: null, playback: false, speed: 1, durationSeconds: 1}), {ready() {}, ended() {}, failed(error) { assert.fail(error); }});
    fixture.frame.load(); fixture.ready();
    assert.equal(fixture.frame.contentWindow.document.documentElement.dataset.tourView, view);
  }
});

// %% recorded query routing
class QueryElement {
  constructor() {
    this.listeners = {};
    this.children = [];
    this.value = '';
    this.innerHTML = '';
    this.classList = {add() {}, remove() {}, toggle() {}};
  }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  appendChild(child) { this.children.push(child); }
  querySelectorAll() { return []; }
  scrollIntoView() {}
  click() { this.listeners.click(); }
}

test('offline questions use query:ask and local EQL while microphone capture stays disabled', async () => {
  let mount, speechStarts = 0, releaseKnowledge;
  const requests = [];
  const elements = new Map();
  const root = {innerHTML: '', querySelector(selector) {
    if (!elements.has(selector)) elements.set(selector, new QueryElement());
    return elements.get(selector);
  }};
  const scope = {
    location: {search: '?scene=saved&offline=1', pathname: '/'},
    localStorage: {getItem() { return null; }, setItem() {}},
    SpeechRecognition: class {start() { speechStarts += 1; }},
    document: {createElement() { return new QueryElement(); }},
    Panels: {define(name, factory) { mount = factory; }},
    EqlSuggestions: {of() { return {forget() {}, handledKey() { return false; }}; }},
    Replay: {popupUrl() { return ''; }},
    fetch: async (url, options) => {
      requests.push({url, options});
      if (url.startsWith('/api/knowledge')) return new Promise(resolve => { releaseKnowledge = () => resolve({ok: true, json: async () => ({ok: true, presets: [], details: {}, status: 'recorded'})}); });
      const payload = url.startsWith('/api/question') ? {ok: true, matched: true, preset: {code: 'an(entity(handle))', text: 'show all handles', scope: 'recording'}} : {ok: true, count: 0, entries: []};
      return {ok: true, json: async () => payload};
    },
  };
  for (const name of ['scene', 'query_source', 'question_display', 'preset_groups', 'answer_table', 'response', 'voice', 'folding']) {
    new Function('window', fs.readFileSync(path.join(web, 'core/' + name + '.js'), 'utf8'))(scope);
  }
  scope.window = scope;
  new Function(...Object.keys(scope), fs.readFileSync(path.join(web, 'panels/eql/panel.js'), 'utf8'))(...Object.values(scope));
  const fixture = frameFixture();
  mount(root, fixture.bus);
  assert.equal(root.querySelector('#voice-ask').disabled, true);
  root.querySelector('#voice-ask').click();
  assert.equal(speechStarts, 0);
  fixture.bus.emit('query:ask', {text: 'show all handles'});
  assert.equal(requests.some(request => request.url.startsWith('/api/question')), false);
  releaseKnowledge();
  await new Promise(resolve => setTimeout(resolve, 0));
  const question = requests.find(request => request.url.startsWith('/api/question'));
  assert.equal(question.url, '/api/question?scene=saved');
  assert.deepEqual(JSON.parse(question.options.body), {text: 'show all handles'});
  const query = requests.find(request => request.url === '/api/eql?scene=saved');
  assert.deepEqual(JSON.parse(query.options.body), {code: 'an(entity(handle))', scope: 'recording'});
  assert.equal(fixture.events.some(event => event.name === 'voice:transcript'), false);
});
