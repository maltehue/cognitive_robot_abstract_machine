'use strict';
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../../../cramera/src/cramera/web/panels/robot_scene/panel.js'), 'utf8');
const sent = [];
const posts = [];
const context = vm.createContext({
  window: {parent: {postMessage(message) {sent.push(message);}}},
  performance: {now: () => 1000}, lastMovePost: 0,
  liveUrl: () => 'http://localhost:8765',
  fetch(url, options) {posts.push({url, options}); return Promise.resolve();},
});
vm.runInContext(source.slice(source.indexOf('  function postLiveMove('), source.indexOf('  // %% joints moved by hand')), context);
context.postLiveMove('milk.stl', 1.2345, 2.3456, 0.8765, false);
assert.strictEqual(sent.length, 0);
context.postLiveMove('milk.stl', 1.2345, 2.3456, 0.8765, true);
assert.strictEqual(sent.length, 1);
assert.strictEqual(sent[0].type, 'cramera-object-settled');
assert.strictEqual(sent[0].key, 'milk.stl');
assert.deepStrictEqual(JSON.parse(JSON.stringify(sent[0].position)), JSON.parse(posts[1].options.body).position);
