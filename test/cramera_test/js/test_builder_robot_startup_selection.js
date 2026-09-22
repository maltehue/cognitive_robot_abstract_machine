'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const elements = new Map();
const element = (identifier) => {
  if (!elements.has(identifier)) elements.set(identifier, {
    value: '', textContent: '', innerHTML: '', style: {}, children: [], src: 'index.html?scene',
    addEventListener() {}, replaceChildren() { this.children = []; }, appendChild(child) { this.children.push(child); },
  });
  return elements.get(identifier);
};
const context = vm.createContext({
  window: {addEventListener() {}, location: {hostname: 'localhost'}},
  document: {getElementById: element, createElement: () => element('option' + elements.size)},
  fetch: async (url) => url.endsWith('/captured_objects') ? new Promise(() => {}) : ({ok: true, json: async () => ({ok: true, nodes: [], returncode: null})}),
  setInterval() { return 1; }, clearInterval() {}, setTimeout() {},
});
const web = path.join(__dirname, '../../../cramera/src/cramera/web');
for (const module of ['base_control', 'execution_environment', 'plan_steps', 'plan_constraints', 'builder_state']) {
  vm.runInContext(fs.readFileSync(path.join(web, 'core', module + '.js'), 'utf8'), context);
}
context.PlanConstraints = context.window.PlanConstraints;
const source = fs.readFileSync(path.join(web, 'plan_builder.js'), 'utf8');
const exportsSource = fs.readFileSync(path.join(__dirname, '../dataset/builder_multi_robot_exports.js'), 'utf8');
vm.runInContext(source.slice(0, source.indexOf('  // ---------- boot ----------')) + exportsSource, context, {filename: path.join(web, 'plan_builder.js')});
const page = context.window.multiRobotPage;
page.initialize([{name: 'PR2', cls: 'PR2', import: 'from semantic_digital_twin.robots.pr2 import PR2', arms: ['BOTH'], steps: ['park_arms']}]);
const first = page.state.activeRobot();
page.addRobotInstance();
const second = page.state.activeRobot();
(async () => {
  await page.startLive();
  await page.selectRobotInstance(first.id);
  assert.equal(page.state.activeRobot(), second);
  assert.equal(element('pb-robot-instance').value, second.id);
  assert.equal(element('pb-robot-instance').disabled, true);
  assert.equal(element('pb-add-robot').disabled, true);
  assert.equal(element('pb-robot').disabled, true);
  console.log('A scene startup keeps its selected robot and authoring controls stable.');
})().catch((error) => { console.error(error); process.exitCode = 1; });
