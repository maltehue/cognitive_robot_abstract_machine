// Exercise both real page generators without starting browser controls or a robot.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const webRoot = process.argv[2];
const scenario = JSON.parse(fs.readFileSync(0, 'utf8'));
const elements = new Map();
const context = vm.createContext({
  scenario,
  window: {addEventListener() {}},
  document: {getElementById(id) {
    if (!elements.has(id)) elements.set(id, {value: scenario.selections[id], addEventListener() {}});
    return elements.get(id);
  }},
});
for (const module of ['base_control', 'execution_environment', 'plan_steps', 'plan_constraints', 'builder_state']) {
  vm.runInContext(fs.readFileSync(path.join(webRoot, 'core', module + '.js'), 'utf8'), context);
}
context.PlanConstraints = context.window.PlanConstraints;
const source = fs.readFileSync(path.join(webRoot, 'plan_builder.js'), 'utf8');
const exportsSource = fs.readFileSync(path.join(__dirname, 'builder_demo_exports.js'), 'utf8');
vm.runInContext(source.slice(0, source.indexOf('  // ---------- boot ----------')) + exportsSource, context);
process.stdout.write(JSON.stringify(context.window.generatedDemos));
