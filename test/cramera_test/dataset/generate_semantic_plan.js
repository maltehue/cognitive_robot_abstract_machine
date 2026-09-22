// Execute the page's real generators without booting the browser controls.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const webRoot = process.argv[2];
const step = JSON.parse(fs.readFileSync(0, 'utf8'));
const context = vm.createContext({
  window: { addEventListener() {} },
  document: { getElementById() { return { addEventListener() {} }; } },
});
for (const module of ['plan_steps', 'plan_constraints', 'builder_state']) {
  vm.runInContext(fs.readFileSync(path.join(webRoot, 'core', module + '.js'), 'utf8'), context);
}
context.PlanConstraints = context.window.PlanConstraints;
const source = fs.readFileSync(path.join(webRoot, 'plan_builder.js'), 'utf8');
const exportsSource = fs.readFileSync(path.join(__dirname, 'semantic_plan_exports.js'), 'utf8');
vm.runInContext(source.slice(0, source.indexOf('  // ---------- boot ----------')) + exportsSource, context);
process.stdout.write(JSON.stringify({
  setup: [context.window.semanticPlan.imports([step]), ...context.window.semanticPlan.setup([step], '')].join('\n'),
  action: context.window.semanticPlan.action(step),
}));
