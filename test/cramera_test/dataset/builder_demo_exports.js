  builderState = new window.PlanBuilderState(scenario.robots);
  if (scenario.instances) {
    scenario.instances.forEach((instance) => builderState.addRobot(instance.model, instance));
    builderState.selectInstance(scenario.activeIdentifier, []);
  }
  objects = scenario.objects;
  steps = scenario.steps;
  if (scenario.robotXY) robotXY = scenario.robotXY;
  builderState.capture(objects, scenario.captured);
  window.generatedDemos = {script: generate(), class: generateClass()};
  if (scenario.inspectStart) {
    window.generatedDemos.start = {position: robotXY, inputs: {x: Number($('pb-rx').value), y: Number($('pb-ry').value)}};
  }
})();
