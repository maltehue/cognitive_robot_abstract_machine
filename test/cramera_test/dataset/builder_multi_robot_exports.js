  window.multiRobotPage = {
    initialize(catalog) {
      builderState = new window.PlanBuilderState(catalog);
      builderState.addRobot(catalog[0].name);
      renderBlocks = renderSteps = renderObjects = showModelStatus = reshowIfGenerated = toast = function () {};
      renderRobotInstances();
    },
    get state() { return builderState; },
    get steps() { return steps; },
    set steps(value) { steps = value; },
    set live(value) { liveOn = value; },
    addRobotInstance, removeRobotInstance, selectRobotInstance, selectRobot,
    renderRobotInstances, synchronizeRobotPoses, synchronizeObjects, wireRobotControls,
    generate, generateClass, robotImportLines, robotSceneLines, startLive, runPlan,
  };
})();
