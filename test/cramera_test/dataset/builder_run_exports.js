  window.builderRunTest = {
    begin(live = true) { liveOn = live; return ++_runMonitor; },
    monitorRun, pollLive, stopRunMonitor, stopLive, startLive, runPlan,
    get live() { return liveOn; },
    clearSteps() { steps = []; },
    rejectCapture() { synchronizeObjects = function () { return Promise.reject(new Error('capture unavailable')); }; },
    prepare() {
      robotInfo = function () { return {name: 'PR2'}; };
      generate = generateSelected = function () { return 'generated fixture'; };
      synchronizeObjects = function () { return Promise.resolve(); };
      fetchSurfaces = showModelStatus = toast = function () {};
      steps = [{type: 'park_arms', params: {arm: 'BOTH'}}];
    },
  };
})();
