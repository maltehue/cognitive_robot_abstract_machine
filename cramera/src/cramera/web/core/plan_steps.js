// Which kinds of plan step act on one of the placed objects.
// Transport drives, picks and places in one step; Pick and Place are the same work spelled
// out, for a world whose floor carries no costmap the transport could search. Whether a
// step acts on an object decides what the generated demo spawns and resolves for it, so
// the two kinds have to be named in one place rather than tested for one at a time.
(function () {
  'use strict';

  const ACTS_ON_AN_OBJECT = ['transport', 'pick', 'place'];
  const PUTS_AN_OBJECT_DOWN = ['transport', 'place'];

  window.PlanSteps = {
    // step kinds naming one of the placed objects
    actingOnAnObject: function () { return ACTS_ON_AN_OBJECT.slice(); },
    actsOnAnObject: function (step) {
      return !!step && ACTS_ON_AN_OBJECT.indexOf(step.type) >= 0;
    },
    // step kinds that put the object down somewhere, and so carry a target
    putsAnObjectDown: function (step) {
      return !!step && PUTS_AN_OBJECT_DOWN.indexOf(step.type) >= 0;
    },
    // ... at a semantic location (a surface or container) rather than an exact pose
    putsAnObjectDownAtASemanticTarget: function (step) {
      return window.PlanSteps.putsAnObjectDown(step)
        && !!step.params && step.params.targetMode === 'semantic';
    },
  };
})();
