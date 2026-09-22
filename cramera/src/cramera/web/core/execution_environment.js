// Collision-enabled coraplex execution for generated plans and live scaffolds.
// Scripts use the environment name; RobotDemonstration uses its matching flag.
(function () {
  'use strict';

  const ENVIRONMENTS = [
    {
      name: 'simulated_robot_advanced',
      label: 'always on — avoid collisions during motion',
      collisionAvoidance: true,
    },
  ];

  window.ExecutionEnvironments = {
    // every environment to offer, in the order they are offered
    all: function () { return ENVIRONMENTS.slice(); },
    // Legacy or missing selections also retain collision avoidance.
    byName: function (name) {
      const found = ENVIRONMENTS.filter(function (e) { return e.name === name; });
      return found.length ? found[0] : ENVIRONMENTS[0];
    },
  };
})();
