// The Plan Builder's robots in the form a saved demo setup takes (cramera.demo_setup):
// the environment and where its own joints stand, every robot with where it stands, the
// joint state and localization topics it follows, whether it repeats its plan, and the
// plan itself, and every box lying about to be carried, where it starts and how big it
// is. The builder writes its state into that form to save it and reads a setup back to
// open it.
(function () {
  'use strict';

  function plainStep(step) { return {type: step.type, params: Object.assign({}, step.params)}; }
  // a file by its path, with the placement an opened setup gave that same file
  function filePayload(path, rootPlacement) {
    if (rootPlacement && rootPlacement.path === path) return {path: path, rootPlacement: rootPlacement.rootPlacement};
    return {path: path};
  }
  // the environment as the server reads it: a file by its path; a map by its kind and
  // class, for the server to refuse by name, since a setup names a file
  function environmentPayload(environment, rootPlacement) {
    if (!environment) return null;
    if (typeof environment === 'string') return filePayload(environment, rootPlacement);
    if (environment.path) return filePayload(environment.path, rootPlacement);
    return {kind: environment.kind, cls: environment.cls};
  }
  // how a setup placed its USD scene, or null for a setup that left the default
  function rootPlacementOf(environment) {
    if (!environment || !environment.rootPlacement) return null;
    return {path: environment.path, rootPlacement: environment.rootPlacement};
  }
  // a box the builder lists, as a setup writes it: only a box with a size is a box the
  // scene can build; a mesh placed for a generated demo is spawned by that demo instead
  function boxPayload(object) {
    return {name: object.mesh, x: object.x, y: object.y, z: object.z, yaw: object.yaw || 0, size: object.size.slice()};
  }

  window.DemoSetupForm = {
    /**
     * @param {object} state The builder's PlanBuilderState.
     * @param {Array<object>} activeSteps The plan being edited, which belongs to the
     *   active robot and is not yet stored on it.
     * @param {string|object} environment The environment file's path, or the offered
     *   environment (see PlanBuilderState.offerEnvironments in core/builder_state.js).
     * @param {Array<object>} [objects] The builder's objects; the ones with a size are
     *   written as the setup's boxes, by their mesh key as the scene names the body.
     * @returns {object} The setup in the form the server saves.
     */
    toPayload: function (state, activeSteps, environment, objects) {
      return {
        environment: environmentPayload(environment, state.environmentRootPlacement),
        objects: (objects || []).filter(function (object) { return Array.isArray(object.size); }).map(boxPayload),
        robots: state.instances.map(function (robot) {
          const steps = robot.id === state.activeIdentifier ? activeSteps : robot.steps;
          return {
            identifier: robot.id, label: robot.label, model: robot.model,
            x: robot.x, y: robot.y, yaw: robot.yaw,
            jointStateTopic: robot.joint_state_topic || '',
            localizationTopic: robot.localization_topic || '',
            repeatsPlan: !!robot.repeats_plan,
            steps: (steps || []).map(plainStep),
          };
        }),
        environmentJointPositions: Object.assign({}, state.environmentJointPositions),
        environmentGeometry: state.environmentGeometry,
      };
    },

    /**
     * Replace the builder's robots with a setup's, and note its boxes on the state as
     * {name, x, y, z, yaw, size} for the page to list.
     * @param {object} state The builder's PlanBuilderState.
     * @param {object} payload A setup in the form the server hands out.
     * @param {function(string, object): object} makeStep Makes a builder step of a type
     *   and its parameters, so opened steps are numbered as the builder numbers its own.
     * @returns {object} The robot selected afterwards: the setup's first.
     * @throws {RangeError} If the setup names a robot model the builder does not offer.
     */
    applyTo: function (state, payload, makeStep) {
      payload.robots.forEach(function (robot) {
        if (!state.robot(robot.model)) throw new RangeError('Unknown robot model: ' + robot.model);
      });
      state.instances = payload.robots.map(function (robot) {
        return {
          id: robot.identifier, model: robot.model, label: robot.label,
          x: robot.x, y: robot.y, yaw: robot.yaw, joint_positions: {},
          joint_state_topic: robot.jointStateTopic || '',
          localization_topic: robot.localizationTopic || '',
          repeats_plan: !!robot.repeatsPlan,
          steps: (robot.steps || []).map(function (step) { return makeStep(step.type, Object.assign({}, step.params)); }),
        };
      });
      state.authoredRobotPoses = new Map();
      state.objects = (payload.objects || []).map(function (box) {
        return {name: box.name, x: box.x, y: box.y, z: box.z, yaw: box.yaw || 0, size: (box.size || []).slice()};
      });
      state.environmentJointPositions = Object.assign({}, payload.environmentJointPositions || {});
      state.environmentGeometry = payload.environmentGeometry || window.PlanBuilderState.DRAWN_GEOMETRY.VISUAL;
      state.environmentRootPlacement = rootPlacementOf(payload.environment);
      state.activeIdentifier = state.instances[0].id;
      return state.instances[0];
    },
  };
})();
