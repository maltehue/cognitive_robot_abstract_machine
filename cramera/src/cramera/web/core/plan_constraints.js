/* ============================================================================
 * core/plan_constraints.js — the Plan Builder's constraints.
 *
 * A constraint is written as a plain sentence ("Robot must look where it operates").
 * This turns it into the giskardpy goal it means, and — where a coraplex action already
 * enforces it — into the switch the generated step carries, so a constraint changes what
 * the robot does even without a live bridge to push it to.
 * ==========================================================================*/
(function (global) {
  'use strict';

  const GOAL = {
    VECTORS_ALIGNED: 'VectorsAligned',
    POINTING_AT: 'PointingAt',
    HEIGHT: 'HeightMonitor',
    DISTANCE: 'DistanceMonitor',
  };
  /* The giskardpy goals a sentence can compile to. */

  const STEP = {
    TRANSPORT: 'transport',
  };
  /* The plan steps a constraint can change the generated code of. */

  const ARGUMENT = {
    LOOK_AT_OPERATION_SITE: 'look_at_operation_site',
  };
  /* The switches a generated step carries to enforce a constraint. TransportAction
     looks at the object before picking it up and at the target before placing it, which
     is the whole of "look where it operates" and the only one coraplex has an action
     for; every other goal still needs the live bridge. */

  const ENFORCED_BY = {};
  ENFORCED_BY[GOAL.POINTING_AT] = { step: STEP.TRANSPORT, argument: ARGUMENT.LOOK_AT_OPERATION_SITE };
  /* Which step a goal is enforced on, and with which switch. */

  const NAMED_OBJECT = /\b(milk|bowl|spoon|fork|knife|plate|cup|mug|tray|bottle|flask|vial|beaker|tube|rack|sample|cereal|box|jar|glass|can|whisk|bread)\b/;
  /* The objects a sentence can name, so "keep the bowl above the table" is about the
     bowl even on a step that carries something else. */

  const LENGTH = /(\d+(?:\.\d+)?)\s*(mm|cm|centimet(?:er|re)s?|m\b|met(?:er|re)s?)/;
  /* A length in the sentence ("10 cm", "0.1 m"), which sets the goal's threshold. */

  const RULES = [
    {
      phrasing: /upright|stand up|stay up|vertical|straight up|tip over|tips?\b|tilt|spill|level|flat|horizontal|steady|balanc|no spill|don.?t (tip|spill|tilt)/,
      goal: GOAL.VECTORS_ALIGNED,
      params: function (object) {
        return { root_link: 'map', tip_link: object, tip_normal: [0, 0, 1], goal_normal: [0, 0, 1], threshold: 0.1 };
      },
    },
    {
      phrasing: /look|watch|gaze|point (at|the camera)|face the|observ|keep .*(in view|an eye)|focus on|keep sight|see the|where it (operat|work)/,
      goal: GOAL.POINTING_AT,
      params: function (object) {
        return { tip_link: 'head_camera', root_link: 'map', pointing_axis: [0, 0, 1], goal_point: '@operation_target', goal_point_body: object, threshold: 0.05 };
      },
    },
    {
      phrasing: /above|higher|over the|off the (table|ground|surface|bench)|keep .*(high|up high|elevated)|lift(ed)? (up|above)?/,
      goal: GOAL.HEIGHT,
      params: function (object, length) {
        return { tip_link: object, lower_limit: (length != null ? length : 0.05), upper_limit: 2.0 };
      },
    },
    {
      phrasing: /below|under(neath)?|lower than|keep .*(low|down|close to the (table|surface|ground))/,
      goal: GOAL.HEIGHT,
      params: function (object, length) {
        return { tip_link: object, lower_limit: 0.0, upper_limit: (length != null ? length : 0.1) };
      },
    },
    {
      phrasing: /away from|keep .*clear|clearance|distance|avoid|don.?t (hit|touch|collide|bump)|too close|stay .*away|far from|min(imum)? distance/,
      goal: GOAL.DISTANCE,
      params: function (object, length) {
        return { tip_link: object, lower_limit: (length != null ? length : 0.05), upper_limit: 5.0 };
      },
    },
  ];
  /* Sentence to goal, in the order they are tried. */

  /* The object a sentence is about: the one it names, else the one its step operates
     on, else nothing in particular. */
  function objectIn(text, stepParams) {
    const named = String(text).toLowerCase().match(NAMED_OBJECT);
    if (named) return named[1];
    if (stepParams && stepParams.object) return String(stepParams.object).replace(/\.(stl|obj|dae)$/i, '');
    return 'object';
  }

  /* The length a sentence gives, in meters, or null if it gives none. */
  function lengthIn(text) {
    const found = String(text).toLowerCase().match(LENGTH);
    if (!found) return null;
    const value = parseFloat(found[1]);
    if (found[2].indexOf('mm') === 0) return value / 1000;
    if (found[2].indexOf('c') === 0) return value / 100;
    return value;
  }

  global.PlanConstraints = {
    GOAL: GOAL,
    STEP: STEP,
    ARGUMENT: ARGUMENT,

    /* The goal a sentence means and the switch the generated step carries for it.
       `step` is the plan step it is attached to ({type, params}), whose transported
       object is the fallback subject of the sentence and whose type decides whether the
       generated plan can enforce the goal at all. A sentence no rule matches compiles to
       no goal. */
    compile: function (text, step) {
      const sentence = String(text).toLowerCase();
      const object = objectIn(text, step && step.params);
      const length = lengthIn(sentence);
      for (let i = 0; i < RULES.length; i++) {
        const rule = RULES[i];
        if (!rule.phrasing.test(sentence)) continue;
        const enforced = ENFORCED_BY[rule.goal];
        return {
          goal: rule.goal,
          params: rule.params(object, length),
          stepArgument: (enforced && step && step.type === enforced.step) ? enforced.argument : null,
        };
      }
      return { goal: null, params: {}, stepArgument: null };
    },

    /* The keyword arguments a step's generated action carries for these constraints,
       each listed once. Every one of them is a switch that is turned on. */
    stepArguments: function (constraints) {
      const keywordArguments = [];
      constraints.forEach(function (constraint) {
        if (!constraint.stepArgument) return;
        const code = constraint.stepArgument + '=True';
        if (keywordArguments.indexOf(code) < 0) keywordArguments.push(code);
      });
      return keywordArguments;
    },
  };
})(window);
