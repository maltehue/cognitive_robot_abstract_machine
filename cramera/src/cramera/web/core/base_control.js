// Whether the robot's base may drive while an arm reaches (whole-body control).
// With it on, the reach's Cartesian goal is solved against the world root, so the base
// drives to help the gripper get there -- past the pose a Navigate step put it in, and
// through whatever stands in the way. A plan built here says where the robot stands, so
// standing still is the default; the other choices keep the robot's own setting or force
// whole-body control on.
(function () {
  'use strict';

  const CHOICES = [
    {
      name: 'stand_still',
      label: 'stand still — the Navigate step decides where it stands',
      fullBodyControlled: false,
    },
    {
      name: 'robot_default',
      label: 'as the robot is configured',
      fullBodyControlled: null,
    },
    {
      name: 'may_drive',
      label: 'may drive to help the arm reach',
      fullBodyControlled: true,
    },
  ];

  window.BaseControl = {
    // every choice to offer, in the order they are offered
    all: function () { return CHOICES.slice(); },
    // the choice of that name, falling back to the first one offered
    byName: function (name) {
      const found = CHOICES.filter(function (c) { return c.name === name; });
      return found.length ? found[0] : CHOICES[0];
    },
    // whether a choice writes the setting at all, as opposed to leaving the robot's own
    pinsTheSetting: function (choice) { return choice.fullBodyControlled !== null; },
  };
})();
