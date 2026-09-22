/* Poses and selection of independently articulated scene models. */
(function (global) {
  'use strict';

  class ModelPoses {
    /** Resolve the selected instance after models finish loading in any order. */
    static primary(models, robot) {
      if (!robot) return null;
      return models.find(function (model) {
        if (!model.robot) return false;
        return robot.identifier ? model.identifier === robot.identifier : model.name === robot.name;
      }) || null;
    }

    /** Apply independent root tracks, retaining a pose when its next frame is absent. */
    static apply(models, current, next, fraction, setPose) {
      if (!current) return;
      models.forEach(function (model) {
        const pose = current[model.prefix];
        if (!pose) return;
        const following = (next && next[model.prefix]) || pose;
        setPose(model.obj, pose, following, fraction);
      });
    }

    /** Read the saved annotation belonging to one loaded model. */
    static metadata(scene, model) {
      const entries = scene.robots || (scene.robot ? [scene.robot] : []);
      return entries.find(function (robot) {
        return robot.identifier ? robot.identifier === model.identifier : robot.name === model.name;
      }) || null;
    }

    /** Resolve a link to the query identity of its own robot part. */
    static partFor(scene, model, link) {
      const robot = ModelPoses.metadata(scene, model);
      if (!robot) return null;
      const parts = robot.parts || {};
      const part = Object.keys(parts).find(function (name) {
        return parts[name].some(function (local) {
          return local === link || (robot.prefix && robot.prefix + '/' + local === link);
        });
      });
      if (!part) return null;
      return scene.robots && scene.robots.length > 1 ? robot.identifier + '/' + part : part;
    }

    /** Match a query's whole robot, part or link without selecting another instance. */
    static highlighted(scene, model, link, selected) {
      const robot = ModelPoses.metadata(scene, model);
      if (!robot) return false;
      const part = ModelPoses.partFor(scene, model, link);
      const annotation = robot.identifier && robot.label ? robot.identifier + '/' + robot.label : null;
      const single = !scene.robots || scene.robots.length <= 1;
      return !!(selected[robot.identifier || robot.name] || (single && selected[robot.name]) || (annotation && selected[annotation]) ||
        (part && (selected[part] || selected[(robot.prefix || '') + '/' + part])) ||
        selected[link] || selected['urdf:' + link]);
    }
  }

  global.ModelPoses = ModelPoses;
})(window);
