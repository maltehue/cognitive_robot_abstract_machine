/* Robot capabilities and authoritative object poses used during plan authoring. */
(function () {
  'use strict';

  class PlanBuilderState {
    /** @returns {{x: number, y: number}} Initial position rehearsed with PR2 in the apartment. */
    static initialRobotPosition() { return {x: 1.0, y: 2.5}; }

    /** @param {object|null} snapshot Published plan tree. @returns {object|null} Its root node. */
    static planRoot(snapshot) {
      return snapshot && Array.isArray(snapshot.nodes) ? snapshot.nodes.find((node) => node.parent === null) || null : null;
    }

    /**
     * Interpret the authoritative result of the current plan.
     * @param {object|null} snapshot Published plan tree.
     * @param {string|null} previousRoot Root belonging to the scene replaced at startup.
     * @returns {{message: string, style: string}|null} A completed run's presentation.
     */
    static planResult(snapshot, previousRoot) {
      const root = PlanBuilderState.planRoot(snapshot);
      if (!root || root.id === previousRoot) return null;
      return PlanBuilderState.PLAN_RESULTS[root.status] || null;
    }

    /** Terminal native plan states and their live-scene presentation. */
    static PLAN_RESULTS = Object.freeze({
      SUCCEEDED: Object.freeze({message: '● completed — live scene remains available', style: 'ok'}),
      FAILED: Object.freeze({message: '● plan failed — open the run log for details', style: 'err'}),
      INTERRUPTED: Object.freeze({message: '● plan interrupted — live scene remains available', style: 'err'}),
    });

    static RUN_PROGRESS = Object.freeze({
      RUNNING: Object.freeze({message: '● running — watch the robot in the 3D view', style: 'ok'}),
      PLACEMENT_SEARCH: Object.freeze({message: '● searching for a reachable placement — the plan is still running', style: 'ok'}),
    });

    static planProgress(snapshot, previousRoot) {
      const root = PlanBuilderState.planRoot(snapshot);
      if (!root || root.id === previousRoot || root.status !== 'RUNNING') return null;
      if (snapshot.nodes.some((node) => node.kind === 'MotionNode' && ['RUNNING', 'PAUSED'].includes(node.status))) return null;
      const nodes = new Map(snapshot.nodes.map((node) => [node.id, node]));
      for (const node of snapshot.nodes) {
        if (node.kind !== 'UnderspecifiedNode' || node.status !== 'RUNNING') continue;
        let ancestor = nodes.get(node.parent);
        while (ancestor && ancestor.status === 'RUNNING') {
          if (ancestor.label === 'MoveAndPlaceAction') return PlanBuilderState.RUN_PROGRESS.PLACEMENT_SEARCH;
          ancestor = nodes.get(ancestor.parent);
        }
      }
      return null;
    }

    /** @param {Array<object>} robots Annotated robot choices from the server. */
    constructor(robots) {
      this.robots = robots;
      this.failures = new Map();
      this.positions = new Map();
      this.instances = [];
      this.activeIdentifier = null;
      this.instanceSequence = 1;
      this.authoredRobotPoses = new Map();
    }

    /** @param {string} model Installed model name. @param {object} [pose] Starting world-frame pose. @returns {object} The newly selected instance. */
    addRobot(model, pose) {
      if (!this.robot(model)) throw new RangeError('Unknown robot model: ' + model);
      const position = Object.assign({yaw: 0}, PlanBuilderState.initialRobotPosition(), pose);
      const instance = {id: 'robot_' + this.instanceSequence++, model: model,
        label: model + ' ' + (this.instances.filter((item) => item.model === model).length + 1),
        x: position.x, y: position.y, yaw: position.yaw, joint_positions: {}, steps: []};
      this.instances.push(instance);
      this.activeIdentifier = instance.id;
      return instance;
    }

    /** @returns {object|null} The instance whose plan is being authored. */
    activeRobot() {
      return this.instances.find((instance) => instance.id === this.activeIdentifier) || null;
    }

    /** @param {string} identifier Stable scene instance identifier. @param {Array<object>} steps Outgoing instance's plan. @returns {object} Selected instance. */
    selectInstance(identifier, steps) {
      const selected = this.instances.find((instance) => instance.id === identifier);
      if (!selected) throw new RangeError('Unknown robot instance: ' + identifier);
      const previous = this.activeRobot();
      if (previous) previous.steps = steps;
      this.activeIdentifier = identifier;
      return selected;
    }

    /** @param {string} identifier Stable scene instance identifier. @param {object} changes Edited name, model, or finite pose coordinates. */
    updateRobot(identifier, changes) {
      const instance = this.instances.find((item) => item.id === identifier);
      if (!instance) throw new RangeError('Unknown robot instance: ' + identifier);
      if (changes.model && !this.robot(changes.model)) throw new RangeError('Unknown robot model: ' + changes.model);
      for (const coordinate of ['x', 'y', 'yaw']) {
        if (coordinate in changes && !Number.isFinite(changes[coordinate])) throw new RangeError('Robot pose must be finite');
      }
      if (changes.model) {
        instance.model = changes.model;
        instance.joint_positions = {};
        instance.steps = this.adaptSteps(instance.steps, changes.model);
      }
      if ('label' in changes) instance.label = changes.label.trim() || instance.model;
      for (const coordinate of ['x', 'y', 'yaw']) {
        if (coordinate in changes) {
          instance[coordinate] = changes[coordinate];
          this.authoredRobotPoses.set(identifier, {x: instance.x, y: instance.y, yaw: instance.yaw});
        }
      }
    }

    /** @returns {Map<string, object>} Pending pose edits included in a scene launch. */
    snapshotRobotPoseEdits() {
      return new Map(this.authoredRobotPoses);
    }

    /** @param {Map<string, object>} edits Pose edits accepted by a successful scene launch. */
    acknowledgeRobotPoseEdits(edits) {
      edits.forEach((pose, identifier) => {
        if (this.authoredRobotPoses.get(identifier) === pose) this.authoredRobotPoses.delete(identifier);
      });
    }

    /** @param {object|null} snapshot Independent live robot poses before replacing a scene. */
    captureRobots(snapshot) {
      if (!snapshot || !Array.isArray(snapshot.robots)) return;
      snapshot.robots.forEach((robot) => {
        const instance = this.instances.find((item) => item.id === robot.identifier && item.model === robot.model);
        const pose = robot.pose;
        if (!instance || !Array.isArray(pose) || pose.length !== 7 || !pose.every(Number.isFinite)) return;
        if (robot.joint_positions && Object.values(robot.joint_positions).every(Number.isFinite)) {
          instance.joint_positions = Object.assign({}, robot.joint_positions);
        }
        const yaw = Math.atan2(2 * (pose[6] * pose[5] + pose[3] * pose[4]), 1 - 2 * (pose[4] ** 2 + pose[5] ** 2));
        const authored = this.authoredRobotPoses.get(instance.id);
        if (authored) {
          const angle = Math.atan2(Math.sin(authored.yaw - yaw), Math.cos(authored.yaw - yaw));
          if (Math.abs(authored.x - pose[0]) > 0.00001 || Math.abs(authored.y - pose[1]) > 0.00001 || Math.abs(angle) > 0.00001) return;
          this.authoredRobotPoses.delete(instance.id);
        }
        instance.x = pose[0]; instance.y = pose[1]; instance.yaw = yaw;
      });
    }

    /** @param {string} identifier Instance to remove. @returns {object|null} Remaining selected robot; null when the last robot is protected. */
    removeRobot(identifier) {
      if (this.instances.length <= 1) return null;
      this.instances = this.instances.filter((instance) => instance.id !== identifier);
      this.authoredRobotPoses.delete(identifier);
      if (this.activeIdentifier === identifier) this.activeIdentifier = this.instances[0].id;
      return this.activeRobot();
    }

    /** @param {string} name Selected robot name. @returns {object|null} Its catalog entry. */
    robot(name) {
      return this.robots.find(function (robot) { return robot.name === name; }) || null;
    }

    /**
     * Preserve supported steps and select a valid arm for the named robot.
     * @param {Array<object>} steps Authored plan steps.
     * @param {string} name Selected robot name.
     * @returns {Array<object>} Steps supported by the selected robot.
     */
    adaptSteps(steps, name) {
      const robot = this.robot(name);
      if (!robot) return [];
      return steps.filter(function (step) {
        return robot.steps.indexOf(step.type) >= 0;
      }).map(function (step) {
        const params = Object.assign({}, step.params);
        if (params.arm && robot.arms.indexOf(params.arm) < 0) params.arm = robot.arms[0];
        return Object.assign({}, step, {params: params});
      });
    }

    /**
     * Record the diagnostic for a failed scene start.
     * @param {string} name Robot whose scene failed to start.
     * @param {string} log Captured process output.
     * @returns {string} Actionable failure description.
     */
    recordFailure(name, log) {
      const dependency = log.match(/package ['"]([^'"]+)['"] not found/i);
      const missing = log.match(/No world entity with name ['"]([^'"]+)['"] found/);
      let message = 'Scene process failed. Open the run log for details.';
      if (dependency) message = 'Missing ROS package: ' + dependency[1] + '. Install the robot description dependencies, then retry.';
      else if (missing && log.indexOf('/robots/') >= 0) message = 'Robot model does not match its annotation: missing ' + missing[1] + '.';
      this.failures.set(name, message);
      return message;
    }

    /** @param {string} name Robot whose last scene-start failure should be shown. */
    failure(name) { return this.failures.get(name) || ''; }

    /** @param {string} name Robot whose scene has now started successfully. */
    clearFailure(name) { this.failures.delete(name); }

    /**
     * Keep an authored pose until the bridge observes the same position and rotation.
     * @param {string} key Object mesh key.
     * @param {Array<number>} position World-frame x, y and z coordinates.
     * @param {Array<number>} [quaternion] World-frame rotation in x, y, z, w order.
     */
    authoredPosition(key, position, quaternion) {
      this.positions.set(key, {position: position.slice(), quaternion: quaternion ? quaternion.slice() : null});
    }

    /**
     * Merge finite world-frame poses into the authored objects.
     * @param {Array<object>} objects Authored objects receiving current poses.
     * @param {Object<string, Array<number>>} captured Bridge poses indexed by mesh key.
     */
    capture(objects, captured) {
      objects.forEach((object) => {
        let pose = captured[object.mesh];
        const authored = this.positions.get(object.mesh);
        if (authored) {
          const valid = Array.isArray(pose) && pose.length === 7 && pose.every(Number.isFinite);
          const samePosition = valid && authored.position.every(function (value, index) { return Math.abs(value - pose[index]) < 0.00001; });
          const sameRotation = !authored.quaternion || (valid && authored.quaternion.every(function (value, index) { return Math.abs(value - pose[index + 3]) < 0.00001; }));
          if (samePosition && sameRotation) this.positions.delete(object.mesh);
          object.x = authored.position[0]; object.y = authored.position[1]; object.z = authored.position[2];
          pose = authored.position.concat(authored.quaternion || (valid ? pose.slice(3) : []));
        }
        if (!Array.isArray(pose) || pose.length !== 7 || !pose.every(Number.isFinite)) return;
        const x = pose[3], y = pose[4], z = pose[5], w = pose[6];
        object.x = pose[0]; object.y = pose[1]; object.z = pose[2];
        object.roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
        object.pitch = Math.asin(Math.max(-1, Math.min(1, 2 * (w * y - z * x))));
        object.yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
      });
    }
  }

  window.PlanBuilderState = PlanBuilderState;
})();
