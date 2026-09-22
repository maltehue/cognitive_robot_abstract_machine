/* ============================================================================
 * core/camera-follow.js — whether the scene camera keeps the moving robot in view.
 *
 * With the follow on, the orbit target glides after the robot while a recording
 * plays or a live demo runs; with it off, the camera stays where the viewer
 * pointed it and the robot may leave the picture.
 *
 * Pure preference state, no DOM and no three.js, so it is testable under node.
 * ==========================================================================*/
(function (global) {
  'use strict';

  const KEY = 'cramera.camera-follow';
  /* localStorage key of whether the camera follows the robot. */

  /* Whether the camera follows the robot; a viewer that never touched the switch does. */
  function on(storage) {
    return storage.getItem(KEY) !== 'false';
  }

  /* Switch the follow on or off; returns the state as stored. */
  function set(storage, follow) {
    storage.setItem(KEY, follow ? 'true' : 'false');
    return on(storage);
  }

  // %% orbit control ownership
  class CameraController {
    /** @param {THREE.OrbitControls} controls Camera controls owned by the scene. */
    constructor(controls) {
      this.controls = controls;
    }

    /** Follow a world point while camera input is enabled. */
    follow(target, smoothing) {
      if (this.controls.enabled) this.controls.target.lerp(target, smoothing);
    }

    /** Preserve the camera during object placement, including orbit damping. */
    update() {
      return this.controls.enabled ? this.controls.update() : false;
    }
  }

  global.CameraFollow = {
    KEY: KEY,
    on: on,
    set: set,
    Controller: CameraController,
  };
})(window);
