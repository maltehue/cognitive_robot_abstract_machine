/* config.js of the published demo site: a plain file server has no EQL or knowledge
 * API, so the site shows the 3D scene alone and drops the topbar's links to the pages
 * that need the server. */
document.documentElement.classList.add(SceneContext.SCENE_ONLY_CLASS);
window.CRAMERA_CONFIG = {
  layout: { left: ['robot-scene'] },
};
