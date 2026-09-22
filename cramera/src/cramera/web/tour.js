/* Offline storyboard: chapter timing and an adapter to the existing viewer. */
(function (global) {
  'use strict';

  const SCENE_NAME = /^[A-Za-z0-9_-]{1,64}$/;
  const VIEWS = Object.freeze({knowledge: 'knowledge', plan: 'plan', statechart: 'chart'});
  const SPEEDS = [0.5, 1, 2, 4];
  const LOAD_TIMEOUT = 45000;

  // %% manifest
  class Chapter {
    /** @param {object} entry One locally saved chapter description. */
    constructor(entry) {
      if (!entry || typeof entry.title !== 'string' || typeof entry.narration !== 'string') throw new Error('Kapitel benötigt Titel und Sprechertext.');
      if (!SCENE_NAME.test(entry.scene || '')) throw new Error('Ungültiger Name der Szene.');
      if (!Object.prototype.hasOwnProperty.call(VIEWS, entry.view)) throw new Error('Unbekannte Kapitelansicht.');
      if (typeof entry.playback !== 'boolean') throw new Error('Wiedergabe muss ausdrücklich gewählt sein.');
      if (!SPEEDS.includes(entry.speed)) throw new Error('Ungültige Geschwindigkeit der Aufnahme.');
      if (!Number.isFinite(entry.durationSeconds) || entry.durationSeconds < 0) throw new Error('Ungültige Mindestdauer des Kapitels.');
      if (entry.query !== null && typeof entry.query !== 'string') throw new Error('Kapitelanfrage muss Text oder null sein.');
      this.title = entry.title;
      this.narration = entry.narration;
      this.scene = entry.scene;
      this.view = entry.view;
      this.query = entry.query;
      this.playback = entry.playback;
      this.speed = entry.speed;
      this.durationSeconds = entry.durationSeconds;
    }
  }

  class Storyboard {
    /** @param {object} manifest Local storyboard JSON. */
    constructor(manifest) {
      if (!manifest || typeof manifest.title !== 'string' || !Array.isArray(manifest.chapters) || !manifest.chapters.length) throw new Error('Die Tour benötigt einen Titel und mindestens ein Kapitel.');
      this.title = manifest.title;
      this.description = manifest.description || '';
      this.chapters = manifest.chapters.map(entry => new Chapter(entry));
    }
  }

  // %% chapter lifecycle
  class Controller {
    /** @param {Storyboard} storyboard Chapters to present.
     * @param {Viewer} viewer Native scene adapter.
     * @param {function():number} clock Monotonic milliseconds.
     * @param {function(Controller):void} changed Update the presentation controls. */
    constructor(storyboard, viewer, clock, changed) {
      this.storyboard = storyboard;
      this.viewer = viewer;
      this.clock = clock;
      this.changed = changed;
      this.running = false;
      this.generation = 0;
      this.select(0);
    }

    /** @param {number} index Chapter to load, preserving the play/pause choice. */
    select(index) {
      if (index < 0 || index >= this.storyboard.chapters.length) return;
      this.viewer.pause();
      this.index = index;
      this.ready = false;
      this.completed = false;
      this.error = '';
      this.elapsed = 0;
      this.ended = false;
      this.lastTick = this.clock();
      const generation = ++this.generation;
      const chapter = this.storyboard.chapters[index];
      this.changed(this);
      this.viewer.load(chapter, {
        ready: () => {
          if (generation !== this.generation) return;
          this.ready = true;
          this.lastTick = this.clock();
          if (this.running && chapter.playback) this.viewer.play(chapter.speed);
          this.changed(this);
        },
        ended: () => { if (generation === this.generation) this.ended = true; },
        failed: message => {
          if (generation !== this.generation) return;
          this.error = message;
          this.ready = false;
          this.running = false;
          this.viewer.pause();
          this.changed(this);
        },
      });
    }

    /** Start or resume presentation time and recorded motion. */
    start() {
      if (this.error || this.completed) this.select(this.completed ? 0 : this.index);
      this.running = true;
      this.lastTick = this.clock();
      const chapter = this.storyboard.chapters[this.index];
      if (this.ready && chapter.playback && !this.ended) this.viewer.play(chapter.speed);
      this.changed(this);
    }

    /** Pause both narration time and the native trajectory. */
    pause() {
      if (this.running && this.ready) this.elapsed += this.clock() - this.lastTick;
      this.running = false;
      this.viewer.pause();
      this.changed(this);
    }

    /** Advance only after minimum presentation time and recorded motion finish. */
    tick() {
      if (!this.running || !this.ready || this.error || this.completed) return;
      const now = this.clock();
      this.elapsed += Math.max(0, now - this.lastTick);
      this.lastTick = now;
      const chapter = this.storyboard.chapters[this.index];
      if (this.elapsed >= chapter.durationSeconds * 1000 && (!chapter.playback || this.ended)) this.next();
      else this.changed(this);
    }

    /** Skip to the next chapter or visibly finish the tour. */
    next() {
      if (this.index + 1 < this.storyboard.chapters.length) this.select(this.index + 1);
      else {
        this.viewer.pause();
        this.completed = true;
        this.running = false;
        this.changed(this);
      }
    }

    /** Return to the preceding chapter. */
    previous() { this.select(Math.max(0, this.index - 1)); }
  }

  // %% same-origin viewer adapter
  class Viewer {
    /** @param {HTMLIFrameElement} frame Existing CRAMERA viewer container.
     * @param {function(function,number):number} schedule Schedule a loading timeout.
     * @param {function(number):void} cancel Cancel a loading timeout. */
    constructor(frame, schedule, cancel) {
      this.frame = frame;
      this.schedule = schedule;
      this.cancel = cancel;
      this.cleanups = [];
      this.robot = null;
    }

    /** @param {Chapter} chapter Scene, graph and optional question to show.
     * @param {object} events Ready, ended and failed lifecycle callbacks. */
    load(chapter, events) {
      const reuse = this.loadedScene === chapter.scene && this.robot !== null;
      this.destroy();
      let failed = false;
      const fail = message => {
        if (failed) return;
        failed = true;
        this.cancel(timeout);
        events.failed(message);
      };
      const timeout = this.schedule(() => fail('Die Szene ist nach 45 Sekunden nicht bereit. Aufnahme und lokale Assets prüfen.'), LOAD_TIMEOUT);
      this.cleanups.push(() => this.cancel(timeout));
      const loaded = () => {
        const child = this.frame.contentWindow;
        if (!child || !child.RobotView || !child.Bus) { fail('Der Viewer konnte nicht starten. Grafikunterstützung und lokale Dateien prüfen.'); return; }
        this.robot = child.RobotView;
        const ended = payload => { if (!failed && payload.step === '__done__') events.ended(); };
        const sceneError = payload => fail(payload.message);
        let robotReady = false, announced = false;
        const knowledgeStatus = child.document.getElementById('knowledge-status');
        let knowledgeReady = !!knowledgeStatus && knowledgeStatus.classList.contains('ready');
        const announce = () => {
          if (failed || announced || !robotReady || !knowledgeReady) return;
          announced = true;
          this.loadedScene = chapter.scene;
          this.cancel(timeout);
          this.robot.stopTrajectory();
          if (!reuse || chapter.playback) this.robot.seek(0);
          child.document.documentElement.dataset.tourView = chapter.view;
          child.Bus.emit('graph:view', {name: VIEWS[chapter.view]});
          if (chapter.query) child.Bus.emit('query:ask', {text: chapter.query});
          events.ready();
        };
        const knowledgeLoaded = () => { knowledgeReady = true; announce(); };
        child.Bus.on('scene:step', ended);
        child.Bus.on('scene:error', sceneError);
        child.Bus.on('knowledge:ready', knowledgeLoaded);
        child.Bus.on('knowledge:error', sceneError);
        this.cleanups.push(() => child.Bus.off('scene:step', ended), () => child.Bus.off('scene:error', sceneError), () => child.Bus.off('knowledge:ready', knowledgeLoaded), () => child.Bus.off('knowledge:error', sceneError));
        this.robot.onReady(() => {
          if (failed) return;
          if (this.robot.frameCount() <= 0) { fail('Die Aufnahme enthält keine abspielbaren Frames.'); return; }
          robotReady = true;
          announce();
        });
      };
      this.cleanups.push(() => { failed = true; });
      if (reuse) { loaded(); return; }
      this.frame.addEventListener('load', loaded);
      this.cleanups.push(() => { failed = true; this.frame.removeEventListener('load', loaded); });
      this.frame.src = 'index.html?scene=' + encodeURIComponent(chapter.scene) + '&offline=1&tour=1';
    }

    /** @param {number} speed Native playback speed multiplier. */
    play(speed) {
      if (!this.robot) return;
      this.robot.setPlaybackSpeed(speed);
      this.robot.playTrajectory();
    }

    /** Pause the current native trajectory without losing its playhead. */
    pause() { if (this.robot) this.robot.stopTrajectory(); }

    /** Release listeners and timers belonging to the preceding scene. */
    destroy() {
      this.pause();
      this.cleanups.forEach(cleanup => cleanup());
      this.cleanups = [];
      this.robot = null;
      this.loadedScene = null;
    }
  }

  // %% presentation page
  class Page {
    /** @param {Document} document Tour controls and same-origin iframe. */
    constructor(document) {
      this.document = document;
      this.buttons = [];
      this.viewer = new Viewer(document.getElementById('tour-viewer'), global.setTimeout.bind(global), global.clearTimeout.bind(global));
    }

    /** Fetch the selected local storyboard and prepare its first chapter. */
    async load() {
      const name = new URLSearchParams(global.location.search).get('tour') || 'offline_tour';
      if (!SCENE_NAME.test(name)) throw new Error('Ungültiger Tourname.');
      const response = await global.fetch('/scenes/' + encodeURIComponent(name) + '/storyboard.json');
      if (!response.ok) throw new Error('Die lokale Tour wurde nicht gefunden: ' + name);
      const storyboard = new Storyboard(await response.json());
      this.document.getElementById('tour-title').textContent = storyboard.title;
      this.document.getElementById('tour-description').textContent = storyboard.description;
      this.document.title = storyboard.title + ' · CRAMERA';
      storyboard.chapters.forEach((chapter, index) => {
        const item = this.document.createElement('li');
        const button = this.document.createElement('button');
        button.type = 'button';
        const number = this.document.createElement('span');
        number.className = 'chapter-number'; number.textContent = String(index + 1).padStart(2, '0');
        button.appendChild(number);
        button.appendChild(this.document.createTextNode(chapter.title));
        button.addEventListener('click', () => this.controller.select(index));
        item.appendChild(button);
        this.document.getElementById('tour-chapters').appendChild(item);
        this.buttons.push(button);
      });
      this.controller = new Controller(storyboard, this.viewer, () => global.performance.now(), controller => this.render(controller));
      this.document.getElementById('tour-start').addEventListener('click', () => this.controller.running ? this.controller.pause() : this.controller.start());
      this.document.getElementById('tour-next').addEventListener('click', () => this.controller.next());
      this.document.getElementById('tour-previous').addEventListener('click', () => this.controller.previous());
      this.timer = global.setInterval(() => this.controller.tick(), 100);
      global.addEventListener('beforeunload', () => { global.clearInterval(this.timer); this.viewer.destroy(); });
    }

    /** @param {Controller} controller Current chapter, readiness and playback state. */
    render(controller) {
      const chapter = controller.storyboard.chapters[controller.index];
      const status = this.document.getElementById('tour-status');
      if (controller.ready && !status.dataset.readyAt) status.dataset.readyAt = String(Date.now());
      this.document.getElementById('chapter-title').textContent = chapter.title;
      this.document.getElementById('chapter-narration').textContent = chapter.narration;
      this.document.getElementById('tour-counter').textContent = 'KAPITEL ' + (controller.index + 1) + ' / ' + this.buttons.length;
      this.buttons.forEach((button, index) => button.setAttribute('aria-current', index === controller.index ? 'step' : 'false'));
      this.document.getElementById('tour-complete').hidden = !controller.completed;
      const start = this.document.getElementById('tour-start');
      start.disabled = !controller.ready && !controller.error && !controller.running;
      start.textContent = controller.completed ? 'Neu starten' : controller.error ? 'Erneut versuchen' : controller.running ? 'Pause' : controller.elapsed ? 'Fortsetzen' : 'Start';
      this.document.getElementById('tour-previous').disabled = controller.index === 0;
      this.document.getElementById('tour-next').disabled = controller.completed;
      this.document.getElementById('tour-next').textContent = controller.index + 1 === this.buttons.length ? 'Abschließen' : 'Weiter →';
      const progress = this.document.getElementById('tour-progress');
      progress.max = this.buttons.length;
      progress.value = controller.completed ? this.buttons.length : controller.index;
      let text = controller.error || (controller.ready ? 'Bereit · Start beginnt die geführte Tour.' : 'Szene und Aufnahme werden geladen …');
      if (controller.completed) text = 'Abgeschlossen · Die Tour kann erneut gestartet werden.';
      else if (controller.running && controller.ready) {
        const remaining = Math.max(0, Math.ceil(chapter.durationSeconds - controller.elapsed / 1000));
        text = remaining ? 'Vorführung läuft · mindestens noch ' + remaining + ' s in diesem Kapitel' : chapter.playback && !controller.ended ? 'Aufnahme läuft · weiter nach dem letzten Frame' : 'Nächstes Kapitel …';
      } else if (controller.ready && controller.elapsed) text = 'Pausiert · Sprecherzeit und Aufnahme sind angehalten.';
      status.classList.toggle('error', !!controller.error);
      if (status.textContent !== text) status.textContent = text;
      this.document.body.dataset.state = controller.error ? 'error' : controller.completed ? 'complete' : controller.ready ? (controller.running ? 'playing' : 'ready') : 'loading';
    }
  }

  global.OfflineTour = {Chapter, Storyboard, Controller, Viewer, Page};
  if (global.document) global.addEventListener('DOMContentLoaded', () => {
    const page = new Page(global.document);
    page.load().catch(error => {
      const status = global.document.getElementById('tour-status');
      status.textContent = error.message;
      status.classList.add('error');
      global.document.body.dataset.state = 'error';
    });
  });
})(window);
