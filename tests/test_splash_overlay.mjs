import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..');

const require = createRequire(import.meta.url);
const splashMod = require('../web/ui/splash_overlay.js');

const {
  BOOT_STAGE_ORDER,
  BOOT_STAGE_LABELS,
  BOOT_PHASE_ORDER,
  BOOT_PHASE_RANGES,
  DISMISS_REASONS,
  computeBootProgress,
  formatElapsedClock,
  resolveBootPhase,
  clampMonotonicPhaseProgress,
  sanitizeScriptedInputPreset,
  sanitizeDismissReason,
  isHudHiddenQuery,
  isGregQuery,
  resolveDefaultGregMode,
  isSplashHoldQuery,
  SplashOverlayController,
  DEFAULT_FADE_MS,
  DEFAULT_FADE_OUT_MS,
  DEFAULT_VISIBLE_FRAME_FADE_MS,
  DEFAULT_HOLD_FALLBACK_TIMEOUT_MS,
  DEFAULT_ELAPSED_INTERVAL_MS,
  DEFAULT_LOADING_ASSETS_MESSAGE,
  DEFAULT_GREG_IMAGE_URL,
  DEFAULT_GREG_CLICK_THRESHOLD,
  GREG_SECRET_WORD,
  REDUCED_MOTION_QUERY,
} = splashMod;

class MockClassList {
  constructor(el) {
    this._el = el;
  }

  _tokens() {
    return new Set(
      String(this._el.className || '')
        .split(/\s+/)
        .filter(Boolean)
    );
  }

  _write(tokens) {
    this._el.className = Array.from(tokens).join(' ');
  }

  add(...names) {
    const tokens = this._tokens();
    for (const name of names) tokens.add(name);
    this._write(tokens);
  }

  remove(...names) {
    const tokens = this._tokens();
    for (const name of names) tokens.delete(name);
    this._write(tokens);
  }

  contains(name) {
    return this._tokens().has(name);
  }
}

class MockElement {
  constructor(tagName, ownerDocument) {
    this.tagName = String(tagName).toUpperCase();
    this.ownerDocument = ownerDocument;
    this.children = [];
    this.parentNode = null;
    this.attributes = new Map();
    this.style = {
      _props: Object.create(null),
      display: '',
      width: '',
      animation: '',
      setProperty(name, value) {
        this._props[name] = String(value);
      },
      getPropertyValue(name) {
        return this._props[name] || '';
      },
    };
    this._hidden = false;
    this._textContent = '';
    this._innerHTML = '';
    this.className = '';
    this.classList = new MockClassList(this);
    this._listeners = new Map();
    this.src = '';
    this.alt = '';
    this.onerror = null;
  }

  get hidden() {
    return this._hidden;
  }

  set hidden(value) {
    this._hidden = Boolean(value);
  }

  get textContent() {
    return this._textContent;
  }

  set textContent(value) {
    this._textContent = String(value ?? '');
  }

  get innerHTML() {
    return this._innerHTML;
  }

  set innerHTML(value) {
    this._innerHTML = String(value ?? '');
  }

  setAttribute(name, value) {
    this.attributes.set(String(name), String(value));
    if (name === 'src') this.src = String(value);
    if (name === 'class') this.className = String(value);
    if (name === 'hidden') this._hidden = true;
  }

  getAttribute(name) {
    if (!this.attributes.has(String(name))) return null;
    return this.attributes.get(String(name));
  }

  removeAttribute(name) {
    this.attributes.delete(String(name));
    if (name === 'src') this.src = '';
    if (name === 'hidden') this._hidden = false;
  }

  appendChild(child) {
    if (child.parentNode && child.parentNode !== this) {
      child.parentNode.removeChild(child);
    }
    child.parentNode = this;
    this.children.push(child);
    return child;
  }

  removeChild(child) {
    const idx = this.children.indexOf(child);
    if (idx >= 0) {
      this.children.splice(idx, 1);
      child.parentNode = null;
    }
    return child;
  }

  addEventListener(type, handler) {
    if (!this._listeners.has(type)) this._listeners.set(type, new Set());
    this._listeners.get(type).add(handler);
  }

  removeEventListener(type, handler) {
    const set = this._listeners.get(type);
    if (set) set.delete(handler);
  }

  listenerCount(type) {
    const set = this._listeners.get(type);
    return set ? set.size : 0;
  }

  dispatch(type, event = {}) {
    const set = this._listeners.get(type);
    if (set) {
      for (const handler of set) handler(event);
    }
    if (type === 'error' && typeof this.onerror === 'function') {
      this.onerror(event);
    }
  }

  querySelector(selector) {
    const match = (el) => {
      if (selector.startsWith('.')) {
        return el.classList.contains(selector.slice(1));
      }
      return false;
    };
    const walk = (nodes) => {
      for (const child of nodes) {
        if (match(child)) return child;
        const found = walk(child.children);
        if (found) return found;
      }
      return null;
    };
    return walk(this.children);
  }

  querySelectorAll(selector) {
    const out = [];
    const match = (el) => {
      if (selector.startsWith('.')) {
        return el.classList.contains(selector.slice(1));
      }
      return false;
    };
    const walk = (nodes) => {
      for (const child of nodes) {
        if (match(child)) out.push(child);
        walk(child.children);
      }
    };
    walk(this.children);
    return out;
  }
}

class MockDocument {
  constructor() {
    this.head = new MockElement('head', this);
    this.body = new MockElement('body', this);
    this.documentElement = new MockElement('html', this);
    this.documentElement.appendChild(this.head);
    this.documentElement.appendChild(this.body);
    this._styles = [];
    this._listeners = new Map();
  }

  addEventListener(type, handler) {
    if (!this._listeners.has(type)) this._listeners.set(type, new Set());
    this._listeners.get(type).add(handler);
  }

  removeEventListener(type, handler) {
    const set = this._listeners.get(type);
    if (set) set.delete(handler);
  }

  listenerCount(type) {
    const set = this._listeners.get(type);
    return set ? set.size : 0;
  }

  dispatch(type, event = {}) {
    const set = this._listeners.get(type);
    if (set) {
      for (const handler of Array.from(set)) handler(event);
    }
  }

  createElement(tag) {
    return new MockElement(tag, this);
  }

  querySelector(selector) {
    if (selector.includes('data-pg6-splash-style')) {
      return this._styles[0] || null;
    }
    return this.body.querySelector(selector) || this.head.querySelector(selector);
  }

  getElementsByTagName(tag) {
    const upper = String(tag).toUpperCase();
    if (upper === 'HEAD') return [this.head];
    if (upper === 'BODY') return [this.body];
    return [];
  }
}

function createTimer() {
  let nextId = 1;
  const pending = new Map();
  return {
    setTimeout(fn, ms) {
      const id = nextId++;
      pending.set(id, { fn, ms: Number(ms) || 0 });
      return id;
    },
    clearTimeout(id) {
      pending.delete(id);
    },
    flush() {
      const entries = Array.from(pending.entries());
      pending.clear();
      for (const [, entry] of entries) entry.fn();
    },
    delays() {
      return Array.from(pending.values()).map((entry) => entry.ms);
    },
    get size() {
      return pending.size;
    },
  };
}

function createController(overrides = {}) {
  const { timer: timerOverride, documentObj: docOverride, onDismiss, ...rest } = overrides;
  const documentObj = docOverride || new MockDocument();
  const timer = timerOverride || createTimer();
  const dismissCalls = [];
  const controller = new SplashOverlayController({
    splashImageUrl: null,
    fadeMs: 50,
    parent: documentObj.body,
    setTimeout: timer.setTimeout.bind(timer),
    clearTimeout: timer.clearTimeout.bind(timer),
    onDismiss:
      typeof onDismiss === 'function'
        ? (...args) => {
            dismissCalls.push(1);
            onDismiss(...args);
          }
        : () => dismissCalls.push(1),
    ...rest,
    documentObj,
  });
  return { controller, documentObj, timer, dismissCalls };
}

test('exports BOOT_STAGE_ORDER, labels, helpers, and UMD global', () => {
  assert.ok(Object.isFrozen(BOOT_STAGE_ORDER));
  assert.deepEqual(BOOT_STAGE_ORDER, [
    'PGIM_LOAD',
    'TLS_INIT',
    'CRT_INITTERM',
    'ENTRY_CALL',
    'XMEM_INIT',
    'GDK_INIT',
    'D3D12_CREATE_DEVICE',
    'BINK_OPEN',
    'FIRST_PRESENT',
  ]);
  assert.equal(BOOT_STAGE_LABELS.PGIM_LOAD, 'Loading guest image');
  assert.equal(BOOT_STAGE_LABELS.FIRST_PRESENT, 'First frame presented');
  assert.equal(typeof computeBootProgress, 'function');
  assert.equal(typeof SplashOverlayController, 'function');
  assert.equal(DEFAULT_FADE_MS, 400);
  assert.equal(globalThis.PlayGta6SplashOverlay.SplashOverlayController, SplashOverlayController);
  assert.equal(globalThis.PlayGta6SplashOverlay.computeBootProgress, computeBootProgress);
});

test('computeBootProgress prefers progressPct then stage then milestonesReached', () => {
  assert.equal(computeBootProgress(null), 0);
  assert.equal(computeBootProgress({}), 0);
  assert.equal(computeBootProgress({ progressPct: 42 }), 42);
  assert.equal(computeBootProgress({ progressPct: 150 }), 100);
  assert.equal(computeBootProgress({ progressPct: -5 }), 0);
  assert.equal(computeBootProgress({ stage: 'PGIM_LOAD' }), Math.round((1 / 9) * 100));
  assert.equal(computeBootProgress({ stage: 'FIRST_PRESENT' }), 100);
  assert.equal(computeBootProgress({ milestonesReached: 3 }), Math.round((3 / 9) * 100));
  assert.equal(computeBootProgress({ progressPct: 10, stage: 'FIRST_PRESENT' }), 10);
});

test('constructor validates options and mounts accessible root', () => {
  assert.throws(() => new SplashOverlayController(null), TypeError);
  assert.throws(() => new SplashOverlayController({ documentObj: {} }), TypeError);
  assert.throws(
    () => new SplashOverlayController({ documentObj: new MockDocument(), onDismiss: 1 }),
    TypeError
  );

  const { controller, documentObj } = createController();
  const state = controller.getState();
  assert.equal(state.visible, true);
  assert.equal(state.destroyed, false);
  assert.equal(state.gregMode, true);
  assert.equal(state.gregBackdropActive, true);
  assert.equal(state.activeSplashImageUrl, './assets/greg.png');
  assert.equal(state.usingFallback, false);
  assert.ok(state.root);
  assert.equal(state.root.parentNode, documentObj.body);
  assert.equal(state.root.getAttribute('role'), 'status');
  assert.equal(state.root.getAttribute('aria-live'), 'polite');
  assert.equal(state.root.getAttribute('aria-busy'), 'true');
  controller.destroy();
});

test('binds splashImageUrl to img src when provided and Greg mode is disabled', () => {
  const { controller } = createController({
    splashImageUrl: '/game/xbs_splash.png',
    gregMode: false,
  });
  const state = controller.getState();
  assert.equal(state.splashImageUrl, '/game/xbs_splash.png');
  assert.equal(state.activeSplashImageUrl, '/game/xbs_splash.png');
  assert.equal(state.gregBackdropActive, false);
  assert.equal(state.usingFallback, false);
  const img = state.root.querySelector('.pg6-splash__img');
  assert.ok(img);
  assert.equal(img.getAttribute('src'), '/game/xbs_splash.png');
  assert.equal(img.style.objectFit, 'cover');
  assert.equal(img.classList.contains('pg6-splash__img--greg'), false);
  assert.equal(img.hidden, false);
  const fallback = state.root.querySelector('.pg6-splash__fallback');
  assert.equal(fallback.hidden, true);
  controller.destroy();
});

test('image onerror falls back to built-in CSS/SVG without throwing', () => {
  const { controller } = createController({
    splashImageUrl: '/game/missing-splash.png',
    gregMode: false,
  });
  const img = controller.getState().root.querySelector('.pg6-splash__img');
  assert.equal(controller.getState().usingFallback, false);

  assert.doesNotThrow(() => img.dispatch('error'));

  const state = controller.getState();
  assert.equal(state.imageFailed, true);
  assert.equal(state.usingFallback, true);
  assert.equal(img.hidden, true);
  const fallback = state.root.querySelector('.pg6-splash__fallback');
  assert.equal(fallback.hidden, false);
  assert.match(fallback.innerHTML, /playgta6web/);
  controller.destroy();
});

test('null splashImageUrl uses fallback immediately when Greg mode is disabled', () => {
  const { controller } = createController({ splashImageUrl: null, gregMode: false });
  const state = controller.getState();
  assert.equal(state.usingFallback, true);
  assert.equal(state.gregBackdropActive, false);
  const img = state.root.querySelector('.pg6-splash__img');
  assert.equal(img.hidden, true);
  const fallback = state.root.querySelector('.pg6-splash__fallback');
  assert.equal(fallback.hidden, false);
  controller.destroy();
});

test('updateBootState integrates BootTracer-style snapshots', () => {
  const { controller } = createController();
  controller.updateBootState({
    stage: 'D3D12_CREATE_DEVICE',
    milestonesReached: 7,
    progressPct: 78,
    elapsedMs: 1250,
    statusText: 'Creating D3D12 device',
    readyForSplash: false,
    streamingProgress: { loadedBytes: 50, totalBytes: 100, pct: 50 },
    unresolvedImports: [],
    traps: [],
  });

  const state = controller.getState();
  assert.equal(state.stage, 'D3D12_CREATE_DEVICE');
  assert.equal(state.progressPct, 78);
  assert.equal(state.streamPct, 50);
  assert.equal(state.elapsedMs, 1250);
  assert.equal(state.statusText, 'Creating D3D12 device');
  assert.equal(state.readyForSplash, false);

  const root = state.root;
  assert.equal(root.querySelector('.pg6-splash__stage').textContent, 'Creating D3D12 device');
  assert.equal(root.querySelector('.pg6-splash__bar'), null);
  assert.equal(root.querySelector('.pg6-splash__bar-fill'), null);
  assert.equal(root.querySelector('.pg6-splash__pct'), null);
  assert.equal(root.querySelector('.pg6-splash__elapsed-label').textContent, 'Elapsed');
  assert.equal(root.querySelector('.pg6-splash__elapsed').textContent, '00:01');
  assert.equal(root.querySelector('.pg6-splash__stream').hidden, false);
  assert.equal(root.querySelector('.pg6-splash__stream-fill').style.width, '50%');
  controller.destroy();
});

test('updateBootState derives progress from stage when progressPct omitted', () => {
  const { controller } = createController();
  controller.updateBootState({ stage: 'TLS_INIT', elapsedMs: 40 });
  const state = controller.getState();
  assert.equal(state.progressPct, Math.round((2 / 9) * 100));
  assert.equal(state.stageLabel, 'Initializing TLS');
  controller.destroy();
});

test('trap badge renders unresolved imports and traps', () => {
  const { controller } = createController();
  controller.updateBootState({
    stage: 'ENTRY_CALL',
    progressPct: 44,
    unresolvedImports: [
      { name: 'kernelx.dll!XMemAlloc', dll: 'kernelx.dll', symbol: 'XMemAlloc', count: 3 },
      { name: 'dxgi.dll!CreateDXGIFactory', count: 1 },
    ],
    traps: [{ kind: 'UNRESOLVED_IMPORT', message: 'missing' }, { kind: 'WASM_TRAP' }],
  });

  const state = controller.getState();
  assert.match(state.badgeText, /Unresolved imports \(2\)/);
  assert.match(state.badgeText, /kernelx\.dll!XMemAlloc/);
  assert.match(state.badgeText, /Traps: 2/);
  assert.match(state.badgeText, /WASM_TRAP/);

  const badge = state.root.querySelector('.pg6-splash__badge');
  assert.equal(badge.hidden, false);
  assert.equal(badge.textContent, state.badgeText);

  controller.updateBootState({ stage: 'ENTRY_CALL', progressPct: 44, unresolvedImports: [], traps: [] });
  assert.equal(controller.getState().badgeText, '');
  assert.equal(badge.hidden, true);
  controller.destroy();
});

test('accepts BootTracer recentTraps alias for traps', () => {
  const { controller } = createController();
  controller.updateBootState({
    stage: 'CRT_INITTERM',
    progressPct: 33,
    recentTraps: [{ kind: 'INDIRECT_CALL_MISS' }],
    unresolvedImports: ['foo.dll!Bar'],
  });
  const text = controller.getState().badgeText;
  assert.match(text, /foo\.dll!Bar/);
  assert.match(text, /INDIRECT_CALL_MISS/);
  controller.destroy();
});

test('onFirstPresent fades then dismisses and fires onDismiss once', () => {
  const { controller, timer, dismissCalls } = createController({ fadeMs: 25 });
  const state1 = controller.onFirstPresent();
  assert.equal(state1.dismissed, true);
  assert.equal(state1.fading, true);
  assert.equal(state1.progressPct, 100);
  assert.equal(state1.stage, 'FIRST_PRESENT');
  assert.equal(state1.visible, true);
  assert.ok(state1.root.classList.contains('pg6-splash--fading'));
  assert.equal(dismissCalls.length, 0);

  timer.flush();
  const state2 = controller.getState();
  assert.equal(state2.visible, false);
  assert.equal(state2.fading, false);
  assert.equal(state2.root.hidden, true);
  assert.equal(dismissCalls.length, 1);

  // Idempotent dismiss / onFirstPresent
  controller.onFirstPresent();
  controller.dismiss({ fadeMs: 10 });
  assert.equal(dismissCalls.length, 1);
  controller.destroy();
});

test('dismiss with fadeMs 0 hides immediately', () => {
  const { controller, dismissCalls, timer } = createController();
  controller.dismiss({ fadeMs: 0 });
  assert.equal(controller.getState().visible, false);
  assert.equal(controller.getState().dismissed, true);
  assert.equal(controller.getState().fading, false);
  assert.equal(dismissCalls.length, 1);
  assert.equal(timer.size, 0);
  controller.destroy();
});

test('manual show and hide toggle visibility without dismissing', () => {
  const { controller, dismissCalls } = createController();
  controller.hide();
  assert.equal(controller.getState().visible, false);
  assert.equal(controller.getState().dismissed, false);
  assert.equal(dismissCalls.length, 0);

  controller.show();
  assert.equal(controller.getState().visible, true);
  assert.equal(controller.getState().root.hidden, false);
  assert.equal(dismissCalls.length, 0);
  controller.destroy();
});

test('ARIA attributes stay present across boot updates', () => {
  const { controller } = createController();
  const root = controller.getState().root;
  assert.equal(root.getAttribute('role'), 'status');
  assert.equal(root.getAttribute('aria-live'), 'polite');
  assert.equal(root.getAttribute('aria-atomic'), 'true');

  controller.updateBootState({
    stage: 'BINK_OPEN',
    progressPct: 89,
    statusText: 'Opening intro stream',
  });
  assert.equal(root.getAttribute('aria-label'), 'Opening intro stream');
  assert.equal(root.getAttribute('aria-busy'), 'true');

  controller.onFirstPresent({ fadeMs: 0 });
  assert.equal(root.getAttribute('aria-busy'), 'false');
  controller.destroy();
});

test('destroy is idempotent and detaches the overlay', () => {
  const { controller, documentObj } = createController({
    splashImageUrl: '/game/xbs_splash.png',
  });
  const root = controller.getState().root;
  assert.equal(root.parentNode, documentObj.body);

  controller.destroy();
  assert.equal(controller.getState().destroyed, true);
  assert.equal(controller.getState().visible, false);
  assert.equal(root.parentNode, null);

  assert.doesNotThrow(() => controller.destroy());
  assert.doesNotThrow(() => controller.updateBootState({ stage: 'PGIM_LOAD' }));
  assert.doesNotThrow(() => controller.show());
  assert.doesNotThrow(() => controller.hide());
  assert.doesNotThrow(() => controller.dismiss());
  assert.equal(controller.getState().destroyed, true);
});

test('readyForSplash in snapshot marks stage ready without auto-dismiss', () => {
  const { controller, dismissCalls } = createController();
  controller.updateBootState({
    stage: 'FIRST_PRESENT',
    progressPct: 100,
    readyForSplash: true,
  });
  const state = controller.getState();
  assert.equal(state.readyForSplash, true);
  assert.equal(state.dismissed, false);
  assert.equal(state.visible, true);
  assert.equal(dismissCalls.length, 0);
  controller.destroy();
});

test('streamingProgress accepts numeric pct and hides stream when absent', () => {
  const { controller } = createController();
  controller.updateBootState({ stage: 'PGIM_LOAD', progressPct: 11, streamingProgress: 64 });
  assert.equal(controller.getState().streamPct, 64);
  assert.equal(controller.getState().root.querySelector('.pg6-splash__stream').hidden, false);

  controller.updateBootState({ stage: 'PGIM_LOAD', progressPct: 11 });
  assert.equal(controller.getState().streamPct, null);
  assert.equal(controller.getState().root.querySelector('.pg6-splash__stream').hidden, true);
  controller.destroy();
});

test('W151: exports sanitizeScriptedInputPreset and REDUCED_MOTION_QUERY on module and UMD global', () => {
  assert.equal(typeof sanitizeScriptedInputPreset, 'function');
  assert.equal(REDUCED_MOTION_QUERY, '(prefers-reduced-motion: reduce)');
  assert.equal(
    globalThis.PlayGta6SplashOverlay.sanitizeScriptedInputPreset,
    sanitizeScriptedInputPreset
  );
  assert.equal(globalThis.PlayGta6SplashOverlay.REDUCED_MOTION_QUERY, REDUCED_MOTION_QUERY);
});

test('W151: validates nowFn, reducedMotion, and matchMedia constructor options', () => {
  assert.throws(
    () => new SplashOverlayController({ documentObj: new MockDocument(), nowFn: 123 }),
    TypeError
  );
  assert.throws(
    () => new SplashOverlayController({ documentObj: new MockDocument(), reducedMotion: 'yes' }),
    TypeError
  );
  assert.throws(
    () => new SplashOverlayController({ documentObj: new MockDocument(), matchMedia: {} }),
    TypeError
  );
  assert.doesNotThrow(() => {
    const c = new SplashOverlayController({
      documentObj: new MockDocument(),
      nowFn: null,
      reducedMotion: null,
      matchMedia: null,
    });
    c.destroy();
  });
});

test('W151: tracks FIRST_PRESENT dismissal telemetry via getStats() and getState() using injectable nowFn', () => {
  let now = 1000;
  let statsInsideOnDismiss = null;
  let controllerRef = null;

  const { controller, timer, dismissCalls } = createController({
    fadeMs: 40,
    nowFn: () => now,
    onDismiss: () => {
      statsInsideOnDismiss = controllerRef.getStats();
    },
  });
  controllerRef = controller;

  const initialStats = controller.getStats();
  assert.equal(initialStats.visible, true);
  assert.equal(initialStats.dismissed, false);
  assert.equal(initialStats.fading, false);
  assert.equal(initialStats.destroyed, false);
  assert.equal(initialStats.dismissedAtMs, null);
  assert.equal(initialStats.dismissCount, 0);
  assert.equal(initialStats.fadeDurationMs, null);
  assert.equal(initialStats.fadeMs, 40);
  assert.equal(initialStats.reducedMotion, false);
  assert.equal(initialStats.scriptedInputPreset, null);
  assert.equal('root' in initialStats, false);
  assert.doesNotThrow(() => JSON.stringify(initialStats));

  // Trigger FIRST_PRESENT fade at t=1000.
  controller.onFirstPresent();
  const fadingStats = controller.getStats();
  assert.equal(fadingStats.visible, true);
  assert.equal(fadingStats.dismissed, true);
  assert.equal(fadingStats.fading, true);
  assert.equal(fadingStats.dismissCount, 0);
  assert.equal(fadingStats.dismissedAtMs, null);
  assert.equal(fadingStats.fadeDurationMs, null);
  assert.equal(timer.size, 1);

  // Advance clock to t=1045 and flush fade timer.
  now = 1045;
  timer.flush();

  assert.equal(dismissCalls.length, 1);
  assert.ok(statsInsideOnDismiss);
  assert.equal(statsInsideOnDismiss.visible, false);
  assert.equal(statsInsideOnDismiss.dismissCount, 1);
  assert.equal(statsInsideOnDismiss.dismissedAtMs, 1045);
  assert.equal(statsInsideOnDismiss.fadeDurationMs, 45);

  const afterStats = controller.getStats();
  const afterState = controller.getState();
  assert.equal(afterStats.visible, false);
  assert.equal(afterStats.dismissed, true);
  assert.equal(afterStats.fading, false);
  assert.equal(afterStats.dismissCount, 1);
  assert.equal(afterStats.dismissedAtMs, 1045);
  assert.equal(afterStats.fadeDurationMs, 45);
  assert.equal(afterState.dismissCount, 1);
  assert.equal(afterState.dismissedAtMs, 1045);
  assert.equal(afterState.fadeDurationMs, 45);

  // Repeated idempotent dismiss does not increment count or overwrite timestamp.
  now = 1500;
  controller.onFirstPresent();
  controller.dismiss({ fadeMs: 0 });
  assert.equal(controller.getStats().dismissCount, 1);
  assert.equal(controller.getStats().dismissedAtMs, 1045);
  assert.equal(controller.getStats().fadeDurationMs, 45);

  // Manual show() resets visibility without clearing historical telemetry; hide() does not count as dismiss.
  controller.show();
  assert.equal(controller.getStats().visible, true);
  assert.equal(controller.getStats().dismissed, false);
  assert.equal(controller.getStats().dismissCount, 1);

  controller.hide();
  assert.equal(controller.getStats().visible, false);
  assert.equal(controller.getStats().dismissCount, 1);

  // Re-show and perform a second immediate dismissal at t=2500.
  controller.show();
  now = 2500;
  controller.onFirstPresent({ fadeMs: 0 });
  const secondStats = controller.getStats();
  assert.equal(secondStats.visible, false);
  assert.equal(secondStats.dismissCount, 2);
  assert.equal(secondStats.dismissedAtMs, 2500);
  assert.equal(secondStats.fadeDurationMs, 0);
  assert.equal(dismissCalls.length, 2);

  controller.destroy();
  const destroyedStats = controller.getStats();
  assert.equal(destroyedStats.destroyed, true);
  assert.equal(destroyedStats.visible, false);
  assert.equal(destroyedStats.dismissCount, 2);
  assert.equal(destroyedStats.dismissedAtMs, 2500);
});

test('W151: sanitizes telemetry when nowFn is static, negative, NaN, or throws, and aborts cleanly on destroy', () => {
  // Static clock during timed fade falls back to effective fadeMs.
  const staticCase = createController({ fadeMs: 60, nowFn: () => 500 });
  staticCase.controller.onFirstPresent();
  staticCase.timer.flush();
  assert.equal(staticCase.controller.getStats().dismissedAtMs, 500);
  assert.equal(staticCase.controller.getStats().fadeDurationMs, 60);
  staticCase.controller.destroy();

  // Throwing / invalid nowFn values are sanitized to 0 and never throw.
  const faultyClocks = [
    () => {
      throw new Error('clock failure');
    },
    () => Number.NaN,
    () => -100,
    () => Number.POSITIVE_INFINITY,
    () => 'not-a-number',
  ];
  for (const nowFn of faultyClocks) {
    const { controller, timer } = createController({ fadeMs: 30, nowFn });
    assert.doesNotThrow(() => controller.onFirstPresent());
    assert.doesNotThrow(() => timer.flush());
    const stats = controller.getStats();
    assert.equal(stats.dismissedAtMs, 0);
    assert.equal(stats.fadeDurationMs, 30);
    assert.equal(stats.dismissCount, 1);
    controller.destroy();
  }

  // Destroy mid-fade cancels dismissal without incrementing dismissCount.
  const midFade = createController({ fadeMs: 50, nowFn: () => 900 });
  midFade.controller.onFirstPresent();
  assert.equal(midFade.timer.size, 1);
  midFade.controller.destroy();
  assert.equal(midFade.timer.size, 0);
  midFade.timer.flush();
  assert.equal(midFade.dismissCalls.length, 0);
  assert.equal(midFade.controller.getStats().dismissCount, 0);
  assert.equal(midFade.controller.getStats().dismissedAtMs, null);
  assert.equal(midFade.controller.getStats().destroyed, true);
});

test('W151: prefers-reduced-motion option and matchMedia detection dismiss cleanly without transition delay', () => {
  // 1) Explicit reducedMotion: true option skips fade timer completely.
  let now = 3000;
  const explicit = createController({
    fadeMs: 400,
    reducedMotion: true,
    nowFn: () => now,
  });
  assert.equal(explicit.controller.getStats().reducedMotion, true);
  assert.equal(
    explicit.controller.getState().root.style.getPropertyValue('--pg6-splash-fade'),
    '0ms'
  );
  explicit.controller.onFirstPresent();
  assert.equal(explicit.timer.size, 0);
  assert.equal(explicit.dismissCalls.length, 1);
  assert.equal(explicit.controller.getStats().visible, false);
  assert.equal(explicit.controller.getStats().fading, false);
  assert.equal(explicit.controller.getStats().dismissCount, 1);
  assert.equal(explicit.controller.getStats().dismissedAtMs, 3000);
  assert.equal(explicit.controller.getStats().fadeDurationMs, 0);
  explicit.controller.destroy();

  // 2) Injected matchMedia returning matches: true triggers immediate dismissal.
  const seenQueries = [];
  const viaOptionMedia = createController({
    fadeMs: 250,
    nowFn: () => 4100,
    matchMedia: (query) => {
      seenQueries.push(query);
      return { matches: query === REDUCED_MOTION_QUERY };
    },
  });
  assert.equal(viaOptionMedia.controller.getStats().reducedMotion, true);
  viaOptionMedia.controller.dismiss();
  assert.equal(viaOptionMedia.timer.size, 0);
  assert.equal(viaOptionMedia.controller.getStats().visible, false);
  assert.equal(viaOptionMedia.controller.getStats().fadeDurationMs, 0);
  assert.equal(viaOptionMedia.controller.getStats().dismissedAtMs, 4100);
  assert.ok(seenQueries.includes(REDUCED_MOTION_QUERY));
  viaOptionMedia.controller.destroy();

  // 3) documentObj.defaultView.matchMedia detection works automatically.
  const docWithView = new MockDocument();
  docWithView.defaultView = {
    matchMedia: (query) => ({ matches: query === REDUCED_MOTION_QUERY }),
  };
  const viaView = createController({
    documentObj: docWithView,
    fadeMs: 300,
    nowFn: () => 5200,
  });
  assert.equal(viaView.controller.getStats().reducedMotion, true);
  viaView.controller.onFirstPresent();
  assert.equal(viaView.timer.size, 0);
  assert.equal(viaView.controller.getStats().visible, false);
  assert.equal(viaView.controller.getStats().fadeDurationMs, 0);
  viaView.controller.destroy();

  // 4) Per-call reducedMotion override on dismiss() works in both directions.
  const perCall = createController({ fadeMs: 100, reducedMotion: false, nowFn: () => 6000 });
  perCall.controller.dismiss({ reducedMotion: true });
  assert.equal(perCall.timer.size, 0);
  assert.equal(perCall.controller.getStats().visible, false);
  assert.equal(perCall.controller.getStats().fadeDurationMs, 0);
  perCall.controller.show();
  perCall.controller.dismiss({ reducedMotion: false, fadeMs: 35 });
  assert.equal(perCall.timer.size, 1);
  assert.equal(perCall.controller.getStats().fading, true);
  perCall.timer.flush();
  assert.equal(perCall.controller.getStats().fadeDurationMs, 35);
  perCall.controller.destroy();

  // 5) Faulty matchMedia falls back safely to false and never throws.
  const faultyMedia = createController({
    fadeMs: 25,
    matchMedia: () => {
      throw new Error('matchMedia unavailable');
    },
  });
  assert.equal(faultyMedia.controller.getStats().reducedMotion, false);
  assert.doesNotThrow(() => faultyMedia.controller.onFirstPresent());
  assert.equal(faultyMedia.timer.size, 1);
  faultyMedia.timer.flush();
  assert.equal(faultyMedia.controller.getStats().visible, false);
  faultyMedia.controller.destroy();

  // 6) Injected shared stylesheet contains prefers-reduced-motion media query.
  const styleDoc = new MockDocument();
  const styled = createController({ documentObj: styleDoc });
  const styleEl = styleDoc.head.children.find((el) => el.getAttribute('data-pg6-splash-style') === '1');
  assert.ok(styleEl);
  assert.match(styleEl.textContent, /@media \(prefers-reduced-motion: reduce\)/);
  styled.controller.destroy();
});

test('W151: setScriptedInput renders sanitized indicator pill, hides on null, and rejects malicious inputs', () => {
  // Direct allowlist verification.
  assert.equal(sanitizeScriptedInputPreset('start-screen'), 'start-screen');
  assert.equal(sanitizeScriptedInputPreset('  menu_nav-01  '), 'menu_nav-01');
  assert.equal(sanitizeScriptedInputPreset('A'), 'A');
  assert.equal(sanitizeScriptedInputPreset('a'.repeat(32)), 'a'.repeat(32));
  assert.equal(sanitizeScriptedInputPreset('a'.repeat(33)), null);
  assert.equal(sanitizeScriptedInputPreset(''), null);
  assert.equal(sanitizeScriptedInputPreset('   '), null);
  assert.equal(sanitizeScriptedInputPreset(null), null);
  assert.equal(sanitizeScriptedInputPreset(undefined), null);
  assert.equal(sanitizeScriptedInputPreset(42), null);
  assert.equal(sanitizeScriptedInputPreset({ preset: 'start-screen' }), null);
  assert.equal(sanitizeScriptedInputPreset('-leading-dash'), null);
  assert.equal(sanitizeScriptedInputPreset('_leading_underscore'), null);
  assert.equal(sanitizeScriptedInputPreset('start screen'), null);
  assert.equal(sanitizeScriptedInputPreset('../etc/passwd'), null);
  assert.equal(sanitizeScriptedInputPreset('C:\\Users\\alice\\script.json'), null);
  assert.equal(sanitizeScriptedInputPreset('/home/alice/private/input.json'), null);
  assert.equal(sanitizeScriptedInputPreset('file:///Users/bob/script'), null);
  assert.equal(sanitizeScriptedInputPreset('start-screen?token=secret'), null);
  assert.equal(sanitizeScriptedInputPreset('<script>alert(1)</script>'), null);
  assert.equal(sanitizeScriptedInputPreset('start\nscreen'), null);

  const { controller } = createController();
  const pill = controller.getState().root.querySelector('.pg6-splash__scripted');
  assert.ok(pill);

  // Initially inactive and hidden.
  assert.equal(controller.getState().scriptedInputPreset, null);
  assert.equal(controller.getStats().scriptedInputPreset, null);
  assert.equal(pill.hidden, true);
  assert.equal(pill.style.display, 'none');
  assert.equal(pill.getAttribute('aria-hidden'), 'true');
  assert.equal(pill.getAttribute('data-active'), 'false');
  assert.equal(pill.getAttribute('data-preset'), null);
  assert.equal(pill.textContent, '');

  // Valid preset shows pill with clean text and attributes.
  const state1 = controller.setScriptedInput('  start-screen  ');
  assert.equal(state1.scriptedInputPreset, 'start-screen');
  assert.equal(controller.getStats().scriptedInputPreset, 'start-screen');
  assert.equal(pill.hidden, false);
  assert.equal(pill.style.display, '');
  assert.equal(pill.getAttribute('aria-hidden'), 'false');
  assert.equal(pill.getAttribute('data-active'), 'true');
  assert.equal(pill.getAttribute('data-preset'), 'start-screen');
  assert.equal(pill.textContent, 'Scripted input: start-screen');

  // Clearing with null hides pill and removes preset metadata.
  const state2 = controller.setScriptedInput(null);
  assert.equal(state2.scriptedInputPreset, null);
  assert.equal(controller.getStats().scriptedInputPreset, null);
  assert.equal(pill.hidden, true);
  assert.equal(pill.style.display, 'none');
  assert.equal(pill.getAttribute('aria-hidden'), 'true');
  assert.equal(pill.getAttribute('data-active'), 'false');
  assert.equal(pill.getAttribute('data-preset'), null);
  assert.equal(pill.textContent, '');

  // Malicious inputs clear active preset and never leak into DOM or telemetry.
  const maliciousInputs = [
    'C:\\Users\\secretuser\\GTA6\\macros.json',
    '../secret/preset',
    '<img src=x onerror=alert(1)>',
    'preset\r\ninjection',
    'preset with spaces',
    'x'.repeat(64),
  ];
  for (const bad of maliciousInputs) {
    controller.setScriptedInput('demo_preset');
    assert.equal(controller.getStats().scriptedInputPreset, 'demo_preset');
    controller.setScriptedInput(bad);
    assert.equal(controller.getStats().scriptedInputPreset, null);
    assert.equal(controller.getState().scriptedInputPreset, null);
    assert.equal(pill.hidden, true);
    assert.equal(pill.textContent, '');
    assert.equal(pill.getAttribute('data-preset'), null);
    assert.equal(JSON.stringify(controller.getStats()).includes(bad), false);
  }

  // Initial constructor scriptedInput option works and sanitizes.
  const withInitial = createController({ scriptedInput: 'boot_menu-1' });
  assert.equal(withInitial.controller.getStats().scriptedInputPreset, 'boot_menu-1');
  const initPill = withInitial.controller.getState().root.querySelector('.pg6-splash__scripted');
  assert.equal(initPill.hidden, false);
  assert.equal(initPill.textContent, 'Scripted input: boot_menu-1');
  withInitial.controller.destroy();

  // Safe no-op after destroy.
  controller.destroy();
  assert.doesNotThrow(() => controller.setScriptedInput('start-screen'));
  assert.equal(controller.getStats().destroyed, true);
});

test('W213: exports DISMISS_REASONS, sanitizeDismissReason, and isHudHiddenQuery on module and UMD global', () => {
  assert.ok(Object.isFrozen(DISMISS_REASONS));
  assert.equal(DISMISS_REASONS.FIRST_PRESENT, 'first_present');
  assert.equal(DISMISS_REASONS.IMMEDIATE, 'immediate');
  assert.equal(DISMISS_REASONS.HUD_HIDDEN, 'hud_hidden');
  assert.equal(DISMISS_REASONS.INSTANT_HIDE, 'instant_hide');
  assert.equal(DISMISS_REASONS.REDUCED_MOTION, 'reduced_motion');
  assert.equal(DISMISS_REASONS.MANUAL, 'manual');
  assert.equal(typeof sanitizeDismissReason, 'function');
  assert.equal(typeof isHudHiddenQuery, 'function');
  assert.equal(globalThis.PlayGta6SplashOverlay.DISMISS_REASONS, DISMISS_REASONS);
  assert.equal(globalThis.PlayGta6SplashOverlay.sanitizeDismissReason, sanitizeDismissReason);
  assert.equal(globalThis.PlayGta6SplashOverlay.isHudHiddenQuery, isHudHiddenQuery);

  // sanitizeDismissReason allowlist & injection protection.
  assert.equal(sanitizeDismissReason('first_present'), 'first_present');
  assert.equal(sanitizeDismissReason('  bink.frame:0-ready  '), 'bink.frame:0-ready');
  assert.equal(sanitizeDismissReason(''), null);
  assert.equal(sanitizeDismissReason(null), null);
  assert.equal(sanitizeDismissReason(42), null);
  assert.equal(sanitizeDismissReason('../etc/passwd'), null);
  assert.equal(sanitizeDismissReason('C:\\Users\\secret\\file'), null);
  assert.equal(sanitizeDismissReason('<script>alert(1)</script>'), null);
  assert.equal(sanitizeDismissReason('reason with spaces'), null);
  assert.equal(sanitizeDismissReason('reason\ninjection'), null);
  assert.equal(sanitizeDismissReason('a'.repeat(65)), null);

  // isHudHiddenQuery parsing across string, URLSearchParams, Location, and full URLs.
  assert.equal(isHudHiddenQuery('?hud=0'), true);
  assert.equal(isHudHiddenQuery('hud=0'), true);
  assert.equal(isHudHiddenQuery('?HUD=0'), true);
  assert.equal(isHudHiddenQuery('?foo=1&hud=0&bar=2'), true);
  assert.equal(isHudHiddenQuery('?hud=false'), true);
  assert.equal(isHudHiddenQuery('?hud=off'), true);
  assert.equal(isHudHiddenQuery('?hud=no'), true);
  assert.equal(isHudHiddenQuery('?hud=1&hud=0'), true);
  assert.equal(isHudHiddenQuery('?hud=0&hud=1'), false);
  assert.equal(isHudHiddenQuery('?hud=1'), false);
  assert.equal(isHudHiddenQuery('?hud=true'), false);
  assert.equal(isHudHiddenQuery('?hud='), false);
  assert.equal(isHudHiddenQuery(''), false);
  assert.equal(isHudHiddenQuery(null), false);
  assert.equal(isHudHiddenQuery(undefined), false);
  assert.equal(isHudHiddenQuery(new URLSearchParams('?hud=0')), true);
  assert.equal(isHudHiddenQuery(new URLSearchParams('?hud=1')), false);
  assert.equal(isHudHiddenQuery({ search: '?hud=0' }), true);
  assert.equal(isHudHiddenQuery('http://localhost:8000/?hud=0#top'), true);
  assert.equal(isHudHiddenQuery('http://localhost:8000/?hud=1#?hud=0'), false);
});

test('W213: validates instantHide and windowObj constructor options', () => {
  assert.throws(
    () => new SplashOverlayController({ documentObj: new MockDocument(), instantHide: 'yes' }),
    TypeError
  );
  assert.throws(
    () => new SplashOverlayController({ documentObj: new MockDocument(), windowObj: 'not-an-obj' }),
    TypeError
  );
  assert.doesNotThrow(() => {
    const c = new SplashOverlayController({
      documentObj: new MockDocument(),
      instantHide: null,
      windowObj: null,
    });
    c.destroy();
  });
});

test('W213: dismissImmediate and instant-hide options dismiss synchronously vs animated dismiss', () => {
  // 1) dismissImmediate() bypasses CSS fade and timer completely.
  let now = 1200;
  const imm = createController({ fadeMs: 400, nowFn: () => now });
  const immState = imm.controller.dismissImmediate();
  assert.equal(immState.visible, false);
  assert.equal(immState.dismissed, true);
  assert.equal(immState.fading, false);
  assert.equal(immState.root.hidden, true);
  assert.equal(immState.root.style.display, 'none');
  assert.equal(immState.root.style.getPropertyValue('--pg6-splash-fade'), '0ms');
  assert.equal(immState.root.classList.contains('pg6-splash--fading'), false);
  assert.equal(immState.root.getAttribute('aria-busy'), 'false');
  assert.equal(imm.timer.size, 0);
  assert.equal(imm.dismissCalls.length, 1);

  const immStats = imm.controller.getStats();
  assert.equal(immStats.visible, false);
  assert.equal(immStats.dismissedAtMs, 1200);
  assert.equal(immStats.dismissReason, DISMISS_REASONS.IMMEDIATE);
  assert.equal(immStats.fadeDurationMs, 0);
  assert.equal(immStats.dismissCount, 1);

  // Idempotent second dismissImmediate does not double-fire or mutate telemetry.
  now = 1900;
  imm.controller.dismissImmediate('later_reason');
  assert.equal(imm.dismissCalls.length, 1);
  assert.equal(imm.controller.getStats().dismissedAtMs, 1200);
  assert.equal(imm.controller.getStats().dismissReason, DISMISS_REASONS.IMMEDIATE);
  assert.equal(imm.controller.getStats().dismissCount, 1);
  imm.controller.destroy();

  // 2) dismissImmediate accepts custom reason string or object and sanitizes malicious values.
  const custom = createController({ fadeMs: 400, nowFn: () => 2200 });
  custom.controller.dismissImmediate('bink2_first_frame');
  assert.equal(custom.controller.getStats().dismissReason, 'bink2_first_frame');
  custom.controller.show();
  custom.controller.dismissImmediate({ reason: '../bad/path' });
  assert.equal(custom.controller.getStats().dismissReason, DISMISS_REASONS.IMMEDIATE);
  custom.controller.destroy();

  // 3) Constructor instantHide: true starts visible and dismisses immediately on FIRST_PRESENT.
  const instantOpt = createController({ fadeMs: 400, instantHide: true, nowFn: () => 3100 });
  assert.equal(instantOpt.controller.getStats().visible, true);
  assert.equal(instantOpt.controller.getStats().instantHide, true);
  assert.equal(instantOpt.controller.getStats().hudHidden, false);
  assert.equal(
    instantOpt.controller.getState().root.style.getPropertyValue('--pg6-splash-fade'),
    '0ms'
  );
  instantOpt.controller.onFirstPresent();
  assert.equal(instantOpt.timer.size, 0);
  assert.equal(instantOpt.dismissCalls.length, 1);
  assert.equal(instantOpt.controller.getStats().visible, false);
  assert.equal(instantOpt.controller.getStats().dismissedAtMs, 3100);
  assert.equal(instantOpt.controller.getStats().dismissReason, DISMISS_REASONS.FIRST_PRESENT);
  assert.equal(instantOpt.controller.getStats().fadeDurationMs, 0);

  // Per-call immediate: false can still opt into animated fade.
  instantOpt.controller.show();
  instantOpt.controller.dismiss({ immediate: false, fadeMs: 35, reason: 'manual_fade' });
  assert.equal(instantOpt.timer.size, 1);
  assert.equal(instantOpt.controller.getStats().fading, true);
  assert.equal(instantOpt.controller.getStats().visible, true);
  instantOpt.timer.flush();
  assert.equal(instantOpt.controller.getStats().visible, false);
  assert.equal(instantOpt.controller.getStats().dismissReason, 'manual_fade');
  assert.equal(instantOpt.controller.getStats().fadeDurationMs, 35);
  instantOpt.controller.destroy();

  // 4) Animated onFirstPresent() vs immediate onFirstPresent({ immediate: true }).
  let animNow = 4000;
  const anim = createController({ fadeMs: 80, nowFn: () => animNow });
  anim.controller.onFirstPresent();
  assert.equal(anim.timer.size, 1);
  assert.equal(anim.controller.getStats().visible, true);
  assert.equal(anim.controller.getStats().fading, true);
  assert.equal(anim.controller.getStats().dismissedAtMs, null);
  assert.equal(anim.controller.getStats().dismissReason, null);
  animNow = 4082;
  anim.timer.flush();
  assert.equal(anim.controller.getStats().visible, false);
  assert.equal(anim.controller.getStats().dismissedAtMs, 4082);
  assert.equal(anim.controller.getStats().dismissReason, DISMISS_REASONS.FIRST_PRESENT);
  assert.equal(anim.controller.getStats().fadeDurationMs, 82);
  anim.controller.destroy();

  const immFirstPresent = createController({ fadeMs: 400, nowFn: () => 5000 });
  immFirstPresent.controller.onFirstPresent({ immediate: true });
  assert.equal(immFirstPresent.timer.size, 0);
  assert.equal(immFirstPresent.dismissCalls.length, 1);
  assert.equal(immFirstPresent.controller.getStats().visible, false);
  assert.equal(immFirstPresent.controller.getStats().dismissedAtMs, 5000);
  assert.equal(immFirstPresent.controller.getStats().dismissReason, DISMISS_REASONS.FIRST_PRESENT);
  assert.equal(immFirstPresent.controller.getStats().fadeDurationMs, 0);
  immFirstPresent.controller.destroy();

  // 5) Upgrading an in-flight animated fade via dismissImmediate cancels timer & fires onDismiss once.
  let upgradeNow = 6000;
  const upgrade = createController({ fadeMs: 250, nowFn: () => upgradeNow });
  upgrade.controller.onFirstPresent();
  assert.equal(upgrade.timer.size, 1);
  assert.equal(upgrade.controller.getStats().fading, true);
  assert.equal(upgrade.dismissCalls.length, 0);

  upgradeNow = 6012;
  upgrade.controller.dismissImmediate({ reason: 'capture_override' });
  assert.equal(upgrade.timer.size, 0);
  assert.equal(upgrade.dismissCalls.length, 1);
  assert.equal(upgrade.controller.getStats().visible, false);
  assert.equal(upgrade.controller.getStats().fading, false);
  assert.equal(upgrade.controller.getStats().dismissedAtMs, 6012);
  assert.equal(upgrade.controller.getStats().dismissReason, 'capture_override');
  assert.equal(upgrade.controller.getStats().fadeDurationMs, 0);
  assert.equal(upgrade.controller.getStats().dismissCount, 1);

  // Flushing timers after cancellation never double-fires.
  upgrade.timer.flush();
  assert.equal(upgrade.dismissCalls.length, 1);
  upgrade.controller.destroy();
});

test('W213: ?hud=0 query instantly hides overlay on construction and avoids fade occlusion', () => {
  // 1) Detected via windowObj.location.search (matches web/app_bootstrap.js wiring).
  const viaWindow = createController({
    fadeMs: 400,
    nowFn: () => 7100,
    windowObj: { location: { search: '?inputScript=intro-watch&hud=0' } },
  });
  const stats = viaWindow.controller.getStats();
  const state = viaWindow.controller.getState();
  assert.equal(stats.visible, false);
  assert.equal(stats.dismissed, true);
  assert.equal(stats.fading, false);
  assert.equal(stats.hudHidden, true);
  assert.equal(stats.instantHide, true);
  assert.equal(stats.dismissedAtMs, 7100);
  assert.equal(stats.dismissReason, DISMISS_REASONS.HUD_HIDDEN);
  assert.equal(stats.fadeDurationMs, 0);
  assert.equal(stats.dismissCount, 1);
  assert.equal(state.root.hidden, true);
  assert.equal(state.root.style.display, 'none');
  assert.equal(state.root.style.getPropertyValue('--pg6-splash-fade'), '0ms');
  assert.equal(state.root.getAttribute('aria-busy'), 'false');
  assert.equal(viaWindow.timer.size, 0);

  // Subsequent onFirstPresent is an idempotent no-op and never schedules a fade.
  viaWindow.controller.onFirstPresent();
  assert.equal(viaWindow.timer.size, 0);
  assert.equal(viaWindow.controller.getStats().visible, false);
  assert.equal(viaWindow.controller.getStats().dismissReason, DISMISS_REASONS.HUD_HIDDEN);
  assert.equal(viaWindow.controller.getStats().dismissCount, 1);
  viaWindow.controller.destroy();

  // 2) Detected via documentObj.defaultView.location.search.
  const docWithLoc = new MockDocument();
  docWithLoc.defaultView = { location: { search: '?hud=false' } };
  const viaDoc = createController({ documentObj: docWithLoc, nowFn: () => 7200 });
  assert.equal(viaDoc.controller.getStats().visible, false);
  assert.equal(viaDoc.controller.getStats().hudHidden, true);
  assert.equal(viaDoc.controller.getStats().dismissReason, DISMISS_REASONS.HUD_HIDDEN);
  viaDoc.controller.destroy();

  // 3) Explicit visible: true with ?hud=0 keeps overlay visible initially but dismisses instantly.
  const visibleHud0 = createController({
    locationSearch: '?hud=0',
    visible: true,
    fadeMs: 400,
    nowFn: () => 7300,
  });
  assert.equal(visibleHud0.controller.getStats().visible, true);
  assert.equal(visibleHud0.controller.getStats().hudHidden, true);
  assert.equal(visibleHud0.controller.getStats().instantHide, true);
  visibleHud0.controller.onFirstPresent();
  assert.equal(visibleHud0.timer.size, 0);
  assert.equal(visibleHud0.controller.getStats().visible, false);
  assert.equal(visibleHud0.controller.getStats().dismissReason, DISMISS_REASONS.FIRST_PRESENT);
  visibleHud0.controller.destroy();

  // 4) Explicit instantHide: false overrides ?hud=0 auto-hide.
  const overrideHud0 = createController({
    locationSearch: '?hud=0',
    instantHide: false,
    fadeMs: 50,
  });
  assert.equal(overrideHud0.controller.getStats().visible, true);
  assert.equal(overrideHud0.controller.getStats().instantHide, false);
  overrideHud0.controller.onFirstPresent();
  assert.equal(overrideHud0.timer.size, 1);
  overrideHud0.controller.destroy();

  // 5) Throwing location getter never breaks construction.
  const hostileWindow = {};
  Object.defineProperty(hostileWindow, 'location', {
    get() {
      throw new Error('SecurityError');
    },
  });
  assert.doesNotThrow(() => {
    const safe = createController({ windowObj: hostileWindow });
    assert.equal(safe.controller.getStats().visible, true);
    assert.equal(safe.controller.getStats().hudHidden, false);
    safe.controller.destroy();
  });
});

test('W213: idempotent double-hide and double-show guards avoid redundant DOM writes', () => {
  const { controller, timer, dismissCalls } = createController({ fadeMs: 60, nowFn: () => 8000 });
  const root = controller.getState().root;

  let hiddenWrites = 0;
  let backingHidden = root._hidden;
  Object.defineProperty(root, 'hidden', {
    configurable: true,
    get() {
      return backingHidden;
    },
    set(v) {
      hiddenWrites += 1;
      backingHidden = Boolean(v);
      root._hidden = backingHidden;
    },
  });

  // Already visible & active: show() is a no-op with zero DOM writes.
  controller.show();
  controller.show();
  assert.equal(hiddenWrites, 0);
  assert.equal(controller.getStats().visible, true);

  // First hide() transitions DOM once; second hide() is an idempotent no-op.
  controller.hide();
  assert.equal(hiddenWrites, 1);
  assert.equal(controller.getStats().visible, false);
  assert.equal(controller.getStats().dismissed, false);
  assert.equal(dismissCalls.length, 0);

  controller.hide();
  controller.hide();
  assert.equal(hiddenWrites, 1);
  assert.equal(controller.getStats().visible, false);

  // Re-show writes once; duplicate show() does not write again.
  controller.show();
  assert.equal(hiddenWrites, 2);
  assert.equal(controller.getStats().visible, true);
  controller.show();
  assert.equal(hiddenWrites, 2);

  // hide() during an in-flight fade cancels the timer without firing onDismiss.
  controller.onFirstPresent();
  assert.equal(timer.size, 1);
  assert.equal(controller.getStats().fading, true);
  controller.hide();
  assert.equal(timer.size, 0);
  assert.equal(controller.getStats().visible, false);
  assert.equal(controller.getStats().fading, false);
  assert.equal(controller.getStats().dismissed, false);
  timer.flush();
  assert.equal(dismissCalls.length, 0);

  // hide() after completed dismiss is also a no-op and preserves dismissal telemetry.
  controller.show();
  controller.dismissImmediate('manual_hide_guard');
  const writesAfterDismiss = hiddenWrites;
  controller.hide();
  assert.equal(hiddenWrites, writesAfterDismiss);
  assert.equal(controller.getStats().dismissed, true);
  assert.equal(controller.getStats().dismissReason, 'manual_hide_guard');
  assert.equal(controller.getStats().dismissedAtMs, 8000);
  controller.destroy();
});

test('W213: re-show on restart cleanly restores overlay and tracks subsequent dismissal telemetry', () => {
  let now = 9000;
  const { controller, timer, dismissCalls } = createController({ fadeMs: 120, nowFn: () => now });

  // First boot reaches FIRST_PRESENT and dismisses immediately.
  controller.updateBootState({ stage: 'BINK_OPEN', progressPct: 89, traps: [{ kind: 'WARN' }] });
  controller.onFirstPresent({ immediate: true });
  assert.equal(controller.getStats().visible, false);
  assert.equal(controller.getStats().dismissed, true);
  assert.equal(controller.getStats().dismissedAtMs, 9000);
  assert.equal(controller.getStats().dismissReason, DISMISS_REASONS.FIRST_PRESENT);
  assert.equal(controller.getStats().dismissCount, 1);
  assert.equal(dismissCalls.length, 1);

  // Restart with show({ reset: true }) restores visibility and resets boot progress, keeping historical telemetry.
  const restarted = controller.show({ reset: true });
  assert.equal(restarted.visible, true);
  assert.equal(restarted.dismissed, false);
  assert.equal(restarted.fading, false);
  assert.equal(restarted.stage, 'NOT_STARTED');
  assert.equal(restarted.progressPct, 0);
  assert.equal(restarted.readyForSplash, false);
  assert.equal(restarted.badgeText, '');
  assert.equal(restarted.root.hidden, false);
  assert.equal(restarted.root.style.display, '');
  assert.equal(restarted.root.classList.contains('pg6-splash--fading'), false);
  assert.equal(restarted.root.style.getPropertyValue('--pg6-splash-fade'), '120ms');
  assert.equal(restarted.root.getAttribute('aria-busy'), 'true');
  assert.equal(restarted.dismissCount, 1);
  assert.equal(restarted.dismissedAtMs, 9000);
  assert.equal(restarted.dismissReason, DISMISS_REASONS.FIRST_PRESENT);

  // Mid-fade restart cancels pending fade so onDismiss does not fire spuriously.
  now = 9500;
  controller.onFirstPresent();
  assert.equal(timer.size, 1);
  assert.equal(controller.getStats().fading, true);
  controller.show({ reset: true });
  assert.equal(timer.size, 0);
  assert.equal(controller.getStats().visible, true);
  assert.equal(controller.getStats().fading, false);
  assert.equal(controller.getStats().dismissed, false);
  timer.flush();
  assert.equal(dismissCalls.length, 1);

  // Second completed dismissal updates telemetry accurately.
  now = 10250;
  controller.dismissImmediate({ reason: 'second_boot_ready' });
  const finalStats = controller.getStats();
  assert.equal(finalStats.visible, false);
  assert.equal(finalStats.dismissed, true);
  assert.equal(finalStats.dismissCount, 2);
  assert.equal(finalStats.dismissedAtMs, 10250);
  assert.equal(finalStats.dismissReason, 'second_boot_ready');
  assert.equal(finalStats.fadeDurationMs, 0);
  assert.equal(dismissCalls.length, 2);
  assert.doesNotThrow(() => JSON.stringify(finalStats));
  controller.destroy();
});

test('W228: exports isGregQuery, Spinning Greg constants, and web/assets/greg.png exists on disk', () => {
  assert.equal(typeof isGregQuery, 'function');
  assert.equal(DEFAULT_GREG_IMAGE_URL, './assets/greg.png');
  assert.equal(DEFAULT_GREG_CLICK_THRESHOLD, 3);
  assert.equal(GREG_SECRET_WORD, 'greg');
  assert.equal(globalThis.PlayGta6SplashOverlay.isGregQuery, isGregQuery);
  assert.equal(globalThis.PlayGta6SplashOverlay.DEFAULT_GREG_IMAGE_URL, DEFAULT_GREG_IMAGE_URL);
  assert.equal(
    globalThis.PlayGta6SplashOverlay.DEFAULT_GREG_CLICK_THRESHOLD,
    DEFAULT_GREG_CLICK_THRESHOLD
  );
  assert.equal(globalThis.PlayGta6SplashOverlay.GREG_SECRET_WORD, GREG_SECRET_WORD);

  // Verify org mascot image exists on disk and is a valid, compact PNG.
  const gregPath = path.join(REPO_ROOT, 'web', 'assets', 'greg.png');
  assert.equal(fs.existsSync(gregPath), true, 'web/assets/greg.png must exist on disk');
  const bytes = fs.readFileSync(gregPath);
  assert.ok(bytes.length > 0 && bytes.length < 512 * 1024, 'greg.png must be non-empty and < 512 KiB');
  assert.deepEqual(
    Array.from(bytes.subarray(0, 8)),
    [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a],
    'greg.png must have a valid PNG signature'
  );

  // isGregQuery parsing across string, URLSearchParams, Location, and full URLs.
  assert.equal(isGregQuery('?greg=1'), true);
  assert.equal(isGregQuery('greg=1'), true);
  assert.equal(isGregQuery('?GREG=TRUE'), true);
  assert.equal(isGregQuery('?greg=true'), true);
  assert.equal(isGregQuery('?greg=on'), true);
  assert.equal(isGregQuery('?greg=yes'), true);
  assert.equal(isGregQuery('?foo=1&greg=yes&bar=2'), true);
  assert.equal(isGregQuery('?greg=0&greg=1'), true);
  assert.equal(isGregQuery('?greg=1&greg=0'), false);
  assert.equal(isGregQuery('?greg=0'), false);
  assert.equal(isGregQuery('?greg=false'), false);
  assert.equal(isGregQuery('?greg='), false);
  assert.equal(isGregQuery(''), false);
  assert.equal(isGregQuery(null), false);
  assert.equal(isGregQuery(undefined), false);
  assert.equal(isGregQuery(new URLSearchParams('?greg=1')), true);
  assert.equal(isGregQuery(new URLSearchParams('?greg=0')), false);
  assert.equal(isGregQuery({ search: '?greg=true' }), true);
  assert.equal(isGregQuery('http://localhost:8000/?greg=1#top'), true);
  assert.equal(isGregQuery('http://localhost:8000/?greg=0#?greg=1'), false);
});

test('W228: default Greg Theft Auto hero renders cleanly with non-spinning banner, monochrome ring spinner, hides on dismiss, and respects reduced motion', () => {
  const doc = new MockDocument();
  const { controller } = createController({ documentObj: doc });
  const state = controller.getState();
  const stats = controller.getStats();

  assert.equal(state.gregMode, true);
  assert.equal(stats.gregMode, true);
  assert.equal(state.gregBackdropActive, true);

  const panel = state.root.querySelector('.pg6-splash__panel');
  assert.ok(panel);
  const heroImg = state.root.querySelector('.pg6-splash__img');
  const spinner = panel.querySelector('.pg6-splash__spinner');
  const ring = panel.querySelector('.pg6-splash__spinner-ring');
  const greg = panel.querySelector('.pg6-splash__greg');
  const stage = panel.querySelector('.pg6-splash__stage');

  assert.ok(heroImg, 'hero splash img element must render');
  assert.ok(spinner, 'spinner element must render inside panel');
  assert.ok(ring, 'monochrome CSS ring spinner must render inside spinner');
  assert.ok(greg, 'legacy greg img element remains hidden and non-spinning');
  assert.ok(stage, 'stage label must remain intact inside panel');
  assert.equal(heroImg.hidden, false);
  assert.equal(heroImg.getAttribute('src'), './assets/greg.png');
  assert.equal(heroImg.getAttribute('alt'), 'Greg Theft Auto');
  assert.equal(heroImg.classList.contains('pg6-splash__img--greg'), true);
  assert.equal(spinner.hidden, false);
  assert.equal(spinner.style.display, '');
  assert.equal(spinner.getAttribute('aria-hidden'), 'true');
  assert.equal(spinner.getAttribute('data-greg-mode'), 'true');
  assert.equal(spinner.classList.contains('pg6-splash__spinner--greg'), false);
  assert.equal(ring.hidden, false);
  assert.equal(ring.style.display, '');
  assert.equal(greg.hidden, true);
  assert.equal(greg.style.display, 'none');
  assert.equal(greg.style.animation, 'none');

  // Shared stylesheet includes spin keyframes for ring, subtle pulse for hero banner, and reduced-motion suppression.
  const styleEl = doc.head.children.find((el) => el.getAttribute('data-pg6-splash-style') === '1');
  assert.ok(styleEl);
  assert.match(styleEl.textContent, /@keyframes pg6-spin/);
  assert.match(styleEl.textContent, /@keyframes pg6-greg-pulse/);
  assert.match(styleEl.textContent, /\.pg6-splash__spinner/);
  assert.match(styleEl.textContent, /\.pg6-splash__greg\{[^}]*animation:none;/);
  assert.match(
    styleEl.textContent,
    /@media \(prefers-reduced-motion: reduce\)\{[\s\S]*\.pg6-splash__spinner-ring,\.pg6-splash__greg,\.pg6-splash__img--greg\{animation:none!important;transform:none!important;\}\}/
  );

  // Dismissing hides the spinner cleanly; show() restores it.
  controller.dismiss({ fadeMs: 0 });
  assert.equal(spinner.hidden, true);
  assert.equal(spinner.style.display, 'none');

  controller.show();
  assert.equal(spinner.hidden, false);
  assert.equal(spinner.style.display, '');
  controller.destroy();

  // Reduced motion disables ring spin & hero pulse animations while keeping Greg Theft Auto visible cleanly.
  const reduced = createController({ reducedMotion: true, gregMode: true });
  const rRoot = reduced.controller.getState().root;
  const rHero = rRoot.querySelector('.pg6-splash__img');
  const rSpinner = rRoot.querySelector('.pg6-splash__spinner');
  const rRing = rRoot.querySelector('.pg6-splash__spinner-ring');
  const rGreg = rRoot.querySelector('.pg6-splash__greg');
  assert.equal(reduced.controller.getStats().gregMode, true);
  assert.equal(reduced.controller.getStats().reducedMotion, true);
  assert.equal(rSpinner.hidden, false);
  assert.equal(rSpinner.getAttribute('data-reduced-motion'), 'true');
  assert.equal(rRing.style.animation, 'none');
  assert.equal(rHero.hidden, false);
  assert.equal(rHero.style.animation, 'none');
  assert.equal(rGreg.hidden, true);
  assert.equal(rGreg.style.display, 'none');
  assert.equal(rGreg.style.animation, 'none');
  reduced.controller.destroy();
});

test('W228: ?greg=1 / ?greg=0 query and constructor options control Greg Theft Auto mode and preserve ?hud=0', () => {
  // Option validation.
  assert.throws(
    () => new SplashOverlayController({ documentObj: new MockDocument(), gregMode: 'yes' }),
    TypeError
  );
  assert.throws(
    () => new SplashOverlayController({ documentObj: new MockDocument(), gregClickThreshold: 0 }),
    TypeError
  );
  assert.throws(
    () => new SplashOverlayController({ documentObj: new MockDocument(), gregClickThreshold: 1.5 }),
    TypeError
  );
  assert.throws(
    () => new SplashOverlayController({ documentObj: new MockDocument(), gregImageUrl: 42 }),
    TypeError
  );

  // ?greg=1 via locationSearch keeps Greg Theft Auto hero active with clean ring spinner.
  const viaQuery1 = createController({ locationSearch: '?greg=1' });
  assert.equal(viaQuery1.controller.getState().gregMode, true);
  assert.equal(viaQuery1.controller.getStats().gregMode, true);
  const root1 = viaQuery1.controller.getState().root;
  assert.equal(root1.querySelector('.pg6-splash__img').getAttribute('src'), './assets/greg.png');
  assert.equal(root1.querySelector('.pg6-splash__img').getAttribute('alt'), 'Greg Theft Auto');
  assert.equal(root1.querySelector('.pg6-splash__spinner').getAttribute('data-greg-mode'), 'true');
  assert.equal(root1.querySelector('.pg6-splash__spinner-ring').hidden, false);
  assert.equal(root1.querySelector('.pg6-splash__greg').hidden, true);
  viaQuery1.controller.destroy();

  // ?greg=0 via locationSearch opts out of Greg Theft Auto hero.
  const viaQuery0 = createController({ locationSearch: '?greg=0' });
  assert.equal(viaQuery0.controller.getState().gregMode, false);
  assert.equal(viaQuery0.controller.getStats().gregMode, false);
  const root0 = viaQuery0.controller.getState().root;
  assert.equal(root0.querySelector('.pg6-splash__spinner').getAttribute('data-greg-mode'), 'false');
  assert.equal(root0.querySelector('.pg6-splash__spinner-ring').hidden, false);
  assert.equal(root0.querySelector('.pg6-splash__greg').hidden, true);
  assert.equal(root0.querySelector('.pg6-splash__img').hidden, true);
  viaQuery0.controller.destroy();

  // ?greg=true via windowObj.location.search enables Greg Theft Auto automatically.
  const viaWindow = createController({ windowObj: { location: { search: '?foo=bar&greg=true' } } });
  assert.equal(viaWindow.controller.getState().gregMode, true);
  assert.equal(viaWindow.controller.getStats().gregMode, true);
  viaWindow.controller.destroy();

  // Explicit constructor gregMode option overrides query in both directions.
  const explicitOn = createController({ locationSearch: '?greg=0', gregMode: true });
  assert.equal(explicitOn.controller.getStats().gregMode, true);
  explicitOn.controller.destroy();

  const explicitOff = createController({ locationSearch: '?greg=1', gregMode: false });
  assert.equal(explicitOff.controller.getStats().gregMode, false);
  explicitOff.controller.destroy();

  // ?hud=0 still hides the overlay cleanly even when combined with default/explicit Greg.
  const hudHiddenOnly = createController({ locationSearch: '?hud=0', nowFn: () => 5000 });
  assert.equal(hudHiddenOnly.controller.getStats().visible, false);
  assert.equal(hudHiddenOnly.controller.getStats().dismissed, true);
  assert.equal(hudHiddenOnly.controller.getStats().hudHidden, true);
  assert.equal(hudHiddenOnly.controller.getStats().gregMode, true);
  assert.equal(hudHiddenOnly.controller.getState().root.hidden, true);
  hudHiddenOnly.controller.destroy();

  const hudHiddenWithGreg = createController({ locationSearch: '?hud=0&greg=1', nowFn: () => 5100 });
  assert.equal(hudHiddenWithGreg.controller.getStats().visible, false);
  assert.equal(hudHiddenWithGreg.controller.getStats().dismissed, true);
  assert.equal(hudHiddenWithGreg.controller.getStats().hudHidden, true);
  assert.equal(hudHiddenWithGreg.controller.getStats().gregMode, true);
  assert.equal(hudHiddenWithGreg.controller.getState().root.hidden, true);
  assert.equal(hudHiddenWithGreg.controller.getState().root.querySelector('.pg6-splash__spinner').hidden, true);
  hudHiddenWithGreg.controller.destroy();
});

test('W228: 3x spinner click trigger toggles Greg Theft Auto hero cleanly and respects gregClickThreshold', () => {
  const { controller } = createController();
  const root = controller.getState().root;
  const heroImg = root.querySelector('.pg6-splash__img');
  const spinner = root.querySelector('.pg6-splash__spinner');
  const ring = root.querySelector('.pg6-splash__spinner-ring');
  const greg = root.querySelector('.pg6-splash__greg');

  // Greg Theft Auto hero is enabled by default; monochrome ring spinner stays active and non-spinning banner never spins.
  assert.equal(controller.getState().gregMode, true);
  spinner.dispatch('click');
  spinner.dispatch('click');
  assert.equal(controller.getState().gregMode, true);
  assert.equal(heroImg.hidden, false);
  assert.equal(ring.hidden, false);
  assert.equal(greg.hidden, true);

  // 3rd click toggles Greg Theft Auto hero OFF.
  spinner.dispatch('click');
  assert.equal(controller.getState().gregMode, false);
  assert.equal(controller.getStats().gregMode, false);
  assert.equal(heroImg.hidden, true);
  assert.equal(ring.hidden, false);
  assert.equal(greg.hidden, true);

  // Clicking while hidden is ignored.
  controller.hide();
  spinner.dispatch('click');
  spinner.dispatch('click');
  spinner.dispatch('click');
  assert.equal(controller.getState().gregMode, false);

  // Show again and click 3x to toggle back ON.
  controller.show();
  spinner.dispatch('click');
  spinner.dispatch('click');
  assert.equal(controller.getState().gregMode, false);
  spinner.dispatch('click');
  assert.equal(controller.getState().gregMode, true);
  assert.equal(heroImg.hidden, false);
  assert.equal(ring.hidden, false);
  assert.equal(greg.hidden, true);

  // Listener is cleaned up on destroy.
  assert.equal(spinner.listenerCount('click'), 1);
  controller.destroy();
  assert.equal(spinner.listenerCount('click'), 0);

  // Custom gregClickThreshold works.
  const customThreshold = createController({ gregClickThreshold: 2 });
  const customSpinner = customThreshold.controller.getState().root.querySelector('.pg6-splash__spinner');
  customSpinner.dispatch('click');
  assert.equal(customThreshold.controller.getStats().gregMode, true);
  customSpinner.dispatch('click');
  assert.equal(customThreshold.controller.getStats().gregMode, false);
  customThreshold.controller.destroy();
});

test('W228: typing g-r-e-g toggles Greg Theft Auto hero, ignores editable inputs & modifiers, and cleans up on destroy', () => {
  const { controller, documentObj } = createController();
  assert.equal(documentObj.listenerCount('keydown'), 1);
  assert.equal(controller.getStats().gregMode, true);

  const typeWord = (target, word, extra = {}) => {
    for (const ch of word) {
      target.dispatch('keydown', { key: ch, ...extra });
    }
  };

  // Ignored when modifiers are held.
  typeWord(documentObj, 'greg', { ctrlKey: true });
  assert.equal(controller.getStats().gregMode, true);
  typeWord(documentObj, 'greg', { altKey: true });
  assert.equal(controller.getStats().gregMode, true);
  typeWord(documentObj, 'greg', { metaKey: true });
  assert.equal(controller.getStats().gregMode, true);

  // Ignored when event originates from editable inputs/textareas/select/contenteditable.
  typeWord(documentObj, 'greg', { target: { tagName: 'INPUT' } });
  typeWord(documentObj, 'greg', { target: { tagName: 'TEXTAREA' } });
  typeWord(documentObj, 'greg', { target: { tagName: 'SELECT' } });
  typeWord(documentObj, 'greg', { target: { tagName: 'DIV', isContentEditable: true } });
  typeWord(documentObj, 'greg', {
    target: { tagName: 'DIV', getAttribute: (name) => (name === 'contenteditable' ? 'true' : null) },
  });
  assert.equal(controller.getStats().gregMode, true);

  // Typing g-r-e-g (even after unrelated keys) toggles Greg Theft Auto hero OFF.
  typeWord(documentObj, 'abgreg');
  assert.equal(controller.getState().gregMode, false);
  assert.equal(controller.getStats().gregMode, false);
  assert.equal(controller.getState().gregBackdropActive, false);
  assert.equal(controller.getState().root.querySelector('.pg6-splash__img').hidden, true);

  // Ignored while splash overlay is hidden or dismissed.
  controller.hide();
  typeWord(documentObj, 'greg');
  assert.equal(controller.getStats().gregMode, false);
  controller.show();

  // Case-insensitive typing toggles Greg Theft Auto hero ON again.
  typeWord(documentObj, 'GREG');
  assert.equal(controller.getStats().gregMode, true);
  assert.equal(controller.getState().gregBackdropActive, true);
  assert.equal(controller.getState().root.querySelector('.pg6-splash__img').hidden, false);

  // destroy() removes keydown listener so no listeners leak.
  controller.destroy();
  assert.equal(documentObj.listenerCount('keydown'), 0);
  typeWord(documentObj, 'greg');
  assert.equal(controller.getStats().gregMode, true);

  // Also binds and cleans up when windowObj provides addEventListener.
  const winListeners = new Map();
  const mockWin = {
    location: { search: '' },
    addEventListener(type, fn) {
      if (!winListeners.has(type)) winListeners.set(type, new Set());
      winListeners.get(type).add(fn);
    },
    removeEventListener(type, fn) {
      winListeners.get(type)?.delete(fn);
    },
    dispatch(type, event) {
      for (const fn of winListeners.get(type) || []) fn(event);
    },
  };
  const viaWin = createController({ windowObj: mockWin });
  assert.equal(winListeners.get('keydown')?.size, 1);
  typeWord(mockWin, 'greg');
  assert.equal(viaWin.controller.getStats().gregMode, false);
  viaWin.controller.destroy();
  assert.equal(winListeners.get('keydown')?.size, 0);
});

test('W228: setGregMode and toggleGregMode update DOM & telemetry cleanly and are safe no-ops after destroy', () => {
  const { controller } = createController();
  const root = controller.getState().root;
  const heroImg = root.querySelector('.pg6-splash__img');
  const spinner = root.querySelector('.pg6-splash__spinner');
  const ring = root.querySelector('.pg6-splash__spinner-ring');
  const greg = root.querySelector('.pg6-splash__greg');

  const s1 = controller.setGregMode(true);
  assert.equal(s1.gregMode, true);
  assert.equal(controller.getStats().gregMode, true);
  assert.equal(root.getAttribute('data-greg-mode'), 'true');
  assert.equal(spinner.getAttribute('data-greg-mode'), 'true');
  assert.equal(heroImg.hidden, false);
  assert.equal(ring.hidden, false);
  assert.equal(greg.hidden, true);
  assert.doesNotThrow(() => JSON.stringify(controller.getStats()));

  const s2 = controller.toggleGregMode();
  assert.equal(s2.gregMode, false);
  assert.equal(controller.getStats().gregMode, false);
  assert.equal(root.getAttribute('data-greg-mode'), 'false');
  assert.equal(spinner.getAttribute('data-greg-mode'), 'false');
  assert.equal(heroImg.hidden, true);
  assert.equal(ring.hidden, false);
  assert.equal(greg.hidden, true);

  const s3 = controller.toggleGregMode();
  assert.equal(s3.gregMode, true);
  const s4 = controller.setGregMode(false);
  assert.equal(s4.gregMode, false);

  controller.destroy();
  assert.doesNotThrow(() => controller.setGregMode(true));
  assert.doesNotThrow(() => controller.toggleGregMode());
  assert.equal(controller.getStats().gregMode, false);
  assert.equal(controller.getStats().destroyed, true);
});

test('W230: exports isSplashHoldQuery, W230 constants, and dismiss reasons on module and UMD global', () => {
  assert.equal(typeof isSplashHoldQuery, 'function');
  assert.equal(DEFAULT_VISIBLE_FRAME_FADE_MS, 450);
  assert.equal(DEFAULT_HOLD_FALLBACK_TIMEOUT_MS, 240000);
  assert.equal(DEFAULT_LOADING_ASSETS_MESSAGE, 'Loading game assets...');
  assert.equal(DISMISS_REASONS.FIRST_VISIBLE_FRAME, 'first_visible_frame');
  assert.equal(DISMISS_REASONS.HOLD_FALLBACK_TIMEOUT, 'hold_fallback_timeout');
  assert.equal(globalThis.PlayGta6SplashOverlay.isSplashHoldQuery, isSplashHoldQuery);
  assert.equal(
    globalThis.PlayGta6SplashOverlay.DEFAULT_VISIBLE_FRAME_FADE_MS,
    DEFAULT_VISIBLE_FRAME_FADE_MS
  );
  assert.equal(
    globalThis.PlayGta6SplashOverlay.DEFAULT_HOLD_FALLBACK_TIMEOUT_MS,
    DEFAULT_HOLD_FALLBACK_TIMEOUT_MS
  );
  assert.equal(
    globalThis.PlayGta6SplashOverlay.DEFAULT_LOADING_ASSETS_MESSAGE,
    DEFAULT_LOADING_ASSETS_MESSAGE
  );

  // isSplashHoldQuery parsing across string, URLSearchParams, Location, and full URLs.
  assert.equal(isSplashHoldQuery('?splashHold=1'), true);
  assert.equal(isSplashHoldQuery('splashHold=1'), true);
  assert.equal(isSplashHoldQuery('?SPLASHHOLD=TRUE'), true);
  assert.equal(isSplashHoldQuery('?splashhold=true'), true);
  assert.equal(isSplashHoldQuery('?splashHold=on'), true);
  assert.equal(isSplashHoldQuery('?splashHold=yes'), true);
  assert.equal(isSplashHoldQuery('?foo=1&splashHold=1&bar=2'), true);
  assert.equal(isSplashHoldQuery('?splashHold=0&splashHold=1'), true);
  assert.equal(isSplashHoldQuery('?splashHold=1&splashHold=0'), false);
  assert.equal(isSplashHoldQuery('?splashHold=0'), false);
  assert.equal(isSplashHoldQuery('?splashHold=false'), false);
  assert.equal(isSplashHoldQuery('?splashHold='), false);
  assert.equal(isSplashHoldQuery(''), false);
  assert.equal(isSplashHoldQuery(null), false);
  assert.equal(isSplashHoldQuery(undefined), false);
  assert.equal(isSplashHoldQuery(new URLSearchParams('?splashHold=1')), true);
  assert.equal(isSplashHoldQuery(new URLSearchParams('?splashHold=0')), false);
  assert.equal(isSplashHoldQuery({ search: '?splashHold=true' }), true);
  assert.equal(isSplashHoldQuery('http://localhost:8000/?splashHold=1#top'), true);
  assert.equal(isSplashHoldQuery('http://localhost:8000/?splashHold=0#?splashHold=1'), false);
});

test('W230: validates holdUntilVisible options, defaults false for backwards compatibility, and honors ?splashHold=1', () => {
  assert.throws(
    () => new SplashOverlayController({ documentObj: new MockDocument(), holdUntilVisible: 'yes' }),
    TypeError
  );
  assert.throws(
    () => new SplashOverlayController({ documentObj: new MockDocument(), holdFallbackTimeoutMs: -1 }),
    TypeError
  );
  assert.throws(
    () =>
      new SplashOverlayController({
        documentObj: new MockDocument(),
        holdFallbackTimeoutMs: Number.POSITIVE_INFINITY,
      }),
    TypeError
  );
  assert.throws(
    () => new SplashOverlayController({ documentObj: new MockDocument(), holdFallbackTimeoutMs: '240000' }),
    TypeError
  );
  assert.throws(
    () => new SplashOverlayController({ documentObj: new MockDocument(), visibleFrameFadeMs: -5 }),
    TypeError
  );
  assert.throws(
    () => new SplashOverlayController({ documentObj: new MockDocument(), visibleFrameFadeMs: '450' }),
    TypeError
  );

  // Backwards compatibility: default holdUntilVisible is false and onFirstPresent dismisses immediately/fades.
  let now = 1000;
  const compat = createController({ fadeMs: 40, nowFn: () => now });
  assert.equal(compat.controller.getStats().holdUntilVisible, false);
  assert.equal(compat.controller.getStats().holdingForVisibleFrame, false);
  assert.equal(compat.controller.getStats().heldForVisibleFrame, false);
  assert.equal(compat.controller.getStats().firstVisibleFrameMs, null);
  assert.equal(compat.controller.getStats().blackPresentsWhileHeld, 0);
  compat.controller.onFirstPresent();
  assert.equal(compat.controller.getStats().dismissed, true);
  assert.equal(compat.controller.getStats().fading, true);
  assert.equal(compat.controller.getStats().heldForVisibleFrame, false);
  now = 1040;
  compat.timer.flush();
  assert.equal(compat.controller.getStats().visible, false);
  assert.equal(compat.controller.getStats().dismissReason, DISMISS_REASONS.FIRST_PRESENT);
  assert.equal(compat.controller.getStats().heldForVisibleFrame, false);
  assert.equal(compat.controller.getStats().firstVisibleFrameMs, null);
  assert.equal(compat.controller.getStats().blackPresentsWhileHeld, 0);
  assert.equal(compat.dismissCalls.length, 1);
  compat.controller.destroy();

  // ?splashHold=1 via locationSearch enables holdUntilVisible automatically.
  const viaQuery = createController({ locationSearch: '?splashHold=1' });
  assert.equal(viaQuery.controller.getStats().holdUntilVisible, true);
  assert.equal(viaQuery.controller.getState().holdUntilVisible, true);
  viaQuery.controller.destroy();

  // ?splashHold=1 via windowObj.location.search enables holdUntilVisible automatically.
  const viaWindow = createController({ windowObj: { location: { search: '?foo=1&splashHold=true' } } });
  assert.equal(viaWindow.controller.getStats().holdUntilVisible, true);
  viaWindow.controller.destroy();

  // Explicit holdUntilVisible: false overrides ?splashHold=1 query.
  const explicitOff = createController({ locationSearch: '?splashHold=1', holdUntilVisible: false });
  assert.equal(explicitOff.controller.getStats().holdUntilVisible, false);
  explicitOff.controller.destroy();

  // ?hud=0 still hides the overlay cleanly even when combined with ?splashHold=1.
  const hudHidden = createController({ locationSearch: '?hud=0&splashHold=1', nowFn: () => 2000 });
  assert.equal(hudHidden.controller.getStats().visible, false);
  assert.equal(hudHidden.controller.getStats().dismissed, true);
  assert.equal(hudHidden.controller.getStats().holdUntilVisible, true);
  hudHidden.controller.onFirstPresent();
  assert.equal(hudHidden.controller.getStats().visible, false);
  assert.equal(hudHidden.controller.getStats().holdingForVisibleFrame, false);
  assert.equal(hudHidden.timer.size, 0);
  hudHidden.controller.destroy();
});

test('W230: holdUntilVisible holds past FIRST_PRESENT, updates loading progress, preserves Spinning Greg, and fades out smoothly on notifyFirstVisibleFrame', () => {
  let now = 45000;
  const { controller, documentObj, timer, dismissCalls } = createController({
    holdUntilVisible: true,
    nowFn: () => now,
  });
  const root = controller.getState().root;
  const stageEl = root.querySelector('.pg6-splash__stage');
  const spinner = root.querySelector('.pg6-splash__spinner');
  const gregImg = root.querySelector('.pg6-splash__greg');

  // Pre-FIRST_PRESENT boot progress works normally.
  controller.updateBootState({ stage: 'BINK_OPEN', progressPct: 89, elapsedMs: 42000 });
  assert.equal(controller.getStats().holdingForVisibleFrame, false);
  assert.equal(controller.getStats().heldForVisibleFrame, false);

  // FIRST_PRESENT enters smooth hold phase instead of dismissing.
  const heldState = controller.onFirstPresent();
  assert.equal(heldState.visible, true);
  assert.equal(heldState.dismissed, false);
  assert.equal(heldState.fading, false);
  assert.equal(heldState.holdingForVisibleFrame, true);
  assert.equal(heldState.heldForVisibleFrame, true);
  assert.equal(heldState.blackPresentsWhileHeld, 1);
  assert.equal(heldState.statusText, 'Loading game assets...');
  assert.equal(stageEl.textContent, 'Loading game assets...');
  assert.equal(heldState.progressPct, 90);
  assert.equal(root.getAttribute('aria-busy'), 'true');
  assert.equal(root.getAttribute('data-holding-for-visible'), 'true');
  assert.equal(root.classList.contains('pg6-splash--holding'), true);
  assert.equal(spinner.hidden, false);
  assert.equal(dismissCalls.length, 0);
  // Default 240000 ms fallback safety timer is armed.
  assert.deepEqual(timer.delays(), [240000]);

  // setLoadingAssetsProgress updates status text and progress smoothly without dismissing.
  const p1 = controller.setLoadingAssetsProgress({ presents: 30, realPackDraws: 0 });
  assert.equal(p1.visible, true);
  assert.equal(p1.dismissed, false);
  assert.equal(p1.holdingForVisibleFrame, true);
  assert.equal(p1.blackPresentsWhileHeld, 30);
  assert.equal(p1.loadingRealPackDraws, 0);
  assert.match(p1.statusText, /Loading game assets\.\.\..*30 frames/);
  assert.ok(p1.progressPct >= 90 && p1.progressPct <= 99);

  const p2 = controller.setLoadingAssetsProgress({
    presents: 120,
    realPackDraws: 18,
    elapsedMs: 110000,
  });
  assert.equal(p2.blackPresentsWhileHeld, 120);
  assert.equal(p2.loadingRealPackDraws, 18);
  assert.equal(p2.elapsedMs, 110000);
  assert.match(p2.statusText, /Loading game assets\.\.\..*18 draws.*120 frames/);
  assert.ok(p2.progressPct >= p1.progressPct && p2.progressPct <= 99);

  // Explicit custom message is rendered verbatim (sanitized of control chars).
  const p3 = controller.setLoadingAssetsProgress({
    presents: 180,
    realPackDraws: 42,
    message: 'Streaming shaders.rpf & levels.rpf...',
  });
  assert.equal(p3.statusText, 'Streaming shaders.rpf & levels.rpf...');
  assert.equal(stageEl.textContent, 'Streaming shaders.rpf & levels.rpf...');
  assert.equal(p3.blackPresentsWhileHeld, 180);

  // Late BootTracer FIRST_PRESENT snapshot does not overwrite loading-assets text or jump to 100%.
  controller.updateBootState({ stage: 'FIRST_PRESENT', progressPct: 100, elapsedMs: 125000 });
  assert.equal(controller.getState().statusText, 'Streaming shaders.rpf & levels.rpf...');
  assert.ok(controller.getState().progressPct <= 99);
  assert.equal(controller.getState().dismissed, false);

  // W228 Greg Theft Auto hero triggers still work while held (default ON -> 3x click OFF -> typing greg ON)!
  const heroImg = root.querySelector('.pg6-splash__img');
  assert.equal(controller.getStats().gregMode, true);
  assert.equal(controller.getStats().gregBackdropActive, true);
  assert.equal(heroImg.hidden, false);
  assert.equal(gregImg.hidden, true);
  spinner.dispatch('click');
  spinner.dispatch('click');
  spinner.dispatch('click');
  assert.equal(controller.getStats().gregMode, false);
  assert.equal(controller.getStats().gregBackdropActive, false);
  assert.equal(heroImg.hidden, true);
  for (const ch of 'greg') {
    documentObj.dispatch('keydown', { key: ch });
  }
  assert.equal(controller.getStats().gregMode, true);
  assert.equal(controller.getStats().gregBackdropActive, true);
  assert.equal(heroImg.hidden, false);

  // notifyFirstVisibleFrame() cancels fallback timer and fades out smoothly (450 ms default).
  now = 195000;
  const fadingState = controller.notifyFirstVisibleFrame();
  assert.equal(fadingState.visible, true);
  assert.equal(fadingState.dismissed, true);
  assert.equal(fadingState.fading, true);
  assert.equal(fadingState.holdingForVisibleFrame, false);
  assert.equal(fadingState.heldForVisibleFrame, true);
  assert.equal(fadingState.firstVisibleFrameMs, 195000);
  assert.equal(fadingState.blackPresentsWhileHeld, 180);
  assert.equal(fadingState.progressPct, 100);
  assert.equal(root.classList.contains('pg6-splash--fading'), true);
  assert.equal(root.classList.contains('pg6-splash--holding'), false);
  assert.equal(root.getAttribute('data-holding-for-visible'), 'false');
  assert.equal(root.getAttribute('aria-busy'), 'false');
  assert.equal(root.style.getPropertyValue('--pg6-splash-fade'), '450ms');
  // Fallback timer was cancelled; only the 450 ms fade timer remains.
  assert.deepEqual(timer.delays(), [450]);
  assert.equal(dismissCalls.length, 0);

  // Complete fade transition.
  now = 195450;
  timer.flush();
  const doneStats = controller.getStats();
  assert.equal(doneStats.visible, false);
  assert.equal(doneStats.dismissed, true);
  assert.equal(doneStats.fading, false);
  assert.equal(doneStats.dismissCount, 1);
  assert.equal(doneStats.dismissReason, DISMISS_REASONS.FIRST_VISIBLE_FRAME);
  assert.equal(doneStats.fadeDurationMs, 450);
  assert.equal(doneStats.heldForVisibleFrame, true);
  assert.equal(doneStats.firstVisibleFrameMs, 195000);
  assert.equal(doneStats.blackPresentsWhileHeld, 180);
  assert.equal(doneStats.holdFallbackTimedOut, false);
  assert.equal(dismissCalls.length, 1);
  assert.doesNotThrow(() => JSON.stringify(doneStats));

  // Idempotent second notifyFirstVisibleFrame & setLoadingAssetsProgress after dismiss.
  now = 210000;
  controller.notifyFirstVisibleFrame();
  controller.setLoadingAssetsProgress({ presents: 999, message: 'Should be ignored' });
  assert.equal(controller.getStats().firstVisibleFrameMs, 195000);
  assert.equal(controller.getStats().blackPresentsWhileHeld, 180);
  assert.equal(controller.getStats().dismissCount, 1);
  assert.equal(dismissCalls.length, 1);

  // show({ reset: true }) resets hold telemetry for a fresh boot cycle.
  const resetState = controller.show({ reset: true });
  assert.equal(resetState.visible, true);
  assert.equal(resetState.dismissed, false);
  assert.equal(resetState.holdingForVisibleFrame, false);
  assert.equal(resetState.heldForVisibleFrame, false);
  assert.equal(resetState.firstVisibleFrameMs, null);
  assert.equal(resetState.blackPresentsWhileHeld, 0);
  assert.equal(resetState.loadingRealPackDraws, 0);
  controller.destroy();
});

test('W230: notifyFirstVisibleFrame supports custom fadeDurationMs, immediate, and reducedMotion', () => {
  // 1) Custom fadeDurationMs override.
  let now = 5000;
  const custom = createController({ holdUntilVisible: true, nowFn: () => now });
  custom.controller.onFirstPresent();
  now = 6000;
  custom.controller.notifyFirstVisibleFrame({ fadeDurationMs: 600 });
  assert.deepEqual(custom.timer.delays(), [600]);
  assert.equal(custom.controller.getState().root.style.getPropertyValue('--pg6-splash-fade'), '600ms');
  now = 6600;
  custom.timer.flush();
  assert.equal(custom.controller.getStats().fadeDurationMs, 600);
  assert.equal(custom.controller.getStats().visible, false);
  custom.controller.destroy();

  // 2) fadeDurationMs: 0 hides synchronously without scheduling a fade timer.
  const zero = createController({ holdUntilVisible: true, nowFn: () => 7000 });
  zero.controller.onFirstPresent();
  zero.controller.notifyFirstVisibleFrame({ fadeDurationMs: 0 });
  assert.equal(zero.timer.size, 0);
  assert.equal(zero.controller.getStats().visible, false);
  assert.equal(zero.controller.getStats().fadeDurationMs, 0);
  assert.equal(zero.controller.getStats().firstVisibleFrameMs, 7000);
  assert.equal(zero.dismissCalls.length, 1);
  zero.controller.destroy();

  // 3) reducedMotion skips transition delay on notifyFirstVisibleFrame.
  const reduced = createController({
    holdUntilVisible: true,
    reducedMotion: true,
    nowFn: () => 8000,
  });
  reduced.controller.onFirstPresent();
  assert.equal(reduced.controller.getStats().holdingForVisibleFrame, true);
  reduced.controller.notifyFirstVisibleFrame();
  assert.equal(reduced.timer.size, 0);
  assert.equal(reduced.controller.getStats().visible, false);
  assert.equal(reduced.controller.getStats().fadeDurationMs, 0);
  assert.equal(reduced.controller.getStats().dismissReason, DISMISS_REASONS.FIRST_VISIBLE_FRAME);
  assert.equal(reduced.dismissCalls.length, 1);
  reduced.controller.destroy();
});

test('W230: holdFallbackTimeoutMs safety cap prevents infinite splash hang, 0 disables, and destroy/dismiss cancel it', () => {
  // 1) Custom fallback timeout fires smooth fade-out if notifyFirstVisibleFrame is never called.
  let now = 10000;
  const fallback = createController({
    holdUntilVisible: true,
    holdFallbackTimeoutMs: 15000,
    nowFn: () => now,
  });
  fallback.controller.onFirstPresent();
  fallback.controller.setLoadingAssetsProgress({ presents: 45, realPackDraws: 2 });
  assert.deepEqual(fallback.timer.delays(), [15000]);
  assert.equal(fallback.dismissCalls.length, 0);

  // Trigger the fallback timeout.
  now = 25000;
  fallback.timer.flush();
  assert.equal(fallback.controller.getStats().holdFallbackTimedOut, true);
  assert.equal(fallback.controller.getStats().holdingForVisibleFrame, false);
  assert.equal(fallback.controller.getStats().heldForVisibleFrame, true);
  assert.equal(fallback.controller.getStats().fading, true);
  assert.equal(fallback.controller.getStats().visible, true);
  assert.deepEqual(fallback.timer.delays(), [450]);

  // Complete fade transition.
  now = 25450;
  fallback.timer.flush();
  const stats = fallback.controller.getStats();
  assert.equal(stats.visible, false);
  assert.equal(stats.dismissed, true);
  assert.equal(stats.dismissReason, DISMISS_REASONS.HOLD_FALLBACK_TIMEOUT);
  assert.equal(stats.holdFallbackTimedOut, true);
  assert.equal(stats.firstVisibleFrameMs, null);
  assert.equal(stats.blackPresentsWhileHeld, 45);
  assert.equal(fallback.dismissCalls.length, 1);
  fallback.controller.destroy();

  // 2) holdFallbackTimeoutMs: 0 disables fallback timer completely.
  const disabled = createController({
    holdUntilVisible: true,
    holdFallbackTimeoutMs: 0,
  });
  disabled.controller.onFirstPresent();
  disabled.controller.setLoadingAssetsProgress({ presents: 10 });
  assert.equal(disabled.timer.size, 0);
  assert.equal(disabled.controller.getStats().holdFallbackTimeoutMs, 0);
  assert.equal(disabled.controller.getStats().holdingForVisibleFrame, true);
  disabled.controller.destroy();

  // 3) destroy() cancels pending fallback timer cleanly.
  const destroyed = createController({
    holdUntilVisible: true,
    holdFallbackTimeoutMs: 5000,
  });
  destroyed.controller.onFirstPresent();
  assert.equal(destroyed.timer.size, 1);
  destroyed.controller.destroy();
  assert.equal(destroyed.timer.size, 0);
  destroyed.timer.flush();
  assert.equal(destroyed.dismissCalls.length, 0);

  // 4) Manual dismissImmediate() during hold cancels pending fallback timer cleanly.
  const manual = createController({
    holdUntilVisible: true,
    holdFallbackTimeoutMs: 5000,
    nowFn: () => 33000,
  });
  manual.controller.onFirstPresent();
  assert.equal(manual.timer.size, 1);
  manual.controller.dismissImmediate('user_skip');
  assert.equal(manual.timer.size, 0);
  assert.equal(manual.controller.getStats().visible, false);
  assert.equal(manual.controller.getStats().dismissReason, 'user_skip');
  assert.equal(manual.controller.getStats().heldForVisibleFrame, true);
  assert.equal(manual.controller.getStats().holdFallbackTimedOut, false);
  assert.equal(manual.dismissCalls.length, 1);
  manual.controller.destroy();
});

test('W252: Astryx-matched compact splash card uses #hud tokens, removes nested inner pill box, and replaces visible progress bar with live elapsed readout', () => {
  const doc = new MockDocument();
  const { controller } = createController({ documentObj: doc });
  const root = controller.getState().root;

  const panel = root.querySelector('.pg6-splash__panel');
  const statusRow = root.querySelector('.pg6-splash__status-row');
  const stageRow = root.querySelector('.pg6-splash__stage-row');
  const elapsedLabel = root.querySelector('.pg6-splash__elapsed-label');
  const elapsed = root.querySelector('.pg6-splash__elapsed');
  assert.ok(panel, '.pg6-splash__panel must exist');
  assert.ok(statusRow, '.pg6-splash__status-row must exist');
  assert.equal(statusRow, stageRow, '.pg6-splash__status-row and .pg6-splash__stage-row alias the same row');
  assert.ok(statusRow.querySelector('.pg6-splash__spinner'), 'spinner must be inside status row');
  assert.ok(statusRow.querySelector('.pg6-splash__spinner-ring'), 'spinner ring must be inside status row');
  assert.ok(statusRow.querySelector('.pg6-splash__stage'), 'stage text must be inside status row');
  assert.equal(root.querySelector('.pg6-splash__bar'), null, 'visible progress bar must be removed');
  assert.equal(root.querySelector('.pg6-splash__bar-fill'), null, 'visible progress bar fill must be removed');
  assert.equal(root.querySelector('.pg6-splash__pct'), null, 'visible percentage readout must be removed');
  assert.ok(elapsedLabel, '.pg6-splash__elapsed-label must exist');
  assert.equal(elapsedLabel.textContent, 'Elapsed');
  assert.ok(elapsed, '.pg6-splash__elapsed must exist');
  assert.equal(elapsed.getAttribute('role'), 'timer');
  assert.equal(elapsed.textContent, '00:00');

  // Verify shared stylesheet matches #hud Astryx tokens with zero nested inner pill box and zero neon blue.
  const styleEl = doc.head.children.find((el) => el.getAttribute('data-pg6-splash-style') === '1');
  assert.ok(styleEl);
  const css = styleEl.textContent;

  const panelRuleMatch = /\.pg6-splash__panel\{([^}]+)\}/.exec(css);
  assert.ok(panelRuleMatch, 'SHARED_CSS must define .pg6-splash__panel');
  const panelCss = panelRuleMatch[1];
  assert.match(panelCss, /width:\s*min\(340px,\s*calc\(100vw - 24px\)\)/);
  assert.match(panelCss, /background:\s*var\(--color-background-surface/);
  assert.match(panelCss, /border:\s*none/);
  assert.match(panelCss, /border-radius:\s*var\(--radius-container/);
  assert.match(panelCss, /box-shadow:\s*var\(--shadow-high/);
  assert.match(panelCss, /backdrop-filter:\s*blur\(18px\)/);
  assert.match(panelCss, /-webkit-backdrop-filter:\s*blur\(18px\)/);
  assert.match(panelCss, /padding:\s*12px 14px/);
  assert.match(panelCss, /gap:\s*8px/);

  const rowRuleMatch = /\.pg6-splash__stage-row,\.pg6-splash__status-row\{([^}]+)\}/.exec(css);
  assert.ok(rowRuleMatch, 'SHARED_CSS must style .pg6-splash__stage-row,.pg6-splash__status-row');
  const rowCss = rowRuleMatch[1];
  assert.match(rowCss, /background:\s*transparent/);
  assert.match(rowCss, /border:\s*none/);
  assert.match(rowCss, /padding:\s*0/);
  assert.match(rowCss, /box-shadow:\s*none/);
  assert.match(rowCss, /backdrop-filter:\s*none/);

  // Monochrome Astryx spinner (14px) and non-spinning hero styling.
  assert.match(css, /\.pg6-splash__spinner-ring\{[^}]*width:14px;height:14px[^}]*border:2px solid var\(--color-track[^}]*border-top-color:var\(--color-accent/);
  assert.match(css, /\.pg6-splash__greg\{[^}]*animation:none;/);
  assert.match(css, /\.pg6-splash__elapsed-label\{/);
  assert.match(css, /\.pg6-splash__elapsed\{[^}]*font-variant-numeric:tabular-nums;/);
  assert.doesNotMatch(css, /#6db3ff/i, 'SHARED_CSS must not use legacy neon blue #6db3ff');

  // Telemetry progressPct still tracks boot stages cleanly even without a visible progress bar.
  controller.updateBootState({ stage: 'D3D12_CREATE_DEVICE', progressPct: 78 });
  assert.equal(controller.getStats().progressPct, 78);
  controller.updateBootState({ stage: 'FIRST_PRESENT', progressPct: 100, readyForSplash: true });
  assert.equal(controller.getStats().progressPct, 100);
  controller.show({ reset: true });
  assert.equal(controller.getStats().progressPct, 0);
  controller.destroy();

  // While holding for visible frame (W230), progressPct stays capped at 99 until notifyFirstVisibleFrame().
  const held = createController({ holdUntilVisible: true });
  held.controller.onFirstPresent();
  assert.equal(held.controller.getStats().progressPct, 90);
  held.controller.notifyFirstVisibleFrame({ fadeDurationMs: 0 });
  assert.equal(held.controller.getStats().progressPct, 100);
  held.controller.destroy();
});

test('W311: exports monotonic phase constants, DEFAULT_FADE_OUT_MS (220), and phase clamping helpers on module and UMD global', () => {
  assert.ok(Object.isFrozen(BOOT_PHASE_ORDER));
  assert.ok(Object.isFrozen(BOOT_PHASE_RANGES));
  assert.deepEqual(BOOT_PHASE_ORDER, [
    'PGIM_LOAD',
    'WASM_INIT',
    'D3D12_INIT',
    'ASSET_STREAMING',
    'FIRST_VISIBLE_FRAME',
  ]);
  assert.equal(DEFAULT_FADE_OUT_MS, 220);
  assert.equal(typeof resolveBootPhase, 'function');
  assert.equal(typeof clampMonotonicPhaseProgress, 'function');
  assert.equal(globalThis.PlayGta6SplashOverlay.BOOT_PHASE_ORDER, BOOT_PHASE_ORDER);
  assert.equal(globalThis.PlayGta6SplashOverlay.BOOT_PHASE_RANGES, BOOT_PHASE_RANGES);
  assert.equal(globalThis.PlayGta6SplashOverlay.DEFAULT_FADE_OUT_MS, 220);
  assert.equal(globalThis.PlayGta6SplashOverlay.resolveBootPhase, resolveBootPhase);
  assert.equal(
    globalThis.PlayGta6SplashOverlay.clampMonotonicPhaseProgress,
    clampMonotonicPhaseProgress
  );

  // Stage-to-phase resolution covers both canonical phases and fine-grained BootTracer milestones.
  assert.equal(resolveBootPhase('PGIM_LOAD'), 'PGIM_LOAD');
  assert.equal(resolveBootPhase('WASM_INIT'), 'WASM_INIT');
  assert.equal(resolveBootPhase('TLS_INIT'), 'WASM_INIT');
  assert.equal(resolveBootPhase('CRT_INITTERM'), 'WASM_INIT');
  assert.equal(resolveBootPhase('ENTRY_CALL'), 'WASM_INIT');
  assert.equal(resolveBootPhase('XMEM_INIT'), 'WASM_INIT');
  assert.equal(resolveBootPhase('GDK_INIT'), 'WASM_INIT');
  assert.equal(resolveBootPhase('D3D12_INIT'), 'D3D12_INIT');
  assert.equal(resolveBootPhase('D3D12_CREATE_DEVICE'), 'D3D12_INIT');
  assert.equal(resolveBootPhase('ASSET_STREAMING'), 'ASSET_STREAMING');
  assert.equal(resolveBootPhase('BINK_OPEN'), 'ASSET_STREAMING');
  assert.equal(resolveBootPhase('FIRST_VISIBLE_FRAME'), 'FIRST_VISIBLE_FRAME');
  assert.equal(resolveBootPhase('FIRST_PRESENT'), 'FIRST_VISIBLE_FRAME');
  assert.equal(
    resolveBootPhase('FIRST_PRESENT', { holdingForVisibleFrame: true }),
    'ASSET_STREAMING'
  );
  assert.equal(resolveBootPhase('NOT_STARTED'), null);
  assert.equal(resolveBootPhase('UNKNOWN_STAGE'), null);

  // clampMonotonicPhaseProgress enforces per-phase bounds and never moves backwards.
  assert.equal(clampMonotonicPhaseProgress('PGIM_LOAD', 95, 0), 20);
  assert.equal(clampMonotonicPhaseProgress('WASM_INIT', 5, 11), 20);
  assert.equal(clampMonotonicPhaseProgress('WASM_INIT', 45, 20), 45);
  assert.equal(clampMonotonicPhaseProgress('PGIM_LOAD', 15, 45), 45);
  assert.equal(clampMonotonicPhaseProgress('D3D12_INIT', 99, 45), 85);
  assert.equal(clampMonotonicPhaseProgress('ASSET_STREAMING', 100, 85), 99);
  assert.equal(clampMonotonicPhaseProgress('FIRST_VISIBLE_FRAME', 100, 99), 100);
  assert.equal(
    clampMonotonicPhaseProgress('FIRST_VISIBLE_FRAME', 100, 95, { holdingForVisibleFrame: true }),
    99
  );
});

test('W311: monotonic phase-aware progress clamping prevents backwards jumps during parallel preloads and reaches 100% right before dismiss', () => {
  const { controller } = createController({ holdUntilVisible: true, fadeOutMs: 220 });
  const root = controller.getState().root;
  assert.equal(root.querySelector('.pg6-splash__bar'), null);
  assert.equal(root.querySelector('.pg6-splash__bar-fill'), null);

  // 1. PGIM_LOAD clamps upper bound to 20% even if a sub-loader reports 100%.
  let state = controller.updateBootState({ stage: 'PGIM_LOAD', progressPct: 12, elapsedMs: 100 });
  assert.equal(state.phase, 'PGIM_LOAD');
  assert.equal(state.progressPct, 12);
  state = controller.updateBootState({ stage: 'PGIM_LOAD', progressPct: 100, elapsedMs: 150 });
  assert.equal(state.progressPct, 20);

  // 2. WASM_INIT advances smoothly using phaseProgress interpolation and ignores lower values.
  state = controller.updateBootState({ stage: 'WASM_INIT', phaseProgress: 0.5, elapsedMs: 400 });
  assert.equal(state.phase, 'WASM_INIT');
  assert.equal(state.progressPct, Math.round(20 + (67 - 20) * 0.5));
  const wasmPct = state.progressPct;
  const wasmStatus = state.statusText;

  // Late parallel PGIM_LOAD update NEVER jumps progress, phase, or status text backwards.
  state = controller.updateBootState({ stage: 'PGIM_LOAD', progressPct: 15, elapsedMs: 120 });
  assert.equal(state.phase, 'WASM_INIT');
  assert.equal(state.progressPct, wasmPct);
  assert.equal(state.statusText, wasmStatus);
  assert.equal(state.elapsedMs, 400);

  // Lower progress within the same WASM_INIT phase also never jumps backwards.
  state = controller.updateBootState({ stage: 'WASM_INIT', progressPct: 25, elapsedMs: 450 });
  assert.equal(state.progressPct, wasmPct);

  // 3. D3D12_INIT clamps within [67, 85].
  state = controller.updateBootState({ stage: 'D3D12_INIT', progressPct: 10, elapsedMs: 700 });
  assert.equal(state.phase, 'D3D12_INIT');
  assert.equal(state.progressPct, 67);
  state = controller.updateBootState({ stage: 'D3D12_INIT', progressPct: 99, elapsedMs: 900 });
  assert.equal(state.progressPct, 85);

  // 4. ASSET_STREAMING holds cleanly within [85, 99] and never drops on lower setLoadingAssetsProgress pct.
  state = controller.onFirstPresent();
  assert.equal(state.phase, 'ASSET_STREAMING');
  assert.equal(state.progressPct, 90);

  state = controller.setLoadingAssetsProgress({ presents: 50, realPackDraws: 20, progressPct: 95 });
  assert.equal(state.progressPct, 95);

  // Out-of-order lower progressPct or late WASM_INIT never regresses progress.
  state = controller.setLoadingAssetsProgress({ presents: 60, realPackDraws: 21, progressPct: 40 });
  assert.equal(state.progressPct, 95);
  state = controller.updateBootState({ stage: 'WASM_INIT', progressPct: 30 });
  assert.equal(state.phase, 'ASSET_STREAMING');
  assert.equal(state.progressPct, 95);

  // Overshooting 100% during ASSET_STREAMING stays capped at 99% until first visible frame.
  state = controller.setLoadingAssetsProgress({ presents: 100, realPackDraws: 50, progressPct: 100 });
  assert.equal(state.progressPct, 99);

  // 5. First non-black frame smoothly reaches 100% right before fade-out dismiss.
  state = controller.notifyFirstVisibleFrame();
  assert.equal(state.phase, 'FIRST_VISIBLE_FRAME');
  assert.equal(state.progressPct, 100);
  assert.equal(state.fading, true);
  assert.equal(state.fadeOutCompleted, false);

  // Resetting via show({ reset: true }) allows a fresh boot starting cleanly from 0%.
  state = controller.show({ reset: true });
  assert.equal(state.phase, 'NOT_STARTED');
  assert.equal(state.progressPct, 0);
  assert.equal(state.fadeOutCompleted, false);
  state = controller.updateBootState({ stage: 'PGIM_LOAD', progressPct: 8 });
  assert.equal(state.phase, 'PGIM_LOAD');
  assert.equal(state.progressPct, 8);
  controller.destroy();
});

test('W311: smooth fade-out dismissal state (fadeOutMs default 220ms, instant fadeOutMs: 0) and leak-free idempotent hide()/destroy()', () => {
  // Constructor validation for fadeOutMs.
  assert.throws(
    () => new SplashOverlayController({ documentObj: new MockDocument(), fadeOutMs: -1 }),
    TypeError
  );
  assert.throws(
    () => new SplashOverlayController({ documentObj: new MockDocument(), fadeOutMs: Number.NaN }),
    TypeError
  );
  assert.throws(
    () => new SplashOverlayController({ documentObj: new MockDocument(), fadeOutMs: '220' }),
    TypeError
  );

  // Default fadeOutMs is 220 ms in stats/state.
  const defaultCtrl = createController();
  assert.equal(defaultCtrl.controller.getStats().fadeOutMs, 220);
  assert.equal(defaultCtrl.controller.getState().fadeOutMs, 220);
  assert.equal(defaultCtrl.controller.getStats().fadeOutCompleted, false);
  defaultCtrl.controller.destroy();

  // 1) Smooth 220 ms fade-out on first non-black frame transitions opacity cleanly and completes.
  let now = 1000;
  const smooth = createController({
    holdUntilVisible: true,
    fadeOutMs: 220,
    fadeMs: undefined,
    nowFn: () => now,
  });
  const root = smooth.controller.getState().root;
  assert.equal(root.style.getPropertyValue('--pg6-splash-fade'), '220ms');

  smooth.controller.onFirstPresent();
  now = 2500;
  const fadingState = smooth.controller.notifyFirstVisibleFrame();
  assert.equal(fadingState.visible, true);
  assert.equal(fadingState.dismissed, true);
  assert.equal(fadingState.fading, true);
  assert.equal(fadingState.fadeOutCompleted, false);
  assert.equal(fadingState.progressPct, 100);
  assert.equal(root.classList.contains('pg6-splash--fading'), true);
  assert.equal(root.style.opacity, '0');
  assert.equal(root.style.getPropertyValue('--pg6-splash-fade'), '220ms');
  assert.deepEqual(smooth.timer.delays(), [220]);
  assert.equal(smooth.dismissCalls.length, 0);

  now = 2720;
  smooth.timer.flush();
  const doneStats = smooth.controller.getStats();
  assert.equal(doneStats.visible, false);
  assert.equal(doneStats.fading, false);
  assert.equal(doneStats.dismissed, true);
  assert.equal(doneStats.fadeOutCompleted, true);
  assert.equal(doneStats.fadeDurationMs, 220);
  assert.equal(root.hidden, true);
  assert.equal(root.style.display, 'none');
  assert.equal(root.style.opacity, '');
  assert.equal(smooth.timer.size, 0);
  assert.equal(smooth.dismissCalls.length, 1);
  smooth.controller.destroy();

  // 2) Instant hide option fadeOutMs: 0 (constructor and per-call) hides synchronously with zero timers.
  const instantCtor = createController({
    holdUntilVisible: true,
    fadeOutMs: 0,
    fadeMs: undefined,
    nowFn: () => 5000,
  });
  assert.equal(instantCtor.controller.getStats().fadeOutMs, 0);
  assert.equal(
    instantCtor.controller.getState().root.style.getPropertyValue('--pg6-splash-fade'),
    '0ms'
  );
  instantCtor.controller.onFirstPresent();
  const instState = instantCtor.controller.notifyFirstVisibleFrame();
  assert.equal(instState.visible, false);
  assert.equal(instState.dismissed, true);
  assert.equal(instState.fading, false);
  assert.equal(instState.fadeOutCompleted, true);
  assert.equal(instState.fadeDurationMs, 0);
  assert.equal(instState.progressPct, 100);
  assert.equal(instantCtor.timer.size, 0);
  assert.equal(instantCtor.dismissCalls.length, 1);
  instantCtor.controller.destroy();

  const instantPerCall = createController({ holdUntilVisible: true, fadeOutMs: 220, nowFn: () => 6000 });
  instantPerCall.controller.onFirstPresent();
  instantPerCall.controller.notifyFirstVisibleFrame({ fadeOutMs: 0 });
  assert.equal(instantPerCall.timer.size, 0);
  assert.equal(instantPerCall.controller.getStats().visible, false);
  assert.equal(instantPerCall.controller.getStats().fadeOutCompleted, true);
  assert.equal(instantPerCall.controller.getStats().fadeDurationMs, 0);
  assert.equal(instantPerCall.dismissCalls.length, 1);
  instantPerCall.controller.destroy();

  // 3) hide({ fadeOutMs }) smooth fade + idempotent hide() / destroy() cancel all timers cleanly (zero leaks).
  const hideFade = createController({ fadeOutMs: 220 });
  hideFade.controller.hide({ fadeOutMs: 220 });
  assert.equal(hideFade.timer.size, 1);
  assert.deepEqual(hideFade.timer.delays(), [220]);
  assert.equal(hideFade.controller.getStats().fading, true);
  assert.equal(hideFade.controller.getStats().fadeOutCompleted, false);
  assert.equal(hideFade.controller.getState().root.style.opacity, '0');

  // Calling hide() again mid-fade cancels timer and hides synchronously without firing onDismiss.
  hideFade.controller.hide();
  assert.equal(hideFade.timer.size, 0);
  assert.equal(hideFade.controller.getStats().visible, false);
  assert.equal(hideFade.controller.getStats().fading, false);
  assert.equal(hideFade.dismissCalls.length, 0);
  // Repeated idempotent hide() stays clean.
  hideFade.controller.hide();
  hideFade.controller.hide();
  assert.equal(hideFade.timer.size, 0);
  hideFade.controller.destroy();

  // Completing hide({ fadeOutMs: 220 }) marks fadeOutCompleted without firing onDismiss.
  const hideFadeDone = createController({ fadeOutMs: 220 });
  hideFadeDone.controller.hide({ fadeOutMs: 220 });
  assert.equal(hideFadeDone.timer.size, 1);
  hideFadeDone.timer.flush();
  assert.equal(hideFadeDone.timer.size, 0);
  assert.equal(hideFadeDone.controller.getStats().visible, false);
  assert.equal(hideFadeDone.controller.getStats().fading, false);
  assert.equal(hideFadeDone.controller.getStats().fadeOutCompleted, true);
  assert.equal(hideFadeDone.dismissCalls.length, 0);
  hideFadeDone.controller.destroy();

  // destroy() mid-fade cancels all pending timers, detaches root, and is idempotent.
  const midDestroy = createController({ holdUntilVisible: true, fadeOutMs: 220, holdFallbackTimeoutMs: 5000 });
  midDestroy.controller.onFirstPresent();
  assert.equal(midDestroy.timer.size, 1);
  midDestroy.controller.notifyFirstVisibleFrame();
  assert.equal(midDestroy.timer.size, 1);
  assert.equal(midDestroy.controller.getStats().fading, true);
  midDestroy.controller.destroy();
  assert.equal(midDestroy.timer.size, 0);
  assert.equal(midDestroy.controller.getStats().destroyed, true);
  assert.equal(midDestroy.controller.getStats().fading, false);
  assert.equal(midDestroy.controller.getStats().fadeOutCompleted, false);
  // Idempotent repeated destroy & timer flush never throw or fire callbacks.
  assert.doesNotThrow(() => midDestroy.controller.destroy());
  midDestroy.timer.flush();
  assert.equal(midDestroy.dismissCalls.length, 0);
});

test('W311: surfaces sanitized phase timing telemetry (stageDurationsMs, totalSplashMs, fadeOutCompleted) and preserves W252 Astryx & Spinning Greg', () => {
  let now = 1000;
  const doc = new MockDocument();
  const { controller, timer } = createController({
    documentObj: doc,
    holdUntilVisible: true,
    fadeOutMs: 220,
    fadeMs: undefined,
    nowFn: () => now,
  });

  // Initial telemetry is clean and JSON-safe.
  const initStats = controller.getStats();
  assert.deepEqual(initStats.stageDurationsMs, {});
  assert.equal(initStats.totalSplashMs, 0);
  assert.equal(initStats.fadeOutCompleted, false);
  assert.equal(initStats.phase, 'NOT_STARTED');

  // Walk through all 5 phases with deterministic clock advances.
  now = 1100;
  controller.updateBootState({ stage: 'PGIM_LOAD', progressPct: 12 });

  now = 1450; // PGIM_LOAD = 350 ms
  controller.updateBootState({ stage: 'WASM_INIT', progressPct: 35 });

  // Stale parallel PGIM_LOAD event must NOT corrupt PGIM_LOAD duration or add unsanitized keys.
  now = 1600;
  controller.updateBootState({ stage: 'PGIM_LOAD', progressPct: 18 });
  controller.updateBootState({ stage: '../malicious/path', progressPct: 40 });

  now = 2000; // WASM_INIT = 550 ms
  controller.updateBootState({ stage: 'D3D12_INIT', progressPct: 75 });

  // Verify Greg Theft Auto hero is active by default mid-boot and toggles cleanly while preserving W252 Astryx DOM.
  const root = controller.getState().root;
  const heroImg = root.querySelector('.pg6-splash__img');
  const spinner = root.querySelector('.pg6-splash__spinner');
  const gregImg = root.querySelector('.pg6-splash__greg');
  assert.equal(controller.getStats().gregMode, true);
  assert.equal(heroImg.hidden, false);
  assert.equal(gregImg.hidden, true);
  spinner.dispatch('click');
  spinner.dispatch('click');
  spinner.dispatch('click');
  assert.equal(controller.getStats().gregMode, false);
  assert.equal(heroImg.hidden, true);
  spinner.dispatch('click');
  spinner.dispatch('click');
  spinner.dispatch('click');
  assert.equal(controller.getStats().gregMode, true);
  assert.equal(heroImg.hidden, false);

  now = 2300; // D3D12_INIT = 300 ms
  controller.onFirstPresent();
  now = 3100;
  controller.setLoadingAssetsProgress({ presents: 40, realPackDraws: 12 });

  // Live telemetry during ASSET_STREAMING includes elapsed duration so far (800 ms).
  const midStats = controller.getStats();
  assert.deepEqual(midStats.stageDurationsMs, {
    PGIM_LOAD: 350,
    WASM_INIT: 550,
    D3D12_INIT: 300,
    ASSET_STREAMING: 800,
  });
  assert.equal(midStats.totalSplashMs, 2100);
  assert.equal(midStats.fadeOutCompleted, false);

  // Mutating returned stageDurationsMs does not corrupt controller state.
  midStats.stageDurationsMs.PGIM_LOAD = 999999;
  midStats.stageDurationsMs.EVIL = 123;
  assert.equal(controller.getStats().stageDurationsMs.PGIM_LOAD, 350);
  assert.equal('EVIL' in controller.getStats().stageDurationsMs, false);

  // First non-black frame at t=3500 (ASSET_STREAMING = 1200 ms), fade finishes at t=3720 (FIRST_VISIBLE_FRAME = 220 ms).
  now = 3500;
  controller.notifyFirstVisibleFrame();
  assert.equal(controller.getStats().fadeOutCompleted, false);
  now = 3720;
  timer.flush();

  const finalStats = controller.getStats();
  const finalState = controller.getState();
  const expectedDurations = {
    PGIM_LOAD: 350,
    WASM_INIT: 550,
    D3D12_INIT: 300,
    ASSET_STREAMING: 1200,
    FIRST_VISIBLE_FRAME: 220,
  };
  assert.deepEqual(finalStats.stageDurationsMs, expectedDurations);
  assert.deepEqual(finalState.stageDurationsMs, expectedDurations);
  assert.equal(finalStats.totalSplashMs, 2720);
  assert.equal(finalState.totalSplashMs, 2720);
  assert.equal(finalStats.fadeOutCompleted, true);
  assert.equal(finalState.fadeOutCompleted, true);
  assert.equal(finalStats.gregMode, true);
  assert.doesNotThrow(() => JSON.stringify(finalStats));

  // Verify W252 Astryx stylesheet tokens remain intact.
  const styleEl = doc.head.children.find((el) => el.getAttribute('data-pg6-splash-style') === '1');
  assert.ok(styleEl);
  assert.match(styleEl.textContent, /var\(--color-background-surface/);
  assert.match(styleEl.textContent, /var\(--color-accent/);
  assert.match(styleEl.textContent, /var\(--color-success/);

  // show({ reset: true }) resets phase durations, totalSplashMs, and fadeOutCompleted for a fresh cycle.
  now = 5000;
  const restarted = controller.show({ reset: true });
  assert.deepEqual(restarted.stageDurationsMs, {});
  assert.equal(restarted.totalSplashMs, 0);
  assert.equal(restarted.fadeOutCompleted, false);
  assert.equal(restarted.phase, 'NOT_STARTED');
  controller.destroy();

  // Faulty / backwards clock values are strictly sanitized to non-negative finite integers.
  let clockVal = 5000;
  const faulty = createController({ fadeOutMs: 0, fadeMs: undefined, nowFn: () => clockVal });
  faulty.controller.updateBootState({ stage: 'PGIM_LOAD', progressPct: 10 });
  clockVal = 1000; // clock jumped backwards
  faulty.controller.updateBootState({ stage: 'WASM_INIT', progressPct: 30 });
  clockVal = Number.NaN;
  faulty.controller.notifyFirstVisibleFrame();
  const safeStats = faulty.controller.getStats();
  assert.equal(safeStats.stageDurationsMs.PGIM_LOAD, 0);
  assert.equal(safeStats.stageDurationsMs.WASM_INIT, 0);
  assert.equal(safeStats.totalSplashMs, 0);
  assert.equal(safeStats.fadeOutCompleted, true);
  assert.doesNotThrow(() => JSON.stringify(safeStats));
  faulty.controller.destroy();
});

test('Greg Theft Auto hero banner renders assets/greg.png without spinning, keeps ring spinner, degrades gracefully on load error or minimal mocks, and honors reduced motion', () => {
  const doc = new MockDocument();
  const { controller } = createController({ documentObj: doc });
  const root = controller.getState().root;
  const heroImg = root.querySelector('.pg6-splash__img');
  const spinner = root.querySelector('.pg6-splash__spinner');
  const ring = root.querySelector('.pg6-splash__spinner-ring');
  const greg = root.querySelector('.pg6-splash__greg');

  // 1) Default hero displays Greg Theft Auto (./assets/greg.png) framed cleanly; bottom spinner shows monochrome ring and NEVER spins the banner.
  assert.ok(heroImg);
  assert.ok(spinner);
  assert.ok(ring);
  assert.ok(greg);
  assert.equal(heroImg.tagName, 'IMG');
  assert.equal(heroImg.getAttribute('src'), './assets/greg.png');
  assert.equal(heroImg.getAttribute('alt'), 'Greg Theft Auto');
  assert.equal(heroImg.getAttribute('draggable'), 'false');
  assert.equal(heroImg.hidden, false);
  assert.equal(heroImg.classList.contains('pg6-splash__img--greg'), true);
  assert.equal(spinner.getAttribute('aria-hidden'), 'true');
  assert.equal(spinner.classList.contains('pg6-splash__spinner--greg'), false);
  assert.equal(ring.hidden, false);
  assert.equal(ring.style.display, '');
  assert.equal(ring.style.animation, '');
  assert.equal(greg.hidden, true);
  assert.equal(greg.style.display, 'none');
  assert.equal(greg.style.animation, 'none');

  // Verify shared stylesheet & web/ui/overlay.css use subtle pulse (not 360-deg spin) on the hero banner and disable spin on .pg6-splash__greg.
  const styleEl = doc.head.children.find((el) => el.getAttribute('data-pg6-splash-style') === '1');
  assert.ok(styleEl);
  assert.match(styleEl.textContent, /\.pg6-splash__img\.pg6-splash__img--greg\{[^}]*aspect-ratio:2\/1[^}]*animation:pg6-greg-pulse 4800ms ease-in-out infinite;/);
  assert.match(styleEl.textContent, /@keyframes pg6-greg-pulse\{/);
  assert.match(styleEl.textContent, /\.pg6-splash__greg\{[^}]*animation:none;/);
  assert.match(styleEl.textContent, /\.pg6-splash\[data-reduced-motion="true"\] \.pg6-splash__img--greg\{animation:none!important;transform:none!important;\}/);

  const overlayCss = fs.readFileSync(path.join(REPO_ROOT, 'web', 'ui', 'overlay.css'), 'utf8');
  assert.match(overlayCss, /@keyframes\s+pg6-greg-pulse\s*\{[\s\S]*scale\(1\.012\)/);
  assert.match(
    overlayCss,
    /#splash-overlay-root\s+\.pg6-splash__greg\s*\{[\s\S]*display:\s*none;[\s\S]*animation:\s*none;/
  );
  assert.match(
    overlayCss,
    /\.pg6-splash\[data-reduced-motion="true"\]\s+\.pg6-splash__img--greg[\s\S]*animation:\s*none\s*!important;/
  );
  assert.match(
    overlayCss,
    /@media\s*\(\s*prefers-reduced-motion\s*:\s*reduce\s*\)[\s\S]*\.pg6-splash__img--greg[\s\S]*animation:\s*none\s*!important;/
  );

  // 2) If assets/greg.png fails to load on the hero img, degrades gracefully to fallback without throwing.
  assert.equal(heroImg.listenerCount('error'), 1);
  assert.doesNotThrow(() => heroImg.dispatch('error'));
  assert.equal(heroImg.hidden, true);
  assert.equal(spinner.getAttribute('data-greg-mode'), 'false');
  assert.equal(ring.hidden, false);

  // Error listeners are cleaned up on destroy.
  controller.destroy();
  assert.equal(heroImg.listenerCount('error'), 0);
  assert.equal(greg.listenerCount('error'), 0);

  // 3) Reduced motion stops pulse & ring spin while keeping Greg Theft Auto visible cleanly.
  const reduced = createController({
    matchMedia: (q) => ({ matches: q === REDUCED_MOTION_QUERY }),
  });
  const rRoot = reduced.controller.getState().root;
  const rHero = rRoot.querySelector('.pg6-splash__img');
  const rSpinner = rRoot.querySelector('.pg6-splash__spinner');
  const rRing = rRoot.querySelector('.pg6-splash__spinner-ring');
  assert.equal(rRoot.getAttribute('data-reduced-motion'), 'true');
  assert.equal(rSpinner.getAttribute('data-reduced-motion'), 'true');
  assert.equal(rSpinner.getAttribute('aria-hidden'), 'true');
  assert.equal(rHero.hidden, false);
  assert.equal(rHero.style.animation, 'none');
  assert.equal(rRing.style.animation, 'none');
  reduced.controller.destroy();

  // 4) Minimal DOM test mocks (omitting style/classList/addEventListener) degrade gracefully without throwing.
  const minimalDoc = {
    createElement(tag) {
      return {
        tagName: String(tag).toUpperCase(),
        className: '',
        hidden: false,
        textContent: '',
        innerHTML: '',
        children: [],
        style: { setProperty() {} },
        setAttribute() {},
        appendChild(child) {
          this.children.push(child);
          return child;
        },
      };
    },
  };
  assert.doesNotThrow(() => {
    const minCtrl = new SplashOverlayController({ documentObj: minimalDoc, autoMount: false });
    minCtrl.updateBootState({ stage: 'PGIM_LOAD', progressPct: 15 });
    minCtrl.dismiss({ fadeMs: 0 });
    minCtrl.destroy();
  });
});

test('default Greg Theft Auto hero backdrop replaces /game/xbs_splash.png with wide rounded hero framing on dark Astryx canvas', () => {
  assert.equal(typeof resolveDefaultGregMode, 'function');
  assert.equal(resolveDefaultGregMode(''), true);
  assert.equal(resolveDefaultGregMode(null), true);
  assert.equal(resolveDefaultGregMode('?greg=1'), true);
  assert.equal(resolveDefaultGregMode('?greg=0'), false);
  assert.equal(resolveDefaultGregMode('?greg=false'), false);
  assert.equal(resolveDefaultGregMode('?greg=off'), false);
  assert.equal(resolveDefaultGregMode('?greg=no'), false);

  const doc = new MockDocument();
  const { controller } = createController({
    documentObj: doc,
    splashImageUrl: '/game/xbs_splash.png',
  });

  const root = controller.getState().root;
  const media = root.querySelector('.pg6-splash__media');
  const img = root.querySelector('.pg6-splash__img');
  const fallback = root.querySelector('.pg6-splash__fallback');
  const spinner = root.querySelector('.pg6-splash__spinner');
  const ring = root.querySelector('.pg6-splash__spinner-ring');
  const gregSpinner = root.querySelector('.pg6-splash__greg');

  // 1) Initial mount uses Greg Theft Auto hero backdrop and monochrome ring spinner in the bottom Astryx card.
  const initial = controller.getState();
  assert.equal(initial.gregMode, true);
  assert.equal(initial.gregBackdropActive, true);
  assert.equal(initial.splashImageUrl, '/game/xbs_splash.png');
  assert.equal(initial.activeSplashImageUrl, './assets/greg.png');
  assert.equal(initial.usingFallback, false);
  assert.equal(controller.getStats().gregBackdropActive, true);
  assert.equal(root.getAttribute('data-greg-backdrop'), 'true');
  assert.equal(media.classList.contains('pg6-splash__media--greg'), true);
  assert.equal(media.getAttribute('data-greg-backdrop'), 'true');
  assert.equal(img.getAttribute('src'), './assets/greg.png');
  assert.equal(img.getAttribute('alt'), 'Greg Theft Auto');
  assert.equal(img.getAttribute('data-greg-backdrop'), 'true');
  assert.equal(img.classList.contains('pg6-splash__img--greg'), true);
  assert.equal(img.style.objectFit, 'contain');
  assert.equal(img.hidden, false);
  assert.equal(fallback.hidden, true);
  assert.equal(spinner.getAttribute('data-greg-mode'), 'true');
  assert.equal(ring.hidden, false);
  assert.equal(gregSpinner.hidden, true);

  // 2) Calling setSplashImageUrl() / setSplashImage() later stores original URL but keeps Greg Theft Auto hero active.
  const afterSet = controller.setSplashImageUrl('/game/Assets/xbs_splash.png');
  assert.equal(afterSet.splashImageUrl, '/game/Assets/xbs_splash.png');
  assert.equal(afterSet.activeSplashImageUrl, './assets/greg.png');
  assert.equal(afterSet.gregBackdropActive, true);
  assert.equal(img.getAttribute('src'), './assets/greg.png');
  assert.equal(img.style.objectFit, 'contain');
  assert.equal(img.classList.contains('pg6-splash__img--greg'), true);

  // 3) Verify SHARED_CSS and web/ui/overlay.css style the wide rounded Greg Theft Auto hero frame on a dark #0b0c10 vignette.
  const styleEl = doc.head.children.find((el) => el.getAttribute('data-pg6-splash-style') === '1');
  assert.ok(styleEl);
  assert.match(styleEl.textContent, /\.pg6-splash\[data-greg-backdrop="true"\]\{background:#0b0c10;\}/);
  assert.match(styleEl.textContent, /\.pg6-splash__media--greg\{[^}]*radial-gradient\([^}]*#0b0c10/);
  assert.match(
    styleEl.textContent,
    /\.pg6-splash__img\.pg6-splash__img--greg\{[^}]*width:min\(1040px,92vw\)[^}]*aspect-ratio:2\/1;object-fit:contain;[^}]*border-radius:var\(--radius-container,16px\);/
  );

  const overlayCss = fs.readFileSync(path.join(REPO_ROOT, 'web', 'ui', 'overlay.css'), 'utf8');
  assert.match(overlayCss, /#splash-overlay-root\s+\.pg6-splash\[data-greg-backdrop="true"\][\s\S]*#0b0c10/);
  assert.match(overlayCss, /#splash-overlay-root\s+\.pg6-splash__media--greg[\s\S]*radial-gradient/);
  assert.match(
    overlayCss,
    /#splash-overlay-root\s+\.pg6-splash__img\.pg6-splash__img--greg[\s\S]*width:\s*min\(1040px,\s*92vw\)[\s\S]*aspect-ratio:\s*2\s*\/\s*1;[\s\S]*object-fit:\s*contain;[\s\S]*border-radius:\s*var\(--radius-container\);/
  );

  controller.destroy();
});

test('opting out via ?greg=0 or gregMode: false renders original /game/xbs_splash.png with object-fit: cover', () => {
  // 1) Opt-out via ?greg=0 query string.
  const viaQuery = createController({
    locationSearch: '?greg=0',
    splashImageUrl: null,
  });
  const qRoot = viaQuery.controller.getState().root;
  const qImg = qRoot.querySelector('.pg6-splash__img');
  const qMedia = qRoot.querySelector('.pg6-splash__media');
  const qFallback = qRoot.querySelector('.pg6-splash__fallback');

  assert.equal(viaQuery.controller.getState().gregMode, false);
  assert.equal(viaQuery.controller.getState().gregBackdropActive, false);
  assert.equal(viaQuery.controller.getState().usingFallback, true);
  assert.equal(qImg.hidden, true);
  assert.equal(qFallback.hidden, false);

  viaQuery.controller.setSplashImageUrl('/game/xbs_splash.png');
  assert.equal(viaQuery.controller.getState().splashImageUrl, '/game/xbs_splash.png');
  assert.equal(viaQuery.controller.getState().activeSplashImageUrl, '/game/xbs_splash.png');
  assert.equal(viaQuery.controller.getState().gregBackdropActive, false);
  assert.equal(viaQuery.controller.getState().usingFallback, false);
  assert.equal(qRoot.getAttribute('data-greg-backdrop'), 'false');
  assert.equal(qMedia.classList.contains('pg6-splash__media--greg'), false);
  assert.equal(qImg.getAttribute('src'), '/game/xbs_splash.png');
  assert.equal(qImg.getAttribute('alt'), 'Local splash');
  assert.equal(qImg.classList.contains('pg6-splash__img--greg'), false);
  assert.equal(qImg.style.objectFit, 'cover');
  assert.equal(qImg.hidden, false);
  assert.equal(qFallback.hidden, true);
  viaQuery.controller.destroy();

  // 2) Opt-out via constructor gregMode: false.
  const viaOpt = createController({
    gregMode: false,
    splashImageUrl: '/game/xbs_splash.png',
  });
  const oRoot = viaOpt.controller.getState().root;
  const oImg = oRoot.querySelector('.pg6-splash__img');
  assert.equal(viaOpt.controller.getState().gregBackdropActive, false);
  assert.equal(oImg.getAttribute('src'), '/game/xbs_splash.png');
  assert.equal(oImg.style.objectFit, 'cover');
  assert.equal(oImg.classList.contains('pg6-splash__img--greg'), false);
  viaOpt.controller.destroy();
});

test('toggling Greg mode switches cleanly between Greg backdrop and original splash URL or SVG fallback', () => {
  const { controller, documentObj } = createController({ splashImageUrl: null });
  const root = controller.getState().root;
  const media = root.querySelector('.pg6-splash__media');
  const img = root.querySelector('.pg6-splash__img');
  const fallback = root.querySelector('.pg6-splash__fallback');
  const spinner = root.querySelector('.pg6-splash__spinner');

  // Late discovery of /game/xbs_splash.png while Greg mode is ON.
  controller.setSplashImageUrl('/game/xbs_splash.png');
  assert.equal(img.getAttribute('src'), './assets/greg.png');
  assert.equal(img.style.objectFit, 'contain');
  assert.equal(controller.getState().gregBackdropActive, true);

  // 3x click toggles Greg OFF -> immediately restores /game/xbs_splash.png with cover fit.
  spinner.dispatch('click');
  spinner.dispatch('click');
  spinner.dispatch('click');
  assert.equal(controller.getState().gregMode, false);
  assert.equal(controller.getState().gregBackdropActive, false);
  assert.equal(controller.getState().activeSplashImageUrl, '/game/xbs_splash.png');
  assert.equal(img.getAttribute('src'), '/game/xbs_splash.png');
  assert.equal(img.style.objectFit, 'cover');
  assert.equal(img.classList.contains('pg6-splash__img--greg'), false);
  assert.equal(media.classList.contains('pg6-splash__media--greg'), false);
  assert.equal(root.getAttribute('data-greg-backdrop'), 'false');
  assert.equal(img.hidden, false);
  assert.equal(fallback.hidden, true);

  // Typing "greg" toggles Greg back ON -> restores centered Greg backdrop.
  for (const ch of 'greg') {
    documentObj.dispatch('keydown', { key: ch });
  }
  assert.equal(controller.getState().gregMode, true);
  assert.equal(controller.getState().gregBackdropActive, true);
  assert.equal(controller.getState().activeSplashImageUrl, './assets/greg.png');
  assert.equal(img.getAttribute('src'), './assets/greg.png');
  assert.equal(img.style.objectFit, 'contain');
  assert.equal(img.classList.contains('pg6-splash__img--greg'), true);
  assert.equal(media.classList.contains('pg6-splash__media--greg'), true);
  assert.equal(root.getAttribute('data-greg-backdrop'), 'true');

  // Clearing original splashImageUrl and toggling Greg OFF falls back cleanly to built-in SVG.
  controller.setSplashImageUrl(null);
  assert.equal(img.getAttribute('src'), './assets/greg.png');
  controller.setGregMode(false);
  assert.equal(controller.getState().gregBackdropActive, false);
  assert.equal(controller.getState().usingFallback, true);
  assert.equal(controller.getState().activeSplashImageUrl, null);
  assert.equal(img.hidden, true);
  assert.equal(fallback.hidden, false);

  // Toggling back ON brings Greg backdrop back.
  controller.toggleGregMode();
  assert.equal(controller.getState().gregBackdropActive, true);
  assert.equal(controller.getState().usingFallback, false);
  assert.equal(img.hidden, false);
  assert.equal(img.getAttribute('src'), './assets/greg.png');
  assert.equal(fallback.hidden, true);
  controller.destroy();
});

test('Greg backdrop degrades gracefully when ./assets/greg.png fails to load', () => {
  // 1) With a stored original splashImageUrl, Greg load error falls back to /game/xbs_splash.png,
  //    and a second error falls back to built-in CSS/SVG.
  const withOriginal = createController({ splashImageUrl: '/game/xbs_splash.png' });
  const wRoot = withOriginal.controller.getState().root;
  const wImg = wRoot.querySelector('.pg6-splash__img');
  const wMedia = wRoot.querySelector('.pg6-splash__media');
  const wFallback = wRoot.querySelector('.pg6-splash__fallback');

  assert.equal(wImg.getAttribute('src'), './assets/greg.png');
  assert.doesNotThrow(() => wImg.dispatch('error'));

  let state = withOriginal.controller.getState();
  assert.equal(state.gregBackdropFailed, true);
  assert.equal(state.gregBackdropActive, false);
  assert.equal(state.imageFailed, false);
  assert.equal(state.usingFallback, false);
  assert.equal(state.activeSplashImageUrl, '/game/xbs_splash.png');
  assert.equal(wImg.getAttribute('src'), '/game/xbs_splash.png');
  assert.equal(wImg.style.objectFit, 'cover');
  assert.equal(wImg.classList.contains('pg6-splash__img--greg'), false);
  assert.equal(wMedia.classList.contains('pg6-splash__media--greg'), false);
  assert.equal(wImg.hidden, false);
  assert.equal(wFallback.hidden, true);

  // If the fallback /game/xbs_splash.png also fails, degrade cleanly to built-in SVG.
  assert.doesNotThrow(() => wImg.dispatch('error'));
  state = withOriginal.controller.getState();
  assert.equal(state.imageFailed, true);
  assert.equal(state.usingFallback, true);
  assert.equal(state.activeSplashImageUrl, null);
  assert.equal(wImg.hidden, true);
  assert.equal(wFallback.hidden, false);
  withOriginal.controller.destroy();

  // 2) Without a stored original splashImageUrl, Greg load error falls back immediately to SVG.
  const withoutOriginal = createController({ splashImageUrl: null });
  const nRoot = withoutOriginal.controller.getState().root;
  const nImg = nRoot.querySelector('.pg6-splash__img');
  const nFallback = nRoot.querySelector('.pg6-splash__fallback');

  assert.equal(nImg.getAttribute('src'), './assets/greg.png');
  assert.doesNotThrow(() => nImg.dispatch('error'));
  const nState = withoutOriginal.controller.getState();
  assert.equal(nState.gregBackdropFailed, true);
  assert.equal(nState.gregBackdropActive, false);
  assert.equal(nState.usingFallback, true);
  assert.equal(nImg.hidden, true);
  assert.equal(nFallback.hidden, false);
  withoutOriginal.controller.destroy();
});

test('Greg Theft Auto asset dimensions and live elapsed-time clock replace 99%-stuck progress bar with zero leaked timers', () => {
  // 1) Verify web/assets/greg.png is a valid 2:1 widescreen PNG (1280x640) and compact (< 512 KiB).
  const gregPath = path.join(REPO_ROOT, 'web', 'assets', 'greg.png');
  const pngBytes = fs.readFileSync(gregPath);
  assert.ok(pngBytes.length > 0 && pngBytes.length < 512 * 1024);
  const width = pngBytes.readUInt32BE(16);
  const height = pngBytes.readUInt32BE(20);
  assert.equal(width, 1280);
  assert.equal(height, 640);

  // 2) formatElapsedClock and DEFAULT_ELAPSED_INTERVAL_MS exports.
  assert.equal(typeof formatElapsedClock, 'function');
  assert.equal(DEFAULT_ELAPSED_INTERVAL_MS, 250);
  assert.equal(globalThis.PlayGta6SplashOverlay.formatElapsedClock, formatElapsedClock);
  assert.equal(
    globalThis.PlayGta6SplashOverlay.DEFAULT_ELAPSED_INTERVAL_MS,
    DEFAULT_ELAPSED_INTERVAL_MS
  );
  assert.equal(formatElapsedClock(0), '00:00');
  assert.equal(formatElapsedClock(999), '00:00');
  assert.equal(formatElapsedClock(1000), '00:01');
  assert.equal(formatElapsedClock(42500), '00:42');
  assert.equal(formatElapsedClock(65000), '01:05');
  assert.equal(formatElapsedClock(3661000), '1:01:01');
  assert.equal(formatElapsedClock(-500), '00:00');
  assert.equal(formatElapsedClock(Number.NaN), '00:00');
  assert.equal(formatElapsedClock(Number.POSITIVE_INFINITY), '00:00');

  // 3) Validate setInterval, clearInterval, and elapsedIntervalMs options.
  assert.throws(
    () => new SplashOverlayController({ documentObj: new MockDocument(), setInterval: 123 }),
    TypeError
  );
  assert.throws(
    () => new SplashOverlayController({ documentObj: new MockDocument(), clearInterval: 123 }),
    TypeError
  );
  assert.throws(
    () => new SplashOverlayController({ documentObj: new MockDocument(), elapsedIntervalMs: -10 }),
    TypeError
  );

  // 4) Live elapsed timer ticks smoothly while visible (even with no boot-stage updates) and stops immediately on dismiss/hide/destroy.
  let nextIntervalId = 1;
  const activeIntervals = new Map();
  const fakeIntervals = {
    setInterval(fn, ms) {
      const id = nextIntervalId++;
      activeIntervals.set(id, { fn, ms });
      return id;
    },
    clearInterval(id) {
      activeIntervals.delete(id);
    },
    tick() {
      for (const entry of Array.from(activeIntervals.values())) {
        entry.fn();
      }
    },
  };

  let now = 10000;
  const { controller, timer } = createController({
    holdUntilVisible: true,
    fadeOutMs: 220,
    nowFn: () => now,
    setInterval: fakeIntervals.setInterval,
    clearInterval: fakeIntervals.clearInterval,
    elapsedIntervalMs: 250,
  });

  const root = controller.getState().root;
  const elapsedEl = root.querySelector('.pg6-splash__elapsed');
  assert.equal(root.querySelector('.pg6-splash__bar'), null);
  assert.equal(root.querySelector('.pg6-splash__pct'), null);
  assert.equal(activeIntervals.size, 1);
  assert.equal(elapsedEl.textContent, '00:00');
  assert.equal(controller.getStats().elapsedText, '00:00');

  // Simulate long 99%-held boot with zero progress updates: wall clock still ticks cleanly!
  controller.setStage('ASSET_STREAMING', 'Loading game assets...');
  controller.updateProgress(99);
  assert.equal(controller.getStats().progressPct, 99);
  assert.equal(root.querySelector('.pg6-splash__stage').textContent, 'Loading game assets...');

  now = 52400; // +42.4s
  fakeIntervals.tick();
  assert.equal(elapsedEl.textContent, '00:42');
  assert.equal(elapsedEl.getAttribute('aria-label'), 'Elapsed: 00:42');
  assert.equal(controller.getStats().liveElapsedMs, 42400);
  assert.equal(controller.getStats().elapsedText, '00:42');

  now = 75000; // +65s
  fakeIntervals.tick();
  assert.equal(elapsedEl.textContent, '01:05');

  // hide() stops the interval immediately; show() resumes it cleanly.
  controller.hide();
  assert.equal(activeIntervals.size, 0);
  controller.show();
  assert.equal(activeIntervals.size, 1);

  // dismiss() stops the interval immediately (before fade completes) and freezes final elapsed time.
  now = 80000; // +70s
  controller.dismiss({ fadeOutMs: 220 });
  assert.equal(activeIntervals.size, 0);
  assert.equal(elapsedEl.textContent, '01:10');
  now = 80220;
  timer.flush();
  assert.equal(activeIntervals.size, 0);
  assert.equal(controller.getStats().totalSplashMs, 70220);

  // show({ reset: true }) resets elapsed time to 00:00 and restarts interval.
  now = 90000;
  controller.show({ reset: true });
  assert.equal(activeIntervals.size, 1);
  assert.equal(elapsedEl.textContent, '00:00');
  now = 93000;
  fakeIntervals.tick();
  assert.equal(elapsedEl.textContent, '00:03');

  // destroy() clears active interval cleanly with zero leaks.
  controller.destroy();
  assert.equal(activeIntervals.size, 0);

  // 5) ?hud=0 and visible: false do not start background intervals.
  const hud0 = createController({
    locationSearch: '?hud=0',
    setInterval: fakeIntervals.setInterval,
    clearInterval: fakeIntervals.clearInterval,
  });
  assert.equal(activeIntervals.size, 0);
  hud0.controller.destroy();

  const hiddenInit = createController({
    visible: false,
    setInterval: fakeIntervals.setInterval,
    clearInterval: fakeIntervals.clearInterval,
  });
  assert.equal(activeIntervals.size, 0);
  hiddenInit.controller.destroy();

  const disabledInterval = createController({
    elapsedIntervalMs: 0,
    setInterval: fakeIntervals.setInterval,
    clearInterval: fakeIntervals.clearInterval,
  });
  assert.equal(activeIntervals.size, 0);
  disabledInterval.controller.destroy();
});

test('Orch-32 P1 #4a: isSplashHoldQuery and SplashOverlayController unify ?holdUntilVisible= and ?splashHold=', () => {
  for (const q of [
    '?holdUntilVisible=1',
    '?holdUntilVisible=true',
    '?holdUntilVisible=on',
    '?holdUntilVisible=yes',
    '?HOLDUNTILVISIBLE=TRUE',
    '?holdUntilVisible=0&splashHold=1',
  ]) {
    assert.equal(isSplashHoldQuery(q), true, `expected true for ${q}`);
  }
  for (const q of [
    '?holdUntilVisible=0',
    '?holdUntilVisible=false',
    '?holdUntilVisible=off',
    '?holdUntilVisible=no',
    '?splashHold=1&holdUntilVisible=0',
    '?splashHold=true&holdUntilVisible=off',
  ]) {
    assert.equal(isSplashHoldQuery(q), false, `expected false for ${q}`);
  }

  const viaHoldOn = createController({ locationSearch: '?holdUntilVisible=1' });
  assert.equal(viaHoldOn.controller.getStats().holdUntilVisible, true);
  viaHoldOn.controller.destroy();

  const viaHoldOff = createController({ locationSearch: '?splashHold=1&holdUntilVisible=0' });
  assert.equal(viaHoldOff.controller.getStats().holdUntilVisible, false);
  viaHoldOff.controller.destroy();
});
