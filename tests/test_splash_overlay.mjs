import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const splashMod = require('../web/ui/splash_overlay.js');

const {
  BOOT_STAGE_ORDER,
  BOOT_STAGE_LABELS,
  computeBootProgress,
  sanitizeScriptedInputPreset,
  SplashOverlayController,
  DEFAULT_FADE_MS,
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
  assert.equal(state.usingFallback, true);
  assert.ok(state.root);
  assert.equal(state.root.parentNode, documentObj.body);
  assert.equal(state.root.getAttribute('role'), 'status');
  assert.equal(state.root.getAttribute('aria-live'), 'polite');
  assert.equal(state.root.getAttribute('aria-busy'), 'true');
  controller.destroy();
});

test('binds splashImageUrl to img src when provided', () => {
  const { controller } = createController({
    splashImageUrl: '/game/xbs_splash.png',
  });
  const state = controller.getState();
  assert.equal(state.splashImageUrl, '/game/xbs_splash.png');
  assert.equal(state.usingFallback, false);
  const img = state.root.querySelector('.pg6-splash__img');
  assert.ok(img);
  assert.equal(img.getAttribute('src'), '/game/xbs_splash.png');
  assert.equal(img.hidden, false);
  const fallback = state.root.querySelector('.pg6-splash__fallback');
  assert.equal(fallback.hidden, true);
  controller.destroy();
});

test('image onerror falls back to built-in CSS/SVG without throwing', () => {
  const { controller } = createController({
    splashImageUrl: '/game/missing-splash.png',
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

test('null splashImageUrl uses fallback immediately', () => {
  const { controller } = createController({ splashImageUrl: null });
  const state = controller.getState();
  assert.equal(state.usingFallback, true);
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
  assert.equal(root.querySelector('.pg6-splash__pct').textContent, '78%');
  assert.equal(root.querySelector('.pg6-splash__bar-fill').style.width, '78%');
  assert.equal(root.querySelector('.pg6-splash__stream').hidden, false);
  assert.equal(root.querySelector('.pg6-splash__stream-fill').style.width, '50%');
  assert.equal(root.querySelector('.pg6-splash__bar').getAttribute('aria-valuenow'), '78');
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
