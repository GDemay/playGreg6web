import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const splashMod = require('../web/ui/splash_overlay.js');

const {
  BOOT_STAGE_ORDER,
  BOOT_STAGE_LABELS,
  computeBootProgress,
  SplashOverlayController,
  DEFAULT_FADE_MS,
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
