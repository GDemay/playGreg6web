import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const {
  SplashOverlayController,
  computeBootProgress,
  DISMISS_REASONS,
} = require('../web/ui/splash_overlay.js');

function createMockElement(tagName = 'div') {
  const attrs = new Map();
  const listeners = new Map();
  const children = [];
  const styleProps = new Map();
  const classSet = new Set();

  const el = {
    tagName: String(tagName).toUpperCase(),
    children,
    parentNode: null,
    hidden: false,
    textContent: '',
    innerHTML: '',
    _className: '',
    get className() {
      return this._className;
    },
    set className(val) {
      this._className = String(val || '');
      classSet.clear();
      for (const tok of this._className.split(/\s+/)) {
        if (tok) classSet.add(tok);
      }
    },
    classList: {
      add(cls) {
        classSet.add(cls);
        el._className = Array.from(classSet).join(' ');
      },
      remove(cls) {
        classSet.delete(cls);
        el._className = Array.from(classSet).join(' ');
      },
      contains(cls) {
        return classSet.has(cls);
      },
    },
    style: {
      display: '',
      opacity: '',
      width: '',
      objectFit: '',
      animation: '',
      backgroundColor: '',
      setProperty(k, v) {
        styleProps.set(k, String(v));
      },
      getPropertyValue(k) {
        return styleProps.get(k) || '';
      },
    },
    setAttribute(k, v) {
      attrs.set(k, String(v));
    },
    getAttribute(k) {
      return attrs.has(k) ? attrs.get(k) : null;
    },
    removeAttribute(k) {
      attrs.delete(k);
    },
    appendChild(child) {
      if (child && typeof child === 'object') {
        child.parentNode = el;
        children.push(child);
      }
      return child;
    },
    removeChild(child) {
      const idx = children.indexOf(child);
      if (idx >= 0) {
        children.splice(idx, 1);
        child.parentNode = null;
      }
      return child;
    },
    addEventListener(type, fn) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add(fn);
    },
    removeEventListener(type, fn) {
      if (listeners.has(type)) listeners.get(type).delete(fn);
    },
    dispatchEvent(type, event = {}) {
      const set = listeners.get(type);
      if (!set) return 0;
      let count = 0;
      for (const fn of Array.from(set)) {
        fn(event);
        count += 1;
      }
      return count;
    },
    listenerCount(type) {
      const set = listeners.get(type);
      return set ? set.size : 0;
    },
    querySelector(selector) {
      const matchClass = selector.startsWith('.') ? selector.slice(1) : null;
      const matchStyle = selector === 'style[data-pg6-splash-style]';
      const stack = [...children];
      while (stack.length > 0) {
        const node = stack.shift();
        if (matchClass && node.classList && node.classList.contains(matchClass)) {
          return node;
        }
        if (matchStyle && node.tagName === 'STYLE' && node.getAttribute('data-pg6-splash-style')) {
          return node;
        }
        if (Array.isArray(node.children)) {
          stack.push(...node.children);
        }
      }
      return null;
    },
  };
  return el;
}

function createMockDocument() {
  const head = createMockElement('head');
  const body = createMockElement('body');
  const docEl = createMockElement('html');
  docEl.appendChild(head);
  docEl.appendChild(body);
  return {
    head,
    body,
    documentElement: docEl,
    createElement(tag) {
      return createMockElement(tag);
    },
    querySelector(sel) {
      return docEl.querySelector(sel);
    },
    addEventListener(type, fn) {
      docEl.addEventListener(type, fn);
    },
    removeEventListener(type, fn) {
      docEl.removeEventListener(type, fn);
    },
    dispatchEvent(type, ev) {
      return docEl.dispatchEvent(type, ev);
    },
    listenerCount(type) {
      return docEl.listenerCount(type);
    },
  };
}

function createFakeTimers() {
  let nextId = 1;
  const pending = new Map();
  return {
    setTimeout(fn, ms) {
      const id = nextId++;
      pending.set(id, { fn, ms });
      return id;
    },
    clearTimeout(id) {
      pending.delete(id);
    },
    flushAll() {
      const items = Array.from(pending.entries());
      pending.clear();
      for (const [, item] of items) {
        item.fn();
      }
    },
    size() {
      return pending.size;
    },
  };
}

test('Console-clean deep-black backdrop replaces navy fallback gradient and aligns non-Greg root to #000', () => {
  const doc = createMockDocument();
  const controller = new SplashOverlayController({
    documentObj: doc,
    gregMode: false,
    splashImageUrl: null,
    elapsedIntervalMs: 0,
  });

  const root = controller.getState().root;
  const styleEl = doc.querySelector('style[data-pg6-splash-style]');
  const fallbackEl = root.querySelector('.pg6-splash__fallback');

  assert.ok(styleEl, 'shared splash style element should be mounted');
  assert.doesNotMatch(styleEl.textContent, /#0b1220|#1a2a3a|#20202c/i);
  assert.match(styleEl.textContent, /\.pg6-splash__fallback\{[^}]*background:#000;/);
  assert.ok(fallbackEl, 'fallback backdrop element should exist');
  assert.doesNotMatch(fallbackEl.innerHTML, /#0b1220|#132033|#1a2a3a|#8fa3bc/i);
  assert.match(fallbackEl.innerHTML, /#000000/);
  assert.match(fallbackEl.innerHTML, /playgta6web/);
  assert.equal(root.style.backgroundColor, '#000');

  // Enabling Greg backdrop clears the inline #000 override so Greg CSS backdrop rules apply cleanly.
  controller.setGregMode(true);
  assert.equal(root.style.backgroundColor, '');

  // Disabling Greg backdrop restores deep-black #000 alignment with #viewport.
  controller.setGregMode(false);
  assert.equal(root.style.backgroundColor, '#000');

  controller.destroy();
});

test('Monotonic progress never jumps backward (100% -> 99%) when FIRST_PRESENT snapshot precedes onFirstPresent under holdUntilVisible', () => {
  const doc = createMockDocument();
  const timers = createFakeTimers();
  let now = 1000;
  const controller = new SplashOverlayController({
    documentObj: doc,
    holdUntilVisible: true,
    nowFn: () => now,
    setTimeout: timers.setTimeout,
    clearTimeout: timers.clearTimeout,
    elapsedIntervalMs: 0,
  });

  // PGIM preload -> shader pack skipped -> straight into WASM_INIT -> D3D12_INIT
  const s1 = controller.updateBootState({ stage: 'PGIM_LOAD', progressPct: 18, elapsedMs: 250 });
  assert.equal(s1.progressPct, 18);
  assert.equal(s1.phase, 'PGIM_LOAD');

  // NaN progressPct falls back to stage/phase rather than stalling at 0
  assert.equal(computeBootProgress({ stage: 'D3D12_CREATE_DEVICE', progressPct: Number.NaN }), 78);

  const s2 = controller.updateBootState({ stage: 'WASM_INIT', progressPct: 45, elapsedMs: 600 });
  assert.equal(s2.progressPct, 45);
  assert.equal(s2.phase, 'WASM_INIT');

  // Parallel streaming update with omitted stage must NOT clobber stage, statusText, or elapsedMs back to NOT_STARTED / 0
  const sStream = controller.updateBootState({
    streamingProgress: { loadedBytes: 40, totalBytes: 100 },
  });
  assert.equal(sStream.stage, 'WASM_INIT');
  assert.equal(sStream.elapsedMs, 600);
  assert.equal(sStream.streamPct, 40);
  assert.equal(sStream.progressPct, 45);

  // Stale earlier phase update must NOT regress stage, progress, or streamPct
  const sStale = controller.updateBootState({
    stage: 'PGIM_LOAD',
    progressPct: 12,
    streamingProgress: { loadedBytes: 10, totalBytes: 100 },
    elapsedMs: 100,
  });
  assert.equal(sStale.stage, 'WASM_INIT');
  assert.equal(sStale.progressPct, 45);
  assert.equal(sStale.streamPct, 40);
  assert.equal(sStale.elapsedMs, 600);

  // BootTracer emits FIRST_PRESENT (progressPct: 100) BEFORE onFirstPresent() is invoked
  now = 2500;
  const sFirstPresentSnap = controller.updateBootState({
    stage: 'FIRST_PRESENT',
    progressPct: 100,
    elapsedMs: 1500,
  });
  assert.ok(sFirstPresentSnap.progressPct <= 99, 'must stay <= 99 while holding for visible frame');
  assert.equal(sFirstPresentSnap.phase, 'ASSET_STREAMING');

  // Calling onFirstPresent() after FIRST_PRESENT snapshot must NEVER jump backward
  const sHold = controller.onFirstPresent();
  assert.ok(sHold.progressPct >= sFirstPresentSnap.progressPct);
  assert.ok(sHold.progressPct <= 99);
  assert.equal(sHold.holdingForVisibleFrame, true);
  assert.equal(sHold.phase, 'ASSET_STREAMING');

  // Real visible frame lifts smoothly to 100% and fades out
  now = 4000;
  const sVisible = controller.notifyFirstVisibleFrame({ fadeOutMs: 220 });
  assert.equal(sVisible.progressPct, 100);
  assert.equal(sVisible.phase, 'FIRST_VISIBLE_FRAME');
  assert.equal(sVisible.fading, true);

  // Spinner remains in layout (no abrupt layout snap) with animation paused during fade
  const spinner = sVisible.root.querySelector('.pg6-splash__spinner');
  const ring = sVisible.root.querySelector('.pg6-splash__spinner-ring');
  assert.equal(spinner.hidden, false);
  assert.equal(ring.style.animation, 'none');

  // Once fade completes, spinner hides and overlay is hidden with zero active timers
  now = 4220;
  timers.flushAll();
  assert.equal(spinner.hidden, true);
  assert.equal(controller.getState().visible, false);
  assert.equal(controller.getStats().dismissReason, DISMISS_REASONS.FIRST_VISIBLE_FRAME);
  assert.equal(timers.size(), 0);

  controller.destroy();
});

test('Zero idle overhead after dismiss/hide: keydown listener detaches and re-attaches on show()', () => {
  const doc = createMockDocument();
  const win = createMockElement('window');
  const timers = createFakeTimers();

  const controller = new SplashOverlayController({
    documentObj: doc,
    windowObj: win,
    setTimeout: timers.setTimeout,
    clearTimeout: timers.clearTimeout,
    elapsedIntervalMs: 0,
  });

  // Active while splash is visible
  assert.equal(win.listenerCount('keydown'), 1);

  // Typing 'greg' toggles Greg mode while visible
  const beforeGreg = controller.getState().gregMode;
  for (const key of ['g', 'r', 'e', 'g']) {
    win.dispatchEvent('keydown', { key });
  }
  assert.equal(controller.getState().gregMode, !beforeGreg);

  // Dismissing detaches the global keydown listener immediately (0 in-game keypress overhead)
  controller.dismiss({ fadeOutMs: 220 });
  assert.equal(win.listenerCount('keydown'), 0);
  timers.flushAll();
  assert.equal(win.listenerCount('keydown'), 0);

  // Post-dismiss updateBootState / onFirstPresent do not mutate hidden stage text
  const stageEl = controller.getState().root.querySelector('.pg6-splash__stage');
  const frozenText = stageEl.textContent;
  controller.updateBootState({ stage: 'BINK_OPEN', statusText: 'Should not churn DOM' });
  controller.onFirstPresent();
  assert.equal(stageEl.textContent, frozenText);

  // Re-showing on restart re-attaches the keydown listener and clears frozen totalSplashMs
  controller.show();
  assert.equal(win.listenerCount('keydown'), 1);

  // hide() also detaches keydown cleanly
  controller.hide();
  assert.equal(win.listenerCount('keydown'), 0);

  controller.destroy();

  // ?hud=0 auto-dismissed on construction never attaches keydown at all
  const hud0 = new SplashOverlayController({
    documentObj: doc,
    windowObj: win,
    locationSearch: '?hud=0',
    elapsedIntervalMs: 0,
  });
  assert.equal(win.listenerCount('keydown'), 0);
  hud0.destroy();
});
