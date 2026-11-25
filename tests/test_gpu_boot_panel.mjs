import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const panelMod = require('../web/ui/gpu_boot_panel.js');
const fallbackRouterMod = require('../web/workers/shader_fallback_router.js');
import { BootTracer } from '../web/telemetry/boot_tracer.js';

const {
  DEFAULT_MAX_ROWS,
  DEFAULT_MAX_TOP_MISSING_SHADERS,
  MAX_GPU_ADAPTER_LENGTH,
  EXPORT_FORMAT,
  EXPORT_SCHEMA_VERSION,
  BOOT_STAGE_ORDER,
  STAGE_NAMES,
  ALLOWED_SWAPCHAIN_FORMATS,
  sanitizeSwapchainFormat,
  sanitizeGpuAdapter,
  sanitizeUnitRatio,
  sanitizePositiveInt,
  formatDurationMs,
  formatOnePercentLowFps,
  formatP99FrameMs,
  formatRatioPct,
  formatFrameJitterMs,
  formatMemoryMb,
  formatMailboxLanes,
  formatSummary,
  normalizeTopMissingShaders,
  formatTopMissingShaders,
  formatMissingShaderRow,
  buildMissingShaderExport,
  GpuBootPanelController,
  GpuBootPanel,
} = panelMod;

const { ShaderFallbackRouter, SHADER_STAGE } = fallbackRouterMod;

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
      display: '',
      width: '',
    };
    this._hidden = false;
    this._textContent = '';
    this.textContentWrites = 0;
    this.childMutations = 0;
    this.className = '';
    this.classList = new MockClassList(this);
    this._listeners = new Map();
    this.onclick = null;
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
    this.textContentWrites += 1;
    this._textContent = String(value ?? '');
  }

  setAttribute(name, value) {
    this.attributes.set(String(name), String(value));
    if (name === 'class') this.className = String(value);
    if (name === 'hidden') this._hidden = true;
  }

  getAttribute(name) {
    if (!this.attributes.has(String(name))) return null;
    return this.attributes.get(String(name));
  }

  removeAttribute(name) {
    this.attributes.delete(String(name));
    if (name === 'hidden') this._hidden = false;
  }

  appendChild(child) {
    if (child.parentNode && child.parentNode !== this) {
      child.parentNode.removeChild(child);
    }
    child.parentNode = this;
    this.children.push(child);
    this.childMutations += 1;
    return child;
  }

  removeChild(child) {
    const idx = this.children.indexOf(child);
    if (idx >= 0) {
      this.children.splice(idx, 1);
      child.parentNode = null;
      this.childMutations += 1;
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
      for (const handler of Array.from(set)) handler(event);
    }
    if (type === 'click' && typeof this.onclick === 'function') {
      this.onclick(event);
    }
  }

  querySelector(selector) {
    const matches = this.querySelectorAll(selector);
    return matches.length > 0 ? matches[0] : null;
  }

  querySelectorAll(selector) {
    const out = [];
    const match = (el) => {
      if (selector.startsWith('.')) {
        return el.classList.contains(selector.slice(1));
      }
      if (selector.startsWith('style[')) {
        const attr = selector.slice('style['.length, -1);
        return el.tagName === 'STYLE' && el.getAttribute(attr) !== null;
      }
      return el.tagName === selector.toUpperCase();
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
  }

  createElement(tag) {
    return new MockElement(tag, this);
  }

  querySelector(selector) {
    return this.head.querySelector(selector) || this.body.querySelector(selector);
  }

  getElementsByTagName(tag) {
    const upper = String(tag).toUpperCase();
    if (upper === 'HEAD') return [this.head];
    if (upper === 'BODY') return [this.body];
    return [];
  }
}

function createPanel(overrides = {}) {
  const documentObj = overrides.documentObj || new MockDocument();
  const container =
    overrides.container !== undefined ? overrides.container : documentObj.body;
  const exportCalls = [];
  const controller = new GpuBootPanelController({
    container,
    documentObj,
    now: () => Date.UTC(2026, 0, 1, 12, 0, 0),
    onExportMissingShaders:
      overrides.onExportMissingShaders !== undefined
        ? overrides.onExportMissingShaders
        : (jsonText, payload) => {
            exportCalls.push({ jsonText, payload });
          },
    ...overrides,
    documentObj,
    container,
  });
  return { controller, documentObj, container, exportCalls };
}

test('exports UMD global, helpers, constants, and GpuBootPanelController', () => {
  assert.equal(DEFAULT_MAX_ROWS, 32);
  assert.equal(EXPORT_FORMAT, 'playgta6-missing-shaders-v1');
  assert.equal(EXPORT_SCHEMA_VERSION, 1);
  assert.ok(Object.isFrozen(BOOT_STAGE_ORDER));
  assert.ok(Object.isFrozen(STAGE_NAMES));
  assert.equal(typeof formatMissingShaderRow, 'function');
  assert.equal(typeof buildMissingShaderExport, 'function');
  assert.equal(typeof GpuBootPanelController, 'function');
  assert.equal(globalThis.PlayGta6GpuBootPanel.GpuBootPanelController, GpuBootPanelController);
  assert.equal(globalThis.PlayGta6GpuBootPanel.formatMissingShaderRow, formatMissingShaderRow);
  assert.equal(globalThis.PlayGta6GpuBootPanel.buildMissingShaderExport, buildMissingShaderExport);
});

test('formatMissingShaderRow normalizes hashes, stages, and hit counts while rejecting invalid entries', () => {
  const r1 = formatMissingShaderRow({
    hash: 0xdeadbeefn,
    stage: 1,
    hitCount: 14,
    firstSeenMs: 12.5,
    lastSeenMs: 90,
  });
  assert.deepEqual(r1, {
    hash: '0x00000000deadbeef',
    stage: 'vertex',
    hits: 14,
    firstSeenMs: 12.5,
    lastSeenMs: 90,
    text: '0x00000000deadbeef | vertex | 14',
  });

  const r2 = formatMissingShaderRow({
    shaderHash: 'CAFEBABE',
    stage: 'PS',
    drawCount: 7,
  });
  assert.equal(r2.hash, '0x00000000cafebabe');
  assert.equal(r2.stage, 'fragment');
  assert.equal(r2.hits, 7);

  const r3 = formatMissingShaderRow({ hash: 255, stage: 'cs' });
  assert.equal(r3.hash, '0x00000000000000ff');
  assert.equal(r3.stage, 'compute');
  assert.equal(r3.hits, 1);

  const r4 = formatMissingShaderRow({ hash: '0x1', stage: 'C:\\Users\\alice\\shader.hlsl', hits: 0 });
  assert.equal(r4.stage, 'unknown');
  assert.equal(r4.hits, 0);

  // Invalid hashes, host paths, and raw byte buffers are rejected cleanly.
  assert.equal(formatMissingShaderRow(null), null);
  assert.equal(formatMissingShaderRow({}), null);
  assert.equal(formatMissingShaderRow({ hash: 'C:\\Users\\alice\\game.rpf', stage: 'vertex' }), null);
  assert.equal(formatMissingShaderRow({ hash: new Uint8Array([1, 2, 3, 4]), stage: 'fragment' }), null);
  assert.equal(formatMissingShaderRow({ hash: -1n, stage: 'compute' }), null);
});

test('buildMissingShaderExport deduplicates, sorts, and strips host paths and raw bytes', () => {
  const payload = buildMissingShaderExport({
    bootSummary: {
      stage: 'C:\\Users\\alice\\GTA6\\game.exe',
      milestonesReached: 4,
      progressPct: 44,
      readyForSplash: false,
      hostPath: '/home/alice/secret/game',
      rawDump: new Uint8Array([0xde, 0xad, 0xbe, 0xef]),
    },
    frameStats: {
      framesPresented: 3,
      droppedFrames: 1,
      localFile: 'file:///C:/Users/alice/frame.bin',
    },
    fallbackStats: {
      packHits: 15,
      fallbackHits: 3,
      builtinHits: 2,
      rawShaderBuffer: new ArrayBuffer(32),
    },
    missingShaders: [
      {
        hash: '0x10',
        stage: 'vs',
        hitCount: 2,
        sourcePath: 'C:\\Users\\alice\\shaders\\sky.dxil',
        bytecode: new Uint8Array([1, 2, 3, 4]),
      },
      {
        hash: 0x10n,
        stage: 'vertex',
        hits: 5,
        firstSeenMs: 10,
        lastSeenMs: 40,
      },
      {
        hash: '0x20',
        stage: 'fragment',
        hitCount: 12,
        wgslBytes: new Uint8Array([9, 8, 7]),
      },
      {
        hash: '/home/alice/not-a-hash',
        stage: 'compute',
        hits: 99,
      },
    ],
    exportedAt: Date.UTC(2026, 5, 1, 0, 0, 0),
  });

  assert.equal(payload.format, EXPORT_FORMAT);
  assert.equal(payload.schemaVersion, EXPORT_SCHEMA_VERSION);
  assert.equal(payload.exportedAt, '2026-06-01T00:00:00.000Z');
  assert.equal(payload.boot.stage, 'UNKNOWN');
  assert.equal(payload.gpu.framesPresented, 3);
  assert.equal(payload.gpu.droppedFrames, 1);
  assert.equal(payload.gpu.hitRatePct, 75);
  assert.equal(payload.missingShaderCount, 2);
  assert.deepEqual(
    payload.missingShaders.map((r) => [r.hash, r.stage, r.hits]),
    [
      ['0x0000000000000020', 'fragment', 12],
      ['0x0000000000000010', 'vertex', 7],
    ]
  );

  const json = JSON.stringify(payload);
  assert.doesNotMatch(json, /Users|alice|home|file:\/\/|bytecode|rawDump|wgslBytes|sourcePath/i);
});

test('constructor validates options and throws descriptive errors', () => {
  assert.throws(() => new GpuBootPanelController(null), TypeError);
  assert.throws(() => new GpuBootPanelController({ documentObj: {} }), TypeError);
  assert.throws(
    () => new GpuBootPanelController({ documentObj: new MockDocument(), maxRows: 0 }),
    RangeError
  );
  assert.throws(
    () => new GpuBootPanelController({ documentObj: new MockDocument(), maxRows: -4 }),
    RangeError
  );
  assert.throws(
    () => new GpuBootPanelController({ documentObj: new MockDocument(), maxRows: 2.5 }),
    RangeError
  );
  assert.throws(
    () =>
      new GpuBootPanelController({
        documentObj: new MockDocument(),
        onExportMissingShaders: 'not-a-fn',
      }),
    TypeError
  );
});

test('initial render mounts panel into container with accessible defaults and empty table', () => {
  const { controller, documentObj, container } = createPanel();
  const state = controller.getState();

  assert.equal(state.visible, true);
  assert.equal(state.collapsed, false);
  assert.equal(state.destroyed, false);
  assert.equal(state.maxRows, DEFAULT_MAX_ROWS);
  assert.equal(state.root.parentNode, container);
  assert.equal(state.boot.stage, 'NOT_STARTED');
  assert.equal(state.boot.milestonesReached, 0);
  assert.equal(state.boot.progressPct, 0);
  assert.equal(state.boot.readyForSplash, false);

  // Shared style injected once in head.
  assert.equal(documentObj.head.querySelectorAll('style[data-pg6-gpu-boot-panel-style]').length, 1);

  // Default empty-state table row and hidden overflow indicator.
  const root = state.root;
  assert.equal(root.querySelectorAll('.pg6-gpu-panel__empty-row').length, 1);
  assert.equal(root.querySelector('.pg6-gpu-panel__overflow').hidden, true);
  assert.equal(root.querySelector('.pg6-gpu-panel__hit-rate').textContent, '0.0%');
  assert.equal(root.querySelector('.pg6-gpu-panel__missing-header').textContent, 'Missing Shaders (0)');

  controller.destroy();
});

test('ARIA attributes are set properly on region, toggle button, progressbar, and table', () => {
  const { controller } = createPanel({ ariaLabel: 'Custom GPU Diagnostics' });
  const root = controller.getState().root;
  const toggleBtn = root.querySelector('.pg6-gpu-panel__toggle-btn');
  const exportBtn = root.querySelector('.pg6-gpu-panel__export-btn');
  const body = root.querySelector('.pg6-gpu-panel__body');
  const progressBar = root.querySelector('.pg6-gpu-panel__progress-bar');
  const table = root.querySelector('.pg6-gpu-panel__table');

  assert.equal(root.getAttribute('role'), 'region');
  assert.equal(root.getAttribute('aria-label'), 'Custom GPU Diagnostics');
  assert.equal(toggleBtn.getAttribute('aria-expanded'), 'true');
  assert.equal(toggleBtn.getAttribute('aria-controls'), body.getAttribute('id'));
  assert.equal(exportBtn.getAttribute('aria-label'), 'Export missing shaders JSON');
  assert.equal(progressBar.getAttribute('role'), 'progressbar');
  assert.equal(progressBar.getAttribute('aria-valuemin'), '0');
  assert.equal(progressBar.getAttribute('aria-valuemax'), '100');
  assert.equal(progressBar.getAttribute('aria-valuenow'), '0');
  assert.equal(table.getAttribute('aria-label'), 'Missing shaders');

  controller.destroy();
});

test('update({ bootSummary }) renders milestone, progressPct, milestonesReached, and readyForSplash', () => {
  const { controller } = createPanel();
  controller.update({
    bootSummary: {
      stage: 'D3D12_CREATE_DEVICE',
      milestonesReached: 7,
      progressPct: 78,
      readyForSplash: false,
    },
  });

  const root = controller.getState().root;
  assert.equal(root.querySelector('.pg6-gpu-panel__boot-stage').textContent, 'D3D12_CREATE_DEVICE');
  assert.equal(root.querySelector('.pg6-gpu-panel__boot-milestones').textContent, '7/9');
  assert.equal(root.querySelector('.pg6-gpu-panel__boot-progress').textContent, '78%');
  assert.equal(root.querySelector('.pg6-gpu-panel__boot-ready').textContent, 'Pending');
  assert.equal(root.querySelector('.pg6-gpu-panel__progress-bar').getAttribute('aria-valuenow'), '78');
  assert.equal(root.querySelector('.pg6-gpu-panel__progress-fill').style.width, '78%');

  // Derives progressPct, milestonesReached, and readyForSplash when omitted.
  controller.update({
    bootSummary: { stage: 'FIRST_PRESENT' },
  });
  const s2 = controller.getState();
  assert.equal(s2.boot.progressPct, 100);
  assert.equal(s2.boot.milestonesReached, 9);
  assert.equal(s2.boot.readyForSplash, true);
  assert.equal(root.querySelector('.pg6-gpu-panel__boot-ready').textContent, 'Ready');

  controller.destroy();
});

test('update({ frameStats, fallbackStats }) updates Frame-1 GPU counters and hitRatePct formatting', () => {
  const { controller } = createPanel();
  controller.update({
    frameStats: { framesPresented: 42, droppedFrames: 3 },
    fallbackStats: {
      packHits: 2,
      fallbackHits: 1,
      builtinHits: 0,
    },
  });

  const state = controller.getState();
  const root = state.root;
  assert.equal(state.frameStats.framesPresented, 42);
  assert.equal(state.frameStats.droppedFrames, 3);
  assert.equal(state.fallbackStats.packHits, 2);
  assert.equal(state.fallbackStats.fallbackHits, 1);
  assert.equal(state.fallbackStats.builtinHits, 0);
  assert.equal(state.fallbackStats.hitRatePct, 66.67);
  assert.equal(state.hitRateText, '66.7%');

  assert.equal(root.querySelector('.pg6-gpu-panel__frames-presented').textContent, '42');
  assert.equal(root.querySelector('.pg6-gpu-panel__dropped-frames').textContent, '3');
  assert.equal(root.querySelector('.pg6-gpu-panel__pack-hits').textContent, '2');
  assert.equal(root.querySelector('.pg6-gpu-panel__fallback-hits').textContent, '1');
  assert.equal(root.querySelector('.pg6-gpu-panel__builtin-hits').textContent, '0');
  assert.equal(root.querySelector('.pg6-gpu-panel__hit-rate').textContent, '66.7%');

  // Explicit hitRatePct overrides derived computation and clamps to [0, 100].
  controller.update({
    fallbackStats: { packHits: 10, fallbackHits: 0, builtinHits: 0, hitRatePct: 120 },
  });
  assert.equal(controller.getState().fallbackStats.hitRatePct, 100);
  assert.equal(root.querySelector('.pg6-gpu-panel__hit-rate').textContent, '100.0%');

  controller.destroy();
});

test('partial update() preserves previously populated telemetry sections', () => {
  const { controller } = createPanel();
  controller.update({
    bootSummary: { stage: 'GDK_INIT', milestonesReached: 6, progressPct: 67, readyForSplash: false },
    frameStats: { framesPresented: 1, droppedFrames: 0 },
  });

  controller.update({
    missingShaders: [{ hash: '0xabc', stage: 'vertex', hits: 4 }],
  });

  const state = controller.getState();
  assert.equal(state.boot.stage, 'GDK_INIT');
  assert.equal(state.frameStats.framesPresented, 1);
  assert.equal(state.totalMissingShaders, 1);
  assert.equal(state.missingShaders[0].hash, '0x0000000000000abc');

  controller.destroy();
});

test('missing-shader table deduplicates entries and sorts by hits descending then hash ascending', () => {
  const { controller } = createPanel();
  controller.update({
    missingShaders: [
      { hash: '0x0000000000000002', stage: 'fragment', hitCount: 3 },
      { hash: '0x1', stage: 'vs', hits: 5 },
      { hash: 1n, stage: 'vertex', hitCount: 4 }, // merges with 0x1 vertex -> 9
      { hash: '0x0000000000000003', stage: 'compute', hits: 9 }, // ties with 0x1 -> sorted after 0x1
      { hash: '0x1', stage: 'fragment', hits: 1 }, // distinct stage -> stays separate
    ],
  });

  const state = controller.getState();
  assert.equal(state.totalMissingShaders, 4);
  assert.deepEqual(
    state.missingShaders.map((r) => [r.hash, r.stage, r.hits]),
    [
      ['0x0000000000000001', 'vertex', 9],
      ['0x0000000000000003', 'compute', 9],
      ['0x0000000000000002', 'fragment', 3],
      ['0x0000000000000001', 'fragment', 1],
    ]
  );

  const rows = state.root.querySelectorAll('.pg6-gpu-panel__shader-row');
  assert.equal(rows.length, 4);
  assert.equal(rows[0].querySelector('.pg6-gpu-panel__cell-hash').textContent, '0x0000000000000001');
  assert.equal(rows[0].querySelector('.pg6-gpu-panel__cell-stage').textContent, 'vertex');
  assert.equal(rows[0].querySelector('.pg6-gpu-panel__cell-hits').textContent, '9');
  assert.equal(rows[3].querySelector('.pg6-gpu-panel__cell-hits').textContent, '1');

  controller.destroy();
});

test('missing-shader table caps rendered rows to maxRows and shows +N more overflow indicator', () => {
  const { controller } = createPanel({ maxRows: 3 });
  const entries = [];
  for (let i = 1; i <= 7; i += 1) {
    entries.push({ hash: BigInt(i), stage: 'fragment', hits: i * 10 });
  }
  controller.update({ missingShaders: entries });

  const state = controller.getState();
  const root = state.root;
  assert.equal(state.totalMissingShaders, 7);
  assert.equal(state.visibleMissingShaders.length, 3);
  assert.equal(state.overflowCount, 4);
  assert.equal(root.querySelectorAll('.pg6-gpu-panel__shader-row').length, 3);
  assert.equal(
    root.querySelectorAll('.pg6-gpu-panel__shader-row')[0].querySelector('.pg6-gpu-panel__cell-hits')
      .textContent,
    '70'
  );

  const overflowEl = root.querySelector('.pg6-gpu-panel__overflow');
  assert.equal(overflowEl.hidden, false);
  assert.equal(overflowEl.textContent, '+4 more');

  // Shrinking back within maxRows hides the overflow indicator.
  controller.update({ missingShaders: entries.slice(0, 2) });
  assert.equal(controller.getState().overflowCount, 0);
  assert.equal(overflowEl.hidden, true);
  assert.equal(overflowEl.textContent, '');

  controller.destroy();
});

test('export button and exportMissingShadersJson() invoke callback with full sanitized JSON', () => {
  const { controller, exportCalls } = createPanel({ maxRows: 2 });
  controller.update({
    bootSummary: { stage: 'BINK_OPEN', milestonesReached: 8, progressPct: 89, readyForSplash: false },
    frameStats: { framesPresented: 5, droppedFrames: 0 },
    fallbackStats: { packHits: 8, fallbackHits: 2, builtinHits: 0 },
    missingShaders: [
      { hash: '0xaa', stage: 'vertex', hits: 10, hostPath: 'C:\\Users\\secret\\a.bin' },
      { hash: '0xbb', stage: 'fragment', hits: 6, rawBytes: new Uint8Array([1, 2, 3]) },
      { hash: '0xcc', stage: 'compute', hits: 2 },
    ],
  });

  const exportBtn = controller.getState().root.querySelector('.pg6-gpu-panel__export-btn');
  exportBtn.dispatch('click');

  assert.equal(exportCalls.length, 1);
  const { jsonText, payload } = exportCalls[0];
  assert.deepEqual(JSON.parse(jsonText), payload);
  // Export contains all 3 deduplicated shaders even though table is capped at maxRows=2.
  assert.equal(payload.missingShaderCount, 3);
  assert.equal(payload.missingShaders.length, 3);
  assert.equal(payload.boot.stage, 'BINK_OPEN');
  assert.equal(payload.gpu.hitRatePct, 80);
  assert.doesNotMatch(jsonText, /Users|secret|hostPath|rawBytes/i);

  // Direct API call returns JSON string and invokes callback.
  const directJson = controller.exportMissingShadersJson();
  assert.equal(exportCalls.length, 2);
  assert.equal(JSON.parse(directJson).missingShaderCount, 3);
  assert.equal(controller.getState().exportCount, 2);

  controller.destroy();
});

test('toggle(), header toggle button, show(), and hide() update visibility and collapsed ARIA state', () => {
  const { controller } = createPanel();
  const root = controller.getState().root;
  const body = root.querySelector('.pg6-gpu-panel__body');
  const toggleBtn = root.querySelector('.pg6-gpu-panel__toggle-btn');

  assert.equal(controller.getState().collapsed, false);
  assert.equal(body.hidden, false);
  assert.equal(toggleBtn.getAttribute('aria-expanded'), 'true');
  assert.equal(root.getAttribute('data-collapsed'), 'false');

  // Toggle via method collapses body.
  controller.toggle();
  assert.equal(controller.getState().collapsed, true);
  assert.equal(body.hidden, true);
  assert.equal(toggleBtn.getAttribute('aria-expanded'), 'false');
  assert.equal(root.getAttribute('data-collapsed'), 'true');
  assert.equal(toggleBtn.textContent, 'Expand');

  // Toggle via button click expands body.
  toggleBtn.dispatch('click');
  assert.equal(controller.getState().collapsed, false);
  assert.equal(body.hidden, false);
  assert.equal(toggleBtn.getAttribute('aria-expanded'), 'true');
  assert.equal(toggleBtn.textContent, 'Collapse');

  // hide() / show() control panel visibility independently of collapsed state.
  controller.toggle(true);
  controller.hide();
  assert.equal(controller.getState().visible, false);
  assert.equal(controller.getState().collapsed, true);
  assert.equal(root.hidden, true);
  assert.equal(root.style.display, 'none');

  controller.show();
  assert.equal(controller.getState().visible, true);
  assert.equal(controller.getState().collapsed, true);
  assert.equal(root.hidden, false);
  assert.equal(root.style.display, '');

  controller.destroy();
});

test('error isolation prevents callback failures and malformed snapshots from throwing', () => {
  const { controller } = createPanel({
    onExportMissingShaders() {
      throw new Error('export callback exploded');
    },
  });

  // Throwing export callback must not throw out of exportMissingShadersJson() or click.
  let returnedJson = null;
  assert.doesNotThrow(() => {
    returnedJson = controller.exportMissingShadersJson();
  });
  assert.equal(JSON.parse(returnedJson).format, EXPORT_FORMAT);

  const exportBtn = controller.getState().root.querySelector('.pg6-gpu-panel__export-btn');
  assert.doesNotThrow(() => exportBtn.dispatch('click'));

  // Malformed snapshots and throwing property getters must never crash update().
  assert.doesNotThrow(() => controller.update(null));
  assert.doesNotThrow(() => controller.update(undefined));
  assert.doesNotThrow(() => controller.update('bad-input'));
  assert.doesNotThrow(() =>
    controller.update({
      get bootSummary() {
        throw new Error('getter fault');
      },
      missingShaders: [
        {
          get hash() {
            throw new Error('row getter fault');
          },
        },
        { hash: '0x42', stage: 'compute', hits: 3 },
      ],
    })
  );
  assert.equal(controller.getState().totalMissingShaders, 1);
  assert.equal(controller.getState().missingShaders[0].hash, '0x0000000000000042');

  controller.destroy();
});

test('destroy() is idempotent, removes DOM listeners and node, and guards post-destroy calls', () => {
  const { controller, container, exportCalls } = createPanel();
  const root = controller.getState().root;
  const toggleBtn = root.querySelector('.pg6-gpu-panel__toggle-btn');
  const exportBtn = root.querySelector('.pg6-gpu-panel__export-btn');

  assert.equal(root.parentNode, container);
  assert.equal(toggleBtn.listenerCount('click'), 1);
  assert.equal(exportBtn.listenerCount('click'), 1);

  controller.destroy();
  assert.equal(controller.getState().destroyed, true);
  assert.equal(controller.getState().visible, false);
  assert.equal(controller.getState().root, null);
  assert.equal(root.parentNode, null);
  assert.equal(toggleBtn.listenerCount('click'), 0);
  assert.equal(exportBtn.listenerCount('click'), 0);

  // Clicking detached buttons after destroy does nothing.
  toggleBtn.dispatch('click');
  exportBtn.dispatch('click');
  assert.equal(exportCalls.length, 0);

  // Idempotent destroy and safe post-destroy methods.
  assert.doesNotThrow(() => controller.destroy());
  assert.doesNotThrow(() => controller.update({ bootSummary: { stage: 'PGIM_LOAD' } }));
  assert.doesNotThrow(() => controller.toggle());
  assert.doesNotThrow(() => controller.show());
  assert.doesNotThrow(() => controller.hide());
  assert.doesNotThrow(() => controller.exportMissingShadersJson());
  assert.equal(exportCalls.length, 0);
});

test('integrates directly with live BootTracer and ShaderFallbackRouter snapshots', () => {
  let clockMs = 100;
  const tracer = new BootTracer({ now: () => clockMs, startMs: 100 });
  tracer.markMilestone('PGIM_LOAD');
  clockMs += 50;
  tracer.markMilestone('D3D12_CREATE_DEVICE');

  const pack = new Map([
    [
      '0x0000000000000001',
      '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }',
    ],
  ]);
  const router = new ShaderFallbackRouter({ packLoader: pack, now: () => clockMs });
  router.resolve(1n, SHADER_STAGE.VERTEX); // pack hit
  router.resolve(0xdeadn, SHADER_STAGE.FRAGMENT); // miss 1
  router.resolve(0xdeadn, SHADER_STAGE.FRAGMENT); // miss 2
  router.resolve(0xbeefn, SHADER_STAGE.COMPUTE); // miss 1

  const { controller, exportCalls } = createPanel();
  controller.update({
    bootSummary: tracer.getSummary(),
    missingShaders: router.getMissingReport(),
    fallbackStats: router.getStats(),
    frameStats: { framesPresented: 1, droppedFrames: 0 },
  });

  const state = controller.getState();
  assert.equal(state.boot.stage, 'D3D12_CREATE_DEVICE');
  assert.equal(state.boot.milestonesReached, 2);
  assert.equal(state.boot.progressPct, Math.round((7 / 9) * 100));
  assert.equal(state.fallbackStats.packHits, 1);
  assert.equal(state.fallbackStats.builtinHits, 3);
  assert.equal(state.fallbackStats.hitRatePct, 25);
  assert.equal(state.totalMissingShaders, 2);
  assert.equal(state.missingShaders[0].hash, '0x000000000000dead');
  assert.equal(state.missingShaders[0].hits, 2);
  assert.equal(state.missingShaders[1].hash, '0x000000000000beef');
  assert.equal(state.missingShaders[1].hits, 1);

  const json = controller.exportMissingShadersJson();
  assert.equal(exportCalls.length, 1);
  const parsed = JSON.parse(json);
  assert.equal(parsed.missingShaderCount, 2);
  assert.equal(parsed.gpu.hitRatePct, 25);

  controller.destroy();
});

test('W114 default/omitted compat and HTTPFS telemetry preserves exact backwards-compatible defaults', () => {
  assert.ok(Array.isArray(ALLOWED_SWAPCHAIN_FORMATS) && Object.isFrozen(ALLOWED_SWAPCHAIN_FORMATS));
  assert.equal(typeof sanitizeSwapchainFormat, 'function');

  const { controller } = createPanel();
  const state = controller.getState();
  const root = state.root;

  assert.deepEqual(state.compatStats, {
    bindGroupKindMismatches: 0,
    autoLayoutFallbacks: 0,
    bc3dFallbacks: 0,
    bc3dFallbackUploadsSkipped: 0,
    swapchainFormat: null,
    usesBlitFallback: false,
    blitCount: 0,
  });
  assert.deepEqual(state.ioStats, {
    pgimFullFetches: 0,
    slabCacheHits: 0,
    slabCacheMisses: 0,
    syncFastPathHits: 0,
  });

  // Accessible groups & default DOM values
  const compatSection = root.querySelector('.pg6-gpu-panel__compat');
  const ioSection = root.querySelector('.pg6-gpu-panel__io');
  assert.equal(compatSection.getAttribute('role'), 'group');
  assert.equal(compatSection.getAttribute('aria-label'), 'WebGPU Compatibility & Swapchain');
  assert.equal(ioSection.getAttribute('role'), 'group');
  assert.equal(ioSection.getAttribute('aria-label'), 'HTTPFS & Cache Telemetry');

  assert.equal(root.querySelector('.pg6-gpu-panel__bind-mismatches').textContent, '0');
  assert.equal(root.querySelector('.pg6-gpu-panel__auto-layout-fallbacks').textContent, '0');
  assert.equal(root.querySelector('.pg6-gpu-panel__bc3d-fallbacks').textContent, '0');
  assert.equal(root.querySelector('.pg6-gpu-panel__bc3d-uploads-skipped').textContent, '0');
  assert.equal(root.querySelector('.pg6-gpu-panel__swapchain-format').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__blit-fallback').textContent, 'Direct (0)');
  assert.equal(root.querySelector('.pg6-gpu-panel__pgim-fetches').textContent, '0');
  assert.equal(root.querySelector('.pg6-gpu-panel__slab-hits').textContent, '0');
  assert.equal(root.querySelector('.pg6-gpu-panel__slab-misses').textContent, '0');
  assert.equal(root.querySelector('.pg6-gpu-panel__sync-fast-hits').textContent, '0');

  // Export preserves format, schemaVersion, existing keys, and safe additive defaults.
  const exported = JSON.parse(controller.exportMissingShadersJson());
  assert.equal(exported.format, EXPORT_FORMAT);
  assert.equal(exported.schemaVersion, EXPORT_SCHEMA_VERSION);
  assert.equal(exported.gpu.framesPresented, 0);
  assert.equal(exported.gpu.bindGroupKindMismatches, 0);
  assert.equal(exported.gpu.autoLayoutFallbacks, 0);
  assert.equal(exported.gpu.bc3dFallbacks, 0);
  assert.equal(exported.gpu.bc3dFallbackUploadsSkipped, 0);
  assert.equal(exported.gpu.swapchainFormat, null);
  assert.equal(exported.gpu.usesBlitFallback, false);
  assert.equal(exported.gpu.blitCount, 0);
  assert.deepEqual(exported.io, {
    pgimFullFetches: 0,
    slabCacheHits: 0,
    slabCacheMisses: 0,
    syncFastPathHits: 0,
  });

  controller.destroy();
});

test('W114 update() ingests bind-group mismatches, 3D BC fallbacks, swapchain/blit, and HTTPFS cache metrics into DOM and JSON export', () => {
  const { controller, exportCalls } = createPanel();

  // Ingest via top-level fields and nested ioStats.
  controller.update({
    bindGroupKindMismatches: 4,
    autoLayoutFallbacks: 7,
    bc3dFallbacks: 3,
    bc3dFallbackUploadsSkipped: 11,
    swapchainFormat: 'RGB10A2UNORM',
    usesBlitFallback: true,
    blitCount: 25,
    ioStats: {
      pgimFullFetches: 2,
      slabCacheHits: 148,
      slabCacheMisses: 6,
      syncFastPathHits: 142,
    },
  });

  const state = controller.getState();
  const root = state.root;
  assert.deepEqual(state.compatStats, {
    bindGroupKindMismatches: 4,
    autoLayoutFallbacks: 7,
    bc3dFallbacks: 3,
    bc3dFallbackUploadsSkipped: 11,
    swapchainFormat: 'rgb10a2unorm',
    usesBlitFallback: true,
    blitCount: 25,
  });
  assert.deepEqual(state.ioStats, {
    pgimFullFetches: 2,
    slabCacheHits: 148,
    slabCacheMisses: 6,
    syncFastPathHits: 142,
  });

  assert.equal(root.querySelector('.pg6-gpu-panel__bind-mismatches').textContent, '4');
  assert.equal(root.querySelector('.pg6-gpu-panel__auto-layout-fallbacks').textContent, '7');
  assert.equal(root.querySelector('.pg6-gpu-panel__bc3d-fallbacks').textContent, '3');
  assert.equal(root.querySelector('.pg6-gpu-panel__bc3d-uploads-skipped').textContent, '11');
  assert.equal(root.querySelector('.pg6-gpu-panel__swapchain-format').textContent, 'rgb10a2unorm');
  assert.equal(root.querySelector('.pg6-gpu-panel__blit-fallback').textContent, 'Active (25)');
  assert.equal(
    root.querySelector('.pg6-gpu-panel__blit-fallback').getAttribute('data-uses-blit'),
    'true'
  );
  assert.equal(root.querySelector('.pg6-gpu-panel__pgim-fetches').textContent, '2');
  assert.equal(root.querySelector('.pg6-gpu-panel__slab-hits').textContent, '148');
  assert.equal(root.querySelector('.pg6-gpu-panel__slab-misses').textContent, '6');
  assert.equal(root.querySelector('.pg6-gpu-panel__sync-fast-hits').textContent, '142');

  // Partial update via worker sub-objects preserves unaffected fields.
  controller.update({
    renderStateStats: { bindGroupKindMismatches: 9, autoLayoutFallbacks: 12 },
    textureStats: { bc3dFallbacks: 5, bc3dFallbackUploadsSkipped: 19 },
    swapchainStats: { format: 'bgra8unorm', usesBlitFallback: false, blitCount: 25 },
  });

  const s2 = controller.getState();
  assert.equal(s2.compatStats.bindGroupKindMismatches, 9);
  assert.equal(s2.compatStats.autoLayoutFallbacks, 12);
  assert.equal(s2.compatStats.bc3dFallbacks, 5);
  assert.equal(s2.compatStats.bc3dFallbackUploadsSkipped, 19);
  assert.equal(s2.compatStats.swapchainFormat, 'bgra8unorm');
  assert.equal(s2.compatStats.usesBlitFallback, false);
  assert.equal(s2.compatStats.blitCount, 25);
  // HTTPFS counters preserved across partial GPU update.
  assert.equal(s2.ioStats.pgimFullFetches, 2);
  assert.equal(s2.ioStats.slabCacheHits, 148);
  assert.equal(root.querySelector('.pg6-gpu-panel__swapchain-format').textContent, 'bgra8unorm');
  assert.equal(root.querySelector('.pg6-gpu-panel__blit-fallback').textContent, 'Direct (25)');

  // Top-level HTTPFS fields also update selectively without wiping other counters.
  controller.update({
    pgimFullFetches: 3,
    syncFastPathHits: 200,
  });
  assert.equal(controller.getState().ioStats.pgimFullFetches, 3);
  assert.equal(controller.getState().ioStats.slabCacheHits, 148);
  assert.equal(controller.getState().ioStats.syncFastPathHits, 200);

  // Verify JSON export contains all sanitized additive fields alongside existing schema keys.
  const jsonText = controller.exportMissingShadersJson();
  assert.equal(exportCalls.length, 1);
  const exported = JSON.parse(jsonText);
  assert.equal(exported.format, EXPORT_FORMAT);
  assert.equal(exported.schemaVersion, EXPORT_SCHEMA_VERSION);
  assert.equal(exported.gpu.bindGroupKindMismatches, 9);
  assert.equal(exported.gpu.autoLayoutFallbacks, 12);
  assert.equal(exported.gpu.bc3dFallbacks, 5);
  assert.equal(exported.gpu.bc3dFallbackUploadsSkipped, 19);
  assert.equal(exported.gpu.swapchainFormat, 'bgra8unorm');
  assert.equal(exported.gpu.usesBlitFallback, false);
  assert.equal(exported.gpu.blitCount, 25);
  assert.deepEqual(exported.io, {
    pgimFullFetches: 3,
    slabCacheHits: 148,
    slabCacheMisses: 6,
    syncFastPathHits: 200,
  });

  // Explicit null reset clears compatStats / ioStats cleanly.
  controller.update({ compatStats: null, ioStats: null });
  assert.equal(controller.getState().compatStats.swapchainFormat, null);
  assert.equal(controller.getState().compatStats.bindGroupKindMismatches, 0);
  assert.equal(controller.getState().ioStats.slabCacheHits, 0);

  controller.destroy();
});

test('W114 malicious or overly long swapchainFormat strings and invalid counters are sanitized/rejected', () => {
  assert.equal(sanitizeSwapchainFormat('rgba8unorm'), 'rgba8unorm');
  assert.equal(sanitizeSwapchainFormat('  BGRA8UNORM  '), 'bgra8unorm');
  assert.equal(sanitizeSwapchainFormat('rgb10a2unorm'), 'rgb10a2unorm');
  assert.equal(sanitizeSwapchainFormat('C:\\Users\\alice\\GTA6\\game.exe'), null);
  assert.equal(sanitizeSwapchainFormat('/home/alice/secret/swapchain.bin'), null);
  assert.equal(sanitizeSwapchainFormat('file:///C:/Users/alice/rgba8unorm'), null);
  assert.equal(sanitizeSwapchainFormat('rgba8unorm; ../../etc/passwd'), null);
  assert.equal(sanitizeSwapchainFormat('a'.repeat(512)), null);
  assert.equal(sanitizeSwapchainFormat(new Uint8Array([1, 2, 3])), null);
  assert.equal(sanitizeSwapchainFormat(12345), null);
  assert.equal(sanitizeSwapchainFormat(null), null);

  const { controller } = createPanel();
  const root = controller.getState().root;

  // Set a valid format first, then pass a malicious host path — it must be rejected to null / '—'.
  controller.update({ swapchainFormat: 'rgba8unorm' });
  assert.equal(controller.getState().compatStats.swapchainFormat, 'rgba8unorm');
  assert.equal(root.querySelector('.pg6-gpu-panel__swapchain-format').textContent, 'rgba8unorm');

  controller.update({
    swapchainFormat: 'C:\\Users\\alice\\AppData\\Local\\GTA6\\secret_dump.rpf',
    bindGroupKindMismatches: -5,
    autoLayoutFallbacks: Number.NaN,
    bc3dFallbacks: 'not-a-number',
    bc3dFallbackUploadsSkipped: Infinity,
    blitCount: -99,
    ioStats: {
      pgimFullFetches: -1,
      slabCacheHits: 'C:\\Users\\alice\\slab',
      slabCacheMisses: 3.9,
      syncFastPathHits: undefined,
      hostPath: '/home/alice/game/main.pgim',
      rawBytes: new Uint8Array([0xca, 0xfe, 0xba, 0xbe]),
    },
  });

  const state = controller.getState();
  assert.equal(state.compatStats.swapchainFormat, null);
  assert.equal(state.compatStats.bindGroupKindMismatches, 0);
  assert.equal(state.compatStats.autoLayoutFallbacks, 0);
  assert.equal(state.compatStats.bc3dFallbacks, 0);
  assert.equal(state.compatStats.bc3dFallbackUploadsSkipped, 0);
  assert.equal(state.compatStats.blitCount, 0);
  assert.equal(state.ioStats.pgimFullFetches, 0);
  assert.equal(state.ioStats.slabCacheHits, 0);
  assert.equal(state.ioStats.slabCacheMisses, 3);
  assert.equal(state.ioStats.syncFastPathHits, 0);

  assert.equal(root.querySelector('.pg6-gpu-panel__swapchain-format').textContent, '—');

  const exportedJson = controller.exportMissingShadersJson();
  const exported = JSON.parse(exportedJson);
  assert.equal(exported.gpu.swapchainFormat, null);
  assert.doesNotMatch(
    exportedJson,
    /Users|alice|AppData|secret_dump|home|main\.pgim|rawBytes|hostPath/i
  );

  // Also test buildMissingShaderExport directly with malicious swapchainFormat & extra fields.
  const directExport = buildMissingShaderExport({
    swapchainFormat: '/var/private/users/bob/game.xvc',
    compatStats: {
      bindGroupKindMismatches: 2,
      secretDump: new Uint8Array([1, 2, 3]),
    },
    ioStats: {
      pgimFullFetches: 1,
      localFile: 'D:\\Games\\Secret\\file.rpf',
    },
    exportedAt: Date.UTC(2026, 6, 1, 0, 0, 0),
  });
  assert.equal(directExport.gpu.swapchainFormat, null);
  assert.equal(directExport.gpu.bindGroupKindMismatches, 2);
  assert.equal(directExport.io.pgimFullFetches, 1);
  assert.doesNotMatch(
    JSON.stringify(directExport),
    /private|bob|game\.xvc|secretDump|Games|Secret|file\.rpf/i
  );

  controller.destroy();
});

test('W212 exports perf telemetry formatters and defaults cleanly when fields are omitted', () => {
  assert.equal(DEFAULT_MAX_TOP_MISSING_SHADERS, 8);
  assert.equal(typeof formatOnePercentLowFps, 'function');
  assert.equal(typeof formatP99FrameMs, 'function');
  assert.equal(typeof normalizeTopMissingShaders, 'function');
  assert.equal(typeof formatTopMissingShaders, 'function');
  assert.equal(globalThis.PlayGta6GpuBootPanel.formatOnePercentLowFps, formatOnePercentLowFps);
  assert.equal(globalThis.PlayGta6GpuBootPanel.formatP99FrameMs, formatP99FrameMs);
  assert.equal(globalThis.PlayGta6GpuBootPanel.formatTopMissingShaders, formatTopMissingShaders);

  // Pure formatter edge cases
  assert.equal(formatOnePercentLowFps(58.44), '58.4 fps');
  assert.equal(formatOnePercentLowFps(0), '0.0 fps');
  assert.equal(formatOnePercentLowFps('59.95'), '60.0 fps');
  assert.equal(formatOnePercentLowFps(null), '—');
  assert.equal(formatOnePercentLowFps(undefined), '—');
  assert.equal(formatOnePercentLowFps(-1), '—');
  assert.equal(formatOnePercentLowFps(Number.NaN), '—');
  assert.equal(formatOnePercentLowFps(Infinity), '—');
  assert.equal(formatOnePercentLowFps(true), '—');
  assert.equal(formatOnePercentLowFps('C:\\Users\\alice\\60'), '—');

  assert.equal(formatP99FrameMs(16.666), '16.67 ms');
  assert.equal(formatP99FrameMs(0), '0.00 ms');
  assert.equal(formatP99FrameMs('8.333'), '8.33 ms');
  assert.equal(formatP99FrameMs(null), '—');
  assert.equal(formatP99FrameMs(-0.5), '—');
  assert.equal(formatP99FrameMs(Infinity), '—');
  assert.equal(formatP99FrameMs(false), '—');
  assert.equal(formatP99FrameMs('/home/alice/frame.bin'), '—');

  assert.equal(formatTopMissingShaders([]), '—');
  assert.equal(formatTopMissingShaders(null), '—');

  // Invalid maxTopMissingShaders option throws RangeError
  assert.throws(
    () => new GpuBootPanelController({ documentObj: new MockDocument(), maxTopMissingShaders: 0 }),
    RangeError
  );
  assert.throws(
    () => new GpuBootPanelController({ documentObj: new MockDocument(), maxTopMissingShaders: -2 }),
    RangeError
  );
  assert.throws(
    () => new GpuBootPanelController({ documentObj: new MockDocument(), maxTopMissingShaders: 1.5 }),
    RangeError
  );

  // Initial panel state & DOM defaults
  const { controller } = createPanel();
  const state = controller.getState();
  const root = state.root;

  assert.deepEqual(state.perfStats, {
    onePercentLowFps: null,
    p99FrameMs: null,
    zeroCopyUploads: 0,
    totalElidedStateCalls: 0,
    bindGroupCacheHits: 0,
    topMissingShaders: [],
    topMissingShaderOverflow: 0,
  });

  const perfSection = root.querySelector('.pg6-gpu-panel__perf');
  assert.equal(perfSection.getAttribute('role'), 'group');
  assert.equal(perfSection.getAttribute('aria-label'), 'Performance & GPU Telemetry');
  assert.equal(root.querySelector('.pg6-gpu-panel__one-percent-low-fps').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__p99-frame-ms').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__zero-copy-uploads').textContent, '0');
  assert.equal(root.querySelector('.pg6-gpu-panel__elided-state-calls').textContent, '0');
  assert.equal(root.querySelector('.pg6-gpu-panel__bind-group-cache-hits').textContent, '0');
  assert.equal(root.querySelector('.pg6-gpu-panel__top-missing-count').textContent, '0');
  assert.equal(root.querySelector('.pg6-gpu-panel__top-missing-shaders').textContent, '—');

  const exported = JSON.parse(controller.exportMissingShadersJson());
  assert.deepEqual(exported.perf, {
    onePercentLowFps: null,
    p99FrameMs: null,
    zeroCopyUploads: 0,
    totalElidedStateCalls: 0,
    bindGroupCacheHits: 0,
    topMissingShaders: [],
  });

  controller.destroy();
});

test('W212 update() formats 1%-low FPS, P99 frame time, 4K Bink2 zero-copy, state elision, BG cache hits, and capped top missing shaders', () => {
  const { controller } = createPanel({ maxTopMissingShaders: 3 });
  const root = controller.getState().root;

  // Top-level W212 telemetry fields
  controller.update({
    onePercentLowFps: 57.36,
    p99FrameMs: 17.456,
    zeroCopyUploads: 128,
    totalElidedStateCalls: 4096,
    bindGroupCacheHits: 2048,
    topMissingShaders: [
      { hash: '0x20', stage: 'fragment', hits: 9 },
      { hash: 0x10n, stage: 'vs', hitCount: 15 },
      { hash: '0x10', stage: 'vertex', count: 5 }, // merges with 0x10 vertex -> 20
      { hashHex: '0x30', stage: 'compute', count: 12 },
      { hash: '0x40', stage: 'fragment', hits: 4 },
      { hash: '0x50', stage: 'vertex', hits: 2 },
    ],
  });

  const state = controller.getState();
  assert.equal(state.perfStats.onePercentLowFps, 57.36);
  assert.equal(state.perfStats.p99FrameMs, 17.456);
  assert.equal(state.perfStats.zeroCopyUploads, 128);
  assert.equal(state.perfStats.totalElidedStateCalls, 4096);
  assert.equal(state.perfStats.bindGroupCacheHits, 2048);
  assert.equal(state.perfStats.topMissingShaders.length, 3);
  assert.equal(state.perfStats.topMissingShaderOverflow, 2);
  assert.deepEqual(
    state.perfStats.topMissingShaders.map((r) => [r.hash, r.stage, r.hits]),
    [
      ['0x0000000000000010', 'vertex', 20],
      ['0x0000000000000030', 'compute', 12],
      ['0x0000000000000020', 'fragment', 9],
    ]
  );

  assert.equal(root.querySelector('.pg6-gpu-panel__one-percent-low-fps').textContent, '57.4 fps');
  assert.equal(root.querySelector('.pg6-gpu-panel__p99-frame-ms').textContent, '17.46 ms');
  assert.equal(root.querySelector('.pg6-gpu-panel__zero-copy-uploads').textContent, '128');
  assert.equal(root.querySelector('.pg6-gpu-panel__elided-state-calls').textContent, '4096');
  assert.equal(root.querySelector('.pg6-gpu-panel__bind-group-cache-hits').textContent, '2048');
  assert.equal(root.querySelector('.pg6-gpu-panel__top-missing-count').textContent, '3');
  assert.equal(
    root.querySelector('.pg6-gpu-panel__top-missing-shaders').textContent,
    '0x0000000000000010 (20), 0x0000000000000030 (12), 0x0000000000000020 (9) (+2 more)'
  );

  // Partial update via nested worker/monitor sub-objects preserves untouched fields
  controller.update({
    frameStats: { framesPresented: 60, p99FrameTimeMs: 16.25 },
    binkStats: { directZeroCopyUploads: 256 },
    drawStats: {
      redundantPipelineSkips: 100,
      redundantBindGroupSkips: 200,
      redundantVertexBufferSkips: 50,
      redundantIndexBufferSkips: 25,
      bindGroupCacheHits: 3333,
    },
  });

  const s2 = controller.getState();
  assert.equal(s2.perfStats.onePercentLowFps, 57.36); // preserved
  assert.equal(s2.perfStats.p99FrameMs, 16.25); // updated from frameStats.p99FrameTimeMs
  assert.equal(s2.perfStats.zeroCopyUploads, 256); // updated from binkStats.directZeroCopyUploads
  assert.equal(s2.perfStats.totalElidedStateCalls, 375); // summed from redundant*Skips
  assert.equal(s2.perfStats.bindGroupCacheHits, 3333); // updated from drawStats
  assert.equal(s2.perfStats.topMissingShaders.length, 3); // preserved

  assert.equal(root.querySelector('.pg6-gpu-panel__p99-frame-ms').textContent, '16.25 ms');
  assert.equal(root.querySelector('.pg6-gpu-panel__zero-copy-uploads').textContent, '256');
  assert.equal(root.querySelector('.pg6-gpu-panel__elided-state-calls').textContent, '375');
  assert.equal(root.querySelector('.pg6-gpu-panel__bind-group-cache-hits').textContent, '3333');

  // Bare hash array preserves deduplicated order and formats cleanly without fake counts
  controller.update({
    topMissingShaders: ['0xbeef', '0x000000000000beef', '0xdead'],
  });
  assert.equal(
    root.querySelector('.pg6-gpu-panel__top-missing-shaders').textContent,
    '0x000000000000beef, 0x000000000000dead'
  );
  assert.equal(root.querySelector('.pg6-gpu-panel__top-missing-count').textContent, '2');

  // Explicit perfStats: null resets all perf telemetry fields cleanly
  controller.update({ perfStats: null });
  assert.equal(controller.getState().perfStats.onePercentLowFps, null);
  assert.equal(controller.getState().perfStats.zeroCopyUploads, 0);
  assert.equal(controller.getState().perfStats.topMissingShaders.length, 0);
  assert.equal(root.querySelector('.pg6-gpu-panel__one-percent-low-fps').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__top-missing-shaders').textContent, '—');

  controller.destroy();
});

test('W212 sanitizes hostile perf fields and strips host paths and raw bytes from topMissingShaders and export', () => {
  const { controller } = createPanel();
  const root = controller.getState().root;

  controller.update({
    onePercentLowFps: 'C:\\Users\\alice\\GTA6\\fps.txt',
    p99FrameMs: '/home/alice/secret/p99.bin',
    zeroCopyUploads: -12,
    totalElidedStateCalls: 'file:///C:/Users/alice/elided',
    bindGroupCacheHits: Number.NaN,
    topMissingShaders: [
      'C:\\Users\\alice\\shaders\\sky.dxil',
      '/home/alice/game/main.rpf',
      'file:///C:/secret/shader.wgsl',
      new Uint8Array([0xde, 0xad, 0xbe, 0xef]),
      {
        hash: '0xcafebabe',
        stage: 'C:\\Users\\alice\\bad_stage',
        hits: 7,
        sourcePath: 'D:\\Games\\GTA6\\shader_dump.bin',
        bytecode: new Uint8Array([1, 2, 3, 4]),
      },
      {
        hash: '/var/private/not-a-hash',
        stage: 'fragment',
        hits: 99,
      },
      {
        get hash() {
          throw new Error('hostile getter');
        },
      },
    ],
  });

  const state = controller.getState();
  assert.equal(state.perfStats.onePercentLowFps, null);
  assert.equal(state.perfStats.p99FrameMs, null);
  assert.equal(state.perfStats.zeroCopyUploads, 0);
  assert.equal(state.perfStats.totalElidedStateCalls, 0);
  assert.equal(state.perfStats.bindGroupCacheHits, 0);
  assert.deepEqual(state.perfStats.topMissingShaders, [
    { hash: '0x00000000cafebabe', stage: 'unknown', hits: 7 },
  ]);

  assert.equal(root.querySelector('.pg6-gpu-panel__one-percent-low-fps').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__p99-frame-ms').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__zero-copy-uploads').textContent, '0');
  assert.equal(root.querySelector('.pg6-gpu-panel__elided-state-calls').textContent, '0');
  assert.equal(root.querySelector('.pg6-gpu-panel__bind-group-cache-hits').textContent, '0');
  assert.equal(
    root.querySelector('.pg6-gpu-panel__top-missing-shaders').textContent,
    '0x00000000cafebabe (7)'
  );

  const exportedJson = controller.exportMissingShadersJson();
  assert.doesNotMatch(
    exportedJson,
    /Users|alice|home|secret|file:\/\/|sourcePath|bytecode|shader_dump|private/i
  );
  const parsed = JSON.parse(exportedJson);
  assert.deepEqual(parsed.perf.topMissingShaders, [
    { hash: '0x00000000cafebabe', stage: 'unknown', hits: 7 },
  ]);

  // Direct formatTopMissingShaders also rejects hostile paths
  assert.equal(
    formatTopMissingShaders(['C:\\Users\\alice\\a.bin', '/etc/passwd', '0x1234']),
    '0x0000000000001234'
  );

  controller.destroy();
});

test('W212 dirty-checked DOM updates skip redundant textContent writes and table rebuilds on unchanged telemetry', () => {
  const { controller } = createPanel();
  const root = controller.getState().root;

  const snapshot = {
    bootSummary: { stage: 'FIRST_PRESENT', milestonesReached: 9, progressPct: 100, readyForSplash: true },
    frameStats: { framesPresented: 120, droppedFrames: 1 },
    fallbackStats: { packHits: 90, fallbackHits: 8, builtinHits: 2 },
    onePercentLowFps: 59.1,
    p99FrameMs: 16.92,
    zeroCopyUploads: 44,
    totalElidedStateCalls: 812,
    bindGroupCacheHits: 512,
    topMissingShaders: [{ hash: '0xabc', stage: 'fragment', hits: 6 }],
    missingShaders: [{ hash: '0xabc', stage: 'fragment', hits: 6 }],
  };

  controller.update(snapshot);

  const fpsEl = root.querySelector('.pg6-gpu-panel__one-percent-low-fps');
  const p99El = root.querySelector('.pg6-gpu-panel__p99-frame-ms');
  const zeroCopyEl = root.querySelector('.pg6-gpu-panel__zero-copy-uploads');
  const elidedEl = root.querySelector('.pg6-gpu-panel__elided-state-calls');
  const bgHitsEl = root.querySelector('.pg6-gpu-panel__bind-group-cache-hits');
  const topMissingEl = root.querySelector('.pg6-gpu-panel__top-missing-shaders');
  const framesEl = root.querySelector('.pg6-gpu-panel__frames-presented');
  const tbodyEl = root.querySelector('.pg6-gpu-panel__table-body');

  const before = {
    fps: fpsEl.textContentWrites,
    p99: p99El.textContentWrites,
    zeroCopy: zeroCopyEl.textContentWrites,
    elided: elidedEl.textContentWrites,
    bgHits: bgHitsEl.textContentWrites,
    topMissing: topMissingEl.textContentWrites,
    frames: framesEl.textContentWrites,
    tbodyMutations: tbodyEl.childMutations,
  };

  // Repeated identical updates must produce zero DOM writes / table mutations.
  controller.update(snapshot);
  controller.update({ ...snapshot, topMissingShaders: [{ hash: 0xabcn, stage: 'ps', count: 6 }] });
  controller.update(snapshot);

  assert.equal(fpsEl.textContentWrites, before.fps);
  assert.equal(p99El.textContentWrites, before.p99);
  assert.equal(zeroCopyEl.textContentWrites, before.zeroCopy);
  assert.equal(elidedEl.textContentWrites, before.elided);
  assert.equal(bgHitsEl.textContentWrites, before.bgHits);
  assert.equal(topMissingEl.textContentWrites, before.topMissing);
  assert.equal(framesEl.textContentWrites, before.frames);
  assert.equal(tbodyEl.childMutations, before.tbodyMutations);

  // Updating a single perf counter must touch ONLY that counter's DOM element.
  controller.update({ zeroCopyUploads: 45 });
  assert.equal(zeroCopyEl.textContent, '45');
  assert.equal(zeroCopyEl.textContentWrites, before.zeroCopy + 1);
  assert.equal(fpsEl.textContentWrites, before.fps);
  assert.equal(p99El.textContentWrites, before.p99);
  assert.equal(elidedEl.textContentWrites, before.elided);
  assert.equal(bgHitsEl.textContentWrites, before.bgHits);
  assert.equal(topMissingEl.textContentWrites, before.topMissing);
  assert.equal(framesEl.textContentWrites, before.frames);
  assert.equal(tbodyEl.childMutations, before.tbodyMutations);

  // Changing formatted 1%-low FPS updates only fpsEl.
  controller.update({ onePercentLowFps: 58.2 });
  assert.equal(fpsEl.textContent, '58.2 fps');
  assert.equal(fpsEl.textContentWrites, before.fps + 1);
  assert.equal(zeroCopyEl.textContentWrites, before.zeroCopy + 1);
  assert.equal(p99El.textContentWrites, before.p99);

  controller.destroy();
});

test('W240 exports adapter/duration helpers, GpuBootPanel alias, and clean default state/DOM/snapshot', () => {
  assert.equal(MAX_GPU_ADAPTER_LENGTH, 96);
  assert.equal(typeof sanitizeGpuAdapter, 'function');
  assert.equal(typeof formatDurationMs, 'function');
  assert.equal(GpuBootPanel, GpuBootPanelController);
  assert.equal(globalThis.PlayGta6GpuBootPanel.MAX_GPU_ADAPTER_LENGTH, 96);
  assert.equal(globalThis.PlayGta6GpuBootPanel.sanitizeGpuAdapter, sanitizeGpuAdapter);
  assert.equal(globalThis.PlayGta6GpuBootPanel.formatDurationMs, formatDurationMs);
  assert.equal(globalThis.PlayGta6GpuBootPanel.GpuBootPanel, GpuBootPanelController);

  // Pure sanitizeGpuAdapter checks
  assert.equal(sanitizeGpuAdapter('NVIDIA / Turing'), 'nvidia / turing');
  assert.equal(sanitizeGpuAdapter('  Intel/Gen-12LP  '), 'intel / gen-12lp');
  assert.equal(sanitizeGpuAdapter({ vendor: 'AMD', architecture: 'RDNA2' }), 'amd / rdna2');
  assert.equal(sanitizeGpuAdapter({ vendor: 'Intel', arch: 'Xe-LPG' }), 'intel / xe-lpg');
  assert.equal(sanitizeGpuAdapter({ vendor: 'Apple' }), 'apple');
  assert.equal(sanitizeGpuAdapter({ architecture: 'Metal-3' }), 'metal-3');
  assert.equal(sanitizeGpuAdapter('amd / gfx10.3'), 'amd / gfx10.3');
  assert.equal(sanitizeGpuAdapter('\x1b[32mNVIDIA\x1b[0m / Turing\r\n\0'), 'nvidia / turing');
  assert.equal(
    sanitizeGpuAdapter('NVIDIA / Turing (C:\\Users\\alice\\driver.dll)'),
    'nvidia / turing'
  );

  // Max 96 chars enforced without trailing slash or whitespace
  const longAdapter = `nvidia / ${'a'.repeat(120)}`;
  const sanitizedLong = sanitizeGpuAdapter(longAdapter);
  assert.equal(sanitizedLong.length, 96);
  assert.ok(sanitizedLong.startsWith('nvidia / a'));
  assert.equal(sanitizeGpuAdapter(`${'a'.repeat(94)} / bbb`), 'a'.repeat(94));

  // Host paths, URLs, raw bytes, and invalid types are rejected cleanly
  assert.equal(sanitizeGpuAdapter('C:\\Users\\alice\\GTA6\\game.exe'), null);
  assert.equal(sanitizeGpuAdapter('C:\\Program Files\\NVIDIA\\driver.dll'), null);
  assert.equal(sanitizeGpuAdapter('D:/Games/GTA6/shader.bin'), null);
  assert.equal(sanitizeGpuAdapter('\\\\fileserver\\share\\gpu.dll'), null);
  assert.equal(sanitizeGpuAdapter('/home/alice/secret/gpu.bin'), null);
  assert.equal(sanitizeGpuAdapter('/Users/alice/Library/Application Support/gpu'), null);
  assert.equal(sanitizeGpuAdapter('/etc/passwd'), null);
  assert.equal(sanitizeGpuAdapter('../../etc/passwd'), null);
  assert.equal(sanitizeGpuAdapter('file:///C:/Users/alice/nvidia'), null);
  assert.equal(sanitizeGpuAdapter('https://user:pass@example.test/gpu'), null);
  assert.equal(sanitizeGpuAdapter(new Uint8Array([1, 2, 3])), null);
  assert.equal(sanitizeGpuAdapter(new ArrayBuffer(16)), null);
  assert.equal(sanitizeGpuAdapter(['nvidia', 'turing']), null);
  assert.equal(sanitizeGpuAdapter(12345), null);
  assert.equal(sanitizeGpuAdapter(null), null);
  assert.equal(sanitizeGpuAdapter(undefined), null);
  assert.equal(sanitizeGpuAdapter('   ///   '), null);
  assert.equal(
    sanitizeGpuAdapter({
      get vendor() {
        throw new Error('hostile getter');
      },
    }),
    null
  );

  // Pure formatDurationMs checks
  assert.equal(formatDurationMs(0), '0.0 ms');
  assert.equal(formatDurationMs(1420.56), '1420.6 ms');
  assert.equal(formatDurationMs('350.25'), '350.3 ms');
  assert.equal(formatDurationMs(null), '—');
  assert.equal(formatDurationMs(undefined), '—');
  assert.equal(formatDurationMs(-5), '—');
  assert.equal(formatDurationMs(Number.NaN), '—');
  assert.equal(formatDurationMs(Infinity), '—');
  assert.equal(formatDurationMs(true), '—');
  assert.equal(formatDurationMs('C:\\Users\\alice\\time.txt'), '—');

  // Default panel state, getSnapshot(), DOM, and export
  const { controller } = createPanel();
  const state = controller.getState();
  const snapshot = controller.getSnapshot();
  const root = state.root;

  assert.deepEqual(snapshot, state);
  assert.equal(state.gpuAdapter, null);
  assert.equal(state.firstVisibleFrameMs, null);
  assert.equal(state.blackLoadingMs, null);
  assert.equal(state.blackPresentsTotal, 0);
  assert.equal(state.reflectedStages, 0);
  assert.equal(state.inlineSamplerStages, 0);
  assert.equal(state.resourceHeapBindings, 0);
  assert.deepEqual(state.visibilityStats, {
    firstVisibleFrameMs: null,
    blackLoadingMs: null,
    blackPresentsTotal: 0,
  });
  assert.deepEqual(state.reflectionStats, {
    reflectedStages: 0,
    inlineSamplerStages: 0,
    resourceHeapBindings: 0,
  });

  const section = root.querySelector('.pg6-gpu-panel__adapter-reflection');
  assert.equal(section.getAttribute('role'), 'group');
  assert.equal(section.getAttribute('aria-label'), 'Adapter, Visibility & Shader Reflection');
  assert.equal(root.querySelector('.pg6-gpu-panel__gpu-adapter').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__first-visible-frame-ms').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__black-loading-ms').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__black-presents-total').textContent, '0');
  assert.equal(root.querySelector('.pg6-gpu-panel__reflected-stages').textContent, '0');
  assert.equal(root.querySelector('.pg6-gpu-panel__inline-sampler-stages').textContent, '0');
  assert.equal(root.querySelector('.pg6-gpu-panel__resource-heap-bindings').textContent, '0');

  const exported = JSON.parse(controller.exportMissingShadersJson());
  assert.equal(exported.gpu.gpuAdapter, null);
  assert.equal(exported.gpu.firstVisibleFrameMs, null);
  assert.equal(exported.gpu.blackLoadingMs, null);
  assert.equal(exported.gpu.blackPresentsTotal, 0);
  assert.deepEqual(exported.reflection, {
    reflectedStages: 0,
    inlineSamplerStages: 0,
    resourceHeapBindings: 0,
  });

  controller.destroy();
});

test('W240 update() ingests gpuAdapter, visibility timings, and S44 shader reflection breakdown from top-level and nested sources', () => {
  const { controller } = createPanel();
  const root = controller.getState().root;

  // 1. Top-level fields
  controller.update({
    gpuAdapter: 'NVIDIA / Turing',
    firstVisibleFrameMs: 1842.4,
    blackLoadingMs: 412.75,
    blackPresentsTotal: 19,
    reflectedStages: 248,
    inlineSamplerStages: 64,
    resourceHeapBindings: 512,
  });

  const s1 = controller.getSnapshot();
  assert.equal(s1.gpuAdapter, 'nvidia / turing');
  assert.equal(s1.firstVisibleFrameMs, 1842.4);
  assert.equal(s1.blackLoadingMs, 412.75);
  assert.equal(s1.blackPresentsTotal, 19);
  assert.equal(s1.reflectedStages, 248);
  assert.equal(s1.inlineSamplerStages, 64);
  assert.equal(s1.resourceHeapBindings, 512);

  assert.equal(root.querySelector('.pg6-gpu-panel__gpu-adapter').textContent, 'nvidia / turing');
  assert.equal(
    root.querySelector('.pg6-gpu-panel__first-visible-frame-ms').textContent,
    '1842.4 ms'
  );
  assert.equal(root.querySelector('.pg6-gpu-panel__black-loading-ms').textContent, '412.8 ms');
  assert.equal(root.querySelector('.pg6-gpu-panel__black-presents-total').textContent, '19');
  assert.equal(root.querySelector('.pg6-gpu-panel__reflected-stages').textContent, '248');
  assert.equal(root.querySelector('.pg6-gpu-panel__inline-sampler-stages').textContent, '64');
  assert.equal(root.querySelector('.pg6-gpu-panel__resource-heap-bindings').textContent, '512');

  // 2. Partial BootTracer summary with null optional fields must NOT clobber existing values.
  controller.update({
    bootSummary: {
      stage: 'D3D12_CREATE_DEVICE',
      gpuAdapter: null,
      firstVisibleFrameMs: null,
      blackLoadingMs: null,
    },
  });
  assert.equal(controller.getState().gpuAdapter, 'nvidia / turing');
  assert.equal(controller.getState().firstVisibleFrameMs, 1842.4);
  assert.equal(controller.getState().blackLoadingMs, 412.75);

  // 3. Nested W236 bootSummary + W231 swapchainStats + S44 reflectionStats (stageCount alias)
  controller.update({
    bootSummary: {
      stage: 'FIRST_VISIBLE_FRAME',
      gpuAdapter: { vendor: 'Intel', architecture: 'Gen-12LP' },
      firstVisibleFrameMs: 2150,
      blackLoadingMs: 605.5,
    },
    swapchainStats: {
      blackPresentsTotal: 27,
    },
    reflectionStats: {
      stageCount: 310,
      inlineSamplerStages: 92,
      resourceHeapBindings: 1024,
    },
  });

  const s2 = controller.getState();
  assert.equal(s2.boot.stage, 'FIRST_VISIBLE_FRAME');
  assert.equal(s2.boot.readyForSplash, true);
  assert.equal(s2.gpuAdapter, 'intel / gen-12lp');
  assert.equal(s2.firstVisibleFrameMs, 2150);
  assert.equal(s2.blackLoadingMs, 605.5);
  assert.equal(s2.blackPresentsTotal, 27);
  assert.equal(s2.reflectedStages, 310);
  assert.equal(s2.inlineSamplerStages, 92);
  assert.equal(s2.resourceHeapBindings, 1024);

  assert.equal(root.querySelector('.pg6-gpu-panel__gpu-adapter').textContent, 'intel / gen-12lp');
  assert.equal(
    root.querySelector('.pg6-gpu-panel__first-visible-frame-ms').textContent,
    '2150.0 ms'
  );
  assert.equal(root.querySelector('.pg6-gpu-panel__black-loading-ms').textContent, '605.5 ms');
  assert.equal(root.querySelector('.pg6-gpu-panel__black-presents-total').textContent, '27');
  assert.equal(root.querySelector('.pg6-gpu-panel__reflected-stages').textContent, '310');
  assert.equal(root.querySelector('.pg6-gpu-panel__inline-sampler-stages').textContent, '92');
  assert.equal(root.querySelector('.pg6-gpu-panel__resource-heap-bindings').textContent, '1024');

  // Verify JSON export contains all W240 sanitized fields
  const exported = JSON.parse(controller.exportMissingShadersJson());
  assert.equal(exported.gpu.gpuAdapter, 'intel / gen-12lp');
  assert.equal(exported.gpu.firstVisibleFrameMs, 2150);
  assert.equal(exported.gpu.blackLoadingMs, 605.5);
  assert.equal(exported.gpu.blackPresentsTotal, 27);
  assert.deepEqual(exported.reflection, {
    reflectedStages: 310,
    inlineSamplerStages: 92,
    resourceHeapBindings: 1024,
  });

  // Explicit top-level / sub-object null clears individual sections cleanly
  controller.update({ gpuAdapter: null, visibilityStats: null, reflectionStats: null });
  const s3 = controller.getSnapshot();
  assert.equal(s3.gpuAdapter, null);
  assert.equal(s3.firstVisibleFrameMs, null);
  assert.equal(s3.blackLoadingMs, null);
  assert.equal(s3.blackPresentsTotal, 0);
  assert.equal(s3.reflectedStages, 0);
  assert.equal(s3.inlineSamplerStages, 0);
  assert.equal(s3.resourceHeapBindings, 0);

  controller.destroy();
});

test('W240 sanitizes hostile adapter paths, invalid timings/counts, and strips leaks from DOM and export', () => {
  const { controller } = createPanel();
  const root = controller.getState().root;

  controller.update({
    gpuAdapter: 'C:\\Users\\alice\\AppData\\Local\\GTA6\\secret_gpu.dll',
    firstVisibleFrameMs: '/home/alice/secret/frame.bin',
    blackLoadingMs: -42,
    blackPresentsTotal: Number.NaN,
    reflectedStages: 'file:///C:/Users/alice/stages.json',
    inlineSamplerStages: -7,
    resourceHeapBindings: Infinity,
    reflectionStats: {
      hostPath: 'D:\\Games\\Secret\\manifest.json',
      rawBytes: new Uint8Array([0xde, 0xad, 0xbe, 0xef]),
    },
  });

  const state = controller.getSnapshot();
  assert.equal(state.gpuAdapter, null);
  assert.equal(state.firstVisibleFrameMs, null);
  assert.equal(state.blackLoadingMs, null);
  assert.equal(state.blackPresentsTotal, 0);
  assert.equal(state.reflectedStages, 0);
  assert.equal(state.inlineSamplerStages, 0);
  assert.equal(state.resourceHeapBindings, 0);

  assert.equal(root.querySelector('.pg6-gpu-panel__gpu-adapter').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__first-visible-frame-ms').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__black-loading-ms').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__black-presents-total').textContent, '0');
  assert.equal(root.querySelector('.pg6-gpu-panel__reflected-stages').textContent, '0');
  assert.equal(root.querySelector('.pg6-gpu-panel__inline-sampler-stages').textContent, '0');
  assert.equal(root.querySelector('.pg6-gpu-panel__resource-heap-bindings').textContent, '0');

  const exportedJson = controller.exportMissingShadersJson();
  assert.doesNotMatch(
    exportedJson,
    /Users|alice|AppData|secret_gpu|home|file:\/\/|Games|Secret|hostPath|rawBytes/i
  );

  // Direct buildMissingShaderExport also sanitizes hostile W240 fields
  const direct = buildMissingShaderExport({
    gpuAdapter: '/Users/bob/Library/Application Support/GTA6/adapter',
    firstVisibleFrameMs: 'not-a-number',
    blackLoadingMs: -1,
    blackPresentsTotal: 'C:\\secret',
    reflectedStages: 14.9,
    inlineSamplerStages: 3.2,
    resourceHeapBindings: 88.7,
    exportedAt: Date.UTC(2026, 7, 1, 0, 0, 0),
  });
  assert.equal(direct.gpu.gpuAdapter, null);
  assert.equal(direct.gpu.firstVisibleFrameMs, null);
  assert.equal(direct.gpu.blackLoadingMs, null);
  assert.equal(direct.gpu.blackPresentsTotal, 0);
  assert.deepEqual(direct.reflection, {
    reflectedStages: 14,
    inlineSamplerStages: 3,
    resourceHeapBindings: 88,
  });
  assert.doesNotMatch(JSON.stringify(direct), /Users|bob|Library|secret/i);

  controller.destroy();
});

test('W240 reset() cleanly resets all W240 and existing telemetry state and DOM while keeping panel mounted', () => {
  const { controller, container } = createPanel();
  const root = controller.getState().root;

  controller.update({
    bootSummary: {
      stage: 'FIRST_VISIBLE_FRAME',
      milestonesReached: 9,
      progressPct: 100,
      readyForSplash: true,
    },
    frameStats: { framesPresented: 50, droppedFrames: 2 },
    fallbackStats: { packHits: 40, fallbackHits: 5, builtinHits: 5 },
    compatStats: {
      bindGroupKindMismatches: 2,
      swapchainFormat: 'bgra8unorm',
      usesBlitFallback: true,
      blitCount: 5,
    },
    ioStats: { pgimFullFetches: 1, slabCacheHits: 10, slabCacheMisses: 2, syncFastPathHits: 8 },
    perfStats: {
      onePercentLowFps: 59.0,
      p99FrameMs: 16.8,
      zeroCopyUploads: 12,
      totalElidedStateCalls: 300,
      bindGroupCacheHits: 120,
      topMissingShaders: [{ hash: '0x1234', stage: 'fragment', hits: 4 }],
    },
    missingShaders: [{ hash: '0x1234', stage: 'fragment', hits: 4 }],
    gpuAdapter: 'AMD / RDNA2',
    firstVisibleFrameMs: 950.5,
    blackLoadingMs: 210.2,
    blackPresentsTotal: 8,
    reflectedStages: 180,
    inlineSamplerStages: 44,
    resourceHeapBindings: 612,
  });
  controller.exportMissingShadersJson();
  assert.equal(controller.getState().exportCount, 1);

  const afterReset = controller.reset();
  assert.equal(afterReset.destroyed, false);
  assert.equal(afterReset.visible, true);
  assert.equal(afterReset.root, root);
  assert.equal(root.parentNode, container);
  assert.equal(afterReset.exportCount, 0);
  assert.equal(afterReset.boot.stage, 'NOT_STARTED');
  assert.equal(afterReset.boot.milestonesReached, 0);
  assert.equal(afterReset.boot.progressPct, 0);
  assert.equal(afterReset.boot.readyForSplash, false);
  assert.equal(afterReset.frameStats.framesPresented, 0);
  assert.equal(afterReset.fallbackStats.packHits, 0);
  assert.equal(afterReset.compatStats.swapchainFormat, null);
  assert.equal(afterReset.ioStats.slabCacheHits, 0);
  assert.equal(afterReset.perfStats.onePercentLowFps, null);
  assert.equal(afterReset.totalMissingShaders, 0);

  // All W240 fields cleanly reset in getState(), getSnapshot(), and DOM
  assert.equal(afterReset.gpuAdapter, null);
  assert.equal(afterReset.firstVisibleFrameMs, null);
  assert.equal(afterReset.blackLoadingMs, null);
  assert.equal(afterReset.blackPresentsTotal, 0);
  assert.equal(afterReset.reflectedStages, 0);
  assert.equal(afterReset.inlineSamplerStages, 0);
  assert.equal(afterReset.resourceHeapBindings, 0);
  assert.deepEqual(controller.getSnapshot(), afterReset);

  assert.equal(root.querySelector('.pg6-gpu-panel__gpu-adapter').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__first-visible-frame-ms').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__black-loading-ms').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__black-presents-total').textContent, '0');
  assert.equal(root.querySelector('.pg6-gpu-panel__reflected-stages').textContent, '0');
  assert.equal(root.querySelector('.pg6-gpu-panel__inline-sampler-stages').textContent, '0');
  assert.equal(root.querySelector('.pg6-gpu-panel__resource-heap-bindings').textContent, '0');

  // Panel remains fully usable after reset()
  controller.update({ gpuAdapter: 'nvidia / ampere', reflectedStages: 12 });
  assert.equal(controller.getSnapshot().gpuAdapter, 'nvidia / ampere');
  assert.equal(controller.getSnapshot().reflectedStages, 12);
  assert.equal(root.querySelector('.pg6-gpu-panel__gpu-adapter').textContent, 'nvidia / ampere');
  assert.equal(root.querySelector('.pg6-gpu-panel__reflected-stages').textContent, '12');

  // Safe no-op after destroy()
  controller.destroy();
  assert.doesNotThrow(() => controller.reset());
  assert.equal(controller.getSnapshot().destroyed, true);
});

test('W240 dirty-checked DOM updates skip redundant writes for adapter, visibility, and reflection fields', () => {
  const { controller } = createPanel();
  const root = controller.getState().root;

  const snapshot = {
    gpuAdapter: 'nvidia / turing',
    firstVisibleFrameMs: 1200.5,
    blackLoadingMs: 300.2,
    blackPresentsTotal: 11,
    reflectedStages: 200,
    inlineSamplerStages: 50,
    resourceHeapBindings: 400,
  };
  controller.update(snapshot);

  const adapterEl = root.querySelector('.pg6-gpu-panel__gpu-adapter');
  const firstVisEl = root.querySelector('.pg6-gpu-panel__first-visible-frame-ms');
  const blackLoadEl = root.querySelector('.pg6-gpu-panel__black-loading-ms');
  const blackPresEl = root.querySelector('.pg6-gpu-panel__black-presents-total');
  const reflStagesEl = root.querySelector('.pg6-gpu-panel__reflected-stages');
  const inlineSampEl = root.querySelector('.pg6-gpu-panel__inline-sampler-stages');
  const heapBindEl = root.querySelector('.pg6-gpu-panel__resource-heap-bindings');

  const before = {
    adapter: adapterEl.textContentWrites,
    firstVis: firstVisEl.textContentWrites,
    blackLoad: blackLoadEl.textContentWrites,
    blackPres: blackPresEl.textContentWrites,
    reflStages: reflStagesEl.textContentWrites,
    inlineSamp: inlineSampEl.textContentWrites,
    heapBind: heapBindEl.textContentWrites,
  };

  // Identical repeated updates produce zero DOM writes
  controller.update(snapshot);
  controller.update({ ...snapshot, gpuAdapter: '  NVIDIA / TURING  ' });
  assert.equal(adapterEl.textContentWrites, before.adapter);
  assert.equal(firstVisEl.textContentWrites, before.firstVis);
  assert.equal(blackLoadEl.textContentWrites, before.blackLoad);
  assert.equal(blackPresEl.textContentWrites, before.blackPres);
  assert.equal(reflStagesEl.textContentWrites, before.reflStages);
  assert.equal(inlineSampEl.textContentWrites, before.inlineSamp);
  assert.equal(heapBindEl.textContentWrites, before.heapBind);

  // Updating a single field touches ONLY that DOM node
  controller.update({ resourceHeapBindings: 401 });
  assert.equal(heapBindEl.textContent, '401');
  assert.equal(heapBindEl.textContentWrites, before.heapBind + 1);
  assert.equal(adapterEl.textContentWrites, before.adapter);
  assert.equal(firstVisEl.textContentWrites, before.firstVis);
  assert.equal(blackLoadEl.textContentWrites, before.blackLoad);
  assert.equal(blackPresEl.textContentWrites, before.blackPres);
  assert.equal(reflStagesEl.textContentWrites, before.reflStages);
  assert.equal(inlineSampEl.textContentWrites, before.inlineSamp);

  controller.destroy();
});

test('W292 exports pacing/memory/mailbox formatters, formatSummary, and clean defaults when omitted', () => {
  assert.equal(typeof sanitizeUnitRatio, 'function');
  assert.equal(typeof sanitizePositiveInt, 'function');
  assert.equal(typeof formatRatioPct, 'function');
  assert.equal(typeof formatFrameJitterMs, 'function');
  assert.equal(typeof formatMemoryMb, 'function');
  assert.equal(typeof formatMailboxLanes, 'function');
  assert.equal(typeof formatSummary, 'function');
  assert.equal(globalThis.PlayGta6GpuBootPanel.sanitizeUnitRatio, sanitizeUnitRatio);
  assert.equal(globalThis.PlayGta6GpuBootPanel.sanitizePositiveInt, sanitizePositiveInt);
  assert.equal(globalThis.PlayGta6GpuBootPanel.formatRatioPct, formatRatioPct);
  assert.equal(globalThis.PlayGta6GpuBootPanel.formatFrameJitterMs, formatFrameJitterMs);
  assert.equal(globalThis.PlayGta6GpuBootPanel.formatMemoryMb, formatMemoryMb);
  assert.equal(globalThis.PlayGta6GpuBootPanel.formatMailboxLanes, formatMailboxLanes);
  assert.equal(globalThis.PlayGta6GpuBootPanel.formatSummary, formatSummary);

  // Unit ratio clamping [0, 1] and rejection of non-finite / hostile values
  assert.equal(sanitizeUnitRatio(0), 0);
  assert.equal(Object.is(sanitizeUnitRatio(-0), 0), true);
  assert.equal(sanitizeUnitRatio(0.945), 0.945);
  assert.equal(sanitizeUnitRatio(1), 1);
  assert.equal(sanitizeUnitRatio(-0.25), 0);
  assert.equal(sanitizeUnitRatio(1.75), 1);
  assert.equal(sanitizeUnitRatio('0.85'), 0.85);
  assert.equal(sanitizeUnitRatio('-0.5'), 0);
  assert.equal(sanitizeUnitRatio('2.5'), 1);
  assert.equal(sanitizeUnitRatio(null), null);
  assert.equal(sanitizeUnitRatio(undefined), null);
  assert.equal(sanitizeUnitRatio(Number.NaN), null);
  assert.equal(sanitizeUnitRatio(Infinity), null);
  assert.equal(sanitizeUnitRatio(-Infinity), null);
  assert.equal(sanitizeUnitRatio(true), null);
  assert.equal(sanitizeUnitRatio(false), null);
  assert.equal(sanitizeUnitRatio('C:\\Users\\alice\\rate'), null);
  assert.equal(sanitizeUnitRatio('<script>alert(1)</script>'), null);

  assert.equal(formatRatioPct(0.985), '98.5%');
  assert.equal(formatRatioPct(-0.4), '0.0%');
  assert.equal(formatRatioPct(1.4), '100.0%');
  assert.equal(formatRatioPct(null), '—');
  assert.equal(formatRatioPct(Number.NaN), '—');

  // Frame jitter (>= 0) & JS heap MB (>= 0) formatters
  assert.equal(formatFrameJitterMs(0), '0.00 ms');
  assert.equal(formatFrameJitterMs(1.456), '1.46 ms');
  assert.equal(formatFrameJitterMs('2.5'), '2.50 ms');
  assert.equal(formatFrameJitterMs(-0.1), '—');
  assert.equal(formatFrameJitterMs(Number.NaN), '—');
  assert.equal(formatFrameJitterMs(Infinity), '—');
  assert.equal(formatFrameJitterMs(null), '—');
  assert.equal(formatFrameJitterMs('<img src=x onerror=alert(1)>'), '—');

  assert.equal(formatMemoryMb(0), '0.0 MB');
  assert.equal(formatMemoryMb(256.46), '256.5 MB');
  assert.equal(formatMemoryMb('512'), '512.0 MB');
  assert.equal(formatMemoryMb(-1), '—');
  assert.equal(formatMemoryMb(Number.NaN), '—');
  assert.equal(formatMemoryMb(Infinity), '—');
  assert.equal(formatMemoryMb(null), '—');
  assert.equal(formatMemoryMb('/home/alice/heap.bin'), '—');

  // Positive integer (>= 1) mailboxLanes sanitizer & formatter
  assert.equal(sanitizePositiveInt(1), 1);
  assert.equal(sanitizePositiveInt(4), 4);
  assert.equal(sanitizePositiveInt(4.9), 4);
  assert.equal(sanitizePositiveInt('8'), 8);
  assert.equal(sanitizePositiveInt(0), null);
  assert.equal(sanitizePositiveInt(0.9), null);
  assert.equal(sanitizePositiveInt(-3), null);
  assert.equal(sanitizePositiveInt(Number.NaN), null);
  assert.equal(sanitizePositiveInt(Infinity), null);
  assert.equal(sanitizePositiveInt(true), null);
  assert.equal(sanitizePositiveInt(null), null);
  assert.equal(sanitizePositiveInt('<svg onload=alert(1)>'), null);
  assert.equal(formatMailboxLanes(4), '4');
  assert.equal(formatMailboxLanes(0), '—');
  assert.equal(formatMailboxLanes(null), '—');

  // Default panel state, snapshot, DOM, summary, and export
  const { controller } = createPanel();
  const state = controller.getState();
  const snapshot = controller.getSnapshot();
  const root = state.root;

  assert.deepEqual(snapshot, state);
  assert.equal(state.fps30ComplianceRate, null);
  assert.equal(state.frameJitterMs, null);
  assert.equal(state.jsHeapUsedMb, null);
  assert.equal(state.vramBudgetUtilization, null);
  assert.equal(state.bindGroupHitRate, null);
  assert.equal(state.mailboxLanes, null);
  assert.deepEqual(state.pacingStats, {
    fps30ComplianceRate: null,
    frameJitterMs: null,
  });
  assert.deepEqual(state.memoryStats, {
    jsHeapUsedMb: null,
    vramBudgetUtilization: null,
    bindGroupHitRate: null,
  });

  const section = root.querySelector('.pg6-gpu-panel__pacing-memory');
  assert.equal(section.getAttribute('role'), 'group');
  assert.equal(section.getAttribute('aria-label'), '30 FPS Pacing, Memory & I/O Telemetry');
  assert.equal(root.querySelector('.pg6-gpu-panel__fps30-compliance-rate').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__frame-jitter-ms').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__js-heap-used-mb').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__vram-budget-utilization').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__bind-group-hit-rate').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__mailbox-lanes').textContent, '—');

  // Missing/null optional W292 metrics are gracefully omitted from formatSummary()
  const summary = controller.formatSummary();
  assert.equal(summary, 'NOT_STARTED (0%) | frames=0 | packHit=0.0%');
  assert.equal(formatSummary(snapshot), summary);
  assert.doesNotMatch(summary, /30fps=|jitter=|jsHeap=|vram=|bgHit=|mailboxLanes=/);

  const exported = JSON.parse(controller.exportMissingShadersJson());
  assert.deepEqual(exported.pacing, {
    fps30ComplianceRate: null,
    frameJitterMs: null,
  });
  assert.deepEqual(exported.memory, {
    jsHeapUsedMb: null,
    vramBudgetUtilization: null,
    bindGroupHitRate: null,
  });
  assert.equal(exported.mailboxLanes, null);

  controller.destroy();
});

test('W292 update() normalizes, clamps, and renders 30 FPS pacing, memory/VRAM, bind-group hit-rate, and mailboxLanes across DOM, getSnapshot(), formatSummary(), and reset()', () => {
  const { controller } = createPanel();
  const root = controller.getState().root;

  // 1. Top-level fields with clamping and integer flooring
  controller.update({
    bootSummary: { stage: 'FIRST_VISIBLE_FRAME', progressPct: 100, readyForSplash: true },
    frameStats: { framesPresented: 120, droppedFrames: 1 },
    fallbackStats: { packHits: 95, fallbackHits: 5, builtinHits: 0 },
    fps30ComplianceRate: 1.35, // clamps to 1
    frameJitterMs: 1.426,
    jsHeapUsedMb: 312.74,
    vramBudgetUtilization: -0.2, // clamps to 0
    bindGroupHitRate: 0.9125,
    mailboxLanes: 4.8, // floors to 4
  });

  const s1 = controller.getSnapshot();
  assert.equal(s1.fps30ComplianceRate, 1);
  assert.equal(s1.frameJitterMs, 1.426);
  assert.equal(s1.jsHeapUsedMb, 312.74);
  assert.equal(s1.vramBudgetUtilization, 0);
  assert.equal(s1.bindGroupHitRate, 0.9125);
  assert.equal(s1.mailboxLanes, 4);

  assert.equal(root.querySelector('.pg6-gpu-panel__fps30-compliance-rate').textContent, '100.0%');
  assert.equal(root.querySelector('.pg6-gpu-panel__frame-jitter-ms').textContent, '1.43 ms');
  assert.equal(root.querySelector('.pg6-gpu-panel__js-heap-used-mb').textContent, '312.7 MB');
  assert.equal(root.querySelector('.pg6-gpu-panel__vram-budget-utilization').textContent, '0.0%');
  assert.equal(root.querySelector('.pg6-gpu-panel__bind-group-hit-rate').textContent, '91.3%');
  assert.equal(root.querySelector('.pg6-gpu-panel__mailbox-lanes').textContent, '4');

  const summary1 = controller.formatSummary();
  assert.equal(
    summary1,
    'FIRST_VISIBLE_FRAME (100%) | frames=120 | packHit=95.0% | 30fps=100.0% | jitter=1.43 ms | jsHeap=312.7 MB | vram=0.0% | bgHit=91.3% | mailboxLanes=4'
  );
  assert.equal(formatSummary(s1), summary1);

  // 2. Partial unrelated bootSummary with null optional fields must NOT clobber existing values
  controller.update({
    bootSummary: {
      stage: 'FIRST_VISIBLE_FRAME',
      fps30ComplianceRate: null,
      frameJitterMs: null,
      jsHeapUsedMb: null,
      vramBudgetUtilization: null,
      bindGroupHitRate: null,
      mailboxLanes: null,
    },
  });
  const sPreserved = controller.getSnapshot();
  assert.equal(sPreserved.fps30ComplianceRate, 1);
  assert.equal(sPreserved.frameJitterMs, 1.426);
  assert.equal(sPreserved.jsHeapUsedMb, 312.74);
  assert.equal(sPreserved.vramBudgetUtilization, 0);
  assert.equal(sPreserved.bindGroupHitRate, 0.9125);
  assert.equal(sPreserved.mailboxLanes, 4);

  // 3. Nested pacingStats, memoryStats, resourceTableStats, and ioStats.mailboxLanes
  controller.update({
    pacingStats: { fps30ComplianceRate: 0.967, frameJitterMs: 0.85 },
    memoryStats: { jsHeapUsedMb: 420.5, vramBudgetUtilization: 0.625 },
    resourceTableStats: { bindGroupHitRate: 0.982 },
    ioStats: { pgimFullFetches: 2, slabCacheHits: 50, slabCacheMisses: 1, syncFastPathHits: 49, mailboxLanes: 8 },
  });

  const s2 = controller.getSnapshot();
  assert.equal(s2.fps30ComplianceRate, 0.967);
  assert.equal(s2.frameJitterMs, 0.85);
  assert.equal(s2.jsHeapUsedMb, 420.5);
  assert.equal(s2.vramBudgetUtilization, 0.625);
  assert.equal(s2.bindGroupHitRate, 0.982);
  assert.equal(s2.mailboxLanes, 8);
  // Existing ioStats shape remains strictly backwards-compatible
  assert.deepEqual(s2.ioStats, {
    pgimFullFetches: 2,
    slabCacheHits: 50,
    slabCacheMisses: 1,
    syncFastPathHits: 49,
  });

  assert.equal(root.querySelector('.pg6-gpu-panel__fps30-compliance-rate').textContent, '96.7%');
  assert.equal(root.querySelector('.pg6-gpu-panel__frame-jitter-ms').textContent, '0.85 ms');
  assert.equal(root.querySelector('.pg6-gpu-panel__js-heap-used-mb').textContent, '420.5 MB');
  assert.equal(root.querySelector('.pg6-gpu-panel__vram-budget-utilization').textContent, '62.5%');
  assert.equal(root.querySelector('.pg6-gpu-panel__bind-group-hit-rate').textContent, '98.2%');
  assert.equal(root.querySelector('.pg6-gpu-panel__mailbox-lanes').textContent, '8');

  // Verify JSON export contains sanitized W292 sections
  const exported = JSON.parse(controller.exportMissingShadersJson());
  assert.deepEqual(exported.pacing, {
    fps30ComplianceRate: 0.967,
    frameJitterMs: 0.85,
  });
  assert.deepEqual(exported.memory, {
    jsHeapUsedMb: 420.5,
    vramBudgetUtilization: 0.625,
    bindGroupHitRate: 0.982,
  });
  assert.equal(exported.mailboxLanes, 8);

  // 4. Explicit null omission removes metrics cleanly from DOM and formatSummary()
  controller.update({
    fps30ComplianceRate: null,
    jsHeapUsedMb: null,
    mailboxLanes: null,
  });
  const s3 = controller.getSnapshot();
  assert.equal(s3.fps30ComplianceRate, null);
  assert.equal(s3.frameJitterMs, 0.85);
  assert.equal(s3.jsHeapUsedMb, null);
  assert.equal(s3.vramBudgetUtilization, 0.625);
  assert.equal(s3.bindGroupHitRate, 0.982);
  assert.equal(s3.mailboxLanes, null);
  assert.equal(root.querySelector('.pg6-gpu-panel__fps30-compliance-rate').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__js-heap-used-mb').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__mailbox-lanes').textContent, '—');

  const summary3 = controller.formatSummary();
  assert.doesNotMatch(summary3, /30fps=|jsHeap=|mailboxLanes=/);
  assert.match(summary3, /jitter=0\.85 ms/);
  assert.match(summary3, /vram=62\.5%/);
  assert.match(summary3, /bgHit=98\.2%/);

  // 5. reset() clears all W292 metrics back to defaults
  const afterReset = controller.reset();
  assert.equal(afterReset.fps30ComplianceRate, null);
  assert.equal(afterReset.frameJitterMs, null);
  assert.equal(afterReset.jsHeapUsedMb, null);
  assert.equal(afterReset.vramBudgetUtilization, null);
  assert.equal(afterReset.bindGroupHitRate, null);
  assert.equal(afterReset.mailboxLanes, null);
  assert.equal(root.querySelector('.pg6-gpu-panel__frame-jitter-ms').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__vram-budget-utilization').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__bind-group-hit-rate').textContent, '—');
  assert.equal(controller.formatSummary(), 'NOT_STARTED (0%) | frames=0 | packHit=0.0%');

  controller.destroy();
});

test('W292 rejects XSS payloads and host paths across DOM, getSnapshot(), formatSummary(), and export', () => {
  const { controller } = createPanel();
  const root = controller.getState().root;

  controller.update({
    bootSummary: { stage: '<script>alert(1)</script>' },
    gpuAdapter: '<img src=x onerror=alert(1)>',
    fps30ComplianceRate: '<script>alert("fps")</script>',
    frameJitterMs: 'C:\\Users\\alice\\GTA6\\jitter.bin',
    jsHeapUsedMb: '/home/alice/secret/heap.dump',
    vramBudgetUtilization: '"><svg onload=alert(1)>',
    bindGroupHitRate: 'file:///C:/Users/alice/bind_groups.bin',
    mailboxLanes: '0; ../../etc/passwd',
    pacingStats: {
      hostPath: 'D:\\Games\\Secret\\pacing.log',
      rawBytes: new Uint8Array([0xde, 0xad, 0xbe, 0xef]),
    },
    memoryStats: {
      localFile: '/Users/alice/Library/Application Support/vram.bin',
      payload: '<iframe src="javascript:alert(1)"></iframe>',
    },
  });

  const state = controller.getSnapshot();
  assert.equal(state.boot.stage, 'UNKNOWN');
  assert.equal(state.fps30ComplianceRate, null);
  assert.equal(state.frameJitterMs, null);
  assert.equal(state.jsHeapUsedMb, null);
  assert.equal(state.vramBudgetUtilization, null);
  assert.equal(state.bindGroupHitRate, null);
  assert.equal(state.mailboxLanes, null);

  assert.equal(root.querySelector('.pg6-gpu-panel__fps30-compliance-rate').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__frame-jitter-ms').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__js-heap-used-mb').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__vram-budget-utilization').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__bind-group-hit-rate').textContent, '—');
  assert.equal(root.querySelector('.pg6-gpu-panel__mailbox-lanes').textContent, '—');

  // Ensure zero unescaped HTML tags or host paths leak into formatSummary() or JSON export
  const summary = controller.formatSummary();
  assert.doesNotMatch(summary, /[<>]|Users|alice|home|secret|file:\/\/|passwd|Games|Library/i);
  assert.doesNotMatch(summary, /30fps=|jitter=|jsHeap=|vram=|bgHit=|mailboxLanes=/);

  const exportedJson = controller.exportMissingShadersJson();
  assert.doesNotMatch(
    exportedJson,
    /[<>]|Users|alice|home|secret|file:\/\/|passwd|Games|Library|hostPath|rawBytes|localFile|javascript:/i
  );

  // Also test zero/negative values for mailboxLanes & non-negative scalars
  controller.update({
    fps30ComplianceRate: Number.NaN,
    frameJitterMs: -5,
    jsHeapUsedMb: -128,
    vramBudgetUtilization: Infinity,
    bindGroupHitRate: -Infinity,
    mailboxLanes: 0,
  });
  const s2 = controller.getSnapshot();
  assert.equal(s2.fps30ComplianceRate, null);
  assert.equal(s2.frameJitterMs, null);
  assert.equal(s2.jsHeapUsedMb, null);
  assert.equal(s2.vramBudgetUtilization, null);
  assert.equal(s2.bindGroupHitRate, null);
  assert.equal(s2.mailboxLanes, null);

  controller.destroy();
});

test('W292 dirty-checked DOM updates skip redundant writes for pacing, memory/VRAM, and mailboxLanes', () => {
  const { controller } = createPanel();
  const root = controller.getState().root;

  const snapshot = {
    fps30ComplianceRate: 1.0,
    frameJitterMs: 1.25,
    jsHeapUsedMb: 250.4,
    vramBudgetUtilization: 0.5,
    bindGroupHitRate: 0.875,
    mailboxLanes: 4,
  };
  controller.update(snapshot);

  const complianceEl = root.querySelector('.pg6-gpu-panel__fps30-compliance-rate');
  const jitterEl = root.querySelector('.pg6-gpu-panel__frame-jitter-ms');
  const heapEl = root.querySelector('.pg6-gpu-panel__js-heap-used-mb');
  const vramEl = root.querySelector('.pg6-gpu-panel__vram-budget-utilization');
  const bgHitEl = root.querySelector('.pg6-gpu-panel__bind-group-hit-rate');
  const lanesEl = root.querySelector('.pg6-gpu-panel__mailbox-lanes');

  const before = {
    compliance: complianceEl.textContentWrites,
    jitter: jitterEl.textContentWrites,
    heap: heapEl.textContentWrites,
    vram: vramEl.textContentWrites,
    bgHit: bgHitEl.textContentWrites,
    lanes: lanesEl.textContentWrites,
  };

  // Identical and clamp/floor-equivalent updates must produce zero DOM writes
  controller.update(snapshot);
  controller.update({
    ...snapshot,
    fps30ComplianceRate: 1.5, // still clamps to 1 -> '100.0%'
    mailboxLanes: 4.9, // still floors to 4 -> '4'
  });
  assert.equal(complianceEl.textContentWrites, before.compliance);
  assert.equal(jitterEl.textContentWrites, before.jitter);
  assert.equal(heapEl.textContentWrites, before.heap);
  assert.equal(vramEl.textContentWrites, before.vram);
  assert.equal(bgHitEl.textContentWrites, before.bgHit);
  assert.equal(lanesEl.textContentWrites, before.lanes);

  // Updating a single metric touches ONLY that DOM node
  controller.update({ mailboxLanes: 8 });
  assert.equal(lanesEl.textContent, '8');
  assert.equal(lanesEl.textContentWrites, before.lanes + 1);
  assert.equal(complianceEl.textContentWrites, before.compliance);
  assert.equal(jitterEl.textContentWrites, before.jitter);
  assert.equal(heapEl.textContentWrites, before.heap);
  assert.equal(vramEl.textContentWrites, before.vram);
  assert.equal(bgHitEl.textContentWrites, before.bgHit);

  controller.destroy();
});
