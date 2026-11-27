import test from 'node:test';
import assert from 'node:assert/strict';
import {
  HudController,
  normalizeSwapchainColorMode,
  sanitizeForJson,
  safeJsonStringify,
} from '../web/ui/hud_controller.js';
import { CrashReporter } from '../web/telemetry/crash_report.js';

function createDisplayEl(initialText = '') {
  const attrs = new Map();
  const children = [];
  const el = {
    textContent: initialText,
    title: '',
    hidden: false,
    dataset: {},
    style: { display: '' },
    children,
    setAttribute(k, v) {
      attrs.set(k, String(v));
    },
    getAttribute(k) {
      return attrs.has(k) ? attrs.get(k) : null;
    },
    appendChild(child) {
      children.push(child);
      child.parentNode = el;
      return child;
    },
    removeChild(child) {
      const idx = children.indexOf(child);
      if (idx >= 0) children.splice(idx, 1);
      return child;
    },
  };
  return el;
}

function createMockHud() {
  const statusCalls = [];
  const logCalls = [];
  const visibilityCalls = [];
  return {
    statusCalls,
    logCalls,
    visibilityCalls,
    setStatus(text, level) {
      statusCalls.push({ text, level });
    },
    appendLog(text, level) {
      logCalls.push({ text, level });
    },
    updateMetrics() {},
    setVisible(v) {
      visibilityCalls.push(Boolean(v));
    },
  };
}

test('HudController ingests #1077 GPU memory, Wasm linear memory, shader streaming, swapchainColorMode & frameJitterMs', () => {
  assert.equal(normalizeSwapchainColorMode('hdr-tonemap-aces'), 'hdr-tonemap-aces');
  assert.equal(
    normalizeSwapchainColorMode(null, { toneMapActive: true, toneMapMode: 'aces' }),
    'hdr-tonemap-aces'
  );
  assert.equal(
    normalizeSwapchainColorMode(null, { swapchainFormat: 'bgra8unorm-srgb' }),
    'sdr-srgb'
  );
  assert.equal(
    normalizeSwapchainColorMode('C:\\Users\\secret\\bad.exe', { applySrgbEncode: true }),
    'srgb-encode'
  );

  const gpuMemoryEl = createDisplayEl();
  const wasmMemoryEl = createDisplayEl();
  const shaderStreamingEl = createDisplayEl();
  const swapchainColorModeEl = createDisplayEl();

  const controller = new HudController({
    displayElements: {
      gpuMemoryEl,
      wasmMemoryEl,
      shaderStreamingEl,
      swapchainColorModeEl,
    },
  });

  controller.updateTelemetry({
    gpuMemory: {
      liveBufferBytes: 64 * 1024 * 1024,
      liveTextureBytes: 192 * 1024 * 1024,
      peakBufferBytes: 96 * 1024 * 1024,
      peakTextureBytes: 256 * 1024 * 1024,
      stagingPoolBytes: 16 * 1024 * 1024,
    },
    wasmLinearMemoryBytes: 512 * 1024 * 1024,
    shaderPackStats: {
      rangedShaderReady: 128,
      rangedShaderFetches: 42,
      rangedShaderErrors: 0,
      asyncShaderPendingCount: 3,
      rangedPackActive: true,
    },
    swapchainStats: {
      swapchainFormat: 'rgba16float',
      toneMapMode: 'aces',
      isHdrStream: true,
      toneMapActive: true,
      applySrgbEncode: true,
    },
    frameBudget: {
      fps30ComplianceRate: 0.98,
      dominantBottleneck: 'within_budget',
      frameJitterMs: 1.25,
    },
  });

  const snap = controller.getTelemetry();
  assert.equal(snap.memory.liveBufferBytes, 64 * 1024 * 1024);
  assert.equal(snap.memory.liveTextureBytes, 192 * 1024 * 1024);
  assert.equal(snap.memory.peakBufferBytes, 96 * 1024 * 1024);
  assert.equal(snap.memory.peakTextureBytes, 256 * 1024 * 1024);
  assert.equal(snap.memory.totalLiveGpuBytes, 256 * 1024 * 1024);
  assert.equal(snap.memory.wasmLinearMemoryBytes, 512 * 1024 * 1024);
  assert.equal(snap.gpu.rangedShaderReady, 128);
  assert.equal(snap.gpu.rangedShaderFetches, 42);
  assert.equal(snap.gpu.asyncShaderPendingCount, 3);
  assert.equal(snap.gpu.swapchainColorMode, 'hdr-tonemap-aces');
  assert.equal(snap.frameBudget.frameJitterMs, 1.25);

  assert.match(gpuMemoryEl.textContent, /256\.0 MiB/);
  assert.match(gpuMemoryEl.textContent, /buf 64\.0 MiB/);
  assert.match(gpuMemoryEl.textContent, /tex 192\.0 MiB/);
  assert.match(gpuMemoryEl.textContent, /peak 352\.0 MiB/);
  assert.equal(wasmMemoryEl.textContent, '512.0 MiB');
  assert.equal(shaderStreamingEl.textContent, '128 ready · 42 fetches (3 pending)');
  assert.equal(swapchainColorModeEl.textContent, 'hdr-tonemap-aces');

  controller.destroy();
});

test('HUD status bar never sticks red after clearValidationErrors, setValidationErrors([]), resetTelemetry, or restartSession', async () => {
  const hud = createMockHud();
  const validationCountEl = createDisplayEl('0');
  const validationLatestEl = createDisplayEl('None');
  const validationBannerEl = createDisplayEl('');

  // Simulate app_bootstrap holding stale pre-restart validationErrorCount = 2
  let appPresentCount = 10;
  let appValErrors = 2;
  const app = {
    getBootTelemetry() {
      return {
        milestone: 'FIRST_VISIBLE_FRAME',
        presentCount: appPresentCount,
        gpu: {
          presentCount: appPresentCount,
          validationErrorCount: appValErrors,
        },
      };
    },
  };
  const sessionController = {
    async restart() {
      return { ok: true };
    },
    getState() {
      return 'running_wasm';
    },
    getMode() {
      return 'wasm';
    },
  };

  const controller = new HudController({
    hud,
    app,
    sessionController,
    bootMode: 'real-wasm64',
    displayElements: {
      validationCountEl,
      validationLatestEl,
      validationBannerEl,
    },
  });

  // 1) Counter-only error must NOT show "None" inside validationLatestEl
  controller.updateTelemetry({ gpu: { validationErrorCount: 2 } });
  assert.equal(validationCountEl.dataset.state, 'err');
  assert.equal(validationBannerEl.hidden, false);
  assert.notEqual(validationLatestEl.textContent, 'None');
  assert.match(validationLatestEl.textContent, /2 WebGPU validation errors/);

  // Explicit 0 clears counter-only error back to clean green/ok
  controller.updateTelemetry({ gpu: { validationErrorCount: 0 } });
  assert.equal(validationCountEl.dataset.state, 'ok');
  assert.equal(validationBannerEl.hidden, true);
  assert.equal(validationLatestEl.textContent, 'None');

  // 2) Recorded error -> clearValidationErrors() recovers status dot to green ('ok')
  controller.recordValidationError({ type: 'validation', message: 'Transient bind mismatch' });
  assert.equal(validationCountEl.dataset.state, 'err');
  assert.equal(controller.getBootStatus().statusLevel, 'error');

  hud.statusCalls.length = 0;
  controller.clearValidationErrors();
  assert.equal(validationCountEl.dataset.state, 'ok');
  assert.equal(validationBannerEl.hidden, true);
  assert.equal(controller.getBootStatus().statusLevel, 'ok');
  assert.equal(hud.statusCalls.at(-1)?.level, 'ok');

  // 3) setValidationErrors([], { totalCount: 0 }) syncs _gpuTelemetry.validationErrorCount
  controller.recordValidationError({ type: 'validation', message: 'Another transient error' });
  controller.setValidationErrors([], { totalCount: 0 });
  assert.equal(controller.getTelemetry().gpu.validationErrorCount, 0);
  assert.equal(validationCountEl.dataset.state, 'ok');

  // 4) resetTelemetry() clears validation errors too (no split-brain red banner)
  controller.recordValidationError({ type: 'validation', message: 'Before reset' });
  controller.resetTelemetry();
  assert.equal(controller.getValidationSummary().count, 0);
  assert.equal(validationCountEl.dataset.state, 'ok');
  assert.equal(validationBannerEl.hidden, true);

  // 5) restartSession() clears validation errors AND baselines stale app.getBootTelemetry() count
  controller.recordValidationError({ type: 'validation', message: 'Pre-restart error' });
  assert.equal(validationCountEl.dataset.state, 'err');
  const res = await controller.restartSession();
  assert.equal(res.ok, true);
  assert.equal(controller.getValidationSummary().count, 0);
  assert.equal(validationCountEl.dataset.state, 'ok');
  assert.equal(validationBannerEl.hidden, true);

  // New frame after restart with stale appValErrors=2 must NOT re-latch red!
  appPresentCount = 11;
  controller._syncFromAppSources();
  assert.equal(controller.getValidationSummary().count, 0);
  assert.equal(validationCountEl.dataset.state, 'ok');

  // Only a genuinely NEW error (appValErrors=3) increments post-restart count to 1
  appPresentCount = 12;
  appValErrors = 3;
  controller._syncFromAppSources();
  assert.equal(controller.getValidationSummary().count, 1);

  controller.destroy();
});

test('Export Report includes memory_stats, swapchainColorMode, shaderStreaming, deduplicated GPU errors, and survives BigInt/circular values', async () => {
  const downloads = [];
  const crashReporter = new CrashReporter({ appVersion: '1.0.0-test' });
  const controller = new HudController({
    crashReporter,
    downloadBlob(filename, data, mimeType) {
      downloads.push({ filename, data, mimeType });
    },
    storageManager: {
      async exportUserdataSnapshot() {
        const circular = { supported: true, fileCount: 1, totalBytes: 64n, files: [] };
        circular.self = circular;
        return circular;
      },
    },
  });

  controller.updateTelemetry({
    memory: {
      jsHeapUsedMb: 210.5,
      jsHeapTotalMb: 512,
      liveBufferBytes: 32 * 1024 * 1024,
      liveTextureBytes: 96 * 1024 * 1024,
      peakBufferBytes: 48 * 1024 * 1024,
      peakTextureBytes: 128 * 1024 * 1024,
      wasmLinearMemoryBytes: 256 * 1024 * 1024,
    },
    shaderPackStats: {
      rangedShaderReady: 55,
      rangedShaderFetches: 14,
      rangedPackActive: true,
    },
    swapchainStats: {
      swapchainFormat: 'bgra8unorm-srgb',
      applySrgbEncode: true,
    },
  });

  // Duplicate GPU validation errors should be cleanly deduplicated in export
  controller.recordValidationError({ type: 'validation', message: 'Bind group slot 2 unbound', timestampMs: 100 });
  controller.recordValidationError({ type: 'validation', message: 'Bind group slot 2 unbound', timestampMs: 250 });
  controller.recordValidationError({ type: 'out-of-memory', message: 'Buffer alloc failed', timestampMs: 300 });

  // Caller extra with BigInt and circular reference must NEVER crash exportCrashReport
  const callerExtra = { guestPtr: 0x100000004n };
  callerExtra.cycle = callerExtra;

  const res = await controller.exportCrashReport({ extra: callerExtra });
  assert.equal(res.ok, true);
  const parsed = JSON.parse(res.json);

  assert.equal(parsed.extra.memory_stats.liveBufferBytes, 32 * 1024 * 1024);
  assert.equal(parsed.extra.memory_stats.liveTextureBytes, 96 * 1024 * 1024);
  assert.equal(parsed.extra.memory_stats.peakBufferBytes, 48 * 1024 * 1024);
  assert.equal(parsed.extra.memory_stats.peakTextureBytes, 128 * 1024 * 1024);
  assert.equal(parsed.extra.memory_stats.wasmLinearMemoryBytes, 256 * 1024 * 1024);
  assert.equal(parsed.extra.swapchainColorMode, 'srgb-encode');
  assert.equal(parsed.extra.shaderStreaming.rangedShaderReady, 55);
  assert.equal(parsed.extra.shaderStreaming.rangedShaderFetches, 14);
  assert.equal(parsed.extra.gpuValidationErrors.totalCount, 3);
  assert.equal(parsed.extra.gpuValidationErrors.uniqueCount, 2);
  assert.equal(parsed.extra.gpuValidationErrors.deduplicated[0].occurrences, 2);
  assert.equal(parsed.extra.guestPtr, String(0x100000004n));
  assert.equal(parsed.extra.cycle, '[Circular]');

  // exportSaves also survives BigInt + circular objects without throwing
  const saveRes = await controller.exportSaves();
  assert.equal(saveRes.ok, true);
  const parsedSave = JSON.parse(saveRes.json);
  assert.equal(parsedSave.totalBytes, '64');
  assert.equal(parsedSave.self, '[Circular]');

  // Direct helpers sanity
  const cyclic = { a: 42n };
  cyclic.self = cyclic;
  assert.equal(sanitizeForJson(cyclic).a, '42');
  assert.equal(sanitizeForJson(cyclic).self, '[Circular]');
  assert.doesNotThrow(() => safeJsonStringify(cyclic));

  controller.destroy();
});

test('Hide/Show HUD UX blurs focused descendants, syncs gpuBootPanel, clears inline display:none, guards re-entrancy & supports custom key codes', () => {
  let blurred = 0;
  const focusedChild = {
    blur() {
      blurred += 1;
    },
  };
  const hudRootEl = createDisplayEl();
  hudRootEl.style.display = 'none';
  hudRootEl.ownerDocument = { activeElement: focusedChild };
  hudRootEl.contains = (el) => el === focusedChild;

  const panelCalls = [];
  const gpuBootPanel = {
    hide() {
      panelCalls.push('hide');
    },
    show() {
      panelCalls.push('show');
    },
  };

  let hudSetVisibleCalls = 0;
  const hud = {
    setVisible(v) {
      hudSetVisibleCalls += 1;
      // Re-entrant callback must not recurse infinitely
      controller.setVisible(v);
    },
  };

  const controller = new HudController({
    hud,
    gpuBootPanel,
    displayElements: { hudRootEl },
  });

  // Hide HUD: blurs focused descendant, syncs gpuBootPanel.hide(), guards re-entrancy
  assert.equal(controller.setVisible(false), false);
  assert.equal(blurred, 1);
  assert.deepEqual(panelCalls, ['hide']);
  assert.equal(hudSetVisibleCalls, 1);
  assert.equal(hudRootEl.hidden, true);
  assert.equal(hudRootEl.getAttribute('aria-hidden'), 'true');

  // Show HUD: clears stale inline display:none and syncs gpuBootPanel.show()
  hudRootEl.style.display = 'none';
  assert.equal(controller.setVisible(true), true);
  assert.equal(hudRootEl.style.display, '');
  assert.deepEqual(panelCalls, ['hide', 'show']);
  assert.equal(hudSetVisibleCalls, 2);

  // Custom toggle key matches event.code generically (e.g. F9)
  const keyListeners = new Set();
  const target = {
    addEventListener(type, fn) {
      if (type === 'keydown') keyListeners.add(fn);
    },
    removeEventListener(type, fn) {
      if (type === 'keydown') keyListeners.delete(fn);
    },
  };
  controller.bindToggleKey(target, 'F9');
  for (const fn of keyListeners) {
    fn({ key: 'Unidentified', code: 'F9', preventDefault() {} });
  }
  assert.equal(controller.isVisible(), false);

  controller.destroy();
});
