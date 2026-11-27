import test from 'node:test';
import assert from 'node:assert/strict';

import {
  DEFAULT_SHADER_PACK_URL,
  PlayGta6AppHandle,
  bootstrapPlayGta6App,
} from '../web/app_bootstrap.js';
import { SESSION_MODE } from '../web/runtime/session_controller.js';

function createClock(startMs = 10000) {
  let current = startMs;
  return {
    now: () => current,
    advance(deltaMs) {
      current += deltaMs;
      return current;
    },
  };
}

function createFakeEventTarget(extra = {}) {
  const listeners = new Map();
  return {
    ...extra,
    addEventListener(type, fn, opts) {
      if (!listeners.has(type)) {
        listeners.set(type, new Set());
      }
      listeners.get(type).add({ fn, once: Boolean(opts && opts.once) });
    },
    removeEventListener(type, fn) {
      const set = listeners.get(type);
      if (!set) return;
      for (const entry of set) {
        if (entry.fn === fn) {
          set.delete(entry);
        }
      }
    },
    dispatch(type, event = {}) {
      const set = listeners.get(type);
      if (!set) return;
      for (const entry of [...set]) {
        if (entry.once) {
          set.delete(entry);
        }
        entry.fn({ type, ...event });
      }
    },
    listenerCount(type) {
      if (type) {
        return listeners.get(type)?.size ?? 0;
      }
      let total = 0;
      for (const set of listeners.values()) {
        total += set.size;
      }
      return total;
    },
  };
}

function createFakeTimers() {
  let nextId = 1;
  const active = new Map();
  return {
    setInterval(fn, ms) {
      const id = nextId++;
      active.set(id, { fn, ms });
      return id;
    },
    clearInterval(id) {
      active.delete(id);
    },
    activeCount() {
      return active.size;
    },
  };
}

function createMockFetch(payload) {
  return async () => ({
    ok: true,
    status: 200,
    async json() {
      return typeof payload === 'function' ? payload() : payload;
    },
  });
}

function createTestHost(statusPayload, extraBootstrapOptions = {}) {
  const clock = createClock(30000);
  const timers = createFakeTimers();
  const broadcastInstances = [];

  class MockBroadcastChannel {
    constructor(name) {
      this.name = name;
      this.onmessage = null;
      this.closed = false;
      broadcastInstances.push(this);
    }
    close() {
      this.closed = true;
    }
  }

  const windowObj = createFakeEventTarget({
    crossOriginIsolated: true,
    innerWidth: 1280,
    innerHeight: 720,
    devicePixelRatio: 1,
    location: { origin: 'https://playgta6.local' },
    BroadcastChannel: MockBroadcastChannel,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
  });

  const documentObj = createFakeEventTarget({
    pointerLockElement: null,
    hidden: false,
    visibilityState: 'visible',
  });

  const navigatorObj = {
    gpu: {
      getPreferredCanvasFormat() {
        return 'bgra8unorm';
      },
      async requestAdapter() {
        return null;
      },
    },
    getGamepads() {
      return [];
    },
  };

  const canvas = createFakeEventTarget({
    width: 1280,
    height: 720,
    clientWidth: 1280,
    clientHeight: 720,
  });

  const mockWorker = createFakeEventTarget({
    postMessage() {},
    terminate() {},
  });

  const hudMetrics = [];
  const hudLogs = [];
  const hud = {
    metrics: hudMetrics,
    logs: hudLogs,
    setStatus() {},
    appendLog(message, level = 'info') {
      hudLogs.push({ message, level });
    },
    updateMetrics(metrics) {
      hudMetrics.push({ ...metrics });
    },
  };

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch(statusPayload),
    canvas,
    hud,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker() {
      return mockWorker;
    },
    ...extraBootstrapOptions,
  });

  return {
    app,
    windowObj,
    documentObj,
    canvas,
    mockWorker,
    broadcastInstances,
    hud,
    timers,
  };
}

test('W88 gpu-stats real-pack vs fallback draws, gpu-validation-error sanitization, and HUD forwarding', async () => {
  const gpuUpdates = [];
  const hudController = {
    updateGpuStats(stats) {
      gpuUpdates.push(stats);
    },
  };

  const { app, windowObj, mockWorker, broadcastInstances, hud } = createTestHost(
    {
      ok: true,
      coopCoep: true,
      hasWasmBinary: true,
      preferredShaderPackUrl: '/shaders/splash.wgsl.pack',
    },
    { hudController }
  );

  try {
    assert.ok(app instanceof PlayGta6AppHandle);
    const started = await app.start();
    assert.equal(started.ok, true);
    assert.equal(started.mode, SESSION_MODE.WASM);
    assert.equal(broadcastInstances.length, 1);

    // 1. Dispatch gpu-stats from worker with W83 real-pack vs fallback draw metrics.
    mockWorker.dispatch('message', {
      data: {
        type: 'gpu-stats',
        realPackDrawCount: 42,
        fallbackDrawCount: 3,
        presentCount: 7,
        unboundBindGroupCount: 2,
        skippedDrawCount: 1,
      },
    });

    // 2. Dispatch gpu-validation-error with host paths, stack trace, and code.
    mockWorker.dispatch('message', {
      data: {
        type: 'gpu-validation-error',
        errorType: 'validation',
        code: 'WGPU_BIND_GROUP_MISMATCH',
        message:
          'BindGroup mismatch at C:\\Users\\alice\\gta6\\shader.wgsl\n    at validateDraw (file:///C:/Users/alice/wgpu_worker.js:123:45)\n    at loop (/home/bob/src/worker.js:88:12)',
      },
    });

    // 3. Dispatch a second gpu-stats via BroadcastChannel('game-progress') with updated counts.
    broadcastInstances[0].onmessage({
      data: {
        type: 'stats',
        stats: {
          realPackDrawCount: 64,
          fallbackDrawCount: 5,
          presentCount: 12,
          unboundBindGroupCount: 4,
          skippedDrawCount: 2,
        },
      },
    });

    const snap = app.getBootTelemetry();
    assert.equal(Object.isFrozen(snap), true);
    assert.equal(Object.isFrozen(snap.gpu), true);
    assert.equal(Object.isFrozen(snap.audio), true);
    assert.equal(Object.isFrozen(snap.loader), true);

    assert.equal(snap.realPackDrawCount, 64);
    assert.equal(snap.fallbackDrawCount, 5);
    assert.equal(snap.presentCount, 12);
    assert.equal(snap.unboundBindGroupCount, 4);
    assert.equal(snap.skippedDrawCount, 2);
    assert.equal(snap.validationErrorCount, 1);
    assert.equal(snap.lastValidationErrorCode, 'WGPU_BIND_GROUP_MISMATCH');
    assert.ok(snap.lastValidationErrorMessage);
    assert.ok(snap.lastValidationErrorMessage.includes('BindGroup mismatch'));
    assert.ok(!snap.lastValidationErrorMessage.includes('alice'));
    assert.ok(!snap.lastValidationErrorMessage.includes('bob'));
    assert.ok(!snap.lastValidationErrorMessage.includes('validateDraw'));
    assert.ok(!snap.lastValidationErrorMessage.includes('wgpu_worker.js:123'));

    // Structured subsystem view mirrors top-level fields.
    assert.equal(snap.gpu.realPackDrawCount, 64);
    assert.equal(snap.gpu.fallbackDrawCount, 5);
    assert.equal(snap.gpu.presentCount, 12);
    assert.equal(snap.gpu.unboundBindGroupCount, 4);
    assert.equal(snap.gpu.skippedDrawCount, 2);
    assert.equal(snap.gpu.validationErrorCount, 1);
    assert.equal(snap.gpu.lastValidationErrorCode, 'WGPU_BIND_GROUP_MISMATCH');
    assert.equal(snap.gpu.lastValidationErrorMessage, snap.lastValidationErrorMessage);

    // HUD controller received both GPU stats updates.
    assert.equal(gpuUpdates.length, 2);
    assert.equal(gpuUpdates[0].realPackDrawCount, 42);
    assert.equal(gpuUpdates[0].fallbackDrawCount, 3);
    assert.equal(gpuUpdates[1].realPackDrawCount, 64);
    assert.equal(gpuUpdates[1].validationErrorCount, 1);
    assert.equal(gpuUpdates[1].lastValidationErrorCode, 'WGPU_BIND_GROUP_MISMATCH');

    // Primary hud.updateMetrics also received fallback metrics because hud had no updateGpuStats.
    assert.ok(
      hud.metrics.some((m) => m.realPackDrawCount === 64 && m.presentCount === 12),
      'hud.updateMetrics should receive GPU draw/present fallback metrics'
    );

    // Global diagnostics hooks expose identical sanitized snapshots.
    assert.deepEqual(windowObj.__playgta6BootState, snap);
    assert.equal(typeof globalThis.__playgta6Telemetry, 'function');
    assert.deepEqual(globalThis.__playgta6Telemetry(), snap);
    assert.deepEqual(globalThis.__playgta6Telemetry.getBootTelemetry(), snap);
  } finally {
    await app.destroy();
  }

  assert.equal(windowObj.__playgta6BootState, undefined);
  assert.equal(globalThis.__playgta6Telemetry, undefined);
});

test('W88 audio-stats outputState, framesDrained, underruns, and overruns tracking and HUD forwarding', async () => {
  const audioUpdates = [];
  const genericStats = [];
  const hudController = {
    setAudioStats(stats) {
      audioUpdates.push(stats);
    },
  };

  const { app, windowObj, mockWorker, broadcastInstances, hud } = createTestHost(
    {
      ok: true,
      coopCoep: true,
      hasWasmBinary: true,
    },
    { hudController }
  );

  // Secondary HUD controller with only generic updateStats().
  windowObj.playgta6HudController = {
    updateStats(stats) {
      genericStats.push(stats);
    },
  };

  try {
    await app.start();

    mockWorker.dispatch('message', {
      data: {
        type: 'audio-stats',
        outputState: 'running',
        framesDrained: 48000,
        underruns: 2,
        overruns: 1,
      },
    });

    let snap = app.getBootTelemetry();
    assert.equal(snap.outputState, 'running');
    assert.equal(snap.framesDrained, 48000);
    assert.equal(snap.underruns, 2);
    assert.equal(snap.overruns, 1);
    assert.deepEqual(snap.audio, {
      outputState: 'running',
      framesDrained: 48000,
      underruns: 2,
      overruns: 1,
    });

    // Nested stats + overflows alias via BroadcastChannel.
    broadcastInstances[0].onmessage({
      data: {
        type: 'audio-stats',
        stats: {
          state: 'suspended',
          totalPulledFrames: 96000,
          underrunCount: 5,
          overflows: 3,
        },
      },
    });

    snap = app.getBootTelemetry();
    assert.equal(snap.outputState, 'suspended');
    assert.equal(snap.framesDrained, 96000);
    assert.equal(snap.underruns, 5);
    assert.equal(snap.overruns, 3);

    assert.equal(audioUpdates.length, 2);
    assert.equal(audioUpdates[0].outputState, 'running');
    assert.equal(audioUpdates[0].framesDrained, 48000);
    assert.equal(audioUpdates[1].outputState, 'suspended');
    assert.equal(audioUpdates[1].overruns, 3);

    assert.equal(genericStats.length, 2);
    assert.equal(genericStats[1].type, 'audio-stats');
    assert.equal(genericStats[1].framesDrained, 96000);

    assert.ok(
      hud.metrics.some((m) => m.audioOutputState === 'suspended' && m.framesDrained === 96000),
      'hud.updateMetrics should receive audio stats fallback'
    );
  } finally {
    await app.destroy();
  }
});

test('W88 indirectDispatch shard, 52-thread pthread counters, and preferredShaderPackUrl surface in getBootTelemetry()', async () => {
  const loaderUpdates = [];
  const hudController = {
    updatePthreadStats(stats) {
      loaderUpdates.push(stats);
    },
  };

  const { app, windowObj, mockWorker, broadcastInstances, hud } = createTestHost(
    {
      ok: true,
      coopCoep: true,
      hasWasmBinary: true,
      sideModuleCount: 4,
      sideModuleUrls: [
        '/dist/guest_part_0.wasm',
        '/dist/guest_part_1.wasm',
        '/dist/guest_part_2.wasm',
        '/dist/guest_part_3.wasm',
      ],
      preferredShaderPackUrl: '/shaders/splash.wgsl.pack',
      shaderPackUrl: '/shaders/fallback.wgsl.pack',
    },
    { hudController }
  );

  try {
    await app.start();

    // Verify W79 preferredShaderPackUrl is already surfaced from /api/status.
    assert.equal(app.getBootTelemetry().preferredShaderPackUrl, '/shaders/splash.wgsl.pack');
    assert.equal(app.getBootTelemetry().indirectDispatch, 'unknown');

    // Progress message on BroadcastChannel carrying both label and W86 metrics.
    broadcastInstances[0].onmessage({
      data: {
        label: 'pthread pool active (52 threads)',
        metrics: {
          indirectDispatch: 'shard',
          pthreadSpawnedCount: 52,
          pthreadActiveCount: 52,
          pthreadPeakCount: 52,
          pthreadLoadQueueDepth: 6,
        },
      },
    });

    // Label still flowed through to HUD log (was not swallowed by telemetry ingestion).
    assert.ok(hud.logs.some((entry) => entry.message.includes('52 threads')));

    let snap = app.getBootTelemetry();
    assert.equal(snap.indirectDispatch, 'shard');
    assert.equal(snap.pthreadSpawnedCount, 52);
    assert.equal(snap.pthreadActiveCount, 52);
    assert.equal(snap.pthreadPeakCount, 52);
    assert.equal(snap.pthreadLoadQueueDepth, 6);
    assert.equal(snap.sideModuleCount, 4);
    assert.equal(snap.preferredShaderPackUrl, '/shaders/splash.wgsl.pack');
    assert.deepEqual(snap.loader, {
      indirectDispatch: 'shard',
      pthreadSpawnedCount: 52,
      pthreadActiveCount: 52,
      pthreadPeakCount: 52,
      pthreadLoadQueueDepth: 6,
      sideModuleCount: 4,
      preferredShaderPackUrl: '/shaders/splash.wgsl.pack',
    });

    // Dedicated pthread-stats update dropping active count while preserving peak count 52.
    mockWorker.dispatch('message', {
      data: {
        type: 'pthread-stats',
        pthreadSpawnedCount: 52,
        pthreadActiveCount: 48,
        pthreadLoadQueueDepth: 0,
      },
    });

    snap = app.getBootTelemetry();
    assert.equal(snap.pthreadSpawnedCount, 52);
    assert.equal(snap.pthreadActiveCount, 48);
    assert.equal(snap.pthreadPeakCount, 52);
    assert.equal(snap.pthreadLoadQueueDepth, 0);

    // Table dispatch mode alias is normalized cleanly.
    mockWorker.dispatch('message', {
      data: {
        type: 'shard-dispatch',
        dispatchMode: 'indirect-table',
      },
    });
    assert.equal(app.getBootTelemetry().indirectDispatch, 'table');

    assert.ok(loaderUpdates.length >= 2);
    assert.equal(loaderUpdates[0].indirectDispatch, 'shard');
    assert.equal(loaderUpdates[0].pthreadSpawnedCount, 52);
    assert.equal(loaderUpdates[0].preferredShaderPackUrl, '/shaders/splash.wgsl.pack');

    // window.__playgta6BootState and globalThis.__playgta6Telemetry reflect live state.
    assert.deepEqual(windowObj.__playgta6BootState, app.getBootTelemetry());
    assert.deepEqual(globalThis.__playgta6Telemetry(), app.getBootTelemetry());
  } finally {
    await app.destroy();
  }
});

test('W88 handles missing optional HUD/worker methods, throwing hooks, malformed/NaN payloads, and path redaction gracefully', async () => {
  const gpuWorker = createFakeEventTarget();
  const audioWorker = createFakeEventTarget();
  const ioWorker = createFakeEventTarget();

  // HUD with NO optional updater methods and a throwing secondary controller.
  const bareHud = {};
  const throwingHudController = {
    updateGpuStats() {
      throw new Error('gpu HUD crash');
    },
    updateAudioStats() {
      throw new Error('audio HUD crash');
    },
    updateLoaderStats() {
      throw new Error('loader HUD crash');
    },
  };

  const { app, mockWorker } = createTestHost(
    {
      ok: true,
      coopCoep: true,
      hasWasmBinary: true,
      // Malicious local host path in /api/status must be rejected by telemetry sanitization.
      preferredShaderPackUrl: 'C:\\Users\\alice\\Desktop\\secret.wgsl.pack',
    },
    {
      hud: bareHud,
      hudController: throwingHudController,
      gpuWorker,
      audioWorker,
      ioWorker,
    }
  );

  try {
    await app.start();

    // Direct gpuWorker / audioWorker / ioWorker messages are wired and sanitized.
    gpuWorker.dispatch('message', {
      data: {
        type: 'gpu-stats',
        realPackDrawCount: Number.NaN,
        fallbackDrawCount: Number.POSITIVE_INFINITY,
        presentCount: -99,
        unboundBindGroupCount: 'not-a-number',
        skippedDrawCount: true,
        validationErrorCount: Number.NEGATIVE_INFINITY,
        lastValidationErrorCode: 'C:\\Users\\secret\\code',
        lastValidationErrorMessage:
          'Shader fault in /Users/alice/project/foo.wgsl\n    at compile (/home/alice/x.js:1:1)',
      },
    });

    audioWorker.dispatch('message', {
      data: {
        type: 'audio-stats',
        outputState: 'corrupted-state-value',
        framesDrained: Number.NaN,
        underruns: -10,
        overruns: { bad: true },
      },
    });

    ioWorker.dispatch('message', {
      data: {
        type: 'loader-stats',
        indirectDispatch: 12345,
        pthreadSpawnedCount: Number.POSITIVE_INFINITY,
        pthreadActiveCount: -5,
        pthreadPeakCount: Number.NaN,
        pthreadLoadQueueDepth: 'NaN',
        preferredShaderPackUrl: '/home/alice/private.wgsl.pack',
      },
    });

    // Null / primitive / malformed worker messages must never throw.
    mockWorker.dispatch('message', { data: null });
    mockWorker.dispatch('message', { data: undefined });
    mockWorker.dispatch('message', { data: 'plain string' });
    mockWorker.dispatch('message', { data: 42 });
    mockWorker.dispatch('message', { data: { type: 'gpu-stats', stats: null } });
    mockWorker.dispatch('message', { data: { type: 'audio-stats', stats: 'bad' } });

    const snap = app.getBootTelemetry();
    assert.equal(snap.realPackDrawCount, 0);
    assert.equal(snap.fallbackDrawCount, 0);
    assert.equal(snap.presentCount, 0);
    assert.equal(snap.unboundBindGroupCount, 0);
    assert.equal(snap.skippedDrawCount, 0);
    assert.equal(snap.validationErrorCount, 0);
    assert.equal(snap.lastValidationErrorCode, 'validation');
    assert.ok(snap.lastValidationErrorMessage);
    assert.ok(!snap.lastValidationErrorMessage.includes('alice'));
    assert.ok(!snap.lastValidationErrorMessage.includes('compile'));

    assert.equal(snap.outputState, 'unknown');
    assert.equal(snap.framesDrained, 0);
    assert.equal(snap.underruns, 0);
    assert.equal(snap.overruns, 0);

    assert.equal(snap.indirectDispatch, 'unknown');
    assert.equal(snap.pthreadSpawnedCount, 0);
    assert.equal(snap.pthreadActiveCount, 0);
    assert.equal(snap.pthreadPeakCount, 0);
    assert.equal(snap.pthreadLoadQueueDepth, 0);
    assert.equal(snap.preferredShaderPackUrl, DEFAULT_SHADER_PACK_URL);
  } finally {
    await app.destroy();
  }

  // External worker listeners are removed cleanly on destroy().
  assert.equal(gpuWorker.listenerCount(), 0);
  assert.equal(audioWorker.listenerCount(), 0);
  assert.equal(ioWorker.listenerCount(), 0);
});

test('W88 respects attachGlobalHook opt-out, frozen globalScope, and preserves replaced global hooks on destroy', async () => {
  // 1. attachGlobalHook: false skips all global telemetry hooks.
  const optOutScope = {};
  const optOutHost = createTestHost(
    { ok: true, hasWasmBinary: true },
    { globalScope: optOutScope, attachGlobalHook: false }
  );
  assert.equal(optOutScope.__playgta6, undefined);
  assert.equal(optOutScope.__playgta6BootState, undefined);
  assert.equal(optOutScope.__playgta6Telemetry, undefined);
  assert.equal(globalThis.__playgta6Telemetry, undefined);
  assert.equal(optOutHost.app.getBootTelemetry().indirectDispatch, 'unknown');
  await optOutHost.app.destroy();

  // 2. Frozen globalScope never throws.
  const frozenScope = Object.freeze({});
  const frozenHost = createTestHost(
    { ok: true, hasWasmBinary: true },
    { globalScope: frozenScope }
  );
  assert.equal(frozenScope.__playgta6BootState, undefined);
  assert.equal(typeof globalThis.__playgta6Telemetry, 'function');
  await frozenHost.app.destroy();
  assert.equal(globalThis.__playgta6Telemetry, undefined);

  // 3. Replacing global hooks before destroy() must not clobber the replacement.
  const customScope = {};
  const host = createTestHost(
    { ok: true, hasWasmBinary: true },
    { globalScope: customScope }
  );
  assert.ok(customScope.__playgta6BootState);
  assert.equal(typeof globalThis.__playgta6Telemetry, 'function');

  const sentinelBootState = Object.freeze({ sentinel: true });
  const sentinelTelemetry = Object.freeze(() => sentinelBootState);
  Object.defineProperty(customScope, '__playgta6BootState', {
    configurable: true,
    enumerable: true,
    value: sentinelBootState,
  });
  globalThis.__playgta6Telemetry = sentinelTelemetry;

  await host.app.destroy();
  assert.equal(customScope.__playgta6BootState, sentinelBootState);
  assert.equal(globalThis.__playgta6Telemetry, sentinelTelemetry);

  // Clean up test sentinel from globalThis.
  delete globalThis.__playgta6Telemetry;
});

test('W296 forwards 30 FPS frame timing, VRAM/bind-group/mailbox/JS heap telemetry into PerfMonitor, HudController, and GpuBootPanel', async () => {
  const { createRequire } = await import('node:module');
  const require = createRequire(import.meta.url);
  const { GpuBootPanelController } = require('../web/ui/gpu_boot_panel.js');
  const { HudController } = await import('../web/ui/hud_controller.js');

  const clock = createClock(40000);
  const hudController = new HudController({ now: clock.now });

  // Spy wrapper around a real HudController to confirm individual W289 methods are invoked.
  const hookCalls = {
    setFrameBudgetStats: [],
    setMemoryStats: [],
    setIoMailboxLanes: [],
    updatePerfStats: [],
  };
  const origSetFrameBudget = hudController.setFrameBudgetStats.bind(hudController);
  const origSetMemory = hudController.setMemoryStats.bind(hudController);
  const origSetMailbox = hudController.setIoMailboxLanes.bind(hudController);
  const origUpdatePerf = hudController.updatePerfStats.bind(hudController);
  hudController.setFrameBudgetStats = (stats, opts) => {
    hookCalls.setFrameBudgetStats.push(stats);
    return origSetFrameBudget(stats, opts);
  };
  hudController.setMemoryStats = (stats, opts) => {
    hookCalls.setMemoryStats.push(stats);
    return origSetMemory(stats, opts);
  };
  hudController.setIoMailboxLanes = (lanes, opts) => {
    hookCalls.setIoMailboxLanes.push(lanes);
    return origSetMailbox(lanes, opts);
  };
  hudController.updatePerfStats = (stats, opts) => {
    hookCalls.updatePerfStats.push(stats);
    return origUpdatePerf(stats, opts);
  };

  const makeDomNode = (tag = 'div') => {
    const attrs = new Map();
    const node = {
      tagName: String(tag).toUpperCase(),
      className: '',
      hidden: false,
      textContent: '',
      children: [],
      parentNode: null,
      style: {},
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
        child.parentNode = node;
        node.children.push(child);
        return child;
      },
      removeChild(child) {
        const idx = node.children.indexOf(child);
        if (idx >= 0) node.children.splice(idx, 1);
        child.parentNode = null;
        return child;
      },
      addEventListener() {},
      removeEventListener() {},
      querySelector() {
        return null;
      },
    };
    return node;
  };
  const panelDocument = {
    head: makeDomNode('head'),
    createElement: (tag) => makeDomNode(tag),
    querySelector: () => null,
  };
  const gpuBootPanel = new GpuBootPanelController({
    container: makeDomNode('div'),
    documentObj: panelDocument,
    now: clock.now,
  });

  const { app, mockWorker } = createTestHost(
    {
      ok: true,
      coopCoep: true,
      hasWasmBinary: true,
    },
    {
      hudController,
      gpuBootPanel,
      performanceObj: {
        memory: {
          usedJSHeapSize: 128 * 1024 * 1024,
          totalJSHeapSize: 256 * 1024 * 1024,
          jsHeapSizeLimit: 2048 * 1024 * 1024,
        },
      },
    }
  );

  try {
    const started = await app.start();
    assert.equal(started.ok, true);

    // 1. Within-budget frame + GPU VRAM/bind-group/mailbox metrics via gpu-stats.
    mockWorker.dispatch('message', {
      data: {
        type: 'gpu-stats',
        realPackDrawCount: 12,
        presentCount: 1,
        frameMs: 24.0,
        gpuMs: 11.0,
        cpuMs: 9.0,
        ioWaitMs: 2.0,
        shaderCompileMs: 2.0,
        frameJitterMs: 1.45,
        vramBudgetUtilization: 0.62,
        bindGroupHitRate: 0.94,
        mailboxLanes: 8,
      },
    });

    assert.equal(app.perfMonitor.getSnapshot().fps30ComplianceRate, 1);
    assert.equal(app.perfMonitor.getSnapshot().dominantBottleneck, 'within_budget');
    assert.equal(hudController.getTelemetry().frameBudget.fps30ComplianceRate, 1);
    assert.equal(hudController.getTelemetry().frameBudget.dominantBottleneck, 'within_budget');
    assert.equal(hudController.getTelemetry().memory.jsHeapUsedMb, 128);
    assert.equal(hudController.getTelemetry().memory.jsHeapTotalMb, 256);
    assert.equal(hudController.getTelemetry().io.mailboxLanes, 8);

    let panelState = gpuBootPanel.getState();
    assert.equal(panelState.fps30ComplianceRate, 1);
    assert.equal(panelState.frameJitterMs, 1.45);
    assert.equal(panelState.jsHeapUsedMb, 128);
    assert.equal(panelState.vramBudgetUtilization, 0.62);
    assert.equal(panelState.bindGroupHitRate, 0.94);
    assert.equal(panelState.mailboxLanes, 8);

    // 2. Over-budget I/O-bound frame via perf-stats -> PerfMonitor computes 50% 30 FPS compliance & io_bound.
    mockWorker.dispatch('message', {
      data: {
        type: 'perf-stats',
        frameMs: 44.0,
        ioWaitMs: 30.0,
        gpuMs: 8.0,
        cpuMs: 6.0,
      },
    });

    assert.equal(app.perfMonitor.getSnapshot().fps30ComplianceRate, 0.5);
    assert.equal(app.perfMonitor.getSnapshot().dominantBottleneck, 'io_bound');
    assert.equal(hudController.getTelemetry().frameBudget.fps30ComplianceRate, 0.5);
    assert.equal(hudController.getTelemetry().frameBudget.dominantBottleneck, 'io_bound');
    assert.equal(gpuBootPanel.getState().fps30ComplianceRate, 0.5);

    // 3. Dedicated io-stats and gpu-telemetry updates flow cleanly to HudController and GpuBootPanel.
    mockWorker.dispatch('message', {
      data: {
        type: 'io-stats',
        httpfsMailboxLanes: 6,
      },
    });
    mockWorker.dispatch('message', {
      data: {
        type: 'gpu-telemetry',
        frameJitterMs: 2.1,
        vramBudgetUtilization: 0.78,
        bindGroupHitRate: 0.97,
      },
    });

    assert.equal(hudController.getTelemetry().io.mailboxLanes, 6);
    panelState = gpuBootPanel.getState();
    assert.equal(panelState.frameJitterMs, 2.1);
    assert.equal(panelState.vramBudgetUtilization, 0.78);
    assert.equal(panelState.bindGroupHitRate, 0.97);
    assert.equal(panelState.mailboxLanes, 6);

    assert.ok(hookCalls.setFrameBudgetStats.length >= 2);
    assert.ok(hookCalls.setMemoryStats.length >= 1);
    assert.ok(hookCalls.setIoMailboxLanes.includes(8));
    assert.ok(hookCalls.setIoMailboxLanes.includes(6));
    assert.ok(hookCalls.updatePerfStats.length >= 3);

    const snap = app.getBootTelemetry();
    assert.equal(Object.isFrozen(snap.perf), true);
    assert.equal(snap.fps30ComplianceRate, 0.5);
    assert.equal(snap.dominantBottleneck, 'io_bound');
    assert.equal(snap.frameJitterMs, 2.1);
    assert.equal(snap.jsHeapUsedMb, 128);
    assert.equal(snap.jsHeapTotalMb, 256);
    assert.equal(snap.vramBudgetUtilization, 0.78);
    assert.equal(snap.bindGroupHitRate, 0.97);
    assert.equal(snap.mailboxLanes, 6);
    assert.equal(snap.perf.fps30ComplianceRate, 0.5);
    assert.equal(snap.perf.dominantBottleneck, 'io_bound');
    assert.equal(snap.perf.vramBudgetUtilization, 0.78);
    assert.equal(snap.perf.bindGroupHitRate, 0.97);
    assert.equal(snap.perf.mailboxLanes, 6);

    const diag = app.getDiagnosticsSnapshot();
    assert.equal(diag.fps30ComplianceRate, 0.5);
    assert.equal(diag.dominantBottleneck, 'io_bound');
    assert.equal(diag.frameJitterMs, 2.1);
    assert.equal(diag.jsHeapUsedMb, 128);
    assert.equal(diag.vramBudgetUtilization, 0.78);
    assert.equal(diag.bindGroupHitRate, 0.97);
    assert.equal(diag.mailboxLanes, 6);
  } finally {
    hudController.destroy();
    gpuBootPanel.destroy();
    await app.destroy();
  }
});

test('W296 ingests BootTracer fast-boot milestones (START_SCREEN, INTRO_VIDEO, MAIN_MENU, IN_GAME) from logs and worker events', async () => {
  const clock = createClock(50000);
  const panelUpdates = [];
  const mockGpuBootPanel = {
    update(payload) {
      panelUpdates.push(payload);
    },
  };

  const { app, mockWorker, broadcastInstances } = createTestHost(
    {
      ok: true,
      coopCoep: true,
      hasWasmBinary: true,
    },
    {
      now: clock.now,
      gpuBootPanel: mockGpuBootPanel,
    }
  );

  try {
    await app.start();

    // Reach FIRST_PRESENT via BroadcastChannel log.
    clock.advance(500);
    broadcastInstances[0].onmessage({
      data: { label: '[boot] FIRST_PRESENT' },
    });

    // Failure log containing MAIN_MENU must NOT falsely record MAIN_MENU.
    mockWorker.dispatch('message', {
      data: { type: 'log', message: 'MAIN_MENU failed to initialize: timeout error' },
    });
    assert.equal(app.bootTracer.getSummary().bootToMainMenuMs, null);

    // 1. Tagged log -> START_SCREEN (and PerfMonitor transitions to 'menu').
    clock.advance(700);
    mockWorker.dispatch('message', {
      data: { type: 'log', message: '[boot] START_SCREEN ready (press any button)' },
    });
    assert.equal(app.bootTracer.getSummary().bootToStartScreenMs, 1200);
    assert.equal(app.perfMonitor.currentPhase, 'menu');

    // 2. Untagged BroadcastChannel label -> INTRO_VIDEO.
    clock.advance(800);
    broadcastInstances[0].onmessage({
      data: { label: 'Opening intro video playback started' },
    });
    assert.equal(app.bootTracer.getSummary().bootToIntroVideoMs, 2000);

    // 3. Explicit worker event -> MAIN_MENU with explicit elapsedMs.
    mockWorker.dispatch('message', {
      data: { type: 'main-menu', elapsedMs: 4200 },
    });
    assert.equal(app.bootTracer.getSummary().bootToMainMenuMs, 4200);
    assert.equal(app.perfMonitor.currentPhase, 'menu');

    // 4. Structured boot-milestone worker event -> IN_GAME (and PerfMonitor transitions to 'gameplay').
    mockWorker.dispatch('message', {
      data: { type: 'boot-milestone', milestone: 'IN_GAME', elapsedMs: 7500 },
    });
    assert.equal(app.bootTracer.getSummary().bootToInGameMs, 7500);
    assert.equal(app.perfMonitor.currentPhase, 'gameplay');

    // Verify all 4 fast-boot milestones surface in getBootTelemetry() and getDiagnosticsSnapshot().
    const snap = app.getBootTelemetry();
    assert.equal(snap.milestone, 'IN_GAME');
    assert.equal(snap.bootToStartScreenMs, 1200);
    assert.equal(snap.bootToIntroVideoMs, 2000);
    assert.equal(snap.bootToMainMenuMs, 4200);
    assert.equal(snap.bootToInGameMs, 7500);
    assert.equal(Object.isFrozen(snap.fastBootMilestones), true);
    assert.deepEqual(snap.fastBootMilestones, {
      START_SCREEN: 1200,
      INTRO_VIDEO: 2000,
      MAIN_MENU: 4200,
      IN_GAME: 7500,
    });

    const diag = app.getDiagnosticsSnapshot();
    assert.equal(diag.bootToStartScreenMs, 1200);
    assert.equal(diag.bootToIntroVideoMs, 2000);
    assert.equal(diag.bootToMainMenuMs, 4200);
    assert.equal(diag.bootToInGameMs, 7500);

    // Panel received updates for fast-boot milestone progression.
    assert.ok(panelUpdates.some((u) => u.bootSummary && u.bootSummary.stage === 'IN_GAME'));
  } finally {
    await app.destroy();
  }
});

test('W296 queries StorageManager.getFastBootPlan() during boot and triggers runQuotaPressureEviction() when shouldEvictColdChunks is true', async () => {
  const { BrowserStorageManager } = await import('../web/storage/storage_manager.js');

  // Case A: Healthy storage plan (shouldEvictColdChunks: false) -> exposes plan, skips eviction.
  {
    const planCalls = [];
    let evictionCalls = 0;
    const postedMessages = [];
    const customWorker = createFakeEventTarget({
      postMessage(msg) {
        postedMessages.push(msg);
      },
      terminate() {},
    });

    const mockStorageManager = {
      async getFastBootPlan(opts) {
        planCalls.push(opts);
        return {
          warmCacheReady: true,
          recommendedSlabMaxBytes: 192 * 1024 * 1024,
          recommendedOpfsBudgetBytes: 512 * 1024 * 1024,
          quotaHeadroomBytes: 2 * 1024 * 1024 * 1024,
          pressureLevel: 'low',
          shouldEvictColdChunks: false,
        };
      },
      async runQuotaPressureEviction() {
        evictionCalls += 1;
        return { evictedBytes: 0, evictedEntries: 0, ok: true };
      },
    };

    const { app } = createTestHost(
      { ok: true, coopCoep: true, hasWasmBinary: true },
      {
        storageManager: mockStorageManager,
        createWasmWorker() {
          return customWorker;
        },
      }
    );

    try {
      const started = await app.start();
      assert.equal(started.ok, true);
      assert.equal(planCalls.length, 1);
      assert.equal(evictionCalls, 0);

      const plan = app.getFastBootPlan();
      assert.equal(Object.isFrozen(plan), true);
      assert.equal(plan.warmCacheReady, true);
      assert.equal(plan.recommendedSlabMaxBytes, 192 * 1024 * 1024);
      assert.equal(plan.pressureLevel, 'low');
      assert.equal(plan.shouldEvictColdChunks, false);

      const snap = app.getBootTelemetry();
      assert.deepEqual(snap.fastBootPlan, plan);
      assert.equal(snap.quotaEviction, null);
      assert.equal(Object.isFrozen(snap.storage), true);
      assert.deepEqual(snap.storage.fastBootPlan, plan);
      assert.equal(snap.storage.quotaEviction, null);
      assert.equal(postedMessages[0].recommendedSlabMaxBytes, 192 * 1024 * 1024);
    } finally {
      await app.destroy();
    }
  }

  // Case B: Real BrowserStorageManager under high quota pressure -> triggers runQuotaPressureEviction() cleanly.
  {
    let evictToTargetBytes = null;
    const mockOpfsCache = {
      getStats() {
        return {
          initialized: true,
          totalBytes: 200 * 1024 * 1024,
          entryCount: 12,
          shaderPackCount: 1,
          rpfChunkCount: 11,
        };
      },
      async evictToTarget(targetBytes) {
        evictToTargetBytes = targetBytes;
        return { evictedBytes: 64 * 1024 * 1024, evictedEntries: 4, ok: true };
      },
    };

    const realStorageManager = new BrowserStorageManager({
      storageObj: {
        async estimate() {
          return {
            usage: 900 * 1024 * 1024,
            quota: 1000 * 1024 * 1024,
          };
        },
        async persisted() {
          return true;
        },
        async getDirectory() {
          return null;
        },
      },
      opfsRoot: null,
    });
    realStorageManager.attachOpfsCache(mockOpfsCache);
    // Simulate OPFS supported via warmBootReadiness while delegating eviction to attached opfsCache.
    const origGetFastBootPlan = realStorageManager.getFastBootPlan.bind(realStorageManager);
    realStorageManager.getFastBootPlan = (opts) =>
      origGetFastBootPlan({
        ...opts,
        warmBootReadiness: {
          opfsSupported: true,
          warmBootReady: true,
          quotaBytes: 1000 * 1024 * 1024,
          usageBytes: 900 * 1024 * 1024,
          cachedBytes: 200 * 1024 * 1024,
        },
      });

    const { app } = createTestHost(
      { ok: true, coopCoep: true, hasWasmBinary: true },
      { storageManager: realStorageManager }
    );

    try {
      const started = await app.start();
      assert.equal(started.ok, true);

      const plan = app.getFastBootPlan();
      assert.equal(plan.pressureLevel, 'critical');
      assert.equal(plan.shouldEvictColdChunks, true);
      assert.ok(evictToTargetBytes > 0);

      const snap = app.getBootTelemetry();
      assert.equal(Object.isFrozen(snap.quotaEviction), true);
      assert.deepEqual(snap.quotaEviction, {
        triggered: true,
        ok: true,
        evictedBytes: 64 * 1024 * 1024,
        evictedEntries: 4,
      });
      assert.deepEqual(snap.storage.quotaEviction, snap.quotaEviction);
    } finally {
      await app.destroy();
    }
  }
});

test('W296 defensive feature checks tolerate legacy mocks and sanitize hostile/NaN telemetry safely', async () => {
  const legacyPerfMonitor = {};
  const markedMilestones = [];
  const legacyBootTracer = {
    markMilestone(name) {
      markedMilestones.push(name);
    },
    getSummary() {
      return { stage: 'FIRST_PRESENT' };
    },
  };
  const throwingStorageManager = {
    async getFastBootPlan() {
      throw new Error('quota API exploded');
    },
  };
  const throwingHudController = {
    setFrameBudgetStats() {
      throw new Error('frameBudget HUD crash');
    },
    setMemoryStats() {
      throw new Error('memory HUD crash');
    },
    setIoMailboxLanes() {
      throw new Error('mailbox HUD crash');
    },
    updatePerfStats() {
      throw new Error('perf HUD crash');
    },
  };

  const { app, mockWorker } = createTestHost(
    { ok: true, coopCoep: true, hasWasmBinary: true },
    {
      perfMonitor: legacyPerfMonitor,
      bootTracer: legacyBootTracer,
      storageManager: throwingStorageManager,
      hudController: throwingHudController,
    }
  );

  try {
    const started = await app.start();
    assert.equal(started.ok, true);
    assert.equal(app.getFastBootPlan(), null);

    // Hostile / out-of-range / NaN telemetry must sanitize safely without throwing.
    mockWorker.dispatch('message', {
      data: {
        type: 'perf-stats',
        fps30ComplianceRate: 99,
        dominantBottleneck: '<script>alert(1)</script>',
        frameJitterMs: Number.POSITIVE_INFINITY,
        vramBudgetUtilization: -5,
        bindGroupHitRate: Number.NaN,
        jsHeapUsedMb: -100,
        mailboxLanes: 0,
      },
    });

    // Legacy bootTracer without recordMainMenu falls back to markMilestone('MAIN_MENU').
    mockWorker.dispatch('message', {
      data: { type: 'main-menu', elapsedMs: 3100 },
    });
    assert.ok(markedMilestones.includes('MAIN_MENU'));

    const snap = app.getBootTelemetry();
    assert.equal(snap.fps30ComplianceRate, 1);
    assert.equal(snap.dominantBottleneck, null);
    assert.equal(snap.frameJitterMs, null);
    assert.equal(snap.vramBudgetUtilization, 0);
    assert.equal(snap.bindGroupHitRate, null);
    assert.equal(snap.jsHeapUsedMb, null);
    assert.equal(snap.mailboxLanes, null);
    assert.equal(snap.bootToMainMenuMs, 3100);
  } finally {
    await app.destroy();
  }
});

test('W366 app_bootstrap uses session.getPhaseTelemetry() (zero session.getStats() churn), fast-paths frame-presented labels, and throttles hidden-HUD churn', async () => {
  const clock = createClock(60000);
  let gpuStatsForwarded = 0;
  let memoryStatsForwarded = 0;
  let perfStatsForwarded = 0;
  const hiddenHudController = {
    visible: false,
    isVisible() {
      return this.visible;
    },
    updateGpuStats() {
      gpuStatsForwarded += 1;
    },
    setMemoryStats() {
      memoryStatsForwarded += 1;
    },
    updatePerfStats() {
      perfStatsForwarded += 1;
    },
  };

  const { app, mockWorker, broadcastInstances, hud } = createTestHost(
    { ok: true, coopCoep: true, hasWasmBinary: true },
    {
      now: clock.now,
      hudController: hiddenHudController,
      performanceObj: {
        memory: {
          usedJSHeapSize: 128 * 1024 * 1024,
          totalJSHeapSize: 256 * 1024 * 1024,
        },
      },
    }
  );

  let sessionGetStatsCalls = 0;
  let sessionGetPhaseTelemetryCalls = 0;
  const origSessionGetStats = app.session.getStats.bind(app.session);
  const origSessionGetPhaseTel = app.session.getPhaseTelemetry.bind(app.session);
  app.session.getStats = (...args) => {
    sessionGetStatsCalls += 1;
    return origSessionGetStats(...args);
  };
  app.session.getPhaseTelemetry = (...args) => {
    sessionGetPhaseTelemetryCalls += 1;
    return origSessionGetPhaseTel(...args);
  };

  try {
    const started = await app.start();
    assert.equal(started.ok, true);

    // 1. Repeated telemetry reads call lightweight session.getPhaseTelemetry() and NEVER session.getStats().
    for (let i = 0; i < 10; i += 1) {
      const bt = app.getBootTelemetry();
      const diag = app.getDiagnosticsSnapshot();
      const pt = app.getPhaseTelemetry();
      assert.equal(bt.sessionPhase, 'boot');
      assert.equal(diag.sessionPhase, 'boot');
      assert.equal(pt.currentPhase, 'boot');
    }
    assert.equal(sessionGetStatsCalls, 0, 'getBootTelemetry/getDiagnosticsSnapshot must not call session.getStats()');
    assert.equal(sessionGetPhaseTelemetryCalls, 30);

    // 2. BroadcastChannel per-frame progress labels bypass 32-source perf/memory scanning & HUD spam.
    const hudMetricsBefore = hud.metrics.length;
    for (let frame = 1; frame <= 120; frame += 1) {
      clock.advance(16);
      broadcastInstances[0].onmessage({
        data: { label: `WebGPU frame #${frame} presented` },
      });
    }
    assert.equal(app.perfMonitor.getTotalFrames(), 120);
    assert.equal(app.getBootTelemetry().presentCount, 120);
    assert.equal(sessionGetStatsCalls, 0);
    assert.equal(memoryStatsForwarded, 0, 'pure frame-presented labels must not spam setMemoryStats');
    assert.equal(perfStatsForwarded, 0, 'pure frame-presented labels must not spam updatePerfStats');
    // 120 frames * 16 ms = 1920 ms -> at 500 ms hidden-HUD throttle, <= 6 hud.updateMetrics calls (vs 20+ unthrottled).
    const hiddenHudMetricCalls = hud.metrics.length - hudMetricsBefore;
    assert.ok(hiddenHudMetricCalls >= 1 && hiddenHudMetricCalls <= 6, `expected <= 6 hidden HUD metric updates, got ${hiddenHudMetricCalls}`);

    // 3. Rapid worker frame-presented messages while HUD is hidden throttle forwardGpuStatsToObservers
    //    while keeping getBootTelemetry().presentCount 100% up-to-date.
    const gpuBefore = gpuStatsForwarded;
    for (let frame = 121; frame <= 180; frame += 1) {
      clock.advance(16);
      mockWorker.dispatch('message', {
        data: { type: 'frame-presented', presentCount: frame },
      });
    }
    assert.equal(app.getBootTelemetry().presentCount, 180);
    const hiddenGpuForwards = gpuStatsForwarded - gpuBefore;
    assert.ok(hiddenGpuForwards >= 1 && hiddenGpuForwards <= 4, `expected <= 4 hidden GPU observer forwards, got ${hiddenGpuForwards}`);
    assert.equal(sessionGetStatsCalls, 0);
    // performance.memory fallback sampled once on first frame-presented and throttled thereafter.
    assert.equal(memoryStatsForwarded, 1);

    // 4. Explicit session.getStats() still works and stays 100% accurate whenever explicitly queried.
    const explicitStats = app.session.getStats();
    assert.equal(explicitStats.currentPhase, 'boot');
    assert.equal(sessionGetStatsCalls, 1);
  } finally {
    await app.destroy();
  }
});

test('Orch-13 gpu-validation-error anti-double-counting across session, HUD, crashReporter, BroadcastChannel, duplicates, and totalCount reconciliation', async () => {
  const validationErrors = [];
  const hudLogs = [];
  const hudMetrics = [];
  const sharedHud = {
    validationErrors,
    logs: hudLogs,
    metrics: hudMetrics,
    setStatus() {},
    appendLog(message, level = 'info') {
      hudLogs.push({ message, level });
    },
    updateMetrics(metrics) {
      hudMetrics.push({ ...metrics });
    },
    recordValidationError(ev) {
      validationErrors.push(ev);
    },
  };

  const { app, mockWorker, broadcastInstances } = createTestHost(
    {
      ok: true,
      coopCoep: true,
      hasWasmBinary: true,
    },
    { hud: sharedHud }
  );

  try {
    const started = await app.start();
    assert.equal(started.ok, true);
    assert.equal(started.mode, SESSION_MODE.WASM);
    assert.equal(broadcastInstances.length, 1);

    // 1. Single gpu-validation-error delivered to app_bootstrap (with session sharing the same hud
    //    and crashReporter) PLUS mirrored BroadcastChannel progress with gpuValidationForwarded: true.
    mockWorker.dispatch('message', {
      data: {
        type: 'gpu-validation-error',
        seq: 1,
        timestampMs: 50_001,
        errorType: 'validation',
        message: 'Invalid bind group layout',
        totalCount: 1,
      },
    });
    broadcastInstances[0].onmessage({
      data: {
        error: 'wgpu_worker: [validation] Invalid bind group layout',
        gpuValidationForwarded: true,
      },
    });

    assert.equal(app.getBootTelemetry().validationErrorCount, 1);
    assert.equal(app.session.getValidationErrorCount(), 1);
    assert.equal(validationErrors.length, 1, 'hud.recordValidationError must be called once');
    assert.equal(
      app.crashReporter.getErrors().length,
      1,
      'crashReporter must record the validation error once without BroadcastChannel worker_error duplication'
    );
    assert.equal(
      hudLogs.filter((entry) => /WebGPU validation|Invalid bind group layout/i.test(entry.message)).length,
      0,
      'session must not duplicate hud.appendLog when hud.recordValidationError is present'
    );

    // 2. Identical duplicate gpu-validation-error (same seq + timestampMs + message) is deduplicated.
    mockWorker.dispatch('message', {
      data: {
        type: 'gpu-validation-error',
        seq: 1,
        timestampMs: 50_001,
        errorType: 'validation',
        message: 'Invalid bind group layout',
        totalCount: 1,
      },
    });

    assert.equal(app.getBootTelemetry().validationErrorCount, 1);
    assert.equal(app.session.getValidationErrorCount(), 1);
    assert.equal(validationErrors.length, 1);
    assert.equal(app.crashReporter.getErrors().length, 1);

    // 3. Periodic gpu-stats arriving ahead of an in-flight gpu-validation-error reconciles via
    //    totalCount instead of overshooting validationErrorCount (2 -> 2, never 3).
    mockWorker.dispatch('message', {
      data: {
        type: 'gpu-stats',
        validationErrorCount: 2,
      },
    });
    assert.equal(app.getBootTelemetry().validationErrorCount, 2);

    mockWorker.dispatch('message', {
      data: {
        type: 'gpu-validation-error',
        seq: 2,
        timestampMs: 50_050,
        errorType: 'validation',
        message: 'Pipeline layout mismatch',
        totalCount: 2,
      },
    });

    assert.equal(app.getBootTelemetry().validationErrorCount, 2);
    assert.equal(validationErrors.length, 2);
    assert.equal(app.crashReporter.getErrors().length, 2);
  } finally {
    await app.destroy();
  }
});

test('Orch-32 P1 #4a + P1 #4b + P2 #5: holdUntilVisible query + requireNonBlackFrame + gpuMemory + swapchainProbeSupported sync', async () => {
  const { parseSplashHoldQuery } = await import('../web/app_bootstrap.js');

  // 1. parseSplashHoldQuery accepts both ?holdUntilVisible= and ?splashHold= (last wins).
  assert.equal(parseSplashHoldQuery('?holdUntilVisible=0'), false);
  assert.equal(parseSplashHoldQuery('?holdUntilVisible=false'), false);
  assert.equal(parseSplashHoldQuery('?holdUntilVisible=off'), false);
  assert.equal(parseSplashHoldQuery('?holdUntilVisible=no'), false);
  assert.equal(parseSplashHoldQuery('?holdUntilVisible=1'), true);
  assert.equal(parseSplashHoldQuery('?holdUntilVisible=true'), true);
  assert.equal(parseSplashHoldQuery('?holdUntilVisible=on'), true);
  assert.equal(parseSplashHoldQuery('?holdUntilVisible=yes'), true);
  assert.equal(parseSplashHoldQuery('?splashHold=1&holdUntilVisible=0'), false);
  assert.equal(parseSplashHoldQuery('?holdUntilVisible=0&splashHold=1'), true);

  // 2. ?holdUntilVisible=0 resolves holdUntilVisible=false and forwards requireNonBlackFrame: false to worker init.
  const { app, mockWorker } = createTestHost(
    {
      ok: true,
      coopCoep: true,
      hasWasmBinary: true,
    },
    {
      urlSearch: '?holdUntilVisible=0',
      defaultHoldUntilVisible: true,
      attachGlobalHook: false,
    }
  );

  const postedInits = [];
  mockWorker.postMessage = (msg) => {
    postedInits.push(msg);
  };

  try {
    await app.start();
    assert.equal(app.getBootTelemetry().holdUntilVisible, false);
    assert.equal(app.getDiagnosticsSnapshot().holdUntilVisible, false);
    assert.ok(postedInits.length >= 1);
    assert.equal(postedInits[0].requireNonBlackFrame, false);

    // 3. gpu-telemetry carries gpuMemory (with monotonic peaks) + syncs swapchainProbeSupported to inputBridge.
    mockWorker.dispatch('message', {
      data: {
        type: 'gpu-telemetry',
        swapchainProbeSupported: true,
        frameStats: {
          framesPresented: 3,
          droppedFrames: 0,
          gpuMemory: {
            liveBufferBytes: 1000,
            liveTextureBytes: 2000,
            peakBufferBytes: 1500,
            peakTextureBytes: 2500,
            stagingPoolBytes: 500,
            totalLiveGpuBytes: 3500,
          },
        },
      },
    });
    assert.equal(app.inputBridge.getStats().swapchainProbeSupported, true);
    assert.equal(app.getBootTelemetry().gpu.gpuMemory.liveBufferBytes, 1000);
    assert.equal(app.getBootTelemetry().gpu.gpuMemory.totalLiveGpuBytes, 3500);
    assert.equal(app.getDiagnosticsSnapshot().gpuMemory.peakTextureBytes, 2500);

    // Lower live counts on later sample preserve monotonic peak counters.
    mockWorker.dispatch('message', {
      data: {
        type: 'gpu-telemetry',
        frameStats: {
          framesPresented: 4,
          droppedFrames: 0,
          gpuMemory: {
            liveBufferBytes: 800,
            liveTextureBytes: 1200,
            stagingPoolBytes: 400,
            totalLiveGpuBytes: 2400,
          },
        },
      },
    });
    assert.equal(app.getBootTelemetry().gpu.gpuMemory.liveBufferBytes, 800);
    assert.equal(app.getBootTelemetry().gpu.gpuMemory.peakBufferBytes, 1500);
    assert.equal(app.getBootTelemetry().gpu.gpuMemory.peakTextureBytes, 2500);
    assert.equal(app.getDiagnosticsSnapshot().gpuMemory.totalLiveGpuBytes, 2400);
  } finally {
    await app.destroy();
  }
});
