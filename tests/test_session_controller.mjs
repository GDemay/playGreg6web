import test from 'node:test';
import assert from 'node:assert/strict';

import RuntimeSessionControllerDefault, {
  BOOT_MODE,
  DEFAULT_MAX_VALIDATION_ERRORS,
  DEFAULT_STATUS_URL,
  SESSION_MODE,
  SESSION_STATE,
  RuntimeSessionController,
  detectWasm64,
  probeBrowserCapabilities,
} from '../web/runtime/session_controller.js';
import { PerfMonitor } from '../web/telemetry/perf_monitor.js';

function createClock(startMs = 1000) {
  let current = startMs;
  return {
    now: () => current,
    advance(deltaMs) {
      current += deltaMs;
      return current;
    },
    set(ms) {
      current = ms;
      return current;
    },
  };
}

function createFakeTimers() {
  let nextId = 1;
  const active = new Map();
  const clearedIds = [];
  return {
    setInterval(fn, ms) {
      const id = nextId++;
      active.set(id, { fn, ms });
      return id;
    },
    clearInterval(id) {
      clearedIds.push(id);
      active.delete(id);
    },
    tickAll() {
      for (const entry of [...active.values()]) {
        entry.fn();
      }
    },
    activeCount() {
      return active.size;
    },
    clearedIds,
  };
}

function createMockHud() {
  const statuses = [];
  const logs = [];
  const metricsUpdates = [];
  return {
    statuses,
    logs,
    metricsUpdates,
    setStatus(text, level = 'info') {
      statuses.push({ text, level });
    },
    appendLog(message, level = 'info') {
      logs.push({ message, level });
    },
    updateMetrics(metrics) {
      metricsUpdates.push(metrics);
    },
  };
}

function createMockFetch(payload, { ok = true, status = 200, throwError = null } = {}) {
  const calls = [];
  const fetchImpl = async (url, init) => {
    calls.push({ url, init });
    if (throwError) {
      throw throwError;
    }
    return {
      ok,
      status,
      async json() {
        return typeof payload === 'function' ? payload() : payload;
      },
    };
  };
  fetchImpl.calls = calls;
  return fetchImpl;
}

function createMockEventTarget() {
  const listeners = new Map();
  return {
    addEventListener(type, fn) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add(fn);
    },
    removeEventListener(type, fn) {
      listeners.get(type)?.delete(fn);
    },
    listenerCount(type) {
      if (type) return listeners.get(type)?.size ?? 0;
      let total = 0;
      for (const set of listeners.values()) total += set.size;
      return total;
    },
    dispatch(type) {
      for (const fn of [...(listeners.get(type) || [])]) {
        fn({ type });
      }
    },
  };
}

function fullCapabilities(overrides = {}) {
  return {
    crossOriginIsolated: true,
    sharedArrayBuffer: true,
    webgpu: true,
    wasm64: true,
    opfs: true,
    gamepad: true,
    audioWorklet: true,
    ...overrides,
  };
}

function readyStatus(overrides = {}) {
  return {
    ok: true,
    coopCoep: true,
    dataDirConfigured: true,
    hasManifest: true,
    hasBootset: true,
    hasWasmBinary: true,
    shaderPackCount: 2,
    ...overrides,
  };
}

test('exports, defaults, and constructor argument validation', () => {
  assert.equal(RuntimeSessionControllerDefault, RuntimeSessionController);
  assert.equal(DEFAULT_STATUS_URL, '/api/status');
  assert.deepEqual(SESSION_STATE, {
    IDLE: 'idle',
    PROBING: 'probing',
    STARTING: 'starting',
    RUNNING_WASM: 'running_wasm',
    RUNNING_SMOKE: 'running_smoke',
    STOPPED: 'stopped',
    ERROR: 'error',
  });
  assert.deepEqual(SESSION_MODE, {
    NONE: 'none',
    WASM: 'wasm',
    SMOKE: 'smoke',
  });

  const controller = new RuntimeSessionController({ fetchImpl: null });
  assert.equal(controller.getState(), SESSION_STATE.IDLE);
  assert.equal(controller.getMode(), SESSION_MODE.NONE);

  const diag = controller.getDiagnostics();
  assert.equal(diag.state, 'idle');
  assert.equal(diag.mode, 'none');
  assert.equal(diag.capabilities, null);
  assert.equal(diag.serverStatus, null);
  assert.equal(diag.storageSummary, null);
  assert.equal(diag.perfSnapshot, null);
  assert.equal(diag.uptimeMs, 0);

  const caps = probeBrowserCapabilities();
  for (const key of [
    'crossOriginIsolated',
    'sharedArrayBuffer',
    'webgpu',
    'wasm64',
    'opfs',
    'gamepad',
    'audioWorklet',
  ]) {
    assert.equal(typeof caps[key], 'boolean');
  }

  assert.throws(() => new RuntimeSessionController(null), TypeError);
  assert.throws(() => new RuntimeSessionController({ statusUrl: '' }), TypeError);
  assert.throws(() => new RuntimeSessionController({ now: 'not-fn' }), TypeError);
  assert.throws(() => new RuntimeSessionController({ fetchImpl: 42 }), TypeError);
  assert.throws(() => new RuntimeSessionController({ startWasmRuntime: {} }), TypeError);
  assert.throws(() => new RuntimeSessionController({ startSmokeDemo: {} }), TypeError);
  assert.throws(() => new RuntimeSessionController({ capabilityProbe: 'nope' }), TypeError);
  assert.throws(() => controller.onStateChange('nope'), TypeError);
});

test('Wasm64 happy path: state transition order, /api/status probing, and diagnostics', async () => {
  const clock = createClock(5000);
  const hud = createMockHud();
  const fetchImpl = createMockFetch(readyStatus({ hasWasmBinary: true }));
  const storageSummary = {
    totalManagedBytes: 4096,
    totalManagedFormatted: '4 KiB',
  };
  const storageManager = {
    async getSummary() {
      return storageSummary;
    },
  };

  let wasmCalls = 0;
  let smokeCalls = 0;
  let stoppedCount = 0;

  const controller = new RuntimeSessionController({
    fetchImpl,
    statusUrl: '/api/status',
    hud,
    storageManager,
    capabilityProbe: () => fullCapabilities(),
    now: clock.now,
    startWasmRuntime: async (ctx) => {
      wasmCalls += 1;
      assert.equal(ctx.mode, SESSION_MODE.WASM);
      assert.equal(ctx.serverStatus.hasWasmBinary, true);
      assert.equal(ctx.capabilities.webgpu, true);
      assert.deepEqual(ctx.storageSummary, storageSummary);
      return {
        stop() {
          stoppedCount += 1;
        },
      };
    },
    startSmokeDemo: async () => {
      smokeCalls += 1;
      return { stop() {} };
    },
  });

  const events = [];
  const unsubscribe = controller.onStateChange((ev) => {
    events.push({ prev: ev.previousState, next: ev.state, mode: ev.mode });
  });

  const res = await controller.start();
  assert.equal(res.ok, true);
  assert.equal(res.state, SESSION_STATE.RUNNING_WASM);
  assert.equal(res.mode, SESSION_MODE.WASM);
  assert.equal(res.fallbackReason, null);
  assert.equal(wasmCalls, 1);
  assert.equal(smokeCalls, 0);
  assert.equal(fetchImpl.calls.length, 1);
  assert.equal(fetchImpl.calls[0].url, '/api/status');
  assert.deepEqual(fetchImpl.calls[0].init, { cache: 'no-store' });

  assert.deepEqual(events, [
    { prev: 'idle', next: 'probing', mode: 'none' },
    { prev: 'probing', next: 'starting', mode: 'none' },
    { prev: 'starting', next: 'running_wasm', mode: 'wasm' },
  ]);

  clock.advance(2500);
  const diag = controller.getDiagnostics();
  assert.equal(diag.state, SESSION_STATE.RUNNING_WASM);
  assert.equal(diag.mode, SESSION_MODE.WASM);
  assert.equal(diag.uptimeMs, 2500);
  assert.equal(diag.serverStatus.hasWasmBinary, true);
  assert.equal(diag.serverStatus.shaderPackCount, 2);
  assert.deepEqual(diag.storageSummary, storageSummary);
  assert.equal(diag.capabilities.crossOriginIsolated, true);
  assert.equal(diag.capabilities.webgpu, true);

  unsubscribe();
  await controller.stop();
  assert.equal(stoppedCount, 1);
  assert.equal(controller.getState(), SESSION_STATE.STOPPED);
  assert.equal(controller.getDiagnostics().uptimeMs, 0);
  // Unsubscribed listener did not receive the 'stopped' event.
  assert.equal(events.length, 3);
});

test('automatic fallback to running_smoke when hasWasmBinary is false and logs fallback reason to HUD', async () => {
  const hud = createMockHud();
  const fetchImpl = createMockFetch(readyStatus({ hasWasmBinary: false }));
  let wasmCalls = 0;
  let smokeCalls = 0;

  const controller = new RuntimeSessionController({
    fetchImpl,
    hud,
    capabilityProbe: () => fullCapabilities(),
    startWasmRuntime: async () => {
      wasmCalls += 1;
      return { stop() {} };
    },
    startSmokeDemo: async (ctx) => {
      smokeCalls += 1;
      assert.equal(ctx.mode, SESSION_MODE.SMOKE);
      assert.equal(ctx.serverStatus.hasWasmBinary, false);
      return { stop() {} };
    },
  });

  const states = [];
  controller.onStateChange((ev) => states.push(ev.state));

  const res = await controller.start();
  assert.equal(res.ok, true);
  assert.equal(res.state, SESSION_STATE.RUNNING_SMOKE);
  assert.equal(res.mode, SESSION_MODE.SMOKE);
  assert.equal(wasmCalls, 0);
  assert.equal(smokeCalls, 1);
  assert.deepEqual(states, ['probing', 'starting', 'running_smoke']);
  assert.match(res.fallbackReason, /hasWasmBinary=false/i);
  assert.ok(
    hud.logs.some((entry) => entry.level === 'warn' && /smoke demo/i.test(entry.message)),
    'Expected HUD warning log explaining fallback to smoke demo'
  );
  assert.equal(controller.getDiagnostics().fallbackReason, res.fallbackReason);
});

test('preferSmoke: true forces WebGPU G5 smoke demo even when hasWasmBinary is true', async () => {
  const hud = createMockHud();
  const fetchImpl = createMockFetch(readyStatus({ hasWasmBinary: true }));
  let wasmCalls = 0;
  let smokeCalls = 0;

  const controller = new RuntimeSessionController({
    fetchImpl,
    hud,
    capabilityProbe: () => fullCapabilities(),
    startWasmRuntime: async () => {
      wasmCalls += 1;
      return { stop() {} };
    },
    startSmokeDemo: async () => {
      smokeCalls += 1;
      return { stop() {} };
    },
  });

  const res = await controller.start({ preferSmoke: true });
  assert.equal(res.ok, true);
  assert.equal(res.state, SESSION_STATE.RUNNING_SMOKE);
  assert.equal(res.mode, SESSION_MODE.SMOKE);
  assert.equal(wasmCalls, 0);
  assert.equal(smokeCalls, 1);
  assert.match(res.fallbackReason, /preferSmoke=true/i);
  assert.ok(hud.logs.some((entry) => /preferSmoke=true/i.test(entry.message)));
});

test('probeEnvironment handles offline/network and HTTP errors gracefully via crashReporter and falls back to smoke demo', async () => {
  const hud = createMockHud();
  const reportedErrors = [];
  const crashReporter = {
    captureError(err, ctx) {
      reportedErrors.push({ message: err.message, ctx });
    },
  };

  const netFetch = createMockFetch(null, { throwError: new Error('ECONNREFUSED') });
  const controller = new RuntimeSessionController({
    fetchImpl: netFetch,
    hud,
    crashReporter,
    capabilityProbe: () => fullCapabilities(),
    startWasmRuntime: async () => ({ stop() {} }),
    startSmokeDemo: async () => ({ stop() {} }),
  });

  // Direct probeEnvironment must never throw unhandled.
  const probe = await controller.probeEnvironment();
  assert.equal(probe.ready, true);
  assert.equal(probe.serverStatus.ok, false);
  assert.equal(probe.serverStatus.offline, true);
  assert.equal(probe.serverStatus.hasWasmBinary, false);
  assert.match(probe.serverStatus.error, /ECONNREFUSED/);
  assert.equal(reportedErrors.length, 1);
  assert.equal(reportedErrors[0].ctx.source, 'status');

  // Starting while offline falls back cleanly to smoke demo.
  const res = await controller.start();
  assert.equal(res.ok, true);
  assert.equal(res.state, SESSION_STATE.RUNNING_SMOKE);
  assert.equal(res.mode, SESSION_MODE.SMOKE);
  assert.match(res.fallbackReason, /unreachable/i);

  // HTTP 503 status response also captured without throwing.
  const http503Fetch = createMockFetch(null, { ok: false, status: 503 });
  const c2 = new RuntimeSessionController({
    fetchImpl: http503Fetch,
    crashReporter,
    capabilityProbe: () => fullCapabilities(),
  });
  const p2 = await c2.probeEnvironment();
  assert.equal(p2.serverStatus.ok, false);
  assert.equal(p2.serverStatus.httpStatus, 503);
  assert.ok(reportedErrors.some((r) => r.ctx.httpStatus === 503));
});

test('missing crossOriginIsolated or webgpu transitions to error, reports to crashReporter, and blocks launch', async () => {
  for (const scenario of [
    {
      name: 'missing crossOriginIsolated',
      caps: fullCapabilities({ crossOriginIsolated: false }),
      expectedPattern: /Cross-Origin Isolation/i,
      expectedMissing: 'crossOriginIsolated',
    },
    {
      name: 'missing webgpu',
      caps: fullCapabilities({ webgpu: false }),
      expectedPattern: /WebGPU is unavailable/i,
      expectedMissing: 'webgpu',
    },
    {
      name: 'missing both crossOriginIsolated and webgpu',
      caps: fullCapabilities({ crossOriginIsolated: false, webgpu: false }),
      expectedPattern: /WebGPU is unavailable/i,
      expectedMissing: 'webgpu',
    },
    {
      name: 'missing sharedArrayBuffer',
      caps: fullCapabilities({ sharedArrayBuffer: false }),
      expectedPattern: /SharedArrayBuffer is unavailable/i,
      expectedMissing: 'sharedArrayBuffer',
    },
  ]) {
    const hud = createMockHud();
    const reported = [];
    const crashReporter = {
      recordError(err, ctx) {
        reported.push({ err, ctx });
      },
    };
    let launched = false;
    const states = [];

    const controller = new RuntimeSessionController({
      fetchImpl: createMockFetch(readyStatus()),
      hud,
      crashReporter,
      capabilityProbe: () => scenario.caps,
      startWasmRuntime: async () => {
        launched = true;
        return { stop() {} };
      },
      startSmokeDemo: async () => {
        launched = true;
        return { stop() {} };
      },
    });
    controller.onStateChange((ev) => states.push(ev.state));

    const res = await controller.start();
    assert.equal(res.ok, false, scenario.name);
    assert.equal(res.state, SESSION_STATE.ERROR, scenario.name);
    assert.equal(res.mode, SESSION_MODE.NONE, scenario.name);
    assert.equal(launched, false, `${scenario.name} must not launch runtime`);
    assert.deepEqual(states, ['probing', 'error'], scenario.name);
    assert.ok(res.reasons.some((r) => scenario.expectedPattern.test(r)), scenario.name);
    assert.equal(reported.length, 1, scenario.name);
    assert.equal(reported[0].ctx.code, 'MISSING_BROWSER_CAPABILITIES', scenario.name);
    assert.ok(reported[0].ctx.missingCapabilities.includes(scenario.expectedMissing), scenario.name);
    assert.ok(hud.logs.some((l) => l.level === 'error' && scenario.expectedPattern.test(l.message)));
    assert.equal(controller.getDiagnostics().diagnosticReasons.length, res.reasons.length);
  }
});

test('PerfMonitor + HUD wiring, InputBridge start/stop, and AudioEngine user-gesture registration', async () => {
  const clock = createClock(2000);
  const timers = createFakeTimers();
  const hud = createMockHud();
  const perfMonitor = new PerfMonitor({
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
  });

  let inputStarts = 0;
  let inputStops = 0;
  const inputBridge = {
    running: false,
    start() {
      inputStarts += 1;
      this.running = true;
    },
    stop() {
      inputStops += 1;
      this.running = false;
    },
  };

  const gestureTarget = createMockEventTarget();
  let audioGestureRegistrations = 0;
  let audioGestureCleanups = 0;
  const audioEngine = {
    resumeOnUserGesture(target) {
      audioGestureRegistrations += 1;
      const onGesture = () => {};
      const resolved = target || gestureTarget;
      resolved.addEventListener('pointerdown', onGesture);
      resolved.addEventListener('keydown', onGesture);
      return () => {
        audioGestureCleanups += 1;
        resolved.removeEventListener('pointerdown', onGesture);
        resolved.removeEventListener('keydown', onGesture);
      };
    },
  };

  const controller = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus({ hasWasmBinary: true })),
    hud,
    perfMonitor,
    inputBridge,
    audioEngine,
    audioGestureTarget: gestureTarget,
    capabilityProbe: () => fullCapabilities(),
    now: clock.now,
    startWasmRuntime: async () => ({ stop() {} }),
  });

  const startRes = await controller.start();
  assert.equal(startRes.ok, true);
  assert.equal(perfMonitor.isHudAttached, true);
  assert.equal(timers.activeCount(), 1);
  assert.equal(inputStarts, 1);
  assert.equal(inputBridge.running, true);
  assert.equal(audioGestureRegistrations, 1);
  assert.equal(gestureTarget.listenerCount(), 2);

  // Record frames and trigger HUD interval tick.
  perfMonitor.recordFrameTimeMs(16.0);
  perfMonitor.recordFrameTimeMs(16.0);
  perfMonitor.recordGpuBatch({ packets: 12, drawCalls: 4, bytes: 512, durationMs: 16 });
  timers.tickAll();

  assert.equal(hud.metricsUpdates.length, 1);
  assert.equal(hud.metricsUpdates[0].gpuPackets, 12);
  assert.ok(Math.abs(hud.metricsUpdates[0].fps - 62.5) < 0.01);

  const diag = controller.getDiagnostics();
  assert.ok(diag.perfSnapshot);
  assert.equal(diag.perfSnapshot.gpu.totalGpuPackets, 12);

  // Stop detaches PerfMonitor, stops InputBridge, and removes audio gesture listeners.
  await controller.stop();
  assert.equal(perfMonitor.isHudAttached, false);
  assert.equal(timers.activeCount(), 0);
  assert.equal(inputStops, 1);
  assert.equal(inputBridge.running, false);
  assert.equal(audioGestureCleanups, 1);
  assert.equal(gestureTarget.listenerCount(), 0);
});

test('restart() cleans up handles, timers, and listeners without leaks across multiple cycles', async () => {
  const clock = createClock(100);
  const timers = createFakeTimers();
  const hud = createMockHud();
  const perfMonitor = new PerfMonitor({
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
  });
  const gestureTarget = createMockEventTarget();

  let inputStarts = 0;
  let inputStops = 0;
  const inputBridge = {
    running: false,
    start() {
      inputStarts += 1;
      this.running = true;
    },
    stop() {
      inputStops += 1;
      this.running = false;
    },
  };

  const audioEngine = {
    resumeOnUserGesture(target) {
      const fn = () => {};
      target.addEventListener('pointerdown', fn);
      target.addEventListener('keydown', fn);
      return () => {
        target.removeEventListener('pointerdown', fn);
        target.removeEventListener('keydown', fn);
      };
    },
  };

  const activeHandles = new Set();
  let handleSeq = 0;

  const makeHandle = (kind) => {
    const id = `${kind}-${++handleSeq}`;
    activeHandles.add(id);
    return {
      id,
      stop() {
        activeHandles.delete(id);
      },
    };
  };

  const controller = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus({ hasWasmBinary: true })),
    hud,
    perfMonitor,
    inputBridge,
    audioEngine,
    audioGestureTarget: gestureTarget,
    capabilityProbe: () => fullCapabilities(),
    now: clock.now,
    startWasmRuntime: async () => makeHandle('wasm'),
    startSmokeDemo: async () => makeHandle('smoke'),
  });

  const transitions = [];
  controller.onStateChange((ev) => transitions.push(ev.state));

  // Cycle 1: Wasm64
  const r1 = await controller.start();
  assert.equal(r1.state, SESSION_STATE.RUNNING_WASM);
  assert.equal(activeHandles.size, 1);
  assert.equal(timers.activeCount(), 1);
  assert.equal(gestureTarget.listenerCount(), 2);

  // Cycle 2: Restart into smoke demo
  const r2 = await controller.restart({ preferSmoke: true });
  assert.equal(r2.state, SESSION_STATE.RUNNING_SMOKE);
  assert.equal(activeHandles.size, 1);
  assert.ok(activeHandles.has('smoke-2'));
  assert.equal(timers.activeCount(), 1);
  assert.equal(gestureTarget.listenerCount(), 2);

  // Cycle 3: Restart back into Wasm64
  const r3 = await controller.restart({ preferSmoke: false });
  assert.equal(r3.state, SESSION_STATE.RUNNING_WASM);
  assert.equal(activeHandles.size, 1);
  assert.ok(activeHandles.has('wasm-3'));
  assert.equal(timers.activeCount(), 1);
  assert.equal(gestureTarget.listenerCount(), 2);

  // Final stop leaves zero active handles, timers, or listeners.
  await controller.stop();
  assert.equal(activeHandles.size, 0);
  assert.equal(timers.activeCount(), 0);
  assert.equal(gestureTarget.listenerCount(), 0);
  assert.equal(inputStarts, 3);
  assert.equal(inputStops, 3);

  assert.deepEqual(transitions, [
    'probing',
    'starting',
    'running_wasm',
    'stopped',
    'probing',
    'starting',
    'running_smoke',
    'stopped',
    'probing',
    'starting',
    'running_wasm',
    'stopped',
  ]);
});

test('idempotent stop() invokes handle teardown once and emits stopped only once', async () => {
  let stopCalls = 0;
  let inputStopCalls = 0;

  const inputBridge = {
    running: false,
    start() {
      this.running = true;
    },
    stop() {
      inputStopCalls += 1;
      this.running = false;
    },
  };

  const controller = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus({ hasWasmBinary: false })),
    inputBridge,
    capabilityProbe: () => fullCapabilities(),
    startSmokeDemo: async () => () => {
      stopCalls += 1;
    },
  });

  const states = [];
  controller.onStateChange((ev) => states.push(ev.state));

  await controller.start();
  const s1 = await controller.stop();
  const s2 = await controller.stop();
  const s3 = await controller.stop();

  assert.equal(s1.ok, true);
  assert.equal(s1.alreadyStopped, false);
  assert.equal(s2.ok, true);
  assert.equal(s2.alreadyStopped, true);
  assert.equal(s3.ok, true);
  assert.equal(s3.alreadyStopped, true);

  assert.equal(stopCalls, 1);
  assert.equal(inputStopCalls, 1);
  assert.deepEqual(states, ['probing', 'starting', 'running_smoke', 'stopped']);
});

test('launcher failure transitions starting -> error, reports to crashReporter, and allows recovery', async () => {
  const hud = createMockHud();
  const reported = [];
  const crashReporter = {
    captureException(err, ctx) {
      reported.push({ message: err.message, ctx });
    },
  };

  let shouldFail = true;
  const controller = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus({ hasWasmBinary: true })),
    hud,
    crashReporter,
    capabilityProbe: () => fullCapabilities(),
    startWasmRuntime: async () => {
      if (shouldFail) {
        throw new Error('Wasm64 instantiation OOM');
      }
      return { dispose() {} };
    },
  });

  const states = [];
  controller.onStateChange((ev) => states.push(ev.state));

  const failed = await controller.start();
  assert.equal(failed.ok, false);
  assert.equal(failed.state, SESSION_STATE.ERROR);
  assert.equal(failed.mode, SESSION_MODE.NONE);
  assert.match(failed.error.message, /Wasm64 instantiation OOM/);
  assert.equal(reported.length, 1);
  assert.equal(reported[0].ctx.phase, 'starting');
  assert.equal(controller.getDiagnostics().lastError, 'Wasm64 instantiation OOM');

  // Recovery from error state via start() succeeds cleanly.
  shouldFail = false;
  const recovered = await controller.start();
  assert.equal(recovered.ok, true);
  assert.equal(recovered.state, SESSION_STATE.RUNNING_WASM);
  assert.deepEqual(states, ['probing', 'starting', 'error', 'probing', 'starting', 'running_wasm']);
});

test('mid-flight stop() cancels pending start() and disposes late-resolving handle', async () => {
  let resolveLaunch;
  let lateDisposed = false;

  const controller = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus({ hasWasmBinary: true })),
    capabilityProbe: () => fullCapabilities(),
    startWasmRuntime: () =>
      new Promise((resolve) => {
        resolveLaunch = resolve;
      }),
  });

  const startPromise = controller.start();
  // Yield until start() enters launcher.
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(controller.getState(), SESSION_STATE.STARTING);

  const stopRes = await controller.stop();
  assert.equal(stopRes.state, SESSION_STATE.STOPPED);

  resolveLaunch({
    terminate() {
      lateDisposed = true;
    },
  });

  const startRes = await startPromise;
  assert.equal(startRes.ok, false);
  assert.equal(startRes.aborted, true);
  assert.equal(lateDisposed, true);
  assert.equal(controller.getState(), SESSION_STATE.STOPPED);
});

test('restart() while an earlier start() is in flight cancels the first and launches a fresh session', async () => {
  let resolveFirst;
  let firstDisposed = false;
  let calls = 0;

  const controller = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus({ hasWasmBinary: true })),
    capabilityProbe: () => fullCapabilities(),
    startWasmRuntime: () => {
      calls += 1;
      if (calls === 1) {
        return new Promise((resolve) => {
          resolveFirst = resolve;
        });
      }
      return Promise.resolve({ stop() {} });
    },
    startSmokeDemo: async () => ({ stop() {} }),
  });

  const firstStart = controller.start();
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(controller.getState(), SESSION_STATE.STARTING);

  const restartPromise = controller.restart({ preferSmoke: true });
  const restarted = await restartPromise;
  assert.equal(restarted.ok, true);
  assert.equal(restarted.state, SESSION_STATE.RUNNING_SMOKE);
  assert.equal(controller.getState(), SESSION_STATE.RUNNING_SMOKE);

  resolveFirst({
    stop() {
      firstDisposed = true;
    },
  });
  const firstRes = await firstStart;
  assert.equal(firstRes.aborted, true);
  assert.equal(firstDisposed, true);
  assert.equal(controller.getState(), SESSION_STATE.RUNNING_SMOKE);
});

test('W60 detectWasm64 supports spec-standard address: i64, legacy index: i64, BigInt rejection, and type reflection', () => {
  // Simulates WebIDL ToNumber coercion on i32 memory limits, which throws TypeError on BigInt.
  function assertI32Limit(value) {
    return +value;
  }

  // 1. Modern Chrome/V8/Firefox/Node engine accepting only `{ address: 'i64' }` with BigInt limits.
  const modernCalls = [];
  const modernWasmApi = {
    Memory: class MockModernMemory {
      constructor(desc) {
        modernCalls.push(desc);
        if (desc.address === 'i64') {
          if (typeof desc.initial !== 'bigint' || typeof desc.maximum !== 'bigint') {
            throw new TypeError('Expected BigInt limits for i64 memory');
          }
          return;
        }
        assertI32Limit(desc.initial);
      }
      type() {
        return { minimum: 1n, maximum: 2n, address: 'i64' };
      }
    },
  };
  assert.equal(detectWasm64(modernWasmApi), true);
  assert.equal(modernCalls.length, 1);
  assert.equal(modernCalls[0].address, 'i64');
  assert.equal(modernCalls[0].initial, 1n);
  assert.equal(modernCalls[0].maximum, 2n);

  // 2. Older memory64 proposal engine accepting only legacy `{ index: 'i64' }`.
  const legacyCalls = [];
  const legacyWasmApi = {
    Memory: class MockLegacyMemory {
      constructor(desc) {
        legacyCalls.push(desc);
        if (desc.index === 'i64') {
          if (typeof desc.initial !== 'bigint' || typeof desc.maximum !== 'bigint') {
            throw new TypeError('Expected BigInt limits for i64 memory');
          }
          return;
        }
        // Unknown `address` key is ignored -> treated as i32 -> BigInt throws TypeError.
        assertI32Limit(desc.initial);
      }
    },
  };
  assert.equal(detectWasm64(legacyWasmApi), true);
  assert.equal(legacyCalls.length, 2);
  assert.equal(legacyCalls[0].address, 'i64');
  assert.equal(legacyCalls[1].index, 'i64');

  // 3. Non-Wasm64 engine that ignores both keys and throws on BigInt initial/maximum.
  const wasm32Calls = [];
  const wasm32OnlyApi = {
    Memory: class MockWasm32Memory {
      constructor(desc) {
        wasm32Calls.push(desc);
        assertI32Limit(desc.initial);
      }
    },
  };
  assert.equal(detectWasm64(wasm32OnlyApi), false);
  assert.equal(wasm32Calls.length, 2);
  assert.equal(wasm32Calls[0].address, 'i64');
  assert.equal(wasm32Calls[1].index, 'i64');

  // 4. Non-i64 type reflection must be rejected (both `address` and `index` fields).
  assert.equal(
    detectWasm64({
      Memory: class MockI32AddressReflectingMemory {
        type() {
          return { minimum: 1, maximum: 2, address: 'i32' };
        }
      },
    }),
    false
  );
  assert.equal(
    detectWasm64({
      Memory: class MockI32IndexReflectingMemory {
        type() {
          return { minimum: 1, maximum: 2, index: 'i32' };
        }
      },
    }),
    false
  );

  // If `address` probe reflects i32, detectWasm64 still falls back to `index` probe.
  assert.equal(
    detectWasm64({
      Memory: class MockReflectFallbackMemory {
        constructor(desc) {
          this._addr = desc.index === 'i64' ? 'i64' : 'i32';
        }
        type() {
          return { minimum: 1n, maximum: 2n, index: this._addr };
        }
      },
    }),
    true
  );

  // Optional type-reflection errors are ignored once memory construction succeeds.
  assert.equal(
    detectWasm64({
      Memory: class MockThrowingReflectMemory {
        type() {
          throw new Error('type reflection unsupported');
        }
      },
    }),
    true
  );

  // 5. Defensive edge cases: missing/invalid API and zero-argument global probe.
  assert.equal(detectWasm64(null), false);
  assert.equal(detectWasm64(undefined), false);
  assert.equal(detectWasm64({}), false);
  assert.equal(detectWasm64({ Memory: 'not-a-function' }), false);
  assert.equal(typeof detectWasm64(), 'boolean');
});

test('W60 probeBrowserCapabilities reflects detectWasm64 across address: i64, legacy index: i64, and non-Wasm64 engines', () => {
  const originalWebAssembly = globalThis.WebAssembly;
  const originalDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'WebAssembly');

  const setGlobalWebAssembly = (value) => {
    Object.defineProperty(globalThis, 'WebAssembly', {
      value,
      configurable: true,
      writable: true,
      enumerable: false,
    });
  };

  try {
    setGlobalWebAssembly({
      Memory: class MockModernMemory {
        constructor(desc) {
          if (desc.address !== 'i64') {
            return +desc.initial;
          }
        }
      },
    });
    assert.equal(probeBrowserCapabilities().wasm64, true);

    setGlobalWebAssembly({
      Memory: class MockLegacyMemory {
        constructor(desc) {
          if (desc.index !== 'i64') {
            return +desc.initial;
          }
        }
      },
    });
    assert.equal(probeBrowserCapabilities().wasm64, true);

    setGlobalWebAssembly({
      Memory: class MockWasm32Memory {
        constructor(desc) {
          return +desc.initial;
        }
      },
    });
    assert.equal(probeBrowserCapabilities().wasm64, false);

    setGlobalWebAssembly({
      Memory: class MockNonI64ReflectMemory {
        type() {
          return { address: 'i32' };
        }
      },
    });
    assert.equal(probeBrowserCapabilities().wasm64, false);

    setGlobalWebAssembly(null);
    assert.equal(probeBrowserCapabilities().wasm64, false);
  } finally {
    if (originalDescriptor) {
      Object.defineProperty(globalThis, 'WebAssembly', originalDescriptor);
    } else {
      globalThis.WebAssembly = originalWebAssembly;
    }
  }
});

// --- W72: Split-Module Boot Mode, WebGPU Validation Errors & Splash Presentation Telemetry ---

function createMockWorker() {
  const listeners = new Map();
  return {
    addEventListener(type, fn) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add(fn);
    },
    removeEventListener(type, fn) {
      listeners.get(type)?.delete(fn);
    },
    emitMessage(data) {
      for (const fn of [...(listeners.get('message') || [])]) {
        fn({ type: 'message', data });
      }
    },
    listenerCount(type) {
      if (type) return listeners.get(type)?.size ?? 0;
      let total = 0;
      for (const set of listeners.values()) total += set.size;
      return total;
    },
  };
}

function createExtendedMockHud() {
  const base = createMockHud();
  const bootStatuses = [];
  const bootModes = [];
  const validationErrors = [];
  const splashFrames = [];
  const telemetryEvents = [];
  return {
    ...base,
    bootStatuses,
    bootModes,
    validationErrors,
    splashFrames,
    telemetryEvents,
    updateBootStatus(status) {
      bootStatuses.push(status);
    },
    setBootMode(bootMode, meta) {
      bootModes.push({ bootMode, meta });
    },
    recordValidationError(ev) {
      validationErrors.push(ev);
    },
    onSplashFramePresented(ev) {
      splashFrames.push(ev);
    },
    updateTelemetry(ev) {
      telemetryEvents.push(ev);
    },
  };
}

test('W72 exports, defaults, and option/listener validation for telemetry & split-module APIs', () => {
  assert.deepEqual(BOOT_MODE, {
    IDLE: 'idle',
    REAL_WASM64: 'real-wasm64',
    SMOKE: 'smoke',
  });
  assert.equal(DEFAULT_MAX_VALIDATION_ERRORS, 32);

  const controller = new RuntimeSessionController({ fetchImpl: null });
  assert.equal(controller.getBootMode(), BOOT_MODE.IDLE);
  assert.deepEqual(controller.getSideModuleMetadata(), {
    sideModuleCount: 0,
    sideModuleUrls: [],
  });
  assert.equal(controller.getValidationErrorCount(), 0);
  assert.deepEqual(controller.getValidationErrors(), []);
  assert.deepEqual(controller.getSplashStats(), {
    framesPresented: 0,
    errorCount: 0,
    firstPresented: false,
    lastFrame: null,
  });

  const snap = controller.getSnapshot();
  assert.equal(snap.bootMode, 'idle');
  assert.equal(snap.sideModuleCount, 0);
  assert.deepEqual(snap.sideModuleUrls, []);
  assert.equal(snap.validationErrorCount, 0);
  assert.equal(snap.validationErrorHistorySize, 0);
  assert.deepEqual(snap.recentValidationErrors, []);
  assert.equal(snap.splashFramesPresented, 0);
  assert.equal(snap.splashFrameErrorCount, 0);
  assert.equal(snap.firstSplashPresented, false);
  assert.equal(snap.lastSplashFrame, null);

  assert.throws(() => new RuntimeSessionController({ maxValidationErrors: 0 }), RangeError);
  assert.throws(() => new RuntimeSessionController({ maxValidationErrors: -5 }), RangeError);
  assert.throws(() => new RuntimeSessionController({ maxValidationErrors: 1.5 }), RangeError);
  assert.throws(() => new RuntimeSessionController({ wgpuWorker: {} }), TypeError);
  assert.throws(() => new RuntimeSessionController({ onStatusUpdate: 123 }), TypeError);
  assert.throws(() => new RuntimeSessionController({ onTelemetry: 'bad' }), TypeError);
  assert.throws(() => new RuntimeSessionController({ onValidationError: {} }), TypeError);
  assert.throws(() => new RuntimeSessionController({ onSplashFrame: true }), TypeError);

  assert.throws(() => controller.onStatusUpdate('nope'), TypeError);
  assert.throws(() => controller.onTelemetry(null), TypeError);
  assert.throws(() => controller.onValidationError(42), TypeError);
  assert.throws(() => controller.onSplashFrame({}), TypeError);
  assert.throws(() => controller.attachWorker(null), TypeError);
  assert.throws(() => controller.attachWorker({}), TypeError);
});

test('W72 tracks bootMode (idle -> real-wasm64 -> smoke -> idle) and sanitizes split-module metadata from /api/status', async () => {
  const hud = createExtendedMockHud();
  const statusUpdates = [];
  const stateEvents = [];
  let capturedLaunchCtx = null;

  const fetchImpl = createMockFetch(
    readyStatus({
      hasWasmBinary: true,
      bootMode: 'real-wasm64',
      sideModuleCount: 5,
      sideModuleUrls: [
        '/dist/guest_part_0.wasm',
        '/dist/guest_part_1.wasm',
        '/dist/guest_part_0.wasm', // duplicate
        '../escape/evil.wasm',
        'C:\\Users\\alice\\secret.wasm',
        '/Users/alice/secret.wasm',
        'file:///etc/passwd.wasm',
        'javascript:alert(1).wasm',
        '/dist/not_a_wasm.js',
        'https://localhost:8000/dist/guest_part_2.wasm',
      ],
    })
  );

  const controller = new RuntimeSessionController({
    fetchImpl,
    hud,
    capabilityProbe: () => fullCapabilities(),
    startWasmRuntime: async (ctx) => {
      capturedLaunchCtx = ctx;
      return { stop() {} };
    },
    startSmokeDemo: async () => ({ stop() {} }),
  });

  const unsubStatus = controller.onStatusUpdate((ev) => statusUpdates.push(ev));
  controller.onStateChange((ev) => stateEvents.push(ev));

  assert.equal(controller.getBootMode(), 'idle');

  const res = await controller.start();
  assert.equal(res.ok, true);
  assert.equal(res.bootMode, 'real-wasm64');
  assert.equal(controller.getBootMode(), 'real-wasm64');

  const expectedSideUrls = [
    '/dist/guest_part_0.wasm',
    '/dist/guest_part_1.wasm',
    'https://localhost:8000/dist/guest_part_2.wasm',
  ];
  assert.equal(res.sideModuleCount, 3);
  assert.deepEqual(res.sideModuleUrls, expectedSideUrls);
  assert.deepEqual(controller.getSideModuleMetadata(), {
    sideModuleCount: 3,
    sideModuleUrls: expectedSideUrls,
  });

  // Launch context receives sanitized split-module metadata and callbacks.
  assert.equal(capturedLaunchCtx.bootMode, 'real-wasm64');
  assert.equal(capturedLaunchCtx.sideModuleCount, 3);
  assert.deepEqual(capturedLaunchCtx.sideModuleUrls, expectedSideUrls);
  assert.equal(typeof capturedLaunchCtx.onBootLifecycle, 'function');
  assert.equal(typeof capturedLaunchCtx.onWorkerMessage, 'function');
  assert.equal(typeof capturedLaunchCtx.onValidationError, 'function');
  assert.equal(typeof capturedLaunchCtx.onSplashFramePresented, 'function');

  // Diagnostics & snapshot reflect bootMode and side-module metadata without leaking mutations.
  const diag = controller.getDiagnostics();
  assert.equal(diag.bootMode, 'real-wasm64');
  assert.equal(diag.sideModuleCount, 3);
  assert.deepEqual(diag.sideModuleUrls, expectedSideUrls);
  assert.equal(diag.serverStatus.sideModuleCount, 3);
  assert.deepEqual(diag.serverStatus.sideModuleUrls, expectedSideUrls);
  diag.sideModuleUrls.push('/dist/mutated.wasm');
  assert.equal(controller.getSideModuleMetadata().sideModuleUrls.length, 3);

  // HUD and status/state listeners received bootMode + side-module info.
  assert.ok(hud.bootStatuses.some((s) => s.bootMode === 'real-wasm64' && s.sideModuleCount === 3));
  assert.ok(hud.bootModes.some((m) => m.bootMode === 'real-wasm64' && m.meta.sideModuleCount === 3));
  assert.ok(hud.logs.some((l) => /Split-module Wasm64 layout detected \(3 side modules\)/i.test(l.message)));
  assert.ok(statusUpdates.some((u) => u.bootMode === 'real-wasm64' && u.sideModuleCount === 3));
  assert.equal(stateEvents[stateEvents.length - 1].bootMode, 'real-wasm64');
  assert.equal(stateEvents[stateEvents.length - 1].sideModuleCount, 3);

  // Restart into smoke demo updates bootMode to 'smoke'.
  const smokeRes = await controller.restart({ preferSmoke: true });
  assert.equal(smokeRes.ok, true);
  assert.equal(smokeRes.bootMode, 'smoke');
  assert.equal(controller.getBootMode(), 'smoke');
  assert.equal(controller.getSnapshot().bootMode, 'smoke');

  // Stop resets bootMode to 'idle'.
  unsubStatus();
  const countBeforeStop = statusUpdates.length;
  const stopRes = await controller.stop();
  assert.equal(stopRes.bootMode, 'idle');
  assert.equal(controller.getBootMode(), 'idle');
  assert.equal(statusUpdates.length, countBeforeStop);
});

test('W72 supports summary-only sideModuleCount and dynamic boot lifecycle callbacks', async () => {
  const statusUpdates = [];
  const fetchImpl = createMockFetch(
    readyStatus({
      hasWasmBinary: true,
      sideModuleCount: 4,
    })
  );

  const controller = new RuntimeSessionController({
    fetchImpl,
    capabilityProbe: () => fullCapabilities(),
    onStatusUpdate: (ev) => statusUpdates.push(ev),
    startWasmRuntime: async (ctx) => {
      // Preflight count from /api/status is preserved even without explicit URLs.
      assert.equal(ctx.sideModuleCount, 4);
      assert.deepEqual(ctx.sideModuleUrls, []);

      // Boot lifecycle callback discovers actual side-module URLs during startup.
      ctx.onBootLifecycle({
        bootMode: 'wasm-guest', // alias normalized to 'real-wasm64'
        sideModuleUrls: ['/dist/guest_part_0.wasm', '/dist/guest_part_1.wasm'],
        stage: 'side-modules-discovered',
      });
      return { stop() {} };
    },
  });

  const res = await controller.start();
  assert.equal(res.ok, true);
  assert.equal(res.bootMode, 'real-wasm64');
  assert.equal(res.sideModuleCount, 2);
  assert.deepEqual(res.sideModuleUrls, ['/dist/guest_part_0.wasm', '/dist/guest_part_1.wasm']);

  // Direct notifyBootLifecycle updates & normalizes aliases without throwing on bad inputs.
  const updated = controller.notifyBootLifecycle({
    bootMode: 'smoke-demo',
    sideModuleCount: 0,
    sideModuleUrls: [],
    reason: 'fallback-triggered',
  });
  assert.equal(updated.bootMode, 'smoke');
  assert.equal(updated.sideModuleCount, 0);
  assert.deepEqual(updated.sideModuleUrls, []);
  assert.equal(controller.getBootMode(), 'smoke');

  assert.doesNotThrow(() => controller.notifyBootLifecycle(null));
  assert.doesNotThrow(() => controller.notifyBootLifecycle('invalid'));
  assert.ok(statusUpdates.some((u) => u.reason === 'side-modules-discovered'));
  assert.ok(statusUpdates.some((u) => u.reason === 'fallback-triggered'));

  await controller.stop();
});

test('W72 handles gpu-validation-error messages with sanitization, bounded history, counters, and CrashReporter/HUD forwarding', async () => {
  const clock = createClock(10_000);
  const hud = createExtendedMockHud();
  const reported = [];
  const crashReporter = {
    captureError(err, ctx) {
      reported.push({ message: err.message, ctx });
    },
  };
  const validationEvents = [];
  const telemetryEvents = [];

  const controller = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus({ hasWasmBinary: true })),
    hud,
    crashReporter,
    now: clock.now,
    maxValidationErrors: 3,
    capabilityProbe: () => fullCapabilities(),
    startWasmRuntime: async () => ({ stop() {} }),
  });

  const unsubVal = controller.onValidationError((ev) => validationEvents.push(ev));
  const unsubTel = controller.onTelemetry((ev) => telemetryEvents.push(ev));

  await controller.start();

  // 1. G35 WgpuSwapchain-shaped nested error with host paths & embedded byte dump.
  const hexBlob = 'ab'.repeat(32);
  const r1 = controller.handleWorkerMessage({
    data: {
      type: 'gpu-validation-error',
      source: 'wgpu_swapchain',
      error: {
        seq: 1,
        type: 'validation',
        message: `Shader compile error in C:\\Users\\alice\\gta6\\shader.wgsl and /Users/alice/work/dump.bin payload=${hexBlob}\nnext line`,
        timestampMs: 10_050,
      },
    },
  });
  assert.equal(r1.handled, true);
  assert.equal(r1.validationErrorCount, 1);
  assert.equal(r1.validationErrorHistorySize, 1);
  assert.equal(r1.record.type, 'validation');
  assert.equal(r1.record.workerSeq, 1);
  assert.doesNotMatch(r1.record.message, /alice|C:\\|\/Users|abababab/);
  assert.match(r1.record.message, /<path>/);
  assert.match(r1.record.message, /<buffer>/);

  // 2. Out-of-memory flat message format.
  clock.advance(100);
  const r2 = controller.handleWorkerMessage({
    type: 'gpu-validation-error',
    errorType: 'out-of-memory',
    message: 'GPUOutOfMemoryError allocating 256MB texture at file:///home/dev/app.js',
  });
  assert.equal(r2.handled, true);
  assert.equal(r2.record.type, 'out-of-memory');
  assert.doesNotMatch(r2.record.message, /\/home\/dev/);

  // 3. Internal error with oversized message (>512 chars) is truncated.
  clock.advance(100);
  const longMsg = `Internal driver fault: ${'bind-group mismatch detail; '.repeat(30)}`;
  const r3 = controller.handleWorkerMessage({
    type: 'gpu-validation-error',
    error: { type: 'internal', message: longMsg },
  });
  assert.equal(r3.record.type, 'internal');
  assert.equal(r3.record.message.length, 512);
  assert.ok(r3.record.message.endsWith('...'));

  // 4. Fourth error evicts oldest from bounded history (maxValidationErrors = 3),
  //    while total validationErrorCount remains monotonic (4).
  clock.advance(100);
  controller.handleWorkerMessage({
    type: 'gpu-validation-error',
    message: 'Bind group layout mismatch at binding 2',
  });

  assert.equal(controller.getValidationErrorCount(), 4);
  const history = controller.getValidationErrors();
  assert.equal(history.length, 3);
  assert.deepEqual(
    history.map((e) => e.seq),
    [2, 3, 4]
  );
  assert.equal(history[2].type, 'validation');
  assert.equal(history[2].message, 'Bind group layout mismatch at binding 2');

  // Snapshot & diagnostics expose counts and bounded history cleanly.
  const diag = controller.getDiagnostics();
  assert.equal(diag.validationErrorCount, 4);
  assert.equal(diag.validationErrorHistorySize, 3);
  assert.equal(diag.recentValidationErrors.length, 3);

  // HUD, CrashReporter, and listeners all received sanitized events.
  assert.equal(hud.validationErrors.length, 4);
  assert.equal(validationEvents.length, 4);
  assert.equal(
    telemetryEvents.filter((e) => e.type === 'gpu-validation-error').length,
    4
  );
  assert.equal(reported.length, 4);
  assert.equal(reported[1].ctx.category, 'webgpu_validation');
  assert.equal(reported[1].ctx.errorType, 'out-of-memory');
  assert.equal(reported[1].ctx.bootMode, 'real-wasm64');

  // clearValidationErrors clears history buffer but preserves cumulative count.
  controller.clearValidationErrors();
  assert.equal(controller.getValidationErrors().length, 0);
  assert.equal(controller.getValidationErrorCount(), 4);

  unsubVal();
  unsubTel();
  await controller.stop();
});

test('W72 handles splash-frame-presented messages (success, firstPresent, and failure) from wgpu_worker.js', async () => {
  const clock = createClock(20_000);
  const hud = createExtendedMockHud();
  const splashEvents = [];
  const telemetryEvents = [];

  const controller = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus({ hasWasmBinary: true })),
    hud,
    now: clock.now,
    capabilityProbe: () => fullCapabilities(),
    startWasmRuntime: async () => ({ stop() {} }),
  });

  controller.onSplashFrame((ev) => splashEvents.push(ev));
  controller.onTelemetry((ev) => telemetryEvents.push(ev));

  await controller.start();

  // First presented splash frame.
  const s1 = controller.handleWorkerMessage({
    data: {
      type: 'splash-frame-presented',
      frameIndex: 0,
      framesPresented: 1,
      ok: true,
      firstPresent: true,
    },
  });
  assert.equal(s1.handled, true);
  assert.equal(s1.splashFramesPresented, 1);
  assert.equal(s1.firstSplashPresented, true);
  assert.equal(s1.record.ok, true);
  assert.equal(s1.record.frameIndex, 0);
  assert.equal(s1.record.firstPresent, true);

  // Subsequent presented splash frame.
  clock.advance(33);
  const s2 = controller.handleWorkerMessage({
    type: 'splash-frame-presented',
    frameIndex: 1,
    framesPresented: 2,
    ok: true,
  });
  assert.equal(s2.handled, true);
  assert.equal(s2.splashFramesPresented, 2);
  assert.equal(s2.record.firstPresent, false);

  // Failed splash presentation (e.g. WebGPU device not ready yet) sanitizes error & counts failure separately.
  clock.advance(33);
  const s3 = controller.handleWorkerMessage({
    type: 'splash-frame-presented',
    frameIndex: 2,
    framesPresented: 2,
    ok: false,
    reason: 'device-not-ready',
    error: 'WebGPU device not ready at C:\\Users\\bob\\worker.js',
  });
  assert.equal(s3.handled, true);
  assert.equal(s3.splashFramesPresented, 2);
  assert.equal(s3.splashFrameErrorCount, 1);
  assert.equal(s3.record.ok, false);
  assert.doesNotMatch(s3.record.error, /bob|C:\\/);
  assert.match(s3.record.error, /<path>/);

  const stats = controller.getSplashStats();
  assert.equal(stats.framesPresented, 2);
  assert.equal(stats.errorCount, 1);
  assert.equal(stats.firstPresented, true);
  assert.equal(stats.lastFrame.frameIndex, 2);
  assert.equal(stats.lastFrame.ok, false);

  const diag = controller.getDiagnostics();
  assert.equal(diag.splashFramesPresented, 2);
  assert.equal(diag.splashFrameErrorCount, 1);
  assert.equal(diag.firstSplashPresented, true);

  assert.equal(hud.splashFrames.length, 3);
  assert.equal(splashEvents.length, 3);
  assert.equal(
    telemetryEvents.filter((e) => e.type === 'splash-frame-presented').length,
    3
  );
  assert.ok(hud.logs.some((l) => l.level === 'ok' && /First splash frame presented/i.test(l.message)));
  assert.ok(hud.logs.some((l) => l.level === 'warn' && /Splash frame presentation warning/i.test(l.message)));

  await controller.stop();
});

test('W72 attaches/detaches wgpuWorker without listener leaks across restart and isolates throwing observers', async () => {
  const worker = createMockWorker();
  const throwingHud = {
    setStatus() {
      throw new Error('HUD setStatus boom');
    },
    appendLog() {
      throw new Error('HUD appendLog boom');
    },
    updateBootStatus() {
      throw new Error('HUD updateBootStatus boom');
    },
    setBootMode() {
      throw new Error('HUD setBootMode boom');
    },
    recordValidationError() {
      throw new Error('HUD recordValidationError boom');
    },
    onSplashFramePresented() {
      throw new Error('HUD onSplashFramePresented boom');
    },
    updateTelemetry() {
      throw new Error('HUD updateTelemetry boom');
    },
  };
  const throwingCrashReporter = {
    captureError() {
      throw new Error('CrashReporter boom');
    },
  };

  let receivedByGoodListener = 0;

  const controller = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus({ hasWasmBinary: true })),
    hud: throwingHud,
    crashReporter: throwingCrashReporter,
    wgpuWorker: worker,
    capabilityProbe: () => fullCapabilities(),
    startWasmRuntime: async () => ({ stop() {} }),
  });

  controller.onStatusUpdate(() => {
    throw new Error('status observer boom');
  });
  controller.onValidationError(() => {
    throw new Error('validation observer boom');
  });
  controller.onSplashFrame(() => {
    throw new Error('splash observer boom');
  });
  controller.onTelemetry(() => {
    throw new Error('telemetry observer boom');
  });
  controller.onTelemetry(() => {
    receivedByGoodListener += 1;
  });

  await controller.start();
  assert.equal(worker.listenerCount('message'), 1);

  // Duplicate attachWorker is idempotent (no duplicate listener).
  const detach = controller.attachWorker(worker);
  assert.equal(worker.listenerCount('message'), 1);

  // Worker messages flow in cleanly and throwing observers never break delivery.
  assert.doesNotThrow(() => {
    worker.emitMessage({
      type: 'gpu-validation-error',
      error: { type: 'validation', message: 'Invalid texture view' },
    });
    worker.emitMessage({
      type: 'splash-frame-presented',
      frameIndex: 0,
      framesPresented: 1,
      ok: true,
      firstPresent: true,
    });
    worker.emitMessage({ type: 'unrelated-worker-event', foo: 1 });
    worker.emitMessage(null);
  });

  assert.equal(controller.getValidationErrorCount(), 1);
  assert.equal(controller.getSplashStats().framesPresented, 1);
  assert.ok(receivedByGoodListener >= 2);

  // Unknown/malformed handleWorkerMessage inputs return handled: false without throwing.
  assert.deepEqual(controller.handleWorkerMessage(null), { handled: false, type: null });
  assert.deepEqual(controller.handleWorkerMessage(123), { handled: false, type: null });
  assert.deepEqual(controller.handleWorkerMessage({ type: 'gpu-telemetry' }), {
    handled: false,
    type: 'gpu-telemetry',
  });

  // Restart resets per-session counters and re-attaches wgpuWorker with exactly 1 listener.
  await controller.restart();
  assert.equal(worker.listenerCount('message'), 1);
  assert.equal(controller.getValidationErrorCount(), 0);
  assert.equal(controller.getValidationErrors().length, 0);
  assert.equal(controller.getSplashStats().framesPresented, 0);

  // Detach & stop leave zero worker listeners.
  detach();
  detach(); // idempotent
  await controller.stop();
  assert.equal(worker.listenerCount('message'), 0);
});

