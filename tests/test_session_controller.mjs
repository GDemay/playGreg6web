import test from 'node:test';
import assert from 'node:assert/strict';

import RuntimeSessionControllerDefault, {
  DEFAULT_STATUS_URL,
  SESSION_MODE,
  SESSION_STATE,
  RuntimeSessionController,
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

