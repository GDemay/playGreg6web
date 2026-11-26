import test from 'node:test';
import assert from 'node:assert/strict';

import RuntimeSessionControllerDefault, {
  BOOT_MODE,
  DEFAULT_MAX_TRANSITION_HISTORY,
  DEFAULT_MAX_VALIDATION_ERRORS,
  DEFAULT_STATUS_URL,
  SESSION_MODE,
  SESSION_PHASE,
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

// --- W148: Transition History Ring Buffer & Session Uptime Telemetry ---

test('W148 defaults, constructor option validation (maxTransitionHistory, nowFn), and initial telemetry state', () => {
  assert.equal(DEFAULT_MAX_TRANSITION_HISTORY, 64);

  assert.throws(() => new RuntimeSessionController({ maxTransitionHistory: 0 }), RangeError);
  assert.throws(() => new RuntimeSessionController({ maxTransitionHistory: -4 }), RangeError);
  assert.throws(() => new RuntimeSessionController({ maxTransitionHistory: 2.5 }), RangeError);
  assert.throws(() => new RuntimeSessionController({ maxTransitionHistory: '64' }), RangeError);
  assert.throws(() => new RuntimeSessionController({ maxTransitionHistory: NaN }), RangeError);
  assert.throws(() => new RuntimeSessionController({ nowFn: 'not-a-fn' }), TypeError);
  assert.throws(() => new RuntimeSessionController({ nowFn: 123 }), TypeError);

  const clock = createClock(10_000);
  const controller = new RuntimeSessionController({
    fetchImpl: null,
    nowFn: clock.now,
  });

  assert.deepEqual(controller.getTransitionHistory(), []);
  assert.equal(controller.getSessionUptimeMs(), 0);
  assert.equal(controller.getSessionUptimeMs(25_000), 0);
  assert.equal(controller.getRestartCount(), 0);
  assert.equal(controller.getLastTransitionReason(), null);

  assert.deepEqual(controller.getStateDurations(), {
    idle: 0,
    probing: 0,
    starting: 0,
    running_wasm: 0,
    running_smoke: 0,
    stopped: 0,
    error: 0,
  });

  clock.advance(250);
  assert.equal(controller.getStateDurations().idle, 250);
  assert.equal(controller.getStateDurations(10_750).idle, 750);

  const snap = controller.getSnapshot();
  assert.deepEqual(snap.transitionHistory, []);
  assert.equal(snap.transitionCount, 0);
  assert.equal(snap.transitionHistorySize, 0);
  assert.equal(snap.transitionEvictedCount, 0);
  assert.equal(snap.maxTransitionHistory, 64);
  assert.equal(snap.sessionUptimeMs, 0);
  assert.equal(snap.uptimeMs, 0);
  assert.equal(snap.lastTransitionReason, null);
  assert.equal(snap.restartCount, 0);
  assert.equal(snap.recoveryCount, 0);
  assert.equal(snap.stateDurations.idle, 250);

  const diag = controller.getDiagnostics(11_000);
  assert.equal(diag.stateDurations.idle, 1000);
  assert.equal(diag.sessionUptimeMs, 0);
});

test('W148 exact per-state duration tracking, transition records, and uptime computation', async () => {
  const clock = createClock(1_000);
  const statusEvents = [];
  const telemetryEvents = [];

  const controller = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus({ hasWasmBinary: true })),
    nowFn: clock.now,
    capabilityProbe: async () => {
      // Spend 150ms in PROBING.
      clock.advance(150);
      return fullCapabilities();
    },
    startWasmRuntime: async () => {
      // Spend 350ms in STARTING.
      clock.advance(350);
      return {
        async stop() {
          // Handle teardown takes 25ms before transitioning to STOPPED.
          clock.advance(25);
        },
      };
    },
    onStatusUpdate: (ev) => statusEvents.push(ev),
    onTelemetry: (ev) => telemetryEvents.push(ev),
  });

  // Spend 200ms in IDLE before start().
  clock.advance(200);

  const startRes = await controller.start();
  assert.equal(startRes.ok, true);
  assert.equal(controller.getState(), SESSION_STATE.RUNNING_WASM);

  // Immediately upon entering running_wasm (t = 1700), uptime is 0.
  assert.equal(controller.getSessionUptimeMs(), 0);
  assert.equal(controller.getSessionUptimeMs(2_700), 1_000);
  // Clock before startedAtMs clamps to 0.
  assert.equal(controller.getSessionUptimeMs(500), 0);

  // Advance 1200ms in RUNNING_WASM (t = 2900).
  clock.advance(1_200);
  assert.equal(controller.getSessionUptimeMs(), 1_200);
  assert.equal(controller.getSessionUptimeMs(3_400), 1_700);

  const durationsWhileRunning = controller.getStateDurations();
  assert.deepEqual(durationsWhileRunning, {
    idle: 200,
    probing: 150,
    starting: 350,
    running_wasm: 1_200,
    running_smoke: 0,
    stopped: 0,
    error: 0,
  });
  // Total duration across all states equals total elapsed wall time (2900 - 1000 = 1900).
  const sumRunning = Object.values(durationsWhileRunning).reduce((a, b) => a + b, 0);
  assert.equal(sumRunning, 1_900);

  // Explicit nowMs override on getStateDurations and getSnapshot.
  assert.equal(controller.getStateDurations(3_900).running_wasm, 2_200);
  const snapAt3900 = controller.getSnapshot(3_900);
  assert.equal(snapAt3900.sessionUptimeMs, 2_200);
  assert.equal(snapAt3900.uptimeMs, 2_200);
  assert.equal(snapAt3900.stateDurations.running_wasm, 2_200);

  // Stop at t = 2900 (+25ms handle stop -> transition to stopped at t = 2925).
  await controller.stop();
  assert.equal(controller.getState(), SESSION_STATE.STOPPED);
  assert.equal(controller.getSessionUptimeMs(), 0);
  assert.equal(controller.getSessionUptimeMs(9_999), 0);

  // Spend 75ms in STOPPED (t = 3000).
  clock.advance(75);
  const durationsAfterStop = controller.getStateDurations();
  assert.deepEqual(durationsAfterStop, {
    idle: 200,
    probing: 150,
    starting: 350,
    running_wasm: 1_225,
    running_smoke: 0,
    stopped: 75,
    error: 0,
  });
  assert.equal(
    Object.values(durationsAfterStop).reduce((a, b) => a + b, 0),
    2_000
  );

  // Verify exact transition history records.
  const history = controller.getTransitionHistory();
  assert.deepEqual(history, [
    {
      seq: 1,
      from: 'idle',
      to: 'probing',
      atMs: 1_200,
      durationInPrevMs: 200,
      reason: 'Probing browser capabilities and server readiness',
    },
    {
      seq: 2,
      from: 'probing',
      to: 'starting',
      atMs: 1_350,
      durationInPrevMs: 150,
      reason: 'Starting Wasm64 runtime',
    },
    {
      seq: 3,
      from: 'starting',
      to: 'running_wasm',
      atMs: 1_700,
      durationInPrevMs: 350,
      reason: 'Wasm64 runtime running',
    },
    {
      seq: 4,
      from: 'running_wasm',
      to: 'stopped',
      atMs: 2_925,
      durationInPrevMs: 1_225,
      reason: 'Session stopped',
    },
  ]);
  assert.equal(controller.getLastTransitionReason(), 'Session stopped');

  // Verify status & telemetry observers received non-mutating history & stateDurations.
  assert.ok(statusEvents.length >= 2);
  const lastStatus = statusEvents[statusEvents.length - 1];
  assert.equal(lastStatus.transitionCount, 4);
  assert.equal(lastStatus.lastTransitionReason, 'Session stopped');
  assert.equal(lastStatus.sessionUptimeMs, 0);
  assert.equal(lastStatus.transitionHistory.length, 4);
  assert.equal(lastStatus.stateDurations.running_wasm, 1_225);

  const bootTelemetry = telemetryEvents.filter((e) => e.type === 'boot-status');
  assert.ok(bootTelemetry.length >= 2);
  assert.equal(bootTelemetry[bootTelemetry.length - 1].transitionHistory.length, 4);
});

test('W148 bounded ring buffer eviction, default 64 cap, and defensive clone immutability', async () => {
  // 1. Small configurable capacity (4) evicts oldest transitions in exact FIFO order.
  const clock = createClock(500);
  const controller = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus({ hasWasmBinary: true })),
    now: clock.now,
    maxTransitionHistory: 4,
    capabilityProbe: () => {
      clock.advance(10);
      return fullCapabilities();
    },
    startWasmRuntime: async () => {
      clock.advance(20);
      return { stop() {} };
    },
    startSmokeDemo: async () => {
      clock.advance(30);
      return { stop() {} };
    },
  });

  // Cycle 1: idle->probing(1)->starting(2)->running_wasm(3)->stopped(4)
  await controller.start();
  clock.advance(100);
  await controller.stop();

  // Cycle 2: stopped->probing(5)->starting(6)->running_smoke(7)->stopped(8)
  clock.advance(50);
  await controller.start({ preferSmoke: true });
  clock.advance(200);
  await controller.stop();

  // Cycle 3: stopped->probing(9)->starting(10)->running_wasm(11)
  clock.advance(40);
  await controller.start();

  const history = controller.getTransitionHistory();
  assert.equal(history.length, 4);
  assert.deepEqual(
    history.map((e) => ({ seq: e.seq, from: e.from, to: e.to })),
    [
      { seq: 8, from: 'running_smoke', to: 'stopped' },
      { seq: 9, from: 'stopped', to: 'probing' },
      { seq: 10, from: 'probing', to: 'starting' },
      { seq: 11, from: 'starting', to: 'running_wasm' },
    ]
  );

  const snap = controller.getSnapshot();
  assert.equal(snap.transitionCount, 11);
  assert.equal(snap.transitionHistorySize, 4);
  assert.equal(snap.transitionEvictedCount, 7);
  assert.equal(snap.maxTransitionHistory, 4);

  // Mutating returned arrays and objects never affects internal ring buffer or duration state.
  history[0].to = 'mutated';
  history.push({ seq: 999, from: 'x', to: 'y', atMs: 0, durationInPrevMs: 0, reason: 'bad' });
  snap.transitionHistory.length = 0;
  snap.stateDurations.running_wasm = 999_999;

  const freshHistory = controller.getTransitionHistory();
  assert.equal(freshHistory.length, 4);
  assert.equal(freshHistory[0].to, 'stopped');
  assert.equal(freshHistory[0].seq, 8);

  const freshDurations = controller.getStateDurations();
  freshDurations.idle = 123_456;
  assert.equal(controller.getStateDurations().idle, 0);
  assert.equal(controller.getStateDurations().running_wasm, 100);
  assert.equal(controller.getStateDurations().running_smoke, 200);

  await controller.stop();

  // 2. Default capacity (64): 18 start/stop cycles = 72 transitions -> retains newest 64 (seq 9..72).
  const defaultClock = createClock(0);
  const defaultCtrl = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus({ hasWasmBinary: true })),
    now: defaultClock.now,
    capabilityProbe: () => fullCapabilities(),
    startWasmRuntime: async () => ({ stop() {} }),
  });

  for (let i = 0; i < 18; i += 1) {
    defaultClock.advance(5);
    await defaultCtrl.start();
    defaultClock.advance(10);
    await defaultCtrl.stop();
  }

  const defaultHist = defaultCtrl.getTransitionHistory();
  assert.equal(defaultHist.length, 64);
  assert.equal(defaultHist[0].seq, 9);
  assert.equal(defaultHist[63].seq, 72);
  assert.equal(defaultCtrl.getSnapshot().transitionCount, 72);
  assert.equal(defaultCtrl.getSnapshot().transitionEvictedCount, 8);
});

test('W148 reason sanitization, error recovery, restart(), clearTransitionHistory(), and resetTelemetry()', async () => {
  const clock = createClock(5_000);
  let failNext = true;
  const hexBlob = 'de'.repeat(32);

  const controller = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus({ hasWasmBinary: true })),
    nowFn: clock.now,
    capabilityProbe: () => {
      clock.advance(20);
      return fullCapabilities();
    },
    startWasmRuntime: async () => {
      clock.advance(30);
      if (failNext) {
        failNext = false;
        throw new Error(
          `Boot failed at C:\\Users\\alice\\gta6\\main.wasm and /Users/alice/dump.bin data=${hexBlob}\r\n`
        );
      }
      return { stop() {} };
    },
    startSmokeDemo: async () => {
      clock.advance(40);
      return { stop() {} };
    },
  });

  // 1. Failing start sanitizes error reason in transition history & lastTransitionReason.
  const failRes = await controller.start();
  assert.equal(failRes.ok, false);
  assert.equal(controller.getState(), SESSION_STATE.ERROR);
  assert.equal(controller.getSessionUptimeMs(), 0);
  assert.equal(controller.getRestartCount(), 0);

  const errReason = controller.getLastTransitionReason();
  assert.ok(errReason);
  assert.doesNotMatch(errReason, /alice|C:\\|\/Users|dededede|[\r\n]/);
  assert.match(errReason, /<path>/);
  assert.match(errReason, /<buffer>/);

  // 2. Recover from ERROR via start() increments recoveryCount (not restartCount).
  clock.advance(100);
  const recoverRes = await controller.start();
  assert.equal(recoverRes.ok, true);
  assert.equal(controller.getState(), SESSION_STATE.RUNNING_WASM);
  assert.equal(controller.getSnapshot().recoveryCount, 1);
  assert.equal(controller.getRestartCount(), 0);
  assert.equal(controller.getStateDurations().error, 100);

  clock.advance(500);
  assert.equal(controller.getSessionUptimeMs(), 500);

  // 3. restart() into smoke demo increments restartCount, resets active session uptime,
  //    preserves transition history and cumulative state durations across modes.
  const r1 = await controller.restart({ preferSmoke: true });
  assert.equal(r1.ok, true);
  assert.equal(r1.state, SESSION_STATE.RUNNING_SMOKE);
  assert.equal(controller.getRestartCount(), 1);
  assert.equal(controller.getSessionUptimeMs(), 0);

  clock.advance(300);
  assert.equal(controller.getSessionUptimeMs(), 300);

  const durations = controller.getStateDurations();
  assert.equal(durations.running_wasm, 500);
  assert.equal(durations.running_smoke, 300);
  assert.equal(durations.error, 100);

  // 4. Second restart back to Wasm64.
  const r2 = await controller.restart({ preferSmoke: false });
  assert.equal(r2.ok, true);
  assert.equal(r2.state, SESSION_STATE.RUNNING_WASM);
  assert.equal(controller.getRestartCount(), 2);
  assert.equal(controller.getSessionUptimeMs(), 0);

  clock.advance(250);
  assert.equal(controller.getSessionUptimeMs(), 250);
  assert.equal(controller.getStateDurations().running_wasm, 750);
  assert.equal(controller.getStateDurations().running_smoke, 300);

  // 5. clearTransitionHistory empties buffer while preserving seq, durations, restartCount, and uptime.
  const seqBeforeClear = controller.getSnapshot().transitionCount;
  controller.clearTransitionHistory();
  assert.deepEqual(controller.getTransitionHistory(), []);
  assert.equal(controller.getSnapshot().transitionHistorySize, 0);
  assert.equal(controller.getSnapshot().transitionCount, seqBeforeClear);
  assert.equal(controller.getRestartCount(), 2);
  assert.equal(controller.getSessionUptimeMs(), 250);

  // Next transition continues monotonic seq numbering.
  await controller.stop();
  const afterStopHist = controller.getTransitionHistory();
  assert.equal(afterStopHist.length, 1);
  assert.equal(afterStopHist[0].seq, seqBeforeClear + 1);
  assert.equal(afterStopHist[0].from, 'running_wasm');
  assert.equal(afterStopHist[0].to, 'stopped');

  // 6. resetTelemetry resets counters, history, and state durations while keeping active session intact.
  await controller.start();
  clock.advance(400);
  assert.equal(controller.getSessionUptimeMs(), 400);

  controller.resetTelemetry();
  assert.deepEqual(controller.getTransitionHistory(), []);
  assert.equal(controller.getRestartCount(), 0);
  assert.equal(controller.getLastTransitionReason(), null);
  const resetSnap = controller.getSnapshot();
  assert.equal(resetSnap.transitionCount, 0);
  assert.equal(resetSnap.transitionEvictedCount, 0);
  assert.equal(resetSnap.recoveryCount, 0);
  assert.equal(resetSnap.stateDurations.running_wasm, 0);
  assert.equal(resetSnap.stateDurations.running_smoke, 0);
  // Active running session uptime is unaffected by telemetry reset.
  assert.equal(controller.getState(), SESSION_STATE.RUNNING_WASM);
  assert.equal(controller.getSessionUptimeMs(), 400);

  clock.advance(60);
  assert.equal(controller.getStateDurations().running_wasm, 60);
  assert.equal(controller.getSessionUptimeMs(), 460);

  await controller.stop();
  const postResetHist = controller.getTransitionHistory();
  assert.equal(postResetHist.length, 1);
  assert.equal(postResetHist[0].seq, 1);
  assert.equal(postResetHist[0].durationInPrevMs, 60);
});

// --- W276: Leak-Free Restart Teardown, Re-entrancy Guard & Device-Loss Recovery Tracking ---

test('W276 option validation and initial getStats() stability counters', () => {
  assert.throws(() => new RuntimeSessionController({ bootOrchestrator: 'invalid' }), TypeError);
  assert.throws(() => new RuntimeSessionController({ orchestrator: 123 }), TypeError);
  assert.throws(() => new RuntimeSessionController({ onDeviceLost: 'nope' }), TypeError);
  assert.throws(() => new RuntimeSessionController({ onDeviceRecovered: {} }), TypeError);

  const clock = createClock(4000);
  const controller = new RuntimeSessionController({
    fetchImpl: null,
    now: clock.now,
  });

  assert.throws(() => controller.onDeviceLost('nope'), TypeError);
  assert.throws(() => controller.onDeviceRecovered(42), TypeError);
  assert.equal(controller.isDestroyed(), false);

  const stats = controller.getStats();
  assert.equal(stats.state, SESSION_STATE.IDLE);
  assert.equal(stats.mode, SESSION_MODE.NONE);
  assert.equal(stats.bootMode, BOOT_MODE.IDLE);
  assert.equal(stats.destroyed, false);
  assert.equal(stats.restartInFlight, false);
  assert.equal(stats.uptimeMs, 0);
  assert.equal(stats.restartCount, 0);
  assert.equal(stats.coalescedRestarts, 0);
  assert.equal(stats.recoveryCount, 0);
  assert.equal(stats.deviceLost, false);
  assert.equal(stats.deviceLossCount, 0);
  assert.equal(stats.deviceRecoveryCount, 0);
  assert.equal(stats.lastDeviceLossReason, null);
  assert.equal(stats.cleanShutdownCount, 0);
  assert.equal(stats.preloadBufferReleaseCount, 0);
  assert.equal(stats.attachedWorkerCount, 0);
  assert.equal(stats.activeListenerCount, 0);

  const snap = controller.getSnapshot();
  assert.equal(snap.coalescedRestarts, 0);
  assert.equal(snap.deviceLossCount, 0);
  assert.equal(snap.deviceRecoveryCount, 0);
  assert.equal(snap.cleanShutdownCount, 0);
});

test('W276 concurrent restart() calls coalesce safely without spawning duplicate workers or handles', async () => {
  const clock = createClock(10_000);
  const timers = createFakeTimers();
  const hud = createMockHud();
  const worker = createMockWorker();
  const gestureTarget = createMockEventTarget();
  const perfMonitor = new PerfMonitor({
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
  });

  let releaseStopGate;
  let releaseLaunchGate;
  let stopGateArmed = false;
  let launchGateArmed = false;
  let wasmLaunchCount = 0;
  let handleStopCount = 0;
  let preloadReleaseCount = 0;
  const activeHandles = new Set();

  const bootOrchestrator = {
    releasePreloadBuffers() {
      preloadReleaseCount += 1;
      return { releasedFields: ['wasmBytes'], releasedBytes: 1024 };
    },
  };

  const controller = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus({ hasWasmBinary: true })),
    hud,
    perfMonitor,
    wgpuWorker: worker,
    bootOrchestrator,
    audioGestureTarget: gestureTarget,
    audioEngine: {
      resumeOnUserGesture(target) {
        const fn = () => {};
        target.addEventListener('pointerdown', fn);
        return () => target.removeEventListener('pointerdown', fn);
      },
    },
    capabilityProbe: () => fullCapabilities(),
    now: clock.now,
    startWasmRuntime: async () => {
      const id = `wasm-${++wasmLaunchCount}`;
      if (launchGateArmed) {
        launchGateArmed = false;
        await new Promise((resolve) => {
          releaseLaunchGate = resolve;
        });
      }
      activeHandles.add(id);
      return {
        id,
        async stop() {
          handleStopCount += 1;
          if (stopGateArmed) {
            stopGateArmed = false;
            await new Promise((resolve) => {
              releaseStopGate = resolve;
            });
          }
          activeHandles.delete(id);
        },
      };
    },
  });

  await controller.start();
  assert.equal(wasmLaunchCount, 1);
  assert.equal(activeHandles.size, 1);

  // Trigger 4 overlapping concurrent restarts plus a concurrent start() while stop & launch are in flight.
  stopGateArmed = true;
  launchGateArmed = true;

  const r1 = controller.restart();
  const r2 = controller.restart();
  const r3 = controller.restart();
  const r4 = controller.restart();
  const sDuringRestart = controller.start();

  assert.equal(controller.getStats().restartInFlight, true);
  assert.equal(controller.getStats().restartCount, 1);
  assert.equal(controller.getStats().coalescedRestarts, 3);

  // Let stop finish, then let the single fresh launch finish.
  await new Promise((r) => setTimeout(r, 5));
  releaseStopGate();
  await new Promise((r) => setTimeout(r, 5));
  releaseLaunchGate();

  const results = await Promise.all([r1, r2, r3, r4, sDuringRestart]);
  for (const res of results) {
    assert.equal(res.ok, true);
    assert.equal(res.state, SESSION_STATE.RUNNING_WASM);
  }

  // Only ONE teardown and ONE new launch happened (total 2 launches including initial start).
  assert.equal(wasmLaunchCount, 2);
  assert.equal(handleStopCount, 1);
  assert.equal(preloadReleaseCount, 1);
  assert.equal(activeHandles.size, 1);
  assert.ok(activeHandles.has('wasm-2'));
  assert.equal(timers.activeCount(), 1);
  assert.equal(worker.listenerCount('message'), 1);
  assert.equal(gestureTarget.listenerCount('pointerdown'), 1);

  const statsAfter = controller.getStats();
  assert.equal(statsAfter.restartInFlight, false);
  assert.equal(statsAfter.restartCount, 1);
  assert.equal(statsAfter.coalescedRestarts, 3);
  assert.equal(statsAfter.cleanShutdownCount, 1);
  assert.equal(statsAfter.preloadBufferReleaseCount, 1);

  // A subsequent non-overlapping restart works cleanly and increments restartCount again.
  const r5 = await controller.restart();
  assert.equal(r5.ok, true);
  assert.equal(wasmLaunchCount, 3);
  assert.equal(activeHandles.size, 1);
  assert.ok(activeHandles.has('wasm-3'));
  assert.equal(controller.getStats().restartCount, 2);
  assert.equal(controller.getStats().coalescedRestarts, 3);
  assert.equal(controller.getStats().cleanShutdownCount, 2);
  assert.equal(preloadReleaseCount, 2);

  // External stop() during an in-flight restart aborts launch cleanly.
  stopGateArmed = true;
  const abortedRestart = controller.restart();
  await new Promise((r) => setTimeout(r, 5));
  const externalStop = controller.stop();
  releaseStopGate();

  const [abortedRes, extStopRes] = await Promise.all([abortedRestart, externalStop]);
  assert.equal(abortedRes.ok, false);
  assert.equal(abortedRes.aborted, true);
  assert.equal(extStopRes.ok, true);
  assert.equal(controller.getState(), SESSION_STATE.STOPPED);
  assert.equal(wasmLaunchCount, 3);
  assert.equal(activeHandles.size, 0);
  assert.equal(timers.activeCount(), 0);
  assert.equal(worker.listenerCount('message'), 0);
  assert.equal(gestureTarget.listenerCount('pointerdown'), 0);
});

test('W276 stop(), restart(), destroy(), and dispose() invoke releasePreloadBuffers/releaseBootBundleBytes and clear all refs', async () => {
  const clock = createClock(5_000);
  const timers = createFakeTimers();
  const hud = createMockHud();
  const worker = createMockWorker();
  const gestureTarget = createMockEventTarget();
  const perfMonitor = new PerfMonitor({
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
  });

  let orchReleaseCalls = 0;
  let handleReleaseCalls = 0;
  let handleStopCalls = 0;
  let inputStopCalls = 0;

  const orchestrator = {
    releasePreloadBuffers() {
      orchReleaseCalls += 1;
      return { releasedFields: ['wasmBytes', 'sideModules'], releasedBytes: 2048 };
    },
  };

  const controller = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus({ hasWasmBinary: true })),
    hud,
    perfMonitor,
    wgpuWorker: worker,
    orchestrator,
    audioGestureTarget: gestureTarget,
    inputBridge: {
      running: false,
      start() {
        this.running = true;
      },
      stop() {
        inputStopCalls += 1;
        this.running = false;
      },
    },
    audioEngine: {
      resumeOnUserGesture(target) {
        const fn = () => {};
        target.addEventListener('keydown', fn);
        return () => target.removeEventListener('keydown', fn);
      },
    },
    capabilityProbe: () => fullCapabilities(),
    now: clock.now,
    startWasmRuntime: async (ctx) => {
      assert.equal(ctx.bootOrchestrator, orchestrator);
      return {
        stop() {
          handleStopCalls += 1;
        },
        releaseBootBundleBytes() {
          handleReleaseCalls += 1;
        },
      };
    },
  });

  const observedStates = [];
  controller.onStateChange((ev) => observedStates.push(ev.state));
  controller.onStatusUpdate(() => {});
  controller.onTelemetry(() => {});
  controller.onValidationError(() => {});
  controller.onSplashFrame(() => {});
  controller.onDeviceLost(() => {});
  controller.onDeviceRecovered(() => {});

  assert.equal(controller.getStats().activeListenerCount, 7);

  await controller.start();
  assert.equal(timers.activeCount(), 1);
  assert.equal(worker.listenerCount('message'), 1);
  assert.equal(gestureTarget.listenerCount('keydown'), 1);

  // dispose() tears down everything, releases preload buffers on both orchestrator & handle,
  // notifies state listener of final 'stopped' transition, then clears all listener sets.
  const d1 = await controller.dispose();
  assert.equal(d1.ok, true);
  assert.equal(d1.destroyed, true);
  assert.equal(d1.alreadyDestroyed, false);
  assert.equal(controller.isDestroyed(), true);
  assert.equal(controller.getState(), SESSION_STATE.STOPPED);
  assert.equal(observedStates[observedStates.length - 1], SESSION_STATE.STOPPED);

  assert.equal(handleStopCalls, 1);
  assert.equal(orchReleaseCalls, 1);
  assert.equal(handleReleaseCalls, 1);
  assert.equal(inputStopCalls, 1);
  assert.equal(timers.activeCount(), 0);
  assert.equal(worker.listenerCount('message'), 0);
  assert.equal(gestureTarget.listenerCount('keydown'), 0);

  const statsAfterDispose = controller.getStats();
  assert.equal(statsAfterDispose.destroyed, true);
  assert.equal(statsAfterDispose.cleanShutdownCount, 1);
  assert.equal(statsAfterDispose.preloadBufferReleaseCount, 1);
  assert.equal(statsAfterDispose.attachedWorkerCount, 0);
  assert.equal(statsAfterDispose.activeListenerCount, 0);

  // Idempotent destroy()/dispose() does not double-stop or double-count clean shutdowns.
  const d2 = await controller.destroy();
  assert.equal(d2.ok, true);
  assert.equal(d2.destroyed, true);
  assert.equal(d2.alreadyDestroyed, true);
  assert.equal(handleStopCalls, 1);
  assert.equal(controller.getStats().cleanShutdownCount, 1);

  // start() and restart() after dispose() are rejected cleanly without spawning handles.
  const startAfter = await controller.start();
  assert.equal(startAfter.ok, false);
  assert.equal(startAfter.destroyed, true);
  const restartAfter = await controller.restart();
  assert.equal(restartAfter.ok, false);
  assert.equal(restartAfter.destroyed, true);

  // Fallback to releaseBootBundleBytes() on orchestrator when releasePreloadBuffers() is absent.
  let bundleReleaseCalls = 0;
  const c2 = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus({ hasWasmBinary: false })),
    bootOrchestrator: {
      releaseBootBundleBytes() {
        bundleReleaseCalls += 1;
      },
    },
    capabilityProbe: () => fullCapabilities(),
    startSmokeDemo: async () => ({ stop() {} }),
  });
  await c2.start();
  await c2.stop();
  assert.equal(bundleReleaseCalls, 1);
  assert.equal(c2.getStats().cleanShutdownCount, 1);
  assert.equal(c2.getStats().preloadBufferReleaseCount, 1);
  // Idempotent second stop does not double-count cleanShutdownCount.
  await c2.stop();
  assert.equal(c2.getStats().cleanShutdownCount, 1);
});

test('W276 recordDeviceLost / recordDeviceRecovered track sanitized counters, worker messages, and observers', async () => {
  const clock = createClock(30_000);
  const hud = createExtendedMockHud();
  const hudLostEvents = [];
  const hudRecoveredEvents = [];
  hud.onDeviceLost = (ev) => hudLostEvents.push(ev);
  hud.onDeviceRecovered = (ev) => hudRecoveredEvents.push(ev);

  const reported = [];
  const crashReporter = {
    captureError(err, ctx) {
      reported.push({ message: err.message, ctx });
    },
  };

  const worker = createMockWorker();
  const lostEvents = [];
  const recoveredEvents = [];
  const telemetryEvents = [];
  let capturedLaunchCtx = null;

  const controller = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus({ hasWasmBinary: true })),
    hud,
    crashReporter,
    wgpuWorker: worker,
    now: clock.now,
    capabilityProbe: () => fullCapabilities(),
    onDeviceLost: (ev) => lostEvents.push(ev),
    onDeviceRecovered: (ev) => recoveredEvents.push(ev),
    onTelemetry: (ev) => telemetryEvents.push(ev),
    startWasmRuntime: async (ctx) => {
      capturedLaunchCtx = ctx;
      return { stop() {} };
    },
  });

  // Throwing observer must not break delivery to other observers.
  const unsubBadLost = controller.onDeviceLost(() => {
    throw new Error('deviceLost observer boom');
  });
  const unsubBadRec = controller.onDeviceRecovered(() => {
    throw new Error('deviceRecovered observer boom');
  });

  await controller.start();

  // 1. Direct recordDeviceLost(reason) sanitizes host paths and hex dumps.
  const hexDump = 'ff'.repeat(32);
  const l1 = controller.recordDeviceLost(
    `GPU driver reset at C:\\Users\\alice\\gpu.dll and /Users/alice/trace.bin dump=${hexDump}\n`
  );
  assert.equal(l1.handled, true);
  assert.equal(l1.type, 'gpu-device-lost');
  assert.equal(l1.deviceLost, true);
  assert.equal(l1.deviceLossCount, 1);
  assert.equal(l1.deviceRecoveryCount, 0);
  assert.doesNotMatch(l1.reason, /alice|C:\\|\/Users|ffffffff|[\r\n]/);
  assert.match(l1.reason, /<path>/);
  assert.match(l1.reason, /<buffer>/);

  // 2. Direct recordDeviceRecovered() clears deviceLost flag and tracks recovery duration.
  clock.advance(450);
  const rec1 = controller.recordDeviceRecovered();
  assert.equal(rec1.handled, true);
  assert.equal(rec1.type, 'gpu-device-recovered');
  assert.equal(rec1.deviceLost, false);
  assert.equal(rec1.deviceLossCount, 1);
  assert.equal(rec1.deviceRecoveryCount, 1);
  assert.equal(rec1.recoveryDurationMs, 450);

  // 3. Worker messages (`gpu-device-lost` and `gpu-device-recovered`) flow automatically.
  clock.advance(100);
  worker.emitMessage({
    type: 'gpu-device-lost',
    reason: 'destroyed',
    message: 'Adapter lost due to TDR',
    source: 'wgpu_worker',
  });
  clock.advance(250);
  worker.emitMessage({
    type: 'gpu-device-recovered',
    source: 'wgpu_worker',
  });

  // 4. Launch context callbacks also forward while session is active.
  clock.advance(50);
  capturedLaunchCtx.onDeviceLost(new Error('Device lost via launcher callback'));
  clock.advance(120);
  capturedLaunchCtx.onDeviceRecovered({ reason: 're-initialized swapchain' });

  const stats = controller.getStats();
  assert.equal(stats.deviceLost, false);
  assert.equal(stats.deviceLossCount, 3);
  assert.equal(stats.deviceRecoveryCount, 3);
  assert.equal(stats.lastDeviceLossReason, 'Device lost via launcher callback');
  assert.equal(stats.lastDeviceRecoveredAtMs, clock.now());

  const diag = controller.getDiagnostics();
  assert.equal(diag.deviceLossCount, 3);
  assert.equal(diag.deviceRecoveryCount, 3);
  assert.equal(diag.deviceLost, false);

  assert.equal(lostEvents.length, 3);
  assert.equal(recoveredEvents.length, 3);
  assert.equal(hudLostEvents.length, 3);
  assert.equal(hudRecoveredEvents.length, 3);
  assert.equal(telemetryEvents.filter((e) => e.type === 'gpu-device-lost').length, 3);
  assert.equal(telemetryEvents.filter((e) => e.type === 'gpu-device-recovered').length, 3);
  assert.equal(reported.filter((r) => r.ctx.category === 'webgpu_device_lost').length, 3);

  // 5. After stop(), stale launchContext callbacks are ignored.
  await controller.stop();
  const staleRes = capturedLaunchCtx.onDeviceLost('stale loss');
  assert.equal(staleRes.handled, false);
  assert.equal(controller.getStats().deviceLossCount, 3);

  // 6. resetTelemetry() clears device-loss, recovery, coalescedRestarts, and cleanShutdownCount.
  controller.resetTelemetry();
  const resetStats = controller.getStats();
  assert.equal(resetStats.deviceLossCount, 0);
  assert.equal(resetStats.deviceRecoveryCount, 0);
  assert.equal(resetStats.deviceLost, false);
  assert.equal(resetStats.lastDeviceLossReason, null);
  assert.equal(resetStats.cleanShutdownCount, 0);
  assert.equal(resetStats.coalescedRestarts, 0);
  assert.equal(resetStats.preloadBufferReleaseCount, 0);

  unsubBadLost();
  unsubBadRec();
});

test('W276 cleanShutdownCount does not increment when handle teardown throws', async () => {
  const reported = [];
  const controller = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus({ hasWasmBinary: true })),
    crashReporter: {
      captureError(err, ctx) {
        reported.push({ message: err.message, ctx });
      },
    },
    capabilityProbe: () => fullCapabilities(),
    startWasmRuntime: async () => ({
      async stop() {
        throw new Error('Worker terminate fault');
      },
    }),
  });

  await controller.start();
  const stopRes = await controller.stop();
  assert.equal(stopRes.ok, true);
  assert.equal(stopRes.state, SESSION_STATE.STOPPED);
  assert.equal(controller.getStats().cleanShutdownCount, 0);
  assert.ok(reported.some((r) => r.ctx.source === 'runtimeHandle'));
});

// --- W321: Session Phase Durations & Soft Memory-Pressure Reclaim Hooks ---

test('W321 exports, option validation, and initial phase/reclaim telemetry in getSnapshot() and getStats()', () => {
  assert.deepEqual(SESSION_PHASE, {
    BOOT: 'boot',
    START_SCREEN: 'start_screen',
    INTRO_VIDEO: 'intro_video',
    MAIN_MENU: 'main_menu',
    IN_GAME: 'in_game',
  });

  assert.throws(() => new RuntimeSessionController({ onMemoryReclaim: 'bad' }), TypeError);
  assert.throws(() => new RuntimeSessionController({ onMemoryReclaim: 42 }), TypeError);

  const clock = createClock(1000);
  const controller = new RuntimeSessionController({
    fetchImpl: null,
    now: clock.now,
  });

  assert.throws(() => controller.registerReclaimHook('nope'), TypeError);
  assert.throws(() => controller.onMemoryReclaim({}), TypeError);

  const expectedEmptyPhases = {
    boot: 0,
    start_screen: 0,
    intro_video: 0,
    main_menu: 0,
    in_game: 0,
  };

  assert.equal(controller.getCurrentPhase(), null);
  assert.deepEqual(controller.getPhaseDurations(), expectedEmptyPhases);

  const stats = controller.getStats();
  assert.equal(stats.currentPhase, null);
  assert.deepEqual(stats.phaseDurationsMs, expectedEmptyPhases);
  assert.equal(stats.phaseTransitionCount, 0);
  assert.equal(stats.reclaimRuns, 0);
  assert.equal(stats.reclaimFailures, 0);
  assert.equal(stats.reclaimInFlight, false);
  assert.equal(stats.reclaimHookCount, 0);
  assert.equal(stats.lastReclaimReason, null);
  assert.equal(stats.lastReclaimAtMs, null);

  const snap = controller.getSnapshot();
  assert.equal(snap.currentPhase, null);
  assert.deepEqual(snap.phaseDurationsMs, expectedEmptyPhases);
  assert.equal(snap.phaseTransitionCount, 0);
  assert.equal(snap.reclaimRuns, 0);
  assert.equal(snap.reclaimFailures, 0);
  assert.equal(snap.reclaimInFlight, false);
  assert.equal(snap.reclaimHookCount, 0);
  assert.equal(snap.lastReclaimReason, null);
  assert.equal(snap.lastReclaimAtMs, null);
});

test('W321 recordPhase tracks sanitized phase durations across boot -> start_screen -> intro_video -> main_menu -> in_game', async () => {
  const clock = createClock(10_000);
  const worker = createMockWorker();
  const statusUpdates = [];
  let capturedLaunchCtx = null;

  const controller = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus({ hasWasmBinary: true })),
    wgpuWorker: worker,
    now: clock.now,
    capabilityProbe: () => fullCapabilities(),
    onStatusUpdate: (ev) => statusUpdates.push(ev),
    startWasmRuntime: async (ctx) => {
      capturedLaunchCtx = ctx;
      return { stop() {} };
    },
  });

  // 1. Standalone explicit-timestamp progression + alias normalization + sanitization.
  const rBoot = controller.recordPhase('boot', 1000);
  assert.equal(rBoot.ok, true);
  assert.equal(rBoot.changed, true);
  assert.equal(rBoot.phase, SESSION_PHASE.BOOT);
  assert.equal(rBoot.previousPhase, null);

  // Duplicate same phase is a no-op and does not alter elapsed baseline.
  const rDup = controller.recordPhase('BOOT', 1200);
  assert.equal(rDup.ok, true);
  assert.equal(rDup.changed, false);
  assert.equal(rDup.durationInPrevMs, 0);

  // Unsafe / invalid phases are rejected cleanly without polluting phaseDurationsMs.
  for (const badPhase of [null, undefined, '', 123, '../../etc/passwd', 'C:\\Users\\alice', '__proto__', 'unknown']) {
    const badRes = controller.recordPhase(badPhase, 1300);
    assert.equal(badRes.ok, false);
    assert.equal(badRes.changed, false);
    assert.equal(badRes.phase, null);
    assert.equal(controller.getCurrentPhase(), SESSION_PHASE.BOOT);
  }

  const rStart = controller.recordPhase('Start Screen', 1500);
  assert.equal(rStart.ok, true);
  assert.equal(rStart.changed, true);
  assert.equal(rStart.phase, SESSION_PHASE.START_SCREEN);
  assert.equal(rStart.durationInPrevMs, 500);

  const rIntro = controller.recordPhase('intro-video', 2200);
  assert.equal(rIntro.phase, SESSION_PHASE.INTRO_VIDEO);
  assert.equal(rIntro.durationInPrevMs, 700);

  const rMenu = controller.recordPhase({ phase: 'main menu', timestampMs: 3000 });
  assert.equal(rMenu.phase, SESSION_PHASE.MAIN_MENU);
  assert.equal(rMenu.durationInPrevMs, 800);

  // Out-of-order earlier timestamp clamps duration to 0 (never negative).
  const rInGame = controller.recordPhase('in-game', 2900);
  assert.equal(rInGame.ok, true);
  assert.equal(rInGame.phase, SESSION_PHASE.IN_GAME);
  assert.equal(rInGame.durationInPrevMs, 0);

  const explicitSnap = controller.getSnapshot(3400);
  assert.equal(explicitSnap.currentPhase, SESSION_PHASE.IN_GAME);
  assert.deepEqual(explicitSnap.phaseDurationsMs, {
    boot: 500,
    start_screen: 700,
    intro_video: 800,
    main_menu: 0,
    in_game: 500,
  });

  // Defensive clone immutability.
  explicitSnap.phaseDurationsMs.boot = 999_999;
  const freshStats = controller.getStats(3400);
  assert.equal(freshStats.phaseDurationsMs.boot, 500);
  assert.equal(freshStats.currentPhase, SESSION_PHASE.IN_GAME);

  // 2. Clock-driven session lifecycle: start() resets phases and enters 'boot'.
  await controller.start();
  assert.equal(controller.getCurrentPhase(), SESSION_PHASE.BOOT);
  assert.deepEqual(controller.getPhaseDurations(), {
    boot: 0,
    start_screen: 0,
    intro_video: 0,
    main_menu: 0,
    in_game: 0,
  });

  clock.advance(400);
  capturedLaunchCtx.recordPhase(SESSION_PHASE.START_SCREEN);
  clock.advance(600);
  worker.emitMessage({ type: 'session-phase', phase: 'intro_video' });
  clock.advance(350);
  worker.emitMessage({ type: 'phase-transition', phase: 'main_menu' });
  clock.advance(900);
  controller.recordPhase(SESSION_PHASE.IN_GAME);
  clock.advance(1250);

  const runningStats = controller.getStats();
  assert.equal(runningStats.currentPhase, SESSION_PHASE.IN_GAME);
  assert.deepEqual(runningStats.phaseDurationsMs, {
    boot: 400,
    start_screen: 600,
    intro_video: 350,
    main_menu: 900,
    in_game: 1250,
  });
  assert.ok(statusUpdates.some((u) => u.reason === 'phase:in_game' && u.currentPhase === 'in_game'));

  // 3. stop() finalizes active phase duration and freezes accumulation while stopped.
  await controller.stop();
  assert.equal(controller.getCurrentPhase(), null);
  clock.advance(5000);
  assert.deepEqual(controller.getSnapshot().phaseDurationsMs, {
    boot: 400,
    start_screen: 600,
    intro_video: 350,
    main_menu: 900,
    in_game: 1250,
  });

  // Stale launchContext.recordPhase after stop() is ignored.
  const stalePhase = capturedLaunchCtx.recordPhase(SESSION_PHASE.MAIN_MENU);
  assert.equal(stalePhase.ok, false);
  assert.equal(controller.getCurrentPhase(), null);
});

test('W321 requestMemoryReclaim executes sync & async hooks, isolates errors, coalesces concurrent calls, and guards re-entrancy', async () => {
  const clock = createClock(20_000);
  const hud = createMockHud();
  const reported = [];
  const telemetryEvents = [];
  const invoked = [];

  const controller = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus({ hasWasmBinary: true })),
    hud,
    crashReporter: {
      captureError(err, ctx) {
        reported.push({ message: err.message, ctx });
      },
    },
    now: clock.now,
    capabilityProbe: () => fullCapabilities(),
    onTelemetry: (ev) => telemetryEvents.push(ev),
    onMemoryReclaim: (ctx) => {
      invoked.push({ hook: 'ctor-rt-pool', reason: ctx.reason, phase: ctx.currentPhase });
    },
    startWasmRuntime: async () => ({ stop() {} }),
  });

  await controller.start();
  clock.advance(100);
  controller.recordPhase(SESSION_PHASE.IN_GAME);

  const hexBlob = 'ab'.repeat(32);
  let releaseAsyncHook;
  let asyncHookGateArmed = false;

  const unsubStaging = controller.registerReclaimHook((ctx) => {
    invoked.push({ hook: 'staging-trim', reason: ctx.reason });
    // Verify synchronous re-entrancy is safely rejected without deadlock.
    const nested = controller.requestMemoryReclaim('nested-reentrant');
    nested.then((res) => {
      invoked.push({ hook: 'nested-result', ok: res.ok, skipped: res.skipped, reentrant: res.reentrant });
    });
  });

  const unsubBadSync = controller.onMemoryReclaim(() => {
    invoked.push({ hook: 'bad-sync' });
    throw new Error(`RT pool fault at C:\\Users\\alice\\pool.js dump=${hexBlob}\n`);
  });

  const unsubBadAsync = controller.registerReclaimHook(async () => {
    invoked.push({ hook: 'bad-async' });
    throw new Error('OPFS eviction rejected at /Users/alice/cache.bin');
  });

  const unsubOpfs = controller.registerReclaimHook(async (ctx) => {
    invoked.push({ hook: 'opfs-evict', reason: ctx.reason });
    if (asyncHookGateArmed) {
      asyncHookGateArmed = false;
      await new Promise((resolve) => {
        releaseAsyncHook = resolve;
      });
    }
  });

  assert.equal(controller.getStats().reclaimHookCount, 5);

  // 1. Run reclaim with mixed passing and failing hooks (sync + async).
  const res1 = await controller.requestMemoryReclaim('gpu-oom at /Users/alice/heap.bin');
  assert.equal(res1.ok, false);
  assert.equal(res1.hooksInvoked, 5);
  assert.equal(res1.hooksSucceeded, 3);
  assert.equal(res1.hooksFailed, 2);
  assert.equal(res1.reclaimRuns, 1);
  assert.equal(res1.reclaimFailures, 2);
  assert.doesNotMatch(res1.reason, /alice|\/Users/);
  assert.match(res1.reason, /<path>/);
  assert.equal(res1.errors.length, 2);
  assert.doesNotMatch(res1.errors[0], /alice|C:\\|abababab|[\r\n]/);
  assert.match(res1.errors[0], /<path>/);
  assert.match(res1.errors[0], /<buffer>/);
  assert.doesNotMatch(res1.errors[1], /alice|\/Users/);

  // Re-entrant sync call inside hook did not deadlock and reported reentrant: true.
  const nestedEntry = invoked.find((entry) => entry.hook === 'nested-result');
  assert.ok(nestedEntry);
  assert.equal(nestedEntry.ok, false);
  assert.equal(nestedEntry.skipped, true);
  assert.equal(nestedEntry.reentrant, true);

  // All 5 hooks ran in order despite 2 failures.
  assert.deepEqual(
    invoked.filter((e) => e.hook !== 'nested-result').map((e) => e.hook),
    ['ctor-rt-pool', 'staging-trim', 'bad-sync', 'bad-async', 'opfs-evict']
  );

  // CrashReporter, HUD, and telemetry received sanitized notifications.
  assert.equal(reported.filter((r) => r.ctx.source === 'memoryReclaim').length, 2);
  assert.ok(hud.logs.some((l) => l.level === 'warn' && /Memory reclaim hook warning/i.test(l.message)));
  assert.ok(telemetryEvents.some((e) => e.type === 'memory-reclaim' && e.reclaimRuns === 1 && e.reclaimFailures === 2));

  // 2. Unregister failing hooks (idempotent) -> clean run succeeds and coalesces concurrent calls.
  unsubBadSync();
  unsubBadSync();
  unsubBadAsync();
  assert.equal(controller.getStats().reclaimHookCount, 3);

  invoked.length = 0;
  asyncHookGateArmed = true;
  const p1 = controller.requestMemoryReclaim('pressure-burst');
  const p2 = controller.requestMemoryReclaim('pressure-burst-2');
  assert.equal(controller.getStats().reclaimInFlight, true);

  await new Promise((r) => setTimeout(r, 5));
  releaseAsyncHook();
  const [c1, c2] = await Promise.all([p1, p2]);

  assert.equal(c1.ok, true);
  assert.equal(c1.hooksInvoked, 3);
  assert.equal(c1.hooksSucceeded, 3);
  assert.equal(c1.hooksFailed, 0);
  assert.equal(c1.reclaimRuns, 2);
  assert.equal(c1.reclaimFailures, 2);
  assert.equal(c2.ok, true);
  assert.equal(c2.coalesced, true);
  assert.equal(c2.reclaimRuns, 2);
  assert.equal(controller.getStats().reclaimInFlight, false);

  // Only one execution of each remaining hook occurred across both coalesced calls.
  assert.deepEqual(
    invoked.filter((e) => e.hook !== 'nested-result').map((e) => e.hook),
    ['ctor-rt-pool', 'staging-trim', 'opfs-evict']
  );

  unsubStaging();
  unsubOpfs();
  assert.equal(controller.getStats().reclaimHookCount, 1);

  await controller.stop();
});

test('W321 restart invokes soft reclaim before teardown, maintains phase reset parity, and cleans up hooks on destroy()', async () => {
  const clock = createClock(50_000);
  const trace = [];
  let launchSeq = 0;
  let releasePreRestartReclaim = null;
  let gatePreRestartReclaim = false;

  const bootOrchestrator = {
    releasePreloadBuffers() {
      trace.push('preload:release');
    },
  };

  const controller = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus({ hasWasmBinary: true })),
    bootOrchestrator,
    now: clock.now,
    capabilityProbe: () => fullCapabilities(),
    onMemoryReclaim: async (ctx) => {
      trace.push(`reclaim:controller:${ctx.reason}`);
      if (gatePreRestartReclaim) {
        gatePreRestartReclaim = false;
        await new Promise((resolve) => {
          releasePreRestartReclaim = resolve;
        });
      }
    },
    startWasmRuntime: async (ctx) => {
      const id = ++launchSeq;
      trace.push(`launch:${id}`);
      // Register a session-scoped reclaim hook (e.g. per-session RT pool + staging trim).
      ctx.registerReclaimHook((reclaimCtx) => {
        trace.push(`reclaim:session-${id}:${reclaimCtx.reason}`);
      });
      return {
        stop() {
          trace.push(`handle:stop:${id}`);
        },
      };
    },
  });

  // 1. Initial start enters 'boot', progresses through all phases.
  await controller.start();
  assert.equal(controller.getStats().reclaimHookCount, 2);
  clock.advance(300);
  controller.recordPhase(SESSION_PHASE.START_SCREEN);
  clock.advance(450);
  controller.recordPhase(SESSION_PHASE.INTRO_VIDEO);
  clock.advance(250);
  controller.recordPhase(SESSION_PHASE.MAIN_MENU);
  clock.advance(800);
  controller.recordPhase(SESSION_PHASE.IN_GAME);
  clock.advance(1000);

  assert.deepEqual(controller.getSnapshot().phaseDurationsMs, {
    boot: 300,
    start_screen: 450,
    intro_video: 250,
    main_menu: 800,
    in_game: 1000,
  });

  // 2. restart() invokes soft reclaim BEFORE stopping handles/preload release,
  //    cleans up session-1 reclaim hook, and resets phase durations to fresh 'boot' parity.
  trace.length = 0;
  const r1 = await controller.restart();
  assert.equal(r1.ok, true);
  assert.deepEqual(trace, [
    'reclaim:controller:restart',
    'reclaim:session-1:restart',
    'handle:stop:1',
    'preload:release',
    'launch:2',
  ]);

  // Session-1 hook was torn down; only controller hook + session-2 hook remain (zero leaks).
  const statsAfterRestart = controller.getStats();
  assert.equal(statsAfterRestart.reclaimRuns, 1);
  assert.equal(statsAfterRestart.reclaimFailures, 0);
  assert.equal(statsAfterRestart.reclaimHookCount, 2);
  assert.equal(statsAfterRestart.currentPhase, SESSION_PHASE.BOOT);
  assert.deepEqual(statsAfterRestart.phaseDurationsMs, {
    boot: 0,
    start_screen: 0,
    intro_video: 0,
    main_menu: 0,
    in_game: 0,
  });

  // 3. restart({ skipReclaim: true }) skips soft reclaim while still resetting phase parity.
  clock.advance(200);
  controller.recordPhase(SESSION_PHASE.START_SCREEN);
  clock.advance(150);
  trace.length = 0;
  const r2 = await controller.restart({ skipReclaim: true });
  assert.equal(r2.ok, true);
  assert.deepEqual(trace, ['handle:stop:2', 'preload:release', 'launch:3']);
  assert.equal(controller.getStats().reclaimRuns, 1);
  assert.equal(controller.getStats().reclaimHookCount, 2);
  assert.equal(controller.getStats().currentPhase, SESSION_PHASE.BOOT);
  assert.deepEqual(controller.getSnapshot().phaseDurationsMs, {
    boot: 0,
    start_screen: 0,
    intro_video: 0,
    main_menu: 0,
    in_game: 0,
  });

  // 4. External stop() during in-flight pre-restart soft reclaim aborts restart cleanly.
  gatePreRestartReclaim = true;
  const abortedRestartPromise = controller.restart();
  await new Promise((r) => setTimeout(r, 5));
  const externalStopPromise = controller.stop();
  releasePreRestartReclaim();

  const [abortedRes, stopRes] = await Promise.all([abortedRestartPromise, externalStopPromise]);
  assert.equal(abortedRes.ok, false);
  assert.equal(abortedRes.aborted, true);
  assert.equal(stopRes.ok, true);
  assert.equal(controller.getState(), SESSION_STATE.STOPPED);
  assert.equal(launchSeq, 3);
  // Session-3 hook was cleaned up on stop; only controller hook remains.
  assert.equal(controller.getStats().reclaimHookCount, 1);

  // 5. resetTelemetry() clears reclaim counters and resets active phase baseline without losing currentPhase.
  await controller.start();
  clock.advance(120);
  controller.recordPhase(SESSION_PHASE.MAIN_MENU);
  clock.advance(80);
  controller.resetTelemetry();
  const afterReset = controller.getStats();
  assert.equal(afterReset.reclaimRuns, 0);
  assert.equal(afterReset.reclaimFailures, 0);
  assert.equal(afterReset.lastReclaimReason, null);
  assert.equal(afterReset.currentPhase, SESSION_PHASE.MAIN_MENU);
  assert.deepEqual(afterReset.phaseDurationsMs, {
    boot: 0,
    start_screen: 0,
    intro_video: 0,
    main_menu: 0,
    in_game: 0,
  });
  clock.advance(45);
  assert.equal(controller.getPhaseDurations().main_menu, 45);

  // 6. destroy() clears all reclaim hooks, blocks future requestMemoryReclaim(), and ignores post-destroy registrations.
  await controller.destroy();
  assert.equal(controller.getStats().reclaimHookCount, 0);
  assert.equal(controller.getCurrentPhase(), null);

  let postDestroyCalled = false;
  const noopUnsub = controller.registerReclaimHook(() => {
    postDestroyCalled = true;
  });
  noopUnsub();
  assert.equal(controller.getStats().reclaimHookCount, 0);

  const postDestroyReclaim = await controller.requestMemoryReclaim('after-destroy');
  assert.equal(postDestroyReclaim.ok, false);
  assert.equal(postDestroyReclaim.destroyed, true);
  assert.equal(postDestroyReclaim.hooksInvoked, 0);
  assert.equal(postDestroyCalled, false);
});

test('W366 getPhaseTelemetry() and getFastStats() match getStats() accurately with defensive phase clones', async () => {
  const clock = createClock(10_000);
  const controller = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus({ hasWasmBinary: true })),
    now: clock.now,
    capabilityProbe: () => fullCapabilities(),
    startWasmRuntime: async () => ({ stop() {} }),
  });

  await controller.start();
  clock.advance(250);
  controller.recordPhase(SESSION_PHASE.START_SCREEN);
  clock.advance(400);
  controller.recordPhase(SESSION_PHASE.IN_GAME);
  clock.advance(150);
  await controller.requestMemoryReclaim('vram-trim');

  const fullStats = controller.getStats();
  const phaseTel = controller.getPhaseTelemetry();
  const fastStats = controller.getFastStats();

  assert.equal(phaseTel.currentPhase, fullStats.currentPhase);
  assert.deepEqual(phaseTel.phaseDurationsMs, fullStats.phaseDurationsMs);
  assert.equal(phaseTel.phaseTransitionCount, fullStats.phaseTransitionCount);
  assert.equal(phaseTel.reclaimRuns, fullStats.reclaimRuns);
  assert.equal(phaseTel.reclaimFailures, fullStats.reclaimFailures);
  assert.equal(phaseTel.lastReclaimReason, fullStats.lastReclaimReason);
  assert.equal(phaseTel.lastReclaimAtMs, fullStats.lastReclaimAtMs);

  // Defensive clone immutability on getPhaseTelemetry().
  phaseTel.phaseDurationsMs.boot = 999_999;
  assert.equal(controller.getPhaseTelemetry().phaseDurationsMs.boot, 250);

  // getFastStats() matches core getStats() counters.
  assert.equal(fastStats.state, fullStats.state);
  assert.equal(fastStats.mode, fullStats.mode);
  assert.equal(fastStats.bootMode, fullStats.bootMode);
  assert.equal(fastStats.currentPhase, fullStats.currentPhase);
  assert.equal(fastStats.phaseTransitionCount, fullStats.phaseTransitionCount);
  assert.equal(fastStats.destroyed, fullStats.destroyed);
  assert.equal(fastStats.uptimeMs, fullStats.uptimeMs);
  assert.equal(fastStats.reclaimRuns, fullStats.reclaimRuns);
  assert.equal(fastStats.reclaimFailures, fullStats.reclaimFailures);
  assert.equal(fastStats.validationErrorCount, fullStats.validationErrorCount);
  assert.equal(fastStats.splashFramesPresented, fullStats.splashFramesPresented);

  await controller.destroy();
});


