import test from 'node:test';
import assert from 'node:assert/strict';

import {
  WorkerPool,
  WorkerPoolError,
  WorkerCrashError,
  WorkerStallError,
  PoolTerminatedError,
  DEFAULT_WATCHDOG_INTERVAL_MS,
  DEFAULT_STALL_TIMEOUT_MS,
  serveWorkerPool,
} from '../web/workers/worker_pool.js';

import {
  RuntimeSessionController,
  SESSION_STATE,
  SESSION_MODE,
  DEFAULT_MAX_DEVICE_RECOVERIES,
} from '../web/runtime/session_controller.js';

import { PerfMonitor } from '../web/telemetry/perf_monitor.js';

const flush = () => new Promise((resolve) => setImmediate(resolve));

function createFullClock(startMs = 0) {
  let time = startMs;
  let nextId = 1;
  const timeouts = new Map();
  const intervals = new Map();

  return {
    now: () => time,
    setTimeout(fn, ms) {
      const id = nextId++;
      timeouts.set(id, { at: time + Math.max(0, ms), fn });
      return id;
    },
    clearTimeout(id) {
      timeouts.delete(id);
    },
    setInterval(fn, ms) {
      const id = nextId++;
      const step = Math.max(1, ms);
      intervals.set(id, { nextAt: time + step, step, fn });
      return id;
    },
    clearInterval(id) {
      intervals.delete(id);
    },
    advance(ms) {
      const target = time + ms;
      for (;;) {
        let best = null;
        for (const [id, t] of timeouts) {
          if (t.at <= target && (!best || t.at < best.at)) {
            best = { kind: 'timeout', id, at: t.at, fn: t.fn };
          }
        }
        for (const [id, iv] of intervals) {
          if (iv.nextAt <= target && (!best || iv.nextAt < best.at)) {
            best = { kind: 'interval', id, at: iv.nextAt, iv };
          }
        }
        if (!best) break;
        time = Math.max(time, best.at);
        if (best.kind === 'timeout') {
          timeouts.delete(best.id);
          best.fn();
        } else {
          // Reschedule before calling in case fn clears the interval.
          best.iv.nextAt = time + best.iv.step;
          best.iv.fn();
        }
      }
      time = target;
    },
    get pendingTimeouts() {
      return timeouts.size;
    },
    get pendingIntervals() {
      return intervals.size;
    },
    get pending() {
      return timeouts.size + intervals.size;
    },
  };
}

class TrackedFakeWorker {
  constructor({ respondToPing = false } = {}) {
    this.listeners = { message: new Set(), error: new Set(), messageerror: new Set(), exit: new Set() };
    this.received = [];
    this.transfers = [];
    this.terminated = false;
    this.respondToPing = respondToPing;
    this.throwOnPing = false;
  }
  addEventListener(type, fn) {
    if (!this.listeners[type]) this.listeners[type] = new Set();
    this.listeners[type].add(fn);
  }
  removeEventListener(type, fn) {
    this.listeners[type]?.delete(fn);
  }
  listenerCount(type) {
    if (type) return this.listeners[type]?.size ?? 0;
    let total = 0;
    for (const set of Object.values(this.listeners)) total += set.size;
    return total;
  }
  postMessage(msg, transfer = []) {
    if (this.terminated) throw new Error('postMessage on terminated worker');
    this.received.push(msg);
    this.transfers.push(transfer);
    if (msg && msg.__watchdogPing === true) {
      if (this.throwOnPing) {
        throw new Error('broken worker channel');
      }
      if (this.respondToPing) {
        this.reply({ __watchdogPong: true, pingId: msg.pingId });
      }
    }
  }
  terminate() {
    this.terminated = true;
  }
  emit(type, ev) {
    if (this.terminated) return;
    for (const fn of [...(this.listeners[type] || [])]) fn(ev);
  }
  reply(msg) {
    this.emit('message', { data: msg });
  }
  pings() {
    return this.received.filter((m) => m && m.__watchdogPing === true);
  }
  jobs() {
    return this.received.filter((m) => m && typeof m.type === 'string');
  }
  lastJob() {
    return this.jobs().at(-1);
  }
}

function createConnectedWorker(handlers) {
  const poolListeners = { message: new Set(), error: new Set(), messageerror: new Set(), exit: new Set() };
  const workerListeners = new Set();
  let terminated = false;
  let frozen = false;

  const workerSideScope = {
    addEventListener(type, fn) {
      if (type === 'message') workerListeners.add(fn);
    },
    removeEventListener(type, fn) {
      if (type === 'message') workerListeners.delete(fn);
    },
    postMessage(msg) {
      if (terminated || frozen) return;
      for (const fn of [...poolListeners.message]) fn({ data: msg });
    },
  };

  const unsubServe = serveWorkerPool(handlers, { scope: workerSideScope });

  return {
    get terminated() {
      return terminated;
    },
    setFrozen(val) {
      frozen = Boolean(val);
    },
    addEventListener(type, fn) {
      poolListeners[type]?.add(fn);
    },
    removeEventListener(type, fn) {
      poolListeners[type]?.delete(fn);
    },
    listenerCount() {
      let total = 0;
      for (const set of Object.values(poolListeners)) total += set.size;
      return total;
    },
    postMessage(msg) {
      if (terminated) throw new Error('terminated');
      if (frozen) return;
      for (const fn of [...workerListeners]) fn({ data: msg });
    },
    terminate() {
      terminated = true;
      unsubServe();
    },
  };
}

function createMockEventTarget(extra = {}) {
  const listeners = new Map();
  return {
    ...extra,
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
    dispatch(type, event = {}) {
      for (const fn of [...(listeners.get(type) || [])]) {
        fn({ type, ...event });
      }
    },
  };
}

function createMockGpuDevice(label = 'mock-gpu') {
  let resolveLost;
  let rejectLost;
  const lost = new Promise((resolve, reject) => {
    resolveLost = resolve;
    rejectLost = reject;
  });
  return {
    label,
    lost,
    destroyed: false,
    destroy() {
      if (this.destroyed) return;
      this.destroyed = true;
      resolveLost({ reason: 'destroyed', message: 'Device destroyed by application' });
    },
    simulateLost(reason = 'unknown', message = 'GPU adapter reset') {
      resolveLost({ reason, message });
    },
    simulateLostRejection(err) {
      rejectLost(err);
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

function fullCapabilities() {
  return {
    crossOriginIsolated: true,
    sharedArrayBuffer: true,
    webgpu: true,
    wasm64: true,
    opfs: true,
    gamepad: true,
    audioWorklet: true,
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
    shaderPackCount: 4,
    ...overrides,
  };
}

// ============================================================================
// PART 1: WorkerPool Health Watchdog, Stall Diagnostics & Zero-Leak Cleanup
// ============================================================================

test('W383 WorkerPool: validates watchdog & callback options and disables watchdog by default', async () => {
  assert.equal(DEFAULT_WATCHDOG_INTERVAL_MS, 0);
  assert.equal(DEFAULT_STALL_TIMEOUT_MS, 5000);

  const err = new WorkerStallError(3000, 1, 2);
  assert.ok(err instanceof WorkerStallError);
  assert.ok(err instanceof WorkerPoolError);
  assert.equal(err.name, 'WorkerStallError');
  assert.equal(err.code, 'WORKER_STALL');
  assert.equal(err.stallTimeoutMs, 3000);
  assert.equal(err.slotIndex, 1);
  assert.equal(err.missedPings, 2);

  const factory = () => new TrackedFakeWorker();
  assert.throws(() => new WorkerPool({ workerFactory: factory, watchdogIntervalMs: -1 }), RangeError);
  assert.throws(() => new WorkerPool({ workerFactory: factory, watchdogIntervalMs: NaN }), RangeError);
  assert.throws(() => new WorkerPool({ workerFactory: factory, watchdogIntervalMs: Infinity }), RangeError);
  assert.throws(() => new WorkerPool({ workerFactory: factory, stallTimeoutMs: 0 }), RangeError);
  assert.throws(() => new WorkerPool({ workerFactory: factory, stallTimeoutMs: -50 }), RangeError);
  assert.throws(() => new WorkerPool({ workerFactory: factory, onWorkerStall: 'bad' }), TypeError);
  assert.throws(() => new WorkerPool({ workerFactory: factory, onWorkerCrash: 123 }), TypeError);
  assert.throws(() => new WorkerPool({ workerFactory: factory, setInterval: 'bad' }), TypeError);
  assert.throws(() => new WorkerPool({ workerFactory: factory, clearInterval: {} }), TypeError);

  const clock = createFullClock();
  const pool = new WorkerPool({
    workerFactory: factory,
    maxWorkers: 2,
    now: clock.now,
    setTimeout: clock.setTimeout,
    clearTimeout: clock.clearTimeout,
    setInterval: clock.setInterval,
    clearInterval: clock.clearInterval,
  });

  assert.equal(clock.pending, 0);
  const health = pool.getHealthSnapshot();
  assert.equal(health.healthy, true);
  assert.equal(health.watchdog.enabled, false);
  assert.equal(health.watchdog.active, false);
  assert.equal(health.activeTimers, 0);
  assert.equal(pool.stats().watchdogEnabled, false);
  assert.equal(pool.stats().stalledWorkers, 0);
  await pool.terminate();
});

test('W383 WorkerPool: serveWorkerPool responds transparently to watchdog pings without interfering with jobs', async () => {
  const clock = createFullClock(1000);
  const workers = [];
  const pool = new WorkerPool({
    workerFactory: () => {
      const w = createConnectedWorker({
        double: (x) => x * 2,
        slow: async (x, ctx) => {
          ctx.progress(0.5);
          return x + 10;
        },
      });
      workers.push(w);
      return w;
    },
    maxWorkers: 1,
    watchdogIntervalMs: 500,
    stallTimeoutMs: 1500,
    now: clock.now,
    setTimeout: clock.setTimeout,
    clearTimeout: clock.clearTimeout,
    setInterval: clock.setInterval,
    clearInterval: clock.clearInterval,
  });

  assert.equal(clock.pendingIntervals, 1);
  assert.equal(await pool.submit('double', 21), 42);

  // Advance through 4 watchdog cycles while idle: pongs keep worker healthy.
  clock.advance(2000);
  const health = pool.getHealthSnapshot();
  assert.equal(health.watchdog.enabled, true);
  assert.equal(health.watchdog.active, true);
  assert.equal(health.watchdogPingsSent, 4);
  assert.equal(health.watchdogPongsReceived, 4);
  assert.equal(health.stalledWorkers, 0);
  assert.equal(health.slots.length, 1);
  assert.equal(health.slots[0].consecutiveMissedPings, 0);
  assert.equal(health.slots[0].pendingPingId, null);

  assert.equal(await pool.submit('slow', 5), 15);
  await pool.terminate();
  assert.equal(clock.pending, 0);
  assert.equal(workers[0].terminated, true);
  assert.equal(workers[0].listenerCount(), 0);
});

test('W383 WorkerPool: detects stalled worker mid-job, invokes onWorkerStall safely, recycles & retries without hanging', async () => {
  const clock = createFullClock(0);
  const workers = [];
  const stallEvents = [];

  const pool = new WorkerPool({
    workerFactory: () => {
      // First worker hangs (ignores pings); second worker is healthy.
      const w = new TrackedFakeWorker({ respondToPing: workers.length > 0 });
      workers.push(w);
      return w;
    },
    maxWorkers: 1,
    retries: 1,
    watchdogIntervalMs: 500,
    stallTimeoutMs: 1500,
    onWorkerStall: (info) => {
      stallEvents.push(info);
      // Throwing callback must be isolated.
      throw new Error('telemetry callback boom');
    },
    now: clock.now,
    setTimeout: clock.setTimeout,
    clearTimeout: clock.clearTimeout,
    setInterval: clock.setInterval,
    clearInterval: clock.clearInterval,
  });

  const jobPromise = pool.submit('decode', { chunk: 7 });
  assert.equal(workers.length, 1);

  // t=500 -> ping #1 sent; t=1000 -> 1 missed; t=1500 -> silentMs=1500 >= stallTimeoutMs -> stalled & recycled!
  clock.advance(1499);
  assert.equal(stallEvents.length, 0);
  assert.equal(workers[0].terminated, false);
  assert.equal(pool.getHealthSnapshot().slots[0].consecutiveMissedPings, 1);

  clock.advance(1);
  assert.equal(stallEvents.length, 1);
  assert.equal(stallEvents[0].slotIndex, 0);
  assert.equal(stallEvents[0].jobType, 'decode');
  assert.ok(stallEvents[0].error instanceof WorkerStallError);
  assert.equal(stallEvents[0].error.code, 'WORKER_STALL');
  assert.equal(workers[0].terminated, true);
  assert.equal(workers[0].listenerCount(), 0);

  // Replacement worker was spawned and received the retried job!
  assert.equal(workers.length, 2);
  assert.equal(workers[1].lastJob().payload.chunk, 7);
  workers[1].reply({ id: workers[1].lastJob().id, ok: true, result: 'decoded-ok' });

  assert.equal(await jobPromise, 'decoded-ok');
  const s = pool.stats();
  assert.equal(s.stalledWorkers, 1);
  assert.equal(s.watchdogRecycled, 1);
  assert.equal(s.retried, 1);
  assert.equal(s.crashed, 0);
  assert.equal(s.completed, 1);

  pool.resetStats();
  assert.equal(pool.stats().stalledWorkers, 0);
  assert.equal(pool.stats().watchdogRecycled, 0);
  assert.equal(pool.stats().watchdogPingsSent, 0);
  assert.equal(pool.stats().watchdogPongsReceived, 0);

  await pool.destroy();
  assert.equal(clock.pending, 0);
  assert.equal(pool.getHealthSnapshot().activeTimers, 0);
});

test('W383 WorkerPool: stalled non-retryable/transfer job rejects with WorkerStallError; progress resets stall window', async () => {
  const clock = createFullClock(0);
  const workers = [];
  const pool = new WorkerPool({
    workerFactory: () => {
      const w = new TrackedFakeWorker({ respondToPing: false });
      workers.push(w);
      return w;
    },
    maxWorkers: 1,
    retries: 2,
    watchdogIntervalMs: 500,
    stallTimeoutMs: 1500,
    now: clock.now,
    setTimeout: clock.setTimeout,
    clearTimeout: clock.clearTimeout,
    setInterval: clock.setInterval,
    clearInterval: clock.clearInterval,
  });

  // 1. Progress updates keep a busy worker alive even if pings are unanswered.
  const progressSeen = [];
  const longJob = pool.submit('stream', 1, { onProgress: (v) => progressSeen.push(v) });
  clock.advance(1000);
  workers[0].reply({ id: 1, progress: 0.3 });
  clock.advance(1000);
  workers[0].reply({ id: 1, progress: 0.7 });
  clock.advance(1000);
  assert.equal(workers[0].terminated, false);
  workers[0].reply({ id: 1, ok: true, result: 'stream-done' });
  assert.equal(await longJob, 'stream-done');
  assert.deepEqual(progressSeen, [0.3, 0.7]);

  // 2. Transferable job cannot be retried on stall -> rejects cleanly with WorkerStallError.
  const buf = new ArrayBuffer(16);
  const xferJob = pool.submit('upload', { buf }, { transfer: [buf] });
  clock.advance(2000);
  await assert.rejects(
    xferJob,
    (e) => e instanceof WorkerStallError && e.code === 'WORKER_STALL' && e instanceof WorkerPoolError
  );
  assert.equal(workers[0].terminated, true);
  assert.equal(pool.stats().stalledWorkers, 1);

  await pool.terminate();
  assert.equal(clock.pending, 0);
});

test('W383 WorkerPool: idle worker stall recycles warm slot, onWorkerCrash fires safely, and multi-cycle create/destroy leaks zero timers or listeners', async () => {
  const clock = createFullClock(0);
  const crashEvents = [];
  const allWorkers = [];

  for (let cycle = 0; cycle < 4; cycle++) {
    const cycleWorkers = [];
    const pool = new WorkerPool({
      workerFactory: () => {
        const w = new TrackedFakeWorker({ respondToPing: cycleWorkers.length > 0 });
        cycleWorkers.push(w);
        allWorkers.push(w);
        return w;
      },
      maxWorkers: 2,
      minWarmWorkers: 1,
      idleTimeoutMs: 5000,
      defaultTimeoutMs: 4000,
      watchdogIntervalMs: 250,
      stallTimeoutMs: 750,
      onWorkerCrash: (info) => {
        crashEvents.push(info);
        throw new Error('crash listener error');
      },
      now: clock.now,
      setTimeout: clock.setTimeout,
      clearTimeout: clock.clearTimeout,
      setInterval: clock.setInterval,
      clearInterval: clock.clearInterval,
    });

    // Warm 1 idle worker that ignores pings -> watchdog recycles it and respawns healthy warm replacement.
    pool.warmUp(1);
    clock.advance(1000);
    assert.equal(cycleWorkers[0].terminated, true);
    assert.equal(cycleWorkers.length, 2);
    assert.equal(pool.stats().stalledWorkers, 1);
    assert.equal(pool.stats().watchdogRecycled, 1);

    // Trigger a crash on the replacement worker and verify onWorkerCrash callback fired.
    const ac = new AbortController();
    const p1 = pool.submit('work', cycle, { signal: ac.signal });
    cycleWorkers[1].emit('error', { message: `crash-cycle-${cycle}` });
    await assert.rejects(p1, WorkerCrashError);
    assert.equal(crashEvents.length, cycle + 1);
    assert.equal(crashEvents[cycle].reason, `crash-cycle-${cycle}`);

    // Submit another job with AbortSignal and terminate mid-flight.
    const ac2 = new AbortController();
    const p2 = pool.submit('hang', cycle, { signal: ac2.signal });
    await pool.terminate();
    await assert.rejects(p2, PoolTerminatedError);

    // Zero timers and zero listener leaks after every cycle.
    assert.equal(clock.pending, 0);
    assert.equal(pool.getHealthSnapshot().activeTimers, 0);
    assert.equal(pool.getHealthSnapshot().workers, 0);
    ac2.abort(); // detached abort listener is a no-op
  }

  for (const w of allWorkers) {
    assert.equal(w.terminated, true);
    assert.equal(w.listenerCount(), 0);
  }
});

// ============================================================================
// PART 2: RuntimeSessionController WebGPU device.lost, Pause/Resume & Teardown
// ============================================================================

test('W383 SessionController: attachGpuDevice wires device.lost safely, deduplicates, suppresses destroyed, and recovers cleanly', async () => {
  assert.equal(DEFAULT_MAX_DEVICE_RECOVERIES, 3);
  const clock = createFullClock(10_000);
  const lostEvents = [];
  const recoveredEvents = [];
  const recoveryCalls = [];

  // Option validation checks
  assert.throws(() => new RuntimeSessionController({ maxDeviceRecoveries: -1 }), RangeError);
  assert.throws(() => new RuntimeSessionController({ onDeviceRecovery: 'bad' }), TypeError);
  assert.throws(() => new RuntimeSessionController({ recoverDevice: 42 }), TypeError);
  assert.throws(() => new RuntimeSessionController({ visibilityTarget: {} }), TypeError);

  const devices = [];
  const controller = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus()),
    capabilityProbe: fullCapabilities,
    now: clock.now,
    maxDeviceRecoveries: 3,
    onDeviceLost: (ev) => lostEvents.push(ev),
    onDeviceRecovered: (ev) => recoveredEvents.push(ev),
    recoverDevice: async (ctx) => {
      recoveryCalls.push(ctx);
      const nextDev = createMockGpuDevice(`gpu-gen-${ctx.generation + 1}`);
      devices.push(nextDev);
      return nextDev;
    },
    startWasmRuntime: async () => ({ stop() {} }),
  });

  assert.throws(() => controller.attachGpuDevice(null), TypeError);
  await controller.start();

  // 1. Intentional destroy (`reason === 'destroyed'`) must NOT trigger recovery or error transitions!
  const devIntentional = createMockGpuDevice('gpu-intentional');
  controller.attachGpuDevice(devIntentional);
  assert.equal(controller.getDeviceHealth().generation, 1);
  devIntentional.destroy();
  await flush();

  assert.equal(lostEvents.length, 0);
  assert.equal(recoveryCalls.length, 0);
  assert.equal(controller.getState(), SESSION_STATE.RUNNING_WASM);
  assert.equal(controller.getDeviceHealth().lost, false);
  assert.equal(controller.getDeviceHealth().lossCount, 0);

  // 2. Unexpected loss triggers sanitized diagnostics + bounded recovery returning replacement GPUDevice.
  const dev1 = createMockGpuDevice('gpu-1');
  devices.push(dev1);
  const detach1 = controller.attachGpuDevice(dev1);
  // Idempotent re-attach of same device returns same detach without bumping generation.
  assert.equal(controller.attachGpuDevice(dev1), detach1);
  assert.equal(controller.getDeviceHealth().generation, 2);

  dev1.simulateLost('unknown', 'Driver TDR at C:\\Users\\bob\\dx12.dll');
  await flush();
  await flush();

  assert.equal(lostEvents.length, 1);
  assert.equal(lostEvents[0].generation, 2);
  assert.doesNotMatch(lostEvents[0].message, /bob|C:\\/);
  assert.match(lostEvents[0].message, /<path>/);
  assert.equal(recoveryCalls.length, 1);
  assert.equal(recoveredEvents.length, 1);

  const health = controller.getDeviceHealth();
  assert.equal(health.lost, false);
  assert.equal(health.generation, 3);
  assert.equal(health.lossCount, 1);
  assert.equal(health.recoveryCount, 1);
  assert.equal(health.consecutiveLosses, 0);
  assert.equal(health.recoveryInFlight, false);
  assert.equal(health.maxRecoveries, 3);
  assert.equal(health.exhausted, false);

  // Duplicate loss notification for generation 2 is deduplicated cleanly.
  const dup = controller.recordDeviceLost({ reason: 'unknown', generation: 2 });
  assert.equal(dup.handled, true); // cleared after recovery; now duplicate for gen 2 again:
  const dup2 = controller.recordDeviceLost({ reason: 'unknown', generation: 2 });
  assert.equal(dup2.handled, false);
  assert.equal(dup2.duplicate, true);

  // 3. Rejected device.lost promise is caught safely without unhandled rejection.
  const devRej = createMockGpuDevice('gpu-reject');
  controller.attachGpuDevice(devRej, { autoRecover: false });
  devRej.simulateLostRejection(new Error('IPC lost'));
  await flush();
  assert.equal(controller.getDeviceHealth().lost, true);
  assert.match(controller.getDeviceHealth().lastLossReason, /IPC lost/);

  // Manual notifyDeviceRecovered clears lost state and attaches optional new device.
  const devManual = createMockGpuDevice('gpu-manual');
  controller.notifyDeviceRecovered({ reason: 'manual-reinit', device: devManual });
  assert.equal(controller.getDeviceHealth().lost, false);

  // Detaching before device.lost resolves suppresses stale notification.
  controller.detachGpuDevice();
  devManual.simulateLost('unknown', 'stale after detach');
  await flush();
  assert.equal(controller.getDeviceHealth().lost, false);

  await controller.destroy();
});

test('W383 SessionController: device.lost enforces maxDeviceRecoveries bound and transitions to error when exhausted', async () => {
  const clock = createFullClock(5000);
  let attempts = 0;
  const reported = [];
  const devices = [];

  const controller = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus()),
    capabilityProbe: fullCapabilities,
    now: clock.now,
    maxDeviceRecoveries: 2,
    crashReporter: {
      captureError(err, ctx) {
        reported.push({ message: err.message, ctx });
      },
    },
    recoverDevice: async () => {
      attempts += 1;
      const d = createMockGpuDevice(`gpu-recovered-${attempts}`);
      devices.push(d);
      return d;
    },
    startWasmRuntime: async () => ({ stop() {} }),
  });

  await controller.start();

  // Loss #1 -> recovery #1 succeeds -> returns devices[0]
  const d0 = createMockGpuDevice('gpu-0');
  controller.attachGpuDevice(d0);
  d0.simulateLost('unknown', 'TDR #1');
  await flush();
  await flush();
  assert.equal(attempts, 1);
  assert.equal(controller.getDeviceHealth().recoveryCount, 1);
  assert.equal(controller.getDeviceHealth().exhausted, false);

  // Loss #2 -> recovery #2 succeeds -> returns devices[1]
  devices[0].simulateLost('unknown', 'TDR #2');
  await flush();
  await flush();
  assert.equal(attempts, 2);
  assert.equal(controller.getDeviceHealth().recoveryCount, 2);
  assert.equal(controller.getDeviceHealth().exhausted, false);

  // Loss #3 -> budget exhausted (2/2 used) -> no 3rd recovery, transitions to ERROR!
  devices[1].simulateLost('unknown', 'TDR #3');
  await flush();
  await flush();
  assert.equal(attempts, 2);
  assert.equal(controller.getState(), SESSION_STATE.ERROR);

  const health = controller.getDeviceHealth();
  assert.equal(health.lost, true);
  assert.equal(health.lossCount, 3);
  assert.equal(health.recoveryCount, 2);
  assert.equal(health.exhausted, true);
  assert.equal(health.maxRecoveries, 2);
  assert.ok(reported.some((r) => r.ctx.category === 'webgpu_device_lost_exhausted'));

  await controller.destroy();
});

test('W383 SessionController: autoRecoverOnDeviceLost falls back to session restart when no recoverDevice hook is set', async () => {
  let launches = 0;
  let stops = 0;
  const recovered = [];

  const controller = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus()),
    capabilityProbe: fullCapabilities,
    autoRecoverOnDeviceLost: true,
    maxDeviceRecoveries: 2,
    onDeviceRecovered: (ev) => recovered.push(ev),
    startWasmRuntime: async () => {
      launches += 1;
      return {
        gpuDevice: createMockGpuDevice(`session-gpu-${launches}`),
        stop() {
          stops += 1;
        },
      };
    },
  });

  const startRes = await controller.start();
  assert.equal(launches, 1);
  const firstGpu = startRes.handle.gpuDevice;

  firstGpu.simulateLost('unknown', 'GPU hung');
  await flush();
  await flush();
  await flush();

  assert.equal(stops, 1);
  assert.equal(launches, 2);
  assert.equal(controller.getState(), SESSION_STATE.RUNNING_WASM);
  assert.equal(recovered.length, 1);
  assert.equal(controller.getDeviceHealth().lost, false);
  assert.equal(controller.getDeviceHealth().recoveryCount, 1);

  await controller.destroy();
});

test('W383 SessionController: pause/resume and visibilitychange coordination without corrupting state or leaking listeners', async () => {
  const clock = createFullClock(20_000);
  const trace = [];
  const doc = createMockEventTarget({ hidden: false, visibilityState: 'visible' });

  const audioEngine = {
    pause(reason) {
      trace.push(`audio:pause:${reason}`);
    },
    resume(reason) {
      trace.push(`audio:resume:${reason}`);
    },
  };
  const inputBridge = {
    running: false,
    start() {
      this.running = true;
    },
    stop() {
      this.running = false;
    },
    pause(reason) {
      trace.push(`input:pause:${reason}`);
    },
    resume(reason) {
      trace.push(`input:resume:${reason}`);
    },
  };
  const perfMonitor = {
    pause(reason) {
      trace.push(`perf:pause:${reason}`);
    },
    resume(reason) {
      trace.push(`perf:resume:${reason}`);
    },
    getSnapshot() {
      return { fps: 60 };
    },
  };

  const stateChanges = [];
  const controller = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus()),
    capabilityProbe: fullCapabilities,
    now: clock.now,
    audioEngine,
    inputBridge,
    perfMonitor,
    visibilityTarget: doc,
    startWasmRuntime: async () => ({
      pause(reason) {
        trace.push(`handle:pause:${reason}`);
      },
      resume(reason) {
        trace.push(`handle:resume:${reason}`);
      },
      stop() {},
    }),
  });
  controller.onStateChange((ev) => stateChanges.push(ev.state));

  assert.equal(doc.listenerCount('visibilitychange'), 1);

  // Pausing while idle returns ok: false cleanly.
  const idlePause = controller.pause('too-early');
  assert.equal(idlePause.ok, false);
  assert.equal(controller.isPaused(), false);

  await controller.start();
  assert.deepEqual(stateChanges, ['probing', 'starting', 'running_wasm']);

  // 1. Tab hidden -> auto-pause
  doc.hidden = true;
  doc.visibilityState = 'hidden';
  doc.dispatch('visibilitychange');

  assert.equal(controller.isPaused(), true);
  assert.equal(controller.getState(), SESSION_STATE.RUNNING_WASM);
  assert.deepEqual(stateChanges, ['probing', 'starting', 'running_wasm']);
  assert.deepEqual(trace, [
    'handle:pause:visibilitychange:hidden',
    'audio:pause:visibilitychange:hidden',
    'input:pause:visibilitychange:hidden',
    'perf:pause:visibilitychange:hidden',
  ]);

  // Live totalPausedMs advances while paused
  clock.advance(750);
  assert.equal(controller.getDiagnostics().paused, true);
  assert.equal(controller.getDiagnostics().pauseCount, 1);
  assert.equal(controller.getDiagnostics().totalPausedMs, 750);

  // Idempotent pause does not double-count
  const dupPause = controller.pause('again');
  assert.equal(dupPause.alreadyPaused, true);
  assert.equal(controller.getDiagnostics().pauseCount, 1);

  // 2. Tab visible -> auto-resume
  trace.length = 0;
  doc.hidden = false;
  doc.visibilityState = 'visible';
  doc.dispatch('visibilitychange');

  assert.equal(controller.isPaused(), false);
  assert.equal(controller.getState(), SESSION_STATE.RUNNING_WASM);
  assert.equal(controller.getDiagnostics().totalPausedMs, 750);
  assert.deepEqual(trace, [
    'perf:resume:visibilitychange:visible',
    'input:resume:visibilitychange:visible',
    'audio:resume:visibilitychange:visible',
    'handle:resume:visibilitychange:visible',
  ]);

  // 3. Manual pause is NOT overridden when tab switches hidden -> visible
  trace.length = 0;
  controller.pause('user-pause-menu');
  clock.advance(250);
  doc.hidden = true;
  doc.visibilityState = 'hidden';
  doc.dispatch('visibilitychange');
  doc.hidden = false;
  doc.visibilityState = 'visible';
  doc.dispatch('visibilitychange');
  assert.equal(controller.isPaused(), true);

  controller.resume('user-unpause');
  assert.equal(controller.isPaused(), false);
  assert.equal(controller.getDiagnostics().pauseCount, 2);
  assert.equal(controller.getDiagnostics().totalPausedMs, 1000);

  // 4. destroy() detaches visibilitychange listener cleanly
  await controller.destroy();
  assert.equal(doc.listenerCount('visibilitychange'), 0);
  assert.equal(controller.getDiagnostics().visibilityAttached, false);
});

test('W383 SessionController: leak-free multi-restart teardown across 48 pthreads, WorkerPool, SAB hooks, GPU devices, and listeners (#1077)', async () => {
  const clock = createFullClock(1000);
  const wgpuWorker = createMockEventTarget();
  const gestureTarget = createMockEventTarget();
  const doc = createMockEventTarget({ hidden: false, visibilityState: 'visible' });
  const perfMonitor = new PerfMonitor({
    now: clock.now,
    setInterval: clock.setInterval,
    clearInterval: clock.clearInterval,
  });

  const livePthreadHandles = new Set();
  const createdPoolWorkers = [];
  const createdDevices = [];
  let sabReleases = 0;
  let workerTerminates = 0;
  let cycleSeq = 0;

  const controller = new RuntimeSessionController({
    fetchImpl: createMockFetch(readyStatus()),
    capabilityProbe: fullCapabilities,
    now: clock.now,
    hud: {
      setStatus() {},
      appendLog() {},
      updateMetrics() {},
    },
    perfMonitor,
    wgpuWorker,
    visibilityTarget: doc,
    audioGestureTarget: gestureTarget,
    inputBridge: {
      running: false,
      start() {
        this.running = true;
      },
      stop() {
        this.running = false;
      },
    },
    audioEngine: {
      resumeOnUserGesture(target) {
        const fn = () => {};
        target.addEventListener('pointerdown', fn);
        return () => target.removeEventListener('pointerdown', fn);
      },
    },
    autoRecoverOnDeviceLost: true,
    startWasmRuntime: async (ctx) => {
      const id = ++cycleSeq;
      const pthreads = new Array(48).fill(null).map((_, idx) => `c${id}-pthread-${idx}`);
      for (const p of pthreads) livePthreadHandles.add(p);

      // Register a live WorkerPool with watchdog enabled
      const pool = new WorkerPool({
        workerFactory: () => {
          const w = new TrackedFakeWorker({ respondToPing: true });
          createdPoolWorkers.push(w);
          return w;
        },
        maxWorkers: 2,
        minWarmWorkers: 2,
        watchdogIntervalMs: 500,
        stallTimeoutMs: 2000,
        now: clock.now,
        setTimeout: clock.setTimeout,
        clearTimeout: clock.clearTimeout,
        setInterval: clock.setInterval,
        clearInterval: clock.clearInterval,
      });
      pool.warmUp(2);
      ctx.registerWorkerPool(pool);

      // Register SAB cleanup resource
      let sabRef = new SharedArrayBuffer(1024);
      ctx.registerCleanupResource(`cmd-ring-sab-${id}`, () => {
        sabRef = null;
        sabReleases += 1;
      }, { sab: true });

      const gpuDevice = createMockGpuDevice(`gpu-cycle-${id}`);
      createdDevices.push(gpuDevice);

      return {
        id,
        gpuDevice,
        activeWorkerCount: pthreads.length,
        stop() {
          // Simulate intentional GPU destroy during stop
          gpuDevice.destroy();
        },
        terminateWorkers() {
          workerTerminates += 1;
          for (const p of pthreads) livePthreadHandles.delete(p);
          this.activeWorkerCount = 0;
        },
        releaseBuffers() {
          sabReleases += 1;
          // Ensure ref stays used until released
          assert.equal(sabRef, null);
        },
      };
    },
  });

  // Cycle 1: start
  await controller.start();
  let diag = controller.getDiagnostics();
  assert.equal(diag.state, SESSION_STATE.RUNNING_WASM);
  // 1 wgpuWorker + 2 WorkerPool workers + 48 pthreads = 51 active workers!
  assert.equal(diag.activeWorkerCount, 51);
  assert.equal(diag.registeredWorkerPoolCount, 1);
  assert.equal(diag.registeredCleanupCount, 1);
  assert.equal(livePthreadHandles.size, 48);

  // Cycles 2..6: 5 consecutive restarts
  for (let i = 0; i < 5; i++) {
    const res = await controller.restart();
    assert.equal(res.ok, true);
    assert.equal(res.state, SESSION_STATE.RUNNING_WASM);
    await flush();

    // Only current session's 48 pthreads + 2 pool workers + 1 wgpuWorker exist — ZERO leaks!
    assert.equal(livePthreadHandles.size, 48);
    assert.equal(wgpuWorker.listenerCount('message'), 1);
    assert.equal(gestureTarget.listenerCount('pointerdown'), 1);
    assert.equal(doc.listenerCount('visibilitychange'), 1);
    // Only 1 PerfMonitor HUD interval + 1 current WorkerPool watchdog interval = 2 active intervals.
    assert.equal(clock.pendingIntervals, 2);

    diag = controller.getDiagnostics();
    assert.equal(diag.activeWorkerCount, 51);
    assert.equal(diag.registeredWorkerPoolCount, 1);
    assert.equal(diag.registeredCleanupCount, 1);
    assert.equal(diag.cleanShutdownCount, i + 1);
    // Each stop runs 1 registered SAB cleanup + 1 handle.releaseBuffers() = 2 per teardown
    assert.equal(diag.sabReleaseCount, (i + 1) * 2);
    // Intentional gpuDevice.destroy() during restart never caused spurious loss/recovery!
    assert.equal(diag.deviceHealth.lossCount, 0);
    assert.equal(diag.deviceHealth.recoveryCount, 0);
  }

  // Final destroy tears down EVERY resource cleanly
  await controller.destroy();
  await flush();

  const finalDiag = controller.getDiagnostics();
  assert.equal(finalDiag.state, SESSION_STATE.STOPPED);
  assert.equal(finalDiag.activeWorkerCount, 0);
  assert.equal(finalDiag.registeredWorkerPoolCount, 0);
  assert.equal(finalDiag.registeredCleanupCount, 0);
  assert.equal(finalDiag.cleanShutdownCount, 6);
  assert.equal(finalDiag.sabReleaseCount, 12);
  assert.equal(finalDiag.deviceHealth.lossCount, 0);
  assert.equal(livePthreadHandles.size, 0);
  assert.equal(workerTerminates, 6);
  assert.equal(sabReleases, 12);
  assert.equal(clock.pending, 0);
  assert.equal(wgpuWorker.listenerCount(), 0);
  assert.equal(gestureTarget.listenerCount(), 0);
  assert.equal(doc.listenerCount(), 0);

  for (const w of createdPoolWorkers) {
    assert.equal(w.terminated, true);
    assert.equal(w.listenerCount(), 0);
  }
});
