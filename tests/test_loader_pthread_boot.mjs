import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');

const {
  PTHREAD_HTTPFS_SAB_CHANNEL_NAME,
  PTHREAD_CMD_LOAD,
  PTHREAD_CMD_LOADED,
  PTHREAD_CMD_RUN,
  PTHREAD_CMD_OBJECT_TRANSFER,
  DEFAULT_MAX_CONCURRENT_PTHREAD_LOADS,
  createPatchedPthreadEnv,
  installPthreadHostImportShim,
  bootstrapEmscriptenPthreadWorker,
  createPthreadPoolCoordinator,
  installPthreadWorkerProxy,
  parseGuestManifest,
  loadGuestManifest,
  buildRuntimeModuleConfig,
} = require('../web/loader.js');

const { createHttpfsMailboxBuffer } = require('../web/workers/httpfs_mailbox.js');

class StressMockWorker {
  constructor(script, opts) {
    this.script = script;
    this.opts = opts;
    this.name = opts && typeof opts.name === 'string' ? opts.name : '';
    this.onmessage = null;
    this.onerror = null;
    this.listeners = { message: [], error: [] };
    this.received = [];
    this.terminated = false;
    this.seenLoad = false;
    this.loaded = false;
    this.onPostLoad = null;
    this.throwOnLoad = false;
    this.orderViolations = [];
  }

  addEventListener(type, fn) {
    if (!this.listeners[type]) {
      this.listeners[type] = [];
    }
    this.listeners[type].push(fn);
  }

  removeEventListener(type, fn) {
    const arr = this.listeners[type];
    if (!arr) return;
    const idx = arr.indexOf(fn);
    if (idx >= 0) {
      arr.splice(idx, 1);
    }
  }

  postMessage(msg, transfer) {
    if (this.terminated) {
      return;
    }
    if (msg && msg.cmd === PTHREAD_CMD_LOAD) {
      if (this.throwOnLoad) {
        throw new Error(`simulated CMD_LOAD postMessage failure on ${this.name}`);
      }
      this.seenLoad = true;
    } else if (
      msg &&
      (msg.cmd === PTHREAD_CMD_RUN || msg.cmd === PTHREAD_CMD_OBJECT_TRANSFER) &&
      !this.loaded
    ) {
      this.orderViolations.push({
        worker: this.name,
        cmd: msg.cmd,
        seenLoad: this.seenLoad,
        loaded: this.loaded,
      });
    }

    this.received.push({ msg, transfer });

    if (msg && msg.cmd === PTHREAD_CMD_LOAD && typeof this.onPostLoad === 'function') {
      this.onPostLoad(this, msg);
    }
  }

  emitMessage(data, { useEventTarget = true } = {}) {
    if (data && data.cmd === PTHREAD_CMD_LOADED) {
      this.loaded = true;
    }
    const ev = { data };
    if (useEventTarget) {
      for (const fn of this.listeners.message.slice()) {
        fn.call(this, ev);
      }
    }
    if (typeof this.onmessage === 'function') {
      this.onmessage(ev);
    }
  }

  emitError(message = 'worker error', { useEventTarget = true } = {}) {
    const ev = { message };
    if (useEventTarget) {
      for (const fn of this.listeners.error.slice()) {
        fn.call(this, ev);
      }
    }
    if (typeof this.onerror === 'function') {
      this.onerror(ev);
    }
  }

  terminate() {
    this.terminated = true;
  }
}

test('W86 52-thread burst spawn: SAB delivery, strict CMD_LOAD-before-CMD_RUN/OBJECT_TRANSFER ordering, bounded concurrency, and exact telemetry', async () => {
  const sab = createHttpfsMailboxBuffer(1024);
  const maxConcurrent = 4;
  const spawned = [];
  let inFlightLoads = 0;
  let observedMaxInFlight = 0;
  const pendingLoadResolvers = [];

  class BurstWorker extends StressMockWorker {
    constructor(script, opts) {
      super(script, opts);
      this.onPostLoad = (w) => {
        inFlightLoads += 1;
        observedMaxInFlight = Math.max(observedMaxInFlight, inFlightLoads);
        assert.ok(
          inFlightLoads <= maxConcurrent,
          `Concurrent CMD_LOAD exceeded cap: ${inFlightLoads} > ${maxConcurrent}`,
        );
        pendingLoadResolvers.push(() => {
          inFlightLoads -= 1;
          // Alternate EventTarget + onmessage delivery styles across the 52 threads.
          const idNum = Number(w.name.split('-').pop()) || 0;
          w.emitMessage({ cmd: PTHREAD_CMD_LOADED, threadId: idNum }, { useEventTarget: idNum % 2 === 0 });
        });
      };
      spawned.push(this);
    }
  }

  const { WrappedWorker, coordinator } = installPthreadWorkerProxy({
    WorkerCtor: BurstWorker,
    maxConcurrentPthreadLoads: maxConcurrent,
    getHttpfsMailboxSab: () => sab,
    redirectHref: 'http://127.0.0.1:8080/loader.js',
  });

  // Also verify non-pthread workers are never redirected or counted.
  const ioWorker = new WrappedWorker('./workers/io_worker.js');
  const gpuWorker = new WrappedWorker('./workers/wgpu_worker.js');
  assert.equal(ioWorker.script, './workers/io_worker.js');
  assert.equal(gpuWorker.script, './workers/wgpu_worker.js');
  assert.equal(coordinator.getTelemetry().pthreadSpawnedCount, 0);

  const workers = [];
  const emscriptenLoadedEvents = [];

  // Tight burst spawn of all 52 real Xbox guest pthreads: em-pthread-1 .. em-pthread-52.
  for (let i = 1; i <= 52; i += 1) {
    const w = new WrappedWorker('http://127.0.0.1:8080/dist/runtime.js', {
      name: `em-pthread-${i}`,
    });
    w.onmessage = (ev) => {
      if (ev.data && ev.data.cmd === PTHREAD_CMD_LOADED) {
        emscriptenLoadedEvents.push(i);
      }
    };
    workers.push(w);

    const transferBuf = new ArrayBuffer(16);

    if (i % 4 === 1) {
      // Hostile case 1: CMD_RUN and CMD_OBJECT_TRANSFER arrive BEFORE CMD_LOAD is even posted!
      w.postMessage({ cmd: PTHREAD_CMD_RUN, seq: 'early-run', thread: i });
      w.postMessage({ cmd: PTHREAD_CMD_OBJECT_TRANSFER, seq: 'early-xfer', buf: transferBuf }, [transferBuf]);
      w.postMessage({ cmd: PTHREAD_CMD_LOAD, thread: i });
      w.postMessage({ cmd: 'custom', seq: 'post-load-custom', thread: i });
    } else if (i % 4 === 2) {
      // Hostile case 2: CMD_LOAD followed immediately by multiple interleaved CMD_RUN & custom messages.
      w.postMessage({ cmd: PTHREAD_CMD_LOAD, thread: i });
      w.postMessage({ cmd: 'custom', seq: 'c1', thread: i });
      w.postMessage({ cmd: PTHREAD_CMD_RUN, seq: 'r1', thread: i });
      w.postMessage({ cmd: PTHREAD_CMD_OBJECT_TRANSFER, seq: 'x1', buf: transferBuf }, [transferBuf]);
      w.postMessage({ cmd: PTHREAD_CMD_RUN, seq: 'r2', thread: i });
    } else if (i % 4 === 3) {
      // Hostile case 3: Duplicate CMD_LOAD must not consume extra slots or duplicate loads.
      w.postMessage({ cmd: PTHREAD_CMD_LOAD, thread: i });
      w.postMessage({ cmd: PTHREAD_CMD_LOAD, thread: i, duplicate: true });
      w.postMessage({ cmd: PTHREAD_CMD_RUN, seq: 'r-dup', thread: i });
    } else {
      // Standard case 4: CMD_LOAD + CMD_RUN + transfer.
      w.postMessage({ cmd: PTHREAD_CMD_LOAD, thread: i });
      w.postMessage({ cmd: PTHREAD_CMD_RUN, seq: 'r-std', thread: i });
      w.postMessage({ cmd: PTHREAD_CMD_OBJECT_TRANSFER, seq: 'x-std', buf: transferBuf }, [transferBuf]);
    }
  }

  // Immediately after burst: exactly 4 in-flight, 48 queued.
  const midStats = coordinator.getTelemetry();
  assert.equal(midStats.pthreadSpawnedCount, 52);
  assert.equal(midStats.pthreadActiveCount, 52);
  assert.equal(midStats.pthreadPeakCount, 52);
  assert.equal(midStats.pthreadActiveLoads, maxConcurrent);
  assert.equal(midStats.pthreadLoadQueueDepth, 52 - maxConcurrent);
  assert.equal(midStats.pthreadFailedCount, 0);

  // While the first batch is in-flight (loading, not yet loaded), push an extra CMD_RUN to worker #1.
  workers[0].postMessage({ cmd: PTHREAD_CMD_RUN, seq: 'inflight-run', thread: 1 });

  // Drain all 52 loads in waves, simulating async Wasm instantiation completion.
  let safetyCounter = 0;
  while (pendingLoadResolvers.length > 0 && safetyCounter < 200) {
    safetyCounter += 1;
    const resolveNext = pendingLoadResolvers.shift();
    resolveNext();
  }

  assert.equal(emscriptenLoadedEvents.length, 52, 'All 52 pthreads must report CMD_LOADED');
  assert.equal(observedMaxInFlight, maxConcurrent);

  for (let i = 0; i < workers.length; i += 1) {
    const w = workers[i];
    const id = i + 1;
    assert.equal(w.script, 'http://127.0.0.1:8080/loader.js');
    assert.deepEqual(w.orderViolations, [], `Ordering violation on ${w.name}`);

    // First message is always httpfsMailboxSab.
    assert.equal(w.received[0].msg.__playgta6_httpfs_sab, sab, `${w.name} missing httpfsMailboxSab`);
    // Second message is always CMD_LOAD.
    assert.equal(w.received[1].msg.cmd, PTHREAD_CMD_LOAD, `${w.name} did not get CMD_LOAD second`);

    // Verify exact FIFO order of buffered post-load messages and transfer preservation.
    const postLoad = w.received.slice(2);
    if (id % 4 === 1) {
      const expectedSeqs =
        id === 1
          ? ['early-run', 'early-xfer', 'post-load-custom', 'inflight-run']
          : ['early-run', 'early-xfer', 'post-load-custom'];
      assert.deepEqual(
        postLoad.map((r) => r.msg.seq),
        expectedSeqs,
      );
      assert.equal(postLoad[1].transfer.length, 1);
      assert.ok(postLoad[1].transfer[0] instanceof ArrayBuffer);
    } else if (id % 4 === 2) {
      assert.deepEqual(
        postLoad.map((r) => r.msg.seq),
        ['c1', 'r1', 'x1', 'r2'],
      );
      assert.equal(postLoad[2].transfer.length, 1);
    } else if (id % 4 === 3) {
      assert.deepEqual(
        postLoad.map((r) => r.msg.seq),
        ['r-dup'],
      );
      const loadCount = w.received.filter((r) => r.msg && r.msg.cmd === PTHREAD_CMD_LOAD).length;
      assert.equal(loadCount, 1, `${w.name} received duplicate CMD_LOAD`);
    } else {
      assert.deepEqual(
        postLoad.map((r) => r.msg.seq),
        ['r-std', 'x-std'],
      );
      assert.equal(postLoad[1].transfer.length, 1);
    }

    // Post-loaded messages pass straight through immediately.
    w.postMessage({ cmd: PTHREAD_CMD_RUN, seq: 'live-after-load' });
    assert.equal(w.received[w.received.length - 1].msg.seq, 'live-after-load');
  }

  const finalStats = coordinator.getTelemetry();
  assert.equal(finalStats.pthreadSpawnedCount, 52);
  assert.equal(finalStats.pthreadActiveCount, 52);
  assert.equal(finalStats.pthreadPeakCount, 52);
  assert.equal(finalStats.pthreadLoadedCount, 52);
  assert.equal(finalStats.pthreadLoadQueueDepth, 0);
  assert.equal(finalStats.pthreadActiveLoads, 0);
  assert.equal(finalStats.pthreadFailedCount, 0);
  assert.equal(finalStats.pthreadPeakConcurrentLoads, maxConcurrent);
  assert.equal(finalStats.maxConcurrentPthreadLoads, maxConcurrent);
});

test('W86 worker termination and error free load slots without deadlock and leak zero tracked entries', () => {
  const sab = createHttpfsMailboxBuffer(1024);
  const maxConcurrent = 3;
  const activeResolvers = new Map();

  class FaultyBurstWorker extends StressMockWorker {
    constructor(script, opts) {
      super(script, opts);
      this.onPostLoad = (w) => {
        activeResolvers.set(w.name, () => {
          activeResolvers.delete(w.name);
          w.emitMessage({ cmd: PTHREAD_CMD_LOADED });
        });
      };
    }
  }

  const { WrappedWorker, coordinator } = installPthreadWorkerProxy({
    WorkerCtor: FaultyBurstWorker,
    maxConcurrentPthreadLoads: maxConcurrent,
    getHttpfsMailboxSab: () => sab,
  });

  const workers = [];
  for (let i = 1; i <= 52; i += 1) {
    const w = new WrappedWorker('./dist/runtime.js', { name: `em-pthread-${i}` });
    if (i === 10) {
      // Worker 10 will throw synchronously when CMD_LOAD is dispatched.
      w.throwOnLoad = true;
    }
    workers.push(w);
    w.postMessage({ cmd: PTHREAD_CMD_LOAD, thread: i });
    w.postMessage({ cmd: PTHREAD_CMD_RUN, thread: i });
  }

  assert.equal(coordinator.getTelemetry().pthreadPeakCount, 52);
  assert.equal(coordinator.getTelemetry().pthreadActiveLoads, 3);
  assert.equal(coordinator.getTelemetry().pthreadLoadQueueDepth, 49);

  // 1. Terminate an in-flight loading worker (#1) -> must free slot and immediately start #4!
  workers[0].terminate();
  workers[0].terminate(); // idempotent double terminate must not double-decrement!
  assert.equal(activeResolvers.has('em-pthread-4'), true, 'Terminating #1 must dispatch queued #4');

  // 2. Error an in-flight loading worker (#2) via EventTarget error -> must free slot & dispatch #5!
  workers[1].emitError('wasm OOM', { useEventTarget: true });
  workers[1].terminate(); // terminating already-failed worker must be idempotent!
  assert.equal(activeResolvers.has('em-pthread-5'), true, 'Errored #2 must dispatch queued #5');

  // 3. Terminate queued workers (#20, #21) and error queued workers (#30 via EventTarget, #31 via onerror) BEFORE they dispatch.
  workers[19].terminate();
  workers[20].terminate();
  workers[29].emitError('script load error', { useEventTarget: true });
  let onErrorObserved = 0;
  workers[30].onerror = () => {
    onErrorObserved += 1;
  };
  workers[30].emitError('network failure', { useEventTarget: false });
  assert.equal(onErrorObserved, 1);

  // Drain all remaining loads (including #10 which throws on CMD_LOAD dispatch and auto-advances!).
  let guard = 0;
  while (activeResolvers.size > 0 && guard < 200) {
    guard += 1;
    const [firstKey, resolve] = activeResolvers.entries().next().value;
    if (firstKey === 'em-pthread-1' || firstKey === 'em-pthread-2') {
      activeResolvers.delete(firstKey);
      continue;
    }
    resolve();
  }

  // Terminated: #1, #20, #21 (3 workers).
  // Failed: #2 (error while loading), #10 (throw on CMD_LOAD), #30, #31 (error while queued) = 4 workers.
  // Healthy loaded: 52 - 3 - 4 = 45 workers.
  const drainedStats = coordinator.getTelemetry();
  assert.equal(drainedStats.pthreadSpawnedCount, 52);
  assert.equal(drainedStats.pthreadPeakCount, 52);
  assert.equal(drainedStats.pthreadFailedCount, 4);
  assert.equal(drainedStats.pthreadLoadedCount, 45);
  assert.equal(drainedStats.pthreadActiveCount, 45);
  assert.equal(drainedStats.pthreadLoadQueueDepth, 0);
  assert.equal(drainedStats.pthreadActiveLoads, 0);
  assert.equal(coordinator.getTrackedCount(), 45);

  // Healthy loaded workers all received CMD_LOAD strictly before CMD_RUN with zero violations.
  const deadSet = new Set([1, 2, 10, 20, 21, 30, 31]);
  for (let i = 0; i < workers.length; i += 1) {
    const id = i + 1;
    const w = workers[i];
    assert.deepEqual(w.orderViolations, []);
    if (!deadSet.has(id)) {
      assert.equal(w.loaded, true, `Healthy worker em-pthread-${id} did not load`);
      assert.equal(
        w.received.some((r) => r.msg && r.msg.cmd === PTHREAD_CMD_RUN),
        true,
        `Healthy worker em-pthread-${id} did not receive CMD_RUN`,
      );
    }
  }

  // Terminate all remaining 45 workers and confirm ZERO tracked entries remain (no memory leak).
  for (const w of workers) {
    w.terminate();
  }
  const emptyStats = coordinator.getTelemetry();
  assert.equal(emptyStats.pthreadActiveCount, 0);
  assert.equal(emptyStats.pthreadLoadQueueDepth, 0);
  assert.equal(emptyStats.pthreadActiveLoads, 0);
  assert.equal(coordinator.getTrackedCount(), 0);
  for (const w of workers) {
    assert.equal(w.listeners.message.length, 0);
    assert.equal(w.listeners.error.length, 0);
  }
});

test('W86 synchronous re-entrant CMD_LOADED drain completes 64 pthreads with O(1) stack depth', () => {
  const sab = createHttpfsMailboxBuffer(1024);

  class SyncWorker extends StressMockWorker {
    constructor(script, opts) {
      super(script, opts);
      this.onPostLoad = (w) => {
        // Synchronously fire CMD_LOADED inside postMessage(CMD_LOAD).
        w.emitMessage({ cmd: PTHREAD_CMD_LOADED });
      };
    }
  }

  const { WrappedWorker, coordinator } = installPthreadWorkerProxy({
    WorkerCtor: SyncWorker,
    maxConcurrentPthreadLoads: DEFAULT_MAX_CONCURRENT_PTHREAD_LOADS,
    getHttpfsMailboxSab: () => sab,
  });

  const workers = [];
  for (let i = 1; i <= 64; i += 1) {
    const w = new WrappedWorker('./dist/runtime.js', { name: `em-pthread-${i}` });
    w.postMessage({ cmd: PTHREAD_CMD_RUN, thread: i });
    w.postMessage({ cmd: PTHREAD_CMD_LOAD, thread: i });
    workers.push(w);
  }

  const stats = coordinator.getTelemetry();
  assert.equal(stats.pthreadSpawnedCount, 64);
  assert.equal(stats.pthreadLoadedCount, 64);
  assert.equal(stats.pthreadLoadQueueDepth, 0);
  assert.equal(stats.pthreadActiveLoads, 0);
  assert.ok(stats.pthreadPeakConcurrentLoads <= DEFAULT_MAX_CONCURRENT_PTHREAD_LOADS);
  for (const w of workers) {
    assert.deepEqual(w.orderViolations, []);
    assert.equal(w.received[0].msg.__playgta6_httpfs_sab, sab);
    assert.equal(w.received[1].msg.cmd, PTHREAD_CMD_LOAD);
    assert.equal(w.received[2].msg.cmd, PTHREAD_CMD_RUN);
  }
});

test('W86 52-thread bootstrapEmscriptenPthreadWorker deduplicates SAB client creation across BroadcastChannel bursts', () => {
  const sab = createHttpfsMailboxBuffer(1024);
  const channels = new Map();
  let clientConstructCount = 0;

  class CountingHttpfsClient {
    constructor(buf) {
      this.buf = buf;
      clientConstructCount += 1;
    }
  }

  class MockBusChannel {
    constructor(name) {
      this.name = name;
      this.onmessage = null;
      this.closed = false;
      if (!channels.has(name)) {
        channels.set(name, new Set());
      }
      channels.get(name).add(this);
    }
    postMessage(data) {
      if (this.closed) return;
      for (const peer of channels.get(this.name) || []) {
        if (peer !== this && !peer.closed && typeof peer.onmessage === 'function') {
          peer.onmessage({ data });
        }
      }
    }
    close() {
      this.closed = true;
      const set = channels.get(this.name);
      if (set) set.delete(this);
    }
  }

  const coordSabChannel = new MockBusChannel(PTHREAD_HTTPFS_SAB_CHANNEL_NAME);
  coordSabChannel.onmessage = (ev) => {
    if (ev.data && ev.data.type === 'request-httpfs-sab') {
      coordSabChannel.postMessage({ type: 'httpfs-sab', sab });
    }
  };

  const handles = [];
  for (let i = 1; i <= 52; i += 1) {
    const scope = {
      name: `em-pthread-${i}`,
      location: { href: 'http://127.0.0.1:8080/loader.js' },
      WebAssembly: { Instance: WebAssembly.Instance },
      listeners: [],
      addEventListener(type, fn) {
        if (type === 'message') this.listeners.push(fn);
      },
      removeEventListener(type, fn) {
        const idx = this.listeners.indexOf(fn);
        if (idx >= 0) this.listeners.splice(idx, 1);
      },
    };
    const h = bootstrapEmscriptenPthreadWorker({
      scope,
      importScriptsFn: () => {},
      BroadcastChannelCtor: MockBusChannel,
      httpfsMailboxApi: { HttpfsMailboxClient: CountingHttpfsClient },
    });
    // Also deliver via direct Worker.postMessage({ __playgta6_httpfs_sab: sab }).
    for (const fn of scope.listeners.slice()) {
      fn({ data: { __playgta6_httpfs_sab: sab }, stopImmediatePropagation() {} });
    }
    assert.equal(h.hasHttpfsSab(), true);
    handles.push(h);
  }

  // Without deduplication, 52 BroadcastChannel handshakes + 52 postMessages would construct ~2,756 clients!
  assert.equal(clientConstructCount, 52, 'Each of 52 pthreads must construct HttpfsMailboxClient exactly once');

  for (const h of handles) {
    h.cleanup();
    assert.equal(h.hasHttpfsSab(), false);
  }
  coordSabChannel.close();
});

test('W86 parseGuestManifest and loadGuestManifest handle legacy table and shard-dispatch wasmMetrics cleanly', async () => {
  // 1. Legacy manifest without indirectDispatch or wasmMetrics (full backwards compatibility).
  const legacyNoMetrics = parseGuestManifest({
    runtimeJs: 'dist/runtime.js',
    sideModules: ['guest_part_0.wasm', 'guest_part_1.wasm', '../blocked.wasm'],
  });
  assert.deepEqual(legacyNoMetrics, {
    indirectDispatch: 'table',
    sideModules: ['guest_part_0.wasm', 'guest_part_1.wasm'],
    sideModuleCount: 2,
    wasmMetrics: null,
    maxConcurrentPthreadLoads: null,
    runtimeScriptUrl: 'dist/runtime.js',
  });

  // 2. Legacy table manifest with snake_case wasm_metrics + untrusted fields stripped.
  const legacyWithMetrics = parseGuestManifest({
    indirect_dispatch: 'TABLE',
    side_modules: [
      { name: 'guest_part_0.wasm', wasm_bytes: 1000 },
      { name: 'guest_part_1.wasm', wasm_bytes: 2000 },
      { name: 'guest_part_2.wasm', wasm_bytes: 3000 },
      { name: 'guest_part_3.wasm', wasm_bytes: 4000 },
      { name: 'guest_part_4.wasm', wasm_bytes: 5000 },
    ],
    wasm_metrics: {
      total_table_size: 437813,
      total_got_func_imports: 118420,
      side_module_count: 5,
      shard_count: 512,
      hostPath: 'C:\\secret\\game.exe',
      negativeVal: -5,
      nanVal: 'not-a-number',
    },
  });
  assert.equal(legacyWithMetrics.indirectDispatch, 'table');
  assert.equal(legacyWithMetrics.sideModuleCount, 5);
  assert.deepEqual(legacyWithMetrics.sideModules, [
    'guest_part_0.wasm',
    'guest_part_1.wasm',
    'guest_part_2.wasm',
    'guest_part_3.wasm',
    'guest_part_4.wasm',
  ]);
  assert.deepEqual(legacyWithMetrics.wasmMetrics, {
    sideModuleCount: 5,
    totalTableSize: 437813,
    totalGotFuncImports: 118420,
    shardCount: 512,
  });

  // 3. R37 shard-dispatch manifest with shrunken wasmTable (~379) & 0 side modules (functions.o linked into main).
  const shardZeroSides = parseGuestManifest(
    JSON.stringify({
      indirectDispatch: 'shard',
      sideModules: [],
      maxConcurrentPthreadLoads: 4,
      wasmMetrics: {
        totalTableSize: 379,
        totalGotFuncImports: 0,
        sideModuleCount: 0,
        dispatchShardCount: 379,
      },
    }),
  );
  assert.deepEqual(shardZeroSides, {
    indirectDispatch: 'shard',
    sideModules: [],
    sideModuleCount: 0,
    wasmMetrics: {
      sideModuleCount: 0,
      totalTableSize: 379,
      totalGotFuncImports: 0,
      dispatchShardCount: 379,
    },
    maxConcurrentPthreadLoads: 4,
    runtimeScriptUrl: null,
  });

  // 4. loadGuestManifest forwards manifest summary to onProgress and handles 404/invalid gracefully.
  const progressEvents = [];
  const reports = [];
  const loaded = await loadGuestManifest({
    url: './dist/guest_manifest.json',
    onProgress: (ev) => progressEvents.push(ev),
    reportFn: (msg) => reports.push(msg),
    fetchFn: async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        indirectDispatch: 'shard',
        sideModules: ['guest_part_0.wasm', 'guest_part_1.wasm', 'guest_part_2.wasm'],
        wasmMetrics: {
          totalTableSize: 379,
          totalGotFuncImports: 42,
          sideModuleCount: 3,
        },
      }),
    }),
  });
  assert.equal(loaded.indirectDispatch, 'shard');
  assert.equal(loaded.sideModuleCount, 3);
  assert.equal(progressEvents.length, 1);
  assert.deepEqual(progressEvents[0], {
    phase: 'manifest',
    indirectDispatch: 'shard',
    sideModules: ['guest_part_0.wasm', 'guest_part_1.wasm', 'guest_part_2.wasm'],
    sideModuleCount: 3,
    wasmMetrics: {
      sideModuleCount: 3,
      totalTableSize: 379,
      totalGotFuncImports: 42,
    },
  });
  assert.ok(reports[0].includes('dispatch=shard'));
  assert.ok(reports[0].includes('sideModules=3'));
  assert.ok(reports[0].includes('table=379'));

  // Missing or malformed manifest returns null without throwing.
  assert.equal(parseGuestManifest(null), null);
  assert.equal(parseGuestManifest('not json'), null);
  assert.equal(parseGuestManifest([]), null);
  assert.equal(
    await loadGuestManifest({ fetchFn: async () => ({ ok: false, status: 404 }) }),
    null,
  );
  assert.equal(
    await loadGuestManifest({
      fetchFn: async () => {
        throw new Error('offline');
      },
    }),
    null,
  );
});

test('W86 sideModules progress reporting works cleanly for 0, 1, 2, 3, and 6 side modules', () => {
  for (const count of [0, 1, 2, 3, 6]) {
    const names = Array.from({ length: count }, (_, idx) => `guest_part_${idx}.wasm`);
    const progressEvents = [];
    const reports = [];
    const cfg = buildRuntimeModuleConfig({
      locationHref: 'http://127.0.0.1:8080/loader.js',
      reportFn: (m) => reports.push(m),
      onProgress: (ev) => progressEvents.push(ev),
      guestManifest: {
        indirectDispatch: count <= 3 ? 'shard' : 'table',
        sideModules: names,
        wasmMetrics: {
          totalTableSize: count <= 3 ? 379 : 437813,
          totalGotFuncImports: count * 12,
          sideModuleCount: count,
        },
      },
    });

    assert.equal(cfg.indirectDispatch, count <= 3 ? 'shard' : 'table');
    assert.equal(cfg.manifestSummary.sideModuleCount, count);
    assert.equal(cfg.wasmMetrics.sideModuleCount, count);

    if (count === 0) {
      assert.equal('dynamicLibraries' in cfg, false);
      const snap = cfg.getSideModuleProgress();
      assert.deepEqual(snap, {
        loadedCount: 0,
        totalCount: 0,
        remainingCount: 0,
        complete: true,
        pct: 100,
        loadedModules: [],
        expectedModules: [],
      });
      continue;
    }

    assert.equal(cfg.dynamicLibraries.length, count);
    assert.equal(cfg.getSideModuleProgress().loadedCount, 0);
    assert.equal(cfg.getSideModuleProgress().totalCount, count);
    assert.equal(cfg.getSideModuleProgress().complete, false);

    for (let idx = 0; idx < count; idx += 1) {
      const modName = names[idx];
      assert.equal(cfg.locateFile(modName), `http://127.0.0.1:8080/dist/${modName}`);
      // Duplicate locate call must not double-count or double-emit.
      assert.equal(cfg.locateFile(`./${modName}`), `http://127.0.0.1:8080/dist/${modName}`);
    }

    assert.equal(reports.length, count);
    assert.equal(progressEvents.length, count);
    assert.equal(progressEvents[count - 1].loadedCount, count);
    assert.equal(progressEvents[count - 1].totalCount, count);
    assert.equal(progressEvents[count - 1].complete, true);
    assert.equal(progressEvents[count - 1].pct, 100);
    assert.equal(progressEvents[count - 1].indirectDispatch, count <= 3 ? 'shard' : 'table');
    assert.equal(progressEvents[count - 1].wasmMetrics.sideModuleCount, count);
    assert.equal(cfg.getSideModuleProgress().complete, true);
  }
});

test('W86 coordinator loader.js VM end-to-end: loads shard guest_manifest.json, coordinates 52 pthreads, and answers request-loader-status', async () => {
  const loaderPath = path.join(repoRoot, 'web/loader.js');
  const loaderSource = fs.readFileSync(loaderPath, 'utf8');
  const mainMessages = [];
  const progressMessages = [];
  const pthreadWorkers = [];
  let receivedRuntimeOpts = null;

  class MockBroadcastChannel {
    constructor(name) {
      this.name = name;
      this.onmessage = null;
    }
    postMessage(msg) {
      if (this.name === 'game-progress') {
        progressMessages.push(msg);
      }
    }
  }

  class MockCoordWorker extends StressMockWorker {
    constructor(script, opts) {
      super(script, opts);
      if (opts && typeof opts.name === 'string' && opts.name.startsWith('em-pthread')) {
        pthreadWorkers.push(this);
      }
    }
    postMessage(msg, transfer) {
      super.postMessage(msg, transfer);
      if (msg && msg.init) {
        queueMicrotask(() => {
          this.emitMessage({ ready: true, summary: 'ok', adapter: 'mock-gpu' });
        });
      }
    }
  }

  const workerSelf = {
    name: '',
    location: { href: 'http://127.0.0.1:8080/loader.js' },
    Worker: MockCoordWorker,
    onmessage: null,
    postMessage(msg) {
      mainMessages.push(msg);
    },
  };

  const sandbox = {
    self: workerSelf,
    globalThis: null,
    URL,
    BroadcastChannel: MockBroadcastChannel,
    Worker: MockCoordWorker,
    indexedDB: null,
    Uint8Array,
    Int32Array,
    Uint32Array,
    DataView,
    ArrayBuffer,
    SharedArrayBuffer,
    Atomics,
    BigInt,
    TextEncoder,
    TextDecoder,
    Map,
    Set,
    Promise,
    Error,
    Number,
    Math,
    Date,
    Object,
    Array,
    Proxy,
    Reflect,
    JSON,
    setTimeout,
    clearTimeout,
    fetch: async (url) => {
      if (url === './dist/runtime.js') {
        return { ok: true, status: 200 };
      }
      if (url === './dist/guest_manifest.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            indirectDispatch: 'shard',
            sideModules: ['guest_part_0.wasm', 'guest_part_1.wasm'],
            maxConcurrentPthreadLoads: 4,
            wasmMetrics: {
              totalTableSize: 379,
              totalGotFuncImports: 36,
              sideModuleCount: 2,
            },
          }),
        };
      }
      return { ok: false, status: 404 };
    },
    importScripts(scriptPath) {
      if (scriptPath === './dist/runtime.js') {
        workerSelf.createRuntimeModule = async (opts) => {
          receivedRuntimeOpts = opts;
          opts.locateFile('guest_part_0.wasm');
          opts.locateFile('guest_part_1.wasm');
          // Spawn 52 real pthreads during runtime boot.
          for (let i = 1; i <= 52; i += 1) {
            const w = new workerSelf.Worker('./dist/runtime.js', { name: `em-pthread-${i}` });
            w.postMessage({ cmd: PTHREAD_CMD_LOAD, thread: i });
            w.postMessage({ cmd: PTHREAD_CMD_RUN, thread: i });
          }
          return {};
        };
        return;
      }
      const resolved = path.resolve(repoRoot, 'web', scriptPath);
      vm.runInContext(fs.readFileSync(resolved, 'utf8'), context, { filename: resolved });
    },
  };
  sandbox.globalThis = sandbox;
  const context = vm.createContext(sandbox);
  vm.runInContext(loaderSource, context, { filename: loaderPath });

  workerSelf.onmessage({
    data: { canvas: { id: 'c' }, width: 1280, height: 720, dataBase: '/data/' },
  });
  await new Promise((r) => setTimeout(r, 30));

  assert.ok(receivedRuntimeOpts);
  assert.equal(receivedRuntimeOpts.indirectDispatch, 'shard');
  assert.equal(receivedRuntimeOpts.wasmMetrics.totalTableSize, 379);
  assert.equal(receivedRuntimeOpts.wasmMetrics.totalGotFuncImports, 36);
  assert.equal(pthreadWorkers.length, 52);

  // Mid-boot status check via request-loader-status (4 active loads, 48 queued).
  workerSelf.onmessage({ data: { type: 'request-loader-status', id: 101 } });
  const midStatus = mainMessages.find((m) => m && m.type === 'loader-status' && m.id === 101);
  assert.ok(midStatus);
  assert.equal(midStatus.pthreadSpawnedCount, 52);
  assert.equal(midStatus.pthreadActiveCount, 52);
  assert.equal(midStatus.pthreadPeakCount, 52);
  assert.equal(midStatus.pthreadLoadQueueDepth, 48);
  assert.equal(midStatus.pthreadActiveLoads, 4);
  assert.equal(midStatus.pthreadFailedCount, 0);
  assert.equal(midStatus.indirectDispatch, 'shard');
  assert.equal(midStatus.wasmMetrics.totalTableSize, 379);
  assert.equal(midStatus.sideModules.loadedCount, 2);
  assert.equal(midStatus.sideModules.complete, true);

  // Complete all 52 pthreads.
  for (let round = 0; round < 20; round += 1) {
    let progressed = false;
    for (const w of pthreadWorkers) {
      if (w.seenLoad && !w.loaded) {
        w.emitMessage({ cmd: PTHREAD_CMD_LOADED });
        progressed = true;
      }
    }
    if (!progressed) break;
  }

  for (const w of pthreadWorkers) {
    assert.equal(w.loaded, true);
    assert.deepEqual(w.orderViolations, []);
    assert.ok(w.received[0].msg.__playgta6_httpfs_sab instanceof SharedArrayBuffer);
    assert.equal(w.received[1].msg.cmd, PTHREAD_CMD_LOAD);
    assert.equal(w.received[2].msg.cmd, PTHREAD_CMD_RUN);
  }

  workerSelf.onmessage({ data: { type: 'request-loader-status', id: 102 } });
  const endStatus = mainMessages.find((m) => m && m.type === 'loader-status' && m.id === 102);
  assert.equal(endStatus.pthreadSpawnedCount, 52);
  assert.equal(endStatus.pthreadActiveCount, 52);
  assert.equal(endStatus.pthreadPeakCount, 52);
  assert.equal(endStatus.pthreadLoadedCount, 52);
  assert.equal(endStatus.pthreadLoadQueueDepth, 0);
  assert.equal(endStatus.pthreadActiveLoads, 0);
  assert.equal(endStatus.pthreadFailedCount, 0);

  // Confirm BroadcastChannel metrics update carried shard wasmMetrics.
  const metricsMsg = progressMessages.find((m) => m && m.metrics && m.metrics.indirectDispatch === 'shard');
  assert.ok(metricsMsg);
  assert.equal(metricsMsg.metrics.wasmMetrics.totalTableSize, 379);
});

test('W86 createPatchedPthreadEnv and installPthreadHostImportShim handle .stub === true edge cases', () => {
  const makeStub = (name, stubVal = true) => {
    const fn = () => {
      throw new Error(`abort("missing function: ${name}")`);
    };
    fn.stub = stubVal;
    return fn;
  };
  const hostEnv = {
    host_log: () => 'real-log',
    host_httpfs_open: () => 1,
    host_httpfs_read: () => 2n,
    host_httpfs_close: () => 'real-close',
    host_wgpu_submit: () => 'real-wgpu',
  };

  // 1. Frozen baseEnv with non-enumerable own property preserved on clone + stub replaced.
  const frozenBase = { host_log: makeStub('host_log'), keep: 42 };
  Object.defineProperty(frozenBase, 'hiddenHelper', {
    value: () => 'hidden-ok',
    enumerable: false,
    writable: false,
    configurable: false,
  });
  Object.freeze(frozenBase);
  const patchedFrozen = createPatchedPthreadEnv(frozenBase, hostEnv, null);
  assert.notEqual(patchedFrozen, frozenBase);
  assert.equal(patchedFrozen.host_log, hostEnv.host_log);
  assert.equal(patchedFrozen.keep, 42);
  assert.equal(patchedFrozen.hiddenHelper(), 'hidden-ok');

  // 2. Configurable non-writable own .stub === true property is redefined in-place.
  const configNonWritable = {};
  Object.defineProperty(configNonWritable, 'host_log', {
    value: makeStub('host_log'),
    writable: false,
    configurable: true,
    enumerable: true,
  });
  const patchedCfg = createPatchedPthreadEnv(configNonWritable, hostEnv, null);
  assert.equal(configNonWritable.host_log, hostEnv.host_log);
  assert.equal(patchedCfg.host_log, hostEnv.host_log);

  // 3. Non-configurable non-writable own .stub === true property triggers clone (no Proxy invariant TypeError).
  const nonConfigStub = {};
  Object.defineProperty(nonConfigStub, 'host_wgpu_submit', {
    value: makeStub('host_wgpu_submit'),
    writable: false,
    configurable: false,
    enumerable: true,
  });
  const patchedNonCfg = createPatchedPthreadEnv(nonConfigStub, hostEnv, null);
  assert.equal(patchedNonCfg.host_wgpu_submit, hostEnv.host_wgpu_submit);

  // 4. Sealed object with writable .stub === true is patched in-place.
  const sealedEnv = Object.seal({
    host_log: makeStub('host_log'),
    host_httpfs_open: makeStub('host_httpfs_open'),
  });
  const patchedSealed = createPatchedPthreadEnv(sealedEnv, hostEnv, null);
  assert.equal(sealedEnv.host_log, hostEnv.host_log);
  assert.equal(sealedEnv.host_httpfs_open, hostEnv.host_httpfs_open);
  assert.equal(patchedSealed.host_httpfs_read, hostEnv.host_httpfs_read);

  // 5. Functions with stub: false, stub: 1, stub: 'true' (not strictly === true) are preserved!
  const falseStub = makeStub('host_log', false);
  const numericStub = makeStub('host_httpfs_open', 1);
  const stringStub = makeStub('host_httpfs_read', 'true');
  const strictEnv = {
    host_log: falseStub,
    host_httpfs_open: numericStub,
    host_httpfs_read: stringStub,
    host_wgpu_submit: makeStub('host_wgpu_submit', true),
  };
  const patchedStrict = createPatchedPthreadEnv(strictEnv, hostEnv, null);
  assert.equal(strictEnv.host_log, falseStub);
  assert.equal(strictEnv.host_httpfs_open, numericStub);
  assert.equal(strictEnv.host_httpfs_read, stringStub);
  assert.equal(strictEnv.host_wgpu_submit, hostEnv.host_wgpu_submit);
  assert.equal(patchedStrict.host_log, falseStub);
  assert.equal(patchedStrict.host_httpfs_open, numericStub);
  assert.equal(patchedStrict.host_httpfs_read, stringStub);

  // 6. Proxy baseEnv whose getOwnPropertyDescriptor and set throw still safely patches via Proxy get!
  const hostileProxy = new Proxy(
    { host_log: makeStub('host_log'), lifted: () => 7 },
    {
      getOwnPropertyDescriptor() {
        throw new TypeError('descriptor trap blocked');
      },
      set() {
        throw new TypeError('set trap blocked');
      },
    },
  );
  const patchedHostile = createPatchedPthreadEnv(hostileProxy, hostEnv, null);
  assert.equal(patchedHostile.host_log, hostEnv.host_log);
  assert.equal(patchedHostile.lifted(), 7);

  // 7. Subset requiredHostNames only patches listed symbols.
  const subsetEnv = {
    host_log: makeStub('host_log'),
    host_wgpu_submit: makeStub('host_wgpu_submit'),
  };
  const origWgpuStub = subsetEnv.host_wgpu_submit;
  const patchedSubset = createPatchedPthreadEnv(subsetEnv, hostEnv, new Set(['host_log']));
  assert.equal(subsetEnv.host_log, hostEnv.host_log);
  assert.equal(subsetEnv.host_wgpu_submit, origWgpuStub);
  assert.equal(patchedSubset.host_wgpu_submit, origWgpuStub);

  // 8. Null/non-object baseEnv returns safely without throwing.
  assert.equal(createPatchedPthreadEnv(null, hostEnv, null), null);
  assert.equal(createPatchedPthreadEnv(undefined, hostEnv, null), undefined);

  // 9. installPthreadHostImportShim end-to-end with non-configurable stub in MockWasmInstance.
  let captured = null;
  class MockInst {
    constructor(_mod, imports) {
      captured = imports;
    }
  }
  const scope = { WebAssembly: { Instance: MockInst } };
  const restore = installPthreadHostImportShim(scope, { env: hostEnv });
  new scope.WebAssembly.Instance('mod', { env: nonConfigStub });
  assert.equal(captured.env.host_wgpu_submit, hostEnv.host_wgpu_submit);
  restore();
});

test('W86 regression: bare manifest runtimeJs resolves under dist/ so runtime.wasm never 404s', () => {
  const cfg = buildRuntimeModuleConfig({
    locationHref: 'http://127.0.0.1:8080/loader.js',
    guestManifest: {
      runtimeJs: 'runtime.js',
      indirectDispatch: 'shard',
      sideModules: ['guest_part_0.wasm', 'guest_part_1.wasm', 'guest_part_2.wasm'],
    },
  });
  assert.equal(cfg.mainScriptUrlOrBlob, 'http://127.0.0.1:8080/dist/runtime.js');
  assert.equal(cfg.locateFile('runtime.wasm'), 'http://127.0.0.1:8080/dist/runtime.wasm');
  assert.equal(
    cfg.locateFile('guest_part_0.wasm'),
    'http://127.0.0.1:8080/dist/guest_part_0.wasm',
  );
});
