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

const schedulerMod = require('../web/workers/io_scheduler.js');
const rangeCacheMod = require('../web/workers/range_cache.js');

const {
  IO_PRIORITY,
  IoPriorityScheduler,
  DEFAULT_AGING_INTERVAL_MS,
  DEFAULT_AGING_WINDOW_MS,
  DEFAULT_DEDUPE_IN_FLIGHT,
  makeReadKey,
  classifyIoPriority,
  classifyBootArchivePriority,
  isBootCriticalRpfPath,
  isBinkVideoPath,
  isShaderPackPath,
  isUiFontRpfPath,
  isRetryableError,
} = schedulerMod;
const { RangeCache } = rangeCacheMod;

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function makeBytes(tag, length = 8) {
  const out = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = (tag + i) & 0xff;
  }
  return out;
}

test('UMD exports IO_PRIORITY and IoPriorityScheduler in CommonJS and Worker globalThis', () => {
  assert.equal(IO_PRIORITY.CRITICAL, 0);
  assert.equal(IO_PRIORITY.HIGH, 1);
  assert.equal(IO_PRIORITY.NORMAL, 2);
  assert.equal(IO_PRIORITY.PREFETCH, 3);
  assert.ok(Object.isFrozen(IO_PRIORITY));
  assert.equal(typeof IoPriorityScheduler, 'function');

  // Verify classic Web Worker (importScripts) global binding without CommonJS module.exports.
  const src = fs.readFileSync(path.join(repoRoot, 'web/workers/io_scheduler.js'), 'utf8');
  const sandbox = {};
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;
  vm.runInNewContext(src, sandbox);

  assert.ok(sandbox.PlayGta6IoScheduler);
  assert.equal(sandbox.PlayGta6IoScheduler.IO_PRIORITY.CRITICAL, 0);
  assert.equal(sandbox.PlayGta6IoScheduler.IO_PRIORITY.PREFETCH, 3);
  assert.equal(typeof sandbox.PlayGta6IoScheduler.IoPriorityScheduler, 'function');
});

test('constructor validates rangeCache and applies defaults', () => {
  assert.throws(() => new IoPriorityScheduler(), TypeError);
  assert.throws(() => new IoPriorityScheduler({ rangeCache: {} }), TypeError);

  const scheduler = new IoPriorityScheduler({
    rangeCache: { read: async () => new Uint8Array(0) },
  });
  assert.equal(scheduler.maxConcurrency, 6);
  assert.equal(scheduler.maxRetries, 2);
  assert.equal(scheduler.baseRetryDelayMs, 50);
});

test('strict priority ordering: CRITICAL > HIGH > NORMAL > PREFETCH and FIFO within same priority', async () => {
  const dispatchOrder = [];
  const gates = new Map();

  const rangeCache = {
    read(filePath, offset, length) {
      dispatchOrder.push(filePath);
      const gate = gates.get(filePath);
      if (gate) {
        return gate.promise;
      }
      return Promise.resolve(makeBytes(filePath.charCodeAt(0), length));
    },
  };

  const blockerGate = deferred();
  gates.set('blocker.bin', blockerGate);

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 1,
    sleep: async () => {},
  });

  // Occupy the single concurrency slot so subsequent reads queue up.
  const pBlocker = scheduler.scheduleRead('blocker.bin', 0, 4, {
    priority: IO_PRIORITY.PREFETCH,
  });

  // Queue reads in reverse / mixed priority order.
  const pPrefetch1 = scheduler.scheduleRead('prefetch-1.bin', 0, 4, {
    priority: IO_PRIORITY.PREFETCH,
  });
  const pPrefetch2 = scheduler.scheduleRead('prefetch-2.bin', 0, 4, {
    priority: IO_PRIORITY.PREFETCH,
  });
  const pNormal1 = scheduler.scheduleRead('normal-1.bin', 0, 4, {
    priority: IO_PRIORITY.NORMAL,
  });
  const pNormal2 = scheduler.scheduleRead('normal-2.bin', 0, 4); // default NORMAL
  const pHigh1 = scheduler.scheduleRead('high-1.bin', 0, 4, {
    priority: IO_PRIORITY.HIGH,
  });
  const pCritical1 = scheduler.scheduleRead('critical-1.bin', 0, 4, {
    priority: IO_PRIORITY.CRITICAL,
  });
  const pCritical2 = scheduler.scheduleRead('critical-2.bin', 0, 4, {
    priority: IO_PRIORITY.CRITICAL,
  });
  const pHigh2 = scheduler.scheduleRead('high-2.bin', 0, 4, {
    priority: IO_PRIORITY.HIGH,
  });

  assert.deepEqual(dispatchOrder, ['blocker.bin']);
  assert.equal(scheduler.getStats().queued, 8);
  assert.equal(scheduler.getStats().inFlight, 1);

  blockerGate.resolve(makeBytes(1, 4));

  await Promise.all([
    pBlocker,
    pPrefetch1,
    pPrefetch2,
    pNormal1,
    pNormal2,
    pHigh1,
    pHigh2,
    pCritical1,
    pCritical2,
  ]);

  assert.deepEqual(dispatchOrder, [
    'blocker.bin',
    'critical-1.bin',
    'critical-2.bin',
    'high-1.bin',
    'high-2.bin',
    'normal-1.bin',
    'normal-2.bin',
    'prefetch-1.bin',
    'prefetch-2.bin',
  ]);
});

test('maxConcurrency strictly bounds active in-flight rangeCache reads', async () => {
  let active = 0;
  let peakActive = 0;
  const pendingGates = [];

  const rangeCache = {
    read(filePath, offset, length) {
      active += 1;
      if (active > peakActive) {
        peakActive = active;
      }
      const gate = deferred();
      pendingGates.push({ gate, filePath, length });
      return gate.promise.finally(() => {
        active -= 1;
      });
    },
  };

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 3,
    sleep: async () => {},
  });

  const promises = [];
  for (let i = 0; i < 10; i += 1) {
    promises.push(scheduler.scheduleRead(`chunk_${i}.rpf`, i * 16, 16));
  }

  assert.equal(active, 3);
  assert.equal(peakActive, 3);
  assert.equal(scheduler.getStats().inFlight, 3);
  assert.equal(scheduler.getStats().queued, 7);

  // Release reads one by one and confirm concurrency never exceeds 3.
  while (pendingGates.length > 0) {
    const next = pendingGates.shift();
    next.gate.resolve(makeBytes(1, next.length));
    // Flush microtasks so _drain() dispatches the next queued read.
    await Promise.resolve();
    await Promise.resolve();
    assert.ok(active <= 3, `Expected active <= 3, saw ${active}`);
  }

  const results = await Promise.all(promises);
  assert.equal(results.length, 10);
  assert.equal(peakActive, 3);
  assert.equal(active, 0);
  assert.equal(scheduler.getStats().inFlight, 0);
  assert.equal(scheduler.getStats().queued, 0);
  assert.equal(scheduler.getStats().completed, 10);
});

test('AbortSignal cancellation before dispatch rejects with AbortError without calling rangeCache', async () => {
  const calledPaths = [];
  const blockerGate = deferred();

  const rangeCache = {
    read(filePath) {
      calledPaths.push(filePath);
      if (filePath === 'blocker.bin') {
        return blockerGate.promise;
      }
      return Promise.resolve(makeBytes(42, 4));
    },
  };

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 1,
    sleep: async () => {},
  });

  // 1. Pre-aborted signal rejects immediately without touching rangeCache.
  const preAborted = new AbortController();
  preAborted.abort();
  await assert.rejects(
    scheduler.scheduleRead('already_aborted.bin', 0, 8, {
      priority: IO_PRIORITY.HIGH,
      signal: preAborted.signal,
    }),
    { name: 'AbortError' }
  );
  assert.deepEqual(calledPaths, []);

  // 2. Queued read aborted while waiting in queue.
  const pBlocker = scheduler.scheduleRead('blocker.bin', 0, 8, {
    priority: IO_PRIORITY.CRITICAL,
  });

  const queuedController = new AbortController();
  const pQueuedCancelled = scheduler.scheduleRead('queued_cancel.bin', 0, 8, {
    priority: IO_PRIORITY.NORMAL,
    signal: queuedController.signal,
  });
  const pQueuedKept = scheduler.scheduleRead('queued_kept.bin', 0, 8, {
    priority: IO_PRIORITY.NORMAL,
  });

  assert.equal(scheduler.getStats().queued, 2);

  queuedController.abort();
  await assert.rejects(pQueuedCancelled, { name: 'AbortError' });

  // Immediately removed from queue and never dispatched.
  assert.equal(scheduler.getStats().queued, 1);
  assert.equal(scheduler.getStats().cancelled, 2);

  blockerGate.resolve(makeBytes(7, 8));
  await Promise.all([pBlocker, pQueuedKept]);

  assert.deepEqual(calledPaths, ['blocker.bin', 'queued_kept.bin']);
  const stats = scheduler.getStats();
  assert.equal(stats.completed, 2);
  assert.equal(stats.cancelled, 2);
  assert.equal(stats.failed, 0);
});

test('AbortSignal cancels across retries during backoff sleep without retrying again', async () => {
  let readCalls = 0;
  const recordedDelays = [];
  const controller = new AbortController();

  const rangeCache = {
    async read() {
      readCalls += 1;
      throw new Error('HTTP 503 reading streaming.rpf[0..255]');
    },
  };

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 2,
    maxRetries: 3,
    baseRetryDelayMs: 40,
    sleep: async (ms) => {
      recordedDelays.push(ms);
      // Abort while waiting in retry backoff before retry attempt 1 starts.
      controller.abort();
    },
  });

  await assert.rejects(
    scheduler.scheduleRead('streaming.rpf', 0, 256, {
      priority: IO_PRIORITY.HIGH,
      signal: controller.signal,
    }),
    { name: 'AbortError' }
  );

  assert.equal(readCalls, 1);
  assert.deepEqual(recordedDelays, [40]);

  const stats = scheduler.getStats();
  assert.equal(stats.cancelled, 1);
  assert.equal(stats.failed, 0);
  assert.equal(stats.retries, 0);
  assert.equal(stats.inFlight, 0);
});

test('AbortSignal aborts a pending retry sleep immediately when triggered externally', async () => {
  let readCalls = 0;
  const sleepStarted = deferred();
  const neverEndingSleep = deferred();
  const controller = new AbortController();

  const rangeCache = {
    async read() {
      readCalls += 1;
      throw new TypeError('Failed to fetch');
    },
  };

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 1,
    maxRetries: 2,
    baseRetryDelayMs: 100,
    sleep: () => {
      sleepStarted.resolve();
      return neverEndingSleep.promise;
    },
  });

  const p = scheduler.scheduleRead('world.rpf', 0, 128, {
    signal: controller.signal,
  });

  await sleepStarted.promise;
  assert.equal(scheduler.getStats().inFlight, 1);

  controller.abort();
  await assert.rejects(p, { name: 'AbortError' });

  assert.equal(readCalls, 1);
  assert.equal(scheduler.getStats().inFlight, 0);
  assert.equal(scheduler.getStats().cancelled, 1);
  assert.equal(scheduler.getStats().retries, 0);
});

test('retry backoff succeeds on 2nd attempt for transient HTTP 5xx, 429, and network errors', async () => {
  const transientErrors = [
    new Error('HTTP 503 reading file.rpf[0..63]'),
    Object.assign(new Error('Too Many Requests'), { status: 429 }),
    new TypeError('Failed to fetch'),
    Object.assign(new Error('socket reset'), { code: 'ECONNRESET' }),
  ];

  for (let i = 0; i < transientErrors.length; i += 1) {
    const errToThrow = transientErrors[i];
    const delays = [];
    let attempts = 0;

    const rangeCache = {
      async read(filePath, offset, length) {
        attempts += 1;
        if (attempts === 1) {
          throw errToThrow;
        }
        return makeBytes(99, length);
      },
    };

    const scheduler = new IoPriorityScheduler({
      rangeCache,
      maxConcurrency: 2,
      maxRetries: 2,
      baseRetryDelayMs: 50,
      sleep: async (ms) => {
        delays.push(ms);
      },
    });

    const bytes = await scheduler.scheduleRead(`transient_${i}.rpf`, 0, 16, {
      priority: IO_PRIORITY.NORMAL,
    });

    assert.equal(attempts, 2);
    assert.deepEqual(delays, [50]);
    assert.deepEqual(bytes, makeBytes(99, 16));

    const stats = scheduler.getStats();
    assert.equal(stats.completed, 1);
    assert.equal(stats.failed, 0);
    assert.equal(stats.retries, 1);
    assert.equal(stats.byPriority.NORMAL.retries, 1);
    assert.equal(stats.byPriority.NORMAL.completed, 1);
  }
});

test('retry backoff uses exponential delays (baseRetryDelayMs * 2^attempt) and fails after maxRetries', async () => {
  const delays = [];
  let attempts = 0;

  const rangeCache = {
    async read() {
      attempts += 1;
      throw new Error(`HTTP 502 reading fail.rpf attempt ${attempts}`);
    },
  };

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 2,
    maxRetries: 3,
    baseRetryDelayMs: 25,
    sleep: async (ms) => {
      delays.push(ms);
    },
  });

  await assert.rejects(
    scheduler.scheduleRead('fail.rpf', 0, 64, { priority: IO_PRIORITY.CRITICAL }),
    /HTTP 502 reading fail\.rpf attempt 4/
  );

  // 1 initial attempt + 3 retries = 4 total attempts, delays: 25*2^0, 25*2^1, 25*2^2.
  assert.equal(attempts, 4);
  assert.deepEqual(delays, [25, 50, 100]);

  const stats = scheduler.getStats();
  assert.equal(stats.completed, 0);
  assert.equal(stats.failed, 1);
  assert.equal(stats.retries, 3);
  assert.equal(stats.inFlight, 0);
  assert.equal(stats.byPriority.CRITICAL.failed, 1);
  assert.equal(stats.byPriority.CRITICAL.retries, 3);
});

test('non-retryable errors (404, 403, AbortError, invalid args) fast-fail with zero retries', async () => {
  const delays = [];
  let readCalls = 0;

  const rangeCache = {
    async read(filePath) {
      readCalls += 1;
      if (filePath === 'missing.rpf') {
        throw new Error('HTTP 404 reading missing.rpf[0..15]');
      }
      if (filePath === 'forbidden.rpf') {
        throw Object.assign(new Error('Forbidden'), { status: 403 });
      }
      if (filePath === 'aborted_inside.rpf') {
        const e = new Error('aborted inside fetch');
        e.name = 'AbortError';
        throw e;
      }
      return makeBytes(1, 4);
    },
  };

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxRetries: 3,
    baseRetryDelayMs: 50,
    sleep: async (ms) => {
      delays.push(ms);
    },
  });

  await assert.rejects(scheduler.scheduleRead('missing.rpf', 0, 16), /HTTP 404/);
  await assert.rejects(scheduler.scheduleRead('forbidden.rpf', 0, 16), /Forbidden/);
  await assert.rejects(scheduler.scheduleRead('aborted_inside.rpf', 0, 16), {
    name: 'AbortError',
  });

  // Invalid args fail before ever touching rangeCache.
  await assert.rejects(scheduler.scheduleRead('', 0, 16), TypeError);
  await assert.rejects(scheduler.scheduleRead('../escape.rpf', 0, 16), RangeError);
  await assert.rejects(scheduler.scheduleRead('ok.rpf', -1, 16), RangeError);
  await assert.rejects(scheduler.scheduleRead('ok.rpf', 0, -5), RangeError);
  await assert.rejects(
    scheduler.scheduleRead('ok.rpf', 0, 16, { priority: 99 }),
    RangeError
  );

  assert.equal(readCalls, 3);
  assert.deepEqual(delays, []);

  assert.equal(isRetryableError(new Error('HTTP 404 reading x')), false);
  assert.equal(isRetryableError(new Error('HTTP 500 reading x')), true);
  assert.equal(isRetryableError(new Error('HTTP 429 reading x')), true);

  const stats = scheduler.getStats();
  assert.equal(stats.retries, 0);
  assert.equal(stats.cancelled, 1);
  assert.equal(stats.failed, 7);
});

test('schedulePrefetchList yields to CRITICAL/HIGH reads and cancelPrefetch cancels only queued PREFETCH', async () => {
  const dispatchOrder = [];
  const firstPrefetchGate = deferred();
  const highGate = deferred();

  const rangeCache = {
    read(filePath, offset, length) {
      dispatchOrder.push(`${filePath}@${offset}`);
      if (filePath === 'boot_0.rpf') {
        return firstPrefetchGate.promise;
      }
      if (filePath === 'urgent_high.rpf') {
        return highGate.promise;
      }
      return Promise.resolve(makeBytes(5, length));
    },
  };

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 1,
    sleep: async () => {},
  });

  // Queue a bootset.json-style prefetch list (object form with .entries).
  const prefetchDone = scheduler.schedulePrefetchList({
    entries: [
      { path: 'boot_0.rpf', offset: 0, length: 16 },
      { path: 'boot_1.rpf', offset: 16, length: 16 },
      { path: 'boot_2.rpf', offset: 32, length: 16 },
      { path: 'boot_3.rpf', offset: 48, length: 16 },
    ],
  });

  // boot_0 is in-flight; boot_1..boot_3 are queued as PREFETCH.
  assert.equal(scheduler.getStats().inFlight, 1);
  assert.equal(scheduler.getStats().byPriority.PREFETCH.inFlight, 1);
  assert.equal(scheduler.getStats().byPriority.PREFETCH.queued, 3);

  // Schedule HIGH and CRITICAL reads while prefetch is queued.
  const pHigh = scheduler.scheduleRead('urgent_high.rpf', 0, 16, {
    priority: IO_PRIORITY.HIGH,
  });
  const pCritical = scheduler.scheduleRead('urgent_critical.rpf', 0, 16, {
    priority: IO_PRIORITY.CRITICAL,
  });

  // Cancel queued PREFETCH items — must NOT affect in-flight boot_0 or queued HIGH/CRITICAL.
  const cancelledCount = scheduler.cancelPrefetch();
  assert.equal(cancelledCount, 3);
  assert.equal(scheduler.getStats().byPriority.PREFETCH.queued, 0);
  assert.equal(scheduler.getStats().byPriority.PREFETCH.inFlight, 1);
  assert.equal(scheduler.getStats().byPriority.HIGH.queued, 1);
  assert.equal(scheduler.getStats().byPriority.CRITICAL.queued, 1);

  // Calling cancelPrefetch() again when queue is empty returns 0.
  assert.equal(scheduler.cancelPrefetch(), 0);

  // Finish the already-in-flight boot_0 read -> CRITICAL must run before HIGH!
  firstPrefetchGate.resolve(makeBytes(10, 16));
  highGate.resolve(makeBytes(20, 16));

  const [prefetchSummary] = await Promise.all([prefetchDone, pCritical, pHigh]);

  assert.deepEqual(dispatchOrder, [
    'boot_0.rpf@0',
    'urgent_critical.rpf@0',
    'urgent_high.rpf@0',
  ]);
  assert.deepEqual(prefetchSummary, {
    requested: 4,
    scheduled: 4,
    completed: 1,
    cancelled: 3,
    failed: 0,
  });

  const stats = scheduler.getStats();
  assert.equal(stats.queued, 0);
  assert.equal(stats.inFlight, 0);
  assert.equal(stats.completed, 3);
  assert.equal(stats.cancelled, 3);
  assert.equal(stats.failed, 0);
  assert.equal(stats.retries, 0);
  assert.equal(stats.byPriority.CRITICAL.completed, 1);
  assert.equal(stats.byPriority.HIGH.completed, 1);
  assert.equal(stats.byPriority.PREFETCH.completed, 1);
  assert.equal(stats.byPriority.PREFETCH.cancelled, 3);
});

test('getStats returns accurate snapshots across all lifecycle states and isolates mutations', async () => {
  const gate = deferred();
  let flakeyAttempts = 0;

  const rangeCache = {
    read(filePath, offset, length) {
      if (filePath === 'hold.bin') return gate.promise;
      if (filePath === 'flakey.bin') {
        flakeyAttempts += 1;
        if (flakeyAttempts === 1) {
          return Promise.reject(new Error('HTTP 503 reading flakey.bin[0..3]'));
        }
        return Promise.resolve(makeBytes(3, length));
      }
      if (filePath === 'err404.bin') {
        return Promise.reject(new Error('HTTP 404 reading err404.bin[0..3]'));
      }
      return Promise.resolve(makeBytes(1, length));
    },
  };

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 1,
    maxRetries: 1,
    baseRetryDelayMs: 10,
    sleep: async () => {},
  });

  const pHold = scheduler.scheduleRead('hold.bin', 0, 4, { priority: IO_PRIORITY.CRITICAL });
  const pFlakey = scheduler.scheduleRead('flakey.bin', 0, 4, { priority: IO_PRIORITY.HIGH });
  const p404 = scheduler.scheduleRead('err404.bin', 0, 4, { priority: IO_PRIORITY.NORMAL });
  const pPrefetch = scheduler.scheduleRead('bg.bin', 0, 4, { priority: IO_PRIORITY.PREFETCH });

  const midStats = scheduler.getStats();
  assert.equal(midStats.inFlight, 1);
  assert.equal(midStats.queued, 3);
  assert.equal(midStats.byPriority.CRITICAL.inFlight, 1);
  assert.equal(midStats.byPriority.HIGH.queued, 1);
  assert.equal(midStats.byPriority.NORMAL.queued, 1);
  assert.equal(midStats.byPriority.PREFETCH.queued, 1);

  // Mutating the returned snapshot must not corrupt internal counters.
  midStats.completed = 999;
  midStats.byPriority.CRITICAL.completed = 999;
  assert.equal(scheduler.getStats().completed, 0);

  scheduler.cancelPrefetch();
  gate.resolve(makeBytes(0, 4));

  await pHold;
  await pFlakey;
  await assert.rejects(p404, /HTTP 404/);
  await assert.rejects(pPrefetch, { name: 'AbortError' });

  const finalStats = scheduler.getStats();
  assert.equal(finalStats.queued, 0);
  assert.equal(finalStats.inFlight, 0);
  assert.equal(finalStats.completed, 2);
  assert.equal(finalStats.failed, 1);
  assert.equal(finalStats.cancelled, 1);
  assert.equal(finalStats.retries, 1);

  // Verify per-priority totals sum to top-level totals.
  const prios = Object.values(finalStats.byPriority);
  assert.equal(prios.reduce((s, p) => s + p.queued, 0), finalStats.queued);
  assert.equal(prios.reduce((s, p) => s + p.inFlight, 0), finalStats.inFlight);
  assert.equal(prios.reduce((s, p) => s + p.completed, 0), finalStats.completed);
  assert.equal(prios.reduce((s, p) => s + p.failed, 0), finalStats.failed);
  assert.equal(prios.reduce((s, p) => s + p.cancelled, 0), finalStats.cancelled);
  assert.equal(prios.reduce((s, p) => s + p.retries, 0), finalStats.retries);
});

test('integration with real RangeCache: retries HTTP 503 chunk fetch, fast-fails 404, and prioritizes CRITICAL over prefetch', async () => {
  const files = new Map([
    ['boot_a.rpf', makeBytes(11, 32)],
    ['boot_b.rpf', makeBytes(22, 32)],
    ['critical_hdr.rpf', makeBytes(33, 32)],
  ]);

  const fetchLog = [];
  let transientFailedOnce = false;
  const delays = [];

  async function mockFetch(url, init = {}) {
    const relPath = url.replace(/^.*\/data\//, '');
    fetchLog.push(relPath);

    if (relPath === 'critical_hdr.rpf' && !transientFailedOnce) {
      transientFailedOnce = true;
      return {
        ok: false,
        status: 503,
        headers: new Map(),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }

    const data = files.get(relPath);
    if (!data) {
      return {
        ok: false,
        status: 404,
        headers: new Map(),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }

    const range = init.headers && init.headers.Range;
    const match = /^bytes=(\d+)-(\d+)$/.exec(range || '');
    const start = match ? Number(match[1]) : 0;
    const end = match ? Math.min(Number(match[2]), data.byteLength - 1) : data.byteLength - 1;
    const slice = data.slice(start, end + 1);
    return {
      ok: true,
      status: 206,
      headers: new Map([
        ['Content-Range', `bytes ${start}-${end}/${data.byteLength}`],
        ['Content-Length', String(slice.byteLength)],
      ]),
      async arrayBuffer() {
        return slice.buffer;
      },
    };
  }

  const rangeCache = new RangeCache({
    dataBase: '/data/',
    chunkSize: 16,
    fetchImpl: mockFetch,
  });

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 1,
    maxRetries: 2,
    baseRetryDelayMs: 30,
    sleep: async (ms) => {
      delays.push(ms);
    },
  });

  const prefetchPromise = scheduler.schedulePrefetchList([
    { path: 'boot_a.rpf', offset: 0, length: 16 },
    { path: 'boot_b.rpf', offset: 0, length: 16 },
  ]);

  const criticalPromise = scheduler.scheduleRead('critical_hdr.rpf', 0, 16, {
    priority: IO_PRIORITY.CRITICAL,
  });

  const missingPromise = scheduler.scheduleRead('does_not_exist.rpf', 0, 16, {
    priority: IO_PRIORITY.NORMAL,
  });

  const criticalBytes = await criticalPromise;
  await assert.rejects(missingPromise, /HTTP 404/);

  const summary = await prefetchPromise;

  assert.deepEqual(criticalBytes, files.get('critical_hdr.rpf').slice(0, 16));
  assert.deepEqual(delays, [30]);
  assert.equal(summary.completed, 2);
  // boot_a started first (slot was idle), then CRITICAL jumped ahead of queued NORMAL/PREFETCH
  // and retried once on 503, then NORMAL fast-failed 404, then queued PREFETCH boot_b ran.
  assert.deepEqual(fetchLog, [
    'boot_a.rpf',
    'critical_hdr.rpf',
    'critical_hdr.rpf',
    'does_not_exist.rpf',
    'boot_b.rpf',
  ]);
});

test('AbortSignal while in-flight rejects caller immediately and preserves maxConcurrency until underlying read settles', async () => {
  const inflightGate = deferred();
  const started = [];

  const rangeCache = {
    read(filePath, offset, length) {
      started.push(filePath);
      if (filePath === 'slow.rpf') {
        return inflightGate.promise;
      }
      return Promise.resolve(makeBytes(7, length));
    },
  };

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 1,
    sleep: async () => {},
  });

  const controller = new AbortController();
  const pSlow = scheduler.scheduleRead('slow.rpf', 0, 16, {
    priority: IO_PRIORITY.NORMAL,
    signal: controller.signal,
  });
  const pNext = scheduler.scheduleRead('next.rpf', 0, 16, {
    priority: IO_PRIORITY.CRITICAL,
  });

  controller.abort();
  await assert.rejects(pSlow, { name: 'AbortError' });

  // Underlying rangeCache.read for slow.rpf has not settled yet, so maxConcurrency=1 holds next.rpf queued.
  assert.deepEqual(started, ['slow.rpf']);
  assert.equal(scheduler.getStats().cancelled, 1);
  assert.equal(scheduler.getStats().inFlight, 1);

  // Once the underlying read rejects/resolves, slot is freed and next.rpf dispatches cleanly.
  inflightGate.reject(new Error('HTTP 503 late failure'));
  const nextBytes = await pNext;

  assert.deepEqual(nextBytes, makeBytes(7, 16));
  assert.deepEqual(started, ['slow.rpf', 'next.rpf']);
  const stats = scheduler.getStats();
  assert.equal(stats.completed, 1);
  assert.equal(stats.cancelled, 1);
  assert.equal(stats.failed, 0);
  assert.equal(stats.inFlight, 0);
  assert.equal(stats.queued, 0);
});

test('priority-starvation aging promotes starved low-priority tasks ahead of newer high-priority bursts', async () => {
  let fakeNow = 0;
  const dispatchOrder = [];
  const gates = new Map();

  const rangeCache = {
    read(filePath, offset, length) {
      dispatchOrder.push(filePath);
      const gate = gates.get(filePath);
      if (gate) {
        return gate.promise;
      }
      return Promise.resolve(makeBytes(filePath.charCodeAt(0), length));
    },
  };

  const blockerGate = deferred();
  const prefetchGate = deferred();
  const normalGate = deferred();
  gates.set('blocker.bin', blockerGate);
  gates.set('prefetch-old.bin', prefetchGate);
  gates.set('normal-mid.bin', normalGate);

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 1,
    agingThresholdMs: 100,
    nowFn: () => fakeNow,
    sleep: async () => {},
  });

  assert.equal(scheduler.agingThresholdMs, 100);

  // t = 0: occupy slot and enqueue PREFETCH.
  const pBlocker = scheduler.scheduleRead('blocker.bin', 0, 4, {
    priority: IO_PRIORITY.CRITICAL,
  });
  const pPrefetch = scheduler.scheduleRead('prefetch-old.bin', 0, 4, {
    priority: IO_PRIORITY.PREFETCH,
  });

  // t = 20: enqueue NORMAL.
  fakeNow = 20;
  const pNormal = scheduler.scheduleRead('normal-mid.bin', 0, 4, {
    priority: IO_PRIORITY.NORMAL,
  });

  // t = 110: PREFETCH has waited 110ms (>= 100ms), NORMAL has waited 90ms (< 100ms).
  // Fresh CRITICAL and HIGH requests arrive in a burst.
  fakeNow = 110;
  const pCritical = scheduler.scheduleRead('critical-new.bin', 0, 4, {
    priority: IO_PRIORITY.CRITICAL,
  });
  const pHigh = scheduler.scheduleRead('high-new.bin', 0, 4, {
    priority: IO_PRIORITY.HIGH,
  });

  // Release blocker at t = 110 -> starved PREFETCH promotes ahead of CRITICAL and HIGH.
  blockerGate.resolve(makeBytes(1, 4));
  await pBlocker;
  await Promise.resolve();

  assert.deepEqual(dispatchOrder, ['blocker.bin', 'prefetch-old.bin']);
  assert.equal(scheduler.getStats().agedPromotions, 1);
  assert.equal(scheduler.getStats().byPriority.PREFETCH.agedPromotions, 1);

  // t = 130: NORMAL has now waited 110ms (>= 100ms), CRITICAL/HIGH only 20ms.
  fakeNow = 130;
  prefetchGate.resolve(makeBytes(2, 4));
  await pPrefetch;
  await Promise.resolve();

  assert.deepEqual(dispatchOrder, ['blocker.bin', 'prefetch-old.bin', 'normal-mid.bin']);
  assert.equal(scheduler.getStats().agedPromotions, 2);
  assert.equal(scheduler.getStats().byPriority.NORMAL.agedPromotions, 1);

  // t = 140: remaining CRITICAL (wait 30ms) and HIGH (wait 30ms) are not starved -> strict priority order.
  fakeNow = 140;
  normalGate.resolve(makeBytes(3, 4));
  await Promise.all([pNormal, pCritical, pHigh]);

  assert.deepEqual(dispatchOrder, [
    'blocker.bin',
    'prefetch-old.bin',
    'normal-mid.bin',
    'critical-new.bin',
    'high-new.bin',
  ]);

  const stats = scheduler.getStats();
  assert.equal(stats.agedPromotions, 2);
  assert.equal(stats.byPriority.CRITICAL.agedPromotions, 0);
  assert.equal(stats.byPriority.HIGH.agedPromotions, 0);
  assert.equal(stats.byPriority.NORMAL.agedPromotions, 1);
  assert.equal(stats.byPriority.PREFETCH.agedPromotions, 1);
  assert.equal(stats.maxQueueWaitMs, 110);
  assert.equal(stats.avgQueueWaitMsByPriority.CRITICAL, 15); // (0 + 30) / 2
  assert.equal(stats.avgQueueWaitMsByPriority.HIGH, 30);
  assert.equal(stats.avgQueueWaitMsByPriority.NORMAL, 110);
  assert.equal(stats.avgQueueWaitMsByPriority.PREFETCH, 110);
});

test('aging selects oldest starved task first and preserves priority on equal age or solo queue', async () => {
  let fakeNow = 0;
  const dispatchOrder = [];
  const blockerGate = deferred();
  const prefetchGate = deferred();
  const normalGate = deferred();

  const rangeCache = {
    read(filePath, offset, length) {
      dispatchOrder.push(filePath);
      if (filePath === 'blocker.bin') return blockerGate.promise;
      if (filePath === 'prefetch-oldest.bin') return prefetchGate.promise;
      if (filePath === 'normal-older.bin') return normalGate.promise;
      return Promise.resolve(makeBytes(9, length));
    },
  };

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 1,
    agingThresholdMs: 100,
    nowFn: () => fakeNow,
    sleep: async () => {},
  });

  // Case 1: Two starved tasks at different enqueue times -> oldest starved runs first.
  const pBlocker = scheduler.scheduleRead('blocker.bin', 0, 4, {
    priority: IO_PRIORITY.CRITICAL,
  });

  fakeNow = 10;
  const pPrefetchOldest = scheduler.scheduleRead('prefetch-oldest.bin', 0, 4, {
    priority: IO_PRIORITY.PREFETCH,
  });

  fakeNow = 30;
  const pNormalOlder = scheduler.scheduleRead('normal-older.bin', 0, 4, {
    priority: IO_PRIORITY.NORMAL,
  });

  fakeNow = 150;
  const pHighFresh = scheduler.scheduleRead('high-fresh.bin', 0, 4, {
    priority: IO_PRIORITY.HIGH,
  });

  blockerGate.resolve(makeBytes(1, 4));
  await pBlocker;
  await Promise.resolve();
  assert.deepEqual(dispatchOrder, ['blocker.bin', 'prefetch-oldest.bin']);

  prefetchGate.resolve(makeBytes(2, 4));
  await pPrefetchOldest;
  await Promise.resolve();
  assert.deepEqual(dispatchOrder, ['blocker.bin', 'prefetch-oldest.bin', 'normal-older.bin']);

  normalGate.resolve(makeBytes(3, 4));
  await Promise.all([pNormalOlder, pHighFresh]);
  assert.deepEqual(dispatchOrder, [
    'blocker.bin',
    'prefetch-oldest.bin',
    'normal-older.bin',
    'high-fresh.bin',
  ]);
  assert.equal(scheduler.getStats().agedPromotions, 2);

  // Case 2: Equal-age starved tasks prefer higher priority and do not falsely count promotion.
  scheduler.reset();
  dispatchOrder.length = 0;
  fakeNow = 0;
  const gate2 = deferred();
  const rangeCache2 = {
    read(filePath, offset, length) {
      dispatchOrder.push(filePath);
      if (filePath === 'hold2.bin') return gate2.promise;
      return Promise.resolve(makeBytes(4, length));
    },
  };
  const scheduler2 = new IoPriorityScheduler({
    rangeCache: rangeCache2,
    maxConcurrency: 1,
    agingThresholdMs: 100,
    nowFn: () => fakeNow,
    sleep: async () => {},
  });

  const pHold2 = scheduler2.scheduleRead('hold2.bin', 0, 4, { priority: IO_PRIORITY.NORMAL });
  const pPrefEq = scheduler2.scheduleRead('pref-eq.bin', 0, 4, { priority: IO_PRIORITY.PREFETCH });
  const pCritEq = scheduler2.scheduleRead('crit-eq.bin', 0, 4, { priority: IO_PRIORITY.CRITICAL });

  fakeNow = 200;
  gate2.resolve(makeBytes(1, 4));
  await Promise.all([pHold2, pPrefEq, pCritEq]);

  // Equal enqueue timestamp -> CRITICAL runs before PREFETCH; then PREFETCH is the only remaining queue item.
  assert.deepEqual(dispatchOrder, ['hold2.bin', 'crit-eq.bin', 'pref-eq.bin']);
  assert.equal(scheduler2.getStats().agedPromotions, 0);
});

test('agingThresholdMs: 0 (and default) preserves strict priority ordering regardless of wait time', async () => {
  for (const agingOption of [undefined, 0]) {
    let fakeNow = 0;
    const dispatchOrder = [];
    const blockerGate = deferred();

    const rangeCache = {
      read(filePath, offset, length) {
        dispatchOrder.push(filePath);
        if (filePath === 'blocker.bin') {
          return blockerGate.promise;
        }
        return Promise.resolve(makeBytes(5, length));
      },
    };

    const opts = {
      rangeCache,
      maxConcurrency: 1,
      agingWindowMs: 0,
      nowFn: () => fakeNow,
      sleep: async () => {},
    };
    if (agingOption !== undefined) {
      opts.agingThresholdMs = agingOption;
    }

    const scheduler = new IoPriorityScheduler(opts);
    assert.equal(scheduler.agingThresholdMs, 0);

    const pBlocker = scheduler.scheduleRead('blocker.bin', 0, 4, {
      priority: IO_PRIORITY.NORMAL,
    });
    const pPrefetch = scheduler.scheduleRead('prefetch-waited.bin', 0, 4, {
      priority: IO_PRIORITY.PREFETCH,
    });
    const pNormal = scheduler.scheduleRead('normal-waited.bin', 0, 4, {
      priority: IO_PRIORITY.NORMAL,
    });

    // Advance fake clock far ahead before high-priority tasks arrive.
    fakeNow = 10000;
    const pHigh = scheduler.scheduleRead('high-new.bin', 0, 4, {
      priority: IO_PRIORITY.HIGH,
    });
    const pCritical = scheduler.scheduleRead('critical-new.bin', 0, 4, {
      priority: IO_PRIORITY.CRITICAL,
    });

    blockerGate.resolve(makeBytes(1, 4));
    await Promise.all([pBlocker, pPrefetch, pNormal, pHigh, pCritical]);

    assert.deepEqual(dispatchOrder, [
      'blocker.bin',
      'critical-new.bin',
      'high-new.bin',
      'normal-waited.bin',
      'prefetch-waited.bin',
    ]);

    const stats = scheduler.getStats();
    assert.equal(stats.agedPromotions, 0);
    assert.equal(stats.maxQueueWaitMs, 10000);
    assert.equal(stats.avgQueueWaitMsByPriority.PREFETCH, 10000);
    assert.equal(stats.avgQueueWaitMsByPriority.NORMAL, 5000); // (0 + 10000) / 2
    assert.equal(stats.avgQueueWaitMsByPriority.HIGH, 0);
    assert.equal(stats.avgQueueWaitMsByPriority.CRITICAL, 0);
  }
});

test('queue-wait telemetry computes exact deterministic per-priority averages, max, and ignores aborted queued tasks', async () => {
  let fakeNow = 0;
  const gates = new Map();

  const rangeCache = {
    read(filePath, offset, length) {
      const gate = gates.get(filePath);
      if (gate) {
        return gate.promise;
      }
      return Promise.resolve(makeBytes(7, length));
    },
  };

  const gBlocker = deferred();
  const gCrit2 = deferred();
  const gHigh1 = deferred();
  const gHigh2 = deferred();
  gates.set('crit-1.bin', gBlocker);
  gates.set('crit-2.bin', gCrit2);
  gates.set('high-1.bin', gHigh1);
  gates.set('high-2.bin', gHigh2);

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 1,
    agingThresholdMs: 0,
    nowFn: () => fakeNow,
    sleep: async () => {},
  });

  // t = 0: crit-1 dispatches immediately (wait = 0ms).
  const pCrit1 = scheduler.scheduleRead('crit-1.bin', 0, 4, { priority: IO_PRIORITY.CRITICAL });
  // t = 0: prefetch-1 queued.
  const pPrefetch1 = scheduler.scheduleRead('prefetch-1.bin', 0, 4, {
    priority: IO_PRIORITY.PREFETCH,
  });

  // t = 10: crit-2 queued.
  fakeNow = 10;
  const pCrit2 = scheduler.scheduleRead('crit-2.bin', 0, 4, { priority: IO_PRIORITY.CRITICAL });

  // t = 20: high-1 queued, plus a queued NORMAL read that will be aborted before dispatch.
  fakeNow = 20;
  const pHigh1 = scheduler.scheduleRead('high-1.bin', 0, 4, { priority: IO_PRIORITY.HIGH });
  const abortController = new AbortController();
  const pAbortedNormal = scheduler.scheduleRead('aborted-normal.bin', 0, 4, {
    priority: IO_PRIORITY.NORMAL,
    signal: abortController.signal,
  });

  // t = 30: high-2 queued.
  fakeNow = 30;
  const pHigh2 = scheduler.scheduleRead('high-2.bin', 0, 4, { priority: IO_PRIORITY.HIGH });

  // Abort the queued NORMAL read at t = 45 before it ever dispatches.
  fakeNow = 45;
  abortController.abort();
  await assert.rejects(pAbortedNormal, { name: 'AbortError' });

  // t = 50: crit-1 finishes -> crit-2 dispatches (wait = 50 - 10 = 40ms).
  fakeNow = 50;
  gBlocker.resolve(makeBytes(1, 4));
  await pCrit1;
  await Promise.resolve();

  // t = 70: crit-2 finishes -> high-1 dispatches (wait = 70 - 20 = 50ms).
  fakeNow = 70;
  gCrit2.resolve(makeBytes(2, 4));
  await pCrit2;
  await Promise.resolve();

  // t = 100: high-1 finishes -> high-2 dispatches (wait = 100 - 30 = 70ms).
  fakeNow = 100;
  gHigh1.resolve(makeBytes(3, 4));
  await pHigh1;
  await Promise.resolve();

  // t = 120: high-2 finishes -> prefetch-1 dispatches (wait = 120 - 0 = 120ms).
  fakeNow = 120;
  gHigh2.resolve(makeBytes(4, 4));
  await Promise.all([pHigh2, pPrefetch1]);

  const stats = scheduler.getStats();
  assert.deepEqual(stats.avgQueueWaitMsByPriority, {
    CRITICAL: 20, // (0 + 40) / 2
    HIGH: 60, // (50 + 70) / 2
    NORMAL: 0, // aborted task was never dispatched
    PREFETCH: 120, // 120 / 1
  });
  assert.equal(stats.maxQueueWaitMs, 120);
  assert.equal(stats.agedPromotions, 0);
  assert.equal(stats.byPriority.CRITICAL.maxQueueWaitMs, 40);
  assert.equal(stats.byPriority.HIGH.maxQueueWaitMs, 70);
  assert.equal(stats.byPriority.NORMAL.dispatched, 0);
  assert.equal(stats.byPriority.PREFETCH.maxQueueWaitMs, 120);

  // Snapshot isolation: mutating avgQueueWaitMsByPriority must not affect subsequent getStats().
  stats.avgQueueWaitMsByPriority.CRITICAL = 9999;
  assert.equal(scheduler.getStats().avgQueueWaitMsByPriority.CRITICAL, 20);
});

test('reset() and clear() cleanly reset telemetry counters and cancel queued tasks safely', async () => {
  let fakeNow = 0;
  let blockerGate = deferred();

  const rangeCache = {
    read(filePath, offset, length) {
      if (filePath.startsWith('blocker')) {
        return blockerGate.promise;
      }
      return Promise.resolve(makeBytes(6, length));
    },
  };

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 1,
    agingThresholdMs: 50,
    nowFn: () => fakeNow,
    sleep: async () => {},
  });

  // Accumulate completions, wait latency, and an aged promotion.
  const pB1 = scheduler.scheduleRead('blocker-1.bin', 0, 4, { priority: IO_PRIORITY.NORMAL });
  const pPref = scheduler.scheduleRead('pref-1.bin', 0, 4, { priority: IO_PRIORITY.PREFETCH });
  fakeNow = 80;
  const pCrit = scheduler.scheduleRead('crit-1.bin', 0, 4, { priority: IO_PRIORITY.CRITICAL });

  blockerGate.resolve(makeBytes(1, 4));
  await Promise.all([pB1, pPref, pCrit]);

  const beforeReset = scheduler.getStats();
  assert.equal(beforeReset.completed, 3);
  assert.equal(beforeReset.agedPromotions, 1);
  assert.equal(beforeReset.maxQueueWaitMs, 80);
  assert.equal(beforeReset.avgQueueWaitMsByPriority.PREFETCH, 80);

  // 1. reset() while idle zeroes all counters and telemetry.
  assert.equal(scheduler.reset(), 0);
  const afterReset = scheduler.getStats();
  assert.equal(afterReset.queued, 0);
  assert.equal(afterReset.inFlight, 0);
  assert.equal(afterReset.completed, 0);
  assert.equal(afterReset.failed, 0);
  assert.equal(afterReset.cancelled, 0);
  assert.equal(afterReset.retries, 0);
  assert.equal(afterReset.agedPromotions, 0);
  assert.equal(afterReset.maxQueueWaitMs, 0);
  assert.deepEqual(afterReset.avgQueueWaitMsByPriority, {
    CRITICAL: 0,
    HIGH: 0,
    NORMAL: 0,
    PREFETCH: 0,
  });

  // 2. clear() with queued tasks rejects queued tasks with AbortError, zeroes telemetry,
  //    and preserves inFlight tracking until the active read settles.
  blockerGate = deferred();
  fakeNow = 100;
  const pB2 = scheduler.scheduleRead('blocker-2.bin', 0, 4, { priority: IO_PRIORITY.HIGH });
  const pQueued1 = scheduler.scheduleRead('queued-1.bin', 0, 4, { priority: IO_PRIORITY.NORMAL });
  const pQueued2 = scheduler.scheduleRead('queued-2.bin', 0, 4, { priority: IO_PRIORITY.PREFETCH });

  const clearedCount = scheduler.clear();
  assert.equal(clearedCount, 2);
  await assert.rejects(pQueued1, { name: 'AbortError' });
  await assert.rejects(pQueued2, { name: 'AbortError' });

  const midClearStats = scheduler.getStats();
  assert.equal(midClearStats.queued, 0);
  assert.equal(midClearStats.inFlight, 1);
  assert.equal(midClearStats.agedPromotions, 0);
  assert.equal(midClearStats.maxQueueWaitMs, 0);
  assert.deepEqual(midClearStats.avgQueueWaitMsByPriority, {
    CRITICAL: 0,
    HIGH: 0,
    NORMAL: 0,
    PREFETCH: 0,
  });

  blockerGate.resolve(makeBytes(2, 4));
  await pB2;
  assert.equal(scheduler.getStats().inFlight, 0);

  // Subsequent reads after clear() work normally and report fresh metrics.
  scheduler.reset();
  fakeNow = 200;
  const freshBytes = await scheduler.scheduleRead('fresh.bin', 0, 4, {
    priority: IO_PRIORITY.NORMAL,
  });
  assert.deepEqual(freshBytes, makeBytes(6, 4));
  assert.equal(scheduler.getStats().completed, 1);
  assert.equal(scheduler.getStats().maxQueueWaitMs, 0);
  assert.equal(scheduler.getStats().avgQueueWaitMsByPriority.NORMAL, 0);
});

// ---------------------------------------------------------------------------
// W188: Automatic path-based priority classification, Bink2/shader/UI reads,
//       telemetry (priorityCounts, binkVideoReads, shaderPackReads, promotions),
//       and anti-starvation aging under media/shader bursts.
// ---------------------------------------------------------------------------

test('classifyIoPriority is exported in CommonJS and Worker globalThis', () => {
  assert.equal(typeof classifyIoPriority, 'function');
  assert.equal(typeof isBinkVideoPath, 'function');
  assert.equal(typeof isShaderPackPath, 'function');
  assert.equal(typeof isUiFontRpfPath, 'function');

  const src = fs.readFileSync(path.join(repoRoot, 'web/workers/io_scheduler.js'), 'utf8');
  const sandbox = {};
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;
  vm.runInNewContext(src, sandbox);

  assert.equal(typeof sandbox.PlayGta6IoScheduler.classifyIoPriority, 'function');
  assert.equal(
    sandbox.PlayGta6IoScheduler.classifyIoPriority('shaders/main.wgsl.pack'),
    IO_PRIORITY.CRITICAL
  );
  assert.equal(
    sandbox.PlayGta6IoScheduler.classifyIoPriority('movies/intro.bk2'),
    IO_PRIORITY.HIGH
  );
});

test('classifyIoPriority maps shader packs, boot metadata, Bink2, UI/font RPF, general RPF/TTPF, and prefetch paths', () => {
  // CRITICAL: shader packs, reflection JSON, boot-options.json, guest .pgim images
  assert.equal(classifyIoPriority('shaders/frame1.wgsl.pack'), IO_PRIORITY.CRITICAL);
  assert.equal(classifyIoPriority('SHADERS/MAIN.WGSL.PACK?v=2#frag'), IO_PRIORITY.CRITICAL);
  assert.equal(classifyIoPriority('shaders/frame1.reflect.json'), IO_PRIORITY.CRITICAL);
  assert.equal(classifyIoPriority('shaders/main.wgsl.pack.reflect.json'), IO_PRIORITY.CRITICAL);
  assert.equal(classifyIoPriority('boot-options.json'), IO_PRIORITY.CRITICAL);
  assert.equal(classifyIoPriority('config/boot-options.json'), IO_PRIORITY.CRITICAL);
  assert.equal(classifyIoPriority('guest\\game_main.pgim'), IO_PRIORITY.CRITICAL);

  // HIGH: Bink2 video streams (.bk2 / .bik) and UI/font .rpf / .ttpf / font assets
  assert.equal(classifyIoPriority('movies/intro_logo.bk2'), IO_PRIORITY.HIGH);
  assert.equal(classifyIoPriority('movies/INTRO.BIK?range=1'), IO_PRIORITY.HIGH);
  assert.equal(classifyIoPriority('x64/data/ui/fonts.rpf'), IO_PRIORITY.HIGH);
  assert.equal(classifyIoPriority('common/ui.rpf'), IO_PRIORITY.HIGH);
  assert.equal(classifyIoPriority('x64/data/scaleform_generic.rpf'), IO_PRIORITY.HIGH);
  assert.equal(classifyIoPriority('x64/data/hud_icons.ttpf'), IO_PRIORITY.HIGH);
  assert.equal(classifyIoPriority('x64/data/frontend_menu.rpf'), IO_PRIORITY.HIGH);
  assert.equal(classifyIoPriority('x64/data/splash_screen.rpf'), IO_PRIORITY.HIGH);
  assert.equal(classifyIoPriority('rpf://x64/data/ui/fonts.rpf/font_lib.gfx'), IO_PRIORITY.HIGH);
  assert.equal(classifyIoPriority('fonts/ui_sans.ttf'), IO_PRIORITY.HIGH);

  // NORMAL: general on-demand .rpf / .ttpf reads (including false-positive substring guards)
  assert.equal(classifyIoPriority('x64/levels/world.rpf'), IO_PRIORITY.NORMAL);
  assert.equal(classifyIoPriority('x64/textures/terrain.ttpf'), IO_PRIORITY.NORMAL);
  assert.equal(classifyIoPriority('x64/build_assets.rpf'), IO_PRIORITY.NORMAL);
  assert.equal(classifyIoPriority('x64/liquid_sim.rpf'), IO_PRIORITY.NORMAL);
  assert.equal(classifyIoPriority('x64/fruit_props.rpf'), IO_PRIORITY.NORMAL);
  assert.equal(classifyIoPriority('misc/data.bin'), IO_PRIORITY.NORMAL);
  assert.equal(classifyIoPriority(''), IO_PRIORITY.NORMAL);
  assert.equal(classifyIoPriority(null), IO_PRIORITY.NORMAL);
  assert.equal(classifyIoPriority(undefined), IO_PRIORITY.NORMAL);

  // PREFETCH (Low): background speculative prefetch paths
  assert.equal(classifyIoPriority('prefetch/world_lod.rpf'), IO_PRIORITY.PREFETCH);
  assert.equal(classifyIoPriority('x64/speculative/chunk_01.rpf'), IO_PRIORITY.PREFETCH);
  assert.equal(classifyIoPriority('chunk_02.rpf.prefetch'), IO_PRIORITY.PREFETCH);

  // Helper predicates
  assert.equal(isBinkVideoPath('movies/intro.bk2'), true);
  assert.equal(isBinkVideoPath('movies/logo.BIK'), true);
  assert.equal(isBinkVideoPath('movies/intro.mp4'), false);
  assert.equal(isShaderPackPath('shaders/main.wgsl.pack'), true);
  assert.equal(isShaderPackPath('shaders/main.wgsl.pack.reflect.json'), false);
  assert.equal(isUiFontRpfPath('x64/data/ui/fonts.rpf'), true);
  assert.equal(isUiFontRpfPath('x64/levels/world.rpf'), false);
});

test('classifyIoPriority explicit caller priority always overrides auto-classification', () => {
  // Demote CRITICAL/HIGH assets explicitly to PREFETCH or NORMAL.
  assert.equal(
    classifyIoPriority('shaders/main.wgsl.pack', IO_PRIORITY.PREFETCH),
    IO_PRIORITY.PREFETCH
  );
  assert.equal(
    classifyIoPriority('boot-options.json', IO_PRIORITY.NORMAL),
    IO_PRIORITY.NORMAL
  );
  assert.equal(
    classifyIoPriority('movies/intro.bk2', IO_PRIORITY.PREFETCH),
    IO_PRIORITY.PREFETCH
  );
  assert.equal(
    classifyIoPriority('x64/data/ui/fonts.rpf', IO_PRIORITY.NORMAL),
    IO_PRIORITY.NORMAL
  );

  // Promote NORMAL/PREFETCH assets explicitly to CRITICAL or HIGH.
  assert.equal(
    classifyIoPriority('x64/levels/world.rpf', IO_PRIORITY.CRITICAL),
    IO_PRIORITY.CRITICAL
  );
  assert.equal(
    classifyIoPriority('prefetch/world_lod.rpf', IO_PRIORITY.HIGH),
    IO_PRIORITY.HIGH
  );
  assert.equal(
    classifyIoPriority('movies/intro.bk2', IO_PRIORITY.CRITICAL),
    IO_PRIORITY.CRITICAL
  );

  // Undefined/null explicit priority falls back to auto-classification.
  assert.equal(classifyIoPriority('movies/intro.bk2', undefined), IO_PRIORITY.HIGH);
  assert.equal(classifyIoPriority('shaders/main.wgsl.pack', null), IO_PRIORITY.CRITICAL);
});

test('auto-classified shader packs, boot metadata, Bink2 video, and UI/font RPF preempt general RPF and prefetch traffic', async () => {
  const dispatchOrder = [];
  const blockerGate = deferred();

  const rangeCache = {
    read(filePath, offset, length) {
      dispatchOrder.push(filePath);
      if (filePath === 'blocker.bin') {
        return blockerGate.promise;
      }
      return Promise.resolve(makeBytes(filePath.charCodeAt(0), length));
    },
  };

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 1,
    sleep: async () => {},
  });

  // Occupy single slot so all subsequent reads queue up.
  const pBlocker = scheduler.scheduleRead('blocker.bin', 0, 16, {
    priority: IO_PRIORITY.PREFETCH,
  });

  // Queue background prefetch, normal RPF/TTPF, Bink2 video, UI/font RPF, and critical shader/boot assets
  // WITHOUT explicit priorities (relying on auto-classification, plus schedulePrefetchList/prefetch flag).
  const pPrefetchList = scheduler.schedulePrefetchList([
    { path: 'x64/levels/bg_chunk_1.rpf', offset: 0, length: 16 },
    { path: 'x64/levels/bg_chunk_2.rpf', offset: 16, length: 16 },
  ]);
  const pPrefetchPath = scheduler.scheduleRead('prefetch/world_lod.rpf', 0, 16);
  const pPrefetchFlag = scheduler.scheduleRead('x64/levels/spec_world.rpf', 0, 16, {
    prefetch: true,
  });
  const pNormalRpf = scheduler.scheduleRead('x64/levels/world.rpf', 0, 16);
  const pNormalTtpf = scheduler.scheduleRead('x64/textures/terrain.ttpf', 0, 16);
  const pBink2 = scheduler.scheduleRead('movies/intro.bk2', 0, 64);
  const pBink1 = scheduler.scheduleRead('movies/logo.bik', 0, 64);
  const pUiFont = scheduler.scheduleRead('x64/data/ui/fonts.rpf', 0, 32);
  const pShaderPack = scheduler.scheduleRead('shaders/frame1.wgsl.pack', 0, 128);
  const pReflect = scheduler.scheduleRead('shaders/frame1.reflect.json', 0, 32);
  const pBootOpts = scheduler.scheduleRead('boot-options.json', 0, 32);
  const pPgim = scheduler.scheduleRead('guest/main.pgim', 0, 64);

  blockerGate.resolve(makeBytes(1, 16));

  await Promise.all([
    pBlocker,
    pPrefetchList,
    pPrefetchPath,
    pPrefetchFlag,
    pNormalRpf,
    pNormalTtpf,
    pBink2,
    pBink1,
    pUiFont,
    pShaderPack,
    pReflect,
    pBootOpts,
    pPgim,
  ]);

  assert.deepEqual(dispatchOrder, [
    'blocker.bin',
    // CRITICAL (FIFO):
    'shaders/frame1.wgsl.pack',
    'shaders/frame1.reflect.json',
    'boot-options.json',
    'guest/main.pgim',
    // HIGH (FIFO):
    'movies/intro.bk2',
    'movies/logo.bik',
    'x64/data/ui/fonts.rpf',
    // NORMAL (FIFO):
    'x64/levels/world.rpf',
    'x64/textures/terrain.ttpf',
    // PREFETCH (FIFO):
    'x64/levels/bg_chunk_1.rpf',
    'x64/levels/bg_chunk_2.rpf',
    'prefetch/world_lod.rpf',
    'x64/levels/spec_world.rpf',
  ]);

  const stats = scheduler.getStats();
  assert.deepEqual(stats.priorityCounts, {
    CRITICAL: 4,
    HIGH: 3,
    NORMAL: 2,
    PREFETCH: 5,
  });
  assert.equal(stats.binkVideoReads, 2);
  assert.equal(stats.shaderPackReads, 1);
  assert.equal(stats.uiFontReads, 1);
  assert.equal(stats.promotions, 0);
  assert.equal(stats.agedPromotions, 0);
  assert.equal(stats.byPriority.CRITICAL.scheduled, 4);
  assert.equal(stats.byPriority.HIGH.scheduled, 3);
  assert.equal(stats.byPriority.NORMAL.scheduled, 2);
  assert.equal(stats.byPriority.PREFETCH.scheduled, 5);
});

test('explicit caller priority overrides auto-classification in scheduleRead and schedulePrefetchList', async () => {
  const dispatchOrder = [];
  const blockerGate = deferred();

  const rangeCache = {
    read(filePath, offset, length) {
      dispatchOrder.push(filePath);
      if (filePath === 'blocker.bin') {
        return blockerGate.promise;
      }
      return Promise.resolve(makeBytes(3, length));
    },
  };

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 1,
    sleep: async () => {},
  });

  const pBlocker = scheduler.scheduleRead('blocker.bin', 0, 8, {
    priority: IO_PRIORITY.NORMAL,
  });

  // Explicitly schedule a .wgsl.pack and .bk2 as PREFETCH via schedulePrefetchList.
  const pPrefetchMedia = scheduler.schedulePrefetchList([
    { path: 'shaders/background.wgsl.pack', offset: 0, length: 16 },
    { path: 'movies/attract.bk2', offset: 0, length: 16 },
  ]);

  // Explicitly demote a .bk2 to NORMAL and promote a general .rpf to CRITICAL.
  const pDemotedBink = scheduler.scheduleRead('movies/low_prio.bk2', 0, 16, {
    priority: IO_PRIORITY.NORMAL,
  });
  const pAutoUi = scheduler.scheduleRead('x64/data/ui/hud.rpf', 0, 16);
  const pPromotedRpf = scheduler.scheduleRead('x64/levels/world.rpf', 0, 16, {
    priority: IO_PRIORITY.CRITICAL,
  });

  // cancelPrefetch() must cancel the prefetched .wgsl.pack and .bk2 cleanly.
  assert.equal(scheduler.cancelPrefetch(), 2);

  blockerGate.resolve(makeBytes(1, 8));
  const [prefetchSummary] = await Promise.all([
    pPrefetchMedia,
    pBlocker,
    pDemotedBink,
    pAutoUi,
    pPromotedRpf,
  ]);

  assert.equal(prefetchSummary.cancelled, 2);
  assert.deepEqual(dispatchOrder, [
    'blocker.bin',
    'x64/levels/world.rpf', // explicit CRITICAL
    'x64/data/ui/hud.rpf', // auto HIGH
    'movies/low_prio.bk2', // explicit NORMAL
  ]);

  const stats = scheduler.getStats();
  assert.deepEqual(stats.priorityCounts, {
    CRITICAL: 1,
    HIGH: 1,
    NORMAL: 2,
    PREFETCH: 2,
  });
  assert.equal(stats.binkVideoReads, 2);
  assert.equal(stats.shaderPackReads, 1);
  assert.equal(stats.uiFontReads, 1);
});

test('anti-starvation aging drains low-priority prefetch reads during continuous Bink2 and shader-pack bursts', async () => {
  let fakeNow = 0;
  const dispatchOrder = [];
  const gates = new Map();

  const rangeCache = {
    read(filePath, offset, length) {
      dispatchOrder.push(filePath);
      const gate = gates.get(filePath);
      if (gate) {
        return gate.promise;
      }
      return Promise.resolve(makeBytes(5, length));
    },
  };

  const blockerGate = deferred();
  const prefetchGate = deferred();
  const binkGate = deferred();
  gates.set('shaders/init.wgsl.pack', blockerGate);
  gates.set('prefetch/bg_lod.rpf', prefetchGate);
  gates.set('movies/stream_01.bk2', binkGate);

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 1,
    agingThresholdMs: 100,
    nowFn: () => fakeNow,
    sleep: async () => {},
  });

  // t = 0: shader pack occupies the slot; background prefetch queues up.
  const pInitPack = scheduler.scheduleRead('shaders/init.wgsl.pack', 0, 64);
  const pBgPrefetch = scheduler.scheduleRead('prefetch/bg_lod.rpf', 0, 64);

  // t = 60: Bink2 stream and another shader pack arrive; prefetch wait is 60ms (< 100ms).
  fakeNow = 60;
  const pShaderBurst = scheduler.scheduleRead('shaders/frame2.wgsl.pack', 0, 64);
  const pBinkStream = scheduler.scheduleRead('movies/stream_01.bk2', 0, 64);

  // t = 110: initial shader pack completes -> prefetch waited 110ms (>= 100ms),
  // so anti-starvation promotes it ahead of the newer CRITICAL shader pack and HIGH Bink2 read!
  fakeNow = 110;
  blockerGate.resolve(makeBytes(1, 64));
  await pInitPack;
  await Promise.resolve();

  assert.deepEqual(dispatchOrder, ['shaders/init.wgsl.pack', 'prefetch/bg_lod.rpf']);
  const midStats = scheduler.getStats();
  assert.equal(midStats.promotions, 1);
  assert.equal(midStats.agedPromotions, 1);
  assert.equal(midStats.byPriority.PREFETCH.promotions, 1);

  // t = 120: prefetch finishes -> shader pack (CRITICAL, wait 60ms) runs before Bink2 (HIGH, wait 60ms).
  fakeNow = 120;
  prefetchGate.resolve(makeBytes(2, 64));
  await pBgPrefetch;
  await pShaderBurst;
  await Promise.resolve();

  binkGate.resolve(makeBytes(3, 64));
  await pBinkStream;

  assert.deepEqual(dispatchOrder, [
    'shaders/init.wgsl.pack',
    'prefetch/bg_lod.rpf',
    'shaders/frame2.wgsl.pack',
    'movies/stream_01.bk2',
  ]);

  const finalStats = scheduler.getStats();
  assert.equal(finalStats.promotions, 1);
  assert.equal(finalStats.agedPromotions, 1);
  assert.equal(finalStats.shaderPackReads, 2);
  assert.equal(finalStats.binkVideoReads, 1);
  assert.deepEqual(finalStats.priorityCounts, {
    CRITICAL: 2,
    HIGH: 1,
    NORMAL: 0,
    PREFETCH: 1,
  });
});

test('telemetry counters isolate snapshot mutations, exclude pre-aborted/invalid reads, and reset cleanly', async () => {
  const rangeCache = {
    read(filePath, offset, length) {
      return Promise.resolve(makeBytes(9, length));
    },
  };

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 2,
    sleep: async () => {},
  });

  // Pre-aborted and invalid reads must not increment priorityCounts / media counters.
  const preAborted = new AbortController();
  preAborted.abort();
  await assert.rejects(
    scheduler.scheduleRead('movies/aborted.bk2', 0, 16, { signal: preAborted.signal }),
    { name: 'AbortError' }
  );
  await assert.rejects(scheduler.scheduleRead('shaders/bad.wgsl.pack', -1, 16), RangeError);

  assert.deepEqual(scheduler.getStats().priorityCounts, {
    CRITICAL: 0,
    HIGH: 0,
    NORMAL: 0,
    PREFETCH: 0,
  });
  assert.equal(scheduler.getStats().binkVideoReads, 0);
  assert.equal(scheduler.getStats().shaderPackReads, 0);

  // Schedule valid reads and verify snapshot isolation.
  await Promise.all([
    scheduler.scheduleRead('shaders/main.wgsl.pack', 0, 32),
    scheduler.scheduleRead('movies/cutscene.bk2', 0, 32),
    scheduler.scheduleRead('x64/data/ui/fonts.rpf', 0, 32),
  ]);

  const snap = scheduler.getStats();
  assert.deepEqual(snap.priorityCounts, {
    CRITICAL: 1,
    HIGH: 2,
    NORMAL: 0,
    PREFETCH: 0,
  });
  assert.equal(snap.shaderPackReads, 1);
  assert.equal(snap.binkVideoReads, 1);
  assert.equal(snap.uiFontReads, 1);

  snap.priorityCounts.CRITICAL = 999;
  snap.binkVideoReads = 999;
  snap.shaderPackReads = 999;
  snap.promotions = 999;

  const fresh = scheduler.getStats();
  assert.equal(fresh.priorityCounts.CRITICAL, 1);
  assert.equal(fresh.binkVideoReads, 1);
  assert.equal(fresh.shaderPackReads, 1);
  assert.equal(fresh.promotions, 0);

  // reset() zeroes all new telemetry counters.
  scheduler.reset();
  const afterReset = scheduler.getStats();
  assert.deepEqual(afterReset.priorityCounts, {
    CRITICAL: 0,
    HIGH: 0,
    NORMAL: 0,
    PREFETCH: 0,
  });
  assert.equal(afterReset.binkVideoReads, 0);
  assert.equal(afterReset.shaderPackReads, 0);
  assert.equal(afterReset.uiFontReads, 0);
  assert.equal(afterReset.promotions, 0);
});

// ---------------------------------------------------------------------------
// W206: Priority aging (starvation prevention), in-flight & queued range
//       deduplication, cancellation/error cleanup, and top-level telemetry.
// ---------------------------------------------------------------------------

test('W206 exports DEFAULT_AGING_INTERVAL_MS, DEFAULT_DEDUPE_IN_FLIGHT, and makeReadKey in CJS and Worker globalThis', () => {
  assert.equal(DEFAULT_AGING_INTERVAL_MS, 0);
  assert.equal(DEFAULT_DEDUPE_IN_FLIGHT, true);
  assert.equal(typeof makeReadKey, 'function');
  assert.equal(makeReadKey('x64/world.rpf', 0, 64), makeReadKey('/x64/world.rpf', 0, 64));
  assert.notEqual(makeReadKey('x64/world.rpf', 0, 64), makeReadKey('x64/world.rpf', 64, 64));

  const src = fs.readFileSync(path.join(repoRoot, 'web/workers/io_scheduler.js'), 'utf8');
  const sandbox = {};
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;
  vm.runInNewContext(src, sandbox);

  assert.equal(sandbox.PlayGta6IoScheduler.DEFAULT_AGING_INTERVAL_MS, 0);
  assert.equal(sandbox.PlayGta6IoScheduler.DEFAULT_DEDUPE_IN_FLIGHT, true);
  assert.equal(typeof sandbox.PlayGta6IoScheduler.makeReadKey, 'function');
});

test('W206 in-flight deduplication coalesces identical concurrent reads and starts fresh after completion', async () => {
  const calls = [];
  let gate = deferred();

  const rangeCache = {
    read(filePath, offset, length) {
      calls.push(`${filePath}@${offset}+${length}`);
      return gate.promise;
    },
  };

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 4,
    sleep: async () => {},
  });
  assert.equal(scheduler.dedupeInFlight, true);

  // Three concurrent requests for the same canonical range (including leading slash).
  const p1 = scheduler.scheduleRead('x64/levels/world.rpf', 4096, 16, {
    priority: IO_PRIORITY.NORMAL,
  });
  const p2 = scheduler.scheduleRead('x64/levels/world.rpf', 4096, 16, {
    priority: IO_PRIORITY.HIGH,
  });
  const p3 = scheduler.scheduleRead('/x64/levels/world.rpf', 4096, 16, {
    priority: IO_PRIORITY.PREFETCH,
  });

  // Distinct offset and distinct length must NOT coalesce.
  const gateOther = deferred();
  const callsBeforeDistinct = calls.length;
  assert.equal(callsBeforeDistinct, 1);

  const midStats = scheduler.getStats();
  assert.equal(midStats.inFlight, 1);
  assert.equal(midStats.queued, 0);
  assert.equal(midStats.enqueued, 1);
  assert.equal(midStats.dispatched, 1);
  assert.equal(midStats.deduped, 2);
  assert.equal(midStats.byPriority.NORMAL.enqueued, 1);
  assert.equal(midStats.byPriority.HIGH.deduped, 1);
  assert.equal(midStats.byPriority.PREFETCH.deduped, 1);

  const expected = makeBytes(77, 16);
  gate.resolve(expected);
  const [r1, r2, r3] = await Promise.all([p1, p2, p3]);
  assert.deepEqual(r1, expected);
  assert.deepEqual(r2, expected);
  assert.deepEqual(r3, expected);
  assert.equal(calls.length, 1);

  // Once settled, a subsequent identical read must trigger a fresh rangeCache.read().
  gate = gateOther;
  const p4 = scheduler.scheduleRead('x64/levels/world.rpf', 4096, 16);
  const expected2 = makeBytes(88, 16);
  gateOther.resolve(expected2);
  const r4 = await p4;
  assert.deepEqual(r4, expected2);
  assert.equal(calls.length, 2);

  const finalStats = scheduler.getStats();
  assert.equal(finalStats.enqueued, 2);
  assert.equal(finalStats.dispatched, 2);
  assert.equal(finalStats.completed, 2);
  assert.equal(finalStats.deduped, 2);
  assert.equal(finalStats.inFlight, 0);
  assert.equal(finalStats.queued, 0);
});

test('W206 dedupeInFlight: false disables coalescing and distinct offsets/lengths never coalesce', async () => {
  const calls = [];
  const gates = [];

  const rangeCache = {
    read(filePath, offset, length) {
      calls.push(`${filePath}@${offset}+${length}`);
      const g = deferred();
      gates.push(g);
      return g.promise;
    },
  };

  // 1. Distinct ranges on same file with dedupe enabled never coalesce.
  const deduping = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 4,
    sleep: async () => {},
  });
  const pA = deduping.scheduleRead('world.rpf', 0, 16);
  const pB = deduping.scheduleRead('world.rpf', 16, 16);
  const pC = deduping.scheduleRead('world.rpf', 0, 32);
  assert.equal(calls.length, 3);
  assert.equal(deduping.getStats().deduped, 0);
  assert.equal(deduping.getStats().enqueued, 3);
  while (gates.length > 0) {
    gates.shift().resolve(makeBytes(1, 16));
  }
  await Promise.all([pA, pB, pC]);

  // 2. dedupeInFlight: false disables coalescing even for identical ranges.
  calls.length = 0;
  const noDedupe = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 4,
    dedupeInFlight: false,
    sleep: async () => {},
  });
  assert.equal(noDedupe.dedupeInFlight, false);
  const p1 = noDedupe.scheduleRead('world.rpf', 0, 16);
  const p2 = noDedupe.scheduleRead('world.rpf', 0, 16);
  assert.equal(calls.length, 2);
  assert.equal(noDedupe.getStats().deduped, 0);
  assert.equal(noDedupe.getStats().enqueued, 2);
  while (gates.length > 0) {
    gates.shift().resolve(makeBytes(2, 16));
  }
  await Promise.all([p1, p2]);
});

test('W206 queued deduplication promotes priority when higher-priority caller joins and demotes if that caller aborts', async () => {
  const dispatchOrder = [];
  const blockerGate = deferred();

  const rangeCache = {
    read(filePath, offset, length) {
      dispatchOrder.push(filePath);
      if (filePath === 'blocker.bin') {
        return blockerGate.promise;
      }
      return Promise.resolve(makeBytes(filePath.charCodeAt(0), length));
    },
  };

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 1,
    sleep: async () => {},
  });

  // Case 1: CRITICAL caller joins queued PREFETCH -> promotes ahead of HIGH/NORMAL.
  const pBlocker = scheduler.scheduleRead('blocker.bin', 0, 8, { priority: IO_PRIORITY.CRITICAL });
  const pPrefetchShared = scheduler.scheduleRead('shared_chunk.rpf', 0, 64, {
    priority: IO_PRIORITY.PREFETCH,
  });
  const pHigh = scheduler.scheduleRead('other_high.rpf', 0, 64, { priority: IO_PRIORITY.HIGH });
  const pNormal = scheduler.scheduleRead('other_normal.rpf', 0, 64, {
    priority: IO_PRIORITY.NORMAL,
  });

  // Also queue another PREFETCH task that gets temporarily promoted and then demoted on abort.
  const pDemoteBase = scheduler.scheduleRead('demoted_chunk.rpf', 0, 64, {
    priority: IO_PRIORITY.PREFETCH,
  });
  const demoteAbort = new AbortController();
  const pDemoteJoin = scheduler.scheduleRead('demoted_chunk.rpf', 0, 64, {
    priority: IO_PRIORITY.CRITICAL,
    signal: demoteAbort.signal,
  });

  // Now CRITICAL joins shared_chunk.rpf -> moves from PREFETCH queue to CRITICAL queue!
  const pCriticalJoin = scheduler.scheduleRead('shared_chunk.rpf', 0, 64, {
    priority: IO_PRIORITY.CRITICAL,
  });

  assert.equal(scheduler.getStats().byPriority.CRITICAL.queued, 2);
  assert.equal(scheduler.getStats().byPriority.PREFETCH.queued, 0);

  // Abort the CRITICAL joiner on demoted_chunk.rpf -> task demotes back to PREFETCH!
  demoteAbort.abort();
  await assert.rejects(pDemoteJoin, { name: 'AbortError' });
  assert.equal(scheduler.getStats().byPriority.CRITICAL.queued, 1);
  assert.equal(scheduler.getStats().byPriority.PREFETCH.queued, 1);

  blockerGate.resolve(makeBytes(1, 8));
  const [resPrefetch, resCritical] = await Promise.all([
    pPrefetchShared,
    pCriticalJoin,
    pBlocker,
    pHigh,
    pNormal,
    pDemoteBase,
  ]);

  assert.deepEqual(resPrefetch, resCritical);
  assert.deepEqual(dispatchOrder, [
    'blocker.bin',
    'shared_chunk.rpf', // promoted to CRITICAL by deduped joiner
    'other_high.rpf',
    'other_normal.rpf',
    'demoted_chunk.rpf', // demoted back to PREFETCH after CRITICAL joiner aborted
  ]);

  const stats = scheduler.getStats();
  assert.equal(stats.deduped, 2);
  assert.equal(stats.enqueued, 5);
  assert.equal(stats.dispatched, 5);
  assert.equal(stats.completed, 5);
  assert.equal(stats.cancelled, 1);
});

test('W206 partial and full AbortSignal cancellation on deduplicated in-flight reads preserves remaining waiters and maxConcurrency', async () => {
  const started = [];
  const gates = [];

  const rangeCache = {
    read(filePath, offset, length) {
      started.push(`${filePath}@${offset}`);
      const g = deferred();
      gates.push(g);
      return g.promise;
    },
  };

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 1,
    sleep: async () => {},
  });

  // 1. Partial abort: waiter A aborts while in-flight, waiter B keeps the read alive and resolves.
  const ctrlA = new AbortController();
  const pA = scheduler.scheduleRead('shared.rpf', 0, 32, { signal: ctrlA.signal });
  const pB = scheduler.scheduleRead('shared.rpf', 0, 32);

  ctrlA.abort();
  await assert.rejects(pA, { name: 'AbortError' });
  assert.equal(scheduler.getStats().cancelled, 1);
  assert.equal(scheduler.getStats().inFlight, 1);

  const bytes1 = makeBytes(42, 32);
  gates.shift().resolve(bytes1);
  assert.deepEqual(await pB, bytes1);
  assert.equal(started.length, 1);
  assert.equal(scheduler.getStats().completed, 1);
  assert.equal(scheduler.getStats().deduped, 1);

  // 2. Full abort: all waiters abort -> rejects both immediately, removes active dedupe key,
  //    holds concurrency slot until underlying read settles, then dispatches a fresh retry cleanly.
  const ctrl1 = new AbortController();
  const ctrl2 = new AbortController();
  const p1 = scheduler.scheduleRead('abort_all.rpf', 0, 16, { signal: ctrl1.signal });
  const p2 = scheduler.scheduleRead('abort_all.rpf', 0, 16, { signal: ctrl2.signal });

  ctrl1.abort();
  await assert.rejects(p1, { name: 'AbortError' });
  assert.equal(scheduler.getStats().inFlight, 1);

  ctrl2.abort();
  await assert.rejects(p2, { name: 'AbortError' });
  assert.equal(scheduler.getStats().cancelled, 3);
  assert.equal(scheduler.getStats().inFlight, 1); // still holding slot until underlying settles

  // Because active key was cleaned up when all waiters aborted, a new request for the same range
  // queues fresh (does not dedupe onto the aborted read) and runs once the slot frees.
  const pFresh = scheduler.scheduleRead('abort_all.rpf', 0, 16);
  assert.equal(scheduler.getStats().queued, 1);
  assert.equal(scheduler.getStats().deduped, 2); // did NOT increment for pFresh

  // Settle the old aborted read -> slot frees -> pFresh dispatches.
  gates.shift().reject(new Error('HTTP 503 late failure'));
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(started.length, 3);

  const bytes2 = makeBytes(55, 16);
  gates.shift().resolve(bytes2);
  assert.deepEqual(await pFresh, bytes2);
  assert.equal(scheduler.getStats().inFlight, 0);
  assert.equal(scheduler.getStats().completed, 2);
});

test('W206 deduplicated error and retry cleanup shares retries and clears active key before rejecting waiters', async () => {
  let flakeyAttempts = 0;
  let missingAttempts = 0;
  const delays = [];

  const rangeCache = {
    async read(filePath, offset, length) {
      if (filePath === 'flakey.rpf') {
        flakeyAttempts += 1;
        if (flakeyAttempts === 1) {
          throw new Error('HTTP 503 reading flakey.rpf');
        }
        return makeBytes(19, length);
      }
      if (filePath === 'missing_once.rpf') {
        missingAttempts += 1;
        if (missingAttempts === 1) {
          throw new Error('HTTP 404 reading missing_once.rpf');
        }
        return makeBytes(29, length);
      }
      return makeBytes(1, length);
    },
  };

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 2,
    maxRetries: 2,
    baseRetryDelayMs: 25,
    sleep: async (ms) => {
      delays.push(ms);
    },
  });

  // 1. Coalesced transient 503 retries once and resolves all waiters.
  const [r1, r2] = await Promise.all([
    scheduler.scheduleRead('flakey.rpf', 0, 16),
    scheduler.scheduleRead('flakey.rpf', 0, 16),
  ]);
  assert.equal(flakeyAttempts, 2);
  assert.deepEqual(delays, [25]);
  assert.deepEqual(r1, makeBytes(19, 16));
  assert.deepEqual(r2, makeBytes(19, 16));
  assert.equal(scheduler.getStats().retries, 1);
  assert.equal(scheduler.getStats().deduped, 1);

  // 2. Non-retryable 404 rejects both waiters AND clears active key before rejection so an
  //    immediate re-schedule inside .catch() starts a fresh read instead of re-attaching.
  let recoveryBytes = null;
  const pErr1 = scheduler.scheduleRead('missing_once.rpf', 0, 16).catch(async (err) => {
    assert.match(err.message, /HTTP 404/);
    recoveryBytes = await scheduler.scheduleRead('missing_once.rpf', 0, 16);
    return 'recovered';
  });
  const pErr2 = scheduler.scheduleRead('missing_once.rpf', 0, 16);

  assert.equal(await pErr1, 'recovered');
  await assert.rejects(pErr2, /HTTP 404/);
  assert.equal(missingAttempts, 2);
  assert.deepEqual(recoveryBytes, makeBytes(29, 16));

  const stats = scheduler.getStats();
  assert.equal(stats.enqueued, 3);
  assert.equal(stats.dispatched, 3);
  assert.equal(stats.completed, 2);
  assert.equal(stats.failed, 1);
  assert.equal(stats.deduped, 2);
});

test('W206 cancelPrefetch and clear reject all deduplicated waiters and clean up active keys', async () => {
  const blockerGate = deferred();
  const rangeCache = {
    read(filePath, offset, length) {
      if (filePath === 'hold.bin') return blockerGate.promise;
      return Promise.resolve(makeBytes(5, length));
    },
  };

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 1,
    sleep: async () => {},
  });

  const pHold = scheduler.scheduleRead('hold.bin', 0, 8, { priority: IO_PRIORITY.CRITICAL });
  const pPref1 = scheduler.scheduleRead('bg.rpf', 0, 16, { priority: IO_PRIORITY.PREFETCH });
  const pPref2 = scheduler.scheduleRead('bg.rpf', 0, 16, { priority: IO_PRIORITY.PREFETCH });

  assert.equal(scheduler.getStats().deduped, 1);
  assert.equal(scheduler.cancelPrefetch(), 2);
  await assert.rejects(pPref1, { name: 'AbortError' });
  await assert.rejects(pPref2, { name: 'AbortError' });

  // Active key was removed, so re-scheduling bg.rpf queues fresh.
  const pPref3 = scheduler.scheduleRead('bg.rpf', 0, 16, { priority: IO_PRIORITY.NORMAL });
  const pPref4 = scheduler.scheduleRead('bg.rpf', 0, 16, { priority: IO_PRIORITY.NORMAL });
  assert.equal(scheduler.clear(), 2);
  await assert.rejects(pPref3, { name: 'AbortError' });
  await assert.rejects(pPref4, { name: 'AbortError' });

  blockerGate.resolve(makeBytes(1, 8));
  await pHold;
  assert.equal(scheduler.getStats().inFlight, 0);
  assert.equal(scheduler.getStats().queued, 0);
});

test('W206 step-based priority aging (agingIntervalMs) prevents starvation under continuous CRITICAL/HIGH load and tracks top-level latency telemetry', async () => {
  let fakeNow = 0;
  const dispatchOrder = [];
  const gates = new Map();

  const rangeCache = {
    read(filePath, offset, length) {
      dispatchOrder.push(filePath);
      const g = gates.get(filePath);
      if (g) return g.promise;
      return Promise.resolve(makeBytes(3, length));
    },
  };

  const gBlocker = deferred();
  const gCrit1 = deferred();
  const gHigh1 = deferred();
  const gCrit2 = deferred();
  const gPrefetch = deferred();
  gates.set('blocker.bin', gBlocker);
  gates.set('crit-1.bin', gCrit1);
  gates.set('high-1.bin', gHigh1);
  gates.set('crit-2.bin', gCrit2);
  gates.set('bg-prefetch.rpf', gPrefetch);

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 1,
    agingIntervalMs: 50,
    nowFn: () => fakeNow,
    sleep: async () => {},
  });
  assert.equal(scheduler.agingIntervalMs, 50);

  // t = 0: blocker runs immediately; background PREFETCH (tier 3) queues up.
  const pBlocker = scheduler.scheduleRead('blocker.bin', 0, 8, { priority: IO_PRIORITY.CRITICAL });
  const pBgPrefetch = scheduler.scheduleRead('bg-prefetch.rpf', 0, 8, {
    priority: IO_PRIORITY.PREFETCH,
  });

  // t = 50: PREFETCH aged 1 step (tier 3 -> eff 2 NORMAL). Fresh CRITICAL (tier 0) arrives -> CRITICAL wins.
  fakeNow = 50;
  const pCrit1 = scheduler.scheduleRead('crit-1.bin', 0, 8, { priority: IO_PRIORITY.CRITICAL });
  const pHigh1 = scheduler.scheduleRead('high-1.bin', 0, 8, { priority: IO_PRIORITY.HIGH });
  gBlocker.resolve(makeBytes(1, 8));
  await pBlocker;
  await Promise.resolve();
  assert.deepEqual(dispatchOrder, ['blocker.bin', 'crit-1.bin']);

  // t = 100: PREFETCH aged 2 steps (tier 3 -> eff 1 HIGH, enqueuedAt 0).
  //          high-1 waited 50ms -> aged 1 step (tier 1 -> eff 0 CRITICAL)! So high-1 runs before PREFETCH.
  fakeNow = 100;
  gCrit1.resolve(makeBytes(2, 8));
  await pCrit1;
  await Promise.resolve();
  assert.deepEqual(dispatchOrder, ['blocker.bin', 'crit-1.bin', 'high-1.bin']);

  // t = 150: fresh CRITICAL burst arrives (crit-2 at t = 150, eff 0, enq 150).
  //          PREFETCH has waited 150ms -> aged 3 steps (tier 3 -> eff 0 CRITICAL, enq 0 < 150)!
  //          PREFETCH MUST promote ahead of fresh crit-2 so it cannot starve!
  fakeNow = 150;
  const pCrit2 = scheduler.scheduleRead('crit-2.bin', 0, 8, { priority: IO_PRIORITY.CRITICAL });
  gHigh1.resolve(makeBytes(3, 8));
  await pHigh1;
  await Promise.resolve();

  assert.deepEqual(dispatchOrder, ['blocker.bin', 'crit-1.bin', 'high-1.bin', 'bg-prefetch.rpf']);
  assert.equal(scheduler.getStats().agedPromotions, 1);
  assert.equal(scheduler.getStats().byPriority.PREFETCH.agedPromotions, 1);

  // t = 180: PREFETCH completes -> crit-2 runs (wait = 30ms).
  fakeNow = 180;
  gPrefetch.resolve(makeBytes(4, 8));
  await pBgPrefetch;
  await Promise.resolve();
  gCrit2.resolve(makeBytes(5, 8));
  await pCrit2;

  assert.deepEqual(dispatchOrder, [
    'blocker.bin',
    'crit-1.bin',
    'high-1.bin',
    'bg-prefetch.rpf',
    'crit-2.bin',
  ]);

  // Exact queue waits:
  // blocker: 0ms, crit-1: 0ms, high-1: 50ms, bg-prefetch: 150ms, crit-2: 30ms
  // total = 230ms across 5 dispatches => avgQueueWaitMs = 46ms, maxQueueWaitMs = 150ms
  const stats = scheduler.getStats();
  assert.equal(stats.enqueued, 5);
  assert.equal(stats.dispatched, 5);
  assert.equal(stats.completed, 5);
  assert.equal(stats.deduped, 0);
  assert.equal(stats.agedPromotions, 1);
  assert.equal(stats.avgQueueWaitMs, 46);
  assert.equal(stats.maxQueueWaitMs, 150);

  // Verify per-priority rollups match top-level telemetry and reset() clears all W206 fields.
  const prios = Object.values(stats.byPriority);
  assert.equal(prios.reduce((s, p) => s + p.enqueued, 0), stats.enqueued);
  assert.equal(prios.reduce((s, p) => s + p.dispatched, 0), stats.dispatched);
  assert.equal(prios.reduce((s, p) => s + p.deduped, 0), stats.deduped);

  scheduler.reset();
  const afterReset = scheduler.getStats();
  assert.equal(afterReset.enqueued, 0);
  assert.equal(afterReset.dispatched, 0);
  assert.equal(afterReset.completed, 0);
  assert.equal(afterReset.deduped, 0);
  assert.equal(afterReset.agedPromotions, 0);
  assert.equal(afterReset.avgQueueWaitMs, 0);
  assert.equal(afterReset.maxQueueWaitMs, 0);
});

// ---------------------------------------------------------------------------
// W248: Boot-critical RPF priority classification & queue-latency telemetry
// ---------------------------------------------------------------------------

test('W248 exports classifyBootArchivePriority and isBootCriticalRpfPath in CommonJS and Worker globalThis', () => {
  assert.equal(typeof classifyBootArchivePriority, 'function');
  assert.equal(typeof isBootCriticalRpfPath, 'function');

  const src = fs.readFileSync(path.join(repoRoot, 'web/workers/io_scheduler.js'), 'utf8');
  const sandbox = {};
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;
  vm.runInNewContext(src, sandbox);

  assert.equal(typeof sandbox.PlayGta6IoScheduler.classifyBootArchivePriority, 'function');
  assert.equal(typeof sandbox.PlayGta6IoScheduler.isBootCriticalRpfPath, 'function');
  assert.equal(
    sandbox.PlayGta6IoScheduler.classifyBootArchivePriority('/game/movies.rpf'),
    IO_PRIORITY.HIGH
  );
  assert.equal(
    sandbox.PlayGta6IoScheduler.classifyBootArchivePriority('/shaders/boot.wgsl.pack'),
    IO_PRIORITY.HIGH
  );
  assert.equal(
    sandbox.PlayGta6IoScheduler.classifyBootArchivePriority('boot/guest.pgim'),
    IO_PRIORITY.HIGH
  );
});

test('W248 classifyBootArchivePriority classifies boot-critical RPFs, shader packs, and pgim assets across paths and URLs without leaking host paths', () => {
  // Boot-critical RPFs: movies.rpf, shaders.rpf, textures.rpf, levels.rpf
  for (const input of [
    '/game/movies.rpf',
    '/game/shaders.rpf',
    '/game/textures.rpf',
    '/game/levels.rpf',
    'game/movies.rpf',
    'movies.rpf',
    '/GAME/SHADERS.RPF?v=2#frag',
    'http://127.0.0.1:8080/game/textures.rpf?range=0-4095',
    'https://cdn.example.invalid/data/game/levels.rpf#top',
    'rpf://game/movies.rpf/intro.bk2',
    'rpf://shaders.rpf/core.bin',
    'C:\\Users\\private_user\\Xbox\\game\\movies.rpf',
    'file:///C:/Users/private_user/game/textures.rpf',
    '\\\\?\\G:\\game\\levels.rpf',
  ]) {
    assert.equal(
      classifyBootArchivePriority(input),
      IO_PRIORITY.HIGH,
      `Expected HIGH from classifyBootArchivePriority for ${input}`
    );
    assert.equal(
      isBootCriticalRpfPath(input),
      true,
      `Expected isBootCriticalRpfPath true for ${input}`
    );
    assert.equal(
      classifyIoPriority(input),
      IO_PRIORITY.HIGH,
      `Expected HIGH from classifyIoPriority for ${input}`
    );
  }

  // Shader packs & .pgim guest images -> HIGH from classifyBootArchivePriority,
  // while classifyIoPriority preserves CRITICAL (W188 compatibility).
  for (const input of [
    '/shaders/frame1.wgsl.pack',
    'shaders/main.wgsl.pack?v=3#entry',
    'http://127.0.0.1:8080/shaders/core.wgsl.pack',
    'guest/game_main.pgim',
    '/boot/game.pgim',
    'https://localhost:8080/boot/GAME.PGIM?cache=1',
    'C:\\Users\\private_user\\work\\game.pgim',
  ]) {
    assert.equal(
      classifyBootArchivePriority(input),
      IO_PRIORITY.HIGH,
      `Expected HIGH from classifyBootArchivePriority for ${input}`
    );
    assert.equal(
      isBootCriticalRpfPath(input),
      false,
      `Expected isBootCriticalRpfPath false for ${input}`
    );
    assert.equal(
      classifyIoPriority(input),
      IO_PRIORITY.CRITICAL,
      `Expected CRITICAL preserved from classifyIoPriority for ${input}`
    );
  }

  // Non-boot RPFs, false-positive substring guards, and invalid/empty inputs -> NORMAL
  for (const input of [
    '/game/audio.rpf',
    '/game/my_movies.rpf',
    '/game/levels_extra.rpf',
    'x64/levels/world.rpf',
    'x64/textures/terrain.ttpf',
    'https://movies.rpf',
    'misc/data.bin',
    '',
    '   ',
    null,
    undefined,
    42,
    {},
  ]) {
    assert.equal(
      classifyBootArchivePriority(input),
      IO_PRIORITY.NORMAL,
      `Expected NORMAL from classifyBootArchivePriority for ${String(input)}`
    );
    assert.equal(isBootCriticalRpfPath(input), false);
  }

  // Speculative prefetch paths remain PREFETCH
  assert.equal(classifyBootArchivePriority('prefetch/world_lod.rpf'), IO_PRIORITY.PREFETCH);
  assert.equal(classifyBootArchivePriority('x64/speculative/movies.rpf'), IO_PRIORITY.PREFETCH);
  assert.equal(classifyBootArchivePriority('movies.rpf.prefetch'), IO_PRIORITY.PREFETCH);

  // Explicit caller override still takes precedence in classifyIoPriority
  assert.equal(classifyIoPriority('/game/movies.rpf', IO_PRIORITY.PREFETCH), IO_PRIORITY.PREFETCH);
  assert.equal(classifyIoPriority('/game/levels.rpf', IO_PRIORITY.CRITICAL), IO_PRIORITY.CRITICAL);
});

test('W248 boot-critical RPFs preempt queued NORMAL/PREFETCH reads and record sanitized telemetry', async () => {
  const dispatchOrder = [];
  const blockerGate = deferred();

  const rangeCache = {
    read(filePath, offset, length) {
      dispatchOrder.push(filePath);
      if (filePath === 'blocker.bin') {
        return blockerGate.promise;
      }
      return Promise.resolve(makeBytes(filePath.charCodeAt(0), length));
    },
  };

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 1,
    sleep: async () => {},
  });

  // Occupy concurrency slot so subsequent reads queue up.
  const pBlocker = scheduler.scheduleRead('blocker.bin', 0, 16, {
    priority: IO_PRIORITY.PREFETCH,
  });

  // Queue lower-priority traffic BEFORE boot-critical archives arrive.
  const pPrefetch = scheduler.scheduleRead('prefetch/bg_chunk.rpf', 0, 16);
  const pNormal = scheduler.scheduleRead('/game/audio.rpf', 0, 16);

  // Queue boot-critical RPFs, shader pack, and guest .pgim without explicit priority.
  const pMovies = scheduler.scheduleRead('/game/movies.rpf', 0, 64);
  const pShaders = scheduler.scheduleRead('/game/shaders.rpf', 0, 64);
  const pTextures = scheduler.scheduleRead('/game/textures.rpf', 0, 64);
  const pLevels = scheduler.scheduleRead('/game/levels.rpf', 0, 64);
  const pPack = scheduler.scheduleRead('/shaders/boot.wgsl.pack', 0, 128);
  const pPgim = scheduler.scheduleRead('/boot/guest.pgim', 0, 128);

  const midStats = scheduler.getStats();
  assert.equal(midStats.queued, 8);
  assert.equal(midStats.peakQueueDepth, 8);
  assert.equal(midStats.highPriorityCount, 6);
  assert.equal(midStats.preemptedCount, 0);

  blockerGate.resolve(makeBytes(1, 16));
  await Promise.all([
    pBlocker,
    pPrefetch,
    pNormal,
    pMovies,
    pShaders,
    pTextures,
    pLevels,
    pPack,
    pPgim,
  ]);

  assert.deepEqual(dispatchOrder, [
    'blocker.bin',
    '/shaders/boot.wgsl.pack',
    '/boot/guest.pgim',
    '/game/movies.rpf',
    '/game/shaders.rpf',
    '/game/textures.rpf',
    '/game/levels.rpf',
    '/game/audio.rpf',
    'prefetch/bg_chunk.rpf',
  ]);

  const finalStats = scheduler.getStats();
  assert.equal(finalStats.queued, 0);
  assert.equal(finalStats.inFlight, 0);
  assert.equal(finalStats.peakQueueDepth, 8);
  assert.equal(finalStats.highPriorityCount, 6);
  // 6 CRITICAL/HIGH reads jumped ahead of earlier NORMAL/PREFETCH, and NORMAL jumped ahead of earlier PREFETCH.
  assert.equal(finalStats.preemptedCount, 7);
  assert.equal(finalStats.completed, 9);

  // Verify getStats() output is strictly sanitized (contains no raw paths or archive names).
  const serialized = JSON.stringify(finalStats);
  assert.equal(serialized.includes('movies.rpf'), false);
  assert.equal(serialized.includes('shaders.rpf'), false);
  assert.equal(serialized.includes('wgsl.pack'), false);
  assert.equal(serialized.includes('guest.pgim'), false);
  assert.equal(serialized.includes('blocker.bin'), false);
});

test('W248 deterministic queue-latency telemetry (peakQueueDepth, avgWaitMs, maxWaitMs, highPriorityCount, preemptedCount) and FIFO/aging boundaries', async () => {
  let fakeNow = 0;
  const gates = new Map();

  const rangeCache = {
    read(filePath, offset, length) {
      const gate = gates.get(filePath);
      if (gate) return gate.promise;
      return Promise.resolve(makeBytes(7, length));
    },
  };

  const gBlocker = deferred();
  const gMovies = deferred();
  const gPrefetch = deferred();
  const gNormal = deferred();
  gates.set('blocker.bin', gBlocker);
  gates.set('/game/movies.rpf', gMovies);
  gates.set('prefetch/bg.rpf', gPrefetch);
  gates.set('/game/audio.rpf', gNormal);

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 1,
    agingThresholdMs: 100,
    nowFn: () => fakeNow,
    sleep: async () => {},
  });

  // Pre-aborted and invalid reads must not affect W248 telemetry counters.
  const preAborted = new AbortController();
  preAborted.abort();
  await assert.rejects(
    scheduler.scheduleRead('/game/movies.rpf', 0, 16, { signal: preAborted.signal }),
    { name: 'AbortError' }
  );
  await assert.rejects(scheduler.scheduleRead('/game/shaders.rpf', -1, 16), RangeError);

  assert.equal(scheduler.getStats().highPriorityCount, 0);
  assert.equal(scheduler.getStats().peakQueueDepth, 0);
  assert.equal(scheduler.getStats().preemptedCount, 0);

  // t = 0: blocker dispatches immediately (wait = 0ms); prefetch queues at t = 0.
  const pBlocker = scheduler.scheduleRead('blocker.bin', 0, 8, { priority: IO_PRIORITY.NORMAL });
  const pPrefetch = scheduler.scheduleRead('prefetch/bg.rpf', 0, 8);

  // t = 10: NORMAL queues.
  fakeNow = 10;
  const pNormal = scheduler.scheduleRead('/game/audio.rpf', 0, 8);

  // t = 25: HIGH boot RPF queues (depth = 3: prefetch, normal, movies).
  fakeNow = 25;
  const pMovies = scheduler.scheduleRead('/game/movies.rpf', 0, 8);

  assert.equal(scheduler.getStats().peakQueueDepth, 3);

  // t = 50: blocker completes -> movies.rpf (wait = 25ms) preempts earlier NORMAL & PREFETCH.
  fakeNow = 50;
  gBlocker.resolve(makeBytes(1, 8));
  await pBlocker;
  await Promise.resolve();

  const afterMoviesDispatch = scheduler.getStats();
  assert.equal(afterMoviesDispatch.preemptedCount, 1);
  assert.equal(afterMoviesDispatch.agedPromotions, 0);

  // t = 120: movies completes -> prefetch waited 120ms (>= 100ms agingThresholdMs) so aging promotes it.
  // Aged promotion must NOT increment preemptedCount.
  fakeNow = 120;
  gMovies.resolve(makeBytes(2, 8));
  await pMovies;
  await Promise.resolve();

  // t = 140: prefetch completes -> audio.rpf dispatches last (wait = 140 - 10 = 130ms).
  fakeNow = 140;
  gPrefetch.resolve(makeBytes(3, 8));
  await pPrefetch;
  await Promise.resolve();

  gNormal.resolve(makeBytes(4, 8));
  await pNormal;

  // Dispatched waits: blocker (0) + movies (25) + prefetch (120) + audio (130)
  // Total wait = 275ms across 4 dispatches => avgWaitMs = 68.75ms, maxWaitMs = 130ms.
  const stats = scheduler.getStats();
  assert.equal(stats.peakQueueDepth, 3);
  assert.equal(stats.avgWaitMs, 68.75);
  assert.equal(stats.avgWaitMs, stats.avgQueueWaitMs);
  assert.equal(stats.maxWaitMs, 130);
  assert.equal(stats.maxWaitMs, stats.maxQueueWaitMs);
  assert.equal(stats.highPriorityCount, 1);
  assert.equal(stats.preemptedCount, 1);
  assert.equal(stats.agedPromotions, 1);

  // Snapshot isolation: mutating returned fields must not corrupt internal state.
  stats.peakQueueDepth = 999;
  stats.avgWaitMs = 999;
  stats.maxWaitMs = 999;
  stats.highPriorityCount = 999;
  stats.preemptedCount = 999;
  const fresh = scheduler.getStats();
  assert.equal(fresh.peakQueueDepth, 3);
  assert.equal(fresh.avgWaitMs, 68.75);
  assert.equal(fresh.maxWaitMs, 130);
  assert.equal(fresh.highPriorityCount, 1);
  assert.equal(fresh.preemptedCount, 1);

  // FIFO ordering when higher priority is enqueued BEFORE lower priority must yield preemptedCount === 0.
  scheduler.resetStats();
  const gHold2 = deferred();
  gates.set('hold2.bin', gHold2);
  fakeNow = 200;
  const pHold2 = scheduler.scheduleRead('hold2.bin', 0, 8, { priority: IO_PRIORITY.NORMAL });
  const pHighFirst = scheduler.scheduleRead('/game/textures.rpf', 0, 8);
  const pNormalSecond = scheduler.scheduleRead('plain.bin', 0, 8);
  gHold2.resolve(makeBytes(9, 8));
  await Promise.all([pHold2, pHighFirst, pNormalSecond]);
  assert.equal(scheduler.getStats().preemptedCount, 0);
  assert.equal(scheduler.getStats().highPriorityCount, 1);
  assert.equal(scheduler.getStats().peakQueueDepth, 2);
});

test('W248 resetStats() cleanly resets telemetry without cancelling queued or in-flight reads', async () => {
  let fakeNow = 0;
  const g1 = deferred();
  const g2 = deferred();

  const rangeCache = {
    read(filePath, offset, length) {
      if (filePath === 'first.bin') return g1.promise;
      if (filePath === 'second.bin') return g2.promise;
      return Promise.resolve(makeBytes(4, length));
    },
  };

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 1,
    nowFn: () => fakeNow,
    sleep: async () => {},
  });

  // Accumulate telemetry while first.bin is in-flight and queued reads wait.
  const p1 = scheduler.scheduleRead('first.bin', 0, 8, { priority: IO_PRIORITY.NORMAL });
  const p2 = scheduler.scheduleRead('second.bin', 0, 8, { priority: IO_PRIORITY.NORMAL });
  const p3 = scheduler.scheduleRead('/game/shaders.rpf', 0, 8);

  assert.equal(scheduler.getStats().inFlight, 1);
  assert.equal(scheduler.getStats().queued, 2);
  assert.equal(scheduler.getStats().peakQueueDepth, 2);
  assert.equal(scheduler.getStats().highPriorityCount, 1);

  // Call resetStats() mid-flight: telemetry zeroes, but active & queued tasks are NOT cancelled.
  scheduler.resetStats();
  const afterMidReset = scheduler.getStats();
  assert.equal(afterMidReset.inFlight, 1);
  assert.equal(afterMidReset.queued, 2);
  assert.equal(afterMidReset.peakQueueDepth, 0);
  assert.equal(afterMidReset.avgWaitMs, 0);
  assert.equal(afterMidReset.maxWaitMs, 0);
  assert.equal(afterMidReset.highPriorityCount, 0);
  assert.equal(afterMidReset.preemptedCount, 0);
  assert.equal(afterMidReset.completed, 0);
  assert.equal(afterMidReset.cancelled, 0);

  // Unblock first.bin at t = 40 -> queued shaders.rpf preempts second.bin and resolves cleanly!
  fakeNow = 40;
  g1.resolve(makeBytes(1, 8));
  await p1;
  await p3;

  fakeNow = 60;
  g2.resolve(makeBytes(2, 8));
  await p2;

  const afterDrain = scheduler.getStats();
  assert.equal(afterDrain.inFlight, 0);
  assert.equal(afterDrain.queued, 0);
  assert.equal(afterDrain.completed, 3);
  assert.equal(afterDrain.preemptedCount, 1);
  assert.equal(afterDrain.maxWaitMs, 40);
  assert.equal(afterDrain.avgWaitMs, 40);

  // Calling resetStats() when idle zeroes every telemetry field cleanly.
  scheduler.resetStats();
  const afterIdleReset = scheduler.getStats();
  assert.equal(afterIdleReset.peakQueueDepth, 0);
  assert.equal(afterIdleReset.avgWaitMs, 0);
  assert.equal(afterIdleReset.maxWaitMs, 0);
  assert.equal(afterIdleReset.avgQueueWaitMs, 0);
  assert.equal(afterIdleReset.maxQueueWaitMs, 0);
  assert.equal(afterIdleReset.highPriorityCount, 0);
  assert.equal(afterIdleReset.preemptedCount, 0);
  assert.equal(afterIdleReset.completed, 0);
  assert.equal(afterIdleReset.dispatched, 0);
  assert.equal(afterIdleReset.enqueued, 0);
});

// ---------------------------------------------------------------------------
// W288: Starvation-free priority aging (agingWindowMs), injectable `now` clock,
//       promotedByAgingCount, maxQueueWaitMs, activeByPriority, and resetStats().
// ---------------------------------------------------------------------------

test('W288 exports DEFAULT_AGING_WINDOW_MS, applies 2000ms default, and classifies core_bvh4.rpf as boot-critical HIGH', () => {
  assert.equal(DEFAULT_AGING_WINDOW_MS, 2000);

  const scheduler = new IoPriorityScheduler({
    rangeCache: { read: async () => new Uint8Array(0) },
  });
  assert.equal(scheduler.agingWindowMs, 2000);

  const src = fs.readFileSync(path.join(repoRoot, 'web/workers/io_scheduler.js'), 'utf8');
  const sandbox = {};
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;
  vm.runInNewContext(src, sandbox);

  assert.equal(sandbox.PlayGta6IoScheduler.DEFAULT_AGING_WINDOW_MS, 2000);
  assert.equal(isBootCriticalRpfPath('/game/core_bvh4.rpf'), true);
  assert.equal(classifyBootArchivePriority('/game/core_bvh4.rpf'), IO_PRIORITY.HIGH);
  assert.equal(classifyIoPriority('/game/core_bvh4.rpf'), IO_PRIORITY.HIGH);
  assert.equal(classifyIoPriority('rpf://game/core_bvh4.rpf/node_0.bin'), IO_PRIORITY.HIGH);
});

test('W288 agingWindowMs promotes NORMAL and PREFETCH one tier per window under heavy HIGH load using injectable `now` clock', async () => {
  let mockNow = 0;
  const dispatchOrder = [];
  const gates = new Map();

  const rangeCache = {
    read(filePath, offset, length) {
      dispatchOrder.push(filePath);
      const gate = gates.get(filePath);
      if (gate) return gate.promise;
      return Promise.resolve(makeBytes(filePath.charCodeAt(0), length));
    },
  };

  const gBlocker = deferred();
  const gHigh1 = deferred();
  const gNormal = deferred();
  const gHigh2 = deferred();
  const gPrefetch = deferred();
  gates.set('blocker.bin', gBlocker);
  gates.set('high-1.rpf', gHigh1);
  gates.set('normal-old.rpf', gNormal);
  gates.set('high-2.rpf', gHigh2);
  gates.set('prefetch-old.rpf', gPrefetch);

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 1,
    agingWindowMs: 500,
    now: () => mockNow,
    sleep: async () => {},
  });
  assert.equal(scheduler.agingWindowMs, 500);

  // t = 0: blocker runs; NORMAL (tier 2) and PREFETCH (tier 3) queue up.
  const pBlocker = scheduler.scheduleRead('blocker.bin', 0, 16, { priority: IO_PRIORITY.HIGH });
  const pNormal = scheduler.scheduleRead('normal-old.rpf', 0, 16, { priority: IO_PRIORITY.NORMAL });
  const pPrefetch = scheduler.scheduleRead('prefetch-old.rpf', 0, 16, {
    priority: IO_PRIORITY.PREFETCH,
  });

  // t = 250 (< 500ms): neither NORMAL nor PREFETCH has aged -> fresh HIGH wins.
  mockNow = 250;
  const pHigh1 = scheduler.scheduleRead('high-1.rpf', 0, 16, { priority: IO_PRIORITY.HIGH });
  gBlocker.resolve(makeBytes(1, 16));
  await pBlocker;
  await Promise.resolve();

  assert.deepEqual(dispatchOrder, ['blocker.bin', 'high-1.rpf']);
  assert.equal(scheduler.getStats().promotedByAgingCount, 0);

  // t = 500 (1 window): NORMAL (wait 500ms -> 1 step -> eff HIGH, enq 0) beats fresh HIGH (enq 500);
  // PREFETCH (wait 500ms -> 1 step -> eff NORMAL) has not reached HIGH yet.
  mockNow = 500;
  const pHigh2 = scheduler.scheduleRead('high-2.rpf', 0, 16, { priority: IO_PRIORITY.HIGH });
  gHigh1.resolve(makeBytes(2, 16));
  await pHigh1;
  await Promise.resolve();

  assert.deepEqual(dispatchOrder, ['blocker.bin', 'high-1.rpf', 'normal-old.rpf']);
  assert.equal(scheduler.getStats().promotedByAgingCount, 1);
  assert.equal(scheduler.getStats().byPriority.NORMAL.promotedByAgingCount, 1);

  // t = 750 (< 1000ms): PREFETCH has only aged 1 step (to eff NORMAL), so queued high-2 (eff HIGH) runs first.
  mockNow = 750;
  const pHigh3 = scheduler.scheduleRead('high-3.rpf', 0, 16, { priority: IO_PRIORITY.HIGH });
  gNormal.resolve(makeBytes(3, 16));
  await pNormal;
  await Promise.resolve();

  assert.deepEqual(dispatchOrder, ['blocker.bin', 'high-1.rpf', 'normal-old.rpf', 'high-2.rpf']);
  assert.equal(scheduler.getStats().promotedByAgingCount, 1);

  // t = 1000 (2 windows): PREFETCH (wait 1000ms -> 2 steps -> eff HIGH, enq 0) beats high-3 (enq 750)!
  mockNow = 1000;
  gHigh2.resolve(makeBytes(4, 16));
  await pHigh2;
  await Promise.resolve();

  assert.deepEqual(dispatchOrder, [
    'blocker.bin',
    'high-1.rpf',
    'normal-old.rpf',
    'high-2.rpf',
    'prefetch-old.rpf',
  ]);
  assert.equal(scheduler.getStats().promotedByAgingCount, 2);
  assert.equal(scheduler.getStats().byPriority.PREFETCH.promotedByAgingCount, 1);

  // Finish PREFETCH -> high-3 drains last.
  mockNow = 1050;
  gPrefetch.resolve(makeBytes(5, 16));
  await Promise.all([pPrefetch, pHigh3]);

  assert.deepEqual(dispatchOrder, [
    'blocker.bin',
    'high-1.rpf',
    'normal-old.rpf',
    'high-2.rpf',
    'prefetch-old.rpf',
    'high-3.rpf',
  ]);

  const stats = scheduler.getStats();
  assert.equal(stats.promotedByAgingCount, 2);
  assert.equal(stats.agedPromotions, 2);
  assert.equal(stats.promotions, 2);
  assert.equal(stats.maxQueueWaitMs, 1000);
  assert.equal(stats.byPriority.NORMAL.maxQueueWaitMs, 500);
  assert.equal(stats.byPriority.PREFETCH.maxQueueWaitMs, 1000);
});

test('W288 CRITICAL stays strictly highest regardless of aging wait time while aged tasks still beat fresh HIGH bursts', async () => {
  let mockNow = 0;
  const dispatchOrder = [];
  const gates = new Map();

  const rangeCache = {
    read(filePath, offset, length) {
      dispatchOrder.push(filePath);
      const gate = gates.get(filePath);
      if (gate) return gate.promise;
      return Promise.resolve(makeBytes(filePath.charCodeAt(0), length));
    },
  };

  const gBlocker = deferred();
  const gCrit1 = deferred();
  const gCrit2 = deferred();
  gates.set('blocker.bin', gBlocker);
  gates.set('shaders/crit-1.wgsl.pack', gCrit1);
  gates.set('shaders/crit-2.wgsl.pack', gCrit2);

  // Uses default agingWindowMs (2000ms) with injectable `now` clock.
  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 1,
    now: () => mockNow,
    sleep: async () => {},
  });
  assert.equal(scheduler.agingWindowMs, DEFAULT_AGING_WINDOW_MS);

  // t = 0: blocker runs; PREFETCH and NORMAL queue up.
  const pBlocker = scheduler.scheduleRead('blocker.bin', 0, 16, { priority: IO_PRIORITY.HIGH });
  const pPrefetch = scheduler.scheduleRead('prefetch/world_lod.rpf', 0, 16);
  const pNormal = scheduler.scheduleRead('x64/levels/world.rpf', 0, 16);

  // t = 20000 (10x agingWindowMs): both PREFETCH and NORMAL are capped at HIGH (below CRITICAL).
  // Fresh CRITICAL and HIGH arrive.
  mockNow = 20000;
  const pHighFresh = scheduler.scheduleRead('/game/core_bvh4.rpf', 0, 16);
  const pCrit1 = scheduler.scheduleRead('shaders/crit-1.wgsl.pack', 0, 16);

  gBlocker.resolve(makeBytes(1, 16));
  await pBlocker;
  await Promise.resolve();

  // CRITICAL MUST run first even though PREFETCH/NORMAL waited 20,000ms!
  assert.deepEqual(dispatchOrder, ['blocker.bin', 'shaders/crit-1.wgsl.pack']);
  assert.equal(scheduler.getStats().promotedByAgingCount, 0);

  // Another CRITICAL arrives before crit-1 completes -> still runs ahead of aged tasks!
  mockNow = 20500;
  const pCrit2 = scheduler.scheduleRead('shaders/crit-2.wgsl.pack', 0, 16);
  gCrit1.resolve(makeBytes(2, 16));
  await pCrit1;
  await Promise.resolve();

  assert.deepEqual(dispatchOrder, [
    'blocker.bin',
    'shaders/crit-1.wgsl.pack',
    'shaders/crit-2.wgsl.pack',
  ]);
  assert.equal(scheduler.getStats().promotedByAgingCount, 0);

  // Once CRITICAL queue is empty, aged NORMAL and PREFETCH (capped at HIGH, enq 0) beat fresh HIGH (enq 20000)!
  mockNow = 21000;
  gCrit2.resolve(makeBytes(3, 16));
  await Promise.all([pCrit2, pNormal, pPrefetch, pHighFresh]);

  assert.deepEqual(dispatchOrder, [
    'blocker.bin',
    'shaders/crit-1.wgsl.pack',
    'shaders/crit-2.wgsl.pack',
    'x64/levels/world.rpf',
    'prefetch/world_lod.rpf',
    '/game/core_bvh4.rpf',
  ]);

  const stats = scheduler.getStats();
  assert.equal(stats.promotedByAgingCount, 2);
  assert.equal(stats.maxQueueWaitMs, 21000);
});

test('W288 agingWindowMs: 0 and agingWindowMs: false cleanly opt out of priority aging', async () => {
  for (const disabledOpt of [0, false]) {
    let mockNow = 0;
    const dispatchOrder = [];
    const gBlocker = deferred();

    const rangeCache = {
      read(filePath, offset, length) {
        dispatchOrder.push(filePath);
        if (filePath === 'blocker.bin') return gBlocker.promise;
        return Promise.resolve(makeBytes(9, length));
      },
    };

    const scheduler = new IoPriorityScheduler({
      rangeCache,
      maxConcurrency: 1,
      agingWindowMs: disabledOpt,
      now: () => mockNow,
      sleep: async () => {},
    });
    assert.equal(scheduler.agingWindowMs, 0);

    const pBlocker = scheduler.scheduleRead('blocker.bin', 0, 8, { priority: IO_PRIORITY.NORMAL });
    const pPrefetch = scheduler.scheduleRead('prefetch-old.rpf', 0, 8, {
      priority: IO_PRIORITY.PREFETCH,
    });
    const pNormal = scheduler.scheduleRead('normal-old.rpf', 0, 8, {
      priority: IO_PRIORITY.NORMAL,
    });

    // Advance well past default 2000ms window before fresh HIGH & CRITICAL arrive.
    mockNow = 60000;
    const pHigh = scheduler.scheduleRead('high-fresh.rpf', 0, 8, { priority: IO_PRIORITY.HIGH });
    const pCritical = scheduler.scheduleRead('critical-fresh.rpf', 0, 8, {
      priority: IO_PRIORITY.CRITICAL,
    });

    gBlocker.resolve(makeBytes(1, 8));
    await Promise.all([pBlocker, pPrefetch, pNormal, pHigh, pCritical]);

    assert.deepEqual(dispatchOrder, [
      'blocker.bin',
      'critical-fresh.rpf',
      'high-fresh.rpf',
      'normal-old.rpf',
      'prefetch-old.rpf',
    ]);

    const stats = scheduler.getStats();
    assert.equal(stats.promotedByAgingCount, 0);
    assert.equal(stats.maxQueueWaitMs, 60000);
  }
});

test('W288 tracks activeByPriority, promotedByAgingCount, and maxQueueWaitMs across concurrent reads with clean resetStats()', async () => {
  let mockNow = 0;
  const gates = new Map();

  const rangeCache = {
    read(filePath, offset, length) {
      const gate = gates.get(filePath);
      if (gate) return gate.promise;
      return Promise.resolve(makeBytes(4, length));
    },
  };

  const gCrit = deferred();
  const gHigh = deferred();
  const gNormal = deferred();
  const gPrefetch = deferred();
  gates.set('crit.bin', gCrit);
  gates.set('high.bin', gHigh);
  gates.set('normal.bin', gNormal);
  gates.set('prefetch.bin', gPrefetch);

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 3,
    agingWindowMs: 200,
    now: () => mockNow,
    sleep: async () => {},
  });

  // Fill all 3 concurrency slots across CRITICAL, HIGH, and NORMAL; queue PREFETCH at t = 0.
  const pCrit = scheduler.scheduleRead('crit.bin', 0, 8, { priority: IO_PRIORITY.CRITICAL });
  const pHigh = scheduler.scheduleRead('high.bin', 0, 8, { priority: IO_PRIORITY.HIGH });
  const pNormal = scheduler.scheduleRead('normal.bin', 0, 8, { priority: IO_PRIORITY.NORMAL });
  const pPrefetch = scheduler.scheduleRead('prefetch.bin', 0, 8, {
    priority: IO_PRIORITY.PREFETCH,
  });

  const activeSnap = scheduler.getStats();
  assert.deepEqual(activeSnap.activeByPriority, {
    CRITICAL: 1,
    HIGH: 1,
    NORMAL: 1,
    PREFETCH: 0,
  });
  assert.equal(activeSnap.inFlight, 3);
  assert.equal(activeSnap.queued, 1);

  // Snapshot isolation on activeByPriority & promotedByAgingCount.
  activeSnap.activeByPriority.CRITICAL = 999;
  activeSnap.promotedByAgingCount = 999;
  assert.equal(scheduler.getStats().activeByPriority.CRITICAL, 1);
  assert.equal(scheduler.getStats().promotedByAgingCount, 0);

  // At t = 450 (2+ aging windows), enqueue a fresh HIGH task and release NORMAL slot ->
  // aged PREFETCH (eff HIGH, enq 0) promotes ahead of fresh HIGH (enq 450).
  mockNow = 450;
  const pHighFresh = scheduler.scheduleRead('high-fresh.bin', 0, 8, {
    priority: IO_PRIORITY.HIGH,
  });
  gNormal.resolve(makeBytes(3, 8));
  await pNormal;
  await Promise.resolve();

  const afterPromotion = scheduler.getStats();
  assert.deepEqual(afterPromotion.activeByPriority, {
    CRITICAL: 1,
    HIGH: 1,
    NORMAL: 0,
    PREFETCH: 1,
  });
  assert.equal(afterPromotion.promotedByAgingCount, 1);
  assert.equal(afterPromotion.maxQueueWaitMs, 450);

  // Mid-flight resetStats(): resets counters (promotedByAgingCount, maxQueueWaitMs) cleanly
  // while keeping live activeByPriority accurate so in-flight release never underflows.
  scheduler.resetStats();
  const midReset = scheduler.getStats();
  assert.equal(midReset.promotedByAgingCount, 0);
  assert.equal(midReset.maxQueueWaitMs, 0);
  assert.deepEqual(midReset.activeByPriority, {
    CRITICAL: 1,
    HIGH: 1,
    NORMAL: 0,
    PREFETCH: 1,
  });

  // Drain all remaining tasks at t = 600 (high-fresh waited 150ms).
  mockNow = 600;
  gCrit.resolve(makeBytes(1, 8));
  gHigh.resolve(makeBytes(2, 8));
  gPrefetch.resolve(makeBytes(4, 8));
  await Promise.all([pCrit, pHigh, pPrefetch, pHighFresh]);

  const afterDrain = scheduler.getStats();
  assert.deepEqual(afterDrain.activeByPriority, {
    CRITICAL: 0,
    HIGH: 0,
    NORMAL: 0,
    PREFETCH: 0,
  });
  assert.equal(afterDrain.inFlight, 0);
  assert.equal(afterDrain.maxQueueWaitMs, 150);

  // Idle resetStats() zeroes everything cleanly.
  scheduler.resetStats();
  const idleReset = scheduler.getStats();
  assert.equal(idleReset.promotedByAgingCount, 0);
  assert.equal(idleReset.maxQueueWaitMs, 0);
  assert.deepEqual(idleReset.activeByPriority, {
    CRITICAL: 0,
    HIGH: 0,
    NORMAL: 0,
    PREFETCH: 0,
  });
});
