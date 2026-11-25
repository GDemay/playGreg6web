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

const { IO_PRIORITY, IoPriorityScheduler, isRetryableError } = schedulerMod;
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
