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

const ioWorkerMod = require('../web/workers/io_worker.js');
const mailboxMod = require('../web/workers/httpfs_mailbox.js');

const {
  DEFAULT_SLAB_SIZE,
  DEFAULT_SLAB_MAX_BYTES,
  MAX_SLAB_CACHE_BYTES,
  DEFAULT_SLAB_PREFETCH_AHEAD,
  DEFAULT_AUTO_PREFETCH_MIN_SLAB_SIZE,
  DEFAULT_SEQUENTIAL_THRESHOLD_READS,
  SLAB_TUNING_PROFILES,
  ImmutableHttpfsCache,
  IoWorkerController,
  emptyStats,
  emptyImmutableCacheStats,
} = ioWorkerMod;

const {
  createHttpfsMailboxBuffer,
  createHttpfsMailboxPool,
  HttpfsMailboxClient,
} = mailboxMod;

function makePatternBytes(length, seed = 41) {
  const out = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = ((i * 67) ^ ((i >> 6) * 23) ^ ((i >> 12) * 11) ^ seed) & 0xff;
  }
  return out;
}

function createDeferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function createControlledRangeFetch(urlBytesMap, options = {}) {
  const requests = [];
  const delayMs = options.delayMs || 0;
  const onRangeStart = typeof options.onRangeStart === 'function' ? options.onRangeStart : null;
  const etags = options.etags || new Map();

  async function fetchImpl(url, init = {}) {
    const method = (init && init.method) || 'GET';
    const rangeHeader = (init && init.headers && (init.headers.Range || init.headers.range)) || null;
    const reqRecord = { method, url, rangeHeader };
    requests.push(reqRecord);

    if (rangeHeader && onRangeStart) {
      await onRangeStart(reqRecord);
    }
    if (delayMs > 0) {
      await new Promise((r) => setTimeout(r, delayMs));
    }

    const data = urlBytesMap.get(url);
    if (!data) {
      return {
        ok: false,
        status: 404,
        headers: new Map(),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
        async json() {
          throw new Error('HTTP 404');
        },
      };
    }

    const headers = new Map();
    if (etags.has(url)) {
      headers.set('ETag', etags.get(url));
    }

    if (!rangeHeader) {
      headers.set('Content-Length', String(data.byteLength));
      return {
        ok: true,
        status: 200,
        headers,
        async arrayBuffer() {
          return data.slice().buffer;
        },
        async json() {
          return JSON.parse(new TextDecoder().decode(data));
        },
      };
    }

    const match = /^bytes=(\d+)-(\d+)$/.exec(rangeHeader);
    assert.ok(match, `Unexpected Range header: ${rangeHeader}`);
    const reqStart = Number(match[1]);
    const reqEnd = Number(match[2]);
    if (reqStart >= data.byteLength) {
      headers.set('Content-Range', `bytes */${data.byteLength}`);
      return {
        ok: false,
        status: 416,
        headers,
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }

    const actualEnd = Math.min(reqEnd, data.byteLength - 1);
    const slice = data.slice(reqStart, actualEnd + 1);
    headers.set('Content-Range', `bytes ${reqStart}-${actualEnd}/${data.byteLength}`);
    headers.set('Content-Length', String(slice.byteLength));
    return {
      ok: true,
      status: 206,
      headers,
      async arrayBuffer() {
        return slice.buffer;
      },
    };
  }

  return { fetchImpl, requests };
}

// ---------------------------------------------------------------------------
// 1. W356 defaults, configurable RAM LRU budget (up to 256 MiB) & aliases
// ---------------------------------------------------------------------------

test('W356: defaults enable +1 sequential prefetch on production slabs and support blockSize / maxRamCacheBytes / fast-boot profile', () => {
  assert.equal(DEFAULT_SLAB_SIZE, 2 * 1024 * 1024);
  assert.equal(DEFAULT_SLAB_MAX_BYTES, 192 * 1024 * 1024);
  assert.equal(MAX_SLAB_CACHE_BYTES, 256 * 1024 * 1024);
  assert.equal(DEFAULT_SLAB_PREFETCH_AHEAD, 1);
  assert.equal(DEFAULT_AUTO_PREFETCH_MIN_SLAB_SIZE, 512 * 1024);
  assert.equal(DEFAULT_SEQUENTIAL_THRESHOLD_READS, 3);
  assert.equal(SLAB_TUNING_PROFILES['fast-boot'].slabMaxBytes, 256 * 1024 * 1024);
  assert.equal(SLAB_TUNING_PROFILES['fast-boot'].prefetchAheadSlabs, 1);

  // Default production cache: 2 MiB aligned slabs, 192 MiB LRU cap, +1 sequential prefetch
  const defaultCache = new ImmutableHttpfsCache();
  assert.equal(defaultCache.slabSize, 2 * 1024 * 1024);
  assert.equal(defaultCache.slabMaxBytes, 192 * 1024 * 1024);
  assert.equal(defaultCache.prefetchAheadSlabs, 1);
  assert.equal(defaultCache.sequentialThresholdReads, 3);

  // Configurable 1 MiB aligned blocks + 256 MiB RAM LRU via blockSize / maxRamCacheBytes aliases
  const custom1m = new ImmutableHttpfsCache({
    blockSize: 1024 * 1024,
    maxRamCacheBytes: 256 * 1024 * 1024,
  });
  assert.equal(custom1m.slabSize, 1024 * 1024);
  assert.equal(custom1m.slabMaxBytes, 256 * 1024 * 1024);
  assert.equal(custom1m.prefetchAheadSlabs, 1);

  // Configurable 512 KiB aligned blocks also enable +1 sequential prefetch
  const custom512k = new ImmutableHttpfsCache({
    blockSize: 512 * 1024,
    maxRamCacheBytes: 256 * 1024 * 1024,
  });
  assert.equal(custom512k.slabSize, 512 * 1024);
  assert.equal(custom512k.prefetchAheadSlabs, 1);

  // Explicit opt-out via prefetchAheadSlabs: 0 and fast-boot profile
  const noPrefetch = new ImmutableHttpfsCache({ prefetchAheadSlabs: 0 });
  assert.equal(noPrefetch.prefetchAheadSlabs, 0);

  const fastBoot = new ImmutableHttpfsCache({ slabProfile: 'fast-boot' });
  assert.equal(fastBoot.slabMaxBytes, 256 * 1024 * 1024);
  assert.equal(fastBoot.prefetchAheadSlabs, 1);

  // StorageManager recommendedSlabMaxBytes forwarding (ignores 0, caps at 256 MiB)
  const recCache = new ImmutableHttpfsCache({ recommendedSlabMaxBytes: 256 * 1024 * 1024 });
  assert.equal(recCache.slabMaxBytes, 256 * 1024 * 1024);
  const zeroRecCache = new ImmutableHttpfsCache({ recommendedSlabMaxBytes: 0 });
  assert.equal(zeroRecCache.slabMaxBytes, DEFAULT_SLAB_MAX_BYTES);
});

// ---------------------------------------------------------------------------
// 2. W356: 8-lane concurrent in-flight deduplication (zero duplicate HTTP 206)
// ---------------------------------------------------------------------------

test('W356: 8 concurrent mailbox lanes + overlapping controller.read() coalesce onto strictly ONE in-flight HTTP 206 per slab', async () => {
  const slabSize = 1024 * 1024; // 1 MiB aligned blocks
  const fileSize = 3 * slabSize; // 3 MiB core_bvh4.rpf
  const bvhBytes = makePatternBytes(fileSize, 0x5c);

  const slab1Gate = createDeferred();
  const slab1Started = createDeferred();
  let slab1FetchCount = 0;

  const mock = createControlledRangeFetch(new Map([['/game/core_bvh4.rpf', bvhBytes]]), {
    async onRangeStart(req) {
      if (req.url === '/game/core_bvh4.rpf' && req.rangeHeader === `bytes=${slabSize}-${slabSize * 2 - 1}`) {
        slab1FetchCount += 1;
        slab1Started.resolve();
        await slab1Gate.promise;
      }
    },
  });

  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    blockSize: slabSize,
    maxRamCacheBytes: 256 * 1024 * 1024,
    prefetchAheadSlabs: 0,
    opfsDir: null,
  });

  const poolSabs = createHttpfsMailboxPool(8, 64 * 1024);
  const clients = poolSabs.map((sab) => new HttpfsMailboxClient(sab));

  try {
    await controller.init({ dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' });
    await controller.attachMailboxPool(poolSabs);

    // Open handle on lane 0 (probes slab 0 once)
    const handle = await clients[0].openAsync('G:\\core_bvh4.rpf');
    assert.ok(handle >= 1);
    const reqsAfterOpen = mock.requests.length;

    // Fire 8 concurrent mailbox reads across all 8 lanes targeting DIFFERENT offsets inside uncached slab 1!
    const laneBuffers = Array.from({ length: 8 }, () => new Uint8Array(32 * 1024));
    const laneOffsets = Array.from({ length: 8 }, (_, i) => slabSize + i * 64 * 1024);
    const lanePromises = clients.map((client, i) =>
      client.readAsync(handle, BigInt(laneOffsets[i]), laneBuffers[i], 32 * 1024),
    );

    // Wait until slab 1 HTTP 206 fetch is actively in-flight
    await slab1Started.promise;

    // Also fire concurrent controller.read() and controller.readBatch() overlapping the SAME in-flight slab 1!
    const directReadPromise = controller.read('G:\\core_bvh4.rpf', slabSize + 12345, 4096);
    const batchReadPromise = controller.readBatch([
      { id: 'b0', path: '/game/core_bvh4.rpf', offset: slabSize + 50000, length: 4096 },
    ]);

    // Give microtasks a tick so all 10 callers attach to the in-flight Promise
    await new Promise((r) => setTimeout(r, 10));
    assert.equal(slab1FetchCount, 1, 'Expected strictly 1 HTTP 206 fetch in-flight for slab 1');

    // Release slab 1 response
    slab1Gate.resolve();

    const [laneLengths, directBytes, batchRes] = await Promise.all([
      Promise.all(lanePromises),
      directReadPromise,
      batchReadPromise,
    ]);

    for (let i = 0; i < 8; i += 1) {
      assert.equal(laneLengths[i], 32 * 1024);
      assert.deepEqual(
        laneBuffers[i],
        bvhBytes.subarray(laneOffsets[i], laneOffsets[i] + 32 * 1024),
      );
    }
    assert.deepEqual(directBytes, bvhBytes.subarray(slabSize + 12345, slabSize + 12345 + 4096));
    assert.equal(batchRes.ok, true);
    assert.deepEqual(
      batchRes.results[0].bytes,
      bvhBytes.subarray(slabSize + 50000, slabSize + 50000 + 4096),
    );

    // Total new HTTP 206 requests for 10 simultaneous readers across 8 lanes + read + readBatch: STRICTLY 1!
    assert.equal(
      mock.requests.length - reqsAfterOpen,
      1,
      `Expected strictly 1 HTTP 206 fetch for all 10 concurrent slab-1 readers, got ${mock.requests.length - reqsAfterOpen}`,
    );

    const stats = controller.getStats();
    assert.ok(
      stats.inflightCoalesced >= 9,
      `Expected inflightCoalesced >= 9 across 10 concurrent callers, got ${stats.inflightCoalesced}`,
    );
    assert.equal(stats.immutableCache.inflightCoalesced, stats.inflightCoalesced);
    assert.equal(stats.immutableCache.inFlightCoalesced, stats.inflightCoalesced);
    assert.equal(stats.bytesFetched, 2 * slabSize); // slab 0 (open) + slab 1 (shared)

    assert.equal(await clients[0].closeAsync(handle), true);
  } finally {
    await controller.detachMailbox();
  }
});

// ---------------------------------------------------------------------------
// 3. W356: Sequential +1 slab background prefetch, prefetchHits & priority promotion
// ---------------------------------------------------------------------------

test('W356: sequential 64 KiB reads on 4 MiB data.rpf auto-prefetch +1 slab in background and report prefetchHits', async () => {
  const slabSize = 1024 * 1024; // 1 MiB aligned blocks (4 slabs = 64 x 64 KiB reads)
  const fileSize = 4 * slabSize;
  const dataRpf = makePatternBytes(fileSize, 0x7a);

  const mock = createControlledRangeFetch(new Map([['/game/data.rpf', dataRpf]]), {
    delayMs: 2,
  });

  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    blockSize: slabSize,
    maxRamCacheBytes: 256 * 1024 * 1024,
    opfsDir: null,
  });

  const sab = createHttpfsMailboxBuffer(64 * 1024);
  const client = new HttpfsMailboxClient(sab);

  try {
    await controller.init({ dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' });
    await controller.attachMailbox(sab);

    assert.equal(controller.immutableCache.prefetchAheadSlabs, 1);

    const reqsAfterInit = mock.requests.length;
    const h = await client.openAsync('G:\\data.rpf');
    assert.ok(h >= 1);

    // Read all 4 MiB sequentially in 64 KiB steps (64 reads)
    const assembled = new Uint8Array(fileSize);
    const stepBuf = new Uint8Array(64 * 1024);
    for (let off = 0; off < fileSize; off += 64 * 1024) {
      const n = await client.readAsync(h, BigInt(off), stepBuf, 64 * 1024);
      assert.equal(n, 64 * 1024);
      assembled.set(stepBuf, off);
      // Allow background prefetch to complete before hitting next slab boundary
      if ((off + 64 * 1024) % slabSize === 0) {
        await controller.immutableCache.waitForPrefetches();
      }
    }
    await controller.immutableCache.waitForPrefetches();

    assert.deepEqual(assembled, dataRpf);

    // 64 x 64 KiB guest reads -> strictly 4 aligned 1 MiB HTTP 206 requests (16x request reduction)!
    const gameReqs = mock.requests.slice(reqsAfterInit);
    assert.equal(gameReqs.length, 4, `Expected 4 x 1 MiB fetches for 4 MiB file, got ${gameReqs.length}`);

    const stats = controller.getStats();
    assert.equal(stats.prefetchHits, 3, `Expected 3 prefetchHits (slabs 1, 2, 3), got ${stats.prefetchHits}`);
    assert.equal(stats.immutableCache.prefetchHits, 3);
    assert.equal(stats.immutableCache.slabPrefetchesScheduled, 3);
    assert.equal(stats.immutableCache.slabPrefetchesCompleted, 3);
    assert.equal(stats.slabCacheMisses, 1, 'Only slab 0 on open() should be a demand miss');
    assert.equal(stats.bytesFetched, fileSize);
    assert.equal(stats.bytesServedFromCache, fileSize);
    assert.ok(stats.syncFastPathHits >= 64, `Expected all 64 reads on sync fast path, got ${stats.syncFastPathHits}`);
    assert.equal(await client.closeAsync(h), true);
  } finally {
    await controller.detachMailbox();
  }
});

test('W356: demand read coalescing onto a queued background prefetch promotes priority and counts prefetchHits once', async () => {
  const slabSize = 512 * 1024; // 512 KiB blocks
  const fileSize = 4 * slabSize;
  const rpfBytes = makePatternBytes(fileSize, 0x2b);

  const gateSlab1 = createDeferred();
  const slab1Started = createDeferred();
  const fetchOrder = [];

  const mock = createControlledRangeFetch(new Map([['/game/promo.rpf', rpfBytes]]), {
    async onRangeStart(req) {
      fetchOrder.push(req.rangeHeader);
      if (req.rangeHeader === `bytes=${slabSize}-${slabSize * 2 - 1}`) {
        slab1Started.resolve();
        await gateSlab1.promise;
      }
    },
  });

  // maxConcurrentSlabFetches: 1 so slab 2 (prefetch) and slab 3 (prefetch) queue behind slab 1!
  const cache = new ImmutableHttpfsCache({
    fetchImpl: mock.fetchImpl,
    slabSize,
    slabMaxBytes: 4 * slabSize,
    prefetchAheadSlabs: 2,
    maxConcurrentSlabFetches: 1,
    sequentialThresholdReads: 1,
  });

  // Trigger slab 1 prefetch (occupies the single slot) + slab 2 prefetch (queued #1) + slab 3 prefetch (queued #2)
  const p1 = cache._fetchSlab('/game/promo.rpf', 1, { isPrefetch: true });
  await slab1Started.promise;
  const p2 = cache._fetchSlab('/game/promo.rpf', 2, { isPrefetch: true });
  const p3 = cache._fetchSlab('/game/promo.rpf', 3, { isPrefetch: true });

  // Demand read for slab 3 MUST coalesce onto p3 AND promote slab 3 ahead of queued prefetch slab 2!
  const demandSlab3 = cache.read('/game/promo.rpf', 3 * slabSize, 4096);
  gateSlab1.resolve();

  const [b3] = await Promise.all([demandSlab3, p1, p2, p3]);
  assert.deepEqual(b3, rpfBytes.subarray(3 * slabSize, 3 * slabSize + 4096));

  // Verify promoted slab 3 was fetched BEFORE unpromoted prefetch slab 2!
  assert.deepEqual(fetchOrder, [
    `bytes=${slabSize}-${slabSize * 2 - 1}`,
    `bytes=${3 * slabSize}-${4 * slabSize - 1}`,
    `bytes=${2 * slabSize}-${3 * slabSize - 1}`,
  ]);

  // Subsequent read of slab 3 must NOT double-count prefetchHits
  const again = await cache.read('/game/promo.rpf', 3 * slabSize + 4096, 4096);
  assert.deepEqual(again, rpfBytes.subarray(3 * slabSize + 4096, 3 * slabSize + 8192));

  const st = cache.getStats();
  assert.equal(st.inflightCoalesced, 1);
  assert.equal(st.prefetchHits, 1, 'Promoted in-flight prefetch must count prefetchHits strictly once');
});

// ---------------------------------------------------------------------------
// 4. W356: Zero redundant copies on cache hit (direct SAB write, zero slice())
// ---------------------------------------------------------------------------

test('W356: warm single-slab, cross-slab, and whole-file SAB mailbox hits perform ZERO Uint8Array.prototype.slice() allocations', async () => {
  const slabSize = 512 * 1024; // 512 KiB slabs
  const fileSize = 2 * slabSize; // 1 MiB file
  const rpfBytes = makePatternBytes(fileSize, 0x91);
  const pgimBytes = makePatternBytes(128 * 1024, 0x44);

  const mock = createControlledRangeFetch(
    new Map([
      ['/game/zero_copy.rpf', rpfBytes],
      ['/boot/main/image/guest.pgim', pgimBytes],
    ]),
  );

  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    slabSize,
    slabMaxBytes: 4 * slabSize,
    prefetchAheadSlabs: 0,
    opfsDir: null,
  });

  const sab = createHttpfsMailboxBuffer(64 * 1024);
  const client = new HttpfsMailboxClient(sab);

  const origSlice = Uint8Array.prototype.slice;
  let sliceCalls = 0;

  try {
    await controller.init({ dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' });
    await controller.attachMailbox(sab);

    const hRpf = await client.openAsync('G:\\zero_copy.rpf');
    const hPgim = await client.openAsync('boot/main/image/guest.pgim');
    assert.ok(hRpf >= 1 && hPgim >= 1);

    // Warm slab 0 (already loaded by open) and slab 1
    const warmBuf = new Uint8Array(64 * 1024);
    assert.equal(await client.readAsync(hRpf, BigInt(slabSize), warmBuf, 64 * 1024), 64 * 1024);

    // Instrument Uint8Array.prototype.slice across warm single-slab, cross-slab, and whole-file reads!
    Uint8Array.prototype.slice = function instrumentedSlice(...args) {
      sliceCalls += 1;
      return origSlice.apply(this, args);
    };

    try {
      // 1. Single-slab warm hit
      const singleOut = new Uint8Array(64 * 1024);
      const n1 = await client.readAsync(hRpf, 4096n, singleOut, 64 * 1024);
      assert.equal(n1, 64 * 1024);
      assert.deepEqual(singleOut, rpfBytes.subarray(4096, 4096 + 64 * 1024));

      // 2. Cross-slab warm hit straddling slab 0 -> slab 1 boundary
      const crossOff = slabSize - 16384;
      const crossOut = new Uint8Array(64 * 1024);
      const n2 = await client.readAsync(hRpf, BigInt(crossOff), crossOut, 64 * 1024);
      assert.equal(n2, 64 * 1024);
      assert.deepEqual(crossOut, rpfBytes.subarray(crossOff, crossOff + 64 * 1024));

      // 3. Whole-file buffered (.pgim) warm hit
      const pgimOut = new Uint8Array(64 * 1024);
      const n3 = await client.readAsync(hPgim, 16384n, pgimOut, 64 * 1024);
      assert.equal(n3, 64 * 1024);
      assert.deepEqual(pgimOut, pgimBytes.subarray(16384, 16384 + 64 * 1024));

      assert.equal(
        sliceCalls,
        0,
        `Expected ZERO Uint8Array.prototype.slice() calls on warm SAB cache hits, got ${sliceCalls}`,
      );
    } finally {
      Uint8Array.prototype.slice = origSlice;
    }

    // Verify bytesServedFromCache tracked all 3 warm reads (3 x 64 KiB)
    const stats = controller.getStats();
    assert.equal(stats.bytesServedFromCache, 3 * 64 * 1024);
    assert.equal(stats.immutableCache.bytesServedFromCache, 2 * 64 * 1024);
    assert.equal(stats.wholeFileCache.bytesServedFromCache, 64 * 1024);

    assert.equal(await client.closeAsync(hRpf), true);
    assert.equal(await client.closeAsync(hPgim), true);
  } finally {
    Uint8Array.prototype.slice = origSlice;
    await controller.detachMailbox();
  }
});

// ---------------------------------------------------------------------------
// 5. W356: Sanitized telemetry & OPFS bypass switch (#1427 opfsL2Enabled: false)
// ---------------------------------------------------------------------------

test('W356: telemetry exposes sanitized cacheHits/cacheMisses/inflightCoalesced/prefetchHits/bytesFetched/bytesServedFromCache and honors opfsL2Enabled=false (#1427)', async () => {
  // 1. Baseline zeroed stats
  const eTop = emptyStats();
  const eImm = emptyImmutableCacheStats();
  for (const field of [
    'cacheHits',
    'cacheMisses',
    'inflightCoalesced',
    'prefetchHits',
    'bytesFetched',
    'bytesServedFromCache',
  ]) {
    assert.equal(eTop[field], 0, `emptyStats().${field} must default to 0`);
    assert.equal(eTop.immutableCache[field], 0, `emptyStats().immutableCache.${field} must default to 0`);
    assert.equal(eImm[field], 0, `emptyImmutableCacheStats().${field} must default to 0`);
  }

  // 2. Verify opfsL2Enabled: false (#1427) completely bypasses OPFS L2 even with an OPFS dir attached
  const slabSize = 512 * 1024;
  const rpfBytes = makePatternBytes(2 * slabSize, 0x3e);
  let opfsDirTouched = 0;
  const guardedOpfsDir = {
    async getFileHandle() {
      opfsDirTouched += 1;
      throw new Error('OPFS must not be touched when opfsL2Enabled=false');
    },
    async *keys() {
      opfsDirTouched += 1;
    },
    async *entries() {
      opfsDirTouched += 1;
    },
  };

  const mock = createControlledRangeFetch(
    new Map([['/game/core_bvh4.rpf', rpfBytes]]),
    { etags: new Map([['/game/core_bvh4.rpf', '"secret-etag-v1"']]) },
  );

  const posted = [];
  const workerSelf = {
    onmessage: null,
    postMessage(msg) {
      posted.push(msg);
    },
  };

  class MockSyncXHR {
    open(_method, url) {
      this._url = url;
    }
    send() {
      const resolved = path.resolve(repoRoot, 'web/workers', this._url);
      this.status = 200;
      this.responseText = fs.readFileSync(resolved, 'utf8');
    }
  }

  const sandbox = {
    self: workerSelf,
    globalThis: null,
    fetch: mock.fetchImpl,
    XMLHttpRequest: MockSyncXHR,
    navigator: {},
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
    AbortController,
    Map,
    Set,
    WeakSet,
    Promise,
    Error,
    Number,
    Math,
    Object,
    Array,
    Date,
    setTimeout,
    clearTimeout,
    importScripts(scriptPath) {
      const resolved = path.resolve(repoRoot, 'web/workers', scriptPath);
      vm.runInContext(fs.readFileSync(resolved, 'utf8'), context, { filename: resolved });
    },
  };
  sandbox.globalThis = sandbox;
  const context = vm.createContext(sandbox);
  const ioWorkerPath = path.join(repoRoot, 'web/workers/io_worker.js');
  vm.runInContext(fs.readFileSync(ioWorkerPath, 'utf8'), context, { filename: ioWorkerPath });

  await workerSelf.onmessage({
    data: {
      init: true,
      dataBase: '/data/',
      gameBase: '/game/',
      blockSize: slabSize,
      maxRamCacheBytes: 256 * 1024 * 1024,
      opfsDir: guardedOpfsDir,
      opfsL2Enabled: false,
    },
  });

  const sab = createHttpfsMailboxBuffer(64 * 1024);
  await workerSelf.onmessage({ data: { attachMailbox: true, id: 1, sab } });

  const client = new HttpfsMailboxClient(sab);
  try {
    const h = await client.openAsync('G:\\core_bvh4.rpf');
    assert.ok(h >= 1);

    // Sequential 64 KiB reads across first 3 chunks trigger +1 prefetch of slab 1
    const buf = new Uint8Array(64 * 1024);
    for (let i = 0; i < 4; i += 1) {
      assert.equal(await client.readAsync(h, BigInt(i * 64 * 1024), buf, 64 * 1024), 64 * 1024);
      assert.deepEqual(buf, rpfBytes.subarray(i * 64 * 1024, (i + 1) * 64 * 1024));
    }
    await sandbox.PlayGta6IoWorker.getDefaultController().immutableCache.waitForPrefetches();
    // Read into slab 1 -> hits prefetched slab 1!
    assert.equal(await client.readAsync(h, BigInt(slabSize), buf, 64 * 1024), 64 * 1024);
    assert.deepEqual(buf, rpfBytes.subarray(slabSize, slabSize + 64 * 1024));
    assert.equal(await client.closeAsync(h), true);

    assert.equal(opfsDirTouched, 0, 'opfsL2Enabled=false (#1427) must never touch OPFS');

    await workerSelf.onmessage({ data: { stats: true, id: 2 } });
    const statsMsg = posted.find((m) => m.id === 2);
    assert.ok(statsMsg && statsMsg.stats);
    const s = statsMsg.stats;

    assert.ok(s.cacheHits >= 5);
    assert.equal(s.cacheMisses, 1);
    assert.equal(s.prefetchHits, 1);
    assert.equal(s.bytesFetched, 2 * slabSize);
    assert.equal(s.bytesServedFromCache, 5 * 64 * 1024);
    assert.equal(s.opfsL2Hits, 0);
    assert.equal(s.opfsL2Writes, 0);

    // Sanitization: zero paths, filenames, URLs, or ETags in stats JSON
    const serialized = JSON.stringify(s);
    assert.equal(serialized.includes('core_bvh4.rpf'), false);
    assert.equal(serialized.includes('/game/'), false);
    assert.equal(serialized.includes('secret-etag-v1'), false);
  } finally {
    await workerSelf.onmessage({ data: { detachMailbox: true, id: 3 } });
  }
});

// ---------------------------------------------------------------------------
// 6. Scan-resistant 2-segment SLRU: hot shaders.rpf survives sequential levels.rpf scan
// ---------------------------------------------------------------------------

test('SLRU protects reused hot shaders.rpf slabs against a 24-slab 64 KiB sequential scan of levels.rpf', async () => {
  const slabSize = 256 * 1024; // 256 KiB slabs
  const shadersBytes = makePatternBytes(4 * slabSize, 0x11);
  const levelsBytes = makePatternBytes(24 * slabSize, 0x22);

  const mock = createControlledRangeFetch(
    new Map([
      ['/game/shaders.rpf', shadersBytes],
      ['/game/levels.rpf', levelsBytes],
    ]),
  );

  // Budget fits 8 slabs (6 Protected max + 2 Probation).
  const cache = new ImmutableHttpfsCache({
    fetchImpl: mock.fetchImpl,
    slabSize,
    slabMaxBytes: 8 * slabSize,
    prefetchAheadSlabs: 1,
    opfsDir: null,
  });

  // Warm 4 hot shaders.rpf slabs and revisit them across slab switches so they promote to Protected.
  for (let pass = 0; pass < 2; pass += 1) {
    for (let s = 0; s < 4; s += 1) {
      const out = await cache.read('/game/shaders.rpf', s * slabSize + 1024, 4096);
      assert.deepEqual(out, shadersBytes.subarray(s * slabSize + 1024, s * slabSize + 5120));
    }
  }
  const stAfterWarm = cache.getStats();
  assert.equal(stAfterWarm.slabProtectedEntries, 4);
  assert.ok(stAfterWarm.slabProtectedPromotions >= 4);

  // Stream sequentially through 24 slabs of levels.rpf in 64 KiB reads (3x total cache capacity!).
  for (let off = 0; off < levelsBytes.byteLength; off += 64 * 1024) {
    const out = await cache.read('/game/levels.rpf', off, 64 * 1024);
    assert.deepEqual(out, levelsBytes.subarray(off, off + 64 * 1024));
    assert.ok(cache.getStats().slabBytesCached <= 8 * slabSize);
  }
  await cache.waitForPrefetches();

  // Every single hot shaders.rpf slab MUST still be resident in RAM (0 network refetches)!
  const reqsBeforeShaderRecheck = mock.requests.length;
  for (let s = 0; s < 4; s += 1) {
    const out = await cache.read('/game/shaders.rpf', s * slabSize + 2048, 4096);
    assert.deepEqual(out, shadersBytes.subarray(s * slabSize + 2048, s * slabSize + 6144));
  }
  assert.equal(
    mock.requests.length,
    reqsBeforeShaderRecheck,
    'Hot Protected shaders.rpf slabs must survive sequential levels.rpf scan with 0 refetches',
  );
});

// ---------------------------------------------------------------------------
// 7. 8-lane out-of-order sequential frontier prefetch, tight forward gap & seek cancellation
// ---------------------------------------------------------------------------

test('8-lane out-of-order sequential reads trigger frontier prefetch, while distant forward jumps do not falsely prefetch and cancel queued prefetches', async () => {
  const slabSize = 512 * 1024;
  const seqFile = makePatternBytes(4 * slabSize, 0x55);
  const randFile = makePatternBytes(12 * slabSize, 0x66);

  const mock = createControlledRangeFetch(
    new Map([
      ['/game/seq8.rpf', seqFile],
      ['/game/rand.rpf', randFile],
    ]),
  );

  const cache = new ImmutableHttpfsCache({
    fetchImpl: mock.fetchImpl,
    slabSize,
    slabMaxBytes: 8 * slabSize,
    prefetchAheadSlabs: 1,
    sequentialThresholdReads: 3,
    opfsDir: null,
  });

  // 8-lane out-of-order sequential arrivals inside slab 0 (lanes arrive 64K, 0, 192K, 128K, 256K).
  const oooOffsets = [64 * 1024, 0, 192 * 1024, 128 * 1024, 256 * 1024];
  for (const off of oooOffsets) {
    const out = await cache.read('/game/seq8.rpf', off, 64 * 1024);
    assert.deepEqual(out, seqFile.subarray(off, off + 64 * 1024));
  }
  await cache.waitForPrefetches();
  assert.ok(
    cache.getStats().slabPrefetchesCompleted >= 1,
    'Out-of-order 8-lane sequential reads must still trigger +1 frontier slab prefetch',
  );

  // Random forward jumps spaced 400 KiB apart (> 256 KiB maxForwardGap) must NOT falsely prefetch!
  const prefBeforeRand = cache.getStats().slabPrefetchesScheduled;
  for (let i = 0; i < 6; i += 1) {
    const off = i * 400 * 1024;
    await cache.read('/game/rand.rpf', off, 4096);
  }
  await cache.waitForPrefetches();
  assert.equal(
    cache.getStats().slabPrefetchesScheduled,
    prefBeforeRand,
    'Random forward jumps > 256 KiB apart must not falsely schedule sequential prefetch',
  );

  // Distant random seek cancels still-queued prefetch waiters for that URL.
  const gate = createDeferred();
  const started = createDeferred();
  const seekFile = makePatternBytes(8 * slabSize, 0x77);
  const gatedMock = createControlledRangeFetch(new Map([['/game/seek.rpf', seekFile]]), {
    async onRangeStart(req) {
      if (req.rangeHeader === `bytes=${slabSize}-${2 * slabSize - 1}`) {
        started.resolve();
        await gate.promise;
      }
    },
  });

  const seekCache = new ImmutableHttpfsCache({
    fetchImpl: gatedMock.fetchImpl,
    slabSize,
    slabMaxBytes: 8 * slabSize,
    prefetchAheadSlabs: 2,
    maxConcurrentSlabFetches: 1,
    sequentialThresholdReads: 2,
    opfsDir: null,
  });

  // Warm slab 0 + slab 7 so the later distant seek into slab 7 hits RAM and cancels queued prefetch slab 2.
  await seekCache.read('/game/seek.rpf', 7 * slabSize, 4096);
  await seekCache.read('/game/seek.rpf', 0, 64 * 1024);
  // Second sequential read in slab 0 triggers prefetch of slab 1 (in-flight) + slab 2 (queued).
  await seekCache.read('/game/seek.rpf', 64 * 1024, 64 * 1024);
  await started.promise;

  // Distant random seek back to cached slab 7 cancels queued prefetch for slab 2!
  await seekCache.read('/game/seek.rpf', 7 * slabSize + 4096, 4096);
  gate.resolve();
  await seekCache.waitForPrefetches();

  assert.ok(
    seekCache.getStats().slabPrefetchCancelled >= 1,
    'Distant random seek should cancel still-queued prefetch waiters for that URL',
  );
  assert.equal(
    gatedMock.requests.some((r) => r.rangeHeader === `bytes=${2 * slabSize}-${3 * slabSize - 1}`),
    false,
    'Cancelled queued prefetch slab 2 must never be fetched over HTTP',
  );
});

// ---------------------------------------------------------------------------
// 8. Bounded memory invariants: OPFS L2 pending write bytes, whole-file total cap & memory-pressure reclaim
// ---------------------------------------------------------------------------

test('Enforces maxPendingOpfsL2WriteBytes, wholeFileTotalMaxBytes aggregate cap, and memory-pressure reclaim preserving Protected slabs', async () => {
  const slabSize = 256 * 1024;
  const rpfBytes = makePatternBytes(8 * slabSize, 0x33);
  const pgimA = makePatternBytes(128 * 1024, 0x41);
  const pgimB = makePatternBytes(128 * 1024, 0x42);

  const mock = createControlledRangeFetch(
    new Map([
      ['/game/shaders.rpf', rpfBytes],
      ['/boot/a.pgim', pgimA],
      ['/boot/b.pgim', pgimB],
    ]),
  );

  // Slow OPFS mock to keep write queue populated while testing byte cap.
  const writeGate = createDeferred();
  let writesStarted = 0;
  const slowOpfsL2 = {
    async readRange() {
      return null;
    },
    async writeRange() {
      writesStarted += 1;
      await writeGate.promise;
      return { stored: true };
    },
  };

  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    slabSize,
    slabMaxBytes: 8 * slabSize,
    prefetchAheadSlabs: 0,
    opfsL2: slowOpfsL2,
    maxOpfsL2WriteConcurrency: 1,
    maxPendingOpfsL2WriteBytes: 2 * slabSize, // Queue cap: at most 2 slabs (512 KiB)
    wholeFileMaxBytes: 128 * 1024,
    wholeFileTotalMaxBytes: 160 * 1024, // Fits 1 x 128 KiB pgim, second must fall back cleanly!
  });

  await controller.init({ dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' });

  // Whole-file aggregate cap: a.pgim buffers whole-file, b.pgim exceeds remaining total cap and falls back.
  const hA = await controller.openHandle('boot/a.pgim');
  const hB = await controller.openHandle('boot/b.pgim');
  assert.equal(controller.getHandleInfo(hA).wholeFileBuffered, true);
  assert.equal(controller.getHandleInfo(hB).wholeFileBuffered, false);
  assert.deepEqual(await controller.readHandle(hB, 0, 4096), pgimB.subarray(0, 4096));
  assert.ok(controller.getStats().wholeFileCache.activeBytes <= 160 * 1024);
  controller.closeHandle(hA);
  controller.closeHandle(hB);

  // Trigger 6 cold slab reads -> 1 active OPFS write + queue capped at <= 2 * slabSize bytes!
  for (let s = 0; s < 6; s += 1) {
    await controller.immutableCache.read('/game/shaders.rpf', s * slabSize, 4096);
  }
  const stQueue = controller.immutableCache.getStats();
  assert.ok(stQueue.opfsL2QueuedBytes <= 2 * slabSize, `Expected queued bytes <= ${2 * slabSize}, got ${stQueue.opfsL2QueuedBytes}`);
  assert.ok(stQueue.opfsL2WriteDrops >= 1, 'Expected byte-cap backpressure drops on OPFS L2 write queue');

  // Promote slabs 0 and 1 to Protected before triggering memory-pressure reclaim.
  await controller.immutableCache.read('/game/shaders.rpf', 0, 4096);
  await controller.immutableCache.read('/game/shaders.rpf', slabSize, 4096);
  assert.ok(controller.immutableCache.getStats().slabProtectedEntries >= 2);

  // Send memory-pressure worker message trimming to 2 slabs (512 KiB) while preserving slabSize and Protected slabs.
  const posted = [];
  controller.postMessage = (m) => posted.push(m);
  await controller.handleMessage({
    id: 99,
    type: 'memory-pressure',
    targetSlabBytes: 2 * slabSize,
    dropQueuedOpfsWrites: true,
  });
  const reply = posted.find((m) => m.id === 99);
  assert.ok(reply && reply.ok === true && reply.bytesFreed >= 4 * slabSize);
  const stAfterReclaim = controller.immutableCache.getStats();
  assert.equal(stAfterReclaim.slabSize, slabSize, 'reclaimMemory must preserve slabSize');
  assert.ok(stAfterReclaim.slabBytesCached <= 2 * slabSize);
  assert.equal(stAfterReclaim.slabProtectedEntries, 2, 'Protected slabs must be retained while Probation slabs are reclaimed first');
  assert.equal(stAfterReclaim.opfsL2QueuedBytes, 0);

  writeGate.resolve();
  await controller.flushOpfsL2Writes();
  assert.ok(writesStarted >= 1);
});

// ---------------------------------------------------------------------------
// 9. OPFS L2 warm reload: HEAD-without-ETag probe, SyncAccessHandle-first read, proactive budget & read-priority gating
// ---------------------------------------------------------------------------

test('OPFS L2 preserves warm slabs when HEAD omits ETag, prefers SyncAccessHandle on read, enforces opfsL2MaxBytes proactively, and gates writes behind demand fetches', async () => {
  const slabSize = 256 * 1024;
  const rpfBytes = makePatternBytes(4 * slabSize, 0x88);
  const realEtag = '"etag-warm-v1"';

  let syncReadCount = 0;
  let getFileReadCount = 0;
  const files = new Map();
  const mockOpfsDir = {
    files,
    async *keys() {
      for (const k of files.keys()) yield k;
    },
    async removeEntry(name) {
      files.delete(name);
    },
    async getFileHandle(name, opts = {}) {
      if (!files.has(name)) {
        if (!opts.create) {
          const err = new Error(`NotFoundError: ${name}`);
          err.name = 'NotFoundError';
          throw err;
        }
        files.set(name, new Uint8Array(0));
      }
      return {
        async getFile() {
          getFileReadCount += 1;
          const data = files.get(name);
          return {
            size: data.byteLength,
            async arrayBuffer() {
              return data.slice().buffer;
            },
          };
        },
        async createSyncAccessHandle() {
          let buf = (files.get(name) || new Uint8Array(0)).slice();
          return {
            getSize() {
              return buf.byteLength;
            },
            truncate(n) {
              buf = buf.slice(0, n);
            },
            write(view, { at = 0 } = {}) {
              const end = at + view.byteLength;
              if (end > buf.byteLength) {
                const grown = new Uint8Array(end);
                grown.set(buf, 0);
                buf = grown;
              }
              buf.set(view, at);
              return view.byteLength;
            },
            read(view, { at = 0 } = {}) {
              syncReadCount += 1;
              const n = Math.min(view.byteLength, Math.max(0, buf.byteLength - at));
              view.set(buf.subarray(at, at + n), 0);
              return n;
            },
            flush() {
              files.set(name, buf.slice());
            },
            close() {},
          };
        },
      };
    },
  };

  // Custom fetch: HEAD returns Content-Length ONLY (omits ETag); Range GET returns ETag.
  let rangeGets = 0;
  let zeroByteProbes = 0;
  const customFetch = async (url, init = {}) => {
    const method = (init.method || 'GET').toUpperCase();
    if (method === 'HEAD') {
      return {
        ok: true,
        status: 200,
        headers: new Map([['Content-Length', String(rpfBytes.byteLength)]]),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }
    const rangeHeader = init.headers && (init.headers.Range || init.headers.range);
    if (rangeHeader === 'bytes=0-0') {
      zeroByteProbes += 1;
      return {
        ok: true,
        status: 206,
        headers: new Map([
          ['ETag', realEtag],
          ['Content-Range', `bytes 0-0/${rpfBytes.byteLength}`],
        ]),
        async arrayBuffer() {
          return rpfBytes.slice(0, 1).buffer;
        },
      };
    }
    rangeGets += 1;
    const m = /^bytes=(\d+)-(\d+)$/.exec(rangeHeader || '');
    const s = m ? Number(m[1]) : 0;
    const e = m ? Math.min(Number(m[2]), rpfBytes.byteLength - 1) : rpfBytes.byteLength - 1;
    const slice = rpfBytes.slice(s, e + 1);
    return {
      ok: true,
      status: 206,
      headers: new Map([
        ['ETag', realEtag],
        ['Content-Range', `bytes ${s}-${e}/${rpfBytes.byteLength}`],
      ]),
      async arrayBuffer() {
        return slice.buffer;
      },
    };
  };

  // Cold session: write 3 slabs with opfsL2MaxBytes = 2 * slabSize -> proactive eviction keeps disk <= 2 slabs!
  const cold = new ImmutableHttpfsCache({
    fetchImpl: customFetch,
    slabSize,
    slabMaxBytes: 4 * slabSize,
    prefetchAheadSlabs: 0,
    opfsDir: mockOpfsDir,
    opfsL2MaxBytes: 2 * slabSize,
  });

  await cold.read('/game/shaders.rpf', 0, 4096);
  await cold.flushOpfsL2Writes();
  await cold.read('/game/shaders.rpf', slabSize, 4096);
  await cold.flushOpfsL2Writes();
  await cold.read('/game/shaders.rpf', 2 * slabSize, 4096);
  await cold.flushOpfsL2Writes();

  assert.ok(files.size <= 2, `Expected proactive opfsL2MaxBytes eviction to keep <= 2 OPFS files, got ${files.size}`);
  assert.ok(cold.getStats().opfsL2ProactiveEvictions >= 1);

  // Warm session: HEAD omits ETag -> probes bytes=0-0 -> recovers real ETag -> hits warm OPFS via SyncAccessHandle (0 full slab GETs!).
  const rangeGetsBeforeWarm = rangeGets;
  const warm = new ImmutableHttpfsCache({
    fetchImpl: customFetch,
    slabSize,
    slabMaxBytes: 4 * slabSize,
    prefetchAheadSlabs: 0,
    opfsDir: mockOpfsDir,
    opfsL2MaxBytes: 2 * slabSize,
  });

  const warmData = await warm.read('/game/shaders.rpf', 2 * slabSize, 4096);
  assert.deepEqual(warmData, rpfBytes.subarray(2 * slabSize, 2 * slabSize + 4096));
  assert.equal(zeroByteProbes, 1, 'Expected single bytes=0-0 ETag probe when HEAD omitted ETag');
  assert.equal(rangeGets, rangeGetsBeforeWarm, 'Warm OPFS slab must NOT be invalidated or refetched over HTTP');
  assert.equal(warm.getStats().opfsL2Hits, 1);
  assert.equal(warm.getStats().opfsL2Invalidations, 0);
  assert.ok(syncReadCount >= 1, 'Expected createSyncAccessHandle fast path to be preferred on OPFS read');
  assert.equal(getFileReadCount, 0, 'getFile() should not be called when createSyncAccessHandle succeeds');

  // Read-priority gating: background OPFS L2 write waits while a demand slab fetch is active, then pumps automatically once demand finishes.
  const demandGate = createDeferred();
  const demandStarted = createDeferred();
  let opfsWriteCount = 0;
  const gatingStore = {
    async readRange() {
      return null;
    },
    async writeRange() {
      opfsWriteCount += 1;
      return { stored: true };
    },
  };
  const gatingMock = createControlledRangeFetch(new Map([['/game/shaders.rpf', rpfBytes]]), {
    async onRangeStart(req) {
      if (req.rangeHeader === `bytes=${slabSize}-${2 * slabSize - 1}`) {
        demandStarted.resolve();
        await demandGate.promise;
      }
    },
  });
  const gatedCache = new ImmutableHttpfsCache({
    fetchImpl: gatingMock.fetchImpl,
    slabSize,
    slabMaxBytes: 4 * slabSize,
    prefetchAheadSlabs: 0,
    opfsL2: gatingStore,
  });
  // Start a long-running demand fetch on slab 1, then finish slab 0 while slab 1 is still active.
  const pSlab1 = gatedCache.read('/game/shaders.rpf', slabSize, 4096);
  await demandStarted.promise;
  await gatedCache.read('/game/shaders.rpf', 0, 4096);
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(
    opfsWriteCount,
    0,
    'Background OPFS L2 write-through must defer while another demand slab fetch is active',
  );
  demandGate.resolve();
  await pSlab1;
  await gatedCache.flushOpfsL2Writes();
  assert.equal(opfsWriteCount, 2, 'Deferred OPFS L2 writes must pump automatically once demand fetches finish');
});
