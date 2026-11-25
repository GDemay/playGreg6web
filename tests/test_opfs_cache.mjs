import test from 'node:test';
import assert from 'node:assert/strict';

import {
  CHUNK_KINDS,
  DEFAULT_INDEX_FILENAME,
  DEFAULT_INDEX_FLUSH_DELAY_MS,
  DEFAULT_MAX_BYTES,
  DEFAULT_MAX_OPEN_SYNC_HANDLES,
  DEFAULT_MAX_PENDING_WRITES,
  DEFAULT_MAX_WRITE_CONCURRENCY,
  DEFAULT_OPFS_CACHE_DIR,
  DEFAULT_PREFETCH_CONCURRENCY,
  OPFS_CACHE_INDEX_VERSION,
  OpfsCache,
  OpfsChunkCache,
  buildFileFingerprint,
  inferChunkKind,
  makeChunkKey,
  validateChunkKey,
} from '../web/storage/opfs_cache.js';

function makePatternBytes(length, seed = 17) {
  const out = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = (i * 31 + seed) & 0xff;
  }
  return out;
}

/**
 * In-memory OPFS FileSystemDirectoryHandle mock with write counters, optional
 * QuotaExceededError injection, and optional async write delay hooks.
 */
function createMockOpfsDirHandle(name = '', initialEntries = null, hooks = {}) {
  const entries = new Map();
  const writeCounts = new Map();
  let totalWrites = 0;

  if (initialEntries) {
    for (const [k, v] of Object.entries(initialEntries)) {
      entries.set(k, v instanceof Uint8Array ? v.slice() : v);
    }
  }

  const handle = {
    kind: 'directory',
    name,
    _entries: entries,
    _writeCounts: writeCounts,
    get totalWrites() {
      return totalWrites;
    },
    getWriteCount(fileName) {
      return writeCounts.get(fileName) || 0;
    },
    async getDirectoryHandle(dirName, opts = {}) {
      const existing = entries.get(dirName);
      if (existing && existing.kind === 'directory') {
        return existing;
      }
      if (!opts.create) {
        const err = new Error(`NotFoundError: ${dirName}`);
        err.name = 'NotFoundError';
        throw err;
      }
      const child = createMockOpfsDirHandle(dirName, null, hooks);
      entries.set(dirName, child);
      return child;
    },
    async getFileHandle(fileName, opts = {}) {
      if (!entries.has(fileName)) {
        if (!opts.create) {
          const err = new Error(`NotFoundError: ${fileName}`);
          err.name = 'NotFoundError';
          throw err;
        }
        entries.set(fileName, new Uint8Array(0));
      }
      return {
        kind: 'file',
        name: fileName,
        async getFile() {
          const data = entries.get(fileName);
          if (!data || data.kind === 'directory') {
            const err = new Error(`NotFoundError: ${fileName}`);
            err.name = 'NotFoundError';
            throw err;
          }
          return {
            size: data.byteLength,
            lastModified: 1000,
            async arrayBuffer() {
              return data.slice().buffer;
            },
          };
        },
        async createWritable() {
          let pending = new Uint8Array(0);
          return {
            async write(chunk) {
              if (typeof hooks.onBeforeWrite === 'function') {
                await hooks.onBeforeWrite(fileName, chunk);
              }
              pending =
                chunk instanceof Uint8Array
                  ? chunk.slice()
                  : chunk instanceof ArrayBuffer
                    ? new Uint8Array(chunk.slice(0))
                    : new TextEncoder().encode(String(chunk));
            },
            async close() {
              entries.set(fileName, pending);
              totalWrites += 1;
              writeCounts.set(fileName, (writeCounts.get(fileName) || 0) + 1);
            },
            async abort() {
              pending = new Uint8Array(0);
            },
          };
        },
      };
    },
    async *entries() {
      for (const [entryName, value] of entries.entries()) {
        if (value && value.kind === 'directory') {
          yield [entryName, value];
        } else {
          yield [
            entryName,
            {
              kind: 'file',
              name: entryName,
              async getFile() {
                return {
                  size: value.byteLength,
                  lastModified: 1000,
                  async arrayBuffer() {
                    return value.slice().buffer;
                  },
                };
              },
            },
          ];
        }
      }
    },
    async removeEntry(entryName, opts = {}) {
      if (!entries.has(entryName)) {
        const err = new Error(`NotFoundError: ${entryName}`);
        err.name = 'NotFoundError';
        throw err;
      }
      const target = entries.get(entryName);
      if (target && target.kind === 'directory' && target._entries.size > 0 && !opts.recursive) {
        const err = new Error(`InvalidModificationError: ${entryName} is not empty`);
        err.name = 'InvalidModificationError';
        throw err;
      }
      entries.delete(entryName);
    },
  };

  return handle;
}

function createFakeTimers() {
  let nextId = 1;
  const scheduled = new Map();
  return {
    scheduled,
    setTimeoutFn(fn, delay) {
      const id = nextId;
      nextId += 1;
      scheduled.set(id, { fn, delay });
      return id;
    },
    clearTimeoutFn(id) {
      scheduled.delete(id);
    },
    async runAll() {
      const items = Array.from(scheduled.entries());
      scheduled.clear();
      for (const [, item] of items) {
        item.fn();
      }
      // Let microtasks from async flushIndex() settle.
      await new Promise((resolve) => setTimeout(resolve, 10));
    },
  };
}

test('exports defaults, validates keys, infers chunk kinds, and builds deterministic descriptor keys', () => {
  assert.equal(DEFAULT_OPFS_CACHE_DIR, 'playgta6-cache');
  assert.equal(DEFAULT_INDEX_FILENAME, '.playgta6-cache-index.json');
  assert.equal(OPFS_CACHE_INDEX_VERSION, 1);
  assert.equal(DEFAULT_INDEX_FLUSH_DELAY_MS, 0);
  assert.equal(DEFAULT_PREFETCH_CONCURRENCY, 4);
  assert.equal(DEFAULT_MAX_BYTES, 512 * 1024 * 1024);
  assert.equal(OpfsCache, OpfsChunkCache);

  assert.equal(validateChunkKey('chunks/a.bin'), 'chunks/a.bin');
  assert.throws(() => validateChunkKey(''), /non-empty/);
  assert.throws(() => validateChunkKey('/leading.bin'), /relative path/);
  assert.throws(() => validateChunkKey('dir/../escape.bin'), /Invalid chunk key segment/);
  assert.throws(() => validateChunkKey('bad\\slash.bin'), /forbidden character/);
  assert.throws(() => validateChunkKey(DEFAULT_INDEX_FILENAME), /reserved/);

  const k1 = makeChunkKey({ path: '/assets/level0.rpf', chunkIndex: 3, chunkSize: 262144 });
  const k2 = makeChunkKey({ path: 'assets/level0.rpf', chunkIndex: 3, chunkSize: 262144 });
  assert.equal(k1, k2);
  assert.match(k1, /^c_262144_3_[0-9a-f]+\.bin$/);
  assert.equal(inferChunkKind(k1), CHUNK_KINDS.RANGE_CHUNK);

  const rangeKey = makeChunkKey({ path: 'movies/intro.bk2', offset: 1024, length: 4096 });
  assert.match(rangeKey, /^r_1024_4096_[0-9a-f]+\.bin$/);
  assert.equal(inferChunkKind(rangeKey), CHUNK_KINDS.RANGE_CHUNK);

  // Very long paths use deterministic 128-bit FNV-1a suffix.
  const longPath = 'a/'.repeat(60) + 'pack.rpf';
  const longKey1 = makeChunkKey({ path: longPath, chunkIndex: 0, chunkSize: 65536 });
  const longKey2 = makeChunkKey({ path: longPath, chunkIndex: 0, chunkSize: 65536 });
  assert.equal(longKey1, longKey2);
  assert.ok(longKey1.length < 130);

  assert.equal(inferChunkKind('boot.pgim'), CHUNK_KINDS.PGIM);
  assert.equal(inferChunkKind('shaders/main.wgsl.pack'), CHUNK_KINDS.SHADER_PACK);
  assert.equal(inferChunkKind('misc/data.dat'), CHUNK_KINDS.GENERIC);
});

test('default indexFlushDelayMs=0 persists index immediately on put/get/delete (backwards-compatible)', async () => {
  let clock = 1000;
  const cacheDir = createMockOpfsDirHandle(DEFAULT_OPFS_CACHE_DIR);
  const opfsRoot = createMockOpfsDirHandle('', { [DEFAULT_OPFS_CACHE_DIR]: cacheDir });

  const cache = new OpfsChunkCache({
    opfsRoot,
    nowFn: () => {
      clock += 10;
      return clock;
    },
  });

  await cache.put('chunk-a.bin', makePatternBytes(64, 1), { etag: 'v1' });
  assert.equal(cacheDir.getWriteCount(DEFAULT_INDEX_FILENAME), 1);
  assert.equal(cache.getStats().indexFlushes, 1);
  assert.equal(cache.getStats().coalescedFlushes, 0);
  assert.equal(cache.getStats().indexDirty, false);

  await cache.put('chunk-b.bin', makePatternBytes(128, 2), { etag: 'v2' });
  assert.equal(cacheDir.getWriteCount(DEFAULT_INDEX_FILENAME), 2);
  assert.equal(cache.getStats().indexFlushes, 2);

  // Verify persisted JSON schema matches version 1 contract.
  const rawIndex = cacheDir._entries.get(DEFAULT_INDEX_FILENAME);
  assert.ok(rawIndex instanceof Uint8Array);
  const parsed = JSON.parse(new TextDecoder().decode(rawIndex));
  assert.equal(parsed.version, OPFS_CACHE_INDEX_VERSION);
  assert.equal(parsed.entries.length, 2);
  assert.deepEqual(
    parsed.entries.map((e) => e.key),
    ['chunk-a.bin', 'chunk-b.bin']
  );

  // Hit touches chunk-a.bin and immediately flushes index when delay is 0.
  const got = await cache.get('chunk-a.bin', { etag: 'v1' });
  assert.deepEqual(got, makePatternBytes(64, 1));
  assert.equal(cacheDir.getWriteCount(DEFAULT_INDEX_FILENAME), 3);
  assert.deepEqual(cache.getLruKeys(), ['chunk-b.bin', 'chunk-a.bin']);

  // Deleting removes both OPFS file and metadata entry and persists immediately.
  const deleted = await cache.delete('chunk-b.bin');
  assert.equal(deleted, true);
  assert.equal(cache.has('chunk-b.bin'), false);
  assert.equal(cacheDir.getWriteCount(DEFAULT_INDEX_FILENAME), 4);
  assert.equal(cache.getStats().deletes, 1);

  await cache.close();
});

test('batched/coalesced chunk-index persistence with indexFlushDelayMs > 0 and explicit flushIndex()', async () => {
  const timers = createFakeTimers();
  let clock = 5000;
  const cacheDir = createMockOpfsDirHandle(DEFAULT_OPFS_CACHE_DIR);
  const opfsRoot = createMockOpfsDirHandle('', { [DEFAULT_OPFS_CACHE_DIR]: cacheDir });

  const cache = new OpfsChunkCache({
    opfsRoot,
    indexFlushDelayMs: 50,
    setTimeoutFn: timers.setTimeoutFn,
    clearTimeoutFn: timers.clearTimeoutFn,
    nowFn: () => {
      clock += 5;
      return clock;
    },
  });

  // Burst of 5 puts should schedule 1 debounced timer and coalesce 4 updates -> 0 OPFS index writes so far.
  for (let i = 0; i < 5; i += 1) {
    await cache.put(`batch-${i}.bin`, makePatternBytes(32, i));
  }
  assert.equal(cacheDir.getWriteCount(DEFAULT_INDEX_FILENAME), 0);
  assert.equal(timers.scheduled.size, 1);
  assert.equal(cache.getStats().indexFlushes, 0);
  assert.equal(cache.getStats().coalescedFlushes, 4);
  assert.equal(cache.getStats().indexDirty, true);

  // Explicit flushIndex() cancels the timer and persists all 5 entries in a single write.
  const flushRes = await cache.flushIndex();
  assert.equal(flushRes.flushed, true);
  assert.equal(flushRes.entryCount, 5);
  assert.equal(flushRes.totalBytes, 160);
  assert.equal(timers.scheduled.size, 0);
  assert.equal(cacheDir.getWriteCount(DEFAULT_INDEX_FILENAME), 1);
  assert.equal(cache.getStats().indexFlushes, 1);
  assert.equal(cache.getStats().indexDirty, false);

  // Calling flushIndex() again when clean is a no-op.
  const noopRes = await cache.flushIndex();
  assert.equal(noopRes.flushed, false);
  assert.equal(cacheDir.getWriteCount(DEFAULT_INDEX_FILENAME), 1);

  // Now test timer-driven flush after hits/touches.
  await cache.get('batch-0.bin');
  await cache.get('batch-1.bin');
  assert.equal(cacheDir.getWriteCount(DEFAULT_INDEX_FILENAME), 1);
  assert.equal(timers.scheduled.size, 1);
  assert.equal(cache.getStats().coalescedFlushes, 5);

  await timers.runAll();
  assert.equal(cacheDir.getWriteCount(DEFAULT_INDEX_FILENAME), 2);
  assert.equal(cache.getStats().indexFlushes, 2);
  assert.equal(cache.getStats().indexDirty, false);

  await cache.close();
});

test('O(1) LRU touch tracking orders evictions accurately and preserves order across warm reload', async () => {
  let clock = 10000;
  const cacheDir = createMockOpfsDirHandle(DEFAULT_OPFS_CACHE_DIR);
  const opfsRoot = createMockOpfsDirHandle('', { [DEFAULT_OPFS_CACHE_DIR]: cacheDir });

  const cache = new OpfsChunkCache({
    opfsRoot,
    maxBytes: 300,
    indexFlushDelayMs: 25,
    nowFn: () => {
      clock += 100;
      return clock;
    },
  });

  await cache.put('a.bin', makePatternBytes(100, 1));
  await cache.put('b.bin', makePatternBytes(100, 2), { pinned: true });
  await cache.put('c.bin', makePatternBytes(100, 3));
  assert.deepEqual(cache.getLruKeys(), ['a.bin', 'b.bin', 'c.bin']);

  // Touching 'a.bin' via get() moves it to MRU tail in O(1); 'c.bin' becomes oldest unpinned.
  const hitA = await cache.get('a.bin');
  assert.equal(hitA.byteLength, 100);
  assert.deepEqual(cache.getLruKeys(), ['b.bin', 'c.bin', 'a.bin']);

  // Adding 'd.bin' (100 B) exceeds maxBytes=300; 'b.bin' is pinned so 'c.bin' MUST be evicted!
  const putD = await cache.put('d.bin', makePatternBytes(100, 4));
  assert.equal(putD.evictedEntries, 1);
  assert.equal(putD.evictedBytes, 100);
  assert.equal(cache.has('c.bin'), false);
  assert.equal(cache.has('b.bin'), true);
  assert.deepEqual(cache.getLruKeys(), ['b.bin', 'a.bin', 'd.bin']);

  // Flush and simulate a warm browser reload with a brand-new OpfsChunkCache instance.
  await cache.close();

  const warmCache = new OpfsChunkCache({
    opfsRoot,
    maxBytes: 300,
    nowFn: () => {
      clock += 100;
      return clock;
    },
  });
  const initRes = await warmCache.init();
  assert.equal(initRes.loadedFromIndex, true);
  assert.equal(initRes.entryCount, 3);
  assert.equal(initRes.totalBytes, 300);
  assert.deepEqual(warmCache.getLruKeys(), ['b.bin', 'a.bin', 'd.bin']);

  // Touching 'a.bin' synchronously via touch() moves it after 'd.bin'.
  assert.equal(warmCache.touch('a.bin', { persist: false }), true);
  assert.equal(warmCache.touch('missing.bin'), false);
  assert.deepEqual(warmCache.getLruKeys(), ['b.bin', 'd.bin', 'a.bin']);

  await warmCache.close();
});

test('prefetchChunks enforces bounded concurrency and deduplicates already-cached & in-flight chunks', async () => {
  const timers = createFakeTimers();
  const cacheDir = createMockOpfsDirHandle(DEFAULT_OPFS_CACHE_DIR);
  const opfsRoot = createMockOpfsDirHandle('', { [DEFAULT_OPFS_CACHE_DIR]: cacheDir });

  const cache = new OpfsChunkCache({
    opfsRoot,
    indexFlushDelayMs: 50,
    setTimeoutFn: timers.setTimeoutFn,
    clearTimeoutFn: timers.clearTimeoutFn,
  });

  // Pre-seed two chunks to simulate warm-boot hits.
  await cache.put('chunk-0.bin', makePatternBytes(64, 0));
  await cache.put('chunk-1.bin', makePatternBytes(64, 1));
  await cache.flushIndex();
  cache.resetStats();

  let activeFetches = 0;
  let maxObservedConcurrency = 0;
  const fetchCallCounts = new Map();

  const fetcher = async (desc, { key }) => {
    activeFetches += 1;
    maxObservedConcurrency = Math.max(maxObservedConcurrency, activeFetches);
    fetchCallCounts.set(key, (fetchCallCounts.get(key) || 0) + 1);
    await new Promise((resolve) => setTimeout(resolve, 15));
    activeFetches -= 1;
    return {
      data: makePatternBytes(128, key.charCodeAt(key.length - 5)),
      etag: `etag-${key}`,
    };
  };

  // Batch includes:
  // - 2 already cached ('chunk-0.bin', 'chunk-1.bin')
  // - 6 uncached ('chunk-2.bin'..'chunk-7.bin')
  // - duplicate 'chunk-2.bin' in same batch
  const batchA = [
    'chunk-0.bin',
    'chunk-1.bin',
    'chunk-2.bin',
    'chunk-2.bin',
    'chunk-3.bin',
    'chunk-4.bin',
    'chunk-5.bin',
    'chunk-6.bin',
    'chunk-7.bin',
  ];

  // Concurrent overlapping prefetch call sharing 'chunk-3.bin' and 'chunk-4.bin' plus new 'chunk-8.bin'.
  const batchB = ['chunk-3.bin', 'chunk-4.bin', 'chunk-8.bin'];

  const [resA, resB] = await Promise.all([
    cache.prefetchChunks(batchA, fetcher, { concurrency: 3 }),
    cache.prefetchChunks(batchB, fetcher, { concurrency: 2 }),
  ]);

  // Every unique uncached chunk must be fetched strictly ONCE across both concurrent batches!
  assert.equal(fetchCallCounts.get('chunk-0.bin') || 0, 0);
  assert.equal(fetchCallCounts.get('chunk-1.bin') || 0, 0);
  for (let i = 2; i <= 8; i += 1) {
    assert.equal(
      fetchCallCounts.get(`chunk-${i}.bin`),
      1,
      `expected chunk-${i}.bin to be fetched exactly once`
    );
  }

  // Concurrency across batchA (3) + batchB (2) stays strictly bounded (<= 5 total, <= 3 for single batch).
  assert.ok(maxObservedConcurrency <= 5, `max concurrency was ${maxObservedConcurrency}`);
  assert.equal(resA.requested, 9);
  assert.equal(resB.requested, 3);
  assert.equal(resA.hits, 2);
  assert.equal(resA.fetched + resB.fetched, 7);
  assert.equal(resA.deduplicatedInFlight + resB.deduplicatedInFlight, 3);
  assert.equal(resA.failed, 0);
  assert.equal(resB.failed, 0);

  // Now test single-call concurrency cap strictly at concurrency = 2.
  activeFetches = 0;
  maxObservedConcurrency = 0;
  const strictBatch = ['s-0.bin', 's-1.bin', 's-2.bin', 's-3.bin', 's-4.bin', 's-5.bin'];
  const strictRes = await cache.prefetchChunks(strictBatch, fetcher, { concurrency: 2 });
  assert.equal(maxObservedConcurrency, 2);
  assert.equal(strictRes.fetched, 6);
  assert.equal(strictRes.bytesFetched, 6 * 128);

  // Verify batched index writes were coalesced instead of writing per chunk!
  const beforeFlushWrites = cacheDir.getWriteCount(DEFAULT_INDEX_FILENAME);
  assert.equal(beforeFlushWrites, 1); // Only the initial pre-seed flush!
  await cache.flushIndex();
  assert.equal(cacheDir.getWriteCount(DEFAULT_INDEX_FILENAME), 2);

  // Verify sanitized warm-boot telemetry exposed via getStats().
  const stats = cache.getStats();
  assert.equal(stats.indexFlushes, 1);
  assert.ok(stats.coalescedFlushes >= 14);
  assert.equal(stats.prefetchRequested, 9 + 3 + 6);
  assert.equal(stats.prefetchHits, 2);
  assert.equal(stats.prefetchFetched, 7 + 6);
  assert.equal(stats.prefetchInFlightDeduped, 3);
  assert.equal(stats.prefetchFailed, 0);
  assert.equal(stats.prefetchBytes, 13 * 128);
  assert.equal(stats.inFlightCount, 0);
  assert.equal(stats.hitRate, 2 / (2 + 13));

  // Verify getStats() contains strictly sanitized primitives (no keys/paths/payloads).
  for (const [k, v] of Object.entries(stats)) {
    assert.ok(
      typeof v === 'number' || typeof v === 'boolean' || v === null,
      `telemetry field ${k} must be primitive number/boolean/null, got ${typeof v}`
    );
  }

  await cache.close();
});

test('prefetchChunks handles fetcher errors, AbortSignal, and ETag/version revalidation cleanly', async () => {
  const cacheDir = createMockOpfsDirHandle(DEFAULT_OPFS_CACHE_DIR);
  const opfsRoot = createMockOpfsDirHandle('', { [DEFAULT_OPFS_CACHE_DIR]: cacheDir });
  const cache = new OpfsChunkCache({ opfsRoot, indexFlushDelayMs: 0 });

  await cache.put('stale.bin', makePatternBytes(40, 1), { etag: 'old-etag', versionKey: 'v1' });

  // Requesting 'stale.bin' with a new etag should bypass the hit and re-fetch!
  let fetchedKeys = [];
  const res = await cache.prefetchChunks(
    [
      { key: 'stale.bin', etag: 'new-etag', versionKey: 'v2' },
      { key: 'boom.bin' },
      { key: 'empty.bin' },
    ],
    async (desc, { key }) => {
      fetchedKeys.push(key);
      if (key === 'boom.bin') {
        throw new Error('NetworkTimeout');
      }
      if (key === 'empty.bin') {
        return null;
      }
      return makePatternBytes(50, 9);
    },
    { concurrency: 2 }
  );

  assert.deepEqual(fetchedKeys.sort(), ['boom.bin', 'empty.bin', 'stale.bin']);
  assert.equal(res.fetched, 1);
  assert.equal(res.failed, 2);
  assert.equal(cache.getStats().prefetchFailed, 2);
  assert.equal(cache.getStats().invalidations, 1);

  const updated = await cache.getEntry('stale.bin', { etag: 'new-etag', versionKey: 'v2' });
  assert.equal(updated.hit, true);
  assert.equal(updated.data.byteLength, 50);

  // Stale etag on getEntry invalidates and removes the file.
  const mismatch = await cache.getEntry('stale.bin', { etag: 'wrong-etag' });
  assert.equal(mismatch.hit, false);
  assert.equal(mismatch.invalidated, true);
  assert.equal(mismatch.reason, 'etag-mismatch');
  assert.equal(cache.has('stale.bin'), false);

  // AbortSignal stops dispatching remaining items.
  const ac = new AbortController();
  ac.abort();
  const abortRes = await cache.prefetchChunks(['x1.bin', 'x2.bin'], async () => makePatternBytes(8), {
    signal: ac.signal,
  });
  assert.equal(abortRes.aborted, true);
  assert.equal(abortRes.fetched, 0);

  await cache.close();
});

test('QuotaExceededError recovery, disk reconciliation on init, and graceful unsupported OPFS fallback', async () => {
  let failOnceOnBigFile = true;
  const cacheDir = createMockOpfsDirHandle(DEFAULT_OPFS_CACHE_DIR, null, {
    onBeforeWrite(fileName) {
      if (fileName === 'big.bin' && failOnceOnBigFile) {
        failOnceOnBigFile = false;
        const err = new Error('QuotaExceededError: storage full');
        err.name = 'QuotaExceededError';
        throw err;
      }
    },
  });
  const opfsRoot = createMockOpfsDirHandle('', { [DEFAULT_OPFS_CACHE_DIR]: cacheDir });

  const cache = new OpfsChunkCache({ opfsRoot, maxBytes: null });
  await cache.put('old-1.bin', makePatternBytes(100, 1));
  await cache.put('old-2.bin', makePatternBytes(100, 2));

  const bigRes = await cache.put('big.bin', makePatternBytes(150, 3));
  assert.equal(bigRes.stored, true);
  assert.equal(bigRes.quotaRecovered, true);
  assert.ok(bigRes.evictedEntries >= 1);
  assert.equal(cache.has('big.bin'), true);

  // Externally add an untracked file and delete an indexed file, then reconcile on reload.
  cacheDir._entries.set('discovered.bin', makePatternBytes(44, 7));
  cacheDir._entries.delete('big.bin');

  const reloaded = new OpfsChunkCache({ opfsRoot });
  const initRes = await reloaded.init();
  assert.equal(initRes.discoveredCount, 1);
  assert.equal(initRes.prunedCount, 1);
  assert.equal(reloaded.has('discovered.bin'), true);
  assert.equal(reloaded.has('big.bin'), false);

  const cleared = await reloaded.clear();
  assert.equal(cleared.supported, true);
  assert.equal(reloaded.getStats().entryCount, 0);
  assert.equal(reloaded.getStats().totalBytes, 0);

  // Graceful fallback when OPFS is unavailable (opfsRoot: null).
  const unsupported = new OpfsChunkCache({ opfsRoot: null });
  const uInit = await unsupported.init();
  assert.equal(uInit.supported, false);
  const uPut = await unsupported.put('a.bin', makePatternBytes(16));
  assert.equal(uPut.supported, false);
  assert.equal(uPut.stored, false);
  assert.equal(await unsupported.get('a.bin'), null);
  assert.equal(unsupported.getStats().supported, false);
  assert.equal(unsupported.getStats().hitRate, 0);
});

test('concurrent index flushes serialize and coalesce without overlapping OPFS index writes', async () => {
  let activeIndexWrites = 0;
  let maxConcurrentIndexWrites = 0;

  const cacheDir = createMockOpfsDirHandle(DEFAULT_OPFS_CACHE_DIR, null, {
    async onBeforeWrite(fileName) {
      if (fileName === DEFAULT_INDEX_FILENAME) {
        activeIndexWrites += 1;
        maxConcurrentIndexWrites = Math.max(maxConcurrentIndexWrites, activeIndexWrites);
        await new Promise((resolve) => setTimeout(resolve, 10));
        activeIndexWrites -= 1;
      }
    },
  });
  const opfsRoot = createMockOpfsDirHandle('', { [DEFAULT_OPFS_CACHE_DIR]: cacheDir });

  // Even with indexFlushDelayMs=0, concurrent puts coalesce into serialized index writes.
  const cache = new OpfsChunkCache({ opfsRoot, indexFlushDelayMs: 0 });
  await Promise.all([
    cache.put('c1.bin', makePatternBytes(16, 1)),
    cache.put('c2.bin', makePatternBytes(16, 2)),
    cache.put('c3.bin', makePatternBytes(16, 3)),
    cache.put('c4.bin', makePatternBytes(16, 4)),
    cache.put('c5.bin', makePatternBytes(16, 5)),
  ]);

  assert.equal(maxConcurrentIndexWrites, 1);
  assert.ok(cacheDir.getWriteCount(DEFAULT_INDEX_FILENAME) < 5);
  assert.ok(cache.getStats().coalescedFlushes >= 1);
  assert.equal(cache.getStats().indexDirty, false);
  assert.equal(cache.getStats().entryCount, 5);

  // Reading with touch: false does not mutate LRU order or trigger an index write.
  const writesBefore = cacheDir.getWriteCount(DEFAULT_INDEX_FILENAME);
  const lruBefore = cache.getLruKeys();
  const data = await cache.get(lruBefore[0], { touch: false });
  assert.equal(data.byteLength, 16);
  assert.deepEqual(cache.getLruKeys(), lruBefore);
  assert.equal(cacheDir.getWriteCount(DEFAULT_INDEX_FILENAME), writesBefore);

  await cache.close();
});

test('structured range descriptors prefetch, alphabetical index hydration, and O(1) LRU scale check', async () => {
  const cacheDir = createMockOpfsDirHandle(DEFAULT_OPFS_CACHE_DIR);
  const opfsRoot = createMockOpfsDirHandle('', { [DEFAULT_OPFS_CACHE_DIR]: cacheDir });

  // Seed an alphabetically-ordered index (matching BrowserStorageManager) with out-of-order accessOrder.
  cacheDir._entries.set('alpha.bin', makePatternBytes(10, 1));
  cacheDir._entries.set('beta.bin', makePatternBytes(10, 2));
  cacheDir._entries.set('gamma.bin', makePatternBytes(10, 3));
  cacheDir._entries.set(
    DEFAULT_INDEX_FILENAME,
    new TextEncoder().encode(
      JSON.stringify({
        version: OPFS_CACHE_INDEX_VERSION,
        entries: [
          { key: 'alpha.bin', sizeBytes: 10, lastAccessedMs: 300, createdAtMs: 100, accessOrder: 3 },
          { key: 'beta.bin', sizeBytes: 10, lastAccessedMs: 100, createdAtMs: 100, accessOrder: 1 },
          { key: 'gamma.bin', sizeBytes: 10, lastAccessedMs: 200, createdAtMs: 100, accessOrder: 2 },
        ],
      })
    )
  );

  const cache = new OpfsChunkCache({ opfsRoot, indexFlushDelayMs: 50 });
  await cache.init();
  assert.deepEqual(cache.getLruKeys(), ['beta.bin', 'gamma.bin', 'alpha.bin']);

  // Prefetch structured { path, chunkIndex, chunkSize } descriptors.
  const descriptors = [
    { path: 'archives/x64a.rpf', chunkIndex: 0, chunkSize: 65536 },
    { path: 'archives/x64a.rpf', chunkIndex: 1, chunkSize: 65536 },
    { path: 'archives/x64a.rpf', chunkIndex: 2, chunkSize: 65536 },
  ];
  const pf = await cache.prefetchChunks(
    descriptors,
    async (desc) => makePatternBytes(32, desc.chunkIndex),
    { concurrency: 2, flushOnComplete: true }
  );
  assert.equal(pf.fetched, 3);
  assert.equal(pf.bytesFetched, 96);
  assert.equal(cache.getStats().indexDirty, false);

  // Warm-boot re-prefetch of identical structured descriptors yields 100% cache hits.
  let fetcherCalled = 0;
  const pfWarm = await cache.prefetchChunks(
    descriptors,
    async () => {
      fetcherCalled += 1;
      return makePatternBytes(32, 99);
    },
    { concurrency: 2 }
  );
  assert.equal(fetcherCalled, 0);
  assert.equal(pfWarm.hits, 3);
  assert.equal(pfWarm.fetched, 0);

  // O(1) LRU touch integrity over 200 entries and 600 touches.
  for (let i = 0; i < 200; i += 1) {
    await cache.put(`scale/k_${i}.bin`, makePatternBytes(8, i));
  }
  for (let i = 0; i < 600; i += 1) {
    assert.equal(cache.touch(`scale/k_${i % 50}.bin`, { persist: false }), true);
  }
  const lruAfter = cache.getLruKeys();
  assert.equal(lruAfter.length, cache.getStats().entryCount);
  assert.equal(lruAfter[lruAfter.length - 1], 'scale/k_49.bin');

  await cache.close();
});

// --- W262: Versioned fingerprint keys, stale invalidation & non-blocking background write-through ---

test('W262: buildFileFingerprint, versioned makeChunkKey, and fingerprint metadata invalidation across warm reload', async () => {
  assert.equal(DEFAULT_MAX_PENDING_WRITES, 64);
  assert.equal(DEFAULT_MAX_WRITE_CONCURRENCY, 4);

  const fpSizeOnly = buildFileFingerprint(4096);
  const fpWithEtag = buildFileFingerprint(4096, '"shaders-v1"');
  const fpV2 = buildFileFingerprint(8192, '"shaders-v2"');
  assert.equal(fpSizeOnly, '4096');
  assert.match(fpWithEtag, /^4096_[0-9a-f]{16}$/);
  assert.notEqual(fpWithEtag, fpV2);

  // Versioned chunk keys stay classified as RANGE_CHUNK and remain backward-compatible without fingerprint.
  const versionedChunkKey = makeChunkKey({
    path: 'game/shaders.rpf',
    chunkIndex: 0,
    chunkSize: 262144,
    fileFingerprint: fpWithEtag,
  });
  assert.match(versionedChunkKey, /^c_262144_0_[0-9a-f]+_fp_4096_[0-9a-f]{16}\.bin$/);
  assert.equal(inferChunkKind(versionedChunkKey), CHUNK_KINDS.RANGE_CHUNK);

  const versionedRangeKey = makeChunkKey({
    path: 'game/movies.rpf',
    offset: 0,
    length: 65536,
    fileSize: 1048576,
    etag: '"movies-v1"',
  });
  assert.match(versionedRangeKey, /^r_0_65536_[0-9a-f]+_fp_1048576_[0-9a-f]{16}\.bin$/);
  assert.equal(inferChunkKind(versionedRangeKey), CHUNK_KINDS.RANGE_CHUNK);

  // Persist chunk with fileFingerprint, reload into fresh OpfsChunkCache, verify hit then stale invalidation.
  const cacheDir = createMockOpfsDirHandle(DEFAULT_OPFS_CACHE_DIR);
  const opfsRoot = createMockOpfsDirHandle('', { [DEFAULT_OPFS_CACHE_DIR]: cacheDir });

  const cold = new OpfsChunkCache({ opfsRoot, indexFlushDelayMs: 0 });
  const baseKey = makeChunkKey({ path: 'game/shaders.rpf', chunkIndex: 0, chunkSize: 16 });
  await cold.put(baseKey, makePatternBytes(16, 7), { fileFingerprint: fpWithEtag });
  assert.equal(cold.getMeta(baseKey).fileFingerprint, fpWithEtag);
  await cold.close();

  const warm = new OpfsChunkCache({ opfsRoot, indexFlushDelayMs: 0 });
  await warm.init();
  const hit = await warm.getEntry(baseKey, { fileFingerprint: fpWithEtag });
  assert.equal(hit.hit, true);
  assert.deepEqual(hit.data, makePatternBytes(16, 7));
  assert.equal(warm.getStats().opfsHits, 1);
  assert.equal(warm.getStats().opfsHitRate, 1);

  // Changing fileFingerprint invalidates the stale OPFS entry and increments opfsStaleInvalidations.
  const stale = await warm.getEntry(baseKey, { fileFingerprint: fpV2 });
  assert.equal(stale.hit, false);
  assert.equal(stale.invalidated, true);
  assert.equal(stale.reason, 'fingerprint-mismatch');
  assert.equal(warm.has(baseKey), false);
  assert.equal(warm.getStats().opfsStaleInvalidations, 1);
  assert.equal(warm.getStats().opfsMisses, 1);
  assert.equal(warm.getStats().opfsHitRate, 0.5);

  // Bulk stale invalidation by archive path descriptor removes outdated fingerprinted chunks.
  const kOld0 = makeChunkKey({ path: 'game/core_bvh4.rpf', chunkIndex: 0, chunkSize: 16, fileFingerprint: '100' });
  const kOld1 = makeChunkKey({ path: 'game/core_bvh4.rpf', chunkIndex: 1, chunkSize: 16, fileFingerprint: '100' });
  const kNew0 = makeChunkKey({ path: 'game/core_bvh4.rpf', chunkIndex: 0, chunkSize: 16, fileFingerprint: '200' });
  await warm.put(kOld0, makePatternBytes(16, 1), { fileFingerprint: '100' });
  await warm.put(kOld1, makePatternBytes(16, 2), { fileFingerprint: '100' });
  await warm.put(kNew0, makePatternBytes(16, 3), { fileFingerprint: '200' });

  const purged = await warm.invalidateStaleFingerprint({ path: 'game/core_bvh4.rpf' }, '200');
  assert.equal(purged.invalidatedCount, 2);
  assert.equal(purged.freedBytes, 32);
  assert.equal(warm.has(kOld0), false);
  assert.equal(warm.has(kOld1), false);
  assert.equal(warm.has(kNew0), true);
  assert.equal(warm.getStats().opfsStaleInvalidations, 3);

  await warm.close();
});

test('W262: putBackground provides non-blocking bounded write-through and degrades gracefully on quota/storage errors', async () => {
  let gateWrites = true;
  const pendingGates = [];
  let failAllWrites = false;

  const cacheDir = createMockOpfsDirHandle(DEFAULT_OPFS_CACHE_DIR, null, {
    async onBeforeWrite(fileName) {
      if (fileName === DEFAULT_INDEX_FILENAME) return;
      if (failAllWrites) {
        const err = new Error('QuotaExceededError: disk full');
        err.name = 'QuotaExceededError';
        throw err;
      }
      if (gateWrites) {
        await new Promise((resolve) => pendingGates.push(resolve));
      }
    },
  });
  const opfsRoot = createMockOpfsDirHandle('', { [DEFAULT_OPFS_CACHE_DIR]: cacheDir });

  const cache = new OpfsChunkCache({
    opfsRoot,
    indexFlushDelayMs: 0,
    maxPendingWrites: 2,
    maxWriteConcurrency: 1,
  });
  await cache.init();

  // Enqueue 4 writes synchronously while writes are gated: never blocks and bounds pending queue to 2!
  const r0 = cache.putBackground('bg-0.bin', makePatternBytes(16, 0), { fileFingerprint: '16' });
  const r1 = cache.putBackground('bg-1.bin', makePatternBytes(16, 1), { fileFingerprint: '16' });
  const r2 = cache.putBackground('bg-2.bin', makePatternBytes(16, 2), { fileFingerprint: '16' });
  const r3 = cache.putBackground('bg-3.bin', makePatternBytes(16, 3), { fileFingerprint: '16' });

  assert.equal(r0.queued, true);
  assert.equal(r1.queued, true);
  assert.equal(r2.queued, true);
  assert.equal(r3.queued, true);
  assert.equal(r3.droppedOldest, true);
  assert.equal(cache.getStats().opfsWrites, 0, 'putBackground must return before OPFS disk write completes');
  assert.ok(cache.getStats().pendingBackgroundWrites <= 3);

  // Release write gate and drain background writes.
  gateWrites = false;
  while (cache.getStats().pendingBackgroundWrites > 0) {
    if (pendingGates.length > 0) {
      pendingGates.shift()();
    }
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  await cache.flushBackgroundWrites();

  assert.equal(cache.getStats().opfsWrites, 3);
  assert.equal(cache.getStats().opfsWriteErrors, 0);
  assert.equal(cache.has('bg-0.bin'), true);
  assert.equal(cache.has('bg-1.bin'), false, 'Oldest queued entry bg-1.bin must be dropped when queue overflows');
  assert.equal(cache.has('bg-2.bin'), true);
  assert.equal(cache.has('bg-3.bin'), true);

  // Persistent QuotaExceededError in putBackground degrades gracefully without throwing and increments opfsWriteErrors.
  failAllWrites = true;
  cache.putBackground('bg-fail.bin', makePatternBytes(16, 9));
  await cache.flushBackgroundWrites();

  const stats = cache.getStats();
  assert.equal(stats.opfsWriteErrors, 1);
  assert.equal(stats.opfsWrites, 3);
  assert.ok(stats.opfsHitRate >= 0 && stats.opfsHitRate <= 1);

  // All telemetry fields remain strictly sanitized primitives.
  for (const [k, v] of Object.entries(stats)) {
    assert.ok(
      typeof v === 'number' || typeof v === 'boolean' || v === null,
      `telemetry field ${k} must be primitive number/boolean/null, got ${typeof v}`
    );
  }

  await cache.close();
});

// --- W313: Web Worker FileSystemSyncAccessHandle fast path, bounded handle pool & cleanup ---

function createSyncMockOpfsDirHandle(name = '', initialEntries = null, opts = {}, sharedState = null, prefix = '') {
  const state = sharedState || {
    activeOpenHandles: new Set(),
    openHandlesByPath: new Map(),
    totalOpenedHandles: 0,
    totalClosedHandles: 0,
    truncateCalls: 0,
    writeCalls: 0,
    flushCalls: 0,
    readCalls: 0,
    getSizeCalls: 0,
    createWritableCalls: 0,
    getFileCalls: 0,
    nextHandleId: 1,
    enforceLocks: opts.enforceLocks !== false,
  };
  const entries = new Map();
  const writeCounts = new Map();
  let totalWrites = 0;

  if (initialEntries) {
    for (const [k, v] of Object.entries(initialEntries)) {
      entries.set(k, v instanceof Uint8Array ? v.slice() : v);
    }
  }

  const relPathFor = (leaf) => (prefix ? `${prefix}/${leaf}` : leaf);

  const hasOpenHandleUnder = (targetPrefix) => {
    for (const openPath of state.openHandlesByPath.keys()) {
      if (openPath === targetPrefix || openPath.startsWith(`${targetPrefix}/`)) {
        return true;
      }
    }
    return false;
  };

  const handle = {
    kind: 'directory',
    name,
    _entries: entries,
    _writeCounts: writeCounts,
    _syncState: state,
    get totalWrites() {
      return totalWrites;
    },
    getWriteCount(fileName) {
      return writeCounts.get(fileName) || 0;
    },
    async getDirectoryHandle(dirName, dirOpts = {}) {
      const existing = entries.get(dirName);
      if (existing && existing.kind === 'directory') {
        return existing;
      }
      if (!dirOpts.create) {
        const err = new Error(`NotFoundError: ${dirName}`);
        err.name = 'NotFoundError';
        throw err;
      }
      const childPrefix = relPathFor(dirName);
      const child = createSyncMockOpfsDirHandle(dirName, null, opts, state, childPrefix);
      entries.set(dirName, child);
      return child;
    },
    async getFileHandle(fileName, fileOpts = {}) {
      if (!entries.has(fileName)) {
        if (!fileOpts.create) {
          const err = new Error(`NotFoundError: ${fileName}`);
          err.name = 'NotFoundError';
          throw err;
        }
        entries.set(fileName, new Uint8Array(0));
      }
      const relPath = relPathFor(fileName);
      const fileHandle = {
        kind: 'file',
        name: fileName,
        async getFile() {
          state.getFileCalls += 1;
          const data = entries.get(fileName);
          if (!data || data.kind === 'directory') {
            const err = new Error(`NotFoundError: ${fileName}`);
            err.name = 'NotFoundError';
            throw err;
          }
          return {
            size: data.byteLength,
            lastModified: 1000,
            async arrayBuffer() {
              return data.slice().buffer;
            },
          };
        },
        async createWritable() {
          state.createWritableCalls += 1;
          let pending = new Uint8Array(0);
          return {
            async write(chunk) {
              if (typeof opts.onBeforeWrite === 'function') {
                await opts.onBeforeWrite(relPath, chunk);
              }
              pending =
                chunk instanceof Uint8Array
                  ? chunk.slice()
                  : chunk instanceof ArrayBuffer
                    ? new Uint8Array(chunk.slice(0))
                    : new TextEncoder().encode(String(chunk));
            },
            async close() {
              entries.set(fileName, pending);
              totalWrites += 1;
              writeCounts.set(fileName, (writeCounts.get(fileName) || 0) + 1);
            },
            async abort() {
              pending = new Uint8Array(0);
            },
          };
        },
      };

      if (opts.exposeSyncAccessHandle !== false) {
        fileHandle.createSyncAccessHandle = async () => {
          if (typeof opts.onCreateSyncAccessHandle === 'function') {
            await opts.onCreateSyncAccessHandle(relPath);
          }
          if (state.enforceLocks && state.openHandlesByPath.has(relPath)) {
            const err = new Error(`NoModificationAllowedError: handle already open for ${relPath}`);
            err.name = 'NoModificationAllowedError';
            throw err;
          }
          const id = state.nextHandleId;
          state.nextHandleId += 1;
          let closed = false;
          let buffer =
            entries.get(fileName) instanceof Uint8Array
              ? entries.get(fileName).slice()
              : new Uint8Array(0);

          state.activeOpenHandles.add(id);
          state.openHandlesByPath.set(relPath, id);
          state.totalOpenedHandles += 1;

          const ensureOpen = () => {
            if (closed) {
              const err = new Error(`InvalidStateError: SyncAccessHandle ${id} is closed`);
              err.name = 'InvalidStateError';
              throw err;
            }
          };

          return {
            getSize() {
              ensureOpen();
              state.getSizeCalls += 1;
              return buffer.byteLength;
            },
            truncate(newSize) {
              ensureOpen();
              state.truncateCalls += 1;
              const target = Math.max(0, Math.floor(newSize));
              if (target === buffer.byteLength) return;
              const next = new Uint8Array(target);
              next.set(buffer.subarray(0, Math.min(target, buffer.byteLength)), 0);
              buffer = next;
            },
            write(view, writeOpts = {}) {
              ensureOpen();
              state.writeCalls += 1;
              const bytes =
                view instanceof Uint8Array
                  ? view
                  : ArrayBuffer.isView(view)
                    ? new Uint8Array(view.buffer, view.byteOffset, view.byteLength)
                    : new Uint8Array(view);
              if (typeof opts.onSyncWrite === 'function') {
                opts.onSyncWrite(relPath, bytes);
              }
              const at = typeof writeOpts.at === 'number' && writeOpts.at >= 0 ? Math.floor(writeOpts.at) : 0;
              const end = at + bytes.byteLength;
              if (end > buffer.byteLength) {
                const grown = new Uint8Array(end);
                grown.set(buffer, 0);
                buffer = grown;
              }
              buffer.set(bytes, at);
              return bytes.byteLength;
            },
            read(view, readOpts = {}) {
              ensureOpen();
              state.readCalls += 1;
              if (typeof opts.onSyncRead === 'function') {
                opts.onSyncRead(relPath);
              }
              const target =
                view instanceof Uint8Array
                  ? view
                  : ArrayBuffer.isView(view)
                    ? new Uint8Array(view.buffer, view.byteOffset, view.byteLength)
                    : new Uint8Array(view);
              const at = typeof readOpts.at === 'number' && readOpts.at >= 0 ? Math.floor(readOpts.at) : 0;
              if (at >= buffer.byteLength) {
                return 0;
              }
              const toCopy = Math.min(target.byteLength, buffer.byteLength - at);
              target.set(buffer.subarray(at, at + toCopy), 0);
              return toCopy;
            },
            flush() {
              ensureOpen();
              state.flushCalls += 1;
              entries.set(fileName, buffer.slice());
              totalWrites += 1;
              writeCounts.set(fileName, (writeCounts.get(fileName) || 0) + 1);
            },
            close() {
              if (closed) return;
              closed = true;
              state.activeOpenHandles.delete(id);
              if (state.openHandlesByPath.get(relPath) === id) {
                state.openHandlesByPath.delete(relPath);
              }
              state.totalClosedHandles += 1;
            },
          };
        };
      }

      return fileHandle;
    },
    async *entries() {
      for (const [entryName, value] of entries.entries()) {
        if (value && value.kind === 'directory') {
          yield [entryName, value];
        } else {
          yield [
            entryName,
            {
              kind: 'file',
              name: entryName,
              async getFile() {
                return {
                  size: value.byteLength,
                  lastModified: 1000,
                  async arrayBuffer() {
                    return value.slice().buffer;
                  },
                };
              },
            },
          ];
        }
      }
    },
    async removeEntry(entryName, removeOpts = {}) {
      if (!entries.has(entryName)) {
        const err = new Error(`NotFoundError: ${entryName}`);
        err.name = 'NotFoundError';
        throw err;
      }
      const relPath = relPathFor(entryName);
      if (state.enforceLocks && hasOpenHandleUnder(relPath)) {
        const err = new Error(`NoModificationAllowedError: cannot remove ${relPath} while SyncAccessHandle is open`);
        err.name = 'NoModificationAllowedError';
        throw err;
      }
      const target = entries.get(entryName);
      if (target && target.kind === 'directory' && target._entries.size > 0 && !removeOpts.recursive) {
        const err = new Error(`InvalidModificationError: ${entryName} is not empty`);
        err.name = 'InvalidModificationError';
        throw err;
      }
      entries.delete(entryName);
    },
  };

  return handle;
}

test('W313: SyncAccessHandle fast path provides read/write parity, overwrite truncation, warm reload, and sanitized telemetry', async () => {
  assert.equal(DEFAULT_MAX_OPEN_SYNC_HANDLES, 8);

  const cacheDir = createSyncMockOpfsDirHandle(DEFAULT_OPFS_CACHE_DIR);
  const opfsRoot = createMockOpfsDirHandle('', { [DEFAULT_OPFS_CACHE_DIR]: cacheDir });
  const syncState = cacheDir._syncState;

  const cache = new OpfsChunkCache({
    opfsRoot,
    indexFlushDelayMs: 0,
  });

  const payloadA = makePatternBytes(256, 11);
  const payloadB = makePatternBytes(128, 22);
  const payloadAShrunk = makePatternBytes(40, 33);
  const emptyPayload = new Uint8Array(0);

  await cache.put('chunks/a.bin', payloadA, { fileFingerprint: 'fp-a-1' });
  await cache.put('chunks/b.bin', payloadB, { fileFingerprint: 'fp-b-1' });
  await cache.put('chunks/empty.bin', emptyPayload);

  // Overwriting a larger chunk with a smaller payload must truncate cleanly via truncate(0).
  await cache.put('chunks/a.bin', payloadAShrunk, { fileFingerprint: 'fp-a-2' });

  assert.equal(syncState.createWritableCalls, 0, 'SyncAccessHandle fast path should bypass createWritable()');
  assert.equal(syncState.getFileCalls, 0, 'SyncAccessHandle fast path should bypass getFile()');
  assert.ok(syncState.truncateCalls >= 4);
  assert.ok(syncState.flushCalls >= 4);

  const gotA = await cache.get('chunks/a.bin', { fileFingerprint: 'fp-a-2' });
  const gotB = await cache.get('chunks/b.bin', { fileFingerprint: 'fp-b-1' });
  const gotEmpty = await cache.get('chunks/empty.bin');
  assert.deepEqual(gotA, payloadAShrunk);
  assert.deepEqual(gotB, payloadB);
  assert.deepEqual(gotEmpty, emptyPayload);
  assert.equal(syncState.getFileCalls, 0);

  const stats = cache.getStats();
  assert.ok(stats.syncHandleWrites >= 4);
  assert.equal(stats.syncHandleReads, 3);
  assert.equal(stats.syncHandleFallbacks, 0);
  assert.ok(stats.openSyncHandles > 0 && stats.openSyncHandles <= DEFAULT_MAX_OPEN_SYNC_HANDLES);
  assert.equal(stats.openSyncHandles, syncState.activeOpenHandles.size);

  // Verify every telemetry field remains a strictly sanitized primitive.
  for (const [k, v] of Object.entries(stats)) {
    assert.ok(
      typeof v === 'number' || typeof v === 'boolean' || v === null,
      `telemetry field ${k} must be primitive number/boolean/null, got ${typeof v}`
    );
  }

  // Closing the cache must flush and release every open SyncAccessHandle (zero leaks).
  await cache.close();
  assert.equal(cache.getStats().openSyncHandles, 0);
  assert.equal(syncState.activeOpenHandles.size, 0, 'Expected zero leaked SyncAccessHandles after close()');
  assert.equal(syncState.totalClosedHandles, syncState.totalOpenedHandles);

  // Warm reload in a brand-new OpfsChunkCache reads persisted index & chunks via SyncAccessHandle.
  const warmCache = new OpfsChunkCache({ opfsRoot, indexFlushDelayMs: 0 });
  const initRes = await warmCache.init();
  assert.equal(initRes.loadedFromIndex, true);
  assert.equal(initRes.entryCount, 3);
  const warmA = await warmCache.getEntry('chunks/a.bin', { fileFingerprint: 'fp-a-2' });
  assert.equal(warmA.hit, true);
  assert.deepEqual(warmA.data, payloadAShrunk);
  assert.ok(warmCache.getStats().syncHandleReads >= 2); // index read + chunk read

  await warmCache.destroy();
  assert.equal(warmCache.getStats().openSyncHandles, 0);
  assert.equal(syncState.activeOpenHandles.size, 0, 'Expected zero leaked SyncAccessHandles after destroy()');
  assert.equal(syncState.totalClosedHandles, syncState.totalOpenedHandles);
});

test('W313: bounded SyncAccessHandle LRU pool evicts cleanly, releases handles on delete/evict/clear, and supports non-pooled mode', async () => {
  const cacheDir = createSyncMockOpfsDirHandle(DEFAULT_OPFS_CACHE_DIR, null, { enforceLocks: true });
  const opfsRoot = createMockOpfsDirHandle('', { [DEFAULT_OPFS_CACHE_DIR]: cacheDir });
  const syncState = cacheDir._syncState;

  const cache = new OpfsChunkCache({
    opfsRoot,
    maxBytes: 256,
    maxOpenSyncHandles: 2,
    indexFlushDelayMs: 50,
  });

  // Writing 4 chunks with maxOpenSyncHandles=2 must strictly cap open handles at 2.
  for (let i = 0; i < 4; i += 1) {
    await cache.put(`k${i}.bin`, makePatternBytes(64, i), { fileFingerprint: `fp-${i}` });
    assert.ok(
      cache.getStats().openSyncHandles <= 2,
      `openSyncHandles exceeded cap: ${cache.getStats().openSyncHandles}`
    );
    assert.ok(
      syncState.activeOpenHandles.size <= 2,
      `active mock handles exceeded cap: ${syncState.activeOpenHandles.size}`
    );
  }

  // Deleting an entry whose handle is currently open in the pool must close it before removeEntry().
  assert.equal(await cache.delete('k3.bin'), true);
  assert.equal(cache.has('k3.bin'), false);

  // Triggering LRU budget eviction must close handles on victims before removeEntry().
  await cache.put('k4.bin', makePatternBytes(128, 4), { fileFingerprint: 'fp-4' });
  assert.ok(cache.getStats().evictions >= 1);

  // Explicit closeAllHandles() releases every pooled handle synchronously.
  const closedCount = cache.closeAllHandles();
  assert.ok(closedCount >= 1);
  assert.equal(cache.getStats().openSyncHandles, 0);
  assert.equal(syncState.activeOpenHandles.size, 0);

  // Clearing the cache with active handles closes all handles before removing files.
  await cache.get('k4.bin');
  assert.equal(cache.getStats().openSyncHandles, 1);
  await cache.clear();
  assert.equal(cache.getStats().openSyncHandles, 0);
  assert.equal(syncState.activeOpenHandles.size, 0);
  await cache.close();
  assert.equal(syncState.totalClosedHandles, syncState.totalOpenedHandles);

  // Non-pooled mode (maxOpenSyncHandles: 0) closes every handle immediately in finally.
  const nonPooled = new OpfsChunkCache({
    opfsRoot,
    maxOpenSyncHandles: 0,
    indexFlushDelayMs: 0,
  });
  await nonPooled.put('ephemeral.bin', makePatternBytes(48, 7));
  assert.equal(nonPooled.getStats().openSyncHandles, 0);
  assert.equal(syncState.activeOpenHandles.size, 0);
  const got = await nonPooled.get('ephemeral.bin');
  assert.deepEqual(got, makePatternBytes(48, 7));
  assert.equal(nonPooled.getStats().openSyncHandles, 0);
  assert.equal(syncState.activeOpenHandles.size, 0);
  assert.ok(nonPooled.getStats().syncHandleWrites >= 1);
  assert.equal(nonPooled.getStats().syncHandleReads, 1);
  assert.equal(syncState.totalClosedHandles, syncState.totalOpenedHandles);

  await nonPooled.close();
});

test('W313: automatic fallback to createWritable()/getFile() on NoModificationAllowedError, NotSupportedError, and useSyncAccessHandle=false', async () => {
  let injectErrorName = 'NoModificationAllowedError';
  const cacheDir = createSyncMockOpfsDirHandle(DEFAULT_OPFS_CACHE_DIR, null, {
    onCreateSyncAccessHandle() {
      if (injectErrorName) {
        const err = new Error(`${injectErrorName}: cannot acquire sync handle`);
        err.name = injectErrorName;
        throw err;
      }
    },
  });
  const opfsRoot = createMockOpfsDirHandle('', { [DEFAULT_OPFS_CACHE_DIR]: cacheDir });
  const syncState = cacheDir._syncState;

  const cache = new OpfsChunkCache({ opfsRoot, indexFlushDelayMs: 0 });
  const data1 = makePatternBytes(96, 5);
  const data2 = makePatternBytes(64, 9);

  // 1. NoModificationAllowedError falls back cleanly on both write and read.
  const putRes1 = await cache.put('fallback-1.bin', data1);
  assert.equal(putRes1.stored, true);
  const got1 = await cache.get('fallback-1.bin');
  assert.deepEqual(got1, data1);
  assert.ok(cache.getStats().syncHandleFallbacks >= 2);
  assert.equal(cache.getStats().syncHandleWrites, 0);
  assert.equal(cache.getStats().syncHandleReads, 0);
  assert.ok(syncState.createWritableCalls >= 1);
  assert.ok(syncState.getFileCalls >= 1);
  assert.equal(syncState.activeOpenHandles.size, 0);

  // 2. NotSupportedError also falls back cleanly without breaking callers.
  injectErrorName = 'NotSupportedError';
  const fallbacksBefore = cache.getStats().syncHandleFallbacks;
  await cache.put('fallback-2.bin', data2);
  const got2 = await cache.get('fallback-2.bin');
  assert.deepEqual(got2, data2);
  assert.ok(cache.getStats().syncHandleFallbacks > fallbacksBefore);
  assert.equal(syncState.activeOpenHandles.size, 0);

  // 3. Once sync access handles become available again, subsequent ops use fast path.
  injectErrorName = null;
  const got1Fast = await cache.get('fallback-1.bin');
  assert.deepEqual(got1Fast, data1);
  assert.equal(cache.getStats().syncHandleReads, 1);

  await cache.close();
  assert.equal(cache.getStats().openSyncHandles, 0);
  assert.equal(syncState.activeOpenHandles.size, 0);

  // 4. Opt-out via useSyncAccessHandle: false skips createSyncAccessHandle completely.
  const openedBeforeOptOut = syncState.totalOpenedHandles;
  const optOutCache = new OpfsChunkCache({
    opfsRoot,
    useSyncAccessHandle: false,
    indexFlushDelayMs: 0,
  });
  await optOutCache.put('async-only.bin', makePatternBytes(32, 3));
  const asyncGot = await optOutCache.get('async-only.bin');
  assert.deepEqual(asyncGot, makePatternBytes(32, 3));
  assert.equal(syncState.totalOpenedHandles, openedBeforeOptOut);
  assert.equal(optOutCache.getStats().syncHandleWrites, 0);
  assert.equal(optOutCache.getStats().syncHandleReads, 0);
  assert.equal(optOutCache.getStats().syncHandleFallbacks, 0);
  assert.equal(optOutCache.getStats().openSyncHandles, 0);
  await optOutCache.close();
});

test('W313: QuotaExceededError recovery on SyncAccessHandle write and concurrent closeAllHandles() race leave zero leaked handles', async () => {
  let failOnceOnBig = true;
  let delayOpenResolve = null;
  let gateTarget = null;

  const cacheDir = createSyncMockOpfsDirHandle(DEFAULT_OPFS_CACHE_DIR, null, {
    enforceLocks: true,
    async onCreateSyncAccessHandle(relPath) {
      if (gateTarget && relPath === gateTarget) {
        await new Promise((resolve) => {
          delayOpenResolve = resolve;
        });
      }
    },
    onSyncWrite(relPath) {
      if (relPath === 'big-sync.bin' && failOnceOnBig) {
        failOnceOnBig = false;
        const err = new Error('QuotaExceededError: OPFS disk full');
        err.name = 'QuotaExceededError';
        throw err;
      }
    },
  });
  const opfsRoot = createMockOpfsDirHandle('', { [DEFAULT_OPFS_CACHE_DIR]: cacheDir });
  const syncState = cacheDir._syncState;

  const cache = new OpfsChunkCache({ opfsRoot, maxBytes: null, indexFlushDelayMs: 0 });
  await cache.put('victim-1.bin', makePatternBytes(100, 1));
  await cache.put('victim-2.bin', makePatternBytes(100, 2));

  // SyncAccessHandle write throws QuotaExceededError -> evicts old entries (closing their handles) and retries cleanly.
  const bigRes = await cache.put('big-sync.bin', makePatternBytes(150, 3));
  assert.equal(bigRes.stored, true);
  assert.equal(bigRes.quotaRecovered, true);
  assert.ok(bigRes.evictedEntries >= 1);
  assert.deepEqual(await cache.get('big-sync.bin'), makePatternBytes(150, 3));

  // Concurrent reads of the same unpooled key share one in-flight createSyncAccessHandle() without lock contention.
  cache.closeAllHandles();
  gateTarget = 'big-sync.bin';
  const readP1 = cache.get('big-sync.bin', { touch: false });
  const readP2 = cache.get('big-sync.bin', { touch: false });
  await new Promise((resolve) => setTimeout(resolve, 5));
  assert.ok(delayOpenResolve, 'expected gated createSyncAccessHandle to be in flight');
  const releaseOpen = delayOpenResolve;
  delayOpenResolve = null;
  gateTarget = null;
  releaseOpen();
  const [b1, b2] = await Promise.all([readP1, readP2]);
  assert.deepEqual(b1, makePatternBytes(150, 3));
  assert.deepEqual(b2, makePatternBytes(150, 3));
  assert.equal(cache.getStats().syncHandleFallbacks, 0);

  // Race: closeAllHandles() called while createSyncAccessHandle() is pending must close the resolved handle immediately.
  cache.closeAllHandles();
  gateTarget = 'big-sync.bin';
  const racingRead = cache.get('big-sync.bin', { touch: false });
  await new Promise((resolve) => setTimeout(resolve, 5));
  assert.ok(delayOpenResolve);
  cache.closeAllHandles();
  const releaseRace = delayOpenResolve;
  delayOpenResolve = null;
  gateTarget = null;
  releaseRace();
  const racedBytes = await racingRead;
  assert.deepEqual(racedBytes, makePatternBytes(150, 3));
  assert.equal(cache.getStats().openSyncHandles, 0);
  assert.equal(syncState.activeOpenHandles.size, 0, 'Pending SyncAccessHandle resolved after closeAllHandles() must not leak');

  // resetStats() clears cumulative sync counters while keeping live openSyncHandles accurate.
  await cache.get('big-sync.bin', { touch: false });
  assert.equal(cache.getStats().openSyncHandles, 1);
  const reset = cache.resetStats();
  assert.equal(reset.syncHandleReads, 0);
  assert.equal(reset.syncHandleWrites, 0);
  assert.equal(reset.syncHandleFallbacks, 0);
  assert.equal(reset.openSyncHandles, 1);

  await cache.close();
  assert.equal(cache.getStats().openSyncHandles, 0);
  assert.equal(syncState.activeOpenHandles.size, 0);
  assert.equal(syncState.totalClosedHandles, syncState.totalOpenedHandles);
});
