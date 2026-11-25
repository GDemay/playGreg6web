import test from 'node:test';
import assert from 'node:assert/strict';

import {
  CHUNK_KINDS,
  DEFAULT_INDEX_FILENAME,
  DEFAULT_INDEX_FLUSH_DELAY_MS,
  DEFAULT_MAX_BYTES,
  DEFAULT_OPFS_CACHE_DIR,
  DEFAULT_PREFETCH_CONCURRENCY,
  OPFS_CACHE_INDEX_VERSION,
  OpfsCache,
  OpfsChunkCache,
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
