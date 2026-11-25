import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { OpfsChunkCache } from '../web/storage/opfs_cache.js';

const require = createRequire(import.meta.url);
const {
  RangeCache,
  DEFAULT_COALESCE_MAX_GAP_BYTES,
  DEFAULT_COALESCE_MAX_MERGED_BYTES,
  DEFAULT_MAX_READ_AHEAD_BYTES,
  DEFAULT_SEQUENTIAL_STREAK_THRESHOLD,
  DEFAULT_MAX_PENDING_OPFS_WRITES,
  DEFAULT_MAX_OPFS_WRITE_CONCURRENCY,
  buildFileFingerprint,
  opfsChunkFileName,
  isQuotaExceededError,
  coalesceAdjacentRanges,
  detectSequentialRead,
} = require('../web/workers/range_cache.js');

function makePatternBytes(length, seed = 17) {
  const out = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = (i * 31 + seed) & 0xff;
  }
  return out;
}

function createMockRangeFetch(filesMap) {
  const calls = [];

  async function mockFetch(url, init = {}) {
    const rangeHeader = (init.headers && (init.headers.Range || init.headers.range)) || null;
    calls.push({ url, rangeHeader });

    const relPath = url.replace(/^.*\/data\//, '');
    const data = filesMap.get(relPath);
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

    if (!rangeHeader) {
      return {
        ok: true,
        status: 200,
        headers: new Map([['Content-Length', String(data.byteLength)]]),
        async arrayBuffer() {
          return data.slice().buffer;
        },
      };
    }

    const match = /^bytes=(\d+)-(\d+)$/.exec(rangeHeader);
    assert.ok(match, `Unexpected Range header: ${rangeHeader}`);
    const reqStart = Number(match[1]);
    const reqEnd = Number(match[2]);

    if (reqStart >= data.byteLength) {
      return {
        ok: false,
        status: 416,
        headers: new Map([['Content-Range', `bytes */${data.byteLength}`]]),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }

    const actualEnd = Math.min(reqEnd, data.byteLength - 1);
    const slice = data.slice(reqStart, actualEnd + 1);
    return {
      ok: true,
      status: 206,
      headers: new Map([
        ['Content-Range', `bytes ${reqStart}-${actualEnd}/${data.byteLength}`],
        ['Content-Length', String(slice.byteLength)],
      ]),
      async arrayBuffer() {
        return slice.buffer;
      },
    };
  }

  return { fetchImpl: mockFetch, calls };
}

function makeQuotaError(variant = 'dom') {
  if (variant === 'code22') {
    const err = new Error('Storage limit reached');
    err.code = 22;
    return err;
  }
  if (variant === 'code1014') {
    const err = new Error('NS_ERROR_DOM_QUOTA_REACHED');
    err.code = 1014;
    return err;
  }
  if (variant === 'message') {
    return new Error('Disk quota exceeded while persisting chunk');
  }
  const err = new Error('The quota has been exceeded.');
  err.name = 'QuotaExceededError';
  return err;
}

function createQuotaMockOpfsDir(options = {}) {
  const files = new Map();
  const maxQuotaBytes = Number.isFinite(options.maxQuotaBytes) ? options.maxQuotaBytes : Infinity;
  const alwaysThrowQuota = Boolean(options.alwaysThrowQuota);
  const useSyncAccessHandle = Boolean(options.useSyncAccessHandle);
  const quotaErrorVariant = options.quotaErrorVariant || 'dom';
  const failReadsOnce = new Set(options.failReadsOnce || []);
  const writeAttempts = [];
  const removedFiles = [];

  function usedBytes(excludeName = null) {
    let total = 0;
    for (const [name, buf] of files.entries()) {
      if (name === excludeName) continue;
      total += buf.byteLength;
    }
    return total;
  }

  function checkQuota(name, incomingLength) {
    if (alwaysThrowQuota || usedBytes(name) + incomingLength > maxQuotaBytes) {
      throw makeQuotaError(quotaErrorVariant);
    }
  }

  function makeFileHandle(name) {
    const handle = {
      async getFile() {
        if (failReadsOnce.has(name)) {
          failReadsOnce.delete(name);
          const err = new Error(`Corrupt backing entry: ${name}`);
          err.name = 'DataError';
          throw err;
        }
        const current = files.get(name);
        return {
          size: current ? current.byteLength : 0,
          async arrayBuffer() {
            return current.slice().buffer;
          },
        };
      },
    };

    if (useSyncAccessHandle) {
      handle.createSyncAccessHandle = async () => {
        let pending = files.has(name) ? files.get(name).slice() : new Uint8Array(0);
        return {
          truncate(len) {
            pending = pending.slice(0, len);
          },
          write(chunk) {
            writeAttempts.push({ name, byteLength: chunk.byteLength });
            checkQuota(name, chunk.byteLength);
            pending = chunk instanceof Uint8Array ? chunk.slice() : new Uint8Array(chunk);
            return pending.byteLength;
          },
          flush() {
            files.set(name, pending);
          },
          close() {},
        };
      };
    } else {
      handle.createWritable = async () => {
        let pending = new Uint8Array(0);
        let aborted = false;
        return {
          async write(chunk) {
            writeAttempts.push({ name, byteLength: chunk.byteLength });
            checkQuota(name, chunk.byteLength);
            pending = chunk instanceof Uint8Array ? chunk.slice() : new Uint8Array(chunk);
          },
          async close() {
            if (!aborted) {
              files.set(name, pending);
            }
          },
          async abort() {
            aborted = true;
          },
        };
      };
    }

    return handle;
  }

  return {
    files,
    writeAttempts,
    removedFiles,
    async getFileHandle(name, opts = {}) {
      if (!files.has(name)) {
        if (!opts.create) {
          const err = new Error(`NotFoundError: ${name}`);
          err.name = 'NotFoundError';
          throw err;
        }
        files.set(name, new Uint8Array(0));
      }
      return makeFileHandle(name);
    },
    async removeEntry(name) {
      removedFiles.push(name);
      if (!files.delete(name)) {
        const err = new Error(`NotFoundError: ${name}`);
        err.name = 'NotFoundError';
        throw err;
      }
    },
    async *entries() {
      for (const name of files.keys()) {
        yield [name, makeFileHandle(name)];
      }
    },
  };
}

test('RangeCache getStats exposes additive resilience counters and isQuotaExceededError classifies errors', () => {
  const cache = new RangeCache({ chunkSize: 16, maxBytes: 128 });
  const stats = cache.getStats();

  assert.equal(stats.hits, 0);
  assert.equal(stats.misses, 0);
  assert.equal(stats.opfsHits, 0);
  assert.equal(stats.opfsMisses, 0);
  assert.equal(stats.opfsWrites, 0);
  assert.equal(stats.opfsWriteErrors, 0);
  assert.equal(stats.opfsStaleInvalidations, 0);
  assert.equal(stats.opfsHitRate, 0);
  assert.equal(stats.evictions, 0);
  assert.equal(stats.quotaEvictions, 0);
  assert.equal(stats.quotaRetrySuccesses, 0);
  assert.equal(stats.corruptEvictions, 0);
  assert.equal(stats.coalescedRangeCount, 0);
  assert.equal(stats.sequentialHits, 0);
  assert.equal(stats.prefetchHits, 0);
  assert.equal(stats.prefetchBytes, 0);
  assert.equal(stats.bytesSavedByCoalescing, 0);
  assert.equal(stats.bytesCached, 0);
  assert.equal(stats.entries, 0);

  assert.equal(DEFAULT_COALESCE_MAX_GAP_BYTES, 64 * 1024);
  assert.equal(DEFAULT_COALESCE_MAX_MERGED_BYTES, 2 * 1024 * 1024);
  assert.equal(DEFAULT_MAX_READ_AHEAD_BYTES, 2 * 1024 * 1024);
  assert.equal(DEFAULT_SEQUENTIAL_STREAK_THRESHOLD, 2);

  assert.equal(isQuotaExceededError(makeQuotaError('dom')), true);
  assert.equal(isQuotaExceededError(makeQuotaError('code22')), true);
  assert.equal(isQuotaExceededError(makeQuotaError('code1014')), true);
  assert.equal(isQuotaExceededError(makeQuotaError('message')), true);
  assert.equal(isQuotaExceededError(null), false);
  assert.equal(isQuotaExceededError(undefined), false);
  assert.equal(isQuotaExceededError(Object.assign(new Error('missing'), { name: 'NotFoundError' })), false);
  assert.equal(isQuotaExceededError(new Error('network reset')), false);
});

test('RangeCache QuotaExceededError triggers LRU eviction and succeeds on single retry', async () => {
  const fileBytes = makePatternBytes(96, 13); // 6 chunks of 16 bytes
  const mockFetch = createMockRangeFetch(new Map([['world.rpf', fileBytes]]));
  // OPFS quota holds exactly 4 chunks (64 bytes).
  const opfsDir = createQuotaMockOpfsDir({ maxQuotaBytes: 64 });

  const cache = new RangeCache({
    chunkSize: 16,
    maxBytes: 64,
    fetchImpl: mockFetch.fetchImpl,
    opfsDir,
  });

  // Populate chunks 0, 1, 2, 3 -> OPFS is at 64/64 bytes.
  await cache.read('world.rpf', 0, 16);
  await cache.flushOpfsWrites();
  await cache.read('world.rpf', 16, 16);
  await cache.flushOpfsWrites();
  await cache.read('world.rpf', 32, 16);
  await cache.flushOpfsWrites();
  await cache.read('world.rpf', 48, 16);
  await cache.flushOpfsWrites();
  assert.equal(opfsDir.files.size, 4);

  // Touch chunk 0 so LRU order becomes [1, 2, 3, 0].
  await cache.read('world.rpf', 0, 8);

  // Reading chunk 4 triggers QuotaExceededError on OPFS write, evicts oldest entry (chunk 1),
  // and retries once cleanly.
  const chunk4 = await cache.read('world.rpf', 64, 16);
  await cache.flushOpfsWrites();
  assert.deepEqual(chunk4, fileBytes.subarray(64, 80));

  const stats = cache.getStats();
  assert.ok(stats.quotaEvictions > 0, `Expected quotaEvictions > 0, got ${stats.quotaEvictions}`);
  assert.equal(stats.quotaRetrySuccesses, 1);

  const chunk0File = opfsChunkFileName('world.rpf', 0, 16);
  const chunk1File = opfsChunkFileName('world.rpf', 1, 16);
  const chunk4File = opfsChunkFileName('world.rpf', 4, 16);
  assert.equal(opfsDir.files.has(chunk1File), false, 'Oldest LRU chunk 1 must be evicted from OPFS');
  assert.equal(opfsDir.files.has(chunk0File), true, 'Recently touched chunk 0 must remain in OPFS');
  assert.equal(opfsDir.files.has(chunk4File), true, 'Retried chunk 4 must be persisted in OPFS');

  // Verify persisted chunk 4 serves a warm OPFS hit with zero network fetches.
  const warmFetch = createMockRangeFetch(new Map([['world.rpf', fileBytes]]));
  const warmCache = new RangeCache({
    chunkSize: 16,
    maxBytes: 64,
    fetchImpl: warmFetch.fetchImpl,
    opfsDir,
  });
  const warmChunk4 = await warmCache.read('world.rpf', 64, 16);
  assert.deepEqual(warmChunk4, fileBytes.subarray(64, 80));
  assert.equal(warmFetch.calls.length, 0);
  assert.equal(warmCache.getStats().opfsHits, 1);
});

test('RangeCache QuotaExceededError frees at least 25% of maxBytes on KV backing store and SyncAccessHandle', async () => {
  const fileBytes = makePatternBytes(128, 29); // 8 chunks of 16 bytes
  const mockFetch = createMockRangeFetch(new Map([['big.rpf', fileBytes]]));

  // Map-like KV backing store capped at 96 bytes (6 chunks) using Firefox code 1014 quota error.
  const kvFiles = new Map();
  let kvBytes = 0;
  const kvStore = {
    async get(key) {
      return kvFiles.get(key);
    },
    async set(key, val) {
      const prev = kvFiles.get(key);
      const prevLen = prev ? prev.byteLength : 0;
      if (kvBytes - prevLen + val.byteLength > 96) {
        throw makeQuotaError('code1014');
      }
      kvFiles.set(key, val);
      kvBytes = kvBytes - prevLen + val.byteLength;
    },
    async delete(key) {
      const prev = kvFiles.get(key);
      if (!prev) return false;
      kvFiles.delete(key);
      kvBytes -= prev.byteLength;
      return true;
    },
  };

  // maxBytes = 256 -> 25% target = 64 bytes (4 chunks of 16 bytes).
  const cache = new RangeCache({
    chunkSize: 16,
    maxBytes: 256,
    fetchImpl: mockFetch.fetchImpl,
    opfsDir: kvStore,
  });

  for (let i = 0; i < 6; i += 1) {
    await cache.read('big.rpf', i * 16, 16);
    await cache.flushOpfsWrites();
  }
  assert.equal(kvFiles.size, 6);

  // Chunk 6 overflows the 96-byte KV quota -> evicts >= 64 bytes (4 oldest chunks), retries once.
  const chunk6 = await cache.read('big.rpf', 6 * 16, 16);
  await cache.flushOpfsWrites();
  assert.deepEqual(chunk6, fileBytes.subarray(96, 112));
  assert.equal(cache.getStats().quotaEvictions, 4);
  assert.equal(cache.getStats().quotaRetrySuccesses, 1);
  assert.equal(kvFiles.size, 3); // chunks 4, 5, 6 remain

  // Also test createSyncAccessHandle() with code 22 quota error.
  const syncOpfs = createQuotaMockOpfsDir({
    maxQuotaBytes: 32,
    useSyncAccessHandle: true,
    quotaErrorVariant: 'code22',
  });
  const syncCache = new RangeCache({
    chunkSize: 16,
    maxBytes: 64,
    fetchImpl: mockFetch.fetchImpl,
    opfsDir: syncOpfs,
  });
  await syncCache.read('big.rpf', 0, 16);
  await syncCache.flushOpfsWrites();
  await syncCache.read('big.rpf', 16, 16);
  await syncCache.flushOpfsWrites();
  const chunk2 = await syncCache.read('big.rpf', 32, 16);
  await syncCache.flushOpfsWrites();
  assert.deepEqual(chunk2, fileBytes.subarray(32, 48));
  assert.equal(syncCache.getStats().quotaEvictions, 1);
  assert.equal(syncCache.getStats().quotaRetrySuccesses, 1);
});

test('RangeCache evicts untracked prior-session OPFS files on QuotaExceededError', async () => {
  const fileBytes = makePatternBytes(64, 41);
  const mockFetch = createMockRangeFetch(new Map([['session.rpf', fileBytes]]));
  const opfsDir = createQuotaMockOpfsDir({ maxQuotaBytes: 32 });

  // Pre-seed 2 chunks from a previous browser session (unknown to the new RangeCache instance).
  opfsDir.files.set(opfsChunkFileName('old.rpf', 0, 16), makePatternBytes(16, 1));
  opfsDir.files.set(opfsChunkFileName('old.rpf', 1, 16), makePatternBytes(16, 2));

  const cache = new RangeCache({
    chunkSize: 16,
    maxBytes: 64,
    fetchImpl: mockFetch.fetchImpl,
    opfsDir,
  });

  const out = await cache.read('session.rpf', 0, 16);
  await cache.flushOpfsWrites();
  assert.deepEqual(out, fileBytes.subarray(0, 16));
  assert.equal(cache.getStats().quotaEvictions, 1);
  assert.equal(cache.getStats().quotaRetrySuccesses, 1);
  assert.equal(opfsDir.files.has(opfsChunkFileName('session.rpf', 0, 16)), true);
});

test('RangeCache persistent QuotaExceededError degrades gracefully without throwing and returns fetched bytes', async () => {
  const fileBytes = makePatternBytes(64, 73);
  const mockFetch = createMockRangeFetch(new Map([['persistent.rpf', fileBytes]]));
  const opfsDir = createQuotaMockOpfsDir({ alwaysThrowQuota: true });

  const cache = new RangeCache({
    chunkSize: 16,
    maxBytes: 64,
    fetchImpl: mockFetch.fetchImpl,
    opfsDir,
  });

  // Must not throw and must return exact fetched bytes despite persistent OPFS quota failure.
  const slice = await cache.read('persistent.rpf', 4, 24);
  await cache.flushOpfsWrites();
  assert.deepEqual(slice, fileBytes.subarray(4, 28));

  const stats = cache.getStats();
  assert.equal(stats.quotaRetrySuccesses, 0);
  assert.equal(stats.opfsWriteErrors, 2);
  assert.equal(stats.opfsWrites, 0);
  assert.equal(stats.misses, 2);

  // Each chunk write should be attempted at most twice (initial + 1 retry), leaving no partial files.
  assert.equal(opfsDir.writeAttempts.length, 4);
  assert.equal(opfsDir.files.size, 0, 'Failed OPFS writes must not leave 0-byte partial files behind');

  // Subsequent read still serves cleanly from in-memory cache with zero network fetches.
  const callsBefore = mockFetch.calls.length;
  const reread = await cache.read('persistent.rpf', 4, 24);
  assert.deepEqual(reread, fileBytes.subarray(4, 28));
  assert.equal(mockFetch.calls.length, callsBefore);
});

test('RangeCache self-heals truncated OPFS slice on warm cache and increments corruptEvictions', async () => {
  const fileBytes = makePatternBytes(48, 91);
  const mockFetch = createMockRangeFetch(new Map([['corrupt.rpf', fileBytes]]));
  const opfsDir = createQuotaMockOpfsDir();

  // Seed chunk 0 as truncated (5 bytes instead of 16) and chunk 1 as valid (16 bytes).
  const chunk0Name = opfsChunkFileName('corrupt.rpf', 0, 16);
  const chunk1Name = opfsChunkFileName('corrupt.rpf', 1, 16);
  opfsDir.files.set(chunk0Name, fileBytes.slice(0, 5));
  opfsDir.files.set(chunk1Name, fileBytes.slice(16, 32));

  const cache = new RangeCache({
    chunkSize: 16,
    maxBytes: 128,
    fetchImpl: mockFetch.fetchImpl,
    opfsDir,
  });

  // Reading across chunks 0 and 1 must evict truncated chunk 0, re-fetch it cleanly, and use chunk 1 from OPFS.
  const out = await cache.read('corrupt.rpf', 0, 32);
  await cache.flushOpfsWrites();
  assert.deepEqual(out, fileBytes.subarray(0, 32));
  assert.equal(mockFetch.calls.length, 1, 'Only the truncated chunk 0 should be re-fetched');
  assert.equal(mockFetch.calls[0].rangeHeader, 'bytes=0-15');

  const stats = cache.getStats();
  assert.equal(stats.corruptEvictions, 1);
  assert.equal(stats.opfsHits, 1);
  assert.equal(stats.misses, 1);

  // Verify chunk 0 was healed in OPFS so a new warm cache reads both chunks with 0 fetches.
  const verifyFetch = createMockRangeFetch(new Map([['corrupt.rpf', fileBytes]]));
  const healedCache = new RangeCache({
    chunkSize: 16,
    maxBytes: 128,
    fetchImpl: verifyFetch.fetchImpl,
    opfsDir,
  });
  const healed = await healedCache.read('corrupt.rpf', 0, 32);
  assert.deepEqual(healed, fileBytes.subarray(0, 32));
  assert.equal(verifyFetch.calls.length, 0);
  assert.equal(healedCache.getStats().corruptEvictions, 0);
  assert.equal(healedCache.getStats().opfsHits, 2);
});

test('RangeCache self-heals 0-byte, oversized, known-length-mismatched, and unreadable OPFS entries', async () => {
  const fileBytes = makePatternBytes(64, 53);
  const mockFetch = createMockRangeFetch(new Map([['heal.rpf', fileBytes]]));
  const chunk2Name = opfsChunkFileName('heal.rpf', 2, 16);
  const opfsDir = createQuotaMockOpfsDir({ failReadsOnce: [chunk2Name] });

  // Chunk 0: 0-byte empty file; Chunk 1: oversized (24 bytes > chunkSize 16); Chunk 2: throws DataError on read.
  opfsDir.files.set(opfsChunkFileName('heal.rpf', 0, 16), new Uint8Array(0));
  opfsDir.files.set(opfsChunkFileName('heal.rpf', 1, 16), makePatternBytes(24, 3));
  opfsDir.files.set(chunk2Name, fileBytes.slice(32, 48));

  const cache = new RangeCache({
    chunkSize: 16,
    maxBytes: 128,
    fetchImpl: mockFetch.fetchImpl,
    opfsDir,
  });

  const out = await cache.read('heal.rpf', 0, 48);
  assert.deepEqual(out, fileBytes.subarray(0, 48));
  assert.equal(cache.getStats().corruptEvictions, 3);
  assert.equal(mockFetch.calls.length, 3);

  // Now simulate external OPFS truncation of chunk 0 to 10 bytes after memory cache clear,
  // and read only the first 4 bytes: known expected length (16) must still catch the corruption.
  cache.clear();
  opfsDir.files.set(opfsChunkFileName('heal.rpf', 0, 16), fileBytes.slice(0, 10));
  const sub = await cache.read('heal.rpf', 0, 4);
  assert.deepEqual(sub, fileBytes.subarray(0, 4));
  assert.equal(cache.getStats().corruptEvictions, 4);
  assert.equal(mockFetch.calls.length, 4);

  // Normal NotFoundError miss on chunk 3 must NOT increment corruptEvictions.
  const c3 = await cache.read('heal.rpf', 48, 16);
  assert.deepEqual(c3, fileBytes.subarray(48, 64));
  assert.equal(cache.getStats().corruptEvictions, 4);
});

test('RangeCache self-heals corrupt in-memory entry and preserves legitimate EOF tail chunks', async () => {
  const fileBytes = makePatternBytes(50, 101); // 3 full 16B chunks + 2B tail chunk at index 3
  const mockFetch = createMockRangeFetch(new Map([['tail.rpf', fileBytes]]));
  const opfsDir = createQuotaMockOpfsDir();

  const cache = new RangeCache({
    chunkSize: 16,
    maxBytes: 128,
    fetchImpl: mockFetch.fetchImpl,
    opfsDir,
  });

  // Read across tail chunk past EOF -> 10 bytes, zero corruptEvictions.
  const tail1 = await cache.read('tail.rpf', 40, 30);
  await cache.flushOpfsWrites();
  assert.deepEqual(tail1, fileBytes.subarray(40, 50));
  assert.equal(cache.getStats().corruptEvictions, 0);

  // Repeat read hits memory cache without re-fetch or false corrupt eviction.
  const callsAfterFirst = mockFetch.calls.length;
  const tail2 = await cache.read('tail.rpf', 40, 30);
  assert.deepEqual(tail2, fileBytes.subarray(40, 50));
  assert.equal(mockFetch.calls.length, callsAfterFirst);
  assert.equal(cache.getStats().corruptEvictions, 0);

  // Brand-new warm cache reading exact tail (2 bytes) gets warm OPFS hit with 0 fetches and 0 corruptEvictions.
  const warmFetch = createMockRangeFetch(new Map([['tail.rpf', fileBytes]]));
  const warmCache = new RangeCache({
    chunkSize: 16,
    maxBytes: 128,
    fetchImpl: warmFetch.fetchImpl,
    opfsDir,
  });
  const warmTailExact = await warmCache.read('tail.rpf', 48, 2);
  assert.deepEqual(warmTailExact, fileBytes.subarray(48, 50));
  assert.equal(warmFetch.calls.length, 0);
  assert.equal(warmCache.getStats().corruptEvictions, 0);

  // Brand-new warm cache reading past EOF verifies legitimate 2-byte tail without counting it corrupt.
  const warmCache2 = new RangeCache({
    chunkSize: 16,
    maxBytes: 128,
    fetchImpl: warmFetch.fetchImpl,
    opfsDir,
  });
  const warmTailPastEof = await warmCache2.read('tail.rpf', 40, 30);
  assert.deepEqual(warmTailPastEof, fileBytes.subarray(40, 50));
  assert.equal(warmCache2.getStats().corruptEvictions, 0);

  // Corrupt an in-memory chunk entry directly -> must detect, evict, re-fetch, and increment corruptEvictions.
  cache._memCache.set('tail.rpf::2', new Uint8Array([0xff, 0xff, 0xff]));
  const healedChunk2 = await cache.read('tail.rpf', 32, 16);
  assert.deepEqual(healedChunk2, fileBytes.subarray(32, 48));
  assert.equal(cache.getStats().corruptEvictions, 1);
});

// --- W181: Coalesced adjacent range merging & adaptive sequential read-ahead tests ---

test('coalesceAdjacentRanges merges unsorted adjacent, overlapping, and small-gap ranges while respecting caps and paths', () => {
  assert.deepEqual(coalesceAdjacentRanges(null), []);
  assert.deepEqual(coalesceAdjacentRanges([]), []);

  // Unsorted 64 KiB DirectStorage reads + small 16 KiB gap + separate file + huge gap split.
  const KiB = 1024;
  const input = [
    { path: 'x64/stream.rpf', offset: 128 * KiB, length: 64 * KiB },
    { path: 'movies/intro.bk2', offset: 0, length: 64 * KiB },
    { path: 'x64/stream.rpf', offset: 0, length: 64 * KiB },
    { path: 'x64/stream.rpf', offset: 64 * KiB, length: 64 * KiB },
    // 16 KiB gap after 192 KiB -> bridges when maxGapBytes = 32 KiB
    { path: 'x64/stream.rpf', offset: 208 * KiB, length: 64 * KiB },
    // 256 KiB gap after 272 KiB -> exceeds maxGapBytes, starts new span
    { path: 'x64/stream.rpf', offset: 528 * KiB, length: 64 * KiB },
    // Second adjacent read on separate file
    { path: 'movies/intro.bk2', offset: 64 * KiB, length: 64 * KiB },
    // Invalid / zero-length entries should be skipped safely
    { path: 'x64/stream.rpf', offset: -10, length: 64 },
    { path: 'x64/stream.rpf', offset: 0, length: 0 },
    null,
  ];

  const merged = coalesceAdjacentRanges(input, 32 * KiB, 512 * KiB);
  assert.equal(merged.length, 3);

  assert.equal(merged[0].path, 'x64/stream.rpf');
  assert.equal(merged[0].offset, 0);
  assert.equal(merged[0].length, 272 * KiB);
  assert.equal(merged[0].end, 272 * KiB);
  assert.equal(merged[0].mergedCount, 4);
  assert.equal(merged[0].gapBytes, 16 * KiB);
  assert.equal(merged[0].originalBytes, 256 * KiB);
  assert.equal(merged[0].overlapBytes, 0);

  assert.equal(merged[1].path, 'x64/stream.rpf');
  assert.equal(merged[1].offset, 528 * KiB);
  assert.equal(merged[1].length, 64 * KiB);
  assert.equal(merged[1].mergedCount, 1);

  assert.equal(merged[2].path, 'movies/intro.bk2');
  assert.equal(merged[2].offset, 0);
  assert.equal(merged[2].length, 128 * KiB);
  assert.equal(merged[2].mergedCount, 2);

  // Overlapping + contained ranges + maxMergedBytes split + pathless { start, end } support.
  const overlapMerged = coalesceAdjacentRanges(
    [
      { start: 0, end: 100 },
      { offset: 40, length: 80 }, // overlaps 60 bytes -> [0..120)
      { offset: 10, length: 20 }, // fully contained -> overlaps 20 bytes
      { start: 120, length: 80 }, // would make 200 > maxMerged 150 -> splits
      { start: 200, length: 50 }, // merges with [120..200) -> [120..250) (130 <= 150)
    ],
    0,
    150,
  );
  assert.equal(overlapMerged.length, 2);
  assert.equal('path' in overlapMerged[0], false);
  assert.equal(overlapMerged[0].offset, 0);
  assert.equal(overlapMerged[0].length, 120);
  assert.equal(overlapMerged[0].mergedCount, 3);
  assert.equal(overlapMerged[0].overlapBytes, 80);

  assert.equal(overlapMerged[1].offset, 120);
  assert.equal(overlapMerged[1].length, 130);
  assert.equal(overlapMerged[1].mergedCount, 2);
  assert.equal(overlapMerged[1].overlapBytes, 0);
});

test('detectSequentialRead ramps read-ahead up to maxReadAheadBytes on consecutive 64 KiB reads and resets on random seek', () => {
  const KiB = 1024;
  const MiB = 1024 * 1024;
  const opts = {
    chunkSize: 256 * KiB,
    maxGapBytes: 64 * KiB,
    maxReadAheadBytes: 2 * MiB,
    streakThreshold: 2,
  };

  let state = null;
  const windows = [];

  // 7 consecutive 64 KiB DirectStorage reads: [0..64K), [64K..128K), ...
  for (let i = 0; i < 7; i += 1) {
    const det = detectSequentialRead(state, i * 64 * KiB, 64 * KiB, opts);
    state = det.nextState;
    windows.push({
      isSequential: det.isSequential,
      streak: det.streak,
      readAheadBytes: det.readAheadBytes,
    });
  }

  assert.deepEqual(windows, [
    { isSequential: false, streak: 0, readAheadBytes: 0 },
    { isSequential: true, streak: 1, readAheadBytes: 0 },
    { isSequential: true, streak: 2, readAheadBytes: 256 * KiB },
    { isSequential: true, streak: 3, readAheadBytes: 512 * KiB },
    { isSequential: true, streak: 4, readAheadBytes: MiB },
    { isSequential: true, streak: 5, readAheadBytes: 2 * MiB },
    { isSequential: true, streak: 6, readAheadBytes: 2 * MiB },
  ]);

  // Small forward gap (32 KiB <= 64 KiB) keeps sequential streak active.
  const withSmallGap = detectSequentialRead(state, 7 * 64 * KiB + 32 * KiB, 64 * KiB, opts);
  assert.equal(withSmallGap.isSequential, true);
  assert.equal(withSmallGap.gapBytes, 32 * KiB);
  assert.equal(withSmallGap.streak, 7);
  assert.equal(withSmallGap.readAheadBytes, 2 * MiB);
  state = withSmallGap.nextState;

  // Large forward random seek resets immediately.
  const forwardJump = detectSequentialRead(state, 20 * MiB, 64 * KiB, opts);
  assert.equal(forwardJump.isSequential, false);
  assert.equal(forwardJump.isRandomSeek, true);
  assert.equal(forwardJump.streak, 0);
  assert.equal(forwardJump.readAheadBytes, 0);
  state = forwardJump.nextState;

  // Backward seek also resets cleanly.
  const seq1 = detectSequentialRead(state, 20 * MiB + 64 * KiB, 64 * KiB, opts);
  assert.equal(seq1.isSequential, true);
  assert.equal(seq1.streak, 1);
  const backward = detectSequentialRead(seq1.nextState, 0, 64 * KiB, opts);
  assert.equal(backward.isSequential, false);
  assert.equal(backward.isRandomSeek, true);
  assert.equal(backward.streak, 0);
  assert.equal(backward.readAheadBytes, 0);

  // maxReadAheadBytes: 0 disables read-ahead while still detecting sequential hits.
  const disabled0 = detectSequentialRead(null, 0, 64 * KiB, { ...opts, maxReadAheadBytes: 0 });
  const disabled1 = detectSequentialRead(disabled0.nextState, 64 * KiB, 64 * KiB, {
    ...opts,
    maxReadAheadBytes: 0,
  });
  const disabled2 = detectSequentialRead(disabled1.nextState, 128 * KiB, 64 * KiB, {
    ...opts,
    maxReadAheadBytes: 0,
  });
  assert.equal(disabled2.isSequential, true);
  assert.equal(disabled2.streak, 2);
  assert.equal(disabled2.readAheadBytes, 0);
});

test('RangeCache coalesces multi-chunk reads and batch readRanges into fewer HTTP 206 requests', async () => {
  const KiB = 1024;
  const fileBytes = makePatternBytes(1024 * KiB, 37); // 1 MiB archive (4 x 256 KiB chunks)
  const mock = createMockRangeFetch(new Map([['x64/world.rpf', fileBytes]]));

  const cache = new RangeCache({
    chunkSize: 256 * KiB,
    maxBytes: 8 * 1024 * KiB,
    enableReadAhead: false,
    coalesceAdjacent: true,
    coalesceMaxGapBytes: 64 * KiB,
    coalesceMaxMergedBytes: 2 * 1024 * KiB,
    fetchImpl: mock.fetchImpl,
  });

  // Read across chunks 0..2 (700 KiB) -> coalesced into ONE HTTP 206 fetch bytes=0-786431!
  const bigSlice = await cache.read('x64/world.rpf', 10 * KiB, 700 * KiB);
  assert.deepEqual(bigSlice, fileBytes.subarray(10 * KiB, 710 * KiB));
  assert.equal(mock.calls.length, 1);
  assert.equal(mock.calls[0].rangeHeader, `bytes=0-${768 * KiB - 1}`);

  const statsAfterRead = cache.getStats();
  assert.equal(statsAfterRead.coalescedRangeCount, 1);
  assert.ok(statsAfterRead.bytesSavedByCoalescing > 0);
  assert.equal(statsAfterRead.entries, 3);

  // Batch readRanges on a second file with 4 adjacent 64 KiB DirectStorage descriptors -> 1 HTTP 206 fetch!
  const bk2Bytes = makePatternBytes(1024 * KiB, 61);
  const mock2 = createMockRangeFetch(new Map([['movies/intro.bk2', bk2Bytes]]));
  const batchCache = new RangeCache({
    chunkSize: 64 * KiB,
    maxBytes: 8 * 1024 * KiB,
    enableReadAhead: false,
    coalesceAdjacent: true,
    fetchImpl: mock2.fetchImpl,
  });

  const slices = await batchCache.readRanges([
    { path: 'movies/intro.bk2', offset: 128 * KiB, length: 64 * KiB },
    { path: 'movies/intro.bk2', offset: 0, length: 64 * KiB },
    { path: 'movies/intro.bk2', offset: 64 * KiB, length: 64 * KiB },
    { path: 'movies/intro.bk2', offset: 192 * KiB, length: 64 * KiB },
  ]);

  assert.equal(slices.length, 4);
  assert.deepEqual(slices[0], bk2Bytes.subarray(128 * KiB, 192 * KiB));
  assert.deepEqual(slices[1], bk2Bytes.subarray(0, 64 * KiB));
  assert.deepEqual(slices[2], bk2Bytes.subarray(64 * KiB, 128 * KiB));
  assert.deepEqual(slices[3], bk2Bytes.subarray(192 * KiB, 256 * KiB));
  assert.equal(mock2.calls.length, 1, '4 adjacent 64 KiB reads must coalesce into 1 HTTP 206 fetch');
  assert.equal(mock2.calls[0].rangeHeader, `bytes=0-${256 * KiB - 1}`);

  const batchStats = batchCache.getStats();
  assert.equal(batchStats.coalescedRangeCount, 1);
  assert.ok(batchStats.bytesSavedByCoalescing > 0);
});

test('RangeCache adaptive sequential read-ahead accelerates consecutive 64 KiB DirectStorage and .bk2 streams and resets on random seek', async () => {
  const KiB = 1024;
  const MiB = 1024 * 1024;
  const totalBytes = 4 * MiB; // 16 chunks of 256 KiB
  const streamBytes = makePatternBytes(totalBytes, 83);
  const mock = createMockRangeFetch(new Map([['x64/dstorage_stream.rpf', streamBytes]]));

  const cache = new RangeCache({
    chunkSize: 256 * KiB,
    maxBytes: 16 * MiB,
    maxReadAheadBytes: 2 * MiB,
    fetchImpl: mock.fetchImpl,
  });

  // Stream 32 consecutive 64 KiB DirectStorage reads (2 MiB total = 8 chunks).
  for (let i = 0; i < 32; i += 1) {
    const offset = i * 64 * KiB;
    const out = await cache.read('x64/dstorage_stream.rpf', offset, 64 * KiB);
    assert.deepEqual(out, streamBytes.subarray(offset, offset + 64 * KiB));
    // Yield microtasks so background coalesced read-ahead completes ahead of next reads.
    await cache.flushPrefetches();
  }

  const stats = cache.getStats();
  assert.equal(stats.sequentialHits, 31);
  assert.ok(stats.prefetchHits >= 6, `Expected >= 6 prefetchHits, got ${stats.prefetchHits}`);
  assert.ok(stats.prefetchBytes >= 2 * MiB, `Expected >= 2 MiB prefetchBytes, got ${stats.prefetchBytes}`);
  assert.ok(stats.coalescedRangeCount >= 1, `Expected coalesced read-ahead fetches, got ${stats.coalescedRangeCount}`);
  assert.ok(stats.bytesSavedByCoalescing > 0);
  // 16 chunks (4 MiB) fetched in far fewer HTTP calls than 16 individual chunk fetches!
  assert.ok(
    mock.calls.length <= 6,
    `Expected <= 6 coalesced HTTP 206 requests for sequential stream, got ${mock.calls.length}`,
  );
  // Every single HTTP Range request must stay bounded by maxReadAheadBytes / coalesceMaxMergedBytes (2 MiB).
  for (const call of mock.calls) {
    const m = /^bytes=(\d+)-(\d+)$/.exec(call.rangeHeader);
    assert.ok(m);
    const span = Number(m[2]) - Number(m[1]) + 1;
    assert.ok(span <= 2 * MiB, `Coalesced range ${call.rangeHeader} exceeded 2 MiB cap`);
  }

  const seqStateBeforeSeek = cache.getSequentialState('x64/dstorage_stream.rpf');
  assert.equal(seqStateBeforeSeek.streak, 31);
  assert.equal(seqStateBeforeSeek.readAheadBytes, 2 * MiB);

  // Random seek back to offset 0 resets sequential streak & read-ahead window immediately.
  const seekOut = await cache.read('x64/dstorage_stream.rpf', 0, 64 * KiB);
  assert.deepEqual(seekOut, streamBytes.subarray(0, 64 * KiB));
  const seqStateAfterSeek = cache.getSequentialState('x64/dstorage_stream.rpf');
  assert.equal(seqStateAfterSeek.streak, 0);
  assert.equal(seqStateAfterSeek.readAheadBytes, 0);
});

test('RangeCache deduplicates demand read against in-flight read-ahead without double-counting prefetchHits', async () => {
  const KiB = 1024;
  const fileBytes = makePatternBytes(1024 * KiB, 19); // 4 chunks of 256 KiB
  const baseMock = createMockRangeFetch(new Map([['movies/cutscene_4k.bk2', fileBytes]]));

  let releasePrefetch = null;
  let delayedChunk1 = false;
  const delayedFetch = async (url, init) => {
    const range = init && init.headers && init.headers.Range;
    if (range && range.startsWith(`bytes=${256 * KiB}-`) && !delayedChunk1) {
      delayedChunk1 = true;
      await new Promise((resolve) => {
        releasePrefetch = resolve;
      });
    }
    return baseMock.fetchImpl(url, init);
  };

  const cache = new RangeCache({
    chunkSize: 256 * KiB,
    maxBytes: 8 * 1024 * KiB,
    maxReadAheadBytes: 256 * KiB,
    fetchImpl: delayedFetch,
  });

  // 3 sequential 64 KiB reads inside chunk 0 trigger background read-ahead for chunk 1.
  await cache.read('movies/cutscene_4k.bk2', 0, 64 * KiB);
  await cache.read('movies/cutscene_4k.bk2', 64 * KiB, 64 * KiB);
  await cache.read('movies/cutscene_4k.bk2', 128 * KiB, 64 * KiB);
  await cache.read('movies/cutscene_4k.bk2', 192 * KiB, 64 * KiB);

  assert.ok(typeof releasePrefetch === 'function', 'Chunk 1 background read-ahead should be in-flight');

  // Foreground demand read for chunk 1 arrives while background prefetch is still pending!
  const demandPromise = cache.read('movies/cutscene_4k.bk2', 256 * KiB, 64 * KiB);
  releasePrefetch();
  const slice = await demandPromise;
  await cache.flushPrefetches();

  assert.deepEqual(slice, fileBytes.subarray(256 * KiB, 320 * KiB));
  // Chunk 1 must have been fetched ONLY once (deduplicated against in-flight prefetch).
  const chunk1Calls = baseMock.calls.filter((c) => c.rangeHeader.startsWith(`bytes=${256 * KiB}-`));
  assert.equal(chunk1Calls.length, 1);

  // Repeat read inside chunk 1 must NOT double-count prefetchHits.
  await cache.read('movies/cutscene_4k.bk2', 320 * KiB, 64 * KiB);
  await cache.flushPrefetches();

  // Chunk 1 was prefetched and consumed once (chunk 2 may also have been prefetched but not yet read).
  assert.equal(cache.getStats().prefetchHits, 1);
});

test('RangeCache handles truncated 206 coalesced responses, 200 OK fallback, EOF clamping, and speculative prefetch errors safely', async () => {
  const KiB = 1024;
  const fileBytes = makePatternBytes(200 * KiB, 77); // 3 full 64 KiB chunks + 8 KiB tail chunk

  // 1. Server truncates a multi-chunk 206 response to only 64 KiB -> RangeCache falls back for uncovered chunks.
  const truncCalls = [];
  const truncFetch = async (url, init = {}) => {
    const range = init.headers && init.headers.Range;
    truncCalls.push(range);
    const m = /^bytes=(\d+)-(\d+)$/.exec(range || '');
    const start = Number(m[1]);
    const reqEnd = Number(m[2]);
    // Cap response at 64 KiB per request to simulate proxy/CDN range truncation.
    const actualEnd = Math.min(reqEnd, start + 64 * KiB - 1, fileBytes.byteLength - 1);
    const slice = fileBytes.slice(start, actualEnd + 1);
    return {
      ok: true,
      status: 206,
      headers: new Map([['Content-Range', `bytes ${start}-${actualEnd}/${fileBytes.byteLength}`]]),
      async arrayBuffer() {
        return slice.buffer;
      },
    };
  };

  const truncCache = new RangeCache({
    chunkSize: 64 * KiB,
    maxBytes: 4 * 1024 * KiB,
    enableReadAhead: false,
    coalesceAdjacent: true,
    fetchImpl: truncFetch,
  });

  const fullRead = await truncCache.read('trunc.rpf', 0, 200 * KiB);
  assert.deepEqual(fullRead, fileBytes);
  // First call requested all 4 chunks coalesced; fallback fetched the 3 uncovered chunks cleanly.
  assert.equal(truncCalls[0], `bytes=0-${256 * KiB - 1}`);
  assert.equal(truncCalls.length, 4);

  // 2. Server returns 200 OK full body on a coalesced request -> slices all chunks with 1 fetch.
  let ok200Calls = 0;
  const ok200Cache = new RangeCache({
    chunkSize: 64 * KiB,
    maxBytes: 4 * 1024 * KiB,
    enableReadAhead: false,
    coalesceAdjacent: true,
    fetchImpl: async () => {
      ok200Calls += 1;
      return {
        ok: true,
        status: 200,
        headers: new Map(),
        async arrayBuffer() {
          return fileBytes.slice().buffer;
        },
      };
    },
  });
  const ok200Out = await ok200Cache.read('full200.rpf', 10 * KiB, 150 * KiB);
  assert.deepEqual(ok200Out, fileBytes.subarray(10 * KiB, 160 * KiB));
  assert.equal(ok200Calls, 1);

  // 3. Background read-ahead 503 error must NEVER reject foreground reads, and EOF clamping prevents past-EOF fetches.
  let failChunk1 = true;
  const errCalls = [];
  const flakyFetch = async (url, init = {}) => {
    const range = init.headers && init.headers.Range;
    errCalls.push(range);
    const m = /^bytes=(\d+)-(\d+)$/.exec(range || '');
    const start = Number(m[1]);
    const reqEnd = Number(m[2]);
    if (start >= 64 * KiB && failChunk1) {
      failChunk1 = false;
      return {
        ok: false,
        status: 503,
        headers: new Map(),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }
    const actualEnd = Math.min(reqEnd, fileBytes.byteLength - 1);
    const slice = fileBytes.slice(start, actualEnd + 1);
    return {
      ok: true,
      status: 206,
      headers: new Map([['Content-Range', `bytes ${start}-${actualEnd}/${fileBytes.byteLength}`]]),
      async arrayBuffer() {
        return slice.buffer;
      },
    };
  };

  const resilientCache = new RangeCache({
    chunkSize: 64 * KiB,
    maxBytes: 4 * 1024 * KiB,
    maxReadAheadBytes: 128 * KiB,
    fetchImpl: flakyFetch,
  });

  // 3 sequential 16 KiB reads trigger read-ahead of chunk 1, which receives HTTP 503 and swallows cleanly.
  assert.deepEqual(await resilientCache.read('flaky.bk2', 0, 16 * KiB), fileBytes.subarray(0, 16 * KiB));
  assert.deepEqual(
    await resilientCache.read('flaky.bk2', 16 * KiB, 16 * KiB),
    fileBytes.subarray(16 * KiB, 32 * KiB),
  );
  assert.deepEqual(
    await resilientCache.read('flaky.bk2', 32 * KiB, 16 * KiB),
    fileBytes.subarray(32 * KiB, 48 * KiB),
  );
  await resilientCache.flushPrefetches();

  // Subsequent demand read into chunk 1 retries cleanly and succeeds!
  const chunk1Slice = await resilientCache.read('flaky.bk2', 64 * KiB, 16 * KiB);
  assert.deepEqual(chunk1Slice, fileBytes.subarray(64 * KiB, 80 * KiB));
  await resilientCache.flushPrefetches();

  // Reading to EOF must not issue any range starting at or beyond fileBytes.byteLength (200 KiB).
  const tailSlice = await resilientCache.read('flaky.bk2', 192 * KiB, 16 * KiB);
  assert.deepEqual(tailSlice, fileBytes.subarray(192 * KiB, 200 * KiB));
  await resilientCache.flushPrefetches();
  for (const r of errCalls) {
    const start = Number(/^bytes=(\d+)-/.exec(r)[1]);
    assert.ok(start < fileBytes.byteLength, `Unexpected past-EOF Range request: ${r}`);
  }
});

test('RangeCache prefetchBootset with coalesce merges adjacent bootset entries and tracks prefetch telemetry', async () => {
  const KiB = 1024;
  const archive = makePatternBytes(512 * KiB, 45);
  const mock = createMockRangeFetch(new Map([['x64/boot.rpf', archive]]));

  const cache = new RangeCache({
    chunkSize: 64 * KiB,
    maxBytes: 4 * 1024 * KiB,
    enableReadAhead: false,
    coalesceAdjacent: true,
    fetchImpl: mock.fetchImpl,
  });

  const summary = await cache.prefetchBootset(
    [
      { path: 'x64/boot.rpf', offset: 0, length: 64 * KiB },
      { path: 'x64/boot.rpf', offset: 64 * KiB, length: 64 * KiB },
      { path: 'x64/boot.rpf', offset: 128 * KiB, length: 64 * KiB },
      { path: 'x64/boot.rpf', offset: 160 * KiB, length: 64 * KiB }, // overlaps 32 KiB
    ],
    { coalesce: true },
  );

  assert.equal(summary.requestedEntries, 4);
  assert.equal(summary.coalescedRanges, 1);
  assert.equal(summary.uniqueChunks, 4);
  assert.equal(summary.prefetched, 4);
  assert.equal(summary.failed, 0);
  assert.equal(mock.calls.length, 1, '4 adjacent/overlapping bootset entries must merge into 1 HTTP 206 fetch');

  const statsBeforeRead = cache.getStats();
  assert.equal(statsBeforeRead.coalescedRangeCount, 1);
  assert.equal(statsBeforeRead.prefetchBytes, 256 * KiB);
  assert.ok(statsBeforeRead.bytesSavedByCoalescing >= 32 * KiB);

  const out = await cache.read('x64/boot.rpf', 0, 224 * KiB);
  assert.deepEqual(out, archive.subarray(0, 224 * KiB));
  assert.equal(mock.calls.length, 1);
  assert.equal(cache.getStats().prefetchHits, 4);
});

// --- W262: Versioned OPFS L2 write-through persistence & fingerprint invalidation tests ---

test('W262: buildFileFingerprint and opfsChunkFileName preserve legacy format and support versioned size/ETag keys', () => {
  assert.equal(DEFAULT_MAX_PENDING_OPFS_WRITES, 64);
  assert.equal(DEFAULT_MAX_OPFS_WRITE_CONCURRENCY, 4);

  // Backward compatibility when no fingerprint is provided.
  const legacy = opfsChunkFileName('game/shaders.rpf', 2, 262144);
  assert.equal(legacy, opfsChunkFileName('game/shaders.rpf', 2, 262144, null));
  assert.match(legacy, /^c_262144_2_[0-9a-f]+\.bin$/);

  // Size-only fingerprint: <size>.
  const fpSize = buildFileFingerprint(1048576);
  assert.equal(fpSize, '1048576');
  const sizeKey = opfsChunkFileName('game/shaders.rpf', 2, 262144, fpSize);
  assert.match(sizeKey, /^c_262144_2_[0-9a-f]+_fp_1048576\.bin$/);
  assert.match(sizeKey, /^c_\d+_\d+_.*\.bin$/);

  // Size + ETag fingerprint: <size>_<etagHash>.
  const fpEtag1 = buildFileFingerprint({ size: 1048576, etag: '"rpf-v1"' });
  const fpEtag2 = buildFileFingerprint({ size: 1048576, etag: '"rpf-v2"' });
  assert.match(fpEtag1, /^1048576_[0-9a-f]{16}$/);
  assert.notEqual(fpEtag1, fpEtag2);
  assert.equal(fpEtag1, buildFileFingerprint(1048576, '"rpf-v1"'));
  const etagKey = opfsChunkFileName('game/shaders.rpf', 2, 262144, fpEtag1);
  assert.match(etagKey, /^c_262144_2_[0-9a-f]+_fp_1048576_[0-9a-f]{16}\.bin$/);
});

test('W262: cold HTTP fetch triggers non-blocking background OPFS L2 write and warm reload hits OPFS with 0 HTTP fetches', async () => {
  const shadersBytes = makePatternBytes(64, 11);
  const bvhBytes = makePatternBytes(48, 22);
  const moviesBytes = makePatternBytes(80, 33);

  const filesMap = new Map([
    ['game/shaders.rpf', shadersBytes],
    ['game/core_bvh4.rpf', bvhBytes],
    ['game/movies.rpf', moviesBytes],
  ]);

  const fingerprints = {
    'game/shaders.rpf': buildFileFingerprint(shadersBytes.byteLength, '"shaders-v1"'),
    'game/core_bvh4.rpf': buildFileFingerprint(bvhBytes.byteLength),
    'game/movies.rpf': buildFileFingerprint(moviesBytes.byteLength, '"movies-v1"'),
  };

  // Wrap OPFS mock with a manual gate on writes to PROVE read() never blocks on OPFS write-through!
  const baseOpfs = createQuotaMockOpfsDir();
  let gateWrites = true;
  const pendingWriteGates = [];
  const gatedOpfs = {
    files: baseOpfs.files,
    async getFileHandle(name, opts = {}) {
      const inner = await baseOpfs.getFileHandle(name, opts);
      if (!opts.create) return inner;
      return {
        getFile: inner.getFile.bind(inner),
        async createWritable() {
          const writable = await inner.createWritable();
          return {
            async write(chunk) {
              if (gateWrites) {
                await new Promise((resolve) => pendingWriteGates.push(resolve));
              }
              return writable.write(chunk);
            },
            close: writable.close.bind(writable),
            abort: writable.abort.bind(writable),
          };
        },
      };
    },
    removeEntry: baseOpfs.removeEntry.bind(baseOpfs),
    entries: baseOpfs.entries.bind(baseOpfs),
  };

  const coldFetch = createMockRangeFetch(filesMap);
  const coldCache = new RangeCache({
    chunkSize: 16,
    maxBytes: 256,
    fetchImpl: coldFetch.fetchImpl,
    opfsDir: gatedOpfs,
    fileFingerprints: fingerprints,
  });

  // Cold HTTP reads resolve IMMEDIATELY even though OPFS writes are gated/blocked!
  const s0 = await coldCache.read('game/shaders.rpf', 0, 32);
  const b0 = await coldCache.read('game/core_bvh4.rpf', 0, 16);
  const m0 = await coldCache.read('game/movies.rpf', 16, 32);

  assert.deepEqual(s0, shadersBytes.subarray(0, 32));
  assert.deepEqual(b0, bvhBytes.subarray(0, 16));
  assert.deepEqual(m0, moviesBytes.subarray(16, 48));
  assert.equal(coldFetch.calls.length, 5);
  // OPFS writes have not completed yet because gateWrites === true -> proves non-blocking return!
  assert.equal(coldCache.getStats().opfsWrites, 0);

  // Unblock background OPFS writes and drain.
  gateWrites = false;
  while (pendingWriteGates.length > 0) {
    pendingWriteGates.shift()();
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  await coldCache.flushOpfsWrites();

  const coldStats = coldCache.getStats();
  assert.equal(coldStats.opfsHits, 0);
  assert.equal(coldStats.opfsMisses, 5);
  assert.equal(coldStats.opfsWrites, 5);
  assert.equal(coldStats.opfsWriteErrors, 0);
  assert.equal(coldStats.opfsStaleInvalidations, 0);
  assert.equal(coldStats.opfsHitRate, 0);

  // Brand-new RangeCache instance (simulating page reload with empty RAM LRU) -> 0 HTTP fetches!
  const warmFetch = createMockRangeFetch(filesMap);
  const warmCache = new RangeCache({
    chunkSize: 16,
    maxBytes: 256,
    fetchImpl: warmFetch.fetchImpl,
    opfsDir: baseOpfs,
    fileFingerprints: fingerprints,
  });

  assert.deepEqual(await warmCache.read('game/shaders.rpf', 0, 32), shadersBytes.subarray(0, 32));
  assert.deepEqual(await warmCache.read('game/core_bvh4.rpf', 0, 16), bvhBytes.subarray(0, 16));
  assert.deepEqual(await warmCache.read('game/movies.rpf', 16, 32), moviesBytes.subarray(16, 48));
  assert.equal(warmFetch.calls.length, 0, 'Warm reload must serve all hot RPF ranges from OPFS with 0 HTTP fetches');

  const warmStats = warmCache.getStats();
  assert.equal(warmStats.opfsHits, 5);
  assert.equal(warmStats.opfsMisses, 0);
  assert.equal(warmStats.opfsHitRate, 1);
  assert.equal(warmStats.opfsStaleInvalidations, 0);
});

test('W262: changing file size or ETag fingerprint invalidates stale OPFS entries and fetches fresh bytes', async () => {
  const v1Bytes = makePatternBytes(64, 10);
  const v2Bytes = makePatternBytes(96, 99);
  const v3Bytes = makePatternBytes(96, 200); // same size as v2, different ETag!
  const opfsDir = createQuotaMockOpfsDir();

  // Session 1: persist shaders.rpf v1 (size 64, ETag "v1").
  const fpV1 = buildFileFingerprint(64, '"v1"');
  const fetchV1 = createMockRangeFetch(new Map([['game/shaders.rpf', v1Bytes]]));
  const cacheV1 = new RangeCache({
    chunkSize: 16,
    maxBytes: 256,
    fetchImpl: fetchV1.fetchImpl,
    opfsDir,
    fileFingerprints: { 'game/shaders.rpf': fpV1 },
  });
  assert.deepEqual(await cacheV1.read('game/shaders.rpf', 0, 32), v1Bytes.subarray(0, 32));
  await cacheV1.flushOpfsWrites();
  const v1Chunk0File = opfsChunkFileName('game/shaders.rpf', 0, 16, fpV1);
  const v1Chunk1File = opfsChunkFileName('game/shaders.rpf', 1, 16, fpV1);
  assert.equal(opfsDir.files.has(v1Chunk0File), true);
  assert.equal(opfsDir.files.has(v1Chunk1File), true);

  // Session 2: archive changes size (64 -> 96) and ETag ("v1" -> "v2").
  const fpV2 = buildFileFingerprint(96, '"v2"');
  const fetchV2 = createMockRangeFetch(new Map([['game/shaders.rpf', v2Bytes]]));
  const cacheV2 = new RangeCache({
    chunkSize: 16,
    maxBytes: 256,
    fetchImpl: fetchV2.fetchImpl,
    opfsDir,
    fileFingerprints: { 'game/shaders.rpf': fpV2 },
  });
  const freshV2 = await cacheV2.read('game/shaders.rpf', 0, 32);
  await cacheV2.flushOpfsWrites();
  assert.deepEqual(freshV2, v2Bytes.subarray(0, 32));
  assert.equal(fetchV2.calls.length, 2, 'Stale fingerprint must re-fetch fresh chunks over HTTP');

  const statsV2 = cacheV2.getStats();
  assert.equal(statsV2.opfsStaleInvalidations, 2);
  assert.equal(statsV2.opfsMisses, 2);
  assert.equal(statsV2.opfsHits, 0);
  assert.equal(statsV2.opfsWrites, 2);
  assert.equal(opfsDir.files.has(v1Chunk0File), false, 'Stale v1 chunk 0 must be deleted from OPFS');
  assert.equal(opfsDir.files.has(v1Chunk1File), false, 'Stale v1 chunk 1 must be deleted from OPFS');
  assert.equal(opfsDir.files.has(opfsChunkFileName('game/shaders.rpf', 0, 16, fpV2)), true);

  // Session 3: same size (96) but different ETag ("v2" -> "v3") via setFileFingerprint on same instance!
  const fpV3 = buildFileFingerprint(96, '"v3"');
  const fetchV3 = createMockRangeFetch(new Map([['game/shaders.rpf', v3Bytes]]));
  cacheV2.fetchImpl = fetchV3.fetchImpl;
  cacheV2.setFileFingerprint('game/shaders.rpf', fpV3);

  const freshV3 = await cacheV2.read('game/shaders.rpf', 0, 16);
  await cacheV2.flushOpfsWrites();
  assert.deepEqual(freshV3, v3Bytes.subarray(0, 16));
  assert.equal(fetchV3.calls.length, 1);
  assert.equal(cacheV2.getStats().opfsStaleInvalidations, 3);
  assert.equal(opfsDir.files.has(opfsChunkFileName('game/shaders.rpf', 0, 16, fpV2)), false);
  assert.equal(opfsDir.files.has(opfsChunkFileName('game/shaders.rpf', 0, 16, fpV3)), true);
});

test('W262: throwing OPFS mock and bounded background write queue degrade gracefully without failing HTTP reads', async () => {
  const fileBytes = makePatternBytes(128, 63);
  const mockFetch = createMockRangeFetch(new Map([['game/levels.rpf', fileBytes]]));

  // 1. Throwing non-quota OPFS mock (e.g., SecurityError / NotAllowedError) degrades cleanly.
  const throwingOpfs = {
    async getFileHandle(_name, opts = {}) {
      if (!opts.create) {
        const err = new Error('NotFoundError');
        err.name = 'NotFoundError';
        throw err;
      }
      const err = new Error('OPFS access denied');
      err.name = 'NotAllowedError';
      throw err;
    },
    async removeEntry() {},
  };

  const errCache = new RangeCache({
    chunkSize: 16,
    maxBytes: 128,
    fetchImpl: mockFetch.fetchImpl,
    opfsDir: throwingOpfs,
  });

  const out = await errCache.read('game/levels.rpf', 0, 32);
  await errCache.flushOpfsWrites();
  assert.deepEqual(out, fileBytes.subarray(0, 32));
  const errStats = errCache.getStats();
  assert.equal(errStats.opfsWriteErrors, 2);
  assert.equal(errStats.opfsWrites, 0);
  assert.equal(errStats.opfsHits, 0);
  assert.equal(errStats.opfsMisses, 2);
  assert.equal(errStats.opfsHitRate, 0);

  // 2. Bounded queue (maxPendingOpfsWrites = 2, maxOpfsWriteConcurrency = 1) drops oldest queued writes under backpressure without blocking reads.
  const baseOpfs = createQuotaMockOpfsDir();
  const gates = [];
  const slowOpfs = {
    files: baseOpfs.files,
    async getFileHandle(name, opts = {}) {
      const inner = await baseOpfs.getFileHandle(name, opts);
      if (!opts.create) return inner;
      return {
        getFile: inner.getFile.bind(inner),
        async createWritable() {
          const w = await inner.createWritable();
          return {
            async write(chunk) {
              await new Promise((resolve) => gates.push(resolve));
              return w.write(chunk);
            },
            close: w.close.bind(w),
            abort: w.abort.bind(w),
          };
        },
      };
    },
    removeEntry: baseOpfs.removeEntry.bind(baseOpfs),
    entries: baseOpfs.entries.bind(baseOpfs),
  };

  const boundedFetch = createMockRangeFetch(new Map([['game/levels.rpf', fileBytes]]));
  const boundedCache = new RangeCache({
    chunkSize: 16,
    maxBytes: 256,
    maxPendingOpfsWrites: 2,
    maxOpfsWriteConcurrency: 1,
    fetchImpl: boundedFetch.fetchImpl,
    opfsDir: slowOpfs,
  });

  // Burst 6 reads while OPFS writes are stalled: all 6 reads return exact bytes immediately!
  for (let i = 0; i < 6; i += 1) {
    const slice = await boundedCache.read('game/levels.rpf', i * 16, 16);
    assert.deepEqual(slice, fileBytes.subarray(i * 16, (i + 1) * 16));
  }
  // At most 1 active + 2 queued = 3 writes should persist after draining!
  while (boundedCache._activeOpfsWrites.size > 0 || boundedCache._opfsWriteQueue.size > 0) {
    if (gates.length > 0) {
      gates.shift()();
    }
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  await boundedCache.flushOpfsWrites();
  assert.ok(
    boundedCache.getStats().opfsWrites <= 3,
    `Expected bounded queue to cap persisted writes to <= 3, got ${boundedCache.getStats().opfsWrites}`,
  );
  assert.ok(boundedCache.getStats().opfsWrites >= 1);
});

test('W262: RangeCache interoperates with OpfsChunkCache L2 backend including versioned fingerprint validation', async () => {
  const v1 = makePatternBytes(48, 5);
  const v2 = makePatternBytes(48, 88);
  const memDir = createQuotaMockOpfsDir();

  const opfsCache = new OpfsChunkCache({ cacheDir: memDir, indexFlushDelayMs: 0 });
  const fp1 = buildFileFingerprint(48, '"etag-1"');
  const fetch1 = createMockRangeFetch(new Map([['game/core_bvh4.rpf', v1]]));

  const cold = new RangeCache({
    chunkSize: 16,
    maxBytes: 128,
    fetchImpl: fetch1.fetchImpl,
    opfsCache,
    fileFingerprints: { 'game/core_bvh4.rpf': fp1 },
  });
  assert.deepEqual(await cold.read('game/core_bvh4.rpf', 0, 32), v1.subarray(0, 32));
  await cold.flushOpfsWrites();
  assert.equal(cold.getStats().opfsWrites, 2);

  // Warm reload with matching fingerprint hits OpfsChunkCache with 0 HTTP fetches!
  const fetch2 = createMockRangeFetch(new Map([['game/core_bvh4.rpf', v1]]));
  const warm = new RangeCache({
    chunkSize: 16,
    maxBytes: 128,
    fetchImpl: fetch2.fetchImpl,
    opfsCache,
    fileFingerprints: { 'game/core_bvh4.rpf': fp1 },
  });
  assert.deepEqual(await warm.read('game/core_bvh4.rpf', 0, 32), v1.subarray(0, 32));
  assert.equal(fetch2.calls.length, 0);
  assert.equal(warm.getStats().opfsHits, 2);
  assert.equal(warm.getStats().opfsHitRate, 1);

  // Reload with updated fingerprint invalidates stale OpfsChunkCache entry and fetches v2!
  const fp2 = buildFileFingerprint(48, '"etag-2"');
  const fetch3 = createMockRangeFetch(new Map([['game/core_bvh4.rpf', v2]]));
  const stale = new RangeCache({
    chunkSize: 16,
    maxBytes: 128,
    fetchImpl: fetch3.fetchImpl,
    opfsCache,
    fileFingerprints: { 'game/core_bvh4.rpf': fp2 },
  });
  assert.deepEqual(await stale.read('game/core_bvh4.rpf', 0, 32), v2.subarray(0, 32));
  await stale.flushOpfsWrites();
  assert.equal(fetch3.calls.length, 2);
  assert.equal(stale.getStats().opfsStaleInvalidations, 2);
  await opfsCache.close();
});
