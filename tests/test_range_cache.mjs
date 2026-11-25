import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { RangeCache, opfsChunkFileName, isQuotaExceededError } = require('../web/workers/range_cache.js');

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
  assert.equal(stats.evictions, 0);
  assert.equal(stats.quotaEvictions, 0);
  assert.equal(stats.quotaRetrySuccesses, 0);
  assert.equal(stats.corruptEvictions, 0);
  assert.equal(stats.bytesCached, 0);
  assert.equal(stats.entries, 0);

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
  await cache.read('world.rpf', 16, 16);
  await cache.read('world.rpf', 32, 16);
  await cache.read('world.rpf', 48, 16);
  assert.equal(opfsDir.files.size, 4);

  // Touch chunk 0 so LRU order becomes [1, 2, 3, 0].
  await cache.read('world.rpf', 0, 8);

  // Reading chunk 4 triggers QuotaExceededError on OPFS write, evicts oldest entry (chunk 1),
  // and retries once cleanly.
  const chunk4 = await cache.read('world.rpf', 64, 16);
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
  }
  assert.equal(kvFiles.size, 6);

  // Chunk 6 overflows the 96-byte KV quota -> evicts >= 64 bytes (4 oldest chunks), retries once.
  const chunk6 = await cache.read('big.rpf', 6 * 16, 16);
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
  await syncCache.read('big.rpf', 16, 16);
  const chunk2 = await syncCache.read('big.rpf', 32, 16);
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
  assert.deepEqual(slice, fileBytes.subarray(4, 28));

  const stats = cache.getStats();
  assert.equal(stats.quotaRetrySuccesses, 0);
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
