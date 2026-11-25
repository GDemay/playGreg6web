import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { Worker as NodeWorker } from 'node:worker_threads';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');

const mailboxMod = require('../web/workers/httpfs_mailbox.js');

const {
  HTTPFS_MAILBOX_VERSION,
  HTTPFS_HEADER_BYTES,
  HTTPFS_PAYLOAD_BYTES,
  HTTPFS_DEFAULT_CACHE_MAX_BYTES,
  HTTPFS_DEFAULT_CACHE_CHUNK_SIZE,
  HTTPFS_DEFAULT_READAHEAD_CHUNKS,
  HttpfsStatus,
  HttpfsOp,
  HttpfsOffsets,
  isCacheableImmutableUrl,
  resolveHttpfsUrl,
  HttpfsLruChunkCache,
  createHttpfsMailboxBuffer,
  createHttpfsMailboxPool,
  HttpfsMailboxHost,
  HttpfsMailboxClient,
  HttpfsMailboxPoolHost,
  HttpfsMailboxPoolClient,
} = mailboxMod;

function makePatternBytes(length, seed = 29) {
  const out = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = ((i * 73) ^ ((i >> 5) * 41) ^ ((i >> 11) * 17) ^ seed) & 0xff;
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

function createMockRangeFetch(urlBytesMap, options = {}) {
  const requests = [];
  const statusOverrides = options.statusOverrides || new Map();
  const maxServeBytes = Number.isInteger(options.maxServeBytes) ? options.maxServeBytes : null;
  const force200 = Boolean(options.force200);
  const omitTotal = Boolean(options.omitTotal);
  const delayMs = options.delayMs || 0;
  const onRequest = typeof options.onRequest === 'function' ? options.onRequest : null;

  async function fetchImpl(url, init = {}) {
    const method = (init && init.method) || 'GET';
    const rangeHeader =
      (init && init.headers && (init.headers.Range || init.headers.range)) || null;
    const reqInfo = { method, url, rangeHeader, index: requests.length };
    requests.push(reqInfo);

    if (onRequest) {
      await onRequest(reqInfo);
    }
    if (delayMs > 0) {
      await new Promise((r) => setTimeout(r, delayMs));
    }

    const override = statusOverrides.get(url);
    const overrideStatus = typeof override === 'function' ? override(reqInfo) : override;
    if (overrideStatus) {
      return {
        ok: false,
        status: overrideStatus,
        headers: new Map(),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
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
      };
    }

    if (!rangeHeader || force200) {
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

    let actualEnd = Math.min(reqEnd, data.byteLength - 1);
    if (maxServeBytes !== null) {
      actualEnd = Math.min(actualEnd, reqStart + maxServeBytes - 1);
    }
    const slice = data.slice(reqStart, actualEnd + 1);
    const totalToken = omitTotal ? '*' : String(data.byteLength);
    return {
      ok: true,
      status: 206,
      headers: new Map([
        ['Content-Range', `bytes ${reqStart}-${actualEnd}/${totalToken}`],
        ['Content-Length', String(slice.byteLength)],
      ]),
      async arrayBuffer() {
        return slice.buffer;
      },
    };
  }

  return { fetchImpl, requests };
}

// ---------------------------------------------------------------------------
// 1. Constants, URL classification, and URL resolution
// ---------------------------------------------------------------------------

test('exports cache defaults, immutable URL classification, and HTTPFS URL routing', () => {
  assert.equal(HTTPFS_MAILBOX_VERSION, 1);
  assert.equal(HTTPFS_HEADER_BYTES, 40);
  assert.equal(HTTPFS_PAYLOAD_BYTES, 64 * 1024);
  assert.equal(HTTPFS_DEFAULT_CACHE_MAX_BYTES, 32 * 1024 * 1024);
  assert.equal(HTTPFS_DEFAULT_CACHE_CHUNK_SIZE, 256 * 1024);
  assert.equal(HTTPFS_DEFAULT_READAHEAD_CHUNKS, 2);

  // Immutable /game/*.rpf and /shaders/* paths are cacheable.
  assert.equal(isCacheableImmutableUrl('/game/shaders.rpf'), true);
  assert.equal(isCacheableImmutableUrl('/game/x64/levels.rpf'), true);
  assert.equal(isCacheableImmutableUrl('/game/UPDATE/MOVIES.RPF'), true);
  assert.equal(isCacheableImmutableUrl('http://127.0.0.1:8080/game/textures.rpf?rev=1'), true);
  assert.equal(isCacheableImmutableUrl('/shaders/splash.wgsl.pack'), true);
  assert.equal(isCacheableImmutableUrl('/shaders/full.wgsl.pack.reflect.json'), true);
  assert.equal(isCacheableImmutableUrl('http://localhost:8000/shaders/pass0/main.wgsl'), true);
  assert.equal(isCacheableImmutableUrl('game/shaders.rpf'), true);
  assert.equal(isCacheableImmutableUrl('shaders/pack.wgsl.pack'), true);

  // Non-immutable paths must bypass the LRU chunk cache.
  assert.equal(isCacheableImmutableUrl('/game/config.xml'), false);
  assert.equal(isCacheableImmutableUrl('/game/'), false);
  assert.equal(isCacheableImmutableUrl('/shaders/'), false);
  assert.equal(isCacheableImmutableUrl('/userdata/profile.sav'), false);
  assert.equal(isCacheableImmutableUrl('/boot/boot-options.json'), false);
  assert.equal(isCacheableImmutableUrl('/data/dynamic.bin'), false);
  assert.equal(isCacheableImmutableUrl(''), false);
  assert.equal(isCacheableImmutableUrl(null), false);

  // URL resolution preserves /game/ and /shaders/ routing cleanly.
  assert.equal(resolveHttpfsUrl('game/shaders.rpf'), '/game/shaders.rpf');
  assert.equal(resolveHttpfsUrl('levels.rpf'), '/game/levels.rpf');
  assert.equal(resolveHttpfsUrl('x64/textures.rpf'), '/game/x64/textures.rpf');
  assert.equal(resolveHttpfsUrl('shaders/splash.wgsl.pack'), '/shaders/splash.wgsl.pack');
  assert.equal(resolveHttpfsUrl('boot/boot-options.json'), '/boot/boot-options.json');
  assert.equal(
    resolveHttpfsUrl('game/movies.rpf', { baseUrl: 'http://127.0.0.1:8080' }),
    'http://127.0.0.1:8080/game/movies.rpf',
  );
  assert.equal(
    resolveHttpfsUrl('shaders/pack.wgsl.pack', { baseUrl: 'http://127.0.0.1:8080/' }),
    'http://127.0.0.1:8080/shaders/pack.wgsl.pack',
  );
});

// ---------------------------------------------------------------------------
// 2. Cache hits, buffer isolation, and non-cacheable bypass
// ---------------------------------------------------------------------------

test('HttpfsLruChunkCache serves cache hits with zero HTTP round-trips, isolates returned buffers, and bypasses non-immutable URLs', async () => {
  const chunkSize = 4096;
  const rpfBytes = makePatternBytes(16 * 1024, 17);
  const shaderBytes = makePatternBytes(12 * 1024, 31);
  const saveBytes = makePatternBytes(8 * 1024, 53);

  const mock = createMockRangeFetch(
    new Map([
      ['/game/shaders.rpf', rpfBytes],
      ['/shaders/full.wgsl.pack', shaderBytes],
      ['/userdata/save0.bin', saveBytes],
    ]),
  );

  const cache = new HttpfsLruChunkCache({
    fetchImpl: mock.fetchImpl,
    cacheMaxBytes: 64 * 1024,
    chunkSize,
    readAheadChunks: 0,
  });

  // Cold miss fetches aligned chunk 0 [0..4095].
  const first = await cache.read('/game/shaders.rpf', 100, 500);
  assert.deepEqual(first, rpfBytes.subarray(100, 600));
  assert.equal(mock.requests.length, 1);
  assert.equal(mock.requests[0].rangeHeader, 'bytes=0-4095');

  // Mutating caller's buffer must NOT poison the cached chunk.
  first.fill(0xff);

  // Repeat read inside same chunk hits cache with ZERO additional HTTP requests.
  const second = await cache.read('/game/shaders.rpf', 100, 500);
  assert.deepEqual(second, rpfBytes.subarray(100, 600));
  assert.equal(mock.requests.length, 1);

  // Different sub-range in chunk 0 also hits synchronously via tryReadSync.
  const syncSlice = cache.tryReadSync('/game/shaders.rpf', 2048, 1024);
  assert.deepEqual(syncSlice, rpfBytes.subarray(2048, 3072));
  assert.equal(mock.requests.length, 1);

  // /shaders/* assets are also cached.
  const sh1 = await cache.read('/shaders/full.wgsl.pack', 0, 1024);
  assert.deepEqual(sh1, shaderBytes.subarray(0, 1024));
  assert.equal(mock.requests.length, 2);
  const sh2 = await cache.read('/shaders/full.wgsl.pack', 512, 1024);
  assert.deepEqual(sh2, shaderBytes.subarray(512, 1536));
  assert.equal(mock.requests.length, 2);

  // Non-immutable URL bypasses LRU cache completely (exact range fetched every time).
  const s1 = await cache.read('/userdata/save0.bin', 200, 300);
  assert.deepEqual(s1, saveBytes.subarray(200, 500));
  assert.equal(mock.requests.length, 3);
  assert.equal(mock.requests[2].rangeHeader, 'bytes=200-499');
  assert.equal(cache.tryReadSync('/userdata/save0.bin', 200, 300), null);

  const s2 = await cache.read('/userdata/save0.bin', 200, 300);
  assert.deepEqual(s2, saveBytes.subarray(200, 500));
  assert.equal(mock.requests.length, 4);

  const stats = cache.getStats();
  assert.equal(stats.cacheMisses, 4); // rpf chunk0 + shader chunk0 + 2 uncached save reads
  assert.equal(stats.cacheHits, 3); // rpf repeat + rpf sync + shader repeat
  assert.equal(stats.cacheBytes, 2 * chunkSize);
  assert.equal(stats.cacheEvictions, 0);
  assert.equal(stats.bytesServedFromCache, 500 + 1024 + 1024);
  assert.equal(stats.bytesFetchedFromNetwork, 4096 + 4096 + 300 + 300);
});

// ---------------------------------------------------------------------------
// 3. Strict LRU eviction & memory bound enforcement
// ---------------------------------------------------------------------------

test('HttpfsLruChunkCache enforces cacheMaxBytes strictly and evicts least-recently-used chunks', async () => {
  const chunkSize = 4096;
  const rpfBytes = makePatternBytes(5 * chunkSize, 43); // 5 chunks: 0..4
  const mock = createMockRangeFetch(new Map([['/game/levels.rpf', rpfBytes]]));

  // Budget fits strictly 2 chunks (8 KiB).
  const cache = new HttpfsLruChunkCache({
    fetchImpl: mock.fetchImpl,
    cacheMaxBytes: 2 * chunkSize,
    chunkSize,
    readAheadChunks: 0,
  });

  // Load chunk 0 and chunk 1 -> LRU order: [0, 1].
  assert.deepEqual(await cache.read('/game/levels.rpf', 0, 256), rpfBytes.subarray(0, 256));
  assert.deepEqual(
    await cache.read('/game/levels.rpf', chunkSize, 256),
    rpfBytes.subarray(chunkSize, chunkSize + 256),
  );
  assert.equal(cache.getStats().cacheBytes, 2 * chunkSize);
  assert.equal(cache.getStats().cacheEvictions, 0);

  // Touch chunk 0 via cache hit -> LRU order becomes [1, 0].
  assert.deepEqual(
    cache.tryReadSync('/game/levels.rpf', 512, 128),
    rpfBytes.subarray(512, 640),
  );

  // Load chunk 2 -> MUST evict chunk 1 (LRU), preserving chunk 0! LRU: [0, 2].
  assert.deepEqual(
    await cache.read('/game/levels.rpf', 2 * chunkSize, 256),
    rpfBytes.subarray(2 * chunkSize, 2 * chunkSize + 256),
  );
  let stats = cache.getStats();
  assert.equal(stats.cacheEvictions, 1);
  assert.equal(stats.cacheBytes, 2 * chunkSize);
  assert.equal(stats.cacheEntries, 2);

  // Chunk 0 is STILL cached (0 new fetches).
  const reqsBefore = mock.requests.length;
  assert.deepEqual(
    cache.tryReadSync('/game/levels.rpf', 1024, 512),
    rpfBytes.subarray(1024, 1536),
  );
  assert.equal(mock.requests.length, reqsBefore);

  // Chunk 1 was evicted -> tryReadSync returns null; async read re-fetches and evicts chunk 2.
  assert.equal(cache.tryReadSync('/game/levels.rpf', chunkSize + 64, 64), null);
  assert.deepEqual(
    await cache.read('/game/levels.rpf', chunkSize + 64, 64),
    rpfBytes.subarray(chunkSize + 64, chunkSize + 128),
  );
  assert.equal(mock.requests.length, reqsBefore + 1);
  stats = cache.getStats();
  assert.equal(stats.cacheEvictions, 2);
  assert.ok(stats.cacheBytes <= 2 * chunkSize);

  // Chunk larger than cacheMaxBytes is served cleanly without exceeding cacheMaxBytes.
  const tinyCache = new HttpfsLruChunkCache({
    fetchImpl: mock.fetchImpl,
    cacheMaxBytes: 2048, // < chunkSize (4096)
    chunkSize: 4096,
    readAheadChunks: 2,
  });
  assert.deepEqual(
    await tinyCache.read('/game/levels.rpf', 0, 512),
    rpfBytes.subarray(0, 512),
  );
  assert.equal(tinyCache.getStats().cacheBytes, 0);
  assert.equal(tinyCache.getStats().prefetchIssued, 0);
});

// ---------------------------------------------------------------------------
// 4. Safe opt-out (cacheMaxBytes: 0)
// ---------------------------------------------------------------------------

test('cacheMaxBytes: 0 cleanly disables caching and prefetching while serving exact HTTP ranges', async () => {
  const rpfBytes = makePatternBytes(32 * 1024, 67);
  const mock = createMockRangeFetch(new Map([['/game/textures.rpf', rpfBytes]]));

  const cache = new HttpfsLruChunkCache({
    fetchImpl: mock.fetchImpl,
    cacheMaxBytes: 0,
    chunkSize: 4096,
    readAheadChunks: 2,
  });

  const a = await cache.read('/game/textures.rpf', 100, 500);
  const b = await cache.read('/game/textures.rpf', 100, 500);
  const c = await cache.read('/game/textures.rpf', 600, 500);

  assert.deepEqual(a, rpfBytes.subarray(100, 600));
  assert.deepEqual(b, rpfBytes.subarray(100, 600));
  assert.deepEqual(c, rpfBytes.subarray(600, 1100));

  // Every read issued a direct exact-range fetch; no chunk amplification or prefetching.
  assert.equal(mock.requests.length, 3);
  assert.deepEqual(
    mock.requests.map((r) => r.rangeHeader),
    ['bytes=100-599', 'bytes=100-599', 'bytes=600-1099'],
  );

  const stats = cache.getStats();
  assert.equal(stats.enabled, false);
  assert.equal(stats.cacheMaxBytes, 0);
  assert.equal(stats.cacheHits, 0);
  assert.equal(stats.cacheMisses, 3);
  assert.equal(stats.cacheBytes, 0);
  assert.equal(stats.cacheEvictions, 0);
  assert.equal(stats.prefetchIssued, 0);
  assert.equal(stats.prefetchHits, 0);
  assert.equal(stats.coalescedReads, 0);
  assert.equal(stats.bytesServedFromCache, 0);
  assert.equal(stats.bytesFetchedFromNetwork, 1500);
});

// ---------------------------------------------------------------------------
// 5. Sequential read-ahead prefetching, random-seek suppression, and error safety
// ---------------------------------------------------------------------------

test('sequential reads trigger read-ahead prefetching, record prefetchHits, suppress prefetches on random seeks, and clamp at EOF', async () => {
  const chunkSize = 4096;
  const totalChunks = 6; // 24 KiB: chunks 0..5
  const rpfBytes = makePatternBytes(totalChunks * chunkSize, 83);
  const mock = createMockRangeFetch(new Map([['/game/movies.rpf', rpfBytes]]));

  const cache = new HttpfsLruChunkCache({
    fetchImpl: mock.fetchImpl,
    cacheMaxBytes: 64 * 1024,
    chunkSize,
    readAheadChunks: 2,
  });

  // Read chunk 0 (initial read -> not sequential yet, 0 prefetches).
  assert.deepEqual(
    await cache.read('/game/movies.rpf', 0, chunkSize),
    rpfBytes.subarray(0, chunkSize),
  );
  await cache.flushPrefetches();
  assert.equal(mock.requests.length, 1);
  assert.equal(cache.getStats().prefetchIssued, 0);

  // Read chunk 1 sequentially -> detects sequential streak, prefetches chunks 2 & 3!
  assert.deepEqual(
    await cache.read('/game/movies.rpf', chunkSize, chunkSize),
    rpfBytes.subarray(chunkSize, 2 * chunkSize),
  );
  await cache.flushPrefetches();
  assert.equal(cache.getStats().prefetchIssued, 2);
  assert.equal(mock.requests.length, 4); // chunks 0, 1, 2 (prefetch), 3 (prefetch)

  // Read chunk 2 -> served directly from prefetched cache! Schedules prefetch for chunk 4.
  const reqsBeforeChunk2 = mock.requests.length;
  assert.deepEqual(
    await cache.read('/game/movies.rpf', 2 * chunkSize, chunkSize),
    rpfBytes.subarray(2 * chunkSize, 3 * chunkSize),
  );
  await cache.flushPrefetches();
  assert.equal(cache.getStats().prefetchHits, 1);
  assert.equal(cache.getStats().prefetchIssued, 3); // chunk 4 prefetched
  assert.equal(mock.requests.length, reqsBeforeChunk2 + 1); // ONLY chunk 4 prefetch!

  // Read chunks 3, 4, 5 sequentially -> chunk 5 prefetched when reading chunk 3;
  // chunks 6+ MUST NOT be prefetched past EOF!
  for (let idx = 3; idx < totalChunks; idx += 1) {
    assert.deepEqual(
      await cache.read('/game/movies.rpf', idx * chunkSize, chunkSize),
      rpfBytes.subarray(idx * chunkSize, (idx + 1) * chunkSize),
    );
    await cache.flushPrefetches();
  }

  const stats = cache.getStats();
  assert.equal(stats.cacheMisses, 2); // ONLY chunks 0 and 1 were demand misses!
  assert.equal(stats.prefetchIssued, 4); // chunks 2, 3, 4, 5 (ZERO past-EOF prefetches!)
  assert.equal(stats.prefetchHits, 4); // ALL 4 prefetched chunks consumed!
  assert.equal(stats.cacheHits, 4);
  assert.equal(stats.bytesServedFromCache, 4 * chunkSize);
  assert.equal(stats.bytesFetchedFromNetwork, totalChunks * chunkSize);
  assert.equal(mock.requests.length, totalChunks);

  // Random seeks on a fresh file must NOT issue prefetches.
  const randBytes = makePatternBytes(16 * chunkSize, 91);
  const randMock = createMockRangeFetch(new Map([['/game/random.rpf', randBytes]]));
  const randCache = new HttpfsLruChunkCache({
    fetchImpl: randMock.fetchImpl,
    cacheMaxBytes: 64 * 1024,
    chunkSize,
    readAheadChunks: 2,
  });

  for (const idx of [0, 7, 2, 12, 4]) {
    const slice = await randCache.read('/game/random.rpf', idx * chunkSize, 1024);
    assert.deepEqual(slice, randBytes.subarray(idx * chunkSize, idx * chunkSize + 1024));
  }
  await randCache.flushPrefetches();
  assert.equal(randCache.getStats().prefetchIssued, 0);
  assert.equal(randMock.requests.length, 5);
});

test('sub-chunk sequential reads trigger read-ahead while reading chunk 0 and safely absorb prefetch errors', async () => {
  const chunkSize = 4096;
  const shaderBytes = makePatternBytes(4 * chunkSize, 109);
  let failChunk2Once = true;

  const mock = createMockRangeFetch(new Map([['/shaders/main.wgsl.pack', shaderBytes]]), {
    statusOverrides: new Map([
      [
        '/shaders/main.wgsl.pack',
        (req) => {
          if (failChunk2Once && req.rangeHeader === 'bytes=8192-12287') {
            failChunk2Once = false;
            return 503;
          }
          return null;
        },
      ],
    ]),
  });

  const cache = new HttpfsLruChunkCache({
    fetchImpl: mock.fetchImpl,
    cacheMaxBytes: 64 * 1024,
    chunkSize,
    readAheadChunks: 2,
  });

  // Read 4 x 1024B slices sequentially across chunk 0:
  //   - Slice 0 [0..1024): fetches chunk 0 (miss)
  //   - Slice 1 [1024..2048): cache hit + sequential detected -> prefetches chunk 1 (ok) & chunk 2 (503 swallowed!)
  for (let i = 0; i < 4; i += 1) {
    const got = await cache.read('/shaders/main.wgsl.pack', i * 1024, 1024);
    assert.deepEqual(got, shaderBytes.subarray(i * 1024, (i + 1) * 1024));
  }
  await cache.flushPrefetches();
  assert.equal(cache.getStats().prefetchIssued, 2);

  // Reading chunk 1 hits the prefetched chunk with zero new network fetches for chunk 1.
  const got1 = cache.tryReadSync('/shaders/main.wgsl.pack', chunkSize, 1024);
  assert.deepEqual(got1, shaderBytes.subarray(chunkSize, chunkSize + 1024));
  assert.equal(cache.getStats().prefetchHits, 1);

  // Reading chunk 2 (whose prefetch failed with 503) retries cleanly and succeeds!
  const got2 = await cache.read('/shaders/main.wgsl.pack', 2 * chunkSize, 1024);
  assert.deepEqual(got2, shaderBytes.subarray(2 * chunkSize, 2 * chunkSize + 1024));
});

// ---------------------------------------------------------------------------
// 6. Concurrent in-flight request coalescing (demand+demand and demand+prefetch)
// ---------------------------------------------------------------------------

test('HttpfsLruChunkCache coalesces duplicate in-flight demand requests and in-flight prefetch requests', async () => {
  const chunkSize = 4096;
  const rpfBytes = makePatternBytes(4 * chunkSize, 127);

  // Part A: 6 concurrent demand reads hitting uncached chunk 1 -> strictly 1 HTTP fetch!
  const gate1 = createDeferred();
  const mockA = createMockRangeFetch(new Map([['/game/coalesce.rpf', rpfBytes]]), {
    onRequest: async (req) => {
      if (req.rangeHeader === 'bytes=4096-8191') {
        await gate1.promise;
      }
    },
  });

  const cacheA = new HttpfsLruChunkCache({
    fetchImpl: mockA.fetchImpl,
    cacheMaxBytes: 64 * 1024,
    chunkSize,
    readAheadChunks: 0,
  });

  const offsets = [4096, 4200, 5000, 6144, 7000, 8000];
  const lens = [100, 300, 512, 1024, 256, 192];
  const pending = offsets.map((off, i) => cacheA.read('/game/coalesce.rpf', off, lens[i]));

  // All 6 reads share the exact same in-flight chunk fetch.
  assert.equal(mockA.requests.length, 1);
  assert.equal(cacheA.getStats().inFlightRequests, 1);
  assert.equal(cacheA.getStats().coalescedReads, 5);
  assert.equal(cacheA.getStats().cacheMisses, 1);

  gate1.resolve();
  const results = await Promise.all(pending);
  for (let i = 0; i < offsets.length; i += 1) {
    assert.deepEqual(results[i], rpfBytes.subarray(offsets[i], offsets[i] + lens[i]));
  }
  assert.equal(mockA.requests.length, 1);
  assert.equal(cacheA.getStats().inFlightRequests, 0);

  // Part B: Demand read arriving while a background prefetch for chunk 2 is STILL in-flight!
  const prefetchGate = createDeferred();
  const mockB = createMockRangeFetch(new Map([['/game/prefetch_join.rpf', rpfBytes]]), {
    onRequest: async (req) => {
      if (req.rangeHeader === 'bytes=8192-12287') {
        await prefetchGate.promise;
      }
    },
  });

  const cacheB = new HttpfsLruChunkCache({
    fetchImpl: mockB.fetchImpl,
    cacheMaxBytes: 64 * 1024,
    chunkSize,
    readAheadChunks: 1,
  });

  // Read chunk 0, then chunk 1 -> triggers in-flight prefetch of chunk 2 (held at prefetchGate).
  await cacheB.read('/game/prefetch_join.rpf', 0, chunkSize);
  await cacheB.read('/game/prefetch_join.rpf', chunkSize, chunkSize);
  assert.equal(cacheB.getStats().prefetchIssued, 1);
  assert.equal(mockB.requests.length, 3); // chunk 0, chunk 1, chunk 2 (in-flight prefetch)

  // Demand read for chunk 2 arrives BEFORE prefetch completes -> coalesces onto in-flight prefetch!
  const demandChunk2 = cacheB.read('/game/prefetch_join.rpf', 2 * chunkSize + 128, 512);
  assert.equal(
    mockB.requests.filter((r) => r.rangeHeader === 'bytes=8192-12287').length,
    1,
    'Must NOT issue duplicate HTTP fetch for in-flight chunk 2 prefetch',
  );
  // Sequential demand read for chunk 2 proactively prefetches chunk 3 while coalescing on chunk 2.
  assert.equal(mockB.requests.length, 4);
  assert.equal(mockB.requests[3].rangeHeader, 'bytes=12288-16383');
  assert.equal(cacheB.getStats().coalescedReads, 1);
  assert.equal(cacheB.getStats().prefetchHits, 1);
  assert.equal(cacheB.getStats().prefetchIssued, 2);

  prefetchGate.resolve();
  const chunk2Slice = await demandChunk2;
  assert.deepEqual(
    chunk2Slice,
    rpfBytes.subarray(2 * chunkSize + 128, 2 * chunkSize + 128 + 512),
  );
  await cacheB.flushPrefetches();

  // Subsequent cache hit on chunk 2 must NOT double-count prefetchHits!
  const chunk2Repeat = cacheB.tryReadSync('/game/prefetch_join.rpf', 2 * chunkSize + 64, 64);
  assert.deepEqual(
    chunk2Repeat,
    rpfBytes.subarray(2 * chunkSize + 64, 2 * chunkSize + 128),
  );
  assert.equal(cacheB.getStats().prefetchHits, 1);
  assert.equal(cacheB.getStats().cacheHits, 1);
});

// ---------------------------------------------------------------------------
// 7. Range boundaries, multi-chunk spans, EOF clamping, truncated 206, 200 OK, and empty files
// ---------------------------------------------------------------------------

test('handles cross-chunk spans, EOF clamping, 416 past-EOF, truncated 206, 200 OK fallback, and empty files', async () => {
  const chunkSize = 4096;
  const partialSize = 2 * chunkSize + 1337; // 9529 bytes (2 full chunks + 1337B tail chunk)
  const rpfBytes = makePatternBytes(partialSize, 151);
  const emptyBytes = new Uint8Array(0);

  const mock = createMockRangeFetch(
    new Map([
      ['/game/partial.rpf', rpfBytes],
      ['/game/empty.rpf', emptyBytes],
    ]),
  );

  const cache = new HttpfsLruChunkCache({
    fetchImpl: mock.fetchImpl,
    cacheMaxBytes: 64 * 1024,
    chunkSize,
    readAheadChunks: 0,
  });

  // Zero-length read performs 0 fetches.
  assert.deepEqual(await cache.read('/game/partial.rpf', 0, 0), new Uint8Array(0));
  assert.equal(mock.requests.length, 0);

  // Spanning read across chunks 0, 1, and partial tail chunk 2 past EOF -> clamped to exact EOF!
  const spanOff = chunkSize - 200;
  const spanGot = await cache.read('/game/partial.rpf', spanOff, 20000);
  assert.equal(spanGot.byteLength, partialSize - spanOff);
  assert.deepEqual(spanGot, rpfBytes.subarray(spanOff, partialSize));
  assert.equal(mock.requests.length, 3); // Strictly chunks 0, 1, 2 (ZERO past-EOF chunk 3/4 fetches!)
  assert.equal(cache.getStats().cacheBytes, partialSize);

  // Synchronous hit on tail chunk clamped past EOF.
  const tailSync = cache.tryReadSync('/game/partial.rpf', partialSize - 100, 500);
  assert.equal(tailSync.byteLength, 100);
  assert.deepEqual(tailSync, rpfBytes.subarray(partialSize - 100, partialSize));
  assert.equal(mock.requests.length, 3);

  // Strictly past EOF returns 0 bytes with ZERO network calls once size is known.
  assert.deepEqual(await cache.read('/game/partial.rpf', partialSize, 1024), new Uint8Array(0));
  assert.deepEqual(await cache.read('/game/partial.rpf', partialSize + 9999, 1024), new Uint8Array(0));
  assert.equal(mock.requests.length, 3);

  // Unknown-size past-EOF triggers single 416, learns total size, and subsequent past-EOF does 0 fetches.
  const unknownMock = createMockRangeFetch(new Map([['/game/oob.rpf', rpfBytes]]));
  const unknownCache = new HttpfsLruChunkCache({
    fetchImpl: unknownMock.fetchImpl,
    cacheMaxBytes: 64 * 1024,
    chunkSize,
    readAheadChunks: 0,
  });
  assert.deepEqual(await unknownCache.read('/game/oob.rpf', 50000, 1024), new Uint8Array(0));
  assert.equal(unknownMock.requests.length, 1);
  assert.equal(unknownCache.getKnownSize('/game/oob.rpf'), partialSize);
  assert.deepEqual(await unknownCache.read('/game/oob.rpf', 60000, 1024), new Uint8Array(0));
  assert.equal(unknownMock.requests.length, 1);

  // Truncated 206 server (serves max 1024B per response for a 4096B chunk).
  const truncMock = createMockRangeFetch(new Map([['/game/trunc.rpf', rpfBytes]]), {
    maxServeBytes: 1024,
  });
  const truncCache = new HttpfsLruChunkCache({
    fetchImpl: truncMock.fetchImpl,
    cacheMaxBytes: 64 * 1024,
    chunkSize,
    readAheadChunks: 0,
  });
  const truncSlice = await truncCache.read('/game/trunc.rpf', 500, 2000);
  assert.deepEqual(truncSlice, rpfBytes.subarray(500, 2500));
  assert.equal(truncMock.requests.length, 4); // 4 x 1024B to complete chunk 0
  const truncCached = truncCache.tryReadSync('/game/trunc.rpf', 3000, 1000);
  assert.deepEqual(truncCached, rpfBytes.subarray(3000, 4000));
  assert.equal(truncMock.requests.length, 4); // Full chunk 0 cached!

  // 200 OK full-body server fallback.
  const full200Mock = createMockRangeFetch(new Map([['/shaders/fallback200.pack', rpfBytes]]), {
    force200: true,
  });
  const full200Cache = new HttpfsLruChunkCache({
    fetchImpl: full200Mock.fetchImpl,
    cacheMaxBytes: 64 * 1024,
    chunkSize,
    readAheadChunks: 0,
  });
  const fullSlice = await full200Cache.read('/shaders/fallback200.pack', chunkSize + 100, 500);
  assert.deepEqual(fullSlice, rpfBytes.subarray(chunkSize + 100, chunkSize + 600));
  assert.equal(full200Cache.getKnownSize('/shaders/fallback200.pack'), partialSize);

  // Empty 0-byte file probes cleanly via 416 bytes */0 and returns 0 bytes on read.
  const probeInfo = await cache.probe('/game/empty.rpf');
  assert.equal(probeInfo.size, 0);
  assert.deepEqual(await cache.read('/game/empty.rpf', 0, 128), new Uint8Array(0));
});

// ---------------------------------------------------------------------------
// 8. End-to-end SAB HttpfsMailboxHost / Client + HttpfsMailboxPoolHost / Client
// ---------------------------------------------------------------------------

test('HttpfsMailboxHost preserves SAB protocol/error codes while accelerating /game/*.rpf & /shaders/* with LRU cache + read-ahead', async () => {
  const chunkSize = 16 * 1024; // 16 KiB chunks (4 x 4 KiB SAB frames per chunk)
  const rpfSize = 5 * chunkSize + 1234; // 83154 bytes
  const shaderSize = 2 * chunkSize + 512; // 33280 bytes
  const rpfBytes = makePatternBytes(rpfSize, 199);
  const shaderBytes = makePatternBytes(shaderSize, 211);

  const mock = createMockRangeFetch(
    new Map([
      ['/game/shaders.rpf', rpfBytes],
      ['/shaders/splash.wgsl.pack', shaderBytes],
    ]),
    {
      statusOverrides: new Map([['/game/broken.rpf', 500]]),
    },
  );

  // Use 4 KiB SAB payload so each 16 KiB chunk serves 4 sequential SAB reads.
  const sab = createHttpfsMailboxBuffer(4096);
  const host = new HttpfsMailboxHost(sab, {
    fetchImpl: mock.fetchImpl,
    cacheMaxBytes: 128 * 1024,
    chunkSize,
    readAheadChunks: 2,
    probeOnOpen: true,
  });
  const client = new HttpfsMailboxClient(sab);

  // Verify exact SAB header layout constants remain intact.
  assert.equal(HttpfsOffsets.STATUS, 0);
  assert.equal(HttpfsOffsets.OP, 4);
  assert.equal(HttpfsOffsets.HANDLE, 8);
  assert.equal(HttpfsOffsets.PAYLOAD_CAPACITY, 12);
  assert.equal(HttpfsOffsets.OFFSET, 16);
  assert.equal(HttpfsOffsets.LENGTH, 24);
  assert.equal(HttpfsOffsets.RESULT, 32);
  assert.equal(HttpfsOffsets.PAYLOAD, 40);
  assert.equal(HttpfsStatus.Idle, 0);
  assert.equal(HttpfsOp.Read, 2);

  host.start();
  try {
    // Error codes preserved: missing file (-1 on client), traversal (-1), HTTP 500 open (-1).
    assert.equal(await client.openAsync('game/does_not_exist.rpf'), -1);
    assert.equal(await client.openAsync('../secret.rpf'), -1);
    assert.equal(await client.openAsync('game/broken.rpf'), -1);

    // Open /game/shaders.rpf and stream entire file sequentially in 4 KiB SAB frames.
    const hRpf = await client.openAsync('game/shaders.rpf');
    assert.ok(hRpf >= 1);

    const outRpf = new Uint8Array(rpfSize);
    const frameBuf = new Uint8Array(4096);
    let offset = 0;
    while (offset < rpfSize) {
      const got = await client.readAsync(hRpf, BigInt(offset), frameBuf, 4096);
      assert.ok(got > 0, `Expected positive bytes at offset ${offset}`);
      outRpf.set(frameBuf.subarray(0, got), offset);
      offset += got;
      await host.flushPrefetches();
    }
    assert.deepEqual(outRpf, rpfBytes);

    // Past-EOF read returns 0 cleanly.
    assert.equal(await client.readAsync(hRpf, BigInt(rpfSize), frameBuf, 4096), 0);

    // Repeat full read of /game/shaders.rpf MUST hit cache with ZERO additional HTTP requests!
    const reqsBeforeRepeat = mock.requests.length;
    const syncHitsBeforeRepeat = host.getStats().syncFastPathHits;
    const repeatRpf = new Uint8Array(rpfSize);
    assert.equal(await client.readAsync(hRpf, 0n, repeatRpf, rpfSize), rpfSize);
    assert.deepEqual(repeatRpf, rpfBytes);
    assert.equal(mock.requests.length, reqsBeforeRepeat);
    assert.ok(
      host.getStats().syncFastPathHits > syncHitsBeforeRepeat,
      'Cached SAB reads must take the synchronous fast path',
    );

    // Open and stream /shaders/splash.wgsl.pack as well.
    const hShader = await client.openAsync('shaders/splash.wgsl.pack');
    assert.ok(hShader >= 1);
    const outShader = new Uint8Array(shaderSize);
    assert.equal(await client.readAsync(hShader, 0n, outShader, shaderSize), shaderSize);
    assert.deepEqual(outShader, shaderBytes);

    // Close handles; post-close read returns -9 (EBADF).
    assert.equal(await client.closeAsync(hRpf), true);
    assert.equal(await client.closeAsync(hRpf), false);
    assert.equal(await client.readAsync(hRpf, 0n, frameBuf, 64), -9);
    assert.equal(await client.closeAsync(hShader), true);

    // Inspect all 9 sanitized telemetry counters on host.getStats().
    const stats = host.getStats();
    assert.ok(stats.cacheHits > 20, `Expected >20 cacheHits, got ${stats.cacheHits}`);
    assert.ok(stats.cacheMisses >= 2, `Expected >=2 cacheMisses, got ${stats.cacheMisses}`);
    assert.equal(stats.cacheBytes, rpfSize + shaderSize);
    assert.equal(stats.cacheEvictions, 0);
    assert.ok(stats.prefetchIssued >= 4, `Expected >=4 prefetchIssued, got ${stats.prefetchIssued}`);
    assert.ok(stats.prefetchHits >= 4, `Expected >=4 prefetchHits, got ${stats.prefetchHits}`);
    assert.equal(typeof stats.coalescedReads, 'number');
    assert.ok(stats.bytesServedFromCache >= rpfSize);
    assert.ok(stats.bytesFetchedFromNetwork >= rpfSize + shaderSize);

    // Zero file paths or URLs leaked in sanitized telemetry JSON!
    const serialized = JSON.stringify(stats);
    assert.equal(serialized.includes('shaders.rpf'), false);
    assert.equal(serialized.includes('splash.wgsl.pack'), false);
    assert.equal(serialized.includes('/game/'), false);
    assert.equal(serialized.includes('/shaders/'), false);
  } finally {
    await host.stop();
  }
});

test('HttpfsMailboxPoolHost shares bounded LRU cache across slots and coalesces concurrent multi-slot chunk reads', async () => {
  const chunkSize = 16 * 1024;
  const rpfBytes = makePatternBytes(4 * chunkSize, 229);
  const gate = createDeferred();

  const mock = createMockRangeFetch(new Map([['/game/pool_shared.rpf', rpfBytes]]), {
    onRequest: async (req) => {
      if (req.rangeHeader === 'bytes=16384-32767') {
        await gate.promise;
      }
    },
  });

  const slots = createHttpfsMailboxPool(4, 4096);
  const poolHost = new HttpfsMailboxPoolHost(slots, {
    fetchImpl: mock.fetchImpl,
    cacheMaxBytes: 64 * 1024,
    chunkSize,
    readAheadChunks: 0,
    probeOnOpen: false,
  });
  const poolClient = new HttpfsMailboxPoolClient(slots);

  poolHost.start();
  try {
    const h = await poolClient.openAsync('game/pool_shared.rpf');
    assert.ok(h >= 1);

    // Issue 4 concurrent reads across all 4 SAB slots targeting different offsets inside chunk 1.
    const batchPromise = poolClient.readBatchAsync([
      { handle: h, offset: 16384n, len: 4096 },
      { handle: h, offset: 20480n, len: 4096 },
      { handle: h, offset: 24576n, len: 4096 },
      { handle: h, offset: 28672n, len: 4096 },
    ]);

    // Wait a tick for all 4 SAB slots to dispatch into the shared chunk cache.
    await new Promise((r) => setTimeout(r, 25));
    assert.equal(
      mock.requests.length,
      1,
      `All 4 concurrent SAB slots hitting chunk 1 must coalesce into 1 HTTP fetch, got ${mock.requests.length}`,
    );

    gate.resolve();
    const batch = await batchPromise;
    for (let i = 0; i < 4; i += 1) {
      assert.equal(batch[i].bytesRead, 4096);
      assert.deepEqual(
        batch[i].dst,
        rpfBytes.subarray(16384 + i * 4096, 16384 + (i + 1) * 4096),
      );
    }

    const stats = poolHost.getStats();
    assert.equal(stats.slotCount, 4);
    assert.equal(stats.cacheMisses, 1);
    assert.equal(stats.coalescedReads, 3);
    assert.equal(stats.cacheBytes, chunkSize);
    assert.equal(stats.bytesFetchedFromNetwork, chunkSize);

    assert.equal(await poolClient.closeAsync(h), true);
  } finally {
    await poolHost.stop();
  }
});

test('cross-thread synchronous Atomics.wait reads via HttpfsMailboxHost LRU cache and sequential prefetch', async () => {
  const chunkSize = 8192;
  const rpfSize = 4 * chunkSize + 777;
  const rpfBytes = makePatternBytes(rpfSize, 241);
  const mock = createMockRangeFetch(new Map([['/game/cross_lru.rpf', rpfBytes]]));

  const sab = createHttpfsMailboxBuffer(4096);
  const host = new HttpfsMailboxHost(sab, {
    fetchImpl: mock.fetchImpl,
    cacheMaxBytes: 64 * 1024,
    chunkSize,
    readAheadChunks: 2,
    probeOnOpen: true,
  });
  host.start();

  const workerCode = `
    const { workerData, parentPort } = require('node:worker_threads');
    const { HttpfsMailboxClient } = require(workerData.mailboxPath);
    const client = new HttpfsMailboxClient(workerData.sab, { timeoutMs: 8000 });

    const h = client.open('game/cross_lru.rpf');
    const first = new Uint8Array(workerData.size);
    const n1 = client.read(h, 0n, first, workerData.size);
    const second = new Uint8Array(workerData.size);
    const n2 = client.read(h, 0n, second, workerData.size);
    const eof = client.read(h, BigInt(workerData.size), new Uint8Array(64), 64);
    const closed = client.close(h);
    parentPort.postMessage({
      h,
      n1,
      n2,
      eof,
      closed,
      first: Array.from(first),
      second: Array.from(second),
    });
  `;

  try {
    const res = await new Promise((resolve, reject) => {
      const w = new NodeWorker(workerCode, {
        eval: true,
        workerData: {
          sab,
          size: rpfSize,
          mailboxPath: path.join(repoRoot, 'web', 'workers', 'httpfs_mailbox.js'),
        },
      });
      w.once('message', resolve);
      w.once('error', reject);
      w.once('exit', (code) => {
        if (code !== 0) reject(new Error(`Worker exited with code ${code}`));
      });
    });

    await host.flushPrefetches();

    assert.ok(res.h >= 1);
    assert.equal(res.n1, rpfSize);
    assert.equal(res.n2, rpfSize);
    assert.equal(res.eof, 0);
    assert.equal(res.closed, true);
    assert.deepEqual(new Uint8Array(res.first), rpfBytes);
    assert.deepEqual(new Uint8Array(res.second), rpfBytes);

    // 1 probe + 5 chunks = 6 HTTP fetches total for TWO full reads of the 5-chunk file!
    assert.equal(mock.requests.length, 6);
    const stats = host.getStats();
    assert.equal(stats.cacheBytes, rpfSize);
    assert.ok(stats.cacheHits > 0);
    assert.ok(stats.prefetchIssued > 0);
    assert.ok(stats.bytesServedFromCache >= rpfSize);
  } finally {
    await host.stop();
  }
});
