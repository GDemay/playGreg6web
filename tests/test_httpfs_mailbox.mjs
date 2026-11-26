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

// ---------------------------------------------------------------------------
// 9. W294: 8-lane pool peak-concurrency telemetry, fairness, cross-lane coalescing & resetStats
// ---------------------------------------------------------------------------

async function waitForCondition(predicate, timeoutMs = 3000, label = 'condition') {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() >= deadline) {
      throw new Error(`Timed out waiting for ${label}`);
    }
    await new Promise((r) => setTimeout(r, 2));
  }
}

test('W294: 8-lane HttpfsMailboxPoolHost tracks peakActiveLanes >= 3, requestsByLane, totalRequests, and prevents head-of-line blocking', async () => {
  const chunkSize = 8192;
  const dsBytes = makePatternBytes(2 * chunkSize, 11);
  const binkBytes = makePatternBytes(2 * chunkSize, 22);
  const readFileBytes = makePatternBytes(2 * chunkSize, 33);
  const lane3Bytes = makePatternBytes(2 * chunkSize, 44);
  const lane4Bytes = makePatternBytes(2 * chunkSize, 55);
  const lane5Bytes = makePatternBytes(2 * chunkSize, 66);

  const dsGate = createDeferred();
  const gate3 = createDeferred();
  const gate4 = createDeferred();
  const gate5 = createDeferred();
  const gatesByUrl = new Map([
    ['/game/directstorage.rpf', dsGate],
    ['/game/stream3.rpf', gate3],
    ['/game/stream4.rpf', gate4],
    ['/game/stream5.rpf', gate5],
  ]);
  const arrivedGatedUrls = new Set();

  const mock = createMockRangeFetch(
    new Map([
      ['/game/directstorage.rpf', dsBytes],
      ['/game/bink2_intro.rpf', binkBytes],
      ['/game/readfile_cfg.rpf', readFileBytes],
      ['/game/stream3.rpf', lane3Bytes],
      ['/game/stream4.rpf', lane4Bytes],
      ['/game/stream5.rpf', lane5Bytes],
    ]),
    {
      onRequest: async (req) => {
        const gate = gatesByUrl.get(req.url);
        if (gate) {
          arrivedGatedUrls.add(req.url);
          await gate.promise;
        }
      },
    },
  );

  const slots = createHttpfsMailboxPool(8, 4096);
  const poolHost = new HttpfsMailboxPoolHost(slots, {
    fetchImpl: mock.fetchImpl,
    cacheMaxBytes: 256 * 1024,
    chunkSize,
    readAheadChunks: 0,
    probeOnOpen: false,
  });
  const clients = slots.map((sab) => new HttpfsMailboxClient(sab));

  poolHost.start();
  try {
    const [hDs, hBink, hRf, h3, h4, h5] = await Promise.all([
      clients[0].openAsync('game/directstorage.rpf'),
      clients[1].openAsync('game/bink2_intro.rpf'),
      clients[2].openAsync('game/readfile_cfg.rpf'),
      clients[3].openAsync('game/stream3.rpf'),
      clients[4].openAsync('game/stream4.rpf'),
      clients[5].openAsync('game/stream5.rpf'),
    ]);
    assert.ok(hDs >= 1 && hBink >= 1 && hRf >= 1 && h3 >= 1 && h4 >= 1 && h5 >= 1);

    // 1. Stall lane 0 (DirectStorage) on a slow HTTP Range fetch.
    const dsOut = new Uint8Array(4096);
    const dsReadPromise = clients[0].readAsync(hDs, 0n, dsOut, 4096);
    await waitForCondition(
      () => arrivedGatedUrls.has('/game/directstorage.rpf'),
      3000,
      'DirectStorage lane 0 in-flight',
    );
    assert.equal(poolHost.getStats().activeLanes, 1);

    // 2. While lane 0 is blocked, lane 1 (Bink2) and lane 2 (ReadFile) must complete with zero head-of-line stall.
    const binkOut1 = new Uint8Array(4096);
    const binkOut2 = new Uint8Array(4096);
    const rfOut1 = new Uint8Array(4096);
    const [nBink1, nRf1] = await Promise.all([
      clients[1].readAsync(hBink, 0n, binkOut1, 4096),
      clients[2].readAsync(hRf, 0n, rfOut1, 4096),
    ]);
    const nBink2 = await clients[1].readAsync(hBink, 4096n, binkOut2, 4096);
    assert.equal(nBink1, 4096);
    assert.equal(nBink2, 4096);
    assert.equal(nRf1, 4096);
    assert.deepEqual(binkOut1, binkBytes.subarray(0, 4096));
    assert.deepEqual(binkOut2, binkBytes.subarray(4096, 8192));
    assert.deepEqual(rfOut1, readFileBytes.subarray(0, 4096));
    assert.equal(poolHost.getStats().activeLanes, 1, 'Lane 0 must still be the sole active lane');

    // 3. Simultaneously drive lanes 3, 4, 5 alongside blocked lane 0 -> 4 active lanes in-flight (>= 3).
    const out3 = new Uint8Array(4096);
    const out4 = new Uint8Array(4096);
    const out5 = new Uint8Array(4096);
    const p3 = clients[3].readAsync(h3, 0n, out3, 4096);
    const p4 = clients[4].readAsync(h4, 0n, out4, 4096);
    const p5 = clients[5].readAsync(h5, 0n, out5, 4096);

    await waitForCondition(
      () => arrivedGatedUrls.size === 4,
      3000,
      '4 concurrent gated lanes in-flight',
    );

    const midStats = poolHost.getStats();
    assert.equal(midStats.slotCount, 8);
    assert.equal(midStats.activeLanes, 4);
    assert.ok(
      midStats.peakActiveLanes >= 3,
      `Expected peakActiveLanes >= 3, got ${midStats.peakActiveLanes}`,
    );
    assert.equal(midStats.peakActiveLanes, 4);

    // Release all 4 gated lanes out of order.
    gate5.resolve();
    gate3.resolve();
    gate4.resolve();
    dsGate.resolve();

    const [nDs, n3, n4, n5] = await Promise.all([dsReadPromise, p3, p4, p5]);
    assert.deepEqual([nDs, n3, n4, n5], [4096, 4096, 4096, 4096]);
    assert.deepEqual(dsOut, dsBytes.subarray(0, 4096));
    assert.deepEqual(out3, lane3Bytes.subarray(0, 4096));
    assert.deepEqual(out4, lane4Bytes.subarray(0, 4096));
    assert.deepEqual(out5, lane5Bytes.subarray(0, 4096));

    const stats = poolHost.getStats();
    assert.equal(stats.slotCount, 8);
    assert.equal(stats.activeLanes, 0);
    assert.equal(stats.peakActiveLanes, 4);
    // Requests per lane: lane 0 (open+read=2), lane 1 (open+2reads=3), lane 2 (open+read=2),
    // lanes 3..5 (open+read=2 each), lanes 6..7 (unused=0).
    assert.deepEqual(stats.requestsByLane, [2, 3, 2, 2, 2, 2, 0, 0]);
    assert.equal(
      stats.totalRequests,
      stats.requestsByLane.reduce((acc, v) => acc + v, 0),
    );
    assert.equal(stats.totalRequests, 13);

    // Verify zero path/URL leaks in sanitized telemetry JSON.
    const serialized = JSON.stringify(stats);
    assert.equal(serialized.includes('directstorage.rpf'), false);
    assert.equal(serialized.includes('bink2_intro.rpf'), false);
    assert.equal(serialized.includes('/game/'), false);
  } finally {
    await poolHost.stop();
  }
});

test('W294: 8-lane HttpfsMailboxPoolHost round-robin polling distributes requestsByLane fairly and skips in-flight lanes without stalling', async () => {
  const chunkSize = 4096;
  const rpfBytes = makePatternBytes(4 * chunkSize, 77);
  const slowGate = createDeferred();
  let gateSlowLane0 = false;

  const mock = createMockRangeFetch(new Map([['/game/fair_rr.rpf', rpfBytes]]), {
    onRequest: async (req) => {
      if (gateSlowLane0 && req.rangeHeader === 'bytes=4096-8191') {
        await slowGate.promise;
      }
    },
  });

  const slots = createHttpfsMailboxPool(8, 4096);
  const poolHost = new HttpfsMailboxPoolHost(slots, {
    fetchImpl: mock.fetchImpl,
    cacheMaxBytes: 64 * 1024,
    chunkSize,
    readAheadChunks: 0,
    probeOnOpen: false,
  });
  const clients = slots.map((sab) => new HttpfsMailboxClient(sab));

  // Warm chunk 0 and open handles on all 8 lanes using manual round-robin processOnce().
  const openPromises = clients.map((c) => c.openAsync('game/fair_rr.rpf'));
  const openDispatched = await poolHost.processOnce();
  assert.deepEqual(openDispatched, [true, true, true, true, true, true, true, true]);
  const handles = await Promise.all(openPromises);
  assert.ok(handles.every((h) => h >= 1));

  // Warm chunk 0 on lane 0 so subsequent chunk-0 reads complete synchronously.
  const warmBuf = new Uint8Array(512);
  const warmPromise = clients[0].readAsync(handles[0], 0n, warmBuf, 512);
  await poolHost.processOnce();
  assert.equal(await warmPromise, 512);

  // Reset telemetry before fairness measurement.
  poolHost.resetStats();
  assert.deepEqual(poolHost.getStats().requestsByLane, [0, 0, 0, 0, 0, 0, 0, 0]);
  assert.equal(poolHost.getStats().totalRequests, 0);
  assert.equal(poolHost.getStats().peakActiveLanes, 0);

  // Queue simultaneous reads on all 8 lanes, then step with pollOnce(1) to verify strict round-robin fairness.
  const round1Bufs = clients.map(() => new Uint8Array(256));
  const round1Promises = clients.map((c, i) =>
    c.readAsync(handles[i], BigInt(i * 256), round1Bufs[i], 256),
  );
  for (let step = 0; step < 8; step += 1) {
    const dispatched = poolHost.pollOnce(1);
    assert.equal(dispatched, 1);
    const snap = poolHost.getStats();
    assert.equal(snap.requestsByLane[step], 1, `Step ${step} must fairly serve lane ${step}`);
    for (let later = step + 1; later < 8; later += 1) {
      assert.equal(snap.requestsByLane[later], 0);
    }
  }
  assert.equal(poolHost.pollOnce(), 0, 'Idle pollOnce must return 0 with zero pending work');
  const round1Lens = await Promise.all(round1Promises);
  assert.deepEqual(round1Lens, [256, 256, 256, 256, 256, 256, 256, 256]);
  for (let i = 0; i < 8; i += 1) {
    assert.deepEqual(round1Bufs[i], rpfBytes.subarray(i * 256, (i + 1) * 256));
  }
  assert.deepEqual(poolHost.getStats().requestsByLane, [1, 1, 1, 1, 1, 1, 1, 1]);
  assert.equal(poolHost.getStats().totalRequests, 8);

  // Non-blocking pollOnce: stall lane 0 on uncached chunk 1 while lanes 1..7 read cached chunk 0.
  gateSlowLane0 = true;
  const slowOut = new Uint8Array(512);
  const slowPromise = clients[0].readAsync(handles[0], 4096n, slowOut, 512);
  assert.equal(poolHost.pollOnce(1), 1);
  assert.equal(poolHost.getStats().activeLanes, 1);

  // Calling pollOnce() again while lane 0 is in-flight must NOT re-dispatch lane 0 or stall siblings!
  const fastBufs = clients.slice(1).map(() => new Uint8Array(256));
  const fastPromises = clients.slice(1).map((c, idx) =>
    c.readAsync(handles[idx + 1], BigInt(idx * 256), fastBufs[idx], 256),
  );
  assert.equal(poolHost.pollOnce(), 7, 'Must dispatch all 7 sibling lanes while skipping in-flight lane 0');
  const fastLens = await Promise.all(fastPromises);
  assert.deepEqual(fastLens, [256, 256, 256, 256, 256, 256, 256]);
  assert.equal(poolHost.getStats().activeLanes, 1, 'Slow lane 0 still cleanly tracked in-flight');
  assert.equal(poolHost.getStats().peakActiveLanes, 2);

  slowGate.resolve();
  assert.equal(await slowPromise, 512);
  assert.deepEqual(slowOut, rpfBytes.subarray(4096, 4096 + 512));
  assert.equal(poolHost.getStats().activeLanes, 0);
  assert.deepEqual(poolHost.getStats().requestsByLane, [2, 2, 2, 2, 2, 2, 2, 2]);
  assert.equal(poolHost.getStats().totalRequests, 16);
});

test('W294: cross-lane in-flight Range coalescing triggers strictly 1 network fetch, reports crossLaneCoalesced >= 1, and resets cleanly via resetStats()', async () => {
  const chunkSize = 16 * 1024;
  const rpfBytes = makePatternBytes(4 * chunkSize, 137);
  const chunk1Gate = createDeferred();
  let chunk1FetchStarted = false;

  const mock = createMockRangeFetch(new Map([['/game/world_shared.rpf', rpfBytes]]), {
    onRequest: async (req) => {
      if (req.rangeHeader === 'bytes=16384-32767') {
        chunk1FetchStarted = true;
        await chunk1Gate.promise;
      }
    },
  });

  const slots = createHttpfsMailboxPool(8, 4096);
  const poolHost = new HttpfsMailboxPoolHost(slots, {
    fetchImpl: mock.fetchImpl,
    cacheMaxBytes: 128 * 1024,
    chunkSize,
    readAheadChunks: 0,
    probeOnOpen: false,
  });
  const clients = slots.map((sab) => new HttpfsMailboxClient(sab));

  poolHost.start();
  try {
    const handles = await Promise.all(clients.map((c) => c.openAsync('game/world_shared.rpf')));
    assert.ok(handles.every((h) => h >= 1));

    // Issue 6 concurrent overlapping sub-range reads across lanes 0..5 targeting the SAME chunk 1 [16384..32768).
    const readSpecs = [
      { lane: 0, off: 16384, len: 4096 },
      { lane: 1, off: 17408, len: 3072 },
      { lane: 2, off: 18432, len: 4096 },
      { lane: 3, off: 20480, len: 4096 },
      { lane: 4, off: 24576, len: 4096 },
      { lane: 5, off: 28672, len: 4096 },
    ];
    const bufs = readSpecs.map((s) => new Uint8Array(s.len));
    const readPromises = readSpecs.map((s, idx) =>
      clients[s.lane].readAsync(handles[s.lane], BigInt(s.off), bufs[idx], s.len),
    );

    await waitForCondition(
      () => chunk1FetchStarted && poolHost.getStats().activeLanes === 6,
      3000,
      'all 6 lanes attached to in-flight chunk 1 fetch',
    );

    const midStats = poolHost.getStats();
    assert.equal(mock.requests.length, 1, 'Concurrent multi-lane reads for chunk 1 must trigger strictly 1 network fetch');
    assert.equal(midStats.activeLanes, 6);
    assert.ok(midStats.peakActiveLanes >= 6);
    assert.equal(midStats.crossLaneCoalesced, 5);
    assert.equal(midStats.coalescedReads, 5);
    assert.equal(poolHost.chunkCache.getStats().crossLaneCoalesced, 5);

    chunk1Gate.resolve();
    const readLens = await Promise.all(readPromises);
    for (let i = 0; i < readSpecs.length; i += 1) {
      assert.equal(readLens[i], readSpecs[i].len);
      assert.deepEqual(
        bufs[i],
        rpfBytes.subarray(readSpecs[i].off, readSpecs[i].off + readSpecs[i].len),
      );
    }

    // Strictly 1 HTTP fetch total across all 6 concurrent lanes!
    assert.equal(mock.requests.length, 1);

    const postStats = poolHost.getStats();
    assert.equal(postStats.activeLanes, 0);
    assert.ok(postStats.peakActiveLanes >= 6);
    assert.ok(postStats.crossLaneCoalesced >= 1);
    assert.equal(postStats.crossLaneCoalesced, 5);
    assert.equal(postStats.cacheMisses, 1);
    assert.equal(postStats.cacheBytes, chunkSize);
    assert.deepEqual(postStats.requestsByLane, [2, 2, 2, 2, 2, 2, 1, 1]);
    assert.equal(postStats.totalRequests, 14);

    // resetStats() clears all cumulative pool & cache counters while preserving cached chunks and open handles.
    poolHost.resetStats();
    const resetSnap = poolHost.getStats();
    assert.equal(resetSnap.totalRequests, 0);
    assert.equal(resetSnap.requests, 0);
    assert.deepEqual(resetSnap.requestsByLane, [0, 0, 0, 0, 0, 0, 0, 0]);
    assert.equal(resetSnap.activeLanes, 0);
    assert.equal(resetSnap.peakActiveLanes, 0);
    assert.equal(resetSnap.crossLaneCoalesced, 0);
    assert.equal(resetSnap.coalescedReads, 0);
    assert.equal(resetSnap.cacheHits, 0);
    assert.equal(resetSnap.cacheMisses, 0);
    assert.equal(resetSnap.cacheBytes, chunkSize, 'Cached chunk must remain resident after resetStats()');
    assert.ok(resetSnap.perSlot.every((s) => s.requests === 0));

    // Subsequent read on lane 7 still serves cleanly from the resident cached chunk (still 1 network fetch total).
    const lane7Buf = new Uint8Array(1024);
    assert.equal(await clients[7].readAsync(handles[7], 20000n, lane7Buf, 1024), 1024);
    assert.deepEqual(lane7Buf, rpfBytes.subarray(20000, 21024));
    assert.equal(mock.requests.length, 1);
    const afterCachedRead = poolHost.getStats();
    assert.equal(afterCachedRead.totalRequests, 1);
    assert.deepEqual(afterCachedRead.requestsByLane, [0, 0, 0, 0, 0, 0, 0, 1]);
    assert.equal(afterCachedRead.cacheHits, 1);
    assert.equal(afterCachedRead.peakActiveLanes, 1);
  } finally {
    await poolHost.stop();
  }

  // Direct HttpfsLruChunkCache check: concurrent multi-lane reads coalesce to 1 fetch and resetStats() clears counters.
  const directGate = createDeferred();
  const directMock = createMockRangeFetch(new Map([['/game/direct_coalesce.rpf', rpfBytes]]), {
    onRequest: async () => {
      await directGate.promise;
    },
  });
  const directCache = new HttpfsLruChunkCache({
    fetchImpl: directMock.fetchImpl,
    cacheMaxBytes: 64 * 1024,
    chunkSize: 4096,
    readAheadChunks: 0,
  });
  const d0 = directCache.read('/game/direct_coalesce.rpf', 0, 512, { laneIndex: 0 });
  const d1 = directCache.read('/game/direct_coalesce.rpf', 256, 1024, { laneIndex: 3 });
  const d2 = directCache.read('/game/direct_coalesce.rpf', 1024, 512, { laneIndex: 7 });
  assert.equal(directMock.requests.length, 1);
  assert.equal(directCache.getStats().crossLaneCoalesced, 2);
  directGate.resolve();
  const [b0, b1, b2] = await Promise.all([d0, d1, d2]);
  assert.deepEqual(b0, rpfBytes.subarray(0, 512));
  assert.deepEqual(b1, rpfBytes.subarray(256, 1280));
  assert.deepEqual(b2, rpfBytes.subarray(1024, 1536));
  assert.equal(directMock.requests.length, 1);

  directCache.resetStats();
  assert.equal(directCache.getStats().crossLaneCoalesced, 0);
  assert.equal(directCache.getStats().coalescedReads, 0);
  assert.equal(directCache.getStats().cacheMisses, 0);
  assert.equal(directCache.getStats().cacheBytes, 4096);
});

// ---------------------------------------------------------------------------
// 10. W367: Adaptive read-ahead slab caching, zero-copy reads, deduplication,
//     LRU budget enforcement, EOF/short-read clamping, and error/retry safety
// ---------------------------------------------------------------------------

test('W367: adaptive sequential read-ahead ramps prefetch window, bounds concurrent prefetches, and supports clustered RPF gaps', async () => {
  const chunkSize = 4096;
  const totalChunks = 16; // 64 KiB across 16 chunks
  const rpfBytes = makePatternBytes(totalChunks * chunkSize, 173);

  let activeFetches = 0;
  let peakActiveFetches = 0;
  const mock = createMockRangeFetch(new Map([['/game/world_stream.rpf', rpfBytes]]), {
    onRequest: async () => {
      activeFetches += 1;
      if (activeFetches > peakActiveFetches) {
        peakActiveFetches = activeFetches;
      }
      await new Promise((r) => setTimeout(r, 2));
      activeFetches -= 1;
    },
  });

  const cache = new HttpfsLruChunkCache({
    fetchImpl: mock.fetchImpl,
    cacheMaxBytes: 128 * 1024, // 32 chunks budget
    chunkSize,
    readAheadChunks: 2,
    maxReadAheadChunks: 6,
    maxConcurrentPrefetches: 3,
    adaptiveReadAhead: true,
  });

  // Stream all 16 chunks sequentially; after chunk 1, background prefetches should satisfy
  // every subsequent chunk without issuing duplicate HTTP requests.
  for (let idx = 0; idx < totalChunks; idx += 1) {
    const got = await cache.read('/game/world_stream.rpf', idx * chunkSize, chunkSize);
    assert.deepEqual(got, rpfBytes.subarray(idx * chunkSize, (idx + 1) * chunkSize));
    // Active background prefetches must strictly obey maxConcurrentPrefetches.
    assert.ok(
      cache.getStats().activePrefetches <= 3,
      `activePrefetches exceeded maxConcurrentPrefetches=3: ${cache.getStats().activePrefetches}`,
    );
  }
  await cache.flushPrefetches();

  const stats = cache.getStats();
  assert.equal(mock.requests.length, totalChunks, 'Strictly 1 HTTP fetch per chunk (zero duplicates)');
  assert.equal(stats.cacheMisses, 2, 'Only chunks 0 and 1 should be demand misses');
  assert.equal(stats.prefetchIssued, totalChunks - 2);
  assert.equal(stats.prefetchHits, totalChunks - 2);
  assert.equal(stats.activePrefetches, 0);
  assert.equal(stats.adaptiveReadAhead, true);
  assert.equal(stats.maxReadAheadChunks, 6);
  assert.equal(stats.maxConcurrentPrefetches, 3);
  assert.ok(peakActiveFetches <= 4, `Peak concurrent fetches should stay bounded, got ${peakActiveFetches}`);

  // Clustered forward reads with small sector/entry alignment gaps (e.g. 256B RPF9 padding)
  // must trigger sequential read-ahead, whereas random seeks must NOT.
  const clusterBytes = makePatternBytes(6 * chunkSize, 181);
  const clusterMock = createMockRangeFetch(new Map([['/game/clustered.rpf', clusterBytes]]));
  const clusterCache = new HttpfsLruChunkCache({
    fetchImpl: clusterMock.fetchImpl,
    cacheMaxBytes: 64 * 1024,
    chunkSize,
    readAheadChunks: 2,
  });

  // Read [0..1024), then [1280..2304) (256B alignment gap <= chunkSize/2) -> triggers read-ahead!
  assert.deepEqual(
    await clusterCache.read('/game/clustered.rpf', 0, 1024),
    clusterBytes.subarray(0, 1024),
  );
  assert.equal(clusterCache.getStats().prefetchIssued, 0);
  assert.deepEqual(
    await clusterCache.read('/game/clustered.rpf', 1280, 1024),
    clusterBytes.subarray(1280, 2304),
  );
  await clusterCache.flushPrefetches();
  assert.equal(
    clusterCache.getStats().prefetchIssued,
    2,
    'Clustered forward read with small alignment gap must trigger read-ahead prefetches',
  );
  // Chunk 1 is now warm in cache (served synchronously via prefetchHit, while proactively prefetching chunk 3).
  assert.deepEqual(
    clusterCache.tryReadSync('/game/clustered.rpf', chunkSize, 1024),
    clusterBytes.subarray(chunkSize, chunkSize + 1024),
  );
  await clusterCache.flushPrefetches();
  assert.equal(clusterCache.getStats().cacheMisses, 1, 'Only chunk 0 was a demand miss');
  assert.equal(clusterCache.getStats().prefetchHits, 1, 'Chunk 1 served directly from prefetch');
  assert.equal(
    clusterMock.requests.filter((r) => r.rangeHeader === 'bytes=4096-8191').length,
    1,
    'Chunk 1 must never be re-fetched',
  );
});

test('W367: zero-copy tryReadIntoSync / readInto avoid extra copies, preserve caller canary bytes, and guard clear() generation', async () => {
  const chunkSize = 4096;
  const rpfBytes = makePatternBytes(3 * chunkSize + 500, 191);
  const saveBytes = makePatternBytes(2048, 197);
  const gate = createDeferred();
  let gateChunk2 = false;

  const mock = createMockRangeFetch(
    new Map([
      ['/game/zerocopy.rpf', rpfBytes],
      ['/userdata/dynamic.sav', saveBytes],
    ]),
    {
      onRequest: async (req) => {
        if (gateChunk2 && req.rangeHeader === 'bytes=8192-12287') {
          await gate.promise;
        }
      },
    },
  );

  const cache = new HttpfsLruChunkCache({
    fetchImpl: mock.fetchImpl,
    cacheMaxBytes: 64 * 1024,
    chunkSize,
    readAheadChunks: 0,
  });

  // Cold miss returns -1 without touching dst canary bytes.
  const dst = new Uint8Array(6000);
  dst.fill(0xa5);
  assert.equal(cache.tryReadIntoSync('/game/zerocopy.rpf', 100, 512, dst, 64), -1);
  assert.ok(dst.every((b) => b === 0xa5), 'Miss must not modify destination buffer');

  // readInto fetches chunk 0 and writes directly at dstOffset=64 without disturbing surrounding canaries.
  const n0 = await cache.readInto('/game/zerocopy.rpf', 100, 512, dst, 64);
  assert.equal(n0, 512);
  assert.deepEqual(dst.subarray(64, 64 + 512), rpfBytes.subarray(100, 612));
  assert.equal(dst[63], 0xa5);
  assert.equal(dst[64 + 512], 0xa5);
  assert.equal(mock.requests.length, 1);

  // Repeat read hits tryReadIntoSync synchronously (0 HTTP fetches, MRU fast path).
  const n0Sync = cache.tryReadIntoSync('/game/zerocopy.rpf', 200, 1024, dst, 128);
  assert.equal(n0Sync, 1024);
  assert.deepEqual(dst.subarray(128, 128 + 1024), rpfBytes.subarray(200, 1224));
  assert.equal(mock.requests.length, 1);

  // Cross-chunk readInto spanning chunks 0 and 1.
  const spanOff = chunkSize - 300;
  const nSpan = await cache.readInto('/game/zerocopy.rpf', spanOff, 900, dst, 32);
  assert.equal(nSpan, 900);
  assert.deepEqual(dst.subarray(32, 32 + 900), rpfBytes.subarray(spanOff, spanOff + 900));
  assert.equal(mock.requests.length, 2);

  // Cross-chunk tryReadIntoSync hits both cached chunks synchronously.
  const nSpanSync = cache.tryReadIntoSync('/game/zerocopy.rpf', spanOff + 50, 700, dst, 16);
  assert.equal(nSpanSync, 700);
  assert.deepEqual(dst.subarray(16, 16 + 700), rpfBytes.subarray(spanOff + 50, spanOff + 750));
  assert.equal(mock.requests.length, 2);

  // Non-cacheable URL bypasses LRU cleanly via readInto.
  assert.equal(cache.tryReadIntoSync('/userdata/dynamic.sav', 0, 256, dst, 0), -1);
  const nSave = await cache.readInto('/userdata/dynamic.sav', 128, 256, dst, 10);
  assert.equal(nSave, 256);
  assert.deepEqual(dst.subarray(10, 10 + 256), saveBytes.subarray(128, 384));

  // Generation guard: clear() while chunk 2 is in-flight prevents stale chunk 2 from repopulating cache.
  gateChunk2 = true;
  const inflightBuf = new Uint8Array(512);
  const inflightRead = cache.readInto('/game/zerocopy.rpf', 2 * chunkSize, 512, inflightBuf, 0);
  assert.equal(cache.getStats().inFlightRequests, 1);
  cache.clear();
  assert.equal(cache.getStats().cacheBytes, 0);

  gate.resolve();
  assert.equal(await inflightRead, 512);
  assert.deepEqual(inflightBuf, rpfBytes.subarray(2 * chunkSize, 2 * chunkSize + 512));
  assert.equal(
    cache.getStats().cacheBytes,
    0,
    'In-flight fetch started before clear() must not repopulate cleared cache',
  );
  assert.equal(cache.getStats().inFlightRequests, 0);
});

test('W367: LRU budget stays strictly bounded under sustained adaptive sequential streaming', async () => {
  const chunkSize = 4096;
  const totalChunks = 12;
  const budgetChunks = 4; // Strict 16 KiB cap for a 48 KiB stream
  const rpfBytes = makePatternBytes(totalChunks * chunkSize, 223);
  const mock = createMockRangeFetch(new Map([['/game/bounded_stream.rpf', rpfBytes]]));

  const cache = new HttpfsLruChunkCache({
    fetchImpl: mock.fetchImpl,
    cacheMaxBytes: budgetChunks * chunkSize,
    chunkSize,
    readAheadChunks: 2,
    maxReadAheadChunks: 8,
    adaptiveReadAhead: true,
  });

  for (let idx = 0; idx < totalChunks; idx += 1) {
    const got = await cache.read('/game/bounded_stream.rpf', idx * chunkSize, chunkSize);
    assert.deepEqual(got, rpfBytes.subarray(idx * chunkSize, (idx + 1) * chunkSize));
    await cache.flushPrefetches();
    const st = cache.getStats();
    assert.ok(
      st.cacheBytes <= budgetChunks * chunkSize,
      `cacheBytes (${st.cacheBytes}) exceeded budget (${budgetChunks * chunkSize}) at chunk ${idx}`,
    );
    assert.ok(st.cacheEntries <= budgetChunks);
  }

  const finalStats = cache.getStats();
  assert.ok(finalStats.cacheEvictions >= totalChunks - budgetChunks);
  assert.equal(finalStats.cacheBytes, budgetChunks * chunkSize);
  assert.equal(finalStats.prefetchHits, totalChunks - 2, 'Adaptive cap must never self-evict prefetched slabs');
  assert.equal(mock.requests.length, totalChunks, 'Zero thrash re-fetches despite tight LRU cap');
});

test('W367: error/retry safety cleans up in-flight state on network throws, supports opt-in transient retries, and clamps unknown-size EOF', async () => {
  const chunkSize = 4096;
  const tailSize = 2 * chunkSize + 777;
  const rpfBytes = makePatternBytes(tailSize, 233);

  // Part A: Default (maxRetries=0) network throw on demand + prefetch leaves zero stuck in-flight state.
  let throwChunk0Once = true;
  let throwChunk2Once = true;
  const flakyFetchA = async (url, init = {}) => {
    const range = init.headers && (init.headers.Range || init.headers.range);
    if (throwChunk0Once && range === 'bytes=0-4095') {
      throwChunk0Once = false;
      throw new TypeError('simulated network reset');
    }
    if (throwChunk2Once && range === 'bytes=8192-12287') {
      throwChunk2Once = false;
      throw new TypeError('simulated prefetch drop');
    }
    const base = createMockRangeFetch(new Map([[url, rpfBytes]]));
    return base.fetchImpl(url, init);
  };

  const cacheA = new HttpfsLruChunkCache({
    fetchImpl: flakyFetchA,
    cacheMaxBytes: 64 * 1024,
    chunkSize,
    readAheadChunks: 2,
  });

  await assert.rejects(() => cacheA.read('/game/flaky.rpf', 0, 512), /simulated network reset/);
  assert.equal(cacheA.getStats().inFlightRequests, 0);
  assert.equal(cacheA.getStats().activePrefetches, 0);
  assert.equal(cacheA.getStats().cacheBytes, 0);

  // Immediate retry succeeds cleanly.
  assert.deepEqual(await cacheA.read('/game/flaky.rpf', 0, 512), rpfBytes.subarray(0, 512));
  // Trigger sequential prefetch (chunk 1 succeeds, chunk 2 throws and is safely absorbed).
  assert.deepEqual(await cacheA.read('/game/flaky.rpf', 512, 512), rpfBytes.subarray(512, 1024));
  await cacheA.flushPrefetches();
  assert.equal(cacheA.getStats().inFlightRequests, 0);
  assert.equal(cacheA.getStats().activePrefetches, 0);

  // Demand read of chunk 2 (whose prefetch threw) retries cleanly and clamps at EOF (777B).
  const tailGot = await cacheA.read('/game/flaky.rpf', 2 * chunkSize, 2048);
  assert.equal(tailGot.byteLength, 777);
  assert.deepEqual(tailGot, rpfBytes.subarray(2 * chunkSize, tailSize));

  // Part B: Opt-in maxRetries retries transient 503 & network throws, does NOT retry 404.
  let chunk0Fails = 2;
  let notFoundCalls = 0;
  const retryMock = createMockRangeFetch(new Map([['/game/retry_ok.rpf', rpfBytes]]), {
    onRequest: async (req) => {
      if (req.url === '/game/missing_404.rpf') {
        notFoundCalls += 1;
      }
      if (req.url === '/game/retry_ok.rpf' && req.rangeHeader === 'bytes=0-4095' && chunk0Fails === 2) {
        chunk0Fails -= 1;
        throw new TypeError('transient socket hangup');
      }
    },
    statusOverrides: new Map([
      [
        '/game/retry_ok.rpf',
        (req) => {
          if (req.rangeHeader === 'bytes=0-4095' && chunk0Fails === 1) {
            chunk0Fails -= 1;
            return 503;
          }
          return null;
        },
      ],
    ]),
  });

  const retryCache = new HttpfsLruChunkCache({
    fetchImpl: retryMock.fetchImpl,
    cacheMaxBytes: 64 * 1024,
    chunkSize,
    readAheadChunks: 0,
    maxRetries: 2,
    retryBackoffMs: 1,
  });

  const recovered = await retryCache.read('/game/retry_ok.rpf', 0, 1024);
  assert.deepEqual(recovered, rpfBytes.subarray(0, 1024));
  assert.equal(retryCache.getStats().retryAttempts, 2);

  // 404 must fail immediately without retrying.
  const retriesBefore404 = retryCache.getStats().retryAttempts;
  await assert.rejects(() => retryCache.read('/game/missing_404.rpf', 0, 64), /HTTP 404/);
  assert.equal(notFoundCalls, 1);
  assert.equal(retryCache.getStats().retryAttempts, retriesBefore404);
  assert.equal(retryCache.getStats().inFlightRequests, 0);

  // Part C: Unknown total size (`Content-Range: bytes start-end/*`) clamps cleanly on partial tail
  // and learns exact EOF size so subsequent past-EOF reads perform zero network fetches.
  const omitMock = createMockRangeFetch(new Map([['/game/unknown_total.rpf', rpfBytes]]), {
    omitTotal: true,
  });
  const omitCache = new HttpfsLruChunkCache({
    fetchImpl: omitMock.fetchImpl,
    cacheMaxBytes: 64 * 1024,
    chunkSize,
    readAheadChunks: 2,
  });
  const spanAll = await omitCache.read('/game/unknown_total.rpf', 0, tailSize + 8192);
  await omitCache.flushPrefetches();
  assert.equal(spanAll.byteLength, tailSize);
  assert.deepEqual(spanAll, rpfBytes);
  assert.equal(omitCache.getKnownSize('/game/unknown_total.rpf'), tailSize);
  const reqsAfterLearn = omitMock.requests.length;
  assert.deepEqual(await omitCache.read('/game/unknown_total.rpf', tailSize, 4096), new Uint8Array(0));
  assert.equal(omitMock.requests.length, reqsAfterLearn, 'Past-EOF read after learning size must issue 0 fetches');
});

test('W367: HttpfsMailboxHost backgroundPrefetch double-buffers sequential windows & cross-thread multi-chunk sync reads stay exact', async () => {
  // Part A: Opt-in host backgroundPrefetch asynchronously fetches the next adaptive window
  // while draining the current window, promoting nextRaBuffer -> raBuffer with zero stall.
  const fileSize = 2 * 1024 * 1024; // 2 MiB = 32 x 64 KiB reads
  const fileBytes = makePatternBytes(fileSize, 251);
  const backendCalls = [];

  const sab = createHttpfsMailboxBuffer(64 * 1024);
  const host = new HttpfsMailboxHost(sab, {
    readAhead: {
      enabled: true,
      backgroundPrefetch: true,
      maxReadAheadBytes: 1024 * 1024,
    },
    open: () => 5,
    read: async (handle, offset, reqLen) => {
      if (handle !== 5) return null;
      backendCalls.push({ offset, reqLen });
      await new Promise((r) => setTimeout(r, 1));
      return fileBytes.subarray(offset, Math.min(fileSize, offset + reqLen));
    },
    close: (handle) => handle === 5,
  });
  const client = new HttpfsMailboxClient(sab);
  host.start();

  try {
    const h = await client.openAsync('update/bg_stream.rpf');
    assert.equal(h, 5);

    const assembled = new Uint8Array(fileSize);
    const stepBuf = new Uint8Array(64 * 1024);
    for (let i = 0; i < 32; i += 1) {
      const got = await client.readAsync(h, BigInt(i * 65536), stepBuf, 65536);
      assert.equal(got, 65536);
      assembled.set(stepBuf, i * 65536);
      // Allow background prefetch promise to settle between reads.
      if (i % 4 === 2) {
        await new Promise((r) => setTimeout(r, 4));
      }
    }
    assert.deepEqual(assembled, fileBytes);

    const tel = host.getHandleTelemetry(h);
    // Background prefetch promotes next window before miss, so >= 29 of 32 reads hit read-ahead!
    assert.ok(
      tel.readAheadHits >= 29,
      `Expected >= 29 readAheadHits with backgroundPrefetch, got ${tel.readAheadHits}`,
    );
    // 5 data windows + at most 1 EOF probe (buf.eof guard prevents repeated past-EOF prefetches)
    assert.ok(backendCalls.length <= 6, `Expected <= 6 backend calls, got ${backendCalls.length}`);

    // Random seek clears both raBuffer and nextRaBuffer cleanly.
    const seekBuf = new Uint8Array(1024);
    assert.equal(await client.readAsync(h, 4096n, seekBuf, 1024), 1024);
    assert.deepEqual(seekBuf, fileBytes.subarray(4096, 5120));
    assert.equal(await client.closeAsync(h), true);
  } finally {
    await host.stop();
  }

  // Part B: Cross-thread synchronous multi-chunk SAB streaming (exercises _executeRoundtripSync
  // bounded spin, direct ResponseReady -> RequestReady chunk chaining, and zero-copy host reads).
  const chunkSize = 32 * 1024;
  const streamSize = 512 * 1024 + 1337; // 16+ slabs, 8+ multi-chunk 64 KiB SAB frames per read
  const streamBytes = makePatternBytes(streamSize, 139);
  const mock = createMockRangeFetch(new Map([['/game/sync_stream.rpf', streamBytes]]));

  const syncSab = createHttpfsMailboxBuffer(64 * 1024);
  const syncHost = new HttpfsMailboxHost(syncSab, {
    fetchImpl: mock.fetchImpl,
    cacheMaxBytes: 1024 * 1024,
    chunkSize,
    readAheadChunks: 2,
    maxReadAheadChunks: 6,
    probeOnOpen: true,
  });
  syncHost.start();

  const workerCode = `
    const { workerData, parentPort } = require('node:worker_threads');
    const { HttpfsMailboxClient } = require(workerData.mailboxPath);
    const client = new HttpfsMailboxClient(workerData.sab, { timeoutMs: 8000 });

    const h = client.open('game/sync_stream.rpf');
    // Multi-chunk 256 KiB reads (4 x 64 KiB SAB roundtrips per call) across full file + EOF.
    const out1 = new Uint8Array(workerData.size);
    let off = 0;
    while (off < workerData.size) {
      const want = Math.min(256 * 1024, workerData.size - off);
      const slice = new Uint8Array(want);
      const got = client.read(h, BigInt(off), slice, want);
      if (got <= 0) break;
      out1.set(slice.subarray(0, got), off);
      off += got;
    }
    // Second pass hits 100% cached slabs via zero-copy tryReadIntoSync + spin fast path.
    const out2 = new Uint8Array(workerData.size);
    const got2 = client.read(h, 0n, out2, workerData.size);
    const eof = client.read(h, BigInt(workerData.size), new Uint8Array(64), 64);
    const closed = client.close(h);
    const checksum = (buf) => buf.reduce((a, b, i) => (a + b * ((i & 7) + 1)) >>> 0, 0);
    parentPort.postMessage({
      h,
      off,
      got2,
      eof,
      closed,
      sum1: checksum(out1),
      sum2: checksum(out2),
    });
  `;

  try {
    const res = await new Promise((resolve, reject) => {
      const w = new NodeWorker(workerCode, {
        eval: true,
        workerData: {
          sab: syncSab,
          size: streamSize,
          mailboxPath: path.join(repoRoot, 'web', 'workers', 'httpfs_mailbox.js'),
        },
      });
      w.once('message', resolve);
      w.once('error', reject);
      w.once('exit', (code) => {
        if (code !== 0) reject(new Error(`Worker exited with code ${code}`));
      });
    });

    await syncHost.flushPrefetches();
    const expectedSum = streamBytes.reduce((a, b, i) => (a + b * ((i & 7) + 1)) >>> 0, 0);
    assert.ok(res.h >= 1);
    assert.equal(res.off, streamSize);
    assert.equal(res.got2, streamSize);
    assert.equal(res.eof, 0);
    assert.equal(res.closed, true);
    assert.equal(res.sum1, expectedSum);
    assert.equal(res.sum2, expectedSum);

    const expectedChunks = Math.ceil(streamSize / chunkSize);
    assert.equal(
      mock.requests.length,
      1 + expectedChunks,
      `Expected 1 probe + ${expectedChunks} chunks across TWO full reads, got ${mock.requests.length}`,
    );
    const stats = syncHost.getStats();
    assert.ok(stats.syncFastPathHits >= 10, `Expected >= 10 syncFastPathHits, got ${stats.syncFastPathHits}`);
    assert.equal(stats.cacheBytes, streamSize);
    assert.equal(stats.activePrefetches, undefined); // host stats stay strictly sanitized
    assert.equal(syncHost.chunkCache.getStats().activePrefetches, 0);
  } finally {
    await syncHost.stop();
  }
});

test('W367: multi-lane out-of-order sequential slab reads deduplicate in-flight fetches and sustain prefetching across lanes', async () => {
  const chunkSize = 8192;
  const totalChunks = 8; // 64 KiB
  const rpfBytes = makePatternBytes(totalChunks * chunkSize, 163);
  const chunkGates = new Map();
  for (let i = 0; i < totalChunks; i += 1) {
    chunkGates.set(i, createDeferred());
  }

  const mock = createMockRangeFetch(new Map([['/game/multilane_seq.rpf', rpfBytes]]), {
    onRequest: async (req) => {
      const m = /^bytes=(\d+)-(\d+)$/.exec(req.rangeHeader || '');
      if (m) {
        const idx = Math.floor(Number(m[1]) / chunkSize);
        const gate = chunkGates.get(idx);
        if (gate) {
          await gate.promise;
        }
      }
    },
  });

  const slots = createHttpfsMailboxPool(4, 4096);
  const poolHost = new HttpfsMailboxPoolHost(slots, {
    fetchImpl: mock.fetchImpl,
    cacheMaxBytes: 128 * 1024,
    chunkSize,
    readAheadChunks: 2,
    maxReadAheadChunks: 4,
    probeOnOpen: false,
  });
  const clients = slots.map((sab) => new HttpfsMailboxClient(sab));
  poolHost.start();

  try {
    const handles = await Promise.all(clients.map((c) => c.openAsync('game/multilane_seq.rpf')));
    assert.ok(handles.every((h) => h >= 1));

    // Wave 1: 4 lanes issue overlapping + out-of-order reads across chunks 0 & 1 concurrently:
    //   lane 0: chunk 0 [0..4096)
    //   lane 1: chunk 0 [4096..8192)  (deduplicates with lane 0!)
    //   lane 2: chunk 1 [12288..16384) (arrives slightly ahead of [8192..12288)!)
    //   lane 3: chunk 1 [8192..12288)  (deduplicates with lane 2!)
    const b0 = new Uint8Array(4096);
    const b1 = new Uint8Array(4096);
    const b2 = new Uint8Array(4096);
    const b3 = new Uint8Array(4096);
    const p0 = clients[0].readAsync(handles[0], 0n, b0, 4096);
    const p1 = clients[1].readAsync(handles[1], 4096n, b1, 4096);
    const p2 = clients[2].readAsync(handles[2], 12288n, b2, 4096);
    const p3 = clients[3].readAsync(handles[3], 8192n, b3, 4096);

    await waitForCondition(
      () => poolHost.getStats().activeLanes === 4 && mock.requests.length >= 2,
      3000,
      '4 lanes dispatched across chunks 0 & 1',
    );

    // Strictly 1 HTTP fetch for chunk 0 and 1 for chunk 1 (no duplicates!).
    assert.equal(
      mock.requests.filter((r) => r.rangeHeader === 'bytes=0-8191').length,
      1,
      'Chunk 0 must have strictly 1 in-flight HTTP fetch across lanes 0 and 1',
    );
    assert.equal(
      mock.requests.filter((r) => r.rangeHeader === 'bytes=8192-16383').length,
      1,
      'Chunk 1 must have strictly 1 in-flight HTTP fetch across lanes 2 and 3',
    );

    // Release all chunk gates so wave 1 and subsequent background prefetches complete.
    for (const gate of chunkGates.values()) {
      gate.resolve();
    }

    const lens = await Promise.all([p0, p1, p2, p3]);
    assert.deepEqual(lens, [4096, 4096, 4096, 4096]);
    assert.deepEqual(b0, rpfBytes.subarray(0, 4096));
    assert.deepEqual(b1, rpfBytes.subarray(4096, 8192));
    assert.deepEqual(b2, rpfBytes.subarray(12288, 16384));
    assert.deepEqual(b3, rpfBytes.subarray(8192, 12288));

    await poolHost.flushPrefetches();
    const st = poolHost.getStats();
    assert.ok(st.crossLaneCoalesced >= 2, `Expected >= 2 crossLaneCoalesced, got ${st.crossLaneCoalesced}`);
    assert.ok(st.prefetchIssued >= 1, `Expected out-of-order multi-lane sequential stream to issue prefetches, got ${st.prefetchIssued}`);
    assert.ok(st.cacheBytes <= 128 * 1024);
  } finally {
    await poolHost.stop();
  }
});

// ---------------------------------------------------------------------------
// 11. W368: Per-lane atomic SAB claim (Claimed = 3), multi-worker contention,
//     host Claimed waiting, shutdown wakeups, and ArrayBuffer responder fallback
// ---------------------------------------------------------------------------

test('W368: two worker_threads sharing the exact same SAB lane run concurrent open/read/close with zero corruption', async () => {
  assert.equal(HttpfsStatus.Claimed, 3);
  assert.equal(HttpfsStatus.CLAIMED, 3);

  const fileSize = 24 * 1024 + 333; // 6+ x 4 KiB SAB chunks per read -> heavy multi-chunk contention
  const fileA = makePatternBytes(fileSize, 41);
  const fileB = makePatternBytes(fileSize, 97);

  // Small async fetch delay guarantees both workers overlap on the same SAB lane.
  const mock = createMockRangeFetch(
    new Map([
      ['/game/lane_share_a.rpf', fileA],
      ['/game/lane_share_b.rpf', fileB],
    ]),
    { delayMs: 2 },
  );

  const sharedSab = createHttpfsMailboxBuffer(4096);
  const barrierSab = new SharedArrayBuffer(4);
  const barrierView = new Int32Array(barrierSab, 0, 1);

  const host = new HttpfsMailboxHost(sharedSab, {
    fetchImpl: mock.fetchImpl,
    cacheMaxBytes: 128 * 1024,
    chunkSize: 8192,
    readAheadChunks: 1,
    probeOnOpen: true,
  });
  host.start();

  const workerCode = `
    const { workerData, parentPort } = require('node:worker_threads');
    const { HttpfsMailboxClient } = require(workerData.mailboxPath);
    const client = new HttpfsMailboxClient(workerData.sab, { timeoutMs: 10000 });
    const barrier = new Int32Array(workerData.barrierSab, 0, 1);

    // Synchronize both workers so they contend on the same SAB lane simultaneously.
    const arrived = Atomics.add(barrier, 0, 1) + 1;
    if (arrived >= 2) {
      Atomics.notify(barrier, 0);
    } else {
      while (Atomics.load(barrier, 0) < 2) {
        Atomics.wait(barrier, 0, 1, 5000);
      }
    }

    const rounds = [];
    for (let round = 0; round < 3; round += 1) {
      const h = client.open(workerData.relPath);
      if (h < 1) {
        throw new Error('open failed for ' + workerData.relPath + ': ' + h);
      }
      const full = new Uint8Array(workerData.size);
      const n1 = client.read(h, 0n, full, workerData.size);
      const midOff = 4096 + round * 512;
      const midLen = 6000;
      const mid = new Uint8Array(midLen);
      const n2 = client.read(h, BigInt(midOff), mid, midLen);
      const eof = client.read(h, BigInt(workerData.size), new Uint8Array(32), 32);
      const closed = client.close(h);
      rounds.push({
        h,
        n1,
        n2,
        midOff,
        eof,
        closed,
        full: Array.from(full),
        mid: Array.from(mid),
      });
    }
    parentPort.postMessage({ relPath: workerData.relPath, rounds });
  `;

  const spawnWorker = (relPath) =>
    new Promise((resolve, reject) => {
      const w = new NodeWorker(workerCode, {
        eval: true,
        workerData: {
          sab: sharedSab,
          barrierSab,
          relPath,
          size: fileSize,
          mailboxPath: path.join(repoRoot, 'web', 'workers', 'httpfs_mailbox.js'),
        },
      });
      w.once('message', resolve);
      w.once('error', reject);
      w.once('exit', (code) => {
        if (code !== 0) reject(new Error(`Worker (${relPath}) exited with code ${code}`));
      });
    });

  try {
    const [resA, resB] = await Promise.all([
      spawnWorker('game/lane_share_a.rpf'),
      spawnWorker('game/lane_share_b.rpf'),
    ]);

    for (const [res, expectedBytes] of [
      [resA, fileA],
      [resB, fileB],
    ]) {
      assert.equal(res.rounds.length, 3);
      for (const r of res.rounds) {
        assert.ok(r.h >= 1);
        assert.equal(r.n1, fileSize);
        assert.equal(r.n2, 6000);
        assert.equal(r.eof, 0);
        assert.equal(r.closed, true);
        assert.deepEqual(new Uint8Array(r.full), expectedBytes);
        assert.deepEqual(
          new Uint8Array(r.mid),
          expectedBytes.subarray(r.midOff, r.midOff + 6000),
        );
      }
    }

    // Also verify two async clients sharing the same SAB lane serialize cleanly via _acquireLaneAsync().
    const asyncClient1 = new HttpfsMailboxClient(sharedSab, { timeoutMs: 5000 });
    const asyncClient2 = new HttpfsMailboxClient(sharedSab, { timeoutMs: 5000 });
    const [hA, hB] = await Promise.all([
      asyncClient1.openAsync('game/lane_share_a.rpf'),
      asyncClient2.openAsync('game/lane_share_b.rpf'),
    ]);
    assert.ok(hA >= 1 && hB >= 1 && hA !== hB);
    const outA = new Uint8Array(fileSize);
    const outB = new Uint8Array(fileSize);
    const [nA, nB] = await Promise.all([
      asyncClient1.readAsync(hA, 0n, outA, fileSize),
      asyncClient2.readAsync(hB, 0n, outB, fileSize),
    ]);
    assert.equal(nA, fileSize);
    assert.equal(nB, fileSize);
    assert.deepEqual(outA, fileA);
    assert.deepEqual(outB, fileB);
    const [cA, cB] = await Promise.all([
      asyncClient1.closeAsync(hA),
      asyncClient2.closeAsync(hB),
    ]);
    assert.equal(cA, true);
    assert.equal(cB, true);

    const statusView = new Int32Array(sharedSab, 0, 1);
    assert.equal(Atomics.load(statusView, 0), HttpfsStatus.Idle);
  } finally {
    await host.stop();
  }
});

test('W368: HttpfsMailboxHost waits cleanly on Claimed (3) until RequestReady (1) without corrupting state', async () => {
  const payloadBytes = makePatternBytes(2048, 73);
  const mock = createMockRangeFetch(new Map([['/game/claimed_wait.rpf', payloadBytes]]));

  const sab = createHttpfsMailboxBuffer(4096);
  const statusView = new Int32Array(sab, 0, 1);
  const headerView = new DataView(sab, 0, HTTPFS_HEADER_BYTES);
  const payloadView = new Uint8Array(sab, HTTPFS_HEADER_BYTES, 4096);

  const host = new HttpfsMailboxHost(sab, {
    fetchImpl: mock.fetchImpl,
    cacheMaxBytes: 32 * 1024,
    chunkSize: 4096,
    readAheadChunks: 0,
    probeOnOpen: false,
  });

  // Pre-claim the lane before any host poll/dispatch: processOnce() must treat Claimed as not ready.
  Atomics.store(statusView, 0, HttpfsStatus.Claimed);
  assert.equal(await host.processOnce(), false);
  assert.equal(Atomics.load(statusView, 0), HttpfsStatus.Claimed);
  assert.equal(host.getStats().requests, 0);

  // Start the host loop while status is Claimed (3); it must wait cleanly without erroring.
  host.start();
  try {
    await new Promise((r) => setTimeout(r, 25));
    assert.equal(Atomics.load(statusView, 0), HttpfsStatus.Claimed);
    assert.equal(host.getStats().requests, 0);

    // Now finish populating the Open request and transition Claimed -> RequestReady + notify.
    const encodedPath = new TextEncoder().encode('game/claimed_wait.rpf');
    headerView.setUint32(HttpfsOffsets.OP, HttpfsOp.Open, true);
    headerView.setInt32(HttpfsOffsets.HANDLE, -1, true);
    headerView.setBigUint64(HttpfsOffsets.OFFSET, 0n, true);
    headerView.setBigUint64(HttpfsOffsets.LENGTH, BigInt(encodedPath.byteLength), true);
    headerView.setBigInt64(HttpfsOffsets.RESULT, -1n, true);
    payloadView.set(encodedPath, 0);
    Atomics.store(statusView, 0, HttpfsStatus.RequestReady);
    Atomics.notify(statusView, 0);

    await waitForCondition(
      () => Atomics.load(statusView, 0) === HttpfsStatus.ResponseReady,
      3000,
      'host wakes from Claimed and completes Open',
    );

    const openedHandle = Number(headerView.getBigInt64(HttpfsOffsets.RESULT, true));
    assert.ok(openedHandle >= 1);
    assert.equal(host.getStats().requests, 1);

    // Release back to Idle and verify normal client read + close succeeds cleanly.
    Atomics.store(statusView, 0, HttpfsStatus.Idle);
    Atomics.notify(statusView, 0);

    const client = new HttpfsMailboxClient(sab, { timeoutMs: 3000 });
    const out = new Uint8Array(2048);
    assert.equal(await client.readAsync(openedHandle, 0n, out, 2048), 2048);
    assert.deepEqual(out, payloadBytes);
    assert.equal(await client.closeAsync(openedHandle), true);
  } finally {
    await host.stop();
  }
});

test('W368: shutdown wakes sync and async clients waiting to acquire a Claimed lane, and plain ArrayBuffer responder hooks remain intact', async () => {
  // Part A: Sync worker waiting in _acquireLaneSync() on a Claimed lane wakes immediately on Shutdown.
  const sabSync = createHttpfsMailboxBuffer(4096);
  const statusSync = new Int32Array(sabSync, 0, 1);
  Atomics.store(statusSync, 0, HttpfsStatus.Claimed);

  const workerCode = `
    const { workerData, parentPort } = require('node:worker_threads');
    const { HttpfsMailboxClient } = require(workerData.mailboxPath);
    const client = new HttpfsMailboxClient(workerData.sab, { timeoutMs: 10000 });
    const t0 = Date.now();
    parentPort.postMessage({ started: true });
    const h = client.open('game/blocked_on_claim.rpf');
    const r = client.read(1, 0n, new Uint8Array(64), 64);
    const c = client.close(1);
    parentPort.postMessage({ done: true, h, r, c, elapsedMs: Date.now() - t0 });
  `;

  const workerResult = await new Promise((resolve, reject) => {
    const w = new NodeWorker(workerCode, {
      eval: true,
      workerData: {
        sab: sabSync,
        mailboxPath: path.join(repoRoot, 'web', 'workers', 'httpfs_mailbox.js'),
      },
    });
    w.on('message', (msg) => {
      if (msg.started) {
        setTimeout(() => {
          HttpfsMailboxClient.shutdownWaiter(sabSync);
        }, 25);
      } else if (msg.done) {
        resolve(msg);
      }
    });
    w.once('error', reject);
    w.once('exit', (code) => {
      if (code !== 0) reject(new Error(`Shutdown worker exited with code ${code}`));
    });
  });

  assert.equal(workerResult.h, -1);
  assert.equal(workerResult.r, -1);
  assert.equal(workerResult.c, false);
  assert.ok(
    workerResult.elapsedMs < 3000,
    `Shutdown must wake client promptly (got ${workerResult.elapsedMs}ms)`,
  );

  // Part B: Async clients waiting in _acquireLaneAsync() on a Claimed lane wake immediately on Shutdown.
  const sabAsync = createHttpfsMailboxBuffer(4096);
  const statusAsync = new Int32Array(sabAsync, 0, 1);
  Atomics.store(statusAsync, 0, HttpfsStatus.Claimed);
  const asyncClient = new HttpfsMailboxClient(sabAsync, { timeoutMs: 10000 });
  const t0Async = Date.now();
  const pendingOpen = asyncClient.openAsync('game/blocked_async.rpf');
  const pendingRead = asyncClient.readAsync(1, 0n, new Uint8Array(64), 64);
  const pendingClose = asyncClient.closeAsync(1);
  setTimeout(() => {
    HttpfsMailboxClient.shutdownWaiter(sabAsync);
  }, 20);
  const [aOpen, aRead, aClose] = await Promise.all([pendingOpen, pendingRead, pendingClose]);
  assert.equal(aOpen, -1);
  assert.equal(aRead, -1);
  assert.equal(aClose, false);
  assert.ok(Date.now() - t0Async < 3000);

  // Part C: Plain (non-Shared) ArrayBuffer with synchronous responder hook works identically.
  const ab = new ArrayBuffer(HTTPFS_HEADER_BYTES + 256);
  const abBytes = makePatternBytes(600, 19);
  const abClient = new HttpfsMailboxClient(ab, (buf) => {
    const status = new Int32Array(buf, 0, 1);
    const header = new DataView(buf, 0, HTTPFS_HEADER_BYTES);
    const payload = new Uint8Array(buf, HTTPFS_HEADER_BYTES, 256);
    const op = header.getUint32(HttpfsOffsets.OP, true);
    if (op === HttpfsOp.Open) {
      header.setBigInt64(HttpfsOffsets.RESULT, 7n, true);
      status[0] = HttpfsStatus.ResponseReady;
    } else if (op === HttpfsOp.Read) {
      const off = Number(header.getBigUint64(HttpfsOffsets.OFFSET, true));
      const len = Number(header.getBigUint64(HttpfsOffsets.LENGTH, true));
      const avail = Math.max(0, Math.min(len, abBytes.byteLength - off));
      if (avail > 0) {
        payload.set(abBytes.subarray(off, off + avail), 0);
      }
      header.setBigInt64(HttpfsOffsets.RESULT, BigInt(avail), true);
      status[0] = HttpfsStatus.ResponseReady;
    } else if (op === HttpfsOp.Close) {
      header.setBigInt64(HttpfsOffsets.RESULT, 0n, true);
      status[0] = HttpfsStatus.ResponseReady;
    }
  });

  assert.equal(abClient.open('game/plain_ab.rpf'), 7);
  const abOut = new Uint8Array(600);
  assert.equal(abClient.read(7, 0n, abOut, 600), 600);
  assert.deepEqual(abOut, abBytes);
  assert.equal(abClient.close(7), true);
  assert.equal(new Int32Array(ab, 0, 1)[0], HttpfsStatus.Idle);
});
