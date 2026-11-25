import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import vm from 'node:vm';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { OpfsChunkCache } from '../web/storage/opfs_cache.js';

const require = createRequire(import.meta.url);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');

const ioWorkerMod = require('../web/workers/io_worker.js');
const mailboxMod = require('../web/workers/httpfs_mailbox.js');
const rangeCacheMod = require('../web/workers/range_cache.js');

const {
  IO_PRIORITY,
  DEFAULT_IMMUTABLE_CHUNK_SIZE,
  DEFAULT_IMMUTABLE_MAX_BYTES,
  DEFAULT_SLAB_SIZE,
  DEFAULT_SLAB_MAX_BYTES,
  DEFAULT_DUMP_ENDPOINT,
  DUMP_VIRTUAL_PREFIX,
  HTTPFS_OP_WRITE,
  HTTPFS_OPEN_FLAG_WRITE,
  HTTPFS_OPEN_FLAG_APPEND,
  MAX_SLAB_SIZE,
  MAX_SLAB_CACHE_BYTES,
  SLAB_TUNING_PROFILES,
  VIDEO_STREAM_SLAB_PRESET,
  DEFAULT_BATCH_COALESCE_MAX_GAP_BYTES,
  DEFAULT_BATCH_COALESCE_MAX_BYTES,
  DEFAULT_HOT_OPFS_RPF_ALLOWLIST,
  DEFAULT_OPFS_L2_MAX_PENDING_WRITES,
  DEFAULT_OPFS_L2_MAX_WRITE_CONCURRENCY,
  ImmutableHttpfsCache,
  IoWorkerController,
  coalesceBatchReadRequests,
  collectTransferableBuffers,
  createSlabCacheTuningProfile,
  emptyStats,
  emptyImmutableCacheStats,
  emptyStreamingStats,
  isDumpVirtualPath,
  isImmutableHttpfsPath,
  isVideoStreamPath,
  buildFileFingerprint,
  normalizeOpfsL2Path,
  isHotOpfsL2Path,
  opfsL2RangeFileName,
  parseContentRangeHeader,
  resolveDumpConfig,
  resolveDumpWriteRoute,
  resolveHttpfsRoute,
} = ioWorkerMod;

const {
  HttpfsStatus,
  HttpfsOp,
  HttpfsOffsets,
  createHttpfsMailboxBuffer,
  HttpfsMailboxClient,
} = mailboxMod;

const utf8Encoder = new TextEncoder();

function makePatternBytes(length, seed = 29) {
  const out = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = ((i * 73) ^ ((i >> 5) * 37) ^ ((i >> 11) * 13) ^ seed) & 0xff;
  }
  return out;
}

function createMockRangeFetch(urlBytesMap, options = {}) {
  const requests = [];
  const statusOverrides = options.statusOverrides || new Map();
  const maxServeBytes = Number.isInteger(options.maxServeBytes) ? options.maxServeBytes : null;
  const force200 = Boolean(options.force200);
  const delayMs = options.delayMs || 0;

  async function fetchImpl(url, init = {}) {
    const method = (init && init.method) || 'GET';
    const rangeHeader = (init && init.headers && (init.headers.Range || init.headers.range)) || null;
    requests.push({ method, url, rangeHeader });

    if (delayMs > 0) {
      await new Promise((r) => setTimeout(r, delayMs));
    }

    const override = statusOverrides.get(url);
    const overrideStatus = typeof override === 'function' ? override(requests.length) : override;
    if (overrideStatus) {
      return {
        ok: false,
        status: overrideStatus,
        headers: new Map(),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
        async json() {
          throw new Error(`HTTP ${overrideStatus}`);
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
        async json() {
          throw new Error('HTTP 404');
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

  return { fetchImpl, requests };
}

async function rawMailboxOpen(sab, relPath) {
  const statusView = new Int32Array(sab, 0, 1);
  const headerView = new DataView(sab, 0, 40);
  const payloadView = new Uint8Array(sab, 40, sab.byteLength - 40);

  const encoded = utf8Encoder.encode(relPath);
  headerView.setUint32(HttpfsOffsets.OP, HttpfsOp.Open, true);
  headerView.setInt32(HttpfsOffsets.HANDLE, -1, true);
  headerView.setBigUint64(HttpfsOffsets.OFFSET, 0n, true);
  headerView.setBigUint64(HttpfsOffsets.LENGTH, BigInt(encoded.byteLength), true);
  headerView.setBigInt64(HttpfsOffsets.RESULT, -1n, true);
  payloadView.set(encoded, 0);

  Atomics.store(statusView, 0, HttpfsStatus.RequestReady);
  Atomics.notify(statusView, 0, 1);

  let cur = Atomics.load(statusView, 0);
  while (cur === HttpfsStatus.RequestReady) {
    const waitObj = Atomics.waitAsync(statusView, 0, HttpfsStatus.RequestReady, 5000);
    const res = waitObj.async ? await waitObj.value : waitObj.value;
    cur = Atomics.load(statusView, 0);
    if (res === 'timed-out' && cur === HttpfsStatus.RequestReady) {
      throw new Error(`Mailbox hung during OPEN(${relPath})`);
    }
  }

  const result = headerView.getBigInt64(HttpfsOffsets.RESULT, true);
  Atomics.store(statusView, 0, HttpfsStatus.Idle);
  Atomics.notify(statusView, 0, 1);
  return { status: cur, result };
}

async function startRangeHttpServer(routesMap) {
  const requests = [];
  const server = http.createServer((req, res) => {
    const reqUrl = new URL(req.url || '/', 'http://127.0.0.1');
    const pathname = reqUrl.pathname;
    const rangeHeader = req.headers.range || null;
    requests.push({ method: req.method, pathname, rangeHeader });

    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
    res.setHeader('Accept-Ranges', 'bytes');

    const data = routesMap.get(pathname);
    if (!data) {
      res.writeHead(404);
      res.end();
      return;
    }

    if (!rangeHeader) {
      res.writeHead(200, { 'Content-Length': String(data.byteLength) });
      res.end(Buffer.from(data.buffer, data.byteOffset, data.byteLength));
      return;
    }

    const match = /^bytes=(\d+)-(\d+)$/.exec(rangeHeader);
    if (!match) {
      res.writeHead(400);
      res.end();
      return;
    }
    const reqStart = Number(match[1]);
    const reqEnd = Number(match[2]);
    if (reqStart >= data.byteLength) {
      res.writeHead(416, { 'Content-Range': `bytes */${data.byteLength}` });
      res.end();
      return;
    }

    const actualEnd = Math.min(reqEnd, data.byteLength - 1);
    const slice = data.subarray(reqStart, actualEnd + 1);
    res.writeHead(206, {
      'Content-Range': `bytes ${reqStart}-${actualEnd}/${data.byteLength}`,
      'Content-Length': String(slice.byteLength),
    });
    res.end(Buffer.from(slice.buffer, slice.byteOffset, slice.byteLength));
  });

  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => resolve());
  });

  const addr = server.address();
  return {
    origin: `http://127.0.0.1:${addr.port}`,
    requests,
    close: () => new Promise((resolve) => server.close(() => resolve())),
  };
}

// ---------------------------------------------------------------------------
// 1. Predicate & Content-Range parser unit tests
// ---------------------------------------------------------------------------

test('isImmutableHttpfsPath matches *.rpf, *.pgim, *.dll, and MicrosoftGame.Config only', () => {
  assert.equal(DEFAULT_IMMUTABLE_CHUNK_SIZE, 64 * 1024);
  assert.equal(DEFAULT_IMMUTABLE_MAX_BYTES, 128 * 1024 * 1024);
  assert.equal(DEFAULT_SLAB_SIZE, 2 * 1024 * 1024);
  assert.equal(DEFAULT_SLAB_MAX_BYTES, 192 * 1024 * 1024);
  assert.equal(MAX_SLAB_CACHE_BYTES, 256 * 1024 * 1024);
  assert.equal(SLAB_TUNING_PROFILES.default.slabMaxBytes, 192 * 1024 * 1024);
  assert.equal(SLAB_TUNING_PROFILES.video4k.slabMaxBytes, 192 * 1024 * 1024);
  assert.equal(SLAB_TUNING_PROFILES['low-memory'].slabMaxBytes, 16 * 1024 * 1024);

  for (const p of [
    'audio.rpf',
    'x64/movies.RPF',
    'main/image/guest.pgim',
    'xgameruntime.dll',
    'sub/WINHTTP.DLL',
    'MicrosoftGame.Config',
    'Game/microsoftgame.config',
  ]) {
    assert.equal(isImmutableHttpfsPath(p), true, `Expected immutable match for ${p}`);
  }

  for (const p of [
    '',
    null,
    'data-manifest.json',
    'bootset.json',
    'program-manifest.json',
    'config/settings.xml',
    'userdata/slot0.sav',
    'notes.txt',
  ]) {
    assert.equal(isImmutableHttpfsPath(p), false, `Expected non-immutable for ${p}`);
  }
});

test('parseContentRangeHeader parses 206 ranges, unknown totals, and 416 unsatisfied headers', () => {
  assert.deepEqual(parseContentRangeHeader('bytes 0-65535/1048576'), {
    start: 0,
    end: 65535,
    total: 1048576,
    unsatisfied: false,
  });
  assert.deepEqual(parseContentRangeHeader('bytes 65536-131071/*'), {
    start: 65536,
    end: 131071,
    total: null,
    unsatisfied: false,
  });
  assert.deepEqual(parseContentRangeHeader('bytes */0'), {
    start: null,
    end: null,
    total: 0,
    unsatisfied: true,
  });
  assert.deepEqual(parseContentRangeHeader('bytes */4096'), {
    start: null,
    end: null,
    total: 4096,
    unsatisfied: true,
  });
  assert.equal(parseContentRangeHeader('invalid'), null);
  assert.equal(parseContentRangeHeader('bytes 100-50/200'), null);
});

// ---------------------------------------------------------------------------
// 2. Repeated open + overlapping reads trigger ONLY 1 HTTP 206 fetch
// ---------------------------------------------------------------------------

test('repeated open + overlapping reads on immutable /game/*.rpf trigger only 1 chunked HTTP 206 fetch', async () => {
  const rpfSize = 180 * 1024; // 180 KiB archive
  const audioRpf = makePatternBytes(rpfSize, 42);

  const mock = createMockRangeFetch(new Map([['/game/audio.rpf', audioRpf]]));
  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    opfsDir: null,
  });

  const sab = createHttpfsMailboxBuffer(64 * 1024);
  const client = new HttpfsMailboxClient(sab);

  try {
    await controller.init({ dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' });
    await controller.attachMailbox(sab);

    const reqsAfterInit = mock.requests.length;

    // 1. Open using Xbox drive path -> triggers 1 aligned slab-0 HTTP 206 fetch (`bytes=0-2097151`, clamped by server to 180 KiB).
    const h1 = await client.openAsync('G:\\audio.rpf');
    assert.ok(h1 >= 1);
    assert.equal(controller.getHandleInfo(h1).size, rpfSize);
    assert.equal(mock.requests.length, reqsAfterInit + 1);
    assert.equal(mock.requests[reqsAfterInit].url, '/game/audio.rpf');
    assert.equal(mock.requests[reqsAfterInit].rangeHeader, `bytes=0-${DEFAULT_SLAB_SIZE - 1}`);

    // 2. Re-open same file via multiple path aliases -> ZERO additional network requests!
    const h2 = await client.openAsync('\\\\?\\G:\\audio.rpf');
    const h3 = await client.openAsync('/game/audio.rpf');
    const h4 = await client.openAsync('audio.rpf');
    assert.ok(h2 >= 1 && h3 >= 1 && h4 >= 1);
    assert.equal(controller.getHandleInfo(h4).size, rpfSize);
    assert.equal(mock.requests.length, reqsAfterInit + 1);

    // 3. Cached stat() returns exact file length with ZERO network round-trips!
    const st = await controller.stat('G:\\audio.rpf');
    assert.ok(st);
    assert.equal(st.size, rpfSize);
    assert.equal(mock.requests.length, reqsAfterInit + 1);

    // 4. Perform overlapping 4 KiB - 64 KiB header/table reads across all handles -> ALL RAM LRU hits!
    const ranges = [
      [h1, 0, 4096],
      [h2, 16, 4096],
      [h3, 2048, 16384],
      [h4, 0, 65536],
      [h1, 49152, 16384],
      [h2, 123, 7777],
    ];

    for (const [h, off, len] of ranges) {
      const dst = new Uint8Array(len);
      const got = await client.readAsync(h, BigInt(off), dst, len);
      assert.equal(got, len);
      assert.deepEqual(dst, audioRpf.subarray(off, off + len));
      // Mutate caller buffer to ensure cached chunk memory is completely isolated.
      dst.fill(0xaa);
    }

    // Re-read offset 0 after caller mutation to confirm cached chunk is pristine.
    const verifyBuf = new Uint8Array(256);
    assert.equal(await client.readAsync(h1, 0n, verifyBuf, 256), 256);
    assert.deepEqual(verifyBuf, audioRpf.subarray(0, 256));

    // Past-EOF read uses cached file size -> returns 0 immediately with ZERO network requests!
    const eofBuf = new Uint8Array(64);
    assert.equal(await client.readAsync(h1, BigInt(rpfSize), eofBuf, 64), 0);
    assert.equal(await client.readAsync(h1, BigInt(rpfSize + 4096), eofBuf, 64), 0);

    // TOTAL HTTP requests for 4 opens + 9 reads + stat: STILL ONLY 1!
    const gameRequests = mock.requests.slice(reqsAfterInit);
    assert.equal(gameRequests.length, 1, `Expected exactly 1 HTTP 206 fetch, got ${gameRequests.length}`);

    assert.equal(await client.closeAsync(h1), true);
    assert.equal(await client.closeAsync(h2), true);
    assert.equal(await client.closeAsync(h3), true);
    assert.equal(await client.closeAsync(h4), true);

    // Verify sanitized stats (numeric only, no host paths or URLs)
    const stats = controller.getStats().immutableCache;
    assert.equal(stats.networkFetches, 1);
    assert.equal(stats.metaMisses, 1);
    assert.equal(stats.metaHits, 3);
    assert.equal(stats.slabCacheMisses, 1);
    assert.equal(stats.slabCacheHits, 7);
    assert.equal(stats.slabBytesCached, rpfSize);
    assert.equal(stats.slabEntries, 1);
    // Chunk path is never populated when slab mode is active (zero chunk+slab overlap).
    assert.equal(stats.misses, 0);
    assert.equal(stats.hits, 0);
    assert.equal(stats.bytesCached, 0);
    assert.equal(stats.entries, 0);
    assert.equal(stats.filesIndexed, 1);
    const statsJson = JSON.stringify(stats);
    assert.equal(statsJson.includes('audio.rpf'), false, 'Stats must not leak file paths');
    assert.equal(statsJson.includes('/game/'), false, 'Stats must not leak URLs');
  } finally {
    await controller.detachMailbox();
  }
});

// ---------------------------------------------------------------------------
// 3. Multi-chunk reads spanning chunk boundaries + real HTTP 206 Range server
// ---------------------------------------------------------------------------

test('multi-chunk reads spanning chunk boundaries assemble exact bytes in order and clamp tail range to file size', async () => {
  const fileSize = 410; // 7 chunks of 64B (last chunk is 26B: [384..409])
  const movieRpf = makePatternBytes(fileSize, 91);

  const mock = createMockRangeFetch(new Map([['/game/movies.rpf', movieRpf]]));
  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    immutableCacheChunkSize: 64,
    immutableCacheMaxBytes: 4096,
    opfsDir: null,
  });

  const sab = createHttpfsMailboxBuffer(1024);
  const client = new HttpfsMailboxClient(sab);

  try {
    await controller.init({ dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' });
    await controller.attachMailbox(sab);

    const reqsBefore = mock.requests.length;
    const h = await client.openAsync('G:\\movies.rpf');
    assert.ok(h >= 1);
    assert.equal(controller.getHandleInfo(h).size, fileSize);
    assert.equal(mock.requests.length, reqsBefore + 1);
    assert.equal(mock.requests[reqsBefore].rangeHeader, 'bytes=0-63');

    // Read spanning chunk 0 (already cached) + chunks 1, 2, 3 (unaligned start 50, length 180 -> [50..230))
    const midBuf = new Uint8Array(180);
    const midGot = await client.readAsync(h, 50n, midBuf, 180);
    assert.equal(midGot, 180);
    assert.deepEqual(midBuf, movieRpf.subarray(50, 230));

    // Coalesced fetch should have fetched chunks 1..3 in ONE single HTTP 206 (`bytes=64-255`)!
    assert.equal(mock.requests.length, reqsBefore + 2);
    assert.equal(mock.requests[reqsBefore + 1].rangeHeader, 'bytes=64-255');

    // Read spanning cached chunk 3 + missing chunks 4, 5, 6 and extending past EOF (offset 240, reqLen 300 -> clamps to 170B)
    const tailBuf = new Uint8Array(300);
    const tailGot = await client.readAsync(h, 240n, tailBuf, 300);
    assert.equal(tailGot, fileSize - 240);
    assert.deepEqual(tailBuf.subarray(0, tailGot), movieRpf.subarray(240, fileSize));

    // Tail range must be clamped to known fileSize - 1 (`bytes=256-409`)!
    assert.equal(mock.requests.length, reqsBefore + 3);
    assert.equal(mock.requests[reqsBefore + 2].rangeHeader, `bytes=256-${fileSize - 1}`);

    // Full-file re-read [0..410) is now 100% RAM LRU hits -> ZERO new HTTP requests!
    const fullBuf = new Uint8Array(fileSize);
    const fullGot = await client.readAsync(h, 0n, fullBuf, fileSize);
    assert.equal(fullGot, fileSize);
    assert.deepEqual(fullBuf, movieRpf);
    assert.equal(mock.requests.length, reqsBefore + 3);

    assert.equal(await client.closeAsync(h), true);
  } finally {
    await controller.detachMailbox();
  }
});

test('multi-chunk reads handle server-truncated HTTP 206 responses and real HTTP server seamlessly', async () => {
  const fileBytes = makePatternBytes(300, 123);

  // Part A: Server caps every 206 response to at most 64 bytes even when asked for 192 bytes.
  const truncMock = createMockRangeFetch(new Map([['/game/script.rpf', fileBytes]]), {
    maxServeBytes: 64,
  });
  const cache = new ImmutableHttpfsCache({
    chunkSize: 64,
    maxBytes: 4096,
    fetchImpl: truncMock.fetchImpl,
  });

  const slice = await cache.read('/game/script.rpf', 30, 210);
  assert.equal(slice.byteLength, 210);
  assert.deepEqual(slice, fileBytes.subarray(30, 240));

  // Subsequent overlapping read must hit cache with ZERO additional fetches.
  const callsAfter = truncMock.requests.length;
  const repeat = await cache.read('/game/script.rpf', 64, 128);
  assert.deepEqual(repeat, fileBytes.subarray(64, 192));
  assert.equal(truncMock.requests.length, callsAfter);

  // Part B: End-to-end against real ephemeral Node HTTP 206 Range server.
  const realBytes = makePatternBytes(150 * 1024, 55);
  const server = await startRangeHttpServer(new Map([['/game/audio.rpf', realBytes]]));
  const controller = new IoWorkerController({ opfsDir: null });
  const sab = createHttpfsMailboxBuffer(64 * 1024);
  const client = new HttpfsMailboxClient(sab);

  try {
    await controller.init({
      dataBase: `${server.origin}/data/`,
      bootBase: `${server.origin}/boot/`,
      gameBase: `${server.origin}/game/`,
    });
    await controller.attachMailbox(sab);

    const h = await client.openAsync('G:\\audio.rpf');
    assert.ok(h >= 1);
    // Cross chunk 0 / chunk 1 / chunk 2 boundary: offset 60 KiB, length 80 KiB (spans 2 mailbox round-trips)
    const crossBuf = new Uint8Array(80 * 1024);
    const got = await client.readAsync(h, BigInt(60 * 1024), crossBuf, 80 * 1024);
    assert.equal(got, 80 * 1024);
    assert.deepEqual(crossBuf, realBytes.subarray(60 * 1024, 140 * 1024));

    // Re-opening & re-reading across the same boundary hits RAM LRU with ZERO new HTTP requests.
    const serverReqsBefore = server.requests.length;
    const hAgain = await client.openAsync('/game/audio.rpf');
    assert.ok(hAgain >= 1);
    const repeatBuf = new Uint8Array(80 * 1024);
    const repeatGot = await client.readAsync(hAgain, BigInt(60 * 1024), repeatBuf, 80 * 1024);
    assert.equal(repeatGot, 80 * 1024);
    assert.deepEqual(repeatBuf, realBytes.subarray(60 * 1024, 140 * 1024));
    assert.equal(server.requests.length, serverReqsBefore);

    assert.equal(await client.closeAsync(h), true);
    assert.equal(await client.closeAsync(hAgain), true);
  } finally {
    await controller.detachMailbox();
    await server.close();
  }
});

// ---------------------------------------------------------------------------
// 4. Bounded LRU eviction enforces maxBytes cap cleanly without memory leaks
// ---------------------------------------------------------------------------

test('LRU eviction strictly enforces maxBytes cap while preserving zero-network metadata open hits', async () => {
  const fileBytes = makePatternBytes(512, 77); // 8 chunks of 64 bytes
  const mock = createMockRangeFetch(new Map([['/game/huge.rpf', fileBytes]]));

  // Budget fits strictly 2 chunks = 128 bytes.
  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    immutableCacheChunkSize: 64,
    immutableCacheMaxBytes: 128,
    opfsDir: null,
  });

  await controller.init({ dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' });

  const h = await controller.openHandle('G:\\huge.rpf');
  assert.ok(h >= 1);
  // Chunk 0 is now cached: LRU = [0], bytesCached = 64.
  assert.equal(controller.getStats().immutableCache.bytesCached, 64);
  assert.equal(controller.getStats().immutableCache.entries, 1);

  // Read chunk 1 -> LRU = [0, 1], bytesCached = 128.
  assert.deepEqual(await controller.readHandle(h, 64, 64), fileBytes.subarray(64, 128));
  assert.equal(controller.getStats().immutableCache.bytesCached, 128);
  assert.equal(controller.getStats().immutableCache.entries, 2);
  assert.equal(controller.getStats().immutableCache.evictions, 0);

  // Touch chunk 0 -> LRU becomes [1, 0].
  assert.deepEqual(await controller.readHandle(h, 8, 16), fileBytes.subarray(8, 24));

  // Read chunk 2 -> must evict chunk 1 (oldest), keeping chunk 0! LRU = [0, 2], bytesCached = 128.
  assert.deepEqual(await controller.readHandle(h, 128, 64), fileBytes.subarray(128, 192));
  let stats = controller.getStats().immutableCache;
  assert.equal(stats.evictions, 1);
  assert.equal(stats.entries, 2);
  assert.equal(stats.bytesCached, 128);

  const reqsBeforeVerify = mock.requests.length;
  // Chunk 0 must STILL be cached (0 fetches).
  assert.deepEqual(await controller.readHandle(h, 0, 32), fileBytes.subarray(0, 32));
  assert.equal(mock.requests.length, reqsBeforeVerify);

  // Chunk 1 was evicted and must re-fetch (evicting chunk 2).
  assert.deepEqual(await controller.readHandle(h, 64, 32), fileBytes.subarray(64, 96));
  assert.equal(mock.requests.length, reqsBeforeVerify + 1);
  assert.equal(controller.getStats().immutableCache.evictions, 2);
  assert.ok(controller.getStats().immutableCache.bytesCached <= 128);

  // Now read chunks 4, 5, 6, 7 sequentially to evict chunk 0 completely.
  for (let c = 4; c < 8; c += 1) {
    const out = await controller.readHandle(h, c * 64, 64);
    assert.deepEqual(out, fileBytes.subarray(c * 64, (c + 1) * 64));
    stats = controller.getStats().immutableCache;
    assert.ok(stats.bytesCached <= 128, `bytesCached ${stats.bytesCached} exceeded cap 128`);
    assert.ok(stats.entries <= 2, `entries ${stats.entries} exceeded 2 chunks`);
  }

  // Even though chunk 0 was evicted from RAM LRU, re-opening G:\huge.rpf STILL hits file metadata with ZERO network calls!
  const reqsBeforeReopen = mock.requests.length;
  const h2 = await controller.openHandle('G:\\huge.rpf');
  assert.ok(h2 >= 1);
  assert.equal(controller.getHandleInfo(h2).size, 512);
  assert.equal(mock.requests.length, reqsBeforeReopen);

  // Verify maxBytes=0 mode never caches anything yet returns exact bytes cleanly.
  const zeroCache = new ImmutableHttpfsCache({
    chunkSize: 64,
    maxBytes: 0,
    fetchImpl: mock.fetchImpl,
  });
  const zeroOut = await zeroCache.read('/game/huge.rpf', 10, 100);
  assert.deepEqual(zeroOut, fileBytes.subarray(10, 110));
  assert.equal(zeroCache.getStats().bytesCached, 0);
  assert.equal(zeroCache.getStats().entries, 0);
});

// ---------------------------------------------------------------------------
// 5. 404 / 403 negative caching, TTL expiry, bounded entries, 503 non-caching, 0-byte 416
// ---------------------------------------------------------------------------

test('404 missing optional DLLs & RPF files return ENOENT cleanly, use bounded negative cache, and never corrupt cache state', async () => {
  const validDll = makePatternBytes(128, 33);
  const files = new Map([['/game/present.dll', validDll]]);
  let transientAttempts = 0;
  let clockMs = 10_000;

  const mock = createMockRangeFetch(files, {
    statusOverrides: new Map([
      ['/game/forbidden.dll', 403],
      [
        '/game/flaky.rpf',
        () => {
          transientAttempts += 1;
          return transientAttempts === 1 ? 503 : null;
        },
      ],
    ]),
  });
  files.set('/game/flaky.rpf', makePatternBytes(96, 88));

  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    immutableCacheChunkSize: 64,
    immutableCacheMaxBytes: 1024,
    negativeCacheTtlMs: 5000,
    negativeCacheMaxEntries: 3,
    now: () => clockMs,
    opfsDir: null,
  });

  const sab = createHttpfsMailboxBuffer(256);
  const client = new HttpfsMailboxClient(sab);

  try {
    await controller.init({ dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' });
    await controller.attachMailbox(sab);

    const reqsAfterInit = mock.requests.length;

    // 1. First probe of missing optional DLL `xgameruntime.dll` -> HTTP 404 -> -2 (ENOENT).
    const first404 = await rawMailboxOpen(sab, 'G:\\xgameruntime.dll');
    assert.equal(first404.status, HttpfsStatus.Error);
    assert.equal(first404.result, -2n);
    assert.equal(mock.requests.length, reqsAfterInit + 1);

    // 2. Next 10 probes of `xgameruntime.dll` (different path spellings) -> negative cache hits, ZERO fetches!
    for (let i = 0; i < 10; i += 1) {
      const spelling = i % 2 === 0 ? 'G:\\xgameruntime.dll' : '/game/xgameruntime.dll';
      const hit = await rawMailboxOpen(sab, spelling);
      assert.equal(hit.status, HttpfsStatus.Error);
      assert.equal(hit.result, -2n);
    }
    assert.equal(mock.requests.length, reqsAfterInit + 1);
    assert.equal(controller.getStats().immutableCache.negativeHits, 10);
    assert.equal(controller.getStats().immutableCache.entries, 0);
    assert.equal(controller.getStats().immutableCache.filesIndexed, 0);

    // 3. HTTP 403 maps to -13 (EACCES) and is also negative-cached cleanly.
    const first403 = await rawMailboxOpen(sab, 'G:\\forbidden.dll');
    assert.equal(first403.status, HttpfsStatus.Error);
    assert.equal(first403.result, -13n);
    const second403 = await rawMailboxOpen(sab, 'G:\\forbidden.dll');
    assert.equal(second403.status, HttpfsStatus.Error);
    assert.equal(second403.result, -13n);
    assert.equal(mock.requests.length, reqsAfterInit + 2);

    // 4. Advance clock past TTL (5000ms) -> xgameruntime.dll expires and re-probes; if file now exists, succeeds!
    clockMs += 5001;
    const appearedBytes = makePatternBytes(80, 19);
    files.set('/game/xgameruntime.dll', appearedBytes);
    const hAppeared = await client.openAsync('G:\\xgameruntime.dll');
    assert.ok(hAppeared >= 1);
    const outAppeared = new Uint8Array(80);
    assert.equal(await client.readAsync(hAppeared, 0n, outAppeared, 80), 80);
    assert.deepEqual(outAppeared, appearedBytes);
    assert.equal(await client.closeAsync(hAppeared), true);

    // 5. Transient 503 error is NOT negative-cached; retry immediately succeeds!
    const flakyFail = await rawMailboxOpen(sab, 'G:\\flaky.rpf');
    assert.equal(flakyFail.status, HttpfsStatus.Error);
    assert.equal(flakyFail.result, -5n);
    const hFlakyOk = await client.openAsync('G:\\flaky.rpf');
    assert.ok(hFlakyOk >= 1);
    assert.equal(await client.closeAsync(hFlakyOk), true);

    // 6. Negative cache size stays strictly capped at negativeCacheMaxEntries (3).
    for (let i = 0; i < 8; i += 1) {
      const res = await rawMailboxOpen(sab, `G:\\missing_${i}.dll`);
      assert.equal(res.status, HttpfsStatus.Error);
      assert.equal(res.result, -2n);
      assert.ok(
        controller.getStats().immutableCache.negativeEntries <= 3,
        'negativeEntries must respect negativeCacheMaxEntries cap',
      );
    }

    // 7. Existing valid files remain completely unaffected and healthy.
    const hPresent = await client.openAsync('G:\\present.dll');
    assert.ok(hPresent >= 1);
    const presentBuf = new Uint8Array(128);
    assert.equal(await client.readAsync(hPresent, 0n, presentBuf, 128), 128);
    assert.deepEqual(presentBuf, validDll);
    assert.equal(await client.closeAsync(hPresent), true);
  } finally {
    await controller.detachMailbox();
  }
});

test('handles 0-byte immutable files (HTTP 416 bytes */0) and HTTP 200 fallback cleanly', async () => {
  const emptyRpf = new Uint8Array(0);
  const full200Config = utf8Encoder.encode('<Game configVersion="1"/>');

  const emptyMock = createMockRangeFetch(new Map([['/game/empty.rpf', emptyRpf]]));
  const c1 = new IoWorkerController({
    fetchImpl: emptyMock.fetchImpl,
    immutableCacheChunkSize: 64,
    opfsDir: null,
  });
  await c1.init({ dataBase: '/data/', gameBase: '/game/' });

  const hEmpty = await c1.openHandle('G:\\empty.rpf');
  assert.ok(hEmpty >= 1);
  assert.equal(c1.getHandleInfo(hEmpty).size, 0);
  assert.equal((await c1.readHandle(hEmpty, 0, 64)).byteLength, 0);
  // Repeat open of 0-byte file hits metadata cache with zero new HTTP requests.
  const emptyReqs = emptyMock.requests.length;
  const hEmpty2 = await c1.openHandle('G:\\empty.rpf');
  assert.ok(hEmpty2 >= 1);
  assert.equal(emptyMock.requests.length, emptyReqs);

  // HTTP 200 OK fallback (server ignores Range header and returns full file)
  const fullMock = createMockRangeFetch(new Map([['/game/MicrosoftGame.Config', full200Config]]), {
    force200: true,
  });
  const c2 = new IoWorkerController({
    fetchImpl: fullMock.fetchImpl,
    immutableCacheChunkSize: 64,
    opfsDir: null,
  });
  await c2.init({ dataBase: '/data/', gameBase: '/game/' });

  const hCfg = await c2.openHandle('MicrosoftGame.Config');
  assert.ok(hCfg >= 1);
  assert.equal(c2.getHandleInfo(hCfg).size, full200Config.byteLength);
  const outCfg = await c2.readHandle(hCfg, 0, 64);
  assert.deepEqual(outCfg, full200Config);
});

// ---------------------------------------------------------------------------
// 6. Concurrent in-flight deduplication + classic Web Worker importScripts check
// ---------------------------------------------------------------------------

test('concurrent open/read calls for the same uncached .rpf coalesce into a single HTTP 206 fetch', async () => {
  const fileBytes = makePatternBytes(128 * 1024, 61);
  const mock = createMockRangeFetch(new Map([['/game/shared.rpf', fileBytes]]), { delayMs: 15 });

  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    opfsDir: null,
  });
  await controller.init({ dataBase: '/data/', gameBase: '/game/' });
  const reqsAfterInit = mock.requests.length;

  // Fire 6 simultaneous opens for the same uncached .rpf!
  const handles = await Promise.all([
    controller.openHandle('G:\\shared.rpf'),
    controller.openHandle('/game/shared.rpf'),
    controller.openHandle('shared.rpf'),
    controller.openHandle('\\\\?\\G:\\shared.rpf'),
    controller.openHandle('G:\\shared.rpf'),
    controller.openHandle('G:\\shared.rpf'),
  ]);

  for (const h of handles) {
    assert.ok(h >= 1);
    assert.equal(controller.getHandleInfo(h).size, fileBytes.byteLength);
  }
  assert.equal(
    mock.requests.length - reqsAfterInit,
    1,
    'Concurrent opens for the same URL must coalesce into 1 HTTP 206 fetch',
  );

  // Fire 4 simultaneous reads into [65536..131072): already covered by slab 0 from open() -> ZERO extra fetches!
  const results = await Promise.all([
    controller.readHandle(handles[0], 65536, 4096),
    controller.readHandle(handles[1], 66000, 8192),
    controller.readHandle(handles[2], 70000, 16384),
    controller.readHandle(handles[3], 65536, 65536),
  ]);

  assert.deepEqual(results[0], fileBytes.subarray(65536, 65536 + 4096));
  assert.deepEqual(results[1], fileBytes.subarray(66000, 66000 + 8192));
  assert.deepEqual(results[2], fileBytes.subarray(70000, 70000 + 16384));
  assert.deepEqual(results[3], fileBytes.subarray(65536, 65536 + 65536));

  assert.equal(
    mock.requests.length - reqsAfterInit,
    1,
    'Expected strictly 1 HTTP 206 slab-0 fetch covering both open and [65536..131072) reads',
  );
  assert.ok(controller.getStats().immutableCache.inFlightCoalesced >= 5);
});

test('classic Web Worker (importScripts) exposes ImmutableHttpfsCache and reports immutableCache stats', async () => {
  const rpfData = makePatternBytes(96 * 1024, 15);
  const mock = createMockRangeFetch(new Map([['/game/audio.rpf', rpfData]]));
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
    Promise,
    Error,
    Number,
    Math,
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

  assert.equal(typeof sandbox.PlayGta6IoWorker.ImmutableHttpfsCache, 'function');

  await workerSelf.onmessage({
    data: { init: true, dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' },
  });
  const sab = createHttpfsMailboxBuffer(64 * 1024);
  await workerSelf.onmessage({ data: { attachMailbox: true, id: 1, sab } });

  const client = new HttpfsMailboxClient(sab);
  try {
    const reqsBefore = mock.requests.length;
    const h = await client.openAsync('G:\\audio.rpf');
    assert.ok(h >= 1);
    const slice = new Uint8Array(8192);
    assert.equal(await client.readAsync(h, 4096n, slice, 8192), 8192);
    assert.deepEqual(slice, rpfData.subarray(4096, 4096 + 8192));
    assert.equal(mock.requests.length, reqsBefore + 1);
    assert.equal(await client.closeAsync(h), true);

    await workerSelf.onmessage({ data: { stats: true, id: 2 } });
    const statsMsg = posted.find((m) => m.id === 2);
    assert.ok(statsMsg && statsMsg.stats && statsMsg.stats.immutableCache);
    assert.equal(statsMsg.stats.immutableCache.networkFetches, 1);
    assert.equal(statsMsg.stats.immutableCache.slabCacheHits, 1);
    assert.equal(statsMsg.stats.immutableCache.hits, 0);
    assert.equal(statsMsg.stats.immutableCache.filesIndexed, 1);
  } finally {
    await workerSelf.onmessage({ data: { detachMailbox: true, id: 3 } });
  }
});

// ---------------------------------------------------------------------------
// 7. W91: shaders/ HTTPFS mount routing, manifest bypass, custom shadersBase,
//    HTTP 206 range reads, traversal rejection, and worker init forwarding
// ---------------------------------------------------------------------------

test('W91: resolveHttpfsRoute routes shaders/... to shadersBase and blocks traversal', () => {
  const bases = {
    dataBase: '/data/',
    bootBase: '/boot/',
    gameBase: '/game/',
    shadersBase: '/shaders/',
  };

  assert.deepEqual(resolveHttpfsRoute('shaders/splash.wgsl.pack', bases), {
    ok: true,
    isRpf: false,
    mount: 'shaders',
    base: '/shaders/',
    relPath: 'splash.wgsl.pack',
    url: '/shaders/splash.wgsl.pack',
  });

  assert.deepEqual(resolveHttpfsRoute('shaders/dump_pipelines.flag', bases), {
    ok: true,
    isRpf: false,
    mount: 'shaders',
    base: '/shaders/',
    relPath: 'dump_pipelines.flag',
    url: '/shaders/dump_pipelines.flag',
  });

  // Leading slash, ./ prefix, and case-insensitive mount prefix
  assert.deepEqual(resolveHttpfsRoute('/shaders/sub/frame1.wgsl.pack', bases), {
    ok: true,
    isRpf: false,
    mount: 'shaders',
    base: '/shaders/',
    relPath: 'sub/frame1.wgsl.pack',
    url: '/shaders/sub/frame1.wgsl.pack',
  });
  assert.deepEqual(resolveHttpfsRoute('./SHADERS/splash.wgsl.pack', bases), {
    ok: true,
    isRpf: false,
    mount: 'shaders',
    base: '/shaders/',
    relPath: 'splash.wgsl.pack',
    url: '/shaders/splash.wgsl.pack',
  });

  // Xbox drive-prefixed paths must NOT hijack into shadersBase (stay on gameBase)
  assert.deepEqual(resolveHttpfsRoute('G:\\shaders\\cache.bin', bases), {
    ok: true,
    isRpf: false,
    mount: 'game',
    base: '/game/',
    relPath: 'shaders/cache.bin',
    url: '/game/shaders/cache.bin',
  });

  // Explicit null/false disables the shaders/ mount
  assert.deepEqual(
    resolveHttpfsRoute('shaders/splash.wgsl.pack', { ...bases, shadersBase: null }),
    {
      ok: true,
      isRpf: false,
      mount: 'data',
      base: '/data/',
      relPath: 'shaders/splash.wgsl.pack',
      url: '/data/shaders/splash.wgsl.pack',
    },
  );
  assert.deepEqual(
    resolveHttpfsRoute('shaders/splash.wgsl.pack', { ...bases, shadersBase: false }),
    {
      ok: true,
      isRpf: false,
      mount: 'data',
      base: '/data/',
      relPath: 'shaders/splash.wgsl.pack',
      url: '/data/shaders/splash.wgsl.pack',
    },
  );

  // Path traversal and NUL injection under shaders/ remain blocked
  for (const bad of [
    'shaders/../secret',
    'shaders/../secret.pack',
    'shaders/sub/../../secret',
    'shaders/%2e%2e/secret',
    'shaders/splash.wgsl.pack\0',
  ]) {
    assert.deepEqual(
      resolveHttpfsRoute(bad, bases),
      { ok: false, error: 'path traversal blocked' },
      `Expected traversal rejection for ${JSON.stringify(bad)}`,
    );
  }
});

test('W91: shaders/splash.wgsl.pack and shaders/dump_pipelines.flag bypass data-manifest.json and serve HTTP 206 range reads', async () => {
  const dataManifest = {
    abi_version: 1,
    file_count: 1,
    files: [{ path: 'notes.txt', size: 5 }],
  };
  const splashPack = makePatternBytes(2048, 0x5a);
  const dumpFlag = utf8Encoder.encode('1\n');

  const mock = createMockRangeFetch(
    new Map([
      ['/data/data-manifest.json', utf8Encoder.encode(JSON.stringify(dataManifest))],
      ['/data/notes.txt', utf8Encoder.encode('hello')],
      ['/shaders/splash.wgsl.pack', splashPack],
      ['/shaders/dump_pipelines.flag', dumpFlag],
    ]),
  );

  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    chunkSize: 256,
    maxBytes: 16 * 1024,
    opfsDir: null,
  });

  // Default shadersBase is '/shaders/' even when omitted from init()
  const ready = await controller.init({
    dataBase: '/data/',
    bootBase: '/boot/',
    gameBase: '/game/',
  });
  assert.equal(ready.shadersBase, '/shaders/');
  assert.equal(controller.shadersBase, '/shaders/');

  // 1. Open and range-read shaders/splash.wgsl.pack (not listed in data-manifest.json)
  const hPack = await controller.openHandle('shaders/splash.wgsl.pack');
  assert.ok(hPack >= 1, `Expected valid handle for shaders/splash.wgsl.pack, got ${hPack}`);
  const packInfo = controller.getHandleInfo(hPack);
  assert.equal(packInfo.mount, 'shaders');
  assert.equal(packInfo.path, 'splash.wgsl.pack');
  assert.equal(packInfo.url, '/shaders/splash.wgsl.pack');

  const packHead = await controller.readHandle(hPack, 0, 512);
  assert.deepEqual(packHead, splashPack.subarray(0, 512));
  const packMid = await controller.readHandle(hPack, 300, 700);
  assert.deepEqual(packMid, splashPack.subarray(300, 1000));
  assert.equal(controller.closeHandle(hPack), true);

  // 2. Open and read shaders/dump_pipelines.flag (not listed in data-manifest.json)
  const hFlag = await controller.openHandle('shaders/dump_pipelines.flag');
  assert.ok(hFlag >= 1, `Expected valid handle for shaders/dump_pipelines.flag, got ${hFlag}`);
  assert.equal(controller.getHandleInfo(hFlag).mount, 'shaders');
  assert.equal(controller.getHandleInfo(hFlag).url, '/shaders/dump_pipelines.flag');
  const flagBytes = await controller.readHandle(hFlag, 0, 16);
  assert.deepEqual(flagBytes, dumpFlag);
  assert.equal(controller.closeHandle(hFlag), true);

  // Verify HTTP 206 range requests went directly to /shaders/*
  const rangeUrls = mock.requests.filter((r) => r.rangeHeader).map((r) => r.url);
  assert.ok(rangeUrls.includes('/shaders/splash.wgsl.pack'));
  assert.ok(rangeUrls.includes('/shaders/dump_pipelines.flag'));
  assert.equal(
    rangeUrls.some((u) => u.includes('/data/shaders/')),
    false,
    'Must not fall through to /data/shaders/',
  );

  // 3. Missing file under /shaders/ returns -2 (ENOENT) via HTTP 404
  const hMissing = await controller.openHandle('shaders/nonexistent.wgsl.pack');
  assert.equal(hMissing, -2);

  // 4. Path traversal under shaders/ is rejected with -22 (EINVAL) and zero outbound fetches
  const reqsBeforeTraversal = mock.requests.length;
  assert.equal(await controller.openHandle('shaders/../secret'), -22);
  assert.equal(await controller.openHandle('shaders/%2e%2e/secret'), -22);
  assert.equal(mock.requests.length, reqsBeforeTraversal);
});

test('W91: custom shadersBase in init() overrides mount route and null disables it', async () => {
  const dataManifest = {
    abi_version: 1,
    file_count: 0,
    files: [],
  };
  const fooPack = makePatternBytes(512, 0x33);

  const mock = createMockRangeFetch(
    new Map([
      ['/data/data-manifest.json', utf8Encoder.encode(JSON.stringify(dataManifest))],
      ['/custom-shaders/foo.pack', fooPack],
    ]),
  );

  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    chunkSize: 128,
    maxBytes: 8192,
    opfsDir: null,
  });

  // Trailing slash omitted -> normalized to '/custom-shaders/'
  const ready = await controller.init({
    dataBase: '/data/',
    shadersBase: '/custom-shaders',
  });
  assert.equal(ready.shadersBase, '/custom-shaders/');
  assert.equal(controller.shadersBase, '/custom-shaders/');

  const h = await controller.openHandle('shaders/foo.pack');
  assert.ok(h >= 1, `Expected valid handle for shaders/foo.pack, got ${h}`);
  assert.equal(controller.getHandleInfo(h).mount, 'shaders');
  assert.equal(controller.getHandleInfo(h).url, '/custom-shaders/foo.pack');

  const out = await controller.readHandle(h, 64, 192);
  assert.deepEqual(out, fooPack.subarray(64, 256));
  assert.equal(controller.closeHandle(h), true);

  const rangeUrls = mock.requests.filter((r) => r.rangeHeader).map((r) => r.url);
  assert.ok(rangeUrls.includes('/custom-shaders/foo.pack'));

  // Explicit shadersBase: null disables the shaders/ mount -> falls back to data-manifest.json (-2)
  const disabledReady = await controller.init({
    dataBase: '/data/',
    shadersBase: null,
  });
  assert.equal(disabledReady.shadersBase, null);
  assert.equal(controller.shadersBase, null);
  assert.equal(await controller.openHandle('shaders/foo.pack'), -2);
});

test('W91: classic Web Worker init message forwards custom shadersBase over self.onmessage', async () => {
  const packData = makePatternBytes(384, 0x71);
  const mock = createMockRangeFetch(new Map([['/alt-shaders/splash.wgsl.pack', packData]]));
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
    Promise,
    Error,
    Number,
    Math,
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
    data: { init: true, dataBase: '/data/', shadersBase: '/alt-shaders/' },
  });
  assert.equal(posted[0].ready, true);
  assert.equal(posted[0].shadersBase, '/alt-shaders/');

  const sab = createHttpfsMailboxBuffer(64 * 1024);
  await workerSelf.onmessage({ data: { attachMailbox: true, id: 1, sab } });

  const client = new HttpfsMailboxClient(sab);
  try {
    const h = await client.openAsync('shaders/splash.wgsl.pack');
    assert.ok(h >= 1, `Expected valid mailbox handle, got ${h}`);
    const out = new Uint8Array(packData.byteLength);
    const n = await client.readAsync(h, 0n, out, packData.byteLength);
    assert.equal(n, packData.byteLength);
    assert.deepEqual(out, packData);
    assert.equal(await client.closeAsync(h), true);
  } finally {
    await workerSelf.onmessage({ data: { detachMailbox: true, id: 2 } });
  }
});

// ---------------------------------------------------------------------------
// 8. W173 (#880): Opt-in /pg6_dump/* write forwarding, strict read-only game
//    file enforcement, bounded sequential slab prefetch & 4K Bink2 cache tuning
// ---------------------------------------------------------------------------

function createMockDumpAndRangeFetch(urlBytesMap, options = {}) {
  const base = createMockRangeFetch(urlBytesMap, options);
  const posts = [];
  const dumpedFiles = new Map();
  const postStatusOverrides = options.postStatusOverrides || new Map();

  async function fetchImpl(url, init = {}) {
    const method = ((init && init.method) || 'GET').toUpperCase();
    if (method === 'POST') {
      const rawBody = init.body;
      const bodyBytes =
        rawBody instanceof Uint8Array
          ? rawBody.slice()
          : rawBody instanceof ArrayBuffer
            ? new Uint8Array(rawBody).slice()
            : typeof rawBody === 'string'
              ? utf8Encoder.encode(rawBody)
              : new Uint8Array(0);
      const parsedUrl = new URL(url, 'http://127.0.0.1');
      const pathname = parsedUrl.pathname;
      const append = parsedUrl.searchParams.get('append') === '1';
      posts.push({ method, url, pathname, append, bytes: bodyBytes });

      if (options.delayMs > 0) {
        await new Promise((r) => setTimeout(r, options.delayMs));
      }

      const override = postStatusOverrides.get(pathname);
      const overrideStatus = typeof override === 'function' ? override(posts.length) : override;
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

      if (append && dumpedFiles.has(pathname)) {
        const prev = dumpedFiles.get(pathname);
        const merged = new Uint8Array(prev.byteLength + bodyBytes.byteLength);
        merged.set(prev, 0);
        merged.set(bodyBytes, prev.byteLength);
        dumpedFiles.set(pathname, merged);
      } else {
        dumpedFiles.set(pathname, bodyBytes);
      }

      return {
        ok: true,
        status: 200,
        headers: new Map(),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }
    return base.fetchImpl(url, init);
  }

  return { fetchImpl, requests: base.requests, posts, dumpedFiles };
}

async function rawMailboxOp(sab, op, handle, offsetBig, payloadBytes) {
  const statusView = new Int32Array(sab, 0, 1);
  const headerView = new DataView(sab, 0, 40);
  const payloadView = new Uint8Array(sab, 40, sab.byteLength - 40);

  const bytes = payloadBytes instanceof Uint8Array ? payloadBytes : new Uint8Array(0);
  headerView.setUint32(HttpfsOffsets.OP, op, true);
  headerView.setInt32(HttpfsOffsets.HANDLE, handle, true);
  headerView.setBigUint64(HttpfsOffsets.OFFSET, offsetBig, true);
  headerView.setBigUint64(HttpfsOffsets.LENGTH, BigInt(bytes.byteLength), true);
  headerView.setBigInt64(HttpfsOffsets.RESULT, -1n, true);
  if (bytes.byteLength > 0) {
    payloadView.set(bytes, 0);
  }

  Atomics.store(statusView, 0, HttpfsStatus.RequestReady);
  Atomics.notify(statusView, 0, 1);

  let cur = Atomics.load(statusView, 0);
  while (cur === HttpfsStatus.RequestReady) {
    const waitObj = Atomics.waitAsync(statusView, 0, HttpfsStatus.RequestReady, 5000);
    const res = waitObj.async ? await waitObj.value : waitObj.value;
    cur = Atomics.load(statusView, 0);
    if (res === 'timed-out' && cur === HttpfsStatus.RequestReady) {
      throw new Error(`Mailbox hung during op=${op}`);
    }
  }

  const result = headerView.getBigInt64(HttpfsOffsets.RESULT, true);
  Atomics.store(statusView, 0, HttpfsStatus.Idle);
  Atomics.notify(statusView, 0, 1);
  return { status: cur, result };
}

test('W173: dump writes are strictly disabled by default and reject /pg6_dump/* with zero outbound requests', async () => {
  assert.equal(DUMP_VIRTUAL_PREFIX, '/pg6_dump/');
  assert.equal(DEFAULT_DUMP_ENDPOINT, '/__pg6_dump/');

  assert.deepEqual(resolveDumpConfig(), { allowDumpWrites: false, dumpEndpoint: null });
  assert.deepEqual(resolveDumpConfig({ allowDumpWrites: false, dumpEndpoint: '/__pg6_dump/' }), {
    allowDumpWrites: false,
    dumpEndpoint: null,
  });
  assert.deepEqual(resolveDumpConfig({ allowDumpWrites: true }), {
    allowDumpWrites: true,
    dumpEndpoint: '/__pg6_dump/',
  });
  assert.deepEqual(resolveDumpConfig({ dumpEndpoint: '/__pg6_dump' }), {
    allowDumpWrites: true,
    dumpEndpoint: '/__pg6_dump/',
  });

  const disabledRoute = resolveDumpWriteRoute('/pg6_dump/pso_001.dxil');
  assert.equal(disabledRoute.ok, false);
  assert.equal(disabledRoute.errno, -30);
  assert.equal(disabledRoute.code, 'DUMP_WRITES_DISABLED');

  const mock = createMockDumpAndRangeFetch(new Map());
  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    opfsDir: null,
  });
  const ready = await controller.init({ dataBase: '/data/', gameBase: '/game/' });
  assert.equal(ready.allowDumpWrites, false);
  assert.equal(ready.dumpEndpoint, null);

  // Disabled by default: openWriteHandle returns -30 (EROFS) and writeDump throws errno -30
  assert.equal(await controller.openWriteHandle('/pg6_dump/pso_001.dxil'), -30);
  assert.equal(await controller.openHandle('/pg6_dump/pso_001.dxil', { mode: 'wb' }), -30);
  await assert.rejects(
    () => controller.writeDump('/pg6_dump/pso_001.dxil', new Uint8Array([1, 2, 3])),
    (err) => err && err.errno === -30,
  );

  // Zero outbound POSTs ever occurred
  assert.equal(mock.posts.length, 0);

  const stats = controller.getStats();
  assert.equal(stats.dumpWritesCompleted, 0);
  assert.equal(stats.dumpBytesWritten, 0);
  assert.equal(stats.dumpWriteErrors, 3);
  assert.equal(stats.dumpWrites.enabled, false);
});

test('W173: strict /pg6_dump/* normalization blocks path traversal and enforces read-only on all game/boot/shader paths', async () => {
  const dumpOpts = { allowDumpWrites: true, dumpEndpoint: '/__pg6_dump/' };

  // Valid /pg6_dump/* virtual paths normalize cleanly
  assert.equal(isDumpVirtualPath('/pg6_dump/pso_01.dxil'), true);
  assert.equal(isDumpVirtualPath('pg6_dump/nested/sub/shader.bin'), true);
  assert.deepEqual(resolveDumpWriteRoute('/pg6_dump/dxil/pso_001.dxil', dumpOpts), {
    ok: true,
    mount: 'dump',
    base: '/__pg6_dump/',
    relPath: 'dxil/pso_001.dxil',
    encodedRelPath: 'dxil/pso_001.dxil',
    url: '/__pg6_dump/dxil/pso_001.dxil',
    appendUrl: '/__pg6_dump/dxil/pso_001.dxil?append=1',
  });

  // Path traversal, NUL bytes, malformed percent-escapes, and empty filenames are strictly rejected (-22)
  for (const badPath of [
    '',
    '/pg6_dump',
    '/pg6_dump/',
    '/pg6_dump/.',
    '/pg6_dump/../secret.bin',
    '/pg6_dump/sub/../../secret.bin',
    '/pg6_dump/sub\\..\\secret.bin',
    '/pg6_dump/%2e%2e/secret.bin',
    '/pg6_dump/a%2f..%2fb.bin',
    '/pg6_dump/a%5c..%5cb.bin',
    '/pg6_dump/file.dxil\0',
    '/pg6_dump/file%00.dxil',
    '/pg6_dump/bad%ZZ.dxil',
    '/pg6_dump/bad?query=1',
    '/pg6_dump/bad*glob.dxil',
  ]) {
    const res = resolveDumpWriteRoute(badPath, dumpOpts);
    assert.equal(res.ok, false, `Expected rejection for ${JSON.stringify(badPath)}`);
    assert.equal(res.errno, -22, `Expected EINVAL (-22) for ${JSON.stringify(badPath)}`);
  }

  // Every path outside /pg6_dump/* is strictly read-only (-30 EROFS)
  for (const readOnlyPath of [
    'G:\\audio.rpf',
    '\\\\?\\G:\\x64\\movies\\intro.bk2',
    '/game/audio.rpf',
    'boot/main/image/guest.pgim',
    'shaders/splash.wgsl.pack',
    'data-manifest.json',
    'rpf://appdata0.rpf/config.xml',
    '/pg6_dump_other/escape.bin',
    'C:\\pg6_dump\\escape.bin',
    'http://127.0.0.1/pg6_dump/escape.bin',
  ]) {
    assert.equal(isDumpVirtualPath(readOnlyPath), false);
    const res = resolveDumpWriteRoute(readOnlyPath, dumpOpts);
    assert.equal(res.ok, false, `Expected read-only refusal for ${readOnlyPath}`);
    assert.equal(res.errno, -30, `Expected EROFS (-30) for ${readOnlyPath}`);
  }

  // Controller-level read-only enforcement even when dump writes are enabled
  const audioRpf = makePatternBytes(4096, 17);
  const mock = createMockDumpAndRangeFetch(new Map([['/game/audio.rpf', audioRpf]]));
  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    opfsDir: null,
  });
  await controller.init({
    dataBase: '/data/',
    bootBase: '/boot/',
    gameBase: '/game/',
    allowDumpWrites: true,
    dumpEndpoint: '/__pg6_dump/',
  });

  // Opening a game file for read works normally, while write attempts on it or its handle fail cleanly
  const readHandle = await controller.openHandle('G:\\audio.rpf');
  assert.ok(readHandle >= 1);
  assert.deepEqual(await controller.readHandle(readHandle, 0, 256), audioRpf.subarray(0, 256));

  assert.equal(await controller.openWriteHandle('G:\\audio.rpf'), -30);
  assert.equal(await controller.openWriteHandle('boot/main/image/guest.pgim'), -30);
  assert.equal(await controller.openWriteHandle('/pg6_dump/../audio.rpf'), -22);

  await assert.rejects(
    () => controller.writeHandle(readHandle, new Uint8Array([0xde, 0xad])),
    (err) => err && err.errno === -30,
  );
  await assert.rejects(
    () => controller.flushHandle(readHandle),
    (err) => err && err.errno === -30,
  );
  await assert.rejects(
    () => controller.writeDump('G:\\audio.rpf', new Uint8Array([1, 2, 3])),
    (err) => err && err.errno === -30,
  );

  // Zero POST requests went out, and game reads remain completely intact
  assert.equal(mock.posts.length, 0);
  assert.deepEqual(await controller.readHandle(readHandle, 256, 256), audioRpf.subarray(256, 512));
  assert.equal(controller.closeHandle(readHandle), true);
  assert.equal(controller.getStats().dumpWriteErrors, 6);
});

test('W173: opt-in /pg6_dump/* write, sequential ?append=1 chunking, buffered flush, and error telemetry', async () => {
  const mock = createMockDumpAndRangeFetch(new Map(), {
    postStatusOverrides: new Map([['/__pg6_dump/fail_413.dxil', 413]]),
  });

  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    dumpFlushChunkBytes: 512,
    dumpMaxBufferBytes: 2048,
    opfsDir: null,
  });

  const ready = await controller.init({
    dataBase: '/data/',
    gameBase: '/game/',
    dumpEndpoint: '/__pg6_dump',
  });
  assert.equal(ready.allowDumpWrites, true);
  assert.equal(ready.dumpEndpoint, '/__pg6_dump/');

  // 1. Streaming handle: first write POSTs without ?append=1, subsequent writes POST with ?append=1
  const h = await controller.openWriteHandle('/pg6_dump/pipelines/pso_001.dxil');
  assert.ok(h >= 1);
  assert.equal(controller.getHandleInfo(h).kind, 'dump_write');
  assert.equal(controller.getHandleInfo(h).writable, true);

  // Reading from a write-only dump handle must fail cleanly
  await assert.rejects(() => controller.readHandle(h, 0, 16), (err) => err && err.errno === -9);

  const c1 = makePatternBytes(300, 11);
  const c2 = makePatternBytes(250, 22);
  const c3 = makePatternBytes(150, 33);

  // Issue concurrent sequential writes on the same handle; internal queue preserves exact order!
  const [r1, r2, r3] = await Promise.all([
    controller.writeHandle(h, c1),
    controller.writeHandle(h, c2),
    controller.writeHandle(h, c3),
  ]);
  assert.equal(r1.append, false);
  assert.equal(r2.append, true);
  assert.equal(r3.append, true);
  assert.equal(await controller.closeDumpHandle(h), true);

  const expectedCombined = new Uint8Array(700);
  expectedCombined.set(c1, 0);
  expectedCombined.set(c2, 300);
  expectedCombined.set(c3, 550);
  assert.deepEqual(mock.dumpedFiles.get('/__pg6_dump/pipelines/pso_001.dxil'), expectedCombined);
  assert.equal(mock.posts[0].url, '/__pg6_dump/pipelines/pso_001.dxil');
  assert.equal(mock.posts[1].url, '/__pg6_dump/pipelines/pso_001.dxil?append=1');
  assert.equal(mock.posts[2].url, '/__pg6_dump/pipelines/pso_001.dxil?append=1');

  // 2. Buffered handle auto-flushes when hitting dumpFlushChunkBytes (512B) and flushes tail on closeDumpHandle
  const hBuf = await controller.openWriteHandle('/pg6_dump/buffered.bin', { buffered: true });
  const b1 = makePatternBytes(200, 44);
  const b2 = makePatternBytes(400, 55);
  const b3 = makePatternBytes(100, 66);

  const q1 = await controller.writeHandle(hBuf, b1);
  assert.equal(q1.buffered, true);
  assert.equal(q1.pendingBytes, 200);

  // Second write pushes pending to 600B >= 512B -> auto-flushes in 512B + 88B chunks!
  const q2 = await controller.writeHandle(hBuf, b2);
  assert.equal(q2.autoFlushed, true);
  assert.equal(controller.getHandleInfo(hBuf).pendingBytes, 0);

  const q3 = await controller.writeHandle(hBuf, b3);
  assert.equal(q3.buffered, true);
  assert.equal(q3.pendingBytes, 100);
  assert.equal(await controller.closeDumpHandle(hBuf), true);

  const expectedBuf = new Uint8Array(700);
  expectedBuf.set(b1, 0);
  expectedBuf.set(b2, 200);
  expectedBuf.set(b3, 600);
  assert.deepEqual(mock.dumpedFiles.get('/__pg6_dump/buffered.bin'), expectedBuf);

  // 3. One-shot writeDump larger than dumpFlushChunkBytes (1100B -> 512 + 512 + 76)
  const bigDump = makePatternBytes(1100, 77);
  const bigRes = await controller.writeDump('/pg6_dump/large_shader.dxil', bigDump);
  assert.equal(bigRes.ok, true);
  assert.equal(bigRes.chunks, 3);
  assert.equal(bigRes.bytesWritten, 1100);
  assert.deepEqual(mock.dumpedFiles.get('/__pg6_dump/large_shader.dxil'), bigDump);

  // 4. Server 413 error maps cleanly to -27 (EFBIG) and increments dumpWriteErrors
  await assert.rejects(
    () => controller.writeDump('/pg6_dump/fail_413.dxil', new Uint8Array([1, 2, 3, 4])),
    (err) => err && err.errno === -27 && err.status === 413,
  );

  const stats = controller.getStats();
  assert.equal(stats.dumpWritesCompleted, 9); // 3 (streaming) + 3 (buffered: 512+88+100) + 3 (large: 512+512+76)
  assert.equal(stats.dumpBytesWritten, 700 + 700 + 1100);
  assert.equal(stats.dumpWriteErrors, 1);
  assert.equal(stats.dumpWrites.enabled, true);
  assert.equal(stats.dumpWrites.openDumpHandles, 0);
  assert.equal(stats.dumpWrites.bufferedBytes, 0);

  // Verify dump telemetry is sanitized (no filenames or URLs leaked)
  const dumpStatsJson = JSON.stringify(stats.dumpWrites);
  assert.equal(dumpStatsJson.includes('pso_001'), false);
  assert.equal(dumpStatsJson.includes('__pg6_dump'), false);
});

test('W173: SAB HttpfsMailbox supports opt-in /pg6_dump/* writes while enforcing read-only on game handles', async () => {
  const gameBytes = makePatternBytes(2048, 99);
  const mock = createMockDumpAndRangeFetch(new Map([['/game/movies.rpf', gameBytes]]));
  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    opfsDir: null,
  });

  const sab = createHttpfsMailboxBuffer(4096);
  try {
    await controller.init({
      dataBase: '/data/',
      gameBase: '/game/',
      allowDumpWrites: true,
      dumpEndpoint: '/__pg6_dump/',
    });
    await controller.attachMailbox(sab);

    // 1. Open /pg6_dump/mailbox_pso.dxil with write flag over SAB mailbox
    const openDump = await rawMailboxOp(
      sab,
      HttpfsOp.Open,
      -1,
      HTTPFS_OPEN_FLAG_WRITE,
      utf8Encoder.encode('/pg6_dump/mailbox_pso.dxil'),
    );
    assert.equal(openDump.status, HttpfsStatus.ResponseReady);
    const dumpHandle = Number(openDump.result);
    assert.ok(dumpHandle >= 1);

    // Write chunk 1 (create) and chunk 2 (append)
    const p1 = makePatternBytes(128, 3);
    const p2 = makePatternBytes(64, 5);
    const w1 = await rawMailboxOp(sab, HTTPFS_OP_WRITE, dumpHandle, 0n, p1);
    assert.equal(w1.status, HttpfsStatus.ResponseReady);
    assert.equal(w1.result, 128n);

    const w2 = await rawMailboxOp(sab, HTTPFS_OP_WRITE, dumpHandle, HTTPFS_OPEN_FLAG_APPEND, p2);
    assert.equal(w2.status, HttpfsStatus.ResponseReady);
    assert.equal(w2.result, 64n);

    const closed = await rawMailboxOp(sab, HttpfsOp.Close, dumpHandle, 0n);
    assert.equal(closed.status, HttpfsStatus.ResponseReady);
    assert.equal(closed.result, 0n);

    const expected = new Uint8Array(192);
    expected.set(p1, 0);
    expected.set(p2, 128);
    assert.deepEqual(mock.dumpedFiles.get('/__pg6_dump/mailbox_pso.dxil'), expected);

    // 2. Attempt to open game file with write flag over SAB mailbox -> refused with -30 (EROFS)
    const openGameWrite = await rawMailboxOp(
      sab,
      HttpfsOp.Open,
      -1,
      HTTPFS_OPEN_FLAG_WRITE,
      utf8Encoder.encode('G:\\movies.rpf'),
    );
    assert.equal(openGameWrite.status, HttpfsStatus.Error);
    assert.equal(openGameWrite.result, -30n);

    // 3. Open game file read-only, then attempt HTTPFS_OP_WRITE on that handle -> refused with -30 (EROFS)
    const openGameRead = await rawMailboxOp(
      sab,
      HttpfsOp.Open,
      -1,
      0n,
      utf8Encoder.encode('G:\\movies.rpf'),
    );
    assert.equal(openGameRead.status, HttpfsStatus.ResponseReady);
    const gameHandle = Number(openGameRead.result);
    const badWrite = await rawMailboxOp(sab, HTTPFS_OP_WRITE, gameHandle, 0n, p1);
    assert.equal(badWrite.status, HttpfsStatus.Error);
    assert.equal(badWrite.result, -30n);

    assert.equal(controller.getStats().dumpWritesCompleted, 2);
    assert.equal(controller.getStats().dumpBytesWritten, 192);
    assert.equal(controller.getStats().dumpWriteErrors, 2);
  } finally {
    await controller.detachMailbox();
  }
});

test('W173: tuned sequential slab prefetch & bounded concurrency for 4K Bink2 streaming alongside live dump POSTs', async () => {
  assert.equal(isVideoStreamPath('x64/movies/intro_4k.bk2'), true);
  assert.equal(isVideoStreamPath('movies/logo.BIK'), true);
  assert.equal(isVideoStreamPath('audio.rpf'), false);
  assert.equal(isImmutableHttpfsPath('x64/movies/intro_4k.bk2'), true);
  assert.equal(isImmutableHttpfsPath('movies/logo.bik'), true);

  // Verify 4K video preset and upper safety clamping
  const video4k = createSlabCacheTuningProfile('video4k');
  assert.deepEqual(video4k, VIDEO_STREAM_SLAB_PRESET);
  const clamped = createSlabCacheTuningProfile({
    slabSize: 512 * 1024 * 1024,
    slabMaxBytes: 2 * 1024 * 1024 * 1024,
    prefetchAheadSlabs: 100,
    maxConcurrentSlabFetches: 100,
  });
  assert.equal(clamped.slabSize, MAX_SLAB_SIZE);
  assert.equal(clamped.slabMaxBytes, MAX_SLAB_CACHE_BYTES);
  assert.ok(clamped.prefetchAheadSlabs <= 8);
  assert.equal(clamped.maxConcurrentSlabFetches, 16);

  // Synthetic 4K Bink2 video file: 12 slabs of 1024 bytes = 12288 bytes
  // Slab budget: 4 slabs max (4096 bytes), prefetchAheadSlabs: 2, maxConcurrentSlabFetches: 2
  const slabSize = 1024;
  const videoBytes = makePatternBytes(12 * slabSize, 137);
  let activeRangeFetches = 0;
  let peakRangeFetches = 0;

  const mock = createMockDumpAndRangeFetch(
    new Map([['/game/x64/movies/intro_4k.bk2', videoBytes]]),
    { delayMs: 8 },
  );

  const instrumentedFetch = async (url, init) => {
    const isRangeGet = (!init || !init.method || init.method === 'GET') && init && init.headers && init.headers.Range;
    if (isRangeGet) {
      activeRangeFetches += 1;
      if (activeRangeFetches > peakRangeFetches) {
        peakRangeFetches = activeRangeFetches;
      }
    }
    try {
      return await mock.fetchImpl(url, init);
    } finally {
      if (isRangeGet) {
        activeRangeFetches -= 1;
      }
    }
  };

  const controller = new IoWorkerController({
    fetchImpl: instrumentedFetch,
    opfsDir: null,
  });

  await controller.init({
    dataBase: '/data/',
    gameBase: '/game/',
    allowDumpWrites: true,
    dumpEndpoint: '/__pg6_dump/',
  });

  // Tune slab cache live for bounded sequential 4K Bink2 video streaming
  const tuned = controller.tuneSlabCache({
    slabSize,
    slabMaxBytes: 4 * slabSize, // Strictly 4 slabs in RAM (4096B cap)
    prefetchAheadSlabs: 2,
    maxConcurrentSlabFetches: 2,
    sequentialThresholdReads: 2,
  });
  assert.equal(tuned.slabSize, slabSize);
  assert.equal(tuned.slabMaxBytes, 4 * slabSize);
  assert.equal(tuned.prefetchAheadSlabs, 2);
  assert.equal(tuned.maxConcurrentSlabFetches, 2);

  // Open Bink2 video via Xbox drive path (routes to /game/ automatically)
  const hVid = await controller.openHandle('G:\\x64\\movies\\intro_4k.bk2', { probeOnOpen: false });
  assert.ok(hVid >= 1);
  assert.equal(controller.getHandleInfo(hVid).mount, 'game');

  // Stream sequentially in 512B reads across the first 8 slabs while concurrently dumping DXIL shaders!
  const assembled = new Uint8Array(8 * slabSize);
  const dumpPromises = [];
  for (let i = 0; i < 16; i += 1) {
    const off = i * 512;
    const chunk = await controller.readHandle(hVid, off, 512);
    assert.equal(chunk.byteLength, 512);
    assembled.set(chunk, off);
    if (i % 4 === 0) {
      dumpPromises.push(
        controller.writeDump(`/pg6_dump/live_pso_${i}.dxil`, makePatternBytes(256, i + 1)),
      );
    }
    // Wait for background prefetches after slab boundary reads to verify hit acceleration
    if (i % 2 === 1) {
      await controller.immutableCache.waitForPrefetches();
    }
    // Worker slab memory must NEVER exceed slabMaxBytes
    assert.ok(
      controller.getStats().immutableCache.slabBytesCached <= 4 * slabSize,
      'slabBytesCached must never exceed slabMaxBytes',
    );
  }
  await Promise.all(dumpPromises);
  await controller.immutableCache.waitForPrefetches();

  assert.deepEqual(assembled, videoBytes.subarray(0, 8 * slabSize));

  // Verify bounded concurrency semaphore enforced maxConcurrentSlabFetches (<= 2)
  assert.ok(
    peakRangeFetches <= 2,
    `Expected peak concurrent slab fetches <= 2, got ${peakRangeFetches}`,
  );

  // Verify telemetry: sequential prefetch warmed slabs so hits >> misses & dumps succeeded concurrently
  const stats = controller.getStats();
  assert.ok(stats.slabCacheHits >= 12, `Expected >= 12 slab hits, got ${stats.slabCacheHits}`);
  assert.ok(
    stats.slabCacheMisses <= 2,
    `Expected <= 2 demand slab misses due to sequential prefetch, got ${stats.slabCacheMisses}`,
  );
  assert.ok(stats.immutableCache.slabPrefetchesScheduled >= 6);
  assert.ok(stats.immutableCache.slabPrefetchesCompleted >= 6);
  assert.ok(stats.immutableCache.slabEvictions >= 1, 'Expected bounded LRU slab evictions');
  assert.ok(stats.immutableCache.peakConcurrentSlabFetches <= 2);
  assert.equal(stats.dumpWritesCompleted, 4);
  assert.equal(stats.dumpBytesWritten, 4 * 256);
  assert.equal(stats.dumpWriteErrors, 0);

  // Verify explicit prefetchVideoSlabs + shrinking slabMaxBytes via tuneSlabCache evicts cleanly
  const preRes = await controller.prefetchVideoSlabs('G:\\x64\\movies\\intro_4k.bk2', 10 * slabSize, 2);
  assert.equal(preRes.completed, 2);
  const shrunk = controller.tuneSlabCache({ slabMaxBytes: 2 * slabSize });
  assert.equal(shrunk.slabMaxBytes, 2 * slabSize);
  assert.ok(controller.getStats().immutableCache.slabBytesCached <= 2 * slabSize);

  assert.equal(controller.closeHandle(hVid), true);
});

test('W173: classic Web Worker self.onmessage forwards dump writes, slab tuning, and telemetry', async () => {
  const binkBytes = makePatternBytes(4096, 51);
  const mock = createMockDumpAndRangeFetch(new Map([['/game/intro.bk2', binkBytes]]));
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
    Promise,
    Error,
    Number,
    Math,
    Object,
    Array,
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
      allowDumpWrites: true,
      dumpEndpoint: '/__pg6_dump/',
    },
  });
  assert.equal(posted[0].ready, true);
  assert.equal(posted[0].allowDumpWrites, true);
  assert.equal(posted[0].dumpEndpoint, '/__pg6_dump/');

  // Tune slab cache via worker message
  await workerSelf.onmessage({
    data: {
      tuneSlabCache: true,
      id: 10,
      slabSize: 1024,
      slabMaxBytes: 4096,
      prefetchAheadSlabs: 2,
      maxConcurrentSlabFetches: 2,
    },
  });
  const tuneMsg = posted.find((m) => m.id === 10);
  assert.equal(tuneMsg.ok, true);
  assert.equal(tuneMsg.slabConfig.prefetchAheadSlabs, 2);

  // Open, write two chunks, and close a dump file via worker messages
  await workerSelf.onmessage({ data: { openDump: true, id: 11, path: '/pg6_dump/worker_pso.dxil' } });
  const openMsg = posted.find((m) => m.id === 11);
  assert.equal(openMsg.ok, true);

  await workerSelf.onmessage({
    data: { writeDump: true, id: 12, handle: openMsg.handle, data: new Uint8Array([1, 2, 3]) },
  });
  await workerSelf.onmessage({
    data: { writeDump: true, id: 13, handle: openMsg.handle, data: new Uint8Array([4, 5]) },
  });
  await workerSelf.onmessage({ data: { closeDump: true, id: 14, handle: openMsg.handle } });

  // Attempt write to read-only game path via worker message -> rejected cleanly
  await workerSelf.onmessage({ data: { openDump: true, id: 15, path: 'G:\\intro.bk2' } });
  const deniedMsg = posted.find((m) => m.id === 15);
  assert.equal(deniedMsg.ok, false);
  assert.equal(deniedMsg.errno, -30);

  await workerSelf.onmessage({ data: { stats: true, id: 16 } });
  const statsMsg = posted.find((m) => m.id === 16);
  assert.equal(statsMsg.stats.dumpWritesCompleted, 2);
  assert.equal(statsMsg.stats.dumpBytesWritten, 5);
  assert.equal(statsMsg.stats.dumpWriteErrors, 1);
  assert.deepEqual(
    mock.dumpedFiles.get('/__pg6_dump/worker_pso.dxil'),
    new Uint8Array([1, 2, 3, 4, 5]),
  );
});

// ---------------------------------------------------------------------------
// 9. W219: Batched multi-range read dispatch, zero-copy slicing & streaming telemetry
// ---------------------------------------------------------------------------

test('W219: coalesceBatchReadRequests merges adjacent/aliased ranges, promotes priority, and isolates invalid/zero entries', () => {
  assert.equal(DEFAULT_BATCH_COALESCE_MAX_GAP_BYTES, 64 * 1024);
  assert.equal(DEFAULT_BATCH_COALESCE_MAX_BYTES, 2 * 1024 * 1024);

  const plan = coalesceBatchReadRequests(
    [
      { id: 'r0', path: 'G:\\audio.rpf', offset: 128, length: 64, priority: IO_PRIORITY.NORMAL },
      { id: 'r1', path: '/game/audio.rpf', offset: 0, length: 64, priority: IO_PRIORITY.PREFETCH },
      { id: 'r2', path: 'audio.rpf', offset: 64, length: 64, priority: IO_PRIORITY.CRITICAL },
      { id: 'r3', path: 'audio.rpf', offset: 5000, length: 32 }, // beyond maxGapBytes=128 -> separate group
      { id: 'r4', path: 'notes.txt', offset: 10, length: 20 }, // different asset
      { id: 'r5', path: 'notes.txt', offset: 40, length: 0 }, // zero-length valid read
      { id: 'r6', path: '../escape.bin', offset: 0, length: 16 }, // traversal -> invalid
      { id: 'r7', path: 'notes.txt', offset: -5, length: 16 }, // negative offset -> invalid
      { id: 'r8', path: 'notes.txt', offset: 0, length: -1 }, // negative length -> invalid
    ],
    {
      dataBase: '/data/',
      bootBase: '/boot/',
      gameBase: '/game/',
      maxGapBytes: 128,
      maxCoalescedBytes: 1024,
    },
  );

  assert.equal(plan.zeroItems.length, 1);
  assert.equal(plan.zeroItems[0].id, 'r5');
  assert.equal(plan.invalidItems.length, 3);
  assert.deepEqual(
    plan.invalidItems.map((x) => [x.id, x.errno]),
    [
      ['r6', -22],
      ['r7', -22],
      ['r8', -22],
    ],
  );

  // Groups: 1 coalesced group [0..192) on /game/audio.rpf, 1 disjoint [5000..5032) on /game/audio.rpf, 1 on /data/notes.txt
  assert.equal(plan.groups.length, 3);
  const mergedGroup = plan.groups.find((g) => g.route.url === '/game/audio.rpf' && g.offset === 0);
  assert.ok(mergedGroup);
  assert.equal(mergedGroup.coalesced, true);
  assert.equal(mergedGroup.offset, 0);
  assert.equal(mergedGroup.length, 192);
  assert.equal(mergedGroup.gapBytes, 0);
  assert.equal(mergedGroup.priority, IO_PRIORITY.CRITICAL);
  assert.deepEqual(
    mergedGroup.items.map((it) => it.id),
    ['r1', 'r2', 'r0'],
  );

  const farGroup = plan.groups.find((g) => g.route.url === '/game/audio.rpf' && g.offset === 5000);
  assert.ok(farGroup);
  assert.equal(farGroup.coalesced, false);
  assert.equal(farGroup.length, 32);

  // collectTransferableBuffers deduplicates shared ArrayBuffers and skips empty/SAB/failed entries
  const shared = new Uint8Array(64);
  const sab = new SharedArrayBuffer(16);
  const transfer = collectTransferableBuffers([
    { ok: true, bytes: shared.subarray(0, 32) },
    { ok: true, bytes: shared.subarray(32, 64) },
    { ok: true, bytes: new Uint8Array(0) },
    { ok: true, bytes: new Uint8Array(sab) },
    { ok: false, bytes: null },
  ]);
  assert.equal(transfer.length, 1);
  assert.equal(transfer[0], shared.buffer);
});

test('W219: readBatch coalesces adjacent ranges into a single HTTP 206 fetch, slices with zero extra copies, and isolates overlapping callers', async () => {
  const assetA = makePatternBytes(512, 0x21);
  const assetB = makePatternBytes(256, 0x42);

  const mock = createMockRangeFetch(
    new Map([
      ['/game/world.rpf', assetA],
      ['/data/config.dat', assetB],
    ]),
  );

  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    chunkSize: 64,
    maxBytes: 8192,
    opfsDir: null,
  });
  await controller.init({ dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' });
  const reqsBefore = mock.requests.length;

  // Batch with out-of-order adjacent ranges, small gap (16B), overlap, EOF clamp, past-EOF, zero-length, and second asset
  const batch = await controller.readBatch(
    [
      { id: 'tail', path: 'G:\\world.rpf', offset: 120, length: 60 }, // [120..180) (gap 20B from [0..100))
      { id: 'head', path: '/game/world.rpf', offset: 0, length: 64 }, // [0..64)
      { id: 'mid', path: 'world.rpf', offset: 64, length: 36 }, // [64..100) adjacent to head
      { id: 'overlap', path: 'world.rpf', offset: 32, length: 48 }, // [32..80) overlaps head & mid -> isolated copy
      { id: 'eof_clamp', path: 'config.dat', offset: 220, length: 100 }, // clamps to [220..256) (36B)
      { id: 'past_eof', path: 'config.dat', offset: 260, length: 32 }, // coalesces with eof_clamp (gap 4B) -> 0B
      { id: 'empty', path: 'config.dat', offset: 10, length: 0 }, // zero-length -> 0B
    ],
    { maxGapBytes: 64, maxCoalescedBytes: 1024 },
  );

  assert.equal(batch.ok, true);
  assert.equal(batch.requested, 7);
  assert.equal(batch.succeeded, 7);
  assert.equal(batch.failed, 0);
  assert.equal(batch.fetches, 2); // 1 coalesced span for world.rpf + 1 coalesced span for config.bin
  assert.equal(batch.coalescedFetches, 2);
  assert.equal(batch.bytesTransferred, 60 + 64 + 36 + 48 + 36 + 0 + 0);

  // Verify ONLY 2 HTTP 206 Range requests went over the wire (NOT 6 individual requests!)
  const rangeReqs = mock.requests.slice(reqsBefore).filter((r) => r.rangeHeader);
  assert.equal(rangeReqs.length, 2, `Expected 2 coalesced HTTP 206 fetches, got ${rangeReqs.length}`);

  // Verify exact byte slicing fidelity in original caller order
  const byId = new Map(batch.results.map((r) => [r.id, r]));
  assert.equal(batch.results[0].id, 'tail');
  assert.equal(batch.results[1].id, 'head');
  assert.equal(batch.results[2].id, 'mid');
  assert.equal(batch.results[3].id, 'overlap');
  assert.equal(batch.results[4].id, 'eof_clamp');
  assert.equal(batch.results[5].id, 'past_eof');
  assert.equal(batch.results[6].id, 'empty');

  assert.deepEqual(byId.get('head').bytes, assetA.subarray(0, 64));
  assert.deepEqual(byId.get('mid').bytes, assetA.subarray(64, 100));
  assert.deepEqual(byId.get('tail').bytes, assetA.subarray(120, 180));
  assert.deepEqual(byId.get('overlap').bytes, assetA.subarray(32, 80));
  assert.deepEqual(byId.get('eof_clamp').bytes, assetB.subarray(220, 256));
  assert.equal(byId.get('past_eof').bytes.byteLength, 0);
  assert.equal(byId.get('empty').bytes.byteLength, 0);

  // Non-overlapping slices from the same coalesced span share ONE underlying ArrayBuffer (zero extra copies!)
  assert.equal(byId.get('head').bytes.buffer, byId.get('mid').bytes.buffer);
  assert.equal(byId.get('head').bytes.buffer, byId.get('tail').bytes.buffer);
  assert.equal(byId.get('head').bytes.byteOffset, 0);
  assert.equal(byId.get('mid').bytes.byteOffset, 64);
  assert.equal(byId.get('tail').bytes.byteOffset, 120);

  // Overlapping slice gets its own isolated ArrayBuffer so caller mutation never corrupts sibling slices
  assert.notEqual(byId.get('overlap').bytes.buffer, byId.get('head').bytes.buffer);
  byId.get('overlap').bytes.fill(0xff);
  assert.deepEqual(byId.get('head').bytes, assetA.subarray(0, 64));
  assert.deepEqual(byId.get('mid').bytes, assetA.subarray(64, 100));

  // transferList is deduplicated: 1 shared buffer for world.rpf non-overlapping + 1 isolated overlap + 1 config.bin
  assert.equal(batch.transferList.length, 3);
  assert.equal(new Set(batch.transferList).size, 3);

  // Repeat identical batch -> 100% cache hits with ZERO additional network fetches!
  const reqsAfterFirst = mock.requests.length;
  const cachedBatch = await controller.readBatch(
    [
      { id: 'c0', path: 'G:\\world.rpf', offset: 0, length: 64 },
      { id: 'c1', path: 'G:\\world.rpf', offset: 64, length: 64 },
    ],
    { isolateBuffers: true },
  );
  assert.equal(cachedBatch.ok, true);
  assert.equal(mock.requests.length, reqsAfterFirst, 'Warm batch read must make zero network fetches');
  assert.deepEqual(cachedBatch.results[0].bytes, assetA.subarray(0, 64));
  assert.deepEqual(cachedBatch.results[1].bytes, assetA.subarray(64, 128));
  // isolateBuffers: true gives each caller its own ArrayBuffer
  assert.notEqual(cachedBatch.results[0].bytes.buffer, cachedBatch.results[1].bytes.buffer);
  assert.equal(cachedBatch.transferList.length, 2);

  const stats = controller.getStats();
  assert.equal(stats.batchedRequests, 2);
  assert.equal(stats.coalescedFetches, 3);
  assert.equal(stats.readsCompleted, 9);
  assert.ok(stats.cacheHits > 0, `Expected positive cacheHits, got ${stats.cacheHits}`);
  assert.equal(stats.errorCount, 0);
});

test('W219: readBatch error propagation isolates per-item failures by default and rejects on failFast or invalid input', async () => {
  const validBytes = makePatternBytes(128, 0x77);
  const mock = createMockRangeFetch(new Map([['/data/good.bin', validBytes]]), {
    statusOverrides: new Map([['/data/forbidden.bin', 403]]),
  });

  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    chunkSize: 64,
    maxBytes: 4096,
    maxRetries: 0,
    opfsDir: null,
  });
  await controller.init({ dataBase: '/data/' });

  // Partial-error batch: 1 valid + 2 coalesced 404s + 1 HTTP 403 + 1 traversal + 1 bad offset
  const mixed = await controller.readBatch([
    { id: 'ok', path: 'good.bin', offset: 16, length: 32 },
    { id: 'miss1', path: 'missing.bin', offset: 0, length: 16 },
    { id: 'miss2', path: 'missing.bin', offset: 16, length: 16 },
    { id: 'deny', path: 'forbidden.bin', offset: 0, length: 16 },
    { id: 'trav', path: '../secret.bin', offset: 0, length: 16 },
    { id: 'neg', path: 'good.bin', offset: -1, length: 16 },
  ]);

  assert.equal(mixed.ok, false);
  assert.equal(mixed.requested, 6);
  assert.equal(mixed.succeeded, 1);
  assert.equal(mixed.failed, 5);

  assert.equal(mixed.results[0].ok, true);
  assert.deepEqual(mixed.results[0].bytes, validBytes.subarray(16, 48));

  assert.equal(mixed.results[1].ok, false);
  assert.equal(mixed.results[1].errno, -2);
  assert.equal(mixed.results[1].bytes, null);

  assert.equal(mixed.results[2].ok, false);
  assert.equal(mixed.results[2].errno, -2);

  assert.equal(mixed.results[3].ok, false);
  assert.equal(mixed.results[3].errno, -13);

  assert.equal(mixed.results[4].ok, false);
  assert.equal(mixed.results[4].errno, -22);

  assert.equal(mixed.results[5].ok, false);
  assert.equal(mixed.results[5].errno, -22);

  // Only the valid item's buffer is in transferList
  assert.equal(mixed.transferList.length, 1);
  assert.equal(mixed.transferList[0], mixed.results[0].bytes.buffer);
  assert.equal(controller.getStats().errorCount, 5);
  assert.equal(controller.getStats().readsCompleted, 1);

  // failFast: true rejects immediately on HTTP 404 or validation error
  await assert.rejects(
    () =>
      controller.readBatch([{ path: 'missing.bin', offset: 0, length: 16 }], {
        failFast: true,
      }),
    /404/,
  );
  assert.equal(controller.getStats().errorCount, 6);

  await assert.rejects(
    () =>
      controller.readBatch([{ path: '../escape.bin', offset: 0, length: 16 }], {
        failFast: true,
      }),
    (err) => err && err.errno === -22,
  );
  assert.equal(controller.getStats().errorCount, 7);

  // Non-array requests rejects with EINVAL (-22)
  await assert.rejects(
    () => controller.readBatch('not-an-array'),
    (err) => err && err.errno === -22,
  );
  assert.equal(controller.getStats().errorCount, 8);

  // Aborted signal propagates AbortError cleanly
  const ac = new AbortController();
  ac.abort('cancelled by caller');
  const aborted = await controller.readBatch(
    [{ id: 'ab', path: 'good.bin', offset: 0, length: 16 }],
    { signal: ac.signal },
  );
  assert.equal(aborted.ok, false);
  assert.equal(aborted.failed, 1);
  assert.equal(controller.getStats().errorCount, 9);
});

test('W219: streaming telemetry tracks exact readsCompleted, bytesTransferred, batchedRequests, coalescedFetches, cacheHits, errorCount, and avgReadMs', async () => {
  // Check baseline zeroed emptyStats() / emptyStreamingStats()
  const baseEmpty = emptyStats();
  const baseStreaming = emptyStreamingStats();
  for (const key of [
    'readsCompleted',
    'bytesTransferred',
    'batchedRequests',
    'coalescedFetches',
    'cacheHits',
    'errorCount',
    'avgReadMs',
  ]) {
    assert.equal(baseEmpty[key], 0, `emptyStats().${key} must default to 0`);
    assert.equal(baseEmpty.streaming[key], 0, `emptyStats().streaming.${key} must default to 0`);
    assert.equal(baseStreaming[key], 0, `emptyStreamingStats().${key} must default to 0`);
  }

  let clockMs = 1000;
  const asset = makePatternBytes(256, 0x99);
  const rawMock = createMockRangeFetch(new Map([['/data/stream.bin', asset]]));
  const timedFetch = async (url, init) => {
    const res = await rawMock.fetchImpl(url, init);
    if (init && init.headers && init.headers.Range) {
      clockMs += 12; // deterministic 12ms per network range fetch
    }
    return res;
  };

  const controller = new IoWorkerController({
    fetchImpl: timedFetch,
    chunkSize: 64,
    maxBytes: 4096,
    maxRetries: 0,
    now: () => clockMs,
    opfsDir: null,
  });
  await controller.init({ dataBase: '/data/' });

  // 1. Single cold read: 40B taking 12ms -> readsCompleted=1, bytesTransferred=40, avgReadMs=12
  const single = await controller.read('stream.bin', 0, 40);
  assert.deepEqual(single, asset.subarray(0, 40));
  let st = controller.getStats();
  assert.equal(st.readsCompleted, 1);
  assert.equal(st.bytesTransferred, 40);
  assert.equal(st.batchedRequests, 0);
  assert.equal(st.coalescedFetches, 0);
  assert.equal(st.cacheHits, 0);
  assert.equal(st.errorCount, 0);
  assert.equal(st.avgReadMs, 12);
  assert.equal(st.streaming.avgReadMs, 12);

  // 2. Coalesced cold batch of 2 adjacent ranges in chunks 1 & 2 (12ms total for both)
  const b1 = await controller.readBatch([
    { path: 'stream.bin', offset: 64, length: 64 },
    { path: 'stream.bin', offset: 128, length: 64 },
  ]);
  assert.equal(b1.ok, true);
  assert.equal(b1.coalescedFetches, 1);
  st = controller.getStats();
  assert.equal(st.readsCompleted, 3); // 1 + 2
  assert.equal(st.bytesTransferred, 40 + 64 + 64);
  assert.equal(st.batchedRequests, 1);
  assert.equal(st.coalescedFetches, 1);
  assert.equal(st.avgReadMs, 12); // (12 + 12 + 12) / 3 = 12

  // 3. Warm single read (0ms cache hit) -> avgReadMs = 36 / 4 = 9, cacheHits >= 1
  const warm = await controller.read('stream.bin', 0, 32);
  assert.deepEqual(warm, asset.subarray(0, 32));
  st = controller.getStats();
  assert.equal(st.readsCompleted, 4);
  assert.equal(st.bytesTransferred, 40 + 64 + 64 + 32);
  assert.equal(st.cacheHits, 1);
  assert.equal(st.avgReadMs, 9);

  // 4. Failed single read increments errorCount once (does not distort readsCompleted/avgReadMs)
  await assert.rejects(() => controller.read('missing.bin', 0, 16), /404/);
  st = controller.getStats();
  assert.equal(st.errorCount, 1);
  assert.equal(st.readsCompleted, 4);
  assert.equal(st.avgReadMs, 9);

  // 5. Sanitized telemetry contains zero paths or URLs
  const json = JSON.stringify(st.streaming);
  assert.equal(json.includes('stream.bin'), false);
  assert.equal(json.includes('/data/'), false);

  // 6. Re-init resets all streaming counters cleanly
  await controller.init({ dataBase: '/data/' });
  st = controller.getStats();
  assert.equal(st.readsCompleted, 0);
  assert.equal(st.bytesTransferred, 0);
  assert.equal(st.batchedRequests, 0);
  assert.equal(st.coalescedFetches, 0);
  assert.equal(st.cacheHits, 0);
  assert.equal(st.errorCount, 0);
  assert.equal(st.avgReadMs, 0);
});

test('W219: classic Web Worker self.onmessage handles readBatch with zero-copy transferable ArrayBuffers and streaming stats', async () => {
  const dataA = makePatternBytes(256, 0x3a);
  const mock = createMockRangeFetch(new Map([['/game/textures.rpf', dataA]]));
  const posted = [];
  const workerSelf = {
    onmessage: null,
    postMessage(msg, transfer) {
      // Simulate real browser structuredClone with transfer list to guarantee zero DataCloneError!
      const cloned =
        Array.isArray(transfer) && transfer.length > 0
          ? structuredClone(msg, { transfer })
          : structuredClone(msg);
      posted.push({
        msg: cloned,
        transferCount: Array.isArray(transfer) ? transfer.length : 0,
      });
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

  // Deliberately omit `performance` to verify Date.now() fallback in minimal Worker scopes.
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

  assert.equal(typeof sandbox.PlayGta6IoWorker.coalesceBatchReadRequests, 'function');
  assert.equal(typeof sandbox.PlayGta6IoWorker.collectTransferableBuffers, 'function');
  assert.equal(typeof sandbox.PlayGta6IoWorker.emptyStreamingStats, 'function');

  await workerSelf.onmessage({
    data: { init: true, dataBase: '/data/', gameBase: '/game/', chunkSize: 64, maxBytes: 4096 },
  });
  assert.equal(posted[0].msg.ready, true);

  const reqsBefore = mock.requests.length;
  await workerSelf.onmessage({
    data: {
      readBatch: true,
      id: 42,
      requests: [
        { id: 'a', path: 'G:\\textures.rpf', offset: 0, length: 64 },
        { id: 'b', path: '/game/textures.rpf', offset: 64, length: 64 },
        { id: 'c', path: 'textures.rpf', offset: 128, length: 32 },
      ],
    },
  });

  const batchReply = posted.find((p) => p.msg.id === 42);
  assert.ok(batchReply);
  assert.equal(batchReply.msg.ok, true);
  assert.equal(batchReply.msg.requested, 3);
  assert.equal(batchReply.msg.succeeded, 3);
  assert.equal(batchReply.msg.failed, 0);
  assert.equal(batchReply.msg.fetches, 1);
  assert.equal(batchReply.msg.coalescedFetches, 1);
  // All 3 non-overlapping slices shared 1 transferred ArrayBuffer (zero extra copies, no duplicate transfer!)
  assert.equal(batchReply.transferCount, 1);
  assert.equal(
    mock.requests.slice(reqsBefore).filter((r) => r.rangeHeader).length,
    1,
    'Worker readBatch must coalesce all 3 adjacent ranges into 1 HTTP 206 fetch',
  );
  assert.deepEqual(batchReply.msg.results[0].bytes, dataA.subarray(0, 64));
  assert.deepEqual(batchReply.msg.results[1].bytes, dataA.subarray(64, 128));
  assert.deepEqual(batchReply.msg.results[2].bytes, dataA.subarray(128, 160));

  // Verify error message on readBatch with invalid payload does not double-count errorCount
  await workerSelf.onmessage({
    data: { readBatch: true, id: 43, requests: 'bad' },
  });
  const errReply = posted.find((p) => p.msg.id === 43);
  assert.ok(errReply && errReply.msg.error);
  assert.equal(errReply.msg.errno, -22);

  await workerSelf.onmessage({ data: { stats: true, id: 44 } });
  const statsReply = posted.find((p) => p.msg.id === 44);
  assert.ok(statsReply && statsReply.msg.stats);
  const s = statsReply.msg.stats;
  assert.equal(s.readsCompleted, 3);
  assert.equal(s.bytesTransferred, 160);
  assert.equal(s.batchedRequests, 1);
  assert.equal(s.coalescedFetches, 1);
  assert.equal(s.errorCount, 1);
  assert.equal(typeof s.avgReadMs, 'number');
  assert.ok(Number.isFinite(s.avgReadMs) && s.avgReadMs >= 0);
  assert.equal(s.streaming.readsCompleted, 3);
  assert.equal(s.streaming.bytesTransferred, 160);
  assert.equal(s.streaming.coalescedFetches, 1);
  assert.equal(s.streaming.errorCount, 1);
});

// ---------------------------------------------------------------------------
// 10. W255 (#1104 / #1015): Zero chunk+slab double-fetch & large slab LRU
//     working-set retention for shaders.rpf / core_bvh4.rpf
// ---------------------------------------------------------------------------

function parseFetchedRangeSpan(rangeHeader, fileSize) {
  const match = /^bytes=(\d+)-(\d+)$/.exec(rangeHeader || '');
  assert.ok(match, `Expected Range header bytes=start-end, got ${rangeHeader}`);
  const start = Number(match[1]);
  const end = Math.min(Number(match[2]), fileSize - 1);
  return { start, end, bytes: end >= start ? end - start + 1 : 0 };
}

function assertZeroOverlappingRanges(requests, fileSize) {
  const spans = requests
    .filter((r) => r.rangeHeader)
    .map((r) => parseFetchedRangeSpan(r.rangeHeader, fileSize))
    .sort((a, b) => a.start - b.start);

  let totalFetchedBytes = 0;
  for (let i = 0; i < spans.length; i += 1) {
    totalFetchedBytes += spans[i].bytes;
    if (i > 0) {
      assert.ok(
        spans[i].start > spans[i - 1].end,
        `Duplicate/overlapping HTTP Range fetch detected: [${spans[i - 1].start}..${spans[i - 1].end}] overlaps [${spans[i].start}..${spans[i].end}]`,
      );
    }
  }
  return { spans, totalFetchedBytes };
}

test('W255: open() + reads across 4 MiB file issue ZERO overlapping HTTP Range fetches and coalesce concurrent open+read', async () => {
  const fileSize = 4 * 1024 * 1024; // 4 MiB = 2 x 2 MiB slabs
  const rpfBytes = makePatternBytes(fileSize, 0xa5);
  const mock = createMockRangeFetch(new Map([['/game/core_bvh4.rpf', rpfBytes]]), { delayMs: 5 });

  const cache = new ImmutableHttpfsCache({ fetchImpl: mock.fetchImpl });
  assert.equal(cache.slabSize, 2 * 1024 * 1024);
  assert.equal(cache.slabMaxBytes, 192 * 1024 * 1024);
  assert.equal(cache.maxBytes, 128 * 1024 * 1024);

  // 1. open(url) probes slab 0 [0..2 MiB - 1] directly instead of chunk 0 [0..64 KiB - 1].
  const openRes = await cache.open('/game/core_bvh4.rpf');
  assert.deepEqual(openRes, { ok: true, size: fileSize });
  assert.equal(mock.requests.length, 1);
  assert.equal(mock.requests[0].rangeHeader, `bytes=0-${2 * 1024 * 1024 - 1}`);

  // 2. read(0, 64 KiB) and read(64 KiB, 1 MiB) hit slab 0 with ZERO additional HTTP requests.
  const first64k = await cache.read('/game/core_bvh4.rpf', 0, 65536);
  assert.deepEqual(first64k, rpfBytes.subarray(0, 65536));
  const next1m = await cache.read('/game/core_bvh4.rpf', 65536, 1024 * 1024);
  assert.deepEqual(next1m, rpfBytes.subarray(65536, 65536 + 1024 * 1024));
  assert.equal(mock.requests.length, 1, 'Reads within slab 0 after open() must issue 0 extra fetches');

  // 3. Cross-slab read spanning slab 0 -> slab 1 fetches ONLY slab 1 (never re-fetches slab 0).
  const crossOff = 2 * 1024 * 1024 - 32768;
  const crossLen = 128 * 1024;
  const crossBytes = await cache.read('/game/core_bvh4.rpf', crossOff, crossLen);
  assert.deepEqual(crossBytes, rpfBytes.subarray(crossOff, crossOff + crossLen));
  assert.equal(mock.requests.length, 2);
  assert.equal(
    mock.requests[1].rangeHeader,
    `bytes=${2 * 1024 * 1024}-${4 * 1024 * 1024 - 1}`,
  );

  // 4. Tail read in slab 1 hits synchronously with ZERO additional HTTP requests.
  const tailBytes = await cache.read('/game/core_bvh4.rpf', 3 * 1024 * 1024, 1024 * 1024);
  assert.deepEqual(tailBytes, rpfBytes.subarray(3 * 1024 * 1024, 4 * 1024 * 1024));
  assert.equal(mock.requests.length, 2);

  const { totalFetchedBytes } = assertZeroOverlappingRanges(mock.requests, fileSize);
  assert.equal(totalFetchedBytes, fileSize, 'Total fetched bytes must strictly equal 4 MiB (0% inflation)');

  const stats = cache.getStats();
  assert.equal(stats.networkFetches, 2);
  assert.equal(stats.slabCacheMisses, 2);
  assert.equal(stats.slabEntries, 2);
  assert.equal(stats.slabBytesCached, fileSize);
  assert.equal(stats.entries, 0, 'Chunk map must remain empty in slab mode (zero chunk+slab duplication)');
  assert.equal(stats.bytesCached, 0);

  // 5. Concurrent open() + read(64 KiB, 64 KiB) on a cold file coalesce onto ONE single slab-0 fetch.
  const concMock = createMockRangeFetch(new Map([['/game/concurrent_bvh.rpf', rpfBytes]]), {
    delayMs: 10,
  });
  const concCache = new ImmutableHttpfsCache({ fetchImpl: concMock.fetchImpl });
  const [concOpen, concRead] = await Promise.all([
    concCache.open('/game/concurrent_bvh.rpf'),
    concCache.read('/game/concurrent_bvh.rpf', 65536, 65536),
  ]);
  assert.deepEqual(concOpen, { ok: true, size: fileSize });
  assert.deepEqual(concRead, rpfBytes.subarray(65536, 131072));
  assert.equal(
    concMock.requests.length,
    1,
    'Concurrent open() and read() in slab 0 must coalesce into a single HTTP 206 fetch',
  );
  assert.ok(concCache.getStats().inFlightCoalesced >= 1);
});

test('W255: sequential 64 KiB and 1 MiB reads across 8 MiB file fetch each 2 MiB slab strictly ONCE (0% re-fetch inflation) and share warm slabs across controller.read/readBatch', async () => {
  const fileSize = 8 * 1024 * 1024; // 8 MiB = 4 x 2 MiB slabs
  const bvhBytes = makePatternBytes(fileSize, 0x3c);
  const mock = createMockRangeFetch(new Map([['/game/core_bvh4.rpf', bvhBytes]]));

  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    opfsDir: null,
  });
  await controller.init({ dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' });

  const reqsBefore = mock.requests.length;
  const h = await controller.openHandle('G:\\core_bvh4.rpf');
  assert.ok(h >= 1);
  assert.equal(controller.getHandleInfo(h).size, fileSize);

  // Stream first 4 MiB in 64 KiB steps (64 reads) and remaining 4 MiB in 1 MiB steps (4 reads).
  const assembled = new Uint8Array(fileSize);
  for (let off = 0; off < 4 * 1024 * 1024; off += 65536) {
    const chunk = await controller.readHandle(h, off, 65536);
    assert.equal(chunk.byteLength, 65536);
    assembled.set(chunk, off);
  }
  for (let off = 4 * 1024 * 1024; off < fileSize; off += 1024 * 1024) {
    const chunk = await controller.readHandle(h, off, 1024 * 1024);
    assert.equal(chunk.byteLength, 1024 * 1024);
    assembled.set(chunk, off);
  }
  assert.deepEqual(assembled, bvhBytes);

  const gameReqs = mock.requests.slice(reqsBefore);
  assert.equal(
    gameReqs.length,
    4,
    `Expected strictly 4 x 2 MiB slab fetches for 8 MiB file, got ${gameReqs.length}`,
  );
  const { totalFetchedBytes } = assertZeroOverlappingRanges(gameReqs, fileSize);
  assert.equal(totalFetchedBytes, fileSize, 'Total fetched bytes must equal exact 8 MiB file size');

  // Re-reading the same warmed .rpf through controller.read() and controller.readBatch()
  // must serve directly from warm ImmutableHttpfsCache slabs with ZERO duplicate fetches!
  const directSlice = await controller.read('G:\\core_bvh4.rpf', 1024 * 1024, 128 * 1024);
  assert.deepEqual(directSlice, bvhBytes.subarray(1024 * 1024, 1024 * 1024 + 128 * 1024));

  const batchRes = await controller.readBatch([
    { id: 's0', path: '/game/core_bvh4.rpf', offset: 0, length: 65536 },
    { id: 's3', path: 'G:\\core_bvh4.rpf', offset: 6 * 1024 * 1024 + 4096, length: 65536 },
  ]);
  assert.equal(batchRes.ok, true);
  assert.deepEqual(batchRes.results[0].bytes, bvhBytes.subarray(0, 65536));
  assert.deepEqual(
    batchRes.results[1].bytes,
    bvhBytes.subarray(6 * 1024 * 1024 + 4096, 6 * 1024 * 1024 + 4096 + 65536),
  );
  assert.equal(
    mock.requests.length - reqsBefore,
    4,
    'Follow-up controller.read() / readBatch() on warm immutable .rpf must trigger 0 extra fetches',
  );

  const stats = controller.getStats().immutableCache;
  assert.equal(stats.slabEntries, 4);
  assert.equal(stats.slabBytesCached, fileSize);
  assert.equal(stats.slabEvictions, 0);
  assert.equal(stats.entries, 0);
  assert.equal(controller.closeHandle(h), true);
});

test('W255: shaders.rpf random working set fits inside enlarged 192 MiB slab LRU with 0 post-warmup evictions while respecting explicit/low-memory overrides', async () => {
  // 1. Assert production defaults & profile caps comfortably hold the ~115 MB shaders.rpf working set.
  assert.ok(DEFAULT_SLAB_MAX_BYTES >= 192 * 1024 * 1024);
  assert.equal(DEFAULT_SLAB_MAX_BYTES, 192 * 1024 * 1024);
  assert.equal(MAX_SLAB_CACHE_BYTES, 256 * 1024 * 1024);
  assert.equal(DEFAULT_IMMUTABLE_MAX_BYTES, 128 * 1024 * 1024);
  assert.ok(SLAB_TUNING_PROFILES.video4k.slabMaxBytes >= 192 * 1024 * 1024);
  assert.equal(SLAB_TUNING_PROFILES['low-memory'].slabMaxBytes, 16 * 1024 * 1024);

  const defaultCache = new ImmutableHttpfsCache({});
  assert.equal(defaultCache.slabMaxBytes, 192 * 1024 * 1024);
  assert.equal(defaultCache.maxBytes, 128 * 1024 * 1024);

  const lowMemCache = new ImmutableHttpfsCache({ slabProfile: 'low-memory' });
  assert.equal(lowMemCache.slabSize, 1024 * 1024);
  assert.equal(lowMemCache.slabMaxBytes, 16 * 1024 * 1024);

  const explicitSmall = new ImmutableHttpfsCache({ slabSize: 64 * 1024, slabMaxBytes: 256 * 1024 });
  assert.equal(explicitSmall.slabSize, 64 * 1024);
  assert.equal(explicitSmall.slabMaxBytes, 256 * 1024);

  // 2. Proportional shaders.rpf working-set test (scale 1:1024 -> 412 KiB archive, 116 KiB random
  //    working set = 58 slabs of 2 KiB; 192 KiB budget = 96 slabs vs old 32 KiB budget = 16 slabs).
  const slabSize = 2048; // 2 KiB proportional to 2 MiB
  const archiveSize = 412 * 1024; // 412 KiB proportional to 412 MB shaders.rpf
  const workingSetSlabs = 58; // 116 KiB proportional to ~115 MB working set
  const shadersBytes = makePatternBytes(archiveSize, 0x6d);

  // Deterministic pseudo-random access schedule over 58 distinct slabs (300 total reads).
  const hotSlabs = [];
  for (let i = 0; i < workingSetSlabs; i += 1) {
    hotSlabs.push((i * 3 + 5) % Math.floor(archiveSize / slabSize));
  }
  const accessSchedule = [];
  for (let step = 0; step < 300; step += 1) {
    const slabIdx = hotSlabs[(step * 17 + Math.floor(step / 5)) % workingSetSlabs];
    const intraOff = ((step * 257) % (slabSize - 256)) & ~15;
    accessSchedule.push({ offset: slabIdx * slabSize + intraOff, length: 256 });
  }

  // Under the W255 proportional 192-unit budget (96 slabs >= 58 working-set slabs): 0 evictions!
  const warmMock = createMockRangeFetch(new Map([['/game/shaders.rpf', shadersBytes]]));
  const warmCache = new ImmutableHttpfsCache({
    slabSize,
    slabMaxBytes: 192 * 1024, // 96 slabs (proportional to 192 MiB)
    fetchImpl: warmMock.fetchImpl,
  });

  // Pass 1: warm every slab in the working set
  for (let i = 0; i < workingSetSlabs; i += 1) {
    const off = hotSlabs[i] * slabSize;
    const got = await warmCache.read('/game/shaders.rpf', off, 512);
    assert.deepEqual(got, shadersBytes.subarray(off, off + 512));
  }
  assert.equal(warmMock.requests.length, workingSetSlabs);
  assert.equal(warmCache.getStats().slabEvictions, 0);

  // Pass 2: 300 random reads across the warmed working set -> ZERO network fetches, ZERO evictions!
  for (const { offset, length } of accessSchedule) {
    const got = await warmCache.read('/game/shaders.rpf', offset, length);
    assert.deepEqual(got, shadersBytes.subarray(offset, offset + length));
  }
  assert.equal(
    warmMock.requests.length,
    workingSetSlabs,
    'Warmed working set must suffer 0 re-fetches under 192-unit slab LRU budget',
  );
  const warmStats = warmCache.getStats();
  assert.equal(warmStats.slabEvictions, 0);
  assert.equal(warmStats.slabCacheHits, 300);
  assert.equal(warmStats.slabEntries, workingSetSlabs);
  assert.equal(warmStats.entries, 0);

  // Guard: slabMaxBytes < slabSize falls back safely to chunk mode instead of dropping every slab.
  const undersizedMock = createMockRangeFetch(new Map([['/game/tiny_cap.rpf', shadersBytes.subarray(0, 8192)]]));
  const undersizedCache = new ImmutableHttpfsCache({
    chunkSize: 512,
    maxBytes: 8192,
    slabSize: 4096,
    slabMaxBytes: 2048, // smaller than 1 slab -> safely disables slab mode
    fetchImpl: undersizedMock.fetchImpl,
  });
  await undersizedCache.open('/game/tiny_cap.rpf');
  const u1 = await undersizedCache.read('/game/tiny_cap.rpf', 0, 512);
  assert.deepEqual(u1, shadersBytes.subarray(0, 512));
  assert.equal(undersizedMock.requests.length, 1, 'Fallback chunk cache must retain fetched chunk 0');
});

// ---------------------------------------------------------------------------
// 11. W298 (#1205 / Refs #1015, #1077): Versioned OPFS L2 write-through for
//     hot .rpf ranges in ImmutableHttpfsCache & IoWorkerController
// ---------------------------------------------------------------------------

function createMockEtagRangeFetch(urlEntriesMap, options = {}) {
  // urlEntriesMap: Map<url, { bytes: Uint8Array, etag?: string|null, omitTotal?: boolean }>
  const requests = [];
  const delayMs = options.delayMs || 0;

  async function fetchImpl(url, init = {}) {
    const method = (init && init.method) || 'GET';
    const rangeHeader = (init && init.headers && (init.headers.Range || init.headers.range)) || null;
    requests.push({ method, url, rangeHeader });

    if (delayMs > 0) {
      await new Promise((r) => setTimeout(r, delayMs));
    }

    const rec = urlEntriesMap.get(url);
    if (!rec || !rec.bytes) {
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

    const data = rec.bytes;
    const headers = new Map();
    if (typeof rec.etag === 'string' && rec.etag.length > 0) {
      headers.set('ETag', rec.etag);
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
    const totalToken = rec.omitTotal ? '*' : String(data.byteLength);
    headers.set('Content-Range', `bytes ${reqStart}-${actualEnd}/${totalToken}`);
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

function createMockOpfsL2Store(options = {}) {
  const entries = new Map(); // `${normalizedPath}::${offset}::${length}` -> { bytes, fileFingerprint }
  const readCalls = [];
  const writeCalls = [];
  const invalidateCalls = [];
  let gateWrites = Boolean(options.gateWrites);
  const writeGates = [];
  let throwOnRead = options.throwOnRead || null;
  let throwOnWrite = options.throwOnWrite || null;

  function makeKey(normalizedPath, offset, length) {
    return `${normalizedPath}::${offset}::${length}`;
  }

  return {
    entries,
    readCalls,
    writeCalls,
    invalidateCalls,
    setGateWrites(val) {
      gateWrites = Boolean(val);
    },
    releaseAllWriteGates() {
      gateWrites = false;
      while (writeGates.length > 0) {
        writeGates.shift()();
      }
    },
    releaseOneWriteGate() {
      if (writeGates.length > 0) {
        writeGates.shift()();
        return true;
      }
      return false;
    },
    pendingWriteGateCount() {
      return writeGates.length;
    },
    setThrowOnRead(err) {
      throwOnRead = err || null;
    },
    setThrowOnWrite(err) {
      throwOnWrite = err || null;
    },
    async readRange(normalizedPath, offset, length, fileFingerprint) {
      readCalls.push({ normalizedPath, offset, length, fileFingerprint });
      if (throwOnRead) {
        throw typeof throwOnRead === 'function' ? throwOnRead() : throwOnRead;
      }
      const k = makeKey(normalizedPath, offset, length);
      const rec = entries.get(k);
      if (!rec) {
        return null;
      }
      if (fileFingerprint && rec.fileFingerprint !== fileFingerprint) {
        entries.delete(k);
        return { hit: false, invalidated: true, bytes: null };
      }
      return { hit: true, bytes: rec.bytes.slice(), fileFingerprint: rec.fileFingerprint };
    },
    async writeRange(normalizedPath, offset, length, bytes, fileFingerprint) {
      writeCalls.push({ normalizedPath, offset, length, byteLength: bytes.byteLength, fileFingerprint });
      if (gateWrites) {
        await new Promise((resolve) => writeGates.push(resolve));
      }
      if (throwOnWrite) {
        throw typeof throwOnWrite === 'function' ? throwOnWrite() : throwOnWrite;
      }
      const k = makeKey(normalizedPath, offset, length);
      entries.set(k, { bytes: bytes.slice(), fileFingerprint });
      return { stored: true };
    },
    async invalidateFingerprint(normalizedPath, expectedFingerprint) {
      invalidateCalls.push({ normalizedPath, expectedFingerprint });
      const prefix = `${normalizedPath}::`;
      let invalidatedCount = 0;
      for (const [k, rec] of Array.from(entries.entries())) {
        if (k.startsWith(prefix) && rec.fileFingerprint !== expectedFingerprint) {
          entries.delete(k);
          invalidatedCount += 1;
        }
      }
      return { invalidatedCount };
    },
  };
}

function createMockOpfsDirHandle(options = {}) {
  const files = new Map();
  const maxQuotaBytes = Number.isFinite(options.maxQuotaBytes) ? options.maxQuotaBytes : Infinity;
  const removed = [];

  function usedBytes(excludeName = null) {
    let total = 0;
    for (const [name, buf] of files.entries()) {
      if (name === excludeName) continue;
      total += buf.byteLength;
    }
    return total;
  }

  function makeHandle(name) {
    return {
      kind: 'file',
      name,
      async getFile() {
        const cur = files.get(name) || new Uint8Array(0);
        return {
          size: cur.byteLength,
          lastModified: 1,
          async arrayBuffer() {
            return cur.slice().buffer;
          },
        };
      },
      async createWritable() {
        let pending = new Uint8Array(0);
        let aborted = false;
        return {
          async write(chunk) {
            const bytes = chunk instanceof Uint8Array ? chunk.slice() : new Uint8Array(chunk);
            if (usedBytes(name) + bytes.byteLength > maxQuotaBytes) {
              const err = new Error('The quota has been exceeded.');
              err.name = 'QuotaExceededError';
              throw err;
            }
            pending = bytes;
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
      },
    };
  }

  return {
    files,
    removed,
    async getFileHandle(name, opts = {}) {
      if (!files.has(name)) {
        if (!opts.create) {
          const err = new Error(`NotFoundError: ${name}`);
          err.name = 'NotFoundError';
          throw err;
        }
        files.set(name, new Uint8Array(0));
      }
      return makeHandle(name);
    },
    async removeEntry(name) {
      removed.push(name);
      if (!files.delete(name)) {
        const err = new Error(`NotFoundError: ${name}`);
        err.name = 'NotFoundError';
        throw err;
      }
    },
    async *entries() {
      for (const name of files.keys()) {
        yield [name, makeHandle(name)];
      }
    },
    async *keys() {
      for (const name of files.keys()) {
        yield name;
      }
    },
  };
}

test('W298: fingerprint helpers, hot RPF allowlist, and zeroed OPFS L2 telemetry match W262 contracts', () => {
  assert.deepEqual(DEFAULT_HOT_OPFS_RPF_ALLOWLIST, ['shaders.rpf', 'core_bvh4.rpf']);
  assert.equal(DEFAULT_OPFS_L2_MAX_PENDING_WRITES, 64);
  assert.equal(DEFAULT_OPFS_L2_MAX_WRITE_CONCURRENCY, 4);

  // Bit-for-bit fingerprint parity with W262 range_cache.js.
  assert.equal(
    buildFileFingerprint(4194304, '"shaders-v1"'),
    rangeCacheMod.buildFileFingerprint(4194304, '"shaders-v1"'),
  );
  assert.equal(buildFileFingerprint(4194304), rangeCacheMod.buildFileFingerprint(4194304));
  assert.equal(buildFileFingerprint(null, '"etag-only"'), rangeCacheMod.buildFileFingerprint(null, '"etag-only"'));
  assert.equal(buildFileFingerprint(null, null), null);

  // Path normalization strips origin, query, and Xbox drive prefix while blocking traversal.
  assert.equal(normalizeOpfsL2Path('/game/shaders.rpf'), 'game/shaders.rpf');
  assert.equal(normalizeOpfsL2Path('http://127.0.0.1:8080/game/core_bvh4.rpf?v=1'), 'game/core_bvh4.rpf');
  assert.equal(normalizeOpfsL2Path('G:\\x64\\shaders.rpf'), 'x64/shaders.rpf');
  assert.equal(normalizeOpfsL2Path('../escape/shaders.rpf'), null);
  assert.equal(normalizeOpfsL2Path('game/bad\0.rpf'), null);

  // Default hot allowlist matches shaders.rpf and core_bvh4.rpf only.
  assert.equal(isHotOpfsL2Path('/game/shaders.rpf'), true);
  assert.equal(isHotOpfsL2Path('http://127.0.0.1:8080/game/CORE_BVH4.RPF'), true);
  assert.equal(isHotOpfsL2Path('/game/audio.rpf'), false);
  assert.equal(isHotOpfsL2Path('/game/movies.rpf'), false);

  // Custom allowlist configurations (array, path suffix, RegExp, predicate, wildcard, empty).
  assert.equal(isHotOpfsL2Path('/game/movies.rpf', ['movies.rpf']), true);
  assert.equal(isHotOpfsL2Path('/game/x64/levels.rpf', ['x64/levels.rpf']), true);
  assert.equal(isHotOpfsL2Path('/game/x64/levels.rpf', [/levels\.rpf$/i]), true);
  assert.equal(isHotOpfsL2Path('/game/audio.rpf', (p) => p.endsWith('audio.rpf')), true);
  assert.equal(isHotOpfsL2Path('/game/anything.rpf', '*'), true);
  assert.equal(isHotOpfsL2Path('/game/shaders.rpf', []), false);
  assert.equal(isHotOpfsL2Path('/game/shaders.rpf', false), false);

  // Versioned range filename matches opfs_cache.js `r_<offset>_<length>_<pathToken>_fp_<fp>.bin`.
  const fp = buildFileFingerprint(4096, '"etag-1"');
  const rangeFile = opfsL2RangeFileName('game/shaders.rpf', 0, 2048, fp);
  assert.match(rangeFile, /^r_0_2048_[0-9a-f]+_fp_4096_[0-9a-f]{16}\.bin$/);

  // Zeroed additive telemetry in emptyStats() & emptyImmutableCacheStats().
  const eTop = emptyStats();
  const eImm = emptyImmutableCacheStats();
  for (const field of ['opfsL2Hits', 'opfsL2Misses', 'opfsL2Writes', 'opfsL2Invalidations']) {
    assert.equal(eTop[field], 0, `emptyStats().${field} must default to 0`);
    assert.equal(eTop.immutableCache[field], 0, `emptyStats().immutableCache.${field} must default to 0`);
    assert.equal(eImm[field], 0, `emptyImmutableCacheStats().${field} must default to 0`);
  }
});

test('W298: cold boot populates OPFS L2 asynchronously (non-blocking) for hot RPFs and warm boot hits OPFS L2 skipping HTTP', async () => {
  const slabSize = 4096;
  const shadersBytes = makePatternBytes(slabSize * 2, 0x11); // 2 slabs
  const bvhBytes = makePatternBytes(slabSize, 0x22); // 1 slab
  const audioBytes = makePatternBytes(slabSize, 0x33); // non-hot RPF (not in allowlist)

  const shadersEtag = '"shaders-rpf-v1"';
  const bvhEtag = '"bvh4-rpf-v1"';
  const audioEtag = '"audio-rpf-v1"';

  const serverMap = new Map([
    ['/game/shaders.rpf', { bytes: shadersBytes, etag: shadersEtag }],
    ['/game/core_bvh4.rpf', { bytes: bvhBytes, etag: bvhEtag }],
    ['/game/audio.rpf', { bytes: audioBytes, etag: audioEtag }],
  ]);

  // Gate OPFS L2 writes to PROVE active cold reads never block on OPFS disk persistence!
  const l2Store = createMockOpfsL2Store({ gateWrites: true });
  const coldFetch = createMockEtagRangeFetch(serverMap);
  const coldController = new IoWorkerController({
    fetchImpl: coldFetch.fetchImpl,
    opfsDir: null,
    opfsL2: l2Store,
    slabSize,
    slabMaxBytes: slabSize * 16,
  });

  await coldController.init({ dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' });
  const reqsAfterInit = coldFetch.requests.length;

  const hShaders = await coldController.openHandle('G:\\shaders.rpf');
  const hBvh = await coldController.openHandle('G:\\core_bvh4.rpf');
  const hAudio = await coldController.openHandle('G:\\audio.rpf');
  assert.ok(hShaders >= 1 && hBvh >= 1 && hAudio >= 1);

  const s0 = await coldController.readHandle(hShaders, 0, 512);
  const s1 = await coldController.readHandle(hShaders, slabSize + 128, 512);
  const b0 = await coldController.readHandle(hBvh, 64, 256);
  const a0 = await coldController.readHandle(hAudio, 0, 256);

  assert.deepEqual(s0, shadersBytes.subarray(0, 512));
  assert.deepEqual(s1, shadersBytes.subarray(slabSize + 128, slabSize + 640));
  assert.deepEqual(b0, bvhBytes.subarray(64, 320));
  assert.deepEqual(a0, audioBytes.subarray(0, 256));

  // All reads completed IMMEDIATELY even while OPFS L2 writes remain gated!
  assert.equal(coldController.getStats().opfsL2Writes, 0);
  assert.equal(l2Store.pendingWriteGateCount(), 3, 'Expected 2 shaders.rpf slabs + 1 core_bvh4.rpf slab queued');
  assert.equal(coldFetch.requests.length - reqsAfterInit, 4);

  // Release write gates and drain background OPFS L2 queue.
  l2Store.releaseAllWriteGates();
  await coldController.flushOpfsL2Writes();

  const coldStats = coldController.getStats();
  assert.equal(coldStats.opfsL2Hits, 0);
  assert.equal(coldStats.opfsL2Misses, 3);
  assert.equal(coldStats.opfsL2Writes, 3);
  assert.equal(coldStats.opfsL2Invalidations, 0);
  assert.equal(coldStats.immutableCache.opfsL2Writes, 3);
  assert.equal(coldStats.immutableCache.opfsL2WriteErrors, 0);

  // Verify non-hot audio.rpf was NEVER written to OPFS L2.
  assert.equal(
    l2Store.writeCalls.some((w) => w.normalizedPath.includes('audio.rpf')),
    false,
    'Non-allowlisted audio.rpf must not pollute OPFS L2',
  );
  // Verify stored fingerprints match buildFileFingerprint(size, etag).
  const expectedShadersFp = buildFileFingerprint(shadersBytes.byteLength, shadersEtag);
  const expectedBvhFp = buildFileFingerprint(bvhBytes.byteLength, bvhEtag);
  assert.equal(l2Store.entries.get(`game/shaders.rpf::0::${slabSize}`).fileFingerprint, expectedShadersFp);
  assert.equal(l2Store.entries.get(`game/shaders.rpf::${slabSize}::${slabSize}`).fileFingerprint, expectedShadersFp);
  assert.equal(l2Store.entries.get(`game/core_bvh4.rpf::0::${slabSize}`).fileFingerprint, expectedBvhFp);

  // --- Warm Boot #1: Pre-seeded fingerprints -> 100% OPFS L2 hits, ZERO HTTP fetches! ---
  const warmFetch1 = createMockEtagRangeFetch(serverMap);
  const warmController1 = new IoWorkerController({
    fetchImpl: warmFetch1.fetchImpl,
    opfsDir: null,
    opfsL2: l2Store,
    slabSize,
    slabMaxBytes: slabSize * 16,
    fileFingerprints: {
      'game/shaders.rpf': expectedShadersFp,
      'game/core_bvh4.rpf': expectedBvhFp,
    },
  });
  await warmController1.init({ dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' });
  const warm1ReqsAfterInit = warmFetch1.requests.length;

  const whShaders = await warmController1.openHandle('G:\\shaders.rpf');
  const whBvh = await warmController1.openHandle('G:\\core_bvh4.rpf');
  assert.equal(warmController1.getHandleInfo(whShaders).size, shadersBytes.byteLength);
  assert.equal(warmController1.getHandleInfo(whBvh).size, bvhBytes.byteLength);

  assert.deepEqual(await warmController1.readHandle(whShaders, 0, 512), shadersBytes.subarray(0, 512));
  assert.deepEqual(
    await warmController1.readHandle(whShaders, slabSize + 128, 512),
    shadersBytes.subarray(slabSize + 128, slabSize + 640),
  );
  assert.deepEqual(await warmController1.readHandle(whBvh, 64, 256), bvhBytes.subarray(64, 320));

  assert.equal(
    warmFetch1.requests.length - warm1ReqsAfterInit,
    0,
    'Warm boot with matching OPFS L2 fingerprints must skip all HTTP Range fetches',
  );
  const warm1Stats = warmController1.getStats();
  assert.equal(warm1Stats.opfsL2Hits, 3);
  assert.equal(warm1Stats.opfsL2Misses, 0);
  assert.equal(warm1Stats.opfsL2Writes, 0);
  assert.equal(warm1Stats.opfsL2Invalidations, 0);
  assert.equal(warm1Stats.immutableCache.networkFetches, 0);
  assert.ok(warm1Stats.slabCacheHits >= 2, 'Repeat reads on L2-warmed slabs must hit RAM L1');

  // Telemetry sanitization check: zero paths, URLs, or ETag strings leaked.
  const statsJson = JSON.stringify(warm1Stats);
  assert.equal(statsJson.includes('shaders.rpf'), false);
  assert.equal(statsJson.includes('core_bvh4.rpf'), false);
  assert.equal(statsJson.includes('shaders-rpf-v1'), false);

  // --- Warm Boot #2: No pre-seeded fingerprints -> slab 0 probes HTTP to learn ETag+size, slab 1 hits OPFS L2! ---
  const warmFetch2 = createMockEtagRangeFetch(serverMap);
  const warmController2 = new IoWorkerController({
    fetchImpl: warmFetch2.fetchImpl,
    opfsDir: null,
    opfsL2: l2Store,
    slabSize,
    slabMaxBytes: slabSize * 16,
  });
  await warmController2.init({ dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' });
  const warm2ReqsAfterInit = warmFetch2.requests.length;

  const wh2 = await warmController2.openHandle('G:\\shaders.rpf');
  assert.deepEqual(await warmController2.readHandle(wh2, 0, 256), shadersBytes.subarray(0, 256));
  assert.deepEqual(
    await warmController2.readHandle(wh2, slabSize, 256),
    shadersBytes.subarray(slabSize, slabSize + 256),
  );
  await warmController2.flushOpfsL2Writes();

  assert.equal(
    warmFetch2.requests.length - warm2ReqsAfterInit,
    1,
    'Only slab 0 probes HTTP to learn ETag+size; slab 1 must hit OPFS L2 with 0 HTTP fetches',
  );
  const warm2Stats = warmController2.getStats();
  assert.equal(warm2Stats.opfsL2Hits, 1);
  assert.equal(warm2Stats.opfsL2Misses, 1);
  assert.equal(warm2Stats.opfsL2Invalidations, 0);
});

test('W298: ETag and size fileFingerprint mismatches automatically invalidate stale OPFS L2 entries and re-fetch fresh bytes', async () => {
  const slabSize = 2048;
  const v1Bytes = makePatternBytes(slabSize * 2, 0x10);
  const v2Bytes = makePatternBytes(slabSize * 2, 0x20); // same size, different ETag!
  const v3Bytes = makePatternBytes(slabSize * 2 + 512, 0x30); // different size + different ETag!
  const v4Bytes = makePatternBytes(slabSize * 2 + 512, 0x40); // mid-session ETag rotation!

  const l2Store = createMockOpfsL2Store();
  const fpV1 = buildFileFingerprint(v1Bytes.byteLength, '"rpf-v1"');
  const fpV2 = buildFileFingerprint(v2Bytes.byteLength, '"rpf-v2"');
  const fpV3 = buildFileFingerprint(v3Bytes.byteLength, '"rpf-v3"');
  const fpV4 = buildFileFingerprint(v4Bytes.byteLength, '"rpf-v4"');

  // 1. Cold seed v1 into OPFS L2.
  const fetchV1 = createMockEtagRangeFetch(
    new Map([['/game/shaders.rpf', { bytes: v1Bytes, etag: '"rpf-v1"' }]]),
  );
  const cacheV1 = new ImmutableHttpfsCache({
    fetchImpl: fetchV1.fetchImpl,
    opfsL2: l2Store,
    slabSize,
    slabMaxBytes: slabSize * 8,
  });
  assert.deepEqual(await cacheV1.read('/game/shaders.rpf', 0, slabSize * 2), v1Bytes);
  await cacheV1.flushOpfsL2Writes();
  assert.equal(cacheV1.getStats().opfsL2Writes, 2);
  assert.equal(l2Store.entries.get(`game/shaders.rpf::0::${slabSize}`).fileFingerprint, fpV1);

  // 2. Warm boot with same size but NEW ETag (v2) pre-seeded -> readRange detects stale v1 & invalidates!
  const fetchV2 = createMockEtagRangeFetch(
    new Map([['/game/shaders.rpf', { bytes: v2Bytes, etag: '"rpf-v2"' }]]),
  );
  const cacheV2 = new ImmutableHttpfsCache({
    fetchImpl: fetchV2.fetchImpl,
    opfsL2: l2Store,
    slabSize,
    slabMaxBytes: slabSize * 8,
    fileFingerprints: { 'game/shaders.rpf': fpV2 },
  });
  const gotV2 = await cacheV2.read('/game/shaders.rpf', 0, slabSize * 2);
  await cacheV2.flushOpfsL2Writes();
  assert.deepEqual(gotV2, v2Bytes);
  assert.equal(fetchV2.requests.length, 2, 'Mismatched ETag fingerprint must re-fetch both slabs over HTTP');
  const statsV2 = cacheV2.getStats();
  assert.equal(statsV2.opfsL2Hits, 0);
  assert.equal(statsV2.opfsL2Misses, 2);
  assert.equal(statsV2.opfsL2Invalidations, 2);
  assert.equal(statsV2.opfsL2Writes, 2);
  assert.equal(l2Store.entries.get(`game/shaders.rpf::0::${slabSize}`).fileFingerprint, fpV2);
  assert.equal(l2Store.entries.get(`game/shaders.rpf::${slabSize}::${slabSize}`).fileFingerprint, fpV2);

  // 3. Warm boot WITHOUT pre-seeded fingerprint when archive size+ETag changed (v2 -> v3):
  //    slab 0 learns fpV3 from HTTP and calls invalidateFingerprint removing stale v2 entries!
  const fetchV3 = createMockEtagRangeFetch(
    new Map([['/game/shaders.rpf', { bytes: v3Bytes, etag: '"rpf-v3"' }]]),
  );
  const cacheV3 = new ImmutableHttpfsCache({
    fetchImpl: fetchV3.fetchImpl,
    opfsL2: l2Store,
    slabSize,
    slabMaxBytes: slabSize * 8,
  });
  const s0V3 = await cacheV3.read('/game/shaders.rpf', 0, 512);
  await cacheV3.flushOpfsL2Writes();
  assert.deepEqual(s0V3, v3Bytes.subarray(0, 512));
  assert.ok(
    cacheV3.getStats().opfsL2Invalidations >= 2,
    `Expected stale v2 entries invalidated on v3 learn, got ${cacheV3.getStats().opfsL2Invalidations}`,
  );
  assert.equal(
    l2Store.entries.has(`game/shaders.rpf::${slabSize}::${slabSize}`),
    false,
    'Stale v2 slab 1 must have been purged by invalidateFingerprint',
  );
  const s1V3 = await cacheV3.read('/game/shaders.rpf', slabSize, slabSize + 512);
  await cacheV3.flushOpfsL2Writes();
  assert.deepEqual(s1V3, v3Bytes.subarray(slabSize, v3Bytes.byteLength));
  assert.equal(l2Store.entries.get(`game/shaders.rpf::0::${slabSize}`).fileFingerprint, fpV3);

  // 4. Mid-session ETag rotation (v3 -> v4) on live server evicts stale RAM slabs and invalidates OPFS L2!
  const dynamicEntries = new Map([['/game/shaders.rpf', { bytes: v3Bytes, etag: '"rpf-v3"' }]]);
  const dynamicFetch = createMockEtagRangeFetch(dynamicEntries);
  const liveCache = new ImmutableHttpfsCache({
    fetchImpl: dynamicFetch.fetchImpl,
    opfsL2: l2Store,
    slabSize,
    slabMaxBytes: slabSize * 8,
  });
  // Read slab 0 from v3 (hits OPFS L2 -> populates RAM L1).
  liveCache.setFileFingerprint('game/shaders.rpf', fpV3);
  assert.deepEqual(await liveCache.read('/game/shaders.rpf', 0, 256), v3Bytes.subarray(0, 256));
  assert.equal(liveCache.getStats().opfsL2Hits, 1);

  // Server rotates file to v4; remove slab 1 from OPFS L2 so slab 1 hits HTTP and observes ETag "rpf-v4".
  dynamicEntries.set('/game/shaders.rpf', { bytes: v4Bytes, etag: '"rpf-v4"' });
  l2Store.entries.delete(`game/shaders.rpf::${slabSize}::${slabSize}`);
  assert.deepEqual(await liveCache.read('/game/shaders.rpf', slabSize, 256), v4Bytes.subarray(slabSize, slabSize + 256));
  await liveCache.flushOpfsL2Writes();
  assert.ok(liveCache.getStats().opfsL2Invalidations >= 1);
  assert.equal(liveCache.getFileFingerprint('/game/shaders.rpf'), fpV4);

  // Re-reading slab 0 must NOT return evicted stale v3 RAM bytes; it must fetch fresh v4 bytes!
  assert.deepEqual(await liveCache.read('/game/shaders.rpf', 0, 256), v4Bytes.subarray(0, 256));
});

test('W298: graceful fallback on OPFS quota/read/write errors, missing ETag, and bounded write backpressure', async () => {
  const slabSize = 2048;
  const rpfBytes = makePatternBytes(slabSize * 6, 0x5a);

  // 1. Throwing OPFS L2 store (QuotaExceededError on write, SecurityError on read) never fails HTTP reads.
  const quotaErr = new Error('The quota has been exceeded.');
  quotaErr.name = 'QuotaExceededError';
  const secErr = new Error('OPFS read denied');
  secErr.name = 'SecurityError';
  const failingL2 = createMockOpfsL2Store({ throwOnRead: secErr, throwOnWrite: quotaErr });
  const fetch1 = createMockEtagRangeFetch(
    new Map([['/game/shaders.rpf', { bytes: rpfBytes, etag: '"q-v1"' }]]),
  );
  const cache1 = new ImmutableHttpfsCache({
    fetchImpl: fetch1.fetchImpl,
    opfsL2: failingL2,
    slabSize,
    slabMaxBytes: slabSize * 8,
    fileFingerprints: { 'game/shaders.rpf': buildFileFingerprint(rpfBytes.byteLength, '"q-v1"') },
  });

  const got = await cache1.read('/game/shaders.rpf', 0, slabSize * 2);
  await cache1.flushOpfsL2Writes();
  assert.deepEqual(got, rpfBytes.subarray(0, slabSize * 2));
  const st1 = cache1.getStats();
  assert.equal(st1.opfsL2Hits, 0);
  assert.equal(st1.opfsL2Misses, 2);
  assert.equal(st1.opfsL2Writes, 0);
  assert.equal(st1.opfsL2WriteErrors, 2);

  // 2. Missing ETag fallback:
  //    (a) Default mode falls back safely to size-only fingerprint when size is known.
  const noEtagL2 = createMockOpfsL2Store();
  const noEtagFetch = createMockEtagRangeFetch(
    new Map([['/game/core_bvh4.rpf', { bytes: rpfBytes.subarray(0, slabSize), etag: null }]]),
  );
  const noEtagCache = new ImmutableHttpfsCache({
    fetchImpl: noEtagFetch.fetchImpl,
    opfsL2: noEtagL2,
    slabSize,
    slabMaxBytes: slabSize * 4,
  });
  assert.deepEqual(
    await noEtagCache.read('/game/core_bvh4.rpf', 0, 512),
    rpfBytes.subarray(0, 512),
  );
  await noEtagCache.flushOpfsL2Writes();
  assert.equal(noEtagCache.getFileFingerprint('/game/core_bvh4.rpf'), String(slabSize));
  assert.equal(noEtagCache.getStats().opfsL2Writes, 1);

  //    (b) Strict opfsL2RequireEtag: true OR unknown total (`bytes 0-N/*`) with no ETag -> skips OPFS L2 safely!
  const strictL2 = createMockOpfsL2Store();
  const strictCache = new ImmutableHttpfsCache({
    fetchImpl: noEtagFetch.fetchImpl,
    opfsL2: strictL2,
    opfsL2RequireEtag: true,
    slabSize,
    slabMaxBytes: slabSize * 4,
  });
  assert.deepEqual(
    await strictCache.read('/game/core_bvh4.rpf', 0, 512),
    rpfBytes.subarray(0, 512),
  );
  await strictCache.flushOpfsL2Writes();
  assert.equal(strictCache.getFileFingerprint('/game/core_bvh4.rpf'), null);
  assert.equal(strictCache.getStats().opfsL2Writes, 0);

  const unknownTotalL2 = createMockOpfsL2Store();
  const unknownTotalFetch = createMockEtagRangeFetch(
    new Map([['/game/core_bvh4.rpf', { bytes: rpfBytes.subarray(0, slabSize), etag: null, omitTotal: true }]]),
  );
  const unknownTotalCache = new ImmutableHttpfsCache({
    fetchImpl: unknownTotalFetch.fetchImpl,
    opfsL2: unknownTotalL2,
    slabSize,
    slabMaxBytes: slabSize * 4,
  });
  assert.deepEqual(
    await unknownTotalCache.read('/game/core_bvh4.rpf', 0, 512),
    rpfBytes.subarray(0, 512),
  );
  await unknownTotalCache.flushOpfsL2Writes();
  assert.equal(unknownTotalCache.getFileFingerprint('/game/core_bvh4.rpf'), null);
  assert.equal(unknownTotalCache.getStats().opfsL2Writes, 0);

  // 3. Bounded background write queue drops oldest writes under backpressure without stalling reads.
  const boundedL2 = createMockOpfsL2Store({ gateWrites: true });
  const boundedFetch = createMockEtagRangeFetch(
    new Map([['/game/shaders.rpf', { bytes: rpfBytes, etag: '"burst-v1"' }]]),
  );
  const boundedCache = new ImmutableHttpfsCache({
    fetchImpl: boundedFetch.fetchImpl,
    opfsL2: boundedL2,
    slabSize,
    slabMaxBytes: slabSize * 8,
    maxPendingOpfsL2Writes: 2,
    maxOpfsL2WriteConcurrency: 1,
  });

  // Burst 6 slab reads while OPFS writes are stalled: all 6 reads complete immediately!
  for (let s = 0; s < 6; s += 1) {
    const out = await boundedCache.read('/game/shaders.rpf', s * slabSize, 256);
    assert.deepEqual(out, rpfBytes.subarray(s * slabSize, s * slabSize + 256));
  }
  boundedL2.releaseAllWriteGates();
  await boundedCache.flushOpfsL2Writes();
  const boundedStats = boundedCache.getStats();
  assert.ok(
    boundedStats.opfsL2Writes >= 1 && boundedStats.opfsL2Writes <= 3,
    `Expected bounded queue (1 active + 2 queued) to cap persisted writes to 1..3, got ${boundedStats.opfsL2Writes}`,
  );
});

test('W298: interoperates with OpfsChunkCache, raw opfsDir (with quota eviction), chunk mode, and W260 8-lane mailbox pool', async () => {
  const slabSize = 2048;
  const shadersV1 = makePatternBytes(slabSize * 2, 0x71);
  const shadersV2 = makePatternBytes(slabSize * 2, 0x72);
  const fpV1 = buildFileFingerprint(shadersV1.byteLength, '"opfs-v1"');
  const fpV2 = buildFileFingerprint(shadersV2.byteLength, '"opfs-v2"');

  // --- Part A: OpfsChunkCache (web/storage/opfs_cache.js) interoperability ---
  const backingDir = createMockOpfsDirHandle();
  const opfsCache = new OpfsChunkCache({ cacheDir: backingDir, indexFlushDelayMs: 0 });
  const fetchA1 = createMockEtagRangeFetch(
    new Map([['/game/shaders.rpf', { bytes: shadersV1, etag: '"opfs-v1"' }]]),
  );
  const coldA = new ImmutableHttpfsCache({
    fetchImpl: fetchA1.fetchImpl,
    opfsCache,
    slabSize,
    slabMaxBytes: slabSize * 4,
  });
  assert.deepEqual(await coldA.read('/game/shaders.rpf', 0, slabSize * 2), shadersV1);
  await coldA.flushOpfsL2Writes();
  assert.equal(coldA.getStats().opfsL2Writes, 2);

  // Warm hit from OpfsChunkCache with 0 HTTP fetches.
  const fetchA2 = createMockEtagRangeFetch(
    new Map([['/game/shaders.rpf', { bytes: shadersV1, etag: '"opfs-v1"' }]]),
  );
  const warmA = new ImmutableHttpfsCache({
    fetchImpl: fetchA2.fetchImpl,
    opfsCache,
    slabSize,
    slabMaxBytes: slabSize * 4,
    fileFingerprints: { 'game/shaders.rpf': fpV1 },
  });
  assert.deepEqual(await warmA.read('/game/shaders.rpf', 0, slabSize * 2), shadersV1);
  assert.equal(fetchA2.requests.length, 0);
  assert.equal(warmA.getStats().opfsL2Hits, 2);

  // Stale fingerprint invalidation through OpfsChunkCache.
  const fetchA3 = createMockEtagRangeFetch(
    new Map([['/game/shaders.rpf', { bytes: shadersV2, etag: '"opfs-v2"' }]]),
  );
  const staleA = new ImmutableHttpfsCache({
    fetchImpl: fetchA3.fetchImpl,
    opfsCache,
    slabSize,
    slabMaxBytes: slabSize * 4,
    fileFingerprints: { 'game/shaders.rpf': fpV2 },
  });
  assert.deepEqual(await staleA.read('/game/shaders.rpf', 0, slabSize * 2), shadersV2);
  await staleA.flushOpfsL2Writes();
  assert.equal(fetchA3.requests.length, 2);
  assert.equal(staleA.getStats().opfsL2Invalidations, 2);
  await opfsCache.close();

  // --- Part B: Raw opfsDir handle (with quota eviction retry) + chunk mode ---
  const rawDir = createMockOpfsDirHandle({ maxQuotaBytes: 256 });
  const chunkFile = makePatternBytes(192, 0x83); // 3 x 64B chunks
  const chunkFp = buildFileFingerprint(chunkFile.byteLength, '"chunk-v1"');
  const fetchB1 = createMockEtagRangeFetch(
    new Map([['/game/core_bvh4.rpf', { bytes: chunkFile, etag: '"chunk-v1"' }]]),
  );
  const chunkCold = new ImmutableHttpfsCache({
    fetchImpl: fetchB1.fetchImpl,
    opfsDir: rawDir,
    chunkSize: 64,
    maxBytes: 512,
    slabCacheEnabled: false,
  });
  assert.deepEqual(await chunkCold.read('/game/core_bvh4.rpf', 0, 192), chunkFile);
  await chunkCold.flushOpfsL2Writes();
  assert.equal(chunkCold.getStats().opfsL2Writes, 3);

  const fetchB2 = createMockEtagRangeFetch(
    new Map([['/game/core_bvh4.rpf', { bytes: chunkFile, etag: '"chunk-v1"' }]]),
  );
  const chunkWarm = new ImmutableHttpfsCache({
    fetchImpl: fetchB2.fetchImpl,
    opfsDir: rawDir,
    chunkSize: 64,
    maxBytes: 512,
    slabCacheEnabled: false,
    fileFingerprints: { 'game/core_bvh4.rpf': chunkFp },
  });
  assert.deepEqual(await chunkWarm.read('/game/core_bvh4.rpf', 0, 192), chunkFile);
  assert.equal(fetchB2.requests.length, 0, 'Chunk mode warm read must hit raw opfsDir with 0 HTTP fetches');
  assert.equal(chunkWarm.getStats().opfsL2Hits, 3);

  // Tight quota on raw opfsDir (fits 2 x 64B chunks, writing 3 sequentially) evicts oldest and retries cleanly!
  const tightDir = createMockOpfsDirHandle({ maxQuotaBytes: 128 });
  const tightFetch = createMockEtagRangeFetch(
    new Map([['/game/core_bvh4.rpf', { bytes: chunkFile, etag: '"chunk-q"' }]]),
  );
  const tightCache = new ImmutableHttpfsCache({
    fetchImpl: tightFetch.fetchImpl,
    opfsDir: tightDir,
    chunkSize: 64,
    maxBytes: 512,
    slabCacheEnabled: false,
    maxOpfsL2WriteConcurrency: 1,
  });
  for (let c = 0; c < 3; c += 1) {
    assert.deepEqual(
      await tightCache.read('/game/core_bvh4.rpf', c * 64, 64),
      chunkFile.subarray(c * 64, (c + 1) * 64),
    );
    await tightCache.flushOpfsL2Writes();
  }
  assert.equal(tightCache.getStats().opfsL2Writes, 3);
  assert.equal(tightCache.getStats().opfsL2WriteErrors, 0);
  assert.ok(tightDir.removed.length >= 1, 'QuotaExceededError on raw opfsDir must evict oldest entry and retry');

  // --- Part C: W260 8-lane HttpfsMailboxPoolHost sharing warm OPFS L2 with 0 HTTP fetches ---
  const poolL2 = createMockOpfsL2Store();
  const poolServer = new Map([['/game/shaders.rpf', { bytes: shadersV1, etag: '"opfs-v1"' }]]);
  const seedFetch = createMockEtagRangeFetch(poolServer);
  const seeder = new IoWorkerController({
    fetchImpl: seedFetch.fetchImpl,
    opfsDir: null,
    opfsL2: poolL2,
    slabSize,
    slabMaxBytes: slabSize * 4,
  });
  await seeder.init({ dataBase: '/data/', gameBase: '/game/' });
  const hSeed = await seeder.openHandle('G:\\shaders.rpf');
  await seeder.readHandle(hSeed, 0, slabSize * 2);
  await seeder.flushOpfsL2Writes();

  const warmPoolFetch = createMockEtagRangeFetch(poolServer);
  const poolController = new IoWorkerController({
    fetchImpl: warmPoolFetch.fetchImpl,
    opfsDir: null,
    opfsL2: poolL2,
    slabSize,
    slabMaxBytes: slabSize * 4,
    fileFingerprints: { 'game/shaders.rpf': fpV1 },
  });
  const laneSabs = Array.from({ length: 8 }, () => createHttpfsMailboxBuffer(8192));
  const laneClients = laneSabs.map((sab) => new HttpfsMailboxClient(sab));

  try {
    await poolController.init({ dataBase: '/data/', gameBase: '/game/' });
    await poolController.attachMailboxPool(laneSabs);
    assert.equal(poolController.mailboxLanes, 8);
    const reqsAfterPoolInit = warmPoolFetch.requests.length;

    // All 8 lanes concurrently open & read shaders.rpf from warm OPFS L2 with ZERO HTTP requests!
    await Promise.all(
      laneClients.map(async (client, idx) => {
        const handle = await client.openAsync('G:\\shaders.rpf');
        assert.ok(handle >= 1);
        const offset = (idx % 2) * slabSize + idx * 32;
        const dst = new Uint8Array(128);
        const gotLen = await client.readAsync(handle, BigInt(offset), dst, 128);
        assert.equal(gotLen, 128);
        assert.deepEqual(dst, shadersV1.subarray(offset, offset + 128));
        assert.equal(await client.closeAsync(handle), true);
      }),
    );

    assert.equal(
      warmPoolFetch.requests.length - reqsAfterPoolInit,
      0,
      '8-lane warm OPFS L2 mailbox reads must trigger 0 HTTP fetches',
    );
    const poolStats = poolController.getStats();
    assert.equal(poolStats.mailboxLanes, 8);
    assert.equal(poolStats.opfsL2Hits, 2, 'Concurrent 8-lane reads must coalesce onto 2 slab OPFS L2 hits');
    assert.equal(poolStats.immutableCache.networkFetches, 0);
  } finally {
    await poolController.detachMailbox();
  }
});
