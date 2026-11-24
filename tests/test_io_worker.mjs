import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
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
  DEFAULT_IMMUTABLE_CHUNK_SIZE,
  DEFAULT_IMMUTABLE_MAX_BYTES,
  ImmutableHttpfsCache,
  IoWorkerController,
  isImmutableHttpfsPath,
  parseContentRangeHeader,
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
  assert.equal(DEFAULT_IMMUTABLE_MAX_BYTES, 64 * 1024 * 1024);

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

    // 1. Open using Xbox drive path -> triggers 1 aligned 64 KiB HTTP 206 fetch (`bytes=0-65535`).
    const h1 = await client.openAsync('G:\\audio.rpf');
    assert.ok(h1 >= 1);
    assert.equal(controller.getHandleInfo(h1).size, rpfSize);
    assert.equal(mock.requests.length, reqsAfterInit + 1);
    assert.equal(mock.requests[reqsAfterInit].url, '/game/audio.rpf');
    assert.equal(mock.requests[reqsAfterInit].rangeHeader, 'bytes=0-65535');

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
    assert.equal(stats.misses, 1);
    assert.equal(stats.hits, 7);
    assert.equal(stats.bytesCached, 65536);
    assert.equal(stats.entries, 1);
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

  // Fire 4 simultaneous reads into uncached chunk 1 [65536..131072) -> must coalesce into 1 fetch!
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
    2,
    'Expected 1 fetch for chunk 0 + 1 coalesced fetch for chunk 1',
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
    assert.equal(statsMsg.stats.immutableCache.hits, 1);
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
