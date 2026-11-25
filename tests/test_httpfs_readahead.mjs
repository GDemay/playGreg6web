import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import path from 'node:path';
import { Worker as NodeWorker } from 'node:worker_threads';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');

const ioWorkerMod = require('../web/workers/io_worker.js');
const mailboxMod = require('../web/workers/httpfs_mailbox.js');

const {
  DEFAULT_WHOLE_FILE_MAX_BYTES,
  DEFAULT_SLAB_SIZE,
  DEFAULT_SLAB_MAX_BYTES,
  ImmutableHttpfsCache,
  IoWorkerController,
  isImmutableHttpfsPath,
  isWholeFileBufferedPath,
} = ioWorkerMod;

const {
  HTTPFS_DIRECTSTORAGE_CHUNK_BYTES,
  HTTPFS_READAHEAD_BASE_BYTES,
  HTTPFS_READAHEAD_MID_BYTES,
  HTTPFS_READAHEAD_MAX_BYTES,
  HTTPFS_DEFAULT_PIPELINE_SLOTS,
  HttpfsStatus,
  isVideoReadAheadPath,
  computeReadAheadWindowSize,
  createHttpfsMailboxBuffer,
  createHttpfsMailboxPool,
  HttpfsMailboxHost,
  HttpfsMailboxClient,
  HttpfsMailboxPoolHost,
  HttpfsMailboxPoolClient,
} = mailboxMod;

const utf8Encoder = new TextEncoder();

function makePatternBytes(length, seed = 41) {
  const out = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = ((i * 67) ^ ((i >> 6) * 31) ^ ((i >> 13) * 19) ^ seed) & 0xff;
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
    const rangeHeader =
      (init && init.headers && (init.headers.Range || init.headers.range)) || null;
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
// 1. Constants & predicates
// ---------------------------------------------------------------------------

test('exports expected PGIM whole-file and slab cache defaults and path predicates', () => {
  assert.equal(DEFAULT_WHOLE_FILE_MAX_BYTES, 256 * 1024 * 1024);
  assert.equal(DEFAULT_SLAB_SIZE, 2 * 1024 * 1024);
  assert.equal(DEFAULT_SLAB_MAX_BYTES, 192 * 1024 * 1024);

  assert.equal(isWholeFileBufferedPath('main/image/guest.pgim'), true);
  assert.equal(isWholeFileBufferedPath('boot/main/image/GUEST.PGIM'), true);
  assert.equal(isWholeFileBufferedPath('program-manifest.json'), true);
  assert.equal(isWholeFileBufferedPath('boot/program-manifest.json'), true);
  assert.equal(isWholeFileBufferedPath('boot/boot-options.json'), true);
  assert.equal(isWholeFileBufferedPath('boot-options.json'), true);
  assert.equal(isWholeFileBufferedPath('shaders/splash.wgsl.pack.reflect.json'), true);
  assert.equal(isWholeFileBufferedPath('shaders/SPLASH.WGSL.PACK.REFLECT.JSON'), true);
  assert.equal(isWholeFileBufferedPath('shaders/splash.wgsl.pack'), false);
  assert.equal(isWholeFileBufferedPath('appdata0.rpf'), false);
  assert.equal(isWholeFileBufferedPath('audio.ttpf'), false);
  assert.equal(isWholeFileBufferedPath('data-manifest.json'), false);

  // Immutable paths include .ttpf, .pack (.wgsl.pack), and .reflect.json in addition to
  // .rpf, .pgim, .dll, and MicrosoftGame.Config
  assert.equal(isImmutableHttpfsPath('x64/audio/dialogue.ttpf'), true);
  assert.equal(isImmutableHttpfsPath('x64/audio/SFX.TTPF'), true);
  assert.equal(isImmutableHttpfsPath('shaders/splash.wgsl.pack'), true);
  assert.equal(isImmutableHttpfsPath('shaders/SPLASH.WGSL.PACK'), true);
  assert.equal(isImmutableHttpfsPath('shaders/splash.wgsl.pack.reflect.json'), true);
  assert.equal(isImmutableHttpfsPath('appdata0.rpf'), true);
  assert.equal(isImmutableHttpfsPath('xgameruntime.dll'), true);
  assert.equal(isImmutableHttpfsPath('main/image/guest.pgim'), true);
});

// ---------------------------------------------------------------------------
// 2. Whole-file eager .pgim fetch: 1 HTTP fetch, exact 64 KiB reads, freed on CLOSE
// ---------------------------------------------------------------------------

test('multi-chunk .pgim read in 64 KiB steps triggers ONLY 1 HTTP fetch, serves exact bytes via sync fast path, and frees buffer on CLOSE', async () => {
  // 1.5 MiB + 12345 bytes -> 25 sequential 64 KiB chunks (24 full + 1 partial tail)
  const pgimSize = 24 * 65536 + 12345;
  const pgimBytes = makePatternBytes(pgimSize, 73);

  const mock = createMockRangeFetch(new Map([['/boot/main/image/guest.pgim', pgimBytes]]));
  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    opfsDir: null,
  });

  const sab = createHttpfsMailboxBuffer(64 * 1024);
  const client = new HttpfsMailboxClient(sab);

  try {
    await controller.init({ dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' });
    await controller.attachMailbox(sab);

    const reqsBefore = mock.requests.length;

    const h = await client.openAsync('boot/main/image/guest.pgim');
    assert.ok(h >= 1, `Expected valid PGIM handle, got ${h}`);
    assert.equal(controller.getHandleInfo(h).size, pgimSize);
    assert.equal(controller.getHandleInfo(h).wholeFileBuffered, true);

    // Exactly 1 HTTP fetch happened on OPEN.
    assert.equal(mock.requests.length - reqsBefore, 1);

    let statsOpen = controller.getStats();
    assert.equal(statsOpen.pgimFullFetches, 1);
    assert.equal(statsOpen.wholeFileCache.activeEntries, 1);
    assert.equal(statsOpen.wholeFileCache.activeBytes, pgimSize);

    // Stream the entire PGIM sequentially in 64 KiB steps (matching WasmBootConfig::read_chunk_bytes).
    const assembled = new Uint8Array(pgimSize);
    const chunkBuf = new Uint8Array(65536);
    let offset = 0;
    let readCalls = 0;
    while (offset < pgimSize) {
      const got = await client.readAsync(h, BigInt(offset), chunkBuf, 65536);
      assert.ok(got > 0, `Expected positive bytes at offset ${offset}, got ${got}`);
      const expectedLen = Math.min(65536, pgimSize - offset);
      assert.equal(got, expectedLen);
      assembled.set(chunkBuf.subarray(0, got), offset);
      offset += got;
      readCalls += 1;
    }

    assert.equal(readCalls, 25);
    assert.deepEqual(assembled, pgimBytes);

    // Past-EOF read returns 0 immediately without network calls.
    assert.equal(await client.readAsync(h, BigInt(pgimSize), chunkBuf, 65536), 0);
    assert.equal(await client.readAsync(h, BigInt(pgimSize + 4096), chunkBuf, 65536), 0);

    // STILL ONLY 1 HTTP FETCH for all 27 reads!
    assert.equal(
      mock.requests.length - reqsBefore,
      1,
      `Expected exactly 1 HTTP fetch for entire PGIM load, got ${mock.requests.length - reqsBefore}`,
    );

    const statsBeforeClose = controller.getStats();
    assert.equal(statsBeforeClose.pgimFullFetches, 1);
    assert.equal(statsBeforeClose.wholeFileCache.hits, 27);
    assert.ok(
      statsBeforeClose.syncFastPathHits >= 27,
      `Expected >= 27 syncFastPathHits, got ${statsBeforeClose.syncFastPathHits}`,
    );

    // CLOSE must immediately release the in-memory PGIM buffer!
    assert.equal(await client.closeAsync(h), true);
    const statsAfterClose = controller.getStats();
    assert.equal(statsAfterClose.wholeFileCache.activeEntries, 0);
    assert.equal(statsAfterClose.wholeFileCache.activeBytes, 0);
    assert.equal(statsAfterClose.wholeFileCache.releases, 1);
    assert.equal(statsAfterClose.openHandles, 0);

    // Post-close read returns -9 (EBADF) cleanly.
    assert.equal(await client.readAsync(h, 0n, chunkBuf, 64), -9);

    // Verify sanitized stats contain zero paths/URLs.
    const statsJson = JSON.stringify(statsAfterClose);
    assert.equal(statsJson.includes('guest.pgim'), false);
    assert.equal(statsJson.includes('/boot/'), false);
  } finally {
    await controller.detachMailbox();
  }
});

test('program-manifest.json eager buffering, concurrent dual-handle refcounting, and lazy probeOnOpen=false buffering', async () => {
  const manifestBytes = utf8Encoder.encode(
    JSON.stringify({ version: 1, entry: 'main/image/guest.pgim', modules: ['main', 'msvcp140'] }),
  );
  const pgimBytes = makePatternBytes(200 * 1024, 19);

  const mock = createMockRangeFetch(
    new Map([
      ['/boot/program-manifest.json', manifestBytes],
      ['/boot/main/image/guest.pgim', pgimBytes],
    ]),
    { delayMs: 10 },
  );

  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    opfsDir: null,
  });
  await controller.init({ dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' });

  const reqsBefore = mock.requests.length;

  // 1. Concurrent opens of the same .pgim coalesce into 1 HTTP fetch and share refcounted buffer.
  const [h1, h2] = await Promise.all([
    controller.openHandle('boot/main/image/guest.pgim'),
    controller.openHandle('main/image/guest.pgim'),
  ]);
  assert.ok(h1 >= 1 && h2 >= 1);
  assert.equal(mock.requests.length - reqsBefore, 1);
  assert.equal(controller.getStats().pgimFullFetches, 1);
  assert.equal(controller.getStats().wholeFileCache.activeEntries, 1);

  // Close h1 -> buffer MUST remain live for h2!
  assert.equal(controller.closeHandle(h1), true);
  assert.equal(controller.getStats().wholeFileCache.activeEntries, 1);
  assert.equal(controller.getStats().wholeFileCache.releases, 0);

  const sliceFromH2 = await controller.readHandle(h2, 65536, 4096);
  assert.deepEqual(sliceFromH2, pgimBytes.subarray(65536, 65536 + 4096));
  // Mutate returned buffer to confirm isolation from retained whole-file buffer.
  sliceFromH2.fill(0xff);
  const verifyH2 = await controller.readHandle(h2, 65536, 4096);
  assert.deepEqual(verifyH2, pgimBytes.subarray(65536, 65536 + 4096));

  // Close h2 -> last ref released, buffer freed!
  assert.equal(controller.closeHandle(h2), true);
  assert.equal(controller.getStats().wholeFileCache.activeEntries, 0);
  assert.equal(controller.getStats().wholeFileCache.activeBytes, 0);
  assert.equal(controller.getStats().wholeFileCache.releases, 1);

  // 2. program-manifest.json whole-file buffered with probeOnOpen=false (lazy on first READ).
  const reqsBeforeManifest = mock.requests.length;
  const hMan = await controller.openHandle('boot/program-manifest.json', { probeOnOpen: false });
  assert.ok(hMan >= 1);
  assert.equal(mock.requests.length, reqsBeforeManifest, 'probeOnOpen=false must not fetch on OPEN');

  const firstPart = await controller.readHandle(hMan, 0, 16);
  assert.deepEqual(firstPart, manifestBytes.subarray(0, 16));
  assert.equal(mock.requests.length, reqsBeforeManifest + 1);

  const restPart = await controller.readHandle(hMan, 16, manifestBytes.byteLength);
  assert.deepEqual(restPart, manifestBytes.subarray(16));
  assert.equal(mock.requests.length, reqsBeforeManifest + 1, 'Subsequent read must hit whole-file buffer');

  assert.equal(controller.closeHandle(hMan), true);
  assert.equal(controller.getStats().wholeFileCache.activeEntries, 0);
});

test('oversized .pgim exceeding wholeFileMaxBytes falls back cleanly to bounded LRU slab cache', async () => {
  const pgimSize = 512 * 1024; // 512 KiB
  const pgimBytes = makePatternBytes(pgimSize, 105);

  const mock = createMockRangeFetch(new Map([['/boot/main/image/huge.pgim', pgimBytes]]));
  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    wholeFileMaxBytes: 128 * 1024, // 128 KiB cap < 512 KiB file
    slabSize: 256 * 1024, // 256 KiB slabs
    slabMaxBytes: 512 * 1024,
    opfsDir: null,
  });

  const sab = createHttpfsMailboxBuffer(64 * 1024);
  const client = new HttpfsMailboxClient(sab);

  try {
    await controller.init({ dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' });
    await controller.attachMailbox(sab);

    const reqsBefore = mock.requests.length;
    const h = await client.openAsync('boot/main/image/huge.pgim');
    assert.ok(h >= 1);
    assert.equal(controller.getHandleInfo(h).size, pgimSize);
    assert.equal(controller.getHandleInfo(h).wholeFileBuffered, false);
    assert.equal(controller.getStats().wholeFileCache.fallbacks, 1);
    assert.equal(controller.getStats().wholeFileCache.activeEntries, 0);
    assert.equal(controller.getStats().pgimFullFetches, 0);

    // Read all 8 x 64 KiB chunks sequentially -> should coalesce into only 2 x 256 KiB slab fetches!
    const out = new Uint8Array(pgimSize);
    const buf = new Uint8Array(65536);
    for (let i = 0; i < 8; i += 1) {
      const got = await client.readAsync(h, BigInt(i * 65536), buf, 65536);
      assert.equal(got, 65536);
      out.set(buf, i * 65536);
    }
    assert.deepEqual(out, pgimBytes);

    // 1 initial range probe on OPEN (detected oversized) + 2 slab fetches = 3 total HTTP fetches (NOT 8!)
    assert.equal(mock.requests.length - reqsBefore, 3);
    const stats = controller.getStats();
    assert.equal(stats.slabCacheMisses, 2);
    assert.equal(stats.slabCacheHits, 6);
    assert.ok(stats.syncFastPathHits >= 6);

    assert.equal(await client.closeAsync(h), true);
  } finally {
    await controller.detachMailbox();
  }
});

// ---------------------------------------------------------------------------
// 3. Sequential read-ahead slab cache for .rpf / .dll / .ttpf + LRU eviction
// ---------------------------------------------------------------------------

test('sequential 64 KiB reads across .rpf, .dll, and .ttpf coalesce into slab-sized Range fetches and hit LRU slab cache', async () => {
  const slabSize = 256 * 1024; // 256 KiB slabs (4 x 64 KiB reads per slab)
  const rpfBytes = makePatternBytes(600 * 1024, 11); // 3 slabs: [0..256K), [256K..512K), [512K..600K)
  const dllBytes = makePatternBytes(320 * 1024, 22); // 2 slabs
  const ttpfBytes = makePatternBytes(300 * 1024, 33); // 2 slabs

  const mock = createMockRangeFetch(
    new Map([
      ['/game/world.rpf', rpfBytes],
      ['/game/engine.dll', dllBytes],
      ['/game/dialogue.ttpf', ttpfBytes],
    ]),
  );

  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    slabSize,
    slabMaxBytes: 4 * 1024 * 1024, // 4 MiB cap
    opfsDir: null,
  });

  const sab = createHttpfsMailboxBuffer(64 * 1024);
  const client = new HttpfsMailboxClient(sab);

  try {
    await controller.init({ dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' });
    await controller.attachMailbox(sab);

    async function verifySequentialRead(openPath, expectedBytes, expectedSlabFetches) {
      const reqsStart = mock.requests.length;
      const h = await client.openAsync(openPath);
      assert.ok(h >= 1, `Failed to open ${openPath}`);
      assert.equal(controller.getHandleInfo(h).size, expectedBytes.byteLength);

      const out = new Uint8Array(expectedBytes.byteLength);
      const stepBuf = new Uint8Array(65536);
      let off = 0;
      while (off < expectedBytes.byteLength) {
        const got = await client.readAsync(h, BigInt(off), stepBuf, 65536);
        assert.ok(got > 0);
        out.set(stepBuf.subarray(0, got), off);
        off += got;
      }
      assert.deepEqual(out, expectedBytes);

      const totalReqs = mock.requests.length - reqsStart;
      // OPEN probes slab 0 directly -> strictly expectedSlabFetches aligned slab fetches (0 duplicate chunk-0 fetch)
      assert.equal(
        totalReqs,
        expectedSlabFetches,
        `Expected ${expectedSlabFetches} HTTP fetches for ${openPath}, got ${totalReqs}`,
      );

      // Re-reading across a slab boundary must hit cached slabs with ZERO new HTTP fetches!
      const reqsBeforeCross = mock.requests.length;
      const crossBuf = new Uint8Array(65536);
      const crossOff = slabSize - 32768;
      const crossGot = await client.readAsync(h, BigInt(crossOff), crossBuf, 65536);
      assert.equal(crossGot, 65536);
      assert.deepEqual(crossBuf, expectedBytes.subarray(crossOff, crossOff + 65536));
      assert.equal(mock.requests.length, reqsBeforeCross);

      assert.equal(await client.closeAsync(h), true);
    }

    await verifySequentialRead('G:\\world.rpf', rpfBytes, 3);
    await verifySequentialRead('G:\\engine.dll', dllBytes, 2);
    await verifySequentialRead('G:\\dialogue.ttpf', ttpfBytes, 2);

    const stats = controller.getStats();
    assert.equal(stats.slabCacheMisses, 7);
    assert.ok(stats.slabCacheHits >= 12, `Expected >= 12 slab hits, got ${stats.slabCacheHits}`);
    assert.equal(stats.immutableCache.slabEntries, 7);
    assert.equal(
      stats.immutableCache.slabBytesCached,
      rpfBytes.byteLength + dllBytes.byteLength + ttpfBytes.byteLength,
    );
    assert.ok(stats.syncFastPathHits >= 15);
  } finally {
    await controller.detachMailbox();
  }
});

test('slab LRU cache strictly enforces slabMaxBytes cap and evicts least-recently-used slabs cleanly', async () => {
  const slabSize = 128 * 1024; // 128 KiB slabs
  const fileBytes = makePatternBytes(512 * 1024, 88); // 4 slabs: 0, 1, 2, 3
  const mock = createMockRangeFetch(new Map([['/game/stream.rpf', fileBytes]]));

  // Budget fits strictly 2 slabs (256 KiB).
  const cache = new ImmutableHttpfsCache({
    slabSize,
    slabMaxBytes: 256 * 1024,
    fetchImpl: mock.fetchImpl,
  });

  // Read slab 0 and slab 1 -> LRU = [0, 1], slabBytesCached = 256 KiB.
  assert.deepEqual(await cache.read('/game/stream.rpf', 0, 4096), fileBytes.subarray(0, 4096));
  assert.deepEqual(
    await cache.read('/game/stream.rpf', slabSize, 4096),
    fileBytes.subarray(slabSize, slabSize + 4096),
  );
  assert.equal(cache.getStats().slabEntries, 2);
  assert.equal(cache.getStats().slabBytesCached, 256 * 1024);
  assert.equal(cache.getStats().slabEvictions, 0);

  // Touch slab 0 via tryReadSync -> LRU becomes [1, 0].
  const hit0 = cache.tryReadSync('/game/stream.rpf', 1024, 2048);
  assert.deepEqual(hit0, fileBytes.subarray(1024, 3072));

  // Read slab 2 -> MUST evict slab 1 (oldest), keeping slab 0! LRU = [0, 2].
  assert.deepEqual(
    await cache.read('/game/stream.rpf', slabSize * 2, 4096),
    fileBytes.subarray(slabSize * 2, slabSize * 2 + 4096),
  );
  let st = cache.getStats();
  assert.equal(st.slabEvictions, 1);
  assert.equal(st.slabEntries, 2);
  assert.equal(st.slabBytesCached, 256 * 1024);

  // Slab 0 is STILL cached (sync hit, 0 fetches)!
  const reqsBefore = mock.requests.length;
  assert.deepEqual(
    cache.tryReadSync('/game/stream.rpf', 8192, 4096),
    fileBytes.subarray(8192, 12288),
  );
  assert.equal(mock.requests.length, reqsBefore);

  // Slab 1 was evicted -> tryReadSync returns null, async read re-fetches and evicts slab 2.
  assert.equal(cache.tryReadSync('/game/stream.rpf', slabSize + 100, 1024), null);
  assert.deepEqual(
    await cache.read('/game/stream.rpf', slabSize + 100, 1024),
    fileBytes.subarray(slabSize + 100, slabSize + 1124),
  );
  assert.equal(mock.requests.length, reqsBefore + 1);
  st = cache.getStats();
  assert.equal(st.slabEvictions, 2);
  assert.ok(st.slabBytesCached <= 256 * 1024);

  // Read slab 3 -> evicts slab 0, never exceeds cap.
  assert.deepEqual(
    await cache.read('/game/stream.rpf', slabSize * 3, 4096),
    fileBytes.subarray(slabSize * 3, slabSize * 3 + 4096),
  );
  st = cache.getStats();
  assert.equal(st.slabEvictions, 3);
  assert.ok(st.slabBytesCached <= 256 * 1024);
  assert.ok(st.slabEntries <= 2);
});

test('concurrent reads hitting the same uncached slab deduplicate into 1 HTTP fetch and handle truncated 206 responses', async () => {
  const slabSize = 256 * 1024;
  const fileBytes = makePatternBytes(512 * 1024, 57);

  // Part A: Concurrent deduplication for slab 1
  const mock = createMockRangeFetch(new Map([['/game/concurrent.rpf', fileBytes]]), {
    delayMs: 15,
  });
  const cache = new ImmutableHttpfsCache({
    slabSize,
    slabMaxBytes: 1024 * 1024,
    fetchImpl: mock.fetchImpl,
  });

  const reads = await Promise.all([
    cache.read('/game/concurrent.rpf', slabSize, 4096),
    cache.read('/game/concurrent.rpf', slabSize + 1024, 8192),
    cache.read('/game/concurrent.rpf', slabSize + 65536, 16384),
    cache.read('/game/concurrent.rpf', slabSize + 131072, 65536),
    cache.read('/game/concurrent.rpf', slabSize + 200000, 32768),
  ]);

  assert.deepEqual(reads[0], fileBytes.subarray(slabSize, slabSize + 4096));
  assert.deepEqual(reads[1], fileBytes.subarray(slabSize + 1024, slabSize + 1024 + 8192));
  assert.deepEqual(reads[2], fileBytes.subarray(slabSize + 65536, slabSize + 65536 + 16384));
  assert.deepEqual(reads[3], fileBytes.subarray(slabSize + 131072, slabSize + 131072 + 65536));
  assert.deepEqual(reads[4], fileBytes.subarray(slabSize + 200000, slabSize + 200000 + 32768));

  assert.equal(mock.requests.length, 1, 'All 5 concurrent reads into slab 1 must share 1 fetch');
  assert.equal(cache.getStats().inFlightCoalesced, 4);
  assert.equal(cache.getStats().slabCacheMisses, 1);

  // Part B: Truncated 206 server (serves at most 64 KiB per response for a 256 KiB slab)
  const truncMock = createMockRangeFetch(new Map([['/game/trunc.ttpf', fileBytes]]), {
    maxServeBytes: 64 * 1024,
  });
  const truncCache = new ImmutableHttpfsCache({
    slabSize,
    slabMaxBytes: 1024 * 1024,
    fetchImpl: truncMock.fetchImpl,
  });

  const slice1 = await truncCache.read('/game/trunc.ttpf', 1000, 150000);
  assert.deepEqual(slice1, fileBytes.subarray(1000, 151000));
  const reqsAfterFirst = truncMock.requests.length;

  // Complete slab 0 is now cached; reading tail of slab 0 hits synchronously with 0 fetches!
  const slice2 = truncCache.tryReadSync('/game/trunc.ttpf', 200000, 50000);
  assert.deepEqual(slice2, fileBytes.subarray(200000, 250000));
  assert.equal(truncMock.requests.length, reqsAfterFirst);
});

// ---------------------------------------------------------------------------
// 4. HttpfsMailboxHost synchronous fast path vs async path parity
// ---------------------------------------------------------------------------

test('HttpfsMailboxHost synchronous fast path supports Uint8Array, { bytes }, ArrayBuffer, errors, and matches async path identically', async () => {
  const payload = makePatternBytes(8192, 99);
  const sabSync = createHttpfsMailboxBuffer(1024);
  const sabAsync = createHttpfsMailboxBuffer(1024);

  let mode = 'uint8';
  const syncHost = new HttpfsMailboxHost(sabSync, {
    open: (relPath) => (relPath === 'ok.bin' ? 7 : -2),
    read: (handle, offset, reqLen) => {
      if (handle !== 7) return null;
      if (mode === 'throw') throw new Error('boom');
      if (mode === 'badtype') return 12345;
      const slice = payload.subarray(offset, Math.min(payload.byteLength, offset + reqLen));
      if (mode === 'object') return { bytes: slice };
      if (mode === 'arraybuffer') return slice.slice().buffer;
      return slice;
    },
    close: (handle) => handle === 7,
  });

  const asyncHost = new HttpfsMailboxHost(sabAsync, {
    open: async (relPath) => (relPath === 'ok.bin' ? 7 : -2),
    read: async (handle, offset, reqLen) => {
      if (handle !== 7) return null;
      const slice = payload.subarray(offset, Math.min(payload.byteLength, offset + reqLen));
      return { bytes: slice };
    },
    close: async (handle) => handle === 7,
  });

  const syncClient = new HttpfsMailboxClient(sabSync);
  const asyncClient = new HttpfsMailboxClient(sabAsync);

  syncHost.start();
  asyncHost.start();

  try {
    assert.equal(await syncClient.openAsync('missing.bin'), -1);
    assert.equal(await asyncClient.openAsync('missing.bin'), -1);

    const hs = await syncClient.openAsync('ok.bin');
    const ha = await asyncClient.openAsync('ok.bin');
    assert.equal(hs, 7);
    assert.equal(ha, 7);

    // 1. Uint8Array sync vs { bytes } async across 8 mailbox chunks
    const outSync = new Uint8Array(payload.byteLength);
    const outAsync = new Uint8Array(payload.byteLength);
    assert.equal(await syncClient.readAsync(hs, 0n, outSync, payload.byteLength), payload.byteLength);
    assert.equal(
      await asyncClient.readAsync(ha, 0n, outAsync, payload.byteLength),
      payload.byteLength,
    );
    assert.deepEqual(outSync, payload);
    assert.deepEqual(outAsync, payload);

    // 2. { bytes } sync return shape
    mode = 'object';
    const objBuf = new Uint8Array(512);
    assert.equal(await syncClient.readAsync(hs, 500n, objBuf, 512), 512);
    assert.deepEqual(objBuf, payload.subarray(500, 1012));

    // 3. ArrayBuffer sync return shape
    mode = 'arraybuffer';
    const abBuf = new Uint8Array(256);
    assert.equal(await syncClient.readAsync(hs, 1024n, abBuf, 256), 256);
    assert.deepEqual(abBuf, payload.subarray(1024, 1280));

    // 4. Error paths on sync fast path
    assert.equal(await syncClient.readAsync(99, 0n, abBuf, 64), -9);
    mode = 'badtype';
    assert.equal(await syncClient.readAsync(hs, 0n, abBuf, 64), -5);
    mode = 'throw';
    assert.equal(await syncClient.readAsync(hs, 0n, abBuf, 64), -5);

    assert.equal(await syncClient.closeAsync(hs), true);
    assert.equal(await asyncClient.closeAsync(ha), true);

    const syncStats = syncHost.getStats();
    const asyncStats = asyncHost.getStats();
    assert.equal(syncStats.asyncPathHits, 0, 'Sync host should handle all requests on sync fast path');
    assert.equal(syncStats.syncFastPathHits, syncStats.requests);
    assert.ok(syncStats.syncFastPathHits >= 15);
    assert.equal(asyncStats.asyncPathHits, asyncStats.requests);
  } finally {
    await syncHost.stop();
    await asyncHost.stop();
  }
});

// ---------------------------------------------------------------------------
// 5. Real HTTP server + cross-thread synchronous Atomics.wait boot simulation
// ---------------------------------------------------------------------------

test('real HTTP server + cross-thread Atomics.wait boot loads PGIM in 1 fetch and coalesces sequential RPF/TTPF reads into slabs', async () => {
  const manifestBytes = utf8Encoder.encode('{"version":1,"entry":"main/image/guest.pgim"}');
  const pgimBytes = makePatternBytes(1024 * 1024 + 4096, 61); // > 1 MiB (17 x 64 KiB reads)
  const rpfBytes = makePatternBytes(700 * 1024, 71); // 11 x 64 KiB reads -> 1 x 2 MiB slab fetch
  const ttpfBytes = makePatternBytes(400 * 1024, 81); // 7 x 64 KiB reads -> 1 x 2 MiB slab fetch

  const server = await startRangeHttpServer(
    new Map([
      ['/boot/program-manifest.json', manifestBytes],
      ['/boot/main/image/guest.pgim', pgimBytes],
      ['/game/pack0.rpf', rpfBytes],
      ['/game/audio.ttpf', ttpfBytes],
    ]),
  );

  const controller = new IoWorkerController({ opfsDir: null });
  const sab = createHttpfsMailboxBuffer(64 * 1024);

  const workerCode = `
    const { workerData, parentPort } = require('node:worker_threads');
    const { HttpfsMailboxClient } = require(workerData.mailboxPath);
    const client = new HttpfsMailboxClient(workerData.sab, { timeoutMs: 10000 });

    function readAll64K(relPath, expectedSize) {
      const h = client.open(relPath);
      if (h < 0) throw new Error('open failed for ' + relPath);
      const out = new Uint8Array(expectedSize);
      const step = new Uint8Array(65536);
      let off = 0;
      while (off < expectedSize) {
        const n = client.read(h, BigInt(off), step, 65536);
        if (n <= 0) break;
        out.set(step.subarray(0, n), off);
        off += n;
      }
      const eof = client.read(h, BigInt(expectedSize), step, 64);
      const closed = client.close(h);
      return { h, bytesRead: off, eof, closed, checksum: out.reduce((a, b, i) => (a + b * ((i & 7) + 1)) >>> 0, 0) };
    }

    const man = readAll64K('boot/program-manifest.json', workerData.sizes.manifest);
    const pgim = readAll64K('boot/main/image/guest.pgim', workerData.sizes.pgim);
    const rpf = readAll64K('G:\\\\pack0.rpf', workerData.sizes.rpf);
    const ttpf = readAll64K('G:\\\\audio.ttpf', workerData.sizes.ttpf);
    parentPort.postMessage({ man, pgim, rpf, ttpf });
  `;

  const checksumOf = (arr) => arr.reduce((a, b, i) => (a + b * ((i & 7) + 1)) >>> 0, 0);

  try {
    await controller.init({
      dataBase: `${server.origin}/data/`,
      bootBase: `${server.origin}/boot/`,
      gameBase: `${server.origin}/game/`,
    });
    await controller.attachMailbox(sab);

    const result = await new Promise((resolve, reject) => {
      const w = new NodeWorker(workerCode, {
        eval: true,
        workerData: {
          sab,
          mailboxPath: path.join(repoRoot, 'web', 'workers', 'httpfs_mailbox.js'),
          sizes: {
            manifest: manifestBytes.byteLength,
            pgim: pgimBytes.byteLength,
            rpf: rpfBytes.byteLength,
            ttpf: ttpfBytes.byteLength,
          },
        },
      });
      w.once('message', resolve);
      w.once('error', reject);
      w.once('exit', (code) => {
        if (code !== 0) reject(new Error(`Worker exited with code ${code}`));
      });
    });

    assert.equal(result.man.bytesRead, manifestBytes.byteLength);
    assert.equal(result.man.checksum, checksumOf(manifestBytes));
    assert.equal(result.man.eof, 0);
    assert.equal(result.man.closed, true);

    assert.equal(result.pgim.bytesRead, pgimBytes.byteLength);
    assert.equal(result.pgim.checksum, checksumOf(pgimBytes));
    assert.equal(result.pgim.eof, 0);
    assert.equal(result.pgim.closed, true);

    assert.equal(result.rpf.bytesRead, rpfBytes.byteLength);
    assert.equal(result.rpf.checksum, checksumOf(rpfBytes));
    assert.equal(result.rpf.eof, 0);
    assert.equal(result.rpf.closed, true);

    assert.equal(result.ttpf.bytesRead, ttpfBytes.byteLength);
    assert.equal(result.ttpf.checksum, checksumOf(ttpfBytes));
    assert.equal(result.ttpf.eof, 0);
    assert.equal(result.ttpf.closed, true);

    // Verify HTTP request counts per path on the real server!
    const countFor = (pathname) => server.requests.filter((r) => r.pathname === pathname).length;
    assert.equal(countFor('/boot/program-manifest.json'), 1);
    assert.equal(countFor('/boot/main/image/guest.pgim'), 1, 'PGIM must take exactly 1 HTTP fetch!');
    assert.equal(countFor('/game/pack0.rpf'), 1, 'RPF: 1 slab-0 fetch on OPEN covers entire 700 KiB file (NOT 11!)');
    assert.equal(countFor('/game/audio.ttpf'), 1, 'TTPF: 1 slab-0 fetch on OPEN covers entire 400 KiB file (NOT 7!)');

    const stats = controller.getStats();
    assert.equal(stats.pgimFullFetches, 1);
    assert.equal(stats.wholeFileCache.activeEntries, 0, 'All whole-file buffers freed after CLOSE');
    assert.equal(stats.wholeFileCache.activeBytes, 0);
    assert.equal(stats.wholeFileCache.releases, 2);
    assert.equal(stats.slabCacheMisses, 2);
    assert.ok(stats.slabCacheHits >= 14);
    assert.ok(stats.syncFastPathHits >= 35);
    assert.equal(stats.openHandles, 0);
  } finally {
    await controller.detachMailbox();
    await server.close();
  }
});

// ---------------------------------------------------------------------------
// 6. W125: .wgsl.pack slab caching & .reflect.json / boot-options.json whole-file buffering
// ---------------------------------------------------------------------------

test('W125: multi-chunk 64 KiB reads of .wgsl.pack coalesce into ImmutableHttpfsCache slabs without whole-file buffering', async () => {
  const slabSize = 256 * 1024; // 256 KiB slabs (4 x 64 KiB reads per slab)
  const packSize = 600 * 1024; // 10 x 64 KiB reads -> 3 slabs: [0..256K), [256K..512K), [512K..600K)
  const packBytes = makePatternBytes(packSize, 137);

  const mock = createMockRangeFetch(new Map([['/shaders/splash.wgsl.pack', packBytes]]));
  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    slabSize,
    slabMaxBytes: 4 * 1024 * 1024,
    opfsDir: null,
  });

  const sab = createHttpfsMailboxBuffer(64 * 1024);
  const client = new HttpfsMailboxClient(sab);

  try {
    await controller.init({
      dataBase: '/data/',
      bootBase: '/boot/',
      gameBase: '/game/',
      shadersBase: '/shaders/',
    });
    await controller.attachMailbox(sab);

    const reqsBefore = mock.requests.length;
    const h = await client.openAsync('shaders/splash.wgsl.pack');
    assert.ok(h >= 1, `Expected valid .wgsl.pack handle, got ${h}`);
    assert.equal(controller.getHandleInfo(h).size, packSize);
    assert.equal(controller.getHandleInfo(h).wholeFileBuffered, false);
    assert.equal(controller.getStats().wholeFileCache.activeEntries, 0);

    const out = new Uint8Array(packSize);
    const chunkBuf = new Uint8Array(65536);
    let offset = 0;
    let readCalls = 0;
    while (offset < packSize) {
      const got = await client.readAsync(h, BigInt(offset), chunkBuf, 65536);
      assert.ok(got > 0, `Expected positive bytes at offset ${offset}, got ${got}`);
      out.set(chunkBuf.subarray(0, got), offset);
      offset += got;
      readCalls += 1;
    }

    assert.equal(readCalls, 10);
    assert.deepEqual(out, packBytes);

    // OPEN probes slab 0 directly -> strictly 3 aligned slab fetches (0 duplicate chunk-0 fetch, NOT 10!)
    assert.equal(
      mock.requests.length - reqsBefore,
      3,
      `Expected 3 HTTP fetches for .wgsl.pack, got ${mock.requests.length - reqsBefore}`,
    );

    // Cross-slab re-read hits cached slabs with 0 additional HTTP fetches.
    const reqsBeforeCross = mock.requests.length;
    const crossBuf = new Uint8Array(65536);
    const crossOff = slabSize - 32768;
    const crossGot = await client.readAsync(h, BigInt(crossOff), crossBuf, 65536);
    assert.equal(crossGot, 65536);
    assert.deepEqual(crossBuf, packBytes.subarray(crossOff, crossOff + 65536));
    assert.equal(mock.requests.length, reqsBeforeCross);

    const stats = controller.getStats();
    assert.equal(stats.slabCacheMisses, 3);
    assert.ok(stats.slabCacheHits >= 7, `Expected >= 7 slab hits, got ${stats.slabCacheHits}`);
    assert.equal(stats.immutableCache.slabEntries, 3);
    assert.equal(stats.immutableCache.slabBytesCached, packSize);
    assert.equal(stats.wholeFileCache.wholeFileFetches, 0);
    assert.ok(stats.syncFastPathHits >= 7);

    assert.equal(await client.closeAsync(h), true);
  } finally {
    await controller.detachMailbox();
  }
});

test('W125: .reflect.json and boot-options.json load in 1 HTTP fetch, release on CLOSE, and fall back to slabs when oversized', async () => {
  // Synthetic .reflect.json spanning 5 x 64 KiB reads (300 KiB)
  const padEntry = 'x'.repeat(1024);
  const entries = [];
  while (entries.length * 1060 < 300 * 1024) {
    entries.push({ hash: `0x${entries.length.toString(16).padStart(8, '0')}`, pad: padEntry });
  }
  const reflectBytes = utf8Encoder.encode(JSON.stringify({ version: 1, shaders: entries }));
  assert.ok(reflectBytes.byteLength > 4 * 65536, 'Synthetic .reflect.json must exceed 4 x 64 KiB chunks');

  const bootOptionsBytes = utf8Encoder.encode(
    JSON.stringify({ streamShaders: true, shaderPack: 'shaders/splash.wgsl.pack' }),
  );

  const mock = createMockRangeFetch(
    new Map([
      ['/shaders/splash.wgsl.pack.reflect.json', reflectBytes],
      ['/boot/boot-options.json', bootOptionsBytes],
    ]),
  );

  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    opfsDir: null,
  });

  const sab = createHttpfsMailboxBuffer(64 * 1024);
  const client = new HttpfsMailboxClient(sab);

  try {
    await controller.init({
      dataBase: '/data/',
      bootBase: '/boot/',
      gameBase: '/game/',
      shadersBase: '/shaders/',
    });
    await controller.attachMailbox(sab);

    // 1. Multi-chunk 64 KiB reads of .reflect.json complete in 1 HTTP fetch and release on CLOSE.
    const reqsBeforeReflect = mock.requests.length;
    const hRef = await client.openAsync('shaders/splash.wgsl.pack.reflect.json');
    assert.ok(hRef >= 1, `Expected valid .reflect.json handle, got ${hRef}`);
    assert.equal(controller.getHandleInfo(hRef).size, reflectBytes.byteLength);
    assert.equal(controller.getHandleInfo(hRef).wholeFileBuffered, true);
    assert.equal(mock.requests.length - reqsBeforeReflect, 1);

    let statsOpen = controller.getStats();
    assert.equal(statsOpen.wholeFileCache.activeEntries, 1);
    assert.equal(statsOpen.wholeFileCache.activeBytes, reflectBytes.byteLength);
    assert.equal(statsOpen.wholeFileCache.wholeFileFetches, 1);
    assert.equal(statsOpen.pgimFullFetches, 0);

    const assembled = new Uint8Array(reflectBytes.byteLength);
    const chunkBuf = new Uint8Array(65536);
    let offset = 0;
    let readCalls = 0;
    while (offset < reflectBytes.byteLength) {
      const got = await client.readAsync(hRef, BigInt(offset), chunkBuf, 65536);
      assert.ok(got > 0);
      assembled.set(chunkBuf.subarray(0, got), offset);
      offset += got;
      readCalls += 1;
    }

    assert.ok(readCalls >= 5, `Expected >= 5 64 KiB reads, got ${readCalls}`);
    assert.deepEqual(assembled, reflectBytes);
    assert.equal(await client.readAsync(hRef, BigInt(reflectBytes.byteLength), chunkBuf, 64), 0);
    assert.equal(
      mock.requests.length - reqsBeforeReflect,
      1,
      `Expected strictly 1 HTTP fetch for .reflect.json, got ${mock.requests.length - reqsBeforeReflect}`,
    );

    assert.equal(await client.closeAsync(hRef), true);
    const statsAfterReflectClose = controller.getStats();
    assert.equal(statsAfterReflectClose.wholeFileCache.activeEntries, 0);
    assert.equal(statsAfterReflectClose.wholeFileCache.activeBytes, 0);
    assert.equal(statsAfterReflectClose.wholeFileCache.releases, 1);

    // 2. boot/boot-options.json loads in 1 HTTP fetch and releases on CLOSE.
    const reqsBeforeOpts = mock.requests.length;
    const hOpts = await client.openAsync('boot/boot-options.json');
    assert.ok(hOpts >= 1, `Expected valid boot-options.json handle, got ${hOpts}`);
    assert.equal(controller.getHandleInfo(hOpts).wholeFileBuffered, true);
    assert.equal(mock.requests.length - reqsBeforeOpts, 1);

    const optsBuf = new Uint8Array(bootOptionsBytes.byteLength);
    assert.equal(await client.readAsync(hOpts, 0n, optsBuf, bootOptionsBytes.byteLength), bootOptionsBytes.byteLength);
    assert.deepEqual(optsBuf, bootOptionsBytes);
    assert.equal(mock.requests.length - reqsBeforeOpts, 1);

    assert.equal(await client.closeAsync(hOpts), true);
    const statsAfterOptsClose = controller.getStats();
    assert.equal(statsAfterOptsClose.wholeFileCache.activeEntries, 0);
    assert.equal(statsAfterOptsClose.wholeFileCache.activeBytes, 0);
    assert.equal(statsAfterOptsClose.wholeFileCache.releases, 2);
  } finally {
    await controller.detachMailbox();
  }

  // 3. Oversized .reflect.json exceeding wholeFileMaxBytes falls back cleanly to slab LRU cache.
  const hugeReflectBytes = makePatternBytes(512 * 1024, 149);
  const hugeMock = createMockRangeFetch(
    new Map([['/shaders/huge.wgsl.pack.reflect.json', hugeReflectBytes]]),
  );
  const hugeController = new IoWorkerController({
    fetchImpl: hugeMock.fetchImpl,
    wholeFileMaxBytes: 128 * 1024,
    slabSize: 256 * 1024,
    slabMaxBytes: 512 * 1024,
    opfsDir: null,
  });
  const hugeSab = createHttpfsMailboxBuffer(64 * 1024);
  const hugeClient = new HttpfsMailboxClient(hugeSab);

  try {
    await hugeController.init({
      dataBase: '/data/',
      bootBase: '/boot/',
      gameBase: '/game/',
      shadersBase: '/shaders/',
    });
    await hugeController.attachMailbox(hugeSab);

    const reqsStart = hugeMock.requests.length;
    const hHuge = await hugeClient.openAsync('shaders/huge.wgsl.pack.reflect.json');
    assert.ok(hHuge >= 1);
    assert.equal(hugeController.getHandleInfo(hHuge).wholeFileBuffered, false);
    assert.equal(hugeController.getStats().wholeFileCache.fallbacks, 1);

    const out = new Uint8Array(hugeReflectBytes.byteLength);
    const stepBuf = new Uint8Array(65536);
    for (let i = 0; i < 8; i += 1) {
      const got = await hugeClient.readAsync(hHuge, BigInt(i * 65536), stepBuf, 65536);
      assert.equal(got, 65536);
      out.set(stepBuf, i * 65536);
    }
    assert.deepEqual(out, hugeReflectBytes);

    // 1 initial oversized probe on OPEN + 2 x 256 KiB slab fetches = 3 HTTP fetches (NOT 8!)
    assert.equal(hugeMock.requests.length - reqsStart, 3);
    assert.equal(hugeController.getStats().slabCacheMisses, 2);
    assert.equal(hugeController.getStats().slabCacheHits, 6);
    assert.equal(await hugeClient.closeAsync(hHuge), true);
  } finally {
    await hugeController.detachMailbox();
  }
});

// ---------------------------------------------------------------------------
// 7. W182: Adaptive read-ahead window sizing, multi-slot mailbox pipelining & telemetry
// ---------------------------------------------------------------------------

test('W182: computeReadAheadWindowSize ramps 256 KiB -> 1 MiB -> 2 MiB for 64 KiB DirectStorage and .bk2 reads capped by maxReadAheadBytes', () => {
  assert.equal(HTTPFS_DIRECTSTORAGE_CHUNK_BYTES, 64 * 1024);
  assert.equal(HTTPFS_READAHEAD_BASE_BYTES, 256 * 1024);
  assert.equal(HTTPFS_READAHEAD_MID_BYTES, 1024 * 1024);
  assert.equal(HTTPFS_READAHEAD_MAX_BYTES, 2 * 1024 * 1024);
  assert.equal(HTTPFS_DEFAULT_PIPELINE_SLOTS, 4);

  assert.equal(isVideoReadAheadPath('movies/intro_4k.bk2'), true);
  assert.equal(isVideoReadAheadPath('MOVIES/SPLASH.BIK'), true);
  assert.equal(isVideoReadAheadPath('update/world.rpf'), false);

  // Non-sequential, invalid, or zero-cap inputs return 0.
  assert.equal(computeReadAheadWindowSize(65536, 0), 0);
  assert.equal(computeReadAheadWindowSize(65536, -3), 0);
  assert.equal(computeReadAheadWindowSize(0, 4), 0);
  assert.equal(computeReadAheadWindowSize(-65536, 4), 0);
  assert.equal(computeReadAheadWindowSize(NaN, 4), 0);
  assert.equal(computeReadAheadWindowSize(65536, 8, { maxReadAheadBytes: 0 }), 0);

  // General 64 KiB DirectStorage ramp: 256 KiB (seq 1..3) -> 1 MiB (seq 4..7) -> 2 MiB (seq >= 8).
  assert.equal(computeReadAheadWindowSize(65536, 1), 256 * 1024);
  assert.equal(computeReadAheadWindowSize(65536, 2), 256 * 1024);
  assert.equal(computeReadAheadWindowSize(65536, 3), 256 * 1024);
  assert.equal(computeReadAheadWindowSize(65536, 4), 1024 * 1024);
  assert.equal(computeReadAheadWindowSize(65536, 7), 1024 * 1024);
  assert.equal(computeReadAheadWindowSize(65536, 8), 2 * 1024 * 1024);
  assert.equal(computeReadAheadWindowSize(65536n, 20), 2 * 1024 * 1024);

  // 4K Bink2 (.bk2) ramp: 256 KiB (seq 1) -> 1 MiB (seq 2..3) -> 2 MiB (seq >= 4).
  assert.equal(computeReadAheadWindowSize(65536, 0, { path: 'movies/intro.bk2' }), 0);
  assert.equal(computeReadAheadWindowSize(65536, 1, { path: 'movies/intro.bk2' }), 256 * 1024);
  assert.equal(computeReadAheadWindowSize(65536, 2, { path: 'movies/intro.bk2' }), 1024 * 1024);
  assert.equal(computeReadAheadWindowSize(65536, 3, { path: 'movies/intro.bk2' }), 1024 * 1024);
  assert.equal(computeReadAheadWindowSize(65536, 4, { path: 'movies/intro.bk2' }), 2 * 1024 * 1024);
  assert.equal(computeReadAheadWindowSize(65536, 2, { isVideoStream: true }), 1024 * 1024);
  assert.equal(computeReadAheadWindowSize(65536, 4, { isVideoStream: true }), 2 * 1024 * 1024);

  // Strict maxReadAheadBytes capping.
  assert.equal(
    computeReadAheadWindowSize(65536, 12, { maxReadAheadBytes: 512 * 1024 }),
    512 * 1024,
  );
  assert.equal(
    computeReadAheadWindowSize(65536, 1, { maxReadAheadBytes: 128 * 1024 }),
    128 * 1024,
  );
  assert.equal(
    computeReadAheadWindowSize(65536, 4, { path: 'movies/intro.bk2', maxReadAheadBytes: 1024 * 1024 }),
    1024 * 1024,
  );
});

test('W182: HttpfsMailboxHost adaptive read-ahead ramps 256 KiB -> 1 MiB -> 2 MiB, tracks per-handle burst/wait telemetry, and handles random seeks and short reads', async () => {
  const fileSize = 4 * 1024 * 1024; // 4 MiB = 64 x 64 KiB DirectStorage chunks
  const fileBytes = makePatternBytes(fileSize, 193);
  const backendReads = [];
  let fakeNowMs = 100;

  const sab = createHttpfsMailboxBuffer(64 * 1024);
  const host = new HttpfsMailboxHost(sab, {
    readAhead: true,
    maxReadAheadBytes: 2 * 1024 * 1024,
    now: () => fakeNowMs,
    open: (relPath) => (relPath === 'update/streaming_world.rpf' ? 11 : -2),
    read: async (handle, offset, reqLen) => {
      if (handle !== 11) return null;
      backendReads.push({ offset, reqLen });
      fakeNowMs += 5; // Deterministic 5ms async backend read latency
      return fileBytes.subarray(offset, Math.min(fileSize, offset + reqLen));
    },
    close: (handle) => handle === 11,
  });
  const client = new HttpfsMailboxClient(sab);
  host.start();

  try {
    const h = await client.openAsync('update/streaming_world.rpf');
    assert.equal(h, 11);

    // Stream all 4 MiB in sequential 64 KiB chunks.
    const assembled = new Uint8Array(fileSize);
    const chunkBuf = new Uint8Array(64 * 1024);
    for (let i = 0; i < 64; i += 1) {
      const got = await client.readAsync(h, BigInt(i * 65536), chunkBuf, 65536);
      assert.equal(got, 65536);
      assembled.set(chunkBuf, i * 65536);
    }
    assert.deepEqual(assembled, fileBytes);

    // Expected adaptive fetch ramp:
    //   Read 0  (seq 0):  64 KiB
    //   Read 1  (seq 1): 256 KiB  -> serves reads 1..4 (3 read-ahead hits)
    //   Read 5  (seq 5):   1 MiB  -> serves reads 5..20 (15 read-ahead hits)
    //   Read 21 (seq 21):  2 MiB  -> serves reads 21..52 (31 read-ahead hits)
    //   Read 53 (seq 53):  2 MiB  -> returns remaining 704 KiB, serves reads 53..63 (10 read-ahead hits)
    assert.deepEqual(
      backendReads.map((r) => r.reqLen),
      [64 * 1024, 256 * 1024, 1024 * 1024, 2 * 1024 * 1024, 2 * 1024 * 1024],
    );

    const handleTel = host.getHandleTelemetry(h);
    assert.ok(handleTel);
    assert.equal(handleTel.handle, 11);
    assert.equal(handleTel.reads, 64);
    assert.equal(handleTel.burstReads, 63);
    assert.equal(handleTel.maxSequentialCount, 63);
    assert.equal(handleTel.peakPendingSlots, 1);
    assert.equal(handleTel.readAheadFetches, 4);
    assert.equal(handleTel.readAheadHits, 59);
    assert.equal(handleTel.readAheadBytesServed, 59 * 65536);
    assert.equal(handleTel.currentWindowBytes, 2 * 1024 * 1024);
    assert.ok(handleTel.avgWaitMs > 0, `Expected positive avgWaitMs, got ${handleTel.avgWaitMs}`);
    assert.equal(handleTel.totalWaitMs, 5 * 5); // 5 async backend reads x 5ms each
    assert.equal(handleTel.maxWaitMs, 5);
    assert.equal(handleTel.isVideoStream, false);

    // Random seek resets sequentialCount and fetches only exact requested length on miss.
    const seekCountBefore = backendReads.length;
    const seekBuf = new Uint8Array(4096);
    assert.equal(await client.readAsync(h, 12345n, seekBuf, 4096), 4096);
    assert.deepEqual(seekBuf, fileBytes.subarray(12345, 12345 + 4096));
    assert.equal(backendReads.length, seekCountBefore + 1);
    assert.equal(backendReads[backendReads.length - 1].reqLen, 4096);
    assert.equal(host.getHandleTelemetry(h).sequentialCount, 0);
    assert.equal(host.getHandleTelemetry(h).currentWindowBytes, 0);

    // Close frees read-ahead buffer while keeping sanitized telemetry with zero paths.
    assert.equal(await client.closeAsync(h), true);
    const stats = host.getStats();
    assert.equal(stats.burstReads, 63);
    assert.equal(stats.peakPendingSlots, 1);
    assert.equal(stats.readAheadBytesServed, 59 * 65536);
    assert.ok(stats.avgWaitMs > 0);
    assert.equal(stats.handles['11'].closed, true);

    const statsJson = JSON.stringify(stats);
    assert.equal(statsJson.includes('streaming_world.rpf'), false);
    assert.equal(statsJson.includes('update/'), false);
  } finally {
    await host.stop();
  }

  // Short-read safety: backend that caps every single read to 64 KiB must never trigger false EOF.
  const shortSize = 200 * 1024; // 3 full 64 KiB + 8 KiB tail
  const shortBytes = makePatternBytes(shortSize, 211);
  const shortSab = createHttpfsMailboxBuffer(64 * 1024);
  const shortHost = new HttpfsMailboxHost(shortSab, {
    readAhead: true,
    open: () => 2,
    read: (_h, offset, reqLen) => {
      const capped = Math.min(reqLen, 64 * 1024);
      return shortBytes.subarray(offset, Math.min(shortSize, offset + capped));
    },
    close: () => true,
  });
  const shortClient = new HttpfsMailboxClient(shortSab);
  shortHost.start();
  try {
    const h2 = await shortClient.openAsync('capped.rpf');
    const out = new Uint8Array(shortSize);
    const step = new Uint8Array(64 * 1024);
    let off = 0;
    while (off < shortSize) {
      const n = await shortClient.readAsync(h2, BigInt(off), step, 65536);
      assert.ok(n > 0);
      out.set(step.subarray(0, n), off);
      off += n;
    }
    assert.deepEqual(out, shortBytes);
    assert.equal(await shortClient.readAsync(h2, BigInt(shortSize), step, 65536), 0);
    assert.equal(await shortClient.closeAsync(h2), true);
  } finally {
    await shortHost.stop();
  }
});

test('W182: multi-slot HttpfsMailboxPoolHost/Client pipelines concurrent 64 KiB DirectStorage and 4K Bink2 .bk2 bursts with coalescing and telemetry', async () => {
  assert.throws(() => createHttpfsMailboxPool(0), RangeError);
  assert.throws(() => createHttpfsMailboxPool(99), RangeError);

  const rpfSize = 2 * 1024 * 1024; // 2 MiB DirectStorage asset (32 x 64 KiB)
  const bk2Size = 3 * 1024 * 1024; // 3 MiB 4K Bink2 video stream (48 x 64 KiB)
  const rpfBytes = makePatternBytes(rpfSize, 43);
  const bk2Bytes = makePatternBytes(bk2Size, 97);

  const backendCalls = [];
  const slots = createHttpfsMailboxPool(4, 64 * 1024);
  assert.equal(slots.length, 4);

  const poolHost = new HttpfsMailboxPoolHost(slots, {
    maxReadAheadBytes: 2 * 1024 * 1024,
    open: (relPath) => {
      if (relPath === 'update/ds_stream.rpf') return 1;
      if (relPath === 'movies/intro_4k.bk2') return 2;
      return -2;
    },
    read: async (handle, offset, reqLen) => {
      backendCalls.push({ handle, offset, reqLen });
      // Small async delay so pipelined sibling slots overlap in flight and coalesce cleanly.
      await new Promise((r) => setTimeout(r, 8));
      const src = handle === 1 ? rpfBytes : handle === 2 ? bk2Bytes : null;
      if (!src) return null;
      return src.subarray(offset, Math.min(src.byteLength, offset + reqLen));
    },
    close: (handle) => handle === 1 || handle === 2,
  });

  const poolClient = new HttpfsMailboxPoolClient(slots);
  poolHost.start();

  try {
    const [hRpf, hBk2] = await Promise.all([
      poolClient.openAsync('update/ds_stream.rpf'),
      poolClient.openAsync('movies/intro_4k.bk2'),
    ]);
    assert.equal(hRpf, 1);
    assert.equal(hBk2, 2);

    // 1. Pipeline 4 concurrent 64 KiB .bk2 reads in one batch -> saturates all 4 slots and coalesces!
    const bk2Batch1 = await poolClient.readBatchAsync([
      { handle: hBk2, offset: 0n, len: 65536 },
      { handle: hBk2, offset: 65536n, len: 65536 },
      { handle: hBk2, offset: 131072n, len: 65536 },
      { handle: hBk2, offset: 196608n, len: 65536 },
    ]);
    for (let i = 0; i < 4; i += 1) {
      assert.equal(bk2Batch1[i].bytesRead, 65536);
      assert.deepEqual(bk2Batch1[i].dst, bk2Bytes.subarray(i * 65536, (i + 1) * 65536));
    }

    // Stream remaining .bk2 chunks in 4-slot pipelined bursts + interleaved DirectStorage .rpf bursts.
    const bk2Out = new Uint8Array(bk2Size);
    for (let i = 0; i < 4; i += 1) {
      bk2Out.set(bk2Batch1[i].dst, i * 65536);
    }
    for (let chunk = 4; chunk < 48; chunk += 4) {
      const batch = await poolClient.readBatchAsync(
        [0, 1, 2, 3].map((k) => ({
          handle: hBk2,
          offset: BigInt((chunk + k) * 65536),
          len: 65536,
        })),
      );
      for (let k = 0; k < 4; k += 1) {
        assert.equal(batch[k].bytesRead, 65536);
        bk2Out.set(batch[k].dst, (chunk + k) * 65536);
      }
    }
    assert.deepEqual(bk2Out, bk2Bytes);

    // Stream all 32 x 64 KiB DirectStorage .rpf chunks in 4-slot pipelined batches.
    const rpfOut = new Uint8Array(rpfSize);
    for (let chunk = 0; chunk < 32; chunk += 4) {
      const batch = await poolClient.readBatchAsync(
        [0, 1, 2, 3].map((k) => ({
          handle: hRpf,
          offset: BigInt((chunk + k) * 65536),
          len: 65536,
        })),
      );
      for (let k = 0; k < 4; k += 1) {
        assert.equal(batch[k].bytesRead, 65536);
        rpfOut.set(batch[k].dst, (chunk + k) * 65536);
      }
    }
    assert.deepEqual(rpfOut, rpfBytes);

    const bk2Tel = poolHost.getHandleTelemetry(hBk2);
    const rpfTel = poolHost.getHandleTelemetry(hRpf);
    assert.ok(bk2Tel && rpfTel);

    // .bk2 stream: detected as video stream, ramped cleanly to 2 MiB window, coalesced pipelined slots!
    assert.equal(bk2Tel.isVideoStream, true);
    assert.equal(bk2Tel.reads, 48);
    assert.equal(bk2Tel.burstReads, 47);
    assert.equal(bk2Tel.peakPendingSlots, 4);
    assert.ok(bk2Tel.coalescedSlotReads >= 2, `Expected >= 2 coalesced .bk2 slot reads, got ${bk2Tel.coalescedSlotReads}`);
    assert.ok(bk2Tel.readAheadBytesServed >= 40 * 65536, `Expected >= 40 chunks served from read-ahead, got ${bk2Tel.readAheadBytesServed}`);
    assert.equal(bk2Tel.currentWindowBytes, 2 * 1024 * 1024);
    assert.ok(bk2Tel.avgWaitMs >= 0);

    // .rpf DirectStorage stream: independent telemetry & read-ahead acceleration.
    assert.equal(rpfTel.isVideoStream, false);
    assert.equal(rpfTel.reads, 32);
    assert.equal(rpfTel.burstReads, 31);
    assert.equal(rpfTel.peakPendingSlots, 4);
    assert.ok(rpfTel.readAheadBytesServed >= 25 * 65536);
    assert.equal(rpfTel.currentWindowBytes, 2 * 1024 * 1024);

    // Total backend calls across 80 x 64 KiB reads must be dramatically reduced by adaptive read-ahead + coalescing!
    const bk2BackendCalls = backendCalls.filter((c) => c.handle === hBk2);
    const rpfBackendCalls = backendCalls.filter((c) => c.handle === hRpf);
    assert.ok(bk2BackendCalls.length <= 5, `Expected <= 5 backend calls for 48 .bk2 reads, got ${bk2BackendCalls.length}`);
    assert.ok(rpfBackendCalls.length <= 5, `Expected <= 5 backend calls for 32 .rpf reads, got ${rpfBackendCalls.length}`);

    // Verify .bk2 ramped all the way up to 2 MiB on the backend!
    assert.ok(
      bk2BackendCalls.some((c) => c.reqLen === 2 * 1024 * 1024),
      `Expected .bk2 stream to ramp to 2 MiB fetch window, saw: ${bk2BackendCalls.map((c) => c.reqLen)}`,
    );

    const hostStats = poolHost.getStats();
    const clientStats = poolClient.getStats();
    assert.equal(hostStats.slotCount, 4);
    assert.equal(hostStats.peakPendingSlots, 4);
    assert.equal(hostStats.burstReads, 78);
    assert.equal(hostStats.activePendingSlots, 0);
    assert.equal(hostStats.perSlot.length, 4);
    assert.ok(hostStats.perSlot.every((s) => s.requests > 0), 'All 4 pipeline slots should serve requests');
    assert.equal(clientStats.slotCount, 4);
    assert.equal(clientStats.peakPendingSlots, 4);
    assert.equal(clientStats.availableSlots, 4);

    assert.equal(await poolClient.closeAsync(hRpf), true);
    assert.equal(await poolClient.closeAsync(hBk2), true);

    // Confirm zero file paths/URLs leak into sanitized telemetry JSON.
    const serialized = JSON.stringify({ hostStats: poolHost.getStats(), clientStats: poolClient.getStats() });
    assert.equal(serialized.includes('intro_4k.bk2'), false);
    assert.equal(serialized.includes('ds_stream.rpf'), false);
  } finally {
    await poolHost.stop();
  }
});
