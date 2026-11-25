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
  HttpfsStatus,
  createHttpfsMailboxBuffer,
  HttpfsMailboxHost,
  HttpfsMailboxClient,
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
  assert.equal(DEFAULT_SLAB_MAX_BYTES, 32 * 1024 * 1024);

  assert.equal(isWholeFileBufferedPath('main/image/guest.pgim'), true);
  assert.equal(isWholeFileBufferedPath('boot/main/image/GUEST.PGIM'), true);
  assert.equal(isWholeFileBufferedPath('program-manifest.json'), true);
  assert.equal(isWholeFileBufferedPath('boot/program-manifest.json'), true);
  assert.equal(isWholeFileBufferedPath('appdata0.rpf'), false);
  assert.equal(isWholeFileBufferedPath('audio.ttpf'), false);
  assert.equal(isWholeFileBufferedPath('data-manifest.json'), false);

  // Immutable paths include .ttpf in addition to .rpf, .pgim, .dll, MicrosoftGame.Config
  assert.equal(isImmutableHttpfsPath('x64/audio/dialogue.ttpf'), true);
  assert.equal(isImmutableHttpfsPath('x64/audio/SFX.TTPF'), true);
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
      // 1 OPEN chunk-0 probe + expectedSlabFetches aligned slab fetches
      assert.equal(
        totalReqs,
        1 + expectedSlabFetches,
        `Expected ${1 + expectedSlabFetches} HTTP fetches for ${openPath}, got ${totalReqs}`,
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
    assert.equal(countFor('/game/pack0.rpf'), 2, 'RPF: 1 OPEN probe + 1 slab fetch (NOT 11!)');
    assert.equal(countFor('/game/audio.ttpf'), 2, 'TTPF: 1 OPEN probe + 1 slab fetch (NOT 7!)');

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
