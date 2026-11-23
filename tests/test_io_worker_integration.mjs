import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import vm from 'node:vm';
import { Worker as NodeWorker } from 'node:worker_threads';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');

const ioWorkerMod = require('../web/workers/io_worker.js');
const mailboxMod = require('../web/workers/httpfs_mailbox.js');
const rpfMod = require('../web/workers/rpf9_vfs.js');
const schedulerMod = require('../web/workers/io_scheduler.js');

const {
  IoWorkerController,
  IO_PRIORITY,
  isRpfVirtualPath,
  parseRpfVirtualPath,
} = ioWorkerMod;

const {
  HttpfsStatus,
  HttpfsOp,
  HttpfsOffsets,
  createHttpfsMailboxBuffer,
  HttpfsMailboxClient,
} = mailboxMod;

const {
  RPF9_MAGIC_LE,
  RPF9_MAGIC_BE,
  RPF_ENCRYPTION_OPEN,
  RPF_ENCRYPTION_AES,
  buildSyntheticRpf9Buffer,
} = rpfMod;

const utf8Encoder = new TextEncoder();
const utf8Decoder = new TextDecoder('utf-8');

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function makePatternBytes(length, seed = 23) {
  const out = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = ((i * 41) ^ ((i >> 7) * 17) ^ seed) & 0xff;
  }
  return out;
}

function createMockRangeFetch(filesMap, hooks = {}) {
  const requests = [];

  async function fetchImpl(url, init = {}) {
    const relPath = url.replace(/^.*\/data\//, '');
    const rangeHeader = (init.headers && (init.headers.Range || init.headers.range)) || null;
    requests.push({ url, relPath, rangeHeader });

    if (typeof hooks.onRequest === 'function') {
      await hooks.onRequest({ url, relPath, rangeHeader });
    }

    const data = filesMap.get(relPath);
    if (!data) {
      return {
        ok: false,
        status: 404,
        headers: new Map(),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
        async json() {
          throw new Error(`HTTP 404 for ${relPath}`);
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
        async json() {
          return JSON.parse(utf8Decoder.decode(data));
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

  return { fetchImpl, requests };
}

async function startEphemeralRangeServer(filesMap) {
  const requests = [];
  let totalRangeBytesServed = 0;

  const server = http.createServer((req, res) => {
    const reqUrl = new URL(req.url || '/', 'http://127.0.0.1');
    const relPath = reqUrl.pathname.replace(/^\/data\//, '');
    const rangeHeader = req.headers.range || null;
    requests.push({ method: req.method, path: reqUrl.pathname, relPath, rangeHeader });

    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
    res.setHeader('Accept-Ranges', 'bytes');

    const data = filesMap.get(relPath);
    if (!data) {
      res.writeHead(404);
      res.end();
      return;
    }

    if (!rangeHeader) {
      res.writeHead(200, {
        'Content-Length': String(data.byteLength),
      });
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
      res.writeHead(416, {
        'Content-Range': `bytes */${data.byteLength}`,
      });
      res.end();
      return;
    }

    const actualEnd = Math.min(reqEnd, data.byteLength - 1);
    const slice = data.subarray(reqStart, actualEnd + 1);
    totalRangeBytesServed += slice.byteLength;

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
  const baseUrl = `http://127.0.0.1:${addr.port}/data/`;

  return {
    baseUrl,
    requests,
    getTotalRangeBytesServed: () => totalRangeBytesServed,
    close: () => new Promise((resolve) => server.close(() => resolve())),
  };
}

// Helper to execute a raw OPEN on the mailbox SAB and return both status and signed errno/handle.
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

// ---------------------------------------------------------------------------
// 1. Virtual rpf:// URI parser unit checks
// ---------------------------------------------------------------------------

test('parseRpfVirtualPath splits simple and nested .rpf URIs and rejects traversal/malformed inputs', () => {
  assert.equal(isRpfVirtualPath('rpf://appdata0.rpf/config/settings.xml'), true);
  assert.equal(isRpfVirtualPath('RPF://archive.rpf/sub/file.bin'), true);
  assert.equal(isRpfVirtualPath('appdata0.rpf'), false);

  assert.deepEqual(parseRpfVirtualPath('rpf://appdata0.rpf/config/settings.xml'), {
    archivePath: 'appdata0.rpf',
    innerPath: 'config/settings.xml',
    mountPath: 'appdata0.rpf/config/settings.xml',
  });

  assert.deepEqual(parseRpfVirtualPath('rpf://update/x64/dlc.rpf/textures/hud.ytd'), {
    archivePath: 'update/x64/dlc.rpf',
    innerPath: 'textures/hud.ytd',
    mountPath: 'update/x64/dlc.rpf/textures/hud.ytd',
  });

  assert.deepEqual(parseRpfVirtualPath('rpf://archive.rpf/sub\\nested\\file.bin'), {
    archivePath: 'archive.rpf',
    innerPath: 'sub/nested/file.bin',
    mountPath: 'archive.rpf/sub/nested/file.bin',
  });

  assert.throws(() => parseRpfVirtualPath('rpf://../escape.rpf/sub/file.bin'));
  assert.throws(() => parseRpfVirtualPath('rpf://archive.rpf/sub/../../escape.bin'));
  assert.throws(() => parseRpfVirtualPath('rpf://archive.rpf/'));
  assert.throws(() => parseRpfVirtualPath('rpf://archive.rpf'));
  assert.throws(() => parseRpfVirtualPath('rpf://'));
});

// ---------------------------------------------------------------------------
// 2. Priority ordering: CRITICAL mailbox reads preempt queued PREFETCH bootset
// ---------------------------------------------------------------------------

test('IoPriorityScheduler priority ordering: CRITICAL mailbox reads preempt queued PREFETCH bootset reads and NORMAL reads', async () => {
  const boot0 = makePatternBytes(32, 1);
  const boot1 = makePatternBytes(32, 2);
  const boot2 = makePatternBytes(32, 3);
  const boot3 = makePatternBytes(32, 4);
  const criticalData = makePatternBytes(32, 99);
  const normalData = makePatternBytes(32, 55);

  const manifestObj = {
    abi_version: 1,
    file_count: 6,
    files: [
      { path: 'boot_0.bin', size: boot0.byteLength },
      { path: 'boot_1.bin', size: boot1.byteLength },
      { path: 'boot_2.bin', size: boot2.byteLength },
      { path: 'boot_3.bin', size: boot3.byteLength },
      { path: 'critical.bin', size: criticalData.byteLength },
      { path: 'normal.bin', size: normalData.byteLength },
    ],
  };

  const bootsetArr = [
    { path: 'boot_0.bin', offset: 0, length: 32 },
    { path: 'boot_1.bin', offset: 0, length: 32 },
    { path: 'boot_2.bin', offset: 0, length: 32 },
    { path: 'boot_3.bin', offset: 0, length: 32 },
  ];

  const files = new Map([
    ['data-manifest.json', utf8Encoder.encode(JSON.stringify(manifestObj))],
    ['bootset.json', utf8Encoder.encode(JSON.stringify(bootsetArr))],
    ['boot_0.bin', boot0],
    ['boot_1.bin', boot1],
    ['boot_2.bin', boot2],
    ['boot_3.bin', boot3],
    ['critical.bin', criticalData],
    ['normal.bin', normalData],
  ]);

  const boot0Started = deferred();
  const boot0Release = deferred();
  const rangeFetchOrder = [];

  const mock = createMockRangeFetch(files, {
    async onRequest({ relPath, rangeHeader }) {
      if (!rangeHeader) return;
      rangeFetchOrder.push(relPath);
      if (relPath === 'boot_0.bin') {
        boot0Started.resolve();
        await boot0Release.promise;
      }
    },
  });

  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    maxConcurrency: 1,
    chunkSize: 32,
    maxBytes: 4096,
    opfsDir: null,
  });

  const sab = createHttpfsMailboxBuffer(64);
  const client = new HttpfsMailboxClient(sab);

  try {
    const ready = await controller.init({ dataBase: '/data/' });
    assert.equal(ready.ready, true);
    assert.equal(ready.bootsetEntries, 4);
    assert.equal(ready.scheduler, true);
    assert.equal(ready.rpfVfs, true);

    await controller.attachMailbox(sab);

    // Wait until boot_0.bin occupies the single concurrency slot while boot_1..boot_3 are queued.
    await boot0Started.promise;

    const midStats = controller.getStats();
    assert.equal(midStats.scheduler.inFlight, 1);
    assert.equal(midStats.scheduler.byPriority.PREFETCH.inFlight, 1);
    assert.equal(midStats.scheduler.byPriority.PREFETCH.queued, 3);

    // Queue a NORMAL streaming read first, then a blocking CRITICAL HttpfsMailbox READ.
    const normalPromise = controller.read('normal.bin', 0, 32); // defaults to PRIORITY.NORMAL

    const handle = await client.openAsync('critical.bin');
    assert.ok(handle >= 0);

    const critBuf = new Uint8Array(32);
    const critReadPromise = client.readAsync(handle, 0n, critBuf, 32);

    // Wait briefly so the mailbox host picks up the READ request and schedules it at CRITICAL.
    await new Promise((r) => setTimeout(r, 15));

    const queuedStats = controller.getStats();
    assert.equal(queuedStats.scheduler.byPriority.CRITICAL.queued, 1);
    assert.equal(queuedStats.scheduler.byPriority.NORMAL.queued, 1);
    assert.equal(queuedStats.scheduler.byPriority.PREFETCH.queued, 3);

    // Unblock boot_0.bin -> scheduler MUST dispatch CRITICAL first, then NORMAL, then PREFETCH!
    boot0Release.resolve();

    const [critLen, normalBytes] = await Promise.all([critReadPromise, normalPromise]);
    await controller.prefetchPromise;

    assert.equal(critLen, 32);
    assert.deepEqual(critBuf, criticalData);
    assert.deepEqual(normalBytes, normalData);

    assert.deepEqual(rangeFetchOrder, [
      'boot_0.bin',
      'critical.bin',
      'normal.bin',
      'boot_1.bin',
      'boot_2.bin',
      'boot_3.bin',
    ]);

    assert.equal(await client.closeAsync(handle), true);

    const finalStats = controller.getStats();
    assert.equal(finalStats.scheduler.completed, 6);
    assert.equal(finalStats.scheduler.byPriority.CRITICAL.completed, 1);
    assert.equal(finalStats.scheduler.byPriority.NORMAL.completed, 1);
    assert.equal(finalStats.scheduler.byPriority.PREFETCH.completed, 4);
    assert.equal(finalStats.openHandles, 0);
  } finally {
    await controller.detachMailbox();
  }
});

test('bootset PREFETCH queue is cancellable via cancelPrefetch() without disrupting CRITICAL reads', async () => {
  const files = new Map([
    ['bootset.json', utf8Encoder.encode(JSON.stringify([
      { path: 'p0.bin', offset: 0, length: 16 },
      { path: 'p1.bin', offset: 0, length: 16 },
      { path: 'p2.bin', offset: 0, length: 16 },
    ]))],
    ['p0.bin', makePatternBytes(16, 1)],
    ['p1.bin', makePatternBytes(16, 2)],
    ['p2.bin', makePatternBytes(16, 3)],
    ['urgent.bin', makePatternBytes(16, 88)],
  ]);

  const p0Started = deferred();
  const p0Release = deferred();
  const fetched = [];

  const mock = createMockRangeFetch(files, {
    async onRequest({ relPath, rangeHeader }) {
      if (!rangeHeader) return;
      fetched.push(relPath);
      if (relPath === 'p0.bin') {
        p0Started.resolve();
        await p0Release.promise;
      }
    },
  });

  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    maxConcurrency: 1,
    chunkSize: 16,
    opfsDir: null,
  });

  await controller.init({ dataBase: '/data/' });
  await p0Started.promise;

  // Cancel queued p1 & p2 while p0 is in-flight, and schedule a CRITICAL read.
  const cancelled = controller.cancelPrefetch();
  assert.equal(cancelled, 2);

  const urgentPromise = controller.read('urgent.bin', 0, 16, {
    priority: IO_PRIORITY.CRITICAL,
  });

  p0Release.resolve();
  const [urgentBytes, prefetchSummary] = await Promise.all([
    urgentPromise,
    controller.prefetchPromise,
  ]);

  assert.deepEqual(urgentBytes, files.get('urgent.bin'));
  assert.deepEqual(fetched, ['p0.bin', 'urgent.bin']);
  assert.equal(prefetchSummary.completed, 1);
  assert.equal(prefetchSummary.cancelled, 2);

  const stats = controller.getStats();
  assert.equal(stats.scheduler.cancelled, 2);
  assert.equal(stats.scheduler.byPriority.PREFETCH.cancelled, 2);
  assert.equal(stats.scheduler.byPriority.CRITICAL.completed, 1);
});

// ---------------------------------------------------------------------------
// 3. Synthetic LE "9FPR" archive over real ephemeral HTTP Range server via HttpfsMailbox
// ---------------------------------------------------------------------------

test('HttpfsMailbox OPEN -> READ -> CLOSE transparently streams sub-files from LE "9FPR" archive over real HTTP 206 server', async () => {
  const subFileBytes = makePatternBytes(1500, 42);
  const xmlText = '<?xml version="1.0"?><Settings><Quality>Ultra</Quality></Settings>';
  const unusedHuge = makePatternBytes(48 * 1024, 7);

  const leArchive = buildSyntheticRpf9Buffer(
    [
      { path: 'sub/file.bin', data: subFileBytes },
      { path: 'sub/empty.dat', data: new Uint8Array(0) },
      { path: 'config/settings.xml', data: xmlText },
      { path: 'unused/giant_blob.bin', data: unusedHuge },
    ],
    {
      magic: RPF9_MAGIC_LE, // Real RAGE little-endian "9FPR" (0x52504639 LE)
      encryptionTag: RPF_ENCRYPTION_OPEN,
      sectorAlign: true,
    },
  );

  // Confirm raw archive header begins with exact LE "9FPR" bytes [0x39, 0x46, 0x50, 0x52].
  assert.deepEqual(Array.from(leArchive.subarray(0, 4)), [0x39, 0x46, 0x50, 0x52]);

  const manifestObj = {
    abi_version: 1,
    file_count: 1,
    total_bytes: leArchive.byteLength,
    files: [{ path: 'archive.rpf', size: leArchive.byteLength }],
  };

  const server = await startEphemeralRangeServer(
    new Map([
      ['archive.rpf', leArchive],
      ['data-manifest.json', utf8Encoder.encode(JSON.stringify(manifestObj))],
    ]),
  );

  const controller = new IoWorkerController({
    chunkSize: 1024,
    maxBytes: 1024 * 1024,
    opfsDir: null,
  });

  // Use 256-byte mailbox payload capacity so 1500-byte sub-file spans 6 SAB roundtrips.
  const sab = createHttpfsMailboxBuffer(256);
  const client = new HttpfsMailboxClient(sab);

  try {
    const ready = await controller.init({ dataBase: server.baseUrl });
    assert.equal(ready.ready, true);
    assert.equal(controller.getStats().mountedRpfCount, 0);

    await controller.attachMailbox(sab);

    // 1. OPEN virtual RPF path -> lazily mounts archive.rpf via RangeCache + IoPriorityScheduler
    const handle = await client.openAsync('rpf://archive.rpf/sub/file.bin');
    assert.ok(handle >= 1, `Expected valid handle >= 1, got ${handle}`);

    const handleInfo = controller.getHandleInfo(handle);
    assert.equal(handleInfo.kind, 'rpf');
    assert.equal(handleInfo.archivePath, 'archive.rpf');
    assert.equal(handleInfo.innerPath, 'sub/file.bin');
    assert.equal(handleInfo.size, subFileBytes.byteLength);

    const fileStat = await controller.stat('rpf://archive.rpf/sub/file.bin');
    assert.ok(fileStat);
    assert.equal(fileStat.isFile, true);
    assert.equal(fileStat.size, subFileBytes.byteLength);

    // 2. Full READ requesting larger buffer (4096B) clamps cleanly to exact file size (1500B)
    const fullBuf = new Uint8Array(4096);
    const fullRead = await client.readAsync(handle, 0n, fullBuf, 4096);
    assert.equal(fullRead, subFileBytes.byteLength);
    assert.deepEqual(fullBuf.subarray(0, fullRead), subFileBytes);

    // 3. Partial offset READ & tail EOF clamp & past-EOF 0-byte read
    const midBuf = new Uint8Array(200);
    const midRead = await client.readAsync(handle, 400n, midBuf, 200);
    assert.equal(midRead, 200);
    assert.deepEqual(midBuf, subFileBytes.subarray(400, 600));

    const tailBuf = new Uint8Array(200);
    const tailRead = await client.readAsync(handle, 1450n, tailBuf, 200);
    assert.equal(tailRead, 50);
    assert.deepEqual(tailBuf.subarray(0, 50), subFileBytes.subarray(1450, 1500));

    const eofBuf = new Uint8Array(32);
    assert.equal(await client.readAsync(handle, 1500n, eofBuf, 32), 0);

    // 4. Case-insensitive inner path & XML sub-file & 0-byte sub-file
    const xmlHandle = await client.openAsync('rpf://archive.rpf/CONFIG/SETTINGS.XML');
    assert.ok(xmlHandle >= 1);
    const xmlOut = new Uint8Array(256);
    const xmlRead = await client.readAsync(xmlHandle, 0n, xmlOut, 256);
    assert.equal(xmlRead, utf8Encoder.encode(xmlText).byteLength);
    assert.equal(utf8Decoder.decode(xmlOut.subarray(0, xmlRead)), xmlText);

    const emptyHandle = await client.openAsync('rpf://archive.rpf/sub/empty.dat');
    assert.ok(emptyHandle >= 1);
    assert.equal(await client.readAsync(emptyHandle, 0n, eofBuf, 32), 0);

    // 5. Verify lazy HTTP 206 streaming NEVER downloaded the 48 KiB unused_large blob!
    assert.ok(
      server.getTotalRangeBytesServed() < 8 * 1024,
      `Expected < 8 KiB streamed over HTTP 206, got ${server.getTotalRangeBytesServed()}`,
    );

    // 6. Repeat read of sub/file.bin hits RangeCache with zero new HTTP requests
    const reqsBeforeRepeat = server.requests.length;
    const hitsBeforeRepeat = controller.getStats().hits;
    const repeatBuf = new Uint8Array(1500);
    const repeatLen = await client.readAsync(handle, 0n, repeatBuf, 1500);
    assert.equal(repeatLen, 1500);
    assert.deepEqual(repeatBuf, subFileBytes);
    assert.equal(server.requests.length, reqsBeforeRepeat);
    assert.ok(controller.getStats().hits > hitsBeforeRepeat);

    // 7. CLOSE handles and verify post-close READ/CLOSE fail cleanly
    assert.equal(await client.closeAsync(handle), true);
    assert.equal(await client.closeAsync(xmlHandle), true);
    assert.equal(await client.closeAsync(emptyHandle), true);
    assert.equal(await client.closeAsync(handle), false);
    assert.equal(await client.readAsync(handle, 0n, eofBuf, 16), -9);

    // 8. Verify telemetry getStats()
    const stats = controller.getStats();
    assert.equal(stats.mountedRpfCount, 1);
    assert.equal(stats.rpf.mountedCount, 1);
    assert.equal(stats.rpf.mountErrors, 0);
    assert.equal(stats.rpf.archives.length, 1);
    assert.equal(stats.rpf.archives[0].archivePath, 'archive.rpf');
    assert.equal(stats.rpf.archives[0].magic, 'RPF9');
    assert.equal(stats.rpf.archives[0].rawMagic, '9FPR');
    assert.equal(stats.rpf.archives[0].magicEndian, 'little');
    assert.ok(stats.hits > 0);
    assert.equal(stats.rangeCache.hits, stats.hits);
    assert.ok(stats.scheduler.completed > 0);
    assert.equal(stats.scheduler.byPriority.CRITICAL.completed, stats.scheduler.completed);
    assert.equal(stats.openHandles, 0);
  } finally {
    await controller.detachMailbox();
    await server.close();
  }
});

// ---------------------------------------------------------------------------
// 4. Both LE "9FPR" and BE "RPF9" archives + nested archive path + cross-thread Atomics.wait
// ---------------------------------------------------------------------------

test('supports both LE "9FPR" and BE "RPF9" archives simultaneously and cross-thread synchronous Atomics.wait reads', async () => {
  const lePayload = utf8Encoder.encode('{"engine":"RAGE9","endian":"little"}');
  const bePayload = makePatternBytes(310, 101);

  const leArchive = buildSyntheticRpf9Buffer(
    [{ path: 'config/settings.xml', data: lePayload }],
    { magic: RPF9_MAGIC_LE, encryptionTag: RPF_ENCRYPTION_OPEN },
  );
  const beArchive = buildSyntheticRpf9Buffer(
    [{ path: 'shaders/core.bin', data: bePayload }],
    { magic: RPF9_MAGIC_BE },
  );

  const mock = createMockRangeFetch(
    new Map([
      ['appdata0.rpf', leArchive],
      ['update/patch.rpf', beArchive],
    ]),
  );

  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    chunkSize: 128,
    maxBytes: 64 * 1024,
    opfsDir: null,
  });

  const sab = createHttpfsMailboxBuffer(128);
  await controller.init({ dataBase: '/data/' });
  await controller.attachMailbox(sab);

  const workerCode = `
    const { workerData, parentPort } = require('node:worker_threads');
    const { HttpfsMailboxClient } = require(workerData.mailboxPath);
    const client = new HttpfsMailboxClient(workerData.sab, { timeoutMs: 5000 });

    const hLe = client.open('rpf://appdata0.rpf/config/settings.xml');
    const leBuf = new Uint8Array(256);
    const leLen = client.read(hLe, 0n, leBuf, 256);
    const leClosed = client.close(hLe);

    const hBe = client.open('rpf://update/patch.rpf/shaders/core.bin');
    const beBuf = new Uint8Array(512);
    const beLen = client.read(hBe, 0n, beBuf, 512);
    const beClosed = client.close(hBe);

    parentPort.postMessage({
      hLe,
      leLen,
      leBytes: Array.from(leBuf.subarray(0, Math.max(0, leLen))),
      leClosed,
      hBe,
      beLen,
      beBytes: Array.from(beBuf.subarray(0, Math.max(0, beLen))),
      beClosed,
    });
  `;

  try {
    const result = await new Promise((resolve, reject) => {
      const w = new NodeWorker(workerCode, {
        eval: true,
        workerData: {
          sab,
          mailboxPath: path.join(repoRoot, 'web', 'workers', 'httpfs_mailbox.js'),
        },
      });
      w.once('message', resolve);
      w.once('error', reject);
      w.once('exit', (code) => {
        if (code !== 0) reject(new Error(`Worker exited with code ${code}`));
      });
    });

    assert.ok(result.hLe >= 1);
    assert.equal(result.leLen, lePayload.byteLength);
    assert.deepEqual(new Uint8Array(result.leBytes), lePayload);
    assert.equal(result.leClosed, true);

    assert.ok(result.hBe >= 1);
    assert.equal(result.beLen, bePayload.byteLength);
    assert.deepEqual(new Uint8Array(result.beBytes), bePayload);
    assert.equal(result.beClosed, true);

    const stats = controller.getStats();
    assert.equal(stats.mountedRpfCount, 2);
    assert.equal(stats.rpf.mountedCount, 2);
    const byPath = new Map(stats.rpf.archives.map((a) => [a.archivePath, a]));
    assert.equal(byPath.get('appdata0.rpf').rawMagic, '9FPR');
    assert.equal(byPath.get('appdata0.rpf').magicEndian, 'little');
    assert.equal(byPath.get('update/patch.rpf').rawMagic, 'RPF9');
    assert.equal(byPath.get('update/patch.rpf').magicEndian, 'big');
  } finally {
    await controller.detachMailbox();
  }
});

// ---------------------------------------------------------------------------
// 5. Error handling for missing virtual paths / missing archives / directories / encrypted / corrupt
// ---------------------------------------------------------------------------

test('missing virtual paths, missing archives, directories, encrypted archives, and traversal return clean negative errnos without hanging mailbox', async () => {
  const validArchive = buildSyntheticRpf9Buffer(
    [{ path: 'sub/ok.bin', data: utf8Encoder.encode('ALIVE_OK') }],
    { magic: RPF9_MAGIC_LE },
  );
  const encryptedArchive = buildSyntheticRpf9Buffer(
    [{ path: 'secret/file.bin', data: 'topsecret' }],
    { magic: RPF9_MAGIC_LE, encryptionTag: RPF_ENCRYPTION_AES },
  );
  const corruptArchive = utf8Encoder.encode('NOT_AN_RPF_ARCHIVE_AT_ALL_1234567890');

  const mock = createMockRangeFetch(
    new Map([
      ['valid.rpf', validArchive],
      ['encrypted.rpf', encryptedArchive],
      ['corrupt.rpf', corruptArchive],
    ]),
  );

  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    chunkSize: 128,
    opfsDir: null,
  });

  const sab = createHttpfsMailboxBuffer(256);
  const client = new HttpfsMailboxClient(sab);

  try {
    await controller.init({ dataBase: '/data/' });
    await controller.attachMailbox(sab);

    // 1. Missing sub-file inside a valid archive -> ENOENT (-2)
    const missingInner = await rawMailboxOpen(sab, 'rpf://valid.rpf/sub/missing.bin');
    assert.equal(missingInner.status, HttpfsStatus.Error);
    assert.equal(missingInner.result, -2n);

    // 2. Missing archive -> ENOENT (-2)
    const missingArchive = await rawMailboxOpen(sab, 'rpf://does_not_exist.rpf/sub/file.bin');
    assert.equal(missingArchive.status, HttpfsStatus.Error);
    assert.equal(missingArchive.result, -2n);

    // 3. Opening a directory inside a valid archive -> EISDIR (-21)
    const dirOpen = await rawMailboxOpen(sab, 'rpf://valid.rpf/sub');
    assert.equal(dirOpen.status, HttpfsStatus.Error);
    assert.equal(dirOpen.result, -21n);

    // 4. Path traversal & malformed rpf:// URIs -> EINVAL (-22)
    for (const badUri of [
      'rpf://../escape.rpf/sub/file.bin',
      'rpf://valid.rpf/sub/../../escape.bin',
      'rpf://valid.rpf/',
      'rpf://valid.rpf',
      'rpf://',
    ]) {
      const res = await rawMailboxOpen(sab, badUri);
      assert.equal(res.status, HttpfsStatus.Error, `Expected Error for ${badUri}`);
      assert.equal(res.result, -22n, `Expected -22 (EINVAL) for ${badUri}, got ${res.result}`);
    }

    // 5. Encrypted archive without decryptBlock hook -> EACCES (-13)
    const encRes = await rawMailboxOpen(sab, 'rpf://encrypted.rpf/secret/file.bin');
    assert.equal(encRes.status, HttpfsStatus.Error);
    assert.equal(encRes.result, -13n);

    // 6. Corrupt / non-RPF archive -> EIO (-5)
    const corruptRes = await rawMailboxOpen(sab, 'rpf://corrupt.rpf/a.bin');
    assert.equal(corruptRes.status, HttpfsStatus.Error);
    assert.equal(corruptRes.result, -5n);

    // 7. Also verify HttpfsMailboxClient.openAsync returns -1 cleanly for missing archive & missing inner path
    assert.equal(await client.openAsync('rpf://no_such.rpf/a.bin'), -1);
    assert.equal(await client.openAsync('rpf://valid.rpf/nope.bin'), -1);

    // 8. Mailbox must remain completely healthy after all errors and serve a valid file immediately!
    const h = await client.openAsync('rpf://valid.rpf/sub/ok.bin');
    assert.ok(h >= 1, `Mailbox should stay healthy after errors, got handle ${h}`);
    const buf = new Uint8Array(32);
    const n = await client.readAsync(h, 0n, buf, 32);
    assert.equal(n, 8);
    assert.equal(utf8Decoder.decode(buf.subarray(0, n)), 'ALIVE_OK');
    assert.equal(await client.closeAsync(h), true);

    const stats = controller.getStats();
    assert.equal(stats.mountedRpfCount, 1);
    assert.ok(stats.rpf.mountErrors >= 3);
    assert.equal(stats.openHandles, 0);
  } finally {
    await controller.detachMailbox();
  }
});

// ---------------------------------------------------------------------------
// 6. Classic Web Worker self.onmessage + importScripts (with XHR isolated fallback) integration
// ---------------------------------------------------------------------------

test('classic Web Worker self.onmessage handles init, attachMailbox, rpf:// read/stat/listDir, and telemetry stats', async () => {
  const payload = makePatternBytes(200, 61);
  const leArchive = buildSyntheticRpf9Buffer(
    [{ path: 'assets/data.bin', data: payload }],
    { magic: RPF9_MAGIC_LE },
  );
  const directBlob = makePatternBytes(64, 19);

  const mock = createMockRangeFetch(
    new Map([
      ['pack.rpf', leArchive],
      ['direct.bin', directBlob],
      ['bootset.json', utf8Encoder.encode(JSON.stringify([{ path: 'direct.bin', offset: 0, length: 64 }]))],
    ]),
  );

  const posted = [];
  const workerSelf = {
    onmessage: null,
    postMessage(msg) {
      posted.push(msg);
    },
  };

  // Simulate a real browser classic Worker's XMLHttpRequest fallback for sibling script isolation.
  class MockSyncXHR {
    open(method, url, asyncFlag) {
      this._url = url;
      this._async = asyncFlag;
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

  // Verify all 4 sibling modules loaded cleanly in the classic Worker environment!
  assert.ok(sandbox.PlayGta6RangeCache);
  assert.ok(sandbox.PlayGta6HttpfsMailbox);
  assert.ok(sandbox.PlayGta6IoScheduler);
  assert.ok(sandbox.PlayGta6Rpf9Vfs);
  assert.ok(sandbox.PlayGta6IoWorker);

  await workerSelf.onmessage({
    data: { init: true, dataBase: '/data/', chunkSize: 64, maxBytes: 16384 },
  });
  assert.equal(posted[0].ready, true);
  assert.equal(posted[0].scheduler, true);
  assert.equal(posted[0].rpfVfs, true);
  assert.equal(posted[0].bootsetEntries, 1);

  // Wait briefly for bootset prefetch to finish.
  await new Promise((r) => setTimeout(r, 20));

  // Direct read via message API (hits prefetched cache)
  await workerSelf.onmessage({
    data: { read: true, id: 1, path: 'direct.bin', offset: 10, length: 20 },
  });
  assert.equal(posted[1].id, 1);
  assert.deepEqual(new Uint8Array(posted[1].bytes), directBlob.subarray(10, 30));

  // Virtual rpf:// read via message API (lazily mounts pack.rpf)
  await workerSelf.onmessage({
    data: { read: true, id: 2, path: 'rpf://pack.rpf/assets/data.bin', offset: 50, length: 40 },
  });
  assert.equal(posted[2].id, 2);
  assert.deepEqual(new Uint8Array(posted[2].bytes), payload.subarray(50, 90));

  // Virtual rpf:// stat and listDir via message API
  await workerSelf.onmessage({
    data: { stat: true, id: 3, path: 'rpf://pack.rpf/assets/data.bin' },
  });
  assert.equal(posted[3].id, 3);
  assert.equal(posted[3].stat.size, payload.byteLength);

  await workerSelf.onmessage({
    data: { listDir: true, id: 4, path: 'rpf://pack.rpf/assets' },
  });
  assert.equal(posted[4].id, 4);
  assert.deepEqual(
    Array.from(posted[4].entries, (e) => e.name),
    ['data.bin'],
  );

  // Attach SAB mailbox and read the same rpf:// sub-file through HttpfsMailbox
  const sab = createHttpfsMailboxBuffer(64);
  await workerSelf.onmessage({ data: { attachMailbox: true, id: 5, sab } });
  assert.equal(posted[5].id, 5);
  assert.equal(posted[5].mailboxAttached, true);

  const client = new HttpfsMailboxClient(sab);
  try {
    const h = await client.openAsync('rpf://pack.rpf/assets/data.bin');
    assert.ok(h >= 1);
    const out = new Uint8Array(256);
    const got = await client.readAsync(h, 0n, out, 256);
    assert.equal(got, payload.byteLength);
    assert.deepEqual(out.subarray(0, got), payload);
    assert.equal(await client.closeAsync(h), true);

    // Query stats via worker message
    await workerSelf.onmessage({ data: { stats: true, id: 6 } });
    const statsMsg = posted[6];
    assert.equal(statsMsg.id, 6);
    assert.ok(statsMsg.stats.hits > 0);
    assert.equal(statsMsg.stats.mountedRpfCount, 1);
    assert.equal(statsMsg.stats.rpf.mountedCount, 1);
    assert.ok(statsMsg.stats.scheduler.completed > 0);
    assert.ok(statsMsg.stats.scheduler.byPriority.CRITICAL.completed > 0);
    assert.ok(statsMsg.stats.scheduler.byPriority.NORMAL.completed > 0);
    assert.ok(statsMsg.stats.scheduler.byPriority.PREFETCH.completed > 0);
  } finally {
    await workerSelf.onmessage({ data: { detachMailbox: true, id: 7 } });
    assert.equal(posted[7].id, 7);
    assert.equal(posted[7].mailboxDetached, true);
  }
});

// Keep reference check so schedulerMod import verifies exact priority constant alignment.
test('io_worker IO_PRIORITY matches io_scheduler IO_PRIORITY constants', () => {
  assert.deepEqual(IO_PRIORITY, schedulerMod.IO_PRIORITY);
});
