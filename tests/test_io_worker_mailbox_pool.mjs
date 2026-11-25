import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');

const { IoWorkerController } = require('../web/workers/io_worker.js');
const {
  HTTPFS_HEADER_BYTES,
  HttpfsStatus,
  HttpfsOffsets,
  createHttpfsMailboxBuffer,
  createHttpfsMailboxPool,
  HttpfsMailboxClient,
  HttpfsMailboxPoolHost,
} = require('../web/workers/httpfs_mailbox.js');
const {
  DEFAULT_HTTPFS_MAILBOX_LANES,
  PTHREAD_HTTPFS_SAB_CHANNEL_NAME,
  normalizeHttpfsMailboxLanes,
  allocateHttpfsMailboxPool,
  bootstrapEmscriptenPthreadWorker,
  createPthreadPoolCoordinator,
  installPthreadWorkerProxy,
} = require('../web/loader.js');

function createDeferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function makePatternBytes(size, fillByte) {
  const buf = new Uint8Array(size);
  for (let i = 0; i < size; i += 1) {
    buf[i] = (fillByte + (i & 0x0f)) & 0xff;
  }
  return buf;
}

/**
 * Create a mock HTTP Range (206) fetch that optionally pauses on matching URLs
 * so tests can deterministically prove multiple HTTP Range reads are in-flight
 * concurrently across independent mailbox lanes.
 */
function createGatedRangeFetch(fileMap, options = {}) {
  const gates = options.gates || new Map();
  let activeFetches = 0;
  let peakInFlight = 0;
  const requests = [];
  const arrivedListeners = [];

  function notifyArrived() {
    for (let i = arrivedListeners.length - 1; i >= 0; i -= 1) {
      const listener = arrivedListeners[i];
      if (listener.predicate()) {
        arrivedListeners.splice(i, 1);
        listener.resolve();
      }
    }
  }

  function waitForInFlight(targetCount, timeoutMs = 3000) {
    if (activeFetches >= targetCount) {
      return Promise.resolve();
    }
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(
          new Error(
            `Timed out waiting for ${targetCount} concurrent in-flight fetches (active=${activeFetches}, peak=${peakInFlight})`,
          ),
        );
      }, timeoutMs);
      arrivedListeners.push({
        predicate: () => activeFetches >= targetCount,
        resolve: () => {
          clearTimeout(timer);
          resolve();
        },
      });
    });
  }

  async function fetchImpl(url, init = {}) {
    const urlStr = String(url);
    const method = (init.method || 'GET').toUpperCase();
    const headers = init.headers || {};
    const rangeHeader = headers.Range || headers.range || null;

    if (urlStr.endsWith('data-manifest.json') || urlStr.endsWith('bootset.json')) {
      return {
        ok: false,
        status: 404,
        headers: { get: () => null },
        async json() {
          return null;
        },
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }

    const bytes = fileMap.get(urlStr);
    if (!bytes) {
      return {
        ok: false,
        status: 404,
        headers: { get: () => null },
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }

    if (method === 'HEAD') {
      return {
        ok: true,
        status: 200,
        headers: {
          get: (k) => (k.toLowerCase() === 'content-length' ? String(bytes.byteLength) : null),
        },
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }

    requests.push({ url: urlStr, range: rangeHeader });
    activeFetches += 1;
    if (activeFetches > peakInFlight) {
      peakInFlight = activeFetches;
    }
    notifyArrived();

    try {
      const gate = gates.get(urlStr);
      if (gate && gate.promise) {
        await gate.promise;
      }

      let start = 0;
      let end = bytes.byteLength - 1;
      let status = 200;
      if (rangeHeader) {
        const match = /^bytes=(\d+)-(\d*)$/.exec(String(rangeHeader));
        if (match) {
          start = Number(match[1]);
          end = match[2] ? Math.min(Number(match[2]), bytes.byteLength - 1) : bytes.byteLength - 1;
          status = 206;
        }
      }
      if (start >= bytes.byteLength) {
        return {
          ok: false,
          status: 416,
          headers: {
            get: (k) =>
              k.toLowerCase() === 'content-range' ? `bytes */${bytes.byteLength}` : null,
          },
          async arrayBuffer() {
            return new ArrayBuffer(0);
          },
        };
      }
      const slice = bytes.slice(start, end + 1);
      return {
        ok: true,
        status,
        headers: {
          get: (k) => {
            const lower = k.toLowerCase();
            if (lower === 'content-range') {
              return `bytes ${start}-${end}/${bytes.byteLength}`;
            }
            if (lower === 'content-length') {
              return String(slice.byteLength);
            }
            return null;
          },
        },
        async arrayBuffer() {
          return slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength);
        },
      };
    } finally {
      activeFetches -= 1;
    }
  }

  return {
    fetchImpl,
    requests,
    waitForInFlight,
    getActiveFetches: () => activeFetches,
    getPeakInFlight: () => peakInFlight,
  };
}

function createMockBroadcastBus() {
  const channelsByName = new Map();
  const postedLog = [];

  class BusBroadcastChannel {
    constructor(name) {
      this.name = name;
      this.onmessage = null;
      this.closed = false;
      if (!channelsByName.has(name)) {
        channelsByName.set(name, new Set());
      }
      channelsByName.get(name).add(this);
    }

    postMessage(data) {
      if (this.closed) return;
      postedLog.push({ channel: this.name, data });
      const peers = channelsByName.get(this.name);
      if (!peers) return;
      for (const peer of Array.from(peers)) {
        if (peer === this || peer.closed) continue;
        if (typeof peer.onmessage === 'function') {
          peer.onmessage({ data });
        }
      }
    }

    close() {
      this.closed = true;
      const peers = channelsByName.get(this.name);
      if (peers) {
        peers.delete(this);
      }
    }
  }

  return { BusBroadcastChannel, postedLog, channelsByName };
}

function bootCoordinatorVm(options = {}) {
  const loaderPath = path.join(repoRoot, 'web/loader.js');
  const loaderSource = fs.readFileSync(loaderPath, 'utf8');
  const bus = createMockBroadcastBus();
  const spawnedWorkers = [];

  class MockWorker {
    constructor(script, opts) {
      this.script = script;
      this.opts = opts;
      this.name = opts && typeof opts.name === 'string' ? opts.name : '';
      this.onmessage = null;
      this.onerror = null;
      this.posted = [];
      this.terminated = false;
      spawnedWorkers.push(this);
    }
    postMessage(msg, transfer) {
      this.posted.push({ msg, transfer });
    }
    terminate() {
      this.terminated = true;
    }
  }

  const workerSelf = {
    name: '',
    location: { href: 'http://127.0.0.1:8080/loader.js' },
    Worker: MockWorker,
    onmessage: null,
    postMessage() {},
  };

  const sandbox = {
    self: workerSelf,
    globalThis: null,
    URL,
    BroadcastChannel: bus.BusBroadcastChannel,
    Worker: MockWorker,
    indexedDB: null,
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
    Map,
    Set,
    Promise,
    Error,
    RangeError,
    TypeError,
    Number,
    Math,
    Date,
    Object,
    Array,
    Proxy,
    Reflect,
    JSON,
    setTimeout,
    clearTimeout,
    fetch: async () => ({ ok: false, status: 404 }),
    importScripts(scriptPath) {
      const resolved = path.resolve(repoRoot, 'web', scriptPath);
      vm.runInContext(fs.readFileSync(resolved, 'utf8'), context, { filename: resolved });
      if (
        options.hideCreateHttpfsMailboxPool &&
        scriptPath.includes('httpfs_mailbox.js') &&
        sandbox.PlayGta6HttpfsMailbox
      ) {
        const cloned = Object.assign({}, sandbox.PlayGta6HttpfsMailbox);
        delete cloned.createHttpfsMailboxPool;
        sandbox.PlayGta6HttpfsMailbox = cloned;
      }
    },
  };
  sandbox.globalThis = sandbox;
  const context = vm.createContext(sandbox);
  vm.runInContext(loaderSource, context, { filename: loaderPath });

  return { workerSelf, sandbox, context, bus, spawnedWorkers };
}

test('W260: io_worker.attachMailbox(poolSabs) serves concurrent reads on lanes 0, 1, 2 simultaneously (in-flight >= 2)', async () => {
  const data0 = makePatternBytes(4096, 0x10);
  const data1 = makePatternBytes(4096, 0x50);
  const data2 = makePatternBytes(4096, 0x90);

  const gate0 = createDeferred();
  const gate1 = createDeferred();
  const gate2 = createDeferred();

  const fileMap = new Map([
    ['/data/lane0.bin', data0],
    ['/data/lane1.bin', data1],
    ['/data/lane2.bin', data2],
  ]);
  const gates = new Map([
    ['/data/lane0.bin', gate0],
    ['/data/lane1.bin', gate1],
    ['/data/lane2.bin', gate2],
  ]);

  const mock = createGatedRangeFetch(fileMap, { gates });
  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    maxConcurrency: 8,
  });
  await controller.init({
    dataBase: '/data/',
    manifest: {
      files: [
        { path: 'lane0.bin', size: data0.byteLength },
        { path: 'lane1.bin', size: data1.byteLength },
        { path: 'lane2.bin', size: data2.byteLength },
      ],
    },
    bootset: [],
  });

  const poolSabs = createHttpfsMailboxPool(4, 64 * 1024);
  const host = await controller.attachMailbox(poolSabs, { probeOnOpen: false });
  assert.ok(host instanceof HttpfsMailboxPoolHost, 'multi-SAB attachMailbox must instantiate HttpfsMailboxPoolHost');
  assert.equal(controller.mailboxLanes, 4);
  assert.equal(controller.getStats().mailboxLanes, 4);
  assert.equal(controller.getStats().mailbox.slotCount, 4);

  const client0 = new HttpfsMailboxClient(poolSabs[0]);
  const client1 = new HttpfsMailboxClient(poolSabs[1]);
  const client2 = new HttpfsMailboxClient(poolSabs[2]);

  try {
    const [h0, h1, h2] = await Promise.all([
      client0.openAsync('lane0.bin'),
      client1.openAsync('lane1.bin'),
      client2.openAsync('lane2.bin'),
    ]);
    assert.ok(h0 >= 1 && h1 >= 1 && h2 >= 1);

    const out0 = new Uint8Array(1024);
    const out1 = new Uint8Array(1024);
    const out2 = new Uint8Array(1024);

    // Kick off all three lane reads concurrently without awaiting them individually.
    const p0 = client0.readAsync(h0, 512n, out0, 1024);
    const p1 = client1.readAsync(h1, 1024n, out1, 1024);
    const p2 = client2.readAsync(h2, 2048n, out2, 1024);

    // Wait until all 3 HTTP Range reads are simultaneously in-flight inside io_worker!
    await mock.waitForInFlight(3);
    assert.ok(
      mock.getPeakInFlight() >= 2,
      `Expected >= 2 concurrent in-flight HTTP reads across mailbox lanes, got ${mock.getPeakInFlight()}`,
    );
    assert.equal(mock.getPeakInFlight(), 3, 'All 3 lanes should execute HTTP reads concurrently');

    // Release gates in reverse order to confirm independent per-lane completion.
    gate2.resolve();
    gate1.resolve();
    gate0.resolve();

    const [n0, n1, n2] = await Promise.all([p0, p1, p2]);
    assert.equal(n0, 1024);
    assert.equal(n1, 1024);
    assert.equal(n2, 1024);
    assert.deepEqual(out0, data0.subarray(512, 512 + 1024));
    assert.deepEqual(out1, data1.subarray(1024, 1024 + 1024));
    assert.deepEqual(out2, data2.subarray(2048, 2048 + 1024));

    assert.deepEqual(
      await Promise.all([client0.closeAsync(h0), client1.closeAsync(h1), client2.closeAsync(h2)]),
      [true, true, true],
    );

    const stats = controller.getStats();
    assert.equal(stats.mailboxLanes, 4);
    assert.equal(stats.mailbox.slotCount, 4);
    assert.equal(stats.mailbox.perSlot[0].requests, 3);
    assert.equal(stats.mailbox.perSlot[1].requests, 3);
    assert.equal(stats.mailbox.perSlot[2].requests, 3);
    assert.equal(stats.mailbox.perSlot[3].requests, 0);
  } finally {
    await controller.detachMailbox();
    assert.equal(controller.mailboxLanes, 0);
    assert.equal(controller.getStats().mailboxLanes, 0);
  }
});

test('W260: single-SAB legacy attachMailbox(singleSab) and 1-element array stay 100% backward-compatible', async () => {
  const payload = makePatternBytes(2048, 0x33);
  const mock = createGatedRangeFetch(new Map([['/data/legacy.bin', payload]]));
  const controller = new IoWorkerController({ fetchImpl: mock.fetchImpl });
  await controller.init({
    dataBase: '/data/',
    manifest: { files: [{ path: 'legacy.bin', size: payload.byteLength }] },
    bootset: [],
  });

  const singleSab = createHttpfsMailboxBuffer(32 * 1024);
  const host = await controller.attachMailbox(singleSab, { probeOnOpen: false });
  assert.equal(host instanceof HttpfsMailboxPoolHost, false);
  assert.equal(controller.mailboxLanes, 1);
  assert.equal(controller.getStats().mailboxLanes, 1);

  const client = new HttpfsMailboxClient(singleSab);
  try {
    const h = await client.openAsync('legacy.bin');
    assert.ok(h >= 1);
    const out = new Uint8Array(512);
    const n = await client.readAsync(h, 256n, out, 512);
    assert.equal(n, 512);
    assert.deepEqual(out, payload.subarray(256, 256 + 512));
    assert.equal(await client.closeAsync(h), true);
  } finally {
    await controller.detachMailbox();
  }

  // 1-element array also cleanly degrades to a single lane.
  const oneLaneSab = createHttpfsMailboxBuffer(32 * 1024);
  const hostOne = await controller.attachMailbox([oneLaneSab], { probeOnOpen: false });
  assert.equal(hostOne instanceof HttpfsMailboxPoolHost, false);
  assert.equal(controller.mailboxLanes, 1);
  await controller.detachMailbox();
});

test('W260: io_worker classic Worker onmessage supports { attachMailbox, sabs }, { attachMailboxPool, sabs }, and pool dump writes', async () => {
  const dataA = makePatternBytes(2048, 0x20);
  const dataB = makePatternBytes(2048, 0x60);
  const gateA = createDeferred();
  const gateB = createDeferred();
  const dumpPosts = [];

  const baseMock = createGatedRangeFetch(
    new Map([
      ['/data/a.bin', dataA],
      ['/data/b.bin', dataB],
    ]),
    {
      gates: new Map([
        ['/data/a.bin', gateA],
        ['/data/b.bin', gateB],
      ]),
    },
  );

  async function fetchWithDump(url, init = {}) {
    const urlStr = String(url);
    if (urlStr.startsWith('/__pg6_dump/') && (init.method || '').toUpperCase() === 'POST') {
      const body = init.body instanceof Uint8Array ? init.body.slice() : new Uint8Array(init.body);
      dumpPosts.push({ url: urlStr, bytes: body });
      return {
        ok: true,
        status: 200,
        headers: { get: () => null },
        async json() {
          return { ok: true, bytesWritten: body.byteLength };
        },
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }
    return baseMock.fetchImpl(url, init);
  }

  const posted = [];
  const workerSelf = {
    onmessage: null,
    postMessage(msg) {
      posted.push(msg);
    },
  };
  const sandbox = {
    self: workerSelf,
    globalThis: null,
    fetch: fetchWithDump,
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
    Map,
    Set,
    WeakSet,
    Promise,
    Error,
    RangeError,
    TypeError,
    Number,
    Math,
    Date,
    Object,
    Array,
    JSON,
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
      allowDumpWrites: true,
      dumpEndpoint: '/__pg6_dump/',
      manifest: {
        files: [
          { path: 'a.bin', size: dataA.byteLength },
          { path: 'b.bin', size: dataB.byteLength },
        ],
      },
      bootset: [],
    },
  });

  const pool4 = createHttpfsMailboxPool(4, 32 * 1024);
  await workerSelf.onmessage({
    data: { attachMailbox: true, id: 1, sabs: pool4, sab: pool4[0], probeOnOpen: false },
  });
  const attachReply = posted.find((m) => m.id === 1);
  assert.deepEqual(Object.assign({}, attachReply), { mailboxAttached: true, mailboxLanes: 4, id: 1 });

  await workerSelf.onmessage({ data: { stats: true, id: 2 } });
  const statsReply = posted.find((m) => m.id === 2);
  assert.equal(statsReply.stats.mailboxLanes, 4);
  assert.equal(statsReply.stats.mailbox.slotCount, 4);

  const client0 = new HttpfsMailboxClient(pool4[0]);
  const client1 = new HttpfsMailboxClient(pool4[1]);
  try {
    const [hA, hB] = await Promise.all([
      client0.openAsync('a.bin'),
      client1.openAsync('b.bin'),
    ]);
    const outA = new Uint8Array(512);
    const outB = new Uint8Array(512);
    const rA = client0.readAsync(hA, 0n, outA, 512);
    const rB = client1.readAsync(hB, 256n, outB, 512);

    await baseMock.waitForInFlight(2);
    assert.ok(baseMock.getPeakInFlight() >= 2);
    gateA.resolve();
    gateB.resolve();

    assert.deepEqual(await Promise.all([rA, rB]), [512, 512]);
    assert.deepEqual(outA, dataA.subarray(0, 512));
    assert.deepEqual(outB, dataB.subarray(256, 256 + 512));
    await Promise.all([client0.closeAsync(hA), client1.closeAsync(hB)]);

    // Also verify IoWorkerMailboxHost custom /pg6_dump/ write handling works on non-zero pool lane (lane 2).
    const lane2Header = new DataView(pool4[2], 0, HTTPFS_HEADER_BYTES);
    const lane2Status = new Int32Array(pool4[2], 0, 1);
    const lane2Payload = new Uint8Array(pool4[2], HTTPFS_HEADER_BYTES);
    const dumpPathBytes = new TextEncoder().encode('/pg6_dump/pool_test.bin');
    lane2Payload.set(dumpPathBytes, 0);
    lane2Header.setUint32(HttpfsOffsets.OP, 1, true); // Open
    lane2Header.setInt32(HttpfsOffsets.HANDLE, -1, true);
    lane2Header.setBigUint64(HttpfsOffsets.OFFSET, 1n, true); // HTTPFS_OPEN_FLAG_WRITE
    lane2Header.setBigUint64(HttpfsOffsets.LENGTH, BigInt(dumpPathBytes.byteLength), true);
    Atomics.store(lane2Status, 0, HttpfsStatus.RequestReady);
    Atomics.notify(lane2Status, 0, 1);
    const openWait = Atomics.waitAsync(lane2Status, 0, HttpfsStatus.RequestReady, 3000);
    if (openWait.async) await openWait.value;
    assert.equal(Atomics.load(lane2Status, 0), HttpfsStatus.ResponseReady);
    const dumpHandle = Number(lane2Header.getBigInt64(HttpfsOffsets.RESULT, true));
    assert.ok(dumpHandle >= 1);
    Atomics.store(lane2Status, 0, HttpfsStatus.Idle);

    const writeBytes = new TextEncoder().encode('hello-pool-lane-2');
    lane2Payload.set(writeBytes, 0);
    lane2Header.setUint32(HttpfsOffsets.OP, 4, true); // HTTPFS_OP_WRITE
    lane2Header.setInt32(HttpfsOffsets.HANDLE, dumpHandle, true);
    lane2Header.setBigUint64(HttpfsOffsets.OFFSET, 0n, true);
    lane2Header.setBigUint64(HttpfsOffsets.LENGTH, BigInt(writeBytes.byteLength), true);
    Atomics.store(lane2Status, 0, HttpfsStatus.RequestReady);
    Atomics.notify(lane2Status, 0, 1);
    const writeWait = Atomics.waitAsync(lane2Status, 0, HttpfsStatus.RequestReady, 3000);
    if (writeWait.async) await writeWait.value;
    assert.equal(Atomics.load(lane2Status, 0), HttpfsStatus.ResponseReady);
    assert.equal(Number(lane2Header.getBigInt64(HttpfsOffsets.RESULT, true)), writeBytes.byteLength);
    Atomics.store(lane2Status, 0, HttpfsStatus.Idle);

    const client2 = new HttpfsMailboxClient(pool4[2]);
    assert.equal(await client2.closeAsync(dumpHandle), true);
    assert.equal(dumpPosts.length, 1);
    assert.equal(new TextDecoder().decode(dumpPosts[0].bytes), 'hello-pool-lane-2');

    // Explicit attachMailboxPool message replaces the active pool cleanly.
    const pool2 = createHttpfsMailboxPool(2, 32 * 1024);
    await workerSelf.onmessage({
      data: { attachMailboxPool: true, id: 3, sabs: pool2, probeOnOpen: false },
    });
    const poolReply = posted.find((m) => m.id === 3);
    assert.deepEqual(Object.assign({}, poolReply), { mailboxAttached: true, mailboxLanes: 2, id: 3 });

    await workerSelf.onmessage({ data: { stats: true, id: 4 } });
    const statsReply2 = posted.find((m) => m.id === 4);
    assert.equal(statsReply2.stats.mailboxLanes, 2);
  } finally {
    await workerSelf.onmessage({ data: { detachMailbox: true, id: 5 } });
  }
});

test('W260: loader.js allocateHttpfsMailboxPool defaults to 8 lanes, falls back to 1 lane on lanes=1 or missing pool API', () => {
  assert.equal(DEFAULT_HTTPFS_MAILBOX_LANES, 8);
  assert.equal(normalizeHttpfsMailboxLanes(undefined), 8);
  assert.equal(normalizeHttpfsMailboxLanes(1), 1);
  assert.equal(normalizeHttpfsMailboxLanes(4), 4);
  assert.equal(normalizeHttpfsMailboxLanes(0), 8);
  assert.equal(normalizeHttpfsMailboxLanes(-3), 8);

  const defaultPool = allocateHttpfsMailboxPool();
  assert.equal(defaultPool.lanes, 8);
  assert.equal(defaultPool.sabs.length, 8);
  assert.equal(defaultPool.sab, defaultPool.sabs[0]);
  assert.equal(new Set(defaultPool.sabs).size, 8, 'All 8 lanes must be distinct SharedArrayBuffers');

  const singlePool = allocateHttpfsMailboxPool(1);
  assert.equal(singlePool.lanes, 1);
  assert.equal(singlePool.sabs.length, 1);
  assert.equal(singlePool.sab, singlePool.sabs[0]);

  const legacyFallbackPool = allocateHttpfsMailboxPool(8, {
    createHttpfsMailboxBuffer: () => createHttpfsMailboxBuffer(1024),
  });
  assert.equal(legacyFallbackPool.lanes, 1);
  assert.equal(legacyFallbackPool.sabs.length, 1);
  assert.equal(legacyFallbackPool.sab, legacyFallbackPool.sabs[0]);
});

test('W260: loader.js round-robin lane assignment distributes 8+ pthreads evenly across all 8 lanes, preserves httpfsMailboxSab === pool[0], and degrades cleanly for lanes=1', () => {
  // 1. Direct coordinator / proxy round-robin unit verification (16 pthreads across 8 lanes).
  const pool8 = createHttpfsMailboxPool(8, 4096);
  const mockWorkers = [];
  class SimpleMockWorker {
    constructor(script, opts) {
      this.script = script;
      this.opts = opts;
      this.posted = [];
      mockWorkers.push(this);
    }
    postMessage(msg) {
      this.posted.push(msg);
    }
  }
  const { WrappedWorker, coordinator } = installPthreadWorkerProxy({
    WorkerCtor: SimpleMockWorker,
    getHttpfsMailboxSabs: () => pool8,
    redirectHref: 'http://127.0.0.1:8080/loader.js',
  });
  assert.equal(coordinator.getTelemetry().httpfsMailboxLanes, 8);

  for (let i = 0; i < 16; i += 1) {
    const w = new WrappedWorker('./dist/runtime.js', { name: `em-pthread-${i + 1}` });
    const expectedLane = i % 8;
    assert.equal(w.posted[0].__playgta6_httpfs_sab, pool8[expectedLane]);
    assert.equal(w.posted[0].__playgta6_httpfs_lane_sab, pool8[expectedLane]);
    assert.equal(w.posted[0].__playgta6_httpfs_lane, expectedLane);
    assert.deepEqual(coordinator.getWorkerHttpfsLane(w), {
      laneIndex: expectedLane,
      sab: pool8[expectedLane],
    });
  }

  // 2. Full loader.js coordinator VM: default 8-lane allocation + ioWorker attach + 12 pthreads + BroadcastChannel round-robin.
  const vm8 = bootCoordinatorVm();
  vm8.workerSelf.onmessage({
    data: { canvas: { id: 'c8' }, width: 1280, height: 720, dataBase: '/data/' },
  });

  const ioWorker8 = vm8.spawnedWorkers.find((w) => w.script === './workers/io_worker.js');
  assert.ok(ioWorker8);
  const attachMsg8 = ioWorker8.posted.map((p) => p.msg).find((m) => m && m.attachMailbox === true);
  assert.ok(attachMsg8, 'Coordinator must post attachMailbox message to io_worker');
  assert.equal(Array.isArray(attachMsg8.sabs), true);
  assert.equal(attachMsg8.sabs.length, 8);
  assert.equal(attachMsg8.sab, attachMsg8.sabs[0], 'httpfsMailboxSab must point to lane 0');
  assert.equal(new Set(attachMsg8.sabs).size, 8);
  assert.equal(vm8.context.PlayGta6Loader.getLoaderStatus().httpfsMailboxLanes, 8);

  const laneCounts = new Array(8).fill(0);
  for (let i = 0; i < 12; i += 1) {
    const pw = new vm8.workerSelf.Worker('./dist/runtime.js', { name: `em-pthread-${i + 1}` });
    const bootMsg = pw.posted[0].msg;
    const expectedLane = i % 8;
    // Legacy __playgta6_httpfs_sab stays lane 0 for backward compatibility, while dedicated
    // round-robin lane arrives in __playgta6_httpfs_lane_sab and is picked by bootstrapEmscriptenPthreadWorker.
    assert.equal(bootMsg.__playgta6_httpfs_sab, attachMsg8.sabs[0]);
    assert.equal(bootMsg.__playgta6_httpfs_lane_sab, attachMsg8.sabs[expectedLane]);
    assert.equal(bootMsg.__playgta6_httpfs_lane, expectedLane);
    laneCounts[expectedLane] += 1;

    // Verify bootstrapEmscriptenPthreadWorker attaches the assigned contention-free lane SAB.
    const listeners = [];
    const pScope = {
      name: `em-pthread-${i + 1}`,
      location: { href: 'http://127.0.0.1:8080/loader.js' },
      WebAssembly: { Instance: WebAssembly.Instance },
      addEventListener(type, fn) {
        if (type === 'message') listeners.push(fn);
      },
      removeEventListener(type, fn) {
        const idx = listeners.indexOf(fn);
        if (idx >= 0) listeners.splice(idx, 1);
      },
    };
    const handle = bootstrapEmscriptenPthreadWorker({
      scope: pScope,
      importScriptsFn: () => {},
      BroadcastChannelCtor: null,
      httpfsMailboxApi: { HttpfsMailboxClient },
    });
    for (const fn of listeners.slice()) {
      fn({ data: bootMsg, stopImmediatePropagation() {} });
    }
    assert.equal(handle.hasHttpfsSab(), true);
    assert.equal(handle.getHttpfsSab(), attachMsg8.sabs[expectedLane]);
    handle.cleanup();
  }
  assert.deepEqual(laneCounts, [2, 2, 2, 2, 1, 1, 1, 1]);

  // Verify BroadcastChannel('playgta6-httpfs-sab') delivers { type: 'httpfs-sab', sab: assignedLaneSab, sabs: poolSabs } round-robin.
  const bcReplies = [];
  const reqChannel = new vm8.bus.BusBroadcastChannel(PTHREAD_HTTPFS_SAB_CHANNEL_NAME);
  reqChannel.onmessage = (ev) => {
    if (ev && ev.data && ev.data.type === 'httpfs-sab') {
      bcReplies.push(ev.data);
    }
  };
  for (let i = 0; i < 9; i += 1) {
    reqChannel.postMessage({ type: 'request-httpfs-sab' });
  }
  reqChannel.close();
  assert.equal(bcReplies.length, 9);
  for (let i = 0; i < 9; i += 1) {
    assert.equal(bcReplies[i].sab, attachMsg8.sabs[i % 8]);
    assert.deepEqual(bcReplies[i].sabs, attachMsg8.sabs);
  }

  // 3. Coordinator VM fallback when httpfsMailboxLanes === 1.
  const vm1 = bootCoordinatorVm();
  vm1.workerSelf.onmessage({
    data: {
      canvas: { id: 'c1' },
      width: 1280,
      height: 720,
      dataBase: '/data/',
      httpfsMailboxLanes: 1,
    },
  });
  const ioWorker1 = vm1.spawnedWorkers.find((w) => w.script === './workers/io_worker.js');
  const attachMsg1 = ioWorker1.posted.map((p) => p.msg).find((m) => m && m.attachMailbox === true);
  assert.equal(attachMsg1.sabs.length, 1);
  assert.equal(attachMsg1.sab, attachMsg1.sabs[0]);
  assert.equal(vm1.context.PlayGta6Loader.getLoaderStatus().httpfsMailboxLanes, 1);

  for (let i = 0; i < 4; i += 1) {
    const pw = new vm1.workerSelf.Worker('./dist/runtime.js', { name: `em-pthread-${i + 1}` });
    assert.equal(pw.posted[0].msg.__playgta6_httpfs_sab, attachMsg1.sabs[0]);
    assert.equal(pw.posted[0].msg.__playgta6_httpfs_lane_sab, attachMsg1.sabs[0]);
    assert.equal(pw.posted[0].msg.__playgta6_httpfs_lane, 0);
  }

  // 4. Coordinator VM fallback when createHttpfsMailboxPool is absent.
  const vmLegacyApi = bootCoordinatorVm({ hideCreateHttpfsMailboxPool: true });
  vmLegacyApi.workerSelf.onmessage({
    data: { canvas: { id: 'c-legacy' }, width: 1280, height: 720, dataBase: '/data/' },
  });
  const ioWorkerLegacy = vmLegacyApi.spawnedWorkers.find(
    (w) => w.script === './workers/io_worker.js',
  );
  const attachMsgLegacy = ioWorkerLegacy.posted
    .map((p) => p.msg)
    .find((m) => m && m.attachMailbox === true);
  assert.equal(attachMsgLegacy.sabs.length, 1);
  assert.equal(attachMsgLegacy.sab, attachMsgLegacy.sabs[0]);
  assert.equal(vmLegacyApi.context.PlayGta6Loader.getLoaderStatus().httpfsMailboxLanes, 1);
});

test('W260 end-to-end: loader.js coordinator pool -> io_worker.js HttpfsMailboxPoolHost -> 3 bootstrapped pthreads reading concurrently', async () => {
  const vmCoord = bootCoordinatorVm();
  vmCoord.workerSelf.onmessage({
    data: { canvas: { id: 'e2e' }, width: 1280, height: 720, dataBase: '/data/' },
  });
  const ioMockWorker = vmCoord.spawnedWorkers.find((w) => w.script === './workers/io_worker.js');
  const attachMsg = ioMockWorker.posted.map((p) => p.msg).find((m) => m && m.attachMailbox === true);
  assert.equal(attachMsg.sabs.length, 8);

  const file0 = makePatternBytes(2048, 0xa0);
  const file1 = makePatternBytes(2048, 0xb0);
  const file2 = makePatternBytes(2048, 0xc0);
  const gate0 = createDeferred();
  const gate1 = createDeferred();
  const gate2 = createDeferred();

  const mock = createGatedRangeFetch(
    new Map([
      ['/data/ds0.rpf', file0],
      ['/data/ds1.rpf', file1],
      ['/data/ds2.rpf', file2],
    ]),
    {
      gates: new Map([
        ['/data/ds0.rpf', gate0],
        ['/data/ds1.rpf', gate1],
        ['/data/ds2.rpf', gate2],
      ]),
    },
  );

  const controller = new IoWorkerController({ fetchImpl: mock.fetchImpl, maxConcurrency: 8 });
  await controller.init({
    dataBase: '/data/',
    manifest: {
      files: [
        { path: 'ds0.rpf', size: file0.byteLength },
        { path: 'ds1.rpf', size: file1.byteLength },
        { path: 'ds2.rpf', size: file2.byteLength },
      ],
    },
    bootset: [],
  });
  // Feed the exact attachMailbox message produced by loader.js into IoWorkerController!
  await controller.handleMessage(Object.assign({}, attachMsg, { probeOnOpen: false }));
  assert.equal(controller.mailboxLanes, 8);
  assert.ok(controller.mailboxHost instanceof HttpfsMailboxPoolHost);

  // Spawn 3 pthreads via the coordinator and bootstrap each with its assigned lane SAB.
  const clients = [];
  const handles = [];
  for (let i = 0; i < 3; i += 1) {
    const pw = new vmCoord.workerSelf.Worker('./dist/runtime.js', { name: `em-pthread-${i + 1}` });
    const listeners = [];
    const pScope = {
      name: `em-pthread-${i + 1}`,
      location: { href: 'http://127.0.0.1:8080/loader.js' },
      WebAssembly: { Instance: WebAssembly.Instance },
      addEventListener(type, fn) {
        if (type === 'message') listeners.push(fn);
      },
      removeEventListener(type, fn) {
        const idx = listeners.indexOf(fn);
        if (idx >= 0) listeners.splice(idx, 1);
      },
    };
    const handle = bootstrapEmscriptenPthreadWorker({
      scope: pScope,
      importScriptsFn: () => {},
      BroadcastChannelCtor: null,
      httpfsMailboxApi: { HttpfsMailboxClient },
    });
    for (const fn of listeners.slice()) {
      fn({ data: pw.posted[0].msg, stopImmediatePropagation() {} });
    }
    const assignedSab = handle.getHttpfsSab();
    assert.equal(assignedSab, attachMsg.sabs[i]);
    clients.push(new HttpfsMailboxClient(assignedSab));
    handles.push(handle);
  }

  try {
    const [h0, h1, h2] = await Promise.all([
      clients[0].openAsync('ds0.rpf'),
      clients[1].openAsync('ds1.rpf'),
      clients[2].openAsync('ds2.rpf'),
    ]);
    const out0 = new Uint8Array(512);
    const out1 = new Uint8Array(512);
    const out2 = new Uint8Array(512);

    const p0 = clients[0].readAsync(h0, 0n, out0, 512);
    const p1 = clients[1].readAsync(h1, 0n, out1, 512);
    const p2 = clients[2].readAsync(h2, 0n, out2, 512);

    await mock.waitForInFlight(3);
    assert.equal(mock.getPeakInFlight(), 3);

    gate0.resolve();
    gate1.resolve();
    gate2.resolve();

    assert.deepEqual(await Promise.all([p0, p1, p2]), [512, 512, 512]);
    assert.deepEqual(out0, file0.subarray(0, 512));
    assert.deepEqual(out1, file1.subarray(0, 512));
    assert.deepEqual(out2, file2.subarray(0, 512));

    await Promise.all([
      clients[0].closeAsync(h0),
      clients[1].closeAsync(h1),
      clients[2].closeAsync(h2),
    ]);
  } finally {
    for (const h of handles) {
      h.cleanup();
    }
    await controller.detachMailbox();
  }
});
