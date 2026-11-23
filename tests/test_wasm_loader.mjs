import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { Worker as NodeWorker } from 'node:worker_threads';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');

const mailboxMod = require('../web/workers/httpfs_mailbox.js');
const rangeCacheMod = require('../web/workers/range_cache.js');
const loaderMod = require('../web/loader.js');

const {
  HTTPFS_MAILBOX_VERSION,
  HTTPFS_HEADER_BYTES,
  HTTPFS_PAYLOAD_BYTES,
  HTTPFS_MAILBOX_BYTES,
  HttpfsStatus,
  HttpfsOp,
  HttpfsOffsets,
  createHttpfsMailboxBuffer,
  initHttpfsMailbox,
  HttpfsMailboxHost,
  HttpfsMailboxClient,
} = mailboxMod;

const { RangeCache } = rangeCacheMod;
const { createHostImports, encodeTestFramePacket, ptrToNumber } = loaderMod;

function makePatternBytes(length, seed = 29) {
  const out = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = (i * 37 + seed) & 0xff;
  }
  return out;
}

function createMockRangeFetch(filesMap) {
  return async function mockFetch(url, init = {}) {
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
        async json() {
          throw new Error('404');
        },
      };
    }

    const rangeHeader = (init.headers && (init.headers.Range || init.headers.range)) || null;
    if (!rangeHeader) {
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
  };
}

// ---------------------------------------------------------------------------
// 1. Exact SAB binary layout verification against runtime/os/httpfs_mailbox.h
// ---------------------------------------------------------------------------

test('HttpfsMailbox UMD export and exact header/payload layout match runtime/os/httpfs_mailbox.h', () => {
  assert.equal(globalThis.PlayGta6HttpfsMailbox.HttpfsMailboxHost, HttpfsMailboxHost);
  assert.equal(globalThis.PlayGta6HttpfsMailbox.HttpfsMailboxClient, HttpfsMailboxClient);

  // Verify constants match runtime/os/httpfs_mailbox.h exactly.
  assert.equal(HTTPFS_MAILBOX_VERSION, 1);
  assert.equal(HTTPFS_HEADER_BYTES, 40);
  assert.equal(HTTPFS_PAYLOAD_BYTES, 64 * 1024);
  assert.equal(HTTPFS_MAILBOX_BYTES, 40 + 64 * 1024);

  assert.deepEqual(
    {
      STATUS: HttpfsOffsets.STATUS,
      OP: HttpfsOffsets.OP,
      HANDLE: HttpfsOffsets.HANDLE,
      PAYLOAD_CAPACITY: HttpfsOffsets.PAYLOAD_CAPACITY,
      OFFSET: HttpfsOffsets.OFFSET,
      LENGTH: HttpfsOffsets.LENGTH,
      RESULT: HttpfsOffsets.RESULT,
      PAYLOAD: HttpfsOffsets.PAYLOAD,
    },
    {
      STATUS: 0,
      OP: 4,
      HANDLE: 8,
      PAYLOAD_CAPACITY: 12,
      OFFSET: 16,
      LENGTH: 24,
      RESULT: 32,
      PAYLOAD: 40,
    },
  );

  // Verify enum values match C++ header.
  assert.equal(HttpfsStatus.Idle, 0);
  assert.equal(HttpfsStatus.RequestReady, 1);
  assert.equal(HttpfsStatus.ResponseReady, 2);
  assert.equal(HttpfsStatus.Error, -1);
  assert.equal(HttpfsStatus.Shutdown, -2);
  assert.equal(HttpfsStatus.IDLE, 0);
  assert.equal(HttpfsStatus.PENDING, 1);
  assert.equal(HttpfsStatus.DONE, 2);
  assert.equal(HttpfsStatus.ERROR, -1);
  assert.equal(HttpfsStatus.SHUTDOWN, -2);

  assert.equal(HttpfsOp.None, 0);
  assert.equal(HttpfsOp.Open, 1);
  assert.equal(HttpfsOp.Read, 2);
  assert.equal(HttpfsOp.Close, 3);

  // Cross-check against static_assert lines in runtime/os/httpfs_mailbox.h.
  const cppHeader = fs.readFileSync(
    path.join(repoRoot, 'runtime/os/httpfs_mailbox.h'),
    'utf8',
  );
  assert.match(cppHeader, /kHttpfsHeaderBytes\s*=\s*40/);
  assert.match(cppHeader, /kHttpfsPayloadBytes\s*=\s*64\s*\*\s*1024/);
  assert.match(cppHeader, /offsetof\(HttpfsMailbox,\s*status\)\s*==\s*0/);
  assert.match(cppHeader, /offsetof\(HttpfsMailbox,\s*op\)\s*==\s*4/);
  assert.match(cppHeader, /offsetof\(HttpfsMailbox,\s*handle\)\s*==\s*8/);
  assert.match(cppHeader, /offsetof\(HttpfsMailbox,\s*payload_capacity\)\s*==\s*12/);
  assert.match(cppHeader, /offsetof\(HttpfsMailbox,\s*offset\)\s*==\s*16/);
  assert.match(cppHeader, /offsetof\(HttpfsMailbox,\s*length\)\s*==\s*24/);
  assert.match(cppHeader, /offsetof\(HttpfsMailbox,\s*result\)\s*==\s*32/);
  assert.match(cppHeader, /offsetof\(HttpfsMailbox,\s*payload\)\s*==\s*kHttpfsHeaderBytes/);

  // Inspect initialized SAB bytes directly.
  const sab = createHttpfsMailboxBuffer();
  assert.ok(sab instanceof SharedArrayBuffer);
  assert.equal(sab.byteLength, HTTPFS_MAILBOX_BYTES);

  const dv = new DataView(sab);
  assert.equal(dv.getInt32(0, true), HttpfsStatus.Idle);
  assert.equal(dv.getUint32(4, true), HttpfsOp.None);
  assert.equal(dv.getInt32(8, true), -1);
  assert.equal(dv.getUint32(12, true), HTTPFS_PAYLOAD_BYTES);
  assert.equal(dv.getBigUint64(16, true), 0n);
  assert.equal(dv.getBigUint64(24, true), 0n);
  assert.equal(dv.getBigInt64(32, true), 0n);
});

// ---------------------------------------------------------------------------
// 2. HttpfsMailboxClient <-> HttpfsMailboxHost end-to-end over SharedArrayBuffer
// ---------------------------------------------------------------------------

test('HttpfsMailboxClient <-> HttpfsMailboxHost Open/Read/Close with multi-chunk, EOF clamp, and error handling', async () => {
  const archive = makePatternBytes(300, 13);
  const files = new Map([['update/x64/dlc.rpf', archive]]);
  const rangeCache = new RangeCache({
    dataBase: '/data/',
    chunkSize: 64,
    maxBytes: 4096,
    fetchImpl: createMockRangeFetch(files),
  });

  // Use a 64-byte payload capacity to exercise multi-chunk SAB transfers (300B => 5 chunks).
  const sab = createHttpfsMailboxBuffer(64);
  const host = new HttpfsMailboxHost(sab, { rangeCache });
  const client = new HttpfsMailboxClient(sab);
  assert.equal(host.payloadCapacity, 64);
  assert.equal(client.payloadCapacity, 64);

  const hostLoop = host.start();

  try {
    // Missing path and traversal attempts must fail cleanly.
    assert.equal(await client.openAsync('update/missing.rpf'), -1);
    assert.equal(await client.openAsync('../etc/passwd'), -1);
    assert.equal(await client.openAsync(''), -1);

    // Valid open returns handle >= 0.
    const handle = await client.openAsync('update/x64/dlc.rpf');
    assert.ok(handle >= 0, `Expected valid handle, got ${handle}`);

    // Full multi-chunk read across 5 x 64-byte mailbox roundtrips.
    const fullOut = new Uint8Array(300);
    const fullRead = await client.readAsync(handle, 0n, fullOut, 300);
    assert.equal(fullRead, 300);
    assert.deepEqual(fullOut, archive);

    // Multi-offset slice read.
    const midOut = new Uint8Array(90);
    const midRead = await client.readAsync(handle, 75, midOut, 90);
    assert.equal(midRead, 90);
    assert.deepEqual(midOut, archive.subarray(75, 165));

    // EOF clamping (offset 260, request 100 -> returns 40 bytes).
    const tailOut = new Uint8Array(100);
    const tailRead = await client.readAsync(handle, 260n, tailOut, 100);
    assert.equal(tailRead, 40);
    assert.deepEqual(tailOut.subarray(0, 40), archive.subarray(260, 300));

    // Reading strictly past EOF returns 0 bytes.
    const pastEof = new Uint8Array(16);
    assert.equal(await client.readAsync(handle, 999, pastEof, 16), 0);

    // Invalid handle read/close return errors.
    assert.ok((await client.readAsync(9999, 0, pastEof, 16)) < 0);
    assert.equal(await client.closeAsync(9999), false);

    // Valid close succeeds; subsequent read/close on closed handle fail.
    assert.equal(await client.closeAsync(handle), true);
    assert.equal(await client.closeAsync(handle), false);
    assert.ok((await client.readAsync(handle, 0, pastEof, 16)) < 0);
  } finally {
    await host.stop();
    await hostLoop;
  }
});

test('HttpfsMailboxHost fallback polling mode (without Atomics.waitAsync) works cleanly', async () => {
  const payload = new TextEncoder().encode('fallback-poll-httpfs-payload');
  const sab = createHttpfsMailboxBuffer(128);
  const handles = new Map();
  let nextHandle = 1;

  const host = new HttpfsMailboxHost(sab, {
    forcePoll: true,
    fallbackPollMs: 1,
    open(relPath) {
      if (relPath !== 'boot.txt') return -2;
      const h = nextHandle++;
      handles.set(h, payload);
      return h;
    },
    read(handle, offset, len) {
      const data = handles.get(handle);
      if (!data) return null;
      return data.subarray(offset, offset + len);
    },
    close(handle) {
      return handles.delete(handle);
    },
  });
  const client = new HttpfsMailboxClient(sab);
  const loop = host.start();

  try {
    const h = await client.openAsync('boot.txt');
    assert.ok(h >= 0);
    const out = new Uint8Array(payload.byteLength);
    const n = await client.readAsync(h, 0, out);
    assert.equal(n, payload.byteLength);
    assert.deepEqual(out, payload);
    assert.equal(await client.closeAsync(h), true);
  } finally {
    await host.stop();
    await loop;
  }
});

test('Cross-thread synchronous Atomics.wait HttpfsMailboxClient <-> HttpfsMailboxHost over SharedArrayBuffer', async () => {
  const blob = makePatternBytes(200, 77);
  const rangeCache = new RangeCache({
    dataBase: '/data/',
    chunkSize: 64,
    maxBytes: 1024,
    fetchImpl: createMockRangeFetch(new Map([['cross_thread.rpf', blob]])),
  });

  const sab = createHttpfsMailboxBuffer(64);
  const host = new HttpfsMailboxHost(sab, { rangeCache });
  const hostLoop = host.start();

  const mailboxPath = path.join(repoRoot, 'web/workers/httpfs_mailbox.js');
  const workerCode = `
    const { workerData, parentPort } = require('node:worker_threads');
    const { HttpfsMailboxClient } = require(workerData.mailboxPath);
    const client = new HttpfsMailboxClient(workerData.sab, { timeoutMs: 5000 });

    const missing = client.open('nope.rpf');
    const h = client.open('cross_thread.rpf');
    const buf = new Uint8Array(200);
    const n = client.read(h, 0n, buf, 200);
    const tail = new Uint8Array(50);
    const tailN = client.read(h, 180n, tail, 50);
    const closed = client.close(h);
    const afterClose = client.read(h, 0, tail, 10);

    parentPort.postMessage({
      missing,
      h,
      n,
      bytes: Array.from(buf),
      tailN,
      tailBytes: Array.from(tail.subarray(0, tailN)),
      closed,
      afterClose,
    });
  `;

  try {
    const result = await new Promise((resolve, reject) => {
      const w = new NodeWorker(workerCode, {
        eval: true,
        workerData: { sab, mailboxPath },
      });
      w.once('message', resolve);
      w.once('error', reject);
      w.once('exit', (code) => {
        if (code !== 0) reject(new Error(`Worker exited with code ${code}`));
      });
    });

    assert.equal(result.missing, -1);
    assert.ok(result.h >= 0);
    assert.equal(result.n, 200);
    assert.deepEqual(new Uint8Array(result.bytes), blob);
    assert.equal(result.tailN, 20);
    assert.deepEqual(new Uint8Array(result.tailBytes), blob.subarray(180, 200));
    assert.equal(result.closed, true);
    assert.ok(result.afterClose < 0);
  } finally {
    await host.stop();
    await hostLoop;
  }
});

// ---------------------------------------------------------------------------
// 3. createHostImports with synthetic WebAssembly.Memory (Wasm64 BigInt + Wasm32 Number)
// ---------------------------------------------------------------------------

test('createHostImports handles Wasm64 BigInt and Wasm32 Number pointers over WebAssembly.Memory', () => {
  // Test with both standard and shared (pthread) WebAssembly.Memory.
  const memories = [
    new WebAssembly.Memory({ initial: 1, maximum: 1 }),
    new WebAssembly.Memory({ initial: 1, maximum: 1, shared: true }),
  ];

  for (const memory of memories) {
    const logs = [];
    const submittedPackets = [];

    // Synchronous in-process HTTPFS mailbox server responder.
    const sab = createHttpfsMailboxBuffer(64);
    const fileData = makePatternBytes(150, 91);
    const openHandles = new Map();
    let nextHandle = 1;

    const httpfsClient = new HttpfsMailboxClient(sab, (sharedBuf) => {
      const statusView = new Int32Array(sharedBuf, 0, 1);
      const dv = new DataView(sharedBuf, 0, HTTPFS_HEADER_BYTES);
      const payload = new Uint8Array(sharedBuf, HTTPFS_HEADER_BYTES, 64);
      const op = dv.getUint32(HttpfsOffsets.OP, true);
      const h = dv.getInt32(HttpfsOffsets.HANDLE, true);
      const off = Number(dv.getBigUint64(HttpfsOffsets.OFFSET, true));
      const len = Number(dv.getBigUint64(HttpfsOffsets.LENGTH, true));

      if (op === HttpfsOp.Open) {
        const copy = new Uint8Array(len);
        copy.set(payload.subarray(0, len));
        const relPath = new TextDecoder().decode(copy);
        if (relPath === 'assets/level0.rpf') {
          const handle = nextHandle++;
          openHandles.set(handle, fileData);
          dv.setBigInt64(HttpfsOffsets.RESULT, BigInt(handle), true);
          Atomics.store(statusView, 0, HttpfsStatus.ResponseReady);
        } else {
          dv.setBigInt64(HttpfsOffsets.RESULT, -2n, true);
          Atomics.store(statusView, 0, HttpfsStatus.Error);
        }
      } else if (op === HttpfsOp.Read) {
        const data = openHandles.get(h);
        if (!data) {
          dv.setBigInt64(HttpfsOffsets.RESULT, -9n, true);
          Atomics.store(statusView, 0, HttpfsStatus.Error);
        } else if (off >= data.byteLength) {
          dv.setBigInt64(HttpfsOffsets.RESULT, 0n, true);
          Atomics.store(statusView, 0, HttpfsStatus.ResponseReady);
        } else {
          const toCopy = Math.min(len, 64, data.byteLength - off);
          payload.set(data.subarray(off, off + toCopy), 0);
          dv.setBigInt64(HttpfsOffsets.RESULT, BigInt(toCopy), true);
          Atomics.store(statusView, 0, HttpfsStatus.ResponseReady);
        }
      } else if (op === HttpfsOp.Close) {
        const ok = openHandles.delete(h);
        dv.setBigInt64(HttpfsOffsets.RESULT, ok ? 0n : -9n, true);
        Atomics.store(statusView, 0, ok ? HttpfsStatus.ResponseReady : HttpfsStatus.Error);
      }
      Atomics.notify(statusView, 0, 1);
    });

    const imports = createHostImports({
      memory,
      log: (msg) => logs.push(msg),
      wgpuSubmit: (bytes) => submittedPackets.push(bytes),
      httpfsClient,
    });

    const memU8 = new Uint8Array(memory.buffer);
    const enc = new TextEncoder();

    // 1. host_log with Wasm64 BigInt ptr/len (including multi-byte UTF-8) and Wasm32 Number ptr/len.
    const msg1 = enc.encode('playgta6web Wasm64 ready — ABI v1');
    memU8.set(msg1, 256);
    imports.env.host_log(256n, BigInt(msg1.byteLength));
    assert.equal(logs[0], 'playgta6web Wasm64 ready — ABI v1');

    const msg2 = enc.encode('wasm32-compat-log');
    memU8.set(msg2, 512);
    imports.env.host_log(512, msg2.byteLength);
    assert.equal(logs[1], 'wasm32-compat-log');

    // 2. host_wgpu_submit copies exact packet bytes out of linear memory.
    const pkt = new Uint8Array(encodeTestFramePacket(0.08, 0.12, 0.22, 1.0));
    memU8.set(pkt, 1024);
    imports.env.host_wgpu_submit(1024n, BigInt(pkt.byteLength));
    assert.equal(submittedPackets.length, 1);
    assert.deepEqual(submittedPackets[0], pkt);
    // Mutating linear memory afterwards must not alter submitted packet copy.
    memU8.fill(0xee, 1024, 1024 + pkt.byteLength);
    assert.deepEqual(submittedPackets[0], pkt);

    // 3. host_httpfs_open / host_httpfs_read / host_httpfs_close using BigInt pointers & offsets.
    const pathBytes = enc.encode('assets/level0.rpf');
    memU8.set(pathBytes, 2048);
    const handle = imports.env.host_httpfs_open(2048n, BigInt(pathBytes.byteLength));
    assert.ok(handle >= 0);

    // Missing file returns -1.
    const badPath = enc.encode('assets/missing.rpf');
    memU8.set(badPath, 2200);
    assert.equal(imports.env.host_httpfs_open(2200n, BigInt(badPath.byteLength)), -1);

    // Read 150 bytes straight into Wasm linear memory at offset 4096 (multi-chunk across 64B mailbox!).
    const n1 = imports.env.host_httpfs_read(handle, 0n, 4096n, 150n);
    assert.equal(n1, 150n);
    assert.deepEqual(memU8.slice(4096, 4096 + 150), fileData);

    // Offset read with EOF clamp (offset 120, request 60 -> 30 bytes) using Number args.
    const n2 = imports.env.host_httpfs_read(handle, 120, 8192, 60);
    assert.equal(n2, 30n);
    assert.deepEqual(memU8.slice(8192, 8192 + 30), fileData.subarray(120, 150));

    // Out-of-bounds Wasm destination pointer returns -1n safely.
    assert.equal(
      imports.env.host_httpfs_read(handle, 0n, BigInt(memory.buffer.byteLength - 4), 16n),
      -1n,
    );

    // Close handle; subsequent read fails with negative BigInt.
    imports.env.host_httpfs_close(handle);
    assert.ok(imports.env.host_httpfs_read(handle, 0n, 4096n, 16n) < 0n);
  }
});

test('ptrToNumber rejects negative, non-integer, and out-of-range pointer values', () => {
  assert.equal(ptrToNumber(0n), 0);
  assert.equal(ptrToNumber(4096n), 4096);
  assert.equal(ptrToNumber(128), 128);
  assert.throws(() => ptrToNumber(-1n), RangeError);
  assert.throws(() => ptrToNumber(-5), RangeError);
  assert.throws(() => ptrToNumber(1.5), RangeError);
  assert.throws(() => ptrToNumber(BigInt(Number.MAX_SAFE_INTEGER) + 1n), RangeError);
  assert.throws(() => ptrToNumber('123'), TypeError);
});

// ---------------------------------------------------------------------------
// 4. io_worker.js attachMailbox & loader.js SAB wiring integration tests
// ---------------------------------------------------------------------------

test('io_worker.js attaches SAB mailbox via { attachMailbox: true, sab } and serves requests', async () => {
  const content = makePatternBytes(96, 41);
  const manifestJson = new TextEncoder().encode(
    JSON.stringify({ abi_version: 1, file_count: 1, files: [{ path: 'pack.rpf', size: 96 }] }),
  );
  const fetchImpl = createMockRangeFetch(
    new Map([
      ['pack.rpf', content],
      ['data-manifest.json', manifestJson],
    ]),
  );

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
    fetch: fetchImpl,
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

  await workerSelf.onmessage({ data: { init: true, dataBase: '/data/', chunkSize: 32, maxBytes: 512 } });
  assert.equal(posted[0].ready, true);

  const sab = createHttpfsMailboxBuffer(32);
  await workerSelf.onmessage({ data: { attachMailbox: true, id: 42, sab } });
  assert.equal(posted[1].id, 42);
  assert.equal(posted[1].mailboxAttached, true);

  const client = new HttpfsMailboxClient(sab);
  try {
    assert.equal(await client.openAsync('missing.rpf'), -1);
    const h = await client.openAsync('pack.rpf');
    assert.ok(h >= 0);

    const out = new Uint8Array(96);
    const got = await client.readAsync(h, 0n, out, 96);
    assert.equal(got, 96);
    assert.deepEqual(out, content);

    assert.equal(await client.closeAsync(h), true);
  } finally {
    await workerSelf.onmessage({ data: { detachMailbox: true, id: 43 } });
    assert.equal(posted[2].id, 43);
    assert.equal(posted[2].mailboxDetached, true);
  }
});

test('loader.js allocates HTTPFS SAB, attaches it to io_worker, and wires hostImports into createRuntimeModule', async () => {
  let receivedOpts = null;
  const workersByScript = new Map();

  class MockBroadcastChannel {
    postMessage() {}
  }

  class MockWorker {
    constructor(script) {
      this.script = script;
      this.onmessage = null;
      this.posted = [];
      workersByScript.set(script, this);
    }
    postMessage(msg) {
      this.posted.push(msg);
      if (msg.init) {
        queueMicrotask(() => {
          if (this.onmessage) {
            this.onmessage({ data: { ready: true, summary: 'ok', adapter: 'mock' } });
          }
        });
      }
    }
  }

  const workerSelf = { onmessage: null };
  const sandbox = {
    self: workerSelf,
    globalThis: null,
    BroadcastChannel: MockBroadcastChannel,
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
    Number,
    Math,
    setTimeout,
    clearTimeout,
    fetch: async () => ({ ok: true, status: 200 }),
    importScripts(scriptPath) {
      if (scriptPath === './dist/runtime.js') {
        workerSelf.createRuntimeModule = async (opts) => {
          receivedOpts = opts;
          return {};
        };
        return;
      }
      const resolved = path.resolve(repoRoot, 'web', scriptPath);
      vm.runInContext(fs.readFileSync(resolved, 'utf8'), context, { filename: resolved });
    },
  };
  sandbox.globalThis = sandbox;
  const context = vm.createContext(sandbox);

  const loaderPath = path.join(repoRoot, 'web/loader.js');
  vm.runInContext(fs.readFileSync(loaderPath, 'utf8'), context, { filename: loaderPath });

  workerSelf.onmessage({
    data: { canvas: { id: 'c' }, width: 800, height: 600, dataBase: '/data/' },
  });
  await new Promise((r) => setTimeout(r, 25));

  const ioWorker = workersByScript.get('./workers/io_worker.js');
  assert.ok(ioWorker, 'Expected io_worker to be created');
  const attachMsg = ioWorker.posted.find((m) => m.attachMailbox === true);
  assert.ok(attachMsg, 'Expected loader.js to send { attachMailbox: true, sab } to io_worker');
  assert.ok(attachMsg.sab instanceof SharedArrayBuffer);
  assert.equal(attachMsg.sab.byteLength, HTTPFS_MAILBOX_BYTES);

  assert.ok(receivedOpts, 'Expected createRuntimeModule to be invoked');
  assert.equal(receivedOpts.httpfsMailboxSab, attachMsg.sab);
  assert.equal(typeof receivedOpts.env.host_log, 'function');
  assert.equal(typeof receivedOpts.env.host_wgpu_submit, 'function');
  assert.equal(typeof receivedOpts.env.host_httpfs_open, 'function');
  assert.equal(typeof receivedOpts.env.host_httpfs_read, 'function');
  assert.equal(typeof receivedOpts.env.host_httpfs_close, 'function');
  assert.equal(typeof receivedOpts.hostImports.setMemory, 'function');
});
