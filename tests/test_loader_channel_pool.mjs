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

const {
  PROGRESS_CHANNEL_NAME,
  PTHREAD_WGPU_CHANNEL_NAME,
  PTHREAD_HTTPFS_SAB_CHANNEL_NAME,
  PTHREAD_TX_MSG_WGPU_SUBMIT,
  PTHREAD_TX_MSG_PROGRESS,
  PTHREAD_TX_MSG_RECYCLE_CMD_BUFFER,
  WGPU_CMD_POOL_MIN_BYTES,
  PTHREAD_CMD_LOAD,
  PTHREAD_CMD_LOADED,
  PTHREAD_CMD_RUN,
  bootstrapEmscriptenPthreadWorker,
  createPthreadPoolCoordinator,
  installPthreadWorkerProxy,
  encodeTestFramePacket,
} = require('../web/loader.js');

const { createHttpfsMailboxBuffer } = require('../web/workers/httpfs_mailbox.js');

function createTrackedBroadcastBus() {
  const openByName = new Map();
  const createdByName = new Map();
  const closedByName = new Map();
  const deliveries = [];

  class TrackedBroadcastChannel {
    constructor(name) {
      this.name = name;
      this.onmessage = null;
      this.closed = false;
      this.queuedCount = 0;
      if (!openByName.has(name)) {
        openByName.set(name, new Set());
      }
      openByName.get(name).add(this);
      createdByName.set(name, (createdByName.get(name) || 0) + 1);
    }

    postMessage(data) {
      if (this.closed) {
        throw new Error(`InvalidStateError: BroadcastChannel ${this.name} is closed`);
      }
      const peers = openByName.get(this.name);
      if (!peers) return;
      for (const peer of Array.from(peers)) {
        if (peer === this || peer.closed) continue;
        peer.queuedCount += 1;
        deliveries.push({ channel: this.name, target: peer, data });
        if (typeof peer.onmessage === 'function') {
          peer.onmessage({ data });
        }
      }
    }

    close() {
      if (this.closed) return;
      this.closed = true;
      closedByName.set(this.name, (closedByName.get(this.name) || 0) + 1);
      const peers = openByName.get(this.name);
      if (peers) {
        peers.delete(this);
      }
    }
  }

  return {
    TrackedBroadcastChannel,
    createdByName,
    closedByName,
    deliveries,
    openCount(name) {
      return openByName.has(name) ? openByName.get(name).size : 0;
    },
    createdCount(name) {
      return createdByName.get(name) || 0;
    },
  };
}

function createTrackedMessageChannelFactory() {
  const channels = [];

  class TrackedMessagePort {
    constructor(side) {
      this.side = side;
      this.peer = null;
      this.onmessage = null;
      this.closed = false;
      this.closeCalls = 0;
      this.sentCount = 0;
      this.receivedCount = 0;
      this.transfersSent = [];
      this.throwOnSend = false;
    }

    postMessage(data, transfer) {
      if (this.closed) {
        throw new Error('InvalidStateError: MessagePort is closed');
      }
      if (this.throwOnSend) {
        throw new Error('simulated MessagePort postMessage failure');
      }
      this.sentCount += 1;
      if (Array.isArray(transfer)) {
        this.transfersSent.push(transfer.slice());
      }
      const peer = this.peer;
      if (!peer || peer.closed) {
        return;
      }
      peer.receivedCount += 1;
      if (typeof peer.onmessage === 'function') {
        peer.onmessage({ data });
      }
    }

    close() {
      this.closeCalls += 1;
      this.closed = true;
    }
  }

  class TrackedMessageChannel {
    constructor() {
      this.port1 = new TrackedMessagePort('coord');
      this.port2 = new TrackedMessagePort('pthread');
      this.port1.peer = this.port2;
      this.port2.peer = this.port1;
      channels.push(this);
    }
  }

  return { TrackedMessageChannel, channels };
}

class MockPthreadWorker {
  constructor(script, opts) {
    this.script = script;
    this.opts = opts;
    this.name = opts && typeof opts.name === 'string' ? opts.name : '';
    this.onmessage = null;
    this.onerror = null;
    this.listeners = { message: [], error: [] };
    this.received = [];
    this.terminated = false;
    this.throwOnTransfer = false;
  }

  addEventListener(type, fn) {
    if (!this.listeners[type]) {
      this.listeners[type] = [];
    }
    this.listeners[type].push(fn);
  }

  removeEventListener(type, fn) {
    const arr = this.listeners[type];
    if (!arr) return;
    const idx = arr.indexOf(fn);
    if (idx >= 0) {
      arr.splice(idx, 1);
    }
  }

  postMessage(msg, transfer) {
    if (this.terminated) return;
    if (this.throwOnTransfer && Array.isArray(transfer) && transfer.length > 0) {
      throw new Error('DataCloneError: MessagePort transfer rejected');
    }
    this.received.push({ msg, transfer });
  }

  emitError(message = 'worker crashed') {
    const ev = { message };
    for (const fn of this.listeners.error.slice()) {
      fn.call(this, ev);
    }
    if (typeof this.onerror === 'function') {
      this.onerror(ev);
    }
  }

  terminate() {
    this.terminated = true;
  }
}

test('W190 (#868): 1,200+ host_wgpu_submit and host_log calls construct ZERO BroadcastChannels when MessagePort is attached', () => {
  const bus = createTrackedBroadcastBus();
  const mcFactory = createTrackedMessageChannelFactory();
  const sab = createHttpfsMailboxBuffer(1024);

  const coordWgpuBatches = [];
  const coordProgressMsgs = [];

  const { WrappedWorker, coordinator } = installPthreadWorkerProxy({
    WorkerCtor: MockPthreadWorker,
    getHttpfsMailboxSab: () => sab,
    MessageChannelCtor: mcFactory.TrackedMessageChannel,
    redirectHref: 'http://127.0.0.1:8080/loader.js',
    onPthreadTxMessage: (msg, workerName) => {
      if (msg && msg.type === PTHREAD_TX_MSG_WGPU_SUBMIT) {
        coordWgpuBatches.push({ workerName, bytes: new Uint8Array(msg.cmdBuffer) });
      } else if (msg && msg.type === PTHREAD_TX_MSG_PROGRESS) {
        coordProgressMsgs.push({ workerName, payload: msg.payload });
      }
    },
  });

  const worker = new WrappedWorker('./dist/runtime.js', { name: 'em-pthread-1' });
  // Single combined bootstrap message carries both SAB and TX port before CMD_LOAD.
  assert.equal(worker.received.length, 1);
  assert.equal(worker.received[0].msg.__playgta6_httpfs_sab, sab);
  assert.ok(worker.received[0].msg.__playgta6_tx_port);
  assert.deepEqual(worker.received[0].transfer, [worker.received[0].msg.__playgta6_tx_port]);
  assert.equal(coordinator.getTelemetry().pthreadActiveTxPorts, 1);

  // Bootstrap the pthread side and feed the initial control message from the coordinator.
  const scopeListeners = [];
  const scope = {
    name: 'em-pthread-1',
    location: { href: 'http://127.0.0.1:8080/loader.js' },
    WebAssembly: { Instance: WebAssembly.Instance },
    addEventListener(type, fn) {
      if (type === 'message') scopeListeners.push(fn);
    },
    removeEventListener(type, fn) {
      const idx = scopeListeners.indexOf(fn);
      if (idx >= 0) scopeListeners.splice(idx, 1);
    },
  };

  class MockHttpfsClient {
    constructor(buf) {
      this.buf = buf;
    }
  }

  const handle = bootstrapEmscriptenPthreadWorker({
    scope,
    importScriptsFn: () => {},
    BroadcastChannelCtor: bus.TrackedBroadcastChannel,
    httpfsMailboxApi: { HttpfsMailboxClient: MockHttpfsClient },
  });

  let stopped = false;
  for (const fn of scopeListeners.slice()) {
    fn({
      data: worker.received[0].msg,
      stopImmediatePropagation() {
        stopped = true;
      },
    });
  }
  assert.equal(stopped, true, 'Bootstrap control message must stop propagation before Emscripten');
  assert.equal(handle.hasHttpfsSab(), true);
  assert.equal(handle.hasTxPort(), true);
  assert.equal(bus.openCount(PTHREAD_HTTPFS_SAB_CHANNEL_NAME), 0);

  const mem = new WebAssembly.Memory({ initial: 1, maximum: 1 });
  const memU8 = new Uint8Array(mem.buffer);
  handle.hostImports.setMemory(mem);

  const frameBytes = new Uint8Array(encodeTestFramePacket(0.25, 0.5, 0.75, 1.0));
  memU8.set(frameBytes, 512);

  const bcProgressBefore = bus.createdCount(PROGRESS_CHANNEL_NAME);
  const bcWgpuBefore = bus.createdCount(PTHREAD_WGPU_CHANNEL_NAME);

  const ITERATIONS = 1200;
  for (let i = 0; i < ITERATIONS; i += 1) {
    const line = `render-batch-${i}`;
    const encoded = new TextEncoder().encode(line);
    memU8.set(encoded, 32);
    handle.hostImports.env.host_log(32n, BigInt(encoded.byteLength));
    handle.hostImports.env.host_wgpu_submit(512n, BigInt(frameBytes.byteLength));
  }

  // Zero BroadcastChannels constructed or left open on the hot paths!
  assert.equal(bus.createdCount(PROGRESS_CHANNEL_NAME), bcProgressBefore);
  assert.equal(bus.createdCount(PTHREAD_WGPU_CHANNEL_NAME), bcWgpuBefore);
  assert.equal(bus.openCount(PROGRESS_CHANNEL_NAME), 0);
  assert.equal(bus.openCount(PTHREAD_WGPU_CHANNEL_NAME), 0);

  // Every batch and log line arrived at the coordinator via the reused MessagePort.
  assert.equal(coordWgpuBatches.length, ITERATIONS);
  assert.equal(coordProgressMsgs.length, ITERATIONS);
  assert.equal(coordProgressMsgs[0].payload.label, '[wasm] render-batch-0');
  assert.equal(coordProgressMsgs[ITERATIONS - 1].payload.label, `[wasm] render-batch-${ITERATIONS - 1}`);
  assert.deepEqual(coordWgpuBatches[0].bytes, frameBytes);
  assert.deepEqual(coordWgpuBatches[ITERATIONS - 1].bytes, frameBytes);

  // Verify ArrayBuffer transfer list was passed on every host_wgpu_submit call.
  const pthreadPort = mcFactory.channels[0].port2;
  assert.equal(pthreadPort.transfersSent.length, ITERATIONS);
  assert.ok(pthreadPort.transfersSent[0][0] instanceof ArrayBuffer);
  // Pthread port is strictly send-only (nothing ever sent or queued to pthread side).
  assert.equal(pthreadPort.onmessage, null);
  assert.equal(pthreadPort.receivedCount, 0);

  handle.cleanup();
  worker.terminate();
  assert.equal(mcFactory.channels[0].port1.closed, true);
  assert.equal(mcFactory.channels[0].port2.closed, true);
  assert.equal(coordinator.getTelemetry().pthreadActiveTxPorts, 0);
});

test('W190 (#868): messages from pthread A go ONLY to coordinator port with ZERO fan-out to sibling pthreads', () => {
  const bus = createTrackedBroadcastBus();
  const mcFactory = createTrackedMessageChannelFactory();
  const sab = createHttpfsMailboxBuffer(1024);
  const receivedByCoordinator = [];

  const { WrappedWorker, coordinator } = installPthreadWorkerProxy({
    WorkerCtor: MockPthreadWorker,
    getHttpfsMailboxSab: () => sab,
    MessageChannelCtor: mcFactory.TrackedMessageChannel,
    onPthreadTxMessage: (msg, workerName) => {
      receivedByCoordinator.push({ workerName, msg });
    },
  });

  const THREAD_COUNT = 16;
  const pthreads = [];
  const mem = new WebAssembly.Memory({ initial: 1, maximum: 1 });
  const memU8 = new Uint8Array(mem.buffer);
  const frameBytes = new Uint8Array(encodeTestFramePacket(0.1, 0.2, 0.3, 1.0));
  memU8.set(frameBytes, 256);

  for (let i = 1; i <= THREAD_COUNT; i += 1) {
    const w = new WrappedWorker('./dist/runtime.js', { name: `em-pthread-${i}` });
    const txPort = w.received[0].msg.__playgta6_tx_port;
    assert.ok(txPort);

    const h = bootstrapEmscriptenPthreadWorker({
      scope: { name: `em-pthread-${i}`, WebAssembly: { Instance: WebAssembly.Instance } },
      importScriptsFn: () => {},
      BroadcastChannelCtor: bus.TrackedBroadcastChannel,
      httpfsMailboxSab: sab,
      httpfsMailboxApi: { HttpfsMailboxClient: class {} },
      txPort,
    });
    h.hostImports.setMemory(mem);
    pthreads.push({ worker: w, handle: h, channel: mcFactory.channels[i - 1] });
  }

  assert.equal(coordinator.getTelemetry().pthreadActiveTxPorts, THREAD_COUNT);

  // Burst 200 sends exclusively from pthread-1.
  const msgBytes = new TextEncoder().encode('from-pthread-1');
  memU8.set(msgBytes, 16);
  for (let n = 0; n < 200; n += 1) {
    pthreads[0].handle.hostImports.env.host_log(16n, BigInt(msgBytes.byteLength));
    pthreads[0].handle.hostImports.env.host_wgpu_submit(256n, BigInt(frameBytes.byteLength));
  }

  // Coordinator received all 400 messages tagged with em-pthread-1.
  assert.equal(receivedByCoordinator.length, 400);
  for (const item of receivedByCoordinator) {
    assert.equal(item.workerName, 'em-pthread-1');
  }

  // Zero BroadcastChannel traffic and zero fan-out to sibling pthreads 2..16!
  assert.equal(bus.createdCount(PROGRESS_CHANNEL_NAME), 0);
  assert.equal(bus.createdCount(PTHREAD_WGPU_CHANNEL_NAME), 0);
  assert.equal(pthreads[0].channel.port1.receivedCount, 400);
  assert.equal(pthreads[0].channel.port2.receivedCount, 0);
  for (let i = 1; i < THREAD_COUNT; i += 1) {
    assert.equal(pthreads[i].channel.port1.receivedCount, 0, `sibling coord port ${i + 1} must see 0 messages`);
    assert.equal(pthreads[i].channel.port2.receivedCount, 0, `sibling pthread port ${i + 1} must see 0 messages`);
  }

  for (const p of pthreads) {
    p.handle.cleanup();
    p.worker.terminate();
  }
  assert.equal(coordinator.getTelemetry().pthreadActiveTxPorts, 0);
});

test('W190 (#868): graceful fallback to BroadcastChannel when MessagePort is unavailable or throws mid-session', () => {
  const bus = createTrackedBroadcastBus();
  const uiLogs = [];
  const coordWgpu = [];

  const uiCh = new bus.TrackedBroadcastChannel(PROGRESS_CHANNEL_NAME);
  uiCh.onmessage = (ev) => uiLogs.push(ev.data);
  const wgpuCh = new bus.TrackedBroadcastChannel(PTHREAD_WGPU_CHANNEL_NAME);
  wgpuCh.onmessage = (ev) => coordWgpu.push(ev.data);

  const mem = new WebAssembly.Memory({ initial: 1, maximum: 1 });
  const memU8 = new Uint8Array(mem.buffer);
  const frameBytes = new Uint8Array(encodeTestFramePacket(0.4, 0.5, 0.6, 1.0));
  memU8.set(frameBytes, 128);

  // Case A: No MessagePort attached at all -> falls back to short-lived BroadcastChannel.
  const noPortHandle = bootstrapEmscriptenPthreadWorker({
    scope: { name: 'em-pthread-fallback-a', WebAssembly: { Instance: WebAssembly.Instance } },
    importScriptsFn: () => {},
    BroadcastChannelCtor: bus.TrackedBroadcastChannel,
    httpfsMailboxApi: null,
  });
  noPortHandle.hostImports.setMemory(mem);
  assert.equal(noPortHandle.hasTxPort(), false);

  const txtA = new TextEncoder().encode('fallback-no-port');
  memU8.set(txtA, 16);
  noPortHandle.hostImports.env.host_log(16n, BigInt(txtA.byteLength));
  noPortHandle.hostImports.env.host_wgpu_submit(128n, BigInt(frameBytes.byteLength));

  assert.equal(uiLogs.length, 1);
  assert.equal(uiLogs[0].label, '[wasm] fallback-no-port');
  assert.equal(coordWgpu.length, 1);
  assert.deepEqual(new Uint8Array(coordWgpu[0].cmdBuffer), frameBytes);
  assert.equal(bus.openCount(PROGRESS_CHANNEL_NAME), 1);
  assert.equal(bus.openCount(PTHREAD_WGPU_CHANNEL_NAME), 1);
  noPortHandle.cleanup();

  // Case B: Attached MessagePort throws mid-session -> closes broken port and immediately falls back without losing the message.
  const mcFactory = createTrackedMessageChannelFactory();
  const mc = new mcFactory.TrackedMessageChannel();
  const portReceived = [];
  mc.port1.onmessage = (ev) => portReceived.push(ev.data);

  const flakyHandle = bootstrapEmscriptenPthreadWorker({
    scope: { name: 'em-pthread-fallback-b', WebAssembly: { Instance: WebAssembly.Instance } },
    importScriptsFn: () => {},
    BroadcastChannelCtor: bus.TrackedBroadcastChannel,
    httpfsMailboxApi: null,
    txPort: mc.port2,
  });
  flakyHandle.hostImports.setMemory(mem);
  assert.equal(flakyHandle.hasTxPort(), true);

  // First send succeeds on MessagePort.
  const txtOk = new TextEncoder().encode('port-ok');
  memU8.set(txtOk, 16);
  flakyHandle.hostImports.env.host_log(16n, BigInt(txtOk.byteLength));
  assert.equal(portReceived.length, 1);
  assert.equal(portReceived[0].payload.label, '[wasm] port-ok');

  // Now simulate detached/failing port -> must close port AND deliver both GPU batch and log via BroadcastChannel.
  mc.port2.throwOnSend = true;
  flakyHandle.hostImports.env.host_wgpu_submit(128n, BigInt(frameBytes.byteLength));
  assert.equal(flakyHandle.hasTxPort(), false);
  assert.equal(flakyHandle.channels.txPort, null);
  assert.equal(mc.port2.closed, true);
  assert.equal(coordWgpu.length, 2);
  assert.deepEqual(new Uint8Array(coordWgpu[1].cmdBuffer), frameBytes);

  const txtAfter = new TextEncoder().encode('after-port-fail');
  memU8.set(txtAfter, 16);
  flakyHandle.hostImports.env.host_log(16n, BigInt(txtAfter.byteLength));
  assert.equal(uiLogs.length, 2);
  assert.equal(uiLogs[1].label, '[wasm] after-port-fail');

  flakyHandle.cleanup();
  mc.port1.close();
  uiCh.close();
  wgpuCh.close();
});

test('W190 (#868): coordinator teardown closes ports on terminate, error, transfer failure, and closeTxPorts()', () => {
  const mcFactory = createTrackedMessageChannelFactory();
  const sab = createHttpfsMailboxBuffer(1024);

  // Custom worker class where worker #1 rejects transfer list on postMessage.
  let spawnIdx = 0;
  class SelectiveRejectWorker extends MockPthreadWorker {
    constructor(script, opts) {
      super(script, opts);
      spawnIdx += 1;
      if (spawnIdx === 1) {
        this.throwOnTransfer = true;
      }
    }
  }

  const { WrappedWorker, coordinator } = installPthreadWorkerProxy({
    WorkerCtor: SelectiveRejectWorker,
    getHttpfsMailboxSab: () => sab,
    MessageChannelCtor: mcFactory.TrackedMessageChannel,
    onPthreadTxMessage: () => {},
  });

  // Worker 1 rejects port transfer -> both ports closed cleanly, SAB still delivered via retry!
  const wReject = new WrappedWorker('./dist/runtime.js', { name: 'em-pthread-reject' });
  assert.equal(mcFactory.channels.length, 1);
  assert.equal(mcFactory.channels[0].port1.closed, true);
  assert.equal(mcFactory.channels[0].port2.closed, true);
  assert.equal(wReject.received.length, 1);
  assert.equal(wReject.received[0].msg.__playgta6_httpfs_sab, sab);
  assert.equal(wReject.received[0].msg.__playgta6_tx_port, undefined);
  assert.equal(coordinator.getTelemetry().pthreadActiveTxPorts, 0);

  // Spawn 5 healthy pthreads -> 5 active coordinator TX ports.
  const workers = [];
  for (let i = 1; i <= 5; i += 1) {
    workers.push(new WrappedWorker('./dist/runtime.js', { name: `em-pthread-${i}` }));
  }
  assert.equal(coordinator.getTelemetry().pthreadActiveTxPorts, 5);

  // Terminate worker 1 (idempotent double terminate).
  workers[0].terminate();
  workers[0].terminate();
  assert.equal(mcFactory.channels[1].port1.closed, true);
  assert.equal(mcFactory.channels[1].port1.onmessage, null);
  assert.equal(mcFactory.channels[1].port1.closeCalls, 1);
  assert.equal(coordinator.getTelemetry().pthreadActiveTxPorts, 4);

  // Error on worker 2 closes its coordinator port.
  workers[1].emitError('oom');
  assert.equal(mcFactory.channels[2].port1.closed, true);
  assert.equal(mcFactory.channels[2].port1.onmessage, null);
  assert.equal(coordinator.getTelemetry().pthreadActiveTxPorts, 3);

  // Session stop / restart via closeTxPorts() closes all remaining 3 coordinator ports.
  coordinator.closeTxPorts();
  assert.equal(coordinator.getTelemetry().pthreadActiveTxPorts, 0);
  for (let idx = 3; idx <= 5; idx += 1) {
    assert.equal(mcFactory.channels[idx].port1.closed, true);
    assert.equal(mcFactory.channels[idx].port1.onmessage, null);
    assert.equal(mcFactory.channels[idx].port1.closeCalls, 1);
  }

  // Terminating already-port-closed workers must not double-close or underflow counters.
  workers[2].terminate();
  workers[3].terminate();
  workers[4].terminate();
  wReject.terminate();
  assert.equal(mcFactory.channels[3].port1.closeCalls, 1);
  assert.equal(coordinator.getTelemetry().pthreadActiveTxPorts, 0);
  assert.equal(coordinator.getTrackedCount(), 0);
});

test('W190 (#868): end-to-end VM boot with real Node MessageChannel delivers 1,050 GPU batches + 50 logs with 0 pthread BroadcastChannels', async () => {
  const bus = createTrackedBroadcastBus();
  const loaderPath = path.join(repoRoot, 'web/loader.js');
  const loaderSource = fs.readFileSync(loaderPath, 'utf8');

  const uiMessages = [];
  const uiProgressChannel = new bus.TrackedBroadcastChannel(PROGRESS_CHANNEL_NAME);
  uiProgressChannel.onmessage = (ev) => uiMessages.push(ev.data);

  const coordWorkers = new Map();
  const spawnedPthreads = [];

  class E2EMockWorker extends MockPthreadWorker {
    constructor(script, opts) {
      super(script, opts);
      coordWorkers.set(opts && opts.name ? opts.name : script, this);
      if (opts && typeof opts.name === 'string' && opts.name.startsWith('em-pthread')) {
        spawnedPthreads.push(this);
      }
    }
  }

  const coordSelf = {
    name: '',
    location: { href: 'http://127.0.0.1:8080/loader.js' },
    MessageChannel,
    onmessage: null,
    postMessage() {},
  };

  const coordSandbox = {
    self: coordSelf,
    globalThis: null,
    URL,
    BroadcastChannel: bus.TrackedBroadcastChannel,
    MessageChannel,
    Worker: E2EMockWorker,
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
      vm.runInContext(fs.readFileSync(resolved, 'utf8'), coordContext, { filename: resolved });
    },
  };
  coordSandbox.globalThis = coordSandbox;
  const coordContext = vm.createContext(coordSandbox);
  vm.runInContext(loaderSource, coordContext, { filename: loaderPath });

  // Initialize coordinator session.
  coordSelf.onmessage({
    data: { canvas: { id: 'c' }, width: 1280, height: 720, dataBase: '/data/' },
  });
  const gpuWorker = coordWorkers.get('./workers/wgpu_worker.js');
  assert.ok(gpuWorker);

  // Spawn 2 pthreads through the coordinator's wrapped Worker constructor.
  const coordPthread1 = new coordSelf.Worker('./dist/runtime.js', { name: 'em-pthread-1' });
  const coordPthread2 = new coordSelf.Worker('./dist/runtime.js', { name: 'em-pthread-2' });
  assert.equal(spawnedPthreads.length, 2);
  assert.equal(coordContext.PlayGta6Loader.getPthreadTelemetry().pthreadActiveTxPorts, 2);

  // Boot 2 isolated pthread VM sandboxes using their own BroadcastChannel counter to verify 0 pthread BC opens.
  const pthreadBus = createTrackedBroadcastBus();
  const pthreadInsts = [];

  for (let p = 0; p < 2; p += 1) {
    const coordW = spawnedPthreads[p];
    let capturedImports = null;
    class CaptureWasmInstance {
      constructor(_mod, imports) {
        capturedImports = imports;
      }
    }
    const listeners = [];
    const pSelf = {
      name: `em-pthread-${p + 1}`,
      location: { href: 'http://127.0.0.1:8080/loader.js' },
      onmessage: null,
      postMessage() {},
      addEventListener(type, fn) {
        if (type === 'message') listeners.push(fn);
      },
      removeEventListener(type, fn) {
        const idx = listeners.indexOf(fn);
        if (idx >= 0) listeners.splice(idx, 1);
      },
    };
    const pSandbox = {
      self: pSelf,
      globalThis: null,
      URL,
      WebAssembly: { Instance: CaptureWasmInstance, Module: WebAssembly.Module, Memory: WebAssembly.Memory },
      BroadcastChannel: pthreadBus.TrackedBroadcastChannel,
      MessageChannel,
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
      Date,
      Object,
      Array,
      Proxy,
      Reflect,
      importScripts(scriptPath) {
        if (scriptPath === './dist/runtime.js') {
          const mem = new WebAssembly.Memory({ initial: 1, maximum: 1 });
          const stub = () => {};
          stub.stub = true;
          new pSandbox.WebAssembly.Instance('mod', {
            env: {
              memory: mem,
              host_log: stub,
              host_wgpu_submit: stub,
            },
          });
          pthreadInsts.push({ mem, env: capturedImports.env, listeners, pSelf });
          return;
        }
        const resolved = path.resolve(repoRoot, 'web', scriptPath);
        vm.runInContext(fs.readFileSync(resolved, 'utf8'), pContext, { filename: resolved });
      },
    };
    pSandbox.globalThis = pSandbox;
    const pContext = vm.createContext(pSandbox);
    vm.runInContext(loaderSource, pContext, { filename: loaderPath });

    // Deliver the coordinator's initial bootstrap message (SAB + real Node MessagePort).
    const bootstrapMsg = coordW.received[0].msg;
    assert.ok(bootstrapMsg.__playgta6_httpfs_sab instanceof SharedArrayBuffer);
    assert.ok(bootstrapMsg.__playgta6_tx_port);
    for (const fn of listeners.slice()) {
      fn({ data: bootstrapMsg, stopImmediatePropagation() {} });
    }
  }

  // Record BroadcastChannel creation counts on pthread side BEFORE heavy sends.
  const pthreadProgressBefore = pthreadBus.createdCount(PROGRESS_CHANNEL_NAME);
  const pthreadWgpuBefore = pthreadBus.createdCount(PTHREAD_WGPU_CHANNEL_NAME);

  const frameBytes = new Uint8Array(encodeTestFramePacket(0.2, 0.4, 0.6, 1.0));
  const { mem, env } = pthreadInsts[0];
  const u8 = new Uint8Array(mem.buffer);
  u8.set(frameBytes, 128);

  const GPU_BATCHES = 1050;
  const LOG_LINES = 50;
  // Simulate a healthy (non-stalled) wgpu_worker that acks presented command buffers.
  const origGpuPostMessage = gpuWorker.postMessage.bind(gpuWorker);
  gpuWorker.postMessage = (msg, transfer = []) => {
    origGpuPostMessage(msg, transfer);
    if (msg && msg.cmdBuffer instanceof ArrayBuffer && typeof gpuWorker.onmessage === 'function') {
      gpuWorker.onmessage({ data: { type: 'gpu-present-ack' } });
    }
  };
  for (let i = 0; i < LOG_LINES; i += 1) {
    const encoded = new TextEncoder().encode(`hot-log-${i}`);
    u8.set(encoded, 16);
    env.host_log(16n, BigInt(encoded.byteLength));
  }
  for (let i = 0; i < GPU_BATCHES; i += 1) {
    env.host_wgpu_submit(128n, BigInt(frameBytes.byteLength));
  }

  // Wait for real Node MessagePort async delivery to drain.
  const deadline = Date.now() + 3000;
  while (Date.now() < deadline) {
    const gpuCount = gpuWorker.received.filter(
      (r) => r.msg && r.msg.cmdBuffer instanceof ArrayBuffer,
    ).length;
    const logCount = uiMessages.filter(
      (m) => m && typeof m.label === 'string' && m.label.startsWith('[wasm] hot-log-'),
    ).length;
    if (gpuCount >= GPU_BATCHES && logCount >= LOG_LINES) {
      break;
    }
    await new Promise((r) => setTimeout(r, 10));
  }

  // Zero BroadcastChannels constructed on pthread side during all 1,100 hot sends!
  assert.equal(pthreadBus.createdCount(PROGRESS_CHANNEL_NAME), pthreadProgressBefore);
  assert.equal(pthreadBus.createdCount(PTHREAD_WGPU_CHANNEL_NAME), pthreadWgpuBefore);
  assert.equal(pthreadBus.openCount(PROGRESS_CHANNEL_NAME), 0);
  assert.equal(pthreadBus.openCount(PTHREAD_WGPU_CHANNEL_NAME), 0);

  // Coordinator forwarded all 1,050 GPU batches with transferable ArrayBuffers + exactly one FIRST_PRESENT.
  const gpuBatches = gpuWorker.received.filter(
    (r) => r.msg && r.msg.cmdBuffer instanceof ArrayBuffer,
  );
  assert.equal(gpuBatches.length, GPU_BATCHES);
  assert.deepEqual(new Uint8Array(gpuBatches[0].msg.cmdBuffer), frameBytes);
  assert.deepEqual(new Uint8Array(gpuBatches[GPU_BATCHES - 1].msg.cmdBuffer), frameBytes);
  assert.equal(gpuBatches[0].transfer.length, 1);

  const labels = uiMessages.filter((m) => m && m.label).map((m) => m.label);
  assert.equal(labels.filter((l) => l === '[boot] FIRST_PRESENT').length, 1);
  assert.equal(labels.filter((l) => l.startsWith('[wasm] hot-log-')).length, LOG_LINES);
  assert.equal(labels.includes('[wasm] hot-log-0'), true);
  assert.equal(labels.includes(`[wasm] hot-log-${LOG_LINES - 1}`), true);

  // Coordinator re-init closes previous session's TX ports cleanly.
  coordSelf.onmessage({
    data: { canvas: { id: 'c2' }, width: 1280, height: 720, dataBase: '/data/' },
  });
  assert.equal(coordContext.PlayGta6Loader.getPthreadTelemetry().pthreadActiveTxPorts, 0);

  // Clean up both pthread ports & UI channel.
  coordPthread1.received[0].msg.__playgta6_tx_port.close();
  coordPthread2.received[0].msg.__playgta6_tx_port.close();
  coordPthread1.terminate();
  coordPthread2.terminate();
  uiProgressChannel.close();
});

test('W345 pthread wgpuSubmit and coordinator handlePthreadWgpuSubmit forward standalone ArrayBuffers with zero extra copy', () => {
  const bus = createTrackedBroadcastBus();
  const mcFactory = createTrackedMessageChannelFactory();
  const loaderPath = path.join(repoRoot, 'web/loader.js');
  const loaderSource = fs.readFileSync(loaderPath, 'utf8');

  const coordWorkers = new Map();
  class ZeroCopyMockWorker extends MockPthreadWorker {
    constructor(script, opts) {
      super(script, opts);
      coordWorkers.set(opts && opts.name ? opts.name : script, this);
    }
  }

  const coordSelf = {
    name: '',
    location: { href: 'http://127.0.0.1:8080/loader.js' },
    MessageChannel: mcFactory.TrackedMessageChannel,
    onmessage: null,
    postMessage() {},
  };
  const coordSandbox = {
    self: coordSelf,
    globalThis: null,
    URL,
    BroadcastChannel: bus.TrackedBroadcastChannel,
    MessageChannel: mcFactory.TrackedMessageChannel,
    Worker: ZeroCopyMockWorker,
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
      vm.runInContext(fs.readFileSync(resolved, 'utf8'), coordContext, { filename: resolved });
    },
  };
  coordSandbox.globalThis = coordSandbox;
  const coordContext = vm.createContext(coordSandbox);
  vm.runInContext(loaderSource, coordContext, { filename: loaderPath });

  coordSelf.onmessage({
    data: { canvas: { id: 'c' }, width: 1280, height: 720, dataBase: '/data/' },
  });
  const gpuWorker = coordWorkers.get('./workers/wgpu_worker.js');
  assert.ok(gpuWorker);
  // W345: live Wasm64 boot disables per-batch resource replay shadow-copying in wgpu_worker init.
  assert.equal(gpuWorker.received[0].msg.enableResourceReplay, false);

  // Spawn one pthread so coordinator creates a tracked MessageChannel pair.
  const pthreadWorker = new coordSelf.Worker('./dist/runtime.js', { name: 'em-pthread-1' });
  const txPort = pthreadWorker.received[0].msg.__playgta6_tx_port;
  assert.ok(txPort);

  // 1. Over txPort: coordinator handlePthreadWgpuSubmit transfers the EXACT ArrayBuffer straight through.
  const frameBuf1 = encodeTestFramePacket(0.1, 0.2, 0.3, 1.0);
  txPort.postMessage({ type: PTHREAD_TX_MSG_WGPU_SUBMIT, cmdBuffer: frameBuf1 }, [frameBuf1]);

  const txSubmitted = gpuWorker.received.filter((r) => r.msg && r.msg.cmdBuffer instanceof ArrayBuffer);
  assert.equal(txSubmitted.length, 1);
  assert.equal(
    txSubmitted[0].msg.cmdBuffer,
    frameBuf1,
    'handlePthreadWgpuSubmit must transfer the exact txPort ArrayBuffer with zero extra copy',
  );
  assert.equal(txSubmitted[0].transfer.length, 1);
  assert.equal(txSubmitted[0].transfer[0], frameBuf1);

  // 2. Over BroadcastChannel fallback: handlePthreadWgpuSubmit ALSO transfers the exact ArrayBuffer without copying.
  const bcSender = new bus.TrackedBroadcastChannel(PTHREAD_WGPU_CHANNEL_NAME);
  const frameBuf2 = encodeTestFramePacket(0.4, 0.5, 0.6, 1.0);
  bcSender.postMessage({ type: PTHREAD_TX_MSG_WGPU_SUBMIT, cmdBuffer: frameBuf2 });
  bcSender.close();

  const allSubmitted = gpuWorker.received.filter((r) => r.msg && r.msg.cmdBuffer instanceof ArrayBuffer);
  assert.equal(allSubmitted.length, 2);
  assert.equal(
    allSubmitted[1].msg.cmdBuffer,
    frameBuf2,
    'handlePthreadWgpuSubmit must transfer the exact BroadcastChannel ArrayBuffer with zero extra copy',
  );
  assert.equal(allSubmitted[1].transfer.length, 1);
  assert.equal(allSubmitted[1].transfer[0], frameBuf2);

  // 3. Pthread-side wgpuSubmit transfers standalone Uint8Array.buffer directly with zero .slice() copies.
  const directPort = {
    sent: [],
    postMessage(msg, transfer) {
      this.sent.push({ msg, transfer });
    },
    close() {},
  };
  const handle = bootstrapEmscriptenPthreadWorker({
    scope: { name: 'em-pthread-zero-copy', WebAssembly: { Instance: WebAssembly.Instance } },
    importScriptsFn: () => {},
    BroadcastChannelCtor: bus.TrackedBroadcastChannel,
    httpfsMailboxApi: null,
    txPort: directPort,
  });

  // Patch Uint8Array.prototype.slice temporarily during host_wgpu_submit to prove zero redundant slice().
  const origSlice = Uint8Array.prototype.slice;
  let u8SliceCalls = 0;
  Uint8Array.prototype.slice = function patchedSlice(...args) {
    u8SliceCalls += 1;
    return origSlice.apply(this, args);
  };
  try {
    const mem = new WebAssembly.Memory({ initial: 1, maximum: 1 });
    const frameBytes = new Uint8Array(encodeTestFramePacket(0.7, 0.8, 0.9, 1.0));
    new Uint8Array(mem.buffer).set(frameBytes, 64);
    handle.hostImports.setMemory(mem);
    handle.hostImports.env.host_wgpu_submit(64n, BigInt(frameBytes.byteLength));
  } finally {
    Uint8Array.prototype.slice = origSlice;
  }

  assert.equal(u8SliceCalls, 0, 'host_wgpu_submit -> pthread wgpuSubmit must perform zero bytes.slice() calls');
  assert.equal(directPort.sent.length, 1);
  assert.ok(directPort.sent[0].msg.cmdBuffer instanceof ArrayBuffer);
  assert.deepEqual(directPort.sent[0].transfer, [directPort.sent[0].msg.cmdBuffer]);
  assert.deepEqual(
    new Uint8Array(directPort.sent[0].msg.cmdBuffer),
    new Uint8Array(encodeTestFramePacket(0.7, 0.8, 0.9, 1.0)),
  );

  handle.cleanup();
  pthreadWorker.terminate();
});

test('G75 pthread warm cmdBuffer pool recycles large ArrayBuffers end-to-end and drops cleanly on port close', () => {
  const bus = createTrackedBroadcastBus();
  const mcFactory = createTrackedMessageChannelFactory();
  const loaderPath = path.join(repoRoot, 'web/loader.js');
  const loaderSource = fs.readFileSync(loaderPath, 'utf8');

  const coordWorkers = new Map();
  class RecycleMockWorker extends MockPthreadWorker {
    constructor(script, opts) {
      super(script, opts);
      coordWorkers.set(opts && opts.name ? opts.name : script, this);
    }
  }

  const coordSelf = {
    name: '',
    location: { href: 'http://127.0.0.1:8080/loader.js' },
    MessageChannel: mcFactory.TrackedMessageChannel,
    onmessage: null,
    postMessage() {},
  };
  const coordSandbox = {
    self: coordSelf,
    globalThis: null,
    URL,
    BroadcastChannel: bus.TrackedBroadcastChannel,
    MessageChannel: mcFactory.TrackedMessageChannel,
    Worker: RecycleMockWorker,
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
      vm.runInContext(fs.readFileSync(resolved, 'utf8'), coordContext, { filename: resolved });
    },
  };
  coordSandbox.globalThis = coordSandbox;
  const coordContext = vm.createContext(coordSandbox);
  vm.runInContext(loaderSource, coordContext, { filename: loaderPath });

  coordSelf.onmessage({
    data: { canvas: { id: 'c' }, width: 1280, height: 720, dataBase: '/data/' },
  });
  const gpuWorker = coordWorkers.get('./workers/wgpu_worker.js');
  assert.ok(gpuWorker && typeof gpuWorker.onmessage === 'function');

  // Spawn one pthread and wire its txPort into bootstrapEmscriptenPthreadWorker.
  const pthreadWorker = new coordSelf.Worker('./dist/runtime.js', { name: 'em-pthread-recycle' });
  const txPort = pthreadWorker.received[0].msg.__playgta6_tx_port;
  assert.ok(txPort);

  const handle = bootstrapEmscriptenPthreadWorker({
    scope: { name: 'em-pthread-recycle', WebAssembly: { Instance: WebAssembly.Instance } },
    importScriptsFn: () => {},
    BroadcastChannelCtor: bus.TrackedBroadcastChannel,
    httpfsMailboxApi: null,
    txPort,
  });

  // 1 MiB Wasm linear memory (16 pages) so we can submit >= 256 KiB batches.
  const mem = new WebAssembly.Memory({ initial: 16, maximum: 16 });
  const memU8 = new Uint8Array(mem.buffer);
  handle.hostImports.setMemory(mem);

  // Frame 1: 300 KiB large batch (>= 256 KiB) with pool empty -> allocates fresh buffer with recycleBuffer: true.
  const len1 = 300 * 1024;
  memU8.fill(0, 0, len1);
  // Write a valid NOP packet header at offset 0 (opcode=0, payloadBytes=len1-8) so FIRST_PRESENT stays false.
  const dv1 = new DataView(mem.buffer, 0, len1);
  dv1.setUint16(0, 0, true);
  dv1.setUint32(4, len1 - 8, true);
  memU8[8] = 0xab;
  memU8[len1 - 1] = 0xcd;

  handle.hostImports.env.host_wgpu_submit(0n, BigInt(len1));
  const submitsAfter1 = gpuWorker.received.filter((r) => r.msg && r.msg.cmdBuffer instanceof ArrayBuffer);
  assert.equal(submitsAfter1.length, 1);
  const buf1 = submitsAfter1[0].msg.cmdBuffer;
  assert.equal(buf1.byteLength, len1);
  assert.equal(submitsAfter1[0].msg.byteLength, len1);
  assert.equal(submitsAfter1[0].msg.recycleBuffer, true);
  assert.equal(submitsAfter1[0].transfer.length, 1);
  assert.equal(submitsAfter1[0].transfer[0], buf1);
  assert.equal(new Uint8Array(buf1)[8], 0xab);
  assert.equal(new Uint8Array(buf1)[len1 - 1], 0xcd);
  assert.equal(handle.cmdBufferPool.size, 0);

  // Poison trailing bytes with a PRESENT packet before recycling to verify Frame 2's byteLength clamp!
  const presentTail = new Uint8Array(encodeTestFramePacket(0.1, 0.2, 0.3, 1.0));
  new Uint8Array(buf1).set(presentTail, 280 * 1024);

  // Simulate wgpu_worker returning buf1 via recycle-cmd-buffer -> coordinator -> txPort -> pthread pool.
  gpuWorker.onmessage({ data: { type: PTHREAD_TX_MSG_RECYCLE_CMD_BUFFER, cmdBuffer: buf1 } });
  assert.equal(handle.cmdBufferPool.size, 1, 'recycled buffer must return to pthread warm pool');

  // Frame 2: 280 KiB batch (<= buf1.byteLength, within slack) -> reuses EXACT buf1 with byteLength = 280 KiB.
  const len2 = 280 * 1024;
  memU8.fill(0, 0, len2);
  const dv2 = new DataView(mem.buffer, 0, len2);
  dv2.setUint16(0, 0, true);
  dv2.setUint32(4, len2 - 8, true);
  memU8[8] = 0x55;
  memU8[len2 - 1] = 0x77;

  handle.hostImports.env.host_wgpu_submit(0n, BigInt(len2));
  const submitsAfter2 = gpuWorker.received.filter((r) => r.msg && r.msg.cmdBuffer instanceof ArrayBuffer);
  assert.equal(submitsAfter2.length, 2);
  assert.equal(
    submitsAfter2[1].msg.cmdBuffer,
    buf1,
    'frame 2 must reuse the exact recycled ArrayBuffer from frame 1',
  );
  assert.equal(submitsAfter2[1].msg.byteLength, len2);
  assert.equal(submitsAfter2[1].msg.recycleBuffer, true);
  assert.equal(new Uint8Array(submitsAfter2[1].msg.cmdBuffer)[8], 0x55);
  assert.equal(new Uint8Array(submitsAfter2[1].msg.cmdBuffer)[len2 - 1], 0x77);
  assert.equal(handle.cmdBufferPool.size, 0);

  // Small (< 256 KiB) batch must NOT set recycleBuffer and must NOT consume from the pool.
  gpuWorker.onmessage({ data: { type: PTHREAD_TX_MSG_RECYCLE_CMD_BUFFER, cmdBuffer: buf1 } });
  assert.equal(handle.cmdBufferPool.size, 1);
  const smallBytes = new Uint8Array(encodeTestFramePacket(0.4, 0.5, 0.6, 1.0));
  assert.ok(smallBytes.byteLength < WGPU_CMD_POOL_MIN_BYTES);
  memU8.set(smallBytes, 1024);
  handle.hostImports.env.host_wgpu_submit(1024n, BigInt(smallBytes.byteLength));
  const submitsAfter3 = gpuWorker.received.filter((r) => r.msg && r.msg.cmdBuffer instanceof ArrayBuffer);
  assert.equal(submitsAfter3.length, 3);
  assert.notEqual(submitsAfter3[2].msg.cmdBuffer, buf1);
  assert.equal(submitsAfter3[2].msg.recycleBuffer, undefined);
  assert.equal(handle.cmdBufferPool.size, 1, 'small submit must leave warm pool untouched');

  // Closing the pthread worker / txPort causes any late recycle-cmd-buffer to drop cleanly.
  handle.cleanup();
  pthreadWorker.terminate();
  const lateBuf = new ArrayBuffer(300 * 1024);
  assert.doesNotThrow(() => {
    gpuWorker.onmessage({ data: { type: PTHREAD_TX_MSG_RECYCLE_CMD_BUFFER, cmdBuffer: lateBuf } });
  });
});

test('Orch-32 P0 #2: handlePthreadWgpuSubmit bounds in-flight pure Presents, never drops resource packets, and recovers on ack/error', () => {
  const bus = createTrackedBroadcastBus();
  const mcFactory = createTrackedMessageChannelFactory();
  const loaderPath = path.join(repoRoot, 'web/loader.js');
  const loaderSource = fs.readFileSync(loaderPath, 'utf8');

  const coordWorkers = new Map();
  class BackpressureMockWorker extends MockPthreadWorker {
    constructor(script, opts) {
      super(script, opts);
      coordWorkers.set(opts && opts.name ? opts.name : script, this);
    }
  }

  const coordSelf = {
    name: '',
    location: { href: 'http://127.0.0.1:8080/loader.js' },
    MessageChannel: mcFactory.TrackedMessageChannel,
    onmessage: null,
    postMessage() {},
  };
  const coordSandbox = {
    self: coordSelf,
    globalThis: null,
    URL,
    BroadcastChannel: bus.TrackedBroadcastChannel,
    MessageChannel: mcFactory.TrackedMessageChannel,
    Worker: BackpressureMockWorker,
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
      vm.runInContext(fs.readFileSync(resolved, 'utf8'), coordContext, { filename: resolved });
    },
  };
  coordSandbox.globalThis = coordSandbox;
  const coordContext = vm.createContext(coordSandbox);
  vm.runInContext(loaderSource, coordContext, { filename: loaderPath });

  coordSelf.onmessage({
    data: { canvas: { id: 'c' }, width: 1280, height: 720, dataBase: '/data/' },
  });
  const gpuWorker = coordWorkers.get('./workers/wgpu_worker.js');
  assert.ok(gpuWorker);

  const pthreadWorker = new coordSelf.Worker('./dist/runtime.js', { name: 'em-pthread-bp' });
  const txPort = pthreadWorker.received[0].msg.__playgta6_tx_port;
  assert.ok(txPort);

  const makeResourcePlusPresentBatch = () => {
    // CREATE_BUFFER (opcode 3, payload 16) + PRESENT (opcode 7, payload 0)
    const buf = new ArrayBuffer(8 + 16 + 8);
    const dv = new DataView(buf);
    dv.setUint16(0, 3, true);
    dv.setUint32(4, 16, true);
    dv.setUint16(24, 7, true);
    dv.setUint32(28, 0, true);
    return buf;
  };

  // 1. Submit 20 pure Present batches while gpuWorker is stalled (0 acks).
  //    Only DEFAULT_MAX_IN_FLIGHT_GPU_PRESENTS (4) must be posted; 16 are coalesced.
  for (let i = 0; i < 20; i += 1) {
    const pureFrame = encodeTestFramePacket(0.1, 0.2, 0.3, 1.0);
    txPort.postMessage({ type: PTHREAD_TX_MSG_WGPU_SUBMIT, cmdBuffer: pureFrame }, [pureFrame]);
  }
  const postedAfterBurst = gpuWorker.received.filter((r) => r.msg && r.msg.cmdBuffer instanceof ArrayBuffer);
  assert.equal(postedAfterBurst.length, 4);
  assert.equal(postedAfterBurst[0].msg.coalescePresent, undefined);

  const statusDuringStall = coordContext.PlayGta6Loader.getLoaderStatus();
  assert.equal(statusDuringStall.inFlightGpuPresents, 4);
  assert.equal(statusDuringStall.maxInFlightPresents, 4);
  assert.equal(statusDuringStall.coalescedPresentSubmissions, 16);

  // 2. Submit a batch containing resource creation + Present while still backpressured:
  //    it MUST NEVER be dropped, and MUST be tagged with coalescePresent: true.
  const resBatch = makeResourcePlusPresentBatch();
  txPort.postMessage({ type: PTHREAD_TX_MSG_WGPU_SUBMIT, cmdBuffer: resBatch }, [resBatch]);
  const postedAfterRes = gpuWorker.received.filter((r) => r.msg && r.msg.cmdBuffer instanceof ArrayBuffer);
  assert.equal(postedAfterRes.length, 5);
  assert.equal(postedAfterRes[4].msg.cmdBuffer, resBatch);
  assert.equal(postedAfterRes[4].msg.coalescePresent, true);
  assert.equal(coordContext.PlayGta6Loader.getLoaderStatus().coalescedPresentSubmissions, 17);
  assert.equal(coordContext.PlayGta6Loader.getLoaderStatus().inFlightGpuPresents, 5);

  // 3. Acks (`gpu-present-ack`, `frame-stats`, `first-present`, `first-visible-frame`, `gpu-telemetry`)
  //    decrement inFlightGpuPresents and flush the latest pending coalesced pure Present once < max.
  gpuWorker.onmessage({ data: { type: 'gpu-present-ack' } }); // 5 -> 4 (still == max, no flush yet)
  assert.equal(coordContext.PlayGta6Loader.getLoaderStatus().inFlightGpuPresents, 4);
  gpuWorker.onmessage({ data: { type: 'frame-stats', framesPresented: 2 } }); // 4 -> 3 -> flushes 1 coalesced -> 4
  const postedAfterFlush = gpuWorker.received.filter((r) => r.msg && r.msg.cmdBuffer instanceof ArrayBuffer);
  assert.equal(postedAfterFlush.length, 6);
  assert.equal(postedAfterFlush[5].msg.coalescePresent, true);
  assert.equal(coordContext.PlayGta6Loader.getLoaderStatus().inFlightGpuPresents, 4);

  gpuWorker.onmessage({ data: { type: 'first-present', framesPresented: 3 } });
  gpuWorker.onmessage({ data: { type: 'first-visible-frame', framesPresented: 4 } });
  gpuWorker.onmessage({ data: { type: 'gpu-telemetry', frameStats: { framesPresented: 5 } } });
  assert.equal(coordContext.PlayGta6Loader.getLoaderStatus().inFlightGpuPresents, 1);

  // 4. Worker error resets inFlightGpuPresents to 0 so backpressure can never deadlock.
  gpuWorker.onmessage({ data: { error: 'simulated device lost' } });
  assert.equal(coordContext.PlayGta6Loader.getLoaderStatus().inFlightGpuPresents, 0);

  pthreadWorker.terminate();
});
