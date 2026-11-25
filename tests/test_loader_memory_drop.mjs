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

const loaderMod = require('../web/loader.js');
const preloaderMod = require('../web/runtime/pgim_preloader.js');
const orchestratorMod = require('../web/runtime/boot_orchestrator.js');

const {
  PTHREAD_CMD_LOAD,
  PTHREAD_CMD_LOADED,
  getLoaderStatus,
  getWasmMemoryTelemetry,
  resetWasmMemoryTelemetry,
  collectTransferableArrayBuffers,
  releaseWireBytesFromObject,
  stripWireBytesFromSharedModules,
  compileWasmModuleStreaming,
  createSharedWasmModuleCache,
  createWorkerBootInitPayloadsWithTransfers,
  buildRuntimeModuleConfig,
  createPthreadPoolCoordinator,
} = loaderMod;

const {
  PgimPreloader,
  collectTransferableBuffers: preloaderCollectTransferables,
  releaseBootBundleBytes: preloaderReleaseBundle,
} = preloaderMod;

const {
  BootOrchestrator,
  collectTransferableBuffers: orchestratorCollectTransferables,
  transferPreloadPayloadToWorker,
  releaseBootBundleBytes: orchestratorReleaseBundle,
} = orchestratorMod;

// Minimal valid WebAssembly binary (empty module: magic + version 1 = 8 bytes).
function makeMinimalWasmBytes(extraCustomPayloadBytes = 0) {
  if (extraCustomPayloadBytes <= 0) {
    return new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
  }
  // Custom section (id=0): name="x" (len=1, 0x78) + zero payload bytes (valid LEB128 for < 127 bytes).
  const payloadLen = Math.min(100, extraCustomPayloadBytes);
  const secLen = 2 + payloadLen;
  const out = new Uint8Array(8 + 2 + secLen);
  out.set([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x00, secLen, 0x01, 0x78], 0);
  return out;
}

// Minimal valid Wasm module importing env.memory and exporting nothing.
function makeMemoryImportWasmBytes() {
  const sec = (id, payload) => [id, payload.length, ...payload];
  const importSec = sec(0x02, [
    0x01,
    0x03, 0x65, 0x6e, 0x76, 0x06, 0x6d, 0x65, 0x6d, 0x6f, 0x72, 0x79, 0x02, 0x01, 0x01, 0x01,
  ]);
  return new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, ...importSec]);
}

function buildValidPgimBytes(payloadSize = 4096) {
  const headerBytes = 64;
  const sectionBytes = 64;
  const payloadOffset = 4096;
  const totalBytes = payloadOffset + payloadSize;
  const buf = new ArrayBuffer(totalBytes);
  const bytes = new Uint8Array(buf);
  const view = new DataView(buf);
  // Magic "PGIM"
  bytes[0] = 0x50;
  bytes[1] = 0x47;
  bytes[2] = 0x49;
  bytes[3] = 0x4d;
  view.setUint32(4, 1, true); // version
  view.setUint32(8, headerBytes, true);
  view.setUint32(12, sectionBytes, true);
  view.setUint32(16, 0, true); // flags
  view.setUint32(20, 4096, true); // page_size
  view.setBigUint64(24, 0x140000000n, true); // image_base
  view.setUint32(32, 0x2000, true); // size_of_image
  view.setUint32(36, 0x1000, true); // section_alignment
  view.setUint32(40, 0x1000, true); // entry_point_rva
  view.setUint32(44, 1, true); // section_count
  view.setUint32(48, 0, true); // dir64_count
  view.setUint32(52, 0, true); // highlow_count
  view.setBigUint64(56, BigInt(totalBytes), true); // file_size
  // Section 0: ".data"
  bytes.set(new TextEncoder().encode('.data'), headerBytes);
  view.setUint32(headerBytes + 16, 0x1000, true); // rva
  view.setUint32(headerBytes + 20, payloadSize, true); // virtual_size
  view.setUint32(headerBytes + 24, 0b011, true); // prot RW
  view.setUint32(headerBytes + 28, 0, true); // flags
  view.setBigUint64(headerBytes + 32, BigInt(payloadOffset), true);
  view.setBigUint64(headerBytes + 40, BigInt(payloadSize), true);
  view.setBigUint64(headerBytes + 48, 0n, true);
  view.setUint32(headerBytes + 56, 0, true);
  view.setUint32(headerBytes + 60, 0, true);
  return bytes;
}

test('W351 loader compileWasmModuleStreaming prefers compileStreaming without materializing ArrayBuffer', async () => {
  resetWasmMemoryTelemetry();
  const wasmBytes = makeMinimalWasmBytes(32);
  let arrayBufferCalls = 0;
  let compileStreamingCalls = 0;

  const mockWasmApi = {
    Module: WebAssembly.Module,
    compile: WebAssembly.compile.bind(WebAssembly),
    compileStreaming: async (respPromise) => {
      compileStreamingCalls += 1;
      const resp = await respPromise;
      assert.equal(resp.headers.get('content-type'), 'application/wasm');
      return WebAssembly.compile(wasmBytes);
    },
  };

  const resp = new Response(wasmBytes, {
    status: 200,
    headers: { 'content-type': 'application/wasm' },
  });
  resp.arrayBuffer = async () => {
    arrayBufferCalls += 1;
    throw new Error('arrayBuffer() must not be called when compileStreaming succeeds');
  };

  const res = await compileWasmModuleStreaming('http://127.0.0.1:8080/dist/guest_part_0.wasm', {
    wasmApi: mockWasmApi,
    fetchFn: async () => resp,
  });

  assert.equal(res.streaming, true);
  assert.equal(res.wireBytesReleased, 0);
  assert.ok(res.module instanceof WebAssembly.Module);
  assert.equal(compileStreamingCalls, 1);
  assert.equal(arrayBufferCalls, 0);

  const tel = getWasmMemoryTelemetry();
  assert.equal(tel.streamingCompiles, 1);
  assert.equal(tel.fallbackCompiles, 0);
  assert.equal(tel.droppedWasmBuffers, 0);
});

test('W351 loader compileWasmModuleStreaming falls back to arrayBuffer + compile and drops wire bytes immediately', async () => {
  resetWasmMemoryTelemetry();
  const wasmBytes = makeMinimalWasmBytes(48);

  // 1. Non-wasm MIME type triggers fallback to arrayBuffer() + WebAssembly.compile + wire drop.
  const fallbackRes = await compileWasmModuleStreaming('http://127.0.0.1:8080/dist/guest_part_1.wasm', {
    fetchFn: async () => ({
      ok: true,
      status: 200,
      headers: { get: () => 'application/octet-stream' },
      arrayBuffer: async () => wasmBytes.buffer.slice(0),
    }),
  });
  assert.equal(fallbackRes.streaming, false);
  assert.ok(fallbackRes.module instanceof WebAssembly.Module);
  assert.equal(fallbackRes.wireBytesReleased, wasmBytes.byteLength);

  // 2. Direct Map-owned wire buffer is compiled and deleted from the owner Map immediately.
  const guestModuleBytes = new Map();
  guestModuleBytes.set('guest_part_0.wasm', wasmBytes.buffer.slice(0));
  const directRes = await compileWasmModuleStreaming(guestModuleBytes.get('guest_part_0.wasm'), {
    owner: guestModuleBytes,
    ownerKey: 'guest_part_0.wasm',
  });
  assert.ok(directRes.module instanceof WebAssembly.Module);
  assert.equal(guestModuleBytes.size, 0, 'owner Map entry must be deleted after compile');

  const tel = getWasmMemoryTelemetry();
  assert.equal(tel.fallbackCompiles, 2);
  assert.equal(tel.droppedWasmBuffers, 2);
  assert.equal(tel.wireBytesReleased, wasmBytes.byteLength * 2);
});

test('W351 loader createSharedWasmModuleCache compiles each side module once, drops wire bytes, and shares WebAssembly.Module across pthreads', async () => {
  resetWasmMemoryTelemetry();
  const part0Bytes = makeMinimalWasmBytes(24);
  const part1Bytes = makeMinimalWasmBytes(40);
  let compileCount = 0;

  const countingWasmApi = {
    Module: WebAssembly.Module,
    compile: async (bytes) => {
      compileCount += 1;
      return WebAssembly.compile(bytes);
    },
  };

  const cache = createSharedWasmModuleCache({ wasmApi: countingWasmApi });
  const preloadedMap = new Map([
    ['guest_part_0.wasm', part0Bytes.buffer.slice(0)],
    ['guest_part_1.wasm', new Uint8Array(part1Bytes)],
  ]);

  const compiled = await cache.ingestPreloadedBuffers(preloadedMap);
  assert.equal(compiled.length, 2);
  assert.equal(preloadedMap.size, 0, 'all raw wire bytes must be deleted from preloaded Map');
  assert.equal(compileCount, 2);

  // Requesting the same module again hits cache without recompiling and still drops caller wire bytes.
  const dupHolder = { bytes: part0Bytes.buffer.slice(0) };
  const cachedMod0 = await cache.compileAndShare('dist/guest_part_0.wasm', dupHolder.bytes, {
    owner: dupHolder,
    ownerKey: 'bytes',
  });
  assert.equal(cachedMod0, compiled[0]);
  assert.equal(compileCount, 2, 'side module must never be recompiled');
  assert.equal(dupHolder.bytes, null, 'duplicate wire bytes must be dropped on cache hit');

  // Wire into pthread pool coordinator and verify CMD_LOAD shares compiled WebAssembly.Module objects
  // across all pthreads while stripping any stray raw wire bytes.
  const coordinator = createPthreadPoolCoordinator({
    maxConcurrentPthreadLoads: 4,
    getSharedModules: () => cache.getSharedModulesArray(),
  });

  const makeMockWorker = () => {
    const received = [];
    return {
      received,
      postMessage(msg, transfer) {
        received.push({ msg, transfer });
      },
      terminate() {},
    };
  };

  const w1 = coordinator.registerWorker(makeMockWorker(), { name: 'em-pthread-1' });
  const w2 = coordinator.registerWorker(makeMockWorker(), { name: 'em-pthread-2' });

  const mainMod = await WebAssembly.compile(makeMinimalWasmBytes(16));
  const strayWire = new ArrayBuffer(256);
  const loadMsg1 = {
    cmd: PTHREAD_CMD_LOAD,
    wasmModule: mainMod,
    wasmBinary: strayWire,
  };
  const loadMsg2 = {
    cmd: PTHREAD_CMD_LOAD,
    wasmModule: mainMod,
  };

  w1.postMessage(loadMsg1);
  w2.postMessage(loadMsg2);

  assert.equal(loadMsg1.wasmBinary, null, 'stray wasmBinary on CMD_LOAD must be stripped when wasmModule is compiled');
  assert.equal(w1.received[0].msg.sharedModules.length, 2);
  assert.equal(w2.received[0].msg.sharedModules.length, 2);
  assert.equal(w1.received[0].msg.sharedModules[0], compiled[0]);
  assert.equal(w2.received[0].msg.sharedModules[0], compiled[0]);
  assert.equal(w1.received[0].msg.sharedModules[1], compiled[1]);
  assert.equal(w2.received[0].msg.sharedModules[1], compiled[1]);

  coordinator.notifyWorkerLoaded(w1);
  coordinator.notifyWorkerLoaded(w2);

  // stripWireBytesFromSharedModules strips raw bytes attached to hybrid entries while preserving Module.
  const hybridShared = [
    { name: 'guest_part_0.wasm', module: compiled[0], bytes: new Uint8Array(128), buffer: new ArrayBuffer(128) },
  ];
  const stripped = stripWireBytesFromSharedModules(hybridShared);
  assert.equal(hybridShared[0].module, compiled[0]);
  assert.equal(hybridShared[0].bytes, null);
  assert.equal(hybridShared[0].buffer, null);
  assert.equal(stripped.releasedBytes, 256);
  assert.ok(stripped.droppedWasmBuffers >= 2);
});

test('W351 loader instantiateWasm & onRuntimeInitialized drop raw wasmBinary wire bytes and update telemetry', async () => {
  resetWasmMemoryTelemetry();
  const wasmBytes = makeMemoryImportWasmBytes();
  const cfg = buildRuntimeModuleConfig({
    runtimeScriptUrl: 'http://127.0.0.1:8080/dist/runtime.js',
    wasmBinary: wasmBytes.buffer.slice(0),
    fetchFn: async () => {
      throw new Error('fetch should not be called when wasmBinary is provided');
    },
  });

  const memory = new WebAssembly.Memory({ initial: 1, maximum: 1 });
  const received = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('instantiateWasm timed out')), 2000);
    cfg.instantiateWasm({ env: { memory } }, (instance, mod) => {
      clearTimeout(timer);
      resolve({ instance, mod });
    });
  });
  assert.ok(received.instance instanceof WebAssembly.Instance);
  assert.ok(received.mod instanceof WebAssembly.Module);

  // Give .finally() microtask a tick to flush telemetry.
  await Promise.resolve();
  const afterInstantiate = getWasmMemoryTelemetry();
  assert.equal(afterInstantiate.droppedWasmBuffers, 1);
  assert.equal(afterInstantiate.wireBytesReleased, wasmBytes.byteLength);

  // Calling onRuntimeInitialized drops Emscripten's own `Module.wasmBinary` wire bytes too.
  const fakeRuntimeMod = { wasmBinary: new Uint8Array(512) };
  cfg.onRuntimeInitialized.call(fakeRuntimeMod);
  assert.equal(fakeRuntimeMod.wasmBinary, null);

  const afterInit = getWasmMemoryTelemetry();
  assert.equal(afterInit.droppedWasmBuffers, 2);
  assert.equal(afterInit.wireBytesReleased, wasmBytes.byteLength + 512);
  assert.equal(getLoaderStatus().wireBytesReleased, wasmBytes.byteLength + 512);
  assert.equal(getLoaderStatus().droppedWasmBuffers, 2);
});

test('W351 loader createWorkerBootInitPayloadsWithTransfers and VM self.onmessage transfer non-SAB buffers zero-copy and drop wire bytes', async () => {
  const pgimBuf = new ArrayBuffer(1024);
  const shaderBuf = new ArrayBuffer(2048);
  const sab = new SharedArrayBuffer(512);
  const fakeCanvas = { id: 'offscreen-canvas' };

  const payloads = createWorkerBootInitPayloadsWithTransfers({
    canvas: fakeCanvas,
    width: 1280,
    height: 720,
    dataBase: '/data/',
    pgimBuffer: pgimBuf,
    shaderPackBuffer: shaderBuf,
  });

  assert.equal(payloads.ioInit.pgimBuffer, pgimBuf);
  assert.deepEqual(payloads.ioTransferList, [pgimBuf]);
  assert.equal(payloads.gpuInit.shaderPackBuffer, shaderBuf);
  assert.deepEqual(payloads.gpuTransferList, [fakeCanvas, shaderBuf]);
  assert.equal(payloads.transferredArrayBufferCount, 2);

  // SharedArrayBuffer must never be placed in transferList.
  const sabPayloads = createWorkerBootInitPayloadsWithTransfers({
    canvas: fakeCanvas,
    dataBase: '/data/',
    pgimBuffer: sab,
    shaderPackBuffer: sab,
  });
  assert.equal(sabPayloads.ioInit.pgimBuffer, sab);
  assert.deepEqual(sabPayloads.ioTransferList, []);
  assert.deepEqual(sabPayloads.gpuTransferList, [fakeCanvas]);
  assert.equal(sabPayloads.transferredArrayBufferCount, 0);

  // Full VM sandbox test for loader.js self.onmessage + tryBootWasmRuntime + request-loader-status.
  const loaderPath = path.join(repoRoot, 'web/loader.js');
  const loaderSource = fs.readFileSync(loaderPath, 'utf8');
  const mainMessages = [];
  const workersByScript = new Map();
  let capturedRuntimeConfig = null;

  class MockBroadcastChannel {
    postMessage() {}
    close() {}
  }

  class MockWorker {
    constructor(script) {
      this.script = script;
      this.onmessage = null;
      this.posted = [];
      workersByScript.set(script, this);
    }
    postMessage(msg, transfer) {
      this.posted.push({ msg, transfer: transfer || [] });
      if (msg && msg.init) {
        queueMicrotask(() => {
          if (typeof this.onmessage === 'function') {
            this.onmessage({ data: { ready: true, summary: 'ok', adapter: 'mock-gpu' } });
          }
        });
      }
    }
  }

  const workerSelf = {
    name: '',
    location: { href: 'http://127.0.0.1:8080/loader.js' },
    Worker: MockWorker,
    onmessage: null,
    postMessage(msg) {
      mainMessages.push(msg);
    },
  };

  const part0Wire = makeMinimalWasmBytes(36).buffer;
  const runtimeWire = makeMemoryImportWasmBytes().buffer;

  const sandbox = {
    self: workerSelf,
    globalThis: null,
    URL,
    BroadcastChannel: MockBroadcastChannel,
    Worker: MockWorker,
    WebAssembly,
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
    TypeError,
    RangeError,
    Number,
    Math,
    Date,
    Object,
    Array,
    Reflect,
    Proxy,
    setTimeout,
    clearTimeout,
    queueMicrotask,
    fetch: async (url, init) => {
      if (init && init.method === 'HEAD') {
        return { ok: true, status: 200 };
      }
      return { ok: false, status: 404 };
    },
    importScripts(scriptPath) {
      if (scriptPath === './dist/runtime.js') {
        workerSelf.createRuntimeModule = async (cfg) => {
          capturedRuntimeConfig = cfg;
          const memory = new WebAssembly.Memory({ initial: 1, maximum: 1 });
          await new Promise((resolve) => {
            cfg.instantiateWasm({ env: { memory } }, () => resolve());
          });
          return { wasmMemory: memory };
        };
        return;
      }
      const resolved = path.resolve(repoRoot, 'web', scriptPath);
      vm.runInContext(fs.readFileSync(resolved, 'utf8'), context, { filename: resolved });
    },
  };
  sandbox.globalThis = sandbox;
  const context = vm.createContext(sandbox);
  vm.runInContext(loaderSource, context, { filename: loaderPath });

  const guestMap = new Map([['guest_part_0.wasm', part0Wire]]);
  const incomingPgim = new ArrayBuffer(512);
  const incomingShader = new ArrayBuffer(1024);
  const initMsg = {
    canvas: { id: 'canvas-1' },
    width: 640,
    height: 360,
    dataBase: '/data/',
    pgimBuffer: incomingPgim,
    shaderPackBuffer: incomingShader,
    wasmBinary: runtimeWire,
    guestModuleBytes: guestMap,
  };

  workerSelf.onmessage({ data: initMsg });

  // Immediately after onmessage: raw binary references on initMsg are nulled,
  // and PGIM/shaderPack buffers were passed via zero-copy transfer lists.
  assert.equal(initMsg.pgimBuffer, null);
  assert.equal(initMsg.shaderPackBuffer, null);
  assert.equal(initMsg.wasmBinary, null);
  assert.equal(initMsg.guestModuleBytes, null);

  const ioWorker = workersByScript.get('./workers/io_worker.js');
  const gpuWorker = workersByScript.get('./workers/wgpu_worker.js');
  assert.ok(ioWorker.posted[0].transfer.includes(incomingPgim));
  assert.ok(gpuWorker.posted[0].transfer.includes(incomingShader));

  // Wait for async worker readiness + tryBootWasmRuntime() compilation to settle.
  await new Promise((resolve) => setTimeout(resolve, 40));

  assert.ok(capturedRuntimeConfig, 'createRuntimeModule should have been invoked');
  assert.equal(guestMap.size, 0, 'guestModuleBytes Map must be cleared after compilation');
  assert.equal(capturedRuntimeConfig.sharedModules.length, 1);
  assert.ok(capturedRuntimeConfig.sharedModules[0] instanceof WebAssembly.Module);

  workerSelf.onmessage({ data: { type: 'request-loader-status', id: 99 } });
  const statusMsg = mainMessages.find((m) => m.type === 'loader-status' && m.id === 99);
  assert.ok(statusMsg);
  assert.equal(statusMsg.sharedModuleCount, 1);
  assert.equal(statusMsg.transferredArrayBuffers, 2);
  assert.equal(statusMsg.droppedWasmBuffers, 2);
  assert.equal(statusMsg.wireBytesReleased, part0Wire.byteLength + runtimeWire.byteLength);
});

test('W351 PgimPreloader compileWasmStreaming prefers streaming, falls back cleanly, and drops wire bytes', async () => {
  const wasmBytes = makeMinimalWasmBytes(44);
  let streamingCalls = 0;

  const preloader = new PgimPreloader({
    fetchImpl: async (url) => {
      if (url.endsWith('guest_part_0.wasm')) {
        const resp = new Response(wasmBytes, {
          status: 200,
          headers: { 'content-type': 'application/wasm', 'content-length': String(wasmBytes.byteLength) },
        });
        resp.arrayBuffer = async () => {
          throw new Error('should not materialize arrayBuffer on streaming path');
        };
        return resp;
      }
      return {
        ok: true,
        status: 200,
        headers: { get: (k) => (k.toLowerCase() === 'content-type' ? 'application/octet-stream' : null) },
        arrayBuffer: async () => wasmBytes.buffer.slice(0),
      };
    },
  });

  const mockWasmApi = {
    Module: WebAssembly.Module,
    compile: WebAssembly.compile.bind(WebAssembly),
    compileStreaming: async () => {
      streamingCalls += 1;
      return WebAssembly.compile(wasmBytes);
    },
  };

  // 1. Streaming compile path.
  const streamRes = await preloader.compileWasmStreaming('/dist/guest_part_0.wasm', { wasmApi: mockWasmApi });
  assert.equal(streamRes.streaming, true);
  assert.ok(streamRes.module instanceof WebAssembly.Module);
  assert.equal('bytes' in streamRes, false);
  assert.equal('buffer' in streamRes, false);
  assert.equal(streamingCalls, 1);

  // 2. Fallback compile path releases wire bytes and records telemetry.
  const fallbackRes = await preloader.compileWasmStreaming('/dist/guest_part_1.wasm', { wasmApi: mockWasmApi });
  assert.equal(fallbackRes.streaming, false);
  assert.ok(fallbackRes.module instanceof WebAssembly.Module);
  assert.equal(fallbackRes.wireBytesReleased, wasmBytes.byteLength);
  assert.equal('bytes' in fallbackRes, false);
  assert.equal('buffer' in fallbackRes, false);

  const memTel = preloader.getMemoryDropTelemetry();
  assert.equal(memTel.streamingCompiles, 1);
  assert.equal(memTel.fallbackCompiles, 1);
  assert.equal(memTel.droppedWasmBuffers, 1);
  assert.equal(memTel.wireBytesReleased, wasmBytes.byteLength);
  assert.equal(preloader.getStats().wireBytesReleased, wasmBytes.byteLength);
  assert.equal(preloader.getStats().droppedWasmBuffers, 1);

  // 3. Full-body PGIM GET reuses backing buffer without double-copying when targetBuffer & SAB are off.
  const pgimBytes = buildValidPgimBytes(4096);
  const originalBuf = pgimBytes.buffer;
  const fullPreloader = new PgimPreloader({
    fetchImpl: async (url, init) => {
      if (init && init.method === 'HEAD') {
        return {
          ok: true,
          status: 200,
          headers: {
            get: (k) => {
              const lower = k.toLowerCase();
              if (lower === 'accept-ranges') return 'none';
              if (lower === 'content-length') return String(pgimBytes.byteLength);
              return null;
            },
          },
        };
      }
      return {
        ok: true,
        status: 200,
        headers: { get: (k) => (k.toLowerCase() === 'content-length' ? String(pgimBytes.byteLength) : null) },
        arrayBuffer: async () => originalBuf,
      };
    },
  });
  const pgimRes = await fullPreloader.preloadPgim('/boot/guest.pgim');
  assert.equal(pgimRes.buffer, originalBuf, 'full-body PGIM fetch should reuse ArrayBuffer without double copy');

  // 4. releaseBootBundleBytes drops guestModuleBytes Map & wasmBinary while preserving compiled WebAssembly.Module.
  const compiledMod = fallbackRes.module;
  const bundleWithWasm = {
    wasmBinary: new ArrayBuffer(128),
    guestModuleBytes: new Map([['guest_part_0.wasm', new ArrayBuffer(256)]]),
    runtimeWasm: { module: compiledMod, bytes: new Uint8Array(64), bytesLoaded: 64 },
    pgim: pgimRes,
  };
  const rel = preloaderReleaseBundle(bundleWithWasm);
  assert.deepEqual(Object.keys(rel).sort(), ['releasedBytes', 'releasedFields']);
  assert.equal(bundleWithWasm.wasmBinary, null);
  assert.equal(bundleWithWasm.guestModuleBytes.size, 0);
  assert.equal(bundleWithWasm.runtimeWasm.bytes, null);
  assert.equal(bundleWithWasm.runtimeWasm.module, compiledMod, 'compiled WebAssembly.Module must stay intact');
  assert.equal(bundleWithWasm.pgim.bytes, null);
  assert.equal(bundleWithWasm.pgim.buffer, null);
  assert.equal(rel.releasedBytes, 128 + 256 + 64 + pgimBytes.byteLength);
});

test('W351 BootOrchestrator transfers non-SAB ArrayBuffers zero-copy, keeps splashOverlay buffer-free, and tracks wireBytesReleased/droppedWasmBuffers', async () => {
  const compiledMod = await WebAssembly.compile(makeMinimalWasmBytes(16));
  const pgimBuf = new ArrayBuffer(1024);
  const wasmWireBuf = new ArrayBuffer(2048);
  const side0WireBuf = new ArrayBuffer(4096);
  const sharedSab = new SharedArrayBuffer(512);

  // Verify collectTransferableBuffers deduplicates aliased buffers and excludes SAB & WebAssembly.Module.
  const aliasedPayload = {
    pgim: { buffer: pgimBuf, bytes: new Uint8Array(pgimBuf) },
    runtimeWasm: { module: compiledMod, wasmBinary: wasmWireBuf },
    guestModuleBytes: new Map([['guest_part_0.wasm', side0WireBuf]]),
    inputSab: sharedSab,
  };
  const transferables = orchestratorCollectTransferables(aliasedPayload);
  assert.equal(transferables.length, 3);
  assert.ok(transferables.includes(pgimBuf));
  assert.ok(transferables.includes(wasmWireBuf));
  assert.ok(transferables.includes(side0WireBuf));
  assert.equal(transferables.includes(sharedSab), false);
  assert.deepEqual(preloaderCollectTransferables(aliasedPayload), transferables);
  assert.deepEqual(collectTransferableArrayBuffers(aliasedPayload), transferables);

  // Verify transferPreloadPayloadToWorker transfers buffers and nulls source references.
  const posted = [];
  const mockWorker = {
    postMessage(msg, transfer) {
      posted.push({ msg, transfer: transfer || [] });
    },
  };
  const xferRes = transferPreloadPayloadToWorker(mockWorker, aliasedPayload, {
    extraTransferables: [{ id: 'canvas' }],
  });
  assert.equal(posted.length, 1);
  assert.equal(posted[0].transfer.length, 4);
  assert.equal(xferRes.transferredBuffers, 3);
  assert.equal(xferRes.transferredBytes, 1024 + 2048 + 4096);
  assert.equal(aliasedPayload.pgim.buffer, null);
  assert.equal(aliasedPayload.pgim.bytes, null);
  assert.equal(aliasedPayload.runtimeWasm.wasmBinary, null);
  assert.equal(aliasedPayload.runtimeWasm.module, compiledMod);
  assert.equal(aliasedPayload.guestModuleBytes.size, 0);

  // End-to-end BootOrchestrator: overlay never receives raw binary bundle, and telemetry tracks drops.
  const overlayUpdates = [];
  const overlay = {
    init() {},
    update(payload) {
      overlayUpdates.push(payload);
    },
    dismiss() {},
    destroy() {},
  };

  const orchPgimBuf = new ArrayBuffer(512);
  const orchWasmWire = new ArrayBuffer(1536);
  const orchSideMap = new Map([['guest_part_0.wasm', new ArrayBuffer(3072)]]);
  const preloader = {
    preloadBootBundle: async () => ({
      pgim: {
        header: { magic: 'PGIM' },
        bytes: new Uint8Array(orchPgimBuf),
        buffer: orchPgimBuf,
        bytesLoaded: 512,
      },
      shaderPackBytes: new Uint8Array(256),
      reflectionManifest: { schema_version: 1 },
      guestModuleBytes: orchSideMap,
      bytesLoaded: 512 + 256 + 3072,
    }),
  };

  const orchestrator = new BootOrchestrator({
    fetchImpl: async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        ok: true,
        hasWasmBinary: true,
        pgimUrl: '/boot/guest.pgim',
        shaderPackUrl: '/shaders/all.wgsl.pack',
        runtimeWasm: '/dist/runtime.wasm',
      }),
    }),
    splashOverlay: overlay,
    preloader,
    preloadRuntimeWasm: async () => ({
      module: compiledMod,
      wasmBinary: orchWasmWire,
      bytesLoaded: 1536,
    }),
    onLaunchRuntime: async () => {},
  });

  const bootRes = await orchestrator.start();
  assert.equal(bootRes.ok, true);

  // Splash overlay must never have been handed a non-null binary bundle.
  for (const upd of overlayUpdates) {
    assert.equal(upd.bundle ?? null, null, 'splashOverlay must never receive raw binary bundle');
  }

  // Internal bundle dropped, all binary buffers nulled, compiled module preserved.
  assert.equal(orchestrator.getState().bundle, null);
  assert.equal(bootRes.bundle.pgim.bytes, null);
  assert.equal(bootRes.bundle.pgim.buffer, null);
  assert.equal(bootRes.bundle.shaderPackBytes, null);
  assert.equal(bootRes.bundle.runtimeWasm.wasmBinary, null);
  assert.equal(bootRes.bundle.runtimeWasm.module, compiledMod);
  assert.equal(orchSideMap.size, 0);

  // Sanitized telemetry counters are populated on getStats(), getDiagnosticSnapshot(), and getMemoryDropTelemetry().
  const expectedReleased = 512 + 256 + 1536 + 3072;
  const stats = orchestrator.getStats();
  const diag = orchestrator.getDiagnosticSnapshot();
  const mem = orchestrator.getMemoryDropTelemetry();
  assert.equal(stats.wireBytesReleased, expectedReleased);
  assert.equal(diag.wireBytesReleased, expectedReleased);
  assert.equal(mem.wireBytesReleased, expectedReleased);
  assert.equal(stats.droppedWasmBuffers, 2);
  assert.equal(diag.droppedWasmBuffers, 2);
  assert.equal(mem.droppedWasmBuffers, 2);
  assert.equal(mem.bundleRetained, false);

  // Direct releaseBootBundleBytes returns exact { releasedFields, releasedBytes } shape.
  const emptyRel = orchestratorReleaseBundle(null);
  assert.deepEqual(emptyRel, { releasedFields: [], releasedBytes: 0 });
  const loaderRel = releaseWireBytesFromObject(null);
  assert.deepEqual(loaderRel, {
    releasedFields: [],
    releasedBytes: 0,
    droppedWasmBuffers: 0,
    droppedPreloadBuffers: 0,
  });
});
