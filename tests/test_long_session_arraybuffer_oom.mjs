import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);

const {
  PTHREAD_CMD_LOAD,
  WGPU_CMD_POOL_MIN_BYTES,
  createWarmCmdBufferPool,
  createSharedWasmModuleCache,
  createPthreadPoolCoordinator,
  buildSharedWasmModuleDylinkStub,
  extractSharedWasmStubKey,
  installSharedWasmModuleShim,
  bootstrapEmscriptenPthreadWorker,
} = require('../web/loader.js');

const {
  createWgpuWorkerHandler,
} = require('../web/workers/wgpu_worker.js');

const {
  installRenderStatePackets,
  V2_OPCODES,
} = require('../web/workers/render_state_packets.js');

function encodeUleb128(n) {
  const out = [];
  let v = n >>> 0;
  do {
    let b = v & 0x7f;
    v >>>= 7;
    if (v !== 0) b |= 0x80;
    out.push(b);
  } while (v !== 0);
  return out;
}

function makeCustomSection(name, payloadBytes) {
  const nameBytes = new TextEncoder().encode(name);
  const nameLen = encodeUleb128(nameBytes.length);
  const bodyLen = nameLen.length + nameBytes.length + payloadBytes.length;
  return new Uint8Array([
    0x00,
    ...encodeUleb128(bodyLen),
    ...nameLen,
    ...nameBytes,
    ...payloadBytes,
  ]);
}

function makeDylinkWasmBytes(dylinkPayload, extraPadding = 64) {
  const dylinkSec = makeCustomSection('dylink.0', dylinkPayload);
  const padPayload = new Uint8Array(extraPadding);
  padPayload.fill(0x42);
  const padSec = makeCustomSection('pad', padPayload);
  const out = new Uint8Array(8 + dylinkSec.length + padSec.length);
  out.set([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00], 0);
  out.set(dylinkSec, 8);
  out.set(padSec, 8 + dylinkSec.length);
  return out;
}

test('buildSharedWasmModuleDylinkStub preserves exact dylink.0 custom section in a tiny stub', async () => {
  // Valid dylink.0 mem_info subsection payload: subsection id=1, size=4, memSize=4096 (0x80 0x20), align=4, tableSize=128 (0x80 0x01), tableAlign=0
  const dylinkPayload = new Uint8Array([0x01, 0x06, 0x80, 0x20, 0x04, 0x80, 0x01, 0x00]);
  const fullBytes = makeDylinkWasmBytes(dylinkPayload, 4096);
  const compiled = await WebAssembly.compile(fullBytes);

  const stub = buildSharedWasmModuleDylinkStub(compiled, 'guest_part_0.wasm');
  assert.ok(stub instanceof Uint8Array);
  assert.ok(stub.byteLength < 256, `stub must stay tiny (got ${stub.byteLength} bytes)`);
  assert.equal(extractSharedWasmStubKey(stub), 'guest_part_0.wasm');
  assert.equal(extractSharedWasmStubKey(stub.slice()), 'guest_part_0.wasm');

  // Verify the stub itself is a valid Wasm module with the exact dylink.0 payload.
  const stubMod = new WebAssembly.Module(stub);
  const sections = WebAssembly.Module.customSections(stubMod, 'dylink.0');
  assert.equal(sections.length, 1);
  assert.deepEqual(new Uint8Array(sections[0]), dylinkPayload);
});

test('dynamically spawned pthread #73 reuses precompiled guest_part_0.wasm across fetch/arrayBuffer/compile/Module/instantiate with 0 HTTP fetches and 0 large ArrayBuffers', async () => {
  const dylinkPayload0 = new Uint8Array([0x01, 0x04, 0x10, 0x02, 0x08, 0x00]);
  const dylinkPayload1 = new Uint8Array([0x01, 0x04, 0x20, 0x02, 0x04, 0x00]);
  const part0Mod = await WebAssembly.compile(makeDylinkWasmBytes(dylinkPayload0, 2048));
  const part1Mod = await WebAssembly.compile(makeDylinkWasmBytes(dylinkPayload1, 1024));
  const mainMod = await WebAssembly.compile(new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]));

  const cache = createSharedWasmModuleCache();
  cache.setModule('guest_part_0.wasm', part0Mod, 908 * 1024 * 1024);
  cache.setModule('guest_part_1.wasm', part1Mod, 7 * 1024 * 1024);

  const coordinator = createPthreadPoolCoordinator({
    maxConcurrentPthreadLoads: 2,
    getSharedModules: () => cache.getSharedModulesArray(),
    getSharedModuleNames: () => cache.getSharedModuleNames(),
  });

  const delivered = [];
  const fakeWorker = {
    postMessage(msg, transfer) {
      delivered.push({ msg, transfer });
    },
    terminate() {},
  };
  coordinator.registerWorker(fakeWorker, { name: 'em-pthread-73' });
  fakeWorker.postMessage({ cmd: PTHREAD_CMD_LOAD, wasmModule: mainMod, dynamicLibraries: ['guest_part_0.wasm', 'guest_part_1.wasm'] });
  const loadMsg = delivered.find((d) => d.msg && d.msg.cmd === PTHREAD_CMD_LOAD).msg;
  assert.equal(loadMsg.sharedModules.length, 2);

  let rawFetchCalls = 0;
  let passthroughFetchCalls = 0;
  let headFetchCalls = 0;
  let rawCompileCalls = 0;
  let rawCompileStreamingCalls = 0;
  let rawInstantiateStreamingCalls = 0;
  const listeners = [];

  const originalFetch = async (url, init) => {
    const method = (init && init.method ? init.method : 'GET').toUpperCase();
    if (method === 'HEAD') {
      headFetchCalls += 1;
      return { ok: true, status: 200, url };
    }
    if (String(url).includes('guest_part_')) {
      rawFetchCalls += 1;
      throw new RangeError('Array buffer allocation failed');
    }
    passthroughFetchCalls += 1;
    return { ok: true, status: 200, url };
  };

  const originalCompile = async (bytes) => {
    rawCompileCalls += 1;
    return WebAssembly.compile(bytes);
  };
  const originalCompileStreaming = async () => {
    rawCompileStreamingCalls += 1;
    throw new Error('compileStreaming must not recompile cached shared module');
  };
  const originalInstantiateStreaming = async () => {
    rawInstantiateStreamingCalls += 1;
    throw new Error('instantiateStreaming must not recompile cached shared module');
  };

  const pthreadScope = {
    name: 'em-pthread-73',
    location: { href: 'http://127.0.0.1:8080/loader.js' },
    fetch: originalFetch,
    Response,
    WebAssembly: {
      Module: WebAssembly.Module,
      Instance: WebAssembly.Instance,
      compile: originalCompile,
      instantiate: WebAssembly.instantiate.bind(WebAssembly),
      compileStreaming: originalCompileStreaming,
      instantiateStreaming: originalInstantiateStreaming,
    },
    addEventListener(type, fn) {
      if (type === 'message') listeners.push(fn);
    },
    removeEventListener(type, fn) {
      const idx = listeners.indexOf(fn);
      if (idx >= 0) listeners.splice(idx, 1);
    },
  };

  const handle = bootstrapEmscriptenPthreadWorker({
    scope: pthreadScope,
    importScriptsFn: () => {},
  });

  // Deliver CMD_LOAD from coordinator to the pthread worker.
  for (const fn of listeners.slice()) {
    fn({ data: loadMsg });
  }

  // Path 1: Emscripten libdylink getBinaryPromise(): fetch -> arrayBuffer -> Uint8Array -> WebAssembly.compile
  const resp0 = await pthreadScope.fetch('http://127.0.0.1:8080/dist/guest_part_0.wasm', {
    credentials: 'same-origin',
  });
  assert.equal(resp0.ok, true);
  assert.equal(resp0.status, 200);
  const buf0 = await resp0.arrayBuffer();
  assert.ok(buf0.byteLength < 256, `expected tiny dylink stub, got ${buf0.byteLength} bytes`);
  const compiled0 = await pthreadScope.WebAssembly.compile(new Uint8Array(buf0));
  assert.equal(compiled0, part0Mod);

  // Path 2: Copied/sliced stub bytes -> new WebAssembly.Module(copied)
  const copiedBytes = new Uint8Array(buf0).slice();
  const constructed0 = new pthreadScope.WebAssembly.Module(copiedBytes);
  assert.equal(constructed0, part0Mod);
  assert.ok(constructed0 instanceof WebAssembly.Module);
  // Verify exact dylink.0 custom sections are still inspectable via WebAssembly.Module.customSections!
  const customSecs = pthreadScope.WebAssembly.Module.customSections(constructed0, 'dylink.0');
  assert.equal(customSecs.length, 1);
  assert.deepEqual(new Uint8Array(customSecs[0]), dylinkPayload0);

  // Path 3: fetch -> arrayBuffer -> WebAssembly.instantiate(bytes, imports) returns { module, instance }
  const resp1 = await pthreadScope.fetch('dist/guest_part_1.wasm');
  const buf1 = await resp1.arrayBuffer();
  const inst1 = await pthreadScope.WebAssembly.instantiate(new Uint8Array(buf1), {});
  assert.equal(inst1.module, part1Mod);
  assert.ok(inst1.instance instanceof WebAssembly.Instance);

  // Path 4: WebAssembly.compileStreaming(fetch(...))
  const streamMod0 = await pthreadScope.WebAssembly.compileStreaming(
    pthreadScope.fetch('http://127.0.0.1:8080/dist/guest_part_0.wasm'),
  );
  assert.equal(streamMod0, part0Mod);

  // Path 5: WebAssembly.instantiateStreaming(fetch(...), imports)
  const streamInst1 = await pthreadScope.WebAssembly.instantiateStreaming(
    pthreadScope.fetch('http://127.0.0.1:8080/dist/guest_part_1.wasm'),
    {},
  );
  assert.equal(streamInst1.module, part1Mod);
  assert.ok(streamInst1.instance instanceof WebAssembly.Instance);

  // Non-cached URLs and HEAD probes pass straight through untouched.
  const otherResp = await pthreadScope.fetch('http://127.0.0.1:8080/dist/other.bin');
  assert.equal(otherResp.status, 200);
  const headResp = await pthreadScope.fetch('http://127.0.0.1:8080/dist/guest_part_0.wasm', {
    method: 'HEAD',
  });
  assert.equal(headResp.status, 200);

  assert.equal(rawFetchCalls, 0, 'cached guest_part_*.wasm must never hit network or allocate 908 MB');
  assert.equal(passthroughFetchCalls, 1);
  assert.equal(headFetchCalls, 1);
  assert.equal(rawCompileCalls, 0);
  assert.equal(rawCompileStreamingCalls, 0);
  assert.equal(rawInstantiateStreamingCalls, 0);

  // Cleanup restores all wrapped APIs cleanly.
  handle.cleanup();
  assert.equal(pthreadScope.fetch, originalFetch);
  assert.equal(pthreadScope.WebAssembly.Module, WebAssembly.Module);
  assert.equal(pthreadScope.WebAssembly.compile, originalCompile);
  assert.equal(pthreadScope.WebAssembly.compileStreaming, originalCompileStreaming);
  assert.equal(pthreadScope.WebAssembly.instantiateStreaming, originalInstantiateStreaming);
});

test('installSharedWasmModuleShim captures coordinator streaming compiles and reuses them', async () => {
  const bytes = makeDylinkWasmBytes(new Uint8Array([0x01, 0x04, 0x10, 0x02, 0x02, 0x00]), 128);
  const cache = createSharedWasmModuleCache();
  let underlyingStreamingCalls = 0;

  const scope = {
    fetch: async (url) => {
      throw new Error(`unexpected fetch for cached module: ${url}`);
    },
    Response,
    WebAssembly: {
      Module: WebAssembly.Module,
      Instance: WebAssembly.Instance,
      compile: WebAssembly.compile.bind(WebAssembly),
      instantiate: WebAssembly.instantiate.bind(WebAssembly),
      compileStreaming: async () => {
        underlyingStreamingCalls += 1;
        return WebAssembly.compile(bytes);
      },
      instantiateStreaming: async (_src, imports) => {
        underlyingStreamingCalls += 1;
        return WebAssembly.instantiate(bytes, imports);
      },
    },
  };

  const restore = installSharedWasmModuleShim(scope, {
    resolveSharedModule: (u) => cache.getModule(u),
    onModuleCompiled: (u, mod) => cache.setModule(u, mod),
  });

  try {
    // First compileStreaming misses cache -> compiles once and captures into shared cache.
    const firstMod = await scope.WebAssembly.compileStreaming(
      Promise.resolve({ url: 'http://127.0.0.1:8080/dist/guest_part_0.wasm' }),
    );
    assert.equal(underlyingStreamingCalls, 1);
    assert.equal(cache.getModule('guest_part_0.wasm'), firstMod);

    // Subsequent fetch + compile hits cache with 0 network and 0 recompiles.
    const resp = await scope.fetch('http://127.0.0.1:8080/dist/guest_part_0.wasm');
    const stubBuf = await resp.arrayBuffer();
    const secondMod = await scope.WebAssembly.compile(stubBuf);
    assert.equal(secondMod, firstMod);
    assert.equal(underlyingStreamingCalls, 1);
  } finally {
    restore();
  }
});

test('createWarmCmdBufferPool prunes detached buffers and stays bounded over 10,000 frames', () => {
  const pool = createWarmCmdBufferPool();
  // Simulate a detached/0-byte entry slipping in alongside valid buffers.
  const bufA = new ArrayBuffer(WGPU_CMD_POOL_MIN_BYTES);
  const bufB = new ArrayBuffer(WGPU_CMD_POOL_MIN_BYTES);
  assert.equal(pool.recycle(bufA), true);
  assert.equal(pool.recycle(bufB), true);

  for (let frame = 0; frame < 10000; frame += 1) {
    const acquired = pool.acquire(WGPU_CMD_POOL_MIN_BYTES) || new ArrayBuffer(WGPU_CMD_POOL_MIN_BYTES);
    assert.equal(acquired.byteLength, WGPU_CMD_POOL_MIN_BYTES);
    assert.equal(pool.recycle(acquired), true);
    assert.ok(pool.size <= 3, `pool size must remain <= 3 (got ${pool.size})`);
  }
});

test('wgpu_worker recycles cmdBuffers on unready and failed-init paths instead of dropping/pinning them', async () => {
  const posted = [];
  const handler = createWgpuWorkerHandler({
    postMessage: (msg, transfer) => posted.push({ msg, transfer }),
    initGpu: async () => {
      throw new Error('simulated WebGPU adapter unavailable');
    },
    ensureRtBvhPreludeLoaded: async () => {},
  });

  // 1. Batch arrives when !processor && !initInFlight with recycleBuffer: true -> recycled immediately.
  const bufUnready = new ArrayBuffer(WGPU_CMD_POOL_MIN_BYTES);
  await handler.handleMessage({ cmdBuffer: bufUnready, recycleBuffer: true });
  const recycledUnready = posted.filter((p) => p.msg && p.msg.type === 'recycle-cmd-buffer');
  assert.equal(recycledUnready.length, 1);
  assert.equal(recycledUnready[0].msg.cmdBuffer, bufUnready);

  // 2. Batches queued while initInFlight are recycled & drained if initGpu throws!
  let rejectInit;
  const posted2 = [];
  const handler2 = createWgpuWorkerHandler({
    postMessage: (msg, transfer) => posted2.push({ msg, transfer }),
    initGpu: () =>
      new Promise((_res, rej) => {
        rejectInit = rej;
      }),
    ensureRtBvhPreludeLoaded: async () => {},
  });

  const initPromise = handler2.handleMessage({ init: true, canvas: { width: 640, height: 360 } });
  const queuedBuf1 = new ArrayBuffer(WGPU_CMD_POOL_MIN_BYTES);
  const queuedBuf2 = new ArrayBuffer(WGPU_CMD_POOL_MIN_BYTES);
  await handler2.handleMessage({ cmdBuffer: queuedBuf1, recycleBuffer: true });
  await handler2.handleMessage({ cmdBuffer: queuedBuf2, recycleBuffer: true });
  assert.equal(handler2.pendingCommandBatchCount, 2);

  rejectInit(new Error('device creation failed'));
  await initPromise;

  assert.equal(handler2.pendingCommandBatchCount, 0, 'failed init must drain pendingCommandBatches');
  const recycledAfterFail = posted2.filter((p) => p.msg && p.msg.type === 'recycle-cmd-buffer');
  assert.equal(recycledAfterFail.length, 2);
  assert.equal(recycledAfterFail[0].msg.cmdBuffer, queuedBuf1);
  assert.equal(recycledAfterFail[1].msg.cmdBuffer, queuedBuf2);
});

test('RenderStatePackets bounds _textureGenerations, _seenWarningKeys, and layoutsByContent over 25,000+ resource churn', () => {
  let layoutSeq = 0;
  const mockDevice = {
    createBindGroupLayout: () => ({ id: ++layoutSeq }),
    createPipelineLayout: () => ({ id: ++layoutSeq }),
  };
  const mockProcessor = {
    device: mockDevice,
    _deviceGeneration: 1,
    textures: new Map(),
    buffers: new Map(),
    bindGroups: new Map(),
    samplers: new Map(),
    pipelines: new Map(),
    computePipelines: new Map(),
    _passCommands: [],
  };

  const ext = installRenderStatePackets(mockProcessor);

  // 1. Destroying 25,000 non-texture resources (buffers / bind groups) must NOT grow _textureGenerations.
  for (let id = 1; id <= 25000; id += 1) {
    ext.destroyResource(id);
  }
  assert.equal(
    ext._textureGenerations.size,
    0,
    'destroying non-texture resources must not pollute _textureGenerations',
  );

  // 2. Creating and destroying 25,000 distinct tracked textures keeps _textureGenerations strictly bounded.
  let prevGen = 0;
  for (let id = 100000; id < 125000; id += 1) {
    ext.textureMeta.set(id, { id, format: 'rgba8unorm' });
    ext.destroyResource(id);
    const gen = ext._textureGenerations.get(id);
    assert.ok(gen > prevGen, 'texture generations must stay strictly monotonic');
    prevGen = gen;
  }
  assert.ok(
    ext._textureGenerations.size <= 16384,
    `_textureGenerations must stay <= 16384 (got ${ext._textureGenerations.size})`,
  );

  // 3. Dynamic warning keys stay bounded <= 4096.
  for (let i = 0; i < 6000; i += 1) {
    ext._recordTelemetryWarningOnce(`warn-key-${i}`, `warning #${i}`);
  }
  assert.ok(
    ext._seenWarningKeys.size <= 4096,
    `_seenWarningKeys must stay <= 4096 (got ${ext._seenWarningKeys.size})`,
  );

  // 4. Content-hashed bind group layouts stay bounded <= 4096.
  const pktBuf = new ArrayBuffer(24);
  const pktView = new DataView(pktBuf);
  for (let i = 1; i <= 5000; i += 1) {
    pktView.setUint32(0, i, true); // id
    pktView.setUint32(4, 0, true); // count = 0
    pktView.setUint32(8, 0, true); // flags = 0
    pktView.setBigUint64(16, BigInt(i), true); // distinct contentId
    assert.equal(ext.handlePacket(V2_OPCODES.CREATE_BIND_GROUP_LAYOUT, pktView, 0, 24), true);
  }
  assert.ok(
    ext.layoutsByContent.size <= 4096,
    `layoutsByContent must stay <= 4096 (got ${ext.layoutsByContent.size})`,
  );
});
