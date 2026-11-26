import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);

const {
  PTHREAD_CMD_LOAD,
  PTHREAD_CMD_LOADED,
  V8_SIDE_MODULE_CAP_BYTES,
  V8_SIDE_MODULE_WARN_RATIO,
  compileWasmModuleStreaming,
  createSharedWasmModuleCache,
  createPthreadPoolCoordinator,
  bootstrapEmscriptenPthreadWorker,
  parseGuestManifest,
  extractSideModuleEntries,
  evaluateV8SideModuleBudget,
  inspectRuntimeExports,
  auditIndirectTableAndGotFunc,
  ensureIndirectFunctionTableCapacity,
  buildRuntimeModuleConfig,
  getWasmMemoryTelemetry,
  resetWasmMemoryTelemetry,
} = require('../web/loader.js');

const {
  GpuStagingPool,
} = require('../web/workers/staging_pool.js');

const {
  ResourceTable,
  GPU_BUFFER_USAGE,
  estimateTextureByteSize,
} = require('../web/workers/resource_table.js');

const {
  WgpuCommandProcessor,
  OPCODES,
  buildGpuTelemetrySnapshot,
} = require('../web/workers/wgpu_worker.js');

const MiB = 1024 * 1024;

function makeMinimalWasmBytes(extraPayload = 0) {
  if (extraPayload <= 0) {
    return new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
  }
  const len = Math.min(100, extraPayload);
  const secLen = 2 + len;
  const out = new Uint8Array(8 + 2 + secLen);
  out.set([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x00, secLen, 0x01, 0x78], 0);
  return out;
}

function makeMemoryImportWasmBytes() {
  const sec = (id, payload) => [id, payload.length, ...payload];
  const importSec = sec(0x02, [
    0x01,
    0x03, 0x65, 0x6e, 0x76, 0x06, 0x6d, 0x65, 0x6d, 0x6f, 0x72, 0x79, 0x02, 0x01, 0x01, 0x01,
  ]);
  return new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, ...importSec]);
}

function makeBeginFramePacket() {
  const buf = new ArrayBuffer(8);
  const view = new DataView(buf);
  view.setUint16(0, OPCODES.BEGIN_FRAME, true);
  view.setUint32(4, 0, true);
  return new Uint8Array(buf);
}

function makeCreateBufferPacket(id, usage, byteSize) {
  const buf = new ArrayBuffer(8 + 16);
  const view = new DataView(buf);
  view.setUint16(0, OPCODES.CREATE_BUFFER, true);
  view.setUint32(4, 16, true);
  view.setUint32(8, id, true);
  view.setUint32(12, usage, true);
  view.setBigUint64(16, BigInt(byteSize), true);
  return new Uint8Array(buf);
}

function makeWriteBufferPacket(id, dstOffset, data, { useSab = false } = {}) {
  const totalBytes = 8 + 16 + data.byteLength;
  const backing = useSab ? new SharedArrayBuffer(totalBytes) : new ArrayBuffer(totalBytes);
  const out = new Uint8Array(backing);
  const view = new DataView(backing);
  view.setUint16(0, OPCODES.WRITE_BUFFER, true);
  view.setUint32(4, 16 + data.byteLength, true);
  view.setUint32(8, id, true);
  view.setUint32(12, dstOffset, true);
  view.setUint32(16, data.byteLength, true);
  view.setUint32(20, 0, true);
  out.set(data, 24);
  return out;
}

function makeCreateTexturePacket(id, width, height, formatCode = 0, usage = 0x06) {
  const buf = new ArrayBuffer(8 + 32);
  const view = new DataView(buf);
  view.setUint16(0, OPCODES.CREATE_TEXTURE, true);
  view.setUint32(4, 32, true);
  view.setUint32(8, id, true);
  view.setUint32(12, width, true);
  view.setUint32(16, height, true);
  view.setUint32(20, 1, true);
  view.setUint32(24, 1, true);
  view.setUint32(28, formatCode, true);
  view.setUint32(32, usage, true);
  view.setUint32(36, 1, true); // 2d
  return new Uint8Array(buf);
}

function makeDestroyResourcePacket(id) {
  const buf = new ArrayBuffer(12);
  const view = new DataView(buf);
  view.setUint16(0, OPCODES.DESTROY_RESOURCE, true);
  view.setUint32(4, 4, true);
  view.setUint32(8, id, true);
  return new Uint8Array(buf);
}

function makePresentPacket() {
  const buf = new ArrayBuffer(8);
  const view = new DataView(buf);
  view.setUint16(0, OPCODES.PRESENT, true);
  view.setUint32(4, 0, true);
  return new Uint8Array(buf);
}

test('W371 #1077/#3: side modules compile strictly ONCE via streaming and share identical WebAssembly.Module instances across 48 pthreads', async () => {
  resetWasmMemoryTelemetry();

  const partBytes = {
    'guest_part_0.wasm': makeMinimalWasmBytes(24),
    'guest_part_1.wasm': makeMinimalWasmBytes(12),
    'guest_part_2.wasm': makeMinimalWasmBytes(36),
  };
  const declaredSizes = {
    'guest_part_0.wasm': 890 * MiB,
    'guest_part_1.wasm': Math.round(0.14 * MiB),
    'guest_part_2.wasm': Math.round(6.7 * MiB),
  };

  let fetchCount = 0;
  let compileStreamingCount = 0;
  let compileFallbackCount = 0;
  let arrayBufferCalls = 0;
  const warnings = [];

  const mockWasmApi = {
    Module: WebAssembly.Module,
    compile: async (bytes) => {
      compileFallbackCount += 1;
      return WebAssembly.compile(bytes);
    },
    compileStreaming: async (respPromise) => {
      compileStreamingCount += 1;
      const resp = await respPromise;
      const base = resp.url.split('/').pop();
      return WebAssembly.compile(partBytes[base]);
    },
  };

  const mockFetch = async (url) => {
    fetchCount += 1;
    const base = url.split('/').pop();
    const resp = new Response(partBytes[base], {
      status: 200,
      headers: {
        'content-type': 'application/wasm',
        'content-length': String(declaredSizes[base]),
      },
    });
    Object.defineProperty(resp, 'url', { value: url, configurable: true });
    resp.arrayBuffer = async () => {
      arrayBufferCalls += 1;
      throw new Error('arrayBuffer() must not be called on streaming path');
    };
    return resp;
  };

  const cache = createSharedWasmModuleCache({
    wasmApi: mockWasmApi,
    fetchFn: mockFetch,
    trackGlobalCount: true,
    reportFn: (msg) => warnings.push(msg),
  });

  // Precompile the 3 asymmetric side modules once.
  const compiled = await cache.precompileSideModules(
    ['guest_part_0.wasm', 'guest_part_1.wasm', 'guest_part_2.wasm'],
    { resolveUrl: (u) => `http://127.0.0.1:8080/dist/${u}` },
  );

  assert.equal(compiled.length, 3);
  assert.equal(fetchCount, 3);
  assert.equal(compileStreamingCount, 3, 'each of the 3 side modules must compile strictly once');
  assert.equal(compileFallbackCount, 0);
  assert.equal(arrayBufferCalls, 0);
  assert.equal(cache.getStats().compileCount, 3);
  assert.equal(warnings.length, 1, '890 MiB guest_part_0.wasm should trigger V8 1 GiB headroom warning');
  assert.ok(warnings[0].includes('guest_part_0.wasm'));
  assert.ok(warnings[0].includes('1024 MiB'));

  // Concurrent duplicate in-flight compileAndShare requests must share 1 compile and drop caller wire bytes.
  let slowCompileCalls = 0;
  let releaseSlow;
  const slowGate = new Promise((r) => {
    releaseSlow = r;
  });
  const slowWasmApi = {
    Module: WebAssembly.Module,
    compile: async (bytes) => {
      slowCompileCalls += 1;
      await slowGate;
      return WebAssembly.compile(bytes);
    },
  };
  const dedupCache = createSharedWasmModuleCache({ wasmApi: slowWasmApi });
  const holderA = { buf: makeMinimalWasmBytes(20).buffer };
  const holderB = { buf: makeMinimalWasmBytes(20).buffer };
  const pA = dedupCache.compileAndShare('guest_part_9.wasm', holderA.buf, {
    owner: holderA,
    ownerKey: 'buf',
  });
  const pB = dedupCache.compileAndShare('guest_part_9.wasm', holderB.buf, {
    owner: holderB,
    ownerKey: 'buf',
  });
  assert.equal(holderA.buf, null, 'first caller wire bytes dropped immediately');
  assert.equal(holderB.buf, null, 'in-flight duplicate caller wire bytes dropped immediately');
  releaseSlow();
  const [mA, mB] = await Promise.all([pA, pB]);
  assert.equal(mA, mB);
  assert.equal(slowCompileCalls, 1, 'concurrent duplicate requests must compile strictly once');

  // Wire into pthread pool coordinator and spawn 48 pthreads!
  const coordinator = createPthreadPoolCoordinator({
    maxConcurrentPthreadLoads: 8,
    getSharedModules: () => cache.getSharedModulesArray(),
    getSharedModuleNames: () => cache.getSharedModuleNames(),
  });

  const mainMod = await WebAssembly.compile(makeMinimalWasmBytes(8));
  const workers = [];
  for (let i = 0; i < 48; i += 1) {
    const received = [];
    const w = {
      received,
      postMessage(msg, transfer) {
        received.push({ msg, transfer });
      },
      terminate() {},
    };
    coordinator.registerWorker(w, { name: `em-pthread-${i + 1}` });
    workers.push(w);

    const loadMsg = {
      cmd: PTHREAD_CMD_LOAD,
      wasmModule: mainMod,
      // Simulate Emscripten sending an empty `sharedModules: []` on half the workers
      // and omitting `sharedModules` on the other half, plus a stray wasmBinary on every 4th.
      ...(i % 2 === 0 ? { sharedModules: [] } : {}),
      ...(i % 4 === 0 ? { wasmBinary: new ArrayBuffer(128) } : {}),
    };
    w.postMessage(loadMsg);
    coordinator.notifyWorkerLoaded(w);
  }

  for (let i = 0; i < 48; i += 1) {
    const loadDelivery = workers[i].received.find((r) => r.msg && r.msg.cmd === PTHREAD_CMD_LOAD);
    assert.ok(loadDelivery, `pthread ${i + 1} must receive CMD_LOAD`);
    assert.equal(loadDelivery.msg.sharedModules.length, 3);
    assert.equal(loadDelivery.msg.sharedModules[0], compiled[0]);
    assert.equal(loadDelivery.msg.sharedModules[1], compiled[1]);
    assert.equal(loadDelivery.msg.sharedModules[2], compiled[2]);
    assert.deepEqual(loadDelivery.msg.__playgta6_shared_module_names, [
      'guest_part_0.wasm',
      'guest_part_1.wasm',
      'guest_part_2.wasm',
    ]);
    if (i % 4 === 0) {
      assert.equal(loadDelivery.msg.wasmBinary, null, 'stray wasmBinary stripped on CMD_LOAD');
    }
  }

  // Still strictly 3 compiles & 3 fetches across all 48 pthreads!
  assert.equal(fetchCount, 3);
  assert.equal(compileStreamingCount, 3);
  assert.equal(coordinator.getTelemetry().pthreadSharedModulesDelivered, 48 * 3);

  const tel = getWasmMemoryTelemetry();
  assert.equal(tel.streamingCompiles, 3);
  assert.equal(tel.largestSideModuleBytes, 890 * MiB);
  assert.equal(tel.sideModuleHeadroomBytes, (1024 - 890) * MiB);
  assert.equal(tel.peakWireBytesInFlight, 32); // only from slowWasmApi direct binary test above (8+2+2+20=32B)

  // Verify pthread worker side (`bootstrapEmscriptenPthreadWorker`) intercepts compileStreaming /
  // instantiateStreaming for precompiled sharedModules delivered in CMD_LOAD with 0 recompiles!
  let underlyingPthreadCompileStreaming = 0;
  let underlyingPthreadInstantiateStreaming = 0;
  const listeners = [];
  const pthreadScope = {
    name: 'em-pthread-1',
    location: { href: 'http://127.0.0.1:8080/loader.js' },
    WebAssembly: {
      Module: WebAssembly.Module,
      Instance: WebAssembly.Instance,
      instantiate: WebAssembly.instantiate.bind(WebAssembly),
      compileStreaming: async () => {
        underlyingPthreadCompileStreaming += 1;
        throw new Error('pthread must never recompile a shared side module');
      },
      instantiateStreaming: async () => {
        underlyingPthreadInstantiateStreaming += 1;
        throw new Error('pthread must never re-stream-instantiate a shared side module');
      },
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

  // Deliver CMD_LOAD from coordinator to pthread worker listener (must NOT stop propagation!).
  let stopPropCalled = false;
  const loadEvent = {
    data: workers[0].received[0].msg,
    stopImmediatePropagation() {
      stopPropCalled = true;
    },
  };
  for (const fn of listeners.slice()) {
    fn(loadEvent);
  }
  assert.equal(stopPropCalled, false, 'CMD_LOAD observation must not block Emscripten handler');
  assert.equal(handle.getSharedModules().length, 3);
  assert.equal(handle.getSharedModuleByName('dist/guest_part_0.wasm'), compiled[0]);

  let bodyCancelled = false;
  const fakeResp = {
    url: 'http://127.0.0.1:8080/dist/guest_part_0.wasm',
    bodyUsed: false,
    body: {
      cancel() {
        bodyCancelled = true;
        return Promise.resolve();
      },
    },
  };
  const reusedMod = await pthreadScope.WebAssembly.compileStreaming(Promise.resolve(fakeResp));
  assert.equal(reusedMod, compiled[0]);
  assert.equal(bodyCancelled, true);
  assert.equal(underlyingPthreadCompileStreaming, 0);

  const instRes = await pthreadScope.WebAssembly.instantiateStreaming(
    Promise.resolve({ url: 'http://127.0.0.1:8080/dist/guest_part_1.wasm' }),
    {},
  );
  assert.equal(instRes.module, compiled[1]);
  assert.ok(instRes.instance instanceof WebAssembly.Instance);
  assert.equal(underlyingPthreadInstantiateStreaming, 0);
  assert.equal(handle.getSharedModuleHits(), 2);

  handle.cleanup();
});

test('W371 #1077/#1: unconsumed wasmBinary and failing compiles drop all wire bytes immediately', async () => {
  resetWasmMemoryTelemetry();

  // 1. Failing WebAssembly.compile still drops owner wire bytes and resets in-flight tracking.
  const ownerMap = new Map([['bad.wasm', new ArrayBuffer(4096)]]);
  await assert.rejects(
    () =>
      compileWasmModuleStreaming(ownerMap.get('bad.wasm'), {
        owner: ownerMap,
        ownerKey: 'bad.wasm',
        wasmApi: {
          Module: WebAssembly.Module,
          compile: async () => {
            throw new WebAssembly.CompileError('simulated invalid module');
          },
        },
      }),
    /simulated invalid module/,
  );
  assert.equal(ownerMap.size, 0, 'failed compile must still delete owner Map entry');
  assert.equal(getWasmMemoryTelemetry().wireBytesReleased, 4096);

  // 2. Unconsumed options.wasmBinary in buildRuntimeModuleConfig is released on destroy().
  const wireBuf = makeMemoryImportWasmBytes().buffer;
  const cfg = buildRuntimeModuleConfig({
    runtimeScriptUrl: 'http://127.0.0.1:8080/dist/runtime.js',
    wasmBinary: wireBuf,
  });
  cfg.destroy();
  assert.equal(getWasmMemoryTelemetry().wireBytesReleased, 4096 + wireBuf.byteLength);
});

test('W371 #663 checklist 1 & 2: asymmetric split, single-side-module collapse, out-of-order loading, and V8 1024 MiB cap telemetry', () => {
  assert.equal(V8_SIDE_MODULE_CAP_BYTES, 1024 * MiB);
  assert.equal(V8_SIDE_MODULE_WARN_RATIO, 0.85);

  // 1. Timon's exact asymmetric build shape: part_0 = 890 MiB (383 shards), part_1 = 0.14 MiB (1 shard), part_2 = 6.7 MiB.
  const asymmetricManifest = {
    runtimeJs: 'dist/runtime.js',
    indirectDispatch: 'shard',
    sideModules: [
      { name: 'guest_part_0.wasm', wasm_bytes: 890 * MiB, shard_count: 383 },
      { name: 'guest_part_1.wasm', wasm_bytes: Math.round(0.14 * MiB), shard_count: 1 },
      { name: 'guest_part_2.wasm', wasm_bytes: Math.round(6.7 * MiB) },
    ],
    wasmMetrics: {
      totalTableSize: 5103,
      totalGotFuncImports: 384,
      sideModuleCount: 3,
      mainWasmBytes: Math.round(5.96 * MiB),
    },
  };

  const parsedAsym = parseGuestManifest(asymmetricManifest);
  assert.equal(parsedAsym.sideModuleCount, 3);
  assert.deepEqual(parsedAsym.sideModules, [
    'guest_part_0.wasm',
    'guest_part_1.wasm',
    'guest_part_2.wasm',
  ]);
  assert.equal(parsedAsym.wasmMetrics.totalTableSize, 5103);
  assert.equal(parsedAsym.wasmMetrics.totalGotFuncImports, 384);

  const entries = extractSideModuleEntries(asymmetricManifest);
  assert.equal(entries.length, 3);
  assert.equal(entries[0].shardCount, 383);
  assert.equal(entries[1].shardCount, 1);

  const asymBudget = evaluateV8SideModuleBudget(asymmetricManifest);
  assert.equal(asymBudget.v8SideModuleCapBytes, 1024 * MiB);
  assert.equal(asymBudget.largestSideModuleBytes, 890 * MiB);
  assert.equal(asymBudget.largestSideModuleName, 'guest_part_0.wasm');
  assert.equal(asymBudget.sideModuleHeadroomBytes, 134 * MiB);
  assert.equal(asymBudget.nearCapWarning, true);
  assert.equal(asymBudget.exceedsCap, false);
  assert.equal(asymBudget.splitShape, 'asymmetric');
  assert.equal(asymBudget.warnings.length, 1);

  // Out-of-order completion (part_2 -> part_0 -> part_1) completes cleanly.
  const cfgAsym = buildRuntimeModuleConfig({
    locationHref: 'http://127.0.0.1:8080/loader.js',
    guestManifest: asymmetricManifest,
    reportFn: () => {},
  });
  cfgAsym.locateFile('guest_part_2.wasm');
  cfgAsym.locateFile('guest_part_0.wasm');
  assert.equal(cfgAsym.getSideModuleProgress().complete, false);
  assert.equal(cfgAsym.getSideModuleProgress().loadedCount, 2);
  cfgAsym.locateFile('guest_part_1.wasm');
  assert.equal(cfgAsym.getSideModuleProgress().complete, true);
  assert.equal(cfgAsym.getSideModuleProgress().pct, 100);

  // 2. Single-side-module collapse (part_0 = 890.14 MiB, 384 shards, part_1 absent, 0 GOT.func).
  const collapsedManifest = {
    indirectDispatch: 'shard',
    sideModules: [{ name: 'guest_part_0.wasm', wasm_bytes: Math.round(890.14 * MiB), shard_count: 384 }],
    wasmMetrics: {
      totalTableSize: 5103,
      totalGotFuncImports: 0,
      sideModuleCount: 1,
    },
  };
  const parsedCollapsed = parseGuestManifest(collapsedManifest);
  assert.equal(parsedCollapsed.sideModuleCount, 1);
  assert.deepEqual(parsedCollapsed.sideModules, ['guest_part_0.wasm']);
  const collapsedBudget = evaluateV8SideModuleBudget(collapsedManifest);
  assert.equal(collapsedBudget.splitShape, 'single-collapsed');
  assert.equal(collapsedBudget.nearCapWarning, true);
  assert.equal(collapsedBudget.exceedsCap, false);

  // 3. Non-contiguous parts (part_0 + part_2, part_1 omitted) and over-cap warning (> 1024 MiB).
  const overCapBudget = evaluateV8SideModuleBudget([
    { name: 'guest_part_0.wasm', byteLength: 1050 * MiB },
    { name: 'guest_part_2.wasm', byteLength: 6 * MiB },
  ]);
  assert.equal(overCapBudget.exceedsCap, true);
  assert.equal(overCapBudget.sideModuleHeadroomBytes, 0);
  assert.ok(overCapBudget.warnings[0].includes('exceeds V8 1024 MiB'));
});

test('W371 #663 checklist 3: runtime.wasm without _main uses fast-boot exports cleanly with zero errors/warnings', () => {
  const reports = [];
  const cfg = buildRuntimeModuleConfig({
    locationHref: 'http://127.0.0.1:8080/loader.js',
    reportFn: (m) => reports.push(m),
  });

  // Runtime module exports fast-boot symbols only — NO `_main` and NO `callMain`!
  const fastBootRuntimeMod = {
    wasmExports: {
      playgta6_get_input_state_ptr: () => 0n,
      playgta6_audio_ring: () => 0n,
      playgta6_prewarm_load_list: () => 0,
      playgta6_relaunch_requested: () => 0,
      playgta6_storage_generation: () => 0n,
    },
  };

  const inspected = inspectRuntimeExports(fastBootRuntimeMod);
  assert.equal(inspected.hasMain, false);
  assert.equal(inspected.hasCallMain, false);
  assert.equal(inspected.mode, 'fast-boot');
  assert.deepEqual(inspected.fastBootExports, [
    'playgta6_get_input_state_ptr',
    'playgta6_audio_ring',
    'playgta6_prewarm_load_list',
    'playgta6_relaunch_requested',
    'playgta6_storage_generation',
  ]);

  // Invoking onRuntimeInitialized without `_main` must not throw or emit warnings.
  assert.doesNotThrow(() => {
    cfg.onRuntimeInitialized.call(fastBootRuntimeMod);
  });
  const summary = cfg.getRuntimeExportsSummary();
  assert.equal(summary.hasMain, false);
  assert.equal(summary.mode, 'fast-boot');
  assert.equal(summary.fastBootExports.length, 5);
  assert.equal(reports.length, 0, 'absence of _main must produce zero warnings');
  cfg.destroy();
});

test('W371 #663 checklist 4: __indirect_function_table growth & GOT.func accounting handle 384 dispatchers, >= 5103 slots, 0 GOT.func, BigInt tableBase, and Proxy GOT.func', async () => {
  // 1. Asymmetric build: 384 GOT.func dispatchers + 5103 table slots on an undersized WebAssembly.Table.
  const table = new WebAssembly.Table({ initial: 64, element: 'anyfunc' });
  const gotFunc = {};
  for (let i = 0; i < 384; i += 1) {
    gotFunc[`dispatch_shard_${i}`] = { value: i };
  }
  const imports = {
    env: {
      __indirect_function_table: table,
      __table_base: 4800n, // Wasm64 BigInt tableBase
    },
    'GOT.func': gotFunc,
  };

  const audit = auditIndirectTableAndGotFunc(imports, {
    totalTableSize: 5103,
    totalGotFuncImports: 384,
  });
  assert.equal(audit.hasTable, true);
  assert.equal(audit.currentTableLength, 64);
  assert.equal(audit.tableBase, 4800);
  assert.equal(audit.gotFuncCount, 384);
  assert.equal(audit.isTable64, true);
  assert.equal(audit.requiredTableSize, 4800 + 384); // 5184 > 5103
  assert.equal(audit.needsGrow, true);

  const growRes = ensureIndirectFunctionTableCapacity(imports, 5103, {
    wasmMetrics: { totalTableSize: 5103, totalGotFuncImports: 384 },
  });
  assert.equal(growRes.error, null);
  assert.equal(growRes.grew, true);
  assert.equal(growRes.previousLength, 64);
  assert.equal(growRes.currentLength, 5184);
  assert.equal(table.length, 5184);

  // 2. Emscripten lazy-stub Proxy on GOT.func must NOT be mutated or queried for unknown keys!
  let proxyGetCalls = 0;
  const gotFuncProxy = new Proxy(
    {},
    {
      get(_target, prop) {
        proxyGetCalls += 1;
        throw new Error(`Unexpected lazy-stub get on GOT.func Proxy: ${String(prop)}`);
      },
      ownKeys() {
        return ['shard_0', 'shard_1'];
      },
      getOwnPropertyDescriptor() {
        return { enumerable: true, configurable: true };
      },
    },
  );
  const proxyAudit = auditIndirectTableAndGotFunc(
    {
      env: {
        __indirect_function_table: table,
        __table_base: { value: 100 }, // WebAssembly.Global-like wrapper
      },
      'GOT.func': gotFuncProxy,
    },
    { totalTableSize: 5103, totalGotFuncImports: 384 },
  );
  assert.equal(proxyGetCalls, 0, 'audit must never trigger GOT.func Proxy get trap');
  assert.equal(proxyAudit.tableBase, 100);
  assert.equal(proxyAudit.gotFuncCount, 384);

  // 3. Single-side-module collapse (0 GOT.func dispatchers, 5103 table slots) with BigInt table64 mock.
  let mockTableLen = 128n;
  let lastGrowArg = null;
  const mockTable64 = {
    get length() {
      return mockTableLen;
    },
    grow(delta) {
      lastGrowArg = delta;
      mockTableLen += delta;
    },
  };
  const collapsedRes = ensureIndirectFunctionTableCapacity(
    { env: { __indirect_function_table: mockTable64, __table_base: 0n }, 'GOT.func': {} },
    5103,
    { wasmMetrics: { totalTableSize: 5103, totalGotFuncImports: 0 } },
  );
  assert.equal(collapsedRes.grew, true);
  assert.equal(collapsedRes.gotFuncCount, 0);
  assert.equal(collapsedRes.isTable64, true);
  assert.equal(typeof lastGrowArg, 'bigint', 'table64 grow() must receive BigInt delta');
  assert.equal(collapsedRes.currentLength, 5103);

  // 4. End-to-end buildRuntimeModuleConfig instantiateWasm grows undersized __indirect_function_table automatically.
  const autoTable = new WebAssembly.Table({ initial: 16, element: 'anyfunc' });
  const cfg = buildRuntimeModuleConfig({
    runtimeScriptUrl: 'http://127.0.0.1:8080/dist/runtime.js',
    wasmBinary: makeMemoryImportWasmBytes().buffer,
    guestManifest: {
      indirectDispatch: 'shard',
      sideModules: ['guest_part_0.wasm'],
      wasmMetrics: { totalTableSize: 5103, totalGotFuncImports: 384, sideModuleCount: 1 },
    },
  });
  const memory = new WebAssembly.Memory({ initial: 1, maximum: 1 });
  await new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error('instantiateWasm timed out')), 2000);
    cfg.instantiateWasm(
      { env: { memory, __indirect_function_table: autoTable, __table_base: 0 } },
      () => {
        clearTimeout(t);
        resolve();
      },
    );
  });
  assert.ok(autoTable.length >= 5103, `expected autoTable.length >= 5103, got ${autoTable.length}`);
});

test('W371 #1077/#2: accurate GPU memory accounting, SAB-safe <=4 MiB direct chunking, and bounded pre-present deferred destroy flushing', async () => {
  const queueWrites = [];
  const createdBuffers = [];
  const createdTextures = [];
  const fences = [];

  const device = {
    features: new Set(['texture-compression-bc']),
    queue: {
      writeBuffer(buffer, offset, data) {
        assert.equal(
          typeof SharedArrayBuffer !== 'undefined' && data.buffer instanceof SharedArrayBuffer,
          false,
          'SharedArrayBuffer must never be passed to queue.writeBuffer',
        );
        assert.equal(offset % 4, 0);
        assert.equal(data.byteLength % 4, 0);
        assert.ok(data.byteLength <= 4 * MiB, `writeBuffer exceeded 4 MiB cap: ${data.byteLength}`);
        queueWrites.push({ buffer, offset, byteLength: data.byteLength });
      },
      writeTexture() {},
      submit() {},
      onSubmittedWorkDone() {
        let resolve;
        const promise = new Promise((r) => {
          resolve = r;
        });
        fences.push({ promise, resolve });
        return promise;
      },
    },
    createBuffer(desc) {
      const buf = {
        kind: 'GPUBuffer',
        size: desc.size,
        usage: desc.usage,
        destroyed: false,
        destroyCount: 0,
        destroy() {
          this.destroyed = true;
          this.destroyCount += 1;
        },
      };
      createdBuffers.push(buf);
      return buf;
    },
    createTexture(desc) {
      const tex = {
        kind: 'GPUTexture',
        format: desc.format,
        dimension: desc.dimension || '2d',
        width: desc.size.width,
        height: desc.size.height,
        destroyed: false,
        destroyCount: 0,
        createView() {
          return { kind: 'GPUTextureView' };
        },
        destroy() {
          this.destroyed = true;
          this.destroyCount += 1;
        },
      };
      createdTextures.push(tex);
      return tex;
    },
    createCommandEncoder() {
      return {
        copyBufferToBuffer() {},
        copyBufferToTexture() {},
        beginRenderPass() {
          return {
            setPipeline() {},
            setVertexBuffer() {},
            setIndexBuffer() {},
            setBindGroup() {},
            draw() {},
            end() {},
          };
        },
        finish() {
          return { kind: 'GPUCommandBuffer' };
        },
      };
    },
  };

  const ctx = {
    getCurrentTexture: () => ({ createView: () => ({ kind: 'swapchain' }) }),
  };

  // 1. Processor WITHOUT resourceRegistry still tracks live/peak buffer & texture bytes accurately!
  const stagingPool = new GpuStagingPool({
    device,
    chunkSize: 2 * MiB,
    maxPoolBytes: 8 * MiB,
    maxWriteChunkBytes: 4 * MiB,
  });
  const resourceTable = new ResourceTable(device, { maxWriteChunkBytes: 4 * MiB });

  const processor = new WgpuCommandProcessor(device, ctx, {
    stagingPool,
    resourceTable,
    maxDeferredGpuDestroys: 16,
    maxDeferredGpuDestroyBytes: 8 * MiB,
  });

  // Allocate a 10 MiB buffer + 1024x1024 RGBA8 (4 MiB) texture on processor, plus a 1 MiB buffer on resourceTable.
  processor.executeCommandBuffer(makeCreateBufferPacket(1, GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST, 10 * MiB));
  processor.executeCommandBuffer(makeCreateTexturePacket(2, 1024, 1024, 0, 0x06));
  resourceTable.createBuffer(100, { size: 1 * MiB, usage: GPU_BUFFER_USAGE.UNIFORM });
  resourceTable.createTexture(101, { size: { width: 512, height: 512 }, format: 'rgba8unorm' });

  const mem1 = processor.getGpuMemoryStats();
  assert.equal(mem1.liveBufferBytes, 11 * MiB);
  assert.equal(mem1.peakBufferBytes, 11 * MiB);
  assert.equal(mem1.liveTextureBytes, 4 * MiB + estimateTextureByteSize(512, 512, 1, 1, 'rgba8unorm', '2d', 1));
  assert.equal(processor.getStats().vramBytes, mem1.totalLiveGpuBytes);
  assert.equal(buildGpuTelemetrySnapshot(processor).gpuMemory.liveBufferBytes, 11 * MiB);

  // 2. SAB-backed >4 MiB odd-length direct writeBuffer on a processor without stagingPool stays <=4 MiB & SAB-safe!
  const directProc = new WgpuCommandProcessor(device, ctx);
  directProc.executeCommandBuffer(makeCreateBufferPacket(1, GPU_BUFFER_USAGE.COPY_DST, 12 * MiB));
  const sabOdd = new Uint8Array(new SharedArrayBuffer(9 * MiB + 3));
  directProc.executeCommandBuffer(makeWriteBufferPacket(1, 0, sabOdd, { useSab: true }));
  assert.equal(queueWrites.length, 3);
  assert.ok(queueWrites.every((w) => w.byteLength <= 4 * MiB));
  assert.equal(directProc.getStats().chunkedWriteBufferUploads, 1);
  assert.ok(directProc.getGpuMemoryStats().directWriteScratchBytes <= 4 * MiB);
  assert.ok(directProc.getGpuMemoryStats().directWriteScratchBytes > 0);
  directProc.trimTransientMemory();
  assert.equal(directProc.getGpuMemoryStats().directWriteScratchBytes, 0);
  directProc.destroy();

  // 3. Pre-present loading burst: BeginFrame followed by 10x 2 MiB buffer overwrites (20 MiB > 8 MiB high-water cap)
  // triggers high-water deferred-destroy drain BEFORE Present so dead GPUBuffers never pile up to gigabytes!
  processor.executeCommandBuffer(makeBeginFramePacket());
  for (let i = 0; i < 10; i += 1) {
    processor.executeCommandBuffer(makeCreateBufferPacket(50, GPU_BUFFER_USAGE.STORAGE, 2 * MiB));
  }
  assert.ok(
    processor.getStats().deferredHighWaterFlushes >= 1,
    'pre-present loading burst must trigger high-water deferred destroy flush',
  );
  // Resolve pending GPU fences and verify old overwritten buffers were actually destroyed!
  while (fences.length > 0) {
    fences.shift().resolve();
  }
  await Promise.resolve();
  await Promise.resolve();
  const destroyedCount = createdBuffers.filter((b) => b.destroyCount === 1).length;
  assert.ok(destroyedCount >= 4, `expected overwritten buffers destroyed before Present, got ${destroyedCount}`);
  assert.ok(createdBuffers.every((b) => b.destroyCount <= 1), 'no buffer may ever be double-destroyed');

  // Finish frame and destroy remaining resources; live bytes return to 0 while peak stays preserved!
  processor.executeCommandBuffer(makePresentPacket());
  while (fences.length > 0) {
    fences.shift().resolve();
  }
  await Promise.resolve();

  processor.executeCommandBuffer(makeDestroyResourcePacket(1));
  processor.executeCommandBuffer(makeDestroyResourcePacket(2));
  processor.executeCommandBuffer(makeDestroyResourcePacket(50));
  resourceTable.destroyBuffer(100);
  resourceTable.destroyTexture(101);
  processor.flushDeferredDestroys({ immediate: true });

  const memEnd = processor.getGpuMemoryStats();
  assert.equal(memEnd.liveBufferBytes, 0);
  assert.equal(memEnd.liveTextureBytes, 0);
  assert.equal(memEnd.pendingDeferredDestroys, 0);
  assert.equal(memEnd.pendingDeferredDestroyBytes, 0);
  assert.ok(memEnd.peakBufferBytes >= 13 * MiB);

  // Verify GpuStagingPool memory stats & trimScratch().
  const spMem = stagingPool.getMemoryStats();
  assert.equal(spMem.stagingPoolBytes, spMem.allocatedBytes);
  assert.equal(typeof spMem.scratchBytes, 'number');
  assert.equal(stagingPool.trimScratch(), 0);

  // Verify ResourceTable memory stats & O(1) dedup.
  const rtMem = resourceTable.getMemoryStats();
  assert.equal(rtMem.liveBufferBytes, 0);
  assert.equal(rtMem.liveTextureBytes, 0);
  assert.equal(rtMem.peakBufferBytes, 1 * MiB);

  processor.destroy();
  resourceTable.destroyAll();
  stagingPool.destroy();
});
