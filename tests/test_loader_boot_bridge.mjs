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
  EMSCRIPTEN_PTHREAD_WORKER_NAME,
  PTHREAD_WGPU_CHANNEL_NAME,
  PTHREAD_HTTPFS_SAB_CHANNEL_NAME,
  HOST_ABI_IMPORT_NAMES,
  GPU_OPCODE_PRESENT,
  GPU_PACKET_HEADER_BYTES,
  INPUT_STATE_BYTES,
  INPUT_STATE_WORDS,
  DEFAULT_INPUT_BRIDGE_POLL_MS,
  DEFAULT_MEMFS_PIPELINE_DUMP_DIR,
  DEFAULT_MEMFS_PIPELINE_DUMP_ENDPOINT,
  DEFAULT_MEMFS_PIPELINE_DUMP_INTERVAL_MS,
  DEFAULT_MEMFS_PIPELINE_DUMP_MAX_CONCURRENT_UPLOADS,
  BOOT_INIT_FORWARD_KEYS,
  isEmscriptenPthreadWorkerName,
  isEmscriptenPthreadWorker,
  createPatchedPthreadEnv,
  installPthreadHostImportShim,
  bootstrapEmscriptenPthreadWorker,
  inspectGpuPacketBatchForPresent,
  normalizeBootHttpfsPath,
  collectBootBridgeInitFields,
  createWorkerBootInitPayloads,
  forwardGpuWorkerMessage,
  sanitizeRenderStateStats,
  getLoaderStatus,
  createHostImports,
  startInputSabBridge,
  resolveDumpPipelinesEnabled,
  sanitizeDumpRelPath,
  createMemfsPipelineDumpSyncer,
  resolveRuntimeScriptUrl,
  resolveSideModuleUrl,
  sanitizeSideModuleUrls,
  buildRuntimeModuleConfig,
  createRuntimeModuleOptions,
  encodeTestFramePacket,
} = require('../web/loader.js');

function packet(opcode, payloadBytes = 0, flags = 0) {
  const buf = new ArrayBuffer(GPU_PACKET_HEADER_BYTES + payloadBytes);
  const view = new DataView(buf);
  view.setUint16(0, opcode, true);
  view.setUint16(2, flags, true);
  view.setUint32(4, payloadBytes, true);
  return new Uint8Array(buf);
}

function concatPackets(chunks) {
  let total = 0;
  for (const chunk of chunks) {
    total += chunk.byteLength;
  }
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out;
}

function submitBytes(imports, memory, bytes, ptr = 64) {
  const memU8 = new Uint8Array(memory.buffer);
  memU8.set(bytes, ptr);
  imports.env.host_wgpu_submit(BigInt(ptr), BigInt(bytes.byteLength));
}

test('GPU_OPCODE_PRESENT and header size match playgta6 ABI (Present=7, not ClearColor=2)', () => {
  assert.equal(GPU_PACKET_HEADER_BYTES, 8);
  assert.equal(GPU_OPCODE_PRESENT, 7);
  assert.notEqual(GPU_OPCODE_PRESENT, 2);
});

test('inspectGpuPacketBatchForPresent finds Present in a valid multi-packet frame', () => {
  const stream = new Uint8Array(encodeTestFramePacket(0.1, 0.2, 0.3, 1.0));
  const info = inspectGpuPacketBatchForPresent(stream);
  assert.equal(info.packetCount, 3);
  assert.equal(info.hasPresent, true);
  assert.equal(info.firstPresentOffset, 32);
  assert.equal(info.truncated, false);
});

test('inspectGpuPacketBatchForPresent reports no Present on BeginFrame + ClearColor only', () => {
  const begin = packet(1, 0);
  const clear = packet(2, 16);
  const stream = concatPackets([begin, clear]);
  const info = inspectGpuPacketBatchForPresent(stream);
  assert.equal(info.packetCount, 2);
  assert.equal(info.hasPresent, false);
  assert.equal(info.firstPresentOffset, -1);
  assert.equal(info.truncated, false);
});

test('inspectGpuPacketBatchForPresent treats empty and null-like input as a safe empty result', () => {
  for (const input of [new Uint8Array(), new ArrayBuffer(0), null, undefined, 12, 'pkt', {}]) {
    const info = inspectGpuPacketBatchForPresent(input);
    assert.deepEqual(info, {
      packetCount: 0,
      hasPresent: false,
      firstPresentOffset: -1,
      truncated: false,
    });
  }
});

test('inspectGpuPacketBatchForPresent flags truncated headers without throwing', () => {
  const truncatedHeader = new Uint8Array([7, 0, 0, 0, 0]);
  const info = inspectGpuPacketBatchForPresent(truncatedHeader);
  assert.equal(info.packetCount, 0);
  assert.equal(info.hasPresent, false);
  assert.equal(info.truncated, true);
});

test('inspectGpuPacketBatchForPresent flags truncated payloads without throwing', () => {
  const buf = new ArrayBuffer(12);
  const view = new DataView(buf);
  view.setUint16(0, 2, true);
  view.setUint16(2, 0, true);
  view.setUint32(4, 16, true);
  const info = inspectGpuPacketBatchForPresent(buf);
  assert.equal(info.packetCount, 0);
  assert.equal(info.hasPresent, false);
  assert.equal(info.truncated, true);
});

test('inspectGpuPacketBatchForPresent flags oversized payload lengths without throwing', () => {
  const buf = new ArrayBuffer(8);
  const view = new DataView(buf);
  view.setUint16(0, GPU_OPCODE_PRESENT, true);
  view.setUint16(2, 0, true);
  view.setUint32(4, 0xffffffff, true);
  const info = inspectGpuPacketBatchForPresent(buf);
  assert.equal(info.packetCount, 0);
  assert.equal(info.hasPresent, false);
  assert.equal(info.firstPresentOffset, -1);
  assert.equal(info.truncated, true);
});

test('inspectGpuPacketBatchForPresent keeps complete packets before a truncated tail', () => {
  const present = packet(GPU_OPCODE_PRESENT, 0);
  const tail = new Uint8Array([1, 0, 0]);
  const stream = concatPackets([present, tail]);
  const info = inspectGpuPacketBatchForPresent(stream);
  assert.equal(info.packetCount, 1);
  assert.equal(info.hasPresent, true);
  assert.equal(info.firstPresentOffset, 0);
  assert.equal(info.truncated, true);
});

test('createHostImports host_wgpu_submit fires onFirstPresent and [boot] FIRST_PRESENT exactly once', () => {
  const memory = new WebAssembly.Memory({ initial: 1, maximum: 1 });
  const presents = [];
  const reports = [];
  const logs = [];
  const submitted = [];
  const imports = createHostImports({
    memory,
    log: (msg) => logs.push(msg),
    report: (msg) => reports.push(msg),
    onFirstPresent: (detail) => presents.push(detail),
    wgpuSubmit: (bytes) => submitted.push(bytes.slice()),
  });

  const frame = new Uint8Array(encodeTestFramePacket(0.08, 0.14, 0.24, 1.0));
  submitBytes(imports, memory, frame, 128);
  submitBytes(imports, memory, frame, 128);
  submitBytes(imports, memory, packet(GPU_OPCODE_PRESENT, 0), 400);

  assert.equal(submitted.length, 3);
  assert.equal(presents.length, 1);
  assert.equal(presents[0].packetCount, 3);
  assert.equal(presents[0].byteLength, frame.byteLength);
  assert.deepEqual(reports, ['[boot] FIRST_PRESENT']);
  assert.equal(logs.length, 0);
});

test('createHostImports does not emit FIRST_PRESENT for ClearColor-only batches', () => {
  const memory = new WebAssembly.Memory({ initial: 1, maximum: 1 });
  const presents = [];
  const reports = [];
  const imports = createHostImports({
    memory,
    report: (msg) => reports.push(msg),
    onFirstPresent: (detail) => presents.push(detail),
  });

  const clearOnly = concatPackets([packet(1, 0), packet(2, 16)]);
  submitBytes(imports, memory, clearOnly, 80);
  assert.equal(presents.length, 0);
  assert.equal(reports.length, 0);

  submitBytes(imports, memory, packet(GPU_OPCODE_PRESENT, 0), 80);
  assert.equal(presents.length, 1);
  assert.deepEqual(reports, ['[boot] FIRST_PRESENT']);
});

test('createHostImports emitFirstPresentLog mirrors FIRST_PRESENT onto logFn once', () => {
  const memory = new WebAssembly.Memory({ initial: 1, maximum: 1 });
  const logs = [];
  const imports = createHostImports({
    memory,
    emitFirstPresentLog: true,
    log: (msg) => logs.push(msg),
    report: () => {},
  });

  const frame = new Uint8Array(encodeTestFramePacket(0.2, 0.3, 0.4, 1.0));
  submitBytes(imports, memory, frame, 16);
  submitBytes(imports, memory, frame, 16);
  assert.deepEqual(logs, ['[boot] FIRST_PRESENT']);
});

test('normalizeBootHttpfsPath routes /boot/ program-manifest and pgim when bootBase is set', () => {
  const opts = { dataBase: '/data/', bootBase: '/boot/' };
  assert.equal(normalizeBootHttpfsPath('boot/program-manifest.json', opts), '/boot/program-manifest.json');
  assert.equal(normalizeBootHttpfsPath('/boot/guest.pgim', opts), '/boot/guest.pgim');
  assert.equal(normalizeBootHttpfsPath('program-manifest.json', opts), '/boot/program-manifest.json');
  assert.equal(normalizeBootHttpfsPath('images/title.pgim', opts), '/boot/images/title.pgim');
});

test('normalizeBootHttpfsPath leaves data-manifest and rpf paths on /data/', () => {
  const opts = { dataBase: '/data/', bootBase: '/boot/' };
  assert.equal(normalizeBootHttpfsPath('data-manifest.json', opts), '/data/data-manifest.json');
  assert.equal(normalizeBootHttpfsPath('x64/update.rpf', opts), '/data/x64/update.rpf');
  assert.equal(normalizeBootHttpfsPath('archives/common.rpf', opts), '/data/archives/common.rpf');
});

test('normalizeBootHttpfsPath uses dataBase when bootBase is omitted', () => {
  assert.equal(normalizeBootHttpfsPath('boot/guest.pgim', { dataBase: '/data/' }), '/data/boot/guest.pgim');
  assert.equal(normalizeBootHttpfsPath('program-manifest.json'), '/data/program-manifest.json');
});

test('normalizeBootHttpfsPath blocks path traversal and empty segments', () => {
  const opts = { dataBase: '/data/', bootBase: '/boot/' };
  assert.throws(() => normalizeBootHttpfsPath('../secret.pgim', opts), /Invalid HTTPFS path/);
  assert.throws(() => normalizeBootHttpfsPath('boot/../etc/passwd', opts), /Invalid HTTPFS path/);
  assert.throws(() => normalizeBootHttpfsPath('boot/./hidden.pgim', opts), /Invalid HTTPFS path/);
  assert.throws(() => normalizeBootHttpfsPath('foo\\bar.pgim', opts), /Invalid HTTPFS path/);
  assert.throws(() => normalizeBootHttpfsPath('', opts), /non-empty string/);
});

test('collectBootBridgeInitFields and worker payloads omit optional URLs when absent', () => {
  assert.deepEqual(BOOT_INIT_FORWARD_KEYS, [
    'shaderPackUrl',
    'reflectionManifestUrl',
    'pgimUrl',
    'bootBase',
    'gameBase',
    'programManifestUrl',
    'opfsL2Enabled',
  ]);
  assert.deepEqual(collectBootBridgeInitFields({ opfsL2Enabled: false }), { opfsL2Enabled: false });
  assert.deepEqual(collectBootBridgeInitFields({ canvas: true, dataBase: '/data/' }), {});
  const payloads = createWorkerBootInitPayloads({
    canvas: 'offscreen',
    width: 1280,
    height: 720,
    dataBase: '/data/',
  });
  assert.deepEqual(payloads.ioInit, { init: true, dataBase: '/data/' });
  assert.deepEqual(payloads.gpuInit, {
    init: true,
    canvas: 'offscreen',
    width: 1280,
    height: 720,
  });
  assert.equal('shaderPackUrl' in payloads.ioInit, false);
  assert.equal('bootBase' in payloads.gpuInit, false);
  assert.equal('gameBase' in payloads.ioInit, false);
  assert.equal('programManifestUrl' in payloads.gpuInit, false);
});

test('createWorkerBootInitPayloads forwards shader pack, reflection, pgim, bootBase, gameBase, and programManifestUrl', () => {
  const msg = {
    canvas: 'gpu-canvas',
    width: 1920,
    height: 1080,
    dataBase: '/data/',
    shaderPackUrl: '/shaders/fallback.wgsl.pack',
    reflectionManifestUrl: '/shaders/reflection-manifest.json',
    pgimUrl: '/boot/guest.pgim',
    bootBase: '/boot/',
    gameBase: 'http://127.0.0.1:8080/game/',
    programManifestUrl: '/boot/program-manifest.json',
  };
  const payloads = createWorkerBootInitPayloads(msg);
  assert.equal(payloads.ioInit.init, true);
  assert.equal(payloads.ioInit.dataBase, '/data/');
  assert.equal(payloads.ioInit.shaderPackUrl, msg.shaderPackUrl);
  assert.equal(payloads.ioInit.reflectionManifestUrl, msg.reflectionManifestUrl);
  assert.equal(payloads.ioInit.pgimUrl, msg.pgimUrl);
  assert.equal(payloads.ioInit.bootBase, '/boot/');
  assert.equal(payloads.ioInit.gameBase, msg.gameBase);
  assert.equal(payloads.ioInit.programManifestUrl, msg.programManifestUrl);
  assert.equal(payloads.gpuInit.canvas, 'gpu-canvas');
  assert.equal(payloads.gpuInit.shaderPackUrl, msg.shaderPackUrl);
  assert.equal(payloads.gpuInit.reflectionManifestUrl, msg.reflectionManifestUrl);
  assert.equal(payloads.gpuInit.pgimUrl, msg.pgimUrl);
  assert.equal(payloads.gpuInit.bootBase, '/boot/');
  assert.equal(payloads.gpuInit.gameBase, msg.gameBase);
  assert.equal(payloads.gpuInit.programManifestUrl, msg.programManifestUrl);
});

test('normalizeBootHttpfsPath routes Xbox game package and drive paths to gameBase when configured', () => {
  const opts = {
    dataBase: '/data/',
    bootBase: '/boot/',
    gameBase: 'http://127.0.0.1:8080/game/',
  };
  assert.equal(
    normalizeBootHttpfsPath('x64/update.rpf', opts),
    'http://127.0.0.1:8080/game/x64/update.rpf',
  );
  assert.equal(
    normalizeBootHttpfsPath('game/x64/common.rpf', opts),
    'http://127.0.0.1:8080/game/x64/common.rpf',
  );
  assert.equal(
    normalizeBootHttpfsPath('/game/config/settings.xml', opts),
    'http://127.0.0.1:8080/game/config/settings.xml',
  );
  assert.equal(
    normalizeBootHttpfsPath('G:\\x64\\update.rpf', opts),
    'http://127.0.0.1:8080/game/x64/update.rpf',
  );
  assert.equal(
    normalizeBootHttpfsPath('\\\\?\\G:\\MicrosoftGame.Config', opts),
    'http://127.0.0.1:8080/game/MicrosoftGame.Config',
  );
  assert.equal(
    normalizeBootHttpfsPath('MicrosoftGame.Config', opts),
    'http://127.0.0.1:8080/game/MicrosoftGame.Config',
  );
  assert.equal(
    normalizeBootHttpfsPath('bin/xgameruntime.dll', opts),
    'http://127.0.0.1:8080/game/bin/xgameruntime.dll',
  );
  assert.equal(
    normalizeBootHttpfsPath('G:\\content\\manifest.txt', opts),
    'http://127.0.0.1:8080/game/content/manifest.txt',
  );

  // Boot-routed and data-pinned paths continue routing to bootBase and dataBase.
  assert.equal(normalizeBootHttpfsPath('data-manifest.json', opts), '/data/data-manifest.json');
  assert.equal(normalizeBootHttpfsPath('boot/guest.pgim', opts), '/boot/guest.pgim');
  assert.equal(normalizeBootHttpfsPath('program-manifest.json', opts), '/boot/program-manifest.json');

  // Traversal and non-drive backslashes stay blocked even with gameBase set.
  assert.throws(() => normalizeBootHttpfsPath('G:\\..\\secret.rpf', opts), /Invalid HTTPFS path/);
  assert.throws(() => normalizeBootHttpfsPath('game/../secret.rpf', opts), /Invalid HTTPFS path/);
  assert.throws(() => normalizeBootHttpfsPath('foo\\bar.rpf', opts), /Invalid HTTPFS path/);

  // Omitting gameBase preserves dataBase fallback for *.rpf and game/...
  assert.equal(
    normalizeBootHttpfsPath('x64/update.rpf', { dataBase: '/data/', bootBase: '/boot/' }),
    '/data/x64/update.rpf',
  );
  assert.equal(
    normalizeBootHttpfsPath('game/x64/common.rpf', { dataBase: '/data/', bootBase: '/boot/' }),
    '/data/game/x64/common.rpf',
  );
});

test('forwardGpuWorkerMessage normalizes and forwards gpu-stats, missing-shaders, and frame-presented payloads', () => {
  const posted = [];
  const post = (msg) => posted.push(msg);

  // 1. wgpu_worker.js { id, stats } response
  const statsPayload = forwardGpuWorkerMessage(
    {
      id: 7,
      stats: {
        frameCount: 12,
        drawCalls: 48,
        computeDispatches: 4,
        packetCount: 99,
        vramBytes: 65536,
        shaderFallback: {
          totalLookups: 20,
          packHits: 15,
          fallbackHits: 4,
          builtinHits: 1,
          overflowMissingCount: 0,
        },
      },
      timestamp: 1700000001000,
    },
    post,
  );
  assert.equal(posted.length, 1);
  assert.deepEqual(statsPayload, posted[0]);
  assert.equal(statsPayload.type, 'gpu-telemetry');
  assert.deepEqual(statsPayload.frameStats, {
    framesPresented: 12,
    droppedFrames: 0,
    drawCalls: 48,
    computeDispatches: 4,
    packetCount: 99,
    vramBytes: 65536,
  });
  assert.deepEqual(statsPayload.fallbackStats, {
    totalLookups: 20,
    packHits: 15,
    fallbackHits: 4,
    builtinHits: 1,
    hitRatePct: 75,
    overflowMissingCount: 0,
  });
  assert.deepEqual(statsPayload.missingShaders, []);
  assert.equal(statsPayload.timestamp, 1700000001000);

  // 2. missing-shaders message with BigInt hash and untrusted extra fields stripped
  const missingPayload = forwardGpuWorkerMessage(
    {
      type: 'missing-shaders',
      missingShaders: [
        {
          hash: 0x1234abcd5678ef01n,
          stage: 2,
          hits: 9,
          firstSeenMs: 120.5,
          lastSeenMs: 340.25,
          hostPath: 'C:\\Users\\secret\\shader.dxil',
          rawBytes: new Uint8Array([1, 2, 3]),
        },
        {
          shaderHash: '0xdeadbeef',
          stage: 'vs',
          hitCount: 3,
        },
        {
          hash: 'not-a-valid-hash',
          stage: 'fragment',
          hits: 1,
        },
      ],
      shaderFallbackStats: {
        packHits: 8,
        fallbackHits: 2,
        builtinHits: 0,
      },
      timestamp: 1700000002000,
    },
    post,
  );
  assert.equal(posted.length, 2);
  assert.equal(missingPayload.type, 'gpu-telemetry');
  assert.equal(missingPayload.frameStats, null);
  assert.deepEqual(missingPayload.fallbackStats, {
    totalLookups: 10,
    packHits: 8,
    fallbackHits: 2,
    builtinHits: 0,
    hitRatePct: 80,
    overflowMissingCount: 0,
  });
  assert.deepEqual(missingPayload.missingShaders, [
    {
      hash: '0x1234abcd5678ef01',
      stage: 'fragment',
      hits: 9,
      firstSeenMs: 120.5,
      lastSeenMs: 340.25,
    },
    {
      hash: '0x00000000deadbeef',
      stage: 'vertex',
      hits: 3,
    },
  ]);

  // 3. frame-presented / { frame } payload
  const framePayload = forwardGpuWorkerMessage({ type: 'frame-presented', framesPresented: 5, droppedFrames: 1 }, post);
  assert.equal(posted.length, 3);
  assert.equal(framePayload.type, 'gpu-telemetry');
  assert.deepEqual(framePayload.frameStats, { framesPresented: 5, droppedFrames: 1 });
  assert.equal(framePayload.fallbackStats, null);
  assert.deepEqual(framePayload.missingShaders, []);
  assert.equal(typeof framePayload.timestamp, 'number');
  assert.ok(Number.isFinite(framePayload.timestamp));

  // 4. Direct { frame } from wgpu_worker Present
  const directFrame = forwardGpuWorkerMessage({ frame: 7 }, post);
  assert.equal(posted.length, 4);
  assert.deepEqual(directFrame.frameStats, { framesPresented: 7, droppedFrames: 0 });
});

test('forwardGpuWorkerMessage ignores unrelated and null-like messages without throwing', () => {
  const posted = [];
  const post = (msg) => posted.push(msg);

  for (const input of [
    null,
    undefined,
    0,
    'frame',
    [],
    {},
    { ready: true, adapter: 'WebGPU' },
    { error: 'adapter unavailable' },
    { ringAttached: true, capacityBytes: 65536 },
    { id: 1, destroyed: true },
    { id: 2, stats: null },
    { type: 'unrelated', value: 123 },
  ]) {
    assert.equal(forwardGpuWorkerMessage(input, post), null);
  }
  assert.equal(posted.length, 0);

  // Calling without postMessageFn or with a throwing postMessageFn must not throw.
  const withoutFn = forwardGpuWorkerMessage({ framesPresented: 2 });
  assert.equal(withoutFn.type, 'gpu-telemetry');
  assert.equal(withoutFn.frameStats.framesPresented, 2);

  const withThrowingFn = forwardGpuWorkerMessage({ framesPresented: 3 }, () => {
    throw new Error('channel closed');
  });
  assert.equal(withThrowingFn.frameStats.framesPresented, 3);
});

test('loader.js worker bridges gameBase/programManifestUrl init and on-demand gpu-telemetry requests', async () => {
  const mainMessages = [];
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
    }
  }

  const workerSelf = {
    onmessage: null,
    postMessage(msg) {
      mainMessages.push(msg);
    },
  };

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
    Date,
    setTimeout,
    clearTimeout,
    fetch: async () => ({ ok: false, status: 404 }),
    importScripts(scriptPath) {
      const resolved = path.resolve(repoRoot, 'web', scriptPath);
      vm.runInContext(fs.readFileSync(resolved, 'utf8'), context, { filename: resolved });
    },
  };
  sandbox.globalThis = sandbox;
  const context = vm.createContext(sandbox);

  const loaderPath = path.join(repoRoot, 'web/loader.js');
  vm.runInContext(fs.readFileSync(loaderPath, 'utf8'), context, { filename: loaderPath });

  // Requesting telemetry before gpuWorker exists must be a safe no-op.
  workerSelf.onmessage({ data: { type: 'request-gpu-telemetry', id: 1 } });
  assert.equal(mainMessages.length, 0);

  // Boot workers with gameBase and programManifestUrl.
  workerSelf.onmessage({
    data: {
      canvas: { id: 'c' },
      width: 1280,
      height: 720,
      dataBase: '/data/',
      bootBase: '/boot/',
      gameBase: 'http://127.0.0.1:8080/game/',
      programManifestUrl: '/boot/program-manifest.json',
      shaderPackUrl: '/shaders/fallback.wgsl.pack',
    },
  });

  const ioWorker = workersByScript.get('./workers/io_worker.js');
  const gpuWorker = workersByScript.get('./workers/wgpu_worker.js');
  assert.ok(ioWorker && gpuWorker);
  assert.equal(ioWorker.posted[0].gameBase, 'http://127.0.0.1:8080/game/');
  assert.equal(ioWorker.posted[0].programManifestUrl, '/boot/program-manifest.json');
  assert.equal(gpuWorker.posted[0].gameBase, 'http://127.0.0.1:8080/game/');
  assert.equal(gpuWorker.posted[0].programManifestUrl, '/boot/program-manifest.json');

  // On-demand telemetry request forwards stats + missingShaders queries to gpuWorker.
  workerSelf.onmessage({ data: { type: 'request-gpu-telemetry', id: 42, limit: 16 } });
  const statsQuery = gpuWorker.posted.find((m) => m.stats === true);
  const missingQuery = gpuWorker.posted.find((m) => m.missingShaders === true);
  assert.deepEqual({ ...statsQuery }, { stats: true, id: 42 });
  assert.deepEqual({ ...missingQuery }, { missingShaders: true, id: 42, limit: 16 });

  // Simulate gpuWorker responding with stats and missingShaders; loader forwards normalized gpu-telemetry.
  gpuWorker.onmessage({
    data: {
      id: 42,
      missingShaders: [{ hash: 0xabcdefn, stage: 'ps', hits: 4 }],
      shaderFallbackStats: { packHits: 6, fallbackHits: 4, builtinHits: 0 },
    },
  });
  assert.equal(mainMessages.length, 1);
  assert.equal(mainMessages[0].type, 'gpu-telemetry');
  assert.deepEqual(
    JSON.parse(JSON.stringify(mainMessages[0].missingShaders)),
    [{ hash: '0x0000000000abcdef', stage: 'fragment', hits: 4 }],
  );
  assert.equal(mainMessages[0].fallbackStats.hitRatePct, 60);

  // Ready/error lifecycle messages do not emit gpu-telemetry.
  gpuWorker.onmessage({ data: { ready: true, adapter: 'mock' } });
  gpuWorker.onmessage({ data: { error: 'test error' } });
  assert.equal(mainMessages.length, 1);
});

// Minimal valid Wasm32 binary importing env.memory and env.host_log(i32, i32) -> (),
// exporting `ping(ptr: i32, len: i32) -> ()` that calls host_log.
function buildMinimalHostLogWasm() {
  const sec = (id, payload) => [id, payload.length, ...payload];
  const typeSec = sec(0x01, [0x01, 0x60, 0x02, 0x7f, 0x7f, 0x00]);
  const importSec = sec(0x02, [
    0x02,
    // env.memory (min=1, max=1)
    0x03, 0x65, 0x6e, 0x76, 0x06, 0x6d, 0x65, 0x6d, 0x6f, 0x72, 0x79, 0x02, 0x01, 0x01, 0x01,
    // env.host_log (func type 0)
    0x03, 0x65, 0x6e, 0x76, 0x08, 0x68, 0x6f, 0x73, 0x74, 0x5f, 0x6c, 0x6f, 0x67, 0x00, 0x00,
  ]);
  const funcSec = sec(0x03, [0x01, 0x00]);
  const exportSec = sec(0x07, [0x01, 0x04, 0x70, 0x69, 0x6e, 0x67, 0x00, 0x01]);
  const codeSec = sec(0x0a, [0x01, 0x08, 0x00, 0x20, 0x00, 0x20, 0x01, 0x10, 0x00, 0x0b]);
  return new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    ...typeSec,
    ...importSec,
    ...funcSec,
    ...exportSec,
    ...codeSec,
  ]);
}

test('resolveRuntimeScriptUrl and buildRuntimeModuleConfig pin mainScriptUrlOrBlob to dist/runtime.js (never loader.js)', () => {
  assert.equal(
    resolveRuntimeScriptUrl(undefined, 'http://127.0.0.1:8080/loader.js'),
    'http://127.0.0.1:8080/dist/runtime.js',
  );
  assert.equal(
    resolveRuntimeScriptUrl(undefined, 'https://play.example.test/sub/loader.js?v=42'),
    'https://play.example.test/sub/dist/runtime.js',
  );
  assert.equal(resolveRuntimeScriptUrl(undefined, ''), 'dist/runtime.js');

  const absCfg = buildRuntimeModuleConfig({
    locationHref: 'http://127.0.0.1:8080/loader.js',
  });
  assert.equal(absCfg.mainScriptUrlOrBlob, 'http://127.0.0.1:8080/dist/runtime.js');
  assert.equal(absCfg.mainScriptUrlOrBlob.includes('loader.js'), false);
  assert.equal(absCfg.locateFile('runtime.wasm'), 'http://127.0.0.1:8080/dist/runtime.wasm');
  assert.equal(
    absCfg.locateFile('runtime.worker.js'),
    'http://127.0.0.1:8080/dist/runtime.worker.js',
  );
  assert.equal(
    absCfg.locateFile('https://cdn.example.test/runtime.wasm'),
    'https://cdn.example.test/runtime.wasm',
  );

  // Relative fallback when no worker locationHref is available.
  const relCfg = buildRuntimeModuleConfig({ locationHref: '' });
  assert.equal(relCfg.mainScriptUrlOrBlob, 'dist/runtime.js');
  assert.equal(relCfg.mainScriptUrlOrBlob.includes('loader.js'), false);
  assert.equal(relCfg.locateFile('runtime.wasm'), 'dist/runtime.wasm');
  assert.equal(relCfg.locateFile('runtime.worker.js'), 'dist/runtime.worker.js');
  assert.equal(relCfg.locateFile('./runtime.wasm'), 'dist/runtime.wasm');

  // Explicit runtimeScriptUrl + locateFileBase overrides.
  const customCfg = buildRuntimeModuleConfig({
    runtimeScriptUrl: 'http://127.0.0.1:9000/assets/dist/runtime.js',
    locateFileBase: 'http://127.0.0.1:9000/wasm-bin',
  });
  assert.equal(customCfg.mainScriptUrlOrBlob, 'http://127.0.0.1:9000/assets/dist/runtime.js');
  assert.equal(customCfg.locateFile('runtime.wasm'), 'http://127.0.0.1:9000/wasm-bin/runtime.wasm');
  assert.equal(
    customCfg.locateFile('runtime.worker.js'),
    'http://127.0.0.1:9000/wasm-bin/runtime.worker.js',
  );
});

test('buildRuntimeModuleConfig instantiateWasm merges all env.host_* imports and passes (instance, module)', async () => {
  const logs = [];
  const reports = [];
  const fetchedUrls = [];
  const hostImports = createHostImports({
    log: (msg) => logs.push(msg),
    report: (msg) => reports.push(msg),
  });

  const wasmBytes = buildMinimalHostLogWasm();
  let wasmErr = null;
  const cfg = buildRuntimeModuleConfig({
    runtimeScriptUrl: 'http://127.0.0.1:8080/dist/runtime.js',
    hostImports,
    reportFn: (msg) => reports.push(msg),
    onWasmError: (err) => {
      wasmErr = err;
    },
    fetchFn: async (url) => {
      fetchedUrls.push(url);
      return {
        ok: true,
        status: 200,
        headers: { get: () => 'application/octet-stream' },
        arrayBuffer: async () => wasmBytes.buffer.slice(0),
      };
    },
  });

  for (const name of [
    'host_log',
    'host_httpfs_open',
    'host_httpfs_read',
    'host_httpfs_close',
    'host_wgpu_submit',
  ]) {
    assert.equal(typeof cfg.env[name], 'function', `Expected cfg.env.${name}`);
  }
  assert.equal(typeof cfg.instantiateWasm, 'function');

  const memory = new WebAssembly.Memory({ initial: 1, maximum: 1 });
  const received = await new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(wasmErr || new Error(`instantiateWasm timed out: ${reports.join('; ')}`)),
      2000,
    );
    const syncRet = cfg.instantiateWasm(
      {
        env: {
          memory,
          // Simulate Emscripten's -sERROR_ON_UNDEFINED_SYMBOLS=0 stub being replaced by host_log.
          host_log: () => {
            throw new Error('stub host_log should be overridden by hostImports.env.host_log');
          },
          emscripten_keepalive_stub: () => 42,
        },
      },
      (instance, mod) => {
        clearTimeout(timer);
        resolve({ instance, mod });
      },
    );
    assert.deepEqual(syncRet, {});
  });

  assert.deepEqual(fetchedUrls, ['http://127.0.0.1:8080/dist/runtime.wasm']);
  assert.ok(received.instance instanceof WebAssembly.Instance);
  assert.ok(received.mod instanceof WebAssembly.Module);

  const text = 'hello from instantiateWasm';
  const encoded = new TextEncoder().encode(text);
  new Uint8Array(memory.buffer).set(encoded, 32);
  received.instance.exports.ping(32, encoded.byteLength);
  assert.deepEqual(logs, [text]);
});

test('buildRuntimeModuleConfig instantiateWasm reports fetch failures without unhandled rejection', async () => {
  const reports = [];
  const errors = [];
  const cfg = buildRuntimeModuleConfig({
    runtimeScriptUrl: 'http://127.0.0.1:8080/dist/runtime.js',
    reportFn: (msg) => reports.push(msg),
    onWasmError: (err) => errors.push(err),
    fetchFn: async () => ({ ok: false, status: 404 }),
  });

  const ret = cfg.instantiateWasm({ env: {} }, () => {
    throw new Error('receiveInstance must not be called on 404');
  });
  assert.deepEqual(ret, {});

  await new Promise((r) => setTimeout(r, 20));
  assert.equal(errors.length, 1);
  assert.match(String(errors[0]), /404/);
  assert.ok(reports.some((m) => m.includes('runtime wasm instantiation failed')));
});

test('loader.js worker passes resolved dist/runtime.js mainScriptUrlOrBlob and locateFile to createRuntimeModule', async () => {
  let receivedOpts = null;

  class MockBroadcastChannel {
    postMessage() {}
  }

  class MockWorker {
    constructor(script) {
      this.script = script;
      this.onmessage = null;
      this.posted = [];
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

  const workerSelf = {
    location: { href: 'http://127.0.0.1:8080/loader.js' },
    onmessage: null,
    postMessage() {},
  };

  const sandbox = {
    self: workerSelf,
    globalThis: null,
    URL,
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
    Date,
    setTimeout,
    clearTimeout,
    fetch: async (url) => {
      if (url === './dist/runtime.js') {
        return { ok: true, status: 200 };
      }
      return { ok: false, status: 404 };
    },
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
    data: { canvas: { id: 'c' }, width: 1280, height: 720, dataBase: '/data/' },
  });
  await new Promise((r) => setTimeout(r, 25));

  assert.ok(receivedOpts, 'Expected createRuntimeModule to be invoked');
  assert.equal(receivedOpts.mainScriptUrlOrBlob, 'http://127.0.0.1:8080/dist/runtime.js');
  assert.equal(receivedOpts.mainScriptUrlOrBlob.includes('loader.js'), false);
  assert.equal(receivedOpts.locateFile('runtime.wasm'), 'http://127.0.0.1:8080/dist/runtime.wasm');
  assert.equal(
    receivedOpts.locateFile('runtime.worker.js'),
    'http://127.0.0.1:8080/dist/runtime.worker.js',
  );
  assert.equal(typeof receivedOpts.instantiateWasm, 'function');
  assert.equal(typeof receivedOpts.env.host_log, 'function');
  assert.equal(typeof receivedOpts.env.host_httpfs_open, 'function');
  assert.equal(typeof receivedOpts.env.host_httpfs_read, 'function');
  assert.equal(typeof receivedOpts.env.host_httpfs_close, 'function');
  assert.equal(typeof receivedOpts.env.host_wgpu_submit, 'function');
});

test('createRuntimeModuleOptions forwards mainScriptUrlOrBlob and locateFile defaults', () => {
  const opts = createRuntimeModuleOptions({
    locationHref: 'http://127.0.0.1:8080/web/loader.js',
  });
  assert.equal(opts.mainScriptUrlOrBlob, 'http://127.0.0.1:8080/web/dist/runtime.js');
  assert.equal(opts.locateFile('runtime.wasm'), 'http://127.0.0.1:8080/web/dist/runtime.wasm');
});

test('isEmscriptenPthreadWorker detects em-pthread and em-pthread-<N> workers and rejects normal workers', () => {
  assert.equal(EMSCRIPTEN_PTHREAD_WORKER_NAME, 'em-pthread');
  assert.equal(PTHREAD_WGPU_CHANNEL_NAME, 'playgta6-wgpu-pthread');
  assert.equal(PTHREAD_HTTPFS_SAB_CHANNEL_NAME, 'playgta6-httpfs-sab');
  assert.deepEqual(HOST_ABI_IMPORT_NAMES, [
    'host_log',
    'host_httpfs_open',
    'host_httpfs_read',
    'host_httpfs_close',
    'host_wgpu_submit',
  ]);

  assert.equal(isEmscriptenPthreadWorkerName('em-pthread'), true);
  assert.equal(isEmscriptenPthreadWorkerName('em-pthread-1'), true);
  assert.equal(isEmscriptenPthreadWorkerName('em-pthread-2'), true);
  assert.equal(isEmscriptenPthreadWorkerName('em-pthread-42'), true);
  assert.equal(isEmscriptenPthreadWorkerName(''), false);
  assert.equal(isEmscriptenPthreadWorkerName('loader'), false);
  assert.equal(isEmscriptenPthreadWorkerName('em-pthreadx'), false);
  assert.equal(isEmscriptenPthreadWorkerName('my-em-pthread-1'), false);
  assert.equal(isEmscriptenPthreadWorkerName(123), false);
  assert.equal(isEmscriptenPthreadWorkerName({}), false);
  assert.equal(isEmscriptenPthreadWorkerName(null), false);
  assert.equal(isEmscriptenPthreadWorkerName(undefined), false);

  assert.equal(isEmscriptenPthreadWorker({ name: 'em-pthread' }), true);
  assert.equal(isEmscriptenPthreadWorker({ name: 'em-pthread-1' }), true);
  assert.equal(isEmscriptenPthreadWorker({ name: 'em-pthread-2' }), true);
  assert.equal(isEmscriptenPthreadWorker({ name: '' }), false);
  assert.equal(isEmscriptenPthreadWorker({ name: 'loader' }), false);
  assert.equal(isEmscriptenPthreadWorker({ name: 'em-pthreadx' }), false);
  assert.equal(isEmscriptenPthreadWorker({ name: 123 }), false);
  assert.equal(isEmscriptenPthreadWorker({}), false);
  assert.equal(isEmscriptenPthreadWorker(null), false);
  assert.equal(isEmscriptenPthreadWorker(undefined), false);
});

test('installPthreadHostImportShim replaces stub/missing ABI imports, binds memory, preserves prototype/instanceof, and restores cleanly', () => {
  const logs = [];
  const submitted = [];
  const hostImports = createHostImports({
    log: (msg) => logs.push(msg),
    wgpuSubmit: (bytes) => submitted.push(bytes.slice()),
    report: () => {},
  });

  // 1. Test against real WebAssembly.Module / WebAssembly.Instance (preserving prototype & instanceof).
  const realScope = { WebAssembly: { Instance: WebAssembly.Instance } };
  const origRealInstance = realScope.WebAssembly.Instance;
  const restoreReal = installPthreadHostImportShim(realScope, hostImports);
  assert.equal(realScope.WebAssembly.Instance.prototype, origRealInstance.prototype);

  const wasmModule = new WebAssembly.Module(buildMinimalHostLogWasm());
  const memory = new WebAssembly.Memory({ initial: 1, maximum: 1 });
  const abortingHostLogStub = () => {
    throw new Error('abort("missing function: host_log")');
  };
  abortingHostLogStub.stub = true;

  const realInstance = new realScope.WebAssembly.Instance(wasmModule, {
    env: {
      memory,
      host_log: abortingHostLogStub,
    },
  });
  assert.ok(realInstance instanceof realScope.WebAssembly.Instance);
  assert.ok(realInstance instanceof origRealInstance);

  const greeting = 'hello from em-pthread wasm';
  const encoded = new TextEncoder().encode(greeting);
  new Uint8Array(memory.buffer).set(encoded, 48);
  realInstance.exports.ping(48, encoded.byteLength);
  assert.deepEqual(logs, [greeting]);

  restoreReal();
  restoreReal(); // idempotent
  assert.equal(realScope.WebAssembly.Instance, origRealInstance);

  // 2. Test all 5 ABI imports (stub + missing), preservation of non-stub functions and non-ABI stubs.
  let capturedImports = null;
  let boundMemory = null;
  class MockWasmInstance {
    constructor(mod, imports) {
      this.mod = mod;
      this.imports = imports;
      capturedImports = imports;
    }
  }
  const mockHostImports = {
    env: {
      host_log: () => 'real-log',
      host_httpfs_open: () => 7,
      host_httpfs_read: () => 12n,
      host_httpfs_close: () => {},
      host_wgpu_submit: () => 'real-wgpu',
    },
    setMemory(mem) {
      boundMemory = mem;
    },
  };
  const mockScope = { WebAssembly: { Instance: MockWasmInstance } };
  const restoreMock = installPthreadHostImportShim(mockScope, mockHostImports);

  const makeStub = (name) => {
    const fn = () => {
      throw new Error(`missing function: ${name}`);
    };
    fn.stub = true;
    return fn;
  };
  const keepaliveFn = () => 99;
  const nonAbiStub = makeStub('some_other_emscripten_sym');
  const realHostClose = () => 'custom-non-stub-close';
  const mockMem = { buffer: new ArrayBuffer(64) };
  const wasiNs = { fd_write: () => 0 };

  const inst = new mockScope.WebAssembly.Instance('fake-mod', {
    env: {
      memory: mockMem,
      host_log: makeStub('host_log'),
      host_httpfs_open: makeStub('host_httpfs_open'),
      // host_httpfs_read intentionally missing
      host_httpfs_close: realHostClose, // non-stub must be preserved
      host_wgpu_submit: makeStub('host_wgpu_submit'),
      emscripten_keepalive: keepaliveFn,
      other_stub: nonAbiStub,
    },
    wasi_snapshot_preview1: wasiNs,
  });

  assert.ok(inst instanceof mockScope.WebAssembly.Instance);
  assert.ok(inst instanceof MockWasmInstance);
  assert.equal(boundMemory, mockMem);
  assert.equal(capturedImports.env.host_log, mockHostImports.env.host_log);
  assert.equal(capturedImports.env.host_httpfs_open, mockHostImports.env.host_httpfs_open);
  assert.equal(capturedImports.env.host_httpfs_read, mockHostImports.env.host_httpfs_read);
  assert.equal(capturedImports.env.host_httpfs_close, realHostClose);
  assert.equal(capturedImports.env.host_wgpu_submit, mockHostImports.env.host_wgpu_submit);
  assert.equal(capturedImports.env.emscripten_keepalive, keepaliveFn);
  assert.equal(capturedImports.env.other_stub, nonAbiStub);
  assert.equal(capturedImports.wasi_snapshot_preview1, wasiNs);

  restoreMock();
  assert.equal(mockScope.WebAssembly.Instance, MockWasmInstance);
});

// Minimal valid Wasm32 binary importing env.memory, env.host_log(i32, i32), and env.host_wgpu_submit(i32, i32),
// exporting `emit(logPtr, logLen, gpuPtr, gpuLen)`.
function buildMinimalPthreadLogAndWgpuWasm() {
  const sec = (id, payload) => [id, payload.length, ...payload];
  // type 0: (i32, i32) -> ()
  // type 1: (i32, i32, i32, i32) -> ()
  const typeSec = sec(0x01, [
    0x02,
    0x60, 0x02, 0x7f, 0x7f, 0x00,
    0x60, 0x04, 0x7f, 0x7f, 0x7f, 0x7f, 0x00,
  ]);
  const encStr = (s) => [s.length, ...Array.from(s, (ch) => ch.charCodeAt(0))];
  const importSec = sec(0x02, [
    0x03,
    // env.memory (min=1, max=1)
    ...encStr('env'), ...encStr('memory'), 0x02, 0x01, 0x01, 0x01,
    // env.host_log (func 0, type 0)
    ...encStr('env'), ...encStr('host_log'), 0x00, 0x00,
    // env.host_wgpu_submit (func 1, type 0)
    ...encStr('env'), ...encStr('host_wgpu_submit'), 0x00, 0x00,
  ]);
  const funcSec = sec(0x03, [0x01, 0x01]);
  const exportSec = sec(0x07, [0x01, ...encStr('emit'), 0x00, 0x02]);
  const codeSec = sec(0x0a, [
    0x01,
    0x0e,
    0x00,
    0x20, 0x00, 0x20, 0x01, 0x10, 0x00,
    0x20, 0x02, 0x20, 0x03, 0x10, 0x01,
    0x0b,
  ]);
  return new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    ...typeSec,
    ...importSec,
    ...funcSec,
    ...exportSec,
    ...codeSec,
  ]);
}

function createMockBroadcastBus({ rejectSabOnChannel = null } = {}) {
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
      if (
        rejectSabOnChannel === this.name &&
        data &&
        typeof data === 'object' &&
        typeof SharedArrayBuffer !== 'undefined' &&
        data.sab instanceof SharedArrayBuffer
      ) {
        throw new Error('DataCloneError: SharedArrayBuffer rejected');
      }
      postedLog.push({ channel: this.name, data });
      const peers = channelsByName.get(this.name);
      if (!peers) return;
      for (const peer of peers) {
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

test('loader.js in em-pthread worker loads dist/runtime.js without overwriting self.onmessage and relays host_log, host_wgpu_submit, and httpfs SAB', () => {
  const bus = createMockBroadcastBus();
  const loaderSource = fs.readFileSync(path.join(repoRoot, 'web/loader.js'), 'utf8');
  const loaderPath = path.join(repoRoot, 'web/loader.js');

  // --- 1. Boot the main coordinator loader.js in its own VM sandbox ---
  const coordWorkers = new Map();
  class CoordMockWorker {
    constructor(script) {
      this.script = script;
      this.onmessage = null;
      this.posted = [];
      coordWorkers.set(script, this);
    }
    postMessage(msg) {
      this.posted.push(msg);
    }
  }
  const coordSelf = {
    name: '',
    onmessage: null,
    postMessage() {},
  };
  const coordSandbox = {
    self: coordSelf,
    globalThis: null,
    BroadcastChannel: bus.BusBroadcastChannel,
    Worker: CoordMockWorker,
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

  // Initialize coordinator so gpuWorker and httpfsMailboxSab exist.
  coordSelf.onmessage({
    data: { canvas: { id: 'c' }, width: 1280, height: 720, dataBase: '/data/' },
  });
  const gpuWorker = coordWorkers.get('./workers/wgpu_worker.js');
  const ioWorker = coordWorkers.get('./workers/io_worker.js');
  assert.ok(gpuWorker && ioWorker);
  const attachMsg = ioWorker.posted.find((m) => m && m.attachMailbox === true);
  assert.ok(attachMsg && attachMsg.sab instanceof SharedArrayBuffer);

  // --- 2. Boot an em-pthread worker running web/loader.js in a second VM sandbox ---
  const pthreadImported = [];
  const pthreadWorkerSpawns = [];
  let emscriptenOnMessage = null;
  let pthreadInstance = null;

  const pthreadSelf = {
    name: 'em-pthread',
    onmessage: null,
    postMessage() {},
  };

  // Use a per-sandbox WebAssembly namespace so we don't mutate Node's global WebAssembly.Instance.
  const pthreadWebAssembly = {
    Module: WebAssembly.Module,
    Memory: WebAssembly.Memory,
    Instance: WebAssembly.Instance,
  };

  const pthreadSandbox = {
    self: pthreadSelf,
    globalThis: null,
    WebAssembly: pthreadWebAssembly,
    BroadcastChannel: bus.BusBroadcastChannel,
    Worker: class ForbiddenWorker {
      constructor(script) {
        pthreadWorkerSpawns.push(script);
      }
    },
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
    importScripts(scriptPath) {
      pthreadImported.push(scriptPath);
      if (scriptPath === './dist/runtime.js') {
        // Simulate Emscripten 7.0.1 pthread bootstrap installing self.onmessage = handleMessage.
        emscriptenOnMessage = (ev) => {
          const { cmd, wasmModule, wasmMemory } = ev.data;
          if (cmd !== 'load') return;
          const makeStub = (name) => {
            const fn = () => {
              throw new Error(`abort("missing function: ${name}")`);
            };
            fn.stub = true;
            return fn;
          };
          pthreadInstance = new pthreadSandbox.WebAssembly.Instance(wasmModule, {
            env: {
              memory: wasmMemory,
              host_log: makeStub('host_log'),
              host_httpfs_open: makeStub('host_httpfs_open'),
              host_httpfs_read: makeStub('host_httpfs_read'),
              host_httpfs_close: makeStub('host_httpfs_close'),
              host_wgpu_submit: makeStub('host_wgpu_submit'),
            },
          });
        };
        pthreadSelf.onmessage = emscriptenOnMessage;
        return;
      }
      const resolved = path.resolve(repoRoot, 'web', scriptPath);
      vm.runInContext(fs.readFileSync(resolved, 'utf8'), pthreadContext, { filename: resolved });
    },
  };
  pthreadSandbox.globalThis = pthreadSandbox;
  const pthreadContext = vm.createContext(pthreadSandbox);
  vm.runInContext(loaderSource, pthreadContext, { filename: loaderPath });

  // Verify pthread fast-path: loaded httpfs_mailbox.js + dist/runtime.js, did NOT spawn workers,
  // and did NOT overwrite Emscripten's self.onmessage!
  assert.deepEqual(pthreadImported, ['./workers/httpfs_mailbox.js', './dist/runtime.js']);
  assert.equal(pthreadWorkerSpawns.length, 0);
  assert.equal(pthreadSelf.onmessage, emscriptenOnMessage);

  // Verify httpfs SAB handshake happened between pthread and coordinator.
  const sabReply = bus.postedLog.find(
    (entry) => entry.channel === PTHREAD_HTTPFS_SAB_CHANNEL_NAME && entry.data.type === 'httpfs-sab',
  );
  assert.ok(sabReply && sabReply.data.sab === attachMsg.sab);

  // Simulate Emscripten PThread.loadWasmModuleToWorker delivering { cmd: 'load', wasmModule, wasmMemory }.
  const wasmModule = new WebAssembly.Module(buildMinimalPthreadLogAndWgpuWasm());
  const wasmMemory = new WebAssembly.Memory({ initial: 1, maximum: 1 });
  pthreadSelf.onmessage({ data: { cmd: 'load', wasmModule, wasmMemory } });
  assert.ok(pthreadInstance instanceof WebAssembly.Instance);

  // Invoke Wasm exports that call host_log and host_wgpu_submit (with a Present frame).
  const msgBytes = new TextEncoder().encode('d3d12 render thread alive');
  const frameBytes = new Uint8Array(encodeTestFramePacket(0.25, 0.5, 0.75, 1.0));
  const memU8 = new Uint8Array(wasmMemory.buffer);
  memU8.set(msgBytes, 16);
  memU8.set(frameBytes, 128);

  pthreadInstance.exports.emit(16, msgBytes.byteLength, 128, frameBytes.byteLength);
  // Submit a second frame to confirm FIRST_PRESENT is emitted only once.
  pthreadInstance.exports.emit(16, msgBytes.byteLength, 128, frameBytes.byteLength);

  // Check host_log reached game-progress channel.
  const progressLabels = bus.postedLog
    .filter((entry) => entry.channel === 'game-progress' && entry.data && entry.data.label)
    .map((entry) => entry.data.label);
  assert.ok(progressLabels.includes('[wasm] d3d12 render thread alive'));

  // Check FIRST_PRESENT was emitted exactly once by the coordinator.
  const firstPresentEvents = progressLabels.filter((label) => label === '[boot] FIRST_PRESENT');
  assert.equal(firstPresentEvents.length, 1);

  // Check GPU command batches were forwarded via playgta6-wgpu-pthread straight to gpuWorker!
  const gpuCmdPosts = gpuWorker.posted.filter((m) => m && m.cmdBuffer instanceof ArrayBuffer);
  assert.equal(gpuCmdPosts.length, 2);
  assert.deepEqual(new Uint8Array(gpuCmdPosts[0].cmdBuffer), frameBytes);
  assert.deepEqual(new Uint8Array(gpuCmdPosts[1].cmdBuffer), frameBytes);
});

test('bootstrapEmscriptenPthreadWorker degrades safely when BroadcastChannel rejects SharedArrayBuffer and supports host_httpfs_open via SAB', () => {
  const { createHttpfsMailboxBuffer, HttpfsMailboxClient, HttpfsOffsets, HttpfsStatus } = require(
    '../web/workers/httpfs_mailbox.js',
  );

  // 1. Degrade safely if BroadcastChannel throws on SAB.
  const rejectingBus = createMockBroadcastBus({ rejectSabOnChannel: PTHREAD_HTTPFS_SAB_CHANNEL_NAME });
  const mockScope = {
    name: 'em-pthread',
    WebAssembly: { Instance: WebAssembly.Instance },
  };
  let importedUrl = null;
  const handle = bootstrapEmscriptenPthreadWorker({
    scope: mockScope,
    importScriptsFn: (url) => {
      importedUrl = url;
    },
    BroadcastChannelCtor: rejectingBus.BusBroadcastChannel,
    httpfsMailboxApi: { HttpfsMailboxClient },
  });
  assert.equal(importedUrl, './dist/runtime.js');

  // Even if coordinator fails to post SAB over BroadcastChannel, host_httpfs_open returns -1 cleanly.
  const mem = new WebAssembly.Memory({ initial: 1, maximum: 1 });
  handle.hostImports.setMemory(mem);
  const pathBytes = new TextEncoder().encode('boot/guest.pgim');
  new Uint8Array(mem.buffer).set(pathBytes, 32);
  assert.equal(handle.hostImports.env.host_httpfs_open(32n, BigInt(pathBytes.byteLength)), -1);

  // 2. When SAB arrives over BroadcastChannel, host_httpfs_open uses HttpfsMailboxClient(sab).
  const sab = createHttpfsMailboxBuffer(1024);
  const clientApi = {
    HttpfsMailboxClient: class ResponderClient extends HttpfsMailboxClient {
      constructor(buf) {
        super(buf, (s) => {
          const header = new DataView(s, 0, 40);
          const statusView = new Int32Array(s, 0, 1);
          header.setBigInt64(HttpfsOffsets.RESULT, 42n, true);
          Atomics.store(statusView, 0, HttpfsStatus.ResponseReady);
          Atomics.notify(statusView, 0, 1);
        });
      }
    },
  };
  const okBus = createMockBroadcastBus();
  const handle2 = bootstrapEmscriptenPthreadWorker({
    scope: { name: 'em-pthread', WebAssembly: { Instance: WebAssembly.Instance } },
    importScriptsFn: () => {},
    BroadcastChannelCtor: okBus.BusBroadcastChannel,
    httpfsMailboxApi: clientApi,
  });
  handle2.hostImports.setMemory(mem);
  const sender = new okBus.BusBroadcastChannel(PTHREAD_HTTPFS_SAB_CHANNEL_NAME);
  sender.postMessage({ type: 'httpfs-sab', sab });
  assert.equal(handle2.hostImports.env.host_httpfs_open(32n, BigInt(pathBytes.byteLength)), 42);

  handle.cleanup();
  handle2.cleanup();
  sender.close();
});

// Minimal valid Wasm32 SIDE_MODULE-style binary importing:
// - env.memory (min=1, max=1)
// - env.__memory_base (i32 immutable global)
// - env.__table_base (i32 immutable global)
// - env.__indirect_function_table (funcref table min=1)
// - GOT.mem.sym_offset (i32 mutable global)
// - GOT.func.sym_idx (i32 mutable global)
// - optionally env.host_log(i32, i32) -> ()
// Exports:
// - read_got_sum() -> i32  (returns __memory_base + __table_base + GOT.mem.sym_offset + GOT.func.sym_idx)
// - when withHostLog=true: log_from_base(len: i32) -> () (calls host_log(__memory_base + GOT.mem.sym_offset, len))
function buildMinimalSideModuleWasm({ withHostLog = false } = {}) {
  const uleb128 = (value) => {
    const out = [];
    let v = value >>> 0;
    do {
      let b = v & 0x7f;
      v >>>= 7;
      if (v !== 0) b |= 0x80;
      out.push(b);
    } while (v !== 0);
    return out;
  };
  const sec = (id, payload) => [id, ...uleb128(payload.length), ...payload];
  const encStr = (s) => [...uleb128(s.length), ...Array.from(s, (ch) => ch.charCodeAt(0))];

  const typeEntries = [
    [0x60, 0x00, 0x01, 0x7f], // type 0: () -> i32
  ];
  if (withHostLog) {
    typeEntries.push([0x60, 0x02, 0x7f, 0x7f, 0x00]); // type 1: (i32, i32) -> ()
    typeEntries.push([0x60, 0x01, 0x7f, 0x00]); // type 2: (i32) -> ()
  }
  const typeSec = sec(0x01, [typeEntries.length, ...typeEntries.flat()]);

  const importEntries = [
    // env.memory (min=1, max=1)
    [...encStr('env'), ...encStr('memory'), 0x02, 0x01, 0x01, 0x01],
    // env.__memory_base (global 0: i32 immutable)
    [...encStr('env'), ...encStr('__memory_base'), 0x03, 0x7f, 0x00],
    // env.__table_base (global 1: i32 immutable)
    [...encStr('env'), ...encStr('__table_base'), 0x03, 0x7f, 0x00],
    // env.__indirect_function_table (table 0: funcref, min=1)
    [...encStr('env'), ...encStr('__indirect_function_table'), 0x01, 0x70, 0x00, 0x01],
    // GOT.mem.sym_offset (global 2: i32 mutable)
    [...encStr('GOT.mem'), ...encStr('sym_offset'), 0x03, 0x7f, 0x01],
    // GOT.func.sym_idx (global 3: i32 mutable)
    [...encStr('GOT.func'), ...encStr('sym_idx'), 0x03, 0x7f, 0x01],
  ];
  if (withHostLog) {
    // env.host_log (func 0, type 1)
    importEntries.push([...encStr('env'), ...encStr('host_log'), 0x00, 0x01]);
  }
  const importSec = sec(0x02, [importEntries.length, ...importEntries.flat()]);

  const definedFuncTypes = withHostLog ? [0x00, 0x02] : [0x00];
  const funcSec = sec(0x03, [definedFuncTypes.length, ...definedFuncTypes]);

  const readGotSumFuncIdx = withHostLog ? 1 : 0;
  const exportEntries = [[...encStr('read_got_sum'), 0x00, readGotSumFuncIdx]];
  if (withHostLog) {
    exportEntries.push([...encStr('log_from_base'), 0x00, 2]);
  }
  const exportSec = sec(0x07, [exportEntries.length, ...exportEntries.flat()]);

  // read_got_sum: global.get 0 + global.get 1 + global.get 2 + global.get 3
  const readGotSumBody = [
    0x00, // 0 locals
    0x23, 0x00,
    0x23, 0x01, 0x6a,
    0x23, 0x02, 0x6a,
    0x23, 0x03, 0x6a,
    0x0b,
  ];
  const codeBodies = [[readGotSumBody.length, ...readGotSumBody]];
  if (withHostLog) {
    // log_from_base(len): call host_log(global.get 0 + global.get 2, local.get 0)
    const logFromBaseBody = [
      0x00, // 0 locals
      0x23, 0x00,
      0x23, 0x02, 0x6a,
      0x20, 0x00,
      0x10, 0x00,
      0x0b,
    ];
    codeBodies.push([logFromBaseBody.length, ...logFromBaseBody]);
  }
  const codeSec = sec(0x0a, [codeBodies.length, ...codeBodies.flat()]);

  return new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    ...typeSec,
    ...importSec,
    ...funcSec,
    ...exportSec,
    ...codeSec,
  ]);
}

test('resolveSideModuleUrl and buildRuntimeModuleConfig.locateFile resolve guest_part_*.wasm cleanly and log once per side module', () => {
  // Direct resolveSideModuleUrl helper checks
  assert.equal(
    resolveSideModuleUrl('guest_part_0.wasm', 'http://localhost:8080/dist/runtime.js'),
    'http://localhost:8080/dist/guest_part_0.wasm',
  );
  assert.equal(
    resolveSideModuleUrl('./guest_part_1.wasm', 'http://localhost:8080/dist/runtime.js'),
    'http://localhost:8080/dist/guest_part_1.wasm',
  );
  assert.equal(
    resolveSideModuleUrl('dist/guest_part_2.wasm', 'http://localhost:8080/dist/runtime.js'),
    'http://localhost:8080/dist/guest_part_2.wasm',
  );
  assert.equal(
    resolveSideModuleUrl('/dist/guest_part_3.wasm', 'http://localhost:8080/dist/runtime.js'),
    'http://localhost:8080/dist/guest_part_3.wasm',
  );
  assert.equal(
    resolveSideModuleUrl('guest_part_0.wasm', 'dist/runtime.js'),
    'dist/guest_part_0.wasm',
  );
  assert.equal(
    resolveSideModuleUrl('dist/guest_part_0.wasm', 'dist/runtime.js'),
    'dist/guest_part_0.wasm',
  );
  assert.equal(resolveSideModuleUrl('guest_part_0.wasm', ''), 'dist/guest_part_0.wasm');
  assert.equal(
    resolveSideModuleUrl('https://cdn.example.test/guest_part_0.wasm', 'dist/runtime.js'),
    'https://cdn.example.test/guest_part_0.wasm',
  );

  // buildRuntimeModuleConfig locateFile with absolute locationHref + once-per-module reporting
  const reports = [];
  const absCfg = buildRuntimeModuleConfig({
    locationHref: 'http://localhost:8080/loader.js',
    reportFn: (msg) => reports.push(msg),
  });
  assert.equal(
    absCfg.locateFile('guest_part_0.wasm'),
    'http://localhost:8080/dist/guest_part_0.wasm',
  );
  assert.equal(
    absCfg.locateFile('./guest_part_0.wasm'),
    'http://localhost:8080/dist/guest_part_0.wasm',
  );
  assert.equal(
    absCfg.locateFile('http://localhost:8080/dist/guest_part_0.wasm'),
    'http://localhost:8080/dist/guest_part_0.wasm',
  );
  assert.equal(
    absCfg.locateFile('guest_part_1.wasm'),
    'http://localhost:8080/dist/guest_part_1.wasm',
  );
  assert.equal(
    absCfg.locateFile('runtime.wasm'),
    'http://localhost:8080/dist/runtime.wasm',
  );
  assert.equal(
    absCfg.locateFile('runtime.worker.js'),
    'http://localhost:8080/dist/runtime.worker.js',
  );
  // Each side module is reported exactly once; runtime.wasm/runtime.worker.js are not logged as side modules.
  assert.deepEqual(reports, [
    'Loading Wasm side module (guest_part_0.wasm)...',
    'Loading Wasm side module (guest_part_1.wasm)...',
  ]);
  assert.equal('dynamicLibraries' in absCfg, false);
  assert.equal('sideModuleUrls' in absCfg, false);

  // Relative fallback when no locationHref is present
  const relReports = [];
  const relCfg = buildRuntimeModuleConfig({
    locationHref: '',
    reportFn: (msg) => relReports.push(msg),
  });
  assert.equal(relCfg.locateFile('guest_part_0.wasm'), 'dist/guest_part_0.wasm');
  assert.equal(relCfg.locateFile('dist/guest_part_0.wasm'), 'dist/guest_part_0.wasm');
  assert.deepEqual(relReports, ['Loading Wasm side module (guest_part_0.wasm)...']);
});

test('buildRuntimeModuleConfig sanitizes and forwards optional sideModuleUrls and dynamicLibraries', () => {
  assert.deepEqual(
    sanitizeSideModuleUrls([
      'guest_part_0.wasm',
      '  guest_part_1.wasm  ',
      '../escape.wasm',
      'foo\\bar.wasm',
      'bad.js',
      'javascript:alert(1).wasm',
      'https://cdn.example.test/guest_part_2.wasm',
      'guest_part_0.wasm',
      '',
      null,
    ]),
    [
      'guest_part_0.wasm',
      'guest_part_1.wasm',
      'https://cdn.example.test/guest_part_2.wasm',
    ],
  );

  const reports = [];
  const cfg = buildRuntimeModuleConfig({
    locationHref: 'http://localhost:8080/loader.js',
    reportFn: (msg) => reports.push(msg),
    sideModuleUrls: [
      'guest_part_0.wasm',
      'guest_part_1.wasm',
      '../blocked.wasm',
      'guest_part_0.wasm',
    ],
  });
  assert.deepEqual(cfg.dynamicLibraries, ['guest_part_0.wasm', 'guest_part_1.wasm']);
  assert.deepEqual(cfg.sideModuleUrls, [
    'http://localhost:8080/dist/guest_part_0.wasm',
    'http://localhost:8080/dist/guest_part_1.wasm',
  ]);

  // Explicit custom dynamicLibraries entry also logs once when located.
  const customReports = [];
  const customCfg = buildRuntimeModuleConfig({
    locationHref: 'http://localhost:8080/loader.js',
    reportFn: (msg) => customReports.push(msg),
    dynamicLibraries: ['custom_chunk.wasm'],
  });
  assert.deepEqual(customCfg.dynamicLibraries, ['custom_chunk.wasm']);
  assert.deepEqual(customCfg.sideModuleUrls, ['http://localhost:8080/dist/custom_chunk.wasm']);
  assert.equal(
    customCfg.locateFile('custom_chunk.wasm'),
    'http://localhost:8080/dist/custom_chunk.wasm',
  );
  assert.equal(
    customCfg.locateFile('custom_chunk.wasm'),
    'http://localhost:8080/dist/custom_chunk.wasm',
  );
  assert.deepEqual(customReports, ['Loading Wasm side module (custom_chunk.wasm)...']);
});

test('installPthreadHostImportShim preserves GOT.mem, GOT.func, __memory_base, __table_base, __indirect_function_table, Proxy env, and non-stub functions for side modules', async () => {
  // 1. Mock side module with Proxy env (Emscripten dylink style), GOT.mem, GOT.func, and stub host_*
  let capturedImports = null;
  let boundMemoryCount = 0;
  let lastBoundMemory = null;
  class MockWasmInstance {
    constructor(mod, imports) {
      this.mod = mod;
      this.imports = imports;
      capturedImports = imports;
    }
  }

  const mockHostImports = {
    env: {
      host_log: () => 'real-host-log',
      host_httpfs_open: () => 5,
      host_httpfs_read: () => 16n,
      host_httpfs_close: () => {},
      host_wgpu_submit: () => 'real-wgpu-submit',
    },
    setMemory(mem) {
      boundMemoryCount += 1;
      lastBoundMemory = mem;
    },
  };
  const mockScope = { WebAssembly: { Instance: MockWasmInstance } };
  const restoreMock = installPthreadHostImportShim(mockScope, mockHostImports);

  const makeStub = (name) => {
    const fn = () => {
      throw new Error(`missing function: ${name}`);
    };
    fn.stub = true;
    return fn;
  };
  const sharedMem = new WebAssembly.Memory({ initial: 1, maximum: 1 });
  const memoryBase = new WebAssembly.Global({ value: 'i32', mutable: false }, 64);
  const tableBase = new WebAssembly.Global({ value: 'i32', mutable: false }, 8);
  const indirectTable = new WebAssembly.Table({ initial: 4, element: 'anyfunc' });
  const liftedFn = () => 0x140001000;
  const dynamicResolvedFn = () => 'resolved-via-dylink-proxy';
  const customHostClose = () => 'real-non-stub-close';
  const gotMemProxy = new Proxy(
    { sym_offset: new WebAssembly.Global({ value: 'i32', mutable: true }, 16) },
    { get: (t, p) => t[p] },
  );
  const gotFuncProxy = new Proxy(
    { sym_idx: new WebAssembly.Global({ value: 'i32', mutable: true }, 3) },
    { get: (t, p) => t[p] },
  );

  const baseEnvTarget = {
    memory: sharedMem,
    __memory_base: memoryBase,
    __table_base: tableBase,
    __indirect_function_table: indirectTable,
    lifted_fn: liftedFn,
    host_log: makeStub('host_log'),
    host_wgpu_submit: makeStub('host_wgpu_submit'),
    host_httpfs_close: customHostClose,
  };
  const dylinkEnvProxy = new Proxy(baseEnvTarget, {
    get(target, prop) {
      if (prop in target) {
        return target[prop];
      }
      if (prop === 'lazy_lifted_0x140002000') {
        return dynamicResolvedFn;
      }
      return undefined;
    },
  });

  new mockScope.WebAssembly.Instance('fake-side-mod', {
    env: dylinkEnvProxy,
    'GOT.mem': gotMemProxy,
    'GOT.func': gotFuncProxy,
  });

  assert.equal(boundMemoryCount, 1);
  assert.equal(lastBoundMemory, sharedMem);
  assert.equal(capturedImports['GOT.mem'], gotMemProxy);
  assert.equal(capturedImports['GOT.func'], gotFuncProxy);
  assert.equal(capturedImports.env.memory, sharedMem);
  assert.equal(capturedImports.env.__memory_base, memoryBase);
  assert.equal(capturedImports.env.__table_base, tableBase);
  assert.equal(capturedImports.env.__indirect_function_table, indirectTable);
  assert.equal(capturedImports.env.lifted_fn, liftedFn);
  assert.equal(capturedImports.env.lazy_lifted_0x140002000, dynamicResolvedFn);
  assert.equal(capturedImports.env.host_log, mockHostImports.env.host_log);
  assert.equal(capturedImports.env.host_wgpu_submit, mockHostImports.env.host_wgpu_submit);
  assert.equal(capturedImports.env.host_httpfs_close, customHostClose);

  // Calling with no env.memory must NOT call setMemory again (preserve previously bound memory).
  new mockScope.WebAssembly.Instance('fake-side-mod-no-mem', {
    env: { __memory_base: memoryBase },
    'GOT.mem': gotMemProxy,
    'GOT.func': gotFuncProxy,
  });
  assert.equal(boundMemoryCount, 1);
  assert.equal(lastBoundMemory, sharedMem);

  restoreMock();
  assert.equal(mockScope.WebAssembly.Instance, MockWasmInstance);

  // 2. Real compiled WebAssembly.Module side modules (both without and with host_log)
  const logs = [];
  const realHostImports = createHostImports({
    log: (msg) => logs.push(msg),
    report: () => {},
  });

  let sawRawImportsOnNoHostMod = null;
  const SpyInstance = new Proxy(WebAssembly.Instance, {
    construct(target, args, newTarget) {
      sawRawImportsOnNoHostMod = args[1];
      return Reflect.construct(target, args, newTarget === SpyInstance ? target : newTarget);
    },
  });
  const realScope = {
    WebAssembly: {
      Module: WebAssembly.Module,
      Instance: SpyInstance,
      instantiate: WebAssembly.instantiate,
    },
  };
  const restoreReal = installPthreadHostImportShim(realScope, realHostImports);

  // 2a. Side module that imports NO host_* functions: imports & imports.env passed completely untouched!
  const noHostSideMod = new WebAssembly.Module(buildMinimalSideModuleWasm({ withHostLog: false }));
  const noHostImports = {
    env: {
      memory: sharedMem,
      __memory_base: memoryBase,
      __table_base: tableBase,
      __indirect_function_table: indirectTable,
    },
    'GOT.mem': gotMemProxy,
    'GOT.func': gotFuncProxy,
  };
  const noHostInst = new realScope.WebAssembly.Instance(noHostSideMod, noHostImports);
  assert.equal(sawRawImportsOnNoHostMod, noHostImports);
  assert.equal(sawRawImportsOnNoHostMod.env, noHostImports.env);
  assert.equal(noHostInst.exports.read_got_sum(), 64 + 8 + 16 + 3);

  // 2b. Side module importing GOT.mem, GOT.func, __memory_base, __indirect_function_table AND env.host_log
  const withHostSideMod = new WebAssembly.Module(buildMinimalSideModuleWasm({ withHostLog: true }));
  const msg = 'hello from side module';
  const encoded = new TextEncoder().encode(msg);
  new Uint8Array(sharedMem.buffer).set(encoded, 64 + 16);

  const sideEnvProxy = new Proxy(
    {
      memory: sharedMem,
      __memory_base: memoryBase,
      __table_base: tableBase,
      __indirect_function_table: indirectTable,
      host_log: makeStub('host_log'),
    },
    { get: (t, p) => t[p] },
  );
  const withHostInst = new realScope.WebAssembly.Instance(withHostSideMod, {
    env: sideEnvProxy,
    'GOT.mem': gotMemProxy,
    'GOT.func': gotFuncProxy,
  });
  assert.equal(withHostInst.exports.read_got_sum(), 64 + 8 + 16 + 3);
  withHostInst.exports.log_from_base(encoded.byteLength);
  assert.deepEqual(logs, [msg]);

  // 2c. Async WebAssembly.instantiate path also patches stub host_log while preserving GOT.*
  const asyncInst = await realScope.WebAssembly.instantiate(withHostSideMod, {
    env: sideEnvProxy,
    'GOT.mem': gotMemProxy,
    'GOT.func': gotFuncProxy,
  });
  assert.equal(asyncInst.exports.read_got_sum(), 64 + 8 + 16 + 3);
  asyncInst.exports.log_from_base(encoded.byteLength);
  assert.deepEqual(logs, [msg, msg]);

  restoreReal();
  assert.equal(realScope.WebAssembly.Instance, SpyInstance);
  assert.equal(realScope.WebAssembly.instantiate, WebAssembly.instantiate);
});

test('bootstrapEmscriptenPthreadWorker configures dist/ locateFile for side modules inside em-pthread workers', () => {
  const bus = createMockBroadcastBus();
  const mockScope = {
    name: 'em-pthread',
    location: { href: 'http://localhost:8080/loader.js' },
    WebAssembly: { Instance: WebAssembly.Instance },
  };
  let importedUrl = null;
  const handle = bootstrapEmscriptenPthreadWorker({
    scope: mockScope,
    importScriptsFn: (url) => {
      importedUrl = url;
    },
    BroadcastChannelCtor: bus.BusBroadcastChannel,
  });

  assert.equal(importedUrl, './dist/runtime.js');
  assert.equal(handle.runtimeScriptUrl, 'http://localhost:8080/dist/runtime.js');
  assert.equal(
    handle.locateFile('guest_part_0.wasm'),
    'http://localhost:8080/dist/guest_part_0.wasm',
  );
  assert.equal(
    mockScope.Module.locateFile('guest_part_1.wasm'),
    'http://localhost:8080/dist/guest_part_1.wasm',
  );
  // Second lookup of guest_part_0.wasm must not duplicate progress logs.
  assert.equal(
    mockScope.Module.locateFile('./guest_part_0.wasm'),
    'http://localhost:8080/dist/guest_part_0.wasm',
  );

  const progressLabels = bus.postedLog
    .filter((entry) => entry.channel === 'game-progress' && entry.data && entry.data.label)
    .map((entry) => entry.data.label);
  assert.deepEqual(progressLabels, [
    'Loading Wasm side module (guest_part_0.wasm)...',
    'Loading Wasm side module (guest_part_1.wasm)...',
  ]);

  handle.cleanup();
  assert.equal('Module' in mockScope, false);

  // Fallback without scope.location resolves cleanly to relative dist/
  const relHandle = bootstrapEmscriptenPthreadWorker({
    scope: { name: 'em-pthread', WebAssembly: { Instance: WebAssembly.Instance } },
    importScriptsFn: () => {},
    BroadcastChannelCtor: bus.BusBroadcastChannel,
  });
  assert.equal(relHandle.locateFile('guest_part_0.wasm'), 'dist/guest_part_0.wasm');
  relHandle.cleanup();
});

test('loader.js coordinator worker forwards sideModuleUrls to createRuntimeModule', async () => {
  let receivedOpts = null;

  class MockBroadcastChannel {
    postMessage() {}
  }

  class MockWorker {
    constructor(script) {
      this.script = script;
      this.onmessage = null;
      this.posted = [];
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

  const workerSelf = {
    location: { href: 'http://localhost:8080/loader.js' },
    onmessage: null,
    postMessage() {},
  };

  const sandbox = {
    self: workerSelf,
    globalThis: null,
    URL,
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
    Date,
    setTimeout,
    clearTimeout,
    fetch: async (url) => {
      if (url === './dist/runtime.js') {
        return { ok: true, status: 200 };
      }
      return { ok: false, status: 404 };
    },
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
    data: {
      canvas: { id: 'c' },
      width: 1280,
      height: 720,
      dataBase: '/data/',
      sideModuleUrls: ['guest_part_0.wasm', 'guest_part_1.wasm', 'guest_part_2.wasm', 'guest_part_3.wasm'],
    },
  });
  await new Promise((r) => setTimeout(r, 25));

  assert.ok(receivedOpts, 'Expected createRuntimeModule to be invoked');
  assert.deepEqual(Array.from(receivedOpts.dynamicLibraries), [
    'guest_part_0.wasm',
    'guest_part_1.wasm',
    'guest_part_2.wasm',
    'guest_part_3.wasm',
  ]);
  assert.deepEqual(Array.from(receivedOpts.sideModuleUrls), [
    'http://localhost:8080/dist/guest_part_0.wasm',
    'http://localhost:8080/dist/guest_part_1.wasm',
    'http://localhost:8080/dist/guest_part_2.wasm',
    'http://localhost:8080/dist/guest_part_3.wasm',
  ]);
  assert.equal(
    receivedOpts.locateFile('guest_part_0.wasm'),
    'http://localhost:8080/dist/guest_part_0.wasm',
  );
});

test('W77 coordinator Worker Proxy redirects em-pthread-<N> workers to self.location.href and posts __playgta6_httpfs_sab', () => {
  const loaderPath = path.join(repoRoot, 'web/loader.js');
  const loaderSource = fs.readFileSync(loaderPath, 'utf8');
  const spawnedWorkers = [];
  let throwOnNextPost = false;

  class MockBroadcastChannel {
    postMessage() {}
  }

  class MockWorker {
    constructor(script, opts) {
      this.script = script;
      this.opts = opts;
      this.onmessage = null;
      this.posted = [];
      spawnedWorkers.push(this);
    }
    postMessage(msg) {
      if (throwOnNextPost) {
        throwOnNextPost = false;
        throw new Error('simulated postMessage failure');
      }
      this.posted.push(msg);
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
    Date,
    setTimeout,
    clearTimeout,
    fetch: async () => ({ ok: false, status: 404 }),
    importScripts(scriptPath) {
      const resolved = path.resolve(repoRoot, 'web', scriptPath);
      vm.runInContext(fs.readFileSync(resolved, 'utf8'), context, { filename: resolved });
    },
  };
  sandbox.globalThis = sandbox;
  const context = vm.createContext(sandbox);
  vm.runInContext(loaderSource, context, { filename: loaderPath });

  // Initialize coordinator so httpfsMailboxSab is created and attached to io_worker.
  workerSelf.onmessage({
    data: { canvas: { id: 'c' }, width: 1280, height: 720, dataBase: '/data/' },
  });

  const ioWorker = spawnedWorkers.find((w) => w.script === './workers/io_worker.js');
  assert.ok(ioWorker);
  const attachMsg = ioWorker.posted.find((m) => m && m.attachMailbox === true);
  assert.ok(attachMsg && attachMsg.sab instanceof SharedArrayBuffer);

  // Regular coordinator workers must NOT be redirected or receive __playgta6_httpfs_sab.
  for (const w of spawnedWorkers) {
    assert.equal(
      w.posted.some((m) => m && m.__playgta6_httpfs_sab),
      false,
    );
  }

  // Emscripten 7.0.1 spawns `new Worker(pthreadMainJs, { name: "em-pthread-" + id })`.
  const pthread1 = new workerSelf.Worker('http://127.0.0.1:8080/dist/runtime.js', {
    name: 'em-pthread-1',
  });
  const pthread2 = new sandbox.Worker('./dist/runtime.js', { name: 'em-pthread-2' });
  const pthreadLegacy = new workerSelf.Worker('./dist/runtime.js', { name: 'em-pthread' });
  const nonPthread = new workerSelf.Worker('./workers/custom.js', { name: 'em-pthreadx' });

  for (const [worker, expectedName] of [
    [pthread1, 'em-pthread-1'],
    [pthread2, 'em-pthread-2'],
    [pthreadLegacy, 'em-pthread'],
  ]) {
    assert.ok(worker instanceof MockWorker);
    assert.equal(worker.script, 'http://127.0.0.1:8080/loader.js');
    assert.deepEqual(worker.opts, { name: expectedName });
    assert.equal(worker.posted.length, 1);
    assert.equal(worker.posted[0].__playgta6_httpfs_sab, attachMsg.sab);
  }

  // Non-matching worker names must keep their original script URL and not get the SAB message.
  assert.equal(nonPthread.script, './workers/custom.js');
  assert.deepEqual(nonPthread.opts, { name: 'em-pthreadx' });
  assert.equal(nonPthread.posted.length, 0);

  // postMessage errors during SAB delivery must be swallowed without breaking Worker construction.
  throwOnNextPost = true;
  const pthread3 = new workerSelf.Worker('http://127.0.0.1:8080/dist/runtime.js', {
    name: 'em-pthread-3',
  });
  assert.equal(pthread3.script, 'http://127.0.0.1:8080/loader.js');
  assert.equal(pthread3.posted.length, 0);
});

test('W77 bootstrapEmscriptenPthreadWorker attaches __playgta6_httpfs_sab via message listener, wires host_httpfs_*, and removes listener on cleanup', () => {
  const { createHttpfsMailboxBuffer } = require('../web/workers/httpfs_mailbox.js');
  const bus = createMockBroadcastBus({ rejectSabOnChannel: PTHREAD_HTTPFS_SAB_CHANNEL_NAME });

  const messageListeners = [];
  const emscriptenReceived = [];
  let listenerRegisteredBeforeImport = false;

  const mockScope = {
    name: 'em-pthread-1',
    location: { href: 'http://127.0.0.1:8080/loader.js' },
    WebAssembly: { Instance: WebAssembly.Instance },
    addEventListener(type, fn) {
      if (type === 'message') {
        messageListeners.push(fn);
      }
    },
    removeEventListener(type, fn) {
      if (type === 'message') {
        const idx = messageListeners.indexOf(fn);
        if (idx >= 0) {
          messageListeners.splice(idx, 1);
        }
      }
    },
  };

  function dispatchMessage(data) {
    let stopped = false;
    const ev = {
      data,
      stopImmediatePropagation() {
        stopped = true;
      },
    };
    const snapshot = messageListeners.slice();
    for (const fn of snapshot) {
      fn(ev);
      if (stopped) {
        break;
      }
    }
    return stopped;
  }

  const clientOps = [];
  class MockHttpfsClient {
    constructor(sab) {
      this.sab = sab;
    }
    open(relPath) {
      clientOps.push({ op: 'open', relPath, sab: this.sab });
      return 17;
    }
    read(handle, offset, dstView, byteLength) {
      clientOps.push({ op: 'read', handle, offset, byteLength });
      const payload = new TextEncoder().encode('GTA6');
      const n = Math.min(payload.byteLength, byteLength);
      dstView.set(payload.subarray(0, n));
      return BigInt(n);
    }
    close(handle) {
      clientOps.push({ op: 'close', handle });
      return true;
    }
  }

  const handle = bootstrapEmscriptenPthreadWorker({
    scope: mockScope,
    importScriptsFn: (url) => {
      assert.equal(url, './dist/runtime.js');
      listenerRegisteredBeforeImport = messageListeners.length === 1;
      // Simulate Emscripten registering its own message handler during importScripts.
      mockScope.addEventListener('message', (ev) => {
        emscriptenReceived.push(ev.data);
      });
    },
    BroadcastChannelCtor: bus.BusBroadcastChannel,
    httpfsMailboxApi: { HttpfsMailboxClient: MockHttpfsClient },
  });

  assert.equal(listenerRegisteredBeforeImport, true);
  assert.equal(messageListeners.length, 2);

  // Deliver SAB via Worker.postMessage({ __playgta6_httpfs_sab: sab }).
  const sab = createHttpfsMailboxBuffer(1024);
  const stoppedForSab = dispatchMessage({ __playgta6_httpfs_sab: sab });
  assert.equal(stoppedForSab, true);
  assert.equal(emscriptenReceived.length, 0);

  // Normal Emscripten messages must pass straight through to Emscripten's listener.
  const stoppedForLoad = dispatchMessage({ cmd: 'load' });
  assert.equal(stoppedForLoad, false);
  assert.deepEqual(emscriptenReceived, [{ cmd: 'load' }]);

  // Exercise host_httpfs_open, host_httpfs_read, and host_httpfs_close backed by the attached SAB.
  const mem = new WebAssembly.Memory({ initial: 1, maximum: 1 });
  handle.hostImports.setMemory(mem);
  const pathBytes = new TextEncoder().encode('boot/guest.pgim');
  new Uint8Array(mem.buffer).set(pathBytes, 64);

  const fd = handle.hostImports.env.host_httpfs_open(64n, BigInt(pathBytes.byteLength));
  assert.equal(fd, 17);

  const bytesRead = handle.hostImports.env.host_httpfs_read(fd, 0n, 256n, 8n);
  assert.equal(bytesRead, 4n);
  assert.equal(new TextDecoder().decode(new Uint8Array(mem.buffer, 256, 4)), 'GTA6');

  handle.hostImports.env.host_httpfs_close(fd);
  assert.deepEqual(clientOps, [
    { op: 'open', relPath: 'boot/guest.pgim', sab },
    { op: 'read', handle: 17, offset: 0n, byteLength: 8 },
    { op: 'close', handle: 17 },
  ]);

  // cleanup() removes the bootstrap message listener cleanly and is idempotent.
  handle.cleanup();
  handle.cleanup();
  assert.equal(messageListeners.length, 1);

  const stoppedAfterCleanup = dispatchMessage({ __playgta6_httpfs_sab: sab });
  assert.equal(stoppedAfterCleanup, false);
  assert.equal(emscriptenReceived.length, 2);
});

test('W77 createPatchedPthreadEnv and installPthreadHostImportShim overwrite writable .stub === true own properties on baseEnv', () => {
  const makeStub = (name) => {
    const fn = () => {
      throw new Error(`abort("missing function: ${name}")`);
    };
    fn.stub = true;
    return fn;
  };

  // 1. Direct createPatchedPthreadEnv check on an Emscripten MAIN_MODULE=1 wasmImports object.
  const logCalls = [];
  const hostEnv = {
    host_log: (ptr, len) => {
      logCalls.push([ptr, len]);
      return 'ok-log';
    },
    host_httpfs_open: () => 9,
    host_httpfs_read: () => 24n,
    host_httpfs_close: () => 'real-close',
    host_wgpu_submit: () => 'ok-wgpu',
  };

  const memoryBase = new WebAssembly.Global({ value: 'i32', mutable: false }, 128);
  const tableBase = new WebAssembly.Global({ value: 'i32', mutable: false }, 4);
  const indirectTable = new WebAssembly.Table({ initial: 2, element: 'anyfunc' });
  const liftedFn = () => 0x140000100;
  const customNonStubClose = () => 'custom-close';
  const nonAbiStub = makeStub('unrelated_emscripten_sym');

  const wasmImports = {
    __memory_base: memoryBase,
    __table_base: tableBase,
    __indirect_function_table: indirectTable,
    lifted_fn: liftedFn,
    host_log: makeStub('host_log'),
    host_httpfs_open: makeStub('host_httpfs_open'),
    host_httpfs_read: makeStub('host_httpfs_read'),
    host_httpfs_close: customNonStubClose,
    host_wgpu_submit: makeStub('host_wgpu_submit'),
    unrelated_emscripten_sym: nonAbiStub,
  };

  const patched = createPatchedPthreadEnv(wasmImports, hostEnv, null);

  // Unwrapped wasmImports itself must have writable .stub === true host_* properties overwritten.
  assert.equal(wasmImports.host_log, hostEnv.host_log);
  assert.equal(wasmImports.host_httpfs_open, hostEnv.host_httpfs_open);
  assert.equal(wasmImports.host_httpfs_read, hostEnv.host_httpfs_read);
  assert.equal(wasmImports.host_wgpu_submit, hostEnv.host_wgpu_submit);

  // Non-stub functions, non-ABI stubs, and dynamic-linking properties stay untouched.
  assert.equal(wasmImports.host_httpfs_close, customNonStubClose);
  assert.equal(wasmImports.unrelated_emscripten_sym, nonAbiStub);
  assert.equal(wasmImports.__memory_base, memoryBase);
  assert.equal(wasmImports.__table_base, tableBase);
  assert.equal(wasmImports.__indirect_function_table, indirectTable);
  assert.equal(wasmImports.lifted_fn, liftedFn);

  // Emscripten MAIN_MODULE=1 JS wrapper reading unwrapped `wasmImports["host_log"]` directly works!
  function emscriptenGeneratedHostLog(...args) {
    if (!wasmImports.host_log || wasmImports.host_log.stub) {
      throw new Error('abort("missing function: host_log")');
    }
    return wasmImports.host_log(...args);
  }
  assert.equal(emscriptenGeneratedHostLog(16, 5), 'ok-log');
  assert.deepEqual(logCalls, [[16, 5]]);
  assert.equal(patched.host_log, hostEnv.host_log);
  assert.equal(patched.host_httpfs_close, customNonStubClose);

  // Proxy baseEnv that throws on set must still succeed via try/catch + Proxy get interception.
  const readOnlyProxyEnv = new Proxy(
    { host_log: makeStub('host_log'), lifted_fn: liftedFn },
    {
      set() {
        throw new TypeError('read-only proxy');
      },
    },
  );
  const patchedReadOnly = createPatchedPthreadEnv(readOnlyProxyEnv, hostEnv, null);
  assert.equal(patchedReadOnly.host_log, hostEnv.host_log);
  assert.equal(patchedReadOnly.lifted_fn, liftedFn);

  // 2. End-to-end check through installPthreadHostImportShim with real WebAssembly.Module.
  const realLogs = [];
  const realHostImports = createHostImports({
    log: (msg) => realLogs.push(msg),
    report: () => {},
  });
  const realScope = {
    WebAssembly: {
      Module: WebAssembly.Module,
      Instance: WebAssembly.Instance,
    },
  };
  const restore = installPthreadHostImportShim(realScope, realHostImports);

  const wasmModule = new WebAssembly.Module(buildMinimalHostLogWasm());
  const memory = new WebAssembly.Memory({ initial: 1, maximum: 1 });
  const rawEnv = {
    memory,
    host_log: makeStub('host_log'),
    host_httpfs_close: customNonStubClose,
  };
  const instance = new realScope.WebAssembly.Instance(wasmModule, { env: rawEnv });

  // Verify rawEnv.host_log was patched in-place while non-stub host_httpfs_close stayed intact.
  assert.equal(rawEnv.host_log, realHostImports.env.host_log);
  assert.equal(rawEnv.host_httpfs_close, customNonStubClose);

  const text = 'main-module wrapper ok';
  const bytes = new TextEncoder().encode(text);
  new Uint8Array(memory.buffer).set(bytes, 32);

  // Both direct Wasm export AND JS wrapper calling rawEnv["host_log"] succeed.
  instance.exports.ping(32, bytes.byteLength);
  if (!rawEnv.host_log || rawEnv.host_log.stub) {
    throw new Error('stub was not overwritten on rawEnv');
  }
  rawEnv.host_log(32, bytes.byteLength);
  assert.deepEqual(realLogs, [text, text]);

  restore();
});

test('W101 forwardGpuWorkerMessage surfaces sanitized renderStateStats and getLoaderStatus keeps the latest', () => {
  const posted = [];
  const post = (msg) => posted.push(msg);

  // Messages without renderStateStats keep the pre-W101 shape (no renderStateStats key).
  const plain = forwardGpuWorkerMessage({ type: 'frame-presented', framesPresented: 2 }, post);
  assert.equal(Object.prototype.hasOwnProperty.call(plain, 'renderStateStats'), false);

  // 1. buildGpuTelemetrySnapshot() shape: top-level renderStateStats.
  const snap = forwardGpuWorkerMessage(
    {
      type: 'gpu-telemetry',
      frameStats: { framesPresented: 3, droppedFrames: 0 },
      renderStateStats: {
        computeDispatchCount: 12,
        computePipelineCount: '4',
        autoLayoutFallbacks: 2.9,
        dynamicOffsetRemaps: 7,
        remappedBindGroupEvictions: -1,
        hostPath: 'C:\\Users\\secret\\x.bin',
      },
    },
    post
  );
  assert.deepEqual(snap.renderStateStats, {
    computeDispatchCount: 12,
    computePipelineCount: 4,
    autoLayoutFallbacks: 2,
    bindGroupKindMismatches: 0,
    dynamicOffsetRemaps: 7,
    remappedBindGroupEvictions: 0,
  });
  assert.deepEqual(snap.frameStats, { framesPresented: 3, droppedFrames: 0 });
  assert.equal(posted.at(-1), snap);
  assert.deepEqual(getLoaderStatus().renderStateStats, snap.renderStateStats);

  // 2. { id, stats: { renderState } } response shape; missing core keys default to 0.
  const fromStats = forwardGpuWorkerMessage(
    { id: 3, stats: { frameCount: 1, renderState: { autoLayoutFallbacks: 5 } } },
    post
  );
  assert.deepEqual(fromStats.renderStateStats, {
    computeDispatchCount: 0,
    computePipelineCount: 0,
    autoLayoutFallbacks: 5,
    bindGroupKindMismatches: 0,
  });
  const status = getLoaderStatus();
  assert.deepEqual(status.renderStateStats, fromStats.renderStateStats);
  status.renderStateStats.autoLayoutFallbacks = 99;
  assert.equal(getLoaderStatus().renderStateStats.autoLayoutFallbacks, 5, 'status returns a copy');

  // Invalid inputs are ignored.
  assert.equal(sanitizeRenderStateStats(null), null);
  assert.equal(sanitizeRenderStateStats([1, 2]), null);
  assert.equal(sanitizeRenderStateStats('x'), null);
});

function writeTestInputSnapshot(sab, snapshot) {
  const i32 = new Int32Array(sab, 0, INPUT_STATE_WORDS);
  const f32 = new Float32Array(sab, 0, INPUT_STATE_WORDS);
  Atomics.add(i32, 0, 1); // odd: write in progress
  if (snapshot.buttons !== undefined) {
    Atomics.store(i32, 1, snapshot.buttons | 0);
  }
  if (snapshot.mouseDx !== undefined) {
    Atomics.add(i32, 2, snapshot.mouseDx | 0);
  }
  if (snapshot.mouseDy !== undefined) {
    Atomics.add(i32, 3, snapshot.mouseDy | 0);
  }
  const fScratch = new Float32Array(1);
  const iScratch = new Int32Array(fScratch.buffer);
  const storeFloat = (wordIdx, val) => {
    if (val === undefined) return;
    fScratch[0] = val;
    Atomics.store(i32, wordIdx, iScratch[0]);
  };
  storeFloat(4, snapshot.lx);
  storeFloat(5, snapshot.ly);
  storeFloat(6, snapshot.rx);
  storeFloat(7, snapshot.ry);
  storeFloat(8, snapshot.lt);
  storeFloat(9, snapshot.rt);
  Atomics.add(i32, 0, 1); // even: publish snapshot
  return { seq: Atomics.load(i32, 0) >>> 0, f32 };
}

test('O36/#796 startInputSabBridge copies InputState buttons, axes, triggers, and mouse deltas into Wasm64 linear memory at BigInt offset with valid seqlock framing', () => {
  assert.equal(INPUT_STATE_BYTES, 40);
  assert.equal(INPUT_STATE_WORDS, 10);
  assert.equal(DEFAULT_INPUT_BRIDGE_POLL_MS, 16);

  const sab = new SharedArrayBuffer(INPUT_STATE_BYTES);
  const wasmMemory = new WebAssembly.Memory({ initial: 1, maximum: 4 });
  const inputPtr = 256n; // Wasm64 BigInt linear-memory address (64-byte aligned)

  // Observe every Atomics.store sequence value written to dst word 0 to verify odd->even seqlock framing.
  const observedDstSeqWrites = [];
  const trackingAtomics = {
    load: (arr, idx) => Atomics.load(arr, idx),
    store: (arr, idx, val) => {
      if (arr.buffer === wasmMemory.buffer && idx === 0) {
        observedDstSeqWrites.push(val >>> 0);
      }
      return Atomics.store(arr, idx, val);
    },
    add: (arr, idx, val) => Atomics.add(arr, idx, val),
    exchange: (arr, idx, val) => Atomics.exchange(arr, idx, val),
    notify: (arr, idx, count) => Atomics.notify(arr, idx, count),
  };

  // Pre-populate an initial button press before attach to verify immediate initial sync.
  const BUTTON_A = 1 << 8;
  const BUTTON_RB = 1 << 13;
  writeTestInputSnapshot(sab, {
    buttons: BUTTON_A,
    mouseDx: 12,
    mouseDy: -7,
    lx: 0.5,
    ly: -0.25,
    rx: -0.75,
    ry: 1.0,
    lt: 0.4,
    rt: 0.9,
  });

  let scheduledFn = null;
  let clearedToken = null;
  const fakeTimerToken = { id: 77 };

  const mod = {
    wasmMemory,
    _playgta6_get_input_state_ptr: () => inputPtr,
  };

  const bridge = startInputSabBridge({
    inputSab: sab,
    module: mod,
    pollIntervalMs: 16,
    atomicsObj: trackingAtomics,
    useWaitAsync: false,
    setIntervalFn: (fn) => {
      scheduledFn = fn;
      return fakeTimerToken;
    },
    clearIntervalFn: (token) => {
      clearedToken = token;
    },
  });

  assert.equal(bridge.active, true);
  assert.equal(bridge.inputStatePtr, 256);
  assert.equal(typeof scheduledFn, 'function');

  // Initial sync wrote odd (1) then even (2).
  assert.deepEqual(observedDstSeqWrites, [1, 2]);

  const dstI32 = new Int32Array(wasmMemory.buffer, 256, INPUT_STATE_WORDS);
  const dstF32 = new Float32Array(wasmMemory.buffer, 256, INPUT_STATE_WORDS);
  assert.equal(dstI32[0] >>> 0, 2);
  assert.equal(dstI32[1] >>> 0, BUTTON_A);
  assert.equal(dstI32[2], 12);
  assert.equal(dstI32[3], -7);
  assert.ok(Math.abs(dstF32[4] - 0.5) < 1e-6);
  assert.ok(Math.abs(dstF32[5] - -0.25) < 1e-6);
  assert.ok(Math.abs(dstF32[6] - -0.75) < 1e-6);
  assert.ok(Math.abs(dstF32[7] - 1.0) < 1e-6);
  assert.ok(Math.abs(dstF32[8] - 0.4) < 1e-6);
  assert.ok(Math.abs(dstF32[9] - 0.9) < 1e-6);

  // Mouse deltas were atomically drained from inputSab so a subsequent button event won't double-count.
  const srcI32 = new Int32Array(sab, 0, INPUT_STATE_WORDS);
  assert.equal(srcI32[2], 0);
  assert.equal(srcI32[3], 0);

  // Simulate C++ GameInputReader consuming mouse deltas, then a new input update arrives.
  dstI32[2] = 0;
  dstI32[3] = 0;
  writeTestInputSnapshot(sab, { buttons: BUTTON_A | BUTTON_RB, lx: -1.0, rt: 0.0 });

  // Simulate an in-progress torn write (odd seq) first: poll tick must skip it cleanly.
  Atomics.add(srcI32, 0, 1);
  scheduledFn();
  assert.equal(dstI32[0] >>> 0, 2, 'must not copy torn snapshot while src seq is odd');

  // Finish the write (even seq): next poll tick copies cleanly and advances dst seq to 4 (3 -> 4).
  Atomics.add(srcI32, 0, 1);
  scheduledFn();
  assert.deepEqual(observedDstSeqWrites, [1, 2, 3, 4]);
  assert.equal(dstI32[0] >>> 0, 4);
  assert.equal(dstI32[1] >>> 0, BUTTON_A | BUTTON_RB);
  assert.equal(dstI32[2], 0);
  assert.equal(dstI32[3], 0);
  assert.ok(Math.abs(dstF32[4] - -1.0) < 1e-6);
  assert.ok(Math.abs(dstF32[9] - 0.0) < 1e-6);

  // Stop clears timer and suppresses any further writes.
  bridge.stop();
  bridge.stop(); // idempotent
  assert.equal(clearedToken, fakeTimerToken);
  assert.equal(bridge.getStats().stopped, true);
  assert.equal(bridge.getStats().active, false);

  writeTestInputSnapshot(sab, { buttons: 0 });
  assert.equal(bridge.syncNow(true), false);
  assert.equal(dstI32[1] >>> 0, BUTTON_A | BUTTON_RB);
});

test('O36/#796 startInputSabBridge re-acquires Int32Array view after wasmMemory.grow() and buffer replacement', () => {
  const sab = new SharedArrayBuffer(INPUT_STATE_BYTES);
  const wasmMemory = new WebAssembly.Memory({ initial: 1, maximum: 4 });
  const oldBuffer = wasmMemory.buffer;

  let currentMem = wasmMemory;
  const mod = {
    get wasmMemory() {
      return currentMem;
    },
    asm: {
      playgta6_get_input_state_ptr: () => 128n,
    },
  };

  const bridge = startInputSabBridge({
    inputSab: sab,
    module: mod,
    useWaitAsync: false,
    disableFallbackInterval: true,
  });

  assert.equal(bridge.active, true);
  writeTestInputSnapshot(sab, { buttons: 1 << 8, lx: 0.25 });
  assert.equal(bridge.syncNow(), true);
  assert.equal(new Int32Array(wasmMemory.buffer, 128, INPUT_STATE_WORDS)[1] >>> 0, 1 << 8);
  assert.equal(bridge.getStats().bufferReacquires, 0);

  // Grow WebAssembly.Memory -> detaches oldBuffer and produces a new ArrayBuffer!
  wasmMemory.grow(1);
  assert.notEqual(wasmMemory.buffer, oldBuffer);

  writeTestInputSnapshot(sab, { buttons: 1 << 9, lx: -0.5 });
  assert.equal(bridge.syncNow(), true);
  const grownI32 = new Int32Array(wasmMemory.buffer, 128, INPUT_STATE_WORDS);
  const grownF32 = new Float32Array(wasmMemory.buffer, 128, INPUT_STATE_WORDS);
  assert.equal(grownI32[0] >>> 0, 4);
  assert.equal(grownI32[1] >>> 0, 1 << 9);
  assert.ok(Math.abs(grownF32[4] - -0.5) < 1e-6);
  assert.equal(bridge.getStats().bufferReacquires, 1);

  // Also test full buffer replacement (e.g. swapped backing SharedArrayBuffer).
  const replacementSab = new SharedArrayBuffer(65536);
  currentMem = { buffer: replacementSab };
  writeTestInputSnapshot(sab, { buttons: 1 << 10, ry: 0.75 });
  assert.equal(bridge.syncNow(), true);
  const replacedI32 = new Int32Array(replacementSab, 128, INPUT_STATE_WORDS);
  const replacedF32 = new Float32Array(replacementSab, 128, INPUT_STATE_WORDS);
  assert.equal(replacedI32[0] >>> 0, 6);
  assert.equal(replacedI32[1] >>> 0, 1 << 10);
  assert.ok(Math.abs(replacedF32[7] - 0.75) < 1e-6);
  assert.equal(bridge.getStats().bufferReacquires, 2);

  bridge.stop();
});

test('O36/#796 startInputSabBridge wakes reactively via Atomics.waitAsync on notify and degrades gracefully when export or inputSab is missing', async () => {
  // 1. Missing inputSab, tiny buffer, missing export, null/zero pointer, unaligned pointer all degrade cleanly.
  const sab = new SharedArrayBuffer(INPUT_STATE_BYTES);
  const mem = new WebAssembly.Memory({ initial: 1, maximum: 1 });
  assert.equal(startInputSabBridge({}).active, false);
  assert.equal(startInputSabBridge({ inputSab: new ArrayBuffer(16), module: { wasmMemory: mem, _playgta6_get_input_state_ptr: () => 64n } }).active, false);
  assert.equal(startInputSabBridge({ inputSab: sab, module: { wasmMemory: mem } }).active, false);
  assert.equal(startInputSabBridge({ inputSab: sab, module: { wasmMemory: mem, _playgta6_get_input_state_ptr: () => 0n } }).active, false);
  assert.equal(startInputSabBridge({ inputSab: sab, module: { wasmMemory: mem, _playgta6_get_input_state_ptr: () => 0 } }).active, false);
  assert.equal(startInputSabBridge({ inputSab: sab, module: { wasmMemory: mem, _playgta6_get_input_state_ptr: () => 65n } }).active, false);
  assert.equal(
    startInputSabBridge({
      inputSab: sab,
      module: {
        wasmMemory: mem,
        _playgta6_get_input_state_ptr: () => {
          throw new Error('not exported');
        },
      },
    }).active,
    false,
  );

  // 2. Reactive Atomics.waitAsync wakeup on Atomics.notify (when supported by Node runtime).
  if (typeof Atomics.waitAsync === 'function') {
    const bridge = startInputSabBridge({
      inputSab: sab,
      module: { wasmMemory: mem, _playgta6_get_input_state_ptr: () => 64n },
      pollIntervalMs: 500,
      disableFallbackInterval: true,
    });
    assert.equal(bridge.active, true);

    const srcI32 = new Int32Array(sab, 0, INPUT_STATE_WORDS);
    writeTestInputSnapshot(sab, { buttons: 1 << 8, lt: 0.6 });
    Atomics.notify(srcI32, 0, 1);

    // Wait a few ticks for waitAsync promise resolution (much faster than 500ms timeout).
    const deadline = Date.now() + 250;
    const dstI32 = new Int32Array(mem.buffer, 64, INPUT_STATE_WORDS);
    while (dstI32[0] === 0 && Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 5));
    }
    assert.equal(dstI32[0] >>> 0, 2);
    assert.equal(dstI32[1] >>> 0, 1 << 8);
    assert.ok(bridge.getStats().waitAsyncWakeups >= 1);

    bridge.stop();
  }
});

test('O36/#796 buildRuntimeModuleConfig wires onRuntimeInitialized and onExit to start and stop the inputSab bridge', () => {
  const sab = new SharedArrayBuffer(INPUT_STATE_BYTES);
  const wasmMemory = new WebAssembly.Memory({ initial: 1, maximum: 2 });
  let userInitCalled = 0;
  let userExitCalled = 0;

  const cfg = buildRuntimeModuleConfig({
    locationHref: 'http://127.0.0.1:8080/loader.js',
    inputSab: sab,
    inputPollIntervalMs: 100,
    onRuntimeInitialized() {
      userInitCalled += 1;
    },
    onExit() {
      userExitCalled += 1;
    },
  });

  const mockModule = {
    wasmMemory,
    _playgta6_get_input_state_ptr: () => 192n,
  };

  writeTestInputSnapshot(sab, { buttons: 1 << 8 });
  cfg.preRun[0](mockModule);
  cfg.onRuntimeInitialized.call(mockModule);

  assert.equal(userInitCalled, 1);
  const bridge = cfg.getInputBridge();
  assert.ok(bridge && bridge.active);
  assert.equal(new Int32Array(wasmMemory.buffer, 192, INPUT_STATE_WORDS)[1] >>> 0, 1 << 8);

  cfg.onExit(0);
  assert.equal(userExitCalled, 1);
  assert.equal(cfg.getInputBridge(), null);
  assert.equal(bridge.getStats().stopped, true);
});

function createFakeMemfs({ withMkdirTree = true } = {}) {
  const S_IFDIR = 0o040000;
  const S_IFREG = 0o100000;
  const dirs = new Set(['/']);
  const files = new Map();
  const extraDirEntries = new Map();
  const counts = {
    mkdir: 0,
    mkdirTree: 0,
    readdir: 0,
    stat: 0,
    readFile: 0,
  };

  function normPath(p) {
    const clean = String(p || '').replace(/\/+/g, '/');
    return clean.length > 1 && clean.endsWith('/') ? clean.slice(0, -1) : clean || '/';
  }

  function parentOf(p) {
    const idx = p.lastIndexOf('/');
    return idx <= 0 ? '/' : p.slice(0, idx);
  }

  function makeErrnoError(code, errno, msg) {
    const err = new Error(msg);
    err.code = code;
    err.errno = errno;
    return err;
  }

  const fakeFs = {
    isDir(mode) {
      return (mode & 0o170000) === S_IFDIR;
    },
    isFile(mode) {
      return (mode & 0o170000) === S_IFREG;
    },
    mkdir(p) {
      counts.mkdir += 1;
      const target = normPath(p);
      if (dirs.has(target) || files.has(target)) {
        throw makeErrnoError('EEXIST', 20, `File exists: ${target}`);
      }
      const parent = parentOf(target);
      if (!dirs.has(parent)) {
        throw makeErrnoError('ENOENT', 44, `No such parent: ${parent}`);
      }
      dirs.add(target);
    },
    readdir(p) {
      counts.readdir += 1;
      const dir = normPath(p);
      if (!dirs.has(dir)) {
        throw makeErrnoError('ENOENT', 44, `No such directory: ${dir}`);
      }
      const prefix = dir === '/' ? '/' : `${dir}/`;
      const names = new Set(['.', '..']);
      for (const d of dirs) {
        if (d === dir || !d.startsWith(prefix)) continue;
        const rest = d.slice(prefix.length);
        if (rest && !rest.includes('/')) {
          names.add(rest);
        }
      }
      for (const f of files.keys()) {
        if (!f.startsWith(prefix)) continue;
        const rest = f.slice(prefix.length);
        if (rest && !rest.includes('/')) {
          names.add(rest);
        }
      }
      const injected = extraDirEntries.get(dir);
      if (injected) {
        for (const bad of injected) {
          names.add(bad);
        }
      }
      return Array.from(names);
    },
    stat(p) {
      counts.stat += 1;
      const target = normPath(p);
      if (dirs.has(target)) {
        return { mode: S_IFDIR | 0o755, size: 0 };
      }
      if (files.has(target)) {
        return { mode: S_IFREG | 0o644, size: files.get(target).byteLength };
      }
      throw makeErrnoError('ENOENT', 44, `No such entry: ${target}`);
    },
    readFile(p) {
      counts.readFile += 1;
      const target = normPath(p);
      if (!files.has(target)) {
        throw makeErrnoError('ENOENT', 44, `No such file: ${target}`);
      }
      return files.get(target).slice();
    },
  };

  if (withMkdirTree) {
    fakeFs.mkdirTree = (p) => {
      counts.mkdirTree += 1;
      const target = normPath(p);
      const parts = target.split('/').filter(Boolean);
      let cur = '';
      for (const part of parts) {
        cur += `/${part}`;
        dirs.add(cur);
      }
    };
  }

  function writeFile(p, data) {
    const target = normPath(p);
    const parts = target.split('/').filter(Boolean);
    let cur = '';
    for (let i = 0; i < parts.length - 1; i += 1) {
      cur += `/${parts[i]}`;
      dirs.add(cur);
    }
    const bytes =
      data instanceof Uint8Array
        ? data.slice()
        : typeof data === 'string'
          ? new TextEncoder().encode(data)
          : new Uint8Array(data);
    files.set(target, bytes);
  }

  function injectUnsafeEntry(dirPath, rawName) {
    const dir = normPath(dirPath);
    dirs.add(dir);
    if (!extraDirEntries.has(dir)) {
      extraDirEntries.set(dir, []);
    }
    extraDirEntries.get(dir).push(rawName);
  }

  return {
    fs: fakeFs,
    dirs,
    files,
    counts,
    writeFile,
    injectUnsafeEntry,
  };
}

test('W180/#918 createMemfsPipelineDumpSyncer is disabled by default with zero overhead and supports opt-in URL/option flags', async () => {
  assert.equal(DEFAULT_MEMFS_PIPELINE_DUMP_DIR, '/pg6_dump');
  assert.equal(DEFAULT_MEMFS_PIPELINE_DUMP_ENDPOINT, '/__pg6_dump/');
  assert.equal(DEFAULT_MEMFS_PIPELINE_DUMP_INTERVAL_MS, 3000);
  assert.equal(DEFAULT_MEMFS_PIPELINE_DUMP_MAX_CONCURRENT_UPLOADS, 4);

  // URL / option opt-in resolution (headless-safe when location is absent).
  assert.equal(resolveDumpPipelinesEnabled(), false);
  assert.equal(resolveDumpPipelinesEnabled({}), false);
  assert.equal(resolveDumpPipelinesEnabled({ dumpPipelines: true }), true);
  assert.equal(
    resolveDumpPipelinesEnabled({ dumpPipelines: false, locationSearch: '?dumpPipelines=1' }),
    false,
  );
  assert.equal(resolveDumpPipelinesEnabled({ locationSearch: '?dumpPipelines=1' }), true);
  assert.equal(resolveDumpPipelinesEnabled({ locationSearch: '?foo=bar&dumpPipelines=true' }), true);
  assert.equal(resolveDumpPipelinesEnabled({ locationSearch: '?dumpPipelines=TRUE' }), true);
  assert.equal(resolveDumpPipelinesEnabled({ locationSearch: '?dumpPipelines=0' }), false);
  assert.equal(resolveDumpPipelinesEnabled({ locationSearch: '?dumpPipelines=false' }), false);
  assert.equal(
    resolveDumpPipelinesEnabled({ locationHref: 'http://127.0.0.1:8080/loader.js?dumpPipelines=1' }),
    true,
  );
  assert.equal(
    resolveDumpPipelinesEnabled({ locationHref: 'http://127.0.0.1:8080/loader.js?other=1' }),
    false,
  );

  // Path sanitization checks.
  assert.equal(sanitizeDumpRelPath('native/0001.bin'), 'native/0001.bin');
  assert.equal(sanitizeDumpRelPath('manifest.json'), 'manifest.json');
  assert.equal(sanitizeDumpRelPath('../escape.bin'), null);
  assert.equal(sanitizeDumpRelPath('native/../escape.bin'), null);
  assert.equal(sanitizeDumpRelPath('native\\0001.bin'), null);
  assert.equal(sanitizeDumpRelPath('native/bad\0.bin'), null);
  assert.equal(sanitizeDumpRelPath('/abs/path.bin'), null);
  assert.equal(sanitizeDumpRelPath('C:drive.bin'), null);

  // Disabled syncer must perform zero FS, timer, or fetch work.
  const memfs = createFakeMemfs();
  memfs.writeFile('/pg6_dump/native/a.bin', new Uint8Array([1, 2, 3, 4]));
  let fetchCalls = 0;
  let timerCalls = 0;
  const syncer = createMemfsPipelineDumpSyncer({
    fs: memfs.fs,
    fetchFn: async () => {
      fetchCalls += 1;
      return { ok: true, status: 200 };
    },
    setIntervalFn: () => {
      timerCalls += 1;
      return 1;
    },
  });

  assert.equal(syncer.enabled, false);
  assert.equal(syncer.ensureDumpDir(), false);
  assert.equal(syncer.start(), false);
  const flushStats = await syncer.flush();
  const stopStats = await syncer.stop();

  assert.deepEqual(flushStats, {
    enabled: false,
    filesUploaded: 0,
    bytesUploaded: 0,
    uploadErrors: 0,
    lastFlushAtMs: null,
    pendingCount: 0,
  });
  assert.deepEqual(stopStats, flushStats);
  assert.equal(fetchCalls, 0);
  assert.equal(timerCalls, 0);
  assert.equal(memfs.counts.mkdir, 0);
  assert.equal(memfs.counts.mkdirTree, 0);
  assert.equal(memfs.counts.readdir, 0);
  assert.equal(memfs.counts.readFile, 0);

  // buildRuntimeModuleConfig keeps dump sync disabled by default.
  const cfg = buildRuntimeModuleConfig({
    locationHref: 'http://127.0.0.1:8080/loader.js',
    dumpFetchFn: async () => {
      fetchCalls += 1;
      return { ok: true, status: 200 };
    },
    dumpSetIntervalFn: () => {
      timerCalls += 1;
      return 99;
    },
  });
  cfg.preRun[0]({ FS: memfs.fs });
  cfg.onExit(0);
  assert.equal(cfg.getPipelineDumpSyncer().enabled, false);
  assert.equal(fetchCalls, 0);
  assert.equal(timerCalls, 0);
  assert.equal(memfs.counts.mkdirTree, 0);
});

test('W180/#918 preRun creates /pg6_dump via mkdirTree or mkdir fallback ignoring EEXIST and activates via URL query', () => {
  // 1. FS.mkdirTree path enabled via URL query parameter.
  const memfsTree = createFakeMemfs({ withMkdirTree: true });
  let scheduledMs = null;
  let clearedId = null;
  const cfgUrl = buildRuntimeModuleConfig({
    locationHref: 'http://127.0.0.1:8080/loader.js?dumpPipelines=1',
    dumpIntervalMs: 2500,
    dumpFetchFn: async () => ({ ok: true, status: 200 }),
    dumpSetIntervalFn: (_fn, ms) => {
      scheduledMs = ms;
      return 'timer-tree';
    },
    dumpClearIntervalFn: (id) => {
      clearedId = id;
    },
  });

  assert.equal(memfsTree.dirs.has('/pg6_dump'), false);
  cfgUrl.preRun[0]({ FS: memfsTree.fs });
  assert.equal(memfsTree.dirs.has('/pg6_dump'), true);
  assert.equal(memfsTree.counts.mkdirTree, 1);
  assert.equal(scheduledMs, 2500);

  // Calling ensureDumpDir again when directory exists does not throw.
  assert.equal(cfgUrl.getPipelineDumpSyncer().ensureDumpDir(), true);
  cfgUrl.onExit(0);
  assert.equal(clearedId, 'timer-tree');

  // 2. FS.mkdir fallback (no mkdirTree) with nested dumpDir and pre-existing EEXIST parent.
  const memfsFlat = createFakeMemfs({ withMkdirTree: false });
  memfsFlat.dirs.add('/work');
  const syncer = createMemfsPipelineDumpSyncer({
    dumpPipelines: true,
    dumpDir: '/work/pg6_dump',
    fs: memfsFlat.fs,
    fetchFn: async () => ({ ok: true, status: 200 }),
    setIntervalFn: () => null,
  });

  assert.equal(syncer.ensureDumpDir(), true);
  assert.equal(memfsFlat.dirs.has('/work/pg6_dump'), true);
  // Second call encounters EEXIST on both /work and /work/pg6_dump and ignores it cleanly.
  assert.equal(syncer.ensureDumpDir(), true);
});

test('W180/#918 recursively uploads native/*.bin, dxil/*.dxbc, manifest.json with deduplication, size-change re-upload, and traversal blocking', async () => {
  const memfs = createFakeMemfs();
  const binBytes = new Uint8Array([0x01, 0x02, 0x03, 0x04, 0x05, 0x06]);
  const metaBytes = new TextEncoder().encode('{"stage":"ps"}');
  const dxbcBytes = new Uint8Array([0x44, 0x58, 0x42, 0x43, 0x10, 0x20, 0x30, 0x40]);
  const manifestV1 = new TextEncoder().encode('{"count":1}');

  memfs.writeFile('/pg6_dump/native/pso_0001.bin', binBytes);
  memfs.writeFile('/pg6_dump/native/pso_0001.meta', metaBytes);
  memfs.writeFile('/pg6_dump/dxil/shader_0002.dxbc', dxbcBytes);
  memfs.writeFile('/pg6_dump/manifest.json', manifestV1);

  // Inject hostile entries that must be rejected cleanly.
  memfs.injectUnsafeEntry('/pg6_dump', '../escape.bin');
  memfs.injectUnsafeEntry('/pg6_dump/native', 'bad\\name.bin');
  memfs.injectUnsafeEntry('/pg6_dump/dxil', 'nul\0byte.dxbc');
  memfs.injectUnsafeEntry('/pg6_dump', 'C:secret.bin');

  const uploads = [];
  let fakeNow = 1700000001000;
  const syncer = createMemfsPipelineDumpSyncer({
    dumpPipelines: true,
    fs: memfs.fs,
    nowFn: () => fakeNow,
    fetchFn: async (url, init) => {
      uploads.push({
        url,
        method: init.method,
        contentType: init.headers['Content-Type'],
        body: new Uint8Array(init.body),
      });
      return { ok: true, status: 200 };
    },
  });

  const stats1 = await syncer.flush();
  assert.equal(uploads.length, 4);
  assert.deepEqual(
    uploads.map((u) => u.url),
    [
      '/__pg6_dump/dxil/shader_0002.dxbc',
      '/__pg6_dump/native/pso_0001.bin',
      '/__pg6_dump/native/pso_0001.meta',
      '/__pg6_dump/manifest.json',
    ],
  );
  for (const u of uploads) {
    assert.equal(u.method, 'POST');
    assert.equal(u.contentType, 'application/octet-stream');
    assert.ok(u.body instanceof Uint8Array);
  }
  assert.deepEqual(uploads[0].body, dxbcBytes);
  assert.deepEqual(uploads[1].body, binBytes);
  assert.deepEqual(uploads[2].body, metaBytes);
  assert.deepEqual(uploads[3].body, manifestV1);

  const expectedBytes1 =
    dxbcBytes.byteLength + binBytes.byteLength + metaBytes.byteLength + manifestV1.byteLength;
  assert.deepEqual(stats1, {
    enabled: true,
    filesUploaded: 4,
    bytesUploaded: expectedBytes1,
    uploadErrors: 0,
    lastFlushAtMs: 1700000001000,
    pendingCount: 0,
  });
  // Verify getStats() returns only sanitized fields (zero host paths or raw shader bytes).
  assert.deepEqual(Object.keys(stats1).sort(), [
    'bytesUploaded',
    'enabled',
    'filesUploaded',
    'lastFlushAtMs',
    'pendingCount',
    'uploadErrors',
  ]);
  stats1.filesUploaded = 999;
  assert.equal(syncer.getStats().filesUploaded, 4);

  // Second flush with no file changes must skip all 4 files (zero POSTs, zero extra readFile).
  const readCountBefore = memfs.counts.readFile;
  fakeNow = 1700000004000;
  const stats2 = await syncer.flush();
  assert.equal(uploads.length, 4);
  assert.equal(memfs.counts.readFile, readCountBefore);
  assert.equal(stats2.filesUploaded, 4);
  assert.equal(stats2.lastFlushAtMs, 1700000004000);

  // Write a newly discovered Bink2 shader and grow manifest.json: only those 2 files upload!
  const dxbc2Bytes = new Uint8Array([0x44, 0x58, 0x42, 0x43, 0x99, 0xaa]);
  const manifestV2 = new TextEncoder().encode('{"count":2,"entries":["pso_0001","bink2_0003"]}');
  memfs.writeFile('/pg6_dump/dxil/bink2_0003.dxbc', dxbc2Bytes);
  memfs.writeFile('/pg6_dump/manifest.json', manifestV2);

  fakeNow = 1700000007000;
  const stats3 = await syncer.flush();
  assert.equal(uploads.length, 6);
  assert.deepEqual(
    uploads.slice(4).map((u) => u.url),
    ['/__pg6_dump/dxil/bink2_0003.dxbc', '/__pg6_dump/manifest.json'],
  );
  assert.deepEqual(uploads[4].body, dxbc2Bytes);
  assert.deepEqual(uploads[5].body, manifestV2);
  assert.deepEqual(stats3, {
    enabled: true,
    filesUploaded: 6,
    bytesUploaded: expectedBytes1 + dxbc2Bytes.byteLength + manifestV2.byteLength,
    uploadErrors: 0,
    lastFlushAtMs: 1700000007000,
    pendingCount: 0,
  });
});

test('W180/#918 enforces maxConcurrentUploads bounded concurrency cap and tracks pendingCount', async () => {
  const memfs = createFakeMemfs();
  for (let i = 0; i < 9; i += 1) {
    memfs.writeFile(`/pg6_dump/native/s${i}.bin`, new Uint8Array([i, i + 1, i + 2]));
  }

  let inFlight = 0;
  let peakInFlight = 0;
  const resolvers = [];

  const syncer = createMemfsPipelineDumpSyncer({
    dumpPipelines: true,
    maxConcurrentUploads: 3,
    fs: memfs.fs,
    fetchFn: (url) =>
      new Promise((resolve) => {
        inFlight += 1;
        if (inFlight > peakInFlight) {
          peakInFlight = inFlight;
        }
        resolvers.push({
          url,
          finish() {
            inFlight -= 1;
            resolve({ ok: true, status: 200 });
          },
        });
      }),
  });

  const flushPromise = syncer.flush();
  // Immediately after starting flush, exactly 3 uploads are in flight and all 9 are pending.
  assert.equal(inFlight, 3);
  assert.equal(peakInFlight, 3);
  assert.equal(syncer.getStats().pendingCount, 9);

  while (resolvers.length > 0) {
    const batch = resolvers.splice(0, resolvers.length);
    for (const item of batch) {
      item.finish();
    }
    await new Promise((r) => setImmediate(r));
    assert.ok(inFlight <= 3, `Expected inFlight <= 3, got ${inFlight}`);
  }

  const finalStats = await flushPromise;
  assert.equal(peakInFlight, 3);
  assert.equal(inFlight, 0);
  assert.equal(finalStats.filesUploaded, 9);
  assert.equal(finalStats.bytesUploaded, 27);
  assert.equal(finalStats.uploadErrors, 0);
  assert.equal(finalStats.pendingCount, 0);
});

test('W180/#918 handles HTTP 403/500 and network rejections without crashing, increments uploadErrors, and retries next flush', async () => {
  const memfs = createFakeMemfs();
  memfs.writeFile('/pg6_dump/native/a.bin', new Uint8Array([1, 2]));
  memfs.writeFile('/pg6_dump/dxil/b.dxbc', new Uint8Array([3, 4, 5]));
  memfs.writeFile('/pg6_dump/manifest.json', '{"v":1}');
  memfs.writeFile('/pg6_dump/native/ok.bin', new Uint8Array([9]));

  let failMode = true;
  const attempts = [];
  const syncer = createMemfsPipelineDumpSyncer({
    dumpPipelines: true,
    fs: memfs.fs,
    fetchFn: async (url) => {
      attempts.push(url);
      if (!failMode) {
        return { ok: true, status: 200 };
      }
      if (url.endsWith('/native/a.bin')) {
        return { ok: false, status: 403 };
      }
      if (url.endsWith('/dxil/b.dxbc')) {
        return { ok: false, status: 500 };
      }
      if (url.endsWith('/manifest.json')) {
        throw new Error('simulated network disconnect');
      }
      return { ok: true, status: 200 };
    },
  });

  const stats1 = await syncer.flush();
  assert.equal(stats1.filesUploaded, 1);
  assert.equal(stats1.bytesUploaded, 1);
  assert.equal(stats1.uploadErrors, 3);
  assert.equal(stats1.pendingCount, 3);

  // Recover server and retry: only the 3 failed files are retried (native/ok.bin is NOT re-posted).
  failMode = false;
  attempts.length = 0;
  const stats2 = await syncer.flush();
  assert.deepEqual(attempts.sort(), [
    '/__pg6_dump/dxil/b.dxbc',
    '/__pg6_dump/manifest.json',
    '/__pg6_dump/native/a.bin',
  ]);
  assert.equal(stats2.filesUploaded, 4);
  assert.equal(stats2.bytesUploaded, 1 + 2 + 3 + 7);
  assert.equal(stats2.uploadErrors, 3);
  assert.equal(stats2.pendingCount, 0);
});

test('W180/#918 buildRuntimeModuleConfig hooks preRun timer + onExit final flush drain and coalesces in-flight stop()', async () => {
  const memfs = createFakeMemfs();
  const postedUrls = [];
  let intervalCb = null;
  let clearedToken = null;
  let userExitCode = null;
  const fakeTimer = { id: 180 };

  const cfg = buildRuntimeModuleConfig({
    locationHref: 'http://127.0.0.1:8080/loader.js',
    dumpPipelines: true,
    dumpIntervalMs: 3000,
    dumpFetchFn: async (url) => {
      postedUrls.push(url);
      return { ok: true, status: 200 };
    },
    dumpSetIntervalFn: (fn, ms) => {
      assert.equal(ms, 3000);
      intervalCb = fn;
      return fakeTimer;
    },
    dumpClearIntervalFn: (token) => {
      clearedToken = token;
    },
    onExit(code) {
      userExitCode = code;
    },
  });

  const mockModule = { FS: memfs.fs };
  cfg.preRun[0](mockModule);
  assert.equal(memfs.dirs.has('/pg6_dump'), true);
  assert.equal(typeof intervalCb, 'function');

  // Periodic timer flush picks up first shader dump.
  memfs.writeFile('/pg6_dump/native/boot0.bin', new Uint8Array([0xaa, 0xbb]));
  intervalCb();
  await new Promise((r) => setImmediate(r));
  assert.deepEqual(postedUrls, ['/__pg6_dump/native/boot0.bin']);

  // New post-XGameSave / Bink2 shaders written right before exit must be drained by onExit().
  memfs.writeFile('/pg6_dump/dxil/bink2_intro.dxbc', new Uint8Array([0xde, 0xad, 0xbe, 0xef]));
  memfs.writeFile('/pg6_dump/manifest.json', '{"shaders":2}');

  cfg.onExit(0);
  assert.equal(clearedToken, fakeTimer);
  assert.equal(userExitCode, 0);
  await new Promise((r) => setImmediate(r));
  assert.deepEqual(postedUrls, [
    '/__pg6_dump/native/boot0.bin',
    '/__pg6_dump/dxil/bink2_intro.dxbc',
    '/__pg6_dump/manifest.json',
  ]);
  const stats = cfg.getPipelineDumpSyncer().getStats();
  assert.equal(stats.filesUploaded, 3);
  assert.equal(stats.uploadErrors, 0);
  assert.equal(stats.pendingCount, 0);

  // Coalesced stop() while an earlier flush() is still in flight runs a follow-up drain.
  const memfs2 = createFakeMemfs();
  memfs2.writeFile('/pg6_dump/native/first.bin', new Uint8Array([1, 2]));
  const uploaded2 = [];
  let releaseFirst = null;
  const syncer2 = createMemfsPipelineDumpSyncer({
    dumpPipelines: true,
    fs: memfs2.fs,
    setIntervalFn: () => 55,
    clearIntervalFn: () => {},
    fetchFn: (url) =>
      new Promise((resolve) => {
        uploaded2.push(url);
        if (url.endsWith('/first.bin')) {
          releaseFirst = () => resolve({ ok: true, status: 200 });
        } else {
          resolve({ ok: true, status: 200 });
        }
      }),
  });
  syncer2.start();
  const inFlightFlush = syncer2.flush();
  assert.equal(typeof releaseFirst, 'function');

  // Write a second file while first.bin upload is still in flight, then call stop().
  memfs2.writeFile('/pg6_dump/native/late_exit.bin', new Uint8Array([3, 4, 5]));
  const stopDrain = syncer2.stop();
  releaseFirst();
  await Promise.all([inFlightFlush, stopDrain]);

  assert.deepEqual(uploaded2, [
    '/__pg6_dump/native/first.bin',
    '/__pg6_dump/native/late_exit.bin',
  ]);
  assert.equal(syncer2.getStats().filesUploaded, 2);
  assert.equal(syncer2.getStats().pendingCount, 0);
});
