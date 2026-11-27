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
  PTHREAD_TX_MSG_FAULT_RECORD,
  FAULT_RECORD_MESSAGE_TYPE,
  FAULT_RECORD_PREFIX,
  FLUSH_PIPELINE_DUMPS_MESSAGE_TYPE,
  FLUSH_PIPELINE_DUMP_MESSAGE_TYPE,
  PIPELINE_DUMPS_FLUSHED_MESSAGE_TYPE,
  HOST_ABI_IMPORT_NAMES,
  GPU_OPCODE_PRESENT,
  GPU_PACKET_HEADER_BYTES,
  INPUT_STATE_BYTES,
  INPUT_STATE_WORDS,
  DEFAULT_INPUT_BRIDGE_POLL_MS,
  DEFAULT_INPUT_BRIDGE_WAIT_ASYNC_TIMEOUT_MS,
  DEFAULT_INPUT_BRIDGE_WAIT_ASYNC_FALLBACK_POLL_MS,
  DEFAULT_MEMFS_PIPELINE_DUMP_DIR,
  DEFAULT_MEMFS_PIPELINE_DUMP_ENDPOINT,
  DEFAULT_MEMFS_PIPELINE_DUMP_INTERVAL_MS,
  DEFAULT_MEMFS_PIPELINE_DUMP_MAX_CONCURRENT_UPLOADS,
  BOOT_INIT_FORWARD_KEYS,
  isEmscriptenPthreadWorkerName,
  isEmscriptenPthreadWorker,
  isFaultRecordLine,
  parseFaultRecordLine,
  getLastFaultRecord,
  resetLastFaultRecord,
  flushPipelineDump,
  flushPipelineDumps,
  createPatchedPthreadEnv,
  installPthreadHostImportShim,
  bootstrapEmscriptenPthreadWorker,
  inspectGpuPacketBatchForPresent,
  normalizeBootHttpfsPath,
  collectBootBridgeInitFields,
  createWorkerBootInitPayloads,
  forwardGpuWorkerMessage,
  sanitizeFrameStats,
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
  toTransferableArrayBuffer,
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
    enableResourceReplay: true,
    enableShadowBufferReplay: false,
  });
  assert.equal('enableResourceReplay' in payloads.ioInit, false);
  assert.equal('enableShadowBufferReplay' in payloads.ioInit, false);
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

test('W365 (#663) startInputSabBridge uses 250ms waitAsync timeout, 50ms fallback cadence when waitAsync is active, and 1-load idle fast-exit', async () => {
  assert.equal(DEFAULT_INPUT_BRIDGE_WAIT_ASYNC_TIMEOUT_MS, 250);
  assert.equal(DEFAULT_INPUT_BRIDGE_WAIT_ASYNC_FALLBACK_POLL_MS, 50);

  const sab = new SharedArrayBuffer(INPUT_STATE_BYTES);
  const wasmMemory = new WebAssembly.Memory({ initial: 1, maximum: 2 });

  const waitAsyncTimeouts = [];
  let resolveWait = null;
  let srcLoadCount = 0;
  let getMemoryCount = 0;
  const scheduledIntervals = [];

  const mockAtomics = {
    load: (arr, idx) => {
      if (arr.buffer === sab && idx === 0) {
        srcLoadCount += 1;
      }
      return Atomics.load(arr, idx);
    },
    store: (arr, idx, val) => Atomics.store(arr, idx, val),
    add: (arr, idx, val) => Atomics.add(arr, idx, val),
    exchange: (arr, idx, val) => Atomics.exchange(arr, idx, val),
    notify: (arr, idx, count) => Atomics.notify(arr, idx, count),
    waitAsync: (_arr, _idx, _expected, timeoutMs) => {
      waitAsyncTimeouts.push(timeoutMs);
      const promise = new Promise((resolve) => {
        resolveWait = resolve;
      });
      return { async: true, value: promise };
    },
  };

  // 1. Default options with waitAsync enabled: waitAsync timeout >= 250ms, fallback setInterval = 50ms.
  const bridge = startInputSabBridge({
    inputSab: sab,
    module: { _playgta6_get_input_state_ptr: () => 128n },
    getMemory: () => {
      getMemoryCount += 1;
      return wasmMemory;
    },
    atomicsObj: mockAtomics,
    setIntervalFn: (fn, ms) => {
      const entry = { fn, ms };
      scheduledIntervals.push(entry);
      return entry;
    },
    clearIntervalFn: () => {},
  });

  assert.equal(bridge.active, true);
  assert.deepEqual(waitAsyncTimeouts, [250], 'default waitAsync timeout must be 250ms');
  assert.equal(scheduledIntervals.length, 1);
  assert.equal(scheduledIntervals[0].ms, 50, 'default fallback cadence must relax to 50ms when waitAsync is active');

  // Idle fallback ticks must fast-exit after 1 single Atomics.load(srcI32, 0) with zero getMemory or waitAsync re-arm.
  srcLoadCount = 0;
  getMemoryCount = 0;
  scheduledIntervals[0].fn();
  scheduledIntervals[0].fn();
  scheduledIntervals[0].fn();
  assert.equal(srcLoadCount, 3, 'each idle tick must execute exactly 1 atomic seq load');
  assert.equal(getMemoryCount, 0, 'idle ticks must not call getMemory()');
  assert.equal(waitAsyncTimeouts.length, 1, 'idle ticks must not re-arm waitAsync while in flight');
  assert.equal(bridge.getStats().pollTicks, 3);

  // Writer omitting Atomics.notify is still synced on the next fallback tick.
  writeTestInputSnapshot(sab, { buttons: 1 << 9, lx: 0.75 });
  scheduledIntervals[0].fn();
  const dstI32 = new Int32Array(wasmMemory.buffer, 128, INPUT_STATE_WORDS);
  assert.equal(dstI32[1] >>> 0, 1 << 9);
  assert.equal(bridge.getStats().syncCount, 1);

  // When waitAsync resolves with 'ok' after a new write, it wakes and re-arms with 250ms timeout.
  writeTestInputSnapshot(sab, { buttons: 1 << 10, rt: 0.5 });
  resolveWait('ok');
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(dstI32[1] >>> 0, 1 << 10);
  assert.equal(bridge.getStats().waitAsyncWakeups, 1);
  assert.deepEqual(waitAsyncTimeouts, [250, 250]);

  bridge.stop();

  // 2. Explicit pollIntervalMs and waitAsyncTimeoutMs are honored verbatim, and disabled waitAsync keeps 16ms default.
  const explicitIntervals = [];
  const explicitWaitTimeouts = [];
  const bExplicit = startInputSabBridge({
    inputSab: sab,
    module: { wasmMemory, _playgta6_get_input_state_ptr: () => 128n },
    pollIntervalMs: 16,
    waitAsyncTimeoutMs: 400,
    atomicsObj: {
      ...mockAtomics,
      waitAsync: (_arr, _idx, _expected, timeoutMs) => {
        explicitWaitTimeouts.push(timeoutMs);
        return { async: true, value: new Promise(() => {}) };
      },
    },
    setIntervalFn: (fn, ms) => {
      explicitIntervals.push(ms);
      return 1;
    },
    clearIntervalFn: () => {},
  });
  assert.deepEqual(explicitIntervals, [16], 'explicit pollIntervalMs must be preserved');
  assert.deepEqual(explicitWaitTimeouts, [400], 'explicit waitAsyncTimeoutMs must be preserved');
  bExplicit.stop();

  const noWaitIntervals = [];
  const bNoWait = startInputSabBridge({
    inputSab: sab,
    module: { wasmMemory, _playgta6_get_input_state_ptr: () => 128n },
    useWaitAsync: false,
    setIntervalFn: (fn, ms) => {
      noWaitIntervals.push(ms);
      return 2;
    },
    clearIntervalFn: () => {},
  });
  assert.deepEqual(noWaitIntervals, [16], 'fallback cadence must stay 16ms when waitAsync is disabled');
  bNoWait.stop();
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

test('W345/Wave-5 createWorkerBootInitPayloads enables lightweight creation replay while disabling WriteBuffer shadow replay by default and honors explicit overrides', () => {
  const defaultPayloads = createWorkerBootInitPayloads({
    canvas: 'c',
    width: 1920,
    height: 1080,
    dataBase: '/data/',
  });
  assert.equal(defaultPayloads.gpuInit.enableResourceReplay, true);
  assert.equal(defaultPayloads.gpuInit.enableShadowBufferReplay, false);
  assert.equal('enableResourceReplay' in defaultPayloads.ioInit, false);
  assert.equal('enableShadowBufferReplay' in defaultPayloads.ioInit, false);

  const optedIn = createWorkerBootInitPayloads({
    canvas: 'c',
    width: 1920,
    height: 1080,
    dataBase: '/data/',
    enableResourceReplay: true,
    enableShadowBufferReplay: true,
  });
  assert.equal(optedIn.gpuInit.enableResourceReplay, true);
  assert.equal(optedIn.gpuInit.enableShadowBufferReplay, true);
  assert.equal('enableResourceReplay' in optedIn.ioInit, false);
  assert.equal('enableShadowBufferReplay' in optedIn.ioInit, false);

  const optedOut = createWorkerBootInitPayloads({
    canvas: 'c',
    width: 1920,
    height: 1080,
    dataBase: '/data/',
    enableResourceReplay: false,
    enableShadowBufferReplay: false,
  });
  assert.equal(optedOut.gpuInit.enableResourceReplay, false);
  assert.equal(optedOut.gpuInit.enableShadowBufferReplay, false);

  // Non-boolean values fall back safely to their respective defaults.
  const nonBoolean = createWorkerBootInitPayloads({
    canvas: 'c',
    width: 1920,
    height: 1080,
    dataBase: '/data/',
    enableResourceReplay: 'yes',
    enableShadowBufferReplay: 'yes',
  });
  assert.equal(nonBoolean.gpuInit.enableResourceReplay, true);
  assert.equal(nonBoolean.gpuInit.enableShadowBufferReplay, false);
});

test('W345 toTransferableArrayBuffer avoids copying standalone non-shared buffers and safely copies subviews/SABs', () => {
  // 1. Standalone ArrayBuffer -> returned directly (same reference, zero copy).
  const standaloneAb = new ArrayBuffer(16);
  new Uint8Array(standaloneAb).set([1, 2, 3, 4]);
  assert.equal(toTransferableArrayBuffer(standaloneAb), standaloneAb);

  // 2. Full-span Uint8Array / DataView / Int32Array -> backing ArrayBuffer returned directly.
  const fullU8 = new Uint8Array([10, 20, 30, 40]);
  assert.equal(toTransferableArrayBuffer(fullU8), fullU8.buffer);
  const fullDv = new DataView(fullU8.buffer);
  assert.equal(toTransferableArrayBuffer(fullDv), fullU8.buffer);
  const fullI32 = new Int32Array(fullU8.buffer);
  assert.equal(toTransferableArrayBuffer(fullI32), fullU8.buffer);

  // 3. Empty standalone Uint8Array -> backing ArrayBuffer returned directly.
  const emptyU8 = new Uint8Array(0);
  assert.equal(toTransferableArrayBuffer(emptyU8), emptyU8.buffer);

  // 4. Subview of larger ArrayBuffer -> copied into an exact-length standalone ArrayBuffer.
  const big = new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7]);
  const sub = big.subarray(2, 6);
  const subBuf = toTransferableArrayBuffer(sub);
  assert.ok(subBuf instanceof ArrayBuffer);
  assert.notEqual(subBuf, big.buffer);
  assert.deepEqual(new Uint8Array(subBuf), new Uint8Array([2, 3, 4, 5]));

  // 5. SharedArrayBuffer and SharedArrayBuffer-backed view -> copied into a non-shared ArrayBuffer.
  const sab = new SharedArrayBuffer(8);
  new Uint8Array(sab).set([9, 8, 7, 6, 5, 4, 3, 2]);
  const sabCopy = toTransferableArrayBuffer(sab);
  assert.ok(sabCopy instanceof ArrayBuffer);
  assert.equal(sabCopy instanceof SharedArrayBuffer, false);
  assert.deepEqual(new Uint8Array(sabCopy), new Uint8Array([9, 8, 7, 6, 5, 4, 3, 2]));

  const sabView = new Uint8Array(sab, 0, 8);
  const sabViewCopy = toTransferableArrayBuffer(sabView);
  assert.ok(sabViewCopy instanceof ArrayBuffer);
  assert.equal(sabViewCopy instanceof SharedArrayBuffer, false);
  assert.deepEqual(new Uint8Array(sabViewCopy), new Uint8Array([9, 8, 7, 6, 5, 4, 3, 2]));

  // 6. Nullish / invalid inputs return null.
  assert.equal(toTransferableArrayBuffer(null), null);
  assert.equal(toTransferableArrayBuffer(undefined), null);
  assert.equal(toTransferableArrayBuffer({}), null);
});

test('W345 coordinator wgpuSubmit transfers standalone ArrayBuffer with zero extra copy and copies subviews/SAB views', () => {
  const posted = [];
  const mockGpuWorker = {
    postMessage(msg, transfer) {
      posted.push({ msg, transfer });
    },
  };

  const cfg = buildRuntimeModuleConfig({
    gpuWorker: mockGpuWorker,
    reportFn: () => {},
  });

  // Standalone Uint8Array (same shape produced by copyMemoryBytes) -> zero copy, exact buffer identity!
  const standalone = new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]);
  let sliceCalls = 0;
  standalone.slice = () => {
    sliceCalls += 1;
    throw new Error('W345 regression: redundant bytes.slice() called on standalone Uint8Array');
  };
  cfg.hostCallbacks.wgpuSubmit(standalone);
  assert.equal(sliceCalls, 0);
  assert.equal(posted.length, 1);
  assert.equal(posted[0].msg.cmdBuffer, standalone.buffer);
  assert.deepEqual(posted[0].transfer, [standalone.buffer]);

  // Subview -> safely copied into a standalone exact-length ArrayBuffer.
  const backing = new Uint8Array([99, 98, 7, 0, 0, 0, 0, 0, 0, 0, 97]);
  const subview = backing.subarray(2, 10);
  cfg.hostCallbacks.wgpuSubmit(subview);
  assert.equal(posted.length, 2);
  assert.notEqual(posted[1].msg.cmdBuffer, backing.buffer);
  assert.equal(posted[1].msg.cmdBuffer.byteLength, 8);
  assert.deepEqual(new Uint8Array(posted[1].msg.cmdBuffer), subview);
  assert.deepEqual(posted[1].transfer, [posted[1].msg.cmdBuffer]);

  // SharedArrayBuffer-backed view -> safely copied into a transferable ArrayBuffer.
  const sab = new SharedArrayBuffer(8);
  const sabView = new Uint8Array(sab);
  sabView.set([7, 0, 0, 0, 0, 0, 0, 0]);
  cfg.hostCallbacks.wgpuSubmit(sabView);
  assert.equal(posted.length, 3);
  assert.ok(posted[2].msg.cmdBuffer instanceof ArrayBuffer);
  assert.equal(posted[2].msg.cmdBuffer instanceof SharedArrayBuffer, false);
  assert.deepEqual(new Uint8Array(posted[2].msg.cmdBuffer), new Uint8Array([7, 0, 0, 0, 0, 0, 0, 0]));
  assert.deepEqual(posted[2].transfer, [posted[2].msg.cmdBuffer]);
});

test('W349 forwardGpuWorkerMessage forwards first-visible-frame/visible-frame and preserves firstVisibleFrame fields on gpu-telemetry', () => {
  const posted = [];
  const post = (msg) => posted.push(msg);

  // 1. Explicit first-visible-frame message from wgpu_worker is forwarded verbatim with all probe fields.
  const adapterObj = {
    vendor: 'nvidia',
    architecture: 'turing',
    device: 'geforce',
    description: 'NVIDIA GeForce',
    label: 'nvidia turing',
    isFallbackAdapter: false,
  };
  const visibleMsg = forwardGpuWorkerMessage(
    {
      type: 'first-visible-frame',
      firstVisibleFrame: true,
      firstVisibleFrameMs: 45120.5,
      firstPresentMs: 320.25,
      blackLoadingMs: 44800.25,
      presentCount: 2700,
      realPackDrawsThisFrame: 18,
      fallbackDrawsThisFrame: 2,
      drawsThisFrame: 20,
      source: 'swapchain_probe',
      maxLuma: 235,
      suppressedBlackFrames: 2699,
      gpuAdapter: adapterObj,
    },
    post,
  );
  assert.equal(posted.length, 1);
  assert.deepEqual(visibleMsg, posted[0]);
  assert.equal(visibleMsg.type, 'first-visible-frame');
  assert.equal(visibleMsg.firstVisibleFrame, true);
  assert.equal(visibleMsg.firstVisibleFrameMs, 45120.5);
  assert.equal(visibleMsg.firstPresentMs, 320.25);
  assert.equal(visibleMsg.blackLoadingMs, 44800.25);
  assert.equal(visibleMsg.presentCount, 2700);
  assert.equal(visibleMsg.realPackDrawsThisFrame, 18);
  assert.equal(visibleMsg.fallbackDrawsThisFrame, 2);
  assert.equal(visibleMsg.drawsThisFrame, 20);
  assert.equal(visibleMsg.source, 'swapchain_probe');
  assert.equal(visibleMsg.maxLuma, 235);
  assert.equal(visibleMsg.suppressedBlackFrames, 2699);
  assert.deepEqual(visibleMsg.gpuAdapter, adapterObj);

  // 2. visible-frame alias is also forwarded preserving string gpuAdapter & timing fields.
  const visibleAlias = forwardGpuWorkerMessage(
    {
      type: 'visible-frame',
      firstVisibleFrame: true,
      firstVisibleFrameMs: 1200,
      firstPresentMs: 100,
      blackLoadingMs: 1100,
      presentCount: 60,
      source: 'pixel_readback',
      maxLuma: 192,
      suppressedBlackFrames: 59,
      gpuAdapter: 'amd rdna2',
    },
    post,
  );
  assert.equal(posted.length, 2);
  assert.equal(visibleAlias.type, 'visible-frame');
  assert.equal(visibleAlias.firstVisibleFrame, true);
  assert.equal(visibleAlias.firstVisibleFrameMs, 1200);
  assert.equal(visibleAlias.firstPresentMs, 100);
  assert.equal(visibleAlias.blackLoadingMs, 1100);
  assert.equal(visibleAlias.presentCount, 60);
  assert.equal(visibleAlias.source, 'pixel_readback');
  assert.equal(visibleAlias.maxLuma, 192);
  assert.equal(visibleAlias.suppressedBlackFrames, 59);
  assert.equal(visibleAlias.gpuAdapter, 'amd rdna2');

  // 3. gpu-telemetry from nested wgpu_worker { id, stats } preserves explicit firstVisibleFrame: false!
  const blackLoadingStats = forwardGpuWorkerMessage(
    {
      id: 11,
      stats: {
        frameCount: 45,
        drawCalls: 120,
        realPackDrawCount: 42,
        firstVisibleFrame: false,
        firstVisibleFrameMs: null,
        blackLoadingMs: null,
        lastSampledMaxLuma: 0,
        suppressedBlackFrames: 45,
        gpuAdapter: adapterObj,
      },
      gpuAdapter: adapterObj,
      timestamp: 1700000010000,
    },
    post,
  );
  assert.equal(posted.length, 3);
  assert.equal(blackLoadingStats.type, 'gpu-telemetry');
  assert.equal(blackLoadingStats.frameStats.realPackDrawCount, 42);
  assert.equal(blackLoadingStats.firstVisibleFrame, false);
  assert.equal(blackLoadingStats.firstVisibleFrameMs, null);
  assert.equal(blackLoadingStats.blackLoadingMs, null);
  assert.equal(blackLoadingStats.maxLuma, 0);
  assert.equal(blackLoadingStats.suppressedBlackFrames, 45);
  assert.deepEqual(blackLoadingStats.gpuAdapter, adapterObj);

  // 4. Flat gpu-stats / gpu-telemetry with firstVisibleFrame: true preserves timing & adapter fields.
  const visibleStats = forwardGpuWorkerMessage(
    {
      type: 'gpu-stats',
      framesPresented: 90,
      realPackDrawCount: 250,
      firstVisibleFrame: true,
      firstVisibleFrameMs: 8400,
      blackLoadingMs: 8100,
      maxLuma: 224,
      suppressedBlackFrames: 89,
      gpuAdapter: 'nvidia turing',
      timestamp: 1700000020000,
    },
    post,
  );
  assert.equal(posted.length, 4);
  assert.equal(visibleStats.type, 'gpu-telemetry');
  assert.equal(visibleStats.firstVisibleFrame, true);
  assert.equal(visibleStats.firstVisibleFrameMs, 8400);
  assert.equal(visibleStats.blackLoadingMs, 8100);
  assert.equal(visibleStats.maxLuma, 224);
  assert.equal(visibleStats.suppressedBlackFrames, 89);
  assert.equal(visibleStats.gpuAdapter, 'nvidia turing');
});

test('W349 loader.js gpuWorker.onmessage forwards firstVisibleFrame: false telemetry and first-visible-frame events end-to-end', () => {
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

  workerSelf.onmessage({
    data: {
      canvas: { id: 'c' },
      width: 1280,
      height: 720,
      dataBase: '/data/',
    },
  });

  const gpuWorker = workersByScript.get('./workers/wgpu_worker.js');
  assert.ok(gpuWorker && typeof gpuWorker.onmessage === 'function');

  // 1. Black-frame stats reply preserves explicit firstVisibleFrame: false despite realPackDrawCount > 0.
  gpuWorker.onmessage({
    data: {
      id: 1,
      stats: {
        frameCount: 15,
        realPackDrawCount: 42,
        firstVisibleFrame: false,
        firstVisibleFrameMs: null,
        blackLoadingMs: null,
        lastSampledMaxLuma: 0,
        suppressedBlackFrames: 15,
      },
      gpuAdapter: { vendor: 'nvidia', architecture: 'turing', label: 'nvidia turing' },
    },
  });

  // 2. First non-black frame event from wgpu_worker arrives verbatim on mainMessages.
  gpuWorker.onmessage({
    data: {
      type: 'first-visible-frame',
      firstVisibleFrame: true,
      firstVisibleFrameMs: 45100,
      firstPresentMs: 250,
      blackLoadingMs: 44850,
      presentCount: 2700,
      source: 'swapchain_probe',
      maxLuma: 235,
      suppressedBlackFrames: 2699,
      gpuAdapter: { vendor: 'nvidia', architecture: 'turing', label: 'nvidia turing' },
    },
  });

  assert.equal(mainMessages.length, 2);
  assert.equal(mainMessages[0].type, 'gpu-telemetry');
  assert.equal(mainMessages[0].firstVisibleFrame, false);
  assert.equal(mainMessages[0].frameStats.realPackDrawCount, 42);
  assert.equal(mainMessages[0].suppressedBlackFrames, 15);
  assert.equal(mainMessages[0].maxLuma, 0);
  assert.equal(mainMessages[0].gpuAdapter.label, 'nvidia turing');

  assert.equal(mainMessages[1].type, 'first-visible-frame');
  assert.equal(mainMessages[1].firstVisibleFrame, true);
  assert.equal(mainMessages[1].firstVisibleFrameMs, 45100);
  assert.equal(mainMessages[1].firstPresentMs, 250);
  assert.equal(mainMessages[1].blackLoadingMs, 44850);
  assert.equal(mainMessages[1].presentCount, 2700);
  assert.equal(mainMessages[1].source, 'swapchain_probe');
  assert.equal(mainMessages[1].maxLuma, 235);
  assert.equal(mainMessages[1].suppressedBlackFrames, 2699);
  assert.equal(mainMessages[1].gpuAdapter.label, 'nvidia turing');
});

test('guestRelaunchPayload only answers the self-relaunch exit code', () => {
  const { guestRelaunchPayload, RELAUNCH_SELF_EXIT_CODE, GUEST_RELAUNCH_MESSAGE_TYPE } = require('../web/loader.js');
  assert.equal(RELAUNCH_SELF_EXIT_CODE, 0x52454c4e);
  assert.deepEqual(guestRelaunchPayload(0x52454c4e), {
    type: GUEST_RELAUNCH_MESSAGE_TYPE,
    exitCode: RELAUNCH_SELF_EXIT_CODE,
  });
  assert.equal(guestRelaunchPayload(0), null);
  assert.equal(guestRelaunchPayload(1), null);
  assert.equal(guestRelaunchPayload(undefined), null);
  assert.equal(guestRelaunchPayload(Number.NaN), null);
});

test('W358 forwardGpuWorkerMessage forwards swapchainProbe* telemetry on first-visible-frame, visible-frame, and gpu-telemetry', () => {
  const posted = [];
  const post = (msg) => posted.push(msg);

  // 1. first-visible-frame with top-level camelCase non-zero probe counts & booleans.
  const visibleNonZero = forwardGpuWorkerMessage(
    {
      type: 'first-visible-frame',
      firstVisibleFrame: true,
      firstVisibleFrameMs: 12345,
      source: 'swapchain_probe',
      maxLuma: 210,
      suppressedBlackFrames: 40,
      swapchainProbeCount: 14,
      swapchainProbeInFlight: false,
      swapchainProbeSupported: true,
      swapchainProbeErrorCount: 2,
    },
    post,
  );
  assert.equal(posted.length, 1);
  assert.deepEqual(visibleNonZero, posted[0]);
  assert.equal(visibleNonZero.swapchainProbeCount, 14);
  assert.equal(visibleNonZero.swapchainProbeInFlight, false);
  assert.equal(visibleNonZero.swapchainProbeSupported, true);
  assert.equal(visibleNonZero.swapchainProbeErrorCount, 2);

  // 2. first-visible-frame preserves explicit 0, false, and null (not dropped as falsy).
  const visibleZerosAndNull = forwardGpuWorkerMessage(
    {
      type: 'first-visible-frame',
      firstVisibleFrame: true,
      swapchainProbeCount: 0,
      swapchainProbeInFlight: false,
      swapchainProbeSupported: null,
      swapchainProbeErrorCount: 0,
    },
    post,
  );
  assert.equal(posted.length, 2);
  assert.equal(visibleZerosAndNull.swapchainProbeCount, 0);
  assert.equal(visibleZerosAndNull.swapchainProbeInFlight, false);
  assert.equal(visibleZerosAndNull.swapchainProbeSupported, null);
  assert.equal(visibleZerosAndNull.swapchainProbeErrorCount, 0);

  // 3. visible-frame alias with snake_case fields nested inside stats.
  const visibleSnakeNested = forwardGpuWorkerMessage(
    {
      type: 'visible-frame',
      firstVisibleFrame: true,
      stats: {
        swapchain_probe_count: 7,
        swapchain_probe_in_flight: true,
        swapchain_probe_supported: false,
        swapchain_probe_error_count: 3,
      },
    },
    post,
  );
  assert.equal(posted.length, 3);
  assert.equal(visibleSnakeNested.type, 'visible-frame');
  assert.equal(visibleSnakeNested.swapchainProbeCount, 7);
  assert.equal(visibleSnakeNested.swapchainProbeInFlight, true);
  assert.equal(visibleSnakeNested.swapchainProbeSupported, false);
  assert.equal(visibleSnakeNested.swapchainProbeErrorCount, 3);

  // 4. gpu-telemetry from nested wgpu_worker { id, stats } preserves 0, false, and null.
  const statsZeroNull = forwardGpuWorkerMessage(
    {
      id: 21,
      stats: {
        frameCount: 10,
        firstVisibleFrame: false,
        swapchainProbeCount: 0,
        swapchainProbeInFlight: false,
        swapchainProbeSupported: null,
        swapchainProbeErrorCount: 0,
      },
      timestamp: 1700000030000,
    },
    post,
  );
  assert.equal(posted.length, 4);
  assert.equal(statsZeroNull.type, 'gpu-telemetry');
  assert.equal(statsZeroNull.swapchainProbeCount, 0);
  assert.equal(statsZeroNull.swapchainProbeInFlight, false);
  assert.equal(statsZeroNull.swapchainProbeSupported, null);
  assert.equal(statsZeroNull.swapchainProbeErrorCount, 0);

  // 5. gpu-telemetry from nested wgpu_worker { id, stats } forwards active non-zero probe state.
  const statsActive = forwardGpuWorkerMessage(
    {
      id: 22,
      stats: {
        frameCount: 95,
        firstVisibleFrame: false,
        swapchainProbeCount: 19,
        swapchainProbeInFlight: true,
        swapchainProbeSupported: true,
        swapchainProbeErrorCount: 1,
        lastSampledMaxLuma: 0,
        suppressedBlackFrames: 95,
      },
      timestamp: 1700000031000,
    },
    post,
  );
  assert.equal(posted.length, 5);
  assert.equal(statsActive.type, 'gpu-telemetry');
  assert.equal(statsActive.swapchainProbeCount, 19);
  assert.equal(statsActive.swapchainProbeInFlight, true);
  assert.equal(statsActive.swapchainProbeSupported, true);
  assert.equal(statsActive.swapchainProbeErrorCount, 1);

  // 6. gpu-telemetry with snake_case fields in nested frameStats sanitizes integers & booleans.
  const frameStatsSnake = forwardGpuWorkerMessage(
    {
      type: 'gpu-telemetry',
      frameStats: {
        framesPresented: 30,
        swapchain_probe_count: 5.9,
        swapchain_probe_in_flight: false,
        swapchain_probe_supported: false,
        swapchain_probe_error_count: 2.4,
      },
      timestamp: 1700000032000,
    },
    post,
  );
  assert.equal(posted.length, 6);
  assert.equal(frameStatsSnake.type, 'gpu-telemetry');
  assert.equal(frameStatsSnake.swapchainProbeCount, 5);
  assert.equal(frameStatsSnake.swapchainProbeInFlight, false);
  assert.equal(frameStatsSnake.swapchainProbeSupported, false);
  assert.equal(frameStatsSnake.swapchainProbeErrorCount, 2);

  // 7. Invalid top-level values fall through cleanly to nested stats; absent fields stay omitted.
  const fallbackFromInvalid = forwardGpuWorkerMessage(
    {
      type: 'gpu-stats',
      framesPresented: 40,
      swapchainProbeCount: Number.NaN,
      swapchainProbeInFlight: 'yes',
      swapchainProbeSupported: 123,
      swapchainProbeErrorCount: null,
      stats: {
        swapchain_probe_count: 8,
        swapchain_probe_in_flight: true,
        swapchain_probe_supported: null,
        swapchain_probe_error_count: 4,
      },
    },
    post,
  );
  assert.equal(posted.length, 7);
  assert.equal(fallbackFromInvalid.swapchainProbeCount, 8);
  assert.equal(fallbackFromInvalid.swapchainProbeInFlight, true);
  assert.equal(fallbackFromInvalid.swapchainProbeSupported, null);
  assert.equal(fallbackFromInvalid.swapchainProbeErrorCount, 4);

  const withoutProbeFields = forwardGpuWorkerMessage({ framesPresented: 12 }, post);
  assert.equal(posted.length, 8);
  assert.equal('swapchainProbeCount' in withoutProbeFields, false);
  assert.equal('swapchainProbeInFlight' in withoutProbeFields, false);
  assert.equal('swapchainProbeSupported' in withoutProbeFields, false);
  assert.equal('swapchainProbeErrorCount' in withoutProbeFields, false);
});

test('W358 loader.js gpuWorker.onmessage forwards swapchainProbe* telemetry end-to-end', () => {
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

  workerSelf.onmessage({
    data: {
      canvas: { id: 'c' },
      width: 1280,
      height: 720,
      dataBase: '/data/',
    },
  });

  const gpuWorker = workersByScript.get('./workers/wgpu_worker.js');
  assert.ok(gpuWorker && typeof gpuWorker.onmessage === 'function');

  // 1. gpu-telemetry stats message with live swapchain probe counts.
  gpuWorker.onmessage({
    data: {
      id: 101,
      stats: {
        frameCount: 64,
        firstVisibleFrame: false,
        lastSampledMaxLuma: 0,
        suppressedBlackFrames: 64,
        swapchainProbeCount: 11,
        swapchainProbeInFlight: true,
        swapchainProbeSupported: true,
        swapchainProbeErrorCount: 0,
      },
    },
  });

  // 2. first-visible-frame message with final probe counts.
  gpuWorker.onmessage({
    data: {
      type: 'first-visible-frame',
      firstVisibleFrame: true,
      firstVisibleFrameMs: 18400,
      source: 'swapchain_probe',
      maxLuma: 228,
      suppressedBlackFrames: 64,
      swapchainProbeCount: 12,
      swapchainProbeInFlight: false,
      swapchainProbeSupported: true,
      swapchainProbeErrorCount: 1,
    },
  });

  assert.equal(mainMessages.length, 2);
  assert.equal(mainMessages[0].type, 'gpu-telemetry');
  assert.equal(mainMessages[0].swapchainProbeCount, 11);
  assert.equal(mainMessages[0].swapchainProbeInFlight, true);
  assert.equal(mainMessages[0].swapchainProbeSupported, true);
  assert.equal(mainMessages[0].swapchainProbeErrorCount, 0);

  assert.equal(mainMessages[1].type, 'first-visible-frame');
  assert.equal(mainMessages[1].swapchainProbeCount, 12);
  assert.equal(mainMessages[1].swapchainProbeInFlight, false);
  assert.equal(mainMessages[1].swapchainProbeSupported, true);
  assert.equal(mainMessages[1].swapchainProbeErrorCount, 1);
});

test('W360 buildRuntimeModuleConfig posts attach-guest-memory without wgpuSubmit and re-posts after WebAssembly.Memory.grow()', () => {
  // 1. options.memory already provided -> attaches immediately, no duplicate on unchanged check, re-attaches after grow().
  const posted1 = [];
  const mockGpuWorker1 = {
    postMessage(msg) {
      posted1.push(msg);
    },
  };
  const sharedMem1 = new WebAssembly.Memory({ initial: 1, maximum: 4, shared: true });
  const initialBuf1 = sharedMem1.buffer;
  const cfg1 = buildRuntimeModuleConfig({
    gpuWorker: mockGpuWorker1,
    memory: sharedMem1,
    reportFn: () => {},
  });

  assert.equal(posted1.length, 1);
  assert.equal(posted1[0].type, 'attach-guest-memory');
  assert.equal(posted1[0].buffer, initialBuf1);

  // Repeated check with unchanged buffer is an O(1) no-op.
  assert.equal(cfg1.attachGuestMemory(), false);
  assert.equal(posted1.length, 1);

  // Growing shared memory produces a new SharedArrayBuffer; re-attaching posts the new buffer.
  sharedMem1.grow(1);
  const grownBuf1 = sharedMem1.buffer;
  assert.notEqual(grownBuf1, initialBuf1);
  assert.equal(cfg1.attachGuestMemory(), true);
  assert.equal(posted1.length, 2);
  assert.equal(posted1[1].type, 'attach-guest-memory');
  assert.equal(posted1[1].buffer, grownBuf1);

  // 2. Real-Chrome path (no options.memory): preRun / onRuntimeInitialized attach without wgpuSubmit ever being called.
  const posted2 = [];
  const mockGpuWorker2 = {
    postMessage(msg) {
      posted2.push(msg);
    },
  };
  const sharedMem2 = new WebAssembly.Memory({ initial: 1, maximum: 4, shared: true });
  const cfg2 = buildRuntimeModuleConfig({
    gpuWorker: mockGpuWorker2,
    reportFn: () => {},
  });
  assert.equal(posted2.length, 0);

  const mockRuntimeMod = { wasmMemory: sharedMem2 };
  cfg2.preRun[0](mockRuntimeMod);
  assert.equal(posted2.length, 1);
  assert.equal(posted2[0].type, 'attach-guest-memory');
  assert.equal(posted2[0].buffer, sharedMem2.buffer);

  // onRuntimeInitialized with unchanged memory does not duplicate.
  cfg2.onRuntimeInitialized.call(mockRuntimeMod);
  assert.equal(posted2.length, 1);

  // After Memory.grow(), host_log (or onRuntimeInitialized) automatically re-sends the grown SAB.
  sharedMem2.grow(1);
  const grownBuf2 = sharedMem2.buffer;
  cfg2.hostImports.env.host_log(0, 0);
  assert.equal(posted2.length, 2);
  assert.equal(posted2[1].type, 'attach-guest-memory');
  assert.equal(posted2[1].buffer, grownBuf2);
});

test('W360 loader.js SPSC gpu_ring boot attaches guest memory on runtime boot and re-attaches after grow() without wgpuSubmit', async () => {
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

  const sharedMem = new WebAssembly.Memory({ initial: 1, maximum: 4, shared: true });
  const initialBuf = sharedMem.buffer;
  let runtimeBootedResolve = null;
  const runtimeBooted = new Promise((resolve) => {
    runtimeBootedResolve = resolve;
  });

  const workerSelf = {
    onmessage: null,
    postMessage() {},
    createRuntimeModule: async (moduleConfig) => {
      const mod = { wasmMemory: sharedMem };
      if (Array.isArray(moduleConfig.preRun)) {
        for (const fn of moduleConfig.preRun) fn(mod);
      }
      if (typeof moduleConfig.onRuntimeInitialized === 'function') {
        moduleConfig.onRuntimeInitialized.call(mod);
      }
      runtimeBootedResolve(mod);
      return mod;
    },
  };

  const sandbox = {
    self: workerSelf,
    globalThis: null,
    BroadcastChannel: MockBroadcastChannel,
    Worker: MockWorker,
    indexedDB: null,
    WebAssembly,
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
      if (String(url).includes('dist/runtime.js')) {
        return { ok: true, status: 200 };
      }
      return { ok: false, status: 404 };
    },
    importScripts(scriptPath) {
      if (String(scriptPath).includes('dist/runtime.js')) {
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
    },
  });

  const ioWorker = workersByScript.get('./workers/io_worker.js');
  const gpuWorker = workersByScript.get('./workers/wgpu_worker.js');
  const audioWorker = workersByScript.get('./workers/audio_worker.js');
  assert.ok(ioWorker && gpuWorker && audioWorker);

  // Mark all 3 workers ready to trigger tryBootWasmRuntime() — wgpuSubmit is NEVER called!
  ioWorker.onmessage({ data: { ready: true, summary: 'ok' } });
  gpuWorker.onmessage({ data: { ready: true, adapter: 'mock' } });
  audioWorker.onmessage({ data: { ready: true } });

  await runtimeBooted;
  await new Promise((r) => setTimeout(r, 10));

  const attachPosts1 = gpuWorker.posted.filter((m) => m && m.type === 'attach-guest-memory');
  assert.equal(attachPosts1.length, 1);
  assert.equal(attachPosts1[0].buffer, initialBuf);

  // Grow guest memory, then notify GPU ring drain via loader -> new SAB must be posted BEFORE drain-ring!
  sharedMem.grow(1);
  const grownBuf = sharedMem.buffer;
  assert.notEqual(grownBuf, initialBuf);

  workerSelf.onmessage({ data: { type: 'drain-ring' } });
  const attachPosts2 = gpuWorker.posted.filter((m) => m && m.type === 'attach-guest-memory');
  assert.equal(attachPosts2.length, 2);
  assert.equal(attachPosts2[1].buffer, grownBuf);
  const lastTwo = gpuWorker.posted.slice(-2);
  assert.equal(lastTwo[0].type, 'attach-guest-memory');
  assert.equal(lastTwo[0].buffer, grownBuf);
  assert.equal(lastTwo[1].type, 'drain-ring');
});

test('Orch-13 createWorkerBootInitPayloads forwards devicePixelRatio and swapchainOptions when provided and omits when absent', () => {
  // Absent or invalid values are omitted from gpuInit and ioInit.
  const omitted = createWorkerBootInitPayloads({
    canvas: 'offscreen',
    width: 1280,
    height: 720,
    dataBase: '/data/',
  });
  assert.equal('devicePixelRatio' in omitted.gpuInit, false);
  assert.equal('swapchainOptions' in omitted.gpuInit, false);
  assert.equal('devicePixelRatio' in omitted.ioInit, false);
  assert.equal('swapchainOptions' in omitted.ioInit, false);

  const invalid = createWorkerBootInitPayloads({
    canvas: 'offscreen',
    width: 1280,
    height: 720,
    dataBase: '/data/',
    devicePixelRatio: 0,
    swapchainOptions: ['not-an-object'],
  });
  assert.equal('devicePixelRatio' in invalid.gpuInit, false);
  assert.equal('swapchainOptions' in invalid.gpuInit, false);

  // Valid values are forwarded to gpuInit (with shallow-cloned swapchainOptions) and omitted from ioInit.
  const swapOpts = { alphaMode: 'opaque', powerPreference: 'high-performance' };
  const forwarded = createWorkerBootInitPayloads({
    canvas: 'offscreen',
    width: 2560,
    height: 1440,
    dataBase: '/data/',
    devicePixelRatio: 1,
    swapchainOptions: swapOpts,
  });
  assert.equal(forwarded.gpuInit.devicePixelRatio, 1);
  assert.deepEqual(forwarded.gpuInit.swapchainOptions, {
    alphaMode: 'opaque',
    powerPreference: 'high-performance',
  });
  assert.notEqual(forwarded.gpuInit.swapchainOptions, swapOpts);
  assert.equal('devicePixelRatio' in forwarded.ioInit, false);
  assert.equal('swapchainOptions' in forwarded.ioInit, false);
});

test('Orch-13 loader.js gpu-validation-error emits gpuValidationForwarded: true on BroadcastChannel while forwarding structured message to main thread', () => {
  const mainMessages = [];
  const progressMessages = [];
  const workersByScript = new Map();

  class MockBroadcastChannel {
    constructor(name) {
      this.name = name;
    }
    postMessage(msg) {
      if (this.name === 'game-progress') {
        progressMessages.push(msg);
      }
    }
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

  workerSelf.onmessage({
    data: {
      canvas: { id: 'c' },
      width: 1280,
      height: 720,
      devicePixelRatio: 1,
      swapchainOptions: { alphaMode: 'opaque' },
      dataBase: '/data/',
    },
  });

  const gpuWorker = workersByScript.get('./workers/wgpu_worker.js');
  assert.ok(gpuWorker);
  assert.equal(gpuWorker.posted[0].devicePixelRatio, 1);
  assert.deepEqual({ ...gpuWorker.posted[0].swapchainOptions }, { alphaMode: 'opaque' });

  // Resize forwards optional devicePixelRatio when valid and omits when absent.
  workerSelf.onmessage({ data: { resize: true, width: 1920, height: 1080, devicePixelRatio: 1.5 } });
  workerSelf.onmessage({ data: { resize: true, width: 1600, height: 900 } });
  const resizePosts = gpuWorker.posted.filter((m) => m && m.resize === true);
  assert.deepEqual(
    resizePosts.map((m) => ({ ...m })),
    [
      { resize: true, width: 1920, height: 1080, devicePixelRatio: 1.5 },
      { resize: true, width: 1600, height: 900 },
    ]
  );

  // gpu-validation-error emits gpuValidationForwarded: true on BroadcastChannel AND forwards structured message to main.
  gpuWorker.onmessage({
    data: {
      type: 'gpu-validation-error',
      seq: 3,
      timestampMs: 12345,
      totalCount: 3,
      error: {
        type: 'validation',
        message: 'Bind group entry mismatch',
      },
    },
  });

  const bcErr = progressMessages.find((m) => m && m.gpuValidationForwarded === true);
  assert.ok(bcErr, 'expected BroadcastChannel message with gpuValidationForwarded: true');
  assert.match(bcErr.error, /wgpu_worker:.*Bind group entry mismatch/);

  const mainErr = mainMessages.find((m) => m && m.type === 'gpu-validation-error');
  assert.ok(mainErr, 'expected structured gpu-validation-error forwarded to main thread');
  assert.equal(mainErr.type, 'gpu-validation-error');
  assert.equal(mainErr.error.type, 'validation');
  assert.match(mainErr.error.message, /Bind group entry mismatch/);
});

test('Orch-39 loader.js off-thread em-pthread host_log [fault-record] flushes synchronously to console.error, BroadcastChannel, and txPort even if txPort closes immediately', () => {
  assert.equal(PTHREAD_TX_MSG_FAULT_RECORD, 'fault-record');
  assert.equal(FAULT_RECORD_MESSAGE_TYPE, 'fault-record');
  assert.equal(FAULT_RECORD_PREFIX, '[fault-record]');

  const faultJson =
    '{"kind":"\\u0075nimplemented_slot","va":"0x140004321","unimplemented_slot":{"interface":"IXStore","slot":71},"registers":{"rax":"0x99"},"call_stack":{"frames":[]}}';
  const rawGuestLog = `[fault-record] ${faultJson}`;

  assert.equal(isFaultRecordLine(rawGuestLog), true);
  const parsed = parseFaultRecordLine(rawGuestLog);
  assert.equal(parsed.kind, 'unimplemented_slot');
  assert.equal(parsed.va, '0x140004321');
  assert.equal(parsed.target, '0x140004321');
  assert.equal(parsed.unimplemented_slot.slot, 71);

  // 1. Open txPort: synchronous console.error + direct BroadcastChannel('game-progress') + structured txPort fault-record message.
  const bcMessages = [];
  class MockBC {
    constructor(name) {
      this.name = name;
    }
    postMessage(msg) {
      bcMessages.push({ channel: this.name, msg });
    }
    close() {}
  }
  const txMessages = [];
  const consoleErrors = [];
  const scope = {
    name: 'em-pthread-4',
    BroadcastChannel: MockBC,
    console: {
      error(line) {
        consoleErrors.push(line);
      },
    },
  };
  const fakeTxPort = {
    postMessage(msg) {
      txMessages.push(msg);
    },
    close() {},
  };

  const booted = bootstrapEmscriptenPthreadWorker({
    scope,
    BroadcastChannel: MockBC,
    txPort: fakeTxPort,
    importScriptsFn: () => {},
  });

  const mem = new WebAssembly.Memory({ initial: 1 });
  booted.hostImports.setMemory(mem);
  const bytes = new TextEncoder().encode(rawGuestLog);
  new Uint8Array(mem.buffer, 64, bytes.byteLength).set(bytes);
  booted.hostImports.env.host_log(64, bytes.byteLength);

  const progressBc1 = bcMessages.filter((m) => m.channel === 'game-progress');
  assert.equal(consoleErrors.length, 1);
  assert.ok(consoleErrors[0].includes('[fault-record]'));
  assert.equal(progressBc1.length, 1);
  assert.ok(progressBc1[0].msg.label.includes('[fault-record]'));
  assert.equal(txMessages.length, 1);
  assert.equal(txMessages[0].type, PTHREAD_TX_MSG_FAULT_RECORD);
  assert.equal(txMessages[0].faultRecord.kind, 'unimplemented_slot');
  assert.equal(txMessages[0].broadcastSent, true);
  assert.equal(txMessages[0].consoleEmitted, true);
  assert.equal(txMessages[0].workerName, 'em-pthread-4');
  assert.equal(booted.getLastFaultRecord().va, '0x140004321');
  assert.equal(scope.__playgta6LastFaultRecord.unimplemented_slot.slot, 71);
  booted.cleanup();

  // 2. Broken / closing txPort right before std::abort(): console.error AND direct BroadcastChannel STILL succeed!
  bcMessages.length = 0;
  consoleErrors.length = 0;
  const brokenTxPort = {
    postMessage() {
      throw new Error('port closed on abort');
    },
    close() {},
  };
  const scope2 = {
    name: 'em-pthread-9',
    BroadcastChannel: MockBC,
    console: {
      error(line) {
        consoleErrors.push(line);
      },
    },
  };
  const booted2 = bootstrapEmscriptenPthreadWorker({
    scope: scope2,
    BroadcastChannel: MockBC,
    txPort: brokenTxPort,
    importScriptsFn: () => {},
  });
  booted2.hostImports.setMemory(mem);
  booted2.hostImports.env.host_log(64, bytes.byteLength);

  const progressBc2 = bcMessages.filter((m) => m.channel === 'game-progress');
  assert.equal(consoleErrors.length, 1);
  assert.equal(progressBc2.length, 1);
  assert.ok(progressBc2[0].msg.label.includes('[fault-record]'));
  assert.equal(booted2.getLastFaultRecord().kind, 'unimplemented_slot');
  booted2.cleanup();
});

test('Orch-39 buildRuntimeModuleConfig captures [fault-record] on host_log/printErr/onAbort and immediately flushes MEMFS /pg6_dump on onAbort even if MEMFS disappears after abort', async () => {
  resetLastFaultRecord();
  const memfs = createFakeMemfs();
  const uploaded = [];
  const mainEvents = [];
  const observerEvents = [];
  const consoleErrors = [];
  const consoleLogs = [];
  const reports = [];
  let abortReasonSeen = null;

  // Write 6 files (> maxConcurrentUploads=2) to verify synchronous snapshotting on onAbort.
  for (let i = 0; i < 5; i += 1) {
    memfs.writeFile(`/pg6_dump/native/pso_${i}.bin`, new Uint8Array([i + 1, i + 2]));
  }
  memfs.writeFile('/pg6_dump/manifest.json', '{"count":5}');

  const pendingResolvers = [];
  const cfg = buildRuntimeModuleConfig({
    locationHref: 'http://127.0.0.1:8080/loader.js',
    dumpPipelines: true,
    dumpMaxConcurrentUploads: 2,
    dumpSetIntervalFn: () => 42,
    dumpClearIntervalFn: () => {},
    dumpFetchFn: (url, init) =>
      new Promise((resolve) => {
        uploaded.push({ url, bytes: new Uint8Array(init.body) });
        pendingResolvers.push(() => resolve({ ok: true, status: 200 }));
      }),
    reportFn: (label) => reports.push(label),
    postToMain: (msg) => mainEvents.push(msg),
    onFaultRecord: (rec, meta) => observerEvents.push({ rec, meta }),
    console: {
      error: (line) => consoleErrors.push(line),
      log: (line) => consoleLogs.push(line),
    },
    onAbort: (what) => {
      abortReasonSeen = what;
    },
  });

  const mockModule = { FS: memfs.fs };
  cfg.preRun[0](mockModule);

  // Regular stdout lines still forward to console.log when options.print is omitted.
  cfg.print('regular stdout line');
  assert.deepEqual(consoleLogs, ['regular stdout line']);

  // printErr delivers a compact KV [fault-record] with trailing newline.
  cfg.printErr(
    '[fault-record] kind=unimplemented_slot va=0x140055555 com_slot=71 hard_stop=true\n',
  );
  assert.equal(cfg.getLastFaultRecord().kind, 'unimplemented_slot');
  assert.equal(cfg.getLastFaultRecord().va, '0x140055555');
  assert.equal(cfg.getLastFaultRecord().com_slot, 71);
  assert.equal(consoleErrors.length, 1);
  assert.equal(observerEvents.length, 1);
  assert.equal(mainEvents.length, 1);
  assert.equal(mainEvents[0].type, FAULT_RECORD_MESSAGE_TYPE);
  assert.equal(reports.some((line) => line.includes('[fault-record]')), true);

  // Write one more late shader right before onAbort(), while the first flush pass is still in flight!
  memfs.writeFile('/pg6_dump/dxil/late_before_abort.dxbc', new Uint8Array([0xde, 0xad, 0xbe, 0xef]));

  // Trigger onAbort, then IMMEDIATELY wipe MEMFS AND invoke a second stop() (matching coordinator onAbort)
  // to verify the second stop() never wipes previously snapshotted bytes!
  cfg.onAbort('Aborted(native code called abort())');
  assert.equal(abortReasonSeen, 'Aborted(native code called abort())');
  memfs.files.clear();
  memfs.dirs.clear();
  cfg.getPipelineDumpSyncer().stop().catch(() => {});

  // Re-emitted fault-record event on abort with onAbort: true.
  const abortEvt = mainEvents.find((e) => e.type === FAULT_RECORD_MESSAGE_TYPE && e.onAbort === true);
  assert.ok(abortEvt, 'expected fault-record re-emission on onAbort');
  assert.equal(abortEvt.faultRecord.va, '0x140055555');

  // Drain all in-flight and queued snapshotted uploads after MEMFS was cleared!
  let guard = 0;
  while (pendingResolvers.length > 0 && guard < 32) {
    guard += 1;
    const batch = pendingResolvers.splice(0, pendingResolvers.length);
    for (const done of batch) {
      done();
    }
    await new Promise((r) => setImmediate(r));
  }

  const uploadedUrls = uploaded.map((u) => u.url);
  assert.equal(uploadedUrls.includes('/__pg6_dump/dxil/late_before_abort.dxbc'), true);
  assert.equal(uploadedUrls.includes('/__pg6_dump/manifest.json'), true);
  assert.equal(uploadedUrls.length, 7);
  assert.equal(cfg.getPipelineDumpSyncer().getStats().filesUploaded, 7);

  // Explicit flushPipelineDump() alias works cleanly.
  assert.equal(typeof cfg.flushPipelineDump, 'function');
  assert.equal(typeof cfg.flushPipelineDumps, 'function');
  const postFlushStats = await cfg.flushPipelineDump();
  assert.equal(postFlushStats.filesUploaded, 7);
  resetLastFaultRecord();
});

test('Orch-39 loader.js coordinator VM handles flush-pipeline-dumps message, global __pg6FlushDumps hook, and request-last-fault-record', async () => {
  const mainMessages = [];
  const progressMessages = [];
  const consoleErrors = [];
  const postedDumpUrls = [];
  const workersByScript = new Map();
  const memfs = createFakeMemfs();
  memfs.writeFile('/pg6_dump/native/boot.bin', new Uint8Array([1, 2, 3, 4]));
  memfs.writeFile('/pg6_dump/manifest.json', '{"count":1}');

  class MockBroadcastChannel {
    constructor(name) {
      this.name = name;
    }
    postMessage(msg) {
      if (this.name === 'game-progress') {
        progressMessages.push(msg);
      }
    }
    close() {}
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
    terminate() {}
  }

  const workerSelf = {
    location: { href: 'http://127.0.0.1:8080/loader.js?dumpPipelines=1' },
    onmessage: null,
    postMessage(msg) {
      mainMessages.push(msg);
    },
    createRuntimeModule: async (moduleConfig) => {
      const mod = { FS: memfs.fs };
      moduleConfig.preRun[0](mod);
      // Simulate runtime emitting a [fault-record] over host_log.
      const mem = new WebAssembly.Memory({ initial: 1 });
      moduleConfig.hostImports.setMemory(mem);
      const line = '[fault-record] {"kind":"\\u0074rap","va":"0x140077777","hard_stop":"S1"}';
      const encoded = new TextEncoder().encode(line);
      new Uint8Array(mem.buffer, 128, encoded.byteLength).set(encoded);
      moduleConfig.env.host_log(128, encoded.byteLength);
      return mod;
    },
  };

  const sandbox = {
    self: workerSelf,
    globalThis: null,
    BroadcastChannel: MockBroadcastChannel,
    Worker: MockWorker,
    WebAssembly,
    indexedDB: null,
    console: {
      error: (line) => consoleErrors.push(line),
      warn: () => {},
      log: () => {},
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
    setTimeout,
    clearTimeout,
    setInterval: () => 777,
    clearInterval: () => {},
    fetch: async (url, init) => {
      if (init && init.method === 'HEAD') {
        return { ok: true, status: 200 };
      }
      if (typeof url === 'string' && url.includes('/__pg6_dump/')) {
        postedDumpUrls.push(url);
        return { ok: true, status: 200 };
      }
      return { ok: false, status: 404 };
    },
    importScripts(scriptPath) {
      if (scriptPath === './dist/runtime.js') {
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

  // Global flush hook exposed matching run_browser_boot.py FLUSH_PIPELINE_DUMPS_JS.
  assert.equal(typeof sandbox.__pg6FlushDumps, 'function');
  assert.equal(typeof flushPipelineDump, 'function');
  assert.equal(typeof flushPipelineDumps, 'function');

  workerSelf.onmessage({
    data: {
      canvas: { id: 'c' },
      width: 1280,
      height: 720,
      dumpPipelines: true,
    },
  });

  workersByScript.get('./workers/io_worker.js').onmessage({ data: { ready: true, summary: 'ok' } });
  workersByScript.get('./workers/wgpu_worker.js').onmessage({ data: { ready: true, adapter: 'mock' } });
  workersByScript.get('./workers/audio_worker.js').onmessage({ data: { ready: true } });

  await new Promise((r) => setTimeout(r, 20));

  // Verify fault-record was captured synchronously and sent to main thread.
  const mainFault = mainMessages.find((m) => m && m.type === FAULT_RECORD_MESSAGE_TYPE);
  assert.ok(mainFault, 'expected structured fault-record sent to main thread');
  assert.equal(mainFault.faultRecord.kind, 'trap');
  assert.equal(mainFault.faultRecord.va, '0x140077777');
  assert.equal(consoleErrors.length, 1);

  // Write a new dump file and trigger explicit flush-pipeline-dumps message from main thread.
  memfs.writeFile('/pg6_dump/dxil/extra.dxbc', new Uint8Array([9, 8, 7]));
  workerSelf.onmessage({ data: { type: FLUSH_PIPELINE_DUMPS_MESSAGE_TYPE, id: 'flush-1' } });
  await new Promise((r) => setTimeout(r, 20));

  const flushedReply = mainMessages.find(
    (m) => m && m.type === PIPELINE_DUMPS_FLUSHED_MESSAGE_TYPE && m.id === 'flush-1',
  );
  assert.ok(flushedReply, 'expected pipeline-dumps-flushed reply');
  assert.equal(flushedReply.ok, true);
  assert.equal(flushedReply.stats.filesUploaded, 3);
  assert.equal(postedDumpUrls.includes('/__pg6_dump/dxil/extra.dxbc'), true);

  // Global __pg6FlushDumps hook and request-last-fault-record query both work.
  const globalStats = await sandbox.__pg6FlushDumps();
  assert.equal(globalStats.filesUploaded, 3);

  workerSelf.onmessage({ data: { type: 'request-last-fault-record', id: 'fr-1' } });
  const frReply = mainMessages.find(
    (m) => m && m.type === FAULT_RECORD_MESSAGE_TYPE && m.id === 'fr-1',
  );
  assert.ok(frReply);
  assert.equal(frReply.faultRecord.va, '0x140077777');

  // request-loader-status includes lastFaultRecord and pipelineDumpStats.
  workerSelf.onmessage({ data: { type: 'request-loader-status', id: 'st-1' } });
  const statusReply = mainMessages.find((m) => m && m.type === 'loader-status' && m.id === 'st-1');
  assert.ok(statusReply);
  assert.equal(statusReply.lastFaultRecord.kind, 'trap');
  assert.equal(statusReply.pipelineDumpStats.filesUploaded, 3);
});

test('Orch-32 P2 #5 + P1 #4a: sanitizeFrameStats preserves gpuMemory and createWorkerBootInitPayloads forwards requireNonBlackFrame', () => {
  const forwarded = [];
  const payload = forwardGpuWorkerMessage(
    {
      type: 'gpu-telemetry',
      frameStats: {
        framesPresented: 12,
        droppedFrames: 1,
        gpuMemory: {
          liveBufferBytes: 16 * 1024 * 1024,
          liveTextureBytes: 32 * 1024 * 1024,
          peakBufferBytes: 24 * 1024 * 1024,
          peakTextureBytes: 48 * 1024 * 1024,
          stagingPoolBytes: 4 * 1024 * 1024,
          totalLiveGpuBytes: 52 * 1024 * 1024,
          customPoolBytes: 1024,
          negativeBad: -5,
          hostPathBad: 'C:\\secret\\path',
          boolBad: true,
        },
      },
    },
    (msg) => forwarded.push(msg),
  );

  assert.ok(payload);
  assert.equal(forwarded.length, 1);
  assert.deepEqual(payload.frameStats.gpuMemory, {
    liveBufferBytes: 16 * 1024 * 1024,
    liveTextureBytes: 32 * 1024 * 1024,
    peakBufferBytes: 24 * 1024 * 1024,
    peakTextureBytes: 48 * 1024 * 1024,
    stagingPoolBytes: 4 * 1024 * 1024,
    totalLiveGpuBytes: 52 * 1024 * 1024,
    customPoolBytes: 1024,
  });
  assert.deepEqual(payload.gpuMemory, payload.frameStats.gpuMemory);
  assert.equal(payload.frameStats.vramBytes, 52 * 1024 * 1024);

  // Also verify direct sanitizeFrameStats computes totalLiveGpuBytes if omitted.
  const direct = sanitizeFrameStats({
    type: 'gpu-stats',
    gpuMemory: {
      liveBufferBytes: 1000,
      liveTextureBytes: 2000,
      stagingPoolBytes: 500,
    },
  });
  assert.equal(direct.gpuMemory.totalLiveGpuBytes, 3500);

  // P1 #4a: requireNonBlackFrame: false forwards into gpuInit only (not ioInit).
  const initPayloads = createWorkerBootInitPayloads({
    canvas: { id: 'c' },
    width: 1280,
    height: 720,
    dataBase: '/data/',
    requireNonBlackFrame: false,
  });
  assert.equal(initPayloads.gpuInit.requireNonBlackFrame, false);
  assert.equal(initPayloads.ioInit.requireNonBlackFrame, undefined);
});
