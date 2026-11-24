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
  BOOT_INIT_FORWARD_KEYS,
  isEmscriptenPthreadWorker,
  installPthreadHostImportShim,
  bootstrapEmscriptenPthreadWorker,
  inspectGpuPacketBatchForPresent,
  normalizeBootHttpfsPath,
  collectBootBridgeInitFields,
  createWorkerBootInitPayloads,
  forwardGpuWorkerMessage,
  createHostImports,
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
  ]);
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

test('isEmscriptenPthreadWorker detects em-pthread workers and rejects normal workers', () => {
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

  assert.equal(isEmscriptenPthreadWorker({ name: 'em-pthread' }), true);
  assert.equal(isEmscriptenPthreadWorker({ name: '' }), false);
  assert.equal(isEmscriptenPthreadWorker({ name: 'loader' }), false);
  assert.equal(isEmscriptenPthreadWorker({ name: 'em-pthread-2' }), false);
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
