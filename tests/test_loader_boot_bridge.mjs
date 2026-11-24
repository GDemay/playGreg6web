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
  GPU_OPCODE_PRESENT,
  GPU_PACKET_HEADER_BYTES,
  BOOT_INIT_FORWARD_KEYS,
  inspectGpuPacketBatchForPresent,
  normalizeBootHttpfsPath,
  collectBootBridgeInitFields,
  createWorkerBootInitPayloads,
  forwardGpuWorkerMessage,
  createHostImports,
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
