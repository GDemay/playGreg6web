import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import {
  WgslPackCache,
  buildWgslPackBuffer,
  fnv1a64 as wgpkFnv1a64,
  formatHashHex as wgpkFormatHashHex,
  SHADER_STAGE as WGPK_SHADER_STAGE,
} from '../web/workers/wgsl_pack_loader.js';

const require = createRequire(import.meta.url);
const {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  buildGpuTelemetrySnapshot,
  DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES,
  DEFAULT_STREAM_SHADERS_THRESHOLD_BYTES,
  resolveShaderPackStreamThresholdBytes,
  OPCODES,
  SHADER_STAGE,
} = require('../web/workers/wgpu_worker.js');
const {
  installRenderStatePackets,
  TEXTURE_FORMAT_NAMES: RS_TEXTURE_FORMAT_NAMES,
} = require('../web/workers/render_state_packets.js');

class BinaryCommandBuilder {
  constructor() {
    this._chunks = [];
  }

  _pushPacket(opcode, payloadBytes) {
    const payload = payloadBytes instanceof Uint8Array ? payloadBytes : new Uint8Array(payloadBytes);
    const buf = new ArrayBuffer(8 + payload.byteLength);
    const view = new DataView(buf);
    view.setUint16(0, opcode, true);
    view.setUint16(2, 0, true);
    view.setUint32(4, payload.byteLength, true);
    new Uint8Array(buf, 8).set(payload);
    this._chunks.push(new Uint8Array(buf));
    return this;
  }

  beginFrame() {
    return this._pushPacket(OPCODES.BEGIN_FRAME, new Uint8Array(0));
  }

  clearColor(r, g, b, a) {
    const buf = new ArrayBuffer(16);
    const v = new DataView(buf);
    v.setFloat32(0, r, true);
    v.setFloat32(4, g, true);
    v.setFloat32(8, b, true);
    v.setFloat32(12, a, true);
    return this._pushPacket(OPCODES.CLEAR_COLOR, new Uint8Array(buf));
  }

  createShaderModule(id, stage, code) {
    const encoded = new TextEncoder().encode(code);
    const buf = new ArrayBuffer(16 + encoded.byteLength);
    const v = new DataView(buf);
    v.setUint32(0, id, true);
    v.setUint32(4, stage, true);
    v.setUint32(8, encoded.byteLength, true);
    v.setUint32(12, 0, true);
    new Uint8Array(buf, 16).set(encoded);
    return this._pushPacket(OPCODES.CREATE_SHADER_MODULE, new Uint8Array(buf));
  }

  createRenderPipeline(id, vsShaderId, fsShaderId, vertexStride = 0, topology = 0) {
    const buf = new ArrayBuffer(24);
    const v = new DataView(buf);
    v.setUint32(0, id, true);
    v.setUint32(4, vsShaderId, true);
    v.setUint32(8, fsShaderId, true);
    v.setUint32(12, vertexStride, true);
    v.setUint32(16, topology, true);
    v.setUint32(20, 0, true);
    return this._pushPacket(OPCODES.CREATE_RENDER_PIPELINE, new Uint8Array(buf));
  }

  createComputePipeline(id, csShaderId) {
    const buf = new ArrayBuffer(16);
    const v = new DataView(buf);
    v.setUint32(0, id, true);
    v.setUint32(4, csShaderId, true);
    v.setUint32(8, 0, true);
    v.setUint32(12, 0, true);
    return this._pushPacket(OPCODES.CREATE_COMPUTE_PIPELINE, new Uint8Array(buf));
  }

  createRenderPipeline2(id, vsShaderId, fsShaderId, layoutId = 0, colorFormat = 'bgra8unorm') {
    const buf = new ArrayBuffer(16 + 376);
    const v = new DataView(buf);
    v.setUint32(0, id, true);
    v.setUint32(4, vsShaderId, true);
    v.setUint32(8, fsShaderId, true);
    v.setUint32(12, layoutId, true);
    // 376-byte RenderPipelineKey starting at byte 16:
    // colorTargetCount at key+20 (u32), target[0].format at key+24 (u8), target[0].writeMask at key+26 (u8)
    const keyBase = 16;
    v.setUint32(keyBase + 20, 1, true);
    const fmtCode = RS_TEXTURE_FORMAT_NAMES.indexOf(colorFormat);
    v.setUint8(keyBase + 24, fmtCode >= 0 ? fmtCode : 27);
    v.setUint8(keyBase + 26, 0x0f);
    // multisample count = 1 at keyBase + 164 (u32), mask = 0xffffffff at keyBase + 168 (u32)
    v.setUint32(keyBase + 164, 1, true);
    v.setUint32(keyBase + 168, 0xffffffff, true);
    return this._pushPacket(OPCODES.CREATE_RENDER_PIPELINE2, new Uint8Array(buf));
  }

  createComputePipeline2(id, csShaderId, layoutId = 0) {
    const buf = new ArrayBuffer(32);
    const v = new DataView(buf);
    v.setUint32(0, id, true);
    v.setUint32(4, csShaderId, true);
    v.setUint32(8, layoutId, true);
    return this._pushPacket(OPCODES.CREATE_COMPUTE_PIPELINE2, new Uint8Array(buf));
  }

  setPipeline(id) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, id, true);
    return this._pushPacket(OPCODES.SET_PIPELINE, new Uint8Array(buf));
  }

  setComputePipeline(id) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, id, true);
    return this._pushPacket(OPCODES.SET_COMPUTE_PIPELINE, new Uint8Array(buf));
  }

  draw(vertexCount, instanceCount = 1, firstVertex = 0, firstInstance = 0) {
    const buf = new ArrayBuffer(16);
    const v = new DataView(buf);
    v.setUint32(0, vertexCount, true);
    v.setUint32(4, instanceCount, true);
    v.setUint32(8, firstVertex, true);
    v.setUint32(12, firstInstance, true);
    return this._pushPacket(OPCODES.DRAW, new Uint8Array(buf));
  }

  dispatchCompute(x, y = 1, z = 1) {
    const buf = new ArrayBuffer(12);
    const v = new DataView(buf);
    v.setUint32(0, x, true);
    v.setUint32(4, y, true);
    v.setUint32(8, z, true);
    return this._pushPacket(OPCODES.DISPATCH_COMPUTE, new Uint8Array(buf));
  }

  destroyResource(id) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, id, true);
    return this._pushPacket(OPCODES.DESTROY_RESOURCE, new Uint8Array(buf));
  }

  present() {
    return this._pushPacket(OPCODES.PRESENT, new Uint8Array(0));
  }

  finish() {
    const total = this._chunks.reduce((sum, c) => sum + c.byteLength, 0);
    const out = new Uint8Array(total);
    let offset = 0;
    for (const c of this._chunks) {
      out.set(c, offset);
      offset += c.byteLength;
    }
    return out.buffer;
  }
}

function createMockWebGpu() {
  const calls = [];
  let nextId = 1;

  const device = {
    features: new Set(),
    createShaderModule(desc) {
      const mod = { kind: 'GPUShaderModule', id: nextId++, code: desc.code };
      calls.push({ op: 'createShaderModule', desc, mod });
      return mod;
    },
    createRenderPipeline(desc) {
      const pipe = {
        kind: 'GPURenderPipeline',
        id: nextId++,
        desc,
        getBindGroupLayout(index) {
          return { kind: 'GPUBindGroupLayout', pipelineId: pipe.id, index };
        },
      };
      calls.push({ op: 'createRenderPipeline', desc, pipe });
      return pipe;
    },
    createComputePipeline(desc) {
      const pipe = {
        kind: 'GPUComputePipeline',
        id: nextId++,
        desc,
        getBindGroupLayout(index) {
          return { kind: 'GPUBindGroupLayout', pipelineId: pipe.id, index };
        },
      };
      calls.push({ op: 'createComputePipeline', desc, pipe });
      return pipe;
    },
    createBindGroupLayout(desc) {
      return { kind: 'GPUBindGroupLayout', id: nextId++, desc };
    },
    createPipelineLayout(desc) {
      return { kind: 'GPUPipelineLayout', id: nextId++, desc };
    },
    createBindGroup(desc) {
      return { kind: 'GPUBindGroup', id: nextId++, desc };
    },
    createBuffer(desc) {
      return { kind: 'GPUBuffer', id: nextId++, size: desc.size, usage: desc.usage, destroy() {} };
    },
    createCommandEncoder() {
      const commands = [];
      return {
        beginRenderPass(desc) {
          commands.push({ op: 'beginRenderPass', desc });
          return {
            setPipeline(pipeline) {
              commands.push({ op: 'setPipeline', pipeline });
            },
            setBindGroup(index, bg) {
              commands.push({ op: 'setBindGroup', index, bg });
            },
            setVertexBuffer(slot, buf, offset) {
              commands.push({ op: 'setVertexBuffer', slot, buf, offset });
            },
            draw(vertexCount, instanceCount, firstVertex, firstInstance) {
              commands.push({ op: 'draw', vertexCount, instanceCount, firstVertex, firstInstance });
            },
            end() {
              commands.push({ op: 'endRenderPass' });
            },
          };
        },
        beginComputePass() {
          commands.push({ op: 'beginComputePass' });
          return {
            setPipeline(pipeline) {
              commands.push({ op: 'setComputePipeline', pipeline });
            },
            setBindGroup(index, bg) {
              commands.push({ op: 'setComputeBindGroup', index, bg });
            },
            dispatchWorkgroups(x, y, z) {
              commands.push({ op: 'dispatchWorkgroups', x, y, z });
            },
            end() {
              commands.push({ op: 'endComputePass' });
            },
          };
        },
        finish() {
          const cb = { kind: 'GPUCommandBuffer', commands };
          calls.push({ op: 'finishCommandBuffer', commands });
          return cb;
        },
      };
    },
    queue: {
      writeBuffer() {},
      writeTexture() {},
      submit(cbs) {
        calls.push({ op: 'submit', count: cbs.length, cbs });
      },
    },
  };

  const ctx = {
    configure() {},
    getCurrentTexture() {
      return {
        createView() {
          return { kind: 'GPUTextureView' };
        },
      };
    },
  };

  return { device, ctx, calls };
}

/**
 * Synthetic HTTP Range server backed by a compact WGPK pack buffer, optionally reporting a
 * large virtual Content-Length (e.g., 1.83 GiB) on HEAD and Content-Range so tests exercise
 * the >128 MiB auto-streaming path without allocating gigabytes of RAM.
 */
function createSyntheticRangePackServer(packBuffer, options = {}) {
  const packBytes = new Uint8Array(packBuffer);
  const reportedTotalBytes =
    options.reportedTotalBytes !== undefined ? options.reportedTotalBytes : packBytes.byteLength;
  const acceptRanges = options.acceptRanges !== undefined ? options.acceptRanges : 'bytes';
  const deferDataRanges = Boolean(options.deferDataRanges);
  const dataStartOffset = options.dataStartOffset || 0;
  const rejectFullGet = Boolean(options.rejectFullGet);

  const requests = [];
  const deferredRanges = [];

  const fetchImpl = async (url, init = {}) => {
    const method = (init && init.method ? String(init.method) : 'GET').toUpperCase();
    const headers = init && init.headers ? init.headers : {};
    const rangeHeader = headers.Range || headers.range || null;
    const req = { url, method, rangeHeader, start: null, end: null };
    requests.push(req);

    if (method === 'HEAD') {
      const respHeaders = new Map([['content-length', String(reportedTotalBytes)]]);
      if (acceptRanges !== null) {
        respHeaders.set('accept-ranges', acceptRanges);
      }
      return {
        ok: true,
        status: 200,
        headers: respHeaders,
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }

    if (rangeHeader) {
      const m = /^bytes=(\d+)-(\d+)$/.exec(rangeHeader);
      assert.ok(m, `Unexpected Range header: ${rangeHeader}`);
      const start = Number(m[1]);
      const requestedEnd = Number(m[2]);
      const clampedEnd = Math.min(requestedEnd, packBytes.byteLength - 1);
      req.start = start;
      req.end = clampedEnd;

      const isDataRange = dataStartOffset > 0 && start >= dataStartOffset;
      if (isDataRange && deferDataRanges) {
        await new Promise((resolve, reject) => {
          deferredRanges.push({ req, start, end: clampedEnd, resolve, reject });
        });
      }

      const slice = packBytes.slice(start, clampedEnd + 1);
      const totalForHeader = Math.max(reportedTotalBytes, clampedEnd + 1);
      return {
        ok: true,
        status: 206,
        headers: new Map([
          ['content-range', `bytes ${start}-${clampedEnd}/${totalForHeader}`],
          ['content-length', String(slice.byteLength)],
          ['accept-ranges', 'bytes'],
        ]),
        async arrayBuffer() {
          return slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength);
        },
      };
    }

    if (rejectFullGet) {
      throw new Error(`Unexpected full-buffer GET for ${url} (expected HTTP 206 Range streaming)`);
    }

    return {
      ok: true,
      status: 200,
      headers: new Map([['content-length', String(packBytes.byteLength)]]),
      async arrayBuffer() {
        return packBuffer.slice(0);
      },
    };
  };

  return {
    requests,
    deferredRanges,
    fetchImpl,
    resolveNextDeferred() {
      const next = deferredRanges.shift();
      assert.ok(next, 'Expected a deferred range fetch');
      next.resolve();
      return next;
    },
    resolveAllDeferred() {
      while (deferredRanges.length > 0) {
        deferredRanges.shift().resolve();
      }
    },
  };
}

const VS_WGSL = '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.2, 0.4, 0.0, 1.0); }\n';
const FS_WGSL = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.9, 0.3, 0.2, 1.0); }\n';
const CS_WGSL = '@compute @workgroup_size(8, 1, 1) fn cs_main() {}\n';
const VS_HASH = wgpkFormatHashHex(wgpkFnv1a64(VS_WGSL));
const FS_HASH = wgpkFormatHashHex(wgpkFnv1a64(FS_WGSL));
const CS_HASH = wgpkFormatHashHex(wgpkFnv1a64(CS_WGSL));

function buildSamplePack() {
  const buffer = buildWgslPackBuffer([
    { hash: VS_HASH, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: VS_WGSL },
    { hash: FS_HASH, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: FS_WGSL },
    { hash: CS_HASH, stage: WGPK_SHADER_STAGE.COMPUTE, wgsl: CS_WGSL },
  ]);
  const dataStart = 16 + 3 * 20;
  return { buffer, dataStart };
}

// ---------------------------------------------------------------------------
// 1. Threshold constant export & resolution precedence
// ---------------------------------------------------------------------------

test('W159 exports DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES (128 MiB) and resolves threshold options cleanly', () => {
  assert.equal(DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES, 128 * 1024 * 1024);
  assert.equal(DEFAULT_STREAM_SHADERS_THRESHOLD_BYTES, 64 * 1024 * 1024);

  // Default auto-threshold when omitted everywhere.
  assert.equal(resolveShaderPackStreamThresholdBytes(null, null), 128 * 1024 * 1024);
  assert.equal(resolveShaderPackStreamThresholdBytes({}, {}), 128 * 1024 * 1024);

  // Explicit 0 or streamShaders: false disables ranged streaming.
  assert.equal(resolveShaderPackStreamThresholdBytes({ streamThresholdBytes: 0 }, null), 0);
  assert.equal(resolveShaderPackStreamThresholdBytes({ streamShaders: false }, null), 0);
  assert.equal(
    resolveShaderPackStreamThresholdBytes(
      { streamThresholdBytes: 0 },
      { streamThresholdBytes: 128 * 1024 * 1024 }
    ),
    0
  );
  assert.equal(
    resolveShaderPackStreamThresholdBytes(
      { streamShaders: false },
      { streamThresholdBytes: 64 * 1024 * 1024 }
    ),
    0
  );
  assert.equal(
    resolveShaderPackStreamThresholdBytes({ streamShaders: false, streamThresholdBytes: 1024 }, null),
    0
  );

  // Explicit streamShaders: true uses 64 MiB; explicit streamThresholdBytes overrides it.
  assert.equal(resolveShaderPackStreamThresholdBytes({ streamShaders: true }, null), 64 * 1024 * 1024);
  assert.equal(
    resolveShaderPackStreamThresholdBytes({ streamShaders: true, streamThresholdBytes: 32 * 1024 * 1024 }, null),
    32 * 1024 * 1024
  );

  // Fallback & tertiary precedence.
  assert.equal(
    resolveShaderPackStreamThresholdBytes({}, { streamThresholdBytes: 0 }, { streamThresholdBytes: 128 }),
    0
  );
  assert.equal(
    resolveShaderPackStreamThresholdBytes({}, {}, { streamThresholdBytes: 0 }),
    0
  );
});

// ---------------------------------------------------------------------------
// 2. Auto-activation for >128 MiB shader packs by default (simulating 1.83 GiB pack)
// ---------------------------------------------------------------------------

test('W159 auto-enables HTTP 206 Range streaming by default for >128 MiB packs on both owned InlineWgslPackCache and WgslPackCache', async () => {
  const { buffer, dataStart } = buildSamplePack();
  const FULL_PACK_SIZE_1_83_GIB = 1_964_947_456; // 1.83 GiB (> 128 MiB)

  // 2a. Default WgpuCommandProcessor (owned InlineWgslPackCache -> auto-upgrades to WgslPackCache).
  const netOwned = createSyntheticRangePackServer(buffer, {
    reportedTotalBytes: FULL_PACK_SIZE_1_83_GIB,
    dataStartOffset: dataStart,
    rejectFullGet: true,
  });
  const mockOwned = createMockWebGpu();
  const procOwned = new WgpuCommandProcessor(mockOwned.device, mockOwned.ctx, {
    format: 'bgra8unorm',
  });

  // Before loading: default threshold is 128 MiB, ranged pack not yet active.
  assert.equal(procOwned.rangedPackActive, false);
  assert.equal(procOwned.rangedPackThresholdBytes, DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES);
  assert.equal(procOwned.asyncShaderPendingCount, 0);

  await procOwned.loadShaderPackUrl('/shaders/full-18504.wgsl.pack', {
    fetch: netOwned.fetchImpl,
    initialRangeBytes: dataStart,
  });

  // Zero full-buffer 200 GETs; HEAD + HTTP 206 Range index load only.
  assert.equal(netOwned.requests.filter((r) => r.method === 'GET' && r.rangeHeader === null).length, 0);
  assert.equal(netOwned.requests.filter((r) => r.method === 'HEAD').length, 1);
  assert.equal(netOwned.requests.filter((r) => r.rangeHeader !== null).length, 1);

  assert.equal(procOwned.rangedPackActive, true);
  assert.equal(procOwned.rangedPackThresholdBytes, DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES);
  assert.equal(procOwned.asyncShaderPendingCount, 0);

  const ownedStats = procOwned.getStats();
  assert.equal(ownedStats.rangedPackActive, true);
  assert.equal(ownedStats.rangedPackThresholdBytes, DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES);
  assert.equal(ownedStats.asyncShaderPendingCount, 0);

  const ownedSnap = buildGpuTelemetrySnapshot(procOwned, { timestamp: 1001 });
  assert.equal(ownedSnap.shaderPackStats.mode, 'ranged');
  assert.equal(ownedSnap.shaderPackStats.rangedPackActive, true);
  assert.equal(
    ownedSnap.shaderPackStats.rangedPackThresholdBytes,
    DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES
  );
  assert.equal(ownedSnap.shaderPackStats.asyncShaderPendingCount, 0);
  assert.equal(ownedSnap.shaderPackStats.shaderCount, 3);

  // 2b. Worker handler init with default options (no streamThresholdBytes specified) + gpu-stats reply.
  const netWorker = createSyntheticRangePackServer(buffer, {
    reportedTotalBytes: FULL_PACK_SIZE_1_83_GIB,
    dataStartOffset: dataStart,
    rejectFullGet: true,
  });
  const mockWorker = createMockWebGpu();
  const procWorker = new WgpuCommandProcessor(mockWorker.device, mockWorker.ctx, {
    format: 'bgra8unorm',
    wgslPackCache: new WgslPackCache(),
  });
  const posted = [];
  const handler = createWgpuWorkerHandler({
    processor: procWorker,
    fetch: netWorker.fetchImpl,
    postMessage: (m) => posted.push(m),
  });

  await handler.handleMessage({
    init: true,
    shaderPackUrl: '/shaders/full-18504.wgsl.pack',
    initialRangeBytes: dataStart,
  });

  assert.equal(netWorker.requests.filter((r) => r.method === 'GET' && r.rangeHeader === null).length, 0);
  assert.ok(netWorker.requests.some((r) => r.rangeHeader !== null));
  assert.equal(procWorker.rangedPackActive, true);

  await handler.handleMessage({ id: 'stats-1', type: 'request-gpu-stats' });
  const gpuStatsMsg = posted.find((m) => m && m.id === 'stats-1' && m.type === 'gpu-stats');
  assert.ok(gpuStatsMsg);
  assert.equal(gpuStatsMsg.rangedPackActive, true);
  assert.equal(gpuStatsMsg.rangedPackThresholdBytes, DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES);
  assert.equal(gpuStatsMsg.asyncShaderPendingCount, 0);

  procOwned.destroy();
  procWorker.destroy();
});

// ---------------------------------------------------------------------------
// 3. Explicit opt-out (`streamThresholdBytes: 0` or `streamShaders: false`)
// ---------------------------------------------------------------------------

test('W159 explicit streamThresholdBytes: 0 and streamShaders: false disable auto-streaming and force full-buffer GET', async () => {
  const { buffer } = buildSamplePack();
  const LARGE_REPORTED_SIZE = 512 * 1024 * 1024; // 512 MiB (> 128 MiB)

  // 3a. Explicit streamThresholdBytes: 0 on loadShaderPackUrl.
  const netZero = createSyntheticRangePackServer(buffer, {
    reportedTotalBytes: LARGE_REPORTED_SIZE,
  });
  const mockZero = createMockWebGpu();
  const procZero = new WgpuCommandProcessor(mockZero.device, mockZero.ctx, { format: 'bgra8unorm' });

  await procZero.loadShaderPackUrl('/shaders/full-optout-0.wgsl.pack', {
    fetch: netZero.fetchImpl,
    streamThresholdBytes: 0,
  });

  assert.equal(netZero.requests.length, 1);
  assert.equal(netZero.requests[0].method, 'GET');
  assert.equal(netZero.requests[0].rangeHeader, null);
  assert.equal(procZero.rangedPackActive, false);
  assert.equal(procZero.rangedPackThresholdBytes, 0);
  assert.equal(procZero.getStats().rangedPackActive, false);
  assert.equal(procZero.getStats().rangedPackThresholdBytes, 0);
  assert.equal(buildGpuTelemetrySnapshot(procZero).shaderPackStats.rangedPackActive, false);
  assert.equal(buildGpuTelemetrySnapshot(procZero).shaderPackStats.rangedPackThresholdBytes, 0);

  // Shaders resolve synchronously from full buffer with zero ranged fetches.
  procZero.executeCommandBuffer(
    new BinaryCommandBuilder().createShaderModule(1, SHADER_STAGE.VERTEX, VS_HASH).finish()
  );
  assert.equal(procZero.shaderModules.get(1).source, 'pack');
  assert.equal(procZero.getStats().rangedShaderFetches, 0);

  // 3b. Explicit streamShaders: false on processor constructor & worker handler message.
  const netFalse = createSyntheticRangePackServer(buffer, {
    reportedTotalBytes: LARGE_REPORTED_SIZE,
  });
  const mockFalse = createMockWebGpu();
  const procFalse = new WgpuCommandProcessor(mockFalse.device, mockFalse.ctx, {
    format: 'bgra8unorm',
    wgslPackCache: new WgslPackCache(),
    streamShaders: false,
  });
  const handlerFalse = createWgpuWorkerHandler({
    processor: procFalse,
    fetch: netFalse.fetchImpl,
  });

  await handlerFalse.handleMessage({
    init: true,
    shaderPackUrl: '/shaders/full-optout-false.wgsl.pack',
  });

  assert.equal(netFalse.requests.length, 1);
  assert.equal(netFalse.requests[0].method, 'GET');
  assert.equal(netFalse.requests[0].rangeHeader, null);
  assert.equal(procFalse.rangedPackActive, false);
  assert.equal(procFalse.rangedPackThresholdBytes, 0);
  assert.equal(procFalse.wgslPackCache.getStats().mode, 'buffer');

  procZero.destroy();
  procFalse.destroy();
});

// ---------------------------------------------------------------------------
// 4. Small-pack full-buffer fast path (< 128 MiB) & Accept-Ranges: none fallback
// ---------------------------------------------------------------------------

test('W159 preserves full-buffer loading for small (<128 MiB) packs and Accept-Ranges: none servers', async () => {
  const { buffer } = buildSamplePack();

  // 4a. Small pack (< 128 MiB) with default auto-threshold stays in buffer mode with zero Range requests.
  const netSmall = createSyntheticRangePackServer(buffer);
  const mockSmall = createMockWebGpu();
  const procSmall = new WgpuCommandProcessor(mockSmall.device, mockSmall.ctx, {
    format: 'bgra8unorm',
  });

  await procSmall.loadShaderPackUrl('/shaders/fallback.wgsl.pack', {
    fetch: netSmall.fetchImpl,
  });

  assert.equal(netSmall.requests.filter((r) => r.rangeHeader !== null).length, 0);
  assert.equal(procSmall.rangedPackActive, false);
  assert.equal(procSmall.rangedPackThresholdBytes, DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES);
  assert.equal(buildGpuTelemetrySnapshot(procSmall).shaderPackStats.mode, 'buffer');
  assert.equal(buildGpuTelemetrySnapshot(procSmall).shaderPackStats.rangedPackActive, false);

  procSmall.executeCommandBuffer(
    new BinaryCommandBuilder()
      .createShaderModule(1, SHADER_STAGE.VERTEX, VS_HASH)
      .createShaderModule(2, SHADER_STAGE.FRAGMENT, FS_HASH)
      .createRenderPipeline(10, 1, 2, 0)
      .beginFrame()
      .setPipeline(10)
      .draw(3, 1, 0, 0)
      .present()
      .finish()
  );
  assert.equal(procSmall.getStats().realPackDrawCount, 1);
  assert.equal(procSmall.getStats().rangedShaderFetches, 0);
  assert.equal(procSmall.getStats().skippedDrawCount, 0);

  // 4b. Large Content-Length but Accept-Ranges: none -> gracefully uses full-buffer GET.
  const netNoRange = createSyntheticRangePackServer(buffer, {
    reportedTotalBytes: 256 * 1024 * 1024,
    acceptRanges: 'none',
  });
  const mockNoRange = createMockWebGpu();
  const procNoRange = new WgpuCommandProcessor(mockNoRange.device, mockNoRange.ctx, {
    format: 'bgra8unorm',
  });

  await procNoRange.loadShaderPackUrl('/shaders/no-range.wgsl.pack', {
    fetch: netNoRange.fetchImpl,
  });

  assert.equal(netNoRange.requests.filter((r) => r.rangeHeader !== null).length, 0);
  assert.equal(procNoRange.rangedPackActive, false);
  assert.equal(procNoRange.wgslPackCache.getStats().shaderCount, 3);

  procSmall.destroy();
  procNoRange.destroy();
});

// ---------------------------------------------------------------------------
// 5. Concurrent async getWgslAsync resolution + pending pipeline creation (zero dropped draws)
// ---------------------------------------------------------------------------

test('W159 concurrent async getWgslAsync resolution upgrades pending v1/v2 render and compute pipelines seamlessly with zero dropped draws', async () => {
  const { buffer, dataStart } = buildSamplePack();
  const FULL_PACK_SIZE_1_83_GIB = 1_964_947_456;

  const net = createSyntheticRangePackServer(buffer, {
    reportedTotalBytes: FULL_PACK_SIZE_1_83_GIB,
    dataStartOffset: dataStart,
    deferDataRanges: true,
    rejectFullGet: true,
  });
  const mock = createMockWebGpu();
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
  });
  installRenderStatePackets(processor);

  // Auto-stream via default 128 MiB threshold (no explicit streamThresholdBytes).
  await processor.loadShaderPackUrl('/shaders/full-18504.wgsl.pack', {
    fetch: net.fetchImpl,
    initialRangeBytes: dataStart,
  });
  assert.equal(processor.rangedPackActive, true);

  // Frame 1: create VS, FS, CS from ranged hashes + v1 & v2 pipelines + draw + compute dispatch while fetches are in flight.
  const frame1 = new BinaryCommandBuilder()
    .beginFrame()
    .createShaderModule(1, SHADER_STAGE.VERTEX, VS_HASH)
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, FS_HASH)
    .createShaderModule(3, SHADER_STAGE.COMPUTE, CS_HASH)
    .createRenderPipeline(10, 1, 2, 0)
    .createComputePipeline(20, 3)
    .createRenderPipeline2(30, 1, 2, 0)
    .createComputePipeline2(40, 3, 0)
    .setPipeline(10)
    .draw(3, 1, 0, 0)
    .setPipeline(30)
    .draw(3, 1, 0, 0)
    .setComputePipeline(20)
    .dispatchCompute(1, 1, 1)
    .present()
    .finish();

  const frame1Res = processor.executeCommandBuffer(frame1);
  assert.deepEqual(frame1Res, { frameCount: 1, presents: 1 });

  // Zero dropped draws on Frame 1 while all 3 ranged shaders are still pending!
  const midStats = processor.getStats();
  assert.equal(midStats.drawCalls, 2);
  assert.equal(midStats.fallbackDrawCount, 2);
  assert.equal(midStats.realPackDrawCount, 0);
  assert.equal(midStats.skippedDrawCount, 0);
  assert.equal(midStats.validationErrorCount, 0);
  assert.equal(midStats.asyncShaderPendingCount, 3);
  assert.equal(midStats.rangedShaderFetches, 3);
  assert.equal(midStats.rangedShaderReady, 0);
  assert.equal(processor.asyncShaderPendingCount, 3);

  const midSnap = buildGpuTelemetrySnapshot(processor, { timestamp: 2001 });
  assert.equal(midSnap.shaderPackStats.rangedPackActive, true);
  assert.equal(midSnap.shaderPackStats.asyncShaderPendingCount, 3);

  // Pipelines 10 & 30 are currently fallback pipelines while fetches are in flight.
  const fallbackPipe10 = processor.pipelines.get(10);
  const fallbackPipe30 = processor.pipelines.get(30);
  const fallbackCompute20 = processor.computePipelines.get(20);
  const fallbackCompute40 = processor.computePipelines.get(40);
  assert.ok(fallbackPipe10 && fallbackPipe30 && fallbackCompute20 && fallbackCompute40);
  assert.equal(processor._pipelineMeta.get(10).isRealPack, false);
  assert.equal(processor._pipelineMeta.get(30).isRealPack, false);

  // Resolve VS (1st of 3) -> render pipelines wait until FS also settles (no wasteful intermediate compile).
  assert.equal(net.deferredRanges.length, 3);
  net.resolveNextDeferred();
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(processor.asyncShaderPendingCount, 2);
  assert.equal(processor.pipelines.get(10), fallbackPipe10, 'Wait for both VS+FS before rebuilding pipeline 10');

  // Resolve FS (2nd of 3) and CS (3rd of 3) concurrently -> all pending pipelines upgrade automatically!
  net.resolveAllDeferred();
  await processor.flushPendingRangedShaders();

  const afterStats = processor.getStats();
  assert.equal(afterStats.asyncShaderPendingCount, 0);
  assert.equal(afterStats.rangedShaderFetches, 3);
  assert.equal(afterStats.rangedShaderReady, 3);
  assert.equal(afterStats.rangedShaderErrors, 0);
  assert.equal(afterStats.validationErrorCount, 0);

  // Verify v1 & v2 render and compute pipelines were upgraded to the real pack shader modules!
  const realPipe10 = processor.pipelines.get(10);
  const realPipe30 = processor.pipelines.get(30);
  const realCompute20 = processor.computePipelines.get(20);
  const realCompute40 = processor.computePipelines.get(40);

  assert.notEqual(realPipe10, fallbackPipe10);
  assert.notEqual(realPipe30, fallbackPipe30);
  assert.notEqual(realCompute20, fallbackCompute20);
  assert.notEqual(realCompute40, fallbackCompute40);

  assert.equal(realPipe10.desc.vertex.module.code, VS_WGSL);
  assert.equal(realPipe10.desc.fragment.module.code, FS_WGSL);
  assert.equal(realPipe30.desc.vertex.module.code, VS_WGSL);
  assert.equal(realPipe30.desc.fragment.module.code, FS_WGSL);
  assert.equal(realCompute20.desc.compute.module.code, CS_WGSL);
  assert.equal(realCompute40.desc.compute.module.code, CS_WGSL);

  assert.equal(processor._pipelineMeta.get(10).isRealPack, true);
  assert.equal(processor._pipelineMeta.get(10).isFallback, false);
  assert.equal(processor._pipelineMeta.get(30).isRealPack, true);
  assert.equal(processor._pipelineMeta.get(30).isFallback, false);

  // Frame 2: draw with the same pipeline IDs -> now draws with REAL pack shaders, zero skipped draws!
  const frame2 = new BinaryCommandBuilder()
    .beginFrame()
    .setPipeline(10)
    .draw(3, 1, 0, 0)
    .setPipeline(30)
    .draw(3, 1, 0, 0)
    .present()
    .finish();
  processor.executeCommandBuffer(frame2);

  const finalStats = processor.getStats();
  assert.equal(finalStats.drawCalls, 4);
  assert.equal(finalStats.realPackDrawCount, 2);
  assert.equal(finalStats.fallbackDrawCount, 2);
  assert.equal(finalStats.skippedDrawCount, 0);
  assert.equal(finalStats.validationErrorCount, 0);

  processor.destroy();
});

test('W159 deferred pipeline creation with enableShaderFallback: false creates pipeline once async ranged shaders settle without errors', async () => {
  const { buffer, dataStart } = buildSamplePack();
  const net = createSyntheticRangePackServer(buffer, {
    reportedTotalBytes: 256 * 1024 * 1024,
    dataStartOffset: dataStart,
    deferDataRanges: true,
    rejectFullGet: true,
  });
  const mock = createMockWebGpu();
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
    enableShaderFallback: false,
  });

  await processor.loadShaderPackUrl('/shaders/full-nofb.wgsl.pack', {
    fetch: net.fetchImpl,
    initialRangeBytes: dataStart,
  });
  assert.equal(processor.rangedPackActive, true);

  // Issue shader + pipeline creation + draw while fetches are still in flight.
  processor.executeCommandBuffer(
    new BinaryCommandBuilder()
      .beginFrame()
      .createShaderModule(1, SHADER_STAGE.VERTEX, VS_HASH)
      .createShaderModule(2, SHADER_STAGE.FRAGMENT, FS_HASH)
      .createRenderPipeline(10, 1, 2, 0)
      .setPipeline(10)
      .draw(3, 1, 0, 0)
      .present()
      .finish()
  );

  // Zero validation errors while pipeline is pending ranged shaders.
  assert.equal(processor.getStats().validationErrorCount, 0);
  assert.equal(processor.isAsyncPipelinePending(10), true);
  assert.equal(processor.pipelines.has(10), false);
  assert.equal(processor.asyncShaderPendingCount, 2);

  // Resolve both ranged fetches -> pipeline 10 is created automatically!
  net.resolveAllDeferred();
  await processor.flushPendingRangedShaders();

  assert.equal(processor.asyncShaderPendingCount, 0);
  assert.equal(processor.isAsyncPipelinePending(10), false);
  assert.equal(processor.pipelines.has(10), true);
  assert.equal(processor.pipelines.get(10).desc.vertex.module.code, VS_WGSL);
  assert.equal(processor.pipelines.get(10).desc.fragment.module.code, FS_WGSL);
  assert.equal(processor.getStats().validationErrorCount, 0);

  // Frame 2 draws cleanly with the real pipeline!
  processor.executeCommandBuffer(
    new BinaryCommandBuilder()
      .beginFrame()
      .setPipeline(10)
      .draw(3, 1, 0, 0)
      .present()
      .finish()
  );
  assert.equal(processor.getStats().realPackDrawCount, 1);
  assert.equal(processor.getStats().skippedDrawCount, 0);
  assert.equal(processor.getStats().validationErrorCount, 0);

  processor.destroy();
});

test('W159 destroying a pending pipeline before ranged shaders resolve prevents stale pipeline resurrection', async () => {
  const { buffer, dataStart } = buildSamplePack();
  const net = createSyntheticRangePackServer(buffer, {
    reportedTotalBytes: 256 * 1024 * 1024,
    dataStartOffset: dataStart,
    deferDataRanges: true,
    rejectFullGet: true,
  });
  const mock = createMockWebGpu();
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
  });

  await processor.loadShaderPackUrl('/shaders/full-destroy.wgsl.pack', {
    fetch: net.fetchImpl,
    initialRangeBytes: dataStart,
  });

  processor.executeCommandBuffer(
    new BinaryCommandBuilder()
      .createShaderModule(1, SHADER_STAGE.VERTEX, VS_HASH)
      .createShaderModule(2, SHADER_STAGE.FRAGMENT, FS_HASH)
      .createRenderPipeline(10, 1, 2, 0)
      .destroyResource(10)
      .finish()
  );
  assert.equal(processor.pipelines.has(10), false);

  net.resolveAllDeferred();
  await processor.flushPendingRangedShaders();

  // Shader modules upgraded, but destroyed pipeline 10 was NOT recreated.
  assert.equal(processor.shaderModules.get(1).source, 'pack');
  assert.equal(processor.shaderModules.get(2).source, 'pack');
  assert.equal(processor.pipelines.has(10), false);
  assert.equal(processor.getStats().asyncShaderPendingCount, 0);

  processor.destroy();
});
