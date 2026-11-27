import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import {
  DEFAULT_MAX_CONCURRENT_RANGE_FETCHES,
  WgslPackCache,
  buildWgslPackBuffer,
  loadIndexFromUrl,
  fnv1a64 as wgpkFnv1a64,
  formatHashHex as wgpkFormatHashHex,
  SHADER_STAGE as WGPK_SHADER_STAGE,
} from '../web/workers/wgsl_pack_loader.js';

const require = createRequire(import.meta.url);
const {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  buildGpuTelemetrySnapshot,
  compactWgslForMetadata,
  DEFAULT_MAX_WGSL_COMPILE_BYTES,
  DEFAULT_MAX_CONCURRENT_SHADER_COMPILES,
  DEFAULT_SHADER_COMPILE_SLICE_BUDGET_MS,
  DEFAULT_MAX_PENDING_RANGED_PIPELINES,
  OPCODES,
  SHADER_STAGE,
  BUFFER_USAGE,
} = require('../web/workers/wgpu_worker.js');
const { GpuRingBuffer } = require('../web/workers/gpu_ring.js');
const { installRenderStatePackets } = require('../web/workers/render_state_packets.js');

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

  createBuffer(id, size, usage = BUFFER_USAGE.VERTEX | BUFFER_USAGE.COPY_DST) {
    const buf = new ArrayBuffer(16);
    const v = new DataView(buf);
    v.setUint32(0, id, true);
    v.setUint32(4, usage, true);
    v.setBigUint64(8, BigInt(size), true);
    return this._pushPacket(OPCODES.CREATE_BUFFER, new Uint8Array(buf));
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

  createRenderPipeline2(id, vsShaderId, fsShaderId, layoutId = 0) {
    const buf = new ArrayBuffer(16 + 376);
    const v = new DataView(buf);
    v.setUint32(0, id, true);
    v.setUint32(4, vsShaderId, true);
    v.setUint32(8, fsShaderId, true);
    v.setUint32(12, layoutId, true);
    const keyBase = 16;
    v.setUint32(keyBase + 20, 1, true);
    v.setUint8(keyBase + 24, 27); // bgra8unorm
    v.setUint8(keyBase + 26, 0x0f);
    v.setUint32(keyBase + 164, 1, true);
    v.setUint32(keyBase + 168, 0xffffffff, true);
    return this._pushPacket(OPCODES.CREATE_RENDER_PIPELINE2, new Uint8Array(buf));
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

function createMockWebGpu(options = {}) {
  let nextId = 1;
  const shaderModulesCreated = [];
  const renderPipelinesCreated = [];
  const computePipelinesCreated = [];
  const submittedCommandBuffers = [];
  const compilationInfoResolvers = [];
  let currentTextureCalls = 0;
  let computeDispatches = 0;

  const device = {
    features: new Set(),
    createShaderModule(desc) {
      if (typeof options.onCreateShaderModule === 'function') {
        options.onCreateShaderModule(desc);
      }
      const mod = {
        kind: 'GPUShaderModule',
        id: nextId++,
        code: desc.code,
      };
      if (options.deferCompilationInfo) {
        mod.getCompilationInfo = () =>
          new Promise((resolve) => {
            compilationInfoResolvers.push(() => resolve({ messages: [] }));
          });
      }
      shaderModulesCreated.push(mod);
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
      renderPipelinesCreated.push(pipe);
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
      computePipelinesCreated.push(pipe);
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
      const ops = [];
      return {
        beginRenderPass(desc) {
          ops.push({ op: 'beginRenderPass', desc });
          return {
            setPipeline() {},
            setBindGroup() {},
            setVertexBuffer() {},
            setIndexBuffer() {},
            draw() {
              ops.push({ op: 'draw' });
            },
            drawIndexed() {},
            end() {
              ops.push({ op: 'endRenderPass' });
            },
          };
        },
        beginComputePass() {
          ops.push({ op: 'beginComputePass' });
          return {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups(x, y, z) {
              computeDispatches += 1;
              ops.push({ op: 'dispatchWorkgroups', x, y, z });
            },
            end() {
              ops.push({ op: 'endComputePass' });
            },
          };
        },
        finish() {
          return { kind: 'GPUCommandBuffer', ops };
        },
      };
    },
    queue: {
      writeBuffer() {},
      writeTexture() {},
      submit(cbs) {
        submittedCommandBuffers.push(...cbs);
      },
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
  };

  const ctx = {
    configure() {},
    getCurrentTexture() {
      currentTextureCalls += 1;
      return {
        width: 1280,
        height: 720,
        createView() {
          return { kind: 'GPUTextureView' };
        },
      };
    },
  };

  return {
    device,
    ctx,
    shaderModulesCreated,
    renderPipelinesCreated,
    computePipelinesCreated,
    submittedCommandBuffers,
    compilationInfoResolvers,
    get currentTextureCalls() {
      return currentTextureCalls;
    },
    get computeDispatches() {
      return computeDispatches;
    },
  };
}

function createSyntheticRangePackServer(packBuffer, options = {}) {
  const packBytes = new Uint8Array(packBuffer);
  const reportedTotalBytes =
    options.reportedTotalBytes !== undefined ? options.reportedTotalBytes : 256 * 1024 * 1024;
  const dataStartOffset = options.dataStartOffset || 0;
  const deferDataRanges = Boolean(options.deferDataRanges);
  const deferredRanges = [];
  let activeDataFetches = 0;
  let peakDataFetches = 0;
  let dataFetchCount = 0;

  const fetchImpl = async (url, init = {}) => {
    const method = (init && init.method ? String(init.method) : 'GET').toUpperCase();
    const headers = init && init.headers ? init.headers : {};
    const rangeHeader = headers.Range || headers.range || null;

    if (method === 'HEAD') {
      return {
        ok: true,
        status: 200,
        headers: new Map([
          ['content-length', String(reportedTotalBytes)],
          ['accept-ranges', 'bytes'],
        ]),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }

    if (rangeHeader) {
      const m = /^bytes=(\d+)-(\d+)$/.exec(rangeHeader);
      assert.ok(m, `Unexpected Range header: ${rangeHeader}`);
      const start = Number(m[1]);
      const clampedEnd = Math.min(Number(m[2]), packBytes.byteLength - 1);
      const isDataRange = dataStartOffset > 0 && start >= dataStartOffset;
      if (isDataRange) {
        dataFetchCount += 1;
        activeDataFetches += 1;
        if (activeDataFetches > peakDataFetches) {
          peakDataFetches = activeDataFetches;
        }
      }
      try {
        if (isDataRange && deferDataRanges) {
          await new Promise((resolve, reject) => {
            deferredRanges.push({ start, end: clampedEnd, resolve, reject });
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
      } finally {
        if (isDataRange) {
          activeDataFetches -= 1;
        }
      }
    }

    throw new Error(`Unexpected full GET for ${url}`);
  };

  return {
    fetchImpl,
    deferredRanges,
    get peakDataFetches() {
      return peakDataFetches;
    },
    get dataFetchCount() {
      return dataFetchCount;
    },
    resolveAllDeferred() {
      while (deferredRanges.length > 0) {
        deferredRanges.shift().resolve();
      }
    },
  };
}

const SMALL_VS_WGSL = '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }\n';
const SMALL_FS_WGSL = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.5, 0.25, 1.0); }\n';
const SMALL_CS_WGSL = '@compute @workgroup_size(1, 1, 1) fn cs_main() {}\n';

// ---------------------------------------------------------------------------
// 1. Hard WGSL single-shader byte-size cap (maxWgslCompileBytes)
// ---------------------------------------------------------------------------

test('Wave-4 #1: oversized WGSL shaders never reach device.createShaderModule, route to safe fallback, and surface oversizedWgslFallbacks telemetry', async () => {
  assert.equal(DEFAULT_MAX_WGSL_COMPILE_BYTES, 1024 * 1024);

  const oversizedCode = `@fragment fn fs_main() -> @location(0) vec4<f32> {\n// ${'x'.repeat(2048)}\nreturn vec4<f32>(1.0);\n}`;
  assert.ok(oversizedCode.length > 1024);

  // 1a. With fallback router enabled (default): oversized shader is quarantined and replaced with safe small fallback.
  const mockWithFb = createMockWebGpu({
    onCreateShaderModule(desc) {
      assert.ok(
        desc.code.length <= 1024,
        `Oversized WGSL (${desc.code.length}B) must NEVER be passed to device.createShaderModule()`
      );
    },
  });
  const procWithFb = new WgpuCommandProcessor(mockWithFb.device, mockWithFb.ctx, {
    format: 'bgra8unorm',
    maxWgslCompileBytes: 1024,
  });

  procWithFb.executeCommandBuffer(
    new BinaryCommandBuilder()
      .createShaderModule(1, SHADER_STAGE.VERTEX, SMALL_VS_WGSL)
      .createShaderModule(2, SHADER_STAGE.FRAGMENT, oversizedCode)
      .createRenderPipeline(10, 1, 2, 0)
      .beginFrame()
      .setPipeline(10)
      .draw(3, 1, 0, 0)
      .present()
      .finish()
  );

  assert.equal(procWithFb.validationErrorCount, 0);
  assert.equal(procWithFb.getStats().oversizedWgslFallbacks, 1);
  assert.equal(procWithFb.getStats().maxWgslCompileBytes, 1024);
  assert.equal(procWithFb.shaderModules.get(2).source, 'builtin');
  assert.equal(procWithFb.pipelines.has(10), true);

  const snap = buildGpuTelemetrySnapshot(procWithFb, { timestamp: 1234 });
  assert.equal(snap.oversizedWgslFallbacks, 1);

  const posted = [];
  const handler = createWgpuWorkerHandler({
    processor: procWithFb,
    postMessage: (m) => posted.push(m),
  });
  await handler.handleMessage({ id: 'wave4-stats', type: 'request-gpu-stats' });
  const statsReply = posted.find((m) => m && m.id === 'wave4-stats' && m.type === 'gpu-stats');
  assert.equal(statsReply.oversizedWgslFallbacks, 1);

  // 1b. With enableShaderFallback: false: returns null cleanly without throwing or recording GPU validation error.
  const mockNoFb = createMockWebGpu({
    onCreateShaderModule(desc) {
      assert.ok(desc.code.length <= 1024, 'Oversized WGSL must never reach device.createShaderModule');
    },
  });
  const procNoFb = new WgpuCommandProcessor(mockNoFb.device, mockNoFb.ctx, {
    format: 'bgra8unorm',
    enableShaderFallback: false,
    maxWgslCompileBytes: 1024,
  });

  procNoFb.executeCommandBuffer(
    new BinaryCommandBuilder()
      .createShaderModule(2, SHADER_STAGE.FRAGMENT, oversizedCode)
      .finish()
  );

  assert.equal(mockNoFb.shaderModulesCreated.length, 0);
  assert.equal(procNoFb.validationErrorCount, 0);
  assert.equal(procNoFb.getStats().oversizedWgslFallbacks, 1);
  assert.equal(procNoFb.ensureShaderModule(2), null);
  assert.equal(procNoFb.getStats().oversizedWgslFallbacks, 1, 'ensureShaderModule must not double-count');

  // 1c. Oversized ranged shader-pack payload is also blocked before createShaderModule and quarantined.
  const hugeHash = wgpkFormatHashHex(wgpkFnv1a64(oversizedCode));
  const packBuf = buildWgslPackBuffer([
    { hash: hugeHash, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: oversizedCode },
  ]);
  const dataStart = 16 + 20;
  const net = createSyntheticRangePackServer(packBuf, { dataStartOffset: dataStart });
  const mockRanged = createMockWebGpu({
    onCreateShaderModule(desc) {
      assert.ok(desc.code.length <= 1024, 'Oversized ranged WGSL must never reach device.createShaderModule');
    },
  });
  const procRanged = new WgpuCommandProcessor(mockRanged.device, mockRanged.ctx, {
    format: 'bgra8unorm',
    maxWgslCompileBytes: 1024,
  });
  await procRanged.loadShaderPackUrl('/shaders/oversized.wgsl.pack', {
    fetch: net.fetchImpl,
    initialRangeBytes: dataStart,
  });

  procRanged.executeCommandBuffer(
    new BinaryCommandBuilder()
      .createShaderModule(1, SHADER_STAGE.VERTEX, SMALL_VS_WGSL)
      .createShaderModule(2, SHADER_STAGE.FRAGMENT, hugeHash)
      .createRenderPipeline(10, 1, 2, 0)
      .finish()
  );
  await procRanged.flushPendingRangedShaders();

  assert.equal(procRanged.validationErrorCount, 0);
  assert.equal(procRanged.getStats().oversizedWgslFallbacks, 1);
  assert.equal(buildGpuTelemetrySnapshot(procRanged).shaderPackStats.oversizedWgslFallbacks, 1);
  assert.equal(procRanged.pipelines.has(10), true, 'Deferred pipeline builds cleanly with safe fallback');

  // Re-issuing the same oversized hash skips ranged fetch due to quarantine.
  const fetchesBefore = net.dataFetchCount;
  procRanged.executeCommandBuffer(
    new BinaryCommandBuilder()
      .createShaderModule(3, SHADER_STAGE.FRAGMENT, hugeHash)
      .finish()
  );
  await procRanged.flushPendingRangedShaders();
  assert.equal(net.dataFetchCount, fetchesBefore, 'Quarantined oversized hash must not re-fetch');

  procWithFb.destroy();
  procNoFb.destroy();
  procRanged.destroy();
});

// ---------------------------------------------------------------------------
// 2. O(N) compactWgslForMetadata & TextEncoder elimination
// ---------------------------------------------------------------------------

test('Wave-4 #2: compactWgslForMetadata runs in linear O(N) time on 512 KB WGSL and _compileShaderModuleEntry avoids TextEncoder allocations', () => {
  const bodySpam = '  let v = f32(1.0) + f32(2.0); // inline comment\n'.repeat(11000);
  const bigWgsl = [
    '/* outer /* nested */ comment */',
    '@group(0) @binding(0) var<uniform> used_cbv: vec4<f32>;',
    '@group(0) @binding(1) var<uniform> unused_cbv: vec4<f32>;',
    '@group(0) @binding(2) var used_tex: texture_2d<f32>;',
    '@fragment fn fs_main() -> @location(0) vec4<f32> {',
    bodySpam,
    '  let _t = textureLoad(used_tex, vec2<i32>(0, 0), 0);',
    '  return used_cbv + _t;',
    '}',
  ].join('\n');
  assert.ok(bigWgsl.length > 480 * 1024);

  const t0 = performance.now();
  const compacted = compactWgslForMetadata(bigWgsl);
  const elapsedMs = performance.now() - t0;

  assert.ok(compacted.length < 512, `Expected compact stub < 512B, got ${compacted.length}B`);
  assert.ok(compacted.includes('used_cbv'));
  assert.ok(compacted.includes('used_tex'));
  assert.equal(compacted.includes('unused_cbv'), false);
  assert.ok(compacted.includes('@fragment fn fs_main()'));
  assert.ok(elapsedMs < 250, `Expected O(N) compaction < 250ms, took ${elapsedMs.toFixed(1)}ms`);

  // Verify _compileShaderModuleEntry registers byteSize using code.length without constructing TextEncoder.
  const registered = [];
  const mock = createMockWebGpu();
  const proc = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
    resourceRegistry: {
      register(kind, id, resource, meta) {
        registered.push({ kind, id, meta });
      },
    },
  });

  const OrigTextEncoder = globalThis.TextEncoder;
  let encoderCtorCalls = 0;
  try {
    globalThis.TextEncoder = function GuardTextEncoder() {
      encoderCtorCalls += 1;
      return new OrigTextEncoder();
    };
    const packWgsl = '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }';
    const entry = {
      module: null,
      stage: SHADER_STAGE.VERTEX,
      code: packWgsl,
      codeLen: packWgsl.length,
      isInlineCode: false,
      hashHex: '0x1234',
      entryPoint: 'vs_main',
      source: 'pack',
      requestedHash: '0x1234',
      resolvedHash: '0x1234',
      fallbackProfile: null,
    };
    const compiled = proc._compileShaderModuleEntry(7, entry, false);
    assert.ok(compiled && compiled.module);
    assert.equal(encoderCtorCalls, 0, '_compileShaderModuleEntry must not allocate TextEncoder');
    assert.equal(registered.length, 1);
    assert.equal(registered[0].meta.byteSize, packWgsl.length);
  } finally {
    globalThis.TextEncoder = OrigTextEncoder;
    proc.destroy();
  }
});

// ---------------------------------------------------------------------------
// 3. Bounded/time-sliced createShaderModule concurrency + deferred fallback PSO + 4096 pending cap
// ---------------------------------------------------------------------------

test('Wave-4 #3a: ranged createShaderModule compilation caps concurrency at maxConcurrentShaderCompiles=2 and yields on >=8ms slices', async () => {
  assert.equal(DEFAULT_MAX_CONCURRENT_SHADER_COMPILES, 2);
  assert.equal(DEFAULT_SHADER_COMPILE_SLICE_BUDGET_MS, 8);
  assert.equal(DEFAULT_MAX_PENDING_RANGED_PIPELINES, 4096);

  const count = 8;
  const entries = [];
  const hashes = [];
  for (let i = 0; i < count; i += 1) {
    const wgsl = `@compute @workgroup_size(${i + 1}, 1, 1) fn cs_main() {}\n`;
    const hash = wgpkFormatHashHex(wgpkFnv1a64(wgsl));
    hashes.push(hash);
    entries.push({ hash, stage: WGPK_SHADER_STAGE.COMPUTE, wgsl });
  }
  const packBuf = buildWgslPackBuffer(entries);
  const dataStart = 16 + count * 20;

  // Part A: deferred getCompilationInfo() verifies hard concurrency cap = 2.
  const netA = createSyntheticRangePackServer(packBuf, { dataStartOffset: dataStart });
  const mockA = createMockWebGpu({ deferCompilationInfo: true });
  const procA = new WgpuCommandProcessor(mockA.device, mockA.ctx, {
    format: 'bgra8unorm',
  });
  await procA.loadShaderPackUrl('/shaders/conc.wgsl.pack', {
    fetch: netA.fetchImpl,
    initialRangeBytes: dataStart,
  });

  const builderA = new BinaryCommandBuilder();
  for (let i = 0; i < count; i += 1) {
    builderA.createShaderModule(i + 1, SHADER_STAGE.COMPUTE, hashes[i]);
  }
  procA.executeCommandBuffer(builderA.finish());

  // Wait for HTTP fetches to complete and enqueue compile jobs.
  await new Promise((r) => setTimeout(r, 15));
  assert.equal(mockA.shaderModulesCreated.length, 2, 'At most 2 concurrent createShaderModule calls active');
  assert.equal(procA.getStats().activeRangedShaderCompiles, 2);
  assert.equal(procA.getStats().peakConcurrentRangedShaderCompiles, 2);

  // Drain compilation info resolvers step-by-step and verify concurrency never exceeds 2.
  while (mockA.compilationInfoResolvers.length > 0 || procA.asyncShaderPendingCount > 0) {
    assert.ok(
      procA.getStats().activeRangedShaderCompiles <= 2,
      `Active compiles exceeded cap: ${procA.getStats().activeRangedShaderCompiles}`
    );
    const next = mockA.compilationInfoResolvers.shift();
    if (next) {
      next();
    }
    await new Promise((r) => setTimeout(r, 2));
  }
  await procA.flushPendingRangedShaders();

  assert.equal(mockA.shaderModulesCreated.length, count);
  assert.equal(procA.getStats().peakConcurrentRangedShaderCompiles, 2);
  assert.equal(procA.getStats().activeRangedShaderCompiles, 0);
  assert.equal(procA.getStats().rangedShaderReady, count);
  procA.destroy();

  // Part B: time-sliced macrotask yielding when batch elapsed >= 8 ms.
  let fakeNow = 100;
  const netB = createSyntheticRangePackServer(packBuf, { dataStartOffset: dataStart });
  const mockB = createMockWebGpu({
    onCreateShaderModule() {
      // Simulate each shader taking 10ms (>= 8ms budget) to compile.
      fakeNow += 10;
    },
  });
  const procB = new WgpuCommandProcessor(mockB.device, mockB.ctx, {
    format: 'bgra8unorm',
    now: () => fakeNow,
  });
  await procB.loadShaderPackUrl('/shaders/yield.wgsl.pack', {
    fetch: netB.fetchImpl,
    initialRangeBytes: dataStart,
  });

  const builderB = new BinaryCommandBuilder();
  for (let i = 0; i < count; i += 1) {
    builderB.createShaderModule(i + 1, SHADER_STAGE.COMPUTE, hashes[i]);
  }
  procB.executeCommandBuffer(builderB.finish());
  await procB.flushPendingRangedShaders();

  assert.equal(mockB.shaderModulesCreated.length, count);
  assert.ok(
    procB.getStats().rangedShaderCompileYields >= 1,
    `Expected >= 1 macrotask yields when compiles exceed 8ms budget, got ${procB.getStats().rangedShaderCompileYields}`
  );
  procB.destroy();
});

test('Wave-4 #3b: ranged index match defers throwaway builtin fallback PSO compilation (1 pipeline/PSO, not 2) and _pendingRangedPipelines cap is 4096', async () => {
  const vsHash = wgpkFormatHashHex(wgpkFnv1a64(SMALL_VS_WGSL));
  const fsHash = wgpkFormatHashHex(wgpkFnv1a64(SMALL_FS_WGSL));
  const csHash = wgpkFormatHashHex(wgpkFnv1a64(SMALL_CS_WGSL));
  const packBuf = buildWgslPackBuffer([
    { hash: vsHash, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: SMALL_VS_WGSL },
    { hash: fsHash, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: SMALL_FS_WGSL },
    { hash: csHash, stage: WGPK_SHADER_STAGE.COMPUTE, wgsl: SMALL_CS_WGSL },
  ]);
  const dataStart = 16 + 3 * 20;

  const net = createSyntheticRangePackServer(packBuf, {
    dataStartOffset: dataStart,
    deferDataRanges: true,
  });
  const mock = createMockWebGpu();
  const proc = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
    // Default options: enableShaderFallback is true, deferRangedFallbackPipelines is true!
  });
  installRenderStatePackets(proc);

  await proc.loadShaderPackUrl('/shaders/defer-pso.wgsl.pack', {
    fetch: net.fetchImpl,
    initialRangeBytes: dataStart,
  });

  // Issue VS, FS, CS + v1 render PSO + v2 render PSO + v1 compute PSO + draws while ranged fetches are in flight.
  proc.executeCommandBuffer(
    new BinaryCommandBuilder()
      .beginFrame()
      .createShaderModule(1, SHADER_STAGE.VERTEX, vsHash)
      .createShaderModule(2, SHADER_STAGE.FRAGMENT, fsHash)
      .createShaderModule(3, SHADER_STAGE.COMPUTE, csHash)
      .createRenderPipeline(10, 1, 2, 0)
      .createRenderPipeline2(20, 1, 2, 0)
      .createComputePipeline(30, 3)
      .setPipeline(10)
      .draw(3, 1, 0, 0)
      .setPipeline(20)
      .draw(3, 1, 0, 0)
      .setComputePipeline(30)
      .dispatchCompute(1, 1, 1)
      .present()
      .finish()
  );

  // Zero throwaway builtin shaders or pipelines compiled while ranged fetches are in flight!
  assert.equal(mock.shaderModulesCreated.length, 0);
  assert.equal(mock.renderPipelinesCreated.length, 0);
  assert.equal(mock.computePipelinesCreated.length, 0);
  assert.equal(proc.validationErrorCount, 0);
  assert.equal(proc.isAsyncPipelinePending(10), true);
  assert.equal(proc.isAsyncPipelinePending(20), true);
  assert.equal(proc.isAsyncPipelinePending(30), true);

  // Also test _pendingRangedPipelines cap raised from 2048 to 4096: enqueue 2500 pending pipelines!
  const manyBuilder = new BinaryCommandBuilder();
  for (let pid = 1000; pid < 3500; pid += 1) {
    manyBuilder.createRenderPipeline(pid, 1, 2, 0);
  }
  proc.executeCommandBuffer(manyBuilder.finish());
  assert.equal(
    proc.getStats().pendingRangedPipelinesCount,
    2503,
    'Pending ranged pipelines must exceed old 2048 cap up to 4096'
  );
  assert.equal(proc.isAsyncPipelinePending(10), true, 'Earliest pending pipeline not evicted at 2503 (< 4096)');

  // Now push past 4096 to verify the new 4096 bound evicts oldest cleanly.
  const overflowBuilder = new BinaryCommandBuilder();
  for (let pid = 3500; pid < 5200; pid += 1) {
    overflowBuilder.createRenderPipeline(pid, 1, 2, 0);
  }
  proc.executeCommandBuffer(overflowBuilder.finish());
  assert.equal(proc.getStats().pendingRangedPipelinesCount, 4096);

  // Clean up bulk test entries and verify the 3 core PSOs compile exactly once on settle.
  proc._pendingRangedPipelines.clear();
  proc.executeCommandBuffer(
    new BinaryCommandBuilder()
      .createRenderPipeline(10, 1, 2, 0)
      .createRenderPipeline2(20, 1, 2, 0)
      .createComputePipeline(30, 3)
      .finish()
  );
  net.resolveAllDeferred();
  await proc.flushPendingRangedShaders();

  assert.equal(mock.shaderModulesCreated.length, 3);
  assert.equal(mock.renderPipelinesCreated.length, 2, 'Exactly 1 real render pipeline per PSO (no throwaway fallback)');
  assert.equal(mock.computePipelinesCreated.length, 1, 'Exactly 1 real compute pipeline per PSO');
  assert.equal(proc.pipelines.get(10).desc.vertex.module.code, SMALL_VS_WGSL);
  assert.equal(proc.pipelines.get(20).desc.fragment.module.code, SMALL_FS_WGSL);
  assert.equal(proc.computePipelines.get(30).desc.compute.module.code, SMALL_CS_WGSL);
  assert.equal(proc.validationErrorCount, 0);

  proc.destroy();
});

// ---------------------------------------------------------------------------
// 4. Worker-side Present coalescing (SAB ring + queued command batches)
// ---------------------------------------------------------------------------

test('Wave-4 #4: SAB ring backlog and queued command batches execute all non-Present commands in order and coalesce redundant Presents', async () => {
  // 4a. SAB ring backlog: 10 frames written during a stall -> 10 buffers created, compute preserved, 1 swapchain Present, 9 coalescedPresents.
  const mockRing = createMockWebGpu();
  const procRing = new WgpuCommandProcessor(mockRing.device, mockRing.ctx, {
    format: 'bgra8unorm',
  });
  const sab = new SharedArrayBuffer(64 + 256 * 1024);
  const ring = new GpuRingBuffer(sab, { initialize: true });
  procRing.attachRing(ring);

  // Pre-create compute shader/pipeline so intermediate compute dispatch executes.
  assert.equal(
    ring.writeBytes(
      new Uint8Array(
        new BinaryCommandBuilder()
          .createShaderModule(1, SHADER_STAGE.COMPUTE, SMALL_CS_WGSL)
          .createComputePipeline(2, 1)
          .finish()
      )
    ),
    true
  );
  for (let f = 1; f <= 10; f += 1) {
    const b = new BinaryCommandBuilder()
      .beginFrame()
      .createBuffer(100 + f, 64 * f)
      .clearColor(f / 10, 0.2, 0.3, 1.0);
    if (f === 3) {
      b.setComputePipeline(2).dispatchCompute(4, 1, 1);
    }
    b.present();
    assert.equal(ring.writeBytes(new Uint8Array(b.finish())), true);
  }

  const drainRes = procRing.drainRing();
  assert.equal(drainRes.presents, 1, 'Only final Present in ring backlog should acquire/blit swapchain');
  assert.equal(drainRes.frameCount, 1);
  assert.equal(mockRing.currentTextureCalls, 1, 'getCurrentTexture called only once for 10 queued frames');
  assert.equal(procRing.buffers.size, 10, 'All 10 non-Present CreateBuffer commands executed in order');
  for (let f = 1; f <= 10; f += 1) {
    assert.equal(procRing.buffers.get(100 + f).size, 64 * f);
  }
  assert.equal(mockRing.computeDispatches, 1, 'Intermediate compute dispatch was preserved');
  assert.equal(procRing.getStats().coalescedPresents, 9);
  assert.equal(buildGpuTelemetrySnapshot(procRing).coalescedPresents, 9);
  procRing.destroy();

  // 4b. Worker queued command batches via createWgpuWorkerHandler (`execute-batch-queue` + `enqueueCommandBatch`).
  const mockQueue = createMockWebGpu();
  const procQueue = new WgpuCommandProcessor(mockQueue.device, mockQueue.ctx, {
    format: 'bgra8unorm',
  });
  const posted = [];
  const handler = createWgpuWorkerHandler({
    processor: procQueue,
    postMessage: (msg) => posted.push(msg),
  });

  const batches = [];
  for (let i = 1; i <= 8; i += 1) {
    const cmdBuffer = new BinaryCommandBuilder()
      .beginFrame()
      .createBuffer(200 + i, 128 * i)
      .present()
      .finish();
    batches.push({ cmdBuffer });
  }
  await handler.handleMessage({ type: 'execute-batch-queue', batches });

  assert.equal(procQueue.frameCount, 1);
  assert.equal(mockQueue.currentTextureCalls, 1);
  assert.equal(procQueue.buffers.size, 8);
  assert.equal(procQueue.getStats().coalescedPresents, 7);

  await handler.handleMessage({ id: 'q-stats', type: 'request-gpu-stats' });
  const qStatsReply = posted.find((m) => m && m.id === 'q-stats' && m.type === 'gpu-stats');
  assert.equal(qStatsReply.coalescedPresents, 7);

  // 4c. Single command buffer with >=3 Presents packed together also coalesces cleanly.
  const multiBuf = new BinaryCommandBuilder();
  for (let i = 1; i <= 5; i += 1) {
    multiBuf.beginFrame().createBuffer(300 + i, 32).present();
  }
  const multiRes = procQueue.executeCommandBuffer(multiBuf.finish());
  assert.equal(multiRes.presents, 1);
  assert.equal(procQueue.frameCount, 2);
  assert.equal(procQueue.getStats().coalescedPresents, 11); // 7 + 4
  for (let i = 1; i <= 5; i += 1) {
    assert.equal(procQueue.buffers.has(300 + i), true);
  }
  procQueue.destroy();
});

// ---------------------------------------------------------------------------
// 5. HTTP/1.1 connection starvation guard (DEFAULT_MAX_CONCURRENT_RANGE_FETCHES = 3)
// ---------------------------------------------------------------------------

test('Wave-4 #5: wgsl_pack_loader defaults DEFAULT_MAX_CONCURRENT_RANGE_FETCHES to 3 to leave >=3 HTTP/1.1 slots for io_worker', async () => {
  assert.equal(DEFAULT_MAX_CONCURRENT_RANGE_FETCHES, 3);

  const count = 10;
  const entries = [];
  const hashes = [];
  for (let i = 0; i < count; i += 1) {
    const wgsl = `@compute @workgroup_size(${i + 1}) fn f_${i}() {}`;
    const hash = wgpkFnv1a64(wgsl);
    hashes.push(hash);
    entries.push({ hash, stage: WGPK_SHADER_STAGE.COMPUTE, wgsl });
  }
  const packBuf = buildWgslPackBuffer(entries);
  const dataStart = 16 + count * 20;
  const net = createSyntheticRangePackServer(packBuf, {
    dataStartOffset: dataStart,
    deferDataRanges: true,
  });

  const pack = await loadIndexFromUrl('/shaders/http11.wgsl.pack', {
    fetch: net.fetchImpl,
    initialRangeBytes: dataStart,
  });
  assert.equal(pack.getStats().maxConcurrentRangeFetches, 3);

  const fetches = hashes.map((h) => pack.getWgslAsync(h));
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(net.deferredRanges.length, 3, 'Default semaphore must hold at most 3 concurrent range fetches');
  assert.equal(net.peakDataFetches, 3);

  while (net.deferredRanges.length > 0) {
    net.resolveAllDeferred();
    await new Promise((r) => setTimeout(r, 5));
  }
  const results = await Promise.all(fetches);
  assert.equal(results.length, count);
  assert.equal(net.peakDataFetches, 3);
});
