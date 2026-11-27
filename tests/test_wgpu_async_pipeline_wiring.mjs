// W103: AsyncPipelineCache wiring in wgpu_worker.js + render_state_packets.js.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { WgpuCommandProcessor, OPCODES, buildGpuTelemetrySnapshot } = require('../web/workers/wgpu_worker.js');
const { installRenderStatePackets, V2_OPCODES } = require('../web/workers/render_state_packets.js');
const { AsyncPipelineCache } = require('../web/workers/async_pipeline_cache.js');

// ---------------------------------------------------------------------------------------------
// Mock WebGPU device. With `async: true` the create*PipelineAsync calls stay pending until the
// test settles them, so a frame can be flushed while a compile is in flight.
// ---------------------------------------------------------------------------------------------

function createMockGpu({ async = true } = {}) {
  const calls = [];
  const passCalls = [];
  const asyncCompiles = [];
  let nextHandle = 1;
  const obj = (kind, extra = {}) => ({ kind, handle: nextHandle++, ...extra });
  const swapchainView = obj('GPUTextureView');
  const ctx = { getCurrentTexture: () => ({ createView: () => swapchainView }) };
  const makePass = (kind) => {
    const record = (op, args) => passCalls.push({ pass: kind, op, args });
    return {
      setPipeline: (p) => record('setPipeline', [p]),
      setBindGroup: (...a) => record('setBindGroup', a),
      setVertexBuffer: (...a) => record('setVertexBuffer', a),
      setIndexBuffer: (...a) => record('setIndexBuffer', a),
      setViewport: (...a) => record('setViewport', a),
      setScissorRect: (...a) => record('setScissorRect', a),
      draw: (...a) => record('draw', a),
      drawIndexed: (...a) => record('drawIndexed', a),
      dispatchWorkgroups: (...a) => record('dispatchWorkgroups', a),
      end: () => record('end', []),
    };
  };
  const pipelineFor = (kind, desc) =>
    obj(kind, { desc, getBindGroupLayout: (i) => obj('GPUBindGroupLayout', { autoFor: i }) });
  const device = {
    createBuffer: (desc) => obj('GPUBuffer', { size: desc.size, usage: desc.usage, destroy() {} }),
    createShaderModule: (desc) => obj('GPUShaderModule', { code: desc.code }),
    createTexture: (desc) => obj('GPUTexture', { desc, createView: () => obj('GPUTextureView'), destroy() {} }),
    createSampler: (desc) => obj('GPUSampler', { desc }),
    createBindGroupLayout: (desc) => obj('GPUBindGroupLayout', { desc }),
    createPipelineLayout: (desc) => obj('GPUPipelineLayout', { desc }),
    createBindGroup: (desc) => obj('GPUBindGroup', { desc }),
    createRenderPipeline(desc) {
      calls.push({ op: 'createRenderPipeline', desc });
      return pipelineFor('GPURenderPipeline', desc);
    },
    createComputePipeline(desc) {
      calls.push({ op: 'createComputePipeline', desc });
      return pipelineFor('GPUComputePipeline', desc);
    },
    createCommandEncoder: () => ({
      beginRenderPass(desc) {
        passCalls.push({ pass: 'render', op: 'beginRenderPass', args: [desc] });
        return makePass('render');
      },
      beginComputePass() {
        passCalls.push({ pass: 'compute', op: 'beginComputePass', args: [] });
        return makePass('compute');
      },
      finish: () => obj('GPUCommandBuffer'),
    }),
    queue: { writeBuffer() {}, writeTexture() {}, submit() {} },
  };
  if (async) {
    const makeAsync = (kind, gpuKind) => (desc) =>
      new Promise((resolve, reject) => {
        const entry = {
          kind,
          desc,
          pipeline: null,
          resolve() {
            entry.pipeline = pipelineFor(gpuKind, desc);
            resolve(entry.pipeline);
          },
          reject(message = 'GPUValidationError: pipeline layout mismatch') {
            reject(new Error(message));
          },
        };
        asyncCompiles.push(entry);
      });
    device.createRenderPipelineAsync = makeAsync('render', 'GPURenderPipeline');
    device.createComputePipelineAsync = makeAsync('compute', 'GPUComputePipeline');
  }
  return { device, ctx, calls, passCalls, asyncCompiles, swapchainView };
}

// Let settled compile promises run their continuations (cache -> render_state_packets install).
const settle = () => new Promise((resolve) => setImmediate(resolve));

// ---------------------------------------------------------------------------------------------
// Packet builders (same wire format as tests/test_render_state_packets.mjs)
// ---------------------------------------------------------------------------------------------

class Stream {
  constructor() {
    this.parts = [];
  }
  packet(opcode, payload = new Uint8Array(0)) {
    const out = new Uint8Array(8 + payload.byteLength);
    const view = new DataView(out.buffer);
    view.setUint16(0, opcode, true);
    view.setUint32(4, payload.byteLength, true);
    out.set(payload, 8);
    this.parts.push(out);
    return this;
  }
  u32s(opcode, values) {
    const payload = new Uint8Array(values.length * 4);
    const view = new DataView(payload.buffer);
    values.forEach((v, i) => view.setUint32(i * 4, v, true));
    return this.packet(opcode, payload);
  }
  shader(id, stage, code) {
    const text = new TextEncoder().encode(code);
    const payload = new Uint8Array(16 + text.byteLength);
    const view = new DataView(payload.buffer);
    view.setUint32(0, id, true);
    view.setUint32(4, stage, true);
    view.setUint32(8, text.byteLength, true);
    payload.set(text, 16);
    return this.packet(OPCODES.CREATE_SHADER_MODULE, payload);
  }
  bindGroupLayout(id, entries) {
    const payload = new Uint8Array(24 + entries.length * 24);
    const view = new DataView(payload.buffer);
    view.setUint32(0, id, true);
    view.setUint32(4, entries.length, true);
    entries.forEach((e, i) => {
      const o = 24 + i * 24;
      view.setUint32(o, e.binding, true);
      view.setUint32(o + 4, e.visibility, true);
      view.setUint8(o + 8, e.type);
      view.setUint8(o + 9, e.subtype || 0);
    });
    return this.packet(V2_OPCODES.CREATE_BIND_GROUP_LAYOUT, payload);
  }
  pipelineLayout(id, groupIds) {
    const payload = new Uint8Array(24 + groupIds.length * 4);
    const view = new DataView(payload.buffer);
    view.setUint32(0, id, true);
    view.setUint32(4, groupIds.length, true);
    groupIds.forEach((g, i) => view.setUint32(24 + i * 4, g, true));
    return this.packet(V2_OPCODES.CREATE_PIPELINE_LAYOUT, payload);
  }
  renderPipeline2(id, vs, fs, layoutId) {
    const payload = new Uint8Array(392);
    const view = new DataView(payload.buffer);
    view.setUint32(0, id, true);
    view.setUint32(4, vs, true);
    view.setUint32(8, fs, true);
    view.setUint32(12, layoutId, true);
    // Pipeline key at +16: one rgba8unorm color target, sampleCount 1, no vertex buffers.
    view.setUint32(16 + 20, 1, true);
    payload.set([27, 0, 0xf, 0], 16 + 24);
    view.setUint32(16 + 160, 1, true);
    return this.packet(V2_OPCODES.CREATE_RENDER_PIPELINE2, payload);
  }
  computePipeline2(id, cs, layoutId) {
    const payload = new Uint8Array(32);
    const view = new DataView(payload.buffer);
    view.setUint32(0, id, true);
    view.setUint32(4, cs, true);
    view.setUint32(8, layoutId, true);
    return this.packet(V2_OPCODES.CREATE_COMPUTE_PIPELINE2, payload);
  }
  bytes() {
    const total = this.parts.reduce((n, p) => n + p.byteLength, 0);
    const out = new Uint8Array(total);
    let offset = 0;
    for (const p of this.parts) {
      out.set(p, offset);
      offset += p.byteLength;
    }
    return out;
  }
}

const VS_WGSL = '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }';
const FS_WGSL = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }';
const VS_GROUP0_WGSL =
  '@group(0) @binding(0) var<uniform> u_frame: vec4<f32>;\n@vertex fn vs_main() -> @builtin(position) vec4<f32> { return u_frame; }';
const CS_GROUP0_WGSL =
  '@group(0) @binding(0) var<storage, read_write> data: array<u32>;\n@compute @workgroup_size(1) fn cs_main() { data[0] = 1u; }';

const drawFrame = (pipelineId) =>
  new Stream()
    .packet(OPCODES.BEGIN_FRAME)
    .u32s(OPCODES.SET_PIPELINE, [pipelineId])
    .u32s(OPCODES.DRAW, [3, 1, 0, 0])
    .packet(OPCODES.PRESENT)
    .bytes();

const dispatchFrame = (pipelineId) =>
  new Stream()
    .packet(OPCODES.BEGIN_FRAME)
    .u32s(OPCODES.SET_COMPUTE_PIPELINE, [pipelineId])
    .u32s(OPCODES.DISPATCH_COMPUTE, [2, 1, 1])
    .packet(OPCODES.PRESENT)
    .bytes();

function makeProcessor(gpu, options = {}) {
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, {
    format: 'bgra8unorm',
    logger: null,
    ...options,
  });
  const handler = installRenderStatePackets(processor);
  return { processor, handler };
}

const opsOf = (gpu, op) => gpu.passCalls.filter((c) => c.op === op);

// ---------------------------------------------------------------------------------------------

test('processor owns an AsyncPipelineCache (maxConcurrent 2) unless enableAsyncPipelines is false', () => {
  const gpu = createMockGpu();
  const { processor } = makeProcessor(gpu);
  assert.ok(processor.asyncPipelineCache instanceof AsyncPipelineCache);
  assert.equal(processor.asyncPipelineCache.device, gpu.device);
  assert.equal(processor.asyncPipelineCache.maxConcurrent, 2);

  const custom = makeProcessor(createMockGpu(), { maxConcurrentPipelineCompiles: 4 }).processor;
  assert.equal(custom.asyncPipelineCache.maxConcurrent, 4);

  const disabled = makeProcessor(createMockGpu(), { enableAsyncPipelines: false }).processor;
  assert.equal(disabled.asyncPipelineCache, null);
  assert.equal(disabled.getAsyncPipelineStats(), null);
});

test('async CreateRenderPipeline2: pending frame skips the draw cleanly, next frame binds and draws', async () => {
  const gpu = createMockGpu();
  const { processor, handler } = makeProcessor(gpu);

  processor.executeCommandBuffer(
    new Stream().shader(1, 0, VS_WGSL).shader(2, 1, FS_WGSL).renderPipeline2(10, 1, 2, 0).bytes()
  );
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.equal(gpu.calls.filter((c) => c.op === 'createRenderPipeline').length, 0, 'no synchronous compile');
  assert.equal(gpu.asyncCompiles.length, 1);
  assert.equal(processor.asyncPipelineCache.isPending(10), true);
  assert.equal(handler.isAsyncPipelinePending(10, false), true);
  assert.equal(processor.pipelines.has(10), false);

  // Frame N: pipeline still compiling -> setPipeline and draw are dropped without errors.
  const first = processor.executeCommandBuffer(drawFrame(10));
  assert.equal(first.presents, 1);
  assert.equal(opsOf(gpu, 'setPipeline').length, 0);
  assert.equal(opsOf(gpu, 'draw').length, 0);
  assert.equal(handler.telemetry.pendingAsyncPipelineSkips, 1);
  assert.equal(processor.validationErrorCount, 0);
  assert.equal(processor._skippedDrawCount, 0);
  assert.equal(processor.drawCalls ?? processor.getStats().drawCalls, 0);

  gpu.asyncCompiles[0].resolve();
  await processor.asyncPipelineCache.whenIdle();
  await settle();
  assert.equal(processor.pipelines.get(10), gpu.asyncCompiles[0].pipeline);
  assert.equal(handler.isAsyncPipelinePending(10, false), false);
  assert.equal(processor._pipelineMeta.get(10).isV2, true);

  // Frame N+1: the compiled pipeline binds and draws normally.
  processor.executeCommandBuffer(drawFrame(10));
  assert.deepEqual(opsOf(gpu, 'setPipeline').map((c) => c.args[0]), [gpu.asyncCompiles[0].pipeline]);
  assert.equal(opsOf(gpu, 'draw').length, 1);
  assert.equal(handler.telemetry.pendingAsyncPipelineSkips, 1);
  assert.equal(processor.validationErrorCount, 0);

  const stats = processor.asyncPipelineCache.getStats();
  assert.equal(stats.compiled, 1);
  assert.equal(stats.syncFallbacks, 0);
  assert.equal(handler.getTelemetry().asyncPipelineCompiles, 1);
});

test('async render compile rejecting the explicit layout retries with layout:auto and remaps group 0', async () => {
  const gpu = createMockGpu();
  const { processor, handler } = makeProcessor(gpu);
  processor.executeCommandBuffer(
    new Stream()
      .shader(1, 0, VS_GROUP0_WGSL)
      .shader(2, 1, FS_WGSL)
      .bindGroupLayout(5, [{ binding: 0, visibility: 3, type: 0, subtype: 0 }]) // uniform @0
      .pipelineLayout(6, [5])
      .renderPipeline2(11, 1, 2, 6)
      .bytes()
  );
  assert.equal(handler.stats.errors, 0, handler.lastError);
  const fallbacksBefore = handler.telemetry.autoLayoutFallbacks;
  assert.equal(gpu.asyncCompiles.length, 1);
  assert.notEqual(gpu.asyncCompiles[0].desc.layout, 'auto', 'explicit root-signature layout first');

  gpu.asyncCompiles[0].reject();
  await settle();
  assert.equal(gpu.asyncCompiles.length, 2);
  assert.equal(gpu.asyncCompiles[1].desc.layout, 'auto');
  assert.equal(processor.asyncPipelineCache.isPending(11), true, 'retry keeps the pipeline pending');

  gpu.asyncCompiles[1].resolve();
  await processor.asyncPipelineCache.whenIdle();
  await settle();
  assert.equal(processor.pipelines.get(11), gpu.asyncCompiles[1].pipeline);
  assert.equal(handler.renderPipelineAutoLayoutMeta.get(11).autoLayoutRemap, true);
  assert.equal(handler.telemetry.autoLayoutFallbacks, fallbacksBefore + 1);
  assert.equal(processor.asyncPipelineCache.getStats().autoLayoutRetries, 1);
  assert.equal(handler.telemetry.asyncPipelineFailures, 0);
});

test('async CreateComputePipeline2: pending dispatch skipped, auto-layout retry, then dispatches', async () => {
  const gpu = createMockGpu();
  const { processor, handler } = makeProcessor(gpu);
  processor.executeCommandBuffer(
    new Stream()
      .shader(3, 2, CS_GROUP0_WGSL)
      .bindGroupLayout(7, [{ binding: 0, visibility: 4, type: 0, subtype: 1 }]) // storage @0
      .pipelineLayout(8, [7])
      .computePipeline2(30, 3, 8)
      .bytes()
  );
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.equal(gpu.calls.filter((c) => c.op === 'createComputePipeline').length, 0);
  assert.equal(gpu.asyncCompiles.length, 1);
  assert.notEqual(gpu.asyncCompiles[0].desc.layout, 'auto');
  assert.equal(handler.telemetry.computePipelineCount, 0, 'counted once the compile resolves');

  // Pending: the dispatch is dropped without a missing-pipeline skip/warning.
  processor.executeCommandBuffer(dispatchFrame(30));
  assert.equal(opsOf(gpu, 'dispatchWorkgroups').length, 0);
  assert.equal(handler.telemetry.pendingAsyncPipelineSkips, 1);
  assert.equal(handler.telemetry.skippedComputeDispatchCount, 0);
  assert.equal(handler.telemetry.computeDispatchCount, 0);
  assert.equal(handler.telemetry.warnings.length, 0);
  assert.equal(processor.validationErrorCount, 0);

  // Dawn rejects the explicit layout asynchronously -> one retry with layout:'auto'.
  gpu.asyncCompiles[0].reject();
  await settle();
  assert.equal(gpu.asyncCompiles[1].desc.layout, 'auto');
  gpu.asyncCompiles[1].resolve();
  await processor.asyncPipelineCache.whenIdle();
  await settle();

  assert.equal(processor.computePipelines.get(30), gpu.asyncCompiles[1].pipeline);
  const meta = handler.computePipelineMeta.get(30);
  assert.equal(meta.autoLayoutRemap, true);
  assert.equal(meta.wgslBindings.length, 1);
  assert.equal(handler.telemetry.autoLayoutFallbacks, 1);
  assert.equal(handler.telemetry.computePipelineCount, 1);
  assert.equal(processor.asyncPipelineCache.getStats().autoLayoutRetries, 1);

  processor.executeCommandBuffer(dispatchFrame(30));
  assert.deepEqual(opsOf(gpu, 'dispatchWorkgroups').map((c) => c.args), [[2, 1, 1]]);
  assert.ok(opsOf(gpu, 'setPipeline').some((c) => c.args[0] === gpu.asyncCompiles[1].pipeline));
  assert.equal(handler.telemetry.computeDispatchCount, 1);
  assert.equal(handler.telemetry.pendingAsyncPipelineSkips, 1);
});

test('async compile failing even after the retry is reported once and leaves the pipeline absent', async () => {
  const gpu = createMockGpu();
  const { processor, handler } = makeProcessor(gpu);
  processor.executeCommandBuffer(
    new Stream().shader(1, 0, VS_WGSL).shader(2, 1, FS_WGSL).renderPipeline2(12, 1, 2, 0).bytes()
  );
  gpu.asyncCompiles[0].reject('GPUValidationError: bad blend state');
  await processor.asyncPipelineCache.whenIdle();
  await settle();
  assert.equal(processor.pipelines.has(12), false);
  assert.equal(handler.isAsyncPipelinePending(12, false), false);
  assert.equal(handler.telemetry.asyncPipelineFailures, 1);
  assert.equal(handler.stats.errors, 1);
  assert.match(handler.lastError, /render pipeline 12: .*bad blend state/);
});

test('v1 CreateComputePipeline: auto layout stays synchronous; reflected layout compiles async', async () => {
  const gpu = createMockGpu();
  const { processor, handler } = makeProcessor(gpu);
  processor.executeCommandBuffer(
    new Stream()
      .shader(3, 2, CS_GROUP0_WGSL)
      .u32s(OPCODES.CREATE_COMPUTE_PIPELINE, [40, 3, 0, 0])
      .bytes()
  );
  assert.equal(gpu.calls.filter((c) => c.op === 'createComputePipeline').length, 1);
  assert.equal(gpu.asyncCompiles.length, 0);
  assert.ok(processor.computePipelines.has(40));

  // Pretend shader reflection produced an explicit layout for pipeline 41.
  const reflectedBgl = gpu.device.createBindGroupLayout({ entries: [] });
  const reflectedLayout = gpu.device.createPipelineLayout({ bindGroupLayouts: [reflectedBgl] });
  const origBuild = processor._buildComputePipelineDescriptorV1.bind(processor);
  processor._buildComputePipelineDescriptorV1 = (view, offset) => {
    const built = origBuild(view, offset);
    processor._computePipelineBindGroupLayouts.set(built.id, [reflectedBgl]);
    return { ...built, reflected: true, descriptor: { ...built.descriptor, layout: reflectedLayout } };
  };
  processor.executeCommandBuffer(
    new Stream().u32s(OPCODES.CREATE_COMPUTE_PIPELINE, [41, 3, 0, 0]).bytes()
  );
  assert.equal(gpu.asyncCompiles.length, 1);
  assert.equal(gpu.asyncCompiles[0].desc.layout, reflectedLayout);
  assert.equal(processor.isAsyncPipelinePending(41), true);
  // v1 CreateBindGroup against the pending pipeline resolves its reflected layout.
  assert.equal(processor._pendingAsyncPipelineBindGroupLayout(41, 0), reflectedBgl);

  processor.executeCommandBuffer(dispatchFrame(41));
  assert.equal(opsOf(gpu, 'dispatchWorkgroups').length, 0);
  assert.equal(handler.telemetry.pendingAsyncPipelineSkips, 1);

  gpu.asyncCompiles[0].resolve();
  await processor.asyncPipelineCache.whenIdle();
  await settle();
  assert.equal(processor.computePipelines.get(41), gpu.asyncCompiles[0].pipeline);
  assert.equal(handler.computePipelineMeta.get(41).autoLayoutRemap, false);
  processor.executeCommandBuffer(dispatchFrame(41));
  assert.equal(opsOf(gpu, 'dispatchWorkgroups').length, 1);
});

test('devices without create*PipelineAsync keep the exact synchronous path', () => {
  const gpu = createMockGpu({ async: false });
  const { processor, handler } = makeProcessor(gpu);
  processor.executeCommandBuffer(
    new Stream()
      .shader(1, 0, VS_WGSL)
      .shader(2, 1, FS_WGSL)
      .shader(3, 2, CS_GROUP0_WGSL)
      .renderPipeline2(10, 1, 2, 0)
      .computePipeline2(30, 3, 0)
      .bytes()
  );
  assert.equal(gpu.calls.filter((c) => c.op === 'createRenderPipeline').length, 1);
  assert.equal(gpu.calls.filter((c) => c.op === 'createComputePipeline').length, 1);
  assert.ok(processor.pipelines.has(10));
  assert.ok(processor.computePipelines.has(30));
  assert.equal(handler.telemetry.asyncPipelineCompiles, 0);
  const stats = processor.asyncPipelineCache.getStats();
  assert.equal(stats.compiled + stats.syncFallbacks + stats.queued + stats.inFlight, 0);

  processor.executeCommandBuffer(drawFrame(10));
  assert.equal(opsOf(gpu, 'draw').length, 1);
  assert.equal(handler.telemetry.pendingAsyncPipelineSkips, 0);
});

test('worker flush never records SetPipeline/Draw errors for a pipeline still compiling', () => {
  const gpu = createMockGpu();
  // No render_state_packets.js extension: the worker's own guards handle the pending pipeline.
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'bgra8unorm', logger: null });
  processor.asyncPipelineCache.compileRenderPipeline(77, { layout: 'auto' });
  assert.equal(processor.isAsyncPipelinePending(77), true);
  processor.executeCommandBuffer(drawFrame(77));
  assert.equal(processor.validationErrorCount, 0);
  assert.equal(opsOf(gpu, 'draw').length, 0);
  assert.equal(processor.getAsyncPipelineStats().pendingAsyncPipelineSkips, 1);

  // A genuinely missing pipeline is still reported.
  processor.executeCommandBuffer(drawFrame(78));
  assert.ok(processor.validationErrorCount > 0);
});

test('rebindDevice resets the cache onto the new device and discards stale in-flight compiles', async () => {
  const oldGpu = createMockGpu();
  const { processor, handler } = makeProcessor(oldGpu);
  processor.executeCommandBuffer(
    new Stream().shader(1, 0, VS_WGSL).shader(2, 1, FS_WGSL).renderPipeline2(10, 1, 2, 0).bytes()
  );
  assert.equal(processor.asyncPipelineCache.isPending(10), true);

  const resets = [];
  const cache = processor.asyncPipelineCache;
  const origReset = cache.reset.bind(cache);
  cache.reset = (device) => {
    resets.push(device);
    return origReset(device);
  };

  const newGpu = createMockGpu();
  processor.rebindDevice(newGpu.device, newGpu.ctx, { format: 'bgra8unorm' });
  assert.deepEqual(resets, [newGpu.device]);
  assert.equal(cache.device, newGpu.device);
  assert.equal(cache.generation, 1);
  assert.equal(cache.isPending(10), false);

  // The dead device finishes its compile late: dropped, never written into the new device's tables.
  oldGpu.asyncCompiles[0].resolve();
  await settle();
  assert.equal(processor.pipelines.has(10), false);
  assert.equal(handler.isAsyncPipelinePending(10, false), false);
  assert.equal(cache.getStats().discardedStale, 1);
  assert.equal(cache.getStats().compiled, 0);

  // Re-installed extensions compile on the new device.
  const newHandler = installRenderStatePackets(processor);
  processor.executeCommandBuffer(
    new Stream().shader(1, 0, VS_WGSL).shader(2, 1, FS_WGSL).renderPipeline2(10, 1, 2, 0).bytes()
  );
  assert.equal(newGpu.asyncCompiles.length, 1);
  newGpu.asyncCompiles[0].resolve();
  await cache.whenIdle();
  await settle();
  assert.equal(processor.pipelines.get(10), newGpu.asyncCompiles[0].pipeline);
  assert.equal(newHandler.stats.errors, 0, newHandler.lastError);
});

test('DestroyResource and destroy() drop pending compiles without installing them', async () => {
  const gpu = createMockGpu();
  const { processor, handler } = makeProcessor(gpu);
  processor.executeCommandBuffer(
    new Stream()
      .shader(1, 0, VS_WGSL)
      .shader(2, 1, FS_WGSL)
      .renderPipeline2(10, 1, 2, 0)
      .renderPipeline2(11, 1, 2, 0)
      .u32s(OPCODES.DESTROY_RESOURCE, [10])
      .bytes()
  );
  assert.equal(processor.asyncPipelineCache.isPending(10), false);
  assert.equal(handler.isAsyncPipelinePending(10, false), false);
  assert.equal(processor.asyncPipelineCache.isPending(11), true);
  gpu.asyncCompiles[0].resolve();
  await settle();
  assert.equal(processor.pipelines.has(10), false, 'destroyed id is never installed');

  processor.destroy();
  assert.equal(processor.asyncPipelineCache.device, null);
  assert.equal(processor.asyncPipelineCache.isPending(11), false);
  gpu.asyncCompiles[1].resolve();
  await settle();
  assert.equal(processor.pipelines.has(11), false);
});

test('buildGpuTelemetrySnapshot surfaces asyncPipelineStats', async () => {
  const gpu = createMockGpu();
  const { processor } = makeProcessor(gpu);
  processor.executeCommandBuffer(
    new Stream().shader(1, 0, VS_WGSL).shader(2, 1, FS_WGSL).renderPipeline2(10, 1, 2, 0).bytes()
  );
  processor.executeCommandBuffer(drawFrame(10));
  let snap = buildGpuTelemetrySnapshot(processor, { timestamp: 1 });
  assert.deepEqual(snap.asyncPipelineStats, {
    generation: 0,
    maxConcurrent: 2,
    queued: 0,
    inFlight: 1,
    peakQueueDepth: 1,
    compiled: 0,
    failed: 0,
    dedupeHits: 0,
    syncFallbacks: 0,
    autoLayoutRetries: 0,
    discardedStale: 0,
    pendingAsyncPipelineSkips: 1,
  });

  gpu.asyncCompiles[0].resolve();
  await processor.asyncPipelineCache.whenIdle();
  snap = buildGpuTelemetrySnapshot(processor, { timestamp: 2 });
  assert.equal(snap.asyncPipelineStats.compiled, 1);
  assert.equal(snap.asyncPipelineStats.inFlight, 0);

  const disabled = makeProcessor(createMockGpu(), { enableAsyncPipelines: false }).processor;
  assert.equal(buildGpuTelemetrySnapshot(disabled, { timestamp: 3 }).asyncPipelineStats, null);
  assert.equal('asyncPipelineStats' in buildGpuTelemetrySnapshot(null, { timestamp: 4 }), false);
});
