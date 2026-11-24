// W104: live GPU resource creation packets & shadow buffer writes are replayed across WebGPU
// device.lost recovery so guest resource ids stay valid on the recovered GPUDevice.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { WgpuCommandProcessor, createWgpuWorkerHandler, buildGpuTelemetrySnapshot, OPCODES } =
  require('../web/workers/wgpu_worker.js');
const rs = require('../web/workers/render_state_packets.js');
const { installRenderStatePackets, V2_OPCODES } = rs;

// ---------------------------------------------------------------------------
// Packet builders
// ---------------------------------------------------------------------------

function packet(opcode, payload = new Uint8Array(0)) {
  const out = new Uint8Array(8 + payload.byteLength);
  const view = new DataView(out.buffer);
  view.setUint16(0, opcode, true);
  view.setUint32(4, payload.byteLength, true);
  out.set(payload, 8);
  return out;
}

function concat(...parts) {
  const total = parts.reduce((n, p) => n + p.byteLength, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const p of parts) {
    out.set(p, offset);
    offset += p.byteLength;
  }
  return out;
}

function u32Packet(opcode, values) {
  const payload = new Uint8Array(values.length * 4);
  const view = new DataView(payload.buffer);
  values.forEach((v, i) => view.setUint32(i * 4, v, true));
  return packet(opcode, payload);
}

function createBufferPacket(id, usage, size) {
  const payload = new Uint8Array(16);
  const view = new DataView(payload.buffer);
  view.setUint32(0, id, true);
  view.setUint32(4, usage, true);
  view.setBigUint64(8, BigInt(size), true);
  return packet(OPCODES.CREATE_BUFFER, payload);
}

function writeBufferPacket(id, dstOffset, bytes) {
  const payload = new Uint8Array(16 + bytes.byteLength);
  const view = new DataView(payload.buffer);
  view.setUint32(0, id, true);
  view.setUint32(4, dstOffset, true);
  view.setUint32(8, bytes.byteLength, true);
  payload.set(bytes, 16);
  return packet(OPCODES.WRITE_BUFFER, payload);
}

function shaderPacket(id, stage, code) {
  const text = new TextEncoder().encode(code);
  const payload = new Uint8Array(16 + text.byteLength);
  const view = new DataView(payload.buffer);
  view.setUint32(0, id, true);
  view.setUint32(4, stage, true);
  view.setUint32(8, text.byteLength, true);
  payload.set(text, 16);
  return packet(OPCODES.CREATE_SHADER_MODULE, payload);
}

function createTexture2Packet({ id, width, height, usage, format }) {
  const payload = new Uint8Array(40);
  const view = new DataView(payload.buffer);
  view.setUint32(0, id, true);
  view.setUint32(4, width, true);
  view.setUint32(8, height, true);
  view.setUint32(12, 1, true);
  view.setUint32(16, 1, true);
  view.setUint32(20, 1, true);
  view.setUint32(24, usage, true);
  view.setUint8(28, 1); // 2d
  view.setUint8(29, format);
  return packet(V2_OPCODES.CREATE_TEXTURE2, payload);
}

function createSamplerPacket(id) {
  // v1 CreateSampler: id, addressU/V/W, mag/min/mip filter, compare (8 x u32).
  return u32Packet(OPCODES.CREATE_SAMPLER, [id, 0, 0, 0, 1, 1, 0, 0]);
}

function bindGroupLayoutPacket(id, entries) {
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
    view.setUint8(o + 10, e.dimension || 0);
    view.setUint8(o + 11, e.flags || 0);
    view.setBigUint64(o + 16, BigInt(e.minBindingSize || 0), true);
  });
  return packet(V2_OPCODES.CREATE_BIND_GROUP_LAYOUT, payload);
}

function pipelineLayoutPacket(id, groupIds) {
  const payload = new Uint8Array(24 + groupIds.length * 4);
  const view = new DataView(payload.buffer);
  view.setUint32(0, id, true);
  view.setUint32(4, groupIds.length, true);
  groupIds.forEach((g, i) => view.setUint32(24 + i * 4, g, true));
  return packet(V2_OPCODES.CREATE_PIPELINE_LAYOUT, payload);
}

function renderKey(formatCode) {
  const key = new Uint8Array(376);
  const v = new DataView(key.buffer);
  v.setUint32(20, 1, true); // 1 color target
  key.set([formatCode, 0, 0xf, 0], 24);
  v.setUint32(160, 1, true); // sampleCount = 1
  return key;
}

function renderPipeline2Packet(id, vs, fs, layoutId, keyBytes) {
  const payload = new Uint8Array(392);
  const view = new DataView(payload.buffer);
  view.setUint32(0, id, true);
  view.setUint32(4, vs, true);
  view.setUint32(8, fs, true);
  view.setUint32(12, layoutId, true);
  payload.set(keyBytes, 16);
  return packet(V2_OPCODES.CREATE_RENDER_PIPELINE2, payload);
}

function bindGroup2Packet(id, layoutId, entries) {
  const payload = new Uint8Array(16 + entries.length * 48);
  const view = new DataView(payload.buffer);
  view.setUint32(0, id, true);
  view.setUint32(4, layoutId, true);
  view.setUint32(8, entries.length, true);
  entries.forEach((e, i) => {
    const o = 16 + i * 48;
    view.setUint32(o, e.binding, true);
    view.setUint32(o + 4, e.type, true);
    view.setUint32(o + 8, e.id, true);
    view.setBigUint64(o + 16, BigInt(e.offset || 0), true);
    view.setBigUint64(o + 24, BigInt(e.size || 0), true);
  });
  return packet(V2_OPCODES.CREATE_BIND_GROUP2, payload);
}

function setBindGroup2Packet(index, id) {
  const payload = new Uint8Array(16);
  const view = new DataView(payload.buffer);
  view.setUint32(0, index, true);
  view.setUint32(4, id, true);
  return packet(V2_OPCODES.SET_BIND_GROUP2, payload);
}

function beginRenderPassPacket(textureId) {
  const payload = new Uint8Array(368);
  const view = new DataView(payload.buffer);
  view.setUint32(0, 1, true);
  view.setUint32(8, textureId, true);
  view.setUint8(12, 2); // clear
  view.setUint8(13, 1); // store
  return packet(V2_OPCODES.BEGIN_RENDER_PASS, payload);
}

const destroyPacket = (id) => u32Packet(OPCODES.DESTROY_RESOURCE, [id]);

// ---------------------------------------------------------------------------
// Scene: buffer + shadow write, texture 2 (render target) + texture 3 (sampled), sampler,
// VS/FS shader modules, bind group layout, pipeline layout, pipeline 880, bind group 5125.
// ---------------------------------------------------------------------------

const RGBA8 = rs.TEXTURE_FORMAT_NAMES.indexOf('rgba8unorm');
const IDS = Object.freeze({
  buffer: 100,
  rt: 2,
  sampled: 3,
  sampler: 50,
  vs: 10,
  fs: 11,
  bgl: 200,
  pl: 201,
  pipeline: 880,
  bindGroup: 5125,
});

const VS = [
  'struct Cb { m: vec4<f32> };',
  '@group(0) @binding(0) var<uniform> cb0_: Cb;',
  '@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {',
  '  return cb0_.m;',
  '}',
].join('\n');
const FS = [
  '@group(0) @binding(1) var tex0_: texture_2d<f32>;',
  '@group(0) @binding(2) var samp0_: sampler;',
  '@fragment fn fs_main() -> @location(0) vec4<f32> {',
  '  return textureSample(tex0_, samp0_, vec2<f32>(0.5, 0.5));',
  '}',
].join('\n');

const UNIFORM_BYTES = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
const PARTIAL_BYTES = new Uint8Array([0xaa, 0xbb, 0xcc, 0xdd]);

function sceneSetupPackets() {
  return concat(
    createBufferPacket(IDS.buffer, 0x0040 | 0x0008, 256),
    writeBufferPacket(IDS.buffer, 0, UNIFORM_BYTES),
    writeBufferPacket(IDS.buffer, 32, PARTIAL_BYTES),
    createTexture2Packet({ id: IDS.rt, width: 64, height: 64, usage: 0x10 | 0x04, format: RGBA8 }),
    createTexture2Packet({ id: IDS.sampled, width: 16, height: 16, usage: 0x04 | 0x02, format: RGBA8 }),
    createSamplerPacket(IDS.sampler),
    shaderPacket(IDS.vs, 0, VS),
    shaderPacket(IDS.fs, 1, FS),
    bindGroupLayoutPacket(IDS.bgl, [
      { binding: 0, visibility: 3, type: 0, subtype: 0, minBindingSize: 16 },
      { binding: 1, visibility: 2, type: 2, subtype: 0, dimension: 2 },
      { binding: 2, visibility: 2, type: 1, subtype: 0 },
    ]),
    pipelineLayoutPacket(IDS.pl, [IDS.bgl]),
    renderPipeline2Packet(IDS.pipeline, IDS.vs, IDS.fs, IDS.pl, renderKey(RGBA8)),
    bindGroup2Packet(IDS.bindGroup, IDS.bgl, [
      { binding: 0, type: 0, id: IDS.buffer, offset: 0, size: 256 },
      { binding: 1, type: 1, id: IDS.sampled },
      { binding: 2, type: 2, id: IDS.sampler },
    ])
  );
}

function drawFramePackets(textureId = IDS.rt) {
  return concat(
    packet(OPCODES.BEGIN_FRAME),
    beginRenderPassPacket(textureId),
    u32Packet(OPCODES.SET_PIPELINE, [IDS.pipeline]),
    setBindGroup2Packet(0, IDS.bindGroup),
    u32Packet(OPCODES.DRAW, [3, 1, 0, 0]),
    packet(V2_OPCODES.END_RENDER_PASS),
    packet(OPCODES.PRESENT)
  );
}

// ---------------------------------------------------------------------------
// Mock WebGPU device (every object remembers the device that created it)
// ---------------------------------------------------------------------------

function createMockDevice(label, { asyncPipelines = false } = {}) {
  const calls = [];
  const passCalls = [];
  let resolveLost;
  const lost = new Promise((resolve) => {
    resolveLost = resolve;
  });
  const obj = (kind, extra = {}) => ({ kind, device: label, ...extra });
  const record = (op, result, desc) => {
    calls.push({ op, desc, result });
    return result;
  };
  const makePass = () => ({
    setPipeline: (p) => passCalls.push({ op: 'setPipeline', args: [p] }),
    setBindGroup: (i, bg) => passCalls.push({ op: 'setBindGroup', args: [i, bg] }),
    setVertexBuffer() {},
    setIndexBuffer() {},
    setViewport() {},
    setScissorRect() {},
    setBlendConstant() {},
    setStencilReference() {},
    draw: (...a) => passCalls.push({ op: 'draw', args: a }),
    drawIndexed() {},
    dispatchWorkgroups() {},
    end: () => passCalls.push({ op: 'end', args: [] }),
  });
  const device = {
    label,
    lost,
    features: new Set(),
    limits: {},
    queue: {
      writeBuffer(buffer, dstOffset, data, dataOffset = 0, size) {
        const src = data instanceof Uint8Array ? data : new Uint8Array(data.buffer || data);
        const len = size === undefined ? src.byteLength - dataOffset : size;
        calls.push({
          op: 'writeBuffer',
          buffer,
          dstOffset,
          bytes: Uint8Array.from(src.subarray(dataOffset, dataOffset + len)),
        });
      },
      writeTexture() {},
      submit(buffers) {
        calls.push({ op: 'submit', buffers });
      },
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
    createBuffer: (desc) =>
      record('createBuffer', obj('GPUBuffer', { size: desc.size, destroyed: false, destroy() { this.destroyed = true; } }), desc),
    createTexture(desc) {
      const texture = obj('GPUTexture', {
        desc,
        createView: (viewDesc) => obj('GPUTextureView', { texture, viewDesc }),
        destroy() {},
      });
      return record('createTexture', texture, desc);
    },
    createSampler: (desc) => record('createSampler', obj('GPUSampler', { desc }), desc),
    createShaderModule: (desc) => record('createShaderModule', obj('GPUShaderModule', { code: desc.code }), desc),
    createBindGroupLayout: (desc) => record('createBindGroupLayout', obj('GPUBindGroupLayout', { desc }), desc),
    createPipelineLayout: (desc) => record('createPipelineLayout', obj('GPUPipelineLayout', { desc }), desc),
    createRenderPipeline: (desc) =>
      record('createRenderPipeline', obj('GPURenderPipeline', { desc, getBindGroupLayout: () => obj('GPUBindGroupLayout') }), desc),
    createComputePipeline: (desc) =>
      record('createComputePipeline', obj('GPUComputePipeline', { desc, getBindGroupLayout: () => obj('GPUBindGroupLayout') }), desc),
    createBindGroup: (desc) => record('createBindGroup', obj('GPUBindGroup', { desc }), desc),
    createCommandEncoder() {
      return {
        beginRenderPass(desc) {
          passCalls.push({ op: 'beginRenderPass', args: [desc] });
          return makePass();
        },
        beginComputePass: () => makePass(),
        copyBufferToBuffer() {},
        copyBufferToTexture() {},
        finish: () => obj('GPUCommandBuffer'),
      };
    },
    pushErrorScope() {},
    popErrorScope: () => Promise.resolve(null),
    addEventListener() {},
    removeEventListener() {},
    destroy() {
      resolveLost({ reason: 'destroyed', message: 'Device was destroyed.' });
    },
  };
  if (asyncPipelines) {
    device.createRenderPipelineAsync = async (desc) => device.createRenderPipeline(desc);
    device.createComputePipelineAsync = async (desc) => device.createComputePipeline(desc);
  }
  device.lose = (info = { reason: 'unknown', message: 'GPU process lost' }) => resolveLost(info);
  const ops = (op) => calls.filter((c) => c.op === op);
  return { device, calls, passCalls, ops };
}

function createMockContext() {
  return {
    configured: [],
    getCurrentTexture() {
      return { createView: () => ({ kind: 'GPUTextureView', id: 'swapchain' }) };
    },
    configure(desc) {
      this.configured.push(desc);
    },
  };
}

async function settle(rounds = 8) {
  for (let i = 0; i < rounds; i += 1) {
    await new Promise((resolve) => setImmediate(resolve));
  }
}

async function withMockNavigatorGpu(gpu, fn) {
  const prev = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', { value: { gpu }, configurable: true, writable: true });
  try {
    return await fn();
  } finally {
    if (prev) Object.defineProperty(globalThis, 'navigator', prev);
    else delete globalThis.navigator;
  }
}

function createMockGpu(device) {
  return {
    async requestAdapter() {
      return {
        features: new Set(),
        limits: {},
        info: { vendor: 'mock' },
        requestDevice: async () => device,
      };
    },
    getPreferredCanvasFormat: () => 'bgra8unorm',
  };
}

function createMockCanvas() {
  const ctx = createMockContext();
  return { ctx, canvas: { width: 0, height: 0, getContext: () => ctx } };
}

function newProcessor(device, extra = {}) {
  const processor = new WgpuCommandProcessor(device, createMockContext(), {
    format: 'bgra8unorm',
    enableShaderFallback: false,
    enableAsyncPipelines: false,
    logger: null,
    ...extra,
  });
  installRenderStatePackets(processor);
  return processor;
}

/** Simulate the worker's recovery sequence (rebind -> reinstall extensions -> replay). */
function recoverOnto(processor, device) {
  processor.beginDeviceRecovery();
  processor.rebindDevice(device, createMockContext(), { format: 'bgra8unorm' });
  installRenderStatePackets(processor);
  const summary = processor.replayRecordedResources();
  processor.endDeviceRecovery(true);
  return summary;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test('W104 handler recovery replays every live resource and draws onto the new device with 0 validation errors', async () => {
  const first = createMockDevice('first');
  const second = createMockDevice('second');
  const { canvas } = createMockCanvas();
  const posted = [];
  const handler = createWgpuWorkerHandler({
    postMessage: (m) => posted.push(m),
    enableSwapchain: false,
    logger: null,
    recoverGpuDevice: async () => ({ device: second.device, ctx: null, format: 'bgra8unorm' }),
  });

  await withMockNavigatorGpu(createMockGpu(first.device), async () => {
    await handler.handleMessage({ init: true, canvas, width: 64, height: 64 });
    const processor = handler.processor;
    assert.ok(processor && processor.extension, 'processor with ABI v2 extensions');

    await handler.handleMessage({ cmdBuffer: concat(sceneSetupPackets(), drawFramePackets()) });
    assert.equal(processor.validationErrorCount, 0, JSON.stringify(processor.getValidationErrors()));
    assert.equal(first.passCalls.filter((c) => c.op === 'draw').length, 1);
    assert.equal(processor.frameCount, 1);

    // Chrome loses the WebGPU device; the guest keeps using its old resource ids.
    first.device.lose({ reason: 'unknown', message: 'GPU process crashed' });
    await settle();

    assert.equal(processor.isDeviceRecovering, false);
    assert.equal(processor.device, second.device);
    assert.equal(processor.getDeviceLossStats().deviceRecoveredCount, 1);

    // Every live resource exists again — created on the NEW device.
    assert.equal(processor.buffers.get(IDS.buffer).device, 'second');
    assert.equal(processor.textures.get(IDS.rt).device, 'second');
    assert.equal(processor.textures.get(IDS.sampled).device, 'second');
    assert.equal(processor.samplers.get(IDS.sampler).device, 'second');
    assert.equal(second.ops('createShaderModule').filter((c) => c.desc.code === VS || c.desc.code === FS).length, 2);
    assert.ok(processor.shaderModules.has(IDS.vs) && processor.shaderModules.has(IDS.fs));
    const rsExt = processor._getRenderStateExtension();
    assert.equal(rsExt.bindGroupLayouts.get(IDS.bgl).layout.device, 'second');
    assert.equal(rsExt.pipelineLayouts.get(IDS.pl).device, 'second');
    const pipelineEntry = processor.pipelines.get(IDS.pipeline);
    const pipeline = pipelineEntry && pipelineEntry.pipeline ? pipelineEntry.pipeline : pipelineEntry;
    assert.equal(pipeline.device, 'second');
    assert.equal(processor.bindGroups.get(IDS.bindGroup).device, 'second');

    // Shadow buffer contents (two merged partial writes) are re-uploaded via queue.writeBuffer.
    const replayWrites = second.ops('writeBuffer');
    assert.equal(replayWrites.length, 1);
    assert.equal(replayWrites[0].buffer, processor.buffers.get(IDS.buffer));
    assert.equal(replayWrites[0].dstOffset, 0);
    const expected = new Uint8Array(36);
    expected.set(UNIFORM_BYTES, 0);
    expected.set(PARTIAL_BYTES, 32);
    assert.deepEqual(replayWrites[0].bytes, expected);

    // The guest's next frame renders into texture 2 on the recovered device without errors.
    const errorsBefore = processor.validationErrorCount;
    await handler.handleMessage({ cmdBuffer: drawFramePackets() });
    assert.equal(
      processor.validationErrorCount,
      errorsBefore,
      JSON.stringify(processor.getValidationErrors().slice(errorsBefore))
    );
    assert.equal(processor.validationErrorCount, 0);
    assert.equal(processor.frameCount, 2);
    const begin = second.passCalls.find((c) => c.op === 'beginRenderPass');
    assert.ok(begin, 'render pass opened on the new device');
    assert.equal(begin.args[0].colorAttachments[0].view.texture, processor.textures.get(IDS.rt));
    assert.equal(second.passCalls.filter((c) => c.op === 'draw').length, 1);
    const setPipe = second.passCalls.find((c) => c.op === 'setPipeline');
    assert.equal(setPipe.args[0].device, 'second');
    const setBg = second.passCalls.find((c) => c.op === 'setBindGroup');
    assert.equal(setBg.args[1].device, 'second');
    assert.ok(second.ops('submit').length >= 1);
  });

  const telemetry = posted.filter((m) => m && m.type === 'gpu-telemetry');
  const recovered = telemetry.find((m) => m.deviceStats && m.deviceStats.deviceRecoveredCount === 1);
  assert.ok(recovered, 'telemetry emitted after recovery');
  // buffer, 2 textures, sampler, 2 shader modules, BGL, PL, pipeline, bind group.
  assert.equal(recovered.deviceStats.replayedResourceCount, 10);
  assert.equal(recovered.deviceStats.replayedBufferWriteCount, 1);
});

test('W104 replay recreates resources in dependency order and never re-records itself', () => {
  const first = createMockDevice('first');
  const processor = newProcessor(first.device);
  processor.executeCommandBuffer(sceneSetupPackets());
  assert.equal(processor.validationErrorCount, 0, JSON.stringify(processor.getValidationErrors()));
  const before = processor.getResourceReplayStats();
  assert.equal(before.recordedResourceCount, 10);
  assert.equal(before.shadowBufferCount, 1);

  const second = createMockDevice('second');
  const summary = recoverOnto(processor, second.device);
  assert.deepEqual(summary, { replayedResources: 10, failedResources: 0, replayedBufferWrites: 1 });

  const order = second.calls
    .map((c) => c.op)
    .filter((op) => /^create|^writeBuffer$/.test(op));
  const firstIdx = (op) => order.indexOf(op);
  const lastIdx = (op) => order.lastIndexOf(op);
  assert.ok(lastIdx('createBuffer') < firstIdx('writeBuffer'));
  assert.ok(lastIdx('createTexture') < firstIdx('createBindGroupLayout'));
  assert.ok(lastIdx('writeBuffer') < firstIdx('createBindGroupLayout'));
  assert.ok(lastIdx('createBindGroupLayout') < firstIdx('createPipelineLayout'));
  assert.ok(lastIdx('createPipelineLayout') < firstIdx('createRenderPipeline'));
  assert.ok(lastIdx('createRenderPipeline') < firstIdx('createBindGroup'));

  // Replay never duplicated log entries; a second recovery replays exactly the same set.
  const after = processor.getResourceReplayStats();
  assert.equal(after.recordedResourceCount, 10);
  assert.equal(after.recordedResourceBytes, before.recordedResourceBytes);
  assert.equal(after.replayedResourceCount, 10);
  assert.equal(processor._replayingResources, false);

  const third = createMockDevice('third');
  const summary2 = recoverOnto(processor, third.device);
  assert.deepEqual(summary2, { replayedResources: 10, failedResources: 0, replayedBufferWrites: 1 });
  assert.equal(processor.getDeviceLossStats().replayedResourceCount, 20);
  assert.equal(processor.getDeviceLossStats().replayedBufferWriteCount, 2);
  assert.equal(processor.validationErrorCount, 0);
});

test('W104 destroyed resources are not replayed; re-created ids replay only their latest version', () => {
  const first = createMockDevice('first');
  const processor = newProcessor(first.device);
  processor.executeCommandBuffer(
    concat(
      sceneSetupPackets(),
      destroyPacket(IDS.bindGroup),
      destroyPacket(IDS.sampled),
      createBufferPacket(101, 0x0040 | 0x0008, 64),
      writeBufferPacket(101, 0, new Uint8Array([9, 9, 9, 9])),
      destroyPacket(101),
      // Buffer 100 is re-created with a new size: its old shadow contents must be dropped.
      createBufferPacket(IDS.buffer, 0x0040 | 0x0008, 512)
    )
  );

  const second = createMockDevice('second');
  const summary = recoverOnto(processor, second.device);
  // 10 scene resources - bind group - texture 3 = 8 (buffer 100 counted once, at its latest size).
  assert.deepEqual(summary, { replayedResources: 8, failedResources: 0, replayedBufferWrites: 0 });
  assert.equal(processor.bindGroups.has(IDS.bindGroup), false);
  assert.equal(processor.textures.has(IDS.sampled), false);
  assert.equal(processor.textures.get(IDS.rt).device, 'second');
  assert.equal(processor.buffers.has(101), false);
  assert.equal(processor.buffers.get(IDS.buffer).size, 512);
  assert.equal(second.ops('createBuffer').length, 1);
  assert.equal(second.ops('createBindGroup').length, 0);
  assert.equal(second.ops('writeBuffer').length, 0);
});

test('W104 shadow buffer cache respects per-buffer and total caps with LRU eviction', () => {
  const first = createMockDevice('first');
  const processor = newProcessor(first.device);

  // Default caps: a 4 MiB buffer is never shadowed, a 2 MiB buffer is.
  processor.executeCommandBuffer(
    concat(
      createBufferPacket(1, 0x0080 | 0x0008, 4 * 1024 * 1024),
      writeBufferPacket(1, 0, new Uint8Array(64).fill(1)),
      createBufferPacket(2, 0x0080 | 0x0008, 2 * 1024 * 1024),
      writeBufferPacket(2, 1024, new Uint8Array(64).fill(2)),
      // Not COPY_DST: never shadowed (its writes cannot be replayed).
      createBufferPacket(3, 0x0080, 64),
      writeBufferPacket(3, 0, new Uint8Array(4))
    )
  );
  let stats = processor.getResourceReplayStats();
  assert.equal(stats.shadowBufferCount, 1);
  assert.equal(stats.shadowBufferBytes, 2 * 1024 * 1024);
  assert.ok(stats.shadowBufferBytes <= 64 * 1024 * 1024);

  // Bounded total cap (configurable): LRU eviction keeps the most recently written buffers.
  const small = newProcessor(createMockDevice('small').device, {
    shadowBufferMaxBytes: 1024,
    shadowBufferTotalMaxBytes: 2048,
  });
  small.executeCommandBuffer(
    concat(
      createBufferPacket(10, 0x0040 | 0x0008, 1024),
      createBufferPacket(11, 0x0040 | 0x0008, 1024),
      createBufferPacket(12, 0x0040 | 0x0008, 1024),
      createBufferPacket(13, 0x0040 | 0x0008, 2048), // over the per-buffer cap
      writeBufferPacket(10, 0, new Uint8Array([10, 10, 10, 10])),
      writeBufferPacket(11, 0, new Uint8Array([11, 11, 11, 11])),
      writeBufferPacket(10, 4, new Uint8Array([10, 10, 10, 10])), // touch 10 -> most recent
      writeBufferPacket(12, 0, new Uint8Array([12, 12, 12, 12])), // evicts 11 (LRU)
      writeBufferPacket(13, 0, new Uint8Array([13, 13, 13, 13]))
    )
  );
  stats = small.getResourceReplayStats();
  assert.equal(stats.shadowBufferCount, 2);
  assert.equal(stats.shadowBufferBytes, 2048);
  assert.equal(stats.shadowBufferEvictions, 1);
  assert.deepEqual([...small._shadowBuffers.keys()].sort((a, b) => a - b), [10, 12]);

  const second = createMockDevice('second');
  const summary = recoverOnto(small, second.device);
  assert.equal(summary.replayedResources, 4);
  assert.equal(summary.replayedBufferWrites, 2);
  const writes = second.ops('writeBuffer');
  const byBuffer = new Map(writes.map((w) => [w.buffer, w.bytes]));
  assert.deepEqual(byBuffer.get(small.buffers.get(10)), new Uint8Array([10, 10, 10, 10, 10, 10, 10, 10]));
  assert.deepEqual(byBuffer.get(small.buffers.get(12)), new Uint8Array([12, 12, 12, 12]));
  assert.equal(small.getDeviceLossStats().replayedBufferWriteCount, 2);
});

test('W104 resources created while recovery is in flight are still replayed', () => {
  const first = createMockDevice('first');
  const processor = newProcessor(first.device);
  processor.executeCommandBuffer(createBufferPacket(1, 0x0040 | 0x0008, 64));

  processor.beginDeviceRecovery();
  // Dropped (not executed) but recorded for replay.
  processor.executeCommandBuffer(
    concat(createBufferPacket(2, 0x0040 | 0x0008, 64), writeBufferPacket(2, 0, new Uint8Array([7, 7, 7, 7])))
  );
  assert.equal(processor.buffers.has(2), false, 'dropped while recovering, not executed');
  assert.equal(processor.getDeviceLossStats().droppedWhileRecovering, 1);

  const second = createMockDevice('second');
  processor.rebindDevice(second.device, createMockContext(), { format: 'bgra8unorm' });
  installRenderStatePackets(processor);
  const summary = processor.replayRecordedResources();
  processor.endDeviceRecovery(true);
  assert.deepEqual(summary, { replayedResources: 2, failedResources: 0, replayedBufferWrites: 1 });
  assert.equal(processor.buffers.get(2).device, 'second');
});

test('W104 one bad recorded packet never aborts the rest of the replay', () => {
  const first = createMockDevice('first');
  const processor = newProcessor(first.device);
  processor.executeCommandBuffer(sceneSetupPackets());

  const second = createMockDevice('second');
  second.device.createSampler = () => {
    throw new Error('sampler creation exploded');
  };
  const summary = recoverOnto(processor, second.device);
  assert.equal(summary.failedResources >= 1, true);
  assert.equal(processor.textures.get(IDS.rt).device, 'second');
  assert.equal(processor.buffers.get(IDS.buffer).device, 'second');
  const pipelineEntry = processor.pipelines.get(IDS.pipeline);
  assert.ok(pipelineEntry, 'pipeline still replayed after a failed sampler');
  assert.equal(processor._replayingResources, false);
});

test('W104 replayed pipelines go through AsyncPipelineCache on async-capable devices', async () => {
  const first = createMockDevice('first', { asyncPipelines: true });
  const processor = new WgpuCommandProcessor(first.device, createMockContext(), {
    format: 'bgra8unorm',
    enableShaderFallback: false,
    logger: null,
  });
  installRenderStatePackets(processor);
  assert.ok(processor.asyncPipelineCache, 'async pipeline cache active');
  processor.executeCommandBuffer(sceneSetupPackets());
  await settle();

  const second = createMockDevice('second', { asyncPipelines: true });
  const summary = recoverOnto(processor, second.device);
  assert.equal(summary.failedResources, 0);
  await settle();
  assert.equal(processor.isAsyncPipelinePending(IDS.pipeline), false);
  const entry = processor.pipelines.get(IDS.pipeline);
  const pipeline = entry && entry.pipeline ? entry.pipeline : entry;
  assert.equal(pipeline.device, 'second');

  processor.executeCommandBuffer(drawFramePackets());
  assert.equal(processor.validationErrorCount, 0, JSON.stringify(processor.getValidationErrors()));
  assert.equal(second.passCalls.filter((c) => c.op === 'draw').length, 1);
});

test('W104 rebindDevice alone still clears stale handles (replay is explicit)', () => {
  const first = createMockDevice('first');
  const processor = newProcessor(first.device);
  processor.executeCommandBuffer(sceneSetupPackets());
  processor.beginDeviceRecovery();
  processor.rebindDevice(createMockDevice('second').device, createMockContext());
  processor.endDeviceRecovery(true);
  assert.equal(processor.buffers.size, 0);
  assert.equal(processor.textures.size, 0);
  assert.equal(processor.pipelines.size, 0);
  assert.equal(processor.bindGroups.size, 0);
  assert.equal(processor.getDeviceLossStats().replayedResourceCount, 0);
});

test('W104 unknown render-pass texture errors are reported once per texture id per device generation', () => {
  const first = createMockDevice('first');
  const processor = newProcessor(first.device);
  processor.executeCommandBuffer(sceneSetupPackets());
  for (let i = 0; i < 5; i += 1) {
    processor.executeCommandBuffer(drawFramePackets(77));
  }
  const unknown = processor
    .getValidationErrors()
    .filter((e) => /unknown texture 77/.test(e.message || (e.error && e.error.message) || JSON.stringify(e)));
  assert.equal(processor.validationErrorCount, 1);
  assert.equal(unknown.length, 1);
  assert.equal(processor.extension.getTelemetry().suppressedMissingTextureErrors, 4);

  // A different missing texture id is still reported.
  processor.executeCommandBuffer(drawFramePackets(78));
  assert.equal(processor.validationErrorCount, 2);
});

test('W104 telemetry deviceStats exposes replay counters with zero defaults', () => {
  const { device } = createMockDevice('first');
  const processor = newProcessor(device);
  const snap = buildGpuTelemetrySnapshot(processor, { timestamp: 1 });
  assert.equal(snap.deviceStats.replayedResourceCount, 0);
  assert.equal(snap.deviceStats.replayedBufferWriteCount, 0);
  assert.equal('deviceStats' in buildGpuTelemetrySnapshot(null, { timestamp: 1 }), false);

  processor.executeCommandBuffer(sceneSetupPackets());
  recoverOnto(processor, createMockDevice('second').device);
  const after = buildGpuTelemetrySnapshot(processor, { timestamp: 2 });
  assert.equal(after.deviceStats.replayedResourceCount, 10);
  assert.equal(after.deviceStats.replayedBufferWriteCount, 1);
});

test('W105 replayRecordedResourcesAsync replays in chunks without giant stream copy and draws cleanly', async () => {
  const first = createMockDevice('first');
  const processor = newProcessor(first.device);
  processor.executeCommandBuffer(sceneSetupPackets());

  const second = createMockDevice('second');
  processor.beginDeviceRecovery();
  processor.rebindDevice(second.device, createMockContext());
  installRenderStatePackets(processor);
  const summary = await processor.replayRecordedResourcesAsync({ chunkSize: 3 });
  processor.endDeviceRecovery(true);

  assert.equal(summary.aborted, false);
  assert.equal(summary.replayedResources, 10);
  assert.equal(summary.replayedBufferWrites, 1);
  assert.equal(summary.failedResources, 0);

  processor.executeCommandBuffer(drawFramePackets(2));
  assert.equal(processor.validationErrorCount, 0, JSON.stringify(processor.getValidationErrors()));
  assert.equal(second.passCalls.filter((c) => c.op === 'draw').length, 1);
});

test('W105 replayMaxBindGroups evicts oldest transient bind groups while keeping persistent resources', () => {
  const first = createMockDevice('first');
  const processor = newProcessor(first.device, { replayMaxBindGroups: 2 });
  const bglEntry = { binding: 0, visibility: 0x3, type: 0, subtype: 0 };
  const bgEntry = { binding: 0, type: 0, id: 10, offset: 0, size: 64 };
  processor.executeCommandBuffer(
    concat(
      createBufferPacket(10, 0x48, 64),
      bindGroupLayoutPacket(100, [bglEntry]),
      bindGroup2Packet(5001, 100, [bgEntry]),
      bindGroup2Packet(5002, 100, [bgEntry]),
      bindGroup2Packet(5003, 100, [bgEntry])
    )
  );

  const stats = processor.getResourceReplayStats();
  assert.equal(stats.replayBindGroupEvictions, 1);

  const second = createMockDevice('second');
  recoverOnto(processor, second.device);
  assert.equal(processor.buffers.has(10), true);
  assert.equal(processor.bindGroups.has(5001), false);
  assert.equal(processor.bindGroups.has(5002), true);
  assert.equal(processor.bindGroups.has(5003), true);
});
