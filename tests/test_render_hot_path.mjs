import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const rs = require('../web/workers/render_state_packets.js');
const { WgpuCommandProcessor, OPCODES } = require('../web/workers/wgpu_worker.js');
const { DrawExecutor } = require('../web/workers/draw_executor.js');

const {
  installRenderStatePackets,
  V2_OPCODES,
  SWAPCHAIN_TEXTURE_ID,
  TEXTURE_FORMAT_NAMES,
} = rs;

function formatCodeOf(name) {
  const idx = TEXTURE_FORMAT_NAMES.indexOf(name);
  if (idx < 0) throw new Error(`unknown format ${name}`);
  return idx;
}

class PacketBuilder {
  constructor() {
    this.chunks = [];
  }
  packet(opcode, payload = new Uint8Array(0)) {
    const out = new Uint8Array(8 + payload.byteLength);
    const dv = new DataView(out.buffer);
    dv.setUint32(0, opcode, true);
    dv.setUint32(4, payload.byteLength, true);
    out.set(payload, 8);
    this.chunks.push(out);
    return this;
  }
  u32s(opcode, values) {
    const payload = new Uint8Array(values.length * 4);
    const dv = new DataView(payload.buffer);
    values.forEach((v, i) => dv.setUint32(i * 4, v >>> 0, true));
    return this.packet(opcode, payload);
  }
  bytes() {
    const total = this.chunks.reduce((n, c) => n + c.byteLength, 0);
    const out = new Uint8Array(total);
    let o = 0;
    for (const c of this.chunks) {
      out.set(c, o);
      o += c.byteLength;
    }
    return out;
  }
}

function concat(...arrays) {
  const total = arrays.reduce((n, a) => n + a.byteLength, 0);
  const out = new Uint8Array(total);
  let o = 0;
  for (const a of arrays) {
    out.set(a, o);
    o += a.byteLength;
  }
  return out;
}

function viewDescBytes({
  format = 0,
  dimension = 0,
  aspect = 0,
  baseMip = 0,
  mipCount = 0,
  baseLayer = 0,
  layerCount = 0,
} = {}) {
  const b = new Uint8Array(16);
  const dv = new DataView(b.buffer);
  dv.setUint8(0, format);
  dv.setUint8(1, dimension);
  dv.setUint8(2, aspect);
  dv.setUint16(4, baseMip, true);
  dv.setUint16(6, mipCount, true);
  dv.setUint32(8, baseLayer, true);
  dv.setUint32(12, layerCount, true);
  return b;
}

function beginRenderPassPacket({ colors = [], depthStencil = null } = {}) {
  const payload = new Uint8Array(368);
  const dv = new DataView(payload.buffer);
  dv.setUint32(0, colors.length, true);
  for (let i = 0; i < colors.length; i += 1) {
    const c = colors[i];
    if (!c) continue;
    const o = 8 + i * 40;
    dv.setUint32(o, c.textureId >>> 0, true);
    dv.setUint8(o + 4, c.loadOp ?? 2);
    dv.setUint8(o + 5, c.storeOp ?? 1);
    if (c.view) payload.set(c.view, o + 8);
    const cv = c.clearValue || [0, 0, 0, 1];
    dv.setFloat32(o + 24, cv[0], true);
    dv.setFloat32(o + 28, cv[1], true);
    dv.setFloat32(o + 32, cv[2], true);
    dv.setFloat32(o + 36, cv[3], true);
  }
  if (depthStencil) {
    const d = 328;
    dv.setUint32(d, depthStencil.textureId >>> 0, true);
    dv.setUint8(d + 4, depthStencil.depthLoadOp ?? 2);
    dv.setUint8(d + 5, depthStencil.depthStoreOp ?? 1);
    dv.setUint8(d + 6, depthStencil.stencilLoadOp ?? 0);
    dv.setUint8(d + 7, depthStencil.stencilStoreOp ?? 0);
    if (depthStencil.view) payload.set(depthStencil.view, d + 8);
    dv.setFloat32(d + 24, depthStencil.depthClearValue ?? 1, true);
    dv.setUint32(d + 28, depthStencil.stencilClearValue ?? 0, true);
    dv.setUint8(d + 32, depthStencil.depthReadOnly ? 1 : 0);
    dv.setUint8(d + 33, depthStencil.stencilReadOnly ? 1 : 0);
  }
  return new PacketBuilder().packet(V2_OPCODES.BEGIN_RENDER_PASS, payload).bytes();
}

function setViewportPacket(x, y, width, height, minDepth = 0, maxDepth = 1) {
  const p = new Uint8Array(24);
  const dv = new DataView(p.buffer);
  dv.setFloat32(0, x, true);
  dv.setFloat32(4, y, true);
  dv.setFloat32(8, width, true);
  dv.setFloat32(12, height, true);
  dv.setFloat32(16, minDepth, true);
  dv.setFloat32(20, maxDepth, true);
  return new PacketBuilder().packet(V2_OPCODES.SET_VIEWPORT, p).bytes();
}

function setScissorPacket(x, y, width, height) {
  const p = new Uint8Array(16);
  const dv = new DataView(p.buffer);
  dv.setUint32(0, x >>> 0, true);
  dv.setUint32(4, y >>> 0, true);
  dv.setUint32(8, width >>> 0, true);
  dv.setUint32(12, height >>> 0, true);
  return new PacketBuilder().packet(V2_OPCODES.SET_SCISSOR_RECT, p).bytes();
}

function setBlendConstantPacket(r, g, b, a) {
  const p = new Uint8Array(16);
  const dv = new DataView(p.buffer);
  dv.setFloat32(0, r, true);
  dv.setFloat32(4, g, true);
  dv.setFloat32(8, b, true);
  dv.setFloat32(12, a, true);
  return new PacketBuilder().packet(V2_OPCODES.SET_BLEND_CONSTANT, p).bytes();
}

function setStencilRefPacket(ref) {
  return new PacketBuilder().u32s(V2_OPCODES.SET_STENCIL_REFERENCE, [ref]).bytes();
}

function bindGroupLayoutPacket(id, entries) {
  const payload = new Uint8Array(24 + entries.length * 24);
  const dv = new DataView(payload.buffer);
  dv.setUint32(0, id, true);
  dv.setUint32(4, entries.length, true);
  entries.forEach((e, i) => {
    const o = 24 + i * 24;
    dv.setUint32(o, e.binding, true);
    dv.setUint32(o + 4, e.visibility ?? 7, true);
    dv.setUint8(o + 8, e.type);
    dv.setUint8(o + 9, e.subtype ?? 0);
    dv.setUint8(o + 10, e.dimension ?? 2);
    dv.setUint8(o + 11, e.flags ?? 0);
    dv.setUint8(o + 12, e.storageFormat ?? 0);
    dv.setBigUint64(o + 16, BigInt(e.minBindingSize ?? 0), true);
  });
  return new PacketBuilder().packet(V2_OPCODES.CREATE_BIND_GROUP_LAYOUT, payload).bytes();
}

function bindGroup2Packet(id, layoutId, entries) {
  const payload = new Uint8Array(16 + entries.length * 48);
  const dv = new DataView(payload.buffer);
  dv.setUint32(0, id, true);
  dv.setUint32(4, layoutId, true);
  dv.setUint32(8, entries.length, true);
  entries.forEach((e, i) => {
    const o = 16 + i * 48;
    dv.setUint32(o, e.binding, true);
    dv.setUint32(o + 4, e.type, true);
    dv.setUint32(o + 8, e.id, true);
    if (e.offset !== undefined) dv.setBigUint64(o + 16, BigInt(e.offset), true);
    if (e.size !== undefined) dv.setBigUint64(o + 24, BigInt(e.size), true);
    if (e.view) payload.set(e.view, o + 32);
  });
  return new PacketBuilder().packet(V2_OPCODES.CREATE_BIND_GROUP2, payload).bytes();
}

function setBindGroup2Packet(index, id, offsets = []) {
  const payload = new Uint8Array(16 + offsets.length * 4);
  const dv = new DataView(payload.buffer);
  dv.setUint32(0, index, true);
  dv.setUint32(4, id, true);
  dv.setUint32(8, offsets.length, true);
  offsets.forEach((off, i) => dv.setUint32(16 + i * 4, off >>> 0, true));
  return new PacketBuilder().packet(V2_OPCODES.SET_BIND_GROUP2, payload).bytes();
}

function createTexture2Packet({
  id,
  width = 256,
  height = 256,
  depth = 1,
  mips = 1,
  samples = 1,
  usage = 0x14,
  dimension = 1,
  format = formatCodeOf('rgba8unorm'),
  viewFormats = [],
}) {
  const p = new Uint8Array(40);
  const dv = new DataView(p.buffer);
  dv.setUint32(0, id, true);
  dv.setUint32(4, width, true);
  dv.setUint32(8, height, true);
  dv.setUint32(12, depth, true);
  dv.setUint32(16, mips, true);
  dv.setUint32(20, samples, true);
  dv.setUint32(24, usage, true);
  dv.setUint8(28, dimension);
  dv.setUint8(29, format);
  dv.setUint8(30, viewFormats.length);
  viewFormats.forEach((vf, i) => dv.setUint8(32 + i, vf));
  return new PacketBuilder().packet(V2_OPCODES.CREATE_TEXTURE2, p).bytes();
}

function createMockWebGpu() {
  let nextHandle = 1;
  const calls = [];
  const passCalls = [];
  const makeSwapchainTex = () => {
    const tex = {
      kind: 'GPUTexture',
      handle: nextHandle++,
      width: 1280,
      height: 720,
      dimension: '2d',
      format: 'bgra8unorm',
      destroyed: false,
      createViewCalls: 0,
      createView(viewDesc) {
        tex.createViewCalls += 1;
        const view = {
          kind: 'GPUTextureView',
          handle: nextHandle++,
          texture: tex,
          textureHandle: tex.handle,
          viewDesc: viewDesc ? { ...viewDesc } : undefined,
          destroyed: false,
        };
        calls.push({ op: 'swapchain.createView', texture: tex, desc: viewDesc, result: view });
        return view;
      },
      destroy() {
        tex.destroyed = true;
      },
    };
    return tex;
  };
  let currentSwapchainTex = makeSwapchainTex();
  const makePass = (kind) => {
    const record = (op, args) => passCalls.push({ pass: kind, op, args });
    return {
      kind,
      setPipeline: (p) => record('setPipeline', [p]),
      setBindGroup: (i, bg, offsets) =>
        record('setBindGroup', offsets === undefined ? [i, bg] : [i, bg, offsets]),
      setVertexBuffer: (slot, buf, off) => record('setVertexBuffer', [slot, buf, off]),
      setIndexBuffer: (buf, fmt, off) => record('setIndexBuffer', [buf, fmt, off]),
      setViewport: (...a) => record('setViewport', a),
      setScissorRect: (...a) => record('setScissorRect', a),
      setBlendConstant: (c) => record('setBlendConstant', [c]),
      setStencilReference: (r) => record('setStencilReference', [r]),
      draw: (...a) => record('draw', a),
      drawIndexed: (...a) => record('drawIndexed', a),
      dispatchWorkgroups: (...a) => record('dispatchWorkgroups', a),
      end: () => record('end', []),
    };
  };

  const device = {
    features: new Set(),
    limits: { maxComputeWorkgroupsPerDimension: 65535 },
    createBuffer(desc) {
      const buf = {
        kind: 'GPUBuffer',
        handle: nextHandle++,
        size: desc.size,
        usage: desc.usage,
        desc,
        destroyed: false,
        destroy() {
          buf.destroyed = true;
        },
      };
      calls.push({ op: 'createBuffer', desc, result: buf });
      return buf;
    },
    createTexture(desc) {
      const tex = {
        kind: 'GPUTexture',
        handle: nextHandle++,
        width: desc.size ? desc.size.width : 1,
        height: desc.size ? desc.size.height : 1,
        depthOrArrayLayers: desc.size ? desc.size.depthOrArrayLayers || 1 : 1,
        mipLevelCount: desc.mipLevelCount || 1,
        dimension: desc.dimension || '2d',
        format: desc.format || 'rgba8unorm',
        usage: desc.usage || 0,
        desc,
        destroyed: false,
        createViewCalls: 0,
        createView(viewDesc) {
          tex.createViewCalls += 1;
          const view = {
            kind: 'GPUTextureView',
            handle: nextHandle++,
            texture: tex,
            textureHandle: tex.handle,
            viewDesc: viewDesc ? { ...viewDesc } : undefined,
            destroyed: false,
          };
          calls.push({ op: 'texture.createView', texture: tex, desc: viewDesc, result: view });
          return view;
        },
        destroy() {
          tex.destroyed = true;
        },
      };
      calls.push({ op: 'createTexture', desc, result: tex });
      return tex;
    },
    createSampler(desc) {
      const s = { kind: 'GPUSampler', handle: nextHandle++, desc, destroyed: false };
      calls.push({ op: 'createSampler', desc, result: s });
      return s;
    },
    createShaderModule(desc) {
      return { kind: 'GPUShaderModule', handle: nextHandle++, desc };
    },
    createBindGroupLayout(desc) {
      const l = { kind: 'GPUBindGroupLayout', handle: nextHandle++, desc };
      calls.push({ op: 'createBindGroupLayout', desc, result: l });
      return l;
    },
    createPipelineLayout(desc) {
      return { kind: 'GPUPipelineLayout', handle: nextHandle++, desc };
    },
    createBindGroup(desc) {
      const bg = { kind: 'GPUBindGroup', handle: nextHandle++, desc, destroyed: false };
      calls.push({ op: 'createBindGroup', desc, result: bg });
      return bg;
    },
    createRenderPipeline(desc) {
      const layout0 = { kind: 'GPUBindGroupLayout', handle: nextHandle++, auto: true };
      const pipe = {
        kind: 'GPURenderPipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayoutCalls: 0,
        getBindGroupLayout(idx) {
          pipe.getBindGroupLayoutCalls += 1;
          return idx === 0 ? layout0 : { kind: 'GPUBindGroupLayout', handle: nextHandle++, idx };
        },
      };
      calls.push({ op: 'createRenderPipeline', desc, result: pipe });
      return pipe;
    },
    createCommandEncoder() {
      return {
        beginRenderPass(desc) {
          passCalls.push({ pass: 'render', op: 'beginRenderPass', args: [desc] });
          return makePass('render');
        },
        beginComputePass() {
          passCalls.push({ pass: 'compute', op: 'beginComputePass', args: [] });
          return makePass('compute');
        },
        copyBufferToBuffer() {},
        finish() {
          return { kind: 'GPUCommandBuffer', handle: nextHandle++ };
        },
      };
    },
    queue: {
      writeBuffer() {},
      writeTexture() {},
      submit(cmds) {
        calls.push({ op: 'submit', cmds });
      },
    },
  };

  const ctx = {
    canvas: { width: 1280, height: 720 },
    configure() {},
    getCurrentTexture() {
      return currentSwapchainTex;
    },
  };

  return {
    device,
    ctx,
    calls,
    passCalls,
    rotateSwapchainTexture() {
      currentSwapchainTex = makeSwapchainTex();
      return currentSwapchainTex;
    },
    get currentSwapchainTex() {
      return currentSwapchainTex;
    },
    results(op) {
      return calls.filter((c) => c.op === op);
    },
  };
}

function makeStrictW347Texture({
  handle = 1,
  width = 4096,
  height = 1,
  depthOrArrayLayers = 1,
  mipLevelCount = 1,
  format = 'rgba8unorm',
  dimension = '2d',
  promoted1DTo2D = false,
  usage = 0x04 | 0x08,
} = {}) {
  const viewCalls = [];
  const tex = {
    kind: 'GPUTexture',
    handle,
    width,
    height,
    depthOrArrayLayers,
    mipLevelCount,
    format,
    dimension,
    promoted1DTo2D,
    usage,
    destroyed: false,
    viewCalls,
    createView(vd) {
      const descCopy = vd ? { ...vd } : undefined;
      viewCalls.push(descCopy);
      const requestedDim = descCopy && descCopy.dimension ? descCopy.dimension : dimension;
      if (dimension === '2d' && requestedDim === '1d') {
        throw new Error(
          `Dawn validation error: TextureViewDimension::e1D incompatible with TextureDimension::e2D (${width}x${height})`
        );
      }
      if (dimension === '1d' && requestedDim !== '1d') {
        throw new Error(
          `Dawn validation error: ${requestedDim} view incompatible with TextureDimension::e1D`
        );
      }
      if (descCopy) {
        const baseLayer = descCopy.baseArrayLayer ?? 0;
        const layerCount = descCopy.arrayLayerCount ?? 1;
        if (baseLayer < 0 || layerCount < 1 || baseLayer + layerCount > depthOrArrayLayers) {
          throw new Error(`Invalid array layer range [${baseLayer}, ${baseLayer + layerCount})`);
        }
        const baseMip = descCopy.baseMipLevel ?? 0;
        const mipCount = descCopy.mipLevelCount ?? mipLevelCount - baseMip;
        if (baseMip < 0 || mipCount < 1 || baseMip + mipCount > mipLevelCount) {
          throw new Error(`Invalid mip range [${baseMip}, ${baseMip + mipCount})`);
        }
      }
      return {
        kind: 'GPUTextureView',
        texture: tex,
        textureHandle: handle,
        viewDesc: descCopy,
        destroyed: false,
      };
    },
    destroy() {
      tex.destroyed = true;
    },
  };
  return tex;
}

// =============================================================================
// 1. Redundant state-call elision inside a pass + reset on new pass
// =============================================================================

test('W354: RenderStatePackets elides redundant viewport, scissor, blendConstant, stencilRef, and bindGroup2 inside a pass and resets on new pass', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'bgra8unorm' });
  const handler = installRenderStatePackets(processor);

  // Set up buffer + BGL + BG for SetBindGroup2 dynamic-offset testing.
  const buf = gpu.device.createBuffer({ size: 1024, usage: 0x0040 | 0x0008 });
  processor.buffers.set(10, buf);
  const setup = concat(
    bindGroupLayoutPacket(1, [
      { binding: 0, visibility: 7, type: 0, subtype: 0, flags: 1, minBindingSize: 256 },
    ]),
    bindGroup2Packet(20, 1, [{ binding: 0, type: 0, id: 10, offset: 0, size: 256 }])
  );
  processor.executeCommandBuffer(setup);
  assert.equal(handler.stats.errors, 0, handler.lastError);

  const frameStream = concat(
    new PacketBuilder().packet(OPCODES.BEGIN_FRAME).bytes(),
    // Pass 1: emit each state, repeat identically (must elide), mutate one field (must emit), repeat (must elide).
    beginRenderPassPacket({ colors: [{ textureId: SWAPCHAIN_TEXTURE_ID }] }),
    setViewportPacket(0, 0, 1280, 720, 0, 1),
    setViewportPacket(0, 0, 1280, 720, 0, 1), // elided
    setViewportPacket(0, 0, 1280, 720, 0, 1), // elided
    setViewportPacket(10, 20, 640, 360, 0, 1), // changed -> emitted
    setViewportPacket(10, 20, 640, 360, 0, 1), // elided
    setScissorPacket(0, 0, 1280, 720),
    setScissorPacket(0, 0, 1280, 720), // elided
    setScissorPacket(8, 8, 600, 300), // changed -> emitted
    setScissorPacket(8, 8, 600, 300), // elided
    setBlendConstantPacket(0.5, 0.25, 0.75, 1.0),
    setBlendConstantPacket(0.5, 0.25, 0.75, 1.0), // elided
    setBlendConstantPacket(0.5, 0.25, 0.75, 0.5), // alpha changed -> emitted
    setStencilRefPacket(42),
    setStencilRefPacket(42), // elided
    setStencilRefPacket(99), // changed -> emitted
    setStencilRefPacket(99), // elided
    setBindGroup2Packet(0, 20, [0]),
    setBindGroup2Packet(0, 20, [0]), // identical dynamic offset -> elided
    setBindGroup2Packet(0, 20, [256]), // dynamic offset changed -> emitted
    setBindGroup2Packet(0, 20, [256]), // elided
    new PacketBuilder().packet(V2_OPCODES.END_RENDER_PASS).bytes(),
    // Pass 2: brand new pass MUST reset tracked state and re-emit identical values!
    beginRenderPassPacket({ colors: [{ textureId: SWAPCHAIN_TEXTURE_ID }] }),
    setViewportPacket(10, 20, 640, 360, 0, 1), // must emit on new pass
    setScissorPacket(8, 8, 600, 300), // must emit on new pass
    setBlendConstantPacket(0.5, 0.25, 0.75, 0.5), // must emit on new pass
    setStencilRefPacket(99), // must emit on new pass
    setBindGroup2Packet(0, 20, [256]), // must emit on new pass
    setBindGroup2Packet(0, 20, [256]), // redundant inside pass 2 -> elided
    new PacketBuilder().packet(V2_OPCODES.END_RENDER_PASS).bytes(),
    new PacketBuilder().packet(OPCODES.PRESENT).bytes()
  );

  const res = processor.executeCommandBuffer(frameStream);
  assert.equal(res.presents, 1);
  assert.equal(handler.stats.errors, 0, handler.lastError);

  assert.equal(gpu.passCalls.filter((c) => c.op === 'setViewport').length, 3);
  assert.equal(gpu.passCalls.filter((c) => c.op === 'setScissorRect').length, 3);
  assert.equal(gpu.passCalls.filter((c) => c.op === 'setBlendConstant').length, 3);
  assert.equal(gpu.passCalls.filter((c) => c.op === 'setStencilReference').length, 3);
  assert.equal(gpu.passCalls.filter((c) => c.op === 'setBindGroup').length, 3);

  const stats = handler.getStats();
  assert.equal(stats.elidedViewportSets, 3);
  assert.equal(stats.elidedScissorSets, 2);
  assert.equal(stats.elidedBlendConstantSets, 1);
  assert.equal(stats.elidedStencilRefSets, 2);
  assert.equal(stats.elidedBindGroupSets, 3);
});

test('W354: RenderStatePackets elideRedundantState=false disables pass state elision', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'bgra8unorm' });
  const handler = installRenderStatePackets(processor, { elideRedundantState: false });

  const stream = concat(
    new PacketBuilder().packet(OPCODES.BEGIN_FRAME).bytes(),
    beginRenderPassPacket({ colors: [{ textureId: SWAPCHAIN_TEXTURE_ID }] }),
    setViewportPacket(0, 0, 1280, 720, 0, 1),
    setViewportPacket(0, 0, 1280, 720, 0, 1),
    setScissorPacket(0, 0, 1280, 720),
    setScissorPacket(0, 0, 1280, 720),
    setStencilRefPacket(7),
    setStencilRefPacket(7),
    new PacketBuilder().packet(V2_OPCODES.END_RENDER_PASS).bytes(),
    new PacketBuilder().packet(OPCODES.PRESENT).bytes()
  );
  processor.executeCommandBuffer(stream);
  assert.equal(gpu.passCalls.filter((c) => c.op === 'setViewport').length, 2);
  assert.equal(gpu.passCalls.filter((c) => c.op === 'setScissorRect').length, 2);
  assert.equal(gpu.passCalls.filter((c) => c.op === 'setStencilReference').length, 2);
  assert.equal(handler.getStats().elidedViewportSets, 0);
  assert.equal(handler.getStats().elidedScissorSets, 0);
  assert.equal(handler.getStats().elidedStencilRefSets, 0);
});

test('W354: DrawExecutor elides redundant state calls inside one pass and resets completely across passes', () => {
  const gpu = createMockWebGpu();
  const executor = new DrawExecutor({ device: gpu.device, reuseDrawResult: true });
  const pipe = { kind: 'GPURenderPipeline', handle: 1, getBindGroupLayout: () => ({}) };
  const vb = gpu.device.createBuffer({ size: 256, usage: 0x0020 });
  const ib = gpu.device.createBuffer({ size: 64, usage: 0x0010 });
  const bg = { kind: 'GPUBindGroup', handle: 2, destroyed: false };

  const pass1 = gpu.device.createCommandEncoder().beginRenderPass({ colorAttachments: [] });
  executor.setPipeline(pipe);
  executor.setVertexBuffer(0, vb, 0);
  executor.setIndexBuffer(ib, 'uint16', 0);
  executor.setBindGroup(0, bg, [0]);
  executor.setViewport(0, 0, 1280, 720, 0, 1);
  executor.setScissorRect(0, 0, 1280, 720);
  executor.setBlendConstant(0.1, 0.2, 0.3, 1.0);
  executor.setStencilReference(5);

  for (let i = 0; i < 10; i += 1) {
    const r = executor.drawIndexed(pass1, 6, 1, 0, 0, 0);
    assert.equal(r.executed, true);
    assert.equal(r.pipelineElided, i > 0);
    assert.equal(r.indexBufferElided, i > 0);
    assert.equal(r.viewportElided, i > 0);
    assert.equal(r.scissorElided, i > 0);
    assert.equal(r.blendConstantElided, i > 0);
    assert.equal(r.stencilReferenceElided, i > 0);
    assert.equal(r.vertexBuffersElided, i > 0 ? 1 : 0);
    assert.equal(r.bindGroupsElided, i > 0 ? 1 : 0);
  }
  assert.equal(gpu.passCalls.filter((c) => c.op === 'setPipeline').length, 1);
  assert.equal(gpu.passCalls.filter((c) => c.op === 'setBindGroup').length, 1);
  assert.equal(gpu.passCalls.filter((c) => c.op === 'setVertexBuffer').length, 1);
  assert.equal(gpu.passCalls.filter((c) => c.op === 'setIndexBuffer').length, 1);
  assert.equal(gpu.passCalls.filter((c) => c.op === 'setViewport').length, 1);
  assert.equal(gpu.passCalls.filter((c) => c.op === 'setScissorRect').length, 1);
  assert.equal(gpu.passCalls.filter((c) => c.op === 'setBlendConstant').length, 1);
  assert.equal(gpu.passCalls.filter((c) => c.op === 'setStencilReference').length, 1);

  // Switch to pass 2 (implicit pass-object switch resets all bound pass state).
  const pass2 = gpu.device.createCommandEncoder().beginRenderPass({ colorAttachments: [] });
  const r2 = executor.drawIndexed(pass2, 6, 1, 0, 0, 0);
  assert.equal(r2.executed, true);
  assert.equal(r2.pipelineElided, false);
  assert.equal(r2.indexBufferElided, false);
  assert.equal(r2.viewportElided, false);
  assert.equal(r2.scissorElided, false);
  assert.equal(r2.blendConstantElided, false);
  assert.equal(r2.stencilReferenceElided, false);
  assert.equal(gpu.passCalls.filter((c) => c.op === 'setPipeline').length, 2);
  assert.equal(gpu.passCalls.filter((c) => c.op === 'setBindGroup').length, 2);
  assert.equal(gpu.passCalls.filter((c) => c.op === 'setViewport').length, 2);
  assert.equal(gpu.passCalls.filter((c) => c.op === 'setScissorRect').length, 2);
});

// =============================================================================
// 2. GPUTextureView WeakMap caching & invalidation on replace / destroy
// =============================================================================

test('W354: RenderStatePackets WeakMap texture-view cache avoids duplicate createView(), caches swapchain format views, and invalidates on replace/destroy', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'bgra8unorm' });
  const handler = installRenderStatePackets(processor);

  const texA = gpu.device.createTexture({
    size: { width: 512, height: 512, depthOrArrayLayers: 1 },
    format: 'rgba8unorm',
    usage: 0x14,
  });
  processor.textures.set(50, texA);

  // Repeated default & format views hit WeakMap cache with 1 createView() per unique desc.
  const def1 = handler._textureView(50, {});
  const def2 = handler._textureView(50, {});
  const def3 = handler._textureView(50, null);
  assert.equal(def1, def2);
  assert.equal(def1, def3);
  assert.equal(texA.createViewCalls, 1);

  const fmt1 = handler._textureView(50, { format: 'rgba8unorm-srgb', dimension: '2d' });
  const fmt2 = handler._textureView(50, { format: 'rgba8unorm-srgb', dimension: '2d' });
  assert.equal(fmt1, fmt2);
  assert.notEqual(fmt1, def1);
  assert.equal(texA.createViewCalls, 2);
  assert.ok(handler.getStats().textureViewCacheHits >= 3);

  // Marking a cached view destroyed forces recreation cleanly.
  fmt1.destroyed = true;
  const fmt3 = handler._textureView(50, { format: 'rgba8unorm-srgb', dimension: '2d' });
  assert.notEqual(fmt3, fmt1);
  assert.equal(texA.createViewCalls, 3);

  // Marking the texture destroyed immediately returns null (never returns stale view).
  texA.destroyed = true;
  assert.equal(handler._textureView(50, {}), null);
  assert.equal(handler.getOrCreateTextureView(texA, {}), null);

  // Replacing texture at slot 50 gets a brand-new view cleanly.
  const texB = gpu.device.createTexture({
    size: { width: 512, height: 512, depthOrArrayLayers: 1 },
    format: 'rgba8unorm',
    usage: 0x14,
  });
  processor.textures.set(50, texB);
  const newView = handler._textureView(50, {});
  assert.ok(newView);
  assert.equal(newView.textureHandle, texB.handle);
  assert.equal(texB.createViewCalls, 1);

  // Swapchain format view caching across multiple passes in the same frame!
  processor._swapchainViewFormat = 'bgra8unorm';
  const srgbCode = formatCodeOf('bgra8unorm-srgb');
  const frameTex1 = gpu.rotateSwapchainTexture();
  const multiPassFrame = concat(
    new PacketBuilder().packet(OPCODES.BEGIN_FRAME).bytes(),
    beginRenderPassPacket({
      colors: [{ textureId: SWAPCHAIN_TEXTURE_ID, view: viewDescBytes({ format: srgbCode }) }],
    }),
    new PacketBuilder().packet(V2_OPCODES.END_RENDER_PASS).bytes(),
    beginRenderPassPacket({
      colors: [{ textureId: SWAPCHAIN_TEXTURE_ID, view: viewDescBytes({ format: srgbCode }) }],
    }),
    new PacketBuilder().packet(V2_OPCODES.END_RENDER_PASS).bytes(),
    beginRenderPassPacket({
      colors: [{ textureId: SWAPCHAIN_TEXTURE_ID, view: viewDescBytes({ format: srgbCode }) }],
    }),
    new PacketBuilder().packet(V2_OPCODES.END_RENDER_PASS).bytes(),
    new PacketBuilder().packet(OPCODES.PRESENT).bytes()
  );
  processor.executeCommandBuffer(multiPassFrame);
  assert.equal(handler.stats.errors, 0, handler.lastError);
  // Swapchain texture gets createView() for default swapchainView + ONLY ONE sRGB view across all 3 passes!
  const srgbViewsFrame1 = gpu
    .results('swapchain.createView')
    .filter((c) => c.texture === frameTex1 && c.desc && c.desc.format === 'bgra8unorm-srgb');
  assert.equal(srgbViewsFrame1.length, 1);

  // Frame 2 with a new swapchainTexture creates a fresh sRGB view for the new frame texture.
  const frameTex2 = gpu.rotateSwapchainTexture();
  processor.executeCommandBuffer(multiPassFrame);
  const srgbViewsFrame2 = gpu
    .results('swapchain.createView')
    .filter((c) => c.texture === frameTex2 && c.desc && c.desc.format === 'bgra8unorm-srgb');
  assert.equal(srgbViewsFrame2.length, 1);
  assert.notEqual(srgbViewsFrame2[0].result, srgbViewsFrame1[0].result);
});

// =============================================================================
// 3. W347 1D -> 2D view coercion honored on cached views (both modules)
// =============================================================================

test('W354 + W347: RenderStatePackets and DrawExecutor honor 1d -> 2d promoted LUT coercion AND cache coerced views', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'rgba8unorm' });
  const handler = installRenderStatePackets(processor);
  const executor = new DrawExecutor({ device: gpu.device });

  // Promoted 4096x1 2D LUT texture with strict Dawn validator that throws on dimension:'1d'.
  const lutTex = makeStrictW347Texture({
    handle: 7001,
    width: 4096,
    height: 1,
    depthOrArrayLayers: 1,
    mipLevelCount: 1,
    format: 'rgba8unorm',
    dimension: '2d',
    promoted1DTo2D: true,
  });
  processor.textures.set(701, lutTex);
  handler.textureMeta.set(701, {
    id: 701,
    requestedFormat: 'rgba8unorm',
    format: 'rgba8unorm',
    dimension: '2d',
    promoted1DTo2D: true,
    mipLevelCount: 1,
    usage: 0x04 | 0x08,
    width: 4096,
    height: 1,
    depthOrArrayLayers: 1,
  });

  // 1. RenderStatePackets: repeated '1d' view descriptors coerce to '2d' and hit WeakMap cache!
  const rsV1 = handler._textureView(701, { format: 'rgba8unorm', dimension: '1d' });
  const rsV2 = handler._textureView(701, { format: 'rgba8unorm', dimension: '1d', baseArrayLayer: 3, arrayLayerCount: 5 });
  const rsV3 = handler._textureView(701, {
    format: 'rgba8unorm',
    dimension: '2d',
    baseArrayLayer: 0,
    arrayLayerCount: 1,
    mipLevelCount: 1,
  });
  assert.ok(rsV1);
  assert.equal(rsV1.viewDesc.dimension, '2d');
  assert.equal(rsV1.viewDesc.baseArrayLayer, 0);
  assert.equal(rsV1.viewDesc.arrayLayerCount, 1);
  assert.equal(rsV1.viewDesc.mipLevelCount, 1);
  assert.equal(rsV2, rsV1, 'repeated coerced 1d requests must hit WeakMap cache');
  assert.equal(rsV3, rsV1, 'explicit equivalent 2d request shares coerced cached view');
  assert.equal(lutTex.viewCalls.length, 1);

  // 2. DrawExecutor: getOrCreateTextureView & getOrCreateBindGroupForPipeline coerce 1d -> 2d and cache!
  const lutTexExec = makeStrictW347Texture({
    handle: 7002,
    width: 4096,
    height: 1,
    depthOrArrayLayers: 1,
    mipLevelCount: 1,
    format: 'r32float',
    dimension: '2d',
    promoted1DTo2D: true,
  });
  const exV1 = executor.getOrCreateTextureView(lutTexExec, { format: 'r32float', dimension: '1d' });
  const exV2 = executor.getOrCreateTextureView(lutTexExec, {
    format: 'r32float',
    dimension: '1d',
    baseMipLevel: 2,
    mipLevelCount: 4,
    baseArrayLayer: 1,
    arrayLayerCount: 2,
  });
  const exV3 = executor.getOrCreateTextureView(lutTexExec, {
    format: 'r32float',
    dimension: '2d',
    baseMipLevel: 0,
    mipLevelCount: 1,
    baseArrayLayer: 0,
    arrayLayerCount: 1,
  });
  assert.ok(exV1);
  assert.equal(exV1.viewDesc.dimension, '2d');
  assert.equal(exV1.viewDesc.baseArrayLayer, 0);
  assert.equal(exV1.viewDesc.arrayLayerCount, 1);
  assert.equal(exV1.viewDesc.mipLevelCount, 1);
  assert.equal(exV2, exV1, 'coerced 1d view descriptor hits WeakMap cache with zero extra createView');
  assert.equal(exV3, exV1, 'explicit equivalent 2d descriptor shares coerced cached view');
  assert.equal(lutTexExec.viewCalls.length, 1);

  // Bind group creation with { texture: lutTexExec, viewDescriptor: { format: 'r32float', dimension: '1d' } }
  const pipe = gpu.device.createRenderPipeline({});
  const bg1 = executor.getOrCreateBindGroupForPipeline(pipe, 0, [
    { binding: 0, resource: { texture: lutTexExec, viewDescriptor: { format: 'r32float', dimension: '1d' } } },
  ]);
  const bg2 = executor.getOrCreateBindGroupForPipeline(pipe, 0, [
    { binding: 0, resource: { texture: lutTexExec, viewDescriptor: { format: 'r32float', dimension: '1d' } } },
  ]);
  assert.ok(bg1);
  assert.equal(bg2, bg1, 'bind group with coerced 1d->2d LUT view hits O(1) cache');
  assert.equal(bg1.desc.entries[0].resource, exV1);

  // 3. Reverse guard on real 1D texture in DrawExecutor: coerces '2d' -> '1d' and caches!
  const real1dTex = makeStrictW347Texture({
    handle: 7003,
    width: 1024,
    height: 1,
    depthOrArrayLayers: 1,
    mipLevelCount: 1,
    format: 'rgba8unorm',
    dimension: '1d',
    promoted1DTo2D: false,
  });
  const r1dA = executor.getOrCreateTextureView(real1dTex, {
    format: 'rgba8unorm',
    dimension: '2d',
    baseMipLevel: 2,
    mipLevelCount: 3,
    baseArrayLayer: 1,
    arrayLayerCount: 4,
  });
  const r1dB = executor.getOrCreateTextureView(real1dTex, {
    format: 'rgba8unorm',
    dimension: '1d',
    baseMipLevel: 0,
    mipLevelCount: 1,
    baseArrayLayer: 0,
    arrayLayerCount: 1,
  });
  assert.ok(r1dA);
  assert.equal(r1dA.viewDesc.dimension, '1d');
  assert.equal(r1dB, r1dA);
  assert.equal(real1dTex.viewCalls.length, 1);
});

// =============================================================================
// 4. Zero stale bind groups after resource destroy/recreate + O(1) cache soundness
// =============================================================================

test('W354: DrawExecutor O(1) bind-group cache handles 1..8 entries, out-of-order bindings, LRU eviction, and zero stale entries on destroy/recreate', () => {
  const gpu = createMockWebGpu();
  const executor = new DrawExecutor({ device: gpu.device, maxBindGroupCacheEntries: 4 });
  const pipe = gpu.device.createRenderPipeline({});
  const layout = gpu.device.createBindGroupLayout({ entries: [] });

  const buf0 = gpu.device.createBuffer({ size: 512, usage: 0x0040 });
  const buf1 = gpu.device.createBuffer({ size: 512, usage: 0x0040 });
  const tex0 = gpu.device.createTexture({ size: { width: 64, height: 64, depthOrArrayLayers: 1 }, format: 'rgba8unorm', usage: 0x04 });
  const tex1 = gpu.device.createTexture({ size: { width: 64, height: 64, depthOrArrayLayers: 1 }, format: 'rgba8unorm', usage: 0x04 });
  const smp0 = gpu.device.createSampler({});
  const smp1 = gpu.device.createSampler({});

  // 8-entry bind group (max fast-path size), passed in reverse binding order on second call -> must hit!
  const entries8 = [
    { binding: 0, resource: { buffer: buf0, offset: 0, size: 256 } },
    { binding: 1, resource: { buffer: buf1, offset: 256, size: 256 } },
    { binding: 2, resource: tex0 },
    { binding: 3, resource: smp0 },
    { binding: 4, resource: tex1 },
    { binding: 5, resource: smp1 },
    { binding: 6, resource: { buffer: buf0, offset: 256, size: 256 } },
    { binding: 7, resource: { buffer: buf1, offset: 0, size: 256 } },
  ];
  const bgA = executor.getOrCreateBindGroupForPipeline(pipe, 0, entries8);
  const bgA2 = executor.getOrCreateBindGroupForPipeline(pipe, 0, entries8.slice().reverse());
  assert.ok(bgA);
  assert.equal(bgA2, bgA, 'out-of-order 8-entry bind group must hit O(1) cache');
  assert.equal(pipe.getBindGroupLayoutCalls, 1);
  assert.equal(tex0.createViewCalls, 1);
  assert.equal(tex1.createViewCalls, 1);

  // Destroy buf1 -> lookup MUST fail (never return stale bgA)!
  buf1.destroyed = true;
  assert.equal(executor.getOrCreateBindGroupForPipeline(pipe, 0, entries8), null);

  // Recreate buf1 and re-query -> creates fresh bind group cleanly.
  const buf1New = gpu.device.createBuffer({ size: 512, usage: 0x0040 });
  const entries8Recreated = entries8.map((e) =>
    e.resource && e.resource.buffer === buf1
      ? { binding: e.binding, resource: { buffer: buf1New, offset: e.resource.offset, size: e.resource.size } }
      : e
  );
  const bgRecreated = executor.getOrCreateBindGroupForPipeline(pipe, 0, entries8Recreated);
  assert.ok(bgRecreated);
  assert.notEqual(bgRecreated, bgA);

  // Mark tex0 destroyed -> lookup MUST NOT return bgRecreated and MUST evict stale texture view.
  tex0.destroyed = true;
  assert.equal(executor.getOrCreateBindGroupForPipeline(pipe, 0, entries8Recreated), null);

  // Explicit invalidateResource(buf0) evicts all referencing entries in O(1).
  const evictedCount = executor.invalidateResource(buf0);
  assert.ok(evictedCount >= 1);
  assert.equal(executor.getStats().bindGroupCacheSize, 0);

  // Generation bump on a live buffer invalidates old cached bind group without returning stale handle.
  const genBuf = gpu.device.createBuffer({ size: 256, usage: 0x0040 });
  genBuf.generation = 1;
  const g1 = executor.getOrCreateBindGroup(layout, [{ binding: 0, resource: { buffer: genBuf, offset: 0, size: 256 } }]);
  const g1Hit = executor.getOrCreateBindGroup(layout, [{ binding: 0, resource: { buffer: genBuf, offset: 0, size: 256 } }]);
  assert.equal(g1Hit, g1);

  genBuf.generation = 2;
  const g2 = executor.getOrCreateBindGroup(layout, [{ binding: 0, resource: { buffer: genBuf, offset: 0, size: 256 } }]);
  assert.ok(g2);
  assert.notEqual(g2, g1, 'generation bump must never return stale bind group');

  // Destroying the cached GPUBindGroup itself forces eviction & recreation.
  g2.destroyed = true;
  const g3 = executor.getOrCreateBindGroup(layout, [{ binding: 0, resource: { buffer: genBuf, offset: 0, size: 256 } }]);
  assert.ok(g3);
  assert.notEqual(g3, g2);

  // >8 entry overflow bind group also caches & invalidates cleanly in the same unified LRU cache.
  const overflowEntries = [];
  for (let i = 0; i < 10; i += 1) {
    overflowEntries.push({ binding: i, resource: { buffer: genBuf, offset: 0, size: 256 } });
  }
  const bgOv1 = executor.getOrCreateBindGroup(layout, overflowEntries);
  const bgOv2 = executor.getOrCreateBindGroup(layout, overflowEntries);
  assert.equal(bgOv2, bgOv1);

  // LRU eviction caps cache at maxBindGroupCacheEntries (4).
  for (let i = 0; i < 6; i += 1) {
    const b = gpu.device.createBuffer({ size: 256, usage: 0x0040 });
    executor.getOrCreateBindGroup(layout, [{ binding: 0, resource: { buffer: b, offset: 0, size: 256 } }]);
  }
  assert.equal(executor.getStats().bindGroupCacheSize, 4);
  assert.ok(executor.getStats().bindGroupCacheEvictions >= 3);
});

test('W354: RenderStatePackets PresentX bind-group cache preserves multi-plane composition and never returns destroyed bind groups/textures', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'bgra8unorm' });
  const handler = installRenderStatePackets(processor);

  const bgraCode = formatCodeOf('bgra8unorm');
  processor.executeCommandBuffer(
    concat(
      createTexture2Packet({ id: 2, width: 1280, height: 720, format: bgraCode, usage: 0x14 }),
      createTexture2Packet({ id: 3, width: 1280, height: 720, format: bgraCode, usage: 0x14 }),
      bindGroupLayoutPacket(10, [
        { binding: 0, visibility: 2, type: 2, subtype: 0, dimension: 2 },
        { binding: 1, visibility: 2, type: 1, subtype: 0 },
      ])
    )
  );
  const smp = gpu.device.createSampler({});
  processor.samplers.set(11, smp);
  const layout10 = handler.bindGroupLayouts.get(10).layout;

  const p1a = handler.getOrCreatePresentXPlaneBindGroup({
    planeTextureId: 2,
    viewFormat: 'bgra8unorm',
    sampler: smp,
    samplerId: 11,
    layout: layout10,
    layoutId: 10,
  });
  const p1b = handler.getOrCreatePresentXPlaneBindGroup({
    planeTextureId: 2,
    viewFormat: 'bgra8unorm',
    sampler: smp,
    samplerId: 11,
    layout: layout10,
    layoutId: 10,
  });
  assert.ok(p1a && p1b);
  assert.equal(p1a.hit, false);
  assert.equal(p1b.hit, true);
  assert.equal(p1b.bindGroup, p1a.bindGroup);
  assert.equal(p1b.view, p1a.view);

  // Destroy cached bindGroup -> must miss and recreate cleanly (never return destroyed bindGroup).
  p1a.bindGroup.destroyed = true;
  const p1c = handler.getOrCreatePresentXPlaneBindGroup({
    planeTextureId: 2,
    viewFormat: 'bgra8unorm',
    sampler: smp,
    samplerId: 11,
    layout: layout10,
    layoutId: 10,
  });
  assert.ok(p1c);
  assert.equal(p1c.hit, false);
  assert.notEqual(p1c.bindGroup, p1a.bindGroup);

  // Destroy underlying plane texture -> must return null cleanly.
  processor.textures.get(2).destroyed = true;
  const p1Destroyed = handler.getOrCreatePresentXPlaneBindGroup({
    planeTextureId: 2,
    viewFormat: 'bgra8unorm',
    sampler: smp,
    samplerId: 11,
    layout: layout10,
    layoutId: 10,
  });
  assert.equal(p1Destroyed, null);
});

// =============================================================================
// 5. >30 FPS steady-state hot-loop throughput sanity check
// =============================================================================

test('W354: 60 frames x 100 draws steady-state hot loop runs with 99%+ cache/elision efficiency well within 30 FPS budget', () => {
  const gpu = createMockWebGpu();
  const executor = new DrawExecutor({ device: gpu.device, reuseDrawResult: true });
  const pipe = gpu.device.createRenderPipeline({});
  const vb = gpu.device.createBuffer({ size: 4096, usage: 0x0020 });
  const cbv = gpu.device.createBuffer({ size: 4096, usage: 0x0040 });
  const tex = gpu.device.createTexture({
    size: { width: 256, height: 256, depthOrArrayLayers: 1 },
    format: 'rgba8unorm',
    usage: 0x04,
  });
  const smp = gpu.device.createSampler({});
  const entries = [
    { binding: 0, resource: { buffer: cbv, offset: 0, size: 256 } },
    { binding: 1, resource: tex },
    { binding: 2, resource: smp },
  ];

  const t0 = performance.now();
  for (let frame = 0; frame < 60; frame += 1) {
    executor.beginFrame();
    const pass = gpu.device.createCommandEncoder().beginRenderPass({ colorAttachments: [] });
    executor.setPipeline(pipe);
    executor.setVertexBuffer(0, vb, 0);
    executor.setViewport(0, 0, 1280, 720, 0, 1);
    executor.setScissorRect(0, 0, 1280, 720);
    for (let d = 0; d < 100; d += 1) {
      const bg = executor.getOrCreateBindGroupForPipeline(pipe, 0, entries);
      executor.setBindGroup(0, bg);
      const res = executor.draw(pass, 36, 1, 0, 0);
      assert.equal(res.executed, true);
    }
    executor.endPass();
  }
  const elapsedMs = performance.now() - t0;

  const stats = executor.getStats();
  assert.equal(stats.totalDrawsExecuted, 6000);
  assert.equal(stats.bindGroupCacheMisses, 1);
  assert.equal(stats.bindGroupCacheHits, 5999);
  assert.equal(stats.textureViewCacheMisses, 1);
  assert.equal(stats.textureViewCacheHits, 5999);
  assert.equal(tex.createViewCalls, 1);
  assert.equal(pipe.getBindGroupLayoutCalls, 1);
  assert.equal(gpu.results('createBindGroup').length, 1);
  assert.equal(gpu.passCalls.filter((c) => c.op === 'setPipeline').length, 60);
  assert.equal(gpu.passCalls.filter((c) => c.op === 'setBindGroup').length, 60);
  assert.equal(gpu.passCalls.filter((c) => c.op === 'setVertexBuffer').length, 60);
  assert.equal(gpu.passCalls.filter((c) => c.op === 'setViewport').length, 60);
  assert.equal(gpu.passCalls.filter((c) => c.op === 'setScissorRect').length, 60);
  // 60 frames in JS worker must complete far below the 2000ms (30 FPS) frame budget.
  assert.ok(elapsedMs < 1500, `60-frame hot loop took ${elapsedMs.toFixed(1)}ms`);
});
