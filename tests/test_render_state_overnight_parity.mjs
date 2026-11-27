import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { WgpuCommandProcessor, OPCODES } = require('../web/workers/wgpu_worker.js');
const rs = require('../web/workers/render_state_packets.js');
const {
  installRenderStatePackets,
  V2_OPCODES,
  SWAPCHAIN_TEXTURE_ID,
  TEXTURE_FORMAT_NAMES,
  BINDING_ARRAYS_LAYOUT_FLAG,
  BINDING_ARRAYS_BG_FLAG,
  GPU_BUFFER_USAGE_INDEX,
  GPU_BUFFER_USAGE_INDIRECT,
  validateIndirectCommand,
  sanitizeTextureDescriptor,
  sanitizeRenderPassAttachments,
  sanitizeSamplerDescriptor,
  sanitizeColorTargetBlendState,
  clampViewportToTarget,
  resolveCompatibleViewFormats,
  getSrgbSiblingFormat,
  validateVertexBufferLayouts,
  parseWgslFragmentOutputLocations,
} = rs;

const formatCodeOf = (name) => {
  const code = TEXTURE_FORMAT_NAMES.indexOf(name);
  assert.ok(code > 0, `unknown texture format ${name}`);
  return code;
};

function header(opcode, payload) {
  const out = new Uint8Array(8 + payload.byteLength);
  const v = new DataView(out.buffer);
  v.setUint32(0, opcode, true);
  v.setUint32(4, payload.byteLength, true);
  out.set(payload, 8);
  return out;
}

function concat(...parts) {
  const total = parts.reduce((n, p) => n + p.byteLength, 0);
  const out = new Uint8Array(total);
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.byteLength;
  }
  return out;
}

class Stream {
  constructor() {
    this.chunks = [];
  }
  raw(bytes) {
    this.chunks.push(bytes);
    return this;
  }
  packet(opcode, bytes = new Uint8Array(0)) {
    this.chunks.push(header(opcode, bytes));
    return this;
  }
  u32s(opcode, values) {
    const b = new Uint8Array(values.length * 4);
    const v = new DataView(b.buffer);
    values.forEach((n, i) => v.setUint32(i * 4, n >>> 0, true));
    return this.packet(opcode, b);
  }
  buffer(id, usage, size) {
    const b = new Uint8Array(16);
    const v = new DataView(b.buffer);
    v.setUint32(0, id, true);
    v.setUint32(4, usage, true);
    v.setBigUint64(8, BigInt(size), true);
    return this.packet(OPCODES.CREATE_BUFFER, b);
  }
  shader(id, stage, code) {
    const enc = new TextEncoder().encode(code);
    const b = new Uint8Array(16 + enc.byteLength);
    const v = new DataView(b.buffer);
    v.setUint32(0, id, true);
    v.setUint32(4, stage, true);
    v.setUint32(8, enc.byteLength, true);
    b.set(enc, 16);
    return this.packet(OPCODES.CREATE_SHADER_MODULE, b);
  }
  indirect(opcode, bufferId, offset) {
    const b = new Uint8Array(16);
    const v = new DataView(b.buffer);
    v.setUint32(0, bufferId, true);
    v.setBigUint64(8, BigInt(offset), true);
    return this.packet(opcode, b);
  }
  bytes() {
    return concat(...this.chunks);
  }
}

function viewDesc({
  format = 0,
  dimension = 0,
  aspect = 0,
  baseMip = 0,
  mipCount = 0,
  baseLayer = 0,
  layerCount = 0,
} = {}) {
  const b = new Uint8Array(16);
  const v = new DataView(b.buffer);
  b[0] = format;
  b[1] = dimension;
  b[2] = aspect;
  v.setUint16(4, baseMip, true);
  v.setUint16(6, mipCount, true);
  v.setUint32(8, baseLayer, true);
  v.setUint32(12, layerCount, true);
  return b;
}

function createTexture2Packet({
  id,
  width = 64,
  height = 64,
  depthOrArrayLayers = 1,
  mipLevelCount = 1,
  sampleCount = 1,
  usage = 0x04,
  dimension = 1, // 0=1d, 1=2d, 2=3d
  format = formatCodeOf('rgba8unorm'),
  viewFormats = [],
} = {}) {
  const b = new Uint8Array(40);
  const v = new DataView(b.buffer);
  v.setUint32(0, id, true);
  v.setUint32(4, width, true);
  v.setUint32(8, height, true);
  v.setUint32(12, depthOrArrayLayers, true);
  v.setUint32(16, mipLevelCount, true);
  v.setUint32(20, sampleCount, true);
  v.setUint32(24, usage, true);
  b[28] = dimension;
  b[29] = format;
  b[30] = viewFormats.length;
  viewFormats.forEach((vf, i) => {
    b[32 + i] = vf;
  });
  return header(V2_OPCODES.CREATE_TEXTURE2, b);
}

function bindGroupLayoutPacket(id, entries, { contentId = 0n, flags = 0 } = {}) {
  const b = new Uint8Array(24 + entries.length * 24);
  const v = new DataView(b.buffer);
  v.setUint32(0, id, true);
  v.setUint32(4, entries.length, true);
  v.setUint32(8, flags, true);
  v.setBigUint64(16, contentId, true);
  entries.forEach((e, i) => {
    const o = 24 + i * 24;
    v.setUint32(o, e.binding, true);
    v.setUint32(o + 4, e.visibility ?? 4, true);
    b[o + 8] = e.type;
    b[o + 9] = e.subtype || 0;
    b[o + 10] = e.dimension || 0;
    b[o + 11] = e.flags || 0;
    b[o + 12] = e.storageFormat || 0;
    v.setUint16(o + 14, e.bindingArraySize || 0, true);
    v.setBigUint64(o + 16, BigInt(e.minBindingSize || 0), true);
  });
  return header(V2_OPCODES.CREATE_BIND_GROUP_LAYOUT, b);
}

function pipelineLayoutPacket(id, groups, { contentId = 0n, flags = 0 } = {}) {
  const b = new Uint8Array(24 + groups.length * 4);
  const v = new DataView(b.buffer);
  v.setUint32(0, id, true);
  v.setUint32(4, groups.length, true);
  v.setUint32(8, flags, true);
  v.setBigUint64(16, contentId, true);
  groups.forEach((g, i) => v.setUint32(24 + i * 4, g, true));
  return header(V2_OPCODES.CREATE_PIPELINE_LAYOUT, b);
}

function computePipeline2Packet(id, csId, layoutId = 0) {
  const b = new Uint8Array(32);
  const v = new DataView(b.buffer);
  v.setUint32(0, id, true);
  v.setUint32(4, csId, true);
  v.setUint32(8, layoutId, true);
  return header(V2_OPCODES.CREATE_COMPUTE_PIPELINE2, b);
}

function renderPipeline2Packet(id, vsId, fsId, layoutId, key) {
  const b = new Uint8Array(16 + key.byteLength);
  const v = new DataView(b.buffer);
  v.setUint32(0, id, true);
  v.setUint32(4, vsId, true);
  v.setUint32(8, fsId, true);
  v.setUint32(12, layoutId, true);
  b.set(key, 16);
  return header(V2_OPCODES.CREATE_RENDER_PIPELINE2, b);
}

function bindGroup2Packet(id, layoutId, entries, { flags = 0 } = {}) {
  const b = new Uint8Array(16 + entries.length * 48);
  const v = new DataView(b.buffer);
  v.setUint32(0, id, true);
  v.setUint32(4, layoutId, true);
  v.setUint32(8, entries.length, true);
  v.setUint32(12, flags, true);
  entries.forEach((e, i) => {
    const o = 16 + i * 48;
    v.setUint32(o, e.binding, true);
    v.setUint32(o + 4, e.type, true);
    v.setUint32(o + 8, e.id, true);
    v.setUint16(o + 12, e.arrayIndex || 0, true);
    v.setBigUint64(o + 16, BigInt(e.offset || 0), true);
    v.setBigUint64(o + 24, BigInt(e.size || 0), true);
    if (e.view) b.set(e.view, o + 32);
  });
  return header(V2_OPCODES.CREATE_BIND_GROUP2, b);
}

function setBindGroup2Packet(index, id, offsets = []) {
  const b = new Uint8Array(16 + offsets.length * 4);
  const v = new DataView(b.buffer);
  v.setUint32(0, index, true);
  v.setUint32(4, id, true);
  v.setUint32(8, offsets.length, true);
  offsets.forEach((off, i) => v.setUint32(16 + i * 4, off, true));
  return header(V2_OPCODES.SET_BIND_GROUP2, b);
}

function beginRenderPassPacket({ colors = [], depthStencil = null } = {}) {
  const b = new Uint8Array(368);
  const v = new DataView(b.buffer);
  v.setUint32(0, colors.length, true);
  colors.forEach((c, i) => {
    if (!c) return;
    const o = 8 + i * 40;
    v.setUint32(o, c.textureId, true);
    b[o + 4] = c.loadOp ?? 2;
    b[o + 5] = c.storeOp ?? 1;
    if (c.view) b.set(c.view, o + 8);
    const cv = c.clearValue || [0, 0, 0, 1];
    v.setFloat32(o + 24, cv[0], true);
    v.setFloat32(o + 28, cv[1], true);
    v.setFloat32(o + 32, cv[2], true);
    v.setFloat32(o + 36, cv[3], true);
  });
  if (depthStencil) {
    const d = 328;
    v.setUint32(d, depthStencil.textureId, true);
    b[d + 4] = depthStencil.depthLoadOp ?? 0;
    b[d + 5] = depthStencil.depthStoreOp ?? 0;
    b[d + 6] = depthStencil.stencilLoadOp ?? 0;
    b[d + 7] = depthStencil.stencilStoreOp ?? 0;
    if (depthStencil.view) b.set(depthStencil.view, d + 8);
    v.setFloat32(d + 24, depthStencil.depthClearValue ?? 1, true);
    v.setUint32(d + 28, depthStencil.stencilClearValue ?? 0, true);
    b[d + 32] = depthStencil.depthReadOnly ? 1 : 0;
    b[d + 33] = depthStencil.stencilReadOnly ? 1 : 0;
  }
  return header(V2_OPCODES.BEGIN_RENDER_PASS, b);
}

function createSampler2Packet(
  id,
  {
    magFilter = 0,
    minFilter = 0,
    mipmapFilter = 0,
    compare = 0,
    maxAnisotropy = 1,
    lodMinClamp = 0,
    lodMaxClamp = 32,
  } = {}
) {
  const b = new Uint8Array(24);
  const v = new DataView(b.buffer);
  v.setUint32(0, id, true);
  b[7] = magFilter;
  b[8] = minFilter;
  b[9] = mipmapFilter;
  b[10] = compare;
  v.setUint16(12, maxAnisotropy, true);
  v.setFloat32(16, lodMinClamp, true);
  v.setFloat32(20, lodMaxClamp, true);
  return header(V2_OPCODES.CREATE_SAMPLER2, b);
}

function createDawnStrictMock({ features = [], limits = {}, swapchainFormat = 'bgra8unorm' } = {}) {
  const calls = [];
  const passCalls = [];
  const validationErrors = [];
  let nextHandle = 1;
  const featureSet = new Set(features);
  const deviceLimits = {
    minUniformBufferOffsetAlignment: 256,
    minStorageBufferOffsetAlignment: 256,
    maxUniformBufferBindingSize: 65536,
    maxStorageBufferBindingSize: 134217728,
    maxComputeWorkgroupsPerDimension: 65535,
    maxBindGroups: 4,
    maxVertexBufferArrayStride: 2048,
    ...limits,
  };
  const swapchainView = { kind: 'GPUTextureView', id: 'swapchain-view', width: 1280, height: 720 };
  const swapchainTexture = {
    kind: 'GPUTexture',
    width: 1280,
    height: 720,
    format: swapchainFormat,
    createView(desc) {
      if (desc && desc.format && desc.format !== swapchainFormat) {
        const sibling = getSrgbSiblingFormat(swapchainFormat);
        if (desc.format !== sibling) {
          validationErrors.push(`swapchain.createView: incompatible format ${desc.format} for ${swapchainFormat}`);
        }
      }
      const v = { kind: 'GPUTextureView', handle: nextHandle++, viewDesc: desc, texture: swapchainTexture };
      calls.push({ op: 'swapchain.createView', desc, result: v });
      return v;
    },
  };
  const ctx = {
    canvas: { width: 1280, height: 720 },
    getCurrentTexture: () => swapchainTexture,
  };

  const makePass = (kind) => ({
    setPipeline(p) {
      passCalls.push({ pass: kind, op: 'setPipeline', args: [p] });
    },
    setBindGroup(i, bg, offsets) {
      if (bg && bg._layout) {
        const dynEntries = bg._layout.entries
          .filter((e) => e.buffer && e.buffer.hasDynamicOffset)
          .sort((a, b) => a.binding - b.binding);
        const got = Array.isArray(offsets) ? offsets : [];
        if (got.length !== dynEntries.length) {
          validationErrors.push(`setBindGroup(${i}): dynOffset count ${got.length} != ${dynEntries.length}`);
        }
        for (let k = 0; k < Math.min(got.length, dynEntries.length); k += 1) {
          const le = dynEntries[k];
          const align =
            (le.buffer.type || 'uniform') === 'uniform'
              ? deviceLimits.minUniformBufferOffsetAlignment
              : deviceLimits.minStorageBufferOffsetAlignment;
          if (got[k] % align !== 0) {
            validationErrors.push(`setBindGroup(${i}): dynOffset[${k}]=${got[k]} not aligned to ${align}`);
          }
          const boundEntry = bg.desc.entries.find((e) => e.binding === le.binding);
          if (boundEntry && boundEntry.resource && boundEntry.resource.buffer) {
            const buf = boundEntry.resource.buffer;
            const baseOff = boundEntry.resource.offset || 0;
            const bindSz = boundEntry.resource.size || le.buffer.minBindingSize || 0;
            if (baseOff + got[k] + bindSz > buf.size) {
              validationErrors.push(
                `setBindGroup(${i}): dynOffset[${k}]=${got[k]} OOB (${baseOff}+${got[k]}+${bindSz} > ${buf.size})`
              );
            }
          }
        }
      }
      passCalls.push({
        pass: kind,
        op: 'setBindGroup',
        args: offsets === undefined ? [i, bg] : [i, bg, [...offsets]],
      });
    },
    setVertexBuffer(...a) {
      passCalls.push({ pass: kind, op: 'setVertexBuffer', args: a });
    },
    setIndexBuffer(...a) {
      passCalls.push({ pass: kind, op: 'setIndexBuffer', args: a });
    },
    setViewport(...a) {
      passCalls.push({ pass: kind, op: 'setViewport', args: a });
    },
    setScissorRect(...a) {
      passCalls.push({ pass: kind, op: 'setScissorRect', args: a });
    },
    setBlendConstant(...a) {
      passCalls.push({ pass: kind, op: 'setBlendConstant', args: a });
    },
    setStencilReference(...a) {
      passCalls.push({ pass: kind, op: 'setStencilReference', args: a });
    },
    draw(...a) {
      passCalls.push({ pass: kind, op: 'draw', args: a });
    },
    drawIndexed(...a) {
      passCalls.push({ pass: kind, op: 'drawIndexed', args: a });
    },
    drawIndirect(buf, off) {
      if (!buf || (buf.usage & GPU_BUFFER_USAGE_INDIRECT) === 0 || off % 4 !== 0 || off + 16 > buf.size) {
        validationErrors.push(`drawIndirect invalid: usage=${buf && buf.usage} off=${off} size=${buf && buf.size}`);
      }
      passCalls.push({ pass: kind, op: 'drawIndirect', args: [buf, off] });
    },
    drawIndexedIndirect(buf, off) {
      if (!buf || (buf.usage & GPU_BUFFER_USAGE_INDIRECT) === 0 || off % 4 !== 0 || off + 20 > buf.size) {
        validationErrors.push(`drawIndexedIndirect invalid: usage=${buf && buf.usage} off=${off} size=${buf && buf.size}`);
      }
      passCalls.push({ pass: kind, op: 'drawIndexedIndirect', args: [buf, off] });
    },
    dispatchWorkgroups(...a) {
      passCalls.push({ pass: kind, op: 'dispatchWorkgroups', args: a });
    },
    dispatchWorkgroupsIndirect(buf, off) {
      if (!buf || (buf.usage & GPU_BUFFER_USAGE_INDIRECT) === 0 || off % 4 !== 0 || off + 12 > buf.size) {
        validationErrors.push(`dispatchWorkgroupsIndirect invalid: usage=${buf && buf.usage} off=${off} size=${buf && buf.size}`);
      }
      passCalls.push({ pass: kind, op: 'dispatchWorkgroupsIndirect', args: [buf, off] });
    },
    end() {
      passCalls.push({ pass: kind, op: 'end', args: [] });
    },
  });

  const device = {
    features: featureSet,
    limits: deviceLimits,
    createBuffer(desc) {
      const buf = {
        kind: 'GPUBuffer',
        handle: nextHandle++,
        size: desc.size,
        usage: desc.usage,
        desc: { ...desc },
        destroy() {
          this.destroyed = true;
        },
      };
      calls.push({ op: 'createBuffer', desc: { ...desc }, result: buf });
      return buf;
    },
    createShaderModule(desc) {
      const m = { kind: 'GPUShaderModule', handle: nextHandle++, code: desc.code };
      calls.push({ op: 'createShaderModule', desc, result: m });
      return m;
    },
    createTexture(desc) {
      if (desc.sampleCount > 1 && (desc.usage & 0x10) === 0) {
        validationErrors.push(`createTexture: multisampled texture lacks RENDER_ATTACHMENT`);
      }
      if (desc.sampleCount > 1 && (desc.usage & 0x08) !== 0) {
        validationErrors.push(`createTexture: multisampled texture cannot have STORAGE_BINDING`);
      }
      if (Array.isArray(desc.viewFormats)) {
        const sibling = getSrgbSiblingFormat(desc.format);
        for (const vf of desc.viewFormats) {
          if (vf !== desc.format && vf !== sibling) {
            validationErrors.push(`createTexture: incompatible viewFormat ${vf} for ${desc.format}`);
          }
        }
      }
      const tex = {
        kind: 'GPUTexture',
        handle: nextHandle++,
        width: desc.size.width,
        height: desc.size.height,
        depthOrArrayLayers: desc.size.depthOrArrayLayers || 1,
        mipLevelCount: desc.mipLevelCount || 1,
        sampleCount: desc.sampleCount || 1,
        dimension: desc.dimension || '2d',
        format: desc.format,
        usage: desc.usage,
        viewFormats: desc.viewFormats ? [...desc.viewFormats] : [],
        desc: { ...desc },
        createView(viewDesc) {
          const vd = viewDesc || {};
          if (vd.format && vd.format !== desc.format) {
            const isAspectFmt =
              (vd.aspect === 'depth-only' && (vd.format === 'depth32float' || vd.format === 'depth24plus')) ||
              (vd.aspect === 'stencil-only' && vd.format === 'stencil8');
            if (!isAspectFmt && !tex.viewFormats.includes(vd.format)) {
              validationErrors.push(`createView: format ${vd.format} not in viewFormats for ${desc.format}`);
            }
          }
          const view = {
            kind: 'GPUTextureView',
            handle: nextHandle++,
            textureHandle: tex.handle,
            texture: tex,
            viewDesc: vd,
          };
          calls.push({ op: 'texture.createView', texture: tex, desc: vd, result: view });
          return view;
        },
        destroy() {
          this.destroyed = true;
        },
      };
      calls.push({ op: 'createTexture', desc, result: tex });
      return tex;
    },
    createSampler(desc) {
      if (
        (desc.maxAnisotropy || 1) > 1 &&
        (desc.magFilter !== 'linear' || desc.minFilter !== 'linear' || desc.mipmapFilter !== 'linear')
      ) {
        validationErrors.push(`createSampler: maxAnisotropy=${desc.maxAnisotropy} requires linear filters`);
      }
      if (desc.lodMinClamp < 0 || desc.lodMaxClamp < desc.lodMinClamp || !Number.isFinite(desc.lodMinClamp) || !Number.isFinite(desc.lodMaxClamp)) {
        validationErrors.push(`createSampler: invalid lod clamps ${desc.lodMinClamp}..${desc.lodMaxClamp}`);
      }
      const s = { kind: 'GPUSampler', handle: nextHandle++, desc: { ...desc } };
      calls.push({ op: 'createSampler', desc: { ...desc }, result: s });
      return s;
    },
    createBindGroupLayout(desc) {
      for (const e of desc.entries) {
        if (e.texture && e.texture.multisampled) {
          if (e.texture.viewDimension !== '2d' || e.texture.sampleType === 'float') {
            validationErrors.push(`createBindGroupLayout: invalid multisampled texture entry`);
          }
        }
        if (e.storageTexture && e.storageTexture.access === 'read-write') {
          const coreRw = new Set(['r32uint', 'r32sint', 'r32float']);
          if (!coreRw.has(e.storageTexture.format) && !featureSet.has('texture-formats-tier2')) {
            validationErrors.push(`createBindGroupLayout: read-write unsupported for ${e.storageTexture.format}`);
          }
        }
      }
      const l = { kind: 'GPUBindGroupLayout', handle: nextHandle++, desc, entries: desc.entries };
      calls.push({ op: 'createBindGroupLayout', desc, result: l });
      return l;
    },
    createPipelineLayout(desc) {
      if (desc.bindGroupLayouts.length > deviceLimits.maxBindGroups) {
        validationErrors.push(`createPipelineLayout: ${desc.bindGroupLayouts.length} > maxBindGroups`);
      }
      const l = { kind: 'GPUPipelineLayout', handle: nextHandle++, desc };
      calls.push({ op: 'createPipelineLayout', desc, result: l });
      return l;
    },
    createRenderPipeline(desc) {
      if (desc.primitive && desc.primitive.unclippedDepth && !featureSet.has('depth-clip-control')) {
        validationErrors.push(`createRenderPipeline: unclippedDepth requires depth-clip-control`);
      }
      if (desc.depthStencil) {
        if (desc.depthStencil.format === 'depth32float-stencil8' && !featureSet.has('depth32float-stencil8')) {
          validationErrors.push(`createRenderPipeline: depth32float-stencil8 requires feature`);
        }
        const topo = desc.primitive ? desc.primitive.topology : 'triangle-list';
        if (
          (topo === 'line-list' || topo === 'line-strip' || topo === 'point-list') &&
          (desc.depthStencil.depthBias || desc.depthStencil.depthBiasSlopeScale || desc.depthStencil.depthBiasClamp)
        ) {
          validationErrors.push(`createRenderPipeline: depthBias forbidden on ${topo}`);
        }
      }
      const p = { kind: 'GPURenderPipeline', handle: nextHandle++, desc };
      calls.push({ op: 'createRenderPipeline', desc, result: p });
      return p;
    },
    createComputePipeline(desc) {
      const p = { kind: 'GPUComputePipeline', handle: nextHandle++, desc };
      calls.push({ op: 'createComputePipeline', desc, result: p });
      return p;
    },
    createBindGroup(desc) {
      // Check Dawn writable storage texture aliasing within the bind group.
      const writableStorageHandles = new Set();
      const writableBufHandles = new Set();
      const readOnlyBufHandles = new Set();
      const layoutEntries = desc.layout && Array.isArray(desc.layout.entries) ? desc.layout.entries : [];
      for (const e of desc.entries) {
        const le = layoutEntries.find(
          (x) =>
            x.binding === e.binding ||
            (x.bindingArraySize >= 1 && e.binding >= x.binding && e.binding < x.binding + x.bindingArraySize)
        );
        if (le && le.storageTexture && e.resource && e.resource.textureHandle) {
          if (writableStorageHandles.has(e.resource.textureHandle)) {
            validationErrors.push(`createBindGroup: writable storage texture handle ${e.resource.textureHandle} aliased`);
          }
          writableStorageHandles.add(e.resource.textureHandle);
        }
        if (le && le.buffer && e.resource && e.resource.buffer) {
          if (le.buffer.type === 'storage') {
            writableBufHandles.add(e.resource.buffer.handle);
          } else {
            readOnlyBufHandles.add(e.resource.buffer.handle);
          }
        }
      }
      for (const h of writableBufHandles) {
        if (readOnlyBufHandles.has(h)) {
          validationErrors.push(`createBindGroup: buffer handle ${h} used as both writable storage and uniform/read-only-storage`);
        }
      }
      const bg = { kind: 'GPUBindGroup', handle: nextHandle++, desc, _layout: desc.layout };
      calls.push({ op: 'createBindGroup', desc, result: bg });
      return bg;
    },
    createCommandEncoder() {
      return {
        beginRenderPass(desc) {
          const hasColor = Array.isArray(desc.colorAttachments) && desc.colorAttachments.some((c) => Boolean(c && c.view));
          const hasDs = Boolean(desc.depthStencilAttachment && desc.depthStencilAttachment.view);
          if (!hasColor && !hasDs) {
            validationErrors.push('beginRenderPass: empty render pass (no color or depthStencil attachments)');
          }
          const ds = desc.depthStencilAttachment;
          if (ds) {
            if (ds.depthReadOnly && (ds.depthLoadOp !== undefined || ds.depthStoreOp !== undefined || ds.depthClearValue !== undefined)) {
              validationErrors.push('beginRenderPass: depthReadOnly forbids depthLoadOp/depthStoreOp/depthClearValue');
            }
            if (ds.stencilReadOnly && (ds.stencilLoadOp !== undefined || ds.stencilStoreOp !== undefined || ds.stencilClearValue !== undefined)) {
              validationErrors.push('beginRenderPass: stencilReadOnly forbids stencilLoadOp/stencilStoreOp/stencilClearValue');
            }
            if (ds.depthLoadOp === 'undefined' || ds.depthStoreOp === 'undefined' || ds.stencilLoadOp === 'undefined' || ds.stencilStoreOp === 'undefined') {
              validationErrors.push('beginRenderPass: invalid string "undefined" loadOp/storeOp');
            }
            const dsFmt = ds.view && ds.view.texture && ds.view.texture.format;
            if (dsFmt === 'depth32float' || dsFmt === 'depth24plus' || dsFmt === 'depth16unorm') {
              if (ds.stencilLoadOp !== undefined || ds.stencilStoreOp !== undefined || ds.stencilReadOnly !== undefined) {
                validationErrors.push(`beginRenderPass: stencil fields forbidden on depth-only format ${dsFmt}`);
              }
              if (!ds.depthReadOnly && (!ds.depthLoadOp || !ds.depthStoreOp)) {
                validationErrors.push(`beginRenderPass: depthLoadOp/depthStoreOp required on ${dsFmt}`);
              }
            }
            if (dsFmt === 'stencil8') {
              if (ds.depthLoadOp !== undefined || ds.depthStoreOp !== undefined || ds.depthReadOnly !== undefined) {
                validationErrors.push('beginRenderPass: depth fields forbidden on stencil8');
              }
              if (!ds.stencilReadOnly && (!ds.stencilLoadOp || !ds.stencilStoreOp)) {
                validationErrors.push('beginRenderPass: stencilLoadOp/stencilStoreOp required on stencil8');
              }
            }
          }
          passCalls.push({ pass: 'render', op: 'beginRenderPass', args: [desc] });
          return makePass('render');
        },
        beginComputePass() {
          passCalls.push({ pass: 'compute', op: 'beginComputePass', args: [] });
          return makePass('compute');
        },
        finish: () => ({ kind: 'GPUCommandBuffer', handle: nextHandle++ }),
      };
    },
    queue: {
      writeBuffer() {},
      writeTexture() {},
      submit() {},
    },
  };

  return {
    device,
    ctx,
    calls,
    passCalls,
    validationErrors,
    swapchainView,
    swapchainTexture,
    results: (op) => calls.filter((c) => c.op === op),
  };
}

// -----------------------------------------------------------------------------
// 1. BeginRenderPass Dawn-strict depth-stencil, attachment views, MSAA, & empty pass guards
// -----------------------------------------------------------------------------

test('Orch-42 BeginRenderPass enforces Dawn depth-stencil readOnly/aspect/loadOp/storeOp invariants and drops invalid attachments', () => {
  const gpu = createDawnStrictMock();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'bgra8unorm' });
  const handler = installRenderStatePackets(processor);

  const s = new Stream()
    // 10: mipmapped layered color texture (128x128, 3 mips, 4 layers)
    .raw(createTexture2Packet({ id: 10, width: 128, height: 128, mipLevelCount: 3, depthOrArrayLayers: 4, format: formatCodeOf('rgba8unorm'), usage: 0x10 }))
    // 11: depth32float (depth-only!)
    .raw(createTexture2Packet({ id: 11, width: 128, height: 128, format: formatCodeOf('depth32float'), usage: 0x10 }))
    // 12: stencil8 (stencil-only!)
    .raw(createTexture2Packet({ id: 12, width: 128, height: 128, format: formatCodeOf('stencil8'), usage: 0x10 }))
    // 13: depth24plus-stencil8 (combined)
    .raw(createTexture2Packet({ id: 13, width: 128, height: 128, format: formatCodeOf('depth24plus-stencil8'), usage: 0x10 }))
    // 14: 4x MSAA color texture (sampleCount mismatch against 1x depth 11)
    .raw(createTexture2Packet({ id: 14, width: 128, height: 128, sampleCount: 4, format: formatCodeOf('rgba8unorm'), usage: 0x10 }))
    // 15: color texture WITHOUT RENDER_ATTACHMENT usage (TEXTURE_BINDING only)
    .raw(createTexture2Packet({ id: 15, width: 128, height: 128, format: formatCodeOf('rgba8unorm'), usage: 0x04 }))
    // 16: 3D color texture (requires depthSlice)
    .raw(createTexture2Packet({ id: 16, width: 32, height: 32, depthOrArrayLayers: 8, dimension: 2, format: formatCodeOf('rgba8unorm'), usage: 0x10 }))
    .packet(OPCODES.BEGIN_FRAME)
    // Pass 1: depth32float with depthReadOnly=1 + illegal depthLoadOp=clear + illegal stencilLoadOp=clear
    .raw(
      beginRenderPassPacket({
        colors: [{ textureId: 10, loadOp: 2, storeOp: 1 }],
        depthStencil: {
          textureId: 11,
          depthLoadOp: 2,
          depthStoreOp: 1,
          depthReadOnly: true,
          stencilLoadOp: 2,
          stencilStoreOp: 1,
          stencilReadOnly: true,
        },
      })
    )
    .raw(header(V2_OPCODES.END_RENDER_PASS, new Uint8Array(0)))
    // Pass 2: depth32float with depthLoadOp=0 (missing!) and depthStoreOp=0 -> normalized to load/store (never 'undefined')
    .raw(
      beginRenderPassPacket({
        colors: [{ textureId: 10, loadOp: 1, storeOp: 1 }],
        depthStencil: { textureId: 11, depthLoadOp: 0, depthStoreOp: 0, depthReadOnly: false },
      })
    )
    .raw(header(V2_OPCODES.END_RENDER_PASS, new Uint8Array(0)))
    // Pass 3: stencil8 with illegal depthLoadOp=clear + missing stencilStoreOp=0
    .raw(
      beginRenderPassPacket({
        colors: [{ textureId: 10, loadOp: 1, storeOp: 1 }],
        depthStencil: { textureId: 12, depthLoadOp: 2, depthStoreOp: 1, stencilLoadOp: 2, stencilStoreOp: 0, stencilClearValue: 5 },
      })
    )
    .raw(header(V2_OPCODES.END_RENDER_PASS, new Uint8Array(0)))
    // Pass 4: combined depth24plus-stencil8 with sub-aspect depth-only view -> coerced to aspect 'all'
    .raw(
      beginRenderPassPacket({
        colors: [{ textureId: 10, loadOp: 1, storeOp: 1 }],
        depthStencil: { textureId: 13, view: viewDesc({ aspect: 1 }), depthLoadOp: 1, depthStoreOp: 1, stencilReadOnly: true },
      })
    )
    .raw(header(V2_OPCODES.END_RENDER_PASS, new Uint8Array(0)))
    // Pass 5: 4x MSAA color + 1x depth -> mismatched sampleCount drops depth
    .raw(
      beginRenderPassPacket({
        colors: [{ textureId: 14, loadOp: 2, storeOp: 1 }],
        depthStencil: { textureId: 11, depthLoadOp: 2, depthStoreOp: 1 },
      })
    )
    .raw(header(V2_OPCODES.END_RENDER_PASS, new Uint8Array(0)))
    // Pass 6: non-RENDER_ATTACHMENT color only -> all attachments dropped -> empty pass skipped cleanly!
    .raw(beginRenderPassPacket({ colors: [{ textureId: 15, loadOp: 2, storeOp: 1 }] }))
    .raw(header(V2_OPCODES.END_RENDER_PASS, new Uint8Array(0)))
    // Pass 7: 3D color attachment -> depthSlice set and dimension '3d'
    .raw(
      beginRenderPassPacket({
        colors: [{ textureId: 16, view: viewDesc({ baseLayer: 3 }), loadOp: 2, storeOp: 1 }],
      })
    )
    .raw(header(V2_OPCODES.END_RENDER_PASS, new Uint8Array(0)))
    .packet(OPCODES.PRESENT);

  processor.executeCommandBuffer(s.bytes());
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.deepEqual(gpu.validationErrors, [], 'zero Dawn validation errors across all render passes');

  const begins = gpu.passCalls.filter((c) => c.op === 'beginRenderPass').map((c) => c.args[0]);
  // Pass 6 was skipped because it had zero valid attachments -> 6 passes executed, not 7.
  assert.equal(begins.length, 6);

  // Pass 1: depthReadOnly kept, depthLoadOp/depthStoreOp/stencil* all stripped; mipmapped/layered color coerced to 2d single-mip/single-layer
  assert.equal(begins[0].colorAttachments[0].view.viewDesc.dimension, '2d');
  assert.equal(begins[0].colorAttachments[0].view.viewDesc.mipLevelCount, 1);
  assert.equal(begins[0].colorAttachments[0].view.viewDesc.arrayLayerCount, 1);
  assert.deepEqual(begins[0].depthStencilAttachment, {
    view: begins[0].depthStencilAttachment.view,
    depthReadOnly: true,
  });

  // Pass 2: missing depthLoadOp/depthStoreOp normalized to load/store
  assert.equal(begins[1].depthStencilAttachment.depthLoadOp, 'load');
  assert.equal(begins[1].depthStencilAttachment.depthStoreOp, 'store');
  assert.equal('stencilLoadOp' in begins[1].depthStencilAttachment, false);

  // Pass 3: stencil8 has zero depth fields and normalized stencilStoreOp='store'
  assert.equal('depthLoadOp' in begins[2].depthStencilAttachment, false);
  assert.equal('depthReadOnly' in begins[2].depthStencilAttachment, false);
  assert.equal(begins[2].depthStencilAttachment.stencilLoadOp, 'clear');
  assert.equal(begins[2].depthStencilAttachment.stencilStoreOp, 'store');
  assert.equal(begins[2].depthStencilAttachment.stencilClearValue, 5);

  // Pass 4: combined depth-stencil attachment view coerced from depth-only -> all
  assert.equal(begins[3].depthStencilAttachment.view.viewDesc.aspect, 'all');

  // Pass 5: 4x MSAA vs 1x depth dropped mismatched depth
  assert.equal('depthStencilAttachment' in begins[4], false);

  // Pass 7 (index 5): 3D color attachment has depthSlice=3 and 3d view
  assert.equal(begins[5].colorAttachments[0].depthSlice, 3);
  assert.equal(begins[5].colorAttachments[0].view.viewDesc.dimension, '3d');
});

test('Orch-42 sanitizeRenderPassAttachments validates resolveTarget sampleCount/format/usage and sanitizes non-finite clearValue', () => {
  const singleSampleTex = { width: 64, height: 64, sampleCount: 1, format: 'rgba8unorm', usage: 0x10 };
  const msaaTex = { width: 64, height: 64, sampleCount: 4, format: 'rgba8unorm', usage: 0x10 };
  const intMsaaTex = { width: 64, height: 64, sampleCount: 4, format: 'rgba8uint', usage: 0x10 };
  const noRaTex = { width: 64, height: 64, sampleCount: 1, format: 'rgba8unorm', usage: 0x04 };
  const r1 = sanitizeRenderPassAttachments({
    colorAttachments: [
      {
        view: { texture: singleSampleTex },
        resolveTarget: { texture: singleSampleTex },
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: Number.NaN, g: Number.POSITIVE_INFINITY, b: -0, a: Number.NaN },
      },
      {
        view: { texture: msaaTex },
        resolveTarget: { texture: noRaTex },
        loadOp: 'load',
        storeOp: 'store',
      },
    ],
  });
  assert.equal('resolveTarget' in r1.descriptor.colorAttachments[0], false, 'single-sampled attachment cannot have resolveTarget');
  assert.deepEqual(r1.descriptor.colorAttachments[0].clearValue, { r: 0, g: 0, b: 0, a: 1 });
  assert.equal(r1.descriptor.colorAttachments[1], null, 'mismatched 4x MSAA secondary attachment dropped against 1x reference');

  const r2 = sanitizeRenderPassAttachments({
    colorAttachments: [
      {
        view: { texture: msaaTex },
        resolveTarget: { texture: noRaTex },
        loadOp: 'load',
        storeOp: 'store',
      },
      {
        view: { texture: intMsaaTex },
        resolveTarget: { texture: { width: 64, height: 64, sampleCount: 1, format: 'rgba8uint', usage: 0x10 } },
        loadOp: 'load',
        storeOp: 'store',
      },
    ],
  });
  assert.equal('resolveTarget' in r2.descriptor.colorAttachments[0], false, 'resolveTarget without RENDER_ATTACHMENT dropped');
  assert.equal('resolveTarget' in r2.descriptor.colorAttachments[1], false, 'integer format cannot resolve');
});

// -----------------------------------------------------------------------------
// 2. Indirect draw/dispatch validation + #1941 ExecuteIndirect dynamic offsets & compute pipeline switch
// -----------------------------------------------------------------------------

test('Orch-42 validateIndirectCommand and _preparePassCommandsForFlush enforce INDIRECT usage, 4B alignment, bounds (16/20/12B), and index buffer', () => {
  assert.equal(validateIndirectCommand('drawIndirect', null, 0).reason, 'missing-or-destroyed');
  assert.equal(
    validateIndirectCommand('drawIndirect', { size: 64, usage: 0x0080 }, 0).reason,
    'missing-indirect-usage'
  );
  assert.equal(
    validateIndirectCommand('drawIndirect', { size: 64, usage: GPU_BUFFER_USAGE_INDIRECT }, 2).reason,
    'unaligned-or-invalid-offset'
  );
  assert.equal(
    validateIndirectCommand('drawIndirect', { size: 16, usage: GPU_BUFFER_USAGE_INDIRECT }, 4).reason,
    'offset-out-of-bounds'
  );
  assert.equal(
    validateIndirectCommand('drawIndexedIndirect', { size: 16, usage: GPU_BUFFER_USAGE_INDIRECT }, 0).reason,
    'offset-out-of-bounds'
  );
  assert.equal(
    validateIndirectCommand('dispatchIndirect', { size: 12, usage: GPU_BUFFER_USAGE_INDIRECT }, 4).reason,
    'offset-out-of-bounds'
  );
  assert.equal(validateIndirectCommand('drawIndirect', { size: 16, usage: GPU_BUFFER_USAGE_INDIRECT }, 0).valid, true);
  assert.equal(validateIndirectCommand('drawIndexedIndirect', { size: 20, usage: GPU_BUFFER_USAGE_INDIRECT }, 0).valid, true);
  assert.equal(validateIndirectCommand('dispatchIndirect', { size: 12, usage: GPU_BUFFER_USAGE_INDIRECT }, 0).valid, true);

  // End-to-end stream verification on Dawn-strict mock:
  const gpu = createDawnStrictMock();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'bgra8unorm' });
  const handler = installRenderStatePackets(processor);

  const key = new Uint8Array(376);
  const kv = new DataView(key.buffer);
  kv.setUint32(20, 1, true);
  key[24] = formatCodeOf('bgra8unorm');
  key[26] = 0x0f;
  kv.setUint32(160, 1, true);
  kv.setUint32(164, 0xffffffff, true);

  const stream = new Stream()
    .shader(1, 0, '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }')
    .shader(2, 1, '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }')
    .shader(3, 2, '@compute @workgroup_size(1) fn cs_main() {}')
    .raw(renderPipeline2Packet(10, 1, 2, 0, key))
    .raw(computePipeline2Packet(11, 3, 0))
    // 20: valid INDIRECT buffer (32 bytes)
    .buffer(20, GPU_BUFFER_USAGE_INDIRECT | 0x0008, 32)
    // 21: storage-only buffer (lacks INDIRECT!)
    .buffer(21, 0x0080 | 0x0008, 64)
    // 22: valid index buffer
    .buffer(22, GPU_BUFFER_USAGE_INDEX | 0x0008, 64)
    .packet(OPCODES.BEGIN_FRAME)
    .raw(beginRenderPassPacket({ colors: [{ textureId: SWAPCHAIN_TEXTURE_ID }] }))
    .u32s(OPCODES.SET_PIPELINE, [10])
    // Bad drawIndirect: missing INDIRECT usage
    .indirect(OPCODES.DRAW_INDIRECT, 21, 0)
    // Bad drawIndirect: unaligned offset 2
    .indirect(OPCODES.DRAW_INDIRECT, 20, 2)
    // Bad drawIndirect: OOB (20 + 16 = 36 > 32)
    .indirect(OPCODES.DRAW_INDIRECT, 20, 20)
    // Bad drawIndexedIndirect: no index buffer bound yet!
    .indirect(OPCODES.DRAW_INDEXED_INDIRECT, 20, 0)
    // Good drawIndirect: 16 + 16 == 32
    .indirect(OPCODES.DRAW_INDIRECT, 20, 16)
    // Bind index buffer, then bad OOB (16 + 20 = 36 > 32) and good (12 + 20 == 32)
    .u32s(OPCODES.SET_INDEX_BUFFER, [22, 0, 0, 0])
    .indirect(OPCODES.DRAW_INDEXED_INDIRECT, 20, 16)
    .indirect(OPCODES.DRAW_INDEXED_INDIRECT, 20, 12)
    .raw(header(V2_OPCODES.END_RENDER_PASS, new Uint8Array(0)))
    .u32s(OPCODES.SET_COMPUTE_PIPELINE, [11])
    // Bad dispatchIndirect: missing INDIRECT, misaligned, OOB (24 + 12 = 36 > 32)
    .indirect(OPCODES.DISPATCH_INDIRECT, 21, 0)
    .indirect(OPCODES.DISPATCH_INDIRECT, 20, 6)
    .indirect(OPCODES.DISPATCH_INDIRECT, 20, 24)
    // Good dispatchIndirect: 20 + 12 == 32
    .indirect(OPCODES.DISPATCH_INDIRECT, 20, 20)
    .packet(OPCODES.PRESENT);

  processor.executeCommandBuffer(stream.bytes());
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.deepEqual(gpu.validationErrors, [], 'zero Dawn validation errors on indirect ops');
  assert.equal(gpu.passCalls.filter((c) => c.op === 'drawIndirect').length, 1);
  assert.equal(gpu.passCalls.filter((c) => c.op === 'drawIndexedIndirect').length, 1);
  assert.equal(gpu.passCalls.filter((c) => c.op === 'dispatchWorkgroupsIndirect').length, 1);
  assert.equal(handler.getStats().skippedInvalidIndirectOps, 8);
});

test('Orch-42 #1941 ExecuteIndirect: explicit-layout SetBindGroup2 validates dynamic offsets & compute pipeline switches reset incompatible groups', () => {
  const gpu = createDawnStrictMock();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'bgra8unorm' });
  const handler = installRenderStatePackets(processor);

  const csIndirect = `
    @group(0) @binding(0) var<storage, read_write> ring: array<u32>;
    @compute @workgroup_size(1) fn cs_main() { ring[0] = 1u; }
  `;
  const csUser = `
    @group(0) @binding(0) var<uniform> u_data: vec4<f32>;
    @compute @workgroup_size(1) fn cs_main() { _ = u_data; }
  `;

  const s = new Stream()
    .shader(1, 2, csIndirect)
    .shader(2, 2, csUser)
    .buffer(10, 0x0040 | 0x0080 | 0x0008, 1024) // 1024B root-constant ring
    // BGL 30: storage + hasDynamicOffset (size 256)
    .raw(bindGroupLayoutPacket(30, [{ binding: 0, visibility: 4, type: 0, subtype: 1, flags: 0x1, minBindingSize: 256 }]))
    // BGL 31: uniform + hasDynamicOffset (user pipeline — distinct BGL!)
    .raw(bindGroupLayoutPacket(31, [{ binding: 0, visibility: 4, type: 0, subtype: 0, flags: 0x1, minBindingSize: 64 }]))
    .raw(pipelineLayoutPacket(40, [30]))
    .raw(pipelineLayoutPacket(41, [31]))
    .raw(computePipeline2Packet(50, 1, 40))
    .raw(computePipeline2Packet(51, 2, 41))
    // BG 60: real ring buffer (offset=0, size=256)
    .raw(bindGroup2Packet(60, 30, [{ binding: 0, type: 0, id: 10, offset: 0, size: 256 }]))
    // BG 61: null resource -> 256B dummy buffer (non-zero dynamic offset MUST be zeroed!)
    .raw(bindGroup2Packet(61, 30, [{ binding: 0, type: 0, id: 0 }]))
    .packet(OPCODES.BEGIN_FRAME)
    .u32s(OPCODES.SET_COMPUTE_PIPELINE, [50])
    // Misaligned dynamic offset 300 -> aligned to 256; OOB offset 2048 -> clamped to 768
    .raw(setBindGroup2Packet(0, 60, [300]))
    .u32s(OPCODES.DISPATCH_COMPUTE, [1, 1, 1])
    .raw(setBindGroup2Packet(0, 60, [2048]))
    .u32s(OPCODES.DISPATCH_COMPUTE, [1, 1, 1])
    // Dummy fallback buffer with dynamic offset 512 -> zeroed to 0
    .raw(setBindGroup2Packet(0, 61, [512]))
    .u32s(OPCODES.DISPATCH_COMPUTE, [1, 1, 1])
    // Switch to user pipeline 51 (expects BGL 31, NOT BGL 30!) WITHOUT binding a BGL-31 group -> dispatch MUST be skipped!
    .u32s(OPCODES.SET_COMPUTE_PIPELINE, [51])
    .u32s(OPCODES.DISPATCH_COMPUTE, [1, 1, 1])
    .packet(OPCODES.PRESENT);

  processor.executeCommandBuffer(s.bytes());
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.deepEqual(gpu.validationErrors, [], 'zero Dawn validation errors on explicit dynamic offsets');

  const setBgs = gpu.passCalls.filter((c) => c.pass === 'compute' && c.op === 'setBindGroup');
  assert.equal(setBgs.length, 3);
  assert.deepEqual(setBgs[0].args[2], [256]);
  assert.deepEqual(setBgs[1].args[2], [768]);
  assert.deepEqual(setBgs[2].args[2], [0]);

  const dispatches = gpu.passCalls.filter((c) => c.op === 'dispatchWorkgroups');
  assert.equal(dispatches.length, 3, 'user compute dispatch after incompatible BGL switch must be skipped');
  assert.equal(handler.getTelemetry().skippedUnboundPassOps, 1);
  assert.equal(handler.getTelemetry().dynamicOffsetClamps, 3);
});

// -----------------------------------------------------------------------------
// 3. CreateTexture2 viewFormats propagation (#1896/#1904) & #1937/#1944 COPY_SRC
// -----------------------------------------------------------------------------

test('Orch-42 #1896/#1904/#1937 CreateTexture2 auto-registers sRGB/linear sibling viewFormats, strips invalid viewFormats, and preserves COPY_SRC on float UAVs', () => {
  const gpu = createDawnStrictMock({ features: ['texture-compression-bc'] });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'bgra8unorm' });
  const handler = installRenderStatePackets(processor);

  const s = new Stream()
    // 100: rgba8unorm-srgb with NO viewFormats specified -> must auto-register 'rgba8unorm' for #1896 PresentX / #1904 copies
    .raw(createTexture2Packet({ id: 100, width: 64, height: 64, format: formatCodeOf('rgba8unorm-srgb'), usage: 0x04 | 0x10 }))
    // 101: bgra8unorm with incompatible viewFormat r32uint -> strips r32uint and registers 'bgra8unorm-srgb'
    .raw(
      createTexture2Packet({
        id: 101,
        width: 64,
        height: 64,
        format: formatCodeOf('bgra8unorm'),
        usage: 0x04 | 0x10,
        viewFormats: [formatCodeOf('r32uint')],
      })
    )
    // 102: compute-only float32 UAV texture (STORAGE_BINDING | TEXTURE_BINDING | COPY_DST, no RENDER_ATTACHMENT) -> auto-adds COPY_SRC (#1937/#1944)
    .raw(createTexture2Packet({ id: 102, width: 64, height: 64, format: formatCodeOf('rgba32float'), usage: 0x04 | 0x08 | 0x02 }))
    // 103: bc7-rgba-unorm-srgb with non-multiple-of-4 dims (10x10) -> falls back to rgba8unorm-srgb + sibling rgba8unorm
    .raw(createTexture2Packet({ id: 103, width: 10, height: 10, format: formatCodeOf('bc7-rgba-unorm-srgb'), usage: 0x04 }))
    // 104: MSAA with STORAGE_BINDING and no RENDER_ATTACHMENT -> strips STORAGE_BINDING and downgrades sampleCount to 1
    .raw(createTexture2Packet({ id: 104, width: 32, height: 32, sampleCount: 4, format: formatCodeOf('rgba8unorm'), usage: 0x04 | 0x08 }))
    // 105: norm16 without texture-formats-tier1 -> falls back to rgba16float
    .raw(createTexture2Packet({ id: 105, width: 32, height: 32, format: formatCodeOf('rgba16unorm'), usage: 0x04 | 0x10 }))
    // Create linear sibling view on 100 via BindGroup2 (exact #1896/#1904 pattern)
    .raw(bindGroupLayoutPacket(1, [{ binding: 0, visibility: 2, type: 2, subtype: 0, dimension: 2 }]))
    .raw(bindGroup2Packet(2, 1, [{ binding: 0, type: 1, id: 100, view: viewDesc({ format: formatCodeOf('rgba8unorm'), dimension: 2 }) }]));

  processor.executeCommandBuffer(s.bytes());
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.deepEqual(gpu.validationErrors, [], 'zero Dawn validation errors on CreateTexture2 & sibling view');

  const tex100 = processor.textures.get(100);
  assert.ok(tex100.viewFormats.includes('rgba8unorm'), 'rgba8unorm-srgb must auto-register rgba8unorm sibling');

  const tex101 = processor.textures.get(101);
  assert.deepEqual(tex101.viewFormats, ['bgra8unorm-srgb'], 'incompatible r32uint stripped and bgra8unorm-srgb added');

  const tex102 = processor.textures.get(102);
  assert.equal(tex102.usage & 0x01, 0x01, 'compute-only float UAV texture must include COPY_SRC for #1937 readback');

  const tex103 = processor.textures.get(103);
  assert.equal(tex103.format, 'rgba8unorm-srgb');
  assert.ok(tex103.viewFormats.includes('rgba8unorm'));

  const tex104 = processor.textures.get(104);
  assert.equal(tex104.sampleCount, 1);
  assert.equal(tex104.usage & 0x08, 0);

  const tex105 = processor.textures.get(105);
  assert.equal(tex105.format, 'rgba16float');

  // Verify resolveCompatibleViewFormats helper directly
  const rvf = resolveCompatibleViewFormats('bc1-rgba-unorm', ['rgba8uint', 'bc1-rgba-unorm']);
  assert.deepEqual(rvf.viewFormats, ['bc1-rgba-unorm', 'bc1-rgba-unorm-srgb']);
  assert.deepEqual(rvf.dropped, ['rgba8uint']);
});

test('Orch-42 #1896 PresentX _attachmentView never creates cross-channel rgba8unorm view on bgra8unorm swapchain', () => {
  const gpu = createDawnStrictMock({ swapchainFormat: 'bgra8unorm' });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'bgra8unorm' });
  const handler = installRenderStatePackets(processor);
  processor._swapchainFormat = 'bgra8unorm';
  processor._swapchainViewFormat = 'bgra8unorm-srgb';

  const resolver = handler._attachmentView(SWAPCHAIN_TEXTURE_ID, { format: 'rgba8unorm' });
  const view = resolver({ swapchainTexture: gpu.swapchainTexture, swapchainView: gpu.swapchainView });
  assert.equal(view, gpu.swapchainView, 'cross-channel swapchain view must fall back to ctl.swapchainView');
  assert.deepEqual(gpu.validationErrors, []);
});

// -----------------------------------------------------------------------------
// 4. #1977 RT Arena & #1892 SM 6.6 Dummy Resource Isolation + Group 3 layout:'auto' guard
// -----------------------------------------------------------------------------

test('Orch-42 #1977/#1892 dummy buffers split rw vs ro, dummy storageTextures are distinct per slot, MSAA dummy has RENDER_ATTACHMENT, and kWholeSize/zero-size handled', () => {
  const gpu = createDawnStrictMock();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'bgra8unorm' });
  const handler = installRenderStatePackets(processor);

  // Mix of null writable storage buffers (RT arena slots 17..19), null read-only-storage (slot 20),
  // null uniform (slot 0), 2 null storageTextures + 1 binding_array of 3 null storageTextures, and 1 null MSAA texture!
  const s = new Stream()
    .buffer(700, 0x0080 | 0x0008, 0) // zero-size buffer -> dummy fallback
    .buffer(701, 0x0080 | 0x0008, 512) // valid buffer with kWholeSize (0xFFFFFFFFFFFFFFFF)
    .raw(
      bindGroupLayoutPacket(
        10,
        [
          { binding: 0, visibility: 4, type: 0, subtype: 0 }, // uniform
          { binding: 1, visibility: 4, type: 0, subtype: 1 }, // storage (rw)
          { binding: 2, visibility: 4, type: 0, subtype: 2 }, // read-only-storage
          { binding: 3, visibility: 4, type: 3, subtype: 0, dimension: 2, storageFormat: formatCodeOf('rgba8unorm') },
          { binding: 4, visibility: 4, type: 3, subtype: 0, dimension: 2, storageFormat: formatCodeOf('rgba8unorm') },
          { binding: 10, visibility: 4, type: 3, subtype: 0, dimension: 2, storageFormat: formatCodeOf('rgba8unorm'), bindingArraySize: 3 },
          { binding: 15, visibility: 4, type: 2, subtype: 1, dimension: 2, flags: 0x2 }, // multisampled texture
          { binding: 16, visibility: 4, type: 0, subtype: 1 }, // zero-size buf test
          { binding: 17, visibility: 4, type: 0, subtype: 1 }, // kWholeSize buf test
        ],
        { flags: BINDING_ARRAYS_LAYOUT_FLAG }
      )
    )
    .raw(
      bindGroup2Packet(
        20,
        10,
        [
          { binding: 0, type: 0, id: 0 },
          { binding: 1, type: 0, id: 0 },
          { binding: 2, type: 0, id: 0 },
          { binding: 3, type: 1, id: 0 },
          { binding: 4, type: 1, id: 0 },
          { binding: 10, arrayIndex: 0, type: 1, id: 0 },
          { binding: 10, arrayIndex: 1, type: 1, id: 0 },
          { binding: 10, arrayIndex: 2, type: 1, id: 0 },
          { binding: 15, type: 1, id: 0 },
          { binding: 16, type: 0, id: 700, offset: 0, size: 64 },
          { binding: 17, type: 0, id: 701, offset: 0, size: 0xffffffffffffffffn },
        ],
        { flags: BINDING_ARRAYS_BG_FLAG }
      )
    );

  processor.executeCommandBuffer(s.bytes());
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.deepEqual(gpu.validationErrors, [], 'zero Dawn aliasing or MSAA dummy validation errors');

  const [bg] = gpu.results('createBindGroup');
  const byBinding = new Map(bg.desc.entries.map((e) => [e.binding, e.resource]));
  // Writable storage buffer (binding 1) MUST NOT alias uniform/read-only-storage (bindings 0, 2)
  assert.notEqual(byBinding.get(1).buffer.handle, byBinding.get(0).buffer.handle);
  assert.notEqual(byBinding.get(1).buffer.handle, byBinding.get(2).buffer.handle);
  // All 5 storageTexture slots (3, 4, 10, 11, 12) must have DISTINCT texture handles!
  const stHandles = new Set([3, 4, 10, 11, 12].map((b) => byBinding.get(b).textureHandle));
  assert.equal(stHandles.size, 5, 'all 5 null storageTexture bindings must have distinct GPUTextures');
  // kWholeSize binding 17 omits huge size (leaves undefined for whole remaining buffer)
  assert.equal(byBinding.get(17).buffer.handle, processor.buffers.get(701).handle);
  assert.equal('size' in byBinding.get(17), false);
});

// -----------------------------------------------------------------------------
// 5. Dawn-strict CreateSampler2, CreateBindGroupLayout, CreatePipelineLayout, CreateRenderPipeline2,
//    SetViewport f32-edge, & _syncDeviceGeneration layoutsByContent invalidation
// -----------------------------------------------------------------------------

test('Orch-42 CreateSampler2, CreateBindGroupLayout, CreatePipelineLayout, and CreateRenderPipeline2 enforce Dawn rules', () => {
  const gpu = createDawnStrictMock();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'bgra8unorm' });
  const handler = installRenderStatePackets(processor);

  // Build render pipeline key with:
  // - target 0: rgba8unorm with min blend op and non-'one' factors
  // - target 1: rgba8unorm with dual-source blend factor src1 (unsupported without feature & MRT)
  // - depthStencil: depth32float-stencil8 (unsupported -> depth24plus-stencil8) + depthBias on line-list
  // - primitive: line-list + unclippedDepth=1 (unsupported without depth-clip-control)
  // - multisample: count=2 (invalid -> 1), alphaToCoverageEnabled=1 (invalid at count=1 -> false)
  const key = new Uint8Array(376);
  const kv = new DataView(key.buffer);
  kv.setUint32(20, 2, true); // 2 targets
  // target 0: rgba8unorm, blendEnable=1, writeMask=0xf, color=(min, zero, src-alpha), alpha=(max, dst, zero)
  key[24] = formatCodeOf('rgba8unorm');
  key[25] = 1;
  key[26] = 0x0f;
  key.set([3, 0, 4], 28); // min, zero, src-alpha
  key.set([4, 6, 0], 32); // max, dst, zero
  // target 1: rgba8unorm, blendEnable=1, color=(add, src1=13, zero)
  key[36] = formatCodeOf('rgba8unorm');
  key[37] = 1;
  key[38] = 0x0f;
  key.set([0, 13, 0], 40);
  key.set([0, 1, 0], 44);
  // depthStencil at offset 120: depth32float-stencil8, depthWrite=1, depthCompare=less(2), depthBias=4
  key[120] = formatCodeOf('depth32float-stencil8');
  key[121] = 1;
  key[122] = 2;
  kv.setInt32(140, 4, true);
  kv.setFloat32(144, 1.5, true);
  kv.setFloat32(148, 0.5, true);
  // primitive at offset 152: line-list (2), no strip, ccw, none, unclippedDepth=1
  key.set([2, 0, 0, 0, 1], 152);
  // multisample at offset 160: count=2, mask=0xffffffff, alphaToCoverage=1
  kv.setUint32(160, 2, true);
  kv.setUint32(164, 0xffffffff, true);
  key[168] = 1;

  const s = new Stream()
    .shader(1, 0, '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }')
    .shader(2, 1, '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }')
    // Sampler with nearest mipmap + maxAnisotropy=8 + negative lodMinClamp + lodMaxClamp < lodMinClamp
    .raw(createSampler2Packet(1, { magFilter: 1, minFilter: 1, mipmapFilter: 0, maxAnisotropy: 8, lodMinClamp: -5, lodMaxClamp: -10 }))
    // BGL with read-write rgba8unorm (unsupported without tier2 -> write-only) + multisampled float 2d-array (coerced to unfilterable-float 2d)
    .raw(
      bindGroupLayoutPacket(
        2,
        [
          { binding: 0, visibility: 4, type: 3, subtype: 2, dimension: 2, storageFormat: formatCodeOf('rgba8unorm') },
          { binding: 1, visibility: 2, type: 2, subtype: 0, dimension: 3, flags: 0x2 },
        ],
        { contentId: 0x999n }
      )
    )
    // PipelineLayout with 6 groups (> maxBindGroups=4 -> clamped to 4)
    .raw(pipelineLayoutPacket(3, [2, 0, 0, 0, 0, 0]))
    .raw(renderPipeline2Packet(4, 1, 2, 0, key));

  processor.executeCommandBuffer(s.bytes());
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.deepEqual(gpu.validationErrors, [], 'zero Dawn validation errors across sampler/BGL/PL/PSO');

  const [smp] = gpu.results('createSampler');
  assert.equal(smp.desc.maxAnisotropy, 1);
  assert.equal(smp.desc.lodMinClamp, 0);
  assert.equal(smp.desc.lodMaxClamp, 0);

  const [bgl] = gpu.results('createBindGroupLayout');
  assert.equal(bgl.desc.entries[0].storageTexture.access, 'write-only');
  assert.equal(bgl.desc.entries[1].texture.viewDimension, '2d');
  assert.equal(bgl.desc.entries[1].texture.sampleType, 'unfilterable-float');

  const [pl] = gpu.results('createPipelineLayout');
  assert.equal(pl.desc.bindGroupLayouts.length, 4);

  const [pso] = gpu.results('createRenderPipeline');
  assert.deepEqual(pso.desc.fragment.targets[0].blend, {
    color: { operation: 'min', srcFactor: 'one', dstFactor: 'one' },
    alpha: { operation: 'max', srcFactor: 'one', dstFactor: 'one' },
  });
  assert.equal('blend' in pso.desc.fragment.targets[1], false, 'dual-source blend stripped');
  assert.equal(pso.desc.depthStencil.format, 'depth24plus-stencil8');
  assert.equal(pso.desc.depthStencil.depthBias, 0);
  assert.equal(pso.desc.depthStencil.depthBiasSlopeScale, 0);
  assert.equal(pso.desc.depthStencil.depthBiasClamp, 0);
  assert.equal('unclippedDepth' in pso.desc.primitive, false);
  assert.equal(pso.desc.multisample.count, 1);
  assert.equal(pso.desc.multisample.alphaToCoverageEnabled, false);

  // _syncDeviceGeneration clears layoutsByContent on device recovery
  assert.ok(handler.layoutsByContent.has('bgl:999'));
  processor._deviceGeneration = (processor._deviceGeneration || 0) + 1;
  handler._syncDeviceGeneration();
  assert.equal(handler.layoutsByContent.size, 0, 'layoutsByContent must clear on device generation bump');

  // Pure helper checks: clampViewportToTarget fractional targetSize, validateVertexBufferLayouts, parseWgslFragmentOutputLocations
  const vpFrac = clampViewportToTarget({ x: 0, y: 0, width: 1280.4, height: 720.4, minDepth: 0, maxDepth: 1 }, { width: 1280.5, height: 720.5 });
  assert.equal(vpFrac.width, 1280);
  assert.equal(vpFrac.height, 720);

  assert.equal(
    validateVertexBufferLayouts([{ arrayStride: 4096, stepMode: 'vertex', attributes: [] }]).valid,
    false
  );
  assert.equal(
    validateVertexBufferLayouts([
      { arrayStride: 16, stepMode: 'vertex', attributes: [{ format: 'float32x2', offset: 2, shaderLocation: 0 }] },
    ]).valid,
    false
  );

  const outLocs = parseWgslFragmentOutputLocations(`
    struct FsOut { @location(0) albedo: vec4<f32>, @location(2) normal: vec4<f32> };
    @fragment fn fs_main() -> FsOut { var o: FsOut; return o; }
  `);
  assert.deepEqual([...outLocs].sort(), [0, 2]);
  assert.deepEqual(sanitizeSamplerDescriptor({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', maxAnisotropy: 32, lodMinClamp: Number.NaN, lodMaxClamp: 8 }), {
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    maxAnisotropy: 16,
    lodMinClamp: 0,
    lodMaxClamp: 8,
  });
  assert.equal(
    sanitizeColorTargetBlendState(
      { format: 'rgba8unorm', blend: { color: { operation: 'min', srcFactor: 'src', dstFactor: 'dst' }, alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'zero' } } },
      null
    ).target.blend.color.srcFactor,
    'one'
  );
});
