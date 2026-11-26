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
    return this.u32s(OPCODES.CREATE_BUFFER, [id, usage, size, 0]);
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
  v.setUint16(8, baseLayer, true);
  v.setUint16(10, layerCount, true);
  return b;
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

function createComputePipeline2Packet(id, csId, layoutId = 0) {
  const b = new Uint8Array(32);
  const v = new DataView(b.buffer);
  v.setUint32(0, id, true);
  v.setUint32(4, csId, true);
  v.setUint32(8, layoutId, true);
  return header(V2_OPCODES.CREATE_COMPUTE_PIPELINE2, b);
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

function createTexture2Packet({
  id,
  width = 16,
  height = 16,
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

function createSampler2Packet(id, { magFilter = 0, minFilter = 0, mipmapFilter = 0, compare = 0, maxAnisotropy = 1 } = {}) {
  const b = new Uint8Array(24);
  const v = new DataView(b.buffer);
  v.setUint32(0, id, true);
  b[7] = magFilter;
  b[8] = minFilter;
  b[9] = mipmapFilter;
  b[10] = compare;
  v.setUint16(12, maxAnisotropy, true);
  v.setFloat32(16, 0, true);
  v.setFloat32(20, 32, true);
  return header(V2_OPCODES.CREATE_SAMPLER2, b);
}

function buildMinimalRenderKey() {
  const key = new Uint8Array(376);
  const v = new DataView(key.buffer);
  key[0] = 0; // triangle-list
  key[5] = 1; // sampleCount = 1
  v.setUint32(8, 0xffffffff, true);
  key[16] = 1; // 1 color target
  key[20] = formatCodeOf('rgba8unorm');
  key[21] = 0xf;
  return key;
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

function beginRenderPassSwapchain() {
  const p = new Uint8Array(368);
  const v = new DataView(p.buffer);
  v.setUint32(0, 1, true);
  v.setUint32(8, SWAPCHAIN_TEXTURE_ID, true);
  p[12] = 2; // clear
  p[13] = 1; // store
  return header(V2_OPCODES.BEGIN_RENDER_PASS, p);
}

// Strict Chrome-Dawn-like WebGPU mock that records any spec validation error on createBindGroup /
// createView / setBindGroup / dispatchWorkgroups / draw!
function createStrictDawnWebGpu({ features = [], opaqueSamplers = false } = {}) {
  const validationErrors = [];
  const calls = [];
  const passCalls = [];
  let nextHandle = 1;
  const samplerMetaMap = new WeakMap();
  const limits = {
    minUniformBufferOffsetAlignment: 256,
    minStorageBufferOffsetAlignment: 256,
    maxUniformBufferBindingSize: 65536,
    maxStorageBufferBindingSize: 134217728,
    maxComputeWorkgroupsPerDimension: 65535,
  };
  const featureSet = new Set(features);

  const swapchainView = { kind: 'GPUTextureView', id: 'swapchain-view' };
  const ctx = {
    getCurrentTexture() {
      return { width: 640, height: 360, createView: () => swapchainView };
    },
  };

  const makePass = (kind) => {
    const boundGroups = new Map();
    let activePipeline = null;
    return {
      setPipeline(p) {
        activePipeline = p;
        passCalls.push({ pass: kind, op: 'setPipeline', args: [p] });
      },
      setBindGroup(i, bg, offsets) {
        if (!bg) {
          validationErrors.push(`${kind}.setBindGroup(${i}): null bindGroup`);
        }
        if (bg && bg._layout) {
          const dynCount = bg._layout.entries.filter((e) => e.buffer && e.buffer.hasDynamicOffset).length;
          const gotCount = Array.isArray(offsets) || ArrayBuffer.isView(offsets) ? offsets.length : 0;
          if (gotCount !== dynCount) {
            validationErrors.push(`${kind}.setBindGroup(${i}): dynamicOffset count ${gotCount} != layout ${dynCount}`);
          }
        }
        boundGroups.set(i, bg);
        passCalls.push({ pass: kind, op: 'setBindGroup', args: offsets === undefined ? [i, bg] : [i, bg, [...offsets]] });
      },
      setViewport(...a) { passCalls.push({ pass: kind, op: 'setViewport', args: a }); },
      setScissorRect(...a) { passCalls.push({ pass: kind, op: 'setScissorRect', args: a }); },
      setBlendConstant(...a) { passCalls.push({ pass: kind, op: 'setBlendConstant', args: a }); },
      setStencilReference(...a) { passCalls.push({ pass: kind, op: 'setStencilReference', args: a }); },
      setVertexBuffer(...a) { passCalls.push({ pass: kind, op: 'setVertexBuffer', args: a }); },
      setIndexBuffer(...a) { passCalls.push({ pass: kind, op: 'setIndexBuffer', args: a }); },
      draw(...a) {
        if (activePipeline && activePipeline._requiredGroups) {
          for (const g of activePipeline._requiredGroups) {
            if (!boundGroups.has(g)) {
              validationErrors.push(`No bind group set at group index ${g} during draw`);
            }
          }
        }
        passCalls.push({ pass: kind, op: 'draw', args: a });
      },
      drawIndexed(...a) { passCalls.push({ pass: kind, op: 'drawIndexed', args: a }); },
      dispatchWorkgroups(...a) {
        if (activePipeline && activePipeline._requiredGroups) {
          for (const g of activePipeline._requiredGroups) {
            if (!boundGroups.has(g)) {
              validationErrors.push(`No bind group set at group index ${g} during DispatchWorkgroups(${a.join(', ')})`);
            }
          }
        }
        passCalls.push({ pass: kind, op: 'dispatchWorkgroups', args: a });
      },
      dispatchWorkgroupsIndirect(...a) {
        if (activePipeline && activePipeline._requiredGroups) {
          for (const g of activePipeline._requiredGroups) {
            if (!boundGroups.has(g)) {
              validationErrors.push(`No bind group set at group index ${g} during DispatchWorkgroupsIndirect`);
            }
          }
        }
        passCalls.push({ pass: kind, op: 'dispatchWorkgroupsIndirect', args: a });
      },
      end() { passCalls.push({ pass: kind, op: 'end', args: [] }); },
    };
  };

  const device = {
    features: featureSet,
    limits,
    createShaderModule(desc) {
      const result = { kind: 'GPUShaderModule', handle: nextHandle++, desc };
      calls.push({ op: 'createShaderModule', desc, result });
      return result;
    },
    createBuffer(desc) {
      if (!Number.isFinite(desc.size) || desc.size <= 0) {
        validationErrors.push(`createBuffer: invalid size ${desc.size}`);
      }
      const result = {
        kind: 'GPUBuffer',
        handle: nextHandle++,
        size: desc.size,
        usage: desc.usage,
        desc,
        destroyed: false,
        destroy() { this.destroyed = true; },
      };
      calls.push({ op: 'createBuffer', desc, result });
      return result;
    },
    createTexture(desc) {
      const dim = desc.dimension || '2d';
      const layers = (desc.size && desc.size.depthOrArrayLayers) || 1;
      const mips = desc.mipLevelCount || 1;
      const sampleCount = desc.sampleCount || 1;
      const width = desc.size.width;
      const height = desc.size.height;
      const result = {
        kind: 'GPUTexture',
        handle: nextHandle++,
        width,
        height,
        depthOrArrayLayers: layers,
        mipLevelCount: mips,
        sampleCount,
        dimension: dim,
        format: desc.format,
        usage: desc.usage,
        desc,
        destroyed: false,
        createView(vd) {
          const effDim = (vd && vd.dimension) || (dim === '2d' && layers > 1 ? '2d-array' : dim);
          const effAspect = (vd && vd.aspect) || 'all';
          const effFormat = (vd && vd.format) || desc.format;
          const effMips = (vd && vd.mipLevelCount) || mips;
          if (dim === '2d' && (effDim === '1d' || effDim === '3d')) {
            validationErrors.push(`createView: incompatible dim ${effDim} on ${dim} texture`);
          }
          if ((effDim === 'cube' || effDim === 'cube-array') && (layers < 6 || width !== height)) {
            validationErrors.push(`createView: cube view requires layers>=6 & square (${width}x${height}x${layers})`);
          }
          const view = {
            kind: 'GPUTextureView',
            handle: nextHandle++,
            textureHandle: result.handle,
            format: effFormat,
            dimension: effDim,
            aspect: effAspect,
            mipLevelCount: effMips,
            sampleCount,
            usage: desc.usage,
            viewDesc: vd ? { ...vd } : undefined,
          };
          calls.push({ op: 'texture.createView', desc: vd, result: view });
          return view;
        },
        destroy() { this.destroyed = true; },
      };
      calls.push({ op: 'createTexture', desc, result });
      return result;
    },
    createSampler(desc = {}) {
      const result = opaqueSamplers
        ? { kind: 'GPUSampler', handle: nextHandle++ }
        : { kind: 'GPUSampler', handle: nextHandle++, desc };
      samplerMetaMap.set(result, {
        compare: desc.compare !== undefined,
        filtering:
          desc.magFilter === 'linear' ||
          desc.minFilter === 'linear' ||
          desc.mipmapFilter === 'linear' ||
          (desc.maxAnisotropy || 1) > 1,
      });
      calls.push({ op: 'createSampler', desc, result });
      return result;
    },
    createBindGroupLayout(desc) {
      const result = { kind: 'GPUBindGroupLayout', handle: nextHandle++, entries: desc.entries };
      calls.push({ op: 'createBindGroupLayout', desc, result });
      return result;
    },
    createPipelineLayout(desc) {
      const requiredGroups = [];
      (desc.bindGroupLayouts || []).forEach((bgl, idx) => {
        if (bgl && Array.isArray(bgl.entries) && bgl.entries.length > 0) requiredGroups.push(idx);
      });
      const result = { kind: 'GPUPipelineLayout', handle: nextHandle++, desc, requiredGroups };
      calls.push({ op: 'createPipelineLayout', desc, result });
      return result;
    },
    createRenderPipeline(desc) {
      const result = {
        kind: 'GPURenderPipeline',
        handle: nextHandle++,
        desc,
        _requiredGroups: desc.layout && desc.layout.requiredGroups ? desc.layout.requiredGroups : [],
      };
      calls.push({ op: 'createRenderPipeline', desc, result });
      return result;
    },
    createComputePipeline(desc) {
      const result = {
        kind: 'GPUComputePipeline',
        handle: nextHandle++,
        desc,
        _requiredGroups: desc.layout && desc.layout.requiredGroups ? desc.layout.requiredGroups : [],
      };
      calls.push({ op: 'createComputePipeline', desc, result });
      return result;
    },
    createBindGroup(desc) {
      const layoutEntries = (desc.layout && desc.layout.entries) || [];
      const expectedSlots = new Map();
      for (const le of layoutEntries) {
        if (le.bindingArraySize >= 1) {
          for (let ai = 0; ai < le.bindingArraySize; ai += 1) {
            expectedSlots.set(le.binding + ai, le);
          }
        } else {
          expectedSlots.set(le.binding, le);
        }
      }
      if (desc.entries.length !== expectedSlots.size) {
        validationErrors.push(
          `createBindGroup: entry count ${desc.entries.length} != layout expected ${expectedSlots.size}`
        );
      }
      for (const e of desc.entries) {
        const le = expectedSlots.get(e.binding);
        if (!le) {
          validationErrors.push(`createBindGroup: unexpected binding ${e.binding}`);
          continue;
        }
        const r = e.resource;
        if (le.buffer) {
          if (!r || !r.buffer || r.buffer.destroyed) {
            validationErrors.push(`binding ${e.binding}: missing or destroyed buffer`);
            continue;
          }
          const isUniform = (le.buffer.type || 'uniform') === 'uniform';
          const reqUsage = isUniform ? 0x40 : 0x80;
          if ((r.buffer.usage & reqUsage) === 0) {
            validationErrors.push(`binding ${e.binding}: buffer lacks required usage 0x${reqUsage.toString(16)}`);
          }
          const off = r.offset || 0;
          const align = isUniform ? limits.minUniformBufferOffsetAlignment : limits.minStorageBufferOffsetAlignment;
          if (off % align !== 0) {
            validationErrors.push(`binding ${e.binding}: offset ${off} not aligned to ${align}`);
          }
          const boundSize = r.size !== undefined ? r.size : r.buffer.size - off;
          if (off >= r.buffer.size || off + boundSize > r.buffer.size) {
            validationErrors.push(`binding ${e.binding}: range [${off}, ${off + boundSize}) exceeds buffer size ${r.buffer.size}`);
          }
          const maxSize = isUniform ? limits.maxUniformBufferBindingSize : limits.maxStorageBufferBindingSize;
          if (boundSize > maxSize) {
            validationErrors.push(`binding ${e.binding}: size ${boundSize} > max ${maxSize}`);
          }
          if (!isUniform && (boundSize < 4 || boundSize % 4 !== 0)) {
            validationErrors.push(`binding ${e.binding}: storage size ${boundSize} must be >=4 and multiple of 4`);
          }
          if (le.buffer.minBindingSize && boundSize < le.buffer.minBindingSize) {
            validationErrors.push(`binding ${e.binding}: size ${boundSize} < minBindingSize ${le.buffer.minBindingSize}`);
          }
        } else if (le.texture) {
          if (!r || r.kind !== 'GPUTextureView') {
            validationErrors.push(`binding ${e.binding}: expected GPUTextureView`);
            continue;
          }
          if ((r.usage & 0x04) === 0) {
            validationErrors.push(`binding ${e.binding}: texture lacks TEXTURE_BINDING`);
          }
          if ((r.format === 'depth24plus-stencil8' || r.format === 'depth32float-stencil8') && r.aspect === 'all') {
            validationErrors.push(`binding ${e.binding}: combined depth-stencil view requires depth-only or stencil-only aspect`);
          }
          if (r.dimension !== (le.texture.viewDimension || '2d')) {
            validationErrors.push(`binding ${e.binding}: view dim ${r.dimension} != layout ${le.texture.viewDimension || '2d'}`);
          }
          if ((r.sampleCount > 1) !== Boolean(le.texture.multisampled)) {
            validationErrors.push(`binding ${e.binding}: multisampled mismatch`);
          }
          const effFmt = rs.resolveEffectiveViewFormat(r.format, r.aspect);
          if (!rs.isTextureSampleTypeCompatible(effFmt, le.texture.sampleType || 'float', device)) {
            validationErrors.push(`binding ${e.binding}: format ${effFmt} incompatible with sampleType ${le.texture.sampleType}`);
          }
        } else if (le.storageTexture) {
          if (!r || r.kind !== 'GPUTextureView') {
            validationErrors.push(`binding ${e.binding}: expected storage GPUTextureView`);
            continue;
          }
          if ((r.usage & 0x08) === 0) {
            validationErrors.push(`binding ${e.binding}: storageTexture lacks STORAGE_BINDING`);
          }
          if (r.sampleCount > 1) {
            validationErrors.push(`binding ${e.binding}: storageTexture cannot be multisampled`);
          }
          if (r.mipLevelCount !== 1) {
            validationErrors.push(`binding ${e.binding}: storageTexture mipLevelCount ${r.mipLevelCount} != 1`);
          }
          if (r.format !== le.storageTexture.format) {
            validationErrors.push(`binding ${e.binding}: storage format ${r.format} != layout ${le.storageTexture.format}`);
          }
          if (r.dimension !== (le.storageTexture.viewDimension || '2d')) {
            validationErrors.push(`binding ${e.binding}: storage dim ${r.dimension} != layout ${le.storageTexture.viewDimension || '2d'}`);
          }
        } else if (le.sampler) {
          if (!r || r.kind !== 'GPUSampler') {
            validationErrors.push(`binding ${e.binding}: expected GPUSampler`);
            continue;
          }
          const sm = samplerMetaMap.get(r);
          const lType = le.sampler.type || 'filtering';
          if (sm) {
            if (lType === 'comparison' && !sm.compare) {
              validationErrors.push(`binding ${e.binding}: non-comparison sampler bound to comparison layout`);
            }
            if (lType !== 'comparison' && sm.compare) {
              validationErrors.push(`binding ${e.binding}: comparison sampler bound to ${lType} layout`);
            }
            if (lType === 'non-filtering' && sm.filtering) {
              validationErrors.push(`binding ${e.binding}: filtering sampler bound to non-filtering layout`);
            }
          }
        }
      }
      const result = { kind: 'GPUBindGroup', handle: nextHandle++, desc, _layout: desc.layout };
      calls.push({ op: 'createBindGroup', desc, result });
      return result;
    },
    createCommandEncoder() {
      return {
        beginRenderPass: (desc) => {
          passCalls.push({ pass: 'render', op: 'beginRenderPass', args: [desc] });
          return makePass('render');
        },
        beginComputePass: (desc) => {
          passCalls.push({ pass: 'compute', op: 'beginComputePass', args: [desc] });
          return makePass('compute');
        },
        copyBufferToBuffer() {},
        copyBufferToTexture() {},
        copyTextureToTexture() {},
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
    results: (op) => calls.filter((c) => c.op === op),
  };
}

test('Orch-02: missing/destroyed resources in CreateBindGroup2 substitute dummies with zero cascade errors', () => {
  const gpu = createStrictDawnWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'rgba8unorm' });
  const handler = installRenderStatePackets(processor);

  const csWgsl = [
    '@group(0) @binding(0) var<storage, read_write> outBuf: array<u32>;',
    '@group(0) @binding(1) var inTex: texture_2d<f32>;',
    '@group(0) @binding(2) var inSmp: sampler;',
    '@group(0) @binding(3) var<uniform> cb0: vec4<f32>;',
    '@compute @workgroup_size(1) fn cs_main() {}',
  ].join('\n');

  const bglEntries = [
    { binding: 0, visibility: 4, type: 0, subtype: 1, minBindingSize: 16 }, // storage
    { binding: 1, visibility: 4, type: 2, subtype: 0, dimension: 2 }, // float 2d
    { binding: 2, visibility: 4, type: 1, subtype: 0 }, // filtering sampler
    { binding: 3, visibility: 4, type: 0, subtype: 0, minBindingSize: 16 }, // uniform
  ];

  // Create buffer 6236 and destroy it before CreateBindGroup2 to test destroyed-resource resilience.
  const destroyedBuf = gpu.device.createBuffer({ size: 256, usage: 0x40 });
  destroyedBuf.destroy();
  processor.buffers.set(6236, destroyedBuf);

  const stream = concat(
    new Stream().shader(10, 2, csWgsl).bytes(),
    bindGroupLayoutPacket(100, bglEntries),
    pipelineLayoutPacket(101, [100]),
    createComputePipeline2Packet(200, 10, 101),
    // Exact Chrome failure pattern: bind group 6332 references unavailable resource 6233 (buffer),
    // unavailable texture 6234, unavailable sampler 6235 (without pipelineCache), and destroyed buffer 6236!
    bindGroup2Packet(6332, 100, [
      { binding: 0, type: 0, id: 6233, offset: 0, size: 64 },
      { binding: 1, type: 1, id: 6234 },
      { binding: 2, type: 2, id: 6235 },
      { binding: 3, type: 0, id: 6236, offset: 0, size: 64 },
    ]),
    // Second bind group repeating missing resource 6233 verifies warning deduplication.
    bindGroup2Packet(6333, 100, [
      { binding: 0, type: 0, id: 6233, offset: 0, size: 64 },
      { binding: 1, type: 1, id: 6234 },
      { binding: 2, type: 2, id: 6235 },
      { binding: 3, type: 0, id: 6236, offset: 0, size: 64 },
    ]),
    new Stream().packet(OPCODES.BEGIN_FRAME).u32s(OPCODES.SET_COMPUTE_PIPELINE, [200]).bytes(),
    setBindGroup2Packet(0, 6332, []),
    new Stream().u32s(OPCODES.DISPATCH_COMPUTE, [5, 5, 33]).packet(OPCODES.PRESENT).bytes()
  );

  processor.executeCommandBuffer(stream);
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.deepEqual(gpu.validationErrors, [], 'zero WebGPU validation errors must occur');
  assert.equal(handler.getStats().missingResourceFallbacks, 8);
  assert.equal(handler.getTelemetry().missingResourceFallbacks, 8);
  assert.equal(handler.getStats().computeDispatchCount, 1);

  const dispatches = gpu.passCalls.filter((c) => c.op === 'dispatchWorkgroups');
  assert.equal(dispatches.length, 1);
  assert.deepEqual(dispatches[0].args, [5, 5, 33]);

  const missingBufWarnings = handler
    .getTelemetry()
    .warnings.filter((w) => w.includes('buffer 6233 unavailable'));
  assert.equal(missingBufWarnings.length, 1, 'missing resource warning must be deduplicated');
});

test('Orch-02: failed CreateBindGroup2 (unknown layout) prevents unbound DispatchCompute and Draw cascade', () => {
  const gpu = createStrictDawnWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'rgba8unorm' });
  const handler = installRenderStatePackets(processor);

  const csWgsl = '@group(0) @binding(0) var<uniform> cb0: vec4<f32>; @compute @workgroup_size(1) fn cs_main() {}';
  const vsWgsl = '@group(0) @binding(0) var<uniform> cb0: vec4<f32>; @vertex fn vs_main() -> @builtin(position) vec4<f32> { return cb0; }';
  const fsWgsl = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }';

  const setup = concat(
    new Stream()
      .shader(1, 2, csWgsl)
      .shader(2, 0, vsWgsl)
      .shader(3, 1, fsWgsl)
      .buffer(10, 0x40 | 0x08, 256)
      .bytes(),
    bindGroupLayoutPacket(20, [{ binding: 0, visibility: 7, type: 0, subtype: 0, minBindingSize: 16 }]),
    pipelineLayoutPacket(21, [20]),
    createComputePipeline2Packet(30, 1, 21),
    renderPipeline2Packet(31, 2, 3, 21, buildMinimalRenderKey()),
    bindGroup2Packet(40, 20, [{ binding: 0, type: 0, id: 10, offset: 0, size: 64 }]), // valid bind group
    bindGroup2Packet(6332, 9999, []) // unknown layout -> genuinely fails!
  );
  processor.executeCommandBuffer(setup);
  assert.equal(handler.stats.errors, 1);
  assert.match(handler.lastError, /unknown bind group layout 9999/);

  const frame = concat(
    new Stream().packet(OPCODES.BEGIN_FRAME).u32s(OPCODES.SET_COMPUTE_PIPELINE, [30]).bytes(),
    // 1) Explicit-layout compute dispatch with NO SetBindGroup -> must skip cleanly (zero Dawn errors)
    new Stream().u32s(OPCODES.DISPATCH_COMPUTE, [2, 2, 1]).bytes(),
    // 2) SetBindGroup2 referencing failed bind group 6332 -> subsequent dispatch skipped cleanly
    setBindGroup2Packet(0, 6332, []),
    new Stream().u32s(OPCODES.DISPATCH_COMPUTE, [5, 5, 33]).bytes(),
    // 3) Valid SetBindGroup2(0, 40) recovers the pass -> next dispatch executes normally!
    setBindGroup2Packet(0, 40, []),
    new Stream().u32s(OPCODES.DISPATCH_COMPUTE, [4, 1, 1]).bytes(),
    // 4) Render pass: SetBindGroup2(0, 40) succeeds, then failed SetBindGroup2(0, 6332) invalidates group 0 -> draw skipped!
    beginRenderPassSwapchain(),
    new Stream().u32s(OPCODES.SET_PIPELINE, [31]).bytes(),
    setBindGroup2Packet(0, 40, []),
    new Stream().u32s(OPCODES.DRAW, [3, 1, 0, 0]).bytes(),
    setBindGroup2Packet(0, 6332, []),
    new Stream().u32s(OPCODES.DRAW, [3, 1, 0, 0]).bytes(),
    header(V2_OPCODES.END_RENDER_PASS, new Uint8Array(0)),
    new Stream().packet(OPCODES.PRESENT).bytes()
  );

  processor.executeCommandBuffer(frame);
  assert.deepEqual(gpu.validationErrors, [], 'zero WebGPU validation errors on failed bind group cascade');

  const dispatches = gpu.passCalls.filter((c) => c.op === 'dispatchWorkgroups');
  assert.deepEqual(
    dispatches.map((d) => d.args),
    [[4, 1, 1]],
    'only the dispatch after valid SetBindGroup2(0, 40) must execute'
  );
  const draws = gpu.passCalls.filter((c) => c.op === 'draw');
  assert.equal(draws.length, 1, 'draw after failed SetBindGroup2(0, 6332) must be skipped');
  assert.equal(handler.getStats().skippedUnboundPassOps, 3);
  assert.equal(handler.getStats().skippedComputeDispatchCount, 2);
});

test('Orch-02: buffer alignment, maxBindingSize clamp, and storage 4B alignment guards keep real valid buffers intact', () => {
  const gpu = createStrictDawnWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'rgba8unorm' });
  const handler = installRenderStatePackets(processor);

  const bglEntries = [
    { binding: 0, visibility: 4, type: 0, subtype: 0, minBindingSize: 16 }, // uniform: unaligned offset 64 -> dummy
    { binding: 1, visibility: 4, type: 0, subtype: 0, minBindingSize: 16 }, // uniform: 128 KiB buffer size=0 -> clamp to 65536 (real!)
    { binding: 2, visibility: 4, type: 0, subtype: 1, minBindingSize: 16 }, // storage: size 18 on 256B -> aligned to 20 (real!)
    { binding: 3, visibility: 4, type: 0, subtype: 1, minBindingSize: 0 }, // storage: 2B buffer -> dummy (< 4B)
    { binding: 4, visibility: 4, type: 0, subtype: 0, minBindingSize: 16 }, // valid tight 16B uniform -> real untouched!
  ];

  const stream = concat(
    new Stream()
      .buffer(1, 0x40 | 0x08, 1024)
      .buffer(2, 0x40 | 0x08, 131072) // 128 KiB > 65536 maxUniformBufferBindingSize
      .buffer(3, 0x80 | 0x08, 256)
      .buffer(4, 0x80 | 0x08, 2) // < 4B
      .buffer(5, 0x40 | 0x08, 16) // valid tight 16B CBV
      .bytes(),
    bindGroupLayoutPacket(10, bglEntries),
    bindGroup2Packet(20, 10, [
      { binding: 0, type: 0, id: 1, offset: 64, size: 64 }, // unaligned offset!
      { binding: 1, type: 0, id: 2, offset: 0, size: 0 }, // oversized whole-buffer uniform!
      { binding: 2, type: 0, id: 3, offset: 0, size: 18 }, // non-4B storage size!
      { binding: 3, type: 0, id: 4, offset: 0, size: 2 }, // < 4B storage size!
      { binding: 4, type: 0, id: 5, offset: 0, size: 16 }, // valid tight CBV!
    ])
  );

  processor.executeCommandBuffer(stream);
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.deepEqual(gpu.validationErrors, [], 'zero WebGPU validation errors on buffer edge cases');

  const [bg] = gpu.results('createBindGroup');
  assert.notEqual(bg.desc.entries[0].resource.buffer, processor.buffers.get(1), 'unaligned offset uses dummy');
  assert.equal(bg.desc.entries[1].resource.buffer, processor.buffers.get(2), 'oversized uniform preserves real buffer');
  assert.equal(bg.desc.entries[1].resource.size, 65536, 'oversized uniform clamped to maxUniformBufferBindingSize');
  assert.equal(bg.desc.entries[2].resource.buffer, processor.buffers.get(3), 'unaligned storage size preserves real buffer');
  assert.equal(bg.desc.entries[2].resource.size, 20, 'storage size aligned up to 4B multiple within remaining buffer');
  assert.notEqual(bg.desc.entries[3].resource.buffer, processor.buffers.get(4), '<4B storage buffer uses dummy');
  assert.equal(bg.desc.entries[4].resource.buffer, processor.buffers.get(5), 'valid 16B uniform buffer preserved');
  assert.equal(bg.desc.entries[4].resource.size, 16);
  assert.equal(handler.getStats().bindGroupValidationFallbacks, 2);

  // Also verify _dummyResource handles undefined minBindingSize without NaN!
  const dummyBuf = handler._dummyResource({ buffer: { type: 'uniform' } });
  assert.equal(dummyBuf.buffer.size, 256);
});

test('Orch-02: sampled & storage texture sampleType/aspect/dimension/multisample/usage/format guards (incl. PR #1868 disjoint @binding(64+))', () => {
  const gpu = createStrictDawnWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'rgba8unorm' });
  const handler = installRenderStatePackets(processor);

  const bglEntries = [
    // Sampled texture mismatches & coercions (including PR #1868 disjoint PS offsets 64+)
    { binding: 0, visibility: 6, type: 2, subtype: 0, dimension: 2 }, // float 2d <- r32uint -> dummy
    { binding: 1, visibility: 6, type: 2, subtype: 0, dimension: 2 }, // float 2d <- depth32float -> dummy
    { binding: 2, visibility: 6, type: 2, subtype: 1, dimension: 2 }, // unfilterable-float 2d <- depth32float -> REAL!
    { binding: 3, visibility: 6, type: 2, subtype: 2, dimension: 2 }, // depth 2d <- depth24plus-stencil8 (all) -> REAL depth-only!
    { binding: 4, visibility: 6, type: 2, subtype: 4, dimension: 2 }, // uint 2d <- depth24plus-stencil8 (all) -> REAL stencil-only!
    { binding: 5, visibility: 6, type: 2, subtype: 0, dimension: 2 }, // float 2d <- no TEXTURE_BINDING -> dummy
    { binding: 6, visibility: 6, type: 2, subtype: 0, dimension: 2, flags: 0x2 }, // multisampled 2d <- 1x tex -> dummy
    { binding: 7, visibility: 6, type: 2, subtype: 0, dimension: 4 }, // float cube <- non-square 6-layer 2d -> dummy
    { binding: 64, visibility: 2, type: 2, subtype: 0, dimension: 2 }, // PS disjoint @64 float 2d <- 6-layer 2d -> adapted REAL 2d!
    // Storage texture mismatches
    { binding: 65, visibility: 6, type: 3, subtype: 0, dimension: 2, storageFormat: formatCodeOf('rgba8unorm') }, // <- rgba16float -> dummy (format mismatch!)
    { binding: 66, visibility: 6, type: 3, subtype: 0, dimension: 2, storageFormat: formatCodeOf('rgba8unorm') }, // <- rgba8unorm no STORAGE_BINDING -> dummy
    { binding: 67, visibility: 6, type: 3, subtype: 0, dimension: 2, storageFormat: formatCodeOf('rgba8unorm') }, // <- 3d texture -> dummy (dim mismatch!)
    { binding: 68, visibility: 6, type: 3, subtype: 2, dimension: 2, storageFormat: formatCodeOf('r32uint') }, // read-write r32uint <- REAL r32uint!
  ];

  const stream = concat(
    createTexture2Packet({ id: 100, width: 16, height: 16, format: formatCodeOf('r32uint'), usage: 0x04 | 0x08 }),
    createTexture2Packet({ id: 101, width: 16, height: 16, format: formatCodeOf('depth32float'), usage: 0x04 | 0x10 }),
    createTexture2Packet({ id: 102, width: 16, height: 16, format: formatCodeOf('depth24plus-stencil8'), usage: 0x04 | 0x10 }),
    createTexture2Packet({ id: 103, width: 16, height: 16, format: formatCodeOf('rgba8unorm'), usage: 0x10 }), // no TEXTURE_BINDING or STORAGE_BINDING
    createTexture2Packet({ id: 104, width: 32, height: 16, depthOrArrayLayers: 6, format: formatCodeOf('rgba8unorm'), usage: 0x04 }), // non-square 6-layer 2D
    createTexture2Packet({ id: 105, width: 16, height: 16, format: formatCodeOf('rgba16float'), usage: 0x04 | 0x08 }),
    createTexture2Packet({ id: 106, width: 16, height: 16, depthOrArrayLayers: 4, dimension: 2, format: formatCodeOf('rgba8unorm'), usage: 0x04 | 0x08 }), // 3D storage texture
    bindGroupLayoutPacket(200, bglEntries),
    bindGroup2Packet(300, 200, [
      { binding: 0, type: 1, id: 100 },
      { binding: 1, type: 1, id: 101 },
      { binding: 2, type: 1, id: 101 },
      { binding: 3, type: 1, id: 102 },
      { binding: 4, type: 1, id: 102 },
      { binding: 5, type: 1, id: 103 },
      { binding: 6, type: 1, id: 104 },
      { binding: 7, type: 1, id: 104 },
      { binding: 64, type: 1, id: 104 },
      { binding: 65, type: 1, id: 105 },
      { binding: 66, type: 1, id: 103 },
      { binding: 67, type: 1, id: 106 },
      { binding: 68, type: 1, id: 100 },
    ])
  );

  processor.executeCommandBuffer(stream);
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.deepEqual(gpu.validationErrors, [], 'zero WebGPU validation errors on texture/storageTexture matrix');

  const [bg] = gpu.results('createBindGroup');
  const byBinding = new Map(bg.desc.entries.map((e) => [e.binding, e.resource]));

  // Mismatches substituted dummies (different textureHandle from source textures)
  assert.notEqual(byBinding.get(0).textureHandle, processor.textures.get(100).handle);
  assert.notEqual(byBinding.get(1).textureHandle, processor.textures.get(101).handle);
  assert.notEqual(byBinding.get(5).textureHandle, processor.textures.get(103).handle);
  assert.notEqual(byBinding.get(6).textureHandle, processor.textures.get(104).handle);
  assert.notEqual(byBinding.get(7).textureHandle, processor.textures.get(104).handle);
  assert.notEqual(byBinding.get(65).textureHandle, processor.textures.get(105).handle);
  assert.notEqual(byBinding.get(66).textureHandle, processor.textures.get(103).handle);
  assert.notEqual(byBinding.get(67).textureHandle, processor.textures.get(106).handle);

  // Valid & auto-coerced bindings preserved REAL textures!
  assert.equal(byBinding.get(2).textureHandle, processor.textures.get(101).handle, 'depth32float valid for unfilterable-float');
  assert.equal(byBinding.get(3).textureHandle, processor.textures.get(102).handle, 'combined DS auto-coerced to depth-only');
  assert.equal(byBinding.get(3).aspect, 'depth-only');
  assert.equal(byBinding.get(4).textureHandle, processor.textures.get(102).handle, 'combined DS auto-coerced to stencil-only for uint');
  assert.equal(byBinding.get(4).aspect, 'stencil-only');
  assert.equal(byBinding.get(64).textureHandle, processor.textures.get(104).handle, '6-layer 2d adapted to single-layer 2d view');
  assert.equal(byBinding.get(64).dimension, '2d');
  assert.equal(byBinding.get(68).textureHandle, processor.textures.get(100).handle, 'read-write r32uint real storage view preserved');
});

test('Orch-02: opaque GPUSampler comparison/filtering validation via samplerMeta, omitted/extra bindings, and dynamicOffset normalization', () => {
  // opaqueSamplers: true simulates real browser GPUSampler (no .desc property!)
  const gpu = createStrictDawnWebGpu({ opaqueSamplers: true });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'rgba8unorm' });
  const handler = installRenderStatePackets(processor);

  const bglEntries = [
    { binding: 0, visibility: 6, type: 1, subtype: 0 }, // filtering <- comparison sampler -> fallback!
    { binding: 1, visibility: 6, type: 1, subtype: 2 }, // comparison <- linear sampler -> fallback!
    { binding: 2, visibility: 6, type: 1, subtype: 1 }, // non-filtering <- linear sampler -> fallback!
    { binding: 3, visibility: 6, type: 1, subtype: 1 }, // non-filtering <- nearest sampler -> REAL!
    { binding: 4, visibility: 6, type: 1, subtype: 2 }, // comparison <- comparison sampler -> REAL!
    { binding: 5, visibility: 6, type: 0, subtype: 0, flags: 0x1, minBindingSize: 16 }, // omitted dynamic uniform -> synthesized!
    { binding: 6, visibility: 6, type: 1, subtype: 0 }, // omitted sampler (no pipelineCache!) -> synthesized!
  ];

  const stream = concat(
    createSampler2Packet(10, { magFilter: 1, minFilter: 1 }), // linear filtering
    createSampler2Packet(11, { magFilter: 0, minFilter: 0 }), // nearest non-filtering
    createSampler2Packet(12, { magFilter: 1, minFilter: 1, compare: 2 }), // comparison (less)
    bindGroupLayoutPacket(50, bglEntries),
    pipelineLayoutPacket(51, [50]),
    new Stream().shader(1, 2, '@compute @workgroup_size(1) fn cs_main() {}').bytes(),
    createComputePipeline2Packet(60, 1, 51),
    bindGroup2Packet(70, 50, [
      { binding: 0, type: 2, id: 12 }, // comparison -> filtering (mismatch)
      { binding: 1, type: 2, id: 10 }, // linear -> comparison (mismatch)
      { binding: 2, type: 2, id: 10 }, // linear -> non-filtering (mismatch)
      { binding: 3, type: 2, id: 11 }, // nearest -> non-filtering (valid!)
      { binding: 4, type: 2, id: 12 }, // comparison -> comparison (valid!)
      // bindings 5 and 6 intentionally omitted!
      { binding: 99, type: 2, id: 11 }, // extra undeclared binding -> skipped!
    ]),
    new Stream().packet(OPCODES.BEGIN_FRAME).u32s(OPCODES.SET_COMPUTE_PIPELINE, [60]).bytes(),
    // Layout has 1 dynamic uniform (binding 5), packet passes 0 dynamic offsets -> normalized to [0]!
    setBindGroup2Packet(0, 70, []),
    new Stream().u32s(OPCODES.DISPATCH_COMPUTE, [1, 1, 1]).packet(OPCODES.PRESENT).bytes()
  );

  processor.executeCommandBuffer(stream);
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.deepEqual(gpu.validationErrors, [], 'zero WebGPU validation errors on opaque sampler & omitted/extra binding checks');

  const [bg] = gpu.results('createBindGroup');
  const byBinding = new Map(bg.desc.entries.map((e) => [e.binding, e.resource]));
  assert.equal(bg.desc.entries.length, 7, 'omitted bindings synthesized and extra binding 99 dropped');
  assert.notEqual(byBinding.get(0), processor.samplers.get(12));
  assert.notEqual(byBinding.get(1), processor.samplers.get(10));
  assert.notEqual(byBinding.get(2), processor.samplers.get(10));
  assert.equal(byBinding.get(3), processor.samplers.get(11), 'nearest sampler preserved for non-filtering');
  assert.equal(byBinding.get(4), processor.samplers.get(12), 'comparison sampler preserved for comparison');
  assert.ok(byBinding.get(5) && byBinding.get(5).buffer, 'omitted uniform buffer synthesized');
  assert.ok(byBinding.get(6) && byBinding.get(6).kind === 'GPUSampler', 'omitted sampler synthesized without pipelineCache');

  const setBgCalls = gpu.passCalls.filter((c) => c.op === 'setBindGroup');
  assert.equal(setBgCalls.length, 1);
  assert.deepEqual(setBgCalls[0].args[2], [0], 'dynamic offsets padded to layout expected count 1');
});

test('Orch-02: binding-array bind group synthesizes omitted slots and skips out-of-range arrayIndex', () => {
  const gpu = createStrictDawnWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'rgba8unorm' });
  const handler = installRenderStatePackets(processor);

  const stream = concat(
    createTexture2Packet({ id: 201, width: 16, height: 16, format: formatCodeOf('rgba8unorm'), usage: 0x04 }),
    bindGroupLayoutPacket(
      210,
      [{ binding: 4, visibility: 2, type: 2, subtype: 0, dimension: 2, bindingArraySize: 3 }],
      { flags: BINDING_ARRAYS_LAYOUT_FLAG }
    ),
    bindGroup2Packet(
      220,
      210,
      [
        { binding: 4, type: 1, id: 201, arrayIndex: 0 },
        // arrayIndex 1 omitted, arrayIndex 2 points to missing texture 999, arrayIndex 5 out-of-range!
        { binding: 4, type: 1, id: 999, arrayIndex: 2 },
        { binding: 4, type: 1, id: 201, arrayIndex: 5 },
      ],
      { flags: BINDING_ARRAYS_BG_FLAG }
    )
  );

  processor.executeCommandBuffer(stream);
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.deepEqual(gpu.validationErrors, [], 'zero WebGPU validation errors for binding-array gaps/OOB');

  const [bg] = gpu.results('createBindGroup');
  assert.deepEqual(
    bg.desc.entries.map((e) => e.binding).sort((a, b) => a - b),
    [4, 5, 6]
  );
  assert.equal(handler.getStats().missingResourceFallbacks, 1);
});

test('Orch-02: review edge cases (sampler comparison metadata, OOB arrayIndex non-pollution, 3D/cube-array dim checks, pipeline switch, tier2 read-write)', () => {
  const gpu = createStrictDawnWebGpu({ opaqueSamplers: true });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'rgba8unorm' });
  const handler = installRenderStatePackets(processor);

  // 1) Sampler mismatch must keep true comparison metadata on descriptorResources for layout:'auto' remap
  // 2) OOB arrayIndex (binding 4 + arrayIndex 2 == 6) must NOT pollute seenBindings for valid scalar binding 6
  // 3) 3D texture with explicit viewDesc.dimension:'2d' bound to 2d layout must fall back to dummy (never pass 3D tex!)
  // 4) Square 7-layer 2D texture with viewDesc.dimension:'cube-array' bound to cube-array layout must fall back (7 % 6 !== 0)
  const stream = concat(
    createSampler2Packet(10, { magFilter: 1, minFilter: 1, compare: 2 }), // comparison sampler
    createTexture2Packet({ id: 20, width: 16, height: 16, depthOrArrayLayers: 4, dimension: 2, format: formatCodeOf('rgba8unorm'), usage: 0x04 }), // 3D tex
    createTexture2Packet({ id: 21, width: 16, height: 16, depthOrArrayLayers: 7, dimension: 1, format: formatCodeOf('rgba8unorm'), usage: 0x04 }), // 7-layer 2D tex
    createTexture2Packet({ id: 22, width: 16, height: 16, format: formatCodeOf('rgba8unorm'), usage: 0x04 }), // valid 2D tex
    bindGroupLayoutPacket(
      30,
      [
        { binding: 0, visibility: 6, type: 1, subtype: 0 }, // filtering sampler
        { binding: 1, visibility: 6, type: 2, subtype: 0, dimension: 2 }, // 2d texture
        { binding: 2, visibility: 6, type: 2, subtype: 0, dimension: 5 }, // cube-array texture
        { binding: 4, visibility: 6, type: 2, subtype: 0, dimension: 2, bindingArraySize: 2 }, // slots 4..5
        { binding: 6, visibility: 6, type: 2, subtype: 0, dimension: 2 }, // scalar slot 6
      ],
      { flags: BINDING_ARRAYS_LAYOUT_FLAG }
    ),
    bindGroup2Packet(
      40,
      30,
      [
        { binding: 0, type: 2, id: 10 }, // comparison sampler in filtering slot
        { binding: 1, type: 1, id: 20, view: viewDesc({ dimension: 2 }) }, // 3D tex claiming 2d view!
        { binding: 2, type: 1, id: 21, view: viewDesc({ dimension: 5 }) }, // 7-layer 2D tex claiming cube-array!
        { binding: 4, type: 1, id: 22, arrayIndex: 0 },
        { binding: 4, type: 1, id: 22, arrayIndex: 2 }, // OOB arrayIndex collides with binding 6!
        { binding: 6, type: 1, id: 22, arrayIndex: 0 }, // valid binding 6 must NOT trigger duplicate!
      ],
      { flags: BINDING_ARRAYS_BG_FLAG }
    )
  );
  processor.executeCommandBuffer(stream);
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.deepEqual(gpu.validationErrors, []);

  const bgDesc40 = handler.bindGroupDescriptors.get(40);
  const smpEntry = bgDesc40.resources.find((r) => r.binding === 0);
  assert.equal(smpEntry.comparison, true, 'real comparison sampler must keep comparison:true in descriptorResources');

  const [bg40] = gpu.results('createBindGroup');
  const byB = new Map(bg40.desc.entries.map((e) => [e.binding, e.resource]));
  assert.notEqual(byB.get(1).textureHandle, processor.textures.get(20).handle, '3D texture rejected for 2d layout');
  assert.notEqual(byB.get(2).textureHandle, processor.textures.get(21).handle, '7-layer texture rejected for cube-array');
  assert.equal(byB.get(6).textureHandle, processor.textures.get(22).handle, 'binding 6 unaffected by prior OOB arrayIndex');

  // 5) Pipeline switch to a 0-group explicit-layout pipeline after failed SetBindGroup2(0, unknown) must NOT skip valid dispatch!
  const csNoGroup = '@compute @workgroup_size(1) fn cs_main() {}';
  const frame2 = concat(
    new Stream().shader(50, 2, csNoGroup).bytes(),
    bindGroupLayoutPacket(51, []), // empty group 0
    pipelineLayoutPacket(52, [51]), // known explicit layout requiring 0 groups
    createComputePipeline2Packet(53, 50, 52),
    new Stream().packet(OPCODES.BEGIN_FRAME).u32s(OPCODES.SET_COMPUTE_PIPELINE, [53]).bytes(),
    setBindGroup2Packet(0, 99999, []), // fails, sets invalidComputeGroupsMask bit 0
    new Stream().u32s(OPCODES.DISPATCH_COMPUTE, [3, 1, 1]).packet(OPCODES.PRESENT).bytes()
  );
  processor.executeCommandBuffer(frame2);
  assert.deepEqual(gpu.validationErrors, []);
  const dispatches = gpu.passCalls.filter((c) => c.op === 'dispatchWorkgroups');
  assert.equal(dispatches.length, 1, '0-group explicit compute pipeline must not be falsely skipped by invalid group 0');
  assert.deepEqual(dispatches[0].args, [3, 1, 1]);

  // 6) isStorageAccessCompatible under texture-formats-tier2 rejects *snorm and rg* read-write formats
  const tier2Dev = { features: new Set(['texture-formats-tier2']) };
  assert.equal(rs.isStorageAccessCompatible('r32uint', 'read-write', { features: new Set() }), true);
  assert.equal(rs.isStorageAccessCompatible('rgba8unorm', 'read-write', { features: new Set() }), false);
  assert.equal(rs.isStorageAccessCompatible('rgba8unorm', 'read-write', tier2Dev), true);
  assert.equal(rs.isStorageAccessCompatible('rgba8snorm', 'read-write', tier2Dev), false);
  assert.equal(rs.isStorageAccessCompatible('rg32float', 'read-write', tier2Dev), false);
});

