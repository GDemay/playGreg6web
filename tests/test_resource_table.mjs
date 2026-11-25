import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const rt = require('../web/workers/resource_table.js');
const tp = require('../web/workers/texture_packets.js');
const { GpuResourceRegistry, RESOURCE_KIND } = require('../web/workers/resource_registry.js');

const {
  CREATE_TEXTURE2_OPCODE,
  CREATE_TEXTURE2_PAYLOAD_BYTES,
  MAX_TEXTURE_VIEW_FORMATS,
  GPU_BUFFER_USAGE,
  GPU_TEXTURE_USAGE,
  TEXTURE_FORMAT_NAMES,
  TEXTURE_DIMENSION_NAMES,
  BC_TEXTURE_FORMATS,
  BC_FALLBACK_FORMAT,
  TEXTURE_COMPRESSION_BC_FEATURE,
  BUFFER_ALIGNMENT_BYTES,
  UNIFORM_BUFFER_MIN_BYTES,
  DEFAULT_FAST_SLOT_CAPACITY,
  DEFAULT_MAX_BIND_GROUPS,
  DEFAULT_FAST_BIND_GROUP_BUCKETS,
  isBcFormat,
  deviceSupportsBc,
  computeMaxMipLevels,
  clampMipLevelCount,
  clampSampleCount,
  alignBcDimension,
  alignBufferSize,
  padBufferUploadData,
  decodeCreateTexture2,
  sanitizeTextureDescriptor,
  computeBindGroupHashFast,
  ResourceTable,
} = rt;

const WORKERS_DIR = fileURLToPath(new URL('../web/workers/', import.meta.url));
const repoFile = (rel) => readFileSync(new URL(`../${rel}`, import.meta.url), 'utf8');

function createMockDevice({ features = new Set([TEXTURE_COMPRESSION_BC_FEATURE]) } = {}) {
  const calls = {
    createBuffer: [],
    writeBuffer: [],
    createTexture: [],
    createSampler: [],
    createShaderModule: [],
    createBindGroup: [],
  };
  let nextHandle = 1;

  const device = {
    features,
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
      calls.createBuffer.push({ desc, result: buf });
      return buf;
    },
    createTexture(desc) {
      const tex = {
        kind: 'GPUTexture',
        handle: nextHandle++,
        desc,
        destroyed: false,
        createView(viewDesc) {
          return {
            kind: 'GPUTextureView',
            textureHandle: tex.handle,
            texture: tex,
            viewDesc,
          };
        },
        destroy() {
          tex.destroyed = true;
        },
      };
      calls.createTexture.push({ desc, result: tex });
      return tex;
    },
    createSampler(desc) {
      const sampler = { kind: 'GPUSampler', handle: nextHandle++, desc };
      calls.createSampler.push({ desc, result: sampler });
      return sampler;
    },
    createShaderModule(desc) {
      const mod = { kind: 'GPUShaderModule', handle: nextHandle++, desc };
      calls.createShaderModule.push({ desc, result: mod });
      return mod;
    },
    createBindGroup(desc) {
      // Mirror WebGPU validation: reject any destroyed buffer or texture in entries.
      const entries = Array.isArray(desc.entries) ? desc.entries : [];
      for (let i = 0; i < entries.length; i += 1) {
        const res = entries[i] && entries[i].resource;
        if (!res) {
          throw new Error(`WebGPU validation: bindGroup entry ${i} missing resource`);
        }
        if (res.buffer && res.buffer.destroyed) {
          throw new Error('WebGPU validation: Destroyed buffer used in createBindGroup');
        }
        if (res.destroyed) {
          throw new Error('WebGPU validation: Destroyed resource used in createBindGroup');
        }
        if (res.texture && res.texture.destroyed) {
          throw new Error('WebGPU validation: Destroyed texture used in createBindGroup');
        }
      }
      const bg = {
        kind: 'GPUBindGroup',
        handle: nextHandle++,
        layout: desc.layout,
        entries,
        label: desc.label,
      };
      calls.createBindGroup.push({ desc, result: bg });
      return bg;
    },
    queue: {
      writeBuffer(buffer, bufferOffset, data) {
        // Mirror WebGPU validation requirements on bufferOffset and byteLength.
        if (bufferOffset % 4 !== 0) {
          throw new Error(`WebGPU validation: bufferOffset (${bufferOffset}) must be 4-byte aligned`);
        }
        const byteLength = data.byteLength;
        if (byteLength % 4 !== 0) {
          throw new Error(`WebGPU validation: writeBuffer size (${byteLength}) must be 4-byte aligned`);
        }
        if (bufferOffset + byteLength > buffer.size) {
          throw new Error(
            `WebGPU validation: writeBuffer [${bufferOffset}, ${bufferOffset + byteLength}) exceeds buffer.size (${buffer.size})`
          );
        }
        const snapshot = new Uint8Array(byteLength);
        snapshot.set(new Uint8Array(data.buffer, data.byteOffset, byteLength));
        calls.writeBuffer.push({ buffer, bufferOffset, data: snapshot, byteLength });
      },
    },
  };

  return { device, calls };
}

function buildCreateTexture2Payload({
  id = 1,
  width = 256,
  height = 256,
  layers = 1,
  mips = 1,
  samples = 1,
  usage = GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.COPY_DST,
  dimension = 1, // 2d
  format = 22, // rgba8unorm
  viewFormats = [],
} = {}) {
  const buf = new ArrayBuffer(CREATE_TEXTURE2_PAYLOAD_BYTES);
  const view = new DataView(buf);
  view.setUint32(0, id, true);
  view.setUint32(4, width, true);
  view.setUint32(8, height, true);
  view.setUint32(12, layers, true);
  view.setUint32(16, mips, true);
  view.setUint32(20, samples, true);
  view.setUint32(24, usage, true);
  view.setUint8(28, dimension);
  view.setUint8(29, format);
  view.setUint8(30, viewFormats.length);
  view.setUint8(31, 0);
  for (let i = 0; i < viewFormats.length && i < MAX_TEXTURE_VIEW_FORMATS; i += 1) {
    view.setUint8(32 + i, viewFormats[i]);
  }
  return view;
}

test('UMD/CommonJS/ESM exports and ABI v2 CreateTexture2 constants match playgta6_abi.h', async () => {
  const esmMod = await import('../web/workers/resource_table.js');
  assert.equal(typeof esmMod.ResourceTable, 'function');
  assert.equal(esmMod.ResourceTable, ResourceTable);
  assert.equal(globalThis.PlayGta6ResourceTable.ResourceTable, ResourceTable);
  assert.equal(CREATE_TEXTURE2_OPCODE, 35);
  assert.equal(CREATE_TEXTURE2_PAYLOAD_BYTES, 40);
  assert.equal(MAX_TEXTURE_VIEW_FORMATS, 8);
  assert.equal(BUFFER_ALIGNMENT_BYTES, 4);
  assert.equal(UNIFORM_BUFFER_MIN_BYTES, 16);

  const header = repoFile('runtime/abi/playgta6_abi.h');
  assert.match(header, /CreateTexture2 = 35,/);
  assert.match(header, /sizeof\(GpuCreateTexture2Payload\) == 40/);
  assert.match(header, /kGpuMaxTextureViewFormats = 8/);

  // Format code table stays strictly aligned with texture_packets.js.
  assert.deepEqual([...TEXTURE_FORMAT_NAMES], [...tp.TEXTURE_FORMAT_NAMES]);
  assert.deepEqual([...TEXTURE_DIMENSION_NAMES], ['1d', '2d', '3d']);
  assert.equal(BC_TEXTURE_FORMATS.length, 14);
  assert.equal(BC_FALLBACK_FORMAT, 'rgba8unorm');
});

test('classic-worker importScripts IIFE isolation avoids top-level const collisions (G31)', () => {
  const context = vm.createContext({
    console,
    Uint8Array,
    Int32Array,
    Uint32Array,
    Float32Array,
    DataView,
    ArrayBuffer,
    Map,
    Set,
    WeakSet,
    Promise,
  });
  context.globalThis = context;
  context.self = context;

  const scripts = [
    'texture_formats.js',
    'texture_packets.js',
    'render_state_packets.js',
    'resource_registry.js',
    'resource_table.js',
  ];

  for (const name of scripts) {
    const src = readFileSync(path.join(WORKERS_DIR, name), 'utf8');
    vm.runInContext(src, context, { filename: name });
  }

  assert.equal(typeof context.PlayGta6ResourceTable, 'object');
  assert.equal(typeof context.PlayGta6ResourceTable.ResourceTable, 'function');
  assert.equal(typeof context.PlayGta6TexturePackets.TexturePackets, 'function');
  assert.equal(typeof context.PlayGta6ResourceRegistry.GpuResourceRegistry, 'function');
});

test('mipLevelCount clamping enforces [1, floor(log2(max(width, height))) + 1] and tracks mipClampCount', () => {
  assert.equal(computeMaxMipLevels(1, 1), 1);
  assert.equal(computeMaxMipLevels(4, 4), 3);
  assert.equal(computeMaxMipLevels(256, 128), 9);
  assert.equal(computeMaxMipLevels(3840, 2160), 12);
  assert.equal(computeMaxMipLevels(4096, 4096), 13);
  // 2D array layers do not increase max mip levels; 3D depth does.
  assert.equal(computeMaxMipLevels(64, 32, 16, '2d'), 7);
  assert.equal(computeMaxMipLevels(16, 16, 64, '3d'), 7);

  const { device, calls } = createMockDevice();
  const table = new ResourceTable(device);

  // Valid mip count (no clamp).
  table.createTexture(1, {
    size: { width: 256, height: 128, depthOrArrayLayers: 6 },
    mipLevelCount: 9,
    format: 'rgba8unorm',
    usage: GPU_TEXTURE_USAGE.TEXTURE_BINDING,
  });
  assert.equal(calls.createTexture[0].desc.mipLevelCount, 9);
  assert.equal(table.getStats().mipClampCount, 0);

  // Excessive mip count (20 -> 12 for 4K 3840x2160).
  table.createTexture(2, {
    size: { width: 3840, height: 2160, depthOrArrayLayers: 1 },
    mipLevelCount: 20,
    format: 'bgra8unorm',
    usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT | GPU_TEXTURE_USAGE.TEXTURE_BINDING,
  });
  assert.equal(calls.createTexture[1].desc.mipLevelCount, 12);
  assert.equal(table.getStats().mipClampCount, 1);
  assert.equal(table.getTextureMeta(2).mipClamped, true);

  // Zero / negative mip count (0 -> 1).
  table.createTexture(3, {
    size: { width: 64, height: 64, depthOrArrayLayers: 1 },
    mipLevelCount: 0,
    format: 'rgba8unorm',
    usage: GPU_TEXTURE_USAGE.TEXTURE_BINDING,
  });
  assert.equal(calls.createTexture[2].desc.mipLevelCount, 1);
  assert.equal(table.getStats().mipClampCount, 2);

  // Pure helper validation.
  assert.deepEqual(clampMipLevelCount(15, 32, 16), { mipLevelCount: 6, maxMipLevels: 6, clamped: true });
  assert.deepEqual(clampMipLevelCount(undefined, 32, 16), { mipLevelCount: 1, maxMipLevels: 6, clamped: false });
});

test('sampleCount clamping enforces {1, 4} and falls back to 1 when mipLevelCount > 1 or unsupported', () => {
  const { device, calls } = createMockDevice();
  const table = new ResourceTable(device);

  // 4x MSAA on single-mip 2D render target -> preserved.
  table.createTexture(10, {
    size: { width: 1920, height: 1080, depthOrArrayLayers: 1 },
    mipLevelCount: 1,
    sampleCount: 4,
    format: 'rgba16float',
    usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT,
  });
  assert.equal(calls.createTexture[0].desc.sampleCount, 4);
  assert.equal(table.getStats().sampleClampCount, 0);

  // 8x MSAA on valid 2D target -> clamped to 4.
  table.createTexture(11, {
    size: { width: 640, height: 360, depthOrArrayLayers: 1 },
    mipLevelCount: 1,
    sampleCount: 8,
    format: 'rgba8unorm',
    usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT,
  });
  assert.equal(calls.createTexture[1].desc.sampleCount, 4);
  assert.equal(table.getStats().sampleClampCount, 1);

  // 2x MSAA (unsupported in WebGPU) -> falls back to 1.
  table.createTexture(12, {
    size: { width: 640, height: 360, depthOrArrayLayers: 1 },
    mipLevelCount: 1,
    sampleCount: 2,
    format: 'rgba8unorm',
    usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT,
  });
  assert.equal(calls.createTexture[2].desc.sampleCount, 1);
  assert.equal(table.getStats().sampleClampCount, 2);

  // 4x MSAA with mipLevelCount > 1 -> falls back to 1.
  table.createTexture(13, {
    size: { width: 256, height: 256, depthOrArrayLayers: 1 },
    mipLevelCount: 4,
    sampleCount: 4,
    format: 'rgba8unorm',
    usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT,
  });
  assert.equal(calls.createTexture[3].desc.sampleCount, 1);
  assert.equal(table.getStats().sampleClampCount, 3);

  // 4x MSAA on BC7 compressed texture -> falls back to 1.
  table.createTexture(14, {
    size: { width: 64, height: 64, depthOrArrayLayers: 1 },
    mipLevelCount: 1,
    sampleCount: 4,
    format: 'bc7-rgba-unorm',
    usage: GPU_TEXTURE_USAGE.TEXTURE_BINDING,
  });
  assert.equal(calls.createTexture[4].desc.sampleCount, 1);
  assert.equal(table.getStats().sampleClampCount, 4);

  // 4x MSAA on 3D or 2D array texture -> falls back to 1.
  assert.equal(
    clampSampleCount(4, { dimension: '3d', mipLevelCount: 1, format: 'rgba8unorm' }).sampleCount,
    1
  );
  assert.equal(
    clampSampleCount(4, { dimension: '2d', depthOrArrayLayers: 6, mipLevelCount: 1, format: 'rgba8unorm' }).sampleCount,
    1
  );
});

test('BC1..BC7 formats fall back to rgba8unorm when device.features lacks texture-compression-bc', () => {
  const { device, calls } = createMockDevice({ features: new Set() });
  const table = new ResourceTable(device);

  assert.equal(deviceSupportsBc(device), false);
  assert.equal(isBcFormat('bc1-rgba-unorm'), true);
  assert.equal(isBcFormat('bc7-rgba-unorm-srgb'), true);
  assert.equal(isBcFormat('rgba8unorm'), false);

  for (let i = 0; i < BC_TEXTURE_FORMATS.length; i += 1) {
    const bcFormat = BC_TEXTURE_FORMATS[i];
    table.createTexture(100 + i, {
      size: { width: 13, height: 7, depthOrArrayLayers: 1 },
      mipLevelCount: 1,
      format: bcFormat,
      usage: GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.COPY_DST,
      viewFormats: [bcFormat],
    });
    const createdDesc = calls.createTexture[i].desc;
    assert.equal(createdDesc.format, 'rgba8unorm', `expected fallback for ${bcFormat}`);
    // Dimensions stay unaligned to 4 when falling back to uncompressed rgba8unorm.
    assert.equal(createdDesc.size.width, 13);
    assert.equal(createdDesc.size.height, 7);
    // BC viewFormats are remapped to compatible rgba8unorm / rgba8unorm-srgb.
    const expectedVf = bcFormat.endsWith('-srgb') ? 'rgba8unorm-srgb' : 'rgba8unorm';
    assert.deepEqual(createdDesc.viewFormats, [expectedVf]);
  }

  const stats = table.getStats();
  assert.equal(stats.bcFormatFallbackCount, BC_TEXTURE_FORMATS.length);
  assert.equal(stats.bcDimensionAlignCount, 0);
});

test('BC1..BC7 formats align dimensions up to multiples of 4 when texture-compression-bc is supported', () => {
  const { device, calls } = createMockDevice({
    features: new Set([TEXTURE_COMPRESSION_BC_FEATURE]),
  });
  const table = new ResourceTable(device);

  assert.equal(deviceSupportsBc(device), true);
  assert.equal(alignBcDimension(1), 4);
  assert.equal(alignBcDimension(4), 4);
  assert.equal(alignBcDimension(5), 8);
  assert.equal(alignBcDimension(13), 16);

  // Odd dimensions -> rounded up to multiples of 4 (13x7 -> 16x8).
  table.createTexture(200, {
    size: { width: 13, height: 7, depthOrArrayLayers: 1 },
    mipLevelCount: 1,
    format: 'bc7-rgba-unorm',
    usage: GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.COPY_DST,
  });
  assert.equal(calls.createTexture[0].desc.format, 'bc7-rgba-unorm');
  assert.equal(calls.createTexture[0].desc.size.width, 16);
  assert.equal(calls.createTexture[0].desc.size.height, 8);
  assert.equal(table.getStats().bcDimensionAlignCount, 1);
  assert.equal(table.getStats().bcFormatFallbackCount, 0);

  // Already multiple-of-4 dimensions (4096x2160) -> unchanged, counter does not increment.
  table.createTexture(201, {
    size: { width: 4096, height: 2160, depthOrArrayLayers: 1 },
    mipLevelCount: 1,
    format: 'bc3-rgba-unorm',
    usage: GPU_TEXTURE_USAGE.TEXTURE_BINDING,
  });
  assert.equal(calls.createTexture[1].desc.size.width, 4096);
  assert.equal(calls.createTexture[1].desc.size.height, 2160);
  assert.equal(table.getStats().bcDimensionAlignCount, 1);

  // Device without `features` property keeps BC format and still aligns dimensions.
  const minimalDevice = {
    createTexture(desc) {
      return { desc, createView: () => ({}) };
    },
  };
  const sanitized = sanitizeTextureDescriptor(
    { size: { width: 6, height: 10 }, format: 'bc1-rgba-unorm' },
    minimalDevice
  );
  assert.equal(sanitized.bcFormatFallback, false);
  assert.equal(sanitized.bcDimensionAligned, true);
  assert.equal(sanitized.descriptor.size.width, 8);
  assert.equal(sanitized.descriptor.size.height, 12);
});

test('CreateTexture2 binary payload decoding & creation handles 4K, array layers, viewFormats, BC fallback, and errors', () => {
  const { device, calls } = createMockDevice({ features: new Set() });
  const table = new ResourceTable(device);

  // 4K BGRA8_UNORM (code 27) with BGRA8_UNORM_SRGB view format (code 28), 6 array layers, excessive mips (99 -> 12).
  const payload4k = buildCreateTexture2Payload({
    id: 301,
    width: 3840,
    height: 2160,
    layers: 6,
    mips: 99,
    samples: 4, // must fall back to 1 because mipLevelCount > 1 and layers > 1
    usage: GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.COPY_DST,
    dimension: 1,
    format: 27,
    viewFormats: [28],
  });
  const res4k = table.createTexture2(payload4k, 0);
  assert.equal(res4k.error, undefined);
  assert.equal(res4k.id, 301);
  assert.ok(table.getTexture(301));
  assert.ok(table.getTextureView(301));
  assert.deepEqual(calls.createTexture[0].desc, {
    size: { width: 3840, height: 2160, depthOrArrayLayers: 6 },
    mipLevelCount: 12,
    sampleCount: 1,
    dimension: '2d',
    format: 'bgra8unorm',
    usage: GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.COPY_DST,
    viewFormats: ['bgra8unorm-srgb'],
  });

  // BC7_RGBA_UNORM (code 62) with BC7_RGBA_UNORM_SRGB view format (code 63) -> falls back on no-BC device.
  const payloadBc7 = buildCreateTexture2Payload({
    id: 302,
    width: 1024,
    height: 512,
    mips: 11,
    samples: 1,
    format: 62,
    viewFormats: [63],
  });
  const resBc7 = table.createTexture2FromPacket(payloadBc7, 0);
  assert.equal(resBc7.error, undefined);
  assert.equal(calls.createTexture[1].desc.format, 'rgba8unorm');
  assert.deepEqual(calls.createTexture[1].desc.viewFormats, ['rgba8unorm-srgb']);
  assert.equal(calls.createTexture[1].desc.mipLevelCount, 11);

  // Invalid format code (0) and truncated payload record errors without throwing.
  const invalidFormat = buildCreateTexture2Payload({ id: 303, format: 0 });
  const err1 = table.createTexture2FromPacket(invalidFormat, 0);
  assert.match(err1.error, /invalid texture format code/);
  assert.equal(table.hasTexture(303), false);

  const truncated = decodeCreateTexture2(new DataView(new ArrayBuffer(16)), 0);
  assert.match(truncated.error, /payload 16 < 40 bytes/);

  const stats = table.getStats();
  assert.equal(stats.texture2Created, 2);
  assert.equal(stats.bcFormatFallbackCount, 1);
  assert.equal(stats.mipClampCount, 1);
  assert.equal(stats.sampleClampCount, 1);
  assert.equal(stats.errors, 1);
});

test('createBuffer aligns sizes to 4 bytes (and minimum 16 bytes for UNIFORM) without 32-bit overflow', () => {
  const { device, calls } = createMockDevice();
  const table = new ResourceTable(device);

  // Real-title 4,080-byte VB -> already 4-byte aligned, no pad increment.
  table.createBuffer(1, { size: 4080, usage: GPU_BUFFER_USAGE.VERTEX | GPU_BUFFER_USAGE.COPY_DST });
  assert.equal(calls.createBuffer[0].desc.size, 4080);
  assert.equal(table.getStats().bufferAlignPadCount, 0);

  // Odd 13-byte VB -> aligned to 16 bytes.
  table.createBuffer(2, { size: 13, usage: GPU_BUFFER_USAGE.VERTEX | GPU_BUFFER_USAGE.COPY_DST });
  assert.equal(calls.createBuffer[1].desc.size, 16);
  assert.equal(table.getStats().bufferAlignPadCount, 1);

  // Zero-byte buffer -> aligned to minimum 4 bytes.
  table.createBuffer(3, 0, GPU_BUFFER_USAGE.STORAGE);
  assert.equal(calls.createBuffer[2].desc.size, 4);
  assert.equal(table.getStats().bufferAlignPadCount, 2);

  // Small UNIFORM buffer (4 bytes) -> padded to minimum 16 bytes.
  table.createBuffer(4, { size: 4, usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST });
  assert.equal(calls.createBuffer[3].desc.size, 16);
  assert.equal(table.getStats().bufferAlignPadCount, 3);

  // 17-byte UNIFORM buffer -> padded to 20 bytes.
  table.createBuffer(5, { size: 17, usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST });
  assert.equal(calls.createBuffer[4].desc.size, 20);
  assert.equal(table.getStats().bufferAlignPadCount, 4);

  // 16-byte UNIFORM buffer -> already valid, no pad increment.
  table.createBuffer(6, { size: 16, usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST });
  assert.equal(calls.createBuffer[5].desc.size, 16);
  assert.equal(table.getStats().bufferAlignPadCount, 4);

  // > 2 GiB buffer size must not wrap negative via 32-bit bitwise arithmetic.
  const largeAligned = alignBufferSize(3 * 1024 * 1024 * 1024 + 1, GPU_BUFFER_USAGE.STORAGE);
  assert.equal(largeAligned.alignedSize, 3 * 1024 * 1024 * 1024 + 4);
  assert.equal(largeAligned.padded, true);
});

test('writeBuffer pads non-4-byte-aligned payloads into zero-padded 4-byte-aligned copies', () => {
  const { device, calls } = createMockDevice();
  const table = new ResourceTable(device);

  // 17-byte CBV -> buffer created at 20 bytes.
  table.createBuffer(10, { size: 17, usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST });
  assert.equal(table.getStats().bufferAlignPadCount, 1);

  // Write 13 odd bytes (e.g. odd CBV upload) -> padded to 16 bytes with trailing zeros.
  const oddCbv = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
  const okOdd = table.writeBuffer(10, 0, oddCbv);
  assert.equal(okOdd, true);
  assert.equal(calls.writeBuffer.length, 1);
  assert.equal(calls.writeBuffer[0].byteLength, 16);
  assert.deepEqual(
    Array.from(calls.writeBuffer[0].data),
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 0, 0]
  );
  assert.equal(table.getStats().bufferAlignPadCount, 2);

  // Write 1-byte payload at offset 4 -> padded to 4 bytes with 3 trailing zeros.
  table.writeBuffer(10, 4, new Uint8Array([0xab]));
  assert.equal(calls.writeBuffer[1].bufferOffset, 4);
  assert.equal(calls.writeBuffer[1].byteLength, 4);
  assert.deepEqual(Array.from(calls.writeBuffer[1].data), [0xab, 0, 0, 0]);
  assert.equal(table.getStats().bufferAlignPadCount, 3);

  // Write already-aligned 16-byte Float32Array -> no extra padding increment.
  const alignedFloats = new Float32Array([1, 2, 3, 4]);
  table.writeBuffer(10, 0, alignedFloats);
  assert.equal(calls.writeBuffer[2].byteLength, 16);
  assert.equal(table.getStats().bufferAlignPadCount, 3);

  // SharedArrayBuffer-backed view is safely copied without throwing or inflating pad count.
  if (typeof SharedArrayBuffer !== 'undefined') {
    const sab = new SharedArrayBuffer(8);
    const sabView = new Uint8Array(sab);
    sabView.set([10, 20, 30, 40, 50, 60, 70, 80]);
    const paddedSab = padBufferUploadData(sabView);
    assert.equal(paddedSab.padded, false);
    assert.equal(paddedSab.bytes.buffer instanceof ArrayBuffer, true);
    assert.deepEqual(Array.from(paddedSab.bytes), [10, 20, 30, 40, 50, 60, 70, 80]);
  }

  // Unknown buffer id returns false safely.
  assert.equal(table.writeBuffer(9999, 0, new Uint8Array([1, 2, 3, 4])), false);
});

test('ResourceTable lifecycle, GpuResourceRegistry integration, and sanitized getStats()', () => {
  const { device } = createMockDevice({ features: new Set() });
  const registry = new GpuResourceRegistry({ framesInFlight: 2 });
  const warnings = [];
  const table = new ResourceTable(device, {
    resourceRegistry: registry,
    onWarning: (ev) => warnings.push(ev),
  });

  const buf1 = table.createBuffer(1, { size: 5, usage: GPU_BUFFER_USAGE.VERTEX });
  assert.equal(table.getBuffer(1), buf1);
  assert.equal(registry.get(RESOURCE_KIND.BUFFER, 1), buf1);

  // Overwriting id 1 retires old buffer via GpuResourceRegistry (not immediately destroyed).
  const buf2 = table.createBuffer(1, { size: 8, usage: GPU_BUFFER_USAGE.VERTEX });
  assert.equal(buf1.destroyed, false);
  assert.equal(table.getBuffer(1), buf2);

  // Advancing framesInFlight collects retired buf1.
  registry.beginFrame(1);
  registry.beginFrame(2);
  assert.equal(buf1.destroyed, true);
  assert.equal(buf2.destroyed, false);

  const tex = table.createTexture(2, {
    size: { width: 16, height: 16 },
    mipLevelCount: 10,
    format: 'bc7-rgba-unorm',
  });
  assert.equal(table.getTexture(2), tex);
  assert.ok(table.getTextureView(2));
  assert.equal(registry.get(RESOURCE_KIND.TEXTURE, 2), tex);

  const sampler = table.createSampler(3, { magFilter: 'linear' });
  const shader = table.createShaderModule(4, { code: '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }' });
  assert.equal(table.getSampler(3), sampler);
  assert.equal(table.getShaderModule(4), shader);

  const stats = table.getStats();
  assert.equal(stats.bcFormatFallbackCount, 1);
  assert.equal(stats.bcDimensionAlignCount, 0);
  assert.equal(stats.bufferAlignPadCount, 1);
  assert.equal(stats.mipClampCount, 1);
  assert.equal(stats.buffersActive, 1);
  assert.equal(stats.texturesActive, 1);
  assert.equal(stats.samplersActive, 1);
  assert.equal(stats.shaderModulesActive, 1);

  // All getStats() values are sanitized finite numbers (no paths or raw data).
  for (const [k, v] of Object.entries(stats)) {
    assert.equal(typeof v, 'number', `stat ${k} must be numeric`);
    assert.ok(Number.isFinite(v), `stat ${k} must be finite`);
  }
  assert.ok(warnings.length >= 3);

  table.destroyAll();
  assert.equal(table.getStats().buffersActive, 0);
  assert.equal(table.getStats().texturesActive, 0);
  assert.equal(table.getStats().samplersActive, 0);
  assert.equal(table.getStats().shaderModulesActive, 0);
});

test('ResourceTable tracks peakLiveHandles, totalInserts, and totalDeletes across insert/delete waves (W141)', () => {
  const { device } = createMockDevice();
  const table = new ResourceTable(device);

  let stats = table.getStats();
  assert.equal(stats.liveHandles, 0);
  assert.equal(stats.peakLiveHandles, 0);
  assert.equal(stats.overwrites, 0);
  assert.equal(stats.missingLookups, 0);
  assert.equal(stats.totalInserts, 0);
  assert.equal(stats.totalDeletes, 0);

  // Wave 1: insert 6 handles across all 4 resource kinds (buffers, textures, texture2, samplers, shaders).
  table.createBuffer(1, { size: 16, usage: GPU_BUFFER_USAGE.VERTEX });
  table.createBuffer(2, { size: 32, usage: GPU_BUFFER_USAGE.UNIFORM });
  table.createTexture(10, { size: { width: 32, height: 32 }, format: 'rgba8unorm' });
  table.createTexture2(buildCreateTexture2Payload({ id: 11, width: 64, height: 64 }), 0);
  table.createSampler(20, { magFilter: 'linear' });
  table.createShaderModule(30, { code: '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }' });

  stats = table.getStats();
  assert.equal(stats.liveHandles, 6);
  assert.equal(stats.peakLiveHandles, 6);
  assert.equal(stats.totalInserts, 6);
  assert.equal(stats.totalDeletes, 0);
  assert.equal(stats.overwrites, 0);

  // Wave 2: delete 4 handles + attempt deleting unknown handles (must not inflate totalDeletes).
  assert.equal(table.destroyBuffer(1), true);
  assert.equal(table.destroyBuffer(1), false);
  assert.equal(table.destroyTexture(10), true);
  assert.equal(table.destroyTexture(999), false);
  assert.equal(table.destroy(20), 1);
  assert.equal(table.destroy(30), 1);
  assert.equal(table.destroy(12345), 0);

  stats = table.getStats();
  assert.equal(stats.liveHandles, 2);
  assert.equal(stats.peakLiveHandles, 6);
  assert.equal(stats.totalInserts, 6);
  assert.equal(stats.totalDeletes, 4);

  // Wave 3: insert 2 more (live=4 < peak=6 -> peak stays 6).
  table.createBuffer(3, { size: 64, usage: GPU_BUFFER_USAGE.STORAGE });
  table.createSampler(21, { minFilter: 'nearest' });

  stats = table.getStats();
  assert.equal(stats.liveHandles, 4);
  assert.equal(stats.peakLiveHandles, 6);
  assert.equal(stats.totalInserts, 8);
  assert.equal(stats.totalDeletes, 4);

  // Wave 4: insert 5 more (live=9 > peak=6 -> peak rises to 9).
  for (let i = 0; i < 5; i += 1) {
    table.createBuffer(100 + i, { size: 16, usage: GPU_BUFFER_USAGE.VERTEX });
  }
  stats = table.getStats();
  assert.equal(stats.liveHandles, 9);
  assert.equal(stats.peakLiveHandles, 9);
  assert.equal(stats.totalInserts, 13);
  assert.equal(stats.totalDeletes, 4);
  assert.equal(stats.liveHandles, stats.totalInserts - stats.overwrites - stats.totalDeletes);
});

test('ResourceTable accounts for overwrites across all resource kinds without double-counting deletes (W141)', () => {
  const { device } = createMockDevice();
  const table = new ResourceTable(device);

  const b1 = table.createBuffer(1, { size: 16, usage: GPU_BUFFER_USAGE.VERTEX });
  const t1 = table.createTexture(2, { size: { width: 16, height: 16 }, format: 'rgba8unorm' });
  const s1 = table.createSampler(3, { magFilter: 'linear' });
  const m1 = table.createShaderModule(4, { code: '/* v1 */' });

  assert.equal(table.getStats().liveHandles, 4);
  assert.equal(table.getStats().peakLiveHandles, 4);
  assert.equal(table.getStats().overwrites, 0);
  assert.equal(table.getStats().totalInserts, 4);
  assert.equal(table.getStats().totalDeletes, 0);

  // Overwrite each live handle id once (plus a second texture overwrite via createTexture2).
  const b2 = table.createBuffer(1, { size: 32, usage: GPU_BUFFER_USAGE.VERTEX });
  const t2 = table.createTexture(2, { size: { width: 32, height: 32 }, format: 'rgba8unorm' });
  const t3Res = table.createTexture2(buildCreateTexture2Payload({ id: 2, width: 64, height: 64 }), 0);
  const s2 = table.createSampler(3, { magFilter: 'nearest' });
  const m2 = table.createShaderModule(4, { code: '/* v2 */' });

  assert.equal(b1.destroyed, true);
  assert.equal(t1.destroyed, true);
  assert.equal(t2.destroyed, true);
  assert.notEqual(table.getBuffer(1), b1);
  assert.equal(table.getBuffer(1), b2);
  assert.equal(table.getTexture(2), t3Res.texture);
  assert.notEqual(table.getSampler(3), s1);
  assert.equal(table.getSampler(3), s2);
  assert.notEqual(table.getShaderModule(4), m1);
  assert.equal(table.getShaderModule(4), m2);

  const stats = table.getStats();
  assert.equal(stats.liveHandles, 4);
  assert.equal(stats.peakLiveHandles, 4);
  assert.equal(stats.overwrites, 5);
  assert.equal(stats.totalInserts, 9);
  assert.equal(stats.totalDeletes, 0);
  assert.equal(stats.missingLookups, 0);
  assert.equal(stats.liveHandles, stats.totalInserts - stats.overwrites - stats.totalDeletes);

  // Re-inserting an explicitly deleted id is a fresh insert, not an overwrite.
  assert.equal(table.destroyBuffer(1), true);
  table.createBuffer(1, { size: 64, usage: GPU_BUFFER_USAGE.VERTEX });
  const afterReinsert = table.getStats();
  assert.equal(afterReinsert.overwrites, 5);
  assert.equal(afterReinsert.totalInserts, 10);
  assert.equal(afterReinsert.totalDeletes, 1);
  assert.equal(afterReinsert.liveHandles, 4);
  assert.equal(
    afterReinsert.liveHandles,
    afterReinsert.totalInserts - afterReinsert.overwrites - afterReinsert.totalDeletes
  );
});

test('ResourceTable tracks missingLookups for unknown and deleted handles without counting hits or has* checks (W141)', () => {
  const { device } = createMockDevice();
  const table = new ResourceTable(device);

  // Existence predicates must NOT increment missingLookups.
  assert.equal(table.hasBuffer(999), false);
  assert.equal(table.hasTexture(999), false);
  assert.equal(table.getStats().missingLookups, 0);

  // Lookups on unknown handle ids increment missingLookups.
  assert.equal(table.getBuffer(101), undefined);
  assert.equal(table.getBufferMeta(101), undefined);
  assert.equal(table.getTexture(102), undefined);
  assert.equal(table.getTextureView(102), undefined);
  assert.equal(table.getTextureMeta(102), undefined);
  assert.equal(table.getSampler(103), undefined);
  assert.equal(table.getShaderModule(104), undefined);
  assert.equal(table.writeBuffer(105, 0, new Uint8Array([1, 2, 3, 4])), false);
  assert.equal(table.getStats().missingLookups, 8);

  // Create valid handles and verify hits do NOT increment missingLookups.
  table.createBuffer(1, { size: 16, usage: GPU_BUFFER_USAGE.VERTEX | GPU_BUFFER_USAGE.COPY_DST });
  table.createTexture(2, { size: { width: 16, height: 16 }, format: 'rgba8unorm' });
  table.createSampler(3, {});
  table.createShaderModule(4, { code: '/* ok */' });

  assert.ok(table.getBuffer(1));
  assert.ok(table.getBufferMeta(1));
  assert.equal(table.writeBuffer(1, 0, new Uint8Array([1, 2, 3, 4])), true);
  assert.ok(table.getTexture(2));
  assert.ok(table.getTextureView(2));
  assert.ok(table.getTextureMeta(2));
  assert.ok(table.getSampler(3));
  assert.ok(table.getShaderModule(4));
  assert.equal(table.getStats().missingLookups, 8);

  // Lookups on deleted handles increment missingLookups.
  assert.equal(table.destroyBuffer(1), true);
  assert.equal(table.destroyTexture(2), true);
  assert.equal(table.destroy(3), 1);
  assert.equal(table.destroy(4), 1);

  assert.equal(table.getBuffer(1), undefined);
  assert.equal(table.getTexture(2), undefined);
  assert.equal(table.getTextureView(2), undefined);
  assert.equal(table.getSampler(3), undefined);
  assert.equal(table.getShaderModule(4), undefined);
  assert.equal(table.writeBuffer(1, 0, new Uint8Array([1, 2, 3, 4])), false);
  assert.equal(table.getStats().missingLookups, 14);
});

test('ResourceTable resetStats() preserves live handles with peak=currentLive and clear() resets deterministically (W141)', () => {
  const { device } = createMockDevice({ features: new Set() });
  const table = new ResourceTable(device);

  // Build up a peak of 5 handles, trigger overwrite + missing lookup + hardening counters.
  const b1 = table.createBuffer(1, { size: 5, usage: GPU_BUFFER_USAGE.VERTEX });
  table.createBuffer(2, { size: 16, usage: GPU_BUFFER_USAGE.VERTEX });
  table.createTexture(3, { size: { width: 16, height: 16 }, mipLevelCount: 10, format: 'bc7-rgba-unorm' });
  table.createSampler(4, {});
  table.createShaderModule(5, { code: '/* s */' });

  // Overwrite id 2 and probe missing id 999, then delete 3 handles so live=2 while peak=5.
  table.createBuffer(2, { size: 32, usage: GPU_BUFFER_USAGE.VERTEX });
  assert.equal(table.getBuffer(999), undefined);
  table.destroyBuffer(2);
  table.destroy(4);
  table.destroy(5);

  const beforeReset = table.getStats();
  assert.equal(beforeReset.liveHandles, 2);
  assert.equal(beforeReset.peakLiveHandles, 5);
  assert.equal(beforeReset.overwrites, 1);
  assert.equal(beforeReset.missingLookups, 1);
  assert.equal(beforeReset.totalInserts, 6);
  assert.equal(beforeReset.totalDeletes, 3);

  // resetStats() zeroes counters, preserves live handles, and clamps peakLiveHandles to liveHandles (2).
  table.resetStats();
  const afterReset = table.getStats();
  assert.equal(afterReset.buffersActive, 1);
  assert.equal(afterReset.texturesActive, 1);
  assert.equal(afterReset.liveHandles, 2);
  assert.equal(afterReset.peakLiveHandles, 2);
  assert.equal(afterReset.overwrites, 0);
  assert.equal(afterReset.missingLookups, 0);
  assert.equal(afterReset.totalInserts, 0);
  assert.equal(afterReset.totalDeletes, 0);
  assert.equal(afterReset.buffersCreated, 0);
  assert.equal(afterReset.texturesCreated, 0);
  assert.equal(afterReset.destroyCount, 0);
  assert.equal(afterReset.bcFormatFallbackCount, 0);
  assert.equal(afterReset.bufferAlignPadCount, 0);
  assert.equal(afterReset.mipClampCount, 0);

  // Preserved handles are still accessible.
  assert.equal(table.getBuffer(1), b1);
  assert.equal(b1.destroyed, false);
  assert.ok(table.getTexture(3));

  // New insert after resetStats() pushes peakLiveHandles from 2 -> 3.
  table.createSampler(6, {});
  assert.equal(table.getStats().liveHandles, 3);
  assert.equal(table.getStats().peakLiveHandles, 3);
  assert.equal(table.getStats().totalInserts, 1);

  // clear() destroys all live handles while preserving cumulative peak and recording deletes.
  table.clear();
  const afterClear = table.getStats();
  assert.equal(b1.destroyed, true);
  assert.equal(afterClear.liveHandles, 0);
  assert.equal(afterClear.buffersActive, 0);
  assert.equal(afterClear.texturesActive, 0);
  assert.equal(afterClear.textureViewsActive, 0);
  assert.equal(afterClear.samplersActive, 0);
  assert.equal(afterClear.shaderModulesActive, 0);
  assert.equal(afterClear.peakLiveHandles, 3);
  assert.equal(afterClear.totalDeletes, 3);
  assert.equal(table.getBuffer(1), undefined);
  assert.equal(table.getStats().missingLookups, 1);

  // clear({ resetStats: true }) destroys live handles and resets all telemetry including peakLiveHandles=0.
  table.createBuffer(7, { size: 16, usage: GPU_BUFFER_USAGE.VERTEX });
  table.createBuffer(7, { size: 32, usage: GPU_BUFFER_USAGE.VERTEX });
  assert.equal(table.getStats().overwrites, 1);
  table.clear({ resetStats: true });

  const afterClearReset = table.getStats();
  assert.equal(afterClearReset.liveHandles, 0);
  assert.equal(afterClearReset.peakLiveHandles, 0);
  assert.equal(afterClearReset.overwrites, 0);
  assert.equal(afterClearReset.missingLookups, 0);
  assert.equal(afterClearReset.totalInserts, 0);
  assert.equal(afterClearReset.totalDeletes, 0);
  assert.equal(afterClearReset.destroyCount, 0);

  for (const [k, v] of Object.entries(afterClearReset)) {
    assert.equal(typeof v, 'number', `stat ${k} must be numeric`);
    assert.ok(Number.isFinite(v), `stat ${k} must be finite`);
  }
});

test('ResourceTable fast descriptor slot cache elides no-op writes and increments versions monotonically on real changes (W207)', () => {
  const { device } = createMockDevice();
  const table = new ResourceTable(device);

  assert.equal(DEFAULT_FAST_SLOT_CAPACITY, 1024);
  assert.equal(table.tableVersion, 0);
  assert.equal(table.getTableVersion(), 0);
  assert.equal(table.getSlotVersion(0), 0);
  assert.equal(table.hasSlot(0), false);
  assert.equal(table.getSlot(0), undefined);

  const buf1 = table.createBuffer(1, { size: 256, usage: GPU_BUFFER_USAGE.UNIFORM });
  const buf2 = table.createBuffer(2, { size: 256, usage: GPU_BUFFER_USAGE.UNIFORM });
  table.createTexture(10, { size: { width: 64, height: 64 }, mipLevelCount: 4, format: 'rgba8unorm' });
  // Creating resources that are not referenced by any slot must NOT bump tableVersion.
  assert.equal(table.tableVersion, 0);

  // Initial write to slot 0 increments slot version and tableVersion.
  assert.equal(table.setSlot(0, { kind: 'buffer', handle: 1, offset: 0, size: 64 }), true);
  assert.equal(table.hasSlot(0), true);
  assert.equal(table.getSlotVersion(0), 1);
  assert.equal(table.tableVersion, 1);

  // Identical overwrite (including default offset omission) is elided as a no-op.
  assert.equal(table.setSlot(0, { kind: 'buffer', handle: 1, offset: 0, size: 64 }), false);
  assert.equal(table.setSlot(0, { kind: 'buffer', handle: 1, size: 64 }), false);
  assert.equal(table.setSlot(0, 1, { kind: 'buffer', offset: 0, size: 64 }), false);
  assert.equal(table.getSlotVersion(0), 1);
  assert.equal(table.tableVersion, 1);

  let stats = table.getStats();
  assert.equal(stats.slotWrites, 1);
  assert.equal(stats.elidedNoOpWrites, 3);
  assert.equal(stats.tableVersion, 1);
  assert.equal(stats.activeSlots, 1);

  // Real view-param change (offset 0 -> 64) increments both versions.
  assert.equal(table.setSlot(0, { kind: 'buffer', handle: 1, offset: 64, size: 64 }), true);
  assert.equal(table.getSlotVersion(0), 2);
  assert.equal(table.tableVersion, 2);

  // Real handle change (1 -> 2) increments both versions.
  assert.equal(table.setSlot(0, { kind: 'buffer', handle: 2, offset: 64, size: 64 }), true);
  assert.equal(table.getSlotVersion(0), 3);
  assert.equal(table.tableVersion, 3);

  // Real kind change on same numeric id increments both versions.
  assert.equal(table.setSlot(0, { kind: 'texture', handle: 2 }), true);
  assert.equal(table.getSlotVersion(0), 4);
  assert.equal(table.tableVersion, 4);

  // Texture view-param normalization: case-insensitive format + flat vs nested viewParams + key order.
  assert.equal(
    table.setSlot(1, {
      kind: 'texture',
      handle: 10,
      format: 'RGBA8Unorm',
      dimension: '2D',
      aspect: 'ALL',
      baseMipLevel: 1,
      mipLevelCount: 2,
      viewParams: { swizzleR: 'r', swizzleG: 'g' },
    }),
    true
  );
  assert.equal(table.getSlotVersion(1), 1);
  assert.equal(table.tableVersion, 5);

  // Equivalent descriptor with flat/nested representation & reversed extra key order -> elided.
  assert.equal(
    table.setSlot(1, {
      kind: 'texture',
      handle: 10,
      view: {
        format: 'rgba8unorm',
        dimension: '2d',
        aspect: 'all',
        baseMipLevel: 1,
        mipLevelCount: 2,
        swizzleG: 'g',
        swizzleR: 'r',
      },
    }),
    false
  );
  assert.equal(table.getSlotVersion(1), 1);
  assert.equal(table.tableVersion, 5);

  // Mutating an extra viewParam triggers a real version bump.
  assert.equal(
    table.setSlot(1, {
      kind: 'texture',
      handle: 10,
      format: 'rgba8unorm',
      dimension: '2d',
      aspect: 'all',
      baseMipLevel: 1,
      mipLevelCount: 2,
      viewParams: { swizzleR: 'b', swizzleG: 'g' },
    }),
    true
  );
  assert.equal(table.getSlotVersion(1), 2);
  assert.equal(table.tableVersion, 6);

  // Direct GPU resource reference equality & elision.
  assert.equal(table.setSlot(2, buf1, { offset: 0, size: 128 }), true);
  assert.equal(table.setSlot(2, buf1, { offset: 0, size: 128 }), false);
  assert.equal(table.setSlot(2, buf2, { offset: 0, size: 128 }), true);
  assert.equal(table.resolveSlot(2), buf2);

  // Unbinding an already-empty slot via setSlot(idx, null) is elided; unbinding occupied slot mutates.
  assert.equal(table.setSlot(99, null), false);
  assert.equal(table.getSlotVersion(99), 0);
  const beforeUnbindTableVer = table.tableVersion;
  assert.equal(table.setSlot(2, null), true);
  assert.equal(table.hasSlot(2), false);
  assert.equal(table.getSlotVersion(2), 3);
  assert.equal(table.tableVersion, beforeUnbindTableVer + 1);

  // Repeated getSlot returns the exact same cached frozen snapshot object until mutated.
  const snapA = table.getSlot(1);
  const snapB = table.getSlot(1);
  assert.strictEqual(snapA, snapB);
  assert.equal(Object.isFrozen(snapA), true);
  assert.equal(snapA.version, 2);

  // Validation on bad slot indices.
  assert.throws(() => table.setSlot(-1, { kind: 'buffer', handle: 1 }), RangeError);
  assert.throws(() => table.setSlot(1.5, { kind: 'buffer', handle: 1 }), RangeError);
  assert.equal(table.getSlot(-1), undefined);
  assert.equal(table.hasSlot(-1), false);
  assert.equal(table.getSlotVersion(-1), 0);
  assert.equal(table.clearSlot(-1), false);

  stats = table.getStats();
  assert.equal(stats.activeSlots, 2);
  assert.equal(stats.slotWrites, 9);
  assert.equal(stats.elidedNoOpWrites, 6);
  assert.equal(stats.tableVersion, table.tableVersion);
});

test('ResourceTable O(1) bind-group version validation, resource-lifecycle invalidation, and custom view caching (W207)', () => {
  const { device } = createMockDevice();
  let customViewCreates = 0;
  const origCreateTexture = device.createTexture.bind(device);
  device.createTexture = (desc) => {
    const tex = origCreateTexture(desc);
    const origCreateView = tex.createView.bind(tex);
    tex.createView = (viewDesc) => {
      if (viewDesc !== undefined) {
        customViewCreates += 1;
      }
      return origCreateView(viewDesc);
    };
    return tex;
  };

  const table = new ResourceTable(device, { fastSlotCapacity: 8 });
  const b1 = table.createBuffer(1, { size: 128, usage: GPU_BUFFER_USAGE.UNIFORM });
  table.createTexture(2, { size: { width: 64, height: 64 }, mipLevelCount: 4, format: 'rgba8unorm' });
  const s1 = table.createSampler(3, { magFilter: 'linear' });

  // Bind dense fast slots (0, 1, 2) and an overflow slot (64 >= fastSlotCapacity 8).
  table.setSlot(0, { kind: 'buffer', handle: 1, offset: 16, size: 64 });
  table.setSlot(1, { kind: 'texture', handle: 2, baseMipLevel: 1, mipLevelCount: 2 });
  table.setSlot(2, { kind: 'sampler', handle: 3 });
  table.setSlot(64, { kind: 'buffer', handle: 1, offset: 0, size: 32 });

  // Custom texture view is lazily created once and reused across lookups & elided writes.
  const view1 = table.resolveSlot(1);
  const view2 = table.getSlotResource(1);
  assert.strictEqual(view1, view2);
  assert.equal(customViewCreates, 1);
  assert.equal(table.setSlot(1, { kind: 'texture', handle: 2, baseMipLevel: 1, mipLevelCount: 2 }), false);
  assert.strictEqual(table.resolveSlot(1), view1);
  assert.equal(customViewCreates, 1);

  // getBindGroupEntry returns valid WebGPU bind group entries.
  assert.deepEqual(table.getBindGroupEntry(0, 0), {
    binding: 0,
    resource: { buffer: b1, offset: 16, size: 64 },
  });
  assert.deepEqual(table.getBindGroupEntry(1, 1), {
    binding: 1,
    resource: view1,
  });
  assert.deepEqual(table.getBindGroupEntry(2, 2), {
    binding: 2,
    resource: s1,
  });
  assert.equal(table.getBindGroupEntry(3, 5), null);

  // Capture bind-group version stamp for slots [0, 1, 2].
  const bgSnap = table.captureSlotVersions([0, 1, 2]);
  const keyBefore = table.getSlotVersionKey([0, 1, 2]);
  assert.equal(table.isBindGroupCacheValid(bgSnap), true);

  // Mutating an UNRELATED overflow slot (64) advances tableVersion, but bgSnap remains valid
  // and fast-forwards its tableVersion for subsequent O(1) checks.
  assert.equal(table.setSlot(64, { kind: 'buffer', handle: 1, offset: 32, size: 32 }), true);
  assert.notEqual(bgSnap.tableVersion, table.tableVersion);
  assert.equal(table.isBindGroupCacheValid(bgSnap), true);
  assert.equal(bgSnap.tableVersion, table.tableVersion);
  assert.equal(table.getSlotVersionKey([0, 1, 2]), keyBefore);

  // Overwriting buffer handle 1 automatically invalidates slot 0 (and slot 64) and bgSnap!
  const v0Before = table.getSlotVersion(0);
  const v64Before = table.getSlotVersion(64);
  const b1New = table.createBuffer(1, { size: 256, usage: GPU_BUFFER_USAGE.UNIFORM });
  assert.equal(table.getSlotVersion(0), v0Before + 1);
  assert.equal(table.getSlotVersion(64), v64Before + 1);
  assert.equal(table.isBindGroupCacheValid(bgSnap), false);
  assert.notEqual(table.getSlotVersionKey([0, 1, 2]), keyBefore);
  assert.strictEqual(table.resolveSlot(0), b1New);

  // Re-capturing after rebuild stays valid; repeating identical setSlot on slot 0 is elided.
  const bgSnap2 = table.captureSlotVersions([0, 1, 2]);
  assert.equal(table.setSlot(0, { kind: 'buffer', handle: 1, offset: 16, size: 64 }), false);
  assert.equal(table.isBindGroupCacheValid(bgSnap2), true);

  // Overwriting texture handle 2 invalidates slot 1 AND clears its cached custom view.
  const v1Before = table.getSlotVersion(1);
  table.createTexture(2, { size: { width: 128, height: 128 }, mipLevelCount: 5, format: 'rgba8unorm' });
  assert.equal(table.getSlotVersion(1), v1Before + 1);
  assert.equal(table.isBindGroupCacheValid(bgSnap2), false);
  const view3 = table.resolveSlot(1);
  assert.notStrictEqual(view3, view1);
  assert.equal(customViewCreates, 2);

  // Destroying sampler handle 3 invalidates slot 2.
  const bgSnap3 = table.captureSlotVersions([0, 1, 2]);
  const v2Before = table.getSlotVersion(2);
  assert.equal(table.destroy(3), 1);
  assert.equal(table.getSlotVersion(2), v2Before + 1);
  assert.equal(table.isBindGroupCacheValid(bgSnap3), false);
  assert.equal(table.resolveSlot(2), undefined);
});

test('ResourceTable clearSlot, clearSlots, clear(), and resetStats() preserve monotonic versions and stats (W207)', () => {
  const { device } = createMockDevice();
  const table = new ResourceTable(device, { fastSlotCapacity: 4 });

  table.createBuffer(1, { size: 64, usage: GPU_BUFFER_USAGE.UNIFORM });
  table.createBuffer(2, { size: 64, usage: GPU_BUFFER_USAGE.UNIFORM });

  // Bind dense slot 0, dense slot 1, and overflow slot 10.
  table.setSlot(0, { kind: 'buffer', handle: 1, offset: 0, size: 32 });
  table.setSlot(1, { kind: 'buffer', handle: 2, offset: 0, size: 64 });
  table.setSlot(10, { kind: 'buffer', handle: 1, offset: 32, size: 32 });
  // Trigger a no-op write.
  table.setSlot(0, { kind: 'buffer', handle: 1, offset: 0, size: 32 });

  const snapAll = table.captureSlotVersions([0, 1, 10]);
  assert.equal(table.getStats().activeSlots, 3);
  assert.equal(table.getStats().slotWrites, 3);
  assert.equal(table.getStats().elidedNoOpWrites, 1);

  // clearSlot on occupied slot increments its version & tableVersion; second call is a no-op.
  const v0BeforeClear = table.getSlotVersion(0);
  const tvBeforeClear = table.tableVersion;
  assert.equal(table.clearSlot(0), true);
  assert.equal(table.hasSlot(0), false);
  assert.equal(table.getSlot(0), undefined);
  assert.equal(table.getSlotVersion(0), v0BeforeClear + 1);
  assert.equal(table.tableVersion, tvBeforeClear + 1);
  assert.equal(table.getStats().activeSlots, 2);
  assert.equal(table.isBindGroupCacheValid(snapAll), false);

  // Repeated clearSlot on already-unbound slot does NOT bump version.
  assert.equal(table.clearSlot(0), false);
  assert.equal(table.getSlotVersion(0), v0BeforeClear + 1);
  assert.equal(table.tableVersion, tvBeforeClear + 1);

  // Re-binding slot 0 after clearSlot yields strictly higher version (no ABA reuse).
  assert.equal(table.setSlot(0, { kind: 'buffer', handle: 1, offset: 0, size: 32 }), true);
  assert.equal(table.getSlotVersion(0), v0BeforeClear + 2);
  assert.equal(table.getStats().activeSlots, 3);

  // resetStats() zeroes slotWrites & elidedNoOpWrites while preserving activeSlots & monotonic versions.
  const tvBeforeReset = table.tableVersion;
  const v1BeforeReset = table.getSlotVersion(1);
  table.resetStats();
  const afterReset = table.getStats();
  assert.equal(afterReset.slotWrites, 0);
  assert.equal(afterReset.elidedNoOpWrites, 0);
  assert.equal(afterReset.activeSlots, 3);
  assert.equal(afterReset.tableVersion, tvBeforeReset);
  assert.equal(table.getSlotVersion(1), v1BeforeReset);
  assert.equal(table.hasSlot(0), true);
  assert.equal(table.hasSlot(10), true);

  // clearSlots() clears all dense + overflow active slots and bumps their versions monotonically.
  const snapBeforeBulkClear = table.captureSlotVersions([0, 1, 10]);
  const v10BeforeBulkClear = table.getSlotVersion(10);
  assert.equal(table.clearSlots(), 3);
  assert.equal(table.getStats().activeSlots, 0);
  assert.equal(table.hasSlot(0), false);
  assert.equal(table.hasSlot(1), false);
  assert.equal(table.hasSlot(10), false);
  assert.equal(table.getSlotVersion(10), v10BeforeBulkClear + 1);
  assert.equal(table.tableVersion, tvBeforeReset + 1);
  assert.equal(table.isBindGroupCacheValid(snapBeforeBulkClear), false);

  // Calling clearSlots() when already empty is a no-op (no version bump).
  assert.equal(table.clearSlots(), 0);
  assert.equal(table.tableVersion, tvBeforeReset + 1);

  // Re-populate and test clear() and clear({ resetStats: true }).
  table.setSlot(0, { kind: 'buffer', handle: 1, offset: 0, size: 32 });
  table.setSlot(0, { kind: 'buffer', handle: 1, offset: 0, size: 32 }); // no-op
  const snapBeforeClear = table.captureSlotVersions([0]);
  const tvBeforeFullClear = table.tableVersion;
  table.clear();
  const afterClear = table.getStats();
  assert.equal(afterClear.activeSlots, 0);
  assert.equal(afterClear.liveHandles, 0);
  assert.equal(afterClear.slotWrites, 1);
  assert.equal(afterClear.elidedNoOpWrites, 1);
  assert.ok(afterClear.tableVersion > tvBeforeFullClear);
  assert.equal(table.isBindGroupCacheValid(snapBeforeClear), false);

  // clear({ resetStats: true }) resets telemetry counters while keeping tableVersion strictly monotonic.
  table.createBuffer(5, { size: 64, usage: GPU_BUFFER_USAGE.UNIFORM });
  table.setSlot(0, { kind: 'buffer', handle: 5 });
  table.setSlot(0, { kind: 'buffer', handle: 5 }); // no-op
  const tvBeforeClearReset = table.tableVersion;
  const v0BeforeClearReset = table.getSlotVersion(0);
  table.clear({ resetStats: true });

  const afterClearReset = table.getStats();
  assert.equal(afterClearReset.activeSlots, 0);
  assert.equal(afterClearReset.slotWrites, 0);
  assert.equal(afterClearReset.elidedNoOpWrites, 0);
  assert.ok(afterClearReset.tableVersion > tvBeforeClearReset);
  assert.ok(table.getSlotVersion(0) > v0BeforeClearReset);

  for (const [k, v] of Object.entries(afterClearReset)) {
    assert.equal(typeof v, 'number', `stat ${k} must be numeric`);
    assert.ok(Number.isFinite(v), `stat ${k} must be finite`);
  }
});

test('ResourceTable tracks monotonic resource generation stamps across create, overwrite, and destroy for buffers, textures, texture views, and samplers (W267)', async () => {
  const esmMod = await import('../web/workers/resource_table.js');
  assert.equal(DEFAULT_MAX_BIND_GROUPS, 4096);
  assert.equal(esmMod.DEFAULT_MAX_BIND_GROUPS, 4096);
  assert.equal(globalThis.PlayGta6ResourceTable.DEFAULT_MAX_BIND_GROUPS, 4096);

  const { device } = createMockDevice();
  const table = new ResourceTable(device);
  assert.equal(table.maxBindGroups, DEFAULT_MAX_BIND_GROUPS);
  assert.throws(() => new ResourceTable(device, { maxBindGroups: -1 }), RangeError);
  assert.throws(() => new ResourceTable(device, { maxBindGroups: 1.5 }), RangeError);

  // Never-created handles report generation 0 and are not current.
  assert.equal(table.getResourceGeneration('buffer', 1), 0);
  assert.equal(table.getResourceGeneration('texture', 2), 0);
  assert.equal(table.getResourceGeneration('textureView', 2), 0);
  assert.equal(table.getResourceGeneration('sampler', 3), 0);
  assert.equal(table.isResourceGenerationCurrent('buffer', 1, 0), false);

  // Buffer lifecycle: create -> overwrite -> destroy -> recreate.
  table.createBuffer(1, { size: 64, usage: GPU_BUFFER_USAGE.UNIFORM });
  const bg1 = table.getResourceGeneration('buffer', 1);
  assert.ok(bg1 >= 1);
  assert.equal(table.isResourceGenerationCurrent('buffer', 1, bg1), true);

  table.createBuffer(1, { size: 128, usage: GPU_BUFFER_USAGE.UNIFORM });
  const bg2 = table.getResourceGeneration('buffer', 1);
  assert.ok(bg2 > bg1);
  assert.equal(table.isResourceGenerationCurrent('buffer', 1, bg1), false);
  assert.equal(table.isResourceGenerationCurrent('buffer', 1, bg2), true);

  assert.equal(table.destroyBuffer(1), true);
  const bg3 = table.getResourceGeneration('buffer', 1);
  assert.ok(bg3 > bg2);
  assert.equal(table.isResourceGenerationCurrent('buffer', 1, bg2), false);
  assert.equal(table.isResourceGenerationCurrent('buffer', 1, bg3), false);

  table.createBuffer(1, { size: 64, usage: GPU_BUFFER_USAGE.UNIFORM });
  const bg4 = table.getResourceGeneration('buffer', 1);
  assert.ok(bg4 > bg3);
  assert.equal(table.isResourceGenerationCurrent('buffer', 1, bg4), true);

  // Texture + default view + standalone child view lifecycle.
  table.createTexture(2, { size: { width: 64, height: 64 }, mipLevelCount: 4, format: 'rgba8unorm' });
  const tg1 = table.getResourceGeneration('texture', 2);
  const vg1 = table.getResourceGeneration('textureView', 2);
  assert.ok(tg1 >= 1);
  assert.ok(vg1 > tg1);
  assert.equal(table.isResourceGenerationCurrent('texture', 2, tg1), true);
  assert.equal(table.isResourceGenerationCurrent('textureView', 2, vg1), true);

  // Overwriting default textureView via createTextureView bumps view generation without destroying texture.
  const customDefaultView = table.createTextureView(2, { baseMipLevel: 1, mipLevelCount: 1 });
  assert.ok(customDefaultView);
  const vg2 = table.getResourceGeneration('textureView', 2);
  assert.ok(vg2 > vg1);
  assert.equal(table.getResourceGeneration('texture', 2), tg1);
  assert.equal(table.isResourceGenerationCurrent('textureView', 2, vg1), false);
  assert.equal(table.isResourceGenerationCurrent('textureView', 2, vg2), true);

  // Child textureView handle (200) bound to parent texture (2).
  const childView = table.createTextureView(200, 2, { baseMipLevel: 2, mipLevelCount: 1 });
  assert.ok(childView);
  const childGen1 = table.getResourceGeneration('textureView', 200);
  assert.ok(childGen1 > vg2);
  assert.equal(table.isResourceGenerationCurrent('textureView', 200, childGen1), true);

  // Destroying parent texture (2) advances texture 2, default view 2, AND child view 200 generations!
  assert.equal(table.destroyTexture(2), true);
  const tg2 = table.getResourceGeneration('texture', 2);
  const vg3 = table.getResourceGeneration('textureView', 2);
  const childGen2 = table.getResourceGeneration('textureView', 200);
  assert.ok(tg2 > tg1);
  assert.ok(vg3 > vg2);
  assert.ok(childGen2 > childGen1);
  assert.equal(table.isResourceGenerationCurrent('texture', 2, tg1), false);
  assert.equal(table.isResourceGenerationCurrent('textureView', 2, vg2), false);
  assert.equal(table.isResourceGenerationCurrent('textureView', 200, childGen1), false);
  assert.equal(table.getTextureView(200), undefined);

  // Sampler lifecycle: create -> overwrite -> destroySampler.
  table.createSampler(3, { magFilter: 'linear' });
  const sg1 = table.getResourceGeneration('sampler', 3);
  assert.ok(sg1 > childGen2);
  assert.equal(table.isResourceGenerationCurrent('sampler', 3, sg1), true);

  table.createSampler(3, { magFilter: 'nearest' });
  const sg2 = table.getResourceGeneration('sampler', 3);
  assert.ok(sg2 > sg1);
  assert.equal(table.isResourceGenerationCurrent('sampler', 3, sg1), false);
  assert.equal(table.isResourceGenerationCurrent('sampler', 3, sg2), true);

  assert.equal(table.destroySampler(3), true);
  const sg3 = table.getResourceGeneration('sampler', 3);
  assert.ok(sg3 > sg2);
  assert.equal(table.isResourceGenerationCurrent('sampler', 3, sg2), false);
  assert.equal(table.isResourceGenerationCurrent('sampler', 3, sg3), false);
});

test('ResourceTable stamps cached GPUBindGroups with (handle, generation) tuples and proactively invalidates on destroy/overwrite (W267)', () => {
  const { device, calls } = createMockDevice();
  const table = new ResourceTable(device);
  const layout = { kind: 'GPUBindGroupLayout', id: 'main-bgl' };

  table.createBuffer(1, { size: 64, usage: GPU_BUFFER_USAGE.UNIFORM });
  table.createTexture(2, { size: { width: 128, height: 128 }, mipLevelCount: 4, format: 'rgba8unorm' });
  table.createSampler(3, { magFilter: 'linear' });

  const bGen1 = table.getResourceGeneration('buffer', 1);
  const tGen1 = table.getResourceGeneration('texture', 2);
  const tvGen1 = table.getResourceGeneration('textureView', 2);
  const sGen1 = table.getResourceGeneration('sampler', 3);

  // First call: miss -> creates and caches bind group stamped with exact (handle, generation) tuples.
  const bg1 = table.getOrCreateBindGroup('pass0:group0', {
    layout,
    entries: [
      { binding: 0, kind: 'buffer', handle: 1, offset: 0, size: 64 },
      { binding: 1, kind: 'texture', handle: 2 },
      { binding: 2, kind: 'sampler', handle: 3 },
    ],
    label: 'pass0_group0',
  });
  assert.ok(bg1);
  assert.equal(calls.createBindGroup.length, 1);
  assert.equal(table.hasBindGroup('pass0:group0'), true);

  const meta = table.getBindGroupMeta('pass0:group0');
  assert.ok(meta);
  assert.equal(meta.valid, true);
  assert.deepEqual(meta.stamps, [
    { kind: 'buffer', handle: 1, generation: bGen1 },
    { kind: 'texture', handle: 2, generation: tGen1 },
    { kind: 'textureView', handle: 2, generation: tvGen1 },
    { kind: 'sampler', handle: 3, generation: sGen1 },
  ]);

  // Repeat lookup via getBindGroup and getOrCreateBindGroup -> hits without calling device.createBindGroup!
  assert.strictEqual(table.getBindGroup('pass0:group0'), bg1);
  assert.strictEqual(
    table.getOrCreateBindGroup('pass0:group0', {
      layout,
      entries: [
        { binding: 0, kind: 'buffer', handle: 1, offset: 0, size: 64 },
        { binding: 1, kind: 'texture', handle: 2 },
        { binding: 2, kind: 'sampler', handle: 3 },
      ],
    }),
    bg1
  );
  assert.equal(calls.createBindGroup.length, 1);

  let stats = table.getStats();
  assert.equal(stats.bindGroupHits, 2);
  assert.equal(stats.bindGroupMisses, 1);
  assert.ok(Math.abs(stats.bindGroupHitRate - 2 / 3) < 1e-12);
  assert.equal(stats.staleBindGroupEvictions, 0);
  assert.equal(stats.lruBindGroupEvictions, 0);
  assert.equal(stats.cachedBindGroups, 1);

  // Add a second bind group sharing texture 2 and an unrelated third bind group using only buffer 1.
  const bgShared = table.getOrCreateBindGroup('pass1:group0', {
    layout,
    entries: [{ binding: 0, kind: 'textureView', handle: 2 }],
  });
  const bgUnrelated = table.getOrCreateBindGroup('pass2:bufOnly', {
    layout,
    entries: [{ binding: 0, kind: 'buffer', handle: 1 }],
  });
  assert.ok(bgShared);
  assert.ok(bgUnrelated);
  assert.equal(table.getStats().cachedBindGroups, 3);

  // Destroying texture 2 proactively evicts BOTH pass0:group0 and pass1:group0 (once each!),
  // while leaving pass2:bufOnly completely untouched!
  assert.equal(table.destroyTexture(2), true);
  assert.equal(table.hasBindGroup('pass0:group0'), false);
  assert.equal(table.hasBindGroup('pass1:group0'), false);
  assert.equal(table.hasBindGroup('pass2:bufOnly'), true);

  stats = table.getStats();
  assert.equal(stats.staleBindGroupEvictions, 2);
  assert.equal(stats.cachedBindGroups, 1);

  // Looking up pass0:group0 while texture 2 is destroyed misses (no double-count of stale evictions)
  // and getOrCreateBindGroup returns null without invoking device.createBindGroup.
  const createCountBefore = calls.createBindGroup.length;
  assert.equal(table.getBindGroup('pass0:group0'), undefined);
  assert.equal(
    table.getOrCreateBindGroup('pass0:group0', {
      layout,
      entries: [
        { binding: 0, kind: 'buffer', handle: 1, offset: 0, size: 64 },
        { binding: 1, kind: 'texture', handle: 2 },
        { binding: 2, kind: 'sampler', handle: 3 },
      ],
    }),
    null
  );
  assert.equal(calls.createBindGroup.length, createCountBefore);
  assert.equal(table.getStats().staleBindGroupEvictions, 2);

  // Recreating texture 2 allows creating a fresh bind group stamped with the new generation!
  table.createTexture(2, { size: { width: 256, height: 256 }, format: 'rgba8unorm' });
  const tGen2 = table.getResourceGeneration('texture', 2);
  assert.ok(tGen2 > tGen1);
  const bg1Recreated = table.getOrCreateBindGroup('pass0:group0', {
    layout,
    entries: [
      { binding: 0, kind: 'buffer', handle: 1, offset: 0, size: 64 },
      { binding: 1, kind: 'texture', handle: 2 },
      { binding: 2, kind: 'sampler', handle: 3 },
    ],
  });
  assert.ok(bg1Recreated);
  assert.notStrictEqual(bg1Recreated, bg1);
  const recreatedMeta = table.getBindGroupMeta('pass0:group0');
  assert.equal(
    recreatedMeta.stamps.find((s) => s.kind === 'texture' && s.handle === 2).generation,
    tGen2
  );

  // Overwriting sampler 3 proactively evicts pass0:group0 (which references sampler 3).
  const staleBeforeSampler = table.getStats().staleBindGroupEvictions;
  table.createSampler(3, { magFilter: 'nearest' });
  assert.equal(table.hasBindGroup('pass0:group0'), false);
  assert.equal(table.getStats().staleBindGroupEvictions, staleBeforeSampler + 1);

  // Overwriting buffer 1 proactively evicts pass2:bufOnly.
  table.createBuffer(1, { size: 128, usage: GPU_BUFFER_USAGE.UNIFORM });
  assert.equal(table.hasBindGroup('pass2:bufOnly'), false);
  assert.equal(table.getStats().staleBindGroupEvictions, staleBeforeSampler + 2);
  assert.equal(table.getStats().cachedBindGroups, 0);
});

test('ResourceTable detects and evicts stale bind groups on lookup (outdated generation stamps, external destroy, and stale direct GPU objects) (W267)', () => {
  const { device, calls } = createMockDevice();
  const table = new ResourceTable(device);
  const layout = { kind: 'GPUBindGroupLayout', id: 'lookup-stale-bgl' };

  const buf1 = table.createBuffer(10, { size: 64, usage: GPU_BUFFER_USAGE.UNIFORM });
  const tex1 = table.createTexture(20, { size: { width: 32, height: 32 }, format: 'rgba8unorm' });
  const oldBufGen = table.getResourceGeneration('buffer', 10);

  // Overwrite buffer 10 so oldBufGen is stale.
  const buf2 = table.createBuffer(10, { size: 64, usage: GPU_BUFFER_USAGE.UNIFORM });
  assert.equal(buf1.destroyed, true);
  assert.equal(buf2.destroyed, false);

  // 1. Entry manually cached with an outdated (handle, generation) tuple is detected & evicted on lookup.
  const staleManualBg = { kind: 'GPUBindGroup', id: 'stale-manual' };
  table.setBindGroup('manual-stale', staleManualBg, [
    { kind: 'buffer', handle: 10, generation: oldBufGen },
  ]);
  assert.equal(table.getStats().cachedBindGroups, 1);
  assert.equal(table.hasBindGroup('manual-stale'), false);

  const staleBefore = table.getStats().staleBindGroupEvictions;
  assert.equal(table.getBindGroup('manual-stale'), undefined);
  assert.equal(table.getStats().staleBindGroupEvictions, staleBefore + 1);
  assert.equal(table.getStats().cachedBindGroups, 0);
  assert.equal(table.getStats().bindGroupMisses, 1);

  // 2. Direct GPU object passed to getOrCreateBindGroup(layout, entries) auto-tracks origin and detects external destroy().
  const bgDirect = table.getOrCreateBindGroup(layout, [
    { binding: 0, resource: { buffer: buf2, offset: 0, size: 64 } },
    { binding: 1, resource: table.getTextureView(20) },
  ]);
  assert.ok(bgDirect);
  // Out-of-order entries deduplicate to the exact same cached bind group!
  const bgDirectHit = table.getOrCreateBindGroup(layout, [
    { binding: 1, resource: table.getTextureView(20) },
    { binding: 0, resource: { buffer: buf2, offset: 0, size: 64 } },
  ]);
  assert.strictEqual(bgDirectHit, bgDirect);
  assert.equal(calls.createBindGroup.length, 1);

  // Externally destroy tex1 (bypassing table.destroyTexture) -> lookup MUST detect and evict as stale!
  tex1.destroy();
  const staleBeforeExt = table.getStats().staleBindGroupEvictions;
  const retryAfterExtDestroy = table.getOrCreateBindGroup(layout, [
    { binding: 0, resource: { buffer: buf2, offset: 0, size: 64 } },
    { binding: 1, resource: table.getTextureView(20) },
  ]);
  assert.equal(retryAfterExtDestroy, null);
  assert.equal(table.getStats().staleBindGroupEvictions, staleBeforeExt + 1);
  assert.equal(table.getStats().cachedBindGroups, 0);
  assert.equal(calls.createBindGroup.length, 1);

  // 3. Attempting to build a bind group using a previously overwritten/destroyed direct GPUBuffer (buf1) returns null safely.
  const staleBufResult = table.getOrCreateBindGroup(layout, [
    { binding: 0, resource: { buffer: buf1, offset: 0, size: 64 } },
  ]);
  assert.equal(staleBufResult, null);
  assert.equal(calls.createBindGroup.length, 1);

  // 4. evictStaleBindGroups() sweeps externally destroyed direct resources proactively.
  const extBuf = device.createBuffer({ size: 32, usage: GPU_BUFFER_USAGE.UNIFORM });
  const bgExt = table.getOrCreateBindGroup('ext-key', layout, [
    { binding: 0, resource: { buffer: extBuf } },
  ]);
  assert.ok(bgExt);
  assert.strictEqual(table.getBindGroup('ext-key'), bgExt);
  extBuf.destroy();
  const swept = table.evictStaleBindGroups();
  assert.equal(swept, 1);
  assert.equal(table.getStats().cachedBindGroups, 0);
});

test('ResourceTable enforces bounded LRU cap (maxBindGroups) with recency refresh and accurate lruBindGroupEvictions (W267)', () => {
  const { device } = createMockDevice();
  const table = new ResourceTable(device, { maxBindGroups: 3 });
  assert.equal(table.maxBindGroups, 3);

  table.createBuffer(1, { size: 64, usage: GPU_BUFFER_USAGE.UNIFORM });
  table.createBuffer(2, { size: 64, usage: GPU_BUFFER_USAGE.UNIFORM });
  table.createBuffer(3, { size: 64, usage: GPU_BUFFER_USAGE.UNIFORM });
  table.createBuffer(4, { size: 64, usage: GPU_BUFFER_USAGE.UNIFORM });

  const bgA = { kind: 'GPUBindGroup', name: 'A' };
  const bgB = { kind: 'GPUBindGroup', name: 'B' };
  const bgC = { kind: 'GPUBindGroup', name: 'C' };
  const bgD = { kind: 'GPUBindGroup', name: 'D' };
  const bgE = { kind: 'GPUBindGroup', name: 'E' };

  table.setBindGroup('A', bgA, [{ kind: 'buffer', handle: 1 }]);
  table.setBindGroup('B', bgB, [{ kind: 'buffer', handle: 2 }]);
  table.setBindGroup('C', bgC, [{ kind: 'buffer', handle: 3 }]);
  assert.equal(table.getStats().cachedBindGroups, 3);
  assert.equal(table.getStats().lruBindGroupEvictions, 0);

  // Touch 'A' so LRU order becomes [B (oldest), C, A (newest)].
  assert.strictEqual(table.getBindGroup('A'), bgA);

  // Inserting 'D' must evict 'B' (LRU oldest), preserving 'A', 'C', 'D'.
  table.setBindGroup('D', bgD, [{ kind: 'buffer', handle: 4 }]);
  assert.equal(table.getStats().cachedBindGroups, 3);
  assert.equal(table.getStats().lruBindGroupEvictions, 1);
  assert.equal(table.hasBindGroup('B'), false);
  assert.equal(table.hasBindGroup('A'), true);
  assert.equal(table.hasBindGroup('C'), true);
  assert.equal(table.hasBindGroup('D'), true);

  // Overwriting an existing key ('C') refreshes its recency without triggering an LRU eviction.
  const bgC2 = { kind: 'GPUBindGroup', name: 'C2' };
  table.setBindGroup('C', bgC2, [{ kind: 'buffer', handle: 3 }]);
  assert.equal(table.getStats().cachedBindGroups, 3);
  assert.equal(table.getStats().lruBindGroupEvictions, 1);

  // Now order is [A (oldest), D, C (newest)]; inserting 'E' evicts 'A'.
  table.setBindGroup('E', bgE, [{ kind: 'buffer', handle: 1 }]);
  assert.equal(table.getStats().cachedBindGroups, 3);
  assert.equal(table.getStats().lruBindGroupEvictions, 2);
  assert.equal(table.hasBindGroup('A'), false);
  assert.strictEqual(table.getBindGroup('C'), bgC2);
  assert.strictEqual(table.getBindGroup('D'), bgD);
  assert.strictEqual(table.getBindGroup('E'), bgE);

  // Destroying buffer 2 (whose only bind group 'B' was already LRU-evicted) must NOT increment stale evictions!
  const staleBefore = table.getStats().staleBindGroupEvictions;
  table.destroyBuffer(2);
  assert.equal(table.getStats().staleBindGroupEvictions, staleBefore);

  // maxBindGroups: 0 cleanly disables retention without errors.
  const uncachedTable = new ResourceTable(device, { maxBindGroups: 0 });
  const layout = { kind: 'GPUBindGroupLayout', id: 'zero-cap' };
  uncachedTable.createBuffer(1, { size: 16, usage: GPU_BUFFER_USAGE.UNIFORM });
  const g1 = uncachedTable.getOrCreateBindGroup('k', {
    layout,
    entries: [{ binding: 0, kind: 'buffer', handle: 1 }],
  });
  const g2 = uncachedTable.getOrCreateBindGroup('k', {
    layout,
    entries: [{ binding: 0, kind: 'buffer', handle: 1 }],
  });
  assert.ok(g1);
  assert.ok(g2);
  assert.notStrictEqual(g1, g2);
  assert.equal(uncachedTable.getStats().cachedBindGroups, 0);
  assert.equal(uncachedTable.getStats().bindGroupMisses, 2);
  assert.equal(uncachedTable.getStats().bindGroupHits, 0);
  assert.equal(uncachedTable.getStats().bindGroupHitRate, 0);
  assert.equal(uncachedTable.getStats().lruBindGroupEvictions, 0);
});

test('ResourceTable bind-group cache integrates with descriptor slots, resetStats(), and clear() (W267)', () => {
  const { device, calls } = createMockDevice();
  const table = new ResourceTable(device, { maxBindGroups: 16 });
  const layout = { kind: 'GPUBindGroupLayout', id: 'slot-bgl' };

  table.createBuffer(1, { size: 256, usage: GPU_BUFFER_USAGE.UNIFORM });
  table.createTexture(2, { size: { width: 64, height: 64 }, format: 'rgba8unorm' });
  table.setSlot(0, { kind: 'buffer', handle: 1, offset: 0, size: 64 });
  table.setSlot(1, { kind: 'texture', handle: 2 });

  const bg1 = table.getOrCreateBindGroup({
    key: 'slot-group',
    layout,
    slots: [0, 1],
  });
  assert.ok(bg1);
  assert.strictEqual(
    table.getOrCreateBindGroup({ key: 'slot-group', layout, slots: [0, 1] }),
    bg1
  );
  assert.equal(calls.createBindGroup.length, 1);

  // Rebinding slot 0 to a different offset invalidates the slot-backed bind group on lookup.
  table.setSlot(0, { kind: 'buffer', handle: 1, offset: 64, size: 64 });
  const bg2 = table.getOrCreateBindGroup({
    key: 'slot-group',
    layout,
    slots: [0, 1],
  });
  assert.ok(bg2);
  assert.notStrictEqual(bg2, bg1);
  assert.equal(calls.createBindGroup.length, 2);
  assert.equal(table.getStats().staleBindGroupEvictions, 1);

  // resetStats() clears bind-group counters while preserving cached bind groups & monotonic generations.
  const genBeforeReset = table.getResourceGeneration('buffer', 1);
  table.resetStats();
  const afterReset = table.getStats();
  assert.equal(afterReset.bindGroupHits, 0);
  assert.equal(afterReset.bindGroupMisses, 0);
  assert.equal(afterReset.bindGroupHitRate, 0);
  assert.equal(afterReset.staleBindGroupEvictions, 0);
  assert.equal(afterReset.lruBindGroupEvictions, 0);
  assert.equal(afterReset.cachedBindGroups, 1);
  assert.strictEqual(table.getBindGroup('slot-group'), bg2);
  assert.equal(table.getStats().bindGroupHits, 1);
  assert.equal(table.getStats().bindGroupHitRate, 1);

  // clear() evicts all cached bind groups while preserving strictly monotonic generations.
  table.clear();
  const afterClear = table.getStats();
  assert.equal(afterClear.cachedBindGroups, 0);
  assert.ok(table.getResourceGeneration('buffer', 1) > genBeforeReset);

  table.createBuffer(1, { size: 64, usage: GPU_BUFFER_USAGE.UNIFORM });
  assert.ok(table.getResourceGeneration('buffer', 1) > genBeforeReset + 1);
  table.clear({ resetStats: true });
  const afterClearReset = table.getStats();
  assert.equal(afterClearReset.bindGroupHits, 0);
  assert.equal(afterClearReset.bindGroupMisses, 0);
  assert.equal(afterClearReset.bindGroupHitRate, 0);
  assert.equal(afterClearReset.staleBindGroupEvictions, 0);
  assert.equal(afterClearReset.lruBindGroupEvictions, 0);
  assert.equal(afterClearReset.cachedBindGroups, 0);
});

test('ResourceTable scene transition stress (Start Screen -> 4K Bink2 -> Main Menu) prevents destroyed-resource submits and bounds GPU handles (W267)', () => {
  const { device, calls } = createMockDevice();
  const table = new ResourceTable(device, { maxBindGroups: 32 });
  const uiLayout = { kind: 'GPUBindGroupLayout', id: 'ui' };
  const binkLayout = { kind: 'GPUBindGroupLayout', id: 'bink2-yuv' };
  const menuLayout = { kind: 'GPUBindGroupLayout', id: 'main-menu' };

  // Phase 1: Start Screen (reused across 20 frames -> high cache hit rate).
  table.createBuffer(1, { size: 256, usage: GPU_BUFFER_USAGE.UNIFORM });
  table.createTexture(10, { size: { width: 1920, height: 1080 }, format: 'rgba8unorm' });
  table.createSampler(100, { magFilter: 'linear', minFilter: 'linear' });

  for (let frame = 0; frame < 20; frame += 1) {
    const bg = table.getOrCreateBindGroup('start-screen', {
      layout: uiLayout,
      entries: [
        { binding: 0, kind: 'buffer', handle: 1 },
        { binding: 1, kind: 'texture', handle: 10 },
        { binding: 2, kind: 'sampler', handle: 100 },
      ],
    });
    assert.ok(bg);
  }
  assert.equal(calls.createBindGroup.length, 1);
  assert.equal(table.getStats().bindGroupHits, 19);
  assert.equal(table.getStats().bindGroupMisses, 1);
  assert.equal(table.getStats().bindGroupHitRate, 19 / 20);

  // Phase 2: Transition to 4K Bink2 (destroy Start Screen textures, reuse identical handle IDs 10, 11, 12).
  const startTexGen = table.getResourceGeneration('texture', 10);
  table.destroyTexture(10);
  assert.equal(table.getStats().staleBindGroupEvictions, 1);
  assert.equal(table.getStats().cachedBindGroups, 0);

  // Allocate 4K Y, U, V planes reusing handle 10 + 11, 12.
  table.createTexture(10, { size: { width: 3840, height: 2160 }, format: 'r8unorm' });
  table.createTexture(11, { size: { width: 1920, height: 1080 }, format: 'r8unorm' });
  table.createTexture(12, { size: { width: 1920, height: 1080 }, format: 'r8unorm' });
  assert.ok(table.getResourceGeneration('texture', 10) > startTexGen);

  for (let frame = 0; frame < 60; frame += 1) {
    const binkBg = table.getOrCreateBindGroup('bink2-4k', {
      layout: binkLayout,
      entries: [
        { binding: 0, kind: 'buffer', handle: 1 },
        { binding: 1, kind: 'textureView', handle: 10 },
        { binding: 2, kind: 'textureView', handle: 11 },
        { binding: 3, kind: 'textureView', handle: 12 },
        { binding: 4, kind: 'sampler', handle: 100 },
      ],
    });
    assert.ok(binkBg);
  }
  assert.equal(calls.createBindGroup.length, 2);

  // Phase 3: Transition to Main Menu (destroy Bink2 planes + stream 128 unique draw bind groups under cap 32).
  table.destroyTexture(10);
  table.destroyTexture(11);
  table.destroyTexture(12);
  // All 3 planes were in 'bink2-4k', so 'bink2-4k' is evicted EXACTLY ONCE on destroyTexture(10)!
  assert.equal(table.getStats().staleBindGroupEvictions, 2);
  assert.equal(table.getStats().cachedBindGroups, 0);

  for (let i = 0; i < 128; i += 1) {
    const texId = 500 + i;
    table.createTexture(texId, { size: { width: 256, height: 256 }, format: 'rgba8unorm' });
    const bg = table.getOrCreateBindGroup(menuLayout, [
      { binding: 0, kind: 'buffer', handle: 1 },
      { binding: 1, kind: 'texture', handle: texId },
      { binding: 2, kind: 'sampler', handle: 100 },
    ]);
    assert.ok(bg);
    assert.ok(table.getStats().cachedBindGroups <= 32);
  }

  const finalStats = table.getStats();
  assert.equal(finalStats.cachedBindGroups, 32);
  assert.equal(finalStats.lruBindGroupEvictions, 96);
  assert.equal(finalStats.staleBindGroupEvictions, 2);
  assert.ok(finalStats.bindGroupHitRate >= 0 && finalStats.bindGroupHitRate <= 1);
  for (const [k, v] of Object.entries(finalStats)) {
    assert.equal(typeof v, 'number', `stat ${k} must be numeric`);
    assert.ok(Number.isFinite(v), `stat ${k} must be finite`);
  }
});

test('computeBindGroupHashFast produces deterministic 64-bit dual-u32 hashes sensitive to layout, slots, handles, generations, and ranges (W320)', async () => {
  const esmMod = await import('../web/workers/resource_table.js');
  assert.equal(typeof computeBindGroupHashFast, 'function');
  assert.equal(esmMod.computeBindGroupHashFast, computeBindGroupHashFast);
  assert.equal(globalThis.PlayGta6ResourceTable.computeBindGroupHashFast, computeBindGroupHashFast);
  assert.equal(DEFAULT_FAST_BIND_GROUP_BUCKETS, 512);
  assert.equal(esmMod.DEFAULT_FAST_BIND_GROUP_BUCKETS, 512);

  const pairA = new Uint32Array(2);
  const pairB = new Uint32Array(2);
  const slots = new Uint32Array([0, 1, 2]);
  const handles = new Uint32Array([10, 20, 30]);
  const gens = new Uint32Array([1, 2, 3]);
  const kinds = new Uint8Array([1, 2, 4]);
  const offsets = new Float64Array([0, 0, 0]);
  const sizes = new Float64Array([256, -1, -1]);

  const h1 = computeBindGroupHashFast(1, 3, slots, handles, gens, pairA, kinds, offsets, sizes);
  const h2 = computeBindGroupHashFast(
    1,
    3,
    [0, 1, 2],
    [10, 20, 30],
    [1, 2, 3],
    pairB,
    [1, 2, 4],
    [0, 0, 0],
    [256, -1, -1]
  );

  assert.equal(typeof h1, 'number');
  assert.equal(Number.isSafeInteger(h1), true);
  assert.ok(h1 >= 0);
  assert.equal(h1, h2);
  assert.equal(pairA[0], pairB[0]);
  assert.equal(pairA[1], pairB[1]);

  // Changing any single component (layoutId, entryCount, slot, handle, generation, kind, offset, size) alters hash.
  const variants = [
    computeBindGroupHashFast(2, 3, slots, handles, gens, null, kinds, offsets, sizes),
    computeBindGroupHashFast(1, 2, slots, handles, gens, null, kinds, offsets, sizes),
    computeBindGroupHashFast(1, 3, new Uint32Array([0, 1, 3]), handles, gens, null, kinds, offsets, sizes),
    computeBindGroupHashFast(1, 3, slots, new Uint32Array([10, 21, 30]), gens, null, kinds, offsets, sizes),
    computeBindGroupHashFast(1, 3, slots, handles, new Uint32Array([1, 2, 4]), null, kinds, offsets, sizes),
    computeBindGroupHashFast(1, 3, slots, handles, gens, null, new Uint8Array([1, 3, 4]), offsets, sizes),
    computeBindGroupHashFast(1, 3, slots, handles, gens, null, kinds, new Float64Array([64, 0, 0]), sizes),
    computeBindGroupHashFast(1, 3, slots, handles, gens, null, kinds, offsets, new Float64Array([512, -1, -1])),
    computeBindGroupHashFast('layout-str', 3, slots, handles, gens),
    computeBindGroupHashFast(1, 0, slots, handles, gens),
  ];
  for (let i = 0; i < variants.length; i += 1) {
    assert.equal(Number.isSafeInteger(variants[i]), true);
    assert.notEqual(variants[i], h1, `variant ${i} must differ from baseline hash`);
  }
});

test('ResourceTable zero-alloc repeated bind-group probes bypass string key building on steady-state hits (W320)', () => {
  const { device, calls } = createMockDevice();
  const table = new ResourceTable(device);
  const layoutA = { kind: 'GPUBindGroupLayout', id: 'draw-a' };
  const layoutB = { kind: 'GPUBindGroupLayout', id: 'draw-b' };

  const buf1 = table.createBuffer(1, { size: 512, usage: GPU_BUFFER_USAGE.UNIFORM });
  table.createTexture(2, { size: { width: 128, height: 128 }, format: 'rgba8unorm' });
  const view2 = table.getTextureView(2);
  table.createSampler(3, { magFilter: 'linear' });

  table.setSlot(0, { kind: 'buffer', handle: 1, offset: 0, size: 64 });
  table.setSlot(1, { kind: 'texture', handle: 2 });

  // Instrument slow-path methods to verify ZERO invocations on steady-state hits!
  let resolveSlowCalls = 0;
  let layoutKeyStringCalls = 0;
  const origResolve = table._resolveBindGroupEntries.bind(table);
  const origGetLayoutId = table._getLayoutId.bind(table);
  table._resolveBindGroupEntries = (...args) => {
    resolveSlowCalls += 1;
    return origResolve(...args);
  };
  table._getLayoutId = (...args) => {
    layoutKeyStringCalls += 1;
    return origGetLayoutId(...args);
  };

  // Warm cache: handle-based, different buffer offset, different layout, direct GPU entries, and slot-backed.
  const entriesA = [
    { binding: 0, kind: 'buffer', handle: 1, offset: 0, size: 64 },
    { binding: 1, kind: 'texture', handle: 2 },
    { binding: 2, kind: 'sampler', handle: 3 },
  ];
  const entriesAOffset = [
    { binding: 0, kind: 'buffer', handle: 1, offset: 256, size: 64 },
    { binding: 1, kind: 'texture', handle: 2 },
    { binding: 2, kind: 'sampler', handle: 3 },
  ];
  const entriesDirect = [
    { binding: 0, resource: { buffer: buf1, offset: 0, size: 64 } },
    { binding: 1, resource: view2 },
  ];
  const entriesDirectUnsorted = [
    { binding: 1, resource: view2 },
    { binding: 0, resource: { buffer: buf1, offset: 0, size: 64 } },
  ];

  const bgA = table.getOrCreateBindGroup(layoutA, entriesA);
  const bgAOffset = table.getOrCreateBindGroup(layoutA, entriesAOffset);
  const bgB = table.getOrCreateBindGroup(layoutB, entriesA);
  const bgDirect = table.getOrCreateBindGroup(layoutA, entriesDirect);
  const bgSlots = table.getOrCreateBindGroup({ layout: layoutA, slots: [0, 1] });

  assert.ok(bgA && bgAOffset && bgB && bgDirect && bgSlots);
  assert.notStrictEqual(bgA, bgAOffset);
  assert.notStrictEqual(bgA, bgB);
  assert.notStrictEqual(bgA, bgDirect);
  assert.equal(calls.createBindGroup.length, 5);
  assert.equal(resolveSlowCalls, 5);
  assert.equal(layoutKeyStringCalls, 5);

  // Steady-state 30 FPS draw loop: 200 probes across all 5 bind groups (including out-of-order entries).
  for (let i = 0; i < 40; i += 1) {
    assert.strictEqual(table.getOrCreateBindGroup(layoutA, entriesA), bgA);
    assert.strictEqual(table.getOrCreateBindGroup(layoutA, entriesAOffset), bgAOffset);
    assert.strictEqual(table.getOrCreateBindGroup(layoutB, entriesA), bgB);
    assert.strictEqual(table.getOrCreateBindGroup(layoutA, entriesDirectUnsorted), bgDirect);
    assert.strictEqual(table.getOrCreateBindGroup({ layout: layoutA, slots: [0, 1] }), bgSlots);
  }

  // Slow path & string key builders were NEVER called during the 200 steady-state probes!
  assert.equal(calls.createBindGroup.length, 5);
  assert.equal(resolveSlowCalls, 5);
  assert.equal(layoutKeyStringCalls, 5);

  const stats = table.getStats();
  assert.equal(stats.zeroAllocKeyProbes, 205);
  assert.equal(stats.zeroAllocKeyHits, 200);
  assert.equal(stats.hashCollisions, 0);
  assert.equal(stats.bindGroupHits, 200);
  assert.equal(stats.bindGroupMisses, 5);
  assert.ok(Math.abs(stats.bindGroupHitRate - 200 / 205) < 1e-12);
});

test('ResourceTable zero-alloc fast-cache invalidates on generation bumps, destroys, and slot mutations (W320)', () => {
  const { device, calls } = createMockDevice();
  const table = new ResourceTable(device);
  const layout = { kind: 'GPUBindGroupLayout', id: 'gen-inval-bgl' };

  table.createBuffer(1, { size: 128, usage: GPU_BUFFER_USAGE.UNIFORM });
  const tex1 = table.createTexture(2, { size: { width: 64, height: 64 }, format: 'rgba8unorm' });
  table.createSampler(3, { magFilter: 'linear' });

  const entries = [
    { binding: 0, kind: 'buffer', handle: 1, offset: 0, size: 64 },
    { binding: 1, kind: 'texture', handle: 2 },
    { binding: 2, kind: 'sampler', handle: 3 },
  ];

  const bg1 = table.getOrCreateBindGroup(layout, entries);
  assert.strictEqual(table.getOrCreateBindGroup(layout, entries), bg1);
  assert.equal(table.getStats().zeroAllocKeyHits, 1);
  assert.equal(calls.createBindGroup.length, 1);

  // Overwriting buffer 1 bumps generation and proactively evicts fast-cache + LRU entry.
  table.createBuffer(1, { size: 256, usage: GPU_BUFFER_USAGE.UNIFORM });
  assert.equal(table.getStats().staleBindGroupEvictions, 1);
  assert.equal(table.getStats().cachedBindGroups, 0);

  const bg2 = table.getOrCreateBindGroup(layout, entries);
  assert.ok(bg2);
  assert.notStrictEqual(bg2, bg1);
  assert.strictEqual(table.getOrCreateBindGroup(layout, entries), bg2);
  assert.equal(calls.createBindGroup.length, 2);
  assert.equal(table.getStats().zeroAllocKeyHits, 2);

  // Overwriting default textureView bumps textureView generation and evicts fast-cache entry.
  table.createTextureView(2, { baseMipLevel: 0, mipLevelCount: 1 });
  assert.equal(table.getStats().staleBindGroupEvictions, 2);
  const bg3 = table.getOrCreateBindGroup(layout, entries);
  assert.ok(bg3);
  assert.notStrictEqual(bg3, bg2);
  assert.strictEqual(table.getOrCreateBindGroup(layout, entries), bg3);
  assert.equal(calls.createBindGroup.length, 3);

  // External texture destroy() (bypassing destroyTexture) is caught safely and evicts stale entry.
  tex1.destroy();
  const staleBeforeExt = table.getStats().staleBindGroupEvictions;
  assert.equal(table.getOrCreateBindGroup(layout, entries), null);
  assert.equal(table.getStats().staleBindGroupEvictions, staleBeforeExt + 1);
  assert.equal(table.getStats().cachedBindGroups, 0);
  assert.equal(calls.createBindGroup.length, 3);

  // Slot-backed fast-cache invalidates cleanly when setSlot() changes offset.
  table.createTexture(2, { size: { width: 64, height: 64 }, format: 'rgba8unorm' });
  table.setSlot(0, { kind: 'buffer', handle: 1, offset: 0, size: 64 });
  table.setSlot(1, { kind: 'texture', handle: 2 });
  const bgSlot1 = table.getOrCreateBindGroup({ layout, slots: [0, 1] });
  assert.strictEqual(table.getOrCreateBindGroup({ layout, slots: [0, 1] }), bgSlot1);

  table.setSlot(0, { kind: 'buffer', handle: 1, offset: 64, size: 64 });
  const bgSlot2 = table.getOrCreateBindGroup({ layout, slots: [0, 1] });
  assert.ok(bgSlot2);
  assert.notStrictEqual(bgSlot2, bgSlot1);
  assert.strictEqual(table.getOrCreateBindGroup({ layout, slots: [0, 1] }), bgSlot2);
});

test('ResourceTable forced 64-bit hash-collision verification guarantees ZERO wrong bind groups (W320)', () => {
  const { device, calls } = createMockDevice();
  // Force EVERY bind group to produce the exact same 64-bit (lo, hi) hash!
  const table = new ResourceTable(device, {
    maxBindGroups: 32,
    fastBindGroupBuckets: 8,
    bindGroupHashFn: (_layoutId, _count, _slots, _handles, _gens, outPair) => {
      if (outPair) {
        outPair[0] = 0xdeadbeef;
        outPair[1] = 0xcafebabe;
      }
      return 0xdeadbeef;
    },
  });

  const layoutA = { kind: 'GPUBindGroupLayout', id: 'col-a' };
  const layoutB = { kind: 'GPUBindGroupLayout', id: 'col-b' };

  // Phase A: 6 distinct bind groups all sharing the exact same 64-bit hash inside the 8-slot window.
  for (let i = 1; i <= 12; i += 1) {
    table.createBuffer(i, { size: 256, usage: GPU_BUFFER_USAGE.UNIFORM });
  }

  const created = [];
  for (let i = 1; i <= 4; i += 1) {
    const bg = table.getOrCreateBindGroup(layoutA, [
      { binding: 0, kind: 'buffer', handle: i, offset: 0, size: 64 },
    ]);
    assert.ok(bg);
    created.push(bg);
  }
  // Distinct offset on buffer 1 and distinct layout on buffer 1 must also never alias!
  const bgBuf1Offset = table.getOrCreateBindGroup(layoutA, [
    { binding: 0, kind: 'buffer', handle: 1, offset: 64, size: 64 },
  ]);
  const bgBuf1LayoutB = table.getOrCreateBindGroup(layoutB, [
    { binding: 0, kind: 'buffer', handle: 1, offset: 0, size: 64 },
  ]);
  assert.equal(calls.createBindGroup.length, 6);

  // Re-probing all 6 colliding bind groups hits directly in the fast-cache with exact collision verification!
  for (let i = 1; i <= 4; i += 1) {
    assert.strictEqual(
      table.getOrCreateBindGroup(layoutA, [
        { binding: 0, kind: 'buffer', handle: i, offset: 0, size: 64 },
      ]),
      created[i - 1]
    );
  }
  assert.strictEqual(
    table.getOrCreateBindGroup(layoutA, [
      { binding: 0, kind: 'buffer', handle: 1, offset: 64, size: 64 },
    ]),
    bgBuf1Offset
  );
  assert.strictEqual(
    table.getOrCreateBindGroup(layoutB, [
      { binding: 0, kind: 'buffer', handle: 1, offset: 0, size: 64 },
    ]),
    bgBuf1LayoutB
  );
  assert.equal(table.getStats().zeroAllocKeyHits, 6);
  assert.ok(table.getStats().hashCollisions > 0);
  assert.equal(calls.createBindGroup.length, 6);

  // Phase B: Add 8 more colliding bind groups (total 14 > 8-slot window!) to test overflow
  // fallback to _bindGroupCache with zero duplicate creates and zero cross-contamination.
  for (let i = 5; i <= 12; i += 1) {
    const bg = table.getOrCreateBindGroup(layoutA, [
      { binding: 0, kind: 'buffer', handle: i, offset: 0, size: 64 },
    ]);
    assert.ok(bg);
    created.push(bg);
  }
  assert.equal(calls.createBindGroup.length, 14);
  assert.equal(new Set([...created, bgBuf1Offset, bgBuf1LayoutB]).size, 14);

  // Re-probing all 14 distinct bind groups must return the exact matching object every time
  // with ZERO extra device.createBindGroup calls!
  for (let pass = 0; pass < 3; pass += 1) {
    for (let i = 1; i <= 12; i += 1) {
      const hit = table.getOrCreateBindGroup(layoutA, [
        { binding: 0, kind: 'buffer', handle: i, offset: 0, size: 64 },
      ]);
      assert.strictEqual(hit, created[i - 1], `buffer ${i} must never collide with another bind group`);
    }
    assert.strictEqual(
      table.getOrCreateBindGroup(layoutA, [
        { binding: 0, kind: 'buffer', handle: 1, offset: 64, size: 64 },
      ]),
      bgBuf1Offset
    );
    assert.strictEqual(
      table.getOrCreateBindGroup(layoutB, [
        { binding: 0, kind: 'buffer', handle: 1, offset: 0, size: 64 },
      ]),
      bgBuf1LayoutB
    );
  }

  assert.equal(calls.createBindGroup.length, 14);
  const stats = table.getStats();
  assert.ok(stats.hashCollisions > 0, 'forced hash collisions must be recorded in telemetry');
  assert.ok(stats.zeroAllocKeyProbes >= 62);
  assert.ok(stats.zeroAllocKeyHits >= 6);
  assert.equal(stats.bindGroupHits, 48);
  assert.equal(stats.bindGroupMisses, 14);
  assert.equal(stats.cachedBindGroups, 14);
});

test('ResourceTable zero-alloc fast-cache preserves bounded LRU recency, clear(), and resetStats() (W320)', () => {
  const { device, calls } = createMockDevice();
  assert.throws(() => new ResourceTable(device, { fastBindGroupBuckets: -1 }), RangeError);
  assert.throws(() => new ResourceTable(device, { fastBindGroupBuckets: 2.5 }), RangeError);

  const table = new ResourceTable(device, { maxBindGroups: 3, fastBindGroupBuckets: 16 });
  const layout = { kind: 'GPUBindGroupLayout', id: 'lru-fast-bgl' };

  for (let i = 1; i <= 4; i += 1) {
    table.createBuffer(i, { size: 64, usage: GPU_BUFFER_USAGE.UNIFORM });
  }

  const e1 = [{ binding: 0, kind: 'buffer', handle: 1 }];
  const e2 = [{ binding: 0, kind: 'buffer', handle: 2 }];
  const e3 = [{ binding: 0, kind: 'buffer', handle: 3 }];
  const e4 = [{ binding: 0, kind: 'buffer', handle: 4 }];

  const bg1 = table.getOrCreateBindGroup(layout, e1);
  const bg2 = table.getOrCreateBindGroup(layout, e2);
  const bg3 = table.getOrCreateBindGroup(layout, e3);
  assert.equal(calls.createBindGroup.length, 3);

  // Zero-alloc hit on e1 refreshes its LRU recency so order becomes [e2 (oldest), e3, e1 (newest)].
  assert.strictEqual(table.getOrCreateBindGroup(layout, e1), bg1);
  assert.equal(table.getStats().zeroAllocKeyHits, 1);

  // Inserting e4 evicts e2 (LRU oldest) from BOTH _bindGroupCache and fast-cache!
  const bg4 = table.getOrCreateBindGroup(layout, e4);
  assert.ok(bg4);
  assert.equal(table.getStats().lruBindGroupEvictions, 1);
  assert.equal(table.getStats().cachedBindGroups, 3);

  // e1, e3, e4 still hit via zero-alloc fast-cache without creating new bind groups.
  assert.strictEqual(table.getOrCreateBindGroup(layout, e1), bg1);
  assert.strictEqual(table.getOrCreateBindGroup(layout, e3), bg3);
  assert.strictEqual(table.getOrCreateBindGroup(layout, e4), bg4);
  assert.equal(calls.createBindGroup.length, 4);
  assert.equal(table.getStats().zeroAllocKeyHits, 4);

  // Re-probing evicted e2 misses and creates a brand-new GPUBindGroup (evicting e1, now oldest).
  const bg2Recreated = table.getOrCreateBindGroup(layout, e2);
  assert.ok(bg2Recreated);
  assert.notStrictEqual(bg2Recreated, bg2);
  assert.equal(calls.createBindGroup.length, 5);
  assert.equal(table.getStats().lruBindGroupEvictions, 2);

  // resetStats() clears zeroAllocKeyProbes, zeroAllocKeyHits, and hashCollisions while keeping cached entries live.
  table.resetStats();
  let stats = table.getStats();
  assert.equal(stats.zeroAllocKeyProbes, 0);
  assert.equal(stats.zeroAllocKeyHits, 0);
  assert.equal(stats.hashCollisions, 0);
  assert.equal(stats.bindGroupHits, 0);
  assert.equal(stats.bindGroupMisses, 0);
  assert.equal(stats.cachedBindGroups, 3);

  // Live cached entry e4 immediately hits via zero-alloc fast-cache after resetStats().
  assert.strictEqual(table.getOrCreateBindGroup(layout, e4), bg4);
  stats = table.getStats();
  assert.equal(stats.zeroAllocKeyProbes, 1);
  assert.equal(stats.zeroAllocKeyHits, 1);
  assert.equal(stats.hashCollisions, 0);
  assert.equal(stats.bindGroupHits, 1);
  assert.equal(stats.bindGroupHitRate, 1);

  // clear({ resetStats: true }) clears fast-cache slots and resets all counters cleanly.
  table.clear({ resetStats: true });
  stats = table.getStats();
  assert.equal(stats.cachedBindGroups, 0);
  assert.equal(stats.zeroAllocKeyProbes, 0);
  assert.equal(stats.zeroAllocKeyHits, 0);
  assert.equal(stats.hashCollisions, 0);

  // fastBindGroupBuckets: 0 disables fast-cache cleanly while preserving W267 caching.
  const noFastTable = new ResourceTable(device, { fastBindGroupBuckets: 0 });
  noFastTable.createBuffer(1, { size: 64, usage: GPU_BUFFER_USAGE.UNIFORM });
  const nf1 = noFastTable.getOrCreateBindGroup(layout, e1);
  const nf2 = noFastTable.getOrCreateBindGroup(layout, e1);
  assert.strictEqual(nf1, nf2);
  assert.equal(noFastTable.getStats().zeroAllocKeyProbes, 0);
  assert.equal(noFastTable.getStats().zeroAllocKeyHits, 0);
  assert.equal(noFastTable.getStats().bindGroupHits, 1);

  for (const [k, v] of Object.entries(stats)) {
    assert.equal(typeof v, 'number', `stat ${k} must be numeric`);
    assert.ok(Number.isFinite(v), `stat ${k} must be finite`);
  }
});
