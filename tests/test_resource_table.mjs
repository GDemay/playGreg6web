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
          return { kind: 'GPUTextureView', textureHandle: tex.handle, viewDesc };
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
