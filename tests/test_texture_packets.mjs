import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const { WgpuCommandProcessor, OPCODES } = require('../web/workers/wgpu_worker.js');
const { installRenderStatePackets, TEXTURE_FORMAT_NAMES: RS_FORMATS } = require('../web/workers/render_state_packets.js');
const { GpuStagingPool } = require('../web/workers/staging_pool.js');
const tp = require('../web/workers/texture_packets.js');

const WORKERS_DIR = fileURLToPath(new URL('../web/workers/', import.meta.url));
const repoFile = (rel) => readFileSync(new URL(`../${rel}`, import.meta.url), 'utf8');

function createMockWebGpu({ strictWebGpu = false, features = undefined } = {}) {
  const calls = [];
  const buffers = new Map();
  let nextHandle = 1;

  const device = {
    ...(features !== undefined ? { features } : {}),
    createBuffer(desc) {
      const backing = new Uint8Array(desc.size);
      const buffer = {
        kind: 'GPUBuffer',
        handle: nextHandle++,
        size: desc.size,
        usage: desc.usage,
        desc,
        backing,
        destroyed: false,
        destroy() {
          buffer.destroyed = true;
        },
      };
      buffers.set(buffer.handle, buffer);
      calls.push({ op: 'createBuffer', desc, result: buffer });
      return buffer;
    },
    createTexture(desc) {
      if (desc.format === 'bc7-rgba-unorm' && desc.sampleCount > 1) {
        throw new Error('not multisampleable');
      }
      if (
        strictWebGpu &&
        desc.dimension === '3d' &&
        typeof desc.format === 'string' &&
        desc.format.startsWith('bc') &&
        !(features && typeof features.has === 'function' && features.has('texture-compression-bc-sliced-3d'))
      ) {
        throw new Error(
          `WebGPU validation: 3d texture format ${desc.format} requires texture-compression-bc-sliced-3d`
        );
      }
      const texture = {
        kind: 'GPUTexture',
        handle: nextHandle++,
        format: desc.format,
        desc,
        destroyed: false,
        createView(viewDesc) {
          if (
            strictWebGpu &&
            viewDesc &&
            viewDesc.format &&
            viewDesc.format !== desc.format
          ) {
            const allowed = Array.isArray(desc.viewFormats) ? desc.viewFormats : [];
            if (!allowed.includes(viewDesc.format)) {
              throw new Error(
                `WebGPU validation: texture view format ${viewDesc.format} is not in texture viewFormats for ${desc.format}`
              );
            }
          }
          return { kind: 'GPUTextureView', textureHandle: texture.handle, viewDesc };
        },
        destroy() {
          texture.destroyed = true;
        },
      };
      calls.push({ op: 'createTexture', desc, result: texture });
      return texture;
    },
    createCommandEncoder() {
      const encoder = {
        beginRenderPass: () => ({ end() {}, setPipeline() {}, draw() {} }),
        beginComputePass: () => ({ end() {} }),
        copyBufferToBuffer(srcBuf, srcOff, dstBuf, dstOff, size) {
          if (strictWebGpu && size % 4 !== 0) {
            throw new Error(`WebGPU validation: copyBufferToBuffer size (${size}) not 4B aligned`);
          }
          if (srcBuf.backing && dstBuf.backing) {
            dstBuf.backing.set(srcBuf.backing.subarray(srcOff, srcOff + size), dstOff);
          }
          calls.push({ op: 'copyBufferToBuffer', srcBuf, srcOff, dstBuf, dstOff, size });
        },
        copyBufferToTexture(source, destination, copySize) {
          if (
            strictWebGpu &&
            source.bytesPerRow !== undefined &&
            source.bytesPerRow % 256 !== 0
          ) {
            throw new Error(
              `WebGPU validation: copyBufferToTexture bytesPerRow (${source.bytesPerRow}) must be a multiple of 256`
            );
          }
          const snapshot =
            source.buffer && source.buffer.backing
              ? source.buffer.backing.slice(source.offset || 0)
              : null;
          calls.push({
            op: 'copyBufferToTexture',
            source: { ...source },
            destination: { ...destination },
            copySize: { ...copySize },
            bytes: snapshot,
          });
        },
        finish: () => ({ kind: 'GPUCommandBuffer' }),
      };
      return encoder;
    },
    queue: {
      writeBuffer(buffer, bufferOffset, data) {
        if (
          strictWebGpu &&
          typeof SharedArrayBuffer !== 'undefined' &&
          data &&
          data.buffer instanceof SharedArrayBuffer
        ) {
          throw new TypeError('WebGPU validation: SharedArrayBuffer view disallowed in writeBuffer');
        }
        const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        if (buffer && buffer.backing) {
          buffer.backing.set(bytes, bufferOffset);
        }
        calls.push({
          op: 'writeBuffer',
          buffer,
          bufferOffset,
          byteLength: bytes.byteLength,
          isSab: typeof SharedArrayBuffer !== 'undefined' && data.buffer instanceof SharedArrayBuffer,
          bytes: bytes.slice(),
        });
      },
      writeTexture(destination, data, dataLayout, size) {
        if (
          strictWebGpu &&
          typeof SharedArrayBuffer !== 'undefined' &&
          data &&
          data.buffer instanceof SharedArrayBuffer
        ) {
          throw new TypeError('WebGPU validation: SharedArrayBuffer view disallowed in writeTexture');
        }
        const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        calls.push({
          op: 'writeTexture',
          destination: { ...destination },
          dataLayout: { ...dataLayout },
          size: { ...size },
          isSab: typeof SharedArrayBuffer !== 'undefined' && data.buffer instanceof SharedArrayBuffer,
          dataRef: data,
          bufferRef: data.buffer,
          byteOffset: data.byteOffset,
          byteLength: data.byteLength,
          bytes: bytes.slice(),
        });
      },
      submit() {
        calls.push({ op: 'submit' });
      },
    },
  };
  const ctx = { getCurrentTexture: () => ({ createView: () => ({ kind: 'swapchain' }) }) };
  return { device, ctx, calls };
}

function createTexture1Packet({
  id,
  width,
  height,
  layers = 1,
  mips = 1,
  format = 0,
  usage = 0x06,
  dimension = 1,
}) {
  const out = new Uint8Array(8 + 32);
  const view = new DataView(out.buffer);
  view.setUint16(0, OPCODES.CREATE_TEXTURE, true);
  view.setUint32(4, 32, true);
  const p = 8;
  view.setUint32(p, id, true);
  view.setUint32(p + 4, width, true);
  view.setUint32(p + 8, height, true);
  view.setUint32(p + 12, layers, true);
  view.setUint32(p + 16, mips, true);
  view.setUint32(p + 20, format, true);
  view.setUint32(p + 24, usage, true);
  view.setUint32(p + 28, dimension, true);
  return out;
}

function createTexture2Packet({
  id,
  width,
  height,
  layers = 1,
  mips = 1,
  samples = 1,
  usage = 0x14,
  dimension = 1,
  format,
  viewFormats = [],
}) {
  const out = new Uint8Array(8 + 40);
  const view = new DataView(out.buffer);
  view.setUint16(0, 35, true);
  view.setUint32(4, 40, true);
  const p = 8;
  view.setUint32(p, id, true);
  view.setUint32(p + 4, width, true);
  view.setUint32(p + 8, height, true);
  view.setUint32(p + 12, layers, true);
  view.setUint32(p + 16, mips, true);
  view.setUint32(p + 20, samples, true);
  view.setUint32(p + 24, usage, true);
  view.setUint8(p + 28, dimension);
  view.setUint8(p + 29, format);
  view.setUint8(p + 30, viewFormats.length);
  viewFormats.forEach((f, i) => view.setUint8(p + 32 + i, f));
  return out;
}

function writeTexturePacket({
  textureId,
  mipLevel = 0,
  originX = 0,
  originY = 0,
  originZ = 0,
  width,
  height,
  layers = 1,
  bytesPerRow = 0,
  rowsPerImage = 0,
  data,
  useSab = false,
}) {
  const totalBytes = 8 + 48 + data.byteLength;
  const backing = useSab ? new SharedArrayBuffer(totalBytes) : new ArrayBuffer(totalBytes);
  const out = new Uint8Array(backing);
  const view = new DataView(backing);
  view.setUint16(0, OPCODES.WRITE_TEXTURE, true);
  view.setUint32(4, 48 + data.byteLength, true);
  const p = 8;
  view.setUint32(p, textureId, true);
  view.setUint32(p + 4, mipLevel, true);
  view.setUint32(p + 8, originX, true);
  view.setUint32(p + 12, originY, true);
  view.setUint32(p + 16, originZ, true);
  view.setUint32(p + 20, width, true);
  view.setUint32(p + 24, height, true);
  view.setUint32(p + 28, layers, true);
  view.setUint32(p + 32, bytesPerRow, true);
  view.setUint32(p + 36, rowsPerImage, true);
  view.setUint32(p + 40, data.byteLength, true);
  view.setUint32(p + 44, 0, true);
  out.set(data, p + 48);
  return out;
}

function destroyPacket(id) {
  const out = new Uint8Array(12);
  const view = new DataView(out.buffer);
  view.setUint16(0, OPCODES.DESTROY_RESOURCE, true);
  view.setUint32(4, 4, true);
  view.setUint32(8, id, true);
  return out;
}

test('CreateTexture2 opcode and payload size match the ABI header', () => {
  assert.equal(OPCODES.CREATE_TEXTURE2, 35);
  assert.equal(tp.CREATE_TEXTURE2, 35);
  assert.equal(tp.CREATE_TEXTURE2_BYTES, 40);
  assert.equal(tp.CREATE_TEXTURE, 12);
  assert.equal(tp.WRITE_TEXTURE, 13);
  assert.equal(tp.WEBGPU_BYTES_PER_ROW_ALIGNMENT, 256);
  const header = repoFile('runtime/abi/playgta6_abi.h');
  assert.match(header, /CreateTexture2 = 35,/);
  assert.match(header, /sizeof\(GpuCreateTexture2Payload\) == 40/);
  // Same v2 format code table as the render state handler (docs/gpu_commands.md).
  assert.deepEqual([...tp.TEXTURE_FORMAT_NAMES], [...RS_FORMATS]);
});

test('CreateTexture2 creates a texture with format, samples and view formats in the shared tables', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  installRenderStatePackets(processor);
  const handler = tp.installTexturePackets(processor);

  // rgba16float (40) 4x MSAA render target; bgra8unorm (27) with an sRGB view format (28).
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 7, width: 640, height: 360, samples: 4, usage: 0x10, format: 40 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 8,
      width: 256,
      height: 128,
      layers: 6,
      mips: 9,
      usage: 0x06,
      format: 27,
      viewFormats: [28],
    })
  );
  assert.deepEqual(handler.getStats(), {
    created: 2,
    errors: 0,
    createTexture2Count: 2,
    rowPitchRepackCount: 0,
    sabCopyCount: 0,
    bc3dFallbacks: 0,
    bc3dFallbackUploadsSkipped: 0,
    uploadCount: 0,
    uploadBytes: 0,
    zeroCopyUploads: 0,
    scratchRepackUploads: 0,
    directZeroCopyUploads: 0,
    repackedUploads: 0,
    scratchBufferReuses: 0,
    scratchBufferAllocations: 0,
    scratchBufferBytes: 0,
    r8unormUploads: 0,
    bytesUploaded: 0,
    srgbSiblingViewFormatsAdded: 0,
  });
  const [rt, tex] = gpu.calls.filter((c) => c.op === 'createTexture').map((c) => c.desc);
  assert.deepEqual(rt, {
    size: { width: 640, height: 360, depthOrArrayLayers: 1 },
    mipLevelCount: 1,
    sampleCount: 4,
    dimension: '2d',
    format: 'rgba16float',
    usage: 0x10,
  });
  assert.equal(tex.format, 'bgra8unorm');
  assert.deepEqual(tex.viewFormats, ['bgra8unorm-srgb']);
  assert.equal(tex.size.depthOrArrayLayers, 6);
  assert.equal(tex.mipLevelCount, 9);
  assert.ok(processor.textures.get(7));
  assert.ok(processor.textureViews.get(8));

  // Re-creating an id replaces (and destroys) the previous texture; DestroyResource removes it.
  const first = processor.textures.get(7);
  processor.executeCommandBuffer(createTexture2Packet({ id: 7, width: 64, height: 64, format: 22 }));
  assert.equal(first.destroyed, true);
  assert.equal(processor.textures.get(7).desc.format, 'rgba8unorm');
  assert.equal(handler.stats.createTexture2Count, 3);
  processor.executeCommandBuffer(destroyPacket(7));
  assert.equal(processor.textures.has(7), false);
});

test('invalid CreateTexture2 payloads are counted and leave the id unbound', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);
  processor.executeCommandBuffer(createTexture2Packet({ id: 1, width: 4, height: 4, format: 0 }));
  processor.executeCommandBuffer(createTexture2Packet({ id: 2, width: 4, height: 4, format: 200 }));
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 3, width: 4, height: 4, format: 22, dimension: 9 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 4, width: 4, height: 4, format: 62, samples: 4 })
  );
  assert.equal(handler.stats.errors, 4);
  assert.equal(handler.stats.createTexture2Count, 0);
  assert.match(handler.lastError, /not multisampleable/);
  for (const id of [1, 2, 3, 4]) assert.equal(processor.textures.has(id), false);

  // Truncated payload decoded directly returns a clean error instead of throwing RangeError.
  const truncated = tp.decodeCreateTexture2(new DataView(new ArrayBuffer(20)), 0);
  assert.match(truncated.error, /payload 20 < 40 bytes/);
});

test('the texture handler chains other opcodes to the previously installed extension', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const seen = [];
  processor.extension = {
    handlePacket: (opcode) => seen.push(['packet', opcode]),
    destroyResource: (id) => seen.push(['destroy', id]),
  };
  tp.installTexturePackets(processor);
  const unknown = new Uint8Array(8);
  new DataView(unknown.buffer).setUint16(0, 77, true);
  processor.executeCommandBuffer(unknown);
  processor.executeCommandBuffer(destroyPacket(5));
  assert.deepEqual(seen, [
    ['packet', 77],
    ['destroy', 5],
  ]);
});

test('CreateTexture2 and CreateTexture safely clamp mipLevels, sampleCount, and dimensions', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);

  // Excessive mip levels (99 -> 9 for 256x128) and MSAA with mips > 1 (4 -> 1).
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 10, width: 256, height: 128, mips: 99, samples: 4, format: 22 })
  );
  // 8x MSAA on single-mip 2D -> clamped to 4.
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 11, width: 64, height: 64, mips: 0, samples: 8, format: 22 })
  );
  // 2x MSAA (unsupported in WebGPU) -> clamped to 1.
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 12, width: 64, height: 64, mips: 1, samples: 2, format: 22 })
  );
  // 4x MSAA on 2D array (layers=6) -> clamped to 1.
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 13, width: 64, height: 64, layers: 6, mips: 1, samples: 4, format: 22 })
  );
  // 3D texture: depth participates in max mip calculation (16x16x64 -> 7 mips), MSAA -> 1.
  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 14,
      width: 16,
      height: 16,
      layers: 64,
      mips: 20,
      samples: 4,
      dimension: 2,
      format: 22,
    })
  );
  // 1D texture: height=1, depthOrArrayLayers=1, mipLevelCount=1, sampleCount=1.
  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 15,
      width: 128,
      height: 8,
      layers: 4,
      mips: 5,
      samples: 4,
      dimension: 0,
      format: 22,
    })
  );
  // v1 CreateTexture alongside CreateTexture2 also clamps excessive mipLevels (50 -> 6 for 32x16).
  processor.executeCommandBuffer(
    createTexture1Packet({ id: 16, width: 32, height: 16, mips: 50, format: 0, dimension: 1 })
  );

  const byId = (id) => processor.textures.get(id).desc;
  assert.equal(byId(10).mipLevelCount, 9);
  assert.equal(byId(10).sampleCount, 1);

  assert.equal(byId(11).mipLevelCount, 1);
  assert.equal(byId(11).sampleCount, 4);

  assert.equal(byId(12).sampleCount, 1);
  assert.equal(byId(13).sampleCount, 1);

  assert.equal(byId(14).dimension, '3d');
  assert.equal(byId(14).mipLevelCount, 7);
  assert.equal(byId(14).sampleCount, 1);

  assert.deepEqual(byId(15).size, { width: 128, height: 1, depthOrArrayLayers: 1 });
  assert.equal(byId(15).mipLevelCount, 1);
  assert.equal(byId(15).sampleCount, 1);

  assert.equal(byId(16).mipLevelCount, 6);
  assert.equal(byId(16).format, 'rgba8unorm');

  // createTexture2Count counts only v2 CreateTexture2 (6), while created counts both v1 + v2 (7).
  assert.equal(handler.stats.created, 7);
  assert.equal(handler.stats.createTexture2Count, 6);
  assert.equal(handler.stats.errors, 0);
});

test('alignBytesPerRow256 and repackRowsTo256 enforce WebGPU 256-byte row pitch alignment', () => {
  assert.equal(tp.alignBytesPerRow256(0), 0);
  assert.equal(tp.alignBytesPerRow256(1), 256);
  assert.equal(tp.alignBytesPerRow256(12), 256);
  assert.equal(tp.alignBytesPerRow256(256), 256);
  assert.equal(tp.alignBytesPerRow256(257), 512);
  assert.equal(tp.alignBytesPerRow256(768), 768);
  assert.throws(() => tp.alignBytesPerRow256(-4), RangeError);
  assert.throws(() => tp.alignBytesPerRow256(12.5), RangeError);

  // Unaligned 3 rows x 12 bytes -> 3 rows x 256 bytes, zero-padded tails.
  const srcUnaligned = new Uint8Array(36);
  for (let i = 0; i < 36; i += 1) srcUnaligned[i] = i + 1;
  const repacked = tp.repackRowsTo256(srcUnaligned, 12, 3);
  assert.equal(repacked.repacked, true);
  assert.equal(repacked.copiedFromSab, false);
  assert.equal(repacked.bytesPerRow, 256);
  assert.equal(repacked.originalBytesPerRow, 12);
  assert.equal(repacked.rowCount, 3);
  assert.equal(repacked.data.byteLength, 768);
  for (let row = 0; row < 3; row += 1) {
    const rowStart = row * 256;
    assert.deepEqual(
      Array.from(repacked.data.subarray(rowStart, rowStart + 12)),
      Array.from(srcUnaligned.subarray(row * 12, row * 12 + 12))
    );
    for (let pad = 12; pad < 256; pad += 1) {
      assert.equal(repacked.data[rowStart + pad], 0);
    }
  }

  // Already 256-byte aligned plain ArrayBuffer view -> zero-copy, repacked: false.
  const srcAligned = new Uint8Array(512);
  srcAligned[0] = 0xaa;
  srcAligned[256] = 0xbb;
  const alignedRes = tp.repackRowsTo256(srcAligned, 256, 2);
  assert.equal(alignedRes.repacked, false);
  assert.equal(alignedRes.copiedFromSab, false);
  assert.equal(alignedRes.bytesPerRow, 256);
  assert.equal(alignedRes.data, srcAligned);

  // Multi-layer (2D array / 3D) repacking preserves byteOffset on subarray input.
  const outer = new Uint8Array(16 + 16 * 2 * 3);
  outer.fill(0xff, 0, 16);
  for (let i = 0; i < 96; i += 1) outer[16 + i] = (i * 7) & 0xff;
  const sub = outer.subarray(16);
  const multi = tp.repackRowsTo256(sub, 16, 2, 3);
  assert.equal(multi.repacked, true);
  assert.equal(multi.rowCount, 6);
  assert.equal(multi.data.byteLength, 6 * 256);
  assert.equal(multi.data[0], 0);
  assert.equal(multi.data[5 * 256 + 15], (95 * 7) & 0xff);

  // SharedArrayBuffer-backed views are safely copied out even when already aligned.
  const sabAligned = new SharedArrayBuffer(512);
  const sabAlignedView = new Uint8Array(sabAligned);
  sabAlignedView[0] = 42;
  sabAlignedView[256] = 99;
  const sabRes = tp.repackRowsTo256(sabAlignedView, 256, 2);
  assert.equal(sabRes.repacked, false);
  assert.equal(sabRes.copiedFromSab, true);
  assert.equal(sabRes.data.buffer instanceof ArrayBuffer, true);
  assert.equal(sabRes.data[0], 42);
  assert.equal(sabRes.data[256], 99);

  // SharedArrayBuffer-backed unaligned view is both repacked and copied out.
  const sabUnaligned = new SharedArrayBuffer(32);
  const sabUnalignedView = new Uint8Array(sabUnaligned);
  for (let i = 0; i < 32; i += 1) sabUnalignedView[i] = i + 10;
  const sabRepacked = tp.repackRowsTo256(sabUnalignedView, 16, 2);
  assert.equal(sabRepacked.repacked, true);
  assert.equal(sabRepacked.copiedFromSab, true);
  assert.equal(sabRepacked.data.buffer instanceof ArrayBuffer, true);
  assert.equal(sabRepacked.bytesPerRow, 256);
  assert.equal(sabRepacked.data[256 + 15], 41);

  // Insufficient bytes throw RangeError.
  assert.throws(() => tp.repackRowsTo256(new Uint8Array(10), 16, 2), RangeError);
});

test('staged WriteTexture & copyBufferToTexture repack unaligned bytesPerRow to 256B and copy SAB views', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const stagingPool = new GpuStagingPool({ device: gpu.device, chunkSize: 64 * 1024 });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { stagingPool });
  const handler = tp.installTexturePackets(processor);

  // Create a 4x4 rgba8unorm texture via ABI v2 CreateTexture2.
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 50, width: 4, height: 4, usage: 0x06, format: 22 })
  );

  // Upload 4x4 RGBA8 (bytesPerRow = 16, unaligned!) from a SharedArrayBuffer command packet.
  const texels = new Uint8Array(64);
  for (let i = 0; i < 64; i += 1) texels[i] = i + 1;

  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 50,
      width: 4,
      height: 4,
      bytesPerRow: 16,
      rowsPerImage: 4,
      data: texels,
      useSab: true,
    })
  );

  // Verify copyBufferToTexture received 256-byte aligned bytesPerRow and zero SAB views reached queue.writeBuffer.
  const c2tCalls = gpu.calls.filter((c) => c.op === 'copyBufferToTexture');
  assert.equal(c2tCalls.length, 1);
  assert.equal(c2tCalls[0].source.bytesPerRow, 256);
  assert.equal(c2tCalls[0].source.rowsPerImage, 4);
  for (let row = 0; row < 4; row += 1) {
    assert.deepEqual(
      Array.from(c2tCalls[0].bytes.subarray(row * 256, row * 256 + 16)),
      Array.from(texels.subarray(row * 16, row * 16 + 16))
    );
  }
  const wbCalls = gpu.calls.filter((c) => c.op === 'writeBuffer');
  assert.ok(wbCalls.length >= 1);
  assert.ok(wbCalls.every((c) => c.isSab === false));

  assert.deepEqual(handler.getStats(), {
    created: 1,
    errors: 0,
    createTexture2Count: 1,
    rowPitchRepackCount: 1,
    sabCopyCount: 1,
    bc3dFallbacks: 0,
    bc3dFallbackUploadsSkipped: 0,
    uploadCount: 1,
    uploadBytes: 1024,
    zeroCopyUploads: 0,
    scratchRepackUploads: 1,
    directZeroCopyUploads: 0,
    repackedUploads: 1,
    scratchBufferReuses: 0,
    scratchBufferAllocations: 1,
    scratchBufferBytes: 1024,
    r8unormUploads: 0,
    bytesUploaded: 1024,
    srgbSiblingViewFormatsAdded: 1,
  });
});

test('direct queue.writeTexture safely copies SharedArrayBuffer views and tracks sabCopyCount', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);

  processor.executeCommandBuffer(
    createTexture2Packet({ id: 60, width: 2, height: 2, usage: 0x06, format: 22 })
  );

  const texels = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);

  // Upload 1: SharedArrayBuffer-backed WriteTexture packet -> must copy out of SAB.
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 60,
      width: 2,
      height: 2,
      bytesPerRow: 8,
      rowsPerImage: 2,
      data: texels,
      useSab: true,
    })
  );

  // Upload 2: plain ArrayBuffer-backed WriteTexture packet -> no SAB copy needed.
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 60,
      width: 2,
      height: 2,
      bytesPerRow: 8,
      rowsPerImage: 2,
      data: texels,
      useSab: false,
    })
  );

  const wtCalls = gpu.calls.filter((c) => c.op === 'writeTexture');
  assert.equal(wtCalls.length, 2);
  assert.equal(wtCalls[0].isSab, false);
  assert.deepEqual(Array.from(wtCalls[0].bytes), Array.from(texels));
  assert.equal(wtCalls[1].isSab, false);

  assert.deepEqual(handler.getStats(), {
    created: 1,
    errors: 0,
    createTexture2Count: 1,
    rowPitchRepackCount: 0,
    sabCopyCount: 1,
    bc3dFallbacks: 0,
    bc3dFallbackUploadsSkipped: 0,
    uploadCount: 2,
    uploadBytes: 32,
    zeroCopyUploads: 1,
    scratchRepackUploads: 0,
    directZeroCopyUploads: 1,
    repackedUploads: 0,
    scratchBufferReuses: 0,
    scratchBufferAllocations: 1,
    scratchBufferBytes: 16,
    r8unormUploads: 0,
    bytesUploaded: 32,
    srgbSiblingViewFormatsAdded: 1,
  });
});

test('handler.copyBufferToTexture repacks both CPU-backed and GPU-buffer-only unaligned copies', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);

  processor.executeCommandBuffer(
    createTexture2Packet({ id: 70, width: 4, height: 2, usage: 0x06, format: 22 })
  );
  const texture = processor.textures.get(70);
  const encoder = gpu.device.createCommandEncoder();

  // Case 1: CPU-backed SAB bytes with unaligned bytesPerRow = 16.
  const sab = new SharedArrayBuffer(32);
  const sabView = new Uint8Array(sab);
  for (let i = 0; i < 32; i += 1) sabView[i] = 100 + i;

  const resCpu = handler.copyBufferToTexture(
    encoder,
    { bytesPerRow: 16, rowsPerImage: 2 },
    { texture, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
    { width: 4, height: 2, depthOrArrayLayers: 1 },
    sabView
  );
  assert.equal(resCpu.repacked, true);
  assert.equal(resCpu.copiedFromSab, true);
  assert.equal(resCpu.bytesPerRow, 256);

  // Case 2: GPU-buffer-only copy with unaligned 4-byte-multiple bytesPerRow = 16.
  const gpuBuf = gpu.device.createBuffer({ size: 32, usage: 0x0004 | 0x0008 });
  for (let i = 0; i < 32; i += 1) gpuBuf.backing[i] = 200 + i;
  const resGpu = handler.copyBufferToTexture(
    encoder,
    { buffer: gpuBuf, offset: 0, bytesPerRow: 16, rowsPerImage: 2 },
    { texture, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
    { width: 4, height: 2, depthOrArrayLayers: 1 }
  );
  assert.equal(resGpu.repacked, true);
  assert.equal(resGpu.copiedFromSab, false);
  assert.equal(resGpu.bytesPerRow, 256);

  const c2tCalls = gpu.calls.filter((c) => c.op === 'copyBufferToTexture');
  assert.equal(c2tCalls.length, 2);
  assert.equal(c2tCalls[0].source.bytesPerRow, 256);
  assert.equal(c2tCalls[1].source.bytesPerRow, 256);
  assert.deepEqual(Array.from(c2tCalls[1].bytes.subarray(0, 16)), Array.from(gpuBuf.backing.subarray(0, 16)));
  assert.deepEqual(
    Array.from(c2tCalls[1].bytes.subarray(256, 272)),
    Array.from(gpuBuf.backing.subarray(16, 32))
  );

  // Case 3: Already-aligned GPU buffer passes straight through without repacking.
  const alignedBuf = gpu.device.createBuffer({ size: 512, usage: 0x0004 });
  const resAligned = handler.copyBufferToTexture(
    encoder,
    { buffer: alignedBuf, offset: 0, bytesPerRow: 256, rowsPerImage: 2 },
    { texture, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
    { width: 4, height: 2, depthOrArrayLayers: 1 }
  );
  assert.equal(resAligned.repacked, false);
  assert.equal(resAligned.bytesPerRow, 256);

  const stats = handler.getStats();
  assert.deepEqual(stats, {
    created: 1,
    errors: 0,
    createTexture2Count: 1,
    rowPitchRepackCount: 2,
    sabCopyCount: 1,
    bc3dFallbacks: 0,
    bc3dFallbackUploadsSkipped: 0,
    uploadCount: 3,
    uploadBytes: 1536,
    zeroCopyUploads: 0,
    scratchRepackUploads: 1,
    directZeroCopyUploads: 0,
    repackedUploads: 2,
    scratchBufferReuses: 0,
    scratchBufferAllocations: 1,
    scratchBufferBytes: 512,
    r8unormUploads: 0,
    bytesUploaded: 1536,
    srgbSiblingViewFormatsAdded: 1,
  });
  for (const [key, val] of Object.entries(stats)) {
    assert.equal(typeof val, 'number', `stat ${key} must be numeric`);
    assert.ok(Number.isInteger(val) && val >= 0, `stat ${key} must be a non-negative integer`);
  }
});

test('G31 classic-worker importScripts IIFE isolation preserves zero top-level lexical leaks', () => {
  const ctx = vm.createContext({
    console,
    Uint8Array,
    DataView,
    ArrayBuffer,
    SharedArrayBuffer,
    Map,
    Set,
  });
  ctx.globalThis = ctx;
  ctx.self = ctx;

  const load = (file) => {
    const src = readFileSync(path.join(WORKERS_DIR, file), 'utf8');
    vm.runInContext(src, ctx, { filename: file });
  };

  // Load texture_formats.js, render_state_packets.js, and texture_packets.js twice without SyntaxError.
  assert.doesNotThrow(() => {
    load('texture_formats.js');
    load('render_state_packets.js');
    load('texture_packets.js');
    load('texture_packets.js');
  });

  // Verify no internal identifiers leaked into global lexical scope.
  assert.doesNotThrow(() => {
    vm.runInContext(
      `
      const CREATE_TEXTURE2 = 0;
      const WEBGPU_BYTES_PER_ROW_ALIGNMENT = 0;
      const TEXTURE_FORMAT_NAMES = [];
      const alignBytesPerRow256 = null;
      const repackRowsTo256 = null;
      const isBcCompressedFormat = null;
      const deviceSupportsBcSliced3d = null;
      const SRGB_SIBLING_FORMATS = null;
      const DEFAULT_MAX_SCRATCH_BYTES = 0;
      const getSrgbSiblingFormat = null;
      const resolveCompatibleViewFormats = null;
      class TexturePackets {}
      `,
      ctx
    );
  });

  assert.equal(typeof ctx.PlayGta6TexturePackets.alignBytesPerRow256, 'function');
  assert.equal(typeof ctx.PlayGta6TexturePackets.repackRowsTo256, 'function');
  assert.equal(ctx.PlayGta6TexturePackets.alignBytesPerRow256(16), 256);
});

test('W108 (#734): 3D BC textures fall back to rgba8unorm without bc-sliced-3d and skip compressed uploads', () => {
  const gpu = createMockWebGpu({
    strictWebGpu: true,
    features: new Set(['texture-compression-bc']),
  });
  const stagingPool = new GpuStagingPool({ device: gpu.device, chunkSize: 64 * 1024 });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { stagingPool });
  const rsHandler = installRenderStatePackets(processor);
  const handler = tp.installTexturePackets(processor);

  // CreateTexture2: 3D bc5-rg-unorm (code 58) with incompatible bc5 viewFormats.
  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 201,
      width: 16,
      height: 16,
      layers: 4,
      mips: 1,
      dimension: 2, // 3d
      format: 58, // bc5-rg-unorm
      usage: 0x06,
      viewFormats: [58, 59], // bc5-rg-unorm, bc5-rg-snorm
    })
  );
  // CreateTexture2: 3D bc7-rgba-unorm (code 62) with bc7-rgba-unorm-srgb (code 63) viewFormat.
  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 202,
      width: 8,
      height: 8,
      layers: 4,
      mips: 1,
      dimension: 2, // 3d
      format: 62, // bc7-rgba-unorm
      usage: 0x06,
      viewFormats: [63], // bc7-rgba-unorm-srgb
    })
  );
  // v1 CreateTexture: 3D bc7-rgba-unorm (v1 format code 5).
  processor.executeCommandBuffer(
    createTexture1Packet({
      id: 203,
      width: 8,
      height: 8,
      layers: 2,
      mips: 1,
      dimension: 2, // 3d
      format: 5, // bc7-rgba-unorm
      usage: 0x06,
    })
  );

  const tex201 = processor.textures.get(201);
  const tex202 = processor.textures.get(202);
  const tex203 = processor.textures.get(203);
  assert.ok(tex201 && tex202 && tex203);
  assert.equal(tex201.desc.format, 'rgba8unorm');
  assert.equal(tex201.desc.dimension, '3d');
  assert.equal(tex201.desc.viewFormats, undefined);
  assert.equal(tex202.desc.format, 'rgba8unorm');
  assert.equal(tex202.desc.viewFormats, undefined);
  assert.equal(tex203.desc.format, 'rgba8unorm');
  assert.equal(tex203.desc.dimension, '3d');
  assert.equal(handler._bc3dFallbackTextures.has(201), true);
  assert.equal(handler._bc3dFallbackTextures.has(202), true);
  assert.equal(handler._bc3dFallbackTextures.has(203), true);
  assert.equal(handler.stats.bc3dFallbacks, 3);
  assert.equal(handler.stats.errors, 0);

  // Chained RenderStatePackets view creation must not pass bc* format against the rgba8unorm fallback.
  const view201 = rsHandler._textureView(201, {
    format: 'bc5-rg-unorm',
    dimension: '3d',
    baseMipLevel: 0,
    mipLevelCount: 1,
    baseArrayLayer: 0,
    arrayLayerCount: 1,
    aspect: 'all',
  });
  assert.ok(view201);
  assert.equal(view201.viewDesc.format, 'rgba8unorm');

  // Compressed WriteTexture uploads to fallback 3D textures must be skipped cleanly (both staged and direct).
  const bc5Bytes = new Uint8Array(4 * 4 * 16 * 4); // 16x16x4 BC5 blocks
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 201,
      width: 16,
      height: 16,
      layers: 4,
      bytesPerRow: 64,
      rowsPerImage: 4,
      data: bc5Bytes,
    })
  );
  // Direct writeTexture() and copyBufferToTexture() wrappers also skip fallback texture objects.
  const directWt = handler.writeTexture(
    { texture: tex202, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
    new Uint8Array(64),
    { offset: 0, bytesPerRow: 32, rowsPerImage: 2 },
    { width: 8, height: 8, depthOrArrayLayers: 4 }
  );
  assert.equal(directWt.uploaded, false);
  assert.equal(directWt.skippedBc3dFallback, true);

  const encoder = gpu.device.createCommandEncoder();
  const directC2t = handler.copyBufferToTexture(
    encoder,
    { bytesPerRow: 32, rowsPerImage: 2 },
    { texture: tex202, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
    { width: 8, height: 8, depthOrArrayLayers: 4 },
    new Uint8Array(256)
  );
  assert.equal(directC2t.skippedBc3dFallback, true);

  assert.equal(gpu.calls.filter((c) => c.op === 'writeTexture').length, 0);
  assert.equal(gpu.calls.filter((c) => c.op === 'copyBufferToTexture').length, 0);
  assert.deepEqual(handler.getStats(), {
    created: 3,
    errors: 0,
    createTexture2Count: 2,
    rowPitchRepackCount: 0,
    sabCopyCount: 0,
    bc3dFallbacks: 3,
    bc3dFallbackUploadsSkipped: 3,
    uploadCount: 0,
    uploadBytes: 0,
    zeroCopyUploads: 0,
    scratchRepackUploads: 0,
    directZeroCopyUploads: 0,
    repackedUploads: 0,
    scratchBufferReuses: 0,
    scratchBufferAllocations: 0,
    scratchBufferBytes: 0,
    r8unormUploads: 0,
    bytesUploaded: 0,
    srgbSiblingViewFormatsAdded: 0,
  });
});

test('W108 (#734): 3D BC textures preserve original format and upload normally when bc-sliced-3d is present', () => {
  const gpu = createMockWebGpu({
    strictWebGpu: true,
    features: new Set(['texture-compression-bc', 'texture-compression-bc-sliced-3d']),
  });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);

  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 210,
      width: 16,
      height: 16,
      layers: 4,
      mips: 1,
      dimension: 2, // 3d
      format: 58, // bc5-rg-unorm
      usage: 0x06,
    })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 211,
      width: 8,
      height: 8,
      layers: 2,
      mips: 1,
      dimension: 2, // 3d
      format: 62, // bc7-rgba-unorm
      usage: 0x06,
      viewFormats: [63], // bc7-rgba-unorm-srgb
    })
  );

  assert.equal(processor.textures.get(210).desc.format, 'bc5-rg-unorm');
  assert.equal(processor.textures.get(211).desc.format, 'bc7-rgba-unorm');
  assert.deepEqual(processor.textures.get(211).desc.viewFormats, ['bc7-rgba-unorm-srgb']);
  assert.equal(handler._bc3dFallbackTextures.size, 0);
  assert.equal(handler.stats.bc3dFallbacks, 0);

  const bc5Bytes = new Uint8Array(4 * 4 * 16 * 4);
  bc5Bytes[0] = 0x5a;
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 210,
      width: 16,
      height: 16,
      layers: 4,
      bytesPerRow: 64,
      rowsPerImage: 4,
      data: bc5Bytes,
    })
  );

  const wtCalls = gpu.calls.filter((c) => c.op === 'writeTexture');
  assert.equal(wtCalls.length, 1);
  assert.equal(wtCalls[0].bytes[0], 0x5a);
  assert.equal(handler.stats.bc3dFallbackUploadsSkipped, 0);
  assert.equal(handler.stats.errors, 0);
});

test('W108 (#734): 2D BC textures remain unaffected when bc-sliced-3d is absent', () => {
  const gpu = createMockWebGpu({
    strictWebGpu: true,
    features: new Set(['texture-compression-bc']),
  });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);

  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 220,
      width: 16,
      height: 16,
      layers: 1,
      mips: 1,
      dimension: 1, // 2d
      format: 58, // bc5-rg-unorm
      usage: 0x06,
    })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 221,
      width: 16,
      height: 16,
      layers: 6,
      mips: 1,
      dimension: 1, // 2d array
      format: 62, // bc7-rgba-unorm
      usage: 0x06,
      viewFormats: [63],
    })
  );

  assert.equal(processor.textures.get(220).desc.format, 'bc5-rg-unorm');
  assert.equal(processor.textures.get(221).desc.format, 'bc7-rgba-unorm');
  assert.deepEqual(processor.textures.get(221).desc.viewFormats, ['bc7-rgba-unorm-srgb']);
  assert.equal(handler._bc3dFallbackTextures.size, 0);
  assert.equal(handler.stats.bc3dFallbacks, 0);

  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 220,
      width: 16,
      height: 16,
      layers: 1,
      bytesPerRow: 64,
      rowsPerImage: 4,
      data: new Uint8Array(256),
    })
  );
  assert.equal(gpu.calls.filter((c) => c.op === 'writeTexture').length, 1);
  assert.equal(handler.stats.bc3dFallbackUploadsSkipped, 0);
});

test('W108 (#734): destroying or recreating a fallback texture ID clears fallback tracking', () => {
  const gpu = createMockWebGpu({
    strictWebGpu: true,
    features: new Set(['texture-compression-bc']),
  });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);

  // Create two 3D BC fallback textures.
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 230, width: 8, height: 8, layers: 4, dimension: 2, format: 58, usage: 0x06 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 231, width: 8, height: 8, layers: 4, dimension: 2, format: 62, usage: 0x06 })
  );
  assert.equal(handler._bc3dFallbackTextures.has(230), true);
  assert.equal(handler._bc3dFallbackTextures.has(231), true);

  // 1) DestroyResource removes ID 230 from the fallback tracking set.
  processor.executeCommandBuffer(destroyPacket(230));
  assert.equal(handler._bc3dFallbackTextures.has(230), false);
  assert.equal(handler._bc3dFallbackTextures.has(231), true);

  // 2) Recreating ID 231 with a non-3D/non-BC format clears it and allows normal uploads.
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 231, width: 4, height: 4, layers: 1, dimension: 1, format: 22, usage: 0x06 })
  );
  assert.equal(handler._bc3dFallbackTextures.has(231), false);
  assert.equal(processor.textures.get(231).desc.format, 'rgba8unorm');

  const rgbaBytes = new Uint8Array(4 * 4 * 4);
  rgbaBytes[0] = 0x77;
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 231,
      width: 4,
      height: 4,
      layers: 1,
      bytesPerRow: 16,
      rowsPerImage: 4,
      data: rgbaBytes,
    })
  );
  const wtCalls = gpu.calls.filter((c) => c.op === 'writeTexture');
  assert.equal(wtCalls.length, 1);
  assert.equal(wtCalls[0].bytes[0], 0x77);
  assert.equal(handler.stats.bc3dFallbacks, 2);
  assert.equal(handler.stats.bc3dFallbackUploadsSkipped, 0);
});

test('W164 (#871): 4K Bink2 3840x2160 r8unorm Y plane takes zero-copy direct queue.writeTexture path even with stagingPool', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const stagingPool = new GpuStagingPool({ device: gpu.device, chunkSize: 4 * 1024 * 1024 });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { stagingPool });
  const handler = tp.installTexturePackets(processor);

  const W = 3840;
  const H = 2160;
  assert.equal(W % 256, 0);

  // Create 4K r8unorm (format code 1) Y plane texture.
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 300, width: W, height: H, usage: 0x06, format: 1 })
  );
  assert.equal(processor.textures.get(300).desc.format, 'r8unorm');
  assert.equal(processor.textures.get(300).desc.viewFormats, undefined);

  const yBytes = new Uint8Array(W * H);
  yBytes[0] = 0x11;
  yBytes[W - 1] = 0x22;
  yBytes[(H - 1) * W] = 0x33;
  yBytes[W * H - 1] = 0x44;

  const pkt = writeTexturePacket({
    textureId: 300,
    width: W,
    height: H,
    bytesPerRow: W,
    rowsPerImage: H,
    data: yBytes,
    useSab: false,
  });

  processor.executeCommandBuffer(pkt);

  // Must go directly to queue.writeTexture with the exact command packet ArrayBuffer (zero-copy).
  const wtCalls = gpu.calls.filter((c) => c.op === 'writeTexture');
  const c2tCalls = gpu.calls.filter((c) => c.op === 'copyBufferToTexture');
  assert.equal(wtCalls.length, 1);
  assert.equal(c2tCalls.length, 0);
  assert.equal(wtCalls[0].bufferRef, pkt.buffer, '3840x2160 aligned r8unorm upload must be zero-copy');
  assert.equal(wtCalls[0].byteOffset, 8 + 48);
  assert.equal(wtCalls[0].byteLength, W * H);
  assert.equal(wtCalls[0].dataLayout.bytesPerRow, W);
  assert.equal(wtCalls[0].dataLayout.rowsPerImage, H);
  assert.equal(wtCalls[0].bytes[0], 0x11);
  assert.equal(wtCalls[0].bytes[W - 1], 0x22);
  assert.equal(wtCalls[0].bytes[(H - 1) * W], 0x33);
  assert.equal(wtCalls[0].bytes[W * H - 1], 0x44);

  // No staging buffer chunks allocated and no scratch buffer touched.
  assert.equal(stagingPool.getStats().totalChunksAllocated, 0);
  assert.deepEqual(handler.getStats(), {
    created: 1,
    errors: 0,
    createTexture2Count: 1,
    rowPitchRepackCount: 0,
    sabCopyCount: 0,
    bc3dFallbacks: 0,
    bc3dFallbackUploadsSkipped: 0,
    uploadCount: 1,
    uploadBytes: W * H,
    zeroCopyUploads: 1,
    scratchRepackUploads: 0,
    directZeroCopyUploads: 1,
    repackedUploads: 0,
    scratchBufferReuses: 0,
    scratchBufferAllocations: 0,
    scratchBufferBytes: 0,
    r8unormUploads: 1,
    bytesUploaded: W * H,
    srgbSiblingViewFormatsAdded: 0,
  });
});

test('W164 (#871): 4K Bink2 1920x1080 r8unorm Cr/Cb planes reuse persistent scratch Uint8Array across frames and zero padding tails', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const stagingPool = new GpuStagingPool({
    device: gpu.device,
    chunkSize: 4 * 1024 * 1024,
    maxPoolBytes: 64 * 1024 * 1024,
  });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { stagingPool });
  const handler = tp.installTexturePackets(processor);

  const YW = 3840;
  const YH = 2160;
  const CW = 1920;
  const CH = 1080;
  const ALIGNED_CW = 2048;
  assert.equal(tp.alignBytesPerRow256(CW), ALIGNED_CW);

  // Create Y, Cr, Cb r8unorm planes.
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 310, width: YW, height: YH, usage: 0x06, format: 1 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 311, width: CW, height: CH, usage: 0x06, format: 1 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 312, width: CW, height: CH, usage: 0x06, format: 1 })
  );

  const yData = new Uint8Array(YW * YH);
  const crData = new Uint8Array(CW * CH);
  const cbData = new Uint8Array(CW * CH);

  // Simulate 3 frames at 30 FPS cadence: 3x Y (zero-copy) + 3x Cr + 3x Cb (scratch-reused repack).
  const FRAMES = 3;
  let firstScratchRef = null;
  for (let frame = 0; frame < FRAMES; frame += 1) {
    stagingPool.beginFrame(frame);
    yData[0] = 0x10 + frame;
    crData.fill(0xa0 + frame);
    cbData.fill(0x50 + frame);
    // Mark first and last active bytes of row 0 and last row.
    crData[0] = 0xc0 + frame;
    crData[CW - 1] = 0xd0 + frame;
    cbData[0] = 0x60 + frame;
    cbData[CW - 1] = 0x70 + frame;
    cbData[(CH - 1) * CW + (CW - 1)] = 0x80 + frame;

    processor.executeCommandBuffer(
      writeTexturePacket({
        textureId: 310,
        width: YW,
        height: YH,
        bytesPerRow: YW,
        rowsPerImage: YH,
        data: yData,
      })
    );
    processor.executeCommandBuffer(
      writeTexturePacket({
        textureId: 311,
        width: CW,
        height: CH,
        bytesPerRow: CW,
        rowsPerImage: CH,
        data: crData,
      })
    );
    if (frame === 0) {
      firstScratchRef = handler._scratchBuffer;
      assert.ok(firstScratchRef instanceof Uint8Array);
      assert.equal(firstScratchRef.byteLength, ALIGNED_CW * CH);
      // Poison padding tail in scratch buffer to verify next upload strictly zeroes it.
      firstScratchRef.fill(0xff, CW, ALIGNED_CW);
    }
    processor.executeCommandBuffer(
      writeTexturePacket({
        textureId: 312,
        width: CW,
        height: CH,
        bytesPerRow: CW,
        rowsPerImage: CH,
        data: cbData,
      })
    );
    assert.equal(
      handler._scratchBuffer,
      firstScratchRef,
      `Frame ${frame}: scratch Uint8Array must be reused without reallocating`
    );
    stagingPool.endFrame();
  }

  const wtCalls = gpu.calls.filter((c) => c.op === 'writeTexture');
  const c2tCalls = gpu.calls.filter((c) => c.op === 'copyBufferToTexture');
  assert.equal(wtCalls.length, FRAMES, 'Y plane uses direct zero-copy writeTexture each frame');
  assert.equal(c2tCalls.length, FRAMES * 2, 'Cr + Cb planes stage repacked 2048B rows');

  // Verify Cb frame 0 repacked rows have exact active pixels and strictly zeroed padding [1920..2048).
  const cbFrame0 = c2tCalls[1];
  assert.equal(cbFrame0.source.bytesPerRow, ALIGNED_CW);
  assert.equal(cbFrame0.source.rowsPerImage, CH);
  assert.equal(cbFrame0.bytes[0], 0x60);
  assert.equal(cbFrame0.bytes[CW - 1], 0x70);
  for (let p = CW; p < ALIGNED_CW; p += 1) {
    assert.equal(cbFrame0.bytes[p], 0, `padding byte at offset ${p} must be zeroed`);
  }
  assert.equal(cbFrame0.bytes[(CH - 1) * ALIGNED_CW + (CW - 1)], 0x80);

  const expectedBytes = FRAMES * (YW * YH) + FRAMES * 2 * (ALIGNED_CW * CH);
  assert.deepEqual(handler.getStats(), {
    created: 3,
    errors: 0,
    createTexture2Count: 3,
    rowPitchRepackCount: FRAMES * 2,
    sabCopyCount: 0,
    bc3dFallbacks: 0,
    bc3dFallbackUploadsSkipped: 0,
    uploadCount: FRAMES * 3,
    uploadBytes: expectedBytes,
    zeroCopyUploads: FRAMES,
    scratchRepackUploads: FRAMES * 2,
    directZeroCopyUploads: FRAMES,
    repackedUploads: FRAMES * 2,
    scratchBufferReuses: FRAMES * 2 - 1,
    scratchBufferAllocations: 1,
    scratchBufferBytes: ALIGNED_CW * CH,
    r8unormUploads: FRAMES * 3,
    bytesUploaded: expectedBytes,
    srgbSiblingViewFormatsAdded: 0,
  });
});

test('W164 (#871): scratch buffer grows on demand, respects maxScratchBytes cap, and handles SAB-backed aligned uploads', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const stagingPool = new GpuStagingPool({ device: gpu.device, chunkSize: 64 * 1024 });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { stagingPool });
  // Small cap (4096 bytes) to test growth + cap enforcement deterministically.
  const handler = tp.installTexturePackets(processor, { maxScratchBytes: 4096 });

  processor.executeCommandBuffer(
    createTexture2Packet({ id: 320, width: 16, height: 4, usage: 0x06, format: 1 }) // r8unorm
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 321, width: 16, height: 8, usage: 0x06, format: 1 }) // r8unorm
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 322, width: 16, height: 32, usage: 0x06, format: 1 }) // r8unorm -> 256*32=8192 > 4096 cap
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 323, width: 256, height: 8, usage: 0x06, format: 1 }) // aligned 256x8 = 2048B SAB
  );

  // 1) Initial 16x4 -> repacked 256*4 = 1024 bytes (alloc #1).
  processor.executeCommandBuffer(
    writeTexturePacket({ textureId: 320, width: 16, height: 4, bytesPerRow: 16, rowsPerImage: 4, data: new Uint8Array(64) })
  );
  assert.equal(handler.getStats().scratchBufferAllocations, 1);
  assert.equal(handler.getStats().scratchBufferReuses, 0);
  assert.equal(handler.getStats().scratchBufferBytes, 1024);

  // 2) Same size -> reuses 1024B scratch (reuse #1).
  processor.executeCommandBuffer(
    writeTexturePacket({ textureId: 320, width: 16, height: 4, bytesPerRow: 16, rowsPerImage: 4, data: new Uint8Array(64) })
  );
  assert.equal(handler.getStats().scratchBufferAllocations, 1);
  assert.equal(handler.getStats().scratchBufferReuses, 1);

  // 3) Larger 16x8 -> repacked 256*8 = 2048 bytes <= 4096 cap -> grows (alloc #2).
  processor.executeCommandBuffer(
    writeTexturePacket({ textureId: 321, width: 16, height: 8, bytesPerRow: 16, rowsPerImage: 8, data: new Uint8Array(128) })
  );
  assert.equal(handler.getStats().scratchBufferAllocations, 2);
  assert.equal(handler.getStats().scratchBufferBytes, 2048);

  // 4) Oversized 16x32 -> repacked 256*32 = 8192 > 4096 cap -> one-off alloc, persistent scratch stays 2048B.
  processor.executeCommandBuffer(
    writeTexturePacket({ textureId: 322, width: 16, height: 32, bytesPerRow: 16, rowsPerImage: 32, data: new Uint8Array(512) })
  );
  assert.equal(handler.getStats().scratchBufferAllocations, 2);
  assert.equal(handler.getStats().scratchBufferBytes, 2048);

  // 5) SAB-backed aligned 256x8 (2048B) -> copies out of SAB using existing 2048B scratch (reuse #2), direct writeTexture!
  const sabData = new Uint8Array(256 * 8);
  sabData[0] = 0xab;
  sabData[2047] = 0xcd;
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 323,
      width: 256,
      height: 8,
      bytesPerRow: 256,
      rowsPerImage: 8,
      data: sabData,
      useSab: true,
    })
  );
  const wtCalls = gpu.calls.filter((c) => c.op === 'writeTexture');
  assert.equal(wtCalls.length, 1);
  assert.equal(wtCalls[0].isSab, false);
  assert.equal(wtCalls[0].bytes[0], 0xab);
  assert.equal(wtCalls[0].bytes[2047], 0xcd);
  assert.equal(handler.getStats().sabCopyCount, 1);
  assert.equal(handler.getStats().scratchBufferReuses, 2);
  assert.equal(handler.getStats().directZeroCopyUploads, 0, 'SAB copy must not count as zero-copy');
  assert.equal(handler.getStats().r8unormUploads, 5);
});

test('W164 (#871): CreateTexture and CreateTexture2 attach compatible sRGB/linear sibling viewFormats safely', () => {
  // Unit check helper exports directly.
  assert.equal(tp.getSrgbSiblingFormat('rgba8unorm'), 'rgba8unorm-srgb');
  assert.equal(tp.getSrgbSiblingFormat('rgba8unorm-srgb'), 'rgba8unorm');
  assert.equal(tp.getSrgbSiblingFormat('bgra8unorm'), 'bgra8unorm-srgb');
  assert.equal(tp.getSrgbSiblingFormat('bgra8unorm-srgb'), 'bgra8unorm');
  assert.equal(tp.getSrgbSiblingFormat('bc1-rgba-unorm'), 'bc1-rgba-unorm-srgb');
  assert.equal(tp.getSrgbSiblingFormat('bc1-rgba-unorm-srgb'), 'bc1-rgba-unorm');
  assert.equal(tp.getSrgbSiblingFormat('bc2-rgba-unorm'), 'bc2-rgba-unorm-srgb');
  assert.equal(tp.getSrgbSiblingFormat('bc2-rgba-unorm-srgb'), 'bc2-rgba-unorm');
  assert.equal(tp.getSrgbSiblingFormat('bc3-rgba-unorm'), 'bc3-rgba-unorm-srgb');
  assert.equal(tp.getSrgbSiblingFormat('bc3-rgba-unorm-srgb'), 'bc3-rgba-unorm');
  assert.equal(tp.getSrgbSiblingFormat('bc7-rgba-unorm'), 'bc7-rgba-unorm-srgb');
  assert.equal(tp.getSrgbSiblingFormat('bc7-rgba-unorm-srgb'), 'bc7-rgba-unorm');
  assert.equal(tp.getSrgbSiblingFormat('r8unorm'), null);
  assert.equal(tp.getSrgbSiblingFormat('bc4-r-unorm'), null);
  assert.equal(tp.getSrgbSiblingFormat('bc5-rg-unorm'), null);
  assert.equal(tp.getSrgbSiblingFormat('bc6h-rgb-ufloat'), null);
  assert.equal(tp.getSrgbSiblingFormat('rgba16float'), null);

  // resolveCompatibleViewFormats strips incompatible formats and deduplicates.
  assert.deepEqual(tp.resolveCompatibleViewFormats('rgba8unorm', ['rgba16float', 'rgba8unorm']), {
    viewFormats: ['rgba8unorm', 'rgba8unorm-srgb'],
    siblingAdded: true,
  });
  assert.deepEqual(tp.resolveCompatibleViewFormats('bgra8unorm', ['bgra8unorm-srgb', 'bgra8unorm-srgb']), {
    viewFormats: ['bgra8unorm-srgb'],
    siblingAdded: false,
  });

  const gpu = createMockWebGpu({
    strictWebGpu: true,
    features: new Set(['texture-compression-bc']),
  });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);

  // v1 CreateTexture: rgba8unorm (0), bgra8unorm (1), rgba16float (2), bc7-rgba-unorm (5).
  processor.executeCommandBuffer(createTexture1Packet({ id: 400, width: 64, height: 64, format: 0 }));
  processor.executeCommandBuffer(createTexture1Packet({ id: 401, width: 64, height: 64, format: 1 }));
  processor.executeCommandBuffer(createTexture1Packet({ id: 402, width: 64, height: 64, format: 2 }));
  processor.executeCommandBuffer(createTexture1Packet({ id: 403, width: 64, height: 64, format: 5 }));

  assert.deepEqual(processor.textures.get(400).desc.viewFormats, ['rgba8unorm-srgb']);
  assert.deepEqual(processor.textures.get(401).desc.viewFormats, ['bgra8unorm-srgb']);
  assert.equal(processor.textures.get(402).desc.viewFormats, undefined);
  assert.deepEqual(processor.textures.get(403).desc.viewFormats, ['bc7-rgba-unorm-srgb']);

  // Creating sRGB views on v1 linear textures must pass strict WebGPU validation.
  assert.doesNotThrow(() => processor.textures.get(400).createView({ format: 'rgba8unorm-srgb' }));
  assert.doesNotThrow(() => processor.textures.get(401).createView({ format: 'bgra8unorm-srgb' }));
  assert.doesNotThrow(() => processor.textures.get(403).createView({ format: 'bc7-rgba-unorm-srgb' }));

  // v2 CreateTexture2: all sRGB-capable pairs without explicit viewFormats.
  const v2Pairs = [
    [410, 22, 'rgba8unorm', 'rgba8unorm-srgb'],
    [411, 23, 'rgba8unorm-srgb', 'rgba8unorm'],
    [412, 27, 'bgra8unorm', 'bgra8unorm-srgb'],
    [413, 28, 'bgra8unorm-srgb', 'bgra8unorm'],
    [414, 50, 'bc1-rgba-unorm', 'bc1-rgba-unorm-srgb'],
    [415, 51, 'bc1-rgba-unorm-srgb', 'bc1-rgba-unorm'],
    [416, 52, 'bc2-rgba-unorm', 'bc2-rgba-unorm-srgb'],
    [417, 53, 'bc2-rgba-unorm-srgb', 'bc2-rgba-unorm'],
    [418, 54, 'bc3-rgba-unorm', 'bc3-rgba-unorm-srgb'],
    [419, 55, 'bc3-rgba-unorm-srgb', 'bc3-rgba-unorm'],
    [420, 62, 'bc7-rgba-unorm', 'bc7-rgba-unorm-srgb'],
    [421, 63, 'bc7-rgba-unorm-srgb', 'bc7-rgba-unorm'],
  ];
  for (const [id, code, expectedFmt, expectedSibling] of v2Pairs) {
    processor.executeCommandBuffer(
      createTexture2Packet({ id, width: 16, height: 16, format: code, usage: 0x06 })
    );
    const tex = processor.textures.get(id);
    assert.equal(tex.desc.format, expectedFmt);
    assert.deepEqual(tex.desc.viewFormats, [expectedSibling]);
    assert.doesNotThrow(() => tex.createView({ format: expectedSibling }));
  }

  // Incompatible viewFormat in packet payload is stripped safely while valid sibling is attached.
  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 430,
      width: 16,
      height: 16,
      format: 22, // rgba8unorm
      viewFormats: [40], // incompatible rgba16float
      usage: 0x06,
    })
  );
  assert.deepEqual(processor.textures.get(430).desc.viewFormats, ['rgba8unorm-srgb']);

  // Formats without sRGB siblings (r8unorm, bc4, bc5, bc6h) keep viewFormats omitted.
  for (const [id, code] of [
    [440, 1], // r8unorm
    [441, 56], // bc4-r-unorm
    [442, 58], // bc5-rg-unorm
    [443, 60], // bc6h-rgb-ufloat
  ]) {
    processor.executeCommandBuffer(
      createTexture2Packet({ id, width: 16, height: 16, format: code, usage: 0x06 })
    );
    assert.equal(processor.textures.get(id).desc.viewFormats, undefined);
  }

  // 3 v1 textures + 12 v2 pairs + 1 stripped-incompatible v2 = 16 sibling viewFormats added.
  assert.equal(handler.getStats().srgbSiblingViewFormatsAdded, 16);
  assert.equal(handler.getStats().errors, 0);
});

test('W164 (#871): telemetry resets cleanly on reset(), destroy(), and processor.destroy()', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const stagingPool = new GpuStagingPool({ device: gpu.device, chunkSize: 64 * 1024 });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { stagingPool });
  const handler = tp.installTexturePackets(processor);

  processor.executeCommandBuffer(
    createTexture2Packet({ id: 500, width: 256, height: 4, usage: 0x06, format: 1 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 501, width: 16, height: 4, usage: 0x06, format: 1 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 502, width: 8, height: 8, usage: 0x06, format: 22 })
  );
  processor.executeCommandBuffer(
    writeTexturePacket({ textureId: 500, width: 256, height: 4, bytesPerRow: 256, rowsPerImage: 4, data: new Uint8Array(1024) })
  );
  processor.executeCommandBuffer(
    writeTexturePacket({ textureId: 501, width: 16, height: 4, bytesPerRow: 16, rowsPerImage: 4, data: new Uint8Array(64) })
  );
  processor.executeCommandBuffer(
    writeTexturePacket({ textureId: 501, width: 16, height: 4, bytesPerRow: 16, rowsPerImage: 4, data: new Uint8Array(64) })
  );

  const beforeReset = handler.getStats();
  assert.equal(beforeReset.created, 3);
  assert.equal(beforeReset.directZeroCopyUploads, 1);
  assert.equal(beforeReset.repackedUploads, 2);
  assert.equal(beforeReset.scratchBufferAllocations, 1);
  assert.equal(beforeReset.scratchBufferReuses, 1);
  assert.equal(beforeReset.scratchBufferBytes, 1024);
  assert.equal(beforeReset.r8unormUploads, 3);
  assert.equal(beforeReset.bytesUploaded, 1024 + 1024 + 1024);
  assert.equal(beforeReset.srgbSiblingViewFormatsAdded, 1);
  for (const [key, val] of Object.entries(beforeReset)) {
    assert.equal(typeof val, 'number', `stat ${key} must be numeric`);
    assert.ok(Number.isInteger(val) && val >= 0, `stat ${key} must be a non-negative integer`);
  }

  const expectedZeroStats = {
    created: 0,
    errors: 0,
    createTexture2Count: 0,
    rowPitchRepackCount: 0,
    sabCopyCount: 0,
    bc3dFallbacks: 0,
    bc3dFallbackUploadsSkipped: 0,
    uploadCount: 0,
    uploadBytes: 0,
    zeroCopyUploads: 0,
    scratchRepackUploads: 0,
    directZeroCopyUploads: 0,
    repackedUploads: 0,
    scratchBufferReuses: 0,
    scratchBufferAllocations: 0,
    scratchBufferBytes: 0,
    r8unormUploads: 0,
    bytesUploaded: 0,
    srgbSiblingViewFormatsAdded: 0,
  };

  // 1) reset() clears counters and releases scratch buffer, while handler stays usable.
  handler.reset();
  assert.deepEqual(handler.getStats(), expectedZeroStats);
  assert.equal(handler._scratchBuffer, null);
  assert.equal(handler.lastError, null);

  processor.executeCommandBuffer(
    createTexture2Packet({ id: 510, width: 256, height: 2, usage: 0x06, format: 1 })
  );
  processor.executeCommandBuffer(
    writeTexturePacket({ textureId: 510, width: 256, height: 2, bytesPerRow: 256, rowsPerImage: 2, data: new Uint8Array(512) })
  );
  assert.equal(handler.getStats().directZeroCopyUploads, 1);
  assert.equal(handler.getStats().r8unormUploads, 1);
  assert.equal(handler.getStats().bytesUploaded, 512);

  // 2) processor.destroy() triggers handler.destroy() cleanly and idempotently.
  processor.destroy();
  assert.deepEqual(handler.getStats(), expectedZeroStats);
  assert.equal(handler._scratchBuffer, null);
  handler.destroy();
  assert.deepEqual(handler.getStats(), expectedZeroStats);
});

test('W191: tightly packed and already-aligned WriteTexture payloads pass exact zero-copy ArrayBuffer subarray views', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);

  const YW = 3840;
  const YH = 2160;
  const CW = 1920;
  const CH = 1080;

  processor.executeCommandBuffer(
    createTexture2Packet({ id: 600, width: YW, height: YH, usage: 0x06, format: 1 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 601, width: CW, height: CH, usage: 0x06, format: 1 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 602, width: CW, height: CH, usage: 0x06, format: 1 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 603, width: 8, height: 4, layers: 2, usage: 0x06, format: 1 })
  );

  // Build one combined multi-packet command buffer with a leading non-zero prefix to verify
  // exact view.buffer identity and byteOffset preservation without any .slice() / copy.
  const yData = new Uint8Array(YW * YH);
  yData[0] = 0x1a;
  yData[yData.length - 1] = 0x2b;

  const uData = new Uint8Array(CW * CH);
  uData[0] = 0x3c;
  uData[uData.length - 1] = 0x4d;

  const vData = new Uint8Array(CW * CH);
  vData[0] = 0x5e;
  vData[vData.length - 1] = 0x6f;

  const layerData = new Uint8Array(8 * 4 * 2);
  layerData[0] = 0x71;
  layerData[layerData.length - 1] = 0x82;

  // Packet 1: 4K Y with implicit tightly-packed bytesPerRow=0 (inferred 3840, 256B aligned).
  const pktY = writeTexturePacket({
    textureId: 600,
    width: YW,
    height: YH,
    bytesPerRow: 0,
    rowsPerImage: 0,
    data: yData,
  });
  // Packet 2: 1920x1080 U with explicit tightly-packed bytesPerRow=1920.
  const pktU = writeTexturePacket({
    textureId: 601,
    width: CW,
    height: CH,
    bytesPerRow: CW,
    rowsPerImage: CH,
    data: uData,
  });
  // Packet 3: 1920x1080 V with implicit tightly-packed bytesPerRow=0.
  const pktV = writeTexturePacket({
    textureId: 602,
    width: CW,
    height: CH,
    bytesPerRow: 0,
    rowsPerImage: 0,
    data: vData,
  });
  // Packet 4: multi-layer texture with rowsPerImage=0 (must populate rowsPerImage=height).
  const pktLayers = writeTexturePacket({
    textureId: 603,
    width: 8,
    height: 4,
    layers: 2,
    bytesPerRow: 8,
    rowsPerImage: 0,
    data: layerData,
  });

  const prefixPad = 64;
  const combined = new Uint8Array(
    prefixPad + pktY.byteLength + pktU.byteLength + pktV.byteLength + pktLayers.byteLength
  );
  let cursor = prefixPad;
  const offY = cursor;
  combined.set(pktY, cursor);
  cursor += pktY.byteLength;
  const offU = cursor;
  combined.set(pktU, cursor);
  cursor += pktU.byteLength;
  const offV = cursor;
  combined.set(pktV, cursor);
  cursor += pktV.byteLength;
  const offLayers = cursor;
  combined.set(pktLayers, cursor);

  const cmdSlice = combined.subarray(prefixPad);
  processor.executeCommandBuffer(cmdSlice);

  const wtCalls = gpu.calls.filter((c) => c.op === 'writeTexture');
  assert.equal(wtCalls.length, 4);

  // Every single upload must reference the exact underlying combined.buffer (zero copies!).
  assert.equal(wtCalls[0].bufferRef, combined.buffer);
  assert.equal(wtCalls[0].byteOffset, offY + 8 + 48);
  assert.equal(wtCalls[0].byteLength, YW * YH);
  assert.equal(wtCalls[0].dataLayout.bytesPerRow, YW);
  assert.equal(wtCalls[0].dataLayout.rowsPerImage, YH);
  assert.equal(wtCalls[0].bytes[0], 0x1a);
  assert.equal(wtCalls[0].bytes[YW * YH - 1], 0x2b);

  assert.equal(wtCalls[1].bufferRef, combined.buffer);
  assert.equal(wtCalls[1].byteOffset, offU + 8 + 48);
  assert.equal(wtCalls[1].byteLength, CW * CH);
  assert.equal(wtCalls[1].dataLayout.bytesPerRow, CW);
  assert.equal(wtCalls[1].bytes[0], 0x3c);
  assert.equal(wtCalls[1].bytes[CW * CH - 1], 0x4d);

  assert.equal(wtCalls[2].bufferRef, combined.buffer);
  assert.equal(wtCalls[2].byteOffset, offV + 8 + 48);
  assert.equal(wtCalls[2].byteLength, CW * CH);
  assert.equal(wtCalls[2].dataLayout.bytesPerRow, CW);
  assert.equal(wtCalls[2].bytes[0], 0x5e);
  assert.equal(wtCalls[2].bytes[CW * CH - 1], 0x6f);

  assert.equal(wtCalls[3].bufferRef, combined.buffer);
  assert.equal(wtCalls[3].byteOffset, offLayers + 8 + 48);
  assert.equal(wtCalls[3].byteLength, 64);
  assert.equal(wtCalls[3].dataLayout.bytesPerRow, 8);
  assert.equal(wtCalls[3].dataLayout.rowsPerImage, 4);
  assert.equal(wtCalls[3].bytes[0], 0x71);
  assert.equal(wtCalls[3].bytes[63], 0x82);

  const expectedTotalBytes = YW * YH + CW * CH * 2 + 64;
  const uploadStats = handler.getUploadStats();
  assert.deepEqual(uploadStats, {
    uploadCount: 4,
    uploadBytes: expectedTotalBytes,
    zeroCopyUploads: 4,
    scratchRepackUploads: 0,
    scratchBufferAllocations: 0,
    scratchBufferReuses: 0,
    scratchBufferBytes: 0,
    sabCopyCount: 0,
  });
  assert.deepEqual(handler.uploadStats, uploadStats);
  assert.equal(handler._scratchBuffer, null);
});

test('W191: upload telemetry distinguishes zero-copy, scratch-repack, oversized repack, SAB copy, and resets cleanly', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const stagingPool = new GpuStagingPool({
    device: gpu.device,
    chunkSize: 4 * 1024 * 1024,
    maxPoolBytes: 64 * 1024 * 1024,
  });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { stagingPool });
  // Cap persistent scratch at 4096 bytes to verify bounded growth and oversized fallback telemetry.
  const handler = tp.installTexturePackets(processor, { maxScratchBytes: 4096 });

  // Verify default cap constant is 16 MiB.
  assert.equal(tp.DEFAULT_MAX_SCRATCH_BYTES, 16 * 1024 * 1024);

  processor.executeCommandBuffer(
    createTexture2Packet({ id: 700, width: 256, height: 4, usage: 0x06, format: 1 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 701, width: 20, height: 4, usage: 0x06, format: 1 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 702, width: 20, height: 32, usage: 0x06, format: 1 })
  );

  // 1) Aligned non-SAB upload -> zero-copy direct writeTexture.
  const alignedPkt = writeTexturePacket({
    textureId: 700,
    width: 256,
    height: 4,
    bytesPerRow: 256,
    rowsPerImage: 4,
    data: new Uint8Array(1024),
    useSab: false,
  });
  processor.executeCommandBuffer(alignedPkt);

  // 2) Three repeated padded uploads (20x4 -> 256x4 = 1024B <= 4096B) -> 1 alloc + 2 reuses.
  let scratchRef = null;
  for (let i = 0; i < 3; i += 1) {
    const rowData = new Uint8Array(80);
    rowData.fill(0x20 + i);
    processor.executeCommandBuffer(
      writeTexturePacket({
        textureId: 701,
        width: 20,
        height: 4,
        bytesPerRow: 20,
        rowsPerImage: 4,
        data: rowData,
      })
    );
    if (i === 0) {
      scratchRef = handler._scratchBuffer;
      assert.ok(scratchRef instanceof Uint8Array);
      assert.equal(scratchRef.byteLength, 1024);
    } else {
      assert.equal(handler._scratchBuffer, scratchRef, 'Scratch Uint8Array must be reused');
    }
  }

  // 3) Oversized padded upload (20x32 -> 256x32 = 8192B > 4096B cap) -> one-off repack, NOT scratchRepackUploads.
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 702,
      width: 20,
      height: 32,
      bytesPerRow: 20,
      rowsPerImage: 32,
      data: new Uint8Array(640),
    })
  );
  assert.equal(handler._scratchBuffer, scratchRef, 'Oversized upload must not replace persistent scratch');

  // 4) Aligned SAB upload (256x4 = 1024B) -> copies out of SAB using existing scratch (reuse #3), NOT zeroCopy.
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 700,
      width: 256,
      height: 4,
      bytesPerRow: 256,
      rowsPerImage: 4,
      data: new Uint8Array(1024),
      useSab: true,
    })
  );

  // Verify copyFromSharedIfNeeded optional scratch reuse helper as well.
  const sabBuf = new SharedArrayBuffer(64);
  const sabView = new Uint8Array(sabBuf);
  sabView[0] = 0xaa;
  sabView[63] = 0xbb;
  const externalScratch = new Uint8Array(128);
  const sabCopied = tp.copyFromSharedIfNeeded(sabView, externalScratch);
  assert.equal(sabCopied.copiedFromSab, true);
  assert.equal(sabCopied.reusedScratch, true);
  assert.equal(sabCopied.data.buffer, externalScratch.buffer);
  assert.equal(sabCopied.data.byteLength, 64);
  assert.equal(sabCopied.data[0], 0xaa);
  assert.equal(sabCopied.data[63], 0xbb);

  const expectedBytes = 1024 + 3 * 1024 + 8192 + 1024;
  assert.deepEqual(handler.getUploadStats(), {
    uploadCount: 6,
    uploadBytes: expectedBytes,
    zeroCopyUploads: 1,
    scratchRepackUploads: 3,
    scratchBufferAllocations: 1,
    scratchBufferReuses: 3,
    scratchBufferBytes: 1024,
    sabCopyCount: 1,
  });
  assert.equal(handler.getStats().repackedUploads, 4);
  assert.equal(handler.getStats().scratchRepackUploads, 3);
  assert.equal(handler.getStats().uploadCount, 6);
  assert.equal(handler.getStats().uploadBytes, expectedBytes);
  assert.equal(handler.getStats().bytesUploaded, expectedBytes);

  // Reset zeros all upload telemetry cleanly.
  handler.reset();
  assert.deepEqual(handler.getUploadStats(), {
    uploadCount: 0,
    uploadBytes: 0,
    zeroCopyUploads: 0,
    scratchRepackUploads: 0,
    scratchBufferAllocations: 0,
    scratchBufferReuses: 0,
    scratchBufferBytes: 0,
    sabCopyCount: 0,
  });
});
