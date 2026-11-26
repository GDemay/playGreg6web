import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const {
  GpuStagingPool,
  STAGING_MODE,
  DEFAULT_MAX_WRITE_CHUNK_BYTES: STAGING_MAX_WRITE_CHUNK,
} = require('../web/workers/staging_pool.js');
const {
  ResourceTable,
  GPU_BUFFER_USAGE,
  GPU_TEXTURE_USAGE,
  DEFAULT_MAX_WRITE_CHUNK_BYTES: RT_MAX_WRITE_CHUNK,
} = require('../web/workers/resource_table.js');
const tp = require('../web/workers/texture_packets.js');
const { WgpuCommandProcessor, OPCODES } = require('../web/workers/wgpu_worker.js');

const MiB = 1024 * 1024;

function createBoundedMockGpu({
  maxAllowedWriteBufferBytes = 4 * MiB,
  enableWorkDoneFence = false,
  features = new Set(['texture-compression-bc']),
  allocateBufferBacking = true,
} = {}) {
  let nextHandle = 1;
  const createdBuffers = [];
  const createdTextures = [];
  const queueWrites = [];
  const writeTextureCalls = [];
  const deferredFences = [];

  const queue = {
    writeBuffer(buffer, bufferOffset, data) {
      if (typeof SharedArrayBuffer !== 'undefined' && data && data.buffer instanceof SharedArrayBuffer) {
        throw new TypeError('WebGPU validation: SharedArrayBuffer disallowed in writeBuffer');
      }
      const bytes =
        data instanceof Uint8Array
          ? data
          : ArrayBuffer.isView(data)
            ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
            : new Uint8Array(data);
      assert.equal(bufferOffset % 4, 0, 'writeBuffer offset must be 4-byte aligned');
      assert.equal(bytes.byteLength % 4, 0, 'writeBuffer size must be 4-byte aligned');
      assert.ok(
        bytes.byteLength <= maxAllowedWriteBufferBytes,
        `writeBuffer spike: ${bytes.byteLength} B > cap ${maxAllowedWriteBufferBytes} B`
      );
      if (buffer && typeof buffer.size === 'number') {
        assert.ok(bufferOffset + bytes.byteLength <= buffer.size, 'writeBuffer out of bounds');
      }
      if (buffer && buffer._data) {
        buffer._data.set(bytes, bufferOffset);
      }
      queueWrites.push({
        buffer,
        bufferOffset,
        byteLength: bytes.byteLength,
        firstByte: bytes.byteLength > 0 ? bytes[0] : -1,
        lastByte: bytes.byteLength > 0 ? bytes[bytes.byteLength - 1] : -1,
      });
    },
    writeTexture(destination, data, dataLayout, size) {
      if (typeof SharedArrayBuffer !== 'undefined' && data && data.buffer instanceof SharedArrayBuffer) {
        throw new TypeError('WebGPU validation: SharedArrayBuffer disallowed in writeTexture');
      }
      const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      writeTextureCalls.push({
        destination: {
          ...destination,
          origin: destination.origin ? { ...destination.origin } : { x: 0, y: 0, z: 0 },
        },
        dataLayout: { ...dataLayout },
        size: { ...size },
        byteLength: bytes.byteLength,
        firstByte: bytes.byteLength > 0 ? bytes[0] : -1,
        lastByte: bytes.byteLength > 0 ? bytes[bytes.byteLength - 1] : -1,
      });
    },
    submit() {},
  };

  if (enableWorkDoneFence) {
    queue.onSubmittedWorkDone = () => {
      let resolve;
      let reject;
      const promise = new Promise((res, rej) => {
        resolve = res;
        reject = rej;
      });
      deferredFences.push({ promise, resolve, reject });
      return promise;
    };
  }

  const device = {
    features,
    queue,
    createBuffer(desc) {
      const id = nextHandle++;
      const backing = allocateBufferBacking ? new ArrayBuffer(desc.size) : null;
      const dataView = backing ? new Uint8Array(backing) : null;
      const buf = {
        kind: 'GPUBuffer',
        id,
        handle: id,
        size: desc.size,
        usage: desc.usage,
        label: desc.label || `buf-${id}`,
        _data: dataView,
        _mapped: Boolean(desc.mappedAtCreation),
        destroyed: false,
        destroyCount: 0,
        getMappedRange() {
          if (!this._mapped) throw new Error('buffer not mapped');
          return backing;
        },
        unmap() {
          this._mapped = false;
        },
        async mapAsync() {
          this._mapped = true;
        },
        destroy() {
          this._mapped = false;
          this.destroyed = true;
          this.destroyCount += 1;
        },
      };
      createdBuffers.push(buf);
      return buf;
    },
    createTexture(desc) {
      const id = nextHandle++;
      const tex = {
        kind: 'GPUTexture',
        id,
        handle: id,
        format: desc.format,
        dimension: desc.dimension || '2d',
        width: desc.size && desc.size.width ? desc.size.width : 1,
        height: desc.size && desc.size.height ? desc.size.height : 1,
        depthOrArrayLayers:
          desc.size && desc.size.depthOrArrayLayers ? desc.size.depthOrArrayLayers : 1,
        mipLevelCount: desc.mipLevelCount || 1,
        desc,
        destroyed: false,
        destroyCount: 0,
        createView(viewDesc) {
          return { kind: 'GPUTextureView', textureHandle: id, texture: tex, viewDesc };
        },
        destroy() {
          this.destroyed = true;
          this.destroyCount += 1;
        },
      };
      createdTextures.push(tex);
      return tex;
    },
    createCommandEncoder() {
      return createTrackingEncoder();
    },
  };

  const ctx = { getCurrentTexture: () => ({ createView: () => ({ kind: 'swapchain' }) }) };
  return { device, ctx, queue, createdBuffers, createdTextures, queueWrites, writeTextureCalls, deferredFences };
}

function createTrackingEncoder() {
  const bufferCopies = [];
  const textureCopies = [];
  return {
    bufferCopies,
    textureCopies,
    copyBufferToBuffer(srcBuffer, srcOffset, dstBuffer, dstOffset, size) {
      if (srcBuffer._data && dstBuffer._data) {
        dstBuffer._data.set(srcBuffer._data.subarray(srcOffset, srcOffset + size), dstOffset);
      }
      bufferCopies.push({ srcBuffer, srcOffset, dstBuffer, dstOffset, size });
    },
    copyBufferToTexture(source, destination, copySize) {
      textureCopies.push({
        source: { ...source },
        destination: {
          ...destination,
          origin: destination.origin ? { ...destination.origin } : { x: 0, y: 0, z: 0 },
        },
        copySize: { ...copySize },
      });
    },
    finish() {
      return { kind: 'GPUCommandBuffer' };
    },
  };
}

function createTexture2Packet({
  id,
  width,
  height,
  layers = 1,
  mips = 1,
  samples = 1,
  usage = 0x06,
  dimension = 1,
  format,
}) {
  const out = new Uint8Array(8 + 40);
  const view = new DataView(out.buffer);
  view.setUint16(0, 35, true);
  view.setUint32(4, 40, true);
  view.setUint32(8, id, true);
  view.setUint32(12, width, true);
  view.setUint32(16, height, true);
  view.setUint32(20, layers, true);
  view.setUint32(24, mips, true);
  view.setUint32(28, samples, true);
  view.setUint32(32, usage, true);
  view.setUint8(36, dimension);
  view.setUint8(37, format);
  view.setUint8(38, 0);
  return out;
}

function writeTexturePacket({
  textureId,
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
  view.setUint32(8, textureId, true);
  view.setUint32(12, 0, true);
  view.setUint32(16, 0, true);
  view.setUint32(20, 0, true);
  view.setUint32(24, 0, true);
  view.setUint32(28, width, true);
  view.setUint32(32, height, true);
  view.setUint32(36, layers, true);
  view.setUint32(40, bytesPerRow, true);
  view.setUint32(44, rowsPerImage, true);
  view.setUint32(48, data.byteLength, true);
  view.setUint32(52, 0, true);
  out.set(data, 56);
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

test('W352: GpuStagingPool chunks huge (>4 MiB) buffer uploads through bounded <=4 MiB chunks in writeBuffer and mapped modes', () => {
  assert.equal(STAGING_MAX_WRITE_CHUNK, 4 * MiB);

  const { device, createdBuffers, queueWrites } = createBoundedMockGpu({
    maxAllowedWriteBufferBytes: 4 * MiB,
    allocateBufferBacking: true,
  });

  const pool = new GpuStagingPool({
    device,
    chunkSize: 4 * MiB,
    maxPoolBytes: 64 * MiB,
    maxWriteChunkBytes: 4 * MiB,
    framesInFlight: 3,
  });

  const encoder = createTrackingEncoder();
  const dst1 = device.createBuffer({ size: 16 * MiB, usage: GPU_BUFFER_USAGE.COPY_DST });

  // 1) Plain 12 MiB upload -> 3x 4 MiB bounded staging chunks, zero oversized >4 MiB GPUBuffer!
  const bigPlain = new Uint8Array(12 * MiB);
  bigPlain[0] = 0x11;
  bigPlain[4 * MiB] = 0x22;
  bigPlain[8 * MiB] = 0x33;
  bigPlain[12 * MiB - 1] = 0x44;

  pool.beginFrame(0);
  const res1 = pool.stageBufferUpload(encoder, dst1, 0, bigPlain);
  assert.equal(res1.size, 12 * MiB);
  assert.equal(res1.srcByteLength, 12 * MiB);
  assert.equal(queueWrites.length, 3);
  assert.ok(queueWrites.every((w) => w.byteLength <= 4 * MiB));
  assert.equal(encoder.bufferCopies.length, 3);
  assert.equal(dst1._data[0], 0x11);
  assert.equal(dst1._data[4 * MiB], 0x22);
  assert.equal(dst1._data[8 * MiB], 0x33);
  assert.equal(dst1._data[12 * MiB - 1], 0x44);

  const stagingBufsAfter1 = createdBuffers.filter((b) => b !== dst1);
  assert.ok(
    stagingBufsAfter1.every((b) => b.size <= 4 * MiB),
    'chunked buffer upload must never allocate an oversized >4 MiB staging GPUBuffer'
  );
  pool.endFrame();

  // 2) SAB-backed odd-length upload (9 MiB + 3 bytes -> pads final tail to +4 bytes) reuses bounded <=4 MiB scratch!
  const sabLen = 9 * MiB + 3;
  const sab = new SharedArrayBuffer(sabLen);
  const sabView = new Uint8Array(sab);
  sabView[0] = 0xa1;
  sabView[4 * MiB] = 0xb2;
  sabView[8 * MiB] = 0xc3;
  sabView[sabLen - 1] = 0xd4;
  const dst2 = device.createBuffer({ size: 10 * MiB, usage: GPU_BUFFER_USAGE.COPY_DST });

  pool.beginFrame(1);
  const res2 = pool.stageBufferUpload(encoder, dst2, 0, sabView);
  assert.equal(res2.size, 9 * MiB + 4);
  assert.equal(res2.srcByteLength, sabLen);
  assert.ok(pool._uploadScratch instanceof Uint8Array);
  assert.ok(
    pool._uploadScratch.byteLength <= 4 * MiB,
    'SAB chunked buffer upload scratch must stay <= 4 MiB'
  );
  assert.equal(dst2._data[0], 0xa1);
  assert.equal(dst2._data[4 * MiB], 0xb2);
  assert.equal(dst2._data[8 * MiB], 0xc3);
  assert.equal(dst2._data[sabLen - 1], 0xd4);
  assert.equal(dst2._data[sabLen], 0x00, 'final unaligned tail byte must be zero-padded');
  pool.endFrame();

  const stats = pool.getStats();
  assert.equal(stats.chunkedWrites, 2);
  assert.equal(stats.sabCopyCount, 1);
  assert.equal(stats.alignPadCount, 1);
  assert.equal(stats.oversizedAllocations, 0);

  // 3) Mapped mode also chunks >4 MiB uploads cleanly without calling queue.writeBuffer!
  const mappedGpu = createBoundedMockGpu({ maxAllowedWriteBufferBytes: 4 * MiB });
  const mappedPool = new GpuStagingPool({
    device: mappedGpu.device,
    chunkSize: 4 * MiB,
    maxPoolBytes: 32 * MiB,
    maxWriteChunkBytes: 4 * MiB,
    mode: STAGING_MODE.MAPPED,
  });
  const mappedEnc = createTrackingEncoder();
  const mappedDst = mappedGpu.device.createBuffer({ size: 10 * MiB, usage: GPU_BUFFER_USAGE.COPY_DST });
  const mappedSrc = new Uint8Array(6 * MiB + 1);
  mappedSrc[0] = 0x55;
  mappedSrc[4 * MiB] = 0x66;
  mappedSrc[6 * MiB] = 0x77;

  mappedPool.beginFrame(0);
  mappedPool.stageBufferUpload(mappedEnc, mappedDst, 0, mappedSrc);
  assert.equal(mappedGpu.queueWrites.length, 0);
  assert.equal(mappedEnc.bufferCopies.length, 2);
  assert.equal(mappedDst._data[0], 0x55);
  assert.equal(mappedDst._data[4 * MiB], 0x66);
  assert.equal(mappedDst._data[6 * MiB], 0x77);
  assert.equal(mappedDst._data[6 * MiB + 1], 0x00);
  assert.equal(mappedPool.getStats().chunkedWrites, 1);

  pool.destroy();
  mappedPool.destroy();
  assert.ok(pool.getStats().destroyedBuffers > 0);
  assert.ok(mappedPool.getStats().destroyedBuffers > 0);
});

test('W352: GpuStagingPool chunks 4K texture writeBuffer calls <=4 MiB, reuses scratch across frames, and tracks trimmedStagingBytes', () => {
  const { device, queueWrites } = createBoundedMockGpu({
    maxAllowedWriteBufferBytes: 4 * MiB,
    allocateBufferBacking: true,
  });
  const pool = new GpuStagingPool({
    device,
    chunkSize: 4 * MiB,
    maxPoolBytes: 64 * MiB,
    maxWriteChunkBytes: 4 * MiB,
    framesInFlight: 3,
    idleShrinkFrames: 10,
  });

  const encoder = createTrackingEncoder();
  const dstY = { id: 'y-4k' };
  const dstU = { id: 'u-4k' };

  // 4K Bink2 Y (3840x2160 = 7.91 MiB, SAB-backed) and unaligned U (1920x1080 -> 2048x1080 = 2.11 MiB, SAB-backed).
  const ySab = new Uint8Array(new SharedArrayBuffer(3840 * 2160));
  const uSab = new Uint8Array(new SharedArrayBuffer(1920 * 1080));
  ySab[0] = 0x12;
  ySab[4 * MiB] = 0x34;
  ySab[3840 * 2160 - 1] = 0x56;
  uSab[0] = 0x78;
  uSab[1919] = 0x9a;

  let scratchRef = null;
  for (let frame = 0; frame < 5; frame += 1) {
    pool.beginFrame(frame);
    const yRes = pool.stageTextureUpload(encoder, dstY, ySab, {
      width: 3840,
      height: 2160,
      bytesPerRow: 3840,
    });
    if (frame === 0) {
      scratchRef = pool._uploadScratch;
      assert.ok(scratchRef instanceof Uint8Array);
      assert.equal(scratchRef.byteLength, 4 * MiB);
      // Poison chroma padding zone to prove zero-padding on reused scratch.
      scratchRef.fill(0xff, 1920, 2048);
    }
    const uRes = pool.stageTextureUpload(encoder, dstU, uSab, {
      width: 1920,
      height: 1080,
      bytesPerRow: 1920,
    });
    assert.equal(
      pool._uploadScratch,
      scratchRef,
      `Frame ${frame}: stageTextureUpload must reuse the single 4 MiB scratch buffer with zero reallocations`
    );
    // Verify zeroed trailing alignment bytes on row 0 of U plane.
    for (let pad = 1920; pad < 2048; pad += 1) {
      assert.equal(uRes.buffer._data[uRes.offset + pad], 0);
    }
    assert.equal(yRes.buffer._data[yRes.offset], 0x12);
    assert.equal(yRes.buffer._data[yRes.offset + 4 * MiB], 0x34);
    assert.equal(yRes.buffer._data[yRes.offset + 3840 * 2160 - 1], 0x56);
    pool.endFrame();
  }

  // Every single queue.writeBuffer call across all 5 frames stayed <= 4 MiB!
  assert.ok(queueWrites.length >= 15); // 5 * (2 chunks for Y + 1 chunk for U)
  assert.ok(queueWrites.every((w) => w.byteLength <= 4 * MiB));
  assert.equal(pool.getStats().chunkedWrites, 5);

  // Advance past framesInFlight (7 - 4 = 3) before idleShrinkFrames (10), then trim for scene transition!
  pool.beginFrame(7);
  const trimRes = pool.trimForSceneTransition();
  assert.ok(trimRes.reclaimedSlabs >= 1);
  assert.ok(trimRes.reclaimedBytes >= 8 * MiB);
  assert.equal(pool._uploadScratch, null, 'trimForSceneTransition must release idle CPU upload scratch');

  const statsAfterTrim = pool.getStats();
  assert.equal(statsAfterTrim.trimmedStagingBytes, trimRes.reclaimedBytes);
  assert.equal(statsAfterTrim.destroyedBuffers, trimRes.reclaimedSlabs);

  // Also verify automatic idleShrinkFrames trims oversized slabs when idle >= 10 frames!
  pool.stageTextureUpload(encoder, dstY, ySab, { width: 3840, height: 2160, bytesPerRow: 3840 });
  pool.endFrame();
  const autoShrinkBegin = pool.beginFrame(20);
  assert.ok(autoShrinkBegin.shrunkSlabs >= 1);
  assert.ok(autoShrinkBegin.shrunkBytes >= 8 * MiB);
  assert.equal(
    pool.getStats().trimmedStagingBytes,
    trimRes.reclaimedBytes + autoShrinkBegin.shrunkBytes
  );
  assert.equal(
    pool.getStats().destroyedBuffers,
    trimRes.reclaimedSlabs + autoShrinkBegin.shrunkSlabs
  );

  const reset = pool.resetStats();
  assert.equal(reset.chunkedWrites, 0);
  assert.equal(reset.trimmedStagingBytes, 0);
  assert.equal(reset.destroyedBuffers, 0);
  pool.destroy();
});

test('W352: ResourceTable chunks >4 MiB writeBuffer calls and destroys every replaced/deferred buffer & texture cleanly', async () => {
  assert.equal(RT_MAX_WRITE_CHUNK, 4 * MiB);

  const { device, queue, queueWrites, deferredFences } = createBoundedMockGpu({
    maxAllowedWriteBufferBytes: 4 * MiB,
    enableWorkDoneFence: true,
    allocateBufferBacking: true,
  });

  const table = new ResourceTable(device, { maxWriteChunkBytes: 4 * MiB });

  // 1) Huge plain + SAB odd-length writeBuffer calls stay strictly <= 4 MiB per queue.writeBuffer!
  const dstBuf = table.createBuffer(1, {
    size: 16 * MiB,
    usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST,
  });
  const plain12 = new Uint8Array(12 * MiB);
  plain12[0] = 0x21;
  plain12[4 * MiB] = 0x43;
  plain12[8 * MiB] = 0x65;
  plain12[12 * MiB - 1] = 0x87;

  assert.equal(table.writeBuffer(1, 0, plain12), true);
  assert.equal(queueWrites.length, 3);
  assert.ok(queueWrites.every((w) => w.byteLength <= 4 * MiB));
  assert.equal(dstBuf._data[0], 0x21);
  assert.equal(dstBuf._data[4 * MiB], 0x43);
  assert.equal(dstBuf._data[8 * MiB], 0x65);
  assert.equal(dstBuf._data[12 * MiB - 1], 0x87);

  const sabOddLen = 9 * MiB + 1;
  const sabOdd = new Uint8Array(new SharedArrayBuffer(sabOddLen));
  sabOdd[0] = 0xaa;
  sabOdd[4 * MiB] = 0xbb;
  sabOdd[8 * MiB] = 0xcc;
  sabOdd[sabOddLen - 1] = 0xdd;
  assert.equal(table.writeBuffer(1, 0, sabOdd), true);
  assert.equal(dstBuf._data[0], 0xaa);
  assert.equal(dstBuf._data[4 * MiB], 0xbb);
  assert.equal(dstBuf._data[8 * MiB], 0xcc);
  assert.equal(dstBuf._data[sabOddLen - 1], 0xdd);
  assert.equal(dstBuf._data[sabOddLen], 0x00, 'tail byte must be zero-padded');
  assert.ok(table._writeScratch.byteLength <= 4 * MiB);
  assert.equal(table.getStats().chunkedWrites, 2);

  // 2) Mid-frame overwrites & explicit destroys defer safely and execute .destroy() exactly once at endFrame(queue)!
  const texA = table.createTexture(10, { size: { width: 64, height: 64 }, format: 'rgba8unorm' });
  const texB = table.createTexture(11, { size: { width: 64, height: 64 }, format: 'rgba8unorm' });
  const bufA = table.createBuffer(20, { size: 256, usage: GPU_BUFFER_USAGE.UNIFORM });

  table.beginFrame();
  // Recreate IDs 10 and 20 (overwrites) + destroy ID 11 and ID 1.
  const texA2 = table.createTexture(10, { size: { width: 128, height: 128 }, format: 'rgba8unorm' });
  const bufA2 = table.createBuffer(20, { size: 512, usage: GPU_BUFFER_USAGE.UNIFORM });
  table.destroyTexture(11);
  table.destroyBuffer(1);

  assert.equal(texA.destroyed, false);
  assert.equal(texB.destroyed, false);
  assert.equal(bufA.destroyed, false);
  assert.equal(dstBuf.destroyed, false);
  assert.equal(table.getStats().pendingDeferredDestroyCount, 4);
  assert.equal(table.getStats().destroyedBuffers, 0);
  assert.equal(table.getStats().destroyedTextures, 0);

  const endPromise = table.endFrame(queue);
  assert.equal(deferredFences.length, 1);
  deferredFences[0].resolve();
  const destroyedCount = await endPromise;
  assert.equal(destroyedCount, 4);

  assert.equal(texA.destroyCount, 1, 'replaced texture must be destroyed exactly once');
  assert.equal(texB.destroyCount, 1, 'destroyed texture must be destroyed exactly once');
  assert.equal(bufA.destroyCount, 1, 'replaced buffer must be destroyed exactly once');
  assert.equal(dstBuf.destroyCount, 1, 'destroyed buffer must be destroyed exactly once');
  assert.equal(texA2.destroyed, false);
  assert.equal(bufA2.destroyed, false);

  const statsAfterEnd = table.getStats();
  assert.equal(statsAfterEnd.pendingDeferredDestroyCount, 0);
  assert.equal(statsAfterEnd.destroyedBuffers, 2);
  assert.equal(statsAfterEnd.destroyedTextures, 2);

  // 3) Failing createView() during createTexture() must destroy the orphaned GPUTexture immediately!
  let orphanedTex = null;
  const origCreateTexture = device.createTexture;
  device.createTexture = (desc) => {
    const t = origCreateTexture(desc);
    orphanedTex = t;
    t.createView = () => {
      throw new Error('simulated createView failure');
    };
    return t;
  };
  assert.throws(
    () => table.createTexture(99, { size: { width: 32, height: 32 }, format: 'rgba8unorm' }),
    /simulated createView failure/
  );
  device.createTexture = origCreateTexture;
  assert.ok(orphanedTex);
  assert.equal(orphanedTex.destroyCount, 1, 'orphaned texture on createView failure must be destroyed');
  assert.equal(table.hasTexture(99), false);
  assert.equal(table.getStats().destroyedTextures, 3);

  table.destroyAll();
  assert.equal(texA2.destroyCount, 1);
  assert.equal(bufA2.destroyCount, 1);
  assert.equal(table._writeScratch, null);
  assert.equal(table.getStats().destroyedBuffers, 3);
  assert.equal(table.getStats().destroyedTextures, 4);

  table.resetStats();
  assert.equal(table.getStats().chunkedWrites, 0);
  assert.equal(table.getStats().destroyedBuffers, 0);
  assert.equal(table.getStats().destroyedTextures, 0);
});

test('W352: TexturePackets chunks >8 MiB SAB 4K RGBA/array uploads through bounded scratch, preserves W341/BC3D guards, and reports telemetry', () => {
  assert.equal(tp.DEFAULT_MAX_WRITE_TEXTURE_CHUNK_BYTES, 8 * MiB);

  const gpu = createBoundedMockGpu({
    maxAllowedWriteBufferBytes: 4 * MiB,
    features: new Set(['texture-compression-bc']), // no texture-compression-bc-sliced-3d
  });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor, {
    maxScratchBytes: 8 * MiB,
    maxWriteChunkBytes: 4 * MiB,
  });

  // 1) Huge 2048x2048 RGBA8 (16 MiB) upload from SAB -> chunked into 4 strips of <= 4 MiB,
  // reusing ONE 4 MiB scratch buffer (ZERO 16 MiB transient allocation!).
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 100, width: 2048, height: 2048, format: 22 }) // rgba8unorm
  );
  const rgbaBytesPerRow = 2048 * 4; // 8192 bytes/row -> 512 rows = 4 MiB per strip!
  const rgbaData = new Uint8Array(rgbaBytesPerRow * 2048);
  rgbaData[0] = 0x10;
  rgbaData[512 * rgbaBytesPerRow] = 0x20;
  rgbaData[1024 * rgbaBytesPerRow] = 0x30;
  rgbaData[1536 * rgbaBytesPerRow] = 0x40;
  rgbaData[rgbaData.byteLength - 1] = 0x99;

  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 100,
      width: 2048,
      height: 2048,
      bytesPerRow: rgbaBytesPerRow,
      rowsPerImage: 2048,
      data: rgbaData,
      useSab: true,
    })
  );

  assert.equal(gpu.writeTextureCalls.length, 4, '16 MiB texture at 4 MiB cap must split into 4 strips');
  assert.ok(gpu.writeTextureCalls.every((c) => c.byteLength <= 4 * MiB));
  assert.deepEqual(
    gpu.writeTextureCalls.map((c) => c.destination.origin.y),
    [0, 512, 1024, 1536]
  );
  assert.deepEqual(
    gpu.writeTextureCalls.map((c) => c.size.height),
    [512, 512, 512, 512]
  );
  assert.deepEqual(
    gpu.writeTextureCalls.map((c) => c.firstByte),
    [0x10, 0x20, 0x30, 0x40]
  );
  assert.equal(gpu.writeTextureCalls[3].lastByte, 0x99);
  assert.equal(
    handler._scratchBuffer.byteLength,
    4 * MiB,
    '16 MiB SAB upload must reuse a 4 MiB scratch buffer across strips instead of allocating 16 MiB'
  );
  assert.equal(handler.getStats().scratchBufferAllocations, 1);
  assert.equal(handler.getStats().scratchBufferReuses, 3);
  assert.equal(handler.getStats().chunkedWrites, 1);

  // 2) 2D array texture (512x512x8 RGBA8 = 8 MiB -> 2 batches of 4 slices <= 4 MiB).
  gpu.writeTextureCalls.length = 0;
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 101, width: 512, height: 512, layers: 8, format: 22 })
  );
  const sliceSize = 512 * 4 * 512; // 1 MiB per slice
  const arrData = new Uint8Array(sliceSize * 8);
  arrData[0] = 0xa0;
  arrData[4 * sliceSize] = 0xb0;
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 101,
      width: 512,
      height: 512,
      layers: 8,
      bytesPerRow: 512 * 4,
      rowsPerImage: 512,
      data: arrData,
      useSab: true,
    })
  );
  assert.equal(gpu.writeTextureCalls.length, 2);
  assert.deepEqual(
    gpu.writeTextureCalls.map((c) => c.destination.origin.z),
    [0, 4]
  );
  assert.deepEqual(
    gpu.writeTextureCalls.map((c) => c.size.depthOrArrayLayers),
    [4, 4]
  );
  assert.deepEqual(
    gpu.writeTextureCalls.map((c) => c.firstByte),
    [0xa0, 0xb0]
  );
  assert.equal(handler.getStats().chunkedWrites, 2);

  // 3) W341 1D->2D LUT promotion + W108 BC3D fallback guard remain intact!
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 102, width: 4096, height: 1, dimension: 0, format: 16 }) // 1d r32float
  );
  assert.equal(processor.textures.get(102).desc.dimension, '2d');
  assert.equal(handler.getStats().promoted1DTo2DCount, 1);

  processor.executeCommandBuffer(
    createTexture2Packet({ id: 103, width: 8, height: 8, layers: 4, dimension: 2, format: 62 }) // 3d bc7
  );
  assert.equal(processor.textures.get(103).desc.format, 'rgba8unorm');
  assert.equal(handler.getStats().bc3dFallbacks, 1);

  // 4) Overwrite + explicit destroy track destroyedTextures; trimScratch tracks trimmedStagingBytes!
  const prevTex100 = processor.textures.get(100);
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 100, width: 16, height: 16, format: 22 })
  );
  assert.equal(prevTex100.destroyCount, 1);
  processor.executeCommandBuffer(destroyPacket(100));
  assert.equal(handler.getStats().destroyedTextures, 2);

  const trimmed = handler.trimScratch(0);
  assert.equal(trimmed.trimmedSlots, 1);
  assert.equal(trimmed.trimmedBytes, 4 * MiB);
  assert.equal(handler._scratchBuffer, null);
  assert.equal(handler.getStats().trimmedStagingBytes, 4 * MiB);

  handler.resetStats();
  const afterReset = handler.getStats();
  assert.equal(afterReset.chunkedWrites, 0);
  assert.equal(afterReset.trimmedStagingBytes, 0);
  assert.equal(afterReset.destroyedTextures, 0);
});
