import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import fs from 'node:fs';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const stagingMod = require('../web/workers/staging_pool.js');

const {
  alignUp,
  alignBufferSize,
  isSharedBuffer,
  sanitizeWriteBufferPayload,
  GpuStagingPool,
  BUFFER_USAGE,
  MAP_MODE,
  STAGING_MODE,
  WEBGPU_BUFFER_COPY_ALIGNMENT,
  WEBGPU_BYTES_PER_ROW_ALIGNMENT,
} = stagingMod;

/**
 * Create a deterministic mock WebGPU device, buffers, queue, and command encoder
 * supporting both `queue.writeBuffer` and `mappedAtCreation` / `mapAsync` modes.
 */
function createMockGpuDevice(options = {}) {
  let nextBufferId = 1;
  const createdBuffers = [];
  const queueWrites = [];
  const deferredFences = [];

  const queue = {
    writeBuffer(buffer, bufferOffset, data) {
      const bytes =
        data instanceof Uint8Array
          ? data
          : ArrayBuffer.isView(data)
            ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
            : new Uint8Array(data);
      const copy = bytes.slice();
      buffer._data.set(copy, bufferOffset);
      queueWrites.push({
        buffer,
        bufferOffset,
        byteLength: copy.byteLength,
        data: copy,
      });
    },
  };

  if (options.enableWorkDoneFence) {
    queue.onSubmittedWorkDone = () => {
      let resolve;
      const promise = new Promise((res) => {
        resolve = res;
      });
      deferredFences.push({ promise, resolve });
      return promise;
    };
  }

  const device = {
    queue,
    createBuffer(descriptor) {
      const id = nextBufferId++;
      const backing = new ArrayBuffer(descriptor.size);
      const dataView = new Uint8Array(backing);
      const buffer = {
        id,
        label: descriptor.label || `buf-${id}`,
        size: descriptor.size,
        usage: descriptor.usage,
        mappedAtCreation: Boolean(descriptor.mappedAtCreation),
        _data: dataView,
        _mapped: Boolean(descriptor.mappedAtCreation),
        unmapCount: 0,
        mapAsyncCount: 0,
        destroyCount: 0,
        getMappedRange(offset = 0, size = descriptor.size - offset) {
          if (!this._mapped) {
            throw new Error(`Buffer ${this.id} is not mapped`);
          }
          if (offset === 0 && size === descriptor.size) {
            return backing;
          }
          return backing.slice(offset, offset + size);
        },
        unmap() {
          if (!this._mapped) {
            throw new Error(`Buffer ${this.id} is already unmapped`);
          }
          this._mapped = false;
          this.unmapCount += 1;
        },
        async mapAsync(mode) {
          assert.equal(mode, MAP_MODE.WRITE);
          this.mapAsyncCount += 1;
          this._mapped = true;
        },
        destroy() {
          this._mapped = false;
          this.destroyCount += 1;
        },
      };
      createdBuffers.push(buffer);
      return buffer;
    },
  };

  return { device, createdBuffers, queueWrites, deferredFences };
}

function createMockEncoder() {
  const bufferCopies = [];
  const textureCopies = [];
  return {
    bufferCopies,
    textureCopies,
    copyBufferToBuffer(srcBuffer, srcOffset, dstBuffer, dstOffset, size) {
      bufferCopies.push({
        srcBuffer,
        srcOffset,
        dstBuffer,
        dstOffset,
        size,
        snapshot: srcBuffer._data.slice(srcOffset, srcOffset + size),
      });
    },
    copyBufferToTexture(source, destination, copySize) {
      const totalBytes = source.bytesPerRow * source.rowsPerImage * copySize.depthOrArrayLayers;
      textureCopies.push({
        source: { ...source },
        destination: {
          ...destination,
          origin: { ...destination.origin },
        },
        copySize: { ...copySize },
        snapshot: source.buffer._data.slice(source.offset, source.offset + totalBytes),
      });
    },
  };
}

test('UMD export attaches to globalThis.PlayGta6StagingPool and module.exports', () => {
  assert.equal(globalThis.PlayGta6StagingPool, stagingMod);
  assert.equal(typeof alignUp, 'function');
  assert.equal(typeof GpuStagingPool, 'function');
  assert.equal(WEBGPU_BUFFER_COPY_ALIGNMENT, 4);
  assert.equal(WEBGPU_BYTES_PER_ROW_ALIGNMENT, 256);
});

test('alignUp aligns 4, 16, 256, and large values accurately and validates inputs', () => {
  assert.equal(alignUp(0, 4), 0);
  assert.equal(alignUp(1, 4), 4);
  assert.equal(alignUp(4, 4), 4);
  assert.equal(alignUp(5, 4), 8);

  assert.equal(alignUp(0, 16), 0);
  assert.equal(alignUp(1, 16), 16);
  assert.equal(alignUp(16, 16), 16);
  assert.equal(alignUp(17, 16), 32);

  assert.equal(alignUp(0, 256), 0);
  assert.equal(alignUp(1, 256), 256);
  assert.equal(alignUp(256, 256), 256);
  assert.equal(alignUp(257, 256), 512);
  assert.equal(alignUp(300, 256), 512);

  // Beyond 32-bit signed/unsigned range.
  assert.equal(alignUp(0x100000001, 256), 0x100000100);

  assert.throws(() => alignUp(-1, 4), RangeError);
  assert.throws(() => alignUp(1.5, 4), RangeError);
  assert.throws(() => alignUp(8, 0), RangeError);
  assert.throws(() => alignUp(8, -4), RangeError);
});

test('sub-allocation enforces 4, 16, and 256-byte alignments within a chunk', () => {
  const { device, createdBuffers } = createMockGpuDevice();
  const pool = new GpuStagingPool({
    device,
    chunkSize: 1024,
    maxPoolBytes: 4096,
    framesInFlight: 3,
  });

  pool.beginFrame(0);

  const s1 = pool.allocate(5, 4);
  assert.equal(s1.offset, 0);
  assert.equal(s1.size, 5);

  const s2 = pool.allocate(3, 4);
  assert.equal(s2.offset, 8);
  assert.equal(s2.size, 3);
  assert.equal(s2.buffer, s1.buffer);

  const s3 = pool.allocate(10, 16);
  assert.equal(s3.offset, 16);
  assert.equal(s3.size, 10);
  assert.equal(s3.buffer, s1.buffer);

  const s4 = pool.allocate(64, 256);
  assert.equal(s4.offset, 256);
  assert.equal(s4.size, 64);
  assert.equal(s4.buffer, s1.buffer);

  const s5 = pool.allocate(32, 256);
  assert.equal(s5.offset, 512);
  assert.equal(s5.size, 32);
  assert.equal(s5.buffer, s1.buffer);

  assert.equal(createdBuffers.length, 1);
  assert.equal(createdBuffers[0].usage, BUFFER_USAGE.COPY_SRC | BUFFER_USAGE.COPY_DST);
  assert.equal(pool.getStats().allocationCount, 5);

  pool.destroy();
});

test('chunk rollover creates additional chunks when current chunk cannot fit aligned allocation', () => {
  const { device, createdBuffers } = createMockGpuDevice();
  const pool = new GpuStagingPool({
    device,
    chunkSize: 512,
    maxPoolBytes: 2048,
    framesInFlight: 2,
  });

  pool.beginFrame(0);

  const a = pool.allocate(300, 4);
  assert.equal(a.offset, 0);

  // Aligning 300 up to 256 gives 512, so 512 + 64 > 512 -> must roll over to a 2nd chunk!
  const b = pool.allocate(64, 256);
  assert.equal(b.offset, 0);
  assert.notEqual(b.buffer, a.buffer);

  // Next small allocation fits in chunk 2.
  const c = pool.allocate(128, 16);
  assert.equal(c.buffer, b.buffer);
  assert.equal(c.offset, 64);

  assert.equal(createdBuffers.length, 2);
  const stats = pool.getStats();
  assert.equal(stats.totalChunksAllocated, 2);
  assert.equal(stats.activeChunks, 2);
  assert.equal(stats.allocatedBytes, 1024);
  assert.equal(stats.peakAllocatedBytes, 1024);
  assert.equal(stats.bufferReuseMisses, 2);
  assert.equal(stats.bufferReuseHits, 0);

  pool.destroy();
});

test('frame-epoch recycling avoids device.createBuffer churn on steady-state frames', () => {
  const { device, createdBuffers } = createMockGpuDevice();
  const pool = new GpuStagingPool({
    device,
    chunkSize: 1024,
    maxPoolBytes: 8192,
    framesInFlight: 3,
  });

  const encoder = createMockEncoder();
  const dstBuffer = { id: 'dst-vb' };
  const payload = new Uint8Array(256).fill(0xab);

  // Warmup frames 0, 1, 2 allocate 3 chunks (one per in-flight frame).
  for (let frame = 0; frame < 3; frame += 1) {
    pool.beginFrame(frame);
    pool.stageBufferUpload(encoder, dstBuffer, 0, payload);
    const frameStats = pool.endFrame();
    assert.equal(frameStats.frameIndex, frame);
    assert.equal(frameStats.chunksRetired, 1);
    assert.equal(frameStats.bytesUploaded, 256);
  }

  assert.equal(createdBuffers.length, 3);
  assert.equal(pool.getStats().inFlightChunks, 3);
  assert.equal(pool.getStats().recycledChunks, 0);
  assert.equal(pool.getStats().bufferReuseHits, 0);

  // Steady-state frames 3..11: ZERO new device.createBuffer() calls!
  for (let frame = 3; frame < 12; frame += 1) {
    const beginInfo = pool.beginFrame(frame);
    assert.equal(beginInfo.recycledCount, 1);
    pool.stageBufferUpload(encoder, dstBuffer, 0, payload);
    pool.endFrame();
  }

  assert.equal(createdBuffers.length, 3, 'steady-state frames must allocate zero new GPUBuffers');

  const stats = pool.getStats();
  assert.equal(stats.totalChunksAllocated, 3);
  assert.equal(stats.activeChunks, 0);
  assert.equal(stats.inFlightChunks, 3);
  assert.equal(stats.recycledChunks, 0);
  assert.equal(stats.allocatedBytes, 3072);
  assert.equal(stats.peakAllocatedBytes, 3072);
  assert.equal(stats.allocationCount, 12);
  assert.equal(stats.bytesUploaded, 12 * 256);
  assert.equal(stats.bufferReuseMisses, 3);
  assert.equal(stats.bufferReuseHits, 9);
  assert.ok(stats.bufferReuseHits > 0);

  pool.destroy();
});

test('onSubmittedWorkDone fence allows early chunk recycling before framesInFlight elapses', async () => {
  const { device, createdBuffers, deferredFences } = createMockGpuDevice({
    enableWorkDoneFence: true,
  });
  const pool = new GpuStagingPool({
    device,
    chunkSize: 1024,
    maxPoolBytes: 4096,
    framesInFlight: 3,
  });

  pool.beginFrame(0);
  const s0 = pool.allocate(128, 4);
  pool.endFrame();

  assert.equal(createdBuffers.length, 1);
  assert.equal(deferredFences.length, 1);

  // Resolve GPU work fence early (before 3 frames elapse).
  deferredFences[0].resolve();
  await deferredFences[0].promise;
  await Promise.resolve();

  // Frame 1 (1 - 0 < 3) still recycles chunk 0 because onSubmittedWorkDone resolved!
  const beginInfo = pool.beginFrame(1);
  assert.equal(beginInfo.recycledCount, 1);

  const s1 = pool.allocate(128, 4);
  assert.equal(s1.buffer, s0.buffer);
  assert.equal(s1.offset, 0);
  assert.equal(createdBuffers.length, 1);
  assert.equal(pool.getStats().bufferReuseHits, 1);

  pool.destroy();
});

test('oversized allocation uses dedicated chunk while preserving active chunk and enforces maxPoolBytes', () => {
  const { device, createdBuffers } = createMockGpuDevice();
  const pool = new GpuStagingPool({
    device,
    chunkSize: 512,
    maxPoolBytes: 2048,
    framesInFlight: 3,
  });

  pool.beginFrame(0);

  // Regular small allocation in standard 512B chunk.
  const small1 = pool.allocate(64, 4);
  assert.equal(small1.offset, 0);

  // Oversized allocation (1024B > 512B chunkSize) gets its own dedicated chunk.
  const big = pool.allocate(1024, 4);
  assert.equal(big.offset, 0);
  assert.equal(big.size, 1024);
  assert.notEqual(big.buffer, small1.buffer);
  assert.equal(big.buffer.size, 1024);

  // Next small allocation still sub-allocates from the preserved standard chunk!
  const small2 = pool.allocate(64, 4);
  assert.equal(small2.buffer, small1.buffer);
  assert.equal(small2.offset, 64);

  assert.equal(createdBuffers.length, 2);
  assert.equal(pool.getStats().allocatedBytes, 512 + 1024);

  // Another 1024B allocation would need 1536 + 1024 = 2560 > 2048 -> must throw!
  assert.throws(
    () => pool.allocate(1024, 4),
    (err) => err instanceof RangeError && /budget exceeded/i.test(err.message)
  );

  // Standard chunk rollover within remaining 512B budget still works!
  const fill = pool.allocate(400, 4);
  assert.notEqual(fill.buffer, small1.buffer);
  assert.equal(pool.getStats().allocatedBytes, 2048);

  // Pool is now at 2048 / 2048 with all chunks active -> next rollover must throw.
  assert.throws(
    () => pool.allocate(200, 4),
    (err) => err instanceof RangeError && /budget exceeded/i.test(err.message)
  );

  pool.destroy();
});

test('stageBufferUpload pads unaligned lengths to 4 bytes and records copyBufferToBuffer (writeBuffer mode)', () => {
  const { device, queueWrites } = createMockGpuDevice();
  const pool = new GpuStagingPool({
    device,
    chunkSize: 1024,
    maxPoolBytes: 4096,
  });

  const encoder = createMockEncoder();
  const dstBuffer = { id: 'gpu-dst-buffer' };

  pool.beginFrame(0);

  // Unaligned 5-byte upload must pad to 8 bytes per WebGPU copyBufferToBuffer spec.
  const unaligned = new Uint8Array([0x11, 0x22, 0x33, 0x44, 0x55]);
  const res1 = pool.stageBufferUpload(encoder, dstBuffer, 16, unaligned);

  assert.equal(res1.offset, 0);
  assert.equal(res1.size, 8);
  assert.equal(res1.srcByteLength, 5);
  assert.equal(queueWrites.length, 1);
  assert.equal(queueWrites[0].bufferOffset, 0);
  assert.equal(queueWrites[0].byteLength, 8);
  assert.deepEqual(
    Array.from(queueWrites[0].data),
    [0x11, 0x22, 0x33, 0x44, 0x55, 0x00, 0x00, 0x00]
  );

  assert.equal(encoder.bufferCopies.length, 1);
  assert.equal(encoder.bufferCopies[0].srcBuffer, res1.buffer);
  assert.equal(encoder.bufferCopies[0].srcOffset, 0);
  assert.equal(encoder.bufferCopies[0].dstBuffer, dstBuffer);
  assert.equal(encoder.bufferCopies[0].dstOffset, 16);
  assert.equal(encoder.bufferCopies[0].size, 8);

  // Aligned Float32Array upload (12 bytes).
  const floats = new Float32Array([1.0, 2.0, 3.0]);
  const res2 = pool.stageBufferUpload(encoder, dstBuffer, 64, floats);
  assert.equal(res2.offset, 8);
  assert.equal(res2.size, 12);
  assert.equal(res2.buffer, res1.buffer);
  assert.equal(encoder.bufferCopies.length, 2);
  assert.equal(encoder.bufferCopies[1].srcOffset, 8);
  assert.equal(encoder.bufferCopies[1].dstOffset, 64);
  assert.equal(encoder.bufferCopies[1].size, 12);
  assert.deepEqual(
    Array.from(new Float32Array(encoder.bufferCopies[1].snapshot.buffer)),
    [1.0, 2.0, 3.0]
  );

  // Reject non-4-byte-aligned dstOffset.
  assert.throws(() => pool.stageBufferUpload(encoder, dstBuffer, 2, unaligned), RangeError);

  assert.equal(pool.getStats().bytesUploaded, 8 + 12);
  pool.destroy();
});

test('mapped mode writes directly to mapped range without queue.writeBuffer and recycles after mapAsync', async () => {
  const { device, createdBuffers, queueWrites, deferredFences } = createMockGpuDevice({
    enableWorkDoneFence: true,
  });
  const pool = new GpuStagingPool({
    device,
    chunkSize: 1024,
    maxPoolBytes: 4096,
    framesInFlight: 2,
    mode: STAGING_MODE.MAPPED,
  });

  const encoder = createMockEncoder();
  const dstBuffer = { id: 'gpu-dst-mapped' };

  pool.beginFrame(0);
  const src = new Uint8Array([1, 2, 3, 4, 5, 6]); // pads to 8 bytes
  const res = pool.stageBufferUpload(encoder, dstBuffer, 0, src);

  assert.equal(queueWrites.length, 0, 'mapped mode must not call device.queue.writeBuffer');
  assert.equal(createdBuffers.length, 1);
  assert.equal(createdBuffers[0].usage, BUFFER_USAGE.MAP_WRITE | BUFFER_USAGE.COPY_SRC);
  assert.equal(createdBuffers[0].mappedAtCreation, true);
  assert.equal(encoder.bufferCopies.length, 1);
  assert.deepEqual(
    Array.from(encoder.bufferCopies[0].snapshot),
    [1, 2, 3, 4, 5, 6, 0, 0]
  );

  // Direct allocate() exposes a live Uint8Array view in mapped mode.
  const direct = pool.allocate(16, 16);
  assert.ok(direct.view instanceof Uint8Array);
  assert.equal(direct.view.byteLength, 16);
  assert.equal(direct.offset, 16);

  // endFrame() unmaps active chunks before submit / retirement.
  pool.endFrame();
  assert.equal(createdBuffers[0].unmapCount, 1);
  assert.equal(createdBuffers[0]._mapped, false);

  // Resolve GPU work fence -> triggers mapAsync(GPUMapMode.WRITE).
  deferredFences[0].resolve();
  await deferredFences[0].promise;
  await pool.waitForPendingMaps();

  assert.equal(createdBuffers[0].mapAsyncCount, 1);
  assert.equal(createdBuffers[0]._mapped, true);

  // Frame 1 reuses chunk 0 with zero new buffer allocations!
  pool.beginFrame(1);
  const res2 = pool.stageBufferUpload(encoder, dstBuffer, 8, new Uint8Array([9, 8, 7, 6]));
  assert.equal(res2.buffer, res.buffer);
  assert.equal(res2.offset, 0);
  assert.equal(createdBuffers.length, 1);
  assert.equal(pool.getStats().bufferReuseHits, 1);
  assert.deepEqual(Array.from(encoder.bufferCopies[1].snapshot), [9, 8, 7, 6]);

  pool.destroy();
});

test('stageTextureUpload enforces 256-byte bytesPerRow alignment and records copyBufferToTexture', () => {
  const { device, queueWrites } = createMockGpuDevice();
  const pool = new GpuStagingPool({
    device,
    chunkSize: 4096,
    maxPoolBytes: 16384,
  });

  const encoder = createMockEncoder();
  const dstTexture = { id: 'tex-albedo' };

  pool.beginFrame(0);

  // 4x3 RGBA8 texture: unaligned bytesPerRow = 16 -> must pad to 256 bytes/row!
  const width = 4;
  const height = 3;
  const unalignedBytesPerRow = width * 4; // 16
  const src = new Uint8Array(unalignedBytesPerRow * height);
  for (let row = 0; row < height; row += 1) {
    for (let col = 0; col < unalignedBytesPerRow; col += 1) {
      src[row * unalignedBytesPerRow + col] = (row + 1) * 16 + col;
    }
  }

  const info = pool.stageTextureUpload(encoder, dstTexture, src, {
    width,
    height,
    bytesPerRow: unalignedBytesPerRow,
    mipLevel: 1,
    origin: { x: 4, y: 8, z: 0 },
  });

  assert.equal(info.bytesPerRow, 256);
  assert.equal(info.unalignedBytesPerRow, 16);
  assert.equal(info.rowsPerImage, 3);
  assert.equal(info.size, 256 * 3);
  assert.equal(info.offset, 0);

  assert.equal(queueWrites.length, 1);
  assert.equal(encoder.textureCopies.length, 1);

  const copy = encoder.textureCopies[0];
  assert.equal(copy.source.buffer, info.buffer);
  assert.equal(copy.source.offset, 0);
  assert.equal(copy.source.bytesPerRow, 256);
  assert.equal(copy.source.rowsPerImage, 3);
  assert.equal(copy.destination.texture, dstTexture);
  assert.equal(copy.destination.mipLevel, 1);
  assert.deepEqual(copy.destination.origin, { x: 4, y: 8, z: 0 });
  assert.deepEqual(copy.copySize, { width: 4, height: 3, depthOrArrayLayers: 1 });

  // Verify every row's texels and zero-padded tail bytes [16..255].
  for (let row = 0; row < height; row += 1) {
    const rowBase = row * 256;
    for (let col = 0; col < 16; col += 1) {
      assert.equal(copy.snapshot[rowBase + col], (row + 1) * 16 + col);
    }
    for (let pad = 16; pad < 256; pad += 1) {
      assert.equal(copy.snapshot[rowBase + pad], 0);
    }
  }

  // Multi-layer / BCn texture with explicit rowsPerImage (e.g., 8x8 BC7 -> 2x2 blocks, 32B/row, 2 layers).
  const bcRowsPerImage = 2;
  const bcLayers = 2;
  const bcBytesPerRow = 32;
  const bcSrc = new Uint8Array(bcBytesPerRow * bcRowsPerImage * bcLayers).fill(0x5a);
  const bcInfo = pool.stageTextureUpload(encoder, dstTexture, bcSrc, {
    width: 8,
    height: 8,
    depthOrLayers: bcLayers,
    bytesPerRow: bcBytesPerRow,
    rowsPerImage: bcRowsPerImage,
  });

  assert.equal(bcInfo.offset % 256, 0);
  assert.equal(bcInfo.bytesPerRow, 256);
  assert.equal(bcInfo.size, 256 * bcRowsPerImage * bcLayers);
  assert.equal(encoder.textureCopies[1].copySize.depthOrArrayLayers, 2);
  assert.equal(encoder.textureCopies[1].source.rowsPerImage, 2);

  assert.equal(pool.getStats().bytesUploaded, 256 * 3 + 256 * 2 * 2);
  pool.destroy();
});

test('stageTextureUpload works in mapped mode without calling queue.writeBuffer', () => {
  const { device, queueWrites } = createMockGpuDevice();
  const pool = new GpuStagingPool({
    device,
    chunkSize: 4096,
    maxPoolBytes: 8192,
    mode: STAGING_MODE.MAPPED,
  });

  const encoder = createMockEncoder();
  const dstTexture = { id: 'tex-mapped' };

  pool.beginFrame(0);
  const src = new Uint8Array(8 * 2); // 2 rows of 8 bytes
  src.set([1, 2, 3, 4, 5, 6, 7, 8], 0);
  src.set([9, 10, 11, 12, 13, 14, 15, 16], 8);

  const info = pool.stageTextureUpload(encoder, dstTexture, src, {
    width: 2,
    height: 2,
    bytesPerRow: 8,
  });

  assert.equal(queueWrites.length, 0);
  assert.equal(info.bytesPerRow, 256);
  assert.equal(info.size, 512);
  assert.deepEqual(Array.from(encoder.textureCopies[0].snapshot.subarray(0, 8)), [1, 2, 3, 4, 5, 6, 7, 8]);
  assert.deepEqual(
    Array.from(encoder.textureCopies[0].snapshot.subarray(256, 264)),
    [9, 10, 11, 12, 13, 14, 15, 16]
  );
  assert.equal(encoder.textureCopies[0].snapshot[20], 0);

  pool.destroy();
});

test('getStats reports accurate counts and destroy is idempotent', () => {
  const { device, createdBuffers } = createMockGpuDevice();
  const pool = new GpuStagingPool({
    device,
    chunkSize: 512,
    maxPoolBytes: 4096,
    framesInFlight: 2,
  });

  // Frame 0 uses 2 chunks.
  pool.beginFrame(0);
  pool.allocate(400, 4);
  pool.allocate(400, 4);
  pool.endFrame();

  // Frame 1 uses 1 chunk.
  pool.beginFrame(1);
  pool.allocate(200, 4);
  pool.endFrame();

  // Frame 2 recycles frame 0's 2 chunks, uses 1 of them.
  pool.beginFrame(2);
  pool.allocate(100, 4);

  const midStats = pool.getStats();
  assert.equal(midStats.totalChunksAllocated, 3);
  assert.equal(midStats.activeChunks, 1);
  assert.equal(midStats.recycledChunks, 1);
  assert.equal(midStats.inFlightChunks, 1);
  assert.equal(midStats.allocatedBytes, 1536);
  assert.equal(midStats.peakAllocatedBytes, 1536);
  assert.equal(midStats.allocationCount, 4);
  assert.equal(midStats.bufferReuseHits, 1);
  assert.equal(midStats.bufferReuseMisses, 3);

  // First destroy destroys all 3 GPUBuffers once.
  pool.destroy();
  for (const buf of createdBuffers) {
    assert.equal(buf.destroyCount, 1);
  }

  const afterDestroy = pool.getStats();
  assert.equal(afterDestroy.activeChunks, 0);
  assert.equal(afterDestroy.recycledChunks, 0);
  assert.equal(afterDestroy.inFlightChunks, 0);
  assert.equal(afterDestroy.allocatedBytes, 0);
  assert.equal(afterDestroy.peakAllocatedBytes, 1536);
  assert.equal(afterDestroy.totalChunksAllocated, 3);

  // Second destroy is a safe no-op (no double-destroy).
  pool.destroy();
  for (const buf of createdBuffers) {
    assert.equal(buf.destroyCount, 1);
  }

  // Operations after destroy throw cleanly.
  assert.throws(() => pool.beginFrame(3), /destroyed/i);
  assert.throws(() => pool.allocate(16, 4), /destroyed/i);
  assert.throws(() => pool.endFrame(), /destroyed/i);
});

test('auto-incrementing beginFrame, empty uploads, aligned textures, and recycled eviction within budget', () => {
  const { device, createdBuffers } = createMockGpuDevice();
  const pool = new GpuStagingPool({
    device,
    chunkSize: 512,
    maxPoolBytes: 1024,
    framesInFlight: 2,
  });

  const encoder = createMockEncoder();
  const dstBuffer = { id: 'dst-auto' };
  const dstTexture = { id: 'tex-aligned' };

  // Auto-incrementing frame indices (0, 1, 2, 3) with framesInFlight = 2.
  for (let i = 0; i < 4; i += 1) {
    const info = pool.beginFrame();
    assert.equal(info.frameIndex, i);
    pool.stageBufferUpload(encoder, dstBuffer, 0, new Uint8Array(64));
    pool.endFrame();
  }
  assert.equal(createdBuffers.length, 2);
  assert.equal(pool.getStats().bufferReuseHits, 2);
  assert.equal(pool.getStats().bufferReuseMisses, 2);

  // Empty buffer upload is a safe no-op.
  pool.beginFrame(); // frame 4 (recycles frame 2)
  const emptyRes = pool.stageBufferUpload(encoder, dstBuffer, 0, new Uint8Array(0));
  assert.equal(emptyRes.size, 0);
  assert.equal(emptyRes.buffer, null);

  // Already 256-byte aligned texture upload takes zero-padding fast path.
  const alignedTexBytes = new Uint8Array(256 * 2).fill(0x7e);
  const texInfo = pool.stageTextureUpload(encoder, dstTexture, alignedTexBytes, {
    width: 64,
    height: 2,
    bytesPerRow: 256,
  });
  assert.equal(texInfo.bytesPerRow, 256);
  assert.equal(texInfo.size, 512);
  pool.endFrame();

  // Advance 2 frames so all chunks are idle/recycled, then allocate an oversized 768B chunk:
  // pool evicts idle recycled 512B chunks to stay within maxPoolBytes (1024).
  pool.beginFrame(); // frame 5
  pool.endFrame();
  pool.beginFrame(); // frame 6 (frame 4's chunk recycled; both 512B chunks idle)
  const oversized = pool.allocate(768, 4);
  assert.equal(oversized.size, 768);
  assert.ok(pool.getStats().allocatedBytes <= 1024);

  // Failed budget allocation must not inflate bufferReuseMisses.
  const missesBefore = pool.getStats().bufferReuseMisses;
  assert.throws(() => pool.allocate(1024, 4), RangeError);
  assert.equal(pool.getStats().bufferReuseMisses, missesBefore);

  pool.destroy();
});

// ---------------------------------------------------------------------------
// G46 Hardening: 4-byte alignment, SharedArrayBuffer-safe uploads,
// double-release / use-after-destroy guards, and Frame-1 burst recycling.
// ---------------------------------------------------------------------------

function createStrictWebGpuDevice(options = {}) {
  const mock = createMockGpuDevice(options);
  const origWriteBuffer = mock.device.queue.writeBuffer;
  const rawWriteArgs = [];

  mock.device.queue.writeBuffer = (buffer, bufferOffset, data) => {
    if (bufferOffset % 4 !== 0) {
      throw new RangeError(`WebGPU writeBuffer bufferOffset (${bufferOffset}) must be 4-byte aligned`);
    }
    if (typeof SharedArrayBuffer !== 'undefined') {
      if (data instanceof SharedArrayBuffer) {
        throw new TypeError('WebGPU writeBuffer rejected SharedArrayBuffer directly');
      }
      if (ArrayBuffer.isView(data) && data.buffer instanceof SharedArrayBuffer) {
        throw new TypeError('WebGPU writeBuffer rejected SharedArrayBuffer-backed view');
      }
    }
    const byteLength = ArrayBuffer.isView(data) ? data.byteLength : data.byteLength;
    if (byteLength % 4 !== 0) {
      throw new RangeError(`WebGPU writeBuffer size (${byteLength}) must be 4-byte aligned`);
    }
    rawWriteArgs.push({ buffer, bufferOffset, data });
    return origWriteBuffer(buffer, bufferOffset, data);
  };

  return { ...mock, rawWriteArgs };
}

test('G46: alignBufferSize and 4-byte staging allocation alignment track alignPadCount accurately', () => {
  assert.equal(alignBufferSize(0), 4);
  assert.equal(alignBufferSize(1), 4);
  assert.equal(alignBufferSize(2), 4);
  assert.equal(alignBufferSize(3), 4);
  assert.equal(alignBufferSize(4), 4);
  assert.equal(alignBufferSize(5), 8);
  assert.equal(alignBufferSize(7), 8);
  assert.equal(alignBufferSize(8), 8);
  assert.equal(alignBufferSize(0x100000001), 0x100000004);
  assert.throws(() => alignBufferSize(-1), RangeError);
  assert.throws(() => alignBufferSize(1.25), RangeError);

  const { device, createdBuffers } = createStrictWebGpuDevice();
  // Unaligned chunkSize (513) rounds up to 516 and increments alignPadCount once.
  const pool = new GpuStagingPool({
    device,
    chunkSize: 513,
    maxPoolBytes: 4096,
    framesInFlight: 2,
  });

  assert.equal(pool.chunkSize, 516);
  assert.equal(pool.getStats().alignPadCount, 1);

  pool.beginFrame(0);

  // Unaligned sizes + sub-4-byte alignments still produce 4-byte aligned offsets and reservations.
  const a1 = pool.allocate(1, 1);
  assert.equal(a1.offset, 0);
  assert.equal(a1.size, 1);
  assert.equal(a1.alignedSize, 4);
  assert.equal(createdBuffers[0].size, 516);

  const a2 = pool.allocate(3, 2);
  assert.equal(a2.offset, 4);
  assert.equal(a2.offset % 4, 0);
  assert.equal(a2.size, 3);
  assert.equal(a2.alignedSize, 4);

  // alignment=3 -> lcm(3, 4) = 12, so next offset is 12 (divisible by both 3 and 4).
  const a3 = pool.allocate(5, 3);
  assert.equal(a3.offset, 12);
  assert.equal(a3.offset % 4, 0);
  assert.equal(a3.offset % 3, 0);
  assert.equal(a3.size, 5);
  assert.equal(a3.alignedSize, 8);

  // Already 4-byte aligned allocation does not increment alignPadCount.
  const a4 = pool.allocate(16, 4);
  assert.equal(a4.offset, 20);
  assert.equal(a4.size, 16);
  assert.equal(a4.alignedSize, 16);

  // Oversized unaligned allocation (517 > 516) creates a 520B GPUBuffer and pads once.
  const big = pool.allocate(517, 4);
  assert.equal(big.offset, 0);
  assert.equal(big.size, 517);
  assert.equal(big.alignedSize, 520);
  assert.equal(big.buffer.size, 520);

  const stats = pool.getStats();
  // 1 (chunkSize 513->516) + 3 (a1, a2, a3) + 1 (big 517->520) = 5
  assert.equal(stats.alignPadCount, 5);
  assert.equal(stats.activeBuffers, 2);
  assert.equal(stats.activeChunks, 2);
  assert.equal(stats.recycledBuffers, 0);
  assert.equal(stats.recycledChunks, 0);

  pool.destroy();
});

test('G46: sanitizeWriteBufferPayload copies SharedArrayBuffer views and zero-pads non-4B lengths', () => {
  // Regular ArrayBuffer, 4B aligned -> zero-copy fast path.
  const alignedAb = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
  const r1 = sanitizeWriteBufferPayload(alignedAb);
  assert.equal(r1.bytes, alignedAb);
  assert.equal(r1.padded, false);
  assert.equal(r1.copiedFromSab, false);
  assert.equal(r1.alignedByteLength, 8);

  // Regular ArrayBuffer, unaligned -> zero-padded copy.
  const oddAb = new Uint8Array([0xaa, 0xbb, 0xcc]);
  const r2 = sanitizeWriteBufferPayload(oddAb);
  assert.notEqual(r2.bytes, oddAb);
  assert.ok(r2.bytes.buffer instanceof ArrayBuffer);
  assert.equal(r2.padded, true);
  assert.equal(r2.copiedFromSab, false);
  assert.deepEqual(Array.from(r2.bytes), [0xaa, 0xbb, 0xcc, 0x00]);

  // SharedArrayBuffer, 4B aligned -> copied into regular ArrayBuffer.
  const sab = new SharedArrayBuffer(16);
  const sabBytes = new Uint8Array(sab);
  for (let i = 0; i < 16; i += 1) sabBytes[i] = i + 1;
  assert.equal(isSharedBuffer(sab), true);
  assert.equal(isSharedBuffer(alignedAb.buffer), false);

  const r3 = sanitizeWriteBufferPayload(new Uint8Array(sab, 4, 8));
  assert.ok(r3.bytes.buffer instanceof ArrayBuffer);
  assert.equal(isSharedBuffer(r3.bytes.buffer), false);
  assert.equal(r3.padded, false);
  assert.equal(r3.copiedFromSab, true);
  assert.deepEqual(Array.from(r3.bytes), [5, 6, 7, 8, 9, 10, 11, 12]);

  // SharedArrayBuffer, unaligned DataView at non-zero byteOffset -> copied + zero-padded.
  const dv = new DataView(sab, 3, 5);
  const r4 = sanitizeWriteBufferPayload(dv);
  assert.ok(r4.bytes.buffer instanceof ArrayBuffer);
  assert.equal(isSharedBuffer(r4.bytes.buffer), false);
  assert.equal(r4.padded, true);
  assert.equal(r4.copiedFromSab, true);
  assert.equal(r4.originalByteLength, 5);
  assert.equal(r4.alignedByteLength, 8);
  assert.deepEqual(Array.from(r4.bytes), [4, 5, 6, 7, 8, 0, 0, 0]);

  // Empty SharedArrayBuffer -> empty regular Uint8Array.
  const r5 = sanitizeWriteBufferPayload(new SharedArrayBuffer(0));
  assert.equal(r5.bytes.byteLength, 0);
  assert.ok(r5.bytes.buffer instanceof ArrayBuffer);
  assert.equal(r5.padded, false);
  assert.equal(r5.copiedFromSab, false);
});

test('G46: stageBufferUpload & stageTextureUpload safely copy SharedArrayBuffer views before queue.writeBuffer', () => {
  const { device, queueWrites, rawWriteArgs } = createStrictWebGpuDevice();
  const pool = new GpuStagingPool({
    device,
    chunkSize: 4096,
    maxPoolBytes: 16384,
    framesInFlight: 2,
  });

  const encoder = createMockEncoder();
  const dstBuffer = { id: 'dst-sab' };
  const dstTexture = { id: 'tex-sab' };

  pool.beginFrame(0);

  // 1. Aligned SharedArrayBuffer-backed Float32Array upload.
  const sab1 = new SharedArrayBuffer(16);
  const f32 = new Float32Array(sab1, 4, 3);
  f32.set([1.5, 2.5, 3.5]);
  const res1 = pool.stageBufferUpload(encoder, dstBuffer, 0, f32);
  // Immediately mutate SAB to verify copy isolation!
  f32.set([999, 999, 999]);

  assert.equal(res1.size, 12);
  assert.equal(res1.srcByteLength, 12);
  assert.ok(rawWriteArgs[0].data.buffer instanceof ArrayBuffer);
  assert.equal(isSharedBuffer(rawWriteArgs[0].data.buffer), false);
  assert.deepEqual(Array.from(new Float32Array(queueWrites[0].data.buffer)), [1.5, 2.5, 3.5]);

  // 2. Unaligned SharedArrayBuffer-backed Uint8Array upload (5 bytes -> 8 bytes zero-padded).
  const sab2 = new SharedArrayBuffer(10);
  const u8 = new Uint8Array(sab2, 2, 5);
  u8.set([0xde, 0xad, 0xbe, 0xef, 0x42]);
  const res2 = pool.stageBufferUpload(encoder, dstBuffer, 16, u8);
  u8.fill(0xff);

  assert.equal(res2.size, 8);
  assert.equal(res2.srcByteLength, 5);
  assert.ok(rawWriteArgs[1].data.buffer instanceof ArrayBuffer);
  assert.deepEqual(
    Array.from(encoder.bufferCopies[1].snapshot),
    [0xde, 0xad, 0xbe, 0xef, 0x42, 0x00, 0x00, 0x00]
  );

  // 3. Direct SharedArrayBuffer (not wrapped in a TypedArray), unaligned 3 bytes -> 4 bytes.
  const sab3 = new SharedArrayBuffer(3);
  new Uint8Array(sab3).set([7, 8, 9]);
  const res3 = pool.stageBufferUpload(encoder, dstBuffer, 32, sab3);
  assert.equal(res3.size, 4);
  assert.deepEqual(Array.from(encoder.bufferCopies[2].snapshot), [7, 8, 9, 0]);

  // 4. SharedArrayBuffer-backed texture upload (both aligned 256B/row and unaligned 16B/row).
  const sabTexAligned = new SharedArrayBuffer(256 * 2);
  new Uint8Array(sabTexAligned).fill(0x3c);
  const texRes1 = pool.stageTextureUpload(encoder, dstTexture, new Uint8Array(sabTexAligned), {
    width: 64,
    height: 2,
    bytesPerRow: 256,
  });
  assert.equal(texRes1.size, 512);
  assert.ok(rawWriteArgs[3].data.buffer instanceof ArrayBuffer);
  assert.equal(isSharedBuffer(rawWriteArgs[3].data.buffer), false);

  const sabTexUnaligned = new SharedArrayBuffer(16 * 2);
  new Uint8Array(sabTexUnaligned).fill(0x5f);
  const texRes2 = pool.stageTextureUpload(encoder, dstTexture, new Uint8Array(sabTexUnaligned), {
    width: 4,
    height: 2,
    bytesPerRow: 16,
  });
  assert.equal(texRes2.size, 512);
  assert.ok(rawWriteArgs[4].data.buffer instanceof ArrayBuffer);
  assert.equal(encoder.textureCopies[1].snapshot[0], 0x5f);
  assert.equal(encoder.textureCopies[1].snapshot[16], 0);

  const stats = pool.getStats();
  assert.equal(stats.sabCopyCount, 5);
  assert.equal(stats.alignPadCount, 2); // res2 (5->8) and res3 (3->4)
  assert.equal(stats.activeBuffers, 1);

  pool.destroy();
});

test('G46: double-release, double-endFrame, double-destroy, and use-after-destroy are guarded safely', async () => {
  const { device, createdBuffers } = createStrictWebGpuDevice({ enableWorkDoneFence: true });
  const pool = new GpuStagingPool({
    device,
    chunkSize: 512,
    maxPoolBytes: 2048,
    framesInFlight: 2,
    mode: STAGING_MODE.MAPPED,
  });

  const encoder = createMockEncoder();
  const dstBuffer = { id: 'dst-guard' };
  const dstTexture = { id: 'tex-guard' };

  pool.beginFrame(0);
  const slice = pool.allocate(5, 4);
  assert.equal(slice.size, 5);
  assert.equal(slice.alignedSize, 8);
  assert.equal(pool.getStats().activeBuffers, 1);
  assert.equal(pool.getStats().recycledBuffers, 0);

  // Calling prepareSubmit twice must not double-unmap.
  pool.prepareSubmit();
  pool.prepareSubmit();
  assert.equal(createdBuffers[0].unmapCount, 1);

  // Explicit release retires the chunk once; second release is a safe no-op!
  assert.equal(pool.release(slice), true);
  assert.equal(pool.getStats().activeBuffers, 0);
  assert.equal(pool.getStats().inFlightChunks, 1);
  assert.equal(pool.release(slice), false);
  assert.equal(pool.release(slice.buffer), false);
  assert.equal(pool.getStats().inFlightChunks, 1);
  assert.equal(pool.getStats().doubleReleaseIgnored, 2);

  // Invalid / unknown release arguments return false without throwing.
  assert.equal(pool.release(null), false);
  assert.equal(pool.release(undefined), false);
  assert.equal(pool.release({ buffer: { id: 'unknown' } }), false);

  // Double endFrame() without intervening beginFrame() is guarded and idempotent.
  const firstEnd = pool.endFrame();
  assert.equal(firstEnd.chunksRetired, 0);
  const secondEnd = pool.endFrame();
  assert.equal(secondEnd.chunksRetired, 0);
  assert.equal(pool.getStats().doubleReleaseIgnored, 3);

  // Destroy twice: single GPUBuffer.destroy(), increments doubleReleaseIgnored on second call.
  pool.destroy();
  assert.equal(createdBuffers[0].destroyCount, 1);
  assert.equal(pool.getStats().destroyed, true);
  assert.equal(pool.getStats().activeBuffers, 0);
  assert.equal(pool.getStats().recycledBuffers, 0);

  pool.destroy();
  assert.equal(createdBuffers[0].destroyCount, 1);
  assert.equal(pool.getStats().doubleReleaseIgnored, 4);

  // waitForPendingMaps() resolves cleanly after destroy without hanging.
  await pool.waitForPendingMaps();

  // All mutating methods throw cleanly after destroy.
  assert.throws(() => pool.beginFrame(1), /destroyed/i);
  assert.throws(() => pool.prepareSubmit(), /destroyed/i);
  assert.throws(() => pool.endFrame(), /destroyed/i);
  assert.throws(() => pool.allocate(16, 4), /destroyed/i);
  assert.throws(() => pool.release(slice), /destroyed/i);
  assert.throws(() => pool.recycleCompleted(), /destroyed/i);
  assert.throws(() => pool.stageBufferUpload(encoder, dstBuffer, 0, new Uint8Array(4)), /destroyed/i);
  assert.throws(
    () => pool.stageTextureUpload(encoder, dstTexture, new Uint8Array(16), { width: 2, height: 2, bytesPerRow: 8 }),
    /destroyed/i
  );
  assert.throws(() => pool.writeBuffer(encoder, dstBuffer, 0, new Uint8Array(4)), /destroyed/i);
  assert.throws(
    () => pool.writeTexture(encoder, dstTexture, new Uint8Array(16), { width: 2, height: 2, bytesPerRow: 8 }),
    /destroyed/i
  );
});

test('G46: Frame-1 burst recycling reclaims GPU-completed chunks on demand and evicts oversized burst chunks first', async () => {
  const { device, createdBuffers, deferredFences } = createStrictWebGpuDevice({
    enableWorkDoneFence: true,
  });
  const pool = new GpuStagingPool({
    device,
    chunkSize: 512,
    maxPoolBytes: 2560, // fits 1 standard (512B) + 1 oversized (1024B) + 2 standard (1024B) = 2560B
    framesInFlight: 3,
  });

  const encoder = createMockEncoder();
  const dstBuffer = { id: 'dst-burst' };

  // Frame 1 burst: 2 standard chunks + 1 oversized chunk (1024B) using SAB-backed unaligned data.
  pool.beginFrame(1);
  const sabChunk1 = new SharedArrayBuffer(401);
  new Uint8Array(sabChunk1).fill(0x11);
  pool.stageBufferUpload(encoder, dstBuffer, 0, new Uint8Array(sabChunk1)); // chunk 1 (512B, 404B used)

  const sabChunk2 = new SharedArrayBuffer(303);
  new Uint8Array(sabChunk2).fill(0x22);
  pool.stageBufferUpload(encoder, dstBuffer, 512, new Uint8Array(sabChunk2)); // rolls over -> chunk 2 (512B)

  const sabOversized = new SharedArrayBuffer(1021);
  new Uint8Array(sabOversized).fill(0x33);
  pool.stageBufferUpload(encoder, dstBuffer, 1024, new Uint8Array(sabOversized)); // oversized -> chunk 3 (1024B)

  assert.equal(pool.getStats().activeBuffers, 3);
  assert.equal(pool.getStats().allocatedBytes, 512 + 512 + 1024);
  assert.equal(pool.getStats().sabCopyCount, 3);
  assert.equal(pool.getStats().alignPadCount, 3);

  const end1 = pool.endFrame();
  assert.equal(end1.chunksRetired, 3);
  assert.equal(pool.getStats().activeBuffers, 0);
  assert.equal(pool.getStats().inFlightChunks, 3);

  // Frame 2 starts before fence resolves -> allocates 1 fresh standard chunk (total = 2560B, at budget cap).
  pool.beginFrame(2);
  pool.stageBufferUpload(encoder, dstBuffer, 0, new Uint8Array(256));
  assert.equal(createdBuffers.length, 4);
  assert.equal(pool.getStats().allocatedBytes, 2560);

  // Now Frame 1's GPU fence resolves mid-Frame 2!
  deferredFences[0].resolve();
  await deferredFences[0].promise;
  await Promise.resolve();

  // Another allocation in Frame 2 (rolling over chunk 4) reclaims Frame 1's GPU-completed chunks
  // on demand instead of throwing budget-exceeded or creating a 5th buffer!
  pool.stageBufferUpload(encoder, dstBuffer, 256, new Uint8Array(400));
  assert.equal(createdBuffers.length, 4, 'must recycle GPU-completed Frame-1 burst chunks without extra createBuffer');
  assert.equal(pool.getStats().activeBuffers, 2);
  assert.equal(pool.getStats().recycledBuffers, 2); // 1 standard + 1 oversized left warm
  assert.equal(pool.getStats().bufferReuseHits, 1);

  // Requesting a new 1024B oversized chunk while at budget (2560B) reuses or evicts the idle
  // oversized burst chunk cleanly while preserving standard chunks!
  const oversizedReuse = pool.allocate(1024, 4);
  assert.equal(oversizedReuse.buffer, createdBuffers[2]); // reused the 1024B oversized chunk!
  assert.equal(createdBuffers.length, 4);
  pool.endFrame();

  // Advance 3 frames to recycle everything, then request a 1280B oversized chunk:
  // pool must evict the idle 1024B oversized chunk FIRST (keeping standard 512B chunks intact).
  pool.beginFrame(5);
  assert.equal(pool.getStats().recycledBuffers, 4);
  const huge = pool.allocate(1280, 4);
  assert.equal(huge.size, 1280);
  assert.equal(createdBuffers[2].destroyCount, 1, 'idle oversized chunk must be evicted before standard chunks');
  assert.equal(createdBuffers[0].destroyCount, 0, 'warm standard chunk should stay alive');
  assert.ok(pool.getStats().allocatedBytes <= 2560);

  pool.destroy();
});

test('G46: classic-worker importScripts IIFE isolation attaches PlayGta6StagingPool cleanly', () => {
  const source = fs.readFileSync(
    new URL('../web/workers/staging_pool.js', import.meta.url),
    'utf8'
  );
  const sandbox = {
    Uint8Array,
    ArrayBuffer,
    SharedArrayBuffer,
    DataView,
    Float32Array,
    Math,
    Number,
    Object,
    Set,
    Map,
    Array,
    Promise,
    Error,
    TypeError,
    RangeError,
    JSON,
    Infinity,
  };
  sandbox.self = sandbox;
  sandbox.globalThis = sandbox;
  const ctx = vm.createContext(sandbox);

  // Running the script twice (simulating duplicate importScripts or sibling scripts with
  // matching helper names) must NOT throw SyntaxError / Redeclaration errors.
  vm.runInContext(source, ctx, { filename: 'staging_pool.js' });
  vm.runInContext(source, ctx, { filename: 'staging_pool.js' });

  assert.ok(ctx.PlayGta6StagingPool);
  assert.equal(typeof ctx.PlayGta6StagingPool.GpuStagingPool, 'function');
  assert.equal(typeof ctx.PlayGta6StagingPool.alignBufferSize, 'function');
  assert.equal(typeof ctx.PlayGta6StagingPool.sanitizeWriteBufferPayload, 'function');
  // Internal constants must stay isolated inside the IIFE.
  assert.equal(ctx.DEFAULT_CHUNK_SIZE, undefined);
  assert.equal(ctx.WEBGPU_BUFFER_COPY_ALIGNMENT, undefined);
});

