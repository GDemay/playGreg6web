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
  nextPowerOfTwo,
  buildPowerOfTwoBuckets,
  selectPowerOfTwoBucket,
  selectLargeSlabBucket,
  isSharedBuffer,
  sanitizeWriteBufferPayload,
  GpuStagingPool,
  DEFAULT_FRAMES_IN_FLIGHT,
  DEFAULT_MAX_FRAMES_IN_FLIGHT,
  DEFAULT_IDLE_SHRINK_FRAMES,
  DEFAULT_LARGE_SLAB_THRESHOLD,
  DEFAULT_MAX_LARGE_SLABS,
  DEFAULT_POW2_MIN_SLAB_BYTES,
  DEFAULT_POW2_MAX_BUCKET_BYTES,
  DEFAULT_LARGE_SLAB_BUCKETS,
  POW2_LARGE_SLAB_BUCKETS,
  BUCKET_STRATEGY,
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

// ---------------------------------------------------------------------------
// W126: Idle chunk reclamation (`reclaimIdleChunks`, `minWarmChunks`,
// `idleEvictFrames`) and peak VRAM / oversize telemetry.
// ---------------------------------------------------------------------------

test('W126: burst allocation updates peakBytesAllocated and oversizeAllocations without inflating on failure', () => {
  const { device, createdBuffers } = createMockGpuDevice();
  const pool = new GpuStagingPool({
    device,
    chunkSize: 512,
    maxPoolBytes: 2560,
    framesInFlight: 2,
  });

  // Initial stats before any allocation.
  const initStats = pool.getStats();
  assert.equal(initStats.peakBytesAllocated, 0);
  assert.equal(initStats.peakAllocatedBytes, 0);
  assert.equal(initStats.reclaimedChunks, 0);
  assert.equal(initStats.reclaimedBytes, 0);
  assert.equal(initStats.oversizeAllocations, 0);
  assert.equal(pool.minWarmChunks, 1);
  assert.equal(pool.idleEvictFrames, 0);

  pool.beginFrame(0);
  pool.allocate(400, 4); // chunk 1 (512B)
  pool.allocate(400, 4); // rollover -> chunk 2 (512B)
  pool.allocate(1024, 4); // oversized -> chunk 3 (1024B)

  let stats = pool.getStats();
  assert.equal(createdBuffers.length, 3);
  assert.equal(stats.allocatedBytes, 2048);
  assert.equal(stats.peakBytesAllocated, 2048);
  assert.equal(stats.peakAllocatedBytes, 2048);
  assert.equal(stats.oversizeAllocations, 1);

  // Over-budget oversized allocation must throw and NOT increment oversizeAllocations or peakBytesAllocated.
  assert.throws(() => pool.allocate(1024, 4), RangeError);
  stats = pool.getStats();
  assert.equal(stats.oversizeAllocations, 1);
  assert.equal(stats.peakBytesAllocated, 2048);
  pool.endFrame();

  // Recycle all 3 chunks at frame 2 and reuse the oversized chunk -> increments oversizeAllocations again.
  pool.beginFrame(2);
  const reusedBig = pool.allocate(1024, 4);
  assert.equal(reusedBig.buffer, createdBuffers[2]);
  stats = pool.getStats();
  assert.equal(stats.oversizeAllocations, 2);
  assert.equal(stats.allocatedBytes, 2048);
  assert.equal(stats.peakBytesAllocated, 2048);
  pool.endFrame();

  // destroy() clears live bytes while keeping peakBytesAllocated intact.
  pool.destroy();
  const destroyedStats = pool.getStats();
  assert.equal(destroyedStats.allocatedBytes, 0);
  assert.equal(destroyedStats.peakBytesAllocated, 2048);
  assert.equal(destroyedStats.oversizeAllocations, 2);
});

test('W126: reclaimIdleChunks destroys idle chunks after N frames, prefers oversized first, and preserves minWarmChunks', () => {
  const { device, createdBuffers } = createMockGpuDevice();
  const pool = new GpuStagingPool({
    device,
    chunkSize: 512,
    maxPoolBytes: 4096,
    framesInFlight: 2,
    minWarmChunks: 1,
  });

  // Frame 0 burst: 2 standard (512B each) + 1 oversized (1024B) = 2048B.
  pool.beginFrame(0);
  pool.allocate(400, 4); // chunk 0 (512B, lastUsedFrame = 0)
  pool.allocate(400, 4); // chunk 1 (512B, lastUsedFrame = 0)
  pool.allocate(1024, 4); // chunk 2 (1024B oversized, lastUsedFrame = 0)
  pool.endFrame();

  // Frame 2 recycles all 3 chunks; idle age is 2 frames (< 5).
  pool.beginFrame(2);
  assert.equal(pool.getStats().recycledChunks, 3);
  assert.deepEqual(pool.reclaimIdleChunks(5), { reclaimedChunks: 0, reclaimedBytes: 0 });
  // No-arg reclaimIdleChunks is a safe no-op when idleEvictFrames === 0.
  assert.deepEqual(pool.reclaimIdleChunks(), { reclaimedChunks: 0, reclaimedBytes: 0 });
  pool.endFrame();

  // Frame 5: all 3 recycled chunks now have currentFrame - lastUsedFrame = 5 >= 5.
  pool.beginFrame(5);
  const reclaimed = pool.reclaimIdleChunks(5);
  assert.deepEqual(reclaimed, { reclaimedChunks: 2, reclaimedBytes: 1536 });

  // Oversized chunk (createdBuffers[2]) and oldest standard chunk (createdBuffers[0]) destroyed;
  // 1 warm standard chunk (createdBuffers[1]) preserved to satisfy minWarmChunks = 1.
  assert.equal(createdBuffers[2].destroyCount, 1, 'oversized idle chunk must be reclaimed first');
  assert.equal(createdBuffers[0].destroyCount, 1, 'older idle standard chunk reclaimed second');
  assert.equal(createdBuffers[1].destroyCount, 0, 'minWarmChunks warm standard chunk must stay alive');

  const stats = pool.getStats();
  assert.equal(stats.recycledChunks, 1);
  assert.equal(stats.allocatedBytes, 512);
  assert.equal(stats.peakBytesAllocated, 2048);
  assert.equal(stats.reclaimedChunks, 2);
  assert.equal(stats.reclaimedBytes, 1536);

  // Calling reclaimIdleChunks again stays at minWarmChunks = 1.
  assert.deepEqual(pool.reclaimIdleChunks(0), { reclaimedChunks: 0, reclaimedBytes: 0 });
  assert.equal(pool.getStats().recycledChunks, 1);

  // destroy() cleanly destroys the remaining warm chunk without double-destroying reclaimed chunks.
  pool.destroy();
  assert.equal(createdBuffers[0].destroyCount, 1);
  assert.equal(createdBuffers[1].destroyCount, 1);
  assert.equal(createdBuffers[2].destroyCount, 1);
  assert.equal(pool.getStats().allocatedBytes, 0);
  assert.equal(pool.getStats().reclaimedChunks, 2);
  assert.equal(pool.getStats().reclaimedBytes, 1536);
  assert.throws(() => pool.reclaimIdleChunks(1), /destroyed/i);

  // minWarmChunks = 0 allows reclaiming all idle chunks down to 0 bytes.
  const zeroWarmMock = createMockGpuDevice();
  const zeroWarmPool = new GpuStagingPool({
    device: zeroWarmMock.device,
    chunkSize: 512,
    maxPoolBytes: 2048,
    framesInFlight: 1,
    minWarmChunks: 0,
  });
  zeroWarmPool.beginFrame(0);
  zeroWarmPool.allocate(400, 4);
  zeroWarmPool.allocate(400, 4);
  zeroWarmPool.endFrame();

  zeroWarmPool.beginFrame(3);
  assert.deepEqual(zeroWarmPool.reclaimIdleChunks(3), { reclaimedChunks: 2, reclaimedBytes: 1024 });
  assert.equal(zeroWarmPool.getStats().recycledChunks, 0);
  assert.equal(zeroWarmPool.getStats().allocatedBytes, 0);
  assert.equal(zeroWarmPool.getStats().peakBytesAllocated, 1024);
  zeroWarmPool.destroy();

  // Option and argument validation.
  assert.throws(
    () => new GpuStagingPool({ device, minWarmChunks: -1 }),
    RangeError
  );
  assert.throws(
    () => new GpuStagingPool({ device, idleEvictFrames: -2 }),
    RangeError
  );
  const validPool = new GpuStagingPool({ device, chunkSize: 512, maxPoolBytes: 1024 });
  assert.throws(() => validPool.reclaimIdleChunks(-1), RangeError);
  assert.throws(() => validPool.reclaimIdleChunks(1.5), RangeError);
  validPool.destroy();
});

test('W126: idleEvictFrames automatically reclaims idle chunks on frame advance and recycleCompleted', async () => {
  const { device, createdBuffers, deferredFences } = createMockGpuDevice({
    enableWorkDoneFence: true,
  });
  const pool = new GpuStagingPool({
    device,
    chunkSize: 512,
    maxPoolBytes: 4096,
    framesInFlight: 2,
    minWarmChunks: 1,
    idleEvictFrames: 4,
  });

  // Frame 0: allocate 3 chunks (2 standard + 1 oversized).
  pool.beginFrame(0);
  pool.allocate(400, 4);
  pool.allocate(400, 4);
  pool.allocate(1024, 4);
  pool.endFrame();

  // Frames 1..3: idle age < 4, nothing auto-reclaimed yet.
  for (let f = 1; f <= 3; f += 1) {
    const beginRes = pool.beginFrame(f);
    assert.equal(beginRes.reclaimedChunks, 0);
    const endRes = pool.endFrame();
    assert.equal(endRes.reclaimedChunks, 0);
  }
  assert.equal(pool.getStats().recycledChunks, 3);
  assert.equal(pool.getStats().reclaimedChunks, 0);

  // Frame 4: idle age == 4 >= idleEvictFrames -> auto-reclaims oversized + 1 standard chunk!
  const begin4 = pool.beginFrame(4);
  assert.equal(begin4.reclaimedChunks, 2);
  assert.equal(begin4.reclaimedBytes, 1536);
  assert.equal(begin4.recycledChunks, 1);
  assert.equal(pool.getStats().reclaimedChunks, 2);
  assert.equal(pool.getStats().reclaimedBytes, 1536);
  assert.equal(pool.getStats().allocatedBytes, 512);
  assert.equal(pool.getStats().peakBytesAllocated, 2048);

  // Also verify recycleCompleted() triggers auto-reclamation when a fence resolves after idleEvictFrames.
  // Allocate 2 chunks in frame 4 (reusing warm chunk + 1 fresh chunk), retire them, advance to frame 8
  // with long framesInFlight pool so only fence resolution recycles them.
  pool.endFrame();
  pool.destroy();

  const fenceMock = createMockGpuDevice({ enableWorkDoneFence: true });
  const fencePool = new GpuStagingPool({
    device: fenceMock.device,
    chunkSize: 512,
    maxPoolBytes: 4096,
    framesInFlight: 10,
    minWarmChunks: 1,
    idleEvictFrames: 3,
  });

  fencePool.beginFrame(0);
  fencePool.allocate(400, 4);
  fencePool.allocate(400, 4);
  fencePool.endFrame();

  // Frame 4: still in-flight because framesInFlight=10 and fence not yet resolved -> NOT reclaimed!
  fencePool.beginFrame(4);
  assert.equal(fencePool.getStats().inFlightChunks, 2);
  assert.equal(fencePool.getStats().reclaimedChunks, 0);

  // Resolve GPU fence and call recycleCompleted(): both move to recycled, 1 immediately auto-reclaimed!
  fenceMock.deferredFences[0].resolve();
  await fenceMock.deferredFences[0].promise;
  await Promise.resolve();

  const moved = fencePool.recycleCompleted();
  assert.equal(moved, 2);
  assert.equal(fencePool.getStats().recycledChunks, 1);
  assert.equal(fencePool.getStats().reclaimedChunks, 1);
  assert.equal(fencePool.getStats().reclaimedBytes, 512);
  fencePool.destroy();

  // Default idleEvictFrames === 0 must never unexpectedly auto-evict warm chunks.
  const defaultMock = createMockGpuDevice();
  const defaultPool = new GpuStagingPool({
    device: defaultMock.device,
    chunkSize: 512,
    maxPoolBytes: 2048,
    framesInFlight: 1,
  });
  defaultPool.beginFrame(0);
  defaultPool.allocate(400, 4);
  defaultPool.allocate(400, 4);
  defaultPool.endFrame();
  defaultPool.beginFrame(100);
  defaultPool.endFrame();
  assert.equal(defaultPool.getStats().recycledChunks, 2);
  assert.equal(defaultPool.getStats().reclaimedChunks, 0);
  defaultPool.destroy();
  assert.ok(createdBuffers.length === 3 && deferredFences.length === 1);
});

test('W126: active and in-flight chunks are never reclaimed prematurely in writeBuffer or mapped mode', async () => {
  // 1. writeBuffer mode: active + in-flight chunks must never be destroyed by reclaimIdleChunks(0).
  const { device, createdBuffers } = createMockGpuDevice();
  const pool = new GpuStagingPool({
    device,
    chunkSize: 512,
    maxPoolBytes: 4096,
    framesInFlight: 4,
    minWarmChunks: 0,
    idleEvictFrames: 1,
  });

  // Frame 0: allocate chunk 0 and retire it (in-flight until frame 4).
  pool.beginFrame(0);
  pool.allocate(400, 4);
  pool.endFrame();
  assert.equal(pool.getStats().inFlightChunks, 1);

  // Frame 2: chunk 0 is still in-flight (2 - 0 = 2 < framesInFlight=4), chunk 1 is active.
  pool.beginFrame(2);
  pool.allocate(400, 4);
  assert.equal(pool.getStats().inFlightChunks, 1);
  assert.equal(pool.getStats().activeChunks, 1);

  // Explicit aggressive reclaimIdleChunks(0) must NOT touch active or in-flight chunks!
  assert.deepEqual(pool.reclaimIdleChunks(0), { reclaimedChunks: 0, reclaimedBytes: 0 });
  assert.equal(createdBuffers[0].destroyCount, 0);
  assert.equal(createdBuffers[1].destroyCount, 0);
  assert.equal(pool.getStats().inFlightChunks, 1);
  assert.equal(pool.getStats().activeChunks, 1);
  pool.endFrame();

  // Frame 4: chunk 0 becomes recycled (age 4 >= idleEvictFrames=1) and is reclaimed,
  // while chunk 1 (retired at frame 2, in-flight until frame 6) stays untouched!
  const begin4 = pool.beginFrame(4);
  assert.equal(begin4.reclaimedChunks, 1);
  assert.equal(begin4.reclaimedBytes, 512);
  assert.equal(createdBuffers[0].destroyCount, 1);
  assert.equal(createdBuffers[1].destroyCount, 0);
  assert.equal(pool.getStats().inFlightChunks, 1);
  assert.equal(pool.getStats().recycledChunks, 0);
  pool.destroy();
  assert.equal(createdBuffers[1].destroyCount, 1);

  // 2. mapped mode: in-flight unmapped/pending-remap chunks are protected; recycled mapped chunks reclaim cleanly.
  const mappedMock = createMockGpuDevice({ enableWorkDoneFence: true });
  const mappedPool = new GpuStagingPool({
    device: mappedMock.device,
    chunkSize: 512,
    maxPoolBytes: 2048,
    framesInFlight: 2,
    minWarmChunks: 1,
    mode: STAGING_MODE.MAPPED,
  });

  mappedPool.beginFrame(0);
  mappedPool.allocate(400, 4); // chunk 0
  mappedPool.allocate(400, 4); // chunk 1
  mappedPool.endFrame();

  // Still in-flight at frame 1 before fence resolves.
  mappedPool.beginFrame(1);
  assert.deepEqual(mappedPool.reclaimIdleChunks(0), { reclaimedChunks: 0, reclaimedBytes: 0 });
  assert.equal(mappedMock.createdBuffers[0].destroyCount, 0);
  assert.equal(mappedMock.createdBuffers[1].destroyCount, 0);

  // Resolve fence and await mapAsync -> chunks move to recycled.
  mappedMock.deferredFences[0].resolve();
  await mappedMock.deferredFences[0].promise;
  await mappedPool.waitForPendingMaps();
  mappedPool.endFrame();

  mappedPool.beginFrame(5);
  const mappedReclaim = mappedPool.reclaimIdleChunks(5);
  assert.deepEqual(mappedReclaim, { reclaimedChunks: 1, reclaimedBytes: 512 });
  assert.equal(mappedPool.getStats().recycledChunks, 1);
  assert.equal(mappedPool.getStats().reclaimedChunks, 1);
  assert.equal(mappedPool.getStats().reclaimedBytes, 512);

  // Remaining mapped chunk is cleanly destroyed on pool.destroy() with zero double-destroy.
  mappedPool.destroy();
  assert.equal(mappedMock.createdBuffers[0].destroyCount, 1);
  assert.equal(mappedMock.createdBuffers[1].destroyCount, 1);
});

// ---------------------------------------------------------------------------
// W176: 4K Bink2 YUV large-slab persistent bucket reuse (2–16 MiB) &
// upload bandwidth / reuse telemetry.
// ---------------------------------------------------------------------------

/**
 * Lightweight mock WebGPU device for multi-frame 4K (12.5 MiB/frame) tests.
 * Validates buffer bounds & 4-byte alignment without duplicating full multi-MiB
 * snapshots on every frame.
 */
function createLightweight4kMockDevice(options = {}) {
  let nextBufferId = 1;
  const createdBuffers = [];
  const queueWrites = [];
  const deferredFences = [];
  const allocateBacking = Boolean(options.allocateBacking);

  const queue = {
    writeBuffer(buffer, bufferOffset, data) {
      const byteLength = ArrayBuffer.isView(data) ? data.byteLength : data.byteLength;
      assert.equal(bufferOffset % 4, 0);
      assert.equal(byteLength % 4, 0);
      assert.ok(bufferOffset + byteLength <= buffer.size, 'writeBuffer must stay in bounds');
      if (buffer._data) {
        const bytes =
          data instanceof Uint8Array
            ? data
            : ArrayBuffer.isView(data)
              ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
              : new Uint8Array(data);
        buffer._data.set(bytes, bufferOffset);
      }
      queueWrites.push({ buffer, bufferOffset, byteLength });
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
      const needsBacking = allocateBacking || Boolean(descriptor.mappedAtCreation);
      const backing = needsBacking ? new ArrayBuffer(descriptor.size) : null;
      const dataView = backing ? new Uint8Array(backing) : null;
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
        getMappedRange() {
          if (!this._mapped) {
            throw new Error(`Buffer ${this.id} is not mapped`);
          }
          return backing;
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

function createLightweightEncoder() {
  const bufferCopies = [];
  const textureCopies = [];
  return {
    bufferCopies,
    textureCopies,
    copyBufferToBuffer(srcBuffer, srcOffset, dstBuffer, dstOffset, size) {
      assert.ok(srcOffset + size <= srcBuffer.size);
      bufferCopies.push({ srcBuffer, srcOffset, dstBuffer, dstOffset, size });
    },
    copyBufferToTexture(source, destination, copySize) {
      const totalBytes = source.bytesPerRow * source.rowsPerImage * copySize.depthOrArrayLayers;
      assert.ok(source.offset + totalBytes <= source.buffer.size);
      assert.equal(source.bytesPerRow % 256, 0);
      textureCopies.push({
        source: { ...source },
        destination: { ...destination, origin: { ...destination.origin } },
        copySize: { ...copySize },
        totalBytes,
      });
    },
  };
}

test('W176: selectLargeSlabBucket and constructor option validation for large-slab buckets', () => {
  const MiB = 1024 * 1024;
  assert.equal(DEFAULT_LARGE_SLAB_THRESHOLD, 2 * MiB);
  assert.equal(DEFAULT_MAX_LARGE_SLABS, 12);
  assert.deepEqual(
    Array.from(DEFAULT_LARGE_SLAB_BUCKETS),
    [2 * MiB, 3 * MiB, 4 * MiB, 6 * MiB, 8 * MiB, 12 * MiB, 16 * MiB]
  );

  // 4K Bink2 YUV plane sizes map cleanly onto 2–16 MiB buckets.
  assert.equal(selectLargeSlabBucket(2 * MiB), 2 * MiB);
  assert.equal(selectLargeSlabBucket(2048 * 1080), 3 * MiB); // 4K Cr/Cb 256B-aligned (2.11 MiB)
  assert.equal(selectLargeSlabBucket(3840 * 1080), 4 * MiB); // 4K NV12 UV (3.96 MiB)
  assert.equal(selectLargeSlabBucket(5 * MiB), 6 * MiB);
  assert.equal(selectLargeSlabBucket(3840 * 2160), 8 * MiB); // 4K Y plane (7.91 MiB)
  assert.equal(selectLargeSlabBucket(Math.ceil(8.3 * MiB)), 12 * MiB); // 8.3 MiB slab
  assert.equal(selectLargeSlabBucket(15 * MiB), 16 * MiB);
  assert.equal(selectLargeSlabBucket(20 * MiB + 1), 20 * MiB + 4); // >16 MiB falls back to 4B-aligned exact

  assert.throws(() => selectLargeSlabBucket(0), RangeError);
  assert.throws(() => selectLargeSlabBucket(-4), RangeError);

  const { device } = createLightweight4kMockDevice();
  assert.throws(() => new GpuStagingPool({ device, maxLargeSlabs: -1 }), RangeError);
  assert.throws(() => new GpuStagingPool({ device, maxLargeSlabs: 1.5 }), RangeError);
  assert.throws(() => new GpuStagingPool({ device, largeSlabMinBytes: 0 }), RangeError);
  assert.throws(() => new GpuStagingPool({ device, largeSlabBuckets: 'bad' }), TypeError);
  assert.throws(() => new GpuStagingPool({ device, largeSlabBuckets: [0] }), RangeError);
  assert.throws(() => new GpuStagingPool({ device, now: 'not-a-fn' }), TypeError);
});

test('W176: 30 FPS 4K Bink2 YUV plane uploads reuse persistent large slabs with zero steady-state GPUBuffer churn', () => {
  const MiB = 1024 * 1024;
  let fakeTimeMs = 0;
  const { device, createdBuffers } = createLightweight4kMockDevice();
  const pool = new GpuStagingPool({
    device,
    // Default chunkSize = 4 MiB, maxPoolBytes = 64 MiB, framesInFlight = 3, maxLargeSlabs = 12
    now: () => fakeTimeMs,
  });

  const encoder = createLightweightEncoder();
  const dstY = { id: 'bink-y-4k' };
  const dstCb = { id: 'bink-cb-4k' };
  const dstCr = { id: 'bink-cr-4k' };
  const dstUniform = { id: 'frame-meta' };

  // 4K YUV420 planes:
  //  - Y:  3840x2160 R8 (bytesPerRow = 3840, 256B-aligned -> 8,294,400 B -> 8 MiB bucket)
  //  - Cb: 1920x1080 R8 (bytesPerRow = 2048 pre-aligned -> 2,211,840 B -> 3 MiB bucket)
  //  - Cr: 1920x1080 R8 (bytesPerRow = 2048 pre-aligned -> 2,211,840 B -> 3 MiB bucket)
  const yBytes = new Uint8Array(3840 * 2160);
  const cbBytes = new Uint8Array(2048 * 1080);
  const crBytes = new Uint8Array(2048 * 1080);
  const metaBytes = new Uint8Array(256);
  const frameVideoBytes = yBytes.byteLength + cbBytes.byteLength + crBytes.byteLength;
  const frameTotalBytes = frameVideoBytes + metaBytes.byteLength;

  const totalFrames = 30;
  const frameMs = 1000 / 30;

  for (let frame = 0; frame < totalFrames; frame += 1) {
    fakeTimeMs = frame * frameMs;
    pool.beginFrame(frame);

    // Small uniform upload before video planes -> uses standard 4 MiB chunk.
    const metaUpload = pool.stageBufferUpload(encoder, dstUniform, 0, metaBytes);

    // 4K Y, Cb, Cr texture uploads -> persistent large-slab buckets (8 MiB + 2x 3 MiB).
    const yUpload = pool.stageTextureUpload(encoder, dstY, yBytes, {
      width: 3840,
      height: 2160,
      bytesPerRow: 3840,
    });
    const cbUpload = pool.stageTextureUpload(encoder, dstCb, cbBytes, {
      width: 1920,
      height: 1080,
      bytesPerRow: 2048,
    });
    const crUpload = pool.stageTextureUpload(encoder, dstCr, crBytes, {
      width: 1920,
      height: 1080,
      bytesPerRow: 2048,
    });

    // Small upload after video planes still packs into the preserved standard _currentChunk!
    const metaTail = pool.stageBufferUpload(encoder, dstUniform, 256, metaBytes);
    assert.equal(metaTail.buffer, metaUpload.buffer);
    assert.equal(metaTail.offset, 256);

    assert.equal(yUpload.buffer.size, 8 * MiB);
    assert.equal(cbUpload.buffer.size, 3 * MiB);
    assert.equal(crUpload.buffer.size, 3 * MiB);

    fakeTimeMs = (frame + 1) * frameMs;
    const endStats = pool.endFrame();
    assert.equal(endStats.bytesUploaded, frameTotalBytes + metaBytes.byteLength);

    if (frame === 2) {
      // After 3 warmup frames (framesInFlight = 3):
      // 3 frames * (1 standard 4 MiB chunk + 1x 8 MiB Y + 2x 3 MiB chroma) = 12 buffers total, 54 MiB <= 64 MiB.
      assert.equal(createdBuffers.length, 12);
      assert.equal(pool.getStats().largeSlabMisses, 9);
      assert.equal(pool.getStats().largeSlabHits, 0);
    }
  }

  // Frames 3..29 (27 steady-state frames) must create ZERO new GPUBuffers!
  assert.equal(createdBuffers.length, 12, 'steady-state 30 FPS 4K video must not churn GPUBuffers');

  const stats = pool.getStats();
  // 30 frames * 3 video planes = 90 large-slab uploads: 9 warmup misses + 81 bucket hits!
  assert.equal(stats.largeSlabMisses, 9);
  assert.equal(stats.largeSlabHits, 81);
  assert.equal(stats.largeSlabReuseRate, 81 / 90);
  assert.equal(stats.largeSlabEvictions, 0);
  assert.equal(stats.oversizeAllocations, 30); // Y plane (7.91 MiB > 4 MiB chunkSize) x 30 frames

  // 30 frames * 5 allocations/frame = 150 total allocations:
  // 12 fresh GPUBuffers created, 138 allocations served from existing/reused buffers!
  assert.equal(stats.allocationCount, 150);
  assert.equal(stats.freshAllocations, 12);
  assert.equal(stats.reusedAllocations, 138);
  assert.equal(stats.freshAllocations + stats.reusedAllocations, stats.allocationCount);
  assert.equal(stats.reuseRate, 138 / 150);

  // Pool footprint: 3 frames * (4 MiB std + 8 MiB Y + 3 MiB Cb + 3 MiB Cr) = 54 MiB <= 64 MiB!
  const expectedPeakPoolBytes = 3 * (4 + 8 + 3 + 3) * MiB;
  assert.equal(stats.allocatedBytes, expectedPeakPoolBytes);
  assert.equal(stats.peakAllocationBytes, expectedPeakPoolBytes);
  assert.equal(stats.peakAllocatedBytes, expectedPeakPoolBytes);
  assert.equal(stats.peakSingleAllocationBytes, 3840 * 2160);
  assert.equal(stats.largeSlabBytes, 3 * (8 + 3 + 3) * MiB);

  // Upload bandwidth telemetry across 1.0 second (30 frames @ 30 FPS).
  const expectedPerFrameUpload = frameTotalBytes + metaBytes.byteLength;
  assert.equal(stats.totalBytesUploaded, expectedPerFrameUpload * totalFrames);
  assert.equal(stats.bytesUploaded, stats.totalBytesUploaded);
  assert.equal(stats.framesCompleted, totalFrames);
  assert.equal(stats.lastFrameBytesUploaded, expectedPerFrameUpload);
  assert.equal(stats.peakFrameBytesUploaded, expectedPerFrameUpload);
  assert.equal(stats.avgBytesPerFrame, expectedPerFrameUpload);
  assert.ok(Math.abs(stats.uploadBytesPerSec - stats.totalBytesUploaded) < 1);
  assert.ok(stats.uploadMiBPerSec > 360 && stats.uploadMiBPerSec < 370);

  // Clean teardown destroys all 12 buffers once and zeroes live large-slab bytes.
  pool.destroy();
  for (const buf of createdBuffers) {
    assert.equal(buf.destroyCount, 1);
  }
  const destroyed = pool.getStats();
  assert.equal(destroyed.allocatedBytes, 0);
  assert.equal(destroyed.largeSlabBytes, 0);
  assert.equal(destroyed.idleLargeSlabs, 0);
  assert.equal(destroyed.peakAllocationBytes, expectedPeakPoolBytes);
  assert.equal(destroyed.totalBytesUploaded, expectedPerFrameUpload * totalFrames);
});

test('W176: 4K video large slabs (8.3 MiB Y + 2x 2.1 MiB chroma) reuse cleanly in mapped mode with GPU work fence', async () => {
  const MiB = 1024 * 1024;
  const { device, createdBuffers, queueWrites, deferredFences } = createLightweight4kMockDevice({
    enableWorkDoneFence: true,
    allocateBacking: true,
  });

  const pool = new GpuStagingPool({
    device,
    chunkSize: 4 * MiB,
    maxPoolBytes: 64 * MiB,
    framesInFlight: 3,
    mode: STAGING_MODE.MAPPED,
  });

  const encoder = createLightweightEncoder();
  const dstY = { id: 'mapped-y' };
  const dstU = { id: 'mapped-u' };
  const dstV = { id: 'mapped-v' };

  // 8.3 MiB (~8,703,232 B, 256B-aligned row width 4096 x 2124 = 8,699,904 B -> 12 MiB bucket)
  // + 2x 2.1 MiB (2048 x 1080 = 2,211,840 B -> 3 MiB bucket).
  const ySrc = new Uint8Array(4096 * 2124);
  const uSrc = new Uint8Array(2048 * 1080);
  const vSrc = new Uint8Array(2048 * 1080);
  ySrc[0] = 0x41;
  ySrc[ySrc.length - 1] = 0x42;
  uSrc[0] = 0x51;
  vSrc[0] = 0x61;

  pool.beginFrame(0);
  const y0 = pool.stageTextureUpload(encoder, dstY, ySrc, { width: 4096, height: 2124, bytesPerRow: 4096 });
  const u0 = pool.stageTextureUpload(encoder, dstU, uSrc, { width: 1920, height: 1080, bytesPerRow: 2048 });
  const v0 = pool.stageTextureUpload(encoder, dstV, vSrc, { width: 1920, height: 1080, bytesPerRow: 2048 });

  assert.equal(queueWrites.length, 0, 'mapped mode must never call queue.writeBuffer');
  assert.equal(createdBuffers.length, 3);
  assert.equal(y0.buffer.size, 12 * MiB);
  assert.equal(u0.buffer.size, 3 * MiB);
  assert.equal(v0.buffer.size, 3 * MiB);
  assert.equal(y0.buffer._data[0], 0x41);
  assert.equal(y0.buffer._data[ySrc.length - 1], 0x42);
  assert.equal(u0.buffer._data[0], 0x51);
  assert.equal(v0.buffer._data[0], 0x61);
  assert.equal(pool.getStats().activeLargeSlabs, 3);

  pool.endFrame();
  assert.equal(pool.getStats().activeLargeSlabs, 0);
  assert.equal(pool.getStats().inFlightLargeSlabs, 3);
  for (const buf of createdBuffers) {
    assert.equal(buf.unmapCount, 1);
    assert.equal(buf._mapped, false);
  }

  // Resolve Frame 0 GPU fence early -> triggers mapAsync on all 3 slabs.
  deferredFences[0].resolve();
  await deferredFences[0].promise;
  await pool.waitForPendingMaps();

  // Frame 1 immediately reuses all 3 mapped large slabs with ZERO new GPUBuffers!
  pool.beginFrame(1);
  assert.equal(pool.getStats().idleLargeSlabs, 3);
  assert.deepEqual(pool.getStats().largeSlabBucketCounts, {
    [12 * MiB]: 1,
    [3 * MiB]: 2,
  });

  ySrc[0] = 0x77;
  const y1 = pool.stageTextureUpload(encoder, dstY, ySrc, { width: 4096, height: 2124, bytesPerRow: 4096 });
  const u1 = pool.stageTextureUpload(encoder, dstU, uSrc, { width: 1920, height: 1080, bytesPerRow: 2048 });
  const v1 = pool.stageTextureUpload(encoder, dstV, vSrc, { width: 1920, height: 1080, bytesPerRow: 2048 });

  assert.equal(createdBuffers.length, 3, 'mapped mode must reuse remapped large slabs');
  assert.equal(y1.buffer, y0.buffer);
  assert.equal(y1.buffer._data[0], 0x77);
  assert.ok([u0.buffer, v0.buffer].includes(u1.buffer));
  assert.ok([u0.buffer, v0.buffer].includes(v1.buffer));
  assert.notEqual(u1.buffer, v1.buffer);

  const stats = pool.getStats();
  assert.equal(stats.largeSlabMisses, 3);
  assert.equal(stats.largeSlabHits, 3);
  assert.equal(stats.freshAllocations, 3);
  assert.equal(stats.reusedAllocations, 3);

  pool.endFrame();
  pool.destroy();
  for (const buf of createdBuffers) {
    assert.equal(buf.destroyCount, 1);
  }
});

test('W176: maxLargeSlabs bounded idle cap and maxPoolBytes eviction prevent unbounded large-slab growth', () => {
  const MiB = 1024 * 1024;
  const { device, createdBuffers } = createLightweight4kMockDevice();

  // 1. Bounded idle cap: maxLargeSlabs = 2 with 4 slabs allocated in Frame 0.
  const pool = new GpuStagingPool({
    device,
    chunkSize: 4 * MiB,
    maxPoolBytes: 64 * MiB,
    framesInFlight: 2,
    maxLargeSlabs: 2,
  });

  pool.beginFrame(0);
  const s1 = pool.allocate(2 * MiB, 256); // 2 MiB bucket
  const s2 = pool.allocate(3 * MiB, 256); // 3 MiB bucket
  const s3 = pool.allocate(8 * MiB, 256); // 8 MiB bucket
  const s4 = pool.allocate(16 * MiB, 256); // 16 MiB bucket
  assert.equal(s1.isLargeSlab, true);
  assert.equal(s1.slabBucketSize, 2 * MiB);
  assert.equal(s4.slabBucketSize, 16 * MiB);
  assert.equal(pool.getStats().activeLargeSlabs, 4);
  assert.equal(pool.getStats().largeSlabBytes, (2 + 3 + 8 + 16) * MiB);
  pool.endFrame();

  // Frame 2 recycles all 4 slabs -> cap = 2 must immediately evict the 2 largest (16 MiB & 8 MiB)!
  const begin2 = pool.beginFrame(2);
  assert.equal(begin2.reclaimedChunks, 2);
  assert.equal(begin2.reclaimedBytes, (16 + 8) * MiB);
  assert.equal(createdBuffers[3].destroyCount, 1, 'largest idle slab (16 MiB) evicted first on tie');
  assert.equal(createdBuffers[2].destroyCount, 1, 'second largest idle slab (8 MiB) evicted next');
  assert.equal(createdBuffers[0].destroyCount, 0);
  assert.equal(createdBuffers[1].destroyCount, 0);

  const capStats = pool.getStats();
  assert.equal(capStats.idleLargeSlabs, 2);
  assert.equal(capStats.largeSlabEvictions, 2);
  assert.equal(capStats.largeSlabBytes, (2 + 3) * MiB);
  assert.equal(capStats.allocatedBytes, (2 + 3) * MiB);
  assert.equal(capStats.peakAllocationBytes, (2 + 3 + 8 + 16) * MiB);
  pool.endFrame();
  pool.destroy();

  // 2. maxLargeSlabs = 0 retains zero idle large slabs upon recycle.
  const zeroCapMock = createLightweight4kMockDevice();
  const zeroCapPool = new GpuStagingPool({
    device: zeroCapMock.device,
    chunkSize: 4 * MiB,
    maxPoolBytes: 32 * MiB,
    framesInFlight: 1,
    maxLargeSlabs: 0,
  });
  zeroCapPool.beginFrame(0);
  zeroCapPool.allocate(8 * MiB, 256);
  zeroCapPool.endFrame();
  zeroCapPool.beginFrame(1);
  assert.equal(zeroCapPool.getStats().idleLargeSlabs, 0);
  assert.equal(zeroCapPool.getStats().largeSlabEvictions, 1);
  assert.equal(zeroCapMock.createdBuffers[0].destroyCount, 1);
  zeroCapPool.destroy();

  // 3. Budget eviction frees idle large slabs to stay under maxPoolBytes, and failed
  //    allocations throw RangeError without inflating telemetry.
  const tightMock = createLightweight4kMockDevice();
  const tightPool = new GpuStagingPool({
    device: tightMock.device,
    chunkSize: 4 * MiB,
    maxPoolBytes: 18 * MiB,
    framesInFlight: 1,
    maxLargeSlabs: 8,
  });

  tightPool.beginFrame(0);
  tightPool.allocate(1024, 4); // 4 MiB standard chunk
  tightPool.allocate(8 * MiB, 256); // 8 MiB large slab (total 12 MiB)
  tightPool.endFrame();

  // Frame 1: both buffers idle. Allocating a 12 MiB slab needs 12 + 12 = 24 > 18 MiB ->
  // must evict the idle 8 MiB large slab FIRST (keeping the 4 MiB standard chunk warm: 4 + 12 = 16 <= 18 MiB)!
  tightPool.beginFrame(1);
  const slab12 = tightPool.allocate(10 * MiB, 256);
  assert.equal(slab12.buffer.size, 12 * MiB);
  assert.equal(tightMock.createdBuffers[1].destroyCount, 1, 'idle large slab evicted before standard chunk');
  assert.equal(tightMock.createdBuffers[0].destroyCount, 0, 'warm standard chunk preserved');
  assert.equal(tightPool.getStats().allocatedBytes, 16 * MiB);
  assert.equal(tightPool.getStats().largeSlabEvictions, 1);

  // Over-budget 8 MiB request while 12 MiB slab is active (only 4 MiB standard idle -> 12 + 8 = 20 > 18 MiB) throws.
  const beforeFail = tightPool.getStats();
  assert.throws(() => tightPool.allocate(8 * MiB, 256), RangeError);
  const afterFail = tightPool.getStats();
  assert.equal(afterFail.largeSlabMisses, beforeFail.largeSlabMisses);
  assert.equal(afterFail.freshAllocations, beforeFail.freshAllocations);
  assert.equal(afterFail.peakAllocationBytes, beforeFail.peakAllocationBytes);
  tightPool.destroy();

  // 4. Tight maxPoolBytes clamping: 8.3 MiB request inside a 10 MiB pool clamps bucket to exact aligned size
  //    instead of spuriously failing on 12 MiB bucket rounding, and reuses that clamped bucket on next frame!
  const clampMock = createLightweight4kMockDevice();
  const clampPool = new GpuStagingPool({
    device: clampMock.device,
    chunkSize: 4 * MiB,
    maxPoolBytes: 10 * MiB,
    framesInFlight: 1,
  });
  const reqBytes = Math.ceil(8.3 * MiB / 4) * 4;
  clampPool.beginFrame(0);
  const c0 = clampPool.allocate(reqBytes, 256);
  assert.equal(c0.buffer.size, reqBytes);
  assert.equal(c0.slabBucketSize, reqBytes);
  clampPool.endFrame();

  clampPool.beginFrame(1);
  const c1 = clampPool.allocate(reqBytes, 256);
  assert.equal(c1.buffer, c0.buffer);
  assert.equal(clampMock.createdBuffers.length, 1);
  assert.equal(clampPool.getStats().largeSlabHits, 1);
  clampPool.destroy();
});

test('W176: bucket isolation, explicit release, >16 MiB fallback, and idleEvictFrames post-cutscene reclamation', () => {
  const MiB = 1024 * 1024;
  const { device, createdBuffers } = createLightweight4kMockDevice();
  const pool = new GpuStagingPool({
    device,
    chunkSize: 4 * MiB,
    maxPoolBytes: 64 * MiB,
    framesInFlight: 2,
    minWarmChunks: 1,
    idleEvictFrames: 5,
  });

  // Frame 0: allocate a standard chunk + 8 MiB Y slab, then explicitly release the Y slab.
  pool.beginFrame(0);
  pool.allocate(1024, 4); // chunk 0: 4 MiB standard
  const ySlice = pool.allocate(3840 * 2160, 256); // chunk 1: 8 MiB large slab
  assert.equal(pool.release(ySlice), true);
  assert.equal(pool.release(ySlice), false); // double-release guarded
  assert.equal(pool.getStats().inFlightLargeSlabs, 1);
  pool.endFrame();

  // Frame 2: 8 MiB Y slab is idle. Requesting a 2.1 MiB chroma slab must NOT steal the 8 MiB Y bucket!
  pool.beginFrame(2);
  const cbSlice = pool.allocate(2048 * 1080, 256); // chunk 2: fresh 3 MiB bucket
  assert.equal(cbSlice.buffer.size, 3 * MiB);
  assert.notEqual(cbSlice.buffer, ySlice.buffer);

  // Immediately requesting Y plane hits the preserved 8 MiB bucket!
  const yReuse = pool.allocate(3840 * 2160, 256);
  assert.equal(yReuse.buffer, ySlice.buffer);
  assert.equal(pool.getStats().largeSlabHits, 1);

  // >16 MiB slab (e.g., 20 MiB) allocates exact aligned size and reuses on future frames.
  const mega = pool.allocate(20 * MiB, 256);
  assert.equal(mega.buffer.size, 20 * MiB);
  assert.equal(mega.isLargeSlab, true);
  pool.endFrame();

  // Frame 4: reuse the 20 MiB slab.
  pool.beginFrame(4);
  const megaReuse = pool.allocate(20 * MiB, 256);
  assert.equal(megaReuse.buffer, mega.buffer);
  assert.equal(pool.getStats().largeSlabHits, 2);
  pool.endFrame();

  // Cutscene stops! Advance to frame 10 (>= idleEvictFrames = 5 past last usage at frames 0, 2, 4):
  // all idle large slabs (20 MiB, 8 MiB, 3 MiB) are reclaimed while 1 warm standard chunk stays alive!
  const begin10 = pool.beginFrame(10);
  assert.equal(begin10.reclaimedChunks, 3);
  assert.equal(begin10.reclaimedBytes, (20 + 8 + 3) * MiB);
  assert.equal(createdBuffers[0].destroyCount, 0, 'minWarmChunks standard chunk remains warm');
  assert.equal(createdBuffers[1].destroyCount, 1, 'stale 8 MiB slab reclaimed');
  assert.equal(createdBuffers[2].destroyCount, 1, 'stale 3 MiB slab reclaimed');
  assert.equal(createdBuffers[3].destroyCount, 1, 'stale 20 MiB slab reclaimed');

  const afterCutscene = pool.getStats();
  assert.equal(afterCutscene.idleLargeSlabs, 0);
  assert.equal(afterCutscene.largeSlabBytes, 0);
  assert.equal(afterCutscene.largeSlabEvictions, 3);
  assert.equal(afterCutscene.allocatedBytes, 4 * MiB);
  assert.equal(afterCutscene.recycledChunks, 1);

  pool.destroy();
});

// ---------------------------------------------------------------------------
// W196: Power-of-two bucket scaling through 16 MiB for 4K Bink2 texture
// uploads, per-bucket caps, deterministic LRU eviction, and pool telemetry.
// ---------------------------------------------------------------------------

test('W196: power-of-two bucket helpers scale smoothly through 16 MiB and validate constructor options', () => {
  const KiB = 1024;
  const MiB = 1024 * 1024;

  assert.equal(DEFAULT_POW2_MIN_SLAB_BYTES, 1 * MiB);
  assert.equal(DEFAULT_POW2_MAX_BUCKET_BYTES, 16 * MiB);
  assert.deepEqual(
    Array.from(POW2_LARGE_SLAB_BUCKETS),
    [1 * MiB, 2 * MiB, 4 * MiB, 8 * MiB, 16 * MiB]
  );
  assert.deepEqual(BUCKET_STRATEGY, { DEFAULT: 'default', POW2: 'pow2' });

  // nextPowerOfTwo across small, 4K plane, and >32-bit sizes.
  assert.equal(nextPowerOfTwo(1), 1);
  assert.equal(nextPowerOfTwo(4), 4);
  assert.equal(nextPowerOfTwo(5), 8);
  assert.equal(nextPowerOfTwo(1920 * 1080), 2 * MiB); // 2,073,600 B (~2.07 MB) -> 2 MiB
  assert.equal(nextPowerOfTwo(Math.ceil(2.07 * MiB)), 4 * MiB); // 2.07 MiB -> 4 MiB
  assert.equal(nextPowerOfTwo(2048 * 1080), 4 * MiB); // 2.11 MiB aligned chroma -> 4 MiB
  assert.equal(nextPowerOfTwo(3840 * 2160), 8 * MiB); // 7.91 MiB / 8.29 MB 4K Y -> 8 MiB
  assert.equal(nextPowerOfTwo(Math.ceil(8.3 * MiB)), 16 * MiB); // 8.3 MiB 4K Y -> 16 MiB
  assert.equal(nextPowerOfTwo(16 * MiB), 16 * MiB);
  assert.equal(nextPowerOfTwo(0x100000001), 0x200000000);
  assert.throws(() => nextPowerOfTwo(0), RangeError);
  assert.throws(() => nextPowerOfTwo(-8), RangeError);
  assert.throws(() => nextPowerOfTwo(1.5), RangeError);

  // buildPowerOfTwoBuckets default and fine-grained ladders.
  assert.deepEqual(
    Array.from(buildPowerOfTwoBuckets()),
    [1 * MiB, 2 * MiB, 4 * MiB, 8 * MiB, 16 * MiB]
  );
  assert.deepEqual(
    Array.from(buildPowerOfTwoBuckets(64 * KiB, 16 * MiB)),
    [
      64 * KiB,
      128 * KiB,
      256 * KiB,
      512 * KiB,
      1 * MiB,
      2 * MiB,
      4 * MiB,
      8 * MiB,
      16 * MiB,
    ]
  );
  assert.throws(() => buildPowerOfTwoBuckets(16 * MiB, 8 * MiB), RangeError);

  // selectPowerOfTwoBucket scales smoothly through 16 MiB and falls back cleanly above 16 MiB.
  assert.equal(selectPowerOfTwoBucket(1), 4);
  assert.equal(selectPowerOfTwoBucket(1000, 64 * KiB, 16 * MiB), 64 * KiB);
  assert.equal(selectPowerOfTwoBucket(1920 * 1080, 1 * MiB, 16 * MiB), 2 * MiB);
  assert.equal(selectPowerOfTwoBucket(Math.ceil(2.07 * MiB), 1 * MiB, 16 * MiB), 4 * MiB);
  assert.equal(selectPowerOfTwoBucket(3840 * 2160, 1 * MiB, 16 * MiB), 8 * MiB);
  assert.equal(selectPowerOfTwoBucket(Math.ceil(8.3 * MiB), 1 * MiB, 16 * MiB), 16 * MiB);
  assert.equal(selectPowerOfTwoBucket(16 * MiB, 1 * MiB, 16 * MiB), 16 * MiB);
  assert.equal(selectPowerOfTwoBucket(17 * MiB + 1, 1 * MiB, 16 * MiB), 17 * MiB + 4);
  assert.throws(() => selectPowerOfTwoBucket(1024, 8 * MiB, 4 * MiB), RangeError);

  // Constructor option aliases and validation.
  const { device } = createLightweight4kMockDevice();
  const aliasedPool = new GpuStagingPool({
    device,
    maxPooledBytes: 96 * MiB,
    maxPerBucket: 3,
    powerOfTwoBuckets: true,
  });
  assert.equal(aliasedPool.maxPoolBytes, 96 * MiB);
  assert.equal(aliasedPool.maxPooledBytes, 96 * MiB);
  assert.equal(aliasedPool.maxSlabsPerBucket, 3);
  assert.equal(aliasedPool.bucketStrategy, 'pow2');
  assert.deepEqual(Array.from(aliasedPool.largeSlabBuckets), Array.from(POW2_LARGE_SLAB_BUCKETS));
  const initStats = aliasedPool.getStats();
  assert.equal(initStats.poolHits, 0);
  assert.equal(initStats.poolMisses, 0);
  assert.equal(initStats.bytesAllocated, 0);
  assert.equal(initStats.bytesReused, 0);
  assert.equal(initStats.evictedCount, 0);
  assert.equal(initStats.activeCount, 0);
  assert.equal(initStats.idleCount, 0);
  assert.equal(initStats.peakBytes, 0);
  assert.equal(initStats.maxPooledBytes, 96 * MiB);
  assert.equal(initStats.maxSlabsPerBucket, 3);
  assert.equal(initStats.bucketStrategy, 'pow2');
  aliasedPool.destroy();

  assert.throws(
    () => new GpuStagingPool({ device, maxPoolBytes: 16 * MiB, maxPooledBytes: 32 * MiB }),
    RangeError
  );
  assert.throws(
    () => new GpuStagingPool({ device, maxSlabsPerBucket: 2, maxPerBucket: 4 }),
    RangeError
  );
  assert.throws(() => new GpuStagingPool({ device, maxSlabsPerBucket: -1 }), RangeError);
  assert.throws(() => new GpuStagingPool({ device, maxPerBucket: 1.5 }), RangeError);
  assert.throws(() => new GpuStagingPool({ device, bucketStrategy: 'invalid' }), RangeError);
  assert.throws(() => new GpuStagingPool({ device, powerOfTwoBuckets: 'yes' }), TypeError);
  assert.throws(
    () => new GpuStagingPool({ device, bucketStrategy: 'default', powerOfTwoBuckets: true }),
    RangeError
  );
  assert.throws(
    () => new GpuStagingPool({ device, bucketStrategy: 'pow2', powerOfTwoBuckets: false }),
    RangeError
  );
  assert.throws(() => new GpuStagingPool({ device, maxBucketBytes: 0 }), RangeError);
});

test('W196: repeated 8.3 MiB Y + 2.07 MiB U/V 4K plane acquire/release cycles achieve 100% pow2 reuse after warmup', async () => {
  const MiB = 1024 * 1024;
  const yPlane83MiB = Math.ceil(8.3 * MiB); // 8,703,181 B -> 16 MiB pow2 bucket
  const uvPlane207MiB = Math.ceil(2.07 * MiB); // 2,170,553 B -> 4 MiB pow2 bucket
  const perFrameBucketBytes = 16 * MiB + 4 * MiB + 4 * MiB; // 24 MiB / frame

  // 1. Triple-buffered (framesInFlight = 3) explicit allocate/release loop over 33 frames.
  const { device, createdBuffers } = createLightweight4kMockDevice();
  const pool = new GpuStagingPool({
    device,
    bucketStrategy: BUCKET_STRATEGY.POW2,
    maxPooledBytes: 96 * MiB,
    framesInFlight: 3,
    maxLargeSlabs: 12,
    maxSlabsPerBucket: 6,
  });

  const warmupFrames = 3;
  const steadyFrames = 30;
  const totalFrames = warmupFrames + steadyFrames;

  for (let frame = 0; frame < totalFrames; frame += 1) {
    pool.beginFrame(frame);
    const ySlice = pool.allocate(yPlane83MiB, 256);
    const uSlice = pool.allocate(uvPlane207MiB, 256);
    const vSlice = pool.allocate(uvPlane207MiB, 256);

    assert.equal(ySlice.buffer.size, 16 * MiB);
    assert.equal(uSlice.buffer.size, 4 * MiB);
    assert.equal(vSlice.buffer.size, 4 * MiB);
    assert.equal(ySlice.slabBucketSize, 16 * MiB);
    assert.equal(uSlice.slabBucketSize, 4 * MiB);
    assert.equal(vSlice.slabBucketSize, 4 * MiB);

    const midStats = pool.getStats();
    assert.equal(midStats.activeCount, 3);
    assert.equal(midStats.idleCount, 0);

    assert.equal(pool.release(ySlice), true);
    assert.equal(pool.release(uSlice), true);
    assert.equal(pool.release(vSlice), true);
    assert.equal(pool.getStats().activeCount, 0);
    pool.endFrame();

    if (frame === warmupFrames - 1) {
      assert.equal(createdBuffers.length, 9);
      const warmStats = pool.getStats();
      assert.equal(warmStats.poolMisses, 9);
      assert.equal(warmStats.poolHits, 0);
      assert.equal(warmStats.bytesAllocated, 3 * perFrameBucketBytes);
      assert.equal(warmStats.bytesReused, 0);
      assert.equal(warmStats.peakBytes, 3 * perFrameBucketBytes);
      assert.equal(warmStats.evictedCount, 0);
    }
  }

  // Zero new GPUBuffer allocations across all 30 steady-state frames (100% reuse after warmup).
  assert.equal(createdBuffers.length, 9);
  const stats = pool.getStats();
  assert.equal(stats.poolMisses, 9);
  assert.equal(stats.poolHits, steadyFrames * 3);
  assert.equal(stats.largeSlabMisses, 9);
  assert.equal(stats.largeSlabHits, steadyFrames * 3);
  assert.equal(stats.bytesAllocated, 3 * perFrameBucketBytes);
  assert.equal(stats.bytesReused, steadyFrames * perFrameBucketBytes);
  assert.equal(stats.peakBytes, 3 * perFrameBucketBytes);
  assert.equal(stats.allocatedBytes, 3 * perFrameBucketBytes);
  assert.equal(stats.evictedCount, 0);
  assert.equal(stats.activeCount, 0);
  assert.equal(stats.idleCount, 0); // last 3 frames still in-flight

  // Advance framesInFlight frames so all 9 slabs return to idle buckets.
  pool.beginFrame(totalFrames + 3);
  const idleStats = pool.getStats();
  assert.equal(idleStats.activeCount, 0);
  assert.equal(idleStats.idleCount, 9);
  assert.deepEqual(idleStats.largeSlabBucketCounts, {
    [16 * MiB]: 3,
    [4 * MiB]: 6,
  });
  pool.destroy();

  // 2. GPU fence-driven 4K Bink2 stageTextureUpload (3840x2160 Y -> 8 MiB, 1920x1080 U/V -> 4 MiB)
  //    plus raw 2.07 MB (1920x1080) -> 2 MiB bucket reuse.
  const fencedMock = createLightweight4kMockDevice({ enableWorkDoneFence: true });
  const fencedPool = new GpuStagingPool({
    device: fencedMock.device,
    powerOfTwoBuckets: true,
    maxPooledBytes: 64 * MiB,
    framesInFlight: 3,
  });
  const encoder = createLightweightEncoder();
  const yData = new Uint8Array(3840 * 2160); // 7.91 MiB -> 8 MiB pow2 bucket
  const uData = new Uint8Array(1920 * 1080); // 1920B/row -> 2048B/row aligned = 2.11 MiB -> 4 MiB pow2 bucket
  const vData = new Uint8Array(1920 * 1080);
  const dstY = { id: 'pow2-y' };
  const dstU = { id: 'pow2-u' };
  const dstV = { id: 'pow2-v' };

  for (let frame = 0; frame < 11; frame += 1) {
    fencedPool.beginFrame(frame);
    const yUp = fencedPool.stageTextureUpload(encoder, dstY, yData, {
      width: 3840,
      height: 2160,
      bytesPerRow: 3840,
    });
    const uUp = fencedPool.stageTextureUpload(encoder, dstU, uData, {
      width: 1920,
      height: 1080,
      bytesPerRow: 1920,
    });
    const vUp = fencedPool.stageTextureUpload(encoder, dstV, vData, {
      width: 1920,
      height: 1080,
      bytesPerRow: 1920,
    });
    assert.equal(yUp.buffer.size, 8 * MiB);
    assert.equal(uUp.buffer.size, 4 * MiB);
    assert.equal(vUp.buffer.size, 4 * MiB);
    fencedPool.endFrame();

    const fence = fencedMock.deferredFences[fencedMock.deferredFences.length - 1];
    fence.resolve();
    await fence.promise;
    await Promise.resolve();
  }

  assert.equal(fencedMock.createdBuffers.length, 3);
  const fStats = fencedPool.getStats();
  assert.equal(fStats.poolMisses, 3);
  assert.equal(fStats.poolHits, 30);
  assert.equal(fStats.bytesAllocated, 16 * MiB);
  assert.equal(fStats.bytesReused, 160 * MiB);
  assert.equal(fStats.peakBytes, 16 * MiB);
  assert.equal(fStats.evictedCount, 0);

  // Raw 2.07 MB (1920*1080 = 2,073,600 B) hits the 2 MiB pow2 bucket directly and reuses on next frame.
  fencedPool.beginFrame(11);
  const rawChroma0 = fencedPool.allocate(1920 * 1080, 256);
  assert.equal(rawChroma0.buffer.size, 2 * MiB);
  fencedPool.endFrame();
  fencedMock.deferredFences[fencedMock.deferredFences.length - 1].resolve();
  await fencedMock.deferredFences[fencedMock.deferredFences.length - 1].promise;
  await Promise.resolve();

  fencedPool.beginFrame(12);
  const rawChroma1 = fencedPool.allocate(1920 * 1080, 256);
  assert.equal(rawChroma1.buffer, rawChroma0.buffer);
  assert.equal(fencedMock.createdBuffers.length, 4);
  fencedPool.destroy();
});

test('W196: per-bucket cap and maxPooledBytes VRAM cap enforce deterministic LRU eviction and telemetry', () => {
  const MiB = 1024 * 1024;

  // 1. Per-bucket cap (maxSlabsPerBucket = 2) evicts LRU slab inside the overflowing bucket only.
  const bucketMock = createLightweight4kMockDevice();
  const bucketPool = new GpuStagingPool({
    device: bucketMock.device,
    bucketStrategy: BUCKET_STRATEGY.POW2,
    maxPooledBytes: 128 * MiB,
    maxLargeSlabs: 16,
    maxSlabsPerBucket: 2,
    framesInFlight: 1,
  });

  bucketPool.beginFrame(0);
  const c0 = bucketPool.allocate(Math.ceil(2.07 * MiB), 256); // buf 0 (4 MiB, seq 1)
  const c1 = bucketPool.allocate(Math.ceil(2.07 * MiB), 256); // buf 1 (4 MiB, seq 2)
  const c2 = bucketPool.allocate(Math.ceil(2.07 * MiB), 256); // buf 2 (4 MiB, seq 3)
  const y0 = bucketPool.allocate(Math.ceil(8.3 * MiB), 256); // buf 3 (16 MiB, seq 4)
  assert.equal(bucketPool.getStats().activeCount, 4);
  assert.equal(bucketPool.getStats().bytesAllocated, 28 * MiB);
  assert.equal(bucketPool.getStats().peakBytes, 28 * MiB);
  bucketPool.endFrame();

  // Frame 1 recycles all 4 slabs: 4 MiB bucket has 3 > maxSlabsPerBucket(2) -> evicts LRU buf 0!
  const b1 = bucketPool.beginFrame(1);
  assert.equal(b1.reclaimedChunks, 1);
  assert.equal(b1.reclaimedBytes, 4 * MiB);
  assert.equal(c0.buffer.destroyCount, 1, 'oldest (LRU) 4 MiB slab evicted by per-bucket cap');
  assert.equal(c1.buffer.destroyCount, 0);
  assert.equal(c2.buffer.destroyCount, 0);
  assert.equal(y0.buffer.destroyCount, 0, 'unrelated 16 MiB bucket stays warm');

  const b1Stats = bucketPool.getStats();
  assert.equal(b1Stats.evictedCount, 1);
  assert.equal(b1Stats.evictedBytes, 4 * MiB);
  assert.equal(b1Stats.idleCount, 3);
  assert.equal(b1Stats.activeCount, 0);
  assert.equal(b1Stats.allocatedBytes, 24 * MiB);
  assert.equal(b1Stats.peakBytes, 28 * MiB);
  assert.deepEqual(b1Stats.largeSlabBucketCounts, {
    [4 * MiB]: 2,
    [16 * MiB]: 1,
  });

  // Touch MRU slab (c2) in frame 1 while leaving c1 untouched at frame 0, and allocate
  // 2 additional 4 MiB slabs in frame 1 (c3 -> hits c1? No: reuse c2 + fresh c3/c4 after cap=1).
  // Specifically, verify cross-frame per-bucket LRU with maxSlabsPerBucket = 1:
  const c2Reused = bucketPool.allocate(Math.ceil(2.07 * MiB), 256);
  assert.equal(c2Reused.buffer, c2.buffer, 'MRU slab is reused first from bucket');
  bucketPool.endFrame();
  bucketPool.destroy();

  const perBucketCrossFrameMock = createLightweight4kMockDevice();
  const perBucketCrossFramePool = new GpuStagingPool({
    device: perBucketCrossFrameMock.device,
    bucketStrategy: BUCKET_STRATEGY.POW2,
    maxPooledBytes: 64 * MiB,
    maxSlabsPerBucket: 1,
    framesInFlight: 2,
  });
  perBucketCrossFramePool.beginFrame(0);
  const f0Slab = perBucketCrossFramePool.allocate(Math.ceil(2.07 * MiB), 256); // frame 0
  perBucketCrossFramePool.endFrame();
  perBucketCrossFramePool.beginFrame(1);
  const f1Slab = perBucketCrossFramePool.allocate(Math.ceil(2.07 * MiB), 256); // frame 1
  perBucketCrossFramePool.endFrame();
  perBucketCrossFramePool.beginFrame(2); // recycles f0Slab (idle = 1 <= cap 1)
  assert.equal(f0Slab.buffer.destroyCount, 0);
  perBucketCrossFramePool.endFrame();
  perBucketCrossFramePool.beginFrame(3); // recycles f1Slab -> 2 idle > cap 1 -> evicts older f0Slab!
  assert.equal(f0Slab.buffer.destroyCount, 1, 'older frame-0 slab evicted before frame-1 slab');
  assert.equal(f1Slab.buffer.destroyCount, 0, 'newer frame-1 slab retained in bucket');
  assert.equal(perBucketCrossFramePool.getStats().evictedCount, 1);
  assert.equal(perBucketCrossFramePool.getStats().idleCount, 1);
  perBucketCrossFramePool.destroy();

  // 2. Cross-frame deterministic LRU eviction under maxPooledBytes VRAM cap:
  //    older idle oversized slab (frame 0) is evicted BEFORE newer larger slab (frame 1).
  const lruMock = createLightweight4kMockDevice();
  const lruPool = new GpuStagingPool({
    device: lruMock.device,
    bucketStrategy: BUCKET_STRATEGY.POW2,
    maxPooledBytes: 24 * MiB,
    maxLargeSlabs: 8,
    framesInFlight: 1,
  });

  // Frame 0: allocate 4 MiB slab (buf 0, lastUsedFrame = 0).
  lruPool.beginFrame(0);
  const old4 = lruPool.allocate(Math.ceil(2.07 * MiB), 256);
  assert.equal(old4.buffer.size, 4 * MiB);
  lruPool.endFrame();

  // Frame 1: allocate 16 MiB slab (buf 1, lastUsedFrame = 1). Live = 20 MiB <= 24 MiB.
  lruPool.beginFrame(1);
  const newer16 = lruPool.allocate(Math.ceil(8.3 * MiB), 256);
  assert.equal(newer16.buffer.size, 16 * MiB);
  lruPool.endFrame();

  // Frame 2: both slabs idle (4 MiB @ frame 0, 16 MiB @ frame 1).
  // Requesting an 8 MiB slab needs 20 + 8 = 28 > 24 MiB ->
  // deterministic LRU must evict old4 (frame 0) FIRST, leaving newer16 (frame 1) warm (16 + 8 = 24 MiB)!
  lruPool.beginFrame(2);
  assert.equal(lruPool.getStats().idleCount, 2);
  const fresh8 = lruPool.allocate(3840 * 2160, 256); // 8 MiB bucket
  assert.equal(fresh8.buffer.size, 8 * MiB);
  assert.equal(old4.buffer.destroyCount, 1, 'older frame-0 idle slab must be LRU-evicted first');
  assert.equal(newer16.buffer.destroyCount, 0, 'newer frame-1 16 MiB slab must stay warm');

  let lruStats = lruPool.getStats();
  assert.equal(lruStats.allocatedBytes, 24 * MiB);
  assert.equal(lruStats.peakBytes, 24 * MiB);
  assert.equal(lruStats.bytesAllocated, (4 + 16 + 8) * MiB);
  assert.equal(lruStats.bytesReused, 0);
  assert.equal(lruStats.poolMisses, 3);
  assert.equal(lruStats.poolHits, 0);
  assert.equal(lruStats.evictedCount, 1);
  assert.equal(lruStats.evictedBytes, 4 * MiB);
  assert.equal(lruStats.activeCount, 1);
  assert.equal(lruStats.idleCount, 1);

  // Immediately acquiring 8.3 MiB hits the preserved newer16 slab with zero eviction or new allocation!
  const hit16 = lruPool.allocate(Math.ceil(8.3 * MiB), 256);
  assert.equal(hit16.buffer, newer16.buffer);
  lruStats = lruPool.getStats();
  assert.equal(lruStats.poolHits, 1);
  assert.equal(lruStats.bytesReused, 16 * MiB);
  assert.equal(lruStats.activeCount, 2);
  assert.equal(lruStats.idleCount, 0);
  assert.equal(lruStats.allocatedBytes, 24 * MiB);

  // Over-budget allocation while all 24 MiB are active must throw without altering telemetry!
  const beforeOom = lruPool.getStats();
  assert.throws(() => lruPool.allocate(Math.ceil(2.07 * MiB), 256), RangeError);
  const afterOom = lruPool.getStats();
  assert.equal(afterOom.poolMisses, beforeOom.poolMisses);
  assert.equal(afterOom.bytesAllocated, beforeOom.bytesAllocated);
  assert.equal(afterOom.evictedCount, beforeOom.evictedCount);
  assert.equal(afterOom.peakBytes, beforeOom.peakBytes);
  assert.equal(afterOom.allocatedBytes, 24 * MiB);

  lruPool.endFrame();
  lruPool.destroy();
  assert.equal(lruPool.getStats().allocatedBytes, 0);
  assert.equal(lruPool.getStats().activeCount, 0);
  assert.equal(lruPool.getStats().idleCount, 0);
  assert.equal(lruPool.getStats().peakBytes, 24 * MiB);
});

test('W196: classic-worker importScripts exposes pow2 bucket exports without leaking internals', () => {
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

  vm.runInContext(source, ctx, { filename: 'staging_pool.js' });
  const mod = ctx.PlayGta6StagingPool;
  assert.equal(typeof mod.nextPowerOfTwo, 'function');
  assert.equal(typeof mod.buildPowerOfTwoBuckets, 'function');
  assert.equal(typeof mod.selectPowerOfTwoBucket, 'function');
  assert.deepEqual(Array.from(mod.POW2_LARGE_SLAB_BUCKETS), Array.from(POW2_LARGE_SLAB_BUCKETS));
  assert.equal(mod.BUCKET_STRATEGY.POW2, 'pow2');
  assert.equal(ctx.POW2_LARGE_SLAB_BUCKETS, undefined);
  assert.equal(ctx.compareEvictionLru, undefined);
});

// ---------------------------------------------------------------------------
// W245: Sanitized 4K upload pool telemetry, VRAM pressure tracking,
// clamped hitRate [0, 1], lruEvictions, oversizedAllocations,
// totalBytesStaged, and resetStats().
// ---------------------------------------------------------------------------

test('W245: initial getStats exposes sanitized telemetry fields, legacy aliases, and immutable snapshots', () => {
  const MiB = 1024 * 1024;
  const { device } = createLightweight4kMockDevice();
  const pool = new GpuStagingPool({
    device,
    bucketStrategy: BUCKET_STRATEGY.POW2,
    maxPooledBytes: 64 * MiB,
    framesInFlight: 3,
  });

  const stats = pool.getStats();
  assert.equal(stats.poolHits, 0);
  assert.equal(stats.poolMisses, 0);
  assert.equal(stats.hitRate, 0);
  assert.equal(stats.poolHitRate, 0);
  assert.equal(stats.peakAllocatedBytes, 0);
  assert.equal(stats.lruEvictions, 0);
  assert.equal(stats.evictedCount, 0);
  assert.equal(stats.oversizedAllocations, 0);
  assert.equal(stats.oversizeAllocations, 0);
  assert.equal(stats.totalBytesStaged, 0);
  assert.equal(stats.bytesUploaded, 0);
  assert.equal(stats.totalBytesUploaded, 0);

  // Sanitized JSON-serializable telemetry: no GPUBuffer/ArrayBuffer references, finite bounds.
  const roundTrip = JSON.parse(JSON.stringify(stats));
  assert.equal(roundTrip.hitRate, 0);
  assert.ok(roundTrip.hitRate >= 0 && roundTrip.hitRate <= 1);
  assert.equal(roundTrip.totalBytesStaged, 0);

  // Mutating the returned snapshot must never corrupt internal pool telemetry.
  stats.poolHits = 9999;
  stats.poolMisses = 9999;
  stats.hitRate = 99;
  stats.peakAllocatedBytes = 9999;
  stats.lruEvictions = 9999;
  stats.oversizedAllocations = 9999;
  stats.totalBytesStaged = 9999;
  stats.largeSlabBucketCounts[4 * MiB] = 42;

  const fresh = pool.getStats();
  assert.equal(fresh.poolHits, 0);
  assert.equal(fresh.poolMisses, 0);
  assert.equal(fresh.hitRate, 0);
  assert.equal(fresh.peakAllocatedBytes, 0);
  assert.equal(fresh.lruEvictions, 0);
  assert.equal(fresh.oversizedAllocations, 0);
  assert.equal(fresh.totalBytesStaged, 0);
  assert.deepEqual(fresh.largeSlabBucketCounts, {});

  pool.destroy();
});

test('W245: 4K pow2 upload telemetry tracks poolHits, poolMisses, hitRate, peakAllocatedBytes, oversizedAllocations, and totalBytesStaged with zero cache-hit allocations', async () => {
  const MiB = 1024 * 1024;
  const { device, createdBuffers } = createLightweight4kMockDevice();
  const pool = new GpuStagingPool({
    device,
    chunkSize: 4 * MiB,
    bucketStrategy: BUCKET_STRATEGY.POW2,
    maxPooledBytes: 96 * MiB,
    framesInFlight: 3,
    maxLargeSlabs: 12,
  });

  const encoder = createLightweightEncoder();
  const dstY = { id: 'w245-y' };
  const dstU = { id: 'w245-u' };
  const dstV = { id: 'w245-v' };
  const dstUniform = { id: 'w245-ubo' };

  const yBytes = new Uint8Array(3840 * 2160); // 7.91 MiB (> 4 MiB chunkSize -> oversized, 8 MiB pow2 bucket)
  const uBytes = new Uint8Array(2048 * 1080); // 2.11 MiB (<= 4 MiB chunkSize, 4 MiB pow2 bucket)
  const vBytes = new Uint8Array(2048 * 1080); // 2.11 MiB (<= 4 MiB chunkSize, 4 MiB pow2 bucket)
  const uboBytes = new Uint8Array(512);
  const perFrameStaged = yBytes.byteLength + uBytes.byteLength + vBytes.byteLength + uboBytes.byteLength;

  // 3 warmup frames (all misses) + 12 steady-state frames (100% cache hits).
  const warmupFrames = 3;
  const steadyFrames = 12;
  const totalFrames = warmupFrames + steadyFrames;

  for (let frame = 0; frame < totalFrames; frame += 1) {
    pool.beginFrame(frame);
    pool.stageBufferUpload(encoder, dstUniform, 0, uboBytes);
    // Empty upload is a no-op and must not alter totalBytesStaged or poolMisses.
    pool.stageBufferUpload(encoder, dstUniform, 0, new Uint8Array(0));
    pool.stageTextureUpload(encoder, dstY, yBytes, { width: 3840, height: 2160, bytesPerRow: 3840 });
    pool.stageTextureUpload(encoder, dstU, uBytes, { width: 1920, height: 1080, bytesPerRow: 2048 });
    pool.stageTextureUpload(encoder, dstV, vBytes, { width: 1920, height: 1080, bytesPerRow: 2048 });
    pool.endFrame();

    if (frame === warmupFrames - 1) {
      assert.equal(createdBuffers.length, 12); // 3 * (4 MiB std + 8 MiB Y + 4 MiB U + 4 MiB V)
      const warm = pool.getStats();
      assert.equal(warm.poolMisses, 12);
      assert.equal(warm.poolHits, 0);
      assert.equal(warm.hitRate, 0);
      assert.equal(warm.poolHitRate, 0);
      assert.equal(warm.oversizedAllocations, 3); // Y plane > 4 MiB chunkSize
      assert.equal(warm.oversizeAllocations, 3);
      assert.equal(warm.lruEvictions, 0);
      assert.equal(warm.peakAllocatedBytes, 3 * (4 + 8 + 4 + 4) * MiB);
      assert.equal(warm.totalBytesStaged, warmupFrames * perFrameStaged);
      assert.equal(warm.totalBytesUploaded, warm.totalBytesStaged);
    }
  }

  // Zero extra GPUBuffer allocations during steady-state cache hits!
  assert.equal(createdBuffers.length, 12);
  const steady = pool.getStats();
  const expectedHits = steadyFrames * 4;
  const expectedMisses = warmupFrames * 4;
  assert.equal(steady.poolHits, expectedHits);
  assert.equal(steady.poolMisses, expectedMisses);
  assert.equal(steady.hitRate, expectedHits / (expectedHits + expectedMisses));
  assert.equal(steady.poolHitRate, steady.hitRate);
  assert.ok(steady.hitRate >= 0 && steady.hitRate <= 1);
  assert.equal(steady.peakAllocatedBytes, 3 * (4 + 8 + 4 + 4) * MiB);
  assert.equal(steady.lruEvictions, 0);
  assert.equal(steady.oversizedAllocations, totalFrames); // 1 oversized Y plane per frame
  assert.equal(steady.oversizeAllocations, steady.oversizedAllocations);
  assert.equal(steady.totalBytesStaged, totalFrames * perFrameStaged);
  assert.equal(steady.totalBytesUploaded, steady.totalBytesStaged);

  pool.destroy();

  // Mapped mode direct allocate() staging also tracks totalBytesStaged and hitRate cleanly.
  const mappedMock = createLightweight4kMockDevice({ enableWorkDoneFence: true, allocateBacking: true });
  const mappedPool = new GpuStagingPool({
    device: mappedMock.device,
    chunkSize: 4 * MiB,
    bucketStrategy: BUCKET_STRATEGY.POW2,
    maxPooledBytes: 32 * MiB,
    framesInFlight: 2,
    mode: STAGING_MODE.MAPPED,
  });

  const yDirectSize = Math.ceil(8.3 * MiB); // > 4 MiB chunkSize -> oversized, 16 MiB pow2 bucket
  mappedPool.beginFrame(0);
  const direct0 = mappedPool.allocate(yDirectSize, 256);
  assert.ok(direct0.view instanceof Uint8Array);
  mappedPool.endFrame();

  mappedMock.deferredFences[0].resolve();
  await mappedMock.deferredFences[0].promise;
  await mappedPool.waitForPendingMaps();

  mappedPool.beginFrame(1);
  const direct1 = mappedPool.allocate(yDirectSize, 256);
  assert.equal(direct1.buffer, direct0.buffer);
  assert.equal(mappedMock.createdBuffers.length, 1, 'mapped cache hit must allocate zero new GPUBuffers');
  mappedPool.endFrame();

  const mappedStats = mappedPool.getStats();
  assert.equal(mappedStats.poolMisses, 1);
  assert.equal(mappedStats.poolHits, 1);
  assert.equal(mappedStats.hitRate, 0.5);
  assert.equal(mappedStats.peakAllocatedBytes, 16 * MiB);
  assert.equal(mappedStats.oversizedAllocations, 2);
  assert.equal(mappedStats.totalBytesStaged, yDirectSize * 2);
  assert.equal(mappedStats.lruEvictions, 0);
  mappedPool.destroy();
});

test('W245: lruEvictions tracks per-bucket, global cap, VRAM budget, and idle evictions while ignoring OOM failures and destroy', () => {
  const MiB = 1024 * 1024;
  const { device } = createLightweight4kMockDevice();
  const pool = new GpuStagingPool({
    device,
    chunkSize: 4 * MiB,
    bucketStrategy: BUCKET_STRATEGY.POW2,
    maxPooledBytes: 24 * MiB,
    maxLargeSlabs: 4,
    maxSlabsPerBucket: 2,
    framesInFlight: 1,
    minWarmChunks: 0,
  });

  // 1. Per-bucket cap eviction (3x 4 MiB slabs > maxSlabsPerBucket=2 -> 1 LRU eviction).
  const chromaSize = Math.ceil(2.07 * MiB);
  pool.beginFrame(0);
  pool.allocate(chromaSize, 256);
  pool.allocate(chromaSize, 256);
  pool.allocate(chromaSize, 256);
  pool.endFrame();

  pool.beginFrame(1);
  let stats = pool.getStats();
  assert.equal(stats.lruEvictions, 1);
  assert.equal(stats.evictedCount, 1);
  assert.equal(stats.allocatedBytes, 8 * MiB);
  assert.equal(stats.peakAllocatedBytes, 12 * MiB);
  assert.equal(stats.oversizedAllocations, 0);
  assert.equal(stats.totalBytesStaged, chromaSize * 3);

  // 2. VRAM budget cap eviction: live = 8 MiB (2x 4 MiB idle). Allocate 16 MiB (8.3 MiB > 4 MiB -> oversized)
  //    then allocate 8 MiB (3840x2160 > 4 MiB -> oversized) on frame 2 -> 24 + 8 > 24 MiB forces 2x 4 MiB LRU evictions!
  const y83Size = Math.ceil(8.3 * MiB);
  pool.allocate(y83Size, 256); // 16 MiB bucket; live = 24 MiB, peak = 24 MiB
  pool.endFrame();

  pool.beginFrame(2); // all 3 slabs idle (2x 4 MiB from frame 0, 1x 16 MiB from frame 1)
  const y4kSize = 3840 * 2160; // 8 MiB bucket
  pool.allocate(y4kSize, 256); // evicts both older frame-0 4 MiB slabs (2 LRU evictions), keeps frame-1 16 MiB
  stats = pool.getStats();
  assert.equal(stats.lruEvictions, 3);
  assert.equal(stats.evictedCount, 3);
  assert.equal(stats.allocatedBytes, 24 * MiB);
  assert.equal(stats.peakAllocatedBytes, 24 * MiB);
  assert.equal(stats.oversizedAllocations, 2);
  assert.equal(stats.totalBytesStaged, chromaSize * 3 + y83Size + y4kSize);

  // 3. Over-budget allocation failure must NOT inflate any telemetry counters.
  //    Hold both 16 MiB and 8 MiB active (24 / 24 MiB active), then request another 4 MiB.
  pool.allocate(y83Size, 256); // hits idle 16 MiB slab
  const beforeOom = pool.getStats();
  assert.equal(beforeOom.poolHits, 1);
  assert.equal(beforeOom.oversizedAllocations, 3);
  assert.throws(() => pool.allocate(y4kSize, 256), RangeError);
  const afterOom = pool.getStats();
  assert.equal(afterOom.poolHits, beforeOom.poolHits);
  assert.equal(afterOom.poolMisses, beforeOom.poolMisses);
  assert.equal(afterOom.hitRate, beforeOom.hitRate);
  assert.equal(afterOom.peakAllocatedBytes, beforeOom.peakAllocatedBytes);
  assert.equal(afterOom.lruEvictions, beforeOom.lruEvictions);
  assert.equal(afterOom.oversizedAllocations, beforeOom.oversizedAllocations);
  assert.equal(afterOom.totalBytesStaged, beforeOom.totalBytesStaged);
  pool.endFrame();

  // 4. Idle-frame reclamation increments lruEvictions while preserving peakAllocatedBytes.
  pool.beginFrame(10);
  const reclaimed = pool.reclaimIdleChunks(5);
  assert.equal(reclaimed.reclaimedChunks, 2);
  stats = pool.getStats();
  assert.equal(stats.lruEvictions, 5);
  assert.equal(stats.allocatedBytes, 0);
  assert.equal(stats.peakAllocatedBytes, 24 * MiB);

  // 5. destroy() must NOT increment lruEvictions and keeps peakAllocatedBytes & totalBytesStaged intact.
  pool.allocate(chromaSize, 256);
  const beforeDestroy = pool.getStats();
  pool.destroy();
  const afterDestroy = pool.getStats();
  assert.equal(afterDestroy.lruEvictions, beforeDestroy.lruEvictions);
  assert.equal(afterDestroy.peakAllocatedBytes, 24 * MiB);
  assert.equal(afterDestroy.totalBytesStaged, beforeDestroy.totalBytesStaged);
  assert.equal(afterDestroy.allocatedBytes, 0);
});

test('W245: resetStats zeroes cumulative telemetry cleanly, preserves live pooled buffers, clamps hitRate to 1.0 on post-reset hits, and re-baselines peakAllocatedBytes', () => {
  const MiB = 1024 * 1024;
  const { device, createdBuffers } = createLightweight4kMockDevice();
  const pool = new GpuStagingPool({
    device,
    chunkSize: 4 * MiB,
    bucketStrategy: BUCKET_STRATEGY.POW2,
    maxPooledBytes: 32 * MiB,
    maxSlabsPerBucket: 2,
    framesInFlight: 1,
  });

  const encoder = createLightweightEncoder();
  const dstY = { id: 'reset-y' };
  const dstU = { id: 'reset-u' };
  const yData = new Uint8Array(3840 * 2160); // 8 MiB bucket (> 4 MiB -> oversized)
  const uData = new Uint8Array(2048 * 1080); // 4 MiB bucket

  // Frame 0 spike: 3x 4 MiB slabs + 1x 8 MiB slab = 20 MiB peak; per-bucket cap evicts 1x 4 MiB on recycle -> 16 MiB live.
  pool.beginFrame(0);
  pool.stageTextureUpload(encoder, dstY, yData, { width: 3840, height: 2160, bytesPerRow: 3840 });
  pool.stageTextureUpload(encoder, dstU, uData, { width: 1920, height: 1080, bytesPerRow: 2048 });
  pool.stageTextureUpload(encoder, dstU, uData, { width: 1920, height: 1080, bytesPerRow: 2048 });
  pool.stageTextureUpload(encoder, dstU, uData, { width: 1920, height: 1080, bytesPerRow: 2048 });
  pool.endFrame();

  // Frame 1: recycle & evict 1x 4 MiB slab, then get 2 hits (Y + U).
  pool.beginFrame(1);
  pool.stageTextureUpload(encoder, dstY, yData, { width: 3840, height: 2160, bytesPerRow: 3840 });
  pool.stageTextureUpload(encoder, dstU, uData, { width: 1920, height: 1080, bytesPerRow: 2048 });
  pool.endFrame();

  pool.beginFrame(2); // all 3 remaining slabs (1x 8 MiB + 2x 4 MiB = 16 MiB) are idle
  const preReset = pool.getStats();
  assert.equal(preReset.poolMisses, 4);
  assert.equal(preReset.poolHits, 2);
  assert.equal(preReset.hitRate, 2 / 6);
  assert.equal(preReset.lruEvictions, 1);
  assert.equal(preReset.oversizedAllocations, 2);
  assert.equal(preReset.peakAllocatedBytes, 20 * MiB);
  assert.equal(preReset.allocatedBytes, 16 * MiB);
  assert.equal(preReset.totalBytesStaged, yData.byteLength * 2 + uData.byteLength * 4);
  assert.equal(createdBuffers.length, 4);

  // Reset telemetry while 16 MiB of warm pooled buffers remain live!
  const resetReturn = pool.resetStats();
  const postReset = pool.getStats();
  assert.deepEqual(resetReturn, postReset);

  // All cumulative telemetry counters reset cleanly to 0.
  assert.equal(postReset.poolHits, 0);
  assert.equal(postReset.poolMisses, 0);
  assert.equal(postReset.hitRate, 0);
  assert.equal(postReset.poolHitRate, 0);
  assert.equal(postReset.lruEvictions, 0);
  assert.equal(postReset.evictedCount, 0);
  assert.equal(postReset.evictedBytes, 0);
  assert.equal(postReset.oversizedAllocations, 0);
  assert.equal(postReset.oversizeAllocations, 0);
  assert.equal(postReset.totalBytesStaged, 0);
  assert.equal(postReset.bytesUploaded, 0);
  assert.equal(postReset.totalBytesUploaded, 0);
  assert.equal(postReset.bytesReused, 0);
  assert.equal(postReset.bytesAllocated, 0);
  assert.equal(postReset.allocationCount, 0);
  assert.equal(postReset.largeSlabHits, 0);
  assert.equal(postReset.largeSlabMisses, 0);
  assert.equal(postReset.largeSlabEvictions, 0);
  assert.equal(postReset.reclaimedChunks, 0);
  assert.equal(postReset.reclaimedBytes, 0);
  assert.equal(postReset.framesCompleted, 0);
  assert.equal(postReset.alignPadCount, 0);
  assert.equal(postReset.sabCopyCount, 0);
  assert.equal(postReset.doubleReleaseIgnored, 0);

  // Live pool state is 100% preserved and peakAllocatedBytes re-baselines to current live VRAM (16 MiB).
  assert.equal(postReset.allocatedBytes, 16 * MiB);
  assert.equal(postReset.peakAllocatedBytes, 16 * MiB);
  assert.equal(postReset.peakBytes, 16 * MiB);
  assert.equal(postReset.idleCount, 3);
  assert.deepEqual(postReset.largeSlabBucketCounts, {
    [8 * MiB]: 1,
    [4 * MiB]: 2,
  });

  // Subsequent frames reuse preserved warm slabs -> 0 misses, 100% hitRate === 1.0, 0 new GPUBuffers!
  for (let f = 2; f < 6; f += 1) {
    if (f > 2) {
      pool.beginFrame(f);
    }
    pool.stageTextureUpload(encoder, dstY, yData, { width: 3840, height: 2160, bytesPerRow: 3840 });
    pool.stageTextureUpload(encoder, dstU, uData, { width: 1920, height: 1080, bytesPerRow: 2048 });
    pool.stageTextureUpload(encoder, dstU, uData, { width: 1920, height: 1080, bytesPerRow: 2048 });
    pool.endFrame();
  }

  assert.equal(createdBuffers.length, 4, 'post-reset steady-state frames must create zero new GPUBuffers');
  const steadyPostReset = pool.getStats();
  assert.equal(steadyPostReset.poolHits, 12);
  assert.equal(steadyPostReset.poolMisses, 0);
  assert.equal(steadyPostReset.hitRate, 1);
  assert.equal(steadyPostReset.poolHitRate, 1);
  assert.equal(steadyPostReset.oversizedAllocations, 4);
  assert.equal(steadyPostReset.lruEvictions, 0);
  assert.equal(steadyPostReset.peakAllocatedBytes, 16 * MiB);
  assert.equal(steadyPostReset.totalBytesStaged, 4 * (yData.byteLength + uData.byteLength * 2));

  // New allocation expanding live VRAM from 16 MiB -> 32 MiB advances peakAllocatedBytes cleanly.
  pool.beginFrame(6);
  pool.allocate(Math.ceil(8.3 * MiB), 256); // 16 MiB bucket -> live = 32 MiB
  assert.equal(pool.getStats().peakAllocatedBytes, 32 * MiB);
  assert.equal(pool.getStats().poolMisses, 1);
  assert.equal(pool.getStats().hitRate, 12 / 13);
  pool.endFrame();

  // After destroy(), resetStats() zeroes peakAllocatedBytes as well (allocatedBytes === 0).
  pool.destroy();
  const destroyedReset = pool.resetStats();
  assert.equal(destroyedReset.destroyed, true);
  assert.equal(destroyedReset.allocatedBytes, 0);
  assert.equal(destroyedReset.peakAllocatedBytes, 0);
  assert.equal(destroyedReset.poolHits, 0);
  assert.equal(destroyedReset.poolMisses, 0);
  assert.equal(destroyedReset.hitRate, 0);
  assert.equal(destroyedReset.lruEvictions, 0);
  assert.equal(destroyedReset.oversizedAllocations, 0);
  assert.equal(destroyedReset.totalBytesStaged, 0);
});

test('W245: classic-worker importScripts keeps W245 telemetry helpers isolated inside IIFE', () => {
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

  vm.runInContext(source, ctx, { filename: 'staging_pool.js' });
  assert.equal(typeof ctx.PlayGta6StagingPool.GpuStagingPool.prototype.resetStats, 'function');
  assert.equal(ctx.clampUnitRate, undefined);
  assert.equal(ctx.sanitizeNonNegativeCounter, undefined);
});

// ---------------------------------------------------------------------------
// W283: 30 FPS stall-free streaming — 3-deep frame-in-flight slab recycling
// (`maxFramesInFlight`), bump-pointer sub-allocation coalescing
// (`subAllocCoalesced`), idle oversized-slab shrinking (`idleShrinkFrames`,
// `shrinkIdleOversizedSlabs`, `idleSlabShrinks`), and `stallAvoidedCount`.
// ---------------------------------------------------------------------------

test('W283: constructor options, defaults, validation, and initial telemetry for maxFramesInFlight & idleShrinkFrames', () => {
  assert.equal(DEFAULT_FRAMES_IN_FLIGHT, 3);
  assert.equal(DEFAULT_MAX_FRAMES_IN_FLIGHT, 3);
  assert.equal(DEFAULT_IDLE_SHRINK_FRAMES, 120);

  const { device } = createMockGpuDevice();
  const defaultPool = new GpuStagingPool({ device });
  assert.equal(defaultPool.framesInFlight, 3);
  assert.equal(defaultPool.maxFramesInFlight, 3);
  assert.equal(defaultPool.idleShrinkFrames, 120);

  const initStats = defaultPool.getStats();
  assert.equal(initStats.maxFramesInFlight, 3);
  assert.equal(initStats.idleShrinkFrames, 120);
  assert.equal(initStats.subAllocCoalesced, 0);
  assert.equal(initStats.idleSlabShrinks, 0);
  assert.equal(initStats.stallAvoidedCount, 0);

  // Sanitized JSON round-trip & snapshot immutability.
  const roundTrip = JSON.parse(JSON.stringify(initStats));
  assert.equal(roundTrip.subAllocCoalesced, 0);
  assert.equal(roundTrip.idleSlabShrinks, 0);
  assert.equal(roundTrip.stallAvoidedCount, 0);
  initStats.subAllocCoalesced = 999;
  initStats.idleSlabShrinks = 999;
  initStats.stallAvoidedCount = 999;
  const fresh = defaultPool.getStats();
  assert.equal(fresh.subAllocCoalesced, 0);
  assert.equal(fresh.idleSlabShrinks, 0);
  assert.equal(fresh.stallAvoidedCount, 0);
  defaultPool.destroy();

  // Explicit maxFramesInFlight & idleShrinkFrames options.
  const customPool = new GpuStagingPool({
    device,
    maxFramesInFlight: 4,
    idleShrinkFrames: 60,
  });
  assert.equal(customPool.framesInFlight, 4);
  assert.equal(customPool.maxFramesInFlight, 4);
  assert.equal(customPool.idleShrinkFrames, 60);
  customPool.destroy();

  // Matching framesInFlight & maxFramesInFlight succeeds; mismatching throws.
  const matchedPool = new GpuStagingPool({
    device,
    framesInFlight: 2,
    maxFramesInFlight: 2,
    idleShrinkFrames: 0,
  });
  assert.equal(matchedPool.maxFramesInFlight, 2);
  assert.equal(matchedPool.idleShrinkFrames, 0);
  matchedPool.destroy();

  assert.throws(
    () => new GpuStagingPool({ device, framesInFlight: 2, maxFramesInFlight: 3 }),
    RangeError
  );
  assert.throws(() => new GpuStagingPool({ device, maxFramesInFlight: 0 }), RangeError);
  assert.throws(() => new GpuStagingPool({ device, maxFramesInFlight: -1 }), RangeError);
  assert.throws(() => new GpuStagingPool({ device, maxFramesInFlight: 2.5 }), RangeError);
  assert.throws(() => new GpuStagingPool({ device, idleShrinkFrames: -1 }), RangeError);
  assert.throws(() => new GpuStagingPool({ device, idleShrinkFrames: 1.5 }), RangeError);
});

test('W283: bump-pointer sub-allocation coalesces multiple small/medium uploads into one slab per frame and tracks subAllocCoalesced', () => {
  const { device, createdBuffers } = createMockGpuDevice();
  const pool = new GpuStagingPool({
    device,
    chunkSize: 16 * 1024, // 16 KiB standard slab
    maxPoolBytes: 128 * 1024,
    maxFramesInFlight: 3,
  });

  const encoder = createMockEncoder();
  const dstBuffer = { id: 'dst-coalesced-buf' };
  const dstTexture = { id: 'dst-coalesced-tex' };

  pool.beginFrame(0);

  // 5 small/medium buffer + texture uploads pack into a SINGLE 16 KiB staging GPUBuffer.
  const u1 = pool.stageBufferUpload(encoder, dstBuffer, 0, new Uint8Array(64).fill(1));
  const u2 = pool.stageBufferUpload(encoder, dstBuffer, 64, new Uint8Array(128).fill(2));
  const u3 = pool.stageTextureUpload(encoder, dstTexture, new Uint8Array(16 * 4).fill(3), {
    width: 4,
    height: 4,
    bytesPerRow: 16,
  }); // 256B aligned offset, 1024B staged
  const u4 = pool.stageBufferUpload(encoder, dstBuffer, 256, new Uint8Array(512).fill(4));
  const u5 = pool.allocate(1024, 256);

  assert.equal(createdBuffers.length, 1, 'all 5 uploads must coalesce into 1 GPUBuffer');
  assert.equal(u2.buffer, u1.buffer);
  assert.equal(u3.buffer, u1.buffer);
  assert.equal(u4.buffer, u1.buffer);
  assert.equal(u5.buffer, u1.buffer);
  assert.equal(u1.offset, 0);
  assert.equal(u2.offset, 64);
  assert.equal(u3.offset, 256);
  assert.equal(u3.offset % 256, 0);
  assert.equal(u4.offset, 256 + 1024);
  assert.equal(u5.offset % 256, 0);
  assert.equal(pool.getStats().subAllocCoalesced, 4);

  // Oversized allocation does NOT disrupt _currentChunk or inflate subAllocCoalesced;
  // a following small allocation still coalesces into u1.buffer!
  const big = pool.allocate(20 * 1024, 256);
  assert.notEqual(big.buffer, u1.buffer);
  assert.equal(pool.getStats().subAllocCoalesced, 4);

  const u6 = pool.allocate(256, 4);
  assert.equal(u6.buffer, u1.buffer);
  assert.equal(pool.getStats().subAllocCoalesced, 5);

  // Rollover into a 2nd standard chunk: first alloc in chunk 2 is NOT coalesced, next one IS.
  const rollover = pool.allocate(15 * 1024, 256);
  assert.notEqual(rollover.buffer, u1.buffer);
  assert.notEqual(rollover.buffer, big.buffer);
  assert.equal(rollover.offset, 0);
  assert.equal(pool.getStats().subAllocCoalesced, 5);

  const afterRollover = pool.allocate(128, 4);
  assert.equal(afterRollover.buffer, rollover.buffer);
  assert.equal(afterRollover.offset, 15 * 1024);
  assert.equal(pool.getStats().subAllocCoalesced, 6);

  pool.endFrame();
  pool.destroy();
});

test('W283: 3-deep frame-in-flight ring sustains 30 FPS steady-state streaming with zero buffer churn and stall avoidance (writeBuffer & mapped modes)', async () => {
  // 1. writeBuffer mode: 30 frames @ 30 FPS with 8 uploads per frame coalesced into 1 slab/frame.
  const { device, createdBuffers } = createMockGpuDevice();
  const pool = new GpuStagingPool({
    device,
    chunkSize: 32 * 1024,
    maxPoolBytes: 256 * 1024,
    maxFramesInFlight: 3,
  });

  const encoder = createMockEncoder();
  const dstBuf = { id: 'stream-buf' };
  const payload = new Uint8Array(512).fill(0x5c);

  const totalFrames = 30;
  const uploadsPerFrame = 8;
  for (let frame = 0; frame < totalFrames; frame += 1) {
    pool.beginFrame(frame);
    let firstBuf = null;
    for (let u = 0; u < uploadsPerFrame; u += 1) {
      const res = pool.stageBufferUpload(encoder, dstBuf, u * 512, payload);
      if (u === 0) {
        firstBuf = res.buffer;
      } else {
        assert.equal(res.buffer, firstBuf, 'all uploads in one frame must coalesce into 1 slab');
      }
    }
    pool.endFrame();
  }

  assert.equal(createdBuffers.length, 3, '3-deep ring must allocate strictly 3 slabs for 30 FPS streaming');
  const wbStats = pool.getStats();
  assert.equal(wbStats.poolMisses, 3);
  assert.equal(wbStats.poolHits, totalFrames - 3);
  assert.equal(wbStats.subAllocCoalesced, totalFrames * (uploadsPerFrame - 1));
  // Frame 0 has 0 in-flight; frames 1..29 each acquire a slab while 1–2 frames remain in-flight!
  assert.equal(wbStats.stallAvoidedCount, totalFrames - 1);
  pool.destroy();

  // 2. mapped mode: 3-deep frame ring with deferred GPU fences avoids mapAsync stalls completely.
  const mappedMock = createMockGpuDevice({ enableWorkDoneFence: true });
  const mappedPool = new GpuStagingPool({
    device: mappedMock.device,
    chunkSize: 4096,
    maxPoolBytes: 32 * 1024,
    maxFramesInFlight: 3,
    mode: STAGING_MODE.MAPPED,
  });

  const mappedEnc = createMockEncoder();
  const mappedDst = { id: 'mapped-stream-dst' };

  // Warmup frames 0, 1, 2: GPU fences have NOT resolved yet; 3-deep ring avoids stalling!
  for (let frame = 0; frame < 3; frame += 1) {
    mappedPool.beginFrame(frame);
    const a = mappedPool.stageBufferUpload(mappedEnc, mappedDst, 0, new Uint8Array(64).fill(frame + 1));
    const b = mappedPool.stageBufferUpload(mappedEnc, mappedDst, 64, new Uint8Array(64).fill(frame + 11));
    assert.equal(b.buffer, a.buffer);
    mappedPool.endFrame();
  }

  assert.equal(mappedMock.createdBuffers.length, 3);
  assert.equal(mappedMock.queueWrites.length, 0);
  assert.equal(mappedPool.getStats().inFlightChunks, 3);
  assert.equal(mappedPool.getStats().subAllocCoalesced, 3);
  assert.equal(mappedPool.getStats().stallAvoidedCount, 2); // frames 1 and 2 avoided waiting on in-flight slabs

  // Steady-state frames 3..11: resolve oldest GPU fence (frame - 3) before each frame,
  // keeping 2 frames in-flight while recycling the 3rd mapped slab with zero new allocations!
  for (let frame = 3; frame < 12; frame += 1) {
    const oldestFence = mappedMock.deferredFences[frame - 3];
    oldestFence.resolve();
    await oldestFence.promise;
    await mappedPool.waitForPendingMaps();

    const beginInfo = mappedPool.beginFrame(frame);
    assert.equal(beginInfo.recycledCount, 1);
    assert.equal(beginInfo.inFlightChunks, 2);
    const a = mappedPool.stageBufferUpload(mappedEnc, mappedDst, 0, new Uint8Array(128).fill(0x2a));
    const b = mappedPool.stageBufferUpload(mappedEnc, mappedDst, 128, new Uint8Array(128).fill(0x3b));
    assert.equal(b.buffer, a.buffer);
    mappedPool.endFrame();
  }

  assert.equal(mappedMock.createdBuffers.length, 3, '3-deep mapped ring must allocate zero extra buffers');
  const mStats = mappedPool.getStats();
  assert.equal(mStats.poolHits, 9);
  assert.equal(mStats.poolMisses, 3);
  assert.equal(mStats.subAllocCoalesced, 12);
  assert.equal(mStats.stallAvoidedCount, 11); // frames 1..11 all bypassed in-flight GPU work
  mappedPool.destroy();
});

test('W283: idle oversized-slab shrinking destroys transient boot slabs after idleShrinkFrames while preserving warm standard chunks', async () => {
  const MiB = 1024 * 1024;
  const { device, createdBuffers } = createLightweight4kMockDevice();
  const pool = new GpuStagingPool({
    device,
    chunkSize: 4 * MiB,
    maxPoolBytes: 64 * MiB,
    maxFramesInFlight: 3,
    minWarmChunks: 1,
    idleEvictFrames: 0, // standard warm chunks stay warm
    // default idleShrinkFrames = 120
  });

  // Frame 0 boot burst: 1 standard 4 MiB chunk + 1 oversized 8 MiB Y slab + 1 oversized 16 MiB texture slab.
  pool.beginFrame(0);
  const stdChunk = pool.allocate(1024, 4); // 4 MiB standard chunk
  const bootSlab8 = pool.allocate(3840 * 2160, 256); // 8 MiB large slab (oversized)
  const bootSlab16 = pool.allocate(16 * MiB, 256); // 16 MiB large slab (oversized)
  assert.equal(createdBuffers.length, 3);
  assert.equal(pool.getStats().allocatedBytes, (4 + 8 + 16) * MiB);
  assert.equal(pool.getStats().peakAllocatedBytes, 28 * MiB);
  pool.endFrame();

  // Frame 3: all 3 buffers recycle. Frame 119: idle age 119 < 120 -> nothing shrunk yet.
  pool.beginFrame(3);
  pool.endFrame();
  const begin119 = pool.beginFrame(119);
  assert.equal(begin119.shrunkSlabs, 0);
  assert.equal(begin119.shrunkBytes, 0);
  assert.equal(pool.getStats().idleSlabShrinks, 0);
  assert.equal(bootSlab8.buffer.destroyCount, 0);
  assert.equal(bootSlab16.buffer.destroyCount, 0);
  pool.endFrame();

  // Frame 120: idle age == 120 >= idleShrinkFrames -> both oversized boot slabs destroyed!
  // Standard 4 MiB warm chunk remains alive!
  const begin120 = pool.beginFrame(120);
  assert.equal(begin120.shrunkSlabs, 2);
  assert.equal(begin120.shrunkBytes, 24 * MiB);
  assert.equal(begin120.reclaimedChunks, 2);
  assert.equal(begin120.reclaimedBytes, 24 * MiB);
  assert.equal(bootSlab8.buffer.destroyCount, 1, 'idle 8 MiB boot slab must be shrunk');
  assert.equal(bootSlab16.buffer.destroyCount, 1, 'idle 16 MiB boot slab must be shrunk');
  assert.equal(stdChunk.buffer.destroyCount, 0, 'warm standard chunk must remain alive');

  const afterShrink = pool.getStats();
  assert.equal(afterShrink.idleSlabShrinks, 2);
  assert.equal(afterShrink.allocatedBytes, 4 * MiB);
  assert.equal(afterShrink.largeSlabBytes, 0);
  assert.equal(afterShrink.idleLargeSlabs, 0);
  assert.equal(afterShrink.recycledChunks, 1);
  assert.equal(afterShrink.peakAllocatedBytes, 28 * MiB, 'peak VRAM high-water mark stays intact');
  assert.equal(afterShrink.lruEvictions, 2);
  assert.equal(afterShrink.largeSlabEvictions, 2);
  pool.endFrame();
  pool.destroy();

  // Verify non-bucketed oversized burst chunks shrink even when minWarmChunks = 1 and 0 standard chunks exist,
  // and reusing a slab before idleShrinkFrames resets its idle timer.
  const customMock = createMockGpuDevice();
  const customPool = new GpuStagingPool({
    device: customMock.device,
    chunkSize: 512,
    maxPoolBytes: 8192,
    maxFramesInFlight: 2,
    minWarmChunks: 1,
    idleShrinkFrames: 10,
    largeSlabBuckets: [],
  });

  customPool.beginFrame(0);
  const ov1 = customPool.allocate(1024, 4); // non-bucketed oversized chunk
  customPool.endFrame();

  // Reuse ov1 at frame 6 (refreshes lastUsedFrame = 6).
  customPool.beginFrame(6);
  const ov1Reuse = customPool.allocate(1024, 4);
  assert.equal(ov1Reuse.buffer, ov1.buffer);
  customPool.endFrame();

  // Frame 15 (15 - 6 = 9 < 10): still warm!
  customPool.beginFrame(15);
  assert.equal(customPool.getStats().idleSlabShrinks, 0);
  assert.equal(ov1.buffer.destroyCount, 0);
  customPool.endFrame();

  // Frame 16 (16 - 6 = 10 >= 10): shrunk cleanly down to 0 bytes even though minWarmChunks = 1!
  const b16 = customPool.beginFrame(16);
  assert.equal(b16.shrunkSlabs, 1);
  assert.equal(b16.shrunkBytes, 1024);
  assert.equal(ov1.buffer.destroyCount, 1);
  assert.equal(customPool.getStats().idleSlabShrinks, 1);
  assert.equal(customPool.getStats().allocatedBytes, 0);
  customPool.destroy();

  // Active and in-flight oversized slabs must NEVER be shrunk prematurely;
  // idleShrinkFrames = 0 disables auto-shrink while explicit shrinkIdleOversizedSlabs(N) works.
  const guardMock = createMockGpuDevice({ enableWorkDoneFence: true });
  const guardPool = new GpuStagingPool({
    device: guardMock.device,
    chunkSize: 512,
    maxPoolBytes: 4096,
    maxFramesInFlight: 10,
    idleShrinkFrames: 0,
    largeSlabBuckets: [],
  });

  guardPool.beginFrame(0);
  const inFlightOv = guardPool.allocate(1024, 4);
  guardPool.endFrame();

  guardPool.beginFrame(5);
  const activeOv = guardPool.allocate(1024, 4);
  // Explicit shrink(0) must NOT touch active or in-flight slabs!
  assert.deepEqual(guardPool.shrinkIdleOversizedSlabs(0), { shrunkSlabs: 0, shrunkBytes: 0 });
  assert.equal(inFlightOv.buffer.destroyCount, 0);
  assert.equal(activeOv.buffer.destroyCount, 0);
  // No-arg shrink is a no-op when idleShrinkFrames === 0.
  assert.deepEqual(guardPool.shrinkIdleOversizedSlabs(), { shrunkSlabs: 0, shrunkBytes: 0 });

  // Resolve frame-0 fence and recycleCompleted -> now inFlightOv is recycled and explicit shrink destroys it.
  guardMock.deferredFences[0].resolve();
  await guardMock.deferredFences[0].promise;
  await Promise.resolve();
  assert.equal(guardPool.recycleCompleted(), 1);
  assert.deepEqual(guardPool.shrinkIdleOversizedSlabs(5), { shrunkSlabs: 1, shrunkBytes: 1024 });
  assert.equal(inFlightOv.buffer.destroyCount, 1);
  assert.equal(activeOv.buffer.destroyCount, 0);
  assert.equal(guardPool.getStats().idleSlabShrinks, 1);

  assert.throws(() => guardPool.shrinkIdleOversizedSlabs(-1), RangeError);
  assert.throws(() => guardPool.shrinkIdleOversizedSlabs(1.5), RangeError);
  guardPool.destroy();
  assert.throws(() => guardPool.shrinkIdleOversizedSlabs(0), /destroyed/i);
});

test('W283: resetStats zeroes subAllocCoalesced, idleSlabShrinks, and stallAvoidedCount, OOM never inflates them, and classic-worker IIFE stays isolated', () => {
  const { device } = createMockGpuDevice();
  const pool = new GpuStagingPool({
    device,
    chunkSize: 512,
    maxPoolBytes: 2048,
    maxFramesInFlight: 2,
    idleShrinkFrames: 5,
    largeSlabBuckets: [],
  });

  // Frame 0: 2 coalesced standard allocs + 1 oversized chunk (512 + 1024 = 1536B).
  pool.beginFrame(0);
  pool.allocate(128, 4);
  pool.allocate(128, 4); // coalesced #1
  pool.allocate(1024, 4); // oversized
  pool.endFrame();

  // Frame 1: frame 0 still in-flight -> stall avoided #1; 2nd alloc coalesced #2.
  pool.beginFrame(1);
  pool.allocate(64, 4); // 512B -> total 2048B (at budget cap)
  pool.allocate(64, 4); // coalesced #2

  // OOM allocation must NOT inflate subAllocCoalesced, idleSlabShrinks, or stallAvoidedCount!
  const beforeOom = pool.getStats();
  assert.equal(beforeOom.subAllocCoalesced, 2);
  assert.equal(beforeOom.stallAvoidedCount, 1);
  assert.equal(beforeOom.idleSlabShrinks, 0);
  assert.throws(() => pool.allocate(1024, 4), RangeError);
  const afterOom = pool.getStats();
  assert.equal(afterOom.subAllocCoalesced, beforeOom.subAllocCoalesced);
  assert.equal(afterOom.stallAvoidedCount, beforeOom.stallAvoidedCount);
  assert.equal(afterOom.idleSlabShrinks, beforeOom.idleSlabShrinks);
  pool.endFrame();

  // Frame 6: idle oversized chunk (lastUsedFrame = 0, age 6 >= 5) shrinks -> idleSlabShrinks = 1.
  pool.beginFrame(6);
  const preReset = pool.getStats();
  assert.equal(preReset.subAllocCoalesced, 2);
  assert.equal(preReset.stallAvoidedCount, 1);
  assert.equal(preReset.idleSlabShrinks, 1);
  assert.equal(preReset.recycledChunks, 2); // 2 warm standard 512B chunks preserved

  // resetStats zeroes all three W283 counters while preserving warm buffers & config.
  const postReset = pool.resetStats();
  assert.equal(postReset.subAllocCoalesced, 0);
  assert.equal(postReset.stallAvoidedCount, 0);
  assert.equal(postReset.idleSlabShrinks, 0);
  assert.equal(postReset.maxFramesInFlight, 2);
  assert.equal(postReset.idleShrinkFrames, 5);
  assert.equal(postReset.recycledChunks, 2);
  assert.equal(postReset.allocatedBytes, 1024);
  pool.destroy();

  // Classic-worker importScripts IIFE exports W283 additions cleanly without leaking internals.
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
  vm.runInContext(source, ctx, { filename: 'staging_pool.js' });
  assert.equal(ctx.PlayGta6StagingPool.DEFAULT_MAX_FRAMES_IN_FLIGHT, 3);
  assert.equal(ctx.PlayGta6StagingPool.DEFAULT_IDLE_SHRINK_FRAMES, 120);
  assert.equal(
    typeof ctx.PlayGta6StagingPool.GpuStagingPool.prototype.shrinkIdleOversizedSlabs,
    'function'
  );
  assert.equal(ctx.DEFAULT_IDLE_SHRINK_FRAMES, undefined);
  assert.equal(ctx.DEFAULT_MAX_FRAMES_IN_FLIGHT, undefined);
});

