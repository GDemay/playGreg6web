import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const stagingMod = require('../web/workers/staging_pool.js');

const {
  alignUp,
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

