import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const uniformRingMod = require('../web/workers/uniform_ring.js');

const {
  DEFAULT_UNIFORM_ALIGNMENT,
  DEFAULT_MIN_UNIFORM_BINDING_SIZE,
  DEFAULT_BYTE_SIZE,
  DEFAULT_MAX_BYTE_SIZE,
  DEFAULT_FRAMES_IN_FLIGHT,
  DEFAULT_MAX_BINDING_SIZE,
  WEBGPU_BUFFER_COPY_ALIGNMENT,
  BUFFER_USAGE,
  SHADER_STAGE,
  alignUp,
  alignUniformSliceSize,
  copyForWriteBuffer,
  DynamicUniformRing,
} = uniformRingMod;

/**
 * Create a deterministic mock WebGPU device tracking all buffer allocations,
 * destructions, and `queue.writeBuffer` uploads with strict Chrome WebGPU validation
 * (rejects SharedArrayBuffer directly, requires 4-byte alignment & bounds).
 */
function createMockGpuDevice(options = {}) {
  let nextBufferId = 1;
  const createdBuffers = [];
  const queueWrites = [];

  const queue = {
    writeBuffer(buffer, bufferOffset, data) {
      // Chrome WebGPU rejects SharedArrayBuffer or SAB-backed views directly in writeBuffer.
      if (
        typeof SharedArrayBuffer !== 'undefined' &&
        (data instanceof SharedArrayBuffer ||
          (ArrayBuffer.isView(data) && data.buffer instanceof SharedArrayBuffer))
      ) {
        throw new TypeError(
          "Failed to execute 'writeBuffer' on 'GPUQueue': The provided ArrayBufferView value must not be shared."
        );
      }
      const bytes =
        data instanceof Uint8Array
          ? data
          : ArrayBuffer.isView(data)
            ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
            : new Uint8Array(data);
      assert.equal(
        bytes.byteOffset % 4,
        0,
        'WebGPU queue.writeBuffer source byteOffset must be 4-byte aligned'
      );
      const copy = bytes.slice();
      assert.equal(
        bufferOffset % 4,
        0,
        'WebGPU queue.writeBuffer bufferOffset must be 4-byte aligned'
      );
      assert.equal(
        copy.byteLength % 4,
        0,
        'WebGPU queue.writeBuffer byteLength must be 4-byte aligned'
      );
      assert.ok(
        bufferOffset + copy.byteLength <= buffer.size,
        'WebGPU queue.writeBuffer must stay within destination buffer bounds'
      );
      buffer._data.set(copy, bufferOffset);
      queueWrites.push({
        buffer,
        bufferOffset,
        byteLength: copy.byteLength,
        data: copy,
      });
    },
  };

  const device = {
    limits: options.limits || {
      minUniformBufferOffsetAlignment: 256,
      maxUniformBufferBindingSize: 65536,
    },
    queue,
    createBuffer(descriptor) {
      const id = nextBufferId++;
      const buffer = {
        id,
        label: descriptor.label || `uniform-buf-${id}`,
        size: descriptor.size,
        usage: descriptor.usage,
        _data: new Uint8Array(descriptor.size),
        destroyCount: 0,
        destroy() {
          this.destroyCount += 1;
        },
      };
      createdBuffers.push(buffer);
      return buffer;
    },
  };

  return { device, createdBuffers, queueWrites };
}

test('UMD export attaches to globalThis.PlayGta6UniformRing and module.exports', () => {
  assert.equal(globalThis.PlayGta6UniformRing, uniformRingMod);
  assert.equal(DEFAULT_UNIFORM_ALIGNMENT, 256);
  assert.equal(DEFAULT_BYTE_SIZE, 1024 * 1024);
  assert.equal(DEFAULT_MAX_BYTE_SIZE, 16 * 1024 * 1024);
  assert.equal(DEFAULT_FRAMES_IN_FLIGHT, 3);
  assert.equal(DEFAULT_MAX_BINDING_SIZE, 65536);
  assert.equal(typeof DynamicUniformRing, 'function');
  assert.equal(typeof alignUp, 'function');
});

test('constructor uses device.limits.minUniformBufferOffsetAlignment and validates options', () => {
  const { device: dev512, createdBuffers } = createMockGpuDevice({
    limits: { minUniformBufferOffsetAlignment: 512 },
  });

  const ring512 = new DynamicUniformRing({
    device: dev512,
    byteSize: 2048,
    maxByteSize: 8192,
    framesInFlight: 2,
  });

  assert.equal(ring512.alignment, 512);
  assert.equal(createdBuffers.length, 2);
  for (const buf of createdBuffers) {
    assert.equal(buf.size, 2048);
    assert.equal(buf.usage, BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST);
  }

  ring512.beginFrame();
  const d0 = ring512.pushConstants(new Float32Array([1, 2, 3, 4]));
  const d1 = ring512.pushConstants(new Float32Array([5, 6, 7, 8]));
  const d2 = ring512.pushConstants(new Uint8Array(513).fill(0xaa));
  assert.equal(d0.dynamicOffset, 0);
  assert.equal(d0.alignedSize, 512);
  assert.equal(d1.dynamicOffset, 512);
  assert.equal(d1.alignedSize, 512);
  assert.equal(d2.dynamicOffset, 1024);
  assert.equal(d2.alignedSize, 1024);
  ring512.destroy();

  // Fallback to 256 when device.limits is absent.
  const devNoLimits = {
    queue: { writeBuffer() {} },
    createBuffer(desc) {
      return { size: desc.size, usage: desc.usage, destroy() {} };
    },
  };
  const ringDefault = new DynamicUniformRing({ device: devNoLimits, byteSize: 1024 });
  assert.equal(ringDefault.alignment, 256);
  ringDefault.destroy();

  // Invalid constructor options reject cleanly.
  assert.throws(() => new DynamicUniformRing(null), TypeError);
  assert.throws(() => new DynamicUniformRing({}), TypeError);
  assert.throws(
    () => new DynamicUniformRing({ device: dev512, alignment: 300 }),
    RangeError
  );
  assert.throws(
    () => new DynamicUniformRing({ device: dev512, alignment: 2 }),
    RangeError
  );
  assert.throws(
    () => new DynamicUniformRing({ device: dev512, byteSize: 4096, maxByteSize: 2048 }),
    RangeError
  );
  assert.throws(
    () => new DynamicUniformRing({ device: dev512, maxByteSize: 128, alignment: 256 }),
    RangeError
  );
  assert.throws(
    () => new DynamicUniformRing({ device: dev512, framesInFlight: 0 }),
    RangeError
  );
});

test('256-byte dynamic offset alignment across multiple pushConstants and ArrayBuffer/TypedArray views', () => {
  const { device, queueWrites } = createMockGpuDevice();
  const ring = new DynamicUniformRing({
    device,
    byteSize: 4096,
    maxByteSize: 16384,
    framesInFlight: 3,
  });

  ring.beginFrame();

  // Draw 0: Float32Array (16 bytes -> aligned to 256).
  const mat0 = new Float32Array([1.0, 2.0, 3.0, 4.0]);
  const r0 = ring.pushConstants(mat0);
  assert.equal(r0.dynamicOffset, 0);
  assert.equal(r0.size, 16);
  assert.equal(r0.alignedSize, 256);
  assert.equal(r0.buffer, ring.currentBuffer);

  // Draw 1: ArrayBuffer (64 bytes -> offset 256, aligned to 256).
  const ab1 = new ArrayBuffer(64);
  const f32View1 = new Float32Array(ab1);
  for (let i = 0; i < 16; i += 1) {
    f32View1[i] = i + 10;
  }
  const r1 = ring.pushConstants(ab1);
  assert.equal(r1.dynamicOffset, 256);
  assert.equal(r1.size, 64);
  assert.equal(r1.alignedSize, 256);
  assert.equal(r1.buffer, r0.buffer);

  // Draw 2: Subarray view with non-zero byteOffset (300 bytes -> offset 512, aligned to 512).
  const backing = new Uint8Array(512);
  for (let i = 0; i < backing.length; i += 1) {
    backing[i] = i & 0xff;
  }
  const subView = backing.subarray(40, 340); // 300 bytes
  const r2 = ring.pushConstants(subView);
  assert.equal(r2.dynamicOffset, 512);
  assert.equal(r2.size, 300);
  assert.equal(r2.alignedSize, 512);

  // Draw 3: DataView (12 bytes -> offset 1024, aligned to 256).
  const dvBacking = new ArrayBuffer(32);
  const dv = new DataView(dvBacking, 8, 12);
  dv.setUint32(0, 0xdeadbeef, true);
  dv.setUint32(4, 0xcafebabe, true);
  dv.setUint32(8, 0x12345678, true);
  const r3 = ring.pushConstants(dv);
  assert.equal(r3.dynamicOffset, 1024);
  assert.equal(r3.size, 12);
  assert.equal(r3.alignedSize, 256);

  // Zero GPU uploads before flush().
  assert.equal(queueWrites.length, 0);

  // Single flush uploads exact dirty prefix [0, 1280) in ONE writeBuffer call!
  const flushRes = ring.flush();
  assert.deepEqual(flushRes, { uploadedBytes: 1280, drawCount: 4 });
  assert.equal(queueWrites.length, 1);
  assert.equal(queueWrites[0].buffer, r0.buffer);
  assert.equal(queueWrites[0].bufferOffset, 0);
  assert.equal(queueWrites[0].byteLength, 1280);

  // Verify exact uploaded bytes in GPU buffer for all 4 draws + zero padding.
  const gpuBytes = r0.buffer._data;
  const readDraw0 = new Float32Array(gpuBytes.buffer, 0, 4);
  assert.deepEqual(Array.from(readDraw0), [1.0, 2.0, 3.0, 4.0]);
  for (let p = 16; p < 256; p += 1) {
    assert.equal(gpuBytes[p], 0, `expected zero padding at byte ${p}`);
  }

  const readDraw1 = new Float32Array(gpuBytes.buffer, 256, 16);
  assert.deepEqual(Array.from(readDraw1), Array.from(f32View1));

  assert.deepEqual(
    Array.from(gpuBytes.subarray(512, 512 + 300)),
    Array.from(subView)
  );
  for (let p = 512 + 300; p < 1024; p += 1) {
    assert.equal(gpuBytes[p], 0, `expected zero padding at byte ${p}`);
  }

  const readDv = new DataView(gpuBytes.buffer, 1024, 12);
  assert.equal(readDv.getUint32(0, true), 0xdeadbeef);
  assert.equal(readDv.getUint32(4, true), 0xcafebabe);
  assert.equal(readDv.getUint32(8, true), 0x12345678);

  // Calling flush() a second time with no new pushes is a safe no-op.
  assert.deepEqual(ring.flush(), { uploadedBytes: 0, drawCount: 0 });
  assert.equal(queueWrites.length, 1);

  ring.destroy();
});

test('batches 500 per-draw uniform pushes into ONE single queue.writeBuffer call on flush()', () => {
  const { device, queueWrites } = createMockGpuDevice();
  const ring = new DynamicUniformRing({
    device,
    byteSize: 256 * 1024,
    framesInFlight: 3,
  });

  ring.beginFrame();
  const drawPayload = new Uint32Array(16); // 64 bytes per draw
  const offsets = [];

  for (let draw = 0; draw < 500; draw += 1) {
    drawPayload[0] = draw;
    drawPayload[1] = draw * 3 + 7;
    const alloc = ring.pushConstants(drawPayload);
    offsets.push(alloc.dynamicOffset);
    assert.equal(alloc.dynamicOffset, draw * 256);
    assert.equal(alloc.size, 64);
    assert.equal(alloc.alignedSize, 256);
  }

  assert.equal(queueWrites.length, 0);
  const res = ring.flush();
  assert.deepEqual(res, { uploadedBytes: 500 * 256, drawCount: 500 });
  assert.equal(queueWrites.length, 1, '500 draws must coalesce into 1 queue.writeBuffer call');

  // Spot-check first, middle, and last draw payloads in the uploaded GPU buffer.
  const gpuU32 = new Uint32Array(ring.currentBuffer._data.buffer);
  for (const idx of [0, 123, 499]) {
    const u32Index = offsets[idx] / 4;
    assert.equal(gpuU32[u32Index], idx);
    assert.equal(gpuU32[u32Index + 1], idx * 3 + 7);
  }

  // Empty frame flush issues zero queue.writeBuffer calls.
  ring.beginFrame();
  assert.deepEqual(ring.flush(), { uploadedBytes: 0, drawCount: 0 });
  assert.equal(queueWrites.length, 1);

  ring.destroy();
});

test('multi-frame slot rotation across framesInFlight isolates in-flight GPU buffers', () => {
  const { device, createdBuffers, queueWrites } = createMockGpuDevice();
  const ring = new DynamicUniformRing({
    device,
    byteSize: 1024,
    framesInFlight: 3,
  });

  assert.equal(createdBuffers.length, 3);

  const frameBuffers = [];
  for (let frame = 0; frame < 6; frame += 1) {
    const frameInfo = ring.beginFrame();
    assert.equal(frameInfo.frameIndex, frame);
    assert.equal(frameInfo.slotIndex, frame % 3);

    const marker = new Uint32Array([0xa0000000 + frame, frame * 11]);
    const push = ring.pushConstants(marker);
    assert.equal(push.dynamicOffset, 0);
    assert.equal(push.buffer, frameInfo.buffer);
    frameBuffers.push(push.buffer);

    const flushed = ring.flush();
    assert.deepEqual(flushed, { uploadedBytes: 256, drawCount: 1 });
  }

  // Frames 0, 1, 2 use 3 distinct GPUBuffers so frame N+1 never overwrites frame N!
  assert.notEqual(frameBuffers[0], frameBuffers[1]);
  assert.notEqual(frameBuffers[1], frameBuffers[2]);
  assert.notEqual(frameBuffers[0], frameBuffers[2]);

  // Frames 3, 4, 5 rotate back to slots 0, 1, 2 with zero extra GPUBuffer allocations.
  assert.equal(frameBuffers[3], frameBuffers[0]);
  assert.equal(frameBuffers[4], frameBuffers[1]);
  assert.equal(frameBuffers[5], frameBuffers[2]);
  assert.equal(createdBuffers.length, 3);
  assert.equal(queueWrites.length, 6);

  // Verify each slot retained its latest frame's marker independently.
  assert.equal(new Uint32Array(frameBuffers[0]._data.buffer)[0], 0xa0000003);
  assert.equal(new Uint32Array(frameBuffers[1]._data.buffer)[0], 0xa0000004);
  assert.equal(new Uint32Array(frameBuffers[2]._data.buffer)[0], 0xa0000005);

  ring.destroy();
});

test('auto-grows capacity by doubling up to maxByteSize, preserves prior frame pushes, and rejects overflow', () => {
  const { device, createdBuffers, queueWrites } = createMockGpuDevice();
  const ring = new DynamicUniformRing({
    device,
    byteSize: 512,
    maxByteSize: 2048,
    framesInFlight: 2,
  });

  // Initial allocation: 2 slots x 512B = 1024B.
  assert.equal(createdBuffers.length, 2);
  const initialSlot0 = createdBuffers[0];
  const initialSlot1 = createdBuffers[1];

  ring.beginFrame(); // Frame 0 -> slot 0 (512B)

  const p0 = ring.pushConstants(new Uint32Array([101, 102, 103, 104])); // [0, 256)
  const p1 = ring.pushConstants(new Uint32Array([201, 202, 203, 204])); // [256, 512)
  assert.equal(p0.dynamicOffset, 0);
  assert.equal(p1.dynamicOffset, 256);
  assert.equal(ring.capacity, 512);

  // Draw 2 requires [512, 768) -> doubles 512 -> 1024!
  const p2 = ring.pushConstants(new Uint32Array([301, 302, 303, 304]));
  assert.equal(p2.dynamicOffset, 512);
  assert.equal(ring.capacity, 1024);
  assert.equal(p2.buffer.size, 1024);
  assert.notEqual(p2.buffer, initialSlot0);
  assert.equal(initialSlot0.destroyCount, 1, 'old active slot buffer should be destroyed on grow');
  assert.equal(
    initialSlot1.destroyCount,
    0,
    'other in-flight slot buffer must NOT be destroyed mid-frame'
  );

  // Push 3 more draws to exceed 1024 -> doubles 1024 -> 2048 (maxByteSize).
  const p3 = ring.pushConstants(new Uint32Array([401])); // [768, 1024)
  const p4 = ring.pushConstants(new Uint32Array([501])); // [1024, 1280) -> grows to 2048!
  assert.equal(p3.dynamicOffset, 768);
  assert.equal(p4.dynamicOffset, 1024);
  assert.equal(ring.capacity, 2048);
  assert.equal(p4.buffer.size, 2048);

  // Fill remaining 3 slots up to 2048 bytes: [1280, 1536), [1536, 1792), [1792, 2048).
  ring.pushConstants(new Uint32Array([601]));
  ring.pushConstants(new Uint32Array([701]));
  ring.pushConstants(new Uint32Array([801]));
  assert.equal(ring.usedBytes, 2048);

  // Exceeding maxByteSize must throw a clear RangeError and leave state intact!
  const statsBeforeOverflow = ring.getStats();
  assert.throws(
    () => ring.pushConstants(new Uint32Array([901])),
    (err) => err instanceof RangeError && /maxByteSize|capacity exceeded/i.test(err.message)
  );
  assert.equal(ring.usedBytes, statsBeforeOverflow.usedBytes);
  assert.equal(ring.getStats().frameDrawCount, statsBeforeOverflow.frameDrawCount);

  // Single flush uploads ALL 8 draws (including draws 0..1 pushed before both grows!).
  const flush0 = ring.flush();
  assert.deepEqual(flush0, { uploadedBytes: 2048, drawCount: 8 });
  assert.equal(queueWrites.length, 1);

  const gpuU32 = new Uint32Array(ring.currentBuffer._data.buffer);
  assert.equal(gpuU32[0], 101);
  assert.equal(gpuU32[256 / 4], 201);
  assert.equal(gpuU32[512 / 4], 301);
  assert.equal(gpuU32[768 / 4], 401);
  assert.equal(gpuU32[1024 / 4], 501);
  assert.equal(gpuU32[1280 / 4], 601);
  assert.equal(gpuU32[1536 / 4], 701);
  assert.equal(gpuU32[1792 / 4], 801);

  // Frame 1 rotates to slot 1 and resizes slot 1 lazily from 512B to 2048B.
  const frame1 = ring.beginFrame();
  assert.equal(frame1.slotIndex, 1);
  assert.equal(frame1.byteSize, 2048);
  assert.equal(initialSlot1.destroyCount, 1);
  assert.equal(ring.getStats().allocatedGpuBytes, 4096);
  assert.equal(ring.getStats().growCount, 2);

  ring.destroy();
  for (const buf of createdBuffers) {
    assert.equal(buf.destroyCount, 1, `buffer ${buf.id} must be destroyed exactly once`);
  }
});

test('maxBindingSize and empty payload validation reject invalid pushes cleanly', () => {
  const { device } = createMockGpuDevice();
  const ring = new DynamicUniformRing({
    device,
    byteSize: 4096,
    maxByteSize: 16384,
    maxBindingSize: 512,
  });

  ring.beginFrame();

  // Exact maxBindingSize succeeds.
  const exact = ring.pushConstants(new Uint8Array(512).fill(0x55));
  assert.equal(exact.dynamicOffset, 0);
  assert.equal(exact.size, 512);
  assert.equal(exact.alignedSize, 512);

  // Exceeding maxBindingSize throws RangeError.
  assert.throws(
    () => ring.pushConstants(new Uint8Array(513)),
    (err) => err instanceof RangeError && /maxBindingSize/i.test(err.message)
  );

  // Empty data throws RangeError.
  assert.throws(() => ring.pushConstants(new Uint8Array(0)), RangeError);

  // Invalid input type throws TypeError.
  assert.throws(() => ring.pushConstants('not-a-buffer'), TypeError);

  // Cursor and draw count were not corrupted by failed pushes.
  assert.equal(ring.usedBytes, 512);
  assert.equal(ring.getStats().frameDrawCount, 1);

  ring.destroy();
});

test('createBindGroupLayoutEntry and createBindGroupEntry emit valid hasDynamicOffset entries', () => {
  const { device } = createMockGpuDevice();
  const ring = new DynamicUniformRing({
    device,
    byteSize: 2048,
    maxByteSize: 8192,
    framesInFlight: 3,
    maxBindingSize: 1024,
  });

  // Default visibility (VERTEX | FRAGMENT).
  const layoutDefault = ring.createBindGroupLayoutEntry(0);
  assert.deepEqual(layoutDefault, {
    binding: 0,
    visibility: SHADER_STAGE.VERTEX | SHADER_STAGE.FRAGMENT,
    buffer: {
      type: 'uniform',
      hasDynamicOffset: true,
      minBindingSize: 0,
    },
  });

  // Explicit compute visibility & minBindingSize.
  const layoutCompute = ring.createBindGroupLayoutEntry(2, SHADER_STAGE.COMPUTE, 256);
  assert.deepEqual(layoutCompute, {
    binding: 2,
    visibility: SHADER_STAGE.COMPUTE,
    buffer: {
      type: 'uniform',
      hasDynamicOffset: true,
      minBindingSize: 256,
    },
  });

  // Default sliceSize uses min(maxBindingSize, slot.byteSize).
  ring.beginFrame(); // slot 0
  const entryDefault = ring.createBindGroupEntry(0);
  assert.equal(entryDefault.binding, 0);
  assert.equal(entryDefault.resource.buffer, ring.currentBuffer);
  assert.equal(entryDefault.resource.offset, 0);
  assert.equal(entryDefault.resource.size, 1024);

  // Explicit sliceSize.
  const entryExplicit = ring.createBindGroupEntry(1, 256);
  assert.equal(entryExplicit.binding, 1);
  assert.equal(entryExplicit.resource.buffer, ring.currentBuffer);
  assert.equal(entryExplicit.resource.offset, 0);
  assert.equal(entryExplicit.resource.size, 256);

  // Rotating to frame 1 updates createBindGroupEntry to slot 1's GPUBuffer.
  const f1 = ring.beginFrame();
  const entrySlot1 = ring.createBindGroupEntry(0, 512);
  assert.equal(entrySlot1.resource.buffer, f1.buffer);
  assert.notEqual(entrySlot1.resource.buffer, entryDefault.resource.buffer);
  assert.equal(entrySlot1.resource.size, 512);

  // Validation errors on bind group helpers.
  assert.throws(() => ring.createBindGroupLayoutEntry(-1), RangeError);
  assert.throws(() => ring.createBindGroupLayoutEntry(0, SHADER_STAGE.VERTEX, 2048), RangeError);
  assert.throws(() => ring.createBindGroupEntry(0, 2048), RangeError); // > maxBindingSize
  assert.throws(() => ring.createBindGroupEntry(0, 0), RangeError);
  assert.throws(() => ring.createBindGroupEntry(0, 256, 3), RangeError); // slotIndex out of range

  ring.destroy();
});

test('getStats tracks peak usage and destroy() is idempotent and guards post-destroy calls', () => {
  const { device, createdBuffers } = createMockGpuDevice();
  const ring = new DynamicUniformRing({
    device,
    byteSize: 1024,
    maxByteSize: 4096,
    framesInFlight: 3,
  });

  // Frame 0: 3 draws (768B peak).
  ring.beginFrame();
  ring.pushConstants(new Float32Array(16)); // 64B -> 256
  ring.pushConstants(new Float32Array(16)); // 64B -> 512
  ring.pushConstants(new Float32Array(16)); // 64B -> 768
  ring.flush();

  // Frame 1: 1 draw (256B; peak stays 768B).
  ring.beginFrame();
  ring.pushConstants(new Float32Array(8)); // 32B -> 256
  ring.flush();

  const stats = ring.getStats();
  assert.equal(stats.frameIndex, 1);
  assert.equal(stats.slotIndex, 1);
  assert.equal(stats.usedBytes, 256);
  assert.equal(stats.peakUsedBytes, 768);
  assert.equal(stats.allocatedGpuBytes, 3072);
  assert.equal(stats.peakAllocatedGpuBytes, 3072);
  assert.equal(stats.frameDrawCount, 1);
  assert.equal(stats.totalDrawCount, 4);
  assert.equal(stats.totalBytesPushed, 64 * 3 + 32);
  assert.equal(stats.totalBytesUploaded, 768 + 256);
  assert.equal(stats.flushCount, 2);
  assert.equal(stats.growCount, 0);
  assert.equal(stats.destroyed, false);

  // First destroy destroys all 3 GPUBuffers once.
  ring.destroy();
  assert.equal(ring.currentBuffer, null);
  for (const buf of createdBuffers) {
    assert.equal(buf.destroyCount, 1);
  }

  const afterDestroy = ring.getStats();
  assert.equal(afterDestroy.destroyed, true);
  assert.equal(afterDestroy.allocatedGpuBytes, 0);
  assert.equal(afterDestroy.peakUsedBytes, 768);
  assert.equal(afterDestroy.totalDrawCount, 4);

  // Second destroy is a safe idempotent no-op.
  ring.destroy();
  for (const buf of createdBuffers) {
    assert.equal(buf.destroyCount, 1);
  }

  // Operations after destroy throw cleanly.
  assert.throws(() => ring.beginFrame(), /destroyed/i);
  assert.throws(() => ring.pushConstants(new Uint8Array(16)), /destroyed/i);
  assert.throws(() => ring.flush(), /destroyed/i);
  assert.throws(() => ring.createBindGroupEntry(0), /destroyed/i);
  assert.throws(() => ring.createBindGroupLayoutEntry(0), /destroyed/i);
});

test('supports mid-frame incremental flush and full re-upload when a grow occurs after flush', () => {
  const { device, queueWrites } = createMockGpuDevice();
  const ring = new DynamicUniformRing({
    device,
    byteSize: 512,
    maxByteSize: 2048,
    framesInFlight: 2,
  });

  // First push without explicit beginFrame() auto-initializes frame 0 / slot 0.
  const a = ring.pushConstants(new Uint32Array([11, 22]));
  assert.equal(ring.frameIndex, 0);
  assert.equal(ring.slotIndex, 0);
  assert.equal(a.dynamicOffset, 0);

  // Mid-frame flush uploads [0, 256).
  assert.deepEqual(ring.flush(), { uploadedBytes: 256, drawCount: 1 });
  assert.equal(queueWrites.length, 1);
  assert.equal(queueWrites[0].bufferOffset, 0);
  assert.equal(queueWrites[0].byteLength, 256);

  // Second push fits in [256, 512) -> incremental flush uploads only [256, 512).
  const b = ring.pushConstants(new Uint32Array([33, 44]));
  assert.equal(b.dynamicOffset, 256);
  assert.deepEqual(ring.flush(), { uploadedBytes: 256, drawCount: 1 });
  assert.equal(queueWrites.length, 2);
  assert.equal(queueWrites[1].bufferOffset, 256);
  assert.equal(queueWrites[1].byteLength, 256);

  // Third push triggers grow (new 1024B GPUBuffer) -> next flush uploads full [0, 768) from offset 0!
  const c = ring.pushConstants(new Uint32Array([55, 66]));
  assert.equal(c.dynamicOffset, 512);
  assert.deepEqual(ring.flush(), { uploadedBytes: 768, drawCount: 3 });
  assert.equal(queueWrites.length, 3);
  assert.equal(queueWrites[2].buffer, c.buffer);
  assert.equal(queueWrites[2].bufferOffset, 0);
  assert.equal(queueWrites[2].byteLength, 768);

  const u32 = new Uint32Array(c.buffer._data.buffer);
  assert.equal(u32[0], 11);
  assert.equal(u32[256 / 4], 33);
  assert.equal(u32[512 / 4], 55);

  ring.destroy();
});

test('G43: enforces device.limits.minUniformBufferOffsetAlignment (default 256B) even when caller passes smaller alignment', () => {
  const { device: dev256 } = createMockGpuDevice({
    limits: { minUniformBufferOffsetAlignment: 256 },
  });
  const ringClamped256 = new DynamicUniformRing({
    device: dev256,
    alignment: 64,
    byteSize: 1024,
    framesInFlight: 2,
  });
  assert.equal(ringClamped256.alignment, 256);
  assert.equal(ringClamped256.deviceMinUniformBufferOffsetAlignment, 256);
  ringClamped256.beginFrame();
  const a0 = ringClamped256.pushConstants(new Uint32Array([1, 2, 3, 4]));
  const a1 = ringClamped256.pushConstants(new Uint32Array([5, 6, 7, 8]));
  assert.equal(a0.dynamicOffset % 256, 0);
  assert.equal(a1.dynamicOffset % 256, 0);
  assert.equal(a1.dynamicOffset, 256);
  ringClamped256.destroy();

  const { device: dev512 } = createMockGpuDevice({
    limits: { minUniformBufferOffsetAlignment: 512 },
  });
  const ringClamped512 = new DynamicUniformRing({
    device: dev512,
    alignment: 256,
    byteSize: 2048,
    framesInFlight: 2,
  });
  assert.equal(ringClamped512.alignment, 512);
  ringClamped512.beginFrame();
  const b0 = ringClamped512.pushConstants(new Uint8Array(16).fill(1));
  const b1 = ringClamped512.pushConstants(new Uint8Array(16).fill(2));
  assert.equal(b0.dynamicOffset, 0);
  assert.equal(b1.dynamicOffset, 512);
  ringClamped512.destroy();

  // Invalid / non-power-of-two device limits fall back or round up safely.
  const { device: devInvalid } = createMockGpuDevice({
    limits: { minUniformBufferOffsetAlignment: NaN },
  });
  const ringFallback = new DynamicUniformRing({ device: devInvalid, byteSize: 1024 });
  assert.equal(ringFallback.alignment, DEFAULT_UNIFORM_ALIGNMENT);
  ringFallback.destroy();

  const { device: devQuirk } = createMockGpuDevice({
    limits: { minUniformBufferOffsetAlignment: 300 },
  });
  const ringQuirk = new DynamicUniformRing({ device: devQuirk, byteSize: 2048 });
  assert.equal(ringQuirk.alignment, 512);
  ringQuirk.destroy();
});

test('G43: pads uniform slice sizes to 4-byte multiples with 16B minimum (or minUniformBindingSize) and supports empty/tiny Frame-1 CBVs', () => {
  assert.equal(DEFAULT_MIN_UNIFORM_BINDING_SIZE, 16);
  assert.equal(WEBGPU_BUFFER_COPY_ALIGNMENT, 4);
  assert.equal(alignUniformSliceSize(0), 16);
  assert.equal(alignUniformSliceSize(1), 16);
  assert.equal(alignUniformSliceSize(12), 16);
  assert.equal(alignUniformSliceSize(16), 16);
  assert.equal(alignUniformSliceSize(17), 20);
  assert.equal(alignUniformSliceSize(21), 24);
  assert.equal(alignUniformSliceSize(12, 32), 32);
  assert.equal(alignUniformSliceSize(33, 32), 36);

  const { device, queueWrites } = createMockGpuDevice();
  const ring = new DynamicUniformRing({
    device,
    byteSize: 2048,
    maxByteSize: 8192,
    framesInFlight: 2,
  });

  ring.beginFrame();

  // Tiny 5-byte CBV -> paddedSize 16, alignedSize 256, zero-padded tail.
  const tiny = ring.pushConstants(new Uint8Array([0x11, 0x22, 0x33, 0x44, 0x55]));
  assert.equal(tiny.size, 5);
  assert.equal(tiny.paddedSize, 16);
  assert.equal(tiny.alignedSize, 256);
  assert.equal(tiny.dynamicOffset, 0);
  assert.equal(tiny.overflow, false);

  // Odd 19-byte CBV -> paddedSize 20, alignedSize 256.
  const odd = ring.pushConstants(new Uint8Array(19).fill(0x7a));
  assert.equal(odd.size, 19);
  assert.equal(odd.paddedSize, 20);
  assert.equal(odd.alignedSize, 256);
  assert.equal(odd.dynamicOffset, 256);

  // Empty CBV for Frame-1 DrawInstanced(3, 1) via pushCbv() and pushConstants(..., { allowEmpty: true }).
  const empty0 = ring.pushCbv(new Uint8Array(0));
  assert.equal(empty0.size, 0);
  assert.equal(empty0.paddedSize, 16);
  assert.equal(empty0.alignedSize, 256);
  assert.equal(empty0.dynamicOffset, 512);

  const empty1 = ring.pushCbv(null);
  assert.equal(empty1.size, 0);
  assert.equal(empty1.paddedSize, 16);
  assert.equal(empty1.alignedSize, 256);
  assert.equal(empty1.dynamicOffset, 768);

  const empty2 = ring.pushConstants(new Uint8Array(0), { allowEmpty: true });
  assert.equal(empty2.size, 0);
  assert.equal(empty2.paddedSize, 16);
  assert.equal(empty2.dynamicOffset, 1024);

  // createBindGroupEntry pads tiny/unaligned sliceSize to 4B multiple >= 16B.
  const bgTiny = ring.createBindGroupEntry(0, 5);
  assert.equal(bgTiny.resource.size, 16);
  const bgOdd = ring.createBindGroupEntry(1, 17);
  assert.equal(bgOdd.resource.size, 20);
  const bgAlloc = ring.createBindGroupEntryForAllocation(2, empty0);
  assert.equal(bgAlloc.binding, 2);
  assert.equal(bgAlloc.resource.buffer, empty0.buffer);
  assert.equal(bgAlloc.resource.offset, 0);
  assert.equal(bgAlloc.resource.size, 16);

  // Flush and verify exact bytes + zero padding for all 5 draws.
  assert.deepEqual(ring.flush(), { uploadedBytes: 1280, drawCount: 5 });
  assert.equal(queueWrites.length, 1);
  const gpuBytes = tiny.buffer._data;
  assert.deepEqual(Array.from(gpuBytes.subarray(0, 5)), [0x11, 0x22, 0x33, 0x44, 0x55]);
  for (let i = 5; i < 256; i += 1) {
    assert.equal(gpuBytes[i], 0);
  }
  for (let i = 512; i < 1280; i += 1) {
    assert.equal(gpuBytes[i], 0, `empty CBV slice byte ${i} must be zero-filled`);
  }

  ring.destroy();

  // Constructor allowEmptyPayloads + custom minUniformBindingSize (32B).
  const ringCustom = new DynamicUniformRing({
    device,
    byteSize: 1024,
    minUniformBindingSize: 32,
    allowEmptyPayloads: true,
  });
  assert.equal(ringCustom.minUniformBindingSize, 32);
  assert.equal(ringCustom.getStats().minUniformBindingSize, 32);
  const c0 = ringCustom.pushConstants(new Uint8Array(0));
  assert.equal(c0.paddedSize, 32);
  assert.equal(c0.alignedSize, 256);
  assert.equal(ringCustom.createBindGroupEntry(0, 12).resource.size, 32);
  assert.equal(ringCustom.createBindGroupEntry(0, 33).resource.size, 36);
  ringCustom.destroy();
});

test('G43: safely copies SharedArrayBuffer-backed and non-4-byte-aligned views into regular ArrayBuffer before queue.writeBuffer', () => {
  const { device, queueWrites } = createMockGpuDevice();
  const ring = new DynamicUniformRing({
    device,
    byteSize: 4096,
    framesInFlight: 2,
  });

  // Direct helper tests on copyForWriteBuffer.
  const directSab = new SharedArrayBuffer(16);
  new Uint32Array(directSab).set([0x11111111, 0x22222222, 0x33333333, 0x44444444]);
  const sabPrepared = copyForWriteBuffer(new Uint8Array(directSab));
  assert.equal(sabPrepared.copiedFromSab, true);
  assert.equal(sabPrepared.copiedForAlignment, false);
  assert.ok(sabPrepared.bytes.buffer instanceof ArrayBuffer);
  assert.equal(sabPrepared.bytes.byteLength, 16);

  const unalignedAb = new ArrayBuffer(32);
  const unalignedView = new Uint8Array(unalignedAb, 3, 13);
  unalignedView.fill(0xab);
  const unalignedPrepared = copyForWriteBuffer(unalignedView);
  assert.equal(unalignedPrepared.copiedFromSab, false);
  assert.equal(unalignedPrepared.copiedForAlignment, true);
  assert.equal(unalignedPrepared.bytes.byteOffset, 0);
  assert.equal(unalignedPrepared.bytes.byteLength, 16);
  assert.deepEqual(Array.from(unalignedPrepared.bytes.subarray(13, 16)), [0, 0, 0]);

  const alignedAb = new Uint8Array(new ArrayBuffer(16));
  const fastPath = copyForWriteBuffer(alignedAb);
  assert.equal(fastPath.copiedFromSab, false);
  assert.equal(fastPath.copiedForAlignment, false);
  assert.equal(fastPath.bytes, alignedAb, 'aligned ArrayBuffer Uint8Array should take zero-copy fast path');

  // End-to-end ring pushes with SAB, SAB Float32Array, SAB unaligned subarray, and unaligned AB view.
  ring.beginFrame();

  // 1. Direct SharedArrayBuffer (16B)
  const sab0 = new SharedArrayBuffer(16);
  new Float32Array(sab0).set([1.5, 2.5, 3.5, 4.5]);
  const r0 = ring.pushConstants(sab0);
  assert.equal(r0.dynamicOffset, 0);
  assert.equal(r0.size, 16);
  assert.equal(r0.paddedSize, 16);

  // Mutate source SAB immediately after pushConstants() to verify snapshot isolation before flush().
  new Float32Array(sab0).set([99, 99, 99, 99]);

  // 2. Unaligned Uint8Array view into SharedArrayBuffer (byteOffset = 3, byteLength = 13)
  const sab1 = new SharedArrayBuffer(32);
  const sab1Bytes = new Uint8Array(sab1);
  for (let i = 0; i < 13; i += 1) {
    sab1Bytes[3 + i] = 0x40 + i;
  }
  const sabUnaligned = new Uint8Array(sab1, 3, 13);
  const r1 = ring.pushConstants(sabUnaligned);
  assert.equal(r1.dynamicOffset, 256);
  assert.equal(r1.size, 13);
  assert.equal(r1.paddedSize, 16);

  // 3. DataView over SharedArrayBuffer (8B)
  const sab2 = new SharedArrayBuffer(24);
  const sabDv = new DataView(sab2, 4, 8);
  sabDv.setUint32(0, 0xfeedface, true);
  sabDv.setUint32(4, 0x0badc0de, true);
  const r2 = ring.pushConstants(sabDv);
  assert.equal(r2.dynamicOffset, 512);
  assert.equal(r2.size, 8);
  assert.equal(r2.paddedSize, 16);

  // 4. Non-4-byte-aligned regular ArrayBuffer view (byteOffset = 5, byteLength = 7)
  const r3 = ring.pushConstants(new Uint8Array(unalignedAb, 5, 7));
  assert.equal(r3.dynamicOffset, 768);
  assert.equal(r3.size, 7);
  assert.equal(r3.paddedSize, 16);

  // Flush must pass strict Chrome WebGPU writeBuffer checks (non-SAB, 4B aligned).
  const flushed = ring.flush();
  assert.deepEqual(flushed, { uploadedBytes: 1024, drawCount: 4 });
  assert.equal(queueWrites.length, 1);

  const gpuData = r0.buffer._data;
  assert.deepEqual(Array.from(new Float32Array(gpuData.buffer, 0, 4)), [1.5, 2.5, 3.5, 4.5]);
  for (let i = 0; i < 13; i += 1) {
    assert.equal(gpuData[256 + i], 0x40 + i);
  }
  for (let i = 13; i < 256; i += 1) {
    assert.equal(gpuData[256 + i], 0);
  }
  const readDv = new DataView(gpuData.buffer, 512, 8);
  assert.equal(readDv.getUint32(0, true), 0xfeedface);
  assert.equal(readDv.getUint32(4, true), 0x0badc0de);

  const stats = ring.getStats();
  assert.equal(stats.sabCopyCount, 3, 'expected 3 SharedArrayBuffer copies');
  assert.equal(stats.unalignedCopyCount, 2, 'expected 2 unaligned input copies');

  ring.destroy();
});

test('G43: mid-frame grow after prior flush keeps previously-flushed buffer alive until slot wrap reset', () => {
  const { device, createdBuffers } = createMockGpuDevice();
  const ring = new DynamicUniformRing({
    device,
    byteSize: 512,
    maxByteSize: 2048,
    framesInFlight: 2,
  });

  const initialSlot0 = createdBuffers[0];
  const initialSlot1 = createdBuffers[1];

  // Frame 0 -> slot 0: push 2 draws and flush to initialSlot0.
  ring.beginFrame();
  ring.pushConstants(new Uint32Array([1, 2, 3, 4]));
  ring.pushConstants(new Uint32Array([5, 6, 7, 8]));
  ring.flush();
  assert.equal(initialSlot0.destroyCount, 0);

  // Draw 3 overflows 512B -> grows slot 0 to 1024B.
  // Because initialSlot0 was already flushed this frame, it MUST stay alive until slot 0 resets!
  const p2 = ring.pushConstants(new Uint32Array([9, 10, 11, 12]));
  assert.equal(p2.dynamicOffset, 512);
  assert.notEqual(p2.buffer, initialSlot0);
  assert.equal(
    initialSlot0.destroyCount,
    0,
    'previously-flushed buffer must stay alive for the remainder of the frame'
  );
  assert.equal(ring.getStats().growCount, 1);
  assert.equal(ring.getStats().overflowAllocCount, 1);
  assert.equal(ring.getStats().overflowBufferCount, 1);
  ring.flush();

  // Frame 1 -> slot 1: Frame 0 is still in flight, so initialSlot0 MUST STILL stay alive!
  ring.beginFrame();
  assert.equal(initialSlot0.destroyCount, 0, 'previous frame retired buffer must stay alive during frame 1');
  assert.equal(initialSlot1.destroyCount, 1, 'slot 1 resized to 1024B on its turn');
  assert.equal(ring.getStats().wrapCount, 0);

  // Frame 2 -> wraps back to slot 0: now Frame 0 completed, so initialSlot0 is destroyed cleanly!
  ring.beginFrame();
  assert.equal(initialSlot0.destroyCount, 1, 'retired buffer must be destroyed on slot wrap reset');
  assert.equal(ring.getStats().wrapCount, 1);
  assert.equal(ring.getStats().overflowBufferCount, 0);

  ring.destroy();
  for (const buf of createdBuffers) {
    assert.equal(buf.destroyCount, 1, `buffer ${buf.id} must be destroyed exactly once`);
  }
});

test('G43: allowOverflow recovers from single-frame CBV bursts beyond maxByteSize, batches overflow uploads, and preserves in-flight buffers until reset', () => {
  const { device, createdBuffers, queueWrites } = createMockGpuDevice();
  const ring = new DynamicUniformRing({
    device,
    byteSize: 512,
    maxByteSize: 1024,
    framesInFlight: 2,
    allowOverflow: true,
  });

  ring.beginFrame(); // Frame 0 -> slot 0

  // Push 10 draws (2560B total):
  // - Draws 0..1 fit in initial 512B
  // - Draw 2 grows primary slot 0 to 1024B (overflowAllocCount = 1, growCount = 1)
  // - Draw 3 fills primary slot 0 (1024B)
  // - Draws 4..7 spill into overflow chunk #0 (1024B, overflowAllocCount = 2)
  // - Draws 8..9 spill into overflow chunk #1 (1024B, overflowAllocCount = 3)
  const allocs = [];
  for (let i = 0; i < 10; i += 1) {
    let payload;
    if (i === 5) {
      // Also mix a SharedArrayBuffer payload into the overflow chunk!
      const sab = new SharedArrayBuffer(16);
      new Uint32Array(sab).set([0x1000 + i, 0x2000 + i, 0x3000 + i, 0x4000 + i]);
      payload = sab;
    } else {
      payload = new Uint32Array([0x1000 + i, 0x2000 + i, 0x3000 + i, 0x4000 + i]);
    }
    allocs.push(ring.pushConstants(payload));
  }

  for (let i = 0; i < 4; i += 1) {
    assert.equal(allocs[i].overflow, false);
    assert.equal(allocs[i].dynamicOffset, i * 256);
  }
  for (let i = 4; i < 8; i += 1) {
    assert.equal(allocs[i].overflow, true);
    assert.equal(allocs[i].buffer, allocs[4].buffer);
    assert.equal(allocs[i].dynamicOffset, (i - 4) * 256);
  }
  for (let i = 8; i < 10; i += 1) {
    assert.equal(allocs[i].overflow, true);
    assert.equal(allocs[i].buffer, allocs[8].buffer);
    assert.equal(allocs[i].dynamicOffset, (i - 8) * 256);
  }
  assert.notEqual(allocs[4].buffer, allocs[0].buffer);
  assert.notEqual(allocs[8].buffer, allocs[4].buffer);

  // Bind group helper works seamlessly on overflow allocations too.
  const overflowBind = ring.createBindGroupEntryForAllocation(0, allocs[5]);
  assert.equal(overflowBind.resource.buffer, allocs[5].buffer);
  assert.equal(overflowBind.resource.offset, 0);
  assert.equal(overflowBind.resource.size, 16);

  // Single flush uploads primary (1024B) + overflow chunk 0 (1024B) + overflow chunk 1 (512B).
  const flush0 = ring.flush();
  assert.deepEqual(flush0, { uploadedBytes: 2560, drawCount: 10 });
  assert.equal(queueWrites.length, 3);

  // Verify every single draw's uploaded data arrived at the exact dynamicOffset in its GPUBuffer
  // (primary draws live in ring.currentBuffer after mid-frame grow; overflow draws in their chunk).
  for (let i = 0; i < 10; i += 1) {
    const a = allocs[i];
    const targetBuf = a.overflow ? a.buffer : ring.currentBuffer;
    const u32 = new Uint32Array(targetBuf._data.buffer, a.dynamicOffset, 4);
    assert.deepEqual(Array.from(u32), [0x1000 + i, 0x2000 + i, 0x3000 + i, 0x4000 + i]);
  }

  const statsF0 = ring.getStats();
  assert.equal(statsF0.usedBytes, 2560);
  assert.equal(statsF0.peakUsedBytes, 2560);
  assert.equal(statsF0.growCount, 1);
  assert.equal(statsF0.overflowAllocCount, 3);
  assert.equal(statsF0.overflowBufferCount, 2);
  assert.equal(statsF0.sabCopyCount, 1);

  // Calling flush again with no new draws is a safe no-op.
  assert.deepEqual(ring.flush(), { uploadedBytes: 0, drawCount: 0 });
  assert.equal(queueWrites.length, 3);

  const overflowBuf0 = allocs[4].buffer;
  const overflowBuf1 = allocs[8].buffer;

  // Frame 1 -> slot 1: Frame 0's overflow buffers MUST remain alive while Frame 0 is in flight!
  ring.beginFrame();
  assert.equal(overflowBuf0.destroyCount, 0);
  assert.equal(overflowBuf1.destroyCount, 0);
  assert.equal(ring.usedBytes, 0);
  ring.pushConstants(new Uint32Array([0xdead, 0xbeef, 1, 2]));
  assert.deepEqual(ring.flush(), { uploadedBytes: 256, drawCount: 1 });

  // Frame 2 -> wraps back to slot 0: Frame 0's overflow buffers are now destroyed cleanly!
  ring.beginFrame();
  assert.equal(overflowBuf0.destroyCount, 1);
  assert.equal(overflowBuf1.destroyCount, 1);
  assert.equal(ring.getStats().overflowBufferCount, 0);
  assert.equal(ring.getStats().wrapCount, 1);
  assert.equal(ring.getStats().allocatedGpuBytes, 2048); // 2 slots x 1024B

  ring.destroy();
  for (const buf of createdBuffers) {
    assert.equal(buf.destroyCount, 1, `buffer ${buf.id} must be destroyed exactly once`);
  }
});

test('G43: preserves G31 classic-worker importScripts IIFE isolation with zero top-level lexical leaks', () => {
  const source = readFileSync(new URL('../web/workers/uniform_ring.js', import.meta.url), 'utf8');
  const ctx = vm.createContext({ console });
  ctx.self = ctx;

  // Load twice or alongside colliding top-level const declarations to confirm full IIFE isolation.
  vm.runInContext(source, ctx, { filename: 'uniform_ring.js' });
  vm.runInContext('const BUFFER_USAGE = {}; const SHADER_STAGE = {}; const alignUp = () => 0;', ctx);

  assert.ok(ctx.PlayGta6UniformRing, 'must attach exports to globalThis.PlayGta6UniformRing');
  assert.equal(typeof ctx.PlayGta6UniformRing.DynamicUniformRing, 'function');
  assert.equal(typeof ctx.PlayGta6UniformRing.alignUniformSliceSize, 'function');
  assert.equal(typeof ctx.PlayGta6UniformRing.copyForWriteBuffer, 'function');
  assert.equal(ctx.PlayGta6UniformRing.DEFAULT_MIN_UNIFORM_BINDING_SIZE, 16);
});
