import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const ringMod = require('../web/workers/gpu_ring.js');
const wgpuMod = require('../web/workers/wgpu_worker.js');

const {
  GpuRingBuffer,
  GPU_RING_MAGIC,
  GPU_RING_VERSION,
  GPU_RING_HEADER_BYTES,
  GPU_PACKET_HEADER_BYTES,
  DEFAULT_BATCH_MAX_PACKETS,
  HEADER_INDEX,
} = ringMod;

const {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  OPCODES,
  SHADER_STAGE,
  PRIMITIVE_TOPOLOGY,
  BUFFER_USAGE,
} = wgpuMod;

class BinaryCommandBuilder {
  constructor() {
    this.chunks = [];
  }

  _pushPacket(opcode, payloadBytes) {
    const buf = new ArrayBuffer(8 + payloadBytes.byteLength);
    const view = new DataView(buf);
    view.setUint16(0, opcode, true);
    view.setUint16(2, 0, true);
    view.setUint32(4, payloadBytes.byteLength, true);
    new Uint8Array(buf, 8).set(payloadBytes);
    this.chunks.push(new Uint8Array(buf));
  }

  beginFrame() {
    this._pushPacket(OPCODES.BEGIN_FRAME, new Uint8Array(0));
    return this;
  }

  clearColor(r, g, b, a) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setFloat32(0, r, true);
    view.setFloat32(4, g, true);
    view.setFloat32(8, b, true);
    view.setFloat32(12, a, true);
    this._pushPacket(OPCODES.CLEAR_COLOR, new Uint8Array(buf));
    return this;
  }

  createBuffer(id, usageFlags, byteSize) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, usageFlags, true);
    view.setBigUint64(8, BigInt(byteSize), true);
    this._pushPacket(OPCODES.CREATE_BUFFER, new Uint8Array(buf));
    return this;
  }

  writeBuffer(id, dstOffset, bytes) {
    const src = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    const buf = new ArrayBuffer(16 + src.byteLength);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, dstOffset, true);
    view.setUint32(8, src.byteLength, true);
    view.setUint32(12, 0, true);
    new Uint8Array(buf, 16).set(src);
    this._pushPacket(OPCODES.WRITE_BUFFER, new Uint8Array(buf));
    return this;
  }

  createShaderModule(id, stage, wgslCode) {
    const encoded = new TextEncoder().encode(wgslCode);
    const buf = new ArrayBuffer(16 + encoded.byteLength);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, stage, true);
    view.setUint32(8, encoded.byteLength, true);
    view.setUint32(12, 0, true);
    new Uint8Array(buf, 16).set(encoded);
    this._pushPacket(OPCODES.CREATE_SHADER_MODULE, new Uint8Array(buf));
    return this;
  }

  createRenderPipeline(
    id,
    vsShaderId,
    fsShaderId,
    vertexStride,
    topology = PRIMITIVE_TOPOLOGY.TRIANGLE_LIST,
    flags = 0
  ) {
    const buf = new ArrayBuffer(24);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, vsShaderId, true);
    view.setUint32(8, fsShaderId, true);
    view.setUint32(12, vertexStride, true);
    view.setUint32(16, topology, true);
    view.setUint32(20, flags, true);
    this._pushPacket(OPCODES.CREATE_RENDER_PIPELINE, new Uint8Array(buf));
    return this;
  }

  setPipeline(pipelineId) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, pipelineId, true);
    this._pushPacket(OPCODES.SET_PIPELINE, new Uint8Array(buf));
    return this;
  }

  setVertexBuffer(slot, bufferId, offset = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, slot, true);
    view.setUint32(4, bufferId, true);
    view.setBigUint64(8, BigInt(offset), true);
    this._pushPacket(OPCODES.SET_VERTEX_BUFFER, new Uint8Array(buf));
    return this;
  }

  draw(vertexCount, instanceCount = 1, firstVertex = 0, firstInstance = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, vertexCount, true);
    view.setUint32(4, instanceCount, true);
    view.setUint32(8, firstVertex, true);
    view.setUint32(12, firstInstance, true);
    this._pushPacket(OPCODES.DRAW, new Uint8Array(buf));
    return this;
  }

  present() {
    this._pushPacket(OPCODES.PRESENT, new Uint8Array(0));
    return this;
  }

  destroyResource(id) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, id, true);
    this._pushPacket(OPCODES.DESTROY_RESOURCE, new Uint8Array(buf));
    return this;
  }

  finish() {
    const total = this.chunks.reduce((sum, c) => sum + c.byteLength, 0);
    const out = new Uint8Array(total);
    let offset = 0;
    for (const chunk of this.chunks) {
      out.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return out;
  }
}

function createMockWebGpu() {
  const calls = [];
  let nextHandle = 1;
  const mockTextureView = { kind: 'GPUTextureView', id: 'swapchain-view-0' };

  const ctx = {
    getCurrentTexture() {
      return {
        createView() {
          return mockTextureView;
        },
      };
    },
  };

  const device = {
    createBuffer(desc) {
      const buffer = {
        kind: 'GPUBuffer',
        handle: nextHandle++,
        size: desc.size,
        usage: desc.usage,
        destroyed: false,
        destroy() {
          this.destroyed = true;
          calls.push({ op: 'buffer.destroy', buffer });
        },
      };
      calls.push({ op: 'createBuffer', desc: { ...desc }, result: buffer });
      return buffer;
    },

    createShaderModule(desc) {
      const module = {
        kind: 'GPUShaderModule',
        handle: nextHandle++,
        code: desc.code,
      };
      calls.push({ op: 'createShaderModule', desc: { ...desc }, result: module });
      return module;
    },

    createRenderPipeline(desc) {
      const pipeline = {
        kind: 'GPURenderPipeline',
        handle: nextHandle++,
        desc,
      };
      calls.push({ op: 'createRenderPipeline', desc, result: pipeline });
      return pipeline;
    },

    createCommandEncoder() {
      const commandBuffer = { kind: 'GPUCommandBuffer', handle: nextHandle++ };
      return {
        beginRenderPass(desc) {
          calls.push({ op: 'beginRenderPass', desc });
          return {
            setPipeline(pipeline) {
              calls.push({ op: 'setPipeline', pipeline });
            },
            setVertexBuffer(slot, buffer, offset) {
              calls.push({ op: 'setVertexBuffer', slot, buffer, offset });
            },
            draw(vertexCount, instanceCount, firstVertex, firstInstance) {
              calls.push({
                op: 'draw',
                vertexCount,
                instanceCount,
                firstVertex,
                firstInstance,
              });
            },
            end() {
              calls.push({ op: 'renderPass.end' });
            },
          };
        },
        finish() {
          calls.push({ op: 'encoder.finish', result: commandBuffer });
          return commandBuffer;
        },
      };
    },

    queue: {
      writeBuffer(buffer, dstOffset, data) {
        calls.push({
          op: 'writeBuffer',
          buffer,
          dstOffset,
          data: new Uint8Array(data),
        });
      },
      submit(commandBuffers) {
        calls.push({ op: 'submit', commandBuffers });
      },
    },
  };

  return { device, ctx, calls, mockTextureView };
}

test('UMD export and 64-byte GpuRingHeader binary layout match runtime/gpu/gpu_ring.h', () => {
  assert.equal(globalThis.PlayGta6GpuRing.GpuRingBuffer, GpuRingBuffer);
  assert.equal(GPU_RING_MAGIC, 0x47505552);
  assert.equal(GPU_RING_VERSION, 1);
  assert.equal(GPU_RING_HEADER_BYTES, 64);

  const ring = GpuRingBuffer.create(128);
  assert.equal(ring.sab.byteLength, 64 + 128);

  const raw = new DataView(ring.sab);
  assert.equal(raw.getUint32(0, true), 0x47505552); // magic
  assert.equal(raw.getUint32(4, true), 1);          // version
  assert.equal(raw.getUint32(8, true), 128);        // capacity_bytes
  assert.equal(raw.getUint32(12, true), 0);         // reserved0
  assert.equal(raw.getUint32(16, true), 0);         // write_pos
  assert.equal(raw.getUint32(20, true), 0);         // read_pos
  assert.equal(raw.getUint32(24, true), 0);         // dropped_packets
  assert.equal(raw.getUint32(28, true), 0);         // notify_seq
  for (let off = 32; off < 64; off += 4) {
    assert.equal(raw.getUint32(off, true), 0);
  }

  assert.throws(() => GpuRingBuffer.create(96), RangeError);
  assert.throws(() => GpuRingBuffer.create(0), RangeError);
  assert.throws(() => GpuRingBuffer.attach(new SharedArrayBuffer(128)), /Invalid GpuRingHeader magic/);
});

test('GpuRingBuffer handles wrap-around, zero-copy contiguous drain, and full-buffer backpressure', () => {
  const producer = GpuRingBuffer.create(64);
  const consumer = GpuRingBuffer.attach(producer.sab);

  assert.equal(producer.availableBytes(), 0);
  assert.equal(producer.freeBytes(), 64);

  // Write 48 bytes -> contiguous drain should be zero-copy directly on SAB.
  const first = Uint8Array.from({ length: 48 }, (_, i) => (i + 1) & 0xff);
  assert.equal(producer.writeBytes(first), true);
  assert.equal(producer.availableBytes(), 48);
  assert.equal(producer.freeBytes(), 16);
  assert.equal(producer.notifySeq, 1);

  // Backpressure: 17 bytes does not fit into 16 free bytes.
  const rejected = new Uint8Array(17).fill(0xee);
  assert.equal(producer.writeBytes(rejected), false);
  assert.equal(producer.droppedPackets, 1);
  assert.equal(producer.availableBytes(), 48);
  assert.equal(producer.freeBytes(), 16);

  // Oversized write (> capacity) returns false without corrupting ring.
  assert.equal(producer.writeBytes(new Uint8Array(65)), false);
  assert.equal(producer.droppedPackets, 2);

  let drainedFirst = null;
  let wasZeroCopy = false;
  const n1 = consumer.drainTo((bytes) => {
    wasZeroCopy = bytes.buffer === producer.sab;
    drainedFirst = bytes.slice();
  });
  assert.equal(n1, 48);
  assert.equal(wasZeroCopy, true);
  assert.deepEqual(drainedFirst, first);
  assert.equal(consumer.availableBytes(), 0);
  assert.equal(producer.freeBytes(), 64);

  // Write 40 bytes at offset 48 -> wraps across [48..64) and [0..24).
  const wrapped = Uint8Array.from({ length: 40 }, (_, i) => (0xa0 + i) & 0xff);
  const tail = Uint8Array.from({ length: 24 }, (_, i) => (0x50 + i) & 0xff);
  assert.equal(producer.writeBytes(wrapped), true);
  assert.equal(producer.writeBytes(tail), true);
  assert.equal(producer.availableBytes(), 64);
  assert.equal(producer.freeBytes(), 0);

  // Full ring rejects even 1 byte and keeps existing data intact.
  assert.equal(producer.writeBytes(Uint8Array.of(0xff)), false);
  assert.equal(producer.droppedPackets, 3);

  let drainedWrapped = null;
  const n2 = consumer.drainTo((bytes) => {
    drainedWrapped = bytes.slice();
  });
  assert.equal(n2, 64);
  assert.deepEqual(drainedWrapped.subarray(0, 40), wrapped);
  assert.deepEqual(drainedWrapped.subarray(40, 64), tail);
  assert.equal(consumer.availableBytes(), 0);
  assert.equal(producer.freeBytes(), 64);

  // Verify 32-bit monotonic counter wraparound across 0xFFFFFFFF -> 0.
  const headerU32 = new Uint32Array(producer.sab, 0, 16);
  Atomics.store(headerU32, HEADER_INDEX.WRITE_POS, 0xfffffff0);
  Atomics.store(headerU32, HEADER_INDEX.READ_POS, 0xfffffff0);

  const wrap32 = Uint8Array.from({ length: 32 }, (_, i) => (i ^ 0x5a) & 0xff);
  assert.equal(producer.writeBytes(wrap32), true);
  assert.equal(producer.writePos, 0x00000010);
  assert.equal(consumer.availableBytes(), 32);

  let wrap32Out = null;
  const n3 = consumer.drainTo((bytes) => {
    wrap32Out = bytes.slice();
  });
  assert.equal(n3, 32);
  assert.equal(consumer.readPos, 0x00000010);
  assert.deepEqual(wrap32Out, wrap32);
});

test('Zero-corruption streaming of wrapped GpuPacketHeader commands into WgpuCommandProcessor', () => {
  const { device, ctx, calls } = createMockWebGpu();
  const processor = new WgpuCommandProcessor(device, ctx, { format: 'bgra8unorm' });

  // Small 512-byte ring forces repeated wrap-arounds across multi-packet frames.
  const producer = GpuRingBuffer.create(512);
  processor.attachRing(producer.sab);

  const vertexFloats = new Float32Array([
    0.0, 0.5, 0.0,
    -0.5, -0.5, 0.0,
    0.5, -0.5, 0.0,
  ]);
  const vertexBytes = new Uint8Array(vertexFloats.buffer);
  const vsWgsl = '@vertex fn vs_main(@location(0) p: vec3<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(p, 1.0); }';
  const fsWgsl = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.2, 0.6, 1.0, 1.0); }';

  const frameStream = new BinaryCommandBuilder()
    .beginFrame()
    .createBuffer(1, BUFFER_USAGE.VERTEX | BUFFER_USAGE.COPY_DST, vertexBytes.byteLength)
    .writeBuffer(1, 0, vertexBytes)
    .createShaderModule(2, SHADER_STAGE.VERTEX, vsWgsl)
    .createShaderModule(3, SHADER_STAGE.FRAGMENT, fsWgsl)
    .createRenderPipeline(4, 2, 3, 12, PRIMITIVE_TOPOLOGY.TRIANGLE_LIST, 0)
    .clearColor(0.25, 0.5, 0.75, 1.0)
    .setPipeline(4)
    .setVertexBuffer(0, 1, 0)
    .draw(3, 1, 0, 0)
    .present()
    .destroyResource(1)
    .finish();

  assert.ok(frameStream.byteLength > 300 && frameStream.byteLength < 512);

  // Stream 10 frames so every frame starts at a different ring offset and wraps across 512B.
  for (let frameIdx = 1; frameIdx <= 10; frameIdx++) {
    calls.length = 0;
    assert.equal(producer.writeBytes(frameStream), true);

    // Backpressure: a second full frame cannot fit in the 512B ring while the first is queued.
    assert.equal(producer.writeBytes(frameStream), false);

    const res = processor.drainRing();
    assert.deepEqual(res, {
      frameCount: frameIdx,
      presents: 1,
      bytesDrained: frameStream.byteLength,
    });
    assert.equal(producer.availableBytes(), 0);

    assert.deepEqual(
      calls.map((c) => c.op),
      [
        'createBuffer',
        'writeBuffer',
        'createShaderModule',
        'createShaderModule',
        'createRenderPipeline',
        'beginRenderPass',
        'setPipeline',
        'setVertexBuffer',
        'draw',
        'renderPass.end',
        'encoder.finish',
        'submit',
        'buffer.destroy',
      ]
    );
    assert.deepEqual(calls[1].data, vertexBytes);
    assert.equal(calls[2].desc.code, vsWgsl);
    assert.equal(calls[3].desc.code, fsWgsl);
    assert.deepEqual(calls[5].desc.colorAttachments[0].clearValue, {
      r: 0.25,
      g: 0.5,
      b: 0.75,
      a: 1.0,
    });
    assert.equal(calls[8].vertexCount, 3);
  }

  assert.equal(producer.droppedPackets, 10);
});

test('wgpu_worker message handler supports attachRing, drainRing, injected GpuRingBuffer, and legacy cmdBuffer', async () => {
  const { device, ctx, calls } = createMockWebGpu();
  const posted = [];
  let injectedConstructed = 0;

  class InjectedGpuRing extends GpuRingBuffer {
    constructor(sab, opts) {
      super(sab, opts);
      injectedConstructed += 1;
    }
  }

  const producer = GpuRingBuffer.create(256);
  const worker = createWgpuWorkerHandler({
    postMessage: (msg) => posted.push(msg),
    GpuRingBuffer: InjectedGpuRing,
    initGpu: async () => new WgpuCommandProcessor(device, ctx, { format: 'bgra8unorm' }),
  });

  // Attach ring BEFORE init to verify pending ring binding carries over once GPU initializes.
  await worker.handleMessage({ attachRing: true, ringSab: producer.sab });
  assert.equal(injectedConstructed, 1);
  assert.deepEqual(posted.shift(), { ringAttached: true, capacityBytes: 256 });

  await worker.handleMessage({ init: true, canvas: { width: 0, height: 0 }, width: 640, height: 360 });
  assert.ok(worker.processor);
  assert.equal(worker.processor.ring, worker.ring);

  // Stream a ring frame that wraps around the 256B ring.
  const padStream = new BinaryCommandBuilder().beginFrame().clearColor(0.1, 0.1, 0.1, 1.0).finish();
  for (let i = 0; i < 6; i++) {
    assert.equal(producer.writeBytes(padStream), true);
    await worker.handleMessage({ drainRing: true });
  }
  assert.equal(posted.length, 0); // no Present yet

  const presentStream = new BinaryCommandBuilder()
    .beginFrame()
    .clearColor(0.9, 0.2, 0.1, 1.0)
    .present()
    .finish();
  assert.equal(producer.writeBytes(presentStream), true);
  await worker.handleMessage({ drainRing: true });

  assert.deepEqual(posted.shift(), { frame: 1 });
  const frame1Telemetry = posted.shift();
  assert.equal(frame1Telemetry.type, 'gpu-telemetry');
  assert.equal(frame1Telemetry.frameStats.framesPresented, 1);
  assert.equal(worker.processor.frameCount, 1);
  assert.ok(calls.some((c) => c.op === 'submit'));

  // Backward-compatible { cmdBuffer } path still works alongside attached ring.
  const legacyStream = new BinaryCommandBuilder()
    .beginFrame()
    .clearColor(0.0, 1.0, 0.0, 1.0)
    .present()
    .finish();
  await worker.handleMessage({ cmdBuffer: legacyStream.buffer });
  assert.deepEqual(posted.shift(), { frame: 2 });
  assert.equal(worker.processor.frameCount, 2);
});

function encodeRawPacket(opcode, flags, payloadBytes) {
  const payload =
    payloadBytes instanceof Uint8Array
      ? payloadBytes
      : new Uint8Array(payloadBytes || 0);
  const out = new Uint8Array(8 + payload.byteLength);
  const view = new DataView(out.buffer);
  view.setUint16(0, opcode, true);
  view.setUint16(2, flags, true);
  view.setUint32(4, payload.byteLength, true);
  out.set(payload, 8);
  return out;
}

test('W187 drainBatch decodes packets, enforces maxPackets/default 256 cap, and publishes READ_POS once per batch', () => {
  assert.equal(GPU_PACKET_HEADER_BYTES, 8);
  assert.equal(DEFAULT_BATCH_MAX_PACKETS, 256);
  assert.equal(GpuRingBuffer.PACKET_HEADER_BYTES, 8);
  assert.equal(GpuRingBuffer.DEFAULT_BATCH_MAX_PACKETS, 256);

  const producer = GpuRingBuffer.create(16384);
  const consumer = GpuRingBuffer.attach(producer.sab);

  // Argument validation.
  assert.throws(() => consumer.drainBatch(null), TypeError);
  assert.throws(() => consumer.drainBatch(() => {}, 0), RangeError);
  assert.throws(() => consumer.drainBatch(() => {}, -4), RangeError);
  assert.throws(() => consumer.drainBatch(() => {}, 1.5), RangeError);
  assert.throws(() => consumer.drainBatch(() => {}, NaN), RangeError);
  assert.throws(() => consumer.drainBatch(() => {}, Infinity), RangeError);
  assert.throws(() => consumer.drainBatch(() => {}, '8'), RangeError);

  // Empty ring returns zero-work summary without incrementing batchesDrained.
  const emptyRes = consumer.drainBatch(() => {
    throw new Error('should not be called on empty ring');
  });
  assert.deepEqual(emptyRes, { packets: 0, bytes: 0, remainingBytes: 0, wrapped: false });
  assert.equal(consumer.getStats().batchesDrained, 0);
  assert.equal(consumer.getStats().avgBatchSize, 0);

  // Enqueue 10 heterogeneous packets (including 0-byte payloads and non-zero flags).
  const expected = [];
  for (let i = 0; i < 10; i++) {
    const opcode = (i % 22) + 1;
    const flags = (i * 7) & 0xffff;
    const payloadLen = i % 3 === 0 ? 0 : (i * 4);
    const payload = Uint8Array.from({ length: payloadLen }, (_, b) => ((i * 31 + b) & 0xff));
    const pkt = encodeRawPacket(opcode, flags, payload);
    assert.equal(producer.writeBytes(pkt), true);
    expected.push({ opcode, flags, payload, raw: pkt });
  }

  // Drain bounded to 4 packets and verify exactly ONE Atomics.store and ONE Atomics.notify on READ_POS.
  const origStore = Atomics.store;
  const origNotify = Atomics.notify;
  let readPosStores = 0;
  let readPosNotifies = 0;
  Atomics.store = (ta, idx, val) => {
    if (idx === HEADER_INDEX.READ_POS) {
      readPosStores += 1;
    }
    return origStore(ta, idx, val);
  };
  Atomics.notify = (ta, idx, count) => {
    if (idx === HEADER_INDEX.READ_POS) {
      readPosNotifies += 1;
    }
    return origNotify(ta, idx, count);
  };

  try {
    const batch1 = [];
    const r1 = consumer.drainBatch((pkt, idx) => {
      assert.equal(idx, batch1.length);
      assert.equal(pkt.bytes.buffer, producer.sab, 'contiguous batch packet must be zero-copy');
      assert.equal(pkt.payload.buffer, producer.sab, 'contiguous payload must be zero-copy');
      assert.equal(pkt.wrapped, false);
      batch1.push({
        opcode: pkt.opcode,
        flags: pkt.flags,
        payloadBytes: pkt.payloadBytes,
        payload: pkt.payload.slice(),
        bytes: pkt.bytes.slice(),
        byteLength: pkt.byteLength,
      });
    }, 4);

    assert.equal(r1.packets, 4);
    assert.equal(r1.wrapped, false);
    assert.ok(r1.remainingBytes > 0);
    assert.equal(readPosStores, 1, 'drainBatch must advance read_pos with a single Atomics.store');
    assert.equal(readPosNotifies, 1, 'drainBatch must issue a single Atomics.notify per batch');

    for (let i = 0; i < 4; i++) {
      assert.equal(batch1[i].opcode, expected[i].opcode);
      assert.equal(batch1[i].flags, expected[i].flags);
      assert.equal(batch1[i].payloadBytes, expected[i].payload.byteLength);
      assert.equal(batch1[i].byteLength, expected[i].raw.byteLength);
      assert.deepEqual(batch1[i].payload, expected[i].payload);
      assert.deepEqual(batch1[i].bytes, expected[i].raw);
    }

    // Drain next 4 and final 2 packets.
    const rest = [];
    const r2 = consumer.drainBatch((pkt) => rest.push(pkt.opcode), 4);
    const r3 = consumer.drainBatch((pkt) => rest.push(pkt.opcode), 4);
    assert.equal(r2.packets, 4);
    assert.equal(r3.packets, 2);
    assert.equal(r3.remainingBytes, 0);
    assert.deepEqual(
      rest,
      expected.slice(4).map((e) => e.opcode)
    );
    assert.equal(readPosStores, 3);
    assert.equal(readPosNotifies, 3);
  } finally {
    Atomics.store = origStore;
    Atomics.notify = origNotify;
  }

  const stats = consumer.getStats();
  assert.equal(stats.batchesDrained, 3);
  assert.equal(stats.packetsDrained, 10);
  assert.equal(stats.maxBatchSize, 4);
  assert.equal(stats.avgBatchSize, 10 / 3);

  // Default maxPackets = 256 cap check: enqueue 300 12-byte packets (3600 bytes).
  consumer.reset();
  for (let i = 0; i < 300; i++) {
    const payload = new Uint8Array(4);
    new DataView(payload.buffer).setUint32(0, i, true);
    assert.equal(producer.writeBytes(encodeRawPacket(OPCODES.DRAW, 0, payload)), true);
  }
  const seenIds = [];
  const d1 = consumer.drainBatch((pkt) => {
    seenIds.push(new DataView(pkt.payload.buffer, pkt.payload.byteOffset, 4).getUint32(0, true));
  });
  assert.equal(d1.packets, 256);
  assert.equal(d1.bytes, 256 * 12);
  assert.equal(d1.remainingBytes, 44 * 12);

  const d2 = consumer.drainBatch((pkt) => {
    seenIds.push(new DataView(pkt.payload.buffer, pkt.payload.byteOffset, 4).getUint32(0, true));
  });
  assert.equal(d2.packets, 44);
  assert.equal(d2.bytes, 44 * 12);
  assert.equal(d2.remainingBytes, 0);
  assert.equal(seenIds.length, 300);
  for (let i = 0; i < 300; i++) {
    assert.equal(seenIds[i], i);
  }
  assert.equal(consumer.getStats().batchesDrained, 2);
  assert.equal(consumer.getStats().maxBatchSize, 256);
  assert.equal(consumer.getStats().avgBatchSize, 150);
  assert.equal(consumer.getStats().packetsDrained, 300);
});

test('W187 drainBatch handles wrap-around batches (header split, payload split, exact boundary, and u32 counter wrap)', () => {
  const producer = GpuRingBuffer.create(64);
  const consumer = GpuRingBuffer.attach(producer.sab);

  // Case A: Advance read/write to offset 60 so an 8-byte GpuPacketHeader splits [60..64) + [0..4).
  assert.equal(producer.writeBytes(new Uint8Array(60)), true);
  assert.equal(consumer.drainTo(() => {}), 60);
  assert.equal(consumer.readPos, 60);

  const pA0 = encodeRawPacket(6, 0x1234, Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8])); // 16B: [60..64) + [0..12)
  const pA1 = encodeRawPacket(7, 0x5678, Uint8Array.from([9, 10, 11, 12]));          // 12B: [12..24)
  const pA2 = encodeRawPacket(9, 0x00ff, new Uint8Array(0));                         // 8B:  [24..32)
  assert.equal(producer.writeBytes(pA0), true);
  assert.equal(producer.writeBytes(pA1), true);
  assert.equal(producer.writeBytes(pA2), true);

  const receivedA = [];
  const resA = consumer.drainBatch((pkt) => {
    receivedA.push({
      opcode: pkt.opcode,
      flags: pkt.flags,
      payload: pkt.payload.slice(),
      bytes: pkt.bytes.slice(),
      wrapped: pkt.wrapped,
      zeroCopy: pkt.bytes.buffer === producer.sab,
    });
  });

  assert.equal(resA.packets, 3);
  assert.equal(resA.bytes, 36);
  assert.equal(resA.remainingBytes, 0);
  assert.equal(resA.wrapped, true);
  assert.equal(receivedA[0].opcode, 6);
  assert.equal(receivedA[0].flags, 0x1234);
  assert.deepEqual(receivedA[0].payload, Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8]));
  assert.deepEqual(receivedA[0].bytes, pA0);
  assert.equal(receivedA[0].wrapped, true);
  assert.equal(receivedA[0].zeroCopy, false);
  // Subsequent packets after wrap remain zero-copy directly on SAB.
  assert.equal(receivedA[1].opcode, 7);
  assert.equal(receivedA[1].flags, 0x5678);
  assert.deepEqual(receivedA[1].payload, Uint8Array.from([9, 10, 11, 12]));
  assert.equal(receivedA[1].wrapped, false);
  assert.equal(receivedA[1].zeroCopy, true);
  assert.equal(receivedA[2].opcode, 9);
  assert.equal(receivedA[2].flags, 0x00ff);
  assert.equal(receivedA[2].payload.byteLength, 0);
  assert.equal(receivedA[2].wrapped, false);
  assert.equal(receivedA[2].zeroCopy, true);

  // Case B: Header fits at [56..64), payload wraps into [0..12).
  // Current readPos = 96 (offset 32). Advance 24 bytes -> offset 56.
  assert.equal(producer.writeBytes(new Uint8Array(24)), true);
  assert.equal(consumer.drainTo(() => {}), 24);
  assert.equal(consumer.readPos & 63, 56);

  const pB0 = encodeRawPacket(4, 0xaa55, Uint8Array.from({ length: 12 }, (_, i) => 0x80 + i)); // 20B
  const pB1 = encodeRawPacket(2, 0x0001, Uint8Array.from([0xfe, 0xed, 0xfa, 0xce]));           // 12B
  assert.equal(producer.writeBytes(pB0), true);
  assert.equal(producer.writeBytes(pB1), true);

  const receivedB = [];
  const resB = consumer.drainBatch((pkt) => {
    receivedB.push({
      opcode: pkt.opcode,
      flags: pkt.flags,
      payload: pkt.payload.slice(),
      wrapped: pkt.wrapped,
    });
  });
  assert.equal(resB.packets, 2);
  assert.equal(resB.bytes, 32);
  assert.equal(resB.wrapped, true);
  assert.equal(receivedB[0].opcode, 4);
  assert.equal(receivedB[0].flags, 0xaa55);
  assert.deepEqual(receivedB[0].payload, Uint8Array.from({ length: 12 }, (_, i) => 0x80 + i));
  assert.equal(receivedB[0].wrapped, true);
  assert.equal(receivedB[1].opcode, 2);
  assert.deepEqual(receivedB[1].payload, Uint8Array.from([0xfe, 0xed, 0xfa, 0xce]));
  assert.equal(receivedB[1].wrapped, false);

  // Case C: Exact-boundary wrap between two contiguous packets + 32-bit 0xFFFFFFFF -> 0 counter wrap.
  const headerU32 = new Uint32Array(producer.sab, 0, 16);
  Atomics.store(headerU32, HEADER_INDEX.WRITE_POS, 0xfffffff0); // offset 48
  Atomics.store(headerU32, HEADER_INDEX.READ_POS, 0xfffffff0);

  const pC0 = encodeRawPacket(10, 1, Uint8Array.from([10, 20, 30, 40, 50, 60, 70, 80])); // 16B: [48..64)
  const pC1 = encodeRawPacket(11, 2, Uint8Array.from([81, 82, 83, 84]));                  // 12B: [0..12)
  assert.equal(producer.writeBytes(pC0), true);
  assert.equal(producer.writeBytes(pC1), true);
  assert.equal(producer.writePos, 0x0000000c);

  const receivedC = [];
  const resC = consumer.drainBatch((pkt) => {
    receivedC.push({
      opcode: pkt.opcode,
      payload: pkt.payload.slice(),
      wrapped: pkt.wrapped,
      zeroCopy: pkt.bytes.buffer === producer.sab,
    });
  });
  assert.equal(resC.packets, 2);
  assert.equal(resC.bytes, 28);
  assert.equal(resC.wrapped, true);
  assert.equal(consumer.readPos, 0x0000000c);
  assert.equal(receivedC[0].opcode, 10);
  assert.equal(receivedC[0].wrapped, false);
  assert.equal(receivedC[0].zeroCopy, true);
  assert.equal(receivedC[1].opcode, 11);
  assert.equal(receivedC[1].wrapped, false);
  assert.equal(receivedC[1].zeroCopy, true);
});

test('W187 drainBatch error mid-batch commits only completed packets and preserves remaining stream', () => {
  const producer = GpuRingBuffer.create(256);
  const consumer = GpuRingBuffer.attach(producer.sab);

  // Advance to offset 220 so the 8-packet batch also spans across the 256B wrap boundary.
  assert.equal(producer.writeBytes(new Uint8Array(220)), true);
  assert.equal(consumer.drainTo(() => {}), 220);

  const packets = [];
  for (let i = 0; i < 8; i++) {
    const payload = Uint8Array.from({ length: 8 }, (_, b) => (i * 16 + b) & 0xff);
    const pkt = encodeRawPacket(100 + i, i, payload); // 16 bytes each
    assert.equal(producer.writeBytes(pkt), true);
    packets.push({ opcode: 100 + i, flags: i, payload, byteLength: pkt.byteLength });
  }

  // Case 1: Throw on index 0 -> 0 packets committed, READ_POS unchanged, no batch recorded.
  const readPosBeforeZeroFail = consumer.readPos;
  const origStore = Atomics.store;
  let storeCalls = 0;
  Atomics.store = (ta, idx, val) => {
    if (idx === HEADER_INDEX.READ_POS) {
      storeCalls += 1;
    }
    return origStore(ta, idx, val);
  };
  try {
    assert.throws(
      () => {
        consumer.drainBatch(() => {
          throw new Error('fail-on-first');
        });
      },
      /fail-on-first/
    );
  } finally {
    Atomics.store = origStore;
  }
  assert.equal(consumer.readPos, readPosBeforeZeroFail);
  assert.equal(storeCalls, 0);
  assert.equal(consumer.getStats().batchesDrained, 0);
  assert.equal(consumer.getStats().packetsDrained, 0);

  // Case 2: Succeed on packets 0, 1, 2 (spanning the wrap boundary!) and throw on packet 3.
  const firstPhase = [];
  assert.throws(
    () => {
      consumer.drainBatch((pkt, idx) => {
        if (idx === 3) {
          throw new Error('boom-mid-batch');
        }
        firstPhase.push({ opcode: pkt.opcode, payload: pkt.payload.slice() });
      });
    },
    /boom-mid-batch/
  );

  assert.equal(firstPhase.length, 3);
  assert.deepEqual(
    firstPhase.map((p) => p.opcode),
    [100, 101, 102]
  );
  // Only the 3 completed packets (3 * 16 = 48 bytes) were committed!
  assert.equal(consumer.readPos, (readPosBeforeZeroFail + 48) >>> 0);
  assert.equal(consumer.availableBytes(), 5 * 16);

  const midStats = consumer.getStats();
  assert.equal(midStats.batchesDrained, 1);
  assert.equal(midStats.packetsDrained, 3);
  assert.equal(midStats.maxBatchSize, 3);
  assert.equal(midStats.avgBatchSize, 3);

  // Retry resumes cleanly at packet 3 and drains packets 3..7 with zero loss or duplication.
  const secondPhase = [];
  const res = consumer.drainBatch((pkt) => {
    secondPhase.push({ opcode: pkt.opcode, flags: pkt.flags, payload: pkt.payload.slice() });
  });
  assert.equal(res.packets, 5);
  assert.equal(res.bytes, 5 * 16);
  assert.equal(res.remainingBytes, 0);
  assert.equal(consumer.availableBytes(), 0);
  assert.deepEqual(
    secondPhase.map((p) => p.opcode),
    [103, 104, 105, 106, 107]
  );
  for (let i = 0; i < 5; i++) {
    assert.deepEqual(secondPhase[i].payload, packets[3 + i].payload);
  }

  const finalStats = consumer.getStats();
  assert.equal(finalStats.batchesDrained, 2);
  assert.equal(finalStats.packetsDrained, 8);
  assert.equal(finalStats.maxBatchSize, 5);
  assert.equal(finalStats.avgBatchSize, 4);
});

test('W187 drainBatch leaves incomplete trailing header/payload uncommitted until completed', () => {
  const producer = GpuRingBuffer.create(128);
  const consumer = GpuRingBuffer.attach(producer.sab);

  const p0 = encodeRawPacket(1, 0, new Uint8Array(0));
  const p1 = encodeRawPacket(2, 1, Uint8Array.from([1, 2, 3, 4]));
  const p2 = encodeRawPacket(3, 2, Uint8Array.from([10, 20, 30, 40, 50, 60]));

  // Write p0, p1, and only first 4 bytes of p2's header.
  assert.equal(producer.writeBytes(p0), true);
  assert.equal(producer.writeBytes(p1), true);
  assert.equal(producer.writeBytes(p2.subarray(0, 4)), true);

  const seen = [];
  const r1 = consumer.drainBatch((pkt) => seen.push(pkt.opcode));
  assert.equal(r1.packets, 2);
  assert.equal(r1.bytes, p0.byteLength + p1.byteLength);
  assert.equal(r1.remainingBytes, 4);
  assert.deepEqual(seen, [1, 2]);

  // Write remaining 4 header bytes + only 3 of 6 payload bytes.
  assert.equal(producer.writeBytes(p2.subarray(4, 11)), true);
  const r2 = consumer.drainBatch((pkt) => seen.push(pkt.opcode));
  assert.equal(r2.packets, 0);
  assert.equal(r2.bytes, 0);
  assert.equal(r2.remainingBytes, 11);
  assert.equal(consumer.getStats().batchesDrained, 1);

  // Write final 3 payload bytes -> packet 2 now completes cleanly.
  assert.equal(producer.writeBytes(p2.subarray(11)), true);
  let p2Payload = null;
  const r3 = consumer.drainBatch((pkt) => {
    seen.push(pkt.opcode);
    p2Payload = pkt.payload.slice();
  });
  assert.equal(r3.packets, 1);
  assert.equal(r3.bytes, p2.byteLength);
  assert.equal(r3.remainingBytes, 0);
  assert.deepEqual(seen, [1, 2, 3]);
  assert.deepEqual(p2Payload, Uint8Array.from([10, 20, 30, 40, 50, 60]));
});

test('W187 sanitized throughput & backpressure telemetry in getStats() and reset()', () => {
  const ring = GpuRingBuffer.create(64);

  const initial = ring.getStats();
  assert.equal(initial.capacityBytes, 64);
  assert.equal(initial.batchesDrained, 0);
  assert.equal(initial.maxBatchSize, 0);
  assert.equal(initial.avgBatchSize, 0);
  assert.equal(initial.packetsDrained, 0);
  assert.equal(initial.backpressureWaits, 0);
  assert.equal(initial.peakOccupancyBytes, 0);

  // Write 48 bytes -> peakOccupancyBytes = 48.
  const p16 = encodeRawPacket(6, 0, new Uint8Array(8)); // 16 bytes
  assert.equal(ring.writeBytes(p16), true);
  assert.equal(ring.writeBytes(p16), true);
  assert.equal(ring.writeBytes(p16), true);
  assert.equal(ring.getStats().peakOccupancyBytes, 48);

  // Oversized write (> capacity) drops packet but is NOT a backpressure wait.
  assert.equal(ring.writeBytes(new Uint8Array(65)), false);
  assert.equal(ring.getStats().droppedPackets, 1);
  assert.equal(ring.getStats().backpressureWaits, 0);

  // Full-ring backpressure (32B does not fit in 16B free) increments backpressureWaits.
  assert.equal(ring.writeBytes(new Uint8Array(32)), false);
  assert.equal(ring.writeBytes(new Uint8Array(24)), false);
  assert.equal(ring.getStats().droppedPackets, 3);
  assert.equal(ring.getStats().backpressureWaits, 2);

  // External backpressure wait recording + sanitization of invalid counts.
  assert.equal(ring.recordBackpressureWait(), 3);
  assert.equal(ring.recordBackpressureWait(2), 5);
  assert.equal(ring.recordBackpressureWait(0), 5);
  assert.equal(ring.recordBackpressureWait(-10), 5);
  assert.equal(ring.recordBackpressureWait(NaN), 5);

  // Drain all 3 packets; peakOccupancyBytes remains at high-water mark (48).
  const drained = ring.drainBatch(() => {});
  assert.equal(drained.packets, 3);
  const afterDrain = ring.getStats();
  assert.equal(afterDrain.availableBytes, 0);
  assert.equal(afterDrain.peakOccupancyBytes, 48);
  assert.equal(afterDrain.batchesDrained, 1);
  assert.equal(afterDrain.packetsDrained, 3);
  assert.equal(afterDrain.maxBatchSize, 3);
  assert.equal(afterDrain.avgBatchSize, 3);
  assert.equal(afterDrain.backpressureWaits, 5);

  // Sanitization guard if internal counters are ever corrupted with NaN / Infinity / negative.
  ring._batchesDrained = NaN;
  ring._packetsDrained = -5;
  ring._maxBatchSize = Infinity;
  ring._backpressureWaits = -1;
  ring._peakOccupancyBytes = 999999;
  const sanitized = ring.getStats();
  assert.equal(sanitized.batchesDrained, 0);
  assert.equal(sanitized.packetsDrained, 0);
  assert.equal(sanitized.maxBatchSize, 0);
  assert.equal(sanitized.avgBatchSize, 0);
  assert.equal(sanitized.backpressureWaits, 0);
  assert.equal(sanitized.peakOccupancyBytes, 64);

  // reset() clears both SAB pointers and telemetry stats.
  ring.reset();
  const afterReset = ring.getStats();
  assert.equal(afterReset.writePos, 0);
  assert.equal(afterReset.readPos, 0);
  assert.equal(afterReset.droppedPackets, 0);
  assert.equal(afterReset.notifySeq, 0);
  assert.equal(afterReset.batchesDrained, 0);
  assert.equal(afterReset.maxBatchSize, 0);
  assert.equal(afterReset.avgBatchSize, 0);
  assert.equal(afterReset.packetsDrained, 0);
  assert.equal(afterReset.backpressureWaits, 0);
  assert.equal(afterReset.peakOccupancyBytes, 0);
});

test('W187 10,000-packet burst streams losslessly across bounded wrapping ring with batched atomic publishes', () => {
  const TOTAL_PACKETS = 10000;
  // 4096-byte ring forces >60 wrap-arounds and repeated backpressure across 10,000 packets.
  const producer = GpuRingBuffer.create(4096);
  const consumer = GpuRingBuffer.attach(producer.sab);

  const origStore = Atomics.store;
  let readPosAtomicsStores = 0;
  Atomics.store = (ta, idx, val) => {
    if (idx === HEADER_INDEX.READ_POS) {
      readPosAtomicsStores += 1;
    }
    return origStore(ta, idx, val);
  };

  let produced = 0;
  let consumed = 0;

  try {
    while (consumed < TOTAL_PACKETS) {
      // Producer burst: fill ring until backpressure triggers (or all 10,000 produced).
      while (produced < TOTAL_PACKETS) {
        // Mix small 16B draw-style packets with periodic 256B Bink2-style upload chunks.
        const payloadLen = produced % 128 === 0 ? 256 : 16;
        const payload = new Uint8Array(payloadLen);
        const dv = new DataView(payload.buffer);
        dv.setUint32(0, produced, true);
        dv.setUint32(4, (produced * 2654435761) >>> 0, true);
        payload[payloadLen - 1] = produced & 0xff;

        const opcode = produced % 128 === 0 ? OPCODES.WRITE_BUFFER : OPCODES.DRAW;
        const flags = produced & 0xffff;
        const pkt = encodeRawPacket(opcode, flags, payload);

        if (!producer.writeBytes(pkt)) {
          // Ring hit backpressure -> break out to let consumer drain a batch.
          break;
        }
        produced += 1;
      }

      // Consumer drains one bounded batch (default 256 maxPackets).
      const batch = consumer.drainBatch((pkt) => {
        const expectedOpcode = consumed % 128 === 0 ? OPCODES.WRITE_BUFFER : OPCODES.DRAW;
        const expectedLen = consumed % 128 === 0 ? 256 : 16;
        assert.equal(pkt.opcode, expectedOpcode);
        assert.equal(pkt.flags, consumed & 0xffff);
        assert.equal(pkt.payloadBytes, expectedLen);
        const dv = new DataView(
          pkt.payload.buffer,
          pkt.payload.byteOffset,
          pkt.payload.byteLength
        );
        assert.equal(dv.getUint32(0, true), consumed);
        assert.equal(dv.getUint32(4, true), (consumed * 2654435761) >>> 0);
        assert.equal(pkt.payload[expectedLen - 1], consumed & 0xff);
        consumed += 1;
      });
      assert.ok(batch.packets > 0, 'consumer must make forward progress on non-empty ring');
    }
  } finally {
    Atomics.store = origStore;
  }

  assert.equal(produced, TOTAL_PACKETS);
  assert.equal(consumed, TOTAL_PACKETS);
  assert.equal(consumer.availableBytes(), 0);

  const prodStats = producer.getStats();
  const consStats = consumer.getStats();

  assert.equal(consStats.packetsDrained, TOTAL_PACKETS);
  assert.ok(consStats.batchesDrained > 0 && consStats.batchesDrained < TOTAL_PACKETS / 50);
  assert.equal(readPosAtomicsStores, consStats.batchesDrained);
  assert.ok(consStats.maxBatchSize > 1 && consStats.maxBatchSize <= DEFAULT_BATCH_MAX_PACKETS);
  assert.equal(consStats.avgBatchSize, TOTAL_PACKETS / consStats.batchesDrained);
  assert.ok(consStats.peakOccupancyBytes > 3500 && consStats.peakOccupancyBytes <= 4096);
  assert.ok(prodStats.backpressureWaits > 0, 'bounded 4KB burst must exercise backpressureWaits');
  assert.equal(prodStats.peakOccupancyBytes, consStats.peakOccupancyBytes);
});

test('W198 drainBatch(maxPackets, callback) performs zero-extra-atomics batching across wrap-around with partial caps and early stop', () => {
  const producer = GpuRingBuffer.create(128);
  const consumer = GpuRingBuffer.attach(producer.sab);

  // Argument validation for (maxPackets, callback) order.
  assert.throws(() => consumer.drainBatch(4), TypeError);
  assert.throws(() => consumer.drainBatch(0, () => {}), RangeError);
  assert.throws(() => consumer.drainBatch(-3, () => {}), RangeError);
  assert.throws(() => consumer.drainBatch(2.5, () => {}), RangeError);
  assert.throws(() => consumer.drainBatch(NaN, () => {}), RangeError);
  assert.throws(() => consumer.drainBatch(Infinity, () => {}), RangeError);
  assert.throws(() => consumer.drainBatch('4', () => {}), RangeError);

  // Advance ring offset to 108 so the batch wraps across the 128-byte boundary
  // (packet 1 header splits [124..128) + [0..4), and later packets continue contiguously).
  assert.equal(producer.writeBytes(new Uint8Array(108)), true);
  assert.equal(consumer.drainTo(() => {}), 108);
  assert.equal(consumer.readPos & 127, 108);

  const expected = [];
  for (let i = 0; i < 7; i++) {
    const payload = Uint8Array.from({ length: 8 }, (_, b) => ((i * 29 + b * 7 + 3) & 0xff));
    const pkt = encodeRawPacket(20 + i, (i + 1) * 0x11, payload); // 16B each; 7 * 16 = 112B <= 128B
    assert.equal(producer.writeBytes(pkt), true);
    expected.push({ opcode: 20 + i, flags: (i + 1) * 0x11, payload, raw: pkt });
  }

  // Instrument Atomics.load / Atomics.store / Atomics.notify to assert zero extra atomics.
  const origLoad = Atomics.load;
  const origStore = Atomics.store;
  const origNotify = Atomics.notify;
  let writePosLoads = 0;
  let readPosLoads = 0;
  let readPosStores = 0;
  let readPosNotifies = 0;

  Atomics.load = (ta, idx) => {
    if (idx === HEADER_INDEX.WRITE_POS) writePosLoads += 1;
    if (idx === HEADER_INDEX.READ_POS) readPosLoads += 1;
    return origLoad(ta, idx);
  };
  Atomics.store = (ta, idx, val) => {
    if (idx === HEADER_INDEX.READ_POS) readPosStores += 1;
    return origStore(ta, idx, val);
  };
  Atomics.notify = (ta, idx, count) => {
    if (idx === HEADER_INDEX.READ_POS) readPosNotifies += 1;
    return origNotify(ta, idx, count);
  };

  const collected = [];
  try {
    // Partial cap: drain first 3 packets (spans wrap boundary) with (maxPackets, callback).
    const r1 = consumer.drainBatch(3, (pkt, idx) => {
      assert.equal(idx, collected.length);
      // Zero extra atomics during packet iteration!
      assert.equal(writePosLoads, 1);
      assert.equal(readPosLoads, 1);
      assert.equal(readPosStores, 0);
      collected.push({
        opcode: pkt.opcode,
        flags: pkt.flags,
        payload: pkt.payload.slice(),
        bytes: pkt.bytes.slice(),
        wrapped: pkt.wrapped,
      });
    });

    assert.equal(r1.packets, 3);
    assert.equal(r1.bytes, 48);
    assert.equal(r1.remainingBytes, 64);
    assert.equal(r1.wrapped, true);
    assert.equal(writePosLoads, 1, 'must read WRITE_POS exactly once per drainBatch');
    assert.equal(readPosLoads, 1, 'must read READ_POS exactly once per drainBatch');
    assert.equal(readPosStores, 1, 'must write READ_POS exactly once at batch end');
    assert.equal(readPosNotifies, 1);

    // Cooperative early stop: ask for up to 4 packets, return false on index 1 -> commits 2 packets.
    const r2 = consumer.drainBatch(4, (pkt, idx) => {
      collected.push({
        opcode: pkt.opcode,
        flags: pkt.flags,
        payload: pkt.payload.slice(),
        bytes: pkt.bytes.slice(),
        wrapped: pkt.wrapped,
      });
      if (idx === 1) {
        return false;
      }
    });

    assert.equal(r2.packets, 2);
    assert.equal(r2.bytes, 32);
    assert.equal(r2.remainingBytes, 32);
    assert.equal(writePosLoads, 2);
    assert.equal(readPosLoads, 2);
    assert.equal(readPosStores, 2);
    assert.equal(readPosNotifies, 2);

    // Default maxPackets via (undefined, callback) drains remaining 2 packets.
    const r3 = consumer.drainBatch(undefined, (pkt) => {
      collected.push({
        opcode: pkt.opcode,
        flags: pkt.flags,
        payload: pkt.payload.slice(),
        bytes: pkt.bytes.slice(),
        wrapped: pkt.wrapped,
      });
    });

    assert.equal(r3.packets, 2);
    assert.equal(r3.bytes, 32);
    assert.equal(r3.remainingBytes, 0);
    assert.equal(writePosLoads, 3);
    assert.equal(readPosLoads, 3);
    assert.equal(readPosStores, 3);
    assert.equal(readPosNotifies, 3);
  } finally {
    Atomics.load = origLoad;
    Atomics.store = origStore;
    Atomics.notify = origNotify;
  }

  // Exact byte-for-byte preservation across all 7 packets.
  assert.equal(collected.length, 7);
  for (let i = 0; i < 7; i++) {
    assert.equal(collected[i].opcode, expected[i].opcode);
    assert.equal(collected[i].flags, expected[i].flags);
    assert.deepEqual(collected[i].payload, expected[i].payload);
    assert.deepEqual(collected[i].bytes, expected[i].raw);
  }
  assert.equal(collected[0].wrapped, false);
  assert.equal(collected[1].wrapped, true);
  assert.equal(collected[2].wrapped, false);
  assert.equal(consumer.availableBytes(), 0);
});

test('W198 high-watermark utilization, peakUtilizationPct, resetTelemetry(), and untouched 64B SAB header', () => {
  const ring = GpuRingBuffer.create(64);
  const rawHeader = new DataView(ring.sab);

  // Initial telemetry state.
  assert.equal(ring.highWatermarkBytes, 0);
  assert.equal(ring.peakUtilizationPct, 0);
  assert.equal(ring.batchesDrained, 0);
  assert.equal(ring.packetsDrained, 0);
  assert.equal(ring.backpressureWaits, 0);
  const initStats = ring.getStats();
  assert.equal(initStats.highWatermarkBytes, 0);
  assert.equal(initStats.peakUtilizationPct, 0);
  assert.equal(initStats.utilizationPct, 0);

  // Write 16B (25%), then 32B more -> 48B (75%).
  const pkt16 = encodeRawPacket(OPCODES.DRAW, 1, new Uint8Array(8));
  assert.equal(ring.writeBytes(pkt16), true);
  assert.equal(ring.highWatermarkBytes, 16);
  assert.equal(ring.peakUtilizationPct, 25);
  assert.equal(ring.getStats().utilizationPct, 25);

  assert.equal(ring.writeBytes(pkt16), true);
  assert.equal(ring.writeBytes(pkt16), true);
  assert.equal(ring.highWatermarkBytes, 48);
  assert.equal(ring.peakUtilizationPct, 75);
  assert.equal(ring.getStats().utilizationPct, 75);

  // Backpressure when 32B doesn't fit in 16B free.
  assert.equal(ring.writeBytes(new Uint8Array(32)), false);
  assert.equal(ring.backpressureWaits, 1);

  // Fill remaining 16B -> 64B (100% peak).
  assert.equal(ring.writeBytes(pkt16), true);
  assert.equal(ring.highWatermarkBytes, 64);
  assert.equal(ring.peakUtilizationPct, 100);
  assert.equal(ring.getStats().utilizationPct, 100);

  // Drain 2 of 4 packets using drainBatch(2, cb): current utilization drops to 50%,
  // while highWatermarkBytes & peakUtilizationPct remain at 64B / 100%.
  const b1 = ring.drainBatch(2, () => {});
  assert.equal(b1.packets, 2);
  assert.equal(ring.highWatermarkBytes, 64);
  assert.equal(ring.peakUtilizationPct, 100);
  assert.equal(ring.batchesDrained, 1);
  assert.equal(ring.packetsDrained, 2);
  const midStats = ring.getStats();
  assert.equal(midStats.highWatermarkBytes, 64);
  assert.equal(midStats.peakOccupancyBytes, 64);
  assert.equal(midStats.peakUtilizationPct, 100);
  assert.equal(midStats.utilizationPct, 50);
  assert.equal(midStats.batchesDrained, 1);
  assert.equal(midStats.packetsDrained, 2);
  assert.equal(midStats.backpressureWaits, 1);

  // resetTelemetry() clears local telemetry counters without discarding queued SAB packets!
  const writePosBefore = ring.writePos;
  const readPosBefore = ring.readPos;
  ring.resetTelemetry();
  assert.equal(ring.highWatermarkBytes, 0);
  assert.equal(ring.peakUtilizationPct, 0);
  assert.equal(ring.batchesDrained, 0);
  assert.equal(ring.packetsDrained, 0);
  assert.equal(ring.backpressureWaits, 0);
  assert.equal(ring.writePos, writePosBefore);
  assert.equal(ring.readPos, readPosBefore);

  // Draining remaining 2 queued packets still succeeds and records fresh watermark = 32B (50%).
  const b2 = ring.drainBatch(16, () => {});
  assert.equal(b2.packets, 2);
  assert.equal(ring.highWatermarkBytes, 32);
  assert.equal(ring.peakUtilizationPct, 50);
  assert.equal(ring.batchesDrained, 1);
  assert.equal(ring.packetsDrained, 2);
  assert.equal(ring.getStats().utilizationPct, 0);

  // Self-healing if internal _peakOccupancyBytes is corrupted to NaN or oversized.
  ring._peakOccupancyBytes = NaN;
  assert.equal(ring.highWatermarkBytes, 0);
  assert.equal(ring.peakUtilizationPct, 0);
  assert.equal(ring.writeBytes(pkt16), true);
  assert.equal(ring.highWatermarkBytes, 16);
  assert.equal(ring.peakUtilizationPct, 25);

  ring._peakOccupancyBytes = 99999;
  assert.equal(ring.highWatermarkBytes, 64);
  assert.equal(ring.peakUtilizationPct, 100);
  assert.equal(ring.getStats().peakUtilizationPct, 100);

  // Verify the 64-byte SAB header ABI layout is untouched (reserved words 8..15 stay 0).
  assert.equal(rawHeader.getUint32(0, true), GPU_RING_MAGIC);
  assert.equal(rawHeader.getUint32(4, true), GPU_RING_VERSION);
  assert.equal(rawHeader.getUint32(8, true), 64);
  assert.equal(rawHeader.getUint32(12, true), 0);
  for (let off = 32; off < 64; off += 4) {
    assert.equal(rawHeader.getUint32(off, true), 0, `SAB header reserved word at byte ${off} must remain 0`);
  }
});

test('W244 backpressure & batch-drain telemetry tracks drainCalls, maxDrainBatchBytes, totalBytesDrained, backpressureStalls, peakOccupancyRatio, emptyDrains, and resetStats()', () => {
  const ring = GpuRingBuffer.create(64);
  const rawHeader = new DataView(ring.sab);

  // Initial stats & getters.
  assert.equal(ring.drainCalls, 0);
  assert.equal(ring.maxDrainBatchBytes, 0);
  assert.equal(ring.totalBytesDrained, 0);
  assert.equal(ring.backpressureStalls, 0);
  assert.equal(ring.peakOccupancyRatio, 0);
  assert.equal(ring.emptyDrains, 0);

  const init = ring.getStats();
  assert.equal(init.drainCalls, 0);
  assert.equal(init.maxDrainBatchBytes, 0);
  assert.equal(init.totalBytesDrained, 0);
  assert.equal(init.backpressureStalls, 0);
  assert.equal(init.peakOccupancyRatio, 0);
  assert.equal(init.emptyDrains, 0);
  // Existing fields preserved.
  assert.equal(init.capacityBytes, 64);
  assert.equal(init.batchesDrained, 0);
  assert.equal(init.packetsDrained, 0);
  assert.equal(init.backpressureWaits, 0);
  assert.equal(init.highWatermarkBytes, 0);
  assert.equal(init.peakUtilizationPct, 0);

  // Invalid drain arguments throw without incrementing drainCalls or emptyDrains.
  assert.throws(() => ring.drainTo(null), TypeError);
  assert.throws(() => ring.drainBatch(null), TypeError);
  assert.throws(() => ring.drainBatch(() => {}, 0), RangeError);
  assert.equal(ring.drainCalls, 0);
  assert.equal(ring.emptyDrains, 0);

  // Empty drains via both drainTo and drainBatch increment drainCalls and emptyDrains.
  assert.equal(ring.drainTo(() => { throw new Error('unexpected'); }), 0);
  const emptyBatch = ring.drainBatch(() => { throw new Error('unexpected'); });
  assert.equal(emptyBatch.bytes, 0);
  assert.equal(ring.drainCalls, 2);
  assert.equal(ring.emptyDrains, 2);
  assert.equal(ring.totalBytesDrained, 0);
  assert.equal(ring.maxDrainBatchBytes, 0);
  assert.equal(ring.batchesDrained, 0);

  // Write 3x16B packets -> 48B (peakOccupancyRatio = 0.75).
  const pkt16 = encodeRawPacket(OPCODES.DRAW, 3, new Uint8Array(8));
  assert.equal(ring.writeBytes(pkt16), true);
  assert.equal(ring.peakOccupancyRatio, 0.25);
  assert.equal(ring.writeBytes(pkt16), true);
  assert.equal(ring.writeBytes(pkt16), true);
  assert.equal(ring.peakOccupancyRatio, 0.75);
  assert.equal(ring.getStats().peakOccupancyRatio, 0.75);

  // Oversized write (> capacity) is NOT a backpressure stall.
  assert.equal(ring.writeBytes(new Uint8Array(65)), false);
  assert.equal(ring.backpressureStalls, 0);
  assert.equal(ring.droppedPackets, 1);

  // Full/insufficient-space writes increment backpressureStalls & backpressureWaits.
  assert.equal(ring.writeBytes(new Uint8Array(32)), false);
  assert.equal(ring.writeBytes(new Uint8Array(17)), false);
  assert.equal(ring.backpressureStalls, 2);
  assert.equal(ring.backpressureWaits, 2);

  // External wait recording also updates backpressureStalls and ignores invalid counts.
  assert.equal(ring.recordBackpressureWait(3), 5);
  assert.equal(ring.recordBackpressureWait(0), 5);
  assert.equal(ring.recordBackpressureWait(-2), 5);
  assert.equal(ring.recordBackpressureWait(NaN), 5);
  assert.equal(ring.backpressureStalls, 5);

  // Fill remaining 16B -> 64B (peakOccupancyRatio = 1.0).
  assert.equal(ring.writeBytes(pkt16), true);
  assert.equal(ring.peakOccupancyRatio, 1);

  // Drain 1 packet (16B) via drainBatch, then 2 packets (32B) via drainBatch, then remaining 16B via drainTo.
  const b1 = ring.drainBatch(1, () => {});
  assert.equal(b1.bytes, 16);
  assert.equal(ring.drainCalls, 3);
  assert.equal(ring.emptyDrains, 2);
  assert.equal(ring.maxDrainBatchBytes, 16);
  assert.equal(ring.totalBytesDrained, 16);

  const b2 = ring.drainBatch(2, () => {});
  assert.equal(b2.bytes, 32);
  assert.equal(ring.drainCalls, 4);
  assert.equal(ring.maxDrainBatchBytes, 32);
  assert.equal(ring.totalBytesDrained, 48);

  const dTo = ring.drainTo(() => {});
  assert.equal(dTo, 16);
  assert.equal(ring.drainCalls, 5);
  assert.equal(ring.maxDrainBatchBytes, 32);
  assert.equal(ring.totalBytesDrained, 64);

  // Drain a larger wrapped batch (48B) -> maxDrainBatchBytes advances to 48.
  assert.equal(ring.writeBytes(pkt16), true);
  assert.equal(ring.writeBytes(pkt16), true);
  assert.equal(ring.writeBytes(pkt16), true);
  const b3 = ring.drainBatch(() => {});
  assert.equal(b3.bytes, 48);
  assert.equal(ring.drainCalls, 6);
  assert.equal(ring.maxDrainBatchBytes, 48);
  assert.equal(ring.totalBytesDrained, 112);

  // Incomplete trailing packet counts as a drainCall + emptyDrain without changing byte totals.
  assert.equal(ring.writeBytes(pkt16.subarray(0, 4)), true);
  const partial = ring.drainBatch(() => {});
  assert.equal(partial.bytes, 0);
  assert.equal(ring.drainCalls, 7);
  assert.equal(ring.emptyDrains, 3);
  assert.equal(ring.totalBytesDrained, 112);
  assert.equal(ring.maxDrainBatchBytes, 48);

  // Finish the incomplete packet, then verify resetStats() clears counters without discarding queued SAB bytes.
  assert.equal(ring.writeBytes(pkt16.subarray(4)), true);
  const writePosBefore = ring.writePos;
  const readPosBefore = ring.readPos;

  const snapBeforeReset = ring.getStats();
  assert.equal(snapBeforeReset.drainCalls, 7);
  assert.equal(snapBeforeReset.maxDrainBatchBytes, 48);
  assert.equal(snapBeforeReset.totalBytesDrained, 112);
  assert.equal(snapBeforeReset.backpressureStalls, 5);
  assert.equal(snapBeforeReset.peakOccupancyRatio, 1);
  assert.equal(snapBeforeReset.emptyDrains, 3);

  ring.resetStats();
  assert.equal(ring.drainCalls, 0);
  assert.equal(ring.maxDrainBatchBytes, 0);
  assert.equal(ring.totalBytesDrained, 0);
  assert.equal(ring.backpressureStalls, 0);
  assert.equal(ring.peakOccupancyRatio, 0);
  assert.equal(ring.emptyDrains, 0);
  assert.equal(ring.batchesDrained, 0);
  assert.equal(ring.packetsDrained, 0);
  assert.equal(ring.backpressureWaits, 0);
  assert.equal(ring.highWatermarkBytes, 0);
  assert.equal(ring.writePos, writePosBefore);
  assert.equal(ring.readPos, readPosBefore);

  // Draining the still-queued 16B packet after resetStats() works cleanly and records fresh ratio 0.25.
  const bAfter = ring.drainBatch(() => {});
  assert.equal(bAfter.packets, 1);
  assert.equal(bAfter.bytes, 16);
  const afterStats = ring.getStats();
  assert.equal(afterStats.drainCalls, 1);
  assert.equal(afterStats.maxDrainBatchBytes, 16);
  assert.equal(afterStats.totalBytesDrained, 16);
  assert.equal(afterStats.backpressureStalls, 0);
  assert.equal(afterStats.peakOccupancyRatio, 0.25);
  assert.equal(afterStats.emptyDrains, 0);

  // Verify 64B SAB header reserved words remain untouched.
  for (let off = 32; off < 64; off += 4) {
    assert.equal(rawHeader.getUint32(off, true), 0);
  }
});

test('W244 telemetry sanitization, mid-batch throw semantics, and zero-extra-atomics burst stability', () => {
  const producer = GpuRingBuffer.create(128);
  const consumer = GpuRingBuffer.attach(producer.sab);
  const pkt16 = encodeRawPacket(OPCODES.DRAW, 0, new Uint8Array(8));

  // Throw on packet 0 increments drainCalls, but NOT emptyDrains/totalBytesDrained/maxDrainBatchBytes.
  assert.equal(producer.writeBytes(pkt16), true);
  assert.equal(producer.writeBytes(pkt16), true);
  assert.equal(producer.writeBytes(pkt16), true);
  assert.throws(() => {
    consumer.drainBatch(() => {
      throw new Error('fail-first');
    });
  }, /fail-first/);
  assert.equal(consumer.drainCalls, 1);
  assert.equal(consumer.emptyDrains, 0);
  assert.equal(consumer.totalBytesDrained, 0);
  assert.equal(consumer.maxDrainBatchBytes, 0);

  // Throw on packet 2 after 2 completed packets commits 32B to totalBytesDrained & maxDrainBatchBytes.
  assert.throws(() => {
    consumer.drainBatch((_pkt, idx) => {
      if (idx === 2) {
        throw new Error('fail-third');
      }
    });
  }, /fail-third/);
  assert.equal(consumer.drainCalls, 2);
  assert.equal(consumer.emptyDrains, 0);
  assert.equal(consumer.totalBytesDrained, 32);
  assert.equal(consumer.maxDrainBatchBytes, 32);

  // Drain remaining 16B packet cleanly.
  const tailRes = consumer.drainBatch(() => {});
  assert.equal(tailRes.bytes, 16);
  assert.equal(consumer.drainCalls, 3);
  assert.equal(consumer.totalBytesDrained, 48);
  assert.equal(consumer.maxDrainBatchBytes, 32);

  // Corruption sanitization & clamping of all W244 telemetry fields.
  consumer._drainCalls = NaN;
  consumer._maxDrainBatchBytes = 999999;
  consumer._totalBytesDrained = -50;
  consumer._backpressureStalls = Infinity;
  consumer._emptyDrains = -7;
  consumer._peakOccupancyBytes = 999999;
  let sanitized = consumer.getStats();
  assert.equal(sanitized.drainCalls, 0);
  assert.equal(sanitized.maxDrainBatchBytes, 128);
  assert.equal(sanitized.totalBytesDrained, 0);
  assert.equal(sanitized.backpressureStalls, 0);
  assert.equal(sanitized.emptyDrains, 0);
  assert.equal(sanitized.peakOccupancyRatio, 1);

  consumer._peakOccupancyBytes = -12;
  assert.equal(consumer.peakOccupancyRatio, 0);
  assert.equal(consumer.getStats().peakOccupancyRatio, 0);

  consumer._peakOccupancyBytes = NaN;
  assert.equal(consumer.peakOccupancyRatio, 0);
  assert.equal(consumer.getStats().peakOccupancyRatio, 0);

  // Self-healing after corrupted fields: subsequent operations increment from sanitized 0.
  consumer._maxDrainBatchBytes = NaN;
  assert.equal(consumer.drainBatch(() => {}).bytes, 0);
  assert.equal(consumer.drainCalls, 1);
  assert.equal(consumer.emptyDrains, 1);
  assert.equal(producer.writeBytes(pkt16), true);
  assert.equal(consumer.drainBatch(() => {}).bytes, 16);
  assert.equal(consumer.drainCalls, 2);
  assert.equal(consumer.totalBytesDrained, 16);
  assert.equal(consumer.maxDrainBatchBytes, 16);
  assert.equal(consumer.peakOccupancyRatio, 16 / 128);

  // Full reset() clears pointers and W244 telemetry.
  producer.reset();
  consumer.resetStats();
  const afterReset = consumer.getStats();
  assert.equal(afterReset.drainCalls, 0);
  assert.equal(afterReset.maxDrainBatchBytes, 0);
  assert.equal(afterReset.totalBytesDrained, 0);
  assert.equal(afterReset.backpressureStalls, 0);
  assert.equal(afterReset.peakOccupancyRatio, 0);
  assert.equal(afterReset.emptyDrains, 0);

  // Burst stress: 500 cycles of fill + backpressure stall + batch drain + empty poll.
  const origStore = Atomics.store;
  let readPosStores = 0;
  Atomics.store = (ta, idx, val) => {
    if (idx === HEADER_INDEX.READ_POS) {
      readPosStores += 1;
    }
    return origStore(ta, idx, val);
  };

  let expectedStalls = 0;
  let expectedDrainedBytes = 0;
  try {
    for (let cycle = 0; cycle < 500; cycle++) {
      // Fill 8x16B = 128B (full ring), then trigger 1 stall.
      for (let p = 0; p < 8; p++) {
        assert.equal(producer.writeBytes(pkt16), true);
      }
      assert.equal(producer.writeBytes(pkt16), false);
      expectedStalls += 1;

      const batch = consumer.drainBatch(() => {});
      assert.equal(batch.packets, 8);
      assert.equal(batch.bytes, 128);
      expectedDrainedBytes += 128;

      // One empty poll per cycle.
      const empty = consumer.drainBatch(() => {});
      assert.equal(empty.bytes, 0);
    }
  } finally {
    Atomics.store = origStore;
  }

  assert.equal(readPosStores, 500, 'single atomic READ_POS store per non-empty drainBatch');
  const prodStats = producer.getStats();
  const consStats = consumer.getStats();
  assert.equal(prodStats.backpressureStalls, expectedStalls);
  assert.equal(prodStats.peakOccupancyRatio, 1);
  assert.equal(consStats.drainCalls, 1000);
  assert.equal(consStats.emptyDrains, 500);
  assert.equal(consStats.totalBytesDrained, expectedDrainedBytes);
  assert.equal(consStats.maxDrainBatchBytes, 128);
  assert.equal(consStats.peakOccupancyRatio, 1);
});
