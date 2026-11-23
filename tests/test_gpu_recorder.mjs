import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const require = createRequire(import.meta.url);
const recorderMod = require('../web/workers/gpu_recorder.js');
const wgpuMod = require('../web/workers/wgpu_worker.js');
const ringMod = require('../web/workers/gpu_ring.js');

const {
  GpuFrameRecorder,
  createInterceptor,
  PACKET_HEADER_BYTES,
  DEFAULT_MAX_FRAMES,
  DEFAULT_MAX_BYTES,
  OPCODES,
  OPCODE_NAMES,
  getOpcodeName,
} = recorderMod;

const { WgpuCommandProcessor, BUFFER_USAGE, SHADER_STAGE, PRIMITIVE_TOPOLOGY } = wgpuMod;
const { GpuRingBuffer } = ringMod;

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');

class BinaryPacketBuilder {
  constructor() {
    this.chunks = [];
  }

  packet(opcode, payloadBytes = new Uint8Array(0), flags = 0) {
    const payload =
      payloadBytes instanceof Uint8Array ? payloadBytes : new Uint8Array(payloadBytes);
    const buf = new ArrayBuffer(PACKET_HEADER_BYTES + payload.byteLength);
    const view = new DataView(buf);
    view.setUint16(0, opcode, true);
    view.setUint16(2, flags, true);
    view.setUint32(4, payload.byteLength, true);
    new Uint8Array(buf, PACKET_HEADER_BYTES).set(payload);
    this.chunks.push(new Uint8Array(buf));
    return this;
  }

  nop(flags = 0) {
    return this.packet(OPCODES.NOP, new Uint8Array(0), flags);
  }

  beginFrame(flags = 0) {
    return this.packet(OPCODES.BEGIN_FRAME, new Uint8Array(0), flags);
  }

  clearColor(r, g, b, a, flags = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setFloat32(0, r, true);
    view.setFloat32(4, g, true);
    view.setFloat32(8, b, true);
    view.setFloat32(12, a, true);
    return this.packet(OPCODES.CLEAR_COLOR, new Uint8Array(buf), flags);
  }

  createBuffer(id, usageFlags, byteSize, flags = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, usageFlags, true);
    view.setBigUint64(8, BigInt(byteSize), true);
    return this.packet(OPCODES.CREATE_BUFFER, new Uint8Array(buf), flags);
  }

  writeBuffer(id, dstOffset, data, reserved = 0, flags = 0) {
    const src = data instanceof Uint8Array ? data : new Uint8Array(data);
    const buf = new ArrayBuffer(16 + src.byteLength);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, dstOffset, true);
    view.setUint32(8, src.byteLength, true);
    view.setUint32(12, reserved, true);
    new Uint8Array(buf, 16).set(src);
    return this.packet(OPCODES.WRITE_BUFFER, new Uint8Array(buf), flags);
  }

  createShaderModule(id, stage, wgslCode, reserved = 0, flags = 0) {
    const encoded = new TextEncoder().encode(wgslCode);
    const buf = new ArrayBuffer(16 + encoded.byteLength);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, stage, true);
    view.setUint32(8, encoded.byteLength, true);
    view.setUint32(12, reserved, true);
    new Uint8Array(buf, 16).set(encoded);
    return this.packet(OPCODES.CREATE_SHADER_MODULE, new Uint8Array(buf), flags);
  }

  createRenderPipeline(
    id,
    vsShaderId,
    fsShaderId,
    vertexStride,
    topology = PRIMITIVE_TOPOLOGY.TRIANGLE_LIST,
    pipelineFlags = 0,
    flags = 0
  ) {
    const buf = new ArrayBuffer(24);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, vsShaderId, true);
    view.setUint32(8, fsShaderId, true);
    view.setUint32(12, vertexStride, true);
    view.setUint32(16, topology, true);
    view.setUint32(20, pipelineFlags, true);
    return this.packet(OPCODES.CREATE_RENDER_PIPELINE, new Uint8Array(buf), flags);
  }

  setPipeline(pipelineId, flags = 0) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, pipelineId, true);
    return this.packet(OPCODES.SET_PIPELINE, new Uint8Array(buf), flags);
  }

  setVertexBuffer(slot, bufferId, offset = 0, flags = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, slot, true);
    view.setUint32(4, bufferId, true);
    view.setBigUint64(8, BigInt(offset), true);
    return this.packet(OPCODES.SET_VERTEX_BUFFER, new Uint8Array(buf), flags);
  }

  draw(vertexCount, instanceCount = 1, firstVertex = 0, firstInstance = 0, flags = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, vertexCount, true);
    view.setUint32(4, instanceCount, true);
    view.setUint32(8, firstVertex, true);
    view.setUint32(12, firstInstance, true);
    return this.packet(OPCODES.DRAW, new Uint8Array(buf), flags);
  }

  present(flags = 0) {
    return this.packet(OPCODES.PRESENT, new Uint8Array(0), flags);
  }

  dispatchCompute(x, y, z, flags = 0) {
    const buf = new ArrayBuffer(12);
    const view = new DataView(buf);
    view.setUint32(0, x, true);
    view.setUint32(4, y, true);
    view.setUint32(8, z, true);
    return this.packet(OPCODES.DISPATCH_COMPUTE, new Uint8Array(buf), flags);
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

function buildSimpleFrame(clearR, vertexCount = 3) {
  return new BinaryPacketBuilder()
    .beginFrame()
    .clearColor(clearR, 0.25, 0.5, 1.0)
    .draw(vertexCount, 1, 0, 0)
    .present()
    .finish();
}

function createMockWebGpu() {
  const calls = [];
  const ctx = {
    getCurrentTexture() {
      return {
        createView() {
          return { kind: 'GPUTextureView' };
        },
      };
    },
  };
  const device = {
    createBuffer(desc) {
      const buf = { kind: 'GPUBuffer', size: desc.size, usage: desc.usage, destroy() {} };
      calls.push({ op: 'createBuffer', desc });
      return buf;
    },
    createCommandEncoder() {
      return {
        beginRenderPass(desc) {
          calls.push({ op: 'beginRenderPass', desc });
          return {
            setPipeline() {},
            setVertexBuffer() {},
            draw(v, i, fv, fi) {
              calls.push({ op: 'draw', v, i, fv, fi });
            },
            end() {
              calls.push({ op: 'renderPass.end' });
            },
          };
        },
        finish() {
          return { kind: 'GPUCommandBuffer' };
        },
      };
    },
    queue: {
      writeBuffer() {},
      submit(cbs) {
        calls.push({ op: 'submit', count: cbs.length });
      },
    },
  };
  return { device, ctx, calls };
}

test('UMD global export, defaults, and constructor validation', () => {
  assert.equal(globalThis.PlayGta6GpuRecorder, recorderMod);
  assert.equal(PACKET_HEADER_BYTES, 8);
  assert.equal(DEFAULT_MAX_FRAMES, 60);
  assert.equal(DEFAULT_MAX_BYTES, 16 * 1024 * 1024);
  assert.equal(OPCODES.BEGIN_FRAME, 1);
  assert.equal(OPCODES.PRESENT, 7);
  assert.equal(OPCODE_NAMES[OPCODES.BEGIN_FRAME], 'BeginFrame');
  assert.equal(OPCODE_NAMES[OPCODES.PRESENT], 'Present');
  assert.equal(getOpcodeName(999), 'Unknown(999)');

  const rec = new GpuFrameRecorder();
  assert.equal(rec.isRecording(), false);
  const meta = rec.exportMetadata();
  assert.equal(meta.frameCount, 0);
  assert.equal(meta.totalBytes, 0);
  assert.equal(meta.totalPackets, 0);
  assert.deepEqual(meta.opcodeCounts, {});
  assert.equal(meta.droppedFrames, 0);
  assert.equal(meta.recording, false);
  assert.equal(meta.maxFrames, 60);
  assert.equal(meta.maxBytes, 16 * 1024 * 1024);

  const autoRec = new GpuFrameRecorder({ autoStart: true });
  assert.equal(autoRec.isRecording(), true);

  assert.throws(() => new GpuFrameRecorder({ maxFrames: 0 }), RangeError);
  assert.throws(() => new GpuFrameRecorder({ maxFrames: -5 }), RangeError);
  assert.throws(() => new GpuFrameRecorder({ maxFrames: 1.5 }), RangeError);
  assert.throws(() => new GpuFrameRecorder({ maxBytes: 0 }), RangeError);
  assert.throws(() => new GpuFrameRecorder({ maxBytes: -10 }), RangeError);
});

test('start(), stop(), and clear() lifecycle behavior', () => {
  const rec = new GpuFrameRecorder({ maxFrames: 10 });
  const frame0 = buildSimpleFrame(0.1, 3);
  const frame1 = buildSimpleFrame(0.2, 6);
  const frame2 = buildSimpleFrame(0.3, 9);

  // Ignored while stopped.
  assert.equal(rec.isRecording(), false);
  const ignored = rec.recordChunk(frame0);
  assert.equal(ignored.recordedBytes, 0);
  assert.equal(rec.exportMetadata().frameCount, 0);

  // Start recording and capture frame0 + half of frame1.
  rec.start();
  assert.equal(rec.isRecording(), true);
  rec.recordChunk(frame0);
  // Feed a partial packet/frame before stopping.
  rec.recordChunk(frame1.subarray(0, 13));
  assert.equal(rec.exportMetadata().frameCount, 1);

  // Stop discards in-progress partial packet/frame but retains frame0.
  rec.stop();
  assert.equal(rec.isRecording(), false);
  rec.recordChunk(frame1.subarray(13)); // ignored while stopped
  assert.equal(rec.exportMetadata().frameCount, 1);
  assert.deepEqual(rec.exportTraceBytes(), frame0);

  // Resuming cleanly starts from frame2 without stale partial bytes from frame1.
  rec.start();
  rec.recordChunk(frame2);
  assert.equal(rec.exportMetadata().frameCount, 2);
  const expectedTwo = new Uint8Array(frame0.byteLength + frame2.byteLength);
  expectedTwo.set(frame0, 0);
  expectedTwo.set(frame2, frame0.byteLength);
  assert.deepEqual(rec.exportTraceBytes(), expectedTwo);

  // clear() empties captured data and counters while keeping recording=true.
  rec.clear();
  assert.equal(rec.isRecording(), true);
  const clearedMeta = rec.exportMetadata();
  assert.equal(clearedMeta.frameCount, 0);
  assert.equal(clearedMeta.totalBytes, 0);
  assert.equal(clearedMeta.totalPackets, 0);
  assert.deepEqual(clearedMeta.opcodeCounts, {});
  assert.equal(clearedMeta.droppedFrames, 0);
  assert.equal(clearedMeta.recording, true);
  assert.equal(rec.exportTraceBytes().byteLength, 0);

  // Still recording after clear().
  rec.recordChunk(frame1);
  assert.equal(rec.exportMetadata().frameCount, 1);
  assert.deepEqual(rec.exportTraceBytes(), frame1);
});

test('packet recording across arbitrary split recordChunk calls (including 1-byte chunks)', () => {
  const rec = new GpuFrameRecorder({ autoStart: true });
  const vertexPayload = new Uint8Array(40);
  for (let i = 0; i < vertexPayload.length; i++) {
    vertexPayload[i] = (i * 7) & 0xff;
  }

  const stream = new BinaryPacketBuilder()
    .beginFrame(0x0001)
    .clearColor(0.2, 0.4, 0.6, 1.0, 0x0002)
    .createBuffer(10, BUFFER_USAGE.VERTEX | BUFFER_USAGE.COPY_DST, 256)
    .writeBuffer(10, 0, vertexPayload)
    .draw(12, 2, 0, 0)
    .present(0x0004)
    .finish();

  // Feed stream 1 byte at a time to stress header & payload reassembly.
  for (let i = 0; i < stream.byteLength; i++) {
    rec.recordChunk(stream.subarray(i, i + 1));
  }

  const meta = rec.exportMetadata();
  assert.equal(meta.frameCount, 1);
  assert.equal(meta.totalBytes, stream.byteLength);
  assert.equal(meta.totalPackets, 6);
  assert.deepEqual(meta.opcodeCounts, {
    BeginFrame: 1,
    ClearColor: 1,
    CreateBuffer: 1,
    WriteBuffer: 1,
    Draw: 1,
    Present: 1,
  });

  // Exported bytes must match input stream byte-for-byte including header flags.
  const exported = rec.exportTraceBytes();
  assert.ok(exported instanceof Uint8Array);
  assert.ok(exported.buffer instanceof ArrayBuffer);
  assert.deepEqual(exported, stream);

  // Also test odd multi-byte split boundaries spanning multiple frames.
  const rec2 = new GpuFrameRecorder({ autoStart: true });
  const twoFrames = new Uint8Array(stream.byteLength * 2);
  twoFrames.set(stream, 0);
  twoFrames.set(stream, stream.byteLength);

  const splitSizes = [3, 5, 1, 7, 19, 11, 29, 13, 41, 64, 1000];
  let pos = 0;
  let idx = 0;
  while (pos < twoFrames.byteLength) {
    const step = splitSizes[idx % splitSizes.length];
    rec2.recordChunk(twoFrames.subarray(pos, Math.min(pos + step, twoFrames.byteLength)));
    pos += step;
    idx += 1;
  }

  assert.equal(rec2.exportMetadata().frameCount, 2);
  assert.deepEqual(rec2.exportTraceBytes(), twoFrames);
});

test('frame delimiting ignores out-of-frame packets and unclosed frames cleanly', () => {
  const rec = new GpuFrameRecorder({ autoStart: true });

  // Stray packets before first BeginFrame (e.g. recording started mid-frame).
  const strayBefore = new BinaryPacketBuilder()
    .clearColor(1, 0, 0, 1)
    .draw(3)
    .present()
    .finish();

  // Unclosed frame interrupted by a new BeginFrame.
  const unclosed = new BinaryPacketBuilder()
    .beginFrame()
    .clearColor(0.5, 0.5, 0.5, 1)
    .draw(99)
    .finish();

  // Complete frame 1 with G5 compute opcode + unknown future opcode.
  const validFrame1 = new BinaryPacketBuilder()
    .beginFrame()
    .clearColor(0.1, 0.2, 0.3, 1.0)
    .dispatchCompute(4, 2, 1)
    .packet(777, new Uint8Array([0xde, 0xad, 0xbe, 0xef]), 0x0009)
    .present()
    .finish();

  // Stray packet between Present and next BeginFrame.
  const strayBetween = new BinaryPacketBuilder().nop().finish();

  // Complete frame 2.
  const validFrame2 = buildSimpleFrame(0.9, 6);

  rec.recordChunk(strayBefore);
  assert.equal(rec.exportMetadata().frameCount, 0);

  rec.recordChunk(unclosed);
  assert.equal(rec.exportMetadata().frameCount, 0);

  rec.recordChunk(validFrame1);
  rec.recordChunk(strayBetween);
  rec.recordChunk(validFrame2);

  const meta = rec.exportMetadata();
  assert.equal(meta.frameCount, 2);
  assert.equal(meta.droppedFrames, 0);
  assert.equal(meta.totalPackets, 5 + 4);
  assert.deepEqual(meta.opcodeCounts, {
    BeginFrame: 2,
    ClearColor: 2,
    DispatchCompute: 1,
    'Unknown(777)': 1,
    Draw: 1,
    Present: 2,
  });

  assert.deepEqual(rec.exportTraceBytes({ frameStart: 0, frameCount: 1 }), validFrame1);
  assert.deepEqual(rec.exportTraceBytes({ frameStart: 1, frameCount: 1 }), validFrame2);
});

test('maxFrames and maxBytes bounded ring eviction and oversized frame handling', () => {
  // 1) maxFrames eviction
  const recFrames = new GpuFrameRecorder({ maxFrames: 3, autoStart: true });
  const frames = [
    new BinaryPacketBuilder().beginFrame().nop().present().finish(), // unique Nop in frame 0
    buildSimpleFrame(0.2, 3),
    buildSimpleFrame(0.4, 6),
    buildSimpleFrame(0.6, 9),
    buildSimpleFrame(0.8, 12),
  ];

  for (const f of frames) {
    recFrames.recordChunk(f);
  }

  const metaFrames = recFrames.exportMetadata();
  assert.equal(metaFrames.frameCount, 3);
  assert.equal(metaFrames.droppedFrames, 2);
  assert.equal(metaFrames.capturedFrames, 5);
  // Nop from evicted frame 0 must not appear in opcodeCounts.
  assert.equal(metaFrames.opcodeCounts.Nop, undefined);
  assert.deepEqual(metaFrames.opcodeCounts, {
    BeginFrame: 3,
    ClearColor: 3,
    Draw: 3,
    Present: 3,
  });

  // Retained frames should be frames[2], frames[3], frames[4].
  assert.deepEqual(recFrames.exportTraceBytes({ frameStart: 0, frameCount: 1 }), frames[2]);
  assert.deepEqual(recFrames.exportTraceBytes({ frameStart: 1, frameCount: 1 }), frames[3]);
  assert.deepEqual(recFrames.exportTraceBytes({ frameStart: 2, frameCount: 1 }), frames[4]);

  // 2) maxBytes eviction
  const singleFrameSize = buildSimpleFrame(0.1, 3).byteLength; // 8 + 24 + 24 + 8 = 64 bytes
  assert.equal(singleFrameSize, 64);

  const recBytes = new GpuFrameRecorder({
    maxFrames: 100,
    maxBytes: singleFrameSize * 2, // 128 bytes -> at most 2 of these frames
    autoStart: true,
  });

  const fA = buildSimpleFrame(0.1, 3);
  const fB = buildSimpleFrame(0.2, 6);
  const fC = buildSimpleFrame(0.3, 9);
  recBytes.recordChunk(fA);
  recBytes.recordChunk(fB);
  recBytes.recordChunk(fC);

  const metaBytes = recBytes.exportMetadata();
  assert.equal(metaBytes.frameCount, 2);
  assert.equal(metaBytes.totalBytes, singleFrameSize * 2);
  assert.equal(metaBytes.droppedFrames, 1);
  assert.deepEqual(recBytes.exportTraceBytes({ frameStart: 0, frameCount: 1 }), fB);
  assert.deepEqual(recBytes.exportTraceBytes({ frameStart: 1, frameCount: 1 }), fC);

  // 3) Single frame larger than maxBytes is dropped without evicting valid retained frames.
  const oversized = new BinaryPacketBuilder()
    .beginFrame()
    .writeBuffer(1, 0, new Uint8Array(200))
    .present()
    .finish();
  assert.ok(oversized.byteLength > singleFrameSize * 2);
  const dropRes = recBytes.recordChunk(oversized);
  assert.equal(dropRes.completedFrames, 1);
  assert.equal(dropRes.droppedFrames, 1);

  const metaAfterOversized = recBytes.exportMetadata();
  assert.equal(metaAfterOversized.frameCount, 2);
  assert.equal(metaAfterOversized.droppedFrames, 2);
  assert.deepEqual(recBytes.exportTraceBytes({ frameStart: 0, frameCount: 1 }), fB);
  assert.deepEqual(recBytes.exportTraceBytes({ frameStart: 1, frameCount: 1 }), fC);

  // 4) Ring compaction stress test (> 150 frames into maxFrames = 4).
  const recStress = new GpuFrameRecorder({ maxFrames: 4, autoStart: true });
  let lastFour = [];
  for (let i = 0; i < 160; i++) {
    const f = buildSimpleFrame(i / 255, i + 1);
    recStress.recordChunk(f);
    lastFour.push(f);
    if (lastFour.length > 4) {
      lastFour.shift();
    }
  }
  const stressMeta = recStress.exportMetadata();
  assert.equal(stressMeta.frameCount, 4);
  assert.equal(stressMeta.droppedFrames, 156);
  for (let i = 0; i < 4; i++) {
    assert.deepEqual(recStress.exportTraceBytes({ frameStart: i, frameCount: 1 }), lastFour[i]);
  }
});

test('exportTraceBytes slicing, copy isolation (SAB & subarrays), and JSON-safe exportMetadata', () => {
  const rec = new GpuFrameRecorder({ autoStart: true });
  const f0 = buildSimpleFrame(0.1, 3);
  const f1 = buildSimpleFrame(0.5, 6);
  const f2 = buildSimpleFrame(0.9, 9);

  // Feed f0 via SharedArrayBuffer subarray and mutate the SAB afterward.
  const sab = new SharedArrayBuffer(f0.byteLength + 16);
  const sabView = new Uint8Array(sab, 8, f0.byteLength);
  sabView.set(f0);
  rec.recordChunk(sabView);
  sabView.fill(0xff); // overwrite source SAB

  // Feed f1 via ArrayBuffer.
  rec.recordChunk(f1.buffer.slice(f1.byteOffset, f1.byteOffset + f1.byteLength));

  // Feed f2 via DataView with non-zero byteOffset.
  const padded = new ArrayBuffer(f2.byteLength + 20);
  new Uint8Array(padded, 12, f2.byteLength).set(f2);
  rec.recordChunk(new DataView(padded, 12, f2.byteLength));

  // Verify copy isolation & regular ArrayBuffer backing.
  const all = rec.exportTraceBytes();
  assert.equal(all.buffer instanceof ArrayBuffer, true);
  assert.deepEqual(rec.exportTraceBytes({ frameStart: 0, frameCount: 1 }), f0);
  assert.deepEqual(rec.exportTraceBytes({ frameStart: 1, frameCount: 2 }), (() => {
    const out = new Uint8Array(f1.byteLength + f2.byteLength);
    out.set(f1, 0);
    out.set(f2, f1.byteLength);
    return out;
  })());
  assert.deepEqual(rec.exportTraceBytes({ frameStart: 2 }), f2);
  assert.equal(rec.exportTraceBytes({ frameStart: 10 }).byteLength, 0);
  assert.equal(rec.exportTraceBytes({ frameStart: 0, frameCount: 0 }).byteLength, 0);

  assert.throws(() => rec.exportTraceBytes({ frameStart: -1 }), RangeError);
  assert.throws(() => rec.exportTraceBytes({ frameCount: -1 }), RangeError);
  assert.throws(() => rec.recordChunk('not-binary'), TypeError);

  // exportMetadata must be JSON-safe and decoupled from internal state.
  const meta = rec.exportMetadata();
  const roundTripped = JSON.parse(JSON.stringify(meta));
  assert.deepEqual(roundTripped, meta);

  meta.opcodeCounts.BeginFrame = 9999;
  meta.frames.push({ fake: true });
  const freshMeta = rec.exportMetadata();
  assert.equal(freshMeta.opcodeCounts.BeginFrame, 3);
  assert.equal(freshMeta.frames.length, 3);
});

test('createInterceptor wraps WgpuCommandProcessor non-invasively (direct, drainRing, and detach)', () => {
  const { device, ctx, calls } = createMockWebGpu();
  const processor = new WgpuCommandProcessor(device, ctx);
  const recorder = new GpuFrameRecorder({ autoStart: true });

  const originalProtoMethod = WgpuCommandProcessor.prototype.executeCommandBuffer;
  const interceptor = recorder.createInterceptor(processor);
  assert.equal(interceptor.isAttached(), true);
  assert.equal(interceptor.recorder, recorder);
  assert.equal(interceptor.processor, processor);

  // 1) Direct processor.executeCommandBuffer pass-through
  const frame0 = buildSimpleFrame(0.25, 3);
  const res0 = processor.executeCommandBuffer(frame0);
  assert.deepEqual(res0, { frameCount: 1, presents: 1 });
  assert.equal(recorder.exportMetadata().frameCount, 1);

  // 2) Calling via interceptor handle records once and passes through
  const frame1 = buildSimpleFrame(0.5, 6);
  const res1 = interceptor.executeCommandBuffer(frame1);
  assert.deepEqual(res1, { frameCount: 2, presents: 1 });
  assert.equal(recorder.exportMetadata().frameCount, 2);

  // 3) GpuRingBuffer.drainRing() invokes this.executeCommandBuffer -> recorded automatically!
  const ring = GpuRingBuffer.create(4096);
  processor.attachRing(ring);
  const frame2 = buildSimpleFrame(0.75, 9);
  assert.equal(ring.writeBytes(frame2), true);
  const drainRes = processor.drainRing();
  assert.equal(drainRes.presents, 1);
  assert.equal(drainRes.frameCount, 3);
  assert.equal(recorder.exportMetadata().frameCount, 3);

  // 4) Stopping recorder skips capture while still executing GPU commands.
  recorder.stop();
  const frame3 = buildSimpleFrame(0.9, 12);
  const res3 = processor.executeCommandBuffer(frame3);
  assert.deepEqual(res3, { frameCount: 4, presents: 1 });
  assert.equal(recorder.exportMetadata().frameCount, 3);

  // 5) Detaching restores original prototype method cleanly.
  recorder.start();
  assert.equal(interceptor.detach(), true);
  assert.equal(interceptor.detach(), false); // idempotent
  assert.equal(interceptor.isAttached(), false);
  assert.equal(processor.executeCommandBuffer, originalProtoMethod);
  assert.equal(Object.prototype.hasOwnProperty.call(processor, 'executeCommandBuffer'), false);

  const frame4 = buildSimpleFrame(1.0, 15);
  const res4 = processor.executeCommandBuffer(frame4);
  assert.deepEqual(res4, { frameCount: 5, presents: 1 });
  assert.equal(recorder.exportMetadata().frameCount, 3);
  assert.equal(calls.filter((c) => c.op === 'submit').length, 5);

  // Verify exported trace matches exact frames 0, 1, 2.
  const expected = new Uint8Array(frame0.byteLength + frame1.byteLength + frame2.byteLength);
  expected.set(frame0, 0);
  expected.set(frame1, frame0.byteLength);
  expected.set(frame2, frame0.byteLength + frame1.byteLength);
  assert.deepEqual(recorder.exportTraceBytes(), expected);

  // 6) Module-level createInterceptor helper & frozen processor fallback.
  const executed = [];
  const frozenProcessor = Object.freeze({
    executeCommandBuffer(buf) {
      executed.push(buf.byteLength);
      return { frameCount: executed.length, presents: 1 };
    },
  });
  const frozenHandle = createInterceptor(frozenProcessor, { autoStart: true });
  const fRes = frozenHandle.executeCommandBuffer(frame0);
  assert.deepEqual(fRes, { frameCount: 1, presents: 1 });
  assert.equal(frozenHandle.recorder.exportMetadata().frameCount, 1);
  frozenHandle.detach();
  frozenHandle.executeCommandBuffer(frame1);
  assert.equal(frozenHandle.recorder.exportMetadata().frameCount, 1);
  assert.equal(executed.length, 2);

  assert.throws(() => recorder.createInterceptor(null), TypeError);
  assert.throws(() => recorder.createInterceptor({}), TypeError);
});

test('exported .pg6trace binary stream disassembles cleanly with tools/gpu_trace.py --strict', (t) => {
  const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
  const probe = spawnSync(pythonCmd, ['--version'], { encoding: 'utf-8' });
  if (probe.error || probe.status !== 0) {
    t.skip('python interpreter unavailable for gpu_trace.py cross-check');
    return;
  }

  const vsWgsl = '@vertex fn vs_main(@location(0) p: vec3<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(p, 1.0); }\n';
  const fsWgsl = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.0, 0.0, 1.0); }\n';
  const vertices = new Uint8Array(36);
  for (let i = 0; i < vertices.length; i++) {
    vertices[i] = i & 0xff;
  }

  const frame0 = new BinaryPacketBuilder()
    .beginFrame()
    .clearColor(0.25, 0.5, 0.75, 1.0)
    .createBuffer(1, BUFFER_USAGE.VERTEX | BUFFER_USAGE.COPY_DST, 36)
    .writeBuffer(1, 0, vertices)
    .createShaderModule(2, SHADER_STAGE.VERTEX, vsWgsl)
    .createShaderModule(3, SHADER_STAGE.FRAGMENT, fsWgsl)
    .createRenderPipeline(4, 2, 3, 12, PRIMITIVE_TOPOLOGY.TRIANGLE_LIST, 0)
    .setPipeline(4)
    .setVertexBuffer(0, 1, 0)
    .draw(3, 1, 0, 0)
    .present()
    .finish();

  const frame1 = buildSimpleFrame(0.5, 6);

  const rec = new GpuFrameRecorder({ autoStart: true });
  // Feed split across two chunks.
  const combined = new Uint8Array(frame0.byteLength + frame1.byteLength);
  combined.set(frame0, 0);
  combined.set(frame1, frame0.byteLength);
  rec.recordChunk(combined.subarray(0, 37));
  rec.recordChunk(combined.subarray(37));

  const traceBytes = rec.exportTraceBytes();
  const meta = rec.exportMetadata();

  const traceScript = path.join(REPO_ROOT, 'tools', 'gpu_trace.py');
  const proc = spawnSync(pythonCmd, [traceScript, '-', '--json', '--strict'], {
    input: Buffer.from(traceBytes),
    maxBuffer: 4 * 1024 * 1024,
  });

  assert.equal(proc.status, 0, `gpu_trace.py failed: ${proc.stderr.toString('utf-8')}`);
  const parsed = JSON.parse(proc.stdout.toString('utf-8'));

  assert.equal(parsed.ok, true);
  assert.deepEqual(parsed.diagnostics, []);
  assert.equal(parsed.total_bytes, traceBytes.byteLength);
  assert.equal(parsed.consumed_bytes, traceBytes.byteLength);
  assert.equal(parsed.frame_count, 2);
  assert.equal(parsed.packet_count, meta.totalPackets);
  assert.equal(parsed.draw_calls, 2);
  assert.deepEqual(parsed.opcode_histogram, meta.opcodeCounts);
  assert.equal(parsed.frames[0].complete, true);
  assert.equal(parsed.frames[0].has_begin_frame, true);
  assert.equal(parsed.frames[1].complete, true);
  assert.equal(parsed.frames[1].has_begin_frame, true);
});
