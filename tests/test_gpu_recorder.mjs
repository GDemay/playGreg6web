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
  GpuTraceReplayer,
  createInterceptor,
  splitTraceIntoFrames,
  parseFrameCaptureSidecar,
  FRAME_CAPTURE_SCHEMA,
  FRAME_CAPTURE_SCHEMA_VERSION,
  PACKET_HEADER_BYTES,
  WRITE_TEXTURE_HEADER_BYTES,
  DEFAULT_START_FRAME,
  DEFAULT_MAX_FRAMES,
  DEFAULT_MAX_BYTES,
  DEFAULT_MAX_RECORDED_BYTES,
  DEFAULT_SKIP_WRITE_TEXTURE_ABOVE_BYTES,
  OPCODES,
  OPCODE_NAMES,
  getOpcodeName,
} = recorderMod;

const {
  WgpuCommandProcessor,
  OPCODES: WGPU_OPCODES,
  BUFFER_USAGE,
  SHADER_STAGE,
  PRIMITIVE_TOPOLOGY,
} = wgpuMod;
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

  writeTexture(
    textureId,
    width,
    height,
    data,
    {
      mipLevel = 0,
      originX = 0,
      originY = 0,
      originZ = 0,
      depthOrLayers = 1,
      bytesPerRow = 0,
      rowsPerImage = 0,
      reserved = 0,
      flags = 0,
    } = {}
  ) {
    const src = data instanceof Uint8Array ? data : new Uint8Array(data);
    const buf = new ArrayBuffer(WRITE_TEXTURE_HEADER_BYTES + src.byteLength);
    const view = new DataView(buf);
    view.setUint32(0, textureId, true);
    view.setUint32(4, mipLevel, true);
    view.setUint32(8, originX, true);
    view.setUint32(12, originY, true);
    view.setUint32(16, originZ, true);
    view.setUint32(20, width, true);
    view.setUint32(24, height, true);
    view.setUint32(28, depthOrLayers, true);
    view.setUint32(32, bytesPerRow, true);
    view.setUint32(36, rowsPerImage, true);
    view.setUint32(40, src.byteLength, true);
    view.setUint32(44, reserved, true);
    new Uint8Array(buf, WRITE_TEXTURE_HEADER_BYTES).set(src);
    return this.packet(OPCODES.WRITE_TEXTURE, new Uint8Array(buf), flags);
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

test('ABI v2 opcodes (22..35) match playgta6_abi.h and wgpu_worker.js', () => {
  const expectedV2 = [
    ['CREATE_BIND_GROUP_LAYOUT', 22, 'CreateBindGroupLayout'],
    ['CREATE_PIPELINE_LAYOUT', 23, 'CreatePipelineLayout'],
    ['CREATE_RENDER_PIPELINE2', 24, 'CreateRenderPipeline2'],
    ['CREATE_COMPUTE_PIPELINE2', 25, 'CreateComputePipeline2'],
    ['BEGIN_RENDER_PASS', 26, 'BeginRenderPass'],
    ['END_RENDER_PASS', 27, 'EndRenderPass'],
    ['SET_VIEWPORT', 28, 'SetViewport'],
    ['SET_SCISSOR_RECT', 29, 'SetScissorRect'],
    ['SET_BLEND_CONSTANT', 30, 'SetBlendConstant'],
    ['SET_STENCIL_REFERENCE', 31, 'SetStencilReference'],
    ['CREATE_BIND_GROUP2', 32, 'CreateBindGroup2'],
    ['SET_BIND_GROUP2', 33, 'SetBindGroup2'],
    ['CREATE_SAMPLER2', 34, 'CreateSampler2'],
    ['CREATE_TEXTURE2', 35, 'CreateTexture2'],
  ];

  // Exact parity with WgpuCommandProcessor's OPCODES table (0..35).
  assert.deepEqual(OPCODES, WGPU_OPCODES);
  assert.equal(Object.keys(OPCODES).length, 36);
  assert.equal(Object.keys(OPCODE_NAMES).length, 36);

  for (const [constName, value, canonicalName] of expectedV2) {
    assert.equal(OPCODES[constName], value);
    assert.equal(OPCODE_NAMES[value], canonicalName);
    assert.equal(getOpcodeName(value), canonicalName);
  }
  assert.equal(OPCODES.CREATE_TEXTURE2, 35);
  assert.equal(getOpcodeName(35), 'CreateTexture2');
});

test('splitTraceIntoFrames splits multi-frame, omitted-payload, empty, and truncated traces', () => {
  // 1) Empty trace -> []
  assert.deepEqual(splitTraceIntoFrames(new Uint8Array(0)), []);
  assert.deepEqual(splitTraceIntoFrames(new ArrayBuffer(0)), []);
  assert.throws(() => splitTraceIntoFrames('not-binary'), TypeError);

  // 2) Multi-frame real-capture style trace:
  //    Frame 0 starts with resource creation (no BeginFrame) + zero-length omitted payloads
  //    (--gpu-capture default mode: WriteBuffer/CreateShaderModule/WriteTexture with data_len=0)
  //    plus ABI v2 CreateTexture2, ending with Present.
  const writeTextureOmitted = new Uint8Array(48); // all zeros => data_len = 0
  const createTexture2Payload = new Uint8Array(40);
  const frame0 = new BinaryPacketBuilder()
    .createBuffer(1, BUFFER_USAGE.VERTEX | BUFFER_USAGE.COPY_DST, 1024)
    .writeBuffer(1, 0, new Uint8Array(0)) // omitted WriteBuffer payload (data_len = 0)
    .createShaderModule(2, SHADER_STAGE.VERTEX, '') // omitted CreateShaderModule (code_len = 0)
    .packet(OPCODES.CREATE_TEXTURE2, createTexture2Payload)
    .packet(OPCODES.WRITE_TEXTURE, writeTextureOmitted) // omitted WriteTexture (data_len = 0)
    .packet(OPCODES.BEGIN_RENDER_PASS, new Uint8Array(0)) // zero-length payload test
    .draw(3, 1, 0, 0)
    .packet(OPCODES.END_RENDER_PASS, new Uint8Array(0))
    .present()
    .finish();

  const frame1 = new BinaryPacketBuilder()
    .packet(OPCODES.SET_VIEWPORT, new Uint8Array(24))
    .packet(OPCODES.SET_SCISSOR_RECT, new Uint8Array(16))
    .draw(6, 1, 0, 0)
    .present()
    .finish();

  // Trailing unclosed partial frame (valid packets, no Present)
  const partialFrame2 = new BinaryPacketBuilder()
    .beginFrame()
    .clearColor(0.2, 0.3, 0.4, 1.0)
    .finish();

  const combined = new Uint8Array(
    frame0.byteLength + frame1.byteLength + partialFrame2.byteLength
  );
  combined.set(frame0, 0);
  combined.set(frame1, frame0.byteLength);
  combined.set(partialFrame2, frame0.byteLength + frame1.byteLength);

  // Pass via SharedArrayBuffer subarray to verify copy isolation.
  const sab = new SharedArrayBuffer(combined.byteLength + 8);
  const sabView = new Uint8Array(sab, 4, combined.byteLength);
  sabView.set(combined);
  const frames = splitTraceIntoFrames(sabView);
  sabView.fill(0xaa);

  assert.equal(frames.length, 3);

  assert.equal(frames[0].index, 0);
  assert.equal(frames[0].offset, 0);
  assert.equal(frames[0].packetCount, 9);
  assert.equal(frames[0].byteLength, frame0.byteLength);
  assert.equal(frames[0].hasPresent, true);
  assert.equal(frames[0].complete, true);
  assert.equal(frames[0].truncated, false);
  assert.equal(frames[0].truncatedBytes, 0);
  assert.deepEqual(frames[0].bytes, frame0);
  assert.equal(frames[0].bytes.buffer instanceof ArrayBuffer, true);
  assert.deepEqual(frames[0].opcodeCounts, {
    CreateBuffer: 1,
    WriteBuffer: 1,
    CreateShaderModule: 1,
    CreateTexture2: 1,
    WriteTexture: 1,
    BeginRenderPass: 1,
    Draw: 1,
    EndRenderPass: 1,
    Present: 1,
  });

  assert.equal(frames[1].index, 1);
  assert.equal(frames[1].offset, frame0.byteLength);
  assert.equal(frames[1].packetCount, 4);
  assert.equal(frames[1].byteLength, frame1.byteLength);
  assert.equal(frames[1].hasPresent, true);
  assert.equal(frames[1].complete, true);
  assert.equal(frames[1].truncated, false);
  assert.deepEqual(frames[1].bytes, frame1);
  assert.deepEqual(frames[1].opcodeCounts, {
    SetViewport: 1,
    SetScissorRect: 1,
    Draw: 1,
    Present: 1,
  });

  assert.equal(frames[2].index, 2);
  assert.equal(frames[2].offset, frame0.byteLength + frame1.byteLength);
  assert.equal(frames[2].packetCount, 2);
  assert.equal(frames[2].byteLength, partialFrame2.byteLength);
  assert.equal(frames[2].hasPresent, false);
  assert.equal(frames[2].complete, false);
  assert.equal(frames[2].truncated, false);
  assert.equal(frames[2].truncatedBytes, 0);
  assert.deepEqual(frames[2].bytes, partialFrame2);

  // 3) Truncated header (< 8 bytes)
  const truncatedHeader = splitTraceIntoFrames(new Uint8Array([1, 0, 0, 0, 5]));
  assert.equal(truncatedHeader.length, 1);
  assert.equal(truncatedHeader[0].packetCount, 0);
  assert.equal(truncatedHeader[0].byteLength, 0);
  assert.equal(truncatedHeader[0].hasPresent, false);
  assert.equal(truncatedHeader[0].truncated, true);
  assert.equal(truncatedHeader[0].truncatedBytes, 5);

  // 4) Truncated payload (valid frame0 followed by half of a ClearColor packet)
  const truncatedPayloadStream = new Uint8Array(frame0.byteLength + 12);
  truncatedPayloadStream.set(frame0, 0);
  // ClearColor header declaring 16-byte payload, but only 4 payload bytes follow
  const truncView = new DataView(
    truncatedPayloadStream.buffer,
    frame0.byteLength,
    12
  );
  truncView.setUint16(0, OPCODES.CLEAR_COLOR, true);
  truncView.setUint16(2, 0, true);
  truncView.setUint32(4, 16, true);

  const splitTrunc = splitTraceIntoFrames(truncatedPayloadStream);
  assert.equal(splitTrunc.length, 2);
  assert.equal(splitTrunc[0].complete, true);
  assert.equal(splitTrunc[0].truncated, false);
  assert.equal(splitTrunc[1].packetCount, 0);
  assert.equal(splitTrunc[1].byteLength, 0);
  assert.equal(splitTrunc[1].hasPresent, false);
  assert.equal(splitTrunc[1].truncated, true);
  assert.equal(splitTrunc[1].truncatedBytes, 12);
});

test('parseFrameCaptureSidecar validates schema, normalizes fields, and strips host paths', () => {
  assert.equal(FRAME_CAPTURE_SCHEMA, 'playgta6-frame-capture');
  assert.equal(FRAME_CAPTURE_SCHEMA_VERSION, 1);

  const rawSidecar = {
    schema: 'playgta6-frame-capture',
    schema_version: 1,
    trace: 'C:\\Users\\alice\\secret_local\\captures\\frame1.pg6trace',
    payloads: 'omitted',
    frames_requested: 2,
    frames_complete: 2,
    presents_seen: 3,
    packets: 13,
    bytes: 512,
    omitted_bytes: 4096,
    frames: [
      { index: 0, offset: 0, bytes: 320, packets: 9 },
      { index: 1, offset: 320, bytes: 192, packets: 4 },
    ],
    presents: [
      {
        index: 0,
        wall_ms: 14.25,
        dispatches: 120,
        import_calls: 45,
        reserved_bytes: 1048576,
        committed_bytes: 524288,
      },
    ],
    omitted: [
      { packet: 1, opcode: 4, length: 1024, fnv1a64: '0x0123456789ABCDEF' },
      { packet: 2, opcode: 5, length: 3072, fnv1a64: '0xfedcba9876543210' },
    ],
    // Extra host-specific property that must never be copied through.
    host_cwd: 'C:\\Users\\alice\\secret_local',
  };

  // Parse from JSON string
  const parsedFromJson = parseFrameCaptureSidecar(JSON.stringify(rawSidecar));
  assert.equal(parsedFromJson.schema, 'playgta6-frame-capture');
  assert.equal(parsedFromJson.schemaVersion, 1);
  assert.equal(parsedFromJson.trace, 'frame1.pg6trace');
  assert.equal(parsedFromJson.payloads, 'omitted');
  assert.equal(parsedFromJson.payloadsOmitted, true);
  assert.equal(parsedFromJson.framesRequested, 2);
  assert.equal(parsedFromJson.framesComplete, 2);
  assert.equal(parsedFromJson.presentsSeen, 3);
  assert.equal(parsedFromJson.packets, 13);
  assert.equal(parsedFromJson.bytes, 512);
  assert.equal(parsedFromJson.omittedBytes, 4096);
  assert.equal(parsedFromJson.frames.length, 2);
  assert.deepEqual(parsedFromJson.presents[0], {
    index: 0,
    wallMs: 14.25,
    dispatches: 120,
    importCalls: 45,
    reservedBytes: 1048576,
    committedBytes: 524288,
  });
  assert.deepEqual(parsedFromJson.omitted[0], {
    packet: 1,
    opcode: 4,
    opcodeName: 'WriteBuffer',
    length: 1024,
    fnv1a64: '0x0123456789abcdef',
  });
  assert.equal(parsedFromJson.host_cwd, undefined);

  const serialized = JSON.stringify(parsedFromJson);
  assert.equal(serialized.includes('alice'), false);
  assert.equal(serialized.includes('C:\\'), false);
  assert.equal(serialized.includes('secret_local'), false);

  // POSIX & UNC path sanitization from plain object
  const parsedPosix = parseFrameCaptureSidecar({
    schema: 'playgta6-frame-capture',
    schema_version: 1,
    trace: '/home/bob/work/capture.pg6trace',
    payloads: 'kept',
    frames: [],
  });
  assert.equal(parsedPosix.trace, 'capture.pg6trace');
  assert.equal(parsedPosix.payloadsOmitted, false);
  assert.deepEqual(parsedPosix.presents, []);
  assert.deepEqual(parsedPosix.omitted, []);

  const parsedUnc = parseFrameCaptureSidecar({
    schema: 'playgta6-frame-capture',
    schema_version: 1,
    trace: '\\\\fileserver\\share\\run2.pg6trace',
    payloads: 'omitted',
    frames: [],
  });
  assert.equal(parsedUnc.trace, 'run2.pg6trace');

  // Bad schemas & invalid field rejections
  assert.throws(() => parseFrameCaptureSidecar('{bad json'), SyntaxError);
  assert.throws(() => parseFrameCaptureSidecar(null), TypeError);
  assert.throws(() => parseFrameCaptureSidecar([]), TypeError);
  assert.throws(
    () =>
      parseFrameCaptureSidecar({
        schema: 'wrong-schema',
        schema_version: 1,
        payloads: 'omitted',
        frames: [],
      }),
    Error
  );
  assert.throws(
    () =>
      parseFrameCaptureSidecar({
        schema: 'playgta6-frame-capture',
        schema_version: 2,
        payloads: 'omitted',
        frames: [],
      }),
    Error
  );
  assert.throws(
    () =>
      parseFrameCaptureSidecar({
        schema: 'playgta6-frame-capture',
        schema_version: 1,
        payloads: 'invalid',
        frames: [],
      }),
    Error
  );
  assert.throws(
    () =>
      parseFrameCaptureSidecar({
        schema: 'playgta6-frame-capture',
        schema_version: 1,
        payloads: 'omitted',
        frames: 'not-array',
      }),
    TypeError
  );
  assert.throws(
    () =>
      parseFrameCaptureSidecar({
        schema: 'playgta6-frame-capture',
        schema_version: 1,
        payloads: 'omitted',
        frames: [{ index: -1, offset: 0, bytes: 8, packets: 1 }],
      }),
    RangeError
  );
  assert.throws(
    () =>
      parseFrameCaptureSidecar({
        schema: 'playgta6-frame-capture',
        schema_version: 1,
        payloads: 'omitted',
        frames: [],
        omitted: [{ packet: 0, opcode: 4, length: 16, fnv1a64: 'not-hex' }],
      }),
    TypeError
  );
});

test('GpuTraceReplayer replays multi-frame traces, captures errors, and integrates with WgpuCommandProcessor', () => {
  const frame0 = buildSimpleFrame(0.1, 3);
  const frame1 = buildSimpleFrame(0.5, 6);
  const frame2 = buildSimpleFrame(0.9, 9);
  const trace = new Uint8Array(frame0.byteLength + frame1.byteLength + frame2.byteLength);
  trace.set(frame0, 0);
  trace.set(frame1, frame0.byteLength);
  trace.set(frame2, frame0.byteLength + frame1.byteLength);

  const sidecar = {
    schema: 'playgta6-frame-capture',
    schema_version: 1,
    trace: 'C:\\Users\\dev\\captures\\demo.pg6trace',
    payloads: 'omitted',
    frames_requested: 3,
    frames_complete: 3,
    presents_seen: 3,
    packets: 12,
    bytes: trace.byteLength,
    omitted_bytes: 256,
    frames: [
      { index: 0, offset: 0, bytes: frame0.byteLength, packets: 4 },
      { index: 1, offset: frame0.byteLength, bytes: frame1.byteLength, packets: 4 },
      {
        index: 2,
        offset: frame0.byteLength + frame1.byteLength,
        bytes: frame2.byteLength,
        packets: 4,
      },
    ],
    omitted: [{ packet: 1, opcode: 4, length: 256, fnv1a64: '0x0000000000001234' }],
  };

  // Deterministic clock for testing durationMs.
  let clockMs = 100;
  const replayer = new GpuTraceReplayer({
    now() {
      const t = clockMs;
      clockMs += 2.5;
      return t;
    },
  });

  assert.equal(replayer.getFrameCount(), 0);
  assert.equal(replayer.getSummary().loaded, false);

  assert.equal(replayer.loadTrace(trace, sidecar), replayer);
  assert.equal(replayer.getFrameCount(), 3);

  // 1) Mock processor implementing executeBatch(bytes) with an error injected on frame 1
  const receivedSlices = [];
  const mockBatchProcessor = {
    executeBatch(bytes) {
      assert.ok(bytes instanceof Uint8Array);
      receivedSlices.push(new Uint8Array(bytes));
      if (receivedSlices.length === 2) {
        throw new Error('simulated WebGPU validation error on frame 1');
      }
      return { executedBytes: bytes.byteLength };
    },
  };

  const run = replayer.replayAll(mockBatchProcessor);
  assert.equal(receivedSlices.length, 3);
  assert.deepEqual(receivedSlices[0], frame0);
  assert.deepEqual(receivedSlices[1], frame1);
  assert.deepEqual(receivedSlices[2], frame2);

  assert.equal(run.ok, false);
  assert.equal(run.frameCount, 3);
  assert.equal(run.replayedFrames, 3);
  assert.equal(run.succeededFrames, 2);
  assert.equal(run.failedFrames, 1);
  assert.equal(run.totalPackets, 12);
  assert.equal(run.totalBytes, trace.byteLength);
  assert.equal(run.totalDurationMs, 7.5);
  assert.equal(run.payloadsOmitted, true);
  assert.equal(run.omittedPayloadCount, 1);
  assert.equal(run.omittedBytes, 256);
  assert.deepEqual(run.errors, [
    { frameIndex: 1, error: 'simulated WebGPU validation error on frame 1' },
  ]);

  assert.equal(run.frames[0].frameIndex, 0);
  assert.equal(run.frames[0].packetCount, 4);
  assert.equal(run.frames[0].byteLength, frame0.byteLength);
  assert.equal(run.frames[0].ok, true);
  assert.equal(run.frames[0].error, null);
  assert.equal(run.frames[0].durationMs, 2.5);

  assert.equal(run.frames[1].frameIndex, 1);
  assert.equal(run.frames[1].ok, false);
  assert.equal(run.frames[1].error, 'simulated WebGPU validation error on frame 1');
  assert.equal(run.frames[1].durationMs, 2.5);

  assert.equal(run.frames[2].frameIndex, 2);
  assert.equal(run.frames[2].ok, true);
  assert.equal(run.frames[2].error, null);

  const summary = replayer.getSummary();
  assert.deepEqual(JSON.parse(JSON.stringify(summary)), summary);
  assert.equal(summary.loaded, true);
  assert.equal(summary.frameCount, 3);
  assert.equal(summary.completeFrameCount, 3);
  assert.equal(summary.truncatedFrameCount, 0);
  assert.equal(summary.totalPackets, 12);
  assert.equal(summary.totalBytes, trace.byteLength);
  assert.equal(summary.sidecar.trace, 'demo.pg6trace');
  assert.equal(summary.replay.replayedFrames, 3);
  assert.equal(summary.replay.succeededFrames, 2);
  assert.equal(summary.replay.failedFrames, 1);
  assert.deepEqual(summary.replay.errors, [
    { frameIndex: 1, error: 'simulated WebGPU validation error on frame 1' },
  ]);

  // 2) stopOnError halts immediately on first failing frame
  let stopCalls = 0;
  const stopRun = replayer.replayAll(
    {
      executeBatch() {
        stopCalls += 1;
        if (stopCalls === 1) {
          throw new Error('boom on frame 0');
        }
      },
    },
    { stopOnError: true }
  );
  assert.equal(stopRun.ok, false);
  assert.equal(stopRun.replayedFrames, 1);
  assert.equal(stopRun.failedFrames, 1);
  assert.equal(stopCalls, 1);

  // 3) Replaying directly into a real WgpuCommandProcessor (via executeCommandBuffer)
  const { device, ctx, calls } = createMockWebGpu();
  const wgpuProcessor = new WgpuCommandProcessor(device, ctx);
  const wgpuReplayer = new GpuTraceReplayer().loadTrace(trace);
  const wgpuRun = wgpuReplayer.replayAll(wgpuProcessor);
  assert.equal(wgpuRun.ok, true);
  assert.equal(wgpuRun.succeededFrames, 3);
  assert.equal(wgpuRun.failedFrames, 0);
  assert.equal(calls.filter((c) => c.op === 'submit').length, 3);
  assert.deepEqual(wgpuRun.frames[2].result, { frameCount: 3, presents: 1 });

  // 4) Validation errors
  assert.throws(() => replayer.replayFrame(-1, mockBatchProcessor), RangeError);
  assert.throws(() => replayer.replayFrame(3, mockBatchProcessor), RangeError);
  assert.throws(() => replayer.replayFrame(0, null), TypeError);
  assert.throws(() => replayer.replayAll({}), TypeError);
});

function concatUint8Arrays(arrays) {
  const total = arrays.reduce((sum, a) => sum + a.byteLength, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const a of arrays) {
    out.set(a, offset);
    offset += a.byteLength;
  }
  return out;
}

test('W124: startFrame: 3 skips frames 0..2 and captures cleanly from frame 3 onward', () => {
  assert.equal(DEFAULT_START_FRAME, 0);
  assert.equal(GpuFrameRecorder.DEFAULT_START_FRAME, 0);

  const frames = [
    buildSimpleFrame(0.0, 3),
    buildSimpleFrame(0.1, 6),
    buildSimpleFrame(0.2, 9),
    buildSimpleFrame(0.3, 12),
    buildSimpleFrame(0.4, 15),
    buildSimpleFrame(0.5, 18),
  ];

  // 1) Via startCapture({ startFrame: 3 })
  const rec = new GpuFrameRecorder();
  assert.equal(rec.startCapture({ startFrame: 3 }), rec);
  assert.equal(rec.isRecording(), true);

  for (const f of frames) {
    rec.recordChunk(f);
  }

  const stats = rec.getStats();
  assert.equal(stats.startFrame, 3);
  assert.equal(stats.observedFrames, 6);
  assert.equal(stats.skippedFrames, 3);
  assert.equal(stats.capturedFrames, 3);
  assert.equal(stats.frameCount, 3);
  assert.equal(stats.capturedBytes, frames[3].byteLength + frames[4].byteLength + frames[5].byteLength);
  assert.equal(stats.totalBytes, stats.capturedBytes);
  assert.equal(stats.droppedBudgetFrames, 0);
  assert.equal(stats.budgetReached, false);
  assert.equal(stats.recording, true);

  const expected = concatUint8Arrays([frames[3], frames[4], frames[5]]);
  assert.deepEqual(rec.exportTraceBytes(), expected);

  const meta = rec.exportMetadata();
  assert.deepEqual(
    meta.frames.map((f) => f.sequence),
    [3, 4, 5]
  );

  // 2) Via constructor options ({ startFrame: 3, autoStart: true }) with huge skipped frames
  //    that exceed maxBytes — skipped frames must not trip budgetReached!
  const hugeSkipped = new BinaryPacketBuilder()
    .beginFrame()
    .writeBuffer(1, 0, new Uint8Array(512))
    .present()
    .finish();
  const smallFrame3 = buildSimpleFrame(0.75, 9);
  const smallFrame4 = buildSimpleFrame(0.85, 12);

  const ctorRec = new GpuFrameRecorder({
    startFrame: 3,
    maxBytes: smallFrame3.byteLength * 2,
    autoStart: true,
  });
  ctorRec.recordChunk(hugeSkipped);
  ctorRec.recordChunk(hugeSkipped);
  ctorRec.recordChunk(hugeSkipped);
  ctorRec.recordChunk(smallFrame3);
  ctorRec.recordChunk(smallFrame4);

  const ctorStats = ctorRec.getStats();
  assert.equal(ctorStats.startFrame, 3);
  assert.equal(ctorStats.skippedFrames, 3);
  assert.equal(ctorStats.capturedFrames, 2);
  assert.equal(ctorStats.capturedBytes, smallFrame3.byteLength + smallFrame4.byteLength);
  assert.equal(ctorStats.budgetReached, false);
  assert.equal(ctorStats.droppedBudgetFrames, 0);
  assert.deepEqual(ctorRec.exportTraceBytes(), concatUint8Arrays([smallFrame3, smallFrame4]));

  // 3) Option validation (constructor and startCapture leave state intact on error)
  assert.throws(() => new GpuFrameRecorder({ startFrame: -1 }), RangeError);
  assert.throws(() => new GpuFrameRecorder({ startFrame: 1.5 }), RangeError);
  assert.throws(() => new GpuFrameRecorder({ startFrame: '3' }), RangeError);
  assert.throws(() => ctorRec.startCapture({ startFrame: -2 }), RangeError);
  assert.throws(() => ctorRec.startCapture({ maxFrames: 0 }), RangeError);
  assert.throws(() => ctorRec.startCapture({ maxBytes: 0 }), RangeError);
  assert.equal(ctorRec.getStats().capturedFrames, 2);
});

test('W124: maxFrames: 2 stops cleanly after 2 captured frames with budgetReached and droppedBudgetFrames', () => {
  const frames = [
    buildSimpleFrame(0.1, 3),
    buildSimpleFrame(0.2, 6),
    buildSimpleFrame(0.3, 9),
    buildSimpleFrame(0.4, 12),
    buildSimpleFrame(0.5, 15),
  ];

  // 1) Frame-by-frame delivery via startCapture({ maxFrames: 2 })
  const rec = new GpuFrameRecorder();
  rec.startCapture({ maxFrames: 2 });

  rec.recordChunk(frames[0]);
  rec.recordChunk(frames[1]);
  const midStats = rec.getStats();
  assert.equal(midStats.capturedFrames, 2);
  assert.equal(midStats.budgetReached, false);
  assert.equal(midStats.droppedBudgetFrames, 0);
  assert.equal(rec.isRecording(), true);

  // Frame 2 exceeds maxFrames -> budgetReached=true, droppedBudgetFrames>0, clean auto-stop.
  const hitRes = rec.recordChunk(frames[2]);
  assert.equal(hitRes.completedFrames, 0);
  assert.equal(hitRes.droppedFrames, 1);

  const afterHit = rec.getStats();
  assert.equal(afterHit.capturedFrames, 2);
  assert.equal(afterHit.frameCount, 2);
  assert.equal(afterHit.capturedBytes, frames[0].byteLength + frames[1].byteLength);
  assert.equal(afterHit.budgetReached, true);
  assert.ok(afterHit.droppedBudgetFrames > 0);
  assert.equal(afterHit.droppedBudgetFrames, 1);
  assert.equal(afterHit.recording, false);
  assert.equal(rec.isRecording(), false);

  // Subsequent frames after clean stop are ignored without corrupting captured frames.
  const ignoredRes = rec.recordChunk(frames[3]);
  assert.equal(ignoredRes.recordedBytes, 0);
  assert.equal(rec.getStats().capturedFrames, 2);
  assert.deepEqual(rec.exportTraceBytes(), concatUint8Arrays([frames[0], frames[1]]));

  // 2) Single multi-frame chunk delivery with combined startFrame: 2 & maxFrames: 2
  const windowedRec = new GpuFrameRecorder();
  windowedRec.startCapture({ startFrame: 2, maxFrames: 2 });
  windowedRec.recordChunk(concatUint8Arrays(frames));

  const wStats = windowedRec.getStats();
  assert.equal(wStats.startFrame, 2);
  assert.equal(wStats.skippedFrames, 2);
  assert.equal(wStats.capturedFrames, 2);
  assert.equal(wStats.budgetReached, true);
  assert.ok(wStats.droppedBudgetFrames > 0);
  assert.equal(wStats.recording, false);
  assert.deepEqual(windowedRec.exportTraceBytes(), concatUint8Arrays([frames[2], frames[3]]));

  // 3) Re-starting a new capture resets budget state cleanly.
  windowedRec.startCapture({ startFrame: 0, maxFrames: 3 });
  const resetStats = windowedRec.getStats();
  assert.equal(resetStats.startFrame, 0);
  assert.equal(resetStats.capturedFrames, 0);
  assert.equal(resetStats.capturedBytes, 0);
  assert.equal(resetStats.droppedBudgetFrames, 0);
  assert.equal(resetStats.budgetReached, false);
  assert.equal(resetStats.recording, true);

  // 4) Interceptor continues passing frames to WgpuCommandProcessor after recorder hits budget.
  const { device, ctx, calls } = createMockWebGpu();
  const processor = new WgpuCommandProcessor(device, ctx);
  const intRec = new GpuFrameRecorder();
  const handle = intRec.createInterceptor(processor);
  intRec.startCapture({ startFrame: 1, maxFrames: 2 });

  for (const f of frames) {
    processor.executeCommandBuffer(f);
  }
  handle.detach();

  assert.equal(calls.filter((c) => c.op === 'submit').length, 5);
  assert.equal(intRec.getStats().capturedFrames, 2);
  assert.equal(intRec.getStats().budgetReached, true);
  assert.ok(intRec.getStats().droppedBudgetFrames > 0);
  assert.deepEqual(intRec.exportTraceBytes(), concatUint8Arrays([frames[1], frames[2]]));
});

test('W124: maxBytes enforces byte budget cleanly without partial or corrupt packets in .pg6trace', (t) => {
  const frame0 = buildSimpleFrame(0.1, 3); // 64 bytes
  const frame1 = buildSimpleFrame(0.2, 6); // 64 bytes
  const frame2 = buildSimpleFrame(0.3, 9); // 64 bytes
  assert.equal(frame0.byteLength, 64);

  // 1) Budget hit mid-frame (maxBytes = 100: frame0=64 fits, frame1 overflows on Draw packet)
  //    Verify both single-chunk and 1-byte-chunk delivery leave ONLY intact frame0!
  for (const oneByteChunks of [false, true]) {
    const rec = new GpuFrameRecorder();
    rec.startCapture({ maxBytes: 100 });
    const stream = concatUint8Arrays([frame0, frame1, frame2]);

    if (oneByteChunks) {
      for (let i = 0; i < stream.byteLength; i++) {
        rec.recordChunk(stream.subarray(i, i + 1));
      }
    } else {
      rec.recordChunk(stream);
    }

    const stats = rec.getStats();
    assert.equal(stats.capturedFrames, 1);
    assert.equal(stats.capturedBytes, 64);
    assert.ok(stats.capturedBytes <= 100);
    assert.equal(stats.budgetReached, true);
    assert.ok(stats.droppedBudgetFrames > 0);
    assert.equal(stats.recording, false);

    const exported = rec.exportTraceBytes();
    assert.deepEqual(exported, frame0);

    // splitTraceIntoFrames must report 1 complete, non-truncated frame.
    const split = splitTraceIntoFrames(exported);
    assert.equal(split.length, 1);
    assert.equal(split[0].complete, true);
    assert.equal(split[0].truncated, false);
    assert.equal(split[0].truncatedBytes, 0);
  }

  // 2) Oversized split packet header exceeding maxBytes stops immediately without OOM buffering.
  const recSplitHuge = new GpuFrameRecorder();
  recSplitHuge.startCapture({ maxBytes: 128 });
  recSplitHuge.recordChunk(frame0); // 64 bytes retained

  // Start frame 1 with BeginFrame (8B) + WriteBuffer header claiming 1 MiB payload (only 16B sent).
  const hugePartial = new Uint8Array(8 + 8 + 16);
  const hugeView = new DataView(hugePartial.buffer);
  // BeginFrame header
  hugeView.setUint16(0, OPCODES.BEGIN_FRAME, true);
  hugeView.setUint16(2, 0, true);
  hugeView.setUint32(4, 0, true);
  // WriteBuffer header claiming 1,048,576-byte payload
  hugeView.setUint16(8, OPCODES.WRITE_BUFFER, true);
  hugeView.setUint16(10, 0, true);
  hugeView.setUint32(12, 1024 * 1024, true);

  recSplitHuge.recordChunk(hugePartial);
  const splitStats = recSplitHuge.getStats();
  assert.equal(splitStats.budgetReached, true);
  assert.equal(splitStats.droppedBudgetFrames, 1);
  assert.equal(splitStats.capturedFrames, 1);
  assert.equal(splitStats.capturedBytes, 64);
  assert.equal(splitStats.recording, false);
  assert.deepEqual(recSplitHuge.exportTraceBytes(), frame0);

  // 3) First frame itself exceeds maxBytes -> 0 frames captured, empty valid trace.
  const recTiny = new GpuFrameRecorder({ maxBytes: 32, stopOnBudget: true, autoStart: true });
  recTiny.recordChunk(frame0);
  const tinyStats = recTiny.getStats();
  assert.equal(tinyStats.capturedFrames, 0);
  assert.equal(tinyStats.capturedBytes, 0);
  assert.equal(tinyStats.budgetReached, true);
  assert.equal(tinyStats.droppedBudgetFrames, 1);
  assert.equal(tinyStats.recording, false);
  assert.equal(recTiny.exportTraceBytes().byteLength, 0);

  // 4) Cross-check windowed + byte-budgeted .pg6trace with tools/gpu_trace.py --strict
  //    --shader-summary --binding-audit.
  const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
  const probe = spawnSync(pythonCmd, ['--version'], { encoding: 'utf-8' });
  if (probe.error || probe.status !== 0) {
    t.skip('python interpreter unavailable for gpu_trace.py cross-check');
    return;
  }

  const vsWgsl = '@vertex fn vs_main(@location(0) p: vec3<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(p, 1.0); }\n';
  const fsWgsl = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.0, 1.0, 0.0, 1.0); }\n';
  const richFrame = new BinaryPacketBuilder()
    .beginFrame()
    .clearColor(0.1, 0.2, 0.3, 1.0)
    .createShaderModule(1, SHADER_STAGE.VERTEX, vsWgsl)
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, fsWgsl)
    .createRenderPipeline(3, 1, 2, 12, PRIMITIVE_TOPOLOGY.TRIANGLE_LIST, 0)
    .setPipeline(3)
    .draw(3, 1, 0, 0)
    .present()
    .finish();

  const recAudit = new GpuFrameRecorder();
  recAudit.startCapture({
    startFrame: 1,
    maxFrames: 10,
    maxBytes: richFrame.byteLength + 40, // fits frame 1, overflows mid-frame on frame 2
  });
  recAudit.recordChunk(frame0); // skipped (frame 0)
  recAudit.recordChunk(richFrame); // captured (frame 1)
  recAudit.recordChunk(richFrame); // exceeds maxBytes mid-frame -> cleanly dropped

  assert.equal(recAudit.getStats().capturedFrames, 1);
  assert.equal(recAudit.getStats().budgetReached, true);
  const traceBytes = recAudit.exportTraceBytes();
  assert.deepEqual(traceBytes, richFrame);

  const traceScript = path.join(REPO_ROOT, 'tools', 'gpu_trace.py');
  const proc = spawnSync(
    pythonCmd,
    [traceScript, '-', '--json', '--strict', '--shader-summary', '--binding-audit'],
    {
      input: Buffer.from(traceBytes),
      maxBuffer: 4 * 1024 * 1024,
    }
  );
  assert.equal(proc.status, 0, `gpu_trace.py failed: ${proc.stderr.toString('utf-8')}`);
  const parsed = JSON.parse(proc.stdout.toString('utf-8'));
  assert.equal(parsed.ok, true);
  assert.deepEqual(parsed.diagnostics, []);
  assert.equal(parsed.frame_count, 1);
  assert.equal(parsed.total_bytes, richFrame.byteLength);
  assert.equal(parsed.consumed_bytes, richFrame.byteLength);
});

test('W124: default behavior without options remains 100% backwards-compatible and getStats() is JSON-safe', () => {
  const rec = new GpuFrameRecorder();
  const initial = rec.getStats();
  assert.equal(initial.abiVersion, 1);
  assert.equal(initial.frameCount, 0);
  assert.equal(initial.totalBytes, 0);
  assert.equal(initial.totalPackets, 0);
  assert.deepEqual(initial.opcodeCounts, {});
  assert.equal(initial.droppedFrames, 0);
  assert.equal(initial.capturedFrames, 0);
  assert.equal(initial.capturedBytes, 0);
  assert.equal(initial.droppedBudgetFrames, 0);
  assert.equal(initial.budgetReached, false);
  assert.equal(initial.startFrame, 0);
  assert.equal(initial.stopOnBudget, false);
  assert.equal(initial.recording, false);
  assert.equal(initial.maxFrames, DEFAULT_MAX_FRAMES);
  assert.equal(initial.maxBytes, DEFAULT_MAX_BYTES);

  // Default mode retains rolling ring eviction when started with start().
  rec.start();
  const frames = [];
  for (let i = 0; i < DEFAULT_MAX_FRAMES + 3; i++) {
    const f = buildSimpleFrame(i / 100, i + 1);
    frames.push(f);
    rec.recordChunk(f);
  }

  const stats = rec.getStats();
  assert.equal(stats.frameCount, DEFAULT_MAX_FRAMES);
  assert.equal(stats.capturedFrames, DEFAULT_MAX_FRAMES + 3);
  assert.equal(stats.droppedFrames, 3);
  assert.equal(stats.droppedBudgetFrames, 0);
  assert.equal(stats.budgetReached, false);
  assert.equal(stats.recording, true);
  assert.equal(stats.capturedBytes, stats.totalBytes);
  assert.deepEqual(JSON.parse(JSON.stringify(stats)), stats);

  // Mutating returned getStats().opcodeCounts must not affect internal state.
  stats.opcodeCounts.BeginFrame = 999999;
  assert.equal(rec.getStats().opcodeCounts.BeginFrame, DEFAULT_MAX_FRAMES);

  // Retained frames in default ring mode are the newest 60 frames.
  assert.deepEqual(rec.exportTraceBytes({ frameStart: 0, frameCount: 1 }), frames[3]);
  assert.deepEqual(
    rec.exportTraceBytes({ frameStart: DEFAULT_MAX_FRAMES - 1, frameCount: 1 }),
    frames[frames.length - 1]
  );
});

test('W216: full capture default (skipWriteTextureAboveBytes = 0), option validation, and alias support', () => {
  assert.equal(WRITE_TEXTURE_HEADER_BYTES, 48);
  assert.equal(DEFAULT_MAX_RECORDED_BYTES, DEFAULT_MAX_BYTES);
  assert.equal(DEFAULT_SKIP_WRITE_TEXTURE_ABOVE_BYTES, 0);
  assert.equal(GpuFrameRecorder.WRITE_TEXTURE_HEADER_BYTES, 48);
  assert.equal(GpuFrameRecorder.DEFAULT_MAX_RECORDED_BYTES, DEFAULT_MAX_BYTES);
  assert.equal(GpuFrameRecorder.DEFAULT_SKIP_WRITE_TEXTURE_ABOVE_BYTES, 0);

  // Default recorder retains large WriteTexture payloads verbatim.
  const rec = new GpuFrameRecorder({ autoStart: true });
  const initialStats = rec.getStats();
  assert.equal(initialStats.skipWriteTextureAboveBytes, 0);
  assert.equal(initialStats.maxRecordedBytes, DEFAULT_MAX_BYTES);
  assert.equal(initialStats.recordedPackets, 0);
  assert.equal(initialStats.recordedBytes, 0);
  assert.equal(initialStats.skippedLargeUploadPackets, 0);
  assert.equal(initialStats.skippedLargeUploadBytes, 0);
  assert.equal(initialStats.droppedRingPackets, 0);

  const largeTexels = new Uint8Array(512 * 1024);
  largeTexels.fill(0x5a);
  const fullFrame = new BinaryPacketBuilder()
    .beginFrame()
    .clearColor(0.1, 0.2, 0.3, 1.0)
    .writeTexture(42, 1024, 512, largeTexels, { bytesPerRow: 1024, rowsPerImage: 512 })
    .draw(6, 1, 0, 0)
    .present()
    .finish();

  const chunkRes = rec.recordChunk(fullFrame);
  assert.equal(chunkRes.completedFrames, 1);
  assert.equal(chunkRes.skippedLargeUploadPackets, 0);
  assert.equal(chunkRes.skippedLargeUploadBytes, 0);

  const stats = rec.getStats();
  assert.equal(stats.recordedPackets, 5);
  assert.equal(stats.recordedBytes, fullFrame.byteLength);
  assert.equal(stats.skippedLargeUploadPackets, 0);
  assert.equal(stats.skippedLargeUploadBytes, 0);
  assert.equal(stats.droppedRingPackets, 0);
  assert.deepEqual(rec.exportTraceBytes(), fullFrame);

  // Alias skipLargeWriteTextureBytes works identically to skipWriteTextureAboveBytes.
  const aliasRec = new GpuFrameRecorder({
    autoStart: true,
    skipLargeWriteTextureBytes: 64 * 1024,
  });
  assert.equal(aliasRec.getStats().skipWriteTextureAboveBytes, 64 * 1024);
  aliasRec.recordChunk(fullFrame);
  assert.equal(aliasRec.getStats().skippedLargeUploadPackets, 1);
  assert.equal(aliasRec.getStats().recordedPackets, 4);

  // Option validation in constructor and startCapture leaves existing state untouched.
  assert.throws(() => new GpuFrameRecorder({ maxRecordedBytes: 0 }), RangeError);
  assert.throws(() => new GpuFrameRecorder({ maxRecordedBytes: -100 }), RangeError);
  assert.throws(() => new GpuFrameRecorder({ maxRecordedBytes: 12.5 }), RangeError);
  assert.throws(() => new GpuFrameRecorder({ skipWriteTextureAboveBytes: -1 }), RangeError);
  assert.throws(() => new GpuFrameRecorder({ skipWriteTextureAboveBytes: 1.5 }), RangeError);
  assert.throws(() => new GpuFrameRecorder({ skipLargeWriteTextureBytes: -8 }), RangeError);
  assert.throws(() => rec.startCapture({ maxRecordedBytes: 0 }), RangeError);
  assert.throws(() => rec.startCapture({ skipWriteTextureAboveBytes: -4 }), RangeError);
  assert.throws(() => rec.startCapture({ skipLargeWriteTextureBytes: 'bad' }), RangeError);
  assert.equal(rec.getStats().recordedPackets, 5);
  assert.equal(rec.getStats().recordedBytes, fullFrame.byteLength);
});

test('W216: skipWriteTextureAboveBytes elides 8.3 MiB 4K Y planes while preserving UI uploads, pipelines, draws, and replayability', (t) => {
  const Y_WIDTH = 3840;
  const Y_HEIGHT = 2160;
  const Y_PLANE_BYTES = Y_WIDTH * Y_HEIGHT; // 8,294,400 bytes (~8.3 MB)
  const CHROMA_WIDTH = 1920;
  const CHROMA_HEIGHT = 1080;
  const CHROMA_PLANE_BYTES = CHROMA_WIDTH * CHROMA_HEIGHT; // 2,073,600 bytes
  const THRESHOLD_BYTES = 256 * 1024; // 256 KiB

  const yPlaneData = new Uint8Array(Y_PLANE_BYTES);
  yPlaneData[0] = 0x10;
  yPlaneData[Y_PLANE_BYTES - 1] = 0xeb;
  const uPlaneData = new Uint8Array(CHROMA_PLANE_BYTES);
  const vPlaneData = new Uint8Array(CHROMA_PLANE_BYTES);
  // Exact-threshold UI atlas (256 KiB) must be preserved; threshold + 1 must be skipped.
  const exactThresholdData = new Uint8Array(THRESHOLD_BYTES);
  exactThresholdData.fill(0x7f);
  const overThresholdByOneData = new Uint8Array(THRESHOLD_BYTES + 1);

  const vsWgsl = '@vertex fn vs_main(@location(0) p: vec3<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(p, 1.0); }\n';
  const fsWgsl = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 1.0, 1.0, 1.0); }\n';

  const binkFrame = new BinaryPacketBuilder()
    .beginFrame()
    .clearColor(0.0, 0.0, 0.0, 1.0)
    .createShaderModule(1, SHADER_STAGE.VERTEX, vsWgsl)
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, fsWgsl)
    .createRenderPipeline(3, 1, 2, 12, PRIMITIVE_TOPOLOGY.TRIANGLE_LIST, 0)
    .writeTexture(100, 256, 256, exactThresholdData, { bytesPerRow: 1024, rowsPerImage: 256 })
    .writeTexture(101, Y_WIDTH, Y_HEIGHT, yPlaneData, { bytesPerRow: Y_WIDTH, rowsPerImage: Y_HEIGHT })
    .writeTexture(102, CHROMA_WIDTH, CHROMA_HEIGHT, uPlaneData, {
      bytesPerRow: CHROMA_WIDTH,
      rowsPerImage: CHROMA_HEIGHT,
    })
    .writeTexture(103, CHROMA_WIDTH, CHROMA_HEIGHT, vPlaneData, {
      bytesPerRow: CHROMA_WIDTH,
      rowsPerImage: CHROMA_HEIGHT,
    })
    .writeTexture(104, 256, 257, overThresholdByOneData, { bytesPerRow: 1024, rowsPerImage: 257 })
    .setPipeline(3)
    .draw(6, 1, 0, 0)
    .present()
    .finish();

  const expectedFilteredFrame = new BinaryPacketBuilder()
    .beginFrame()
    .clearColor(0.0, 0.0, 0.0, 1.0)
    .createShaderModule(1, SHADER_STAGE.VERTEX, vsWgsl)
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, fsWgsl)
    .createRenderPipeline(3, 1, 2, 12, PRIMITIVE_TOPOLOGY.TRIANGLE_LIST, 0)
    .writeTexture(100, 256, 256, exactThresholdData, { bytesPerRow: 1024, rowsPerImage: 256 })
    .setPipeline(3)
    .draw(6, 1, 0, 0)
    .present()
    .finish();

  const expectedSkippedBytes =
    (PACKET_HEADER_BYTES + WRITE_TEXTURE_HEADER_BYTES + Y_PLANE_BYTES) +
    (PACKET_HEADER_BYTES + WRITE_TEXTURE_HEADER_BYTES + CHROMA_PLANE_BYTES) * 2 +
    (PACKET_HEADER_BYTES + WRITE_TEXTURE_HEADER_BYTES + THRESHOLD_BYTES + 1);

  assert.equal(expectedFilteredFrame.byteLength + expectedSkippedBytes, binkFrame.byteLength);

  // 1) Single-chunk capture
  const rec = new GpuFrameRecorder({
    autoStart: true,
    maxRecordedBytes: 512 * 1024, // far smaller than a single 8.3 MiB Y plane!
    skipWriteTextureAboveBytes: THRESHOLD_BYTES,
  });

  const res = rec.recordChunk(binkFrame);
  assert.equal(res.completedFrames, 1);
  assert.equal(res.droppedFrames, 0);
  assert.equal(res.skippedLargeUploadPackets, 4);
  assert.equal(res.skippedLargeUploadBytes, expectedSkippedBytes);

  const stats = rec.getStats();
  assert.equal(stats.frameCount, 1);
  assert.equal(stats.recordedPackets, 9);
  assert.equal(stats.totalPackets, 9);
  assert.equal(stats.recordedBytes, expectedFilteredFrame.byteLength);
  assert.equal(stats.totalBytes, expectedFilteredFrame.byteLength);
  assert.ok(stats.recordedBytes < binkFrame.byteLength / 20);
  assert.equal(stats.skippedLargeUploadPackets, 4);
  assert.equal(stats.skippedLargeUploadBytes, expectedSkippedBytes);
  assert.equal(stats.droppedRingPackets, 0);
  assert.equal(stats.skipWriteTextureAboveBytes, THRESHOLD_BYTES);
  assert.equal(stats.maxRecordedBytes, 512 * 1024);
  assert.deepEqual(stats.opcodeCounts, {
    BeginFrame: 1,
    ClearColor: 1,
    CreateShaderModule: 2,
    CreateRenderPipeline: 1,
    WriteTexture: 1, // exact-threshold UI upload preserved
    SetPipeline: 1,
    Draw: 1,
    Present: 1,
  });

  const exported = rec.exportTraceBytes();
  assert.deepEqual(exported, expectedFilteredFrame);

  // Metadata records details of skipped large uploads and is JSON-safe & isolated.
  const meta = rec.exportMetadata();
  assert.deepEqual(JSON.parse(JSON.stringify(meta)), meta);
  assert.equal(meta.frames.length, 1);
  assert.equal(meta.frames[0].skippedLargeUploadPackets, 4);
  assert.equal(meta.frames[0].skippedLargeUploadBytes, expectedSkippedBytes);
  assert.equal(meta.skippedLargeUploads.length, 4);
  assert.deepEqual(meta.skippedLargeUploads[0], {
    frameSequence: 0,
    textureId: 101,
    mipLevel: 0,
    width: Y_WIDTH,
    height: Y_HEIGHT,
    depthOrArrayLayers: 1,
    bytesPerRow: Y_WIDTH,
    rowsPerImage: Y_HEIGHT,
    dataLen: Y_PLANE_BYTES,
    texelBytes: Y_PLANE_BYTES,
    packetBytes: PACKET_HEADER_BYTES + WRITE_TEXTURE_HEADER_BYTES + Y_PLANE_BYTES,
  });
  meta.skippedLargeUploads[0].textureId = 99999;
  assert.equal(rec.exportMetadata().skippedLargeUploads[0].textureId, 101);

  // 2) Streaming / arbitrary split chunks across packet header & WriteTexture payload boundaries
  const recSplit = new GpuFrameRecorder({
    autoStart: true,
    maxRecordedBytes: 512 * 1024,
    skipWriteTextureAboveBytes: THRESHOLD_BYTES,
  });
  // Deliberately split across tiny header fragments first, then 64 KiB chunks.
  const firstCutPoints = [3, 8, 12, 50, 90, 200, 1024];
  let pos = 0;
  for (const step of firstCutPoints) {
    if (pos >= binkFrame.byteLength) break;
    const end = Math.min(pos + step, binkFrame.byteLength);
    recSplit.recordChunk(binkFrame.subarray(pos, end));
    pos = end;
  }
  const chunkStep = 65536;
  while (pos < binkFrame.byteLength) {
    // Also split right around a WriteTexture packet header (odd stride).
    const step = (pos % 3 === 0) ? 19 : chunkStep;
    const end = Math.min(pos + step, binkFrame.byteLength);
    recSplit.recordChunk(binkFrame.subarray(pos, end));
    pos = end;
  }

  assert.deepEqual(recSplit.exportTraceBytes(), expectedFilteredFrame);
  assert.equal(recSplit.getStats().recordedPackets, 9);
  assert.equal(recSplit.getStats().recordedBytes, expectedFilteredFrame.byteLength);
  assert.equal(recSplit.getStats().skippedLargeUploadPackets, 4);
  assert.equal(recSplit.getStats().skippedLargeUploadBytes, expectedSkippedBytes);
  assert.deepEqual(recSplit.exportMetadata().skippedLargeUploads, rec.exportMetadata().skippedLargeUploads);

  // 3) Replay filtered trace through GpuTraceReplayer -> WgpuCommandProcessor
  const { device, ctx, calls } = createMockWebGpu();
  device.createShaderModule = (desc) => ({ kind: 'GPUShaderModule', code: desc.code });
  device.createRenderPipeline = (desc) => ({ kind: 'GPURenderPipeline', desc });
  const processor = new WgpuCommandProcessor(device, ctx);
  const replayer = new GpuTraceReplayer().loadTrace(exported);
  const replayRun = replayer.replayAll(processor);
  assert.equal(replayRun.ok, true);
  assert.equal(replayRun.succeededFrames, 1);
  assert.equal(replayRun.failedFrames, 0);
  assert.equal(calls.filter((c) => c.op === 'draw').length, 1);
  assert.equal(calls.filter((c) => c.op === 'submit').length, 1);

  // 4) Disassemble filtered trace with tools/gpu_trace.py --strict
  const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
  const probe = spawnSync(pythonCmd, ['--version'], { encoding: 'utf-8' });
  if (probe.error || probe.status !== 0) {
    t.skip('python interpreter unavailable for gpu_trace.py cross-check');
    return;
  }
  const traceScript = path.join(REPO_ROOT, 'tools', 'gpu_trace.py');
  const proc = spawnSync(pythonCmd, [traceScript, '-', '--json', '--strict'], {
    input: Buffer.from(exported),
    maxBuffer: 4 * 1024 * 1024,
  });
  assert.equal(proc.status, 0, `gpu_trace.py failed: ${proc.stderr.toString('utf-8')}`);
  const parsed = JSON.parse(proc.stdout.toString('utf-8'));
  assert.equal(parsed.ok, true);
  assert.deepEqual(parsed.diagnostics, []);
  assert.equal(parsed.frame_count, 1);
  assert.equal(parsed.packet_count, stats.recordedPackets);
  assert.equal(parsed.draw_calls, 1);
  assert.deepEqual(parsed.opcode_histogram, stats.opcodeCounts);
});

test('W216: maxRecordedBytes bounded ring-buffer eviction, droppedRingPackets, stopOnBudget, and clear()', () => {
  const frameA = buildSimpleFrame(0.1, 3); // 4 packets, 64 bytes
  const frameB = buildSimpleFrame(0.2, 6); // 4 packets, 64 bytes
  const frameC = buildSimpleFrame(0.3, 9); // 4 packets, 64 bytes
  const frameD = buildSimpleFrame(0.4, 12); // 4 packets, 64 bytes
  assert.equal(frameA.byteLength, 64);

  // 1) Rolling ring buffer bounded by maxRecordedBytes = 128 (holds 2 frames = 8 packets)
  const ringRec = new GpuFrameRecorder({
    maxFrames: 100,
    maxRecordedBytes: 128,
    autoStart: true,
  });

  ringRec.recordChunk(frameA);
  ringRec.recordChunk(frameB);
  let stats = ringRec.getStats();
  assert.equal(stats.frameCount, 2);
  assert.equal(stats.recordedPackets, 8);
  assert.equal(stats.recordedBytes, 128);
  assert.equal(stats.droppedFrames, 0);
  assert.equal(stats.droppedRingPackets, 0);

  // Pushing frameC and frameD evicts frameA and frameB (4 packets each -> 8 droppedRingPackets).
  ringRec.recordChunk(frameC);
  ringRec.recordChunk(frameD);
  stats = ringRec.getStats();
  assert.equal(stats.frameCount, 2);
  assert.equal(stats.recordedPackets, 8);
  assert.equal(stats.recordedBytes, 128);
  assert.ok(stats.recordedBytes <= stats.maxRecordedBytes);
  assert.equal(stats.droppedFrames, 2);
  assert.equal(stats.droppedRingPackets, 8);
  assert.deepEqual(ringRec.exportTraceBytes(), concatUint8Arrays([frameC, frameD]));

  // 2) Single oversized frame (> maxRecordedBytes) is dropped without evicting retained frames,
  //    and its packets are accounted in droppedRingPackets.
  const oversized = new BinaryPacketBuilder()
    .beginFrame()
    .writeBuffer(1, 0, new Uint8Array(200))
    .draw(3)
    .present()
    .finish(); // 4 packets, > 128 bytes
  ringRec.recordChunk(oversized);
  stats = ringRec.getStats();
  assert.equal(stats.frameCount, 2);
  assert.equal(stats.recordedPackets, 8);
  assert.equal(stats.recordedBytes, 128);
  assert.equal(stats.droppedFrames, 3);
  assert.equal(stats.droppedRingPackets, 12);

  // 3) Combined maxRecordedBytes + skipWriteTextureAboveBytes in startCapture (stopOnBudget: true):
  //    4K 8.3 MiB Y-plane frames fit inside a 200-byte budget instead of tripping budgetReached on frame 0!
  const yPlane = new Uint8Array(3840 * 2160);
  const binkLightFrame = new BinaryPacketBuilder()
    .beginFrame()
    .clearColor(0.2, 0.4, 0.6, 1.0)
    .writeTexture(77, 3840, 2160, yPlane, { bytesPerRow: 3840, rowsPerImage: 2160 })
    .draw(3, 1, 0, 0)
    .present()
    .finish();
  const expectedCompactFrame = buildSimpleFrame(0.2, 3);
  // Make clearColor match (0.2, 0.4, 0.6, 1.0)
  const expectedCompact = new BinaryPacketBuilder()
    .beginFrame()
    .clearColor(0.2, 0.4, 0.6, 1.0)
    .draw(3, 1, 0, 0)
    .present()
    .finish();
  assert.equal(expectedCompact.byteLength, expectedCompactFrame.byteLength); // 64 bytes

  const capRec = new GpuFrameRecorder();
  capRec.startCapture({
    maxFrames: 10,
    maxRecordedBytes: 150, // fits 2 compact frames (128B), stops cleanly on 3rd (192B > 150B)
    skipWriteTextureAboveBytes: 1024 * 1024,
  });

  capRec.recordChunk(binkLightFrame);
  capRec.recordChunk(binkLightFrame);
  assert.equal(capRec.isRecording(), true);
  assert.equal(capRec.getStats().frameCount, 2);
  assert.equal(capRec.getStats().recordedBytes, 128);
  assert.equal(capRec.getStats().recordedPackets, 8);
  assert.equal(capRec.getStats().skippedLargeUploadPackets, 2);
  assert.equal(
    capRec.getStats().skippedLargeUploadBytes,
    2 * (PACKET_HEADER_BYTES + WRITE_TEXTURE_HEADER_BYTES + yPlane.byteLength)
  );

  // 3rd frame trips 150B cap cleanly without corrupting the 2 captured frames.
  capRec.recordChunk(binkLightFrame);
  const capStats = capRec.getStats();
  assert.equal(capStats.budgetReached, true);
  assert.equal(capStats.recording, false);
  assert.equal(capStats.frameCount, 2);
  assert.equal(capStats.recordedBytes, 128);
  assert.equal(capStats.recordedPackets, 8);
  assert.deepEqual(capRec.exportTraceBytes(), concatUint8Arrays([expectedCompact, expectedCompact]));

  // 4) clear() resets all W216 counters and skippedLargeUploads metadata.
  ringRec.clear();
  const cleared = ringRec.exportMetadata();
  assert.equal(cleared.recordedPackets, 0);
  assert.equal(cleared.recordedBytes, 0);
  assert.equal(cleared.skippedLargeUploadPackets, 0);
  assert.equal(cleared.skippedLargeUploadBytes, 0);
  assert.equal(cleared.droppedRingPackets, 0);
  assert.deepEqual(cleared.skippedLargeUploads, []);
});
