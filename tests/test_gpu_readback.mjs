// G66 GPU -> CPU read-back in the WebGPU worker: ReadbackBuffer / ReadbackTexture packets decode,
// copy into a MAP_READ staging buffer in stream order, and after mapping write the bytes into the
// runtime's memory before the completion counter advances (also when the copy fails).
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  OPCODES,
  decodeReadbackPayload,
  readbackStagingLayout,
  readbackPieces,
  deliverReadback,
} = require('../web/workers/wgpu_worker.js');
const { GpuRingBuffer } = require('../web/workers/gpu_ring.js');

function bufferPayload({ seq, bufferId, srcOffset, size, dstAddr, doneAddr }) {
  const view = new DataView(new ArrayBuffer(40));
  view.setUint32(0, seq, true);
  view.setUint32(4, bufferId, true);
  view.setBigUint64(8, BigInt(srcOffset), true);
  view.setBigUint64(16, BigInt(size), true);
  view.setBigUint64(24, BigInt(dstAddr), true);
  view.setBigUint64(32, BigInt(doneAddr), true);
  return view;
}

function texturePayload(f) {
  const view = new DataView(new ArrayBuffer(72));
  const u32 = [f.seq, f.textureId, f.mipLevel, f.originX, f.originY, f.originZ, f.width, f.height, f.depth,
    f.bytesPerRow, f.rowsPerImage, 0];
  u32.forEach((v, i) => view.setUint32(4 * i, v, true));
  view.setBigUint64(48, BigInt(f.dstAddr), true);
  view.setBigUint64(56, BigInt(f.doneAddr), true);
  view.setBigUint64(64, BigInt(f.size), true);
  return view;
}

// A device whose copies really move bytes: GPU buffers and textures carry `contents`.
function mockDevice() {
  const pending = [];
  const device = {
    createBuffer(desc) {
      const b = {
        size: desc.size,
        usage: desc.usage,
        contents: new Uint8Array(desc.size),
        mapped: false,
        destroyed: false,
        mapAsync() {
          return new Promise((resolve) => pending.push(() => { b.mapped = true; resolve(); }));
        },
        getMappedRange() {
          return b.contents.slice().buffer;
        },
        unmap() { b.mapped = false; },
        destroy() { b.destroyed = true; },
      };
      return b;
    },
    queue: { submit() {} },
  };
  const encoder = {
    copyBufferToBuffer(src, srcOffset, dst, dstOffset, size) {
      dst.contents.set(src.contents.subarray(srcOffset, srcOffset + size), dstOffset);
    },
    copyTextureToBuffer(src, dst, size) {
      // Texture `contents` are tightly packed rows of `rowBytes`.
      const t = src.texture;
      for (let y = 0; y < size.height; y += 1) {
        const from = (src.origin.y + y) * t.rowBytes + src.origin.x * t.texelBytes;
        dst.buffer.contents.set(
          t.contents.subarray(from, from + size.width * t.texelBytes),
          y * dst.bytesPerRow
        );
      }
    },
  };
  return { device, encoder, resolveMaps: () => pending.splice(0).forEach((f) => f()) };
}

test('read-back payloads decode with the ABI layout', () => {
  const b = decodeReadbackPayload(OPCODES.READBACK_BUFFER,
    bufferPayload({ seq: 3, bufferId: 9, srcOffset: 6, size: 10, dstAddr: 0x1234, doneAddr: 0x40 }), 0, 40);
  assert.deepEqual(b, { kind: 'buffer', seq: 3, bufferId: 9, srcOffset: 6, size: 10, dstAddr: 0x1234, doneAddr: 0x40 });
  assert.equal(decodeReadbackPayload(OPCODES.READBACK_BUFFER, bufferPayload({ seq: 1, bufferId: 1, srcOffset: 0,
    size: 4, dstAddr: 0, doneAddr: 0 }), 0, 39), null);
  const t = decodeReadbackPayload(OPCODES.READBACK_TEXTURE, texturePayload({ seq: 4, textureId: 7, mipLevel: 1,
    originX: 2, originY: 3, originZ: 0, width: 16, height: 4, depth: 1, bytesPerRow: 256, rowsPerImage: 4,
    dstAddr: 0x800, doneAddr: 0x40, size: 832 }), 0, 72);
  assert.equal(t.kind, 'texture');
  assert.equal(t.textureId, 7);
  assert.equal(t.bytesPerRow, 256);
  assert.equal(t.size, 832);
});

test('staging layout aligns buffer copies to 4 bytes and texture rows to 256', () => {
  const b = readbackStagingLayout({ kind: 'buffer', srcOffset: 6, size: 9 });
  assert.deepEqual(b, { copyOffset: 4, skip: 2, stagingSize: 12 });
  const t = readbackStagingLayout({ kind: 'texture', width: 16, height: 4, depthOrLayers: 1, bytesPerRow: 512,
    rowsPerImage: 4, size: 512 * 3 + 64 });
  assert.equal(t.rowBytes, 64);
  assert.equal(t.stagingRow, 256);
  assert.equal(t.stagingSize, 1024);
  // Texture rows land at the guest pitch.
  const mapped = new Uint8Array(1024).map((_, i) => i & 0xff);
  const pieces = readbackPieces({ kind: 'texture', bytesPerRow: 512, rowsPerImage: 4, size: 512 * 3 + 64 }, t, mapped);
  assert.deepEqual(pieces.map((p) => p.offset), [0, 512, 1024, 1536]);
  assert.equal(pieces[1].bytes[0], 0);  // staging row 1 starts at byte 256 (& 0xff == 0)
});

test('deliverReadback writes the bytes before the completion counter, and counts failures too', () => {
  const memory = new SharedArrayBuffer(4096);
  const req = { kind: 'buffer', srcOffset: 0, size: 4, dstAddr: 100, doneAddr: 64 };
  assert.equal(deliverReadback(memory, req, [{ offset: 0, bytes: new Uint8Array([1, 2, 3, 4]) }]), true);
  assert.deepEqual([...new Uint8Array(memory, 100, 4)], [1, 2, 3, 4]);
  assert.equal(Atomics.load(new Uint32Array(memory, 64, 1), 0), 1);
  assert.equal(deliverReadback(memory, req, null), false);
  assert.equal(Atomics.load(new Uint32Array(memory, 64, 1), 0), 2);
});

test('processor records the copy in stream order and delivers after the map', async () => {
  const { device, encoder, resolveMaps } = mockDevice();
  const processor = new WgpuCommandProcessor(device, null, {});
  const memory = new SharedArrayBuffer(8192);
  processor.attachGuestMemory(memory);
  const src = device.createBuffer({ size: 64, usage: 0 });
  src.contents.set([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 6);
  processor.buffers.set(5, src);
  const tex = { format: 'rgba8unorm', rowBytes: 64, texelBytes: 4, contents: new Uint8Array(64 * 4) };
  tex.contents.fill(0x5c);
  processor.textures.set(8, tex);
  processor._handleReadback(OPCODES.READBACK_BUFFER,
    bufferPayload({ seq: 1, bufferId: 5, srcOffset: 6, size: 10, dstAddr: 200, doneAddr: 64 }), 0, 40);
  processor._handleReadback(OPCODES.READBACK_TEXTURE, texturePayload({ seq: 2, textureId: 8, mipLevel: 0, originX: 0,
    originY: 0, originZ: 0, width: 16, height: 4, depth: 1, bytesPerRow: 256, rowsPerImage: 4, dstAddr: 1024,
    doneAddr: 64, size: 256 * 3 + 64 }), 0, 72);
  // A packet for an unknown buffer still completes (without data).
  processor._handleReadback(OPCODES.READBACK_BUFFER,
    bufferPayload({ seq: 3, bufferId: 99, srcOffset: 0, size: 4, dstAddr: 300, doneAddr: 64 }), 0, 40);
  const cmds = processor._passCommands.filter((c) => c.type === 'readback');
  assert.equal(cmds.length, 3);
  let ended = 0;
  for (const c of cmds) c.run({ encoder, endPass: () => { ended += 1; } });
  assert.equal(ended, 3);
  const counter = new Uint32Array(memory, 64, 1);
  processor._startRecordedReadbacks(true);
  assert.equal(Atomics.load(counter, 0), 1);  // the unknown buffer failed at once
  assert.deepEqual([...new Uint8Array(memory, 200, 10)], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  resolveMaps();
  await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual([...new Uint8Array(memory, 200, 10)], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]);
  assert.equal(new Uint8Array(memory, 1024 + 256 * 3, 64).every((v) => v === 0x5c), true);
  assert.equal(new Uint8Array(memory, 1024 + 64, 192).every((v) => v === 0), true);  // pitch padding untouched
  assert.equal(Atomics.load(counter, 0), 3);
  assert.deepEqual(processor.readbackStats, { requested: 3, delivered: 2, failed: 1, bytes: 10 + 256 * 3 + 64 });
});

test('a title blocked on the guarded fence gets its read-back flushed without a Present', async () => {
  const { device } = mockDevice();
  const processor = new WgpuCommandProcessor(device, null, {});
  processor.attachGuestMemory(new SharedArrayBuffer(8192));
  const flushes = [];
  processor.ctx = {};
  processor._flushPresent = (opts) => {
    flushes.push({ opts, cmds: processor._passCommands.map((c) => c.type) });
  };
  processor._handleReadback(OPCODES.READBACK_BUFFER,
    bufferPayload({ seq: 1, bufferId: 5, srcOffset: 0, size: 4, dstAddr: 200, doneAddr: 64 }), 0, 40);
  processor._passCommands.push({ type: 'draw' });
  assert.equal(processor._readbackFlushWanted, true);
  // The end of a command buffer (no Present packet) triggers the flush.
  processor.executeCommandBuffer(new ArrayBuffer(0));
  assert.equal(flushes.length, 1);
  assert.deepEqual(flushes[0].opts, { readbackOnly: true });
  assert.deepEqual(flushes[0].cmds, ['readback']);          // only the work up to the read-back
  assert.deepEqual(processor._passCommands.map((c) => c.type), ['draw']);  // the rest waits for the frame
  assert.equal(processor._readbackFlushWanted, false);
  // Nothing recorded: no flush.
  processor._passCommands = [];
  assert.equal(processor._flushReadbacksNow(), false);
});

function encodePacket(opcode, payloadViewOrBytes = null) {
  const payload =
    payloadViewOrBytes === null
      ? new Uint8Array(0)
      : payloadViewOrBytes instanceof Uint8Array
        ? payloadViewOrBytes
        : new Uint8Array(
            payloadViewOrBytes.buffer,
            payloadViewOrBytes.byteOffset,
            payloadViewOrBytes.byteLength
          );
  const out = new Uint8Array(8 + payload.byteLength);
  const dv = new DataView(out.buffer);
  dv.setUint16(0, opcode, true);
  dv.setUint16(2, 0, true);
  dv.setUint32(4, payload.byteLength, true);
  out.set(payload, 8);
  return out;
}

function concatBytes(chunks) {
  let total = 0;
  for (const c of chunks) total += c.byteLength;
  const out = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.byteLength;
  }
  return out;
}

function setPipelinePayload(pipelineId) {
  const v = new DataView(new ArrayBuffer(4));
  v.setUint32(0, pipelineId, true);
  return v;
}

function drawPayload(vertexCount, instanceCount = 1, firstVertex = 0, firstInstance = 0) {
  const v = new DataView(new ArrayBuffer(16));
  v.setUint32(0, vertexCount, true);
  v.setUint32(4, instanceCount, true);
  v.setUint32(8, firstVertex, true);
  v.setUint32(12, firstInstance, true);
  return v;
}

function indirectPayload(bufferId, offset) {
  const v = new DataView(new ArrayBuffer(16));
  v.setUint32(0, bufferId, true);
  v.setUint32(4, 0, true);
  v.setBigUint64(8, BigInt(offset), true);
  return v;
}

function createFullMockWebGpu() {
  const pendingMaps = [];
  const recordedOps = [];
  let submitCount = 0;
  const device = {
    createBuffer(desc) {
      const b = {
        size: desc.size,
        usage: desc.usage,
        contents: new Uint8Array(desc.size),
        mapped: false,
        destroyed: false,
        mapAsync() {
          return new Promise((resolve) =>
            pendingMaps.push(() => {
              b.mapped = true;
              resolve();
            })
          );
        },
        getMappedRange() {
          return b.contents.slice().buffer;
        },
        unmap() {
          b.mapped = false;
        },
        destroy() {
          b.destroyed = true;
        },
      };
      return b;
    },
    createCommandEncoder() {
      const encoder = {
        beginRenderPass() {
          return {
            setPipeline(p) {
              recordedOps.push({ op: 'setPipeline', id: p.id });
            },
            setVertexBuffer(slot, buf, off) {
              recordedOps.push({ op: 'setVertexBuffer', slot, buf, off });
            },
            setIndexBuffer(buf, fmt, off) {
              recordedOps.push({ op: 'setIndexBuffer', buf, fmt, off });
            },
            setBindGroup(idx, bg) {
              recordedOps.push({ op: 'setBindGroup', idx, bg });
            },
            draw(vc, ic, fv, fi) {
              recordedOps.push({ op: 'draw', vc, ic, fv, fi });
            },
            drawIndexed(ic, inst, fi, bv, finst) {
              recordedOps.push({ op: 'drawIndexed', ic, inst, fi, bv, finst });
            },
            drawIndirect(buf, off) {
              recordedOps.push({ op: 'drawIndirect', bufId: buf.id, off });
            },
            drawIndexedIndirect(buf, off) {
              recordedOps.push({ op: 'drawIndexedIndirect', bufId: buf.id, off });
            },
            end() {},
          };
        },
        beginComputePass() {
          return {
            setPipeline(p) {
              recordedOps.push({ op: 'setComputePipeline', id: p.id });
            },
            setBindGroup(idx, bg) {
              recordedOps.push({ op: 'setComputeBindGroup', idx, bg });
            },
            dispatchWorkgroups(x, y, z) {
              recordedOps.push({ op: 'dispatch', x, y, z });
            },
            dispatchWorkgroupsIndirect(buf, off) {
              recordedOps.push({ op: 'dispatchIndirect', bufId: buf.id, off });
            },
            end() {},
          };
        },
        copyBufferToBuffer(src, srcOffset, dst, dstOffset, size) {
          recordedOps.push({ op: 'copyBufferToBuffer', srcOffset, size });
          dst.contents.set(src.contents.subarray(srcOffset, srcOffset + size), dstOffset);
        },
        copyTextureToBuffer(src, dst, size) {
          recordedOps.push({ op: 'copyTextureToBuffer', size });
        },
        finish() {
          return { finished: true };
        },
      };
      return encoder;
    },
    queue: {
      submit() {
        submitCount += 1;
        recordedOps.push({ op: 'submit', submitCount });
      },
    },
  };
  const ctx = {
    getCurrentTexture() {
      return {
        width: 1280,
        height: 720,
        createView() {
          return { view: true };
        },
      };
    },
  };
  return {
    device,
    ctx,
    recordedOps,
    getSubmitCount: () => submitCount,
    resolveMaps: () => pendingMaps.splice(0).forEach((fn) => fn()),
  };
}

test('W360: read-back delivered via SPSC GpuRingBuffer drainRing() flushes immediately without Present', async () => {
  const { device, ctx, recordedOps, getSubmitCount, resolveMaps } = createFullMockWebGpu();
  const processor = new WgpuCommandProcessor(device, ctx, {
    enableStagingPool: false,
    enableUniformRing: false,
    enablePassTimer: false,
    enableRtPool: false,
    enableResourceReplay: false,
  });
  const guestMemory = new SharedArrayBuffer(8192);
  processor.attachGuestMemory(guestMemory);

  const srcBuf = device.createBuffer({ size: 64, usage: 0 });
  srcBuf.contents.set([0xde, 0xad, 0xbe, 0xef, 0x11, 0x22, 0x33, 0x44], 8);
  processor.buffers.set(7, srcBuf);

  const ring = GpuRingBuffer.create(4096);
  processor.attachRing(ring);

  // Push ONLY a ReadbackBuffer packet into the SPSC ring (NO Present packet).
  const pkt = encodePacket(
    OPCODES.READBACK_BUFFER,
    bufferPayload({ seq: 1, bufferId: 7, srcOffset: 8, size: 8, dstAddr: 256, doneAddr: 64 })
  );
  assert.equal(ring.writeBytes(pkt), true);

  const res = processor.drainRing();
  assert.equal(res.bytesDrained, pkt.byteLength);
  assert.equal(res.presents, 0);
  assert.equal(processor.frameCount, 0);
  assert.equal(processor._earlyReadbackFlushes, 1);
  assert.equal(processor._readbackFlushWanted, false);
  assert.equal(processor._passCommands.length, 0);
  assert.equal(processor._passCommandPoolUsed, 0);
  assert.equal(getSubmitCount(), 1);
  assert.equal(recordedOps.some((e) => e.op === 'copyBufferToBuffer'), true);

  resolveMaps();
  await new Promise((r) => setTimeout(r, 0));

  assert.deepEqual(
    [...new Uint8Array(guestMemory, 256, 8)],
    [0xde, 0xad, 0xbe, 0xef, 0x11, 0x22, 0x33, 0x44]
  );
  assert.equal(Atomics.load(new Uint32Array(guestMemory, 64, 1), 0), 1);
  assert.deepEqual(processor.readbackStats, { requested: 1, delivered: 1, failed: 0, bytes: 8 });
});

test('W360: createWgpuWorkerHandler attaches pendingGuestMemory before drainRing() when attach-guest-memory arrives pre-init', async () => {
  const { device, ctx, resolveMaps } = createFullMockWebGpu();
  const posted = [];
  let createdProcessor = null;
  const handler = createWgpuWorkerHandler({
    postMessage: (m) => posted.push(m),
    GpuRingBuffer,
    initGpu: async () => {
      createdProcessor = new WgpuCommandProcessor(device, ctx, {
        enableStagingPool: false,
        enableUniformRing: false,
        enablePassTimer: false,
        enableRtPool: false,
        enableResourceReplay: false,
      });
      return createdProcessor;
    },
  });

  const guestMemory = new SharedArrayBuffer(4096);
  // attach-guest-memory arrives BEFORE init creates the processor.
  await handler.handleMessage({ type: 'attach-guest-memory', buffer: guestMemory });
  await handler.handleMessage({ init: true, canvas: { width: 1280, height: 720 }, width: 1280, height: 720 });
  assert.equal(createdProcessor._guestMemory, guestMemory);

  const srcBuf = device.createBuffer({ size: 32, usage: 0 });
  srcBuf.contents.set([42, 43, 44, 45], 0);
  createdProcessor.buffers.set(3, srcBuf);

  const ring = GpuRingBuffer.create(4096);
  await handler.handleMessage({ attachRing: true, ringSab: ring.sab });
  const pkt = encodePacket(
    OPCODES.READBACK_BUFFER,
    bufferPayload({ seq: 1, bufferId: 3, srcOffset: 0, size: 4, dstAddr: 128, doneAddr: 32 })
  );
  ring.writeBytes(pkt);

  await handler.handleMessage({ type: 'drain-ring' });
  resolveMaps();
  await new Promise((r) => setTimeout(r, 0));

  assert.deepEqual([...new Uint8Array(guestMemory, 128, 4)], [42, 43, 44, 45]);
  assert.equal(Atomics.load(new Uint32Array(guestMemory, 32, 1), 0), 1);
});

test('W360: _flushReadbacksNow() followed by pooled pass commands, indirect draws, and Present executes every command once with zero pool corruption', async () => {
  const { device, ctx, recordedOps, getSubmitCount, resolveMaps } = createFullMockWebGpu();
  const processor = new WgpuCommandProcessor(device, ctx, {
    enableStagingPool: false,
    enableUniformRing: false,
    enablePassTimer: false,
    enableRtPool: false,
    enableResourceReplay: false,
  });
  const guestMemory = new SharedArrayBuffer(8192);
  processor.attachGuestMemory(guestMemory);

  processor.pipelines.set(10, { id: 10 });
  processor.pipelines.set(20, { id: 20 });
  processor.pipelines.set(30, { id: 30 });
  processor.computePipelines.set(40, { id: 40 });

  const srcBuf = device.createBuffer({ size: 64, usage: 0 });
  srcBuf.contents.set([1, 2, 3, 4], 0);
  processor.buffers.set(5, srcBuf);

  const indirectDrawBuf = device.createBuffer({ size: 256, usage: 0 });
  indirectDrawBuf.id = 50;
  processor.buffers.set(50, indirectDrawBuf);

  const indirectDispatchBuf = device.createBuffer({ size: 256, usage: 0 });
  indirectDispatchBuf.id = 51;
  processor.buffers.set(51, indirectDispatchBuf);

  // Batch 1: pooled commands BEFORE readback + readback + pooled tail commands AFTER readback.
  const batch1 = concatBytes([
    encodePacket(OPCODES.SET_PIPELINE, setPipelinePayload(10)),
    encodePacket(OPCODES.DRAW, drawPayload(3, 1, 0, 0)),
    encodePacket(
      OPCODES.READBACK_BUFFER,
      bufferPayload({ seq: 1, bufferId: 5, srcOffset: 0, size: 4, dstAddr: 200, doneAddr: 64 })
    ),
    encodePacket(OPCODES.SET_PIPELINE, setPipelinePayload(20)),
    encodePacket(OPCODES.DRAW, drawPayload(6, 2, 1, 0)),
  ]);
  processor.executeCommandBuffer(batch1);

  assert.equal(getSubmitCount(), 1);
  assert.equal(processor.frameCount, 0);
  assert.equal(processor._earlyReadbackFlushes, 1);
  // Tail commands remain queued in `_passCommands` with intact pooled slot values.
  assert.equal(processor._passCommands.length, 2);
  assert.equal(processor._passCommands[0].type, 'setPipeline');
  assert.equal(processor._passCommands[0].pipelineId, 20);
  assert.equal(processor._passCommands[1].type, 'draw');
  assert.equal(processor._passCommands[1].vertexCount, 6);
  assert.equal(processor._passCommands[1].instanceCount, 2);
  // Pool cursor was NOT reset because tail still holds pooled slots.
  assert.equal(processor._passCommandPoolUsed, 4);

  resolveMaps();
  await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual([...new Uint8Array(guestMemory, 200, 4)], [1, 2, 3, 4]);
  assert.equal(Atomics.load(new Uint32Array(guestMemory, 64, 1), 0), 1);

  // Batch 2: more pooled commands (including indirect draws/dispatch) + Present.
  // Acquiring new pooled slots MUST NOT corrupt the preserved tail commands from Batch 1!
  const batch2 = concatBytes([
    encodePacket(OPCODES.SET_PIPELINE, setPipelinePayload(30)),
    encodePacket(OPCODES.DRAW, drawPayload(9, 1, 0, 0)),
    encodePacket(OPCODES.DRAW_INDIRECT, indirectPayload(50, 32)),
    encodePacket(OPCODES.DRAW_INDEXED_INDIRECT, indirectPayload(50, 64)),
    encodePacket(OPCODES.SET_COMPUTE_PIPELINE, setPipelinePayload(40)),
    encodePacket(OPCODES.DISPATCH_INDIRECT, indirectPayload(51, 96)),
    encodePacket(OPCODES.PRESENT),
  ]);
  const presentRes = processor.executeCommandBuffer(batch2);

  assert.equal(presentRes.presents, 1);
  assert.equal(processor.frameCount, 1);
  assert.equal(getSubmitCount(), 2);
  assert.equal(processor._passCommands.length, 0);
  assert.equal(processor._passCommandPoolUsed, 0);
  assert.equal(processor._validationErrorCount, 0);

  // Verify exact execution sequence across both submits: every command ran ONCE in order.
  assert.deepEqual(recordedOps, [
    // Submit 1 (early readback flush)
    { op: 'setPipeline', id: 10 },
    { op: 'draw', vc: 3, ic: 1, fv: 0, fi: 0 },
    { op: 'copyBufferToBuffer', srcOffset: 0, size: 4 },
    { op: 'submit', submitCount: 1 },
    // Submit 2 (frame Present: tail from batch 1 + batch 2)
    { op: 'setPipeline', id: 20 },
    { op: 'draw', vc: 6, ic: 2, fv: 1, fi: 0 },
    { op: 'setPipeline', id: 30 },
    { op: 'draw', vc: 9, ic: 1, fv: 0, fi: 0 },
    { op: 'drawIndirect', bufId: 50, off: 32 },
    { op: 'drawIndexedIndirect', bufId: 50, off: 64 },
    { op: 'setComputePipeline', id: 40 },
    { op: 'dispatchIndirect', bufId: 51, off: 96 },
    { op: 'submit', submitCount: 2 },
  ]);
});
