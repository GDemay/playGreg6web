// G66 GPU -> CPU read-back in the WebGPU worker: ReadbackBuffer / ReadbackTexture packets decode,
// copy into a MAP_READ staging buffer in stream order, and after mapping write the bytes into the
// runtime's memory before the completion counter advances (also when the copy fails).
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const {
  WgpuCommandProcessor,
  OPCODES,
  decodeReadbackPayload,
  readbackStagingLayout,
  readbackPieces,
  deliverReadback,
} = require('../web/workers/wgpu_worker.js');

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
