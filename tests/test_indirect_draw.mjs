import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const mod = require('../web/workers/indirect_draw.js');

const {
  U32_MAX,
  DEFAULT_MAX_INDIRECT_BATCH_COUNT,
  BUFFER_USAGE,
  INDIRECT_KIND,
  DRAW_INDIRECT_LAYOUT,
  DRAW_INDEXED_INDIRECT_LAYOUT,
  DISPATCH_INDIRECT_LAYOUT,
  IndirectDrawError,
  recordByteSize,
  wgslStructSource,
  wgslArrayStride,
  encodeDrawIndirect,
  encodeDrawIndexedIndirect,
  encodeDispatchIndirect,
  decodeDrawIndirect,
  decodeDrawIndexedIndirect,
  decodeDispatchIndirect,
  encodeBatch,
  isValidIndirectRange,
  checkIndirectRange,
  isZeroCountRecord,
  validateIndirectBufferArgs,
  validateIndirectDrawArgs,
  validateIndirectRange,
  computeIndirectBatchStepPlan,
  validateIndirectBatch,
  validateDispatchCounts,
  planDispatches,
  encodeDispatches,
  encodeDispatchForItems,
  encodeIndirectDraw,
  encodeIndirectDrawIndexed,
  encodeIndirectDispatch,
  encodeMultiDraw,
  encodeMultiDispatchIndirect,
  executeIndirect,
  executeIndirectDraw,
  executeIndirectDrawIndexed,
  executeIndirectDispatch,
  dispatchIndirectDraw,
  dispatchIndexedIndirectDraw,
  dispatchIndirectBatch,
  getStats,
  resetStats,
  IndirectArgsBuffer,
} = mod;

function createMockDevice() {
  const buffers = [];
  const writes = [];
  const submits = [];
  const copies = [];
  const queue = {
    writeBuffer(buffer, offset, data) {
      const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
      buffer._data.set(bytes, offset);
      writes.push({ buffer, offset, length: bytes.byteLength });
    },
    submit(cmds) {
      submits.push(cmds);
      for (const cmd of cmds) {
        for (const c of cmd.copies) {
          c.dst._data.set(c.src._data.subarray(c.srcOffset, c.srcOffset + c.size), c.dstOffset);
        }
      }
    },
  };
  const device = {
    queue,
    createBuffer(desc) {
      const buffer = {
        ...desc,
        _data: new Uint8Array(desc.size),
        destroyed: false,
        destroy() {
          this.destroyed = true;
        },
      };
      buffers.push(buffer);
      return buffer;
    },
    createCommandEncoder() {
      const mine = [];
      return {
        copyBufferToBuffer(src, srcOffset, dst, dstOffset, size) {
          const c = { src, srcOffset, dst, dstOffset, size };
          mine.push(c);
          copies.push(c);
        },
        finish() {
          return { copies: mine };
        },
      };
    },
  };
  return { device, buffers, writes, submits, copies };
}

function createMockPass(withNative = false) {
  const calls = [];
  const pass = {
    calls,
    drawIndirect: (b, o) => calls.push(['drawIndirect', o]),
    drawIndexedIndirect: (b, o) => calls.push(['drawIndexedIndirect', o]),
    dispatchWorkgroups: (x, y, z) => calls.push(['dispatchWorkgroups', x, y, z]),
    dispatchWorkgroupsIndirect: (b, o) => calls.push(['dispatchWorkgroupsIndirect', o]),
  };
  if (withNative) {
    pass.multiDrawIndirect = (b, o, n) => calls.push(['multiDrawIndirect', o, n]);
    pass.multiDrawIndexedIndirect = (b, o, n) => calls.push(['multiDrawIndexedIndirect', o, n]);
  }
  return pass;
}

const rec = (n) => ({ indexCount: n, instanceCount: 1, firstIndex: 0, baseVertex: 0, firstInstance: 0 });

test('layout sizes are 16 / 20 / 12 bytes', () => {
  assert.equal(DRAW_INDIRECT_LAYOUT.byteSize, 16);
  assert.equal(DRAW_INDEXED_INDIRECT_LAYOUT.byteSize, 20);
  assert.equal(DISPATCH_INDIRECT_LAYOUT.byteSize, 12);
  assert.equal(recordByteSize(INDIRECT_KIND.DRAW_INDEXED), 20);
  assert.throws(() => recordByteSize('nope'), TypeError);
});

test('drawIndirect encodes byte-exact little-endian', () => {
  const bytes = encodeDrawIndirect({ vertexCount: 3, instanceCount: 0x01020304, firstVertex: 7, firstInstance: 0 });
  assert.deepEqual(
    [...bytes],
    [3, 0, 0, 0, 4, 3, 2, 1, 7, 0, 0, 0, 0, 0, 0, 0]
  );
});

test('drawIndexedIndirect encodes negative baseVertex and max u32 byte-exact', () => {
  const bytes = encodeDrawIndexedIndirect({
    indexCount: U32_MAX,
    instanceCount: 1,
    firstIndex: 2,
    baseVertex: -1,
    firstInstance: U32_MAX,
  });
  assert.equal(bytes.byteLength, 20);
  assert.deepEqual(
    [...bytes],
    [255, 255, 255, 255, 1, 0, 0, 0, 2, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255]
  );
  const min = encodeDrawIndexedIndirect({ indexCount: 0, instanceCount: 0, baseVertex: -0x80000000 });
  assert.deepEqual([...min.subarray(12, 16)], [0, 0, 0, 0x80]);
});

test('dispatchWorkgroupsIndirect encodes byte-exact', () => {
  assert.deepEqual([...encodeDispatchIndirect({ x: 1, y: 2, z: 0x10000 })], [1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0]);
});

test('round-trip decode for all three layouts', () => {
  const d = { vertexCount: 36, instanceCount: 5, firstVertex: 9, firstInstance: 11 };
  assert.deepEqual(decodeDrawIndirect(encodeDrawIndirect(d)), d);
  const i = { indexCount: U32_MAX, instanceCount: 2, firstIndex: 100, baseVertex: -12345, firstInstance: 3 };
  assert.deepEqual(decodeDrawIndexedIndirect(encodeDrawIndexedIndirect(i)), i);
  const c = { x: U32_MAX, y: 1, z: 7 };
  assert.deepEqual(decodeDispatchIndirect(encodeDispatchIndirect(c)), c);
});

test('encode/decode work on ArrayBuffer, DataView and offset typed arrays', () => {
  const ab = new ArrayBuffer(48);
  mod.encodeRecord(INDIRECT_KIND.DRAW_INDEXED, rec(9), new DataView(ab), 24);
  assert.equal(decodeDrawIndexedIndirect(ab, 24).indexCount, 9);
  const u32 = new Uint32Array(ab, 4, 11);
  assert.equal(decodeDrawIndexedIndirect(u32, 20).indexCount, 9);
});

test('validation: non-integer field rejected', () => {
  assert.throws(() => encodeDrawIndirect({ vertexCount: 1.5, instanceCount: 1 }), TypeError);
});
test('validation: missing required count rejected', () => {
  assert.throws(() => encodeDrawIndirect({ vertexCount: 1 }), TypeError);
});
test('validation: u32 overflow rejected', () => {
  assert.throws(() => encodeDrawIndirect({ vertexCount: U32_MAX + 1, instanceCount: 1 }), RangeError);
});
test('validation: negative u32 rejected', () => {
  assert.throws(() => encodeDrawIndexedIndirect({ indexCount: -1, instanceCount: 1 }), RangeError);
});
test('validation: baseVertex outside i32 rejected', () => {
  assert.throws(() => encodeDrawIndexedIndirect({ indexCount: 1, instanceCount: 1, baseVertex: 0x80000000 }), RangeError);
  assert.throws(() => encodeDrawIndexedIndirect({ indexCount: 1, instanceCount: 1, baseVertex: -0x80000001 }), RangeError);
});
test('validation: undersized target rejected', () => {
  assert.throws(() => mod.encodeRecord(INDIRECT_KIND.DRAW, { vertexCount: 1, instanceCount: 1 }, new Uint8Array(15)), RangeError);
});

test('encodeBatch writes records at a stride', () => {
  const { bytes, stride, patched } = encodeBatch(INDIRECT_KIND.DRAW_INDEXED, [rec(1), rec(2), rec(3)], { stride: 32 });
  assert.equal(stride, 32);
  assert.deepEqual(patched, []);
  assert.equal(bytes.byteLength, 32 * 2 + 20);
  assert.equal(decodeDrawIndexedIndirect(bytes, 64).indexCount, 3);
  assert.equal(decodeDrawIndexedIndirect(bytes, 32).indexCount, 2);
});

test('encodeBatch rejects bad strides', () => {
  assert.throws(() => encodeBatch(INDIRECT_KIND.DRAW_INDEXED, [rec(1)], { stride: 16 }), RangeError);
  assert.throws(() => encodeBatch(INDIRECT_KIND.DRAW, [{ vertexCount: 1, instanceCount: 1 }], { stride: 18 }), RangeError);
});

test('firstInstance without feature throws FIRST_INSTANCE_UNSUPPORTED', () => {
  assert.throws(
    () => encodeBatch(INDIRECT_KIND.DRAW_INDEXED, [rec(1), { ...rec(2), firstInstance: 4 }]),
    (e) => e instanceof IndirectDrawError && e.code === 'FIRST_INSTANCE_UNSUPPORTED' && /record 1/.test(e.message)
  );
});

test('firstInstance allowed with feature flag', () => {
  const { bytes, patched } = encodeBatch(INDIRECT_KIND.DRAW, [{ vertexCount: 3, instanceCount: 1, firstInstance: 4 }], {
    indirectFirstInstance: true,
  });
  assert.equal(decodeDrawIndirect(bytes).firstInstance, 4);
  assert.deepEqual(patched, []);
});

test('firstInstance patch fallback zeroes and reports indices', () => {
  const { bytes, patched } = encodeBatch(
    INDIRECT_KIND.DRAW_INDEXED,
    [{ ...rec(1), firstInstance: 9 }, rec(2)],
    { patchFirstInstance: true }
  );
  assert.deepEqual(patched, [0]);
  assert.equal(decodeDrawIndexedIndirect(bytes, 0).firstInstance, 0);
});

test('encodeBatch can write into a caller buffer at an offset', () => {
  const out = new Uint8Array(64);
  encodeBatch(INDIRECT_KIND.DRAW_INDEXED, [rec(5)], { out, outOffset: 40 });
  assert.equal(decodeDrawIndexedIndirect(out, 40).indexCount, 5);
});

test('validateIndirectRange: missing INDIRECT usage', () => {
  assert.throws(
    () => validateIndirectRange({ usage: BUFFER_USAGE.STORAGE, size: 64 }, 0, 16),
    (e) => e.code === 'NOT_INDIRECT'
  );
});
test('validateIndirectRange: offset not multiple of 4', () => {
  assert.throws(
    () => validateIndirectRange({ usage: BUFFER_USAGE.INDIRECT, size: 64 }, 2, 16),
    (e) => e.code === 'MISALIGNED_OFFSET'
  );
});
test('validateIndirectRange: offset + record beyond buffer', () => {
  assert.throws(
    () => validateIndirectRange({ usage: BUFFER_USAGE.INDIRECT, size: 64 }, 48, 20),
    (e) => e.code === 'OUT_OF_BOUNDS'
  );
});
test('validateIndirectRange: exact fit accepted', () => {
  validateIndirectRange({ usage: BUFFER_USAGE.INDIRECT, size: 64 }, 44, 20);
});

test('validateDispatchCounts: limit enforced per axis', () => {
  assert.deepEqual(validateDispatchCounts({ x: 65535 }), { x: 65535, y: 1, z: 1 });
  for (const axis of ['x', 'y', 'z']) {
    assert.throws(
      () => validateDispatchCounts({ x: 1, y: 1, z: 1, [axis]: 65536 }),
      (e) => e.code === 'DISPATCH_LIMIT' && e.message.includes(`${axis}=65536`)
    );
  }
  assert.throws(() => validateDispatchCounts({ x: 11 }, 10), (e) => e.code === 'DISPATCH_LIMIT');
  assert.throws(() => validateDispatchCounts({ x: -1 }), RangeError);
});

test('planDispatches: fits one dispatch at exactly the limit', () => {
  const plan = planDispatches(64 * 10, 64, 10);
  assert.deepEqual(plan, [{ x: 10, y: 1, z: 1, firstWorkgroup: 0, firstItem: 0 }]);
});

test('planDispatches: one over the limit splits in two', () => {
  const plan = planDispatches(64 * 10 + 1, 64, 10);
  assert.equal(plan.length, 2);
  assert.deepEqual(plan[0], { x: 10, y: 1, z: 1, firstWorkgroup: 0, firstItem: 0 });
  assert.deepEqual(plan[1], { x: 1, y: 1, z: 1, firstWorkgroup: 10, firstItem: 640 });
});

test('planDispatches: rounding and totals at default limit', () => {
  assert.deepEqual(planDispatches(0, 64), []);
  assert.deepEqual(planDispatches(1, 64).map((d) => d.x), [1]);
  const items = 65535 * 64 * 2 + 5;
  const plan = planDispatches(items, 64);
  assert.deepEqual(plan.map((d) => d.x), [65535, 65535, 1]);
  assert.equal(plan.reduce((s, d) => s + d.x, 0), Math.ceil(items / 64));
  assert.throws(() => planDispatches(1, 0), RangeError);
});

test('encodeDispatches calls dispatchWorkgroups with hook', () => {
  const pass = createMockPass();
  const seen = [];
  const n = encodeDispatches(pass, planDispatches(25, 1, 10), (d, i) => seen.push([i, d.firstItem]));
  assert.equal(n, 3);
  assert.deepEqual(pass.calls, [
    ['dispatchWorkgroups', 10, 1, 1],
    ['dispatchWorkgroups', 10, 1, 1],
    ['dispatchWorkgroups', 5, 1, 1],
  ]);
  assert.deepEqual(seen, [[0, 0], [1, 10], [2, 20]]);
});

test('encodeDispatchForItems encodes an indirect record and enforces the limit', () => {
  assert.deepEqual(decodeDispatchIndirect(encodeDispatchForItems(1000, 64)), { x: 16, y: 1, z: 1 });
  assert.throws(() => encodeDispatchForItems(64 * 11, 64, 10), (e) => e.code === 'DISPATCH_LIMIT');
});

test('WGSL struct layout offsets and strides', () => {
  assert.equal(wgslArrayStride(INDIRECT_KIND.DRAW), 16);
  assert.equal(wgslArrayStride(INDIRECT_KIND.DRAW_INDEXED), 20);
  assert.equal(wgslArrayStride(INDIRECT_KIND.DISPATCH), 12);
  // Member offsets follow declaration order in the WGSL struct source.
  for (const [kind, layout] of [
    [INDIRECT_KIND.DRAW, DRAW_INDIRECT_LAYOUT],
    [INDIRECT_KIND.DRAW_INDEXED, DRAW_INDEXED_INDIRECT_LAYOUT],
    [INDIRECT_KIND.DISPATCH, DISPATCH_INDIRECT_LAYOUT],
  ]) {
    const members = [...wgslStructSource(kind).matchAll(/^\s+(\w+): (u32|i32),$/gm)];
    assert.equal(members.length * 4, layout.byteSize);
    const camel = (s) => s.replace(/_(\w)/g, (_, c) => c.toUpperCase());
    members.forEach((m, idx) => {
      const field = layout.fields[camel(m[1])];
      assert.ok(field, `${kind}.${m[1]} maps to a field`);
      assert.equal(field.offset, idx * 4);
      assert.equal(field.signed, m[2] === 'i32');
    });
  }
  assert.match(wgslStructSource(INDIRECT_KIND.DRAW_INDEXED), /base_vertex: i32/);
  assert.equal(DRAW_INDEXED_INDIRECT_LAYOUT.fields.instanceCount.offset, 4);
  // args[7].instance_count byte offset for the 20-byte indexed record
  assert.equal(7 * wgslArrayStride(INDIRECT_KIND.DRAW_INDEXED) + DRAW_INDEXED_INDIRECT_LAYOUT.fields.instanceCount.offset, 144);
});

test('IndirectArgsBuffer allocates GPU buffer with INDIRECT|STORAGE|COPY_DST|COPY_SRC', () => {
  const { device, buffers } = createMockDevice();
  const args = new IndirectArgsBuffer({ device, capacity: 4 });
  assert.equal(buffers.length, 1);
  assert.equal(buffers[0].size, 80);
  assert.equal(
    buffers[0].usage,
    BUFFER_USAGE.INDIRECT | BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST | BUFFER_USAGE.COPY_SRC
  );
  const noSrc = new IndirectArgsBuffer({ device, capacity: 4, copySrc: false });
  assert.equal(noSrc.buffer.usage & BUFFER_USAGE.COPY_SRC, 0);
  assert.throws(() => noSrc.grow(8), (e) => e.code === 'GROW_UNSUPPORTED');
  assert.equal(args.getStorageBinding().size, 80);
});

test('IndirectArgsBuffer slot offsets are stable and freed slots are reused', () => {
  const { device } = createMockDevice();
  const args = new IndirectArgsBuffer({ device, capacity: 4 });
  const a = args.allocate(rec(1));
  const b = args.allocate(rec(2));
  const c = args.allocate(rec(3));
  assert.deepEqual([a, b, c], [0, 1, 2]);
  assert.equal(args.slotOffset(c), 40);
  args.free(b);
  assert.equal(args.liveCount, 2);
  assert.equal(args.allocate(rec(9)), b);
  assert.equal(args.getArgs(b).indexCount, 9);
  assert.throws(() => args.free(3), (e) => e.code === 'BAD_SLOT');
  args.free(a);
  assert.throws(() => args.free(a), (e) => e.code === 'BAD_SLOT');
});

test('IndirectArgsBuffer reused slot without args is zeroed', () => {
  const { device } = createMockDevice();
  const args = new IndirectArgsBuffer({ device, capacity: 2 });
  const s = args.allocate(rec(5));
  args.free(s);
  assert.equal(args.allocate(), s);
  assert.equal(args.getArgs(s).indexCount, 0);
});

test('dirty-range coalescing: adjacent slots merge into one write', () => {
  const { device, writes } = createMockDevice();
  const args = new IndirectArgsBuffer({ device, capacity: 8 });
  for (let i = 0; i < 3; i += 1) args.allocate(rec(i + 1));
  assert.equal(args.flush(), 1);
  assert.deepEqual(writes.map((w) => [w.offset, w.length]), [[0, 60]]);
  assert.equal(args.flush(), 0);
});

test('dirty-range coalescing: gaps stay separate unless mergeGap covers them', () => {
  const { device, writes } = createMockDevice();
  const args = new IndirectArgsBuffer({ device, capacity: 8 });
  for (let i = 0; i < 6; i += 1) args.allocate(rec(i));
  args.flush();
  writes.length = 0;
  args.setArgs(0, rec(10));
  args.setArgs(2, rec(12));
  args.setArgs(5, rec(15));
  args.setArgs(2, rec(13));
  assert.equal(args.flush(), 3);
  assert.deepEqual(writes.map((w) => [w.offset, w.length]), [[0, 20], [40, 20], [100, 20]]);

  const gap = new IndirectArgsBuffer({ device, capacity: 8, mergeGap: 20 });
  for (let i = 0; i < 6; i += 1) gap.allocate(rec(i));
  gap.flush();
  writes.length = 0;
  gap.setArgs(0, rec(10));
  gap.setArgs(2, rec(12));
  assert.equal(gap.flush(), 1);
  assert.deepEqual(writes.map((w) => [w.offset, w.length]), [[0, 60]]);
});

test('flush uploads exact bytes into the GPU buffer', () => {
  const { device } = createMockDevice();
  const args = new IndirectArgsBuffer({ device, capacity: 2 });
  const s = args.allocate({ ...rec(7), baseVertex: -3 });
  args.flush();
  assert.deepEqual(decodeDrawIndexedIndirect(args.buffer._data, args.slotOffset(s)), { ...rec(7), baseVertex: -3 });
});

test('growth preserves data (including GPU-side edits), notifies, and keeps slot indices', () => {
  const { device, buffers, submits, copies } = createMockDevice();
  const events = [];
  const args = new IndirectArgsBuffer({ device, capacity: 2, onGrow: (e) => events.push(e) });
  args.allocate(rec(1));
  args.allocate(rec(2));
  args.flush();
  // Simulate the culling shader zeroing instanceCount of slot 1 on the GPU.
  args.buffer._data.set([0, 0, 0, 0], 20 + 4);
  const old = args.buffer;
  const third = args.allocate(rec(3));
  assert.equal(third, 2);
  assert.equal(args.capacity, 4);
  assert.equal(buffers.length, 2);
  assert.equal(old.destroyed, true);
  assert.equal(submits.length, 1);
  assert.equal(copies.length, 1);
  assert.equal(events.length, 1);
  assert.equal(events[0].oldBuffer, old);
  assert.equal(events[0].buffer, args.buffer);
  assert.equal(events[0].capacity, 4);
  assert.equal(events[0].oldCapacity, 2);
  args.flush();
  assert.equal(decodeDrawIndexedIndirect(args.buffer._data, 0).indexCount, 1);
  assert.equal(decodeDrawIndexedIndirect(args.buffer._data, 20).indexCount, 2);
  assert.equal(decodeDrawIndexedIndirect(args.buffer._data, 20).instanceCount, 0, "GPU-side cull survives growth");
  assert.equal(decodeDrawIndexedIndirect(args.buffer._data, 40).indexCount, 3);
  assert.equal(args.getStorageBinding().buffer, args.buffer);
});

test('growth copies GPU-side edits for clean slots', () => {
  const { device } = createMockDevice();
  const args = new IndirectArgsBuffer({ device, capacity: 2 });
  args.allocate(rec(1));
  args.allocate(rec(2));
  args.flush();
  args.buffer._data.set([0, 0, 0, 0], 4);
  args.grow(3);
  assert.equal(decodeDrawIndexedIndirect(args.buffer._data, 0).instanceCount, 0);
  assert.equal(args.grow(2), false);
});

test('autoGrow false throws FULL', () => {
  const { device } = createMockDevice();
  const args = new IndirectArgsBuffer({ device, capacity: 1, autoGrow: false });
  args.allocate();
  assert.throws(() => args.allocate(), (e) => e.code === 'FULL');
});

test('IndirectArgsBuffer applies firstInstance policy', () => {
  const { device } = createMockDevice();
  const strict = new IndirectArgsBuffer({ device, capacity: 2 });
  const s = strict.allocate();
  assert.throws(() => strict.setArgs(s, { ...rec(1), firstInstance: 2 }), (e) => e.code === 'FIRST_INSTANCE_UNSUPPORTED');
  const patching = new IndirectArgsBuffer({ device, capacity: 2, patchFirstInstance: true });
  const p = patching.allocate({ ...rec(1), firstInstance: 2 });
  assert.equal(patching.getArgs(p).firstInstance, 0);
  const native = new IndirectArgsBuffer({ device, capacity: 2, indirectFirstInstance: true });
  assert.equal(native.getArgs(native.allocate({ ...rec(1), firstInstance: 2 })).firstInstance, 2);
});

test('storage binding exposes buffer, size, stride and instanceCount offset', () => {
  const { device } = createMockDevice();
  const idx = new IndirectArgsBuffer({ device, capacity: 3 }).getStorageBinding();
  assert.equal(idx.offset, 0);
  assert.equal(idx.size, 60);
  assert.equal(idx.stride, 20);
  assert.equal(idx.instanceCountOffset, 4);
  assert.deepEqual(idx.bindGroupLayoutEntry, { buffer: { type: 'storage' } });
  assert.match(idx.wgsl, /struct DrawIndexedIndirectArgs/);
  const plain = new IndirectArgsBuffer({ device, kind: INDIRECT_KIND.DRAW, capacity: 3 }).getStorageBinding();
  assert.equal(plain.size, 48);
  assert.equal(plain.stride, 16);
  assert.throws(() => new IndirectArgsBuffer({ device, kind: INDIRECT_KIND.DISPATCH }), TypeError);
});

test('destroy releases the buffer and blocks further use', () => {
  const { device } = createMockDevice();
  const args = new IndirectArgsBuffer({ device, capacity: 2 });
  args.destroy();
  args.destroy();
  assert.equal(args.buffer.destroyed, true);
  assert.throws(() => args.allocate(), (e) => e.code === 'DESTROYED');
  assert.throws(() => args.flush(), (e) => e.code === 'DESTROYED');
});

const indirectBuf = (size) => ({ usage: BUFFER_USAGE.INDIRECT, size });

test('multi-draw emulation issues one drawIndirect per slot', () => {
  const pass = createMockPass();
  const r = encodeMultiDraw(pass, indirectBuf(80), { count: 3, firstSlot: 1 });
  assert.deepEqual(pass.calls, [
    ['drawIndirect', 16],
    ['drawIndirect', 32],
    ['drawIndirect', 48],
  ]);
  assert.equal(r.calls, 3);
});

test('multi-draw emulation: exact call sequence without mask (indexed)', () => {
  const pass = createMockPass();
  const r = encodeMultiDraw(pass, indirectBuf(100), { indexed: true, count: 3, firstSlot: 1, baseOffset: 0 });
  assert.deepEqual(pass.calls, [
    ['drawIndexedIndirect', 20],
    ['drawIndexedIndirect', 40],
    ['drawIndexedIndirect', 60],
  ]);
  assert.deepEqual(r, { calls: 3, draws: 3, skipped: 0, native: false });
});

test('multi-draw emulation: visibility mask skips draws', () => {
  const pass = createMockPass();
  const r = encodeMultiDraw(pass, indirectBuf(100), {
    indexed: true,
    count: 5,
    visibility: [true, false, 1, 0, true],
  });
  assert.deepEqual(pass.calls, [
    ['drawIndexedIndirect', 0],
    ['drawIndexedIndirect', 40],
    ['drawIndexedIndirect', 80],
  ]);
  assert.deepEqual(r, { calls: 3, draws: 3, skipped: 2, native: false });
});

test('multi-draw validates every offset against the buffer', () => {
  const pass = createMockPass();
  assert.throws(() => encodeMultiDraw(pass, indirectBuf(30), { indexed: true, count: 2 }), (e) => e.code === 'OUT_OF_BOUNDS');
  assert.throws(() => encodeMultiDraw(pass, { usage: 0, size: 100 }, { count: 1 }), (e) => e.code === 'NOT_INDIRECT');
  assert.throws(() => encodeMultiDraw(pass, indirectBuf(100), { count: 1, baseOffset: 2 }), (e) => e.code === 'MISALIGNED_OFFSET');
  assert.throws(() => encodeMultiDraw(pass, indirectBuf(100), { count: 1, stride: 8 }), RangeError);
});

test('native multi-draw is used for contiguous visible runs only when requested', () => {
  const pass = createMockPass(true);
  const off = encodeMultiDraw(pass, indirectBuf(200), { indexed: true, count: 2 });
  assert.equal(off.native, false);
  assert.deepEqual(pass.calls.map((c) => c[0]), ['drawIndexedIndirect', 'drawIndexedIndirect']);

  pass.calls.length = 0;
  const r = encodeMultiDraw(pass, indirectBuf(200), {
    indexed: true,
    useNative: true,
    count: 6,
    visibility: [1, 1, 0, 1, 1, 1],
  });
  assert.deepEqual(pass.calls, [
    ['multiDrawIndexedIndirect', 0, 2],
    ['multiDrawIndexedIndirect', 60, 3],
  ]);
  assert.deepEqual(r, { calls: 2, draws: 5, skipped: 1, native: true });

  pass.calls.length = 0;
  encodeMultiDraw(pass, indirectBuf(200), { useNative: true, count: 4 });
  assert.deepEqual(pass.calls, [['multiDrawIndirect', 0, 4]]);
});

test('useNative falls back to emulation when the pass lacks the method', () => {
  const pass = createMockPass(false);
  const r = encodeMultiDraw(pass, indirectBuf(100), { useNative: true, count: 2 });
  assert.equal(r.native, false);
  assert.deepEqual(pass.calls, [['drawIndirect', 0], ['drawIndirect', 16]]);
});

test('IndirectArgsBuffer.encodeDraws covers handed-out slots by default', () => {
  const { device } = createMockDevice();
  const args = new IndirectArgsBuffer({ device, capacity: 4 });
  args.allocate(rec(1));
  args.allocate(rec(2));
  const pass = createMockPass();
  const r = args.encodeDraws(pass, { visibility: [true, false] });
  assert.deepEqual(pass.calls, [['drawIndexedIndirect', 0]]);
  assert.equal(r.skipped, 1);
});

test('bounds & alignment validation rejects unaligned and out-of-bounds offsets and increments boundsRejectedCount', () => {
  resetStats();
  const pass = createMockPass();

  // drawIndirect requires 16 bytes, 4-byte aligned
  assert.throws(
    () => encodeIndirectDraw(pass, indirectBuf(32), 2),
    (e) => e instanceof IndirectDrawError && e.code === 'MISALIGNED_OFFSET'
  );
  assert.throws(
    () => encodeIndirectDraw(pass, indirectBuf(32), 20),
    (e) => e instanceof IndirectDrawError && e.code === 'OUT_OF_BOUNDS'
  );

  // drawIndexedIndirect requires 20 bytes
  assert.throws(
    () => encodeIndirectDrawIndexed(pass, indirectBuf(32), 16),
    (e) => e instanceof IndirectDrawError && e.code === 'OUT_OF_BOUNDS'
  );
  assert.throws(
    () => encodeIndirectDrawIndexed(pass, indirectBuf(64), 6),
    (e) => e instanceof IndirectDrawError && e.code === 'MISALIGNED_OFFSET'
  );

  // dispatchWorkgroupsIndirect requires 12 bytes
  assert.throws(
    () => encodeIndirectDispatch(pass, indirectBuf(20), 12),
    (e) => e instanceof IndirectDrawError && e.code === 'OUT_OF_BOUNDS'
  );
  assert.throws(
    () => encodeIndirectDispatch(pass, indirectBuf(20), 1),
    (e) => e instanceof IndirectDrawError && e.code === 'MISALIGNED_OFFSET'
  );

  // Explicit bufferSize option works even when buffer.size is omitted, and cannot exceed buffer.size
  assert.throws(
    () => encodeIndirectDispatch(pass, { usage: BUFFER_USAGE.INDIRECT }, 8, { bufferSize: 16 }),
    (e) => e instanceof IndirectDrawError && e.code === 'OUT_OF_BOUNDS'
  );
  assert.throws(
    () => encodeIndirectDraw(pass, indirectBuf(16), 4, { bufferSize: 64 }),
    (e) => e instanceof IndirectDrawError && e.code === 'OUT_OF_BOUNDS'
  );

  // Unknown bufferSize still enforces 4-byte alignment
  assert.throws(
    () => encodeIndirectDraw(pass, { usage: BUFFER_USAGE.INDIRECT }, 2),
    (e) => e instanceof IndirectDrawError && e.code === 'MISALIGNED_OFFSET'
  );

  assert.equal(pass.calls.length, 0, 'no pass calls emitted for rejected offsets');
  assert.equal(getStats().boundsRejectedCount, 9);

  // Exact-fit calls succeed for all three record sizes (16 / 20 / 12) and unknown bufferSize when aligned
  encodeIndirectDraw(pass, indirectBuf(32), 16);
  encodeIndirectDrawIndexed(pass, indirectBuf(40), 20);
  encodeIndirectDispatch(pass, { usage: BUFFER_USAGE.INDIRECT }, 8, { bufferSize: 20 });
  encodeIndirectDispatch(pass, { usage: BUFFER_USAGE.INDIRECT }, 4);
  assert.deepEqual(pass.calls, [
    ['drawIndirect', 16],
    ['drawIndexedIndirect', 20],
    ['dispatchWorkgroupsIndirect', 8],
    ['dispatchWorkgroupsIndirect', 4],
  ]);
  const s = getStats();
  assert.equal(s.indirectDrawsEncoded, 1);
  assert.equal(s.indirectIndexedDrawsEncoded, 1);
  assert.equal(s.indirectDispatchesEncoded, 2);
  assert.equal(s.boundsRejectedCount, 9);
});

test('multi-draw and multi-dispatch validate full batch atomically before issuing pass calls', () => {
  resetStats();
  const pass = createMockPass();

  // Slot 0 fits (0..20 <= 30) but slot 1 overflows (20..40 > 30): zero draws should be emitted
  assert.throws(
    () => encodeMultiDraw(pass, indirectBuf(30), { indexed: true, count: 2 }),
    (e) => e instanceof IndirectDrawError && e.code === 'OUT_OF_BOUNDS'
  );
  assert.equal(pass.calls.length, 0);

  // Multi-dispatch atomic bounds rejection (stride 16, count 3 -> 16*2 + 12 = 44 > 40)
  assert.throws(
    () => encodeMultiDispatchIndirect(pass, indirectBuf(40), { count: 3, stride: 16 }),
    (e) => e instanceof IndirectDrawError && e.code === 'OUT_OF_BOUNDS'
  );
  assert.equal(pass.calls.length, 0);

  // Stride misalignment rejection increments boundsRejectedCount
  assert.throws(
    () => encodeMultiDispatchIndirect(pass, indirectBuf(128), { count: 2, stride: 14 }),
    RangeError
  );
  assert.equal(getStats().boundsRejectedCount, 3);
});

test('maxIndirectBatchCount rejects oversized batches by default and clamps when clampBatch is true', () => {
  resetStats();
  assert.equal(DEFAULT_MAX_INDIRECT_BATCH_COUNT, 4096);
  const pass = createMockPass();

  // encodeMultiDraw rejects count > maxIndirectBatchCount
  assert.throws(
    () => encodeMultiDraw(pass, indirectBuf(1024), { count: 5, maxIndirectBatchCount: 3 }),
    (e) => e instanceof IndirectDrawError && e.code === 'BATCH_LIMIT_EXCEEDED'
  );
  assert.equal(pass.calls.length, 0);
  assert.equal(getStats().boundsRejectedCount, 1);
  assert.equal(getStats().batchLimitRejectedCount, 1);

  // encodeMultiDraw with clampBatch: true bounds execution to maxIndirectBatchCount
  const clampedDraw = encodeMultiDraw(pass, indirectBuf(1024), {
    count: 5,
    maxIndirectBatchCount: 3,
    clampBatch: true,
  });
  assert.deepEqual(clampedDraw, { calls: 3, draws: 3, skipped: 0, native: false });
  assert.equal(pass.calls.length, 3);
  assert.equal(getStats().batchClampedCount, 1);
  assert.equal(getStats().indirectDrawsEncoded, 3);

  // encodeMultiDispatchIndirect enforces maxIndirectBatchCount (reject + clamp)
  pass.calls.length = 0;
  assert.throws(
    () => encodeMultiDispatchIndirect(pass, indirectBuf(1024), { count: 4, maxIndirectBatchCount: 2 }),
    (e) => e instanceof IndirectDrawError && e.code === 'BATCH_LIMIT_EXCEEDED'
  );
  assert.equal(pass.calls.length, 0);
  const clampedDispatch = encodeMultiDispatchIndirect(pass, indirectBuf(1024), {
    count: 4,
    maxIndirectBatchCount: 2,
    clampBatch: true,
  });
  assert.deepEqual(clampedDispatch, { calls: 2, dispatches: 2, skipped: 0, clamped: true });
  assert.deepEqual(pass.calls, [
    ['dispatchWorkgroupsIndirect', 0],
    ['dispatchWorkgroupsIndirect', 12],
  ]);

  // encodeBatch enforces maxIndirectBatchCount (reject + clamp)
  assert.throws(
    () => encodeBatch(INDIRECT_KIND.DRAW_INDEXED, [rec(1), rec(2), rec(3)], { maxIndirectBatchCount: 2 }),
    (e) => e instanceof IndirectDrawError && e.code === 'BATCH_LIMIT_EXCEEDED'
  );
  const clampedBatch = encodeBatch(INDIRECT_KIND.DRAW_INDEXED, [rec(1), rec(2), rec(3)], {
    maxIndirectBatchCount: 2,
    clampBatch: true,
  });
  assert.equal(clampedBatch.bytes.byteLength, 40);

  // validateIndirectBatch enforces maxIndirectBatchCount and validates spanned bytes
  assert.throws(
    () => validateIndirectBatch(indirectBuf(1024), { kind: INDIRECT_KIND.DISPATCH, count: 10, maxIndirectBatchCount: 4 }),
    (e) => e instanceof IndirectDrawError && e.code === 'BATCH_LIMIT_EXCEEDED'
  );
  const validated = validateIndirectBatch(indirectBuf(64), {
    kind: INDIRECT_KIND.DISPATCH,
    count: 10,
    maxIndirectBatchCount: 4,
    clampBatch: true,
  });
  assert.deepEqual(validated, {
    count: 4,
    requestedCount: 10,
    clamped: true,
    recordSize: 12,
    stride: 12,
    firstOffset: 0,
    requiredBytes: 48,
  });

  // Invalid maxIndirectBatchCount throws RangeError
  assert.throws(
    () => encodeMultiDraw(pass, indirectBuf(64), { count: 1, maxIndirectBatchCount: 0 }),
    RangeError
  );
});

test('count === 0 batches cleanly skip without GPU calls and increment zeroBatchSkips', () => {
  resetStats();
  const pass = createMockPass();

  const r1 = encodeMultiDraw(pass, indirectBuf(0), { count: 0 });
  assert.deepEqual(r1, { calls: 0, draws: 0, skipped: 0, native: false });

  const r2 = encodeMultiDispatchIndirect(pass, indirectBuf(0), { count: 0 });
  assert.deepEqual(r2, { calls: 0, dispatches: 0, skipped: 0, clamped: false });

  const r3 = encodeBatch(INDIRECT_KIND.DRAW, []);
  assert.equal(r3.bytes.byteLength, 0);
  assert.deepEqual(r3.patched, []);

  const { device } = createMockDevice();
  const args = new IndirectArgsBuffer({ device, capacity: 4 });
  const r4 = args.encodeDraws(pass);
  assert.deepEqual(r4, { calls: 0, draws: 0, skipped: 0, native: false });

  assert.equal(pass.calls.length, 0);
  assert.equal(getStats().zeroBatchSkips, 4);
  assert.equal(args.getStats().zeroBatchSkips, 1);
});

test('telemetry tracks drawIndirect, drawIndexedIndirect, dispatchWorkgroupsIndirect and resetStats zeroes all counters', () => {
  resetStats();
  const pass = createMockPass(true);

  // Single-call helpers
  encodeIndirectDraw(pass, indirectBuf(64), 0);
  encodeIndirectDrawIndexed(pass, indirectBuf(64), 20);
  encodeIndirectDispatch(pass, indirectBuf(64), 12);

  // Batched helpers (emulated + native + visibility-masked dispatch)
  encodeMultiDraw(pass, indirectBuf(128), { indexed: false, count: 3 });
  encodeMultiDraw(pass, indirectBuf(200), {
    indexed: true,
    useNative: true,
    count: 5,
    visibility: [1, 1, 0, 1, 1],
  });
  encodeMultiDispatchIndirect(pass, indirectBuf(128), {
    count: 4,
    visibility: [true, false, true, true],
  });

  // Zero-batch skip + bounds rejection
  encodeMultiDraw(pass, indirectBuf(64), { count: 0 });
  assert.throws(
    () => encodeIndirectDraw(pass, indirectBuf(16), 4),
    (e) => e.code === 'OUT_OF_BOUNDS'
  );

  const snapshot = getStats();
  assert.equal(snapshot.indirectDrawsEncoded, 4);
  assert.equal(snapshot.indirectIndexedDrawsEncoded, 5);
  assert.equal(snapshot.indirectDispatchesEncoded, 4);
  assert.equal(snapshot.indirectDraws, 4);
  assert.equal(snapshot.indirectIndexedDraws, 5);
  assert.equal(snapshot.indirectDispatches, 4);
  assert.equal(snapshot.boundsRejectedCount, 1);
  assert.equal(snapshot.clampedOrSkippedInvalid, 1);
  assert.equal(snapshot.maxMultiDrawCount, 4);
  assert.equal(snapshot.zeroBatchSkips, 1);

  // Returned snapshot is sanitized (finite non-negative integers only) and detached
  for (const [key, val] of Object.entries(snapshot)) {
    assert.equal(typeof val, 'number', `stat ${key} must be numeric`);
    assert.ok(Number.isInteger(val) && val >= 0, `stat ${key} must be a non-negative integer`);
  }
  snapshot.indirectDrawsEncoded = 999999;
  snapshot.indirectDraws = 999999;
  assert.equal(getStats().indirectDrawsEncoded, 4);
  assert.equal(getStats().indirectDraws, 4);

  // resetStats zeroes module counters
  resetStats();
  assert.deepEqual(getStats(), {
    indirectDrawsEncoded: 0,
    indirectIndexedDrawsEncoded: 0,
    indirectDispatchesEncoded: 0,
    boundsRejectedCount: 0,
    zeroBatchSkips: 0,
    batchLimitRejectedCount: 0,
    batchClampedCount: 0,
    indirectDraws: 0,
    indirectIndexedDraws: 0,
    indirectDispatches: 0,
    clampedOrSkippedInvalid: 0,
    maxMultiDrawCount: 0,
    validatedIndirectDraws: 0,
    skippedOutOfBoundsDraws: 0,
    skippedMisalignedDraws: 0,
    skippedZeroCountDraws: 0,
    batchesDispatched: 0,
    commandsValidated: 0,
    zeroCountElided: 0,
    outOfBoundsElided: 0,
  });

  // IndirectArgsBuffer instance telemetry + maxIndirectBatchCount + resetStats
  const { device } = createMockDevice();
  const args = new IndirectArgsBuffer({ device, capacity: 8, maxIndirectBatchCount: 2 });
  args.allocate(rec(1));
  args.allocate(rec(2));
  args.allocate(rec(3));

  assert.throws(
    () => args.encodeDraws(pass),
    (e) => e instanceof IndirectDrawError && e.code === 'BATCH_LIMIT_EXCEEDED'
  );
  args.encodeDraws(pass, { clampBatch: true });
  args.encodeDraws(pass, { count: 0 });

  const bufStats = args.getStats();
  assert.equal(bufStats.indirectIndexedDrawsEncoded, 2);
  assert.equal(bufStats.indirectIndexedDraws, 2);
  assert.equal(bufStats.indirectDrawsEncoded, 0);
  assert.equal(bufStats.indirectDraws, 0);
  assert.equal(bufStats.indirectDispatchesEncoded, 0);
  assert.equal(bufStats.indirectDispatches, 0);
  assert.equal(bufStats.boundsRejectedCount, 1);
  assert.equal(bufStats.batchLimitRejectedCount, 1);
  assert.equal(bufStats.batchClampedCount, 1);
  assert.equal(bufStats.clampedOrSkippedInvalid, 2);
  assert.equal(bufStats.maxMultiDrawCount, 2);
  assert.equal(bufStats.zeroBatchSkips, 1);
  assert.equal(bufStats.liveSlots, 3);
  assert.equal(bufStats.maxIndirectBatchCount, 2);

  args.resetStats();
  const clearedBufStats = args.getStats();
  assert.equal(clearedBufStats.indirectIndexedDrawsEncoded, 0);
  assert.equal(clearedBufStats.indirectIndexedDraws, 0);
  assert.equal(clearedBufStats.boundsRejectedCount, 0);
  assert.equal(clearedBufStats.zeroBatchSkips, 0);
  assert.equal(clearedBufStats.batchLimitRejectedCount, 0);
  assert.equal(clearedBufStats.batchClampedCount, 0);
  assert.equal(clearedBufStats.clampedOrSkippedInvalid, 0);
  assert.equal(clearedBufStats.maxMultiDrawCount, 0);
  assert.equal(clearedBufStats.liveSlots, 3, 'resetStats preserves live slot metadata');
});

test('W205: fast-path isValidIndirectRange & checkIndirectRange validate alignment, bounds, usage, and liveness without mutating stats', () => {
  resetStats();
  const buf64 = indirectBuf(64);
  const destroyedBuf = { usage: BUFFER_USAGE.INDIRECT, size: 64, destroyed: true };
  const noIndirectBuf = { usage: BUFFER_USAGE.STORAGE, size: 64 };

  // Aligned & exact-fit offsets pass for all three record sizes (16 / 20 / 12)
  assert.equal(isValidIndirectRange(buf64, 0, 16), true);
  assert.equal(isValidIndirectRange(buf64, 48, 16), true);
  assert.equal(isValidIndirectRange(buf64, 44, 20), true);
  assert.equal(isValidIndirectRange(buf64, 52, 12), true);
  // > 2 GiB safe-integer 4-byte-aligned offset passes
  assert.equal(isValidIndirectRange({ usage: BUFFER_USAGE.INDIRECT, size: 0x100000100 }, 0x100000004, 16), true);

  // Misaligned offsets fail (both Smi and > 2 GiB)
  for (const badOffset of [1, 2, 3, 6, 18, 0x100000002]) {
    const buf = { usage: BUFFER_USAGE.INDIRECT, size: 0x100000100 };
    assert.equal(isValidIndirectRange(buf, badOffset, 16), false);
    const res = checkIndirectRange(buf, badOffset, 16);
    assert.equal(res.ok, false);
    assert.equal(res.code, 'MISALIGNED_OFFSET');
  }

  // Out-of-bounds offsets fail (1 byte short & explicit bufferSize cap)
  assert.equal(checkIndirectRange(buf64, 52, 16).code, 'OUT_OF_BOUNDS');
  assert.equal(checkIndirectRange(buf64, 48, 20).code, 'OUT_OF_BOUNDS');
  assert.equal(checkIndirectRange(buf64, 56, 12).code, 'OUT_OF_BOUNDS');
  assert.equal(checkIndirectRange(buf64, 0, 16, { bufferSize: 12 }).code, 'OUT_OF_BOUNDS');
  assert.equal(checkIndirectRange(buf64, 0, 16, { bufferSize: -4 }).code, 'OUT_OF_BOUNDS');

  // Non-integer / negative / NaN offsets fail cleanly
  for (const invalidOffset of [-4, 1.5, NaN, Infinity, '0', null]) {
    assert.equal(checkIndirectRange(buf64, invalidOffset, 16).code, 'INVALID_OFFSET');
  }

  // Missing INDIRECT usage and destroyed buffer fail cleanly
  assert.equal(checkIndirectRange(noIndirectBuf, 0, 16).code, 'NOT_INDIRECT');
  assert.equal(checkIndirectRange(null, 0, 16).code, 'NOT_INDIRECT');
  assert.equal(checkIndirectRange(destroyedBuf, 0, 16).code, 'DESTROYED_BUFFER');
  assert.throws(
    () => validateIndirectRange(destroyedBuf, 0, 16),
    (e) => e instanceof IndirectDrawError && e.code === 'DESTROYED_BUFFER'
  );

  // Only the throwing validateIndirectRange call mutated stats; pure checks did not
  assert.equal(getStats().boundsRejectedCount, 1);
  assert.equal(getStats().clampedOrSkippedInvalid, 1);
});

test('W205: encodeIndirect* with skipInvalid skips misaligned/OOB/destroyed calls without emitting GPU commands', () => {
  resetStats();
  const pass = createMockPass();
  const buf32 = indirectBuf(32);
  const destroyedBuf = { usage: BUFFER_USAGE.INDIRECT, size: 64, destroyed: true };

  // Misaligned offsets skipped cleanly
  const r1 = encodeIndirectDraw(pass, buf32, 2, { skipInvalid: true });
  assert.equal(r1.executed, false);
  assert.equal(r1.skipped, true);
  assert.equal(r1.reason, 'MISALIGNED_OFFSET');

  const r2 = encodeIndirectDrawIndexed(pass, buf32, 6, { skipInvalid: true });
  assert.equal(r2.skipped, true);
  assert.equal(r2.reason, 'MISALIGNED_OFFSET');

  const r3 = encodeIndirectDispatch(pass, buf32, 1, { skipInvalid: true });
  assert.equal(r3.skipped, true);
  assert.equal(r3.reason, 'MISALIGNED_OFFSET');

  // Out-of-bounds skipped cleanly
  const r4 = encodeIndirectDraw(pass, buf32, 20, { skipInvalid: true });
  assert.equal(r4.skipped, true);
  assert.equal(r4.reason, 'OUT_OF_BOUNDS');

  const r5 = encodeIndirectDrawIndexed(pass, buf32, 16, { skipInvalid: true });
  assert.equal(r5.skipped, true);
  assert.equal(r5.reason, 'OUT_OF_BOUNDS');

  const r6 = encodeIndirectDispatch(pass, buf32, 24, { skipInvalid: true });
  assert.equal(r6.skipped, true);
  assert.equal(r6.reason, 'OUT_OF_BOUNDS');

  // Destroyed buffer & missing pass method skipped cleanly
  const r7 = encodeIndirectDraw(pass, destroyedBuf, 0, { skipInvalid: true });
  assert.equal(r7.skipped, true);
  assert.equal(r7.reason, 'DESTROYED_BUFFER');

  const r8 = encodeIndirectDispatch({}, buf32, 0, { skipInvalid: true });
  assert.equal(r8.skipped, true);
  assert.equal(r8.reason, 'MISSING_PASS_METHOD');

  assert.equal(pass.calls.length, 0, 'zero WebGPU pass calls emitted for invalid inputs');
  assert.equal(getStats().clampedOrSkippedInvalid, 8);
  assert.equal(getStats().boundsRejectedCount, 8);

  // Valid calls still execute normally with skipInvalid: true
  encodeIndirectDraw(pass, buf32, 16, { skipInvalid: true });
  encodeIndirectDrawIndexed(pass, buf32, 12, { skipInvalid: true });
  encodeIndirectDispatch(pass, buf32, 20, { skipInvalid: true });
  assert.deepEqual(pass.calls, [
    ['drawIndirect', 16],
    ['drawIndexedIndirect', 12],
    ['dispatchWorkgroupsIndirect', 20],
  ]);
  const s = getStats();
  assert.equal(s.indirectDraws, 1);
  assert.equal(s.indirectIndexedDraws, 1);
  assert.equal(s.indirectDispatches, 1);
  assert.equal(s.clampedOrSkippedInvalid, 8);
});

test('W205: executeIndirect* steps ExecuteIndirect (count > 1) offsets with packed and custom strides', () => {
  resetStats();
  const pass = createMockPass(true);

  // Non-indexed multi-draw with custom D3D12 command signature stride (32B) and argOffset 12
  // Even with useNative: true, custom stride (> 16B) must step via drawIndirect (never native packed multiDraw)
  const rDraw = executeIndirectDraw(pass, indirectBuf(256), 12, {
    count: 4,
    stride: 32,
    useNative: true,
  });
  assert.deepEqual(rDraw, {
    ok: true,
    executed: 4,
    calls: 4,
    requestedCount: 4,
    clamped: false,
    skipped: 0,
    reason: null,
    native: false,
    firstOffset: 12,
    stride: 32,
  });
  assert.deepEqual(pass.calls, [
    ['drawIndirect', 12],
    ['drawIndirect', 44],
    ['drawIndirect', 76],
    ['drawIndirect', 108],
  ]);

  // Indexed multi-draw with packed stride (20B) + firstSlot + maxCommandCount alias + native multiDraw
  pass.calls.length = 0;
  const rIndexed = executeIndirectDrawIndexed(pass, indirectBuf(256), 20, {
    maxCommandCount: 5,
    firstSlot: 1,
    useNative: true,
  });
  assert.deepEqual(rIndexed, {
    ok: true,
    executed: 5,
    calls: 1,
    requestedCount: 5,
    clamped: false,
    skipped: 0,
    reason: null,
    native: true,
    firstOffset: 40,
    stride: 20,
  });
  assert.deepEqual(pass.calls, [['multiDrawIndexedIndirect', 40, 5]]);

  // Multi-dispatch stepping with 16-byte stride (12B payload + 4B padding)
  pass.calls.length = 0;
  const rDisp = executeIndirectDispatch(pass, indirectBuf(128), 8, {
    count: 3,
    stride: 16,
  });
  assert.deepEqual(rDisp, {
    ok: true,
    executed: 3,
    calls: 3,
    requestedCount: 3,
    clamped: false,
    skipped: 0,
    reason: null,
    native: false,
    firstOffset: 8,
    stride: 16,
  });
  assert.deepEqual(pass.calls, [
    ['dispatchWorkgroupsIndirect', 8],
    ['dispatchWorkgroupsIndirect', 24],
    ['dispatchWorkgroupsIndirect', 40],
  ]);

  // Single-command fast path (count omitted -> 1)
  pass.calls.length = 0;
  const rSingle = executeIndirectDraw(pass, indirectBuf(64), 16);
  assert.equal(rSingle.ok, true);
  assert.equal(rSingle.executed, 1);
  assert.deepEqual(pass.calls, [['drawIndirect', 16]]);

  const s = getStats();
  assert.equal(s.indirectDraws, 5);
  assert.equal(s.indirectIndexedDraws, 5);
  assert.equal(s.indirectDispatches, 3);
  assert.equal(s.clampedOrSkippedInvalid, 0);
  assert.equal(s.maxMultiDrawCount, 5);
});

test('W205: executeIndirect* and computeIndirectBatchStepPlan guard misaligned/OOB offsets and clamp overflowing counts', () => {
  resetStats();
  const pass = createMockPass();

  // Buffer of 84 bytes with stride 32 and 20B indexed records fits exactly 3 records:
  // [0..20], [32..52], [64..84] -> requesting count=7 clamps to 3
  const plan = computeIndirectBatchStepPlan(indirectBuf(84), {
    kind: INDIRECT_KIND.DRAW_INDEXED,
    count: 7,
    stride: 32,
  });
  assert.deepEqual(plan, {
    valid: true,
    reason: 'CLAMPED_TO_BUFFER',
    requestedCount: 7,
    effectiveCount: 3,
    clamped: true,
    clampedReason: 'CLAMPED_TO_BUFFER',
    recordSize: 20,
    stride: 32,
    firstOffset: 0,
    lastOffset: 64,
    requiredBytes: 84,
    bufferSize: 84,
  });

  const clamped = executeIndirectDrawIndexed(pass, indirectBuf(84), 0, {
    count: 7,
    stride: 32,
  });
  assert.equal(clamped.ok, true);
  assert.equal(clamped.executed, 3);
  assert.equal(clamped.clamped, true);
  assert.equal(clamped.skipped, 4);
  assert.equal(clamped.reason, 'CLAMPED_TO_BUFFER');
  assert.deepEqual(pass.calls, [
    ['drawIndexedIndirect', 0],
    ['drawIndexedIndirect', 32],
    ['drawIndexedIndirect', 64],
  ]);

  // Buffer of 83 bytes (1 byte too small for slot 2) clamps to 2
  pass.calls.length = 0;
  const clamped2 = executeIndirectDrawIndexed(pass, indirectBuf(83), 0, {
    count: 7,
    stride: 32,
  });
  assert.equal(clamped2.executed, 2);
  assert.deepEqual(pass.calls, [
    ['drawIndexedIndirect', 0],
    ['drawIndexedIndirect', 32],
  ]);

  // With clampToBuffer: false, overflowing count is rejected atomically (0 calls)
  pass.calls.length = 0;
  const rejectedOob = executeIndirectDrawIndexed(pass, indirectBuf(83), 0, {
    count: 3,
    stride: 32,
    clampToBuffer: false,
  });
  assert.equal(rejectedOob.ok, false);
  assert.equal(rejectedOob.executed, 0);
  assert.equal(rejectedOob.reason, 'OUT_OF_BOUNDS');
  assert.equal(pass.calls.length, 0);

  // Even slot 0 OOB -> skipped cleanly even when clampToBuffer is true
  const slot0Oob = executeIndirectDraw(pass, indirectBuf(12), 0, { count: 4 });
  assert.equal(slot0Oob.ok, false);
  assert.equal(slot0Oob.executed, 0);
  assert.equal(slot0Oob.reason, 'OUT_OF_BOUNDS');
  assert.equal(pass.calls.length, 0);

  // Misaligned argOffset / bad stride / destroyed buffer / batch limit rejected without pass calls
  const mis = executeIndirectDraw(pass, indirectBuf(256), 2, { count: 3 });
  assert.equal(mis.ok, false);
  assert.equal(mis.reason, 'MISALIGNED_OFFSET');

  const badStride = executeIndirectDraw(pass, indirectBuf(256), 0, { count: 3, stride: 18 });
  assert.equal(badStride.ok, false);
  assert.equal(badStride.reason, 'INVALID_STRIDE');

  const dest = executeIndirectDispatch(pass, { usage: BUFFER_USAGE.INDIRECT, size: 128, destroyed: true }, 0, { count: 2 });
  assert.equal(dest.ok, false);
  assert.equal(dest.reason, 'DESTROYED_BUFFER');

  const batchRej = executeIndirectDraw(pass, indirectBuf(1024), 0, {
    count: 10,
    maxIndirectBatchCount: 4,
    clampBatch: false,
  });
  assert.equal(batchRej.ok, false);
  assert.equal(batchRej.reason, 'BATCH_LIMIT_EXCEEDED');
  assert.equal(pass.calls.length, 0);

  // Strict mode throws on invalid args
  assert.throws(
    () => executeIndirect(pass, INDIRECT_KIND.DRAW, indirectBuf(64), 2, { count: 2, strict: true }),
    (e) => e instanceof IndirectDrawError && e.code === 'MISALIGNED_OFFSET'
  );
  assert.throws(
    () => executeIndirect(pass, INDIRECT_KIND.DRAW, indirectBuf(64), 0, { count: 1, strict: true, bufferSize: 8 }),
    (e) => e instanceof IndirectDrawError && e.code === 'OUT_OF_BOUNDS'
  );

  // Verify stats: 2 clamped batches + 8 skipped/rejected invalid calls = 10
  const s = getStats();
  assert.equal(s.indirectIndexedDraws, 5);
  assert.equal(s.batchClampedCount, 2);
  assert.equal(s.clampedOrSkippedInvalid, 10);
  assert.equal(s.maxMultiDrawCount, 3);
});

test('W205: encodeMultiDraw / encodeMultiDispatchIndirect clampToBuffer & skipInvalid + IndirectArgsBuffer.executeDraws', () => {
  resetStats();
  const pass = createMockPass(true);

  // encodeMultiDraw with clampToBuffer clamps 6 requested 16B draws in a 52B buffer to 3 (0, 16, 32; 48+16>52)
  const md = encodeMultiDraw(pass, indirectBuf(52), {
    count: 6,
    clampToBuffer: true,
    useNative: true,
  });
  assert.deepEqual(md, { calls: 1, draws: 3, skipped: 3, native: true });
  assert.deepEqual(pass.calls, [['multiDrawIndirect', 0, 3]]);

  // encodeMultiDraw with skipInvalid skips misaligned offset without throwing
  pass.calls.length = 0;
  const mdSkip = encodeMultiDraw(pass, indirectBuf(128), {
    count: 3,
    baseOffset: 2,
    skipInvalid: true,
  });
  assert.deepEqual(mdSkip, { calls: 0, draws: 0, skipped: 3, native: false });
  assert.equal(pass.calls.length, 0);

  // encodeMultiDispatchIndirect with clampToBuffer clamps 5 dispatches (12B) in 40B buffer to 3
  const mdisp = encodeMultiDispatchIndirect(pass, indirectBuf(40), {
    count: 5,
    clampToBuffer: true,
  });
  assert.deepEqual(mdisp, { calls: 3, dispatches: 3, skipped: 2, clamped: true });
  assert.deepEqual(pass.calls, [
    ['dispatchWorkgroupsIndirect', 0],
    ['dispatchWorkgroupsIndirect', 12],
    ['dispatchWorkgroupsIndirect', 24],
  ]);

  // IndirectArgsBuffer.executeDraws tracks per-instance W205 telemetry
  pass.calls.length = 0;
  const { device } = createMockDevice();
  const args = new IndirectArgsBuffer({ device, capacity: 4 });
  args.allocate(rec(10));
  args.allocate(rec(20));
  args.allocate(rec(30));
  const execRes = args.executeDraws(pass, { count: 6 }); // buffer capacity is 4 slots (80B) -> clamps to 4
  assert.equal(execRes.ok, true);
  assert.equal(execRes.executed, 4);
  assert.equal(execRes.clamped, true);
  assert.equal(execRes.reason, 'CLAMPED_TO_BUFFER');
  assert.equal(pass.calls.length, 4);

  const bufStats = args.getStats();
  assert.equal(bufStats.indirectIndexedDraws, 4);
  assert.equal(bufStats.clampedOrSkippedInvalid, 1);
  assert.equal(bufStats.maxMultiDrawCount, 4);

  // Classic Worker global export exposes all new W205 helpers
  assert.equal(globalThis.PlayGta6IndirectDraw.executeIndirect, executeIndirect);
  assert.equal(globalThis.PlayGta6IndirectDraw.executeIndirectDraw, executeIndirectDraw);
  assert.equal(globalThis.PlayGta6IndirectDraw.executeIndirectDrawIndexed, executeIndirectDrawIndexed);
  assert.equal(globalThis.PlayGta6IndirectDraw.executeIndirectDispatch, executeIndirectDispatch);
  assert.equal(globalThis.PlayGta6IndirectDraw.isValidIndirectRange, isValidIndirectRange);
  assert.equal(globalThis.PlayGta6IndirectDraw.checkIndirectRange, checkIndirectRange);
  assert.equal(globalThis.PlayGta6IndirectDraw.computeIndirectBatchStepPlan, computeIndirectBatchStepPlan);
});

test('W269: validateIndirectBufferArgs enforces 16B/20B/12B strides, alignment, bounds, and zero-count without mutating stats', () => {
  resetStats();
  const buf32 = indirectBuf(32);

  // Exact fits for drawIndirect (16B), drawIndexedIndirect (20B), dispatchWorkgroupsIndirect (12B)
  const vDraw = validateIndirectBufferArgs(INDIRECT_KIND.DRAW, buf32, 16);
  assert.equal(vDraw.ok, true);
  assert.equal(vDraw.requiredStrideBytes, 16);
  assert.equal(vDraw.zeroCount, false);

  const vIndexed = validateIndirectBufferArgs(INDIRECT_KIND.DRAW_INDEXED, buf32, 12);
  assert.equal(vIndexed.ok, true);
  assert.equal(vIndexed.requiredStrideBytes, 20);

  const vDisp = validateIndirectBufferArgs(INDIRECT_KIND.DISPATCH, buf32, 20);
  assert.equal(vDisp.ok, true);
  assert.equal(vDisp.requiredStrideBytes, 12);

  // Misaligned offsets (offset % 4 !== 0)
  for (const badOff of [1, 2, 3, 6, 14]) {
    const res = validateIndirectBufferArgs(INDIRECT_KIND.DRAW, buf32, badOff);
    assert.equal(res.ok, false);
    assert.equal(res.code, 'MISALIGNED_OFFSET');
  }

  // Negative & out-of-bounds offsets for each record size
  assert.equal(validateIndirectBufferArgs(INDIRECT_KIND.DRAW, buf32, -4).code, 'INVALID_OFFSET');
  assert.equal(validateIndirectBufferArgs(INDIRECT_KIND.DRAW, buf32, 20).code, 'OUT_OF_BOUNDS');
  assert.equal(validateIndirectBufferArgs(INDIRECT_KIND.DRAW_INDEXED, buf32, 16).code, 'OUT_OF_BOUNDS');
  assert.equal(validateIndirectBufferArgs(INDIRECT_KIND.DISPATCH, buf32, 24).code, 'OUT_OF_BOUNDS');
  assert.equal(
    validateIndirectBufferArgs(INDIRECT_KIND.DRAW, indirectBuf(64), 48, { bufferSize: 60 }).code,
    'OUT_OF_BOUNDS'
  );

  // Custom requiredStrideBytes enforcement
  assert.equal(
    validateIndirectBufferArgs(INDIRECT_KIND.DRAW, buf32, 0, { requiredStrideBytes: 32 }).ok,
    true
  );
  assert.equal(
    validateIndirectBufferArgs(INDIRECT_KIND.DRAW, buf32, 4, { requiredStrideBytes: 32 }).code,
    'OUT_OF_BOUNDS'
  );
  assert.equal(
    validateIndirectBufferArgs(INDIRECT_KIND.DRAW, buf32, 0, { requiredStrideBytes: 18 }).code,
    'INVALID_STRIDE'
  );

  // Zero-count detection
  assert.equal(isZeroCountRecord(INDIRECT_KIND.DRAW, { vertexCount: 0, instanceCount: 1 }), true);
  assert.equal(isZeroCountRecord(INDIRECT_KIND.DRAW, { vertexCount: 3, instanceCount: 0 }), true);
  assert.equal(isZeroCountRecord(INDIRECT_KIND.DRAW, { vertexCount: 3, instanceCount: 1 }), false);
  assert.equal(isZeroCountRecord(INDIRECT_KIND.DRAW_INDEXED, { indexCount: 0, instanceCount: 1 }), true);
  assert.equal(isZeroCountRecord(INDIRECT_KIND.DRAW_INDEXED, { indexCount: 36, instanceCount: 0 }), true);
  assert.equal(isZeroCountRecord(INDIRECT_KIND.DISPATCH, { x: 1, y: 0, z: 1 }), true);
  assert.equal(isZeroCountRecord(INDIRECT_KIND.DISPATCH, { x: 1, y: 1, z: 1 }), false);

  const zCount = validateIndirectBufferArgs(INDIRECT_KIND.DRAW, indirectBuf(0), 0, { count: 0 });
  assert.equal(zCount.ok, true);
  assert.equal(zCount.zeroCount, true);
  assert.equal(zCount.code, 'ZERO_COUNT');

  const zRec = validateIndirectBufferArgs(INDIRECT_KIND.DRAW_INDEXED, buf32, 0, {
    record: { indexCount: 36, instanceCount: 0 },
  });
  assert.equal(zRec.ok, true);
  assert.equal(zRec.zeroCount, true);

  // Pure validation does not mutate telemetry
  const stats = getStats();
  assert.equal(stats.validatedIndirectDraws, 0);
  assert.equal(stats.skippedOutOfBoundsDraws, 0);
  assert.equal(stats.skippedMisalignedDraws, 0);
  assert.equal(stats.skippedZeroCountDraws, 0);

  // UMD global export exposes W269 helpers
  assert.equal(globalThis.PlayGta6IndirectDraw.isZeroCountRecord, isZeroCountRecord);
  assert.equal(globalThis.PlayGta6IndirectDraw.validateIndirectBufferArgs, validateIndirectBufferArgs);
});

test('W269: valid draws, misaligned offsets, and out-of-bounds offsets update telemetry and never emit invalid WebGPU calls', () => {
  resetStats();
  const pass = createMockPass();
  const buf32 = indirectBuf(32);

  // Valid draws (16B draw, 20B indexed draw, 12B dispatch)
  assert.equal(executeIndirectDraw(pass, buf32, 16).ok, true);
  assert.equal(executeIndirectDrawIndexed(pass, buf32, 12).ok, true);
  assert.equal(executeIndirectDispatch(pass, buf32, 20).ok, true);
  assert.equal(pass.calls.length, 3);

  // Misaligned offsets safely elided without throwing
  const m1 = executeIndirectDraw(pass, buf32, 2);
  const m2 = executeIndirectDrawIndexed(pass, buf32, 6);
  const m3 = executeIndirectDispatch(pass, buf32, 1);
  const m4 = encodeIndirectDraw(pass, buf32, 3, { skipInvalid: true });
  assert.equal(m1.ok, false);
  assert.equal(m1.reason, 'MISALIGNED_OFFSET');
  assert.equal(m2.reason, 'MISALIGNED_OFFSET');
  assert.equal(m3.reason, 'MISALIGNED_OFFSET');
  assert.equal(m4.skipped, true);
  assert.equal(m4.reason, 'MISALIGNED_OFFSET');

  // Out-of-bounds & negative offsets safely elided without throwing
  const o1 = executeIndirectDraw(pass, buf32, 20); // 20 + 16 = 36 > 32
  const o2 = executeIndirectDrawIndexed(pass, buf32, 16); // 16 + 20 = 36 > 32
  const o3 = executeIndirectDispatch(pass, buf32, 24); // 24 + 12 = 36 > 32
  const o4 = executeIndirectDraw(pass, buf32, -4); // offset < 0
  const o5 = encodeIndirectDrawIndexed(pass, indirectBuf(16), 0, { skipInvalid: true }); // 0 + 20 > 16
  assert.equal(o1.ok, false);
  assert.equal(o1.reason, 'OUT_OF_BOUNDS');
  assert.equal(o2.reason, 'OUT_OF_BOUNDS');
  assert.equal(o3.reason, 'OUT_OF_BOUNDS');
  assert.equal(o4.reason, 'INVALID_OFFSET');
  assert.equal(o5.skipped, true);
  assert.equal(o5.reason, 'OUT_OF_BOUNDS');

  // Clamped multi-draw counts validated draws and OOB-skipped tail draws separately
  // 48B buffer with 16B drawIndirect fits 3 draws; requesting 5 executes 3 and skips 2 OOB
  const clamped = executeIndirectDraw(pass, indirectBuf(48), 0, { count: 5 });
  assert.equal(clamped.ok, true);
  assert.equal(clamped.executed, 3);
  assert.equal(clamped.skipped, 2);

  assert.equal(pass.calls.length, 6, 'only the 6 valid draws reached the WebGPU pass encoder');

  const s = getStats();
  assert.equal(s.validatedIndirectDraws, 6);
  assert.equal(s.skippedMisalignedDraws, 4);
  assert.equal(s.skippedOutOfBoundsDraws, 7); // 5 single OOB/negative + 2 clamped OOB tail draws
  assert.equal(s.skippedZeroCountDraws, 0);
});

test('W269: zero-count indirect draws and dispatches are safely elided across single, batched, shadowData, and IndirectArgsBuffer paths', () => {
  resetStats();
  const pass = createMockPass(true);
  const buf64 = indirectBuf(64);

  // 1. count === 0 on executeIndirect*, encodeIndirect*, encodeMultiDraw, encodeMultiDispatchIndirect
  const z1 = executeIndirectDraw(pass, indirectBuf(0), 0, { count: 0 });
  assert.equal(z1.ok, true);
  assert.equal(z1.executed, 0);

  const z2 = encodeIndirectDraw(pass, indirectBuf(0), 0, { count: 0 });
  assert.equal(z2.skipped, true);
  assert.equal(z2.reason, 'ZERO_COUNT');

  const z3 = encodeIndirectDrawIndexed(pass, indirectBuf(0), 0, { count: 0 });
  assert.equal(z3.skipped, true);
  assert.equal(z3.reason, 'ZERO_COUNT');

  const z4 = encodeIndirectDispatch(pass, indirectBuf(0), 0, { count: 0 });
  assert.equal(z4.skipped, true);
  assert.equal(z4.reason, 'ZERO_COUNT');

  // 2. Explicit zero-count record / field elision on single draws and dispatches
  const z5 = executeIndirectDraw(pass, buf64, 0, { record: { vertexCount: 0, instanceCount: 1 } });
  assert.equal(z5.ok, true);
  assert.equal(z5.executed, 0);
  assert.equal(z5.skipped, 1);
  assert.equal(z5.reason, 'ZERO_COUNT');

  const z6 = executeIndirectDrawIndexed(pass, buf64, 0, { indexCount: 36, instanceCount: 0 });
  assert.equal(z6.ok, true);
  assert.equal(z6.executed, 0);
  assert.equal(z6.reason, 'ZERO_COUNT');

  const z7 = executeIndirectDispatch(pass, buf64, 0, { record: { x: 4, y: 0, z: 1 } });
  assert.equal(z7.ok, true);
  assert.equal(z7.executed, 0);
  assert.equal(z7.reason, 'ZERO_COUNT');

  const z8 = encodeIndirectDraw(pass, buf64, 0, { vertexCount: 3, instanceCount: 0 });
  assert.equal(z8.skipped, true);
  assert.equal(z8.reason, 'ZERO_COUNT');

  assert.equal(pass.calls.length, 0, 'zero WebGPU calls emitted for zero-count draws');
  assert.equal(getStats().skippedZeroCountDraws, 8);
  assert.equal(getStats().validatedIndirectDraws, 0);

  // 3. CPU shadowData elision in multi-draw (native run coalescing skips zero-count holes)
  const { bytes: batchBytes } = encodeBatch(INDIRECT_KIND.DRAW_INDEXED, [
    { indexCount: 36, instanceCount: 1 }, // slot 0: active
    { indexCount: 0, instanceCount: 1 }, // slot 1: zero indexCount -> elided
    { indexCount: 24, instanceCount: 2 }, // slot 2: active
    { indexCount: 12, instanceCount: 0 }, // slot 3: zero instanceCount -> elided
    { indexCount: 18, instanceCount: 1 }, // slot 4: active
  ]);
  const resShadow = executeIndirectDrawIndexed(pass, indirectBuf(batchBytes.byteLength), 0, {
    count: 5,
    shadowData: batchBytes,
    useNative: true,
  });
  assert.equal(resShadow.ok, true);
  assert.equal(resShadow.executed, 3);
  assert.equal(resShadow.skipped, 2);
  assert.deepEqual(pass.calls, [
    ['multiDrawIndexedIndirect', 0, 1],
    ['multiDrawIndexedIndirect', 40, 1],
    ['multiDrawIndexedIndirect', 80, 1],
  ]);

  // 4. IndirectArgsBuffer.executeDraws with elideZeroCount: true elides zeroed/freed slots
  pass.calls.length = 0;
  const { device } = createMockDevice();
  const args = new IndirectArgsBuffer({ device, capacity: 4 });
  const s0 = args.allocate(rec(12));
  const s1 = args.allocate({ indexCount: 0, instanceCount: 1 });
  const s2 = args.allocate(rec(24));
  args.free(s0); // freed slot is zeroed in shadow
  assert.equal(s1, 1);
  assert.equal(s2, 2);

  const bufExec = args.executeDraws(pass, { elideZeroCount: true });
  assert.equal(bufExec.ok, true);
  assert.equal(bufExec.executed, 1);
  assert.equal(bufExec.skipped, 2);
  assert.deepEqual(pass.calls, [['drawIndexedIndirect', 40]]);

  const bufStats = args.getStats();
  assert.equal(bufStats.validatedIndirectDraws, 1);
  assert.equal(bufStats.skippedZeroCountDraws, 2);
  assert.equal(bufStats.skippedOutOfBoundsDraws, 0);
  assert.equal(bufStats.skippedMisalignedDraws, 0);

  const modStats = getStats();
  assert.equal(modStats.validatedIndirectDraws, 4); // 3 from shadowData + 1 from IndirectArgsBuffer
  assert.equal(modStats.skippedZeroCountDraws, 12); // 8 + 2 + 2

  // 5. resetStats cleanly zeroes all W269 counters on both module and IndirectArgsBuffer
  args.resetStats();
  const clearedBuf = args.getStats();
  assert.equal(clearedBuf.validatedIndirectDraws, 0);
  assert.equal(clearedBuf.skippedOutOfBoundsDraws, 0);
  assert.equal(clearedBuf.skippedMisalignedDraws, 0);
  assert.equal(clearedBuf.skippedZeroCountDraws, 0);
  assert.equal(clearedBuf.liveSlots, 2, 'resetStats preserves live slot metadata');

  resetStats();
  const clearedMod = getStats();
  assert.equal(clearedMod.validatedIndirectDraws, 0);
  assert.equal(clearedMod.skippedOutOfBoundsDraws, 0);
  assert.equal(clearedMod.skippedMisalignedDraws, 0);
  assert.equal(clearedMod.skippedZeroCountDraws, 0);
});

test('W306: validateIndirectDrawArgs / validateIndirectBufferArgs / checkIndirectRange support zero-alloc out scratch reuse and preserve default return shape', () => {
  resetStats();
  const buf64 = indirectBuf(64);
  const scratch = {};

  // 1. Positional `out` scratch reuse on validateIndirectDrawArgs (kind-first & buffer-first forms)
  const r1 = validateIndirectDrawArgs(INDIRECT_KIND.DRAW, buf64, 16, null, scratch);
  assert.equal(r1, scratch, 'returns caller scratch object directly');
  assert.deepEqual(r1, {
    ok: true,
    valid: true,
    code: null,
    reason: null,
    kind: INDIRECT_KIND.DRAW,
    offset: 16,
    recordSize: 16,
    requiredStrideBytes: 16,
    bufferSize: 64,
    zeroCount: false,
  });

  // Subsequent invalid call overwrites all fields on the same scratch object (no stale state)
  const r2 = validateIndirectDrawArgs(INDIRECT_KIND.DRAW_INDEXED, buf64, 48, { out: scratch });
  assert.equal(r2, scratch);
  assert.equal(r2.ok, false);
  assert.equal(r2.valid, false);
  assert.equal(r2.code, 'OUT_OF_BOUNDS');
  assert.equal(r2.reason, 'OUT_OF_BOUNDS');
  assert.equal(r2.kind, INDIRECT_KIND.DRAW_INDEXED);
  assert.equal(r2.recordSize, 20);
  assert.equal(r2.zeroCount, false);

  // Buffer-first convenience forms: (buffer, offset, options, out) and (buffer, offset, indexedBool, out)
  const r3 = validateIndirectDrawArgs(buf64, 44, { indexed: true }, scratch);
  assert.equal(r3, scratch);
  assert.equal(r3.ok, true);
  assert.equal(r3.code, null);
  assert.equal(r3.recordSize, 20);

  const r4 = validateIndirectDrawArgs(buf64, 2, true, scratch);
  assert.equal(r4, scratch);
  assert.equal(r4.ok, false);
  assert.equal(r4.code, 'MISALIGNED_OFFSET');

  // Zero-count detection via Uint32Array shadowData with zero-alloc scratch reuse
  const u32Shadow = new Uint32Array([36, 0, 0, 0, 12, 1, 0, 0]);
  const r5 = validateIndirectDrawArgs(
    INDIRECT_KIND.DRAW,
    buf64,
    0,
    { shadowData: u32Shadow },
    scratch
  );
  assert.equal(r5, scratch);
  assert.equal(r5.ok, true);
  assert.equal(r5.zeroCount, true);
  assert.equal(r5.code, 'ZERO_COUNT');

  const r6 = validateIndirectBufferArgs(
    INDIRECT_KIND.DRAW,
    buf64,
    16,
    { shadowData: u32Shadow, out: scratch }
  );
  assert.equal(r6, scratch);
  assert.equal(r6.ok, true);
  assert.equal(r6.zeroCount, false);
  assert.equal(r6.code, null);

  // Omitting `out` allocates a fresh result object preserving exact default return shape
  const d1 = validateIndirectDrawArgs(INDIRECT_KIND.DRAW, buf64, 0);
  const d2 = validateIndirectDrawArgs(INDIRECT_KIND.DRAW, buf64, 0);
  assert.notEqual(d1, d2);
  assert.deepEqual(d1, {
    ok: true,
    valid: true,
    code: null,
    reason: null,
    kind: INDIRECT_KIND.DRAW,
    offset: 0,
    recordSize: 16,
    requiredStrideBytes: 16,
    bufferSize: 64,
    zeroCount: false,
  });

  // checkIndirectRange also supports `out` scratch reuse without mutating telemetry
  const rangeScratch = {};
  const cr1 = checkIndirectRange(buf64, 44, 20, null, rangeScratch);
  assert.equal(cr1, rangeScratch);
  assert.deepEqual(cr1, { ok: true, code: null, offset: 44, recordSize: 20, bufferSize: 64 });
  const cr2 = checkIndirectRange(buf64, 48, 20, { out: rangeScratch });
  assert.equal(cr2, rangeScratch);
  assert.equal(cr2.ok, false);
  assert.equal(cr2.code, 'OUT_OF_BOUNDS');

  // Pure validation never mutates telemetry
  const stats = getStats();
  assert.equal(stats.commandsValidated, 0);
  assert.equal(stats.batchesDispatched, 0);
  assert.equal(stats.zeroCountElided, 0);
  assert.equal(stats.outOfBoundsElided, 0);

  // UMD global export exposes all W306 helpers
  assert.equal(globalThis.PlayGta6IndirectDraw.validateIndirectDrawArgs, validateIndirectDrawArgs);
  assert.equal(globalThis.PlayGta6IndirectDraw.dispatchIndirectDraw, dispatchIndirectDraw);
  assert.equal(globalThis.PlayGta6IndirectDraw.dispatchIndexedIndirectDraw, dispatchIndexedIndirectDraw);
  assert.equal(globalThis.PlayGta6IndirectDraw.dispatchIndirectBatch, dispatchIndirectBatch);
});

test('W306: dispatchIndirectDraw & dispatchIndexedIndirectDraw reuse out scratch, preserve W269 bounds checks, and update telemetry', () => {
  resetStats();
  const pass = createMockPass();
  const buf64 = indirectBuf(64);
  const scratch = {};

  // Valid non-indexed draw (16B) with positional `out` scratch
  const r1 = dispatchIndirectDraw(pass, buf64, 48, null, scratch);
  assert.equal(r1, scratch);
  assert.deepEqual(r1, {
    ok: true,
    dispatched: true,
    executed: 1,
    skipped: 0,
    elided: false,
    reason: null,
    offset: 48,
    byteSize: 16,
    indexed: false,
  });

  // Valid indexed draw (20B) with `options.out` scratch
  const r2 = dispatchIndexedIndirectDraw(pass, buf64, 44, { out: scratch });
  assert.equal(r2, scratch);
  assert.deepEqual(r2, {
    ok: true,
    dispatched: true,
    executed: 1,
    skipped: 0,
    elided: false,
    reason: null,
    offset: 44,
    byteSize: 20,
    indexed: true,
  });

  // Zero-count elision via Uint8Array shadow and inline fields reuses scratch cleanly
  const shadowBytes = encodeDrawIndirect({ vertexCount: 36, instanceCount: 0 });
  const r3 = dispatchIndirectDraw(pass, buf64, 0, { shadowData: shadowBytes }, scratch);
  assert.equal(r3, scratch);
  assert.equal(r3.ok, true);
  assert.equal(r3.dispatched, false);
  assert.equal(r3.executed, 0);
  assert.equal(r3.skipped, 1);
  assert.equal(r3.elided, true);
  assert.equal(r3.reason, 'ZERO_COUNT');

  const r4 = dispatchIndexedIndirectDraw(pass, buf64, 0, { indexCount: 0, instanceCount: 2, out: scratch });
  assert.equal(r4, scratch);
  assert.equal(r4.ok, true);
  assert.equal(r4.dispatched, false);
  assert.equal(r4.elided, true);
  assert.equal(r4.reason, 'ZERO_COUNT');

  // Out-of-bounds elision (16B draw at 52 > 64, 20B indexed draw at 48 > 64)
  const r5 = dispatchIndirectDraw(pass, buf64, 52, null, scratch);
  assert.equal(r5, scratch);
  assert.equal(r5.ok, false);
  assert.equal(r5.dispatched, false);
  assert.equal(r5.elided, true);
  assert.equal(r5.reason, 'OUT_OF_BOUNDS');

  const r6 = dispatchIndexedIndirectDraw(pass, buf64, 48, null, scratch);
  assert.equal(r6, scratch);
  assert.equal(r6.ok, false);
  assert.equal(r6.reason, 'OUT_OF_BOUNDS');

  // Misaligned offset and destroyed buffer elided safely
  const r7 = dispatchIndirectDraw(pass, buf64, 2, null, scratch);
  assert.equal(r7, scratch);
  assert.equal(r7.ok, false);
  assert.equal(r7.reason, 'MISALIGNED_OFFSET');

  const r8 = dispatchIndexedIndirectDraw(
    pass,
    { usage: BUFFER_USAGE.INDIRECT, size: 64, destroyed: true },
    0,
    null,
    scratch
  );
  assert.equal(r8, scratch);
  assert.equal(r8.ok, false);
  assert.equal(r8.reason, 'DESTROYED_BUFFER');

  // Omitting `out` returns fresh objects with exact default shape
  const fresh1 = dispatchIndirectDraw(pass, buf64, 0);
  const fresh2 = dispatchIndirectDraw(pass, buf64, 0);
  assert.notEqual(fresh1, fresh2);
  assert.equal(fresh1.dispatched, true);
  assert.equal(fresh2.dispatched, true);

  // Strict mode throws W269 errors on invalid args
  assert.throws(
    () => dispatchIndirectDraw(pass, buf64, 52, { strict: true }),
    (e) => e instanceof IndirectDrawError && e.code === 'OUT_OF_BOUNDS'
  );
  assert.throws(
    () => dispatchIndexedIndirectDraw(pass, buf64, 6, { strict: true }),
    (e) => e instanceof IndirectDrawError && e.code === 'MISALIGNED_OFFSET'
  );

  // executeIndirectDraw & encodeIndirectDraw also support `out` scratch reuse
  const execScratch = {};
  const ex1 = executeIndirectDraw(pass, buf64, 0, null, execScratch);
  assert.equal(ex1, execScratch);
  assert.equal(ex1.ok, true);
  assert.equal(ex1.executed, 1);

  const encScratch = {};
  const en1 = encodeIndirectDraw(pass, buf64, 0, null, encScratch);
  assert.equal(en1, encScratch);
  assert.equal(en1.offset, 0);
  assert.equal(en1.byteSize, 16);
  // Default encodeIndirectDraw without `out` preserves exact { offset, byteSize } shape
  assert.deepEqual(encodeIndirectDraw(pass, buf64, 16), { offset: 16, byteSize: 16 });

  assert.deepEqual(pass.calls, [
    ['drawIndirect', 48],
    ['drawIndexedIndirect', 44],
    ['drawIndirect', 0],
    ['drawIndirect', 0],
    ['drawIndirect', 0],
    ['drawIndirect', 0],
    ['drawIndirect', 16],
  ]);

  const s = getStats();
  assert.equal(s.commandsValidated, 15);
  assert.equal(s.validatedIndirectDraws, 7);
  assert.equal(s.zeroCountElided, 2);
  assert.equal(s.skippedZeroCountDraws, 2);
  assert.equal(s.outOfBoundsElided, 2);
  assert.equal(s.skippedOutOfBoundsDraws, 3); // 2 elided + 1 strict throw
  assert.equal(s.skippedMisalignedDraws, 2); // 1 elided + 1 strict throw
});

test('W306: dispatchIndirectBatch validates consecutive structs in a single pass with zero-count & OOB elision, native coalescing, and scratch reuse', () => {
  resetStats();
  const pass = createMockPass(true);
  const scratch = {};

  // 1. Mixed indexed batch (20B stride): 5 requested commands in an 80B buffer (fits slots 0..3; slot 4 is OOB)
  // Slot 0: active (36, 1)
  // Slot 1: zero indexCount -> elided
  // Slot 2: active (24, 2)
  // Slot 3: zero instanceCount -> elided
  // Slot 4: offset 80 + 20 = 100 > 80 -> OOB elided
  const { bytes: shadowBytes } = encodeBatch(INDIRECT_KIND.DRAW_INDEXED, [
    { indexCount: 36, instanceCount: 1 },
    { indexCount: 0, instanceCount: 1 },
    { indexCount: 24, instanceCount: 2 },
    { indexCount: 12, instanceCount: 0 },
  ]);

  const b1 = dispatchIndirectBatch(
    pass,
    indirectBuf(80),
    {
      indexed: true,
      count: 5,
      shadowData: shadowBytes,
    },
    scratch
  );
  assert.equal(b1, scratch, 'reuses caller scratch object');
  assert.deepEqual(b1, {
    ok: true,
    dispatched: 2,
    executed: 2,
    calls: 2,
    validated: 5,
    requestedCount: 5,
    clamped: true,
    skipped: 3,
    zeroCountElided: 2,
    outOfBoundsElided: 1,
    visibilitySkipped: 0,
    reason: 'CLAMPED_TO_BUFFER',
    native: false,
    firstOffset: 0,
    stride: 20,
  });
  assert.deepEqual(pass.calls, [
    ['drawIndexedIndirect', 0],
    ['drawIndexedIndirect', 40],
  ]);

  // 2. Native multi-draw coalescing around zero-count holes + OOB tail in one pass (Uint32Array shadow)
  pass.calls.length = 0;
  // 5 non-indexed draw records (16B each = 80B); buffer is 96B (fits 6), request 8 -> 2 OOB tail
  const u32DrawShadow = new Uint32Array([
    3, 1, 0, 0, // slot 0: active
    6, 2, 0, 0, // slot 1: active -> coalesced with slot 0 (run of 2 at offset 0)
    0, 1, 0, 0, // slot 2: zero vertexCount -> elided
    9, 1, 0, 0, // slot 3: active
    12, 1, 0, 0, // slot 4: active
    15, 0, 0, 0, // slot 5: zero instanceCount -> elided (run of 2 flushed at offset 48)
  ]);
  const b2 = dispatchIndirectBatch(pass, indirectBuf(96), {
    indexed: false,
    count: 8,
    shadowData: u32DrawShadow,
    useNative: true,
    out: scratch,
  });
  assert.equal(b2, scratch);
  assert.equal(b2.ok, true);
  assert.equal(b2.dispatched, 4);
  assert.equal(b2.calls, 2);
  assert.equal(b2.validated, 8);
  assert.equal(b2.zeroCountElided, 2);
  assert.equal(b2.outOfBoundsElided, 2);
  assert.equal(b2.native, true);
  assert.equal(b2.reason, 'CLAMPED_TO_BUFFER');
  assert.deepEqual(pass.calls, [
    ['multiDrawIndirect', 0, 2],
    ['multiDrawIndirect', 48, 2],
  ]);

  // 3. Custom stride (32B) with records array + visibility mask + baseOffset
  pass.calls.length = 0;
  const b3 = dispatchIndirectBatch(pass, indirectBuf(160), {
    indexed: true,
    baseOffset: 16,
    stride: 32,
    records: [
      { indexCount: 12, instanceCount: 1 }, // off 16: active
      { indexCount: 18, instanceCount: 1 }, // off 48: visibility false -> skipped
      { indexCount: 24, instanceCount: 0 }, // off 80: zero instanceCount -> elided
      { indexCount: 30, instanceCount: 1 }, // off 112: active (112+20=132 <= 160)
      { indexCount: 36, instanceCount: 1 }, // off 144: OOB (144+20=164 > 160) -> elided
    ],
    visibility: [1, 0, 1, 1, 1],
  });
  assert.notEqual(b3, scratch, 'omitting out allocates fresh batch result');
  assert.equal(b3.ok, true);
  assert.equal(b3.dispatched, 2);
  assert.equal(b3.validated, 5);
  assert.equal(b3.visibilitySkipped, 1);
  assert.equal(b3.zeroCountElided, 1);
  assert.equal(b3.outOfBoundsElided, 1);
  assert.deepEqual(pass.calls, [
    ['drawIndexedIndirect', 16],
    ['drawIndexedIndirect', 112],
  ]);

  // 4. Full-batch OOB emits zero GPU commands and records all commands as outOfBoundsElided
  pass.calls.length = 0;
  const b4 = dispatchIndirectBatch(
    pass,
    indirectBuf(16),
    { indexed: true, count: 3 },
    scratch
  );
  assert.equal(b4, scratch);
  assert.equal(b4.ok, false);
  assert.equal(b4.dispatched, 0);
  assert.equal(b4.validated, 3);
  assert.equal(b4.outOfBoundsElided, 3);
  assert.equal(b4.reason, 'OUT_OF_BOUNDS');
  assert.equal(pass.calls.length, 0);

  // 5. All-zero-count batch emits zero GPU commands, does not increment batchesDispatched
  const b5 = dispatchIndirectBatch(
    pass,
    indirectBuf(64),
    {
      indexed: false,
      records: [
        { vertexCount: 0, instanceCount: 1 },
        { vertexCount: 3, instanceCount: 0 },
      ],
    },
    scratch
  );
  assert.equal(b5.ok, true);
  assert.equal(b5.dispatched, 0);
  assert.equal(b5.zeroCountElided, 2);
  assert.equal(b5.reason, 'ZERO_COUNT');
  assert.equal(pass.calls.length, 0);

  // 6. Empty count === 0 batch and misaligned offset rejection
  const b6 = dispatchIndirectBatch(pass, indirectBuf(64), { count: 0 }, scratch);
  assert.equal(b6.ok, true);
  assert.equal(b6.dispatched, 0);
  assert.equal(b6.validated, 0);

  const b7 = dispatchIndirectBatch(pass, indirectBuf(64), { count: 2, baseOffset: 2 }, scratch);
  assert.equal(b7.ok, false);
  assert.equal(b7.reason, 'MISALIGNED_OFFSET');
  assert.equal(pass.calls.length, 0);

  // Verify sanitized telemetry totals
  const stats = getStats();
  assert.equal(stats.batchesDispatched, 3); // b1, b2, b3
  assert.equal(stats.commandsValidated, 5 + 8 + 5 + 3 + 2 + 2); // 25
  assert.equal(stats.zeroCountElided, 2 + 2 + 1 + 2); // 7
  assert.equal(stats.outOfBoundsElided, 1 + 2 + 1 + 3); // 7
  assert.equal(stats.validatedIndirectDraws, 2 + 4 + 2); // 8
  assert.equal(stats.skippedZeroCountDraws, 8); // 7 zero-count commands + 1 count===0 batch
  assert.equal(stats.skippedOutOfBoundsDraws, 7);
  assert.equal(stats.skippedMisalignedDraws, 2);

  for (const [key, val] of Object.entries(stats)) {
    assert.equal(typeof val, 'number', `stat ${key} must be numeric`);
    assert.ok(Number.isInteger(val) && val >= 0, `stat ${key} must be a non-negative integer`);
  }

  // resetStats clears all W306 counters cleanly on both module and IndirectArgsBuffer
  resetStats();
  const cleared = getStats();
  assert.equal(cleared.batchesDispatched, 0);
  assert.equal(cleared.commandsValidated, 0);
  assert.equal(cleared.zeroCountElided, 0);
  assert.equal(cleared.outOfBoundsElided, 0);
  assert.equal(cleared.validatedIndirectDraws, 0);
  assert.equal(cleared.skippedZeroCountDraws, 0);
  assert.equal(cleared.skippedOutOfBoundsDraws, 0);
  assert.equal(cleared.skippedMisalignedDraws, 0);

  const { device } = createMockDevice();
  const args = new IndirectArgsBuffer({ device, capacity: 4 });
  args.allocate(rec(12));
  args.allocate(rec(24));
  args.executeDraws(pass);
  assert.equal(args.getStats().batchesDispatched, 1);
  assert.equal(args.getStats().commandsValidated, 2);
  args.resetStats();
  const clearedBuf = args.getStats();
  assert.equal(clearedBuf.batchesDispatched, 0);
  assert.equal(clearedBuf.commandsValidated, 0);
  assert.equal(clearedBuf.zeroCountElided, 0);
  assert.equal(clearedBuf.outOfBoundsElided, 0);
  assert.equal(clearedBuf.liveSlots, 2);
});
