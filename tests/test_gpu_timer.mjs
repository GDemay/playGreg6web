import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const gpuTimerMod = require('../web/workers/gpu_timer.js');
const {
  DEFAULT_MAX_PASSES_PER_FRAME,
  DEFAULT_READBACK_SLOTS,
  DEFAULT_HISTORY_SIZE,
  BUFFER_USAGE,
  MAP_MODE,
  GpuPassTimer,
} = gpuTimerMod;

/**
 * Creates a realistic mock WebGPU device + command encoder harness that simulates:
 * - `GPUQuerySet({ type: 'timestamp', count })`
 * - `GPUBuffer` with `mapState`, deferred or immediate `mapAsync`, `getMappedRange`, `unmap`, `destroy`
 * - `GPUCommandEncoder` with `beginComputePass`, `beginRenderPass`, `resolveQuerySet`, `copyBufferToBuffer`
 * - Per-frame simulated nanosecond timestamps copied from querySet -> resolveBuffer -> readbackBuffer
 */
function createMockWebGpu({ supported = true, deferMap = false } = {}) {
  const querySets = [];
  const buffers = [];
  const resolveCalls = [];
  const copyCalls = [];
  const passDescriptors = [];
  const pendingMaps = [];

  let nextFrameTimestamps = [];

  function setFrameTimestamps(nsPairs) {
    nextFrameTimestamps = nsPairs.slice();
  }

  const device = {
    features: new Set(supported ? ['timestamp-query'] : []),
    createQuerySet(desc) {
      const qs = {
        kind: 'GPUQuerySet',
        desc: { ...desc },
        values: new BigUint64Array(desc.count),
        destroyed: false,
        destroyCalls: 0,
        destroy() {
          this.destroyed = true;
          this.destroyCalls += 1;
        },
      };
      querySets.push(qs);
      return qs;
    },
    createBuffer(desc) {
      const storage = new ArrayBuffer(desc.size);
      const buf = {
        kind: 'GPUBuffer',
        desc: { ...desc },
        size: desc.size,
        usage: desc.usage,
        _storage: storage,
        mapState: 'unmapped',
        mapCalls: 0,
        unmapCalls: 0,
        destroyed: false,
        destroyCalls: 0,
        mapAsync(mode) {
          this.mapCalls += 1;
          if (mode !== MAP_MODE.READ) {
            return Promise.reject(new Error(`Unexpected map mode: ${mode}`));
          }
          if (this.destroyed) {
            return Promise.reject(new Error('Buffer destroyed'));
          }
          if (this.mapState !== 'unmapped') {
            return Promise.reject(new Error(`Buffer not unmapped (state=${this.mapState})`));
          }
          this.mapState = 'pending';

          if (deferMap) {
            return new Promise((resolve, reject) => {
              pendingMaps.push({
                buffer: this,
                resolve: () => {
                  if (this.destroyed) {
                    reject(new Error('Buffer destroyed while mapping'));
                    return;
                  }
                  this.mapState = 'mapped';
                  resolve();
                },
                reject: (err) => {
                  this.mapState = 'unmapped';
                  reject(err);
                },
              });
            });
          }

          return Promise.resolve().then(() => {
            if (this.destroyed) {
              throw new Error('Buffer destroyed while mapping');
            }
            this.mapState = 'mapped';
          });
        },
        getMappedRange(offset = 0, size = this.size - offset) {
          if (this.mapState !== 'mapped') {
            throw new Error(`getMappedRange called while mapState=${this.mapState}`);
          }
          return this._storage.slice(offset, offset + size);
        },
        unmap() {
          this.unmapCalls += 1;
          this.mapState = 'unmapped';
        },
        destroy() {
          this.destroyed = true;
          this.destroyCalls += 1;
        },
      };
      buffers.push(buf);
      return buf;
    },
    createCommandEncoder() {
      return {
        beginComputePass(desc = {}) {
          passDescriptors.push({ kind: 'compute', desc });
          return { end() {} };
        },
        beginRenderPass(desc = {}) {
          passDescriptors.push({ kind: 'render', desc });
          return { end() {} };
        },
        resolveQuerySet(querySet, firstQuery, queryCount, destination, destinationOffset) {
          resolveCalls.push({ querySet, firstQuery, queryCount, destination, destinationOffset });
          // Populate querySet from test-injected timestamps before resolving into destination buffer.
          for (let i = 0; i < queryCount; i += 1) {
            querySet.values[firstQuery + i] =
              i < nextFrameTimestamps.length ? BigInt(nextFrameTimestamps[i]) : 0n;
          }
          const dstView = new BigUint64Array(
            destination._storage,
            destinationOffset,
            queryCount
          );
          for (let i = 0; i < queryCount; i += 1) {
            dstView[i] = querySet.values[firstQuery + i];
          }
        },
        copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
          copyCalls.push({ source, sourceOffset, destination, destinationOffset, size });
          const srcBytes = new Uint8Array(source._storage, sourceOffset, size);
          const dstBytes = new Uint8Array(destination._storage, destinationOffset, size);
          dstBytes.set(srcBytes);
        },
      };
    },
  };

  return {
    device,
    querySets,
    buffers,
    resolveCalls,
    copyCalls,
    passDescriptors,
    pendingMaps,
    setFrameTimestamps,
  };
}

test('UMD export attaches PlayGta6GpuTimer on globalThis and exposes defaults', () => {
  assert.equal(globalThis.PlayGta6GpuTimer.GpuPassTimer, GpuPassTimer);
  assert.equal(DEFAULT_MAX_PASSES_PER_FRAME, 32);
  assert.equal(DEFAULT_READBACK_SLOTS, 3);
  assert.equal(DEFAULT_HISTORY_SIZE, 60);
  assert.equal(BUFFER_USAGE.QUERY_RESOLVE, 0x0200);
  assert.equal(BUFFER_USAGE.MAP_READ, 0x0001);
  assert.equal(MAP_MODE.READ, 0x0001);
});

test('graceful no-op degradation when device lacks timestamp-query feature or is missing', async () => {
  const { device, querySets, buffers, resolveCalls, copyCalls } = createMockWebGpu({
    supported: false,
  });

  const timer = new GpuPassTimer({ device });
  assert.equal(timer.supported, false);
  assert.equal(querySets.length, 0);
  assert.equal(buffers.length, 0);

  // Every method degrades cleanly to a no-op and never throws.
  timer.beginFrame();
  assert.equal(timer.getPassTimestampWrites('compute'), undefined);
  assert.equal(timer.getPassTimestampWrites('main_render'), undefined);

  const encoder = device.createCommandEncoder();
  assert.equal(timer.resolveFrame(encoder), false);
  assert.equal(resolveCalls.length, 0);
  assert.equal(copyCalls.length, 0);

  const readback = await timer.collectCompletedReadback();
  assert.equal(readback, null);

  assert.deepEqual(timer.getStats(), {
    supported: false,
    resolvedFrames: 0,
    skippedFrames: 0,
    droppedPasses: 0,
    lastFrameGpuMs: 0,
    passStats: {},
  });

  timer.destroy();
  timer.destroy();

  // Also test missing device / malformed features object.
  const noDeviceTimer = new GpuPassTimer({});
  assert.equal(noDeviceTimer.supported, false);
  noDeviceTimer.beginFrame();
  assert.equal(noDeviceTimer.getPassTimestampWrites('pass'), undefined);
  assert.equal(noDeviceTimer.resolveFrame(encoder), false);
  assert.equal(await noDeviceTimer.collectCompletedReadback(), null);
  noDeviceTimer.destroy();
});

test('supported lifecycle: allocates resources, profiles compute + render passes, and converts ns to exact ms', async () => {
  const env = createMockWebGpu({ supported: true });
  const timer = new GpuPassTimer({
    device: env.device,
    maxPassesPerFrame: 8,
    readbackSlots: 3,
    historySize: 10,
  });

  assert.equal(timer.supported, true);
  assert.equal(env.querySets.length, 1);
  assert.deepEqual(env.querySets[0].desc, {
    label: 'playgta6-gpu-timer-queries',
    type: 'timestamp',
    count: 16,
  });

  // 1 resolve buffer + 3 readback staging buffers = 4 buffers total, each 16 * 8 = 128 bytes.
  assert.equal(env.buffers.length, 4);
  assert.equal(env.buffers[0].size, 128);
  assert.equal(env.buffers[0].usage, BUFFER_USAGE.QUERY_RESOLVE | BUFFER_USAGE.COPY_SRC);
  for (let i = 1; i <= 3; i += 1) {
    assert.equal(env.buffers[i].size, 128);
    assert.equal(env.buffers[i].usage, BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ);
  }

  timer.beginFrame();
  const encoder = env.device.createCommandEncoder();

  const computeTs = timer.getPassTimestampWrites('compute');
  assert.deepEqual(computeTs, {
    querySet: env.querySets[0],
    beginningOfPassWriteIndex: 0,
    endOfPassWriteIndex: 1,
  });
  const computePass = encoder.beginComputePass({ timestampWrites: computeTs });
  computePass.end();

  const renderTs = timer.getPassTimestampWrites('main_render');
  assert.deepEqual(renderTs, {
    querySet: env.querySets[0],
    beginningOfPassWriteIndex: 2,
    endOfPassWriteIndex: 3,
  });
  const renderPass = encoder.beginRenderPass({
    colorAttachments: [],
    timestampWrites: renderTs,
  });
  renderPass.end();

  assert.equal(env.passDescriptors.length, 2);
  assert.deepEqual(env.passDescriptors[0].desc.timestampWrites, computeTs);
  assert.deepEqual(env.passDescriptors[1].desc.timestampWrites, renderTs);

  // Inject large 64-bit nanosecond values (> Number.MAX_SAFE_INTEGER) to verify BigInt delta precision:
  // compute:     1.25 ms (1_250_000 ns)
  // main_render: 4.50 ms (4_500_000 ns)
  const baseNs = 100_000_000_000_000_000n;
  env.setFrameTimestamps([
    baseNs,
    baseNs + 1_250_000n,
    baseNs + 2_000_000n,
    baseNs + 6_500_000n,
  ]);

  assert.equal(timer.resolveFrame(encoder), true);
  assert.equal(env.resolveCalls.length, 1);
  assert.equal(env.resolveCalls[0].firstQuery, 0);
  assert.equal(env.resolveCalls[0].queryCount, 4);
  assert.equal(env.resolveCalls[0].destination, env.buffers[0]);
  assert.equal(env.resolveCalls[0].destinationOffset, 0);

  assert.equal(env.copyCalls.length, 1);
  assert.equal(env.copyCalls[0].source, env.buffers[0]);
  assert.equal(env.copyCalls[0].destination, env.buffers[1]);
  assert.equal(env.copyCalls[0].size, 32); // 4 timestamps * 8 bytes

  const result = await timer.collectCompletedReadback();
  assert.deepEqual(result, {
    totalGpuMs: 5.75,
    passes: [
      { name: 'compute', durationMs: 1.25 },
      { name: 'main_render', durationMs: 4.5 },
    ],
  });

  // Readback buffer must be unmapped after collection.
  assert.equal(env.buffers[1].mapCalls, 1);
  assert.equal(env.buffers[1].unmapCalls, 1);
  assert.equal(env.buffers[1].mapState, 'unmapped');

  // Collecting again when no readback is pending returns null.
  assert.equal(await timer.collectCompletedReadback(), null);

  assert.deepEqual(timer.getStats(), {
    supported: true,
    resolvedFrames: 1,
    skippedFrames: 0,
    droppedPasses: 0,
    lastFrameGpuMs: 5.75,
    passStats: {
      compute: { count: 1, lastMs: 1.25, avgMs: 1.25, maxMs: 1.25 },
      main_render: { count: 1, lastMs: 4.5, avgMs: 4.5, maxMs: 4.5 },
    },
  });

  timer.destroy();
});

test('skips resolveFrame cleanly without stalling when all readback slots are busy', async () => {
  const env = createMockWebGpu({ supported: true, deferMap: true });
  const timer = new GpuPassTimer({
    device: env.device,
    maxPassesPerFrame: 4,
    readbackSlots: 2,
    historySize: 8,
  });

  // Frame 1 -> occupies slot 0
  timer.beginFrame();
  timer.getPassTimestampWrites('main');
  env.setFrameTimestamps([1_000_000n, 3_000_000n]); // 2.0 ms
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);

  // Frame 2 -> occupies slot 1
  timer.beginFrame();
  timer.getPassTimestampWrites('main');
  env.setFrameTimestamps([1_000_000n, 4_000_000n]); // 3.0 ms
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);

  // Frame 3 -> all 2 slots are pending, must skip without throwing or encoding resolve
  timer.beginFrame();
  timer.getPassTimestampWrites('main');
  env.setFrameTimestamps([1_000_000n, 9_000_000n]);
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), false);
  assert.equal(env.resolveCalls.length, 2);
  assert.equal(timer.getStats().skippedFrames, 1);

  // Start async readback for Frame 1 (slot 0 enters 'mapping' state, still busy!)
  const p1 = timer.collectCompletedReadback();
  assert.equal(env.pendingMaps.length, 1);

  // Frame 4 -> slot 0 is still mapping, slot 1 is still pending -> still skips!
  timer.beginFrame();
  timer.getPassTimestampWrites('main');
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), false);
  assert.equal(timer.getStats().skippedFrames, 2);

  // Complete Frame 1 map -> slot 0 frees up!
  env.pendingMaps.shift().resolve();
  const r1 = await p1;
  assert.deepEqual(r1, {
    totalGpuMs: 2.0,
    passes: [{ name: 'main', durationMs: 2.0 }],
  });

  // Frame 5 -> now slot 0 is idle again and can be reused immediately!
  timer.beginFrame();
  timer.getPassTimestampWrites('main');
  env.setFrameTimestamps([1_000_000n, 6_000_000n]); // 5.0 ms
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
  assert.equal(env.resolveCalls.length, 3);

  // Collect Frame 2 then Frame 5 in FIFO order.
  const p2 = timer.collectCompletedReadback();
  env.pendingMaps.shift().resolve();
  const r2 = await p2;
  assert.equal(r2.totalGpuMs, 3.0);

  const p5 = timer.collectCompletedReadback();
  env.pendingMaps.shift().resolve();
  const r5 = await p5;
  assert.equal(r5.totalGpuMs, 5.0);

  const stats = timer.getStats();
  assert.equal(stats.resolvedFrames, 3);
  assert.equal(stats.skippedFrames, 2);
  assert.equal(stats.lastFrameGpuMs, 5.0);
  assert.equal(stats.passStats.main.count, 3);
  assert.equal(stats.passStats.main.lastMs, 5.0);
  assert.equal(stats.passStats.main.maxMs, 5.0);
  assert.ok(Math.abs(stats.passStats.main.avgMs - 10 / 3) < 1e-9);

  timer.destroy();
});

test('query-budget exhaustion returns undefined and increments droppedPasses', async () => {
  const env = createMockWebGpu({ supported: true });
  const timer = new GpuPassTimer({
    device: env.device,
    maxPassesPerFrame: 2,
    readbackSlots: 2,
  });

  timer.beginFrame();
  const ts0 = timer.getPassTimestampWrites('shadow');
  const ts1 = timer.getPassTimestampWrites('gbuffer');
  const ts2 = timer.getPassTimestampWrites('post');
  const ts3 = timer.getPassTimestampWrites('ui');

  assert.ok(ts0);
  assert.equal(ts0.beginningOfPassWriteIndex, 0);
  assert.equal(ts0.endOfPassWriteIndex, 1);

  assert.ok(ts1);
  assert.equal(ts1.beginningOfPassWriteIndex, 2);
  assert.equal(ts1.endOfPassWriteIndex, 3);

  assert.equal(ts2, undefined);
  assert.equal(ts3, undefined);
  assert.equal(timer.getStats().droppedPasses, 2);

  env.setFrameTimestamps([0n, 1_500_000n, 2_000_000n, 4_500_000n]);
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);

  const result = await timer.collectCompletedReadback();
  assert.deepEqual(result, {
    totalGpuMs: 4.0,
    passes: [
      { name: 'shadow', durationMs: 1.5 },
      { name: 'gbuffer', durationMs: 2.5 },
    ],
  });

  // Next frame resets query cursor cleanly.
  timer.beginFrame();
  const nextTs = timer.getPassTimestampWrites('shadow');
  assert.equal(nextTs.beginningOfPassWriteIndex, 0);
  assert.equal(nextTs.endOfPassWriteIndex, 1);

  timer.destroy();
});

test('clamps negative and clock-wrap timestamp deltas safely to 0', async () => {
  const env = createMockWebGpu({ supported: true });
  const timer = new GpuPassTimer({
    device: env.device,
    maxPassesPerFrame: 4,
    readbackSlots: 2,
  });

  timer.beginFrame();
  timer.getPassTimestampWrites('backwards');
  timer.getPassTimestampWrites('equal');
  timer.getPassTimestampWrites('clock_wrap');
  timer.getPassTimestampWrites('normal');

  env.setFrameTimestamps([
    5_000_000n,
    4_000_000n, // end < start
    7_000_000n,
    7_000_000n, // end === start
    0xffffffffffff0000n,
    0x0000000000001000n, // uint64 wraparound (end < start)
    10_000_000n,
    12_250_000n, // +2.25 ms
  ]);

  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
  const result = await timer.collectCompletedReadback();

  assert.deepEqual(result, {
    totalGpuMs: 2.25,
    passes: [
      { name: 'backwards', durationMs: 0 },
      { name: 'equal', durationMs: 0 },
      { name: 'clock_wrap', durationMs: 0 },
      { name: 'normal', durationMs: 2.25 },
    ],
  });

  const stats = timer.getStats();
  assert.equal(stats.passStats.backwards.lastMs, 0);
  assert.equal(stats.passStats.clock_wrap.maxMs, 0);
  assert.equal(stats.passStats.normal.avgMs, 2.25);

  timer.destroy();
});

test('rolling getStats() evicts samples older than historySize for avgMs and maxMs', async () => {
  const env = createMockWebGpu({ supported: true });
  const timer = new GpuPassTimer({
    device: env.device,
    maxPassesPerFrame: 4,
    readbackSlots: 2,
    historySize: 3,
  });

  const frameDurationsNs = [
    10_000_000n, // 10 ms (spike that should age out after frame 4)
    2_000_000n,  // 2 ms
    4_000_000n,  // 4 ms
    6_000_000n,  // 6 ms -> rolling window is now [2, 4, 6]
  ];

  for (const durNs of frameDurationsNs) {
    timer.beginFrame();
    timer.getPassTimestampWrites('lighting');
    env.setFrameTimestamps([1_000_000n, 1_000_000n + durNs]);
    assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
    await timer.collectCompletedReadback();
  }

  const stats = timer.getStats();
  assert.equal(stats.resolvedFrames, 4);
  assert.equal(stats.lastFrameGpuMs, 6);
  assert.deepEqual(stats.passStats.lighting, {
    count: 4,
    lastMs: 6,
    avgMs: 4,
    maxMs: 6,
  });

  // Returned stats snapshot is defensive (mutating it does not affect internal state).
  stats.passStats.lighting.avgMs = 999;
  assert.equal(timer.getStats().passStats.lighting.avgMs, 4);

  timer.destroy();
});

test('idempotent destroy() releases GPUQuerySet and all buffers cleanly and turns subsequent calls into no-ops', async () => {
  const env = createMockWebGpu({ supported: true, deferMap: true });
  const timer = new GpuPassTimer({
    device: env.device,
    maxPassesPerFrame: 4,
    readbackSlots: 2,
  });

  // Start an in-flight readback before destroying to ensure no unhandled rejection occurs.
  timer.beginFrame();
  timer.getPassTimestampWrites('main');
  env.setFrameTimestamps([0n, 1_000_000n]);
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
  const inFlight = timer.collectCompletedReadback();

  timer.destroy();
  timer.destroy(); // Idempotent second call

  assert.equal(timer.isDestroyed, true);
  assert.equal(env.querySets[0].destroyed, true);
  assert.equal(env.querySets[0].destroyCalls, 1);

  // Resolve buffer + 2 readback buffers = 3 buffers, each destroyed exactly once.
  assert.equal(env.buffers.length, 3);
  for (const buf of env.buffers) {
    assert.equal(buf.destroyed, true);
    assert.equal(buf.destroyCalls, 1);
  }

  // Completing the deferred map after destroy resolves cleanly to null without throwing.
  env.pendingMaps.shift().resolve();
  assert.equal(await inFlight, null);

  // Subsequent API calls degrade cleanly to no-ops.
  timer.beginFrame();
  assert.equal(timer.getPassTimestampWrites('after_destroy'), undefined);
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), false);
  assert.equal(await timer.collectCompletedReadback(), null);
});
