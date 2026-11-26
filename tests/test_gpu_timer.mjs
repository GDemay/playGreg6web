import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const gpuTimerMod = require('../web/workers/gpu_timer.js');
const {
  DEFAULT_MAX_PASSES_PER_FRAME,
  DEFAULT_READBACK_SLOTS,
  DEFAULT_QUERY_RING_SLOTS,
  DEFAULT_HISTORY_SIZE,
  DEFAULT_FRAME_BUDGET_MS,
  DEFAULT_PASS_BUDGET_MS,
  DEFAULT_30FPS_BUDGET_MS,
  PASS_KINDS,
  BUFFER_USAGE,
  MAP_MODE,
  computePercentiles,
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
  assert.equal(globalThis.PlayGta6GpuTimer.computePercentiles, computePercentiles);
  assert.equal(DEFAULT_MAX_PASSES_PER_FRAME, 32);
  assert.equal(DEFAULT_READBACK_SLOTS, 3);
  assert.equal(DEFAULT_QUERY_RING_SLOTS, 1);
  assert.equal(DEFAULT_HISTORY_SIZE, 120);
  assert.equal(DEFAULT_FRAME_BUDGET_MS, 16.67);
  assert.equal(DEFAULT_PASS_BUDGET_MS, 33.333);
  assert.equal(DEFAULT_30FPS_BUDGET_MS, 33.333);
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
    droppedQueryFrames: 0,
    mapErrors: 0,
    lastFrameGpuMs: 0,
    p50PassMs: 0,
    p95PassMs: 0,
    maxPassMs: 0,
    passBudgetMs: 33.333,
    passesOverBudget: 0,
    querySlotRecycles: 0,
    querySlotWraps: 0,
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
    droppedQueryFrames: 0,
    mapErrors: 0,
    lastFrameGpuMs: 5.75,
    p50PassMs: 1.25,
    p95PassMs: 4.5,
    maxPassMs: 4.5,
    passBudgetMs: 33.333,
    passesOverBudget: 0,
    querySlotRecycles: 0,
    querySlotWraps: 1,
    passStats: {
      compute: {
        count: 1,
        lastMs: 1.25,
        avgMs: 1.25,
        p50Ms: 1.25,
        p95Ms: 1.25,
        maxMs: 1.25,
        passesOverBudget: 0,
      },
      main_render: {
        count: 1,
        lastMs: 4.5,
        avgMs: 4.5,
        p50Ms: 4.5,
        p95Ms: 4.5,
        maxMs: 4.5,
        passesOverBudget: 0,
      },
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
    p50Ms: 4,
    p95Ms: 6,
    maxMs: 6,
    passesOverBudget: 0,
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

test('computePercentiles handles 0, 1, and N samples deterministically without mutating input', () => {
  const zeroExpected = {
    count: 0,
    avgMs: 0,
    minMs: 0,
    p50Ms: 0,
    p95Ms: 0,
    p99Ms: 0,
    maxMs: 0,
  };
  assert.deepEqual(computePercentiles(null), zeroExpected);
  assert.deepEqual(computePercentiles(undefined), zeroExpected);
  assert.deepEqual(computePercentiles([]), zeroExpected);
  assert.deepEqual(computePercentiles(new Float64Array(0)), zeroExpected);
  assert.deepEqual(computePercentiles([Number.NaN, Number.POSITIVE_INFINITY]), zeroExpected);

  assert.deepEqual(computePercentiles([4.25]), {
    count: 1,
    avgMs: 4.25,
    minMs: 4.25,
    p50Ms: 4.25,
    p95Ms: 4.25,
    p99Ms: 4.25,
    maxMs: 4.25,
  });

  // Unsorted array must not be mutated.
  const unsorted = [50, 10, 40, 20, 30];
  const snapshot = unsorted.slice();
  assert.deepEqual(computePercentiles(unsorted), {
    count: 5,
    avgMs: 30,
    minMs: 10,
    p50Ms: 30,
    p95Ms: 50,
    p99Ms: 50,
    maxMs: 50,
  });
  assert.deepEqual(unsorted, snapshot);

  // 100 samples (1..100 in reverse order) => exact nearest-rank P50=50, P95=95, P99=99.
  const hundred = [];
  for (let i = 100; i >= 1; i -= 1) {
    hundred.push(i);
  }
  const hundredCopy = hundred.slice();
  assert.deepEqual(computePercentiles(hundred), {
    count: 100,
    avgMs: 50.5,
    minMs: 1,
    p50Ms: 50,
    p95Ms: 95,
    p99Ms: 99,
    maxMs: 100,
  });
  assert.deepEqual(hundred, hundredCopy);

  // TypedArray input is also preserved without mutation.
  const typed = new Float64Array([9.5, 1.5, 5.5]);
  assert.deepEqual(computePercentiles(typed), {
    count: 3,
    avgMs: 5.5,
    minMs: 1.5,
    p50Ms: 5.5,
    p95Ms: 9.5,
    p99Ms: 9.5,
    maxMs: 9.5,
  });
  assert.deepEqual(Array.from(typed), [9.5, 1.5, 5.5]);
});

test('rolling per-pass percentiles evict oldest samples beyond historySize', async () => {
  const env = createMockWebGpu({ supported: true });
  const timer = new GpuPassTimer({
    device: env.device,
    maxPassesPerFrame: 4,
    readbackSlots: 2,
    historySize: 5,
    frameBudgetMs: 16.67,
  });

  assert.equal(timer.historySize, 5);
  assert.equal(timer.frameBudgetMs, 16.67);
  assert.equal(timer.getPassPercentiles('gbuffer'), null);

  // Feed 7 frames: first two are huge spikes (30ms, 25ms) that must be evicted from the 5-sample window,
  // leaving [2, 6, 4, 10, 8] -> sorted [2, 4, 6, 8, 10].
  const durationsMs = [30, 25, 2, 6, 4, 10, 8];
  for (const ms of durationsMs) {
    timer.beginFrame();
    timer.getPassTimestampWrites('gbuffer');
    env.setFrameTimestamps([1_000_000n, 1_000_000n + BigInt(Math.round(ms * 1_000_000))]);
    assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
    await timer.collectCompletedReadback();
  }

  const p = timer.getPassPercentiles('gbuffer');
  assert.deepEqual(p, {
    count: 5,
    totalCount: 7,
    lastMs: 8,
    avgMs: 6,
    minMs: 2,
    p50Ms: 6,
    p95Ms: 10,
    p99Ms: 10,
    maxMs: 10,
    budgetExceedCount: 2,
  });

  // Defensive snapshot: mutating returned object does not corrupt internal state.
  p.p99Ms = 999;
  p.budgetExceedCount = 999;
  assert.equal(timer.getPassPercentiles('gbuffer').p99Ms, 10);
  assert.equal(timer.getPassPercentiles('gbuffer').budgetExceedCount, 2);

  timer.destroy();
});

test('budgetExceedCount tracks spikes strictly above custom frameBudgetMs and aggregates in getSummary()', async () => {
  const env = createMockWebGpu({ supported: true });
  const timer = new GpuPassTimer({
    device: env.device,
    maxPassesPerFrame: 4,
    readbackSlots: 2,
    historySize: 20,
    frameBudgetMs: 4.0,
  });

  assert.equal(timer.frameBudgetMs, 4.0);

  // Validate option rejection for invalid frameBudgetMs values.
  assert.throws(() => new GpuPassTimer({ device: env.device, frameBudgetMs: 0 }), RangeError);
  assert.throws(() => new GpuPassTimer({ device: env.device, frameBudgetMs: -2 }), RangeError);
  assert.throws(() => new GpuPassTimer({ device: env.device, frameBudgetMs: Number.NaN }), RangeError);

  // 4 frames with 2 passes each:
  // shadow:   [2.0, 4.0 (at boundary, not spike), 4.5 (spike), 6.0 (spike)] -> 2 spikes
  // lighting: [1.0, 5.0 (spike), 3.0, 2.0]                                  -> 1 spike
  const frames = [
    [2.0, 1.0],
    [4.0, 5.0],
    [4.5, 3.0],
    [6.0, 2.0],
  ];

  for (const [shadowMs, lightingMs] of frames) {
    timer.beginFrame();
    timer.getPassTimestampWrites('shadow');
    timer.getPassTimestampWrites('lighting');
    const sNs = BigInt(Math.round(shadowMs * 1_000_000));
    const lNs = BigInt(Math.round(lightingMs * 1_000_000));
    env.setFrameTimestamps([0n, sNs, 10_000_000n, 10_000_000n + lNs]);
    assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
    await timer.collectCompletedReadback();
  }

  const summary = timer.getSummary();
  assert.equal(summary.supported, true);
  assert.equal(summary.frameBudgetMs, 4.0);
  assert.equal(summary.historySize, 20);
  assert.equal(summary.resolvedFrames, 4);
  assert.equal(summary.skippedFrames, 0);
  assert.equal(summary.droppedPasses, 0);
  assert.equal(summary.lastFrameGpuMs, 8.0);
  assert.equal(summary.budgetExceedCount, 3);

  assert.deepEqual(summary.passes.shadow, {
    count: 4,
    totalCount: 4,
    lastMs: 6.0,
    avgMs: 4.125,
    minMs: 2.0,
    p50Ms: 4.0,
    p95Ms: 6.0,
    p99Ms: 6.0,
    maxMs: 6.0,
    budgetExceedCount: 2,
  });

  assert.deepEqual(summary.passes.lighting, {
    count: 4,
    totalCount: 4,
    lastMs: 2.0,
    avgMs: 2.75,
    minMs: 1.0,
    p50Ms: 2.0,
    p95Ms: 5.0,
    p99Ms: 5.0,
    maxMs: 5.0,
    budgetExceedCount: 1,
  });

  // Unknown pass returns null.
  assert.equal(timer.getPassPercentiles('nonexistent'), null);
  assert.equal(timer.getPassPercentiles(''), null);
  assert.equal(timer.getPassPercentiles(null), null);

  timer.destroy();
});

test('getPassPercentiles, getSummary, and reset degrade cleanly when timestamp-query is unsupported', () => {
  const env = createMockWebGpu({ supported: false });
  const timer = new GpuPassTimer({
    device: env.device,
    frameBudgetMs: 8.33,
    historySize: 30,
  });

  assert.equal(timer.supported, false);
  assert.equal(timer.getPassPercentiles('main'), null);
  assert.deepEqual(timer.getSummary(), {
    supported: false,
    frameBudgetMs: 8.33,
    historySize: 30,
    resolvedFrames: 0,
    skippedFrames: 0,
    droppedPasses: 0,
    lastFrameGpuMs: 0,
    budgetExceedCount: 0,
    passes: {},
  });

  timer.reset();
  assert.deepEqual(timer.getSummary(), {
    supported: false,
    frameBudgetMs: 8.33,
    historySize: 30,
    resolvedFrames: 0,
    skippedFrames: 0,
    droppedPasses: 0,
    lastFrameGpuMs: 0,
    budgetExceedCount: 0,
    passes: {},
  });

  timer.destroy();
});

test('reset() and destroy() clear rolling histories, spike counters, and discard stale pre-reset readbacks', async () => {
  const env = createMockWebGpu({ supported: true, deferMap: true });
  const timer = new GpuPassTimer({
    device: env.device,
    maxPassesPerFrame: 4,
    readbackSlots: 2,
    historySize: 10,
    frameBudgetMs: 5.0,
  });

  // Frame 1 (8ms -> spike)
  timer.beginFrame();
  timer.getPassTimestampWrites('post');
  env.setFrameTimestamps([0n, 8_000_000n]);
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
  const p1 = timer.collectCompletedReadback();
  env.pendingMaps.shift().resolve();
  await p1;

  assert.equal(timer.getPassPercentiles('post').budgetExceedCount, 1);
  assert.equal(timer.getSummary().budgetExceedCount, 1);

  // Frame 2 (12ms -> submitted before reset, mapped after reset)
  timer.beginFrame();
  timer.getPassTimestampWrites('post');
  env.setFrameTimestamps([0n, 12_000_000n]);
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
  const preResetReadback = timer.collectCompletedReadback();

  // Call reset() while Frame 2 is in-flight.
  timer.reset();

  assert.equal(timer.supported, true);
  assert.equal(timer.isDestroyed, false);
  assert.equal(timer.getPassPercentiles('post'), null);
  assert.deepEqual(timer.getSummary(), {
    supported: true,
    frameBudgetMs: 5.0,
    historySize: 10,
    resolvedFrames: 0,
    skippedFrames: 0,
    droppedPasses: 0,
    lastFrameGpuMs: 0,
    budgetExceedCount: 0,
    passes: {},
  });

  // Resolving the stale pre-reset readback must return null and not pollute post-reset telemetry.
  env.pendingMaps.shift().resolve();
  assert.equal(await preResetReadback, null);
  assert.equal(timer.getPassPercentiles('post'), null);
  assert.equal(timer.getSummary().budgetExceedCount, 0);

  // Timer remains usable after reset() and records fresh post-reset frames cleanly.
  timer.beginFrame();
  timer.getPassTimestampWrites('post');
  env.setFrameTimestamps([0n, 3_000_000n]); // 3ms (within 5ms budget)
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
  const p3 = timer.collectCompletedReadback();
  env.pendingMaps.shift().resolve();
  await p3;

  assert.deepEqual(timer.getPassPercentiles('post'), {
    count: 1,
    totalCount: 1,
    lastMs: 3.0,
    avgMs: 3.0,
    minMs: 3.0,
    p50Ms: 3.0,
    p95Ms: 3.0,
    p99Ms: 3.0,
    maxMs: 3.0,
    budgetExceedCount: 0,
  });

  // destroy() clears rolling history & spike counters too.
  timer.destroy();
  assert.equal(timer.isDestroyed, true);
  assert.equal(timer.getPassPercentiles('post'), null);
  assert.deepEqual(timer.getSummary(), {
    supported: true,
    frameBudgetMs: 5.0,
    historySize: 10,
    resolvedFrames: 0,
    skippedFrames: 0,
    droppedPasses: 0,
    lastFrameGpuMs: 0,
    budgetExceedCount: 0,
    passes: {},
  });
});

test('W209 GPU timestamp-query beginPass/endPass aggregates repeated labels and computes getTopPasses(limit) without hot-path descriptor reallocations', async () => {
  const env = createMockWebGpu({ supported: true });
  const timer = new GpuPassTimer({
    device: env.device,
    maxPassesPerFrame: 8,
    readbackSlots: 3,
    historySize: 8,
  });

  assert.equal(timer.mode, 'gpu');
  assert.deepEqual(timer.getTopPasses(), []);

  // Frame 1: bink2_yuv runs twice (2.0ms + 3.0ms = 5.0ms), ui_overlay (3.0ms), shadow (2.0ms) -> 10.0ms total
  timer.beginFrame();
  const ts0Frame1 = timer.beginPass('bink2_yuv');
  assert.equal(timer.endPass(), null);
  const ts1Frame1 = timer.beginPass('ui_overlay');
  assert.equal(timer.endPass(), null);
  const ts2Frame1 = timer.beginPass('bink2_yuv');
  assert.equal(timer.endPass(), null);
  const ts3Frame1 = timer.beginPass('shadow');
  assert.equal(timer.endPass(), null);

  assert.ok(Object.isFrozen(ts0Frame1));
  assert.deepEqual(ts0Frame1, {
    querySet: env.querySets[0],
    beginningOfPassWriteIndex: 0,
    endOfPassWriteIndex: 1,
  });
  assert.deepEqual(ts3Frame1, {
    querySet: env.querySets[0],
    beginningOfPassWriteIndex: 6,
    endOfPassWriteIndex: 7,
  });

  env.setFrameTimestamps([
    0n,
    2_000_000n, // bink2_yuv #1: 2.0 ms
    2_000_000n,
    5_000_000n, // ui_overlay:   3.0 ms
    5_000_000n,
    8_000_000n, // bink2_yuv #2: 3.0 ms
    8_000_000n,
    10_000_000n, // shadow:      2.0 ms
  ]);
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
  const res1 = await timer.collectCompletedReadback();
  assert.equal(res1.totalGpuMs, 10.0);

  // Frame 2: verify pre-allocated timestampWrites descriptor reuse (zero hot-path object allocations)
  // bink2_yuv (7.0ms), ui_overlay (1.0ms), shadow (2.0ms) -> 10.0ms total
  timer.beginFrame();
  const ts0Frame2 = timer.beginPass('bink2_yuv');
  assert.equal(ts0Frame2, ts0Frame1);
  assert.equal(timer.endPass(), null);
  const ts1Frame2 = timer.beginPass('ui_overlay');
  assert.equal(ts1Frame2, ts1Frame1);
  assert.equal(timer.endPass(), null);
  const ts2Frame2 = timer.beginPass('shadow');
  assert.equal(ts2Frame2, ts2Frame1);
  assert.equal(timer.endPass(), null);

  env.setFrameTimestamps([
    0n,
    7_000_000n, // bink2_yuv:  7.0 ms
    7_000_000n,
    8_000_000n, // ui_overlay: 1.0 ms
    8_000_000n,
    10_000_000n, // shadow:    2.0 ms
  ]);
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
  await timer.collectCompletedReadback();

  // Over 2 frames (20ms total):
  // bink2_yuv:  frame sums [5.0, 7.0] -> total 12.0ms, avg 6.0ms, max 7.0ms, share 60%
  // ui_overlay: frame sums [3.0, 1.0] -> total  4.0ms, avg 2.0ms, max 3.0ms, share 20%
  // shadow:     frame sums [2.0, 2.0] -> total  4.0ms, avg 2.0ms, max 2.0ms, share 20%
  // Tie between ui_overlay and shadow breaks by maxMs descending (3.0 > 2.0).
  const allTop = timer.getTopPasses();
  assert.deepEqual(allTop, [
    {
      label: 'bink2_yuv',
      name: 'bink2_yuv',
      count: 2,
      totalCount: 3,
      avgMs: 6.0,
      maxMs: 7.0,
      lastMs: 7.0,
      totalMs: 12.0,
      shareOfFramePct: 60.0,
      mode: 'gpu',
    },
    {
      label: 'ui_overlay',
      name: 'ui_overlay',
      count: 2,
      totalCount: 2,
      avgMs: 2.0,
      maxMs: 3.0,
      lastMs: 1.0,
      totalMs: 4.0,
      shareOfFramePct: 20.0,
      mode: 'gpu',
    },
    {
      label: 'shadow',
      name: 'shadow',
      count: 2,
      totalCount: 2,
      avgMs: 2.0,
      maxMs: 2.0,
      lastMs: 2.0,
      totalMs: 4.0,
      shareOfFramePct: 20.0,
      mode: 'gpu',
    },
  ]);

  // Share of frame percentages sum to 100%.
  const shareSum = allTop.reduce((acc, p) => acc + p.shareOfFramePct, 0);
  assert.ok(Math.abs(shareSum - 100) < 1e-9);

  // Limit handling: positive, 0, negative, NaN, Infinity.
  const top2 = timer.getTopPasses(2);
  assert.equal(top2.length, 2);
  assert.equal(top2[0].label, 'bink2_yuv');
  assert.equal(top2[1].label, 'ui_overlay');
  assert.equal(timer.getTopPasses(1).length, 1);
  assert.deepEqual(timer.getTopPasses(0), []);
  assert.deepEqual(timer.getTopPasses(-3), []);
  assert.deepEqual(timer.getTopPasses(Number.NaN), []);
  assert.equal(timer.getTopPasses(Number.POSITIVE_INFINITY).length, 3);

  // Defensive snapshot: mutating returned objects does not corrupt internal state.
  top2[0].avgMs = 999;
  assert.equal(timer.getTopPasses(1)[0].avgMs, 6.0);

  timer.destroy();
});

test('W209 CPU performance.now() fallback when timestamp-query is unavailable', () => {
  const { device } = createMockWebGpu({ supported: false });

  assert.throws(() => new GpuPassTimer({ device, now: 'not-a-fn' }), TypeError);

  let clockMs = 100.0;
  const timer = new GpuPassTimer({
    device,
    historySize: 5,
    frameBudgetMs: 10.0,
    now: () => clockMs,
  });

  assert.equal(timer.supported, false);
  assert.equal(timer.mode, 'cpu');
  assert.equal(timer.endPass(), null);

  // Frame 1 via explicit beginFrame / beginPass / endPass / endFrame
  timer.beginFrame();
  assert.equal(timer.beginPass('bink2_decode'), undefined);
  clockMs += 6.0;
  assert.equal(timer.endPass(), 6.0);

  assert.equal(timer.beginPass('ui_hud'), undefined);
  clockMs += 2.0;
  assert.equal(timer.endPass(), 2.0);

  assert.equal(timer.endFrame(), true);
  assert.equal(timer.endFrame(), false); // idempotent when no passes recorded since last finalize

  // Frame 2 via resolveFrame(encoder) (returns false since no GPU commands, still commits CPU frame)
  timer.beginFrame();
  assert.equal(timer.beginPass('bink2_decode'), undefined);
  clockMs += 10.0;
  assert.equal(timer.endPass(), 10.0);

  assert.equal(timer.beginPass('ui_hud'), undefined);
  clockMs += 2.0;
  assert.equal(timer.endPass(), 2.0);

  // Backwards clock delta clamps safely to 0
  assert.equal(timer.beginPass('clock_skew'), undefined);
  clockMs -= 5.0;
  assert.equal(timer.endPass(), 0);
  clockMs += 5.0;

  assert.equal(timer.resolveFrame(device.createCommandEncoder()), false);

  const stats = timer.getStats();
  assert.equal(stats.supported, false);
  assert.equal(stats.resolvedFrames, 2);
  assert.equal(stats.lastFrameGpuMs, 12.0);
  assert.equal(stats.passStats.bink2_decode.avgMs, 8.0);
  assert.equal(stats.passStats.bink2_decode.maxMs, 10.0);

  // Total rolling window across 2 frames = 8.0 + 12.0 = 20.0ms:
  // bink2_decode: 16.0ms (80%), ui_hud: 4.0ms (20%), clock_skew: 0ms (0%)
  const top = timer.getTopPasses();
  assert.deepEqual(top, [
    {
      label: 'bink2_decode',
      name: 'bink2_decode',
      count: 2,
      totalCount: 2,
      avgMs: 8.0,
      maxMs: 10.0,
      lastMs: 10.0,
      totalMs: 16.0,
      shareOfFramePct: 80.0,
      mode: 'cpu',
    },
    {
      label: 'ui_hud',
      name: 'ui_hud',
      count: 2,
      totalCount: 2,
      avgMs: 2.0,
      maxMs: 2.0,
      lastMs: 2.0,
      totalMs: 4.0,
      shareOfFramePct: 20.0,
      mode: 'cpu',
    },
    {
      label: 'clock_skew',
      name: 'clock_skew',
      count: 1,
      totalCount: 1,
      avgMs: 0,
      maxMs: 0,
      lastMs: 0,
      totalMs: 0,
      shareOfFramePct: 0,
      mode: 'cpu',
    },
  ]);

  // Direct beginPass/endPass without calling beginFrame/endFrame immediately reflects in getTopPasses()
  const directTimer = new GpuPassTimer({ device: null });
  assert.equal(directTimer.supported, false);
  assert.equal(directTimer.mode, 'cpu');
  directTimer.beginPass('bink2_direct');
  const d1 = directTimer.endPass();
  assert.equal(typeof d1, 'number');
  assert.ok(Number.isFinite(d1) && d1 >= 0);
  const directTop = directTimer.getTopPasses(1);
  assert.equal(directTop.length, 1);
  assert.equal(directTop[0].label, 'bink2_direct');
  assert.equal(directTop[0].mode, 'cpu');

  timer.destroy();
  assert.equal(timer.beginPass('after_destroy'), undefined);
  assert.equal(timer.endPass(), null);
  assert.equal(timer.endFrame(), false);
  assert.deepEqual(timer.getTopPasses(), []);
  directTimer.destroy();
});

test('W209 ring wrap-around evicts aged-out passes and keeps share-of-frame % exact at 100%', async () => {
  const env = createMockWebGpu({ supported: true });
  const timer = new GpuPassTimer({
    device: env.device,
    maxPassesPerFrame: 4,
    readbackSlots: 2,
    historySize: 3,
  });

  // Frame 1: one-time 40ms boot splash + 2ms ui (should age out completely after frame 4)
  timer.beginFrame();
  timer.beginPass('bink2_boot_splash');
  timer.endPass();
  timer.beginPass('ui');
  timer.endPass();
  env.setFrameTimestamps([0n, 40_000_000n, 40_000_000n, 42_000_000n]);
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
  await timer.collectCompletedReadback();

  // Frames 2, 3, 4: lighting [6, 4, 8] (sum 18), ui [2, 4, 0] (sum 6) -> window total = 24ms
  const steadyFrames = [
    [6_000_000n, 2_000_000n],
    [4_000_000n, 4_000_000n],
    [8_000_000n, 0n],
  ];
  for (const [lightingNs, uiNs] of steadyFrames) {
    timer.beginFrame();
    timer.getPassTimestampWrites('lighting');
    timer.getPassTimestampWrites('ui');
    env.setFrameTimestamps([0n, lightingNs, 10_000_000n, 10_000_000n + uiNs]);
    assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
    await timer.collectCompletedReadback();
  }

  const top = timer.getTopPasses();
  assert.equal(top.length, 2);
  assert.equal(top.find((p) => p.label === 'bink2_boot_splash'), undefined);

  assert.deepEqual(top, [
    {
      label: 'lighting',
      name: 'lighting',
      count: 3,
      totalCount: 3,
      avgMs: 6.0,
      maxMs: 8.0,
      lastMs: 8.0,
      totalMs: 18.0,
      shareOfFramePct: 75.0,
      mode: 'gpu',
    },
    {
      label: 'ui',
      name: 'ui',
      count: 3,
      totalCount: 4,
      avgMs: 2.0,
      maxMs: 4.0,
      lastMs: 0,
      totalMs: 6.0,
      shareOfFramePct: 25.0,
      mode: 'gpu',
    },
  ]);

  // Also verify CPU fallback ring wrap-around with an open frame in progress.
  let cpuNow = 0;
  const cpuTimer = new GpuPassTimer({
    device: null,
    historySize: 2,
    now: () => cpuNow,
  });

  // Frame 1: spike (50ms) -> will be evicted once frames 2 and 3 exist
  cpuTimer.beginFrame();
  cpuTimer.beginPass('boot_spike');
  cpuNow += 50;
  cpuTimer.endPass();
  cpuTimer.endFrame();

  // Frame 2: bink2 (6ms), ui (4ms)
  cpuTimer.beginFrame();
  cpuTimer.beginPass('bink2');
  cpuNow += 6;
  cpuTimer.endPass();
  cpuTimer.beginPass('ui');
  cpuNow += 4;
  cpuTimer.endPass();
  cpuTimer.endFrame();

  // Frame 3 (open, not yet endFrame'd): bink2 (8ms), ui (2ms)
  cpuTimer.beginFrame();
  cpuTimer.beginPass('bink2');
  cpuNow += 8;
  cpuTimer.endPass();
  cpuTimer.beginPass('ui');
  cpuNow += 2;
  cpuTimer.endPass();

  const cpuTop = cpuTimer.getTopPasses();
  assert.equal(cpuTop.length, 2);
  assert.equal(cpuTop[0].label, 'bink2');
  assert.equal(cpuTop[0].avgMs, 7.0);
  assert.equal(cpuTop[0].maxMs, 8.0);
  assert.equal(cpuTop[0].shareOfFramePct, 70.0);
  assert.equal(cpuTop[1].label, 'ui');
  assert.equal(cpuTop[1].avgMs, 3.0);
  assert.equal(cpuTop[1].maxMs, 4.0);
  assert.equal(cpuTop[1].shareOfFramePct, 30.0);

  timer.destroy();
  cpuTimer.destroy();
});

test('W209 reset() clears top-pass rankings, frame totals, and open CPU passes cleanly', async () => {
  // GPU mode reset()
  const env = createMockWebGpu({ supported: true });
  const gpuTimer = new GpuPassTimer({
    device: env.device,
    maxPassesPerFrame: 4,
    readbackSlots: 2,
    historySize: 4,
  });

  gpuTimer.beginFrame();
  gpuTimer.beginPass('alpha');
  gpuTimer.endPass();
  gpuTimer.beginPass('beta');
  gpuTimer.endPass();
  env.setFrameTimestamps([0n, 3_000_000n, 3_000_000n, 4_000_000n]);
  assert.equal(gpuTimer.resolveFrame(env.device.createCommandEncoder()), true);
  await gpuTimer.collectCompletedReadback();
  assert.equal(gpuTimer.getTopPasses().length, 2);

  gpuTimer.reset();
  assert.deepEqual(gpuTimer.getTopPasses(), []);

  // Fresh post-reset frame computes 100% share cleanly without pre-reset frame total leakage.
  gpuTimer.beginFrame();
  gpuTimer.beginPass('gamma');
  gpuTimer.endPass();
  env.setFrameTimestamps([0n, 2_500_000n]);
  assert.equal(gpuTimer.resolveFrame(env.device.createCommandEncoder()), true);
  await gpuTimer.collectCompletedReadback();
  assert.deepEqual(gpuTimer.getTopPasses(), [
    {
      label: 'gamma',
      name: 'gamma',
      count: 1,
      totalCount: 1,
      avgMs: 2.5,
      maxMs: 2.5,
      lastMs: 2.5,
      totalMs: 2.5,
      shareOfFramePct: 100.0,
      mode: 'gpu',
    },
  ]);

  // CPU fallback mode reset() also clears in-progress open passes and zero-duration tie-breaking.
  let t = 0;
  const cpuTimer = new GpuPassTimer({ device: null, now: () => t });
  cpuTimer.beginFrame();
  cpuTimer.beginPass('in_progress');
  t += 5;
  cpuTimer.reset();
  // Calling endPass() after reset must be a safe no-op (open pass stack was cleared).
  assert.equal(cpuTimer.endPass(), null);
  assert.deepEqual(cpuTimer.getTopPasses(), []);

  // All-zero passes break ties lexicographically and report 0% share (never NaN).
  cpuTimer.beginFrame();
  cpuTimer.beginPass('z_pass');
  cpuTimer.endPass();
  cpuTimer.beginPass('a_pass');
  cpuTimer.endPass();
  cpuTimer.endFrame();

  const zeroTop = cpuTimer.getTopPasses();
  assert.deepEqual(
    zeroTop.map((p) => ({ label: p.label, shareOfFramePct: p.shareOfFramePct })),
    [
      { label: 'a_pass', shareOfFramePct: 0 },
      { label: 'z_pass', shareOfFramePct: 0 },
    ]
  );

  gpuTimer.destroy();
  cpuTimer.destroy();
});

test('W277: guards mapAsync, getMappedRange, unmap, and resolveFrame against device-loss / abort / torn-down buffer errors without unhandled rejections or dangling mapped buffers', async () => {
  // 1) mapAsync rejection (AbortError / device loss), getMappedRange throw, and unmap throw
  const env = createMockWebGpu({ supported: true, deferMap: true });
  const timer = new GpuPassTimer({
    device: env.device,
    maxPassesPerFrame: 4,
    readbackSlots: 2,
    historySize: 8,
  });

  // Frame 1: mapAsync rejects with AbortError (simulated device loss / abort)
  timer.beginFrame();
  timer.getPassTimestampWrites('shadow');
  env.setFrameTimestamps([0n, 2_000_000n]);
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
  const pAbort = timer.collectCompletedReadback();
  assert.equal(env.pendingMaps.length, 1);
  env.pendingMaps.shift().reject(new DOMException('Device was lost', 'AbortError'));
  assert.equal(await pAbort, null);

  let stats = timer.getStats();
  assert.equal(stats.mapErrors, 1);
  assert.equal(stats.droppedQueryFrames, 1);
  assert.equal(stats.resolvedFrames, 0);
  assert.equal(env.buffers[1].mapState, 'unmapped');

  // Frame 2: getMappedRange throws after mapAsync resolves -> buffer MUST be unmapped (no dangling mapped buffer)
  const readbackBuf0 = env.buffers[1];
  const origGetMappedRange = readbackBuf0.getMappedRange.bind(readbackBuf0);
  const unmapCallsBefore = readbackBuf0.unmapCalls;
  readbackBuf0.getMappedRange = () => {
    throw new Error('Buffer torn down during getMappedRange');
  };

  timer.beginFrame();
  timer.getPassTimestampWrites('gbuffer');
  env.setFrameTimestamps([0n, 3_000_000n]);
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
  const pRangeErr = timer.collectCompletedReadback();
  env.pendingMaps.shift().resolve();
  assert.equal(await pRangeErr, null);

  assert.equal(readbackBuf0.mapState, 'unmapped');
  assert.equal(readbackBuf0.unmapCalls, unmapCallsBefore + 1);
  stats = timer.getStats();
  assert.equal(stats.mapErrors, 2);
  assert.equal(stats.droppedQueryFrames, 2);
  assert.equal(stats.resolvedFrames, 0);
  readbackBuf0.getMappedRange = origGetMappedRange;

  // Frame 3: unmap() throws (torn-down buffer / device loss during unmap) -> no unhandled rejection, no double unmap
  const origUnmap = readbackBuf0.unmap.bind(readbackBuf0);
  let throwingUnmapCalls = 0;
  readbackBuf0.unmap = function () {
    throwingUnmapCalls += 1;
    this.mapState = 'unmapped';
    throw new Error('Device lost during unmap');
  };

  timer.beginFrame();
  timer.getPassTimestampWrites('lighting');
  env.setFrameTimestamps([0n, 4_000_000n]);
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
  const pUnmapErr = timer.collectCompletedReadback();
  env.pendingMaps.shift().resolve();
  assert.equal(await pUnmapErr, null);

  assert.equal(throwingUnmapCalls, 1);
  stats = timer.getStats();
  assert.equal(stats.mapErrors, 3);
  assert.equal(stats.droppedQueryFrames, 3);
  assert.equal(stats.resolvedFrames, 0);
  readbackBuf0.unmap = origUnmap;

  // Frame 4: externally destroyed readback slot is skipped cleanly while healthy slot still resolves
  readbackBuf0.destroy();
  timer.beginFrame();
  timer.getPassTimestampWrites('post');
  env.setFrameTimestamps([0n, 5_000_000n]);
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
  const pHealthy = timer.collectCompletedReadback();
  env.pendingMaps.shift().resolve();
  const resHealthy = await pHealthy;
  assert.deepEqual(resHealthy, {
    totalGpuMs: 5.0,
    passes: [{ name: 'post', durationMs: 5.0 }],
  });
  assert.equal(timer.getStats().resolvedFrames, 1);
  assert.equal(timer.getStats().mapErrors, 3);
  assert.equal(timer.getStats().droppedQueryFrames, 3);

  // Frame 5: resolveQuerySet throwing on encoder is caught and increments droppedQueryFrames
  timer.beginFrame();
  timer.getPassTimestampWrites('broken_encode');
  const throwingEncoder = {
    resolveQuerySet() {
      throw new Error('GPUQuerySet destroyed');
    },
    copyBufferToBuffer() {},
  };
  assert.equal(timer.resolveFrame(throwingEncoder), false);
  assert.equal(timer.getStats().droppedQueryFrames, 4);
  assert.equal(timer.getStats().mapErrors, 3);

  timer.destroy();

  // 2) WebGPU device.lost promise guards subsequent resolveFrame calls cleanly
  let triggerDeviceLost;
  const lostEnv = createMockWebGpu({ supported: true });
  lostEnv.device.lost = new Promise((resolve) => {
    triggerDeviceLost = resolve;
  });
  const lostTimer = new GpuPassTimer({
    device: lostEnv.device,
    maxPassesPerFrame: 4,
    readbackSlots: 2,
  });

  triggerDeviceLost({ reason: 'destroyed', message: 'Simulated GPU device loss' });
  await Promise.resolve();

  lostTimer.beginFrame();
  lostTimer.getPassTimestampWrites('after_loss');
  assert.equal(lostTimer.resolveFrame(lostEnv.device.createCommandEncoder()), false);
  assert.equal(lostEnv.resolveCalls.length, 0);
  assert.equal(lostTimer.getStats().droppedQueryFrames, 1);
  lostTimer.destroy();
});

test('W277: bounded rolling pass duration window computes p50PassMs and p95PassMs over multi-thousand-frame sessions with clean resetStats()', async () => {
  const env = createMockWebGpu({ supported: true });
  const timer = new GpuPassTimer({
    device: env.device,
    maxPassesPerFrame: 4,
    readbackSlots: 2,
    historySize: 100,
  });

  // Simulate 2,500 frames: first 2,400 frames have huge 50ms spikes that must evict cleanly,
  // final 100 frames have pass durations 1ms..100ms -> exact nearest-rank p50=50ms, p95=95ms.
  for (let frame = 0; frame < 2500; frame += 1) {
    const durationMs = frame < 2400 ? 50 : frame - 2400 + 1;
    timer.beginFrame();
    timer.getPassTimestampWrites('main_render');
    env.setFrameTimestamps([0n, BigInt(durationMs * 1_000_000)]);
    assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
    await timer.collectCompletedReadback();
  }

  const stats = timer.getStats();
  assert.equal(stats.resolvedFrames, 2500);
  assert.equal(stats.droppedQueryFrames, 0);
  assert.equal(stats.mapErrors, 0);
  assert.equal(stats.p50PassMs, 50);
  assert.equal(stats.p95PassMs, 95);

  // resetStats() clears rolling window, counters, and errors while keeping GPU resources usable.
  const afterReset = timer.resetStats();
  assert.deepEqual(afterReset, {
    supported: true,
    resolvedFrames: 0,
    skippedFrames: 0,
    droppedPasses: 0,
    droppedQueryFrames: 0,
    mapErrors: 0,
    lastFrameGpuMs: 0,
    p50PassMs: 0,
    p95PassMs: 0,
    maxPassMs: 0,
    passBudgetMs: 33.333,
    passesOverBudget: 0,
    querySlotRecycles: 0,
    querySlotWraps: 0,
    passStats: {},
  });
  assert.equal(timer.isDestroyed, false);

  // Fresh post-reset frames compute new p50PassMs / p95PassMs without leakage from prior 2,500 frames.
  for (const ms of [2, 4, 6, 8]) {
    timer.beginFrame();
    timer.getPassTimestampWrites('post_reset');
    env.setFrameTimestamps([0n, BigInt(ms * 1_000_000)]);
    assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
    await timer.collectCompletedReadback();
  }
  const freshStats = timer.getStats();
  assert.equal(freshStats.resolvedFrames, 4);
  assert.equal(freshStats.p50PassMs, 4);
  assert.equal(freshStats.p95PassMs, 8);

  timer.destroy();

  // In-flight mapAsync during resetStats() must still unmap cleanly when it resolves.
  const deferEnv = createMockWebGpu({ supported: true, deferMap: true });
  const deferTimer = new GpuPassTimer({
    device: deferEnv.device,
    maxPassesPerFrame: 2,
    readbackSlots: 2,
  });
  deferTimer.beginFrame();
  deferTimer.getPassTimestampWrites('stale_pass');
  deferEnv.setFrameTimestamps([0n, 9_000_000n]);
  assert.equal(deferTimer.resolveFrame(deferEnv.device.createCommandEncoder()), true);
  const inFlight = deferTimer.collectCompletedReadback();
  deferTimer.resetStats();

  deferEnv.pendingMaps.shift().resolve();
  assert.equal(await inFlight, null);
  assert.equal(deferEnv.buffers[1].mapState, 'unmapped');
  assert.equal(deferEnv.buffers[1].unmapCalls, 1);
  assert.equal(deferTimer.getStats().p50PassMs, 0);
  assert.equal(deferTimer.getStats().mapErrors, 0);
  deferTimer.destroy();
});

test('W324: rolling 30 FPS per-pass histograms compute p50Ms/p95Ms/maxMs and passesOverBudget against 33.333ms default budget', async () => {
  const env = createMockWebGpu({ supported: true });

  // Validate option rejection for invalid passBudgetMs / queryRingSlots values.
  assert.throws(() => new GpuPassTimer({ device: env.device, passBudgetMs: 0 }), RangeError);
  assert.throws(() => new GpuPassTimer({ device: env.device, passBudgetMs: -5 }), RangeError);
  assert.throws(
    () => new GpuPassTimer({ device: env.device, passBudgetMs: Number.NaN }),
    RangeError
  );
  assert.throws(() => new GpuPassTimer({ device: env.device, queryRingSlots: 0 }), RangeError);
  assert.throws(() => new GpuPassTimer({ device: env.device, queryRingSlots: -2 }), RangeError);
  assert.throws(() => new GpuPassTimer({ device: env.device, queryRingSlots: 1.5 }), RangeError);

  const timer = new GpuPassTimer({
    device: env.device,
    maxPassesPerFrame: 4,
    readbackSlots: 2,
    historySize: 20,
  });

  assert.equal(timer.passBudgetMs, 33.333);
  assert.equal(timer.queryRingSlots, 1);
  assert.deepEqual(timer.getPassStats(), {});
  assert.equal(timer.getPassStats('gbuffer'), null);
  assert.equal(timer.getPassStats(''), null);
  assert.equal(timer.getPassStats(null), null);

  // Feed 25 frames for 'gbuffer' and 'shadow':
  // - First 5 frames: huge spikes (50ms gbuffer, 40ms shadow) that MUST age out of the 20-sample rolling window
  //   for p50Ms/p95Ms/maxMs, while passesOverBudget retains the 5 over-budget events.
  // - Next 20 frames for gbuffer: 1..18ms, 33.333ms (exact boundary, NOT over budget), 33.334ms (over budget #6)
  //   -> sorted window of 20 samples: [1,2,...,18, 33.333, 33.334]
  //      p50 (ceil(0.5*20)-1 = idx 9) = 10ms, p95 (ceil(0.95*20)-1 = idx 18) = 33.333ms, max = 33.334ms
  //      passesOverBudget = 5 (early spikes) + 1 (33.334ms) = 6
  // - Next 20 frames for shadow: 2..21ms (all <= 33.333ms)
  //   -> sorted window of 20 samples: [2..21]
  //      p50 (idx 9) = 11ms, p95 (idx 18) = 20ms, max = 21ms
  //      passesOverBudget = 5 (early spikes only)
  for (let frame = 0; frame < 25; frame += 1) {
    let gbufMs;
    let shadowMs;
    if (frame < 5) {
      gbufMs = 50;
      shadowMs = 40;
    } else {
      const idx = frame - 5; // 0..19
      if (idx < 18) {
        gbufMs = idx + 1;
      } else if (idx === 18) {
        gbufMs = 33.333; // exact 30 FPS boundary -> NOT strictly over budget
      } else {
        gbufMs = 33.334; // strictly over 33.333 ms budget
      }
      shadowMs = idx + 2;
    }

    timer.beginFrame();
    timer.getPassTimestampWrites('gbuffer');
    timer.getPassTimestampWrites('shadow');
    const gNs = BigInt(Math.round(gbufMs * 1_000_000));
    const sNs = BigInt(Math.round(shadowMs * 1_000_000));
    env.setFrameTimestamps([0n, gNs, 100_000_000n, 100_000_000n + sNs]);
    assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
    await timer.collectCompletedReadback();
  }

  const gbufStats = timer.getPassStats('gbuffer');
  assert.equal(gbufStats.count, 25);
  assert.equal(gbufStats.lastMs, 33.334);
  assert.equal(gbufStats.p50Ms, 10);
  assert.equal(gbufStats.p95Ms, 33.333);
  assert.equal(gbufStats.maxMs, 33.334);
  assert.equal(gbufStats.passesOverBudget, 6);

  const shadowStats = timer.getPassStats('shadow');
  assert.equal(shadowStats.count, 25);
  assert.equal(shadowStats.lastMs, 21);
  assert.equal(shadowStats.p50Ms, 11);
  assert.equal(shadowStats.p95Ms, 20);
  assert.equal(shadowStats.maxMs, 21);
  assert.equal(shadowStats.passesOverBudget, 5);

  // getPassStats() with no argument matches getStats().passStats and is defensive.
  const allPassStats = timer.getPassStats();
  const stats = timer.getStats();
  assert.deepEqual(allPassStats, stats.passStats);
  assert.equal(stats.passBudgetMs, 33.333);
  assert.equal(stats.passesOverBudget, 11);
  assert.equal(stats.maxPassMs, 33.334);

  allPassStats.gbuffer.p95Ms = 9999;
  gbufStats.passesOverBudget = 9999;
  assert.equal(timer.getPassStats('gbuffer').p95Ms, 33.333);
  assert.equal(timer.getPassStats('gbuffer').passesOverBudget, 6);

  // Custom passBudgetMs and CPU fallback mode also track passesOverBudget & rolling p50/p95/max.
  let nowMs = 0;
  const cpuTimer = new GpuPassTimer({
    device: null,
    historySize: 4,
    passBudgetMs: 10.0,
    now: () => nowMs,
  });
  assert.equal(cpuTimer.passBudgetMs, 10.0);
  for (const dur of [5, 10, 12, 15, 8]) {
    cpuTimer.beginFrame();
    cpuTimer.beginPass('cpu_pass');
    nowMs += dur;
    cpuTimer.endPass();
    cpuTimer.endFrame();
  }
  // Rolling window of last 4 samples: [10, 12, 15, 8] -> sorted [8, 10, 12, 15]
  // p50 = 10, p95 = 15, max = 15; samples strictly > 10 are [12, 15] => passesOverBudget = 2
  assert.deepEqual(cpuTimer.getPassStats('cpu_pass'), {
    count: 5,
    lastMs: 8,
    avgMs: 11.25,
    p50Ms: 10,
    p95Ms: 15,
    maxMs: 15,
    passesOverBudget: 2,
  });
  assert.equal(cpuTimer.getStats().passesOverBudget, 2);

  timer.destroy();
  cpuTimer.destroy();
});

test('W324: query-slot ring wrap-around recycles pre-allocated descriptors with zero GC churn across pipelined frames', async () => {
  const env = createMockWebGpu({ supported: true, deferMap: true });
  const timer = new GpuPassTimer({
    device: env.device,
    maxPassesPerFrame: 4,
    readbackSlots: 3,
    queryRingSlots: 3,
    historySize: 30,
  });

  assert.equal(timer.queryRingSlots, 3);
  // GPUQuerySet holds queryRingSlots * maxPassesPerFrame * 2 = 3 * 4 * 2 = 24 timestamp queries.
  assert.equal(env.querySets.length, 1);
  assert.equal(env.querySets[0].desc.count, 24);
  // Resolve buffer + 3 readback buffers each hold 1 frame slice = 4 * 2 * 8 = 64 bytes.
  assert.equal(env.buffers.length, 4);
  for (const buf of env.buffers) {
    assert.equal(buf.size, 64);
  }

  const firstSlicePass0 = [];
  const firstSlicePass1 = [];

  // Run 9 frames (3 full query-ring wrap cycles) with 2 pipelined frames in flight at a time.
  const inFlight = [];
  for (let frame = 0; frame < 9; frame += 1) {
    timer.beginFrame();
    const ts0 = timer.beginPass('gbuffer');
    assert.equal(timer.endPass(), null);
    const ts1 = timer.beginPass('lighting');
    assert.equal(timer.endPass(), null);

    const expectedSlice = frame % 3;
    const expectedQueryBase = expectedSlice * 8;
    assert.ok(Object.isFrozen(ts0));
    assert.ok(Object.isFrozen(ts1));
    assert.equal(ts0.beginningOfPassWriteIndex, expectedQueryBase);
    assert.equal(ts0.endOfPassWriteIndex, expectedQueryBase + 1);
    assert.equal(ts1.beginningOfPassWriteIndex, expectedQueryBase + 2);
    assert.equal(ts1.endOfPassWriteIndex, expectedQueryBase + 3);

    // Verify exact descriptor object identity reuse after ring wrap-around (zero per-pass allocation).
    if (frame < 3) {
      firstSlicePass0[expectedSlice] = ts0;
      firstSlicePass1[expectedSlice] = ts1;
    } else {
      assert.equal(ts0, firstSlicePass0[expectedSlice]);
      assert.equal(ts1, firstSlicePass1[expectedSlice]);
    }

    const gMs = (frame + 1) * 2;
    const lMs = (frame + 1) * 3;
    env.setFrameTimestamps([
      0n,
      BigInt(gMs * 1_000_000),
      100_000_000n,
      100_000_000n + BigInt(lMs * 1_000_000),
    ]);

    assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
    assert.equal(env.resolveCalls[frame].firstQuery, expectedQueryBase);
    assert.equal(env.resolveCalls[frame].queryCount, 4);

    // Keep 2 frames pipelined, draining the oldest once 2 are queued.
    if (frame >= 1) {
      const p = timer.collectCompletedReadback();
      inFlight.push(p);
      env.pendingMaps.shift().resolve();
      const resolved = await p;
      assert.ok(resolved);
      assert.equal(resolved.passes[0].name, 'gbuffer');
      assert.equal(resolved.passes[1].name, 'lighting');
    }
  }

  // Drain the final remaining frame.
  const finalReadback = timer.collectCompletedReadback();
  env.pendingMaps.shift().resolve();
  const finalResult = await finalReadback;
  assert.deepEqual(finalResult, {
    totalGpuMs: 45,
    passes: [
      { name: 'gbuffer', durationMs: 18 },
      { name: 'lighting', durationMs: 27 },
    ],
  });

  // 9 frames across 3 ring slices => 3 full wraps, 6 slice recycles, 0 extra GPUQuerySet allocations.
  assert.equal(env.querySets.length, 1);
  const stats = timer.getStats();
  assert.equal(stats.resolvedFrames, 9);
  assert.equal(stats.skippedFrames, 0);
  assert.equal(stats.querySlotWraps, 3);
  assert.equal(stats.querySlotRecycles, 6);
  assert.equal(stats.passStats.gbuffer.p50Ms, 10);
  assert.equal(stats.passStats.gbuffer.p95Ms, 18);
  assert.equal(stats.passStats.gbuffer.maxMs, 18);
  assert.equal(stats.passStats.lighting.p50Ms, 15);
  assert.equal(stats.passStats.lighting.p95Ms, 27);
  assert.equal(stats.passStats.lighting.maxMs, 27);

  // Skipped resolveFrame (all readback slots busy) must NOT advance the query-slot ring slice.
  for (let i = 0; i < 3; i += 1) {
    timer.beginFrame();
    timer.getPassTimestampWrites('busy');
    env.setFrameTimestamps([0n, 1_000_000n]);
    assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
  }
  const wrapsBeforeSkip = timer.getStats().querySlotWraps;
  const recyclesBeforeSkip = timer.getStats().querySlotRecycles;
  timer.beginFrame();
  const tsBeforeSkip = timer.getPassTimestampWrites('skipped');
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), false);
  assert.equal(timer.getStats().querySlotWraps, wrapsBeforeSkip);
  assert.equal(timer.getStats().querySlotRecycles, recyclesBeforeSkip);

  // After freeing one readback slot, next frame uses the exact same slice that was skipped.
  const drainOne = timer.collectCompletedReadback();
  env.pendingMaps.shift().resolve();
  await drainOne;
  timer.beginFrame();
  const tsAfterSkip = timer.getPassTimestampWrites('reused_slice');
  assert.equal(tsAfterSkip, tsBeforeSkip);

  // Drain remaining 2 pending slots before teardown.
  for (let i = 0; i < 2; i += 1) {
    const p = timer.collectCompletedReadback();
    env.pendingMaps.shift().resolve();
    await p;
  }

  timer.destroy();
});

test('W324: preserves W277 device-loss / mapAsync guards and clears histogram + ring counters cleanly on resetStats()', async () => {
  let triggerLost;
  const env = createMockWebGpu({ supported: true, deferMap: true });
  env.device.lost = new Promise((resolve) => {
    triggerLost = resolve;
  });

  const timer = new GpuPassTimer({
    device: env.device,
    maxPassesPerFrame: 4,
    readbackSlots: 2,
    queryRingSlots: 2,
    historySize: 16,
  });

  // Frame 1 (slice 0): over 30 FPS budget (42ms)
  timer.beginFrame();
  const tsSlice0 = timer.beginPass('volumetrics');
  timer.endPass();
  assert.equal(tsSlice0.beginningOfPassWriteIndex, 0);
  env.setFrameTimestamps([0n, 42_000_000n]);
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
  const p1 = timer.collectCompletedReadback();
  env.pendingMaps.shift().resolve();
  await p1;

  // Frame 2 (slice 1): mapAsync rejects (W277 guard) -> must NOT corrupt passStats or leak mapped state
  timer.beginFrame();
  const tsSlice1 = timer.beginPass('volumetrics');
  timer.endPass();
  assert.equal(tsSlice1.beginningOfPassWriteIndex, 8);
  env.setFrameTimestamps([0n, 50_000_000n]);
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
  const p2 = timer.collectCompletedReadback();
  env.pendingMaps.shift().reject(new DOMException('GPU aborted', 'AbortError'));
  assert.equal(await p2, null);

  let stats = timer.getStats();
  assert.equal(stats.resolvedFrames, 1);
  assert.equal(stats.mapErrors, 1);
  assert.equal(stats.droppedQueryFrames, 1);
  assert.equal(stats.passesOverBudget, 1);
  assert.equal(stats.querySlotWraps, 1);
  assert.equal(timer.getPassStats('volumetrics').passesOverBudget, 1);
  assert.equal(timer.getPassStats('volumetrics').maxMs, 42);

  // resetStats() clears per-pass histograms, passesOverBudget, query-slot ring counters, and slice cursor.
  const resetSnapshot = timer.resetStats();
  assert.deepEqual(resetSnapshot, {
    supported: true,
    resolvedFrames: 0,
    skippedFrames: 0,
    droppedPasses: 0,
    droppedQueryFrames: 0,
    mapErrors: 0,
    lastFrameGpuMs: 0,
    p50PassMs: 0,
    p95PassMs: 0,
    maxPassMs: 0,
    passBudgetMs: 33.333,
    passesOverBudget: 0,
    querySlotRecycles: 0,
    querySlotWraps: 0,
    passStats: {},
  });
  assert.deepEqual(timer.getPassStats(), {});
  assert.equal(timer.getPassStats('volumetrics'), null);

  // Next frame after resetStats() restarts cleanly at slice 0 (beginningOfPassWriteIndex === 0).
  timer.beginFrame();
  const tsAfterReset = timer.beginPass('volumetrics');
  timer.endPass();
  assert.equal(tsAfterReset, tsSlice0);
  env.setFrameTimestamps([0n, 12_000_000n]);
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
  const p3 = timer.collectCompletedReadback();
  env.pendingMaps.shift().resolve();
  await p3;

  assert.deepEqual(timer.getPassStats('volumetrics'), {
    count: 1,
    lastMs: 12,
    avgMs: 12,
    p50Ms: 12,
    p95Ms: 12,
    maxMs: 12,
    passesOverBudget: 0,
  });

  // Trigger device.lost -> subsequent resolveFrame must be safely guarded and drop cleanly.
  triggerLost({ reason: 'destroyed', message: 'Simulated W324 device loss' });
  await Promise.resolve();

  timer.beginFrame();
  timer.beginPass('after_loss');
  timer.endPass();
  const resolveCountBeforeLoss = env.resolveCalls.length;
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), false);
  assert.equal(env.resolveCalls.length, resolveCountBeforeLoss);
  assert.equal(timer.getStats().droppedQueryFrames, 1);
  assert.equal(timer.getPassStats('after_loss'), null);

  timer.destroy();
});

// ---------------------------------------------------------------------------
// W380 (#1763): High-Pass-Count 3D Category Tracking, Zero-Alloc `out` Reuse,
//               CPU Budget Overflow & Pending-Slot Device-Loss Hardening
// ---------------------------------------------------------------------------

test('W380 tracks render/compute pass categories and supports zero-alloc out object reuse on getFrameSummary(out), getStats(out), and getCategoryStats(out)', async () => {
  assert.deepEqual(PASS_KINDS, ['unknown', 'render', 'compute']);

  const env = createMockWebGpu({ supported: true });
  const timer = new GpuPassTimer({
    device: env.device,
    maxPassesPerFrame: 16,
    readbackSlots: 3,
    historySize: 8,
  });

  // High-pass-count 3D frame: 2 compute passes + 5 render passes + 1 unlabeled pass = 8 passes.
  const passes = [
    ['compute_cull', 'compute', 1.0],
    ['compute_mesh', 'compute', 1.5],
    ['shadow_csm', 'render', 2.0],
    ['gbuffer', 'render', 3.0],
    ['lighting', 'render', 2.5],
    ['post_fx', 'render', 1.0],
    ['ui_blit', 'render', 0.5],
    [undefined, undefined, 0.25], // unlabeled -> uses pre-allocated 'pass_7'
  ];

  const reusableSummary = {};
  const reusableStats = {};
  const reusableCategories = {};

  for (let frame = 0; frame < 3; frame += 1) {
    timer.beginFrame();
    const timestamps = [];
    let cursorNs = 10_000_000n;
    for (const [label, kind, durMs] of passes) {
      const ts = timer.beginPass(label, kind);
      assert.ok(ts);
      assert.equal(timer.endPass(), null);
      timestamps.push(cursorNs);
      cursorNs += BigInt(Math.round(durMs * 1_000_000));
      timestamps.push(cursorNs);
    }
    env.setFrameTimestamps(timestamps);
    assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);
    const res = await timer.collectCompletedReadback();
    assert.equal(res.passes.length, 8);
    assert.equal(res.passes[7].name, 'pass_7');
    assert.ok(Math.abs(res.totalGpuMs - 11.75) < 1e-9);

    const summaryRef = timer.getFrameSummary(reusableSummary);
    assert.strictEqual(summaryRef, reusableSummary);
    assert.equal(summaryRef.mode, 'gpu');
    assert.equal(summaryRef.supported, true);
    assert.equal(summaryRef.deviceLost, false);
    assert.equal(summaryRef.lastFramePassCount, 8);
    assert.ok(Math.abs(summaryRef.lastFrameRenderMs - 9.0) < 1e-9);
    assert.ok(Math.abs(summaryRef.lastFrameComputeMs - 2.5) < 1e-9);
    assert.ok(Math.abs(summaryRef.lastFrameOtherMs - 0.25) < 1e-9);

    const prevPassStatsObj = reusableStats.passStats;
    const prevGbufSlot = prevPassStatsObj ? prevPassStatsObj.gbuffer : null;
    const statsRef = timer.getStats(reusableStats);
    assert.strictEqual(statsRef, reusableStats);
    if (frame > 0) {
      // Steady-state out reuse preserves exact nested passStats & per-pass object identities!
      assert.strictEqual(statsRef.passStats, prevPassStatsObj);
      assert.strictEqual(statsRef.passStats.gbuffer, prevGbufSlot);
    }

    const prevRenderObj = reusableCategories.render;
    const catRef = timer.getCategoryStats(reusableCategories);
    assert.strictEqual(catRef, reusableCategories);
    if (frame > 0) {
      assert.strictEqual(catRef.render, prevRenderObj);
    }
  }

  assert.equal(timer.getPassKind('compute_cull'), 'compute');
  assert.equal(timer.getPassKind('gbuffer'), 'render');
  assert.equal(timer.getPassKind('pass_7'), 'unknown');
  assert.equal(timer.getPassKind('nonexistent'), null);

  assert.equal(reusableCategories.render.count, 15); // 5 render passes * 3 frames
  assert.equal(reusableCategories.compute.count, 6); // 2 compute passes * 3 frames
  assert.equal(reusableCategories.other.count, 3); // 1 unknown pass * 3 frames
  assert.ok(Math.abs(reusableCategories.render.avgFrameMs - 9.0) < 1e-9);
  assert.ok(Math.abs(reusableCategories.compute.avgFrameMs - 2.5) < 1e-9);

  // reset() cleans up stale keys in reusable out.passStats cleanly.
  timer.reset();
  timer.getStats(reusableStats);
  assert.deepEqual(reusableStats.passStats, {});
  assert.equal(timer.getFrameSummary(reusableSummary).lastFrameRenderMs, 0);
  assert.equal(timer.getFrameSummary(reusableSummary).lastFrameComputeMs, 0);

  timer.destroy();
});

test('W380 CPU fallback mode tracks categories, enforces maxPassesPerFrame overflow cleanly, and supports zero-alloc getFrameSummary(out)', () => {
  let clockMs = 0;
  const timer = new GpuPassTimer({
    device: null,
    maxPassesPerFrame: 3,
    historySize: 4,
    now: () => clockMs,
  });

  timer.beginFrame();
  timer.beginPass('cull', 'compute');
  clockMs += 1.5;
  assert.equal(timer.endPass(), 1.5);

  timer.beginPass('shadow', 'render');
  clockMs += 2.5;
  assert.equal(timer.endPass(), 2.5);

  timer.beginPass(undefined, 'render'); // uses pre-allocated 'pass_2'
  clockMs += 1.0;
  assert.equal(timer.endPass(), 1.0);

  // 4th pass exceeds maxPassesPerFrame=3 -> dropped cleanly and increments droppedPasses!
  assert.equal(timer.beginPass('overflow_pass', 'render'), undefined);
  assert.equal(timer.endPass(), null);
  assert.equal(timer.endFrame(), true);

  const summary = timer.getFrameSummary({});
  assert.equal(summary.mode, 'cpu');
  assert.equal(summary.supported, false);
  assert.equal(summary.droppedPasses, 1);
  assert.equal(summary.lastFramePassCount, 3);
  assert.equal(summary.lastFrameComputeMs, 1.5);
  assert.equal(summary.lastFrameRenderMs, 3.5);
  assert.equal(summary.lastFrameOtherMs, 0);
  assert.equal(timer.getPassKind('cull'), 'compute');
  assert.equal(timer.getPassKind('pass_2'), 'render');
  assert.equal(timer.getPassKind('overflow_pass'), null);

  timer.destroy();
});

test('W380 device.lost while readback buffers are pending and mapping drops safely with zero unhandled rejections or leaked slots', async () => {
  let triggerLost;
  const env = createMockWebGpu({ supported: true, deferMap: true });
  env.device.lost = new Promise((resolve) => {
    triggerLost = resolve;
  });

  const timer = new GpuPassTimer({
    device: env.device,
    maxPassesPerFrame: 4,
    readbackSlots: 3,
  });

  // Queue 2 frames: start mapping Frame 1, leave Frame 2 pending in ring.
  timer.beginFrame();
  timer.beginPass('gbuffer', 'render');
  timer.endPass();
  env.setFrameTimestamps([0n, 3_000_000n]);
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);

  timer.beginFrame();
  timer.beginPass('lighting', 'render');
  timer.endPass();
  env.setFrameTimestamps([0n, 4_000_000n]);
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), true);

  // Start async mapping on slot 0 while slot 1 stays queued in pendingRing.
  const inFlightMap = timer.collectCompletedReadback();
  assert.equal(timer.getFrameSummary().pendingReadbacks, 1);

  // Trigger device.lost while slot 0 is mapping and slot 1 is pending!
  triggerLost({ reason: 'unknown', message: 'driver timeout' });
  await env.device.lost;
  await Promise.resolve();

  // Queued pending slot 1 was flushed cleanly with zero leaked pending entries.
  assert.equal(timer.isDeviceLost, true);
  assert.equal(timer.getFrameSummary().deviceLost, true);
  assert.equal(timer.getFrameSummary().pendingReadbacks, 0);
  assert.equal(await timer.collectCompletedReadback(), null);

  // Rejecting the in-flight mapAsync resolves cleanly to null with zero unhandled rejections.
  env.pendingMaps.shift().reject(new DOMException('Device lost', 'AbortError'));
  assert.equal(await inFlightMap, null);

  // All readback slots returned to unmapped/idle state.
  assert.equal(env.buffers[1].mapState, 'unmapped');
  assert.equal(env.buffers[2].mapState, 'unmapped');

  timer.destroy();
});
