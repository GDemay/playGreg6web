// W203 / W263 / W305: concurrency throttling, priority-tiered prewarm, canonical coalescing,
// priority-lane scheduling, device-loss recovery, bounded compile-latency percentiles & stall
// telemetry tests for web/workers/async_pipeline_cache.js.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const {
  AsyncPipelineCache,
  DEFAULT_MAX_CONCURRENT_PIPELINES,
  MIN_CONCURRENT_COMPILES,
  MAX_CONCURRENT_COMPILES,
  DEFAULT_PIPELINE_PRIORITY,
  DEFAULT_COMPILE_STALL_THRESHOLD_MS,
  DEFAULT_COMPILE_LATENCY_RING_SIZE,
  PIPELINE_KIND,
  PIPELINE_PRIORITY_LANE,
  PIPELINE_PRIORITY_TIER,
  PIPELINE_STATUS,
  canonicalizePipelineKey,
  classifyPriorityTier,
  parsePriorityTier,
  isDeviceLostError,
} = require('../web/workers/async_pipeline_cache.js');

const unhandledRejections = [];
process.on('unhandledRejection', (reason) => unhandledRejections.push(reason));

const tick = () => new Promise((resolve) => setImmediate(resolve));

function createManualDevice(name = 'device') {
  const calls = [];
  let inFlight = 0;
  let peakInFlight = 0;

  const makeAsync = (kind) => (descriptor) => {
    inFlight += 1;
    peakInFlight = Math.max(peakInFlight, inFlight);
    let resolve;
    let reject;
    const promise = new Promise((res, rej) => {
      resolve = res;
      reject = rej;
    });
    const call = {
      kind,
      descriptor,
      settled: false,
      resolve(pipeline = { device: name, kind, label: descriptor?.label }) {
        if (call.settled) return;
        call.settled = true;
        inFlight -= 1;
        resolve(pipeline);
      },
      reject(err = new Error(`compile failed: ${descriptor?.label}`)) {
        if (call.settled) return;
        call.settled = true;
        inFlight -= 1;
        reject(err);
      },
    };
    calls.push(call);
    return promise;
  };

  return {
    name,
    calls,
    createRenderPipelineAsync: makeAsync('render'),
    createComputePipelineAsync: makeAsync('compute'),
    get inFlight() {
      return inFlight;
    },
    get peakInFlight() {
      return peakInFlight;
    },
  };
}

test('W203: configurable maxConcurrentCompiles cap enforcement and dynamic adjustment via setMaxConcurrentCompiles', async () => {
  const device = createManualDevice();
  const cache = new AsyncPipelineCache({ device, maxConcurrentCompiles: 8 });

  assert.equal(DEFAULT_MAX_CONCURRENT_PIPELINES, 4);
  assert.equal(DEFAULT_COMPILE_STALL_THRESHOLD_MS, 16);
  assert.equal(cache.maxConcurrentCompiles, 8);
  assert.equal(cache.maxConcurrent, 8);

  // Burst 20 PSOs under cap = 8: strictly 8 in flight, 12 queued.
  const promises = [];
  for (let i = 0; i < 20; i += 1) {
    promises.push(cache.compileRenderPipeline(`pso_${i}`, { label: `pso_${i}` }));
  }

  assert.equal(device.calls.length, 8);
  assert.equal(device.inFlight, 8);
  assert.equal(device.peakInFlight, 8);

  let tel = cache.getTelemetry();
  assert.equal(tel.maxConcurrentCompiles, 8);
  assert.equal(tel.inFlightCount, 8);
  assert.equal(tel.peakInFlightCount, 8);
  assert.equal(tel.queuedCount, 12);
  assert.equal(tel.peakQueuedCount, 12);
  assert.equal(tel.completedCount, 0);
  assert.equal(tel.failedCount, 0);

  // Throttle down to 2 mid-burst (e.g., during 4K Bink2 intro playback):
  // already in-flight compiles finish naturally, and no new compiles start until inFlight < 2.
  assert.equal(cache.setMaxConcurrentCompiles(2), 2);
  assert.equal(cache.getTelemetry().maxConcurrentCompiles, 2);

  for (let i = 0; i < 6; i += 1) {
    device.calls[i].resolve();
    await tick();
  }
  assert.equal(device.calls.length, 8, 'no new compiles dispatched while inFlight >= 2');
  assert.equal(device.inFlight, 2);
  assert.equal(cache.getTelemetry().inFlightCount, 2);
  assert.equal(cache.getTelemetry().queuedCount, 12);
  assert.equal(cache.getTelemetry().completedCount, 6);

  // One more completes (inFlight drops to 1 < 2) -> exactly 1 queued job starts.
  device.calls[6].resolve();
  await tick();
  assert.equal(device.calls.length, 9);
  assert.equal(device.inFlight, 2);
  assert.equal(cache.getTelemetry().queuedCount, 11);

  // Raise cap to 6 -> immediately drains 4 more jobs into slots (2 -> 6 in flight).
  assert.equal(cache.setMaxConcurrentCompiles(6), 6);
  assert.equal(device.calls.length, 13);
  assert.equal(device.inFlight, 6);
  assert.equal(cache.getTelemetry().inFlightCount, 6);
  assert.equal(cache.getTelemetry().queuedCount, 7);

  // Drain the rest while enforcing the 6-slot cap.
  let idx = 7;
  while (idx < device.calls.length) {
    assert.ok(device.inFlight <= 6, `concurrency cap 6 violated: ${device.inFlight}`);
    device.calls[idx].resolve();
    idx += 1;
    await tick();
  }
  const results = await Promise.all(promises);
  await cache.whenIdle();

  assert.equal(results.length, 20);
  assert.ok(results.every((r) => r.status === PIPELINE_STATUS.READY));
  assert.equal(device.peakInFlight, 8);
  tel = cache.getTelemetry();
  assert.equal(tel.inFlightCount, 0);
  assert.equal(tel.queuedCount, 0);
  assert.equal(tel.completedCount, 20);
  assert.equal(tel.failedCount, 0);
});

test('W203: priority-lane scheduling and promotion of queued pre-warm jobs to urgent on active draw', async () => {
  const device = createManualDevice();
  const cache = new AsyncPipelineCache({ device, maxConcurrentCompiles: 1 });

  // Occupy the single compiler slot.
  const active = cache.compileRenderPipeline('active_0', { label: 'active_0' });

  // Enqueue background pre-warm jobs, a low compute job, and a normal render job.
  const prewarmA = cache.compileRenderPipeline(
    'prewarm_a',
    { label: 'prewarm_a' },
    { dedupeKey: 'pso-prewarm-a', priority: 'prewarm' },
  );
  const prewarmB = cache.compileRenderPipeline(
    'prewarm_b',
    { label: 'prewarm_b' },
    { dedupeKey: 'pso-prewarm-b', lane: 'background' },
  );
  const computeLow = cache.compileComputePipeline(
    'compute_low',
    { label: 'compute_low' },
    { priority: 'low' },
  );
  const renderNormal = cache.compileRenderPipeline(
    'render_normal',
    { label: 'render_normal' },
    { priority: 'normal' },
  );

  let tel = cache.getTelemetry();
  assert.equal(tel.inFlightCount, 1);
  assert.equal(tel.queuedCount, 4);
  assert.equal(tel.promotedCount, 0, 'initial enqueueing does not inflate queued-entry promotedCount');
  assert.deepEqual(tel.queueByLane, {
    urgent: 0,
    high: 0,
    normal: 1,
    low: 1,
    background: 2,
  });

  // Active draw call hits prewarm_b via same ID at 'urgent' priority -> promotes to front of queue!
  const drawPromotedSameId = cache.compileRenderPipeline(
    'prewarm_b',
    { label: 'prewarm_b' },
    { priority: 'urgent' },
  );
  assert.strictEqual(drawPromotedSameId, prewarmB, 'returns exact same promise on queued ID hit');

  // Another active draw call hits prewarm_a via dedupeKey at 'high' priority with a fallbackDescriptor!
  const drawPromotedAlias = cache.compileRenderPipeline(
    'draw_alias_a',
    { label: 'prewarm_a' },
    {
      dedupeKey: 'pso-prewarm-a',
      priority: 'high',
      fallbackDescriptor: { label: 'prewarm_a_fallback', layout: 'auto' },
    },
  );

  // Explicit promote() on queued compute job moves it to urgent (behind prewarm_b only if seq is later;
  // prewarm_b had seq=3 < compute_low seq=4, so prewarm_b stays first among urgent jobs).
  assert.equal(cache.promote('compute_low', PIPELINE_PRIORITY_LANE.URGENT), true);

  // Re-promoting at equal or lower priority is a no-op and returns false.
  assert.equal(cache.promote('compute_low', 'high'), false);
  assert.equal(cache.promote('active_0', 'urgent'), false, 'in-flight job cannot be promoted');
  assert.equal(cache.promote('does_not_exist', 'urgent'), false);

  tel = cache.getTelemetry();
  assert.equal(tel.queuedCount, 4, 'promotions never duplicate queued entries');
  assert.equal(tel.promotedCount, 3, 'prewarm_b + prewarm_a + compute_low promoted');
  assert.equal(tel.dedupedCount, 2, 'same-id hit + dedupeKey hit counted');
  assert.equal(tel.queuedDedupedCount, 2);
  assert.equal(tel.inFlightDedupedCount, 0);
  assert.deepEqual(tel.queueByLane, {
    urgent: 2,
    high: 1,
    normal: 1,
    low: 0,
    background: 0,
  });

  // Also test promote() by dedupeKey: elevate prewarm_a from high -> urgent (seq=2 < prewarm_b seq=3!).
  assert.equal(cache.promote('pso-prewarm-a', 'urgent', PIPELINE_KIND.RENDER), true);
  assert.equal(cache.getTelemetry().promotedCount, 4);

  // Resolve active_0 -> prewarm_a must dispatch FIRST (urgent, earliest seq=2).
  device.calls[0].resolve();
  await active;
  assert.equal(device.calls[1].descriptor.label, 'prewarm_a');

  // Reject prewarm_a primary -> verifies adopted fallbackDescriptor from the draw caller is used!
  device.calls[1].reject(new Error('explicit layout rejected'));
  await tick();
  assert.equal(device.calls[2].descriptor.label, 'prewarm_a_fallback');
  device.calls[2].resolve();
  await Promise.all([prewarmA, drawPromotedAlias]);

  // Next in order: prewarm_b (urgent, seq=3), compute_low (urgent, seq=4), render_normal (normal, seq=5).
  assert.equal(device.calls[3].descriptor.label, 'prewarm_b');
  device.calls[3].resolve();
  await prewarmB;

  assert.equal(device.calls[4].descriptor.label, 'compute_low');
  device.calls[4].resolve();
  await computeLow;

  assert.equal(device.calls[5].descriptor.label, 'render_normal');
  device.calls[5].resolve();
  await renderNormal;
  await cache.whenIdle();

  assert.deepEqual(
    device.calls.map((c) => c.descriptor.label),
    ['active_0', 'prewarm_a', 'prewarm_a_fallback', 'prewarm_b', 'compute_low', 'render_normal'],
  );

  tel = cache.getTelemetry();
  assert.equal(tel.inFlightCount, 0);
  assert.equal(tel.queuedCount, 0);
  assert.equal(tel.completedCount, 5);
  assert.equal(tel.failedCount, 0);
  assert.equal(tel.promotedCount, 4);
  assert.equal(tel.dedupedCount, 2);
});

test('W203: in-flight and queued deduplication prevents duplicate create*PipelineAsync calls across repeated draws', async () => {
  const device = createManualDevice();
  const cache = new AsyncPipelineCache({ device, maxConcurrentCompiles: 2 });

  // Launch 2 in-flight compiles (render + compute) and 1 queued render compile.
  const r0 = cache.compileRenderPipeline('r0', { label: 'r0' }, { dedupeKey: 'shared-pso' });
  const c0 = cache.compileComputePipeline('c0', { label: 'c0' }, { dedupeKey: 'shared-pso' });
  const r1 = cache.compileRenderPipeline('r1', { label: 'r1' }, { dedupeKey: 'queued-pso', priority: 'low' });

  // Simulate 5 draw calls referencing r0 while it is in flight (mix of same-id and distinct-id dedupeKey).
  const r0DupId1 = cache.compileRenderPipeline('r0', { label: 'r0' }, { dedupeKey: 'shared-pso' });
  const r0DupId2 = cache.compileRenderPipeline('r0', { label: 'r0' }, { priority: 'urgent' });
  const r0Alias1 = cache.compileRenderPipeline('r0_draw2', { label: 'r0' }, { dedupeKey: 'shared-pso' });
  const r0Alias2 = cache.compileRenderPipeline('r0_draw3', { label: 'r0' }, { dedupeKey: 'shared-pso' });

  // Compute draw referencing c0 while in flight (same dedupeKey string, isolated compute namespace!).
  const c0Alias = cache.compileComputePipeline('c0_dispatch2', { label: 'c0' }, { dedupeKey: 'shared-pso' });

  // Multiple draws referencing queued r1 before it even starts.
  const r1DupId = cache.compileRenderPipeline('r1', { label: 'r1' }, { priority: 'urgent' });
  const r1Alias = cache.compileRenderPipeline('r1_draw2', { label: 'r1' }, { dedupeKey: 'queued-pso' });

  assert.strictEqual(r0DupId1, r0);
  assert.strictEqual(r0DupId2, r0);
  assert.strictEqual(r1DupId, r1);
  assert.equal(device.calls.length, 2, 'only 2 unique in-flight compiles started');

  let tel = cache.getTelemetry();
  assert.equal(tel.inFlightCount, 2);
  assert.equal(tel.queuedCount, 1);
  assert.equal(tel.dedupedCount, 7);
  assert.equal(tel.inFlightDedupedCount, 5);
  assert.equal(tel.queuedDedupedCount, 2);
  assert.equal(tel.promotedCount, 1, 'r1 promoted from low to urgent');

  // Legacy getStats().dedupeHits still tracks distinct-id dedupeKey hits (r0Alias1, r0Alias2, c0Alias, r1Alias = 4).
  const legacyStats = cache.getStats();
  assert.equal(legacyStats.dedupeHits, 4);
  assert.equal(legacyStats.dedupedCount, 7, 'non-enumerable W203 alias available on getStats()');
  assert.equal(legacyStats.inFlightCount, 2);
  assert.equal(legacyStats.queuedCount, 1);
  assert.equal(legacyStats.promotedCount, 1);

  device.calls[0].resolve();
  device.calls[1].resolve();
  await Promise.all([r0, r0DupId1, r0DupId2, r0Alias1, r0Alias2, c0, c0Alias]);

  assert.equal(device.calls.length, 3, 'r1 dispatched once after slot opened');
  device.calls[2].resolve();
  const [resR0, resR0Alias1, resC0, resC0Alias, resR1, resR1Alias] = await Promise.all([
    r0,
    r0Alias1,
    c0,
    c0Alias,
    r1,
    r1Alias,
  ]);
  await cache.whenIdle();

  assert.strictEqual(resR0.pipeline, resR0Alias1.pipeline);
  assert.equal(resR0.pipeline.kind, 'render');
  assert.strictEqual(resC0.pipeline, resC0Alias.pipeline);
  assert.equal(resC0.pipeline.kind, 'compute');
  assert.notStrictEqual(resR0.pipeline, resC0.pipeline, 'render and compute namespaces isolated');
  assert.strictEqual(resR1.pipeline, resR1Alias.pipeline);
  assert.equal(device.calls.length, 3, 'zero duplicate create*PipelineAsync calls');

  tel = cache.getTelemetry();
  assert.equal(tel.completedCount, 3);
  assert.equal(tel.failedCount, 0);
  assert.equal(tel.inFlightCount, 0);
  assert.equal(tel.queuedCount, 0);
});

test('W203: failure cleanup frees slots, clears dedupe keys, preserves priority drain, and records telemetry', async () => {
  const device = createManualDevice();
  const origRender = device.createRenderPipelineAsync;
  device.createRenderPipelineAsync = (desc) => {
    if (desc.label === 'sync-boom') {
      throw new Error('sync driver validation error');
    }
    return origRender(desc);
  };

  const cache = new AsyncPipelineCache({ device, maxConcurrentCompiles: 1 });

  const pFail = cache.compileRenderPipeline('bad_1', { label: 'bad_1' }, { dedupeKey: 'bad-key' });
  const pFailAlias = cache.compileRenderPipeline('bad_2', { label: 'bad_1' }, { dedupeKey: 'bad-key' });
  const pBg = cache.compileRenderPipeline('bg_1', { label: 'bg_1' }, { priority: 'background' });
  const pSyncThrow = cache.compileRenderPipeline('sync_1', { label: 'sync-boom' }, {
    dedupeKey: 'sync-key',
    priority: 'urgent',
  });
  const pHigh = cache.compileRenderPipeline('high_1', { label: 'high_1' }, { priority: 'high' });

  // Reject bad_1 -> both bad_1 and bad_2 fail, dedupe key 'bad-key' is cleared, sync_1 runs next (urgent) and throws synchronously,
  // freeing the slot immediately so high_1 runs ahead of bg_1.
  device.calls[0].reject(new Error('WGSL compile failure'));
  const [resBad1, resBad2, resSync] = await Promise.all([pFail, pFailAlias, pSyncThrow]);
  await tick();

  assert.equal(resBad1.status, PIPELINE_STATUS.FAILED);
  assert.equal(resBad2.status, PIPELINE_STATUS.FAILED);
  assert.equal(resSync.status, PIPELINE_STATUS.FAILED);
  assert.match(resSync.error.message, /sync driver validation error/);
  assert.equal(cache.hasFailed('bad_1'), true);
  assert.equal(cache.hasFailed('bad_2'), true);

  assert.equal(device.calls.length, 2);
  assert.equal(device.calls[1].descriptor.label, 'high_1', 'high priority drained ahead of background');

  device.calls[1].resolve();
  await pHigh;
  assert.equal(device.calls[2].descriptor.label, 'bg_1');
  device.calls[2].resolve();
  await pBg;

  // Retrying previously failed dedupeKey succeeds cleanly.
  const retry = cache.compileRenderPipeline('bad_1', { label: 'bad_1_fixed' }, { dedupeKey: 'bad-key' });
  assert.equal(device.calls.length, 4);
  device.calls[3].resolve();
  assert.equal((await retry).status, PIPELINE_STATUS.READY);
  assert.equal(cache.hasFailed('bad_1'), false);
  await cache.whenIdle();

  const tel = cache.getTelemetry();
  assert.equal(tel.inFlightCount, 0);
  assert.equal(tel.queuedCount, 0);
  assert.equal(tel.completedCount, 3);
  assert.equal(tel.failedCount, 2);
  assert.equal(tel.failedAsyncCompiles, 2);
  assert.equal(tel.dedupedCount, 1);

  const stats = cache.getStats();
  assert.equal(stats.completedCount, 3);
  assert.equal(stats.failedCount, 2);
});

test('W203: compile latency, queue wait, and stall telemetry with deterministic clock and onStall callback', async () => {
  let now = 1000;
  const stalls = [];
  const warnings = [];
  const device = createManualDevice();
  const cache = new AsyncPipelineCache({
    device,
    maxConcurrentCompiles: 1,
    stallThresholdMs: 20,
    nowFn: () => now,
    onStall: (detail) => {
      stalls.push(detail);
      if (detail.id === 'p_stall_throw') {
        throw new Error('onStall listener bug');
      }
    },
    logger: { warn: (msg) => warnings.push(msg) },
  });

  // t=1000: p0 starts immediately (queueWait=0); p1 queues at t=1000.
  const p0 = cache.compileRenderPipeline('p_fast', { label: 'p_fast' });
  const p1 = cache.compileRenderPipeline('p_stall', { label: 'p_stall' }, { dedupeKey: 'heavy-pso' });
  const p2 = cache.compileComputePipeline('p_stall_throw', { label: 'p_stall_throw' });

  // t=1008: p0 finishes in 8ms (< 20ms threshold, not a stall). p1 starts after 8ms queue wait.
  now = 1008;
  device.calls[0].resolve();
  await p0;

  let tel = cache.getTelemetry();
  assert.equal(tel.completedCount, 1);
  assert.equal(tel.maxCompileMs, 8);
  assert.equal(tel.avgCompileMs, 8);
  assert.equal(tel.lastCompileMs, 8);
  assert.equal(tel.maxQueueWaitMs, 8);
  assert.equal(tel.avgQueueWaitMs, 4); // (0 + 8) / 2 started jobs
  assert.equal(tel.stallCount, 0);
  assert.equal(stalls.length, 0);

  // t=1048: p1 finishes in 40ms (>= 20ms stall threshold). p2 starts after 48ms queue wait.
  now = 1048;
  device.calls[1].resolve();
  await p1;

  tel = cache.getTelemetry();
  assert.equal(tel.completedCount, 2);
  assert.equal(tel.maxCompileMs, 40);
  assert.equal(tel.avgCompileMs, 24); // (8 + 40) / 2
  assert.equal(tel.lastCompileMs, 40);
  assert.equal(tel.maxQueueWaitMs, 48);
  assert.equal(tel.stallCount, 1);
  assert.equal(stalls.length, 1);
  assert.equal(stalls[0].id, 'p_stall');
  assert.equal(stalls[0].dedupeKey, 'heavy-pso');
  assert.equal(stalls[0].elapsedMs, 40);
  assert.equal(stalls[0].thresholdMs, 20);
  assert.equal(stalls[0].failed, false);

  // t=1072: p2 completes in 24ms (stall #2); throwing onStall is contained and logged.
  now = 1072;
  device.calls[2].resolve();
  const r2 = await p2;
  await cache.whenIdle();

  assert.equal(r2.status, PIPELINE_STATUS.READY);
  tel = cache.getTelemetry();
  assert.equal(tel.completedCount, 3);
  assert.equal(tel.failedCount, 0);
  assert.equal(tel.maxCompileMs, 40);
  assert.equal(tel.avgCompileMs, 24); // (8 + 40 + 24) / 3
  assert.equal(tel.lastCompileMs, 24);
  assert.equal(tel.maxQueueWaitMs, 48);
  assert.equal(tel.stallCount, 2);
  assert.equal(stalls.length, 2);
  assert.ok(warnings.some((w) => /onStall callback threw/.test(w)));

  // getStats() preserves exact legacy enumerable shape while exposing non-enumerable telemetry aliases.
  const stats = cache.getStats();
  assert.equal(stats.maxCompileMs, 40);
  assert.equal(stats.avgCompileMs, 24);
  assert.equal(stats.stallCount, 2);
  assert.equal(stats.maxQueueWaitMs, 48);
  assert.equal(stats.completedCount, 3);
  assert.equal(stats.failedCount, 0);
});

test('W203: classic Web Worker UMD export attaches PlayGta6AsyncPipelineCache without module.exports', () => {
  const source = readFileSync(new URL('../web/workers/async_pipeline_cache.js', import.meta.url), 'utf8');
  const workerGlobal = {};
  workerGlobal.self = workerGlobal;
  workerGlobal.globalThis = workerGlobal;
  vm.runInNewContext(source, workerGlobal, { filename: 'async_pipeline_cache.js' });

  const exported = workerGlobal.PlayGta6AsyncPipelineCache;
  assert.ok(exported, 'globalThis.PlayGta6AsyncPipelineCache must be defined in classic Worker');
  assert.equal(typeof exported.AsyncPipelineCache, 'function');
  assert.equal(exported.DEFAULT_MAX_CONCURRENT_PIPELINES, 4);
  assert.equal(exported.DEFAULT_PIPELINE_PRIORITY.render, DEFAULT_PIPELINE_PRIORITY.render);
  assert.equal(exported.PIPELINE_PRIORITY_LANE.URGENT, 100);
  assert.equal(exported.PIPELINE_PRIORITY_LANE.BACKGROUND, -10);

  const instance = new exported.AsyncPipelineCache({ maxConcurrentCompiles: 6 });
  assert.equal(instance.maxConcurrentCompiles, 6);
  assert.equal(typeof instance.getTelemetry, 'function');
  assert.equal(typeof instance.promote, 'function');
  assert.equal(typeof instance.setMaxConcurrentCompiles, 'function');
  assert.equal(typeof instance.resetStats, 'function');
  assert.equal(exported.DEFAULT_COMPILE_LATENCY_RING_SIZE, 128);
  assert.equal(typeof exported.canonicalizePipelineKey, 'function');
  assert.equal(typeof exported.isDeviceLostError, 'function');
  assert.equal(exported.PIPELINE_PRIORITY_LANE.DEMAND, 100);
  assert.equal(exported.PIPELINE_PRIORITY_LANE.WARMUP, -10);
  assert.deepEqual(unhandledRejections, []);
});

test('W263: canonical key coalescing prevents duplicate concurrent render/compute compiles and tracks coalescedCompiles', async () => {
  class MockShaderModule {
    constructor(name) {
      this.name = name;
    }
  }
  const vsMod = new MockShaderModule('vs_main');
  const fsMod = new MockShaderModule('fs_main');
  const csMod = new MockShaderModule('cs_main');
  const otherVsMod = new MockShaderModule('vs_other');

  // Verify deterministic canonicalization regardless of object key order, opaque modules, or cycles.
  const keyA = canonicalizePipelineKey({ vs: vsMod, fs: fsMod, targets: [{ format: 'bgra8unorm', blend: null }] });
  const keyB = canonicalizePipelineKey({ targets: [{ blend: null, format: 'bgra8unorm' }], fs: fsMod, vs: vsMod });
  const keyC = canonicalizePipelineKey({ vs: otherVsMod, fs: fsMod, targets: [{ format: 'bgra8unorm', blend: null }] });
  assert.equal(keyA, keyB);
  assert.notEqual(keyA, keyC);

  const cyclic = { name: 'cycle' };
  cyclic.self = cyclic;
  assert.equal(typeof canonicalizePipelineKey(cyclic), 'string');

  const device = createManualDevice();
  const cache = new AsyncPipelineCache({ device, maxConcurrentCompiles: 4 });

  // Concurrent render compiles using equivalent structured canonicalKey objects (different property order).
  const r1 = cache.compileRenderPipeline(
    'render_1',
    { label: 'draw_1' },
    { canonicalKey: { vs: vsMod, fs: fsMod, format: 'bgra8unorm', topology: 'triangle-list' } },
  );
  const r2 = cache.compileRenderPipeline(
    'render_2',
    { label: 'draw_2' },
    { canonicalKey: { topology: 'triangle-list', format: 'bgra8unorm', fs: fsMod, vs: vsMod } },
  );
  const r1SameId = cache.compileRenderPipeline(
    'render_1',
    { label: 'draw_1_repeat' },
    { canonicalKey: { vs: vsMod, fs: fsMod, format: 'bgra8unorm', topology: 'triangle-list' } },
  );

  // Concurrent compute compiles using the same canonicalKey shape stay isolated in compute namespace.
  const c1 = cache.compileComputePipeline(
    'compute_1',
    { label: 'cs_1' },
    { canonicalKey: { cs: csMod, entry: 'cs_main' } },
  );
  const c2 = cache.compileComputePipeline(
    'compute_2',
    { label: 'cs_2' },
    { canonicalKey: { entry: 'cs_main', cs: csMod } },
  );

  assert.strictEqual(r1SameId, r1);
  assert.equal(device.calls.length, 2, 'Tint only compiles 1 render + 1 compute PSO concurrently');
  assert.equal(cache.getStats().activeCompiles, 2);
  assert.equal(cache.getStats().coalescedCompiles, 3, 'r2 + r1SameId + c2 coalesced');
  assert.equal(cache.getTelemetry().coalescedCompiles, 3);

  device.calls[0].resolve();
  device.calls[1].resolve();
  const [resR1, resR2, resC1, resC2] = await Promise.all([r1, r2, c1, c2]);
  await cache.whenIdle();

  assert.strictEqual(resR1.pipeline, resR2.pipeline);
  assert.strictEqual(resC1.pipeline, resC2.pipeline);
  assert.notStrictEqual(resR1.pipeline, resC1.pipeline);

  // Post-completion cache reuse hits dedupeHits without inflating concurrent coalescedCompiles.
  const r3 = await cache.compileRenderPipeline(
    'render_3',
    { label: 'draw_3' },
    { canonicalKey: { format: 'bgra8unorm', fs: fsMod, topology: 'triangle-list', vs: vsMod } },
  );
  assert.strictEqual(r3.pipeline, resR1.pipeline);
  assert.equal(device.calls.length, 2);
  assert.equal(cache.getStats().coalescedCompiles, 3);
  assert.equal(cache.getStats().dedupeHits, 3);

  // Opt-in coalesceByDescriptor ignores cosmetic top-level label and coalesces identical descriptors.
  const descDevice = createManualDevice();
  const descCache = new AsyncPipelineCache({
    device: descDevice,
    maxConcurrentCompiles: 4,
    coalesceByDescriptor: true,
  });
  const d1 = descCache.compileRenderPipeline('d1', {
    label: 'pass_a',
    layout: 'auto',
    vertex: { module: vsMod, entryPoint: 'vs_main' },
    fragment: { module: fsMod, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
  });
  const d2 = descCache.compileRenderPipeline('d2', {
    label: 'pass_b',
    fragment: { targets: [{ format: 'bgra8unorm' }], entryPoint: 'fs_main', module: fsMod },
    vertex: { entryPoint: 'vs_main', module: vsMod },
    layout: 'auto',
  });
  assert.equal(descDevice.calls.length, 1, 'identical descriptors with different labels coalesce');
  assert.equal(descCache.getStats().coalescedCompiles, 1);
  descDevice.calls[0].resolve();
  const [rd1, rd2] = await Promise.all([d1, d2]);
  assert.strictEqual(rd1.pipeline, rd2.pipeline);
});

test('W263: bounded concurrency prioritizes demand draws ahead of background warmups and promotes coalesced warmups', async () => {
  const device = createManualDevice();
  const cache = new AsyncPipelineCache({ device, maxConcurrentCompiles: 2 });

  // Fill both active compile slots.
  const active0 = cache.compileRenderPipeline('active_0', { label: 'active_0' });
  const active1 = cache.compileRenderPipeline('active_1', { label: 'active_1' });

  // Queue 3 speculative/background warmups.
  const warm0 = cache.compileRenderPipeline('warm_0', { label: 'warm_0' }, { warmup: true, canonicalKey: 'warm-0' });
  const warm1 = cache.compileRenderPipeline('warm_1', { label: 'warm_1' }, { priority: 'warmup', canonicalKey: 'warm-1' });
  const warm2 = cache.compileComputePipeline('warm_2', { label: 'warm_2' }, { lane: 'speculative', canonicalKey: 'warm-2' });

  // A visible demand draw arrives for a brand-new PSO AND another demand draw hits queued warm_1!
  const demandNew = cache.compileRenderPipeline('demand_new', { label: 'demand_new' }, { demand: true });
  const demandHitWarm1 = cache.compileRenderPipeline(
    'demand_warm1_draw',
    { label: 'warm_1' },
    { canonicalKey: 'warm-1', priority: 'demand' },
  );

  assert.equal(device.calls.length, 2, 'strictly respects maxConcurrentCompiles=2');
  assert.equal(cache.getStats().activeCompiles, 2);
  assert.equal(cache.getStats().coalescedCompiles, 1);
  assert.equal(cache.getTelemetry().queueByLane.urgent, 2);
  assert.equal(cache.getTelemetry().queueByLane.background, 2);

  // Drain all slots while checking concurrency cap at every step.
  let idx = 0;
  while (idx < device.calls.length) {
    assert.ok(device.inFlight <= 2, `concurrency cap violated: ${device.inFlight}`);
    device.calls[idx].resolve();
    idx += 1;
    await tick();
  }
  await Promise.all([active0, active1, warm0, warm1, warm2, demandNew, demandHitWarm1]);
  await cache.whenIdle();

  // warm_1 (seq 4, promoted to demand) and demand_new (seq 6, demand) MUST dispatch ahead of warm_0 and warm_2!
  assert.deepEqual(
    device.calls.map((c) => c.descriptor.label),
    ['active_0', 'active_1', 'warm_1', 'demand_new', 'warm_0', 'warm_2'],
  );
  assert.equal(device.peakInFlight, 2);
  assert.equal(cache.getStats().activeCompiles, 0);
});

test('W263: device-loss and compile rejections remove failed in-flight keys, increment compileErrors, allow recovery, and leak zero unhandled rejections', async () => {
  assert.equal(isDeviceLostError(new Error('GPUDevice lost during compile')), true);
  assert.equal(isDeviceLostError({ reason: 'destroyed', message: 'destroyed' }), true);
  assert.equal(isDeviceLostError(new Error('WGSL syntax error')), false);

  const errors = [];
  const device = createManualDevice('dev-v1');
  const cache = new AsyncPipelineCache({
    device,
    maxConcurrentCompiles: 2,
    onError: (id, err) => errors.push({ id, msg: err.message }),
  });

  // Launch a render compile with a coalesced waiter and a fallbackDescriptor; reject with device-loss.
  const lostErr = Object.assign(new Error('GPUDevice was lost'), { name: 'GPUDeviceLostError' });
  const pLost1 = cache.compileRenderPipeline(
    'pso_lost_1',
    { label: 'pso_lost' },
    { canonicalKey: 'key-lost', fallbackDescriptor: { label: 'pso_lost_fallback' } },
  );
  const pLost2 = cache.compileRenderPipeline(
    'pso_lost_2',
    { label: 'pso_lost' },
    { canonicalKey: 'key-lost' },
  );
  const pValErr = cache.compileComputePipeline(
    'pso_bad_wgsl',
    { label: 'pso_bad_wgsl' },
    { canonicalKey: 'key-bad' },
  );

  // Device-lost rejection must NOT trigger fallbackDescriptor retry on the dead device.
  device.calls[0].reject(lostErr);
  device.calls[1].reject(new Error('Tint WGSL validation failed'));
  const [rLost1, rLost2, rValErr] = await Promise.all([pLost1, pLost2, pValErr]);
  await cache.whenIdle();

  assert.equal(device.calls.length, 2, 'device-lost rejection skipped pointless fallbackDescriptor retry');
  assert.equal(rLost1.status, PIPELINE_STATUS.FAILED);
  assert.equal(rLost2.status, PIPELINE_STATUS.FAILED);
  assert.equal(rValErr.status, PIPELINE_STATUS.FAILED);
  assert.equal(cache.getStats().activeCompiles, 0);
  assert.equal(cache.getStats().compileErrors, 2);
  assert.equal(cache.getStats().coalescedCompiles, 1);
  assert.equal(cache.getStats().deviceLostRejections, 1);
  assert.equal(cache.getStats().autoLayoutRetries, 0);
  assert.equal(errors.length, 3);

  // Failed in-flight keys were cleanly evicted so immediate retries on a recovered device succeed!
  const recoveredDevice = createManualDevice('dev-v2');
  cache.device = recoveredDevice;
  const retryLost = cache.compileRenderPipeline('pso_lost_1', { label: 'pso_lost_retry' }, { canonicalKey: 'key-lost' });
  const retryBad = cache.compileComputePipeline('pso_bad_wgsl', { label: 'pso_bad_retry' }, { canonicalKey: 'key-bad' });
  assert.equal(recoveredDevice.calls.length, 2);
  recoveredDevice.calls[0].resolve();
  recoveredDevice.calls[1].resolve();
  const [resRetryLost, resRetryBad] = await Promise.all([retryLost, retryBad]);
  assert.equal(resRetryLost.status, PIPELINE_STATUS.READY);
  assert.equal(resRetryBad.status, PIPELINE_STATUS.READY);
  assert.equal(resRetryLost.pipeline.device, 'dev-v2');

  // Mid-queue device vanishing (device = null) fails queued compiles cleanly without crashing or leaking rejections.
  const activeGood = cache.compileRenderPipeline('g0', { label: 'g0' });
  const activeGood2 = cache.compileRenderPipeline('g1', { label: 'g1' });
  const queuedWhenNull = cache.compileRenderPipeline('g2_null', { label: 'g2_null' }, { canonicalKey: 'key-null' });
  cache.device = null;
  recoveredDevice.calls[2].resolve();
  recoveredDevice.calls[3].resolve();
  const resQueuedNull = await queuedWhenNull;
  await Promise.all([activeGood, activeGood2]);
  await cache.whenIdle();

  assert.equal(resQueuedNull.status, PIPELINE_STATUS.FAILED);
  assert.match(resQueuedNull.error.message, /unavailable or lost/);
  assert.equal(cache.getStats().compileErrors, 3);
  assert.equal(cache.getStats().activeCompiles, 0);

  // Device reset with late old-device rejection never leaks unhandled rejections and allows fresh compile.
  const devV3 = createManualDevice('dev-v3');
  cache.reset(devV3);
  const pReset = cache.compileRenderPipeline('after_reset', { label: 'after_reset' }, { canonicalKey: 'key-null' });
  const devV4 = createManualDevice('dev-v4');
  cache.reset(devV4);
  devV3.calls[0].reject(new Error('GPUDevice lost late'));
  assert.equal((await pReset).status, PIPELINE_STATUS.STALE);

  const pFresh = cache.compileRenderPipeline('after_reset', { label: 'after_reset_v4' }, { canonicalKey: 'key-null' });
  devV4.calls[0].resolve();
  assert.equal((await pFresh).status, PIPELINE_STATUS.READY);
  await tick();
  assert.deepEqual(unhandledRejections, []);
});

test('W263: bounded compile-latency ring computes p50CompileMs/p95CompileMs, wraps cleanly, and resetStats() resets telemetry without dropping live state', async () => {
  let now = 0;
  const device = createManualDevice();
  const cache = new AsyncPipelineCache({
    device,
    maxConcurrentCompiles: 1,
    compileLatencyRingSize: 20,
    nowFn: () => now,
  });

  // Initial stats before any samples.
  assert.equal(DEFAULT_COMPILE_LATENCY_RING_SIZE, 128);
  let stats = cache.getStats();
  assert.equal(stats.p50CompileMs, 0);
  assert.equal(stats.p95CompileMs, 0);
  assert.equal(stats.activeCompiles, 0);
  assert.equal(stats.coalescedCompiles, 0);
  assert.equal(stats.compileErrors, 0);
  assert.equal(stats.compileLatencyRingSize, 20);
  assert.equal(stats.compileLatencySampleCount, 0);

  // Record 20 deterministic compiles with latencies [1, 2, ..., 20] ms.
  for (let i = 1; i <= 20; i += 1) {
    const start = now;
    const p = cache.compileRenderPipeline(`lat_${i}`, { label: `lat_${i}` });
    now = start + i;
    device.calls[device.calls.length - 1].resolve();
    await p;
  }

  stats = cache.getStats();
  assert.equal(stats.compileLatencySampleCount, 20);
  assert.equal(stats.p50CompileMs, 10);
  assert.equal(stats.p95CompileMs, 19);
  assert.equal(stats.maxCompileMs, 20);

  let tel = cache.getTelemetry();
  assert.equal(tel.p50CompileMs, 10);
  assert.equal(tel.p95CompileMs, 19);
  assert.equal(tel.compileLatencySampleCount, 20);

  // Overwrite all 20 ring slots with slower/different latencies (ring stays bounded at 20).
  // Push 20 samples of [5, 5, ..., 5, 50] (19 samples of 5ms, 1 sample of 50ms -> p50=5, p95=5, max in ring=50).
  // Wait: nearest-rank p95 of 20 items is index ceil(0.95*20)-1 = 18 (19th item).
  // So pushing 18 samples of 4ms and 2 samples of 40ms yields sorted [4 x 18, 40, 40] -> p50=4, p95=40!
  for (let i = 0; i < 20; i += 1) {
    const dur = i < 18 ? 4 : 40;
    const start = now;
    const p = cache.compileComputePipeline(`wrap_${i}`, { label: `wrap_${i}` });
    now = start + dur;
    device.calls[device.calls.length - 1].resolve();
    await p;
  }

  stats = cache.getStats();
  assert.equal(stats.compileLatencySampleCount, 20, 'ring buffer capacity strictly bounded at 20');
  assert.equal(stats.p50CompileMs, 4);
  assert.equal(stats.p95CompileMs, 40);

  // Verify resetStats() clears all cumulative counters & latency ring while keeping cached pipelines and active work live!
  const inFlightP = cache.compileRenderPipeline('live_inflight', { label: 'live_inflight' }, { canonicalKey: 'live-key' });
  const coalescedLive = cache.compileRenderPipeline('live_alias', { label: 'live_inflight' }, { canonicalKey: 'live-key' });
  assert.equal(cache.getStats().activeCompiles, 1);
  assert.equal(cache.getStats().coalescedCompiles, 1);

  cache.resetStats();

  stats = cache.getStats();
  assert.equal(stats.activeCompiles, 1, 'live activeCompiles preserved across resetStats()');
  assert.equal(stats.coalescedCompiles, 0);
  assert.equal(stats.compileErrors, 0);
  assert.equal(stats.compiled, 0);
  assert.equal(stats.failed, 0);
  assert.equal(stats.p50CompileMs, 0);
  assert.equal(stats.p95CompileMs, 0);
  assert.equal(stats.avgCompileMs, 0);
  assert.equal(stats.maxCompileMs, 0);
  assert.equal(stats.compileLatencySampleCount, 0);
  assert.equal(stats.stallCount, 0);
  assert.equal(stats.deviceLostRejections, 0);
  assert.ok(cache.get('lat_1') !== null, 'cached pipelines preserved across resetStats()');

  // Complete the in-flight compile after resetStats() -> records cleanly into fresh stats.
  now += 12;
  device.calls[device.calls.length - 1].resolve();
  const [rLive, rAlias] = await Promise.all([inFlightP, coalescedLive]);
  await cache.whenIdle();

  assert.strictEqual(rLive.pipeline, rAlias.pipeline);
  stats = cache.getStats();
  assert.equal(stats.activeCompiles, 0);
  assert.equal(stats.compiled, 1);
  assert.equal(stats.compileLatencySampleCount, 1);
  assert.equal(stats.p50CompileMs, 12);
  assert.equal(stats.p95CompileMs, 12);
  assert.deepEqual(unhandledRejections, []);
});

test('W305: priority tiers (CRITICAL=0..BACKGROUND=3) enforce strict ordering under bounded concurrency [1, 16]', async () => {
  assert.deepEqual(PIPELINE_PRIORITY_TIER, {
    CRITICAL: 0,
    HIGH: 1,
    NORMAL: 2,
    BACKGROUND: 3,
  });
  assert.equal(MIN_CONCURRENT_COMPILES, 1);
  assert.equal(MAX_CONCURRENT_COMPILES, 16);
  assert.equal(DEFAULT_MAX_CONCURRENT_PIPELINES, 4);
  assert.equal(parsePriorityTier(PIPELINE_PRIORITY_TIER.CRITICAL), PIPELINE_PRIORITY_LANE.CRITICAL);
  assert.equal(parsePriorityTier(PIPELINE_PRIORITY_TIER.BACKGROUND), PIPELINE_PRIORITY_LANE.BACKGROUND);
  assert.equal(classifyPriorityTier(PIPELINE_PRIORITY_LANE.CRITICAL), 'critical');
  assert.equal(classifyPriorityTier(PIPELINE_PRIORITY_LANE.HIGH), 'high');
  assert.equal(classifyPriorityTier(PIPELINE_PRIORITY_LANE.NORMAL), 'normal');
  assert.equal(classifyPriorityTier(PIPELINE_PRIORITY_LANE.BACKGROUND), 'background');

  // Verify bounded concurrency clamping to [1, 16] (default 4).
  assert.equal(new AsyncPipelineCache().maxConcurrentCompiles, 4);
  assert.equal(new AsyncPipelineCache({ maxConcurrentCompiles: 1 }).maxConcurrentCompiles, 1);
  assert.equal(new AsyncPipelineCache({ maxConcurrentCompiles: 16 }).maxConcurrentCompiles, 16);
  assert.equal(new AsyncPipelineCache({ maxConcurrentCompiles: 32 }).maxConcurrentCompiles, 16);
  assert.equal(new AsyncPipelineCache({ maxConcurrentCompiles: 999 }).maxConcurrentCompiles, 16);
  assert.equal(new AsyncPipelineCache({ maxConcurrent: 64 }).maxConcurrent, 16);

  const clampCache = new AsyncPipelineCache({ device: createManualDevice(), maxConcurrentCompiles: 4 });
  assert.equal(clampCache.setMaxConcurrentCompiles(64), 16);
  assert.equal(clampCache.setMaxConcurrentCompiles(1), 1);
  assert.equal(clampCache.setMaxConcurrentCompiles(-5), 4);

  // Bounded concurrency cap = 2: occupy both slots, then enqueue mixed W305 tiers.
  const device = createManualDevice();
  const cache = new AsyncPipelineCache({ device, maxConcurrentCompiles: 2 });

  const active0 = cache.prewarm('active_0', { label: 'active_0' }, { priority: PIPELINE_PRIORITY_TIER.CRITICAL });
  const active1 = cache.prewarm('active_1', { label: 'active_1' }, { priority: PIPELINE_PRIORITY_TIER.CRITICAL });

  const bgWorld1 = cache.prewarm('world_far_1', { label: 'world_far_1' }); // default BACKGROUND = 3
  const normProps = cache.prewarm('scene_props', { label: 'scene_props' }, { priority: PIPELINE_PRIORITY_TIER.NORMAL });
  const bgWorld2 = cache.prewarm('world_far_2', { label: 'world_far_2' }, { priority: PIPELINE_PRIORITY_TIER.BACKGROUND });
  const highHud = cache.prewarm('menu_hud', { label: 'menu_hud' }, { priority: PIPELINE_PRIORITY_TIER.HIGH });
  const critStart = cache.prewarm('start_screen', { label: 'start_screen' }, { priority: PIPELINE_PRIORITY_TIER.CRITICAL });
  const critUiText = cache.compileRenderPipeline('ui_text', { label: 'ui_text' }, {
    priorityTier: PIPELINE_PRIORITY_TIER.CRITICAL,
  });
  const highCompute = cache.prewarm('menu_blur_cs', { label: 'menu_blur_cs' }, {
    kind: PIPELINE_KIND.COMPUTE,
    priority: PIPELINE_PRIORITY_TIER.HIGH,
  });

  assert.equal(device.calls.length, 2, 'bounded concurrency strictly capped at 2');
  const stats = cache.getStats();
  assert.equal(stats.activeCompiles, 2);
  assert.equal(stats.peakActiveCompiles, 2);
  assert.deepEqual(stats.queuedByPriority, {
    critical: 2,
    high: 2,
    normal: 1,
    background: 2,
  });

  // Drain one by one and confirm strict CRITICAL -> HIGH -> NORMAL -> BACKGROUND dispatch order (FIFO within tier).
  for (let i = 0; i < 9; i += 1) {
    assert.ok(device.inFlight <= 2, `concurrency cap violated: ${device.inFlight}`);
    device.calls[i].resolve();
    await tick();
  }
  const settled = await Promise.all([
    active0,
    active1,
    bgWorld1,
    normProps,
    bgWorld2,
    highHud,
    critStart,
    critUiText,
    highCompute,
  ]);
  await cache.whenIdle();

  assert.ok(settled.every((r) => r.status === PIPELINE_STATUS.READY));
  assert.equal(device.peakInFlight, 2);
  assert.deepEqual(
    device.calls.map((c) => c.descriptor.label),
    [
      'active_0',
      'active_1',
      'start_screen',
      'ui_text',
      'menu_hud',
      'menu_blur_cs',
      'scene_props',
      'world_far_1',
      'world_far_2',
    ],
  );
  assert.deepEqual(cache.getStats().queuedByPriority, {
    critical: 0,
    high: 0,
    normal: 0,
    background: 0,
  });

  // Batch prewarm([...]) on an idle cache dispatches CRITICAL/HIGH into active slots first while
  // preserving caller's original index order in the resolved array.
  const batchDev = createManualDevice('batch');
  const batchCache = new AsyncPipelineCache({ device: batchDev, maxConcurrentCompiles: 2 });
  const batchPromise = batchCache.prewarm([
    { id: 'b_bg', descriptor: { label: 'b_bg' }, priority: PIPELINE_PRIORITY_TIER.BACKGROUND },
    { id: 'b_norm', descriptor: { label: 'b_norm' }, priority: PIPELINE_PRIORITY_TIER.NORMAL },
    { id: 'b_crit', descriptor: { label: 'b_crit' }, priority: PIPELINE_PRIORITY_TIER.CRITICAL },
    { id: 'b_high', descriptor: { label: 'b_high' }, priority: PIPELINE_PRIORITY_TIER.HIGH },
  ]);

  assert.deepEqual(
    batchDev.calls.map((c) => c.descriptor.label),
    ['b_crit', 'b_high'],
    'batch prewarm fills initial slots with CRITICAL and HIGH ahead of NORMAL/BACKGROUND',
  );
  assert.deepEqual(batchCache.getStats().queuedByPriority, {
    critical: 0,
    high: 0,
    normal: 1,
    background: 1,
  });

  for (let i = 0; i < 4; i += 1) {
    batchDev.calls[i].resolve();
    await tick();
  }
  const batchResults = await batchPromise;
  assert.deepEqual(
    batchResults.map((r) => r.id),
    ['b_bg', 'b_norm', 'b_crit', 'b_high'],
    'batch prewarm resolves results aligned 1:1 with caller request order',
  );
  assert.deepEqual(
    batchDev.calls.map((c) => c.descriptor.label),
    ['b_crit', 'b_high', 'b_norm', 'b_bg'],
  );
});

test('W305: coalescing duplicate keys elevates queued BACKGROUND request to CRITICAL without double-queuing', async () => {
  const device = createManualDevice();
  const cache = new AsyncPipelineCache({ device, maxConcurrentCompiles: 1 });

  // Occupy the single compiler slot.
  const active = cache.prewarm('active_slot', { label: 'active_slot' }, { priority: PIPELINE_PRIORITY_TIER.CRITICAL });

  // Queue BACKGROUND, NORMAL, and HIGH prewarm jobs behind the active compile.
  const bgWorld = cache.prewarm('world_mesh_bg', { label: 'world_mesh' }, {
    canonicalKey: 'pso-world-mesh',
    priority: PIPELINE_PRIORITY_TIER.BACKGROUND,
  });
  const bgSameId = cache.prewarm('menu_bg_shader', { label: 'menu_bg_shader' }, {
    priority: PIPELINE_PRIORITY_TIER.BACKGROUND,
  });
  const normTerrain = cache.prewarm('terrain_norm', { label: 'terrain_norm' }, {
    canonicalKey: 'pso-terrain',
    priority: PIPELINE_PRIORITY_TIER.NORMAL,
  });
  const highUi = cache.prewarm('hud_high', { label: 'hud_high' }, {
    priority: PIPELINE_PRIORITY_TIER.HIGH,
  });

  assert.deepEqual(cache.getStats().queuedByPriority, {
    critical: 0,
    high: 1,
    normal: 1,
    background: 2,
  });

  // Urgent draw needs 'pso-world-mesh' right now -> coalesces onto queued BACKGROUND job & elevates to CRITICAL!
  const urgentDrawAlias = cache.compileRenderPipeline('urgent_world_draw', { label: 'world_mesh' }, {
    canonicalKey: 'pso-world-mesh',
    priorityTier: PIPELINE_PRIORITY_TIER.CRITICAL,
  });

  // Same-id prewarm elevates 'menu_bg_shader' from BACKGROUND -> CRITICAL and returns exact same promise.
  const urgentSameId = cache.prewarm('menu_bg_shader', { label: 'menu_bg_shader' }, {
    priority: PIPELINE_PRIORITY_TIER.CRITICAL,
  });
  assert.strictEqual(urgentSameId, bgSameId);

  // Lower-priority duplicate arriving for already-CRITICAL job must NOT demote it.
  const lowDuplicate = cache.prewarm('world_mesh_low_dup', { label: 'world_mesh' }, {
    canonicalKey: 'pso-world-mesh',
    priority: PIPELINE_PRIORITY_TIER.BACKGROUND,
  });

  // Explicit promoteTier() elevates 'pso-terrain' from NORMAL -> HIGH (after hud_high? no, terrain_norm has earlier seq!).
  assert.equal(cache.promoteTier('pso-terrain', PIPELINE_PRIORITY_TIER.HIGH), true);
  assert.equal(cache.promoteTier('pso-terrain', PIPELINE_PRIORITY_TIER.NORMAL), false, 'never demotes');

  const stats = cache.getStats();
  assert.equal(stats.queued, 4, 'coalesced duplicates never double-queue');
  assert.equal(stats.coalescedCompiles, 3, 'urgentDrawAlias + urgentSameId + lowDuplicate coalesced');
  assert.deepEqual(stats.queuedByPriority, {
    critical: 2,
    high: 2,
    normal: 0,
    background: 0,
  });

  // Drain all 5 unique compiles.
  for (let i = 0; i < 5; i += 1) {
    device.calls[i].resolve();
    await tick();
  }
  const [rActive, rBgWorld, rUrgentAlias, rLowDup, rBgSameId, rNormTerrain, rHighUi] = await Promise.all([
    active,
    bgWorld,
    urgentDrawAlias,
    lowDuplicate,
    bgSameId,
    normTerrain,
    highUi,
  ]);
  await cache.whenIdle();

  assert.equal(device.calls.length, 5, 'zero duplicate createRenderPipelineAsync calls');
  assert.deepEqual(
    device.calls.map((c) => c.descriptor.label),
    ['active_slot', 'world_mesh', 'menu_bg_shader', 'terrain_norm', 'hud_high'],
  );
  assert.strictEqual(rBgWorld.pipeline, rUrgentAlias.pipeline);
  assert.strictEqual(rBgWorld.pipeline, rLowDup.pipeline);
  assert.equal(rActive.status, PIPELINE_STATUS.READY);
  assert.equal(rBgSameId.status, PIPELINE_STATUS.READY);
  assert.equal(rNormTerrain.status, PIPELINE_STATUS.READY);
  assert.equal(rHighUi.status, PIPELINE_STATUS.READY);
});

test('W305: error recovery under priority tiers frees slots, evicts failed keys, drains highest priority, and leaks zero rejections', async () => {
  const device = createManualDevice();
  const origRender = device.createRenderPipelineAsync;
  device.createRenderPipelineAsync = (desc) => {
    if (desc.label === 'crit_sync_throw') {
      throw new Error('Tint sync validation error');
    }
    return origRender(desc);
  };

  const cache = new AsyncPipelineCache({ device, maxConcurrentCompiles: 1 });

  const failCrit = cache.prewarm('fail_crit', { label: 'fail_crit' }, {
    canonicalKey: 'key-crit-fail',
    priority: PIPELINE_PRIORITY_TIER.CRITICAL,
  });
  const bgQueued = cache.prewarm('bg_after_err', { label: 'bg_after_err' }, {
    priority: PIPELINE_PRIORITY_TIER.BACKGROUND,
  });
  const syncThrowCrit = cache.prewarm('crit_sync', { label: 'crit_sync_throw' }, {
    canonicalKey: 'key-sync-throw',
    priority: PIPELINE_PRIORITY_TIER.CRITICAL,
  });
  const highQueued = cache.prewarm('high_after_err', { label: 'high_after_err' }, {
    priority: PIPELINE_PRIORITY_TIER.HIGH,
  });

  // Reject active CRITICAL compile -> slot frees -> crit_sync throws synchronously -> slot frees -> high_after_err runs ahead of bg!
  device.calls[0].reject(new Error('D3D12 PSO compile rejected'));
  const [resFail, resSync] = await Promise.all([failCrit, syncThrowCrit]);
  await tick();

  assert.equal(resFail.status, PIPELINE_STATUS.FAILED);
  assert.equal(resSync.status, PIPELINE_STATUS.FAILED);
  assert.equal(cache.getStats().compileErrors, 2);
  assert.equal(cache.getStats().activeCompiles, 1);
  assert.equal(device.calls[1].descriptor.label, 'high_after_err');

  device.calls[1].resolve();
  await highQueued;
  assert.equal(device.calls[2].descriptor.label, 'bg_after_err');
  device.calls[2].resolve();
  await bgQueued;

  // Evicted failed dedupe key retries cleanly at CRITICAL tier.
  const retryCrit = cache.prewarm('fail_crit_retry', { label: 'fail_crit_fixed' }, {
    canonicalKey: 'key-crit-fail',
    priority: PIPELINE_PRIORITY_TIER.CRITICAL,
  });
  assert.equal(device.calls.length, 4);
  device.calls[3].resolve();
  assert.equal((await retryCrit).status, PIPELINE_STATUS.READY);

  // Batch prewarm gracefully fails malformed entries without throwing or rejecting.
  const malformedBatch = await cache.prewarm([null]);
  assert.equal(malformedBatch.length, 1);
  assert.equal(malformedBatch[0].status, PIPELINE_STATUS.FAILED);
  await cache.whenIdle();
  assert.deepEqual(unhandledRejections, []);
});

test('W305: sanitized additive telemetry (queuedByPriority, activeCompiles, peakActiveCompiles, compileStallCount, avgCompileMs) and resetStats()', async () => {
  let now = 500;
  const device = createManualDevice();
  const cache = new AsyncPipelineCache({
    device,
    maxConcurrentCompiles: 2,
    stallThresholdMs: 16,
    nowFn: () => now,
  });

  let stats = cache.getStats();
  assert.deepEqual(stats.queuedByPriority, { critical: 0, high: 0, normal: 0, background: 0 });
  assert.equal(stats.activeCompiles, 0);
  assert.equal(stats.peakActiveCompiles, 0);
  assert.equal(stats.compileStallCount, 0);
  assert.equal(stats.avgCompileMs, 0);

  // Launch 2 active compiles + 2 queued compiles (1 HIGH, 1 BACKGROUND).
  const c0 = cache.prewarm('c0', { label: 'c0' }, { priority: PIPELINE_PRIORITY_TIER.CRITICAL });
  const c1 = cache.prewarm('c1', { label: 'c1' }, { priority: PIPELINE_PRIORITY_TIER.CRITICAL });
  const qHigh = cache.prewarm('q_high', { label: 'q_high' }, { priority: PIPELINE_PRIORITY_TIER.HIGH });
  const qBg = cache.prewarm('q_bg', { label: 'q_bg' }, { priority: PIPELINE_PRIORITY_TIER.BACKGROUND });

  stats = cache.getStats();
  assert.equal(stats.activeCompiles, 2);
  assert.equal(stats.peakActiveCompiles, 2);
  assert.deepEqual(stats.queuedByPriority, { critical: 0, high: 1, normal: 0, background: 1 });

  // t=510: c0 resolves in 10ms (< 16ms -> no stall); q_high starts at t=510.
  now = 510;
  device.calls[0].resolve();
  await c0;
  stats = cache.getStats();
  assert.equal(stats.compileStallCount, 0);
  assert.equal(stats.avgCompileMs, 10);
  assert.deepEqual(stats.queuedByPriority, { critical: 0, high: 0, normal: 0, background: 1 });

  // t=530: c1 resolves in 30ms (>= 16ms -> stall #1); q_bg starts at t=530.
  now = 530;
  device.calls[1].resolve();
  await c1;
  stats = cache.getStats();
  assert.equal(stats.compileStallCount, 1);
  assert.equal(stats.avgCompileMs, 20); // (10 + 30) / 2
  assert.equal(stats.peakActiveCompiles, 2);

  // Mid-flight resetStats(): clears cumulative stall/latency counters while preserving live active compiles & cached PSOs.
  cache.resetStats();
  stats = cache.getStats();
  assert.equal(stats.activeCompiles, 2);
  assert.equal(stats.peakActiveCompiles, 2, 'live active floor preserved');
  assert.equal(stats.compileStallCount, 0);
  assert.equal(stats.avgCompileMs, 0);
  assert.ok(cache.get('c0') !== null, 'cached pipelines preserved across resetStats()');

  // Complete q_high (550 - 510 = 40ms -> stall #1 post-reset) and q_bg (550 - 530 = 20ms -> stall #2 post-reset).
  now = 550;
  device.calls[2].resolve();
  device.calls[3].resolve();
  await Promise.all([qHigh, qBg]);
  await cache.whenIdle();

  stats = cache.getStats();
  const tel = cache.getTelemetry();
  assert.equal(stats.activeCompiles, 0);
  assert.equal(stats.peakActiveCompiles, 2);
  assert.equal(stats.compileStallCount, 2);
  assert.equal(stats.avgCompileMs, 30); // (40 + 20) / 2
  assert.deepEqual(stats.queuedByPriority, { critical: 0, high: 0, normal: 0, background: 0 });
  assert.equal(tel.peakActiveCompiles, 2);
  assert.equal(tel.compileStallCount, 2);
  assert.deepEqual(tel.queuedByPriority, { critical: 0, high: 0, normal: 0, background: 0 });

  // Idle resetStats() clears peakActiveCompiles, compileStallCount, and avgCompileMs to 0.
  cache.resetStats();
  stats = cache.getStats();
  assert.equal(stats.activeCompiles, 0);
  assert.equal(stats.peakActiveCompiles, 0);
  assert.equal(stats.compileStallCount, 0);
  assert.equal(stats.avgCompileMs, 0);
  assert.deepEqual(stats.queuedByPriority, { critical: 0, high: 0, normal: 0, background: 0 });

  // Verify UMD worker export exposes W305 symbols & prewarm().
  const source = readFileSync(new URL('../web/workers/async_pipeline_cache.js', import.meta.url), 'utf8');
  const workerGlobal = {};
  workerGlobal.self = workerGlobal;
  workerGlobal.globalThis = workerGlobal;
  vm.runInNewContext(source, workerGlobal, { filename: 'async_pipeline_cache.js' });
  const umd = workerGlobal.PlayGta6AsyncPipelineCache;
  assert.deepEqual({ ...umd.PIPELINE_PRIORITY_TIER }, PIPELINE_PRIORITY_TIER);
  assert.equal(umd.MIN_CONCURRENT_COMPILES, 1);
  assert.equal(umd.MAX_CONCURRENT_COMPILES, 16);
  assert.equal(typeof new umd.AsyncPipelineCache().prewarm, 'function');
  assert.equal(typeof new umd.AsyncPipelineCache().promoteTier, 'function');
  assert.deepEqual(unhandledRejections, []);
});
