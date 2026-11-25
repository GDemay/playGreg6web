// W203: concurrency throttling, priority-lane deduplication & stall telemetry tests
// for web/workers/async_pipeline_cache.js.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const {
  AsyncPipelineCache,
  DEFAULT_MAX_CONCURRENT_PIPELINES,
  DEFAULT_PIPELINE_PRIORITY,
  DEFAULT_COMPILE_STALL_THRESHOLD_MS,
  PIPELINE_KIND,
  PIPELINE_PRIORITY_LANE,
  PIPELINE_STATUS,
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
  assert.deepEqual(unhandledRejections, []);
});
