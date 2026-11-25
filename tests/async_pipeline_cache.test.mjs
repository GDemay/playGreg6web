import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const {
  AsyncPipelineCache,
  DEFAULT_MAX_CONCURRENT_PIPELINES,
  DEFAULT_PIPELINE_PRIORITY,
  PIPELINE_STATUS,
} = require('../web/workers/async_pipeline_cache.js');

// Any unhandled rejection in this file fails the "zero unhandled rejections" test at the end.
const unhandledRejections = [];
process.on('unhandledRejection', (reason) => unhandledRejections.push(reason));

const tick = () => new Promise((resolve) => setImmediate(resolve));

/**
 * Mock GPUDevice whose *Async calls stay pending until the test settles them by hand, so the
 * concurrency cap and late (post-reset) resolution can be observed deterministically.
 */
function createManualDevice(name = 'device', { async = true, sync = true } = {}) {
  const calls = [];
  let inFlight = 0;
  let peakInFlight = 0;
  const device = { name, calls, syncCalls: [] };
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
      resolve(pipeline = { device: name, kind, label: descriptor?.label }) {
        inFlight -= 1;
        resolve(pipeline);
      },
      reject(err = new Error(`compile failed: ${descriptor?.label}`)) {
        inFlight -= 1;
        reject(err);
      },
    };
    calls.push(call);
    return promise;
  };
  if (async) {
    device.createRenderPipelineAsync = makeAsync('render');
    device.createComputePipelineAsync = makeAsync('compute');
  }
  if (sync) {
    device.createRenderPipeline = (descriptor) => {
      device.syncCalls.push(descriptor);
      return { device: name, kind: 'render', sync: true, label: descriptor?.label };
    };
    device.createComputePipeline = (descriptor) => {
      device.syncCalls.push(descriptor);
      return { device: name, kind: 'compute', sync: true, label: descriptor?.label };
    };
  }
  Object.defineProperty(device, 'peakInFlight', { get: () => peakInFlight });
  Object.defineProperty(device, 'inFlight', { get: () => inFlight });
  return device;
}

/** Mock device that settles every async compile after 0..3 random event-loop turns. */
function createAutoDevice({ failLabels = new Set() } = {}) {
  let inFlight = 0;
  let peakInFlight = 0;
  let total = 0;
  const run = (kind) => (descriptor) => {
    inFlight += 1;
    total += 1;
    peakInFlight = Math.max(peakInFlight, inFlight);
    return new Promise((resolve, reject) => {
      const settle = (hops) => {
        if (hops > 0) {
          setImmediate(() => settle(hops - 1));
          return;
        }
        inFlight -= 1;
        if (failLabels.has(descriptor.label)) reject(new Error(`bad ${descriptor.label}`));
        else resolve({ kind, label: descriptor.label });
      };
      settle(Math.floor(Math.random() * 4));
    });
  };
  return {
    createRenderPipelineAsync: run('render'),
    createComputePipelineAsync: run('compute'),
    get peakInFlight() { return peakInFlight; },
    get total() { return total; },
  };
}

function recordCallbacks() {
  const ready = [];
  const errors = [];
  return {
    ready,
    errors,
    onReady: (id, pipeline, meta) => ready.push({ id, pipeline, meta }),
    onError: (id, err, meta) => errors.push({ id, err, meta }),
  };
}

test('defaults: maxConcurrent 4 and initial stats', () => {
  assert.equal(DEFAULT_MAX_CONCURRENT_PIPELINES, 4);
  assert.ok(DEFAULT_PIPELINE_PRIORITY.render > DEFAULT_PIPELINE_PRIORITY.compute);
  const cache = new AsyncPipelineCache({ device: createManualDevice() });
  assert.equal(cache.maxConcurrentCompiles, 4);
  assert.equal(cache.prioritizeGraphics, true);
  assert.deepEqual(cache.getStats(), {
    generation: 0,
    maxConcurrent: 4,
    queued: 0,
    inFlight: 0,
    peakQueueDepth: 0,
    compiled: 0,
    failed: 0,
    dedupeHits: 0,
    syncFallbacks: 0,
    autoLayoutRetries: 0,
    discardedStale: 0,
    inFlightCompiles: 0,
    peakInFlightCompiles: 0,
    activeCompiles: 0,
    peakConcurrentCompiles: 0,
    queuedCompiles: 0,
    failedAsyncCompiles: 0,
    avgAsyncCompileMs: 0,
    avgCompileMs: 0,
    maxCompileMs: 0,
    priorityPromotions: 0,
  });
  assert.equal(new AsyncPipelineCache({ maxConcurrent: 0 }).maxConcurrent, 4);
  assert.equal(new AsyncPipelineCache({ maxConcurrent: 2.5 }).maxConcurrent, 4);
  assert.equal(new AsyncPipelineCache({ maxConcurrentCompiles: -1 }).maxConcurrentCompiles, 4);
  assert.equal(new AsyncPipelineCache({ maxConcurrentCompiles: 2.5 }).maxConcurrentCompiles, 4);
  assert.equal(new AsyncPipelineCache({ maxConcurrentCompiles: null }).maxConcurrentCompiles, 4);
  assert.equal(new AsyncPipelineCache({ maxConcurrentCompiles: 0 }).maxConcurrentCompiles, 0);
});

test('concurrency cap: 12 pipelines with maxConcurrent 3 never exceed 3 in flight (FIFO)', async () => {
  const device = createManualDevice();
  const cb = recordCallbacks();
  // Opt out of W168 graphics-over-compute default to verify legacy cross-kind FIFO behavior.
  const cache = new AsyncPipelineCache({ device, maxConcurrent: 3, prioritizeGraphics: false, ...cb });

  const results = [];
  for (let i = 0; i < 12; i += 1) {
    const compile = i % 2 === 0 ? 'compileRenderPipeline' : 'compileComputePipeline';
    results.push(cache[compile](`p${i}`, { label: `p${i}` }));
  }
  assert.equal(device.calls.length, 3);
  assert.deepEqual(device.calls.map((c) => c.descriptor.label), ['p0', 'p1', 'p2']);
  let stats = cache.getStats();
  assert.equal(stats.inFlight, 3);
  assert.equal(stats.queued, 9);
  assert.equal(stats.peakQueueDepth, 9);
  assert.ok(cache.isPending('p11'));
  assert.equal(cache.get('p0'), null);

  // Settle one compile at a time, out of order; the cap must hold throughout.
  let settled = 0;
  while (settled < 12) {
    const open = device.calls.filter((c) => !c.done);
    assert.ok(open.length <= 3, `in flight ${open.length}`);
    const call = open[open.length - 1];
    call.done = true;
    call.resolve();
    settled += 1;
    await tick();
    assert.ok(cache.getStats().inFlight <= 3);
  }
  await cache.whenIdle();

  assert.equal(device.peakInFlight, 3);
  assert.deepEqual(device.calls.map((c) => c.descriptor.label),
    Array.from({ length: 12 }, (_, i) => `p${i}`), 'dispatched in FIFO order');
  const resolved = await Promise.all(results);
  assert.ok(resolved.every((r) => r.status === PIPELINE_STATUS.READY));
  assert.equal(cb.ready.length, 12);
  assert.equal(cb.errors.length, 0);
  assert.equal(cache.get('p5').label, 'p5');
  assert.equal(cache.get('p5').kind, 'compute');
  stats = cache.getStats();
  assert.equal(stats.compiled, 12);
  assert.equal(stats.inFlight, 0);
  assert.equal(stats.queued, 0);
});

test('concurrency cap holds under randomized async completion (200-PSO burst)', async () => {
  const device = createAutoDevice({ failLabels: new Set(['p17', 'p99']) });
  const cache = new AsyncPipelineCache({ device });
  const promises = [];
  for (let i = 0; i < 200; i += 1) {
    promises.push(cache.compileRenderPipeline(i, { label: `p${i}` }));
  }
  await cache.flush();
  const results = await Promise.all(promises);
  assert.equal(device.peakInFlight, DEFAULT_MAX_CONCURRENT_PIPELINES);
  assert.equal(device.total, 200);
  assert.equal(results.filter((r) => r.status === 'failed').length, 2);
  assert.ok(cache.hasFailed(17));
  assert.ok(cache.hasFailed(99));
  assert.equal(cache.get(18).label, 'p18');
  const stats = cache.getStats();
  assert.equal(stats.compiled, 198);
  assert.equal(stats.failed, 2);
  assert.equal(stats.peakQueueDepth, 196);
});

test('dedupe: ids sharing a dedupeKey attach to the in-flight compile', async () => {
  const device = createManualDevice();
  const cb = recordCallbacks();
  const cache = new AsyncPipelineCache({ device, ...cb });

  const a = cache.compileRenderPipeline('a', { label: 'shared' }, { dedupeKey: 'k1' });
  const b = cache.compileRenderPipeline('b', { label: 'shared' }, { dedupeKey: 'k1' });
  const c = cache.compileRenderPipeline('c', { label: 'shared' }, { dedupeKey: 'k1' });
  assert.equal(device.calls.length, 1, 'compiled once');
  assert.ok(cache.isPending('b'));
  assert.equal(cache.getStats().dedupeHits, 2);

  device.calls[0].resolve();
  const [ra, rb, rc] = await Promise.all([a, b, c]);
  assert.equal(ra.status, 'ready');
  assert.equal(ra.dedupe, false);
  assert.equal(rb.dedupe, true);
  assert.strictEqual(ra.pipeline, rb.pipeline);
  assert.strictEqual(rb.pipeline, rc.pipeline);
  assert.strictEqual(cache.get('c'), cache.get('a'));
  assert.deepEqual(cb.ready.map((r) => r.id), ['a', 'b', 'c']);
  assert.equal(cache.getStats().compiled, 1);
});

test('dedupe: a later id with a completed dedupeKey reuses the ready pipeline synchronously', async () => {
  const device = createManualDevice();
  const cb = recordCallbacks();
  const cache = new AsyncPipelineCache({ device, ...cb });

  const first = cache.compileComputePipeline(1, { label: 'cs' }, { dedupeKey: 'cs-key' });
  device.calls[0].resolve();
  const r1 = await first;

  const second = cache.compileComputePipeline(2, { label: 'cs' }, { dedupeKey: 'cs-key' });
  assert.strictEqual(cache.get(2), r1.pipeline, 'available without waiting');
  const r2 = await second;
  assert.equal(r2.status, 'ready');
  assert.equal(r2.dedupe, true);
  assert.equal(device.calls.length, 1);
  assert.equal(cache.getStats().dedupeHits, 1);
  assert.equal(cb.ready.length, 2);

  // Same dedupeKey on a different pipeline kind is not shared.
  const third = cache.compileRenderPipeline(3, { label: 'vs' }, { dedupeKey: 'cs-key' });
  assert.equal(device.calls.length, 2);
  device.calls[1].resolve();
  assert.equal((await third).pipeline.kind, 'render');
});

test('dedupe: a failed compile is not cached, so the key can be compiled again', async () => {
  const device = createManualDevice();
  const cb = recordCallbacks();
  const cache = new AsyncPipelineCache({ device, ...cb });

  const a = cache.compileRenderPipeline('a', { label: 'x' }, { dedupeKey: 'k' });
  const b = cache.compileRenderPipeline('b', { label: 'x' }, { dedupeKey: 'k' });
  device.calls[0].reject();
  const [ra, rb] = await Promise.all([a, b]);
  assert.equal(ra.status, 'failed');
  assert.equal(rb.status, 'failed');
  assert.ok(cache.hasFailed('a') && cache.hasFailed('b'));
  assert.deepEqual(cb.errors.map((e) => e.id), ['a', 'b']);
  assert.equal(cache.getStats().failed, 1);

  const again = cache.compileRenderPipeline('a', { label: 'x' }, { dedupeKey: 'k' });
  assert.equal(device.calls.length, 2, 'retried after failure');
  assert.ok(cache.isPending('a'));
  device.calls[1].resolve();
  assert.equal((await again).status, 'ready');
  assert.equal(cache.hasFailed('a'), false);
});

test('fallbackDescriptor: primary rejection retries once with the fallback', async () => {
  const device = createManualDevice();
  const cb = recordCallbacks();
  const cache = new AsyncPipelineCache({ device, maxConcurrent: 1, ...cb });

  const primary = { label: 'explicit', layout: { explicit: true } };
  const fallback = { label: 'auto', layout: 'auto' };
  const p = cache.compileRenderPipeline('pso', primary, { fallbackDescriptor: fallback });
  const queued = cache.compileRenderPipeline('next', { label: 'next' });
  device.calls[0].reject(new Error('layout mismatch'));
  await tick();

  assert.equal(device.calls.length, 2);
  assert.strictEqual(device.calls[1].descriptor, fallback);
  assert.equal(cache.getStats().inFlight, 1, 'retry keeps the slot; next is still queued');
  assert.equal(cache.getStats().queued, 1);
  assert.equal(cache.getStats().autoLayoutRetries, 1);

  device.calls[1].resolve();
  const result = await p;
  assert.equal(result.status, 'ready');
  assert.equal(result.retried, true);
  assert.equal(result.pipeline.label, 'auto');
  assert.equal(cb.errors.length, 0);

  device.calls[2].resolve();
  assert.equal((await queued).status, 'ready');
});

test('fallbackDescriptor: failure of the retry is reported once via onError', async () => {
  const device = createManualDevice();
  const cb = recordCallbacks();
  const cache = new AsyncPipelineCache({ device, ...cb });

  const p = cache.compileComputePipeline('cs', { label: 'a' }, { fallbackDescriptor: { label: 'b' } });
  device.calls[0].reject();
  await tick();
  device.calls[1].reject(new Error('fallback also failed'));
  const result = await p;
  assert.equal(result.status, 'failed');
  assert.equal(result.error.message, 'fallback also failed');
  assert.equal(device.calls.length, 2, 'only one retry');
  assert.equal(cb.errors.length, 1);
  assert.equal(cb.errors[0].meta.retried, true);
  const stats = cache.getStats();
  assert.equal(stats.autoLayoutRetries, 1);
  assert.equal(stats.failed, 1);
  assert.equal(stats.inFlight, 0);
});

test('reset(newDevice): late results from the old device are discarded', async () => {
  const oldDevice = createManualDevice('old');
  const newDevice = createManualDevice('new');
  const cb = recordCallbacks();
  const cache = new AsyncPipelineCache({ device: oldDevice, maxConcurrent: 2, ...cb });

  const a = cache.compileRenderPipeline('a', { label: 'a' }, { dedupeKey: 'ka' });
  const b = cache.compileRenderPipeline('b', { label: 'b' });
  const c = cache.compileRenderPipeline('c', { label: 'c' }); // queued
  const d = cache.compileComputePipeline('d', { label: 'd' }, { fallbackDescriptor: { label: 'd2' } });
  assert.equal(oldDevice.calls.length, 2);

  let idleBeforeReset = false;
  cache.whenIdle().then(() => { idleBeforeReset = true; });

  cache.reset(newDevice);
  assert.equal(cache.generation, 1);
  assert.strictEqual(cache.device, newDevice);
  assert.deepEqual(
    (await Promise.all([a, b, c, d])).map((r) => r.status),
    ['stale', 'stale', 'stale', 'stale'],
  );
  await tick();
  assert.equal(idleBeforeReset, true, 'whenIdle waiters released by reset');
  assert.equal(cache.isPending('a'), false);
  assert.equal(cache.getStats().queued, 0);
  assert.equal(cache.getStats().inFlight, 0);

  // New-generation compiles of the same ids / dedupe key go to the new device.
  const a2 = cache.compileRenderPipeline('a', { label: 'a' }, { dedupeKey: 'ka' });
  const b2 = cache.compileRenderPipeline('b', { label: 'b' });
  assert.equal(newDevice.calls.length, 2, 'dedupe table was cleared');
  assert.equal(cache.getStats().dedupeHits, 0);

  // The old device now settles late: one resolve, one reject.
  oldDevice.calls[0].resolve({ stale: true });
  oldDevice.calls[1].reject(new Error('device lost'));
  await tick();
  assert.equal(cache.getStats().discardedStale, 2);
  assert.equal(cb.ready.length, 0, 'no onReady for stale results');
  assert.equal(cb.errors.length, 0, 'no onError for stale results');
  assert.equal(cache.get('a'), null, 'stale pipeline not written into the new maps');
  assert.ok(cache.isPending('a'));
  assert.ok(cache.isPending('b'));
  assert.equal(cache.hasFailed('b'), false);
  assert.equal(cache.getStats().inFlight, 2, 'stale settlement does not free new-generation slots');
  assert.equal(oldDevice.calls.length, 2, 'queued old job never reached the old device');

  newDevice.calls[0].resolve();
  newDevice.calls[1].resolve();
  const [ra2, rb2] = await Promise.all([a2, b2]);
  assert.equal(ra2.pipeline.device, 'new');
  assert.equal(rb2.generation, 1);
  assert.equal(cache.get('a').device, 'new');
  assert.deepEqual(cb.ready.map((r) => r.id), ['a', 'b']);
  await cache.whenIdle();
  const stats = cache.getStats();
  assert.equal(stats.compiled, 2);
  assert.equal(stats.failed, 0);
  assert.equal(stats.discardedStale, 2);
});

test('reset(): a stale rejection never triggers the fallbackDescriptor retry', async () => {
  const oldDevice = createManualDevice('old');
  const cache = new AsyncPipelineCache({ device: oldDevice });
  const p = cache.compileRenderPipeline('x', { label: 'x' }, { fallbackDescriptor: { label: 'y' } });
  cache.reset(createManualDevice('new'));
  oldDevice.calls[0].reject();
  await tick();
  assert.equal((await p).status, 'stale');
  assert.equal(oldDevice.calls.length, 1);
  assert.equal(cache.getStats().autoLayoutRetries, 0);
  assert.equal(cache.getStats().discardedStale, 1);
});

test('sync fallback when create*PipelineAsync is absent', async () => {
  const device = createManualDevice('sync-only', { async: false });
  const cb = recordCallbacks();
  const cache = new AsyncPipelineCache({ device, ...cb });

  const p = cache.compileRenderPipeline('vs', { label: 'vs' }, { dedupeKey: 'k' });
  assert.equal(cache.get('vs').sync, true, 'ready immediately');
  const result = await p;
  assert.equal(result.status, 'ready');
  assert.equal(result.syncFallback, true);
  assert.equal(result.pipeline.label, 'vs');

  const cs = await cache.compileComputePipeline('cs', { label: 'cs' });
  assert.equal(cs.pipeline.kind, 'compute');

  const dup = await cache.compileRenderPipeline('vs2', { label: 'vs' }, { dedupeKey: 'k' });
  assert.strictEqual(dup.pipeline, result.pipeline);
  assert.equal(device.syncCalls.length, 2);

  const stats = cache.getStats();
  assert.equal(stats.syncFallbacks, 2);
  assert.equal(stats.compiled, 2);
  assert.equal(stats.dedupeHits, 1);
  assert.equal(stats.inFlight, 0);
  assert.equal(cb.ready.length, 3);
});

test('sync fallback: disabled, missing, or throwing sync APIs report failure cleanly', async () => {
  const cb = recordCallbacks();
  const syncOnly = createManualDevice('sync-only', { async: false });
  const cache = new AsyncPipelineCache({ device: syncOnly, ...cb });

  const disabled = await cache.compileRenderPipeline('a', { label: 'a' }, { allowSyncFallback: false });
  assert.equal(disabled.status, 'failed');
  assert.match(disabled.error.message, /allowSyncFallback is false/);
  assert.equal(syncOnly.syncCalls.length, 0);

  syncOnly.createRenderPipeline = (desc) => {
    if (desc.layout !== 'auto') throw new Error('layout mismatch');
    return { label: desc.label, auto: true };
  };
  const retried = await cache.compileRenderPipeline('b', { label: 'b', layout: {} }, {
    fallbackDescriptor: { label: 'b', layout: 'auto' },
  });
  assert.equal(retried.status, 'ready');
  assert.equal(retried.pipeline.auto, true);
  assert.equal(cache.getStats().autoLayoutRetries, 1);

  const thrown = await cache.compileRenderPipeline('c', { label: 'c', layout: {} });
  assert.equal(thrown.status, 'failed');
  assert.equal(thrown.error.message, 'layout mismatch');

  const noDevice = new AsyncPipelineCache({ ...cb });
  const none = await noDevice.compileComputePipeline('d', { label: 'd' });
  assert.equal(none.status, 'failed');
  assert.match(none.error.message, /no GPU device/);

  const bare = new AsyncPipelineCache({ device: {}, ...cb });
  assert.equal((await bare.compileRenderPipeline('e', {})).status, 'failed');

  assert.deepEqual(cb.errors.map((e) => e.id), ['a', 'c', 'd', 'e']);
});

test('delete(): queued jobs are dropped, in-flight results no longer land', async () => {
  const device = createManualDevice();
  const cb = recordCallbacks();
  const cache = new AsyncPipelineCache({ device, maxConcurrent: 1, ...cb });

  const a = cache.compileRenderPipeline('a', { label: 'a' });
  const b = cache.compileRenderPipeline('b', { label: 'b' }, { dedupeKey: 'kb' });
  assert.equal(cache.getStats().queued, 1);

  assert.equal(cache.delete('b'), true);
  assert.equal((await b).status, 'deleted');
  assert.equal(cache.getStats().queued, 0);
  assert.equal(cache.delete('b'), false);

  assert.equal(cache.delete('a'), true);
  assert.equal((await a).status, 'deleted');
  device.calls[0].resolve();
  await cache.whenIdle();
  assert.equal(cache.get('a'), null);
  assert.equal(cb.ready.length, 0);
  assert.equal(device.calls.length, 1, 'deleted queued job never compiled');

  // Recompiling after delete works; a fresh compile of the same key is not a dedupe hit.
  const b2 = cache.compileRenderPipeline('b', { label: 'b' }, { dedupeKey: 'kb' });
  device.calls[1].resolve();
  assert.equal((await b2).status, 'ready');
  assert.equal(cache.getStats().dedupeHits, 0);
});

test('repeat compile of a pending or ready id does not recompile', async () => {
  const device = createManualDevice();
  const cache = new AsyncPipelineCache({ device });
  const p1 = cache.compileRenderPipeline('x', { label: 'x' });
  const p2 = cache.compileRenderPipeline('x', { label: 'x' });
  assert.strictEqual(p1, p2);
  device.calls[0].resolve();
  await p1;
  const p3 = await cache.compileRenderPipeline('x', { label: 'x' });
  assert.equal(p3.status, 'ready');
  assert.equal(device.calls.length, 1);
});

test('W138: maxConcurrentCompiles: 2 caps 5 distinct pipelines in FIFO order with telemetry', async () => {
  const device = createManualDevice();
  const cache = new AsyncPipelineCache({ device, maxConcurrentCompiles: 2, prioritizeGraphics: false });
  assert.equal(cache.maxConcurrentCompiles, 2);

  const promises = [
    cache.compileRenderPipeline('k0', { label: 'k0' }),
    cache.compileComputePipeline('k1', { label: 'k1' }),
    cache.compileRenderPipeline('k2', { label: 'k2' }),
    cache.compileComputePipeline('k3', { label: 'k3' }),
    cache.compileRenderPipeline('k4', { label: 'k4' }),
  ];

  assert.equal(device.calls.length, 2, 'only 2 async compiles active simultaneously');
  assert.deepEqual(device.calls.map((c) => c.descriptor.label), ['k0', 'k1']);
  let stats = cache.getStats();
  assert.equal(stats.inFlightCompiles, 2);
  assert.equal(stats.peakInFlightCompiles, 2);
  assert.equal(stats.queuedCompiles, 3);
  assert.equal(stats.failedAsyncCompiles, 0);

  // Resolve k0 -> k2 starts next (FIFO)
  device.calls[0].resolve();
  await promises[0];
  assert.equal(device.calls.length, 3);
  assert.equal(device.calls[2].descriptor.label, 'k2');
  stats = cache.getStats();
  assert.equal(stats.inFlightCompiles, 2);
  assert.equal(stats.peakInFlightCompiles, 2);
  assert.equal(stats.queuedCompiles, 2);

  // Resolve k1 -> k3 starts next (FIFO)
  device.calls[1].resolve();
  await promises[1];
  assert.equal(device.calls.length, 4);
  assert.equal(device.calls[3].descriptor.label, 'k3');
  assert.equal(cache.getStats().queuedCompiles, 1);

  // Resolve k2 -> k4 starts next (FIFO)
  device.calls[2].resolve();
  await promises[2];
  assert.equal(device.calls.length, 5);
  assert.equal(device.calls[4].descriptor.label, 'k4');
  assert.equal(cache.getStats().queuedCompiles, 0);

  device.calls[3].resolve();
  device.calls[4].resolve();
  const results = await Promise.all(promises);
  await cache.whenIdle();

  assert.ok(results.every((r) => r.status === PIPELINE_STATUS.READY));
  assert.equal(device.peakInFlight, 2);
  assert.deepEqual(
    device.calls.map((c) => c.descriptor.label),
    ['k0', 'k1', 'k2', 'k3', 'k4'],
  );
  stats = cache.getStats();
  assert.equal(stats.inFlightCompiles, 0);
  assert.equal(stats.peakInFlightCompiles, 2);
  assert.equal(stats.queuedCompiles, 0);
  assert.equal(stats.failedAsyncCompiles, 0);
  assert.equal(stats.compiled, 5);
});

test('W138: maxConcurrentCompiles: 0 allows unlimited concurrent async compiles', async () => {
  const device = createManualDevice();
  const cache = new AsyncPipelineCache({ device, maxConcurrentCompiles: 0 });
  assert.equal(cache.maxConcurrentCompiles, 0);

  const promises = [];
  for (let i = 0; i < 5; i += 1) {
    promises.push(cache.compileRenderPipeline(`u${i}`, { label: `u${i}` }));
  }

  assert.equal(device.calls.length, 5, 'all 5 compiles start immediately when unlimited');
  let stats = cache.getStats();
  assert.equal(stats.inFlightCompiles, 5);
  assert.equal(stats.peakInFlightCompiles, 5);
  assert.equal(stats.queuedCompiles, 0);

  for (const call of device.calls) call.resolve();
  await Promise.all(promises);
  await cache.whenIdle();

  stats = cache.getStats();
  assert.equal(stats.inFlightCompiles, 0);
  assert.equal(stats.peakInFlightCompiles, 5);
  assert.equal(stats.queuedCompiles, 0);
  assert.equal(stats.compiled, 5);
});

test('W138: deduplication while queued and in-flight returns same promise without double-queuing', async () => {
  const device = createManualDevice();
  const cache = new AsyncPipelineCache({ device, maxConcurrentCompiles: 2 });

  const p0a = cache.compileRenderPipeline('p0', { label: 'p0' }, { dedupeKey: 'dk0' });
  const p1 = cache.compileRenderPipeline('p1', { label: 'p1' }, { dedupeKey: 'dk1' });
  const p2a = cache.compileRenderPipeline('p2', { label: 'p2' }, { dedupeKey: 'dk2' });

  // Repeat request for already in-flight id returns exact same promise.
  const p0b = cache.compileRenderPipeline('p0', { label: 'p0' }, { dedupeKey: 'dk0' });
  assert.strictEqual(p0a, p0b);

  // Repeat request for already queued id returns exact same promise without double-queuing.
  const p2b = cache.compileRenderPipeline('p2', { label: 'p2' }, { dedupeKey: 'dk2' });
  assert.strictEqual(p2a, p2b);

  // Distinct id sharing the queued dedupeKey attaches to the queued job without double-queuing.
  const p2Alias = cache.compileRenderPipeline('p2_alias', { label: 'p2' }, { dedupeKey: 'dk2' });

  let stats = cache.getStats();
  assert.equal(stats.inFlightCompiles, 2);
  assert.equal(stats.queuedCompiles, 1, 'queued job was not double-queued');
  assert.equal(stats.dedupeHits, 1);

  device.calls[0].resolve();
  device.calls[1].resolve();
  await Promise.all([p0a, p1]);
  assert.equal(device.calls.length, 3);
  assert.equal(device.calls[2].descriptor.label, 'p2');

  device.calls[2].resolve();
  const [r2a, r2b, r2Alias] = await Promise.all([p2a, p2b, p2Alias]);
  await cache.whenIdle();

  assert.equal(device.calls.length, 3, 'each distinct pipeline compiled once');
  assert.strictEqual(r2a.pipeline, r2b.pipeline);
  assert.strictEqual(r2a.pipeline, r2Alias.pipeline);
  assert.equal(r2Alias.dedupe, true);
  stats = cache.getStats();
  assert.equal(stats.inFlightCompiles, 0);
  assert.equal(stats.queuedCompiles, 0);
  assert.equal(stats.compiled, 3);
});

test('W138: failed async compiles release concurrency slot, increment failedAsyncCompiles, and drain queue', async () => {
  const device = createManualDevice();
  const origComputeAsync = device.createComputePipelineAsync;
  device.createComputePipelineAsync = (descriptor) => {
    if (descriptor.label === 'sync-throw') {
      throw new Error('sync throw during async dispatch');
    }
    return origComputeAsync(descriptor);
  };

  const cache = new AsyncPipelineCache({ device, maxConcurrentCompiles: 2 });

  const p0 = cache.compileRenderPipeline('f0', { label: 'f0' });
  const p1 = cache.compileComputePipeline('f1', { label: 'sync-throw' });
  const p2 = cache.compileRenderPipeline('f2', { label: 'f2' });
  const p3 = cache.compileRenderPipeline('f3', { label: 'f3' });
  const p4 = cache.compileComputePipeline('f4', { label: 'f4' });

  // Wait for f1's sync throw rejection microtask to settle and drain f2 into the freed slot.
  const r1 = await p1;
  assert.equal(r1.status, PIPELINE_STATUS.FAILED);
  assert.match(r1.error.message, /sync throw during async dispatch/);
  assert.equal(cache.getStats().failedAsyncCompiles, 1);
  assert.equal(cache.getStats().inFlightCompiles, 2, 'slot released and refilled with f2');
  assert.equal(cache.getStats().queuedCompiles, 2);
  assert.deepEqual(device.calls.map((c) => c.descriptor.label), ['f0', 'f2']);

  // Reject f0 -> must release its slot, increment failedAsyncCompiles, and drain f3.
  device.calls[0].reject(new Error('shader validation error'));
  const r0 = await p0;
  assert.equal(r0.status, PIPELINE_STATUS.FAILED);
  assert.equal(r0.error.message, 'shader validation error');
  assert.equal(cache.getStats().failedAsyncCompiles, 2);
  assert.equal(cache.getStats().inFlightCompiles, 2);
  assert.equal(cache.getStats().queuedCompiles, 1);
  assert.equal(device.calls[2].descriptor.label, 'f3');

  // Resolve remaining queued compiles (f2, f3, f4).
  device.calls[1].resolve();
  await p2;
  assert.equal(device.calls[3].descriptor.label, 'f4');
  device.calls[2].resolve();
  device.calls[3].resolve();
  const [r2, r3, r4] = await Promise.all([p2, p3, p4]);
  await cache.whenIdle();

  assert.equal(r2.status, PIPELINE_STATUS.READY);
  assert.equal(r3.status, PIPELINE_STATUS.READY);
  assert.equal(r4.status, PIPELINE_STATUS.READY);
  const stats = cache.getStats();
  assert.equal(stats.inFlightCompiles, 0);
  assert.equal(stats.peakInFlightCompiles, 2);
  assert.equal(stats.queuedCompiles, 0);
  assert.equal(stats.failedAsyncCompiles, 2);
  assert.equal(stats.failed, 2);
  assert.equal(stats.compiled, 3);
});

test('W138: deterministic fake-clock nowFn tracks avgAsyncCompileMs without queue wait time', async () => {
  let now = 100;
  const device = createManualDevice();
  const cache = new AsyncPipelineCache({
    device,
    maxConcurrentCompiles: 2,
    nowFn: () => now,
  });

  assert.equal(cache.getStats().avgAsyncCompileMs, 0);

  // t=100: p0 and p1 launch; p2 waits in queue.
  const p0 = cache.compileRenderPipeline('t0', { label: 't0' });
  const p1 = cache.compileRenderPipeline('t1', { label: 't1' });
  const p2 = cache.compileComputePipeline('t2', { label: 't2' });

  // t=110: p0 completes in 10ms; p2 launches at t=110 (not t=100).
  now = 110;
  device.calls[0].resolve();
  await p0;
  assert.equal(cache.getStats().avgAsyncCompileMs, 10);

  // t=130: p1 completes in 30ms -> avg = (10 + 30) / 2 = 20ms.
  now = 130;
  device.calls[1].resolve();
  await p1;
  assert.equal(cache.getStats().avgAsyncCompileMs, 20);

  // t=160: p2 started at t=110, completes in 50ms -> avg = (10 + 30 + 50) / 3 = 30ms.
  now = 160;
  device.calls[2].resolve();
  await p2;
  assert.equal(cache.getStats().avgAsyncCompileMs, 30);

  // Non-monotonic clock jump clamps elapsed time to 0 instead of going negative.
  const p3 = cache.compileRenderPipeline('t3', { label: 't3' });
  now = 140;
  device.calls[3].resolve();
  await p3;
  assert.equal(cache.getStats().avgAsyncCompileMs, 22.5);
});

test('W168: graphics-over-compute priority schedules UI render pipelines ahead of queued compute burst', async () => {
  const device = createManualDevice();
  const cache = new AsyncPipelineCache({ device, maxConcurrentCompiles: 2 });

  // Fill both concurrency slots with background compute pipelines, then queue 3 more compute jobs.
  const cs0 = cache.compileComputePipeline('cs0', { label: 'cs0' });
  const cs1 = cache.compileComputePipeline('cs1', { label: 'cs1' });
  const cs2 = cache.compileComputePipeline('cs2', { label: 'cs2' });
  const cs3 = cache.compileComputePipeline('cs3', { label: 'cs3' });
  const cs4 = cache.compileComputePipeline('cs4', { label: 'cs4' });

  assert.deepEqual(device.calls.map((c) => c.descriptor.label), ['cs0', 'cs1']);
  let stats = cache.getStats();
  assert.equal(stats.activeCompiles, 2);
  assert.equal(stats.peakConcurrentCompiles, 2);
  assert.equal(stats.queuedCompiles, 3);
  assert.equal(stats.priorityPromotions, 0);

  // Start Screen -> UI TEXT SIZE transition bursts 3 render pipelines while compute backlog is queued.
  const uiBg = cache.compileRenderPipeline('ui_bg', { label: 'ui_bg' });
  const uiText = cache.compileRenderPipeline('ui_text', { label: 'ui_text' });
  const uiCursor = cache.compileRenderPipeline('ui_cursor', { label: 'ui_cursor' });

  stats = cache.getStats();
  assert.equal(stats.activeCompiles, 2);
  assert.equal(stats.queuedCompiles, 6);
  assert.equal(stats.priorityPromotions, 3, 'each render pipeline promoted ahead of queued compute jobs');

  // Drain all slots one by one; UI render pipelines must compile in FIFO order before queued compute.
  const dispatchOrder = ['cs0', 'cs1'];
  for (let i = 0; i < 8; i += 1) {
    assert.ok(device.inFlight <= 2, `concurrency cap violated: ${device.inFlight}`);
    const call = device.calls[i];
    call.resolve();
    await tick();
    if (device.calls.length > dispatchOrder.length) {
      dispatchOrder.push(device.calls[device.calls.length - 1].descriptor.label);
    }
  }

  const results = await Promise.all([cs0, cs1, cs2, cs3, cs4, uiBg, uiText, uiCursor]);
  await cache.whenIdle();

  assert.ok(results.every((r) => r.status === PIPELINE_STATUS.READY));
  assert.equal(device.peakInFlight, 2);
  assert.deepEqual(dispatchOrder, [
    'cs0',
    'cs1',
    'ui_bg',
    'ui_text',
    'ui_cursor',
    'cs2',
    'cs3',
    'cs4',
  ]);
  stats = cache.getStats();
  assert.equal(stats.activeCompiles, 0);
  assert.equal(stats.peakConcurrentCompiles, 2);
  assert.equal(stats.queuedCompiles, 0);
  assert.equal(stats.priorityPromotions, 3);
  assert.equal(stats.compiled, 8);
});

test('W168: explicit numeric priority overrides default render/compute order with FIFO tie-breaking', async () => {
  const device = createManualDevice();
  const cache = new AsyncPipelineCache({ device, maxConcurrentCompiles: 1 });

  // Slot 0 occupied by initial compute job.
  const p0 = cache.compileComputePipeline('c0', { label: 'c0' });
  // Queue a mix of default render (10), default compute (0), low-priority render (-5),
  // urgent compute (50), and equal-priority ties (20).
  const p1 = cache.compileRenderPipeline('r_low', { label: 'r_low' }, { priority: -5 });
  const p2 = cache.compileComputePipeline('c_def', { label: 'c_def' });
  const p3 = cache.compileRenderPipeline('r_def', { label: 'r_def' });
  const p4 = cache.compileComputePipeline('c_tie_a', { label: 'c_tie_a' }, { priority: 20 });
  const p5 = cache.compileRenderPipeline('r_tie_b', { label: 'r_tie_b' }, { priority: 20 });
  const p6 = cache.compileComputePipeline('c_urgent', { label: 'c_urgent' }, { priority: 50 });

  assert.equal(device.calls.length, 1);
  assert.equal(cache.getStats().queuedCompiles, 6);
  // Promotions: c_def(0 > -5), r_def(10 > 0), c_tie_a(20 > 10), r_tie_b(20 > 10), c_urgent(50 > 20) => 5
  assert.equal(cache.getStats().priorityPromotions, 5);

  for (let i = 0; i < 7; i += 1) {
    device.calls[i].resolve();
    await tick();
  }
  await Promise.all([p0, p1, p2, p3, p4, p5, p6]);
  await cache.whenIdle();

  assert.deepEqual(
    device.calls.map((c) => c.descriptor.label),
    ['c0', 'c_urgent', 'c_tie_a', 'r_tie_b', 'r_def', 'c_def', 'r_low'],
  );

  // prioritizeGraphics: false disables render-over-compute default while still honoring explicit priority.
  const dev2 = createManualDevice();
  const fifoCache = new AsyncPipelineCache({
    device: dev2,
    maxConcurrentCompiles: 1,
    prioritizeGraphics: false,
  });
  const q0 = fifoCache.compileComputePipeline('a0', { label: 'a0' });
  const q1 = fifoCache.compileComputePipeline('a1', { label: 'a1' });
  const q2 = fifoCache.compileRenderPipeline('a2', { label: 'a2' });
  const q3 = fifoCache.compileComputePipeline('a3_hi', { label: 'a3_hi' }, { priority: 5 });
  assert.equal(fifoCache.getStats().priorityPromotions, 1);
  for (let i = 0; i < 4; i += 1) {
    dev2.calls[i].resolve();
    await tick();
  }
  await Promise.all([q0, q1, q2, q3]);
  assert.deepEqual(
    dev2.calls.map((c) => c.descriptor.label),
    ['a0', 'a3_hi', 'a1', 'a2'],
  );
});

test('W168: duplicate key coalescing promotes queued jobs and isolates render/compute dedupe namespaces', async () => {
  const device = createManualDevice();
  const cache = new AsyncPipelineCache({ device, maxConcurrentCompiles: 1 });

  // Slot 0 occupied by active compile.
  const active = cache.compileRenderPipeline('active', { label: 'active' });
  // Queue a default render job (10), then a lower-priority render job (2) with dedupeKey 'ui-shared',
  // and a compute job with the SAME dedupeKey 'ui-shared'.
  const rMid = cache.compileRenderPipeline('r_mid', { label: 'r_mid' });
  const rLow = cache.compileRenderPipeline('r_low', { label: 'r_low' }, {
    dedupeKey: 'ui-shared',
    priority: 2,
  });
  const cSameKey = cache.compileComputePipeline('c_same_key', { label: 'c_same_key' }, {
    dedupeKey: 'ui-shared',
  });

  assert.equal(cache.getStats().queuedCompiles, 3);
  assert.equal(cache.getStats().priorityPromotions, 0);

  // Coalesce onto queued 'ui-shared' render job with higher priority (25) -> must promote ahead of r_mid.
  const rAlias = cache.compileRenderPipeline('r_low_alias', { label: 'r_low' }, {
    dedupeKey: 'ui-shared',
    priority: 25,
  });
  // Coalesce onto queued compute job with same 'ui-shared' key -> must attach to compute, NOT render!
  const cAlias = cache.compileComputePipeline('c_same_key_alias', { label: 'c_same_key' }, {
    dedupeKey: 'ui-shared',
  });

  // Same-id re-request on queued r_mid with higher priority (30) promotes r_mid ahead of r_low (25)
  // and returns the exact same promise without double-queuing.
  const rMidSame = cache.compileRenderPipeline('r_mid', { label: 'r_mid' }, { priority: 30 });
  assert.strictEqual(rMidSame, rMid);

  let stats = cache.getStats();
  assert.equal(stats.queuedCompiles, 3, 'coalesced duplicates do not double-queue');
  assert.equal(stats.dedupeHits, 2);
  assert.equal(stats.priorityPromotions, 2, 'r_low promoted to 25 and r_mid promoted to 30');

  // Settle all 4 distinct compiles.
  for (let i = 0; i < 4; i += 1) {
    device.calls[i].resolve();
    await tick();
  }
  const [rActive, rMidRes, rLowRes, rAliasRes, cRes, cAliasRes] = await Promise.all([
    active,
    rMid,
    rLow,
    rAlias,
    cSameKey,
    cAlias,
  ]);
  await cache.whenIdle();

  assert.equal(device.calls.length, 4, 'each distinct (kind, dedupeKey) compiled once');
  assert.deepEqual(
    device.calls.map((c) => c.descriptor.label),
    ['active', 'r_mid', 'r_low', 'c_same_key'],
  );
  assert.strictEqual(rLowRes.pipeline, rAliasRes.pipeline);
  assert.equal(rLowRes.pipeline.kind, 'render');
  assert.strictEqual(cRes.pipeline, cAliasRes.pipeline);
  assert.equal(cRes.pipeline.kind, 'compute');
  assert.notStrictEqual(rLowRes.pipeline, cRes.pipeline);
  assert.equal(rActive.status, PIPELINE_STATUS.READY);
  assert.equal(rMidRes.status, PIPELINE_STATUS.READY);
});

test('W168: failure recovery releases slot, clears failed dedupe key, and drains highest-priority queued pipeline', async () => {
  const device = createManualDevice();
  const origRenderAsync = device.createRenderPipelineAsync;
  device.createRenderPipelineAsync = (descriptor) => {
    if (descriptor.label === 'boom-sync') {
      throw new Error('sync driver throw');
    }
    return origRenderAsync(descriptor);
  };

  const cache = new AsyncPipelineCache({ device, maxConcurrentCompiles: 1 });

  // Active compute compile with fallbackDescriptor that will fail both primary and fallback.
  const cFail = cache.compileComputePipeline(
    'c_fail',
    { label: 'c_primary' },
    { dedupeKey: 'bad-cs', fallbackDescriptor: { label: 'c_fallback' } },
  );
  // Queue a compute job first, then a render job that synchronously throws, then a healthy render job.
  const cQueued = cache.compileComputePipeline('c_queued', { label: 'c_queued' });
  const rThrow = cache.compileRenderPipeline('r_throw', { label: 'boom-sync' }, { dedupeKey: 'throw-rs' });
  const rOk = cache.compileRenderPipeline('r_ok', { label: 'r_ok' });

  assert.equal(cache.getStats().activeCompiles, 1);
  assert.equal(cache.getStats().queuedCompiles, 3);
  assert.equal(cache.getStats().priorityPromotions, 2);

  // Reject primary -> fallback runs in the same slot; reject fallback -> slot frees for r_throw (render > compute).
  device.calls[0].reject(new Error('primary rejected'));
  await tick();
  assert.equal(cache.getStats().activeCompiles, 1);
  assert.equal(device.calls[1].descriptor.label, 'c_fallback');

  device.calls[1].reject(new Error('fallback rejected'));
  const [resCFail, resRThrow] = await Promise.all([cFail, rThrow]);
  await tick();

  assert.equal(resCFail.status, PIPELINE_STATUS.FAILED);
  assert.equal(resCFail.retried, true);
  assert.equal(resRThrow.status, PIPELINE_STATUS.FAILED);
  assert.match(resRThrow.error.message, /sync driver throw/);

  // After both failures, r_ok (render) must be dispatched ahead of c_queued (compute).
  assert.equal(device.calls.length, 3);
  assert.equal(device.calls[2].descriptor.label, 'r_ok');
  assert.equal(cache.getStats().activeCompiles, 1);
  assert.equal(cache.getStats().queuedCompiles, 1);
  assert.equal(cache.getStats().failedAsyncCompiles, 2);

  device.calls[2].resolve();
  await rOk;
  assert.equal(device.calls[3].descriptor.label, 'c_queued');

  device.calls[3].resolve();
  await cQueued;

  // Previously failed dedupeKey can be retried cleanly now that it was cleared on failure.
  const cRetry = cache.compileComputePipeline('c_retry', { label: 'c_retry' }, { dedupeKey: 'bad-cs' });
  assert.equal(device.calls.length, 5);
  device.calls[4].resolve();
  assert.equal((await cRetry).status, PIPELINE_STATUS.READY);
  await cache.whenIdle();

  const stats = cache.getStats();
  assert.equal(stats.activeCompiles, 0);
  assert.equal(stats.queuedCompiles, 0);
  assert.equal(stats.peakConcurrentCompiles, 1);
  assert.equal(stats.failedAsyncCompiles, 2);
  assert.equal(stats.compiled, 3);
});

test('W168: compile-stall and latency telemetry tracks activeCompiles, peakConcurrentCompiles, maxCompileMs, avgCompileMs', async () => {
  let now = 1000;
  const device = createManualDevice();
  const cache = new AsyncPipelineCache({
    device,
    maxConcurrentCompiles: 2,
    nowFn: () => now,
  });

  // t=1000: c0 (compute) and c1 (compute) start immediately; c2 (compute) queues; r0 (render) promotes ahead of c2.
  const c0 = cache.compileComputePipeline('c0', { label: 'c0' });
  const c1 = cache.compileComputePipeline('c1', { label: 'c1' }, { fallbackDescriptor: { label: 'c1_fb' } });
  const c2 = cache.compileComputePipeline('c2', { label: 'c2' });
  const r0 = cache.compileRenderPipeline('r0', { label: 'r0' });

  let stats = cache.getStats();
  assert.equal(stats.activeCompiles, 2);
  assert.equal(stats.peakConcurrentCompiles, 2);
  assert.equal(stats.queuedCompiles, 2);
  assert.equal(stats.priorityPromotions, 1);
  assert.equal(stats.maxCompileMs, 0);
  assert.equal(stats.avgCompileMs, 0);

  // t=1005: c0 finishes quickly (5ms) -> r0 launches at t=1005 (not t=1000).
  now = 1005;
  device.calls[0].resolve();
  await c0;
  stats = cache.getStats();
  assert.equal(stats.maxCompileMs, 5);
  assert.equal(stats.avgCompileMs, 5);
  assert.equal(stats.avgAsyncCompileMs, 5);
  assert.equal(stats.activeCompiles, 2);
  assert.equal(stats.queuedCompiles, 1);
  assert.equal(device.calls[2].descriptor.label, 'r0');

  // t=1030: c1 primary rejects -> retries fallback at t=1030 keeping startedAtMs=1000;
  //         r0 finishes in 25ms (1030 - 1005) -> c2 launches at t=1030.
  now = 1030;
  device.calls[1].reject(new Error('layout mismatch'));
  device.calls[2].resolve();
  await r0;
  await tick();
  stats = cache.getStats();
  assert.equal(stats.maxCompileMs, 25);
  assert.equal(stats.avgCompileMs, 15); // (5 + 25) / 2
  assert.equal(stats.activeCompiles, 2);
  assert.equal(stats.queuedCompiles, 0);

  // t=1120: c1 fallback resolves after a 120ms stall spike (1120 - 1000).
  now = 1120;
  const c1FallbackCall = device.calls.find((c) => c.descriptor.label === 'c1_fb');
  c1FallbackCall.resolve();
  await c1;
  stats = cache.getStats();
  assert.equal(stats.maxCompileMs, 120);
  assert.equal(stats.avgCompileMs, 50); // (5 + 25 + 120) / 3
  assert.equal(stats.activeCompiles, 1);

  // t=1140: c2 (started at t=1030) fails after 110ms; maxCompileMs remains 120.
  now = 1140;
  const c2Call = device.calls.find((c) => c.descriptor.label === 'c2');
  c2Call.reject(new Error('compile stall fail'));
  await c2;
  await cache.whenIdle();

  stats = cache.getStats();
  assert.equal(stats.activeCompiles, 0);
  assert.equal(stats.peakConcurrentCompiles, 2);
  assert.equal(stats.queuedCompiles, 0);
  assert.equal(stats.maxCompileMs, 120);
  assert.equal(stats.avgCompileMs, 65); // (5 + 25 + 120 + 110) / 4
  assert.equal(stats.avgAsyncCompileMs, 65);
  assert.equal(stats.priorityPromotions, 1);
  assert.equal(stats.failedAsyncCompiles, 1);
  assert.equal(stats.compiled, 3);
  assert.equal(stats.failed, 1);
});

test('zero unhandled rejections: throwing APIs, callbacks, and logger are contained', async () => {
  const warnings = [];
  const device = createManualDevice();
  device.createComputePipelineAsync = () => {
    throw new Error('sync throw from async API');
  };
  device.createRenderPipelineAsync = () => Promise.reject('string rejection');
  const cache = new AsyncPipelineCache({
    device,
    onReady: () => { throw new Error('onReady boom'); },
    onError: () => Promise.reject(new Error('async onError boom')),
    logger: { warn: (msg) => warnings.push(msg), debug: () => { throw new Error('logger boom'); } },
  });

  const r1 = await cache.compileComputePipeline('c', { label: 'c' });
  assert.equal(r1.status, 'failed');
  assert.equal(r1.error.message, 'sync throw from async API');
  const r2 = await cache.compileRenderPipeline('r', { label: 'r' });
  assert.equal(r2.status, 'failed');
  assert.ok(r2.error instanceof Error);
  assert.equal(r2.error.message, 'string rejection');

  device.createRenderPipelineAsync = async (desc) => ({ label: desc.label });
  const r3 = await cache.compileRenderPipeline('ok', { label: 'ok' });
  assert.equal(r3.status, 'ready', 'a throwing onReady does not fail the compile');
  assert.equal(cache.get('ok').label, 'ok');

  device.createRenderPipelineAsync = async () => undefined;
  const r4 = await cache.compileRenderPipeline('empty', { label: 'empty' });
  assert.equal(r4.status, 'failed', 'resolving no pipeline is a failure');

  // Stale path logs via debug, which throws; must still be contained.
  let rejectLate;
  device.createRenderPipelineAsync = () => new Promise((_, rej) => { rejectLate = rej; });
  const stale = cache.compileRenderPipeline('late', { label: 'late' });
  cache.reset(device);
  rejectLate(new Error('late failure'));
  assert.equal((await stale).status, 'stale');

  await cache.whenIdle();
  await tick();
  await tick();
  assert.ok(warnings.some((w) => /onReady callback threw/.test(w)));
  assert.ok(warnings.some((w) => /onError callback rejected/.test(w)));
  assert.equal(cache.getStats().discardedStale, 1);
  assert.deepEqual(unhandledRejections, []);
});
