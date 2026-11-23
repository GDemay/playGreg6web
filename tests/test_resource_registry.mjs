import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const registryCjs = require('../web/workers/resource_registry.js');
const {
  DEFAULT_FRAMES_IN_FLIGHT,
  DEFAULT_SOFT_BUDGET_BYTES,
  RESOURCE_KIND,
  RESOURCE_KINDS,
  GpuResourceRegistry,
} = registryCjs;

function createDeferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function createDestroyableMock(name, { throwOnDestroy = false } = {}) {
  return {
    name,
    destroyCount: 0,
    destroy() {
      this.destroyCount += 1;
      if (throwOnDestroy) {
        throw new Error(`destroy failed for ${name}`);
      }
    },
  };
}

function createNonDestroyableMock(name) {
  // Mirrors GPUSampler, GPUBindGroup, GPUShaderModule, pipelines, etc.
  return { name };
}

test('UMD/ESM/CommonJS/globalThis exports match expected RESOURCE_KIND enum', async () => {
  const esmMod = await import('../web/workers/resource_registry.js');

  assert.equal(typeof esmMod.GpuResourceRegistry, 'function');
  assert.equal(esmMod.GpuResourceRegistry, GpuResourceRegistry);
  assert.equal(esmMod.RESOURCE_KIND, RESOURCE_KIND);
  assert.equal(globalThis.PlayGta6ResourceRegistry.GpuResourceRegistry, GpuResourceRegistry);
  assert.equal(DEFAULT_FRAMES_IN_FLIGHT, 3);
  assert.ok(DEFAULT_SOFT_BUDGET_BYTES > 0);

  const expectedKinds = [
    'BUFFER',
    'TEXTURE',
    'TEXTURE_VIEW',
    'SAMPLER',
    'BIND_GROUP',
    'BIND_GROUP_LAYOUT',
    'PIPELINE_LAYOUT',
    'SHADER_MODULE',
    'RENDER_PIPELINE',
    'COMPUTE_PIPELINE',
  ];

  for (const key of expectedKinds) {
    assert.ok(typeof RESOURCE_KIND[key] === 'string' && RESOURCE_KIND[key].length > 0, `missing ${key}`);
  }
  assert.equal(RESOURCE_KINDS.length, expectedKinds.length);
  assert.ok(Object.isFrozen(RESOURCE_KIND));
  assert.ok(Object.isFrozen(RESOURCE_KINDS));
});

test('constructor validation rejects invalid options', () => {
  assert.throws(() => new GpuResourceRegistry(null), TypeError);
  assert.throws(() => new GpuResourceRegistry({ framesInFlight: 0 }), RangeError);
  assert.throws(() => new GpuResourceRegistry({ framesInFlight: -2 }), RangeError);
  assert.throws(() => new GpuResourceRegistry({ framesInFlight: 1.5 }), RangeError);
  assert.throws(() => new GpuResourceRegistry({ softBudgetBytes: 0 }), RangeError);
  assert.throws(() => new GpuResourceRegistry({ softBudgetBytes: -10 }), RangeError);
  assert.throws(() => new GpuResourceRegistry({ onBudgetWarning: 'nope' }), TypeError);
  assert.throws(() => new GpuResourceRegistry({ onDestroyError: 123 }), TypeError);

  const unlimited = new GpuResourceRegistry({ softBudgetBytes: null });
  assert.equal(unlimited.softBudgetBytes, null);

  const infBudget = new GpuResourceRegistry({ softBudgetBytes: Infinity });
  assert.equal(infBudget.softBudgetBytes, null);
});

test('register, get, has, getEntry, list across kinds and input validation', () => {
  const reg = new GpuResourceRegistry();

  const buf1 = createDestroyableMock('vertex-buf');
  const tex1 = createDestroyableMock('albedo-tex');
  const sampler1 = createNonDestroyableMock('linear-sampler');

  const returned = reg.register(RESOURCE_KIND.BUFFER, 1, buf1, {
    byteSize: 4096,
    label: 'vbuf',
    tag: 'mesh',
  });
  assert.equal(returned, buf1);

  // Same numeric handle in a different kind does not collide.
  reg.register(RESOURCE_KIND.TEXTURE, 1, tex1, { byteSize: 65536, label: 'albedo' });
  reg.register(RESOURCE_KIND.SAMPLER, 7, sampler1, { label: 'aniso' });

  assert.equal(reg.has(RESOURCE_KIND.BUFFER, 1), true);
  assert.equal(reg.has(RESOURCE_KIND.TEXTURE, 1), true);
  assert.equal(reg.has(RESOURCE_KIND.SAMPLER, 7), true);
  assert.equal(reg.has(RESOURCE_KIND.BUFFER, 7), false);

  assert.equal(reg.get(RESOURCE_KIND.BUFFER, 1), buf1);
  assert.equal(reg.get(RESOURCE_KIND.TEXTURE, 1), tex1);
  assert.equal(reg.get(RESOURCE_KIND.SAMPLER, 7), sampler1);
  assert.equal(reg.get(RESOURCE_KIND.BUFFER, 999), undefined);

  const entry = reg.getEntry(RESOURCE_KIND.BUFFER, 1);
  assert.deepEqual(entry, {
    kind: RESOURCE_KIND.BUFFER,
    handle: 1,
    resource: buf1,
    byteSize: 4096,
    label: 'vbuf',
    tag: 'mesh',
    registeredAtFrame: 0,
  });

  const bufferList = reg.list(RESOURCE_KIND.BUFFER);
  assert.equal(bufferList.length, 1);
  assert.equal(bufferList[0].resource, buf1);

  const allList = reg.list();
  assert.equal(allList.length, 3);

  const stats = reg.getStats();
  assert.equal(stats.countsByKind[RESOURCE_KIND.BUFFER], 1);
  assert.equal(stats.countsByKind[RESOURCE_KIND.TEXTURE], 1);
  assert.equal(stats.countsByKind[RESOURCE_KIND.SAMPLER], 1);
  assert.equal(stats.countsByKind[RESOURCE_KIND.RENDER_PIPELINE], 0);
  assert.equal(stats.activeCount, 3);
  assert.equal(stats.activeBytes, 4096 + 65536);
  assert.equal(stats.pendingRetireBytes, 0);
  assert.equal(stats.peakBytes, 4096 + 65536);
  assert.equal(stats.totalDestroyed, 0);
  assert.equal(stats.currentFrame, 0);

  // Validation checks
  assert.throws(() => reg.register('invalidKind', 1, buf1), RangeError);
  assert.throws(() => reg.register(RESOURCE_KIND.BUFFER, -1, buf1), RangeError);
  assert.throws(() => reg.register(RESOURCE_KIND.BUFFER, 1.5, buf1), RangeError);
  assert.throws(() => reg.register(RESOURCE_KIND.BUFFER, NaN, buf1), RangeError);
  assert.throws(() => reg.register(RESOURCE_KIND.BUFFER, '1', buf1), RangeError);
  assert.throws(() => reg.register(RESOURCE_KIND.BUFFER, 2, null), TypeError);
  assert.throws(() => reg.register(RESOURCE_KIND.BUFFER, 2, buf1, { byteSize: -4 }), RangeError);
  assert.throws(() => reg.register(RESOURCE_KIND.BUFFER, 2, buf1, { label: 42 }), TypeError);
  assert.throws(() => reg.get('bogus', 1), RangeError);
  assert.throws(() => reg.has(RESOURCE_KIND.BUFFER, -5), RangeError);
  assert.throws(() => reg.list('bogus'), RangeError);
});

test('overwriting an existing (kind, handle) retires prior resource cleanly without leaking or premature destroy', () => {
  const reg = new GpuResourceRegistry({ framesInFlight: 3 });

  const oldBuf = createDestroyableMock('old-buf');
  const newBuf = createDestroyableMock('new-buf');

  reg.beginFrame(1);
  reg.register(RESOURCE_KIND.BUFFER, 10, oldBuf, { byteSize: 1024, label: 'v1' });

  // Overwrite handle 10 with a brand-new buffer.
  reg.register(RESOURCE_KIND.BUFFER, 10, newBuf, { byteSize: 2048, label: 'v2' });

  // Active table points to newBuf immediately; oldBuf is deferred, not immediately destroyed.
  assert.equal(reg.get(RESOURCE_KIND.BUFFER, 10), newBuf);
  assert.equal(oldBuf.destroyCount, 0);
  assert.equal(newBuf.destroyCount, 0);

  let stats = reg.getStats();
  assert.equal(stats.countsByKind[RESOURCE_KIND.BUFFER], 1);
  assert.equal(stats.pendingRetireCount, 1);
  assert.equal(stats.activeBytes, 2048);
  assert.equal(stats.pendingRetireBytes, 1024);
  assert.equal(stats.peakBytes, 3072);

  // Re-registering the exact same object updates metadata in place without retiring it.
  reg.register(RESOURCE_KIND.BUFFER, 10, newBuf, { byteSize: 4096, label: 'v2-resized' });
  assert.equal(reg.getStats().pendingRetireCount, 1);
  assert.equal(reg.getStats().activeBytes, 4096);
  assert.equal(reg.getEntry(RESOURCE_KIND.BUFFER, 10).label, 'v2-resized');

  // Advance framesInFlight frames -> oldBuf destroyed, newBuf untouched.
  reg.endFrame();
  reg.beginFrame(2);
  assert.equal(oldBuf.destroyCount, 0);
  reg.beginFrame(3);
  assert.equal(oldBuf.destroyCount, 0);
  reg.beginFrame(4);
  assert.equal(oldBuf.destroyCount, 1);
  assert.equal(newBuf.destroyCount, 0);

  stats = reg.getStats();
  assert.equal(stats.pendingRetireCount, 0);
  assert.equal(stats.pendingRetireBytes, 0);
  assert.equal(stats.activeBytes, 4096);
  assert.equal(stats.totalDestroyed, 1);
});

test('deferred retirement across framesInFlight frames', () => {
  const reg = new GpuResourceRegistry({ framesInFlight: 3 });

  const b1 = createDestroyableMock('b1');
  const b2 = createDestroyableMock('b2');

  reg.beginFrame(10);
  reg.register(RESOURCE_KIND.BUFFER, 1, b1, { byteSize: 512 });
  reg.register(RESOURCE_KIND.BUFFER, 2, b2, { byteSize: 1024 });

  // Retire b1 on frame 10 -> immediately removed from active lookups.
  assert.equal(reg.retire(RESOURCE_KIND.BUFFER, 1), true);
  assert.equal(reg.retire(RESOURCE_KIND.BUFFER, 1), false); // Already retired
  assert.equal(reg.has(RESOURCE_KIND.BUFFER, 1), false);
  assert.equal(reg.get(RESOURCE_KIND.BUFFER, 1), undefined);
  assert.equal(b1.destroyCount, 0);

  let stats = reg.getStats();
  assert.equal(stats.activeBytes, 1024);
  assert.equal(stats.pendingRetireBytes, 512);
  assert.equal(stats.pendingRetireCount, 1);

  reg.endFrame();

  // Frame 11: retire b2 (elapsed for b1 = 1 < 3)
  let f = reg.beginFrame();
  assert.equal(f.frameIndex, 11);
  assert.equal(f.destroyedCount, 0);
  assert.equal(b1.destroyCount, 0);
  assert.equal(reg.retire(RESOURCE_KIND.BUFFER, 2), true);
  reg.endFrame();

  // Frame 12: elapsed b1=2, b2=1 -> neither destroyed
  f = reg.beginFrame();
  assert.equal(f.frameIndex, 12);
  assert.equal(f.destroyedCount, 0);
  assert.equal(b1.destroyCount, 0);
  assert.equal(b2.destroyCount, 0);
  reg.endFrame();

  // Frame 13: elapsed b1=3 -> b1 destroyed; b2=2 -> still pending
  f = reg.beginFrame();
  assert.equal(f.frameIndex, 13);
  assert.equal(f.destroyedCount, 1);
  assert.equal(f.freedBytes, 512);
  assert.equal(f.remainingCount, 1);
  assert.equal(b1.destroyCount, 1);
  assert.equal(b2.destroyCount, 0);
  assert.equal(reg.getStats().pendingRetireBytes, 1024);
  reg.endFrame();

  // Frame 14: elapsed b2=3 -> b2 destroyed
  f = reg.beginFrame();
  assert.equal(f.frameIndex, 14);
  assert.equal(f.destroyedCount, 1);
  assert.equal(f.freedBytes, 1024);
  assert.equal(f.remainingCount, 0);
  assert.equal(b2.destroyCount, 1);

  stats = reg.getStats();
  assert.equal(stats.activeBytes, 0);
  assert.equal(stats.pendingRetireBytes, 0);
  assert.equal(stats.totalDestroyed, 2);
  assert.equal(stats.currentFrame, 14);

  // Frame index cannot go backwards.
  assert.throws(() => reg.beginFrame(13), RangeError);
});

test('fence-driven early collection destroys retired resources before framesInFlight elapses', async () => {
  const reg = new GpuResourceRegistry({ framesInFlight: 3 });
  const tex = createDestroyableMock('streaming-tex');
  const fence = createDeferred();

  reg.beginFrame(1);
  reg.register(RESOURCE_KIND.TEXTURE, 42, tex, { byteSize: 16384 });
  reg.retire(RESOURCE_KIND.TEXTURE, 42);

  const ended = reg.endFrame(fence.promise);
  assert.equal(ended.sealedCount, 1);
  assert.equal(tex.destroyCount, 0);

  // Fence has not resolved yet -> collect() keeps it pending.
  assert.equal(reg.collect().destroyedCount, 0);
  assert.equal(tex.destroyCount, 0);

  // Resolve GPU fence while still on frame 1 (0 frames elapsed < framesInFlight).
  fence.resolve();
  await fence.promise;
  // Let internal .then() reaction run.
  await Promise.resolve();

  const collected = reg.collect();
  assert.equal(collected.destroyedCount, 1);
  assert.equal(collected.freedBytes, 16384);
  assert.equal(collected.remainingCount, 0);
  assert.equal(tex.destroyCount, 1);

  const stats = reg.getStats();
  assert.equal(stats.pendingRetireBytes, 0);
  assert.equal(stats.totalDestroyed, 1);
});

test('fence-driven gated collection holds resources while fence is pending, supports force and rejected fence fallback', async () => {
  const reg = new GpuResourceRegistry({ framesInFlight: 2 });
  const bufSlow = createDestroyableMock('slow-gpu-buf');
  const bufRejected = createDestroyableMock('rejected-fence-buf');
  const bufForced = createDestroyableMock('forced-buf');

  const slowFence = createDeferred();
  const rejectedFence = createDeferred();
  const neverFence = createDeferred();

  // Frame 1: retire bufSlow bound to slowFence
  reg.beginFrame(1);
  reg.register(RESOURCE_KIND.BUFFER, 1, bufSlow, { byteSize: 400 });
  reg.retire(RESOURCE_KIND.BUFFER, 1);
  reg.endFrame(slowFence.promise);

  // Frame 2: retire bufRejected bound to rejectedFence
  reg.beginFrame(2);
  reg.register(RESOURCE_KIND.BUFFER, 2, bufRejected, { byteSize: 600 });
  reg.retire(RESOURCE_KIND.BUFFER, 2);
  reg.endFrame(rejectedFence.promise);

  // Advance well past framesInFlight (frame 5) while slowFence is still pending.
  reg.beginFrame(3);
  reg.beginFrame(4);
  reg.beginFrame(5);

  // Both are still gated because their fences have not resolved.
  assert.equal(bufSlow.destroyCount, 0);
  assert.equal(bufRejected.destroyCount, 0);
  assert.equal(reg.getStats().pendingRetireBytes, 1000);

  // Reject rejectedFence -> must NOT produce unhandled rejection and falls back to epoch.
  rejectedFence.reject(new Error('device lost'));
  await rejectedFence.promise.catch(() => {});
  await Promise.resolve();

  let col = reg.collect();
  assert.equal(col.destroyedCount, 1);
  assert.equal(bufRejected.destroyCount, 1);
  assert.equal(bufSlow.destroyCount, 0);

  // Resolve slowFence -> bufSlow is now collected on next collect()/beginFrame().
  slowFence.resolve();
  await slowFence.promise;
  await Promise.resolve();

  col = reg.flush(); // alias for collect()
  assert.equal(col.destroyedCount, 1);
  assert.equal(bufSlow.destroyCount, 1);
  assert.equal(reg.getStats().pendingRetireBytes, 0);

  // Force collection bypasses an unresolved fence.
  reg.register(RESOURCE_KIND.BUFFER, 3, bufForced, { byteSize: 800 });
  reg.retire(RESOURCE_KIND.BUFFER, 3);
  reg.endFrame(neverFence.promise);
  assert.equal(reg.collect(false).destroyedCount, 0);
  assert.equal(bufForced.destroyCount, 0);

  col = reg.collect(true);
  assert.equal(col.destroyedCount, 1);
  assert.equal(col.freedBytes, 800);
  assert.equal(bufForced.destroyCount, 1);

  assert.throws(() => reg.endFrame('not-a-promise'), TypeError);
});

test('VRAM soft budget warning fires once per threshold crossing and re-arms after reclamation', () => {
  const warnings = [];
  const reg = new GpuResourceRegistry({
    framesInFlight: 2,
    softBudgetBytes: 1000,
    onBudgetWarning: (info) => warnings.push(info),
  });

  const b1 = createDestroyableMock('b1');
  const b2 = createDestroyableMock('b2');
  const b3 = createDestroyableMock('b3');
  const b4 = createDestroyableMock('b4');

  reg.beginFrame(1);
  // Exactly at budget (1000 <= 1000) -> no warning.
  reg.register(RESOURCE_KIND.BUFFER, 1, b1, { byteSize: 1000, label: 'exact' });
  assert.equal(warnings.length, 0);
  assert.equal(reg.getStats().budgetExceeded, false);

  // Crossing above 1000 -> fires once.
  reg.register(RESOURCE_KIND.BUFFER, 2, b2, { byteSize: 200, label: 'over-1' });
  assert.equal(warnings.length, 1);
  assert.equal(warnings[0].totalBytes, 1200);
  assert.equal(warnings[0].softBudgetBytes, 1000);
  assert.equal(warnings[0].label, 'over-1');
  assert.equal(warnings[0].kind, RESOURCE_KIND.BUFFER);
  assert.equal(warnings[0].handle, 2);
  assert.equal(reg.getStats().budgetExceeded, true);
  assert.equal(reg.getStats().budgetWarningCount, 1);

  // Another allocation while still above budget -> does NOT re-fire.
  reg.register(RESOURCE_KIND.BUFFER, 3, b3, { byteSize: 500, label: 'over-2' });
  assert.equal(warnings.length, 1);
  assert.equal(reg.getStats().peakBytes, 1700);

  // Retiring alone does NOT free VRAM yet, so budget remains exceeded until collected.
  reg.retire(RESOURCE_KIND.BUFFER, 1);
  assert.equal(reg.getStats().activeBytes, 700);
  assert.equal(reg.getStats().pendingRetireBytes, 1000);
  assert.equal(reg.getStats().totalBytes, 1700);

  // Collecting drops totalBytes back to 700 <= 1000 -> re-arms threshold.
  reg.beginFrame(2);
  reg.beginFrame(3);
  assert.equal(b1.destroyCount, 1);
  assert.equal(reg.getStats().totalBytes, 700);
  assert.equal(reg.getStats().budgetExceeded, false);

  // Crossing above 1000 again -> fires a second time.
  reg.register(RESOURCE_KIND.BUFFER, 4, b4, { byteSize: 400, label: 'over-again' });
  assert.equal(warnings.length, 2);
  assert.equal(warnings[1].totalBytes, 1100);
  assert.equal(warnings[1].label, 'over-again');
  assert.equal(reg.getStats().budgetWarningCount, 2);
});

test('handles resources without .destroy() and guards against .destroy() errors & double-destroy', () => {
  const destroyErrors = [];
  const reg = new GpuResourceRegistry({
    framesInFlight: 1,
    onDestroyError: (err, info) => destroyErrors.push({ err, info }),
  });

  const sampler = createNonDestroyableMock('sampler-0');
  const bindGroup = createNonDestroyableMock('bg-0');
  const shader = createNonDestroyableMock('wgsl-0');
  const pipeline = createNonDestroyableMock('pso-0');
  const failingBuf = createDestroyableMock('bad-buf', { throwOnDestroy: true });
  const healthyBuf = createDestroyableMock('good-buf');

  reg.beginFrame(1);
  reg.register(RESOURCE_KIND.SAMPLER, 1, sampler);
  reg.register(RESOURCE_KIND.BIND_GROUP, 2, bindGroup);
  reg.register(RESOURCE_KIND.SHADER_MODULE, 3, shader, { byteSize: 256 });
  reg.register(RESOURCE_KIND.RENDER_PIPELINE, 4, pipeline);
  reg.register(RESOURCE_KIND.BUFFER, 5, failingBuf, { byteSize: 128, label: 'bad' });
  reg.register(RESOURCE_KIND.BUFFER, 6, healthyBuf, { byteSize: 512, label: 'good' });

  // Also test registering the same underlying buffer object under a second handle:
  // double-destroy guard must ensure healthyBuf.destroy() is invoked only once.
  reg.register(RESOURCE_KIND.BUFFER, 7, healthyBuf, { byteSize: 0, label: 'alias' });

  reg.retire(RESOURCE_KIND.SAMPLER, 1);
  reg.retire(RESOURCE_KIND.BIND_GROUP, 2);
  reg.retire(RESOURCE_KIND.SHADER_MODULE, 3);
  reg.retire(RESOURCE_KIND.RENDER_PIPELINE, 4);
  reg.retire(RESOURCE_KIND.BUFFER, 5);
  reg.retire(RESOURCE_KIND.BUFFER, 6);
  reg.retire(RESOURCE_KIND.BUFFER, 7);

  // Next frame collects everything; failingBuf throws inside destroy(),
  // yet healthyBuf and all other entries are still collected cleanly.
  const res = reg.beginFrame(2);
  assert.equal(res.destroyedCount, 7);
  assert.equal(res.freedBytes, 256 + 128 + 512);
  assert.equal(res.remainingCount, 0);

  assert.equal(failingBuf.destroyCount, 1);
  assert.equal(healthyBuf.destroyCount, 1);
  assert.equal(destroyErrors.length, 1);
  assert.equal(destroyErrors[0].info.handle, 5);
  assert.equal(destroyErrors[0].info.label, 'bad');

  const stats = reg.getStats();
  assert.equal(stats.destroyErrors, 1);
  assert.equal(stats.totalDestroyed, 7);
  assert.equal(stats.activeBytes, 0);
  assert.equal(stats.pendingRetireBytes, 0);
});

test('retireHandle retires matching numeric handle across all kinds', () => {
  const reg = new GpuResourceRegistry({ framesInFlight: 1 });
  const tex = createDestroyableMock('tex-9');
  const view = createNonDestroyableMock('view-9');
  const other = createDestroyableMock('buf-10');

  reg.register(RESOURCE_KIND.TEXTURE, 9, tex, { byteSize: 4096 });
  reg.register(RESOURCE_KIND.TEXTURE_VIEW, 9, view);
  reg.register(RESOURCE_KIND.BUFFER, 10, other, { byteSize: 256 });

  const retired = reg.retireHandle(9);
  assert.equal(retired, 2);
  assert.equal(reg.has(RESOURCE_KIND.TEXTURE, 9), false);
  assert.equal(reg.has(RESOURCE_KIND.TEXTURE_VIEW, 9), false);
  assert.equal(reg.has(RESOURCE_KIND.BUFFER, 10), true);

  reg.beginFrame(1);
  assert.equal(tex.destroyCount, 1);
  assert.equal(other.destroyCount, 0);
});

test('destroyAll() is idempotent and destroys both active and pending resources safely', async () => {
  const reg = new GpuResourceRegistry({ framesInFlight: 3, softBudgetBytes: 500 });

  const activeBuf = createDestroyableMock('active-buf');
  const pendingTex = createDestroyableMock('pending-tex');
  const activeSampler = createNonDestroyableMock('active-sampler');
  const pendingFence = createDeferred();

  reg.beginFrame(1);
  reg.register(RESOURCE_KIND.BUFFER, 1, activeBuf, { byteSize: 300 });
  reg.register(RESOURCE_KIND.TEXTURE, 2, pendingTex, { byteSize: 400 });
  reg.register(RESOURCE_KIND.SAMPLER, 3, activeSampler);

  // Move pendingTex into deferred retirement queue bound to an unresolved fence.
  reg.retire(RESOURCE_KIND.TEXTURE, 2);
  reg.endFrame(pendingFence.promise);

  assert.equal(reg.getStats().budgetExceeded, true);
  assert.equal(activeBuf.destroyCount, 0);
  assert.equal(pendingTex.destroyCount, 0);

  // First destroyAll() tears down everything (active + pending).
  const first = reg.destroyAll();
  assert.equal(first.destroyedCount, 3);
  assert.equal(first.freedBytes, 700);
  assert.equal(activeBuf.destroyCount, 1);
  assert.equal(pendingTex.destroyCount, 1);

  assert.equal(reg.list().length, 0);
  assert.equal(reg.has(RESOURCE_KIND.BUFFER, 1), false);
  assert.equal(reg.getStats().activeCount, 0);
  assert.equal(reg.getStats().pendingRetireCount, 0);
  assert.equal(reg.getStats().activeBytes, 0);
  assert.equal(reg.getStats().pendingRetireBytes, 0);
  assert.equal(reg.getStats().budgetExceeded, false);
  assert.equal(reg.getStats().totalDestroyed, 3);

  // Late resolution of a pre-destroyAll fence is safely ignored.
  pendingFence.resolve();
  await pendingFence.promise;
  await Promise.resolve();
  assert.equal(reg.collect().destroyedCount, 0);

  // Second destroyAll() is a safe idempotent no-op.
  const second = reg.destroyAll();
  assert.equal(second.destroyedCount, 0);
  assert.equal(second.freedBytes, 0);
  assert.equal(activeBuf.destroyCount, 1);
  assert.equal(pendingTex.destroyCount, 1);
  assert.equal(reg.getStats().totalDestroyed, 3);

  // Registry can be reused cleanly after device.lost recovery.
  const recoveredBuf = createDestroyableMock('recovered-buf');
  reg.register(RESOURCE_KIND.BUFFER, 1, recoveredBuf, { byteSize: 128 });
  assert.equal(reg.get(RESOURCE_KIND.BUFFER, 1), recoveredBuf);
  assert.equal(reg.getStats().activeBytes, 128);
});
