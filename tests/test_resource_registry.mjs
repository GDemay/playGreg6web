import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const WORKERS_DIR = fileURLToPath(new URL('../web/workers/', import.meta.url));
const registryCjs = require('../web/workers/resource_registry.js');
const {
  DEFAULT_FRAMES_IN_FLIGHT,
  DEFAULT_SOFT_BUDGET_BYTES,
  CREATE_TEXTURE2_OPCODE,
  CREATE_TEXTURE2_PAYLOAD_BYTES,
  MAX_TEXTURE_VIEW_FORMATS,
  GPU_BUFFER_USAGE,
  GPU_TEXTURE_USAGE,
  TEXTURE_FORMAT_NAMES,
  BC_TEXTURE_FORMATS,
  BC_FALLBACK_FORMAT,
  TEXTURE_COMPRESSION_BC_FEATURE,
  RESOURCE_KIND,
  RESOURCE_KINDS,
  isBcFormat,
  deviceSupportsBc,
  computeMaxMipLevels,
  clampMipLevelCount,
  clampSampleCount,
  alignBcDimension,
  alignBufferSize,
  padBufferUploadData,
  estimateTextureBytes,
  decodeCreateTexture2,
  sanitizeTextureDescriptor,
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

// ---------------------------------------------------------------------------
// G41 Hardening Tests: CreateTexture2 (ABI v2 opcode 35), BC1..BC7 fallback &
// 4x4 alignment, mip/sampleCount clamping, 4-byte buffer alignment/padding,
// SharedArrayBuffer upload safety, and classic-worker importScripts isolation.
// ---------------------------------------------------------------------------

function createMockWebGpuDevice({ features = undefined, throwOnCreateTexture = false } = {}) {
  const createdTextures = [];
  const createdBuffers = [];
  const writeBuffers = [];
  let nextId = 1;

  const device = {
    createTexture(desc) {
      if (throwOnCreateTexture) {
        throw new Error('simulated WebGPU createTexture failure');
      }
      const tex = {
        kind: 'GPUTexture',
        internalId: nextId++,
        desc: {
          ...desc,
          size: { ...desc.size },
          ...(desc.viewFormats ? { viewFormats: desc.viewFormats.slice() } : {}),
        },
        destroyCount: 0,
        createView(viewDesc) {
          return {
            kind: 'GPUTextureView',
            textureInternalId: tex.internalId,
            viewDesc: viewDesc || null,
          };
        },
        destroy() {
          this.destroyCount += 1;
        },
      };
      createdTextures.push(tex);
      return tex;
    },
    createBuffer(desc) {
      const buf = {
        kind: 'GPUBuffer',
        internalId: nextId++,
        size: desc.size,
        usage: desc.usage,
        label: desc.label || '',
        mappedAtCreation: Boolean(desc.mappedAtCreation),
        destroyCount: 0,
        destroy() {
          this.destroyCount += 1;
        },
      };
      createdBuffers.push(buf);
      return buf;
    },
    queue: {
      writeBuffer(buffer, dstOffset, data) {
        writeBuffers.push({
          buffer,
          dstOffset,
          data,
          byteLength: data.byteLength,
          isShared: typeof SharedArrayBuffer !== 'undefined' && data.buffer instanceof SharedArrayBuffer,
          bytesCopy: new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice(),
        });
      },
    },
  };

  if (features !== undefined) {
    device.features = features;
  }

  return { device, createdTextures, createdBuffers, writeBuffers };
}

function buildCreateTexture2Payload({
  id = 1,
  width = 256,
  height = 256,
  depthOrLayers = 1,
  mipLevels = 1,
  sampleCount = 1,
  usageFlags = GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.COPY_DST,
  dimension = 1, // 2d
  format = 22, // rgba8unorm
  viewFormats = [],
  viewFormatCount = undefined,
} = {}) {
  const buf = new ArrayBuffer(CREATE_TEXTURE2_PAYLOAD_BYTES);
  const view = new DataView(buf);
  view.setUint32(0, id, true);
  view.setUint32(4, width, true);
  view.setUint32(8, height, true);
  view.setUint32(12, depthOrLayers, true);
  view.setUint32(16, mipLevels, true);
  view.setUint32(20, sampleCount, true);
  view.setUint32(24, usageFlags, true);
  view.setUint8(28, dimension);
  view.setUint8(29, format);
  view.setUint8(30, viewFormatCount !== undefined ? viewFormatCount : viewFormats.length);
  view.setUint8(31, 0);
  for (let i = 0; i < Math.min(viewFormats.length, MAX_TEXTURE_VIEW_FORMATS); i += 1) {
    view.setUint8(32 + i, viewFormats[i]);
  }
  return view;
}

test('G41 classic-worker importScripts IIFE isolation loads resource_registry.js twice cleanly with zero top-level leaks', () => {
  const sandbox = { console, TextEncoder, TextDecoder };
  const vmCtx = vm.createContext(sandbox);
  vmCtx.self = vmCtx;
  vmCtx.importScripts = (...urls) => {
    for (const rel of urls) {
      const fullPath = path.resolve(WORKERS_DIR, rel);
      const source = readFileSync(fullPath, 'utf8');
      vm.runInContext(source, vmCtx, { filename: fullPath });
    }
  };

  assert.doesNotThrow(() => {
    vmCtx.importScripts('./texture_formats.js', './resource_registry.js');
    vmCtx.importScripts('./resource_registry.js');
  });

  assert.doesNotThrow(() => {
    vm.runInContext(
      `
      const TEXTURE_FORMAT_NAMES = [];
      const TEXTURE_DIMENSION_NAMES = [];
      const BC_TEXTURE_FORMATS = [];
      const GPU_BUFFER_USAGE = {};
      const GPU_TEXTURE_USAGE = {};
      const RESOURCE_KIND = {};
      class GpuResourceRegistry {}
      `,
      vmCtx
    );
  });

  assert.equal(vmCtx.PlayGta6ResourceRegistry.CREATE_TEXTURE2_OPCODE, 35);
  assert.equal(typeof vmCtx.PlayGta6ResourceRegistry.GpuResourceRegistry, 'function');
  assert.equal(typeof vmCtx.PlayGta6ResourceRegistry.sanitizeTextureDescriptor, 'function');
});

test('G41 CreateTexture2 (ABI v2 opcode 35) decodes binary payload & structured descriptors and tracks VRAM + deferred GC', () => {
  assert.equal(CREATE_TEXTURE2_OPCODE, 35);
  assert.equal(CREATE_TEXTURE2_PAYLOAD_BYTES, 40);

  const { device, createdTextures } = createMockWebGpuDevice({
    features: new Set([TEXTURE_COMPRESSION_BC_FEATURE]),
  });
  const reg = new GpuResourceRegistry({ framesInFlight: 2, device });

  // Binary CreateTexture2 payload: 64x32x6 array texture, rgba16float (code 40), 4 mips, sampleCount 1, viewFormats [40]
  const payload = buildCreateTexture2Payload({
    id: 77,
    width: 64,
    height: 32,
    depthOrLayers: 6,
    mipLevels: 4,
    sampleCount: 1,
    usageFlags: GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.COPY_DST,
    dimension: 1,
    format: 40,
    viewFormats: [40],
  });

  const res = reg.createTexture2(payload, 0, CREATE_TEXTURE2_PAYLOAD_BYTES);
  assert.equal(res.error, undefined);
  assert.equal(res.id, 77);
  assert.equal(res.texture, createdTextures[0]);
  assert.equal(reg.get(RESOURCE_KIND.TEXTURE, 77), createdTextures[0]);
  assert.equal(reg.get(RESOURCE_KIND.TEXTURE_VIEW, 77), res.view);

  assert.deepEqual(createdTextures[0].desc, {
    size: { width: 64, height: 32, depthOrArrayLayers: 6 },
    mipLevelCount: 4,
    sampleCount: 1,
    dimension: '2d',
    format: 'rgba16float',
    usage: GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.COPY_DST,
    viewFormats: ['rgba16float'],
  });
  assert.equal(res.meta.isTexture2, true);
  assert.equal(res.meta.byteSize, estimateTextureBytes(64, 32, 6, 4, 'rgba16float', '2d', 1));
  assert.equal(reg.getStats().activeBytes, res.meta.byteSize);

  // Structured CreateTexture2 call with flat ABI fields (arrayLayers, mipLevels, sampleCount, numeric format)
  const tex2 = reg.createTexture2(77, {
    width: 32,
    height: 32,
    arrayLayers: 4,
    mipLevels: 3,
    sampleCount: 1,
    dimension: '2d',
    format: 22, // rgba8unorm
    usage: GPU_TEXTURE_USAGE.TEXTURE_BINDING,
  });
  assert.equal(tex2, createdTextures[1]);
  assert.deepEqual(createdTextures[1].desc.size, { width: 32, height: 32, depthOrArrayLayers: 4 });
  assert.equal(createdTextures[1].desc.mipLevelCount, 3);

  // Previous texture on handle 77 was cleanly retired (deferred, not immediately destroyed).
  assert.equal(createdTextures[0].destroyCount, 0);
  reg.beginFrame(1);
  assert.equal(createdTextures[0].destroyCount, 0);
  reg.beginFrame(2);
  assert.equal(createdTextures[0].destroyCount, 1);

  // Invalid CreateTexture2 binary payloads return structured errors instead of throwing.
  const truncated = new DataView(new ArrayBuffer(20));
  assert.match(reg.createTexture2(truncated).error, /payload 20 < 40/);

  const badFmt = buildCreateTexture2Payload({ id: 80, format: 0 });
  assert.match(reg.createTexture2(badFmt).error, /invalid texture format code 0/);

  const badDim = buildCreateTexture2Payload({ id: 81, dimension: 9 });
  assert.match(reg.createTexture2(badDim).error, /invalid dimension 9/);

  const badVfCount = buildCreateTexture2Payload({ id: 82, viewFormatCount: 9 });
  assert.match(reg.createTexture2(badVfCount).error, /view_format_count 9 > 8/);

  const badVfCode = buildCreateTexture2Payload({ id: 83, viewFormats: [0], viewFormatCount: 1 });
  assert.match(reg.createTexture2(badVfCode).error, /invalid view format code 0/);

  // Device error inside createTexture2FromPacket is captured cleanly.
  const failing = createMockWebGpuDevice({ throwOnCreateTexture: true });
  const failingReg = new GpuResourceRegistry({ device: failing.device });
  const failRes = failingReg.createTexture2(payload);
  assert.match(failRes.error, /simulated WebGPU createTexture failure/);
});

test('G41 mipLevelCount clamping for 1D/2D/2D-array/3D textures tracks mipClampCount accurately', () => {
  assert.equal(computeMaxMipLevels(128, 64, 1, '2d'), 8);
  assert.equal(computeMaxMipLevels(16, 8, 64, '2d'), 5); // arrayLayers do not increase 2D max mips
  assert.equal(computeMaxMipLevels(16, 8, 64, '3d'), 7); // 3D depth DOES contribute to max mips
  assert.equal(computeMaxMipLevels(32, 999, 1, '1d'), 6); // 1D ignores height

  const { device, createdTextures } = createMockWebGpuDevice();
  const warnings = [];
  const reg = new GpuResourceRegistry({
    device,
    onSanitizeWarning: (w) => warnings.push(w),
  });

  // 1. Over-limit 2D mips (128x64 -> max 8, requested 12 -> clamped to 8)
  reg.createTexture(1, {
    size: { width: 128, height: 64, depthOrArrayLayers: 1 },
    mipLevelCount: 12,
    format: 'rgba8unorm',
    usage: GPU_TEXTURE_USAGE.TEXTURE_BINDING,
  });
  assert.equal(createdTextures[0].desc.mipLevelCount, 8);
  assert.equal(reg.getTextureMeta(1).mipClamped, true);
  assert.equal(reg.getStats().mipClampCount, 1);

  // 2. Zero / negative mips clamped to 1
  reg.createTexture(2, {
    width: 64,
    height: 64,
    mipLevels: 0,
    format: 'rgba8unorm',
  });
  assert.equal(createdTextures[1].desc.mipLevelCount, 1);
  assert.equal(reg.getStats().mipClampCount, 2);

  // 3. 3D texture uses max(w, h, d): 16x8x64 -> max 7; requesting 7 is NOT clamped
  reg.createTexture2(3, {
    width: 16,
    height: 8,
    depth: 64,
    mipLevels: 7,
    dimension: '3d',
    format: 'rgba8unorm',
  });
  assert.equal(createdTextures[2].desc.mipLevelCount, 7);
  assert.equal(reg.getTextureMeta(3).mipClamped, false);
  assert.equal(reg.getStats().mipClampCount, 2);

  // 4. 3D texture requesting 10 -> clamped to 7
  reg.createTexture2(4, {
    width: 16,
    height: 8,
    depth: 64,
    mipLevels: 10,
    dimension: '3d',
    format: 'rgba8unorm',
  });
  assert.equal(createdTextures[3].desc.mipLevelCount, 7);
  assert.equal(reg.getStats().mipClampCount, 3);

  // 5. 2D array texture with 64 layers & 16x16 -> max 5, requesting 7 clamps to 5
  reg.createTexture2(5, {
    width: 16,
    height: 16,
    arrayLayers: 64,
    mipLevels: 7,
    dimension: '2d',
    format: 'rgba8unorm',
  });
  assert.equal(createdTextures[4].desc.mipLevelCount, 5);
  assert.equal(reg.getStats().mipClampCount, 4);

  assert.equal(warnings.filter((w) => w.kind === 'mip_clamp').length, 4);
});

test('G41 sampleCount clamping enforces {1, 4} and falls back to 1 when multisampling is invalid', () => {
  const { device, createdTextures } = createMockWebGpuDevice({
    features: new Set([TEXTURE_COMPRESSION_BC_FEATURE]),
  });
  const reg = new GpuResourceRegistry({ device });

  // Valid 4x MSAA 2D single-mip render target
  reg.createTexture2(1, {
    width: 1280,
    height: 720,
    mipLevels: 1,
    sampleCount: 4,
    dimension: '2d',
    format: 'rgba16float',
    usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT,
  });
  assert.equal(createdTextures[0].desc.sampleCount, 4);
  assert.equal(reg.getTextureMeta(1).sampleClamped, false);

  // 8x MSAA clamps to 4x when valid
  reg.createTexture2(2, {
    width: 640,
    height: 360,
    mipLevels: 1,
    sampleCount: 8,
    dimension: '2d',
    format: 'depth32float',
    usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT,
  });
  assert.equal(createdTextures[1].desc.sampleCount, 4);
  assert.equal(reg.getTextureMeta(2).sampleClamped, true);

  // 2x MSAA (unsupported in WebGPU) clamps to 1
  reg.createTexture2(3, {
    width: 256,
    height: 256,
    sampleCount: 2,
    format: 'rgba8unorm',
    usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT,
  });
  assert.equal(createdTextures[2].desc.sampleCount, 1);

  // mipLevelCount > 1 forces sampleCount = 1
  reg.createTexture2(4, {
    width: 256,
    height: 256,
    mipLevels: 3,
    sampleCount: 4,
    format: 'rgba8unorm',
    usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT,
  });
  assert.equal(createdTextures[3].desc.sampleCount, 1);

  // 3D texture forces sampleCount = 1
  reg.createTexture2(5, {
    width: 32,
    height: 32,
    depth: 8,
    sampleCount: 4,
    dimension: '3d',
    format: 'rgba8unorm',
  });
  assert.equal(createdTextures[4].desc.sampleCount, 1);

  // 2D array texture (layers > 1) forces sampleCount = 1
  reg.createTexture2(6, {
    width: 64,
    height: 64,
    arrayLayers: 2,
    sampleCount: 4,
    dimension: '2d',
    format: 'rgba8unorm',
  });
  assert.equal(createdTextures[5].desc.sampleCount, 1);

  // Non-multisampleable formats (BC7, rgba32float, rgb9e5ufloat) force sampleCount = 1
  reg.createTexture2(7, {
    width: 64,
    height: 64,
    sampleCount: 4,
    format: 'bc7-rgba-unorm',
  });
  assert.equal(createdTextures[6].desc.sampleCount, 1);

  reg.createTexture2(8, {
    width: 64,
    height: 64,
    sampleCount: 4,
    format: 'rgba32float',
  });
  assert.equal(createdTextures[7].desc.sampleCount, 1);

  reg.createTexture2(9, {
    width: 64,
    height: 64,
    sampleCount: 4,
    format: 'rgb9e5ufloat',
  });
  assert.equal(createdTextures[8].desc.sampleCount, 1);

  // STORAGE_BINDING usage forces sampleCount = 1
  reg.createTexture2(10, {
    width: 64,
    height: 64,
    sampleCount: 4,
    format: 'rgba8unorm',
    usage: GPU_TEXTURE_USAGE.STORAGE_BINDING | GPU_TEXTURE_USAGE.TEXTURE_BINDING,
  });
  assert.equal(createdTextures[9].desc.sampleCount, 1);

  assert.equal(reg.getStats().sampleClampCount, 9);
});

test('G41 BC1..BC7 fallback to rgba8unorm when device.features lacks texture-compression-bc', () => {
  assert.equal(BC_TEXTURE_FORMATS.length, 14);
  for (const fmt of BC_TEXTURE_FORMATS) {
    assert.equal(isBcFormat(fmt), true);
  }
  assert.equal(isBcFormat('rgba8unorm'), false);

  const { device, createdTextures } = createMockWebGpuDevice({
    features: new Set(['timestamp-query']),
  });
  assert.equal(deviceSupportsBc(device), false);

  const reg = new GpuResourceRegistry({ device });

  BC_TEXTURE_FORMATS.forEach((bcFmt, idx) => {
    reg.createTexture2(100 + idx, {
      width: 13, // non-multiple of 4: kept as-is when falling back to rgba8unorm
      height: 7,
      mipLevels: 1,
      format: bcFmt,
      viewFormats: [bcFmt],
      usage: GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.COPY_DST,
    });
    const desc = createdTextures[idx].desc;
    assert.equal(desc.format, BC_FALLBACK_FORMAT);
    assert.equal(desc.size.width, 13);
    assert.equal(desc.size.height, 7);
    assert.deepEqual(desc.viewFormats, [
      bcFmt.endsWith('-srgb') ? 'rgba8unorm-srgb' : 'rgba8unorm',
    ]);
    const meta = reg.getTextureMeta(100 + idx);
    assert.equal(meta.originalFormat, bcFmt);
    assert.equal(meta.format, 'rgba8unorm');
    assert.equal(meta.bcFormatFallback, true);
    assert.equal(meta.bcDimensionAligned, false);
  });

  const stats = reg.getStats();
  assert.equal(stats.bcFormatFallbackCount, BC_TEXTURE_FORMATS.length);
  assert.equal(stats.bcDimensionAlignCount, 0);
});

test('G41 BC1..BC7 aligns dimensions to multiples of 4 when BC is supported and tracks bcDimensionAlignCount', () => {
  assert.equal(alignBcDimension(1), 4);
  assert.equal(alignBcDimension(3), 4);
  assert.equal(alignBcDimension(4), 4);
  assert.equal(alignBcDimension(5), 8);
  assert.equal(alignBcDimension(1366), 1368);

  const { device: bcDevice, createdTextures } = createMockWebGpuDevice({
    features: new Set([TEXTURE_COMPRESSION_BC_FEATURE]),
  });
  assert.equal(deviceSupportsBc(bcDevice), true);
  assert.equal(deviceSupportsBc({ features: [TEXTURE_COMPRESSION_BC_FEATURE] }), true);
  assert.equal(deviceSupportsBc({ features: { [TEXTURE_COMPRESSION_BC_FEATURE]: true } }), true);
  assert.equal(deviceSupportsBc({}), true); // no device.features -> assumed supported

  const reg = new GpuResourceRegistry({ device: bcDevice });

  // Non-multiple-of-4 BC7 texture -> aligned to 4x4 multiple (1366x767 -> 1368x768)
  reg.createTexture2(1, {
    width: 1366,
    height: 767,
    mipLevels: 1,
    format: 'bc7-rgba-unorm',
  });
  assert.equal(createdTextures[0].desc.format, 'bc7-rgba-unorm');
  assert.equal(createdTextures[0].desc.size.width, 1368);
  assert.equal(createdTextures[0].desc.size.height, 768);
  assert.equal(reg.getTextureMeta(1).bcDimensionAligned, true);
  assert.equal(reg.getStats().bcDimensionAlignCount, 1);
  assert.equal(reg.getStats().bcFormatFallbackCount, 0);

  // Tiny 1x2 BC1 texture -> aligned to 4x4
  reg.createTexture2(2, {
    width: 1,
    height: 2,
    mipLevels: 1,
    format: 'bc1-rgba-unorm-srgb',
  });
  assert.equal(createdTextures[1].desc.size.width, 4);
  assert.equal(createdTextures[1].desc.size.height, 4);
  assert.equal(reg.getStats().bcDimensionAlignCount, 2);

  // Already-aligned 1920x1080 BC3 texture -> no alignment needed
  reg.createTexture2(3, {
    width: 1920,
    height: 1080,
    mipLevels: 1,
    format: 'bc3-rgba-unorm',
  });
  assert.equal(createdTextures[2].desc.size.width, 1920);
  assert.equal(createdTextures[2].desc.size.height, 1080);
  assert.equal(reg.getTextureMeta(3).bcDimensionAligned, false);
  assert.equal(reg.getStats().bcDimensionAlignCount, 2);

  // Device without .features property keeps BC format and aligns dimensions
  const { device: noFeaturesDevice, createdTextures: nfTex } = createMockWebGpuDevice();
  reg.createTexture2(
    4,
    { width: 5, height: 5, format: 'bc5-rg-unorm' },
    { device: noFeaturesDevice }
  );
  assert.equal(nfTex[0].desc.format, 'bc5-rg-unorm');
  assert.equal(nfTex[0].desc.size.width, 8);
  assert.equal(nfTex[0].desc.size.height, 8);
  assert.equal(reg.getStats().bcDimensionAlignCount, 3);
  assert.equal(reg.getStats().bcFormatFallbackCount, 0);
});

test('G41 createBuffer aligns sizes to 4 bytes (16 bytes min for UNIFORM) and tracks bufferAlignPadCount', () => {
  assert.deepEqual(alignBufferSize(0, GPU_BUFFER_USAGE.VERTEX), {
    requestedSize: 0,
    alignedSize: 4,
    padded: true,
    isUniform: false,
  });
  assert.deepEqual(alignBufferSize(5, GPU_BUFFER_USAGE.VERTEX), {
    requestedSize: 5,
    alignedSize: 8,
    padded: true,
    isUniform: false,
  });
  assert.deepEqual(alignBufferSize(8, GPU_BUFFER_USAGE.UNIFORM), {
    requestedSize: 8,
    alignedSize: 16,
    padded: true,
    isUniform: true,
  });
  assert.deepEqual(alignBufferSize(16, GPU_BUFFER_USAGE.UNIFORM), {
    requestedSize: 16,
    alignedSize: 16,
    padded: false,
    isUniform: true,
  });
  assert.deepEqual(alignBufferSize(21n, GPU_BUFFER_USAGE.UNIFORM), {
    requestedSize: 21,
    alignedSize: 24,
    padded: true,
    isUniform: true,
  });

  const { device, createdBuffers } = createMockWebGpuDevice();
  const reg = new GpuResourceRegistry({ framesInFlight: 2, device });

  // Descriptor form: non-aligned vertex buffer 5 -> 8
  const b1 = reg.createBuffer(1, {
    size: 5,
    usage: GPU_BUFFER_USAGE.VERTEX | GPU_BUFFER_USAGE.COPY_DST,
    label: 'tri-vb',
  });
  assert.equal(b1, createdBuffers[0]);
  assert.equal(b1.size, 8);
  assert.equal(reg.getBufferMeta(1).requestedSize, 5);
  assert.equal(reg.getBufferMeta(1).alignedSize, 8);
  assert.equal(reg.getBufferMeta(1).padded, true);
  assert.equal(reg.getStats().activeBytes, 8);
  assert.equal(reg.getStats().bufferAlignPadCount, 1);

  // Positional form: UNIFORM buffer 12 -> 16
  const b2 = reg.createBuffer(2, 12, GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST);
  assert.equal(b2.size, 16);
  assert.equal(reg.getBufferMeta(2).isUniform, true);
  assert.equal(reg.getStats().bufferAlignPadCount, 2);

  // Already aligned 64-byte UNIFORM buffer -> no pad increment
  const b3 = reg.createBuffer(3, {
    size: 64,
    usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST,
  });
  assert.equal(b3.size, 64);
  assert.equal(reg.getBufferMeta(3).padded, false);
  assert.equal(reg.getStats().bufferAlignPadCount, 2);

  // Overwriting handle 1 retires previous buffer safely via deferred GC
  const b1New = reg.createBuffer(1, { size: 32, usage: GPU_BUFFER_USAGE.VERTEX });
  assert.equal(reg.get(RESOURCE_KIND.BUFFER, 1), b1New);
  assert.equal(b1.destroyCount, 0);
  reg.beginFrame(1);
  reg.beginFrame(2);
  assert.equal(b1.destroyCount, 1);
  assert.equal(b1New.destroyCount, 0);
});

test('G41 writeBuffer zero-pads non-4-byte payloads and safely copies SharedArrayBuffer views', () => {
  const { device, writeBuffers } = createMockWebGpuDevice();
  const reg = new GpuResourceRegistry({ device });

  reg.createBuffer(10, {
    size: 64,
    usage: GPU_BUFFER_USAGE.VERTEX | GPU_BUFFER_USAGE.COPY_DST,
  });

  // 1. Odd-length 5-byte Uint8Array -> zero-padded to 8 bytes
  const odd5 = new Uint8Array([0xaa, 0xbb, 0xcc, 0xdd, 0xee]);
  assert.equal(reg.writeBuffer(10, 0, odd5), true);
  assert.equal(writeBuffers.length, 1);
  assert.equal(writeBuffers[0].dstOffset, 0);
  assert.equal(writeBuffers[0].byteLength, 8);
  assert.deepEqual(
    Array.from(writeBuffers[0].bytesCopy),
    [0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0x00, 0x00, 0x00]
  );
  assert.equal(reg.getStats().bufferAlignPadCount, 1);

  // 2. Odd-length 3-element Uint16Array (6 bytes) -> zero-padded to 8 bytes
  const u16 = new Uint16Array([0x0102, 0x0304, 0x0506]);
  assert.equal(reg.writeBuffer(10, 8, u16), true);
  assert.equal(writeBuffers.length, 2);
  assert.equal(writeBuffers[1].byteLength, 8);
  assert.equal(writeBuffers[1].bytesCopy[6], 0);
  assert.equal(writeBuffers[1].bytesCopy[7], 0);
  assert.equal(reg.getStats().bufferAlignPadCount, 2);

  // 3. Aligned 8-byte Uint8Array -> no copy/pad increment
  const aligned8 = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
  assert.equal(reg.writeBuffer(10, 16, aligned8), true);
  assert.equal(writeBuffers.length, 3);
  assert.equal(writeBuffers[2].byteLength, 8);
  assert.equal(writeBuffers[2].data, aligned8);
  assert.equal(reg.getStats().bufferAlignPadCount, 2);

  // 4. SharedArrayBuffer view (unaligned 5 bytes) -> copied to non-shared & zero-padded to 8 bytes
  const sab = new SharedArrayBuffer(16);
  const sabAll = new Uint8Array(sab);
  sabAll.set([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]);
  const sabSlice5 = new Uint8Array(sab, 2, 5); // [30, 40, 50, 60, 70]
  assert.equal(reg.writeBuffer(10, 24, sabSlice5), true);
  assert.equal(writeBuffers.length, 4);
  assert.equal(writeBuffers[3].isShared, false, 'SharedArrayBuffer view must be copied to non-shared buffer');
  assert.equal(writeBuffers[3].byteLength, 8);
  assert.deepEqual(Array.from(writeBuffers[3].bytesCopy), [30, 40, 50, 60, 70, 0, 0, 0]);
  assert.equal(reg.getStats().bufferAlignPadCount, 3);

  // 5. SharedArrayBuffer view (aligned 8 bytes) -> copied to non-shared WITHOUT incrementing bufferAlignPadCount
  const sabSlice8 = new Uint8Array(sab, 4, 8);
  assert.equal(reg.writeBuffer(10, 32, sabSlice8), true);
  assert.equal(writeBuffers.length, 5);
  assert.equal(writeBuffers[4].isShared, false, 'Aligned SharedArrayBuffer view must still be copied to non-shared buffer');
  assert.equal(writeBuffers[4].byteLength, 8);
  assert.deepEqual(Array.from(writeBuffers[4].bytesCopy), [50, 60, 70, 80, 90, 100, 110, 120]);
  assert.equal(reg.getStats().bufferAlignPadCount, 3);

  // 6. Direct SharedArrayBuffer with dataOffset & size
  const sabUpload = padBufferUploadData(sab, 1, 3);
  assert.equal(sabUpload.copiedFromShared, true);
  assert.equal(sabUpload.padded, true);
  assert.equal(sabUpload.alignedByteLength, 4);
  assert.equal(sabUpload.bytes.buffer instanceof SharedArrayBuffer, false);
  assert.deepEqual(Array.from(sabUpload.bytes), [20, 30, 40, 0]);

  // 7. Empty upload is a safe no-op; missing handle returns false; direct GPUBuffer works
  assert.equal(reg.writeBuffer(10, 0, new Uint8Array(0)), true);
  assert.equal(writeBuffers.length, 5);
  assert.equal(reg.writeBuffer(9999, 0, odd5), false);

  const directBuf = reg.get(RESOURCE_KIND.BUFFER, 10);
  assert.equal(reg.writeBuffer(directBuf, 40, new Uint8Array([7, 8, 9])), true);
  assert.equal(writeBuffers.length, 6);
  assert.equal(writeBuffers[5].byteLength, 4);
  assert.deepEqual(Array.from(writeBuffers[5].bytesCopy), [7, 8, 9, 0]);
  assert.equal(reg.getStats().bufferAlignPadCount, 4);
});

test('G41 setDevice, missing device guards, and getStats() sanitized counters default to 0', () => {
  const reg = new GpuResourceRegistry();
  const initialStats = reg.getStats();
  assert.equal(initialStats.bcFormatFallbackCount, 0);
  assert.equal(initialStats.bcDimensionAlignCount, 0);
  assert.equal(initialStats.bufferAlignPadCount, 0);
  assert.equal(initialStats.mipClampCount, 0);
  assert.equal(initialStats.sampleClampCount, 0);

  assert.throws(() => reg.createBuffer(1, { size: 16 }), TypeError);
  assert.throws(() => reg.createTexture(1, { width: 16, height: 16 }), TypeError);
  assert.throws(() => reg.setDevice('not-a-device'), TypeError);
  assert.throws(() => new GpuResourceRegistry({ device: 'bad' }), TypeError);
  assert.throws(() => new GpuResourceRegistry({ onSanitizeWarning: 42 }), TypeError);

  const { device } = createMockWebGpuDevice({ features: new Set() });
  reg.setDevice(device);
  const tex = reg.createTexture2(1, { width: 32, height: 32, format: 'bc7-rgba-unorm', mipLevels: 99 });
  assert.ok(tex);
  assert.equal(reg.getStats().bcFormatFallbackCount, 1);
  assert.equal(reg.getStats().mipClampCount, 1);

  // Confirm pure helper validation
  assert.throws(() => decodeCreateTexture2('not-a-dataview'), TypeError);
  assert.throws(() => sanitizeTextureDescriptor(null), TypeError);
  assert.throws(() => padBufferUploadData('not-a-buffer'), TypeError);
  assert.deepEqual(clampMipLevelCount(undefined, 64, 64), {
    mipLevelCount: 1,
    maxMipLevels: 7,
    clamped: false,
  });
  assert.deepEqual(clampSampleCount(undefined), {
    sampleCount: 1,
    clamped: false,
  });
});

// ---------------------------------------------------------------------------
// W129 (#785): Per-kind VRAM accounting (byKind), peakTotalBytes high-water
// mark, fence-deferred retirement without double-decrementing, and sanitized
// getOldestLiveResources(limit) diagnostics.
// ---------------------------------------------------------------------------

test('W129 getStats().byKind tracks per-kind live counts and VRAM bytes accurately across kinds', () => {
  const reg = new GpuResourceRegistry({ framesInFlight: 2 });

  // Initial empty state exposes every RESOURCE_KIND with { count: 0, bytes: 0 }
  const emptyStats = reg.getStats();
  assert.equal(emptyStats.peakTotalBytes, 0);
  for (const kind of RESOURCE_KINDS) {
    assert.deepEqual(emptyStats.byKind[kind], { count: 0, bytes: 0 });
  }

  reg.beginFrame(1);
  reg.register(RESOURCE_KIND.BUFFER, 1, createDestroyableMock('vb-1'), { byteSize: 4096 });
  reg.register(RESOURCE_KIND.BUFFER, 2, createDestroyableMock('ib-2'), { byteSize: 2048 });
  reg.register(RESOURCE_KIND.TEXTURE, 10, createDestroyableMock('tex-10'), { byteSize: 65536 });
  reg.register(RESOURCE_KIND.TEXTURE_VIEW, 10, createNonDestroyableMock('view-10'), { byteSize: 0 });
  reg.register(RESOURCE_KIND.SAMPLER, 20, createNonDestroyableMock('samp-20'));
  reg.register(RESOURCE_KIND.BIND_GROUP, 30, createNonDestroyableMock('bg-30'), { byteSize: 128 });
  reg.register(RESOURCE_KIND.RENDER_PIPELINE, 40, createNonDestroyableMock('pso-40'), { byteSize: 512 });

  const stats = reg.getStats();
  assert.deepEqual(stats.byKind[RESOURCE_KIND.BUFFER], { count: 2, bytes: 6144 });
  assert.deepEqual(stats.byKind[RESOURCE_KIND.TEXTURE], { count: 1, bytes: 65536 });
  assert.deepEqual(stats.byKind[RESOURCE_KIND.TEXTURE_VIEW], { count: 1, bytes: 0 });
  assert.deepEqual(stats.byKind[RESOURCE_KIND.SAMPLER], { count: 1, bytes: 0 });
  assert.deepEqual(stats.byKind[RESOURCE_KIND.BIND_GROUP], { count: 1, bytes: 128 });
  assert.deepEqual(stats.byKind[RESOURCE_KIND.RENDER_PIPELINE], { count: 1, bytes: 512 });
  assert.deepEqual(stats.byKind[RESOURCE_KIND.COMPUTE_PIPELINE], { count: 0, bytes: 0 });

  const sumCount = Object.values(stats.byKind).reduce((acc, k) => acc + k.count, 0);
  const sumBytes = Object.values(stats.byKind).reduce((acc, k) => acc + k.bytes, 0);
  assert.equal(sumCount, stats.activeCount);
  assert.equal(sumBytes, stats.activeBytes);
  assert.equal(stats.peakTotalBytes, 6144 + 65536 + 128 + 512);
  assert.equal(stats.peakTotalBytes, stats.peakBytes);

  // Snapshot isolation: mutating stats.byKind must not corrupt internal counters.
  stats.byKind[RESOURCE_KIND.BUFFER].bytes = 999999;
  stats.byKind[RESOURCE_KIND.BUFFER].count = 999;
  assert.deepEqual(reg.getStats().byKind[RESOURCE_KIND.BUFFER], { count: 2, bytes: 6144 });
});

test('W129 overwrite replacement updates byKind without double-counting and tracks peakTotalBytes', () => {
  const reg = new GpuResourceRegistry({ framesInFlight: 2 });
  const bufV1 = createDestroyableMock('buf-v1');
  const bufV2 = createDestroyableMock('buf-v2');
  const texOther = createDestroyableMock('tex-other');

  reg.beginFrame(1);
  reg.register(RESOURCE_KIND.BUFFER, 5, bufV1, { byteSize: 1024, label: 'v1' });
  reg.register(RESOURCE_KIND.TEXTURE, 6, texOther, { byteSize: 4096, label: 't1' });
  assert.deepEqual(reg.getStats().byKind[RESOURCE_KIND.BUFFER], { count: 1, bytes: 1024 });
  assert.equal(reg.getStats().peakTotalBytes, 5120);

  // Replace (buffer, 5) with a new object (2048 bytes): old buffer moves to pendingRetire,
  // live byKind.buffer must reflect ONLY bufV2 (no double-counting), while peakTotalBytes
  // captures active (2048 + 4096) + pendingRetire (1024) = 7168.
  reg.register(RESOURCE_KIND.BUFFER, 5, bufV2, { byteSize: 2048, label: 'v2' });
  let stats = reg.getStats();
  assert.deepEqual(stats.byKind[RESOURCE_KIND.BUFFER], { count: 1, bytes: 2048 });
  assert.deepEqual(stats.byKind[RESOURCE_KIND.TEXTURE], { count: 1, bytes: 4096 });
  assert.equal(stats.activeBytes, 6144);
  assert.equal(stats.pendingRetireBytes, 1024);
  assert.equal(stats.totalBytes, 7168);
  assert.equal(stats.peakTotalBytes, 7168);

  // Re-registering the exact same object updates byteSize in-place without double-counting.
  reg.register(RESOURCE_KIND.BUFFER, 5, bufV2, { byteSize: 3072, label: 'v2-grown' });
  stats = reg.getStats();
  assert.deepEqual(stats.byKind[RESOURCE_KIND.BUFFER], { count: 1, bytes: 3072 });
  assert.equal(stats.peakTotalBytes, 8192);

  // Shrinking same object updates bytes cleanly while preserving peakTotalBytes.
  reg.register(RESOURCE_KIND.BUFFER, 5, bufV2, { byteSize: 512, label: 'v2-shrunk' });
  stats = reg.getStats();
  assert.deepEqual(stats.byKind[RESOURCE_KIND.BUFFER], { count: 1, bytes: 512 });
  assert.equal(stats.peakTotalBytes, 8192);

  // Advance framesInFlight so bufV1 is destroyed -> byKind.buffer must NOT double-decrement.
  reg.endFrame();
  reg.beginFrame(2);
  reg.beginFrame(3);
  assert.equal(bufV1.destroyCount, 1);
  assert.equal(bufV2.destroyCount, 0);
  stats = reg.getStats();
  assert.deepEqual(stats.byKind[RESOURCE_KIND.BUFFER], { count: 1, bytes: 512 });
  assert.equal(stats.pendingRetireBytes, 0);
  assert.equal(stats.peakTotalBytes, 8192);
});

test('W129 fence-deferred retirement and destroyAll never double-decrement byKind and preserve peakTotalBytes', async () => {
  const reg = new GpuResourceRegistry({ framesInFlight: 3 });
  const b1 = createDestroyableMock('b1');
  const b2 = createDestroyableMock('b2');
  const t1 = createDestroyableMock('t1');
  const fence = createDeferred();

  reg.beginFrame(1);
  reg.register(RESOURCE_KIND.BUFFER, 1, b1, { byteSize: 4000 });
  reg.register(RESOURCE_KIND.BUFFER, 2, b2, { byteSize: 6000 });
  reg.register(RESOURCE_KIND.TEXTURE, 3, t1, { byteSize: 16000 });

  assert.equal(reg.getStats().peakTotalBytes, 26000);

  // Retire b1 and t1, bind to GPU fence.
  assert.equal(reg.retire(RESOURCE_KIND.BUFFER, 1), true);
  assert.equal(reg.retire(RESOURCE_KIND.BUFFER, 1), false); // duplicate retire is a no-op
  assert.equal(reg.retire(RESOURCE_KIND.TEXTURE, 3), true);
  reg.endFrame(fence.promise);

  // Immediately after retire (before fence resolves), byKind reflects only live b2.
  let stats = reg.getStats();
  assert.deepEqual(stats.byKind[RESOURCE_KIND.BUFFER], { count: 1, bytes: 6000 });
  assert.deepEqual(stats.byKind[RESOURCE_KIND.TEXTURE], { count: 0, bytes: 0 });
  assert.equal(stats.activeBytes, 6000);
  assert.equal(stats.pendingRetireBytes, 20000);
  assert.equal(stats.totalBytes, 26000);
  assert.equal(stats.peakTotalBytes, 26000);

  // Resolve fence and collect -> b1 and t1 destroyed; byKind must NOT double-decrement!
  fence.resolve();
  await fence.promise;
  await Promise.resolve();

  const col = reg.collect();
  assert.equal(col.destroyedCount, 2);
  assert.equal(col.freedBytes, 20000);
  assert.equal(b1.destroyCount, 1);
  assert.equal(t1.destroyCount, 1);
  assert.equal(b2.destroyCount, 0);

  stats = reg.getStats();
  assert.deepEqual(stats.byKind[RESOURCE_KIND.BUFFER], { count: 1, bytes: 6000 });
  assert.deepEqual(stats.byKind[RESOURCE_KIND.TEXTURE], { count: 0, bytes: 0 });
  assert.equal(stats.activeBytes, 6000);
  assert.equal(stats.pendingRetireBytes, 0);
  assert.equal(stats.totalBytes, 6000);
  assert.equal(stats.peakTotalBytes, 26000);

  // Register a texture and retire it so destroyAll sees BOTH active (b2) and pending (t2).
  const t2 = createDestroyableMock('t2');
  reg.register(RESOURCE_KIND.TEXTURE, 4, t2, { byteSize: 8000 });
  reg.retire(RESOURCE_KIND.TEXTURE, 4);
  assert.deepEqual(reg.getStats().byKind[RESOURCE_KIND.TEXTURE], { count: 0, bytes: 0 });

  const torn = reg.destroyAll();
  assert.equal(torn.destroyedCount, 2);
  assert.equal(torn.freedBytes, 14000);

  stats = reg.getStats();
  for (const kind of RESOURCE_KINDS) {
    assert.deepEqual(stats.byKind[kind], { count: 0, bytes: 0 });
  }
  assert.equal(stats.activeBytes, 0);
  assert.equal(stats.pendingRetireBytes, 0);
  assert.equal(stats.totalBytes, 0);
  assert.equal(stats.peakTotalBytes, 26000);

  // Idempotent second destroyAll() keeps all zeroes and peakTotalBytes intact.
  reg.destroyAll();
  assert.deepEqual(reg.getStats().byKind[RESOURCE_KIND.BUFFER], { count: 0, bytes: 0 });
  assert.equal(reg.getStats().peakTotalBytes, 26000);

  // Re-allocation above prior peak raises peakTotalBytes cleanly.
  reg.register(RESOURCE_KIND.BUFFER, 9, createDestroyableMock('big-buf'), { byteSize: 32000 });
  stats = reg.getStats();
  assert.deepEqual(stats.byKind[RESOURCE_KIND.BUFFER], { count: 1, bytes: 32000 });
  assert.equal(stats.peakTotalBytes, 32000);
});

test('W129 getOldestLiveResources(limit) orders by createdFrame then registration order and strictly sanitizes output', () => {
  const reg = new GpuResourceRegistry({ framesInFlight: 2 });

  assert.deepEqual(reg.getOldestLiveResources(), []);
  assert.deepEqual(reg.getOldestLiveResources(0), []);

  // Frame 2: register texture first, then buffer, then sampler (cross-kind registration order).
  reg.beginFrame(2);
  const tex2 = createDestroyableMock('tex-f2');
  const buf2 = createDestroyableMock('buf-f2');
  reg.register(RESOURCE_KIND.TEXTURE, 100, tex2, {
    byteSize: 8192,
    label: 'C:\\secret\\host\\path\\albedo.dds',
    tag: { secretHostInfo: '/home/user/private' },
  });
  reg.register(RESOURCE_KIND.BUFFER, 10, buf2, {
    byteSize: 1024,
    label: '/Users/dev/mesh.bin',
    tag: 'mesh',
  });
  reg.register(RESOURCE_KIND.SAMPLER, 5, createNonDestroyableMock('samp-f2'), {
    byteSize: 0,
    label: 'private-sampler-label',
  });

  // Frame 5: register bindGroup, overwrite (buffer, 10) with a NEW resource (moves its createdFrame to 5),
  // and update tex2 in-place (preserves its original createdFrame 2 and registration order).
  reg.beginFrame(5);
  reg.register(RESOURCE_KIND.BIND_GROUP, 20, createNonDestroyableMock('bg-f5'), { byteSize: 64 });
  reg.register(RESOURCE_KIND.BUFFER, 10, createDestroyableMock('buf-f5-replaced'), {
    byteSize: 2048,
    label: 'replaced-buf',
  });
  reg.register(RESOURCE_KIND.TEXTURE, 100, tex2, {
    byteSize: 16384,
    label: 'updated-label-must-not-leak',
  });

  // Frame 7: register a renderPipeline and retire sampler 5 (retired entries must be excluded).
  reg.beginFrame(7);
  reg.register(RESOURCE_KIND.RENDER_PIPELINE, 30, createNonDestroyableMock('pso-f7'), { byteSize: 512 });
  reg.retire(RESOURCE_KIND.SAMPLER, 5);

  const allOldest = reg.getOldestLiveResources(10);
  assert.deepEqual(allOldest, [
    { kind: RESOURCE_KIND.TEXTURE, id: 100, byteSize: 16384, createdFrame: 2 },
    { kind: RESOURCE_KIND.BIND_GROUP, id: 20, byteSize: 64, createdFrame: 5 },
    { kind: RESOURCE_KIND.BUFFER, id: 10, byteSize: 2048, createdFrame: 5 },
    { kind: RESOURCE_KIND.RENDER_PIPELINE, id: 30, byteSize: 512, createdFrame: 7 },
  ]);

  // Strict sanitization: ONLY { kind, id, byteSize, createdFrame } keys; zero labels/tags/GPU handles.
  for (const item of allOldest) {
    assert.deepEqual(Object.keys(item).sort(), ['byteSize', 'createdFrame', 'id', 'kind']);
    assert.equal('label' in item, false);
    assert.equal('tag' in item, false);
    assert.equal('resource' in item, false);
    assert.equal('handle' in item, false);
    assert.equal('registeredAtFrame' in item, false);
    assert.equal('registrationSeq' in item, false);
  }
  const serialized = JSON.stringify(allOldest);
  assert.equal(serialized.includes('secret'), false);
  assert.equal(serialized.includes('host'), false);
  assert.equal(serialized.includes('private'), false);

  // Limit truncation works accurately.
  assert.deepEqual(reg.getOldestLiveResources(2), allOldest.slice(0, 2));
  assert.deepEqual(reg.getOldestLiveResources(0), []);

  // Default limit is 16 when > 16 resources are live.
  reg.beginFrame(8);
  for (let i = 0; i < 20; i += 1) {
    reg.register(RESOURCE_KIND.BUFFER, 500 + i, createDestroyableMock(`extra-${i}`), {
      byteSize: 256,
    });
  }
  const defaultLimited = reg.getOldestLiveResources();
  assert.equal(defaultLimited.length, 16);
  assert.deepEqual(defaultLimited.slice(0, 4), allOldest);
  assert.equal(defaultLimited[4].id, 500);
  assert.equal(defaultLimited[15].id, 511);

  // Validation rejects invalid limits.
  assert.throws(() => reg.getOldestLiveResources(-1), RangeError);
  assert.throws(() => reg.getOldestLiveResources(1.5), RangeError);
  assert.throws(() => reg.getOldestLiveResources(NaN), RangeError);
  assert.throws(() => reg.getOldestLiveResources('16'), RangeError);

  // destroyAll clears live entries.
  reg.destroyAll();
  assert.deepEqual(reg.getOldestLiveResources(), []);
});
