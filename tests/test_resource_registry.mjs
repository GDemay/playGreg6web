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
  TEXTURE_VIEW_DIMENSIONS,
  TEXTURE_VIEW_ASPECTS,
  SRGB_SIBLING_FORMATS,
  BINK_VIDEO_TEXTURE_FORMATS,
  BC_TEXTURE_FORMATS,
  BC_FALLBACK_FORMAT,
  TEXTURE_COMPRESSION_BC_FEATURE,
  RESOURCE_KIND,
  RESOURCE_KINDS,
  isBcFormat,
  isSrgbFormat,
  getSrgbSiblingFormat,
  isViewFormatCompatible,
  resolveTextureViewFormats,
  buildTextureViewCacheKey,
  normalizeTextureViewDescriptor,
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

// ---------------------------------------------------------------------------
// W178 (#885): sRGB/linear sibling viewFormats, r8unorm Bink2 video textures,
// 8-tuple cached GPUTextureView lookup, textureGeneration tracking, and
// automatic stale-view invalidation on 1080p <-> 4K texture recreation.
// ---------------------------------------------------------------------------

test('W178 sRGB/linear sibling format helpers and resolveTextureViewFormats handle all WebGPU pairs and r8unorm Bink2 planes', () => {
  assert.ok(Object.isFrozen(SRGB_SIBLING_FORMATS));
  assert.ok(Object.isFrozen(TEXTURE_VIEW_DIMENSIONS));
  assert.ok(Object.isFrozen(TEXTURE_VIEW_ASPECTS));
  assert.ok(Object.isFrozen(BINK_VIDEO_TEXTURE_FORMATS));
  assert.ok(BINK_VIDEO_TEXTURE_FORMATS.includes('r8unorm'));

  // All 6 WebGPU sRGB <-> linear pairs are bidirectional.
  const expectedPairs = [
    ['rgba8unorm', 'rgba8unorm-srgb'],
    ['bgra8unorm', 'bgra8unorm-srgb'],
    ['bc1-rgba-unorm', 'bc1-rgba-unorm-srgb'],
    ['bc2-rgba-unorm', 'bc2-rgba-unorm-srgb'],
    ['bc3-rgba-unorm', 'bc3-rgba-unorm-srgb'],
    ['bc7-rgba-unorm', 'bc7-rgba-unorm-srgb'],
  ];
  for (const [linear, srgb] of expectedPairs) {
    assert.equal(getSrgbSiblingFormat(linear), srgb);
    assert.equal(getSrgbSiblingFormat(srgb), linear);
    assert.equal(isSrgbFormat(linear), false);
    assert.equal(isSrgbFormat(srgb), true);
    assert.equal(isViewFormatCompatible(linear, srgb), true);
    assert.equal(isViewFormatCompatible(srgb, linear), true);
    assert.equal(isViewFormatCompatible(linear, linear), true);
  }

  // r8unorm (Bink2 video), rg8unorm, rgba16float, depth/stencil have NO sRGB sibling.
  for (const nonSrgb of ['r8unorm', 'rg8unorm', 'rgba16float', 'depth24plus-stencil8', 'depth32float']) {
    assert.equal(getSrgbSiblingFormat(nonSrgb), null);
    assert.equal(isSrgbFormat(nonSrgb), false);
  }
  assert.equal(getSrgbSiblingFormat(null), null);
  assert.equal(getSrgbSiblingFormat(123), null);

  // Depth/stencil aspect compatibility.
  assert.equal(isViewFormatCompatible('depth24plus-stencil8', 'depth24plus', 'depth-only'), true);
  assert.equal(isViewFormatCompatible('depth24plus-stencil8', 'stencil8', 'stencil-only'), true);
  assert.equal(isViewFormatCompatible('depth32float-stencil8', 'depth32float', 'depth-only'), true);
  assert.equal(isViewFormatCompatible('depth32float-stencil8', 'stencil8', 'stencil-only'), true);
  assert.equal(isViewFormatCompatible('r8unorm', 'rgba8unorm-srgb', 'all'), false);
  assert.equal(isViewFormatCompatible('rgba8unorm', 'bgra8unorm-srgb', 'all'), false);

  // resolveTextureViewFormats strips incompatible formats, deduplicates, and adds sibling when requested.
  const resolvedRgba = resolveTextureViewFormats(
    'rgba8unorm',
    ['rgba8unorm', 23, 'bgra8unorm-srgb', 'r8unorm', 'rgba8unorm-srgb'],
    { includeSrgbSiblingViewFormat: true }
  );
  assert.deepEqual(resolvedRgba.viewFormats, ['rgba8unorm', 'rgba8unorm-srgb']);
  assert.equal(resolvedRgba.siblingViewFormat, 'rgba8unorm-srgb');
  assert.deepEqual(resolvedRgba.droppedViewFormats, ['bgra8unorm-srgb', 'r8unorm']);

  // r8unorm with includeSrgbSiblingViewFormat: true never gets a bogus sRGB viewFormat.
  const resolvedR8 = resolveTextureViewFormats('r8unorm', ['rgba8unorm-srgb', 'r8unorm'], {
    includeSrgbSiblingViewFormat: true,
  });
  assert.deepEqual(resolvedR8.viewFormats, ['r8unorm']);
  assert.equal(resolvedR8.siblingViewFormat, null);
  assert.deepEqual(resolvedR8.droppedViewFormats, ['rgba8unorm-srgb']);

  // Classic-worker importScripts isolation also exposes all W178 exports without leaking top-level bindings.
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
  vmCtx.importScripts('./texture_formats.js', './resource_registry.js');
  assert.doesNotThrow(() => {
    vm.runInContext(
      `
      const SRGB_SIBLING_FORMATS = {};
      const TEXTURE_VIEW_DIMENSIONS = [];
      const TEXTURE_VIEW_ASPECTS = [];
      const BINK_VIDEO_TEXTURE_FORMATS = [];
      function getSrgbSiblingFormat() {}
      function resolveTextureViewFormats() {}
      function normalizeTextureViewDescriptor() {}
      `,
      vmCtx
    );
  });
  assert.equal(typeof vmCtx.PlayGta6ResourceRegistry.getSrgbSiblingFormat, 'function');
  assert.equal(typeof vmCtx.PlayGta6ResourceRegistry.normalizeTextureViewDescriptor, 'function');
  assert.equal(typeof vmCtx.PlayGta6ResourceRegistry.buildTextureViewCacheKey, 'function');
});

test('W178 r8unorm Bink2 video textures support CreateTexture2, 1-bpp VRAM accounting, view caching, and 1080p <-> 4K recreation', () => {
  const { device, createdTextures } = createMockWebGpuDevice({
    features: new Set([TEXTURE_COMPRESSION_BC_FEATURE]),
  });
  const warnings = [];
  const reg = new GpuResourceRegistry({
    framesInFlight: 2,
    device,
    includeSrgbSiblingViewFormat: true,
    onSanitizeWarning: (w) => warnings.push(w),
  });

  // Binary CreateTexture2 packet for a 1080p Bink2 Y-plane (format code 1 = r8unorm).
  const bink1080Payload = buildCreateTexture2Payload({
    id: 500,
    width: 1920,
    height: 1080,
    depthOrLayers: 1,
    mipLevels: 1,
    sampleCount: 1,
    usageFlags: GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.COPY_DST,
    dimension: 1, // 2d
    format: 1, // r8unorm
    viewFormats: [23], // incompatible rgba8unorm-srgb must be cleanly dropped
  });

  const res1080 = reg.createTexture2(bink1080Payload, 0, CREATE_TEXTURE2_PAYLOAD_BYTES);
  assert.equal(res1080.error, undefined);
  assert.equal(res1080.id, 500);
  assert.equal(createdTextures[0].desc.format, 'r8unorm');
  assert.equal(createdTextures[0].desc.viewFormats, undefined);
  assert.equal(res1080.meta.format, 'r8unorm');
  assert.equal(res1080.meta.siblingViewFormat, null);
  assert.deepEqual(res1080.meta.viewFormats, []);
  assert.equal(res1080.meta.byteSize, 1920 * 1080); // 1 bpp = 2,073,600 bytes
  assert.equal(res1080.meta.textureGeneration, 1);
  assert.equal(reg.getTextureGeneration(500), 1);
  assert.equal(warnings.some((w) => w.kind === 'view_format_drop' && w.handle === 500), true);

  // Cached view lookup on r8unorm Bink2 plane: default and explicit 8-tuple hit the same entry.
  let createViewCalls = 0;
  const origCreateView0 = createdTextures[0].createView.bind(createdTextures[0]);
  createdTextures[0].createView = (vd) => {
    createViewCalls += 1;
    return origCreateView0(vd);
  };

  const viewY1080 = reg.getTextureView(500);
  assert.ok(viewY1080);
  assert.equal(createViewCalls, 1);
  assert.equal(viewY1080.textureInternalId, createdTextures[0].internalId);
  assert.equal(viewY1080.viewDesc.format, 'r8unorm');
  assert.equal(reg.isTextureViewValid(viewY1080), true);
  assert.equal(reg.isViewValid(viewY1080), true);

  const viewY1080Hit = reg.getOrCreateTextureView(500, {
    format: 'r8unorm',
    dimension: '2d',
    baseMipLevel: 0,
    mipLevelCount: 1,
    baseArrayLayer: 0,
    arrayLayerCount: 1,
    aspect: 'all',
  });
  assert.equal(viewY1080Hit, viewY1080);
  assert.equal(createViewCalls, 1);

  // Requesting an sRGB view on an r8unorm texture is rejected.
  assert.throws(() => reg.getTextureView(500, { format: 'rgba8unorm-srgb' }), RangeError);

  let stats = reg.getStats();
  assert.equal(stats.viewCacheMisses, 1);
  assert.equal(stats.viewCacheHits, 1);
  assert.equal(stats.textureRecreations, 0);
  assert.equal(stats.staleViewsInvalidated, 0);

  // Recreate Bink2 Y-plane at 4K (3840x2160) on the exact same texture ID 500.
  const defaultView1080 = reg.get(RESOURCE_KIND.TEXTURE_VIEW, 500);
  assert.equal(reg.isTextureViewValid(defaultView1080), true);

  const tex4k = reg.createTexture2(500, {
    width: 3840,
    height: 2160,
    mipLevels: 1,
    format: 'r8unorm',
    usage: GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.COPY_DST,
  });
  assert.equal(tex4k, createdTextures[1]);
  assert.equal(reg.getTextureGeneration(500), 2);
  assert.equal(reg.isTextureGenerationCurrent(500, 1), false);
  assert.equal(reg.isTextureGenerationCurrent(500, 2), true);
  assert.equal(reg.getTextureMeta(500).byteSize, 3840 * 2160);
  assert.equal(reg.getTextureMeta(500).textureGeneration, 2);

  // Both the cached 1080p view and the old default view are immediately invalidated!
  assert.equal(reg.isTextureViewValid(viewY1080), false);
  assert.equal(reg.isTextureViewValid(defaultView1080), false);
  assert.equal(reg.getTextureViewInfo(viewY1080).valid, false);
  assert.equal(reg.getTextureViewInfo(viewY1080).textureGeneration, 1);
  assert.equal(reg.getTextureViewInfo(viewY1080).currentGeneration, 2);

  stats = reg.getStats();
  assert.equal(stats.textureRecreations, 1);
  assert.equal(stats.staleViewsInvalidated, 2); // 1 cached view + 1 default view
  assert.equal(stats.cachedViewCount, 0);

  // Next view lookup creates a fresh view bound to the 4K texture.
  const viewY4k = reg.getTextureView(500, { format: 'r8unorm' });
  assert.notEqual(viewY4k, viewY1080);
  assert.equal(viewY4k.textureInternalId, createdTextures[1].internalId);
  assert.equal(reg.isTextureViewValid(viewY4k), true);
  assert.equal(reg.getStats().viewCacheMisses, 2);
  assert.equal(reg.getStats().cachedViewCount, 1);
});

test('W178 sibling sRGB/linear viewFormats metadata and 8-tuple cached view lookup across format/dimension/mip/layer/aspect', () => {
  const { device, createdTextures } = createMockWebGpuDevice({
    features: new Set([TEXTURE_COMPRESSION_BC_FEATURE]),
  });
  const reg = new GpuResourceRegistry({ framesInFlight: 2, device });

  // 1. 2D array texture (6 layers, 4 mips) with sibling sRGB viewFormat auto-included.
  reg.createTexture2(
    10,
    {
      width: 256,
      height: 256,
      arrayLayers: 6,
      mipLevels: 4,
      dimension: '2d',
      format: 'rgba8unorm',
      usage: GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.RENDER_ATTACHMENT,
    },
    { includeSrgbSiblingViewFormat: true }
  );

  assert.deepEqual(createdTextures[0].desc.viewFormats, ['rgba8unorm-srgb']);
  const meta10 = reg.getTextureMeta(10);
  assert.equal(meta10.siblingViewFormat, 'rgba8unorm-srgb');
  assert.deepEqual(meta10.viewFormats, ['rgba8unorm-srgb']);
  assert.equal(meta10.textureGeneration, 1);

  let viewCalls10 = 0;
  const origCv10 = createdTextures[0].createView.bind(createdTextures[0]);
  createdTextures[0].createView = (vd) => {
    viewCalls10 += 1;
    return origCv10(vd);
  };

  // Distinct keys by format (linear vs sRGB sibling).
  const vLinearArray = reg.getTextureView(10);
  const vSrgbArray = reg.getTextureView(10, { format: 'rgba8unorm-srgb' });
  assert.notEqual(vLinearArray, vSrgbArray);
  assert.equal(vLinearArray.viewDesc.format, 'rgba8unorm');
  assert.equal(vLinearArray.viewDesc.dimension, '2d-array');
  assert.equal(vLinearArray.viewDesc.arrayLayerCount, 6);
  assert.equal(vLinearArray.viewDesc.mipLevelCount, 4);
  assert.equal(vSrgbArray.viewDesc.format, 'rgba8unorm-srgb');

  // Exact canonical 8-tuple hit for default view.
  const vLinearArrayHit = reg.getTextureView(10, {
    format: 'rgba8unorm',
    dimension: '2d-array',
    baseMipLevel: 0,
    mipLevelCount: 4,
    baseArrayLayer: 0,
    arrayLayerCount: 6,
    aspect: 'all',
  });
  assert.equal(vLinearArrayHit, vLinearArray);

  // Distinct keys by dimension ('cube' vs '2d' single layer vs '2d-array'), mip slice, and layer slice.
  const vCubeSrgb = reg.getTextureView(10, { format: 'rgba8unorm-srgb', dimension: 'cube' });
  const vMip1 = reg.getTextureView(10, { dimension: '2d', baseMipLevel: 1, mipLevelCount: 2, baseArrayLayer: 3, arrayLayerCount: 1 });
  const vMip2Remaining = reg.getTextureView(10, { dimension: '2d', baseMipLevel: 2, baseArrayLayer: 0 });
  const vMip2Explicit = reg.getTextureView(10, { dimension: '2d', baseMipLevel: 2, mipLevelCount: 2, baseArrayLayer: 0, arrayLayerCount: 1 });
  assert.equal(vMip2Explicit, vMip2Remaining);
  assert.notEqual(vCubeSrgb, vSrgbArray);
  assert.notEqual(vMip1, vMip2Remaining);
  assert.equal(viewCalls10, 5);

  const cached10 = reg.getCachedTextureViews(10);
  assert.equal(cached10.length, 5);
  assert.equal(
    cached10.some(
      (c) => c.key === buildTextureViewCacheKey(10, normalizeTextureViewDescriptor(10, { format: 'rgba8unorm-srgb', dimension: 'cube' }, meta10))
    ),
    true
  );

  // 2. Depth-stencil texture: distinct cached views by aspect ('all', 'depth-only', 'stencil-only').
  reg.createTexture2(11, {
    width: 1920,
    height: 1080,
    format: 'depth24plus-stencil8',
    usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT | GPU_TEXTURE_USAGE.TEXTURE_BINDING,
  });
  const vDepthAll = reg.getTextureView(11);
  const vDepthOnly = reg.getTextureView(11, { aspect: 'depth-only' });
  const vDepthOnlyHit = reg.getTextureView(11, { aspect: 'depth-only', format: 'depth24plus' });
  const vStencilOnly = reg.getTextureView(11, { aspect: 'stencil-only' });
  assert.equal(vDepthOnlyHit, vDepthOnly);
  assert.equal(vDepthOnly.viewDesc.format, 'depth24plus');
  assert.equal(vStencilOnly.viewDesc.format, 'stencil8');
  assert.notEqual(vDepthAll, vDepthOnly);
  assert.notEqual(vDepthOnly, vStencilOnly);

  // 3. Undeclared sibling format rejected unless declared in viewFormats or allowUndeclaredSibling is true.
  reg.createTexture2(12, {
    width: 64,
    height: 64,
    format: 'bgra8unorm',
    usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT | GPU_TEXTURE_USAGE.TEXTURE_BINDING,
  });
  assert.throws(() => reg.getTextureView(12, { format: 'bgra8unorm-srgb' }), RangeError);
  const vAllowed = reg.getTextureView(12, { format: 'bgra8unorm-srgb' }, { allowUndeclaredSibling: true });
  assert.equal(vAllowed.viewDesc.format, 'bgra8unorm-srgb');

  // 4. Out-of-bounds mip/layer/aspect/dimension validation throws RangeError.
  assert.throws(() => reg.getTextureView(10, { baseMipLevel: 4 }), RangeError);
  assert.throws(() => reg.getTextureView(10, { baseMipLevel: 2, mipLevelCount: 3 }), RangeError);
  assert.throws(() => reg.getTextureView(10, { baseArrayLayer: 6 }), RangeError);
  assert.throws(() => reg.getTextureView(10, { baseArrayLayer: 4, arrayLayerCount: 3 }), RangeError);
  assert.throws(() => reg.getTextureView(10, { dimension: 'cube', arrayLayerCount: 4 }), RangeError);
  assert.throws(() => reg.getTextureView(10, { aspect: 'bogus-aspect' }), RangeError);
  assert.throws(() => reg.getTextureView(10, { dimension: '4d' }), RangeError);
  assert.throws(() => reg.getTextureView(12, { aspect: 'depth-only' }), RangeError);

  const stats = reg.getStats();
  assert.equal(stats.viewCacheMisses, 9);
  assert.equal(stats.viewCacheHits, 3);
  assert.equal(stats.cachedViewCount, 9);
});

test('W178 automatic stale-view invalidation across 1080p <-> 4K recreation, format transition, retire, linked views, and destroyAll()', () => {
  const { device, createdTextures } = createMockWebGpuDevice({
    features: new Set([TEXTURE_COMPRESSION_BC_FEATURE]),
  });
  const reg = new GpuResourceRegistry({
    framesInFlight: 2,
    device,
    includeSrgbSiblingViewFormat: true,
  });

  // Initial stats default to 0.
  const initStats = reg.getStats();
  assert.equal(initStats.staleViewsInvalidated, 0);
  assert.equal(initStats.viewCacheHits, 0);
  assert.equal(initStats.viewCacheMisses, 0);
  assert.equal(initStats.textureRecreations, 0);
  assert.equal(initStats.cachedViewCount, 0);

  // Frame 1: Create 1080p bgra8unorm render target (with sibling bgra8unorm-srgb).
  reg.beginFrame(1);
  reg.createTexture(100, {
    width: 1920,
    height: 1080,
    format: 'bgra8unorm',
    usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT | GPU_TEXTURE_USAGE.TEXTURE_BINDING,
  });
  const default1080 = reg.get(RESOURCE_KIND.TEXTURE_VIEW, 100);
  const linear1080 = reg.getTextureView(100, { format: 'bgra8unorm' });
  const srgb1080 = reg.getTextureView(100, { format: 'bgra8unorm-srgb' });

  // Also register an explicitly linked external TEXTURE_VIEW handle bound to texture 100.
  const externalLinkedView = createNonDestroyableMock('external-rt-view-1080');
  reg.register(RESOURCE_KIND.TEXTURE_VIEW, 9001, externalLinkedView, {
    textureId: 100,
    label: 'rt-pass-view',
  });
  assert.equal(reg.isTextureViewValid(externalLinkedView), true);
  assert.equal(reg.has(RESOURCE_KIND.TEXTURE_VIEW, 9001), true);

  // Re-registering the exact same texture object in place does NOT invalidate views or bump generation.
  reg.register(RESOURCE_KIND.TEXTURE, 100, createdTextures[0], {
    byteSize: 1920 * 1080 * 4,
    label: 'same-1080p-tex',
  });
  assert.equal(reg.getTextureGeneration(100), 1);
  assert.equal(reg.getStats().textureRecreations, 0);
  assert.equal(reg.isTextureViewValid(linear1080), true);
  assert.equal(reg.isTextureViewValid(srgb1080), true);
  assert.equal(reg.getTextureView(100, { format: 'bgra8unorm-srgb' }), srgb1080);
  assert.equal(reg.getStats().viewCacheHits, 1);

  // Frame 2: Resize 1080p -> 4K (3840x2160) on handle 100 (in-place replacement).
  reg.beginFrame(2);
  reg.createTexture(100, {
    width: 3840,
    height: 2160,
    format: 'bgra8unorm',
    usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT | GPU_TEXTURE_USAGE.TEXTURE_BINDING,
  });
  assert.equal(reg.getTextureGeneration(100), 2);
  assert.equal(reg.isTextureViewValid(default1080), false);
  assert.equal(reg.isTextureViewValid(linear1080), false);
  assert.equal(reg.isTextureViewValid(srgb1080), false);
  assert.equal(reg.isTextureViewValid(externalLinkedView), false);
  assert.equal(reg.has(RESOURCE_KIND.TEXTURE_VIEW, 9001), false); // linked view auto-retired

  let stats = reg.getStats();
  assert.equal(stats.textureRecreations, 1);
  assert.equal(stats.staleViewsInvalidated, 4); // default1080 + linear1080 + srgb1080 + externalLinkedView
  assert.equal(stats.cachedViewCount, 0);

  const srgb4k = reg.getTextureView(100, { format: 'bgra8unorm-srgb' });
  assert.notEqual(srgb4k, srgb1080);
  assert.equal(srgb4k.textureInternalId, createdTextures[1].internalId);
  assert.equal(reg.isTextureViewValid(srgb4k), true);

  // Frame 3: Resize back 4K -> 1080p with a format transition (bgra8unorm -> rgba16float).
  reg.beginFrame(3);
  reg.createTexture(100, {
    width: 1920,
    height: 1080,
    format: 'rgba16float',
    usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT | GPU_TEXTURE_USAGE.TEXTURE_BINDING,
  });
  assert.equal(reg.getTextureGeneration(100), 3);
  assert.equal(reg.isTextureViewValid(srgb4k), false);
  assert.equal(reg.getStats().textureRecreations, 2);
  assert.equal(reg.getStats().staleViewsInvalidated, 6); // + default4k + srgb4k
  assert.throws(() => reg.getTextureView(100, { format: 'bgra8unorm-srgb' }), RangeError);
  const hdr1080 = reg.getTextureView(100, { format: 'rgba16float' });
  assert.equal(hdr1080.textureInternalId, createdTextures[2].internalId);
  assert.equal(reg.isTextureViewValid(hdr1080), true);

  // Pattern B: Explicit retire(TEXTURE, 100) then recreate on handle 100.
  assert.equal(reg.retire(RESOURCE_KIND.TEXTURE, 100), true);
  assert.equal(reg.getTextureGeneration(100), 0);
  assert.equal(reg.isTextureViewValid(hdr1080), false);
  assert.equal(reg.get(RESOURCE_KIND.TEXTURE_VIEW, 100), undefined);
  assert.equal(reg.getTextureView(100), undefined);
  assert.equal(reg.getStats().staleViewsInvalidated, 8); // + defaultHdr1080 + hdr1080

  // Recreate after explicit retirement bumps generation to 4 and counts as a recreation.
  reg.createTexture(100, {
    width: 3840,
    height: 2160,
    format: 'rgba8unorm',
    usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT | GPU_TEXTURE_USAGE.TEXTURE_BINDING,
  });
  assert.equal(reg.getTextureGeneration(100), 4);
  assert.equal(reg.getStats().textureRecreations, 3);
  const postRetireSrgb = reg.getTextureView(100, { format: 'rgba8unorm-srgb' });
  assert.equal(reg.isTextureViewValid(postRetireSrgb), true);

  // Manual invalidateTextureViews(100) clears cached & linked views immediately.
  const manuallyInvalidated = reg.invalidateTextureViews(100);
  assert.equal(manuallyInvalidated, 2); // default + postRetireSrgb
  assert.equal(reg.isTextureViewValid(postRetireSrgb), false);
  assert.equal(reg.getStats().cachedViewCount, 0);

  // Cache a new view, then trigger destroyAll() (device.lost recovery).
  const preLossView = reg.getTextureView(100, { format: 'rgba8unorm' });
  assert.equal(reg.isTextureViewValid(preLossView), true);
  const invalidatedBeforeLoss = reg.getStats().staleViewsInvalidated;

  reg.destroyAll();
  assert.equal(reg.getTextureGeneration(100), 0);
  assert.equal(reg.isTextureViewValid(preLossView), false);
  assert.equal(reg.getTextureView(100), undefined);
  assert.equal(reg.getStats().cachedViewCount, 0);
  assert.equal(reg.getStats().staleViewsInvalidated, invalidatedBeforeLoss + 1);

  // Re-registering handle 100 after destroyAll() MUST advance generation monotonically (to 5)
  // so pre-loss views/tokens can NEVER match the post-recovery texture.
  reg.createTexture(100, {
    width: 1920,
    height: 1080,
    format: 'rgba8unorm',
    usage: GPU_TEXTURE_USAGE.TEXTURE_BINDING,
  });
  assert.equal(reg.getTextureGeneration(100), 5);
  assert.equal(reg.isTextureViewValid(preLossView), false);
  assert.equal(reg.isTextureGenerationCurrent(100, 4), false);
  assert.equal(reg.isTextureGenerationCurrent(100, 5), true);
});

test('W178 direct register(RESOURCE_KIND.TEXTURE, ...) populates descriptor viewFormats metadata and invalidates cached views on overwrite', () => {
  const reg = new GpuResourceRegistry({ framesInFlight: 2 });
  let nextViewId = 1;
  const makeTex = (label, format, width, height) => ({
    label,
    format,
    width,
    height,
    destroyCount: 0,
    createView(desc) {
      return { viewId: nextViewId++, ownerLabel: label, desc };
    },
    destroy() {
      this.destroyCount += 1;
    },
  });

  const tex1 = makeTex('albedo-1080', 'rgba8unorm', 1920, 1080);
  const tex2 = makeTex('albedo-4k', 'rgba8unorm', 3840, 2160);

  reg.register(RESOURCE_KIND.TEXTURE, 7, tex1, {
    byteSize: 1920 * 1080 * 4,
    label: 'albedo',
    descriptor: {
      size: { width: 1920, height: 1080, depthOrArrayLayers: 1 },
      format: 'rgba8unorm',
      viewFormats: ['rgba8unorm-srgb'],
    },
  });

  const meta1 = reg.getTextureMeta(7);
  assert.equal(meta1.format, 'rgba8unorm');
  assert.equal(meta1.siblingViewFormat, 'rgba8unorm-srgb');
  assert.deepEqual(meta1.viewFormats, ['rgba8unorm-srgb']);
  assert.equal(meta1.textureGeneration, 1);

  const v1 = reg.getTextureView(7, { format: 'rgba8unorm-srgb' });
  assert.equal(v1.ownerLabel, 'albedo-1080');
  assert.equal(reg.isTextureViewValid(v1), true);

  // Overwriting handle 7 via register() replaces tex1, bumps generation, and invalidates v1.
  reg.register(RESOURCE_KIND.TEXTURE, 7, tex2, {
    byteSize: 3840 * 2160 * 4,
    label: 'albedo',
    format: 'rgba8unorm',
    width: 3840,
    height: 2160,
    viewFormats: ['rgba8unorm-srgb'],
  });

  assert.equal(reg.getTextureGeneration(7), 2);
  assert.equal(reg.isTextureViewValid(v1), false);
  assert.equal(reg.getStats().textureRecreations, 1);
  assert.equal(reg.getStats().staleViewsInvalidated, 1);

  const v2 = reg.getTextureView(7, { format: 'rgba8unorm-srgb' });
  assert.equal(v2.ownerLabel, 'albedo-4k');
  assert.equal(reg.isTextureViewValid(v2), true);
  assert.equal(reg.getTextureViewInfo(null), null);
  assert.equal(reg.isTextureViewValid(null), false);
  assert.equal(reg.isTextureViewValid({}), false);
});
