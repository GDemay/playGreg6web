import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const rtPoolMod = require('../web/workers/rt_pool.js');

const {
  DEFAULT_MAX_POOL_BYTES,
  DEFAULT_IDLE_FRAME_TTL,
  TEXTURE_USAGE,
  DEFAULT_TEXTURE_USAGE,
  FORMAT_BYTES_PER_PIXEL,
  normalizeRenderTargetDesc,
  makeRenderTargetKey,
  estimateRenderTargetBytes,
  RenderTargetPool,
} = rtPoolMod;

/**
 * Create a deterministic mock WebGPU device tracking texture creation, view creation,
 * and texture destruction counts.
 */
function createMockGpuDevice(options = {}) {
  let nextTexId = 1;
  let nextViewId = 1;
  const createdTextures = [];

  const device = {
    createdTextures,
    createTexture(descriptor) {
      const texId = nextTexId++;
      const texture = {
        mockId: texId,
        label: descriptor.label,
        descriptor: {
          ...descriptor,
          size: { ...descriptor.size },
        },
        viewCreateCount: 0,
        destroyCount: 0,
        destroyed: false,
        createView(viewDesc) {
          if (options.throwOnCreateView) {
            throw new Error('simulated createView failure');
          }
          this.viewCreateCount += 1;
          return {
            kind: 'GPUTextureView',
            viewId: nextViewId++,
            textureMockId: texId,
            viewDesc: viewDesc || null,
          };
        },
        destroy() {
          this.destroyed = true;
          this.destroyCount += 1;
        },
      };
      createdTextures.push(texture);
      return texture;
    },
  };

  return { device, createdTextures };
}

test('UMD export attaches PlayGta6RenderTargetPool on globalThis and module.exports', () => {
  assert.equal(globalThis.PlayGta6RenderTargetPool, rtPoolMod);
  assert.equal(typeof estimateRenderTargetBytes, 'function');
  assert.equal(typeof RenderTargetPool, 'function');
  assert.equal(typeof normalizeRenderTargetDesc, 'function');
  assert.equal(typeof makeRenderTargetKey, 'function');
  assert.equal(DEFAULT_IDLE_FRAME_TTL, 8);
  assert.equal(DEFAULT_MAX_POOL_BYTES, 256 * 1024 * 1024);
  assert.equal(DEFAULT_TEXTURE_USAGE, TEXTURE_USAGE.RENDER_ATTACHMENT | TEXTURE_USAGE.TEXTURE_BINDING);
  assert.ok(Object.isFrozen(rtPoolMod));
  assert.ok(Object.isFrozen(FORMAT_BYTES_PER_PIXEL));
});

test('descriptor normalization and keying handles flat and WebGPU size shapes consistently', () => {
  const usage = TEXTURE_USAGE.RENDER_ATTACHMENT | TEXTURE_USAGE.TEXTURE_BINDING;

  const flatKey = makeRenderTargetKey({
    width: 1280,
    height: 720,
    format: 'rgba16float',
    usage,
  });

  const objSizeKey = makeRenderTargetKey({
    size: { width: 1280, height: 720, depthOrArrayLayers: 1 },
    format: 'RGBA16FLOAT',
    usage,
    sampleCount: 1,
    mipLevelCount: 1,
    dimension: '2d',
  });

  const arrSizeKey = makeRenderTargetKey({
    size: [1280, 720, 1],
    format: 'rgba16float',
  });

  assert.equal(flatKey, objSizeKey);
  assert.equal(flatKey, arrSizeKey);

  // Every normalized axis must produce a distinct key when changed.
  const base = { width: 512, height: 512, format: 'rgba8unorm', usage };
  const baseKey = makeRenderTargetKey(base);
  assert.notEqual(makeRenderTargetKey({ ...base, width: 256 }), baseKey);
  assert.notEqual(makeRenderTargetKey({ ...base, height: 256 }), baseKey);
  assert.notEqual(makeRenderTargetKey({ ...base, format: 'bgra8unorm' }), baseKey);
  assert.notEqual(
    makeRenderTargetKey({ ...base, usage: TEXTURE_USAGE.RENDER_ATTACHMENT }),
    baseKey
  );
  assert.notEqual(makeRenderTargetKey({ ...base, sampleCount: 4 }), baseKey);
  assert.notEqual(makeRenderTargetKey({ ...base, mipLevelCount: 4 }), baseKey);
  assert.notEqual(makeRenderTargetKey({ ...base, depthOrArrayLayers: 4 }), baseKey);
  assert.notEqual(makeRenderTargetKey({ ...base, dimension: '3d' }), baseKey);

  // Validation errors on malformed descriptors.
  assert.throws(() => normalizeRenderTargetDesc(null), TypeError);
  assert.throws(() => normalizeRenderTargetDesc({ width: 0, height: 64, format: 'rgba8unorm' }), RangeError);
  assert.throws(() => normalizeRenderTargetDesc({ width: 64, height: -1, format: 'rgba8unorm' }), RangeError);
  assert.throws(() => normalizeRenderTargetDesc({ width: 64, height: 64, format: '' }), TypeError);
  assert.throws(() => normalizeRenderTargetDesc({ width: 64, height: 64, format: 'rgba8unorm', usage: 0 }), RangeError);
  assert.throws(() => normalizeRenderTargetDesc({ width: 64, height: 64, format: 'rgba8unorm', sampleCount: 3 }), RangeError);
  assert.throws(() => normalizeRenderTargetDesc({ width: 64, height: 64, format: 'rgba8unorm', mipLevelCount: 0 }), RangeError);
  assert.throws(() => normalizeRenderTargetDesc({ width: 64, height: 64, format: 'rgba8unorm', dimension: '4d' }), RangeError);
});

test('estimateRenderTargetBytes accurately computes bytes for all common formats, MSAA, mips, array layers, and 3D', () => {
  const expectedBpp = {
    rgba8unorm: 4,
    'rgba8unorm-srgb': 4,
    bgra8unorm: 4,
    'bgra8unorm-srgb': 4,
    rgb10a2unorm: 4,
    rg11b10ufloat: 4,
    rgba16float: 8,
    rgba32float: 16,
    r8unorm: 1,
    rg8unorm: 2,
    r16float: 2,
    rg16float: 4,
    r32float: 4,
    rg32float: 8,
    depth16unorm: 2,
    depth24plus: 4,
    'depth24plus-stencil8': 4,
    depth32float: 4,
    'depth32float-stencil8': 8,
    stencil8: 1,
  };

  for (const [format, bpp] of Object.entries(expectedBpp)) {
    assert.equal(
      estimateRenderTargetBytes({ width: 100, height: 50, format }),
      100 * 50 * bpp,
      `byte estimate mismatch for ${format}`
    );
  }

  // 4x MSAA 1920x1080 rgba16float
  assert.equal(
    estimateRenderTargetBytes({
      width: 1920,
      height: 1080,
      format: 'rgba16float',
      sampleCount: 4,
    }),
    1920 * 1080 * 8 * 4
  );

  // Mip chain: 256x256 rgba8unorm with 3 mip levels (256x256 + 128x128 + 64x64) * 4
  assert.equal(
    estimateRenderTargetBytes({
      width: 256,
      height: 256,
      format: 'rgba8unorm',
      mipLevelCount: 3,
    }),
    (256 * 256 + 128 * 128 + 64 * 64) * 4
  );

  // Non-power-of-two mip chain: 5x3 r8unorm with 3 mips (5x3 + 2x1 + 1x1 = 18 bytes)
  assert.equal(
    estimateRenderTargetBytes({
      width: 5,
      height: 3,
      format: 'r8unorm',
      mipLevelCount: 3,
    }),
    18
  );

  // 4-cascade shadow map array: 1024x1024x4 depth32float (4 layers stay constant across mips)
  assert.equal(
    estimateRenderTargetBytes({
      width: 1024,
      height: 1024,
      depthOrArrayLayers: 4,
      format: 'depth32float',
      mipLevelCount: 2,
    }),
    (1024 * 1024 + 512 * 512) * 4 * 4
  );

  // 3D texture mip chain halves depth per mip: 4x4x4 + 2x2x2 + 1x1x1 = 73 texels
  assert.equal(
    estimateRenderTargetBytes({
      width: 4,
      height: 4,
      depthOrArrayLayers: 4,
      dimension: '3d',
      format: 'rgba8unorm',
      mipLevelCount: 3,
    }),
    (64 + 8 + 1) * 4
  );

  // 1D texture: height stays 1
  assert.equal(
    estimateRenderTargetBytes({
      width: 16,
      dimension: '1d',
      format: 'rg16float',
      mipLevelCount: 2,
    }),
    (16 + 8) * 4
  );

  // Unknown format throws RangeError
  assert.throws(
    () => estimateRenderTargetBytes({ width: 64, height: 64, format: 'unknown-format' }),
    RangeError
  );
});

test('same-frame attachment aliasing reuses released target immediately and prevents double-release corruption', () => {
  const { device, createdTextures } = createMockGpuDevice();
  const pool = new RenderTargetPool({ device });

  const hdrDesc = {
    width: 1280,
    height: 720,
    format: 'rgba16float',
    usage: TEXTURE_USAGE.RENDER_ATTACHMENT | TEXTURE_USAGE.TEXTURE_BINDING,
  };
  const expectedBytes = 1280 * 720 * 8;

  // Pass 1 acquires HDR target (miss).
  const rt1 = pool.acquire(hdrDesc, 'gbuffer-hdr');
  assert.equal(rt1.reused, false);
  assert.equal(rt1.byteSize, expectedBytes);
  assert.equal(rt1.texture, createdTextures[0]);
  assert.equal(rt1.texture.viewCreateCount, 1);
  assert.equal(rt1.view.textureMockId, rt1.texture.mockId);
  assert.equal(createdTextures.length, 1);

  // Release rt1 within the same frame.
  assert.equal(pool.release(rt1), true);
  // Double-release on the same handle must be a safe no-op returning false.
  assert.equal(pool.release(rt1), false);
  assert.equal(pool.release(rt1.id), false);

  let stats = pool.getStats();
  assert.equal(stats.activeCount, 0);
  assert.equal(stats.idleCount, 1);
  assert.equal(stats.activeBytes, 0);
  assert.equal(stats.idleBytes, expectedBytes);

  // Pass 2 in the SAME frame acquires an identical descriptor -> aliased hit!
  const rt2 = pool.acquire(hdrDesc, 'postfx-bloom');
  assert.equal(rt2.reused, true);
  assert.equal(rt2.id, rt1.id);
  assert.equal(rt2.texture, rt1.texture);
  assert.equal(rt2.view, rt1.view);
  // Original handle object's `reused` field must remain untouched.
  assert.equal(rt1.reused, false);
  // Zero extra createTexture or createView calls on hit!
  assert.equal(createdTextures.length, 1);
  assert.equal(rt2.texture.viewCreateCount, 1);

  // Stale handle rt1 double-release while rt2 owns the underlying id must NOT release rt2!
  assert.equal(pool.release(rt1), false);
  assert.equal(pool.getStats().activeCount, 1);

  // While rt2 is active, acquiring a 2nd simultaneous HDR target must allocate a second texture.
  const rt3 = pool.acquire(hdrDesc, 'postfx-pingpong');
  assert.equal(rt3.reused, false);
  assert.notEqual(rt3.id, rt2.id);
  assert.notEqual(rt3.texture, rt2.texture);
  assert.equal(createdTextures.length, 2);

  // Release rt2 by handle and rt3 by numeric id.
  assert.equal(pool.release(rt2), true);
  assert.equal(pool.release(rt3.id), true);

  stats = pool.getStats();
  assert.equal(stats.hits, 1);
  assert.equal(stats.misses, 2);
  assert.ok(Math.abs(stats.hitRate - 1 / 3) < 1e-9);
  assert.equal(stats.activeCount, 0);
  assert.equal(stats.idleCount, 2);
  assert.equal(stats.totalBytes, expectedBytes * 2);
  assert.equal(stats.peakBytes, expectedBytes * 2);

  pool.destroy();
});

test('cross-frame steady-state reuse eliminates createTexture churn and tracks hitRate accurately', () => {
  const { device, createdTextures } = createMockGpuDevice();
  const pool = new RenderTargetPool({ device });

  const colorDesc = { width: 640, height: 360, format: 'rg11b10ufloat' };
  const depthDesc = {
    width: 640,
    height: 360,
    format: 'depth32float',
    usage: TEXTURE_USAGE.RENDER_ATTACHMENT,
  };

  // Run 10 frames: each frame uses 1 color + 1 depth, then aliases the color target in post-FX.
  for (let frame = 0; frame < 10; frame += 1) {
    const color = pool.acquire(colorDesc, 'scene-color');
    const depth = pool.acquire(depthDesc, 'scene-depth');
    assert.equal(color.reused, frame > 0);
    assert.equal(depth.reused, frame > 0);

    pool.release(color);
    const postFx = pool.acquire(colorDesc, 'postfx-tonemap');
    assert.equal(postFx.reused, true);
    assert.equal(postFx.texture, color.texture);

    pool.release(postFx);
    pool.release(depth);
    pool.endFrame();
  }

  // Only 2 physical textures ever allocated across 30 total acquires!
  assert.equal(createdTextures.length, 2);

  const stats = pool.getStats();
  assert.equal(stats.misses, 2);
  assert.equal(stats.hits, 28);
  assert.ok(Math.abs(stats.hitRate - 28 / 30) < 1e-9);
  assert.equal(stats.activeCount, 0);
  assert.equal(stats.idleCount, 2);
  assert.equal(stats.evictions, 0);
  assert.equal(stats.leakedReleases, 0);
  assert.equal(stats.currentFrame, 10);

  pool.destroy();
});

test('idleFrameTtl destroys idle textures unused for more than idleFrameTtl frames', () => {
  const { device, createdTextures } = createMockGpuDevice();
  const pool = new RenderTargetPool({ device, idleFrameTtl: 3 });

  const shadowDesc = { width: 512, height: 512, format: 'depth24plus' };
  const colorDesc = { width: 256, height: 256, format: 'rgba8unorm' };

  // Frame 0: acquire both, release both (lastUsedFrame = 0).
  const s = pool.acquire(shadowDesc);
  const c = pool.acquire(colorDesc);
  pool.release(s);
  pool.release(c);

  // endFrame #1 -> currentFrame = 1 (age = 1 <= 3): both survive.
  let res = pool.endFrame();
  assert.equal(res.frame, 1);
  assert.equal(res.evictedCount, 0);
  assert.equal(createdTextures[0].destroyed, false);
  assert.equal(createdTextures[1].destroyed, false);

  // Frame 1: touch only colorDesc (lastUsedFrame = 1); leave shadowDesc idle.
  const c1 = pool.acquire(colorDesc);
  assert.equal(c1.reused, true);
  pool.release(c1);
  // endFrame #2 -> currentFrame = 2 (shadow age = 2 <= 3, color age = 1 <= 3).
  res = pool.endFrame();
  assert.equal(res.evictedCount, 0);

  // Frame 2: idle for both -> endFrame #3 -> currentFrame = 3 (shadow age = 3 <= 3: still survives).
  res = pool.endFrame();
  assert.equal(res.evictedCount, 0);
  assert.equal(createdTextures[0].destroyed, false);

  // Frame 3: idle for both -> endFrame #4 -> currentFrame = 4 (shadow age = 4 > 3: EVICTED; color age = 3 <= 3: survives).
  res = pool.endFrame();
  assert.equal(res.frame, 4);
  assert.equal(res.evictedCount, 1);
  assert.equal(createdTextures[0].destroyed, true);
  assert.equal(createdTextures[0].destroyCount, 1);
  assert.equal(createdTextures[1].destroyed, false);

  const stats = pool.getStats();
  assert.equal(stats.idleCount, 1);
  assert.equal(stats.evictions, 1);
  assert.equal(stats.idleBytes, 256 * 256 * 4);

  // One more idle frame evicts colorDesc too (currentFrame = 5, 5 - 1 = 4 > 3).
  res = pool.endFrame();
  assert.equal(res.evictedCount, 1);
  assert.equal(createdTextures[1].destroyed, true);
  assert.equal(pool.getStats().idleCount, 0);
  assert.equal(pool.getStats().idleBytes, 0);
  assert.equal(pool.getStats().evictions, 2);

  pool.destroy();
});

test('idleFrameTtl = 0 permits same-frame aliasing and destroys idle targets at endFrame()', () => {
  const { device, createdTextures } = createMockGpuDevice();
  const pool = new RenderTargetPool({ device, idleFrameTtl: 0 });

  const desc = { width: 128, height: 128, format: 'rgba8unorm' };
  const a = pool.acquire(desc);
  pool.release(a);
  const b = pool.acquire(desc);
  assert.equal(b.reused, true);
  pool.release(b);

  const res = pool.endFrame();
  assert.equal(res.evictedCount, 1);
  assert.equal(createdTextures[0].destroyed, true);
  assert.equal(pool.getStats().idleCount, 0);

  pool.destroy();
});

test('MRU per-key reuse allows unused peak-demand duplicate targets to age out via idleFrameTtl', () => {
  const { device, createdTextures } = createMockGpuDevice();
  const pool = new RenderTargetPool({ device, idleFrameTtl: 2 });

  const desc = { width: 128, height: 128, format: 'rgba8unorm' };

  // Frame 0 spike needs 2 simultaneous targets.
  const t1 = pool.acquire(desc);
  const t2 = pool.acquire(desc);
  pool.release(t1);
  pool.release(t2);
  pool.endFrame(); // currentFrame = 1

  // Frames 1..3 only need 1 target per frame; MRU reuse keeps t2 warm so t1 expires after frame 2.
  for (let i = 0; i < 3; i += 1) {
    const t = pool.acquire(desc);
    assert.equal(t.reused, true);
    assert.equal(t.id, t2.id);
    pool.release(t);
    pool.endFrame();
  }

  assert.equal(createdTextures[0].destroyed, true, 'unused duplicate t1 should expire');
  assert.equal(createdTextures[1].destroyed, false, 'steadily reused t2 should stay warm');
  assert.equal(pool.getStats().idleCount, 1);
  assert.equal(pool.getStats().evictions, 1);

  pool.destroy();
});

test('maxPoolBytes LRU-evicts least-recently-used idle targets and enforces strictBudget', () => {
  const { device, createdTextures } = createMockGpuDevice();
  // Each 256x256 rgba8unorm = 262,144 bytes (256 KiB). Budget = 512 KiB (fits 2 targets).
  const oneTargetBytes = 256 * 256 * 4;
  const pool = new RenderTargetPool({
    device,
    maxPoolBytes: oneTargetBytes * 2,
    strictBudget: true,
  });

  const descA = { width: 256, height: 256, format: 'rgba8unorm' };
  const descB = { width: 256, height: 256, format: 'bgra8unorm' };
  const descC = { width: 256, height: 256, format: 'rg11b10ufloat' };
  const descD = { width: 256, height: 256, format: 'rgb10a2unorm' };

  // Acquire & release A, then B -> idle LRU order is [A (oldest), B (newest)].
  const a = pool.acquire(descA, 'rt-A');
  pool.release(a);
  const b = pool.acquire(descB, 'rt-B');
  pool.release(b);

  // Re-acquire & release A -> refreshes A so LRU order becomes [B (oldest), A (newest)].
  const a2 = pool.acquire(descA, 'rt-A-touch');
  assert.equal(a2.reused, true);
  pool.release(a2);

  // Acquire C (new format) -> must evict B (the true LRU), NOT A!
  const c = pool.acquire(descC, 'rt-C');
  assert.equal(c.reused, false);
  assert.equal(createdTextures[0].destroyed, false, 'recently-used A must survive LRU eviction');
  assert.equal(createdTextures[1].destroyed, true, 'least-recently-used B must be evicted');
  assert.equal(pool.getStats().evictions, 1);
  assert.equal(pool.getStats().totalBytes, oneTargetBytes * 2);

  // Now C is active (256 KiB) and A is idle (256 KiB).
  // Acquire D (256 KiB) -> evicts idle A so C + D fit exactly in 512 KiB active.
  const d = pool.acquire(descD, 'rt-D');
  assert.equal(d.reused, false);
  assert.equal(createdTextures[0].destroyed, true, 'idle A evicted to fit active D');
  assert.equal(pool.getStats().activeCount, 2);
  assert.equal(pool.getStats().idleCount, 0);
  assert.equal(pool.getStats().evictions, 2);

  // Both C and D are active (512 KiB / 512 KiB). Requesting a 3rd active target exceeds strictBudget!
  const missesBefore = pool.getStats().misses;
  assert.throws(
    () => pool.acquire(descA, 'rt-over-budget'),
    (err) => err instanceof RangeError && /budget exceeded/i.test(err.message)
  );
  assert.equal(pool.getStats().misses, missesBefore, 'failed budget acquire must not inflate misses');

  // Also test strictBudget: true fails BEFORE evicting idle textures if active + requested > maxPoolBytes.
  pool.release(c); // Now D is active (256 KiB), C is idle (256 KiB).
  const hugeDesc = { width: 512, height: 256, format: 'rgba8unorm' }; // 512 KiB > remaining 256 KiB active headroom
  assert.throws(() => pool.acquire(hugeDesc), RangeError);
  assert.equal(
    createdTextures[2].destroyed,
    false,
    'idle C should not be evicted when strictBudget allocation is impossible anyway'
  );

  pool.destroy();
});

test('strictBudget: false evicts idle targets first then allows over-budget active allocations', () => {
  const { device, createdTextures } = createMockGpuDevice();
  const oneTargetBytes = 128 * 128 * 4;
  const pool = new RenderTargetPool({
    device,
    maxPoolBytes: oneTargetBytes,
    strictBudget: false,
  });

  const descA = { width: 128, height: 128, format: 'rgba8unorm' };
  const descB = { width: 128, height: 128, format: 'bgra8unorm' };
  const descC = { width: 128, height: 128, format: 'r32float' };

  const a = pool.acquire(descA);
  pool.release(a);

  // Acquiring B evicts idle A first.
  const b = pool.acquire(descB);
  assert.equal(createdTextures[0].destroyed, true);
  assert.equal(pool.getStats().evictions, 1);

  // While B is still active (at budget), acquiring C is permitted when strictBudget is false.
  const c = pool.acquire(descC);
  assert.equal(c.reused, false);
  assert.equal(pool.getStats().activeCount, 2);
  assert.equal(pool.getStats().activeBytes, oneTargetBytes * 2);
  assert.equal(pool.getStats().peakBytes, oneTargetBytes * 2);

  pool.destroy();
});

test('invalidateAll() destroys all idle targets on resize while leaving active targets intact', () => {
  const { device, createdTextures } = createMockGpuDevice();
  const pool = new RenderTargetPool({ device });

  const oldDesc1 = { width: 1280, height: 720, format: 'rgba16float' };
  const oldDesc2 = { width: 1280, height: 720, format: 'depth32float' };
  const activeDesc = { width: 256, height: 256, format: 'rgba8unorm' };

  const rt1 = pool.acquire(oldDesc1);
  const rt2 = pool.acquire(oldDesc2);
  const activeRt = pool.acquire(activeDesc);

  pool.release(rt1);
  pool.release(rt2);

  const destroyedCount = pool.invalidateAll();
  assert.equal(destroyedCount, 2);
  assert.equal(createdTextures[0].destroyed, true);
  assert.equal(createdTextures[1].destroyed, true);
  assert.equal(createdTextures[2].destroyed, false, 'active texture must not be destroyed by invalidateAll()');

  const stats = pool.getStats();
  assert.equal(stats.idleCount, 0);
  assert.equal(stats.idleBytes, 0);
  assert.equal(stats.activeCount, 1);
  assert.equal(stats.activeBytes, 256 * 256 * 4);
  assert.equal(stats.totalBytes, 256 * 256 * 4);

  // Re-acquiring oldDesc1 after invalidateAll() must allocate a fresh texture.
  const rt1New = pool.acquire(oldDesc1);
  assert.equal(rt1New.reused, false);
  assert.notEqual(rt1New.texture, rt1.texture);

  pool.release(activeRt);
  pool.release(rt1New);
  pool.destroy();
});

test('endFrame() leak detection supports both autoReleaseLeaked=true and autoReleaseLeaked=false', () => {
  // 1. autoReleaseLeaked: true (default)
  {
    const { device } = createMockGpuDevice();
    const leaks = [];
    const warnings = [];
    const pool = new RenderTargetPool({
      device,
      autoReleaseLeaked: true,
      onLeak: (info) => leaks.push(info),
      warn: (msg) => warnings.push(msg),
    });

    const desc = { width: 320, height: 180, format: 'rgba8unorm' };
    const leaked = pool.acquire(desc, 'forgotten-rt');
    assert.equal(pool.getStats().activeCount, 1);

    const frameResult = pool.endFrame();
    assert.equal(frameResult.leakedCount, 1);
    assert.equal(frameResult.autoReleasedCount, 1);
    assert.equal(frameResult.activeCount, 0);
    assert.equal(frameResult.idleCount, 1);

    assert.equal(leaks.length, 1);
    assert.equal(leaks[0].id, leaked.id);
    assert.equal(leaks[0].label, 'forgotten-rt');
    assert.equal(leaks[0].autoReleased, true);
    assert.equal(leaks[0].frame, 0);
    assert.equal(warnings.length, 1);

    const stats = pool.getStats();
    assert.equal(stats.leakedReleases, 1);
    assert.equal(stats.activeCount, 0);
    assert.equal(stats.idleCount, 1);

    // Next frame can reuse the auto-released target!
    const next = pool.acquire(desc);
    assert.equal(next.reused, true);
    assert.equal(next.texture, leaked.texture);

    pool.destroy();
  }

  // 2. autoReleaseLeaked: false (warn only; target remains active)
  {
    const { device } = createMockGpuDevice();
    const leaks = [];
    const warnings = [];
    const pool = new RenderTargetPool({
      device,
      autoReleaseLeaked: false,
      onLeak: (info) => leaks.push(info),
      warn: (msg) => warnings.push(msg),
    });

    const desc = { width: 320, height: 180, format: 'rgba8unorm' };
    const leaked = pool.acquire(desc, 'still-in-use');

    const frameResult = pool.endFrame();
    assert.equal(frameResult.leakedCount, 1);
    assert.equal(frameResult.autoReleasedCount, 0);
    assert.equal(frameResult.activeCount, 1);
    assert.equal(frameResult.idleCount, 0);

    assert.equal(leaks.length, 1);
    assert.equal(leaks[0].autoReleased, false);
    assert.equal(warnings.length, 1);
    assert.match(warnings[0], /Leaked active render target/);

    const stats = pool.getStats();
    assert.equal(stats.leakedReleases, 0);
    assert.equal(stats.activeCount, 1);

    // Explicit release later still works cleanly.
    assert.equal(pool.release(leaked), true);
    assert.equal(pool.getStats().activeCount, 0);
    assert.equal(pool.getStats().idleCount, 1);

    pool.destroy();
  }
});

test('destroy() destroys all active and idle targets cleanly, is idempotent, and guards post-destroy calls', () => {
  const { device, createdTextures } = createMockGpuDevice();
  const pool = new RenderTargetPool({ device });

  const active1 = pool.acquire({ width: 128, height: 128, format: 'rgba8unorm' });
  const idle1 = pool.acquire({ width: 128, height: 128, format: 'depth24plus' });
  const idle2 = pool.acquire({ width: 64, height: 64, format: 'rgba16float' });
  pool.release(idle1);
  pool.release(idle2);

  assert.equal(pool.isDestroyed, false);
  const peakBeforeDestroy = pool.getStats().peakBytes;
  assert.ok(peakBeforeDestroy > 0);

  pool.destroy();
  assert.equal(pool.isDestroyed, true);
  assert.equal(createdTextures.length, 3);
  for (const tex of createdTextures) {
    assert.equal(tex.destroyed, true);
    assert.equal(tex.destroyCount, 1);
  }

  const afterStats = pool.getStats();
  assert.equal(afterStats.activeCount, 0);
  assert.equal(afterStats.idleCount, 0);
  assert.equal(afterStats.activeBytes, 0);
  assert.equal(afterStats.idleBytes, 0);
  assert.equal(afterStats.totalBytes, 0);
  assert.equal(afterStats.peakBytes, peakBeforeDestroy);
  assert.equal(afterStats.misses, 3);

  // Second destroy() is a safe idempotent no-op (no double texture.destroy()).
  pool.destroy();
  for (const tex of createdTextures) {
    assert.equal(tex.destroyCount, 1);
  }

  // Operations after destroy throw cleanly while getStats() stays safe.
  assert.throws(() => pool.acquire({ width: 64, height: 64, format: 'rgba8unorm' }), /destroyed/i);
  assert.throws(() => pool.release(active1), /destroyed/i);
  assert.throws(() => pool.endFrame(), /destroyed/i);
  assert.throws(() => pool.invalidateAll(), /destroyed/i);
  assert.doesNotThrow(() => pool.getStats());
});

test('constructor validation and texture cleanup when createView throws', () => {
  assert.throws(() => new RenderTargetPool(null), TypeError);
  assert.throws(() => new RenderTargetPool({}), TypeError);
  assert.throws(
    () => new RenderTargetPool({ device: { createTexture() {} }, maxPoolBytes: 0 }),
    RangeError
  );
  assert.throws(
    () => new RenderTargetPool({ device: { createTexture() {} }, idleFrameTtl: -1 }),
    RangeError
  );
  assert.throws(
    () => new RenderTargetPool({ device: { createTexture() {} }, onLeak: 'not-a-fn' }),
    TypeError
  );

  // If texture.createView() throws, the newly allocated GPUTexture is destroyed immediately with no accounting leak.
  const { device, createdTextures } = createMockGpuDevice({ throwOnCreateView: true });
  const pool = new RenderTargetPool({ device });

  assert.throws(
    () => pool.acquire({ width: 64, height: 64, format: 'rgba8unorm' }),
    /createView failure/
  );
  assert.equal(createdTextures.length, 1);
  assert.equal(createdTextures[0].destroyed, true);
  assert.equal(pool.getStats().activeCount, 0);
  assert.equal(pool.getStats().totalBytes, 0);
  assert.equal(pool.getStats().misses, 0);

  pool.destroy();
});
