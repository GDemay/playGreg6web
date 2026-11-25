import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const rtPoolMod = require('../web/workers/rt_pool.js');

const {
  DEFAULT_MAX_POOL_BYTES,
  DEFAULT_IDLE_FRAME_TTL,
  DEFAULT_OVERSIZED_TARGET_BYTES,
  DEFAULT_MAX_TEXTURE_DIMENSION_2D,
  TEXTURE_USAGE,
  DEFAULT_TEXTURE_USAGE,
  FORMAT_BYTES_PER_PIXEL,
  DXGI_FORMAT_ALIASES,
  COMPATIBLE_VIEW_FORMATS,
  normalizeRenderTargetFormat,
  isCompatibleViewFormat,
  normalizeViewFormats,
  normalizeSampleCount,
  sanitizeRenderTargetDesc,
  normalizeRenderTargetDesc,
  makeRenderTargetKey,
  makeTextureViewKey,
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
    ...(options.limits !== undefined ? { limits: options.limits } : {}),
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
          if (options.throwOnDestroy) {
            throw new Error('simulated device-lost destroy error');
          }
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
  assert.equal(DEFAULT_MAX_POOL_BYTES, 512 * 1024 * 1024);
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

// ---------------------------------------------------------------------------
// G45: Xbox Series 4K/Windowed Hardening, Format Normalization & Resize Safety
// ---------------------------------------------------------------------------

test('G45 clamps malformed, zero, negative, and oversized dimensions to [1, maxTextureDimension2D] and tracks dimensionClampCount', () => {
  assert.equal(DEFAULT_MAX_TEXTURE_DIMENSION_2D, 8192);
  assert.equal(typeof sanitizeRenderTargetDesc, 'function');

  // 1. Device without limits defaults to maxTextureDimension2D = 8192.
  {
    const { device, createdTextures } = createMockGpuDevice();
    const pool = new RenderTargetPool({ device, maxPoolBytes: Infinity });
    assert.equal(pool.maxTextureDimension2D, 8192);
    assert.equal(pool.getStats().maxTextureDimension2D, 8192);
    assert.equal(pool.getStats().dimensionClampCount, 0);

    // Valid 4K (3840x2160) & windowed (1280x720) do NOT clamp.
    const rt4k = pool.acquire({ width: 3840, height: 2160, format: 'rgba8unorm' });
    assert.equal(rt4k.desc.width, 3840);
    assert.equal(rt4k.desc.height, 2160);
    assert.equal(createdTextures[0].descriptor.size.width, 3840);
    assert.equal(createdTextures[0].descriptor.size.height, 2160);
    assert.equal(pool.getStats().dimensionClampCount, 0);

    // Zero width/height clamped to 1x1, mipLevelCount clamped to 1.
    const rtZero = pool.acquire({ width: 0, height: 0, format: 'rgba8unorm', mipLevelCount: 6 });
    assert.equal(rtZero.desc.width, 1);
    assert.equal(rtZero.desc.height, 1);
    assert.equal(rtZero.desc.mipLevelCount, 1);
    assert.equal(createdTextures[1].descriptor.size.width, 1);
    assert.equal(createdTextures[1].descriptor.size.height, 1);
    assert.equal(pool.getStats().dimensionClampCount, 1);

    // Negative / NaN / Infinity dimensions clamped safely to 1.
    const rtMalformed = pool.acquire({ width: -640, height: Number.NaN, format: 'bgra8unorm' });
    assert.equal(rtMalformed.desc.width, 1);
    assert.equal(rtMalformed.desc.height, 1);
    assert.equal(pool.getStats().dimensionClampCount, 2);

    // Oversized > 8192 clamped to 8192.
    const rtOversized = pool.acquire({ size: [16384, 12000], format: 'r8unorm' });
    assert.equal(rtOversized.desc.width, 8192);
    assert.equal(rtOversized.desc.height, 8192);
    assert.equal(pool.getStats().dimensionClampCount, 3);

    // Fractional dimensions floored and counted as clamped.
    const rtFrac = pool.acquire({ size: { width: 1280.9, height: 720.2 }, format: 'rgba8unorm' });
    assert.equal(rtFrac.desc.width, 1280);
    assert.equal(rtFrac.desc.height, 720);
    assert.equal(pool.getStats().dimensionClampCount, 4);

    // Releasing rtZero (1x1 rgba8unorm) and re-acquiring with malformed { width: -5, height: 0 }
    // reuses the pooled 1x1 texture AND increments dimensionClampCount + hits.
    pool.release(rtZero);
    const rtZeroHit = pool.acquire({ width: -5, height: 0, format: 'rgba8unorm' });
    assert.equal(rtZeroHit.reused, true);
    assert.equal(rtZeroHit.texture, rtZero.texture);
    assert.equal(pool.getStats().hits, 1);
    assert.equal(pool.getStats().dimensionClampCount, 5);

    pool.destroy();
    // getStats() remains safe after destroy and retains counters & maxTextureDimension2D.
    const postDestroy = pool.getStats();
    assert.equal(postDestroy.dimensionClampCount, 5);
    assert.equal(postDestroy.maxTextureDimension2D, 8192);
  }

  // 2. Device with explicit limits.maxTextureDimension2D = 4096 clamps 4K-oversized requests to 4096.
  {
    const { device } = createMockGpuDevice({ limits: { maxTextureDimension2D: 4096 } });
    const pool = new RenderTargetPool({ device, maxPoolBytes: Infinity });
    assert.equal(pool.maxTextureDimension2D, 4096);

    const rtClamped = pool.acquire({ width: 7680, height: 4320, format: 'rgba8unorm' });
    assert.equal(rtClamped.desc.width, 4096);
    assert.equal(rtClamped.desc.height, 4096);
    assert.equal(pool.getStats().dimensionClampCount, 1);

    // 3840x2160 fits within 4096 without clamping.
    const rt4k = pool.acquire({ width: 3840, height: 2160, format: 'rgba8unorm' });
    assert.equal(rt4k.desc.width, 3840);
    assert.equal(rt4k.desc.height, 2160);
    assert.equal(pool.getStats().dimensionClampCount, 1);

    pool.destroy();
  }
});

test('G45 normalizes DXGI/Xbox color & depth-stencil formats while preserving valid WebGPU formats untouched', () => {
  assert.ok(Object.isFrozen(DXGI_FORMAT_ALIASES));
  assert.equal(typeof normalizeRenderTargetFormat, 'function');

  // Required DXGI/Xbox format normalizations.
  const requiredMappings = [
    ['r8g8b8a8_unorm', 'rgba8unorm'],
    ['R8G8B8A8_UNORM', 'rgba8unorm'],
    ['DXGI_FORMAT_R8G8B8A8_UNORM', 'rgba8unorm'],
    ['b8g8r8a8_unorm', 'bgra8unorm'],
    ['DXGI_FORMAT_B8G8R8A8_UNORM', 'bgra8unorm'],
    ['d32_float', 'depth32float'],
    ['DXGI_FORMAT_D32_FLOAT', 'depth32float'],
    ['d24_unorm_s8_uint', 'depth24plus-stencil8'],
    ['DXGI_FORMAT_D24_UNORM_S8_UINT', 'depth24plus-stencil8'],
    ['d32_float_s8x24_uint', 'depth32float-stencil8'],
    ['DXGI_FORMAT_D32_FLOAT_S8X24_UINT', 'depth32float-stencil8'],
    // Additional Xbox Series HDR/color/depth aliases.
    ['r16g16b16a16_float', 'rgba16float'],
    ['r11g11b10_float', 'rg11b10ufloat'],
    ['r10g10b10a2_unorm', 'rgb10a2unorm'],
    ['b8g8r8a8_unorm_srgb', 'bgra8unorm-srgb'],
    ['r8g8b8a8_unorm_srgb', 'rgba8unorm-srgb'],
    ['d16_unorm', 'depth16unorm'],
  ];

  for (const [input, expected] of requiredMappings) {
    assert.equal(
      normalizeRenderTargetFormat(input),
      expected,
      `expected ${input} -> ${expected}`
    );
  }

  // Valid WebGPU formats pass through untouched.
  const validWebGpuFormats = [
    'rgba8unorm',
    'bgra8unorm',
    'rgba16float',
    'depth32float',
    'depth24plus-stencil8',
    'depth32float-stencil8',
  ];
  for (const fmt of validWebGpuFormats) {
    assert.equal(normalizeRenderTargetFormat(fmt), fmt);
  }

  // Byte estimation works seamlessly with DXGI aliases.
  assert.equal(estimateRenderTargetBytes({ width: 3840, height: 2160, format: 'r8g8b8a8_unorm' }), 3840 * 2160 * 4);
  assert.equal(estimateRenderTargetBytes({ width: 3840, height: 2160, format: 'd32_float' }), 3840 * 2160 * 4);
  assert.equal(estimateRenderTargetBytes({ width: 3840, height: 2160, format: 'd24_unorm_s8_uint' }), 3840 * 2160 * 4);
  assert.equal(estimateRenderTargetBytes({ width: 3840, height: 2160, format: 'd32_float_s8x24_uint' }), 3840 * 2160 * 8);

  // Pool acquisition normalizes DXGI formats, shares pooled textures with canonical WebGPU formats,
  // and only increments formatNormalizeCount when a format alias rewrite occurs.
  const { device, createdTextures } = createMockGpuDevice();
  const pool = new RenderTargetPool({ device });

  // 1. Acquire with canonical WebGPU formats -> formatNormalizeCount stays 0.
  const webgpuColor = pool.acquire({ width: 1280, height: 720, format: 'rgba8unorm' });
  const webgpuDepth = pool.acquire({ width: 1280, height: 720, format: 'depth24plus-stencil8' });
  assert.equal(webgpuColor.desc.format, 'rgba8unorm');
  assert.equal(webgpuDepth.desc.format, 'depth24plus-stencil8');
  assert.equal(pool.getStats().formatNormalizeCount, 0);
  pool.release(webgpuColor);
  pool.release(webgpuDepth);

  // 2. Re-acquire using DXGI aliases -> HITS the exact same pooled textures AND increments formatNormalizeCount!
  const dxgiColor = pool.acquire({ width: 1280, height: 720, format: 'r8g8b8a8_unorm' });
  assert.equal(dxgiColor.reused, true);
  assert.equal(dxgiColor.desc.format, 'rgba8unorm');
  assert.equal(dxgiColor.texture, webgpuColor.texture);
  assert.equal(pool.getStats().formatNormalizeCount, 1);

  const dxgiDepth = pool.acquire({ width: 1280, height: 720, format: 'd24_unorm_s8_uint' });
  assert.equal(dxgiDepth.reused, true);
  assert.equal(dxgiDepth.desc.format, 'depth24plus-stencil8');
  assert.equal(dxgiDepth.texture, webgpuDepth.texture);
  assert.equal(pool.getStats().formatNormalizeCount, 2);

  // 3. Acquire new targets with remaining required DXGI aliases -> device.createTexture receives canonical WebGPU format.
  const dxgiBgra = pool.acquire({ width: 1920, height: 1080, format: 'b8g8r8a8_unorm' });
  const dxgiD32 = pool.acquire({ width: 1920, height: 1080, format: 'd32_float' });
  const dxgiD32S8 = pool.acquire({ width: 1920, height: 1080, format: 'd32_float_s8x24_uint' });

  assert.equal(dxgiBgra.desc.format, 'bgra8unorm');
  assert.equal(dxgiD32.desc.format, 'depth32float');
  assert.equal(dxgiD32S8.desc.format, 'depth32float-stencil8');
  assert.equal(createdTextures[2].descriptor.format, 'bgra8unorm');
  assert.equal(createdTextures[3].descriptor.format, 'depth32float');
  assert.equal(createdTextures[4].descriptor.format, 'depth32float-stencil8');
  assert.equal(createdTextures.length, 5);

  const stats = pool.getStats();
  assert.equal(stats.formatNormalizeCount, 5);
  assert.equal(stats.hits, 2);
  assert.equal(stats.misses, 5);

  pool.destroy();
});

test('G45 normalizes sampleCount strictly to {1, 4} and enforces WebGPU multisampling constraints', () => {
  assert.equal(typeof normalizeSampleCount, 'function');

  assert.deepEqual(normalizeSampleCount(undefined), { sampleCount: 1, normalized: false });
  assert.deepEqual(normalizeSampleCount(1), { sampleCount: 1, normalized: false });
  assert.deepEqual(normalizeSampleCount(4), { sampleCount: 4, normalized: false });
  assert.deepEqual(normalizeSampleCount(0), { sampleCount: 1, normalized: true });
  assert.deepEqual(normalizeSampleCount(-4), { sampleCount: 1, normalized: true });
  assert.deepEqual(normalizeSampleCount(Number.NaN), { sampleCount: 1, normalized: true });
  assert.deepEqual(normalizeSampleCount(2), { sampleCount: 1, normalized: true });
  assert.deepEqual(normalizeSampleCount(3), { sampleCount: 1, normalized: true });
  assert.deepEqual(normalizeSampleCount(8), { sampleCount: 4, normalized: true });
  assert.deepEqual(normalizeSampleCount(16), { sampleCount: 4, normalized: true });

  // WebGPU constraints disallow MSAA for mip chains, 3D textures, texture arrays, or STORAGE_BINDING.
  assert.deepEqual(normalizeSampleCount(4, { mipLevelCount: 2 }), { sampleCount: 1, normalized: true });
  assert.deepEqual(normalizeSampleCount(4, { dimension: '3d' }), { sampleCount: 1, normalized: true });
  assert.deepEqual(normalizeSampleCount(4, { depthOrArrayLayers: 2 }), { sampleCount: 1, normalized: true });
  assert.deepEqual(
    normalizeSampleCount(4, { usage: TEXTURE_USAGE.RENDER_ATTACHMENT | TEXTURE_USAGE.STORAGE_BINDING }),
    { sampleCount: 1, normalized: true }
  );

  const { device, createdTextures } = createMockGpuDevice();
  const pool = new RenderTargetPool({ device });

  // Request 8x MSAA -> clamped to 4x MSAA.
  const msaa8 = pool.acquire({ width: 640, height: 360, format: 'rgba8unorm', sampleCount: 8 });
  assert.equal(msaa8.desc.sampleCount, 4);
  assert.equal(createdTextures[0].descriptor.sampleCount, 4);
  assert.equal(pool.getStats().sampleCountNormalizeCount, 1);

  // Release & re-acquire with canonical 4x MSAA -> pool HIT!
  pool.release(msaa8);
  const msaa4 = pool.acquire({ width: 640, height: 360, format: 'rgba8unorm', sampleCount: 4 });
  assert.equal(msaa4.reused, true);
  assert.equal(msaa4.texture, msaa8.texture);
  assert.equal(pool.getStats().sampleCountNormalizeCount, 1);

  // Request 2x MSAA -> normalized to 1x; release & re-acquire with 1x -> pool HIT!
  const msaa2 = pool.acquire({ width: 640, height: 360, format: 'rgba8unorm', sampleCount: 2 });
  assert.equal(msaa2.desc.sampleCount, 1);
  assert.equal(createdTextures[1].descriptor.sampleCount, 1);
  pool.release(msaa2);

  const msaa1 = pool.acquire({ width: 640, height: 360, format: 'rgba8unorm', sampleCount: 1 });
  assert.equal(msaa1.reused, true);
  assert.equal(msaa1.texture, msaa2.texture);

  // Request 4x MSAA with mipLevelCount: 3 -> normalized to 1x so WebGPU createTexture stays valid.
  const msaaMip = pool.acquire({
    width: 256,
    height: 256,
    format: 'rgba8unorm',
    sampleCount: 4,
    mipLevelCount: 3,
  });
  assert.equal(msaaMip.desc.sampleCount, 1);
  assert.equal(msaaMip.desc.mipLevelCount, 3);
  assert.equal(pool.getStats().sampleCountNormalizeCount, 3);

  pool.destroy();
});

test('G45 resize & eviction destroy old textures safely without double-destroy or stale GPUTextureViews', () => {
  const { device, createdTextures } = createMockGpuDevice();
  const pool = new RenderTargetPool({ device });

  // Simulate Xbox Series 4K Frame-1 splash + depth passes.
  const color4k = pool.acquire({ width: 3840, height: 2160, format: 'r8g8b8a8_unorm' }, 'splash-4k-color');
  const depth4k = pool.acquire({ width: 3840, height: 2160, format: 'd32_float_s8x24_uint' }, 'splash-4k-depth');
  const inFlight4k = pool.acquire({ width: 3840, height: 2160, format: 'b8g8r8a8_unorm' }, 'inflight-4k-ui');

  assert.ok(color4k.view !== null);
  assert.ok(depth4k.view !== null);
  assert.ok(inFlight4k.view !== null);
  assert.equal(color4k.destroyed, false);

  // Release the two completed 4K targets; keep inFlight4k active across resize.
  assert.equal(pool.release(color4k), true);
  assert.equal(pool.release(depth4k), true);

  // Resize from 4K to 1280x720 windowed.
  const destroyedIdle = pool.resize(1280, 720);
  assert.equal(destroyedIdle, 2);

  // Evicted idle textures are destroyed exactly once and their handles NEVER return stale GPUTextureViews!
  assert.equal(createdTextures[0].destroyed, true);
  assert.equal(createdTextures[0].destroyCount, 1);
  assert.equal(createdTextures[1].destroyed, true);
  assert.equal(createdTextures[1].destroyCount, 1);
  assert.equal(color4k.destroyed, true);
  assert.equal(color4k.texture.destroyed, true);
  assert.equal(color4k.view, null);
  assert.equal(depth4k.destroyed, true);
  assert.equal(depth4k.texture.destroyed, true);
  assert.equal(depth4k.view, null);

  // In-flight target stays valid while still active.
  assert.equal(createdTextures[2].destroyed, false);
  assert.equal(inFlight4k.destroyed, false);
  assert.ok(inFlight4k.view !== null);

  // Second resize while idle is empty is a safe no-op with zero double-destroy.
  assert.equal(pool.resize(1280, 720), 0);
  assert.equal(createdTextures[0].destroyCount, 1);
  assert.equal(createdTextures[1].destroyCount, 1);

  // Releasing inFlight4k after resize immediately retires & destroys it instead of putting stale 4K back in idle pool!
  assert.equal(pool.release(inFlight4k), true);
  assert.equal(createdTextures[2].destroyed, true);
  assert.equal(createdTextures[2].destroyCount, 1);
  assert.equal(inFlight4k.destroyed, true);
  assert.equal(inFlight4k.texture.destroyed, true);
  assert.equal(inFlight4k.view, null);
  assert.equal(pool.getStats().idleCount, 0);
  assert.equal(pool.getStats().activeCount, 0);
  assert.equal(pool.getStats().evictions, 3);

  // Duplicate release on retired handle is a safe no-op with zero double-destroy.
  assert.equal(pool.release(inFlight4k), false);
  assert.equal(createdTextures[2].destroyCount, 1);

  // Subsequent windowed acquire allocates a fresh target (never reuses retired 4K target).
  const winRt = pool.acquire({ width: 1280, height: 720, format: 'bgra8unorm' }, 'windowed-ui');
  assert.equal(winRt.reused, false);
  assert.ok(winRt.view !== null);
  assert.equal(winRt.destroyed, false);

  // Full teardown + idempotent second destroy() never double-destroys any texture.
  pool.destroy();
  pool.destroy();
  assert.equal(winRt.destroyed, true);
  assert.equal(winRt.texture.destroyed, true);
  assert.equal(winRt.view, null);
  for (const tex of createdTextures) {
    assert.equal(tex.destroyed, true);
    assert.equal(tex.destroyCount, 1, `texture #${tex.mockId} must be destroyed exactly once`);
  }
});

test('G45 guards against externally-destroyed pooled textures and device-lost destroy() errors', () => {
  // 1. Externally destroyed idle texture must be skipped on acquire (no stale GPUTextureView, no double-destroy).
  {
    const { device, createdTextures } = createMockGpuDevice();
    const pool = new RenderTargetPool({ device });

    const desc = { width: 512, height: 512, format: 'rgba8unorm' };
    const first = pool.acquire(desc);
    const firstTex = createdTextures[0];
    pool.release(first);

    // Simulate external destruction (e.g. device loss or swapchain reset destroying underlying texture).
    firstTex.destroy();
    assert.equal(firstTex.destroyCount, 1);
    assert.equal(first.view, null, 'handle view must immediately become null when underlying texture is destroyed');
    assert.equal(first.destroyed, true);

    // Next acquire must detect the destroyed idle entry, skip it without double-destroying, and allocate fresh.
    const second = pool.acquire(desc);
    assert.equal(second.reused, false);
    assert.notEqual(second.texture, firstTex);
    assert.ok(second.view !== null);
    assert.equal(firstTex.destroyCount, 1, 'externally-destroyed texture must not be double-destroyed');
    assert.equal(pool.getStats().evictions, 1);
    assert.equal(pool.getStats().misses, 2);

    pool.destroy();
    assert.equal(firstTex.destroyCount, 1);
    assert.equal(createdTextures[1].destroyCount, 1);
  }

  // 2. texture.destroy() throwing on device loss does not crash invalidateAll(), eviction, or destroy().
  {
    const { device, createdTextures } = createMockGpuDevice({ throwOnDestroy: true });
    const pool = new RenderTargetPool({ device });

    const rt = pool.acquire({ width: 128, height: 128, format: 'rgba8unorm' });
    pool.release(rt);
    assert.doesNotThrow(() => pool.invalidateAll());
    assert.equal(createdTextures[0].destroyCount, 1);
    assert.equal(rt.view, null);
    assert.doesNotThrow(() => pool.destroy());
    assert.equal(createdTextures[0].destroyCount, 1);
  }

  // 3. In-flight active target flagged by invalidateAll() is retired cleanly on endFrame() auto-release.
  {
    const { device, createdTextures } = createMockGpuDevice();
    const pool = new RenderTargetPool({ device, autoReleaseLeaked: true });

    const leaked = pool.acquire({ width: 640, height: 360, format: 'd32_float' });
    pool.invalidateAll();
    assert.equal(createdTextures[0].destroyed, false);

    const endRes = pool.endFrame();
    assert.equal(endRes.leakedCount, 1);
    assert.equal(endRes.autoReleasedCount, 1);
    assert.equal(endRes.evictedCount, 1);
    assert.equal(endRes.idleCount, 0);
    assert.equal(createdTextures[0].destroyed, true);
    assert.equal(createdTextures[0].destroyCount, 1);
    assert.equal(leaked.view, null);

    pool.destroy();
    assert.equal(createdTextures[0].destroyCount, 1);
  }
});

test('G45 preserves G31 classic-worker importScripts IIFE isolation without top-level lexical collisions', () => {
  const rtPoolPath = fileURLToPath(new URL('../web/workers/rt_pool.js', import.meta.url));
  const source = readFileSync(rtPoolPath, 'utf8');

  const sandbox = { console };
  const ctx = vm.createContext(sandbox);
  ctx.self = ctx;

  // Loading rt_pool.js twice in the same WorkerGlobalScope must throw zero SyntaxErrors.
  assert.doesNotThrow(() => {
    vm.runInContext(source, ctx, { filename: rtPoolPath });
    vm.runInContext(source, ctx, { filename: rtPoolPath });
  });

  // Top-level lexical names must not leak into WorkerGlobalScope.
  assert.doesNotThrow(() => {
    vm.runInContext(
      `
      const DEFAULT_MAX_TEXTURE_DIMENSION_2D = 1;
      const DXGI_FORMAT_ALIASES = {};
      const FORMAT_BYTES_PER_PIXEL = {};
      class RenderTargetPool {}
      `,
      ctx
    );
  });

  assert.ok(ctx.PlayGta6RenderTargetPool);
  assert.equal(ctx.PlayGta6RenderTargetPool.DEFAULT_MAX_TEXTURE_DIMENSION_2D, 8192);
  assert.equal(
    ctx.PlayGta6RenderTargetPool.normalizeRenderTargetFormat('d24_unorm_s8_uint'),
    'depth24plus-stencil8'
  );
});

// ---------------------------------------------------------------------------
// W128: Explicit evictIdle(maxIdleFrames) & Peak VRAM Reuse Telemetry
// ---------------------------------------------------------------------------

test('W128 evictIdle(maxIdleFrames) destroys aged idle targets, updates bytes/stats, and never evicts active targets', () => {
  const { device, createdTextures } = createMockGpuDevice();
  const pool = new RenderTargetPool({
    device,
    maxPoolBytes: Infinity,
    idleFrameTtl: 100,
    autoReleaseLeaked: false,
    onLeak: () => {},
  });

  // Fresh pool exposes additive telemetry fields initialized to 0.
  let stats = pool.getStats();
  assert.equal(stats.peakPooledBytes, 0);
  assert.equal(stats.reuseRatio, 0);
  assert.equal(stats.budgetEvictions, 0);
  assert.equal(stats.idleEvictions, 0);

  // Frame 0: acquire 4K color (released at frame 0) and 1080p depth (kept ACTIVE across frames).
  const oldDesc = { width: 3840, height: 2160, format: 'rgba8unorm' };
  const activeDesc = { width: 1920, height: 1080, format: 'depth32float' };
  const midDesc = { width: 1280, height: 720, format: 'rgba16float' };
  const freshDesc = { width: 640, height: 360, format: 'rgba8unorm' };

  const rtOld = pool.acquire(oldDesc, '4k-old');
  const rtActive = pool.acquire(activeDesc, '1080p-inflight');
  pool.release(rtOld); // lastUsedFrame = 0

  // Advance to frame 2, acquire & release rtMid (lastUsedFrame = 2).
  pool.endFrame(); // frame 1
  pool.endFrame(); // frame 2
  const rtMid = pool.acquire(midDesc, '720p-mid');
  pool.release(rtMid);

  // Advance to frame 4, acquire & release rtFresh (lastUsedFrame = 4).
  pool.endFrame(); // frame 3
  pool.endFrame(); // frame 4
  const rtFresh = pool.acquire(freshDesc, '360p-fresh');
  pool.release(rtFresh);

  const peakAllFour = rtOld.byteSize + rtActive.byteSize + rtMid.byteSize + rtFresh.byteSize;
  stats = pool.getStats();
  assert.equal(stats.currentFrame, 4);
  assert.equal(stats.activeCount, 1);
  assert.equal(stats.idleCount, 3);
  assert.equal(stats.totalBytes, peakAllFour);
  assert.equal(stats.peakPooledBytes, peakAllFour);
  assert.equal(stats.peakBytes, peakAllFour);

  // Threshold 5: max idle age is 4 - 0 = 4 < 5 -> evicts nothing.
  let evictRes = pool.evictIdle(5);
  assert.deepEqual(evictRes, { evictedCount: 0, evictedBytes: 0 });
  assert.equal(pool.getStats().idleCount, 3);
  assert.equal(pool.getStats().idleEvictions, 0);

  // Threshold 4: rtOld has age 4 - 0 = 4 >= 4 -> evicted; rtActive (age 4, ACTIVE) must survive!
  evictRes = pool.evictIdle(4);
  assert.deepEqual(evictRes, { evictedCount: 1, evictedBytes: rtOld.byteSize });
  assert.equal(createdTextures[0].destroyed, true);
  assert.equal(createdTextures[0].destroyCount, 1);
  assert.equal(rtOld.destroyed, true);
  assert.equal(rtOld.view, null);

  assert.equal(createdTextures[1].destroyed, false, 'active target must never be evicted by evictIdle()');
  assert.equal(rtActive.destroyed, false);
  assert.ok(rtActive.view !== null);
  assert.equal(createdTextures[2].destroyed, false);
  assert.equal(createdTextures[3].destroyed, false);

  stats = pool.getStats();
  assert.equal(stats.activeCount, 1);
  assert.equal(stats.activeBytes, rtActive.byteSize);
  assert.equal(stats.idleCount, 2);
  assert.equal(stats.idleBytes, rtMid.byteSize + rtFresh.byteSize);
  assert.equal(stats.totalBytes, rtActive.byteSize + rtMid.byteSize + rtFresh.byteSize);
  assert.equal(stats.peakPooledBytes, peakAllFour, 'peakPooledBytes must retain high-water mark after eviction');
  assert.equal(stats.idleEvictions, 1);
  assert.equal(stats.budgetEvictions, 0);
  assert.equal(stats.evictions, 1);

  // Repeating evictIdle(4) is a safe idempotent no-op with zero double-destroy.
  assert.deepEqual(pool.evictIdle(4), { evictedCount: 0, evictedBytes: 0 });
  assert.equal(createdTextures[0].destroyCount, 1);

  // Threshold 2: rtMid has age 4 - 2 = 2 >= 2 -> evicted; rtFresh (age 0) and rtActive stay intact.
  evictRes = pool.evictIdle(2);
  assert.deepEqual(evictRes, { evictedCount: 1, evictedBytes: rtMid.byteSize });
  assert.equal(createdTextures[2].destroyed, true);
  assert.equal(createdTextures[2].destroyCount, 1);
  assert.equal(createdTextures[3].destroyed, false);
  assert.equal(createdTextures[1].destroyed, false);

  // Threshold 0: evicts all remaining idle targets (rtFresh), still leaves rtActive untouched!
  evictRes = pool.evictIdle(0);
  assert.deepEqual(evictRes, { evictedCount: 1, evictedBytes: rtFresh.byteSize });
  assert.equal(createdTextures[3].destroyed, true);
  assert.equal(createdTextures[3].destroyCount, 1);
  assert.equal(createdTextures[1].destroyed, false, 'active target survives even evictIdle(0)');
  assert.equal(rtActive.destroyed, false);

  stats = pool.getStats();
  assert.equal(stats.activeCount, 1);
  assert.equal(stats.idleCount, 0);
  assert.equal(stats.idleBytes, 0);
  assert.equal(stats.totalBytes, rtActive.byteSize);
  assert.equal(stats.idleEvictions, 3);
  assert.equal(stats.budgetEvictions, 0);
  assert.equal(stats.evictions, 3);

  // Re-acquiring evicted oldDesc allocates a fresh texture rather than returning a destroyed handle.
  const rtOldFresh = pool.acquire(oldDesc, '4k-reallocated');
  assert.equal(rtOldFresh.reused, false);
  assert.notEqual(rtOldFresh.texture, rtOld.texture);
  assert.ok(rtOldFresh.view !== null);
  pool.release(rtOldFresh);

  // Omitting maxIdleFrames defaults to pool.idleFrameTtl (100 -> nothing evicted at age 0).
  assert.deepEqual(pool.evictIdle(), { evictedCount: 0, evictedBytes: 0 });

  // Invalid maxIdleFrames arguments throw RangeError.
  assert.throws(() => pool.evictIdle(-1), RangeError);
  assert.throws(() => pool.evictIdle(1.5), RangeError);
  assert.throws(() => pool.evictIdle(Number.NaN), RangeError);
  assert.throws(() => pool.evictIdle('2'), RangeError);
  assert.throws(() => pool.evictIdle(null), RangeError);

  pool.release(rtActive);
  pool.destroy();

  // Calling evictIdle() after destroy throws cleanly.
  assert.throws(() => pool.evictIdle(0), /destroyed/i);
});

test('W128 telemetry tracks peakPooledBytes, reuseRatio, budgetEvictions, and idleEvictions accurately', () => {
  const { device, createdTextures } = createMockGpuDevice();
  const oneTargetBytes = 512 * 512 * 4; // 1 MiB
  const pool = new RenderTargetPool({
    device,
    maxPoolBytes: oneTargetBytes * 2, // 2 MiB budget (fits 2 targets)
    idleFrameTtl: 2,
    strictBudget: true,
  });

  const descA = { width: 512, height: 512, format: 'rgba8unorm' };
  const descB = { width: 512, height: 512, format: 'bgra8unorm' };
  const descC = { width: 512, height: 512, format: 'rg11b10ufloat' };
  const descD = { width: 512, height: 512, format: 'rgb10a2unorm' };

  // 1. Two initial misses -> reuseRatio = 0, peakPooledBytes = 2 MiB.
  const a1 = pool.acquire(descA, 'rt-A');
  const b1 = pool.acquire(descB, 'rt-B');
  let stats = pool.getStats();
  assert.equal(stats.hits, 0);
  assert.equal(stats.misses, 2);
  assert.equal(stats.reuseRatio, 0);
  assert.equal(stats.peakPooledBytes, oneTargetBytes * 2);
  assert.equal(stats.budgetEvictions, 0);
  assert.equal(stats.idleEvictions, 0);

  // Failed strict-budget acquire must NOT inflate misses, budgetEvictions, or skew reuseRatio.
  assert.throws(() => pool.acquire(descC), RangeError);
  stats = pool.getStats();
  assert.equal(stats.misses, 2);
  assert.equal(stats.reuseRatio, 0);
  assert.equal(stats.budgetEvictions, 0);

  // 2. Release & reuse 3 times -> 3 hits / 5 total acquires = 0.6 reuseRatio.
  pool.release(a1);
  pool.release(b1);
  for (let i = 0; i < 3; i += 1) {
    const reusedA = pool.acquire(descA);
    assert.equal(reusedA.reused, true);
    pool.release(reusedA);
  }
  stats = pool.getStats();
  assert.equal(stats.hits, 3);
  assert.equal(stats.misses, 2);
  assert.ok(Math.abs(stats.reuseRatio - 3 / 5) < 1e-9);
  assert.equal(stats.reuseRatio, stats.hitRate);
  assert.ok(stats.reuseRatio >= 0 && stats.reuseRatio <= 1);

  // 3. Budget LRU eviction: acquiring descC evicts oldest idle target (descB) due to maxPoolBytes.
  const c1 = pool.acquire(descC, 'rt-C');
  assert.equal(c1.reused, false);
  assert.equal(createdTextures[1].destroyed, true);
  assert.equal(createdTextures[1].destroyCount, 1);

  stats = pool.getStats();
  assert.equal(stats.budgetEvictions, 1);
  assert.equal(stats.idleEvictions, 0);
  assert.equal(stats.evictions, 1);
  assert.equal(stats.hits, 3);
  assert.equal(stats.misses, 3);
  assert.ok(Math.abs(stats.reuseRatio - 3 / 6) < 1e-9);
  assert.equal(stats.peakPooledBytes, oneTargetBytes * 2);

  // Release c1 -> both A and C are idle at lastUsedFrame = 0.
  pool.release(c1);

  // 4. Advance frames so auto-idle TTL (> 2) evicts A while C is kept warm via reuse at frame 1.
  pool.endFrame(); // frame 1
  const c2 = pool.acquire(descC);
  assert.equal(c2.reused, true);
  pool.release(c2); // C lastUsedFrame = 1, A lastUsedFrame = 0

  pool.endFrame(); // frame 2: A age = 2 <= 2, C age = 1 <= 2
  assert.equal(pool.getStats().idleEvictions, 0);

  const endRes = pool.endFrame(); // frame 3: A age = 3 > 2 -> auto-idle evicted!
  assert.equal(endRes.evictedCount, 1);
  assert.equal(createdTextures[0].destroyed, true);
  assert.equal(createdTextures[0].destroyCount, 1);

  stats = pool.getStats();
  assert.equal(stats.budgetEvictions, 1);
  assert.equal(stats.idleEvictions, 1);
  assert.equal(stats.evictions, 2);

  // 5. Explicit evictIdle(2) at frame 3 evicts C (3 - 1 = 2 >= 2) and increments idleEvictions.
  const explicitRes = pool.evictIdle(2);
  assert.deepEqual(explicitRes, { evictedCount: 1, evictedBytes: oneTargetBytes });
  assert.equal(createdTextures[2].destroyed, true);
  assert.equal(createdTextures[2].destroyCount, 1);

  stats = pool.getStats();
  assert.equal(stats.budgetEvictions, 1);
  assert.equal(stats.idleEvictions, 2);
  assert.equal(stats.evictions, 3);
  assert.equal(stats.totalBytes, 0);
  assert.equal(stats.peakPooledBytes, oneTargetBytes * 2);

  // 6. invalidateAll() increments evictions without inflating budgetEvictions or idleEvictions.
  const d1 = pool.acquire(descD, 'rt-D');
  pool.release(d1);
  assert.equal(pool.invalidateAll(), 1);
  stats = pool.getStats();
  assert.equal(stats.budgetEvictions, 1);
  assert.equal(stats.idleEvictions, 2);
  assert.equal(stats.evictions, 4);

  // 7. Post-destroy getStats() preserves peakPooledBytes, reuseRatio, budgetEvictions, idleEvictions.
  const expectedReuseRatio = stats.reuseRatio;
  pool.destroy();
  const postDestroy = pool.getStats();
  assert.equal(postDestroy.peakPooledBytes, oneTargetBytes * 2);
  assert.equal(postDestroy.reuseRatio, expectedReuseRatio);
  assert.equal(postDestroy.budgetEvictions, 1);
  assert.equal(postDestroy.idleEvictions, 2);
  assert.equal(postDestroy.totalBytes, 0);
});

// ---------------------------------------------------------------------------
// W175: 4K Scene Plane 1 + 1080p UI Plane 0 Multi-Plane Reuse, viewFormats &
//       Memory-Budget Eviction Guards + Additive Telemetry
// ---------------------------------------------------------------------------

test('W175 distinguishes compatible viewFormats in pool keys, normalizes aliases, and sanitizes incompatible viewFormats', () => {
  assert.ok(Object.isFrozen(COMPATIBLE_VIEW_FORMATS));
  assert.equal(typeof isCompatibleViewFormat, 'function');
  assert.equal(typeof normalizeViewFormats, 'function');

  assert.equal(isCompatibleViewFormat('rgba8unorm', 'rgba8unorm-srgb'), true);
  assert.equal(isCompatibleViewFormat('rgba8unorm-srgb', 'rgba8unorm'), true);
  assert.equal(isCompatibleViewFormat('bgra8unorm', 'bgra8unorm-srgb'), true);
  assert.equal(isCompatibleViewFormat('rgba8unorm', 'DXGI_FORMAT_R8G8B8A8_UNORM_SRGB'), true);
  assert.equal(isCompatibleViewFormat('rgba8unorm', 'rgba8unorm'), true);
  assert.equal(isCompatibleViewFormat('rgba16float', 'rgba8unorm-srgb'), false);
  assert.equal(isCompatibleViewFormat('depth32float-stencil8', 'rgba8unorm'), false);

  const usage = TEXTURE_USAGE.RENDER_ATTACHMENT | TEXTURE_USAGE.TEXTURE_BINDING;
  const base4k = { width: 3840, height: 2160, format: 'rgba8unorm', sampleCount: 1, usage };

  // Keys must distinguish plain rgba8unorm vs rgba8unorm with sRGB viewFormat.
  const keyNoVf = makeRenderTargetKey(base4k);
  const keyWithSrgbVf = makeRenderTargetKey({ ...base4k, viewFormats: ['rgba8unorm-srgb'] });
  assert.notEqual(keyNoVf, keyWithSrgbVf);

  // Self-format, duplicates, DXGI aliases, and ordering canonicalize to identical keys.
  assert.equal(
    makeRenderTargetKey({ ...base4k, viewFormats: ['rgba8unorm'] }),
    keyNoVf,
    'self-format in viewFormats is redundant and matches default key'
  );
  assert.equal(
    makeRenderTargetKey({
      ...base4k,
      viewFormats: ['DXGI_FORMAT_R8G8B8A8_UNORM_SRGB', 'rgba8unorm', 'rgba8unorm-srgb'],
    }),
    keyWithSrgbVf
  );
  assert.equal(
    makeRenderTargetKey({ ...base4k, viewFormats: new Set(['rgba8unorm-srgb']) }),
    keyWithSrgbVf
  );

  // Strict normalizeRenderTargetDesc rejects malformed or incompatible viewFormats.
  assert.throws(
    () => normalizeRenderTargetDesc({ ...base4k, viewFormats: 'rgba8unorm-srgb' }),
    TypeError
  );
  assert.throws(
    () => normalizeRenderTargetDesc({ ...base4k, viewFormats: 42 }),
    TypeError
  );
  assert.throws(
    () => normalizeRenderTargetDesc({ ...base4k, viewFormats: [''] }),
    TypeError
  );
  assert.throws(
    () =>
      normalizeRenderTargetDesc({
        width: 3840,
        height: 2160,
        format: 'rgba16float',
        viewFormats: ['rgba8unorm-srgb'],
      }),
    RangeError
  );

  // Pool acquisition keeps viewFormats-enabled targets isolated from plain targets and passes
  // viewFormats to device.createTexture() only when non-empty.
  const { device, createdTextures } = createMockGpuDevice();
  const pool = new RenderTargetPool({ device });

  const plain4k = pool.acquire(base4k, 'plane1-plain');
  assert.deepEqual(plain4k.desc.viewFormats, []);
  assert.equal('viewFormats' in createdTextures[0].descriptor, false);
  pool.release(plain4k);

  // Requesting the same 4K rgba8unorm WITH viewFormats: ['rgba8unorm-srgb'] must MISS plain4k.
  const srgbView4k = pool.acquire(
    { ...base4k, viewFormats: ['rgba8unorm-srgb'] },
    'plane1-srgb-view'
  );
  assert.equal(srgbView4k.reused, false);
  assert.notEqual(srgbView4k.texture, plain4k.texture);
  assert.deepEqual(srgbView4k.desc.viewFormats, ['rgba8unorm-srgb']);
  assert.deepEqual(createdTextures[1].descriptor.viewFormats, ['rgba8unorm-srgb']);
  pool.release(srgbView4k);

  // Re-acquiring via DXGI alias + redundant self-format hits srgbView4k cleanly!
  const aliasHit = pool.acquire(
    { ...base4k, format: 'r8g8b8a8_unorm', viewFormats: ['r8g8b8a8_unorm_srgb', 'rgba8unorm'] },
    'plane1-srgb-alias-hit'
  );
  assert.equal(aliasHit.reused, true);
  assert.equal(aliasHit.texture, srgbView4k.texture);
  pool.release(aliasHit);

  // Pool acquire() sanitizes incompatible viewFormats on rgba16float & depth32float-stencil8
  // instead of forwarding invalid viewFormats to device.createTexture().
  const hdr4k = pool.acquire(
    { width: 3840, height: 2160, format: 'rgba16float', viewFormats: ['rgba8unorm-srgb'] },
    'plane1-hdr-sanitized'
  );
  assert.deepEqual(hdr4k.desc.viewFormats, []);
  assert.equal('viewFormats' in createdTextures[2].descriptor, false);

  const depth4k = pool.acquire(
    {
      width: 3840,
      height: 2160,
      format: 'depth32float-stencil8',
      viewFormats: ['rgba8unorm'],
    },
    'plane1-depth-sanitized'
  );
  assert.deepEqual(depth4k.desc.viewFormats, []);
  assert.equal('viewFormats' in createdTextures[3].descriptor, false);

  pool.release(hdr4k);
  pool.release(depth4k);
  pool.destroy();
});

test('W175 simultaneous 4K scene plane 1 and 1080p UI plane 0 coexist without thrashing under default budget', () => {
  const { device, createdTextures } = createMockGpuDevice();
  // Uses default maxPoolBytes (512 MiB) — must fit full 4K scene plane 1 + 1080p UI plane 0 set.
  const pool = new RenderTargetPool({ device });
  assert.equal(pool.maxPoolBytes, 512 * 1024 * 1024);

  const colorUsage =
    TEXTURE_USAGE.RENDER_ATTACHMENT | TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_SRC;
  const depthUsage = TEXTURE_USAGE.RENDER_ATTACHMENT | TEXTURE_USAGE.TEXTURE_BINDING;

  // Plane 1 (4K 3840x2160 scene) + Plane 0 (1080p 1920x1080 UI) across all 4 required formats.
  const multiPlaneSpecs = [
    // 4K Scene Plane 1
    {
      label: 'plane1-4k-hdr-a',
      desc: { width: 3840, height: 2160, format: 'rgba16float', usage: colorUsage, sampleCount: 1 },
      expectedBytes: 3840 * 2160 * 8,
    },
    {
      label: 'plane1-4k-hdr-b',
      desc: { width: 3840, height: 2160, format: 'rgba16float', usage: colorUsage, sampleCount: 1 },
      expectedBytes: 3840 * 2160 * 8,
    },
    {
      label: 'plane1-4k-depth',
      desc: {
        width: 3840,
        height: 2160,
        format: 'depth32float-stencil8',
        usage: depthUsage,
        sampleCount: 1,
      },
      expectedBytes: 3840 * 2160 * 8,
    },
    {
      label: 'plane1-4k-gbuffer-linear',
      desc: {
        width: 3840,
        height: 2160,
        format: 'rgba8unorm',
        usage: colorUsage,
        sampleCount: 1,
        viewFormats: ['rgba8unorm-srgb'],
      },
      expectedBytes: 3840 * 2160 * 4,
    },
    {
      label: 'plane1-4k-tonemapped-srgb',
      desc: {
        width: 3840,
        height: 2160,
        format: 'rgba8unorm-srgb',
        usage: colorUsage,
        sampleCount: 1,
        viewFormats: ['rgba8unorm'],
      },
      expectedBytes: 3840 * 2160 * 4,
    },
    // 1080p UI Plane 0
    {
      label: 'plane0-1080p-ui-srgb',
      desc: {
        width: 1920,
        height: 1080,
        format: 'rgba8unorm-srgb',
        usage: colorUsage,
        sampleCount: 1,
      },
      expectedBytes: 1920 * 1080 * 4,
    },
    {
      label: 'plane0-1080p-ui-linear',
      desc: {
        width: 1920,
        height: 1080,
        format: 'rgba8unorm',
        usage: colorUsage,
        sampleCount: 1,
        viewFormats: ['rgba8unorm-srgb'],
      },
      expectedBytes: 1920 * 1080 * 4,
    },
    {
      label: 'plane0-1080p-ui-hdr',
      desc: { width: 1920, height: 1080, format: 'rgba16float', usage: colorUsage, sampleCount: 1 },
      expectedBytes: 1920 * 1080 * 8,
    },
    {
      label: 'plane0-1080p-ui-depth',
      desc: {
        width: 1920,
        height: 1080,
        format: 'depth32float-stencil8',
        usage: depthUsage,
        sampleCount: 1,
      },
      expectedBytes: 1920 * 1080 * 8,
    },
  ];

  const expectedWorkingSetBytes = multiPlaneSpecs.reduce((acc, s) => acc + s.expectedBytes, 0);
  assert.equal(expectedWorkingSetBytes, 315_187_200); // ~300.6 MiB (exceeds legacy 256 MiB, fits 512 MiB)

  const frames = 6;
  const firstFrameTexIds = [];

  for (let frame = 0; frame < frames; frame += 1) {
    const acquired = multiPlaneSpecs.map((spec) => {
      const rt = pool.acquire(spec.desc, spec.label);
      assert.equal(rt.byteSize, spec.expectedBytes, `byteSize mismatch for ${spec.label}`);
      assert.equal(rt.reused, frame > 0, `frame ${frame} ${spec.label} reuse mismatch`);
      return rt;
    });

    // All 9 targets are simultaneously checked out (active).
    const midStats = pool.getStats();
    assert.equal(midStats.activeTargets, multiPlaneSpecs.length);
    assert.equal(midStats.activeCount, multiPlaneSpecs.length);
    assert.equal(midStats.idleTargets, 0);
    assert.equal(midStats.idleCount, 0);
    assert.equal(midStats.estimatedBytes, expectedWorkingSetBytes);
    assert.equal(midStats.peakEstimatedBytes, expectedWorkingSetBytes);

    // Every simultaneously checked-out target must have a distinct physical GPUTexture.
    const uniqueTex = new Set(acquired.map((rt) => rt.texture));
    assert.equal(uniqueTex.size, multiPlaneSpecs.length);

    if (frame === 0) {
      for (const rt of acquired) {
        firstFrameTexIds.push(rt.texture.mockId);
      }
    } else {
      const frameTexIds = new Set(acquired.map((rt) => rt.texture.mockId));
      for (const id of firstFrameTexIds) {
        assert.ok(frameTexIds.has(id), `expected texture #${id} to be reused on frame ${frame}`);
      }
    }

    for (const rt of acquired) {
      assert.equal(pool.release(rt), true);
    }

    const endRes = pool.endFrame();
    assert.equal(endRes.evictedCount, 0);
    assert.equal(endRes.leakedCount, 0);
  }

  // Only 9 physical GPUTextures ever allocated across 54 total acquires (0 thrashing, 0 evictions).
  assert.equal(createdTextures.length, multiPlaneSpecs.length);

  const stats = pool.getStats();
  assert.equal(stats.poolMisses, multiPlaneSpecs.length);
  assert.equal(stats.misses, multiPlaneSpecs.length);
  assert.equal(stats.poolHits, multiPlaneSpecs.length * (frames - 1));
  assert.equal(stats.hits, multiPlaneSpecs.length * (frames - 1));
  assert.equal(stats.evictions, 0);
  assert.equal(stats.budgetEvictions, 0);
  assert.equal(stats.activeTargets, 0);
  assert.equal(stats.idleTargets, multiPlaneSpecs.length);
  assert.equal(stats.estimatedBytes, expectedWorkingSetBytes);
  assert.equal(stats.peakEstimatedBytes, expectedWorkingSetBytes);
  assert.equal(stats.maxPoolBytes, 512 * 1024 * 1024);

  pool.destroy();
  const postDestroy = pool.getStats();
  assert.equal(postDestroy.activeTargets, 0);
  assert.equal(postDestroy.idleTargets, 0);
  assert.equal(postDestroy.estimatedBytes, 0);
  assert.equal(postDestroy.peakEstimatedBytes, expectedWorkingSetBytes);
  assert.equal(postDestroy.poolHits, multiPlaneSpecs.length * (frames - 1));
  assert.equal(postDestroy.poolMisses, multiPlaneSpecs.length);
});

test('W175 memory-budget LRU eviction guards evict only idle targets and never destroy active 4K/1080p targets', () => {
  const bytes4kRgba8 = 3840 * 2160 * 4; // 33,177,600 B
  const bytes4kHdr = 3840 * 2160 * 8; // 66,355,200 B
  const bytes1080Rgba8 = 1920 * 1080 * 4; // 8,294,400 B

  // 1. Tight budget under strictBudget: true evicts LRU idle targets while keeping active targets alive.
  {
    const { device, createdTextures } = createMockGpuDevice();
    // Budget fits 4K HDR (66.35 MB) + 4K rgba8 (33.18 MB) + 1080p rgba8 (8.29 MB) = 107,827,200 B
    const budget = bytes4kHdr + bytes4kRgba8 + bytes1080Rgba8;
    const pool = new RenderTargetPool({ device, maxPoolBytes: budget, strictBudget: true });

    const plane1Hdr = pool.acquire({ width: 3840, height: 2160, format: 'rgba16float' }, 'p1-hdr');
    const plane1Color = pool.acquire({ width: 3840, height: 2160, format: 'rgba8unorm' }, 'p1-color');
    const plane0Ui = pool.acquire({ width: 1920, height: 1080, format: 'rgba8unorm-srgb' }, 'p0-ui');

    // Release plane1Color first (oldest LRU), then plane0Ui (newer LRU). Keep plane1Hdr ACTIVE!
    pool.release(plane1Color);
    pool.release(plane0Ui);

    // Acquiring 4K rgba8unorm-srgb (33.18 MB) needs 33.18 MB -> must LRU-evict plane1Color (oldest idle),
    // keeping plane0Ui (newer idle) AND plane1Hdr (active) untouched!
    const plane1Srgb = pool.acquire(
      { width: 3840, height: 2160, format: 'rgba8unorm-srgb' },
      'p1-srgb'
    );
    assert.equal(plane1Srgb.reused, false);
    assert.equal(createdTextures[0].destroyed, false, 'active 4K HDR target must never be evicted');
    assert.equal(createdTextures[1].destroyed, true, 'oldest idle target must be LRU-evicted');
    assert.equal(createdTextures[2].destroyed, false, 'newer idle 1080p UI target must survive');
    assert.equal(plane1Hdr.destroyed, false);
    assert.ok(plane1Hdr.view !== null);

    let stats = pool.getStats();
    assert.equal(stats.activeTargets, 2);
    assert.equal(stats.idleTargets, 1);
    assert.equal(stats.evictions, 1);
    assert.equal(stats.budgetEvictions, 1);
    assert.equal(stats.estimatedBytes, budget);
    assert.equal(stats.peakEstimatedBytes, budget);

    // Runtime setMaxPoolBytes below active bytes evicts idle targets ONLY, never destroys active targets!
    const shrinkRes = pool.setMaxPoolBytes(bytes4kRgba8);
    assert.deepEqual(shrinkRes, { evictedCount: 1, evictedBytes: bytes1080Rgba8 });
    assert.equal(createdTextures[2].destroyed, true, 'idle 1080p UI evicted when budget shrinks');
    assert.equal(createdTextures[0].destroyed, false, 'checked-out 4K HDR stays alive');
    assert.equal(createdTextures[3].destroyed, false, 'checked-out 4K sRGB stays alive');

    stats = pool.getStats();
    assert.equal(stats.maxPoolBytes, bytes4kRgba8);
    assert.equal(stats.activeTargets, 2);
    assert.equal(stats.idleTargets, 0);
    assert.equal(stats.evictions, 2);
    assert.equal(stats.budgetEvictions, 2);
    assert.equal(stats.estimatedBytes, bytes4kHdr + bytes4kRgba8);

    // Invalid setMaxPoolBytes values throw RangeError.
    assert.throws(() => pool.setMaxPoolBytes(0), RangeError);
    assert.throws(() => pool.setMaxPoolBytes(-1024), RangeError);
    assert.throws(() => pool.setMaxPoolBytes(1.5), RangeError);
    assert.throws(() => pool.setMaxPoolBytes(Number.NaN), RangeError);

    pool.release(plane1Hdr);
    pool.release(plane1Srgb);
    pool.destroy();
    assert.throws(() => pool.setMaxPoolBytes(bytes4kRgba8), /destroyed/i);
  }

  // 2. strictBudget: false permits burst active allocation above maxPoolBytes, then trims excess idle
  //    targets in LRU order at endFrame() while still leaving any checked-out targets untouched.
  {
    const { device, createdTextures } = createMockGpuDevice();
    const pool = new RenderTargetPool({
      device,
      maxPoolBytes: bytes4kRgba8, // 33.18 MB budget
      strictBudget: false,
      autoReleaseLeaked: false,
      onLeak: () => {},
    });

    // Allocate 3 simultaneous targets exceeding budget: 4K HDR (active across frame), 4K rgba8, 1080p UI.
    const activeHdr = pool.acquire({ width: 3840, height: 2160, format: 'rgba16float' }, 'burst-hdr');
    const released4k = pool.acquire({ width: 3840, height: 2160, format: 'rgba8unorm' }, 'burst-4k');
    const released1080 = pool.acquire(
      { width: 1920, height: 1080, format: 'rgba8unorm-srgb' },
      'burst-1080p'
    );

    const peakExpected = bytes4kHdr + bytes4kRgba8 + bytes1080Rgba8;
    let stats = pool.getStats();
    assert.equal(stats.activeTargets, 3);
    assert.equal(stats.estimatedBytes, peakExpected);
    assert.equal(stats.peakEstimatedBytes, peakExpected);

    // Release the two non-active targets; keep activeHdr checked out.
    pool.release(released4k);
    pool.release(released1080);

    // endFrame() must LRU-trim both excess idle targets because activeBytes alone exceeds maxPoolBytes,
    // while NEVER destroying activeHdr!
    const endRes = pool.endFrame();
    assert.equal(endRes.evictedCount, 2);
    assert.equal(createdTextures[0].destroyed, false, 'active 4K HDR must never be destroyed by endFrame budget trim');
    assert.equal(createdTextures[1].destroyed, true);
    assert.equal(createdTextures[2].destroyed, true);
    assert.equal(activeHdr.destroyed, false);
    assert.ok(activeHdr.view !== null);

    stats = pool.getStats();
    assert.equal(stats.activeTargets, 1);
    assert.equal(stats.idleTargets, 0);
    assert.equal(stats.evictions, 2);
    assert.equal(stats.budgetEvictions, 2);
    assert.equal(stats.estimatedBytes, bytes4kHdr);
    assert.equal(stats.peakEstimatedBytes, peakExpected);

    pool.release(activeHdr);
    pool.destroy();
  }
});

// ---------------------------------------------------------------------------
// W200: GPUTextureView Caching, 4K Transient Target Reuse, Oversized Screen-
//       Transition Eviction & Lightweight Telemetry
// ---------------------------------------------------------------------------

test('W200 caches default and parameterized GPUTextureViews for 4K and 1080p reuse with 0 createTexture/createView calls on hits', () => {
  assert.equal(typeof makeTextureViewKey, 'function');
  assert.equal(DEFAULT_OVERSIZED_TARGET_BYTES, 24 * 1024 * 1024);

  // Default / redundant descriptors collapse to default view key.
  assert.equal(makeTextureViewKey(undefined, 'rgba8unorm').isDefault, true);
  assert.equal(makeTextureViewKey(null, 'rgba8unorm').isDefault, true);
  assert.equal(makeTextureViewKey({}, 'rgba8unorm').isDefault, true);
  assert.equal(makeTextureViewKey({ format: 'rgba8unorm' }, 'rgba8unorm').isDefault, true);
  assert.equal(
    makeTextureViewKey({ format: 'DXGI_FORMAT_R8G8B8A8_UNORM', aspect: 'all', baseMipLevel: 0 }, 'rgba8unorm')
      .isDefault,
    true
  );

  // Non-default view descriptors produce canonical deterministic keys independent of label.
  const srgbKey1 = makeTextureViewKey({ format: 'rgba8unorm-srgb', label: 'pass-a' }, 'rgba8unorm');
  const srgbKey2 = makeTextureViewKey(
    { format: 'DXGI_FORMAT_R8G8B8A8_UNORM_SRGB', label: 'pass-b' },
    'rgba8unorm'
  );
  assert.equal(srgbKey1.isDefault, false);
  assert.equal(srgbKey1.key, srgbKey2.key);
  assert.equal(srgbKey1.descriptor.format, 'rgba8unorm-srgb');

  // Validation errors for malformed view descriptors.
  assert.throws(() => makeTextureViewKey('bad', 'rgba8unorm'), TypeError);
  assert.throws(() => makeTextureViewKey([], 'rgba8unorm'), TypeError);
  assert.throws(() => makeTextureViewKey({ dimension: '4d' }, 'rgba8unorm'), RangeError);
  assert.throws(() => makeTextureViewKey({ aspect: 'color-only' }, 'rgba8unorm'), RangeError);
  assert.throws(() => makeTextureViewKey({ baseMipLevel: -1 }, 'rgba8unorm'), RangeError);
  assert.throws(() => makeTextureViewKey({ mipLevelCount: 0 }, 'rgba8unorm'), RangeError);

  const { device, createdTextures } = createMockGpuDevice();
  const pool = new RenderTargetPool({ device });

  // Fresh pool telemetry starts at 0.
  let stats = pool.getStats();
  assert.equal(stats.poolHits, 0);
  assert.equal(stats.poolMisses, 0);
  assert.equal(stats.viewsCached, 0);
  assert.equal(stats.viewCacheHits, 0);
  assert.equal(stats.viewCacheMisses, 0);
  assert.equal(stats.evictions, 0);
  assert.equal(stats.activeTargets, 0);
  assert.equal(stats.idleTargets, 0);
  assert.equal(stats.estimatedVramBytes, 0);

  const desc4kColor = {
    width: 3840,
    height: 2160,
    format: 'rgba8unorm',
    viewFormats: ['rgba8unorm-srgb'],
  };
  const desc4kDepth = {
    width: 3840,
    height: 2160,
    format: 'depth32float-stencil8',
  };
  const desc1080Ui = {
    width: 1920,
    height: 1080,
    format: 'rgba8unorm-srgb',
  };

  // Frame 0: initial misses create 3 textures and 3 default cached views.
  const rt4k = pool.acquire(desc4kColor, '4k-scene-color');
  const rt4kDepth = pool.acquire(desc4kDepth, '4k-scene-depth');
  const rt1080 = pool.acquire(desc1080Ui, '1080p-ui');

  assert.equal(createdTextures.length, 3);
  assert.equal(rt4k.texture.viewCreateCount, 1);
  assert.equal(rt4kDepth.texture.viewCreateCount, 1);
  assert.equal(rt1080.texture.viewCreateCount, 1);

  // Requesting default views via handle.getView / pool.getView performs 0 createView calls.
  assert.equal(rt4k.getView(), rt4k.view);
  assert.equal(rt4k.getView({}), rt4k.view);
  assert.equal(rt4k.getView({ format: 'rgba8unorm' }), rt4k.view);
  assert.equal(pool.getView(rt4k), rt4k.view);
  assert.equal(pool.getView(rt4k.id), rt4k.view);
  assert.equal(rt4k.texture.viewCreateCount, 1);

  // Requesting alternate sRGB view on 4K color and depth-only / stencil-only views on 4K depth
  // creates each view once and caches them by normalized descriptor.
  const srgbView1 = rt4k.getView({ format: 'rgba8unorm-srgb' });
  const srgbView2 = pool.getView(rt4k, { format: 'DXGI_FORMAT_R8G8B8A8_UNORM_SRGB' });
  assert.ok(srgbView1 !== null);
  assert.notEqual(srgbView1, rt4k.view);
  assert.equal(srgbView2, srgbView1);
  assert.equal(rt4k.texture.viewCreateCount, 2);

  const depthOnlyView1 = rt4kDepth.getView({ aspect: 'depth-only' });
  const stencilOnlyView1 = rt4kDepth.getView({ aspect: 'stencil-only' });
  assert.ok(depthOnlyView1 !== null);
  assert.ok(stencilOnlyView1 !== null);
  assert.notEqual(depthOnlyView1, stencilOnlyView1);
  assert.equal(rt4kDepth.getView({ aspect: 'depth-only' }), depthOnlyView1);
  assert.equal(rt4kDepth.texture.viewCreateCount, 3);

  const expectedVramBytes =
    3840 * 2160 * 4 + 3840 * 2160 * 8 + 1920 * 1080 * 4;
  stats = pool.getStats();
  assert.equal(stats.poolMisses, 3);
  assert.equal(stats.poolHits, 0);
  // 3 default views + 1 sRGB view + 1 depth-only + 1 stencil-only = 6 cached views
  assert.equal(stats.viewsCached, 6);
  assert.equal(stats.activeTargets, 3);
  assert.equal(stats.idleTargets, 0);
  assert.equal(stats.estimatedVramBytes, expectedVramBytes);

  pool.release(rt4k);
  pool.release(rt4kDepth);
  pool.release(rt1080);
  pool.endFrame();

  // Frames 1..5: acquiring/reusing 4K and 1080p targets and their cached views performs
  // ZERO createTexture and ZERO createView calls!
  const texCountBeforeHits = createdTextures.length;
  const viewCountsBeforeHits = createdTextures.map((t) => t.viewCreateCount);

  for (let frame = 1; frame <= 5; frame += 1) {
    const c = pool.acquire(desc4kColor, `4k-color-f${frame}`);
    const d = pool.acquire(desc4kDepth, `4k-depth-f${frame}`);
    const u = pool.acquire(desc1080Ui, `1080p-ui-f${frame}`);

    assert.equal(c.reused, true);
    assert.equal(d.reused, true);
    assert.equal(u.reused, true);
    assert.equal(c.texture, rt4k.texture);
    assert.equal(d.texture, rt4kDepth.texture);
    assert.equal(u.texture, rt1080.texture);
    assert.equal(c.view, rt4k.view);
    assert.equal(d.view, rt4kDepth.view);
    assert.equal(u.view, rt1080.view);
    assert.equal(c.getView({ format: 'rgba8unorm-srgb' }), srgbView1);
    assert.equal(d.getView({ aspect: 'depth-only' }), depthOnlyView1);
    assert.equal(d.getView({ aspect: 'stencil-only' }), stencilOnlyView1);

    // Stale handle rt4k must not steal or create views via pool.getView while c owns the lease.
    assert.equal(pool.getView(rt4k, { format: 'rgba8unorm-srgb' }), null);

    pool.release(c);
    pool.release(d);
    pool.release(u);
    pool.endFrame();
  }

  assert.equal(createdTextures.length, texCountBeforeHits, '0 createTexture calls on pool hits');
  assert.deepEqual(
    createdTextures.map((t) => t.viewCreateCount),
    viewCountsBeforeHits,
    '0 createView calls on pool hits'
  );

  stats = pool.getStats();
  assert.equal(stats.poolHits, 15);
  assert.equal(stats.poolMisses, 3);
  assert.equal(stats.viewsCached, 6);
  assert.equal(stats.activeTargets, 0);
  assert.equal(stats.idleTargets, 3);
  assert.equal(stats.estimatedVramBytes, expectedVramBytes);
  assert.equal(stats.evictions, 0);

  // Evicting via invalidateAll() invalidates all cached views cleanly.
  assert.equal(pool.invalidateAll(), 3);
  assert.equal(rt4k.view, null);
  assert.equal(rt4k.getView(), null);
  assert.equal(rt4k.getView({ format: 'rgba8unorm-srgb' }), null);
  assert.equal(pool.getView(rt4k), null);
  assert.equal(rt4kDepth.view, null);
  assert.equal(rt4kDepth.getView({ aspect: 'depth-only' }), null);
  assert.equal(rt1080.view, null);

  stats = pool.getStats();
  assert.equal(stats.viewsCached, 0);
  assert.equal(stats.evictions, 3);
  assert.equal(stats.idleTargets, 0);
  assert.equal(stats.estimatedVramBytes, 0);

  // Re-acquiring after eviction allocates fresh texture + fresh cached view.
  const rt4kFresh = pool.acquire(desc4kColor, '4k-after-evict');
  assert.equal(rt4kFresh.reused, false);
  assert.notEqual(rt4kFresh.texture, rt4k.texture);
  assert.notEqual(rt4kFresh.view, rt4k.view);
  assert.equal(pool.getStats().viewsCached, 1);

  pool.destroy();
  assert.equal(rt4kFresh.view, null);
  assert.equal(rt4kFresh.getView(), null);
  assert.equal(pool.getStats().viewsCached, 0);
  assert.equal(pool.getStats().estimatedVramBytes, 0);
  assert.throws(() => pool.getView(rt4kFresh), /destroyed/i);
});

test('W200 configurable memory budget and oversized idle-frame eviction prevent VRAM leaks across intro -> Start Screen -> menus', () => {
  const bytes4kHdr = 3840 * 2160 * 8; // 66,355,200 B (~63.3 MiB)
  const bytes4kRgba8 = 3840 * 2160 * 4; // 33,177,600 B (~31.6 MiB)
  const bytes1080Rgba8 = 1920 * 1080 * 4; // 8,294,400 B (~7.9 MiB)
  const bytes1080Depth = 1920 * 1080 * 4; // 8,294,400 B (~7.9 MiB)

  const { device, createdTextures } = createMockGpuDevice();
  // Configure via maxVramBytes / maxIdleFrames aliases + oversizedIdleFrameTtl for 4K targets.
  const pool = new RenderTargetPool({
    device,
    maxVramBytes: 256 * 1024 * 1024,
    maxIdleFrames: 8,
    oversizedTargetBytes: DEFAULT_OVERSIZED_TARGET_BYTES, // 24 MiB
    oversizedIdleFrameTtl: 2,
  });

  assert.equal(pool.maxPoolBytes, 256 * 1024 * 1024);
  assert.equal(pool.idleFrameTtl, 8);
  assert.equal(pool.oversizedTargetBytes, DEFAULT_OVERSIZED_TARGET_BYTES);
  assert.equal(pool.oversizedIdleFrameTtl, 2);

  const intro4kHdrDesc = { width: 3840, height: 2160, format: 'rgba16float' };
  const intro4kColorDesc = { width: 3840, height: 2160, format: 'rgba8unorm' };
  const start1080ColorDesc = { width: 1920, height: 1080, format: 'rgba8unorm-srgb' };
  const start1080DepthDesc = { width: 1920, height: 1080, format: 'depth32float' };

  // Phase 1 (Intro): 2 frames using 4K HDR + 4K color targets (reused with 0 churn).
  let introHdr = pool.acquire(intro4kHdrDesc, 'intro-4k-hdr');
  let introColor = pool.acquire(intro4kColorDesc, 'intro-4k-color');
  pool.release(introHdr);
  pool.release(introColor);
  pool.endFrame(); // frame 1

  introHdr = pool.acquire(intro4kHdrDesc, 'intro-4k-hdr-f1');
  introColor = pool.acquire(intro4kColorDesc, 'intro-4k-color-f1');
  assert.equal(introHdr.reused, true);
  assert.equal(introColor.reused, true);
  assert.equal(createdTextures.length, 2);
  pool.release(introHdr); // lastUsedFrame = 1
  pool.release(introColor); // lastUsedFrame = 1
  pool.endFrame(); // frame 2

  // Phase 2 (Start Screen): switches to 1080p UI + depth; 4K intro targets are now unused.
  for (let i = 0; i < 2; i += 1) {
    const uiColor = pool.acquire(start1080ColorDesc, 'start-1080p-color');
    const uiDepth = pool.acquire(start1080DepthDesc, 'start-1080p-depth');
    assert.equal(uiColor.reused, i > 0);
    assert.equal(uiDepth.reused, i > 0);
    pool.release(uiColor);
    pool.release(uiDepth);
    const res = pool.endFrame();
    if (i === 0) {
      // frame 3: 4K idle age = 3 - 1 = 2 <= oversizedIdleFrameTtl (2) -> still warm.
      assert.equal(res.evictedCount, 0);
    } else {
      // frame 4: 4K idle age = 4 - 1 = 3 > oversizedIdleFrameTtl (2) -> BOTH 4K targets evicted!
      assert.equal(res.evictedCount, 2);
    }
  }

  // Oversized 4K targets destroyed & views invalidated; 1080p Start Screen targets stay warm!
  assert.equal(createdTextures[0].destroyed, true);
  assert.equal(createdTextures[1].destroyed, true);
  assert.equal(introHdr.view, null);
  assert.equal(introColor.view, null);
  assert.equal(createdTextures[2].destroyed, false);
  assert.equal(createdTextures[3].destroyed, false);

  let stats = pool.getStats();
  assert.equal(stats.evictions, 2);
  assert.equal(stats.idleEvictions, 2);
  assert.equal(stats.idleTargets, 2);
  assert.equal(stats.activeTargets, 0);
  assert.equal(stats.viewsCached, 2);
  assert.equal(stats.estimatedVramBytes, bytes1080Rgba8 + bytes1080Depth);
  assert.equal(
    stats.peakVramBytes,
    bytes4kHdr + bytes4kRgba8 + bytes1080Rgba8 + bytes1080Depth
  );

  // Phase 3 (Menus): trimToBudget reclaims idle targets without mutating maxPoolBytes
  // and without touching active targets.
  const activeMenuRt = pool.acquire(start1080ColorDesc, 'menu-active-color');
  assert.equal(activeMenuRt.reused, true);
  // Active is 1080p color (8.29 MB), idle is 1080p depth (8.29 MB).
  // trimToBudget(bytes1080Rgba8) evicts the idle 1080p depth while keeping activeMenuRt alive.
  const trimRes = pool.trimToBudget(bytes1080Rgba8);
  assert.deepEqual(trimRes, { evictedCount: 1, evictedBytes: bytes1080Depth });
  assert.equal(createdTextures[3].destroyed, true);
  assert.equal(createdTextures[2].destroyed, false);
  assert.equal(pool.maxPoolBytes, 256 * 1024 * 1024, 'trimToBudget must not alter maxPoolBytes');

  stats = pool.getStats();
  assert.equal(stats.activeTargets, 1);
  assert.equal(stats.idleTargets, 0);
  assert.equal(stats.viewsCached, 1);
  assert.equal(stats.estimatedVramBytes, bytes1080Rgba8);
  assert.equal(stats.evictions, 3);
  assert.equal(stats.budgetEvictions, 1);

  // Release activeMenuRt and verify runtime setIdleFrameTtl + setOversizedEvictionPolicy + evictIdle minBytes.
  pool.release(activeMenuRt);
  const bigTemp = pool.acquire(intro4kColorDesc, 'menu-photo-mode-4k');
  pool.release(bigTemp); // lastUsedFrame = 4

  // evictIdle with minBytes only evicts idle targets >= 24 MiB (bigTemp), sparing 1080p activeMenuRt.
  const bigOnlyRes = pool.evictIdle(0, { minBytes: DEFAULT_OVERSIZED_TARGET_BYTES });
  assert.deepEqual(bigOnlyRes, { evictedCount: 1, evictedBytes: bytes4kRgba8 });
  assert.equal(bigTemp.destroyed, true);
  assert.equal(bigTemp.view, null);
  assert.equal(createdTextures[2].destroyed, false);
  assert.equal(pool.getStats().idleTargets, 1);
  assert.equal(pool.getStats().viewsCached, 1);

  // Advance 2 frames, then tighten idleFrameTtl via setIdleFrameTtl(1) -> immediately evicts aged 1080p target.
  pool.endFrame(); // frame 5
  pool.endFrame(); // frame 6 (age = 6 - 4 = 2 > 1)
  const ttlRes = pool.setIdleFrameTtl(1);
  assert.deepEqual(ttlRes, { evictedCount: 1, evictedBytes: bytes1080Rgba8 });
  assert.equal(pool.idleFrameTtl, 1);
  assert.equal(pool.getStats().idleTargets, 0);
  assert.equal(pool.getStats().viewsCached, 0);
  assert.equal(pool.getStats().estimatedVramBytes, 0);

  // Runtime setOversizedEvictionPolicy updates and validates cleanly.
  assert.deepEqual(
    pool.setOversizedEvictionPolicy({ oversizedTargetBytes: 16 * 1024 * 1024, oversizedIdleFrameTtl: 1 }),
    { evictedCount: 0, evictedBytes: 0 }
  );
  assert.equal(pool.oversizedTargetBytes, 16 * 1024 * 1024);
  assert.equal(pool.oversizedIdleFrameTtl, 1);

  assert.throws(() => pool.setIdleFrameTtl(-1), RangeError);
  assert.throws(() => pool.trimToBudget(-1), RangeError);
  assert.throws(() => pool.evictIdle(0, { minBytes: -1 }), RangeError);
  assert.throws(() => pool.setOversizedEvictionPolicy(null), TypeError);
  assert.throws(() => pool.setOversizedEvictionPolicy({ oversizedTargetBytes: 0 }), RangeError);
  assert.throws(() => pool.setOversizedEvictionPolicy({ oversizedIdleFrameTtl: -1 }), RangeError);

  pool.destroy();
  assert.throws(() => pool.setIdleFrameTtl(2), /destroyed/i);
  assert.throws(() => pool.trimToBudget(0), /destroyed/i);
  assert.throws(() => pool.setOversizedEvictionPolicy({ oversizedIdleFrameTtl: 0 }), /destroyed/i);
});

test('W200 classic-worker importScripts compatibility keeps new helpers isolated inside IIFE', () => {
  const rtPoolPath = fileURLToPath(new URL('../web/workers/rt_pool.js', import.meta.url));
  const source = readFileSync(rtPoolPath, 'utf8');

  const sandbox = { console };
  const ctx = vm.createContext(sandbox);
  ctx.self = ctx;

  assert.doesNotThrow(() => {
    vm.runInContext(source, ctx, { filename: rtPoolPath });
    vm.runInContext(source, ctx, { filename: rtPoolPath });
    vm.runInContext(
      `
      const DEFAULT_OVERSIZED_TARGET_BYTES = 1;
      const VALID_VIEW_DIMENSIONS = [];
      const VALID_VIEW_ASPECTS = [];
      function makeTextureViewKey() {}
      `,
      ctx
    );
  });

  assert.equal(ctx.PlayGta6RenderTargetPool.DEFAULT_OVERSIZED_TARGET_BYTES, 24 * 1024 * 1024);
  assert.equal(typeof ctx.PlayGta6RenderTargetPool.makeTextureViewKey, 'function');
});
