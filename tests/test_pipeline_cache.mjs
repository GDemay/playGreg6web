import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const pipelineCacheMod = require('../web/workers/pipeline_cache.js');
const shaderFallbackMod = require('../web/workers/shader_fallback_router.js');
const asyncPipelineCacheMod = require('../web/workers/async_pipeline_cache.js');
const reflectionLoaderMod = require('../web/workers/reflection_loader.js');
const {
  DEFAULT_MAX_FAST_LOOKUP_ENTRIES,
  DEFAULT_COMPILE_BUDGET_MS,
  DEFAULT_COMPILE_LATENCY_WINDOW_SIZE,
  PIPELINE_PRIORITY_TIERS,
  PIPELINE_PRIORITY_RANKS,
  PIPELINE_FAILURE_REASON,
  DEFAULT_MAX_ERROR_HISTORY,
  DEFAULT_MAX_QUARANTINED_KEYS,
  DEFAULT_MAX_NEGATIVE_CACHE_ENTRIES,
  MAX_PIPELINE_ERROR_MESSAGE_LENGTH,
  FRAME1_VERTEX_STRIDE_24B,
  FRAME1_CBV_MIN_BINDING_SIZE,
  FRAME1_VERTEX_LAYOUT_24B,
  DEFAULT_SAMPLER_DESCRIPTORS,
  SHADER_STAGE_BITS,
  fnv1a64Hex,
  buildFastPipelineKey,
  computeFastDescriptorHash,
  hashDescriptorKey,
  registerShaderModuleId,
  normalizeFallbackProfile,
  normalizePipelinePriority,
  normalizeDefaultSamplerKind,
  buildDefaultSamplerDescriptor,
  isSrgbColorFormat,
  getVertexFormatByteSize,
  buildVertexBufferLayoutForStride,
  normalizeVertexBufferLayouts,
  buildFallbackBindGroupLayoutDescriptor,
  normalizeRenderPipelineDescriptor,
  sanitizePipelineErrorMessage,
  classifyQuarantineReason,
  WebGpuPipelineCache,
} = pipelineCacheMod;
const { ShaderFallbackRouter } = shaderFallbackMod;
const { AsyncPipelineCache, PIPELINE_PRIORITY_TIER } = asyncPipelineCacheMod;
const { parseStageReflectionSidecar } = reflectionLoaderMod;

function createDeferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function createMockGpuDevice({
  withAsync = true,
  withPipelineLayout = true,
  withSampler = true,
  limits = undefined,
  advanceClockOnCompile = null,
  renderSyncImpl = null,
  renderAsyncImpl = null,
  computeSyncImpl = null,
  computeAsyncImpl = null,
  bglImpl = null,
  pipelineLayoutImpl = null,
  samplerImpl = null,
} = {}) {
  let nextId = 1;
  const calls = {
    createRenderPipeline: [],
    createRenderPipelineAsync: [],
    createComputePipeline: [],
    createComputePipelineAsync: [],
    createBindGroupLayout: [],
    createPipelineLayout: [],
    createSampler: [],
    createShaderModule: [],
  };

  const device = {
    calls,
    ...(limits ? { limits } : {}),
    createShaderModule(desc) {
      const id = nextId++;
      const mod = {
        label: desc.label || '',
        getCompilationInfo: async () => ({ messages: [] }),
      };
      calls.createShaderModule.push({ id, desc, mod });
      return mod;
    },
    createBindGroupLayout(desc) {
      if (advanceClockOnCompile) advanceClockOnCompile(1.5);
      const id = nextId++;
      if (bglImpl) {
        const layout = bglImpl(desc, id);
        calls.createBindGroupLayout.push({ id, desc, layout });
        return layout;
      }
      const layout = { kind: 'GPUBindGroupLayout', id, desc };
      calls.createBindGroupLayout.push({ id, desc, layout });
      return layout;
    },
    createRenderPipeline(desc) {
      if (advanceClockOnCompile) advanceClockOnCompile(4.0);
      const id = nextId++;
      if (renderSyncImpl) {
        const pipeline = renderSyncImpl(desc, id);
        calls.createRenderPipeline.push({ id, desc, pipeline });
        return pipeline;
      }
      const pipeline = { kind: 'GPURenderPipeline', id, desc };
      calls.createRenderPipeline.push({ id, desc, pipeline });
      return pipeline;
    },
    createComputePipeline(desc) {
      if (advanceClockOnCompile) advanceClockOnCompile(2.5);
      const id = nextId++;
      if (computeSyncImpl) {
        const pipeline = computeSyncImpl(desc, id);
        calls.createComputePipeline.push({ id, desc, pipeline });
        return pipeline;
      }
      const pipeline = { kind: 'GPUComputePipeline', id, desc };
      calls.createComputePipeline.push({ id, desc, pipeline });
      return pipeline;
    },
  };

  if (withPipelineLayout) {
    device.createPipelineLayout = (desc) => {
      const id = nextId++;
      if (pipelineLayoutImpl) {
        const layout = pipelineLayoutImpl(desc, id);
        calls.createPipelineLayout.push({ id, desc, layout });
        return layout;
      }
      const layout = { kind: 'GPUPipelineLayout', id, desc };
      calls.createPipelineLayout.push({ id, desc, layout });
      return layout;
    };
  }

  if (withSampler) {
    device.createSampler = (desc = {}) => {
      const id = nextId++;
      if (samplerImpl) {
        const sampler = samplerImpl(desc, id);
        calls.createSampler.push({ id, desc, sampler });
        return sampler;
      }
      const sampler = { kind: 'GPUSampler', id, desc };
      calls.createSampler.push({ id, desc, sampler });
      return sampler;
    };
  }

  if (withAsync) {
    device.createRenderPipelineAsync = async (desc) => {
      const id = nextId++;
      const record = { id, desc };
      calls.createRenderPipelineAsync.push(record);
      if (renderAsyncImpl) {
        return renderAsyncImpl(desc, id);
      }
      if (advanceClockOnCompile) advanceClockOnCompile(5.0);
      return { kind: 'GPURenderPipeline', id, async: true, desc };
    };

    device.createComputePipelineAsync = async (desc) => {
      const id = nextId++;
      const record = { id, desc };
      calls.createComputePipelineAsync.push(record);
      if (computeAsyncImpl) {
        return computeAsyncImpl(desc, id);
      }
      if (advanceClockOnCompile) advanceClockOnCompile(3.0);
      return { kind: 'GPUComputePipeline', id, async: true, desc };
    };
  }

  return device;
}

test('UMD module exposes globalThis.PlayGta6PipelineCache and exports expected API', () => {
  assert.equal(globalThis.PlayGta6PipelineCache, pipelineCacheMod);
  assert.equal(typeof hashDescriptorKey, 'function');
  assert.equal(typeof WebGpuPipelineCache, 'function');
  assert.equal(SHADER_STAGE_BITS.VERTEX, 0x1);
  assert.equal(SHADER_STAGE_BITS.FRAGMENT, 0x2);
  assert.equal(SHADER_STAGE_BITS.COMPUTE, 0x4);
});

test('hashDescriptorKey is deterministic, key-order invariant, and handles nested objects and BigInt', () => {
  const h1 = hashDescriptorKey({ a: 1, b: 2 });
  const h2 = hashDescriptorKey({ b: 2, a: 1 });
  assert.match(h1, /^0x[0-9a-f]{16}$/);
  assert.equal(h1, h2);

  // Nested objects with different key insertion orders + undefined omission
  const nestedA = {
    layout: 'auto',
    vertex: {
      entryPoint: 'vs_main',
      buffers: [{ arrayStride: 16, attributes: [{ format: 'float32x4', offset: 0, shaderLocation: 0 }] }],
    },
    fragment: {
      targets: [{ format: 'bgra8unorm', writeMask: 0xf }],
      entryPoint: 'fs_main',
    },
    primitive: { topology: 'triangle-list', cullMode: 'back' },
    depthStencil: undefined,
  };
  const nestedB = {
    primitive: { cullMode: 'back', topology: 'triangle-list' },
    fragment: {
      entryPoint: 'fs_main',
      targets: [{ writeMask: 0xf, format: 'bgra8unorm' }],
    },
    vertex: {
      buffers: [{ attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }], arrayStride: 16 }],
      entryPoint: 'vs_main',
    },
    layout: 'auto',
  };
  assert.equal(hashDescriptorKey(nestedA), hashDescriptorKey(nestedB));

  // Array order IS significant
  assert.notEqual(hashDescriptorKey({ list: [1, 2] }), hashDescriptorKey({ list: [2, 1] }));

  // BigInt support (including 64-bit WGPK hashes) and type separation
  const wgpkHashA = 0xdeadbeefcafebaben;
  const wgpkHashB = 0xdeadbeefcafebabfn;
  assert.equal(
    hashDescriptorKey({ vs: wgpkHashA, fs: wgpkHashB }),
    hashDescriptorKey({ fs: wgpkHashB, vs: wgpkHashA })
  );
  assert.notEqual(hashDescriptorKey({ id: wgpkHashA }), hashDescriptorKey({ id: wgpkHashB }));
  assert.notEqual(hashDescriptorKey({ id: 1n }), hashDescriptorKey({ id: 1 }));
  assert.notEqual(hashDescriptorKey({ id: 1n }), hashDescriptorKey({ id: '1' }));

  // -0 normalizes to 0
  assert.equal(hashDescriptorKey({ v: -0 }), hashDescriptorKey({ v: 0 }));

  // Circular references throw cleanly
  const circular = {};
  circular.self = circular;
  assert.throws(() => hashDescriptorKey(circular), /Circular reference/);
});

test('hashDescriptorKey distinguishes opaque GPUShaderModules and honors WGPK shader IDs', () => {
  const device = createMockGpuDevice();
  const modA = device.createShaderModule({ code: 'fn a() {}' });
  const modB = device.createShaderModule({ code: 'fn b() {}' });

  const descA1 = { layout: 'auto', vertex: { module: modA, entryPoint: 'vs_main' } };
  const descA2 = { vertex: { entryPoint: 'vs_main', module: modA }, layout: 'auto' };
  const descB = { layout: 'auto', vertex: { module: modB, entryPoint: 'vs_main' } };

  assert.equal(hashDescriptorKey(descA1), hashDescriptorKey(descA2));
  assert.notEqual(hashDescriptorKey(descA1), hashDescriptorKey(descB));

  // WGPK-labeled shader modules with identical hash produce identical descriptor keys
  const wgpkMod1 = device.createShaderModule({
    label: 'wgpk:vertex:0x0123456789abcdef',
    code: 'fn vs() {}',
  });
  const wgpkMod2 = device.createShaderModule({
    label: 'wgpk:vertex:0x0123456789abcdef',
    code: 'fn vs() {}',
  });
  assert.equal(
    hashDescriptorKey({ vertex: { module: wgpkMod1, entryPoint: 'vs_main' } }),
    hashDescriptorKey({ vertex: { module: wgpkMod2, entryPoint: 'vs_main' } })
  );

  // Explicit registerShaderModuleId binds opaque module objects to WGPK hashes
  const opaque1 = {};
  const opaque2 = {};
  assert.notEqual(
    hashDescriptorKey({ vertex: { module: opaque1 } }),
    hashDescriptorKey({ vertex: { module: opaque2 } })
  );
  registerShaderModuleId(opaque1, 0xfeedface12345678n);
  registerShaderModuleId(opaque2, 0xfeedface12345678n);
  assert.equal(
    hashDescriptorKey({ vertex: { module: opaque1 } }),
    hashDescriptorKey({ vertex: { module: opaque2 } })
  );
});

test('sync render, compute, and bindGroupLayout cache hits, misses, and telemetry', () => {
  let clock = 100;
  const device = createMockGpuDevice({
    advanceClockOnCompile: (dt) => {
      clock += dt;
    },
  });
  const cache = new WebGpuPipelineCache({ device, now: () => clock });

  const vs = device.createShaderModule({ label: 'wgpk:vertex:0x0000000000000001', code: 'vs' });
  const fs = device.createShaderModule({ label: 'wgpk:fragment:0x0000000000000002', code: 'fs' });
  const cs = device.createShaderModule({ label: 'wgpk:compute:0x0000000000000003', code: 'cs' });

  // BindGroupLayout miss then hit (with reordered descriptor keys)
  const bgl1 = cache.getOrCreateBindGroupLayout({
    entries: [{ binding: 0, visibility: SHADER_STAGE_BITS.VERTEX, buffer: { type: 'uniform' } }],
  });
  const bgl2 = cache.getOrCreateBindGroupLayout({
    entries: [{ buffer: { type: 'uniform' }, visibility: SHADER_STAGE_BITS.VERTEX, binding: 0 }],
  });
  assert.equal(bgl1, bgl2);
  assert.equal(device.calls.createBindGroupLayout.length, 1);

  // RenderPipeline miss then hit (reordered keys)
  const rp1 = cache.getOrCreateRenderPipeline({
    layout: 'auto',
    vertex: { module: vs, entryPoint: 'vs_main' },
    fragment: { module: fs, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
  });
  const rp2 = cache.getOrCreateRenderPipeline({
    fragment: { targets: [{ format: 'bgra8unorm' }], entryPoint: 'fs_main', module: fs },
    vertex: { entryPoint: 'vs_main', module: vs },
    layout: 'auto',
  });
  assert.equal(rp1, rp2);
  assert.equal(device.calls.createRenderPipeline.length, 1);

  // ComputePipeline miss then hit
  const cp1 = cache.getOrCreateComputePipeline({
    layout: 'auto',
    compute: { module: cs, entryPoint: 'cs_main' },
  });
  const cp2 = cache.getOrCreateComputePipeline({
    compute: { entryPoint: 'cs_main', module: cs },
    layout: 'auto',
  });
  assert.equal(cp1, cp2);
  assert.equal(device.calls.createComputePipeline.length, 1);

  // KeyHint override allows precomputed BigInt/string keys
  const rpHint1 = cache.getOrCreateRenderPipeline(
    { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_alt' } },
    0xabcdef01n
  );
  const rpHint2 = cache.getOrCreateRenderPipeline(
    { layout: 'auto', vertex: { module: vs, entryPoint: 'completely_different_desc' } },
    0xabcdef01n
  );
  assert.equal(rpHint1, rpHint2);
  assert.equal(device.calls.createRenderPipeline.length, 2);

  const stats = cache.getStats();
  assert.deepEqual(stats, {
    hits: 4,
    misses: 4,
    pipelineHits: 3,
    pipelineMisses: 3,
    fastLookupHits: 3,
    fastLookupMisses: 3,
    fastLookupHitRate: 0.5,
    negativeHits: 0,
    negativeCacheHits: 0,
    negativeCacheSize: 0,
    negativeCacheEvictions: 0,
    hitRate: 0.5,
    layoutDedupHits: 1,
    inlineSamplerLayouts: 0,
    avgCompileMs: (1.5 + 4.0 + 2.5 + 4.0) / 4,
    shaderPackGeneration: 0,
    renderHits: 2,
    renderMisses: 2,
    computeHits: 1,
    computeMisses: 1,
    layoutHits: 1,
    layoutMisses: 1,
    samplerHits: 0,
    samplerMisses: 0,
    inFlightDedupes: 0,
    evictions: 0,
    reclaimedCount: 0,
    renderCount: 2,
    computeCount: 1,
    layoutCount: 1,
    fallbackSamplerCount: 0,
    compileCount: 4,
    totalCompileMs: 1.5 + 4.0 + 2.5 + 4.0,
    maxCompileMs: 4.0,
    p50Ms: 2.5,
    p95Ms: 4.0,
    maxMs: 4.0,
    slowCompilesOverBudget: 0,
    compileLatency: {
      p50Ms: 2.5,
      p95Ms: 4.0,
      maxMs: 4.0,
      slowCompilesOverBudget: 0,
      budgetMs: DEFAULT_COMPILE_BUDGET_MS,
      sampleCount: 4,
      windowSize: DEFAULT_COMPILE_LATENCY_WINDOW_SIZE,
    },
    quarantineHits: 0,
    quarantinedKeys: 0,
    fallbackNormalizedCount: 0,
    vertexLayoutNormalizedCount: 0,
    layoutNormalizedCount: 0,
    fallbackRecoveryCount: 0,
    pipelineErrorCount: 0,
    pipelineErrorHistorySize: 0,
    maxErrorHistory: DEFAULT_MAX_ERROR_HISTORY,
    fallbackSamplersCreated: 0,
    fallbackSamplerBindingsSynthesized: 0,
    blendPipelineVariants: 0,
    srgbPipelineVariants: 0,
    cacheBreakdown: {
      render: { hits: 2, misses: 2, hitRate: 0.5 },
      compute: { hits: 1, misses: 1, hitRate: 0.5 },
      layout: { hits: 1, misses: 1, hitRate: 0.5 },
      sampler: { hits: 0, misses: 0, hitRate: 0 },
      total: { hits: 4, misses: 4, hitRate: 0.5 },
    },
  });
});

test('concurrent async render & compute pipeline compilations deduplicate in-flight promises', async () => {
  const renderDeferred = createDeferred();
  const computeDeferred = createDeferred();
  let clock = 0;

  const device = createMockGpuDevice({
    renderAsyncImpl: async (desc, id) => {
      const res = await renderDeferred.promise;
      clock += 7.5;
      return { kind: 'GPURenderPipeline', id, tag: res, desc };
    },
    computeAsyncImpl: async (desc, id) => {
      const res = await computeDeferred.promise;
      clock += 4.5;
      return { kind: 'GPUComputePipeline', id, tag: res, desc };
    },
  });

  const cache = new WebGpuPipelineCache({ device, now: () => clock });
  const vs = device.createShaderModule({ code: 'vs' });
  const cs = device.createShaderModule({ code: 'cs' });

  const renderDesc1 = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_main' } };
  const renderDesc2 = { vertex: { entryPoint: 'vs_main', module: vs }, layout: 'auto' };

  // Two simultaneous render async requests for the same PSO
  const rpPromise1 = cache.getOrCreateRenderPipelineAsync(renderDesc1);
  const rpPromise2 = cache.getOrCreateRenderPipelineAsync(renderDesc2);
  assert.equal(device.calls.createRenderPipelineAsync.length, 1);

  // Two simultaneous compute async requests for the same PSO
  const computeDesc = { layout: 'auto', compute: { module: cs, entryPoint: 'cs_main' } };
  const cpPromise1 = cache.getOrCreateComputePipelineAsync(computeDesc);
  const cpPromise2 = cache.getOrCreateComputePipelineAsync(computeDesc);
  assert.equal(device.calls.createComputePipelineAsync.length, 1);

  renderDeferred.resolve('ok-render');
  const [rp1, rp2] = await Promise.all([rpPromise1, rpPromise2]);

  computeDeferred.resolve('ok-compute');
  const [cp1, cp2] = await Promise.all([cpPromise1, cpPromise2]);

  assert.equal(rp1, rp2);
  assert.equal(cp1, cp2);

  // Subsequent call hits the populated cache directly
  const rp3 = await cache.getOrCreateRenderPipelineAsync(renderDesc1);
  const cp3 = await cache.getOrCreateComputePipelineAsync(computeDesc);
  assert.equal(rp3, rp1);
  assert.equal(cp3, cp1);
  assert.equal(device.calls.createRenderPipelineAsync.length, 1);
  assert.equal(device.calls.createComputePipelineAsync.length, 1);

  const stats = cache.getStats();
  assert.equal(stats.renderMisses, 1);
  assert.equal(stats.renderHits, 1);
  assert.equal(stats.computeMisses, 1);
  assert.equal(stats.computeHits, 1);
  assert.equal(stats.inFlightDedupes, 2);
  assert.equal(stats.renderCount, 1);
  assert.equal(stats.computeCount, 1);
  // Render compiled 0 -> 7.5ms (7.5ms); compute compiled 0 -> 12.0ms (12.0ms)
  assert.equal(stats.totalCompileMs, 19.5);
});

test('rejected async pipeline promises are cleaned up and do not poison the cache', async () => {
  let renderAttempts = 0;
  let computeAttempts = 0;

  const device = createMockGpuDevice({
    renderAsyncImpl: async (desc, id) => {
      renderAttempts += 1;
      if (renderAttempts === 1) {
        throw new Error('transient render compile failure');
      }
      return { kind: 'GPURenderPipeline', id, attempt: renderAttempts, desc };
    },
    computeAsyncImpl: async (desc, id) => {
      computeAttempts += 1;
      if (computeAttempts === 1) {
        throw new Error('transient compute compile failure');
      }
      return { kind: 'GPUComputePipeline', id, attempt: computeAttempts, desc };
    },
  });

  const cache = new WebGpuPipelineCache({ device, quarantineFailedKeys: false });
  const vs = device.createShaderModule({ code: 'vs' });
  const cs = device.createShaderModule({ code: 'cs' });
  const renderDesc = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_main' } };
  const computeDesc = { layout: 'auto', compute: { module: cs, entryPoint: 'cs_main' } };

  // Two concurrent callers observe the first failure
  const p1 = cache.getOrCreateRenderPipelineAsync(renderDesc);
  const p2 = cache.getOrCreateRenderPipelineAsync(renderDesc);
  await assert.rejects(p1, /transient render compile failure/);
  await assert.rejects(p2, /transient render compile failure/);
  assert.equal(cache.getStats().renderCount, 0);

  // Retry succeeds and populates the cache
  const recoveredRender = await cache.getOrCreateRenderPipelineAsync(renderDesc);
  assert.equal(recoveredRender.attempt, 2);
  assert.equal(cache.getStats().renderCount, 1);

  // Same for compute pipeline
  await assert.rejects(
    cache.getOrCreateComputePipelineAsync(computeDesc),
    /transient compute compile failure/
  );
  assert.equal(cache.getStats().computeCount, 0);

  const recoveredCompute = await cache.getOrCreateComputePipelineAsync(computeDesc);
  assert.equal(recoveredCompute.attempt, 2);
  assert.equal(cache.getStats().computeCount, 1);
});

test('async methods fall back to synchronous createRenderPipeline / createComputePipeline when async APIs are absent', async () => {
  const device = createMockGpuDevice({ withAsync: false });
  const cache = new WebGpuPipelineCache({ device });
  const vs = device.createShaderModule({ code: 'vs' });
  const cs = device.createShaderModule({ code: 'cs' });

  const rp = await cache.getOrCreateRenderPipelineAsync({
    layout: 'auto',
    vertex: { module: vs, entryPoint: 'vs_main' },
  });
  const cp = await cache.getOrCreateComputePipelineAsync({
    layout: 'auto',
    compute: { module: cs, entryPoint: 'cs_main' },
  });

  assert.equal(rp.kind, 'GPURenderPipeline');
  assert.equal(cp.kind, 'GPUComputePipeline');
  assert.equal(device.calls.createRenderPipeline.length, 1);
  assert.equal(device.calls.createComputePipeline.length, 1);
});

test('LRU eviction drops the least recently used entries when max capacity is exceeded', () => {
  const device = createMockGpuDevice();
  const cache = new WebGpuPipelineCache({
    device,
    maxRenderPipelines: 2,
    maxComputePipelines: 2,
    maxBindGroupLayouts: 2,
  });

  const vs = device.createShaderModule({ code: 'vs' });
  const makeRenderDesc = (entryPoint) => ({
    layout: 'auto',
    vertex: { module: vs, entryPoint },
  });

  const pA = cache.getOrCreateRenderPipeline(makeRenderDesc('vs_a'));
  const pB = cache.getOrCreateRenderPipeline(makeRenderDesc('vs_b'));
  assert.equal(cache.getStats().renderCount, 2);
  assert.equal(cache.getStats().evictions, 0);

  // Touch A so B becomes the least recently used entry
  const pAHit = cache.getOrCreateRenderPipeline(makeRenderDesc('vs_a'));
  assert.equal(pAHit, pA);

  // Insert C -> evicts B, keeps A and C
  const pC = cache.getOrCreateRenderPipeline(makeRenderDesc('vs_c'));
  assert.equal(cache.getStats().renderCount, 2);
  assert.equal(cache.getStats().evictions, 1);
  assert.equal(cache.getOrCreateRenderPipeline(makeRenderDesc('vs_a')), pA);
  assert.equal(cache.getOrCreateRenderPipeline(makeRenderDesc('vs_c')), pC);

  // Requesting B again must recompile (miss) and evict A (since C was touched after A)
  const pBRecompiled = cache.getOrCreateRenderPipeline(makeRenderDesc('vs_b'));
  assert.notEqual(pBRecompiled, pB);
  assert.equal(cache.getStats().evictions, 2);
  assert.equal(device.calls.createRenderPipeline.length, 4);

  // Compute pipeline LRU eviction
  const cs = device.createShaderModule({ code: 'cs' });
  cache.getOrCreateComputePipeline({ layout: 'auto', compute: { module: cs, entryPoint: 'c1' } });
  cache.getOrCreateComputePipeline({ layout: 'auto', compute: { module: cs, entryPoint: 'c2' } });
  cache.getOrCreateComputePipeline({ layout: 'auto', compute: { module: cs, entryPoint: 'c3' } });
  assert.equal(cache.getStats().computeCount, 2);
  assert.equal(cache.getStats().evictions, 3);

  // BindGroupLayout LRU eviction
  cache.getOrCreateBindGroupLayout({ entries: [{ binding: 0, visibility: 1, buffer: {} }] });
  cache.getOrCreateBindGroupLayout({ entries: [{ binding: 1, visibility: 1, buffer: {} }] });
  cache.getOrCreateBindGroupLayout({ entries: [{ binding: 2, visibility: 1, buffer: {} }] });
  assert.equal(cache.getStats().layoutCount, 2);
  assert.equal(cache.getStats().evictions, 4);
});

test('createBindGroupLayoutFromReflection normalizes S6 reflection metadata and deduplicates layouts', () => {
  const device = createMockGpuDevice();
  const cache = new WebGpuPipelineCache({ device });

  const reflectionGroupA = {
    group: 0,
    entries: [
      {
        binding: 2,
        visibility: ['fragment'],
        texture: { sample_type: 'float', view_dimension: '2d', multisampled: false },
      },
      {
        binding: 0,
        visibility: 'vertex | fragment',
        buffer: { type: 'uniform', has_dynamic_offset: true, min_binding_size: 256 },
      },
      {
        binding: 1,
        visibility: SHADER_STAGE_BITS.FRAGMENT,
        sampler: { type: 'filtering' },
      },
      {
        binding: 3,
        visibility: 'compute',
        storage_texture: { access: 'write-only', format: 'rgba8unorm', view_dimension: '2d' },
      },
    ],
  };

  // Same schema provided as a plain array in a different binding order using camelCase
  const reflectionGroupB = [
    {
      binding: 0,
      visibility: SHADER_STAGE_BITS.VERTEX | SHADER_STAGE_BITS.FRAGMENT,
      buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 256 },
    },
    {
      binding: 1,
      visibility: 'fs',
      sampler: { type: 'filtering' },
    },
    {
      binding: 2,
      visibility: 'fragment',
      texture: { sampleType: 'float', viewDimension: '2d', multisampled: false },
    },
    {
      binding: 3,
      visibility: ['cs'],
      storageTexture: { access: 'write-only', format: 'rgba8unorm', viewDimension: '2d' },
    },
  ];

  const layoutA = cache.createBindGroupLayoutFromReflection(reflectionGroupA);
  const layoutB = cache.createBindGroupLayoutFromReflection(reflectionGroupB);

  assert.equal(layoutA, layoutB);
  assert.equal(device.calls.createBindGroupLayout.length, 1);

  const submittedDesc = device.calls.createBindGroupLayout[0].desc;
  assert.deepEqual(submittedDesc, {
    entries: [
      {
        binding: 0,
        visibility: SHADER_STAGE_BITS.VERTEX | SHADER_STAGE_BITS.FRAGMENT,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 256 },
      },
      {
        binding: 1,
        visibility: SHADER_STAGE_BITS.FRAGMENT,
        sampler: { type: 'filtering' },
      },
      {
        binding: 2,
        visibility: SHADER_STAGE_BITS.FRAGMENT,
        texture: { sampleType: 'float', viewDimension: '2d', multisampled: false },
      },
      {
        binding: 3,
        visibility: SHADER_STAGE_BITS.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba8unorm', viewDimension: '2d' },
      },
    ],
  });

  // Duplicate binding index is rejected
  assert.throws(
    () =>
      cache.createBindGroupLayoutFromReflection([
        { binding: 0, visibility: 'vertex', buffer: {} },
        { binding: 0, visibility: 'fragment', buffer: {} },
      ]),
    /Duplicate reflection binding index 0/
  );

  // Unknown stage name is rejected
  assert.throws(
    () =>
      cache.createBindGroupLayoutFromReflection([
        { binding: 0, visibility: 'geometry', buffer: {} },
      ]),
    /Unknown shader stage visibility/
  );
});

test('prewarmRenderPipelines and prewarmComputePipelines bound concurrency and deduplicate', async () => {
  let activeRender = 0;
  let peakRender = 0;
  let activeCompute = 0;
  let peakCompute = 0;

  const device = createMockGpuDevice({
    renderAsyncImpl: async (desc, id) => {
      activeRender += 1;
      peakRender = Math.max(peakRender, activeRender);
      await new Promise((r) => setTimeout(r, 8));
      activeRender -= 1;
      return { kind: 'GPURenderPipeline', id, desc };
    },
    computeAsyncImpl: async (desc, id) => {
      activeCompute += 1;
      peakCompute = Math.max(peakCompute, activeCompute);
      await new Promise((r) => setTimeout(r, 8));
      activeCompute -= 1;
      return { kind: 'GPUComputePipeline', id, desc };
    },
  });

  const cache = new WebGpuPipelineCache({ device });
  const vs = device.createShaderModule({ code: 'vs' });
  const cs = device.createShaderModule({ code: 'cs' });

  // 8 render descriptors with 1 duplicate -> 7 unique compilations, max concurrency = 3
  const renderDescriptors = [];
  for (let i = 0; i < 7; i += 1) {
    renderDescriptors.push({
      layout: 'auto',
      vertex: { module: vs, entryPoint: `vs_${i}` },
    });
  }
  renderDescriptors.push({
    layout: 'auto',
    vertex: { module: vs, entryPoint: 'vs_0' },
  });

  const renderSummary = await cache.prewarmRenderPipelines(renderDescriptors, { concurrency: 3 });
  assert.equal(renderSummary.total, 8);
  assert.equal(renderSummary.pipelines.length, 8);
  assert.equal(renderSummary.pipelines[0], renderSummary.pipelines[7]);
  assert.equal(device.calls.createRenderPipelineAsync.length, 7);
  assert.ok(peakRender <= 3, `expected peak render concurrency <= 3, got ${peakRender}`);
  assert.ok(peakRender >= 2, `expected parallel prewarm execution, got peak ${peakRender}`);

  // 6 compute descriptors, max concurrency = 2
  const computeDescriptors = Array.from({ length: 6 }, (_, i) => ({
    layout: 'auto',
    compute: { module: cs, entryPoint: `cs_${i}` },
  }));

  const computeSummary = await cache.prewarmComputePipelines(computeDescriptors, { concurrency: 2 });
  assert.equal(computeSummary.total, 6);
  assert.equal(computeSummary.compiled, 6);
  assert.equal(computeSummary.cached, 0);
  assert.equal(device.calls.createComputePipelineAsync.length, 6);
  assert.ok(peakCompute <= 2, `expected peak compute concurrency <= 2, got ${peakCompute}`);

  // Second prewarm hits cache without compiling anything
  const cachedSummary = await cache.prewarmComputePipelines(computeDescriptors, { concurrency: 2 });
  assert.equal(cachedSummary.cached, 6);
  assert.equal(cachedSummary.compiled, 0);
  assert.equal(device.calls.createComputePipelineAsync.length, 6);
});

test('evictOlderThan and clear manage lifecycle and isolate stale in-flight compilations', async () => {
  let clock = 1000;
  const pendingRender = createDeferred();

  const device = createMockGpuDevice({
    renderAsyncImpl: async (desc, id) => {
      await pendingRender.promise;
      return { kind: 'GPURenderPipeline', id, desc };
    },
  });

  const cache = new WebGpuPipelineCache({ device, now: () => clock });
  const vs = device.createShaderModule({ code: 'vs' });
  const cs = device.createShaderModule({ code: 'cs' });

  // Populate entries at t = 1000
  cache.getOrCreateRenderPipeline({ layout: 'auto', vertex: { module: vs, entryPoint: 'old_vs' } });
  cache.getOrCreateComputePipeline({ layout: 'auto', compute: { module: cs, entryPoint: 'old_cs' } });
  cache.getOrCreateBindGroupLayout({ entries: [{ binding: 0, visibility: 1, buffer: {} }] });

  // Advance clock to t = 2500 and touch compute + add fresh render
  clock = 2500;
  cache.getOrCreateComputePipeline({ layout: 'auto', compute: { module: cs, entryPoint: 'old_cs' } });
  cache.getOrCreateRenderPipeline({ layout: 'auto', vertex: { module: vs, entryPoint: 'new_vs' } });

  // At t = 2600, evict anything older than 1000ms (cutoff = 1600):
  // old_vs (t=1000) and layout (t=1000) are evicted; old_cs (t=2500) and new_vs (t=2500) survive.
  clock = 2600;
  const evicted = cache.evictOlderThan(1000);
  assert.equal(evicted, 2);
  assert.equal(cache.getStats().renderCount, 1);
  assert.equal(cache.getStats().computeCount, 1);
  assert.equal(cache.getStats().layoutCount, 0);
  assert.equal(cache.getStats().evictions, 2);

  // Start an in-flight async compile, then clear() before it resolves
  const inFlightPromise = cache.getOrCreateRenderPipelineAsync({
    layout: 'auto',
    vertex: { module: vs, entryPoint: 'inflight_vs' },
  });
  cache.clear();
  assert.equal(cache.getStats().renderCount, 0);
  assert.equal(cache.getStats().computeCount, 0);
  assert.equal(cache.getStats().layoutCount, 0);

  // Resolving the pre-clear compilation should still return to caller but NOT repopulate cache
  pendingRender.resolve();
  const stalePipeline = await inFlightPromise;
  assert.equal(stalePipeline.kind, 'GPURenderPipeline');
  assert.equal(cache.getStats().renderCount, 0);

  // clear({ resetStats: true }) resets telemetry counters
  cache.clear({ resetStats: true });
  assert.deepEqual(cache.getStats(), {
    hits: 0,
    misses: 0,
    pipelineHits: 0,
    pipelineMisses: 0,
    fastLookupHits: 0,
    fastLookupMisses: 0,
    fastLookupHitRate: 0,
    negativeHits: 0,
    negativeCacheHits: 0,
    negativeCacheSize: 0,
    negativeCacheEvictions: 0,
    hitRate: 0,
    layoutDedupHits: 0,
    inlineSamplerLayouts: 0,
    avgCompileMs: 0,
    shaderPackGeneration: 0,
    renderHits: 0,
    renderMisses: 0,
    computeHits: 0,
    computeMisses: 0,
    layoutHits: 0,
    layoutMisses: 0,
    samplerHits: 0,
    samplerMisses: 0,
    inFlightDedupes: 0,
    evictions: 0,
    reclaimedCount: 0,
    renderCount: 0,
    computeCount: 0,
    layoutCount: 0,
    fallbackSamplerCount: 0,
    compileCount: 0,
    totalCompileMs: 0,
    maxCompileMs: 0,
    p50Ms: 0,
    p95Ms: 0,
    maxMs: 0,
    slowCompilesOverBudget: 0,
    compileLatency: {
      p50Ms: 0,
      p95Ms: 0,
      maxMs: 0,
      slowCompilesOverBudget: 0,
      budgetMs: DEFAULT_COMPILE_BUDGET_MS,
      sampleCount: 0,
      windowSize: DEFAULT_COMPILE_LATENCY_WINDOW_SIZE,
    },
    quarantineHits: 0,
    quarantinedKeys: 0,
    fallbackNormalizedCount: 0,
    vertexLayoutNormalizedCount: 0,
    layoutNormalizedCount: 0,
    fallbackRecoveryCount: 0,
    pipelineErrorCount: 0,
    pipelineErrorHistorySize: 0,
    maxErrorHistory: DEFAULT_MAX_ERROR_HISTORY,
    fallbackSamplersCreated: 0,
    fallbackSamplerBindingsSynthesized: 0,
    blendPipelineVariants: 0,
    srgbPipelineVariants: 0,
    cacheBreakdown: {
      render: { hits: 0, misses: 0, hitRate: 0 },
      compute: { hits: 0, misses: 0, hitRate: 0 },
      layout: { hits: 0, misses: 0, hitRate: 0 },
      sampler: { hits: 0, misses: 0, hitRate: 0 },
      total: { hits: 0, misses: 0, hitRate: 0 },
    },
  });
});

// --- G36: Fallback-Safe Pipeline & 24B Vertex/CBV Layout Normalization Tests ---

test('G36 preserves classic-worker importScripts IIFE isolation with zero global leakage', () => {
  const scriptPath = path.resolve(
    path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')),
    '../web/workers/pipeline_cache.js'
  );
  const source = fs.readFileSync(scriptPath, 'utf8');
  const sandbox = {
    TextEncoder,
    performance,
    Date,
  };
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox);

  assert.equal(typeof sandbox.PlayGta6PipelineCache, 'object');
  assert.equal(typeof sandbox.PlayGta6PipelineCache.WebGpuPipelineCache, 'function');
  assert.equal(typeof sandbox.PlayGta6PipelineCache.normalizeVertexBufferLayouts, 'function');
  assert.equal(typeof sandbox.PlayGta6PipelineCache.normalizeRenderPipelineDescriptor, 'function');
  assert.equal(typeof sandbox.PlayGta6PipelineCache.buildFallbackBindGroupLayoutDescriptor, 'function');
  assert.equal(typeof sandbox.PlayGta6PipelineCache.sanitizePipelineErrorMessage, 'function');

  // Verify no internal constants or helpers leaked onto global scope.
  assert.equal(sandbox.SHADER_STAGE_BITS, undefined);
  assert.equal(sandbox.FRAME1_VERTEX_LAYOUT_24B, undefined);
  assert.equal(sandbox.normalizeVertexBufferLayouts, undefined);
  assert.equal(sandbox.WebGpuPipelineCache, undefined);
});

test('G36 normalizeFallbackProfile and getVertexFormatByteSize handle aliases and invalid inputs', () => {
  assert.equal(normalizeFallbackProfile('default'), 'default');
  assert.equal(normalizeFallbackProfile('minimal'), 'default');
  assert.equal(normalizeFallbackProfile('fullscreen_triangle_cbv'), 'fullscreen_triangle_cbv');
  assert.equal(normalizeFallbackProfile('fullscreen-cbv'), 'fullscreen_triangle_cbv');
  assert.equal(normalizeFallbackProfile('FULLSCREEN_CBV'), 'fullscreen_triangle_cbv');
  assert.equal(normalizeFallbackProfile('textured_quad_24b'), 'textured_quad_24b');
  assert.equal(normalizeFallbackProfile('textured-24b'), 'textured_quad_24b');
  assert.equal(normalizeFallbackProfile('unknown-profile'), 'default');
  assert.equal(normalizeFallbackProfile(null), 'default');
  assert.equal(normalizeFallbackProfile(undefined), 'default');

  assert.equal(getVertexFormatByteSize('float32x4'), 16);
  assert.equal(getVertexFormatByteSize('float32x3'), 12);
  assert.equal(getVertexFormatByteSize('float32x2'), 8);
  assert.equal(getVertexFormatByteSize('float32'), 4);
  assert.equal(getVertexFormatByteSize('uint8x2'), 2);
  assert.equal(getVertexFormatByteSize('uint8'), 1);
  assert.equal(getVertexFormatByteSize('unorm10-10-10-2'), 4);
  assert.equal(getVertexFormatByteSize('not-a-format'), 0);
  assert.equal(getVertexFormatByteSize(null), 0);
});

test('G36 normalizeVertexBufferLayouts normalizes Frame-1 24B DrawInstanced(3, 1) and G34 profiles', () => {
  // 1. Frame-1 24-byte-stride buffer via stride number or missing/empty attributes -> canonical 24B layout
  const expected24B = [
    {
      arrayStride: 24,
      stepMode: 'vertex',
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x4' },
        { shaderLocation: 1, offset: 16, format: 'float32x2' },
      ],
    },
  ];
  assert.equal(FRAME1_VERTEX_STRIDE_24B, 24);
  assert.deepEqual(buildVertexBufferLayoutForStride(24), expected24B);
  assert.deepEqual(normalizeVertexBufferLayouts([{ arrayStride: 24 }]), expected24B);
  assert.deepEqual(normalizeVertexBufferLayouts([{ arrayStride: 24, attributes: [] }]), expected24B);
  assert.deepEqual(normalizeVertexBufferLayouts([], { vertexStride: 24 }), expected24B);
  assert.deepEqual(normalizeVertexBufferLayouts(24), expected24B);
  assert.equal(FRAME1_VERTEX_LAYOUT_24B.arrayStride, 24);

  // 2. fullscreen_triangle_cbv & default (@builtin(vertex_index)) ALWAYS normalize to empty buffers: []
  assert.deepEqual(
    normalizeVertexBufferLayouts(expected24B, { fallbackProfile: 'fullscreen_triangle_cbv' }),
    []
  );
  assert.deepEqual(
    normalizeVertexBufferLayouts(expected24B, { fallbackProfile: 'fullscreen-cbv' }),
    []
  );
  assert.deepEqual(
    normalizeVertexBufferLayouts(expected24B, { fallbackProfile: 'default' }),
    []
  );
  assert.deepEqual(
    buildVertexBufferLayoutForStride(24, { fallbackProfile: 'fullscreen_triangle_cbv' }),
    []
  );

  // 3. textured_quad_24b enforces canonical 24B layout even if input was missing or single-attribute
  assert.deepEqual(
    normalizeVertexBufferLayouts([], { fallbackProfile: 'textured_quad_24b' }),
    expected24B
  );
  assert.deepEqual(
    normalizeVertexBufferLayouts(
      [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }],
      { fallbackProfile: 'textured-24b' }
    ),
    expected24B
  );

  // 4. Clamping invalid offsets, unaligned strides, oversized strides, unknown formats, duplicate locations
  const clamped = normalizeVertexBufferLayouts([
    {
      arrayStride: 24,
      stepMode: 'bogus-step',
      attributes: [
        { shaderLocation: 0, offset: -8, format: 'float32x4' },
        { shaderLocation: 0, offset: 22, format: 'float32x2' }, // duplicate loc 0 & offset 22+8 > 24 -> loc 1, offset 16
      ],
    },
    {
      arrayStride: 4096, // exceeds WebGPU default 2048 -> clamps to 2048
      stepMode: 'instance',
      attributes: [
        { shaderLocation: 99, offset: 3000, format: 'float32x4' }, // loc -> 2, offset -> 2032
      ],
    },
    {
      arrayStride: -16, // invalid negative stride -> derived from attributes
      attributes: [
        { shaderLocation: 5, offset: 3, format: 'invalid_fmt' }, // fallback float32x4, offset aligned to 0 -> stride 16
      ],
    },
    {
      arrayStride: 8, // stride smaller than float32x4 (16B) -> expands to 16B, offset 0
      attributes: [
        { shaderLocation: 6, offset: 4, format: 'float32x4' },
      ],
    },
  ]);

  assert.deepEqual(clamped, [
    {
      arrayStride: 24,
      stepMode: 'vertex',
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x4' },
        { shaderLocation: 1, offset: 16, format: 'float32x2' },
      ],
    },
    {
      arrayStride: 2048,
      stepMode: 'instance',
      attributes: [
        { shaderLocation: 2, offset: 2032, format: 'float32x4' },
      ],
    },
    {
      arrayStride: 16,
      stepMode: 'vertex',
      attributes: [
        { shaderLocation: 5, offset: 0, format: 'float32x4' },
      ],
    },
    {
      arrayStride: 16,
      stepMode: 'vertex',
      attributes: [
        { shaderLocation: 6, offset: 0, format: 'float32x4' },
      ],
    },
  ]);

  // 5. Unaligned positive stride rounds up to multiple of 4 (e.g., 22 -> 24)
  assert.deepEqual(buildVertexBufferLayoutForStride(22), expected24B);
  assert.deepEqual(buildVertexBufferLayoutForStride(0), []);
  assert.deepEqual(buildVertexBufferLayoutForStride(-4), []);
  assert.deepEqual(buildVertexBufferLayoutForStride(NaN), []);
});

test('G36 fallback bind-group and pipeline layout helpers match G34 CBV and textured_quad_24b profiles', () => {
  assert.equal(FRAME1_CBV_MIN_BINDING_SIZE, 256);

  // fullscreen_triangle_cbv -> group(0) binding(0) uniform CBV (VS | FS, minBindingSize 256)
  const cbvBglDesc = buildFallbackBindGroupLayoutDescriptor('fullscreen_triangle_cbv');
  assert.deepEqual(cbvBglDesc, {
    label: 'fallback-bgl:fullscreen_triangle_cbv:group0',
    entries: [
      {
        binding: 0,
        visibility: SHADER_STAGE_BITS.VERTEX | SHADER_STAGE_BITS.FRAGMENT,
        buffer: {
          type: 'uniform',
          hasDynamicOffset: false,
          minBindingSize: 256,
        },
      },
    ],
  });

  // textured_quad_24b -> group(0) binding(0) 2d float texture + binding(1) filtering sampler
  const texBglDesc = buildFallbackBindGroupLayoutDescriptor('textured_quad_24b');
  assert.deepEqual(texBglDesc, {
    label: 'fallback-bgl:textured_quad_24b:group0',
    entries: [
      {
        binding: 0,
        visibility: SHADER_STAGE_BITS.FRAGMENT,
        texture: {
          sampleType: 'float',
          viewDimension: '2d',
          multisampled: false,
        },
      },
      {
        binding: 1,
        visibility: SHADER_STAGE_BITS.FRAGMENT,
        sampler: {
          type: 'filtering',
        },
      },
    ],
  });

  // default -> empty entries
  const defBglDesc = buildFallbackBindGroupLayoutDescriptor('default');
  assert.deepEqual(defBglDesc, {
    label: 'fallback-bgl:default:group0',
    entries: [],
  });

  // Mutating returned descriptor does not affect subsequent calls
  cbvBglDesc.entries[0].buffer.minBindingSize = 9999;
  assert.equal(
    buildFallbackBindGroupLayoutDescriptor('fullscreen_triangle_cbv').entries[0].buffer.minBindingSize,
    256
  );

  // Cached fallback BindGroupLayout & PipelineLayout deduplication
  const device = createMockGpuDevice();
  const cache = new WebGpuPipelineCache({ device });

  const cbvPl1 = cache.getOrCreateFallbackPipelineLayout('fullscreen_triangle_cbv');
  const cbvPl2 = cache.getOrCreateFallbackPipelineLayout('fullscreen-cbv');
  assert.equal(cbvPl1, cbvPl2);
  assert.equal(cbvPl1.kind, 'GPUPipelineLayout');
  assert.equal(cbvPl1.desc.bindGroupLayouts.length, 1);
  assert.equal(device.calls.createBindGroupLayout.length, 1);
  assert.equal(device.calls.createPipelineLayout.length, 1);

  const texPl = cache.getOrCreateFallbackPipelineLayout('textured_quad_24b');
  assert.equal(texPl.kind, 'GPUPipelineLayout');
  assert.equal(texPl.desc.bindGroupLayouts.length, 1);
  assert.equal(device.calls.createBindGroupLayout.length, 2);
  assert.equal(device.calls.createPipelineLayout.length, 2);

  const defaultPl = cache.getOrCreateFallbackPipelineLayout('default');
  assert.equal(defaultPl.kind, 'GPUPipelineLayout');
  assert.equal(defaultPl.desc.bindGroupLayouts.length, 0);

  // Deterministic hash stability for fallback pipeline layouts
  const vs = device.createShaderModule({ label: 'fallback:fullscreen_triangle_cbv', code: 'vs' });
  assert.equal(
    hashDescriptorKey({ layout: cbvPl1, vertex: { module: vs, entryPoint: 'vs_main' } }),
    hashDescriptorKey({ vertex: { entryPoint: 'vs_main', module: vs }, layout: cbvPl2 })
  );

  // Graceful fallback to 'auto' when device.createPipelineLayout is unavailable
  const noPlDevice = createMockGpuDevice({ withPipelineLayout: false });
  const noPlCache = new WebGpuPipelineCache({ device: noPlDevice });
  assert.equal(noPlCache.getOrCreateFallbackPipelineLayout('fullscreen_triangle_cbv'), 'auto');

  // Graceful fallback to 'auto' + recorded sanitized error when device.createPipelineLayout throws
  const failingPlDevice = createMockGpuDevice({
    pipelineLayoutImpl: () => {
      const err = new Error('GPUValidationError: layout failed at C:\\Users\\dev\\secret\\p.wgsl');
      err.name = 'GPUValidationError';
      throw err;
    },
  });
  const failingPlCache = new WebGpuPipelineCache({ device: failingPlDevice, now: () => 500 });
  assert.equal(failingPlCache.getOrCreateFallbackPipelineLayout('fullscreen_triangle_cbv'), 'auto');
  assert.equal(failingPlCache.getStats().pipelineErrorCount, 1);
  const plErrors = failingPlCache.getPipelineErrors();
  assert.equal(plErrors.length, 1);
  assert.equal(plErrors[0].kind, 'pipeline-layout');
  assert.equal(plErrors[0].errorType, 'validation');
  assert.equal(plErrors[0].recovered, true);
  assert.doesNotMatch(plErrors[0].message, /Users\\dev/);
  assert.match(plErrors[0].message, /<path>/);
});

test('G36 normalizeRenderPipelineDescriptor and PipelineCache normalize fallback layouts without mutating caller descriptors', () => {
  const device = createMockGpuDevice();
  const cache = new WebGpuPipelineCache({ device, defaultColorFormat: 'rgba8unorm' });
  const router = new ShaderFallbackRouter({ defaultProfile: 'fullscreen_triangle_cbv' });

  // Resolve real G34 fallback shaders for Frame-1 DrawInstanced(3, 1)
  const resolvedVs = router.resolve('0x1111222233334444', 'vertex', {
    profile: 'fullscreen_triangle_cbv',
  });
  const resolvedFs = router.resolve('0x5555666677778888', 'fragment', {
    profile: 'fullscreen_triangle_cbv',
  });
  const vsMod = device.createShaderModule({
    label: `fallback:${resolvedVs.profile}:vs`,
    code: resolvedVs.wgsl,
  });
  const fsMod = device.createShaderModule({
    label: `fallback:${resolvedFs.profile}:fs`,
    code: resolvedFs.wgsl,
  });

  // Mismatched custom pipeline layout + 24B vertex buffer + missing fragment targets + invalid topology
  const mismatchedLayout = { kind: 'GPUPipelineLayout', id: 999, custom: true };
  const rawInput = {
    layout: mismatchedLayout,
    vertex: {
      module: vsMod,
      entryPoint: '',
      buffers: [
        {
          arrayStride: 24,
          attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
        },
      ],
    },
    fragment: {
      module: fsMod,
      entryPoint: '',
      targets: [],
    },
    primitive: {
      topology: 'triangle-fan-invalid',
    },
  };
  const inputSnapshot = JSON.stringify({
    vertexBuffers: rawInput.vertex.buffers,
    targets: rawInput.fragment.targets,
    topology: rawInput.primitive.topology,
  });

  // 1. Default auto-layout normalization for fullscreen_triangle_cbv (detected from module label!)
  const rpAuto = cache.getOrCreateRenderPipeline(rawInput);
  assert.equal(rpAuto.desc.layout, 'auto');
  assert.deepEqual(rpAuto.desc.vertex.buffers, []);
  assert.equal(rpAuto.desc.vertex.entryPoint, 'vs_main');
  assert.equal(rpAuto.desc.fragment.entryPoint, 'fs_main');
  assert.deepEqual(rpAuto.desc.fragment.targets, [{ format: 'rgba8unorm' }]);
  assert.equal(rpAuto.desc.primitive.topology, 'triangle-list');

  // Verify caller's raw descriptor was NEVER mutated
  assert.equal(rawInput.layout, mismatchedLayout);
  assert.equal(
    JSON.stringify({
      vertexBuffers: rawInput.vertex.buffers,
      targets: rawInput.fragment.targets,
      topology: rawInput.primitive.topology,
    }),
    inputSnapshot
  );

  // 2. Explicit CBV layout mode binds cached group(0) binding(0) CBV GPUPipelineLayout
  const rpExplicit = cache.getOrCreateFallbackRenderPipeline(rawInput, {
    fallbackProfile: 'fullscreen_triangle_cbv',
    layoutMode: 'explicit',
  });
  assert.equal(rpExplicit.desc.layout.kind, 'GPUPipelineLayout');
  assert.equal(rpExplicit.desc.layout.desc.label, 'fallback-pl:fullscreen_triangle_cbv');
  assert.deepEqual(rpExplicit.desc.vertex.buffers, []);

  // Passing that exact compatible fallback GPUPipelineLayout back preserves it untouched
  const compatNorm = normalizeRenderPipelineDescriptor(
    {
      layout: rpExplicit.desc.layout,
      vertex: { module: vsMod, entryPoint: 'vs_main', buffers: [] },
      fragment: { module: fsMod, entryPoint: 'fs_main', targets: [{ format: 'rgba8unorm' }] },
    },
    { fallbackProfile: 'fullscreen_triangle_cbv' }
  );
  assert.equal(compatNorm.layoutNormalized, false);
  assert.equal(compatNorm.descriptor.layout, rpExplicit.desc.layout);

  // 3. textured_quad_24b fallback normalizes to canonical 24B vertex layout + group(0) tex/sampler layout
  const quadVs = device.createShaderModule({
    label: 'fallback:textured_quad_24b:vs',
    code: router.resolve('0xaaaa000000000001', 'vertex', { profile: 'textured_quad_24b' }).wgsl,
  });
  const quadFs = device.createShaderModule({
    label: 'fallback:textured_quad_24b:fs',
    code: router.resolve('0xaaaa000000000002', 'fragment', { profile: 'textured_quad_24b' }).wgsl,
  });
  const rpQuad = cache.getOrCreateFallbackRenderPipeline(
    {
      // omitted layout + helper metadata stripped before WebGPU call
      fallbackProfile: 'textured_quad_24b',
      vertexStride: 24,
      vertex: { module: quadVs, entryPoint: 'vs_main' },
      fragment: { module: quadFs, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
    },
    { layoutMode: 'explicit' }
  );
  assert.equal('fallbackProfile' in rpQuad.desc, false);
  assert.equal('vertexStride' in rpQuad.desc, false);
  assert.equal(rpQuad.desc.layout.desc.label, 'fallback-pl:textured_quad_24b');
  assert.deepEqual(rpQuad.desc.vertex.buffers, [
    {
      arrayStride: 24,
      stepMode: 'vertex',
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x4' },
        { shaderLocation: 1, offset: 16, format: 'float32x2' },
      ],
    },
  ]);

  // 4. Compute pipeline with missing layout normalizes to 'auto'
  const csMod = device.createShaderModule({ label: 'wgpk:compute:0x0000000000000099', code: 'cs' });
  const cpAuto = cache.getOrCreateComputePipeline({
    compute: { module: csMod, entryPoint: 'cs_main' },
  });
  assert.equal(cpAuto.desc.layout, 'auto');

  const stats = cache.getStats();
  assert.equal(stats.fallbackNormalizedCount, 4);
  assert.equal(stats.vertexLayoutNormalizedCount, 3);
  assert.equal(stats.layoutNormalizedCount, 4);
  assert.equal(stats.pipelineErrorCount, 0);
});

test('G36 auto-layout retry recovery works for sync, async, and deduplicated concurrent fallback pipelines', async () => {
  let syncCalls = 0;
  let asyncCalls = 0;

  const device = createMockGpuDevice({
    renderSyncImpl: (desc, id) => {
      syncCalls += 1;
      if (desc.layout !== 'auto') {
        const err = new Error('BindGroupLayout mismatch on group(0) binding(0)');
        err.name = 'GPUValidationError';
        throw err;
      }
      return { kind: 'GPURenderPipeline', id, recovered: true, desc };
    },
    renderAsyncImpl: async (desc, id) => {
      asyncCalls += 1;
      await new Promise((r) => setTimeout(r, 4));
      if (desc.layout !== 'auto') {
        const err = new Error('Async BindGroupLayout mismatch on group(0)');
        err.name = 'GPUValidationError';
        throw err;
      }
      return { kind: 'GPURenderPipeline', id, asyncRecovered: true, desc };
    },
  });

  const cache = new WebGpuPipelineCache({ device, now: () => 1234 });
  const vs = device.createShaderModule({ label: 'fallback:fullscreen_triangle_cbv:vs', code: 'vs' });
  const fs = device.createShaderModule({ label: 'fallback:fullscreen_triangle_cbv:fs', code: 'fs' });

  // Sync recovery: explicit layout fails -> automatically retried with layout: 'auto'
  const syncRecovered = cache.getOrCreateFallbackRenderPipeline(
    {
      vertex: { module: vs, entryPoint: 'vs_main' },
      fragment: { module: fs, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
    },
    { fallbackProfile: 'fullscreen_triangle_cbv', layoutMode: 'explicit' }
  );
  assert.equal(syncRecovered.recovered, true);
  assert.equal(syncRecovered.desc.layout, 'auto');
  assert.equal(syncCalls, 2);

  // Async recovery with concurrent in-flight dedupe: both callers receive recovered pipeline
  const vs2 = device.createShaderModule({ label: 'fallback:textured_quad_24b:vs', code: 'vs2' });
  const fs2 = device.createShaderModule({ label: 'fallback:textured_quad_24b:fs', code: 'fs2' });
  const asyncDesc = {
    vertex: { module: vs2, entryPoint: 'vs_main' },
    fragment: { module: fs2, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
  };
  const [a1, a2] = await Promise.all([
    cache.getOrCreateFallbackRenderPipelineAsync(asyncDesc, {
      fallbackProfile: 'textured_quad_24b',
      layoutMode: 'explicit',
    }),
    cache.getOrCreateFallbackRenderPipelineAsync(asyncDesc, {
      fallbackProfile: 'textured_quad_24b',
      layoutMode: 'explicit',
    }),
  ]);
  assert.equal(a1, a2);
  assert.equal(a1.asyncRecovered, true);
  assert.equal(a1.desc.layout, 'auto');
  assert.equal(asyncCalls, 2);

  const stats = cache.getStats();
  assert.equal(stats.fallbackRecoveryCount, 2);
  assert.equal(stats.inFlightDedupes, 1);
  assert.equal(stats.pipelineErrorCount, 2);

  const errors = cache.getPipelineErrors();
  assert.equal(errors.length, 2);
  assert.equal(errors[0].recovered, true);
  assert.equal(errors[0].fallbackProfile, 'fullscreen_triangle_cbv');
  assert.equal(errors[1].recovered, true);
  assert.equal(errors[1].fallbackProfile, 'textured_quad_24b');
});

test('G36 structured pipeline creation error tracking is sanitized, bounded, and isolates callbacks', async () => {
  let clock = 100;
  const callbackRecords = [];

  const device = createMockGpuDevice({
    renderSyncImpl: () => {
      const err = new Error(
        'Validation failed at C:\\Users\\alice\\local-data\\gta6\\sh.wgsl and "D:\\My Games\\GTA6\\x.rpf" and \\\\srv01\\share\\a.bin and file:///Users/alice/a.wgsl and /home/alice/dev/p.wgsl (rgba8unorm/bgra8unorm) hash 0x123456789abcdef0'
      );
      err.name = 'GPUValidationError';
      throw err;
    },
    computeSyncImpl: () => {
      const hexBlob = 'cd'.repeat(40);
      const spacedHex = Array.from({ length: 20 }, (_, i) => i.toString(16).padStart(2, '0')).join(' ');
      const byteList = Array.from({ length: 20 }, (_, i) => String(i)).join(', ');
      const b64Blob = 'QUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFB';
      const err = new Error(
        `OOM:\nhex=${hexBlob}\tbytes=[${byteList}] dump=${spacedHex} uri=data:application/octet-stream;base64,${b64Blob} raw=${b64Blob}\x00\x07`
      );
      err.name = 'GPUOutOfMemoryError';
      throw err;
    },
    bglImpl: () => {
      const err = new Error(`Long failure: ${'binding_conflict '.repeat(50)}`);
      err.name = 'GPUInternalError';
      throw err;
    },
  });

  const cache = new WebGpuPipelineCache({
    device,
    maxErrorHistory: 3,
    now: () => {
      clock += 10;
      return clock;
    },
    onPipelineError: (rec) => {
      callbackRecords.push({ ...rec });
      rec.message = 'MUTATED_BY_CALLBACK';
      if (rec.seq === 2) {
        throw new Error('user callback exception must be isolated');
      }
    },
  });

  const vs = device.createShaderModule({ code: 'vs' });
  const cs = device.createShaderModule({ code: 'cs' });

  // 1. Render error (host paths stripped, WGPK hash + format token preserved)
  assert.throws(
    () => cache.getOrCreateRenderPipeline({ layout: 'auto', vertex: { module: vs, entryPoint: 'vs' } }, 'C:\\Users\\alice\\key.pso'),
    /Validation failed/
  );

  // 2. Compute error (raw buffers/hex/base64/control chars stripped; throwing callback isolated)
  assert.throws(
    () => cache.getOrCreateComputePipeline({ layout: 'auto', compute: { module: cs, entryPoint: 'cs' } }),
    /OOM/
  );

  // 3. BindGroupLayout error (>512 chars capped cleanly)
  assert.throws(
    () => cache.getOrCreateBindGroupLayout({ entries: [{ binding: 0, visibility: 1, buffer: {} }] }),
    /Long failure/
  );

  // 4. Fourth error evicts seq 1 from ring (maxErrorHistory = 3)
  assert.throws(
    () => cache.getOrCreateRenderPipeline({ layout: 'auto', vertex: { module: vs, entryPoint: 'vs2' } }),
    /Validation failed/
  );

  const stats = cache.getStats();
  assert.equal(stats.pipelineErrorCount, 4);
  assert.equal(stats.pipelineErrorHistorySize, 3);
  assert.equal(stats.maxErrorHistory, 3);

  // Check callback captured seq 1 before eviction and sanitized paths + key
  assert.equal(callbackRecords.length, 4);
  assert.equal(callbackRecords[0].seq, 1);
  assert.equal(callbackRecords[0].kind, 'render');
  assert.equal(callbackRecords[0].errorType, 'validation');
  assert.doesNotMatch(callbackRecords[0].key, /alice/i);
  assert.match(callbackRecords[0].key, /<path>/);
  assert.doesNotMatch(callbackRecords[0].message, /alice/i);
  assert.doesNotMatch(callbackRecords[0].message, /C:\\/);
  assert.doesNotMatch(callbackRecords[0].message, /\\\\srv01/);
  assert.doesNotMatch(callbackRecords[0].message, /file:\/\//);
  assert.doesNotMatch(callbackRecords[0].message, /\/home\//);
  assert.match(callbackRecords[0].message, /rgba8unorm\/bgra8unorm/);
  assert.match(callbackRecords[0].message, /0x123456789abcdef0/);

  // Ring retains seq [2, 3, 4]
  const history = cache.getPipelineErrors();
  assert.deepEqual(
    history.map((e) => e.seq),
    [2, 3, 4]
  );
  assert.equal(history[0].kind, 'compute');
  assert.equal(history[0].errorType, 'out-of-memory');
  assert.doesNotMatch(history[0].message, /cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd/);
  assert.doesNotMatch(history[0].message, /QUFBQUFBQUFBQUFB/);
  assert.doesNotMatch(history[0].message, /[\x00-\x1F\x7F]/);
  assert.match(history[0].message, /<buffer>/);

  assert.equal(history[1].kind, 'bind-group-layout');
  assert.equal(history[1].errorType, 'internal');
  assert.equal(history[1].message.length, MAX_PIPELINE_ERROR_MESSAGE_LENGTH);
  assert.ok(history[1].message.endsWith('...'));

  // JSON-serializable & defensive copy isolation
  assert.equal(JSON.parse(JSON.stringify(history)).length, 3);
  history[0].message = 'CORRUPTED';
  assert.notEqual(cache.getPipelineErrors()[0].message, 'CORRUPTED');

  // Direct sanitizePipelineErrorMessage helper checks
  assert.equal(sanitizePipelineErrorMessage(null), '');
  assert.equal(sanitizePipelineErrorMessage(undefined), '');

  // clearPipelineErrors clears ring but keeps cumulative count and monotonic seq
  cache.clearPipelineErrors();
  assert.deepEqual(cache.getPipelineErrors(), []);
  assert.equal(cache.getStats().pipelineErrorHistorySize, 0);
  assert.equal(cache.getStats().pipelineErrorCount, 4);

  assert.throws(
    () => cache.getOrCreateRenderPipeline({ layout: 'auto', vertex: { module: vs, entryPoint: 'vs3' } }),
    /Validation failed/
  );
  assert.equal(cache.getPipelineErrors()[0].seq, 5);
  assert.equal(cache.getStats().pipelineErrorCount, 5);

  // clear({ resetStats: true }) resets error counts, seq, and ring
  cache.clear({ resetStats: true });
  assert.equal(cache.getStats().pipelineErrorCount, 0);
  assert.equal(cache.getStats().pipelineErrorHistorySize, 0);
  assert.deepEqual(cache.getPipelineErrors(), []);
});

test('G36 prewarmRenderPipelines deduplicates normalized fallback descriptors with subsequent cache lookups', async () => {
  const device = createMockGpuDevice();
  const cache = new WebGpuPipelineCache({ device });
  const vs = device.createShaderModule({ label: 'fallback:fullscreen_triangle_cbv:vs', code: 'vs' });
  const fs = device.createShaderModule({ label: 'fallback:fullscreen_triangle_cbv:fs', code: 'fs' });

  const summary = await cache.prewarmRenderPipelines([
    {
      descriptor: {
        // missing layout + 24B buffers -> normalizes to layout: 'auto', buffers: []
        vertex: { module: vs, entryPoint: 'vs_main', buffers: [{ arrayStride: 24 }] },
        fragment: { module: fs, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
      },
      options: { fallbackProfile: 'fullscreen_triangle_cbv' },
    },
    {
      descriptor: {
        layout: 'auto',
        vertex: { module: vs, entryPoint: 'vs_main', buffers: [] },
        fragment: { module: fs, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
      },
      options: { fallbackProfile: 'fullscreen_triangle_cbv' },
    },
  ]);

  assert.equal(summary.total, 2);
  assert.equal(summary.pipelines[0], summary.pipelines[1]);
  assert.equal(device.calls.createRenderPipelineAsync.length, 1);

  // Subsequent sync lookup hits the prewarmed normalized entry directly
  const hit = cache.getOrCreateFallbackRenderPipeline(
    {
      vertex: { module: vs, entryPoint: 'vs_main', buffers: [{ arrayStride: 24 }] },
      fragment: { module: fs, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
    },
    { fallbackProfile: 'fullscreen_triangle_cbv' }
  );
  assert.equal(hit, summary.pipelines[0]);
  assert.equal(device.calls.createRenderPipeline.length, 0);
});

// --- W118: Failed-Pipeline Quarantine Cache & Compile-Latency Telemetry Tests ---

test('W118 failed render and compute pipelines are quarantined by default and skip repeat device calls', async () => {
  assert.equal(DEFAULT_MAX_QUARANTINED_KEYS, 512);

  let renderSyncCalls = 0;
  let computeSyncCalls = 0;
  let renderAsyncCalls = 0;
  let computeAsyncCalls = 0;

  const device = createMockGpuDevice({
    renderSyncImpl: () => {
      renderSyncCalls += 1;
      const err = new Error('Validation failed at C:\\Users\\dev\\shader.wgsl');
      err.name = 'GPUValidationError';
      throw err;
    },
    computeSyncImpl: () => {
      computeSyncCalls += 1;
      throw new Error('Invalid workgroup size');
    },
    renderAsyncImpl: async () => {
      renderAsyncCalls += 1;
      throw new Error('Async render validation failure');
    },
    computeAsyncImpl: async () => {
      computeAsyncCalls += 1;
      throw new Error('Async compute validation failure');
    },
  });

  let clock = 100;
  const cache = new WebGpuPipelineCache({ device, nowFn: () => clock });
  const vs = device.createShaderModule({ label: 'wgpk:vertex:0x0000000000000101', code: 'vs' });
  const cs = device.createShaderModule({ label: 'wgpk:compute:0x0000000000000102', code: 'cs' });

  const renderDescA = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_fail_sync' } };
  const renderDescAReordered = { vertex: { entryPoint: 'vs_fail_sync', module: vs }, layout: 'auto' };
  const renderKeyA = hashDescriptorKey(renderDescA);

  // First sync render failure throws and quarantines the canonical key
  assert.throws(() => cache.getOrCreateRenderPipeline(renderDescA), /Validation failed/);
  assert.equal(renderSyncCalls, 1);
  assert.equal(cache.isQuarantined(renderKeyA), true);
  assert.equal(cache.isQuarantined(renderDescAReordered), true);

  // Subsequent sync requests for identical descriptor return null without invoking device
  clock += 5;
  assert.equal(cache.getOrCreateRenderPipeline(renderDescA), null);
  assert.equal(cache.getOrCreateRenderPipeline(renderDescAReordered), null);
  assert.equal(renderSyncCalls, 1);

  // First sync compute failure throws and quarantines
  const computeDescA = { layout: 'auto', compute: { module: cs, entryPoint: 'cs_fail_sync' } };
  assert.throws(() => cache.getOrCreateComputePipeline(computeDescA), /Invalid workgroup size/);
  assert.equal(computeSyncCalls, 1);
  assert.equal(cache.isQuarantined(computeDescA), true);

  // Repeat compute lookup hits quarantine
  assert.equal(cache.getOrCreateComputePipeline(computeDescA), null);
  assert.equal(computeSyncCalls, 1);

  // Async render failure rejects once, then subsequent async calls resolve to null
  const renderDescAsync = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_fail_async' } };
  await assert.rejects(
    cache.getOrCreateRenderPipelineAsync(renderDescAsync),
    /Async render validation failure/
  );
  assert.equal(renderAsyncCalls, 1);
  assert.equal(await cache.getOrCreateRenderPipelineAsync(renderDescAsync), null);
  assert.equal(renderAsyncCalls, 1);

  // Async compute failure rejects once, then subsequent async calls resolve to null
  const computeDescAsync = { layout: 'auto', compute: { module: cs, entryPoint: 'cs_fail_async' } };
  await assert.rejects(
    cache.getOrCreateComputePipelineAsync(computeDescAsync),
    /Async compute validation failure/
  );
  assert.equal(computeAsyncCalls, 1);
  assert.equal(await cache.getOrCreateComputePipelineAsync(computeDescAsync), null);
  assert.equal(computeAsyncCalls, 1);

  const stats = cache.getStats();
  assert.equal(stats.quarantinedKeys, 4);
  assert.equal(stats.quarantineHits, 5);
  assert.equal(stats.compileCount, 0);
  assert.equal(stats.totalCompileMs, 0);
  assert.equal(stats.maxCompileMs, 0);

  // Quarantined records sanitize host paths and track failure/hit counts
  const records = cache.getQuarantinedKeys();
  assert.equal(records.length, 4);
  const renderRec = records.find((r) => r.key === renderKeyA);
  assert.ok(renderRec);
  assert.equal(renderRec.kind, 'render');
  assert.equal(renderRec.failureCount, 1);
  assert.equal(renderRec.hitCount, 2);
  assert.doesNotMatch(renderRec.reason, /Users\\dev/);
  assert.match(renderRec.reason, /<path>/);
});

test('W118 empty/null pipeline creation results are quarantined and never cached as valid pipelines', async () => {
  let syncCalls = 0;
  let asyncCalls = 0;
  const device = createMockGpuDevice({
    renderSyncImpl: () => {
      syncCalls += 1;
      return null;
    },
    computeAsyncImpl: async () => {
      asyncCalls += 1;
      return null;
    },
  });

  const cache = new WebGpuPipelineCache({ device });
  const vs = device.createShaderModule({ code: 'vs' });
  const cs = device.createShaderModule({ code: 'cs' });

  const renderDesc = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_null' } };
  assert.equal(cache.getOrCreateRenderPipeline(renderDesc), null);
  assert.equal(cache.getOrCreateRenderPipeline(renderDesc), null);
  assert.equal(syncCalls, 1);
  assert.equal(cache.getStats().renderCount, 0);
  assert.equal(cache.isQuarantined(renderDesc), true);

  const computeDesc = { layout: 'auto', compute: { module: cs, entryPoint: 'cs_null' } };
  assert.equal(await cache.getOrCreateComputePipelineAsync(computeDesc), null);
  assert.equal(await cache.getOrCreateComputePipelineAsync(computeDesc), null);
  assert.equal(asyncCalls, 1);
  assert.equal(cache.getStats().computeCount, 0);
  assert.equal(cache.isQuarantined(computeDesc), true);
  assert.equal(cache.getStats().quarantineHits, 2);
});

test('W118 clearQuarantine, clear, reset, onDeviceLost, and setDevice clear quarantine and allow retries', async () => {
  let shouldFail = true;
  let calls = 0;
  const makeDevice = () =>
    createMockGpuDevice({
      renderSyncImpl: (desc, id) => {
        calls += 1;
        if (shouldFail) {
          throw new Error('compile error');
        }
        return { kind: 'GPURenderPipeline', id, desc };
      },
    });

  const device1 = makeDevice();
  const cache = new WebGpuPipelineCache({ device: device1 });
  const vs = device1.createShaderModule({ code: 'vs' });
  const desc = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_retry' } };

  // 1. Fail -> quarantined -> clearQuarantine() allows retry to succeed
  assert.throws(() => cache.getOrCreateRenderPipeline(desc), /compile error/);
  assert.equal(cache.getOrCreateRenderPipeline(desc), null);
  assert.equal(calls, 1);
  assert.equal(cache.isQuarantined(desc), true);

  assert.equal(cache.clearQuarantine(), 1);
  assert.equal(cache.isQuarantined(desc), false);
  assert.equal(cache.getStats().quarantinedKeys, 0);

  shouldFail = false;
  const ok1 = cache.getOrCreateRenderPipeline(desc);
  assert.equal(ok1.kind, 'GPURenderPipeline');
  assert.equal(calls, 2);

  // 2. clear() clears quarantine
  shouldFail = true;
  const desc2 = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_clear' } };
  assert.throws(() => cache.getOrCreateRenderPipeline(desc2), /compile error/);
  assert.equal(cache.isQuarantined(desc2), true);
  cache.clear();
  assert.equal(cache.isQuarantined(desc2), false);
  shouldFail = false;
  assert.equal(cache.getOrCreateRenderPipeline(desc2).kind, 'GPURenderPipeline');

  // 3. reset() clears quarantine
  shouldFail = true;
  const desc3 = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_reset' } };
  assert.throws(() => cache.getOrCreateRenderPipeline(desc3), /compile error/);
  assert.equal(cache.isQuarantined(desc3), true);
  cache.reset();
  assert.equal(cache.isQuarantined(desc3), false);

  // 4. onDeviceLost() clears quarantine
  const desc4 = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_lost' } };
  assert.throws(() => cache.getOrCreateRenderPipeline(desc4), /compile error/);
  assert.equal(cache.isQuarantined(desc4), true);
  cache.onDeviceLost();
  assert.equal(cache.isQuarantined(desc4), false);

  // 5. setDevice() swaps device, clears quarantine, and compiles on new device
  const desc5 = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_swap' } };
  assert.throws(() => cache.getOrCreateRenderPipeline(desc5), /compile error/);
  assert.equal(cache.isQuarantined(desc5), true);

  shouldFail = false;
  const device2 = makeDevice();
  assert.equal(cache.setDevice(device1), false);
  assert.equal(cache.setDevice(device2), true);
  assert.equal(cache.isQuarantined(desc5), false);
  const okAfterSwap = cache.getOrCreateRenderPipeline(desc5);
  assert.equal(okAfterSwap.kind, 'GPURenderPipeline');
  assert.equal(device2.calls.createRenderPipeline.length, 1);

  // 6. Stale pre-clear async rejection does not poison post-clear quarantine
  const pendingFail = createDeferred();
  const asyncDevice = createMockGpuDevice({
    renderAsyncImpl: async () => {
      await pendingFail.promise;
      throw new Error('stale pre-clear failure');
    },
  });
  const asyncCache = new WebGpuPipelineCache({ device: asyncDevice });
  const asyncDesc = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_stale' } };
  const pStale = asyncCache.getOrCreateRenderPipelineAsync(asyncDesc);
  asyncCache.onDeviceLost();
  pendingFail.resolve();
  await assert.rejects(pStale, /stale pre-clear failure/);
  assert.equal(asyncCache.isQuarantined(asyncDesc), false);
  assert.equal(asyncCache.getStats().quarantinedKeys, 0);
});

test('W118 maxQuarantinedKeys bounds quarantine map and evicts oldest quarantined key', () => {
  let calls = 0;
  const device = createMockGpuDevice({
    renderSyncImpl: () => {
      calls += 1;
      throw new Error('bad descriptor');
    },
  });

  const cache = new WebGpuPipelineCache({ device, maxQuarantinedKeys: 2 });
  const vs = device.createShaderModule({ code: 'vs' });
  const d1 = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_q1' } };
  const d2 = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_q2' } };
  const d3 = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_q3' } };

  assert.throws(() => cache.getOrCreateRenderPipeline(d1), /bad descriptor/);
  assert.throws(() => cache.getOrCreateRenderPipeline(d2), /bad descriptor/);
  assert.equal(cache.getStats().quarantinedKeys, 2);
  assert.equal(cache.isQuarantined(d1), true);
  assert.equal(cache.isQuarantined(d2), true);

  // Adding d3 evicts oldest quarantined key (d1)
  assert.throws(() => cache.getOrCreateRenderPipeline(d3), /bad descriptor/);
  assert.equal(cache.getStats().quarantinedKeys, 2);
  assert.equal(cache.isQuarantined(d1), false);
  assert.equal(cache.isQuarantined(d2), true);
  assert.equal(cache.isQuarantined(d3), true);

  // d2 and d3 still hit quarantine without device calls
  assert.equal(cache.getOrCreateRenderPipeline(d2), null);
  assert.equal(cache.getOrCreateRenderPipeline(d3), null);
  assert.equal(calls, 3);

  // Evicted d1 retries compilation on the device
  assert.throws(() => cache.getOrCreateRenderPipeline(d1), /bad descriptor/);
  assert.equal(calls, 4);
  assert.equal(cache.getStats().quarantinedKeys, 2);
});

test('W118 quarantineFailedKeys: false disables quarantine and retries normally', () => {
  let renderAttempts = 0;
  let computeAttempts = 0;
  const device = createMockGpuDevice({
    renderSyncImpl: (desc, id) => {
      renderAttempts += 1;
      if (renderAttempts < 3) {
        throw new Error('transient render error');
      }
      return { kind: 'GPURenderPipeline', id, desc };
    },
    computeSyncImpl: (desc, id) => {
      computeAttempts += 1;
      if (computeAttempts < 2) {
        throw new Error('transient compute error');
      }
      return { kind: 'GPUComputePipeline', id, desc };
    },
  });

  const cache = new WebGpuPipelineCache({ device, quarantineFailedKeys: false });
  const vs = device.createShaderModule({ code: 'vs' });
  const cs = device.createShaderModule({ code: 'cs' });
  const renderDesc = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_no_q' } };
  const computeDesc = { layout: 'auto', compute: { module: cs, entryPoint: 'cs_no_q' } };

  assert.throws(() => cache.getOrCreateRenderPipeline(renderDesc), /transient render error/);
  assert.equal(cache.isQuarantined(renderDesc), false);
  assert.throws(() => cache.getOrCreateRenderPipeline(renderDesc), /transient render error/);
  const rp = cache.getOrCreateRenderPipeline(renderDesc);
  assert.equal(rp.kind, 'GPURenderPipeline');
  assert.equal(renderAttempts, 3);

  assert.throws(() => cache.getOrCreateComputePipeline(computeDesc), /transient compute error/);
  assert.equal(cache.isQuarantined(computeDesc), false);
  const cp = cache.getOrCreateComputePipeline(computeDesc);
  assert.equal(cp.kind, 'GPUComputePipeline');
  assert.equal(computeAttempts, 2);

  assert.equal(cache.getStats().quarantinedKeys, 0);
  assert.equal(cache.getStats().quarantineHits, 0);
});

test('W118 compile-latency telemetry tracks compileCount, totalCompileMs, and maxCompileMs via injectable clock', async () => {
  let clock = 500;
  const renderDurations = [3.5, 18.25];
  const computeDurations = [7.0, 12.5];

  const device = createMockGpuDevice({
    renderSyncImpl: (desc, id) => {
      clock += renderDurations.shift();
      return { kind: 'GPURenderPipeline', id, desc };
    },
    renderAsyncImpl: async (desc, id) => {
      clock += renderDurations.shift();
      return { kind: 'GPURenderPipeline', id, desc };
    },
    computeSyncImpl: (desc, id) => {
      clock += computeDurations.shift();
      return { kind: 'GPUComputePipeline', id, desc };
    },
    computeAsyncImpl: async (desc, id) => {
      clock += computeDurations.shift();
      return { kind: 'GPUComputePipeline', id, desc };
    },
  });

  const cache = new WebGpuPipelineCache({ device, nowFn: () => clock });
  const vs = device.createShaderModule({ code: 'vs' });
  const cs = device.createShaderModule({ code: 'cs' });

  // 1. Sync render compile: 3.5ms
  const r1 = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_t1' } };
  cache.getOrCreateRenderPipeline(r1);
  // Cache hit should NOT increment compileCount or change timing
  cache.getOrCreateRenderPipeline(r1);

  assert.equal(cache.getStats().compileCount, 1);
  assert.equal(cache.getStats().totalCompileMs, 3.5);
  assert.equal(cache.getStats().maxCompileMs, 3.5);

  // 2. Async render compile: 18.25ms (new max)
  await cache.getOrCreateRenderPipelineAsync({
    layout: 'auto',
    vertex: { module: vs, entryPoint: 'vs_t2' },
  });
  assert.equal(cache.getStats().compileCount, 2);
  assert.equal(cache.getStats().totalCompileMs, 21.75);
  assert.equal(cache.getStats().maxCompileMs, 18.25);

  // 3. Sync compute compile: 7.0ms (max stays 18.25)
  cache.getOrCreateComputePipeline({
    layout: 'auto',
    compute: { module: cs, entryPoint: 'cs_t1' },
  });
  assert.equal(cache.getStats().compileCount, 3);
  assert.equal(cache.getStats().totalCompileMs, 28.75);
  assert.equal(cache.getStats().maxCompileMs, 18.25);

  // 4. Async compute compile: 12.5ms
  await cache.getOrCreateComputePipelineAsync({
    layout: 'auto',
    compute: { module: cs, entryPoint: 'cs_t2' },
  });
  assert.equal(cache.getStats().compileCount, 4);
  assert.equal(cache.getStats().totalCompileMs, 41.25);
  assert.equal(cache.getStats().maxCompileMs, 18.25);

  // Reset stats clears compile timing telemetry
  cache.clear({ resetStats: true });
  assert.equal(cache.getStats().compileCount, 0);
  assert.equal(cache.getStats().totalCompileMs, 0);
  assert.equal(cache.getStats().maxCompileMs, 0);

  // Constructor option validation checks
  assert.throws(
    () => new WebGpuPipelineCache({ device, quarantineFailedKeys: 'yes' }),
    /quarantineFailedKeys must be a boolean/
  );
  assert.throws(
    () => new WebGpuPipelineCache({ device, maxQuarantinedKeys: 0 }),
    /maxQuarantinedKeys must be a positive integer/
  );
  assert.throws(
    () => new WebGpuPipelineCache({ device, nowFn: 'not-a-fn' }),
    /nowFn must be a function/
  );
});

// --- W167: Static-Sampler Fallback Synthesis & Menu Blend/sRGB Cache Hardening Tests ---

test('W167 normalizeDefaultSamplerKind, buildDefaultSamplerDescriptor, and IIFE isolation', () => {
  assert.equal(normalizeDefaultSamplerKind('filtering'), 'filtering');
  assert.equal(normalizeDefaultSamplerKind('linear'), 'filtering');
  assert.equal(normalizeDefaultSamplerKind('linear-clamp'), 'filtering');
  assert.equal(normalizeDefaultSamplerKind('default'), 'filtering');
  assert.equal(normalizeDefaultSamplerKind('non-filtering'), 'non-filtering');
  assert.equal(normalizeDefaultSamplerKind('non_filtering'), 'non-filtering');
  assert.equal(normalizeDefaultSamplerKind('point'), 'non-filtering');
  assert.equal(normalizeDefaultSamplerKind('nearest-clamp'), 'non-filtering');
  assert.equal(normalizeDefaultSamplerKind('comparison'), 'comparison');
  assert.equal(normalizeDefaultSamplerKind('compare'), 'comparison');
  assert.equal(normalizeDefaultSamplerKind('less-equal'), 'comparison');
  assert.equal(normalizeDefaultSamplerKind('shadow'), 'comparison');
  assert.equal(normalizeDefaultSamplerKind('unknown-kind'), 'filtering');
  assert.equal(normalizeDefaultSamplerKind(null), 'filtering');
  assert.equal(normalizeDefaultSamplerKind(undefined), 'filtering');

  const filteringDesc = buildDefaultSamplerDescriptor('filtering');
  assert.deepEqual(filteringDesc, {
    label: 'fallback-sampler:filtering',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
    addressModeW: 'clamp-to-edge',
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
  });

  const nonFilteringDesc = buildDefaultSamplerDescriptor('non-filtering');
  assert.deepEqual(nonFilteringDesc, {
    label: 'fallback-sampler:non-filtering',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
    addressModeW: 'clamp-to-edge',
    magFilter: 'nearest',
    minFilter: 'nearest',
    mipmapFilter: 'nearest',
  });

  const comparisonDesc = buildDefaultSamplerDescriptor('comparison');
  assert.deepEqual(comparisonDesc, {
    label: 'fallback-sampler:comparison',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
    addressModeW: 'clamp-to-edge',
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'nearest',
    compare: 'less-equal',
  });

  // Mutating returned descriptor does not affect frozen DEFAULT_SAMPLER_DESCRIPTORS
  filteringDesc.magFilter = 'nearest';
  assert.equal(DEFAULT_SAMPLER_DESCRIPTORS.filtering.magFilter, 'linear');
  assert.equal(buildDefaultSamplerDescriptor('filtering').magFilter, 'linear');

  // sRGB format detection helper
  assert.equal(isSrgbColorFormat('rgba8unorm-srgb'), true);
  assert.equal(isSrgbColorFormat('bgra8unorm-srgb'), true);
  assert.equal(isSrgbColorFormat('RGBA8UNORM-SRGB'), true);
  assert.equal(isSrgbColorFormat('rgba8unorm'), false);
  assert.equal(isSrgbColorFormat('bgra8unorm'), false);
  assert.equal(isSrgbColorFormat(null), false);

  // Classic worker IIFE isolation verifies W167 helpers do not leak globals
  const scriptPath = path.resolve(
    path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')),
    '../web/workers/pipeline_cache.js'
  );
  const source = fs.readFileSync(scriptPath, 'utf8');
  const sandbox = { TextEncoder, performance, Date };
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox);
  assert.equal(typeof sandbox.PlayGta6PipelineCache.normalizeDefaultSamplerKind, 'function');
  assert.equal(typeof sandbox.PlayGta6PipelineCache.buildDefaultSamplerDescriptor, 'function');
  assert.equal(typeof sandbox.PlayGta6PipelineCache.isSrgbColorFormat, 'function');
  assert.equal(sandbox.DEFAULT_SAMPLER_DESCRIPTORS, undefined);
  assert.equal(sandbox.normalizeDefaultSamplerKind, undefined);
});

test('W167 getOrCreateDefaultSampler caches per kind, invalidates on device reset, and handles errors', () => {
  const device1 = createMockGpuDevice();
  const cache = new WebGpuPipelineCache({ device: device1 });

  const sLinear1 = cache.getOrCreateDefaultSampler('filtering');
  const sLinear2 = cache.getOrCreateDefaultSampler('linear-clamp');
  const sLinear3 = cache.getOrCreateDefaultSampler();
  assert.equal(sLinear1, sLinear2);
  assert.equal(sLinear1, sLinear3);
  assert.equal(sLinear1.desc.magFilter, 'linear');
  assert.equal(sLinear1.desc.minFilter, 'linear');
  assert.equal(sLinear1.desc.addressModeU, 'clamp-to-edge');

  const sPoint1 = cache.getOrCreateDefaultSampler('non-filtering');
  const sPoint2 = cache.getOrCreateDefaultSampler('point');
  assert.equal(sPoint1, sPoint2);
  assert.notEqual(sPoint1, sLinear1);
  assert.equal(sPoint1.desc.magFilter, 'nearest');
  assert.equal(sPoint1.desc.minFilter, 'nearest');

  const sCmp1 = cache.getOrCreateDefaultSampler('comparison');
  const sCmp2 = cache.getOrCreateDefaultSampler('less-equal');
  assert.equal(sCmp1, sCmp2);
  assert.notEqual(sCmp1, sLinear1);
  assert.equal(sCmp1.desc.compare, 'less-equal');

  assert.equal(device1.calls.createSampler.length, 3);

  // Deterministic hashing via registeredObjectIds
  assert.equal(
    hashDescriptorKey({ sampler: sLinear1 }),
    hashDescriptorKey({ sampler: sLinear2 })
  );
  assert.notEqual(
    hashDescriptorKey({ sampler: sLinear1 }),
    hashDescriptorKey({ sampler: sPoint1 })
  );
  assert.notEqual(
    hashDescriptorKey({ sampler: sLinear1 }),
    hashDescriptorKey({ sampler: sCmp1 })
  );

  const stats = cache.getStats();
  assert.equal(stats.fallbackSamplersCreated, 3);
  assert.equal(stats.fallbackSamplerCount, 3);
  assert.equal(stats.samplerMisses, 3);
  assert.equal(stats.samplerHits, 4);
  assert.equal(stats.cacheBreakdown.sampler.hits, 4);
  assert.equal(stats.cacheBreakdown.sampler.misses, 3);

  // Device swap clears cached samplers and creates fresh ones on new device
  const device2 = createMockGpuDevice();
  assert.equal(cache.setDevice(device2), true);
  assert.equal(cache.getStats().fallbackSamplerCount, 0);
  const sLinearNew = cache.getOrCreateDefaultSampler('filtering');
  assert.notEqual(sLinearNew, sLinear1);
  assert.equal(device2.calls.createSampler.length, 1);
  assert.equal(cache.getStats().fallbackSamplersCreated, 4);

  // Missing createSampler throws TypeError cleanly
  const noSamplerCache = new WebGpuPipelineCache({
    device: createMockGpuDevice({ withSampler: false }),
  });
  assert.throws(
    () => noSamplerCache.getOrCreateDefaultSampler('filtering'),
    /device must provide a createSampler/
  );

  // Failing / empty createSampler records sanitized pipeline error
  const failSamplerCache = new WebGpuPipelineCache({
    device: createMockGpuDevice({
      samplerImpl: (desc) => {
        if (desc.compare) {
          return null;
        }
        const err = new Error('Sampler failed at C:\\Users\\dev\\menu.wgsl');
        err.name = 'GPUValidationError';
        throw err;
      },
    }),
  });
  assert.throws(() => failSamplerCache.getOrCreateDefaultSampler('filtering'), /Sampler failed/);
  assert.throws(() => failSamplerCache.getOrCreateDefaultSampler('comparison'), /empty sampler/);
  assert.equal(failSamplerCache.getStats().fallbackSamplersCreated, 0);
  assert.equal(failSamplerCache.getStats().pipelineErrorCount, 2);
  const errs = failSamplerCache.getPipelineErrors();
  assert.equal(errs[0].kind, 'sampler');
  assert.equal(errs[0].errorType, 'validation');
  assert.doesNotMatch(errs[0].message, /Users\\dev/);
  assert.match(errs[0].message, /<path>/);
});

test('W167 synthesizeFallbackSamplerEntries fills missing sampler slots without mutating caller entries', () => {
  const device = createMockGpuDevice();
  const cache = new WebGpuPipelineCache({ device });

  // Reflection group with a texture + 3 sampler slots (linear filtering, nearest non-filtering, comparison)
  const reflectionGroup = {
    group: 0,
    entries: [
      {
        binding: 0,
        visibility: 'fragment',
        texture: { sampleType: 'float', viewDimension: '2d', multisampled: false },
      },
      {
        binding: 1,
        visibility: 'fragment',
        sampler: {}, // omitted type -> defaults to 'filtering'
      },
      {
        binding: 2,
        visibility: 'fragment',
        sampler: { type: 'non-filtering' },
      },
      {
        binding: 3,
        visibility: 'fragment',
        sampler: { type: 'comparison' },
      },
      {
        binding: 4,
        visibility: 'fragment',
        sampler: { type: 'filtering' },
      },
    ],
  };

  const mockTexView = { kind: 'GPUTextureView', id: 100 };
  const explicitStaticSampler = { kind: 'GPUSampler', id: 200, customStatic: true };
  const existingEntries = [
    { binding: 0, resource: mockTexView },
    { binding: 1, resource: null }, // null placeholder -> treated as missing & synthesized
    { binding: 4, resource: explicitStaticSampler }, // explicit static sampler -> preserved
  ];
  const existingSnapshot = JSON.stringify(existingEntries);

  const result = cache.synthesizeFallbackSamplerEntries(reflectionGroup, existingEntries);

  // Caller's array must not be mutated
  assert.equal(JSON.stringify(existingEntries), existingSnapshot);

  assert.equal(result.synthesizedCount, 3);
  assert.equal(result.entries.length, 5);
  assert.deepEqual(
    result.entries.map((e) => e.binding),
    [0, 1, 2, 3, 4]
  );
  assert.equal(result.entries[0].resource, mockTexView);
  assert.equal(result.entries[1].resource.desc.magFilter, 'linear');
  assert.equal(result.entries[2].resource.desc.magFilter, 'nearest');
  assert.equal(result.entries[3].resource.desc.compare, 'less-equal');
  assert.equal(result.entries[4].resource, explicitStaticSampler);

  assert.deepEqual(
    result.synthesized.map((s) => ({ binding: s.binding, kind: s.kind })),
    [
      { binding: 1, kind: 'filtering' },
      { binding: 2, kind: 'non-filtering' },
      { binding: 3, kind: 'comparison' },
    ]
  );

  // Second synthesis reuses cached fallback samplers without creating new device samplers
  const result2 = cache.synthesizeFallbackSamplerEntries(reflectionGroup.entries, [
    { binding: 0, resource: mockTexView },
  ]);
  assert.equal(result2.synthesizedCount, 4);
  assert.equal(result2.entries[1].resource, result.entries[1].resource);
  assert.equal(result2.entries[4].resource, result.entries[1].resource);
  assert.equal(device.calls.createSampler.length, 3);

  const stats = cache.getStats();
  assert.equal(stats.fallbackSamplersCreated, 3);
  assert.equal(stats.fallbackSamplerBindingsSynthesized, 7);
  assert.equal(stats.samplerMisses, 3);
  assert.equal(stats.samplerHits, 4);

  // Reflection layout creation normalizes sampler type aliases cleanly
  const bglA = cache.createBindGroupLayoutFromReflection([
    { binding: 0, visibility: 'fragment', sampler: { type: 'linear' } },
    { binding: 1, visibility: 'fragment', sampler: { type: 'point' } },
    { binding: 2, visibility: 'fragment', sampler: { type: 'compare' } },
  ]);
  const bglB = cache.createBindGroupLayoutFromReflection([
    { binding: 0, visibility: 'fragment', sampler: { type: 'filtering' } },
    { binding: 1, visibility: 'fragment', sampler: { type: 'non-filtering' } },
    { binding: 2, visibility: 'fragment', sampler: { type: 'comparison' } },
  ]);
  assert.equal(bglA, bglB);
  assert.equal(device.calls.createBindGroupLayout.length, 1);
});

test('W167 menu sRGB/linear color formats, alpha-blend states, writeMask, depthStencil, and vertex layouts never collide', async () => {
  const device = createMockGpuDevice();
  const cache = new WebGpuPipelineCache({ device });

  const vs = device.createShaderModule({ label: 'wgpk:vertex:0x000000000000a001', code: 'vs' });
  const fs = device.createShaderModule({ label: 'wgpk:fragment:0x000000000000a002', code: 'fs' });

  const alphaBlend = {
    color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  };
  const premultipliedBlend = {
    color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  };
  const additiveBlend = {
    color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
  };
  const subtractBlend = {
    color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'subtract' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  };

  const vbl24 = [
    {
      arrayStride: 24,
      stepMode: 'vertex',
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x4' },
        { shaderLocation: 1, offset: 16, format: 'float32x2' },
      ],
    },
  ];
  const vbl32UiGlyph = [
    {
      arrayStride: 32,
      stepMode: 'vertex',
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x4' },
        { shaderLocation: 1, offset: 16, format: 'float32x2' },
        { shaderLocation: 2, offset: 24, format: 'unorm8x4' },
      ],
    },
  ];
  const vbl24Instanced = [
    {
      arrayStride: 24,
      stepMode: 'instance',
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x4' },
        { shaderLocation: 1, offset: 16, format: 'float32x2' },
      ],
    },
  ];

  const makeDesc = ({ targets, depthStencil, buffers = vbl24 }) => ({
    layout: 'auto',
    vertex: { module: vs, entryPoint: 'vs_main', buffers },
    fragment: { module: fs, entryPoint: 'fs_main', targets },
    ...(depthStencil !== undefined ? { depthStencil } : {}),
  });

  const variantDescriptors = [
    // 1. Opaque intro video quads across linear/sRGB/BGRA formats
    makeDesc({ targets: [{ format: 'rgba8unorm' }] }),
    makeDesc({ targets: [{ format: 'rgba8unorm-srgb' }] }),
    makeDesc({ targets: [{ format: 'bgra8unorm' }] }),
    makeDesc({ targets: [{ format: 'bgra8unorm-srgb' }] }),

    // 2. Alpha-blended UI text/glyphs vs premultiplied vs additive vs subtract (same bgra8unorm format)
    makeDesc({ targets: [{ format: 'bgra8unorm', blend: alphaBlend }] }),
    makeDesc({ targets: [{ format: 'bgra8unorm', blend: premultipliedBlend }] }),
    makeDesc({ targets: [{ format: 'bgra8unorm', blend: additiveBlend }] }),
    makeDesc({ targets: [{ format: 'bgra8unorm', blend: subtractBlend }] }),

    // 3. sRGB + alpha blend UI target
    makeDesc({ targets: [{ format: 'rgba8unorm-srgb', blend: alphaBlend }] }),

    // 4. writeMask variants (depth-only 0x0, RGB-only 0x7, alpha-only 0x8, RGBA 0xf)
    makeDesc({ targets: [{ format: 'bgra8unorm', writeMask: 0x0 }] }),
    makeDesc({ targets: [{ format: 'bgra8unorm', writeMask: 0x7 }] }),
    makeDesc({ targets: [{ format: 'bgra8unorm', writeMask: 0x8 }] }),
    makeDesc({ targets: [{ format: 'bgra8unorm', writeMask: 0xf }] }),

    // 5. depthStencil variants (depth write on/off, compare op, format, stencil)
    makeDesc({
      targets: [{ format: 'bgra8unorm' }],
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    }),
    makeDesc({
      targets: [{ format: 'bgra8unorm' }],
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' },
    }),
    makeDesc({
      targets: [{ format: 'bgra8unorm' }],
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' },
    }),
    makeDesc({
      targets: [{ format: 'bgra8unorm' }],
      depthStencil: {
        format: 'depth24plus-stencil8',
        depthWriteEnabled: false,
        depthCompare: 'less-equal',
        stencilFront: { compare: 'equal', passOp: 'keep' },
      },
    }),

    // 6. Vertex buffer layout variants (fullscreen [], 32B UI glyph, instanced 24B)
    makeDesc({ targets: [{ format: 'bgra8unorm' }], buffers: [] }),
    makeDesc({ targets: [{ format: 'bgra8unorm' }], buffers: vbl32UiGlyph }),
    makeDesc({ targets: [{ format: 'bgra8unorm' }], buffers: vbl24Instanced }),

    // 7. MRT order and sparse null target sensitivity
    makeDesc({ targets: [{ format: 'rgba8unorm' }, { format: 'rgba8unorm-srgb' }] }),
    makeDesc({ targets: [{ format: 'rgba8unorm-srgb' }, { format: 'rgba8unorm' }] }),
    makeDesc({ targets: [null, { format: 'rgba8unorm' }] }),
  ];

  const seenKeys = new Set();
  const seenPipelines = new Set();

  for (let i = 0; i < variantDescriptors.length; i += 1) {
    const desc = variantDescriptors[i];
    const key = hashDescriptorKey(desc);
    assert.equal(seenKeys.has(key), false, `variant index ${i} collided on cache key ${key}`);
    seenKeys.add(key);

    const pipeline =
      i % 2 === 0
        ? cache.getOrCreateRenderPipeline(desc)
        : await cache.getOrCreateRenderPipelineAsync(desc);
    assert.equal(seenPipelines.has(pipeline), false, `variant index ${i} returned colliding pipeline`);
    seenPipelines.add(pipeline);
  }

  assert.equal(seenPipelines.size, variantDescriptors.length);

  // Re-querying every variant hits the cache with zero new compiles
  for (let i = 0; i < variantDescriptors.length; i += 1) {
    const hit = cache.getOrCreateRenderPipeline(variantDescriptors[i]);
    assert.equal(seenPipelines.has(hit), true);
  }

  const stats = cache.getStats();
  assert.equal(stats.renderCount, variantDescriptors.length);
  assert.equal(stats.renderMisses, variantDescriptors.length);
  assert.equal(stats.renderHits, variantDescriptors.length);
  // 5 blended variants: alphaBlend, premultipliedBlend, additiveBlend, subtractBlend, sRGB+alphaBlend
  assert.equal(stats.blendPipelineVariants, 5);
  // 5 sRGB variants: rgba8unorm-srgb, bgra8unorm-srgb, rgba8unorm-srgb+alphaBlend, and 2 MRT pairs
  assert.equal(stats.srgbPipelineVariants, 5);
  assert.equal(stats.cacheBreakdown.render.hits, variantDescriptors.length);
  assert.equal(stats.cacheBreakdown.render.misses, variantDescriptors.length);
  assert.equal(stats.cacheBreakdown.render.hitRate, 0.5);
});

test('W167 color target blend/sRGB/writeMask/depthStencil normalization deduplicates safely without mutating caller descriptors', () => {
  const device = createMockGpuDevice();
  const cache = new WebGpuPipelineCache({ device, defaultColorFormat: 'bgra8unorm' });
  const vs = device.createShaderModule({ label: 'wgpk:vertex:0x000000000000b001', code: 'vs' });
  const fs = device.createShaderModule({ label: 'wgpk:fragment:0x000000000000b002', code: 'fs' });

  // 1. blend: null and depthStencil: null normalize to omitted and share key with clean opaque descriptor
  const cleanOpaque = {
    layout: 'auto',
    vertex: { module: vs, entryPoint: 'vs_main' },
    fragment: { module: fs, entryPoint: 'fs_main', targets: [{ format: 'rgba8unorm-srgb' }] },
  };
  const nullableOpaque = {
    layout: 'auto',
    vertex: { module: vs, entryPoint: 'vs_main' },
    fragment: {
      module: fs,
      entryPoint: 'fs_main',
      targets: [{ format: '  RGBA8UNORM-SRGB ', blend: null }],
    },
    depthStencil: null,
  };

  const pClean = cache.getOrCreateRenderPipeline(cleanOpaque);
  const pFromNullable = cache.getOrCreateRenderPipeline(nullableOpaque);
  assert.equal(pFromNullable, pClean);
  assert.equal(device.calls.createRenderPipeline.length, 1);
  // Caller descriptor untouched
  assert.equal(nullableOpaque.depthStencil, null);
  assert.equal(nullableOpaque.fragment.targets[0].blend, null);
  assert.equal(nullableOpaque.fragment.targets[0].format, '  RGBA8UNORM-SRGB ');

  // 2. Partial blend component (missing operation: 'add') normalizes and deduplicates with explicit blend
  const explicitBlendDesc = {
    layout: 'auto',
    vertex: { module: vs, entryPoint: 'vs_main' },
    fragment: {
      module: fs,
      entryPoint: 'fs_main',
      targets: [
        {
          format: 'bgra8unorm',
          blend: {
            color: { operation: 'add', srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
            alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          },
        },
      ],
    },
  };
  const partialBlendDesc = {
    layout: 'auto',
    vertex: { module: vs, entryPoint: 'vs_main' },
    fragment: {
      module: fs,
      entryPoint: 'fs_main',
      targets: [
        {
          format: 'bgra8unorm',
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          },
        },
      ],
    },
  };
  const pBlend1 = cache.getOrCreateRenderPipeline(explicitBlendDesc);
  const pBlend2 = cache.getOrCreateRenderPipeline(partialBlendDesc);
  assert.equal(pBlend2, pBlend1);
  assert.equal(device.calls.createRenderPipeline.length, 2);

  // 3. Out-of-range writeMask, missing target format, and min/max blend defaults normalize safely
  const norm = normalizeRenderPipelineDescriptor(
    {
      layout: 'auto',
      vertex: { module: vs, entryPoint: 'vs_main' },
      fragment: {
        module: fs,
        entryPoint: 'fs_main',
        targets: [
          {
            writeMask: 0x37, // clamps to 0x7
            blend: {
              color: { operation: 'min' }, // defaults srcFactor='one', dstFactor='one'
            }, // missing alpha -> synthesized default
          },
        ],
      },
    },
    { defaultColorFormat: 'rgba8unorm' }
  );
  assert.equal(norm.normalized, true);
  assert.equal(norm.descriptor.fragment.targets[0].format, 'rgba8unorm');
  assert.equal(norm.descriptor.fragment.targets[0].writeMask, 0x7);
  assert.deepEqual(norm.descriptor.fragment.targets[0].blend, {
    color: { operation: 'min', srcFactor: 'one', dstFactor: 'one' },
    alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'zero' },
  });
  assert.ok(norm.reasons.includes('default-color-target-format'));
  assert.ok(norm.reasons.includes('clamped-write-mask'));
  assert.ok(norm.reasons.includes('normalized-blend-component'));

  // 4. Failed compilations do not inflate blendPipelineVariants or srgbPipelineVariants
  const failCache = new WebGpuPipelineCache({
    device: createMockGpuDevice({
      renderSyncImpl: () => {
        throw new Error('compile failed');
      },
    }),
  });
  assert.throws(
    () =>
      failCache.getOrCreateRenderPipeline({
        layout: 'auto',
        vertex: { module: vs, entryPoint: 'vs_fail' },
        fragment: {
          module: fs,
          entryPoint: 'fs_fail',
          targets: [
            {
              format: 'rgba8unorm-srgb',
              blend: explicitBlendDesc.fragment.targets[0].blend,
            },
          ],
        },
      }),
    /compile failed/
  );
  assert.equal(failCache.getStats().blendPipelineVariants, 0);
  assert.equal(failCache.getStats().srgbPipelineVariants, 0);
});

// --- W195: Fast Pipeline Lookup Keys, Bounded Negative Cache & Telemetry Tests ---

test('W195 32-bit-limb fnv1a64Hex matches reference BigInt 64-bit FNV-1a across ASCII and UTF-8', () => {
  const encoder = new TextEncoder();
  const refFnv1a64 = (input) => {
    const bytes = encoder.encode(input);
    let h = 0xcbf29ce484222325n;
    const prime = 0x00000100000001b3n;
    const mask = 0xffffffffffffffffn;
    for (let i = 0; i < bytes.length; i += 1) {
      h ^= BigInt(bytes[i]);
      h = (h * prime) & mask;
    }
    return `0x${h.toString(16).padStart(16, '0')}`;
  };

  // Standard FNV-1a 64-bit test vectors + UTF-8 multi-byte & surrogate pairs
  assert.equal(fnv1a64Hex(''), '0xcbf29ce484222325');
  assert.equal(fnv1a64Hex('a'), '0xaf63dc4c8601ec8c');
  assert.equal(fnv1a64Hex('foobar'), '0x85944171f73967e8');

  const samples = [
    '',
    'a',
    'foobar',
    'r|l:4:auto|v:{m:28:gpu:wgpk:0x0000000000000001|ep:7:vs_main}',
    'caf\u00e9', // 2-byte UTF-8
    '\u2603\u4e16\u754c', // 3-byte UTF-8
    '\ud83d\ude80\ud83c\udfae', // 4-byte UTF-8 surrogate pairs
    'a'.repeat(513),
  ];
  for (const s of samples) {
    assert.equal(fnv1a64Hex(s), refFnv1a64(s), `mismatch on input ${JSON.stringify(s)}`);
  }
});

test('W195 fast key builder is deterministic, key-order invariant, JSON.stringify-free, and classic-Worker safe', () => {
  const device = createMockGpuDevice();
  const cache = new WebGpuPipelineCache({ device });
  const vs = device.createShaderModule({ label: 'wgpk:vertex:0x000000000000f001', code: 'vs' });
  const fsMod = device.createShaderModule({ label: 'wgpk:fragment:0x000000000000f002', code: 'fs' });
  const cs = device.createShaderModule({ label: 'wgpk:compute:0x000000000000f003', code: 'cs' });

  const renderA = {
    layout: 'auto',
    vertex: {
      module: vs,
      entryPoint: 'vs_main',
      constants: { b: 2, a: 1 },
      buffers: [
        {
          arrayStride: 24,
          stepMode: 'vertex',
          attributes: [
            { format: 'float32x4', offset: 0, shaderLocation: 0 },
            { format: 'float32x2', offset: 16, shaderLocation: 1 },
          ],
        },
      ],
    },
    fragment: {
      module: fsMod,
      entryPoint: 'fs_main',
      targets: [
        {
          format: 'bgra8unorm',
          writeMask: 0xf,
          blend: {
            color: { operation: 'add', srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
            alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          },
        },
      ],
    },
    primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw', unclippedDepth: false },
    depthStencil: {
      format: 'depth24plus-stencil8',
      depthWriteEnabled: true,
      depthCompare: 'less-equal',
      stencilFront: { compare: 'always', failOp: 'keep', depthFailOp: 'keep', passOp: 'replace' },
      stencilBack: { compare: 'equal', failOp: 'zero', depthFailOp: 'keep', passOp: 'keep' },
      stencilReadMask: 0xff,
      stencilWriteMask: 0xff,
      depthBias: 0,
      depthBiasSlopeScale: 0,
      depthBiasClamp: 0,
    },
    multisample: { count: 4, mask: 0xffffffff, alphaToCoverageEnabled: false },
    label: undefined,
  };

  // Same descriptor with reversed key insertion order at every nesting level
  const renderB = {
    multisample: { alphaToCoverageEnabled: false, mask: 0xffffffff, count: 4 },
    depthStencil: {
      depthBiasClamp: -0,
      depthBiasSlopeScale: 0,
      depthBias: -0,
      stencilWriteMask: 0xff,
      stencilReadMask: 0xff,
      stencilBack: { passOp: 'keep', depthFailOp: 'keep', failOp: 'zero', compare: 'equal' },
      stencilFront: { passOp: 'replace', depthFailOp: 'keep', failOp: 'keep', compare: 'always' },
      depthCompare: 'less-equal',
      depthWriteEnabled: true,
      format: 'depth24plus-stencil8',
    },
    primitive: { unclippedDepth: false, frontFace: 'ccw', cullMode: 'back', topology: 'triangle-list' },
    fragment: {
      targets: [
        {
          blend: {
            alpha: { dstFactor: 'one-minus-src-alpha', srcFactor: 'one', operation: 'add' },
            color: { dstFactor: 'one-minus-src-alpha', srcFactor: 'src-alpha', operation: 'add' },
          },
          writeMask: 0xf,
          format: 'bgra8unorm',
        },
      ],
      entryPoint: 'fs_main',
      module: fsMod,
    },
    vertex: {
      buffers: [
        {
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x4' },
            { shaderLocation: 1, offset: 16, format: 'float32x2' },
          ],
          stepMode: 'vertex',
          arrayStride: 24,
        },
      ],
      constants: { a: 1, b: 2 },
      entryPoint: 'vs_main',
      module: vs,
    },
    layout: 'auto',
  };

  const computeA = { layout: 'auto', compute: { module: cs, entryPoint: 'cs_main', constants: { y: 8, x: 4 } } };
  const computeB = { compute: { constants: { x: 4, y: 8 }, entryPoint: 'cs_main', module: cs }, layout: 'auto' };
  const bglA = { entries: [{ binding: 0, visibility: SHADER_STAGE_BITS.VERTEX, buffer: { type: 'uniform', minBindingSize: 256 } }] };
  const bglB = { entries: [{ buffer: { minBindingSize: 256, type: 'uniform' }, visibility: SHADER_STAGE_BITS.VERTEX, binding: 0 }] };

  // Prove JSON.stringify is NEVER called on the hot key-generation or lookup path
  const origStringify = JSON.stringify;
  JSON.stringify = () => {
    throw new Error('JSON.stringify must not be called on hot pipeline cache path');
  };
  try {
    const keyA = buildFastPipelineKey(renderA);
    const keyB = buildFastPipelineKey(renderB);
    assert.equal(keyA, keyB);
    assert.ok(keyA.startsWith('r|'));

    const hashA = hashDescriptorKey(renderA);
    const hashB = hashDescriptorKey(renderB);
    assert.equal(hashA, hashB);
    assert.equal(fnv1a64Hex(keyA), hashA);

    assert.equal(buildFastPipelineKey(computeA), buildFastPipelineKey(computeB));
    assert.equal(hashDescriptorKey(computeA), hashDescriptorKey(computeB));
    assert.equal(fnv1a64Hex(buildFastPipelineKey(computeA)), hashDescriptorKey(computeA));

    assert.equal(buildFastPipelineKey(bglA), buildFastPipelineKey(bglB));
    assert.equal(hashDescriptorKey(bglA), hashDescriptorKey(bglB));
    assert.equal(fnv1a64Hex(buildFastPipelineKey(bglA)), hashDescriptorKey(bglA));

    const rp1 = cache.getOrCreateRenderPipeline(renderA);
    const rp2 = cache.getOrCreateRenderPipeline(renderB);
    assert.equal(rp1, rp2);

    const cp1 = cache.getOrCreateComputePipeline(computeA);
    const cp2 = cache.getOrCreateComputePipeline(computeB);
    assert.equal(cp1, cp2);

    const l1 = cache.getOrCreateBindGroupLayout(bglA);
    const l2 = cache.getOrCreateBindGroupLayout(bglB);
    assert.equal(l1, l2);
  } finally {
    JSON.stringify = origStringify;
  }

  // Classic-Worker sandbox without TextEncoder still works with zero global leakage
  const scriptPath = path.resolve(
    path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')),
    '../web/workers/pipeline_cache.js'
  );
  const source = fs.readFileSync(scriptPath, 'utf8');
  const sandbox = { performance, Date };
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox);

  assert.equal(typeof sandbox.PlayGta6PipelineCache.buildFastPipelineKey, 'function');
  assert.equal(typeof sandbox.PlayGta6PipelineCache.fnv1a64Hex, 'function');
  assert.equal(sandbox.PlayGta6PipelineCache.DEFAULT_MAX_NEGATIVE_CACHE_ENTRIES, 512);
  assert.equal(sandbox.buildFastPipelineKey, undefined);
  assert.equal(sandbox.fnv1a64Hex, undefined);
  assert.match(sandbox.PlayGta6PipelineCache.hashDescriptorKey({ a: 1 }), /^0x[0-9a-f]{16}$/);
});

test('W195 distinct render and compute descriptors never collide on fast key or FNV-1a hash', () => {
  const device = createMockGpuDevice();
  const vs1 = device.createShaderModule({ label: 'wgpk:vertex:0x0000000000010001', code: 'vs1' });
  const vs2 = device.createShaderModule({ label: 'wgpk:vertex:0x0000000000010002', code: 'vs2' });
  const fs1 = device.createShaderModule({ label: 'wgpk:fragment:0x0000000000020001', code: 'fs1' });
  const fs2 = device.createShaderModule({ label: 'wgpk:fragment:0x0000000000020002', code: 'fs2' });
  const cs1 = device.createShaderModule({ label: 'wgpk:compute:0x0000000000030001', code: 'cs1' });
  const cs2 = device.createShaderModule({ label: 'wgpk:compute:0x0000000000030002', code: 'cs2' });
  const customLayout = device.createPipelineLayout({ bindGroupLayouts: [] });

  const base = () => ({
    layout: 'auto',
    vertex: {
      module: vs1,
      entryPoint: 'vs_main',
      buffers: [
        {
          arrayStride: 24,
          stepMode: 'vertex',
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x4' },
            { shaderLocation: 1, offset: 16, format: 'float32x2' },
          ],
        },
      ],
    },
    fragment: {
      module: fs1,
      entryPoint: 'fs_main',
      targets: [{ format: 'bgra8unorm', writeMask: 0xf }],
    },
    primitive: { topology: 'triangle-list', frontFace: 'ccw', cullMode: 'none' },
  });

  const make = (mutate) => {
    const d = base();
    mutate(d);
    return d;
  };

  const variants = [
    base(),
    // Shader hash / entryPoint / constants / layout variations
    make((d) => { d.vertex.module = vs2; }),
    make((d) => { d.fragment.module = fs2; }),
    make((d) => { d.vertex.entryPoint = 'vs_alt'; }),
    make((d) => { d.fragment.entryPoint = 'fs_alt'; }),
    make((d) => { d.vertex.entryPoint = 'vs_main|f:{m:x}'; }), // delimiter spoofing attempt
    make((d) => { d.vertex.constants = { mode: 1 }; }),
    make((d) => { d.vertex.constants = { mode: 2 }; }),
    make((d) => { d.vertex.constants = { mode: true }; }),
    make((d) => { d.fragment.constants = { alphaCut: 0.5 }; }),
    make((d) => { d.layout = customLayout; }),
    make((d) => { delete d.layout; }),
    make((d) => { d.label = 'pass_a'; }),
    make((d) => { d.label = 'pass_b'; }),
    // Color targets variations
    make((d) => { d.fragment.targets = [{ format: 'rgba8unorm', writeMask: 0xf }]; }),
    make((d) => { d.fragment.targets = [{ format: 'bgra8unorm-srgb', writeMask: 0xf }]; }),
    make((d) => { d.fragment.targets = [{ format: 'bgra8unorm', writeMask: 0x7 }]; }),
    make((d) => { d.fragment.targets = [{ format: 'bgra8unorm' }]; }), // omitted writeMask vs 0xf
    make((d) => { d.fragment.targets = [{ format: 'bgra8unorm', writeMask: 0xf }, { format: 'rgba16float' }]; }),
    make((d) => { d.fragment.targets = [null, { format: 'bgra8unorm', writeMask: 0xf }]; }),
    make((d) => { d.fragment.targets = [{ format: 'bgra8unorm', writeMask: 0xf }, null]; }),
    make((d) => { delete d.fragment; }), // depth-only pass
    // Blend state variations
    make((d) => {
      d.fragment.targets[0].blend = {
        color: { operation: 'add', srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
        alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
      };
    }),
    make((d) => {
      d.fragment.targets[0].blend = {
        color: { operation: 'subtract', srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
        alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
      };
    }),
    make((d) => {
      d.fragment.targets[0].blend = {
        color: { operation: 'add', srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
        alpha: { operation: 'max', srcFactor: 'one', dstFactor: 'one' },
      };
    }),
    make((d) => { d.fragment.targets[0].blend = null; }),
    // Primitive topology / culling / frontFace / stripIndexFormat / unclippedDepth
    make((d) => { d.primitive.topology = 'triangle-strip'; }),
    make((d) => { d.primitive.topology = 'triangle-strip'; d.primitive.stripIndexFormat = 'uint16'; }),
    make((d) => { d.primitive.topology = 'triangle-strip'; d.primitive.stripIndexFormat = 'uint32'; }),
    make((d) => { d.primitive.topology = 'line-list'; }),
    make((d) => { d.primitive.topology = 'point-list'; }),
    make((d) => { d.primitive.frontFace = 'cw'; }),
    make((d) => { d.primitive.cullMode = 'back'; }),
    make((d) => { d.primitive.cullMode = 'front'; }),
    make((d) => { d.primitive.unclippedDepth = true; }),
    make((d) => { d.primitive.unclippedDepth = false; }),
    // Depth / stencil variations
    make((d) => { d.depthStencil = { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' }; }),
    make((d) => { d.depthStencil = { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' }; }),
    make((d) => { d.depthStencil = { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'greater-equal' }; }),
    make((d) => { d.depthStencil = { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' }; }),
    make((d) => {
      d.depthStencil = {
        format: 'depth24plus-stencil8',
        depthWriteEnabled: true,
        depthCompare: 'less',
        stencilFront: { compare: 'always', failOp: 'keep', depthFailOp: 'keep', passOp: 'replace' },
      };
    }),
    make((d) => {
      d.depthStencil = {
        format: 'depth24plus-stencil8',
        depthWriteEnabled: true,
        depthCompare: 'less',
        stencilBack: { compare: 'always', failOp: 'keep', depthFailOp: 'keep', passOp: 'replace' },
      };
    }),
    make((d) => {
      d.depthStencil = {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
        depthBias: 2,
        depthBiasSlopeScale: 1.5,
      };
    }),
    make((d) => { d.depthStencil = null; }),
    // Vertex layout variations
    make((d) => { d.vertex.buffers = []; }),
    make((d) => { d.vertex.buffers[0].arrayStride = 32; }),
    make((d) => { d.vertex.buffers[0].stepMode = 'instance'; }),
    make((d) => { d.vertex.buffers[0].attributes[1].offset = 12; }),
    make((d) => { d.vertex.buffers[0].attributes[1].format = 'float32x3'; }),
    make((d) => { d.vertex.buffers[0].attributes[1].shaderLocation = 2; }),
    make((d) => { d.vertex.buffers.push({ arrayStride: 16, stepMode: 'instance', attributes: [{ shaderLocation: 2, offset: 0, format: 'float32x4' }] }); }),
    make((d) => { d.vertex.buffers = [null, d.vertex.buffers[0]]; }),
    // Multisample variations
    make((d) => { d.multisample = { count: 4 }; }),
    make((d) => { d.multisample = { count: 4, mask: 0x3 }; }),
    make((d) => { d.multisample = { count: 4, alphaToCoverageEnabled: true }; }),
    // Extra custom non-standard property triggers generic fallback without collision
    make((d) => { d.customExtensionFlag = true; }),
    make((d) => { d.customExtensionFlag = false; }),
    // Compute pipelines
    { layout: 'auto', compute: { module: cs1, entryPoint: 'cs_main' } },
    { layout: 'auto', compute: { module: cs2, entryPoint: 'cs_main' } },
    { layout: 'auto', compute: { module: cs1, entryPoint: 'cs_alt' } },
    { layout: 'auto', compute: { module: cs1, entryPoint: 'cs_main', constants: { WG: 64 } } },
    { layout: customLayout, compute: { module: cs1, entryPoint: 'cs_main' } },
  ];

  const seenFastKeys = new Set();
  const seenHashes = new Set();

  for (let i = 0; i < variants.length; i += 1) {
    const fastKey = buildFastPipelineKey(variants[i]);
    const hash = hashDescriptorKey(variants[i]);
    assert.equal(seenFastKeys.has(fastKey), false, `fast key collision at variant ${i}: ${fastKey}`);
    assert.equal(seenHashes.has(hash), false, `FNV-1a hash collision at variant ${i}: ${hash}`);
    assert.equal(fnv1a64Hex(fastKey), hash, `hash mismatch with fnv1a64Hex(fastKey) at variant ${i}`);
    seenFastKeys.add(fastKey);
    seenHashes.add(hash);
  }
  assert.equal(seenFastKeys.size, variants.length);
  assert.equal(seenHashes.size, variants.length);
});

test('W195 bounded negative caching, shader-pack update invalidation, and stale async rejection isolation', async () => {
  let shouldFail = true;
  let syncRenderCalls = 0;
  let syncComputeCalls = 0;
  let asyncRenderCalls = 0;
  const staleDeferred = createDeferred();

  const device = createMockGpuDevice({
    renderSyncImpl: (desc, id) => {
      syncRenderCalls += 1;
      if (shouldFail) {
        const err = new Error('Broken shader WGSL validation error');
        err.name = 'GPUValidationError';
        throw err;
      }
      return { kind: 'GPURenderPipeline', id, desc };
    },
    computeSyncImpl: (desc, id) => {
      syncComputeCalls += 1;
      if (shouldFail) {
        throw new Error('Broken compute shader');
      }
      return { kind: 'GPUComputePipeline', id, desc };
    },
    renderAsyncImpl: async (desc, id) => {
      asyncRenderCalls += 1;
      await staleDeferred.promise;
      throw new Error('Stale async failure from previous shader pack');
    },
  });

  const cache = new WebGpuPipelineCache({
    device,
    enableNegativeCache: true,
    maxNegativeCacheEntries: 2,
  });

  const vs = device.createShaderModule({ label: 'wgpk:vertex:0x0000000000050001', code: 'vs' });
  const cs = device.createShaderModule({ label: 'wgpk:compute:0x0000000000050002', code: 'cs' });
  const d1 = { layout: 'auto', vertex: { module: vs, entryPoint: 'broken_1' } };
  const d2 = { layout: 'auto', vertex: { module: vs, entryPoint: 'broken_2' } };
  const d3 = { layout: 'auto', compute: { module: cs, entryPoint: 'broken_cs' } };

  // 1. First failure populates negative cache; subsequent frames skip device compilation
  assert.throws(() => cache.getOrCreateRenderPipeline(d1), /Broken shader/);
  assert.equal(cache.isNegativeCached(d1), true);
  for (let frame = 0; frame < 5; frame += 1) {
    assert.equal(cache.getOrCreateRenderPipeline(d1), null);
  }
  assert.equal(syncRenderCalls, 1);
  assert.equal(cache.getStats().negativeHits, 5);
  assert.equal(cache.getStats().negativeCacheSize, 1);

  // 2. Bounded LRU eviction on negative cache (capacity = 2):
  // Touch d1 via negative hit so d2 becomes LRU when d3 is inserted
  assert.throws(() => cache.getOrCreateRenderPipeline(d2), /Broken shader/);
  assert.equal(cache.getOrCreateRenderPipeline(d1), null); // refreshes d1 recency
  assert.throws(() => cache.getOrCreateComputePipeline(d3), /Broken compute/);

  assert.equal(cache.getStats().negativeCacheSize, 2);
  assert.equal(cache.getStats().negativeCacheEvictions, 1);
  assert.equal(cache.getStats().evictions, 0); // positive LRU evictions unaffected
  assert.equal(cache.isNegativeCached(d1), true);
  assert.equal(cache.isNegativeCached(d2), false); // d2 evicted
  assert.equal(cache.isNegativeCached(d3), true);
  assert.equal(cache.getNegativeCacheEntries().length, 2);

  // Also compile a valid pipeline before shader-pack update to verify positive cache survives default update
  shouldFail = false;
  const goodDesc = { layout: 'auto', vertex: { module: vs, entryPoint: 'good_vs' } };
  const goodPipeline = cache.getOrCreateRenderPipeline(goodDesc);
  assert.equal(goodPipeline.kind, 'GPURenderPipeline');
  assert.equal(cache.getStats().renderCount, 1);

  // 3. Start an async compile that will reject AFTER onShaderPackUpdated()
  const staleDesc = { layout: 'auto', vertex: { module: vs, entryPoint: 'stale_async_vs' } };
  const stalePromise = cache.getOrCreateRenderPipelineAsync(staleDesc);
  assert.equal(asyncRenderCalls, 1);

  // 4. Shader-pack update clears negative cache & bumps shaderPackGeneration while preserving good pipelines
  const updateSummary = cache.onShaderPackUpdated();
  assert.deepEqual(updateSummary, {
    clearedNegativeEntries: 2,
    clearedPipelines: 0,
    shaderPackGeneration: 1,
  });
  assert.equal(cache.getStats().negativeCacheSize, 0);
  assert.equal(cache.getStats().shaderPackGeneration, 1);
  assert.equal(cache.isNegativeCached(d1), false);
  assert.equal(cache.isNegativeCached(d3), false);
  assert.equal(cache.getOrCreateRenderPipeline(goodDesc), goodPipeline); // still cached!

  // Stale async rejection settles after pack update -> must NOT poison the new negative cache
  staleDeferred.resolve();
  await assert.rejects(stalePromise, /Stale async failure/);
  assert.equal(cache.isNegativeCached(staleDesc), false);
  assert.equal(cache.getStats().negativeCacheSize, 0);

  // Previously broken descriptors now retry and compile cleanly against the updated shader pack
  const fixedRender = cache.getOrCreateRenderPipeline(d1);
  const fixedCompute = cache.getOrCreateComputePipeline(d3);
  assert.equal(fixedRender.kind, 'GPURenderPipeline');
  assert.equal(fixedCompute.kind, 'GPUComputePipeline');
  assert.equal(syncComputeCalls, 2);

  // 5. onShaderPackUpdated({ clearPipelines: true }) clears both negative and positive pipeline caches
  shouldFail = true;
  const d4 = { layout: 'auto', vertex: { module: vs, entryPoint: 'broken_4' } };
  assert.throws(() => cache.getOrCreateRenderPipeline(d4), /Broken shader/);
  const fullUpdate = cache.onShaderPackUpdated({ clearPipelines: true });
  assert.equal(fullUpdate.clearedNegativeEntries, 1);
  assert.equal(fullUpdate.clearedPipelines, 3); // goodDesc + d1 + d3
  assert.equal(fullUpdate.shaderPackGeneration, 2);
  assert.equal(cache.getStats().renderCount, 0);
  assert.equal(cache.getStats().computeCount, 0);
  assert.equal(cache.getStats().negativeCacheSize, 0);

  // 6. clearNegativeCache() removes entries and returns count; option validation works
  assert.throws(() => cache.getOrCreateRenderPipeline(d4), /Broken shader/);
  assert.equal(cache.clearNegativeCache(), 1);
  assert.equal(cache.isNegativeCached(d4), false);

  assert.throws(
    () => new WebGpuPipelineCache({ device, enableNegativeCache: 'yes' }),
    /enableNegativeCache must be a boolean/
  );
  assert.throws(
    () => new WebGpuPipelineCache({ device, maxNegativeCacheEntries: 0 }),
    /maxNegativeCacheEntries must be a positive integer/
  );
});

test('W195 telemetry exposes hits, misses, negativeHits, evictions, avgCompileMs, and hitRate accurately', () => {
  let clock = 0;
  let failNext = false;
  const compileDurations = [4.0, 8.0, 12.0];

  const device = createMockGpuDevice({
    renderSyncImpl: (desc, id) => {
      if (failNext) {
        throw new Error('invalid render descriptor');
      }
      clock += compileDurations.shift();
      return { kind: 'GPURenderPipeline', id, desc };
    },
  });

  const cache = new WebGpuPipelineCache({
    device,
    maxRenderPipelines: 2,
    now: () => clock,
  });
  const vs = device.createShaderModule({ label: 'wgpk:vertex:0x0000000000060001', code: 'vs' });
  const r1 = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_1' } };
  const r2 = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_2' } };
  const r3 = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_3' } };
  const rBad = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_bad' } };

  // Initial empty stats
  let stats = cache.getStats();
  assert.equal(stats.hits, 0);
  assert.equal(stats.misses, 0);
  assert.equal(stats.negativeHits, 0);
  assert.equal(stats.evictions, 0);
  assert.equal(stats.avgCompileMs, 0);
  assert.equal(stats.hitRate, 0);

  // 3 misses (1 eviction since maxRenderPipelines = 2) + 3 hits -> hitRate = 3 / 6 = 0.5
  cache.getOrCreateRenderPipeline(r1); // miss, 4ms
  cache.getOrCreateRenderPipeline(r1); // hit
  cache.getOrCreateRenderPipeline(r2); // miss, 8ms
  cache.getOrCreateRenderPipeline(r2); // hit
  cache.getOrCreateRenderPipeline(r3); // miss, 12ms -> evicts r1
  cache.getOrCreateRenderPipeline(r3); // hit

  // 1 broken compile miss + 2 negative cache hits
  failNext = true;
  assert.throws(() => cache.getOrCreateRenderPipeline(rBad), /invalid render descriptor/);
  assert.equal(cache.getOrCreateRenderPipeline(rBad), null);
  assert.equal(cache.getOrCreateRenderPipeline(rBad), null);

  stats = cache.getStats();
  assert.equal(stats.hits, 3);
  assert.equal(stats.misses, 4);
  assert.equal(stats.negativeHits, 2);
  assert.equal(stats.evictions, 1);
  assert.equal(stats.compileCount, 3);
  assert.equal(stats.totalCompileMs, 24.0);
  assert.equal(stats.avgCompileMs, 8.0);
  assert.equal(stats.hitRate, 3 / 7);
  assert.equal(stats.cacheBreakdown.total.hitRate, 3 / 7);

  // keyHint fast path hits cache and negative cache cleanly
  const keyR3 = hashDescriptorKey(r3);
  const keyBad = hashDescriptorKey(rBad);
  assert.ok(cache.getOrCreateRenderPipeline(r3, keyR3));
  assert.equal(cache.getOrCreateRenderPipeline(rBad, keyBad), null);
  stats = cache.getStats();
  assert.equal(stats.hits, 4);
  assert.equal(stats.negativeHits, 3);
  assert.equal(stats.hitRate, 4 / 8);

  // clear({ resetStats: true }) resets all telemetry cleanly
  cache.clear({ resetStats: true });
  stats = cache.getStats();
  assert.equal(stats.hits, 0);
  assert.equal(stats.misses, 0);
  assert.equal(stats.negativeHits, 0);
  assert.equal(stats.evictions, 0);
  assert.equal(stats.avgCompileMs, 0);
  assert.equal(stats.hitRate, 0);
});

// --- W246: Pipeline & Bind-Group-Layout Dedup, Inline-Sampler & Compile Telemetry Tests ---

test('W246 tracks pipelineHits, pipelineMisses, clamped hitRate, and resetStats() without evicting warm caches', async () => {
  let clock = 100;
  const device = createMockGpuDevice({
    advanceClockOnCompile: (dt) => {
      clock += dt;
    },
  });
  const cache = new WebGpuPipelineCache({ device, now: () => clock });
  const vs = device.createShaderModule({ label: 'wgpk:vertex:0x0000000000070001', code: 'vs' });
  const cs = device.createShaderModule({ label: 'wgpk:compute:0x0000000000070002', code: 'cs' });

  // Initial zeroed snapshot
  let stats = cache.getStats();
  assert.equal(stats.pipelineHits, 0);
  assert.equal(stats.pipelineMisses, 0);
  assert.equal(stats.hitRate, 0);
  assert.equal(stats.negativeCacheHits, 0);
  assert.equal(stats.layoutDedupHits, 0);
  assert.equal(stats.inlineSamplerLayouts, 0);
  assert.equal(stats.avgCompileMs, 0);
  assert.equal(stats.maxCompileMs, 0);

  const renderDesc = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_main' } };
  const computeDesc = { layout: 'auto', compute: { module: cs, entryPoint: 'cs_main' } };

  // 1 sync render miss + 2 hits, 1 async compute miss + 1 hit
  const rp = cache.getOrCreateRenderPipeline(renderDesc);
  assert.equal(cache.getOrCreateRenderPipeline(renderDesc), rp);
  assert.equal(await cache.getOrCreateRenderPipelineAsync(renderDesc), rp);

  const cp = await cache.getOrCreateComputePipelineAsync(computeDesc);
  assert.equal(cache.getOrCreateComputePipeline(computeDesc), cp);

  // Layout & sampler lookups should NOT inflate pipelineHits/pipelineMisses
  const bgl = cache.getOrCreateBindGroupLayout({
    entries: [{ binding: 0, visibility: SHADER_STAGE_BITS.VERTEX, buffer: { type: 'uniform' } }],
  });
  assert.equal(
    cache.getOrCreateBindGroupLayout({
      entries: [{ buffer: { type: 'uniform' }, visibility: SHADER_STAGE_BITS.VERTEX, binding: 0 }],
    }),
    bgl
  );
  cache.getOrCreateDefaultSampler('filtering');
  cache.getOrCreateDefaultSampler('filtering');

  stats = cache.getStats();
  assert.equal(stats.pipelineHits, 3);
  assert.equal(stats.pipelineMisses, 2);
  assert.equal(stats.renderHits, 2);
  assert.equal(stats.renderMisses, 1);
  assert.equal(stats.computeHits, 1);
  assert.equal(stats.computeMisses, 1);
  assert.equal(stats.layoutDedupHits, 1);
  assert.equal(stats.inlineSamplerLayouts, 0);
  assert.equal(stats.hits, 5);
  assert.equal(stats.misses, 4);
  assert.equal(stats.hitRate, 5 / 9);
  assert.ok(stats.hitRate >= 0 && stats.hitRate <= 1);
  // Compiles: render sync 4.0ms + compute async 3.0ms + bgl 1.5ms = 8.5ms across 3 compiles
  assert.equal(stats.compileCount, 3);
  assert.equal(stats.avgCompileMs, 8.5 / 3);
  assert.equal(stats.maxCompileMs, 4.0);

  // resetStats() clears all telemetry counters while preserving warm cached objects
  cache.resetStats();
  stats = cache.getStats();
  assert.equal(stats.pipelineHits, 0);
  assert.equal(stats.pipelineMisses, 0);
  assert.equal(stats.hits, 0);
  assert.equal(stats.misses, 0);
  assert.equal(stats.hitRate, 0);
  assert.equal(stats.negativeCacheHits, 0);
  assert.equal(stats.negativeHits, 0);
  assert.equal(stats.layoutDedupHits, 0);
  assert.equal(stats.layoutHits, 0);
  assert.equal(stats.inlineSamplerLayouts, 0);
  assert.equal(stats.avgCompileMs, 0);
  assert.equal(stats.maxCompileMs, 0);
  assert.equal(stats.compileCount, 0);
  assert.equal(stats.totalCompileMs, 0);
  // Warm caches remain populated
  assert.equal(stats.renderCount, 1);
  assert.equal(stats.computeCount, 1);
  assert.equal(stats.layoutCount, 1);
  assert.equal(stats.fallbackSamplerCount, 1);

  // Subsequent lookup after resetStats() still hits the warm cache -> hitRate clamps cleanly to 1
  assert.equal(cache.getOrCreateRenderPipeline(renderDesc), rp);
  assert.equal(device.calls.createRenderPipeline.length, 1);
  stats = cache.getStats();
  assert.equal(stats.pipelineHits, 1);
  assert.equal(stats.pipelineMisses, 0);
  assert.equal(stats.hitRate, 1);
});

test('W246 tracks negativeCacheHits across sync, async, keyHint fast path, resetStats, and shader-pack reload', async () => {
  let renderAttempts = 0;
  let computeAttempts = 0;
  const device = createMockGpuDevice({
    renderSyncImpl: () => {
      renderAttempts += 1;
      throw new Error('broken render pso');
    },
    computeAsyncImpl: async () => {
      computeAttempts += 1;
      throw new Error('broken compute pso');
    },
  });
  const cache = new WebGpuPipelineCache({ device });
  const vs = device.createShaderModule({ label: 'wgpk:vertex:0x0000000000071001', code: 'vs' });
  const cs = device.createShaderModule({ label: 'wgpk:compute:0x0000000000071002', code: 'cs' });

  const badRender = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_bad' } };
  const badCompute = { layout: 'auto', compute: { module: cs, entryPoint: 'cs_bad' } };
  const renderKey = hashDescriptorKey(badRender);

  assert.throws(() => cache.getOrCreateRenderPipeline(badRender), /broken render pso/);
  await assert.rejects(cache.getOrCreateComputePipelineAsync(badCompute), /broken compute pso/);
  assert.equal(renderAttempts, 1);
  assert.equal(computeAttempts, 1);

  // 3 negative-cache fast-path hits: sync descriptor, sync keyHint, async compute
  assert.equal(cache.getOrCreateRenderPipeline(badRender), null);
  assert.equal(cache.getOrCreateRenderPipeline(badRender, renderKey), null);
  assert.equal(await cache.getOrCreateComputePipelineAsync(badCompute), null);
  assert.equal(renderAttempts, 1);
  assert.equal(computeAttempts, 1);

  let stats = cache.getStats();
  assert.equal(stats.pipelineMisses, 2);
  assert.equal(stats.pipelineHits, 0);
  assert.equal(stats.negativeCacheHits, 3);
  assert.equal(stats.negativeHits, 3);
  assert.equal(stats.quarantineHits, 3);
  assert.equal(stats.negativeCacheSize, 2);
  assert.equal(stats.hitRate, 0);

  // resetStats() zeros negativeCacheHits while retaining quarantined keys
  cache.resetStats();
  stats = cache.getStats();
  assert.equal(stats.negativeCacheHits, 0);
  assert.equal(stats.negativeHits, 0);
  assert.equal(stats.negativeCacheSize, 2);

  // Subsequent hit increments negativeCacheHits again without invoking device
  assert.equal(cache.getOrCreateRenderPipeline(badRender, renderKey), null);
  assert.equal(renderAttempts, 1);
  assert.equal(cache.getStats().negativeCacheHits, 1);

  // onShaderPackUpdated({ resetStats: true }) clears both negative cache and stats
  cache.onShaderPackUpdated({ resetStats: true });
  stats = cache.getStats();
  assert.equal(stats.negativeCacheHits, 0);
  assert.equal(stats.negativeCacheSize, 0);
  assert.equal(stats.shaderPackGeneration, 1);
});

test('W246 tracks layoutDedupHits and inlineSamplerLayouts across direct BGL, S43 reflection, fallback, and error paths', () => {
  let failNextBgl = false;
  const device = createMockGpuDevice({
    bglImpl: (desc, id) => {
      if (failNextBgl) {
        throw new Error('BGL creation failed');
      }
      return { kind: 'GPUBindGroupLayout', id, desc };
    },
  });
  const cache = new WebGpuPipelineCache({ device });

  // 1. Buffer-only BGL: does NOT increment inlineSamplerLayouts; duplicate increments layoutDedupHits
  const bufBgl1 = cache.getOrCreateBindGroupLayout({
    entries: [{ binding: 0, visibility: SHADER_STAGE_BITS.VERTEX, buffer: { type: 'uniform' } }],
  });
  const bufBgl2 = cache.getOrCreateBindGroupLayout({
    entries: [{ buffer: { type: 'uniform' }, binding: 0, visibility: SHADER_STAGE_BITS.VERTEX }],
  });
  assert.equal(bufBgl1, bufBgl2);
  assert.equal(cache.getStats().inlineSamplerLayouts, 0);
  assert.equal(cache.getStats().layoutDedupHits, 1);

  // 2. Direct BGL with sampler binding: increments inlineSamplerLayouts once; dedup hit does not double-count
  const directSamplerBgl1 = cache.getOrCreateBindGroupLayout({
    entries: [
      { binding: 0, visibility: SHADER_STAGE_BITS.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 1, visibility: SHADER_STAGE_BITS.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  });
  const directSamplerBgl2 = cache.getOrCreateBindGroupLayout({
    entries: [
      { texture: { sampleType: 'float' }, visibility: SHADER_STAGE_BITS.FRAGMENT, binding: 0 },
      { sampler: { type: 'filtering' }, visibility: SHADER_STAGE_BITS.FRAGMENT, binding: 1 },
    ],
  });
  assert.equal(directSamplerBgl1, directSamplerBgl2);
  assert.equal(cache.getStats().inlineSamplerLayouts, 1);
  assert.equal(cache.getStats().layoutDedupHits, 2);

  // 3. S43 / #950 reflection group with inline_sampler source metadata:
  //    strips non-WebGPU metadata, deduplicates reordered/aliased groups, and counts unique layout once
  const s43GroupA = {
    group: 0,
    entries: [
      {
        binding: 3,
        visibility: 'fragment',
        sampler: { type: 'linear' },
        source: { kind: 'inline_sampler', sdwords: [0x11111111, 0x22222222, 0x33333333, 0x44444444] },
      },
      {
        binding: 0,
        visibility: 'fragment',
        texture: { sample_type: 'float', view_dimension: '2d' },
      },
    ],
  };
  const s43GroupB = [
    {
      binding: 0,
      visibility: SHADER_STAGE_BITS.FRAGMENT,
      texture: { sampleType: 'float', viewDimension: '2d' },
    },
    {
      binding: 3,
      visibility: ['fs'],
      sampler: { type: 'filtering' },
      source: { inline_sampler: { sdwords: [0x11111111, 0x22222222, 0x33333333, 0x44444444] } },
    },
  ];
  const reflBgl1 = cache.createBindGroupLayoutFromReflection(s43GroupA);
  const reflBgl2 = cache.createBindGroupLayoutFromReflection(s43GroupB);
  assert.equal(reflBgl1, reflBgl2);
  // Submitted WebGPU descriptor stays clean (no raw `source` metadata leaked into GPUBindGroupLayoutDescriptor)
  assert.equal('source' in reflBgl1.desc.entries[1], false);
  assert.equal(cache.getStats().inlineSamplerLayouts, 2);
  assert.equal(cache.getStats().layoutDedupHits, 3);

  // 4. Reflection group with top-level inlineSamplerBindings flag (even without sampler sub-object)
  const reflFlagged = cache.createBindGroupLayoutFromReflection({
    group: 1,
    inlineSamplerBindings: 1,
    entries: [{ binding: 2, visibility: 'compute', buffer: { type: 'read-only-storage' } }],
  });
  assert.equal(reflFlagged.kind, 'GPUBindGroupLayout');
  assert.equal(cache.getStats().inlineSamplerLayouts, 3);

  // 5. Fallback bind-group layouts: fullscreen_triangle_cbv (buffer only) vs textured_quad_24b (texture+sampler)
  cache.getOrCreateFallbackBindGroupLayout('fullscreen_triangle_cbv');
  assert.equal(cache.getStats().inlineSamplerLayouts, 3);
  const quadBgl1 = cache.getOrCreateFallbackBindGroupLayout('textured_quad_24b');
  const quadBgl2 = cache.getOrCreateFallbackBindGroupLayout('textured_quad_24b');
  assert.equal(quadBgl1, quadBgl2);
  assert.equal(cache.getStats().inlineSamplerLayouts, 4);
  assert.equal(cache.getStats().layoutDedupHits, 4);

  // 6. Failed BGL creation does NOT increment inlineSamplerLayouts or layoutDedupHits
  failNextBgl = true;
  assert.throws(
    () =>
      cache.getOrCreateBindGroupLayout({
        entries: [{ binding: 7, visibility: SHADER_STAGE_BITS.FRAGMENT, sampler: { type: 'comparison' } }],
      }),
    /BGL creation failed/
  );
  assert.equal(cache.getStats().inlineSamplerLayouts, 4);
  assert.equal(cache.getStats().layoutDedupHits, 4);

  // 7. resetStats() resets inlineSamplerLayouts & layoutDedupHits cleanly
  cache.resetStats();
  assert.equal(cache.getStats().inlineSamplerLayouts, 0);
  assert.equal(cache.getStats().layoutDedupHits, 0);
  assert.equal(cache.getStats().layoutHits, 0);
  assert.equal(cache.getStats().layoutMisses, 0);
  assert.equal(cache.getStats().layoutCount, 6);
});

test('W246 sanitizes avgCompileMs, maxCompileMs, hitRate clamping, JSON serialization, and IIFE isolation', async () => {
  // Non-monotonic and NaN clock samples must never produce negative or NaN compile timings
  let clock = 100;
  let nextEndClock = 90;
  const device = createMockGpuDevice({
    renderSyncImpl: (desc, id) => {
      clock = nextEndClock;
      return { kind: 'GPURenderPipeline', id, desc };
    },
    renderAsyncImpl: async (desc, id) => {
      clock = nextEndClock;
      return { kind: 'GPURenderPipeline', id, async: true, desc };
    },
  });
  const cache = new WebGpuPipelineCache({
    device,
    now: () => clock,
  });
  const vs = device.createShaderModule({ label: 'wgpk:vertex:0x0000000000072001', code: 'vs' });

  // Compile 1: t0=100, t1=90 (non-monotonic backwards clock) -> clamped to 0ms
  clock = 100;
  nextEndClock = 90;
  cache.getOrCreateRenderPipeline({ layout: 'auto', vertex: { module: vs, entryPoint: 'vs_a' } });

  // Compile 2: t0=200, t1=NaN -> clamped to 0ms
  clock = 200;
  nextEndClock = NaN;
  cache.getOrCreateRenderPipeline({ layout: 'auto', vertex: { module: vs, entryPoint: 'vs_b' } });

  // Compile 3: t0=300, t1=315.5 -> 15.5ms
  clock = 300;
  nextEndClock = 315.5;
  await cache.getOrCreateRenderPipelineAsync({ layout: 'auto', vertex: { module: vs, entryPoint: 'vs_c' } });

  // Compile 4: t0=400, t1=404.5 -> 4.5ms
  clock = 400;
  nextEndClock = 404.5;
  cache.getOrCreateRenderPipeline({ layout: 'auto', vertex: { module: vs, entryPoint: 'vs_d' } });

  const stats = cache.getStats();
  assert.equal(stats.compileCount, 4);
  assert.equal(stats.totalCompileMs, 20.0);
  assert.equal(stats.avgCompileMs, 5.0);
  assert.equal(stats.maxCompileMs, 15.5);
  assert.ok(Number.isFinite(stats.hitRate) && stats.hitRate >= 0 && stats.hitRate <= 1);

  // Defensive clamping check if internal counters are corrupted
  cache._stats.renderHits = 100;
  cache._stats.renderMisses = -50;
  const clamped = cache.getStats();
  assert.equal(clamped.hitRate, 1);
  assert.equal(clamped.cacheBreakdown.render.hitRate, 1);

  // Full stats snapshot is sanitized and cleanly JSON-serializable
  const roundTripped = JSON.parse(JSON.stringify(stats));
  assert.deepEqual(roundTripped, stats);
  assert.equal(typeof roundTripped.pipelineHits, 'number');
  assert.equal(typeof roundTripped.pipelineMisses, 'number');
  assert.equal(typeof roundTripped.hitRate, 'number');
  assert.equal(typeof roundTripped.negativeCacheHits, 'number');
  assert.equal(typeof roundTripped.layoutDedupHits, 'number');
  assert.equal(typeof roundTripped.inlineSamplerLayouts, 'number');
  assert.equal(typeof roundTripped.avgCompileMs, 'number');
  assert.equal(typeof roundTripped.maxCompileMs, 'number');

  // Classic-Worker sandbox isolation: no W246 internal helpers leaked to global scope
  const scriptPath = path.resolve(
    path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')),
    '../web/workers/pipeline_cache.js'
  );
  const source = fs.readFileSync(scriptPath, 'utf8');
  const sandbox = { performance, Date };
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox);

  assert.equal(sandbox.entryHasInlineSampler, undefined);
  assert.equal(sandbox.descriptorHasInlineSampler, undefined);
  assert.equal(sandbox.computeClampedHitRate, undefined);
  assert.equal(typeof sandbox.PlayGta6PipelineCache.WebGpuPipelineCache.prototype.resetStats, 'function');
});

// --- W281: 30 FPS Fast-Path PSO Lookup Cache Tests ---

test('W281 fast-path render & compute lookups hit by numeric/bigint/string key, WeakMap descriptor, and 53-bit int hash without JSON.stringify', async () => {
  const device = createMockGpuDevice();
  const cache = new WebGpuPipelineCache({ device });

  const vs = device.createShaderModule({ label: 'wgpk:vertex:0x0000000000080001', code: 'vs' });
  const fs = device.createShaderModule({ label: 'wgpk:fragment:0x0000000000080002', code: 'fs' });
  const cs = device.createShaderModule({ label: 'wgpk:compute:0x0000000000080003', code: 'cs' });

  const renderDesc = {
    layout: 'auto',
    vertex: {
      module: vs,
      entryPoint: 'vs_main',
      buffers: [
        {
          arrayStride: 24,
          stepMode: 'vertex',
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x4' },
            { shaderLocation: 1, offset: 16, format: 'float32x2' },
          ],
        },
      ],
    },
    fragment: {
      module: fs,
      entryPoint: 'fs_main',
      targets: [
        {
          format: 'bgra8unorm',
          writeMask: 0xf,
          blend: {
            color: { operation: 'add', srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
            alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          },
        },
      ],
    },
    primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  };
  const computeDesc = {
    layout: 'auto',
    compute: { module: cs, entryPoint: 'cs_main' },
  };

  // Pure fast-path lookups on cold cache miss cleanly without compiling
  assert.equal(cache.getFastRenderPipeline(101), null);
  assert.equal(cache.getFastRenderPipeline(renderDesc), null);
  assert.equal(cache.getFastComputePipeline(202), null);
  assert.equal(cache.getFastComputePipeline(computeDesc), null);
  assert.equal(cache.getFastRenderPipeline(''), null);
  assert.equal(cache.getFastRenderPipeline(NaN), null);
  assert.equal(cache.getFastRenderPipeline(null), null);
  assert.equal(device.calls.createRenderPipeline.length, 0);
  assert.equal(device.calls.createComputePipeline.length, 0);
  assert.equal(cache.getStats().fastLookupHits, 0);
  assert.equal(cache.getStats().fastLookupMisses, 7);
  assert.equal(cache.getStats().fastLookupHitRate, 0);

  // Compile render + compute pipelines and also bind explicit numeric/bigint/string fast keys
  const rp = cache.getOrCreateRenderPipeline(renderDesc);
  const rpByNum = cache.getFastRenderPipeline(101, renderDesc);
  const rpByBigInt = cache.getOrCreateRenderPipeline(renderDesc, 0x123456789abcdef0n);
  const rpByStr = cache.getOrCreateRenderPipeline(renderDesc, 'pso:menu:main');
  const cp = cache.getOrCreateComputePipeline(computeDesc);
  const cpByNum = cache.getFastComputePipeline(202, computeDesc);

  assert.equal(device.calls.createRenderPipeline.length, 4);
  assert.equal(device.calls.createComputePipeline.length, 2);

  // Verify hot-loop fast lookups bypass JSON.stringify completely
  const origStringify = JSON.stringify;
  let stringifyCalls = 0;
  JSON.stringify = () => {
    stringifyCalls += 1;
    throw new Error('JSON.stringify must not be called on W281 fast-path lookups');
  };
  try {
    // 1. Direct numeric/bigint/string fastKey lookups
    assert.equal(cache.getFastRenderPipeline(101), rpByNum);
    assert.equal(cache.getFastRenderPipeline(0x123456789abcdef0n), rpByBigInt);
    assert.equal(cache.getFastRenderPipeline('pso:menu:main'), rpByStr);
    assert.equal(cache.getFastComputePipeline(202), cpByNum);
    assert.equal(cache.getOrCreateRenderPipeline(renderDesc, 101), rpByNum);
    assert.equal(await cache.getOrCreateRenderPipelineAsync(renderDesc, 'pso:menu:main'), rpByStr);
    assert.equal(cache.getOrCreateComputePipeline(computeDesc, 202), cpByNum);

    // 2. Same descriptor object (WeakMap memoization hit)
    for (let frame = 0; frame < 30; frame += 1) {
      assert.equal(cache.getFastRenderPipeline(renderDesc), rp);
      assert.equal(cache.getOrCreateRenderPipeline(renderDesc), rp);
      assert.equal(cache.getFastComputePipeline(computeDesc), cp);
    }

    // 3. Freshly constructed descriptor literal with reordered keys (53-bit int-hash hit)
    const freshReorderedRender = {
      depthStencil: { depthCompare: 'less', depthWriteEnabled: true, format: 'depth24plus' },
      primitive: { frontFace: 'ccw', cullMode: 'back', topology: 'triangle-list' },
      fragment: {
        targets: [
          {
            blend: {
              alpha: { dstFactor: 'one-minus-src-alpha', srcFactor: 'one', operation: 'add' },
              color: { dstFactor: 'one-minus-src-alpha', srcFactor: 'src-alpha', operation: 'add' },
            },
            writeMask: 0xf,
            format: 'bgra8unorm',
          },
        ],
        entryPoint: 'fs_main',
        module: fs,
      },
      vertex: {
        buffers: [
          {
            attributes: [
              { format: 'float32x4', offset: 0, shaderLocation: 0 },
              { format: 'float32x2', offset: 16, shaderLocation: 1 },
            ],
            stepMode: 'vertex',
            arrayStride: 24,
          },
        ],
        entryPoint: 'vs_main',
        module: vs,
      },
      layout: 'auto',
    };
    const freshReorderedCompute = {
      compute: { entryPoint: 'cs_main', module: cs },
      layout: 'auto',
    };
    assert.equal(
      computeFastDescriptorHash(renderDesc),
      computeFastDescriptorHash(freshReorderedRender)
    );
    assert.equal(
      computeFastDescriptorHash(computeDesc),
      computeFastDescriptorHash(freshReorderedCompute)
    );
    assert.equal(cache.getFastRenderPipeline(freshReorderedRender), rp);
    assert.equal(cache.getOrCreateRenderPipeline(freshReorderedRender), rp);
    assert.equal(cache.getFastComputePipeline(freshReorderedCompute), cp);
    assert.equal(await cache.getOrCreateComputePipelineAsync(freshReorderedCompute), cp);
  } finally {
    JSON.stringify = origStringify;
  }

  assert.equal(stringifyCalls, 0);
  assert.equal(device.calls.createRenderPipeline.length, 4);
  assert.equal(device.calls.createComputePipeline.length, 2);

  const stats = cache.getStats();
  assert.equal(stats.fastLookupHits, 101); // 7 + 90 + 4
  assert.equal(stats.fastLookupMisses, 13); // 7 initial + 6 compiles
  assert.equal(stats.fastLookupHitRate, 101 / 114);
  assert.ok(stats.fastLookupHitRate > 0.88 && stats.fastLookupHitRate <= 1);

  // resetStats() clears fastLookup counters while keeping warm fast-path tables intact
  cache.resetStats();
  assert.equal(cache.getStats().fastLookupHits, 0);
  assert.equal(cache.getStats().fastLookupMisses, 0);
  assert.equal(cache.getStats().fastLookupHitRate, 0);
  assert.equal(cache.getFastRenderPipeline(101), rpByNum);
  assert.equal(cache.getFastRenderPipeline(renderDesc), rp);
  assert.equal(cache.getStats().fastLookupHits, 2);
  assert.equal(cache.getStats().fastLookupMisses, 0);
  assert.equal(cache.getStats().fastLookupHitRate, 1);
});

test('W281 invalidates fast-path lookups cleanly on clear, reset, destroy, device loss, setDevice, shader-pack reload, and eviction', async () => {
  let clock = 1000;
  let device1 = createMockGpuDevice();
  const cache = new WebGpuPipelineCache({
    device: device1,
    maxRenderPipelines: 4,
    maxFastLookupEntries: 2,
    now: () => clock,
  });

  const makeModules = (dev) => ({
    vs: dev.createShaderModule({ label: 'wgpk:vertex:0x0000000000081001', code: 'vs' }),
    fs: dev.createShaderModule({ label: 'wgpk:fragment:0x0000000000081002', code: 'fs' }),
    cs: dev.createShaderModule({ label: 'wgpk:compute:0x0000000000081003', code: 'cs' }),
  });

  let { vs, fs, cs } = makeModules(device1);
  const descA = {
    layout: 'auto',
    vertex: { module: vs, entryPoint: 'vs_a' },
    fragment: { module: fs, entryPoint: 'fs_a', targets: [{ format: 'bgra8unorm' }] },
  };
  const descB = {
    layout: 'auto',
    vertex: { module: vs, entryPoint: 'vs_b' },
    fragment: { module: fs, entryPoint: 'fs_b', targets: [{ format: 'bgra8unorm' }] },
  };
  const descC = {
    layout: 'auto',
    vertex: { module: vs, entryPoint: 'vs_c' },
    fragment: { module: fs, entryPoint: 'fs_c', targets: [{ format: 'bgra8unorm' }] },
  };
  const computeDesc = { layout: 'auto', compute: { module: cs, entryPoint: 'cs_main' } };

  // 1. Bounded maxFastLookupEntries (capacity = 2) evicts oldest fastKey while backing cache still holds pipeline
  const pA = cache.getOrCreateRenderPipeline(descA, 'pso_a');
  const pB = cache.getOrCreateRenderPipeline(descB, 'pso_b');
  const pC = cache.getOrCreateRenderPipeline(descC, 'pso_c');
  // Each compile inserts canonical key + keyHint; 'pso_a' was evicted from the 2-slot fast table
  assert.equal(cache.getFastRenderPipeline('pso_a'), null);
  // Backing cache still serves pA without recompiling, and re-warms 'pso_a'
  assert.equal(cache.getOrCreateRenderPipeline(descA, 'pso_a'), pA);
  assert.equal(device1.calls.createRenderPipeline.length, 3);
  assert.equal(cache.getFastRenderPipeline('pso_a'), pA);
  assert.ok(pB && pC);

  // 2. clear() invalidates all fast-path entries (key + WeakMap + int-hash)
  cache.clear();
  assert.equal(cache.getFastRenderPipeline('pso_a'), null);
  assert.equal(cache.getFastRenderPipeline(descA), null);
  const pAAfterClear = cache.getOrCreateRenderPipeline(descA);
  assert.notEqual(pAAfterClear, pA);
  assert.equal(cache.getFastRenderPipeline(descA), pAAfterClear);

  // 3. reset() invalidates fast-path entries
  cache.reset();
  assert.equal(cache.getFastRenderPipeline(descA), null);

  // 4. destroy() invalidates fast-path entries
  const pBeforeDestroy = cache.getOrCreateRenderPipeline(descA, 777);
  const cBeforeDestroy = cache.getOrCreateComputePipeline(computeDesc, 888);
  assert.equal(cache.getFastRenderPipeline(777), pBeforeDestroy);
  assert.equal(cache.getFastComputePipeline(888), cBeforeDestroy);
  cache.destroy();
  assert.equal(cache.getFastRenderPipeline(777), null);
  assert.equal(cache.getFastRenderPipeline(descA), null);
  assert.equal(cache.getFastComputePipeline(888), null);
  assert.equal(cache.getFastComputePipeline(computeDesc), null);

  // 5. onDeviceLost() invalidates fast-path entries
  const pBeforeLoss = cache.getOrCreateRenderPipeline(descA, 123);
  assert.equal(cache.getFastRenderPipeline(123), pBeforeLoss);
  cache.onDeviceLost();
  assert.equal(cache.getFastRenderPipeline(123), null);
  assert.equal(cache.getFastRenderPipeline(descA), null);

  // 6. setDevice(newDevice) invalidates fast-path entries so stale pipelines from old device are never returned
  const pOldDevice = cache.getOrCreateRenderPipeline(descA, 555);
  assert.equal(cache.getFastRenderPipeline(555), pOldDevice);
  const device2 = createMockGpuDevice();
  assert.equal(cache.setDevice(device2), true);
  assert.equal(cache.getFastRenderPipeline(555), null);
  assert.equal(cache.getFastRenderPipeline(descA), null);
  const pNewDevice = cache.getOrCreateRenderPipeline(descA, 555);
  assert.notEqual(pNewDevice, pOldDevice);
  assert.equal(device2.calls.createRenderPipeline.length, 1);
  assert.equal(cache.getFastRenderPipeline(555), pNewDevice);

  // 7. onShaderPackUpdated({ clearPipelines: true }) invalidates fast-path entries
  cache.onShaderPackUpdated({ clearPipelines: true });
  assert.equal(cache.getFastRenderPipeline(555), null);
  assert.equal(cache.getFastRenderPipeline(descA), null);

  // 8. evictOlderThan() invalidates stale fast-path entries for expired pipelines only
  clock = 2000;
  const pOld = cache.getOrCreateRenderPipeline(descA, 'old_key');
  clock = 3500;
  const pFresh = cache.getOrCreateRenderPipeline(descB, 'fresh_key');
  clock = 3600;
  assert.equal(cache.evictOlderThan(1000), 1);
  assert.equal(cache.getFastRenderPipeline('old_key'), null);
  assert.equal(cache.getFastRenderPipeline(descA), null);
  assert.equal(cache.getFastRenderPipeline('fresh_key'), pFresh);
  assert.notEqual(pOld, pFresh);

  // 9. prewarmRenderPipelines() warms fast-path lookups immediately
  cache.clear({ resetStats: true });
  await cache.prewarmRenderPipelines([descA, descB]);
  const statsBefore = cache.getStats();
  assert.ok(cache.getFastRenderPipeline(descA));
  assert.ok(cache.getFastRenderPipeline(descB));
  assert.equal(cache.getStats().fastLookupHits, statsBefore.fastLookupHits + 2);
});

test('W281 guards against in-place descriptor mutation, shader re-registration, fastLookupHitRate clamping, and IIFE isolation', () => {
  const device = createMockGpuDevice();
  const cache = new WebGpuPipelineCache({ device });
  const vsMod = device.createShaderModule({ label: 'wgpk:vertex:0x0000000000082001', code: 'vs' });
  const fsMod = device.createShaderModule({ label: 'wgpk:fragment:0x0000000000082002', code: 'fs' });

  const mutableDesc = {
    layout: 'auto',
    vertex: {
      module: vsMod,
      entryPoint: 'vs_v1',
      buffers: [
        {
          arrayStride: 24,
          stepMode: 'vertex',
          attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }],
        },
      ],
    },
    fragment: {
      module: fsMod,
      entryPoint: 'fs_v1',
      targets: [{ format: 'bgra8unorm', writeMask: 0xf }],
    },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
  };

  const p1 = cache.getOrCreateRenderPipeline(mutableDesc);
  assert.equal(cache.getFastRenderPipeline(mutableDesc), p1);

  // Mutate entryPoint in-place -> WeakMap memo must detect mutation and compile distinct PSO
  mutableDesc.vertex.entryPoint = 'vs_v2';
  assert.equal(cache.getFastRenderPipeline(mutableDesc), null);
  const p2 = cache.getOrCreateRenderPipeline(mutableDesc);
  assert.notEqual(p2, p1);
  assert.equal(cache.getFastRenderPipeline(mutableDesc), p2);

  // Mutate nested target format in-place -> must detect mutation
  mutableDesc.fragment.targets[0].format = 'rgba8unorm-srgb';
  assert.equal(cache.getFastRenderPipeline(mutableDesc), null);
  const p3 = cache.getOrCreateRenderPipeline(mutableDesc);
  assert.notEqual(p3, p2);

  // Mutate vertex buffer stride in-place -> must detect mutation
  mutableDesc.vertex.buffers[0].arrayStride = 32;
  assert.equal(cache.getFastRenderPipeline(mutableDesc), null);
  const p4 = cache.getOrCreateRenderPipeline(mutableDesc);
  assert.notEqual(p4, p3);

  // Re-registering an opaque shader module ID invalidates memoized fast entries
  const opaqueVs = {};
  const opaqueDesc = { layout: 'auto', vertex: { module: opaqueVs, entryPoint: 'vs_main' } };
  const pOpaque1 = cache.getOrCreateRenderPipeline(opaqueDesc);
  assert.equal(cache.getFastRenderPipeline(opaqueDesc), pOpaque1);
  cache.registerShaderModule(opaqueVs, 0xdeadbeef00001111n);
  assert.equal(cache.getFastRenderPipeline(opaqueDesc), null);
  const pOpaque2 = cache.getOrCreateRenderPipeline(opaqueDesc);
  assert.notEqual(pOpaque2, pOpaque1);

  // Defensive fastLookupHitRate clamping to [0, 1] under corrupted counters
  cache._stats.fastLookupHits = 50;
  cache._stats.fastLookupMisses = -20;
  assert.equal(cache.getStats().fastLookupHitRate, 1);

  cache._stats.fastLookupHits = -10;
  cache._stats.fastLookupMisses = 5;
  assert.equal(cache.getStats().fastLookupHitRate, 0);
  assert.equal(cache.getStats().fastLookupHits, 0);

  cache._stats.fastLookupHits = NaN;
  cache._stats.fastLookupMisses = Infinity;
  assert.equal(cache.getStats().fastLookupHitRate, 0);
  assert.equal(cache.getStats().fastLookupHits, 0);
  assert.equal(cache.getStats().fastLookupMisses, 0);

  // Option validation for maxFastLookupEntries
  assert.throws(
    () => new WebGpuPipelineCache({ device, maxFastLookupEntries: 0 }),
    /maxFastLookupEntries must be a positive integer/
  );
  assert.equal(DEFAULT_MAX_FAST_LOOKUP_ENTRIES, 2048);

  // Classic-Worker sandbox isolation: no W281 internal helpers leaked to global scope
  const scriptPath = path.resolve(
    path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')),
    '../web/workers/pipeline_cache.js'
  );
  const source = fs.readFileSync(scriptPath, 'utf8');
  const sandbox = { performance, Date };
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox);

  assert.equal(sandbox.computeFastRenderIntHash, undefined);
  assert.equal(sandbox.captureRenderSnapshot, undefined);
  assert.equal(sandbox.matchesRenderSnapshot, undefined);
  assert.equal(typeof sandbox.PlayGta6PipelineCache.computeFastDescriptorHash, 'function');
  assert.equal(
    typeof sandbox.PlayGta6PipelineCache.WebGpuPipelineCache.prototype.getFastRenderPipeline,
    'function'
  );
  assert.equal(
    typeof sandbox.PlayGta6PipelineCache.WebGpuPipelineCache.prototype.getFastComputePipeline,
    'function'
  );
  assert.equal(
    typeof sandbox.PlayGta6PipelineCache.WebGpuPipelineCache.prototype.destroy,
    'function'
  );
});

// --- W334: Compile-Latency Rolling Window, Priority Hints & Bounded LRU Reclaim Tests ---

test('W334 tracks rolling compile-latency percentiles via reusable Float64Array scratch buffer and slowCompilesOverBudget', async () => {
  assert.equal(DEFAULT_COMPILE_BUDGET_MS, 33.333);
  assert.equal(DEFAULT_COMPILE_LATENCY_WINDOW_SIZE, 128);

  let clock = 0;
  const durations = [45.0, 10.0, 20.0, 33.333, 33.334, 5.0, 15.0];
  let idx = 0;

  const device = createMockGpuDevice({
    renderSyncImpl: (desc, id) => {
      const dt = durations[idx++] ?? 1.0;
      clock += dt;
      return { kind: 'GPURenderPipeline', id, desc };
    },
    computeAsyncImpl: async (desc, id) => {
      const dt = durations[idx++] ?? 1.0;
      clock += dt;
      return { kind: 'GPUComputePipeline', id, async: true, desc };
    },
  });

  // Small rolling window (size = 5) with default 33.333ms 30 FPS budget
  const cache = new WebGpuPipelineCache({
    device,
    now: () => clock,
    compileLatencyWindowSize: 5,
  });

  const scratchRef = cache._compileLatencyScratch;
  const windowRef = cache._compileLatencyWindow;
  assert.ok(scratchRef instanceof Float64Array);
  assert.ok(windowRef instanceof Float64Array);
  assert.equal(scratchRef.length, 5);
  assert.equal(windowRef.length, 5);

  // Empty state returns zeros
  const emptyLatency = cache.getCompileLatencyStats();
  assert.deepEqual(emptyLatency, {
    p50Ms: 0,
    p95Ms: 0,
    maxMs: 0,
    slowCompilesOverBudget: 0,
    budgetMs: 33.333,
    sampleCount: 0,
    windowSize: 5,
  });

  const vs = device.createShaderModule({ label: 'wgpk:vertex:0x0000000000090001', code: 'vs' });
  const cs = device.createShaderModule({ label: 'wgpk:compute:0x0000000000090002', code: 'cs' });

  // First 5 compiles: [45.0, 10.0, 20.0, 33.333, 33.334]
  // Sorted: [10.0, 20.0, 33.333, 33.334, 45.0]
  // p50 (ceil(0.5*5)-1 = 2) = 33.333, p95 (ceil(0.95*5)-1 = 4) = 45.0, max = 45.0
  // Over 33.333 budget: 45.0 and 33.334 (exact 33.333 is within budget) -> 2
  cache.getOrCreateRenderPipeline({ layout: 'auto', vertex: { module: vs, entryPoint: 'vs_0' } });
  cache.getOrCreateRenderPipeline({ layout: 'auto', vertex: { module: vs, entryPoint: 'vs_1' } });
  cache.getOrCreateRenderPipeline({ layout: 'auto', vertex: { module: vs, entryPoint: 'vs_2' } });
  await cache.getOrCreateComputePipelineAsync({
    layout: 'auto',
    compute: { module: cs, entryPoint: 'cs_3' },
  });
  cache.getOrCreateRenderPipeline({ layout: 'auto', vertex: { module: vs, entryPoint: 'vs_4' } });

  let stats = cache.getStats();
  assert.equal(stats.p50Ms, 33.333);
  assert.equal(stats.p95Ms, 45.0);
  assert.equal(stats.maxMs, 45.0);
  assert.equal(stats.maxCompileMs, 45.0);
  assert.equal(stats.slowCompilesOverBudget, 2);
  assert.deepEqual(stats.compileLatency, {
    p50Ms: 33.333,
    p95Ms: 45.0,
    maxMs: 45.0,
    slowCompilesOverBudget: 2,
    budgetMs: 33.333,
    sampleCount: 5,
    windowSize: 5,
  });

  // 2 more fast compiles roll out 45.0 and 10.0 -> rolling window holds [5.0, 15.0, 20.0, 33.333, 33.334]
  cache.getOrCreateRenderPipeline({ layout: 'auto', vertex: { module: vs, entryPoint: 'vs_5' } });
  cache.getOrCreateRenderPipeline({ layout: 'auto', vertex: { module: vs, entryPoint: 'vs_6' } });

  stats = cache.getStats();
  assert.equal(stats.p50Ms, 20.0);
  assert.equal(stats.p95Ms, 33.334);
  assert.equal(stats.maxMs, 33.334);
  assert.equal(stats.maxCompileMs, 45.0); // lifetime max preserved
  assert.equal(stats.slowCompilesOverBudget, 2);
  assert.equal(stats.compileLatency.sampleCount, 5);

  // Verify pre-allocated Float64Array buffers are reused without reallocation
  assert.equal(cache._compileLatencyScratch, scratchRef);
  assert.equal(cache._compileLatencyWindow, windowRef);

  // resetStats() resets rolling window & slow count while reusing the exact same typed arrays
  cache.resetStats();
  assert.equal(cache._compileLatencyScratch, scratchRef);
  assert.equal(cache._compileLatencyWindow, windowRef);
  stats = cache.getStats();
  assert.equal(stats.p50Ms, 0);
  assert.equal(stats.p95Ms, 0);
  assert.equal(stats.maxMs, 0);
  assert.equal(stats.slowCompilesOverBudget, 0);
  assert.equal(stats.compileLatency.sampleCount, 0);

  // Configurable compileBudgetMs (e.g. 16.667ms 60 FPS target)
  let clock60 = 0;
  const budgetSamples = [8.0, 16.667, 17.0, 25.0];
  let bIdx = 0;
  const device60 = createMockGpuDevice({
    renderSyncImpl: (desc, id) => {
      clock60 += budgetSamples[bIdx++] ?? 1.0;
      return { kind: 'GPURenderPipeline', id, desc };
    },
  });
  const cache60 = new WebGpuPipelineCache({
    device: device60,
    now: () => clock60,
    compileBudgetMs: 16.667,
  });
  const vs60 = device60.createShaderModule({ code: 'vs60' });
  for (let i = 0; i < budgetSamples.length; i += 1) {
    cache60.getOrCreateRenderPipeline({
      layout: 'auto',
      vertex: { module: vs60, entryPoint: `vs_${i}` },
    });
  }
  const stats60 = cache60.getStats();
  assert.equal(stats60.compileLatency.budgetMs, 16.667);
  assert.equal(stats60.slowCompilesOverBudget, 2); // 17.0 and 25.0
  assert.equal(stats60.p50Ms, 16.667);
  assert.equal(stats60.p95Ms, 25.0);
  assert.equal(stats60.maxMs, 25.0);
});

test('W334 enforces priority-aware bounded LRU eviction via per-map cap and global maxEntries', async () => {
  assert.equal(PIPELINE_PRIORITY_TIERS.CRITICAL, 'critical');
  assert.equal(PIPELINE_PRIORITY_TIERS.HIGH, 'high');
  assert.equal(PIPELINE_PRIORITY_TIERS.NORMAL, 'normal');
  assert.equal(PIPELINE_PRIORITY_TIERS.LOW, 'low');
  assert.ok(PIPELINE_PRIORITY_RANKS.critical > PIPELINE_PRIORITY_RANKS.high);
  assert.ok(PIPELINE_PRIORITY_RANKS.high > PIPELINE_PRIORITY_RANKS.normal);
  assert.ok(PIPELINE_PRIORITY_RANKS.normal > PIPELINE_PRIORITY_RANKS.low);

  assert.equal(normalizePipelinePriority('CRITICAL'), 'critical');
  assert.equal(normalizePipelinePriority('urgent'), 'critical');
  assert.equal(normalizePipelinePriority('High'), 'high');
  assert.equal(normalizePipelinePriority('low'), 'low');
  assert.equal(normalizePipelinePriority('background'), 'low');
  assert.equal(normalizePipelinePriority(undefined), 'normal');
  assert.equal(normalizePipelinePriority('unknown-tier'), 'normal');

  // 1. Per-map priority-aware LRU eviction protects critical/high entries even if less recently touched
  const device = createMockGpuDevice();
  const cache = new WebGpuPipelineCache({
    device,
    maxRenderPipelines: 2,
  });
  const vs = device.createShaderModule({ label: 'wgpk:vertex:0x0000000000091001', code: 'vs' });
  const makeRender = (ep) => ({ layout: 'auto', vertex: { module: vs, entryPoint: ep } });

  const pCrit = cache.getOrCreateRenderPipeline(makeRender('vs_crit'), 'k_crit', {
    priority: 'critical',
  });
  const pLow = cache.getOrCreateRenderPipeline(makeRender('vs_low'), 'k_low', {
    priority: 'low',
  });

  // Touch pLow so it is MORE recently used than pCrit
  assert.equal(cache.getFastRenderPipeline('k_low'), pLow);

  // Inserting a normal entry must evict pLow (lowest priority), keeping pCrit alive
  const pNorm = cache.getOrCreateRenderPipeline(makeRender('vs_norm'), 'k_norm', {
    priority: 'normal',
  });
  assert.equal(cache.getStats().renderCount, 2);
  assert.equal(cache.getStats().evictions, 1);
  assert.equal(cache.getStats().reclaimedCount, 1);
  assert.equal(cache.getFastRenderPipeline('k_crit'), pCrit);
  assert.equal(cache.getFastRenderPipeline('k_norm'), pNorm);
  assert.equal(cache.getFastRenderPipeline('k_low'), null);

  // Priority promotion on hit: promote k_norm to critical, insert high -> older critical evicted only after lower tiers
  const pHigh = cache.getOrCreateRenderPipeline(makeRender('vs_high'), 'k_high', {
    priority: 'high',
  });
  // k_norm was rank 1 (normal) < k_crit rank 3 (critical), so k_norm is evicted
  assert.equal(cache.getFastRenderPipeline('k_crit'), pCrit);
  assert.equal(cache.getFastRenderPipeline('k_high'), pHigh);
  assert.equal(cache.getFastRenderPipeline('k_norm'), null);
  assert.equal(cache.getStats().evictions, 2);
  assert.equal(cache.getStats().reclaimedCount, 2);

  // 2. Global maxEntries bounds total entries across render + compute + bindGroupLayouts
  const device2 = createMockGpuDevice();
  const bounded = new WebGpuPipelineCache({
    device: device2,
    maxEntries: 3,
  });
  const vs2 = device2.createShaderModule({ code: 'vs2' });
  const cs2 = device2.createShaderModule({ code: 'cs2' });

  const rCrit = bounded.getOrCreateRenderPipeline(
    { layout: 'auto', vertex: { module: vs2, entryPoint: 'r_crit' } },
    'r_crit',
    { priority: 'critical' }
  );
  const cLow = bounded.getOrCreateComputePipeline(
    { layout: 'auto', compute: { module: cs2, entryPoint: 'c_low' } },
    'c_low',
    { priority: 'low' }
  );
  const bglHigh = bounded.getOrCreateBindGroupLayout(
    { entries: [{ binding: 0, visibility: SHADER_STAGE_BITS.VERTEX, buffer: { type: 'uniform' } }] },
    'bgl_high',
    { priority: 'high' }
  );
  assert.equal(bounded.getStats().renderCount + bounded.getStats().computeCount + bounded.getStats().layoutCount, 3);
  assert.equal(bounded.getStats().evictions, 0);

  // Touch cLow to make it newest, then insert a 4th entry -> cLow is evicted across maps because it is 'low'
  assert.equal(bounded.getFastComputePipeline('c_low'), cLow);
  const rNorm = bounded.getOrCreateRenderPipeline(
    { layout: 'auto', vertex: { module: vs2, entryPoint: 'r_norm' } },
    'r_norm',
    { priority: 'normal' }
  );

  const bStats = bounded.getStats();
  assert.equal(bStats.renderCount, 2);
  assert.equal(bStats.computeCount, 0);
  assert.equal(bStats.layoutCount, 1);
  assert.equal(bStats.evictions, 1);
  assert.equal(bStats.reclaimedCount, 1);
  assert.equal(bounded.getFastRenderPipeline('r_crit'), rCrit);
  assert.equal(bounded.getFastRenderPipeline('r_norm'), rNorm);
  assert.equal(bounded.getFastComputePipeline('c_low'), null);
  assert.equal(
    bounded.getOrCreateBindGroupLayout(
      { entries: [{ binding: 0, visibility: SHADER_STAGE_BITS.VERTEX, buffer: { type: 'uniform' } }] },
      'bgl_high'
    ),
    bglHigh
  );

  // 3. In-flight async priority promotion: low caller + concurrent critical caller -> stored as critical
  const deferred = createDeferred();
  const asyncDev = createMockGpuDevice({
    renderAsyncImpl: async (desc, id) => {
      await deferred.promise;
      return { kind: 'GPURenderPipeline', id, desc };
    },
  });
  const asyncCache = new WebGpuPipelineCache({ device: asyncDev, maxRenderPipelines: 2 });
  const avs = asyncDev.createShaderModule({ code: 'avs' });
  const sharedDesc = { layout: 'auto', vertex: { module: avs, entryPoint: 'vs_shared' } };

  const p1 = asyncCache.getOrCreateRenderPipelineAsync(sharedDesc, 'k_shared', { priority: 'low' });
  const p2 = asyncCache.getOrCreateRenderPipelineAsync(sharedDesc, 'k_shared', {
    priority: 'critical',
  });
  deferred.resolve();
  const [res1, res2] = await Promise.all([p1, p2]);
  assert.equal(res1, res2);

  // Add a normal entry, then a high entry -> promoted k_shared (critical) must survive, normal gets evicted
  asyncCache.getOrCreateRenderPipeline(
    { layout: 'auto', vertex: { module: avs, entryPoint: 'vs_n' } },
    'k_n',
    { priority: 'normal' }
  );
  asyncCache.getOrCreateRenderPipeline(
    { layout: 'auto', vertex: { module: avs, entryPoint: 'vs_h' } },
    'k_h',
    { priority: 'high' }
  );
  assert.equal(asyncCache.getFastRenderPipeline('k_shared'), res1);
  assert.equal(asyncCache.getFastRenderPipeline('k_n'), null);
});

test('W334 trimTo, evictLowPriority, and priority-ordered prewarm reclaim cleanly and preserve fast-path consistency', async () => {
  const compileOrder = [];
  const device = createMockGpuDevice({
    renderAsyncImpl: async (desc, id) => {
      compileOrder.push(desc.vertex.entryPoint);
      return { kind: 'GPURenderPipeline', id, desc };
    },
    computeAsyncImpl: async (desc, id) => {
      compileOrder.push(desc.compute.entryPoint);
      return { kind: 'GPUComputePipeline', id, desc };
    },
  });
  const cache = new WebGpuPipelineCache({ device });
  const vs = device.createShaderModule({ label: 'wgpk:vertex:0x0000000000092001', code: 'vs' });
  const cs = device.createShaderModule({ label: 'wgpk:compute:0x0000000000092002', code: 'cs' });

  // Priority-ordered prewarm with concurrency: 1 compiles critical -> high -> normal -> low
  // while preserving original input order in summary.pipelines
  const prewarmRes = await cache.prewarmRenderPipelines(
    [
      { descriptor: { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_low_1' } }, keyHint: 'r_low_1', priority: 'low' },
      { descriptor: { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_norm' } }, keyHint: 'r_norm', priority: 'normal' },
      { descriptor: { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_crit' } }, keyHint: 'r_crit', priority: 'critical' },
      { descriptor: { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_low_2' } }, keyHint: 'r_low_2', priority: 'low' },
      { descriptor: { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_high' } }, keyHint: 'r_high', priority: 'high' },
    ],
    { concurrency: 1 }
  );

  assert.deepEqual(compileOrder, ['vs_crit', 'vs_high', 'vs_norm', 'vs_low_1', 'vs_low_2']);
  assert.equal(prewarmRes.compiled, 5);
  assert.equal(prewarmRes.pipelines[0].desc.vertex.entryPoint, 'vs_low_1');
  assert.equal(prewarmRes.pipelines[2].desc.vertex.entryPoint, 'vs_crit');
  assert.equal(prewarmRes.pipelines[4].desc.vertex.entryPoint, 'vs_high');

  // Add a low-priority compute pipeline (most recent low-priority entry)
  const cLow = cache.getOrCreateComputePipeline(
    { layout: 'auto', compute: { module: cs, entryPoint: 'cs_low' } },
    'c_low',
    { priority: 'low' }
  );
  assert.equal(cache.getFastComputePipeline('c_low'), cLow);

  // Touch r_low_1 so r_low_2 becomes the oldest low-priority entry
  assert.ok(cache.getFastRenderPipeline('r_low_1'));

  // evictLowPriority(1) evicts only the oldest 'low' entry (r_low_2)
  assert.equal(cache.evictLowPriority(1), 1);
  assert.equal(cache.getFastRenderPipeline('r_low_2'), null);
  assert.ok(cache.getFastRenderPipeline('r_low_1'));
  assert.equal(cache.getFastComputePipeline('c_low'), cLow);
  assert.equal(cache.getStats().reclaimedCount, 1);
  assert.equal(cache.getStats().evictions, 1);

  // evictLowPriority(10) evicts remaining 2 low entries (r_low_1, c_low) and never touches normal/high/critical
  assert.equal(cache.evictLowPriority(10), 2);
  assert.equal(cache.getFastRenderPipeline('r_low_1'), null);
  assert.equal(cache.getFastComputePipeline('c_low'), null);
  assert.ok(cache.getFastRenderPipeline('r_norm'));
  assert.ok(cache.getFastRenderPipeline('r_high'));
  assert.ok(cache.getFastRenderPipeline('r_crit'));
  assert.equal(cache.getStats().reclaimedCount, 3);
  assert.equal(cache.getStats().evictions, 3);

  // evictLowPriority() with no args on no remaining low entries is a no-op
  assert.equal(cache.evictLowPriority(), 0);

  // trimTo(2) evicts r_norm (lowest remaining tier), keeping r_high and r_crit
  assert.equal(cache.trimTo(2), 1);
  assert.equal(cache.getFastRenderPipeline('r_norm'), null);
  assert.ok(cache.getFastRenderPipeline('r_high'));
  assert.ok(cache.getFastRenderPipeline('r_crit'));
  assert.equal(cache.getStats().reclaimedCount, 4);
  assert.equal(cache.getStats().evictions, 4);

  // trimTo(5) when already <= 5 is a no-op
  assert.equal(cache.trimTo(5), 0);

  // trimTo(0) reclaims remaining 2 entries and invalidates all fast lookups
  assert.equal(cache.trimTo(0), 2);
  assert.equal(cache.getStats().renderCount, 0);
  assert.equal(cache.getStats().computeCount, 0);
  assert.equal(cache.getStats().layoutCount, 0);
  assert.equal(cache.getStats().reclaimedCount, 6);
  assert.equal(cache.getStats().evictions, 6);
  assert.equal(cache.getFastRenderPipeline('r_high'), null);
  assert.equal(cache.getFastRenderPipeline('r_crit'), null);

  // Priority-ordered prewarmComputePipelines also respects priority order
  compileOrder.length = 0;
  const computePrewarm = await cache.prewarmComputePipelines(
    [
      { descriptor: { layout: 'auto', compute: { module: cs, entryPoint: 'cs_bg' } }, priority: 'low' },
      { descriptor: { layout: 'auto', compute: { module: cs, entryPoint: 'cs_crit' } }, priority: 'critical' },
    ],
    { concurrency: 1 }
  );
  assert.deepEqual(compileOrder, ['cs_crit', 'cs_bg']);
  assert.equal(computePrewarm.pipelines[0].desc.compute.entryPoint, 'cs_bg');
  assert.equal(computePrewarm.pipelines[1].desc.compute.entryPoint, 'cs_crit');

  // evictLowPriority() with no args evicts cs_bg while preserving cs_crit
  assert.equal(cache.evictLowPriority(), 1);
  assert.equal(cache.getStats().computeCount, 1);
});

test('W334 validates options, preserves JSON serializability, backward compatibility, and classic-Worker IIFE isolation', () => {
  const device = createMockGpuDevice();

  assert.throws(
    () => new WebGpuPipelineCache({ device, maxEntries: 0 }),
    /maxEntries must be a positive integer/
  );
  assert.throws(
    () => new WebGpuPipelineCache({ device, maxEntries: -5 }),
    /maxEntries must be a positive integer/
  );
  assert.throws(
    () => new WebGpuPipelineCache({ device, compileBudgetMs: 0 }),
    /compileBudgetMs must be a finite positive number/
  );
  assert.throws(
    () => new WebGpuPipelineCache({ device, compileBudgetMs: NaN }),
    /compileBudgetMs must be a finite positive number/
  );
  assert.throws(
    () => new WebGpuPipelineCache({ device, compileLatencyWindowSize: 0 }),
    /compileLatencyWindowSize must be a positive integer/
  );

  const cache = new WebGpuPipelineCache({ device });
  assert.throws(() => cache.trimTo(-1), /limit must be a non-negative integer/);
  assert.throws(() => cache.trimTo(1.5), /limit must be a non-negative integer/);
  assert.throws(() => cache.evictLowPriority(-1), /count must be a non-negative integer/);
  assert.throws(() => cache.evictLowPriority(2.5), /count must be a non-negative integer/);

  // Stats and compile latency snapshots round-trip cleanly through JSON
  const stats = cache.getStats();
  assert.deepEqual(JSON.parse(JSON.stringify(stats)), stats);
  const latency = cache.getCompileLatencyStats();
  assert.deepEqual(JSON.parse(JSON.stringify(latency)), latency);

  // Classic-Worker sandbox isolation: no internal W334 helpers leaked to global scope
  const scriptPath = path.resolve(
    path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')),
    '../web/workers/pipeline_cache.js'
  );
  const source = fs.readFileSync(scriptPath, 'utf8');
  const sandbox = { performance, Date };
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox);

  assert.equal(sandbox.percentileFromSorted, undefined);
  assert.equal(sandbox.extractPriorityOption, undefined);
  assert.equal(sandbox.hasRenderNormalizationOptions, undefined);
  assert.equal(sandbox.validatePositiveNumber, undefined);
  assert.equal(sandbox.validateNonNegativeInt, undefined);
  assert.equal(sandbox.PlayGta6PipelineCache.DEFAULT_COMPILE_BUDGET_MS, 33.333);
  assert.equal(typeof sandbox.PlayGta6PipelineCache.normalizePipelinePriority, 'function');
  assert.equal(
    typeof sandbox.PlayGta6PipelineCache.WebGpuPipelineCache.prototype.trimTo,
    'function'
  );
  assert.equal(
    typeof sandbox.PlayGta6PipelineCache.WebGpuPipelineCache.prototype.evictLowPriority,
    'function'
  );
  assert.equal(
    typeof sandbox.PlayGta6PipelineCache.WebGpuPipelineCache.prototype.getCompileLatencyStats,
    'function'
  );
});

// --- W379: Quarantine Reason Breakdown, Drawn-First Forwarding & Reflection v3 Integration ---

test('W379 tracks sanitized quarantine reason breakdown, skips device_lost, supports single-key clearQuarantine, and preserves LRU bounds', async () => {
  assert.equal(classifyQuarantineReason(new Error('WGSL parse failure in C:\\Users\\bob\\s.wgsl')), PIPELINE_FAILURE_REASON.WGSL_COMPILE);
  assert.equal(classifyQuarantineReason(new Error('bind group layout mismatch')), PIPELINE_FAILURE_REASON.LAYOUT_MISMATCH);
  assert.equal(classifyQuarantineReason(new Error('GPUDevice was lost')), PIPELINE_FAILURE_REASON.DEVICE_LOST);

  let mode = 'wgsl';
  let renderCalls = 0;
  const device = createMockGpuDevice({
    renderSyncImpl: () => {
      renderCalls += 1;
      if (mode === 'wgsl') throw new Error('Tint WGSL compile failed at /home/user/shader.wgsl');
      if (mode === 'layout') throw new Error('pipeline layout mismatch at group 0');
      if (mode === 'lost') throw Object.assign(new Error('GPUDevice was lost'), { name: 'GPUDeviceLostError' });
      if (mode === 'null') return null;
      return { kind: 'GPURenderPipeline', ok: true };
    },
  });

  const cache = new WebGpuPipelineCache({ device, maxQuarantinedKeys: 2 });
  const vs = device.createShaderModule({ label: 'wgpk:vertex:0x0000000000003791', code: 'vs' });
  const dWgsl = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_wgsl' } };
  const dLayout = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_layout' } };
  const dLost = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_lost' } };
  const dNull = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_null' } };

  assert.throws(() => cache.getOrCreateRenderPipeline(dWgsl), /WGSL/);
  mode = 'layout';
  assert.throws(() => cache.getOrCreateRenderPipeline(dLayout), /layout/);
  mode = 'lost';
  assert.throws(() => cache.getOrCreateRenderPipeline(dLost), /lost/);

  assert.equal(cache.isQuarantined(dWgsl), true);
  assert.equal(cache.isQuarantined(dLayout), true);
  assert.equal(cache.isQuarantined(dLost), false, 'device_lost is not quarantined');

  // Hit dWgsl so it becomes MRU before adding dNull (evicting dLayout at maxQuarantinedKeys=2).
  assert.equal(cache.getOrCreateRenderPipeline(dWgsl), null);
  assert.equal(renderCalls, 3);

  mode = 'null';
  assert.equal(cache.getOrCreateRenderPipeline(dNull), null);
  assert.equal(cache.isQuarantined(dLayout), false, 'LRU quarantined entry evicted');
  assert.equal(cache.isQuarantined(dWgsl), true);
  assert.equal(cache.isQuarantined(dNull), true);

  const summary = cache.getQuarantineSummary();
  assert.equal(summary.totalQuarantined, 2);
  assert.equal(summary.quarantineHits, 1);
  assert.equal(summary.evictions, 1);
  assert.equal(summary.byReason.wgsl_compile, 1);
  assert.equal(summary.byReason.null_pipeline, 1);
  assert.equal(summary.byReason.layout_mismatch, 0);
  assert.equal(summary.hitsByReason.wgsl_compile, 1);
  assert.equal(summary.failuresByReason.device_lost, 1);
  assert.doesNotMatch(JSON.stringify(summary), /\/home\/user|shader\.wgsl/);

  // Non-enumerable stats.quarantinedByReason matches summary while keeping JSON stats clean.
  const stats = cache.getStats();
  assert.deepEqual(stats.quarantinedByReason, summary.byReason);

  // Single-key clearQuarantine removes only the requested descriptor.
  assert.equal(cache.clearQuarantine(dWgsl), 1);
  assert.equal(cache.isQuarantined(dWgsl), false);
  assert.equal(cache.isQuarantined(dNull), true);

  mode = 'ok';
  const recovered = cache.getOrCreateRenderPipeline(dWgsl);
  assert.equal(recovered.ok, true);
});

test('W379 forwards drawn-first priority promotions to paired AsyncPipelineCache and live prewarm queues', async () => {
  const asyncCalls = [];
  const asyncDevice = {
    createRenderPipelineAsync(desc) {
      let resolve;
      const promise = new Promise((r) => {
        resolve = r;
      });
      asyncCalls.push({ desc, resolve });
      return promise;
    },
    createComputePipelineAsync(desc) {
      return Promise.resolve({ kind: 'GPUComputePipeline', label: desc?.label });
    },
  };
  const asyncCache = new AsyncPipelineCache({ device: asyncDevice, maxConcurrentCompiles: 1 });

  // Occupy single async slot, queue 2 background compiles.
  const a0 = asyncCache.prewarm('slot_0', { label: 'slot_0' }, { priority: PIPELINE_PRIORITY_TIER.BACKGROUND });
  const a1 = asyncCache.prewarm('bg_a', { label: 'bg_a' }, { canonicalKey: 'pso-bg-a', priority: PIPELINE_PRIORITY_TIER.BACKGROUND });
  const a2 = asyncCache.prewarm('bg_b', { label: 'bg_b' }, { canonicalKey: 'pso-bg-b', priority: PIPELINE_PRIORITY_TIER.BACKGROUND });

  const deferreds = [];
  const prewarmOrder = [];
  const syncDevice = createMockGpuDevice({
    renderAsyncImpl: (desc) => {
      prewarmOrder.push(desc.label);
      const d = createDeferred();
      deferreds.push({ label: desc.label, d });
      return d.promise;
    },
  });
  const cache = new WebGpuPipelineCache({ device: syncDevice, asyncPipelineCache: asyncCache });

  // Promoting 'pso-bg-b' via WebGpuPipelineCache forwards directly to AsyncPipelineCache!
  assert.equal(cache.promoteOnDraw('pso-bg-b', 'render'), true);
  assert.equal(asyncCache.getTelemetry().queuedByPriority.critical, 1);

  // Finish asyncCache compiles and verify bg_b preempted bg_a.
  for (let i = 0; i < 3; i += 1) {
    asyncCalls[i].resolve({ kind: 'GPURenderPipeline', label: asyncCalls[i].desc.label });
    await new Promise((r) => setImmediate(r));
  }
  await Promise.all([a0, a1, a2]);
  assert.deepEqual(asyncCalls.map((c) => c.desc.label), ['slot_0', 'bg_b', 'bg_a']);

  // Also test live drawn-first preemption inside prewarmRenderPipelines(concurrency=1)!
  const vs = syncDevice.createShaderModule({ label: 'wgpk:vertex:0x0000000000003792', code: 'vs' });
  const d0 = { label: 'pw_0', layout: 'auto', vertex: { module: vs, entryPoint: 'vs0' } };
  const d1 = { label: 'pw_1', layout: 'auto', vertex: { module: vs, entryPoint: 'vs1' } };
  const d2 = { label: 'pw_2', layout: 'auto', vertex: { module: vs, entryPoint: 'vs2' } };

  const prewarmPromise = cache.prewarmRenderPipelines(
    [
      { descriptor: d0, priority: 'low' },
      { descriptor: d1, priority: 'low' },
      { descriptor: d2, priority: 'low' },
    ],
    { concurrency: 1 }
  );

  // pw_0 is active; promote d2 via markDrawn so it jumps ahead of pw_1!
  assert.equal(prewarmOrder.length, 1);
  assert.equal(prewarmOrder[0], 'pw_0');
  assert.equal(cache.markDrawn(d2), true);

  deferreds[0].d.resolve({ kind: 'GPURenderPipeline', label: 'pw_0' });
  await new Promise((r) => setImmediate(r));
  assert.equal(prewarmOrder[1], 'pw_2', 'drawn pipeline preempted queued prewarm pipeline');
  deferreds[1].d.resolve({ kind: 'GPURenderPipeline', label: 'pw_2' });
  await new Promise((r) => setImmediate(r));
  deferreds[2].d.resolve({ kind: 'GPURenderPipeline', label: 'pw_1' });
  const summary = await prewarmPromise;
  assert.equal(summary.compiled, 3);
  assert.deepEqual(prewarmOrder, ['pw_0', 'pw_2', 'pw_1']);
});

test('W379 integrates stage reflection v3 metadata (unresolved, lane_indexed, cbv_indexed, inline_sampler, rt_arena, mesh_output)', () => {
  const device = createMockGpuDevice();
  const cache = new WebGpuPipelineCache({ device });

  const sidecar = parseStageReflectionSidecar({
    schema: 'playgta6-stage-reflection',
    version: 3,
    stages: {
      '0x000000000000379a': {
        stage: 'fragment',
        rt_arena: true,
        mesh_output: { max_vertices: 64, max_primitives: 126, vertex_words: 12 },
        bindings: [
          {
            group: 0,
            binding: 0,
            kind: 'uniform',
            min_binding_size_bytes: 64,
            source: { user_sgpr: 16, root_descriptor: true, table_byte_offset: 32 },
          },
          {
            group: 0,
            binding: 1,
            kind: 'texture',
            source: { unresolved: 'dynamic_cb_index' },
          },
          {
            group: 0,
            binding: 2,
            kind: 'sampler',
            source: { inline_sampler: { sdwords: [146, 15728640, 139460608, 0] } },
          },
          {
            group: 0,
            binding: 3,
            kind: 'texture',
            source: {
              lane_indexed_resource_heap: {
                max_count: 64,
                shift: 5,
                window_base: { cbv: { user_sgpr: 18, root_descriptor: true }, byte_offset: 16 },
              },
            },
          },
          {
            group: 3,
            binding: 16,
            source: { rt_arena: true },
          },
        ],
      },
    },
  });

  const stage = sidecar.getStage('0x000000000000379a');
  const bgl = cache.createBindGroupLayoutFromReflection(
    { bindings: stage.bindings, hasInlineSampler: stage.hasInlineSampler },
    undefined,
    { bindingArrays: true }
  );
  assert.ok(bgl);
  assert.equal(cache.getStats().inlineSamplerLayouts, 1);

  const createdDesc = device.calls.createBindGroupLayout[0].desc;
  assert.equal(createdDesc.entries.length, 4);
  assert.equal(createdDesc.entries[3].binding, 3);
  assert.equal(createdDesc.entries[3].bindingArraySize, 64);

  // Fallback sampler synthesis handles v3 stage binding records cleanly (including unresolved/inline samplers).
  const synth = cache.synthesizeFallbackSamplerEntries({ bindings: stage.bindings }, []);
  assert.equal(synth.synthesizedCount, 1);
  assert.equal(synth.synthesized[0].binding, 2);
});
