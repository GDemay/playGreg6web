import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const pipelineCacheMod = require('../web/workers/pipeline_cache.js');
const {
  SHADER_STAGE_BITS,
  hashDescriptorKey,
  registerShaderModuleId,
  WebGpuPipelineCache,
} = pipelineCacheMod;

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
  advanceClockOnCompile = null,
  renderAsyncImpl = null,
  computeAsyncImpl = null,
} = {}) {
  let nextId = 1;
  const calls = {
    createRenderPipeline: [],
    createRenderPipelineAsync: [],
    createComputePipeline: [],
    createComputePipelineAsync: [],
    createBindGroupLayout: [],
    createShaderModule: [],
  };

  const device = {
    calls,
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
      const layout = { kind: 'GPUBindGroupLayout', id, desc };
      calls.createBindGroupLayout.push({ id, desc, layout });
      return layout;
    },
    createRenderPipeline(desc) {
      if (advanceClockOnCompile) advanceClockOnCompile(4.0);
      const id = nextId++;
      const pipeline = { kind: 'GPURenderPipeline', id, desc };
      calls.createRenderPipeline.push({ id, desc, pipeline });
      return pipeline;
    },
    createComputePipeline(desc) {
      if (advanceClockOnCompile) advanceClockOnCompile(2.5);
      const id = nextId++;
      const pipeline = { kind: 'GPUComputePipeline', id, desc };
      calls.createComputePipeline.push({ id, desc, pipeline });
      return pipeline;
    },
  };

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
    renderHits: 2,
    renderMisses: 2,
    computeHits: 1,
    computeMisses: 1,
    layoutHits: 1,
    layoutMisses: 1,
    inFlightDedupes: 0,
    evictions: 0,
    renderCount: 2,
    computeCount: 1,
    layoutCount: 1,
    totalCompileMs: 1.5 + 4.0 + 2.5 + 4.0,
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

  const cache = new WebGpuPipelineCache({ device });
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
    renderHits: 0,
    renderMisses: 0,
    computeHits: 0,
    computeMisses: 0,
    layoutHits: 0,
    layoutMisses: 0,
    inFlightDedupes: 0,
    evictions: 0,
    renderCount: 0,
    computeCount: 0,
    layoutCount: 0,
    totalCompileMs: 0,
  });
});
