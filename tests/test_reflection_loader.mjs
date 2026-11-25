import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const reflectionMod = require('../web/workers/reflection_loader.js');
const pipelineCacheMod = require('../web/workers/pipeline_cache.js');

const {
  LAYOUT_MANIFEST_SCHEMA_VERSION,
  MAX_HEAP_INDEX_SHIFT,
  SHADER_STAGE_BITS,
  VERTEX_FORMAT_BYTE_SIZES,
  ReflectionError,
  ManifestValidationError,
  BindingConflictError,
  ShaderNotFoundError,
  VertexLayoutError,
  normalizeShaderHash,
  formatShaderHash,
  computeBindGroupLayoutHash,
  computePipelineLayoutHash,
  parseBindingSource,
  parseReflectionManifest,
  buildBindGroupLayoutDescriptors,
  buildVertexBufferLayouts,
  ReflectionRegistry,
} = reflectionMod;

const { WebGpuPipelineCache } = pipelineCacheMod;

// Exact S6 manifest fixture matching shaders/src/reflect.rs `VERTEX_WGSL` + `FRAGMENT_WGSL`
// + `COMPUTE_A` (pinned BGL hash 0x472f25790659490f from reflect.rs:3075) + incomplete shader.
function createS6ManifestFixture() {
  const bglEmpty = { entries: [] };
  const bglVsG0 = {
    entries: [
      {
        binding: 0,
        visibility: 1,
        buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: 64 },
      },
      {
        binding: 1,
        visibility: 1,
        texture: { sampleType: 'unfilterable-float', viewDimension: '2d', multisampled: false },
      },
    ],
  };
  const bglVsG2 = {
    entries: [
      {
        binding: 0,
        visibility: 1,
        buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: 16 },
      },
    ],
  };
  const bglFsG0 = {
    entries: [
      {
        binding: 1,
        visibility: 2,
        texture: { sampleType: 'float', viewDimension: '2d', multisampled: false },
      },
      {
        binding: 2,
        visibility: 2,
        sampler: { type: 'filtering' },
      },
    ],
  };
  const bglFsG2 = {
    entries: [
      {
        binding: 0,
        visibility: 2,
        buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: 32 },
      },
    ],
  };
  const bglMergedG0 = {
    entries: [
      {
        binding: 0,
        visibility: 1,
        buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: 64 },
      },
      {
        binding: 1,
        visibility: 3,
        texture: { sampleType: 'float', viewDimension: '2d', multisampled: false },
      },
      {
        binding: 2,
        visibility: 2,
        sampler: { type: 'filtering' },
      },
    ],
  };
  const bglMergedG2 = {
    entries: [
      {
        binding: 0,
        visibility: 3,
        buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: 32 },
      },
    ],
  };
  const bglCompute = {
    entries: [
      {
        binding: 0,
        visibility: 4,
        buffer: { type: 'storage', hasDynamicOffset: false, minBindingSize: 4 },
      },
    ],
  };
  const bglComputeStorageTex = {
    entries: [
      {
        binding: 0,
        visibility: 4,
        storageTexture: { access: 'write-only', format: 'rgba8unorm', viewDimension: '2d' },
      },
    ],
  };
  const bglConflictFsG0 = {
    entries: [
      {
        binding: 0,
        visibility: 2,
        texture: { sampleType: 'unfilterable-float', viewDimension: '2d', multisampled: false },
      },
    ],
  };

  const idEmpty = computeBindGroupLayoutHash(bglEmpty);
  const idVsG0 = computeBindGroupLayoutHash(bglVsG0);
  const idVsG2 = computeBindGroupLayoutHash(bglVsG2);
  const idFsG0 = computeBindGroupLayoutHash(bglFsG0);
  const idFsG2 = computeBindGroupLayoutHash(bglFsG2);
  const idMergedG0 = computeBindGroupLayoutHash(bglMergedG0);
  const idMergedG2 = computeBindGroupLayoutHash(bglMergedG2);
  const idCompute = computeBindGroupLayoutHash(bglCompute);
  const idComputeStorageTex = computeBindGroupLayoutHash(bglComputeStorageTex);
  const idConflictFsG0 = computeBindGroupLayoutHash(bglConflictFsG0);

  const plVs = computePipelineLayoutHash([idVsG0, idEmpty, idVsG2]);
  const plFs = computePipelineLayoutHash([idFsG0, idEmpty, idFsG2]);
  const plMerged = computePipelineLayoutHash([idMergedG0, idEmpty, idMergedG2]);
  const plCompute = computePipelineLayoutHash([idCompute]);
  const plComputeIncomplete = computePipelineLayoutHash([idCompute, idComputeStorageTex]);
  const plConflictFs = computePipelineLayoutHash([idConflictFsG0]);

  return {
    schema_version: 1,
    register_convention: { source: 'dxc-spirv', shifts: { b: 0, t: 0, u: 0, s: 0 } },
    bind_group_layouts: {
      [idEmpty]: bglEmpty,
      [idVsG0]: bglVsG0,
      [idVsG2]: bglVsG2,
      [idFsG0]: bglFsG0,
      [idFsG2]: bglFsG2,
      [idMergedG0]: bglMergedG0,
      [idMergedG2]: bglMergedG2,
      [idCompute]: bglCompute,
      [idComputeStorageTex]: bglComputeStorageTex,
      [idConflictFsG0]: bglConflictFsG0,
    },
    pipeline_layouts: {
      [plVs]: { bindGroupLayouts: [idVsG0, idEmpty, idVsG2] },
      [plFs]: { bindGroupLayouts: [idFsG0, idEmpty, idFsG2] },
      [plMerged]: { bindGroupLayouts: [idMergedG0, idEmpty, idMergedG2] },
      [plCompute]: { bindGroupLayouts: [idCompute] },
      [plComputeIncomplete]: { bindGroupLayouts: [idCompute, idComputeStorageTex] },
      [plConflictFs]: { bindGroupLayouts: [idConflictFsG0] },
    },
    shaders: {
      '0x0000000000003333': {
        stage: 'vertex',
        entry_point: 'vs_main',
        pipeline_layout: plVs,
        bind_group_layouts: [idVsG0, idEmpty, idVsG2],
        layout_complete: true,
        workgroup_size: null,
        bindings: [
          {
            group: 0,
            binding: 0,
            name: 'camera',
            resource: 'buffer',
            in_layout: true,
            register: { class: 'b', index: 0, space: 0 },
            spirv: { set: 0, binding: 0 },
            root_constants: false,
            binding_array: null,
          },
          {
            group: 0,
            binding: 1,
            name: 'albedo',
            resource: 'texture',
            in_layout: true,
            register: { class: 't', index: 1, space: 0 },
            spirv: { set: 0, binding: 1 },
            root_constants: false,
            binding_array: null,
          },
          {
            group: 2,
            binding: 0,
            name: 'shared_small',
            resource: 'buffer',
            in_layout: true,
            register: { class: 'b', index: 0, space: 2 },
            spirv: { set: 2, binding: 0 },
            root_constants: false,
            binding_array: null,
          },
        ],
        vertex_inputs: [
          { location: 0, type: 'vec3<f32>', format: 'float32x3' },
          { location: 1, type: 'vec2<f32>', format: 'float32x2' },
          { location: 2, type: 'i32', format: 'sint32' },
          { location: 3, type: 'vec4<u32>', format: 'uint32x4' },
        ],
        fragment_outputs: [],
        requires_lowering: [],
        limit_violations: [],
      },
      '0x0000000000004444': {
        stage: 'fragment',
        entry_point: 'fs_main',
        pipeline_layout: plFs,
        bind_group_layouts: [idFsG0, idEmpty, idFsG2],
        layout_complete: true,
        workgroup_size: null,
        bindings: [
          {
            group: 0,
            binding: 1,
            name: 'albedo',
            resource: 'texture',
            in_layout: true,
            register: { class: 't', index: 1, space: 0 },
            spirv: { set: 0, binding: 1 },
            root_constants: false,
            binding_array: null,
          },
          {
            group: 0,
            binding: 2,
            name: 'samp',
            resource: 'sampler',
            in_layout: true,
            register: { class: 's', index: 2, space: 0 },
            spirv: { set: 0, binding: 2 },
            root_constants: false,
            binding_array: null,
          },
          {
            group: 2,
            binding: 0,
            name: 'shared_big',
            resource: 'buffer',
            in_layout: true,
            register: { class: 'b', index: 0, space: 2 },
            spirv: { set: 2, binding: 0 },
            root_constants: false,
            binding_array: null,
          },
        ],
        vertex_inputs: [],
        fragment_outputs: [
          { location: 0, type: 'vec4<f32>' },
          { location: 1, type: 'vec2<u32>' },
        ],
        requires_lowering: [],
        limit_violations: [],
      },
      '0x0000000000005555': {
        stage: 'compute',
        entry_point: 'cs_main',
        pipeline_layout: plCompute,
        bind_group_layouts: [idCompute],
        layout_complete: true,
        workgroup_size: [64, 1, 1],
        bindings: [
          {
            group: 0,
            binding: 0,
            name: 'data',
            resource: 'buffer',
            in_layout: true,
            register: { class: 'u', index: 0, space: 0 },
            spirv: { set: 0, binding: 0 },
            root_constants: false,
            binding_array: null,
          },
        ],
        vertex_inputs: [],
        fragment_outputs: [],
        requires_lowering: [],
        limit_violations: [],
      },
      '0x0000000000006666': {
        stage: 'compute',
        entry_point: 'cs_incomplete',
        pipeline_layout: plComputeIncomplete,
        bind_group_layouts: [idCompute, idComputeStorageTex],
        layout_complete: false,
        workgroup_size: [8, 4, 2],
        bindings: [
          {
            group: 0,
            binding: 0,
            name: 'out_buf',
            resource: 'buffer',
            in_layout: true,
            register: { class: 'u', index: 0, space: 0 },
            spirv: { set: 0, binding: 0 },
            root_constants: false,
            binding_array: null,
          },
          {
            group: 1,
            binding: 0,
            name: 'st_w',
            resource: 'storageTexture',
            in_layout: true,
            register: { class: 'u', index: 0, space: 1 },
            spirv: { set: 1, binding: 0 },
            root_constants: false,
            binding_array: null,
          },
          {
            group: 1,
            binding: 1,
            name: 'bindless_tex',
            resource: 'texture',
            in_layout: false,
            register: { class: 't', index: 1, space: 1 },
            spirv: { set: 1, binding: 1 },
            root_constants: false,
            binding_array: { count: 4 },
          },
        ],
        vertex_inputs: [],
        fragment_outputs: [],
        requires_lowering: [
          {
            kind: 'BindingArray',
            group: 1,
            binding: 1,
            detail: "'bindless_tex': binding_array (4 elements) has no baseline WebGPU layout",
          },
        ],
        limit_violations: [],
      },
      '0x0000000000007777': {
        stage: 'fragment',
        entry_point: 'fs_conflict',
        pipeline_layout: plConflictFs,
        bind_group_layouts: [idConflictFsG0],
        layout_complete: true,
        workgroup_size: null,
        bindings: [
          {
            group: 0,
            binding: 0,
            name: 'not_a_buffer',
            resource: 'texture',
            in_layout: true,
            register: { class: 't', index: 0, space: 0 },
            spirv: { set: 0, binding: 0 },
            root_constants: false,
            binding_array: null,
          },
        ],
        vertex_inputs: [],
        fragment_outputs: [{ location: 0, type: 'vec4<f32>' }],
        requires_lowering: [],
        limit_violations: [],
      },
    },
    pipelines: {
      '0x0000000000003333+0x0000000000004444': {
        stages: {
          vertex: '0x0000000000003333',
          fragment: '0x0000000000004444',
        },
        pipeline_layout: plMerged,
        bind_group_layouts: [idMergedG0, idEmpty, idMergedG2],
        layout_complete: true,
        limit_violations: [],
      },
    },
  };
}

function createMockGpuDevice() {
  let nextId = 1;
  const calls = {
    createBindGroupLayout: [],
    createPipelineLayout: [],
  };

  return {
    calls,
    createBindGroupLayout(desc) {
      const id = nextId++;
      const layout = { kind: 'GPUBindGroupLayout', id, desc };
      calls.createBindGroupLayout.push({ id, desc, layout });
      return layout;
    },
    createPipelineLayout(desc) {
      const id = nextId++;
      const pipelineLayout = { kind: 'GPUPipelineLayout', id, desc };
      calls.createPipelineLayout.push({ id, desc, pipelineLayout });
      return pipelineLayout;
    },
  };
}

test('UMD module exposes globalThis.PlayGta6ReflectionLoader and expected exports', () => {
  assert.equal(globalThis.PlayGta6ReflectionLoader, reflectionMod);
  assert.equal(LAYOUT_MANIFEST_SCHEMA_VERSION, 1);
  assert.equal(SHADER_STAGE_BITS.VERTEX, 0x1);
  assert.equal(SHADER_STAGE_BITS.FRAGMENT, 0x2);
  assert.equal(SHADER_STAGE_BITS.COMPUTE, 0x4);
  assert.equal(typeof parseReflectionManifest, 'function');
  assert.equal(typeof ReflectionRegistry, 'function');
  assert.equal(typeof buildBindGroupLayoutDescriptors, 'function');
  assert.equal(typeof buildVertexBufferLayouts, 'function');
  assert.equal(VERTEX_FORMAT_BYTE_SIZES.float32x3, 12);
});

test('computeBindGroupLayoutHash matches pinned S6 FNV-1a content hash from shaders/src/reflect.rs', () => {
  // Pinned in shaders/src/reflect.rs line 3075:
  // [{"binding":0,"visibility":4,"buffer":{"type":"storage","hasDynamicOffset":false,"minBindingSize":4}}]
  // -> "0x472f25790659490f"
  const hash = computeBindGroupLayoutHash({
    entries: [
      {
        binding: 0,
        visibility: SHADER_STAGE_BITS.COMPUTE,
        buffer: { type: 'storage', hasDynamicOffset: false, minBindingSize: 4 },
      },
    ],
  });
  assert.equal(hash, '0x472f25790659490f');
});

test('parseReflectionManifest parses exact S6 JSON string and object, normalizing hashes and indices', () => {
  const fixture = createS6ManifestFixture();
  const parsedFromString = parseReflectionManifest(JSON.stringify(fixture));
  const parsedFromObj = parseReflectionManifest(fixture);

  assert.equal(parsedFromString.schemaVersion, 1);
  assert.equal(parsedFromString.shaders.size, 5);
  assert.equal(parsedFromObj.shadersByHash.size, 5);
  assert.equal(parsedFromString.pipelines.size, 1);

  const vs = parsedFromString.shadersByHash.get(0x3333n);
  assert.ok(vs);
  assert.equal(vs.hash, 0x3333n);
  assert.equal(vs.hashHex, '0x0000000000003333');
  assert.equal(vs.stage, 'vertex');
  assert.equal(vs.visibility, SHADER_STAGE_BITS.VERTEX);
  assert.equal(vs.entryPoint, 'vs_main');
  assert.equal(vs.layoutComplete, true);
  assert.equal(vs.bindGroupLayouts.length, 3);
  assert.equal(vs.bindGroupLayouts[0].entries.length, 2);
  assert.equal(vs.bindGroupLayouts[1].entries.length, 0);
  assert.equal(vs.bindGroupLayouts[2].entries.length, 1);

  // Verify S6 binding metadata cross-linked with layout entry
  assert.equal(vs.bindings[0].name, 'camera');
  assert.deepEqual(vs.bindings[0].register, { class: 'b', index: 0, space: 0 });
  assert.deepEqual(vs.bindings[0].layoutEntry, {
    binding: 0,
    visibility: 1,
    buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: 64 },
  });

  // Incomplete shader preserves requires_lowering and in_layout: false
  const incomplete = parsedFromString.shadersByHash.get(0x6666n);
  assert.equal(incomplete.layoutComplete, false);
  assert.deepEqual(incomplete.workgroupSize, [8, 4, 2]);
  assert.equal(incomplete.requiresLowering.length, 1);
  assert.equal(incomplete.requiresLowering[0].kind, 'BindingArray');
  const bindlessRec = incomplete.bindings.find((b) => b.name === 'bindless_tex');
  assert.equal(bindlessRec.inLayout, false);
  assert.deepEqual(bindlessRec.bindingArray, { count: 4 });
  assert.equal(bindlessRec.layoutEntry, null);

  // Hash normalization accepts 0x hex, uppercase hex, short hex, and BigInt
  assert.equal(normalizeShaderHash('0x0000000000003333'), 0x3333n);
  assert.equal(normalizeShaderHash('0x3333'), 0x3333n);
  assert.equal(normalizeShaderHash(0x3333n), 0x3333n);
  assert.equal(formatShaderHash(0x3333n), '0x0000000000003333');
  assert.equal(formatShaderHash('0xDEADBEEF'), '0x00000000deadbeef');
});

test('ReflectionRegistry loads manifests via object and URL and supports hash/BigInt/name lookup', async () => {
  const fixture = createS6ManifestFixture();
  const registry = new ReflectionRegistry();

  // loadManifestUrl using mock fetch
  const fetchCalls = [];
  const mockFetch = async (url) => {
    fetchCalls.push(url);
    return {
      ok: true,
      status: 200,
      statusText: 'OK',
      text: async () => JSON.stringify(fixture),
    };
  };

  await registry.loadManifestUrl('https://local.test/layout-manifest.json', mockFetch);
  assert.deepEqual(fetchCalls, ['https://local.test/layout-manifest.json']);

  // Lookup by full hex, short hex, uppercase hex, BigInt, number, entry_point, and stage:entry_point
  const byFullHex = registry.getShaderReflection('0x0000000000003333');
  const byShortHex = registry.getShaderReflection('0x3333');
  const byBigInt = registry.getShaderReflection(0x3333n);
  const byNumber = registry.getShaderReflection(0x3333);
  const byName = registry.getShaderReflection('vs_main');
  const byStageName = registry.getShaderReflection('vertex:vs_main');

  assert.ok(byFullHex);
  assert.equal(byShortHex, byFullHex);
  assert.equal(byBigInt, byFullHex);
  assert.equal(byNumber, byFullHex);
  assert.equal(byName, byFullHex);
  assert.equal(byStageName, byFullHex);
  assert.equal(registry.getShaderReflection('0xdeadbeef'), undefined);
  assert.equal(registry.getShaderReflection('nonexistent_shader'), undefined);

  // Pre-merged pipeline lookup works regardless of stage order
  const pipe = registry.getPipelineReflection([0x4444n, '0x3333']);
  assert.ok(pipe);
  assert.equal(pipe.key, '0x0000000000003333+0x0000000000004444');
  assert.equal(pipe.layoutComplete, true);

  // HTTP error handling in loadManifestUrl
  await assert.rejects(
    registry.loadManifestUrl('https://local.test/missing.json', async () => ({
      ok: false,
      status: 404,
      statusText: 'Not Found',
    })),
    /HTTP 404 Not Found/
  );
});

test('buildBindGroupLayoutDescriptors merges multi-stage VS+FS bindings with OR-ed visibility and S6 promotion rules', () => {
  const registry = new ReflectionRegistry();
  registry.loadManifest(createS6ManifestFixture());

  // Order of stages (VS+FS vs FS+VS) produces identical descriptors
  const mergedVsFs = registry.buildBindGroupLayoutDescriptors([0x3333n, '0x0000000000004444']);
  const mergedFsVs = registry.buildBindGroupLayoutDescriptors(['fs_main', 'vs_main']);
  assert.deepEqual(mergedVsFs, mergedFsVs);

  // Groups 0..2 densely populated (group 1 is empty hole)
  assert.equal(mergedVsFs.length, 3);
  assert.deepEqual(mergedVsFs[0], {
    entries: [
      {
        binding: 0,
        visibility: SHADER_STAGE_BITS.VERTEX,
        buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: 64 },
      },
      {
        binding: 1,
        visibility: SHADER_STAGE_BITS.VERTEX | SHADER_STAGE_BITS.FRAGMENT,
        // VS unfilterable-float + FS float promotes to float (matching shaders/src/reflect.rs)
        texture: { sampleType: 'float', viewDimension: '2d', multisampled: false },
      },
      {
        binding: 2,
        visibility: SHADER_STAGE_BITS.FRAGMENT,
        sampler: { type: 'filtering' },
      },
    ],
  });
  assert.deepEqual(mergedVsFs[1], { entries: [] });
  assert.deepEqual(mergedVsFs[2], {
    entries: [
      {
        binding: 0,
        visibility: SHADER_STAGE_BITS.VERTEX | SHADER_STAGE_BITS.FRAGMENT,
        // VS minBindingSize 16 + FS minBindingSize 32 takes max = 32
        buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: 32 },
      },
    ],
  });

  // Verify computed merged BGL hashes match the S6 pre-merged pipeline entry
  const pipe = registry.getPipelineReflection([0x3333n, 0x4444n]);
  assert.deepEqual(
    mergedVsFs.map((d) => computeBindGroupLayoutHash(d)),
    pipe.bindGroupLayoutIds
  );

  // Incomplete layout rejection when requireLayoutComplete=true
  assert.throws(
    () => registry.buildBindGroupLayoutDescriptors([0x6666n], { requireLayoutComplete: true }),
    (err) => err instanceof ReflectionError && err.code === 'INCOMPLETE_LAYOUT'
  );
  // Unknown shader throws typed ShaderNotFoundError
  assert.throws(
    () => registry.buildBindGroupLayoutDescriptors(['0x9999']),
    (err) => err instanceof ShaderNotFoundError && err.code === 'SHADER_NOT_FOUND'
  );
});

test('buildBindGroupLayoutDescriptors rejects conflicting resource types and mismatched parameters at same (group, binding)', () => {
  const registry = new ReflectionRegistry();
  registry.loadManifest(createS6ManifestFixture());

  // VS has uniform buffer at (0,0); fs_conflict (0x7777) has texture_2d at (0,0)
  assert.throws(
    () => registry.buildBindGroupLayoutDescriptors([0x3333n, 0x7777n]),
    (err) => {
      assert.ok(err instanceof BindingConflictError);
      assert.equal(err.code, 'BINDING_CONFLICT');
      assert.equal(err.group, 0);
      assert.equal(err.binding, 0);
      assert.match(err.message, /camera/);
      assert.match(err.message, /not_a_buffer/);
      assert.match(err.message, /buffer vs texture/);
      return true;
    }
  );

  // Buffer type mismatch (uniform vs storage)
  const vsUniform = {
    stage: 'vertex',
    entryPoint: 'vs_u',
    bindings: [
      {
        group: 0,
        binding: 0,
        name: 'ub',
        buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: 16 },
      },
    ],
  };
  const fsStorage = {
    stage: 'fragment',
    entryPoint: 'fs_s',
    bindings: [
      {
        group: 0,
        binding: 0,
        name: 'sb',
        buffer: { type: 'storage', hasDynamicOffset: false, minBindingSize: 16 },
      },
    ],
  };
  assert.throws(
    () => buildBindGroupLayoutDescriptors([vsUniform, fsStorage]),
    (err) => err instanceof BindingConflictError && /buffer type mismatch/.test(err.message)
  );

  // Sampler type mismatch (filtering vs comparison)
  assert.throws(
    () =>
      buildBindGroupLayoutDescriptors([
        {
          stage: 'vertex',
          entryPoint: 'vs_smp',
          bindings: [{ group: 0, binding: 1, sampler: { type: 'filtering' } }],
        },
        {
          stage: 'fragment',
          entryPoint: 'fs_smp',
          bindings: [{ group: 0, binding: 1, sampler: { type: 'comparison' } }],
        },
      ]),
    (err) => err instanceof BindingConflictError && /sampler type mismatch/.test(err.message)
  );

  // Texture sampleType mismatch (float vs sint)
  assert.throws(
    () =>
      buildBindGroupLayoutDescriptors([
        {
          stage: 'vertex',
          entryPoint: 'vs_tex',
          bindings: [
            { group: 1, binding: 2, texture: { sampleType: 'float', viewDimension: '2d', multisampled: false } },
          ],
        },
        {
          stage: 'fragment',
          entryPoint: 'fs_tex',
          bindings: [
            { group: 1, binding: 2, texture: { sampleType: 'sint', viewDimension: '2d', multisampled: false } },
          ],
        },
      ]),
    (err) => err instanceof BindingConflictError && err.group === 1 && err.binding === 2
  );

  // Group index >= 4 rejected under default WebGPU maxBindGroups=4
  assert.throws(
    () =>
      buildBindGroupLayoutDescriptors([
        {
          stage: 'compute',
          entryPoint: 'cs_g4',
          bindings: [
            { group: 4, binding: 0, buffer: { type: 'storage', hasDynamicOffset: false, minBindingSize: 4 } },
          ],
        },
      ]),
    (err) => err instanceof ReflectionError && err.code === 'GROUP_INDEX_OUT_OF_RANGE'
  );

  assert.equal(registry.getStats().bindingConflicts, 1);
});

test('buildPipelineLayout integrates with real WebGpuPipelineCache and deduplicates BindGroupLayouts', () => {
  const device = createMockGpuDevice();
  const pipelineCache = new WebGpuPipelineCache({ device });
  const registry = new ReflectionRegistry();
  registry.loadManifest(createS6ManifestFixture());

  // First build for VS+FS creates 3 BindGroupLayouts (group 0, empty group 1, group 2)
  const pl1 = registry.buildPipelineLayout(device, pipelineCache, [0x3333n, 0x4444n], {
    label: 'pbr_pipeline_layout',
  });
  assert.equal(pl1.kind, 'GPUPipelineLayout');
  assert.equal(pl1.desc.label, 'pbr_pipeline_layout');
  assert.equal(pl1.desc.bindGroupLayouts.length, 3);
  assert.equal(device.calls.createBindGroupLayout.length, 3);
  assert.equal(device.calls.createPipelineLayout.length, 1);

  // Second build with reversed stage order hits ALL 3 cached BindGroupLayouts in WebGpuPipelineCache!
  const pl2 = registry.buildPipelineLayout(device, pipelineCache, ['fs_main', 'vs_main']);
  assert.equal(device.calls.createBindGroupLayout.length, 3);
  assert.equal(device.calls.createPipelineLayout.length, 2);
  assert.equal(pl2.desc.bindGroupLayouts[0], pl1.desc.bindGroupLayouts[0]);
  assert.equal(pl2.desc.bindGroupLayouts[1], pl1.desc.bindGroupLayouts[1]);
  assert.equal(pl2.desc.bindGroupLayouts[2], pl1.desc.bindGroupLayouts[2]);

  const cacheStats = pipelineCache.getStats();
  assert.equal(cacheStats.layoutMisses, 3);
  assert.equal(cacheStats.layoutHits, 3);
  assert.equal(cacheStats.layoutCount, 3);

  // Fallback without pipelineCache calls device.createBindGroupLayout directly
  const directDevice = createMockGpuDevice();
  const plDirect = registry.buildPipelineLayout(directDevice, null, [0x5555n]);
  assert.equal(plDirect.kind, 'GPUPipelineLayout');
  assert.equal(plDirect.desc.bindGroupLayouts.length, 1);
  assert.equal(directDevice.calls.createBindGroupLayout.length, 1);
  assert.deepEqual(directDevice.calls.createBindGroupLayout[0].desc, {
    entries: [
      {
        binding: 0,
        visibility: SHADER_STAGE_BITS.COMPUTE,
        buffer: { type: 'storage', hasDynamicOffset: false, minBindingSize: 4 },
      },
    ],
  });

  const stats = registry.getStats();
  assert.equal(stats.manifestsLoaded, 1);
  assert.equal(stats.shaderCount, 5);
  assert.equal(stats.pipelineLayoutsBuilt, 3);
});

test('buildVertexBufferLayouts builds packed and explicit-stride GPUVertexBufferLayout descriptors', () => {
  const registry = new ReflectionRegistry();
  registry.loadManifest(createS6ManifestFixture());

  // Build directly by shader hash (vs_main has locations 0..3: float32x3, float32x2, sint32, uint32x4)
  // Packed byte sizes: 12 + 8 + 4 + 16 = 40 bytes
  const autoLayouts = registry.buildVertexBufferLayouts(0x3333n);
  assert.deepEqual(autoLayouts, [
    {
      arrayStride: 40,
      stepMode: 'vertex',
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'float32x2' },
        { shaderLocation: 2, offset: 20, format: 'sint32' },
        { shaderLocation: 3, offset: 24, format: 'uint32x4' },
      ],
    },
  ]);

  // Explicit padded stride (e.g., 64-byte aligned vertex) and instance stepMode
  const paddedLayouts = registry.buildVertexBufferLayouts('vs_main', 64, { stepMode: 'instance' });
  assert.equal(paddedLayouts[0].arrayStride, 64);
  assert.equal(paddedLayouts[0].stepMode, 'instance');
  assert.equal(paddedLayouts[0].attributes.length, 4);

  // Unsorted raw vertex_inputs array gets sorted by location; formatOverrides supported
  const overrideLayouts = buildVertexBufferLayouts(
    [
      { location: 2, type: 'vec4<f32>', format: 'float32x4' },
      { location: 0, type: 'vec3<f32>', format: 'float32x3' },
    ],
    { formatOverrides: { 2: 'unorm8x4' } }
  );
  assert.deepEqual(overrideLayouts, [
    {
      arrayStride: 16,
      stepMode: 'vertex',
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 2, offset: 12, format: 'unorm8x4' },
      ],
    },
  ]);

  // Shader with zero vertex inputs returns []
  assert.deepEqual(registry.buildVertexBufferLayouts(0x5555n), []);

  // Rejects explicit stride smaller than attributes
  assert.throws(
    () => registry.buildVertexBufferLayouts(0x3333n, 32),
    (err) => err instanceof VertexLayoutError && /smaller than required/.test(err.message)
  );

  // Rejects missing / null GPUVertexFormat (unless overridden)
  assert.throws(
    () => buildVertexBufferLayouts([{ location: 0, type: 'mat4x4<f32>', format: null }]),
    (err) => err instanceof VertexLayoutError && /unsupported or missing GPUVertexFormat/.test(err.message)
  );

  // Rejects duplicate locations
  assert.throws(
    () =>
      buildVertexBufferLayouts([
        { location: 1, format: 'float32x2' },
        { location: 1, format: 'float32x3' },
      ]),
    (err) => err instanceof VertexLayoutError && /Duplicate vertex input/.test(err.message)
  );
});

test('parseReflectionManifest rejects malformed JSON, bad schema versions, invalid hashes, and broken references', () => {
  // Invalid JSON syntax
  assert.throws(
    () => parseReflectionManifest('{ not valid json'),
    (err) => err instanceof ManifestValidationError && /malformed JSON/.test(err.message)
  );

  // Non-object root
  assert.throws(() => parseReflectionManifest([]), ManifestValidationError);
  assert.throws(() => parseReflectionManifest(null), ManifestValidationError);

  // Wrong schema_version
  assert.throws(
    () => parseReflectionManifest({ schema_version: 2, shaders: {} }),
    (err) => err instanceof ManifestValidationError && /unsupported schema_version/.test(err.message)
  );

  // Missing shaders object
  assert.throws(
    () => parseReflectionManifest({ schema_version: 1 }),
    (err) => err instanceof ManifestValidationError && /shaders must be an object/.test(err.message)
  );

  // Invalid shader hash key
  assert.throws(
    () =>
      parseReflectionManifest({
        schema_version: 1,
        bind_group_layouts: {},
        pipeline_layouts: {},
        shaders: {
          '0xnot_hex': { stage: 'vertex', entry_point: 'main' },
        },
      }),
    (err) => err instanceof ManifestValidationError && /invalid 64-bit hex hash/.test(err.message)
  );

  // Dangling bind_group_layout reference from shader
  assert.throws(
    () =>
      parseReflectionManifest({
        schema_version: 1,
        bind_group_layouts: {},
        pipeline_layouts: {},
        shaders: {
          '0x0000000000000001': {
            stage: 'vertex',
            entry_point: 'vs_main',
            bind_group_layouts: ['0x000000000000dead'],
          },
        },
      }),
    (err) => err instanceof ManifestValidationError && /unknown bind_group_layout/.test(err.message)
  );

  // Entry with two resource members (both buffer and sampler)
  assert.throws(
    () =>
      parseReflectionManifest({
        schema_version: 1,
        bind_group_layouts: {
          '0x0000000000000001': {
            entries: [
              {
                binding: 0,
                visibility: 1,
                buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: 16 },
                sampler: { type: 'filtering' },
              },
            ],
          },
        },
        shaders: {},
      }),
    (err) => err instanceof ManifestValidationError && /exactly one resource member/.test(err.message)
  );

  // Pipeline referencing an unknown shader
  assert.throws(
    () =>
      parseReflectionManifest({
        schema_version: 1,
        bind_group_layouts: {},
        pipeline_layouts: {},
        shaders: {},
        pipelines: {
          '0x0000000000000001': {
            stages: { compute: '0x0000000000000001' },
            bind_group_layouts: [],
            layout_complete: true,
          },
        },
      }),
    (err) => err instanceof ManifestValidationError && /unknown shader/.test(err.message)
  );
});

test('parseBindingSource parses and normalizes S34 sampler_heap, resource_heap, and legacy binding sources', () => {
  assert.equal(MAX_HEAP_INDEX_SHIFT, 31);
  assert.equal(parseBindingSource(undefined), null);
  assert.equal(parseBindingSource(null), null);

  // S34 sampler_heap & resource_heap matching shaders/src/rdna2_to_wgsl.rs & tools/stage_reflection_sidecar.py
  assert.deepEqual(
    parseBindingSource({
      sampler_heap: {
        base_user_sgpr: 11,
        stride: 16,
        index: { user_sgpr: 9, shift: 4 },
      },
    }),
    {
      kind: 'sampler_heap',
      baseUserSgpr: 11,
      stride: 16,
      indexUserSgpr: 9,
      indexShift: 4,
    }
  );

  assert.deepEqual(
    parseBindingSource({
      resource_heap: {
        base_user_sgpr: 4,
        stride: 32,
        index: { user_sgpr: 5, shift: 5 },
      },
    }),
    {
      kind: 'resource_heap',
      baseUserSgpr: 4,
      stride: 32,
      indexUserSgpr: 5,
      indexShift: 5,
    }
  );

  // Boundary values: 0 base/index/shift, stride 1, and max shift 31
  assert.deepEqual(
    parseBindingSource({
      sampler_heap: {
        base_user_sgpr: 0,
        stride: 1,
        index: { user_sgpr: 0, shift: 0 },
      },
    }),
    {
      kind: 'sampler_heap',
      baseUserSgpr: 0,
      stride: 1,
      indexUserSgpr: 0,
      indexShift: 0,
    }
  );
  assert.deepEqual(
    parseBindingSource({
      resource_heap: {
        base_user_sgpr: 15,
        stride: 64,
        index: { user_sgpr: 12, shift: 31 },
      },
    }),
    {
      kind: 'resource_heap',
      baseUserSgpr: 15,
      stride: 64,
      indexUserSgpr: 12,
      indexShift: 31,
    }
  );

  // Existing table, root_constants, root_descriptor, and static_sampler sources
  assert.deepEqual(parseBindingSource({ user_sgpr: 8, table_byte_offset: 32 }), {
    kind: 'table',
    userSgpr: 8,
    tableByteOffset: 32,
  });
  assert.deepEqual(parseBindingSource({ user_sgpr: 4, root_constants: true }), {
    kind: 'root_constants',
    userSgpr: 4,
  });
  assert.deepEqual(parseBindingSource({ user_sgpr: 11, root_descriptor: true }), {
    kind: 'root_descriptor',
    userSgpr: 11,
  });
  assert.deepEqual(parseBindingSource({ static_sampler: 0 }), {
    kind: 'static_sampler',
    index: 0,
  });
});

test('parseReflectionManifest and ReflectionRegistry preserve S34 heap sources and expose additive heap stats', () => {
  const bglG0 = {
    entries: [
      {
        binding: 0,
        visibility: SHADER_STAGE_BITS.FRAGMENT,
        buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: 64 },
      },
      {
        binding: 1,
        visibility: SHADER_STAGE_BITS.FRAGMENT,
        texture: { sampleType: 'float', viewDimension: '2d-array', multisampled: false },
      },
      {
        binding: 2,
        visibility: SHADER_STAGE_BITS.FRAGMENT,
        sampler: { type: 'filtering' },
      },
      {
        binding: 3,
        visibility: SHADER_STAGE_BITS.FRAGMENT,
        sampler: { type: 'comparison' },
      },
    ],
  };
  const bglCs = {
    entries: [
      {
        binding: 0,
        visibility: SHADER_STAGE_BITS.COMPUTE,
        buffer: { type: 'read-only-storage', hasDynamicOffset: false, minBindingSize: 16 },
      },
      {
        binding: 1,
        visibility: SHADER_STAGE_BITS.COMPUTE,
        buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: 16 },
      },
    ],
  };

  const idG0 = computeBindGroupLayoutHash(bglG0);
  const idCs = computeBindGroupLayoutHash(bglCs);
  const plFs = computePipelineLayoutHash([idG0]);
  const plCs = computePipelineLayoutHash([idCs]);

  const s34Manifest = {
    schema_version: 1,
    bind_group_layouts: {
      [idG0]: bglG0,
      [idCs]: bglCs,
    },
    pipeline_layouts: {
      [plFs]: { bindGroupLayouts: [idG0] },
      [plCs]: { bindGroupLayouts: [idCs] },
    },
    shaders: {
      '0x000000000000a001': {
        stage: 'fragment',
        entry_point: 'fs_s34_heaps',
        pipeline_layout: plFs,
        bind_group_layouts: [idG0],
        layout_complete: true,
        workgroup_size: null,
        bindings: [
          {
            group: 0,
            binding: 0,
            name: 'cb_frame',
            resource: 'buffer',
            in_layout: true,
            source: { user_sgpr: 8, table_byte_offset: 0 },
          },
          {
            group: 0,
            binding: 1,
            name: 'dyn_tex',
            resource: 'texture',
            in_layout: true,
            source: {
              resource_heap: {
                base_user_sgpr: 11,
                stride: 32,
                index: { user_sgpr: 10, shift: 5 },
              },
            },
          },
          {
            group: 0,
            binding: 2,
            name: 'dyn_smp',
            resource: 'sampler',
            in_layout: true,
            source: {
              sampler_heap: {
                base_user_sgpr: 11,
                stride: 16,
                index: { user_sgpr: 9, shift: 4 },
              },
            },
          },
          {
            group: 0,
            binding: 3,
            name: 'static_cmp',
            resource: 'sampler',
            in_layout: true,
            source: { static_sampler: 0 },
          },
        ],
        vertex_inputs: [],
        fragment_outputs: [{ location: 0, type: 'vec4<f32>' }],
      },
      '0x000000000000a002': {
        stage: 'compute',
        entry_point: 'cs_s34_resource_heap',
        pipeline_layout: plCs,
        bind_group_layouts: [idCs],
        layout_complete: true,
        workgroup_size: [64, 1, 1],
        bindings: [
          {
            group: 0,
            binding: 0,
            name: 'dyn_srv_buf',
            resource: 'buffer',
            in_layout: true,
            source: {
              resource_heap: {
                base_user_sgpr: 4,
                stride: 16,
                index: { user_sgpr: 6, shift: 4 },
              },
            },
          },
          {
            group: 0,
            binding: 1,
            name: 'root_cbv',
            resource: 'buffer',
            in_layout: true,
            source: { user_sgpr: 12, root_descriptor: true },
          },
        ],
      },
    },
  };

  const parsed = parseReflectionManifest(s34Manifest);
  assert.equal(parsed.samplerHeapBindings, 1);
  assert.equal(parsed.resourceHeapBindings, 2);
  assert.deepEqual(parsed.summary, {
    shaderCount: 2,
    bindGroupLayoutCount: 2,
    pipelineLayoutCount: 2,
    pipelineCount: 0,
    samplerHeapBindings: 1,
    resourceHeapBindings: 2,
  });

  const fs = parsed.shadersByHash.get(0xa001n);
  assert.equal(fs.samplerHeapBindings, 1);
  assert.equal(fs.resourceHeapBindings, 1);
  assert.deepEqual(fs.bindings[0].source, {
    kind: 'table',
    userSgpr: 8,
    tableByteOffset: 0,
  });
  assert.deepEqual(fs.bindings[1].source, {
    kind: 'resource_heap',
    baseUserSgpr: 11,
    stride: 32,
    indexUserSgpr: 10,
    indexShift: 5,
  });
  assert.deepEqual(fs.bindings[2].source, {
    kind: 'sampler_heap',
    baseUserSgpr: 11,
    stride: 16,
    indexUserSgpr: 9,
    indexShift: 4,
  });
  assert.deepEqual(fs.bindings[3].source, {
    kind: 'static_sampler',
    index: 0,
  });

  const cs = parsed.shadersByHash.get(0xa002n);
  assert.equal(cs.samplerHeapBindings, 0);
  assert.equal(cs.resourceHeapBindings, 1);
  assert.deepEqual(cs.bindings[0].source, {
    kind: 'resource_heap',
    baseUserSgpr: 4,
    stride: 16,
    indexUserSgpr: 6,
    indexShift: 4,
  });
  assert.deepEqual(cs.bindings[1].source, {
    kind: 'root_descriptor',
    userSgpr: 12,
  });

  // Existing manifests without `source` remain 100% compatible with zero heap counts
  const registry = new ReflectionRegistry();
  const legacyParsed = registry.loadManifest(createS6ManifestFixture());
  assert.equal(legacyParsed.samplerHeapBindings, 0);
  assert.equal(legacyParsed.resourceHeapBindings, 0);
  assert.equal(legacyParsed.shadersByHash.get(0x3333n).bindings[0].source, null);
  assert.equal(registry.getStats().samplerHeapBindings, 0);
  assert.equal(registry.getStats().resourceHeapBindings, 0);

  // Loading S34 manifest accumulates heap stats cleanly and still builds WebGPU layouts
  registry.loadManifest(s34Manifest);
  assert.equal(registry.getStats().samplerHeapBindings, 1);
  assert.equal(registry.getStats().resourceHeapBindings, 2);

  const descriptors = registry.buildBindGroupLayoutDescriptors([0xa001n]);
  assert.equal(descriptors.length, 1);
  assert.equal(descriptors[0].entries.length, 4);

  // Overwriting an existing shader hash updates heap stats without double-counting
  registry.loadManifest({
    schema_version: 1,
    bind_group_layouts: { [idCs]: bglCs },
    pipeline_layouts: { [plCs]: { bindGroupLayouts: [idCs] } },
    shaders: {
      '0x000000000000a002': {
        stage: 'compute',
        entry_point: 'cs_s34_resource_heap_replaced',
        pipeline_layout: plCs,
        bind_group_layouts: [idCs],
        bindings: [
          {
            group: 0,
            binding: 0,
            name: 'table_buf',
            resource: 'buffer',
            in_layout: true,
            source: { user_sgpr: 4, table_byte_offset: 0 },
          },
          {
            group: 0,
            binding: 1,
            name: 'root_cbv',
            resource: 'buffer',
            in_layout: true,
            source: { user_sgpr: 12, root_constants: true },
          },
        ],
      },
    },
  });
  assert.equal(registry.getStats().samplerHeapBindings, 1);
  assert.equal(registry.getStats().resourceHeapBindings, 1);

  // Clearing registry resets live heap binding counts
  registry.clear();
  assert.equal(registry.getStats().samplerHeapBindings, 0);
  assert.equal(registry.getStats().resourceHeapBindings, 0);
});

test('parseReflectionManifest and parseBindingSource reject malformed S34 heap and legacy sources cleanly', () => {
  const wrapInManifest = (source) => ({
    schema_version: 1,
    bind_group_layouts: {},
    pipeline_layouts: {},
    shaders: {
      '0x000000000000b001': {
        stage: 'fragment',
        entry_point: 'fs_bad_source',
        bindings: [
          {
            group: 0,
            binding: 0,
            name: 'bad_binding',
            resource: 'sampler',
            in_layout: true,
            source,
          },
        ],
      },
    },
  });

  const invalidCases = [
    // Non-object or empty source
    ['not_an_object', /binding source must be an object or null/],
    [42, /binding source must be an object or null/],
    [[], /binding source must be an object or null/],
    [{}, /unrecognized binding source/],
    // Non-object heap payload
    [{ sampler_heap: null }, /sampler_heap must be an object/],
    [{ sampler_heap: 3 }, /sampler_heap must be an object/],
    [{ resource_heap: [] }, /resource_heap must be an object/],
    [{ resource_heap: 'heap' }, /resource_heap must be an object/],
    // Ambiguous / conflicting source fields
    [
      {
        sampler_heap: { base_user_sgpr: 11, stride: 16, index: { user_sgpr: 9, shift: 4 } },
        resource_heap: { base_user_sgpr: 4, stride: 32, index: { user_sgpr: 5, shift: 5 } },
      },
      /cannot define both sampler_heap and resource_heap/,
    ],
    [
      {
        sampler_heap: { base_user_sgpr: 11, stride: 16, index: { user_sgpr: 9, shift: 4 } },
        static_sampler: 0,
      },
      /cannot be combined with other binding source fields/,
    ],
    // Invalid base_user_sgpr
    [
      { sampler_heap: { stride: 16, index: { user_sgpr: 9, shift: 4 } } },
      /sampler_heap\.base_user_sgpr/,
    ],
    [
      { resource_heap: { base_user_sgpr: -1, stride: 32, index: { user_sgpr: 5, shift: 5 } } },
      /resource_heap\.base_user_sgpr/,
    ],
    [
      { resource_heap: { base_user_sgpr: '4', stride: 32, index: { user_sgpr: 5, shift: 5 } } },
      /resource_heap\.base_user_sgpr/,
    ],
    [
      { sampler_heap: { base_user_sgpr: 1.5, stride: 16, index: { user_sgpr: 9, shift: 4 } } },
      /sampler_heap\.base_user_sgpr/,
    ],
    [
      { sampler_heap: { base_user_sgpr: true, stride: 16, index: { user_sgpr: 9, shift: 4 } } },
      /sampler_heap\.base_user_sgpr/,
    ],
    [
      {
        sampler_heap: {
          base_user_sgpr: Number.MAX_SAFE_INTEGER + 1,
          stride: 16,
          index: { user_sgpr: 9, shift: 4 },
        },
      },
      /sampler_heap\.base_user_sgpr/,
    ],
    // Invalid stride (must be > 0 safe uint32 int)
    [
      { sampler_heap: { base_user_sgpr: 11, stride: 0, index: { user_sgpr: 9, shift: 4 } } },
      /sampler_heap\.stride/,
    ],
    [
      { resource_heap: { base_user_sgpr: 4, stride: -32, index: { user_sgpr: 5, shift: 5 } } },
      /resource_heap\.stride/,
    ],
    [
      { sampler_heap: { base_user_sgpr: 11, stride: NaN, index: { user_sgpr: 9, shift: 4 } } },
      /sampler_heap\.stride/,
    ],
    [
      { resource_heap: { base_user_sgpr: 4, stride: Infinity, index: { user_sgpr: 5, shift: 5 } } },
      /resource_heap\.stride/,
    ],
    // Missing or malformed index object
    [{ sampler_heap: { base_user_sgpr: 11, stride: 16 } }, /sampler_heap\.index/],
    [{ resource_heap: { base_user_sgpr: 4, stride: 32, index: null } }, /resource_heap\.index/],
    [{ sampler_heap: { base_user_sgpr: 11, stride: 16, index: [] } }, /sampler_heap\.index/],
    // Invalid index.user_sgpr
    [
      { sampler_heap: { base_user_sgpr: 11, stride: 16, index: { shift: 4 } } },
      /sampler_heap\.index\.user_sgpr/,
    ],
    [
      { resource_heap: { base_user_sgpr: 4, stride: 32, index: { user_sgpr: -2, shift: 5 } } },
      /resource_heap\.index\.user_sgpr/,
    ],
    // Invalid index.shift (must be integer in 0..31)
    [
      { sampler_heap: { base_user_sgpr: 11, stride: 16, index: { user_sgpr: 9 } } },
      /sampler_heap\.index\.shift/,
    ],
    [
      { resource_heap: { base_user_sgpr: 4, stride: 32, index: { user_sgpr: 5, shift: 32 } } },
      /resource_heap\.index\.shift/,
    ],
    [
      { resource_heap: { base_user_sgpr: 4, stride: 32, index: { user_sgpr: 5, shift: 40 } } },
      /resource_heap\.index\.shift/,
    ],
    [
      { sampler_heap: { base_user_sgpr: 11, stride: 16, index: { user_sgpr: 9, shift: -1 } } },
      /sampler_heap\.index\.shift/,
    ],
    [
      { sampler_heap: { base_user_sgpr: 11, stride: 16, index: { user_sgpr: 9, shift: 2.5 } } },
      /sampler_heap\.index\.shift/,
    ],
    // Malformed non-heap sources
    [{ static_sampler: -1 }, /static_sampler/],
    [{ user_sgpr: 8 }, /user_sgpr source must specify exactly one/],
    [{ user_sgpr: 8, root_constants: true, root_descriptor: true }, /user_sgpr source must specify exactly one/],
    [{ user_sgpr: 8, table_byte_offset: -4 }, /table_byte_offset/],
    [{ table_byte_offset: 16 }, /requires user_sgpr/],
  ];

  for (const [badSource, expectedPattern] of invalidCases) {
    assert.throws(
      () => parseBindingSource(badSource, 'test.source'),
      (err) => err instanceof ManifestValidationError && expectedPattern.test(err.message),
      `Expected parseBindingSource(${JSON.stringify(badSource)}) to match ${expectedPattern}`
    );
    assert.throws(
      () => parseReflectionManifest(wrapInManifest(badSource)),
      (err) => err instanceof ManifestValidationError && expectedPattern.test(err.message),
      `Expected parseReflectionManifest with source ${JSON.stringify(badSource)} to match ${expectedPattern}`
    );
  }
});
