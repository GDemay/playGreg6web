import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const drawExecutorMod = require('../web/workers/draw_executor.js');
const routerMod = require('../web/workers/shader_fallback_router.js');

const DRAW_EXECUTOR_PATH = fileURLToPath(
  new URL('../web/workers/draw_executor.js', import.meta.url)
);
const ROUTER_PATH = fileURLToPath(
  new URL('../web/workers/shader_fallback_router.js', import.meta.url)
);

const {
  FALLBACK_PROFILE,
  TEXTURED_QUAD_VERTEX_STRIDE,
  FRAME1_CBV_MIN_BYTES,
  FALLBACK_VERTEX_LAYOUT_24B,
  selectFallbackShaderProfile,
  DrawExecutor,
} = drawExecutorMod;

const { ShaderFallbackRouter, BUILTIN_FALLBACK_WGSL, getBuiltInFallbackProfiles } = routerMod;

const BUFFER_USAGE = Object.freeze({
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
});

function createMockWebGpu(overrides = {}) {
  let nextHandle = 1;
  const calls = {
    createBuffer: [],
    createTexture: [],
    createSampler: [],
    createShaderModule: [],
    createRenderPipeline: [],
    createBindGroup: [],
    writeBuffer: [],
    writeTexture: [],
    pass: [],
  };

  const device = {
    createBuffer(desc) {
      if (overrides.failCreateBuffer) {
        throw new Error('mock createBuffer failure');
      }
      const buf = {
        kind: 'GPUBuffer',
        handle: nextHandle++,
        label: desc.label || '',
        size: desc.size,
        usage: desc.usage,
        destroyed: false,
        destroy() {
          buf.destroyed = true;
        },
      };
      calls.createBuffer.push({ desc, result: buf });
      return buf;
    },
    createTexture(desc) {
      if (overrides.failCreateTexture) {
        throw new Error('mock createTexture failure');
      }
      const tex = {
        kind: 'GPUTexture',
        handle: nextHandle++,
        label: desc.label || '',
        desc,
        destroyed: false,
        createView(viewDesc) {
          return { kind: 'GPUTextureView', textureHandle: tex.handle, viewDesc };
        },
        destroy() {
          tex.destroyed = true;
        },
      };
      calls.createTexture.push({ desc, result: tex });
      return tex;
    },
    createSampler(desc) {
      if (overrides.failCreateSampler) {
        throw new Error('mock createSampler failure');
      }
      const sampler = { kind: 'GPUSampler', handle: nextHandle++, desc };
      calls.createSampler.push({ desc, result: sampler });
      return sampler;
    },
    createShaderModule(desc) {
      if (overrides.failCreateShaderModule) {
        throw new Error('mock createShaderModule failure');
      }
      const mod = {
        kind: 'GPUShaderModule',
        handle: nextHandle++,
        label: desc.label || '',
        code: desc.code,
      };
      calls.createShaderModule.push({ desc, result: mod });
      return mod;
    },
    createRenderPipeline(desc) {
      if (overrides.failCreateRenderPipeline) {
        throw new Error('mock createRenderPipeline failure');
      }
      const pipe = {
        kind: 'GPURenderPipeline',
        handle: nextHandle++,
        label: desc.label || '',
        desc,
        getBindGroupLayout(index) {
          return { kind: 'GPUBindGroupLayout', pipelineHandle: pipe.handle, index };
        },
      };
      calls.createRenderPipeline.push({ desc, result: pipe });
      return pipe;
    },
    createBindGroup(desc) {
      if (overrides.failCreateBindGroup) {
        throw new Error('mock createBindGroup failure');
      }
      const bg = { kind: 'GPUBindGroup', handle: nextHandle++, label: desc.label || '', desc };
      calls.createBindGroup.push({ desc, result: bg });
      return bg;
    },
    queue: {
      writeBuffer(buffer, offset, data) {
        calls.writeBuffer.push({ buffer, offset, byteLength: data.byteLength });
      },
      writeTexture(dst, data, layout, size) {
        calls.writeTexture.push({ dst, byteLength: data.byteLength, layout, size });
      },
    },
  };

  const pass = {
    setPipeline(pipeline) {
      if (overrides.failSetPipeline) {
        throw new Error('mock setPipeline failure');
      }
      calls.pass.push({ op: 'setPipeline', pipeline });
    },
    setVertexBuffer(slot, buffer, offset, size) {
      if (overrides.failSetVertexBuffer) {
        throw new Error('mock setVertexBuffer failure');
      }
      calls.pass.push({ op: 'setVertexBuffer', slot, buffer, offset, size });
    },
    setIndexBuffer(buffer, indexFormat, offset, size) {
      calls.pass.push({ op: 'setIndexBuffer', buffer, indexFormat, offset, size });
    },
    setBindGroup(groupIndex, bindGroup, dynamicOffsets) {
      calls.pass.push({ op: 'setBindGroup', groupIndex, bindGroup, dynamicOffsets });
    },
    draw(vertexCount, instanceCount, firstVertex, firstInstance) {
      if (overrides.failDraw) {
        throw new Error('mock draw failure');
      }
      calls.pass.push({ op: 'draw', vertexCount, instanceCount, firstVertex, firstInstance });
    },
    drawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance) {
      if (overrides.failDrawIndexed) {
        throw new Error('mock drawIndexed failure');
      }
      calls.pass.push({
        op: 'drawIndexed',
        indexCount,
        instanceCount,
        firstIndex,
        baseVertex,
        firstInstance,
      });
    },
  };

  return { device, pass, calls };
}

test('UMD export attaches to globalThis.PlayGta6DrawExecutor and module.exports', () => {
  assert.equal(globalThis.PlayGta6DrawExecutor, drawExecutorMod);
  assert.equal(typeof DrawExecutor, 'function');
  assert.equal(typeof selectFallbackShaderProfile, 'function');
  assert.equal(DrawExecutor.selectFallbackShaderProfile, selectFallbackShaderProfile);
  assert.equal(TEXTURED_QUAD_VERTEX_STRIDE, 24);
  assert.equal(FRAME1_CBV_MIN_BYTES, 256);
  assert.deepEqual(FALLBACK_PROFILE, {
    DEFAULT: 'default',
    FULLSCREEN_TRIANGLE_CBV: 'fullscreen_triangle_cbv',
    TEXTURED_QUAD_24B: 'textured_quad_24b',
  });
  assert.equal(FALLBACK_VERTEX_LAYOUT_24B.arrayStride, 24);
  assert.deepEqual(FALLBACK_VERTEX_LAYOUT_24B.attributes, [
    { shaderLocation: 0, offset: 0, format: 'float32x4' },
    { shaderLocation: 1, offset: 16, format: 'float32x2' },
  ]);
});

test('selectFallbackShaderProfile routes Frame-1 DrawInstanced(3,1)+CBV, 24B textured quad, and default', () => {
  // Null / undefined / empty -> default
  assert.equal(selectFallbackShaderProfile(null), 'default');
  assert.equal(selectFallbackShaderProfile(undefined), 'default');
  assert.equal(selectFallbackShaderProfile({}), 'default');

  // Exact Frame-1 DrawInstanced(3, 1) with bound CBV and no sampled texture -> fullscreen_triangle_cbv
  assert.equal(
    selectFallbackShaderProfile({
      vertexCount: 3,
      instanceCount: 1,
      hasBoundCbv: true,
      hasBoundTexture: false,
    }),
    'fullscreen_triangle_cbv'
  );

  // Even when a 24B-stride VB is also bound alongside the CBV, no sampled texture means Frame-1 fullscreen CBV
  assert.equal(
    selectFallbackShaderProfile({
      vertexCount: 3,
      vertexStride: 24,
      cbvBuffer: { kind: 'GPUBuffer', size: 256 },
    }),
    'fullscreen_triangle_cbv'
  );

  // Bind-group metadata inspection detects CBV
  assert.equal(
    selectFallbackShaderProfile({
      vertexCount: 3,
      bindGroups: [{ entries: [{ binding: 0, type: 'uniform', resource: { buffer: {} } }] }],
    }),
    'fullscreen_triangle_cbv'
  );

  // 24-byte-stride VB + bound texture/sampler -> textured_quad_24b
  assert.equal(
    selectFallbackShaderProfile({
      vertexCount: 6,
      vertexStride: 24,
      hasBoundTexture: true,
      hasBoundSampler: true,
    }),
    'textured_quad_24b'
  );

  // 24-byte-stride VB inside vertexBuffers array + texture only -> textured_quad_24b
  assert.equal(
    selectFallbackShaderProfile({
      vertexCount: 6,
      vertexBuffers: [{ slot: 0, stride: 24, buffer: { size: 144 } }],
      textures: [{ kind: 'GPUTextureView' }],
    }),
    'textured_quad_24b'
  );

  // 24-byte-stride VB + sampler inside bindGroups -> textured_quad_24b
  assert.equal(
    selectFallbackShaderProfile({
      vertexCount: 4,
      vertexBufferLayouts: [{ arrayStride: 24 }],
      bindGroups: [{ entries: [{ binding: 1, type: 'sampler', resource: { kind: 'GPUSampler' } }] }],
    }),
    'textured_quad_24b'
  );

  // vertexCount === 3 with CBV + 24B VB + bound texture -> textured_quad_24b (texture excludes fullscreen_triangle_cbv)
  assert.equal(
    selectFallbackShaderProfile({
      vertexCount: 3,
      vertexStride: 24,
      hasBoundCbv: true,
      hasBoundTexture: true,
      hasBoundSampler: true,
    }),
    'textured_quad_24b'
  );

  // 24B VB without texture/sampler and vertexCount != 3 -> default
  assert.equal(
    selectFallbackShaderProfile({
      vertexCount: 6,
      vertexStride: 24,
      hasBoundCbv: true,
    }),
    'default'
  );

  // 32B VB with texture/sampler -> default
  assert.equal(
    selectFallbackShaderProfile({
      vertexCount: 6,
      vertexStride: 32,
      hasBoundTexture: true,
      hasBoundSampler: true,
    }),
    'default'
  );
});

test('DrawExecutor routes Frame-1 DrawInstanced(3,1) with bound CBV to G34 fullscreen_triangle_cbv', () => {
  const { device, pass, calls } = createMockWebGpu();
  const router = new ShaderFallbackRouter({ now: () => 100 });
  const executor = new DrawExecutor({ device, shaderFallbackRouter: router });
  const profiles = getBuiltInFallbackProfiles();

  const cbv = device.createBuffer({
    size: 256,
    usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST,
  });
  const vb = device.createBuffer({
    size: 72,
    usage: BUFFER_USAGE.VERTEX,
  });

  executor.beginPass();
  executor.setPipeline({
    id: 101,
    vertexShaderHash: '0x1111111111111111',
    fragmentShaderHash: '0x2222222222222222',
    vertexStride: 24,
    isFallback: true,
  });
  executor.setVertexBuffer(0, vb, { offset: 0, stride: 24 });
  executor.setConstantBuffer(0, cbv, { offset: 0, size: 256 });

  const result = executor.draw(pass, 3, 1, 0, 0);
  assert.equal(result.executed, true);
  assert.equal(result.skipped, false);
  assert.equal(result.isFallback, true);
  assert.equal(result.profile, 'fullscreen_triangle_cbv');
  assert.equal(result.effectiveProfile, 'fullscreen_triangle_cbv');
  assert.equal(result.substitutedCbv, false);

  // Verify G34 fullscreen_triangle_cbv WGSL modules were compiled.
  assert.equal(calls.createShaderModule.length, 2);
  assert.equal(calls.createShaderModule[0].desc.code, profiles.fullscreen_triangle_cbv.vertex);
  assert.equal(calls.createShaderModule[1].desc.code, profiles.fullscreen_triangle_cbv.fragment);

  // Pipeline uses empty vertex buffers (fullscreen triangle via @builtin(vertex_index)).
  assert.equal(calls.createRenderPipeline.length, 1);
  assert.deepEqual(calls.createRenderPipeline[0].desc.vertex.buffers, []);

  // Bound the caller's real 256B CBV at group 0 binding 0 without synthesizing a replacement.
  assert.equal(calls.createBindGroup.length, 1);
  assert.equal(calls.createBindGroup[0].desc.entries[0].binding, 0);
  assert.equal(calls.createBindGroup[0].desc.entries[0].resource.buffer, cbv);

  // Pass recorded setPipeline, setBindGroup(0), draw(3, 1, 0, 0).
  assert.deepEqual(
    calls.pass.map((c) => c.op),
    ['setPipeline', 'setBindGroup', 'draw']
  );
  assert.equal(calls.pass[2].vertexCount, 3);
  assert.equal(calls.pass[2].instanceCount, 1);

  const stats = executor.getStats();
  assert.equal(stats.totalDraws, 1);
  assert.equal(stats.fallbackDraws, 1);
  assert.equal(stats.normalDraws, 0);
  assert.deepEqual(stats.fallbackProfileCounts, {
    default: 0,
    fullscreen_triangle_cbv: 1,
    textured_quad_24b: 0,
  });
  assert.equal(stats.invalidCbvSubstitutions, 0);
  assert.equal(stats.fallbackPipelineCacheMisses, 1);

  // Second identical draw hits the fallback pipeline cache.
  const second = executor.draw(pass, 3, 1, 0, 0);
  assert.equal(second.executed, true);
  assert.equal(calls.createRenderPipeline.length, 1);
  assert.equal(executor.getStats().fallbackPipelineCacheHits, 1);
  assert.equal(executor.getStats().fallbackProfileCounts.fullscreen_triangle_cbv, 2);
});

test('DrawExecutor routes 24B-stride textured quad draws to G34 textured_quad_24b with valid layout & bindings', () => {
  const { device, pass, calls } = createMockWebGpu();
  const router = new ShaderFallbackRouter({ now: () => 200 });
  const executor = new DrawExecutor({ device, shaderFallbackRouter: router });
  const profiles = getBuiltInFallbackProfiles();

  const vb = device.createBuffer({ size: 24 * 6, usage: BUFFER_USAGE.VERTEX });
  const tex = device.createTexture({
    size: { width: 64, height: 64, depthOrArrayLayers: 1 },
    format: 'rgba8unorm',
    usage: 0x04,
  });
  const texView = tex.createView();
  const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  executor.beginPass();
  executor.setPipeline({
    id: 202,
    vertexShaderHash: '0x3333333333333333',
    fragmentShaderHash: '0x4444444444444444',
    vertexStride: 24,
    isFallback: true,
  });
  executor.setVertexBuffer(0, vb, { offset: 0, stride: 24 });
  executor.setTexture(0, texView);
  executor.setSampler(0, sampler);

  const result = executor.executeDraw(pass, { vertexCount: 6, instanceCount: 1 });
  assert.equal(result.executed, true);
  assert.equal(result.profile, 'textured_quad_24b');
  assert.equal(result.effectiveProfile, 'textured_quad_24b');
  assert.equal(result.substitutedTexture, false);
  assert.equal(result.substitutedSampler, false);
  assert.equal(result.substitutedVertexBuffer, false);

  assert.equal(calls.createShaderModule[0].desc.code, profiles.textured_quad_24b.vertex);
  assert.equal(calls.createShaderModule[1].desc.code, profiles.textured_quad_24b.fragment);
  assert.equal(calls.createRenderPipeline[0].desc.vertex.buffers.length, 1);
  assert.equal(calls.createRenderPipeline[0].desc.vertex.buffers[0].arrayStride, 24);
  assert.deepEqual(calls.createRenderPipeline[0].desc.vertex.buffers[0].attributes, [
    { shaderLocation: 0, offset: 0, format: 'float32x4' },
    { shaderLocation: 1, offset: 16, format: 'float32x2' },
  ]);

  assert.equal(calls.createBindGroup[0].desc.entries.length, 2);
  assert.equal(calls.createBindGroup[0].desc.entries[0].resource, texView);
  assert.equal(calls.createBindGroup[0].desc.entries[1].resource, sampler);

  assert.deepEqual(
    calls.pass.map((c) => c.op),
    ['setPipeline', 'setBindGroup', 'setVertexBuffer', 'draw']
  );
  assert.equal(calls.pass[2].buffer, vb);
  assert.equal(calls.pass[3].vertexCount, 6);

  assert.deepEqual(executor.getStats().fallbackProfileCounts, {
    default: 0,
    fullscreen_triangle_cbv: 0,
    textured_quad_24b: 1,
  });
});

test('DrawExecutor routes generic missing-shader draws to default fallback profile', () => {
  const { device, pass, calls } = createMockWebGpu();
  const router = new ShaderFallbackRouter({ now: () => 300 });
  const executor = new DrawExecutor({ device, shaderFallbackRouter: router });

  executor.beginPass();
  executor.setPipeline({
    id: 303,
    vertexShaderHash: '0x5555555555555555',
    fragmentShaderHash: '0x6666666666666666',
    vertexStride: 32,
    isFallback: true,
  });

  const result = executor.draw(pass, 36, 2, 0, 0);
  assert.equal(result.executed, true);
  assert.equal(result.profile, 'default');
  assert.equal(result.effectiveProfile, 'default');
  assert.equal(calls.createShaderModule[0].desc.code, BUILTIN_FALLBACK_WGSL.vertex);
  assert.equal(calls.createShaderModule[1].desc.code, BUILTIN_FALLBACK_WGSL.fragment);
  assert.equal(calls.createBindGroup.length, 0);
  assert.deepEqual(executor.getStats().fallbackProfileCounts, {
    default: 1,
    fullscreen_triangle_cbv: 0,
    textured_quad_24b: 0,
  });
});

test('DrawExecutor passes { profile } to modern routers and stays compatible with legacy 2-arg mock routers', () => {
  // 1. Modern spy router verifying exact 3-arg { profile } forwarding
  const modernCalls = [];
  const modernRouter = {
    resolve(hash, stage, options) {
      modernCalls.push({ hash, stage, options });
      const profiles = getBuiltInFallbackProfiles();
      const prof = options && options.profile ? options.profile : 'default';
      return {
        wgsl: profiles[prof][stage],
        source: 'builtin',
        entryPoint: stage === 'fragment' ? 'fs_main' : 'vs_main',
        profile: prof,
      };
    },
  };

  const env1 = createMockWebGpu();
  const exec1 = new DrawExecutor({ device: env1.device, shaderFallbackRouter: modernRouter });
  const cbv = env1.device.createBuffer({ size: 256, usage: BUFFER_USAGE.UNIFORM });
  exec1.setConstantBuffer(0, cbv);
  exec1.setPipeline({ vertexShaderHash: '0xaaaa', fragmentShaderHash: '0xbbbb', isFallback: true });
  const res1 = exec1.draw(env1.pass, 3, 1);
  assert.equal(res1.executed, true);
  assert.equal(modernCalls.length, 2);
  assert.deepEqual(modernCalls[0].options, { profile: 'fullscreen_triangle_cbv' });
  assert.deepEqual(modernCalls[1].options, { profile: 'fullscreen_triangle_cbv' });

  // 2. Legacy 2-arg mock router that ignores extra arguments and omits `profile` in return value
  const legacySoftCalls = [];
  const legacySoftRouter = {
    resolve(hash, stage) {
      legacySoftCalls.push({ hash, stage });
      return {
        wgsl: BUILTIN_FALLBACK_WGSL[stage],
        source: 'builtin',
      };
    },
  };
  const env2 = createMockWebGpu();
  const exec2 = new DrawExecutor({ device: env2.device, shaderFallbackRouter: legacySoftRouter });
  exec2.setConstantBuffer(0, env2.device.createBuffer({ size: 256, usage: BUFFER_USAGE.UNIFORM }));
  const res2 = exec2.draw(env2.pass, 3, 1);
  assert.equal(res2.executed, true);
  assert.equal(res2.profile, 'fullscreen_triangle_cbv');
  // Effective profile safely inferred as default because legacy WGSL declares no @group(0) uniform
  assert.equal(res2.effectiveProfile, 'default');
  assert.equal(legacySoftCalls.length, 2);
  assert.equal(env2.calls.createBindGroup.length, 0);

  // 3. Strict legacy 2-arg mock router that throws if >2 arguments are passed
  let strictCalls = 0;
  const strictLegacyRouter = {
    resolve(hash, stage) {
      if (arguments.length > 2) {
        throw new TypeError('legacy 2-arg router does not accept options');
      }
      strictCalls += 1;
      return {
        wgsl: BUILTIN_FALLBACK_WGSL[stage],
        source: 'builtin',
      };
    },
  };
  const env3 = createMockWebGpu();
  const exec3 = new DrawExecutor({
    device: env3.device,
    shaderFallbackRouter: strictLegacyRouter,
  });
  exec3.setConstantBuffer(0, env3.device.createBuffer({ size: 256, usage: BUFFER_USAGE.UNIFORM }));
  const res3 = exec3.draw(env3.pass, 3, 1);
  assert.equal(res3.executed, true);
  assert.equal(strictCalls, 2);
  assert.equal(exec3.getStats().legacyRouterFallbacks, 2);
  assert.equal(exec3.getStats().fallbackProfileCounts.fullscreen_triangle_cbv, 1);
});

test('DrawExecutor hardens invalid/undersized/misaligned/destroyed CBV slots by substituting a valid 256B uniform buffer', () => {
  const { device, pass, calls } = createMockWebGpu();
  const router = new ShaderFallbackRouter();
  const executor = new DrawExecutor({ device, shaderFallbackRouter: router });

  // Case A: Undersized CBV (only 64 bytes < 256 bytes required by FrameCbv)
  const tinyCbv = device.createBuffer({ size: 64, usage: BUFFER_USAGE.UNIFORM });
  executor.setConstantBuffer(0, tinyCbv, { offset: 0, size: 64 });
  const r1 = executor.draw(pass, 3, 1);
  assert.equal(r1.executed, true);
  assert.equal(r1.profile, 'fullscreen_triangle_cbv');
  assert.equal(r1.substitutedCbv, true);

  // Case B: Destroyed CBV
  const destroyedCbv = device.createBuffer({ size: 256, usage: BUFFER_USAGE.UNIFORM });
  destroyedCbv.destroy();
  executor.setConstantBuffer(0, destroyedCbv);
  const r2 = executor.executeDraw(pass, { vertexCount: 3, hasBoundCbv: true });
  assert.equal(r2.executed, true);
  assert.equal(r2.substitutedCbv, true);

  // Case C: Misaligned CBV offset (128 is not 256-byte aligned)
  const bigCbv = device.createBuffer({ size: 512, usage: BUFFER_USAGE.UNIFORM });
  executor.setConstantBuffer(0, bigCbv, { offset: 128, size: 256 });
  const r3 = executor.draw(pass, 3, 1);
  assert.equal(r3.executed, true);
  assert.equal(r3.substitutedCbv, true);

  // Case D: Wrong usage (VERTEX only, missing UNIFORM)
  const vertexOnlyBuf = device.createBuffer({ size: 256, usage: BUFFER_USAGE.VERTEX });
  executor.setConstantBuffer(0, vertexOnlyBuf);
  const r4 = executor.draw(pass, 3, 1);
  assert.equal(r4.executed, true);
  assert.equal(r4.substitutedCbv, true);

  // Synthetic 256B CBV was allocated ONCE and reused across all 4 guarded draws!
  const syntheticBuffers = calls.createBuffer.filter(
    (c) => c.desc.label === 'draw_executor:fallback_cbv_256b'
  );
  assert.equal(syntheticBuffers.length, 1);
  assert.equal(syntheticBuffers[0].desc.size, 256);

  const stats = executor.getStats();
  assert.equal(stats.fallbackDraws, 4);
  assert.equal(stats.invalidCbvSubstitutions, 4);
  assert.equal(stats.bindGroupGuards, 4);
  assert.equal(stats.skippedDraws, 0);
});

test('DrawExecutor hardens missing/invalid texture, sampler, and 24B vertex buffer on textured_quad_24b draws', () => {
  const { device, pass, calls } = createMockWebGpu();
  const router = new ShaderFallbackRouter();
  const executor = new DrawExecutor({ device, shaderFallbackRouter: router });

  // Bind only a 24B-stride hint + destroyed texture + no sampler + undersized VB (16B < 144B)
  const tinyVb = device.createBuffer({ size: 16, usage: BUFFER_USAGE.VERTEX });
  const destroyedTex = device.createTexture({
    size: { width: 4, height: 4, depthOrArrayLayers: 1 },
    format: 'rgba8unorm',
    usage: 0x04,
  });
  destroyedTex.destroy();

  executor.setPipeline({ vertexStride: 24, isFallback: true });
  executor.setVertexBuffer(0, tinyVb, { offset: 0, stride: 24 });
  executor.setTexture(0, destroyedTex);

  const res = executor.executeDraw(pass, {
    vertexCount: 6,
    instanceCount: 1,
    hasBoundTexture: true,
  });
  assert.equal(res.executed, true);
  assert.equal(res.profile, 'textured_quad_24b');
  assert.equal(res.substitutedTexture, true);
  assert.equal(res.substitutedSampler, true);
  assert.equal(res.substitutedVertexBuffer, true);

  const stats = executor.getStats();
  assert.equal(stats.invalidTextureSubstitutions, 1);
  assert.equal(stats.invalidSamplerSubstitutions, 1);
  assert.equal(stats.vertexBufferSubstitutions, 1);
  assert.equal(stats.invalidVertexBufferGuards, 1);
  assert.equal(stats.bindGroupGuards, 2);
  assert.equal(stats.fallbackProfileCounts.textured_quad_24b, 1);

  // Verify synthetic fallback VB is 144 bytes (6 * 24B).
  const synthVb = calls.createBuffer.find(
    (c) => c.desc.label === 'draw_executor:fallback_vb_24b_quad'
  );
  assert.ok(synthVb);
  assert.equal(synthVb.desc.size, 144);
});

test('DrawExecutor skips gracefully with counters (never throws) when GPU resource creation or pass calls fail', () => {
  const warnings = [];
  const router = new ShaderFallbackRouter();

  // 1. Pipeline creation fails -> skips draw cleanly
  const failPipeEnv = createMockWebGpu({ failCreateRenderPipeline: true });
  const exec1 = new DrawExecutor({
    device: failPipeEnv.device,
    shaderFallbackRouter: router,
    onWarning: (msg) => warnings.push(msg),
  });
  const r1 = exec1.draw(failPipeEnv.pass, 3, 1);
  assert.equal(r1.executed, false);
  assert.equal(r1.skipped, true);
  assert.equal(r1.reason, 'pipeline-creation-failed');
  assert.equal(exec1.getStats().pipelineErrors, 1);
  assert.equal(exec1.getStats().skippedDraws, 1);

  // 2. Bind group creation fails during CBV fallback -> skips draw cleanly
  const failBgEnv = createMockWebGpu({ failCreateBindGroup: true });
  const exec2 = new DrawExecutor({
    device: failBgEnv.device,
    shaderFallbackRouter: router,
    onWarning: (msg) => warnings.push(msg),
  });
  const r2 = exec2.executeDraw(failBgEnv.pass, { vertexCount: 3, hasBoundCbv: true });
  assert.equal(r2.executed, false);
  assert.equal(r2.skipped, true);
  assert.equal(r2.reason, 'fallback-cbv-bind-failed');
  assert.equal(exec2.getStats().bindGroupErrors, 1);
  assert.equal(exec2.getStats().skippedDraws, 1);

  // 3. Invalid slot indices, zero vertex count, null pass, and destroyed executor never throw
  const okEnv = createMockWebGpu();
  const exec3 = new DrawExecutor({ device: okEnv.device, shaderFallbackRouter: router });
  assert.equal(exec3.setVertexBuffer(-1, {}), false);
  assert.equal(exec3.setBindGroup(-2, {}), false);
  assert.equal(exec3.setConstantBuffer(-1, {}), false);
  assert.equal(exec3.setTexture(-1, {}), false);
  assert.equal(exec3.setSampler(-1, {}), false);

  assert.equal(exec3.draw(null, 3, 1).reason, 'invalid-render-pass');
  assert.equal(exec3.draw(okEnv.pass, 0, 1).reason, 'zero-draw-count');
  assert.equal(exec3.draw(okEnv.pass, 3, 0).reason, 'zero-draw-count');

  exec3.destroy();
  assert.equal(exec3.draw(okEnv.pass, 3, 1).reason, 'executor-destroyed');
  assert.equal(exec3.getStats().skippedDraws, 4);
  assert.ok(warnings.length >= 2);
});

test('DrawExecutor normal non-fallback path preserves real pipelines, guards invalid bindings, and supports indexed draws', () => {
  const { device, pass, calls } = createMockWebGpu();
  const router = new ShaderFallbackRouter();
  const executor = new DrawExecutor({ device, shaderFallbackRouter: router });

  const realPipeline = {
    kind: 'GPURenderPipeline',
    handle: 999,
    getBindGroupLayout: () => ({ kind: 'GPUBindGroupLayout' }),
  };
  const validVb = device.createBuffer({ size: 256, usage: BUFFER_USAGE.VERTEX });
  const badVb = device.createBuffer({ size: 64, usage: BUFFER_USAGE.VERTEX });
  badVb.destroy();
  const validIb = device.createBuffer({ size: 64, usage: BUFFER_USAGE.INDEX });
  const validBg = { kind: 'GPUBindGroup', handle: 777 };

  executor.beginPass();
  executor.setPipeline(realPipeline);
  executor.setVertexBuffer(0, validVb, 0);
  executor.setVertexBuffer(1, badVb, 0); // destroyed -> guarded & skipped
  executor.setBindGroup(0, validBg);
  executor.setBindGroup(1, { kind: 'GPUBindGroup', destroyed: true }); // destroyed -> skipped
  executor.setIndexBuffer(validIb, 'uint16', 0);

  const res = executor.drawIndexed(pass, 12, 1, 0, 0, 0);
  assert.equal(res.executed, true);
  assert.equal(res.isFallback, false);
  assert.equal(res.indexed, true);
  assert.equal(res.indexCount, 12);

  // Only valid slot 0 VB and group 0 BG were submitted to the render pass.
  const vbOps = calls.pass.filter((c) => c.op === 'setVertexBuffer');
  const bgOps = calls.pass.filter((c) => c.op === 'setBindGroup');
  const drawOps = calls.pass.filter((c) => c.op === 'drawIndexed');
  assert.equal(vbOps.length, 1);
  assert.equal(vbOps[0].slot, 0);
  assert.equal(bgOps.length, 1);
  assert.equal(bgOps[0].groupIndex, 0);
  assert.equal(drawOps.length, 1);
  assert.equal(drawOps[0].indexCount, 12);

  // Out-of-range indexed draw skips safely instead of triggering OOB index validation error.
  const oob = executor.drawIndexed(pass, 1000, 1, 0, 0, 0);
  assert.equal(oob.executed, false);
  assert.equal(oob.skipped, true);
  assert.equal(oob.reason, 'invalid-index-buffer');

  const stats = executor.getStats();
  assert.equal(stats.totalDraws, 1);
  assert.equal(stats.normalDraws, 1);
  assert.equal(stats.indexedDraws, 1);
  assert.equal(stats.fallbackDraws, 0);
  assert.equal(stats.skippedVertexBufferBindings, 2); // slot 1 skipped on both calls
  assert.equal(stats.skippedBindGroupBindings, 2); // group 1 skipped on both calls
  assert.equal(stats.skippedDraws, 1);

  // Mutating getStats() output cannot corrupt internal stats, and resetStats() clears counters.
  stats.fallbackProfileCounts.default = 999;
  assert.equal(executor.getStats().fallbackProfileCounts.default, 0);
  executor.resetStats();
  assert.equal(executor.getStats().totalDraws, 0);
  assert.equal(executor.getStats().skippedDraws, 0);
});

test('G31 classic Worker importScripts IIFE isolation: double load causes zero global leaks', () => {
  const routerSrc = readFileSync(ROUTER_PATH, 'utf8');
  const executorSrc = readFileSync(DRAW_EXECUTOR_PATH, 'utf8');
  const sandbox = { console, performance };
  const vmCtx = vm.createContext(sandbox);
  vmCtx.self = vmCtx;

  // Execute both scripts twice as in repeated classic-worker importScripts() calls.
  assert.doesNotThrow(() => {
    vm.runInContext(routerSrc, vmCtx, { filename: ROUTER_PATH });
    vm.runInContext(executorSrc, vmCtx, { filename: DRAW_EXECUTOR_PATH });
    vm.runInContext(routerSrc, vmCtx, { filename: ROUTER_PATH });
    vm.runInContext(executorSrc, vmCtx, { filename: DRAW_EXECUTOR_PATH });
  });

  // Confirm zero top-level const/let/class declarations leaked into WorkerGlobalScope.
  assert.doesNotThrow(() => {
    vm.runInContext(
      `
      const FALLBACK_PROFILE = {};
      const BUFFER_USAGE = {};
      const TEXTURE_USAGE = {};
      const exportedApi = {};
      class DrawExecutor {}
      `,
      vmCtx
    );
  });

  assert.ok(vmCtx.PlayGta6DrawExecutor);
  assert.equal(typeof vmCtx.PlayGta6DrawExecutor.DrawExecutor, 'function');
  assert.equal(typeof vmCtx.PlayGta6DrawExecutor.selectFallbackShaderProfile, 'function');
  assert.equal(
    vmCtx.PlayGta6DrawExecutor.selectFallbackShaderProfile({
      vertexCount: 3,
      hasBoundCbv: true,
    }),
    'fullscreen_triangle_cbv'
  );
});
