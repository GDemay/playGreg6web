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
  FRAME1_SPLASH_VERTEX_LAYOUT_24B,
  expandRectListVertices24B,
  selectFallbackShaderProfile,
  DrawExecutor,
} = drawExecutorMod;

const { ShaderFallbackRouter, BUILTIN_FALLBACK_WGSL, getBuiltInFallbackProfiles } = routerMod;

const BUFFER_USAGE = Object.freeze({
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  INDIRECT: 0x0100,
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
    drawIndirect(indirectBuffer, indirectOffset) {
      if (overrides.failDrawIndirect) {
        throw new Error('mock drawIndirect failure');
      }
      calls.pass.push({
        op: 'drawIndirect',
        indirectBuffer,
        indirectOffset,
      });
    },
    drawIndexedIndirect(indirectBuffer, indirectOffset) {
      if (overrides.failDrawIndexedIndirect) {
        throw new Error('mock drawIndexedIndirect failure');
      }
      calls.pass.push({
        op: 'drawIndexedIndirect',
        indirectBuffer,
        indirectOffset,
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
    SPLASH_RECT_24B: 'splash_rect_24b',
  });
  assert.equal(FALLBACK_VERTEX_LAYOUT_24B.arrayStride, 24);
  assert.deepEqual(FALLBACK_VERTEX_LAYOUT_24B.attributes, [
    { shaderLocation: 0, offset: 0, format: 'float32x4' },
    { shaderLocation: 1, offset: 16, format: 'float32x2' },
  ]);
  assert.equal(FRAME1_SPLASH_VERTEX_LAYOUT_24B.arrayStride, 24);
  assert.equal(FRAME1_SPLASH_VERTEX_LAYOUT_24B.stepMode, 'vertex');
  assert.deepEqual(FRAME1_SPLASH_VERTEX_LAYOUT_24B.attributes, [
    { shaderLocation: 0, offset: 0, format: 'float32x3' },
    { shaderLocation: 1, offset: 12, format: 'unorm8x4' },
    { shaderLocation: 2, offset: 16, format: 'float32x2' },
  ]);
  assert.equal(typeof expandRectListVertices24B, 'function');
  assert.equal(DrawExecutor.expandRectListVertices24B, expandRectListVertices24B);
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
    splash_rect_24b: 0,
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
    splash_rect_24b: 0,
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
    splash_rect_24b: 0,
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
      const FRAME1_SPLASH_VERTEX_LAYOUT_24B = {};
      const BUFFER_USAGE = {};
      const TEXTURE_USAGE = {};
      const exportedApi = {};
      function expandRectListVertices24B() {}
      class DrawExecutor {}
      `,
      vmCtx
    );
  });

  assert.ok(vmCtx.PlayGta6DrawExecutor);
  assert.equal(typeof vmCtx.PlayGta6DrawExecutor.DrawExecutor, 'function');
  assert.equal(typeof vmCtx.PlayGta6DrawExecutor.selectFallbackShaderProfile, 'function');
  assert.equal(typeof vmCtx.PlayGta6DrawExecutor.expandRectListVertices24B, 'function');
  assert.ok(vmCtx.PlayGta6DrawExecutor.FRAME1_SPLASH_VERTEX_LAYOUT_24B);
  assert.equal(
    vmCtx.PlayGta6DrawExecutor.selectFallbackShaderProfile({
      vertexCount: 3,
      hasBoundCbv: true,
    }),
    'fullscreen_triangle_cbv'
  );
  assert.equal(
    vmCtx.PlayGta6DrawExecutor.selectFallbackShaderProfile({
      vertexCount: 3,
      vertexStride: 24,
      topology: 'rect-list',
    }),
    'splash_rect_24b'
  );
});

// ---------------------------------------------------------------------------
// G47 Xbox RECTLIST 3->6 vertex expansion & Frame-1 24B splash layout tests
// ---------------------------------------------------------------------------

function packSplashVertex24B(pos, color, uv) {
  const buf = new Uint8Array(24);
  const view = new DataView(buf.buffer);
  view.setFloat32(0, pos[0], true);
  view.setFloat32(4, pos[1], true);
  view.setFloat32(8, pos[2], true);
  buf[12] = color[0] & 0xff;
  buf[13] = color[1] & 0xff;
  buf[14] = color[2] & 0xff;
  buf[15] = color[3] & 0xff;
  view.setFloat32(16, uv[0], true);
  view.setFloat32(20, uv[1], true);
  return buf;
}

function packSplashVertices24B(verts) {
  const out = new Uint8Array(verts.length * 24);
  for (let i = 0; i < verts.length; i += 1) {
    out.set(packSplashVertex24B(verts[i].pos, verts[i].color, verts[i].uv), i * 24);
  }
  return out;
}

function unpackSplashVertex24B(bytes, index) {
  const off = index * 24;
  const view = new DataView(bytes.buffer, bytes.byteOffset + off, 24);
  return {
    pos: [view.getFloat32(0, true), view.getFloat32(4, true), view.getFloat32(8, true)],
    color: [bytes[off + 12], bytes[off + 13], bytes[off + 14], bytes[off + 15]],
    uv: [view.getFloat32(16, true), view.getFloat32(20, true)],
  };
}

function assertApproxArray(actual, expected, eps = 1e-5) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i += 1) {
    assert.ok(
      Math.abs(actual[i] - expected[i]) <= eps,
      `index ${i}: expected ${expected[i]}, got ${actual[i]}`
    );
  }
}

test('G47 expandRectListVertices24B expands 3-vertex RECTLIST quad to 6 triangle-list vertices with exact v3 math', () => {
  const v0 = { pos: [-1.0, 1.0, 0.25], color: [0x11, 0x22, 0x33, 0xff], uv: [0.0, 0.0] };
  const v1 = { pos: [1.0, 1.0, 0.25], color: [0x44, 0x55, 0x66, 0xfe], uv: [1.0, 0.0] };
  const v2 = { pos: [-1.0, -1.0, 0.25], color: [0x77, 0x88, 0x99, 0xee], uv: [0.0, 1.0] };
  const expectedV3 = {
    pos: [1.0, -1.0, 0.25], // v1 + v2 - v0
    color: [0x77, 0x88, 0x99, 0xee], // color3 = color2 bytes 12..15
    uv: [1.0, 1.0], // uv1 + uv2 - uv0
  };

  const src = packSplashVertices24B([v0, v1, v2]);
  const out = expandRectListVertices24B(src, 3);
  assert.ok(out instanceof Uint8Array);
  assert.equal(out.byteLength, 6 * 24);

  const expectedOrder = [v0, v1, v2, v1, expectedV3, v2];
  for (let i = 0; i < 6; i += 1) {
    const actual = unpackSplashVertex24B(out, i);
    assertApproxArray(actual.pos, expectedOrder[i].pos);
    assert.deepEqual(actual.color, expectedOrder[i].color);
    assertApproxArray(actual.uv, expectedOrder[i].uv);
  }
});

test('G47 expandRectListVertices24B handles multi-rect batches, skewed/3D coords, partial trailing verts, and subarrays', () => {
  // 2 complete rects + 2 trailing partial vertices (vertexCount = 8 -> floor(8/3) = 2 rects -> 12 output vertices).
  const r0 = [
    { pos: [10.0, 20.0, 0.5], color: [10, 20, 30, 40], uv: [0.1, 0.2] },
    { pos: [14.0, 21.0, 0.75], color: [50, 60, 70, 80], uv: [0.6, 0.25] },
    { pos: [11.0, 28.0, 1.0], color: [90, 100, 110, 120], uv: [0.15, 0.85] },
  ];
  const r1 = [
    { pos: [-0.5, 0.5, 0.0], color: [255, 0, 0, 255], uv: [0.25, 0.25] },
    { pos: [0.5, 0.5, 0.0], color: [0, 255, 0, 255], uv: [0.75, 0.25] },
    { pos: [-0.5, -0.5, 0.0], color: [0, 0, 255, 200], uv: [0.25, 0.75] },
  ];
  const trailing = [
    { pos: [99.0, 99.0, 99.0], color: [1, 2, 3, 4], uv: [0.0, 0.0] },
    { pos: [88.0, 88.0, 88.0], color: [5, 6, 7, 8], uv: [1.0, 1.0] },
  ];

  // Wrap inside a padded buffer with non-zero byteOffset to verify DataView byteOffset handling.
  const payload = packSplashVertices24B([...r0, ...r1, ...trailing]);
  const padded = new Uint8Array(16 + payload.byteLength + 8);
  padded.fill(0xaa);
  padded.set(payload, 16);
  const sub = padded.subarray(16, 16 + payload.byteLength);

  const out = expandRectListVertices24B(sub, 8);
  assert.equal(out.byteLength, 12 * 24);

  // Verify rect 0 v3 (output index 4): pos1 + pos2 - pos0, uv1 + uv2 - uv0, color2
  const r0v3 = unpackSplashVertex24B(out, 4);
  assertApproxArray(r0v3.pos, [14.0 + 11.0 - 10.0, 21.0 + 28.0 - 20.0, 0.75 + 1.0 - 0.5]);
  assert.deepEqual(r0v3.color, [90, 100, 110, 120]);
  assertApproxArray(r0v3.uv, [0.6 + 0.15 - 0.1, 0.25 + 0.85 - 0.2]);

  // Verify rect 1 v3 (output index 10)
  const r1v3 = unpackSplashVertex24B(out, 10);
  assertApproxArray(r1v3.pos, [0.5, -0.5, 0.0]);
  assert.deepEqual(r1v3.color, [0, 0, 255, 200]);
  assertApproxArray(r1v3.uv, [0.75, 0.75]);

  // Explicit vertexCount smaller than buffer restricts expansion to floor(3/3) = 1 rect.
  const oneRect = expandRectListVertices24B(sub, 3);
  assert.equal(oneRect.byteLength, 6 * 24);

  // Omitted vertexCount infers floor(byteLength / 24) = 8 -> 2 rects.
  const inferred = expandRectListVertices24B(sub);
  assert.equal(inferred.byteLength, 12 * 24);

  // Edge cases: < 3 vertices, zero, negative, null/undefined -> empty Uint8Array(0).
  assert.equal(expandRectListVertices24B(sub, 0).byteLength, 0);
  assert.equal(expandRectListVertices24B(sub, 1).byteLength, 0);
  assert.equal(expandRectListVertices24B(sub, 2).byteLength, 0);
  assert.equal(expandRectListVertices24B(new Uint8Array(48), 2).byteLength, 0);
  assert.equal(expandRectListVertices24B(new Uint8Array(0), 3).byteLength, 0);
  assert.equal(expandRectListVertices24B(null, 3).byteLength, 0);
  assert.equal(expandRectListVertices24B(undefined, 3).byteLength, 0);
});

test('G47 selectFallbackShaderProfile routes rect-list / isRectList with 24B stride to splash_rect_24b', () => {
  // topology === 'rect-list' + 24B stride -> splash_rect_24b (even with vertexCount: 3 + CBV)
  assert.equal(
    selectFallbackShaderProfile({
      topology: 'rect-list',
      vertexCount: 3,
      vertexStride: 24,
      hasBoundCbv: true,
    }),
    'splash_rect_24b'
  );

  // topology === 'rect-list' + 24B VB + CBV + texture + sampler -> splash_rect_24b
  assert.equal(
    selectFallbackShaderProfile({
      topology: 'rect-list',
      vertexCount: 3,
      vertexBuffers: [{ slot: 0, stride: 24, buffer: { size: 72 } }],
      hasBoundCbv: true,
      hasBoundTexture: true,
      hasBoundSampler: true,
    }),
    'splash_rect_24b'
  );

  // isRectList === true + 24B vertexBufferLayouts -> splash_rect_24b
  assert.equal(
    selectFallbackShaderProfile({
      isRectList: true,
      vertexCount: 6,
      vertexBufferLayouts: [{ arrayStride: 24 }],
    }),
    'splash_rect_24b'
  );

  // pipeline.topology === 'RECTLIST' + 24B stride -> splash_rect_24b
  assert.equal(
    selectFallbackShaderProfile({
      pipeline: { topology: 'RECTLIST' },
      vertexCount: 3,
      vertexStride: 24,
    }),
    'splash_rect_24b'
  );

  // rect-list WITHOUT 24B stride does NOT route to splash_rect_24b
  assert.equal(
    selectFallbackShaderProfile({
      topology: 'rect-list',
      vertexCount: 3,
      vertexStride: 32,
      hasBoundCbv: true,
    }),
    'fullscreen_triangle_cbv'
  );
  assert.equal(
    selectFallbackShaderProfile({
      isRectList: true,
      vertexCount: 6,
      vertexStride: 32,
    }),
    'default'
  );
});

test('G47 DrawExecutor executes rect-list 24B draws via splash_rect_24b with triangle-list topology, exact 24B layout, and cb0/tex0/samp0 bindings', () => {
  const { device, pass, calls } = createMockWebGpu();
  const router = new ShaderFallbackRouter({ now: () => 500 });
  const executor = new DrawExecutor({ device, shaderFallbackRouter: router });
  const profiles = getBuiltInFallbackProfiles();

  const cbv = device.createBuffer({ size: 256, usage: BUFFER_USAGE.UNIFORM });
  const vb = device.createBuffer({ size: 6 * 24, usage: BUFFER_USAGE.VERTEX });
  const tex = device.createTexture({
    size: { width: 32, height: 32, depthOrArrayLayers: 1 },
    format: 'rgba8unorm',
    usage: 0x04,
  });
  const texView = tex.createView();
  const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  executor.beginPass();
  executor.setPipeline({
    id: 470,
    vertexShaderHash: '0x7777777777777777',
    fragmentShaderHash: '0x8888888888888888',
    vertexStride: 24,
    topology: 'rect-list',
    isFallback: true,
  });
  executor.setConstantBuffer(0, cbv, { offset: 0, size: 256 });
  executor.setTexture(0, texView);
  executor.setSampler(0, sampler);
  executor.setVertexBuffer(0, vb, { offset: 0, stride: 24 });

  const res = executor.executeDraw(pass, { vertexCount: 6, instanceCount: 1 });
  assert.equal(res.executed, true);
  assert.equal(res.profile, 'splash_rect_24b');
  assert.equal(res.effectiveProfile, 'splash_rect_24b');
  assert.equal(res.substitutedCbv, false);
  assert.equal(res.substitutedTexture, false);
  assert.equal(res.substitutedSampler, false);
  assert.equal(res.substitutedVertexBuffer, false);

  // Compiled splash_rect_24b shaders.
  assert.equal(calls.createShaderModule[0].desc.code, profiles.splash_rect_24b.vertex);
  assert.equal(calls.createShaderModule[1].desc.code, profiles.splash_rect_24b.fragment);

  // Pipeline normalized 'rect-list' -> 'triangle-list' and used exact Frame-1 24B layout.
  const pipeDesc = calls.createRenderPipeline[0].desc;
  assert.equal(pipeDesc.primitive.topology, 'triangle-list');
  assert.equal(pipeDesc.vertex.buffers.length, 1);
  assert.equal(pipeDesc.vertex.buffers[0].arrayStride, 24);
  assert.deepEqual(pipeDesc.vertex.buffers[0].attributes, [
    { shaderLocation: 0, offset: 0, format: 'float32x3' },
    { shaderLocation: 1, offset: 12, format: 'unorm8x4' },
    { shaderLocation: 2, offset: 16, format: 'float32x2' },
  ]);

  // 3-entry bind group: binding 0 = cb0, binding 1 = tex0, binding 2 = samp0.
  assert.equal(calls.createBindGroup[0].desc.entries.length, 3);
  assert.equal(calls.createBindGroup[0].desc.entries[0].binding, 0);
  assert.equal(calls.createBindGroup[0].desc.entries[0].resource.buffer, cbv);
  assert.equal(calls.createBindGroup[0].desc.entries[1].binding, 1);
  assert.equal(calls.createBindGroup[0].desc.entries[1].resource, texView);
  assert.equal(calls.createBindGroup[0].desc.entries[2].binding, 2);
  assert.equal(calls.createBindGroup[0].desc.entries[2].resource, sampler);

  assert.equal(executor.getStats().fallbackProfileCounts.splash_rect_24b, 1);

  // Missing CBV/texture/sampler/VB on a 3-vertex rect-list draw hardens cleanly and expands to 6 vertices.
  executor.beginPass();
  executor.setPipeline({ vertexStride: 24, topology: 'rect-list', isFallback: true });
  const hardened = executor.executeDraw(pass, { vertexCount: 3, instanceCount: 1 });
  assert.equal(hardened.executed, true);
  assert.equal(hardened.profile, 'splash_rect_24b');
  assert.equal(hardened.substitutedCbv, true);
  assert.equal(hardened.substitutedTexture, true);
  assert.equal(hardened.substitutedSampler, true);
  assert.equal(hardened.substitutedVertexBuffer, true);
  assert.equal(hardened.vertexCount, 6);
  const synthSplashVb = calls.createBuffer.find(
    (c) => c.desc.label === 'draw_executor:fallback_vb_24b_splash_rect'
  );
  assert.ok(synthSplashVb);
  assert.equal(synthSplashVb.desc.size, 144);
});

// ---------------------------------------------------------------------------
// W130 Redundant pipeline elision, zero-count draw guard & per-kind telemetry
// ---------------------------------------------------------------------------

test('W130 redundant setPipeline in one pass elides pass.setPipeline and re-binds cleanly on new pass / endPass / reset', () => {
  const { device, pass, calls } = createMockWebGpu();
  const router = new ShaderFallbackRouter();
  const executor = new DrawExecutor({ device, shaderFallbackRouter: router });

  const pipeA = {
    kind: 'GPURenderPipeline',
    handle: 1001,
    getBindGroupLayout: () => ({ kind: 'GPUBindGroupLayout' }),
  };
  const pipeB = {
    kind: 'GPURenderPipeline',
    handle: 1002,
    getBindGroupLayout: () => ({ kind: 'GPUBindGroupLayout' }),
  };

  // Pass 1: redundant setPipeline(pipeA) elides pass.setPipeline once.
  executor.beginPass();
  executor.setPipeline(pipeA);
  const d1 = executor.draw(pass, 3, 1);
  assert.equal(d1.executed, true);
  assert.equal(d1.pipelineElided, false);

  executor.setPipeline(pipeA);
  const d2 = executor.draw(pass, 3, 1);
  assert.equal(d2.executed, true);
  assert.equal(d2.pipelineElided, true);

  let setPipeOps = calls.pass.filter((c) => c.op === 'setPipeline');
  assert.equal(setPipeOps.length, 1);
  assert.equal(setPipeOps[0].pipeline, pipeA);
  assert.equal(executor.getStats().redundantPipelineSkips, 1);

  // Switching to a different pipeline binds it immediately.
  executor.setPipeline(pipeB);
  const d3 = executor.draw(pass, 6, 1);
  assert.equal(d3.executed, true);
  assert.equal(d3.pipelineElided, false);
  setPipeOps = calls.pass.filter((c) => c.op === 'setPipeline');
  assert.equal(setPipeOps.length, 2);
  assert.equal(setPipeOps[1].pipeline, pipeB);
  assert.equal(executor.getStats().redundantPipelineSkips, 1);

  // Starting a new pass resets bound pipeline tracking so pipeB binds again.
  executor.beginPass();
  executor.setPipeline(pipeB);
  const d4 = executor.draw(pass, 6, 1);
  assert.equal(d4.executed, true);
  assert.equal(d4.pipelineElided, false);
  setPipeOps = calls.pass.filter((c) => c.op === 'setPipeline');
  assert.equal(setPipeOps.length, 3);
  assert.equal(executor.getStats().redundantPipelineSkips, 1);

  // endPass() also clears bound pipeline tracking even when the same pass object is reused.
  executor.endPass();
  const d5 = executor.draw(pass, 6, 1);
  assert.equal(d5.executed, true);
  assert.equal(d5.pipelineElided, false);
  setPipeOps = calls.pass.filter((c) => c.op === 'setPipeline');
  assert.equal(setPipeOps.length, 4);

  // reset() clears bound pipeline state without wiping telemetry unless { stats: true }.
  executor.reset();
  executor.setPipeline(pipeA);
  const d6 = executor.draw(pass, 3, 1);
  assert.equal(d6.executed, true);
  assert.equal(d6.pipelineElided, false);
  setPipeOps = calls.pass.filter((c) => c.op === 'setPipeline');
  assert.equal(setPipeOps.length, 5);
  assert.equal(executor.getStats().redundantPipelineSkips, 1);

  // Opt-out option elideRedundantPipelines: false always calls pass.setPipeline.
  const noElideEnv = createMockWebGpu();
  const noElideExec = new DrawExecutor({
    device: noElideEnv.device,
    shaderFallbackRouter: router,
    elideRedundantPipelines: false,
  });
  noElideExec.beginPass();
  noElideExec.setPipeline(pipeA);
  noElideExec.draw(noElideEnv.pass, 3, 1);
  noElideExec.setPipeline(pipeA);
  noElideExec.draw(noElideEnv.pass, 3, 1);
  assert.equal(
    noElideEnv.calls.pass.filter((c) => c.op === 'setPipeline').length,
    2
  );
  assert.equal(noElideExec.getStats().redundantPipelineSkips, 0);
});

test('W130 zero-count draw guard skips zero vertex, index, and instance draws cleanly', () => {
  const { device, pass, calls } = createMockWebGpu();
  const router = new ShaderFallbackRouter();
  const executor = new DrawExecutor({ device, shaderFallbackRouter: router });

  const pipe = {
    kind: 'GPURenderPipeline',
    handle: 2001,
    getBindGroupLayout: () => ({ kind: 'GPUBindGroupLayout' }),
  };
  const ib = device.createBuffer({ size: 64, usage: BUFFER_USAGE.INDEX });

  executor.beginPass();
  executor.setPipeline(pipe);
  executor.setIndexBuffer(ib, 'uint16', 0);

  const zVert = executor.draw(pass, 0, 1);
  const zVertInst = executor.draw(pass, 6, 0);
  const zIdx = executor.drawIndexed(pass, 0, 1);
  const zIdxInst = executor.drawIndexed(pass, 12, 0);

  for (const res of [zVert, zVertInst, zIdx, zIdxInst]) {
    assert.equal(res.executed, false);
    assert.equal(res.skipped, true);
    assert.equal(res.reason, 'zero-draw-count');
  }

  // Zero WebGPU draw or setPipeline calls issued.
  assert.equal(calls.pass.length, 0);

  const stats = executor.getStats();
  assert.equal(stats.zeroCountDrawsSkipped, 4);
  assert.equal(stats.skippedDraws, 4);
  assert.equal(stats.totalDraws, 0);
  assert.equal(stats.indexedDraws, 0);
  assert.equal(stats.nonIndexedDraws, 0);
  assert.equal(stats.indirectDraws, 0);
});

test('W130 per-kind draw telemetry tracks indexedDraws, nonIndexedDraws, and indirectDraws across mixed draws', () => {
  const { device, pass, calls } = createMockWebGpu();
  const router = new ShaderFallbackRouter();
  const executor = new DrawExecutor({ device, shaderFallbackRouter: router });

  const pipe = {
    kind: 'GPURenderPipeline',
    handle: 3001,
    getBindGroupLayout: () => ({ kind: 'GPUBindGroupLayout' }),
  };
  const vb = device.createBuffer({ size: 256, usage: BUFFER_USAGE.VERTEX });
  const ib = device.createBuffer({ size: 128, usage: BUFFER_USAGE.INDEX });
  const indirectBuf = device.createBuffer({ size: 64, usage: BUFFER_USAGE.INDIRECT });

  executor.beginPass();
  executor.setPipeline(pipe);
  executor.setVertexBuffer(0, vb, 0);
  executor.setIndexBuffer(ib, 'uint16', 0);

  // 2 non-indexed draws
  assert.equal(executor.draw(pass, 3, 1).executed, true);
  assert.equal(executor.draw(pass, 6, 2).executed, true);

  // 2 indexed draws
  assert.equal(executor.drawIndexed(pass, 12, 1, 0, 0, 0).executed, true);
  assert.equal(executor.drawIndexed(pass, 18, 2, 0, 0, 0).executed, true);

  // 2 indirect draws (1 non-indexed indirect, 1 indexed indirect)
  const ind1 = executor.drawIndirect(pass, indirectBuf, 0);
  assert.equal(ind1.executed, true);
  assert.equal(ind1.indirect, true);
  assert.equal(ind1.indexed, false);

  const ind2 = executor.drawIndexedIndirect(pass, indirectBuf, 20);
  assert.equal(ind2.executed, true);
  assert.equal(ind2.indirect, true);
  assert.equal(ind2.indexed, true);

  // Invalid indirect buffer (undersized / misaligned) skips cleanly without throwing.
  const badInd = executor.drawIndirect(pass, indirectBuf, 2); // misaligned offset
  assert.equal(badInd.executed, false);
  assert.equal(badInd.skipped, true);
  assert.equal(badInd.reason, 'invalid-indirect-buffer');

  const stats = executor.getStats();
  assert.equal(stats.totalDraws, 6);
  assert.equal(stats.normalDraws, 6);
  assert.equal(stats.nonIndexedDraws, 2);
  assert.equal(stats.indexedDraws, 2);
  assert.equal(stats.indirectDraws, 2);
  assert.equal(stats.skippedDraws, 1);
  assert.equal(stats.zeroCountDrawsSkipped, 0);
  // All 6 draws shared the same pipeline in one pass -> 1 bind, 5 elided.
  assert.equal(stats.redundantPipelineSkips, 5);
  assert.equal(calls.pass.filter((c) => c.op === 'setPipeline').length, 1);
  assert.equal(calls.pass.filter((c) => c.op === 'draw').length, 2);
  assert.equal(calls.pass.filter((c) => c.op === 'drawIndexed').length, 2);
  assert.equal(calls.pass.filter((c) => c.op === 'drawIndirect').length, 1);
  assert.equal(calls.pass.filter((c) => c.op === 'drawIndexedIndirect').length, 1);

  // resetStats() zeroes all new telemetry counters as well.
  executor.resetStats();
  const cleared = executor.getStats();
  assert.equal(cleared.indexedDraws, 0);
  assert.equal(cleared.nonIndexedDraws, 0);
  assert.equal(cleared.indirectDraws, 0);
  assert.equal(cleared.redundantPipelineSkips, 0);
  assert.equal(cleared.zeroCountDrawsSkipped, 0);
});

// ---------------------------------------------------------------------------
// W166 High-frequency menu/video redundant-state elision, bind-group cache,
// dynamic-offset safety, UI/glyph/video accounting, and per-reason skip stats
// ---------------------------------------------------------------------------

test('W166 redundant setBindGroup / setVertexBuffer / setIndexBuffer elision within one pass and invalidation on change or unbind', () => {
  const { device, pass, calls } = createMockWebGpu();
  const router = new ShaderFallbackRouter();
  const executor = new DrawExecutor({ device, shaderFallbackRouter: router });

  const pipe = {
    kind: 'GPURenderPipeline',
    handle: 4001,
    getBindGroupLayout: () => ({ kind: 'GPUBindGroupLayout' }),
  };
  const vb0 = device.createBuffer({ size: 512, usage: BUFFER_USAGE.VERTEX });
  const vb1 = device.createBuffer({ size: 512, usage: BUFFER_USAGE.VERTEX });
  const ib0 = device.createBuffer({ size: 256, usage: BUFFER_USAGE.INDEX });
  const ib1 = device.createBuffer({ size: 256, usage: BUFFER_USAGE.INDEX });
  const bg0 = { kind: 'GPUBindGroup', handle: 4100 };
  const bg1 = { kind: 'GPUBindGroup', handle: 4101 };
  const bg2 = { kind: 'GPUBindGroup', handle: 4102 };

  executor.beginPass();
  executor.setPipeline(pipe);
  executor.setVertexBuffer(0, vb0, 0, 256);
  executor.setVertexBuffer(1, vb1, 16);
  executor.setIndexBuffer(ib0, 'uint16', 0, 128);
  executor.setBindGroup(0, bg0);
  executor.setBindGroup(1, bg1);

  // Draw 1: cold pass binds everything once.
  const d1 = executor.drawIndexed(pass, 6, 1, 0, 0, 0);
  assert.equal(d1.executed, true);
  assert.equal(d1.pipelineElided, false);
  assert.equal(d1.bindGroupsElided, 0);
  assert.equal(d1.vertexBuffersElided, 0);
  assert.equal(d1.indexBufferElided, false);

  // Draw 2: identical state in same pass elides pipeline, both VBs, both BGs, and IB.
  const d2 = executor.drawIndexed(pass, 6, 1, 6, 0, 0);
  assert.equal(d2.executed, true);
  assert.equal(d2.pipelineElided, true);
  assert.equal(d2.bindGroupsElided, 2);
  assert.equal(d2.vertexBuffersElided, 2);
  assert.equal(d2.indexBufferElided, true);

  assert.equal(calls.pass.filter((c) => c.op === 'setPipeline').length, 1);
  assert.equal(calls.pass.filter((c) => c.op === 'setBindGroup').length, 2);
  assert.equal(calls.pass.filter((c) => c.op === 'setVertexBuffer').length, 2);
  assert.equal(calls.pass.filter((c) => c.op === 'setIndexBuffer').length, 1);

  // Draw 3: changing VB0 offset, BG1 handle, and IB format forces ONLY those slots to re-bind.
  executor.setVertexBuffer(0, vb0, 64, 256);
  executor.setBindGroup(1, bg2);
  executor.setIndexBuffer(ib0, 'uint32', 0, 128);
  const d3 = executor.drawIndexed(pass, 6, 1, 0, 0, 0);
  assert.equal(d3.executed, true);
  assert.equal(d3.pipelineElided, true);
  assert.equal(d3.bindGroupsElided, 1); // bg0 elided, bg2 bound
  assert.equal(d3.vertexBuffersElided, 1); // vb1 elided, vb0 bound
  assert.equal(d3.indexBufferElided, false); // uint16 -> uint32 bound

  assert.equal(calls.pass.filter((c) => c.op === 'setBindGroup').length, 3);
  assert.equal(calls.pass.filter((c) => c.op === 'setVertexBuffer').length, 3);
  assert.equal(calls.pass.filter((c) => c.op === 'setIndexBuffer').length, 2);

  // Draw 4: changing IB buffer (same format/offset) re-binds IB, elides rest.
  executor.setIndexBuffer(ib1, 'uint32', 0, 128);
  const d4 = executor.drawIndexed(pass, 6, 1, 0, 0, 0);
  assert.equal(d4.executed, true);
  assert.equal(d4.indexBufferElided, false);
  assert.equal(calls.pass.filter((c) => c.op === 'setIndexBuffer').length, 3);

  // Unbinding slot via null invalidates tracked pass binding for that slot.
  executor.setBindGroup(0, null);
  executor.setBindGroup(0, bg0);
  const d5 = executor.drawIndexed(pass, 6, 1, 0, 0, 0);
  assert.equal(d5.executed, true);
  assert.equal(d5.bindGroupsElided, 1); // bg2 elided, bg0 re-bound after explicit null unbind
  assert.equal(calls.pass.filter((c) => c.op === 'setBindGroup').length, 4);

  const stats = executor.getStats();
  assert.equal(stats.redundantPipelineSkips, 4);
  assert.equal(stats.redundantBindGroupSkips, 6); // 2 + 1 + 2 + 1
  assert.equal(stats.redundantVertexBufferSkips, 7); // 2 + 1 + 2 + 2
  assert.equal(stats.redundantIndexBufferSkips, 2); // d2 + d5

  // Opt-out via elideRedundantState: false disables all redundant-state elision.
  const noElideEnv = createMockWebGpu();
  const noElideExec = new DrawExecutor({
    device: noElideEnv.device,
    shaderFallbackRouter: router,
    elideRedundantState: false,
  });
  noElideExec.beginPass();
  noElideExec.setPipeline(pipe);
  noElideExec.setVertexBuffer(0, vb0, 0);
  noElideExec.setBindGroup(0, bg0);
  noElideExec.setIndexBuffer(ib0, 'uint16', 0);
  noElideExec.drawIndexed(noElideEnv.pass, 6, 1, 0, 0, 0);
  noElideExec.drawIndexed(noElideEnv.pass, 6, 1, 0, 0, 0);
  assert.equal(noElideEnv.calls.pass.filter((c) => c.op === 'setPipeline').length, 2);
  assert.equal(noElideEnv.calls.pass.filter((c) => c.op === 'setBindGroup').length, 2);
  assert.equal(noElideEnv.calls.pass.filter((c) => c.op === 'setVertexBuffer').length, 2);
  assert.equal(noElideEnv.calls.pass.filter((c) => c.op === 'setIndexBuffer').length, 2);
  const noElideStats = noElideExec.getStats();
  assert.equal(noElideStats.redundantPipelineSkips, 0);
  assert.equal(noElideStats.redundantBindGroupSkips, 0);
  assert.equal(noElideStats.redundantVertexBufferSkips, 0);
  assert.equal(noElideStats.redundantIndexBufferSkips, 0);
});

test('W166 pass-boundary invalidation re-binds all state on endPass, beginPass, and implicit pass-object switch', () => {
  const env1 = createMockWebGpu();
  const env2 = createMockWebGpu();
  const router = new ShaderFallbackRouter();
  const executor = new DrawExecutor({ device: env1.device, shaderFallbackRouter: router });

  const pipe = {
    kind: 'GPURenderPipeline',
    handle: 4201,
    getBindGroupLayout: () => ({ kind: 'GPUBindGroupLayout' }),
  };
  const vb = env1.device.createBuffer({ size: 256, usage: BUFFER_USAGE.VERTEX });
  const ib = env1.device.createBuffer({ size: 128, usage: BUFFER_USAGE.INDEX });
  const bg = { kind: 'GPUBindGroup', handle: 4202 };

  executor.beginPass();
  executor.setPipeline(pipe);
  executor.setVertexBuffer(0, vb, 0);
  executor.setIndexBuffer(ib, 'uint16', 0);
  executor.setBindGroup(0, bg);

  // Pass 1 draw 1 (binds) + draw 2 (elides everything).
  executor.drawIndexed(env1.pass, 6, 1, 0, 0, 0);
  const elided = executor.drawIndexed(env1.pass, 6, 1, 0, 0, 0);
  assert.equal(elided.pipelineElided, true);
  assert.equal(elided.bindGroupsElided, 1);
  assert.equal(elided.vertexBuffersElided, 1);
  assert.equal(elided.indexBufferElided, true);

  // Implicitly switching to a second GPURenderPassEncoder without calling endPass() MUST re-bind everything!
  const onPass2 = executor.drawIndexed(env2.pass, 6, 1, 0, 0, 0);
  assert.equal(onPass2.executed, true);
  assert.equal(onPass2.pipelineElided, false);
  assert.equal(onPass2.bindGroupsElided, 0);
  assert.equal(onPass2.vertexBuffersElided, 0);
  assert.equal(onPass2.indexBufferElided, false);
  assert.deepEqual(
    env2.calls.pass.map((c) => c.op),
    ['setPipeline', 'setVertexBuffer', 'setBindGroup', 'setIndexBuffer', 'drawIndexed']
  );

  // endPass() invalidates even when drawing again on the exact same pass2 object.
  executor.endPass();
  const afterEnd = executor.drawIndexed(env2.pass, 6, 1, 0, 0, 0);
  assert.equal(afterEnd.executed, true);
  assert.equal(afterEnd.pipelineElided, false);
  assert.equal(afterEnd.bindGroupsElided, 0);
  assert.equal(afterEnd.vertexBuffersElided, 0);
  assert.equal(afterEnd.indexBufferElided, false);
  assert.equal(env2.calls.pass.filter((c) => c.op === 'setBindGroup').length, 2);
  assert.equal(env2.calls.pass.filter((c) => c.op === 'setVertexBuffer').length, 2);
  assert.equal(env2.calls.pass.filter((c) => c.op === 'setIndexBuffer').length, 2);
});

test('W166 dynamic-offset correctness: never elides when dynamic offsets differ or are mutated in place', () => {
  const { device, pass, calls } = createMockWebGpu();
  const router = new ShaderFallbackRouter();
  const executor = new DrawExecutor({ device, shaderFallbackRouter: router });

  const pipe = {
    kind: 'GPURenderPipeline',
    handle: 4301,
    getBindGroupLayout: () => ({ kind: 'GPUBindGroupLayout' }),
  };
  const bg = { kind: 'GPUBindGroup', handle: 4302 };

  executor.beginPass();
  executor.setPipeline(pipe);

  // 1. Initial bind with Uint32Array([0, 256]).
  const dynOffsets = new Uint32Array([0, 256]);
  executor.setBindGroup(0, bg, dynOffsets);
  const d1 = executor.draw(pass, 6, 1);
  assert.equal(d1.bindGroupsElided, 0);

  // 2. Identical values in a new Array instance -> safe to elide!
  executor.setBindGroup(0, bg, [0, 256]);
  const d2 = executor.draw(pass, 6, 1);
  assert.equal(d2.bindGroupsElided, 1);

  // 3. Different dynamic offset value -> MUST NOT elide!
  executor.setBindGroup(0, bg, [256, 256]);
  const d3 = executor.draw(pass, 6, 1);
  assert.equal(d3.bindGroupsElided, 0);

  // 4. Bind mutable Uint32Array, draw, then mutate IN PLACE without reallocating -> MUST NOT elide!
  const mutableOffsets = new Uint32Array([512, 768]);
  executor.setBindGroup(0, bg, mutableOffsets);
  const d4 = executor.draw(pass, 6, 1);
  assert.equal(d4.bindGroupsElided, 0);

  // Mutate in place without calling setBindGroup again:
  mutableOffsets[0] = 1024;
  const d5 = executor.draw(pass, 6, 1);
  assert.equal(d5.bindGroupsElided, 0);

  // Re-calling setBindGroup with the same mutated reference & unchanged values -> safe to elide!
  executor.setBindGroup(0, bg, mutableOffsets);
  const d6 = executor.draw(pass, 6, 1);
  assert.equal(d6.bindGroupsElided, 1);

  // Mutate second slot in place and re-call setBindGroup -> MUST NOT elide!
  mutableOffsets[1] = 1280;
  executor.setBindGroup(0, bg, mutableOffsets);
  const d7 = executor.draw(pass, 6, 1);
  assert.equal(d7.bindGroupsElided, 0);

  // 5. Different length ([1024] vs [1024, 1280]) -> MUST NOT elide!
  executor.setBindGroup(0, bg, [1024]);
  const d8 = executor.draw(pass, 6, 1);
  assert.equal(d8.bindGroupsElided, 0);

  // 6. Switching from dynamic offsets to no dynamic offsets (null) -> MUST NOT elide!
  executor.setBindGroup(0, bg);
  const d9 = executor.draw(pass, 6, 1);
  assert.equal(d9.bindGroupsElided, 0);

  // 7. Switching from null dynamic offsets to empty array [] -> MUST NOT elide!
  executor.setBindGroup(0, bg, []);
  const d10 = executor.draw(pass, 6, 1);
  assert.equal(d10.bindGroupsElided, 0);

  const bgCalls = calls.pass.filter((c) => c.op === 'setBindGroup');
  assert.equal(bgCalls.length, 8);
  assert.equal(executor.getStats().redundantBindGroupSkips, 2);
});

test('W166 bind-group reuse cache deduplicates fallback and custom bind groups, evicts LRU/destroyed resources, and respects clearBindGroupCache', () => {
  const { device, pass, calls } = createMockWebGpu();
  const router = new ShaderFallbackRouter();
  const executor = new DrawExecutor({
    device,
    shaderFallbackRouter: router,
    maxBindGroupCacheEntries: 3,
  });

  // 1. Fallback textured_quad_24b with GPUTexture passed directly: caches BOTH textureView and GPUBindGroup.
  let createViewCalls = 0;
  const vb = device.createBuffer({ size: 24 * 6, usage: BUFFER_USAGE.VERTEX });
  const tex = device.createTexture({
    size: { width: 64, height: 64, depthOrArrayLayers: 1 },
    format: 'rgba8unorm',
    usage: 0x04,
  });
  const origCreateView = tex.createView.bind(tex);
  tex.createView = (desc) => {
    createViewCalls += 1;
    return origCreateView(desc);
  };
  const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  executor.beginPass();
  executor.setPipeline({ vertexStride: 24, isFallback: true });
  executor.setVertexBuffer(0, vb, { offset: 0, stride: 24 });
  executor.setTexture(0, tex);
  executor.setSampler(0, sampler);

  for (let i = 0; i < 10; i += 1) {
    const r = executor.draw(pass, 6, 1);
    assert.equal(r.executed, true);
    assert.equal(r.profile, 'textured_quad_24b');
    assert.equal(r.bindGroupCacheHit, i > 0);
  }

  assert.equal(createViewCalls, 1);
  assert.equal(calls.createBindGroup.length, 1);
  assert.equal(calls.pass.filter((c) => c.op === 'setBindGroup').length, 1);
  assert.equal(calls.pass.filter((c) => c.op === 'setVertexBuffer').length, 1);
  assert.equal(executor.getStats().bindGroupCacheMisses, 1);
  assert.equal(executor.getStats().bindGroupCacheHits, 9);
  assert.equal(executor.getStats().redundantBindGroupSkips, 9);
  assert.equal(executor.getStats().redundantVertexBufferSkips, 9);

  // 2. clearBindGroupCache() forces recreation and re-binding on next draw.
  executor.clearBindGroupCache();
  assert.equal(executor.getStats().bindGroupCacheSize, 0);
  const afterClear = executor.draw(pass, 6, 1);
  assert.equal(afterClear.executed, true);
  assert.equal(afterClear.bindGroupCacheHit, false);
  assert.equal(calls.createBindGroup.length, 2);
  assert.equal(calls.pass.filter((c) => c.op === 'setBindGroup').length, 2);

  // 3. Public getOrCreateBindGroup deduplicates out-of-order entries, evicts LRU, and invalidates destroyed resources.
  const layout = { kind: 'GPUBindGroupLayout', handle: 9001 };
  const bufA = device.createBuffer({ size: 256, usage: BUFFER_USAGE.UNIFORM });
  const bufB = device.createBuffer({ size: 256, usage: BUFFER_USAGE.UNIFORM });
  const bufC = device.createBuffer({ size: 256, usage: BUFFER_USAGE.UNIFORM });
  const bufD = device.createBuffer({ size: 256, usage: BUFFER_USAGE.UNIFORM });

  executor.clearBindGroupCache();
  const gA1 = executor.getOrCreateBindGroup(layout, [
    { binding: 1, resource: sampler },
    { binding: 0, resource: { buffer: bufA, offset: 0, size: 256 } },
  ]);
  const gA2 = executor.getOrCreateBindGroup(layout, [
    { binding: 0, resource: { buffer: bufA, offset: 0, size: 256 } },
    { binding: 1, resource: sampler },
  ]);
  assert.equal(gA1, gA2); // exact cached GPUBindGroup reused regardless of entry order!

  // Different buffer offset produces distinct bind group.
  const gAOff = executor.getOrCreateBindGroup(layout, [
    { binding: 0, resource: { buffer: bufA, offset: 256, size: 256 } },
    { binding: 1, resource: sampler },
  ]);
  assert.notEqual(gAOff, gA1);

  // Fill to capacity (3) and trigger LRU eviction with 4th distinct entry.
  const gB = executor.getOrCreateBindGroup(layout, [
    { binding: 0, resource: { buffer: bufB, offset: 0, size: 256 } },
  ]);
  assert.ok(gB);
  // Touch gA1 so gAOff becomes oldest LRU.
  assert.equal(
    executor.getOrCreateBindGroup(layout, [
      { binding: 0, resource: { buffer: bufA, offset: 0, size: 256 } },
      { binding: 1, resource: sampler },
    ]),
    gA1
  );
  const evictionsBefore = executor.getStats().bindGroupCacheEvictions;
  const gC = executor.getOrCreateBindGroup(layout, [
    { binding: 0, resource: { buffer: bufC, offset: 0, size: 256 } },
  ]);
  assert.ok(gC);
  assert.equal(executor.getStats().bindGroupCacheSize, 3);
  assert.equal(executor.getStats().bindGroupCacheEvictions, evictionsBefore + 1);

  // Destroyed bind group in cache is automatically evicted & recreated on lookup.
  gC.destroyed = true;
  const gC2 = executor.getOrCreateBindGroup(layout, [
    { binding: 0, resource: { buffer: bufC, offset: 0, size: 256 } },
  ]);
  assert.ok(gC2);
  assert.notEqual(gC2, gC);
  assert.equal(executor.getStats().bindGroupCacheEvictions, evictionsBefore + 2);

  // Destroyed backing buffer is rejected cleanly without returning stale cache entry.
  bufD.destroy();
  assert.equal(
    executor.getOrCreateBindGroup(layout, [
      { binding: 0, resource: { buffer: bufD, offset: 0, size: 256 } },
    ]),
    null
  );

  // setDevice() clears bind-group cache cleanly.
  const envOther = createMockWebGpu();
  executor.setDevice(envOther.device);
  assert.equal(executor.getStats().bindGroupCacheSize, 0);
});

test('W166 per-reason skipped-draw accounting tracks every skip cause and matches skippedDraws invariant', () => {
  const router = new ShaderFallbackRouter();

  // 1. enableShaderFallback: false with no pipeline -> 'no-pipeline'
  const envNoPipe = createMockWebGpu();
  const execNoPipe = new DrawExecutor({
    device: envNoPipe.device,
    enableShaderFallback: false,
  });
  const rNoPipe = execNoPipe.draw(envNoPipe.pass, 6, 1);
  assert.equal(rNoPipe.skipped, true);
  assert.equal(rNoPipe.reason, 'no-pipeline');
  assert.equal(execNoPipe.getStats().skippedNoPipeline, 1);
  assert.equal(execNoPipe.getStats().skippedByReason['no-pipeline'], 1);

  // 2. Comprehensive multi-reason skips on a single executor.
  const env = createMockWebGpu();
  const exec = new DrawExecutor({ device: env.device, shaderFallbackRouter: router });
  const pipe = {
    kind: 'GPURenderPipeline',
    handle: 5001,
    getBindGroupLayout: () => ({ kind: 'GPUBindGroupLayout' }),
  };
  const smallIb = env.device.createBuffer({ size: 12, usage: BUFFER_USAGE.INDEX });
  const smallInd = env.device.createBuffer({ size: 16, usage: BUFFER_USAGE.INDIRECT });

  exec.beginPass();
  exec.setPipeline(pipe);
  exec.setIndexBuffer(smallIb, 'uint16', 0);

  // Invalid pass -> skippedNoPass
  assert.equal(exec.draw(null, 6, 1).reason, 'invalid-render-pass');
  // Zero count -> skippedInvalidRange + zeroCountDrawsSkipped
  assert.equal(exec.draw(env.pass, 0, 1).reason, 'zero-draw-count');
  assert.equal(exec.draw(env.pass, 6, 0).reason, 'zero-draw-count');
  // OOB index draw -> skippedInvalidIndexBuffer + skippedInvalidRange
  assert.equal(exec.drawIndexed(env.pass, 100, 1, 0, 0, 0).reason, 'invalid-index-buffer');
  // Misaligned indirect draw -> skippedInvalidIndirectBuffer + skippedInvalidRange
  assert.equal(exec.drawIndirect(env.pass, smallInd, 2).reason, 'invalid-indirect-buffer');
  // OOB indexed indirect draw (needs 20B, buffer is 16B) -> skippedInvalidIndirectBuffer + skippedInvalidRange
  assert.equal(exec.drawIndexedIndirect(env.pass, smallInd, 0).reason, 'invalid-indirect-buffer');

  // Throwing pass -> skippedPassError
  const failingPass = { ...env.pass, draw() { throw new Error('boom'); } };
  assert.equal(exec.draw(failingPass, 6, 1).reason, 'pass-error');

  // Destroyed executor -> skippedDestroyed
  exec.destroy();
  assert.equal(exec.draw(env.pass, 6, 1).reason, 'executor-destroyed');

  const stats = exec.getStats();
  assert.equal(stats.skippedDraws, 8);
  assert.equal(stats.skippedNoPass, 1);
  assert.equal(stats.zeroCountDrawsSkipped, 2);
  assert.equal(stats.skippedInvalidIndexBuffer, 1);
  assert.equal(stats.skippedInvalidIndirectBuffer, 2);
  assert.equal(stats.skippedInvalidRange, 5); // 2 zero-count + 1 OOB index + 2 invalid indirect range
  assert.equal(stats.skippedPassError, 1);
  assert.equal(stats.skippedDestroyed, 1);

  const sumByReason = Object.values(stats.skippedByReason).reduce((a, b) => a + b, 0);
  assert.equal(sumByReason, stats.skippedDraws);

  // Defensive clone of skippedByReason
  stats.skippedByReason['zero-draw-count'] = 9999;
  assert.equal(exec.getStats().skippedByReason['zero-draw-count'], 2);

  // Pipeline creation failure & bind failure counters
  const failPipeEnv = createMockWebGpu({ failCreateRenderPipeline: true });
  const execFailPipe = new DrawExecutor({ device: failPipeEnv.device, shaderFallbackRouter: router });
  assert.equal(execFailPipe.draw(failPipeEnv.pass, 3, 1).reason, 'pipeline-creation-failed');
  assert.equal(execFailPipe.getStats().skippedNoPipeline, 1);

  const failBgEnv = createMockWebGpu({ failCreateBindGroup: true });
  const execFailBg = new DrawExecutor({ device: failBgEnv.device, shaderFallbackRouter: router });
  assert.equal(
    execFailBg.executeDraw(failBgEnv.pass, { vertexCount: 3, hasBoundCbv: true }).reason,
    'fallback-cbv-bind-failed'
  );
  assert.equal(execFailBg.getStats().skippedBindFailed, 1);

  // resetStats() clears every skip counter and skippedByReason map.
  exec.resetStats();
  const cleared = exec.getStats();
  assert.equal(cleared.skippedDraws, 0);
  assert.equal(cleared.skippedNoPass, 0);
  assert.equal(cleared.skippedNoPipeline, 0);
  assert.equal(cleared.skippedInvalidRange, 0);
  assert.equal(cleared.skippedInvalidIndexBuffer, 0);
  assert.equal(cleared.skippedInvalidIndirectBuffer, 0);
  assert.equal(cleared.skippedBindFailed, 0);
  assert.equal(cleared.skippedPassError, 0);
  assert.equal(cleared.skippedDestroyed, 0);
  assert.deepEqual(cleared.skippedByReason, {});
});

test('W166 high-frequency menu, font-glyph, and intro-video burst (5,000+ draws) minimizes WebGPU bindings and tracks UI/glyph/video telemetry', () => {
  const { device, pass, calls } = createMockWebGpu();
  const router = new ShaderFallbackRouter();
  const executor = new DrawExecutor({ device, shaderFallbackRouter: router });

  const uiPipeline = {
    kind: 'GPURenderPipeline',
    handle: 6001,
    label: 'menu_ui_quad_alpha',
    desc: {
      fragment: {
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
    },
    getBindGroupLayout: () => ({ kind: 'GPUBindGroupLayout', handle: 6101 }),
  };

  const glyphPipeline = {
    kind: 'GPURenderPipeline',
    handle: 6002,
    label: 'scaleform_font_glyph',
    alphaBlended: true,
    getBindGroupLayout: () => ({ kind: 'GPUBindGroupLayout', handle: 6102 }),
  };

  const videoPipeline = {
    kind: 'GPURenderPipeline',
    handle: 6003,
    label: 'bink_intro_video_nv12',
    getBindGroupLayout: () => ({ kind: 'GPUBindGroupLayout', handle: 6103 }),
  };

  const vb = device.createBuffer({ size: 65536, usage: BUFFER_USAGE.VERTEX });
  const ib = device.createBuffer({ size: 65536, usage: BUFFER_USAGE.INDEX });
  const uiBg = { kind: 'GPUBindGroup', handle: 6201 };
  const glyphBg = { kind: 'GPUBindGroup', handle: 6202 };
  const videoBg = { kind: 'GPUBindGroup', handle: 6203 };

  executor.beginPass();
  executor.setVertexBuffer(0, vb, 0);
  executor.setIndexBuffer(ib, 'uint16', 0);

  // 1. 2,000 alpha-blended UI quad draws sharing one pipeline/BG/VB/IB.
  executor.setPipeline(uiPipeline);
  executor.setBindGroup(0, uiBg);
  for (let i = 0; i < 2000; i += 1) {
    const r = executor.drawIndexed(pass, 6, 1, 0, 0, 0);
    assert.equal(r.executed, true);
  }

  // 2. 3,000 alpha-blended font glyph draws; dynamic offset changes every 100 glyphs (30 unique offsets).
  executor.setPipeline(glyphPipeline);
  const dynOffset = new Uint32Array([0]);
  executor.setBindGroup(0, glyphBg, dynOffset);
  for (let i = 0; i < 3000; i += 1) {
    dynOffset[0] = Math.floor(i / 100) * 256;
    const r = executor.drawIndexed(pass, 6, 1, 0, 0, 0);
    assert.equal(r.executed, true);
  }

  // 3. 10 intro-video fullscreen draws.
  executor.setPipeline(videoPipeline);
  executor.setBindGroup(0, videoBg);
  for (let i = 0; i < 10; i += 1) {
    const r = executor.draw(pass, 3, 1);
    assert.equal(r.executed, true);
  }

  const stats = executor.getStats();
  assert.equal(stats.totalDraws, 5010);
  assert.equal(stats.normalDraws, 5010);
  assert.equal(stats.uiQuadDraws, 2000);
  assert.equal(stats.fontGlyphDraws, 3000);
  assert.equal(stats.videoDraws, 10);
  assert.equal(stats.alphaBlendedDraws, 5000);

  // Out of 5,010 draws:
  // - setPipeline: called ONLY 3 times (5,007 elided!)
  // - setVertexBuffer: called ONLY 1 time (5,009 elided!)
  // - setIndexBuffer: called ONLY 1 time across 5,000 indexed draws (4,999 elided!)
  // - setBindGroup: called ONLY 32 times (1 ui + 30 distinct glyph dynamic offsets + 1 video; 4,978 elided!)
  assert.equal(calls.pass.filter((c) => c.op === 'setPipeline').length, 3);
  assert.equal(calls.pass.filter((c) => c.op === 'setVertexBuffer').length, 1);
  assert.equal(calls.pass.filter((c) => c.op === 'setIndexBuffer').length, 1);
  assert.equal(calls.pass.filter((c) => c.op === 'setBindGroup').length, 32);
  assert.equal(stats.redundantPipelineSkips, 5007);
  assert.equal(stats.redundantVertexBufferSkips, 5009);
  assert.equal(stats.redundantIndexBufferSkips, 4999);
  assert.equal(stats.redundantBindGroupSkips, 4978);
});
