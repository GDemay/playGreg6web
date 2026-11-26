import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

import {
  RT_BVH_PRELUDE_VERSION,
  RT_BVH_PRELUDE_WGSL,
  needsRtBvhPrelude,
  applyRtBvhPrelude,
} from '../web/workers/rt_bvh_prelude.js';

import {
  SHADER_STAGE as WGPK_SHADER_STAGE,
  WgslPackCache,
  buildWgslPackBuffer,
  fnv1a64,
} from '../web/workers/wgsl_pack_loader.js';

const require = createRequire(import.meta.url);
const { WgpuCommandProcessor, OPCODES, SHADER_STAGE } = require('../web/workers/wgpu_worker.js');

const REPO_ROOT = path.resolve(fileURLToPath(new URL('..', import.meta.url)));

function extractCppPreludeWgsl(cppSource) {
  const fnIdx = cppSource.indexOf('const char* prelude_wgsl()');
  assert.notEqual(fnIdx, -1, 'Expected prelude_wgsl() in runtime/gpu/rt_bvh.cpp');
  const startMarker = 'R"WGSL(';
  const endMarker = ')WGSL"';
  const startIdx = cppSource.indexOf(startMarker, fnIdx);
  assert.notEqual(startIdx, -1, 'Expected R"WGSL( in prelude_wgsl()');
  const endIdx = cppSource.indexOf(endMarker, startIdx + startMarker.length);
  assert.notEqual(endIdx, -1, 'Expected )WGSL" in prelude_wgsl()');
  return cppSource.slice(startIdx + startMarker.length, endIdx).replace(/\r\n/g, '\n');
}

function buildCreateShaderModulePacket(id, stage, wgsl) {
  const codeBytes = new TextEncoder().encode(wgsl);
  const payloadBytes = 16 + codeBytes.byteLength;
  const buf = new ArrayBuffer(8 + payloadBytes);
  const view = new DataView(buf);
  view.setUint16(0, OPCODES.CREATE_SHADER_MODULE, true);
  view.setUint16(2, 0, true);
  view.setUint32(4, payloadBytes, true);
  view.setUint32(8, id, true);
  view.setUint32(12, stage, true);
  view.setUint32(16, codeBytes.byteLength, true);
  view.setUint32(20, 0, true);
  new Uint8Array(buf, 24).set(codeBytes);
  return new Uint8Array(buf);
}

function createMockDevice() {
  const shaderModules = [];
  let nextHandle = 1;
  const device = {
    createShaderModule(desc) {
      const mod = { kind: 'GPUShaderModule', handle: nextHandle++, label: desc.label, code: desc.code };
      shaderModules.push(mod);
      return mod;
    },
    createCommandEncoder() {
      return { finish: () => ({ kind: 'GPUCommandBuffer' }) };
    },
    queue: {
      submit() {},
    },
  };
  const ctx = {
    getCurrentTexture: () => ({ createView: () => ({ kind: 'swapchainView' }) }),
  };
  return { device, ctx, shaderModules };
}

test('RT_BVH_PRELUDE_VERSION and RT_BVH_PRELUDE_WGSL match C++ runtime/gpu/rt_bvh.{h,cpp} verbatim', () => {
  const headerPath = path.join(REPO_ROOT, 'runtime', 'gpu', 'rt_bvh.h');
  const cppPath = path.join(REPO_ROOT, 'runtime', 'gpu', 'rt_bvh.cpp');

  const headerSrc = readFileSync(headerPath, 'utf8');
  const versionMatch = /constexpr\s+uint32_t\s+kPreludeVersion\s*=\s*(\d+)\s*;/.exec(headerSrc);
  assert.ok(versionMatch, 'Expected kPreludeVersion in runtime/gpu/rt_bvh.h');
  assert.equal(RT_BVH_PRELUDE_VERSION, Number(versionMatch[1]));
  assert.equal(RT_BVH_PRELUDE_VERSION, 1);

  const cppSrc = readFileSync(cppPath, 'utf8');
  const cppPrelude = extractCppPreludeWgsl(cppSrc);
  assert.equal(RT_BVH_PRELUDE_WGSL, cppPrelude);
  assert.ok(RT_BVH_PRELUDE_WGSL.includes('fn pg6_load32(addr: vec2<u32>) -> u32'));
  assert.ok(RT_BVH_PRELUDE_WGSL.includes('fn pg6_bvh_intersect('));
  assert.ok(RT_BVH_PRELUDE_WGSL.includes('@group(3) @binding(16)'));
  assert.ok(RT_BVH_PRELUDE_WGSL.includes('@group(3) @binding(21)'));
});

test('needsRtBvhPrelude detects unresolved pg6_bvh_intersect and pg6_load32 calls accurately', () => {
  assert.equal(needsRtBvhPrelude(''), false);
  assert.equal(needsRtBvhPrelude(null), false);
  assert.equal(needsRtBvhPrelude(undefined), false);
  assert.equal(needsRtBvhPrelude(42), false);

  const normalWgsl = '@compute @workgroup_size(64) fn main() { let x = 1u; }';
  assert.equal(needsRtBvhPrelude(normalWgsl), false);

  // Non-matching identifiers sharing a substring must not trigger prelude injection.
  const unrelatedWgsl = '@compute @workgroup_size(1) fn main() { let a = pg6_load32_stub(vec2<u32>(0u)); let b = my_pg6_bvh_intersect(); }';
  assert.equal(needsRtBvhPrelude(unrelatedWgsl), false);

  const callsLoad = '@compute @workgroup_size(64) fn main() { let w = pg6_load32(vec2<u32>(0u, 0u)); }';
  assert.equal(needsRtBvhPrelude(callsLoad), true);

  const callsLoadSpaced = '@compute @workgroup_size(64) fn main() { let w = pg6_load32 (vec2<u32>(0u, 0u)); }';
  assert.equal(needsRtBvhPrelude(callsLoadSpaced), true);

  const callsIntersect = '@compute @workgroup_size(64) fn main() { let r = pg6_bvh_intersect(vec4<u32>(0u), vec2<u32>(0u), 1.0, vec3<f32>(0.0), vec3<f32>(1.0), vec3<f32>(1.0)); }';
  assert.equal(needsRtBvhPrelude(callsIntersect), true);

  const definesLoad = 'fn pg6_load32(addr: vec2<u32>) -> u32 { return 0u; }\n@compute @workgroup_size(1) fn main() { _ = pg6_load32(vec2<u32>(0u)); }';
  assert.equal(needsRtBvhPrelude(definesLoad), false);

  const definesIntersect = 'fn pg6_bvh_intersect(d: vec4<u32>) -> vec4<u32> { return d; }\n@compute @workgroup_size(1) fn main() { _ = pg6_bvh_intersect(vec4<u32>(0u)); }';
  assert.equal(needsRtBvhPrelude(definesIntersect), false);
});

test('applyRtBvhPrelude leaves normal shaders untouched, injects for RT calls, and is idempotent', () => {
  const normal = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }';
  assert.equal(applyRtBvhPrelude(normal), normal);

  const rtLoad = '@compute @workgroup_size(64) fn main() { let w = pg6_load32(vec2<u32>(0u, 0u)); }';
  const withLoadPrelude = applyRtBvhPrelude(rtLoad);
  assert.ok(withLoadPrelude.startsWith(RT_BVH_PRELUDE_WGSL));
  assert.ok(withLoadPrelude.endsWith(rtLoad));
  assert.equal(applyRtBvhPrelude(withLoadPrelude), withLoadPrelude);
  assert.equal(needsRtBvhPrelude(withLoadPrelude), false);

  const rtIntersect =
    '@compute @workgroup_size(64) fn main() {\n  let hit = pg6_bvh_intersect(vec4<u32>(0u), vec2<u32>(0u), 10.0, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0));\n}';
  const withIntersectPrelude = applyRtBvhPrelude(rtIntersect);
  assert.ok(withIntersectPrelude.startsWith(RT_BVH_PRELUDE_WGSL));
  assert.ok(withIntersectPrelude.includes(rtIntersect));
  assert.equal(applyRtBvhPrelude(withIntersectPrelude), withIntersectPrelude);
});

test('applyRtBvhPrelude places prelude AFTER leading enable, requires, and diagnostic directives', () => {
  const wgsl = [
    '// Generated shader header',
    '/* outer /* nested */ comment */',
    'enable f16; // half precision',
    'enable subgroups;',
    'requires readonly_and_readwrite_storage_textures;',
    'diagnostic(off, derivative_uniformity);',
    '',
    '@group(0) @binding(0) var<storage, read_write> out_buf: array<u32>;',
    '@compute @workgroup_size(64)',
    'fn main() {',
    '  out_buf[0] = pg6_load32(vec2<u32>(0u, 0u));',
    '}',
  ].join('\n');

  const out = applyRtBvhPrelude(wgsl);
  const idxEnableF16 = out.indexOf('enable f16;');
  const idxEnableSubgroups = out.indexOf('enable subgroups;');
  const idxRequires = out.indexOf('requires readonly_and_readwrite_storage_textures;');
  const idxDiagnostic = out.indexOf('diagnostic(off, derivative_uniformity);');
  const idxPrelude = out.indexOf(RT_BVH_PRELUDE_WGSL);
  const idxOutBuf = out.indexOf('@group(0) @binding(0) var<storage, read_write> out_buf');

  assert.ok(idxEnableF16 >= 0);
  assert.ok(idxEnableSubgroups > idxEnableF16);
  assert.ok(idxRequires > idxEnableSubgroups);
  assert.ok(idxDiagnostic > idxRequires);
  assert.ok(idxPrelude > idxDiagnostic, 'Prelude must appear after all top-of-file directives');
  assert.ok(idxOutBuf > idxPrelude, 'Prelude must appear before module declarations');
  assert.ok(out.includes('enable f16; // half precision\n'), 'Inline comment on directive line must stay intact');
  assert.equal(applyRtBvhPrelude(out), out, 'Must remain idempotent with directives');
});

test('applyRtBvhPrelude handles single-line directive + declaration and does not confuse comments/identifiers', () => {
  const inlineDir = 'enable f16; @compute @workgroup_size(1) fn main() { _ = pg6_load32(vec2<u32>(0u)); }';
  const outInline = applyRtBvhPrelude(inlineDir);
  assert.ok(outInline.startsWith('enable f16;\n' + RT_BVH_PRELUDE_WGSL));
  assert.ok(outInline.includes('@compute @workgroup_size(1) fn main()'));

  const commentedDir = '// enable f16;\n@compute @workgroup_size(1) fn main() { _ = pg6_load32(vec2<u32>(0u)); }';
  const outCommented = applyRtBvhPrelude(commentedDir);
  assert.ok(outCommented.startsWith(RT_BVH_PRELUDE_WGSL));
  assert.ok(outCommented.includes('// enable f16;'));

  const identStartingWithEnable = 'const enable_rt: u32 = 1u;\n@compute @workgroup_size(1) fn main() { _ = pg6_load32(vec2<u32>(enable_rt, 0u)); }';
  const outIdent = applyRtBvhPrelude(identStartingWithEnable);
  assert.ok(outIdent.startsWith(RT_BVH_PRELUDE_WGSL));
  assert.ok(outIdent.includes('const enable_rt: u32 = 1u;'));
});

test('wgpu_worker WgpuCommandProcessor prepends RT BVH prelude for RT shaders and skips normal shaders', () => {
  const { device, ctx, shaderModules } = createMockDevice();
  const processor = new WgpuCommandProcessor(device, ctx, {
    lazyShaderCompile: false,
    enableShaderFallback: false,
  });

  const rtWgsl = [
    'enable f16;',
    '@compute @workgroup_size(64)',
    'fn main() {',
    '  let v = pg6_load32(vec2<u32>(0u, 0u));',
    '  let hit = pg6_bvh_intersect(vec4<u32>(v), vec2<u32>(0u), 100.0, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0));',
    '}',
  ].join('\n');

  const normalWgsl = '@compute @workgroup_size(64) fn main() { let x = 42u; }';

  const pkt1 = buildCreateShaderModulePacket(101, SHADER_STAGE.COMPUTE, rtWgsl);
  const pkt2 = buildCreateShaderModulePacket(102, SHADER_STAGE.COMPUTE, normalWgsl);

  processor.executeCommandBuffer(pkt1);
  processor.executeCommandBuffer(pkt2);

  assert.equal(shaderModules.length, 2);
  const compiledRt = shaderModules[0].code;
  assert.ok(compiledRt.startsWith('enable f16;\n'));
  assert.ok(compiledRt.includes(RT_BVH_PRELUDE_WGSL));
  assert.ok(compiledRt.includes('fn pg6_load32(addr: vec2<u32>) -> u32'));
  assert.ok(compiledRt.includes('fn pg6_bvh_intersect('));
  assert.equal(compiledRt, applyRtBvhPrelude(rtWgsl));

  const compiledNormal = shaderModules[1].code;
  assert.equal(compiledNormal, normalWgsl);
  assert.ok(!compiledNormal.includes('pg6_arena0'));
});

test('wgpu_worker lazy shader compile and ranged pack resolution also apply RT BVH prelude', async () => {
  const { device, ctx, shaderModules } = createMockDevice();

  const rtPackWgsl =
    '@compute @workgroup_size(32) fn main() { let x = pg6_bvh_intersect(vec4<u32>(0u), vec2<u32>(0u), 1.0, vec3<f32>(0.0), vec3<f32>(1.0), vec3<f32>(1.0)); }';
  const hashBigInt = fnv1a64(rtPackWgsl);
  const hashHex = `0x${hashBigInt.toString(16).padStart(16, '0')}`;

  const fakeRangedPack = {
    has(key) {
      return key === hashHex || key === hashBigInt;
    },
    async getWgslAsync(key) {
      if (key === hashHex || key === hashBigInt) {
        return rtPackWgsl;
      }
      return null;
    },
  };

  const processor = new WgpuCommandProcessor(device, ctx, {
    lazyShaderCompile: true,
    enableShaderFallback: false,
    wgslPackCache: fakeRangedPack,
  });

  // 1) Lazy inline RT shader
  const lazyRt = '@compute @workgroup_size(1) fn main() { _ = pg6_load32(vec2<u32>(4u, 0u)); }';
  processor.executeCommandBuffer(buildCreateShaderModulePacket(201, SHADER_STAGE.COMPUTE, lazyRt));
  assert.equal(shaderModules.length, 0, 'Lazy mode defers compilation until ensureShaderModule');
  const ensured = processor.ensureShaderModule(201);
  assert.ok(ensured && ensured.module);
  assert.equal(shaderModules.length, 1);
  assert.equal(shaderModules[0].code, applyRtBvhPrelude(lazyRt));

  // 2) Ranged pack RT shader resolved asynchronously by hash
  processor.executeCommandBuffer(buildCreateShaderModulePacket(202, SHADER_STAGE.COMPUTE, hashHex));
  await processor.flushPendingRangedShaders();
  assert.equal(shaderModules.length, 2);
  assert.equal(shaderModules[1].code, applyRtBvhPrelude(rtPackWgsl));
});

test('wgsl_pack_loader WgslPackCache applies RT BVH prelude on sync and ranged async shader compilation', async () => {
  const rtWgsl =
    'enable subgroups;\n@compute @workgroup_size(64) fn main() { _ = pg6_load32(vec2<u32>(0u, 0u)); }';
  const normalWgsl = '@compute @workgroup_size(64) fn main() {}';

  const rtHash = fnv1a64(rtWgsl);
  const normalHash = fnv1a64(normalWgsl);

  const packBuf = buildWgslPackBuffer([
    { hash: rtHash, stage: WGPK_SHADER_STAGE.COMPUTE, wgsl: rtWgsl },
    { hash: normalHash, stage: WGPK_SHADER_STAGE.COMPUTE, wgsl: normalWgsl },
  ]);
  const packBytes = new Uint8Array(packBuf);

  // Sync full-buffer mode
  const syncCache = new WgslPackCache();
  syncCache.loadPackBuffer(packBuf, 'rt_test.wgsl.pack');
  assert.equal(syncCache.getWgsl(rtHash), rtWgsl, 'getWgsl returns raw pack source');

  const syncMock = createMockDevice();
  syncCache.getOrCreateShaderModule(syncMock.device, rtHash);
  syncCache.getOrCreateShaderModule(syncMock.device, normalHash);
  assert.equal(syncMock.shaderModules.length, 2);
  assert.equal(syncMock.shaderModules[0].code, applyRtBvhPrelude(rtWgsl));
  assert.ok(syncMock.shaderModules[0].code.startsWith('enable subgroups;\n' + RT_BVH_PRELUDE_WGSL));
  assert.equal(syncMock.shaderModules[1].code, normalWgsl);

  // Async ranged mode
  const mockFetch = async (_url, init = {}) => {
    const rangeHeader = init.headers && (init.headers.Range || init.headers.range);
    if (rangeHeader) {
      const m = /^bytes=(\d+)-(\d+)$/.exec(rangeHeader);
      const start = Number(m[1]);
      const end = Math.min(Number(m[2]), packBytes.byteLength - 1);
      const slice = packBytes.slice(start, end + 1);
      return {
        ok: true,
        status: 206,
        headers: {
          'content-range': `bytes ${start}-${end}/${packBytes.byteLength}`,
          'content-length': String(slice.byteLength),
        },
        async arrayBuffer() {
          return slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength);
        },
      };
    }
    return {
      ok: true,
      status: 200,
      headers: { 'content-length': String(packBytes.byteLength) },
      async arrayBuffer() {
        return packBuf.slice(0);
      },
    };
  };

  const rangedCache = new WgslPackCache({ fetch: mockFetch });
  await rangedCache.loadIndexFromUrl('http://localhost/rt_test.wgsl.pack');
  const asyncMock = createMockDevice();
  await rangedCache.getOrCreateShaderModuleAsync(asyncMock.device, rtHash);
  await rangedCache.getOrCreateShaderModuleAsync(asyncMock.device, normalHash);
  assert.equal(asyncMock.shaderModules.length, 2);
  assert.equal(asyncMock.shaderModules[0].code, applyRtBvhPrelude(rtWgsl));
  assert.equal(asyncMock.shaderModules[1].code, normalWgsl);
});
