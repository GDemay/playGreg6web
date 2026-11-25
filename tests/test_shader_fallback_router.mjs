import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const routerMod = require('../web/workers/shader_fallback_router.js');
const ROUTER_PATH = fileURLToPath(new URL('../web/workers/shader_fallback_router.js', import.meta.url));

const {
  SHADER_STAGE,
  STAGE_NAMES,
  MISSING_SHADER_REASONS,
  BUILTIN_FALLBACK_WGSL,
  BUILTIN_FALLBACK_PROFILES,
  DEFAULT_ENTRY_POINTS,
  DEFAULT_MAX_TRACKED_MISSING,
  DEFAULT_MAX_TRACKED_MISSING_HASHES,
  DEFAULT_MAX_CATEGORIZED_MISSING_HASHES,
  DEFAULT_MAX_RESOLVE_CACHE_ENTRIES,
  DEFAULT_MAX_COMPILE_FAILURE_ENTRIES,
  DEFAULT_TOP_MISSING_LIMIT,
  DEFAULT_MISSING_SUMMARY_TOP_LIMIT,
  normalizeShaderHash,
  formatShaderHash,
  normalizeStage,
  normalizeMissingReason,
  sanitizeCompileFailureReason,
  normalizeFallbackProfile,
  selectDeterministicFallbackProfile,
  getBuiltInFallbackProfiles,
  ShaderFallbackRouter,
} = routerMod;

const S9_FALLBACK_HASHES = Object.freeze({
  vertex: '0x1111111111111111',
  fragment: '0x2222222222222222',
  compute: '0x3333333333333333',
});

const S9_FALLBACK_WGSL = Object.freeze({
  vertex: [
    '@vertex',
    'fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {',
    '    return vec4<f32>(0.0, 0.0, 0.0, 1.0);',
    '}',
    '',
  ].join('\n'),
  fragment: [
    '@fragment',
    'fn fs_main() -> @location(0) vec4<f32> {',
    '    return vec4<f32>(0.2, 0.85, 0.45, 1.0);',
    '}',
    '',
  ].join('\n'),
  compute: [
    '@compute @workgroup_size(64)',
    'fn cs_main() {}',
    '',
  ].join('\n'),
});

const PACK_VS_WGSL =
  '@vertex fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(pos, 1.0); }';
const PACK_FS_WGSL =
  '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.25, 0.5, 1.0); }';
const PACK_CS_WGSL = '@compute @workgroup_size(1) fn cs_main() {}';

function createMapPackLoader(entries) {
  const byHash = new Map();
  for (const [hash, wgsl] of Object.entries(entries)) {
    const key = normalizeShaderHash(hash);
    byHash.set(key, wgsl);
    byHash.set(formatShaderHash(key), wgsl);
  }
  return {
    get(hash) {
      if (typeof hash === 'bigint' || typeof hash === 'number' || typeof hash === 'string') {
        try {
          const key = normalizeShaderHash(hash);
          return byHash.get(key) ?? byHash.get(formatShaderHash(key));
        } catch (_err) {
          return byHash.get(hash);
        }
      }
      return undefined;
    },
  };
}

function createGetSourcePackLoader(entries) {
  const map = createMapPackLoader(entries);
  return {
    getSource(hash) {
      return map.get(hash);
    },
  };
}

function createGetWgslPackLoader(entries) {
  const map = createMapPackLoader(entries);
  return {
    getWgsl(hash) {
      return map.get(hash);
    },
  };
}

function createLookupPackLoader(entries) {
  const map = createMapPackLoader(entries);
  return {
    lookup(hash) {
      const wgsl = map.get(hash);
      if (wgsl === undefined) return undefined;
      return { wgsl, entryPoint: null };
    },
  };
}

function createS9PackLoader() {
  return createGetSourcePackLoader({
    [S9_FALLBACK_HASHES.vertex]: S9_FALLBACK_WGSL.vertex,
    [S9_FALLBACK_HASHES.fragment]: S9_FALLBACK_WGSL.fragment,
    [S9_FALLBACK_HASHES.compute]: S9_FALLBACK_WGSL.compute,
  });
}

function assertValidWgsl(wgsl, stage) {
  assert.equal(typeof wgsl, 'string');
  assert.ok(wgsl.length > 0, 'WGSL must be non-empty');
  assert.match(wgsl, new RegExp(`@${stage}\\b`));
  assert.match(wgsl, /\bfn\s+[A-Za-z_]/);
}

test('UMD export attaches to globalThis.PlayGta6ShaderFallbackRouter and module.exports', () => {
  assert.equal(globalThis.PlayGta6ShaderFallbackRouter, routerMod);
  assert.equal(typeof ShaderFallbackRouter, 'function');
  assert.deepEqual(SHADER_STAGE, { VERTEX: 1, FRAGMENT: 2, COMPUTE: 3 });
  assert.deepEqual(STAGE_NAMES, { 1: 'vertex', 2: 'fragment', 3: 'compute' });
  assert.equal(DEFAULT_MAX_TRACKED_MISSING, 512);
  assert.deepEqual(DEFAULT_ENTRY_POINTS, {
    vertex: 'vs_main',
    fragment: 'fs_main',
    compute: 'cs_main',
  });
});

test('normalizeShaderHash accepts BigInt, 0x hex, bare hex, and safe integers', () => {
  const expected = 0xdeadbeefcafebaben;
  assert.equal(normalizeShaderHash(expected), expected);
  assert.equal(normalizeShaderHash('0xdeadbeefcafebabe'), expected);
  assert.equal(normalizeShaderHash('0XDEADBEEFCAFEBABE'), expected);
  assert.equal(normalizeShaderHash('deadbeefcafebabe'), expected);
  assert.equal(normalizeShaderHash(42), 42n);
  assert.equal(formatShaderHash(expected), '0xdeadbeefcafebabe');
  assert.equal(formatShaderHash(1), '0x0000000000000001');

  assert.throws(() => normalizeShaderHash(-1n), /u64 range/);
  assert.throws(() => normalizeShaderHash(1.5), /safe integer/);
  assert.throws(() => normalizeShaderHash('0xzz'), /Invalid hex/);
  assert.throws(() => normalizeShaderHash({}), /BigInt, hex string/);
});

test('normalizeStage accepts WGPK codes and stage name aliases', () => {
  assert.equal(normalizeStage(1), 'vertex');
  assert.equal(normalizeStage(2), 'fragment');
  assert.equal(normalizeStage(3), 'compute');
  assert.equal(normalizeStage('VERTEX'), 'vertex');
  assert.equal(normalizeStage('fs'), 'fragment');
  assert.equal(normalizeStage('ps'), 'fragment');
  assert.equal(normalizeStage('cs'), 'compute');
  assert.throws(() => normalizeStage(0), /Invalid shader stage code/);
  assert.throws(() => normalizeStage('geometry'), /Invalid shader stage name/);
});

test('BUILTIN_FALLBACK_WGSL provides minimal valid WGSL for every stage', () => {
  for (const stage of ['vertex', 'fragment', 'compute']) {
    assertValidWgsl(BUILTIN_FALLBACK_WGSL[stage], stage);
    assert.equal(
      Object.isFrozen(BUILTIN_FALLBACK_WGSL) || typeof BUILTIN_FALLBACK_WGSL[stage] === 'string',
      true
    );
  }
  assert.ok(BUILTIN_FALLBACK_WGSL.vertex.includes('vs_main'));
  assert.ok(BUILTIN_FALLBACK_WGSL.fragment.includes('fs_main'));
  assert.ok(BUILTIN_FALLBACK_WGSL.compute.includes('cs_main'));
});

test('resolve returns exact pack hit via getSource without recording missing telemetry', () => {
  const packHash = '0xaaaaaaaaaaaaaaaa';
  const router = new ShaderFallbackRouter({
    packLoader: createGetSourcePackLoader({ [packHash]: PACK_VS_WGSL }),
    fallbackHashes: S9_FALLBACK_HASHES,
  });

  const result = router.resolve(packHash, 'vertex');
  assert.equal(result.source, 'pack');
  assert.equal(result.wgsl, PACK_VS_WGSL);
  assert.equal(result.requestedHash, packHash);
  assert.equal(result.resolvedHash, packHash);
  assert.equal(result.stage, 'vertex');
  assert.equal(result.entryPoint, 'vs_main');

  assert.deepEqual(router.getMissingReport(), []);
  assert.equal(router.getStats().packHits, 1);
  assert.equal(router.getStats().uniqueMissing, 0);
});

test('resolve finds exact pack hit via getWgsl, lookup, and Map.get interfaces', () => {
  const hash = 0xbbbbbbbbbbbbbbbbn;

  const viaGetWgsl = new ShaderFallbackRouter({
    packLoader: createGetWgslPackLoader({ [formatShaderHash(hash)]: PACK_FS_WGSL }),
  });
  assert.equal(viaGetWgsl.resolve(hash, SHADER_STAGE.FRAGMENT).source, 'pack');
  assert.equal(viaGetWgsl.resolve(hash, 'fragment').wgsl, PACK_FS_WGSL);

  const viaLookup = new ShaderFallbackRouter({
    packLoader: createLookupPackLoader({ [formatShaderHash(hash)]: PACK_CS_WGSL }),
  });
  const lookupHit = viaLookup.resolve(formatShaderHash(hash), 'compute');
  assert.equal(lookupHit.source, 'pack');
  assert.equal(lookupHit.wgsl, PACK_CS_WGSL);

  const map = new Map([
    [hash, PACK_VS_WGSL],
    [formatShaderHash(hash), PACK_VS_WGSL],
  ]);
  const viaMap = new ShaderFallbackRouter({ packLoader: map });
  assert.equal(viaMap.resolve(hash, 'vertex').source, 'pack');
});

test('resolve uses stage-matched S9 fallback hash from pack when exact miss', () => {
  const missingHash = '0xcafebabedeadbeef';
  const router = new ShaderFallbackRouter({
    packLoader: createS9PackLoader(),
    fallbackHashes: S9_FALLBACK_HASHES,
  });

  const vs = router.resolve(missingHash, 'vertex');
  assert.equal(vs.source, 'fallback');
  assert.equal(vs.wgsl, S9_FALLBACK_WGSL.vertex);
  assert.equal(vs.requestedHash, missingHash);
  assert.equal(vs.resolvedHash, S9_FALLBACK_HASHES.vertex);
  assert.equal(vs.entryPoint, 'vs_main');

  const fs = router.resolve('0x0102030405060708', SHADER_STAGE.FRAGMENT);
  assert.equal(fs.source, 'fallback');
  assert.equal(fs.resolvedHash, S9_FALLBACK_HASHES.fragment);
  assert.equal(fs.wgsl, S9_FALLBACK_WGSL.fragment);

  const cs = router.resolve(0x9999999999999999n, 'compute');
  assert.equal(cs.source, 'fallback');
  assert.equal(cs.resolvedHash, S9_FALLBACK_HASHES.compute);

  const stats = router.getStats();
  assert.equal(stats.fallbackHits, 3);
  assert.equal(stats.packHits, 0);
  assert.equal(stats.uniqueMissing, 3);
});

test('resolve falls back to builtin WGSL when pack and S9 fallback are absent', () => {
  const router = new ShaderFallbackRouter();
  const vs = router.resolve('0xabcdef0123456789', 'vertex');
  assert.equal(vs.source, 'builtin');
  assert.equal(vs.resolvedHash, null);
  assert.equal(vs.wgsl, BUILTIN_FALLBACK_WGSL.vertex);
  assert.equal(vs.entryPoint, 'vs_main');
  assertValidWgsl(vs.wgsl, 'vertex');

  const fs = router.resolve(1n, 2);
  assert.equal(fs.source, 'builtin');
  assert.equal(fs.wgsl, BUILTIN_FALLBACK_WGSL.fragment);
  assert.equal(fs.entryPoint, 'fs_main');

  const cs = router.resolve(2, 'COMPUTE');
  assert.equal(cs.source, 'builtin');
  assert.equal(cs.wgsl, BUILTIN_FALLBACK_WGSL.compute);
  assert.equal(cs.entryPoint, 'cs_main');

  assert.equal(router.getStats().builtinHits, 3);
  assert.equal(router.getStats().packHits, 0);
  assert.equal(router.getStats().fallbackHits, 0);
});

test('resolve uses builtin when fallbackHashes point at hashes missing from pack', () => {
  const router = new ShaderFallbackRouter({
    packLoader: createGetSourcePackLoader({}),
    fallbackHashes: S9_FALLBACK_HASHES,
  });
  const result = router.resolve('0xfeedfacefeedface', 'fragment');
  assert.equal(result.source, 'builtin');
  assert.equal(result.wgsl, BUILTIN_FALLBACK_WGSL.fragment);
  assert.equal(router.getStats().builtinHits, 1);
  assert.equal(router.getStats().fallbackHits, 0);
});

test('missing telemetry deduplicates by hash+stage and sorts by hit count descending', () => {
  let tick = 1000;
  const router = new ShaderFallbackRouter({
    packLoader: createS9PackLoader(),
    fallbackHashes: S9_FALLBACK_HASHES,
    now: () => {
      tick += 10;
      return tick;
    },
  });

  const a = '0xaaaaaaaaaaaaaaaa';
  const b = '0xbbbbbbbbbbbbbbbb';
  const c = '0xcccccccccccccccc';

  router.resolve(a, 'vertex');
  router.resolve(a, 'vertex');
  router.resolve(a, 'vertex');
  router.resolve(b, 'fragment');
  router.resolve(b, 'fragment');
  router.resolve(c, 'compute');
  // Same hash, different stage — separate telemetry row.
  router.resolve(a, 'fragment');

  const report = router.getMissingReport();
  assert.equal(report.length, 4);
  assert.equal(report[0].hash, a);
  assert.equal(report[0].stage, 'vertex');
  assert.equal(report[0].hitCount, 3);
  assert.equal(report[1].hash, b);
  assert.equal(report[1].hitCount, 2);
  assert.ok(report[0].firstSeenMs <= report[0].lastSeenMs);
  assert.ok(report.every((row) => typeof row.hash === 'string' && row.hash.startsWith('0x')));
  assert.equal(JSON.parse(JSON.stringify(report)).length, 4);

  const limited = router.getMissingReport({ limit: 2 });
  assert.equal(limited.length, 2);
  assert.equal(limited[0].hitCount, 3);
  assert.equal(limited[1].hitCount, 2);
});

test('maxTrackedMissing caps unique rows and increments overflowMissingCount', () => {
  const router = new ShaderFallbackRouter({
    maxTrackedMissing: 2,
    now: () => 42,
  });

  router.resolve('0x0000000000000001', 'vertex');
  router.resolve('0x0000000000000002', 'fragment');
  router.resolve('0x0000000000000003', 'compute');
  router.resolve('0x0000000000000004', 'vertex');
  // Duplicate of an overflowed hash still overflows (not tracked).
  router.resolve('0x0000000000000003', 'compute');

  const stats = router.getStats();
  assert.equal(stats.uniqueMissing, 2);
  assert.equal(stats.overflowMissingCount, 3);
  assert.equal(router.getMissingReport().length, 2);
  assert.equal(router.getMissingReport()[0].hitCount, 1);
});

test('getStats reports hitRatePct from pack hits and reset clears telemetry', () => {
  const packHash = '0xdddddddddddddddd';
  const router = new ShaderFallbackRouter({
    packLoader: createGetSourcePackLoader({
      [packHash]: PACK_VS_WGSL,
      [S9_FALLBACK_HASHES.vertex]: S9_FALLBACK_WGSL.vertex,
    }),
    fallbackHashes: { vertex: S9_FALLBACK_HASHES.vertex },
  });

  router.resolve(packHash, 'vertex');
  router.resolve(packHash, 'vertex');
  router.resolve('0xeeeeeeeeeeeeeeee', 'vertex');
  router.resolve('0xffffffffffffffff', 'fragment');

  let stats = router.getStats();
  assert.equal(stats.totalLookups, 4);
  assert.equal(stats.packHits, 2);
  assert.equal(stats.fallbackHits, 1);
  assert.equal(stats.builtinHits, 1);
  assert.equal(stats.uniqueMissing, 2);
  assert.equal(stats.overflowMissingCount, 0);
  assert.equal(stats.overflowMissingHashes, 0);
  assert.equal(stats.hitRatePct, 50);
  assert.equal(stats.hitRatio, 0.5);

  router.reset();
  stats = router.getStats();
  assert.deepEqual(stats, {
    totalLookups: 0,
    packHits: 0,
    fallbackHits: 0,
    builtinHits: 0,
    uniqueMissing: 0,
    overflowMissingCount: 0,
    overflowMissingHashes: 0,
    hitRatePct: 0,
    hitRatio: 0,
    compileFailureEntries: 0,
    compileFailureCacheHits: 0,
    compileFailureRecorded: 0,
  });
  assert.deepEqual(router.getMissingReport(), []);

  const after = router.resolve('0x1111222233334444', 'compute');
  assert.equal(after.source, 'builtin');
  assert.equal(router.getStats().totalLookups, 1);
  assert.equal(router.getStats().uniqueMissing, 1);
});

test('BigInt and hex forms of the same hash share pack hits and missing telemetry', () => {
  const value = 0x1234567890abcdefn;
  const hex = formatShaderHash(value);
  const router = new ShaderFallbackRouter({
    packLoader: createMapPackLoader({ [hex]: PACK_FS_WGSL }),
  });

  assert.equal(router.resolve(value, 'fragment').source, 'pack');
  assert.equal(router.resolve(hex, 'fragment').source, 'pack');
  assert.equal(router.resolve(hex.toUpperCase().replace('0X', '0x'), 'fragment').source, 'pack');

  const missRouter = new ShaderFallbackRouter({ now: () => 7 });
  missRouter.resolve(value, 'vertex');
  missRouter.resolve(hex, 'vertex');
  missRouter.resolve('1234567890abcdef', 'vertex');
  const report = missRouter.getMissingReport();
  assert.equal(report.length, 1);
  assert.equal(report[0].hash, hex);
  assert.equal(report[0].hitCount, 3);
});

test('constructor validates options and setPackLoader / setFallbackHashes update resolution', () => {
  assert.throws(() => new ShaderFallbackRouter('nope'), /options must be an object/);
  assert.throws(() => new ShaderFallbackRouter({ maxTrackedMissing: -1 }), /maxTrackedMissing/);
  assert.throws(() => new ShaderFallbackRouter({ maxTrackedMissing: 1.5 }), /maxTrackedMissing/);

  const router = new ShaderFallbackRouter();
  let hit = router.resolve('0x0101010101010101', 'vertex');
  assert.equal(hit.source, 'builtin');

  router.setPackLoader(
    createGetSourcePackLoader({
      [S9_FALLBACK_HASHES.vertex]: S9_FALLBACK_WGSL.vertex,
    })
  );
  router.setFallbackHashes({ vertex: S9_FALLBACK_HASHES.vertex });
  hit = router.resolve('0x0202020202020202', 'vertex');
  assert.equal(hit.source, 'fallback');
  assert.equal(hit.wgsl, S9_FALLBACK_WGSL.vertex);

  router.setFallbackHashes(null);
  hit = router.resolve('0x0303030303030303', 'vertex');
  assert.equal(hit.source, 'builtin');
});

// ---------------------------------------------------------------------------
// G34 Frame-1 fallback WGSL profiles (fullscreen_triangle_cbv & textured_quad_24b)
// ---------------------------------------------------------------------------

function assertBalancedWgslDelimiters(wgsl, label) {
  let braces = 0;
  let parens = 0;
  for (let i = 0; i < wgsl.length; i += 1) {
    const ch = wgsl[i];
    if (ch === '{') braces += 1;
    else if (ch === '}') braces -= 1;
    else if (ch === '(') parens += 1;
    else if (ch === ')') parens -= 1;
    assert.ok(braces >= 0, `${label}: unmatched '}' at index ${i}`);
    assert.ok(parens >= 0, `${label}: unmatched ')' at index ${i}`);
  }
  assert.equal(braces, 0, `${label}: unbalanced curly braces`);
  assert.equal(parens, 0, `${label}: unbalanced parentheses`);
}

test('G34 getBuiltInFallbackProfiles returns sanitized, mutation-isolated map of all fallback profiles', () => {
  assert.equal(typeof getBuiltInFallbackProfiles, 'function');
  assert.equal(typeof ShaderFallbackRouter.getBuiltInFallbackProfiles, 'function');
  assert.ok(BUILTIN_FALLBACK_PROFILES && Object.isFrozen(BUILTIN_FALLBACK_PROFILES));

  const profiles = getBuiltInFallbackProfiles();
  assert.deepEqual(Object.keys(profiles).sort(), [
    'default',
    'fullscreen_triangle_cbv',
    'splash_rect_24b',
    'textured_quad_24b',
  ]);

  // Default profile matches legacy BUILTIN_FALLBACK_WGSL verbatim.
  assert.equal(profiles.default.vertex, BUILTIN_FALLBACK_WGSL.vertex);
  assert.equal(profiles.default.fragment, BUILTIN_FALLBACK_WGSL.fragment);
  assert.equal(profiles.default.compute, BUILTIN_FALLBACK_WGSL.compute);

  // Every profile provides valid WGSL for every stage and is JSON-serializable.
  for (const [name, stages] of Object.entries(profiles)) {
    for (const stage of ['vertex', 'fragment', 'compute']) {
      assertValidWgsl(stages[stage], stage);
      assertBalancedWgslDelimiters(stages[stage], `${name}.${stage}`);
    }
  }
  assert.deepEqual(JSON.parse(JSON.stringify(profiles)), profiles);

  // Instance method returns the same catalog and mutating returned copy does not corrupt router state.
  const router = new ShaderFallbackRouter();
  const instanceProfiles = router.getBuiltInFallbackProfiles();
  assert.deepEqual(instanceProfiles, profiles);
  instanceProfiles.fullscreen_triangle_cbv.vertex = 'CORRUPTED';
  instanceProfiles.injected = { vertex: 'bad' };

  const fresh = getBuiltInFallbackProfiles();
  assert.equal(fresh.injected, undefined);
  assert.equal(fresh.fullscreen_triangle_cbv.vertex, profiles.fullscreen_triangle_cbv.vertex);
  assert.equal(
    router.resolve('0x123456789abcdef0', 'vertex', { profile: 'fullscreen_cbv' }).wgsl,
    profiles.fullscreen_triangle_cbv.vertex
  );
});

test('G34 fullscreen_triangle_cbv WGSL pair matches Frame-1 DrawInstanced(3,1) + group(0) binding(0) CBV', () => {
  const { fullscreen_triangle_cbv: pair } = getBuiltInFallbackProfiles();

  // Vertex shader: @builtin(vertex_index) vid: u32 + FrameCbv uniform at group(0) binding(0).
  assert.match(pair.vertex, /struct\s+FrameCbv\s*\{\s*data:\s*array<vec4<f32>,\s*16>,?\s*\};?/);
  assert.match(pair.vertex, /@group\(0\)\s+@binding\(0\)\s+var<uniform>\s+frame_cbv:\s*FrameCbv;/);
  assert.match(pair.vertex, /@vertex\b/);
  assert.match(pair.vertex, /fn\s+vs_main\(@builtin\(vertex_index\)\s+vid:\s*u32\)/);
  assert.match(pair.vertex, /@builtin\(position\)/);
  assert.match(pair.vertex, /@location\(0\)\s+uv:\s*vec2<f32>/);
  // Statically references frame_cbv so WebGPU auto-layout preserves group(0) binding(0).
  assert.match(pair.vertex, /frame_cbv\.data\[0\]/);
  // Wraps vertex_index modulo 3 so draws with >3 vertices never OOB-index.
  assert.match(pair.vertex, /vid\s*%\s*3u/);

  // Fragment shader: matching FrameCbv uniform + @location(0) uv + NaN-safe CBV modulation + alpha 1.0.
  assert.match(pair.fragment, /struct\s+FrameCbv\s*\{\s*data:\s*array<vec4<f32>,\s*16>,?\s*\};?/);
  assert.match(pair.fragment, /@group\(0\)\s+@binding\(0\)\s+var<uniform>\s+frame_cbv:\s*FrameCbv;/);
  assert.match(pair.fragment, /@fragment\b/);
  assert.match(pair.fragment, /fn\s+fs_main\(/);
  assert.match(pair.fragment, /@location\(0\)\s+uv:\s*vec2<f32>/);
  assert.match(pair.fragment, /->\s*@location\(0\)\s+vec4<f32>/);
  assert.match(pair.fragment, /bitcast<u32>\(frame_cbv\.data\[0\]/);
  assert.match(pair.fragment, /return\s+vec4<f32>\([^;]+,\s*1\.0\);/);
});

test('G34 textured_quad_24b WGSL pair matches 24-byte vertex stride + group(0) texture/sampler', () => {
  const { textured_quad_24b: pair } = getBuiltInFallbackProfiles();

  // Vertex shader: @location(0) pos: vec4<f32> (16B) + @location(1) uv: vec2<f32> (8B) = 24B stride.
  assert.match(pair.vertex, /@vertex\b/);
  assert.match(pair.vertex, /fn\s+vs_main\(/);
  assert.match(pair.vertex, /@location\(0\)\s+pos:\s*vec4<f32>/);
  assert.match(pair.vertex, /@location\(1\)\s+uv:\s*vec2<f32>/);
  assert.match(pair.vertex, /@builtin\(position\)/);
  assert.match(pair.vertex, /@location\(0\)\s+uv:\s*vec2<f32>/);

  // Fragment shader: group(0) binding(0) texture_2d<f32> + binding(1) sampler.
  assert.match(pair.fragment, /@group\(0\)\s+@binding\(0\)\s+var\s+[A-Za-z_][A-Za-z0-9_]*:\s*texture_2d<f32>;/);
  assert.match(pair.fragment, /@group\(0\)\s+@binding\(1\)\s+var\s+[A-Za-z_][A-Za-z0-9_]*:\s*sampler;/);
  assert.match(pair.fragment, /@fragment\b/);
  assert.match(pair.fragment, /fn\s+fs_main\(@location\(0\)\s+uv:\s*vec2<f32>\)\s*->\s*@location\(0\)\s+vec4<f32>/);
  assert.match(pair.fragment, /textureSample\(/);
});

test('G34 normalizeFallbackProfile and resolve(hash, stage, { profile }) select profiles and preserve defaults', () => {
  assert.equal(normalizeFallbackProfile(undefined), 'default');
  assert.equal(normalizeFallbackProfile(null), 'default');
  assert.equal(normalizeFallbackProfile(''), 'default');
  assert.equal(normalizeFallbackProfile('default'), 'default');
  assert.equal(normalizeFallbackProfile('unknown-profile'), 'default');
  assert.equal(normalizeFallbackProfile('fullscreen_cbv'), 'fullscreen_triangle_cbv');
  assert.equal(normalizeFallbackProfile('fullscreen_triangle_cbv'), 'fullscreen_triangle_cbv');
  assert.equal(normalizeFallbackProfile('  FULLSCREEN-CBV  '), 'fullscreen_triangle_cbv');
  assert.equal(normalizeFallbackProfile('textured_24b'), 'textured_quad_24b');
  assert.equal(normalizeFallbackProfile('textured_quad_24b'), 'textured_quad_24b');
  assert.equal(normalizeFallbackProfile('TEXTURED-QUAD-24B'), 'textured_quad_24b');

  const profiles = getBuiltInFallbackProfiles();
  const router = new ShaderFallbackRouter({ now: () => 1234 });

  // Default 2-arg call remains 100% backwards compatible.
  const defVs = router.resolve('0x0000000000000001', 'vertex');
  assert.equal(defVs.source, 'builtin');
  assert.equal(defVs.wgsl, BUILTIN_FALLBACK_WGSL.vertex);
  assert.equal(defVs.entryPoint, 'vs_main');
  assert.equal(defVs.profile, 'default');

  // fullscreen_cbv alias & canonical name.
  const cbvVs = router.resolve('0x0000000000000002', 'vertex', { profile: 'fullscreen_cbv' });
  assert.equal(cbvVs.source, 'builtin');
  assert.equal(cbvVs.wgsl, profiles.fullscreen_triangle_cbv.vertex);
  assert.equal(cbvVs.entryPoint, 'vs_main');
  assert.equal(cbvVs.profile, 'fullscreen_triangle_cbv');

  const cbvFs = router.resolve('0x0000000000000003', SHADER_STAGE.FRAGMENT, {
    profile: 'fullscreen_triangle_cbv',
  });
  assert.equal(cbvFs.source, 'builtin');
  assert.equal(cbvFs.wgsl, profiles.fullscreen_triangle_cbv.fragment);
  assert.equal(cbvFs.entryPoint, 'fs_main');
  assert.equal(cbvFs.profile, 'fullscreen_triangle_cbv');

  // textured_24b alias & canonical name.
  const texVs = router.resolve('0x0000000000000004', 'vs', { profile: 'textured_24b' });
  assert.equal(texVs.source, 'builtin');
  assert.equal(texVs.wgsl, profiles.textured_quad_24b.vertex);
  assert.equal(texVs.entryPoint, 'vs_main');
  assert.equal(texVs.profile, 'textured_quad_24b');

  const texFs = router.resolve('0x0000000000000005', 'ps', { profile: 'textured_quad_24b' });
  assert.equal(texFs.source, 'builtin');
  assert.equal(texFs.wgsl, profiles.textured_quad_24b.fragment);
  assert.equal(texFs.entryPoint, 'fs_main');
  assert.equal(texFs.profile, 'textured_quad_24b');

  // Compute stage with render profile still resolves cleanly to valid compute WGSL.
  const cs = router.resolve('0x0000000000000006', 'compute', { profile: 'fullscreen_cbv' });
  assert.equal(cs.source, 'builtin');
  assert.equal(cs.wgsl, BUILTIN_FALLBACK_WGSL.compute);
  assert.equal(cs.entryPoint, 'cs_main');
  assert.equal(cs.profile, 'fullscreen_triangle_cbv');

  // Unknown profile gracefully falls back to default built-in WGSL.
  const unknown = router.resolve('0x0000000000000007', 'fragment', { profile: 'does_not_exist' });
  assert.equal(unknown.source, 'builtin');
  assert.equal(unknown.wgsl, BUILTIN_FALLBACK_WGSL.fragment);
  assert.equal(unknown.profile, 'default');

  // Telemetry accurately records every profiled built-in fallback miss.
  const stats = router.getStats();
  assert.equal(stats.totalLookups, 7);
  assert.equal(stats.builtinHits, 7);
  assert.equal(stats.uniqueMissing, 7);
  assert.equal(router.getMissingReport().length, 7);
});

test('G34 pack hits and S9 fallback hashes retain precedence over options.profile and defaultProfile', () => {
  const packHash = '0xaaaaaaaaaaaaaaaa';
  const router = new ShaderFallbackRouter({
    packLoader: createGetSourcePackLoader({
      [packHash]: PACK_VS_WGSL,
      [S9_FALLBACK_HASHES.fragment]: S9_FALLBACK_WGSL.fragment,
    }),
    fallbackHashes: { fragment: S9_FALLBACK_HASHES.fragment },
    defaultProfile: 'fullscreen_cbv',
    now: () => 500,
  });

  // 1) Exact pack hit wins even when profile is specified, and records zero missing telemetry.
  const packRes = router.resolve(packHash, 'vertex', { profile: 'textured_24b' });
  assert.equal(packRes.source, 'pack');
  assert.equal(packRes.wgsl, PACK_VS_WGSL);
  assert.equal(router.getMissingReport().length, 0);

  // 2) Stage-matched S9 fallback hash wins over built-in profile while recording missing telemetry.
  const s9Res = router.resolve('0xbbbbbbbbbbbbbbbb', 'fragment', { profile: 'fullscreen_cbv' });
  assert.equal(s9Res.source, 'fallback');
  assert.equal(s9Res.wgsl, S9_FALLBACK_WGSL.fragment);
  assert.equal(s9Res.resolvedHash, S9_FALLBACK_HASHES.fragment);

  // 3) Unmatched stage falls through to constructor defaultProfile ('fullscreen_triangle_cbv').
  const defaultProfileRes = router.resolve('0xcccccccccccccccc', 'vertex');
  assert.equal(defaultProfileRes.source, 'builtin');
  assert.equal(defaultProfileRes.profile, 'fullscreen_triangle_cbv');
  assert.equal(
    defaultProfileRes.wgsl,
    getBuiltInFallbackProfiles().fullscreen_triangle_cbv.vertex
  );

  // 4) Per-call profile overrides constructor defaultProfile.
  const overrideRes = router.resolve('0xdddddddddddddddd', 'vertex', { profile: 'textured_24b' });
  assert.equal(overrideRes.source, 'builtin');
  assert.equal(overrideRes.profile, 'textured_quad_24b');
  assert.equal(overrideRes.wgsl, getBuiltInFallbackProfiles().textured_quad_24b.vertex);

  const stats = router.getStats();
  assert.equal(stats.totalLookups, 4);
  assert.equal(stats.packHits, 1);
  assert.equal(stats.fallbackHits, 1);
  assert.equal(stats.builtinHits, 2);
  assert.equal(stats.uniqueMissing, 3);
});

test('G34 shader_fallback_router.js preserves classic Worker importScripts IIFE isolation with zero global leaks', () => {
  const source = readFileSync(ROUTER_PATH, 'utf8');
  const sandbox = { console, performance };
  const vmCtx = vm.createContext(sandbox);
  vmCtx.self = vmCtx;

  // Execute twice as in repeated classic-worker importScripts() calls.
  assert.doesNotThrow(() => {
    vm.runInContext(source, vmCtx, { filename: ROUTER_PATH });
    vm.runInContext(source, vmCtx, { filename: ROUTER_PATH });
  });

  // Verify zero top-level const/let/class declarations leaked into WorkerGlobalScope.
  assert.doesNotThrow(() => {
    vm.runInContext(
      `
      const BUILTIN_FALLBACK_WGSL = {};
      const FULLSCREEN_TRIANGLE_CBV_WGSL = {};
      const TEXTURED_QUAD_24B_WGSL = {};
      const SPLASH_RECT_24B_WGSL = {};
      const BUILTIN_FALLBACK_PROFILES = {};
      const FALLBACK_PROFILE_ALIASES = {};
      const exportedApi = {};
      class ShaderFallbackRouter {}
      `,
      vmCtx
    );
  });

  assert.ok(vmCtx.PlayGta6ShaderFallbackRouter);
  assert.equal(typeof vmCtx.PlayGta6ShaderFallbackRouter.getBuiltInFallbackProfiles, 'function');
  const profiles = vmCtx.PlayGta6ShaderFallbackRouter.getBuiltInFallbackProfiles();
  assert.ok(
    profiles.fullscreen_triangle_cbv && profiles.textured_quad_24b && profiles.splash_rect_24b
  );
});

// ---------------------------------------------------------------------------
// G47 Frame-1 24B splash RECTLIST fallback profile (splash_rect_24b)
// ---------------------------------------------------------------------------

test('G47 splash_rect_24b WGSL pair matches exact Frame-1 24B POSITION/COLOR/TEXCOORD layout + cb0/tex0/samp0', () => {
  const profiles = getBuiltInFallbackProfiles();
  const pair = profiles.splash_rect_24b;
  assert.ok(pair, 'splash_rect_24b profile must be exposed in getBuiltInFallbackProfiles()');

  for (const stage of ['vertex', 'fragment', 'compute']) {
    assertValidWgsl(pair[stage], stage);
    assertBalancedWgslDelimiters(pair[stage], `splash_rect_24b.${stage}`);
  }

  // Vertex shader: exact Frame-1 24B attributes + cb0 uniform binding.
  assert.match(pair.vertex, /struct\s+FrameCbv\s*\{\s*data:\s*array<vec4<f32>,\s*16>,?\s*\};?/);
  assert.match(pair.vertex, /@group\(0\)\s+@binding\(0\)\s+var<uniform>\s+cb0:\s*FrameCbv;/);
  assert.match(pair.vertex, /@vertex\b/);
  assert.match(pair.vertex, /fn\s+vs_main\(/);
  assert.match(pair.vertex, /@location\(0\)\s+pos:\s*vec3<f32>/);
  assert.match(pair.vertex, /@location\(1\)\s+color:\s*vec4<f32>/);
  assert.match(pair.vertex, /@location\(2\)\s+uv:\s*vec2<f32>/);
  assert.match(pair.vertex, /@builtin\(position\)/);
  assert.match(pair.vertex, /cb0\.data\[0\]/);

  // Fragment shader: cb0 @ 0 + tex0 @ 1 + samp0 @ 2 + color * textureSample output.
  assert.match(pair.fragment, /struct\s+FrameCbv\s*\{\s*data:\s*array<vec4<f32>,\s*16>,?\s*\};?/);
  assert.match(pair.fragment, /@group\(0\)\s+@binding\(0\)\s+var<uniform>\s+cb0:\s*FrameCbv;/);
  assert.match(pair.fragment, /@group\(0\)\s+@binding\(1\)\s+var\s+tex0:\s*texture_2d<f32>;/);
  assert.match(pair.fragment, /@group\(0\)\s+@binding\(2\)\s+var\s+samp0:\s*sampler;/);
  assert.match(pair.fragment, /@fragment\b/);
  assert.match(pair.fragment, /fn\s+fs_main\(/);
  assert.match(pair.fragment, /->\s*@location\(0\)\s+vec4<f32>/);
  assert.match(pair.fragment, /textureSample\(\s*tex0\s*,\s*samp0\s*,/);
  assert.match(pair.fragment, /input\.color/);
});

test('G47 normalizeFallbackProfile and resolve() support splash_rect_24b and aliases', () => {
  assert.equal(normalizeFallbackProfile('splash_rect_24b'), 'splash_rect_24b');
  assert.equal(normalizeFallbackProfile('splash-rect-24b'), 'splash_rect_24b');
  assert.equal(normalizeFallbackProfile('rectlist_24b'), 'splash_rect_24b');
  assert.equal(normalizeFallbackProfile('RECTLIST-24B'), 'splash_rect_24b');
  assert.equal(normalizeFallbackProfile('frame1_splash_24b'), 'splash_rect_24b');
  assert.equal(normalizeFallbackProfile('  FRAME1-SPLASH-24B  '), 'splash_rect_24b');

  const profiles = getBuiltInFallbackProfiles();
  const router = new ShaderFallbackRouter({ now: () => 4700 });

  const vsCanonical = router.resolve('0x1010101010101010', 'vertex', {
    profile: 'splash_rect_24b',
  });
  assert.equal(vsCanonical.source, 'builtin');
  assert.equal(vsCanonical.profile, 'splash_rect_24b');
  assert.equal(vsCanonical.entryPoint, 'vs_main');
  assert.equal(vsCanonical.wgsl, profiles.splash_rect_24b.vertex);

  const fsRectAlias = router.resolve('0x2020202020202020', 'fragment', {
    profile: 'rectlist_24b',
  });
  assert.equal(fsRectAlias.source, 'builtin');
  assert.equal(fsRectAlias.profile, 'splash_rect_24b');
  assert.equal(fsRectAlias.entryPoint, 'fs_main');
  assert.equal(fsRectAlias.wgsl, profiles.splash_rect_24b.fragment);

  const vsFrame1Alias = router.resolve('0x3030303030303030', SHADER_STAGE.VERTEX, {
    profile: 'frame1_splash_24b',
  });
  assert.equal(vsFrame1Alias.source, 'builtin');
  assert.equal(vsFrame1Alias.profile, 'splash_rect_24b');
  assert.equal(vsFrame1Alias.wgsl, profiles.splash_rect_24b.vertex);

  const csSplash = router.resolve('0x4040404040404040', 'compute', {
    profile: 'splash_rect_24b',
  });
  assert.equal(csSplash.source, 'builtin');
  assert.equal(csSplash.profile, 'splash_rect_24b');
  assert.equal(csSplash.entryPoint, 'cs_main');
  assert.equal(csSplash.wgsl, BUILTIN_FALLBACK_WGSL.compute);
});

// ---------------------------------------------------------------------------
// W133 Bounded missing-hash tracking, top-N ranking, and hitRatio telemetry
// ---------------------------------------------------------------------------

test('W133 getTopMissingShaders ranks descending by count with deterministic hashHex tie-breaking and sanitized output', () => {
  assert.equal(DEFAULT_TOP_MISSING_LIMIT, 16);

  const router = new ShaderFallbackRouter({ now: () => 133 });
  assert.deepEqual(router.getTopMissingShaders(), []);
  assert.deepEqual(router.getTopMissingShaders(0), []);

  // Insert in non-sorted order with deliberate count ties across hashes & stages.
  const hHigh = '0x0000000000000099';
  const hTieC = '0x00000000000000cc';
  const hTieA = '0x00000000000000aa';
  const hTieB = '0x00000000000000bb';

  for (let i = 0; i < 2; i += 1) router.resolve(hTieC, 'fragment');
  for (let i = 0; i < 5; i += 1) router.resolve(hHigh, 'vertex');
  for (let i = 0; i < 2; i += 1) router.resolve(hTieA, 'vertex');
  for (let i = 0; i < 2; i += 1) router.resolve(hTieB, 'compute');
  // Same hash as hTieA, different stage, same count -> stage ascending breaks secondary tie.
  for (let i = 0; i < 2; i += 1) router.resolve(hTieA, 'fragment');

  // Add 15 more single-hit entries so total unique missing = 20 (> default limit 16).
  for (let i = 1; i <= 15; i += 1) {
    router.resolve(BigInt(0x1000 + i), 'fragment');
  }

  // Default limit returns 16 sanitized records.
  const topDefault = router.getTopMissingShaders();
  assert.equal(topDefault.length, 16);

  // Verify descending count order and deterministic hashHex (then stage) ascending tie-breaking.
  assert.deepEqual(topDefault.slice(0, 5), [
    { hash: hHigh, hashHex: hHigh, stage: 'vertex', count: 5, firstSeenFrame: 0, lastSeenFrame: 0 },
    { hash: hTieA, hashHex: hTieA, stage: 'fragment', count: 2, firstSeenFrame: 0, lastSeenFrame: 0 },
    { hash: hTieA, hashHex: hTieA, stage: 'vertex', count: 2, firstSeenFrame: 0, lastSeenFrame: 0 },
    { hash: hTieB, hashHex: hTieB, stage: 'compute', count: 2, firstSeenFrame: 0, lastSeenFrame: 0 },
    { hash: hTieC, hashHex: hTieC, stage: 'fragment', count: 2, firstSeenFrame: 0, lastSeenFrame: 0 },
  ]);

  // Explicit limit subsets cleanly and limit=0 returns empty array.
  assert.deepEqual(router.getTopMissingShaders(3), topDefault.slice(0, 3));
  assert.deepEqual(router.getTopMissingShaders(0), []);
  assert.equal(router.getTopMissingShaders(null).length, 16);
  assert.equal(router.getTopMissingShaders(100).length, 20);

  // Every record is strictly sanitized: only { hash, hashHex, stage, count, firstSeenFrame, lastSeenFrame },
  // no paths, binary dumps, or WGSL text.
  for (const row of topDefault) {
    assert.deepEqual(Object.keys(row).sort(), [
      'count',
      'firstSeenFrame',
      'hash',
      'hashHex',
      'lastSeenFrame',
      'stage',
    ]);
    assert.equal(row.hash, row.hashHex);
    assert.match(row.hashHex, /^0x[0-9a-f]{16}$/);
    assert.ok(['vertex', 'fragment', 'compute'].includes(row.stage));
    assert.equal(Number.isInteger(row.count) && row.count > 0, true);
    assert.equal(Number.isInteger(row.firstSeenFrame) && row.firstSeenFrame >= 0, true);
    assert.equal(Number.isInteger(row.lastSeenFrame) && row.lastSeenFrame >= row.firstSeenFrame, true);
  }
  const serialized = JSON.stringify(topDefault);
  assert.equal(serialized.includes('@vertex'), false);
  assert.equal(serialized.includes('@fragment'), false);
  assert.equal(serialized.includes('fn '), false);

  // Mutating returned objects must not alter router state.
  topDefault[0].count = 999999;
  assert.equal(router.getTopMissingShaders(1)[0].count, 5);

  // Invalid limit throws RangeError.
  assert.throws(() => router.getTopMissingShaders(-1), RangeError);
  assert.throws(() => router.getTopMissingShaders(2.5), RangeError);
  assert.throws(() => router.getTopMissingShaders(Number.NaN), RangeError);
  assert.throws(() => router.getTopMissingShaders('4'), RangeError);
});

test('W133 maxTrackedMissingHashes defaults to 2048, caps map, tracks overflowMissingHashes, and keeps counting tracked hashes', () => {
  assert.equal(DEFAULT_MAX_TRACKED_MISSING_HASHES, 2048);

  // Default constructor cap is 2048.
  const defaultRouter = new ShaderFallbackRouter({ now: () => 1 });
  for (let i = 0; i < 2048; i += 1) {
    defaultRouter.resolve(BigInt(i + 1), 'vertex');
  }
  assert.equal(defaultRouter.getStats().uniqueMissing, 2048);
  assert.equal(defaultRouter.getStats().overflowMissingHashes, 0);

  // 2049th unseen hash overflows without growing the tracked map.
  defaultRouter.resolve(BigInt(2049), 'vertex');
  assert.equal(defaultRouter.getStats().uniqueMissing, 2048);
  assert.equal(defaultRouter.getStats().overflowMissingHashes, 1);
  assert.equal(defaultRouter.getStats().overflowMissingCount, 1);

  // Already-tracked hash still increments count without incrementing overflow.
  defaultRouter.resolve(BigInt(1), 'vertex');
  assert.equal(defaultRouter.getStats().uniqueMissing, 2048);
  assert.equal(defaultRouter.getStats().overflowMissingHashes, 1);
  assert.deepEqual(defaultRouter.getTopMissingShaders(1), [
    {
      hash: '0x0000000000000001',
      hashHex: '0x0000000000000001',
      stage: 'vertex',
      count: 2,
      firstSeenFrame: 0,
      lastSeenFrame: 0,
    },
  ]);

  // Custom small cap via maxTrackedMissingHashes.
  const capped = new ShaderFallbackRouter({
    maxTrackedMissingHashes: 3,
    now: () => 77,
  });
  capped.resolve('0x0000000000000010', 'vertex');
  capped.resolve('0x0000000000000020', 'fragment');
  capped.resolve('0x0000000000000030', 'compute');

  // Unseen hashes beyond cap increment overflowMissingHashes and are not added.
  capped.resolve('0x0000000000000040', 'vertex');
  capped.resolve('0x0000000000000050', 'fragment');
  capped.resolve('0x0000000000000040', 'vertex');

  // Already-tracked hashes keep incrementing even while full.
  capped.resolve('0x0000000000000020', 'fragment');
  capped.resolve('0x0000000000000020', 'fragment');
  capped.resolve('0x0000000000000030', 'compute');

  const stats = capped.getStats();
  assert.equal(stats.uniqueMissing, 3);
  assert.equal(stats.overflowMissingHashes, 3);
  assert.equal(stats.overflowMissingCount, 3);
  assert.deepEqual(capped.getTopMissingShaders(), [
    {
      hash: '0x0000000000000020',
      hashHex: '0x0000000000000020',
      stage: 'fragment',
      count: 3,
      firstSeenFrame: 0,
      lastSeenFrame: 0,
    },
    {
      hash: '0x0000000000000030',
      hashHex: '0x0000000000000030',
      stage: 'compute',
      count: 2,
      firstSeenFrame: 0,
      lastSeenFrame: 0,
    },
    {
      hash: '0x0000000000000010',
      hashHex: '0x0000000000000010',
      stage: 'vertex',
      count: 1,
      firstSeenFrame: 0,
      lastSeenFrame: 0,
    },
  ]);

  // Zero cap tracks nothing and counts every miss as overflow.
  const zeroCap = new ShaderFallbackRouter({ maxTrackedMissingHashes: 0 });
  zeroCap.resolve('0x0000000000000001', 'vertex');
  zeroCap.resolve('0x0000000000000001', 'vertex');
  assert.equal(zeroCap.getStats().uniqueMissing, 0);
  assert.equal(zeroCap.getStats().overflowMissingHashes, 2);
  assert.deepEqual(zeroCap.getTopMissingShaders(), []);

  // Option validation rejects invalid maxTrackedMissingHashes values.
  assert.throws(
    () => new ShaderFallbackRouter({ maxTrackedMissingHashes: -1 }),
    /maxTrackedMissingHashes/
  );
  assert.throws(
    () => new ShaderFallbackRouter({ maxTrackedMissingHashes: 3.14 }),
    /maxTrackedMissingHashes/
  );
  assert.throws(
    () => new ShaderFallbackRouter({ maxTrackedMissingHashes: '10' }),
    /maxTrackedMissingHashes/
  );

  // reset() clears tracked hashes and overflow counters.
  capped.reset();
  assert.equal(capped.getStats().uniqueMissing, 0);
  assert.equal(capped.getStats().overflowMissingHashes, 0);
  assert.deepEqual(capped.getTopMissingShaders(), []);
});

test('W133 getStats exposes accurate hitRatio in [0, 1] across empty, miss, mixed, and hit workloads', () => {
  const hitHash = '0xaaaaaaaaaaaaaaaa';
  const router = new ShaderFallbackRouter({
    packLoader: createGetSourcePackLoader({
      [hitHash]: PACK_VS_WGSL,
      [S9_FALLBACK_HASHES.vertex]: S9_FALLBACK_WGSL.vertex,
    }),
    fallbackHashes: { vertex: S9_FALLBACK_HASHES.vertex },
  });

  // Empty router: 0 / 0 => hitRatio === 0.
  assert.equal(router.getStats().hitRatio, 0);

  // All misses (fallback + builtin): 0 / 2 => hitRatio === 0.
  router.resolve('0x5555555555555555', 'vertex');
  router.resolve('0x6666666666666666', 'fragment');
  let stats = router.getStats();
  assert.equal(stats.totalLookups, 2);
  assert.equal(stats.packHits, 0);
  assert.equal(stats.hitRatio, 0);
  assert.ok(stats.hitRatio >= 0 && stats.hitRatio <= 1);

  // Mixed: 1 pack hit + 2 misses => 1/3.
  router.resolve(hitHash, 'vertex');
  stats = router.getStats();
  assert.equal(stats.totalLookups, 3);
  assert.equal(stats.packHits, 1);
  assert.ok(Math.abs(stats.hitRatio - 1 / 3) < 1e-12);
  assert.ok(stats.hitRatio >= 0 && stats.hitRatio <= 1);

  // Mixed: 2 pack hits + 2 misses => 0.5.
  router.resolve(hitHash, 'vertex');
  stats = router.getStats();
  assert.equal(stats.hitRatio, 0.5);
  assert.equal(stats.hitRatePct, 50);

  // Reset restores 0; all pack hits => hitRatio === 1.
  router.reset();
  assert.equal(router.getStats().hitRatio, 0);
  router.resolve(hitHash, 'vertex');
  router.resolve(hitHash, 'vertex');
  stats = router.getStats();
  assert.equal(stats.hitRatio, 1);
  assert.equal(stats.hitRatePct, 100);
});

// ---------------------------------------------------------------------------
// W179 Real-shader override guard & categorized missing-shader telemetry
// ---------------------------------------------------------------------------

const BINK2_YUV_PS_WGSL = [
  '@group(0) @binding(0) var y_tex: texture_2d<f32>;',
  '@group(0) @binding(1) var cbcr_tex: texture_2d<f32>;',
  '@group(0) @binding(2) var yuv_sampler: sampler;',
  '@fragment fn fs_bink2_yuv(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {',
  '    let y = textureSample(y_tex, yuv_sampler, uv).r;',
  '    let cbcr = textureSample(cbcr_tex, yuv_sampler, uv).rg - vec2<f32>(0.5, 0.5);',
  '    let r = y + 1.402 * cbcr.y;',
  '    let g = y - 0.344136 * cbcr.x - 0.714136 * cbcr.y;',
  '    let b = y + 1.772 * cbcr.x;',
  '    return vec4<f32>(clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);',
  '}',
].join('\n');

const MENU_UI_PS_WGSL = [
  '@group(0) @binding(0) var ui_atlas: texture_2d<f32>;',
  '@group(0) @binding(1) var ui_sampler: sampler;',
  '@fragment fn fs_menu_ui(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {',
  '    return textureSample(ui_atlas, ui_sampler, uv);',
  '}',
].join('\n');

test('W179 MISSING_SHADER_REASONS and normalizeMissingReason normalize canonical categories and aliases', () => {
  assert.deepEqual(MISSING_SHADER_REASONS, {
    NOT_IN_PACK: 'not_in_pack',
    REFLECTION_FAILED: 'reflection_failed',
    COMPILE_FAILED: 'compile_failed',
    OTHER: 'other',
  });
  assert.equal(Object.isFrozen(MISSING_SHADER_REASONS), true);
  assert.equal(DEFAULT_MAX_CATEGORIZED_MISSING_HASHES, 8192);
  assert.equal(DEFAULT_MISSING_SUMMARY_TOP_LIMIT, 32);

  assert.equal(normalizeMissingReason('not_in_pack'), 'not_in_pack');
  assert.equal(normalizeMissingReason('  NOT-IN-PACK  '), 'not_in_pack');
  assert.equal(normalizeMissingReason('pack_miss'), 'not_in_pack');
  assert.equal(normalizeMissingReason('missing_from_pack'), 'not_in_pack');

  assert.equal(normalizeMissingReason('reflection_failed'), 'reflection_failed');
  assert.equal(normalizeMissingReason('REFLECTION-FAILED'), 'reflection_failed');
  assert.equal(normalizeMissingReason('reflection error'), 'reflection_failed');
  assert.equal(normalizeMissingReason('no_reflection'), 'reflection_failed');

  assert.equal(normalizeMissingReason('compile_failed'), 'compile_failed');
  assert.equal(normalizeMissingReason('COMPILE-FAILED'), 'compile_failed');
  assert.equal(normalizeMissingReason('compilation_failed'), 'compile_failed');
  assert.equal(normalizeMissingReason('wgsl_compile_failed'), 'compile_failed');

  assert.equal(normalizeMissingReason('other'), 'other');
  assert.equal(normalizeMissingReason('unknown_reason'), 'other');
  assert.equal(normalizeMissingReason(''), 'other');
  assert.equal(normalizeMissingReason(null), 'other');
  assert.equal(normalizeMissingReason(undefined), 'other');
  assert.equal(normalizeMissingReason(123), 'other');
});

test('W179 real-pack shaders (menu & Bink2 YUV) are never overwritten by fallback profiles, S9 hashes, or forceFallback', () => {
  const binkHash = '0xb1a2000000000001';
  const menuHash = '0xae00000000000002';
  const missingHash = '0xdead000000000003';
  let clock = 1000;

  const router = new ShaderFallbackRouter({
    packLoader: createGetSourcePackLoader({
      [binkHash]: BINK2_YUV_PS_WGSL,
      [menuHash]: MENU_UI_PS_WGSL,
      [S9_FALLBACK_HASHES.fragment]: S9_FALLBACK_WGSL.fragment,
    }),
    fallbackHashes: S9_FALLBACK_HASHES,
    defaultProfile: 'splash_rect_24b',
    now: () => {
      clock += 5;
      return clock;
    },
  });

  // Real pack shaders are detected as real shaders; S9 fallback hash is not.
  assert.equal(router.hasRealShader(binkHash, 'fragment'), true);
  assert.equal(router.hasRealShader(menuHash, 'fragment'), true);
  assert.equal(router.hasRealShader(S9_FALLBACK_HASHES.fragment, 'fragment'), false);
  assert.equal(router.hasRealShader(missingHash, 'fragment'), false);

  assert.equal(router.canFallbackOverride(binkHash, 'fragment'), false);
  assert.equal(router.canFallbackOverride(menuHash, 'fragment'), false);
  assert.equal(router.canFallbackOverride(S9_FALLBACK_HASHES.fragment, 'fragment'), true);
  assert.equal(router.canFallbackOverride(missingHash, 'fragment'), true);

  // Even when a fallback profile or forceFallback is passed, real pack shaders always win.
  const binkRes = router.resolve(binkHash, 'fragment', {
    profile: 'splash_rect_24b',
    forceFallback: true,
  });
  assert.equal(binkRes.source, 'pack');
  assert.equal(binkRes.wgsl, BINK2_YUV_PS_WGSL);
  assert.equal(binkRes.entryPoint, 'fs_bink2_yuv');
  assert.equal(binkRes.fallbackBlocked, true);
  assert.equal(binkRes.guardReason, 'pack_shader_protected');

  const menuRes = router.resolve(menuHash, 'fragment', {
    profile: 'fullscreen_triangle_cbv',
  });
  assert.equal(menuRes.source, 'pack');
  assert.equal(menuRes.wgsl, MENU_UI_PS_WGSL);
  assert.equal(menuRes.entryPoint, 'fs_menu_ui');
  assert.equal(menuRes.fallbackBlocked, undefined);

  // Zero missing telemetry recorded for protected real pack shaders.
  assert.equal(router.getMissingReport().length, 0);
  const summary = router.getMissingShaderSummary();
  assert.equal(summary.totalMissingEvents, 0);
  assert.equal(summary.realShaderGuard.blockedOverrideCount, 1);

  const guardStats = router.getRealShaderGuardStats();
  assert.equal(guardStats.blockedOverrideCount, 1);
  assert.equal(guardStats.blockedShaders.length, 1);
  assert.equal(guardStats.blockedShaders[0].hashHex, binkHash);
  assert.equal(guardStats.blockedShaders[0].stage, 'fragment');
  assert.equal(guardStats.blockedShaders[0].count, 1);
  assert.equal(guardStats.blockedShaders[0].lastReason, 'pack_shader_protected');
  // Sanitized: no WGSL leaked in guard stats.
  assert.equal(JSON.stringify(guardStats).includes('fs_bink2_yuv'), false);
});

test('W179 registerRealShader and caller existingShader guard protect compiled real modules from silent fallback overwrite', () => {
  const binkHash = '0xb1a2000000000010';
  const menuHash = '0xae00000000000020';
  const inlineHash = '0x1a1a000000000030';
  const fakeBinkModule = { __kind: 'GPUShaderModule', label: 'bink2_yuv' };

  const router = new ShaderFallbackRouter({
    packLoader: createS9PackLoader(),
    fallbackHashes: S9_FALLBACK_HASHES,
    defaultProfile: 'splash_rect_24b',
    realShaders: [
      {
        hash: menuHash,
        stage: 'fragment',
        wgsl: MENU_UI_PS_WGSL,
        label: 'menu_ui_ps',
      },
    ],
    now: () => 500,
  });

  // Register a real Bink2 YUV shader module handle (without WGSL text in pack).
  const regInfo = router.registerRealShader(binkHash, 'fragment', {
    module: fakeBinkModule,
    entryPoint: 'fs_bink2_yuv',
    label: 'bink2_yuv_ps',
  });
  assert.deepEqual(regInfo, {
    hashHex: binkHash,
    stage: 'fragment',
    hasWgsl: false,
    hasModule: true,
  });

  assert.equal(router.hasRealShader(binkHash, 'fragment'), true);
  assert.equal(router.hasRealShader(binkHash, 'vertex'), false);
  assert.equal(router.hasRealShader(menuHash, 'fragment'), true);

  // Attempting to resolve Bink2 YUV fragment shader must NOT return S9 or builtin fallback.
  const guardedBink = router.resolve(binkHash, 'fragment', { profile: 'splash_rect_24b' });
  assert.equal(guardedBink.source, 'real');
  assert.equal(guardedBink.wgsl, null);
  assert.equal(guardedBink.module, fakeBinkModule);
  assert.equal(guardedBink.entryPoint, 'fs_bink2_yuv');
  assert.equal(guardedBink.label, 'bink2_yuv_ps');
  assert.equal(guardedBink.fallbackBlocked, true);
  assert.equal(guardedBink.guardReason, 'real_shader_registered');

  // Registered menu shader with WGSL returns preserved real WGSL instead of opaque fallback.
  const guardedMenu = router.resolve(menuHash, 'fragment', { profile: 'fullscreen_triangle_cbv' });
  assert.equal(guardedMenu.source, 'real');
  assert.equal(guardedMenu.wgsl, MENU_UI_PS_WGSL);
  assert.equal(guardedMenu.entryPoint, 'fs_menu_ui');
  assert.equal(guardedMenu.fallbackBlocked, true);
  assert.equal(guardedMenu.guardReason, 'real_shader_registered');

  // Caller-provided existingShader with source: 'pack' or isRealShader: true is also guarded.
  assert.equal(
    router.canFallbackOverride(inlineHash, 'fragment', { existingSource: 'pack' }),
    false
  );
  assert.equal(
    router.canFallbackOverride(inlineHash, 'fragment', { isRealShader: true }),
    false
  );
  assert.equal(
    router.canFallbackOverride(inlineHash, 'fragment', { source: 'builtin' }),
    true
  );

  const guardedInline = router.resolve(inlineHash, 'fragment', {
    profile: 'textured_quad_24b',
    existingShader: {
      source: 'pack',
      wgsl: BINK2_YUV_PS_WGSL,
      entryPoint: 'fs_bink2_yuv',
    },
  });
  assert.equal(guardedInline.source, 'pack');
  assert.equal(guardedInline.wgsl, BINK2_YUV_PS_WGSL);
  assert.equal(guardedInline.fallbackBlocked, true);
  assert.equal(guardedInline.guardReason, 'existing_real_shader');

  // None of the guarded lookups counted as fallbackHits, builtinHits, or missing shaders.
  const stats = router.getStats();
  assert.equal(stats.fallbackHits, 0);
  assert.equal(stats.builtinHits, 0);
  assert.equal(stats.uniqueMissing, 0);

  const guardStats = router.getRealShaderGuardStats();
  assert.equal(guardStats.registeredRealShaders, 2);
  assert.equal(guardStats.blockedOverrideCount, 3);
  assert.equal(guardStats.blockedShaders.length, 3);
  // Ensure JSON-safe and free of module/WGSL leaks.
  const guardJson = JSON.stringify(guardStats);
  assert.deepEqual(JSON.parse(guardJson), guardStats);
  assert.equal(guardJson.includes('GPUShaderModule'), false);
  assert.equal(guardJson.includes('textureSample'), false);

  // reset() clears blocked counters while keeping registered real shaders protected.
  router.reset();
  assert.equal(router.getRealShaderGuardStats().blockedOverrideCount, 0);
  assert.equal(router.getRealShaderGuardStats().registeredRealShaders, 2);
  assert.equal(router.hasRealShader(binkHash, 'fragment'), true);

  // Explicit unregisterRealShader / clearRealShaders re-enables fallback resolution.
  assert.equal(router.unregisterRealShader(binkHash, 'fragment'), true);
  assert.equal(router.unregisterRealShader(binkHash, 'fragment'), false);
  assert.equal(router.hasRealShader(binkHash, 'fragment'), false);
  const afterUnreg = router.resolve(binkHash, 'fragment');
  assert.equal(afterUnreg.source, 'fallback');

  router.clearRealShaders();
  assert.equal(router.hasRealShader(menuHash, 'fragment'), false);
});

test('W179 recordMissingShader categorizes reasons & stages and getMissingShaderSummary exports JSON-safe sorted triage data', () => {
  let tick = 100;
  const router = new ShaderFallbackRouter({
    now: () => {
      tick += 10;
      return tick;
    },
  });

  const hMenuPs = '0x000000000000aa01';
  const hBinkPs = '0x000000000000bb02';
  const hMenuVs = '0x000000000000cc03';
  const hComputeCs = '0x000000000000dd04';

  // resolve() miss automatically records 'not_in_pack' in categorized summary.
  router.resolve(hMenuPs, 'fragment');
  router.resolve(hMenuPs, 'ps');

  // Explicit recordMissingShader across all 4 categories and all 3 stages.
  const rec1 = router.recordMissingShader(hBinkPs, 'fragment', 'compile_failed');
  assert.equal(rec1.hashHex, hBinkPs);
  assert.equal(rec1.stage, 'fragment');
  assert.equal(rec1.reason, 'compile_failed');
  assert.equal(rec1.hitCount, 1);
  assert.equal(rec1.tracked, true);
  assert.equal(rec1.overflowed, false);

  router.recordMissingShader(0xbb02n, SHADER_STAGE.FRAGMENT, 'compile_failed');
  router.recordMissingShader('bb02', 'fs', 'compile_failed');
  router.recordMissingShader(hBinkPs, 'fragment', 'reflection_failed');

  router.recordMissingShader(hMenuVs, 'vertex', 'reflection_failed');
  router.recordMissingShader(hMenuVs, 1, 'reflection_failed');
  router.recordMissingShader(hComputeCs, 'compute', 'other');
  router.recordMissingShader(hComputeCs, 'cs', 'unrecognized_custom_tag');

  const summary = router.getMissingShaderSummary();
  assert.equal(summary.totalMissingEvents, 10);
  assert.equal(summary.uniqueMissingHashes, 4);
  assert.equal(summary.maxTrackedHashes, DEFAULT_MAX_CATEGORIZED_MISSING_HASHES);
  assert.equal(summary.overflowCount, 0);

  assert.deepEqual(summary.countsByReason, {
    not_in_pack: 2,
    reflection_failed: 3,
    compile_failed: 3,
    other: 2,
  });
  assert.deepEqual(summary.countsByStage, {
    vertex: 2,
    fragment: 6,
    compute: 2,
  });
  assert.deepEqual(summary.uniqueCountsByReason, {
    not_in_pack: 1,
    reflection_failed: 2,
    compile_failed: 1,
    other: 1,
  });
  assert.deepEqual(summary.uniqueCountsByStage, {
    vertex: 1,
    fragment: 2,
    compute: 1,
  });

  // Sorted by hitCount descending, ties broken deterministically by hashHex ascending.
  assert.equal(summary.topMissing.length, 4);
  assert.equal(summary.topMissing[0].hashHex, hBinkPs);
  assert.equal(summary.topMissing[0].hitCount, 4);
  assert.equal(summary.topMissing[0].reason, 'compile_failed'); // dominant (3 vs 1)
  assert.equal(summary.topMissing[0].lastReason, 'reflection_failed');
  assert.deepEqual(summary.topMissing[0].reasons, {
    not_in_pack: 0,
    reflection_failed: 1,
    compile_failed: 3,
    other: 0,
  });

  // Remaining 3 entries all have hitCount === 2 -> ordered lexicographically by hashHex.
  assert.deepEqual(
    summary.topMissing.slice(1).map((r) => ({ hashHex: r.hashHex, stage: r.stage, hitCount: r.hitCount, reason: r.reason })),
    [
      { hashHex: hMenuPs, stage: 'fragment', hitCount: 2, reason: 'not_in_pack' },
      { hashHex: hMenuVs, stage: 'vertex', hitCount: 2, reason: 'reflection_failed' },
      { hashHex: hComputeCs, stage: 'compute', hitCount: 2, reason: 'other' },
    ]
  );

  // Filter by reason, stage, and numeric/object limit.
  const reflOnly = router.getMissingShaderSummary({ reason: 'reflection_failed' });
  assert.equal(reflOnly.topMissing.length, 2);
  // hMenuVs has 2 reflection_failed hits vs hBinkPs 1 reflection_failed hit.
  assert.equal(reflOnly.topMissing[0].hashHex, hMenuVs);
  assert.equal(reflOnly.topMissing[1].hashHex, hBinkPs);

  const fragOnly = router.getMissingShaderSummary({ stage: 'fragment', limit: 1 });
  assert.equal(fragOnly.topMissing.length, 1);
  assert.equal(fragOnly.topMissing[0].hashHex, hBinkPs);

  assert.equal(router.getMissingShaderSummary(2).topMissing.length, 2);
  assert.deepEqual(router.getMissingShaderSummary(0).topMissing, []);

  // Strictly JSON-safe and mutation-isolated.
  const serialized = JSON.stringify(summary);
  assert.deepEqual(JSON.parse(serialized), summary);
  summary.countsByReason.compile_failed = 9999;
  summary.topMissing[0].hitCount = 9999;
  const fresh = router.getMissingShaderSummary();
  assert.equal(fresh.countsByReason.compile_failed, 3);
  assert.equal(fresh.topMissing[0].hitCount, 4);

  // Validation errors on bad inputs.
  assert.throws(() => router.getMissingShaderSummary(-1), RangeError);
  assert.throws(() => router.getMissingShaderSummary({ limit: 1.5 }), RangeError);
  assert.throws(() => router.getMissingShaderSummary('bad'), RangeError);
  assert.throws(() => router.recordMissingShader('0xzz', 'vertex', 'not_in_pack'), RangeError);
  assert.throws(() => router.recordMissingShader('0x1', 'geometry', 'not_in_pack'), RangeError);
});

test('W179 categorized missing-shader tracking is bounded at 8192 by default and handles overflow cleanly', () => {
  const defaultRouter = new ShaderFallbackRouter({ now: () => 1 });
  for (let i = 1; i <= 8192; i += 1) {
    const rec = defaultRouter.recordMissingShader(BigInt(i), 'fragment', 'not_in_pack');
    assert.equal(rec.tracked, true);
    assert.equal(rec.overflowed, false);
  }

  let summary = defaultRouter.getMissingShaderSummary({ limit: 3 });
  assert.equal(summary.uniqueMissingHashes, 8192);
  assert.equal(summary.maxTrackedHashes, 8192);
  assert.equal(summary.overflowCount, 0);
  assert.equal(summary.totalMissingEvents, 8192);
  assert.equal(summary.topMissing.length, 3);

  // 8193rd unseen hash overflows without growing unique map, while aggregate counters stay exact.
  const overflowed = defaultRouter.recordMissingShader(BigInt(8193), 'fragment', 'compile_failed');
  assert.equal(overflowed.tracked, false);
  assert.equal(overflowed.overflowed, true);

  // Already-tracked hash still increments hitCount without overflowing.
  const repeated = defaultRouter.recordMissingShader(BigInt(1), 'fragment', 'reflection_failed');
  assert.equal(repeated.tracked, true);
  assert.equal(repeated.overflowed, false);
  assert.equal(repeated.hitCount, 2);

  summary = defaultRouter.getMissingShaderSummary({ limit: 1 });
  assert.equal(summary.uniqueMissingHashes, 8192);
  assert.equal(summary.overflowCount, 1);
  assert.equal(summary.totalMissingEvents, 8194);
  assert.equal(summary.countsByReason.not_in_pack, 8192);
  assert.equal(summary.countsByReason.compile_failed, 1);
  assert.equal(summary.countsByReason.reflection_failed, 1);
  assert.equal(summary.topMissing[0].hashHex, '0x0000000000000001');
  assert.equal(summary.topMissing[0].hitCount, 2);

  // Custom maxCategorizedMissingHashes validation & zero cap.
  const zeroCap = new ShaderFallbackRouter({ maxCategorizedMissingHashes: 0, now: () => 2 });
  const z = zeroCap.recordMissingShader('0x1234', 'vertex', 'compile_failed');
  assert.equal(z.tracked, false);
  assert.equal(z.overflowed, true);
  const zSum = zeroCap.getMissingShaderSummary();
  assert.equal(zSum.uniqueMissingHashes, 0);
  assert.equal(zSum.overflowCount, 1);
  assert.equal(zSum.countsByReason.compile_failed, 1);

  assert.throws(
    () => new ShaderFallbackRouter({ maxCategorizedMissingHashes: -1 }),
    /maxCategorizedMissingHashes/
  );
  assert.throws(
    () => new ShaderFallbackRouter({ maxCategorizedMissingHashes: 4.2 }),
    /maxCategorizedMissingHashes/
  );

  // reset() clears categorized missing map and totals.
  defaultRouter.reset();
  const afterReset = defaultRouter.getMissingShaderSummary();
  assert.equal(afterReset.totalMissingEvents, 0);
  assert.equal(afterReset.uniqueMissingHashes, 0);
  assert.equal(afterReset.overflowCount, 0);
  assert.deepEqual(afterReset.topMissing, []);
});

test('W179 classic Worker importScripts isolation exposes new APIs with zero global leaks', () => {
  const source = readFileSync(ROUTER_PATH, 'utf8');
  const sandbox = { console, performance };
  const vmCtx = vm.createContext(sandbox);
  vmCtx.self = vmCtx;

  vm.runInContext(source, vmCtx, { filename: ROUTER_PATH });
  vm.runInContext(source, vmCtx, { filename: ROUTER_PATH });

  assert.doesNotThrow(() => {
    vm.runInContext(
      `
      const MISSING_SHADER_REASONS = {};
      const MISSING_SHADER_REASON_ORDER = [];
      const MISSING_SHADER_REASON_ALIASES = {};
      const DEFAULT_MAX_CATEGORIZED_MISSING_HASHES = 0;
      const DEFAULT_MISSING_SUMMARY_TOP_LIMIT = 0;
      `,
      vmCtx
    );
  });

  const api = vmCtx.PlayGta6ShaderFallbackRouter;
  assert.ok(api.MISSING_SHADER_REASONS);
  assert.equal(api.DEFAULT_MAX_CATEGORIZED_MISSING_HASHES, 8192);
  assert.equal(typeof api.normalizeMissingReason, 'function');
  const r = new api.ShaderFallbackRouter();
  assert.equal(typeof r.recordMissingShader, 'function');
  assert.equal(typeof r.getMissingShaderSummary, 'function');
  assert.equal(typeof r.registerRealShader, 'function');
  assert.equal(typeof r.hasRealShader, 'function');
  assert.equal(typeof r.canFallbackOverride, 'function');
});

// ---------------------------------------------------------------------------
// W204 Fast-path memoization, pack-reload invalidation, O(1) missing tracking,
//      frame-indexed top-N ranking, and resetMissingTelemetry()
// ---------------------------------------------------------------------------

test('W204 fast-path resolve memoization returns identical frozen results with 0 pack lookups on cache hits', () => {
  assert.equal(DEFAULT_MAX_RESOLVE_CACHE_ENTRIES, 8192);

  let packLookupCalls = 0;
  const packHashHex = '0x000000000000cafe';
  const packHashBig = 0xcafen;
  const missHashHex = '0x000000000000beef';
  const fallbackMissHex = '0x000000000000fade';

  const backing = createMapPackLoader({
    [packHashHex]: PACK_VS_WGSL,
    [S9_FALLBACK_HASHES.fragment]: S9_FALLBACK_WGSL.fragment,
  });
  const countingLoader = {
    getSource(key) {
      packLookupCalls += 1;
      return backing.get(key);
    },
  };

  let clock = 100;
  const router = new ShaderFallbackRouter({
    packLoader: countingLoader,
    fallbackHashes: { fragment: S9_FALLBACK_HASHES.fragment },
    now: () => {
      clock += 1;
      return clock;
    },
  });

  // 1) Pack hit: first lookup queries packLoader once; repeat lookups across BigInt/hex/number hit cache
  //    returning the exact same frozen result object reference (0 temporary object allocations).
  const p1 = router.resolve(packHashHex, 'vertex');
  assert.equal(p1.source, 'pack');
  assert.equal(p1.wgsl, PACK_VS_WGSL);
  assert.equal(Object.isFrozen(p1), true);
  const callsAfterFirstPackHit = packLookupCalls;
  assert.ok(callsAfterFirstPackHit >= 1);

  for (let i = 0; i < 100; i += 1) {
    const pHitHex = router.resolve(packHashHex, 'vertex');
    const pHitBig = router.resolve(packHashBig, SHADER_STAGE.VERTEX);
    const pHitAlias = router.resolve(packHashHex, 'vs');
    assert.strictEqual(pHitHex, p1);
    assert.strictEqual(pHitBig, p1);
    assert.strictEqual(pHitAlias, p1);
  }
  assert.equal(packLookupCalls, callsAfterFirstPackHit, 'packLoader must not be queried on cache hits');

  // Pack hit ignores profile option and still hits fast-path cache.
  const pWithProfile = router.resolve(packHashHex, 'vertex', { profile: 'splash_rect_24b' });
  assert.strictEqual(pWithProfile, p1);
  assert.equal(packLookupCalls, callsAfterFirstPackHit);

  // 2) S9 fallback hit: memoized with identical frozen object reference and O(1) missing count updates.
  const f1 = router.resolve(fallbackMissHex, 'fragment');
  assert.equal(f1.source, 'fallback');
  assert.equal(Object.isFrozen(f1), true);
  const callsAfterFallbackMiss = packLookupCalls;

  for (let i = 0; i < 10; i += 1) {
    const fHit = router.resolve(fallbackMissHex, 'fs');
    assert.strictEqual(fHit, f1);
  }
  assert.equal(packLookupCalls, callsAfterFallbackMiss, 'fallback cache hits must not re-query packLoader');

  // 3) Builtin hit with profile: memoized for matching profile, re-resolves cleanly for a different profile.
  const b1 = router.resolve(missHashHex, 'vertex', { profile: 'splash_rect_24b' });
  assert.equal(b1.source, 'builtin');
  assert.equal(b1.profile, 'splash_rect_24b');
  assert.equal(Object.isFrozen(b1), true);
  const callsAfterBuiltinMiss = packLookupCalls;

  const b2 = router.resolve(missHashHex, 'vertex', { profile: 'splash_rect_24b' });
  assert.strictEqual(b2, b1);
  assert.equal(packLookupCalls, callsAfterBuiltinMiss);

  // Requesting a different profile must NOT return stale cached profile.
  const bOtherProfile = router.resolve(missHashHex, 'vertex', { profile: 'textured_quad_24b' });
  assert.equal(bOtherProfile.source, 'builtin');
  assert.equal(bOtherProfile.profile, 'textured_quad_24b');
  assert.notStrictEqual(bOtherProfile, b1);

  // Verify aggregate stats and resolve cache telemetry.
  const stats = router.getStats();
  assert.equal(stats.packHits, 302);
  assert.equal(stats.fallbackHits, 11);
  assert.equal(stats.builtinHits, 3);
  assert.equal(stats.totalLookups, 316);
  assert.equal(stats.uniqueMissing, 2);

  const cacheStats = router.getResolveCacheStats();
  assert.equal(cacheStats.entries, 3);
  assert.equal(cacheStats.maxEntries, DEFAULT_MAX_RESOLVE_CACHE_ENTRIES);
  assert.equal(cacheStats.invalidations, 0);
  assert.equal(cacheStats.hits, 312);
  assert.equal(cacheStats.misses, 4);
  assert.ok(cacheStats.hitRatio > 0.98 && cacheStats.hitRatio <= 1);
});

test('W204 resolve cache invalidates cleanly on pack reload (setPackLoader, version bump, Map mutation, onPackReload, fallbackHashes, realShaders)', () => {
  const h1 = '0x000000000000a001';
  const h2 = '0x000000000000a002';
  const v1Wgsl = '@vertex fn vs_v1() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }';
  const v2Wgsl = '@vertex fn vs_v2() -> @builtin(position) vec4<f32> { return vec4<f32>(1.0); }';

  // 1) Explicit setPackLoader reload: previously missing h2 becomes a pack hit; h1 gets updated WGSL.
  const router = new ShaderFallbackRouter({
    packLoader: createGetSourcePackLoader({ [h1]: v1Wgsl }),
    now: () => 10,
  });

  const beforeH1 = router.resolve(h1, 'vertex');
  assert.equal(beforeH1.source, 'pack');
  assert.equal(beforeH1.wgsl, v1Wgsl);
  const beforeH2 = router.resolve(h2, 'vertex');
  assert.equal(beforeH2.source, 'builtin');
  assert.equal(router.getResolveCacheStats().entries, 2);

  router.setPackLoader(createGetSourcePackLoader({ [h1]: v2Wgsl, [h2]: PACK_VS_WGSL }));
  assert.equal(router.getResolveCacheStats().entries, 0);
  assert.equal(router.getResolveCacheStats().invalidations, 1);

  const afterH1 = router.resolve(h1, 'vertex');
  assert.equal(afterH1.source, 'pack');
  assert.equal(afterH1.wgsl, v2Wgsl);
  assert.equal(afterH1.entryPoint, 'vs_v2');

  const afterH2 = router.resolve(h2, 'vertex');
  assert.equal(afterH2.source, 'pack');
  assert.equal(afterH2.wgsl, PACK_VS_WGSL);

  // 2) Versioned in-place pack reload automatically detected on next resolve().
  const versionedEntries = new Map([[h1, v1Wgsl]]);
  const versionedLoader = {
    version: 1,
    getSource(key) {
      return versionedEntries.get(typeof key === 'bigint' ? formatShaderHash(key) : key);
    },
  };
  router.setPackLoader(versionedLoader);
  assert.equal(router.resolve(h1, 'vertex').wgsl, v1Wgsl);
  assert.equal(router.resolve(h2, 'vertex').source, 'builtin');

  versionedEntries.set(h1, v2Wgsl);
  versionedEntries.set(h2, PACK_VS_WGSL);
  versionedLoader.version = 2;

  assert.equal(router.resolve(h1, 'vertex').wgsl, v2Wgsl);
  assert.equal(router.resolve(h2, 'vertex').source, 'pack');

  // 3) Map packLoader size change automatically invalidates cache.
  const mapPack = new Map([[h1, v1Wgsl]]);
  router.setPackLoader(mapPack);
  assert.equal(router.resolve(h2, 'vertex').source, 'builtin');
  mapPack.set(h2, PACK_VS_WGSL);
  assert.equal(router.resolve(h2, 'vertex').source, 'pack');

  // 4) Same-size in-place reload invalidated explicitly via onPackReload() / invalidateCache().
  mapPack.set(h1, v2Wgsl);
  router.onPackReload();
  assert.equal(router.resolve(h1, 'vertex').wgsl, v2Wgsl);

  // 5) setFallbackHashes and registerRealShader / unregisterRealShader invalidate cached misses.
  const fbRouter = new ShaderFallbackRouter({
    packLoader: createS9PackLoader(),
    now: () => 20,
  });
  assert.equal(fbRouter.resolve(h1, 'vertex').source, 'builtin');
  fbRouter.setFallbackHashes({ vertex: S9_FALLBACK_HASHES.vertex });
  assert.equal(fbRouter.resolve(h1, 'vertex').source, 'fallback');

  fbRouter.registerRealShader(h1, 'vertex', { wgsl: PACK_VS_WGSL, entryPoint: 'vs_main' });
  const guarded = fbRouter.resolve(h1, 'vertex');
  assert.equal(guarded.source, 'real');
  assert.equal(guarded.fallbackBlocked, true);

  fbRouter.unregisterRealShader(h1, 'vertex');
  assert.equal(fbRouter.resolve(h1, 'vertex').source, 'fallback');

  // 6) Ranged-mode WGPK entries whose payload is still in flight are not negatively cached,
  //    so once the HTTP 206 Range payload lands in LRU, the next resolve() returns 'pack'.
  const rangedKey = normalizeShaderHash(h1);
  const rangedLru = new Map();
  const rangedPackStub = { mode: 'ranged' };
  const rangedLoader = {
    _shaderEntries: new Map([[rangedKey, { pack: rangedPackStub }]]),
    getWgsl(key) {
      return rangedLru.get(normalizeShaderHash(key));
    },
  };
  const rangedRouter = new ShaderFallbackRouter({ packLoader: rangedLoader, now: () => 30 });
  assert.equal(rangedRouter.resolve(h1, 'vertex').source, 'builtin');
  assert.equal(rangedRouter.getResolveCacheStats().entries, 0);
  rangedLru.set(rangedKey, PACK_VS_WGSL);
  const rangedReady = rangedRouter.resolve(h1, 'vertex');
  assert.equal(rangedReady.source, 'pack');
  assert.equal(rangedReady.wgsl, PACK_VS_WGSL);
  assert.equal(rangedRouter.getResolveCacheStats().entries, 1);
});

test('W204 bounded resolve cache eviction and option validation', () => {
  assert.throws(
    () => new ShaderFallbackRouter({ maxResolveCacheEntries: -1 }),
    /maxResolveCacheEntries/
  );
  assert.throws(
    () => new ShaderFallbackRouter({ maxResolveCacheEntries: 1.5 }),
    /maxResolveCacheEntries/
  );

  let lookups = 0;
  const backing = createMapPackLoader({
    '0x0000000000000001': PACK_VS_WGSL,
    '0x0000000000000002': PACK_VS_WGSL,
    '0x0000000000000003': PACK_VS_WGSL,
  });
  const router = new ShaderFallbackRouter({
    maxResolveCacheEntries: 2,
    packLoader: {
      getSource(key) {
        lookups += 1;
        return backing.get(key);
      },
    },
  });

  router.resolve('0x0000000000000001', 'vertex');
  router.resolve('0x0000000000000002', 'vertex');
  assert.equal(router.getResolveCacheStats().entries, 2);

  // Both 1 and 2 are cached (even across BigInt/hex formats).
  const beforeHits = lookups;
  router.resolve(1n, 'vertex');
  router.resolve(2n, 'vertex');
  assert.equal(lookups, beforeHits);

  // Adding a 3rd entry evicts oldest (1), keeping entries capped at 2.
  router.resolve('0x0000000000000003', 'vertex');
  assert.equal(router.getResolveCacheStats().entries, 2);

  // 2 and 3 hit cache without querying packLoader; 1 misses and re-queries.
  const after3 = lookups;
  router.resolve('0x0000000000000002', 'vertex');
  router.resolve('0x0000000000000003', 'vertex');
  assert.equal(lookups, after3);
  router.resolve('0x0000000000000001', 'vertex');
  assert.ok(lookups > after3);

  // maxResolveCacheEntries: 0 disables memoization cleanly.
  let uncachedLookups = 0;
  const uncached = new ShaderFallbackRouter({
    maxResolveCacheEntries: 0,
    packLoader: {
      getSource(key) {
        uncachedLookups += 1;
        return backing.get(key);
      },
    },
  });
  uncached.resolve('0x0000000000000001', 'vertex');
  uncached.resolve('0x0000000000000001', 'vertex');
  assert.equal(uncachedLookups, 2);
  assert.equal(uncached.getResolveCacheStats().entries, 0);
  assert.equal(uncached.getResolveCacheStats().hits, 0);
});

test('W204 frame tracking and fast min-heap getTopMissingShaders ranking accuracy across 500+ hashes', () => {
  const router = new ShaderFallbackRouter({ initialFrame: 1, now: () => 500 });
  assert.equal(router.getCurrentFrame(), 1);

  assert.throws(() => router.setFrame(-1), RangeError);
  assert.throws(() => router.setFrame(1.5), RangeError);
  assert.throws(() => router.advanceFrame(-1), RangeError);

  const stages = ['vertex', 'fragment', 'compute'];
  const expected = [];

  // Populate 520 unique missing (hash, stage) pairs across multiple frames with deterministic counts.
  for (let i = 1; i <= 520; i += 1) {
    const hashBig = BigInt(0x10000 + i);
    const hashHex = formatShaderHash(hashBig);
    const stage = stages[i % 3];
    const count = (i % 17 === 0) ? 20 + (i % 7) : (i % 5) + 1;
    const firstFrame = i;
    const lastFrame = i + count - 1;

    for (let c = 0; c < count; c += 1) {
      router.setFrame(firstFrame + c);
      router.resolve(hashBig, stage);
    }
    expected.push({
      hash: hashHex,
      hashHex,
      stage,
      count,
      firstSeenFrame: firstFrame,
      lastSeenFrame: lastFrame,
    });
  }

  // Also test advanceFrame() and per-call { frame } override on an existing hot missing hash.
  router.setFrame(900);
  assert.equal(router.advanceFrame(), 901);
  assert.equal(router.advanceFrame(4), 905);
  const hotEntry = expected[0];
  router.resolve(hotEntry.hash, hotEntry.stage, { frame: 999 });
  hotEntry.count += 1;
  hotEntry.lastSeenFrame = 999;

  expected.sort(
    (a, b) =>
      b.count - a.count ||
      (a.hash < b.hash ? -1 : a.hash > b.hash ? 1 : 0) ||
      (a.stage < b.stage ? -1 : a.stage > b.stage ? 1 : 0)
  );

  // Verify exact top-N match against full sort for small, medium, and full limits.
  for (const limit of [1, 5, 16, 32, 100, 520, 1000]) {
    const actual = router.getTopMissingShaders(limit);
    assert.deepEqual(actual, expected.slice(0, limit));
  }

  // Strictly sanitized: zero host paths, binary dumps, or WGSL text.
  const top16 = router.getTopMissingShaders(16);
  const json = JSON.stringify(top16);
  assert.deepEqual(JSON.parse(json), top16);
  assert.equal(json.includes('@vertex'), false);
  assert.equal(json.includes('@fragment'), false);
  assert.equal(json.includes('C:\\'), false);
  assert.equal(json.includes('/Users/'), false);
});

test('W204 resetMissingTelemetry clears missing tracking while preserving pack cache, real shaders, and lookup stats', () => {
  const packHash = '0x000000000000aa11';
  const realHash = '0x000000000000bb22';
  const missA = '0x000000000000cc33';
  const missB = '0x000000000000dd44';
  const missOverflow = '0x000000000000ee55';

  let packCalls = 0;
  const backing = createMapPackLoader({ [packHash]: PACK_VS_WGSL });
  const router = new ShaderFallbackRouter({
    packLoader: {
      getSource(key) {
        packCalls += 1;
        return backing.get(key);
      },
    },
    maxTrackedMissingHashes: 2,
    initialFrame: 10,
    now: () => 1000,
  });

  router.registerRealShader(realHash, 'fragment', { wgsl: MENU_UI_PS_WGSL });

  // Warm pack hit + fill bounded missing cap + trigger overflow.
  router.resolve(packHash, 'vertex');
  router.resolve(missA, 'vertex');
  router.resolve(missA, 'vertex');
  router.resolve(missB, 'fragment');
  router.resolve(missOverflow, 'compute'); // overflows cap of 2
  router.resolve(missOverflow, 'compute'); // cached miss hit while overflowed -> still increments overflow!
  const callsBeforeReset = packCalls;

  assert.equal(router.getStats().uniqueMissing, 2);
  assert.equal(router.getStats().overflowMissingHashes, 2);
  assert.equal(router.getTopMissingShaders().length, 2);
  assert.equal(router.getMissingShaderSummary().totalMissingEvents, 5);

  // resetMissingTelemetry() clears missing & categorized telemetry only.
  router.resetMissingTelemetry();
  assert.equal(router.getStats().uniqueMissing, 0);
  assert.equal(router.getStats().overflowMissingHashes, 0);
  assert.equal(router.getStats().overflowMissingCount, 0);
  assert.equal(router.getStats().packHits, 1, 'packHits preserved by resetMissingTelemetry');
  assert.equal(router.getStats().totalLookups, 6, 'totalLookups preserved by resetMissingTelemetry');
  assert.deepEqual(router.getTopMissingShaders(), []);
  assert.deepEqual(router.getMissingReport(), []);
  assert.equal(router.getMissingShaderSummary().totalMissingEvents, 0);
  assert.equal(router.hasRealShader(realHash, 'fragment'), true);

  // Memoized pack shader still hits fast-path cache without re-querying packLoader.
  router.resolve(packHash, 'vertex');
  assert.equal(packCalls, callsBeforeReset);

  // Previously cached & overflowed missing hashes re-track cleanly with new frame numbers.
  router.setFrame(42);
  router.resolve(missOverflow, 'compute');
  router.setFrame(45);
  router.resolve(missOverflow, 'compute');
  router.resolve(missA, 'vertex');

  assert.deepEqual(router.getTopMissingShaders(), [
    {
      hash: missOverflow,
      hashHex: missOverflow,
      stage: 'compute',
      count: 2,
      firstSeenFrame: 42,
      lastSeenFrame: 45,
    },
    {
      hash: missA,
      hashHex: missA,
      stage: 'vertex',
      count: 1,
      firstSeenFrame: 45,
      lastSeenFrame: 45,
    },
  ]);
  assert.equal(router.getStats().uniqueMissing, 2);
  assert.equal(router.getStats().overflowMissingHashes, 0);
});

test('W204 classic Worker importScripts isolation exposes W204 APIs with zero global leaks', () => {
  const source = readFileSync(ROUTER_PATH, 'utf8');
  const sandbox = { console, performance };
  const vmCtx = vm.createContext(sandbox);
  vmCtx.self = vmCtx;

  vm.runInContext(source, vmCtx, { filename: ROUTER_PATH });
  vm.runInContext(source, vmCtx, { filename: ROUTER_PATH });

  assert.doesNotThrow(() => {
    vm.runInContext(
      `
      const DEFAULT_MAX_RESOLVE_CACHE_ENTRIES = 0;
      const ENTRY_POINT_REGEXES = {};
      const FALLBACK_FN_REGEX = /x/;
      `,
      vmCtx
    );
  });

  const api = vmCtx.PlayGta6ShaderFallbackRouter;
  assert.equal(api.DEFAULT_MAX_RESOLVE_CACHE_ENTRIES, 8192);
  const r = new api.ShaderFallbackRouter();
  assert.equal(typeof r.resetMissingTelemetry, 'function');
  assert.equal(typeof r.invalidateCache, 'function');
  assert.equal(typeof r.onPackReload, 'function');
  assert.equal(typeof r.getResolveCacheStats, 'function');
  assert.equal(typeof r.setFrame, 'function');
  assert.equal(typeof r.advanceFrame, 'function');
  assert.equal(typeof r.getCurrentFrame, 'function');
});

// ---------------------------------------------------------------------------
// W268 Bounded negative compile-failure LRU cache & deterministic fallback routing
// ---------------------------------------------------------------------------

test('W268 recordMissingShader(compile_failed) populates negative compile-failure cache and ignores non-compile reasons', () => {
  assert.equal(DEFAULT_MAX_COMPILE_FAILURE_ENTRIES, 2048);

  let clock = 2000;
  const router = new ShaderFallbackRouter({
    now: () => {
      clock += 5;
      return clock;
    },
  });

  const brokenHash = '0x00000000deadb001';
  const missingHash = '0x00000000deadb002';
  const reflHash = '0x00000000deadb003';
  const otherHash = '0x00000000deadb004';

  // Non-compile failure reasons do NOT enter the negative compile-failure cache.
  router.recordMissingShader(missingHash, 'fragment', 'not_in_pack');
  router.recordMissingShader(missingHash, 'fragment', 'missing_pack_entry');
  router.recordMissingShader(reflHash, 'vertex', 'reflection_failed');
  router.recordMissingShader(otherHash, 'compute', 'other');
  assert.equal(router.hasCompileFailure(missingHash), false);
  assert.equal(router.hasCompileFailure(reflHash), false);
  assert.equal(router.hasCompileFailure(otherHash), false);
  assert.equal(router.getStats().compileFailureEntries, 0);
  assert.equal(router.getStats().compileFailureRecorded, 0);
  assert.equal(router.getStats().compileFailureCacheHits, 0);

  // compile_failed (and aliases) record into the negative compile-failure cache keyed by canonical hash.
  router.recordMissingShader(brokenHash, 'fragment', 'compile_failed');
  router.recordMissingShader(0xdeadb001n, SHADER_STAGE.FRAGMENT, 'wgsl_compile_failed');
  router.recordMissingShader('deadb001', 'fs', 'compilation_failed');

  let stats = router.getStats();
  assert.equal(stats.compileFailureEntries, 1);
  assert.equal(stats.compileFailureRecorded, 3);
  assert.equal(stats.compileFailureCacheHits, 0);

  // Queries across hex, BigInt, bare hex, and safe integer all hit the same canonical entry.
  assert.equal(router.hasCompileFailure(brokenHash), true);
  assert.equal(router.hasCompileFailure(0xdeadb001n), true);
  assert.equal(router.hasCompileFailure('DEADB001'), true);
  assert.equal(router.hasCompileFailure(0xdeadb001), true);
  assert.equal(router.hasCompileFailure(null), false);
  assert.equal(router.hasCompileFailure(undefined), false);
  assert.equal(router.hasCompileFailure(''), false);

  stats = router.getStats();
  assert.equal(stats.compileFailureEntries, 1);
  assert.equal(stats.compileFailureRecorded, 3);
  assert.equal(stats.compileFailureCacheHits, 4);

  // getCompileFailure returns a mutation-isolated sanitized snapshot without changing hit counters.
  const entry = router.getCompileFailure(brokenHash);
  assert.deepEqual(entry, {
    hashHex: brokenHash,
    stage: 'fragment',
    reason: 'compile_failed',
    count: 3,
    firstSeenMs: 2025,
    lastSeenMs: 2035,
  });
  entry.reason = 'mutated';
  assert.equal(router.getCompileFailure(brokenHash).reason, 'compile_failed');
  assert.equal(router.getStats().compileFailureCacheHits, 4);
  assert.equal(router.getCompileFailure(null), null);
  assert.equal(router.getCompileFailure(missingHash), null);
});

test('W268 sanitizeCompileFailureReason and recordCompileFailure store sanitized reasons only', () => {
  assert.equal(sanitizeCompileFailureReason('compile_failed'), 'compile_failed');
  assert.equal(sanitizeCompileFailureReason('missing_pack_entry'), 'missing_pack_entry');
  assert.equal(sanitizeCompileFailureReason('TINT-VALIDATION-FAILED'), 'tint_validation_failed');
  assert.equal(sanitizeCompileFailureReason(''), 'compile_failed');
  assert.equal(sanitizeCompileFailureReason(null), 'compile_failed');
  assert.equal(sanitizeCompileFailureReason(undefined), 'compile_failed');
  // Host paths and raw WGSL snippets are stripped down to safe default token.
  assert.equal(sanitizeCompileFailureReason('C:\\Users\\dev\\game\\shader.wgsl: line 42'), 'compile_failed');
  assert.equal(sanitizeCompileFailureReason('@fragment fn fs_main() { error }'), 'compile_failed');

  const router = new ShaderFallbackRouter({ now: () => 500 });
  const rec1 = router.recordCompileFailure('0x000000000000c001', 'fragment', 'missing_pack_entry');
  assert.equal(rec1.hashHex, '0x000000000000c001');
  assert.equal(rec1.stage, 'fragment');
  assert.equal(rec1.reason, 'missing_pack_entry');
  assert.equal(rec1.cached, true);

  const rec2 = router.recordCompileFailure('0x000000000000c002', {
    stage: 'vertex',
    reason: 'C:\\secret\\host\\path\\bad.wgsl',
  });
  assert.equal(rec2.reason, 'compile_failed');
  assert.equal(router.getCompileFailure('0x000000000000c002').reason, 'compile_failed');

  const serialized = JSON.stringify([
    router.getCompileFailure('0x000000000000c001'),
    router.getCompileFailure('0x000000000000c002'),
  ]);
  assert.equal(serialized.includes('secret'), false);
  assert.equal(serialized.includes('bad.wgsl'), false);
});

test('W268 negative compile-failure cache enforces maxCompileFailureEntries with LRU eviction and clear helpers', () => {
  assert.throws(
    () => new ShaderFallbackRouter({ maxCompileFailureEntries: -1 }),
    /maxCompileFailureEntries/
  );
  assert.throws(
    () => new ShaderFallbackRouter({ maxCompileFailureEntries: 2.5 }),
    /maxCompileFailureEntries/
  );
  assert.throws(
    () => new ShaderFallbackRouter({ maxCompileFailureEntries: '4' }),
    /maxCompileFailureEntries/
  );

  // maxCompileFailureEntries: 0 disables caching while still counting recorded failures.
  const zeroCache = new ShaderFallbackRouter({ maxCompileFailureEntries: 0, now: () => 1 });
  zeroCache.recordMissingShader('0x0000000000000001', 'vertex', 'compile_failed');
  assert.equal(zeroCache.hasCompileFailure('0x0000000000000001'), false);
  assert.equal(zeroCache.getStats().compileFailureEntries, 0);
  assert.equal(zeroCache.getStats().compileFailureRecorded, 1);
  assert.equal(zeroCache.getStats().compileFailureCacheHits, 0);

  // Bounded LRU cache of size 2: touching A makes B the LRU victim when C arrives.
  const router = new ShaderFallbackRouter({ maxCompileFailureEntries: 2, now: () => 10 });
  const hA = '0x000000000000000a';
  const hB = '0x000000000000000b';
  const hC = '0x000000000000000c';

  router.recordMissingShader(hA, 'vertex', 'compile_failed');
  router.recordMissingShader(hB, 'fragment', 'compile_failed');
  assert.equal(router.getStats().compileFailureEntries, 2);

  // Querying hA promotes hA to MRU, leaving hB as LRU.
  assert.equal(router.hasCompileFailure(hA), true);
  assert.equal(router.getStats().compileFailureCacheHits, 1);

  // Inserting hC evicts hB (LRU), keeping hA and hC.
  router.recordMissingShader(hC, 'compute', 'compile_failed');
  assert.equal(router.getStats().compileFailureEntries, 2);
  assert.equal(router.hasCompileFailure(hB), false);
  assert.equal(router.hasCompileFailure(hA), true);
  assert.equal(router.hasCompileFailure(hC), true);
  assert.equal(router.getStats().compileFailureCacheHits, 3);

  // clearCompileFailure removes a single entry and returns boolean status.
  assert.equal(router.clearCompileFailure(hA), true);
  assert.equal(router.clearCompileFailure(hA), false);
  assert.equal(router.clearCompileFailure(null), false);
  assert.equal(router.hasCompileFailure(hA), false);
  assert.equal(router.getStats().compileFailureEntries, 1);

  // clearCompileFailures removes remaining entries while keeping cumulative hit/record counters.
  assert.equal(router.clearCompileFailures(), 1);
  assert.equal(router.clearCompileFailures(), 0);
  assert.equal(router.getStats().compileFailureEntries, 0);
  assert.equal(router.getStats().compileFailureCacheHits, 3);
  assert.equal(router.getStats().compileFailureRecorded, 3);
});

test('W268 resolve skips known compile-failed pack shaders without spamming retries and resumes after clear', () => {
  const brokenPackHash = '0x00000000bad00001';
  const healthyPackHash = '0x00000000cafe0002';
  const brokenWgsl = '@fragment fn fs_broken() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }';

  let packQueries = 0;
  const backing = createMapPackLoader({
    [brokenPackHash]: brokenWgsl,
    [healthyPackHash]: PACK_VS_WGSL,
  });
  const router = new ShaderFallbackRouter({
    packLoader: {
      getSource(key) {
        packQueries += 1;
        return backing.get(key);
      },
    },
    defaultProfile: 'splash_rect_24b',
    now: () => 300,
  });

  // Initial resolve returns pack WGSL (and memoizes it in W204 resolve cache).
  const initial = router.resolve(brokenPackHash, 'fragment');
  assert.equal(initial.source, 'pack');
  assert.equal(initial.wgsl, brokenWgsl);
  assert.equal(router.resolve(healthyPackHash, 'vertex').source, 'pack');

  // Tint rejects brokenPackHash -> recordMissingShader('compile_failed') evicts stale pack cache entry.
  router.recordMissingShader(brokenPackHash, 'fragment', 'compile_failed');
  const queriesAfterFailure = packQueries;

  // Next 64 draws must NEVER re-query packLoader or return brokenWgsl; route straight to fallback.
  const profiles = getBuiltInFallbackProfiles();
  for (let i = 0; i < 64; i += 1) {
    const res = router.resolve(brokenPackHash, 'fragment', {
      bindingSignature: { hasTexture: true, hasSampler: true },
    });
    assert.equal(res.source, 'builtin');
    assert.equal(res.profile, 'textured_quad_24b');
    assert.equal(res.wgsl, profiles.textured_quad_24b.fragment);
    assert.equal(res.compileFailureCached, true);
    assert.equal(res.compileFailureReason, 'compile_failed');
  }
  assert.equal(
    packQueries,
    queriesAfterFailure,
    'negative compile-failure cache must skip packLoader lookups on subsequent draws'
  );

  // Healthy sibling shader stays memoized and unaffected.
  assert.equal(router.resolve(healthyPackHash, 'vertex').source, 'pack');
  assert.equal(packQueries, queriesAfterFailure);

  const stats = router.getStats();
  assert.equal(stats.compileFailureEntries, 1);
  assert.equal(stats.compileFailureRecorded, 1);
  assert.equal(stats.compileFailureCacheHits, 64);

  // Clearing the compile failure restores normal pack resolution.
  assert.equal(router.clearCompileFailure(brokenPackHash), true);
  const recovered = router.resolve(brokenPackHash, 'fragment');
  assert.equal(recovered.source, 'pack');
  assert.equal(recovered.wgsl, brokenWgsl);
  assert.equal(recovered.compileFailureCached, undefined);

  // If an S9 fallback hash itself fails compilation, resolve degrades safely to built-in WGSL.
  const s9Router = new ShaderFallbackRouter({
    packLoader: createS9PackLoader(),
    fallbackHashes: S9_FALLBACK_HASHES,
    now: () => 400,
  });
  assert.equal(s9Router.resolve('0x1111111122222222', 'fragment').source, 'fallback');
  s9Router.recordMissingShader(S9_FALLBACK_HASHES.fragment, 'fragment', 'compile_failed');
  const degraded = s9Router.resolve('0x1111111122222222', 'fragment', {
    bindingSignature: { hasCbv: true },
  });
  assert.equal(degraded.source, 'builtin');
  assert.equal(degraded.profile, 'fullscreen_triangle_cbv');
  assert.equal(degraded.wgsl, profiles.fullscreen_triangle_cbv.fragment);
});

test('W268 deterministic fallback profile routing maps stage + bindingSignature across UI, vertex-colored, fullscreen, and compute', () => {
  // Compute stage always routes deterministically to compute no-op ('default').
  assert.equal(selectDeterministicFallbackProfile('compute', null), 'default');
  assert.equal(
    selectDeterministicFallbackProfile(SHADER_STAGE.COMPUTE, {
      hasCbv: true,
      hasTexture: true,
      hasVertexColor: true,
    }),
    'default'
  );

  // Vertex-colored UI or CBV+texture pass -> splash_rect_24b.
  assert.equal(
    selectDeterministicFallbackProfile('vertex', { hasVertexColor: true }),
    'splash_rect_24b'
  );
  assert.equal(
    selectDeterministicFallbackProfile('fragment', { hasCbv: true, hasTexture: true }),
    'splash_rect_24b'
  );
  assert.equal(
    selectDeterministicFallbackProfile('vertex', { attributeCount: 3 }),
    'splash_rect_24b'
  );

  // Textured UI / 24-byte vertex stride -> textured_quad_24b.
  assert.equal(
    selectDeterministicFallbackProfile('fragment', { hasTexture: true, hasSampler: true }),
    'textured_quad_24b'
  );
  assert.equal(
    selectDeterministicFallbackProfile('vertex', { vertexStride: 24 }),
    'textured_quad_24b'
  );

  // Fullscreen pass with CBV or vertexCount===3 -> fullscreen_triangle_cbv.
  assert.equal(
    selectDeterministicFallbackProfile('vertex', { hasCbv: true }),
    'fullscreen_triangle_cbv'
  );
  assert.equal(
    selectDeterministicFallbackProfile('fragment', { vertexCount: 3 }),
    'fullscreen_triangle_cbv'
  );

  // Minimal / empty signature -> default.
  assert.equal(selectDeterministicFallbackProfile('vertex', {}), 'default');
  assert.equal(selectDeterministicFallbackProfile('fragment', null), 'default');

  const router = new ShaderFallbackRouter({ now: () => 600 });
  const profiles = getBuiltInFallbackProfiles();

  assert.equal(router.selectFallbackProfile('fragment', { hasTexture: true }), 'textured_quad_24b');

  const uiSplash = router.resolve('0x000000000000f001', 'fragment', {
    bindingSignature: { hasVertexColor: true, hasTexture: true, hasCbv: true },
  });
  assert.equal(uiSplash.profile, 'splash_rect_24b');
  assert.equal(uiSplash.wgsl, profiles.splash_rect_24b.fragment);

  const uiTex = router.resolve('0x000000000000f002', 'vertex', {
    bindingSignature: { textureCount: 1, samplerCount: 1 },
  });
  assert.equal(uiTex.profile, 'textured_quad_24b');
  assert.equal(uiTex.wgsl, profiles.textured_quad_24b.vertex);

  const fullscreen = router.resolve('0x000000000000f003', 'fragment', {
    bindingSignature: { cbvCount: 1 },
  });
  assert.equal(fullscreen.profile, 'fullscreen_triangle_cbv');
  assert.equal(fullscreen.wgsl, profiles.fullscreen_triangle_cbv.fragment);

  const computeNoop = router.resolve('0x000000000000f004', 'compute', {
    bindingSignature: { hasCbv: true, hasTexture: true },
  });
  assert.equal(computeNoop.profile, 'default');
  assert.equal(computeNoop.wgsl, BUILTIN_FALLBACK_WGSL.compute);

  // Explicit options.profile takes precedence over bindingSignature.
  const explicitOverride = router.resolve('0x000000000000f005', 'vertex', {
    profile: 'fullscreen_cbv',
    bindingSignature: { hasTexture: true },
  });
  assert.equal(explicitOverride.profile, 'fullscreen_triangle_cbv');
});

test('W268 resetStats clears lookup stats and negative compile-failure cache while preserving packLoader and realShaders', () => {
  const realHash = '0x000000000000aa99';
  const badHash = '0x000000000000bb99';
  const router = new ShaderFallbackRouter({
    packLoader: createGetSourcePackLoader({ [realHash]: PACK_VS_WGSL }),
    now: () => 700,
  });
  router.registerRealShader(realHash, 'vertex', { wgsl: PACK_VS_WGSL });
  router.recordMissingShader(badHash, 'fragment', 'compile_failed');
  assert.equal(router.hasCompileFailure(badHash), true);
  router.resolve(badHash, 'fragment');

  let stats = router.getStats();
  assert.equal(stats.compileFailureEntries, 1);
  assert.equal(stats.compileFailureRecorded, 1);
  assert.equal(stats.compileFailureCacheHits, 2);
  assert.equal(stats.totalLookups, 1);

  router.resetStats();
  stats = router.getStats();
  assert.deepEqual(stats, {
    totalLookups: 0,
    packHits: 0,
    fallbackHits: 0,
    builtinHits: 0,
    uniqueMissing: 0,
    overflowMissingCount: 0,
    overflowMissingHashes: 0,
    hitRatePct: 0,
    hitRatio: 0,
    compileFailureEntries: 0,
    compileFailureCacheHits: 0,
    compileFailureRecorded: 0,
  });
  assert.equal(router.hasCompileFailure(badHash), false);
  assert.equal(router.hasRealShader(realHash, 'vertex'), true);
});

test('W268 classic Worker importScripts isolation exposes negative compile-failure APIs with zero global leaks', () => {
  const source = readFileSync(ROUTER_PATH, 'utf8');
  const sandbox = { console, performance };
  const vmCtx = vm.createContext(sandbox);
  vmCtx.self = vmCtx;

  vm.runInContext(source, vmCtx, { filename: ROUTER_PATH });
  vm.runInContext(source, vmCtx, { filename: ROUTER_PATH });

  assert.doesNotThrow(() => {
    vm.runInContext(
      `
      const DEFAULT_MAX_COMPILE_FAILURE_ENTRIES = 0;
      const SAFE_REASON_TOKEN_REGEX = /x/;
      `,
      vmCtx
    );
  });

  const api = vmCtx.PlayGta6ShaderFallbackRouter;
  assert.equal(api.DEFAULT_MAX_COMPILE_FAILURE_ENTRIES, 2048);
  assert.equal(typeof api.sanitizeCompileFailureReason, 'function');
  assert.equal(typeof api.selectDeterministicFallbackProfile, 'function');
  const r = new api.ShaderFallbackRouter();
  assert.equal(typeof r.hasCompileFailure, 'function');
  assert.equal(typeof r.getCompileFailure, 'function');
  assert.equal(typeof r.recordCompileFailure, 'function');
  assert.equal(typeof r.clearCompileFailure, 'function');
  assert.equal(typeof r.clearCompileFailures, 'function');
  assert.equal(typeof r.selectFallbackProfile, 'function');
  assert.equal(typeof r.resetStats, 'function');
});
