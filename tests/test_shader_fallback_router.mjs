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
  BUILTIN_FALLBACK_WGSL,
  BUILTIN_FALLBACK_PROFILES,
  DEFAULT_ENTRY_POINTS,
  DEFAULT_MAX_TRACKED_MISSING,
  normalizeShaderHash,
  formatShaderHash,
  normalizeStage,
  normalizeFallbackProfile,
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
  assert.equal(stats.hitRatePct, 50);

  router.reset();
  stats = router.getStats();
  assert.deepEqual(stats, {
    totalLookups: 0,
    packHits: 0,
    fallbackHits: 0,
    builtinHits: 0,
    uniqueMissing: 0,
    overflowMissingCount: 0,
    hitRatePct: 0,
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
