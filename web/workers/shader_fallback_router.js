'use strict';

// WebGPU missing-shader fallback router + Frame-1 telemetry (Track 4/3 bridge S12).
// Zero external dependencies; UMD export for Web Workers (globalThis.PlayGta6ShaderFallbackRouter)
// and Node.js (module.exports). Hash / stage conventions match web/workers/wgsl_pack_loader.js,
// web/workers/reflection_loader.js, and tools/gen_fallback_shaders.py.
(function (root) {
  /** WGPK / convert.rs numeric stage codes (Vertex=1, Fragment=2, Compute=3). */
  const SHADER_STAGE = Object.freeze({
    VERTEX: 1,
    FRAGMENT: 2,
    COMPUTE: 3,
  });

  const STAGE_NAMES = Object.freeze({
    [SHADER_STAGE.VERTEX]: 'vertex',
    [SHADER_STAGE.FRAGMENT]: 'fragment',
    [SHADER_STAGE.COMPUTE]: 'compute',
  });

  const STAGE_NAME_TO_CODE = Object.freeze({
    vertex: SHADER_STAGE.VERTEX,
    vs: SHADER_STAGE.VERTEX,
    fragment: SHADER_STAGE.FRAGMENT,
    fs: SHADER_STAGE.FRAGMENT,
    pixel: SHADER_STAGE.FRAGMENT,
    ps: SHADER_STAGE.FRAGMENT,
    compute: SHADER_STAGE.COMPUTE,
    cs: SHADER_STAGE.COMPUTE,
  });

  const DEFAULT_ENTRY_POINTS = Object.freeze({
    vertex: 'vs_main',
    fragment: 'fs_main',
    compute: 'cs_main',
  });

  const U64_MAX = 0xffffffffffffffffn;
  const DEFAULT_MAX_TRACKED_MISSING = 512;
  const DEFAULT_MAX_TRACKED_MISSING_HASHES = 2048;
  const DEFAULT_TOP_MISSING_LIMIT = 16;

  /**
   * Minimal original WebGPU WGSL fallbacks used when no pack (and no S9 fallback hash)
   * is available yet. Never derived from copyrighted game shaders.
   */
  const BUILTIN_FALLBACK_WGSL = Object.freeze({
    vertex: [
      '@vertex',
      'fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {',
      '    var positions = array<vec2<f32>, 3>(',
      '        vec2<f32>(-1.0, -1.0),',
      '        vec2<f32>(3.0, -1.0),',
      '        vec2<f32>(-1.0, 3.0),',
      '    );',
      '    let p = positions[vertex_index];',
      '    return vec4<f32>(p, 0.0, 1.0);',
      '}',
      '',
    ].join('\n'),
    fragment: [
      '@fragment',
      'fn fs_main() -> @location(0) vec4<f32> {',
      '    return vec4<f32>(0.05, 0.07, 0.11, 1.0);',
      '}',
      '',
    ].join('\n'),
    compute: [
      '@compute @workgroup_size(1)',
      'fn cs_main() {}',
      '',
    ].join('\n'),
  });

  /**
   * Frame-1 D3D12 fullscreen-triangle + CBV fallback WGSL pair.
   * Matches `DrawInstanced(3, 1)` with a 1-CBV `group(0) binding(0)` descriptor table
   * (e.g. 256-byte `array<vec4<f32>, 16>` constant buffer) onto an RGBA8 render target.
   * Works with or without bound vertex attributes via `@builtin(vertex_index) vid: u32`,
   * uses bitcast-masked CBV modulation so output is never NaN and always `alpha = 1.0`.
   */
  const FULLSCREEN_TRIANGLE_CBV_WGSL = Object.freeze({
    vertex: [
      'struct FrameCbv {',
      '    data: array<vec4<f32>, 16>,',
      '};',
      '',
      '@group(0) @binding(0) var<uniform> frame_cbv: FrameCbv;',
      '',
      'struct VsOut {',
      '    @builtin(position) position: vec4<f32>,',
      '    @location(0) uv: vec2<f32>,',
      '};',
      '',
      '@vertex',
      'fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {',
      '    var positions = array<vec2<f32>, 3>(',
      '        vec2<f32>(-1.0, -1.0),',
      '        vec2<f32>(3.0, -1.0),',
      '        vec2<f32>(-1.0, 3.0),',
      '    );',
      '    var uvs = array<vec2<f32>, 3>(',
      '        vec2<f32>(0.0, 1.0),',
      '        vec2<f32>(2.0, 1.0),',
      '        vec2<f32>(0.0, -1.0),',
      '    );',
      '    let idx = vid % 3u;',
      '    let cbv_keep = f32(bitcast<u32>(frame_cbv.data[0].x) & 0xffu) * 0.0;',
      '    var out: VsOut;',
      '    out.position = vec4<f32>(positions[idx], cbv_keep, 1.0);',
      '    out.uv = uvs[idx];',
      '    return out;',
      '}',
      '',
    ].join('\n'),
    fragment: [
      'struct FrameCbv {',
      '    data: array<vec4<f32>, 16>,',
      '};',
      '',
      '@group(0) @binding(0) var<uniform> frame_cbv: FrameCbv;',
      '',
      'struct FsIn {',
      '    @location(0) uv: vec2<f32>,',
      '};',
      '',
      '@fragment',
      'fn fs_main(input: FsIn) -> @location(0) vec4<f32> {',
      '    let uv = clamp(input.uv, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));',
      '    let b0 = bitcast<u32>(frame_cbv.data[0].x);',
      '    let b1 = bitcast<u32>(frame_cbv.data[0].y);',
      '    let b2 = bitcast<u32>(frame_cbv.data[0].z) ^ bitcast<u32>(frame_cbv.data[0].w);',
      '    let cbv_mod = vec3<f32>(',
      '        f32(b0 & 0xffu) / 255.0,',
      '        f32(b1 & 0xffu) / 255.0,',
      '        f32(b2 & 0xffu) / 255.0,',
      '    );',
      '    let base = vec3<f32>(',
      '        0.08 + 0.35 * uv.x,',
      '        0.14 + 0.40 * uv.y,',
      '        0.24 + 0.30 * (1.0 - uv.x),',
      '    );',
      '    let color = clamp(',
      '        mix(base, cbv_mod, 0.25),',
      '        vec3<f32>(0.0, 0.0, 0.0),',
      '        vec3<f32>(1.0, 1.0, 1.0),',
      '    );',
      '    return vec4<f32>(color, 1.0);',
      '}',
      '',
    ].join('\n'),
    compute: BUILTIN_FALLBACK_WGSL.compute,
  });

  /**
   * 24-byte-stride textured quad fallback WGSL pair for splash / UI draws.
   * Vertex attributes: `@location(0) pos: vec4<f32>` (16 B) + `@location(1) uv: vec2<f32>` (8 B).
   * Bindings: `@group(0) @binding(0)` texture_2d<f32> + `@group(0) @binding(1)` sampler.
   */
  const TEXTURED_QUAD_24B_WGSL = Object.freeze({
    vertex: [
      'struct VsOut {',
      '    @builtin(position) position: vec4<f32>,',
      '    @location(0) uv: vec2<f32>,',
      '};',
      '',
      '@vertex',
      'fn vs_main(',
      '    @location(0) pos: vec4<f32>,',
      '    @location(1) uv: vec2<f32>,',
      ') -> VsOut {',
      '    var out: VsOut;',
      '    let w = select(pos.w, 1.0, pos.w == 0.0);',
      '    out.position = vec4<f32>(pos.xyz, w);',
      '    out.uv = uv;',
      '    return out;',
      '}',
      '',
    ].join('\n'),
    fragment: [
      '@group(0) @binding(0) var quad_tex: texture_2d<f32>;',
      '@group(0) @binding(1) var quad_sampler: sampler;',
      '',
      '@fragment',
      'fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {',
      '    let texel = textureSample(quad_tex, quad_sampler, uv);',
      '    return texel;',
      '}',
      '',
    ].join('\n'),
    compute: BUILTIN_FALLBACK_WGSL.compute,
  });

  /**
   * Real Frame-1 24-byte splash pass fallback WGSL pair for expanded Xbox RECTLIST quads.
   * Vertex attributes (24 B stride):
   *   `@location(0) pos: vec3<f32>` (float32x3 @ offset 0)
   *   `@location(1) color: vec4<f32>` (unorm8x4 @ offset 12)
   *   `@location(2) uv: vec2<f32>` (float32x2 @ offset 16)
   * Bindings:
   *   `@group(0) @binding(0)` uniform FrameCbv (`cb0`)
   *   `@group(0) @binding(1)` texture_2d<f32> (`tex0`)
   *   `@group(0) @binding(2)` sampler (`samp0`)
   */
  const SPLASH_RECT_24B_WGSL = Object.freeze({
    vertex: [
      'struct FrameCbv {',
      '    data: array<vec4<f32>, 16>,',
      '};',
      '',
      '@group(0) @binding(0) var<uniform> cb0: FrameCbv;',
      '',
      'struct VsOut {',
      '    @builtin(position) position: vec4<f32>,',
      '    @location(0) color: vec4<f32>,',
      '    @location(1) uv: vec2<f32>,',
      '};',
      '',
      '@vertex',
      'fn vs_main(',
      '    @location(0) pos: vec3<f32>,',
      '    @location(1) color: vec4<f32>,',
      '    @location(2) uv: vec2<f32>,',
      ') -> VsOut {',
      '    let cbv_keep = f32(bitcast<u32>(cb0.data[0].x) & 0xffu) * 0.0;',
      '    var out: VsOut;',
      '    out.position = vec4<f32>(pos.xy, pos.z + cbv_keep, 1.0);',
      '    out.color = color;',
      '    out.uv = uv;',
      '    return out;',
      '}',
      '',
    ].join('\n'),
    fragment: [
      'struct FrameCbv {',
      '    data: array<vec4<f32>, 16>,',
      '};',
      '',
      '@group(0) @binding(0) var<uniform> cb0: FrameCbv;',
      '@group(0) @binding(1) var tex0: texture_2d<f32>;',
      '@group(0) @binding(2) var samp0: sampler;',
      '',
      'struct FsIn {',
      '    @location(0) color: vec4<f32>,',
      '    @location(1) uv: vec2<f32>,',
      '};',
      '',
      '@fragment',
      'fn fs_main(input: FsIn) -> @location(0) vec4<f32> {',
      '    let texel = textureSample(tex0, samp0, input.uv);',
      '    let cbv_keep = f32(bitcast<u32>(cb0.data[0].x) & 0xffu) * 0.0;',
      '    let tinted = texel * input.color;',
      '    return clamp(',
      '        vec4<f32>(tinted.rgb + vec3<f32>(cbv_keep, cbv_keep, cbv_keep), tinted.a),',
      '        vec4<f32>(0.0, 0.0, 0.0, 0.0),',
      '        vec4<f32>(1.0, 1.0, 1.0, 1.0),',
      '    );',
      '}',
      '',
    ].join('\n'),
    compute: BUILTIN_FALLBACK_WGSL.compute,
  });

  const BUILTIN_FALLBACK_PROFILES = Object.freeze({
    default: BUILTIN_FALLBACK_WGSL,
    fullscreen_triangle_cbv: FULLSCREEN_TRIANGLE_CBV_WGSL,
    textured_quad_24b: TEXTURED_QUAD_24B_WGSL,
    splash_rect_24b: SPLASH_RECT_24B_WGSL,
  });

  const FALLBACK_PROFILE_ALIASES = Object.freeze({
    default: 'default',
    minimal: 'default',
    fullscreen_triangle_cbv: 'fullscreen_triangle_cbv',
    fullscreen_cbv: 'fullscreen_triangle_cbv',
    'fullscreen-triangle-cbv': 'fullscreen_triangle_cbv',
    'fullscreen-cbv': 'fullscreen_triangle_cbv',
    textured_quad_24b: 'textured_quad_24b',
    textured_24b: 'textured_quad_24b',
    'textured-quad-24b': 'textured_quad_24b',
    'textured-24b': 'textured_quad_24b',
    splash_rect_24b: 'splash_rect_24b',
    'splash-rect-24b': 'splash_rect_24b',
    rectlist_24b: 'splash_rect_24b',
    'rectlist-24b': 'splash_rect_24b',
    frame1_splash_24b: 'splash_rect_24b',
    'frame1-splash-24b': 'splash_rect_24b',
  });

  /**
   * Normalize a built-in fallback profile name or alias to a canonical profile key.
   * Unknown or omitted values safely resolve to `'default'`.
   *
   * @param {string | null | undefined} profile
   * @returns {'default' | 'fullscreen_triangle_cbv' | 'textured_quad_24b' | 'splash_rect_24b'}
   */
  function normalizeFallbackProfile(profile) {
    if (typeof profile !== 'string') {
      return 'default';
    }
    const key = profile.trim().toLowerCase();
    if (key.length === 0) {
      return 'default';
    }
    if (Object.prototype.hasOwnProperty.call(FALLBACK_PROFILE_ALIASES, key)) {
      return FALLBACK_PROFILE_ALIASES[key];
    }
    return 'default';
  }

  /**
   * Return a sanitized, JSON-serializable map of available built-in fallback WGSL profiles.
   * Each call returns fresh copies so caller mutations cannot alter router templates.
   *
   * @returns {Record<string, { vertex: string, fragment: string, compute: string }>}
   */
  function getBuiltInFallbackProfiles() {
    const out = {};
    const names = Object.keys(BUILTIN_FALLBACK_PROFILES);
    for (let i = 0; i < names.length; i += 1) {
      const name = names[i];
      const entry = BUILTIN_FALLBACK_PROFILES[name];
      out[name] = {
        vertex: entry.vertex,
        fragment: entry.fragment,
        compute: entry.compute,
      };
    }
    return out;
  }

  function defaultNow() {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
      return performance.now();
    }
    return Date.now();
  }

  /**
   * Normalize a 64-bit shader hash (`BigInt`, `0x...` hex, bare hex, or safe non-negative
   * integer) into an unsigned 64-bit `BigInt` matching `normalizeHash` / `normalizeShaderHash`.
   *
   * @param {bigint | string | number} hash
   * @returns {bigint}
   */
  function normalizeShaderHash(hash) {
    if (typeof hash === 'bigint') {
      if (hash < 0n || hash > U64_MAX) {
        throw new RangeError(`Shader hash out of u64 range: ${hash.toString()}`);
      }
      return hash;
    }

    if (typeof hash === 'number') {
      if (!Number.isInteger(hash) || !Number.isSafeInteger(hash) || hash < 0) {
        throw new RangeError(`Numeric shader hash must be a non-negative safe integer, got: ${hash}`);
      }
      return BigInt(hash);
    }

    if (typeof hash === 'string') {
      const trimmed = hash.trim();
      const rawHex =
        trimmed.startsWith('0x') || trimmed.startsWith('0X') ? trimmed.slice(2) : trimmed;
      if (!/^[0-9a-fA-F]{1,16}$/.test(rawHex)) {
        throw new RangeError(`Invalid hex shader hash: ${JSON.stringify(hash)}`);
      }
      return BigInt(`0x${rawHex}`);
    }

    throw new TypeError('Shader hash must be a BigInt, hex string, or non-negative safe integer');
  }

  /**
   * Format a 64-bit shader hash as canonical `0x` + 16 lowercase hex digits.
   *
   * @param {bigint | string | number} hash
   * @returns {string}
   */
  function formatShaderHash(hash) {
    const value = normalizeShaderHash(hash);
    return `0x${value.toString(16).padStart(16, '0')}`;
  }

  /**
   * Normalize a shader stage code or name to canonical `"vertex" | "fragment" | "compute"`.
   *
   * @param {number | string} stage
   * @returns {'vertex' | 'fragment' | 'compute'}
   */
  function normalizeStage(stage) {
    if (typeof stage === 'number') {
      const name = STAGE_NAMES[stage];
      if (name !== undefined) {
        return name;
      }
      throw new RangeError(`Invalid shader stage code: ${stage}`);
    }
    if (typeof stage === 'string') {
      const lower = stage.trim().toLowerCase();
      if (Object.prototype.hasOwnProperty.call(STAGE_NAME_TO_CODE, lower)) {
        return STAGE_NAMES[STAGE_NAME_TO_CODE[lower]];
      }
      throw new RangeError(`Invalid shader stage name: ${JSON.stringify(stage)}`);
    }
    throw new TypeError('Shader stage must be a stage code (1..3) or stage name');
  }

  function extractEntryPoint(wgsl, stageName) {
    if (typeof wgsl !== 'string' || wgsl.length === 0) {
      return DEFAULT_ENTRY_POINTS[stageName] || null;
    }
    const re = new RegExp(`@${stageName}\\b[\\s\\S]*?\\bfn\\s+([A-Za-z_][A-Za-z0-9_]*)`);
    const match = re.exec(wgsl);
    if (match) {
      return match[1];
    }
    const fallback = /\bfn\s+([A-Za-z_][A-Za-z0-9_]*)/.exec(wgsl);
    return fallback ? fallback[1] : DEFAULT_ENTRY_POINTS[stageName] || null;
  }

  function unwrapPackResult(raw) {
    if (raw === undefined || raw === null) {
      return undefined;
    }
    if (typeof raw === 'string') {
      return { wgsl: raw, entryPoint: null };
    }
    if (typeof raw === 'object') {
      if (typeof raw.wgsl === 'string') {
        return {
          wgsl: raw.wgsl,
          entryPoint:
            typeof raw.entryPoint === 'string'
              ? raw.entryPoint
              : typeof raw.entry_point === 'string'
                ? raw.entry_point
                : null,
        };
      }
      if (typeof raw.source === 'string') {
        return { wgsl: raw.source, entryPoint: null };
      }
      if (typeof raw.code === 'string') {
        return { wgsl: raw.code, entryPoint: null };
      }
    }
    return undefined;
  }

  /**
   * Look up WGSL from a pack loader supporting getWgsl / getSource / lookup / Map.get
   * (plus getShaderSource for WgslPackCache compatibility).
   *
   * @param {object | Map | null | undefined} packLoader
   * @param {bigint} hashBigInt
   * @returns {{ wgsl: string, entryPoint: string | null } | undefined}
   */
  function lookupPackSource(packLoader, hashBigInt) {
    if (!packLoader) {
      return undefined;
    }

    const hashHex = formatShaderHash(hashBigInt);
    const keys = [hashBigInt, hashHex];

    const tryCall = (fn, key) => {
      if (typeof fn !== 'function') {
        return undefined;
      }
      try {
        return unwrapPackResult(fn.call(packLoader, key));
      } catch (_err) {
        return undefined;
      }
    };

    for (let i = 0; i < keys.length; i += 1) {
      const key = keys[i];
      const hit =
        tryCall(packLoader.getWgsl, key) ||
        tryCall(packLoader.getSource, key) ||
        tryCall(packLoader.getShaderSource, key) ||
        tryCall(packLoader.lookup, key) ||
        tryCall(packLoader.get, key);
      if (hit) {
        return hit;
      }
    }
    return undefined;
  }

  function resolveFallbackHashForStage(fallbackHashes, stageName) {
    if (!fallbackHashes || typeof fallbackHashes !== 'object') {
      return null;
    }
    const raw =
      fallbackHashes[stageName] !== undefined
        ? fallbackHashes[stageName]
        : fallbackHashes[STAGE_NAME_TO_CODE[stageName]];
    if (raw === undefined || raw === null || raw === '') {
      return null;
    }
    return normalizeShaderHash(raw);
  }

  /**
   * Missing-shader fallback router used by the WebGPU worker during early M3/M4 frames.
   */
  class ShaderFallbackRouter {
    /**
     * @param {{
     *   packLoader?: object | Map | null,
     *   fallbackHashes?: Record<string, bigint | string | number> | null,
     *   maxTrackedMissingHashes?: number,
     *   maxTrackedMissing?: number,
     *   defaultProfile?: string | null,
     *   now?: () => number,
     * }} [options]
     */
    constructor(options = {}) {
      if (options !== undefined && options !== null && typeof options !== 'object') {
        throw new TypeError('ShaderFallbackRouter options must be an object');
      }
      const opts = options || {};

      this._packLoader = opts.packLoader !== undefined ? opts.packLoader : null;
      this._fallbackHashes =
        opts.fallbackHashes && typeof opts.fallbackHashes === 'object'
          ? { ...opts.fallbackHashes }
          : {};

      let maxTracked = DEFAULT_MAX_TRACKED_MISSING_HASHES;
      if (opts.maxTrackedMissingHashes !== undefined) {
        if (
          !Number.isInteger(opts.maxTrackedMissingHashes) ||
          opts.maxTrackedMissingHashes < 0
        ) {
          throw new RangeError(
            `maxTrackedMissingHashes must be a non-negative integer, got ${opts.maxTrackedMissingHashes}`
          );
        }
        maxTracked = opts.maxTrackedMissingHashes;
      } else if (opts.maxTrackedMissing !== undefined) {
        if (!Number.isInteger(opts.maxTrackedMissing) || opts.maxTrackedMissing < 0) {
          throw new RangeError(
            `maxTrackedMissing must be a non-negative integer, got ${opts.maxTrackedMissing}`
          );
        }
        maxTracked = opts.maxTrackedMissing;
      }
      this._maxTrackedMissing = maxTracked;

      this._now = typeof opts.now === 'function' ? opts.now : defaultNow;
      this._defaultProfile = normalizeFallbackProfile(opts.defaultProfile);

      this._missing = new Map();
      this._overflowMissingCount = 0;
      this._stats = {
        totalLookups: 0,
        packHits: 0,
        fallbackHits: 0,
        builtinHits: 0,
      };
    }

    /**
     * Return a sanitized map of available built-in fallback WGSL profiles.
     * @returns {Record<string, { vertex: string, fragment: string, compute: string }>}
     */
    getBuiltInFallbackProfiles() {
      return getBuiltInFallbackProfiles();
    }

    /**
     * Replace the pack loader used for exact / stage-fallback lookups.
     * @param {object | Map | null | undefined} packLoader
     */
    setPackLoader(packLoader) {
      this._packLoader = packLoader !== undefined ? packLoader : null;
    }

    /**
     * Replace per-stage S9 fallback hashes (`vertex` / `fragment` / `compute`).
     * @param {Record<string, bigint | string | number> | null | undefined} fallbackHashes
     */
    setFallbackHashes(fallbackHashes) {
      this._fallbackHashes =
        fallbackHashes && typeof fallbackHashes === 'object' ? { ...fallbackHashes } : {};
    }

    /**
     * Resolve a requested shader hash for a stage:
     * 1) exact pack hit, 2) stage-matched S9 fallback hash in pack, 3) built-in WGSL.
     *
     * @param {bigint | string | number} hash
     * @param {number | string} stage
     * @param {{ profile?: string | null }} [options]
     * @returns {{
     *   wgsl: string,
     *   source: 'pack' | 'fallback' | 'builtin',
     *   requestedHash: string,
     *   resolvedHash: string | null,
     *   stage: 'vertex' | 'fragment' | 'compute',
     *   entryPoint: string | null,
     *   profile?: 'default' | 'fullscreen_triangle_cbv' | 'textured_quad_24b' | 'splash_rect_24b',
     * }}
     */
    resolve(hash, stage, options) {
      const stageName = normalizeStage(stage);
      const requested = normalizeShaderHash(hash);
      const requestedHex = formatShaderHash(requested);
      const nowMs = this._now();
      const profileName =
        options &&
        typeof options === 'object' &&
        options.profile !== undefined &&
        options.profile !== null
          ? normalizeFallbackProfile(options.profile)
          : this._defaultProfile;

      this._stats.totalLookups += 1;

      const exact = lookupPackSource(this._packLoader, requested);
      if (exact) {
        this._stats.packHits += 1;
        return {
          wgsl: exact.wgsl,
          source: 'pack',
          requestedHash: requestedHex,
          resolvedHash: requestedHex,
          stage: stageName,
          entryPoint: exact.entryPoint || extractEntryPoint(exact.wgsl, stageName),
        };
      }

      // Exact miss — record Frame-1 missing-shader telemetry.
      this._trackMissing(requestedHex, stageName, nowMs);

      const fallbackHash = resolveFallbackHashForStage(this._fallbackHashes, stageName);
      if (fallbackHash !== null) {
        const fallbackHit = lookupPackSource(this._packLoader, fallbackHash);
        if (fallbackHit) {
          this._stats.fallbackHits += 1;
          const resolvedHex = formatShaderHash(fallbackHash);
          return {
            wgsl: fallbackHit.wgsl,
            source: 'fallback',
            requestedHash: requestedHex,
            resolvedHash: resolvedHex,
            stage: stageName,
            entryPoint: fallbackHit.entryPoint || extractEntryPoint(fallbackHit.wgsl, stageName),
          };
        }
      }

      const profileTemplates = BUILTIN_FALLBACK_PROFILES[profileName] || BUILTIN_FALLBACK_WGSL;
      const builtin = profileTemplates[stageName] || BUILTIN_FALLBACK_WGSL[stageName];
      this._stats.builtinHits += 1;
      return {
        wgsl: builtin,
        source: 'builtin',
        requestedHash: requestedHex,
        resolvedHash: null,
        stage: stageName,
        entryPoint: extractEntryPoint(builtin, stageName),
        profile: profileName,
      };
    }

    _trackMissing(hashHex, stageName, nowMs) {
      const key = `${hashHex}|${stageName}`;
      const existing = this._missing.get(key);
      if (existing) {
        existing.hitCount += 1;
        existing.lastSeenMs = nowMs;
        return;
      }

      if (this._missing.size >= this._maxTrackedMissing) {
        this._overflowMissingCount += 1;
        return;
      }

      this._missing.set(key, {
        hash: hashHex,
        stage: stageName,
        hitCount: 1,
        firstSeenMs: nowMs,
        lastSeenMs: nowMs,
      });
    }

    /**
     * Return top missing shaders sorted by hit count descending (JSON-serializable).
     *
     * @param {{ limit?: number }} [options]
     * @returns {Array<{ hash: string, stage: string, hitCount: number, firstSeenMs: number, lastSeenMs: number }>}
     */
    getMissingReport(options = {}) {
      const opts = options && typeof options === 'object' ? options : {};
      let limit = opts.limit;
      if (limit === undefined || limit === null) {
        limit = this._missing.size;
      }
      if (!Number.isInteger(limit) || limit < 0) {
        throw new RangeError(`limit must be a non-negative integer, got ${limit}`);
      }

      const rows = Array.from(this._missing.values()).map((row) => ({
        hash: row.hash,
        stage: row.stage,
        hitCount: row.hitCount,
        firstSeenMs: row.firstSeenMs,
        lastSeenMs: row.lastSeenMs,
      }));

      rows.sort(
        (a, b) =>
          b.hitCount - a.hitCount ||
          a.hash.localeCompare(b.hash) ||
          a.stage.localeCompare(b.stage)
      );

      return rows.slice(0, limit);
    }

    /**
     * Return top-N missing shaders ranked by miss count descending, breaking ties
     * deterministically by `hashHex` ascending (then `stage` ascending).
     * Records are sanitized (`{ hashHex, stage, count }`) with zero host paths or
     * raw shader text.
     *
     * @param {number} [limit=16]
     * @returns {Array<{ hashHex: string, stage: 'vertex' | 'fragment' | 'compute', count: number }>}
     */
    getTopMissingShaders(limit = DEFAULT_TOP_MISSING_LIMIT) {
      const resolvedLimit =
        limit === undefined || limit === null ? DEFAULT_TOP_MISSING_LIMIT : limit;
      if (!Number.isInteger(resolvedLimit) || resolvedLimit < 0) {
        throw new RangeError(`limit must be a non-negative integer, got ${resolvedLimit}`);
      }
      if (resolvedLimit === 0 || this._missing.size === 0) {
        return [];
      }

      const rows = Array.from(this._missing.values(), (row) => ({
        hashHex: row.hash,
        stage: row.stage,
        count: row.hitCount,
      }));

      rows.sort(
        (a, b) =>
          b.count - a.count ||
          (a.hashHex < b.hashHex ? -1 : a.hashHex > b.hashHex ? 1 : 0) ||
          (a.stage < b.stage ? -1 : a.stage > b.stage ? 1 : 0)
      );

      return rows.slice(0, resolvedLimit);
    }

    /**
     * Aggregate lookup / missing-shader telemetry.
     */
    getStats() {
      const uniqueMissing = this._missing.size;
      const total = this._stats.totalLookups;
      const hitRatio = total > 0 ? this._stats.packHits / total : 0;
      const hitRatePct =
        total > 0 ? Math.round(hitRatio * 10000) / 100 : 0;
      return {
        totalLookups: this._stats.totalLookups,
        packHits: this._stats.packHits,
        fallbackHits: this._stats.fallbackHits,
        builtinHits: this._stats.builtinHits,
        uniqueMissing,
        overflowMissingCount: this._overflowMissingCount,
        overflowMissingHashes: this._overflowMissingCount,
        hitRatePct,
        hitRatio,
      };
    }

    /**
     * Clear missing-shader telemetry and lookup counters.
     */
    reset() {
      this._missing.clear();
      this._overflowMissingCount = 0;
      this._stats.totalLookups = 0;
      this._stats.packHits = 0;
      this._stats.fallbackHits = 0;
      this._stats.builtinHits = 0;
    }
  }

  ShaderFallbackRouter.getBuiltInFallbackProfiles = getBuiltInFallbackProfiles;

  const exportedApi = Object.freeze({
    SHADER_STAGE,
    STAGE_NAMES,
    BUILTIN_FALLBACK_WGSL,
    BUILTIN_FALLBACK_PROFILES,
    DEFAULT_ENTRY_POINTS,
    DEFAULT_MAX_TRACKED_MISSING,
    DEFAULT_MAX_TRACKED_MISSING_HASHES,
    DEFAULT_TOP_MISSING_LIMIT,
    normalizeShaderHash,
    formatShaderHash,
    normalizeStage,
    normalizeFallbackProfile,
    getBuiltInFallbackProfiles,
    ShaderFallbackRouter,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6ShaderFallbackRouter = exportedApi;
  } else if (root) {
    root.PlayGta6ShaderFallbackRouter = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
