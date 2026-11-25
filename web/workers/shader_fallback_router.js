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
  const DEFAULT_MAX_CATEGORIZED_MISSING_HASHES = 8192;
  const DEFAULT_TOP_MISSING_LIMIT = 16;
  const DEFAULT_MISSING_SUMMARY_TOP_LIMIT = 32;

  /**
   * Canonical missing-shader failure categories for live Chrome triage telemetry (W179).
   */
  const MISSING_SHADER_REASONS = Object.freeze({
    NOT_IN_PACK: 'not_in_pack',
    REFLECTION_FAILED: 'reflection_failed',
    COMPILE_FAILED: 'compile_failed',
    OTHER: 'other',
  });

  const MISSING_SHADER_REASON_ORDER = Object.freeze([
    MISSING_SHADER_REASONS.NOT_IN_PACK,
    MISSING_SHADER_REASONS.REFLECTION_FAILED,
    MISSING_SHADER_REASONS.COMPILE_FAILED,
    MISSING_SHADER_REASONS.OTHER,
  ]);

  const MISSING_SHADER_REASON_ALIASES = Object.freeze({
    not_in_pack: MISSING_SHADER_REASONS.NOT_IN_PACK,
    missing: MISSING_SHADER_REASONS.NOT_IN_PACK,
    missing_from_pack: MISSING_SHADER_REASONS.NOT_IN_PACK,
    pack_miss: MISSING_SHADER_REASONS.NOT_IN_PACK,
    reflection_failed: MISSING_SHADER_REASONS.REFLECTION_FAILED,
    reflection_error: MISSING_SHADER_REASONS.REFLECTION_FAILED,
    reflection_miss: MISSING_SHADER_REASONS.REFLECTION_FAILED,
    no_reflection: MISSING_SHADER_REASONS.REFLECTION_FAILED,
    compile_failed: MISSING_SHADER_REASONS.COMPILE_FAILED,
    compilation_failed: MISSING_SHADER_REASONS.COMPILE_FAILED,
    compile_error: MISSING_SHADER_REASONS.COMPILE_FAILED,
    wgsl_compile_failed: MISSING_SHADER_REASONS.COMPILE_FAILED,
    shader_compile_failed: MISSING_SHADER_REASONS.COMPILE_FAILED,
    other: MISSING_SHADER_REASONS.OTHER,
  });

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

  /**
   * Normalize a missing-shader reason string to one of:
   * `'not_in_pack' | 'reflection_failed' | 'compile_failed' | 'other'`.
   *
   * @param {string | null | undefined} reason
   * @returns {'not_in_pack' | 'reflection_failed' | 'compile_failed' | 'other'}
   */
  function normalizeMissingReason(reason) {
    if (typeof reason !== 'string') {
      return MISSING_SHADER_REASONS.OTHER;
    }
    const key = reason
      .trim()
      .toLowerCase()
      .replace(/[\s-]+/g, '_');
    if (key.length === 0) {
      return MISSING_SHADER_REASONS.OTHER;
    }
    if (Object.prototype.hasOwnProperty.call(MISSING_SHADER_REASON_ALIASES, key)) {
      return MISSING_SHADER_REASON_ALIASES[key];
    }
    return MISSING_SHADER_REASONS.OTHER;
  }

  function compareLex(a, b) {
    return a < b ? -1 : a > b ? 1 : 0;
  }

  function createEmptyReasonCounts() {
    return {
      not_in_pack: 0,
      reflection_failed: 0,
      compile_failed: 0,
      other: 0,
    };
  }

  function createEmptyStageCounts() {
    return {
      vertex: 0,
      fragment: 0,
      compute: 0,
    };
  }

  function getDominantMissingReason(row) {
    let bestReason = row.lastReason || MISSING_SHADER_REASONS.OTHER;
    let bestCount = row.reasons[bestReason] || 0;
    for (let i = 0; i < MISSING_SHADER_REASON_ORDER.length; i += 1) {
      const candidate = MISSING_SHADER_REASON_ORDER[i];
      const count = row.reasons[candidate] || 0;
      if (count > bestCount) {
        bestCount = count;
        bestReason = candidate;
      }
    }
    return bestReason;
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
      return raw.trim().length > 0 ? { wgsl: raw, entryPoint: null } : undefined;
    }
    if (typeof raw === 'object') {
      if (typeof raw.wgsl === 'string' && raw.wgsl.trim().length > 0) {
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
      if (typeof raw.source === 'string' && raw.source.trim().length > 0) {
        return { wgsl: raw.source, entryPoint: null };
      }
      if (typeof raw.code === 'string' && raw.code.trim().length > 0) {
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
     *   maxCategorizedMissingHashes?: number,
     *   defaultProfile?: string | null,
     *   realShaders?: Iterable<any> | Map<any, any> | Record<string, any> | null,
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
      let hasExplicitLegacyCap = false;
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
        hasExplicitLegacyCap = true;
      } else if (opts.maxTrackedMissing !== undefined) {
        if (!Number.isInteger(opts.maxTrackedMissing) || opts.maxTrackedMissing < 0) {
          throw new RangeError(
            `maxTrackedMissing must be a non-negative integer, got ${opts.maxTrackedMissing}`
          );
        }
        maxTracked = opts.maxTrackedMissing;
        hasExplicitLegacyCap = true;
      }
      this._maxTrackedMissing = maxTracked;

      let maxCategorized = hasExplicitLegacyCap
        ? maxTracked
        : DEFAULT_MAX_CATEGORIZED_MISSING_HASHES;
      if (opts.maxCategorizedMissingHashes !== undefined) {
        if (
          !Number.isInteger(opts.maxCategorizedMissingHashes) ||
          opts.maxCategorizedMissingHashes < 0
        ) {
          throw new RangeError(
            `maxCategorizedMissingHashes must be a non-negative integer, got ${opts.maxCategorizedMissingHashes}`
          );
        }
        maxCategorized = opts.maxCategorizedMissingHashes;
      }
      this._maxCategorizedMissing = maxCategorized;

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

      // W179: Categorized missing-shader telemetry (bounded unique hash+stage entries).
      this._categorizedMissing = new Map();
      this._categorizedOverflowCount = 0;
      this._categorizedTotals = {
        total: 0,
        byReason: createEmptyReasonCounts(),
        byStage: createEmptyStageCounts(),
      };

      // W179: Real-shader override guard (protects real pack, menu, and Bink2 YUV shaders).
      this._realShaders = new Map();
      this._blockedOverrides = new Map();
      this._blockedOverrideCount = 0;
      this._blockedOverflowCount = 0;

      if (opts.realShaders !== undefined && opts.realShaders !== null) {
        this._registerInitialRealShaders(opts.realShaders);
      }
    }

    _registerInitialRealShaders(realShaders) {
      if (realShaders instanceof Map) {
        for (const [hash, meta] of realShaders.entries()) {
          if (meta && typeof meta === 'object' && meta.stage !== undefined) {
            this.registerRealShader(hash, meta.stage, meta);
          } else {
            this.registerRealShader(hash, '*', meta);
          }
        }
        return;
      }
      if (realShaders instanceof Set || Array.isArray(realShaders)) {
        for (const entry of realShaders) {
          if (
            typeof entry === 'string' ||
            typeof entry === 'bigint' ||
            typeof entry === 'number'
          ) {
            this.registerRealShader(entry, '*');
          } else if (entry && typeof entry === 'object' && entry.hash !== undefined) {
            this.registerRealShader(
              entry.hash,
              entry.stage !== undefined ? entry.stage : '*',
              entry
            );
          }
        }
        return;
      }
      if (typeof realShaders === 'object') {
        const keys = Object.keys(realShaders);
        for (let i = 0; i < keys.length; i += 1) {
          const hash = keys[i];
          const meta = realShaders[hash];
          if (meta && typeof meta === 'object' && meta.stage !== undefined) {
            this.registerRealShader(hash, meta.stage, meta);
          } else {
            this.registerRealShader(hash, '*', meta);
          }
        }
      }
    }

    _isConfiguredFallbackHash(hashBigInt) {
      const stages = ['vertex', 'fragment', 'compute'];
      for (let i = 0; i < stages.length; i += 1) {
        try {
          const fb = resolveFallbackHashForStage(this._fallbackHashes, stages[i]);
          if (fb !== null && fb === hashBigInt) {
            return true;
          }
        } catch (_err) {
          // Ignore malformed fallback hash entry during predicate check.
        }
      }
      return false;
    }

    _lookupRealShader(hashHex, stageName) {
      const exact = this._realShaders.get(`${hashHex}|${stageName}`);
      if (exact) {
        return exact;
      }
      return this._realShaders.get(`${hashHex}|*`) || null;
    }

    _recordBlockedOverride(hashHex, stageName, reason, nowMs) {
      this._blockedOverrideCount += 1;
      const key = `${hashHex}|${stageName}`;
      const existing = this._blockedOverrides.get(key);
      if (existing) {
        existing.count += 1;
        existing.lastReason = reason;
        existing.lastSeenMs = nowMs;
        return;
      }
      if (this._blockedOverrides.size >= this._maxCategorizedMissing) {
        this._blockedOverflowCount += 1;
        return;
      }
      this._blockedOverrides.set(key, {
        hashHex,
        stage: stageName,
        count: 1,
        lastReason: reason,
        firstSeenMs: nowMs,
        lastSeenMs: nowMs,
      });
    }

    /**
     * Register a real shader module (e.g. pack shader, menu shader, or Bink2 YUV shader)
     * so fallback resolution can never silently overwrite it with an opaque fallback.
     *
     * @param {bigint | string | number} hash
     * @param {number | string | object | null} [stageOrMeta]
     * @param {string | {
     *   wgsl?: string | null,
     *   code?: string | null,
     *   entryPoint?: string | null,
     *   entry_point?: string | null,
     *   module?: object | null,
     *   source?: 'pack' | 'real',
     *   label?: string | null,
     *   stage?: number | string | null,
     * } | null} [maybeMeta]
     * @returns {{ hashHex: string, stage: string, hasWgsl: boolean, hasModule: boolean }}
     */
    registerRealShader(hash, stageOrMeta, maybeMeta) {
      const hashHex = formatShaderHash(normalizeShaderHash(hash));
      let stageArg = stageOrMeta;
      let meta = maybeMeta;

      if (
        maybeMeta === undefined &&
        stageOrMeta !== null &&
        typeof stageOrMeta === 'object'
      ) {
        meta = stageOrMeta;
        stageArg = stageOrMeta.stage !== undefined ? stageOrMeta.stage : '*';
      }

      const stageKey =
        stageArg === undefined || stageArg === null || stageArg === '*'
          ? '*'
          : normalizeStage(stageArg);

      let wgsl = null;
      let entryPoint = null;
      let moduleRef = null;
      let source = 'real';
      let label = null;

      if (typeof meta === 'string') {
        if (meta.trim().length > 0) {
          wgsl = meta;
        }
      } else if (meta && typeof meta === 'object') {
        if (typeof meta.wgsl === 'string' && meta.wgsl.trim().length > 0) {
          wgsl = meta.wgsl;
        } else if (typeof meta.code === 'string' && meta.code.trim().length > 0) {
          wgsl = meta.code;
        }
        if (typeof meta.entryPoint === 'string' && meta.entryPoint.length > 0) {
          entryPoint = meta.entryPoint;
        } else if (typeof meta.entry_point === 'string' && meta.entry_point.length > 0) {
          entryPoint = meta.entry_point;
        }
        if (meta.module !== undefined && meta.module !== null) {
          moduleRef = meta.module;
        }
        if (meta.source === 'pack' || meta.source === 'real') {
          source = meta.source;
        }
        if (typeof meta.label === 'string' && meta.label.length > 0) {
          label = meta.label;
        }
      }

      const record = {
        hashHex,
        stage: stageKey,
        wgsl,
        entryPoint,
        module: moduleRef,
        source,
        label,
        registeredAtMs: this._now(),
      };
      this._realShaders.set(`${hashHex}|${stageKey}`, record);

      return {
        hashHex,
        stage: stageKey,
        hasWgsl: wgsl !== null,
        hasModule: moduleRef !== null,
      };
    }

    /**
     * Unregister a previously registered real shader for `(hash, stage)`.
     * When `stage` is omitted, removes all stage entries registered for `hash`.
     *
     * @param {bigint | string | number} hash
     * @param {number | string | null} [stage]
     * @returns {boolean}
     */
    unregisterRealShader(hash, stage) {
      const hashHex = formatShaderHash(normalizeShaderHash(hash));
      if (stage === undefined || stage === null) {
        let removed = false;
        const prefix = `${hashHex}|`;
        for (const key of Array.from(this._realShaders.keys())) {
          if (key.startsWith(prefix)) {
            this._realShaders.delete(key);
            removed = true;
          }
        }
        return removed;
      }
      const stageKey = stage === '*' ? '*' : normalizeStage(stage);
      return this._realShaders.delete(`${hashHex}|${stageKey}`);
    }

    /**
     * Clear all explicitly registered real shaders.
     */
    clearRealShaders() {
      this._realShaders.clear();
    }

    /**
     * Return true if a real shader module exists for `(hash, stage)` either in the
     * explicit real-shader registry or as a non-fallback entry in the active pack loader.
     *
     * @param {bigint | string | number} hash
     * @param {number | string | null} [stage]
     * @returns {boolean}
     */
    hasRealShader(hash, stage) {
      const hashBigInt = normalizeShaderHash(hash);
      const hashHex = formatShaderHash(hashBigInt);

      if (stage === undefined || stage === null || stage === '*') {
        if (this._realShaders.has(`${hashHex}|*`)) {
          return true;
        }
        const prefix = `${hashHex}|`;
        for (const key of this._realShaders.keys()) {
          if (key.startsWith(prefix)) {
            return true;
          }
        }
      } else {
        const stageName = normalizeStage(stage);
        if (this._lookupRealShader(hashHex, stageName) !== null) {
          return true;
        }
      }

      if (!this._isConfiguredFallbackHash(hashBigInt)) {
        const packHit = lookupPackSource(this._packLoader, hashBigInt);
        if (packHit) {
          return true;
        }
      }
      return false;
    }

    /**
     * Guard predicate returning `false` whenever a real shader exists for `(hash, stage)`
     * or is supplied in `context`, ensuring callers never silently overwrite real menu/video
     * shaders with fallback shaders.
     *
     * @param {bigint | string | number} hash
     * @param {number | string | null} [stage]
     * @param {object | null} [context]
     * @returns {boolean}
     */
    canFallbackOverride(hash, stage, context) {
      if (this.hasRealShader(hash, stage)) {
        return false;
      }
      if (context && typeof context === 'object') {
        const existingShader =
          context.existingShader && typeof context.existingShader === 'object'
            ? context.existingShader
            : null;
        const src =
          context.existingSource ||
          context.source ||
          (existingShader && existingShader.source) ||
          null;
        if (
          context.isRealShader === true ||
          src === 'pack' ||
          src === 'real' ||
          (existingShader && existingShader.isRealShader === true)
        ) {
          return false;
        }
      }
      return true;
    }

    /**
     * Return JSON-safe real-shader override guard telemetry (zero raw WGSL or host paths).
     *
     * @param {{ limit?: number }} [options]
     * @returns {{
     *   registeredRealShaders: number,
     *   blockedOverrideCount: number,
     *   overflowBlockedCount: number,
     *   blockedShaders: Array<{
     *     hashHex: string,
     *     stage: string,
     *     count: number,
     *     lastReason: string,
     *     firstSeenMs: number,
     *     lastSeenMs: number,
     *   }>
     * }}
     */
    getRealShaderGuardStats(options = {}) {
      const opts = options && typeof options === 'object' ? options : {};
      let limit = opts.limit;
      if (limit === undefined || limit === null) {
        limit = this._blockedOverrides.size;
      }
      if (!Number.isInteger(limit) || limit < 0) {
        throw new RangeError(`limit must be a non-negative integer, got ${limit}`);
      }

      const blockedShaders = Array.from(this._blockedOverrides.values(), (row) => ({
        hashHex: row.hashHex,
        stage: row.stage,
        count: row.count,
        lastReason: row.lastReason,
        firstSeenMs: row.firstSeenMs,
        lastSeenMs: row.lastSeenMs,
      }));

      blockedShaders.sort(
        (a, b) =>
          b.count - a.count ||
          compareLex(a.hashHex, b.hashHex) ||
          compareLex(a.stage, b.stage)
      );

      return {
        registeredRealShaders: this._realShaders.size,
        blockedOverrideCount: this._blockedOverrideCount,
        overflowBlockedCount: this._blockedOverflowCount,
        blockedShaders: blockedShaders.slice(0, limit),
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
     * 1) exact pack hit (never silently overwritten by fallbacks),
     * 2) registered / caller-supplied real shader guard (never silently overwritten by fallbacks),
     * 3) stage-matched S9 fallback hash in pack,
     * 4) built-in WGSL.
     *
     * @param {bigint | string | number} hash
     * @param {number | string} stage
     * @param {{
     *   profile?: string | null,
     *   forceFallback?: boolean,
     *   isRealShader?: boolean,
     *   existingSource?: string | null,
     *   existingShader?: {
     *     wgsl?: string | null,
     *     code?: string | null,
     *     entryPoint?: string | null,
     *     module?: object | null,
     *     source?: string | null,
     *     isRealShader?: boolean,
     *   } | null,
     *   reason?: string | null,
     *   missingReason?: string | null,
     * }} [options]
     * @returns {{
     *   wgsl: string | null,
     *   source: 'pack' | 'real' | 'fallback' | 'builtin',
     *   requestedHash: string,
     *   resolvedHash: string | null,
     *   stage: 'vertex' | 'fragment' | 'compute',
     *   entryPoint: string | null,
     *   profile?: 'default' | 'fullscreen_triangle_cbv' | 'textured_quad_24b' | 'splash_rect_24b',
     *   fallbackBlocked?: boolean,
     *   guardReason?: string,
     *   module?: object,
     *   label?: string,
     * }}
     */
    resolve(hash, stage, options) {
      const stageName = normalizeStage(stage);
      const requested = normalizeShaderHash(hash);
      const requestedHex = formatShaderHash(requested);
      const nowMs = this._now();
      const opts = options && typeof options === 'object' ? options : null;
      const profileName =
        opts && opts.profile !== undefined && opts.profile !== null
          ? normalizeFallbackProfile(opts.profile)
          : this._defaultProfile;

      this._stats.totalLookups += 1;

      const exact = lookupPackSource(this._packLoader, requested);
      if (exact) {
        this._stats.packHits += 1;
        const result = {
          wgsl: exact.wgsl,
          source: 'pack',
          requestedHash: requestedHex,
          resolvedHash: requestedHex,
          stage: stageName,
          entryPoint: exact.entryPoint || extractEntryPoint(exact.wgsl, stageName),
        };
        if (opts && opts.forceFallback === true && !this._isConfiguredFallbackHash(requested)) {
          this._recordBlockedOverride(requestedHex, stageName, 'pack_shader_protected', nowMs);
          result.fallbackBlocked = true;
          result.guardReason = 'pack_shader_protected';
        }
        return result;
      }

      // W179: Real-shader override guard — never silently replace a registered real shader
      // (e.g. menu or Bink2 YUV shader) with an S9 or built-in opaque fallback.
      const registeredReal = this._lookupRealShader(requestedHex, stageName);
      if (registeredReal) {
        this._recordBlockedOverride(requestedHex, stageName, 'real_shader_registered', nowMs);
        if (registeredReal.source === 'pack') {
          this._stats.packHits += 1;
        }
        const guarded = {
          wgsl: registeredReal.wgsl,
          source: registeredReal.source,
          requestedHash: requestedHex,
          resolvedHash: requestedHex,
          stage: stageName,
          entryPoint:
            registeredReal.entryPoint ||
            (registeredReal.wgsl
              ? extractEntryPoint(registeredReal.wgsl, stageName)
              : DEFAULT_ENTRY_POINTS[stageName] || null),
          fallbackBlocked: true,
          guardReason: 'real_shader_registered',
        };
        if (registeredReal.module !== null) {
          guarded.module = registeredReal.module;
        }
        if (registeredReal.label !== null) {
          guarded.label = registeredReal.label;
        }
        return guarded;
      }

      // W179: Caller-indicated existing real shader guard.
      if (opts) {
        const existingShader =
          opts.existingShader && typeof opts.existingShader === 'object'
            ? opts.existingShader
            : null;
        const existingSource =
          opts.existingSource || (existingShader && existingShader.source) || null;
        const callerClaimsReal =
          opts.isRealShader === true ||
          existingSource === 'pack' ||
          existingSource === 'real' ||
          (existingShader !== null && existingShader.isRealShader === true);

        if (callerClaimsReal) {
          this._recordBlockedOverride(requestedHex, stageName, 'existing_real_shader', nowMs);
          const existingWgsl =
            existingShader && typeof existingShader.wgsl === 'string' && existingShader.wgsl.trim().length > 0
              ? existingShader.wgsl
              : existingShader && typeof existingShader.code === 'string' && existingShader.code.trim().length > 0
                ? existingShader.code
                : null;
          const resolvedSource = existingSource === 'pack' ? 'pack' : 'real';
          if (resolvedSource === 'pack') {
            this._stats.packHits += 1;
          }
          const guarded = {
            wgsl: existingWgsl,
            source: resolvedSource,
            requestedHash: requestedHex,
            resolvedHash: requestedHex,
            stage: stageName,
            entryPoint:
              (existingShader && typeof existingShader.entryPoint === 'string' && existingShader.entryPoint) ||
              (existingWgsl
                ? extractEntryPoint(existingWgsl, stageName)
                : DEFAULT_ENTRY_POINTS[stageName] || null),
            fallbackBlocked: true,
            guardReason: 'existing_real_shader',
          };
          if (existingShader && existingShader.module !== undefined && existingShader.module !== null) {
            guarded.module = existingShader.module;
          }
          return guarded;
        }
      }

      // Exact miss — record Frame-1 & categorized missing-shader telemetry.
      this._trackMissing(requestedHex, stageName, nowMs);
      const missReason =
        opts && (opts.reason !== undefined || opts.missingReason !== undefined)
          ? normalizeMissingReason(opts.reason !== undefined ? opts.reason : opts.missingReason)
          : MISSING_SHADER_REASONS.NOT_IN_PACK;
      this._recordCategorizedMissing(requestedHex, stageName, missReason, nowMs);

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

    _recordCategorizedMissing(hashHex, stageName, reason, nowMs) {
      this._categorizedTotals.total += 1;
      this._categorizedTotals.byReason[reason] += 1;
      this._categorizedTotals.byStage[stageName] += 1;

      const key = `${hashHex}|${stageName}`;
      const existing = this._categorizedMissing.get(key);
      if (existing) {
        existing.hitCount += 1;
        existing.reasons[reason] += 1;
        existing.lastReason = reason;
        existing.lastSeenMs = nowMs;
        return {
          hashHex,
          stage: stageName,
          reason,
          hitCount: existing.hitCount,
          tracked: true,
          overflowed: false,
        };
      }

      if (this._categorizedMissing.size >= this._maxCategorizedMissing) {
        this._categorizedOverflowCount += 1;
        return {
          hashHex,
          stage: stageName,
          reason,
          hitCount: 0,
          tracked: false,
          overflowed: true,
        };
      }

      const reasons = createEmptyReasonCounts();
      reasons[reason] = 1;
      this._categorizedMissing.set(key, {
        hashHex,
        stage: stageName,
        hitCount: 1,
        reasons,
        lastReason: reason,
        firstSeenMs: nowMs,
        lastSeenMs: nowMs,
      });

      return {
        hashHex,
        stage: stageName,
        reason,
        hitCount: 1,
        tracked: true,
        overflowed: false,
      };
    }

    /**
     * Record a categorized missing or failed shader event for live Chrome triage.
     * Supports `'not_in_pack'`, `'reflection_failed'`, `'compile_failed'`, and `'other'`.
     * Bounded to `maxCategorizedMissingHashes` (default 8192) unique `(hashHex, stage)` entries.
     *
     * @param {bigint | string | number} hashHex
     * @param {number | string} stage
     * @param {string} [reason='other']
     * @returns {{
     *   hashHex: string,
     *   stage: 'vertex' | 'fragment' | 'compute',
     *   reason: 'not_in_pack' | 'reflection_failed' | 'compile_failed' | 'other',
     *   hitCount: number,
     *   tracked: boolean,
     *   overflowed: boolean,
     * }}
     */
    recordMissingShader(hashHex, stage, reason = MISSING_SHADER_REASONS.OTHER) {
      const canonicalHash = formatShaderHash(normalizeShaderHash(hashHex));
      const stageName = normalizeStage(stage);
      const canonicalReason = normalizeMissingReason(reason);
      return this._recordCategorizedMissing(
        canonicalHash,
        stageName,
        canonicalReason,
        this._now()
      );
    }

    /**
     * Export a JSON-safe summary of categorized missing shaders:
     * total & unique counts per category/stage, overflow counter, real-shader guard stats,
     * and top hit-count hashes sorted descending (deterministic hashHex/stage tie-breaking).
     *
     * @param {number | { limit?: number | null, reason?: string | null, stage?: number | string | null } | null} [options]
     * @returns {{
     *   totalMissingEvents: number,
     *   uniqueMissingHashes: number,
     *   maxTrackedHashes: number,
     *   overflowCount: number,
     *   countsByReason: { not_in_pack: number, reflection_failed: number, compile_failed: number, other: number },
     *   countsByStage: { vertex: number, fragment: number, compute: number },
     *   uniqueCountsByReason: { not_in_pack: number, reflection_failed: number, compile_failed: number, other: number },
     *   uniqueCountsByStage: { vertex: number, fragment: number, compute: number },
     *   realShaderGuard: {
     *     registeredRealShaders: number,
     *     blockedOverrideCount: number,
     *     overflowBlockedCount: number,
     *   },
     *   topMissing: Array<{
     *     hashHex: string,
     *     stage: 'vertex' | 'fragment' | 'compute',
     *     reason: 'not_in_pack' | 'reflection_failed' | 'compile_failed' | 'other',
     *     lastReason: 'not_in_pack' | 'reflection_failed' | 'compile_failed' | 'other',
     *     hitCount: number,
     *     reasons: { not_in_pack: number, reflection_failed: number, compile_failed: number, other: number },
     *     firstSeenMs: number,
     *     lastSeenMs: number,
     *   }>
     * }}
     */
    getMissingShaderSummary(options) {
      let limit = DEFAULT_MISSING_SUMMARY_TOP_LIMIT;
      let filterReason = null;
      let filterStage = null;

      if (typeof options === 'number') {
        limit = options;
      } else if (options !== undefined && options !== null) {
        if (typeof options !== 'object') {
          throw new RangeError(`options must be an object or non-negative integer limit, got ${options}`);
        }
        if (options.limit !== undefined && options.limit !== null) {
          limit = options.limit;
        }
        if (options.reason !== undefined && options.reason !== null) {
          filterReason = normalizeMissingReason(options.reason);
        }
        if (options.stage !== undefined && options.stage !== null) {
          filterStage = normalizeStage(options.stage);
        }
      }

      if (!Number.isInteger(limit) || limit < 0) {
        throw new RangeError(`limit must be a non-negative integer, got ${limit}`);
      }

      const uniqueCountsByReason = createEmptyReasonCounts();
      const uniqueCountsByStage = createEmptyStageCounts();
      const filteredRows = [];

      for (const row of this._categorizedMissing.values()) {
        uniqueCountsByStage[row.stage] += 1;
        for (let i = 0; i < MISSING_SHADER_REASON_ORDER.length; i += 1) {
          const r = MISSING_SHADER_REASON_ORDER[i];
          if (row.reasons[r] > 0) {
            uniqueCountsByReason[r] += 1;
          }
        }

        if (filterStage !== null && row.stage !== filterStage) {
          continue;
        }
        if (filterReason !== null && !(row.reasons[filterReason] > 0)) {
          continue;
        }

        filteredRows.push({
          hashHex: row.hashHex,
          stage: row.stage,
          reason: filterReason !== null ? filterReason : getDominantMissingReason(row),
          lastReason: row.lastReason,
          hitCount: row.hitCount,
          reasons: {
            not_in_pack: row.reasons.not_in_pack,
            reflection_failed: row.reasons.reflection_failed,
            compile_failed: row.reasons.compile_failed,
            other: row.reasons.other,
          },
          firstSeenMs: row.firstSeenMs,
          lastSeenMs: row.lastSeenMs,
        });
      }

      filteredRows.sort((a, b) => {
        if (filterReason !== null) {
          const diffReason = b.reasons[filterReason] - a.reasons[filterReason];
          if (diffReason !== 0) {
            return diffReason;
          }
        }
        return (
          b.hitCount - a.hitCount ||
          compareLex(a.hashHex, b.hashHex) ||
          compareLex(a.stage, b.stage)
        );
      });

      return {
        totalMissingEvents: this._categorizedTotals.total,
        uniqueMissingHashes: this._categorizedMissing.size,
        maxTrackedHashes: this._maxCategorizedMissing,
        overflowCount: this._categorizedOverflowCount,
        countsByReason: {
          not_in_pack: this._categorizedTotals.byReason.not_in_pack,
          reflection_failed: this._categorizedTotals.byReason.reflection_failed,
          compile_failed: this._categorizedTotals.byReason.compile_failed,
          other: this._categorizedTotals.byReason.other,
        },
        countsByStage: {
          vertex: this._categorizedTotals.byStage.vertex,
          fragment: this._categorizedTotals.byStage.fragment,
          compute: this._categorizedTotals.byStage.compute,
        },
        uniqueCountsByReason,
        uniqueCountsByStage,
        realShaderGuard: {
          registeredRealShaders: this._realShaders.size,
          blockedOverrideCount: this._blockedOverrideCount,
          overflowBlockedCount: this._blockedOverflowCount,
        },
        topMissing: limit === 0 ? [] : filteredRows.slice(0, limit),
      };
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
     * Clear missing-shader telemetry, categorized summary, guard counters, and lookup counters.
     * Preserves configured `packLoader`, `fallbackHashes`, and registered real shaders.
     */
    reset() {
      this._missing.clear();
      this._overflowMissingCount = 0;
      this._stats.totalLookups = 0;
      this._stats.packHits = 0;
      this._stats.fallbackHits = 0;
      this._stats.builtinHits = 0;

      this._categorizedMissing.clear();
      this._categorizedOverflowCount = 0;
      this._categorizedTotals.total = 0;
      this._categorizedTotals.byReason = createEmptyReasonCounts();
      this._categorizedTotals.byStage = createEmptyStageCounts();

      this._blockedOverrides.clear();
      this._blockedOverrideCount = 0;
      this._blockedOverflowCount = 0;
    }
  }

  ShaderFallbackRouter.getBuiltInFallbackProfiles = getBuiltInFallbackProfiles;

  const exportedApi = Object.freeze({
    SHADER_STAGE,
    STAGE_NAMES,
    MISSING_SHADER_REASONS,
    BUILTIN_FALLBACK_WGSL,
    BUILTIN_FALLBACK_PROFILES,
    DEFAULT_ENTRY_POINTS,
    DEFAULT_MAX_TRACKED_MISSING,
    DEFAULT_MAX_TRACKED_MISSING_HASHES,
    DEFAULT_MAX_CATEGORIZED_MISSING_HASHES,
    DEFAULT_TOP_MISSING_LIMIT,
    DEFAULT_MISSING_SUMMARY_TOP_LIMIT,
    normalizeShaderHash,
    formatShaderHash,
    normalizeStage,
    normalizeMissingReason,
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
