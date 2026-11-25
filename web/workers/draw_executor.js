'use strict';

// WebGPU draw executor with Frame-1 DrawInstanced(3, 1) / 24B-stride / CBV fallback profile
// routing (Track 3/4 task G38). Zero external dependencies; UMD export for Web Workers
// (globalThis.PlayGta6DrawExecutor) and Node.js (module.exports).
//
// Bridges missing console-native shaders to G34 fallback WGSL profiles in
// web/workers/shader_fallback_router.js (`fullscreen_triangle_cbv`, `textured_quad_24b`, `default`)
// while hardening bind-group and vertex-buffer bindings against unbound, destroyed, misaligned,
// or undersized Frame-1 bring-up resources.

(function (root) {
  const FALLBACK_PROFILE = Object.freeze({
    DEFAULT: 'default',
    FULLSCREEN_TRIANGLE_CBV: 'fullscreen_triangle_cbv',
    TEXTURED_QUAD_24B: 'textured_quad_24b',
    SPLASH_RECT_24B: 'splash_rect_24b',
  });

  const FALLBACK_PROFILE_ALIASES = Object.freeze({
    default: FALLBACK_PROFILE.DEFAULT,
    minimal: FALLBACK_PROFILE.DEFAULT,
    fullscreen_triangle_cbv: FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV,
    fullscreen_cbv: FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV,
    'fullscreen-triangle-cbv': FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV,
    'fullscreen-cbv': FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV,
    textured_quad_24b: FALLBACK_PROFILE.TEXTURED_QUAD_24B,
    textured_24b: FALLBACK_PROFILE.TEXTURED_QUAD_24B,
    'textured-quad-24b': FALLBACK_PROFILE.TEXTURED_QUAD_24B,
    'textured-24b': FALLBACK_PROFILE.TEXTURED_QUAD_24B,
    splash_rect_24b: FALLBACK_PROFILE.SPLASH_RECT_24B,
    'splash-rect-24b': FALLBACK_PROFILE.SPLASH_RECT_24B,
    rectlist_24b: FALLBACK_PROFILE.SPLASH_RECT_24B,
    'rectlist-24b': FALLBACK_PROFILE.SPLASH_RECT_24B,
    frame1_splash_24b: FALLBACK_PROFILE.SPLASH_RECT_24B,
    'frame1-splash-24b': FALLBACK_PROFILE.SPLASH_RECT_24B,
  });

  const TEXTURED_QUAD_VERTEX_STRIDE = 24;
  const FRAME1_CBV_MIN_BYTES = 256;
  const UNIFORM_OFFSET_ALIGNMENT = 256;
  const VERTEX_OFFSET_ALIGNMENT = 4;
  const DEFAULT_COLOR_FORMAT = 'bgra8unorm';
  const DEFAULT_TOPOLOGY = 'triangle-list';
  const DEFAULT_MAX_FALLBACK_PIPELINES = 128;

  const FALLBACK_VERTEX_LAYOUT_24B = Object.freeze({
    arrayStride: TEXTURED_QUAD_VERTEX_STRIDE,
    stepMode: 'vertex',
    attributes: Object.freeze([
      Object.freeze({ shaderLocation: 0, offset: 0, format: 'float32x4' }),
      Object.freeze({ shaderLocation: 1, offset: 16, format: 'float32x2' }),
    ]),
  });

  /**
   * Exact real Frame-1 24-byte splash pass vertex layout (G37 / G47):
   *   POSITION: R32G32B32_FLOAT (float32x3 @ 0, @location(0))
   *   COLOR:    R8G8B8A8_UNORM  (unorm8x4  @ 12, @location(1))
   *   TEXCOORD: R32G32_FLOAT    (float32x2 @ 16, @location(2))
   */
  const FRAME1_SPLASH_VERTEX_LAYOUT_24B = Object.freeze({
    arrayStride: TEXTURED_QUAD_VERTEX_STRIDE,
    stepMode: 'vertex',
    attributes: Object.freeze([
      Object.freeze({ shaderLocation: 0, offset: 0, format: 'float32x3' }),
      Object.freeze({ shaderLocation: 1, offset: 12, format: 'unorm8x4' }),
      Object.freeze({ shaderLocation: 2, offset: 16, format: 'float32x2' }),
    ]),
  });

  // WebGPU GPUBufferUsage / GPUTextureUsage bitmasks with headless/Node fallbacks.
  const BUFFER_USAGE = Object.freeze({
    COPY_DST:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.COPY_DST === 'number'
        ? GPUBufferUsage.COPY_DST
        : 0x0008,
    INDEX:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.INDEX === 'number'
        ? GPUBufferUsage.INDEX
        : 0x0010,
    VERTEX:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.VERTEX === 'number'
        ? GPUBufferUsage.VERTEX
        : 0x0020,
    UNIFORM:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.UNIFORM === 'number'
        ? GPUBufferUsage.UNIFORM
        : 0x0040,
    INDIRECT:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.INDIRECT === 'number'
        ? GPUBufferUsage.INDIRECT
        : 0x0100,
  });

  const TEXTURE_USAGE = Object.freeze({
    COPY_DST:
      typeof GPUTextureUsage !== 'undefined' && typeof GPUTextureUsage.COPY_DST === 'number'
        ? GPUTextureUsage.COPY_DST
        : 0x02,
    TEXTURE_BINDING:
      typeof GPUTextureUsage !== 'undefined' &&
      typeof GPUTextureUsage.TEXTURE_BINDING === 'number'
        ? GPUTextureUsage.TEXTURE_BINDING
        : 0x04,
  });

  // 64-bit FNV-1a constants matching web/workers/wgsl_pack_loader.js & wgpu_worker.js.
  const FNV1A_64_OFFSET_BASIS = 0xcbf29ce484222325n;
  const FNV1A_64_PRIME = 0x00000100000001b3n;
  const U64_MASK = 0xffffffffffffffffn;

  // Minimal safe built-in WGSL used only when no ShaderFallbackRouter is available at all.
  const EMERGENCY_BUILTIN_WGSL = Object.freeze({
    vertex: [
      '@vertex',
      'fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {',
      '    var positions = array<vec2<f32>, 3>(',
      '        vec2<f32>(-1.0, -1.0),',
      '        vec2<f32>(3.0, -1.0),',
      '        vec2<f32>(-1.0, 3.0),',
      '    );',
      '    let p = positions[vertex_index % 3u];',
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
  });

  function normalizeProfileName(profile) {
    if (typeof profile !== 'string') {
      return FALLBACK_PROFILE.DEFAULT;
    }
    const key = profile.trim().toLowerCase();
    if (key.length === 0) {
      return FALLBACK_PROFILE.DEFAULT;
    }
    if (Object.prototype.hasOwnProperty.call(FALLBACK_PROFILE_ALIASES, key)) {
      return FALLBACK_PROFILE_ALIASES[key];
    }
    return FALLBACK_PROFILE.DEFAULT;
  }

  function toNonNegativeInt(value, fallback) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
      return fallback;
    }
    return Math.trunc(value);
  }

  function fnv1a64Hex(text) {
    const str = typeof text === 'string' ? text : String(text || '');
    let hash = FNV1A_64_OFFSET_BASIS;
    for (let i = 0; i < str.length; i += 1) {
      hash ^= BigInt(str.charCodeAt(i) & 0xff);
      hash = (hash * FNV1A_64_PRIME) & U64_MASK;
    }
    return `0x${hash.toString(16).padStart(16, '0')}`;
  }

  function extractEntryPoint(wgsl, stageName) {
    const defaultName = stageName === 'fragment' ? 'fs_main' : 'vs_main';
    if (typeof wgsl !== 'string' || wgsl.length === 0) {
      return defaultName;
    }
    const re = new RegExp(`@${stageName}\\b[\\s\\S]*?\\bfn\\s+([A-Za-z_][A-Za-z0-9_]*)`);
    const match = re.exec(wgsl);
    if (match) {
      return match[1];
    }
    const fallback = /\bfn\s+([A-Za-z_][A-Za-z0-9_]*)/.exec(wgsl);
    return fallback ? fallback[1] : defaultName;
  }

  function iterateCollectionValues(collection) {
    if (!collection) {
      return [];
    }
    if (collection instanceof Map) {
      return Array.from(collection.values());
    }
    if (Array.isArray(collection)) {
      return collection;
    }
    if (typeof collection === 'object') {
      return Object.values(collection);
    }
    return [];
  }

  function hasAnyTruthyValue(collection) {
    const values = iterateCollectionValues(collection);
    for (let i = 0; i < values.length; i += 1) {
      const item = values[i];
      if (item !== null && item !== undefined && item !== false) {
        if (typeof item === 'object') {
          if (item.destroyed === true) continue;
          if ('buffer' in item && item.buffer === null && item.present !== true) continue;
          if ('texture' in item && item.texture === null && !item.textureView && item.present !== true) continue;
          if ('sampler' in item && item.sampler === null && item.present !== true) continue;
        }
        return true;
      }
    }
    return false;
  }

  function inspectBindGroupMeta(bindGroups) {
    let hasCbv = false;
    let hasTexture = false;
    let hasSampler = false;

    const groups = iterateCollectionValues(bindGroups);
    for (let i = 0; i < groups.length; i += 1) {
      const group = groups[i];
      if (!group || typeof group !== 'object') {
        continue;
      }
      if (group.hasCbv === true || group.hasBoundCbv === true || group.cbv || group.cbvBuffer || group.uniformBuffer) {
        hasCbv = true;
      }
      if (
        group.hasTexture === true ||
        group.hasBoundTexture === true ||
        group.hasSampledTexture === true ||
        group.hasSrv === true ||
        group.texture ||
        group.textureView ||
        group.srv
      ) {
        hasTexture = true;
      }
      if (group.hasSampler === true || group.hasBoundSampler === true || group.sampler) {
        hasSampler = true;
      }

      const entries = Array.isArray(group.entries)
        ? group.entries
        : group.desc && Array.isArray(group.desc.entries)
          ? group.desc.entries
          : null;
      if (!entries) {
        continue;
      }
      for (let j = 0; j < entries.length; j += 1) {
        const entry = entries[j];
        if (!entry || typeof entry !== 'object') {
          continue;
        }
        const type = typeof entry.type === 'string' ? entry.type.toLowerCase() : '';
        const kind = typeof entry.kind === 'string' ? entry.kind.toLowerCase() : '';
        const resourceType = entry.resourceType;
        const resource = entry.resource;
        const resKind =
          resource && typeof resource === 'object' && typeof resource.kind === 'string'
            ? resource.kind.toLowerCase()
            : '';

        if (
          type === 'uniform' ||
          type === 'cbv' ||
          kind === 'cbv' ||
          kind === 'uniform' ||
          entry.isCbv === true ||
          resourceType === 0 ||
          resourceType === 'buffer' ||
          (resource && typeof resource === 'object' && resource.buffer)
        ) {
          hasCbv = true;
        }
        if (
          type === 'texture' ||
          type === 'srv' ||
          type === 'sampled-texture' ||
          kind === 'srv' ||
          kind === 'texture' ||
          kind === 'gputextureview' ||
          resKind === 'gputextureview' ||
          resKind === 'gputexture' ||
          entry.isTexture === true ||
          resourceType === 1 ||
          resourceType === 'textureView' ||
          resourceType === 'texture' ||
          entry.textureView ||
          entry.texture
        ) {
          hasTexture = true;
        }
        if (
          type === 'sampler' ||
          kind === 'sampler' ||
          kind === 'gpusampler' ||
          resKind === 'gpusampler' ||
          entry.isSampler === true ||
          resourceType === 2 ||
          resourceType === 'sampler' ||
          entry.sampler
        ) {
          hasSampler = true;
        }
      }
    }

    return { hasCbv, hasTexture, hasSampler };
  }

  function hasVertexStride(drawState, targetStride) {
    if (
      drawState.vertexStride === targetStride ||
      drawState.arrayStride === targetStride ||
      drawState.vertexBufferStride === targetStride
    ) {
      return true;
    }

    const vbValues = iterateCollectionValues(drawState.vertexBuffers);
    for (let i = 0; i < vbValues.length; i += 1) {
      const vb = vbValues[i];
      if (!vb || typeof vb !== 'object') {
        continue;
      }
      if (vb.stride === targetStride || vb.arrayStride === targetStride) {
        return true;
      }
    }

    const layoutValues = iterateCollectionValues(
      drawState.vertexBufferLayouts || drawState.vertexLayouts
    );
    for (let i = 0; i < layoutValues.length; i += 1) {
      const layout = layoutValues[i];
      if (!layout || typeof layout !== 'object') {
        continue;
      }
      if (layout.arrayStride === targetStride || layout.stride === targetStride) {
        return true;
      }
    }

    return false;
  }

  function isRectListTopology(topology) {
    if (typeof topology !== 'string') {
      return false;
    }
    const norm = topology.trim().toLowerCase();
    return (
      norm === 'rect-list' ||
      norm === 'rectlist' ||
      norm === 'd3d_primitive_topology_rectlist'
    );
  }

  function isRectListDrawState(drawState) {
    if (!drawState || typeof drawState !== 'object') {
      return false;
    }
    if (drawState.isRectList === true || isRectListTopology(drawState.topology)) {
      return true;
    }
    const pipe =
      drawState.pipeline && typeof drawState.pipeline === 'object' ? drawState.pipeline : null;
    if (pipe && (pipe.isRectList === true || isRectListTopology(pipe.topology))) {
      return true;
    }
    return false;
  }

  /**
   * Expand Xbox `D3D_PRIMITIVE_TOPOLOGY_RECTLIST` 24-byte vertices (3 vertices per quad:
   * `v0, v1, v2`) into WebGPU `triangle-list` vertices (6 vertices per quad:
   * `[v0, v1, v2, v1, v3, v2]`).
   *
   * Frame-1 24-byte vertex layout:
   *   bytes  0..11: POSITION (`float32x3`, little-endian)
   *   bytes 12..15: COLOR    (`unorm8x4`)
   *   bytes 16..23: TEXCOORD (`float32x2`, little-endian)
   *
   * Implicit 4th corner `v3`:
   *   `pos3 = pos1 + pos2 - pos0`
   *   `color3 = color2` (bytes 12..15 copied verbatim)
   *   `uv3 = uv1 + uv2 - uv0`
   *
   * @param {Uint8Array | ArrayBufferView | ArrayBuffer} srcBytes
   * @param {number} [vertexCount]
   * @returns {Uint8Array}
   */
  function expandRectListVertices24B(srcBytes, vertexCount) {
    let bytes;
    if (srcBytes instanceof Uint8Array) {
      bytes = srcBytes;
    } else if (ArrayBuffer.isView(srcBytes)) {
      bytes = new Uint8Array(srcBytes.buffer, srcBytes.byteOffset, srcBytes.byteLength);
    } else if (srcBytes instanceof ArrayBuffer) {
      bytes = new Uint8Array(srcBytes);
    } else {
      return new Uint8Array(0);
    }

    const stride = TEXTURED_QUAD_VERTEX_STRIDE;
    const maxAvailableVerts = Math.floor(bytes.byteLength / stride);
    const effectiveVerts =
      vertexCount === undefined || vertexCount === null
        ? maxAvailableVerts
        : Math.min(toNonNegativeInt(vertexCount, 0), maxAvailableVerts);

    const rectCount = Math.floor(effectiveVerts / 3);
    if (rectCount <= 0) {
      return new Uint8Array(0);
    }

    const out = new Uint8Array(rectCount * 6 * stride);
    const srcView = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const dstView = new DataView(out.buffer, out.byteOffset, out.byteLength);

    for (let r = 0; r < rectCount; r += 1) {
      const v0Off = r * 3 * stride;
      const v1Off = v0Off + stride;
      const v2Off = v1Off + stride;
      const dstBase = r * 6 * stride;

      // Triangle 1: [v0, v1, v2]
      out.set(bytes.subarray(v0Off, v0Off + stride), dstBase);
      out.set(bytes.subarray(v1Off, v1Off + stride), dstBase + stride);
      out.set(bytes.subarray(v2Off, v2Off + stride), dstBase + 2 * stride);

      // Triangle 2: [v1, v3, v2]
      out.set(bytes.subarray(v1Off, v1Off + stride), dstBase + 3 * stride);

      const v3DstOff = dstBase + 4 * stride;
      const x3 =
        srcView.getFloat32(v1Off + 0, true) +
        srcView.getFloat32(v2Off + 0, true) -
        srcView.getFloat32(v0Off + 0, true);
      const y3 =
        srcView.getFloat32(v1Off + 4, true) +
        srcView.getFloat32(v2Off + 4, true) -
        srcView.getFloat32(v0Off + 4, true);
      const z3 =
        srcView.getFloat32(v1Off + 8, true) +
        srcView.getFloat32(v2Off + 8, true) -
        srcView.getFloat32(v0Off + 8, true);
      dstView.setFloat32(v3DstOff + 0, x3, true);
      dstView.setFloat32(v3DstOff + 4, y3, true);
      dstView.setFloat32(v3DstOff + 8, z3, true);

      // color3 = color2 (unorm8x4 bytes 12..15)
      out[v3DstOff + 12] = bytes[v2Off + 12];
      out[v3DstOff + 13] = bytes[v2Off + 13];
      out[v3DstOff + 14] = bytes[v2Off + 14];
      out[v3DstOff + 15] = bytes[v2Off + 15];

      const u3 =
        srcView.getFloat32(v1Off + 16, true) +
        srcView.getFloat32(v2Off + 16, true) -
        srcView.getFloat32(v0Off + 16, true);
      const v3 =
        srcView.getFloat32(v1Off + 20, true) +
        srcView.getFloat32(v2Off + 20, true) -
        srcView.getFloat32(v0Off + 20, true);
      dstView.setFloat32(v3DstOff + 16, u3, true);
      dstView.setFloat32(v3DstOff + 20, v3, true);

      out.set(bytes.subarray(v2Off, v2Off + stride), dstBase + 5 * stride);
    }

    return out;
  }

  /**
   * Pure helper that selects the appropriate G34/G47 built-in fallback WGSL profile from active
   * draw state when real shaders are missing or fallback-routed:
   * - Xbox `rect-list` topology (`topology === 'rect-list'` or `isRectList === true`) with
   *   24-byte-stride vertex layout -> `'splash_rect_24b'`
   * - `vertexCount === 3` with a bound CBV and no bound sampled texture (real title Frame-1
   *   `DrawInstanced(3, 1)` pattern) -> `'fullscreen_triangle_cbv'`
   * - 24-byte-stride vertex buffer with a bound texture/sampler -> `'textured_quad_24b'`
   * - Otherwise -> `'default'`
   *
   * @param {object | null | undefined} drawState
   * @returns {'default' | 'fullscreen_triangle_cbv' | 'textured_quad_24b' | 'splash_rect_24b'}
   */
  function selectFallbackShaderProfile(drawState) {
    if (!drawState || typeof drawState !== 'object') {
      return FALLBACK_PROFILE.DEFAULT;
    }

    const vertexCount = toNonNegativeInt(drawState.vertexCount, 0);
    const has24BStride = hasVertexStride(drawState, TEXTURED_QUAD_VERTEX_STRIDE);

    // 0. Xbox RECTLIST draw with Frame-1 24-byte vertex stride (POSITION + COLOR + TEXCOORD).
    if (isRectListDrawState(drawState) && has24BStride) {
      return FALLBACK_PROFILE.SPLASH_RECT_24B;
    }

    const bgInfo = inspectBindGroupMeta(drawState.bindGroups);

    const hasBoundCbv =
      drawState.hasBoundCbv === true ||
      drawState.hasCbv === true ||
      toNonNegativeInt(drawState.cbvCount, 0) > 0 ||
      Boolean(drawState.cbv || drawState.cbvBuffer || drawState.uniformBuffer) ||
      hasAnyTruthyValue(drawState.cbvs) ||
      hasAnyTruthyValue(drawState.uniformBuffers) ||
      bgInfo.hasCbv;

    const hasBoundTexture =
      drawState.hasBoundTexture === true ||
      drawState.hasTexture === true ||
      drawState.hasSampledTexture === true ||
      drawState.hasBoundSampledTexture === true ||
      drawState.hasBoundSrv === true ||
      toNonNegativeInt(drawState.textureCount, 0) > 0 ||
      toNonNegativeInt(drawState.srvCount, 0) > 0 ||
      Boolean(drawState.texture || drawState.textureView || drawState.srv) ||
      hasAnyTruthyValue(drawState.textures) ||
      hasAnyTruthyValue(drawState.textureViews) ||
      hasAnyTruthyValue(drawState.srvs) ||
      bgInfo.hasTexture;

    const hasBoundSampler =
      drawState.hasBoundSampler === true ||
      drawState.hasSampler === true ||
      toNonNegativeInt(drawState.samplerCount, 0) > 0 ||
      Boolean(drawState.sampler) ||
      hasAnyTruthyValue(drawState.samplers) ||
      bgInfo.hasSampler;

    // 1. Frame-1 DrawInstanced(3, 1) with bound CBV and no sampled texture.
    if (vertexCount === 3 && hasBoundCbv && !hasBoundTexture) {
      return FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV;
    }

    // 2. 24-byte-stride vertex buffer with bound texture or sampler.
    if (has24BStride && (hasBoundTexture || hasBoundSampler)) {
      return FALLBACK_PROFILE.TEXTURED_QUAD_24B;
    }

    return FALLBACK_PROFILE.DEFAULT;
  }

  function resolveShaderFallbackRouterCtor() {
    if (
      typeof globalThis !== 'undefined' &&
      globalThis.PlayGta6ShaderFallbackRouter &&
      typeof globalThis.PlayGta6ShaderFallbackRouter.ShaderFallbackRouter === 'function'
    ) {
      return globalThis.PlayGta6ShaderFallbackRouter.ShaderFallbackRouter;
    }
    if (typeof importScripts === 'function') {
      try {
        importScripts('./shader_fallback_router.js');
      } catch (_err) {
        // Optional helper unavailable in this worker context.
      }
      if (
        typeof globalThis !== 'undefined' &&
        globalThis.PlayGta6ShaderFallbackRouter &&
        typeof globalThis.PlayGta6ShaderFallbackRouter.ShaderFallbackRouter === 'function'
      ) {
        return globalThis.PlayGta6ShaderFallbackRouter.ShaderFallbackRouter;
      }
    }
    if (typeof require === 'function') {
      try {
        const mod = require('./shader_fallback_router.js');
        if (mod && typeof mod.ShaderFallbackRouter === 'function') {
          return mod.ShaderFallbackRouter;
        }
      } catch (_err) {
        // Optional helper unavailable in this Node context.
      }
    }
    return null;
  }

  function unwrapRouterResult(raw, stageName) {
    if (raw === null || raw === undefined) {
      return null;
    }
    if (typeof raw === 'string' && raw.length > 0) {
      return {
        wgsl: raw,
        source: 'builtin',
        entryPoint: extractEntryPoint(raw, stageName),
        resolvedHash: null,
        profile: null,
      };
    }
    if (typeof raw === 'object') {
      const wgsl =
        typeof raw.wgsl === 'string'
          ? raw.wgsl
          : typeof raw.code === 'string'
            ? raw.code
            : typeof raw.source === 'string' && raw.source.includes('fn ')
              ? raw.source
              : null;
      if (!wgsl || wgsl.length === 0) {
        return null;
      }
      const source =
        typeof raw.source === 'string' && !raw.source.includes('fn ') ? raw.source : 'builtin';
      const entryPoint =
        typeof raw.entryPoint === 'string' && raw.entryPoint.length > 0
          ? raw.entryPoint
          : extractEntryPoint(wgsl, stageName);
      const profile =
        typeof raw.profile === 'string' && raw.profile.length > 0
          ? normalizeProfileName(raw.profile)
          : null;
      return {
        wgsl,
        source,
        entryPoint,
        resolvedHash: raw.resolvedHash !== undefined ? raw.resolvedHash : null,
        profile,
      };
    }
    return null;
  }

  function inferEffectiveProfileFromWgsl(vsResult, fsResult) {
    const vsProfile = vsResult && vsResult.profile ? vsResult.profile : null;
    const fsProfile = fsResult && fsResult.profile ? fsResult.profile : null;
    if (vsProfile && fsProfile && vsProfile === fsProfile) {
      return vsProfile;
    }

    const vsCode = vsResult && typeof vsResult.wgsl === 'string' ? vsResult.wgsl : '';
    const fsCode = fsResult && typeof fsResult.wgsl === 'string' ? fsResult.wgsl : '';

    if (
      /@location\(0\)[\s\S]*?vec3<f32>[\s\S]*?@location\(1\)[\s\S]*?vec4<f32>[\s\S]*?@location\(2\)[\s\S]*?vec2<f32>/.test(
        vsCode
      ) &&
      /@group\(0\)\s+@binding\(0\)\s+var<uniform>/.test(fsCode) &&
      /@group\(0\)\s+@binding\(1\)[\s\S]*?texture_2d<f32>/.test(fsCode) &&
      /@group\(0\)\s+@binding\(2\)[\s\S]*?sampler/.test(fsCode)
    ) {
      return FALLBACK_PROFILE.SPLASH_RECT_24B;
    }
    if (
      /@group\(0\)\s+@binding\(0\)\s+var<uniform>/.test(vsCode) ||
      /@group\(0\)\s+@binding\(0\)\s+var<uniform>/.test(fsCode)
    ) {
      return FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV;
    }
    if (
      /@location\(0\)[\s\S]*?vec4<f32>[\s\S]*?@location\(1\)[\s\S]*?vec2<f32>/.test(vsCode) &&
      /@group\(0\)\s+@binding\(0\)[\s\S]*?texture_2d<f32>/.test(fsCode) &&
      /@group\(0\)\s+@binding\(1\)[\s\S]*?sampler/.test(fsCode)
    ) {
      return FALLBACK_PROFILE.TEXTURED_QUAD_24B;
    }

    return vsProfile || fsProfile || FALLBACK_PROFILE.DEFAULT;
  }

  function isValidGpuBuffer(buffer, requiredUsage) {
    if (!buffer || typeof buffer !== 'object') {
      return false;
    }
    if (buffer.destroyed === true) {
      return false;
    }
    if (
      typeof requiredUsage === 'number' &&
      requiredUsage !== 0 &&
      typeof buffer.usage === 'number' &&
      (buffer.usage & requiredUsage) === 0
    ) {
      return false;
    }
    return true;
  }

  function isValidIndirectBuffer(indirectBuffer, indirectOffset, indexed) {
    if (!isValidGpuBuffer(indirectBuffer, BUFFER_USAGE.INDIRECT)) {
      return false;
    }
    if (!Number.isInteger(indirectOffset) || indirectOffset < 0 || indirectOffset % 4 !== 0) {
      return false;
    }
    const minBytes = indexed ? 20 : 16;
    const bufSize =
      typeof indirectBuffer.size === 'number' && Number.isFinite(indirectBuffer.size)
        ? indirectBuffer.size
        : null;
    if (bufSize !== null) {
      if (bufSize < minBytes || indirectOffset + minBytes > bufSize) {
        return false;
      }
    }
    return true;
  }

  function isValidCbvBinding(entry) {
    if (!entry) {
      return false;
    }
    const buffer = entry.buffer !== undefined ? entry.buffer : entry;
    if (!isValidGpuBuffer(buffer, BUFFER_USAGE.UNIFORM)) {
      return false;
    }
    const offset = entry.offset !== undefined ? entry.offset : 0;
    if (!Number.isInteger(offset) || offset < 0 || offset % UNIFORM_OFFSET_ALIGNMENT !== 0) {
      return false;
    }
    const bufSize = typeof buffer.size === 'number' && Number.isFinite(buffer.size) ? buffer.size : null;
    const explicitSize =
      entry.size !== undefined && entry.size !== null
        ? entry.size
        : bufSize !== null
          ? bufSize - offset
          : null;
    if (explicitSize !== null) {
      if (!Number.isFinite(explicitSize) || explicitSize < FRAME1_CBV_MIN_BYTES) {
        return false;
      }
      if (bufSize !== null && offset + explicitSize > bufSize) {
        return false;
      }
    }
    return true;
  }

  function isValidVertexBufferBinding(entry, requiredBytes, expectedStride) {
    if (!entry) {
      return false;
    }
    const buffer = entry.buffer !== undefined ? entry.buffer : entry;
    if (!isValidGpuBuffer(buffer, BUFFER_USAGE.VERTEX)) {
      return false;
    }
    const offset = entry.offset !== undefined ? entry.offset : 0;
    if (!Number.isInteger(offset) || offset < 0 || offset % VERTEX_OFFSET_ALIGNMENT !== 0) {
      return false;
    }
    if (
      expectedStride !== undefined &&
      expectedStride !== null &&
      entry.stride !== undefined &&
      entry.stride !== null
    ) {
      if (!Number.isFinite(entry.stride) || entry.stride <= 0 || entry.stride !== expectedStride) {
        return false;
      }
    }
    const bufSize = typeof buffer.size === 'number' && Number.isFinite(buffer.size) ? buffer.size : null;
    if (bufSize !== null) {
      if (bufSize <= 0 || offset >= bufSize) {
        return false;
      }
      const explicitSize =
        entry.size !== undefined && entry.size !== null ? entry.size : bufSize - offset;
      if (!Number.isFinite(explicitSize) || explicitSize <= 0) {
        return false;
      }
      if (offset + explicitSize > bufSize) {
        return false;
      }
      if (requiredBytes > 0 && explicitSize < requiredBytes) {
        return false;
      }
    }
    return true;
  }

  function isValidTextureOrView(resource) {
    if (!resource || typeof resource !== 'object') {
      return false;
    }
    if (resource.destroyed === true) {
      return false;
    }
    return true;
  }

  function resolveTextureView(resource) {
    if (!isValidTextureOrView(resource)) {
      return null;
    }
    if (resource.textureView && isValidTextureOrView(resource.textureView)) {
      return resource.textureView;
    }
    if (resource.texture && isValidTextureOrView(resource.texture)) {
      if (typeof resource.texture.createView === 'function') {
        try {
          return resource.texture.createView();
        } catch (_err) {
          return null;
        }
      }
    }
    if (typeof resource.createView === 'function') {
      try {
        return resource.createView();
      } catch (_err) {
        return null;
      }
    }
    // Already a GPUTextureView (no createView method).
    return resource;
  }

  function isValidSampler(resource) {
    if (!resource || typeof resource !== 'object') {
      return false;
    }
    if (resource.destroyed === true) {
      return false;
    }
    if (resource.sampler && typeof resource.sampler === 'object') {
      return resource.sampler.destroyed !== true;
    }
    return true;
  }

  function unwrapSampler(resource) {
    if (!isValidSampler(resource)) {
      return null;
    }
    return resource.sampler && typeof resource.sampler === 'object' ? resource.sampler : resource;
  }

  function createZeroedStats() {
    return {
      totalDraws: 0,
      normalDraws: 0,
      fallbackDraws: 0,
      indexedDraws: 0,
      nonIndexedDraws: 0,
      indirectDraws: 0,
      skippedDraws: 0,
      zeroCountDrawsSkipped: 0,
      redundantPipelineSkips: 0,
      fallbackProfileCounts: {
        [FALLBACK_PROFILE.DEFAULT]: 0,
        [FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV]: 0,
        [FALLBACK_PROFILE.TEXTURED_QUAD_24B]: 0,
        [FALLBACK_PROFILE.SPLASH_RECT_24B]: 0,
      },
      fallbackPipelineCacheHits: 0,
      fallbackPipelineCacheMisses: 0,
      bindGroupGuards: 0,
      invalidCbvSubstitutions: 0,
      invalidTextureSubstitutions: 0,
      invalidSamplerSubstitutions: 0,
      skippedBindGroupBindings: 0,
      invalidVertexBufferGuards: 0,
      vertexBufferSubstitutions: 0,
      skippedVertexBufferBindings: 0,
      pipelineErrors: 0,
      bindGroupErrors: 0,
      passErrors: 0,
      legacyRouterFallbacks: 0,
    };
  }

  /**
   * WebGPU draw executor with G34 fallback WGSL profile routing and Frame-1
   * bind-group / vertex-buffer hardening.
   */
  class DrawExecutor {
    /**
     * @param {{
     *   device?: object | null,
     *   shaderFallbackRouter?: object | null,
     *   pipelineCache?: object | null,
     *   format?: string,
     *   depthStencilFormat?: string | null,
     *   maxFallbackPipelines?: number,
     *   enableShaderFallback?: boolean,
     *   elideRedundantPipelines?: boolean,
     *   onWarning?: (message: string, meta?: object) => void,
     * }} [options]
     */
    constructor(options = {}) {
      if (options !== undefined && options !== null && typeof options !== 'object') {
        throw new TypeError('DrawExecutor options must be an object');
      }
      const opts = options || {};

      this._device = opts.device !== undefined ? opts.device : null;
      this._pipelineCache = opts.pipelineCache !== undefined ? opts.pipelineCache : null;
      this._format =
        typeof opts.format === 'string' && opts.format.length > 0
          ? opts.format
          : DEFAULT_COLOR_FORMAT;
      this._depthStencilFormat =
        typeof opts.depthStencilFormat === 'string' && opts.depthStencilFormat.length > 0
          ? opts.depthStencilFormat
          : null;

      const maxPipelines =
        opts.maxFallbackPipelines !== undefined
          ? opts.maxFallbackPipelines
          : DEFAULT_MAX_FALLBACK_PIPELINES;
      if (!Number.isInteger(maxPipelines) || maxPipelines < 1) {
        throw new RangeError(
          `maxFallbackPipelines must be a positive integer, got ${maxPipelines}`
        );
      }
      this._maxFallbackPipelines = maxPipelines;
      this._elideRedundantPipelines = opts.elideRedundantPipelines !== false;
      this._onWarning = typeof opts.onWarning === 'function' ? opts.onWarning : null;

      if (opts.shaderFallbackRouter !== undefined) {
        this._shaderFallbackRouter = opts.shaderFallbackRouter;
      } else if (opts.enableShaderFallback !== false) {
        const RouterCtor = resolveShaderFallbackRouterCtor();
        this._shaderFallbackRouter = RouterCtor ? new RouterCtor() : null;
      } else {
        this._shaderFallbackRouter = null;
      }

      // Active per-pass draw state.
      this._activePipeline = null;
      this._boundPass = null;
      this._boundPipeline = null;
      this._vertexBuffers = new Map();
      this._indexBuffer = null;
      this._bindGroups = new Map();
      this._cbvs = new Map();
      this._textures = new Map();
      this._samplers = new Map();

      // Cached fallback resources & pipelines.
      this._fallbackPipelineCache = new Map();
      this._fallbackShaderModuleCache = new Map();
      this._fallbackCbvBuffer = null;
      this._fallbackTexture = null;
      this._fallbackTextureView = null;
      this._fallbackSampler = null;
      this._fallbackVertexBuffer24B = null;
      this._fallbackSplashVertexBuffer24B = null;

      this._destroyed = false;
      this._stats = createZeroedStats();
    }

    _warn(message, meta) {
      if (this._onWarning) {
        try {
          this._onWarning(message, meta);
        } catch (_err) {
          // Never allow diagnostic callbacks to break draw execution.
        }
      }
    }

    setDevice(device) {
      if (this._device !== device) {
        this._destroySyntheticResources();
        this._fallbackPipelineCache.clear();
        this._fallbackShaderModuleCache.clear();
        this._boundPass = null;
        this._boundPipeline = null;
      }
      this._device = device !== undefined ? device : null;
    }

    setShaderFallbackRouter(router) {
      this._shaderFallbackRouter = router !== undefined ? router : null;
    }

    setFormat(format) {
      if (typeof format === 'string' && format.length > 0) {
        this._format = format;
      }
    }

    /**
     * Reset per-pass bound draw state (call at the start of a render pass).
     */
    beginPass(options) {
      this.resetState();
      if (options && typeof options === 'object') {
        if (typeof options.format === 'string' && options.format.length > 0) {
          this._format = options.format;
        }
        if (options.depthStencilFormat !== undefined) {
          this._depthStencilFormat =
            typeof options.depthStencilFormat === 'string' && options.depthStencilFormat.length > 0
              ? options.depthStencilFormat
              : null;
        }
      }
    }

    /**
     * Mark the end of the active render pass so subsequent passes re-bind pipelines cleanly.
     */
    endPass() {
      this._boundPass = null;
      this._boundPipeline = null;
    }

    /**
     * Clear all bound pipeline, buffer, texture, sampler, and bind-group state.
     */
    resetState() {
      this._activePipeline = null;
      this._boundPass = null;
      this._boundPipeline = null;
      this._vertexBuffers.clear();
      this._indexBuffer = null;
      this._bindGroups.clear();
      this._cbvs.clear();
      this._textures.clear();
      this._samplers.clear();
    }

    /**
     * Reset active per-pass state (and optionally telemetry counters when `{ stats: true }`).
     */
    reset(options) {
      this.resetState();
      if (options && typeof options === 'object' && options.stats === true) {
        this.resetStats();
      }
    }

    _bindPipelineOnPass(pass, gpuPipeline) {
      if (
        this._elideRedundantPipelines &&
        this._boundPass === pass &&
        this._boundPipeline === gpuPipeline &&
        gpuPipeline !== null &&
        gpuPipeline !== undefined
      ) {
        this._stats.redundantPipelineSkips += 1;
        return false;
      }
      pass.setPipeline(gpuPipeline);
      this._boundPass = pass;
      this._boundPipeline = gpuPipeline;
      return true;
    }

    setPipeline(pipelineOrDesc) {
      this._activePipeline = pipelineOrDesc !== undefined ? pipelineOrDesc : null;
    }

    setVertexBuffer(slot, buffer, offsetOrOptions, size) {
      if (!Number.isInteger(slot) || slot < 0) {
        this._stats.invalidVertexBufferGuards += 1;
        this._stats.skippedVertexBufferBindings += 1;
        return false;
      }
      if (buffer === null || buffer === undefined) {
        this._vertexBuffers.delete(slot);
        return true;
      }
      let offset = 0;
      let explicitSize = size !== undefined ? size : null;
      let stride = null;

      if (typeof offsetOrOptions === 'object' && offsetOrOptions !== null) {
        offset = offsetOrOptions.offset !== undefined ? offsetOrOptions.offset : 0;
        if (offsetOrOptions.size !== undefined) {
          explicitSize = offsetOrOptions.size;
        }
        if (offsetOrOptions.stride !== undefined) {
          stride = offsetOrOptions.stride;
        } else if (offsetOrOptions.arrayStride !== undefined) {
          stride = offsetOrOptions.arrayStride;
        }
      } else if (typeof offsetOrOptions === 'number') {
        offset = offsetOrOptions;
      }

      if (stride === null && typeof buffer === 'object') {
        if (typeof buffer.stride === 'number') {
          stride = buffer.stride;
        } else if (typeof buffer.arrayStride === 'number') {
          stride = buffer.arrayStride;
        }
      }

      this._vertexBuffers.set(slot, {
        slot,
        buffer,
        offset,
        size: explicitSize,
        stride,
      });
      return true;
    }

    setIndexBuffer(buffer, indexFormat = 'uint16', offset = 0, size = null) {
      if (buffer === null || buffer === undefined) {
        this._indexBuffer = null;
        return true;
      }
      this._indexBuffer = {
        buffer,
        indexFormat,
        offset,
        size,
      };
      return true;
    }

    setBindGroup(groupIndex, bindGroup, dynamicOffsetsOrMeta) {
      if (!Number.isInteger(groupIndex) || groupIndex < 0) {
        this._stats.bindGroupGuards += 1;
        this._stats.skippedBindGroupBindings += 1;
        return false;
      }
      if (bindGroup === null || bindGroup === undefined) {
        this._bindGroups.delete(groupIndex);
        return true;
      }
      let dynamicOffsets = null;
      let meta = null;
      if (Array.isArray(dynamicOffsetsOrMeta) || ArrayBuffer.isView(dynamicOffsetsOrMeta)) {
        dynamicOffsets = dynamicOffsetsOrMeta;
      } else if (dynamicOffsetsOrMeta && typeof dynamicOffsetsOrMeta === 'object') {
        meta = dynamicOffsetsOrMeta;
        if (
          Array.isArray(dynamicOffsetsOrMeta.dynamicOffsets) ||
          ArrayBuffer.isView(dynamicOffsetsOrMeta.dynamicOffsets)
        ) {
          dynamicOffsets = dynamicOffsetsOrMeta.dynamicOffsets;
        }
      }

      this._bindGroups.set(groupIndex, {
        groupIndex,
        bindGroup: bindGroup.bindGroup !== undefined ? bindGroup.bindGroup : bindGroup,
        dynamicOffsets,
        entries:
          (meta && meta.entries) ||
          (bindGroup && bindGroup.entries) ||
          (bindGroup && bindGroup.desc && bindGroup.desc.entries) ||
          null,
        hasCbv: Boolean((meta && meta.hasCbv) || (bindGroup && bindGroup.hasCbv)),
        hasTexture: Boolean((meta && meta.hasTexture) || (bindGroup && bindGroup.hasTexture)),
        hasSampler: Boolean((meta && meta.hasSampler) || (bindGroup && bindGroup.hasSampler)),
        cbvBuffer: (meta && meta.cbvBuffer) || (bindGroup && bindGroup.cbvBuffer) || null,
        textureView: (meta && meta.textureView) || (bindGroup && bindGroup.textureView) || null,
        sampler: (meta && meta.sampler) || (bindGroup && bindGroup.sampler) || null,
      });
      return true;
    }

    setConstantBuffer(slot, buffer, options) {
      if (!Number.isInteger(slot) || slot < 0) {
        this._stats.bindGroupGuards += 1;
        return false;
      }
      if (buffer === null || buffer === undefined) {
        this._cbvs.delete(slot);
        return true;
      }
      const opts = options && typeof options === 'object' ? options : {};
      this._cbvs.set(slot, {
        slot,
        buffer,
        offset: opts.offset !== undefined ? opts.offset : 0,
        size: opts.size !== undefined ? opts.size : null,
      });
      return true;
    }

    setCbv(slot, buffer, options) {
      return this.setConstantBuffer(slot, buffer, options);
    }

    setTexture(slot, textureOrView) {
      if (!Number.isInteger(slot) || slot < 0) {
        this._stats.bindGroupGuards += 1;
        return false;
      }
      if (textureOrView === null || textureOrView === undefined) {
        this._textures.delete(slot);
        return true;
      }
      this._textures.set(slot, textureOrView);
      return true;
    }

    setSrv(slot, textureOrView) {
      return this.setTexture(slot, textureOrView);
    }

    setSampler(slot, sampler) {
      if (!Number.isInteger(slot) || slot < 0) {
        this._stats.bindGroupGuards += 1;
        return false;
      }
      if (sampler === null || sampler === undefined) {
        this._samplers.delete(slot);
        return true;
      }
      this._samplers.set(slot, sampler);
      return true;
    }

    /**
     * Snapshot active draw state merged with optional per-draw overrides.
     */
    getDrawState(overrides) {
      const ov = overrides && typeof overrides === 'object' ? overrides : {};
      const pipe = ov.pipeline !== undefined ? ov.pipeline : this._activePipeline;

      const vertexBuffers = [];
      for (const vb of this._vertexBuffers.values()) {
        vertexBuffers.push({ ...vb });
      }
      if (Array.isArray(ov.vertexBuffers)) {
        for (let i = 0; i < ov.vertexBuffers.length; i += 1) {
          if (ov.vertexBuffers[i]) {
            vertexBuffers.push(ov.vertexBuffers[i]);
          }
        }
      }

      const bindGroups = [];
      for (const bg of this._bindGroups.values()) {
        bindGroups.push(bg);
      }
      if (Array.isArray(ov.bindGroups)) {
        for (let i = 0; i < ov.bindGroups.length; i += 1) {
          if (ov.bindGroups[i]) {
            bindGroups.push(ov.bindGroups[i]);
          }
        }
      }

      const cbvs = Array.from(this._cbvs.values());
      const textures = Array.from(this._textures.values());
      const samplers = Array.from(this._samplers.values());

      let vertexStride =
        ov.vertexStride !== undefined
          ? ov.vertexStride
          : pipe && typeof pipe === 'object' && typeof pipe.vertexStride === 'number'
            ? pipe.vertexStride
            : null;
      if (vertexStride === null) {
        const slot0 = this._vertexBuffers.get(0);
        if (slot0 && typeof slot0.stride === 'number') {
          vertexStride = slot0.stride;
        }
      }

      const state = {
        vertexCount: ov.vertexCount !== undefined ? ov.vertexCount : 0,
        indexCount: ov.indexCount !== undefined ? ov.indexCount : 0,
        instanceCount: ov.instanceCount !== undefined ? ov.instanceCount : 1,
        firstVertex: ov.firstVertex !== undefined ? ov.firstVertex : 0,
        firstIndex: ov.firstIndex !== undefined ? ov.firstIndex : 0,
        baseVertex: ov.baseVertex !== undefined ? ov.baseVertex : 0,
        firstInstance: ov.firstInstance !== undefined ? ov.firstInstance : 0,
        indexed: Boolean(ov.indexed),
        pipeline: pipe,
        vertexStride,
        vertexBuffers,
        vertexBufferLayouts:
          ov.vertexBufferLayouts ||
          (pipe && typeof pipe === 'object' ? pipe.vertexBufferLayouts || pipe.vertexBuffers : null) ||
          null,
        bindGroups,
        cbvs,
        textures,
        samplers,
      };

      // Propagate explicit override hints when passed directly to executeDraw / resolve.
      const passthroughKeys = [
        'hasBoundCbv',
        'hasCbv',
        'cbvCount',
        'cbv',
        'cbvBuffer',
        'uniformBuffer',
        'hasBoundTexture',
        'hasTexture',
        'hasSampledTexture',
        'hasBoundSampledTexture',
        'hasBoundSrv',
        'textureCount',
        'srvCount',
        'texture',
        'textureView',
        'srv',
        'hasBoundSampler',
        'hasSampler',
        'samplerCount',
        'sampler',
        'vertexShaderHash',
        'fragmentShaderHash',
        'vertexShader',
        'fragmentShader',
        'topology',
        'isRectList',
        'format',
        'sampleCount',
        'depthStencilFormat',
        'isFallback',
        'usesFallbackShaders',
      ];
      for (let i = 0; i < passthroughKeys.length; i += 1) {
        const key = passthroughKeys[i];
        if (ov[key] !== undefined) {
          state[key] = ov[key];
        }
      }

      return state;
    }

    _needsFallbackRouting(drawState) {
      if (drawState.isFallback === true || drawState.usesFallbackShaders === true) {
        return true;
      }
      const pipe = drawState.pipeline;
      if (!pipe) {
        return true;
      }
      if (typeof pipe === 'object') {
        if (pipe.isFallback === true || pipe.usesFallbackShaders === true || pipe.missingShaders === true) {
          return true;
        }
        const vs = drawState.vertexShader || pipe.vertexShader;
        const fs = drawState.fragmentShader || pipe.fragmentShader;
        if (
          (vs && (vs.missing === true || vs.isFallback === true || vs.source === 'builtin' || vs.source === 'fallback')) ||
          (fs && (fs.missing === true || fs.isFallback === true || fs.source === 'builtin' || fs.source === 'fallback'))
        ) {
          return true;
        }
        // Descriptor object with no compiled GPURenderPipeline handle attached.
        if (!pipe.pipeline && typeof pipe.getBindGroupLayout !== 'function') {
          return true;
        }
      }
      return false;
    }

    _extractShaderHash(drawState, stageName) {
      const pipe = drawState.pipeline && typeof drawState.pipeline === 'object' ? drawState.pipeline : null;
      const stageObj =
        stageName === 'vertex'
          ? drawState.vertexShader || (pipe && pipe.vertexShader)
          : drawState.fragmentShader || (pipe && pipe.fragmentShader);
      const explicitHash =
        stageName === 'vertex'
          ? drawState.vertexShaderHash || (pipe && pipe.vertexShaderHash)
          : drawState.fragmentShaderHash || (pipe && pipe.fragmentShaderHash);

      if (explicitHash !== undefined && explicitHash !== null && explicitHash !== '') {
        return explicitHash;
      }
      if (stageObj && typeof stageObj === 'object') {
        if (stageObj.requestedHash !== undefined && stageObj.requestedHash !== null) {
          return stageObj.requestedHash;
        }
        if (stageObj.hash !== undefined && stageObj.hash !== null) {
          return stageObj.hash;
        }
        if (stageObj.hashHex !== undefined && stageObj.hashHex !== null) {
          return stageObj.hashHex;
        }
      }
      const pipeId = pipe && pipe.id !== undefined ? pipe.id : 0;
      return fnv1a64Hex(`missing:${stageName}:${pipeId}`);
    }

    _invokeRouterResolve(hash, stageName, profile) {
      const router = this._shaderFallbackRouter;
      if (!router || typeof router.resolve !== 'function') {
        return {
          wgsl: EMERGENCY_BUILTIN_WGSL[stageName],
          source: 'builtin',
          entryPoint: stageName === 'fragment' ? 'fs_main' : 'vs_main',
          resolvedHash: null,
          profile: FALLBACK_PROFILE.DEFAULT,
        };
      }

      let raw = null;
      try {
        raw = router.resolve(hash, stageName, { profile });
      } catch (_err3) {
        // Backwards compatibility with strict legacy 2-arg mock routers that reject a 3rd options arg.
        this._stats.legacyRouterFallbacks += 1;
        try {
          raw = router.resolve(hash, stageName);
        } catch (err2) {
          this._warn('ShaderFallbackRouter.resolve failed', {
            stage: stageName,
            profile,
            error: String(err2 && err2.message ? err2.message : err2),
          });
          return null;
        }
      }

      return unwrapRouterResult(raw, stageName);
    }

    /**
     * Select and resolve fallback vertex + fragment WGSL for a draw state.
     *
     * @param {object} drawState
     * @param {string} [overrideProfile]
     */
    resolveFallbackShaders(drawState, overrideProfile) {
      const state = drawState && typeof drawState === 'object' ? drawState : this.getDrawState();
      const selectedProfile =
        overrideProfile !== undefined && overrideProfile !== null
          ? normalizeProfileName(overrideProfile)
          : selectFallbackShaderProfile(state);

      const vsHash = this._extractShaderHash(state, 'vertex');
      const fsHash = this._extractShaderHash(state, 'fragment');

      const vertex = this._invokeRouterResolve(vsHash, 'vertex', selectedProfile);
      const fragment = this._invokeRouterResolve(fsHash, 'fragment', selectedProfile);

      if (!vertex || !fragment) {
        return null;
      }

      const effectiveProfile = inferEffectiveProfileFromWgsl(vertex, fragment);
      return {
        selectedProfile,
        effectiveProfile,
        vertex,
        fragment,
      };
    }

    _getOrCreateShaderModule(wgsl, label) {
      if (!this._device || typeof this._device.createShaderModule !== 'function') {
        return null;
      }
      const key = fnv1a64Hex(wgsl);
      const cached = this._fallbackShaderModuleCache.get(key);
      if (cached) {
        return cached;
      }
      const module = this._device.createShaderModule({ code: wgsl, label });
      if (this._fallbackShaderModuleCache.size >= this._maxFallbackPipelines * 2) {
        const oldestKey = this._fallbackShaderModuleCache.keys().next().value;
        this._fallbackShaderModuleCache.delete(oldestKey);
      }
      this._fallbackShaderModuleCache.set(key, module);
      return module;
    }

    _getOrCreateFallbackPipeline(resolved, drawState) {
      if (!this._device || typeof this._device.createRenderPipeline !== 'function') {
        return null;
      }

      const pipe = drawState.pipeline && typeof drawState.pipeline === 'object' ? drawState.pipeline : null;
      const format = drawState.format || (pipe && pipe.format) || this._format || DEFAULT_COLOR_FORMAT;
      const rawTopology = drawState.topology || (pipe && pipe.topology) || DEFAULT_TOPOLOGY;
      const topology = isRectListTopology(rawTopology) ? DEFAULT_TOPOLOGY : rawTopology;
      const sampleCount = toNonNegativeInt(
        drawState.sampleCount !== undefined
          ? drawState.sampleCount
          : pipe && pipe.sampleCount !== undefined
            ? pipe.sampleCount
            : 1,
        1
      ) || 1;
      const depthStencilFormat =
        drawState.depthStencilFormat !== undefined
          ? drawState.depthStencilFormat
          : pipe && pipe.depthStencilFormat !== undefined
            ? pipe.depthStencilFormat
            : this._depthStencilFormat;

      const vsKey = fnv1a64Hex(resolved.vertex.wgsl);
      const fsKey = fnv1a64Hex(resolved.fragment.wgsl);
      const cacheKey = `${resolved.effectiveProfile}|${vsKey}|${fsKey}|${format}|${topology}|${sampleCount}|${depthStencilFormat || 'none'}`;

      const cached = this._fallbackPipelineCache.get(cacheKey);
      if (cached) {
        this._stats.fallbackPipelineCacheHits += 1;
        return cached;
      }

      this._stats.fallbackPipelineCacheMisses += 1;
      try {
        const vsModule = this._getOrCreateShaderModule(
          resolved.vertex.wgsl,
          `fallback:${resolved.effectiveProfile}:vs`
        );
        const fsModule = this._getOrCreateShaderModule(
          resolved.fragment.wgsl,
          `fallback:${resolved.effectiveProfile}:fs`
        );
        if (!vsModule || !fsModule) {
          this._stats.pipelineErrors += 1;
          return null;
        }

        const vertexBuffers =
          resolved.effectiveProfile === FALLBACK_PROFILE.SPLASH_RECT_24B
            ? [
                {
                  arrayStride: FRAME1_SPLASH_VERTEX_LAYOUT_24B.arrayStride,
                  stepMode: FRAME1_SPLASH_VERTEX_LAYOUT_24B.stepMode,
                  attributes: FRAME1_SPLASH_VERTEX_LAYOUT_24B.attributes.map((a) => ({ ...a })),
                },
              ]
            : resolved.effectiveProfile === FALLBACK_PROFILE.TEXTURED_QUAD_24B
              ? [
                  {
                    arrayStride: FALLBACK_VERTEX_LAYOUT_24B.arrayStride,
                    stepMode: FALLBACK_VERTEX_LAYOUT_24B.stepMode,
                    attributes: FALLBACK_VERTEX_LAYOUT_24B.attributes.map((a) => ({ ...a })),
                  },
                ]
              : [];

        const descriptor = {
          label: `draw_executor:fallback:${resolved.effectiveProfile}`,
          layout: 'auto',
          vertex: {
            module: vsModule,
            entryPoint: resolved.vertex.entryPoint || 'vs_main',
            buffers: vertexBuffers,
          },
          fragment: {
            module: fsModule,
            entryPoint: resolved.fragment.entryPoint || 'fs_main',
            targets: [{ format }],
          },
          primitive: {
            topology,
          },
        };

        if (sampleCount > 1) {
          descriptor.multisample = { count: sampleCount };
        }
        if (depthStencilFormat) {
          descriptor.depthStencil = {
            format: depthStencilFormat,
            depthWriteEnabled: false,
            depthCompare: 'always',
          };
        }

        const pipeline =
          this._pipelineCache && typeof this._pipelineCache.getOrCreateRenderPipeline === 'function'
            ? this._pipelineCache.getOrCreateRenderPipeline(descriptor)
            : this._device.createRenderPipeline(descriptor);

        if (!pipeline) {
          this._stats.pipelineErrors += 1;
          return null;
        }

        if (this._fallbackPipelineCache.size >= this._maxFallbackPipelines) {
          const oldestKey = this._fallbackPipelineCache.keys().next().value;
          this._fallbackPipelineCache.delete(oldestKey);
        }
        this._fallbackPipelineCache.set(cacheKey, pipeline);
        return pipeline;
      } catch (err) {
        this._stats.pipelineErrors += 1;
        this._warn('Failed to create fallback render pipeline', {
          profile: resolved.effectiveProfile,
          error: String(err && err.message ? err.message : err),
        });
        return null;
      }
    }

    _ensureFallbackCbvBuffer() {
      if (this._fallbackCbvBuffer && this._fallbackCbvBuffer.destroyed !== true) {
        return this._fallbackCbvBuffer;
      }
      if (!this._device || typeof this._device.createBuffer !== 'function') {
        return null;
      }
      try {
        const buf = this._device.createBuffer({
          label: 'draw_executor:fallback_cbv_256b',
          size: FRAME1_CBV_MIN_BYTES,
          usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST,
        });
        if (buf && this._device.queue && typeof this._device.queue.writeBuffer === 'function') {
          // Deterministic non-zero seed bytes (all finite float values) so Frame-1 modulation is visible.
          const seed = new Uint32Array(FRAME1_CBV_MIN_BYTES / 4);
          seed[0] = 0x3f204080;
          seed[1] = 0x3f4080c0;
          seed[2] = 0x3f60a0e0;
          seed[3] = 0x3f800000;
          this._device.queue.writeBuffer(buf, 0, seed);
        }
        this._fallbackCbvBuffer = buf;
        return buf;
      } catch (err) {
        this._warn('Failed to create synthetic fallback CBV buffer', {
          error: String(err && err.message ? err.message : err),
        });
        return null;
      }
    }

    _ensureFallbackTextureView() {
      if (
        this._fallbackTexture &&
        this._fallbackTexture.destroyed !== true &&
        this._fallbackTextureView
      ) {
        return this._fallbackTextureView;
      }
      if (!this._device || typeof this._device.createTexture !== 'function') {
        return null;
      }
      try {
        const tex = this._device.createTexture({
          label: 'draw_executor:fallback_tex_1x1',
          size: { width: 1, height: 1, depthOrArrayLayers: 1 },
          mipLevelCount: 1,
          format: 'rgba8unorm',
          usage: TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
          dimension: '2d',
        });
        if (!tex) {
          return null;
        }
        if (this._device.queue && typeof this._device.queue.writeTexture === 'function') {
          const pixel = new Uint8Array([0x28, 0x38, 0x50, 0xff]);
          this._device.queue.writeTexture(
            { texture: tex },
            pixel,
            { bytesPerRow: 4, rowsPerImage: 1 },
            { width: 1, height: 1, depthOrArrayLayers: 1 }
          );
        }
        const view = typeof tex.createView === 'function' ? tex.createView() : tex;
        this._fallbackTexture = tex;
        this._fallbackTextureView = view;
        return view;
      } catch (err) {
        this._warn('Failed to create synthetic fallback texture', {
          error: String(err && err.message ? err.message : err),
        });
        return null;
      }
    }

    _ensureFallbackSampler() {
      if (this._fallbackSampler && this._fallbackSampler.destroyed !== true) {
        return this._fallbackSampler;
      }
      if (!this._device || typeof this._device.createSampler !== 'function') {
        return null;
      }
      try {
        this._fallbackSampler = this._device.createSampler({
          label: 'draw_executor:fallback_sampler',
          magFilter: 'linear',
          minFilter: 'linear',
          addressModeU: 'clamp-to-edge',
          addressModeV: 'clamp-to-edge',
        });
        return this._fallbackSampler;
      } catch (err) {
        this._warn('Failed to create synthetic fallback sampler', {
          error: String(err && err.message ? err.message : err),
        });
        return null;
      }
    }

    _ensureFallbackVertexBuffer24B() {
      if (this._fallbackVertexBuffer24B && this._fallbackVertexBuffer24B.destroyed !== true) {
        return this._fallbackVertexBuffer24B;
      }
      if (!this._device || typeof this._device.createBuffer !== 'function') {
        return null;
      }
      try {
        // 6 vertices * 24 bytes (vec4<f32> pos + vec2<f32> uv) = 144 bytes (fullscreen quad).
        const quad = new Float32Array([
          // Triangle 1
          -1.0, -1.0, 0.0, 1.0, 0.0, 1.0,
           1.0, -1.0, 0.0, 1.0, 1.0, 1.0,
          -1.0,  1.0, 0.0, 1.0, 0.0, 0.0,
          // Triangle 2
          -1.0,  1.0, 0.0, 1.0, 0.0, 0.0,
           1.0, -1.0, 0.0, 1.0, 1.0, 1.0,
           1.0,  1.0, 0.0, 1.0, 1.0, 0.0,
        ]);
        const buf = this._device.createBuffer({
          label: 'draw_executor:fallback_vb_24b_quad',
          size: quad.byteLength,
          usage: BUFFER_USAGE.VERTEX | BUFFER_USAGE.COPY_DST,
        });
        if (buf && this._device.queue && typeof this._device.queue.writeBuffer === 'function') {
          this._device.queue.writeBuffer(buf, 0, quad);
        }
        this._fallbackVertexBuffer24B = buf;
        return buf;
      } catch (err) {
        this._warn('Failed to create synthetic 24B fallback vertex buffer', {
          error: String(err && err.message ? err.message : err),
        });
        return null;
      }
    }

    _ensureFallbackSplashVertexBuffer24B() {
      if (
        this._fallbackSplashVertexBuffer24B &&
        this._fallbackSplashVertexBuffer24B.destroyed !== true
      ) {
        return this._fallbackSplashVertexBuffer24B;
      }
      if (!this._device || typeof this._device.createBuffer !== 'function') {
        return null;
      }
      try {
        // 3 RECTLIST vertices in exact Frame-1 24B layout (float32x3 pos, unorm8x4 color, float32x2 uv)
        // expanded to 6 triangle-list vertices (144 bytes).
        const rect3 = new Uint8Array(3 * TEXTURED_QUAD_VERTEX_STRIDE);
        const view = new DataView(rect3.buffer);
        const corners = [
          { x: -1.0, y: 1.0, z: 0.0, u: 0.0, v: 0.0 },
          { x: 1.0, y: 1.0, z: 0.0, u: 1.0, v: 0.0 },
          { x: -1.0, y: -1.0, z: 0.0, u: 0.0, v: 1.0 },
        ];
        for (let i = 0; i < 3; i += 1) {
          const base = i * TEXTURED_QUAD_VERTEX_STRIDE;
          view.setFloat32(base + 0, corners[i].x, true);
          view.setFloat32(base + 4, corners[i].y, true);
          view.setFloat32(base + 8, corners[i].z, true);
          rect3[base + 12] = 0xff;
          rect3[base + 13] = 0xff;
          rect3[base + 14] = 0xff;
          rect3[base + 15] = 0xff;
          view.setFloat32(base + 16, corners[i].u, true);
          view.setFloat32(base + 20, corners[i].v, true);
        }
        const expanded = expandRectListVertices24B(rect3, 3);
        const buf = this._device.createBuffer({
          label: 'draw_executor:fallback_vb_24b_splash_rect',
          size: expanded.byteLength,
          usage: BUFFER_USAGE.VERTEX | BUFFER_USAGE.COPY_DST,
        });
        if (buf && this._device.queue && typeof this._device.queue.writeBuffer === 'function') {
          this._device.queue.writeBuffer(buf, 0, expanded);
        }
        this._fallbackSplashVertexBuffer24B = buf;
        return buf;
      } catch (err) {
        this._warn('Failed to create synthetic 24B splash rect fallback vertex buffer', {
          error: String(err && err.message ? err.message : err),
        });
        return null;
      }
    }

    _findBoundCbvCandidate(drawState) {
      if (drawState.cbvBuffer) {
        return { buffer: drawState.cbvBuffer, offset: 0, size: null };
      }
      if (drawState.uniformBuffer) {
        return { buffer: drawState.uniformBuffer, offset: 0, size: null };
      }
      if (drawState.cbv && typeof drawState.cbv === 'object') {
        return drawState.cbv.buffer !== undefined
          ? drawState.cbv
          : { buffer: drawState.cbv, offset: 0, size: null };
      }
      if (this._cbvs.has(0)) {
        return this._cbvs.get(0);
      }
      for (const entry of this._cbvs.values()) {
        return entry;
      }
      const bg0 = this._bindGroups.get(0);
      if (bg0 && bg0.cbvBuffer) {
        return { buffer: bg0.cbvBuffer, offset: 0, size: null };
      }
      if (bg0 && Array.isArray(bg0.entries)) {
        for (let i = 0; i < bg0.entries.length; i += 1) {
          const e = bg0.entries[i];
          if (e && e.resource && typeof e.resource === 'object' && e.resource.buffer) {
            return {
              buffer: e.resource.buffer,
              offset: e.resource.offset || 0,
              size: e.resource.size !== undefined ? e.resource.size : null,
            };
          }
        }
      }
      return null;
    }

    _findBoundTextureCandidate(drawState) {
      if (drawState.textureView) return drawState.textureView;
      if (drawState.texture) return drawState.texture;
      if (drawState.srv) return drawState.srv;
      if (this._textures.has(0)) return this._textures.get(0);
      for (const tex of this._textures.values()) {
        return tex;
      }
      const bg0 = this._bindGroups.get(0);
      if (bg0 && bg0.textureView) return bg0.textureView;
      if (bg0 && Array.isArray(bg0.entries)) {
        for (let i = 0; i < bg0.entries.length; i += 1) {
          const e = bg0.entries[i];
          if (!e) continue;
          if (e.textureView) return e.textureView;
          if (e.texture) return e.texture;
          if (
            e.resource &&
            typeof e.resource === 'object' &&
            (e.type === 'texture' ||
              e.resourceType === 1 ||
              e.resourceType === 'textureView' ||
              e.resource.kind === 'GPUTextureView' ||
              e.resource.kind === 'GPUTexture')
          ) {
            return e.resource;
          }
        }
      }
      return null;
    }

    _findBoundSamplerCandidate(drawState) {
      if (drawState.sampler) return drawState.sampler;
      if (this._samplers.has(0)) return this._samplers.get(0);
      for (const s of this._samplers.values()) {
        return s;
      }
      const bg0 = this._bindGroups.get(0);
      if (bg0 && bg0.sampler) return bg0.sampler;
      if (bg0 && Array.isArray(bg0.entries)) {
        for (let i = 0; i < bg0.entries.length; i += 1) {
          const e = bg0.entries[i];
          if (!e) continue;
          if (e.sampler) return e.sampler;
          if (
            e.resource &&
            typeof e.resource === 'object' &&
            (e.type === 'sampler' ||
              e.resourceType === 2 ||
              e.resourceType === 'sampler' ||
              e.resource.kind === 'GPUSampler')
          ) {
            return e.resource;
          }
        }
      }
      return null;
    }

    _bindFallbackGroupForCbv(pass, pipeline, drawState) {
      if (
        !this._device ||
        typeof this._device.createBindGroup !== 'function' ||
        typeof pipeline.getBindGroupLayout !== 'function'
      ) {
        this._stats.bindGroupErrors += 1;
        return { ok: false, substitutedCbv: false };
      }

      const candidate = this._findBoundCbvCandidate(drawState);
      let buffer;
      let offset = 0;
      let size = FRAME1_CBV_MIN_BYTES;
      let substitutedCbv = false;

      if (candidate && isValidCbvBinding(candidate)) {
        buffer = candidate.buffer !== undefined ? candidate.buffer : candidate;
        offset = candidate.offset || 0;
        if (candidate.size !== undefined && candidate.size !== null) {
          size = candidate.size;
        } else if (typeof buffer.size === 'number' && Number.isFinite(buffer.size)) {
          size = Math.max(FRAME1_CBV_MIN_BYTES, buffer.size - offset);
        }
      } else {
        this._stats.bindGroupGuards += 1;
        this._stats.invalidCbvSubstitutions += 1;
        substitutedCbv = true;
        buffer = this._ensureFallbackCbvBuffer();
        if (!buffer) {
          this._stats.bindGroupErrors += 1;
          return { ok: false, substitutedCbv: true };
        }
      }

      try {
        const layout = pipeline.getBindGroupLayout(0);
        const bindGroup = this._device.createBindGroup({
          label: 'draw_executor:fallback_cbv_bg',
          layout,
          entries: [
            {
              binding: 0,
              resource: { buffer, offset, size },
            },
          ],
        });
        pass.setBindGroup(0, bindGroup);
        return { ok: true, substitutedCbv };
      } catch (err) {
        this._stats.bindGroupErrors += 1;
        this._warn('Failed to bind fallback CBV bind group', {
          error: String(err && err.message ? err.message : err),
        });
        return { ok: false, substitutedCbv };
      }
    }

    _bindFallbackGroupForTexturedQuad(pass, pipeline, drawState) {
      if (
        !this._device ||
        typeof this._device.createBindGroup !== 'function' ||
        typeof pipeline.getBindGroupLayout !== 'function'
      ) {
        this._stats.bindGroupErrors += 1;
        return { ok: false, substitutedTexture: false, substitutedSampler: false };
      }

      let substitutedTexture = false;
      let substitutedSampler = false;

      let textureView = resolveTextureView(this._findBoundTextureCandidate(drawState));
      if (!textureView) {
        this._stats.bindGroupGuards += 1;
        this._stats.invalidTextureSubstitutions += 1;
        substitutedTexture = true;
        textureView = this._ensureFallbackTextureView();
        if (!textureView) {
          this._stats.bindGroupErrors += 1;
          return { ok: false, substitutedTexture: true, substitutedSampler: false };
        }
      }

      let sampler = unwrapSampler(this._findBoundSamplerCandidate(drawState));
      if (!sampler) {
        this._stats.bindGroupGuards += 1;
        this._stats.invalidSamplerSubstitutions += 1;
        substitutedSampler = true;
        sampler = this._ensureFallbackSampler();
        if (!sampler) {
          this._stats.bindGroupErrors += 1;
          return { ok: false, substitutedTexture, substitutedSampler: true };
        }
      }

      try {
        const layout = pipeline.getBindGroupLayout(0);
        const bindGroup = this._device.createBindGroup({
          label: 'draw_executor:fallback_textured_quad_bg',
          layout,
          entries: [
            { binding: 0, resource: textureView },
            { binding: 1, resource: sampler },
          ],
        });
        pass.setBindGroup(0, bindGroup);
        return { ok: true, substitutedTexture, substitutedSampler };
      } catch (err) {
        this._stats.bindGroupErrors += 1;
        this._warn('Failed to bind fallback textured quad bind group', {
          error: String(err && err.message ? err.message : err),
        });
        return { ok: false, substitutedTexture, substitutedSampler };
      }
    }

    _bindFallbackVertexBuffer24B(pass, drawState, vertexCount, firstVertex) {
      const slot0 = this._vertexBuffers.get(0) || (Array.isArray(drawState.vertexBuffers) ? drawState.vertexBuffers[0] : null);
      const requiredBytes = drawState.indexed
        ? TEXTURED_QUAD_VERTEX_STRIDE
        : (firstVertex + vertexCount) * TEXTURED_QUAD_VERTEX_STRIDE;

      if (slot0 && isValidVertexBufferBinding(slot0, requiredBytes, TEXTURED_QUAD_VERTEX_STRIDE)) {
        const buffer = slot0.buffer !== undefined ? slot0.buffer : slot0;
        const offset = slot0.offset || 0;
        try {
          if (slot0.size !== undefined && slot0.size !== null) {
            pass.setVertexBuffer(0, buffer, offset, slot0.size);
          } else {
            pass.setVertexBuffer(0, buffer, offset);
          }
          return {
            ok: true,
            substitutedVertexBuffer: false,
            effectiveVertexCount: vertexCount,
            effectiveFirstVertex: firstVertex,
          };
        } catch (err) {
          this._warn('setVertexBuffer threw on bound 24B vertex buffer; substituting synthetic quad', {
            error: String(err && err.message ? err.message : err),
          });
        }
      }

      this._stats.invalidVertexBufferGuards += 1;
      this._stats.vertexBufferSubstitutions += 1;
      const fallbackVb = this._ensureFallbackVertexBuffer24B();
      if (!fallbackVb) {
        return {
          ok: false,
          substitutedVertexBuffer: true,
          effectiveVertexCount: 0,
          effectiveFirstVertex: 0,
        };
      }

      try {
        pass.setVertexBuffer(0, fallbackVb, 0);
        // Synthetic fallback quad holds 6 vertices (2 triangles); clamp non-indexed draw range safely.
        const clampedCount = drawState.indexed ? vertexCount : Math.min(Math.max(vertexCount, 3), 6);
        return {
          ok: true,
          substitutedVertexBuffer: true,
          effectiveVertexCount: clampedCount,
          effectiveFirstVertex: 0,
        };
      } catch (err) {
        this._stats.passErrors += 1;
        this._warn('Failed to bind synthetic 24B fallback vertex buffer', {
          error: String(err && err.message ? err.message : err),
        });
        return {
          ok: false,
          substitutedVertexBuffer: true,
          effectiveVertexCount: 0,
          effectiveFirstVertex: 0,
        };
      }
    }

    _bindFallbackGroupForSplashRect(pass, pipeline, drawState) {
      if (
        !this._device ||
        typeof this._device.createBindGroup !== 'function' ||
        typeof pipeline.getBindGroupLayout !== 'function'
      ) {
        this._stats.bindGroupErrors += 1;
        return {
          ok: false,
          substitutedCbv: false,
          substitutedTexture: false,
          substitutedSampler: false,
        };
      }

      const cbvCandidate = this._findBoundCbvCandidate(drawState);
      let cbvBuffer;
      let cbvOffset = 0;
      let cbvSize = FRAME1_CBV_MIN_BYTES;
      let substitutedCbv = false;
      let substitutedTexture = false;
      let substitutedSampler = false;

      if (cbvCandidate && isValidCbvBinding(cbvCandidate)) {
        cbvBuffer = cbvCandidate.buffer !== undefined ? cbvCandidate.buffer : cbvCandidate;
        cbvOffset = cbvCandidate.offset || 0;
        if (cbvCandidate.size !== undefined && cbvCandidate.size !== null) {
          cbvSize = cbvCandidate.size;
        } else if (typeof cbvBuffer.size === 'number' && Number.isFinite(cbvBuffer.size)) {
          cbvSize = Math.max(FRAME1_CBV_MIN_BYTES, cbvBuffer.size - cbvOffset);
        }
      } else {
        this._stats.bindGroupGuards += 1;
        this._stats.invalidCbvSubstitutions += 1;
        substitutedCbv = true;
        cbvBuffer = this._ensureFallbackCbvBuffer();
        if (!cbvBuffer) {
          this._stats.bindGroupErrors += 1;
          return {
            ok: false,
            substitutedCbv: true,
            substitutedTexture: false,
            substitutedSampler: false,
          };
        }
      }

      let textureView = resolveTextureView(this._findBoundTextureCandidate(drawState));
      if (!textureView) {
        this._stats.bindGroupGuards += 1;
        this._stats.invalidTextureSubstitutions += 1;
        substitutedTexture = true;
        textureView = this._ensureFallbackTextureView();
        if (!textureView) {
          this._stats.bindGroupErrors += 1;
          return {
            ok: false,
            substitutedCbv,
            substitutedTexture: true,
            substitutedSampler: false,
          };
        }
      }

      let sampler = unwrapSampler(this._findBoundSamplerCandidate(drawState));
      if (!sampler) {
        this._stats.bindGroupGuards += 1;
        this._stats.invalidSamplerSubstitutions += 1;
        substitutedSampler = true;
        sampler = this._ensureFallbackSampler();
        if (!sampler) {
          this._stats.bindGroupErrors += 1;
          return {
            ok: false,
            substitutedCbv,
            substitutedTexture,
            substitutedSampler: true,
          };
        }
      }

      try {
        const layout = pipeline.getBindGroupLayout(0);
        const bindGroup = this._device.createBindGroup({
          label: 'draw_executor:fallback_splash_rect_bg',
          layout,
          entries: [
            { binding: 0, resource: { buffer: cbvBuffer, offset: cbvOffset, size: cbvSize } },
            { binding: 1, resource: textureView },
            { binding: 2, resource: sampler },
          ],
        });
        pass.setBindGroup(0, bindGroup);
        return { ok: true, substitutedCbv, substitutedTexture, substitutedSampler };
      } catch (err) {
        this._stats.bindGroupErrors += 1;
        this._warn('Failed to bind fallback splash rect bind group', {
          error: String(err && err.message ? err.message : err),
        });
        return { ok: false, substitutedCbv, substitutedTexture, substitutedSampler };
      }
    }

    _bindFallbackVertexBufferSplashRect24B(pass, drawState, vertexCount, firstVertex) {
      const slot0 =
        this._vertexBuffers.get(0) ||
        (Array.isArray(drawState.vertexBuffers) ? drawState.vertexBuffers[0] : null);
      const requiredBytes = drawState.indexed
        ? TEXTURED_QUAD_VERTEX_STRIDE
        : (firstVertex + vertexCount) * TEXTURED_QUAD_VERTEX_STRIDE;

      if (slot0 && isValidVertexBufferBinding(slot0, requiredBytes, TEXTURED_QUAD_VERTEX_STRIDE)) {
        const buffer = slot0.buffer !== undefined ? slot0.buffer : slot0;
        const offset = slot0.offset || 0;
        try {
          if (slot0.size !== undefined && slot0.size !== null) {
            pass.setVertexBuffer(0, buffer, offset, slot0.size);
          } else {
            pass.setVertexBuffer(0, buffer, offset);
          }
          return {
            ok: true,
            substitutedVertexBuffer: false,
            effectiveVertexCount: vertexCount,
            effectiveFirstVertex: firstVertex,
          };
        } catch (err) {
          this._warn(
            'setVertexBuffer threw on bound 24B splash vertex buffer; substituting synthetic splash quad',
            {
              error: String(err && err.message ? err.message : err),
            }
          );
        }
      }

      this._stats.invalidVertexBufferGuards += 1;
      this._stats.vertexBufferSubstitutions += 1;
      const fallbackVb = this._ensureFallbackSplashVertexBuffer24B();
      if (!fallbackVb) {
        return {
          ok: false,
          substitutedVertexBuffer: true,
          effectiveVertexCount: 0,
          effectiveFirstVertex: 0,
        };
      }

      try {
        pass.setVertexBuffer(0, fallbackVb, 0);
        const expandedCount =
          isRectListDrawState(drawState) && vertexCount >= 3
            ? Math.floor(vertexCount / 3) * 6
            : vertexCount;
        const clampedCount = drawState.indexed ? vertexCount : Math.min(Math.max(expandedCount, 3), 6);
        return {
          ok: true,
          substitutedVertexBuffer: true,
          effectiveVertexCount: clampedCount,
          effectiveFirstVertex: 0,
        };
      } catch (err) {
        this._stats.passErrors += 1;
        this._warn('Failed to bind synthetic 24B splash fallback vertex buffer', {
          error: String(err && err.message ? err.message : err),
        });
        return {
          ok: false,
          substitutedVertexBuffer: true,
          effectiveVertexCount: 0,
          effectiveFirstVertex: 0,
        };
      }
    }

    _bindNormalResources(pass) {
      // Bind validated vertex buffers.
      for (const [slot, entry] of this._vertexBuffers.entries()) {
        if (!isValidVertexBufferBinding(entry, 0, null)) {
          this._stats.invalidVertexBufferGuards += 1;
          this._stats.skippedVertexBufferBindings += 1;
          continue;
        }
        try {
          if (entry.size !== undefined && entry.size !== null) {
            pass.setVertexBuffer(slot, entry.buffer, entry.offset || 0, entry.size);
          } else {
            pass.setVertexBuffer(slot, entry.buffer, entry.offset || 0);
          }
        } catch (err) {
          this._stats.passErrors += 1;
          this._stats.invalidVertexBufferGuards += 1;
          this._stats.skippedVertexBufferBindings += 1;
          this._warn('setVertexBuffer failed on normal draw', {
            slot,
            error: String(err && err.message ? err.message : err),
          });
        }
      }

      // Bind validated bind groups.
      for (const [groupIndex, entry] of this._bindGroups.entries()) {
        if (!entry || !entry.bindGroup || entry.bindGroup.destroyed === true) {
          this._stats.bindGroupGuards += 1;
          this._stats.skippedBindGroupBindings += 1;
          continue;
        }
        try {
          if (entry.dynamicOffsets) {
            pass.setBindGroup(groupIndex, entry.bindGroup, entry.dynamicOffsets);
          } else {
            pass.setBindGroup(groupIndex, entry.bindGroup);
          }
        } catch (err) {
          this._stats.passErrors += 1;
          this._stats.bindGroupGuards += 1;
          this._stats.skippedBindGroupBindings += 1;
          this._warn('setBindGroup failed on normal draw', {
            groupIndex,
            error: String(err && err.message ? err.message : err),
          });
        }
      }
    }

    _validateAndBindIndexBuffer(pass, indexCount, firstIndex) {
      const ib = this._indexBuffer;
      if (!ib || !isValidGpuBuffer(ib.buffer, BUFFER_USAGE.INDEX)) {
        return false;
      }
      const format = ib.indexFormat === 'uint32' ? 'uint32' : ib.indexFormat === 'uint16' ? 'uint16' : null;
      if (!format) {
        return false;
      }
      const elemBytes = format === 'uint32' ? 4 : 2;
      const offset = ib.offset || 0;
      if (!Number.isInteger(offset) || offset < 0 || offset % elemBytes !== 0) {
        return false;
      }
      const bufSize = typeof ib.buffer.size === 'number' && Number.isFinite(ib.buffer.size) ? ib.buffer.size : null;
      if (bufSize !== null) {
        const effectiveSize = ib.size !== undefined && ib.size !== null ? ib.size : bufSize - offset;
        if (offset >= bufSize || offset + effectiveSize > bufSize) {
          return false;
        }
        const neededBytes = (firstIndex + indexCount) * elemBytes;
        if (effectiveSize < neededBytes) {
          return false;
        }
      }
      try {
        if (ib.size !== undefined && ib.size !== null) {
          pass.setIndexBuffer(ib.buffer, format, offset, ib.size);
        } else {
          pass.setIndexBuffer(ib.buffer, format, offset);
        }
        return true;
      } catch (err) {
        this._stats.passErrors += 1;
        this._warn('setIndexBuffer failed', {
          error: String(err && err.message ? err.message : err),
        });
        return false;
      }
    }

    /**
     * Execute a non-indexed draw (`DrawInstanced`) with automatic fallback profile routing
     * and hardened bind-group / vertex-buffer guards.
     *
     * @param {object} pass GPURenderPassEncoder
     * @param {object} drawArgs `{ vertexCount, instanceCount?, firstVertex?, firstInstance?, ...overrides }`
     */
    executeDraw(pass, drawArgs = {}) {
      return this._executeDrawInternal(pass, drawArgs, false);
    }

    /**
     * Execute an indexed draw (`DrawIndexedInstanced`) with automatic fallback profile routing
     * and hardened index / bind-group / vertex-buffer guards.
     *
     * @param {object} pass GPURenderPassEncoder
     * @param {object} drawArgs `{ indexCount, instanceCount?, firstIndex?, baseVertex?, firstInstance?, ...overrides }`
     */
    executeDrawIndexed(pass, drawArgs = {}) {
      return this._executeDrawInternal(pass, drawArgs, true);
    }

    /**
     * Execute an indirect non-indexed draw (`drawIndirect`) with redundant-pipeline elision
     * and indirect buffer validation.
     *
     * @param {object} pass GPURenderPassEncoder
     * @param {object} drawArgs `{ indirectBuffer, indirectOffset?, ...overrides }`
     */
    executeDrawIndirect(pass, drawArgs = {}) {
      return this._executeDrawIndirectInternal(pass, drawArgs, false);
    }

    /**
     * Execute an indirect indexed draw (`drawIndexedIndirect`) with redundant-pipeline elision
     * and index / indirect buffer validation.
     *
     * @param {object} pass GPURenderPassEncoder
     * @param {object} drawArgs `{ indirectBuffer, indirectOffset?, ...overrides }`
     */
    executeDrawIndexedIndirect(pass, drawArgs = {}) {
      return this._executeDrawIndirectInternal(pass, drawArgs, true);
    }

    /**
     * Positional convenience wrapper for `executeDraw`.
     */
    draw(pass, vertexCount, instanceCount = 1, firstVertex = 0, firstInstance = 0) {
      return this.executeDraw(pass, {
        vertexCount,
        instanceCount,
        firstVertex,
        firstInstance,
      });
    }

    /**
     * Positional convenience wrapper for `executeDrawIndexed`.
     */
    drawIndexed(pass, indexCount, instanceCount = 1, firstIndex = 0, baseVertex = 0, firstInstance = 0) {
      return this.executeDrawIndexed(pass, {
        indexCount,
        instanceCount,
        firstIndex,
        baseVertex,
        firstInstance,
      });
    }

    /**
     * Positional convenience wrapper for `executeDrawIndirect`.
     */
    drawIndirect(pass, indirectBuffer, indirectOffset = 0) {
      return this.executeDrawIndirect(pass, {
        indirectBuffer,
        indirectOffset,
      });
    }

    /**
     * Positional convenience wrapper for `executeDrawIndexedIndirect`.
     */
    drawIndexedIndirect(pass, indirectBuffer, indirectOffset = 0) {
      return this.executeDrawIndexedIndirect(pass, {
        indirectBuffer,
        indirectOffset,
      });
    }

    _executeDrawIndirectInternal(pass, drawArgs, indexed) {
      const args = drawArgs && typeof drawArgs === 'object' ? drawArgs : {};
      const indirectBuffer = args.indirectBuffer !== undefined ? args.indirectBuffer : null;
      const indirectOffset = args.indirectOffset !== undefined ? args.indirectOffset : 0;

      const baseResult = {
        executed: false,
        skipped: false,
        reason: null,
        isFallback: false,
        profile: null,
        effectiveProfile: null,
        indexed,
        indirect: true,
        indirectOffset:
          Number.isInteger(indirectOffset) && indirectOffset >= 0 ? indirectOffset : 0,
        pipelineElided: false,
        vertexCount: 0,
        indexCount: 0,
        instanceCount: 0,
        substitutedCbv: false,
        substitutedTexture: false,
        substitutedSampler: false,
        substitutedVertexBuffer: false,
      };

      if (this._destroyed) {
        this._stats.skippedDraws += 1;
        return { ...baseResult, skipped: true, reason: 'executor-destroyed' };
      }

      const drawMethod = indexed ? 'drawIndexedIndirect' : 'drawIndirect';
      if (
        !pass ||
        typeof pass !== 'object' ||
        typeof pass.setPipeline !== 'function' ||
        typeof pass[drawMethod] !== 'function'
      ) {
        this._stats.skippedDraws += 1;
        return { ...baseResult, skipped: true, reason: 'invalid-render-pass' };
      }

      if (!isValidIndirectBuffer(indirectBuffer, indirectOffset, indexed)) {
        this._stats.skippedDraws += 1;
        return { ...baseResult, skipped: true, reason: 'invalid-indirect-buffer' };
      }

      const drawState = this.getDrawState({
        ...args,
        indexed,
        indirect: true,
        vertexCount: args.vertexCount !== undefined ? args.vertexCount : 0,
        indexCount: 0,
        instanceCount: 1,
      });

      const isFallback = this._needsFallbackRouting(drawState);
      baseResult.isFallback = isFallback;

      if (!isFallback) {
        const realPipeline =
          drawState.pipeline && drawState.pipeline.pipeline
            ? drawState.pipeline.pipeline
            : drawState.pipeline;

        try {
          const bound = this._bindPipelineOnPass(pass, realPipeline);
          baseResult.pipelineElided = !bound;
          this._bindNormalResources(pass);
          if (indexed) {
            if (!this._validateAndBindIndexBuffer(pass, 1, 0)) {
              this._stats.skippedDraws += 1;
              return { ...baseResult, skipped: true, reason: 'invalid-index-buffer' };
            }
            pass.drawIndexedIndirect(indirectBuffer, indirectOffset);
          } else {
            pass.drawIndirect(indirectBuffer, indirectOffset);
          }
          this._stats.totalDraws += 1;
          this._stats.normalDraws += 1;
          this._stats.indirectDraws += 1;
          return { ...baseResult, executed: true };
        } catch (err) {
          this._stats.passErrors += 1;
          this._stats.skippedDraws += 1;
          this._warn('Normal indirect draw execution failed', {
            error: String(err && err.message ? err.message : err),
          });
          return { ...baseResult, skipped: true, reason: 'pass-error' };
        }
      }

      const resolved = this.resolveFallbackShaders(drawState);
      if (!resolved) {
        this._stats.skippedDraws += 1;
        return { ...baseResult, skipped: true, reason: 'shader-resolve-failed' };
      }

      baseResult.profile = resolved.selectedProfile;
      baseResult.effectiveProfile = resolved.effectiveProfile;

      const fallbackPipeline = this._getOrCreateFallbackPipeline(resolved, drawState);
      if (!fallbackPipeline) {
        this._stats.skippedDraws += 1;
        return { ...baseResult, skipped: true, reason: 'pipeline-creation-failed' };
      }

      try {
        const bound = this._bindPipelineOnPass(pass, fallbackPipeline);
        baseResult.pipelineElided = !bound;
      } catch (err) {
        this._stats.passErrors += 1;
        this._stats.skippedDraws += 1;
        this._warn('pass.setPipeline failed on fallback indirect draw', {
          profile: resolved.effectiveProfile,
          error: String(err && err.message ? err.message : err),
        });
        return { ...baseResult, skipped: true, reason: 'pass-error' };
      }

      if (resolved.effectiveProfile === FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV) {
        const bgRes = this._bindFallbackGroupForCbv(pass, fallbackPipeline, drawState);
        baseResult.substitutedCbv = bgRes.substitutedCbv;
        if (!bgRes.ok) {
          this._stats.skippedDraws += 1;
          return { ...baseResult, skipped: true, reason: 'fallback-cbv-bind-failed' };
        }
      } else if (resolved.effectiveProfile === FALLBACK_PROFILE.TEXTURED_QUAD_24B) {
        const bgRes = this._bindFallbackGroupForTexturedQuad(pass, fallbackPipeline, drawState);
        baseResult.substitutedTexture = bgRes.substitutedTexture;
        baseResult.substitutedSampler = bgRes.substitutedSampler;
        if (!bgRes.ok) {
          this._stats.skippedDraws += 1;
          return { ...baseResult, skipped: true, reason: 'fallback-texture-bind-failed' };
        }

        const vbRes = this._bindFallbackVertexBuffer24B(pass, drawState, 1, 0);
        baseResult.substitutedVertexBuffer = vbRes.substitutedVertexBuffer;
        if (!vbRes.ok) {
          this._stats.skippedDraws += 1;
          return { ...baseResult, skipped: true, reason: 'fallback-vb-bind-failed' };
        }
      } else if (resolved.effectiveProfile === FALLBACK_PROFILE.SPLASH_RECT_24B) {
        const bgRes = this._bindFallbackGroupForSplashRect(pass, fallbackPipeline, drawState);
        baseResult.substitutedCbv = bgRes.substitutedCbv;
        baseResult.substitutedTexture = bgRes.substitutedTexture;
        baseResult.substitutedSampler = bgRes.substitutedSampler;
        if (!bgRes.ok) {
          this._stats.skippedDraws += 1;
          return { ...baseResult, skipped: true, reason: 'fallback-splash-bind-failed' };
        }

        const vbRes = this._bindFallbackVertexBufferSplashRect24B(pass, drawState, 1, 0);
        baseResult.substitutedVertexBuffer = vbRes.substitutedVertexBuffer;
        if (!vbRes.ok) {
          this._stats.skippedDraws += 1;
          return { ...baseResult, skipped: true, reason: 'fallback-vb-bind-failed' };
        }
      }

      try {
        if (indexed) {
          if (!this._validateAndBindIndexBuffer(pass, 1, 0)) {
            this._stats.skippedDraws += 1;
            return { ...baseResult, skipped: true, reason: 'invalid-index-buffer' };
          }
          pass.drawIndexedIndirect(indirectBuffer, indirectOffset);
        } else {
          pass.drawIndirect(indirectBuffer, indirectOffset);
        }
      } catch (err) {
        this._stats.passErrors += 1;
        this._stats.skippedDraws += 1;
        this._warn('Fallback indirect draw submission failed', {
          profile: resolved.effectiveProfile,
          error: String(err && err.message ? err.message : err),
        });
        return { ...baseResult, skipped: true, reason: 'pass-error' };
      }

      this._stats.totalDraws += 1;
      this._stats.fallbackDraws += 1;
      this._stats.indirectDraws += 1;
      const countedProfile = Object.prototype.hasOwnProperty.call(
        this._stats.fallbackProfileCounts,
        resolved.selectedProfile
      )
        ? resolved.selectedProfile
        : FALLBACK_PROFILE.DEFAULT;
      this._stats.fallbackProfileCounts[countedProfile] += 1;

      return { ...baseResult, executed: true };
    }

    _executeDrawInternal(pass, drawArgs, indexed) {
      const args = drawArgs && typeof drawArgs === 'object' ? drawArgs : {};
      if (args.indirect === true) {
        return this._executeDrawIndirectInternal(pass, args, indexed);
      }
      const count = indexed
        ? toNonNegativeInt(args.indexCount, 0)
        : toNonNegativeInt(args.vertexCount, 0);
      const instanceCount = toNonNegativeInt(
        args.instanceCount !== undefined ? args.instanceCount : 1,
        0
      );
      const firstVertex = toNonNegativeInt(args.firstVertex, 0);
      const firstIndex = toNonNegativeInt(args.firstIndex, 0);
      const baseVertex = Number.isInteger(args.baseVertex) ? args.baseVertex : 0;
      const firstInstance = toNonNegativeInt(args.firstInstance, 0);

      const baseResult = {
        executed: false,
        skipped: false,
        reason: null,
        isFallback: false,
        profile: null,
        effectiveProfile: null,
        indexed,
        indirect: false,
        pipelineElided: false,
        vertexCount: indexed ? 0 : count,
        indexCount: indexed ? count : 0,
        instanceCount,
        substitutedCbv: false,
        substitutedTexture: false,
        substitutedSampler: false,
        substitutedVertexBuffer: false,
      };

      if (this._destroyed) {
        this._stats.skippedDraws += 1;
        return { ...baseResult, skipped: true, reason: 'executor-destroyed' };
      }

      if (
        !pass ||
        typeof pass !== 'object' ||
        typeof pass.setPipeline !== 'function' ||
        typeof (indexed ? pass.drawIndexed : pass.draw) !== 'function'
      ) {
        this._stats.skippedDraws += 1;
        return { ...baseResult, skipped: true, reason: 'invalid-render-pass' };
      }

      if (count <= 0 || instanceCount <= 0) {
        this._stats.skippedDraws += 1;
        this._stats.zeroCountDrawsSkipped += 1;
        return { ...baseResult, skipped: true, reason: 'zero-draw-count' };
      }

      const drawState = this.getDrawState({
        ...args,
        indexed,
        vertexCount: indexed ? args.vertexCount || 0 : count,
        indexCount: indexed ? count : 0,
        instanceCount,
        firstVertex,
        firstIndex,
        baseVertex,
        firstInstance,
      });

      const isFallback = this._needsFallbackRouting(drawState);
      baseResult.isFallback = isFallback;

      if (!isFallback) {
        const realPipeline =
          drawState.pipeline && drawState.pipeline.pipeline
            ? drawState.pipeline.pipeline
            : drawState.pipeline;

        try {
          const bound = this._bindPipelineOnPass(pass, realPipeline);
          baseResult.pipelineElided = !bound;
          this._bindNormalResources(pass);
          if (indexed) {
            if (!this._validateAndBindIndexBuffer(pass, count, firstIndex)) {
              this._stats.skippedDraws += 1;
              return { ...baseResult, skipped: true, reason: 'invalid-index-buffer' };
            }
            pass.drawIndexed(count, instanceCount, firstIndex, baseVertex, firstInstance);
            this._stats.indexedDraws += 1;
          } else {
            pass.draw(count, instanceCount, firstVertex, firstInstance);
            this._stats.nonIndexedDraws += 1;
          }
          this._stats.totalDraws += 1;
          this._stats.normalDraws += 1;
          return { ...baseResult, executed: true };
        } catch (err) {
          this._stats.passErrors += 1;
          this._stats.skippedDraws += 1;
          this._warn('Normal draw execution failed', {
            error: String(err && err.message ? err.message : err),
          });
          return { ...baseResult, skipped: true, reason: 'pass-error' };
        }
      }

      // Fallback draw path: select G34 profile, resolve shaders, create/cache pipeline,
      // and harden bind-group & vertex-buffer bindings.
      const resolved = this.resolveFallbackShaders(drawState);
      if (!resolved) {
        this._stats.skippedDraws += 1;
        return { ...baseResult, skipped: true, reason: 'shader-resolve-failed' };
      }

      baseResult.profile = resolved.selectedProfile;
      baseResult.effectiveProfile = resolved.effectiveProfile;

      const fallbackPipeline = this._getOrCreateFallbackPipeline(resolved, drawState);
      if (!fallbackPipeline) {
        this._stats.skippedDraws += 1;
        return { ...baseResult, skipped: true, reason: 'pipeline-creation-failed' };
      }

      try {
        const bound = this._bindPipelineOnPass(pass, fallbackPipeline);
        baseResult.pipelineElided = !bound;
      } catch (err) {
        this._stats.passErrors += 1;
        this._stats.skippedDraws += 1;
        this._warn('pass.setPipeline failed on fallback draw', {
          profile: resolved.effectiveProfile,
          error: String(err && err.message ? err.message : err),
        });
        return { ...baseResult, skipped: true, reason: 'pass-error' };
      }

      let effectiveCount = count;
      let effectiveFirstVertex = firstVertex;

      if (resolved.effectiveProfile === FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV) {
        const bgRes = this._bindFallbackGroupForCbv(pass, fallbackPipeline, drawState);
        baseResult.substitutedCbv = bgRes.substitutedCbv;
        if (!bgRes.ok) {
          this._stats.skippedDraws += 1;
          return { ...baseResult, skipped: true, reason: 'fallback-cbv-bind-failed' };
        }
      } else if (resolved.effectiveProfile === FALLBACK_PROFILE.TEXTURED_QUAD_24B) {
        const bgRes = this._bindFallbackGroupForTexturedQuad(pass, fallbackPipeline, drawState);
        baseResult.substitutedTexture = bgRes.substitutedTexture;
        baseResult.substitutedSampler = bgRes.substitutedSampler;
        if (!bgRes.ok) {
          this._stats.skippedDraws += 1;
          return { ...baseResult, skipped: true, reason: 'fallback-texture-bind-failed' };
        }

        const vbRes = this._bindFallbackVertexBuffer24B(pass, drawState, count, firstVertex);
        baseResult.substitutedVertexBuffer = vbRes.substitutedVertexBuffer;
        if (!vbRes.ok) {
          this._stats.skippedDraws += 1;
          return { ...baseResult, skipped: true, reason: 'fallback-vb-bind-failed' };
        }
        effectiveCount = vbRes.effectiveVertexCount;
        effectiveFirstVertex = vbRes.effectiveFirstVertex;
      } else if (resolved.effectiveProfile === FALLBACK_PROFILE.SPLASH_RECT_24B) {
        const bgRes = this._bindFallbackGroupForSplashRect(pass, fallbackPipeline, drawState);
        baseResult.substitutedCbv = bgRes.substitutedCbv;
        baseResult.substitutedTexture = bgRes.substitutedTexture;
        baseResult.substitutedSampler = bgRes.substitutedSampler;
        if (!bgRes.ok) {
          this._stats.skippedDraws += 1;
          return { ...baseResult, skipped: true, reason: 'fallback-splash-bind-failed' };
        }

        const vbRes = this._bindFallbackVertexBufferSplashRect24B(
          pass,
          drawState,
          count,
          firstVertex
        );
        baseResult.substitutedVertexBuffer = vbRes.substitutedVertexBuffer;
        if (!vbRes.ok) {
          this._stats.skippedDraws += 1;
          return { ...baseResult, skipped: true, reason: 'fallback-vb-bind-failed' };
        }
        effectiveCount = vbRes.effectiveVertexCount;
        effectiveFirstVertex = vbRes.effectiveFirstVertex;
      }

      try {
        if (indexed) {
          if (!this._validateAndBindIndexBuffer(pass, effectiveCount, firstIndex)) {
            this._stats.skippedDraws += 1;
            return { ...baseResult, skipped: true, reason: 'invalid-index-buffer' };
          }
          pass.drawIndexed(effectiveCount, instanceCount, firstIndex, baseVertex, firstInstance);
          this._stats.indexedDraws += 1;
        } else {
          pass.draw(effectiveCount, instanceCount, effectiveFirstVertex, firstInstance);
          this._stats.nonIndexedDraws += 1;
        }
      } catch (err) {
        this._stats.passErrors += 1;
        this._stats.skippedDraws += 1;
        this._warn('Fallback draw submission failed', {
          profile: resolved.effectiveProfile,
          error: String(err && err.message ? err.message : err),
        });
        return { ...baseResult, skipped: true, reason: 'pass-error' };
      }

      this._stats.totalDraws += 1;
      this._stats.fallbackDraws += 1;
      const countedProfile = Object.prototype.hasOwnProperty.call(
        this._stats.fallbackProfileCounts,
        resolved.selectedProfile
      )
        ? resolved.selectedProfile
        : FALLBACK_PROFILE.DEFAULT;
      this._stats.fallbackProfileCounts[countedProfile] += 1;

      return {
        ...baseResult,
        executed: true,
        vertexCount: indexed ? 0 : effectiveCount,
        indexCount: indexed ? effectiveCount : 0,
      };
    }

    /**
     * Return a fresh JSON-serializable snapshot of draw executor telemetry,
     * including `fallbackProfileCounts` and per-kind draw counters.
     */
    getStats() {
      return {
        totalDraws: this._stats.totalDraws,
        normalDraws: this._stats.normalDraws,
        fallbackDraws: this._stats.fallbackDraws,
        indexedDraws: this._stats.indexedDraws,
        nonIndexedDraws: this._stats.nonIndexedDraws,
        indirectDraws: this._stats.indirectDraws,
        skippedDraws: this._stats.skippedDraws,
        zeroCountDrawsSkipped: this._stats.zeroCountDrawsSkipped,
        redundantPipelineSkips: this._stats.redundantPipelineSkips,
        fallbackProfileCounts: {
          [FALLBACK_PROFILE.DEFAULT]: this._stats.fallbackProfileCounts[FALLBACK_PROFILE.DEFAULT],
          [FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV]:
            this._stats.fallbackProfileCounts[FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV],
          [FALLBACK_PROFILE.TEXTURED_QUAD_24B]:
            this._stats.fallbackProfileCounts[FALLBACK_PROFILE.TEXTURED_QUAD_24B],
          [FALLBACK_PROFILE.SPLASH_RECT_24B]:
            this._stats.fallbackProfileCounts[FALLBACK_PROFILE.SPLASH_RECT_24B],
        },
        fallbackPipelineCacheHits: this._stats.fallbackPipelineCacheHits,
        fallbackPipelineCacheMisses: this._stats.fallbackPipelineCacheMisses,
        bindGroupGuards: this._stats.bindGroupGuards,
        invalidCbvSubstitutions: this._stats.invalidCbvSubstitutions,
        invalidTextureSubstitutions: this._stats.invalidTextureSubstitutions,
        invalidSamplerSubstitutions: this._stats.invalidSamplerSubstitutions,
        skippedBindGroupBindings: this._stats.skippedBindGroupBindings,
        invalidVertexBufferGuards: this._stats.invalidVertexBufferGuards,
        vertexBufferSubstitutions: this._stats.vertexBufferSubstitutions,
        skippedVertexBufferBindings: this._stats.skippedVertexBufferBindings,
        pipelineErrors: this._stats.pipelineErrors,
        bindGroupErrors: this._stats.bindGroupErrors,
        passErrors: this._stats.passErrors,
        legacyRouterFallbacks: this._stats.legacyRouterFallbacks,
      };
    }

    /**
     * Reset all draw and fallback telemetry counters.
     */
    resetStats() {
      this._stats = createZeroedStats();
    }

    _destroySyntheticResources() {
      const destroySafe = (res) => {
        if (res && typeof res.destroy === 'function') {
          try {
            res.destroy();
          } catch (_err) {
            // Ignore cleanup errors on already-lost devices.
          }
        }
      };
      destroySafe(this._fallbackCbvBuffer);
      destroySafe(this._fallbackTexture);
      destroySafe(this._fallbackVertexBuffer24B);
      destroySafe(this._fallbackSplashVertexBuffer24B);
      this._fallbackCbvBuffer = null;
      this._fallbackTexture = null;
      this._fallbackTextureView = null;
      this._fallbackSampler = null;
      this._fallbackVertexBuffer24B = null;
      this._fallbackSplashVertexBuffer24B = null;
    }

    /**
     * Release synthetic GPU resources and clear caches.
     */
    destroy() {
      this._destroyed = true;
      this._destroySyntheticResources();
      this._fallbackPipelineCache.clear();
      this._fallbackShaderModuleCache.clear();
      this.resetState();
    }
  }

  DrawExecutor.selectFallbackShaderProfile = selectFallbackShaderProfile;
  DrawExecutor.expandRectListVertices24B = expandRectListVertices24B;

  const exportedApi = Object.freeze({
    FALLBACK_PROFILE,
    TEXTURED_QUAD_VERTEX_STRIDE,
    FRAME1_CBV_MIN_BYTES,
    FALLBACK_VERTEX_LAYOUT_24B,
    FRAME1_SPLASH_VERTEX_LAYOUT_24B,
    expandRectListVertices24B,
    selectFallbackShaderProfile,
    DrawExecutor,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6DrawExecutor = exportedApi;
  } else if (root) {
    root.PlayGta6DrawExecutor = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
