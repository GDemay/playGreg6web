'use strict';

// Dedicated OffscreenCanvas WebGPU worker consuming binary GpuPacketHeader streams.
// Opcode contract: docs/gpu_commands.md & runtime/abi/playgta6_abi.h.
const OPCODES = Object.freeze({
  NOP: 0,
  BEGIN_FRAME: 1,
  CLEAR_COLOR: 2,
  CREATE_BUFFER: 3,
  WRITE_BUFFER: 4,
  CREATE_SHADER_MODULE: 5,
  DRAW: 6,
  PRESENT: 7,
  CREATE_RENDER_PIPELINE: 8,
  SET_PIPELINE: 9,
  SET_VERTEX_BUFFER: 10,
  DESTROY_RESOURCE: 11,
  CREATE_TEXTURE: 12,
  WRITE_TEXTURE: 13,
  CREATE_SAMPLER: 14,
  CREATE_BIND_GROUP: 15,
  SET_BIND_GROUP: 16,
  SET_INDEX_BUFFER: 17,
  DRAW_INDEXED: 18,
  CREATE_COMPUTE_PIPELINE: 19,
  SET_COMPUTE_PIPELINE: 20,
  DISPATCH_COMPUTE: 21,
  CREATE_BIND_GROUP_LAYOUT: 22,
  CREATE_PIPELINE_LAYOUT: 23,
  CREATE_RENDER_PIPELINE2: 24,
  CREATE_COMPUTE_PIPELINE2: 25,
  BEGIN_RENDER_PASS: 26,
  END_RENDER_PASS: 27,
  SET_VIEWPORT: 28,
  SET_SCISSOR_RECT: 29,
  SET_BLEND_CONSTANT: 30,
  SET_STENCIL_REFERENCE: 31,
  CREATE_BIND_GROUP2: 32,
  SET_BIND_GROUP2: 33,
  CREATE_SAMPLER2: 34,
  CREATE_TEXTURE2: 35,
});

const SHADER_STAGE = Object.freeze({
  VERTEX: 0,
  FRAGMENT: 1,
  COMPUTE: 2,
});

const PRIMITIVE_TOPOLOGY = Object.freeze({
  TRIANGLE_LIST: 0,
  TRIANGLE_STRIP: 1,
  LINE_LIST: 2,
  LINE_STRIP: 3,
  POINT_LIST: 4,
});

const TOPOLOGY_NAMES = Object.freeze([
  'triangle-list',
  'triangle-strip',
  'line-list',
  'line-strip',
  'point-list',
]);

const BUFFER_USAGE = Object.freeze({
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
});

const TEXTURE_USAGE = Object.freeze({
  COPY_SRC: 0x0001,
  COPY_DST: 0x0002,
  TEXTURE_BINDING: 0x0004,
  STORAGE_BINDING: 0x0008,
  RENDER_ATTACHMENT: 0x0010,
});

const TEXTURE_FORMAT = Object.freeze({
  RGBA8_UNORM: 0,
  BGRA8_UNORM: 1,
  RGBA16_FLOAT: 2,
  DEPTH24_PLUS: 3,
  DEPTH32_FLOAT: 4,
  BC7_RGBA_UNORM: 5,
});

const TEXTURE_FORMAT_NAMES = Object.freeze([
  'rgba8unorm',
  'bgra8unorm',
  'rgba16float',
  'depth24plus',
  'depth32float',
  'bc7-rgba-unorm',
]);

const TEXTURE_DIMENSION = Object.freeze({
  DIM_1D: 0,
  DIM_2D: 1,
  DIM_3D: 2,
});

const TEXTURE_DIMENSION_NAMES = Object.freeze(['1d', '2d', '3d']);

const ADDRESS_MODE = Object.freeze({
  CLAMP_TO_EDGE: 0,
  REPEAT: 1,
  MIRROR_REPEAT: 2,
});

const ADDRESS_MODE_NAMES = Object.freeze(['clamp-to-edge', 'repeat', 'mirror-repeat']);

const FILTER_MODE = Object.freeze({
  NEAREST: 0,
  LINEAR: 1,
});

const FILTER_MODE_NAMES = Object.freeze(['nearest', 'linear']);

const BIND_RESOURCE_TYPE = Object.freeze({
  BUFFER: 0,
  TEXTURE_VIEW: 1,
  SAMPLER: 2,
});

const INDEX_FORMAT = Object.freeze({
  UINT16: 0,
  UINT32: 1,
});

const INDEX_FORMAT_NAMES = Object.freeze(['uint16', 'uint32']);

const DEFAULT_CLEAR_COLOR = Object.freeze({ r: 0.05, g: 0.07, b: 0.11, a: 1.0 });

const DEFAULT_MAX_VALIDATION_ERROR_HISTORY = 32;
const MAX_VALIDATION_ERROR_MESSAGE_LENGTH = 512;

const FALLBACK_PROFILE = Object.freeze({
  DEFAULT: 'default',
  FULLSCREEN_TRIANGLE_CBV: 'fullscreen_triangle_cbv',
  TEXTURED_QUAD_24B: 'textured_quad_24b',
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
});

const TEXTURED_QUAD_24B_STRIDE = 24;

const FALLBACK_VERTEX_LAYOUT_24B = Object.freeze({
  arrayStride: TEXTURED_QUAD_24B_STRIDE,
  stepMode: 'vertex',
  attributes: Object.freeze([
    Object.freeze({ shaderLocation: 0, offset: 0, format: 'float32x4' }),
    Object.freeze({ shaderLocation: 1, offset: 16, format: 'float32x2' }),
  ]),
});

const FNV1A_64_OFFSET_BASIS = 0xcbf29ce484222325n;
const FNV1A_64_PRIME = 0x00000100000001b3n;
const U64_MASK = 0xffffffffffffffffn;

function defaultNowMs() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

function classifyGpuErrorType(raw) {
  if (typeof raw !== 'string') {
    return null;
  }
  const normalized = raw.trim().toLowerCase();
  if (!normalized) {
    return null;
  }
  if (
    normalized === 'out-of-memory' ||
    normalized === 'out_of_memory' ||
    normalized === 'outofmemory' ||
    normalized === 'oom' ||
    normalized.includes('outofmemory') ||
    normalized.includes('out-of-memory') ||
    normalized.includes('out_of_memory')
  ) {
    return 'out-of-memory';
  }
  if (normalized === 'validation' || normalized.includes('validation')) {
    return 'validation';
  }
  if (normalized === 'internal' || normalized.includes('internal')) {
    return 'internal';
  }
  return null;
}

function normalizeValidationErrorType(error, event) {
  if (error && typeof error === 'object') {
    if (typeof GPUOutOfMemoryError === 'function' && error instanceof GPUOutOfMemoryError) {
      return 'out-of-memory';
    }
    if (typeof GPUValidationError === 'function' && error instanceof GPUValidationError) {
      return 'validation';
    }
    if (typeof GPUInternalError === 'function' && error instanceof GPUInternalError) {
      return 'internal';
    }
    const fromType = classifyGpuErrorType(error.type);
    if (fromType) {
      return fromType;
    }
    const fromName = classifyGpuErrorType(error.name);
    if (fromName) {
      return fromName;
    }
    if (error.constructor && typeof error.constructor.name === 'string') {
      const fromCtor = classifyGpuErrorType(error.constructor.name);
      if (fromCtor) {
        return fromCtor;
      }
    }
  }
  if (event && typeof event === 'object') {
    const fromEventType = classifyGpuErrorType(event.errorType);
    if (fromEventType) {
      return fromEventType;
    }
  }
  return 'unknown';
}

function sanitizeValidationErrorMessage(rawMessage) {
  let msg;
  if (typeof rawMessage === 'string') {
    msg = rawMessage;
  } else if (rawMessage === null || rawMessage === undefined) {
    return '';
  } else {
    msg = String(rawMessage);
  }

  msg = msg.replace(/[\r\n\t]+/g, ' ').replace(/[\x00-\x1F\x7F]+/g, '');

  // Strip file:// URLs and host filesystem paths (Windows drive, UNC, and Unix paths).
  msg = msg.replace(/\bfile:\/\/[^\s"'`<>)\]]+/gi, '<path>');
  msg = msg.replace(/\\\\[^\s\\"'`<>)\]]+\\[^\s"'`<>)\]]+/g, '<path>');
  msg = msg.replace(/(["'`])[A-Za-z]:[\\\/][^"'`\r\n]+\1/g, '$1<path>$1');
  msg = msg.replace(/\b[A-Za-z]:[\\\/][^\s"'`<>)\],;]+/g, '<path>');
  msg = msg.replace(
    /(["'`])\/(?:Users|home|var|tmp|private|etc|opt|usr|Volumes|mnt|root|Library|Applications|System)\/[^"'`\r\n]+\1/gi,
    '$1<path>$1'
  );
  msg = msg.replace(
    /(^|[\s"'`(\[=,:])\/(?:[A-Za-z0-9._-]+\/)+[A-Za-z0-9._~+-]+/g,
    '$1<path>'
  );

  // Strip long embedded buffers (data URIs, byte arrays, hex dumps, base64 blobs).
  msg = msg.replace(/\bdata:[^\s;,'"`<>]+;base64,[A-Za-z0-9+/=]{16,}/gi, '<buffer>');
  msg = msg.replace(
    /(?:\b(?:0x)?[0-9a-fA-F]{1,4}\s*,\s*){16,}(?:0x)?[0-9a-fA-F]{1,4}\b/g,
    '<buffer>'
  );
  msg = msg.replace(/(?:\b[0-9a-fA-F]{2}\s+){16,}[0-9a-fA-F]{2}\b/g, '<buffer>');
  msg = msg.replace(/\b(?:0x)?[0-9a-fA-F]{48,}\b/g, '<buffer>');
  msg = msg.replace(
    /(^|[\s"'`(\[=,:])[A-Za-z0-9+/]{64,}={0,2}(?=$|[\s"'`)\]=,.;:])/g,
    '$1<buffer>'
  );

  msg = msg.replace(/\s{2,}/g, ' ').trim();

  if (msg.length > MAX_VALIDATION_ERROR_MESSAGE_LENGTH) {
    msg = `${msg.slice(0, MAX_VALIDATION_ERROR_MESSAGE_LENGTH - 3)}...`;
  }
  return msg;
}

function normalizeFallbackProfileName(profile) {
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

function iterateDrawCollectionValues(collection) {
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

function hasAnyTruthyResource(collection) {
  const values = iterateDrawCollectionValues(collection);
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

function inspectBindGroupCollection(bindGroups) {
  let hasCbv = false;
  let hasTexture = false;
  let hasSampler = false;

  const groups = iterateDrawCollectionValues(bindGroups);
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
    if (group.meta && typeof group.meta === 'object') {
      if (group.meta.hasCbv === true) hasCbv = true;
      if (group.meta.hasTexture === true) hasTexture = true;
      if (group.meta.hasSampler === true) hasSampler = true;
    }

    const entries = Array.isArray(group.entries)
      ? group.entries
      : group.meta && Array.isArray(group.meta.entries)
        ? group.meta.entries
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
        resourceType === BIND_RESOURCE_TYPE.BUFFER ||
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
        resourceType === BIND_RESOURCE_TYPE.TEXTURE_VIEW ||
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
        resourceType === BIND_RESOURCE_TYPE.SAMPLER ||
        resourceType === 'sampler' ||
        entry.sampler
      ) {
        hasSampler = true;
      }
    }
  }

  return { hasCbv, hasTexture, hasSampler };
}

function hasVertexStrideInState(drawState, targetStride) {
  if (
    drawState.vertexStride === targetStride ||
    drawState.arrayStride === targetStride ||
    drawState.vertexBufferStride === targetStride
  ) {
    return true;
  }

  const vbValues = iterateDrawCollectionValues(drawState.vertexBuffers);
  for (let i = 0; i < vbValues.length; i += 1) {
    const vb = vbValues[i];
    if (!vb || typeof vb !== 'object') {
      continue;
    }
    if (vb.stride === targetStride || vb.arrayStride === targetStride) {
      return true;
    }
    if (
      vb.buffer &&
      typeof vb.buffer === 'object' &&
      (vb.buffer.stride === targetStride || vb.buffer.arrayStride === targetStride)
    ) {
      return true;
    }
  }

  const layoutValues = iterateDrawCollectionValues(
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

/**
 * Detect the real Frame-1 draw pattern and select the matching G34 fallback WGSL profile:
 * - `vertexCount === 3` + bound CBV and no bound texture -> `'fullscreen_triangle_cbv'`
 * - 24-byte-stride VB + bound texture -> `'textured_quad_24b'`
 * - Otherwise -> `'default'`
 *
 * @param {object | null | undefined} drawState
 * @returns {'default' | 'fullscreen_triangle_cbv' | 'textured_quad_24b'}
 */
function selectFallbackShaderProfile(drawState) {
  if (!drawState || typeof drawState !== 'object') {
    return FALLBACK_PROFILE.DEFAULT;
  }

  const rawVertexCount = drawState.vertexCount;
  const vertexCount =
    typeof rawVertexCount === 'number' && Number.isFinite(rawVertexCount) && rawVertexCount >= 0
      ? Math.trunc(rawVertexCount)
      : 0;
  const bgInfo = inspectBindGroupCollection(drawState.bindGroups);

  const hasBoundCbv =
    drawState.hasBoundCbv === true ||
    drawState.hasCbv === true ||
    (typeof drawState.cbvCount === 'number' && drawState.cbvCount > 0) ||
    Boolean(drawState.cbv || drawState.cbvBuffer || drawState.uniformBuffer) ||
    hasAnyTruthyResource(drawState.cbvs) ||
    hasAnyTruthyResource(drawState.uniformBuffers) ||
    bgInfo.hasCbv;

  const hasBoundTexture =
    drawState.hasBoundTexture === true ||
    drawState.hasTexture === true ||
    drawState.hasSampledTexture === true ||
    drawState.hasBoundSampledTexture === true ||
    drawState.hasBoundSrv === true ||
    (typeof drawState.textureCount === 'number' && drawState.textureCount > 0) ||
    (typeof drawState.srvCount === 'number' && drawState.srvCount > 0) ||
    Boolean(drawState.texture || drawState.textureView || drawState.srv) ||
    hasAnyTruthyResource(drawState.textures) ||
    hasAnyTruthyResource(drawState.textureViews) ||
    hasAnyTruthyResource(drawState.srvs) ||
    bgInfo.hasTexture;

  if (vertexCount === 3 && hasBoundCbv && !hasBoundTexture) {
    return FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV;
  }

  if (hasVertexStrideInState(drawState, TEXTURED_QUAD_24B_STRIDE) && hasBoundTexture) {
    return FALLBACK_PROFILE.TEXTURED_QUAD_24B;
  }

  return FALLBACK_PROFILE.DEFAULT;
}

function normalizeFallbackRouterResult(raw, lookupHashHex, stageName) {
  if (raw === null || raw === undefined) {
    return null;
  }
  if (typeof raw === 'string' && raw.length > 0) {
    return {
      wgsl: raw,
      source: 'builtin',
      requestedHash: lookupHashHex,
      resolvedHash: null,
      stage: stageName,
      entryPoint: null,
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
      typeof raw.source === 'string' && !raw.source.includes('fn ') ? raw.source : 'fallback';
    return {
      wgsl,
      source,
      requestedHash: raw.requestedHash || lookupHashHex,
      resolvedHash: raw.resolvedHash !== undefined ? raw.resolvedHash : null,
      stage: raw.stage || stageName,
      entryPoint: typeof raw.entryPoint === 'string' ? raw.entryPoint : null,
      profile:
        typeof raw.profile === 'string' && raw.profile.length > 0
          ? normalizeFallbackProfileName(raw.profile)
          : null,
    };
  }
  return null;
}

/**
 * Call `shaderFallbackRouter.resolve(hash, stage, { profile })` while staying 100%
 * backwards-compatible with legacy 2-arg routers (including strict mocks that reject a 3rd arg).
 */
function callFallbackRouterResolve(router, lookupHashHex, stageName, profile) {
  if (!router || typeof router.resolve !== 'function') {
    return null;
  }
  const canonicalProfile = normalizeFallbackProfileName(profile);
  if (canonicalProfile !== FALLBACK_PROFILE.DEFAULT) {
    try {
      const withProfile = router.resolve(lookupHashHex, stageName, { profile: canonicalProfile });
      const normalized = normalizeFallbackRouterResult(withProfile, lookupHashHex, stageName);
      if (normalized) {
        return normalized;
      }
    } catch (_) {
      // Legacy 2-arg router threw on the 3rd options argument — retry with 2 args.
    }
  }
  const legacy = router.resolve(lookupHashHex, stageName);
  return normalizeFallbackRouterResult(legacy, lookupHashHex, stageName);
}

const TEXTURE_FORMAT_BPP = Object.freeze({
  rgba8unorm: 4,
  bgra8unorm: 4,
  rgba16float: 8,
  depth24plus: 4,
  depth32float: 4,
});

function fnv1a64Hex(text) {
  const bytes = new TextEncoder().encode(String(text));
  let hash = FNV1A_64_OFFSET_BASIS;
  for (let i = 0; i < bytes.length; i += 1) {
    hash ^= BigInt(bytes[i]);
    hash = (hash * FNV1A_64_PRIME) & U64_MASK;
  }
  return `0x${hash.toString(16).padStart(16, '0')}`;
}

function estimateTextureBytes(width, height, depthOrArrayLayers, mipLevelCount, format, dimension) {
  const w0 = Math.max(1, width | 0);
  const h0 = Math.max(1, height | 0);
  const d0 = Math.max(1, depthOrArrayLayers | 0);
  const mips = Math.max(1, mipLevelCount | 0);
  const is1D = dimension === '1d';
  const is3D = dimension === '3d';
  const isBc7 = format === 'bc7-rgba-unorm';
  const bpp = TEXTURE_FORMAT_BPP[format] || 4;

  let total = 0;
  for (let mip = 0; mip < mips; mip += 1) {
    const scale = mip >= 31 ? Infinity : Math.pow(2, mip);
    const mw = scale === Infinity ? 1 : Math.max(1, Math.floor(w0 / scale));
    const mh = is1D ? 1 : scale === Infinity ? 1 : Math.max(1, Math.floor(h0 / scale));
    const md = is3D ? (scale === Infinity ? 1 : Math.max(1, Math.floor(d0 / scale))) : d0;
    if (isBc7) {
      total += Math.ceil(mw / 4) * Math.ceil(mh / 4) * 16 * md;
    } else {
      total += mw * mh * md * bpp;
    }
  }
  return total;
}

function extractWgslEntryPoint(code, stage) {
  if (typeof code !== 'string' || code.length === 0) {
    return null;
  }
  const attr =
    stage === SHADER_STAGE.VERTEX
      ? 'vertex'
      : stage === SHADER_STAGE.FRAGMENT
        ? 'fragment'
        : stage === SHADER_STAGE.COMPUTE
          ? 'compute'
          : null;
  if (attr) {
    const re = new RegExp(`@${attr}\\b[\\s\\S]*?\\bfn\\s+([A-Za-z_][A-Za-z0-9_]*)`);
    const m = re.exec(code);
    if (m) return m[1];
  }
  const fallback = /\bfn\s+([A-Za-z_][A-Za-z0-9_]*)/.exec(code);
  return fallback ? fallback[1] : null;
}

function stageCodeToName(stage) {
  if (stage === SHADER_STAGE.VERTEX) return 'vertex';
  if (stage === SHADER_STAGE.FRAGMENT) return 'fragment';
  if (stage === SHADER_STAGE.COMPUTE) return 'compute';
  return null;
}

function resolveSubsystemCtor(injectedCtor, globalNamespace, exportName, scriptRelPath) {
  if (typeof injectedCtor === 'function') {
    return injectedCtor;
  }
  if (
    typeof globalThis !== 'undefined' &&
    globalThis[globalNamespace] &&
    typeof globalThis[globalNamespace][exportName] === 'function'
  ) {
    return globalThis[globalNamespace][exportName];
  }
  if (typeof importScripts === 'function') {
    try {
      importScripts(scriptRelPath);
      if (
        typeof globalThis !== 'undefined' &&
        globalThis[globalNamespace] &&
        typeof globalThis[globalNamespace][exportName] === 'function'
      ) {
        return globalThis[globalNamespace][exportName];
      }
    } catch (_) {
      // Optional helper script unavailable in this worker environment.
    }
  }
  if (typeof require === 'function') {
    try {
      const mod = require(scriptRelPath);
      if (mod && typeof mod[exportName] === 'function') {
        return mod[exportName];
      }
    } catch (_) {
      // Optional helper module unavailable.
    }
  }
  return null;
}

const WGPK_MAGIC_BYTES = Object.freeze([0x57, 0x47, 0x50, 0x4b]);
const WGPK_VERSION = 1;
const WGPK_HEADER_BYTES = 16;
const WGPK_ENTRY_BYTES = 20;
const WGPK_STAGE_NAMES = Object.freeze({ 1: 'vertex', 2: 'fragment', 3: 'compute' });

function normalizeShaderHashBigInt(hash) {
  if (typeof hash === 'bigint') {
    return hash >= 0n && hash <= U64_MASK ? hash : null;
  }
  if (typeof hash === 'number') {
    return Number.isInteger(hash) && Number.isSafeInteger(hash) && hash >= 0 ? BigInt(hash) : null;
  }
  if (typeof hash === 'string') {
    const trimmed = hash.trim();
    const rawHex =
      trimmed.startsWith('0x') || trimmed.startsWith('0X') ? trimmed.slice(2) : trimmed;
    if (/^[0-9a-fA-F]{1,16}$/.test(rawHex)) {
      return BigInt(`0x${rawHex}`);
    }
  }
  return null;
}

function formatShaderHashHex(hashBigInt) {
  return `0x${hashBigInt.toString(16).padStart(16, '0')}`;
}

function parseHashOnlyShaderPayload(trimmed) {
  if (typeof trimmed !== 'string' || trimmed.length === 0) {
    return null;
  }
  const prefixed = /^(?:hash|wgpk)\s*:\s*(?:0[xX])?([0-9a-fA-F]{1,16})$/i.exec(trimmed);
  if (prefixed) {
    return formatShaderHashHex(BigInt(`0x${prefixed[1]}`));
  }
  const hex0x = /^0[xX]([0-9a-fA-F]{1,16})$/.exec(trimmed);
  if (hex0x) {
    return formatShaderHashHex(BigInt(`0x${hex0x[1]}`));
  }
  const bare64 = /^([0-9a-fA-F]{16})$/.exec(trimmed);
  if (bare64) {
    return formatShaderHashHex(BigInt(`0x${bare64[1]}`));
  }
  return null;
}

function unwrapPackLookupResult(raw) {
  if (raw === undefined || raw === null) {
    return null;
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
  return null;
}

function lookupFromPackCache(packCache, hash) {
  if (!packCache) {
    return null;
  }
  const big = normalizeShaderHashBigInt(hash);
  if (big === null) {
    return null;
  }
  const hex = formatShaderHashHex(big);
  const keys = [big, hex];
  const tryCall = (fn, key) => {
    if (typeof fn !== 'function') {
      return null;
    }
    try {
      return unwrapPackLookupResult(fn.call(packCache, key));
    } catch (_) {
      return null;
    }
  };
  for (let i = 0; i < keys.length; i += 1) {
    const key = keys[i];
    const hit =
      tryCall(packCache.getWgsl, key) ||
      tryCall(packCache.getSource, key) ||
      tryCall(packCache.getShaderSource, key) ||
      tryCall(packCache.lookup, key) ||
      tryCall(packCache.get, key);
    if (hit) {
      return hit;
    }
  }
  return null;
}

function loadWgslPackIntoCache(packCache, buffer, packName) {
  if (!packCache) {
    return null;
  }
  if (typeof packCache.loadFromBuffer === 'function') {
    return packCache.loadFromBuffer(buffer, packName);
  }
  if (typeof packCache.loadPackBuffer === 'function') {
    return packCache.loadPackBuffer(buffer, packName);
  }
  if (typeof packCache.parse === 'function') {
    return packCache.parse(buffer, packName);
  }
  return null;
}

/**
 * Classic-worker-compatible WGPK v1 (.wgsl.pack) parser & in-memory cache.
 * Binary layout matches shaders/src/pack.rs and web/workers/wgsl_pack_loader.js byte-for-byte.
 */
class InlineWgslPackCache {
  constructor() {
    this._packs = new Map();
    this._entries = new Map();
    this._nextPackId = 0;
    this._strictDecoder = new TextDecoder('utf-8', { fatal: true });
  }

  _toBytes(buffer) {
    if (buffer instanceof Uint8Array) {
      return buffer;
    }
    if (ArrayBuffer.isView(buffer)) {
      return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    }
    if (
      buffer instanceof ArrayBuffer ||
      (typeof SharedArrayBuffer !== 'undefined' && buffer instanceof SharedArrayBuffer)
    ) {
      return new Uint8Array(buffer);
    }
    throw new TypeError('WGPK buffer must be an ArrayBuffer, SharedArrayBuffer, or ArrayBufferView');
  }

  loadPackBuffer(buffer, packName) {
    const bytes = this._toBytes(buffer);
    if (bytes.byteLength < WGPK_HEADER_BYTES) {
      throw new Error(
        `WGPK pack truncated: ${bytes.byteLength} bytes (minimum header: ${WGPK_HEADER_BYTES})`
      );
    }
    if (
      bytes[0] !== WGPK_MAGIC_BYTES[0] ||
      bytes[1] !== WGPK_MAGIC_BYTES[1] ||
      bytes[2] !== WGPK_MAGIC_BYTES[2] ||
      bytes[3] !== WGPK_MAGIC_BYTES[3]
    ) {
      throw new Error('Invalid WGPK pack magic (expected "WGPK")');
    }

    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const version = view.getUint32(4, true);
    if (version !== WGPK_VERSION) {
      throw new Error(`Unsupported WGPK pack version ${version} (expected ${WGPK_VERSION})`);
    }
    const entryCount = view.getUint32(8, true);
    const flags = view.getUint32(12, true);
    const dataStart = WGPK_HEADER_BYTES + entryCount * WGPK_ENTRY_BYTES;
    if (bytes.byteLength < dataStart) {
      throw new Error(
        `WGPK pack truncated: ${bytes.byteLength} bytes (entry table requires ${dataStart})`
      );
    }

    const parsedEntries = new Array(entryCount);
    const ranges = [];
    const isShared =
      typeof SharedArrayBuffer !== 'undefined' && bytes.buffer instanceof SharedArrayBuffer;

    for (let idx = 0; idx < entryCount; idx += 1) {
      const base = WGPK_HEADER_BYTES + idx * WGPK_ENTRY_BYTES;
      const hash = view.getBigUint64(base, true);
      const stageCode = bytes[base + 8];
      const stage = WGPK_STAGE_NAMES[stageCode];
      if (!stage) {
        throw new Error(`Invalid WGPK shader stage value ${stageCode} at entry ${idx}`);
      }
      const offset = view.getUint32(base + 12, true);
      const length = view.getUint32(base + 16, true);
      const end = offset + length;
      if (offset < dataStart || end > bytes.byteLength) {
        throw new Error(`WGPK entry ${idx} payload [${offset}..+${length}] is out of bounds`);
      }
      if (length > 0) {
        ranges.push({ start: offset, end, index: idx });
      }
      const slice = bytes.subarray(offset, end);
      const wgsl = this._strictDecoder.decode(isShared ? slice.slice() : slice);
      parsedEntries[idx] = {
        hash,
        hashHex: formatShaderHashHex(hash),
        stage,
        stageCode,
        offset,
        length,
        wgsl,
      };
    }

    if (ranges.length > 1) {
      ranges.sort((a, b) => a.start - b.start || a.end - b.end);
      for (let i = 1; i < ranges.length; i += 1) {
        if (ranges[i].start < ranges[i - 1].end) {
          throw new Error('WGPK overlapping payload ranges detected');
        }
      }
    }

    const name =
      typeof packName === 'string' && packName.length > 0
        ? packName
        : `pack-${this._nextPackId++}`;

    if (this._packs.has(name)) {
      for (const [key, rec] of this._entries.entries()) {
        if (rec.packName === name) {
          this._entries.delete(key);
        }
      }
    }

    const packInfo = {
      name,
      version,
      flags,
      entryCount,
      byteLength: bytes.byteLength,
      entries: parsedEntries,
    };
    this._packs.set(name, packInfo);
    for (let i = 0; i < parsedEntries.length; i += 1) {
      const entry = parsedEntries[i];
      this._entries.set(entry.hash, { entry, packName: name });
    }
    return packInfo;
  }

  loadFromBuffer(buffer, packName) {
    return this.loadPackBuffer(buffer, packName);
  }

  parse(buffer, packName) {
    return this.loadPackBuffer(buffer, packName);
  }

  hasShader(hash) {
    const key = normalizeShaderHashBigInt(hash);
    return key !== null && this._entries.has(key);
  }

  has(hash) {
    return this.hasShader(hash);
  }

  getByHash(hash) {
    const key = normalizeShaderHashBigInt(hash);
    if (key === null) return undefined;
    const rec = this._entries.get(key);
    return rec ? rec.entry : undefined;
  }

  getShaderSource(hash) {
    const entry = this.getByHash(hash);
    return entry ? entry.wgsl : undefined;
  }

  getSource(hash) {
    return this.getShaderSource(hash);
  }

  getWgsl(hash) {
    return this.getShaderSource(hash);
  }

  get(hash) {
    return this.getShaderSource(hash);
  }

  clear() {
    this._packs.clear();
    this._entries.clear();
  }

  getStats() {
    let totalPackBytes = 0;
    for (const pack of this._packs.values()) {
      totalPackBytes += pack.byteLength;
    }
    return {
      packCount: this._packs.size,
      shaderCount: this._entries.size,
      totalPackBytes,
    };
  }
}

function deviceSupportsTimestampQuery(device) {
  if (!device || typeof device !== 'object') {
    return false;
  }
  const features = device.features;
  if (!features || typeof features.has !== 'function') {
    return false;
  }
  try {
    if (!features.has('timestamp-query')) {
      return false;
    }
  } catch (_) {
    return false;
  }
  return typeof device.createQuerySet === 'function' && typeof device.createBuffer === 'function';
}

function resolveGpuRingBuffer(injectedCtor) {
  if (typeof injectedCtor === 'function') {
    return injectedCtor;
  }
  if (
    typeof globalThis !== 'undefined' &&
    globalThis.PlayGta6GpuRing &&
    typeof globalThis.PlayGta6GpuRing.GpuRingBuffer === 'function'
  ) {
    return globalThis.PlayGta6GpuRing.GpuRingBuffer;
  }
  if (typeof importScripts === 'function') {
    importScripts('./gpu_ring.js');
    if (
      typeof globalThis !== 'undefined' &&
      globalThis.PlayGta6GpuRing &&
      typeof globalThis.PlayGta6GpuRing.GpuRingBuffer === 'function'
    ) {
      return globalThis.PlayGta6GpuRing.GpuRingBuffer;
    }
  }
  if (typeof require === 'function') {
    const mod = require('./gpu_ring.js');
    if (mod && typeof mod.GpuRingBuffer === 'function') {
      return mod.GpuRingBuffer;
    }
  }
  throw new Error('GpuRingBuffer unavailable; load ./gpu_ring.js or inject GpuRingBuffer');
}

function normalizeSplashPixelFormat(format, payload) {
  if (typeof format === 'string' && format.trim().length > 0) {
    const key = format.trim().toLowerCase();
    if (key === 'rgba8' || key === 'rgba' || key === 'rgba8unorm') {
      return 'rgba8';
    }
    if (key === 'yuv420p' || key === 'yuv420') {
      return 'yuv420p';
    }
    if (key === 'yuva420p' || key === 'yuva420') {
      return 'yuva420p';
    }
    return key;
  }
  const planes = payload && payload.planes && typeof payload.planes === 'object' ? payload.planes : null;
  const hasYuv =
    (planes && (planes.y !== undefined || planes.u !== undefined || planes.v !== undefined)) ||
    (payload && (payload.y !== undefined || payload.u !== undefined || payload.v !== undefined));
  if (hasYuv) {
    const hasAlpha =
      (planes && planes.a !== undefined && planes.a !== null) ||
      (payload && payload.a !== undefined && payload.a !== null);
    return hasAlpha ? 'yuva420p' : 'yuv420p';
  }
  return 'rgba8';
}

function isExternalImageLike(obj) {
  if (!obj || typeof obj !== 'object') {
    return false;
  }
  if (typeof ImageBitmap !== 'undefined' && obj instanceof ImageBitmap) {
    return true;
  }
  if (
    typeof obj.close === 'function' &&
    typeof obj.width === 'number' &&
    typeof obj.height === 'number' &&
    obj.planes === undefined &&
    obj.rgba === undefined &&
    obj.y === undefined
  ) {
    return true;
  }
  return false;
}

class WgpuCommandProcessor {
  constructor(device, ctx, options = {}) {
    this.device = device;
    this.ctx = ctx;
    this.format = options.format || 'bgra8unorm';
    this.buffers = new Map();
    this.textures = new Map();
    this.textureViews = new Map();
    this.samplers = new Map();
    this.bindGroups = new Map();
    this._bindGroupKinds = new Map();
    this.shaderModules = new Map();
    this.pipelines = new Map();
    this.computePipelines = new Map();
    this._pipelineBindGroupLayouts = new Map();
    this._computePipelineBindGroupLayouts = new Map();
    this._shaderReflectionKeys = new Map();
    this.frameCount = 0;
    this.ring = null;
    this._GpuRingBuffer = typeof options.GpuRingBuffer === 'function' ? options.GpuRingBuffer : null;
    this._decoder = new TextDecoder('utf-8');
    this._destroyed = false;

    // Viewport & splash/first-present tracking
    this._viewportWidth =
      typeof options.width === 'number' && Number.isFinite(options.width) && options.width > 0
        ? Math.floor(options.width)
        : 0;
    this._viewportHeight =
      typeof options.height === 'number' && Number.isFinite(options.height) && options.height > 0
        ? Math.floor(options.height)
        : 0;
    this._onFirstPresent =
      typeof options.onFirstPresent === 'function' ? options.onFirstPresent : null;
    this._firstPresentNotified = false;
    this._splashFramesPresented = 0;
    this._lastSplashPresentMs = null;

    // Telemetry counters
    this._packetCount = 0;
    this._drawCallCount = 0;
    this._computeDispatchCount = 0;
    this._bytesProcessed = 0;

    // Pending staging upload encoder & frame lifecycle tracking
    this._pendingEncoder = null;
    this._hasPendingUploadCommands = false;
    this._frameBegunForIndex = -1;
    this._lastPassTimings = null;
    this._pendingTimerReadbacks = new Set();

    // Optional WebGpuPipelineCache for reflection layout deduplication
    this.pipelineCache = options.pipelineCache || null;

    // Saved subsystem options / constructors for lazy construction where supported
    this._reflectionRegistryOptions = options.reflectionRegistryOptions || null;
    this._ReflectionRegistryCtor = options.ReflectionRegistry || null;
    this._rtPoolOptions = options.rtPoolOptions || null;
    this._RenderTargetPoolCtor = options.RenderTargetPool || null;
    this._rtPoolExplicitlyDisabled = options.enableRtPool === false;

    // 1. GpuStagingPool (G14)
    this._ownsStagingPool = false;
    this.stagingPool = null;
    if (options.stagingPool && typeof options.stagingPool === 'object') {
      this.stagingPool = options.stagingPool;
    } else if (
      options.enableStagingPool === true &&
      device &&
      typeof device.createBuffer === 'function'
    ) {
      const StagingCtor = resolveSubsystemCtor(
        options.GpuStagingPool,
        'PlayGta6StagingPool',
        'GpuStagingPool',
        './staging_pool.js'
      );
      if (StagingCtor) {
        this.stagingPool = new StagingCtor({
          ...(options.stagingPoolOptions || {}),
          device,
        });
        this._ownsStagingPool = true;
      }
    }

    // 2. DynamicUniformRing (G18)
    this._ownsUniformRing = false;
    this.uniformRing = null;
    if (options.uniformRing && typeof options.uniformRing === 'object') {
      this.uniformRing = options.uniformRing;
    } else if (
      options.enableUniformRing === true &&
      device &&
      typeof device.createBuffer === 'function'
    ) {
      const UniformRingCtor = resolveSubsystemCtor(
        options.DynamicUniformRing,
        'PlayGta6UniformRing',
        'DynamicUniformRing',
        './uniform_ring.js'
      );
      if (UniformRingCtor) {
        this.uniformRing = new UniformRingCtor({
          ...(options.uniformRingOptions || {}),
          device,
        });
        this._ownsUniformRing = true;
      }
    }

    // 3. GpuPassTimer (G16)
    this._ownsPassTimer = false;
    this.passTimer = null;
    if (options.passTimer && typeof options.passTimer === 'object') {
      this.passTimer = options.passTimer;
    } else if (
      options.enablePassTimer === true ||
      (options.enablePassTimer !== false && deviceSupportsTimestampQuery(device))
    ) {
      const PassTimerCtor = resolveSubsystemCtor(
        options.GpuPassTimer,
        'PlayGta6GpuTimer',
        'GpuPassTimer',
        './gpu_timer.js'
      );
      if (PassTimerCtor) {
        this.passTimer = new PassTimerCtor({
          ...(options.passTimerOptions || {}),
          device,
        });
        this._ownsPassTimer = true;
      }
    }

    // 4. ReflectionRegistry (G17)
    this._ownsReflectionRegistry = false;
    this.reflectionRegistry = null;
    if (options.reflectionRegistry && typeof options.reflectionRegistry === 'object') {
      this.reflectionRegistry = options.reflectionRegistry;
    } else if (options.enableReflectionRegistry === true) {
      const ReflectionCtor = resolveSubsystemCtor(
        options.ReflectionRegistry,
        'PlayGta6ReflectionLoader',
        'ReflectionRegistry',
        './reflection_loader.js'
      );
      if (ReflectionCtor) {
        this.reflectionRegistry = new ReflectionCtor(options.reflectionRegistryOptions || {});
        this._ownsReflectionRegistry = true;
      }
    }

    // 5. GpuResourceRegistry (G21)
    this._ownsResourceRegistry = false;
    this.resourceRegistry = null;
    if (options.resourceRegistry && typeof options.resourceRegistry === 'object') {
      this.resourceRegistry = options.resourceRegistry;
    } else if (options.enableResourceRegistry === true) {
      const RegistryCtor = resolveSubsystemCtor(
        options.GpuResourceRegistry,
        'PlayGta6ResourceRegistry',
        'GpuResourceRegistry',
        './resource_registry.js'
      );
      if (RegistryCtor) {
        this.resourceRegistry = new RegistryCtor(options.resourceRegistryOptions || {});
        this._ownsResourceRegistry = true;
      }
    }

    // 6. RenderTargetPool (G22)
    this._ownsRtPool = false;
    this.rtPool = null;
    if (options.rtPool && typeof options.rtPool === 'object') {
      this.rtPool = options.rtPool;
    } else if (
      options.enableRtPool === true &&
      device &&
      typeof device.createTexture === 'function'
    ) {
      const RtPoolCtor = resolveSubsystemCtor(
        options.RenderTargetPool,
        'PlayGta6RenderTargetPool',
        'RenderTargetPool',
        './rt_pool.js'
      );
      if (RtPoolCtor) {
        this.rtPool = new RtPoolCtor({
          ...(options.rtPoolOptions || {}),
          device,
        });
        this._ownsRtPool = true;
      }
    }

    if (options.reflectionManifest !== undefined && options.reflectionManifest !== null) {
      this.loadReflectionManifest(options.reflectionManifest);
    }

    // 7. WgslPackCache + ShaderFallbackRouter (G28 / S12)
    this._ownsWgslPackCache = false;
    this.wgslPackCache = null;
    if (options.wgslPackCache && typeof options.wgslPackCache === 'object') {
      this.wgslPackCache = options.wgslPackCache;
    } else {
      this.wgslPackCache = new InlineWgslPackCache();
      this._ownsWgslPackCache = true;
    }

    this._ownsShaderFallbackRouter = false;
    this.shaderFallbackRouter = null;
    if (options.shaderFallbackRouter && typeof options.shaderFallbackRouter === 'object') {
      this.shaderFallbackRouter = options.shaderFallbackRouter;
      if (
        options.wgslPackCache &&
        this.shaderFallbackRouter._packLoader == null &&
        typeof this.shaderFallbackRouter.setPackLoader === 'function'
      ) {
        this.shaderFallbackRouter.setPackLoader(this.wgslPackCache);
      }
    } else if (options.enableShaderFallback !== false) {
      const FallbackRouterCtor = resolveSubsystemCtor(
        options.ShaderFallbackRouter,
        'PlayGta6ShaderFallbackRouter',
        'ShaderFallbackRouter',
        './shader_fallback_router.js'
      );
      if (FallbackRouterCtor) {
        const routerOpts = {
          ...(options.shaderFallbackRouterOptions || {}),
          packLoader: this.wgslPackCache,
        };
        if (
          options.fallbackShaderHashes &&
          typeof options.fallbackShaderHashes === 'object' &&
          routerOpts.fallbackHashes === undefined
        ) {
          routerOpts.fallbackHashes = options.fallbackShaderHashes;
        }
        this.shaderFallbackRouter = new FallbackRouterCtor(routerOpts);
        this._ownsShaderFallbackRouter = true;
      }
    }

    if (options.wgslPackBuffer !== undefined && options.wgslPackBuffer !== null) {
      this.loadWgslPack(options.wgslPackBuffer);
    }

    // 8. BinkVideoPresenter (W38) + SplashFrameBridge (G26)
    this._BinkVideoPresenterCtor =
      options.BinkVideoPresenter !== undefined ? options.BinkVideoPresenter : undefined;
    this._SplashFrameBridgeCtor =
      options.SplashFrameBridge !== undefined ? options.SplashFrameBridge : undefined;
    this._binkPresenterOptions = options.binkPresenterOptions || null;
    this._splashBridgeOptions =
      options.splashFrameBridgeOptions || options.splashBridgeOptions || null;
    this._splashBridgeExplicitlyDisabled = options.enableSplashBridge === false;

    this._ownsBinkPresenter = false;
    this.binkPresenter = null;
    if (options.binkPresenter && typeof options.binkPresenter === 'object') {
      this.binkPresenter = options.binkPresenter;
    }

    this._ownsSplashFrameBridge = false;
    this.splashFrameBridge = null;
    if (options.splashFrameBridge && typeof options.splashFrameBridge === 'object') {
      this.splashFrameBridge = options.splashFrameBridge;
      if (!this.binkPresenter && this.splashFrameBridge.presenter) {
        this.binkPresenter = this.splashFrameBridge.presenter;
      }
    } else if (
      options.enableSplashBridge === true &&
      device &&
      typeof device.createTexture === 'function'
    ) {
      this._ensureSplashBridge();
    }

    // Packet ABI v2 handler ({ handlePacket, destroyResource }), see render_state_packets.js.
    this.extension = null;

    // 9. Frame-1 fallback shader profile tracking & cross-buffer specialization (G34 / G38 / G44)
    this._defaultFallbackProfile = normalizeFallbackProfileName(options.defaultFallbackProfile);
    this._explicitFallbackDrawContext = null;
    this._bufferMeta = new Map();
    this._pipelineMeta = new Map();
    this._bindGroupMeta = new Map();
    this._specializedFallbackPipelines = new Map();
    this._specializedFallbackBindGroups = new Map();
    this._pendingCommandStreamShaderProfiles = null;
    this._fallbackProfileCounts = {
      [FALLBACK_PROFILE.DEFAULT]: 0,
      [FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV]: 0,
      [FALLBACK_PROFILE.TEXTURED_QUAD_24B]: 0,
    };
    this._lastFallbackProfile = null;

    // 10. Live WebGPU validation error capture (G35 / G44)
    this._validationNow = typeof options.now === 'function' ? options.now : defaultNowMs;
    const rawMaxErrHistory =
      options.maxValidationErrorHistory !== undefined
        ? options.maxValidationErrorHistory
        : options.maxErrorHistory !== undefined
          ? options.maxErrorHistory
          : DEFAULT_MAX_VALIDATION_ERROR_HISTORY;
    this._maxValidationErrorHistory =
      Number.isInteger(rawMaxErrHistory) && rawMaxErrHistory >= 1
        ? rawMaxErrHistory
        : DEFAULT_MAX_VALIDATION_ERROR_HISTORY;
    this._validationErrors = [];
    this._validationErrorCount = 0;
    this._validationErrorSeq = 0;
    this._validationErrorCallbacks = new Set();
    if (typeof options.onValidationError === 'function') {
      this._validationErrorCallbacks.add(options.onValidationError);
    }
    this._captureValidationErrors = Boolean(
      options.captureValidationErrors === true ||
      options.enableValidationErrors === true ||
      typeof options.onValidationError === 'function' ||
      (options.swapchain && typeof options.swapchain === 'object')
    );
    this._validationErrorHookedDevice = null;
    this._boundDeviceUncapturedErrorHandler = null;
    this._swapchain = null;
    this._ownsSwapchain = false;
    this._swapchainValidationAttached = false;
    this._swapchainOriginalOnValidationError = null;
    this._swapchainWrappedOnValidationError = null;

    if (options.swapchain && typeof options.swapchain === 'object') {
      this.attachSwapchain(options.swapchain, { owned: options.ownsSwapchain === true });
    } else if (this._captureValidationErrors) {
      this._hookDeviceUncapturedError(this.device);
    }

    this._resetFrameState();
  }

  get destroyed() {
    return this._destroyed;
  }

  get framesPresented() {
    return this.frameCount;
  }

  get swapchain() {
    return this._swapchain;
  }

  get validationErrorCount() {
    return this._validationErrorCount;
  }

  _ensureValidationErrorHooked() {
    if (this._destroyed) {
      return false;
    }
    if (this._swapchainValidationAttached) {
      return true;
    }
    return this._hookDeviceUncapturedError(this.device);
  }

  /**
   * Register a callback notified on every captured WebGPU validation/OOM/internal error.
   * Returns an unsubscribe function.
   *
   * @param {(error: { seq: number, type: string, message: string, timestampMs: number }, totalCount: number) => void} callback
   * @returns {() => void}
   */
  addValidationErrorListener(callback) {
    if (typeof callback !== 'function') {
      throw new TypeError('addValidationErrorListener requires a callback function');
    }
    this._validationErrorCallbacks.add(callback);
    this._ensureValidationErrorHooked();
    return () => {
      this._validationErrorCallbacks.delete(callback);
      if (
        this._validationErrorCallbacks.size === 0 &&
        !this._captureValidationErrors &&
        !this._swapchain
      ) {
        this._unhookDeviceUncapturedError();
      }
    };
  }

  /**
   * Return shallow copies of captured sanitized WebGPU validation errors in the bounded history ring.
   *
   * @returns {Array<{ seq: number, type: string, message: string, timestampMs: number }>}
   */
  getValidationErrors() {
    return this._validationErrors.map((entry) => ({ ...entry }));
  }

  /**
   * Clear the bounded validation error history ring (monotonic counter stays intact).
   */
  clearValidationErrors() {
    this._validationErrors.length = 0;
  }

  _unhookDeviceUncapturedError() {
    const prevDevice = this._validationErrorHookedDevice;
    const prevHandler = this._boundDeviceUncapturedErrorHandler;
    this._validationErrorHookedDevice = null;
    this._boundDeviceUncapturedErrorHandler = null;

    if (
      prevDevice &&
      prevHandler &&
      typeof prevDevice.removeEventListener === 'function'
    ) {
      try {
        prevDevice.removeEventListener('uncapturederror', prevHandler);
      } catch (_) {
        // Ignore removal failures on torn-down or mock devices.
      }
    }
  }

  _hookDeviceUncapturedError(device) {
    if (!device || typeof device !== 'object') {
      this._unhookDeviceUncapturedError();
      return false;
    }
    if (device === this._validationErrorHookedDevice && this._boundDeviceUncapturedErrorHandler) {
      return true;
    }
    this._unhookDeviceUncapturedError();
    if (typeof device.addEventListener !== 'function') {
      return false;
    }
    const handler = (event) => {
      if (this._destroyed || this.device !== device) {
        return;
      }
      this._recordValidationError(event);
    };
    this._validationErrorHookedDevice = device;
    this._boundDeviceUncapturedErrorHandler = handler;
    device.addEventListener('uncapturederror', handler);
    return true;
  }

  _unhookSwapchainValidation() {
    const prevSwapchain = this._swapchain;
    if (prevSwapchain && this._swapchainValidationAttached) {
      if (
        '_onValidationError' in prevSwapchain &&
        prevSwapchain._onValidationError === this._swapchainWrappedOnValidationError
      ) {
        prevSwapchain._onValidationError = this._swapchainOriginalOnValidationError;
      }
    }
    this._swapchainValidationAttached = false;
    this._swapchainOriginalOnValidationError = null;
    this._swapchainWrappedOnValidationError = null;
  }

  /**
   * Bind a WgpuSwapchain to this processor and wire its `onValidationError` hook without
   * double-hooking `device.addEventListener('uncapturederror', ...)`.
   *
   * @param {object | null | undefined} swapchain
   * @param {{ owned?: boolean }} [opts]
   */
  attachSwapchain(swapchain, opts = {}) {
    this._unhookSwapchainValidation();
    this._unhookDeviceUncapturedError();

    if (!swapchain || typeof swapchain !== 'object') {
      this._swapchain = null;
      this._ownsSwapchain = false;
      if (this._captureValidationErrors || this._validationErrorCallbacks.size > 0) {
        this._hookDeviceUncapturedError(this.device);
      }
      return null;
    }

    this._swapchain = swapchain;
    this._ownsSwapchain = Boolean(opts && opts.owned);

    if (!this.device && swapchain.device) {
      this.device = swapchain.device;
    }
    if (!this.ctx && swapchain.context) {
      this.ctx = swapchain.context;
    }

    if ('_onValidationError' in swapchain) {
      const prevCb =
        typeof swapchain._onValidationError === 'function' ? swapchain._onValidationError : null;
      const wrappedCb = (record) => {
        if (prevCb) {
          try {
            prevCb(record);
          } catch (_) {
            // Isolate prior callback failures.
          }
        }
        this._recordValidationError(record);
      };
      swapchain._onValidationError = wrappedCb;
      this._swapchainOriginalOnValidationError = prevCb;
      this._swapchainWrappedOnValidationError = wrappedCb;
      this._swapchainValidationAttached = true;

      // If a custom mock swapchain exposes `_onValidationError` without hooking the device itself,
      // fall back to attaching directly on `device`.
      const swapchainHookedDevice =
        swapchain._errorHookedDevice === this.device ||
        typeof swapchain._hookUncapturedError === 'function';
      if (!swapchainHookedDevice) {
        this._hookDeviceUncapturedError(this.device);
      }
    } else if (typeof swapchain.setOnValidationError === 'function') {
      swapchain.setOnValidationError((record) => {
        this._recordValidationError(record);
      });
      this._swapchainValidationAttached = true;
    } else {
      this._hookDeviceUncapturedError(this.device);
    }

    return this._swapchain;
  }

  /**
   * Bind a newly acquired or recovered GPUDevice and re-attach live validation error capture.
   *
   * @param {GPUDevice} newDevice
   * @param {{ format?: string, recoverSwapchain?: boolean }} [opts]
   */
  bindDevice(newDevice, opts = {}) {
    if (this._destroyed) {
      throw new Error('WgpuCommandProcessor has been destroyed');
    }
    this.device = newDevice;
    if (opts && typeof opts.format === 'string' && opts.format.length > 0) {
      this.format = opts.format;
    }
    this._specializedFallbackPipelines.clear();
    this._specializedFallbackBindGroups.clear();

    if (
      this._swapchain &&
      !this._swapchain.isDestroyed &&
      typeof this._swapchain.recover === 'function' &&
      opts.recoverSwapchain !== false
    ) {
      this._swapchain.recover(newDevice, opts.format);
      if (!this._swapchainValidationAttached && (this._captureValidationErrors || this._validationErrorCallbacks.size > 0)) {
        this._hookDeviceUncapturedError(newDevice);
      }
    } else if (this._captureValidationErrors || this._validationErrorCallbacks.size > 0) {
      this._hookDeviceUncapturedError(newDevice);
    } else {
      this._unhookDeviceUncapturedError();
    }
  }

  _recordValidationError(eventOrRecord) {
    if (this._destroyed) {
      return null;
    }
    const rawError =
      eventOrRecord && typeof eventOrRecord === 'object' && 'error' in eventOrRecord
        ? eventOrRecord.error
        : eventOrRecord;
    const type = normalizeValidationErrorType(rawError, eventOrRecord);

    let rawMessage = '';
    if (rawError && typeof rawError === 'object' && typeof rawError.message === 'string') {
      rawMessage = rawError.message;
    } else if (
      eventOrRecord &&
      typeof eventOrRecord === 'object' &&
      typeof eventOrRecord.message === 'string'
    ) {
      rawMessage = eventOrRecord.message;
    } else if (typeof rawError === 'string') {
      rawMessage = rawError;
    } else if (rawError !== null && rawError !== undefined) {
      rawMessage = String(rawError);
    }

    const message = sanitizeValidationErrorMessage(rawMessage);
    let timestampMs;
    if (
      rawError &&
      typeof rawError === 'object' &&
      typeof rawError.timestampMs === 'number' &&
      Number.isFinite(rawError.timestampMs)
    ) {
      timestampMs = rawError.timestampMs;
    } else if (
      eventOrRecord &&
      typeof eventOrRecord === 'object' &&
      typeof eventOrRecord.timestampMs === 'number' &&
      Number.isFinite(eventOrRecord.timestampMs)
    ) {
      timestampMs = eventOrRecord.timestampMs;
    } else {
      const nowVal = this._validationNow();
      timestampMs = typeof nowVal === 'number' && Number.isFinite(nowVal) ? nowVal : Date.now();
    }

    this._validationErrorSeq += 1;
    this._validationErrorCount += 1;

    const record = {
      seq: this._validationErrorSeq,
      type,
      message,
      timestampMs,
    };

    this._validationErrors.push(record);
    if (this._validationErrors.length > this._maxValidationErrorHistory) {
      this._validationErrors.shift();
    }

    const totalCount = this._validationErrorCount;
    for (const cb of Array.from(this._validationErrorCallbacks)) {
      try {
        cb({ ...record }, totalCount);
      } catch (_) {
        // Isolate callback failures so validation error reporting never crashes the worker.
      }
    }

    return { ...record };
  }

  /**
   * Select a G34 fallback shader profile for a draw state (`fullscreen_triangle_cbv`,
   * `textured_quad_24b`, or `default`).
   */
  selectFallbackShaderProfile(drawState) {
    return selectFallbackShaderProfile(drawState);
  }

  /**
   * Set or clear an explicit draw-state hint used for subsequent missing-shader resolutions.
   *
   * @param {object | null | undefined} drawState
   */
  setFallbackDrawContext(drawState) {
    this._explicitFallbackDrawContext =
      drawState && typeof drawState === 'object' ? { ...drawState } : null;
  }

  /**
   * Resolve a missing/untranslatable shader hash via `shaderFallbackRouter` using the detected
   * Frame-1 fallback profile, remaining 100% compatible with legacy 2-arg routers.
   *
   * @param {bigint | string | number} hash
   * @param {number | string} stage
   * @param {object | string | null | undefined} [drawStateOrOptions]
   */
  resolveFallbackShader(hash, stage, drawStateOrOptions) {
    if (!this.shaderFallbackRouter || typeof this.shaderFallbackRouter.resolve !== 'function') {
      return null;
    }
    const stageName =
      typeof stage === 'number' ? stageCodeToName(stage) || 'vertex' : String(stage || 'vertex');
    let profile = this._defaultFallbackProfile;
    if (typeof drawStateOrOptions === 'string') {
      profile = normalizeFallbackProfileName(drawStateOrOptions);
    } else if (drawStateOrOptions && typeof drawStateOrOptions === 'object') {
      if (typeof drawStateOrOptions.profile === 'string') {
        profile = normalizeFallbackProfileName(drawStateOrOptions.profile);
      } else {
        profile = selectFallbackShaderProfile(drawStateOrOptions);
      }
    } else if (this._explicitFallbackDrawContext) {
      profile = selectFallbackShaderProfile(this._explicitFallbackDrawContext);
    }

    const big = normalizeShaderHashBigInt(hash);
    const lookupHashHex = big !== null ? formatShaderHashHex(big) : String(hash);
    const resolved = callFallbackRouterResolve(
      this.shaderFallbackRouter,
      lookupHashHex,
      stageName,
      profile
    );
    if (resolved && resolved.source === 'builtin') {
      const effectiveProfile = normalizeFallbackProfileName(resolved.profile || profile);
      this._recordFallbackProfileUsage(effectiveProfile);
      resolved.profile = effectiveProfile;
    }
    return resolved;
  }

  _recordFallbackProfileUsage(profile) {
    const canonical = normalizeFallbackProfileName(profile);
    this._fallbackProfileCounts[canonical] = (this._fallbackProfileCounts[canonical] || 0) + 1;
    this._lastFallbackProfile = canonical;
  }

  _invalidateSpecializedPipelinesFor(pipelineId) {
    const prefix = `${pipelineId}|`;
    for (const key of Array.from(this._specializedFallbackPipelines.keys())) {
      if (key.startsWith(prefix)) {
        this._specializedFallbackPipelines.delete(key);
      }
    }
    for (const key of Array.from(this._specializedFallbackBindGroups.keys())) {
      if (key.startsWith(prefix)) {
        this._specializedFallbackBindGroups.delete(key);
      }
    }
  }

  _invalidateSpecializedBindGroupsFor(bindGroupId) {
    const suffix = `|${bindGroupId}`;
    for (const key of Array.from(this._specializedFallbackBindGroups.keys())) {
      if (key.endsWith(suffix)) {
        this._specializedFallbackBindGroups.delete(key);
      }
    }
  }

  _scanCommandBufferFallbackProfiles(view) {
    if (!this.shaderFallbackRouter || typeof this.shaderFallbackRouter.resolve !== 'function') {
      return null;
    }

    const localPipelines = new Map();
    const localBindGroups = new Map();
    const shaderProfileHints = new Map();

    let activePipelineId = null;
    const boundBindGroups = new Map();
    const boundVertexBuffers = new Map();

    const recordHint = (shaderId, profile) => {
      if (!shaderId || profile === FALLBACK_PROFILE.DEFAULT) {
        return;
      }
      const existing = shaderProfileHints.get(shaderId);
      if (!existing || existing === FALLBACK_PROFILE.DEFAULT) {
        shaderProfileHints.set(shaderId, profile);
      }
    };

    let offset = 0;
    while (offset + 8 <= view.byteLength) {
      const opcode = view.getUint16(offset, true);
      const payloadBytes = view.getUint32(offset + 4, true);
      const payloadOffset = offset + 8;
      if (payloadOffset + payloadBytes > view.byteLength) break;

      switch (opcode) {
        case OPCODES.BEGIN_FRAME:
        case OPCODES.PRESENT:
          activePipelineId = null;
          boundBindGroups.clear();
          boundVertexBuffers.clear();
          break;
        case OPCODES.CREATE_RENDER_PIPELINE:
          if (payloadBytes >= 24) {
            const pipeId = view.getUint32(payloadOffset, true);
            const vsShaderId = view.getUint32(payloadOffset + 4, true);
            const fsShaderId = view.getUint32(payloadOffset + 8, true);
            const vertexStride = view.getUint32(payloadOffset + 12, true);
            localPipelines.set(pipeId, { vsShaderId, fsShaderId, vertexStride });
          }
          break;
        case OPCODES.CREATE_BIND_GROUP:
          if (payloadBytes >= 16) {
            const bgId = view.getUint32(payloadOffset, true);
            const entryCount = view.getUint32(payloadOffset + 12, true);
            if (16 + entryCount * 32 <= payloadBytes) {
              let hasCbv = false;
              let hasTexture = false;
              let hasSampler = false;
              for (let i = 0; i < entryCount; i += 1) {
                const entryOffset = payloadOffset + 16 + i * 32;
                const resourceType = view.getUint32(entryOffset + 4, true);
                if (resourceType === BIND_RESOURCE_TYPE.BUFFER) {
                  hasCbv = true;
                } else if (resourceType === BIND_RESOURCE_TYPE.TEXTURE_VIEW) {
                  hasTexture = true;
                } else if (resourceType === BIND_RESOURCE_TYPE.SAMPLER) {
                  hasSampler = true;
                }
              }
              localBindGroups.set(bgId, { hasCbv, hasTexture, hasSampler });
            }
          }
          break;
        case OPCODES.SET_PIPELINE:
          if (payloadBytes >= 4) {
            activePipelineId = view.getUint32(payloadOffset, true);
          }
          break;
        case OPCODES.SET_BIND_GROUP:
          if (payloadBytes >= 8) {
            const groupIndex = view.getUint32(payloadOffset, true);
            const bgId = view.getUint32(payloadOffset + 4, true);
            const bgMeta =
              localBindGroups.get(bgId) ||
              this._bindGroupMeta.get(bgId) ||
              this.bindGroups.get(bgId) ||
              null;
            if (bgMeta) {
              boundBindGroups.set(groupIndex, bgMeta);
            }
          }
          break;
        case OPCODES.SET_VERTEX_BUFFER:
          if (payloadBytes >= 16) {
            const slot = view.getUint32(payloadOffset, true);
            const bufferId = view.getUint32(payloadOffset + 4, true);
            const existingBuf = this.buffers.get(bufferId);
            boundVertexBuffers.set(slot, {
              bufferId,
              stride:
                existingBuf && typeof existingBuf.stride === 'number'
                  ? existingBuf.stride
                  : existingBuf && typeof existingBuf.arrayStride === 'number'
                    ? existingBuf.arrayStride
                    : undefined,
            });
          }
          break;
        case OPCODES.DRAW:
        case OPCODES.DRAW_INDEXED: {
          const minBytes = opcode === OPCODES.DRAW ? 16 : 20;
          if (payloadBytes >= minBytes && activePipelineId !== null) {
            const vertexCount = view.getUint32(payloadOffset, true);
            const instanceCount = view.getUint32(payloadOffset + 4, true);
            const pipeMeta =
              localPipelines.get(activePipelineId) || this._pipelineMeta.get(activePipelineId);
            if (pipeMeta) {
              const profile = selectFallbackShaderProfile({
                vertexCount,
                instanceCount,
                vertexStride: pipeMeta.vertexStride || 0,
                bindGroups: boundBindGroups,
                vertexBuffers: boundVertexBuffers,
              });
              if (profile !== FALLBACK_PROFILE.DEFAULT) {
                recordHint(pipeMeta.vsShaderId, profile);
                if (pipeMeta.fsShaderId) {
                  recordHint(pipeMeta.fsShaderId, profile);
                }
              }
            }
          }
          break;
        }
        default:
          break;
      }

      offset = payloadOffset + payloadBytes;
    }

    return shaderProfileHints.size > 0 ? shaderProfileHints : null;
  }

  _getOrCreateSpecializedFallbackPipeline(pipeMeta, profile) {
    if (
      !pipeMeta ||
      !this.device ||
      !this.shaderFallbackRouter ||
      typeof this.shaderFallbackRouter.resolve !== 'function'
    ) {
      return null;
    }
    const canonicalProfile = normalizeFallbackProfileName(profile);
    if (
      canonicalProfile === FALLBACK_PROFILE.DEFAULT ||
      canonicalProfile === pipeMeta.fallbackProfile
    ) {
      return null;
    }

    const cacheKey = `${pipeMeta.id}|${canonicalProfile}`;
    if (this._specializedFallbackPipelines.has(cacheKey)) {
      return this._specializedFallbackPipelines.get(cacheKey);
    }

    const vsOrig = this.shaderModules.get(pipeMeta.vsShaderId);
    const fsOrig = pipeMeta.fsShaderId ? this.shaderModules.get(pipeMeta.fsShaderId) : null;
    if (!vsOrig) {
      return null;
    }

    let vsModule = vsOrig.module;
    let vsEntryPoint = vsOrig.entryPoint || 'vs_main';
    let vsCode = vsOrig.code;
    let changed = false;

    if (vsOrig.source === 'builtin') {
      const vsHash = vsOrig.requestedHash || vsOrig.hashHex || fnv1a64Hex(`empty:vertex:${pipeMeta.vsShaderId}`);
      const resolvedVs = callFallbackRouterResolve(
        this.shaderFallbackRouter,
        vsHash,
        'vertex',
        canonicalProfile
      );
      if (resolvedVs && typeof resolvedVs.wgsl === 'string' && resolvedVs.wgsl.length > 0) {
        if (resolvedVs.wgsl !== vsOrig.code) {
          vsModule = this.device.createShaderModule({ code: resolvedVs.wgsl });
          vsCode = resolvedVs.wgsl;
          vsEntryPoint = resolvedVs.entryPoint || extractWgslEntryPoint(vsCode, SHADER_STAGE.VERTEX) || 'vs_main';
          changed = true;
        }
        this._recordFallbackProfileUsage(resolvedVs.profile || canonicalProfile);
      }
    }

    let fsModule = fsOrig ? fsOrig.module : null;
    let fsEntryPoint = fsOrig ? fsOrig.entryPoint || 'fs_main' : 'fs_main';
    let fsCode = fsOrig ? fsOrig.code : null;

    if (fsOrig && fsOrig.source === 'builtin') {
      const fsHash = fsOrig.requestedHash || fsOrig.hashHex || fnv1a64Hex(`empty:fragment:${pipeMeta.fsShaderId}`);
      const resolvedFs = callFallbackRouterResolve(
        this.shaderFallbackRouter,
        fsHash,
        'fragment',
        canonicalProfile
      );
      if (resolvedFs && typeof resolvedFs.wgsl === 'string' && resolvedFs.wgsl.length > 0) {
        if (resolvedFs.wgsl !== fsOrig.code) {
          fsModule = this.device.createShaderModule({ code: resolvedFs.wgsl });
          fsCode = resolvedFs.wgsl;
          fsEntryPoint = resolvedFs.entryPoint || extractWgslEntryPoint(fsCode, SHADER_STAGE.FRAGMENT) || 'fs_main';
          changed = true;
        }
        this._recordFallbackProfileUsage(resolvedFs.profile || canonicalProfile);
      }
    }

    if (!changed) {
      const existing = this.pipelines.get(pipeMeta.id) || null;
      const entry = { pipeline: existing, profile: canonicalProfile, changed: false };
      this._specializedFallbackPipelines.set(cacheKey, entry);
      return entry;
    }

    let vertexBuffers = [];
    if (canonicalProfile === FALLBACK_PROFILE.TEXTURED_QUAD_24B) {
      vertexBuffers = [
        {
          arrayStride: FALLBACK_VERTEX_LAYOUT_24B.arrayStride,
          stepMode: FALLBACK_VERTEX_LAYOUT_24B.stepMode,
          attributes: FALLBACK_VERTEX_LAYOUT_24B.attributes.map((a) => ({ ...a })),
        },
      ];
    } else if (canonicalProfile === FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV) {
      vertexBuffers = [];
    }

    const descriptor = {
      layout: 'auto',
      vertex: {
        module: vsModule,
        entryPoint: vsEntryPoint,
        buffers: vertexBuffers,
      },
      primitive: {
        topology: pipeMeta.topology || 'triangle-list',
      },
    };
    if (fsOrig && fsModule) {
      descriptor.fragment = {
        module: fsModule,
        entryPoint: fsEntryPoint,
        targets: [{ format: this.format }],
      };
    }

    const pipeline = this.device.createRenderPipeline(descriptor);
    const entry = { pipeline, profile: canonicalProfile, changed: true, vsCode, fsCode };
    this._specializedFallbackPipelines.set(cacheKey, entry);
    return entry;
  }

  _getOrCreateSpecializedBindGroup(pipeId, profile, specPipeline, groupIndex, bgMeta) {
    if (
      !specPipeline ||
      !bgMeta ||
      !Array.isArray(bgMeta.entries) ||
      typeof specPipeline.getBindGroupLayout !== 'function' ||
      !this.device ||
      typeof this.device.createBindGroup !== 'function'
    ) {
      return null;
    }
    const key = `${pipeId}|${profile}|${bgMeta.id}`;
    if (this._specializedFallbackBindGroups.has(key)) {
      return this._specializedFallbackBindGroups.get(key);
    }
    try {
      const layout = specPipeline.getBindGroupLayout(groupIndex);
      if (!layout) {
        return null;
      }
      const bg = this.device.createBindGroup({
        layout,
        entries: bgMeta.entries,
      });
      this._specializedFallbackBindGroups.set(key, bg);
      return bg;
    } catch (_) {
      return null;
    }
  }

  _notifyFirstPresent(info = {}) {
    if (this._firstPresentNotified) {
      return false;
    }
    this._firstPresentNotified = true;
    if (typeof this._onFirstPresent === 'function') {
      try {
        this._onFirstPresent(info);
      } catch (_) {
        // Telemetry hooks must never break frame presentation.
      }
    }
    return true;
  }

  _resolveBinkPresenterCtor() {
    if (this._BinkVideoPresenterCtor === null || this._BinkVideoPresenterCtor === false) {
      return null;
    }
    return resolveSubsystemCtor(
      this._BinkVideoPresenterCtor,
      'PlayGta6BinkPresenter',
      'BinkVideoPresenter',
      './bink_presenter.js'
    );
  }

  _resolveSplashFrameBridgeCtor() {
    if (this._SplashFrameBridgeCtor === null || this._SplashFrameBridgeCtor === false) {
      return null;
    }
    return resolveSubsystemCtor(
      this._SplashFrameBridgeCtor,
      'PlayGta6SplashFrameBridge',
      'SplashFrameBridge',
      './splash_frame_bridge.js'
    );
  }

  _ensureSplashBridge() {
    if (this._destroyed) {
      return null;
    }
    if (this.splashFrameBridge) {
      if (this.splashFrameBridge.isDestroyed) {
        return null;
      }
      if (!this.binkPresenter && this.splashFrameBridge.presenter) {
        this.binkPresenter = this.splashFrameBridge.presenter;
      }
      return this.splashFrameBridge;
    }
    if (this._splashBridgeExplicitlyDisabled && !this.binkPresenter) {
      return null;
    }
    if (!this.binkPresenter) {
      if (!this.device || typeof this.device.createTexture !== 'function') {
        return null;
      }
      const BinkCtor = this._resolveBinkPresenterCtor();
      if (!BinkCtor) {
        return null;
      }
      this.binkPresenter = new BinkCtor({
        presentationFormat: this.format,
        ...(this._binkPresenterOptions || {}),
        device: this.device,
      });
      this._ownsBinkPresenter = true;
    }
    if (!this._splashBridgeExplicitlyDisabled) {
      const BridgeCtor = this._resolveSplashFrameBridgeCtor();
      if (BridgeCtor) {
        const bridgeOpts = {
          presentationFormat: this.format,
          ...(this._splashBridgeOptions || {}),
          device: this.device,
          presenter: this.binkPresenter,
        };
        const userOnFirstPresent =
          typeof bridgeOpts.onFirstPresent === 'function' ? bridgeOpts.onFirstPresent : null;
        bridgeOpts.onFirstPresent = (info) => {
          this._notifyFirstPresent({
            ...info,
            framesPresented: this.frameCount + 1,
          });
          if (userOnFirstPresent) {
            try {
              userOnFirstPresent(info);
            } catch (_) {
              // User callback failures must not break presentation.
            }
          }
        };
        this.splashFrameBridge = new BridgeCtor(bridgeOpts);
        this._ownsSplashFrameBridge = true;
      }
    }
    return this.splashFrameBridge;
  }

  _resolveSplashTargetSize(swapTexture, framePayload, extraOptions) {
    const extImage =
      framePayload && (framePayload.image || framePayload.imageBitmap || framePayload.bitmap);
    const pickDim = (...candidates) => {
      for (let i = 0; i < candidates.length; i += 1) {
        const v = candidates[i];
        if (typeof v === 'number' && Number.isFinite(v) && Math.floor(v) >= 1) {
          return Math.floor(v);
        }
      }
      return null;
    };
    const width =
      pickDim(
        extraOptions && extraOptions.targetWidth,
        framePayload && framePayload.targetWidth,
        swapTexture && swapTexture.width,
        this._viewportWidth,
        this.ctx && this.ctx.canvas && this.ctx.canvas.width,
        framePayload && framePayload.width,
        extImage && extImage.width
      ) || 1280;
    const height =
      pickDim(
        extraOptions && extraOptions.targetHeight,
        framePayload && framePayload.targetHeight,
        swapTexture && swapTexture.height,
        this._viewportHeight,
        this.ctx && this.ctx.canvas && this.ctx.canvas.height,
        framePayload && framePayload.height,
        extImage && extImage.height
      ) || 720;
    return { width, height };
  }

  /**
   * Present an RGBA8, YUV420P/YUVA420P, ImageBitmap, or diagnostic splash/Bink2 frame
   * onto the WebGPU swapchain via SplashFrameBridge -> BinkVideoPresenter.
   *
   * @param {object} framePayload
   * @param {{ nowMs?: number, respectPacing?: boolean, targetWidth?: number, targetHeight?: number }} [options]
   */
  presentSplashFrame(framePayload, options = {}) {
    const requestedFrameIndex =
      framePayload && typeof framePayload === 'object' && Number.isInteger(framePayload.frameIndex)
        ? framePayload.frameIndex
        : null;

    if (this._destroyed) {
      return {
        ok: false,
        presented: false,
        reason: 'destroyed',
        error: 'WgpuCommandProcessor has been destroyed',
        frameIndex: requestedFrameIndex,
        framesPresented: this.frameCount,
        splashFramesPresented: this._splashFramesPresented,
        firstPresent: false,
      };
    }

    if (!this.device || !this.ctx || typeof this.ctx.getCurrentTexture !== 'function') {
      return {
        ok: false,
        presented: false,
        reason: 'not-ready',
        error: 'WebGPU device or swapchain context is not ready',
        frameIndex: requestedFrameIndex,
        framesPresented: this.frameCount,
        splashFramesPresented: this._splashFramesPresented,
        firstPresent: false,
      };
    }

    if (!framePayload || typeof framePayload !== 'object') {
      return {
        ok: false,
        presented: false,
        reason: 'invalid-frame',
        error: 'presentSplashFrame requires a frame payload object',
        frameIndex: null,
        framesPresented: this.frameCount,
        splashFramesPresented: this._splashFramesPresented,
        firstPresent: false,
      };
    }

    const opts = options && typeof options === 'object' ? options : {};

    try {
      const bridge = this._ensureSplashBridge();
      const presenter =
        this.binkPresenter || (bridge && bridge.presenter ? bridge.presenter : null);

      if (!bridge && !presenter) {
        return {
          ok: false,
          presented: false,
          reason: 'presenter-unavailable',
          error: 'BinkVideoPresenter / SplashFrameBridge unavailable',
          frameIndex: requestedFrameIndex,
          framesPresented: this.frameCount,
          splashFramesPresented: this._splashFramesPresented,
          firstPresent: false,
        };
      }

      if (
        bridge &&
        !presenter &&
        !bridge._BinkVideoPresenterCtor &&
        !this._resolveBinkPresenterCtor()
      ) {
        return {
          ok: false,
          presented: false,
          reason: 'presenter-unavailable',
          error: 'BinkVideoPresenter unavailable',
          frameIndex: requestedFrameIndex,
          framesPresented: this.frameCount,
          splashFramesPresented: this._splashFramesPresented,
          firstPresent: false,
        };
      }

      const swapTexture = this.ctx.getCurrentTexture();
      const targetView =
        swapTexture && typeof swapTexture.createView === 'function'
          ? swapTexture.createView()
          : swapTexture;
      if (!targetView || typeof targetView !== 'object') {
        return {
          ok: false,
          presented: false,
          reason: 'not-ready',
          error: 'Swapchain texture view unavailable',
          frameIndex: requestedFrameIndex,
          framesPresented: this.frameCount,
          splashFramesPresented: this._splashFramesPresented,
          firstPresent: false,
        };
      }

      const { width: targetWidth, height: targetHeight } = this._resolveSplashTargetSize(
        swapTexture,
        framePayload,
        opts
      );

      const externalImage =
        framePayload.image ||
        framePayload.imageBitmap ||
        framePayload.bitmap ||
        (isExternalImageLike(framePayload) ? framePayload : null);
      const isDiagnostic =
        framePayload.diagnostic === true ||
        (framePayload.source === 'diagnostic' &&
          !framePayload.planes &&
          !framePayload.rgba &&
          !framePayload.data &&
          !externalImage);

      const explicitNowMs =
        typeof opts.nowMs === 'number' && Number.isFinite(opts.nowMs)
          ? opts.nowMs
          : typeof framePayload.nowMs === 'number' && Number.isFinite(framePayload.nowMs)
            ? framePayload.nowMs
            : null;
      const respectPacing =
        explicitNowMs !== null ||
        opts.respectPacing === true ||
        framePayload.respectPacing === true;

      let renderNowMs = explicitNowMs;
      if (!respectPacing) {
        const targetFps =
          presenter && typeof presenter.targetFps === 'number' && presenter.targetFps > 0
            ? presenter.targetFps
            : 30;
        const minStepMs = 1000 / targetFps + 1;
        const baseNow =
          typeof framePayload.timestampMs === 'number' && Number.isFinite(framePayload.timestampMs)
            ? framePayload.timestampMs
            : typeof performance !== 'undefined' && typeof performance.now === 'function'
              ? performance.now()
              : Date.now();
        const lastAdv =
          presenter && typeof presenter._lastAdvanceMs === 'number'
            ? presenter._lastAdvanceMs
            : this._lastSplashPresentMs;
        renderNowMs =
          lastAdv !== null && Number.isFinite(lastAdv)
            ? Math.max(baseNow, lastAdv + minStepMs)
            : baseNow;
      }

      if (bridge) {
        let pushed;
        if (externalImage) {
          const extOpts = {};
          if (framePayload.width !== undefined) extOpts.width = framePayload.width;
          if (framePayload.height !== undefined) extOpts.height = framePayload.height;
          if (framePayload.timestampMs !== undefined) extOpts.timestampMs = framePayload.timestampMs;
          if (framePayload.frameIndex !== undefined) extOpts.frameIndex = framePayload.frameIndex;
          pushed = bridge.pushExternalImage(externalImage, extOpts);
        } else if (isDiagnostic) {
          const diagW =
            typeof framePayload.width === 'number' && framePayload.width > 0
              ? Math.floor(framePayload.width)
              : targetWidth;
          const diagH =
            typeof framePayload.height === 'number' && framePayload.height > 0
              ? Math.floor(framePayload.height)
              : targetHeight;
          pushed = bridge.pushDiagnosticFrame(diagW, diagH, framePayload.frameIndex, {
            timestampMs: framePayload.timestampMs,
          });
        } else {
          const format = normalizeSplashPixelFormat(framePayload.format, framePayload);
          const rawPlanes =
            framePayload.planes && typeof framePayload.planes === 'object'
              ? framePayload.planes
              : {};
          let planes;
          if (format === 'rgba8') {
            planes = {
              rgba:
                rawPlanes.rgba !== undefined
                  ? rawPlanes.rgba
                  : rawPlanes.data !== undefined
                    ? rawPlanes.data
                    : framePayload.rgba !== undefined
                      ? framePayload.rgba
                      : framePayload.data,
            };
          } else {
            planes = {
              y: rawPlanes.y !== undefined ? rawPlanes.y : framePayload.y,
              u: rawPlanes.u !== undefined ? rawPlanes.u : framePayload.u,
              v: rawPlanes.v !== undefined ? rawPlanes.v : framePayload.v,
            };
            if (format === 'yuva420p') {
              planes.a = rawPlanes.a !== undefined ? rawPlanes.a : framePayload.a;
            }
          }
          pushed = bridge.pushRawFrame({
            width: framePayload.width,
            height: framePayload.height,
            format,
            planes,
            timestampMs: framePayload.timestampMs,
            colorSpace: framePayload.colorSpace,
            source: framePayload.source,
            frameIndex: framePayload.frameIndex,
          });
        }

        const renderResult = bridge.renderFrame(
          this.device,
          targetView,
          targetWidth,
          targetHeight,
          renderNowMs !== null ? renderNowMs : undefined
        );

        if (!this.binkPresenter && bridge.presenter) {
          this.binkPresenter = bridge.presenter;
        }

        if (renderResult && renderResult.presented) {
          if (renderNowMs !== null) {
            this._lastSplashPresentMs = renderNowMs;
          }
          const prevFrameCount = this.frameCount;
          this.frameCount += 1;
          this._splashFramesPresented += 1;
          const isFirstPresent = prevFrameCount === 0;
          if (isFirstPresent) {
            this._notifyFirstPresent({
              frameIndex: renderResult.frameIndex,
              framesPresented: this.frameCount,
              width: renderResult.width,
              height: renderResult.height,
              source: renderResult.source || 'splash',
              timestampMs: renderResult.timestampMs,
            });
          }
          return {
            ok: true,
            presented: true,
            reason: 'ok',
            frameIndex: renderResult.frameIndex,
            framesPresented: this.frameCount,
            splashFramesPresented: this._splashFramesPresented,
            firstPresent: isFirstPresent,
            width: renderResult.width,
            height: renderResult.height,
            targetWidth,
            targetHeight,
            source: renderResult.source,
            viewport: renderResult.viewport || null,
            timestampMs: renderResult.timestampMs,
          };
        }

        return {
          ok: false,
          presented: false,
          reason: (renderResult && renderResult.reason) || 'not-presented',
          frameIndex:
            pushed && pushed.frameIndex !== undefined ? pushed.frameIndex : requestedFrameIndex,
          framesPresented: this.frameCount,
          splashFramesPresented: this._splashFramesPresented,
          firstPresent: false,
          queueDepth: renderResult && renderResult.queueDepth !== undefined ? renderResult.queueDepth : 0,
        };
      }

      // Direct BinkVideoPresenter fallback when SplashFrameBridge is disabled/unavailable
      // but an explicit BinkVideoPresenter instance was injected.
      const format = normalizeSplashPixelFormat(framePayload.format, framePayload);
      const rawPlanes =
        framePayload.planes && typeof framePayload.planes === 'object' ? framePayload.planes : {};
      const width =
        framePayload.width !== undefined
          ? framePayload.width
          : externalImage && externalImage.width;
      const height =
        framePayload.height !== undefined
          ? framePayload.height
          : externalImage && externalImage.height;

      if (format === 'rgba8') {
        const rgba =
          rawPlanes.rgba !== undefined
            ? rawPlanes.rgba
            : rawPlanes.data !== undefined
              ? rawPlanes.data
              : framePayload.rgba !== undefined
                ? framePayload.rgba
                : framePayload.data !== undefined
                  ? framePayload.data
                  : externalImage && externalImage.data;
        presenter.uploadRgbaFrame(rgba, width, height);
      } else {
        const yuv = {
          y: rawPlanes.y !== undefined ? rawPlanes.y : framePayload.y,
          u: rawPlanes.u !== undefined ? rawPlanes.u : framePayload.u,
          v: rawPlanes.v !== undefined ? rawPlanes.v : framePayload.v,
          width,
          height,
        };
        if (format === 'yuva420p') {
          yuv.a = rawPlanes.a !== undefined ? rawPlanes.a : framePayload.a;
        }
        if (framePayload.colorSpace) {
          yuv.colorSpace = framePayload.colorSpace;
        }
        presenter.uploadYuv420Frame(yuv);
      }

      const encoder = this.device.createCommandEncoder({
        label: 'playgta6-wgpu-worker-bink-encoder',
      });
      const presentResult = presenter.presentFrame(encoder, targetView, targetWidth, targetHeight);
      if (this.device.queue && typeof this.device.queue.submit === 'function') {
        const cmdBuf = encoder && typeof encoder.finish === 'function' ? encoder.finish() : encoder;
        this.device.queue.submit([cmdBuf]);
      }

      if (presentResult && presentResult.presented) {
        const frameIndex =
          requestedFrameIndex !== null ? requestedFrameIndex : this._splashFramesPresented;
        const prevFrameCount = this.frameCount;
        this.frameCount += 1;
        this._splashFramesPresented += 1;
        const isFirstPresent = prevFrameCount === 0;
        const timestampMs =
          typeof framePayload.timestampMs === 'number' && Number.isFinite(framePayload.timestampMs)
            ? framePayload.timestampMs
            : renderNowMs !== null
              ? renderNowMs
              : Date.now();
        if (isFirstPresent) {
          this._notifyFirstPresent({
            frameIndex,
            framesPresented: this.frameCount,
            width,
            height,
            source: framePayload.source || 'raw',
            timestampMs,
          });
        }
        return {
          ok: true,
          presented: true,
          reason: 'ok',
          frameIndex,
          framesPresented: this.frameCount,
          splashFramesPresented: this._splashFramesPresented,
          firstPresent: isFirstPresent,
          width,
          height,
          targetWidth,
          targetHeight,
          source: framePayload.source || 'raw',
          viewport: presentResult.viewport || null,
          timestampMs,
        };
      }

      return {
        ok: false,
        presented: false,
        reason: 'not-presented',
        frameIndex: requestedFrameIndex,
        framesPresented: this.frameCount,
        splashFramesPresented: this._splashFramesPresented,
        firstPresent: false,
      };
    } catch (err) {
      return {
        ok: false,
        presented: false,
        reason: 'error',
        error: String(err),
        frameIndex: requestedFrameIndex,
        framesPresented: this.frameCount,
        splashFramesPresented: this._splashFramesPresented,
        firstPresent: false,
      };
    }
  }

  _resetFrameState() {
    this._clearColor = { ...DEFAULT_CLEAR_COLOR };
    this._passCommands = [];
  }

  _ensurePendingEncoder() {
    if (!this._pendingEncoder && this.device && typeof this.device.createCommandEncoder === 'function') {
      this._pendingEncoder = this.device.createCommandEncoder();
      this._hasPendingUploadCommands = false;
    }
    return this._pendingEncoder;
  }

  _flushPendingUploads(retireStaging = true) {
    if (!this._pendingEncoder) {
      return false;
    }
    const encoder = this._pendingEncoder;
    this._pendingEncoder = null;
    const hadCommands = this._hasPendingUploadCommands;
    this._hasPendingUploadCommands = false;

    if (!hadCommands || !this.device || !this.device.queue) {
      return false;
    }

    if (
      this.stagingPool &&
      !this.stagingPool._destroyed &&
      typeof this.stagingPool.prepareSubmit === 'function'
    ) {
      this.stagingPool.prepareSubmit();
    }
    this.device.queue.submit([encoder.finish()]);
    if (
      retireStaging &&
      this.stagingPool &&
      !this.stagingPool._destroyed &&
      typeof this.stagingPool.endFrame === 'function'
    ) {
      this.stagingPool.endFrame();
    }
    return true;
  }

  _beginSubsystemsForFrame(frameIndex) {
    if (
      this.stagingPool &&
      !this.stagingPool._destroyed &&
      typeof this.stagingPool.beginFrame === 'function'
    ) {
      this.stagingPool.beginFrame(frameIndex);
    }
    if (
      this.uniformRing &&
      !this.uniformRing._destroyed &&
      typeof this.uniformRing.beginFrame === 'function'
    ) {
      this.uniformRing.beginFrame(frameIndex);
    }
    if (
      this.passTimer &&
      !this.passTimer.isDestroyed &&
      typeof this.passTimer.beginFrame === 'function'
    ) {
      this.passTimer.beginFrame();
    }
    if (
      this.resourceRegistry &&
      typeof this.resourceRegistry.beginFrame === 'function'
    ) {
      const current =
        typeof this.resourceRegistry.getStats === 'function'
          ? this.resourceRegistry.getStats().currentFrame
          : 0;
      const targetFrame = Math.max(frameIndex, current);
      this.resourceRegistry.beginFrame(targetFrame);
    }
    this._frameBegunForIndex = frameIndex;
  }

  _ensureFrameBegun() {
    if (this._frameBegunForIndex !== this.frameCount) {
      this._beginSubsystemsForFrame(this.frameCount);
    }
  }

  _endSubsystemsForFrame(workDoneFence = null) {
    if (
      this.stagingPool &&
      !this.stagingPool._destroyed &&
      typeof this.stagingPool.endFrame === 'function'
    ) {
      this.stagingPool.endFrame();
    }
    if (
      this.resourceRegistry &&
      typeof this.resourceRegistry.endFrame === 'function'
    ) {
      this.resourceRegistry.endFrame(workDoneFence);
    }
    if (
      this.rtPool &&
      !this.rtPool.isDestroyed &&
      typeof this.rtPool.endFrame === 'function'
    ) {
      this.rtPool.endFrame();
    }
    this._frameBegunForIndex = -1;
  }

  _getWorkDoneFence() {
    if (
      this.device &&
      this.device.queue &&
      typeof this.device.queue.onSubmittedWorkDone === 'function'
    ) {
      try {
        const fence = this.device.queue.onSubmittedWorkDone();
        if (fence && typeof fence.then === 'function') {
          return fence;
        }
      } catch (_) {
        // Ignore unavailable/lost queue fence.
      }
    }
    return null;
  }

  /**
   * Begin a frame epoch across enabled subsystems (staging pool, uniform ring,
   * pass timer, and deferred resource GC) and reset transient pass state.
   *
   * @param {number} [frameIndex]
   */
  beginFrame(frameIndex) {
    if (this._destroyed) {
      throw new Error('WgpuCommandProcessor has been destroyed');
    }
    // Submit any out-of-frame staged uploads (e.g. init buffer writes) before rotating epochs.
    if (this._pendingEncoder && this._frameBegunForIndex === -1) {
      this._flushPendingUploads(true);
    }
    this._resetFrameState();
    const targetFrame = frameIndex !== undefined ? frameIndex : this.frameCount;
    this._beginSubsystemsForFrame(targetFrame);
  }

  /**
   * End the current frame epoch cleanly: flushes any uncommitted upload/uniform data,
   * retires staging chunks and deferred resources, and advances the RT pool epoch.
   */
  endFrame() {
    if (this._destroyed) {
      return;
    }
    if (
      this.uniformRing &&
      !this.uniformRing._destroyed &&
      typeof this.uniformRing.flush === 'function'
    ) {
      this.uniformRing.flush();
    }
    this._flushPendingUploads(false);
    const workDoneFence = this._getWorkDoneFence();
    this._endSubsystemsForFrame(workDoneFence);
  }

  /**
   * Append a per-draw constant payload into the active DynamicUniformRing slot.
   * Flushed automatically in a single queue.writeBuffer before command submission.
   *
   * @param {ArrayBufferView | ArrayBuffer | SharedArrayBuffer} data
   * @returns {{ buffer: GPUBuffer, dynamicOffset: number, size: number, alignedSize: number }}
   */
  pushDrawConstants(data) {
    if (this._destroyed) {
      throw new Error('WgpuCommandProcessor has been destroyed');
    }
    if (!this.uniformRing || this.uniformRing._destroyed) {
      throw new Error(
        'DynamicUniformRing is not enabled; pass { uniformRing } or { enableUniformRing: true }'
      );
    }
    this._ensureFrameBegun();
    return this.uniformRing.pushConstants(data);
  }

  /**
   * Return the most recently resolved GPU pass timings from GpuPassTimer, or null
   * when timestamp queries are unsupported or no readback has completed yet.
   *
   * @returns {{ totalGpuMs: number, passes: Array<{ name: string, durationMs: number }> } | null}
   */
  getLastPassTimings() {
    if (!this._lastPassTimings) {
      return null;
    }
    return {
      totalGpuMs: this._lastPassTimings.totalGpuMs,
      passes: this._lastPassTimings.passes.map((p) => ({ name: p.name, durationMs: p.durationMs })),
    };
  }

  /**
   * Await any in-flight GPU pass timer readbacks (and collect one pending slot if queued),
   * updating and returning the latest pass timings snapshot.
   */
  async collectPassTimings() {
    if (this._destroyed || !this.passTimer || !this.passTimer.supported) {
      return this.getLastPassTimings();
    }
    if (this._pendingTimerReadbacks.size > 0) {
      await Promise.all(Array.from(this._pendingTimerReadbacks));
    }
    if (
      !this._destroyed &&
      typeof this.passTimer.collectCompletedReadback === 'function'
    ) {
      const timing = await this.passTimer.collectCompletedReadback();
      if (timing && !this._destroyed) {
        this._lastPassTimings = timing;
      }
    }
    return this.getLastPassTimings();
  }

  /**
   * Load and index an S6 shader reflection manifest (JSON string or parsed object).
   * Lazily initializes ReflectionRegistry if one was not injected in the constructor.
   *
   * @param {string | object} jsonOrObj
   * @returns {object} Parsed reflection manifest
   */
  loadReflectionManifest(jsonOrObj) {
    if (this._destroyed) {
      throw new Error('WgpuCommandProcessor has been destroyed');
    }
    if (!this.reflectionRegistry) {
      const ReflectionCtor = resolveSubsystemCtor(
        this._ReflectionRegistryCtor,
        'PlayGta6ReflectionLoader',
        'ReflectionRegistry',
        './reflection_loader.js'
      );
      if (!ReflectionCtor) {
        throw new Error(
          'ReflectionRegistry unavailable; load ./reflection_loader.js or inject reflectionRegistry'
        );
      }
      this.reflectionRegistry = new ReflectionCtor(this._reflectionRegistryOptions || {});
      this._ownsReflectionRegistry = true;
    }
    return this.reflectionRegistry.loadManifest(jsonOrObj);
  }

  /**
   * Explicitly associate a command-stream shader ID with a reflection hash or name key.
   *
   * @param {number} shaderId
   * @param {bigint | string | number} hashOrName
   */
  registerShaderReflectionKey(shaderId, hashOrName) {
    this._shaderReflectionKeys.set(shaderId, hashOrName);
  }

  /**
   * Parse and load a binary WGPK (`.wgsl.pack`) buffer into `wgslPackCache`.
   *
   * @param {ArrayBuffer | SharedArrayBuffer | ArrayBufferView} buffer
   * @param {string} [packName]
   */
  loadWgslPack(buffer, packName) {
    if (this._destroyed) {
      throw new Error('WgpuCommandProcessor has been destroyed');
    }
    if (!this.wgslPackCache) {
      this.wgslPackCache = new InlineWgslPackCache();
      this._ownsWgslPackCache = true;
    }
    const result = loadWgslPackIntoCache(this.wgslPackCache, buffer, packName);
    if (
      this.shaderFallbackRouter &&
      this.shaderFallbackRouter._packLoader == null &&
      typeof this.shaderFallbackRouter.setPackLoader === 'function'
    ) {
      this.shaderFallbackRouter.setPackLoader(this.wgslPackCache);
    }
    return result;
  }

  /**
   * Return top missing shaders recorded by `ShaderFallbackRouter` (sorted by hit count desc).
   *
   * @param {{ limit?: number }} [opts]
   */
  getMissingShaders(opts) {
    if (
      this.shaderFallbackRouter &&
      typeof this.shaderFallbackRouter.getMissingReport === 'function'
    ) {
      return this.shaderFallbackRouter.getMissingReport(opts);
    }
    return [];
  }

  /**
   * Return aggregate lookup / fallback telemetry from `ShaderFallbackRouter`, or null if disabled.
   */
  getShaderFallbackStats() {
    if (
      this.shaderFallbackRouter &&
      typeof this.shaderFallbackRouter.getStats === 'function'
    ) {
      return this.shaderFallbackRouter.getStats();
    }
    return null;
  }

  /**
   * Resolve the GPUBindGroupLayout for a pipeline handle and group index, preferring
   * S6-reflected layouts built during pipeline creation and falling back to
   * `pipeline.getBindGroupLayout(groupIndex)`.
   *
   * @param {number} pipelineId
   * @param {number} groupIndex
   * @returns {GPUBindGroupLayout | null}
   */
  getBindGroupLayout(pipelineId, groupIndex) {
    const reflectedRender = this._pipelineBindGroupLayouts.get(pipelineId);
    if (reflectedRender && reflectedRender[groupIndex]) {
      return reflectedRender[groupIndex];
    }
    const reflectedCompute = this._computePipelineBindGroupLayouts.get(pipelineId);
    if (reflectedCompute && reflectedCompute[groupIndex]) {
      return reflectedCompute[groupIndex];
    }
    const pipeline = this.pipelines.get(pipelineId) || this.computePipelines.get(pipelineId);
    if (pipeline && typeof pipeline.getBindGroupLayout === 'function') {
      return pipeline.getBindGroupLayout(groupIndex);
    }
    return null;
  }

  /**
   * Acquire a transient render-target texture + view from RenderTargetPool.
   *
   * @param {object} desc
   * @param {string} [label]
   */
  acquireRenderTarget(desc, label) {
    if (this._destroyed) {
      throw new Error('WgpuCommandProcessor has been destroyed');
    }
    if (!this.rtPool) {
      if (this._rtPoolExplicitlyDisabled) {
        throw new Error('RenderTargetPool is disabled on this WgpuCommandProcessor');
      }
      const RtPoolCtor = resolveSubsystemCtor(
        this._RenderTargetPoolCtor,
        'PlayGta6RenderTargetPool',
        'RenderTargetPool',
        './rt_pool.js'
      );
      if (!RtPoolCtor || !this.device || typeof this.device.createTexture !== 'function') {
        throw new Error(
          'RenderTargetPool unavailable; load ./rt_pool.js or inject rtPool'
        );
      }
      this.rtPool = new RtPoolCtor({
        ...(this._rtPoolOptions || {}),
        device: this.device,
      });
      this._ownsRtPool = true;
    }
    return this.rtPool.acquire(desc, label);
  }

  /**
   * Release a previously acquired transient render-target back into RenderTargetPool
   * for same-frame aliasing or cross-frame reuse.
   *
   * @param {object | number} entry
   * @returns {boolean}
   */
  releaseRenderTarget(entry) {
    if (this._destroyed || !this.rtPool || this.rtPool.isDestroyed) {
      return false;
    }
    return this.rtPool.release(entry);
  }

  /**
   * Notify the processor of a swapchain/viewport resize so idle transient render targets
   * in RenderTargetPool are invalidated cleanly.
   *
   * @param {number} [width]
   * @param {number} [height]
   * @returns {number} Number of invalidated idle render targets
   */
  onResize(width, height) {
    if (typeof width === 'number' && Number.isFinite(width) && width > 0) {
      this._viewportWidth = Math.floor(width);
    }
    if (typeof height === 'number' && Number.isFinite(height) && height > 0) {
      this._viewportHeight = Math.floor(height);
    }
    if (
      !this._destroyed &&
      this.rtPool &&
      !this.rtPool.isDestroyed &&
      typeof this.rtPool.invalidateAll === 'function'
    ) {
      return this.rtPool.invalidateAll();
    }
    return 0;
  }

  handleResize(width, height) {
    return this.onResize(width, height);
  }

  _resolveShaderReflection(shaderId, shaderEntry, stageName) {
    if (
      !this.reflectionRegistry ||
      typeof this.reflectionRegistry.getShaderReflection !== 'function'
    ) {
      return null;
    }
    if (typeof this.reflectionRegistry.getStats === 'function') {
      const stats = this.reflectionRegistry.getStats();
      if (stats && stats.shaderCount === 0) {
        return null;
      }
    }

    const explicitKey = this._shaderReflectionKeys.get(shaderId);
    if (explicitKey !== undefined) {
      const hit = this.reflectionRegistry.getShaderReflection(explicitKey);
      if (hit) return hit;
    }

    if (shaderEntry && shaderEntry.hashHex) {
      const byHash = this.reflectionRegistry.getShaderReflection(shaderEntry.hashHex);
      if (byHash) return byHash;
    }

    if (shaderEntry && shaderEntry.entryPoint && stageName) {
      const byStageName = this.reflectionRegistry.getShaderReflection(
        `${stageName}:${shaderEntry.entryPoint}`
      );
      if (byStageName) return byStageName;
    }

    return null;
  }

  _buildReflectedPipelineLayout(reflections, label) {
    if (
      !this.reflectionRegistry ||
      !Array.isArray(reflections) ||
      reflections.length === 0 ||
      !this.device ||
      typeof this.device.createPipelineLayout !== 'function'
    ) {
      return null;
    }

    const cache = this.pipelineCache;
    const hasCacheBgl =
      Boolean(cache) &&
      (typeof cache.createBindGroupLayoutFromReflection === 'function' ||
        typeof cache.getOrCreateBindGroupLayout === 'function');
    if (!hasCacheBgl && typeof this.device.createBindGroupLayout !== 'function') {
      return null;
    }

    const capturedBgls = [];
    const collectorCache = {
      createBindGroupLayoutFromReflection: (desc) => {
        let bgl;
        if (cache && typeof cache.createBindGroupLayoutFromReflection === 'function') {
          bgl = cache.createBindGroupLayoutFromReflection(desc);
        } else if (cache && typeof cache.getOrCreateBindGroupLayout === 'function') {
          bgl = cache.getOrCreateBindGroupLayout(desc);
        } else {
          bgl = this.device.createBindGroupLayout(desc);
        }
        capturedBgls.push(bgl);
        return bgl;
      },
    };

    const layoutOptions = label ? { label } : {};
    const pipelineLayout = this.reflectionRegistry.buildPipelineLayout(
      this.device,
      collectorCache,
      reflections,
      layoutOptions
    );
    return {
      pipelineLayout,
      bindGroupLayouts: capturedBgls,
    };
  }

  _destroyBufferIfPresent(id) {
    const existing = this.buffers.get(id);
    if (existing) {
      if (this.resourceRegistry && typeof this.resourceRegistry.retire === 'function') {
        const retired = this.resourceRegistry.retire('buffer', id);
        if (!retired && typeof existing.destroy === 'function') {
          existing.destroy();
        }
      } else if (typeof existing.destroy === 'function') {
        existing.destroy();
      }
      this.buffers.delete(id);
    }
  }

  _destroyTextureIfPresent(id) {
    const existing = this.textures.get(id);
    if (existing) {
      if (this.resourceRegistry && typeof this.resourceRegistry.retire === 'function') {
        const retiredTex = this.resourceRegistry.retire('texture', id);
        this.resourceRegistry.retire('textureView', id);
        if (!retiredTex && typeof existing.destroy === 'function') {
          existing.destroy();
        }
      } else if (typeof existing.destroy === 'function') {
        existing.destroy();
      }
      this.textures.delete(id);
    }
    this.textureViews.delete(id);
  }

  _handleCreateBuffer(view, payloadOffset) {
    const id = view.getUint32(payloadOffset, true);
    const usage = view.getUint32(payloadOffset + 4, true);
    const byteSize = Number(view.getBigUint64(payloadOffset + 8, true));
    this._destroyBufferIfPresent(id);
    const buffer = this.device.createBuffer({
      size: byteSize,
      usage,
    });
    this.buffers.set(id, buffer);
    this._bufferMeta.set(id, { usage, byteSize });
    if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
      this.resourceRegistry.register('buffer', id, buffer, {
        byteSize: Math.max(0, Math.trunc(byteSize)),
        label: `buffer:${id}`,
      });
    }
  }

  _handleWriteBuffer(view, payloadOffset, payloadBytes) {
    const id = view.getUint32(payloadOffset, true);
    const dstOffset = view.getUint32(payloadOffset + 4, true);
    const dataLen = view.getUint32(payloadOffset + 8, true);
    if (16 + dataLen > payloadBytes) return;

    const buffer = this.buffers.get(id);
    if (!buffer) return;

    const data = new Uint8Array(view.buffer, view.byteOffset + payloadOffset + 16, dataLen);

    if (
      this.stagingPool &&
      !this.stagingPool._destroyed &&
      typeof this.stagingPool.stageBufferUpload === 'function' &&
      dataLen > 0 &&
      dstOffset % 4 === 0
    ) {
      const rem = dataLen % 4;
      const copySize = rem === 0 ? dataLen : dataLen + (4 - rem);
      const fitsDst = typeof buffer.size !== 'number' || dstOffset + copySize <= buffer.size;
      if (fitsDst) {
        const encoder = this._ensurePendingEncoder();
        if (encoder && typeof encoder.copyBufferToBuffer === 'function') {
          this.stagingPool.stageBufferUpload(encoder, buffer, dstOffset, data);
          this._hasPendingUploadCommands = true;
          return;
        }
        if (!this._hasPendingUploadCommands) {
          this._pendingEncoder = null;
        }
      }
    }

    this.device.queue.writeBuffer(buffer, dstOffset, data);
  }

  _handleCreateShaderModule(view, payloadOffset, payloadBytes) {
    const id = view.getUint32(payloadOffset, true);
    const stage = view.getUint32(payloadOffset + 4, true);
    const codeLen = view.getUint32(payloadOffset + 8, true);
    if (16 + codeLen > payloadBytes) return;

    const rawBytes = new Uint8Array(view.buffer, view.byteOffset + payloadOffset + 16, codeLen);
    const isShared = typeof SharedArrayBuffer !== 'undefined' && view.buffer instanceof SharedArrayBuffer;
    const rawCode = this._decoder.decode(isShared ? rawBytes.slice() : rawBytes);
    const trimmed = rawCode.trim();
    const stageName = stageCodeToName(stage) || 'vertex';
    const hashOnlyHex = parseHashOnlyShaderPayload(trimmed);

    let code = rawCode;
    let hashHex = null;
    let entryPoint = null;
    let source = 'inline';
    let requestedHash = null;
    let resolvedHash = null;
    let fallbackProfile = null;

    if (trimmed.length === 0 || hashOnlyHex !== null) {
      let lookupHashHex = hashOnlyHex;
      if (!lookupHashHex) {
        const explicitKey = this._shaderReflectionKeys.get(id);
        const explicitBigInt =
          explicitKey !== undefined ? normalizeShaderHashBigInt(explicitKey) : null;
        lookupHashHex =
          explicitBigInt !== null
            ? formatShaderHashHex(explicitBigInt)
            : fnv1a64Hex(`empty:${stageName}:${id}`);
      }
      requestedHash = lookupHashHex;

      if (
        this.shaderFallbackRouter &&
        typeof this.shaderFallbackRouter.resolve === 'function'
      ) {
        let hintProfile = this._defaultFallbackProfile;
        if (
          this._pendingCommandStreamShaderProfiles &&
          this._pendingCommandStreamShaderProfiles.has(id)
        ) {
          hintProfile = this._pendingCommandStreamShaderProfiles.get(id);
        } else if (this._explicitFallbackDrawContext) {
          hintProfile = selectFallbackShaderProfile(this._explicitFallbackDrawContext);
        }
        const resolved = callFallbackRouterResolve(
          this.shaderFallbackRouter,
          lookupHashHex,
          stageName,
          hintProfile
        );
        if (resolved && typeof resolved.wgsl === 'string' && resolved.wgsl.length > 0) {
          code = resolved.wgsl;
          source = resolved.source || 'fallback';
          requestedHash = resolved.requestedHash || lookupHashHex;
          resolvedHash = resolved.resolvedHash !== undefined ? resolved.resolvedHash : null;
          hashHex = resolvedHash || fnv1a64Hex(code);
          entryPoint = resolved.entryPoint || extractWgslEntryPoint(code, stage);
          if (source === 'builtin') {
            fallbackProfile = normalizeFallbackProfileName(resolved.profile || hintProfile);
            this._recordFallbackProfileUsage(fallbackProfile);
          }
        }
      } else if (this.wgslPackCache) {
        const packHit = lookupFromPackCache(this.wgslPackCache, lookupHashHex);
        if (packHit && typeof packHit.wgsl === 'string' && packHit.wgsl.length > 0) {
          code = packHit.wgsl;
          source = 'pack';
          resolvedHash = lookupHashHex;
          hashHex = lookupHashHex;
          entryPoint = packHit.entryPoint || extractWgslEntryPoint(code, stage);
        }
      }
    }

    if (hashHex === null) {
      hashHex = fnv1a64Hex(code);
    }
    if (entryPoint === null) {
      entryPoint = extractWgslEntryPoint(code, stage);
    }
    if (requestedHash === null) {
      requestedHash = hashHex;
    }
    if (resolvedHash === null && source === 'inline') {
      resolvedHash = hashHex;
    }

    const module = this.device.createShaderModule({ code });
    this.shaderModules.set(id, {
      module,
      stage,
      code,
      hashHex,
      entryPoint,
      source,
      requestedHash,
      resolvedHash,
      fallbackProfile,
    });
    if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
      const byteSize =
        code === rawCode
          ? Math.max(0, Math.trunc(codeLen))
          : new TextEncoder().encode(code).byteLength;
      this.resourceRegistry.register('shaderModule', id, module, {
        byteSize,
        label: `shaderModule:${id}`,
      });
    }
  }

  _handleCreateRenderPipeline(view, payloadOffset) {
    const id = view.getUint32(payloadOffset, true);
    const vsShaderId = view.getUint32(payloadOffset + 4, true);
    const fsShaderId = view.getUint32(payloadOffset + 8, true);
    const vertexStride = view.getUint32(payloadOffset + 12, true);
    const topologyCode = view.getUint32(payloadOffset + 16, true);
    // Reserved pipeline flags at payloadOffset + 20 for future blend/depth state.

    const vsEntry = this.shaderModules.get(vsShaderId);
    if (!vsEntry) return;
    const fsEntry = fsShaderId !== 0 ? this.shaderModules.get(fsShaderId) : null;

    const isFallback = Boolean(
      (vsEntry && vsEntry.source === 'builtin') || (fsEntry && fsEntry.source === 'builtin')
    );
    const effectiveFallbackProfile =
      vsEntry && vsEntry.fallbackProfile && vsEntry.fallbackProfile !== FALLBACK_PROFILE.DEFAULT
        ? vsEntry.fallbackProfile
        : fsEntry && fsEntry.fallbackProfile && fsEntry.fallbackProfile !== FALLBACK_PROFILE.DEFAULT
          ? fsEntry.fallbackProfile
          : isFallback
            ? FALLBACK_PROFILE.DEFAULT
            : null;

    const vsRefl = this._resolveShaderReflection(vsShaderId, vsEntry, 'vertex');
    const fsRefl = fsEntry ? this._resolveShaderReflection(fsShaderId, fsEntry, 'fragment') : null;
    const matchedRefls = [];
    if (vsRefl) matchedRefls.push(vsRefl);
    if (fsRefl) matchedRefls.push(fsRefl);

    const reflected = this._buildReflectedPipelineLayout(matchedRefls, `render-pipeline-${id}`);
    if (reflected) {
      this._pipelineBindGroupLayouts.set(id, reflected.bindGroupLayouts);
    } else {
      this._pipelineBindGroupLayouts.delete(id);
    }

    const topology = TOPOLOGY_NAMES[topologyCode] || 'triangle-list';
    let vertexBuffers;
    if (effectiveFallbackProfile === FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV && !vsRefl) {
      vertexBuffers = [];
    } else if (
      effectiveFallbackProfile === FALLBACK_PROFILE.TEXTURED_QUAD_24B &&
      vertexStride === TEXTURED_QUAD_24B_STRIDE &&
      !vsRefl
    ) {
      vertexBuffers = [
        {
          arrayStride: FALLBACK_VERTEX_LAYOUT_24B.arrayStride,
          stepMode: FALLBACK_VERTEX_LAYOUT_24B.stepMode,
          attributes: FALLBACK_VERTEX_LAYOUT_24B.attributes.map((a) => ({ ...a })),
        },
      ];
    } else {
      vertexBuffers =
        vertexStride > 0
          ? [
              {
                arrayStride: vertexStride,
                attributes: [
                  {
                    shaderLocation: 0,
                    offset: 0,
                    format: vertexStride >= 12 ? 'float32x3' : vertexStride >= 8 ? 'float32x2' : 'float32',
                  },
                ],
              },
            ]
          : [];
    }

    if (
      vsRefl &&
      vertexStride > 0 &&
      Array.isArray(vsRefl.vertexInputs) &&
      vsRefl.vertexInputs.length > 0 &&
      this.reflectionRegistry &&
      typeof this.reflectionRegistry.buildVertexBufferLayouts === 'function'
    ) {
      try {
        const reflectedVbl = this.reflectionRegistry.buildVertexBufferLayouts(vsRefl, vertexStride);
        if (Array.isArray(reflectedVbl) && reflectedVbl.length > 0) {
          vertexBuffers = reflectedVbl;
        }
      } catch (_) {
        // Keep default single-attribute vertex layout when stride/format overrides don't match.
      }
    }

    const descriptor = {
      layout: reflected ? reflected.pipelineLayout : 'auto',
      vertex: {
        module: vsEntry.module,
        entryPoint: vsRefl && vsRefl.entryPoint ? vsRefl.entryPoint : 'vs_main',
        buffers: vertexBuffers,
      },
      primitive: {
        topology,
      },
    };

    if (fsEntry) {
      descriptor.fragment = {
        module: fsEntry.module,
        entryPoint: fsRefl && fsRefl.entryPoint ? fsRefl.entryPoint : 'fs_main',
        targets: [{ format: this.format }],
      };
    }

    const pipeline = this.device.createRenderPipeline(descriptor);
    this.pipelines.set(id, pipeline);
    this._pipelineMeta.set(id, {
      id,
      vsShaderId,
      fsShaderId,
      vertexStride,
      topology,
      hasReflectedLayout: Boolean(reflected),
      isFallback,
      fallbackProfile: effectiveFallbackProfile,
    });
    this._invalidateSpecializedPipelinesFor(id);
    if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
      this.resourceRegistry.register('renderPipeline', id, pipeline, {
        byteSize: 0,
        label: `renderPipeline:${id}`,
      });
    }
  }

  _handleCreateTexture(view, payloadOffset) {
    const id = view.getUint32(payloadOffset, true);
    const width = view.getUint32(payloadOffset + 4, true);
    const height = view.getUint32(payloadOffset + 8, true);
    const depthOrArrayLayers = Math.max(1, view.getUint32(payloadOffset + 12, true));
    const mipLevelCount = Math.max(1, view.getUint32(payloadOffset + 16, true));
    const formatCode = view.getUint32(payloadOffset + 20, true);
    const usage = view.getUint32(payloadOffset + 24, true);
    const dimensionCode = view.getUint32(payloadOffset + 28, true);

    const format = TEXTURE_FORMAT_NAMES[formatCode] || 'rgba8unorm';
    const dimension = TEXTURE_DIMENSION_NAMES[dimensionCode] || '2d';

    this._destroyTextureIfPresent(id);
    const texture = this.device.createTexture({
      size: {
        width,
        height,
        depthOrArrayLayers,
      },
      mipLevelCount,
      format,
      usage,
      dimension,
    });
    this.textures.set(id, texture);
    let textureView = null;
    if (texture && typeof texture.createView === 'function') {
      textureView = texture.createView();
      this.textureViews.set(id, textureView);
    }
    if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
      const byteSize = estimateTextureBytes(
        width,
        height,
        depthOrArrayLayers,
        mipLevelCount,
        format,
        dimension
      );
      this.resourceRegistry.register('texture', id, texture, {
        byteSize,
        label: `texture:${id}`,
      });
      if (textureView) {
        this.resourceRegistry.register('textureView', id, textureView, {
          byteSize: 0,
          label: `textureView:${id}`,
        });
      }
    }
  }

  _handleWriteTexture(view, payloadOffset, payloadBytes) {
    const textureId = view.getUint32(payloadOffset, true);
    const mipLevel = view.getUint32(payloadOffset + 4, true);
    const originX = view.getUint32(payloadOffset + 8, true);
    const originY = view.getUint32(payloadOffset + 12, true);
    const originZ = view.getUint32(payloadOffset + 16, true);
    const width = view.getUint32(payloadOffset + 20, true);
    const height = view.getUint32(payloadOffset + 24, true);
    const depthOrArrayLayers = Math.max(1, view.getUint32(payloadOffset + 28, true));
    const bytesPerRow = view.getUint32(payloadOffset + 32, true);
    const rowsPerImage = view.getUint32(payloadOffset + 36, true);
    const dataLen = view.getUint32(payloadOffset + 40, true);
    if (48 + dataLen > payloadBytes) return;

    const texture = this.textures.get(textureId);
    if (!texture) return;

    const data = new Uint8Array(view.buffer, view.byteOffset + payloadOffset + 48, dataLen);

    if (
      this.stagingPool &&
      !this.stagingPool._destroyed &&
      typeof this.stagingPool.stageTextureUpload === 'function' &&
      width > 0 &&
      height > 0 &&
      dataLen > 0
    ) {
      const effectiveRowsPerImage = rowsPerImage > 0 ? rowsPerImage : height;
      const totalRows = effectiveRowsPerImage * depthOrArrayLayers;
      const canStage =
        bytesPerRow > 0
          ? dataLen >= bytesPerRow * totalRows
          : totalRows > 0 && dataLen % totalRows === 0;
      if (canStage) {
        const encoder = this._ensurePendingEncoder();
        if (encoder && typeof encoder.copyBufferToTexture === 'function') {
          const layout = {
            width,
            height,
            depthOrArrayLayers,
            mipLevel,
            origin: { x: originX, y: originY, z: originZ },
          };
          if (bytesPerRow > 0) {
            layout.bytesPerRow = bytesPerRow;
          }
          if (rowsPerImage > 0) {
            layout.rowsPerImage = rowsPerImage;
          }
          this.stagingPool.stageTextureUpload(encoder, texture, data, layout);
          this._hasPendingUploadCommands = true;
          return;
        }
        if (!this._hasPendingUploadCommands) {
          this._pendingEncoder = null;
        }
      }
    }

    const dataLayout = { offset: 0 };
    if (bytesPerRow > 0) {
      dataLayout.bytesPerRow = bytesPerRow;
    }
    if (rowsPerImage > 0) {
      dataLayout.rowsPerImage = rowsPerImage;
    }
    this.device.queue.writeTexture(
      {
        texture,
        mipLevel,
        origin: { x: originX, y: originY, z: originZ },
      },
      data,
      dataLayout,
      {
        width,
        height,
        depthOrArrayLayers,
      }
    );
  }

  _handleCreateSampler(view, payloadOffset) {
    const id = view.getUint32(payloadOffset, true);
    const addressModeU = ADDRESS_MODE_NAMES[view.getUint32(payloadOffset + 4, true)] || 'clamp-to-edge';
    const addressModeV = ADDRESS_MODE_NAMES[view.getUint32(payloadOffset + 8, true)] || 'clamp-to-edge';
    const addressModeW = ADDRESS_MODE_NAMES[view.getUint32(payloadOffset + 12, true)] || 'clamp-to-edge';
    const magFilter = FILTER_MODE_NAMES[view.getUint32(payloadOffset + 16, true)] || 'nearest';
    const minFilter = FILTER_MODE_NAMES[view.getUint32(payloadOffset + 20, true)] || 'nearest';
    const mipmapFilter = FILTER_MODE_NAMES[view.getUint32(payloadOffset + 24, true)] || 'nearest';
    const maxAnisotropy = Math.max(1, view.getUint32(payloadOffset + 28, true));

    const sampler = this.device.createSampler({
      addressModeU,
      addressModeV,
      addressModeW,
      magFilter,
      minFilter,
      mipmapFilter,
      maxAnisotropy,
    });
    this.samplers.set(id, sampler);
    if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
      this.resourceRegistry.register('sampler', id, sampler, {
        byteSize: 0,
        label: `sampler:${id}`,
      });
    }
  }

  _handleCreateComputePipeline(view, payloadOffset) {
    const id = view.getUint32(payloadOffset, true);
    const csShaderId = view.getUint32(payloadOffset + 4, true);
    // Reserved flags at payloadOffset + 8 and reserved at payloadOffset + 12.

    const csEntry = this.shaderModules.get(csShaderId);
    if (!csEntry) return;

    const csRefl = this._resolveShaderReflection(csShaderId, csEntry, 'compute');
    const reflected = csRefl
      ? this._buildReflectedPipelineLayout([csRefl], `compute-pipeline-${id}`)
      : null;
    if (reflected) {
      this._computePipelineBindGroupLayouts.set(id, reflected.bindGroupLayouts);
    } else {
      this._computePipelineBindGroupLayouts.delete(id);
    }

    const pipeline = this.device.createComputePipeline({
      layout: reflected ? reflected.pipelineLayout : 'auto',
      compute: {
        module: csEntry.module,
        entryPoint: csRefl && csRefl.entryPoint ? csRefl.entryPoint : 'cs_main',
      },
    });
    this.computePipelines.set(id, pipeline);
    if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
      this.resourceRegistry.register('computePipeline', id, pipeline, {
        byteSize: 0,
        label: `computePipeline:${id}`,
      });
    }
  }

  _handleCreateBindGroup(view, payloadOffset, payloadBytes) {
    const id = view.getUint32(payloadOffset, true);
    const pipelineId = view.getUint32(payloadOffset + 4, true);
    const groupIndex = view.getUint32(payloadOffset + 8, true);
    const entryCount = view.getUint32(payloadOffset + 12, true);
    if (16 + entryCount * 32 > payloadBytes) return;

    const renderPipeline = this.pipelines.get(pipelineId);
    const computePipeline = this.computePipelines.get(pipelineId);
    const pipeline = renderPipeline || computePipeline;
    const pipeMeta = this._pipelineMeta.get(pipelineId) || null;
    if (!pipeline) return;

    let layout = null;
    if (pipeMeta && pipeMeta.isFallback) {
      try {
        layout = this.getBindGroupLayout(pipelineId, groupIndex);
      } catch (_) {
        layout = null;
      }
    } else {
      layout = this.getBindGroupLayout(pipelineId, groupIndex);
      if (!layout) return;
    }

    const entries = [];
    let hasCbv = false;
    let hasTexture = false;
    let hasSampler = false;
    for (let i = 0; i < entryCount; i++) {
      const entryOffset = payloadOffset + 16 + i * 32;
      const binding = view.getUint32(entryOffset, true);
      const resourceType = view.getUint32(entryOffset + 4, true);
      const resourceId = view.getUint32(entryOffset + 8, true);
      const offset = Number(view.getBigUint64(entryOffset + 16, true));
      const size = Number(view.getBigUint64(entryOffset + 24, true));

      if (resourceType === BIND_RESOURCE_TYPE.BUFFER) {
        const buffer = this.buffers.get(resourceId);
        if (!buffer) return;
        const resource = { buffer, offset };
        if (size > 0) {
          resource.size = size;
        }
        hasCbv = true;
        entries.push({ binding, resource });
      } else if (resourceType === BIND_RESOURCE_TYPE.TEXTURE_VIEW) {
        let textureView = this.textureViews.get(resourceId);
        if (!textureView) {
          const texture = this.textures.get(resourceId);
          if (texture && typeof texture.createView === 'function') {
            textureView = texture.createView();
            this.textureViews.set(resourceId, textureView);
            if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
              this.resourceRegistry.register('textureView', resourceId, textureView, {
                byteSize: 0,
                label: `textureView:${resourceId}`,
              });
            }
          }
        }
        if (!textureView) return;
        hasTexture = true;
        entries.push({ binding, resource: textureView });
      } else if (resourceType === BIND_RESOURCE_TYPE.SAMPLER) {
        const sampler = this.samplers.get(resourceId);
        if (!sampler) return;
        hasSampler = true;
        entries.push({ binding, resource: sampler });
      } else {
        return;
      }
    }

    this._bindGroupMeta.set(id, {
      id,
      pipelineId,
      groupIndex,
      hasCbv,
      hasTexture,
      hasSampler,
      entries,
    });
    this._invalidateSpecializedBindGroupsFor(id);

    if (!layout) {
      return;
    }

    const bindGroup = this.device.createBindGroup({
      layout,
      entries,
    });
    this.bindGroups.set(id, bindGroup);
    this._bindGroupKinds.set(id, computePipeline && !renderPipeline ? 'compute' : 'render');
    if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
      this.resourceRegistry.register('bindGroup', id, bindGroup, {
        byteSize: 0,
        label: `bindGroup:${id}`,
      });
    }
  }

  _handleDestroyResource(id) {
    this._destroyBufferIfPresent(id);
    this._destroyTextureIfPresent(id);
    if (this.resourceRegistry && typeof this.resourceRegistry.retireHandle === 'function') {
      this.resourceRegistry.retireHandle(id);
    }
    this._bufferMeta.delete(id);
    this.samplers.delete(id);
    this.bindGroups.delete(id);
    this._bindGroupKinds.delete(id);
    this._bindGroupMeta.delete(id);
    this._invalidateSpecializedBindGroupsFor(id);
    this.shaderModules.delete(id);
    this.pipelines.delete(id);
    this._pipelineMeta.delete(id);
    this._invalidateSpecializedPipelinesFor(id);
    this.computePipelines.delete(id);
    this._pipelineBindGroupLayouts.delete(id);
    this._computePipelineBindGroupLayouts.delete(id);
    this._shaderReflectionKeys.delete(id);
    if (this.extension) this.extension.destroyResource(id);
  }

  _flushPresent() {
    const encoder = this._pendingEncoder || this.device.createCommandEncoder();
    this._pendingEncoder = null;
    this._hasPendingUploadCommands = false;

    const swapchainView = this.ctx.getCurrentTexture().createView();
    let activePass = null;
    let activePassKind = null;
    let renderPassOpened = false;
    let renderPassIdx = 0;
    let computePassIdx = 0;

    const timerSupported = Boolean(this.passTimer && this.passTimer.supported);

    const endActivePass = () => {
      if (activePass) {
        activePass.end();
        activePass = null;
        activePassKind = null;
      }
    };

    const ensureRenderPass = () => {
      if (activePassKind === 'render') return activePass;
      endActivePass();
      const attachment = {
        view: swapchainView,
        loadOp: renderPassOpened ? 'load' : 'clear',
        storeOp: 'store',
      };
      if (!renderPassOpened) {
        attachment.clearValue = { ...this._clearColor };
      }
      const passDesc = {
        colorAttachments: [attachment],
      };
      if (timerSupported && typeof this.passTimer.getPassTimestampWrites === 'function') {
        const tsWrites = this.passTimer.getPassTimestampWrites(`render_${renderPassIdx}`);
        if (tsWrites) {
          passDesc.timestampWrites = tsWrites;
        }
      }
      renderPassIdx += 1;
      activePass = encoder.beginRenderPass(passDesc);
      activePassKind = 'render';
      renderPassOpened = true;
      return activePass;
    };

    const ensureComputePass = () => {
      if (activePassKind === 'compute') return activePass;
      endActivePass();
      if (timerSupported && typeof this.passTimer.getPassTimestampWrites === 'function') {
        const tsWrites = this.passTimer.getPassTimestampWrites(`compute_${computePassIdx}`);
        computePassIdx += 1;
        activePass = tsWrites
          ? encoder.beginComputePass({ timestampWrites: tsWrites })
          : encoder.beginComputePass();
      } else {
        computePassIdx += 1;
        activePass = encoder.beginComputePass();
      }
      activePassKind = 'compute';
      return activePass;
    };

    // Pass access for queued extension commands ({ type, run(passControl) }).
    const passControl = {
      swapchainView,
      get pass() { return activePass; },
      get passKind() { return activePassKind; },
      endPass: endActivePass,
      beginRenderPass: (descriptor) => {
        endActivePass();
        if (timerSupported && typeof this.passTimer.getPassTimestampWrites === 'function') {
          const tsWrites = this.passTimer.getPassTimestampWrites(`render_${renderPassIdx}`);
          if (tsWrites) descriptor = { ...descriptor, timestampWrites: tsWrites };
        }
        renderPassIdx += 1;
        activePass = encoder.beginRenderPass(descriptor);
        activePassKind = 'render';
        renderPassOpened = true;
        return activePass;
      },
    };

    let activePipelineId = null;
    let specializedActiveKey = null;
    const boundBindGroups = new Map();
    const boundVertexBuffers = new Map();

    const maybeSpecializeFallbackPipeline = (pass, vertexCount, instanceCount) => {
      if (activePipelineId === null || !this.shaderFallbackRouter) {
        return;
      }
      const pipeMeta = this._pipelineMeta.get(activePipelineId);
      if (!pipeMeta || !pipeMeta.isFallback) {
        return;
      }
      const desiredProfile = selectFallbackShaderProfile({
        vertexCount,
        instanceCount,
        vertexStride: pipeMeta.vertexStride || 0,
        bindGroups: boundBindGroups,
        vertexBuffers: boundVertexBuffers,
      });
      if (
        desiredProfile === FALLBACK_PROFILE.DEFAULT ||
        desiredProfile === pipeMeta.fallbackProfile
      ) {
        return;
      }
      const spec = this._getOrCreateSpecializedFallbackPipeline(pipeMeta, desiredProfile);
      if (!spec || !spec.changed || !spec.pipeline) {
        return;
      }
      const key = `${pipeMeta.id}|${desiredProfile}`;
      if (specializedActiveKey !== key) {
        pass.setPipeline(spec.pipeline);
        for (const [groupIndex, boundBg] of boundBindGroups.entries()) {
          if (boundBg && boundBg.meta) {
            const specBg = this._getOrCreateSpecializedBindGroup(
              pipeMeta.id,
              desiredProfile,
              spec.pipeline,
              groupIndex,
              boundBg.meta
            );
            if (specBg) {
              pass.setBindGroup(groupIndex, specBg);
            }
          }
        }
        specializedActiveKey = key;
      }
    };

    for (const cmd of this._passCommands) {
      switch (cmd.type) {
        case 'setPipeline': {
          const pass = ensureRenderPass();
          activePipelineId = cmd.pipelineId;
          specializedActiveKey = null;
          const pipeline = this.pipelines.get(cmd.pipelineId);
          if (pipeline) {
            pass.setPipeline(pipeline);
          }
          break;
        }
        case 'setVertexBuffer': {
          const pass = ensureRenderPass();
          const buffer = this.buffers.get(cmd.bufferId);
          if (buffer) {
            boundVertexBuffers.set(cmd.slot, {
              bufferId: cmd.bufferId,
              buffer,
              offset: cmd.offset,
              stride:
                typeof buffer.stride === 'number'
                  ? buffer.stride
                  : typeof buffer.arrayStride === 'number'
                    ? buffer.arrayStride
                    : undefined,
            });
            pass.setVertexBuffer(cmd.slot, buffer, cmd.offset);
          }
          break;
        }
        case 'setIndexBuffer': {
          const pass = ensureRenderPass();
          const buffer = this.buffers.get(cmd.bufferId);
          if (buffer) {
            pass.setIndexBuffer(buffer, cmd.indexFormat, cmd.offset);
          }
          break;
        }
        case 'setBindGroup': {
          const bindGroup = this.bindGroups.get(cmd.bindGroupId);
          const bgMeta = this._bindGroupMeta.get(cmd.bindGroupId) || null;
          if (bindGroup || bgMeta) {
            boundBindGroups.set(cmd.groupIndex, {
              bindGroupId: cmd.bindGroupId,
              bindGroup: bindGroup || null,
              meta: bgMeta,
              hasCbv: Boolean(bgMeta && bgMeta.hasCbv),
              hasTexture: Boolean(bgMeta && bgMeta.hasTexture),
              hasSampler: Boolean(bgMeta && bgMeta.hasSampler),
              entries: bgMeta ? bgMeta.entries : null,
              desc: bindGroup && bindGroup.desc ? bindGroup.desc : null,
            });
          }
          if (bindGroup) {
            const pass =
              activePass ||
              (this._bindGroupKinds.get(cmd.bindGroupId) === 'compute'
                ? ensureComputePass()
                : ensureRenderPass());
            pass.setBindGroup(cmd.groupIndex, bindGroup);
          }
          break;
        }
        case 'draw': {
          const pass = ensureRenderPass();
          maybeSpecializeFallbackPipeline(pass, cmd.vertexCount, cmd.instanceCount);
          pass.draw(cmd.vertexCount, cmd.instanceCount, cmd.firstVertex, cmd.firstInstance);
          this._drawCallCount += 1;
          break;
        }
        case 'drawIndexed': {
          const pass = ensureRenderPass();
          maybeSpecializeFallbackPipeline(pass, cmd.indexCount, cmd.instanceCount);
          pass.drawIndexed(
            cmd.indexCount,
            cmd.instanceCount,
            cmd.firstIndex,
            cmd.baseVertex,
            cmd.firstInstance
          );
          this._drawCallCount += 1;
          break;
        }
        case 'setComputePipeline': {
          const pass = ensureComputePass();
          const pipeline = this.computePipelines.get(cmd.pipelineId);
          if (pipeline) {
            pass.setPipeline(pipeline);
          }
          break;
        }
        case 'dispatchCompute': {
          const pass = ensureComputePass();
          pass.dispatchWorkgroups(cmd.workgroupX, cmd.workgroupY, cmd.workgroupZ);
          this._computeDispatchCount += 1;
          break;
        }
        default:
          if (typeof cmd.run === 'function') cmd.run(passControl);
          break;
      }
    }

    endActivePass();
    if (!renderPassOpened) {
      ensureRenderPass();
      endActivePass();
    }

    let timerResolved = false;
    if (timerSupported && typeof this.passTimer.resolveFrame === 'function') {
      timerResolved = Boolean(this.passTimer.resolveFrame(encoder));
    }

    // Flush batched per-draw dynamic uniform constants once before queue submission.
    if (
      this.uniformRing &&
      !this.uniformRing._destroyed &&
      typeof this.uniformRing.flush === 'function'
    ) {
      this.uniformRing.flush();
    }

    // Unmap any mapped staging chunks before queue submission.
    if (
      this.stagingPool &&
      !this.stagingPool._destroyed &&
      typeof this.stagingPool.prepareSubmit === 'function'
    ) {
      this.stagingPool.prepareSubmit();
    }

    this.device.queue.submit([encoder.finish()]);
    this.frameCount += 1;
    if (this.frameCount === 1) {
      this._notifyFirstPresent({
        frameIndex: 0,
        framesPresented: 1,
        source: 'command-buffer',
        timestampMs: Date.now(),
      });
    }

    const workDoneFence = this._getWorkDoneFence();
    this._endSubsystemsForFrame(workDoneFence);

    if (
      timerResolved &&
      this.passTimer &&
      typeof this.passTimer.collectCompletedReadback === 'function'
    ) {
      const readbackPromise = Promise.resolve(this.passTimer.collectCompletedReadback()).then(
        (timing) => {
          if (timing && !this._destroyed) {
            this._lastPassTimings = timing;
          }
          return timing;
        },
        () => null
      );
      this._pendingTimerReadbacks.add(readbackPromise);
      readbackPromise.finally(() => {
        this._pendingTimerReadbacks.delete(readbackPromise);
      });
    }

    this._resetFrameState();
  }

  attachRing(ringOrSab, options = {}) {
    if (ringOrSab && typeof ringOrSab.drainTo === 'function') {
      this.ring = ringOrSab;
      return this.ring;
    }
    const RingCtor = resolveGpuRingBuffer(options.GpuRingBuffer || this._GpuRingBuffer);
    const ctorOptions = {};
    if (options.byteOffset !== undefined) {
      ctorOptions.byteOffset = options.byteOffset;
    }
    if (options.initialize !== undefined) {
      ctorOptions.initialize = options.initialize;
    }
    if (options.capacityBytes !== undefined) {
      ctorOptions.capacityBytes = options.capacityBytes;
    }
    this.ring = new RingCtor(ringOrSab, ctorOptions);
    return this.ring;
  }

  drainRing() {
    if (this._destroyed || !this.ring || !this.device || !this.ctx) {
      return { frameCount: this.frameCount, presents: 0, bytesDrained: 0 };
    }
    let presents = 0;
    const bytesDrained = this.ring.drainTo((bytes) => {
      const result = this.executeCommandBuffer(bytes);
      presents += result.presents;
    });
    return { frameCount: this.frameCount, presents, bytesDrained };
  }

  executeCommandBuffer(input) {
    if (this._destroyed || !this.device || !this.ctx) {
      return { frameCount: this.frameCount, presents: 0 };
    }

    let view;
    if (
      input instanceof ArrayBuffer ||
      (typeof SharedArrayBuffer !== 'undefined' && input instanceof SharedArrayBuffer)
    ) {
      view = new DataView(input);
    } else if (ArrayBuffer.isView(input)) {
      view = new DataView(input.buffer, input.byteOffset, input.byteLength);
    } else {
      throw new TypeError(
        'WgpuCommandProcessor.executeCommandBuffer expects an ArrayBuffer, SharedArrayBuffer, or ArrayBufferView'
      );
    }

    let offset = 0;
    let presents = 0;

    const prevPendingHints = this._pendingCommandStreamShaderProfiles;
    this._pendingCommandStreamShaderProfiles = this._scanCommandBufferFallbackProfiles(view);
    try {
    while (offset + 8 <= view.byteLength) {
      const opcode = view.getUint16(offset, true);
      const payloadBytes = view.getUint32(offset + 4, true);
      const payloadOffset = offset + 8;
      if (payloadOffset + payloadBytes > view.byteLength) break;

      this._packetCount += 1;
      this._bytesProcessed += 8 + payloadBytes;

      switch (opcode) {
        case OPCODES.NOP:
          break;
        case OPCODES.BEGIN_FRAME:
          if (this._pendingEncoder && this._frameBegunForIndex === -1) {
            this._flushPendingUploads(true);
          }
          this._resetFrameState();
          this._ensureFrameBegun();
          break;
        case OPCODES.CLEAR_COLOR:
          if (payloadBytes >= 16) {
            this._clearColor = {
              r: view.getFloat32(payloadOffset, true),
              g: view.getFloat32(payloadOffset + 4, true),
              b: view.getFloat32(payloadOffset + 8, true),
              a: view.getFloat32(payloadOffset + 12, true),
            };
          }
          break;
        case OPCODES.CREATE_BUFFER:
          if (payloadBytes >= 16) {
            this._handleCreateBuffer(view, payloadOffset);
          }
          break;
        case OPCODES.WRITE_BUFFER:
          if (payloadBytes >= 16) {
            this._handleWriteBuffer(view, payloadOffset, payloadBytes);
          }
          break;
        case OPCODES.CREATE_SHADER_MODULE:
          if (payloadBytes >= 16) {
            this._handleCreateShaderModule(view, payloadOffset, payloadBytes);
          }
          break;
        case OPCODES.CREATE_RENDER_PIPELINE:
          if (payloadBytes >= 24) {
            this._handleCreateRenderPipeline(view, payloadOffset);
          }
          break;
        case OPCODES.SET_PIPELINE:
          if (payloadBytes >= 4) {
            this._passCommands.push({
              type: 'setPipeline',
              pipelineId: view.getUint32(payloadOffset, true),
            });
          }
          break;
        case OPCODES.SET_VERTEX_BUFFER:
          if (payloadBytes >= 16) {
            this._passCommands.push({
              type: 'setVertexBuffer',
              slot: view.getUint32(payloadOffset, true),
              bufferId: view.getUint32(payloadOffset + 4, true),
              offset: Number(view.getBigUint64(payloadOffset + 8, true)),
            });
          }
          break;
        case OPCODES.DRAW:
          if (payloadBytes >= 16) {
            this._passCommands.push({
              type: 'draw',
              vertexCount: view.getUint32(payloadOffset, true),
              instanceCount: view.getUint32(payloadOffset + 4, true),
              firstVertex: view.getUint32(payloadOffset + 8, true),
              firstInstance: view.getUint32(payloadOffset + 12, true),
            });
          }
          break;
        case OPCODES.PRESENT:
          this._flushPresent();
          presents += 1;
          break;
        case OPCODES.DESTROY_RESOURCE:
          if (payloadBytes >= 4) {
            this._handleDestroyResource(view.getUint32(payloadOffset, true));
          }
          break;
        case OPCODES.CREATE_TEXTURE:
          if (payloadBytes >= 32) {
            this._handleCreateTexture(view, payloadOffset);
          }
          break;
        case OPCODES.WRITE_TEXTURE:
          if (payloadBytes >= 48) {
            this._handleWriteTexture(view, payloadOffset, payloadBytes);
          }
          break;
        case OPCODES.CREATE_SAMPLER:
          if (payloadBytes >= 32) {
            this._handleCreateSampler(view, payloadOffset);
          }
          break;
        case OPCODES.CREATE_BIND_GROUP:
          if (payloadBytes >= 16) {
            this._handleCreateBindGroup(view, payloadOffset, payloadBytes);
          }
          break;
        case OPCODES.SET_BIND_GROUP:
          if (payloadBytes >= 8) {
            this._passCommands.push({
              type: 'setBindGroup',
              groupIndex: view.getUint32(payloadOffset, true),
              bindGroupId: view.getUint32(payloadOffset + 4, true),
            });
          }
          break;
        case OPCODES.SET_INDEX_BUFFER:
          if (payloadBytes >= 16) {
            const formatCode = view.getUint32(payloadOffset + 4, true);
            this._passCommands.push({
              type: 'setIndexBuffer',
              bufferId: view.getUint32(payloadOffset, true),
              indexFormat: INDEX_FORMAT_NAMES[formatCode] || 'uint16',
              offset: Number(view.getBigUint64(payloadOffset + 8, true)),
            });
          }
          break;
        case OPCODES.DRAW_INDEXED:
          if (payloadBytes >= 20) {
            this._passCommands.push({
              type: 'drawIndexed',
              indexCount: view.getUint32(payloadOffset, true),
              instanceCount: view.getUint32(payloadOffset + 4, true),
              firstIndex: view.getUint32(payloadOffset + 8, true),
              baseVertex: view.getInt32(payloadOffset + 12, true),
              firstInstance: view.getUint32(payloadOffset + 16, true),
            });
          }
          break;
        case OPCODES.CREATE_COMPUTE_PIPELINE:
          if (payloadBytes >= 16) {
            this._handleCreateComputePipeline(view, payloadOffset);
          }
          break;
        case OPCODES.SET_COMPUTE_PIPELINE:
          if (payloadBytes >= 4) {
            this._passCommands.push({
              type: 'setComputePipeline',
              pipelineId: view.getUint32(payloadOffset, true),
            });
          }
          break;
        case OPCODES.DISPATCH_COMPUTE:
          if (payloadBytes >= 12) {
            this._passCommands.push({
              type: 'dispatchCompute',
              workgroupX: view.getUint32(payloadOffset, true),
              workgroupY: view.getUint32(payloadOffset + 4, true),
              workgroupZ: view.getUint32(payloadOffset + 8, true),
            });
          }
          break;
        default:
          if (this.extension) this.extension.handlePacket(opcode, view, payloadOffset, payloadBytes);
          break;
      }

      offset = payloadOffset + payloadBytes;
    }
    } finally {
      this._pendingCommandStreamShaderProfiles = prevPendingHints;
    }

    return { frameCount: this.frameCount, presents };
  }

  execute(input) {
    return this.executeCommandBuffer(input);
  }

  /**
   * Return aggregated processor and subsystem telemetry (packet counts, frame count,
   * staging pool, dynamic uniform ring, GPU pass timings, VRAM resource registry,
   * render-target pool hit rate, and reflection shader count).
   */
  getStats() {
    const stagingPool =
      this.stagingPool && typeof this.stagingPool.getStats === 'function'
        ? this.stagingPool.getStats()
        : null;
    const uniformRing =
      this.uniformRing && typeof this.uniformRing.getStats === 'function'
        ? this.uniformRing.getStats()
        : null;
    const passTimerBase =
      this.passTimer && typeof this.passTimer.getStats === 'function'
        ? this.passTimer.getStats()
        : null;
    const lastPassTimings = this.getLastPassTimings();
    const passTimer = passTimerBase
      ? { ...passTimerBase, lastTimings: lastPassTimings }
      : null;
    const resourceRegistry =
      this.resourceRegistry && typeof this.resourceRegistry.getStats === 'function'
        ? this.resourceRegistry.getStats()
        : null;
    const rtPool =
      this.rtPool && typeof this.rtPool.getStats === 'function'
        ? this.rtPool.getStats()
        : null;
    const reflectionRegistry =
      this.reflectionRegistry && typeof this.reflectionRegistry.getStats === 'function'
        ? this.reflectionRegistry.getStats()
        : null;
    const shaderFallback = this.getShaderFallbackStats();
    const wgslPackCache =
      this.wgslPackCache && typeof this.wgslPackCache.getStats === 'function'
        ? this.wgslPackCache.getStats()
        : null;
    const splashBridge =
      this.splashFrameBridge && typeof this.splashFrameBridge.getStats === 'function'
        ? this.splashFrameBridge.getStats()
        : null;
    const activeBinkPresenter =
      this.binkPresenter ||
      (this.splashFrameBridge && this.splashFrameBridge.presenter) ||
      null;
    const binkPresenter =
      activeBinkPresenter && typeof activeBinkPresenter.getStats === 'function'
        ? activeBinkPresenter.getStats()
        : splashBridge && splashBridge.presenterStats
          ? splashBridge.presenterStats
          : null;

    return {
      frameCount: this.frameCount,
      framesPresented: this.frameCount,
      splashFramesPresented: this._splashFramesPresented,
      firstPresent: this._firstPresentNotified || this.frameCount > 0,
      packetCount: this._packetCount,
      drawCalls: this._drawCallCount,
      computeDispatches: this._computeDispatchCount,
      bytesProcessed: this._bytesProcessed,
      destroyed: this._destroyed,
      resources: {
        buffers: this.buffers.size,
        textures: this.textures.size,
        textureViews: this.textureViews.size,
        samplers: this.samplers.size,
        bindGroups: this.bindGroups.size,
        shaderModules: this.shaderModules.size,
        pipelines: this.pipelines.size,
        computePipelines: this.computePipelines.size,
      },
      stagingPool,
      uniformRing,
      passTimer,
      lastPassTimings,
      resourceRegistry,
      vramBytes: resourceRegistry ? resourceRegistry.totalBytes : 0,
      rtPool,
      rtPoolHitRate: rtPool ? rtPool.hitRate : 0,
      reflectionRegistry,
      reflectionShaderCount: reflectionRegistry ? reflectionRegistry.shaderCount : 0,
      shaderFallback,
      fallbackProfileCounts: { ...this._fallbackProfileCounts },
      lastFallbackProfile: this._lastFallbackProfile,
      validationErrorCount: this._validationErrorCount,
      validationErrorHistorySize: this._validationErrors.length,
      wgslPackCache,
      splashBridge,
      binkPresenter,
    };
  }

  /**
   * Idempotently destroy all tracked GPU resources and wired subsystems.
   */
  destroy() {
    if (this._destroyed) {
      return;
    }
    this._destroyed = true;
    this._unhookDeviceUncapturedError();
    this._unhookSwapchainValidation();
    if (
      this._swapchain &&
      this._ownsSwapchain &&
      typeof this._swapchain.destroy === 'function'
    ) {
      this._swapchain.destroy();
    }
    this._swapchain = null;
    this._ownsSwapchain = false;
    this._validationErrorCallbacks.clear();
    this._pendingEncoder = null;
    this._hasPendingUploadCommands = false;
    this._passCommands = [];
    this._frameBegunForIndex = -1;

    // 1. Resource registry destroys all registered active & pending-retirement resources.
    const destroyedByRegistry = new Set();
    if (this.resourceRegistry && typeof this.resourceRegistry.destroyAll === 'function') {
      if (typeof this.resourceRegistry.list === 'function') {
        const activeEntries = this.resourceRegistry.list();
        for (let i = 0; i < activeEntries.length; i += 1) {
          destroyedByRegistry.add(activeEntries[i].resource);
        }
      }
      this.resourceRegistry.destroyAll();
    }

    // Destroy any buffers/textures not managed by resourceRegistry.
    for (const buffer of this.buffers.values()) {
      if (buffer && !destroyedByRegistry.has(buffer) && typeof buffer.destroy === 'function') {
        buffer.destroy();
      }
    }
    for (const texture of this.textures.values()) {
      if (texture && !destroyedByRegistry.has(texture) && typeof texture.destroy === 'function') {
        texture.destroy();
      }
    }

    this.buffers.clear();
    this._bufferMeta.clear();
    this.textures.clear();
    this.textureViews.clear();
    this.samplers.clear();
    this.bindGroups.clear();
    this._bindGroupKinds.clear();
    this._bindGroupMeta.clear();
    this.shaderModules.clear();
    this.pipelines.clear();
    this._pipelineMeta.clear();
    this._specializedFallbackPipelines.clear();
    this._specializedFallbackBindGroups.clear();
    this._pendingCommandStreamShaderProfiles = null;
    this._explicitFallbackDrawContext = null;
    this.computePipelines.clear();
    this._pipelineBindGroupLayouts.clear();
    this._computePipelineBindGroupLayouts.clear();
    this._shaderReflectionKeys.clear();

    // 2. Tear down auxiliary subsystems idempotently.
    if (this.stagingPool && typeof this.stagingPool.destroy === 'function') {
      this.stagingPool.destroy();
    }
    if (this.uniformRing && typeof this.uniformRing.destroy === 'function') {
      this.uniformRing.destroy();
    }
    if (this.passTimer && typeof this.passTimer.destroy === 'function') {
      this.passTimer.destroy();
    }
    if (this.rtPool && typeof this.rtPool.destroy === 'function') {
      this.rtPool.destroy();
    }
    if (
      this.reflectionRegistry &&
      this._ownsReflectionRegistry &&
      typeof this.reflectionRegistry.clear === 'function'
    ) {
      this.reflectionRegistry.clear();
    }
    if (
      this.wgslPackCache &&
      this._ownsWgslPackCache &&
      typeof this.wgslPackCache.clear === 'function'
    ) {
      this.wgslPackCache.clear();
    }
    if (this.splashFrameBridge && typeof this.splashFrameBridge.destroy === 'function') {
      this.splashFrameBridge.destroy();
    }
    if (this.binkPresenter && typeof this.binkPresenter.destroy === 'function') {
      this.binkPresenter.destroy();
    }

    this._lastSplashPresentMs = null;
    this._pendingTimerReadbacks.clear();
    this._lastPassTimings = null;
    this.ring = null;
  }
}

const DEFAULT_TELEMETRY_INTERVAL_MS = 250;
const DEFAULT_TELEMETRY_MISSING_LIMIT = 64;

function toNonNegativeInt(value, fallback = 0) {
  if (value === undefined || value === null || value === '') {
    return fallback;
  }
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n) || n < 0) {
    return fallback;
  }
  return Math.floor(n);
}

function toNonNegativeFinite(value) {
  if (value === undefined || value === null || value === '') {
    return null;
  }
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n) || n < 0) {
    return null;
  }
  return n;
}

function sanitizeTelemetryStage(stage) {
  if (typeof stage === 'number') {
    return WGPK_STAGE_NAMES[stage] || 'unknown';
  }
  if (typeof stage === 'string') {
    const key = stage.trim().toLowerCase();
    if (key === 'vertex' || key === 'vs' || key === 'vert' || key === '1') {
      return 'vertex';
    }
    if (
      key === 'fragment' ||
      key === 'fs' ||
      key === 'frag' ||
      key === 'pixel' ||
      key === 'ps' ||
      key === '2'
    ) {
      return 'fragment';
    }
    if (key === 'compute' || key === 'cs' || key === 'comp' || key === '3') {
      return 'compute';
    }
  }
  return 'unknown';
}

function sanitizeMissingShaderRecord(entry) {
  if (!entry || typeof entry !== 'object') {
    return null;
  }
  const rawHash =
    entry.hash !== undefined
      ? entry.hash
      : entry.shaderHash !== undefined
        ? entry.shaderHash
        : entry.hashHex;
  const hashBigInt = normalizeShaderHashBigInt(rawHash);
  if (hashBigInt === null) {
    return null;
  }
  const hash = formatShaderHashHex(hashBigInt);
  const stage = sanitizeTelemetryStage(entry.stage);
  const rawHits =
    entry.hitCount !== undefined
      ? entry.hitCount
      : entry.hits !== undefined
        ? entry.hits
        : entry.drawCount !== undefined
          ? entry.drawCount
          : entry.count;
  const hitCount = rawHits === undefined || rawHits === null ? 1 : toNonNegativeInt(rawHits, 0);
  const firstSeenMs = toNonNegativeFinite(entry.firstSeenMs);
  const lastSeenMs = toNonNegativeFinite(entry.lastSeenMs);

  const sanitized = {
    hash,
    stage,
    hits: hitCount,
    hitCount,
  };
  if (firstSeenMs !== null) {
    sanitized.firstSeenMs = firstSeenMs;
  }
  if (lastSeenMs !== null) {
    sanitized.lastSeenMs = lastSeenMs;
  }
  return sanitized;
}

/**
 * Build a sanitized `{ type: 'gpu-telemetry', frameStats, fallbackStats, missingShaders, timestamp }`
 * payload matching `GpuBootPanelController.update(...)` (zero host paths, zero raw bytes).
 *
 * @param {WgpuCommandProcessor | object | null | undefined} processor
 * @param {{ limit?: number, missingShaderLimit?: number, timestamp?: number, now?: () => number }} [options]
 */
function buildGpuTelemetrySnapshot(processor, options = {}) {
  const opts = options && typeof options === 'object' ? options : {};
  const rawLimit =
    opts.limit !== undefined
      ? opts.limit
      : opts.missingShaderLimit !== undefined
        ? opts.missingShaderLimit
        : DEFAULT_TELEMETRY_MISSING_LIMIT;
  const limit =
    Number.isInteger(rawLimit) && rawLimit >= 0 ? rawLimit : DEFAULT_TELEMETRY_MISSING_LIMIT;

  let stats = null;
  if (processor && typeof processor.getStats === 'function') {
    try {
      stats = processor.getStats();
    } catch (_) {
      stats = null;
    }
  }

  let rawFallback = null;
  if (processor && typeof processor.getShaderFallbackStats === 'function') {
    try {
      rawFallback = processor.getShaderFallbackStats();
    } catch (_) {
      rawFallback = null;
    }
  } else if (stats && stats.shaderFallback && typeof stats.shaderFallback === 'object') {
    rawFallback = stats.shaderFallback;
  }

  let rawMissing = [];
  if (processor && typeof processor.getMissingShaders === 'function') {
    try {
      const list = processor.getMissingShaders({ limit });
      if (Array.isArray(list)) {
        rawMissing = list;
      }
    } catch (_) {
      rawMissing = [];
    }
  }

  const missingShaders = [];
  for (let i = 0; i < rawMissing.length && missingShaders.length < limit; i += 1) {
    const clean = sanitizeMissingShaderRecord(rawMissing[i]);
    if (clean) {
      missingShaders.push(clean);
    }
  }

  const framesPresented = toNonNegativeInt(
    stats && stats.frameCount !== undefined
      ? stats.frameCount
      : stats && stats.framesPresented !== undefined
        ? stats.framesPresented
        : processor && processor.frameCount,
    0
  );
  const droppedFrames = toNonNegativeInt(
    stats && stats.droppedFrames !== undefined
      ? stats.droppedFrames
      : stats && stats.splashBridge && stats.splashBridge.droppedFrames,
    0
  );
  const drawCalls = toNonNegativeInt(stats && stats.drawCalls, 0);
  const computeDispatches = toNonNegativeInt(stats && stats.computeDispatches, 0);
  const packetsProcessed = toNonNegativeInt(
    stats && stats.packetCount !== undefined
      ? stats.packetCount
      : stats && stats.packetsProcessed,
    0
  );

  const packHits = toNonNegativeInt(rawFallback && rawFallback.packHits, 0);
  const fallbackHits = toNonNegativeInt(rawFallback && rawFallback.fallbackHits, 0);
  const builtinHits = toNonNegativeInt(rawFallback && rawFallback.builtinHits, 0);
  const sumHits = packHits + fallbackHits + builtinHits;
  const totalLookups =
    rawFallback && rawFallback.totalLookups !== undefined && rawFallback.totalLookups !== null
      ? toNonNegativeInt(rawFallback.totalLookups, sumHits)
      : rawFallback && rawFallback.totalResolves !== undefined && rawFallback.totalResolves !== null
        ? toNonNegativeInt(rawFallback.totalResolves, sumHits)
        : sumHits;
  const uniqueMissing =
    rawFallback && rawFallback.uniqueMissing !== undefined && rawFallback.uniqueMissing !== null
      ? toNonNegativeInt(rawFallback.uniqueMissing, missingShaders.length)
      : rawFallback && rawFallback.missingCount !== undefined && rawFallback.missingCount !== null
        ? toNonNegativeInt(rawFallback.missingCount, missingShaders.length)
        : missingShaders.length;
  const overflowMissingCount = toNonNegativeInt(rawFallback && rawFallback.overflowMissingCount, 0);

  let hitRatePct;
  if (
    rawFallback &&
    rawFallback.hitRatePct !== undefined &&
    rawFallback.hitRatePct !== null &&
    Number.isFinite(Number(rawFallback.hitRatePct))
  ) {
    const clamped = Math.min(100, Math.max(0, Number(rawFallback.hitRatePct)));
    hitRatePct = Math.round(clamped * 100) / 100;
  } else {
    hitRatePct = totalLookups > 0 ? Math.round((packHits / totalLookups) * 10000) / 100 : 0;
  }

  let timestamp;
  if (typeof opts.timestamp === 'number' && Number.isFinite(opts.timestamp)) {
    timestamp = opts.timestamp;
  } else if (typeof opts.now === 'function') {
    const t = opts.now();
    timestamp = typeof t === 'number' && Number.isFinite(t) ? t : Date.now();
  } else {
    timestamp = Date.now();
  }

  return {
    type: 'gpu-telemetry',
    frameStats: {
      framesPresented,
      droppedFrames,
      drawCalls,
      computeDispatches,
      packetsProcessed,
    },
    fallbackStats: {
      totalLookups,
      totalResolves: totalLookups,
      packHits,
      fallbackHits,
      builtinHits,
      hitRatePct,
      uniqueMissing,
      missingCount: uniqueMissing,
      overflowMissingCount,
    },
    missingShaders,
    timestamp,
  };
}

function createWgpuWorkerHandler(options = {}) {
  const postMessage =
    typeof options.postMessage === 'function' ? options.postMessage : () => {};
  const injectedRingCtor =
    typeof options.GpuRingBuffer === 'function' ? options.GpuRingBuffer : null;
  const telemetryIntervalMs =
    typeof options.telemetryIntervalMs === 'number' &&
    Number.isFinite(options.telemetryIntervalMs) &&
    options.telemetryIntervalMs >= 0
      ? options.telemetryIntervalMs
      : DEFAULT_TELEMETRY_INTERVAL_MS;
  const telemetryMissingLimit =
    Number.isInteger(options.telemetryMissingLimit) && options.telemetryMissingLimit >= 0
      ? options.telemetryMissingLimit
      : DEFAULT_TELEMETRY_MISSING_LIMIT;
  const telemetryNow = typeof options.now === 'function' ? options.now : () => Date.now();
  const onFirstPresent =
    typeof options.onFirstPresent === 'function' ? options.onFirstPresent : null;
  const onValidationError =
    typeof options.onValidationError === 'function' ? options.onValidationError : null;
  let canvas = options.canvas || null;
  let processor = options.processor || null;
  let swapchain = options.swapchain || (processor && processor.swapchain) || null;
  let attachedRing = processor && processor.ring ? processor.ring : null;
  let preloadedInDefaultInit = false;
  let lastTelemetryEmitMs = -Infinity;
  let firstPresentNotified = false;
  let unbindValidationListener = null;

  function emitValidationError(record, totalCount) {
    const cleanRecord = {
      seq: record.seq,
      type: record.type,
      message: record.message,
      timestampMs: record.timestampMs,
    };
    const count = toNonNegativeInt(totalCount, cleanRecord.seq);
    postMessage({
      type: 'gpu-validation-error',
      error: cleanRecord,
      totalCount: count,
    });
    if (onValidationError) {
      try {
        onValidationError({ ...cleanRecord }, count);
      } catch (_) {
        // Non-fatal validation error callback failure.
      }
    }
  }

  function bindProcessorValidationErrors(targetProcessor) {
    if (unbindValidationListener) {
      try {
        unbindValidationListener();
      } catch (_) {
        // Ignore cleanup errors on prior processor/device.
      }
      unbindValidationListener = null;
    }
    if (!targetProcessor) {
      return;
    }
    if (
      swapchain &&
      !targetProcessor.swapchain &&
      typeof targetProcessor.attachSwapchain === 'function'
    ) {
      targetProcessor.attachSwapchain(swapchain);
    }
    if (typeof targetProcessor.addValidationErrorListener === 'function') {
      unbindValidationListener = targetProcessor.addValidationErrorListener(
        (record, totalCount) => {
          emitValidationError(record, totalCount);
        }
      );
      return;
    }
    const dev = targetProcessor.device;
    if (dev && typeof dev.addEventListener === 'function') {
      let localSeq = 0;
      const fallbackHandler = (event) => {
        const rawError =
          event && typeof event === 'object' && 'error' in event ? event.error : event;
        const type = normalizeValidationErrorType(rawError, event);
        const rawMsg =
          rawError && typeof rawError === 'object' && typeof rawError.message === 'string'
            ? rawError.message
            : event && typeof event === 'object' && typeof event.message === 'string'
              ? event.message
              : typeof rawError === 'string'
                ? rawError
                : rawError !== null && rawError !== undefined
                  ? String(rawError)
                  : '';
        const message = sanitizeValidationErrorMessage(rawMsg);
        const tsVal = telemetryNow();
        const timestampMs =
          typeof tsVal === 'number' && Number.isFinite(tsVal) ? tsVal : Date.now();
        localSeq += 1;
        emitValidationError({ seq: localSeq, type, message, timestampMs }, localSeq);
      };
      dev.addEventListener('uncapturederror', fallbackHandler);
      unbindValidationListener = () => {
        if (typeof dev.removeEventListener === 'function') {
          try {
            dev.removeEventListener('uncapturederror', fallbackHandler);
          } catch (_) {
            // Ignore removal failures.
          }
        }
      };
    }
  }

  if (processor) {
    bindProcessorValidationErrors(processor);
  }

  function notifyHandlerFirstPresent(info) {
    if (firstPresentNotified) {
      return false;
    }
    firstPresentNotified = true;
    if (onFirstPresent) {
      try {
        onFirstPresent(info);
      } catch (_) {
        // Non-fatal telemetry callback error.
      }
    }
    return true;
  }

  function emitTelemetry(overrideOptions = {}) {
    const ts = telemetryNow();
    const snap = buildGpuTelemetrySnapshot(processor, {
      limit:
        overrideOptions.limit !== undefined ? overrideOptions.limit : telemetryMissingLimit,
      timestamp: ts,
    });
    if (overrideOptions.id !== undefined) {
      snap.id = overrideOptions.id;
    }
    postMessage(snap);
    return snap;
  }

  function maybeEmitFrameTelemetry(prevFrameCount, nextFrameCount) {
    if (nextFrameCount <= prevFrameCount) {
      return false;
    }
    const nowMs = telemetryNow();
    const reachedFirstFrame = prevFrameCount === 0 && nextFrameCount >= 1;
    const isFrameOne = nextFrameCount === 1;
    const throttleElapsed =
      lastTelemetryEmitMs === -Infinity || nowMs - lastTelemetryEmitMs >= telemetryIntervalMs;

    if (reachedFirstFrame || isFrameOne || throttleElapsed) {
      lastTelemetryEmitMs = nowMs;
      emitTelemetry();
      return true;
    }
    return false;
  }

  async function preloadBootShaderAssets(targetProcessor, initPayload = {}) {
    const summary = {
      shaderPackLoaded: false,
      shaderPackShaders: 0,
      reflectionLoaded: false,
      reflectionShaders: 0,
    };
    if (!targetProcessor || !initPayload || typeof initPayload !== 'object') {
      return summary;
    }

    const activeFetch =
      typeof initPayload.fetch === 'function'
        ? initPayload.fetch
        : typeof options.fetch === 'function'
          ? options.fetch
          : typeof globalThis !== 'undefined' && typeof globalThis.fetch === 'function'
            ? globalThis.fetch
            : null;

    const shaderPackUrl =
      typeof initPayload.shaderPackUrl === 'string' && initPayload.shaderPackUrl.trim().length > 0
        ? initPayload.shaderPackUrl.trim()
        : null;
    const reflectionManifestUrl =
      typeof initPayload.reflectionManifestUrl === 'string' &&
      initPayload.reflectionManifestUrl.trim().length > 0
        ? initPayload.reflectionManifestUrl.trim()
        : null;

    const tasks = [];

    if (shaderPackUrl && activeFetch) {
      tasks.push(
        (async () => {
          try {
            const resp = await activeFetch(shaderPackUrl);
            if (!resp || !resp.ok || typeof resp.arrayBuffer !== 'function') {
              return;
            }
            const buffer = await resp.arrayBuffer();
            if (typeof targetProcessor.loadWgslPack === 'function') {
              targetProcessor.loadWgslPack(buffer, shaderPackUrl);
            } else if (targetProcessor.wgslPackCache) {
              loadWgslPackIntoCache(targetProcessor.wgslPackCache, buffer, shaderPackUrl);
            }
            summary.shaderPackLoaded = true;
            if (
              targetProcessor.wgslPackCache &&
              typeof targetProcessor.wgslPackCache.getStats === 'function'
            ) {
              const packStats = targetProcessor.wgslPackCache.getStats();
              summary.shaderPackShaders = packStats ? packStats.shaderCount || 0 : 0;
            }
          } catch (_) {
            // Non-fatal: worker init must never abort if shader pack fetch/parse fails.
          }
        })()
      );
    }

    if (reflectionManifestUrl && activeFetch) {
      tasks.push(
        (async () => {
          try {
            const resp = await activeFetch(reflectionManifestUrl);
            if (!resp || !resp.ok) {
              return;
            }
            const manifestData =
              typeof resp.text === 'function'
                ? await resp.text()
                : typeof resp.json === 'function'
                  ? await resp.json()
                  : null;
            if (
              manifestData !== null &&
              typeof targetProcessor.loadReflectionManifest === 'function'
            ) {
              targetProcessor.loadReflectionManifest(manifestData);
              summary.reflectionLoaded = true;
              if (typeof targetProcessor.getStats === 'function') {
                const pStats = targetProcessor.getStats();
                summary.reflectionShaders = pStats ? pStats.reflectionShaderCount || 0 : 0;
              }
            }
          } catch (_) {
            // Non-fatal: worker init must never abort if reflection manifest fetch/parse fails.
          }
        })()
      );
    }

    if (tasks.length > 0) {
      await Promise.all(tasks);
    }
    return summary;
  }

  async function defaultInitGpu(offscreen, width, height, initPayload = {}) {
    canvas = offscreen;
    canvas.width = width;
    canvas.height = height;

    if (typeof navigator === 'undefined' || !navigator.gpu) {
      postMessage({ error: 'navigator.gpu unavailable in worker' });
      return null;
    }

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      postMessage({ error: 'no WebGPU adapter available' });
      return null;
    }

    const device = await adapter.requestDevice();
    const format = navigator.gpu.getPreferredCanvasFormat();
    const ctx = canvas.getContext('webgpu');

    const enableSwapchain =
      initPayload.enableSwapchain !== undefined
        ? initPayload.enableSwapchain !== false
        : options.enableSwapchain !== false;
    const injectedSwapchain =
      initPayload.swapchain && typeof initPayload.swapchain === 'object'
        ? initPayload.swapchain
        : options.swapchain && typeof options.swapchain === 'object'
          ? options.swapchain
          : null;

    let activeSwapchain = null;
    let ownsSwapchain = false;
    if (injectedSwapchain) {
      activeSwapchain = injectedSwapchain;
      if (
        !activeSwapchain.isConfigured &&
        typeof activeSwapchain.configure === 'function'
      ) {
        activeSwapchain.configure({
          width,
          height,
          devicePixelRatio: initPayload.devicePixelRatio || 1,
          format,
        });
      }
    } else if (enableSwapchain) {
      const injectedSwapchainCtor =
        initPayload.WgpuSwapchain !== undefined
          ? initPayload.WgpuSwapchain
          : options.WgpuSwapchain;
      const SwapchainCtor =
        injectedSwapchainCtor === null || injectedSwapchainCtor === false
          ? null
          : resolveSubsystemCtor(
              injectedSwapchainCtor,
              'PlayGta6WgpuSwapchain',
              'WgpuSwapchain',
              './wgpu_swapchain.js'
            );
      if (SwapchainCtor) {
        try {
          const swapOpts = {
            ...(options.swapchainOptions || {}),
            ...(initPayload.swapchainOptions || {}),
            canvas,
            device,
            context: ctx,
            format,
            enableDepth:
              initPayload.enableDepth !== undefined
                ? Boolean(initPayload.enableDepth)
                : options.enableDepth !== undefined
                  ? Boolean(options.enableDepth)
                  : false,
            now: telemetryNow,
          };
          activeSwapchain = new SwapchainCtor(swapOpts);
          activeSwapchain.configure({
            width,
            height,
            devicePixelRatio: initPayload.devicePixelRatio || 1,
          });
          ownsSwapchain = true;
        } catch (_) {
          if (activeSwapchain && typeof activeSwapchain.destroy === 'function') {
            try {
              activeSwapchain.destroy();
            } catch (_destroyErr) {
              // Ignore teardown error on fallback.
            }
          }
          activeSwapchain = null;
          ownsSwapchain = false;
        }
      }
    }

    if (!activeSwapchain) {
      ctx.configure({ device, format, alphaMode: 'opaque' });
    }
    swapchain = activeSwapchain;

    processor = new WgpuCommandProcessor(device, ctx, {
      format,
      width,
      height,
      now: telemetryNow,
      swapchain: activeSwapchain,
      ownsSwapchain,
      captureValidationErrors: true,
      maxValidationErrorHistory:
        initPayload.maxValidationErrorHistory ?? options.maxValidationErrorHistory,
      defaultFallbackProfile:
        initPayload.defaultFallbackProfile ?? options.defaultFallbackProfile,
      GpuRingBuffer: injectedRingCtor,
      stagingPool: options.stagingPool,
      uniformRing: options.uniformRing,
      passTimer: options.passTimer,
      reflectionRegistry: options.reflectionRegistry,
      resourceRegistry: options.resourceRegistry,
      rtPool: options.rtPool,
      pipelineCache: options.pipelineCache,
      wgslPackCache: options.wgslPackCache,
      shaderFallbackRouter: options.shaderFallbackRouter,
      ShaderFallbackRouter: options.ShaderFallbackRouter,
      shaderFallbackRouterOptions: options.shaderFallbackRouterOptions,
      binkPresenter: initPayload.binkPresenter ?? options.binkPresenter,
      splashFrameBridge: initPayload.splashFrameBridge ?? options.splashFrameBridge,
      BinkVideoPresenter:
        initPayload.BinkVideoPresenter !== undefined
          ? initPayload.BinkVideoPresenter
          : options.BinkVideoPresenter,
      SplashFrameBridge:
        initPayload.SplashFrameBridge !== undefined
          ? initPayload.SplashFrameBridge
          : options.SplashFrameBridge,
      binkPresenterOptions:
        initPayload.binkPresenterOptions ?? options.binkPresenterOptions,
      splashFrameBridgeOptions:
        initPayload.splashFrameBridgeOptions ??
        initPayload.splashBridgeOptions ??
        options.splashFrameBridgeOptions ??
        options.splashBridgeOptions,
      enableStagingPool: initPayload.enableStagingPool ?? options.enableStagingPool,
      enableUniformRing: initPayload.enableUniformRing ?? options.enableUniformRing,
      enablePassTimer: initPayload.enablePassTimer ?? options.enablePassTimer,
      enableReflectionRegistry:
        initPayload.enableReflectionRegistry ?? options.enableReflectionRegistry,
      enableResourceRegistry:
        initPayload.enableResourceRegistry ?? options.enableResourceRegistry,
      enableRtPool: initPayload.enableRtPool ?? options.enableRtPool,
      enableShaderFallback:
        initPayload.enableShaderFallback ?? options.enableShaderFallback,
      enableSplashBridge:
        initPayload.enableSplashBridge ?? options.enableSplashBridge,
      fallbackShaderHashes:
        initPayload.fallbackShaderHashes ?? options.fallbackShaderHashes,
      reflectionManifest: initPayload.reflectionManifest ?? options.reflectionManifest,
    });
    bindProcessorValidationErrors(processor);
    try {
      if (!globalThis.PlayGta6RenderState && typeof importScripts === 'function') {
        importScripts('./render_state_packets.js');
      }
      if (globalThis.PlayGta6RenderState) globalThis.PlayGta6RenderState.installRenderStatePackets(processor);
      if (!globalThis.PlayGta6TexturePackets && typeof importScripts === 'function') {
        importScripts('./texture_packets.js');
      }
      if (globalThis.PlayGta6TexturePackets) globalThis.PlayGta6TexturePackets.installTexturePackets(processor);
    } catch (err) {
      postMessage({ error: `packet ABI v2 unavailable: ${String(err)}` });
    }
    if (attachedRing) {
      processor.attachRing(attachedRing);
    }

    await preloadBootShaderAssets(processor, initPayload);
    preloadedInDefaultInit = true;

    const info = adapter.info
      ? `${adapter.info.vendor || 'gpu'} ${adapter.info.architecture || ''}`.trim()
      : 'WebGPU';
    postMessage({ ready: true, adapter: info });
    return processor;
  }

  const initGpu = typeof options.initGpu === 'function' ? options.initGpu : defaultInitGpu;

  function attachRingFromMessage(msg) {
    const ringSource = msg.ring || msg.ringSab;
    if (!ringSource) {
      throw new Error('attachRing requires ringSab or ring');
    }
    const RingCtor =
      ringSource && typeof ringSource.drainTo === 'function'
        ? null
        : resolveGpuRingBuffer(msg.GpuRingBuffer || injectedRingCtor);
    attachedRing = RingCtor
      ? new RingCtor(ringSource, { byteOffset: msg.byteOffset || 0 })
      : ringSource;
    if (processor) {
      processor.attachRing(attachedRing);
    }
    postMessage({
      ringAttached: true,
      capacityBytes: attachedRing.capacityBytes,
    });
    return attachedRing;
  }

  async function handleMessage(msg) {
    const payload = msg || {};
    try {
      if (payload.init && payload.canvas) {
        preloadedInDefaultInit = false;
        const created = await initGpu(
          payload.canvas,
          payload.width || 1280,
          payload.height || 720,
          payload
        );
        if (created && created !== processor) {
          processor = created;
          swapchain = created.swapchain || swapchain;
          lastTelemetryEmitMs = -Infinity;
          firstPresentNotified = false;
          bindProcessorValidationErrors(processor);
        }
        if (payload.ringSab || payload.ring) {
          attachRingFromMessage(payload);
        } else if (attachedRing && processor) {
          processor.attachRing(attachedRing);
        }
        if (!preloadedInDefaultInit && processor) {
          await preloadBootShaderAssets(processor, payload);
        }
      } else if (payload.init && processor) {
        await preloadBootShaderAssets(processor, payload);
      } else if (payload.resize && canvas) {
        const nextW = Math.max(1, payload.width | 0);
        const nextH = Math.max(1, payload.height | 0);
        const activeSwapchain = (processor && processor.swapchain) || swapchain;
        if (
          activeSwapchain &&
          !activeSwapchain.isDestroyed &&
          typeof activeSwapchain.resize === 'function'
        ) {
          activeSwapchain.resize(nextW, nextH, payload.devicePixelRatio || 1);
        } else {
          canvas.width = nextW;
          canvas.height = nextH;
        }
        if (processor && typeof processor.onResize === 'function') {
          processor.onResize(nextW, nextH);
        }
      } else if (payload.attachRing) {
        attachRingFromMessage(payload);
      } else if (payload.loadReflectionManifest && processor) {
        processor.loadReflectionManifest(payload.loadReflectionManifest);
        const stats = processor.getStats();
        postMessage({
          id: payload.id,
          reflectionLoaded: true,
          shaderCount: stats.reflectionShaderCount,
        });
      } else if (payload.type === 'request-gpu-telemetry') {
        emitTelemetry({
          id: payload.id,
          limit: payload.limit,
        });
      } else if (payload.missingShaders && processor) {
        const missingOpts =
          payload.limit !== undefined ? { limit: payload.limit } : undefined;
        postMessage({
          id: payload.id,
          missingShaders:
            typeof processor.getMissingShaders === 'function'
              ? processor.getMissingShaders(missingOpts)
              : [],
          shaderFallbackStats:
            typeof processor.getShaderFallbackStats === 'function'
              ? processor.getShaderFallbackStats()
              : null,
        });
      } else if (payload.stats) {
        postMessage({
          id: payload.id,
          stats: processor && typeof processor.getStats === 'function' ? processor.getStats() : null,
        });
      } else if (payload.type === 'present-splash-frame' || payload.type === 'bink-frame') {
        const framePayload =
          payload.frame && typeof payload.frame === 'object' ? payload.frame : payload;
        const requestedFrameIndex =
          framePayload && Number.isInteger(framePayload.frameIndex)
            ? framePayload.frameIndex
            : Number.isInteger(payload.frameIndex)
              ? payload.frameIndex
              : null;

        if (
          !processor ||
          !processor.device ||
          !processor.ctx ||
          typeof processor.presentSplashFrame !== 'function'
        ) {
          const errMsg = !processor
            ? 'WebGPU processor/device is not initialized yet'
            : !processor.device
              ? 'WebGPU device is not ready yet'
              : !processor.ctx
                ? 'WebGPU canvas context is not ready yet'
                : 'WgpuCommandProcessor does not support presentSplashFrame';
          const errorReply = {
            type: 'splash-frame-presented',
            frameIndex: requestedFrameIndex,
            framesPresented: processor ? toNonNegativeInt(processor.frameCount, 0) : 0,
            ok: false,
            error: errMsg,
          };
          if (payload.id !== undefined) {
            errorReply.id = payload.id;
          }
          postMessage(errorReply);
          return;
        }

        const prevFrames = toNonNegativeInt(processor.frameCount, 0);
        const presentOpts = {};
        if (payload.nowMs !== undefined) {
          presentOpts.nowMs = payload.nowMs;
        }
        if (payload.respectPacing !== undefined) {
          presentOpts.respectPacing = payload.respectPacing;
        }
        if (payload.targetWidth !== undefined) {
          presentOpts.targetWidth = payload.targetWidth;
        } else if (canvas && Number.isInteger(canvas.width) && canvas.width > 0) {
          presentOpts.targetWidth = canvas.width;
        }
        if (payload.targetHeight !== undefined) {
          presentOpts.targetHeight = payload.targetHeight;
        } else if (canvas && Number.isInteger(canvas.height) && canvas.height > 0) {
          presentOpts.targetHeight = canvas.height;
        }

        const result = processor.presentSplashFrame(framePayload, presentOpts);
        const framesPresented = toNonNegativeInt(
          result && result.framesPresented !== undefined
            ? result.framesPresented
            : processor.frameCount,
          prevFrames
        );
        const reply = {
          type: 'splash-frame-presented',
          frameIndex:
            result && result.frameIndex !== undefined ? result.frameIndex : requestedFrameIndex,
          framesPresented,
          ok: Boolean(result && result.ok),
        };
        if (payload.id !== undefined) {
          reply.id = payload.id;
        }
        if (result && result.firstPresent) {
          reply.firstPresent = true;
        }
        if (result && !result.ok) {
          if (result.error) {
            reply.error = result.error;
          }
          if (result.reason) {
            reply.reason = result.reason;
          }
        }
        postMessage(reply);

        if (result && result.ok && result.presented) {
          if (result.firstPresent || (prevFrames === 0 && framesPresented >= 1)) {
            const firstPresentInfo = {
              type: 'first-present',
              firstPresent: true,
              source: result.source || 'splash',
              frameIndex: reply.frameIndex,
              framesPresented,
              width: result.width,
              height: result.height,
              timestamp: telemetryNow(),
            };
            if (notifyHandlerFirstPresent(firstPresentInfo)) {
              postMessage(firstPresentInfo);
            }
          }
          maybeEmitFrameTelemetry(prevFrames, framesPresented);
        }
      } else if (payload.destroy && processor) {
        if (unbindValidationListener) {
          try {
            unbindValidationListener();
          } catch (_) {
            // Ignore cleanup errors.
          }
          unbindValidationListener = null;
        }
        if (typeof processor.destroy === 'function') {
          processor.destroy();
        }
        swapchain = null;
        postMessage({ id: payload.id, destroyed: true });
      } else if ((payload.drainRing || payload.type === 'drain-ring') && processor) {
        const prevFrames = toNonNegativeInt(processor.frameCount, 0);
        const result = processor.drainRing();
        if (result.presents > 0) {
          if (prevFrames === 0 && result.frameCount >= 1) {
            notifyHandlerFirstPresent({
              source: 'command-buffer',
              frameIndex: 0,
              framesPresented: result.frameCount,
              timestamp: telemetryNow(),
            });
          }
          postMessage({ frame: result.frameCount });
          maybeEmitFrameTelemetry(prevFrames, result.frameCount);
        }
      } else if (
        processor &&
        (payload.cmdBuffer ||
          ((payload.type === 'submit' || payload.type === 'executeBatch') &&
            (payload.buffer || payload.commands || payload.data)))
      ) {
        const batchBuffer =
          payload.cmdBuffer || payload.buffer || payload.commands || payload.data;
        const prevFrames = toNonNegativeInt(processor.frameCount, 0);
        const result = processor.executeCommandBuffer(batchBuffer);
        if (result.presents > 0) {
          if (prevFrames === 0 && result.frameCount >= 1) {
            notifyHandlerFirstPresent({
              source: 'command-buffer',
              frameIndex: 0,
              framesPresented: result.frameCount,
              timestamp: telemetryNow(),
            });
          }
          postMessage({ frame: result.frameCount });
          maybeEmitFrameTelemetry(prevFrames, result.frameCount);
        }
      }
    } catch (err) {
      postMessage({ error: String(err) });
    }
  }

  return {
    handleMessage,
    get processor() {
      return processor;
    },
    get swapchain() {
      return (processor && processor.swapchain) || swapchain;
    },
    get ring() {
      return attachedRing;
    },
  };
}

const exportedApi = {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  buildGpuTelemetrySnapshot,
  selectFallbackShaderProfile,
  sanitizeValidationErrorMessage,
  normalizeValidationErrorType,
  FALLBACK_PROFILE,
  DEFAULT_MAX_VALIDATION_ERROR_HISTORY,
  MAX_VALIDATION_ERROR_MESSAGE_LENGTH,
  DEFAULT_TELEMETRY_INTERVAL_MS,
  DEFAULT_TELEMETRY_MISSING_LIMIT,
  InlineWgslPackCache,
  parseHashOnlyShaderPayload,
  resolveGpuRingBuffer,
  estimateTextureBytes,
  fnv1a64Hex,
  OPCODES,
  SHADER_STAGE,
  PRIMITIVE_TOPOLOGY,
  BUFFER_USAGE,
  TEXTURE_USAGE,
  TEXTURE_FORMAT,
  TEXTURE_DIMENSION,
  ADDRESS_MODE,
  FILTER_MODE,
  BIND_RESOURCE_TYPE,
  INDEX_FORMAT,
};

if (typeof globalThis !== 'undefined') {
  globalThis.PlayGta6Wgpu = exportedApi;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = exportedApi;
}

if (typeof self !== 'undefined' && typeof self.postMessage === 'function') {
  const workerHandler = createWgpuWorkerHandler({
    postMessage: (data) => self.postMessage(data),
  });
  self.onmessage = async (ev) => {
    await workerHandler.handleMessage(ev ? ev.data : null);
  };
}
