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
const DEFAULT_MAX_VALIDATION_ERROR_POSTS = 64;
const MAX_VALIDATION_ERROR_MESSAGE_LENGTH = 512;
const WGSL_GROUP0_RE = /@group\(\s*0\s*\)/;

const OPCODE_NAMES = Object.freeze({
  [OPCODES.NOP]: 'Nop',
  [OPCODES.BEGIN_FRAME]: 'BeginFrame',
  [OPCODES.CLEAR_COLOR]: 'ClearColor',
  [OPCODES.CREATE_BUFFER]: 'CreateBuffer',
  [OPCODES.WRITE_BUFFER]: 'WriteBuffer',
  [OPCODES.CREATE_SHADER_MODULE]: 'CreateShaderModule',
  [OPCODES.DRAW]: 'Draw',
  [OPCODES.PRESENT]: 'Present',
  [OPCODES.CREATE_RENDER_PIPELINE]: 'CreateRenderPipeline',
  [OPCODES.SET_PIPELINE]: 'SetPipeline',
  [OPCODES.SET_VERTEX_BUFFER]: 'SetVertexBuffer',
  [OPCODES.DESTROY_RESOURCE]: 'DestroyResource',
  [OPCODES.CREATE_TEXTURE]: 'CreateTexture',
  [OPCODES.WRITE_TEXTURE]: 'WriteTexture',
  [OPCODES.CREATE_SAMPLER]: 'CreateSampler',
  [OPCODES.CREATE_BIND_GROUP]: 'CreateBindGroup',
  [OPCODES.SET_BIND_GROUP]: 'SetBindGroup',
  [OPCODES.SET_INDEX_BUFFER]: 'SetIndexBuffer',
  [OPCODES.DRAW_INDEXED]: 'DrawIndexed',
  [OPCODES.CREATE_COMPUTE_PIPELINE]: 'CreateComputePipeline',
  [OPCODES.SET_COMPUTE_PIPELINE]: 'SetComputePipeline',
  [OPCODES.DISPATCH_COMPUTE]: 'DispatchCompute',
  [OPCODES.CREATE_BIND_GROUP_LAYOUT]: 'CreateBindGroupLayout',
  [OPCODES.CREATE_PIPELINE_LAYOUT]: 'CreatePipelineLayout',
  [OPCODES.CREATE_RENDER_PIPELINE2]: 'CreateRenderPipeline2',
  [OPCODES.CREATE_COMPUTE_PIPELINE2]: 'CreateComputePipeline2',
  [OPCODES.BEGIN_RENDER_PASS]: 'BeginRenderPass',
  [OPCODES.END_RENDER_PASS]: 'EndRenderPass',
  [OPCODES.SET_VIEWPORT]: 'SetViewport',
  [OPCODES.SET_SCISSOR_RECT]: 'SetScissorRect',
  [OPCODES.SET_BLEND_CONSTANT]: 'SetBlendConstant',
  [OPCODES.SET_STENCIL_REFERENCE]: 'SetStencilReference',
  [OPCODES.CREATE_BIND_GROUP2]: 'CreateBindGroup2',
  [OPCODES.SET_BIND_GROUP2]: 'SetBindGroup2',
  [OPCODES.CREATE_SAMPLER2]: 'CreateSampler2',
  [OPCODES.CREATE_TEXTURE2]: 'CreateTexture2',
});

function wgslDeclaresGroup0(code) {
  return typeof code === 'string' && code.length > 0 && WGSL_GROUP0_RE.test(code);
}

function opcodeLabel(opcode) {
  return OPCODE_NAMES[opcode] || `Opcode(${opcode})`;
}

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

function resolvePackCacheHasKey(packCache, hash) {
  if (!packCache || typeof packCache !== 'object') {
    return { matched: false, key: null };
  }
  const hasFn =
    typeof packCache.has === 'function'
      ? packCache.has
      : typeof packCache.hasShader === 'function'
        ? packCache.hasShader
        : null;
  if (!hasFn) {
    return { matched: false, key: null };
  }
  const big = normalizeShaderHashBigInt(hash);
  if (big === null) {
    return { matched: false, key: null };
  }
  const hex = formatShaderHashHex(big);
  try {
    if (hasFn.call(packCache, hex)) {
      return { matched: true, key: hex };
    }
  } catch (_) {
    // Fall through to BigInt lookup for caches that only accept BigInt keys.
  }
  try {
    if (hasFn.call(packCache, big)) {
      return { matched: true, key: big };
    }
  } catch (_) {
    // Ignore lookup errors on custom pack caches.
  }
  return { matched: false, key: null };
}

const DEFAULT_STREAM_SHADERS_THRESHOLD_BYTES = 64 * 1024 * 1024;
const SWAPCHAIN_FORMAT_TOKEN_RE = /^[a-z0-9][a-z0-9-]{0,31}$/;

function sanitizeSwapchainFormatToken(raw) {
  if (typeof raw !== 'string') {
    return null;
  }
  const token = raw.trim().toLowerCase();
  if (token.length === 0 || !SWAPCHAIN_FORMAT_TOKEN_RE.test(token)) {
    return null;
  }
  return token;
}

function resolveShaderPackStreamThresholdBytes(primary, fallback) {
  const rawThreshold =
    primary && primary.streamThresholdBytes !== undefined
      ? primary.streamThresholdBytes
      : fallback && fallback.streamThresholdBytes !== undefined
        ? fallback.streamThresholdBytes
        : undefined;
  if (
    typeof rawThreshold === 'number' &&
    Number.isInteger(rawThreshold) &&
    rawThreshold >= 0
  ) {
    return rawThreshold;
  }
  const rawStreamFlag =
    primary && primary.streamShaders !== undefined
      ? primary.streamShaders
      : fallback && fallback.streamShaders !== undefined
        ? fallback.streamShaders
        : undefined;
  if (rawStreamFlag === true) {
    return DEFAULT_STREAM_SHADERS_THRESHOLD_BYTES;
  }
  return undefined;
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

const OPTIONAL_ADAPTER_FEATURES = Object.freeze([
  'texture-compression-bc',
  'depth32float-stencil8',
  'float32-filterable',
  'bgra8unorm-storage',
  'rg11b10ufloat-renderable',
  'texture-formats-tier1',
]);

function selectSupportedAdapterFeatures(adapter) {
  if (!adapter || !adapter.features || typeof adapter.features.has !== 'function') {
    return [];
  }
  const supported = [];
  for (let i = 0; i < OPTIONAL_ADAPTER_FEATURES.length; i += 1) {
    const feature = OPTIONAL_ADAPTER_FEATURES[i];
    try {
      if (adapter.features.has(feature)) {
        supported.push(feature);
      }
    } catch (_) {
      // Ignore feature-probe errors from non-standard or mock adapter feature sets.
    }
  }
  return supported;
}

const OPTIONAL_ADAPTER_LIMIT_SPECS = Object.freeze([
  Object.freeze({ name: 'maxSampledTexturesPerShaderStage', defaultMin: 16, cap: 256 }),
  Object.freeze({ name: 'maxSamplersPerShaderStage', defaultMin: 16, cap: 64 }),
  Object.freeze({ name: 'maxStorageBuffersPerShaderStage', defaultMin: 8, cap: 32 }),
  Object.freeze({ name: 'maxUniformBuffersPerShaderStage', defaultMin: 12, cap: 32 }),
  Object.freeze({ name: 'maxStorageTexturesPerShaderStage', defaultMin: 4, cap: 16 }),
  Object.freeze({ name: 'maxBindingsPerBindGroup', defaultMin: 1000, cap: 4096 }),
  Object.freeze({ name: 'maxBufferSize', defaultMin: 256 * 1024 * 1024, cap: 1024 * 1024 * 1024 }),
  Object.freeze({
    name: 'maxStorageBufferBindingSize',
    defaultMin: 128 * 1024 * 1024,
    cap: 512 * 1024 * 1024,
  }),
  Object.freeze({ name: 'maxComputeWorkgroupStorageSize', defaultMin: 16384, cap: 65536 }),
]);

function selectSupportedAdapterLimits(adapter) {
  if (!adapter || typeof adapter !== 'object') {
    return {};
  }
  let limits;
  try {
    limits = adapter.limits;
  } catch (_) {
    return {};
  }
  if (!limits || typeof limits !== 'object') {
    return {};
  }
  const requiredLimits = {};
  for (let i = 0; i < OPTIONAL_ADAPTER_LIMIT_SPECS.length; i += 1) {
    const { name, defaultMin, cap } = OPTIONAL_ADAPTER_LIMIT_SPECS[i];
    let supported;
    try {
      supported = limits[name];
    } catch (_) {
      continue;
    }
    if (typeof supported !== 'number' || !Number.isFinite(supported)) {
      continue;
    }
    const clamped = Math.min(Math.floor(supported), cap);
    if (clamped > defaultMin) {
      requiredLimits[name] = clamped;
    }
  }
  return requiredLimits;
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

/**
 * Request a GPUDevice from `adapter` using the optional feature / elevated limit negotiation
 * (features + limits, then features only, then a bare device). Shared by initial init and W100
 * device-loss recovery so a recovered device exposes the same capabilities.
 *
 * @param {GPUAdapter} adapter
 * @returns {Promise<{ device: GPUDevice, enabledFeatures: string[] }>}
 */
async function requestNegotiatedDevice(adapter) {
  const requiredFeatures = selectSupportedAdapterFeatures(adapter);
  const requiredLimits = selectSupportedAdapterLimits(adapter);
  const hasRequiredFeatures = requiredFeatures.length > 0;
  const hasRequiredLimits = Object.keys(requiredLimits).length > 0;
  let device;
  let enabledFeatures = [];
  if (hasRequiredFeatures || hasRequiredLimits) {
    const deviceDesc = {};
    if (hasRequiredFeatures) {
      deviceDesc.requiredFeatures = requiredFeatures;
    }
    if (hasRequiredLimits) {
      deviceDesc.requiredLimits = requiredLimits;
    }
    try {
      device = await adapter.requestDevice(deviceDesc);
      enabledFeatures = hasRequiredFeatures ? requiredFeatures.slice() : [];
    } catch (_) {
      if (hasRequiredFeatures && hasRequiredLimits) {
        try {
          device = await adapter.requestDevice({ requiredFeatures });
          enabledFeatures = requiredFeatures.slice();
        } catch (_featureErr) {
          device = await adapter.requestDevice();
          enabledFeatures = [];
        }
      } else {
        device = await adapter.requestDevice();
        enabledFeatures = [];
      }
    }
  } else {
    device = await adapter.requestDevice();
  }
  if (device && device.features && typeof device.features.has === 'function') {
    enabledFeatures = selectSupportedAdapterFeatures(device);
  }
  return { device, enabledFeatures };
}

// W103: concurrent create{Render,Compute}PipelineAsync compiles allowed in flight (async_pipeline_cache.js).
const DEFAULT_MAX_CONCURRENT_PIPELINE_COMPILES = 4;

// W104: live-resource replay log used to recreate guest GPU resources after device-loss recovery.
// The guest never learns that the browser lost its GPUDevice, so every live resource creation
// packet (and a bounded shadow of small/medium buffer contents) is kept and replayed in
// dependency order on the recovered device.
const REPLAY_SHADOW_BUFFER_MAX_BYTES = 2 * 1024 * 1024;
const REPLAY_SHADOW_TOTAL_MAX_BYTES = 64 * 1024 * 1024;
const REPLAY_LOG_MAX_ENTRIES = 65536;
const REPLAY_LOG_MAX_BYTES = 128 * 1024 * 1024;
const REPLAY_LOG_MAX_BIND_GROUPS = 512;
const REPLAY_ASYNC_CHUNK_SIZE = 24;
const REPLAY_PHASE_COUNT = 5;

// opcode -> [resource kind, replay phase (1 = leaf resources ... 5 = bind groups)]
const REPLAY_CREATE_OPCODES = new Map([
  [OPCODES.CREATE_BUFFER, ['buffer', 1]],
  [OPCODES.CREATE_TEXTURE, ['texture', 1]],
  [OPCODES.CREATE_TEXTURE2, ['texture', 1]],
  [OPCODES.CREATE_SAMPLER, ['sampler', 1]],
  [OPCODES.CREATE_SAMPLER2, ['sampler', 1]],
  [OPCODES.CREATE_SHADER_MODULE, ['shaderModule', 1]],
  [OPCODES.CREATE_BIND_GROUP_LAYOUT, ['bindGroupLayout', 2]],
  [OPCODES.CREATE_PIPELINE_LAYOUT, ['pipelineLayout', 3]],
  [OPCODES.CREATE_RENDER_PIPELINE, ['renderPipeline', 4]],
  [OPCODES.CREATE_RENDER_PIPELINE2, ['renderPipeline', 4]],
  [OPCODES.CREATE_COMPUTE_PIPELINE, ['computePipeline', 4]],
  [OPCODES.CREATE_COMPUTE_PIPELINE2, ['computePipeline', 4]],
  [OPCODES.CREATE_BIND_GROUP, ['bindGroup', 5]],
  [OPCODES.CREATE_BIND_GROUP2, ['bindGroup', 5]],
]);

const REPLAY_RESOURCE_KINDS = Object.freeze([
  'buffer',
  'texture',
  'sampler',
  'shaderModule',
  'bindGroupLayout',
  'pipelineLayout',
  'renderPipeline',
  'computePipeline',
  'bindGroup',
]);

// ABI v2 opcodes are only meaningful once the packet extension is installed.
const REPLAY_EXTENSION_OPCODES = new Set([
  OPCODES.CREATE_BIND_GROUP_LAYOUT,
  OPCODES.CREATE_PIPELINE_LAYOUT,
  OPCODES.CREATE_RENDER_PIPELINE2,
  OPCODES.CREATE_COMPUTE_PIPELINE2,
  OPCODES.CREATE_BIND_GROUP2,
  OPCODES.CREATE_SAMPLER2,
  OPCODES.CREATE_TEXTURE2,
]);

function resolvePositiveIntOption(value, fallback) {
  return Number.isInteger(value) && value > 0 ? value : fallback;
}

// W100: bounded WebGPU device-loss recovery.
const MAX_DEVICE_RECOVERY_ATTEMPTS = 3;
const DEFAULT_DEVICE_RECOVERY_MIN_INTERVAL_MS = 1000;
const DEVICE_LOST_ERROR_PATTERN =
  /valid external instance reference no longer exists|instance (?:was |has been |is )?(?:lost|destroyed|dropped)|device (?:is |was |has been )?(?:lost|destroyed)/i;

/**
 * True when `err` looks like a lost GPUDevice / torn-down WebGPU instance (`OperationError`,
 * `A valid external Instance reference no longer exists.`, `... device is lost`).
 *
 * @param {unknown} err
 * @returns {boolean}
 */
function isDeviceLostError(err) {
  if (err === null || err === undefined) {
    return false;
  }
  let name = '';
  let message = '';
  try {
    if (typeof err === 'object') {
      name = typeof err.name === 'string' ? err.name : '';
      message = typeof err.message === 'string' ? err.message : '';
    } else if (typeof err === 'string') {
      message = err;
    }
  } catch (_) {
    return false;
  }
  return name === 'OperationError' || DEVICE_LOST_ERROR_PATTERN.test(message);
}

/**
 * Reduce a `GPUDeviceLostInfo` (`{ reason, message }`) to a sanitized single-line string.
 *
 * @param {unknown} info
 * @returns {string}
 */
function sanitizeDeviceLostInfo(info) {
  let reason = '';
  let message = '';
  try {
    if (info && typeof info === 'object') {
      reason = typeof info.reason === 'string' ? info.reason : '';
      message = typeof info.message === 'string' ? info.message : '';
    } else if (typeof info === 'string') {
      message = info;
    }
  } catch (_) {
    // Fall through with empty fields.
  }
  const safeReason = /^[A-Za-z_-]{1,32}$/.test(reason) ? reason : 'unknown';
  return sanitizeGpuErrorDetail({
    name: `DeviceLost(${safeReason})`,
    message: message || 'device lost',
  });
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
    this._realPackDrawCount = 0;
    this._fallbackDrawCount = 0;
    this._presentCount = 0;
    this._unboundBindGroupCount = 0;
    this._skippedDrawCount = 0;
    this._computeDispatchCount = 0;
    this._bytesProcessed = 0;
    // W99: torn-down/lost GPU instance hardening counters (fence + present submit failures).
    this._workDoneFenceFailures = 0;
    this._presentSubmitFailures = 0;
    // W100: device-loss detection / recovery state (hooked at the end of the constructor).
    this._deviceGeneration = 0;
    this._deviceLostGeneration = -1;
    this._deviceLostHookedDevice = null;
    this._deviceLostCount = 0;
    this._deviceRecoveredCount = 0;
    this._deviceRecoveryFailedCount = 0;
    this._lastDeviceLostReason = null;
    this._deviceLost = false;
    this._deviceRecovering = false;
    this._droppedWhileRecoveringCount = 0;
    this._deviceLossListener =
      typeof options.onDeviceLost === 'function' ? options.onDeviceLost : null;
    this._logger =
      options.logger !== undefined ? options.logger : typeof console !== 'undefined' ? console : null;
    this._ownedSubsystemFactories = {};
    this._v2BindGroupLayouts = new Map();
    this._v2PipelineLayouts = new Map();

    // W104: live-resource replay log + bounded buffer shadow cache for device-loss recovery.
    this._resourceReplayEnabled = options.enableResourceReplay !== false;
    this._replayingResources = false;
    this._replayLog = new Map(); // '<kind>:<id>' -> { kind, id, opcode, phase, payload: Uint8Array }
    this._replayLogBytes = 0;
    this._replayLogMaxEntries = resolvePositiveIntOption(
      options.replayLogMaxEntries,
      REPLAY_LOG_MAX_ENTRIES
    );
    this._replayLogMaxBytes = resolvePositiveIntOption(options.replayLogMaxBytes, REPLAY_LOG_MAX_BYTES);
    this._replayLogMaxBindGroups = resolvePositiveIntOption(
      options.replayMaxBindGroups,
      REPLAY_LOG_MAX_BIND_GROUPS
    );
    this._replayBindGroupKeys = new Set(); // insertion-ordered 'bindGroup:<id>' keys for LRU cap
    this._replayLogOverflowCount = 0;
    this._replayBindGroupEvictions = 0;
    this._shadowBuffers = new Map(); // bufferId -> { data: Uint8Array, dirtyEnd } (LRU order)
    this._shadowBufferBytes = 0;
    this._shadowBufferMaxBytes = resolvePositiveIntOption(
      options.shadowBufferMaxBytes,
      REPLAY_SHADOW_BUFFER_MAX_BYTES
    );
    this._shadowBufferTotalMaxBytes = Math.max(
      this._shadowBufferMaxBytes,
      resolvePositiveIntOption(options.shadowBufferTotalMaxBytes, REPLAY_SHADOW_TOTAL_MAX_BYTES)
    );
    this._shadowBufferEvictions = 0;
    this._replayedResourceCount = 0;
    this._replayedBufferWriteCount = 0;
    this._replayFailedCount = 0;
    this._lastResourceReplay = null;

    // W106: lazy GPUShaderModule compilation — defer device.createShaderModule({ code })
    // until a pipeline actually references the shader id so 4,000+ preload shaders never stall
    // Chrome's GPU process or trigger watchdog loss during boot or device recovery.
    // Values: false (eager), 'pack' (defer pack-resolved WGSL), true (defer all shaders).
    this._lazyShaderCompile =
      options.lazyShaderCompile === 'pack' ? 'pack' : Boolean(options.lazyShaderCompile);
    this._deferredShaderCount = 0;
    this._materializedShaderCount = 0;

    // W136: on-demand HTTP 206 Range shader resolution for ranged WgslPackCache bundles.
    this._nextShaderModuleGen = 0;
    this._shaderModuleGenerations = new Map();
    this._inFlightRangedShaders = new Map();
    this._rangedShaderFetches = 0;
    this._rangedShaderReady = 0;
    this._rangedShaderErrors = 0;

    // Pending staging upload encoder & frame lifecycle tracking
    this._pendingEncoder = null;
    this._hasPendingUploadCommands = false;
    this._frameBegunForIndex = -1;
    this._lastPassTimings = null;
    this._pendingTimerReadbacks = new Set();

    // Optional WebGpuPipelineCache for reflection layout deduplication
    this.pipelineCache = options.pipelineCache || null;

    // W103: bounded-concurrency async PSO compilation. render_state_packets.js routes
    // CreateRenderPipeline2 / CreateComputePipeline2 / v1 CreateComputePipeline through it when the
    // device exposes create*PipelineAsync, so a burst of pipelines no longer stalls the GPU process.
    this._pendingAsyncPipelineSkips = 0;
    this.asyncPipelineCache = null;
    if (options.asyncPipelineCache && typeof options.asyncPipelineCache === 'object') {
      this.asyncPipelineCache = options.asyncPipelineCache;
    } else if (options.enableAsyncPipelines !== false) {
      const AsyncPipelineCacheCtor = resolveSubsystemCtor(
        options.AsyncPipelineCache,
        'PlayGta6AsyncPipelineCache',
        'AsyncPipelineCache',
        './async_pipeline_cache.js'
      );
      if (AsyncPipelineCacheCtor) {
        this.asyncPipelineCache = new AsyncPipelineCacheCtor({
          device,
          maxConcurrent:
            Number.isInteger(options.maxConcurrentPipelineCompiles) &&
            options.maxConcurrentPipelineCompiles >= 1
              ? options.maxConcurrentPipelineCompiles
              : DEFAULT_MAX_CONCURRENT_PIPELINE_COMPILES,
          logger: this._logger ? this._createAsyncPipelineLogger() : null,
        });
      }
    }

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
        this._ownedSubsystemFactories.stagingPool = (dev) =>
          new StagingCtor({ ...(options.stagingPoolOptions || {}), device: dev });
        this.stagingPool = this._ownedSubsystemFactories.stagingPool(device);
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
        this._ownedSubsystemFactories.uniformRing = (dev) =>
          new UniformRingCtor({ ...(options.uniformRingOptions || {}), device: dev });
        this.uniformRing = this._ownedSubsystemFactories.uniformRing(device);
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
        this._ownedSubsystemFactories.passTimer = (dev) =>
          new PassTimerCtor({ ...(options.passTimerOptions || {}), device: dev });
        this.passTimer = this._ownedSubsystemFactories.passTimer(device);
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
        this._ownedSubsystemFactories.rtPool = (dev) =>
          new RtPoolCtor({ ...(options.rtPoolOptions || {}), device: dev });
        this.rtPool = this._ownedSubsystemFactories.rtPool(device);
        this._ownsRtPool = true;
      }
    }

    if (options.reflectionManifest !== undefined && options.reflectionManifest !== null) {
      this.loadReflectionManifest(options.reflectionManifest);
    }

    // 7. WgslPackCache + ShaderFallbackRouter (G28 / S12 / W109 / W121)
    this._shaderPackStreamThresholdBytes = resolveShaderPackStreamThresholdBytes(options, null);
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
    const rawMaxErrPosts = options.maxValidationErrorPosts;
    this._maxValidationErrorPosts =
      Number.isInteger(rawMaxErrPosts) && rawMaxErrPosts >= 1
        ? rawMaxErrPosts
        : DEFAULT_MAX_VALIDATION_ERROR_POSTS;
    this._validationErrors = [];
    this._validationErrorCount = 0;
    this._validationErrorSeq = 0;
    this._validationErrorPostedCount = 0;
    this._validationErrorSuppressedCount = 0;
    this._validationErrorDedup = new Map();
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

    // W100: observe `device.lost` for the initial device.
    this._hookDeviceLost(this.device);

    this._resetFrameState();
  }

  get destroyed() {
    return this._destroyed;
  }

  get framesPresented() {
    return this.frameCount;
  }

  get presentCount() {
    return this._presentCount;
  }

  get realPackDrawCount() {
    return this._realPackDrawCount;
  }

  get fallbackDrawCount() {
    return this._fallbackDrawCount;
  }

  get unboundBindGroupCount() {
    return this._unboundBindGroupCount;
  }

  get skippedDrawCount() {
    return this._skippedDrawCount;
  }

  get swapchain() {
    return this._swapchain;
  }

  get validationErrorCount() {
    return this._validationErrorCount;
  }

  get rangedShaderFetches() {
    return this._rangedShaderFetches;
  }

  get rangedShaderReady() {
    return this._rangedShaderReady;
  }

  get rangedShaderErrors() {
    return this._rangedShaderErrors;
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
   * Clear the bounded validation error history ring and deduplication table
   * (monotonic counter stays intact).
   */
  clearValidationErrors() {
    this._validationErrors.length = 0;
    this._validationErrorDedup.clear();
    this._validationErrorPostedCount = 0;
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
    if (this.asyncPipelineCache && this.asyncPipelineCache.device !== newDevice) {
      this._resetAsyncPipelineCache(newDevice);
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

    this._validationErrorCount += 1;

    const dedupKey = `${type}|${message}`;
    const existingDedup = this._validationErrorDedup.get(dedupKey);
    if (existingDedup) {
      existingDedup.count += 1;
      this._validationErrorSuppressedCount += 1;
      return null;
    }

    if (this._validationErrorPostedCount >= this._maxValidationErrorPosts) {
      this._validationErrorDedup.set(dedupKey, { count: 1, seq: 0, capped: true });
      this._validationErrorSuppressedCount += 1;
      return null;
    }

    this._validationErrorSeq += 1;
    this._validationErrorPostedCount += 1;
    this._validationErrorDedup.set(dedupKey, { count: 1, seq: this._validationErrorSeq });

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

  _recordOpValidationError(operation, errOrMessage, errorType = 'validation') {
    if (this._destroyed) {
      return null;
    }
    const opPrefix = typeof operation === 'string' && operation.length > 0 ? operation : 'WebGPU';
    let detail;
    if (errOrMessage && typeof errOrMessage === 'object' && typeof errOrMessage.message === 'string') {
      detail = errOrMessage.message;
    } else if (typeof errOrMessage === 'string') {
      detail = errOrMessage;
    } else if (errOrMessage !== null && errOrMessage !== undefined) {
      detail = String(errOrMessage);
    } else {
      detail = 'operation failed';
    }
    const trimmed = detail.trim();
    const message =
      trimmed.length === 0
        ? `${opPrefix}: operation failed`
        : trimmed.toLowerCase().startsWith(`${opPrefix.toLowerCase()}:`)
          ? trimmed
          : `${opPrefix}: ${trimmed}`;
    return this._recordValidationError({
      error: {
        type: errorType,
        message,
      },
    });
  }

  _getRenderStateExtension() {
    let ext = this.extension;
    while (ext) {
      if (ext.bindGroupLayouts instanceof Map && ext.pipelineLayouts instanceof Map) {
        return ext;
      }
      ext = ext.next || null;
    }
    return null;
  }

  _getTexturePacketsExtension() {
    let ext = this.extension;
    while (ext) {
      if (
        typeof ext.getStats === 'function' &&
        (ext._bc3dFallbackTextures instanceof Set ||
          (ext.stats &&
            typeof ext.stats === 'object' &&
            ext.stats.bc3dFallbacks !== undefined))
      ) {
        return ext;
      }
      ext = ext.next || null;
    }
    return null;
  }

  _getRenderTargetFormat() {
    if (
      this._swapchain &&
      !this._swapchain.isDestroyed &&
      this._swapchain.usesBlitFallback === true &&
      typeof this._swapchain.streamFormat === 'string' &&
      this._swapchain.streamFormat.length > 0
    ) {
      return this._swapchain.streamFormat;
    }
    return this.format;
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

    const vsOrig = this.ensureShaderModule(pipeMeta.vsShaderId);
    const fsOrig = pipeMeta.fsShaderId ? this.ensureShaderModule(pipeMeta.fsShaderId) : null;
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
        targets: [{ format: this._getRenderTargetFormat() }],
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

  /**
   * Return a work-done fence for the most recent submission, or null when the queue fence is
   * unavailable. The returned promise never rejects: a torn-down/lost GPU instance
   * (`OperationError: A valid external Instance reference no longer exists.`) either throws
   * synchronously or rejects the fence, and both are recorded as a sanitized validation event
   * instead of escaping as an uncaught worker exception or unhandled rejection.
   *
   * @returns {Promise<void> | null}
   */
  _getWorkDoneFence() {
    let queue = null;
    try {
      queue = this.device ? this.device.queue : null;
    } catch (err) {
      this._noteWorkDoneFenceFailure(err);
      return null;
    }
    if (!queue || typeof queue.onSubmittedWorkDone !== 'function') {
      return null;
    }
    let fence;
    try {
      fence = queue.onSubmittedWorkDone();
    } catch (err) {
      this._noteWorkDoneFenceFailure(err);
      return null;
    }
    if (!fence || typeof fence.then !== 'function') {
      return null;
    }
    try {
      return Promise.resolve(fence).then(
        () => undefined,
        (err) => {
          this._noteWorkDoneFenceFailure(err);
          return undefined;
        }
      );
    } catch (err) {
      this._noteWorkDoneFenceFailure(err);
      return null;
    }
  }

  /**
   * Record a failed/rejected `onSubmittedWorkDone()` fence once per processor lifetime as a
   * sanitized validation event; subsequent failures only bump the counter.
   *
   * @param {unknown} err
   */
  _noteWorkDoneFenceFailure(err) {
    this._workDoneFenceFailures += 1;
    if (this._destroyed) {
      return;
    }
    // W100: a torn-down instance / lost device on the fence path triggers recovery.
    this._maybeSignalDeviceLostFromError(err, 'work-done-fence');
    if (this._workDoneFenceFailures !== 1) {
      return;
    }
    try {
      this._recordOpValidationError('OnSubmittedWorkDone', sanitizeGpuErrorDetail(err));
    } catch (_) {
      // Never let telemetry recording escape from fence handling.
    }
  }

  /**
   * Record a failed present encode/submit (e.g. torn-down WebGPU instance) as a sanitized
   * `Present:` validation event without throwing.
   *
   * @param {unknown} err
   */
  _notePresentSubmitFailure(err) {
    this._presentSubmitFailures += 1;
    try {
      this._recordOpValidationError('Present', sanitizeGpuErrorDetail(err));
    } catch (_) {
      // Never let telemetry recording escape from the present path.
    }
    // W100: a torn-down instance / lost device on the present path triggers recovery.
    this._maybeSignalDeviceLostFromError(err, 'present');
  }

  /**
   * W100: emit a concise diagnostic line through the configured logger (default `console`).
   *
   * @param {'info' | 'warn' | 'error'} level
   * @param {string} message
   */
  _logDiagnostic(level, message) {
    const logger = this._logger;
    if (!logger) {
      return;
    }
    try {
      if (typeof logger === 'function') {
        logger(level, message);
      } else if (typeof logger[level] === 'function') {
        logger[level](message);
      } else if (typeof logger.log === 'function') {
        logger.log(message);
      }
    } catch (_) {
      // Logging must never break the render loop.
    }
  }

  /**
   * W100: attach a `device.lost` observer to `device` once. Losses of a device that is no longer
   * current (replaced by recovery) and intentional `device.destroy()` (`reason: 'destroyed'`) or
   * processor teardown are ignored.
   *
   * @param {GPUDevice | null | undefined} device
   * @returns {boolean}
   */
  _hookDeviceLost(device) {
    if (!device || typeof device !== 'object' || device === this._deviceLostHookedDevice) {
      return false;
    }
    let lost;
    try {
      lost = device.lost;
    } catch (_) {
      return false;
    }
    if (!lost || typeof lost.then !== 'function') {
      return false;
    }
    this._deviceLostHookedDevice = device;
    const generation = this._deviceGeneration;
    try {
      Promise.resolve(lost).then(
        (info) => {
          this._handleDeviceLostInfo(info, device, generation);
        },
        () => {
          // A rejected `lost` promise carries no actionable information.
        }
      );
    } catch (_) {
      return false;
    }
    return true;
  }

  _handleDeviceLostInfo(info, device, generation) {
    if (this._destroyed || generation !== this._deviceGeneration || this.device !== device) {
      return;
    }
    let reason = null;
    try {
      reason = info && typeof info === 'object' ? info.reason : null;
    } catch (_) {
      reason = null;
    }
    if (reason === 'destroyed') {
      // `device.destroy()` was called intentionally by the host; do not recover.
      return;
    }
    this._signalDeviceLost(sanitizeDeviceLostInfo(info), 'device.lost');
  }

  /**
   * W100: signal device loss when `err` is a torn-down instance / lost device error.
   *
   * @param {unknown} err
   * @param {string} source
   * @returns {boolean}
   */
  _maybeSignalDeviceLostFromError(err, source) {
    try {
      if (!isDeviceLostError(err)) {
        return false;
      }
      return this._signalDeviceLost(sanitizeGpuErrorDetail(err), source);
    } catch (_) {
      return false;
    }
  }

  /**
   * W100: record a device loss (counted once per device generation) and notify the registered
   * device-loss listener (the worker handler schedules bounded recovery). Never throws.
   *
   * @param {string} reason Sanitized reason string.
   * @param {string} source
   * @returns {boolean}
   */
  _signalDeviceLost(reason, source) {
    if (this._destroyed || this._deviceRecovering) {
      return false;
    }
    const firstForGeneration = this._deviceLostGeneration !== this._deviceGeneration;
    if (firstForGeneration) {
      this._deviceLostGeneration = this._deviceGeneration;
      this._deviceLost = true;
      this._deviceLostCount += 1;
      this._lastDeviceLostReason = typeof reason === 'string' && reason.length > 0 ? reason : null;
      this._logDiagnostic(
        'warn',
        `[wgpu_worker] WebGPU device lost (${source}): ${this._lastDeviceLostReason || 'unknown'}`
      );
    }
    const listener = this._deviceLossListener;
    if (listener) {
      try {
        listener({
          reason: this._lastDeviceLostReason,
          source,
          generation: this._deviceGeneration,
          firstForGeneration,
        });
      } catch (_) {
        // Listener failures must not escape into the render loop.
      }
    }
    return true;
  }

  /**
   * W100: register the device-loss listener (called synchronously on loss; `null` clears it).
   *
   * @param {((event: { reason: string | null, source: string, generation: number, firstForGeneration: boolean }) => void) | null} listener
   */
  setDeviceLossListener(listener) {
    this._deviceLossListener = typeof listener === 'function' ? listener : null;
  }

  get isDeviceLost() {
    return this._deviceLost;
  }

  get isDeviceRecovering() {
    return this._deviceRecovering;
  }

  /** W103: adapt the processor logger to AsyncPipelineCache's { warn, debug }(message, detail). */
  _createAsyncPipelineLogger() {
    const log = (level) => (message, detail) => {
      const suffix = detail && typeof detail.message === 'string' ? `: ${detail.message}` : '';
      this._logDiagnostic(level, `[wgpu_worker] ${message}${suffix}`);
    };
    return { warn: log('warn'), debug: log('debug') };
  }

  /** W103: point the async pipeline cache at `device`, settling pending compiles as stale. */
  _resetAsyncPipelineCache(device) {
    const cache = this.asyncPipelineCache;
    if (!cache || typeof cache.reset !== 'function') {
      return;
    }
    try {
      cache.reset(device || null);
    } catch (err) {
      this._logDiagnostic(
        'warn',
        `[wgpu_worker] async pipeline cache reset failed: ${sanitizeGpuErrorDetail(err)}`
      );
    }
  }

  /**
   * W103: whether pipeline `id` is still compiling asynchronously (draws/dispatches that use it
   * are skipped without a validation error until it is ready).
   *
   * @param {number} id
   * @returns {boolean}
   */
  isAsyncPipelinePending(id) {
    const cache = this.asyncPipelineCache;
    if (!cache || typeof cache.isPending !== 'function') {
      return false;
    }
    try {
      return cache.isPending(id) === true;
    } catch (_) {
      return false;
    }
  }

  /**
   * W103: AsyncPipelineCache counters plus draws/dispatches skipped while a pipeline compiled,
   * or null when async pipeline compilation is disabled.
   */
  getAsyncPipelineStats() {
    const cache = this.asyncPipelineCache;
    if (!cache || typeof cache.getStats !== 'function') {
      return null;
    }
    let stats;
    try {
      stats = cache.getStats();
    } catch (_) {
      return null;
    }
    let renderStateSkips = 0;
    const ext = this.extension;
    if (ext && ext.telemetry && Number.isInteger(ext.telemetry.pendingAsyncPipelineSkips)) {
      renderStateSkips = ext.telemetry.pendingAsyncPipelineSkips;
    }
    return {
      ...(stats && typeof stats === 'object' ? stats : {}),
      pendingAsyncPipelineSkips: this._pendingAsyncPipelineSkips + renderStateSkips,
    };
  }

  /**
   * W100: sanitized device-loss telemetry counters.
   */
  getDeviceLossStats() {
    return {
      deviceLostCount: this._deviceLostCount,
      deviceRecoveredCount: this._deviceRecoveredCount,
      deviceRecoveryFailedCount: this._deviceRecoveryFailedCount,
      lastDeviceLostReason: this._lastDeviceLostReason,
      deviceRecovering: this._deviceRecovering,
      deviceLost: this._deviceLost,
      droppedWhileRecovering: this._droppedWhileRecoveringCount,
      replayedResourceCount: this._replayedResourceCount,
      replayedBufferWriteCount: this._replayedBufferWriteCount,
    };
  }

  /**
   * W100: enter recovery mode. Submissions are dropped (no throw, no frameCount bump) until
   * `endDeviceRecovery()` is called.
   *
   * @returns {boolean}
   */
  beginDeviceRecovery() {
    if (this._destroyed) {
      return false;
    }
    this._deviceRecovering = true;
    this._pendingEncoder = null;
    this._hasPendingUploadCommands = false;
    this._passCommands = [];
    return true;
  }

  /**
   * W100: leave recovery mode and update recovered / failed counters.
   *
   * @param {boolean} success
   */
  endDeviceRecovery(success) {
    const wasRecovering = this._deviceRecovering;
    this._deviceRecovering = false;
    if (!wasRecovering) {
      return;
    }
    if (success && !this._destroyed) {
      this._deviceRecoveredCount += 1;
      this._deviceLost = false;
    } else {
      this._deviceRecoveryFailedCount += 1;
    }
  }

  /**
   * W104: record one packet into the live-resource replay log. Creation packets are copied
   * (latest creation of an id wins), DestroyResource forgets every record of the id, and
   * WriteBuffer updates the bounded shadow copy of the buffer contents. Never records while
   * `replayRecordedResources()` runs.
   */
  _recordReplayPacket(opcode, view, payloadOffset, payloadBytes) {
    if (!this._resourceReplayEnabled || this._replayingResources || payloadBytes < 4) {
      return;
    }
    if (opcode === OPCODES.WRITE_BUFFER) {
      this._recordShadowBufferWrite(view, payloadOffset, payloadBytes);
      return;
    }
    if (opcode === OPCODES.DESTROY_RESOURCE) {
      this._forgetReplayResource(view.getUint32(payloadOffset, true));
      return;
    }
    const spec = REPLAY_CREATE_OPCODES.get(opcode);
    if (!spec) {
      return;
    }
    const [kind, phase] = spec;
    const id = view.getUint32(payloadOffset, true);
    const key = `${kind}:${id}`;
    const previous = this._replayLog.get(key);
    if (previous) {
      this._replayLog.delete(key);
      this._replayLogBytes -= previous.payload.byteLength;
      if (kind === 'bindGroup') {
        this._replayBindGroupKeys.delete(key);
      }
    }
    if (kind === 'buffer') {
      // A re-created buffer starts zero-filled; its old contents must not be replayed.
      this._dropShadowBuffer(id);
    }
    if (kind === 'bindGroup') {
      while (this._replayBindGroupKeys.size >= this._replayLogMaxBindGroups) {
        const oldestKey = this._replayBindGroupKeys.values().next().value;
        this._replayBindGroupKeys.delete(oldestKey);
        const evicted = this._replayLog.get(oldestKey);
        if (evicted) {
          this._replayLog.delete(oldestKey);
          this._replayLogBytes -= evicted.payload.byteLength;
          this._replayBindGroupEvictions += 1;
        }
      }
    }
    if (
      this._replayLog.size >= this._replayLogMaxEntries ||
      this._replayLogBytes + payloadBytes > this._replayLogMaxBytes
    ) {
      this._replayLogOverflowCount += 1;
      return;
    }
    const payload = new Uint8Array(view.buffer, view.byteOffset + payloadOffset, payloadBytes).slice();
    this._replayLog.set(key, { kind, id, opcode, phase, payload });
    this._replayLogBytes += payloadBytes;
    if (kind === 'bindGroup') {
      this._replayBindGroupKeys.add(key);
    }
  }

  /**
   * W104: record the resource packets of `view` starting at `offset` without executing them
   * (command buffers dropped while the device is lost / recovering).
   */
  _recordReplayPacketsFrom(view, offset) {
    if (!this._resourceReplayEnabled || this._replayingResources) {
      return;
    }
    let cursor = offset;
    while (cursor + 8 <= view.byteLength) {
      const opcode = view.getUint16(cursor, true);
      const payloadBytes = view.getUint32(cursor + 4, true);
      const payloadOffset = cursor + 8;
      if (payloadOffset + payloadBytes > view.byteLength) {
        break;
      }
      this._recordReplayPacket(opcode, view, payloadOffset, payloadBytes);
      cursor = payloadOffset + payloadBytes;
    }
  }

  _forgetReplayResource(id) {
    for (let i = 0; i < REPLAY_RESOURCE_KINDS.length; i += 1) {
      const key = `${REPLAY_RESOURCE_KINDS[i]}:${id}`;
      const entry = this._replayLog.get(key);
      if (entry) {
        this._replayLog.delete(key);
        this._replayLogBytes -= entry.payload.byteLength;
        if (entry.kind === 'bindGroup') {
          this._replayBindGroupKeys.delete(key);
        }
      }
    }
    this._dropShadowBuffer(id);
  }

  _dropShadowBuffer(id) {
    const shadow = this._shadowBuffers.get(id);
    if (shadow) {
      this._shadowBuffers.delete(id);
      this._shadowBufferBytes -= shadow.data.byteLength;
    }
  }

  _recordShadowBufferWrite(view, payloadOffset, payloadBytes) {
    if (payloadBytes < 16) {
      return;
    }
    const id = view.getUint32(payloadOffset, true);
    const dstOffset = view.getUint32(payloadOffset + 4, true);
    const dataLen = view.getUint32(payloadOffset + 8, true);
    if (16 + dataLen > payloadBytes || dataLen === 0) {
      return;
    }
    const created = this._replayLog.get(`buffer:${id}`);
    if (!created || created.payload.byteLength < 16) {
      return;
    }
    const createView = new DataView(
      created.payload.buffer,
      created.payload.byteOffset,
      created.payload.byteLength
    );
    const usage = createView.getUint32(4, true);
    const byteSize = Number(createView.getBigUint64(8, true));
    if (
      (usage & BUFFER_USAGE.COPY_DST) === 0 ||
      !Number.isSafeInteger(byteSize) ||
      byteSize < 4 ||
      byteSize > this._shadowBufferMaxBytes ||
      dstOffset >= byteSize
    ) {
      return;
    }
    let shadow = this._shadowBuffers.get(id);
    if (shadow) {
      // LRU: move to the most-recently-written end.
      this._shadowBuffers.delete(id);
    } else {
      shadow = { data: new Uint8Array(byteSize), dirtyEnd: 0 };
      this._shadowBufferBytes += byteSize;
    }
    this._shadowBuffers.set(id, shadow);
    const copyLen = Math.min(dataLen, byteSize - dstOffset);
    shadow.data.set(
      new Uint8Array(view.buffer, view.byteOffset + payloadOffset + 16, copyLen),
      dstOffset
    );
    shadow.dirtyEnd = Math.max(shadow.dirtyEnd, dstOffset + copyLen);

    while (this._shadowBufferBytes > this._shadowBufferTotalMaxBytes && this._shadowBuffers.size > 1) {
      const oldestId = this._shadowBuffers.keys().next().value;
      this._dropShadowBuffer(oldestId);
      this._shadowBufferEvictions += 1;
    }
  }

  /** Total replayable bytes of a shadow (WebGPU writeBuffer sizes must be 4-byte multiples). */
  _shadowWriteLength(shadow) {
    const maxLen = shadow.data.byteLength - (shadow.data.byteLength % 4);
    const aligned = shadow.dirtyEnd + ((4 - (shadow.dirtyEnd % 4)) % 4);
    return Math.min(aligned, maxLen);
  }

  /**
   * W104: recreate every recorded live resource on the current (recovered) device in dependency
   * order — leaf resources (buffers, textures, samplers, shader modules), shadowed buffer
   * contents, bind group layouts, pipeline layouts, pipelines, then bind groups. Each packet is
   * isolated so one stale or invalid record never aborts the rest of the replay. Must run after
   * `rebindDevice()` and after the ABI v2 packet extensions were reinstalled on the new device.
   *
   * @returns {{ replayedResources: number, failedResources: number, replayedBufferWrites: number }}
   */
  _collectOrderedReplayEntries() {
    const phases = [];
    for (let p = 0; p < REPLAY_PHASE_COUNT; p += 1) {
      phases.push([]);
    }
    for (const entry of this._replayLog.values()) {
      phases[entry.phase - 1].push(entry);
    }
    const ordered = [];
    for (let p = 0; p < phases.length; p += 1) {
      for (const entry of phases[p]) {
        ordered.push(entry);
      }
    }
    return ordered;
  }

  replayRecordedResources() {
    const summary = { replayedResources: 0, failedResources: 0, replayedBufferWrites: 0 };
    if (
      this._destroyed ||
      this._replayingResources ||
      !this.device ||
      (this._replayLog.size === 0 && this._shadowBuffers.size === 0)
    ) {
      return summary;
    }

    const ordered = this._collectOrderedReplayEntries();
    const prevPendingHints = this._pendingCommandStreamShaderProfiles;
    const prevPassCommands = this._passCommands;
    this._replayingResources = true;
    try {
      this._pendingCommandStreamShaderProfiles = null;
      this._passCommands = [];
      let shadowWritesDone = false;
      for (let i = 0; i < ordered.length; i += 1) {
        const entry = ordered[i];
        if (!shadowWritesDone && entry.phase > 1) {
          summary.replayedBufferWrites += this._replayShadowBufferWrites();
          shadowWritesDone = true;
        }
        if (this._replayOneEntry(entry)) {
          summary.replayedResources += 1;
        } else {
          summary.failedResources += 1;
        }
      }
      if (!shadowWritesDone) {
        summary.replayedBufferWrites += this._replayShadowBufferWrites();
      }
    } finally {
      this._passCommands = prevPassCommands;
      this._pendingCommandStreamShaderProfiles = prevPendingHints;
      this._replayingResources = false;
    }

    this._replayedResourceCount += summary.replayedResources;
    this._replayedBufferWriteCount += summary.replayedBufferWrites;
    this._replayFailedCount += summary.failedResources;
    this._lastResourceReplay = { ...summary, deviceGeneration: this._deviceGeneration };
    this._logDiagnostic(
      'info',
      `[wgpu_worker] replayed ${summary.replayedResources} GPU resources and ${summary.replayedBufferWrites} buffer writes after device recovery` +
        (summary.failedResources > 0 ? ` (${summary.failedResources} failed)` : '')
    );
    return summary;
  }

  /**
   * W105: chunked async resource replay for live device recovery. Yields between small batches
   * so Chrome's GPU IPC drains cleanly without stalling the worker or re-triggering TDR.
   */
  async replayRecordedResourcesAsync(opts = {}) {
    const summary = { replayedResources: 0, failedResources: 0, replayedBufferWrites: 0, aborted: false };
    if (
      this._destroyed ||
      this._replayingResources ||
      !this.device ||
      (this._replayLog.size === 0 && this._shadowBuffers.size === 0)
    ) {
      return summary;
    }

    const chunkSize = resolvePositiveIntOption(opts && opts.chunkSize, REPLAY_ASYNC_CHUNK_SIZE);
    const ordered = this._collectOrderedReplayEntries();
    const startGen = this._deviceGeneration;
    const startMs = Date.now();
    const logMiB = (this._replayLogBytes / (1024 * 1024)).toFixed(1);
    this._logDiagnostic(
      'info',
      `[wgpu_worker] starting GPU resource replay (${ordered.length} entries, ${logMiB} MiB, ${this._shadowBuffers.size} shadow buffers)`
    );

    const prevPendingHints = this._pendingCommandStreamShaderProfiles;
    const prevPassCommands = this._passCommands;
    this._replayingResources = true;
    try {
      this._pendingCommandStreamShaderProfiles = null;
      this._passCommands = [];
      let shadowWritesDone = false;
      for (let i = 0; i < ordered.length; i += 1) {
        if (this._destroyed || !this.device || this._deviceGeneration !== startGen) {
          summary.aborted = true;
          break;
        }
        const entry = ordered[i];
        if (!shadowWritesDone && entry.phase > 1) {
          summary.replayedBufferWrites += this._replayShadowBufferWrites();
          shadowWritesDone = true;
        }
        if (this._replayOneEntry(entry)) {
          summary.replayedResources += 1;
        } else {
          summary.failedResources += 1;
        }
        if ((i + 1) % chunkSize === 0 && i + 1 < ordered.length) {
          await new Promise((resolve) => setTimeout(resolve, 0));
        }
      }
      if (!shadowWritesDone && !summary.aborted) {
        summary.replayedBufferWrites += this._replayShadowBufferWrites();
      }
    } finally {
      this._passCommands = prevPassCommands;
      this._pendingCommandStreamShaderProfiles = prevPendingHints;
      this._replayingResources = false;
    }

    this._replayedResourceCount += summary.replayedResources;
    this._replayedBufferWriteCount += summary.replayedBufferWrites;
    this._replayFailedCount += summary.failedResources;
    this._lastResourceReplay = { ...summary, deviceGeneration: this._deviceGeneration };
    const elapsedMs = Math.max(0, Date.now() - startMs);
    this._logDiagnostic(
      'info',
      `[wgpu_worker] replayed ${summary.replayedResources} GPU resources and ${summary.replayedBufferWrites} buffer writes in ${elapsedMs}ms after device recovery` +
        (summary.failedResources > 0 ? ` (${summary.failedResources} failed)` : '') +
        (summary.aborted ? ' (aborted)' : '')
    );
    return summary;
  }

  _replayOneEntry(entry) {
    if (REPLAY_EXTENSION_OPCODES.has(entry.opcode) && !this.extension) {
      return false;
    }
    const entryView = new DataView(
      entry.payload.buffer,
      entry.payload.byteOffset,
      entry.payload.byteLength
    );
    const errorsBefore = this._validationErrorCount;
    try {
      this._dispatchPacket(entry.opcode, entryView, 0, entry.payload.byteLength);
    } catch (_) {
      return false;
    }
    return this._validationErrorCount === errorsBefore;
  }

  _replayShadowBufferWrites() {
    const queue = this.device && this.device.queue;
    if (!queue || typeof queue.writeBuffer !== 'function') {
      return 0;
    }
    let written = 0;
    for (const [id, shadow] of this._shadowBuffers) {
      const buffer = this.buffers.get(id);
      const writeLen = this._shadowWriteLength(shadow);
      if (!buffer || writeLen <= 0) {
        continue;
      }
      try {
        queue.writeBuffer(buffer, 0, shadow.data, 0, writeLen);
        written += 1;
      } catch (_) {
        // A stale shadow never aborts the remaining replay.
      }
    }
    return written;
  }

  /**
   * W104: replay-log / shadow-cache counters (sanitized integers only).
   */
  getResourceReplayStats() {
    return {
      recordedResourceCount: this._replayLog.size,
      recordedResourceBytes: this._replayLogBytes,
      replayLogOverflowCount: this._replayLogOverflowCount,
      replayBindGroupEvictions: this._replayBindGroupEvictions,
      shadowBufferCount: this._shadowBuffers.size,
      shadowBufferBytes: this._shadowBufferBytes,
      shadowBufferEvictions: this._shadowBufferEvictions,
      replayedResourceCount: this._replayedResourceCount,
      replayedBufferWriteCount: this._replayedBufferWriteCount,
      replayFailedCount: this._replayFailedCount,
    };
  }

  /**
   * W100: release every GPU resource handle that belongs to the (dead) current device. IDs that
   * the guest references afterwards resolve to `null` and are skipped by the existing lookup
   * paths instead of throwing. Shader WGSL sources in the pack cache are preserved.
   */
  _resetDeviceResources() {
    this._pendingEncoder = null;
    this._hasPendingUploadCommands = false;
    this._passCommands = [];
    this._frameBegunForIndex = -1;

    const safeDestroy = (resource) => {
      if (resource && typeof resource.destroy === 'function') {
        try {
          resource.destroy();
        } catch (_) {
          // Destroying objects of a lost device may throw; they are released either way.
        }
      }
    };

    if (this.resourceRegistry && typeof this.resourceRegistry.destroyAll === 'function') {
      try {
        this.resourceRegistry.destroyAll();
      } catch (_) {
        // Ignore registry teardown failures on a lost device.
      }
    }
    for (const buffer of this.buffers.values()) {
      safeDestroy(buffer);
    }
    for (const texture of this.textures.values()) {
      safeDestroy(texture);
    }

    this.buffers.clear();
    this._bufferMeta.clear();
    this.textures.clear();
    this.textureViews.clear();
    this.samplers.clear();
    this.bindGroups.clear();
    this._bindGroupKinds.clear();
    this._bindGroupMeta.clear();
    this._v2BindGroupLayouts.clear();
    this._v2PipelineLayouts.clear();
    this.shaderModules.clear();
    this._shaderModuleGenerations.clear();
    this._inFlightRangedShaders.clear();
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
    this._pendingTimerReadbacks.clear();
    this._lastPassTimings = null;

    // Packet ABI v2 extensions own device objects too; the worker handler reinstalls them.
    this.extension = null;
    if (Object.prototype.hasOwnProperty.call(this, '__playgta6RsActiveHandler')) {
      this.__playgta6RsActiveHandler = null;
    }

    // Owned splash/Bink presenters are rebuilt lazily by `_ensureSplashBridge()`.
    if (this.splashFrameBridge && this._ownsSplashFrameBridge) {
      safeDestroy(this.splashFrameBridge);
      this.splashFrameBridge = null;
      this._ownsSplashFrameBridge = false;
    }
    if (this.binkPresenter && this._ownsBinkPresenter) {
      safeDestroy(this.binkPresenter);
      this.binkPresenter = null;
      this._ownsBinkPresenter = false;
    }
  }

  /**
   * W100: rebuild owned device-bound subsystems (staging pool, uniform ring, pass timer, RT pool)
   * on `device`. A subsystem that fails to rebuild is disabled instead of throwing.
   *
   * @param {GPUDevice} device
   */
  _recreateOwnedSubsystems(device) {
    const specs = [
      ['stagingPool', '_ownsStagingPool'],
      ['uniformRing', '_ownsUniformRing'],
      ['passTimer', '_ownsPassTimer'],
      ['rtPool', '_ownsRtPool'],
    ];
    for (let i = 0; i < specs.length; i += 1) {
      const [key, ownsKey] = specs[i];
      const factory = this._ownedSubsystemFactories[key];
      if (!this[ownsKey] || typeof factory !== 'function') {
        continue;
      }
      const prev = this[key];
      if (prev && typeof prev.destroy === 'function') {
        try {
          prev.destroy();
        } catch (_) {
          // Ignore teardown failures of subsystems bound to the lost device.
        }
      }
      try {
        this[key] = factory(device);
      } catch (err) {
        this[key] = null;
        this[ownsKey] = false;
        this._logDiagnostic(
          'warn',
          `[wgpu_worker] ${key} unavailable after device recovery: ${sanitizeGpuErrorDetail(err)}`
        );
      }
    }
  }

  /**
   * #709 / W112 / W121: Configure the WebGPU canvas context / swapchain to match the swapchain
   * attachment format declared by the command stream.
   *
   * When an attached `WgpuSwapchain` supports `configureForStreamFormat()`, valid direct canvas
   * formats (`bgra8unorm`, `rgba8unorm`, `rgba16float`) stay on the zero-overhead direct canvas
   * path, while non-canvas formats (e.g. `rgb10a2unorm`, `rg11b10ufloat`, `rgba8unorm-srgb`,
   * `rgba16unorm`) automatically activate the W112 offscreen intermediate texture + fullscreen
   * blit fallback.
   *
   * @param {string} declaredFormat
   * @returns {boolean}
   */
  setSwapchainFormat(declaredFormat) {
    if (typeof declaredFormat !== 'string' || declaredFormat.length === 0) {
      return false;
    }
    const swapchain =
      this._swapchain && !this._swapchain.isDestroyed ? this._swapchain : null;
    if (swapchain && typeof swapchain.configureForStreamFormat === 'function') {
      try {
        swapchain.configureForStreamFormat(declaredFormat);
      } catch (err) {
        this._logDiagnostic(
          'warn',
          `[wgpu_worker] failed to configure swapchain for stream format ${declaredFormat}: ${sanitizeGpuErrorDetail(err)}`
        );
        return false;
      }
      this._streamSwapchainFormat = declaredFormat;
      this._swapchainViewFormat = null;
      if (typeof swapchain.format === 'string' && swapchain.format.length > 0) {
        this.format = swapchain.format;
      }
      return true;
    }

    const isSrgb = declaredFormat.endsWith('-srgb');
    const baseFormat = isSrgb ? declaredFormat.slice(0, -5) : declaredFormat;
    if (
      baseFormat !== 'rgba8unorm' &&
      baseFormat !== 'bgra8unorm' &&
      baseFormat !== 'rgba16float'
    ) {
      return false;
    }
    const viewFormat = isSrgb ? declaredFormat : null;
    this._streamSwapchainFormat = declaredFormat;
    if (this.format === baseFormat && (this._swapchainViewFormat || null) === viewFormat) {
      return true;
    }
    this.format = baseFormat;
    this._swapchainViewFormat = viewFormat;
    const viewFormats = viewFormat ? [viewFormat] : null;
    if (swapchain && typeof swapchain.setFormat === 'function') {
      try {
        swapchain.setFormat(baseFormat, viewFormats);
      } catch (err) {
        this._logDiagnostic(
          'warn',
          `[wgpu_worker] failed to reconfigure swapchain format to ${declaredFormat}: ${sanitizeGpuErrorDetail(err)}`
        );
        return false;
      }
      return true;
    }
    if (this.ctx && typeof this.ctx.configure === 'function' && this.device) {
      try {
        const cfg = {
          device: this.device,
          format: baseFormat,
          alphaMode: 'opaque',
        };
        if (viewFormats) {
          cfg.viewFormats = viewFormats;
        }
        this.ctx.configure(cfg);
      } catch (err) {
        this._logDiagnostic(
          'warn',
          `[wgpu_worker] failed to reconfigure canvas context format to ${declaredFormat}: ${sanitizeGpuErrorDetail(err)}`
        );
        return false;
      }
      return true;
    }
    return true;
  }

  /**
   * W121: Configure the swapchain for a stream-declared back-buffer format
   * (alias for `setSwapchainFormat(declaredFormat)`).
   *
   * @param {string} declaredFormat
   * @returns {boolean}
   */
  configureForStreamFormat(declaredFormat) {
    return this.setSwapchainFormat(declaredFormat);
  }

  /**
   * W100: bind a recovered GPUDevice (and re-acquired canvas context) after device loss. Clears
   * every resource handle owned by the dead device, rebuilds owned subsystems, re-configures the
   * swapchain (via `recover()`) or the raw canvas context, and re-hooks `device.lost`. Telemetry
   * counters, frameCount, the ring and shader pack sources are preserved.
   *
   * @param {GPUDevice} newDevice
   * @param {GPUCanvasContext | null | undefined} newCtx
   * @param {{ format?: string, alphaMode?: string, recoverSwapchain?: boolean }} [opts]
   */
  rebindDevice(newDevice, newCtx, opts = {}) {
    if (this._destroyed) {
      throw new Error('WgpuCommandProcessor has been destroyed');
    }
    if (!newDevice || typeof newDevice !== 'object') {
      throw new TypeError('rebindDevice requires a GPUDevice');
    }
    const options = opts && typeof opts === 'object' ? opts : {};
    this._deviceGeneration += 1;
    this._resetDeviceResources();
    // W103: drop in-flight compiles of the dead device; late results are discarded as stale.
    this._resetAsyncPipelineCache(newDevice);
    if (newCtx && typeof newCtx === 'object') {
      this.ctx = newCtx;
    }
    const swapchainRecovers = Boolean(
      this._swapchain &&
        !this._swapchain.isDestroyed &&
        typeof this._swapchain.recover === 'function' &&
        options.recoverSwapchain !== false
    );
    const effectiveFormat = this._streamSwapchainFormat
      ? swapchainRecovers
        ? this._streamSwapchainFormat
        : this.format
      : options.format;
    this.bindDevice(newDevice, {
      format: effectiveFormat,
      recoverSwapchain: options.recoverSwapchain,
    });
    if (
      swapchainRecovers &&
      this._swapchain &&
      typeof this._swapchain.format === 'string' &&
      this._swapchain.format.length > 0
    ) {
      this.format = this._swapchain.format;
    }
    if (!swapchainRecovers && this.ctx && typeof this.ctx.configure === 'function') {
      const cfg = {
        device: newDevice,
        format: this.format,
        alphaMode: typeof options.alphaMode === 'string' ? options.alphaMode : 'opaque',
      };
      if (this._swapchainViewFormat) {
        cfg.viewFormats = [this._swapchainViewFormat];
      }
      this.ctx.configure(cfg);
    }
    this._recreateOwnedSubsystems(newDevice);
    this._hookDeviceLost(newDevice);
    this._deviceLost = false;
    return true;
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
   * W109 / W121: Fetch and load a `.wgsl.pack` URL into `wgslPackCache`, forwarding optional
   * `streamThresholdBytes` (or `streamShaders: true`) into `WgslPackCache.loadPackUrl(url, opts)`.
   * When omitted, preserves full-buffer loading behavior.
   *
   * @param {string} url
   * @param {{
   *   fetch?: typeof fetch,
   *   packName?: string,
   *   streamThresholdBytes?: number,
   *   streamShaders?: boolean,
   *   initialRangeBytes?: number,
   *   maxCachedWgslBytes?: number,
   *   fetchInit?: RequestInit,
   *   lazyDecode?: boolean,
   * }} [opts]
   */
  async loadShaderPackUrl(url, opts = {}) {
    if (this._destroyed) {
      throw new Error('WgpuCommandProcessor has been destroyed');
    }
    if (typeof url !== 'string' || url.trim().length === 0) {
      throw new TypeError('loadShaderPackUrl requires a non-empty URL string');
    }
    const options = opts && typeof opts === 'object' ? opts : {};
    const activeFetch =
      typeof options.fetch === 'function'
        ? options.fetch
        : typeof globalThis !== 'undefined' && typeof globalThis.fetch === 'function'
          ? globalThis.fetch
          : null;
    if (!activeFetch) {
      throw new Error('No fetch implementation available for loadShaderPackUrl');
    }
    const resolvedUrl = url.trim();
    const packName =
      typeof options.packName === 'string' && options.packName.length > 0
        ? options.packName
        : resolvedUrl;
    const streamThresholdBytes = resolveShaderPackStreamThresholdBytes(options, {
      streamThresholdBytes: this._shaderPackStreamThresholdBytes,
    });

    if (
      streamThresholdBytes !== undefined &&
      streamThresholdBytes > 0 &&
      this._ownsWgslPackCache &&
      (!this.wgslPackCache || typeof this.wgslPackCache.loadPackUrl !== 'function')
    ) {
      try {
        const mod = await import('./wgsl_pack_loader.js');
        if (mod && typeof mod.WgslPackCache === 'function') {
          const prevCache = this.wgslPackCache;
          const nextCache = new mod.WgslPackCache({
            fetch: activeFetch,
            streamThresholdBytes,
          });
          this.wgslPackCache = nextCache;
          if (
            this.shaderFallbackRouter &&
            typeof this.shaderFallbackRouter.setPackLoader === 'function' &&
            (this.shaderFallbackRouter._packLoader == null ||
              this.shaderFallbackRouter._packLoader === prevCache)
          ) {
            this.shaderFallbackRouter.setPackLoader(nextCache);
          }
        }
      } catch (_) {
        // Fall through to full-buffer fetch when dynamic ESM import is unavailable.
      }
    }

    if (this.wgslPackCache && typeof this.wgslPackCache.loadPackUrl === 'function') {
      const loadOpts = {
        fetch: activeFetch,
        packName,
      };
      if (streamThresholdBytes !== undefined) {
        loadOpts.streamThresholdBytes = streamThresholdBytes;
      }
      if (
        Number.isInteger(options.initialRangeBytes) &&
        options.initialRangeBytes >= WGPK_HEADER_BYTES
      ) {
        loadOpts.initialRangeBytes = options.initialRangeBytes;
      }
      if (Number.isInteger(options.maxCachedWgslBytes) && options.maxCachedWgslBytes >= 0) {
        loadOpts.maxCachedWgslBytes = options.maxCachedWgslBytes;
      }
      if (options.fetchInit !== undefined) {
        loadOpts.fetchInit = options.fetchInit;
      }
      if (options.lazyDecode !== undefined) {
        loadOpts.lazyDecode = Boolean(options.lazyDecode);
      }
      const result = await this.wgslPackCache.loadPackUrl(resolvedUrl, loadOpts);
      if (
        this.shaderFallbackRouter &&
        this.shaderFallbackRouter._packLoader == null &&
        typeof this.shaderFallbackRouter.setPackLoader === 'function'
      ) {
        this.shaderFallbackRouter.setPackLoader(this.wgslPackCache);
      }
      return result;
    }

    const resp = await activeFetch(resolvedUrl, options.fetchInit);
    if (!resp || !resp.ok || typeof resp.arrayBuffer !== 'function') {
      const status = resp ? resp.status : 'unknown';
      throw new Error(`Failed to fetch WGSL pack from "${resolvedUrl}": HTTP ${status}`);
    }
    const buffer = await resp.arrayBuffer();
    return this.loadWgslPack(buffer, packName);
  }

  /**
   * Alias for `loadShaderPackUrl(url, opts)`.
   */
  loadShaderPack(url, opts = {}) {
    return this.loadShaderPackUrl(url, opts);
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

  _resolveShaderPayload(id, stage, rawCode, trimmed, hashOnlyHex) {
    const stageName = stageCodeToName(stage) || 'vertex';
    let code = rawCode;
    let hashHex = null;
    let entryPoint = null;
    let source = 'inline';
    let requestedHash = null;
    let resolvedHash = null;
    let fallbackProfile = null;
    let lookupHashHex = null;

    if (trimmed.length === 0 || hashOnlyHex !== null) {
      lookupHashHex = hashOnlyHex;
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
    return { code, hashHex, entryPoint, source, requestedHash, resolvedHash, fallbackProfile, lookupHashHex };
  }

  _compileShaderModuleEntry(id, entry, allowDirectThrowOnEmpty) {
    if (!this.device) {
      return null;
    }
    const code = entry.code;
    let module;
    if (allowDirectThrowOnEmpty) {
      module = this.device.createShaderModule({ code });
    } else {
      try {
        module = this.device.createShaderModule({ code });
      } catch (err) {
        this._recordOpValidationError('CreateShaderModule', err);
        return null;
      }
    }
    if (module && typeof module.getCompilationInfo === 'function') {
      try {
        Promise.resolve(module.getCompilationInfo()).then(
          (info) => {
            if (this._destroyed || !info || !Array.isArray(info.messages)) {
              return;
            }
            for (let i = 0; i < info.messages.length; i += 1) {
              const m = info.messages[i];
              if (m && m.type === 'error') {
                const loc =
                  typeof m.lineNum === 'number' && m.lineNum > 0 ? ` (line ${m.lineNum})` : '';
                this._recordOpValidationError(
                  'CreateShaderModule',
                  `shader ${id}${loc}: ${m.message || 'WGSL compilation error'}`
                );
              }
            }
          },
          () => {}
        );
      } catch (_) {
        // Non-fatal: ignore compilation info inspection errors on mock devices.
      }
    }
    entry.module = module;
    this._materializedShaderCount += 1;
    if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
      const byteSize = entry.isInlineCode
        ? Math.max(0, Math.trunc(entry.codeLen || 0))
        : new TextEncoder().encode(code).byteLength;
      this.resourceRegistry.register('shaderModule', id, module, {
        byteSize,
        label: `shaderModule:${id}`,
      });
    }
    return entry;
  }

  /**
   * W106: return the shader module entry for `id`, materializing the underlying GPUShaderModule
   * on first pipeline reference when `lazyShaderCompile` is enabled.
   */
  ensureShaderModule(id) {
    const entry = this.shaderModules.get(id);
    if (!entry) {
      return null;
    }
    if (entry.module) {
      return entry;
    }
    if (entry.pendingRanged) {
      return null;
    }
    return this._compileShaderModuleEntry(id, entry, false);
  }

  /**
   * W136: await settlement of all currently in-flight ranged shader fetches.
   */
  async flushPendingRangedShaders() {
    if (this._inFlightRangedShaders.size === 0) {
      return;
    }
    const pending = Array.from(this._inFlightRangedShaders.values(), (r) => r.promise).filter(Boolean);
    if (pending.length > 0) {
      await Promise.allSettled(pending);
    }
  }

  _invalidatePipelinesReferencingShader(shaderId) {
    for (const [pipeId, pipeMeta] of this._pipelineMeta.entries()) {
      if (pipeMeta && (pipeMeta.vsShaderId === shaderId || pipeMeta.fsShaderId === shaderId)) {
        this._invalidateSpecializedPipelinesFor(pipeId);
      }
    }
  }

  _scheduleRangedShaderResolve(id, stage, lookupHashHex, shaderGen, packCache, lookupKey) {
    const existing = this._inFlightRangedShaders.get(id);
    if (
      existing &&
      existing.hashHex === lookupHashHex &&
      existing.stage === stage &&
      existing.deviceGeneration === this._deviceGeneration &&
      existing.packCache === packCache
    ) {
      existing.gen = shaderGen;
      return existing.promise;
    }

    const deviceGeneration = this._deviceGeneration;
    const record = {
      id,
      stage,
      hashHex: lookupHashHex,
      gen: shaderGen,
      deviceGeneration,
      packCache,
      promise: null,
    };
    this._inFlightRangedShaders.set(id, record);
    this._rangedShaderFetches += 1;

    const isStillCurrent = () =>
      !this._destroyed &&
      !this._deviceLost &&
      !this._deviceRecovering &&
      Boolean(this.device) &&
      this._deviceGeneration === deviceGeneration &&
      this._shaderModuleGenerations.get(id) === record.gen;

    const cleanup = () => {
      if (this._inFlightRangedShaders.get(id) === record) {
        this._inFlightRangedShaders.delete(id);
      }
    };

    let fetchPromise;
    try {
      fetchPromise = Promise.resolve(packCache.getWgslAsync(lookupKey));
    } catch (err) {
      cleanup();
      if (isStillCurrent()) {
        this._rangedShaderErrors += 1;
        this._recordOpValidationError('CreateShaderModule', err);
      }
      return Promise.resolve(null);
    }

    const settledPromise = fetchPromise.then(
      (rawResult) => {
        cleanup();
        if (!isStillCurrent()) {
          return null;
        }
        const hit = unwrapPackLookupResult(rawResult);
        const wgsl = hit && typeof hit.wgsl === 'string' ? hit.wgsl : null;
        if (!wgsl || wgsl.length === 0) {
          this._rangedShaderErrors += 1;
          this._recordOpValidationError(
            'CreateShaderModule',
            `shader ${id}: ranged WGSL fetch for ${lookupHashHex} returned empty payload`
          );
          return null;
        }
        let metaEntryPoint = hit && hit.entryPoint ? hit.entryPoint : null;
        if (!metaEntryPoint && typeof packCache.getShaderMetadata === 'function') {
          try {
            const meta = packCache.getShaderMetadata(lookupKey);
            if (meta && typeof meta.entryPoint === 'string' && meta.entryPoint.length > 0) {
              metaEntryPoint = meta.entryPoint;
            }
          } catch (_) {
            // Optional metadata lookup failure is non-fatal.
          }
        }
        const realEntry = {
          module: null,
          stage,
          code: wgsl,
          codeLen: wgsl.length,
          isInlineCode: false,
          hashHex: lookupHashHex,
          entryPoint: metaEntryPoint || extractWgslEntryPoint(wgsl, stage),
          source: 'pack',
          requestedHash: lookupHashHex,
          resolvedHash: lookupHashHex,
          fallbackProfile: null,
        };
        const compiled = this._compileShaderModuleEntry(id, realEntry, false);
        if (!compiled) {
          this._rangedShaderErrors += 1;
          return null;
        }
        if (!isStillCurrent()) {
          return null;
        }
        this.shaderModules.set(id, compiled);
        this._invalidatePipelinesReferencingShader(id);
        this._rangedShaderReady += 1;
        return compiled;
      },
      (err) => {
        cleanup();
        if (
          !this._destroyed &&
          this._deviceGeneration === deviceGeneration &&
          this._shaderModuleGenerations.get(id) === record.gen
        ) {
          this._rangedShaderErrors += 1;
          this._recordOpValidationError('CreateShaderModule', err);
        }
        return null;
      }
    );

    record.promise = settledPromise;
    return settledPromise;
  }

  _handleCreateShaderModule(view, payloadOffset, payloadBytes) {
    const id = view.getUint32(payloadOffset, true);
    const stage = view.getUint32(payloadOffset + 4, true);
    const codeLen = view.getUint32(payloadOffset + 8, true);
    if (16 + codeLen > payloadBytes) {
      this._recordOpValidationError(
        'CreateShaderModule',
        `shader ${id}: truncated payload (${payloadBytes}B < ${16 + codeLen}B)`
      );
      return;
    }

    const rawBytes = new Uint8Array(view.buffer, view.byteOffset + payloadOffset + 16, codeLen);
    const isShared = typeof SharedArrayBuffer !== 'undefined' && view.buffer instanceof SharedArrayBuffer;
    const rawCode = this._decoder.decode(isShared ? rawBytes.slice() : rawBytes);
    const trimmed = rawCode.trim();
    const hashOnlyHex = parseHashOnlyShaderPayload(trimmed);

    const shaderGen = ++this._nextShaderModuleGen;
    this._shaderModuleGenerations.set(id, shaderGen);

    const resolved = this._resolveShaderPayload(id, stage, rawCode, trimmed, hashOnlyHex);

    // W136: On sync miss for a hash lookup, check whether a ranged pack cache indexes this hash
    // and supports on-demand HTTP 206 Range resolution via `getWgslAsync(hash)`.
    let rangedPackCache = null;
    let rangedLookupKey = null;
    if (resolved.source !== 'pack' && resolved.lookupHashHex !== null) {
      const candidatePackCache =
        this.wgslPackCache ||
        (this.shaderFallbackRouter && this.shaderFallbackRouter._packLoader) ||
        null;
      if (candidatePackCache && typeof candidatePackCache.getWgslAsync === 'function') {
        const alreadyInMemory =
          this.shaderFallbackRouter &&
          this.shaderFallbackRouter._packLoader !== candidatePackCache
            ? lookupFromPackCache(candidatePackCache, resolved.lookupHashHex)
            : null;
        if (!alreadyInMemory) {
          const hasCheck = resolvePackCacheHasKey(candidatePackCache, resolved.lookupHashHex);
          if (hasCheck.matched) {
            rangedPackCache = candidatePackCache;
            rangedLookupKey = hasCheck.key;
          }
        }
      }
    }

    if (!rangedPackCache) {
      this._inFlightRangedShaders.delete(id);
    }

    const entry = {
      module: null,
      stage,
      code: resolved.code,
      codeLen,
      isInlineCode: resolved.code === rawCode,
      hashHex: resolved.hashHex,
      entryPoint: resolved.entryPoint,
      source: resolved.source,
      requestedHash: resolved.requestedHash,
      resolvedHash: resolved.resolvedHash,
      fallbackProfile: resolved.fallbackProfile,
    };

    if (rangedPackCache && this.shaderFallbackRouter === null) {
      entry.pendingRanged = true;
      this.shaderModules.set(id, entry);
      this._scheduleRangedShaderResolve(
        id,
        stage,
        resolved.lookupHashHex,
        shaderGen,
        rangedPackCache,
        rangedLookupKey
      );
      return;
    }

    const shouldDefer =
      this._lazyShaderCompile === true ||
      (this._lazyShaderCompile === 'pack' && entry.source === 'pack');
    if (shouldDefer) {
      this._deferredShaderCount += 1;
      this.shaderModules.set(id, entry);
      if (rangedPackCache) {
        this._scheduleRangedShaderResolve(
          id,
          stage,
          resolved.lookupHashHex,
          shaderGen,
          rangedPackCache,
          rangedLookupKey
        );
      }
      return;
    }

    const allowDirectThrowOnEmpty =
      this.shaderFallbackRouter === null &&
      !this._captureValidationErrors &&
      this._validationErrorCallbacks.size === 0 &&
      trimmed.length === 0;
    const compiled = this._compileShaderModuleEntry(id, entry, allowDirectThrowOnEmpty);
    if (compiled) {
      this.shaderModules.set(id, compiled);
    }
    if (rangedPackCache) {
      this._scheduleRangedShaderResolve(
        id,
        stage,
        resolved.lookupHashHex,
        shaderGen,
        rangedPackCache,
        rangedLookupKey
      );
    }
  }

  _handleCreateRenderPipeline(view, payloadOffset) {
    const id = view.getUint32(payloadOffset, true);
    const vsShaderId = view.getUint32(payloadOffset + 4, true);
    const fsShaderId = view.getUint32(payloadOffset + 8, true);
    const vertexStride = view.getUint32(payloadOffset + 12, true);
    const topologyCode = view.getUint32(payloadOffset + 16, true);
    // Reserved pipeline flags at payloadOffset + 20 for future blend/depth state.

    const vsEntry = this.ensureShaderModule(vsShaderId);
    if (!vsEntry) {
      this._recordOpValidationError(
        'CreateRenderPipeline',
        `pipeline ${id}: unknown vertex shader module ${vsShaderId}`
      );
      return;
    }
    const fsEntry = fsShaderId !== 0 ? this.ensureShaderModule(fsShaderId) : null;
    if (fsShaderId !== 0 && !fsEntry) {
      this._recordOpValidationError(
        'CreateRenderPipeline',
        `pipeline ${id}: unknown fragment shader module ${fsShaderId}`
      );
      return;
    }

    const isFallback = Boolean(
      (vsEntry && vsEntry.source === 'builtin') || (fsEntry && fsEntry.source === 'builtin')
    );
    const usesFallbackShader = Boolean(
      (vsEntry && (vsEntry.source === 'builtin' || vsEntry.source === 'fallback')) ||
        (fsEntry && (fsEntry.source === 'builtin' || fsEntry.source === 'fallback'))
    );
    const isRealPack = Boolean(
      vsEntry && vsEntry.source === 'pack' && (fsShaderId === 0 || (fsEntry && fsEntry.source === 'pack'))
    );
    const expectsGroup0 = Boolean(
      wgslDeclaresGroup0(vsEntry && vsEntry.code) || wgslDeclaresGroup0(fsEntry && fsEntry.code)
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
        targets: [{ format: this._getRenderTargetFormat() }],
      };
    }

    let pipeline;
    try {
      pipeline = this.device.createRenderPipeline(descriptor);
    } catch (err) {
      this._recordOpValidationError('CreateRenderPipeline', err);
      return;
    }
    this.pipelines.set(id, pipeline);
    this._pipelineMeta.set(id, {
      id,
      vsShaderId,
      fsShaderId,
      vertexStride,
      topology,
      isV2: false,
      hasReflectedLayout: Boolean(reflected),
      isRealPack,
      isFallback,
      usesFallbackShader,
      expectsGroup0,
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
    const built = this._buildComputePipelineDescriptorV1(view, payloadOffset);
    if (!built) return;
    const pipeline = this.device.createComputePipeline(built.descriptor);
    this._registerComputePipelineV1(built.id, pipeline);
  }

  /**
   * Decode a v1 CreateComputePipeline payload into a descriptor (reflected explicit layout when
   * available, else 'auto'). Records the reflected bind group layouts for getBindGroupLayout().
   *
   * @returns {{ id: number, csShaderId: number, reflected: boolean, descriptor: GPUComputePipelineDescriptor } | null}
   */
  _buildComputePipelineDescriptorV1(view, payloadOffset) {
    const id = view.getUint32(payloadOffset, true);
    const csShaderId = view.getUint32(payloadOffset + 4, true);
    // Reserved flags at payloadOffset + 8 and reserved at payloadOffset + 12.

    const csEntry = this.ensureShaderModule(csShaderId);
    if (!csEntry) return null;

    const csRefl = this._resolveShaderReflection(csShaderId, csEntry, 'compute');
    const reflected = csRefl
      ? this._buildReflectedPipelineLayout([csRefl], `compute-pipeline-${id}`)
      : null;
    if (reflected) {
      this._computePipelineBindGroupLayouts.set(id, reflected.bindGroupLayouts);
    } else {
      this._computePipelineBindGroupLayouts.delete(id);
    }

    return {
      id,
      csShaderId,
      reflected: Boolean(reflected),
      descriptor: {
        layout: reflected ? reflected.pipelineLayout : 'auto',
        compute: {
          module: csEntry.module,
          entryPoint: csRefl && csRefl.entryPoint ? csRefl.entryPoint : 'cs_main',
        },
      },
    };
  }

  _registerComputePipelineV1(id, pipeline) {
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
    if (16 + entryCount * 32 > payloadBytes) {
      this._recordOpValidationError(
        'CreateBindGroup',
        `bind group ${id}: truncated payload (${payloadBytes}B < ${16 + entryCount * 32}B)`
      );
      return;
    }

    const renderPipeline = this.pipelines.get(pipelineId);
    const computePipeline = this.computePipelines.get(pipelineId);
    const pipeline = renderPipeline || computePipeline;
    const pipeMeta = this._pipelineMeta.get(pipelineId) || null;
    const renderStateExt = this._getRenderStateExtension();

    let layout = null;
    if (!pipeline) {
      const bglRec =
        renderStateExt && renderStateExt.bindGroupLayouts
          ? renderStateExt.bindGroupLayouts.get(pipelineId)
          : null;
      const pendingLayout =
        bglRec && bglRec.layout
          ? null
          : this._pendingAsyncPipelineBindGroupLayout(pipelineId, groupIndex);
      if (bglRec && bglRec.layout) {
        layout = bglRec.layout;
      } else if (pendingLayout) {
        layout = pendingLayout;
      } else {
        this._recordOpValidationError(
          'CreateBindGroup',
          `bind group ${id}: unknown pipeline or bind group layout ${pipelineId}`
        );
        return;
      }
    } else if (pipeMeta && pipeMeta.isFallback) {
      try {
        layout = this.getBindGroupLayout(pipelineId, groupIndex);
      } catch (_) {
        layout = null;
      }
    } else {
      try {
        layout = this.getBindGroupLayout(pipelineId, groupIndex);
      } catch (err) {
        this._recordOpValidationError('CreateBindGroup', err);
        return;
      }
      if (!layout) {
        this._recordOpValidationError(
          'CreateBindGroup',
          `bind group ${id}: pipeline ${pipelineId} has no bind group layout for @group(${groupIndex})`
        );
        return;
      }
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
        if (!buffer) {
          this._recordOpValidationError(
            'CreateBindGroup',
            `bind group ${id}: binding ${binding} unknown buffer ${resourceId}`
          );
          return;
        }
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
        if (!textureView) {
          this._recordOpValidationError(
            'CreateBindGroup',
            `bind group ${id}: binding ${binding} unknown texture ${resourceId}`
          );
          return;
        }
        hasTexture = true;
        entries.push({ binding, resource: textureView });
      } else if (resourceType === BIND_RESOURCE_TYPE.SAMPLER) {
        const sampler = this.samplers.get(resourceId);
        if (!sampler) {
          this._recordOpValidationError(
            'CreateBindGroup',
            `bind group ${id}: binding ${binding} unknown sampler ${resourceId}`
          );
          return;
        }
        hasSampler = true;
        entries.push({ binding, resource: sampler });
      } else {
        this._recordOpValidationError(
          'CreateBindGroup',
          `bind group ${id}: binding ${binding} invalid resourceType ${resourceType}`
        );
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

    let bindGroup;
    try {
      bindGroup = this.device.createBindGroup({
        layout,
        entries,
      });
    } catch (err) {
      this._recordOpValidationError('CreateBindGroup', err);
      return;
    }
    this.bindGroups.set(id, bindGroup);
    this._bindGroupKinds.set(id, computePipeline && !renderPipeline ? 'compute' : 'render');
    if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
      this.resourceRegistry.register('bindGroup', id, bindGroup, {
        byteSize: 0,
        label: `bindGroup:${id}`,
      });
    }
  }

  _dispatchExtensionPacket(opcode, view, payloadOffset, payloadBytes) {
    if (!this.extension || typeof this.extension.handlePacket !== 'function') {
      return;
    }
    const renderStateExt = this._getRenderStateExtension();
    const prevRsErrors =
      renderStateExt && renderStateExt.stats && typeof renderStateExt.stats.errors === 'number'
        ? renderStateExt.stats.errors
        : 0;
    const prevExtErrors =
      this.extension.stats && typeof this.extension.stats.errors === 'number'
        ? this.extension.stats.errors
        : 0;
    const prevCmdLen = this._passCommands.length;

    let handled = false;
    try {
      handled = Boolean(this.extension.handlePacket(opcode, view, payloadOffset, payloadBytes));
    } catch (err) {
      this._recordOpValidationError(opcodeLabel(opcode), err);
      return;
    }

    if (
      renderStateExt &&
      renderStateExt.stats &&
      typeof renderStateExt.stats.errors === 'number' &&
      renderStateExt.stats.errors > prevRsErrors
    ) {
      this._recordOpValidationError(
        opcodeLabel(opcode),
        renderStateExt.lastError || 'malformed or invalid ABI v2 packet'
      );
      return;
    }

    if (
      this.extension !== renderStateExt &&
      this.extension.stats &&
      typeof this.extension.stats.errors === 'number' &&
      this.extension.stats.errors > prevExtErrors
    ) {
      this._recordOpValidationError(
        opcodeLabel(opcode),
        this.extension.lastError || 'malformed or invalid extension packet'
      );
      return;
    }

    if (!handled) {
      return;
    }

    switch (opcode) {
      case OPCODES.CREATE_BIND_GROUP_LAYOUT: {
        if (payloadBytes >= 24) {
          const id = view.getUint32(payloadOffset, true);
          const count = view.getUint32(payloadOffset + 4, true);
          const flags = view.getUint32(payloadOffset + 8, true);
          this._v2BindGroupLayouts.set(id, {
            id,
            entryCount: count,
            isManifestRef: (flags & 0x1) !== 0,
          });
        }
        break;
      }
      case OPCODES.CREATE_PIPELINE_LAYOUT: {
        if (payloadBytes >= 24) {
          const id = view.getUint32(payloadOffset, true);
          const count = view.getUint32(payloadOffset + 4, true);
          const flags = view.getUint32(payloadOffset + 8, true);
          const isManifestRef = (flags & 0x1) !== 0;
          let groupLayoutIds = null;
          if (!isManifestRef && 24 + count * 4 <= payloadBytes) {
            groupLayoutIds = [];
            for (let g = 0; g < count; g += 1) {
              groupLayoutIds.push(view.getUint32(payloadOffset + 24 + g * 4, true));
            }
          }
          this._v2PipelineLayouts.set(id, { id, groupLayoutIds, isManifestRef });
        }
        break;
      }
      case OPCODES.CREATE_RENDER_PIPELINE2: {
        if (payloadBytes >= 16 + 376) {
          const id = view.getUint32(payloadOffset, true);
          const vsShaderId = view.getUint32(payloadOffset + 4, true);
          const fsShaderId = view.getUint32(payloadOffset + 8, true);
          const layoutId = view.getUint32(payloadOffset + 12, true);
          const vsEntry = this.shaderModules.get(vsShaderId) || null;
          const fsEntry = fsShaderId !== 0 ? this.shaderModules.get(fsShaderId) || null : null;

          const isFallback = Boolean(
            (vsEntry && vsEntry.source === 'builtin') || (fsEntry && fsEntry.source === 'builtin')
          );
          const usesFallbackShader = Boolean(
            (vsEntry && (vsEntry.source === 'builtin' || vsEntry.source === 'fallback')) ||
              (fsEntry && (fsEntry.source === 'builtin' || fsEntry.source === 'fallback'))
          );
          const isRealPack = Boolean(
            vsEntry &&
              vsEntry.source === 'pack' &&
              (fsShaderId === 0 || (fsEntry && fsEntry.source === 'pack'))
          );
          const effectiveFallbackProfile =
            vsEntry && vsEntry.fallbackProfile && vsEntry.fallbackProfile !== FALLBACK_PROFILE.DEFAULT
              ? vsEntry.fallbackProfile
              : fsEntry && fsEntry.fallbackProfile && fsEntry.fallbackProfile !== FALLBACK_PROFILE.DEFAULT
                ? fsEntry.fallbackProfile
                : isFallback
                  ? FALLBACK_PROFILE.DEFAULT
                  : null;

          let expectsGroup0 = Boolean(
            wgslDeclaresGroup0(vsEntry && vsEntry.code) || wgslDeclaresGroup0(fsEntry && fsEntry.code)
          );
          if (layoutId !== 0) {
            const plMeta = this._v2PipelineLayouts.get(layoutId) || null;
            if (plMeta && Array.isArray(plMeta.groupLayoutIds)) {
              const resolvedBgls = plMeta.groupLayoutIds.map((bglId) => {
                if (bglId === 0 || !renderStateExt) return null;
                const rec = renderStateExt.bindGroupLayouts.get(bglId);
                return rec ? rec.layout : null;
              });
              this._pipelineBindGroupLayouts.set(id, resolvedBgls);
              const group0BglId = plMeta.groupLayoutIds.length > 0 ? plMeta.groupLayoutIds[0] : 0;
              if (group0BglId !== 0) {
                const bglMeta = this._v2BindGroupLayouts.get(group0BglId) || null;
                const extBgl = renderStateExt
                  ? renderStateExt.bindGroupLayouts.get(group0BglId)
                  : null;
                if (
                  (bglMeta && (bglMeta.entryCount > 0 || bglMeta.isManifestRef)) ||
                  (extBgl && Array.isArray(extBgl.entries) && extBgl.entries.length > 0)
                ) {
                  expectsGroup0 = true;
                }
              }
            }
          } else {
            this._pipelineBindGroupLayouts.delete(id);
          }

          const keyBase = payloadOffset + 16;
          const bufferCount = Math.min(view.getUint32(keyBase + 172, true), 8);
          const vertexStride = bufferCount > 0 ? view.getUint32(keyBase + 180, true) : 0;

          this._pipelineMeta.set(id, {
            id,
            vsShaderId,
            fsShaderId,
            layoutId,
            vertexStride,
            topology: 'triangle-list',
            isV2: true,
            hasReflectedLayout: layoutId !== 0,
            isRealPack,
            isFallback,
            usesFallbackShader,
            expectsGroup0,
            fallbackProfile: effectiveFallbackProfile,
          });
          this._invalidateSpecializedPipelinesFor(id);
        }
        break;
      }
      case OPCODES.CREATE_BIND_GROUP2: {
        if (payloadBytes >= 16) {
          const id = view.getUint32(payloadOffset, true);
          const layoutId = view.getUint32(payloadOffset + 4, true);
          const count = view.getUint32(payloadOffset + 8, true);
          let hasCbv = false;
          let hasTexture = false;
          let hasSampler = false;
          if (16 + count * 48 <= payloadBytes) {
            for (let i = 0; i < count; i += 1) {
              const entryType = view.getUint32(payloadOffset + 16 + i * 48 + 4, true);
              if (entryType === BIND_RESOURCE_TYPE.BUFFER) hasCbv = true;
              else if (entryType === BIND_RESOURCE_TYPE.TEXTURE_VIEW) hasTexture = true;
              else if (entryType === BIND_RESOURCE_TYPE.SAMPLER) hasSampler = true;
            }
          }
          this._bindGroupMeta.set(id, {
            id,
            pipelineId: 0,
            layoutId,
            isV2: true,
            groupIndex: 0,
            hasCbv,
            hasTexture,
            hasSampler,
            entries: null,
          });
          this._invalidateSpecializedBindGroupsFor(id);
        }
        break;
      }
      case OPCODES.SET_BIND_GROUP2: {
        if (payloadBytes >= 16 && this._passCommands.length > prevCmdLen) {
          this._passCommands[this._passCommands.length - 1]._wgpuMeta = {
            kind: 'setBindGroup2',
            groupIndex: view.getUint32(payloadOffset, true),
            bindGroupId: view.getUint32(payloadOffset + 4, true),
          };
        }
        break;
      }
      case OPCODES.BEGIN_RENDER_PASS: {
        if (this._passCommands.length > prevCmdLen) {
          this._passCommands[this._passCommands.length - 1]._wgpuMeta = {
            kind: 'beginRenderPass',
          };
        }
        break;
      }
      case OPCODES.END_RENDER_PASS: {
        if (this._passCommands.length > prevCmdLen) {
          this._passCommands[this._passCommands.length - 1]._wgpuMeta = {
            kind: 'endRenderPass',
          };
        }
        break;
      }
      default:
        break;
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
    this._v2BindGroupLayouts.delete(id);
    this._v2PipelineLayouts.delete(id);
    this._invalidateSpecializedBindGroupsFor(id);
    this.shaderModules.delete(id);
    this._shaderModuleGenerations.delete(id);
    this._inFlightRangedShaders.delete(id);
    this.pipelines.delete(id);
    this._pipelineMeta.delete(id);
    this._invalidateSpecializedPipelinesFor(id);
    this.computePipelines.delete(id);
    this._pipelineBindGroupLayouts.delete(id);
    this._computePipelineBindGroupLayouts.delete(id);
    this._shaderReflectionKeys.delete(id);
    if (this.asyncPipelineCache && typeof this.asyncPipelineCache.delete === 'function') {
      this.asyncPipelineCache.delete(id);
    }
    if (this.extension) this.extension.destroyResource(id);
  }

  /** W103: reflected bind group layout of a pipeline that is still compiling asynchronously. */
  _pendingAsyncPipelineBindGroupLayout(pipelineId, groupIndex) {
    if (!this.isAsyncPipelinePending(pipelineId)) {
      return null;
    }
    const reflected =
      this._pipelineBindGroupLayouts.get(pipelineId) ||
      this._computePipelineBindGroupLayouts.get(pipelineId);
    return reflected && reflected[groupIndex] ? reflected[groupIndex] : null;
  }

  _flushPresent() {
    const pendingEncoder = this._pendingEncoder;
    this._pendingEncoder = null;
    this._hasPendingUploadCommands = false;

    // W99: a torn-down WebGPU instance throws `OperationError` from createCommandEncoder /
    // getCurrentTexture; record it and drop the frame instead of crashing the message loop.
    // W112 / W121: when WgpuSwapchain activates offscreen blit fallback for a non-canvas
    // stream format, bind the intermediate render-target view for SWAPCHAIN_TEXTURE_ID passes
    // and keep the canvas view handy to blit right before encoder.finish() / queue.submit().
    const activeSwapchain =
      this._swapchain && !this._swapchain.isDestroyed ? this._swapchain : null;
    const useSwapchainBlitFallback = Boolean(
      activeSwapchain &&
        activeSwapchain.usesBlitFallback === true &&
        activeSwapchain.isConfigured !== false &&
        typeof activeSwapchain.getRenderTargetView === 'function'
    );

    let encoder;
    let swapchainTexture;
    let swapchainView;
    let canvasTexture = null;
    let canvasView = null;
    try {
      encoder = pendingEncoder || this.device.createCommandEncoder();
      canvasTexture = this.ctx.getCurrentTexture();
      canvasView =
        canvasTexture && typeof canvasTexture.createView === 'function'
          ? this._swapchainViewFormat && !useSwapchainBlitFallback
            ? canvasTexture.createView({ format: this._swapchainViewFormat })
            : canvasTexture.createView()
          : canvasTexture;
      if (useSwapchainBlitFallback) {
        swapchainView = activeSwapchain.getRenderTargetView();
        swapchainTexture = activeSwapchain.intermediateTexture || canvasTexture;
      } else {
        swapchainTexture = canvasTexture;
        swapchainView = canvasView;
      }
    } catch (err) {
      this._notePresentSubmitFailure(err);
      this._endSubsystemsForFrame(null);
      this._resetFrameState();
      return;
    }
    let activePass = null;
    let activePassKind = null;
    let renderPassOpened = false;
    let renderPassIdx = 0;
    let computePassIdx = 0;

    const timerSupported = Boolean(this.passTimer && this.passTimer.supported);

    let activePipelineId = null;
    let specializedActiveKey = null;
    const boundBindGroups = new Map();
    const boundVertexBuffers = new Map();

    const clearPassBindings = () => {
      activePipelineId = null;
      specializedActiveKey = null;
      boundBindGroups.clear();
      boundVertexBuffers.clear();
    };

    const endActivePass = () => {
      if (activePass) {
        try {
          activePass.end();
        } catch (err) {
          this._recordOpValidationError('EndRenderPass', err);
        }
        activePass = null;
        activePassKind = null;
        clearPassBindings();
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
      swapchainTexture,
      swapchainView,
      canvasWidth: this._viewportWidth > 0 ? this._viewportWidth : undefined,
      canvasHeight: this._viewportHeight > 0 ? this._viewportHeight : undefined,
      swapchainWidth:
        activeSwapchain && activeSwapchain.width > 0 ? activeSwapchain.width : undefined,
      swapchainHeight:
        activeSwapchain && activeSwapchain.height > 0 ? activeSwapchain.height : undefined,
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
              boundBg.bindGroup = specBg;
            }
          }
        }
        specializedActiveKey = key;
      }
    };

    const executeRenderDraw = (drawOpName, count, instanceCount, invokeDraw) => {
      const pass = ensureRenderPass();
      maybeSpecializeFallbackPipeline(pass, count, instanceCount);
      const pipeMeta =
        activePipelineId !== null ? this._pipelineMeta.get(activePipelineId) || null : null;
      const activePipeline =
        activePipelineId !== null ? this.pipelines.get(activePipelineId) || null : null;
      if (activePipelineId !== null && !activePipeline && specializedActiveKey === null) {
        if (this.isAsyncPipelinePending(activePipelineId)) {
          // W103: still compiling; draw normally on a later frame once it is ready.
          this._pendingAsyncPipelineSkips += 1;
          return;
        }
        this._skippedDrawCount += 1;
        this._recordOpValidationError(
          drawOpName,
          `skipped draw: render pipeline ${activePipelineId} is unavailable`
        );
        return;
      }
      const group0Entry = boundBindGroups.get(0);
      const hasLiveGroup0 = Boolean(group0Entry && group0Entry.bindGroup);
      if (pipeMeta && pipeMeta.expectsGroup0 && !hasLiveGroup0) {
        this._unboundBindGroupCount += 1;
        this._skippedDrawCount += 1;
        this._recordOpValidationError(
          drawOpName,
          `skipped draw on pipeline ${activePipelineId}: expected @group(0) bind group is not bound`
        );
        return;
      }
      try {
        invokeDraw(pass);
      } catch (err) {
        this._skippedDrawCount += 1;
        if (/\b(?:group|bind\s*group)\b/i.test(String(err))) {
          this._unboundBindGroupCount += 1;
        }
        this._recordOpValidationError(drawOpName, err);
        return;
      }
      this._drawCallCount += 1;
      if (pipeMeta && pipeMeta.isRealPack && specializedActiveKey === null) {
        this._realPackDrawCount += 1;
      } else if (
        (pipeMeta && (pipeMeta.isFallback || pipeMeta.usesFallbackShader)) ||
        specializedActiveKey !== null
      ) {
        this._fallbackDrawCount += 1;
      }
    };

    for (const cmd of this._passCommands) {
      switch (cmd.type) {
        case 'setPipeline': {
          const pass = ensureRenderPass();
          activePipelineId = cmd.pipelineId;
          specializedActiveKey = null;
          const pipeline = this.pipelines.get(cmd.pipelineId) || cmd.pipeline || null;
          if (pipeline) {
            try {
              pass.setPipeline(pipeline);
            } catch (err) {
              this._recordOpValidationError('SetPipeline', err);
            }
          } else if (!this.isAsyncPipelinePending(cmd.pipelineId)) {
            this._recordOpValidationError(
              'SetPipeline',
              `render pipeline ${cmd.pipelineId} not found`
            );
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
            try {
              pass.setVertexBuffer(cmd.slot, buffer, cmd.offset);
            } catch (err) {
              this._recordOpValidationError('SetVertexBuffer', err);
            }
          }
          break;
        }
        case 'setIndexBuffer': {
          const pass = ensureRenderPass();
          const buffer = this.buffers.get(cmd.bufferId);
          if (buffer) {
            try {
              pass.setIndexBuffer(buffer, cmd.indexFormat, cmd.offset);
            } catch (err) {
              this._recordOpValidationError('SetIndexBuffer', err);
            }
          }
          break;
        }
        case 'setBindGroup': {
          const bindGroup = this.bindGroups.get(cmd.bindGroupId) || cmd.bindGroup || null;
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
          } else {
            this._recordOpValidationError(
              'SetBindGroup',
              `bind group ${cmd.bindGroupId} not found for @group(${cmd.groupIndex})`
            );
          }
          if (bindGroup) {
            const pass =
              activePass ||
              (this._bindGroupKinds.get(cmd.bindGroupId) === 'compute'
                ? ensureComputePass()
                : ensureRenderPass());
            try {
              pass.setBindGroup(cmd.groupIndex, bindGroup);
            } catch (err) {
              this._recordOpValidationError('SetBindGroup', err);
            }
          }
          break;
        }
        case 'draw': {
          executeRenderDraw('Draw', cmd.vertexCount, cmd.instanceCount, (pass) => {
            pass.draw(cmd.vertexCount, cmd.instanceCount, cmd.firstVertex, cmd.firstInstance);
          });
          break;
        }
        case 'drawIndexed': {
          executeRenderDraw('DrawIndexed', cmd.indexCount, cmd.instanceCount, (pass) => {
            pass.drawIndexed(
              cmd.indexCount,
              cmd.instanceCount,
              cmd.firstIndex,
              cmd.baseVertex,
              cmd.firstInstance
            );
          });
          break;
        }
        case 'setComputePipeline': {
          const pass = ensureComputePass();
          const pipeline = this.computePipelines.get(cmd.pipelineId);
          if (pipeline) {
            try {
              pass.setPipeline(pipeline);
            } catch (err) {
              this._recordOpValidationError('SetComputePipeline', err);
            }
          }
          break;
        }
        case 'dispatchCompute': {
          const pass = ensureComputePass();
          try {
            pass.dispatchWorkgroups(cmd.workgroupX, cmd.workgroupY, cmd.workgroupZ);
            this._computeDispatchCount += 1;
          } catch (err) {
            this._recordOpValidationError('DispatchCompute', err);
          }
          break;
        }
        default:
          if (typeof cmd.run === 'function') {
            const meta = cmd._wgpuMeta || null;
            try {
              cmd.run(passControl);
              if (meta && meta.kind === 'setBindGroup2' && activePass) {
                const bg = this.bindGroups.get(meta.bindGroupId) || null;
                const bgMeta = this._bindGroupMeta.get(meta.bindGroupId) || null;
                boundBindGroups.set(meta.groupIndex, {
                  bindGroupId: meta.bindGroupId,
                  bindGroup: bg || { bound: true },
                  meta: bgMeta,
                  hasCbv: Boolean(bgMeta && bgMeta.hasCbv),
                  hasTexture: Boolean(bgMeta && bgMeta.hasTexture),
                  hasSampler: Boolean(bgMeta && bgMeta.hasSampler),
                  entries: bgMeta ? bgMeta.entries : null,
                  desc: bg && bg.desc ? bg.desc : null,
                });
              }
            } catch (err) {
              this._recordOpValidationError(
                meta && meta.kind ? meta.kind : 'RenderStatePassCommand',
                err
              );
            }
          }
          break;
      }
    }

    endActivePass();
    if (!renderPassOpened) {
      try {
        ensureRenderPass();
      } catch (err) {
        this._recordOpValidationError('BeginRenderPass', sanitizeGpuErrorDetail(err));
      }
      endActivePass();
    }

    // W112 / W121: Blit the offscreen intermediate render-target texture onto the canvas
    // right before finishing and submitting when blit fallback is active (zero overhead
    // when usesBlitFallback === false).
    if (
      useSwapchainBlitFallback &&
      typeof activeSwapchain.blitToCanvasIfNeeded === 'function'
    ) {
      try {
        activeSwapchain.blitToCanvasIfNeeded(encoder, canvasView);
      } catch (err) {
        this._recordOpValidationError('SwapchainBlit', sanitizeGpuErrorDetail(err));
      }
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

    // W99: guard encoder.finish() and queue.submit() separately so a torn-down WebGPU instance
    // (`OperationError`) is recorded as a sanitized Present event without crashing the worker.
    let submitted = false;
    let commandBuffer = null;
    try {
      commandBuffer = encoder.finish();
    } catch (err) {
      this._notePresentSubmitFailure(err);
    }
    if (commandBuffer !== null) {
      try {
        this.device.queue.submit([commandBuffer]);
        submitted = true;
      } catch (err) {
        this._notePresentSubmitFailure(err);
      }
    }

    if (submitted) {
      this.frameCount += 1;
      this._presentCount += 1;
      if (this.frameCount === 1) {
        this._notifyFirstPresent({
          frameIndex: 0,
          framesPresented: 1,
          source: 'command-buffer',
          timestampMs: Date.now(),
        });
      }
    }

    const workDoneFence = submitted ? this._getWorkDoneFence() : null;
    this._endSubsystemsForFrame(workDoneFence);

    if (
      submitted &&
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
    } else if (this._deviceRecovering) {
      view = null;
    } else {
      throw new TypeError(
        'WgpuCommandProcessor.executeCommandBuffer expects an ArrayBuffer, SharedArrayBuffer, or ArrayBufferView'
      );
    }

    if (this._deviceRecovering) {
      // W100: drop submissions for the lost device while recovery is in flight.
      this._droppedWhileRecoveringCount += 1;
      // W104: resources the guest creates meanwhile are still recorded for replay.
      if (view) {
        this._recordReplayPacketsFrom(view, 0);
      }
      return { frameCount: this.frameCount, presents: 0 };
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
      if (payloadOffset + payloadBytes > view.byteLength) {
        this._recordOpValidationError(
          'CommandStream',
          `truncated packet ${opcodeLabel(opcode)} (need ${payloadBytes}B, remaining ${view.byteLength - payloadOffset}B)`
        );
        break;
      }

      this._packetCount += 1;
      this._bytesProcessed += 8 + payloadBytes;

      this._recordReplayPacket(opcode, view, payloadOffset, payloadBytes);
      presents += this._dispatchPacket(opcode, view, payloadOffset, payloadBytes);

      offset = payloadOffset + payloadBytes;
      if (this._deviceRecovering) {
        // W100: device loss detected mid-buffer; skip the remaining packets cleanly.
        // W104: still record the skipped resource packets so recovery can replay them.
        this._recordReplayPacketsFrom(view, offset);
        break;
      }
    }
    } finally {
      this._pendingCommandStreamShaderProfiles = prevPendingHints;
    }

    return { frameCount: this.frameCount, presents };
  }

  /**
   * Dispatch one packet of a command stream. Returns the number of presents (0 or 1).
   */
  _dispatchPacket(opcode, view, payloadOffset, payloadBytes) {
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
        } else {
          this._recordOpValidationError('ClearColor', `truncated payload (${payloadBytes}B < 16B)`);
        }
        break;
      case OPCODES.CREATE_BUFFER:
        if (payloadBytes >= 16) {
          this._handleCreateBuffer(view, payloadOffset);
        } else {
          this._recordOpValidationError('CreateBuffer', `truncated payload (${payloadBytes}B < 16B)`);
        }
        break;
      case OPCODES.WRITE_BUFFER:
        if (payloadBytes >= 16) {
          this._handleWriteBuffer(view, payloadOffset, payloadBytes);
        } else {
          this._recordOpValidationError('WriteBuffer', `truncated payload (${payloadBytes}B < 16B)`);
        }
        break;
      case OPCODES.CREATE_SHADER_MODULE:
        if (payloadBytes >= 16) {
          this._handleCreateShaderModule(view, payloadOffset, payloadBytes);
        } else {
          this._recordOpValidationError(
            'CreateShaderModule',
            `truncated payload (${payloadBytes}B < 16B)`
          );
        }
        break;
      case OPCODES.CREATE_RENDER_PIPELINE:
        if (payloadBytes >= 24) {
          this._handleCreateRenderPipeline(view, payloadOffset);
        } else {
          this._recordOpValidationError(
            'CreateRenderPipeline',
            `truncated payload (${payloadBytes}B < 24B)`
          );
        }
        break;
      case OPCODES.SET_PIPELINE:
        if (payloadBytes >= 4) {
          const pipelineId = view.getUint32(payloadOffset, true);
          this._passCommands.push({
            type: 'setPipeline',
            pipelineId,
            pipeline: this.pipelines.get(pipelineId) || null,
          });
        } else {
          this._recordOpValidationError('SetPipeline', `truncated payload (${payloadBytes}B < 4B)`);
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
        } else {
          this._recordOpValidationError('SetVertexBuffer', `truncated payload (${payloadBytes}B < 16B)`);
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
        } else {
          this._recordOpValidationError('Draw', `truncated payload (${payloadBytes}B < 16B)`);
        }
        break;
      case OPCODES.PRESENT:
        this._flushPresent();
        return 1;
      case OPCODES.DESTROY_RESOURCE:
        if (payloadBytes >= 4) {
          this._handleDestroyResource(view.getUint32(payloadOffset, true));
        }
        break;
      case OPCODES.CREATE_TEXTURE:
        if (payloadBytes >= 32) {
          this._handleCreateTexture(view, payloadOffset);
        } else {
          this._recordOpValidationError('CreateTexture', `truncated payload (${payloadBytes}B < 32B)`);
        }
        break;
      case OPCODES.WRITE_TEXTURE:
        if (payloadBytes >= 48) {
          this._handleWriteTexture(view, payloadOffset, payloadBytes);
        } else {
          this._recordOpValidationError('WriteTexture', `truncated payload (${payloadBytes}B < 48B)`);
        }
        break;
      case OPCODES.CREATE_SAMPLER:
        if (payloadBytes >= 32) {
          this._handleCreateSampler(view, payloadOffset);
        } else {
          this._recordOpValidationError('CreateSampler', `truncated payload (${payloadBytes}B < 32B)`);
        }
        break;
      case OPCODES.CREATE_BIND_GROUP:
        if (payloadBytes >= 16) {
          this._handleCreateBindGroup(view, payloadOffset, payloadBytes);
        } else {
          this._recordOpValidationError(
            'CreateBindGroup',
            `truncated payload (${payloadBytes}B < 16B)`
          );
        }
        break;
      case OPCODES.SET_BIND_GROUP:
        if (payloadBytes >= 8) {
          const groupIndex = view.getUint32(payloadOffset, true);
          const bindGroupId = view.getUint32(payloadOffset + 4, true);
          this._passCommands.push({
            type: 'setBindGroup',
            groupIndex,
            bindGroupId,
            bindGroup: this.bindGroups.get(bindGroupId) || null,
          });
        } else {
          this._recordOpValidationError('SetBindGroup', `truncated payload (${payloadBytes}B < 8B)`);
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
        } else {
          this._recordOpValidationError('SetIndexBuffer', `truncated payload (${payloadBytes}B < 16B)`);
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
        } else {
          this._recordOpValidationError('DrawIndexed', `truncated payload (${payloadBytes}B < 20B)`);
        }
        break;
      case OPCODES.CREATE_COMPUTE_PIPELINE:
        if (payloadBytes >= 16) {
          this._handleCreateComputePipeline(view, payloadOffset);
        } else {
          this._recordOpValidationError(
            'CreateComputePipeline',
            `truncated payload (${payloadBytes}B < 16B)`
          );
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
        if (this.extension) {
          this._dispatchExtensionPacket(opcode, view, payloadOffset, payloadBytes);
        }
        break;
    }
    return 0;
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
    let swapchain = null;
    if (this._swapchain && typeof this._swapchain.getStats === 'function') {
      try {
        swapchain = this._swapchain.getStats();
      } catch (_) {
        swapchain = null;
      }
    }
    const texturePacketsExt = this._getTexturePacketsExtension();
    let texturePackets = null;
    if (texturePacketsExt && typeof texturePacketsExt.getStats === 'function') {
      try {
        texturePackets = texturePacketsExt.getStats();
      } catch (_) {
        texturePackets = null;
      }
    }
    const swapchainFormat = sanitizeSwapchainFormatToken(
      (swapchain && (swapchain.streamFormat || swapchain.format)) ||
        this._streamSwapchainFormat ||
        this.format
    );
    const usesBlitFallback = Boolean(
      (swapchain && swapchain.usesBlitFallback === true) ||
        (this._swapchain && this._swapchain.usesBlitFallback === true)
    );
    const blitCount = toNonNegativeInt(
      swapchain && swapchain.blitCount !== undefined
        ? swapchain.blitCount
        : this._swapchain && this._swapchain.blitCount,
      0
    );
    const bc3dFallbacks = toNonNegativeInt(
      texturePackets && texturePackets.bc3dFallbacks,
      0
    );
    const bc3dFallbackUploadsSkipped = toNonNegativeInt(
      texturePackets && texturePackets.bc3dFallbackUploadsSkipped,
      0
    );

    return {
      frameCount: this.frameCount,
      framesPresented: this.frameCount,
      presentCount: this._presentCount,
      splashFramesPresented: this._splashFramesPresented,
      firstPresent: this._firstPresentNotified || this.frameCount > 0,
      packetCount: this._packetCount,
      drawCalls: this._drawCallCount,
      realPackDrawCount: this._realPackDrawCount,
      fallbackDrawCount: this._fallbackDrawCount,
      unboundBindGroupCount: this._unboundBindGroupCount,
      skippedDrawCount: this._skippedDrawCount,
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
      validationErrorSuppressedCount: this._validationErrorSuppressedCount,
      wgslPackCache,
      rangedShaderFetches: this._rangedShaderFetches,
      rangedShaderReady: this._rangedShaderReady,
      rangedShaderErrors: this._rangedShaderErrors,
      splashBridge,
      binkPresenter,
      swapchain,
      swapchainFormat,
      usesBlitFallback,
      blitCount,
      texturePackets,
      bc3dFallbacks,
      bc3dFallbackUploadsSkipped,
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
    // W103: settle queued/in-flight pipeline compiles so nothing writes into a torn-down processor.
    this._resetAsyncPipelineCache(null);
    // W100: intentional teardown — never treat the subsequent `device.lost` as a loss.
    this._deviceLossListener = null;
    this._deviceRecovering = false;
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
    // W104: drop the replay log and shadow buffer cache.
    this._replayLog.clear();
    this._replayBindGroupKeys.clear();
    this._replayLogBytes = 0;
    this._shadowBuffers.clear();
    this._shadowBufferBytes = 0;

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
    this._v2BindGroupLayouts.clear();
    this._v2PipelineLayouts.clear();
    this._validationErrorDedup.clear();
    this.shaderModules.clear();
    this._shaderModuleGenerations.clear();
    this._inFlightRangedShaders.clear();
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

const MAX_GPU_ERROR_DETAIL_CHARS = 256;

/**
 * Reduce a thrown/rejected WebGPU error to a single-line `Name: message` string with URLs
 * stripped and length capped, suitable for validation telemetry.
 *
 * @param {unknown} err
 * @returns {string}
 */
function sanitizeGpuErrorDetail(err) {
  let name = '';
  let message = '';
  try {
    if (err && typeof err === 'object') {
      name = typeof err.name === 'string' ? err.name : '';
      message = typeof err.message === 'string' ? err.message : '';
    } else if (err !== undefined && err !== null) {
      message = String(err);
    }
  } catch (_) {
    message = '';
  }
  message = message
    .replace(/\b(?:https?|file|blob):\/\/\S+/gi, '<url>')
    .replace(/\s+/g, ' ')
    .trim();
  if (message.length === 0) {
    message = 'operation failed';
  }
  const detail = name && name !== 'Error' ? `${name}: ${message}` : message;
  return detail.length > MAX_GPU_ERROR_DETAIL_CHARS
    ? `${detail.slice(0, MAX_GPU_ERROR_DETAIL_CHARS - 3)}...`
    : detail;
}

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
 * Build a sanitized `{ type: 'gpu-telemetry', frameStats, fallbackStats, missingShaders, timestamp, renderStateStats? }`
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
  const realPackDrawCount = toNonNegativeInt(
    stats && stats.realPackDrawCount !== undefined
      ? stats.realPackDrawCount
      : processor && processor.realPackDrawCount,
    0
  );
  const fallbackDrawCount = toNonNegativeInt(
    stats && stats.fallbackDrawCount !== undefined
      ? stats.fallbackDrawCount
      : processor && processor.fallbackDrawCount,
    0
  );
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

  const renderStateTelemetry = resolveRenderStateTelemetry(processor, stats);
  const renderStateStats = {
    computeDispatchCount: toNonNegativeInt(
      renderStateTelemetry && renderStateTelemetry.computeDispatchCount,
      0
    ),
    computePipelineCount: toNonNegativeInt(
      renderStateTelemetry && renderStateTelemetry.computePipelineCount,
      0
    ),
    autoLayoutFallbacks: toNonNegativeInt(
      renderStateTelemetry && renderStateTelemetry.autoLayoutFallbacks,
      0
    ),
    bindGroupKindMismatches: toNonNegativeInt(
      renderStateTelemetry && renderStateTelemetry.bindGroupKindMismatches,
      0
    ),
  };

  let timestamp;
  if (typeof opts.timestamp === 'number' && Number.isFinite(opts.timestamp)) {
    timestamp = opts.timestamp;
  } else if (typeof opts.now === 'function') {
    const t = opts.now();
    timestamp = typeof t === 'number' && Number.isFinite(t) ? t : Date.now();
  } else {
    timestamp = Date.now();
  }

  const snapshot = {
    type: 'gpu-telemetry',
    frameStats: {
      framesPresented,
      droppedFrames,
      drawCalls,
      realPackDrawCount,
      fallbackDrawCount,
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
  // W99: live render_state_packets.js compute/auto-layout counters (0 when not installed).
  // Omitted for a null processor so the empty snapshot shape stays unchanged.
  if (processor && typeof processor === 'object') {
    snapshot.renderStateStats = renderStateStats;
    // W100: WebGPU device-loss / recovery counters (zeros when the processor lacks them).
    snapshot.deviceStats = resolveDeviceLossTelemetry(processor, stats);
    // W103: async pipeline compile queue counters (null when async compilation is disabled).
    snapshot.asyncPipelineStats = resolveAsyncPipelineTelemetry(processor);
    // W121: swapchain format/blit (W112), 3D-BC texture fallback (W108), and WGSL pack mode (W109).
    snapshot.swapchainStats = resolveSwapchainTelemetry(processor, stats);
    snapshot.textureStats = resolveTextureTelemetry(processor, stats);
    snapshot.shaderPackStats = resolveShaderPackTelemetry(processor, stats);
  }
  return snapshot;
}

/**
 * W121: sanitized swapchain format and W112 offscreen blit-fallback counters.
 *
 * @returns {{ swapchainFormat: string | null, canvasFormat: string | null, usesBlitFallback: boolean, blitCount: number }}
 */
function resolveSwapchainTelemetry(processor, stats) {
  let raw = null;
  try {
    const swap =
      (processor && processor.swapchain) ||
      (processor && processor._swapchain) ||
      null;
    if (swap && typeof swap.getStats === 'function') {
      raw = swap.getStats();
    } else if (swap && typeof swap === 'object') {
      raw = swap;
    }
  } catch (_) {
    raw = null;
  }
  if (!raw || typeof raw !== 'object') {
    if (stats && stats.swapchainStats && typeof stats.swapchainStats === 'object') {
      raw = stats.swapchainStats;
    } else if (stats && stats.swapchain && typeof stats.swapchain === 'object') {
      raw = stats.swapchain;
    }
  }
  const src = raw && typeof raw === 'object' ? raw : {};
  const swapchainFormat = sanitizeSwapchainFormatToken(
    src.swapchainFormat ||
      src.streamFormat ||
      src.format ||
      (stats && stats.swapchainFormat) ||
      (processor && processor._streamSwapchainFormat) ||
      (processor && processor.format) ||
      null
  );
  const canvasFormat = sanitizeSwapchainFormatToken(
    src.canvasFormat ||
      src.format ||
      (processor && processor.format) ||
      swapchainFormat ||
      null
  );
  const usesBlitFallback = Boolean(
    src.usesBlitFallback === true ||
      (stats && stats.usesBlitFallback === true)
  );
  const blitCount = toNonNegativeInt(
    src.blitCount !== undefined
      ? src.blitCount
      : stats && stats.blitCount,
    0
  );
  return {
    swapchainFormat,
    canvasFormat,
    usesBlitFallback,
    blitCount,
  };
}

/**
 * W121: sanitized W108 3D-BC texture fallback counters from `TexturePackets.getStats()`.
 *
 * @returns {{ bc3dFallbacks: number, bc3dFallbackUploadsSkipped: number }}
 */
function resolveTextureTelemetry(processor, stats) {
  let raw = null;
  const readSource = (source) => {
    if (!source || typeof source !== 'object') {
      return null;
    }
    try {
      if (typeof source.getStats === 'function') {
        const s = source.getStats();
        if (
          s &&
          typeof s === 'object' &&
          (s.bc3dFallbacks !== undefined || s.bc3dFallbackUploadsSkipped !== undefined)
        ) {
          return s;
        }
      }
      if (
        source.stats &&
        typeof source.stats === 'object' &&
        (source.stats.bc3dFallbacks !== undefined ||
          source.stats.bc3dFallbackUploadsSkipped !== undefined)
      ) {
        return source.stats;
      }
      if (
        source.bc3dFallbacks !== undefined ||
        source.bc3dFallbackUploadsSkipped !== undefined
      ) {
        return source;
      }
    } catch (_) {
      return null;
    }
    return null;
  };

  if (stats && typeof stats === 'object') {
    raw = readSource(stats.textureStats) || readSource(stats.texturePackets);
  }
  if (!raw && processor && typeof processor === 'object') {
    const candidates = [processor.texturePackets, processor._texturePackets];
    if (typeof processor._getTexturePacketsExtension === 'function') {
      try {
        candidates.push(processor._getTexturePacketsExtension());
      } catch (_) {
        // ignore
      }
    } else if (processor.extension && typeof processor.extension === 'object') {
      let ext = processor.extension;
      while (ext) {
        candidates.push(ext);
        ext = ext.next || null;
      }
    }
    for (let i = 0; i < candidates.length; i += 1) {
      const found = readSource(candidates[i]);
      if (found) {
        raw = found;
        break;
      }
    }
  }
  if (!raw && stats && typeof stats === 'object') {
    raw = readSource(stats);
  }

  return {
    bc3dFallbacks: toNonNegativeInt(raw && raw.bc3dFallbacks, 0),
    bc3dFallbackUploadsSkipped: toNonNegativeInt(raw && raw.bc3dFallbackUploadsSkipped, 0),
  };
}

/**
 * W121 / W136: sanitized W109 WGSL pack mode & ranged-fetch counters, or null when unavailable.
 *
 * @returns {{ mode: 'buffer' | 'ranged' | 'mixed', packCount: number, shaderCount: number, rangeFetches: number, cachedWgslBytes: number, rangedShaderFetches: number, rangedShaderReady: number, rangedShaderErrors: number } | null}
 */
function resolveShaderPackTelemetry(processor, stats) {
  let raw = null;
  try {
    if (
      processor &&
      processor.wgslPackCache &&
      typeof processor.wgslPackCache.getStats === 'function'
    ) {
      raw = processor.wgslPackCache.getStats();
    } else if (stats && stats.wgslPackCache && typeof stats.wgslPackCache === 'object') {
      raw = stats.wgslPackCache;
    }
  } catch (_) {
    raw = null;
  }
  if (!raw || typeof raw !== 'object') {
    return null;
  }
  const rawMode = typeof raw.mode === 'string' ? raw.mode.trim().toLowerCase() : 'buffer';
  const mode =
    rawMode === 'ranged' || rawMode === 'mixed' || rawMode === 'buffer' ? rawMode : 'buffer';
  return {
    mode,
    packCount: toNonNegativeInt(raw.packCount, 0),
    shaderCount: toNonNegativeInt(raw.shaderCount, 0),
    rangeFetches: toNonNegativeInt(raw.rangeFetches, 0),
    cachedWgslBytes: toNonNegativeInt(raw.cachedWgslBytes, 0),
    rangedShaderFetches: toNonNegativeInt(
      processor && processor._rangedShaderFetches !== undefined
        ? processor._rangedShaderFetches
        : stats && stats.rangedShaderFetches !== undefined
          ? stats.rangedShaderFetches
          : raw.rangedShaderFetches,
      0
    ),
    rangedShaderReady: toNonNegativeInt(
      processor && processor._rangedShaderReady !== undefined
        ? processor._rangedShaderReady
        : stats && stats.rangedShaderReady !== undefined
          ? stats.rangedShaderReady
          : raw.rangedShaderReady,
      0
    ),
    rangedShaderErrors: toNonNegativeInt(
      processor && processor._rangedShaderErrors !== undefined
        ? processor._rangedShaderErrors
        : stats && stats.rangedShaderErrors !== undefined
          ? stats.rangedShaderErrors
          : raw.rangedShaderErrors,
      0
    ),
  };
}

const ASYNC_PIPELINE_TELEMETRY_KEYS = Object.freeze([
  'generation',
  'maxConcurrent',
  'queued',
  'inFlight',
  'peakQueueDepth',
  'compiled',
  'failed',
  'dedupeHits',
  'syncFallbacks',
  'autoLayoutRetries',
  'discardedStale',
  'pendingAsyncPipelineSkips',
]);

/**
 * W103: sanitized AsyncPipelineCache counters, or null when the processor has no async cache.
 *
 * @returns {object | null}
 */
function resolveAsyncPipelineTelemetry(processor) {
  let raw = null;
  try {
    if (typeof processor.getAsyncPipelineStats === 'function') {
      raw = processor.getAsyncPipelineStats();
    } else if (
      processor.asyncPipelineCache &&
      typeof processor.asyncPipelineCache.getStats === 'function'
    ) {
      raw = processor.asyncPipelineCache.getStats();
    }
  } catch (_) {
    raw = null;
  }
  if (!raw || typeof raw !== 'object') {
    return null;
  }
  const out = {};
  for (let i = 0; i < ASYNC_PIPELINE_TELEMETRY_KEYS.length; i += 1) {
    const key = ASYNC_PIPELINE_TELEMETRY_KEYS[i];
    out[key] = toNonNegativeInt(raw[key], 0);
  }
  return out;
}

/**
 * W100: read sanitized device-loss counters from `processor.getDeviceLossStats()` (or a
 * `stats.deviceLoss` object), defaulting to zeros / null / false.
 *
 * @returns {{ deviceLostCount: number, deviceRecoveredCount: number, lastDeviceLostReason: string | null, deviceRecovering: boolean }}
 */
function resolveDeviceLossTelemetry(processor, stats) {
  let raw = null;
  if (processor && typeof processor.getDeviceLossStats === 'function') {
    try {
      raw = processor.getDeviceLossStats();
    } catch (_) {
      raw = null;
    }
  }
  if (
    (!raw || typeof raw !== 'object') &&
    stats &&
    stats.deviceLoss &&
    typeof stats.deviceLoss === 'object'
  ) {
    raw = stats.deviceLoss;
  }
  const src = raw && typeof raw === 'object' ? raw : {};
  let lastDeviceLostReason = null;
  if (typeof src.lastDeviceLostReason === 'string' && src.lastDeviceLostReason.length > 0) {
    lastDeviceLostReason = sanitizeGpuErrorDetail({ message: src.lastDeviceLostReason });
  }
  const deviceStats = {
    deviceLostCount: toNonNegativeInt(src.deviceLostCount, 0),
    deviceRecoveredCount: toNonNegativeInt(src.deviceRecoveredCount, 0),
    lastDeviceLostReason,
    deviceRecovering: src.deviceRecovering === true,
  };
  // W104: live-resource replay counters (only when the processor tracks replay).
  if (src.replayedResourceCount !== undefined || src.replayedBufferWriteCount !== undefined) {
    deviceStats.replayedResourceCount = toNonNegativeInt(src.replayedResourceCount, 0);
    deviceStats.replayedBufferWriteCount = toNonNegativeInt(src.replayedBufferWriteCount, 0);
  }
  return deviceStats;
}

const RENDER_STATE_TELEMETRY_KEYS = Object.freeze([
  'computeDispatchCount',
  'computePipelineCount',
  'autoLayoutFallbacks',
  'bindGroupKindMismatches',
]);

function hasRenderStateTelemetryKeys(obj) {
  if (!obj || typeof obj !== 'object') {
    return false;
  }
  for (let i = 0; i < RENDER_STATE_TELEMETRY_KEYS.length; i += 1) {
    if (obj[RENDER_STATE_TELEMETRY_KEYS[i]] !== undefined) {
      return true;
    }
  }
  return false;
}

function readRenderStateTelemetry(source) {
  if (!source || typeof source !== 'object') {
    return null;
  }
  try {
    if (typeof source.getTelemetry === 'function') {
      const t = source.getTelemetry();
      if (hasRenderStateTelemetryKeys(t)) return t;
    }
    if (hasRenderStateTelemetryKeys(source.telemetry)) {
      return source.telemetry;
    }
    if (typeof source.getStats === 'function') {
      const s = source.getStats();
      if (hasRenderStateTelemetryKeys(s)) return s;
    }
  } catch (_) {
    return null;
  }
  return hasRenderStateTelemetryKeys(source) ? source : null;
}

/**
 * Locate the live render_state_packets.js telemetry (compute dispatch / compute pipeline /
 * auto-layout fallback counters) attached to a processor, or null when absent.
 *
 * @param {object | null | undefined} processor
 * @param {object | null} stats
 * @returns {object | null}
 */
function resolveRenderStateTelemetry(processor, stats) {
  if (stats && typeof stats === 'object') {
    const fromStats = readRenderStateTelemetry(stats.renderState);
    if (fromStats) return fromStats;
  }
  if (processor && typeof processor === 'object') {
    const candidates = [processor.renderStateProcessor, processor._renderStatePackets];
    if (typeof processor._getRenderStateExtension === 'function') {
      try {
        candidates.push(processor._getRenderStateExtension());
      } catch (_) {
        // ignore
      }
    }
    for (let i = 0; i < candidates.length; i += 1) {
      const t = readRenderStateTelemetry(candidates[i]);
      if (t) return t;
    }
  }
  return hasRenderStateTelemetryKeys(stats) ? stats : null;
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

  // W100: bounded device-loss recovery state (per worker session).
  const deviceRecoveryEnabled = options.enableDeviceRecovery !== false;
  const maxDeviceRecoveries =
    Number.isInteger(options.maxDeviceRecoveries) && options.maxDeviceRecoveries >= 0
      ? options.maxDeviceRecoveries
      : MAX_DEVICE_RECOVERY_ATTEMPTS;
  const deviceRecoveryMinIntervalMs =
    typeof options.deviceRecoveryMinIntervalMs === 'number' &&
    Number.isFinite(options.deviceRecoveryMinIntervalMs) &&
    options.deviceRecoveryMinIntervalMs >= 0
      ? options.deviceRecoveryMinIntervalMs
      : DEFAULT_DEVICE_RECOVERY_MIN_INTERVAL_MS;
  const recoverGpuDevice =
    typeof options.recoverGpuDevice === 'function'
      ? options.recoverGpuDevice
      : (targetProcessor) => defaultRecoverGpuDevice(targetProcessor);
  let deviceRecoveryAttempts = 0;
  let deviceRecoveryInFlight = null;
  let deviceRecoveryExhaustedLogged = false;
  let lastDeviceRecoveryStartMs = -Infinity;
  let boundDeviceLossProcessor = null;

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
    bindProcessorDeviceLoss(targetProcessor);
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
    const streamThresholdBytes = resolveShaderPackStreamThresholdBytes(initPayload, options);
    const initialRangeBytes =
      initPayload.initialRangeBytes !== undefined
        ? initPayload.initialRangeBytes
        : options.initialRangeBytes;
    const maxCachedWgslBytes =
      initPayload.maxCachedWgslBytes !== undefined
        ? initPayload.maxCachedWgslBytes
        : options.maxCachedWgslBytes;
    const fetchInit =
      initPayload.fetchInit !== undefined ? initPayload.fetchInit : options.fetchInit;
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
            const hasLoadPackUrl = Boolean(
              targetProcessor.wgslPackCache &&
                typeof targetProcessor.wgslPackCache.loadPackUrl === 'function'
            );
            if (hasLoadPackUrl) {
              const loadOpts = {
                fetch: activeFetch,
                packName: shaderPackUrl,
              };
              if (streamThresholdBytes !== undefined) {
                loadOpts.streamThresholdBytes = streamThresholdBytes;
              }
              if (
                Number.isInteger(initialRangeBytes) &&
                initialRangeBytes >= WGPK_HEADER_BYTES
              ) {
                loadOpts.initialRangeBytes = initialRangeBytes;
              }
              if (Number.isInteger(maxCachedWgslBytes) && maxCachedWgslBytes >= 0) {
                loadOpts.maxCachedWgslBytes = maxCachedWgslBytes;
              }
              if (fetchInit !== undefined) {
                loadOpts.fetchInit = fetchInit;
              }
              await targetProcessor.wgslPackCache.loadPackUrl(shaderPackUrl, loadOpts);
              if (
                targetProcessor.shaderFallbackRouter &&
                targetProcessor.shaderFallbackRouter._packLoader == null &&
                typeof targetProcessor.shaderFallbackRouter.setPackLoader === 'function'
              ) {
                targetProcessor.shaderFallbackRouter.setPackLoader(targetProcessor.wgslPackCache);
              }
            } else if (
              streamThresholdBytes !== undefined &&
              streamThresholdBytes > 0 &&
              typeof targetProcessor.loadShaderPackUrl === 'function'
            ) {
              await targetProcessor.loadShaderPackUrl(shaderPackUrl, {
                fetch: activeFetch,
                packName: shaderPackUrl,
                streamThresholdBytes,
                initialRangeBytes,
                maxCachedWgslBytes,
                fetchInit,
              });
            } else {
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

  /**
   * Install the packet ABI v2 render-state + texture packet extensions on `targetProcessor`.
   * Also used after W100 device recovery, since extensions own device-bound objects.
   *
   * @returns {boolean} True when installation did not throw.
   */
  function installPacketExtensions(targetProcessor) {
    try {
      let renderStateApi =
        typeof globalThis !== 'undefined' && globalThis.PlayGta6RenderState
          ? globalThis.PlayGta6RenderState
          : null;
      if (!renderStateApi && typeof importScripts === 'function') {
        importScripts('./render_state_packets.js');
        renderStateApi =
          typeof globalThis !== 'undefined' && globalThis.PlayGta6RenderState
            ? globalThis.PlayGta6RenderState
            : null;
      } else if (!renderStateApi && typeof require === 'function') {
        renderStateApi = require('./render_state_packets.js');
      }
      if (renderStateApi && typeof renderStateApi.installRenderStatePackets === 'function') {
        renderStateApi.installRenderStatePackets(targetProcessor);
      }
      let texturePacketsApi =
        typeof globalThis !== 'undefined' && globalThis.PlayGta6TexturePackets
          ? globalThis.PlayGta6TexturePackets
          : null;
      if (!texturePacketsApi && typeof importScripts === 'function') {
        importScripts('./texture_packets.js');
        texturePacketsApi =
          typeof globalThis !== 'undefined' && globalThis.PlayGta6TexturePackets
            ? globalThis.PlayGta6TexturePackets
            : null;
      } else if (!texturePacketsApi && typeof require === 'function') {
        texturePacketsApi = require('./texture_packets.js');
      }
      if (texturePacketsApi && typeof texturePacketsApi.installTexturePackets === 'function') {
        texturePacketsApi.installTexturePackets(targetProcessor);
      }
      return true;
    } catch (err) {
      postMessage({ error: `packet ABI v2 unavailable: ${String(err)}` });
      return false;
    }
  }

  /**
   * W100 default device re-acquisition: re-request adapter + device with the same feature /
   * limit negotiation as init and re-acquire the OffscreenCanvas `webgpu` context.
   *
   * @returns {Promise<{ device: GPUDevice, ctx: GPUCanvasContext | null, format: string }>}
   */
  async function defaultRecoverGpuDevice(targetProcessor) {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      throw new Error('navigator.gpu unavailable in worker');
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      throw new Error('no WebGPU adapter available');
    }
    const { device } = await requestNegotiatedDevice(adapter);
    if (!device || typeof device !== 'object') {
      throw new Error('requestDevice returned no device');
    }
    let format = targetProcessor && targetProcessor.format ? targetProcessor.format : 'bgra8unorm';
    try {
      const preferred = navigator.gpu.getPreferredCanvasFormat();
      if (typeof preferred === 'string' && preferred.length > 0) {
        format = preferred;
      }
    } catch (_) {
      // Keep the previous format.
    }
    let ctx = targetProcessor ? targetProcessor.ctx : null;
    if (canvas && typeof canvas.getContext === 'function') {
      try {
        const nextCtx = canvas.getContext('webgpu');
        if (nextCtx) {
          ctx = nextCtx;
        }
      } catch (_) {
        // Fall back to the existing context object.
      }
    }
    return { device, ctx, format };
  }

  /**
   * W100: one async recovery attempt. Never rejects; always leaves recovery mode.
   */
  async function runDeviceRecovery(targetProcessor, attempt) {
    // Synchronous before the first await so submissions are dropped immediately.
    targetProcessor.beginDeviceRecovery();
    let ok = false;
    let acquiredDevice = null;
    try {
      const acquired = await recoverGpuDevice(targetProcessor, { canvas, attempt });
      if (!acquired || !acquired.device) {
        throw new Error('device recovery returned no device');
      }
      acquiredDevice = acquired.device;
      if (targetProcessor !== processor || targetProcessor.destroyed) {
        throw new Error('processor replaced or destroyed during recovery');
      }
      targetProcessor.rebindDevice(acquired.device, acquired.ctx, {
        format: acquired.format,
        alphaMode: 'opaque',
      });
      installPacketExtensions(targetProcessor);
      // W104/W105: recreate live pre-loss guest resources on the new device in async chunks so
      // Chrome's GPU IPC drains cleanly without stalling the worker.
      if (typeof targetProcessor.replayRecordedResourcesAsync === 'function') {
        try {
          await targetProcessor.replayRecordedResourcesAsync();
        } catch (replayErr) {
          targetProcessor._logDiagnostic(
            'warn',
            `[wgpu_worker] GPU resource replay failed: ${sanitizeGpuErrorDetail(replayErr)}`
          );
        }
      } else if (typeof targetProcessor.replayRecordedResources === 'function') {
        try {
          targetProcessor.replayRecordedResources();
        } catch (replayErr) {
          targetProcessor._logDiagnostic(
            'warn',
            `[wgpu_worker] GPU resource replay failed: ${sanitizeGpuErrorDetail(replayErr)}`
          );
        }
      }
      bindProcessorValidationErrors(targetProcessor);
      ok = true;
      targetProcessor._logDiagnostic(
        'info',
        `[wgpu_worker] WebGPU device recovered (attempt ${attempt}/${maxDeviceRecoveries})`
      );
    } catch (err) {
      if (acquiredDevice && acquiredDevice !== targetProcessor.device) {
        try {
          if (typeof acquiredDevice.destroy === 'function') {
            acquiredDevice.destroy();
          }
        } catch (_) {
          // Ignore cleanup failure of an unused device.
        }
      }
      if (typeof targetProcessor._logDiagnostic === 'function') {
        targetProcessor._logDiagnostic(
          'error',
          `[wgpu_worker] WebGPU device recovery failed (attempt ${attempt}/${maxDeviceRecoveries}): ${sanitizeGpuErrorDetail(err)}`
        );
      }
    } finally {
      try {
        targetProcessor.endDeviceRecovery(ok);
      } catch (_) {
        // Never let bookkeeping escape.
      }
      try {
        if (targetProcessor === processor) {
          emitTelemetry();
        }
      } catch (_) {
        // Telemetry is best-effort.
      }
    }
    return ok;
  }

  /**
   * W100: device-loss listener. Re-entrancy guarded, rate limited (min interval between
   * attempts) and bounded (`maxDeviceRecoveries` attempts per worker session).
   */
  function handleProcessorDeviceLost(targetProcessor) {
    if (!deviceRecoveryEnabled || targetProcessor !== processor || deviceRecoveryInFlight) {
      return;
    }
    if (deviceRecoveryAttempts >= maxDeviceRecoveries) {
      if (!deviceRecoveryExhaustedLogged) {
        deviceRecoveryExhaustedLogged = true;
        targetProcessor._logDiagnostic(
          'error',
          `[wgpu_worker] WebGPU device recovery budget exhausted (${maxDeviceRecoveries} attempts); rendering stays disabled`
        );
      }
      return;
    }
    const nowVal = telemetryNow();
    const now = typeof nowVal === 'number' && Number.isFinite(nowVal) ? nowVal : Date.now();
    if (now - lastDeviceRecoveryStartMs < deviceRecoveryMinIntervalMs) {
      return;
    }
    lastDeviceRecoveryStartMs = now;
    deviceRecoveryAttempts += 1;
    const pending = runDeviceRecovery(targetProcessor, deviceRecoveryAttempts);
    deviceRecoveryInFlight = pending;
    pending.then(
      () => {
        if (deviceRecoveryInFlight === pending) {
          deviceRecoveryInFlight = null;
        }
      },
      () => {
        if (deviceRecoveryInFlight === pending) {
          deviceRecoveryInFlight = null;
        }
      }
    );
  }

  function bindProcessorDeviceLoss(targetProcessor) {
    if (boundDeviceLossProcessor && boundDeviceLossProcessor !== targetProcessor) {
      try {
        boundDeviceLossProcessor.setDeviceLossListener(null);
      } catch (_) {
        // Ignore cleanup errors on prior processor.
      }
    }
    boundDeviceLossProcessor = null;
    if (!targetProcessor || typeof targetProcessor.setDeviceLossListener !== 'function') {
      return;
    }
    targetProcessor.setDeviceLossListener(() => {
      handleProcessorDeviceLost(targetProcessor);
    });
    boundDeviceLossProcessor = targetProcessor;
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
    if (adapter.info && typeof adapter.info === 'object') {
      const desc = [adapter.info.vendor, adapter.info.architecture, adapter.info.description]
        .filter((s) => typeof s === 'string' && s.length > 0)
        .join(' / ');
      const adapterLogger =
        (options && options.logger) || (typeof console !== 'undefined' ? console : null);
      if (desc && adapterLogger && typeof adapterLogger.info === 'function') {
        adapterLogger.info(`[wgpu_worker] WebGPU adapter: ${sanitizeGpuErrorDetail(desc)}`);
      }
    }

    const { device, enabledFeatures } = await requestNegotiatedDevice(adapter);
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
      logger: options.logger,
      captureValidationErrors: true,
      maxValidationErrorHistory:
        initPayload.maxValidationErrorHistory ?? options.maxValidationErrorHistory,
      maxValidationErrorPosts:
        initPayload.maxValidationErrorPosts ?? options.maxValidationErrorPosts,
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
      wgslPackCache: initPayload.wgslPackCache ?? options.wgslPackCache,
      wgslPackBuffer: initPayload.wgslPackBuffer ?? options.wgslPackBuffer,
      streamThresholdBytes:
        initPayload.streamThresholdBytes ?? options.streamThresholdBytes,
      streamShaders: initPayload.streamShaders ?? options.streamShaders,
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
      lazyShaderCompile:
        initPayload.lazyShaderCompile !== undefined
          ? initPayload.lazyShaderCompile
          : options.lazyShaderCompile !== undefined
            ? options.lazyShaderCompile
            : 'pack',
    });
    bindProcessorValidationErrors(processor);
    installPacketExtensions(processor);
    if (attachedRing) {
      processor.attachRing(attachedRing);
    }

    await preloadBootShaderAssets(processor, initPayload);
    preloadedInDefaultInit = true;

    const info = adapter.info
      ? `${adapter.info.vendor || 'gpu'} ${adapter.info.architecture || ''}`.trim()
      : 'WebGPU';
    postMessage({ ready: true, adapter: info, features: enabledFeatures });
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
      } else if (
        (payload.loadShaderPack ||
          payload.type === 'load-shader-pack' ||
          payload.type === 'loadShaderPack') &&
        processor
      ) {
        const packUrl =
          typeof payload.loadShaderPack === 'string'
            ? payload.loadShaderPack
            : typeof payload.shaderPackUrl === 'string'
              ? payload.shaderPackUrl
              : typeof payload.url === 'string'
                ? payload.url
                : null;
        const loadSummary = packUrl
          ? await preloadBootShaderAssets(processor, {
              ...payload,
              shaderPackUrl: packUrl,
            })
          : { shaderPackLoaded: false, shaderPackShaders: 0 };
        const packStats =
          processor.wgslPackCache && typeof processor.wgslPackCache.getStats === 'function'
            ? processor.wgslPackCache.getStats()
            : null;
        postMessage({
          id: payload.id,
          shaderPackLoaded: loadSummary.shaderPackLoaded,
          shaderCount: loadSummary.shaderPackShaders,
          mode: packStats && typeof packStats.mode === 'string' ? packStats.mode : 'buffer',
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
      } else if (
        payload.stats ||
        payload.type === 'gpu-stats' ||
        payload.type === 'request-gpu-stats'
      ) {
        const stats =
          processor && typeof processor.getStats === 'function' ? processor.getStats() : null;
        if (payload.type === 'gpu-stats' || payload.type === 'request-gpu-stats') {
          const reply = {
            type: 'gpu-stats',
            stats,
            framesPresented: stats ? toNonNegativeInt(stats.framesPresented, 0) : 0,
            presentCount: stats ? toNonNegativeInt(stats.presentCount, 0) : 0,
            drawCalls: stats ? toNonNegativeInt(stats.drawCalls, 0) : 0,
            realPackDrawCount: stats ? toNonNegativeInt(stats.realPackDrawCount, 0) : 0,
            fallbackDrawCount: stats ? toNonNegativeInt(stats.fallbackDrawCount, 0) : 0,
            validationErrorCount: stats ? toNonNegativeInt(stats.validationErrorCount, 0) : 0,
          };
          if (payload.id !== undefined) {
            reply.id = payload.id;
          }
          postMessage(reply);
        } else {
          postMessage({
            id: payload.id,
            stats,
          });
        }
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

        if (processor.isDeviceRecovering === true) {
          // W100: drop splash/Bink frames cleanly while the lost device is being re-acquired.
          const recoveringReply = {
            type: 'splash-frame-presented',
            frameIndex: requestedFrameIndex,
            framesPresented: toNonNegativeInt(processor.frameCount, 0),
            ok: false,
            reason: 'device-recovering',
            error: 'WebGPU device recovery in progress',
          };
          if (payload.id !== undefined) {
            recoveringReply.id = payload.id;
          }
          postMessage(recoveringReply);
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
        // W100: intentional host destroy must never trigger device-loss recovery.
        bindProcessorDeviceLoss(null);
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
  selectSupportedAdapterFeatures,
  OPTIONAL_ADAPTER_FEATURES,
  selectSupportedAdapterLimits,
  OPTIONAL_ADAPTER_LIMIT_SPECS,
  selectFallbackShaderProfile,
  sanitizeValidationErrorMessage,
  normalizeValidationErrorType,
  FALLBACK_PROFILE,
  DEFAULT_MAX_VALIDATION_ERROR_HISTORY,
  DEFAULT_MAX_VALIDATION_ERROR_POSTS,
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
