// GPU packet ABI v2 (opcodes 22..35, docs/gpu_commands.md "Opcodes (v2)") for WgpuCommandProcessor:
// explicit bind group / pipeline layouts, pipelines built from the 376-byte render pipeline key,
// explicit render passes, dynamic render state, bind groups with texture view descriptors and
// dynamic offsets, comparison samplers, v2 textures (CreateTexture2), and hardened compute dispatches.
//
// Install with installRenderStatePackets(processor). Objects land in the processor's own tables
// (pipelines, computePipelines, samplers, bindGroups) so v1 SetPipeline / SetComputePipeline /
// SetBindGroup / DestroyResource keep working on them. Pass-level packets are queued in the
// processor's per-frame command list and replayed at Present through the `run(passControl)` hook.

(function (globalScope) {
  'use strict';

// "WebGPU enum codes" (docs/gpu_commands.md): numbering of the pso::Wgpu* enums and the *_name()
// tables in runtime/gpu/pso_cache.{h,cpp}. tests/test_render_state_packets.mjs cross-checks them.
const TEXTURE_FORMAT_NAMES = Object.freeze([
  'undefined', 'r8unorm', 'r8snorm', 'r8uint', 'r8sint', 'r16unorm', 'r16snorm', 'r16uint',
  'r16sint', 'r16float', 'rg8unorm', 'rg8snorm', 'rg8uint', 'rg8sint', 'r32uint', 'r32sint',
  'r32float', 'rg16unorm', 'rg16snorm', 'rg16uint', 'rg16sint', 'rg16float', 'rgba8unorm',
  'rgba8unorm-srgb', 'rgba8snorm', 'rgba8uint', 'rgba8sint', 'bgra8unorm', 'bgra8unorm-srgb',
  'rgb9e5ufloat', 'rgb10a2uint', 'rgb10a2unorm', 'rg11b10ufloat', 'rg32uint', 'rg32sint',
  'rg32float', 'rgba16unorm', 'rgba16snorm', 'rgba16uint', 'rgba16sint', 'rgba16float',
  'rgba32uint', 'rgba32sint', 'rgba32float', 'stencil8', 'depth16unorm', 'depth24plus',
  'depth24plus-stencil8', 'depth32float', 'depth32float-stencil8', 'bc1-rgba-unorm',
  'bc1-rgba-unorm-srgb', 'bc2-rgba-unorm', 'bc2-rgba-unorm-srgb', 'bc3-rgba-unorm',
  'bc3-rgba-unorm-srgb', 'bc4-r-unorm', 'bc4-r-snorm', 'bc5-rg-unorm', 'bc5-rg-snorm',
  'bc6h-rgb-ufloat', 'bc6h-rgb-float', 'bc7-rgba-unorm', 'bc7-rgba-unorm-srgb',
]);
const VERTEX_FORMAT_NAMES = Object.freeze([
  'undefined', 'uint8', 'uint8x2', 'uint8x4', 'sint8', 'sint8x2', 'sint8x4', 'unorm8', 'unorm8x2',
  'unorm8x4', 'snorm8', 'snorm8x2', 'snorm8x4', 'uint16', 'uint16x2', 'uint16x4', 'sint16',
  'sint16x2', 'sint16x4', 'unorm16', 'unorm16x2', 'unorm16x4', 'snorm16', 'snorm16x2', 'snorm16x4',
  'float16', 'float16x2', 'float16x4', 'float32', 'float32x2', 'float32x3', 'float32x4', 'uint32',
  'uint32x2', 'uint32x3', 'uint32x4', 'sint32', 'sint32x2', 'sint32x3', 'sint32x4',
  'unorm10-10-10-2', 'unorm8x4-bgra',
]);
const BLEND_FACTOR_NAMES = Object.freeze([
  'zero', 'one', 'src', 'one-minus-src', 'src-alpha', 'one-minus-src-alpha', 'dst', 'one-minus-dst',
  'dst-alpha', 'one-minus-dst-alpha', 'src-alpha-saturated', 'constant', 'one-minus-constant',
  'src1', 'one-minus-src1', 'src1-alpha', 'one-minus-src1-alpha',
]);
const BLEND_OPERATION_NAMES = Object.freeze(['add', 'subtract', 'reverse-subtract', 'min', 'max']);
const COMPARE_FUNCTION_NAMES = Object.freeze([
  'undefined', 'never', 'less', 'equal', 'less-equal', 'greater', 'not-equal', 'greater-equal', 'always',
]);
const STENCIL_OPERATION_NAMES = Object.freeze([
  'keep', 'zero', 'replace', 'invert', 'increment-clamp', 'decrement-clamp', 'increment-wrap',
  'decrement-wrap',
]);
const CULL_MODE_NAMES = Object.freeze(['none', 'front', 'back']);
const FRONT_FACE_NAMES = Object.freeze(['ccw', 'cw']);
const PRIMITIVE_TOPOLOGY_NAMES = Object.freeze([
  'triangle-list', 'triangle-strip', 'line-list', 'line-strip', 'point-list',
]);
const INDEX_FORMAT_NAMES = Object.freeze(['undefined', 'uint16', 'uint32']);
const VERTEX_STEP_MODE_NAMES = Object.freeze(['unused', 'vertex', 'instance']);
const LOAD_OP_NAMES = Object.freeze(['undefined', 'load', 'clear']);
const STORE_OP_NAMES = Object.freeze(['undefined', 'store', 'discard']);
const TEXTURE_ASPECT_NAMES = Object.freeze(['all', 'depth-only', 'stencil-only']);

// Packet-level enums defined in runtime/abi/playgta6_abi.h.
const VIEW_DIMENSION_NAMES = Object.freeze([
  'undefined', '1d', '2d', '2d-array', 'cube', 'cube-array', '3d',
]);
const BINDING_LAYOUT_TYPE = Object.freeze({ BUFFER: 0, SAMPLER: 1, TEXTURE: 2, STORAGE_TEXTURE: 3 });
const BUFFER_BINDING_TYPE_NAMES = Object.freeze(['uniform', 'storage', 'read-only-storage']);
const SAMPLER_BINDING_TYPE_NAMES = Object.freeze(['filtering', 'non-filtering', 'comparison']);
const TEXTURE_SAMPLE_TYPE_NAMES = Object.freeze(['float', 'unfilterable-float', 'depth', 'sint', 'uint']);
const STORAGE_TEXTURE_ACCESS_NAMES = Object.freeze(['write-only', 'read-only', 'read-write']);
const ADDRESS_MODE_NAMES = Object.freeze(['clamp-to-edge', 'repeat', 'mirror-repeat']);
const FILTER_MODE_NAMES = Object.freeze(['nearest', 'linear']);
const BIND_RESOURCE_TYPE = Object.freeze({ BUFFER: 0, TEXTURE_VIEW: 1, SAMPLER: 2 });

const V2_OPCODES = Object.freeze({
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

const TEXTURE_DIMENSION_NAMES = Object.freeze(['1d', '2d', '3d']);

const SWAPCHAIN_TEXTURE_ID = 0xffffffff;
const MANIFEST_REFERENCE_FLAG = 0x1;
const KEY_OFFSET = 16; // CreateRenderPipeline2: header, then the 376-byte key
const KEY_SIZE = 376;
const CREATE_TEXTURE2_BYTES = 40;
const MAX_TEXTURE_VIEW_FORMATS = 8;
const GPU_BUFFER_USAGE_UNIFORM = 0x0040;
const GPU_BUFFER_USAGE_STORAGE = 0x0080;
const GPU_BUFFER_USAGE_UNIFORM_STORAGE_COPY_DST = GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_STORAGE | 0x0008;
const GPU_TEXTURE_USAGE_COPY_SRC = 0x0001;
const GPU_TEXTURE_USAGE_COPY_DST = 0x0002;
const GPU_TEXTURE_USAGE_TEXTURE_BINDING = 0x0004;
const GPU_TEXTURE_USAGE_STORAGE_BINDING = 0x0008;
const GPU_TEXTURE_USAGE_RENDER_ATTACHMENT = 0x0010;
const EMPTY_FRAGMENT_WGSL = '@fragment fn fs_main() {}\n';
const GPU_SHADER_STAGE_VERTEX = 0x1;
const GPU_SHADER_STAGE_ALL_MASK = 0x7;
const MAX_TELEMETRY_WARNINGS = 32;
// W101: upper bound on cached layout:'auto' remapped bind groups. Root-constant ring offsets rotate
// per draw, so the (pipeline, bind group, dynamic offsets) key space is unbounded; evict LRU-first.
const REMAPPED_BIND_GROUP_CACHE_LIMIT = 2048;
// W186: upper bound on cached PresentX multi-plane (planeTextureId, viewFormat, sampler, pipeline/layout)
// texture views & bind groups across frames; LRU-evicted and invalidated on handle/device/reset changes.
const PRESENTX_BIND_GROUP_CACHE_LIMIT = 256;
// W104: bound on distinct missing render-pass textures remembered for error deduplication.
const MISSING_TEXTURE_REPORT_LIMIT = 4096;

let _cachedTextureFormatsModule = null;
const _gpuObjectIds = new WeakMap();
let _nextGpuObjectId = 1;

function gpuObjectId(obj) {
  if (!obj || (typeof obj !== 'object' && typeof obj !== 'function')) return 0;
  let id = _gpuObjectIds.get(obj);
  if (id === undefined) {
    id = _nextGpuObjectId;
    _nextGpuObjectId += 1;
    _gpuObjectIds.set(obj, id);
  }
  return id;
}

function buildTextureViewKey(desc) {
  if (!desc || typeof desc !== 'object') return '';
  const fmt = desc.format || '';
  const dim = desc.dimension || '';
  const aspect = desc.aspect || '';
  const baseMip = desc.baseMipLevel || 0;
  const mipCount = desc.mipLevelCount || 0;
  const baseLayer = desc.baseArrayLayer || 0;
  const layerCount = desc.arrayLayerCount || 0;
  if (!fmt && !dim && !aspect && !baseMip && !mipCount && !baseLayer && !layerCount) {
    return '';
  }
  return `${fmt}|${dim}|${aspect}|${baseMip}|${mipCount}|${baseLayer}|${layerCount}`;
}

function getResolveStorageCompatibleFormatFn(options) {
  if (options && typeof options.resolveStorageCompatibleFormat === 'function') {
    return options.resolveStorageCompatibleFormat;
  }
  if (
    options &&
    options.textureFormats &&
    typeof options.textureFormats.resolveStorageCompatibleFormat === 'function'
  ) {
    return options.textureFormats.resolveStorageCompatibleFormat;
  }
  if (
    typeof globalThis !== 'undefined' &&
    globalThis.PlayGta6TextureFormats &&
    typeof globalThis.PlayGta6TextureFormats.resolveStorageCompatibleFormat === 'function'
  ) {
    return globalThis.PlayGta6TextureFormats.resolveStorageCompatibleFormat;
  }
  if (
    globalScope &&
    globalScope.PlayGta6TextureFormats &&
    typeof globalScope.PlayGta6TextureFormats.resolveStorageCompatibleFormat === 'function'
  ) {
    return globalScope.PlayGta6TextureFormats.resolveStorageCompatibleFormat;
  }
  if (_cachedTextureFormatsModule === null && typeof require === 'function') {
    try {
      _cachedTextureFormatsModule = require('./texture_formats.js') || false;
    } catch (_) {
      _cachedTextureFormatsModule = false;
    }
  }
  if (
    _cachedTextureFormatsModule &&
    typeof _cachedTextureFormatsModule.resolveStorageCompatibleFormat === 'function'
  ) {
    return _cachedTextureFormatsModule.resolveStorageCompatibleFormat;
  }
  return null;
}
const DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION = 65535;
const VALID_WGSL_IDENT_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;
const WGSL_GROUP0_RE = /@group\(\s*0\s*\)/;

const HOOK_ORIG_FLUSH_PRESENT = '__playgta6RsOrigFlushPresent';
const HOOK_ORIG_CREATE_COMPUTE_PIPELINE = '__playgta6RsOrigCreateComputePipeline';
const HOOK_ACTIVE_HANDLER = '__playgta6RsActiveHandler';

const DEFAULT_ENTRY_POINTS = Object.freeze({
  vertex: 'vs_main',
  fragment: 'fs_main',
  compute: 'cs_main',
});

const DEPTH_TEXTURE_FORMATS = Object.freeze(
  new Set([
    'depth16unorm',
    'depth24plus',
    'depth24plus-stencil8',
    'depth32float',
    'depth32float-stencil8',
  ])
);

const FLOAT32_TEXTURE_FORMATS = Object.freeze(
  new Set(['r32float', 'rg32float', 'rgba32float'])
);

// Core WebGPU storage-compatible texture formats ("Texture Format Capabilities").
const CORE_STORAGE_TEXTURE_FORMATS = Object.freeze(
  new Set([
    'rgba8unorm',
    'rgba8snorm',
    'rgba8uint',
    'rgba8sint',
    'rgba16uint',
    'rgba16sint',
    'rgba16float',
    'r32uint',
    'r32sint',
    'r32float',
    'rg32uint',
    'rg32sint',
    'rg32float',
    'rgba32uint',
    'rgba32sint',
    'rgba32float',
  ])
);

const TIER1_STORAGE_TEXTURE_FORMATS = Object.freeze(
  new Set([
    'r8unorm',
    'r8snorm',
    'r8uint',
    'r8sint',
    'r16unorm',
    'r16snorm',
    'r16uint',
    'r16sint',
    'r16float',
    'rg8unorm',
    'rg8snorm',
    'rg8uint',
    'rg8sint',
    'rg16unorm',
    'rg16snorm',
    'rg16uint',
    'rg16sint',
    'rg16float',
    'rgb10a2uint',
    'rgb10a2unorm',
    'rg11b10ufloat',
    'rgba16unorm',
    'rgba16snorm',
  ])
);

const NON_RENDERABLE_CORE_FORMATS = Object.freeze(
  new Set(['rgb9e5ufloat', 'r8snorm', 'rg8snorm', 'rgba8snorm'])
);

const TIER1_RENDERABLE_FORMATS = Object.freeze(
  new Set(['r16unorm', 'r16snorm', 'rg16unorm', 'rg16snorm', 'rgba16unorm', 'rgba16snorm'])
);

// Core WebGPU blendable color target formats ("Texture Format Capabilities").
const BLENDABLE_CORE_FORMATS = Object.freeze(
  new Set([
    'r8unorm',
    'r16float',
    'rg8unorm',
    'rg16float',
    'rgba8unorm',
    'rgba8unorm-srgb',
    'bgra8unorm',
    'bgra8unorm-srgb',
    'rgb10a2unorm',
    'rgba16float',
  ])
);

const NON_MULTISAMPLEABLE_CORE_FORMATS = Object.freeze(
  new Set([
    'r32uint',
    'r32sint',
    'r32float',
    'rg32uint',
    'rg32sint',
    'rg32float',
    'rgba32uint',
    'rgba32sint',
    'rgba32float',
    'rgb10a2uint',
  ])
);

const VERTEX_FORMAT_BYTE_SIZES = Object.freeze({
  uint8: 1,
  uint8x2: 2,
  uint8x4: 4,
  sint8: 1,
  sint8x2: 2,
  sint8x4: 4,
  unorm8: 1,
  unorm8x2: 2,
  unorm8x4: 4,
  snorm8: 1,
  snorm8x2: 2,
  snorm8x4: 4,
  uint16: 2,
  uint16x2: 4,
  uint16x4: 8,
  sint16: 2,
  sint16x2: 4,
  sint16x4: 8,
  unorm16: 2,
  unorm16x2: 4,
  unorm16x4: 8,
  snorm16: 2,
  snorm16x2: 4,
  snorm16x4: 8,
  float16: 2,
  float16x2: 4,
  float16x4: 8,
  float32: 4,
  float32x2: 8,
  float32x3: 12,
  float32x4: 16,
  uint32: 4,
  uint32x2: 8,
  uint32x3: 12,
  uint32x4: 16,
  sint32: 4,
  sint32x2: 8,
  sint32x3: 12,
  sint32x4: 16,
  'unorm10-10-10-2': 4,
  'unorm8x4-bgra': 4,
});

function isDepthTextureFormat(format) {
  return typeof format === 'string' && DEPTH_TEXTURE_FORMATS.has(format);
}

function isDepthOrStencilTextureFormat(format) {
  return typeof format === 'string' && (DEPTH_TEXTURE_FORMATS.has(format) || format === 'stencil8');
}

function isBcCompressedFormat(format) {
  return typeof format === 'string' && format.startsWith('bc');
}

function isSrgbTextureFormat(format) {
  return typeof format === 'string' && format.endsWith('-srgb');
}

function isFloat32TextureFormat(format) {
  return typeof format === 'string' && FLOAT32_TEXTURE_FORMATS.has(format);
}

function isIntegerTextureFormat(format) {
  if (typeof format !== 'string' || format.length === 0) return false;
  const normalized = format.trim().toLowerCase();
  return normalized.endsWith('uint') || normalized.endsWith('sint');
}

// W266: WebGPU forbids blending on integer (*uint/*sint), depth/stencil, compressed, and
// non-blendable float32 formats (r32float/rg32float/rgba32float without 'float32-blendable').
function isBlendableTextureFormat(format, device) {
  if (typeof format !== 'string' || format.length === 0 || format === 'undefined') return false;
  const normalized = format.trim().toLowerCase();
  if (isIntegerTextureFormat(normalized)) return false;
  if (isDepthOrStencilTextureFormat(normalized) || isBcCompressedFormat(normalized)) return false;
  if (BLENDABLE_CORE_FORMATS.has(normalized)) return true;
  if (FLOAT32_TEXTURE_FORMATS.has(normalized)) {
    return deviceHasFeature(device, 'float32-blendable');
  }
  if (normalized === 'rg11b10ufloat') {
    return deviceHasFeature(device, 'rg11b10ufloat-renderable');
  }
  if (TIER1_RENDERABLE_FORMATS.has(normalized)) {
    return deviceHasFeature(device, 'texture-formats-tier1');
  }
  return false;
}

// W266: strips `.blend` from a GPUColorTargetState when its format is not blendable on `device`,
// preventing WebGPU pipeline creation from failing with "Format is not blendable".
function sanitizeColorTargetBlendState(target, device) {
  if (!target || typeof target !== 'object' || !target.blend) {
    return { target, stripped: false, reason: null };
  }
  const format = typeof target.format === 'string' ? target.format.trim().toLowerCase() : '';
  if (isBlendableTextureFormat(format, device)) {
    return { target, stripped: false, reason: null };
  }
  const cleanTarget = { ...target };
  delete cleanTarget.blend;
  let reason = 'format-not-blendable';
  if (isIntegerTextureFormat(format)) {
    reason = 'integer-format-not-blendable';
  } else if (FLOAT32_TEXTURE_FORMATS.has(format)) {
    reason = 'float32-format-not-blendable';
  }
  return { target: cleanTarget, stripped: true, reason };
}

function deviceHasFeature(device, featureName) {
  const features = device && device.features;
  if (!features || !featureName) return false;
  if (typeof features.has === 'function') return Boolean(features.has(featureName));
  if (Array.isArray(features)) return features.includes(featureName);
  if (typeof features[Symbol.iterator] === 'function') {
    for (const item of features) {
      if (item === featureName) return true;
    }
    return false;
  }
  if (typeof features === 'object') return Boolean(features[featureName]);
  return false;
}

// W163/W186: resolves storage-texture compatibility using texture_formats.js's
// resolveStorageCompatibleFormat when loaded, falling back to built-in tables in isolated VMs.
function resolveStorageTextureFormat(format, device, options) {
  if (typeof format !== 'string' || format.length === 0 || format === 'undefined') {
    return {
      storageFormat: null,
      supportsStorageBinding: false,
      needsLinearSiblingView: false,
      reason: 'invalid-format',
    };
  }
  const helper = getResolveStorageCompatibleFormatFn(options);
  if (helper) {
    try {
      const resolved = helper(format, { deviceFeatures: device && device.features });
      if (resolved && typeof resolved === 'object') {
        return {
          storageFormat: typeof resolved.storageFormat === 'string' ? resolved.storageFormat : null,
          supportsStorageBinding: Boolean(resolved.supportsStorageBinding),
          needsLinearSiblingView: Boolean(resolved.needsLinearSiblingView),
          reason: typeof resolved.reason === 'string' ? resolved.reason : null,
        };
      }
    } catch (_) {
      // Fall through to built-in format tables if an injected helper throws.
    }
  }
  const normalized = format.trim().toLowerCase();
  if (isDepthOrStencilTextureFormat(normalized)) {
    return {
      storageFormat: null,
      supportsStorageBinding: false,
      needsLinearSiblingView: false,
      reason: 'depth-stencil-not-storage-compatible',
    };
  }
  if (isBcCompressedFormat(normalized)) {
    return {
      storageFormat: null,
      supportsStorageBinding: false,
      needsLinearSiblingView: false,
      reason: 'compressed-not-storage-compatible',
    };
  }
  if (CORE_STORAGE_TEXTURE_FORMATS.has(normalized)) {
    return {
      storageFormat: normalized,
      supportsStorageBinding: true,
      needsLinearSiblingView: false,
      reason: null,
    };
  }
  if (normalized === 'bgra8unorm') {
    if (deviceHasFeature(device, 'bgra8unorm-storage')) {
      return {
        storageFormat: 'bgra8unorm',
        supportsStorageBinding: true,
        needsLinearSiblingView: false,
        reason: null,
      };
    }
    return {
      storageFormat: null,
      supportsStorageBinding: false,
      needsLinearSiblingView: false,
      reason: 'missing-bgra8unorm-storage',
    };
  }
  if (TIER1_STORAGE_TEXTURE_FORMATS.has(normalized)) {
    if (deviceHasFeature(device, 'texture-formats-tier1')) {
      return {
        storageFormat: normalized,
        supportsStorageBinding: true,
        needsLinearSiblingView: false,
        reason: null,
      };
    }
    return {
      storageFormat: null,
      supportsStorageBinding: false,
      needsLinearSiblingView: false,
      reason: 'missing-texture-formats-tier1',
    };
  }
  if (normalized === 'rgba8unorm-srgb') {
    return {
      storageFormat: 'rgba8unorm',
      supportsStorageBinding: true,
      needsLinearSiblingView: true,
      reason: 'srgb-requires-linear-view',
    };
  }
  if (normalized === 'bgra8unorm-srgb') {
    if (deviceHasFeature(device, 'bgra8unorm-storage')) {
      return {
        storageFormat: 'bgra8unorm',
        supportsStorageBinding: true,
        needsLinearSiblingView: true,
        reason: 'srgb-requires-linear-view',
      };
    }
    return {
      storageFormat: null,
      supportsStorageBinding: false,
      needsLinearSiblingView: false,
      reason: 'missing-bgra8unorm-storage',
    };
  }
  return {
    storageFormat: null,
    supportsStorageBinding: false,
    needsLinearSiblingView: false,
    reason: 'unsupported-storage-format',
  };
}

function isStorageBindingSupported(format, device, options) {
  const resolved = resolveStorageTextureFormat(format, device, options);
  return Boolean(resolved.supportsStorageBinding && !resolved.needsLinearSiblingView);
}

function isRenderAttachmentSupported(format, device) {
  if (typeof format !== 'string' || format.length === 0 || format === 'undefined') return false;
  if (isBcCompressedFormat(format) || NON_RENDERABLE_CORE_FORMATS.has(format)) return false;
  if (format === 'rg11b10ufloat') return deviceHasFeature(device, 'rg11b10ufloat-renderable');
  if (TIER1_RENDERABLE_FORMATS.has(format)) {
    return deviceHasFeature(device, 'texture-formats-tier1');
  }
  return true;
}

function isMultisampleSupported(format, device) {
  if (!isRenderAttachmentSupported(format, device)) return false;
  if (NON_MULTISAMPLEABLE_CORE_FORMATS.has(format)) return false;
  return true;
}

// Resolves the WebGPU GPUTextureSampleType compatible with `format` on `device`.
// Notably maps rgba32float / rg32float / r32float to 'unfilterable-float' when 'float32-filterable' is absent.
function resolveTextureSampleType(format, device) {
  if (typeof format !== 'string' || format.length === 0 || format === 'undefined') return 'float';
  if (isDepthTextureFormat(format)) return 'depth';
  if (format === 'stencil8' || format.endsWith('uint')) return 'uint';
  if (format.endsWith('sint')) return 'sint';
  if (isFloat32TextureFormat(format)) {
    return deviceHasFeature(device, 'float32-filterable') ? 'float' : 'unfilterable-float';
  }
  return 'float';
}

function computeMaxMipLevels(width, height, depthOrArrayLayers = 1, dimension = '2d') {
  if (dimension === '1d') return 1;
  const w = Math.max(1, Math.floor(Number(width) || 1));
  const h = Math.max(1, Math.floor(Number(height) || 1));
  const d = Math.max(1, Math.floor(Number(depthOrArrayLayers) || 1));
  const maxDim = dimension === '3d' ? Math.max(w, h, d) : Math.max(w, h);
  return Math.floor(Math.log2(maxDim)) + 1;
}

function getMaxComputeWorkgroupsPerDimension(deviceOrLimits) {
  const limits =
    deviceOrLimits && typeof deviceOrLimits === 'object'
      ? deviceOrLimits.limits && typeof deviceOrLimits.limits === 'object'
        ? deviceOrLimits.limits
        : deviceOrLimits
      : null;
  const raw = limits ? limits.maxComputeWorkgroupsPerDimension : undefined;
  if (typeof raw === 'number' && Number.isFinite(raw) && raw >= 1) {
    return Math.floor(raw);
  }
  return DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION;
}

// Validates and clamps compute dispatch workgroup counts against WebGPU limits.
// Zero or invalid dimensions are skipped; oversized dimensions are clamped to maxComputeWorkgroupsPerDimension.
function validateComputeDispatch(x, y, z, deviceOrLimits) {
  const maxDim = getMaxComputeWorkgroupsPerDimension(deviceOrLimits);
  const fx = typeof x === 'number' && Number.isFinite(x) ? Math.floor(x) : 0;
  const fy = typeof y === 'number' && Number.isFinite(y) ? Math.floor(y) : 0;
  const fz = typeof z === 'number' && Number.isFinite(z) ? Math.floor(z) : 0;
  if (fx <= 0 || fy <= 0 || fz <= 0) {
    return {
      skip: true,
      clamped: false,
      x: 0,
      y: 0,
      z: 0,
      maxWorkgroupsPerDimension: maxDim,
      reason: 'zero-dimension',
    };
  }
  if (fx > maxDim || fy > maxDim || fz > maxDim) {
    return {
      skip: false,
      clamped: true,
      x: Math.min(fx, maxDim),
      y: Math.min(fy, maxDim),
      z: Math.min(fz, maxDim),
      maxWorkgroupsPerDimension: maxDim,
      reason: 'exceeds-limit',
    };
  }
  return {
    skip: false,
    clamped: false,
    x: fx,
    y: fy,
    z: fz,
    maxWorkgroupsPerDimension: maxDim,
    reason: null,
  };
}

// Decodes an ABI v2 CreateTexture2 payload (opcode 35, 40-byte GpuCreateTexture2Payload).
function decodeCreateTexture2(view, offset = 0, payloadBytes) {
  const available = payloadBytes !== undefined ? payloadBytes : view.byteLength - offset;
  if (
    offset < 0 ||
    available < CREATE_TEXTURE2_BYTES ||
    offset + CREATE_TEXTURE2_BYTES > view.byteLength
  ) {
    return { id: 0, error: `CreateTexture2: payload ${available} < ${CREATE_TEXTURE2_BYTES} bytes` };
  }
  const id = view.getUint32(offset, true);
  const rawWidth = view.getUint32(offset + 4, true);
  const rawHeight = view.getUint32(offset + 8, true);
  const rawDepth = view.getUint32(offset + 12, true);
  const rawMipLevels = view.getUint32(offset + 16, true);
  const rawSampleCount = view.getUint32(offset + 20, true);
  const usage = view.getUint32(offset + 24, true);
  const dimensionCode = view.getUint8(offset + 28);
  const formatCode = view.getUint8(offset + 29);
  const viewFormatCount = view.getUint8(offset + 30);

  const format = nameOf(TEXTURE_FORMAT_NAMES, formatCode);
  if (formatCode === 0 || !format || format === 'undefined') {
    return { id, error: `CreateTexture2 ${id}: invalid texture format code ${formatCode}` };
  }
  const dimension = nameOf(TEXTURE_DIMENSION_NAMES, dimensionCode);
  if (!dimension) {
    return { id, error: `CreateTexture2 ${id}: invalid dimension ${dimensionCode}` };
  }
  if (viewFormatCount > MAX_TEXTURE_VIEW_FORMATS) {
    return {
      id,
      error: `CreateTexture2 ${id}: view_format_count ${viewFormatCount} > ${MAX_TEXTURE_VIEW_FORMATS}`,
    };
  }
  const viewFormats = [];
  for (let i = 0; i < viewFormatCount; i += 1) {
    const code = view.getUint8(offset + 32 + i);
    const vf = nameOf(TEXTURE_FORMAT_NAMES, code);
    if (code === 0 || !vf || vf === 'undefined') {
      return { id, error: `CreateTexture2 ${id}: invalid view format code ${code}` };
    }
    viewFormats.push(vf);
  }

  const width = Math.max(1, rawWidth);
  const height = dimension === '1d' ? 1 : Math.max(1, rawHeight);
  const depthOrArrayLayers = dimension === '1d' ? 1 : Math.max(1, rawDepth);
  const maxMips = computeMaxMipLevels(width, height, depthOrArrayLayers, dimension);
  const mipLevelCount =
    Number.isFinite(rawMipLevels) && rawMipLevels >= 1
      ? Math.min(maxMips, Math.floor(rawMipLevels))
      : 1;
  const geometryAllowsMsaa =
    dimension === '2d' && mipLevelCount <= 1 && depthOrArrayLayers === 1;
  const sampleCount = rawSampleCount >= 4 && geometryAllowsMsaa ? 4 : 1;

  const descriptor = {
    size: { width, height, depthOrArrayLayers },
    mipLevelCount,
    sampleCount,
    dimension,
    format,
    usage,
  };
  if (viewFormats.length > 0) descriptor.viewFormats = viewFormats;
  return { id, descriptor };
}

// Sanitizes a GPUTextureDescriptor against WebGPU format & device capability constraints
// (stripping illegal STORAGE_BINDING / RENDER_ATTACHMENT usages and falling back cleanly on
// unsupported BC compressed or optional depth formats).
function sanitizeTextureDescriptor(rawDescriptor, device, options) {
  const originalFormat = rawDescriptor.format;
  const wasBc = isBcCompressedFormat(originalFormat);
  let effectiveFormat = originalFormat;
  let usage = rawDescriptor.usage >>> 0;
  let sampleCount = rawDescriptor.sampleCount || 1;
  let bcFallback = false;
  let formatFallback = false;
  const strippedUsages = [];
  const warnings = [];

  const is3d = (rawDescriptor.dimension || '2d') === '3d';
  if (wasBc && !deviceHasFeature(device, 'texture-compression-bc')) {
    bcFallback = true;
    formatFallback = true;
    effectiveFormat = isSrgbTextureFormat(originalFormat) ? 'rgba8unorm-srgb' : 'rgba8unorm';
    warnings.push({
      key: `bc-unsupported:${originalFormat}`,
      message: `texture format ${originalFormat} unavailable without texture-compression-bc; falling back to ${effectiveFormat}`,
    });
  } else if (wasBc && is3d && !deviceHasFeature(device, 'texture-compression-bc-sliced-3d')) {
    // W107 (#695): WebGPU rejects 3D BC-compressed textures without texture-compression-bc-sliced-3d.
    bcFallback = true;
    formatFallback = true;
    effectiveFormat = isSrgbTextureFormat(originalFormat) ? 'rgba8unorm-srgb' : 'rgba8unorm';
    warnings.push({
      key: `bc-3d-unsupported:${originalFormat}`,
      message: `3d texture format ${originalFormat} unavailable without texture-compression-bc-sliced-3d; falling back to ${effectiveFormat}`,
    });
  } else if (
    originalFormat === 'depth32float-stencil8' &&
    !deviceHasFeature(device, 'depth32float-stencil8')
  ) {
    formatFallback = true;
    effectiveFormat = 'depth24plus-stencil8';
    warnings.push({
      key: 'depth32float-stencil8-unsupported',
      message: 'texture format depth32float-stencil8 unavailable; falling back to depth24plus-stencil8',
    });
  }

  if (
    (usage & GPU_TEXTURE_USAGE_STORAGE_BINDING) !== 0 &&
    (wasBc || !isStorageBindingSupported(effectiveFormat, device, options))
  ) {
    usage &= ~GPU_TEXTURE_USAGE_STORAGE_BINDING;
    strippedUsages.push('STORAGE_BINDING');
    warnings.push({
      key: `strip-storage:${originalFormat}`,
      message: `stripped unsupported STORAGE_BINDING usage for texture format ${originalFormat}`,
    });
  }

  if (
    (usage & GPU_TEXTURE_USAGE_RENDER_ATTACHMENT) !== 0 &&
    (wasBc || !isRenderAttachmentSupported(effectiveFormat, device))
  ) {
    usage &= ~GPU_TEXTURE_USAGE_RENDER_ATTACHMENT;
    strippedUsages.push('RENDER_ATTACHMENT');
    warnings.push({
      key: `strip-render-attachment:${originalFormat}`,
      message: `stripped unsupported RENDER_ATTACHMENT usage for texture format ${originalFormat}`,
    });
  }

  if (usage === 0) {
    usage = GPU_TEXTURE_USAGE_TEXTURE_BINDING;
  }

  if (sampleCount > 1 && (wasBc || !isMultisampleSupported(effectiveFormat, device))) {
    sampleCount = 1;
  }

  const descriptor = {
    size: { ...rawDescriptor.size },
    mipLevelCount: rawDescriptor.mipLevelCount || 1,
    sampleCount,
    dimension: rawDescriptor.dimension || '2d',
    format: effectiveFormat,
    usage,
  };

  if (Array.isArray(rawDescriptor.viewFormats) && rawDescriptor.viewFormats.length > 0) {
    const cleanViewFormats = [];
    for (let i = 0; i < rawDescriptor.viewFormats.length; i += 1) {
      let vf = rawDescriptor.viewFormats[i];
      if (bcFallback && isBcCompressedFormat(vf)) {
        vf = isSrgbTextureFormat(vf) ? 'rgba8unorm-srgb' : 'rgba8unorm';
      } else if (
        vf === 'depth32float-stencil8' &&
        effectiveFormat === 'depth24plus-stencil8'
      ) {
        vf = 'depth24plus-stencil8';
      }
      if (!cleanViewFormats.includes(vf)) {
        cleanViewFormats.push(vf);
      }
    }
    if (cleanViewFormats.length > 0) {
      descriptor.viewFormats = cleanViewFormats;
    }
  }

  return {
    descriptor,
    originalFormat,
    effectiveFormat,
    strippedUsages,
    bcFallback,
    formatFallback,
    warnings,
  };
}

// Strips WGSL line (//) and nestable block (/* ... */) comments so commented-out entry points are ignored.
function stripWgslComments(code) {
  if (typeof code !== 'string' || code.length === 0) return '';
  let out = '';
  let blockDepth = 0;
  for (let i = 0; i < code.length; i += 1) {
    const ch = code[i];
    const next = i + 1 < code.length ? code[i + 1] : '';
    if (blockDepth > 0) {
      if (ch === '/' && next === '*') {
        blockDepth += 1;
        i += 1;
      } else if (ch === '*' && next === '/') {
        blockDepth -= 1;
        i += 1;
        out += ' ';
      } else if (ch === '\n') {
        out += '\n';
      }
      continue;
    }
    if (ch === '/' && next === '/') {
      i += 2;
      while (i < code.length && code[i] !== '\n') i += 1;
      if (i < code.length && code[i] === '\n') out += '\n';
      continue;
    }
    if (ch === '/' && next === '*') {
      blockDepth = 1;
      i += 1;
      continue;
    }
    out += ch;
  }
  return out;
}

function findWgslStageEntryPoints(code, stage) {
  if (typeof code !== 'string' || code.length === 0) return [];
  const stageAttr =
    stage === 'vertex' || stage === 'fragment' || stage === 'compute' ? stage : null;
  if (!stageAttr) return [];
  const cleaned = stripWgslComments(code);
  const re = new RegExp(
    `@${stageAttr}\\b(?:\\s*@[A-Za-z_][A-Za-z0-9_]*(?:\\s*\\([^)]*\\))?)*\\s*fn\\s+([A-Za-z_][A-Za-z0-9_]*)`,
    'g'
  );
  const found = [];
  let match;
  while ((match = re.exec(cleaned)) !== null) {
    if (!found.includes(match[1])) {
      found.push(match[1]);
    }
  }
  return found;
}

function normalizeEntryPointName(raw) {
  if (typeof raw !== 'string') return null;
  const trimmed = raw.trim();
  if (!trimmed || !VALID_WGSL_IDENT_RE.test(trimmed)) return null;
  return trimmed;
}

function getShaderSourceText(shaderEntry) {
  if (!shaderEntry || typeof shaderEntry !== 'object') return null;
  if (typeof shaderEntry.code === 'string' && shaderEntry.code.length > 0) return shaderEntry.code;
  if (typeof shaderEntry.wgsl === 'string' && shaderEntry.wgsl.length > 0) return shaderEntry.wgsl;
  if (
    shaderEntry.module &&
    typeof shaderEntry.module.code === 'string' &&
    shaderEntry.module.code.length > 0
  ) {
    return shaderEntry.module.code;
  }
  return null;
}

function wgslDeclaresGroup0(code) {
  return typeof code === 'string' && code.length > 0 && WGSL_GROUP0_RE.test(code);
}

const WGSL_GROUP0_VAR_RE =
  /(?:@group\s*\(\s*0[ui]?\s*\)\s*@binding\s*\(\s*(\d+)[ui]?\s*\)|@binding\s*\(\s*(\d+)[ui]?\s*\)\s*@group\s*\(\s*0[ui]?\s*\))\s*var(?:\s*<\s*([^>]+)\s*>)?\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([^;]+);/g;

function classifyWgslTextureDim(typePrefix) {
  if (typePrefix.endsWith('_2d_array')) return '2d-array';
  if (typePrefix.endsWith('_cube_array')) return 'cube-array';
  if (typePrefix.endsWith('_cube')) return 'cube';
  if (typePrefix.endsWith('_3d')) return '3d';
  if (typePrefix.endsWith('_1d')) return '1d';
  if (typePrefix.endsWith('_2d') || typePrefix === 'texture_external') return '2d';
  return '2d';
}

function extractWgslFunctionBodies(cleaned) {
  const fnRe = /\bfn\s+[A-Za-z_][A-Za-z0-9_]*\s*\(/g;
  let bodies = '';
  let foundFn = false;
  let match;
  while ((match = fnRe.exec(cleaned)) !== null) {
    let i = match.index + match[0].length;
    let parenDepth = 1;
    while (i < cleaned.length && parenDepth > 0) {
      if (cleaned[i] === '(') parenDepth += 1;
      else if (cleaned[i] === ')') parenDepth -= 1;
      i += 1;
    }
    while (i < cleaned.length && cleaned[i] !== '{') {
      if (cleaned[i] === ';') break;
      i += 1;
    }
    if (i >= cleaned.length || cleaned[i] !== '{') continue;
    foundFn = true;
    const bodyStart = i + 1;
    let braceDepth = 1;
    i += 1;
    while (i < cleaned.length && braceDepth > 0) {
      if (cleaned[i] === '{') braceDepth += 1;
      else if (cleaned[i] === '}') braceDepth -= 1;
      i += 1;
    }
    bodies += cleaned.slice(bodyStart, Math.max(bodyStart, i - 1)) + '\n';
    fnRe.lastIndex = i;
  }
  return { hasFunction: foundFn, bodies };
}

// Parses statically referenced @group(0) @binding(N) (or @binding(N) @group(0)) var declarations
// in WGSL source, returning entries sorted and deduplicated by binding index.
function parseWgslGroup0Bindings(wgslSource) {
  if (typeof wgslSource !== 'string' || wgslSource.length === 0) return [];
  const cleaned = stripWgslComments(wgslSource);
  if (cleaned.length === 0) return [];

  const matches = [];
  WGSL_GROUP0_VAR_RE.lastIndex = 0;
  let match;
  while ((match = WGSL_GROUP0_VAR_RE.exec(cleaned)) !== null) {
    const bindingStr = match[1] !== undefined ? match[1] : match[2];
    const binding = parseInt(bindingStr, 10);
    if (!Number.isInteger(binding) || binding < 0) continue;
    matches.push({
      binding,
      addrRaw: match[3] ? match[3].trim() : '',
      varName: match[4],
      typeRaw: match[5].trim(),
    });
  }
  if (matches.length === 0) return [];

  const { hasFunction, bodies: usageSource } = extractWgslFunctionBodies(cleaned);

  const byBinding = new Map();
  for (let i = 0; i < matches.length; i += 1) {
    const m = matches[i];
    if (byBinding.has(m.binding)) continue;
    if (hasFunction) {
      const refRe = new RegExp(`(?:^|[^A-Za-z0-9_.])${m.varName}(?![A-Za-z0-9_])`);
      if (!refRe.test(usageSource)) continue;
    }

    let entry = null;
    if (m.addrRaw.length > 0) {
      const parts = m.addrRaw.split(',').map((s) => s.trim().toLowerCase());
      if (parts[0] === 'uniform') {
        entry = { binding: m.binding, kind: 'uniform' };
      } else if (parts[0] === 'storage') {
        entry = {
          binding: m.binding,
          kind: parts[1] === 'read' ? 'read-only-storage' : 'storage',
        };
      }
    } else if (/^sampler_comparison\b/.test(m.typeRaw)) {
      entry = { binding: m.binding, kind: 'sampler', comparison: true };
    } else if (/^sampler\b/.test(m.typeRaw)) {
      entry = { binding: m.binding, kind: 'sampler' };
    } else if (/^texture_storage_/.test(m.typeRaw)) {
      const prefixMatch = /^(texture_storage_[A-Za-z0-9_]+)/.exec(m.typeRaw);
      const prefix = prefixMatch ? prefixMatch[1] : 'texture_storage_2d';
      const fmtMatch = /^texture_storage_[A-Za-z0-9_]+\s*<\s*([A-Za-z0-9_-]+)/.exec(m.typeRaw);
      entry = {
        binding: m.binding,
        kind: 'storage-texture',
        viewDimension: classifyWgslTextureDim(prefix),
      };
      if (fmtMatch && fmtMatch[1]) {
        entry.storageFormat = fmtMatch[1];
      }
    } else if (/^texture_/.test(m.typeRaw)) {
      const prefixMatch = /^(texture_[A-Za-z0-9_]+)/.exec(m.typeRaw);
      const prefix = prefixMatch ? prefixMatch[1] : 'texture_2d';
      entry = {
        binding: m.binding,
        kind: 'texture',
        viewDimension: classifyWgslTextureDim(prefix),
      };
      if (prefix.includes('_multisampled_')) {
        entry.multisampled = true;
      }
      if (prefix.startsWith('texture_depth_')) {
        entry.sampleType = 'depth';
      } else {
        const genericMatch = /<\s*(u32|i32|f32)\s*>/.exec(m.typeRaw);
        if (genericMatch) {
          if (genericMatch[1] === 'u32') entry.sampleType = 'uint';
          else if (genericMatch[1] === 'i32') entry.sampleType = 'sint';
        }
      }
    }

    if (entry) {
      byBinding.set(m.binding, entry);
    }
  }

  return Array.from(byBinding.values()).sort((a, b) => a.binding - b.binding);
}

function mergeWgslGroup0Bindings(vsBindings, fsBindings) {
  const byBinding = new Map();
  if (Array.isArray(vsBindings)) {
    for (let i = 0; i < vsBindings.length; i += 1) {
      byBinding.set(vsBindings[i].binding, vsBindings[i]);
    }
  }
  if (Array.isArray(fsBindings)) {
    for (let i = 0; i < fsBindings.length; i += 1) {
      const fb = fsBindings[i];
      const existing = byBinding.get(fb.binding);
      if (!existing) {
        byBinding.set(fb.binding, fb);
      } else if (existing.kind === 'read-only-storage' && fb.kind === 'storage') {
        byBinding.set(fb.binding, fb);
      }
    }
  }
  return Array.from(byBinding.values()).sort((a, b) => a.binding - b.binding);
}

// Checks whether an explicit @group(0) BindGroupLayout's entries satisfy every binding declared
// and statically referenced by a WGSL shader (or VS+FS pair).
function isExplicitLayoutCompatibleWithWgsl(group0Entries, wgslBindings) {
  if (!Array.isArray(wgslBindings) || wgslBindings.length === 0) return true;
  if (!Array.isArray(group0Entries)) return true;

  for (let i = 0; i < wgslBindings.length; i += 1) {
    const b = wgslBindings[i];
    const e = group0Entries.find((x) => x && x.binding === b.binding);
    if (!e) return false;
    switch (b.kind) {
      case 'uniform':
        if (!(e.buffer && (!e.buffer.type || e.buffer.type === 'uniform'))) return false;
        break;
      case 'read-only-storage':
        if (!(e.buffer && (e.buffer.type === 'read-only-storage' || e.buffer.type === 'storage'))) {
          return false;
        }
        break;
      case 'storage':
        if (!(e.buffer && e.buffer.type === 'storage')) return false;
        break;
      case 'texture':
        if (!e.texture) return false;
        break;
      case 'storage-texture':
        if (!e.storageTexture) return false;
        break;
      case 'sampler':
        if (!e.sampler) return false;
        break;
      default:
        return false;
    }
  }
  return true;
}

// W101: normalize a setBindGroup dynamicOffsets argument (Array / typed array / null) into a plain
// array of uint32 offsets; anything else yields [].
function normalizeDynamicOffsets(raw) {
  if (!raw || !(Array.isArray(raw) || ArrayBuffer.isView(raw))) return [];
  const out = [];
  for (let i = 0; i < raw.length; i += 1) {
    const v = Number(raw[i]);
    out.push(Number.isFinite(v) && v > 0 ? Math.floor(v) >>> 0 : 0);
  }
  return out;
}

// W101: WebGPU orders setBindGroup dynamicOffsets by ascending layout binding number over the
// entries with buffer.hasDynamicOffset. Returns Map(binding -> dynamicOffsetIndex).
function dynamicOffsetIndicesForLayoutEntries(entries) {
  const indices = new Map();
  if (!Array.isArray(entries)) return indices;
  const dynBindings = entries
    .filter((e) => e && e.buffer && e.buffer.hasDynamicOffset)
    .map((e) => e.binding)
    .sort((a, b) => a - b);
  for (let i = 0; i < dynBindings.length; i += 1) indices.set(dynBindings[i], i);
  return indices;
}

function synthLayoutEntryForWgslBinding(b) {
  if (!b) return null;
  if (b.kind === 'uniform') {
    return {
      binding: b.binding,
      buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: 4096 },
    };
  }
  if (b.kind === 'read-only-storage') {
    return {
      binding: b.binding,
      buffer: { type: 'read-only-storage', hasDynamicOffset: false, minBindingSize: 4096 },
    };
  }
  if (b.kind === 'storage') {
    return {
      binding: b.binding,
      buffer: { type: 'storage', hasDynamicOffset: false, minBindingSize: 4096 },
    };
  }
  if (b.kind === 'sampler') {
    return {
      binding: b.binding,
      sampler: { type: b.comparison ? 'comparison' : 'filtering' },
    };
  }
  if (b.kind === 'storage-texture') {
    return {
      binding: b.binding,
      storageTexture: {
        access: 'write-only',
        format: b.storageFormat || 'rgba8unorm',
        viewDimension: b.viewDimension || '2d',
      },
    };
  }
  return {
    binding: b.binding,
    texture: {
      sampleType: b.sampleType || 'float',
      viewDimension: b.viewDimension || '2d',
      multisampled: Boolean(b.multisampled),
    },
  };
}

// Resolves the WebGPU shader stage entry point: defaults to vs_main/fs_main/cs_main per
// docs/gpu_commands.md, while honoring explicit entryPoint metadata and WGSL source declarations
// (such as legacy `@fragment fn ps_main` from older packs).
function resolveShaderEntryPoint(shaderEntry, stage) {
  const defaultEntryPoint = DEFAULT_ENTRY_POINTS[stage] || 'vs_main';
  const explicit =
    normalizeEntryPointName(shaderEntry && shaderEntry.entryPoint) ||
    normalizeEntryPointName(shaderEntry && shaderEntry.entry_point);
  const sourceText = getShaderSourceText(shaderEntry);
  const declared = sourceText ? findWgslStageEntryPoints(sourceText, stage) : [];

  let entryPoint = defaultEntryPoint;
  let reason = 'default';

  if (declared.length > 0) {
    if (explicit && declared.includes(explicit)) {
      entryPoint = explicit;
      reason = explicit === defaultEntryPoint ? 'default' : 'metadata+source';
    } else if (declared.includes(defaultEntryPoint)) {
      entryPoint = defaultEntryPoint;
      reason = 'default';
    } else {
      entryPoint = declared[0];
      reason = explicit ? 'source-override' : 'source-inspect';
    }
  } else if (explicit) {
    entryPoint = explicit;
    reason = explicit === defaultEntryPoint ? 'default' : 'metadata';
  }

  return {
    entryPoint,
    defaultEntryPoint,
    remapped: entryPoint !== defaultEntryPoint,
    reason,
  };
}

// Validates decoded GPURenderPipelineDescriptor.vertex.buffers against WebGPU layout rules.
function validateVertexBufferLayouts(buffers) {
  if (!Array.isArray(buffers)) {
    return { valid: false, error: 'vertex buffers must be an array' };
  }
  const usedLocations = new Set();
  for (let i = 0; i < buffers.length; i += 1) {
    const slot = buffers[i];
    if (slot === null) continue;
    if (!slot || typeof slot !== 'object') {
      return { valid: false, error: `vertex buffer slot ${i}: invalid layout object` };
    }
    if (slot.stepMode !== 'vertex' && slot.stepMode !== 'instance') {
      return { valid: false, error: `vertex buffer slot ${i}: invalid stepMode ${String(slot.stepMode)}` };
    }
    if (!Number.isInteger(slot.arrayStride) || slot.arrayStride < 0 || slot.arrayStride % 4 !== 0) {
      return { valid: false, error: `vertex buffer slot ${i}: invalid arrayStride ${slot.arrayStride}` };
    }
    if (!Array.isArray(slot.attributes)) {
      return { valid: false, error: `vertex buffer slot ${i}: attributes must be an array` };
    }
    for (let j = 0; j < slot.attributes.length; j += 1) {
      const attr = slot.attributes[j];
      if (!attr || typeof attr !== 'object') {
        return { valid: false, error: `vertex buffer slot ${i} attr ${j}: invalid attribute` };
      }
      const byteSize = VERTEX_FORMAT_BYTE_SIZES[attr.format];
      if (!byteSize) {
        return { valid: false, error: `vertex buffer slot ${i} attr ${j}: invalid format ${String(attr.format)}` };
      }
      if (!Number.isInteger(attr.offset) || attr.offset < 0) {
        return { valid: false, error: `vertex buffer slot ${i} attr ${j}: invalid offset ${attr.offset}` };
      }
      if (!Number.isInteger(attr.shaderLocation) || attr.shaderLocation < 0 || attr.shaderLocation >= 16) {
        return {
          valid: false,
          error: `vertex buffer slot ${i} attr ${j}: invalid shaderLocation ${attr.shaderLocation}`,
        };
      }
      if (usedLocations.has(attr.shaderLocation)) {
        return {
          valid: false,
          error: `vertex buffer slot ${i} attr ${j}: duplicate shaderLocation ${attr.shaderLocation}`,
        };
      }
      usedLocations.add(attr.shaderLocation);
      if (slot.arrayStride > 0 && attr.offset + byteSize > slot.arrayStride) {
        return {
          valid: false,
          error: `vertex buffer slot ${i} attr ${j}: offset ${attr.offset} + ${byteSize} exceeds arrayStride ${slot.arrayStride}`,
        };
      }
    }
  }
  return { valid: true };
}

function nameOf(table, code) {
  return code < table.length ? table[code] : undefined;
}

function finiteOr(value, fallback) {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function validTargetSize(width, height) {
  if (
    typeof width === 'number' &&
    typeof height === 'number' &&
    Number.isFinite(width) &&
    Number.isFinite(height) &&
    width > 0 &&
    height > 0
  ) {
    return { width, height };
  }
  return null;
}

// Clamps a guest viewport to the active render target bounds and WebGPU [0, 1] depth range.
function clampViewportToTarget(viewport, targetSize) {
  const rawX = finiteOr(viewport && viewport.x, 0);
  const rawY = finiteOr(viewport && viewport.y, 0);
  const rawW = finiteOr(viewport && viewport.width, 0);
  const rawH = finiteOr(viewport && viewport.height, 0);

  let minDepth = Math.min(1, Math.max(0, finiteOr(viewport && viewport.minDepth, 0)));
  let maxDepth = Math.min(1, Math.max(0, finiteOr(viewport && viewport.maxDepth, 1)));
  if (minDepth > maxDepth) {
    const tmp = minDepth;
    minDepth = maxDepth;
    maxDepth = tmp;
  }
  minDepth = minDepth || 0;
  maxDepth = maxDepth || 0;

  const target = targetSize ? validTargetSize(targetSize.width, targetSize.height) : null;
  if (!target) {
    const safeX = Number.isFinite(Math.fround(rawX)) ? Math.max(0, rawX) || 0 : 0;
    const safeY = Number.isFinite(Math.fround(rawY)) ? Math.max(0, rawY) || 0 : 0;
    const safeW = Number.isFinite(Math.fround(rawW)) ? Math.max(0, rawW) || 0 : 0;
    const safeH = Number.isFinite(Math.fround(rawH)) ? Math.max(0, rawH) || 0 : 0;
    return {
      x: safeX,
      y: safeY,
      width: safeW,
      height: safeH,
      minDepth,
      maxDepth,
    };
  }

  const targetWidth = target.width;
  const targetHeight = target.height;
  if (
    rawX >= 0 &&
    rawY >= 0 &&
    rawW >= 0 &&
    rawH >= 0 &&
    rawX + rawW <= targetWidth &&
    rawY + rawH <= targetHeight &&
    Math.fround(rawX) + Math.fround(rawW) <= targetWidth &&
    Math.fround(rawY) + Math.fround(rawH) <= targetHeight
  ) {
    return {
      x: rawX || 0,
      y: rawY || 0,
      width: rawW || 0,
      height: rawH || 0,
      minDepth,
      maxDepth,
    };
  }

  let x = Math.min(Math.max(0, rawX), targetWidth) || 0;
  if (Math.fround(x) > targetWidth) {
    x = Math.max(0, Math.floor(targetWidth)) || 0;
  }
  let y = Math.min(Math.max(0, rawY), targetHeight) || 0;
  if (Math.fround(y) > targetHeight) {
    y = Math.max(0, Math.floor(targetHeight)) || 0;
  }
  let width = Math.min(Math.max(0, rawW), targetWidth - x) || 0;
  let height = Math.min(Math.max(0, rawH), targetHeight - y) || 0;
  if (x + width > targetWidth || Math.fround(x) + Math.fround(width) > targetWidth) {
    width = Math.max(0, Math.floor(targetWidth - Math.fround(x))) || 0;
  }
  if (y + height > targetHeight || Math.fround(y) + Math.fround(height) > targetHeight) {
    height = Math.max(0, Math.floor(targetHeight - Math.fround(y))) || 0;
  }
  return { x, y, width, height, minDepth, maxDepth };
}

function isViewportClamped(raw, clamped) {
  if (!raw || typeof raw !== 'object') return true;
  return (
    !Number.isFinite(raw.x) ||
    !Number.isFinite(raw.y) ||
    !Number.isFinite(raw.width) ||
    !Number.isFinite(raw.height) ||
    !Number.isFinite(raw.minDepth) ||
    !Number.isFinite(raw.maxDepth) ||
    clamped.x !== raw.x ||
    clamped.y !== raw.y ||
    clamped.width !== raw.width ||
    clamped.height !== raw.height ||
    clamped.minDepth !== raw.minDepth ||
    clamped.maxDepth !== raw.maxDepth
  );
}

const MAX_SCISSOR_UINT32 = 0xffffffff;

// Clamps a guest integer scissor rect to [0..targetWidth] x [0..targetHeight].
function clampScissorToTarget(scissor, targetSize) {
  const rawX = Math.min(MAX_SCISSOR_UINT32, Math.max(0, Math.floor(finiteOr(scissor && scissor.x, 0)))) || 0;
  const rawY = Math.min(MAX_SCISSOR_UINT32, Math.max(0, Math.floor(finiteOr(scissor && scissor.y, 0)))) || 0;
  const rawW = Math.min(MAX_SCISSOR_UINT32, Math.max(0, Math.floor(finiteOr(scissor && scissor.width, 0)))) || 0;
  const rawH = Math.min(MAX_SCISSOR_UINT32, Math.max(0, Math.floor(finiteOr(scissor && scissor.height, 0)))) || 0;

  const target = targetSize ? validTargetSize(targetSize.width, targetSize.height) : null;
  if (!target) {
    return { x: rawX, y: rawY, width: rawW, height: rawH };
  }

  const targetWidth = Math.floor(target.width);
  const targetHeight = Math.floor(target.height);
  if (targetWidth <= 0 || targetHeight <= 0) {
    return { x: rawX, y: rawY, width: rawW, height: rawH };
  }

  if (rawX + rawW <= targetWidth && rawY + rawH <= targetHeight) {
    return { x: rawX, y: rawY, width: rawW, height: rawH };
  }

  const x = Math.min(rawX, targetWidth) || 0;
  const y = Math.min(rawY, targetHeight) || 0;
  const width = Math.min(rawW, targetWidth - x) || 0;
  const height = Math.min(rawH, targetHeight - y) || 0;
  return { x, y, width, height };
}

function isScissorClamped(raw, clamped) {
  if (!raw || typeof raw !== 'object') return true;
  return (
    !Number.isFinite(raw.x) ||
    !Number.isFinite(raw.y) ||
    !Number.isFinite(raw.width) ||
    !Number.isFinite(raw.height) ||
    clamped.x !== raw.x ||
    clamped.y !== raw.y ||
    clamped.width !== raw.width ||
    clamped.height !== raw.height
  );
}

function resolveAttachmentViewSize(view, explicitSize) {
  if (explicitSize) {
    const valid = validTargetSize(explicitSize.width, explicitSize.height);
    if (valid) return valid;
  }
  if (!view || typeof view !== 'object') return null;
  const fromView = validTargetSize(view.width, view.height);
  if (fromView) return fromView;
  if (view.texture && typeof view.texture === 'object') {
    const fromTex = validTargetSize(view.texture.width, view.texture.height);
    if (fromTex) return fromTex;
  }
  return null;
}

// W266: normalizes a GPURenderPassDescriptor so all attachments share a consistent (width, height),
// dropping mismatched secondary color / depthStencil / resolveTarget attachments and clamping
// depthClearValue / readOnly invariants before WebGPU sees the pass descriptor.
function sanitizeRenderPassAttachments(descriptor, attachmentSizes = null) {
  if (!descriptor || typeof descriptor !== 'object') {
    return { descriptor, referenceSize: null, sanitizedCount: 0, warnings: [] };
  }
  const colorSizesInput =
    attachmentSizes && Array.isArray(attachmentSizes.colors) ? attachmentSizes.colors : [];
  const rawColors = Array.isArray(descriptor.colorAttachments) ? descriptor.colorAttachments : [];
  const colorSizes = rawColors.map((att, i) =>
    att ? resolveAttachmentViewSize(att.view, colorSizesInput[i] || null) : null
  );
  const rawDs = descriptor.depthStencilAttachment || null;
  const dsSize = rawDs
    ? resolveAttachmentViewSize(rawDs.view, attachmentSizes ? attachmentSizes.depthStencil : null)
    : null;

  let referenceSize = null;
  for (let i = 0; i < colorSizes.length; i += 1) {
    if (colorSizes[i]) {
      referenceSize = colorSizes[i];
      break;
    }
  }
  if (!referenceSize && dsSize) {
    referenceSize = dsSize;
  }

  let sanitizedCount = 0;
  const warnings = [];
  let colorsChanged = false;
  const outColors = rawColors.slice();

  for (let i = 0; i < outColors.length; i += 1) {
    const att = outColors[i];
    if (!att) continue;
    const sz = colorSizes[i];
    if (referenceSize && sz && (sz.width !== referenceSize.width || sz.height !== referenceSize.height)) {
      outColors[i] = null;
      colorsChanged = true;
      sanitizedCount += 1;
      warnings.push({
        key: `rp-color-size-mismatch:${i}:${sz.width}x${sz.height}:${referenceSize.width}x${referenceSize.height}`,
        message: `render pass: dropped colorAttachments[${i}] (${sz.width}x${sz.height}) mismatched with pass target (${referenceSize.width}x${referenceSize.height})`,
      });
      continue;
    }
    if (att.resolveTarget) {
      const resolveExplicit =
        attachmentSizes && Array.isArray(attachmentSizes.resolveTargets)
          ? attachmentSizes.resolveTargets[i] || null
          : null;
      const resolveSz = resolveAttachmentViewSize(att.resolveTarget, resolveExplicit);
      if (
        referenceSize &&
        resolveSz &&
        (resolveSz.width !== referenceSize.width || resolveSz.height !== referenceSize.height)
      ) {
        const cleanAtt = { ...att };
        delete cleanAtt.resolveTarget;
        outColors[i] = cleanAtt;
        colorsChanged = true;
        sanitizedCount += 1;
        warnings.push({
          key: `rp-resolve-size-mismatch:${i}:${resolveSz.width}x${resolveSz.height}:${referenceSize.width}x${referenceSize.height}`,
          message: `render pass: dropped colorAttachments[${i}].resolveTarget (${resolveSz.width}x${resolveSz.height}) mismatched with pass target (${referenceSize.width}x${referenceSize.height})`,
        });
      }
    }
  }

  let outDs = rawDs;
  let dsChanged = false;
  if (rawDs) {
    if (referenceSize && dsSize && (dsSize.width !== referenceSize.width || dsSize.height !== referenceSize.height)) {
      outDs = null;
      dsChanged = true;
      sanitizedCount += 1;
      warnings.push({
        key: `rp-depth-size-mismatch:${dsSize.width}x${dsSize.height}:${referenceSize.width}x${referenceSize.height}`,
        message: `render pass: dropped depthStencilAttachment (${dsSize.width}x${dsSize.height}) mismatched with pass target (${referenceSize.width}x${referenceSize.height})`,
      });
    } else {
      let cleanDs = null;
      const ensureDsClone = () => {
        if (!cleanDs) cleanDs = { ...rawDs };
        return cleanDs;
      };
      if (rawDs.depthReadOnly) {
        if (
          rawDs.depthLoadOp !== undefined ||
          rawDs.depthStoreOp !== undefined ||
          rawDs.depthClearValue !== undefined
        ) {
          const cloned = ensureDsClone();
          delete cloned.depthLoadOp;
          delete cloned.depthStoreOp;
          delete cloned.depthClearValue;
          dsChanged = true;
          sanitizedCount += 1;
          warnings.push({
            key: 'rp-depth-readonly-ops',
            message: 'render pass: stripped depthLoadOp/depthStoreOp/depthClearValue on depthReadOnly attachment',
          });
        }
      } else if (rawDs.depthLoadOp === 'clear') {
        const rawClear = rawDs.depthClearValue;
        const clampedClear = Math.min(1, Math.max(0, finiteOr(rawClear, 1))) || 0;
        if (!Number.isFinite(rawClear) || clampedClear !== rawClear) {
          const cloned = ensureDsClone();
          cloned.depthClearValue = clampedClear;
          dsChanged = true;
          sanitizedCount += 1;
          warnings.push({
            key: `rp-depth-clear-clamp:${rawClear}`,
            message: `render pass: clamped depthClearValue ${rawClear} to ${clampedClear}`,
          });
        }
      }
      if (rawDs.stencilReadOnly) {
        if (
          rawDs.stencilLoadOp !== undefined ||
          rawDs.stencilStoreOp !== undefined ||
          rawDs.stencilClearValue !== undefined
        ) {
          const cloned = ensureDsClone();
          delete cloned.stencilLoadOp;
          delete cloned.stencilStoreOp;
          delete cloned.stencilClearValue;
          dsChanged = true;
          sanitizedCount += 1;
          warnings.push({
            key: 'rp-stencil-readonly-ops',
            message: 'render pass: stripped stencilLoadOp/stencilStoreOp/stencilClearValue on stencilReadOnly attachment',
          });
        }
      }
      if (cleanDs) outDs = cleanDs;
    }
  }

  if (!colorsChanged && !dsChanged) {
    return { descriptor, referenceSize, sanitizedCount: 0, warnings };
  }
  const outDescriptor = { ...descriptor, colorAttachments: outColors };
  if (outDs) {
    outDescriptor.depthStencilAttachment = outDs;
  } else {
    delete outDescriptor.depthStencilAttachment;
  }
  return { descriptor: outDescriptor, referenceSize, sanitizedCount, warnings };
}

// ---- Decoders (pure; also used by the tests) ----

function decodeTextureViewDesc(view, offset) {
  const desc = {};
  const format = view.getUint8(offset);
  const dimension = view.getUint8(offset + 1);
  const aspect = view.getUint8(offset + 2);
  const baseMipLevel = view.getUint16(offset + 4, true);
  const mipLevelCount = view.getUint16(offset + 6, true);
  const baseArrayLayer = view.getUint32(offset + 8, true);
  const arrayLayerCount = view.getUint32(offset + 12, true);
  if (format !== 0) desc.format = nameOf(TEXTURE_FORMAT_NAMES, format);
  if (dimension !== 0) desc.dimension = nameOf(VIEW_DIMENSION_NAMES, dimension);
  if (aspect !== 0) desc.aspect = nameOf(TEXTURE_ASPECT_NAMES, aspect);
  if (baseMipLevel !== 0) desc.baseMipLevel = baseMipLevel;
  if (mipLevelCount !== 0) desc.mipLevelCount = mipLevelCount;
  if (baseArrayLayer !== 0) desc.baseArrayLayer = baseArrayLayer;
  if (arrayLayerCount !== 0) desc.arrayLayerCount = arrayLayerCount;
  return desc;
}

function decodeBindGroupLayoutEntry(view, offset) {
  const entry = {
    binding: view.getUint32(offset, true),
    visibility: view.getUint32(offset + 4, true) & GPU_SHADER_STAGE_ALL_MASK,
  };
  const type = view.getUint8(offset + 8);
  const subtype = view.getUint8(offset + 9);
  const dimension = view.getUint8(offset + 10);
  const flags = view.getUint8(offset + 11);
  const storageFormat = view.getUint8(offset + 12);
  const viewDimension = dimension !== 0 ? nameOf(VIEW_DIMENSION_NAMES, dimension) : '2d';
  if (type === BINDING_LAYOUT_TYPE.BUFFER) {
    const bufferType = nameOf(BUFFER_BINDING_TYPE_NAMES, subtype);
    if (!bufferType) return null;
    // WebGPU forbids read-write 'storage' buffers in the vertex stage (e.g., root-signature layouts
    // that tag all entries Vertex|Fragment). Strip the illegal VERTEX visibility bit cleanly.
    if (bufferType === 'storage' && (entry.visibility & GPU_SHADER_STAGE_VERTEX) !== 0) {
      entry.visibility &= ~GPU_SHADER_STAGE_VERTEX;
    }
    entry.buffer = {
      type: bufferType,
      hasDynamicOffset: (flags & 0x1) !== 0,
      minBindingSize: Number(view.getBigUint64(offset + 16, true)),
    };
  } else if (type === BINDING_LAYOUT_TYPE.SAMPLER) {
    const samplerType = nameOf(SAMPLER_BINDING_TYPE_NAMES, subtype);
    if (!samplerType) return null;
    entry.sampler = { type: samplerType };
  } else if (type === BINDING_LAYOUT_TYPE.TEXTURE) {
    const sampleType = nameOf(TEXTURE_SAMPLE_TYPE_NAMES, subtype);
    if (!sampleType || !viewDimension || viewDimension === 'undefined') return null;
    entry.texture = {
      sampleType,
      viewDimension,
      multisampled: (flags & 0x2) !== 0,
    };
  } else if (type === BINDING_LAYOUT_TYPE.STORAGE_TEXTURE) {
    const access = nameOf(STORAGE_TEXTURE_ACCESS_NAMES, subtype);
    const format = storageFormat !== 0 ? nameOf(TEXTURE_FORMAT_NAMES, storageFormat) : undefined;
    if (!access || !format || format === 'undefined' || !viewDimension || viewDimension === 'undefined') {
      return null;
    }
    if ((access === 'write-only' || access === 'read-write') && (entry.visibility & GPU_SHADER_STAGE_VERTEX) !== 0) {
      entry.visibility &= ~GPU_SHADER_STAGE_VERTEX;
    }
    entry.storageTexture = {
      access,
      format,
      viewDimension,
    };
  } else {
    return null;
  }
  return entry;
}

function decodeBlendComponent(view, offset) {
  return {
    operation: nameOf(BLEND_OPERATION_NAMES, view.getUint8(offset)) || 'add',
    srcFactor: nameOf(BLEND_FACTOR_NAMES, view.getUint8(offset + 1)) || 'one',
    dstFactor: nameOf(BLEND_FACTOR_NAMES, view.getUint8(offset + 2)) || 'zero',
  };
}

function decodeStencilFace(view, offset) {
  const compareCode = view.getUint8(offset);
  const compare = compareCode !== 0 ? nameOf(COMPARE_FUNCTION_NAMES, compareCode) : undefined;
  return {
    compare: compare && compare !== 'undefined' ? compare : 'always',
    failOp: nameOf(STENCIL_OPERATION_NAMES, view.getUint8(offset + 1)) || 'keep',
    depthFailOp: nameOf(STENCIL_OPERATION_NAMES, view.getUint8(offset + 2)) || 'keep',
    passOp: nameOf(STENCIL_OPERATION_NAMES, view.getUint8(offset + 3)) || 'keep',
  };
}

function f32FromBits(bits) {
  const scratch = new DataView(new ArrayBuffer(4));
  scratch.setUint32(0, bits, true);
  return scratch.getFloat32(0, true);
}

// Decodes a GpuRenderPipelineKey at `base` into the GPURenderPipelineDescriptor parts it determines.
// Returns { primitive, multisample, depthStencil?, targets, buffers, colorTargetCount }.
function decodeRenderPipelineKey(view, base) {
  const colorTargetCount = Math.min(view.getUint32(base + 20, true), 8);
  const targets = [];
  for (let i = 0; i < colorTargetCount; i++) {
    const o = base + 24 + 12 * i;
    const format = view.getUint8(o);
    if (format === 0) {
      targets.push(null);
      continue;
    }
    const targetFormat = nameOf(TEXTURE_FORMAT_NAMES, format);
    const target = { format: targetFormat, writeMask: view.getUint8(o + 2) & 0x0f };
    if (view.getUint8(o + 1) !== 0) {
      target.blend = { color: decodeBlendComponent(view, o + 4), alpha: decodeBlendComponent(view, o + 8) };
    }
    targets.push(target);
  }

  const ds = base + 120;
  let depthStencil;
  const dsFormat = view.getUint8(ds);
  const dsFormatName = dsFormat !== 0 ? nameOf(TEXTURE_FORMAT_NAMES, dsFormat) : undefined;
  if (dsFormatName && dsFormatName !== 'undefined') {
    const hasDepth = isDepthTextureFormat(dsFormatName);
    const rawDepthCompare = view.getUint8(ds + 2);
    const mappedCompare = rawDepthCompare !== 0 ? nameOf(COMPARE_FUNCTION_NAMES, rawDepthCompare) : undefined;
    const validCompare = mappedCompare && mappedCompare !== 'undefined' ? mappedCompare : undefined;
    depthStencil = {
      format: dsFormatName,
      depthWriteEnabled: hasDepth ? Boolean(view.getUint8(ds + 1) !== 0) : false,
      stencilFront: decodeStencilFace(view, ds + 4),
      stencilBack: decodeStencilFace(view, ds + 8),
      stencilReadMask: view.getUint32(ds + 12, true),
      stencilWriteMask: view.getUint32(ds + 16, true),
      depthBias: view.getInt32(ds + 20, true),
      depthBiasSlopeScale: finiteOr(f32FromBits(view.getUint32(ds + 24, true)), 0),
      depthBiasClamp: finiteOr(f32FromBits(view.getUint32(ds + 28, true)), 0),
    };
    if (hasDepth) {
      // Depth-containing formats (e.g. depth24plus-stencil8, depth32float) must always carry a
      // valid GPUCompareFunction; default to 'always' when depth test is disabled (code 0).
      depthStencil.depthCompare = validCompare || 'always';
    } else if (validCompare === 'always') {
      depthStencil.depthCompare = 'always';
    }
  }

  const pr = base + 152;
  const topology = nameOf(PRIMITIVE_TOPOLOGY_NAMES, view.getUint8(pr)) || 'triangle-list';
  const primitive = {
    topology,
    frontFace: nameOf(FRONT_FACE_NAMES, view.getUint8(pr + 2)) || 'ccw',
    cullMode: nameOf(CULL_MODE_NAMES, view.getUint8(pr + 3)) || 'none',
  };
  const stripIndexFormat = view.getUint8(pr + 1);
  if (stripIndexFormat !== 0 && (topology === 'triangle-strip' || topology === 'line-strip')) {
    const idxFormat = nameOf(INDEX_FORMAT_NAMES, stripIndexFormat);
    if (idxFormat && idxFormat !== 'undefined') {
      primitive.stripIndexFormat = idxFormat;
    }
  }
  if (view.getUint8(pr + 4) !== 0) primitive.unclippedDepth = true;

  const ms = base + 160;
  const multisample = {
    count: view.getUint32(ms, true) || 1,
    mask: view.getUint32(ms + 4, true),
    alphaToCoverageEnabled: view.getUint8(ms + 8) !== 0,
  };

  const bufferCount = Math.min(view.getUint32(base + 172, true), 8);
  const attributeCount = Math.min(view.getUint32(base + 176, true), 16);
  const attributes = [];
  for (let i = 0; i < attributeCount; i++) {
    const o = base + 244 + 8 * i;
    attributes.push({
      bufferIndex: view.getUint8(o + 6),
      attribute: {
        format: nameOf(VERTEX_FORMAT_NAMES, view.getUint8(o + 4)),
        offset: view.getUint32(o, true),
        shaderLocation: view.getUint8(o + 5),
      },
    });
  }
  const buffers = [];
  for (let i = 0; i < bufferCount; i++) {
    const o = base + 180 + 8 * i;
    const stepMode = view.getUint8(o + 4);
    if (stepMode === 0) {
      buffers.push(null);
      continue;
    }
    buffers.push({
      arrayStride: view.getUint32(o, true),
      stepMode: nameOf(VERTEX_STEP_MODE_NAMES, stepMode),
      attributes: attributes.filter((a) => a.bufferIndex === i).map((a) => a.attribute),
    });
  }
  return { primitive, multisample, depthStencil, targets, buffers, colorTargetCount };
}

function decodeBeginRenderPass(view, offset) {
  const colorCount = Math.min(view.getUint32(offset, true), 8);
  const colors = [];
  for (let i = 0; i < colorCount; i++) {
    const o = offset + 8 + 40 * i;
    const textureId = view.getUint32(o, true);
    if (textureId === 0) {
      colors.push(null);
      continue;
    }
    colors.push({
      textureId,
      loadOp: nameOf(LOAD_OP_NAMES, view.getUint8(o + 4)),
      storeOp: nameOf(STORE_OP_NAMES, view.getUint8(o + 5)),
      view: decodeTextureViewDesc(view, o + 8),
      clearValue: {
        r: view.getFloat32(o + 24, true),
        g: view.getFloat32(o + 28, true),
        b: view.getFloat32(o + 32, true),
        a: view.getFloat32(o + 36, true),
      },
    });
  }
  const d = offset + 328;
  let depthStencil = null;
  const dsTexture = view.getUint32(d, true);
  if (dsTexture !== 0) {
    depthStencil = {
      textureId: dsTexture,
      depthLoadOp: view.getUint8(d + 4),
      depthStoreOp: view.getUint8(d + 5),
      stencilLoadOp: view.getUint8(d + 6),
      stencilStoreOp: view.getUint8(d + 7),
      view: decodeTextureViewDesc(view, d + 8),
      depthClearValue: view.getFloat32(d + 24, true),
      stencilClearValue: view.getUint32(d + 28, true),
      depthReadOnly: view.getUint8(d + 32) !== 0,
      stencilReadOnly: view.getUint8(d + 33) !== 0,
    };
  }
  return { colors, depthStencil };
}

// ---- Packet handler ----

class RenderStatePackets {
  constructor(processor, options = {}) {
    this.processor = processor;
    this.device = processor.device;
    this._cachedDeviceGeneration =
      processor && Number.isInteger(processor._deviceGeneration) ? processor._deviceGeneration : 0;
    // Optional: (kind 'bindGroupLayout' | 'pipelineLayout', contentId: bigint) => GPU object, for
    // ManifestReference layouts loaded from layout-manifest.json (e.g. by reflection_loader.js).
    this.resolveManifestLayout =
      typeof options.resolveManifestLayout === 'function' ? options.resolveManifestLayout : null;
    // W167/W186: optional pipelineCache override (falls back to processor.pipelineCache).
    this.pipelineCache =
      options.pipelineCache && typeof options.pipelineCache === 'object' ? options.pipelineCache : null;
    // W163/W186: optional textureFormats / resolveStorageCompatibleFormat override.
    this._textureFormatsOptions =
      typeof options.resolveStorageCompatibleFormat === 'function' ||
      (options.textureFormats && typeof options.textureFormats === 'object')
        ? {
            resolveStorageCompatibleFormat: options.resolveStorageCompatibleFormat,
            textureFormats: options.textureFormats,
          }
        : null;
    this.bindGroupLayouts = new Map(); // id -> { layout, entries, sequentialBindings }
    this.pipelineLayouts = new Map(); // id -> GPUPipelineLayout
    this.pipelineLayoutMeta = new Map(); // id -> { groupCount, isSingleGroup0, sequentialGroup0, group0HasEntries, group0Entries }
    this.computePipelineMeta = new Map(); // id -> { id, csId, layoutId, entryPoint, hasComputeAttr, autoLayoutRemap, wgslBindings }
    this.renderPipelineAutoLayoutMeta = new Map(); // id -> { id, layoutId, autoLayoutRemap, wgslBindings }
    this.bindGroupDescriptors = new Map(); // id -> { id, layoutId, resources }
    // 'c|r:<pipeId>:<bgId>' (+ '@<dynOffsets>' when root-constant dynamic offsets are non-zero)
    // -> GPUBindGroup | null. LRU-bounded by remappedBindGroupCacheLimit.
    this.remappedBindGroups = new Map();
    this.remappedBindGroupCacheLimit =
      Number.isInteger(options.remappedBindGroupCacheLimit) && options.remappedBindGroupCacheLimit > 0
        ? options.remappedBindGroupCacheLimit
        : REMAPPED_BIND_GROUP_CACHE_LIMIT;
    // W186: hot-path PresentX multi-plane cache keyed by (planeTextureId, viewFormat, sampler, pipeline/layout).
    this.presentxBindGroupCache = new Map();
    this.presentxBindGroupCacheLimit =
      Number.isInteger(options.presentxBindGroupCacheLimit) && options.presentxBindGroupCacheLimit > 0
        ? options.presentxBindGroupCacheLimit
        : PRESENTX_BIND_GROUP_CACHE_LIMIT;
    this._textureGenerations = new Map(); // textureId -> monotonic generation
    this._textureViewOwners = new Map(); // textureId -> GPUTexture handle for silent-replacement detection
    this._destroyed = false;
    this._lastBuiltRenderPipelineAutoLayout = null;
    // W103: pipeline id -> token for compiles still running in processor.asyncPipelineCache. A
    // pipeline stays "pending" here until its result is installed, so flushes never see a gap.
    this.pendingAsyncRenderPipelines = new Map();
    this.pendingAsyncComputePipelines = new Map();
    this.layoutsByContent = new Map(); // 'bgl:<hex>' / 'pl:<hex>' -> object
    this.textureViews = new Map(); // textureId -> Map(descKey -> GPUTextureView)
    this.textureMeta = new Map(); // textureId -> { id, requestedFormat, format, bcFallback, usage, width, height }
    this.dummies = new Map();
    this.emptyLayout = null;
    this.emptyFragment = null;
    this.stats = { packets: 0, errors: 0 };
    this.telemetry = {
      legacyEntryPointRemaps: 0,
      depthCompareDefaults: 0,
      rootSigGroup0Layouts: 0,
      autoLayoutFallbacks: 0,
      bindGroupKindMismatches: 0,
      computePipelineCount: 0,
      computeDispatchCount: 0,
      skippedComputeDispatchCount: 0,
      clampedComputeDispatchCount: 0,
      texture2Count: 0,
      bcFormatFallbacks: 0,
      usageStrips: 0,
      unfilterableFloatFallbacks: 0,
      dynamicOffsetRemaps: 0,
      dynamicOffsetClamps: 0,
      remappedBindGroupEvictions: 0,
      asyncPipelineCompiles: 0,
      asyncPipelineFailures: 0,
      pendingAsyncPipelineSkips: 0,
      suppressedMissingTextureErrors: 0,
      presentxBindGroupCacheHits: 0,
      presentxBindGroupCacheMisses: 0,
      fallbackSamplersBound: 0,
      clampedViewports: 0,
      clampedScissors: 0,
      strippedNonBlendableBlends: 0,
      sanitizedAttachments: 0,
      warnings: [],
    };
    this._seenWarningKeys = new Set();
    this.lastError = null;
    this._packetError = null;
    // W104: 'gen:textureId' keys of missing render-pass attachments already reported once.
    this._reportedMissingTextures = new Set();
    this._suppressPacketError = false;
    this._activePass = null;
    this._activePassTargetSize = null;
    this._lastPassReferenceSize = null;
  }

  getStats() {
    return {
      packets: this.stats.packets,
      errors: this.stats.errors,
      legacyEntryPointRemaps: this.telemetry.legacyEntryPointRemaps,
      depthCompareDefaults: this.telemetry.depthCompareDefaults,
      rootSigGroup0Layouts: this.telemetry.rootSigGroup0Layouts,
      autoLayoutFallbacks: this.telemetry.autoLayoutFallbacks,
      bindGroupKindMismatches: this.telemetry.bindGroupKindMismatches,
      computePipelineCount: this.telemetry.computePipelineCount,
      computeDispatchCount: this.telemetry.computeDispatchCount,
      skippedComputeDispatchCount: this.telemetry.skippedComputeDispatchCount,
      clampedComputeDispatchCount: this.telemetry.clampedComputeDispatchCount,
      texture2Count: this.telemetry.texture2Count,
      bcFormatFallbacks: this.telemetry.bcFormatFallbacks,
      usageStrips: this.telemetry.usageStrips,
      unfilterableFloatFallbacks: this.telemetry.unfilterableFloatFallbacks,
      presentxBindGroupCacheHits: this.telemetry.presentxBindGroupCacheHits,
      presentxBindGroupCacheMisses: this.telemetry.presentxBindGroupCacheMisses,
      fallbackSamplersBound: this.telemetry.fallbackSamplersBound,
      clampedViewports: this.telemetry.clampedViewports,
      clampedScissors: this.telemetry.clampedScissors,
      strippedNonBlendableBlends: this.telemetry.strippedNonBlendableBlends,
      sanitizedAttachments: this.telemetry.sanitizedAttachments,
    };
  }

  getTelemetry() {
    return {
      legacyEntryPointRemaps: this.telemetry.legacyEntryPointRemaps,
      depthCompareDefaults: this.telemetry.depthCompareDefaults,
      rootSigGroup0Layouts: this.telemetry.rootSigGroup0Layouts,
      autoLayoutFallbacks: this.telemetry.autoLayoutFallbacks,
      bindGroupKindMismatches: this.telemetry.bindGroupKindMismatches,
      computePipelineCount: this.telemetry.computePipelineCount,
      computeDispatchCount: this.telemetry.computeDispatchCount,
      skippedComputeDispatchCount: this.telemetry.skippedComputeDispatchCount,
      clampedComputeDispatchCount: this.telemetry.clampedComputeDispatchCount,
      texture2Count: this.telemetry.texture2Count,
      bcFormatFallbacks: this.telemetry.bcFormatFallbacks,
      usageStrips: this.telemetry.usageStrips,
      unfilterableFloatFallbacks: this.telemetry.unfilterableFloatFallbacks,
      dynamicOffsetRemaps: this.telemetry.dynamicOffsetRemaps,
      dynamicOffsetClamps: this.telemetry.dynamicOffsetClamps,
      remappedBindGroupEvictions: this.telemetry.remappedBindGroupEvictions,
      asyncPipelineCompiles: this.telemetry.asyncPipelineCompiles,
      asyncPipelineFailures: this.telemetry.asyncPipelineFailures,
      pendingAsyncPipelineSkips: this.telemetry.pendingAsyncPipelineSkips,
      suppressedMissingTextureErrors: this.telemetry.suppressedMissingTextureErrors,
      presentxBindGroupCacheHits: this.telemetry.presentxBindGroupCacheHits,
      presentxBindGroupCacheMisses: this.telemetry.presentxBindGroupCacheMisses,
      fallbackSamplersBound: this.telemetry.fallbackSamplersBound,
      clampedViewports: this.telemetry.clampedViewports,
      clampedScissors: this.telemetry.clampedScissors,
      strippedNonBlendableBlends: this.telemetry.strippedNonBlendableBlends,
      sanitizedAttachments: this.telemetry.sanitizedAttachments,
      warnings: this.telemetry.warnings.slice(),
    };
  }

  _recordTelemetryWarning(message) {
    if (typeof message !== 'string' || message.length === 0) return;
    if (this.telemetry.warnings.length >= MAX_TELEMETRY_WARNINGS) {
      this.telemetry.warnings.shift();
    }
    this.telemetry.warnings.push(message);
  }

  _recordTelemetryWarningOnce(key, message) {
    if (typeof key === 'string' && key.length > 0) {
      if (this._seenWarningKeys.has(key)) return;
      this._seenWarningKeys.add(key);
    }
    this._recordTelemetryWarning(message);
  }

  _resolveEntryPoint(shaderEntry, stage) {
    const resolved = resolveShaderEntryPoint(shaderEntry, stage);
    if (resolved.remapped) {
      this.telemetry.legacyEntryPointRemaps += 1;
      this._recordTelemetryWarning(
        `legacy entry point remapped: stage=${stage} default=${resolved.defaultEntryPoint} resolved=${resolved.entryPoint} reason=${resolved.reason}`
      );
    }
    return resolved.entryPoint;
  }

  // Returns true when the opcode belongs to ABI v2 (handled, possibly with an error recorded).
  handlePacket(opcode, view, offset, payloadBytes) {
    this._syncDeviceGeneration();
    this._packetError = null;
    this._suppressPacketError = false;
    switch (opcode) {
      case V2_OPCODES.CREATE_BIND_GROUP_LAYOUT:
        return this._count(this._createBindGroupLayout(view, offset, payloadBytes));
      case V2_OPCODES.CREATE_PIPELINE_LAYOUT:
        return this._count(this._createPipelineLayout(view, offset, payloadBytes));
      case V2_OPCODES.CREATE_RENDER_PIPELINE2:
        return this._count(payloadBytes >= KEY_OFFSET + KEY_SIZE && this._createRenderPipeline(view, offset));
      case V2_OPCODES.CREATE_COMPUTE_PIPELINE2:
        return this._count(payloadBytes >= 32 && this._createComputePipeline(view, offset));
      case V2_OPCODES.BEGIN_RENDER_PASS:
        return this._count(payloadBytes >= 368 && this._beginRenderPass(view, offset));
      case V2_OPCODES.END_RENDER_PASS: {
        const endCmd = {
          type: 'renderState',
          _rsPassOp: 'endRenderPass',
          _wgpuMeta: { kind: 'endRenderPass' },
          run: (ctl) => {
            if (ctl.passKind === 'render') ctl.endPass();
            this._activePass = null;
            this._activePassTargetSize = null;
            this._lastPassReferenceSize = null;
          },
        };
        this.processor._passCommands.push(endCmd);
        return this._count(true);
      }
      case V2_OPCODES.SET_VIEWPORT: {
        if (payloadBytes < 24) return this._count(false);
        const rawViewport = {
          x: view.getFloat32(offset, true),
          y: view.getFloat32(offset + 4, true),
          width: view.getFloat32(offset + 8, true),
          height: view.getFloat32(offset + 12, true),
          minDepth: view.getFloat32(offset + 16, true),
          maxDepth: view.getFloat32(offset + 20, true),
        };
        this._queue((ctl) => {
          if (ctl.passKind !== 'render') return;
          const targetSize = this._getActivePassTargetSize(ctl);
          const vp = clampViewportToTarget(rawViewport, targetSize);
          if (isViewportClamped(rawViewport, vp)) {
            this.telemetry.clampedViewports += 1;
            const targetTag = targetSize ? `${targetSize.width}x${targetSize.height}` : 'unbounded';
            this._recordTelemetryWarningOnce(
              `clamp-viewport:${targetTag}`,
              `clamped viewport (${rawViewport.x},${rawViewport.y},${rawViewport.width}x${rawViewport.height},d=${rawViewport.minDepth}..${rawViewport.maxDepth}) to (${vp.x},${vp.y},${vp.width}x${vp.height},d=${vp.minDepth}..${vp.maxDepth}) for target ${targetTag}`
            );
          }
          ctl.pass.setViewport(vp.x, vp.y, vp.width, vp.height, vp.minDepth, vp.maxDepth);
        });
        return this._count(true);
      }
      case V2_OPCODES.SET_SCISSOR_RECT: {
        if (payloadBytes < 16) return this._count(false);
        const rawScissor = {
          x: view.getUint32(offset, true),
          y: view.getUint32(offset + 4, true),
          width: view.getUint32(offset + 8, true),
          height: view.getUint32(offset + 12, true),
        };
        this._queue((ctl) => {
          if (ctl.passKind !== 'render') return;
          const targetSize = this._getActivePassTargetSize(ctl);
          const sc = clampScissorToTarget(rawScissor, targetSize);
          if (isScissorClamped(rawScissor, sc)) {
            this.telemetry.clampedScissors += 1;
            const targetTag = targetSize ? `${targetSize.width}x${targetSize.height}` : 'unbounded';
            this._recordTelemetryWarningOnce(
              `clamp-scissor:${targetTag}`,
              `clamped scissor (${rawScissor.x},${rawScissor.y},${rawScissor.width}x${rawScissor.height}) to (${sc.x},${sc.y},${sc.width}x${sc.height}) for target ${targetTag}`
            );
          }
          ctl.pass.setScissorRect(sc.x, sc.y, sc.width, sc.height);
        });
        return this._count(true);
      }
      case V2_OPCODES.SET_BLEND_CONSTANT: {
        if (payloadBytes < 16) return this._count(false);
        const c = [0, 4, 8, 12].map((o) => finiteOr(view.getFloat32(offset + o, true), 0) || 0);
        this._queueRender((pass) => pass.setBlendConstant({ r: c[0], g: c[1], b: c[2], a: c[3] }));
        return this._count(true);
      }
      case V2_OPCODES.SET_STENCIL_REFERENCE: {
        if (payloadBytes < 4) return this._count(false);
        const reference = view.getUint32(offset, true);
        this._queueRender((pass) => pass.setStencilReference(reference));
        return this._count(true);
      }
      case V2_OPCODES.CREATE_BIND_GROUP2:
        return this._count(this._createBindGroup(view, offset, payloadBytes));
      case V2_OPCODES.SET_BIND_GROUP2:
        return this._count(this._setBindGroup(view, offset, payloadBytes));
      case V2_OPCODES.CREATE_SAMPLER2:
        return this._count(payloadBytes >= 24 && this._createSampler(view, offset));
      case V2_OPCODES.CREATE_TEXTURE2:
        return this._count(this._createTexture2(view, offset, payloadBytes));
      default:
        return false;
    }
  }

  _count(ok) {
    this.stats.packets += 1;
    if (!ok && this._suppressPacketError) {
      // W104: repeated error already reported once for this device generation.
      this._suppressPacketError = false;
      this.telemetry.suppressedMissingTextureErrors += 1;
      return true;
    }
    if (!ok) {
      this.stats.errors += 1;
      this.lastError = this._packetError || 'malformed or truncated v2 packet';
    }
    return true;
  }

  _invalidateRemappedBindGroupsForPipeline(pipelineId, isCompute) {
    const prefix = `${isCompute ? 'c' : 'r'}:${pipelineId}:`;
    for (const key of Array.from(this.remappedBindGroups.keys())) {
      if (key.startsWith(prefix)) {
        this.remappedBindGroups.delete(key);
      }
    }
  }

  _invalidateRemappedBindGroupsForBindGroup(bindGroupId) {
    const target = String(bindGroupId);
    for (const key of Array.from(this.remappedBindGroups.keys())) {
      // Key shape: '<c|r>:<pipeId>:<bgId>[@<dynOffsets>]'.
      const at = key.indexOf('@');
      const base = at >= 0 ? key.slice(0, at) : key;
      if (base.slice(base.lastIndexOf(':') + 1) === target) {
        this.remappedBindGroups.delete(key);
      }
    }
  }

  // LRU insert: re-inserting moves the key to the most-recently-used end; the oldest entries are
  // evicted once the cache exceeds remappedBindGroupCacheLimit.
  _cacheRemappedBindGroup(cacheKey, bindGroup) {
    if (this.remappedBindGroups.has(cacheKey)) this.remappedBindGroups.delete(cacheKey);
    this.remappedBindGroups.set(cacheKey, bindGroup);
    while (this.remappedBindGroups.size > this.remappedBindGroupCacheLimit) {
      const oldest = this.remappedBindGroups.keys().next().value;
      this.remappedBindGroups.delete(oldest);
      this.telemetry.remappedBindGroupEvictions += 1;
    }
  }

  // W186: detect GPUDevice replacement or device-generation bump and flush device-bound caches.
  _syncDeviceGeneration() {
    const liveDevice = (this.processor && this.processor.device) || this.device;
    const liveGen =
      this.processor && Number.isInteger(this.processor._deviceGeneration)
        ? this.processor._deviceGeneration
        : this._cachedDeviceGeneration;
    if (liveDevice !== this.device || liveGen !== this._cachedDeviceGeneration) {
      this.invalidatePresentXCache();
      this.textureViews.clear();
      this._textureViewOwners.clear();
      this.dummies.clear();
      this.remappedBindGroups.clear();
      this.emptyLayout = null;
      this.emptyFragment = null;
      this.device = liveDevice;
      this._cachedDeviceGeneration = liveGen;
      return true;
    }
    return false;
  }

  _bumpTextureGeneration(textureId) {
    const next = (this._textureGenerations.get(textureId) || 0) + 1;
    this._textureGenerations.set(textureId, next);
    return next;
  }

  _getTextureGeneration(textureId, liveTexture) {
    const prevOwner = this._textureViewOwners.get(textureId);
    if (liveTexture && prevOwner && prevOwner !== liveTexture) {
      this.textureViews.delete(textureId);
      this._invalidatePresentXCacheForResource(textureId);
      this._textureViewOwners.set(textureId, liveTexture);
      return this._bumpTextureGeneration(textureId);
    }
    if (liveTexture && !prevOwner) {
      this._textureViewOwners.set(textureId, liveTexture);
    }
    return this._textureGenerations.get(textureId) || 0;
  }

  _getPipelineCache() {
    if (this.pipelineCache && typeof this.pipelineCache === 'object') {
      return this.pipelineCache;
    }
    if (this.processor && this.processor.pipelineCache && typeof this.processor.pipelineCache === 'object') {
      return this.processor.pipelineCache;
    }
    return null;
  }

  _resolveDefaultSamplerFromCache(samplerKind) {
    const cache = this._getPipelineCache();
    if (!cache || typeof cache.getOrCreateDefaultSampler !== 'function') {
      return null;
    }
    const kind =
      samplerKind === 'comparison' || samplerKind === 'non-filtering' || samplerKind === 'filtering'
        ? samplerKind
        : 'filtering';
    try {
      const sampler = cache.getOrCreateDefaultSampler(kind);
      return sampler || null;
    } catch (err) {
      this._recordTelemetryWarningOnce(
        `fallback-sampler-error:${kind}`,
        `pipelineCache.getOrCreateDefaultSampler(${kind}) failed: ${String(err && err.message ? err.message : err)}`
      );
      return null;
    }
  }

  _resolveStorageFormat(format) {
    return resolveStorageTextureFormat(format, this.device, this._textureFormatsOptions);
  }

  _cachePresentXBindGroup(cacheKey, entry) {
    if (this.presentxBindGroupCache.has(cacheKey)) {
      this.presentxBindGroupCache.delete(cacheKey);
    }
    this.presentxBindGroupCache.set(cacheKey, entry);
    while (this.presentxBindGroupCache.size > this.presentxBindGroupCacheLimit) {
      const oldest = this.presentxBindGroupCache.keys().next().value;
      this.presentxBindGroupCache.delete(oldest);
    }
  }

  _invalidatePresentXCacheForResource(resourceId) {
    if (this.presentxBindGroupCache.size === 0) return 0;
    let removed = 0;
    for (const [key, entry] of Array.from(this.presentxBindGroupCache.entries())) {
      if (
        entry.planeTextureId === resourceId ||
        entry.samplerId === resourceId ||
        entry.layoutId === resourceId ||
        entry.pipelineId === resourceId
      ) {
        this.presentxBindGroupCache.delete(key);
        removed += 1;
      }
    }
    return removed;
  }

  invalidatePresentXCache(filter = null) {
    if (!filter || typeof filter !== 'object') {
      const count = this.presentxBindGroupCache.size;
      this.presentxBindGroupCache.clear();
      return count;
    }
    let removed = 0;
    for (const [key, entry] of Array.from(this.presentxBindGroupCache.entries())) {
      if (
        (filter.planeTextureId !== undefined && entry.planeTextureId === filter.planeTextureId) ||
        (filter.samplerId !== undefined && entry.samplerId === filter.samplerId) ||
        (filter.sampler !== undefined && entry.sampler === filter.sampler) ||
        (filter.layoutId !== undefined && entry.layoutId === filter.layoutId) ||
        (filter.layout !== undefined && entry.layout === filter.layout) ||
        (filter.pipelineId !== undefined && entry.pipelineId === filter.pipelineId) ||
        (filter.pipeline !== undefined && entry.pipeline === filter.pipeline)
      ) {
        this.presentxBindGroupCache.delete(key);
        removed += 1;
      }
    }
    return removed;
  }

  // W186: hot-path PresentX multi-plane bind-group + texture-view cache.
  // Caches per-(planeTextureId, viewFormat, sampler, pipeline/layout) across frames and invalidates
  // automatically on texture handle/generation change, device change, reset, or destroy.
  getOrCreatePresentXPlaneBindGroup(params) {
    if (this._destroyed || !params || typeof params !== 'object') return null;
    this._syncDeviceGeneration();
    if (!this.device || typeof this.device.createBindGroup !== 'function') return null;

    const planeTextureId = params.planeTextureId;
    if (!Number.isInteger(planeTextureId) || planeTextureId <= 0 || planeTextureId === SWAPCHAIN_TEXTURE_ID) {
      return null;
    }
    const texture =
      this.processor && this.processor.textures ? this.processor.textures.get(planeTextureId) : null;
    if (!texture || typeof texture.createView !== 'function') return null;

    const texGen = this._getTextureGeneration(planeTextureId, texture);
    const baseViewDesc =
      params.viewDesc && typeof params.viewDesc === 'object' ? { ...params.viewDesc } : {};
    if (
      typeof params.viewFormat === 'string' &&
      params.viewFormat.length > 0 &&
      params.viewFormat !== 'undefined' &&
      !baseViewDesc.format
    ) {
      baseViewDesc.format = params.viewFormat;
    }
    const resolvedViewFormat =
      this._resolveTextureFormat(planeTextureId, baseViewDesc) ||
      (typeof params.viewFormat === 'string' && params.viewFormat !== 'undefined'
        ? params.viewFormat
        : 'default');
    const viewDescKey = buildTextureViewKey(baseViewDesc);

    const samplerId = Number.isInteger(params.samplerId) && params.samplerId > 0 ? params.samplerId : 0;
    let samplerObj = params.sampler || null;
    let usedFallbackSampler = false;
    if (!samplerObj && samplerId > 0 && this.processor && this.processor.samplers) {
      samplerObj = this.processor.samplers.get(samplerId) || null;
    }
    if (!samplerObj) {
      const samplerKind =
        params.samplerKind === 'comparison' || params.samplerKind === 'non-filtering'
          ? params.samplerKind
          : 'filtering';
      samplerObj = this._resolveDefaultSamplerFromCache(samplerKind);
      if (samplerObj) {
        usedFallbackSampler = true;
      } else {
        samplerObj = this._dummyResource({ sampler: { type: samplerKind } });
      }
    }
    if (!samplerObj) return null;

    const pipelineId = Number.isInteger(params.pipelineId) && params.pipelineId > 0 ? params.pipelineId : 0;
    let pipelineObj = params.pipeline || null;
    if (!pipelineObj && pipelineId > 0 && this.processor && this.processor.pipelines) {
      pipelineObj = this.processor.pipelines.get(pipelineId) || null;
    }

    const layoutId = Number.isInteger(params.layoutId) && params.layoutId > 0 ? params.layoutId : 0;
    let bindGroupLayout = params.layout || null;
    if (!bindGroupLayout && layoutId > 0) {
      const bglRec = this.bindGroupLayouts.get(layoutId);
      bindGroupLayout = bglRec ? bglRec.layout : null;
    }
    if (!bindGroupLayout && pipelineObj && typeof pipelineObj.getBindGroupLayout === 'function') {
      try {
        bindGroupLayout = pipelineObj.getBindGroupLayout(0);
      } catch (_) {
        bindGroupLayout = null;
      }
    }
    if (!bindGroupLayout) return null;

    const textureBinding = Number.isInteger(params.textureBinding) ? params.textureBinding : 0;
    const samplerBinding = Number.isInteger(params.samplerBinding) ? params.samplerBinding : 1;
    const cacheKey = `px:${planeTextureId}:${texGen}:${resolvedViewFormat}:${viewDescKey}:s=${gpuObjectId(
      samplerObj
    )}:p=${gpuObjectId(pipelineObj)}:l=${gpuObjectId(bindGroupLayout)}:b=${textureBinding},${samplerBinding}`;

    const cached = this.presentxBindGroupCache.get(cacheKey);
    if (
      cached &&
      cached.device === this.device &&
      cached.deviceGeneration === this._cachedDeviceGeneration &&
      cached.texture === texture &&
      cached.textureGen === texGen &&
      cached.sampler === samplerObj &&
      cached.layout === bindGroupLayout &&
      cached.pipeline === (pipelineObj || null)
    ) {
      this._cachePresentXBindGroup(cacheKey, cached);
      this.telemetry.presentxBindGroupCacheHits += 1;
      if (usedFallbackSampler) {
        this.telemetry.fallbackSamplersBound += 1;
      }
      return { view: cached.view, bindGroup: cached.bindGroup, hit: true, cacheKey };
    }

    const view = this._textureView(planeTextureId, baseViewDesc);
    if (!view) return null;

    let bindGroup;
    try {
      bindGroup = this.device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
          { binding: textureBinding, resource: view },
          { binding: samplerBinding, resource: samplerObj },
        ],
      });
    } catch (err) {
      this._packetError = `PresentX bind group (texture ${planeTextureId}): ${String(
        err && err.message ? err.message : err
      )}`;
      return null;
    }

    if (usedFallbackSampler) {
      this.telemetry.fallbackSamplersBound += 1;
    }
    this.telemetry.presentxBindGroupCacheMisses += 1;
    const entry = {
      view,
      bindGroup,
      planeTextureId,
      texture,
      textureGen: texGen,
      viewFormat: resolvedViewFormat,
      viewDescKey,
      samplerId,
      sampler: samplerObj,
      layoutId,
      layout: bindGroupLayout,
      pipelineId,
      pipeline: pipelineObj || null,
      device: this.device,
      deviceGeneration: this._cachedDeviceGeneration,
    };
    this._cachePresentXBindGroup(cacheKey, entry);
    return { view, bindGroup, hit: false, cacheKey };
  }

  reset() {
    this.invalidatePresentXCache();
    this.textureViews.clear();
    this._textureViewOwners.clear();
    this.remappedBindGroups.clear();
    this.dummies.clear();
    this.emptyLayout = null;
    this.emptyFragment = null;
    this._activePass = null;
    this._activePassTargetSize = null;
    this._lastPassReferenceSize = null;
  }

  destroy() {
    this.reset();
    this.bindGroupLayouts.clear();
    this.pipelineLayouts.clear();
    this.pipelineLayoutMeta.clear();
    this.computePipelineMeta.clear();
    this.renderPipelineAutoLayoutMeta.clear();
    this.pendingAsyncRenderPipelines.clear();
    this.pendingAsyncComputePipelines.clear();
    this.bindGroupDescriptors.clear();
    this.layoutsByContent.clear();
    this.textureMeta.clear();
    this._textureGenerations.clear();
    this._reportedMissingTextures.clear();
    this._destroyed = true;
  }

  // DestroyResource hook: forget layouts, metadata and cached views of `id`.
  destroyResource(id) {
    this.bindGroupLayouts.delete(id);
    this.pipelineLayouts.delete(id);
    this.pipelineLayoutMeta.delete(id);
    this.computePipelineMeta.delete(id);
    this.renderPipelineAutoLayoutMeta.delete(id);
    this.pendingAsyncRenderPipelines.delete(id);
    this.pendingAsyncComputePipelines.delete(id);
    this.bindGroupDescriptors.delete(id);
    this._invalidateRemappedBindGroupsForPipeline(id, true);
    this._invalidateRemappedBindGroupsForPipeline(id, false);
    this._invalidateRemappedBindGroupsForBindGroup(id);
    this._invalidatePresentXCacheForResource(id);
    this._bumpTextureGeneration(id);
    this._textureViewOwners.delete(id);
    this.textureViews.delete(id);
    this.textureMeta.delete(id);
    if (
      this.processor &&
      this.processor._pipelineMeta &&
      typeof this.processor._pipelineMeta.delete === 'function'
    ) {
      this.processor._pipelineMeta.delete(id);
    }
  }

  _queue(run) {
    this.processor._passCommands.push({ type: 'renderState', run });
  }

  _queueRender(fn) {
    this._queue((ctl) => {
      if (ctl.passKind === 'render') fn(ctl.pass);
    });
  }

  // ---- Layouts ----

  _createBindGroupLayout(view, offset, payloadBytes) {
    if (payloadBytes < 24) return false;
    const id = view.getUint32(offset, true);
    const count = view.getUint32(offset + 4, true);
    const flags = view.getUint32(offset + 8, true);
    const contentId = view.getBigUint64(offset + 16, true);
    if ((flags & MANIFEST_REFERENCE_FLAG) !== 0) {
      const layout = this.resolveManifestLayout && this.resolveManifestLayout('bindGroupLayout', contentId);
      if (!layout) {
        this._packetError = `manifest bind group layout 0x${contentId.toString(16)} unavailable`;
        return false;
      }
      this.bindGroupLayouts.set(id, { layout, entries: null });
      return true;
    }
    if (24 + count * 24 > payloadBytes) return false;
    const entries = [];
    const seenBindings = new Set();
    let sequentialBindings = true;
    for (let i = 0; i < count; i++) {
      const entry = decodeBindGroupLayoutEntry(view, offset + 24 + i * 24);
      if (!entry) return false;
      if (seenBindings.has(entry.binding)) {
        this._packetError = `bind group layout ${id}: duplicate binding ${entry.binding}`;
        return false;
      }
      if (entry.storageTexture) {
        const origFmt = entry.storageTexture.format;
        const resolvedStorage = this._resolveStorageFormat(origFmt);
        if (resolvedStorage.supportsStorageBinding && resolvedStorage.storageFormat) {
          if (resolvedStorage.storageFormat !== origFmt) {
            entry.storageTexture.format = resolvedStorage.storageFormat;
            this._recordTelemetryWarningOnce(
              `bgl-storage-format:${origFmt}`,
              `bind group layout storageTexture format ${origFmt} remapped to linear sibling ${resolvedStorage.storageFormat}`
            );
          }
        } else {
          entry.storageTexture.format = 'rgba8unorm';
          this._recordTelemetryWarningOnce(
            `bgl-storage-format:${origFmt}`,
            `bind group layout storageTexture format ${origFmt} unsupported${
              resolvedStorage.reason ? ` (${resolvedStorage.reason})` : ''
            }; falling back to rgba8unorm`
          );
        }
        if (
          entry.storageTexture.viewDimension === 'cube' ||
          entry.storageTexture.viewDimension === 'cube-array'
        ) {
          const origDim = entry.storageTexture.viewDimension;
          entry.storageTexture.viewDimension = '2d-array';
          this._recordTelemetryWarningOnce(
            `bgl-storage-dim:${origDim}`,
            `bind group layout storageTexture dimension ${origDim} unsupported; remapping to 2d-array`
          );
        }
      }
      seenBindings.add(entry.binding);
      if (entry.binding !== i) sequentialBindings = false;
      entries.push(entry);
    }
    const contentKey = contentId !== 0n ? `bgl:${contentId.toString(16)}` : null;
    let layout = contentKey ? this.layoutsByContent.get(contentKey) : undefined;
    if (!layout) {
      try {
        layout = this.device.createBindGroupLayout({ entries });
      } catch (err) {
        this._packetError = `bind group layout ${id}: ${String(err && err.message ? err.message : err)}`;
        return false;
      }
      if (contentKey) this.layoutsByContent.set(contentKey, layout);
    }
    this.bindGroupLayouts.set(id, {
      layout,
      entries,
      sequentialBindings,
      dynamicOffsetIndices: dynamicOffsetIndicesForLayoutEntries(entries),
    });
    return true;
  }

  _emptyBindGroupLayout() {
    if (!this.emptyLayout) this.emptyLayout = this.device.createBindGroupLayout({ entries: [] });
    return this.emptyLayout;
  }

  _createPipelineLayout(view, offset, payloadBytes) {
    if (payloadBytes < 24) return false;
    const id = view.getUint32(offset, true);
    const count = view.getUint32(offset + 4, true);
    const flags = view.getUint32(offset + 8, true);
    const contentId = view.getBigUint64(offset + 16, true);
    let layout;
    let meta = {
      groupCount: count,
      isSingleGroup0: false,
      sequentialGroup0: false,
      group0HasEntries: false,
      group0Entries: null,
    };
    if ((flags & MANIFEST_REFERENCE_FLAG) !== 0) {
      layout = this.resolveManifestLayout && this.resolveManifestLayout('pipelineLayout', contentId);
      if (!layout) {
        this._packetError = `manifest pipeline layout 0x${contentId.toString(16)} unavailable`;
        return false;
      }
    } else {
      if (24 + count * 4 > payloadBytes) return false;
      const bindGroupLayouts = [];
      let group0Record = null;
      for (let g = 0; g < count; g++) {
        const bglId = view.getUint32(offset + 24 + g * 4, true);
        const record = bglId !== 0 ? this.bindGroupLayouts.get(bglId) : null;
        if (bglId !== 0 && !record) {
          this._packetError = `pipeline layout ${id}: unknown bind group layout ${bglId}`;
          return false;
        }
        if (g === 0) group0Record = record;
        bindGroupLayouts.push(record ? record.layout : this._emptyBindGroupLayout());
      }
      try {
        layout = this.device.createPipelineLayout({ bindGroupLayouts });
      } catch (err) {
        this._packetError = `pipeline layout ${id}: ${String(err && err.message ? err.message : err)}`;
        return false;
      }
      meta = {
        groupCount: count,
        isSingleGroup0: count === 1 && group0Record !== null,
        sequentialGroup0: Boolean(count === 1 && group0Record && group0Record.sequentialBindings),
        group0HasEntries: Boolean(
          group0Record &&
            (group0Record.entries === null ||
              (Array.isArray(group0Record.entries) && group0Record.entries.length > 0))
        ),
        group0Entries: group0Record ? group0Record.entries : null,
      };
    }
    this.pipelineLayouts.set(id, layout);
    this.pipelineLayoutMeta.set(id, meta);
    return true;
  }

  _layoutFor(layoutId) {
    if (layoutId === 0) return 'auto';
    const layout = this.pipelineLayouts.get(layoutId);
    if (!layout) this._packetError = `unknown pipeline layout ${layoutId}`;
    return layout || null;
  }

  // ---- Pipelines ----

  // W106: materialize lazy GPUShaderModule on first pipeline reference.
  _ensureShaderModule(id) {
    if (this.processor && typeof this.processor.ensureShaderModule === 'function') {
      return this.processor.ensureShaderModule(id);
    }
    return this.processor && this.processor.shaderModules
      ? this.processor.shaderModules.get(id) || null
      : null;
  }

  // The exact GPURenderPipelineDescriptor for a CreateRenderPipeline2 payload at `offset`.
  buildRenderPipelineDescriptor(view, offset) {
    this._lastBuiltRenderPipelineAutoLayout = null;
    const pipeId = view.getUint32(offset, true);
    const vsId = view.getUint32(offset + 4, true);
    const fsId = view.getUint32(offset + 8, true);
    const layoutId = view.getUint32(offset + 12, true);
    let layout = this._layoutFor(layoutId);
    const vs = this._ensureShaderModule(vsId);
    if (!vs || !layout) {
      if (!vs) this._packetError = `unknown vertex shader module ${vsId}`;
      return null;
    }
    const keyBase = offset + KEY_OFFSET;
    const key = decodeRenderPipelineKey(view, keyBase);
    const vertexCheck = validateVertexBufferLayouts(key.buffers);
    if (!vertexCheck.valid) {
      this._packetError = `render pipeline vertex layout invalid: ${vertexCheck.error}`;
      return null;
    }
    const expectedAttrs = Math.min(view.getUint32(keyBase + 176, true), 16);
    const attachedAttrs = key.buffers.reduce((n, b) => n + (b ? b.attributes.length : 0), 0);
    if (attachedAttrs !== expectedAttrs) {
      this._packetError = `render pipeline vertex layout dropped attributes (${attachedAttrs}/${expectedAttrs})`;
      return null;
    }
    for (let i = 0; i < key.targets.length; i += 1) {
      const target = key.targets[i];
      if (target && (!target.format || target.format === 'undefined')) {
        this._packetError = `render pipeline target ${i}: invalid texture format`;
        return null;
      }
    }
    let fs = null;
    let fragmentModule = null;
    let targets = key.targets;
    if (fsId !== 0) {
      fs = this._ensureShaderModule(fsId);
      if (!fs) {
        this._packetError = `unknown fragment shader module ${fsId}`;
        return null;
      }
      fragmentModule = fs.module;
      if (Array.isArray(targets) && targets.length > 0) {
        targets = targets.map((t, idx) => {
          if (!t) return null;
          const checked = sanitizeColorTargetBlendState(t, this.device);
          if (checked.stripped) {
            this.telemetry.strippedNonBlendableBlends += 1;
            this._recordTelemetryWarningOnce(
              `strip-blend:${t.format}:${idx}`,
              `render pipeline ${pipeId} target ${idx}: stripped blend state on non-blendable format ${t.format} (${checked.reason})`
            );
          }
          return checked.target;
        });
      }
    } else if (key.colorTargetCount > 0) {
      // No pixel shader but color targets: an empty fragment stage that writes nothing.
      if (!this.emptyFragment) this.emptyFragment = this.device.createShaderModule({ code: EMPTY_FRAGMENT_WGSL });
      fragmentModule = this.emptyFragment;
      targets = targets.map((t) => (t ? { format: t.format, writeMask: 0 } : null));
    }

    const vsSource = getShaderSourceText(vs);
    const fsSource = fs ? getShaderSourceText(fs) : null;
    const wgslBindings = mergeWgslGroup0Bindings(
      vsSource ? parseWgslGroup0Bindings(vsSource) : [],
      fsSource ? parseWgslGroup0Bindings(fsSource) : []
    );
    let autoLayoutRemap = false;
    const layoutMeta = layoutId !== 0 ? this.pipelineLayoutMeta.get(layoutId) : null;
    if (
      layoutId !== 0 &&
      layoutMeta &&
      !isExplicitLayoutCompatibleWithWgsl(layoutMeta.group0Entries, wgslBindings)
    ) {
      layout = 'auto';
      autoLayoutRemap = true;
      this._lastBuiltRenderPipelineAutoLayout = { autoLayoutRemap: true, wgslBindings };
      this._recordTelemetryWarningOnce(
        `auto-layout-remap:render:${pipeId}:${layoutId}`,
        `render pipeline ${pipeId}: explicit layout ${layoutId} incompatible with WGSL @group(0) bindings; falling back to layout:'auto'`
      );
    } else {
      this._lastBuiltRenderPipelineAutoLayout = { autoLayoutRemap: false, wgslBindings };
    }

    if (layoutId === 0 || autoLayoutRemap) {
      this.telemetry.autoLayoutFallbacks += 1;
    } else if (layoutMeta && layoutMeta.isSingleGroup0) {
      this.telemetry.rootSigGroup0Layouts += 1;
    }
    if (key.depthStencil && isDepthTextureFormat(key.depthStencil.format) && view.getUint8(keyBase + 122) === 0) {
      this.telemetry.depthCompareDefaults += 1;
    }

    const vsEntryPoint = this._resolveEntryPoint(vs, 'vertex');
    const descriptor = {
      layout,
      vertex: { module: vs.module, entryPoint: vsEntryPoint, buffers: key.buffers },
      primitive: key.primitive,
      multisample: key.multisample,
    };
    if (key.depthStencil) descriptor.depthStencil = key.depthStencil;
    if (fragmentModule) {
      const fsEntryPoint = fs ? this._resolveEntryPoint(fs, 'fragment') : DEFAULT_ENTRY_POINTS.fragment;
      descriptor.fragment = { module: fragmentModule, entryPoint: fsEntryPoint, targets };
    }
    return descriptor;
  }

  _createRenderPipeline(view, offset) {
    const descriptor = this.buildRenderPipelineDescriptor(view, offset);
    if (!descriptor) return false;
    const autoLayoutInfo = this._lastBuiltRenderPipelineAutoLayout;
    this._lastBuiltRenderPipelineAutoLayout = null;
    const id = view.getUint32(offset, true);
    const vsId = view.getUint32(offset + 4, true);
    const fsId = view.getUint32(offset + 8, true);
    const layoutId = view.getUint32(offset + 12, true);
    const info = {
      vsId,
      fsId,
      layoutId,
      autoLayoutRemap: Boolean(autoLayoutInfo && autoLayoutInfo.autoLayoutRemap),
      wgslBindings: autoLayoutInfo && Array.isArray(autoLayoutInfo.wgslBindings) ? autoLayoutInfo.wgslBindings : [],
    };
    const asyncCache = this._asyncPipelineCacheFor('render');
    if (asyncCache) {
      return this._compileRenderPipelineAsync(asyncCache, id, descriptor, info);
    }
    let pipeline;
    try {
      pipeline = this.device.createRenderPipeline(descriptor);
    } catch (err) {
      this._packetError = `render pipeline ${id}: ${String(
        err && err.message ? err.message : err
      )}`;
      return false;
    }
    this._installRenderPipeline(id, pipeline, descriptor, info);
    return true;
  }

  // ---- W103: async pipeline compilation (processor.asyncPipelineCache) ----

  /**
   * The processor's AsyncPipelineCache when `kind` pipelines can compile asynchronously on this
   * device, else null (mock devices without create*PipelineAsync keep the synchronous path).
   */
  _asyncPipelineCacheFor(kind) {
    const cache = this.processor && this.processor.asyncPipelineCache;
    if (!cache || typeof cache.compileRenderPipeline !== 'function' || typeof cache.compileComputePipeline !== 'function') {
      return null;
    }
    const method = kind === 'compute' ? 'createComputePipelineAsync' : 'createRenderPipelineAsync';
    return this.device && typeof this.device[method] === 'function' ? cache : null;
  }

  isAsyncPipelinePending(id, isCompute) {
    return (isCompute ? this.pendingAsyncComputePipelines : this.pendingAsyncRenderPipelines).has(id);
  }

  /**
   * Queue `descriptor` on the async cache; the pipeline is installed when the compile resolves.
   * Returns true: the packet is well formed and compile failures are reported asynchronously.
   *
   * @param {object} asyncCache
   * @param {'render' | 'compute'} kind
   * @param {number} id
   * @param {object} descriptor
   * @param {object | null} fallbackDescriptor - retried once (layout:'auto') if `descriptor` rejects.
   * @param {(pipeline: object, retried: boolean) => void} install
   */
  _compilePipelineAsync(asyncCache, kind, id, descriptor, fallbackDescriptor, install) {
    const isCompute = kind === 'compute';
    const pending = isCompute ? this.pendingAsyncComputePipelines : this.pendingAsyncRenderPipelines;
    const live = isCompute ? this.processor.computePipelines : this.processor.pipelines;
    // A re-created id must never draw with its previous pipeline or receive an older compile.
    if (typeof asyncCache.delete === 'function') asyncCache.delete(id);
    if (live && typeof live.delete === 'function') live.delete(id);
    const token = {};
    pending.set(id, token);
    this.telemetry.asyncPipelineCompiles += 1;
    const compile = isCompute ? asyncCache.compileComputePipeline : asyncCache.compileRenderPipeline;
    let promise;
    try {
      promise = Promise.resolve(compile.call(asyncCache, id, descriptor, { fallbackDescriptor }));
    } catch (err) {
      promise = Promise.resolve({ id, status: 'failed', error: err });
    }
    promise.then(
      (result) => this._onAsyncPipelineSettled(kind, id, token, result, install),
      (err) => this._onAsyncPipelineSettled(kind, id, token, { id, status: 'failed', error: err }, install)
    );
    return true;
  }

  _onAsyncPipelineSettled(kind, id, token, result, install) {
    const pending = kind === 'compute' ? this.pendingAsyncComputePipelines : this.pendingAsyncRenderPipelines;
    if (pending.get(id) !== token) return; // destroyed or re-created meanwhile
    pending.delete(id);
    const status = result && result.status;
    if (status === 'stale' || status === 'deleted') return; // device reset / resource destroyed
    if (status === 'ready' && result.pipeline) {
      try {
        install(result.pipeline, Boolean(result.retried));
      } catch (err) {
        this._recordAsyncPipelineFailure(kind, id, err);
      }
      return;
    }
    this._recordAsyncPipelineFailure(kind, id, result && result.error);
  }

  _recordAsyncPipelineFailure(kind, id, err) {
    const message = `${kind} pipeline ${id}: ${String(err && err.message ? err.message : err || 'async compile failed')}`;
    this.telemetry.asyncPipelineFailures += 1;
    this.stats.errors += 1;
    this.lastError = message;
    this._recordTelemetryWarningOnce(`async-pipeline-failed:${kind}:${id}`, message);
  }

  _compileRenderPipelineAsync(asyncCache, id, descriptor, info) {
    // Explicit root-signature layout first; if Dawn rejects it asynchronously, retry with
    // layout:'auto' and remap group 0 at flush time like the synchronous mismatch path does.
    const fallbackDescriptor =
      !info.autoLayoutRemap && descriptor.layout !== 'auto' && info.wgslBindings.length > 0
        ? { ...descriptor, layout: 'auto' }
        : null;
    return this._compilePipelineAsync(asyncCache, 'render', id, descriptor, fallbackDescriptor, (pipeline, retried) => {
      const usedFallback = retried && fallbackDescriptor !== null;
      if (usedFallback) {
        this.telemetry.autoLayoutFallbacks += 1;
        this._recordTelemetryWarningOnce(
          `async-auto-layout:render:${id}:${info.layoutId}`,
          `render pipeline ${id}: explicit layout ${info.layoutId} rejected by async compile; using layout:'auto'`
        );
      }
      this._installRenderPipeline(id, pipeline, usedFallback ? fallbackDescriptor : descriptor, {
        ...info,
        autoLayoutRemap: info.autoLayoutRemap || usedFallback,
      });
    });
  }

  _installRenderPipeline(id, pipeline, descriptor, info) {
    const { vsId, fsId, layoutId } = info;
    this.processor.pipelines.set(id, pipeline);
    this._invalidateRemappedBindGroupsForPipeline(id, false);
    this._invalidatePresentXCacheForResource(id);
    if (info.autoLayoutRemap) {
      this.renderPipelineAutoLayoutMeta.set(id, {
        id,
        layoutId,
        autoLayoutRemap: true,
        wgslBindings: info.wgslBindings,
      });
    } else {
      this.renderPipelineAutoLayoutMeta.delete(id);
    }
    if (this.processor._pipelineMeta && typeof this.processor._pipelineMeta.set === 'function') {
      const vs = this.processor.shaderModules.get(vsId) || null;
      const fs = fsId !== 0 ? this.processor.shaderModules.get(fsId) || null : null;
      const isFallback = Boolean(
        (vs && vs.source === 'builtin') || (fs && fs.source === 'builtin')
      );
      const usesFallbackShader = Boolean(
        (vs && (vs.source === 'builtin' || vs.source === 'fallback')) ||
          (fs && (fs.source === 'builtin' || fs.source === 'fallback'))
      );
      const isRealPack = Boolean(
        vs && vs.source === 'pack' && (fsId === 0 || (fs && fs.source === 'pack'))
      );
      let expectsGroup0 = Boolean(
        wgslDeclaresGroup0(getShaderSourceText(vs)) || wgslDeclaresGroup0(getShaderSourceText(fs))
      );
      if (!expectsGroup0 && layoutId !== 0) {
        const layoutMeta = this.pipelineLayoutMeta.get(layoutId);
        if (layoutMeta && layoutMeta.group0HasEntries) {
          expectsGroup0 = true;
        }
      }
      const effectiveFallbackProfile =
        vs && vs.fallbackProfile && vs.fallbackProfile !== 'default'
          ? vs.fallbackProfile
          : fs && fs.fallbackProfile && fs.fallbackProfile !== 'default'
            ? fs.fallbackProfile
            : isFallback
              ? 'default'
              : null;
      const firstBuffer =
        descriptor.vertex && Array.isArray(descriptor.vertex.buffers) && descriptor.vertex.buffers.length > 0
          ? descriptor.vertex.buffers[0]
          : null;
      const vertexStride =
        firstBuffer && typeof firstBuffer.arrayStride === 'number' ? firstBuffer.arrayStride : 0;
      const topology =
        (descriptor.primitive && descriptor.primitive.topology) || 'triangle-list';
      this.processor._pipelineMeta.set(id, {
        id,
        vsShaderId: vsId,
        fsShaderId: fsId,
        layoutId,
        vertexStride,
        topology,
        isV2: true,
        hasReflectedLayout: layoutId !== 0,
        isRealPack,
        isFallback,
        usesFallbackShader,
        expectsGroup0,
        fallbackProfile: effectiveFallbackProfile,
      });
    }
    if (typeof this.processor._invalidateSpecializedPipelinesFor === 'function') {
      this.processor._invalidateSpecializedPipelinesFor(id);
    }
  }

  _inspectComputeShaderEntryPoint(cs, csId) {
    const sourceText = getShaderSourceText(cs);
    const declared = sourceText ? findWgslStageEntryPoints(sourceText, 'compute') : [];
    const hasComputeAttr = declared.length > 0;
    if (sourceText && !hasComputeAttr) {
      this._recordTelemetryWarningOnce(
        `missing-compute-entry:${csId}`,
        `compute shader module ${csId} lacks @compute / cs_main declaration; defaulting to ${DEFAULT_ENTRY_POINTS.compute}`
      );
      return { entryPoint: DEFAULT_ENTRY_POINTS.compute, hasComputeAttr: false };
    }
    const csEntryPoint = this._resolveEntryPoint(cs, 'compute');
    return { entryPoint: csEntryPoint, hasComputeAttr };
  }

  _createComputePipeline(view, offset) {
    const id = view.getUint32(offset, true);
    const csId = view.getUint32(offset + 4, true);
    const layoutId = view.getUint32(offset + 8, true);
    let layout = this._layoutFor(layoutId);
    const cs = this._ensureShaderModule(csId);
    if (!cs || !layout) {
      if (!cs) this._packetError = `unknown compute shader module ${csId}`;
      return false;
    }
    const { entryPoint: csEntryPoint, hasComputeAttr } = this._inspectComputeShaderEntryPoint(cs, csId);
    const csSource = getShaderSourceText(cs);
    const wgslBindings = csSource ? parseWgslGroup0Bindings(csSource) : [];
    let autoLayoutRemap = false;
    const layoutMeta = layoutId !== 0 ? this.pipelineLayoutMeta.get(layoutId) : null;
    if (
      layoutId !== 0 &&
      layoutMeta &&
      !isExplicitLayoutCompatibleWithWgsl(layoutMeta.group0Entries, wgslBindings)
    ) {
      layout = 'auto';
      autoLayoutRemap = true;
      this._recordTelemetryWarningOnce(
        `auto-layout-remap:compute:${id}:${layoutId}`,
        `compute pipeline ${id}: explicit layout ${layoutId} incompatible with WGSL @group(0) bindings; falling back to layout:'auto'`
      );
    }
    const descriptor = { layout, compute: { module: cs.module, entryPoint: csEntryPoint } };
    const info = {
      csId,
      layoutId,
      entryPoint: csEntryPoint,
      hasComputeAttr,
      autoLayoutRemap,
      wgslBindings,
      isSingleGroup0Layout: Boolean(layoutMeta && layoutMeta.isSingleGroup0),
    };
    const asyncCache = this._asyncPipelineCacheFor('compute');
    if (asyncCache) {
      const fallbackDescriptor =
        !autoLayoutRemap && layout !== 'auto' && wgslBindings.length > 0
          ? { ...descriptor, layout: 'auto' }
          : null;
      return this._compilePipelineAsync(asyncCache, 'compute', id, descriptor, fallbackDescriptor, (pipeline, retried) => {
        const usedFallback = retried && fallbackDescriptor !== null;
        if (usedFallback) {
          this._recordTelemetryWarningOnce(
            `async-auto-layout:compute:${id}:${layoutId}`,
            `compute pipeline ${id}: explicit layout ${layoutId} rejected by async compile; using layout:'auto'`
          );
        }
        this._installComputePipeline(id, pipeline, { ...info, autoLayoutRemap: autoLayoutRemap || usedFallback });
      });
    }
    let pipeline;
    try {
      pipeline = this.device.createComputePipeline(descriptor);
    } catch (err) {
      this._packetError = `compute pipeline ${id}: ${String(err && err.message ? err.message : err)}`;
      return false;
    }
    this._installComputePipeline(id, pipeline, info);
    return true;
  }

  _installComputePipeline(id, pipeline, info) {
    if (info.layoutId === 0 || info.autoLayoutRemap) {
      this.telemetry.autoLayoutFallbacks += 1;
    } else if (info.isSingleGroup0Layout) {
      this.telemetry.rootSigGroup0Layouts += 1;
    }
    this.processor.computePipelines.set(id, pipeline);
    this._invalidateRemappedBindGroupsForPipeline(id, true);
    this.computePipelineMeta.set(id, {
      id,
      csId: info.csId,
      layoutId: info.layoutId,
      entryPoint: info.entryPoint,
      hasComputeAttr: info.hasComputeAttr,
      autoLayoutRemap: info.autoLayoutRemap,
      wgslBindings: info.wgslBindings,
    });
    this.telemetry.computePipelineCount += 1;
  }

  _handleV1CreateComputePipeline(origHandler, view, payloadOffset) {
    const id = view.getUint32(payloadOffset, true);
    const csId = view.getUint32(payloadOffset + 4, true);
    const cs = this._ensureShaderModule(csId);
    let inspected = null;
    if (cs) {
      inspected = this._inspectComputeShaderEntryPoint(cs, csId);
    }
    if (cs && this._handleV1CreateComputePipelineAsync(id, csId, cs, inspected, view, payloadOffset)) {
      return;
    }
    const prevPipeline =
      this.processor && this.processor.computePipelines ? this.processor.computePipelines.get(id) : undefined;
    try {
      origHandler.call(this.processor, view, payloadOffset);
    } catch (err) {
      this._packetError = `compute pipeline ${id}: ${String(err && err.message ? err.message : err)}`;
      this._recordTelemetryWarningOnce(
        `v1-compute-create-fail:${id}`,
        `compute pipeline ${id} creation failed`
      );
      return;
    }
    const nextPipeline =
      this.processor && this.processor.computePipelines ? this.processor.computePipelines.get(id) : undefined;
    if (nextPipeline && nextPipeline !== prevPipeline) {
      this._invalidateRemappedBindGroupsForPipeline(id, true);
      this.computePipelineMeta.set(id, {
        id,
        csId,
        layoutId: 0,
        entryPoint: inspected ? inspected.entryPoint : DEFAULT_ENTRY_POINTS.compute,
        hasComputeAttr: Boolean(inspected && inspected.hasComputeAttr),
        autoLayoutRemap: false,
        wgslBindings: [],
      });
      this.telemetry.computePipelineCount += 1;
    }
  }

  /**
   * W103: compile a v1 CreateComputePipeline through the async cache. Only pipelines with a
   * reflected explicit layout go async (v1 CreateBindGroup resolves their layouts while pending);
   * 'auto'-layout ones stay synchronous. Returns true when the packet was handled here.
   */
  _handleV1CreateComputePipelineAsync(id, csId, cs, inspected, view, payloadOffset) {
    const proc = this.processor;
    const asyncCache = this._asyncPipelineCacheFor('compute');
    if (
      !asyncCache ||
      typeof proc._buildComputePipelineDescriptorV1 !== 'function' ||
      typeof proc._registerComputePipelineV1 !== 'function'
    ) {
      return false;
    }
    let built;
    try {
      built = proc._buildComputePipelineDescriptorV1(view, payloadOffset);
    } catch (err) {
      this._packetError = `compute pipeline ${id}: ${String(err && err.message ? err.message : err)}`;
      this._recordTelemetryWarningOnce(`v1-compute-create-fail:${id}`, `compute pipeline ${id} creation failed`);
      return true;
    }
    if (!built || !built.reflected) return false;
    const csSource = getShaderSourceText(cs);
    const wgslBindings = csSource ? parseWgslGroup0Bindings(csSource) : [];
    const descriptor = built.descriptor;
    const fallbackDescriptor = wgslBindings.length > 0 ? { ...descriptor, layout: 'auto' } : null;
    this._compilePipelineAsync(asyncCache, 'compute', id, descriptor, fallbackDescriptor, (pipeline, retried) => {
      const usedFallback = retried && fallbackDescriptor !== null;
      if (usedFallback) {
        // The reflected layouts no longer describe this pipeline; use its 'auto' layouts.
        if (proc._computePipelineBindGroupLayouts) proc._computePipelineBindGroupLayouts.delete(id);
        this.telemetry.autoLayoutFallbacks += 1;
        this._recordTelemetryWarningOnce(
          `async-auto-layout:compute:${id}:v1`,
          `compute pipeline ${id}: reflected layout rejected by async compile; using layout:'auto'`
        );
      }
      proc._registerComputePipelineV1(id, pipeline);
      this._invalidateRemappedBindGroupsForPipeline(id, true);
      this.computePipelineMeta.set(id, {
        id,
        csId,
        layoutId: 0,
        entryPoint: inspected ? inspected.entryPoint : DEFAULT_ENTRY_POINTS.compute,
        hasComputeAttr: Boolean(inspected && inspected.hasComputeAttr),
        autoLayoutRemap: usedFallback,
        wgslBindings: usedFallback ? wgslBindings : [],
      });
      this.telemetry.computePipelineCount += 1;
    });
    return true;
  }

  _resolveBindGroupDescriptor(bindGroupId) {
    if (!bindGroupId) return null;
    const desc = this.bindGroupDescriptors.get(bindGroupId);
    if (desc) return desc;
    const v1Meta =
      this.processor &&
      this.processor._bindGroupMeta &&
      typeof this.processor._bindGroupMeta.get === 'function'
        ? this.processor._bindGroupMeta.get(bindGroupId)
        : null;
    if (!v1Meta || !Array.isArray(v1Meta.entries)) return null;
    const resources = [];
    for (let i = 0; i < v1Meta.entries.length; i += 1) {
      const e = v1Meta.entries[i];
      if (!e || !e.resource) continue;
      if (e.resource.buffer) {
        const buf = e.resource.buffer;
        let bufferUsage = null;
        if (typeof buf.usage === 'number') {
          bufferUsage = buf.usage;
        } else if (buf.desc && typeof buf.desc.usage === 'number') {
          bufferUsage = buf.desc.usage;
        }
        const kind =
          bufferUsage !== null &&
          (bufferUsage & GPU_BUFFER_USAGE_STORAGE) !== 0 &&
          (bufferUsage & GPU_BUFFER_USAGE_UNIFORM) === 0
            ? 'storage'
            : 'uniform';
        resources.push({ binding: e.binding, kind, resource: e.resource, bufferUsage, isDummy: false });
      } else if (
        e.resource.kind === 'GPUSampler' ||
        (e.resource.desc && ('magFilter' in e.resource.desc || 'compare' in e.resource.desc))
      ) {
        resources.push({
          binding: e.binding,
          kind: 'sampler',
          resource: e.resource,
          comparison: Boolean(e.resource.desc && e.resource.desc.compare),
          isDummy: false,
        });
      } else {
        resources.push({ binding: e.binding, kind: 'texture', resource: e.resource, isDummy: false });
      }
    }
    return { id: bindGroupId, layoutId: v1Meta.layoutId || 0, resources };
  }

  // `dynamicOffsets` are the setBindGroup(0, ..., dynamicOffsets) values (root-constant ring
  // offsets). layout:'auto' bind group layouts never have hasDynamicOffset, so each offset is folded
  // into the static offset of the buffer entry whose dynamicOffsetIndex it belongs to.
  _getOrCreateRemappedBindGroup(
    isCompute,
    pipelineId,
    pipeline,
    wgslBindings,
    bindGroupId,
    bindGroupDesc,
    dynamicOffsets
  ) {
    if (!pipeline || typeof pipeline.getBindGroupLayout !== 'function') {
      return { status: 'unsupported', bindGroup: null };
    }
    if (!Array.isArray(wgslBindings) || wgslBindings.length === 0) {
      return { status: 'empty', bindGroup: null };
    }

    const available =
      bindGroupDesc && Array.isArray(bindGroupDesc.resources) ? bindGroupDesc.resources : [];

    // W186: fast path for 2-binding PresentX plane blits (texture_2d<f32> + sampler) on layout:'auto'
    // render pipelines: reuse the per-(planeTextureId, viewFormat, sampler, pipeline) cache even when
    // bindGroupId rotates every frame.
    if (!isCompute && wgslBindings.length === 2) {
      const texBinding = wgslBindings.find(
        (b) =>
          b &&
          b.kind === 'texture' &&
          !b.multisampled &&
          (b.viewDimension || '2d') === '2d' &&
          (!b.sampleType || b.sampleType === 'float')
      );
      const smpBinding = wgslBindings.find((b) => b && b.kind === 'sampler');
      if (texBinding && smpBinding) {
        const texRes = available.find(
          (r) => r && r.kind === 'texture' && !r.isDummy && Number.isInteger(r.planeTextureId) && r.planeTextureId > 0
        );
        const smpRes = available.find((r) => r && r.kind === 'sampler' && !r.isDummy && r.resource);
        if (texRes) {
          const cachedPx = this.getOrCreatePresentXPlaneBindGroup({
            planeTextureId: texRes.planeTextureId,
            viewDesc: texRes.viewDesc || null,
            viewFormat: texRes.textureFormat || null,
            sampler: smpRes ? smpRes.resource : null,
            samplerId: smpRes && Number.isInteger(smpRes.samplerId) ? smpRes.samplerId : 0,
            samplerKind: smpBinding.comparison ? 'comparison' : 'filtering',
            pipeline,
            pipelineId,
            textureBinding: texBinding.binding,
            samplerBinding: smpBinding.binding,
          });
          if (cachedPx && cachedPx.bindGroup) {
            return { status: 'ok', bindGroup: cachedPx.bindGroup };
          }
        }
      }
    }

    const dynOffsets = normalizeDynamicOffsets(dynamicOffsets);
    const dynOffsetFor = (r) =>
      r &&
      r.hasDynamicOffset &&
      Number.isInteger(r.dynamicOffsetIndex) &&
      r.dynamicOffsetIndex >= 0 &&
      r.dynamicOffsetIndex < dynOffsets.length
        ? dynOffsets[r.dynamicOffsetIndex]
        : 0;
    // Only offsets that land on a real dynamic-offset buffer entry split the cache; all-zero
    // offsets share the plain '<c|r>:<pipeId>:<bgId>' key.
    let dynKey = '';
    if (dynOffsets.some((v) => v > 0)) {
      const parts = [];
      let anyNonZero = false;
      for (let i = 0; i < available.length; i += 1) {
        const r = available[i];
        if (!r || !r.hasDynamicOffset || r.isDummy) continue;
        const v = dynOffsetFor(r);
        if (v > 0) anyNonZero = true;
        parts.push(`${r.binding}=${v}`);
      }
      if (anyNonZero) dynKey = parts.join(',');
    }

    const effectiveBgId = bindGroupId || 0;
    const isLiveDesc =
      effectiveBgId === 0 ||
      (bindGroupDesc && this.bindGroupDescriptors.get(effectiveBgId) === bindGroupDesc);
    const cacheKey = `${isCompute ? 'c' : 'r'}:${pipelineId}:${effectiveBgId}${dynKey ? `@${dynKey}` : ''}`;
    if (isLiveDesc && this.remappedBindGroups.has(cacheKey)) {
      const cached = this.remappedBindGroups.get(cacheKey);
      this._cacheRemappedBindGroup(cacheKey, cached);
      return cached ? { status: 'ok', bindGroup: cached } : { status: 'empty', bindGroup: null };
    }

    let autoBgl;
    try {
      autoBgl = pipeline.getBindGroupLayout(0);
    } catch (_) {
      if (isLiveDesc) this._cacheRemappedBindGroup(cacheKey, null);
      return { status: 'empty', bindGroup: null };
    }
    if (!autoBgl) {
      if (isLiveDesc) this._cacheRemappedBindGroup(cacheKey, null);
      return { status: 'empty', bindGroup: null };
    }

    const usedIndices = new Set();

    // Folds r's dynamic offset into a static { buffer, offset, size } binding, clamped so that
    // offset + size never exceeds the GPUBuffer size; an out-of-bounds offset binds a dummy.
    const bufferResourceWithDynamicOffset = (r, b) => {
      const dyn = dynOffsetFor(r);
      if (dyn <= 0) return r.resource;
      const base = r.resource;
      const effectiveOffset = (Number(base.offset) || 0) + dyn;
      const bufferSize =
        typeof r.bufferSize === 'number' && Number.isFinite(r.bufferSize) && r.bufferSize > 0
          ? r.bufferSize
          : base.buffer && typeof base.buffer.size === 'number' && base.buffer.size > 0
            ? base.buffer.size
            : null;
      let size = typeof base.size === 'number' && base.size > 0 ? base.size : null;
      if (bufferSize !== null) {
        if (effectiveOffset >= bufferSize) {
          this.telemetry.dynamicOffsetClamps += 1;
          this._recordTelemetryWarningOnce(
            `remap-dyn-offset-oob:${effectiveBgId}:${r.binding}`,
            `auto-layout remap: bind group ${effectiveBgId} binding ${r.binding} dynamic offset ${dyn} exceeds buffer size ${bufferSize}; using dummy buffer`
          );
          return this._dummyResource(synthLayoutEntryForWgslBinding(b));
        }
        const remaining = bufferSize - effectiveOffset;
        if (size !== null && size > remaining) {
          size = remaining;
          this.telemetry.dynamicOffsetClamps += 1;
          this._recordTelemetryWarningOnce(
            `remap-dyn-offset-clamp:${effectiveBgId}:${r.binding}`,
            `auto-layout remap: bind group ${effectiveBgId} binding ${r.binding} size clamped to ${remaining} at dynamic offset ${dyn}`
          );
        }
      }
      this.telemetry.dynamicOffsetRemaps += 1;
      const out = { buffer: base.buffer, offset: effectiveOffset };
      if (size !== null) out.size = size;
      return out;
    };

    const isBufferBindingResource = (res) =>
      Boolean(res && typeof res === 'object' && 'buffer' in res && res.buffer);
    const isNonBufferBindingResource = (res) =>
      Boolean(res && !(typeof res === 'object' && 'buffer' in res));
    const isSamplerResource = (res) =>
      Boolean(
        res &&
          ((typeof GPUSampler === 'function' && res instanceof GPUSampler) ||
            (typeof res === 'object' && res.kind === 'GPUSampler'))
      );
    const isTextureViewResource = (res) =>
      Boolean(
        res &&
          ((typeof GPUTextureView === 'function' && res instanceof GPUTextureView) ||
            (typeof res === 'object' && res.kind === 'GPUTextureView'))
      );

    const pickResourceForBinding = (b) => {
      for (let pass = 0; pass < 2; pass += 1) {
        for (let i = 0; i < available.length; i += 1) {
          if (usedIndices.has(i)) continue;
          const r = available[i];
          if (!r || !r.resource || r.isDummy) continue;
          if (b.kind === 'uniform') {
            if (!isBufferBindingResource(r.resource)) continue;
            const isBuf =
              r.kind === 'uniform' || r.kind === 'storage' || r.kind === 'read-only-storage';
            if (!isBuf) continue;
            if (pass === 0 && r.kind !== 'uniform') continue;
            if (r.bufferUsage !== null && r.bufferUsage !== undefined) {
              if ((r.bufferUsage & GPU_BUFFER_USAGE_UNIFORM) === 0) continue;
            } else if (r.kind !== 'uniform') {
              continue;
            }
            usedIndices.add(i);
            return bufferResourceWithDynamicOffset(r, b);
          }
          if (b.kind === 'read-only-storage' || b.kind === 'storage') {
            if (!isBufferBindingResource(r.resource)) continue;
            const isBuf =
              r.kind === 'uniform' || r.kind === 'storage' || r.kind === 'read-only-storage';
            if (!isBuf) continue;
            if (
              pass === 0 &&
              !(r.kind === b.kind || (b.kind === 'read-only-storage' && r.kind === 'storage'))
            ) {
              continue;
            }
            if (r.bufferUsage !== null && r.bufferUsage !== undefined) {
              if ((r.bufferUsage & GPU_BUFFER_USAGE_STORAGE) === 0) continue;
            } else if (
              !(r.kind === 'storage' || (b.kind === 'read-only-storage' && r.kind === 'read-only-storage'))
            ) {
              continue;
            }
            usedIndices.add(i);
            return bufferResourceWithDynamicOffset(r, b);
          }
          if (pass > 0) continue;
          if (b.kind === 'texture') {
            if (r.kind !== 'texture') continue;
            if (!isNonBufferBindingResource(r.resource) || isSamplerResource(r.resource)) continue;
            if (
              r.textureUsage !== null &&
              r.textureUsage !== undefined &&
              (r.textureUsage & GPU_TEXTURE_USAGE_TEXTURE_BINDING) === 0
            ) {
              continue;
            }
            const expectedDim = b.viewDimension || '2d';
            if (r.viewDimension && r.viewDimension !== expectedDim) continue;
            if (
              r.multisampled !== undefined &&
              r.multisampled !== null &&
              Boolean(r.multisampled) !== Boolean(b.multisampled)
            ) {
              continue;
            }
            if (r.textureFormat) {
              const expectedSampleType = b.sampleType || 'float';
              const actualSampleType = resolveTextureSampleType(r.textureFormat, this.device);
              if (expectedSampleType === 'unfilterable-float') {
                if (actualSampleType !== 'float' && actualSampleType !== 'unfilterable-float') {
                  continue;
                }
              } else if (actualSampleType !== expectedSampleType) {
                continue;
              }
            }
            usedIndices.add(i);
            return r.resource;
          }
          if (b.kind === 'storage-texture') {
            if (r.kind !== 'storage-texture') continue;
            if (!isNonBufferBindingResource(r.resource) || isSamplerResource(r.resource)) continue;
            if (
              r.textureUsage !== null &&
              r.textureUsage !== undefined &&
              (r.textureUsage & GPU_TEXTURE_USAGE_STORAGE_BINDING) === 0
            ) {
              continue;
            }
            const expectedDim = b.viewDimension || '2d';
            if (r.viewDimension && r.viewDimension !== expectedDim) continue;
            if (r.textureFormat && b.storageFormat && r.textureFormat !== b.storageFormat) {
              continue;
            }
            usedIndices.add(i);
            return r.resource;
          }
          if (b.kind === 'sampler') {
            if (r.kind !== 'sampler') continue;
            if (!isNonBufferBindingResource(r.resource) || isTextureViewResource(r.resource)) {
              continue;
            }
            if (
              r.comparison !== undefined &&
              r.comparison !== null &&
              Boolean(r.comparison) !== Boolean(b.comparison)
            ) {
              continue;
            }
            usedIndices.add(i);
            return r.resource;
          }
        }
      }
      if (b.kind === 'sampler') {
        const samplerKind = b.comparison ? 'comparison' : 'filtering';
        const fallbackSampler = this._resolveDefaultSamplerFromCache(samplerKind);
        if (fallbackSampler) {
          this.telemetry.fallbackSamplersBound += 1;
          return fallbackSampler;
        }
      }
      return this._dummyResource(synthLayoutEntryForWgslBinding(b));
    };

    const entries = wgslBindings.map((b) => ({
      binding: b.binding,
      resource: pickResourceForBinding(b),
    }));

    let remapped = null;
    try {
      remapped = this.device.createBindGroup({ layout: autoBgl, entries });
    } catch (_) {
      try {
        const fallbackEntries = wgslBindings.map((b) => {
          if (b.kind === 'sampler') {
            const samplerKind = b.comparison ? 'comparison' : 'filtering';
            const fallbackSampler = this._resolveDefaultSamplerFromCache(samplerKind);
            if (fallbackSampler) {
              this.telemetry.fallbackSamplersBound += 1;
              return { binding: b.binding, resource: fallbackSampler };
            }
          }
          return {
            binding: b.binding,
            resource: this._dummyResource(synthLayoutEntryForWgslBinding(b)),
          };
        });
        remapped = this.device.createBindGroup({ layout: autoBgl, entries: fallbackEntries });
      } catch (_err2) {
        remapped = null;
      }
    }

    if (isLiveDesc) {
      this._cacheRemappedBindGroup(cacheKey, remapped);
    }
    return remapped ? { status: 'ok', bindGroup: remapped } : { status: 'empty', bindGroup: null };
  }

  // Guards queued compute dispatches against zero/oversized workgroup dimensions and missing
  // compute pipelines, and remaps @group(0) bind groups for pipelines that fell back to layout:'auto',
  // immediately before WgpuCommandProcessor._flushPresent() executes them.
  _preparePassCommandsForFlush() {
    this._syncDeviceGeneration();
    const cmds = this.processor && this.processor._passCommands;
    if (!Array.isArray(cmds) || cmds.length === 0) return;

    let activePassMode = null; // 'render' | 'compute' | null
    let activeComputePipelineId = null;
    let activeComputePipelineValid = false;
    let activeComputeAutoLayout = null;
    let hadExplicitSetComputePipeline = false;
    let group0BoundForCompute = false;
    let lastComputeGroup0 = null;

    let activeRenderPipelineId = null;
    let activeRenderAutoLayout = null;
    let group0BoundForRender = false;
    let lastRenderGroup0 = null;
    // W103: the bound pipeline is still compiling asynchronously; skip its draws/dispatches.
    let activeRenderPending = false;
    let activeComputePending = false;

    let modified = false;
    const filtered = [];

    const makeRemappedSetBindGroupCmd = (bindGroup, bindGroupId) => ({
      type: 'renderState',
      run: (ctl) => {
        if (ctl.pass) ctl.pass.setBindGroup(0, bindGroup);
      },
      _wgpuMeta: {
        kind: 'setBindGroup2',
        groupIndex: 0,
        bindGroupId: bindGroupId || 0,
      },
    });

    const makeNoopRenderGroup0Cmd = (bindGroupId) => ({
      type: 'renderState',
      run: () => {},
      _wgpuMeta: {
        kind: 'setBindGroup2',
        groupIndex: 0,
        bindGroupId: bindGroupId || 0,
      },
    });

    for (let i = 0; i < cmds.length; i += 1) {
      const cmd = cmds[i];
      if (!cmd) continue;

      const passOpKind =
        (cmd._wgpuMeta && cmd._wgpuMeta.kind) || cmd._rsPassOp || null;
      if (passOpKind === 'beginRenderPass' || passOpKind === 'endRenderPass') {
        activePassMode = passOpKind === 'beginRenderPass' ? 'render' : null;
        activeComputePipelineId = null;
        activeComputePipelineValid = false;
        activeComputeAutoLayout = null;
        hadExplicitSetComputePipeline = false;
        group0BoundForCompute = false;
        lastComputeGroup0 = null;
        activeRenderPipelineId = null;
        activeRenderAutoLayout = null;
        group0BoundForRender = false;
        lastRenderGroup0 = null;
        activeRenderPending = false;
        activeComputePending = false;
        filtered.push(cmd);
        continue;
      }

      if (cmd.type === 'setPipeline') {
        activePassMode = 'render';
        activeComputePipelineId = null;
        activeComputePipelineValid = false;
        activeComputeAutoLayout = null;
        hadExplicitSetComputePipeline = false;
        group0BoundForCompute = false;
        lastComputeGroup0 = null;
        activeComputePending = false;
        activeRenderPipelineId = cmd.pipelineId;
        activeRenderPending = this.isAsyncPipelinePending(cmd.pipelineId, false);
        if (activeRenderPending) {
          // Never bind an undefined pipeline; its draws are dropped until the compile resolves.
          activeRenderAutoLayout = null;
          group0BoundForRender = false;
          modified = true;
          continue;
        }
        const rMeta = this.renderPipelineAutoLayoutMeta.get(cmd.pipelineId) || null;
        activeRenderAutoLayout = rMeta && rMeta.autoLayoutRemap ? rMeta : null;
        group0BoundForRender = false;
        filtered.push(cmd);
        continue;
      }

      if (cmd.type === 'setComputePipeline') {
        activePassMode = 'compute';
        activeRenderPipelineId = null;
        activeRenderAutoLayout = null;
        group0BoundForRender = false;
        lastRenderGroup0 = null;
        activeRenderPending = false;
        hadExplicitSetComputePipeline = true;
        activeComputePipelineId = cmd.pipelineId;
        activeComputePending = this.isAsyncPipelinePending(cmd.pipelineId, true);
        if (activeComputePending) {
          activeComputePipelineValid = false;
          activeComputeAutoLayout = null;
          group0BoundForCompute = false;
          modified = true;
          continue;
        }
        activeComputePipelineValid = Boolean(
          this.processor.computePipelines && this.processor.computePipelines.has(cmd.pipelineId)
        );
        if (!activeComputePipelineValid) {
          activeComputeAutoLayout = null;
          group0BoundForCompute = false;
          this._recordTelemetryWarningOnce(
            `missing-compute-pipeline:${cmd.pipelineId}`,
            `compute pipeline ${cmd.pipelineId} unavailable; subsequent dispatches will be skipped`
          );
          modified = true;
          continue;
        }
        const cMeta = this.computePipelineMeta.get(cmd.pipelineId) || null;
        activeComputeAutoLayout = cMeta && cMeta.autoLayoutRemap ? cMeta : null;
        group0BoundForCompute = false;
        filtered.push(cmd);
        continue;
      }

      const isSetBgV1 = cmd.type === 'setBindGroup';
      const isSetBgV2 = Boolean(cmd._rsSetBindGroup || passOpKind === 'setBindGroup2');
      if (isSetBgV1 || isSetBgV2) {
        let groupIndex;
        let bindGroupId;
        let bindGroupDesc;
        let dynamicOffsets;
        if (cmd._rsSetBindGroup) {
          groupIndex = cmd._rsSetBindGroup.groupIndex;
          bindGroupId = cmd._rsSetBindGroup.bindGroupId;
          bindGroupDesc =
            cmd._rsSetBindGroup.bindGroupDesc || this._resolveBindGroupDescriptor(bindGroupId);
          dynamicOffsets = normalizeDynamicOffsets(
            cmd._rsSetBindGroup.offsets || cmd._rsSetBindGroup.dynamicOffsets
          );
        } else if (isSetBgV1) {
          groupIndex = cmd.groupIndex;
          bindGroupId = cmd.bindGroupId;
          bindGroupDesc = this._resolveBindGroupDescriptor(bindGroupId);
          dynamicOffsets = normalizeDynamicOffsets(cmd.dynamicOffsets || cmd.offsets);
        } else {
          groupIndex = cmd._wgpuMeta.groupIndex;
          bindGroupId = cmd._wgpuMeta.bindGroupId;
          bindGroupDesc = this._resolveBindGroupDescriptor(bindGroupId);
          dynamicOffsets = normalizeDynamicOffsets(
            cmd._wgpuMeta.dynamicOffsets || cmd._wgpuMeta.offsets
          );
        }

        if (groupIndex === 0) {
          if (activePassMode === 'compute') {
            lastComputeGroup0 = { bindGroupId, bindGroupDesc, dynamicOffsets };
            if (activeComputePipelineValid && activeComputeAutoLayout) {
              const pipeline = this.processor.computePipelines.get(activeComputePipelineId);
              const res = this._getOrCreateRemappedBindGroup(
                true,
                activeComputePipelineId,
                pipeline,
                activeComputeAutoLayout.wgslBindings,
                bindGroupId,
                bindGroupDesc,
                dynamicOffsets
              );
              group0BoundForCompute = true;
              if (res.status === 'ok') {
                filtered.push(makeRemappedSetBindGroupCmd(res.bindGroup, bindGroupId));
                modified = true;
                continue;
              }
              if (res.status === 'empty') {
                modified = true;
                continue;
              }
            }
          } else if (activePassMode === 'render') {
            lastRenderGroup0 = { bindGroupId, bindGroupDesc, dynamicOffsets };
            if (activeRenderAutoLayout) {
              const pipeline = this.processor.pipelines.get(activeRenderPipelineId);
              const res = this._getOrCreateRemappedBindGroup(
                false,
                activeRenderPipelineId,
                pipeline,
                activeRenderAutoLayout.wgslBindings,
                bindGroupId,
                bindGroupDesc,
                dynamicOffsets
              );
              group0BoundForRender = true;
              if (res.status === 'ok') {
                filtered.push(makeRemappedSetBindGroupCmd(res.bindGroup, bindGroupId));
                modified = true;
                continue;
              }
              if (res.status === 'empty') {
                filtered.push(makeNoopRenderGroup0Cmd(bindGroupId));
                modified = true;
                continue;
              }
            }
          } else {
            lastRenderGroup0 = { bindGroupId, bindGroupDesc, dynamicOffsets };
            lastComputeGroup0 = { bindGroupId, bindGroupDesc, dynamicOffsets };
          }
        }
        filtered.push(cmd);
        continue;
      }

      if (cmd.type === 'draw' || cmd.type === 'drawIndexed') {
        activePassMode = 'render';
        activeComputePipelineId = null;
        activeComputePipelineValid = false;
        activeComputeAutoLayout = null;
        hadExplicitSetComputePipeline = false;
        group0BoundForCompute = false;
        lastComputeGroup0 = null;
        activeComputePending = false;
        if (activeRenderPending) {
          this.telemetry.pendingAsyncPipelineSkips += 1;
          modified = true;
          continue;
        }
        if (activeRenderAutoLayout && !group0BoundForRender) {
          const pipeline = this.processor.pipelines.get(activeRenderPipelineId);
          const src = lastRenderGroup0 || { bindGroupId: 0, bindGroupDesc: null, dynamicOffsets: [] };
          const res = this._getOrCreateRemappedBindGroup(
            false,
            activeRenderPipelineId,
            pipeline,
            activeRenderAutoLayout.wgslBindings,
            src.bindGroupId,
            src.bindGroupDesc,
            src.dynamicOffsets
          );
          group0BoundForRender = true;
          if (res.status === 'ok') {
            filtered.push(makeRemappedSetBindGroupCmd(res.bindGroup, src.bindGroupId));
            modified = true;
          } else if (res.status === 'empty') {
            filtered.push(makeNoopRenderGroup0Cmd(src.bindGroupId));
            modified = true;
          }
        }
        filtered.push(cmd);
        continue;
      }

      if (cmd.type === 'dispatchCompute') {
        if (activeComputePending) {
          this.telemetry.pendingAsyncPipelineSkips += 1;
          modified = true;
          continue;
        }
        if (hadExplicitSetComputePipeline && !activeComputePipelineValid) {
          this.telemetry.skippedComputeDispatchCount += 1;
          this._recordTelemetryWarningOnce(
            `skip-dispatch-no-pipeline:${activeComputePipelineId}`,
            `skipped compute dispatch: compute pipeline ${activeComputePipelineId} is unavailable`
          );
          modified = true;
          continue;
        }
        const rawX = cmd.workgroupX;
        const rawY = cmd.workgroupY;
        const rawZ = cmd.workgroupZ;
        const check = validateComputeDispatch(rawX, rawY, rawZ, this.device);
        if (check.skip) {
          this.telemetry.skippedComputeDispatchCount += 1;
          this._recordTelemetryWarningOnce(
            `skip-dispatch-zero:${rawX}x${rawY}x${rawZ}`,
            `skipped compute dispatch with zero or invalid workgroups (${rawX}x${rawY}x${rawZ})`
          );
          modified = true;
          continue;
        }
        if (check.clamped) {
          cmd.workgroupX = check.x;
          cmd.workgroupY = check.y;
          cmd.workgroupZ = check.z;
          this.telemetry.clampedComputeDispatchCount += 1;
          this._recordTelemetryWarningOnce(
            `clamp-dispatch:${rawX}x${rawY}x${rawZ}:${check.maxWorkgroupsPerDimension}`,
            `clamped compute dispatch from ${rawX}x${rawY}x${rawZ} to ${check.x}x${check.y}x${check.z} (limit=${check.maxWorkgroupsPerDimension})`
          );
        }
        if (activeComputeAutoLayout && !group0BoundForCompute) {
          const pipeline = this.processor.computePipelines.get(activeComputePipelineId);
          const src = lastComputeGroup0 || { bindGroupId: 0, bindGroupDesc: null, dynamicOffsets: [] };
          const res = this._getOrCreateRemappedBindGroup(
            true,
            activeComputePipelineId,
            pipeline,
            activeComputeAutoLayout.wgslBindings,
            src.bindGroupId,
            src.bindGroupDesc,
            src.dynamicOffsets
          );
          group0BoundForCompute = true;
          if (res.status === 'ok') {
            filtered.push(makeRemappedSetBindGroupCmd(res.bindGroup, src.bindGroupId));
            modified = true;
          }
        }
        this.telemetry.computeDispatchCount += 1;
        filtered.push(cmd);
        continue;
      }

      filtered.push(cmd);
    }

    if (modified) {
      this.processor._passCommands = filtered;
    }
  }

  // ---- Textures and views ----

  _createTexture2(view, offset, payloadBytes) {
    const decoded = decodeCreateTexture2(view, offset, payloadBytes);
    if (decoded.error) {
      this._packetError = decoded.error;
      return false;
    }
    const sanitized = sanitizeTextureDescriptor(
      decoded.descriptor,
      this.device,
      this._textureFormatsOptions
    );
    for (let i = 0; i < sanitized.warnings.length; i += 1) {
      const w = sanitized.warnings[i];
      this._recordTelemetryWarningOnce(w.key, w.message);
    }
    if (sanitized.bcFallback) {
      this.telemetry.bcFormatFallbacks += 1;
    }
    if (sanitized.strippedUsages.length > 0) {
      this.telemetry.usageStrips += sanitized.strippedUsages.length;
    }

    const id = decoded.id;
    if (typeof this.processor._destroyTextureIfPresent === 'function') {
      this.processor._destroyTextureIfPresent(id);
    }
    this._invalidatePresentXCacheForResource(id);
    this._bumpTextureGeneration(id);
    this._textureViewOwners.delete(id);
    this.textureViews.delete(id);
    this.textureMeta.delete(id);

    let texture;
    try {
      texture = this.device.createTexture(sanitized.descriptor);
    } catch (err) {
      this._packetError = `CreateTexture2 ${id}: ${String(err && err.message ? err.message : err)}`;
      return false;
    }
    this.processor.textures.set(id, texture);
    this._textureViewOwners.set(id, texture);
    let defaultView = null;
    if (texture && typeof texture.createView === 'function') {
      try {
        defaultView = texture.createView();
      } catch (_) {
        defaultView = null;
      }
      if (defaultView && this.processor.textureViews) {
        this.processor.textureViews.set(id, defaultView);
      }
    }
    if (this.processor.resourceRegistry && typeof this.processor.resourceRegistry.register === 'function') {
      this.processor.resourceRegistry.register('texture', id, texture, {
        byteSize: 0,
        label: `texture:${id}`,
      });
      if (defaultView) {
        this.processor.resourceRegistry.register('textureView', id, defaultView, {
          byteSize: 0,
          label: `textureView:${id}`,
        });
      }
    }
    this.textureMeta.set(id, {
      id,
      requestedFormat: sanitized.originalFormat,
      format: sanitized.effectiveFormat,
      bcFallback: sanitized.bcFallback,
      formatFallback: sanitized.formatFallback,
      usage: sanitized.descriptor.usage,
      dimension: sanitized.descriptor.dimension,
      sampleCount: sanitized.descriptor.sampleCount,
      width: sanitized.descriptor.size.width,
      height: sanitized.descriptor.size.height,
      depthOrArrayLayers: sanitized.descriptor.size.depthOrArrayLayers,
    });
    this.telemetry.texture2Count += 1;
    return true;
  }

  _resolveTextureFormat(textureId, viewDesc) {
    if (viewDesc && typeof viewDesc.format === 'string' && viewDesc.format !== 'undefined') {
      const meta = this.textureMeta.get(textureId);
      if (meta && meta.bcFallback && isBcCompressedFormat(viewDesc.format)) {
        return meta.format;
      }
      if (
        meta &&
        meta.formatFallback &&
        viewDesc.format === 'depth32float-stencil8' &&
        meta.format === 'depth24plus-stencil8'
      ) {
        return meta.format;
      }
      return viewDesc.format;
    }
    const meta = this.textureMeta.get(textureId);
    if (meta && meta.format) return meta.format;
    const texture = this.processor.textures.get(textureId);
    if (!texture) return null;
    if (typeof texture.format === 'string') return texture.format;
    if (texture.desc && typeof texture.desc.format === 'string') return texture.desc.format;
    return null;
  }

  _textureView(textureId, desc) {
    this._syncDeviceGeneration();
    const texture = this.processor.textures.get(textureId);
    if (!texture || typeof texture.createView !== 'function') return null;
    this._getTextureGeneration(textureId, texture);
    let effectiveDesc = desc;
    const meta = this.textureMeta.get(textureId);
    if (effectiveDesc && typeof effectiveDesc.format === 'string' && meta) {
      if (meta.bcFallback && isBcCompressedFormat(effectiveDesc.format)) {
        effectiveDesc = { ...effectiveDesc, format: meta.format };
      } else if (
        meta.formatFallback &&
        effectiveDesc.format === 'depth32float-stencil8' &&
        meta.format === 'depth24plus-stencil8'
      ) {
        effectiveDesc = { ...effectiveDesc, format: meta.format };
      }
    }
    const key = buildTextureViewKey(effectiveDesc);
    let views = this.textureViews.get(textureId);
    if (!views) {
      views = new Map();
      this.textureViews.set(textureId, views);
    }
    let textureView = views.get(key);
    if (!textureView) {
      try {
        textureView = texture.createView(effectiveDesc);
      } catch (err) {
        this._packetError = `texture ${textureId} createView: ${String(
          err && err.message ? err.message : err
        )}`;
        return null;
      }
      views.set(key, textureView);
    }
    return textureView;
  }

  // Attachment view resolver; the canvas texture only exists at Present (ctl.swapchainView).
  _attachmentView(textureId, desc) {
    if (textureId === SWAPCHAIN_TEXTURE_ID) {
      const requestedFormat =
        desc && typeof desc.format === 'string' && desc.format !== 'undefined' ? desc.format : null;
      return (ctl) => {
        if (
          requestedFormat &&
          ctl &&
          ctl.swapchainTexture &&
          typeof ctl.swapchainTexture.createView === 'function' &&
          this.processor &&
          this.processor._swapchainViewFormat &&
          this.processor._swapchainViewFormat !== requestedFormat &&
          (!this.processor._swapchain || !this.processor._swapchain.usesBlitFallback)
        ) {
          try {
            return ctl.swapchainTexture.createView({ format: requestedFormat });
          } catch (_) {
            // Fall back to ctl.swapchainView if the swapchain texture does not support requestedFormat.
          }
        }
        return ctl.swapchainView;
      };
    }
    const textureView = this._textureView(textureId, desc);
    return textureView ? () => textureView : null;
  }

  _textureAttachmentSize(textureId, viewDesc) {
    const textures = this.processor && this.processor.textures;
    const texture = textures && typeof textures.get === 'function' ? textures.get(textureId) : null;
    let base = texture ? validTargetSize(texture.width, texture.height) : null;
    if (!base) {
      const meta = this.textureMeta.get(textureId);
      if (meta) {
        base = validTargetSize(meta.width, meta.height);
      }
    }
    if (!base) return null;
    const mip =
      viewDesc && typeof viewDesc.baseMipLevel === 'number' && viewDesc.baseMipLevel > 0
        ? Math.floor(viewDesc.baseMipLevel)
        : 0;
    if (mip === 0) return base;
    const scale = mip >= 31 ? Infinity : Math.pow(2, mip);
    return {
      width: scale === Infinity ? 1 : Math.max(1, Math.floor(base.width / scale)),
      height: scale === Infinity ? 1 : Math.max(1, Math.floor(base.height / scale)),
    };
  }

  _resolveSwapchainSize(ctl) {
    if (ctl) {
      if (ctl.swapchainTexture) {
        const fromTex = validTargetSize(ctl.swapchainTexture.width, ctl.swapchainTexture.height);
        if (fromTex) return fromTex;
      }
      const fromCtl =
        validTargetSize(ctl.swapchainWidth, ctl.swapchainHeight) ||
        validTargetSize(ctl.canvasWidth, ctl.canvasHeight);
      if (fromCtl) return fromCtl;
      if (typeof ctl.getCanvasTexture === 'function') {
        try {
          const tex = ctl.getCanvasTexture();
          if (tex) {
            const fromGetter = validTargetSize(tex.width, tex.height);
            if (fromGetter) return fromGetter;
          }
        } catch (_) {
          // Fall through to remaining swapchain size probes.
        }
      }
      if (ctl.swapchainView) {
        const fromView =
          validTargetSize(ctl.swapchainView.width, ctl.swapchainView.height) ||
          (ctl.swapchainView.texture &&
            validTargetSize(ctl.swapchainView.texture.width, ctl.swapchainView.texture.height));
        if (fromView) return fromView;
      }
    }
    const proc = this.processor;
    if (proc) {
      if (proc._swapchain && !proc._swapchain.isDestroyed) {
        const fromSwapchain = validTargetSize(proc._swapchain.width, proc._swapchain.height);
        if (fromSwapchain) return fromSwapchain;
      }
      if (proc.canvas) {
        const fromProcCanvas = validTargetSize(proc.canvas.width, proc.canvas.height);
        if (fromProcCanvas) return fromProcCanvas;
      }
      if (proc.ctx) {
        if (proc.ctx.canvas) {
          const fromCtxCanvas = validTargetSize(proc.ctx.canvas.width, proc.ctx.canvas.height);
          if (fromCtxCanvas) return fromCtxCanvas;
        }
        if (typeof proc.ctx.getCurrentTexture === 'function') {
          try {
            const current = proc.ctx.getCurrentTexture();
            if (current) {
              const fromCurrent = validTargetSize(current.width, current.height);
              if (fromCurrent) return fromCurrent;
            }
          } catch (_) {
            // Fall back to null when swapchain texture dimensions cannot be queried.
          }
        }
      }
    }
    return null;
  }

  _getActivePassTargetSize(ctl) {
    if (this._activePass !== null && ctl.pass === this._activePass) {
      return this._activePassTargetSize;
    }
    this._activePass = null;
    this._activePassTargetSize = null;
    return this._resolveSwapchainSize(ctl);
  }

  // The exact GPURenderPassDescriptor for a decoded BeginRenderPass (views resolved via `ctl`).
  buildRenderPassDescriptor(decoded, resolvers, ctl) {
    const colorSizes = [];
    const colorAttachments = decoded.colors.map((c, i) => {
      if (!c) {
        colorSizes.push(null);
        return null;
      }
      colorSizes.push(
        c.textureId === SWAPCHAIN_TEXTURE_ID
          ? this._resolveSwapchainSize(ctl)
          : this._textureAttachmentSize(c.textureId, c.view)
      );
      const loadOp = c.loadOp === 'clear' || c.loadOp === 'load' ? c.loadOp : 'load';
      const storeOp = c.storeOp === 'discard' || c.storeOp === 'store' ? c.storeOp : 'store';
      const attachment = { view: resolvers.colors[i](ctl), loadOp, storeOp };
      if (loadOp === 'clear') {
        attachment.clearValue = {
          r: finiteOr(c.clearValue && c.clearValue.r, 0) || 0,
          g: finiteOr(c.clearValue && c.clearValue.g, 0) || 0,
          b: finiteOr(c.clearValue && c.clearValue.b, 0) || 0,
          a: finiteOr(c.clearValue && c.clearValue.a, 1) || 0,
        };
      }
      return attachment;
    });
    const descriptor = { colorAttachments };
    const d = decoded.depthStencil;
    let depthStencilSize = null;
    if (d) {
      depthStencilSize =
        d.textureId === SWAPCHAIN_TEXTURE_ID
          ? this._resolveSwapchainSize(ctl)
          : this._textureAttachmentSize(d.textureId, d.view);
      const ds = { view: resolvers.depthStencil(ctl) };
      if (d.depthLoadOp !== 0) {
        ds.depthLoadOp = nameOf(LOAD_OP_NAMES, d.depthLoadOp);
        ds.depthStoreOp = nameOf(STORE_OP_NAMES, d.depthStoreOp);
        if (ds.depthLoadOp === 'clear') ds.depthClearValue = d.depthClearValue;
      }
      if (d.stencilLoadOp !== 0) {
        ds.stencilLoadOp = nameOf(LOAD_OP_NAMES, d.stencilLoadOp);
        ds.stencilStoreOp = nameOf(STORE_OP_NAMES, d.stencilStoreOp);
        if (ds.stencilLoadOp === 'clear') ds.stencilClearValue = d.stencilClearValue;
      }
      if (d.depthReadOnly) ds.depthReadOnly = true;
      if (d.stencilReadOnly) ds.stencilReadOnly = true;
      descriptor.depthStencilAttachment = ds;
    }
    const sanitized = sanitizeRenderPassAttachments(descriptor, {
      colors: colorSizes,
      depthStencil: depthStencilSize,
    });
    this._lastPassReferenceSize = sanitized.referenceSize;
    if (sanitized.sanitizedCount > 0) {
      this.telemetry.sanitizedAttachments += sanitized.sanitizedCount;
      for (let i = 0; i < sanitized.warnings.length; i += 1) {
        const w = sanitized.warnings[i];
        this._recordTelemetryWarningOnce(w.key, w.message);
      }
    }
    return sanitized.descriptor;
  }

  // W104: report a missing render-pass attachment once per texture id and device generation
  // instead of once per frame.
  _missingRenderPassTexture(textureId) {
    this._packetError = `render pass: unknown texture ${textureId}`;
    const generation =
      this.processor && Number.isInteger(this.processor._deviceGeneration)
        ? this.processor._deviceGeneration
        : 0;
    const key = `${generation}:${textureId}`;
    if (this._reportedMissingTextures.has(key)) {
      this._suppressPacketError = true;
    } else {
      if (this._reportedMissingTextures.size >= MISSING_TEXTURE_REPORT_LIMIT) {
        this._reportedMissingTextures.clear();
      }
      this._reportedMissingTextures.add(key);
    }
    return false;
  }

  _beginRenderPass(view, offset) {
    const decoded = decodeBeginRenderPass(view, offset);
    const resolvers = { colors: [], depthStencil: null };
    for (const c of decoded.colors) {
      if (
        c &&
        c.textureId === SWAPCHAIN_TEXTURE_ID &&
        c.view &&
        c.view.format &&
        c.view.format !== 'undefined' &&
        typeof this.processor.setSwapchainFormat === 'function'
      ) {
        // #709: match canvas swapchain format to the stream's declared attachment format.
        this.processor.setSwapchainFormat(c.view.format);
      }
      const resolver = c ? this._attachmentView(c.textureId, c.view) : null;
      if (c && !resolver) {
        return this._missingRenderPassTexture(c.textureId);
      }
      resolvers.colors.push(resolver);
    }
    if (decoded.depthStencil) {
      resolvers.depthStencil = this._attachmentView(decoded.depthStencil.textureId, decoded.depthStencil.view);
      if (!resolvers.depthStencil) {
        return this._missingRenderPassTexture(decoded.depthStencil.textureId);
      }
    }
    const targetAttachment = decoded.colors.find((c) => c !== null) || decoded.depthStencil || null;
    let resolveTargetSize = null;
    if (targetAttachment) {
      if (targetAttachment.textureId === SWAPCHAIN_TEXTURE_ID) {
        resolveTargetSize = (ctl) => this._resolveSwapchainSize(ctl);
      } else {
        const staticSize = this._textureAttachmentSize(targetAttachment.textureId, targetAttachment.view);
        resolveTargetSize = () => staticSize;
      }
    }
    this.processor._passCommands.push({
      type: 'renderState',
      _rsPassOp: 'beginRenderPass',
      _wgpuMeta: { kind: 'beginRenderPass' },
      run: (ctl) => {
        const desc = this.buildRenderPassDescriptor(decoded, resolvers, ctl);
        const pass = ctl.beginRenderPass(desc);
        this._activePass = pass || ctl.pass || null;
        this._activePassTargetSize =
          this._lastPassReferenceSize || (resolveTargetSize ? resolveTargetSize(ctl) : null);
      },
    });
    return true;
  }

  // ---- Bind groups ----

  _dummyResource(layoutEntry) {
    if (layoutEntry.buffer) {
      const size = Math.max(256, Math.ceil(layoutEntry.buffer.minBindingSize / 4) * 4);
      const key = `buffer:${size}`;
      if (!this.dummies.has(key)) {
        this.dummies.set(key, this.device.createBuffer({ size, usage: GPU_BUFFER_USAGE_UNIFORM_STORAGE_COPY_DST }));
      }
      return { buffer: this.dummies.get(key) };
    }
    if (layoutEntry.sampler) {
      const key = `sampler:${layoutEntry.sampler.type}`;
      if (!this.dummies.has(key)) {
        const desc = layoutEntry.sampler.type === 'comparison' ? { compare: 'always' } : {};
        this.dummies.set(key, this.device.createSampler(desc));
      }
      return this.dummies.get(key);
    }
    const tex = layoutEntry.texture || layoutEntry.storageTexture;
    const viewDimension = tex.viewDimension || '2d';
    let format = tex.format;
    if (!format) {
      format =
        { float: 'rgba8unorm', 'unfilterable-float': 'r32float', depth: 'depth32float', sint: 'rgba8sint', uint: 'rgba8uint' }[
          tex.sampleType
        ] || 'rgba8unorm';
    }
    if (layoutEntry.storageTexture) {
      const resolvedStorage = this._resolveStorageFormat(format);
      format =
        resolvedStorage.supportsStorageBinding && resolvedStorage.storageFormat
          ? resolvedStorage.storageFormat
          : 'rgba8unorm';
    }
    const multisampled = Boolean(layoutEntry.texture && layoutEntry.texture.multisampled);
    if (multisampled && !isMultisampleSupported(format, this.device)) {
      format = 'rgba8unorm';
    }
    const key = `texture:${format}:${viewDimension}:${multisampled}:${Boolean(layoutEntry.storageTexture)}`;
    if (!this.dummies.has(key)) {
      const layers = viewDimension === 'cube' || viewDimension === 'cube-array' ? 6 : 1;
      const texture = this.device.createTexture({
        size: { width: 1, height: 1, depthOrArrayLayers: layers },
        format,
        usage: layoutEntry.storageTexture ? GPU_TEXTURE_USAGE_STORAGE_BINDING : GPU_TEXTURE_USAGE_TEXTURE_BINDING,
        dimension: viewDimension === '1d' ? '1d' : viewDimension === '3d' ? '3d' : '2d',
        sampleCount: multisampled ? 4 : 1,
      });
      this.dummies.set(key, texture.createView({ dimension: viewDimension }));
    }
    return this.dummies.get(key);
  }

  _createBindGroup(view, offset, payloadBytes) {
    if (payloadBytes < 16) return false;
    const id = view.getUint32(offset, true);
    const layoutId = view.getUint32(offset + 4, true);
    const count = view.getUint32(offset + 8, true);
    if (16 + count * 48 > payloadBytes) return false;
    const record = this.bindGroupLayouts.get(layoutId);
    if (!record) {
      this._packetError = `bind group ${id}: unknown bind group layout ${layoutId}`;
      return false;
    }
    const entries = [];
    const descriptorResources = [];
    const seenBindings = new Set();
    for (let i = 0; i < count; i++) {
      const o = offset + 16 + i * 48;
      const binding = view.getUint32(o, true);
      if (seenBindings.has(binding)) {
        this._packetError = `bind group ${id}: duplicate binding ${binding}`;
        return false;
      }
      seenBindings.add(binding);
      const type = view.getUint32(o + 4, true);
      const resourceId = view.getUint32(o + 8, true);
      const layoutEntry = record.entries ? record.entries.find((e) => e.binding === binding) : null;
      const layoutExpectsBuffer = Boolean(layoutEntry && layoutEntry.buffer);
      const layoutExpectsTexture = Boolean(
        layoutEntry && (layoutEntry.texture || layoutEntry.storageTexture)
      );
      const layoutExpectsSampler = Boolean(layoutEntry && layoutEntry.sampler);
      const kindMismatch =
        resourceId !== 0 &&
        Boolean(layoutEntry) &&
        ((layoutExpectsBuffer && type !== BIND_RESOURCE_TYPE.BUFFER) ||
          (layoutExpectsTexture && type !== BIND_RESOURCE_TYPE.TEXTURE_VIEW) ||
          (layoutExpectsSampler && type !== BIND_RESOURCE_TYPE.SAMPLER));

      let realResource = null;
      let entryResource = null;
      let entryUsedDummy = false;
      let bufferUsage = null;
      let bufferSizeKnown = null;
      let textureUsage = null;
      let textureFormat = null;
      let textureViewDimension = null;
      let textureMultisampled = null;
      let textureViewDescUsed = null;
      let samplerComparison = null;
      if (resourceId === 0) {
        if (layoutExpectsSampler || type === BIND_RESOURCE_TYPE.SAMPLER) {
          const samplerKind =
            (layoutEntry && layoutEntry.sampler && layoutEntry.sampler.type) || 'filtering';
          const fallbackSampler = this._resolveDefaultSamplerFromCache(samplerKind);
          if (fallbackSampler) {
            entryResource = fallbackSampler;
            entryUsedDummy = true;
            samplerComparison = samplerKind === 'comparison';
            this.telemetry.fallbackSamplersBound += 1;
          } else if (layoutEntry) {
            entryResource = this._dummyResource(layoutEntry);
            if (entryResource) entryUsedDummy = true;
          }
        } else {
          entryResource = layoutEntry ? this._dummyResource(layoutEntry) : null;
          if (entryResource) entryUsedDummy = true;
        }
      } else if (type === BIND_RESOURCE_TYPE.BUFFER) {
        const buffer = this.processor.buffers.get(resourceId);
        if (buffer) {
          const bufMeta =
            this.processor &&
            this.processor._bufferMeta &&
            typeof this.processor._bufferMeta.get === 'function'
              ? this.processor._bufferMeta.get(resourceId)
              : null;
          if (typeof buffer.usage === 'number') {
            bufferUsage = buffer.usage;
          } else if (bufMeta && typeof bufMeta.usage === 'number') {
            bufferUsage = bufMeta.usage;
          } else if (buffer.desc && typeof buffer.desc.usage === 'number') {
            bufferUsage = buffer.desc.usage;
          }
          const bufOffset = Number(view.getBigUint64(o + 16, true));
          const requestedSize = Number(view.getBigUint64(o + 24, true));
          const bufferSize =
            typeof buffer.size === 'number' && Number.isFinite(buffer.size) && buffer.size > 0
              ? buffer.size
              : null;
          bufferSizeKnown = bufferSize;
          const bufLayoutType = layoutExpectsBuffer ? layoutEntry.buffer.type || 'uniform' : null;
          const requiredBufferUsage =
            bufLayoutType === 'uniform'
              ? GPU_BUFFER_USAGE_UNIFORM
              : bufLayoutType
                ? GPU_BUFFER_USAGE_STORAGE
                : 0;
          const bufferUsageMismatch =
            layoutExpectsBuffer &&
            bufferUsage !== null &&
            requiredBufferUsage !== 0 &&
            (bufferUsage & requiredBufferUsage) === 0;
          const minBindingSize =
            layoutExpectsBuffer &&
            !bufferUsageMismatch &&
            typeof layoutEntry.buffer.minBindingSize === 'number' &&
            layoutEntry.buffer.minBindingSize > 0
              ? layoutEntry.buffer.minBindingSize
              : 0;

          if (bufferSize !== null && bufOffset >= bufferSize) {
            if (layoutEntry) {
              this._recordTelemetryWarningOnce(
                `bg-buffer-oob:${layoutId}:${binding}`,
                `bind group ${id}: binding ${binding} offset ${bufOffset} exceeds buffer size ${bufferSize}; using dummy buffer`
              );
              entryResource = this._dummyResource(layoutEntry);
              entryUsedDummy = true;
            }
          } else {
            let effectiveSize = requestedSize;
            if (bufferSize !== null && effectiveSize > 0 && bufOffset + effectiveSize > bufferSize) {
              effectiveSize = bufferSize - bufOffset;
            }
            const remaining = bufferSize !== null ? bufferSize - bufOffset : null;
            let undersizedForLayout = false;
            if (minBindingSize > 0) {
              const boundSize = effectiveSize > 0 ? effectiveSize : remaining;
              if (boundSize !== null && boundSize < minBindingSize) {
                if (remaining !== null && remaining >= minBindingSize) {
                  effectiveSize = minBindingSize;
                  this._recordTelemetryWarningOnce(
                    `bg-buffer-widen:${layoutId}:${binding}:${minBindingSize}`,
                    `bind group ${id}: binding ${binding} size widened to layout minBindingSize ${minBindingSize}`
                  );
                } else if (layoutEntry) {
                  this._recordTelemetryWarningOnce(
                    `bg-buffer-undersized:${layoutId}:${binding}:${minBindingSize}`,
                    `bind group ${id}: binding ${binding} buffer smaller than minBindingSize ${minBindingSize}; using dummy buffer`
                  );
                  undersizedForLayout = true;
                }
              }
            }
            if (undersizedForLayout) {
              entryResource = this._dummyResource(layoutEntry);
              entryUsedDummy = true;
            } else {
              realResource = { buffer, offset: bufOffset };
              if (effectiveSize > 0) realResource.size = effectiveSize;
              if (kindMismatch) {
                this.telemetry.bindGroupKindMismatches += 1;
                this._recordTelemetryWarningOnce(
                  `bg-kind-mismatch:${layoutId}:${binding}:buffer`,
                  `bind group ${id}: binding ${binding} buffer resource mismatches layout entry; substituting dummy resource`
                );
                entryResource = this._dummyResource(layoutEntry);
                entryUsedDummy = true;
              } else if (bufferUsageMismatch) {
                this._recordTelemetryWarningOnce(
                  `bg-buffer-usage-mismatch:${layoutId}:${binding}:${bufLayoutType}`,
                  `bind group ${id}: binding ${binding} buffer usage 0x${bufferUsage.toString(16)} lacks required ${bufLayoutType} usage; using dummy buffer`
                );
                entryResource = this._dummyResource(layoutEntry);
                entryUsedDummy = true;
              } else {
                entryResource = realResource;
              }
            }
          }
        }
      } else if (type === BIND_RESOURCE_TYPE.TEXTURE_VIEW) {
        const viewDesc = decodeTextureViewDesc(view, o + 32);
        const texFormat = this._resolveTextureFormat(resourceId, viewDesc);
        const texMeta = this.textureMeta.get(resourceId) || null;
        const texObj = this.processor.textures.get(resourceId) || null;
        if (texMeta && typeof texMeta.usage === 'number') {
          textureUsage = texMeta.usage;
        } else if (texObj && typeof texObj.usage === 'number') {
          textureUsage = texObj.usage;
        } else if (texObj && texObj.desc && typeof texObj.desc.usage === 'number') {
          textureUsage = texObj.desc.usage;
        }
        textureFormat = texFormat || null;
        if (viewDesc && typeof viewDesc.dimension === 'string' && viewDesc.dimension !== 'undefined') {
          textureViewDimension = viewDesc.dimension;
        } else {
          const baseDim =
            (texMeta && texMeta.dimension) ||
            (texObj && typeof texObj.dimension === 'string' ? texObj.dimension : null) ||
            (texObj && texObj.desc && typeof texObj.desc.dimension === 'string' ? texObj.desc.dimension : null) ||
            '2d';
          const baseLayers =
            (texMeta && typeof texMeta.depthOrArrayLayers === 'number' ? texMeta.depthOrArrayLayers : null) ||
            (texObj && typeof texObj.depthOrArrayLayers === 'number' ? texObj.depthOrArrayLayers : null) ||
            (texObj && texObj.desc && texObj.desc.size && typeof texObj.desc.size.depthOrArrayLayers === 'number'
              ? texObj.desc.size.depthOrArrayLayers
              : 1);
          textureViewDimension = baseDim === '2d' && baseLayers > 1 ? '2d-array' : baseDim;
        }
        const sampleCount =
          (texMeta && typeof texMeta.sampleCount === 'number' ? texMeta.sampleCount : null) ||
          (texObj && typeof texObj.sampleCount === 'number' ? texObj.sampleCount : null) ||
          (texObj && texObj.desc && typeof texObj.desc.sampleCount === 'number' ? texObj.desc.sampleCount : null);
        if (sampleCount !== null) {
          textureMultisampled = sampleCount > 1;
        }
        textureViewDescUsed = viewDesc;
        if (
          !kindMismatch &&
          layoutEntry &&
          layoutEntry.storageTexture &&
          texFormat
        ) {
          const resolvedStorage = this._resolveStorageFormat(texFormat);
          if (!resolvedStorage.supportsStorageBinding || !resolvedStorage.storageFormat) {
            this._recordTelemetryWarningOnce(
              `bg-storage-format-unsupported:${texFormat}:${binding}`,
              `bind group ${id}: binding ${binding} storageTexture format ${texFormat} unsupported${
                resolvedStorage.reason ? ` (${resolvedStorage.reason})` : ''
              }; substituting dummy storage texture`
            );
            entryResource = this._dummyResource(layoutEntry);
            entryUsedDummy = true;
          } else if (resolvedStorage.needsLinearSiblingView) {
            const linearViewDesc = { ...viewDesc, format: resolvedStorage.storageFormat };
            const linearView = this._textureView(resourceId, linearViewDesc);
            if (linearView) {
              realResource = linearView;
              entryResource = linearView;
              textureFormat = resolvedStorage.storageFormat;
              textureViewDescUsed = linearViewDesc;
            } else {
              this._packetError = null;
              this._recordTelemetryWarningOnce(
                `bg-storage-linear-view-fallback:${texFormat}:${binding}`,
                `bind group ${id}: binding ${binding} linear sibling view ${resolvedStorage.storageFormat} unavailable for ${texFormat}; substituting dummy storage texture`
              );
              entryResource = this._dummyResource(layoutEntry);
              entryUsedDummy = true;
            }
          } else {
            const resolvedView = this._textureView(resourceId, viewDesc);
            if (resolvedView) {
              realResource = resolvedView;
              entryResource = resolvedView;
            }
          }
        } else if (
          !kindMismatch &&
          layoutEntry &&
          layoutEntry.texture &&
          layoutEntry.texture.sampleType === 'float' &&
          texFormat &&
          isFloat32TextureFormat(texFormat) &&
          !deviceHasFeature(this.device, 'float32-filterable')
        ) {
          this.telemetry.unfilterableFloatFallbacks += 1;
          this._recordTelemetryWarningOnce(
            `bg-unfilterable-float:${texFormat}:${binding}`,
            `bind group ${id}: binding ${binding} format ${texFormat} requires unfilterable-float without float32-filterable; substituting filterable fallback`
          );
          entryResource = this._dummyResource(layoutEntry);
          entryUsedDummy = true;
        } else {
          const resolvedView = this._textureView(resourceId, viewDesc);
          if (resolvedView) {
            realResource = resolvedView;
            if (kindMismatch) {
              this.telemetry.bindGroupKindMismatches += 1;
              this._recordTelemetryWarningOnce(
                `bg-kind-mismatch:${layoutId}:${binding}:texture`,
                `bind group ${id}: binding ${binding} texture resource mismatches layout entry; substituting dummy resource`
              );
              entryResource = this._dummyResource(layoutEntry);
              entryUsedDummy = true;
            } else {
              entryResource = resolvedView;
            }
          }
        }
      } else if (type === BIND_RESOURCE_TYPE.SAMPLER) {
        const sampler = this.processor.samplers.get(resourceId) || null;
        if (sampler) {
          realResource = sampler;
          if (sampler.desc && sampler.desc.compare !== undefined) {
            samplerComparison = true;
          } else if (sampler.desc) {
            samplerComparison = false;
          }
          if (kindMismatch) {
            this.telemetry.bindGroupKindMismatches += 1;
            this._recordTelemetryWarningOnce(
              `bg-kind-mismatch:${layoutId}:${binding}:sampler`,
              `bind group ${id}: binding ${binding} sampler resource mismatches layout entry; substituting dummy resource`
            );
            entryResource = this._dummyResource(layoutEntry);
            entryUsedDummy = true;
          } else {
            entryResource = sampler;
          }
        } else if (!kindMismatch) {
          const samplerKind =
            (layoutEntry && layoutEntry.sampler && layoutEntry.sampler.type) || 'filtering';
          const fallbackSampler = this._resolveDefaultSamplerFromCache(samplerKind);
          if (fallbackSampler) {
            entryResource = fallbackSampler;
            entryUsedDummy = true;
            samplerComparison = samplerKind === 'comparison';
            this.telemetry.fallbackSamplersBound += 1;
            this._recordTelemetryWarningOnce(
              `bg-missing-sampler:${layoutId}:${binding}:${resourceId}`,
              `bind group ${id}: binding ${binding} sampler ${resourceId} unavailable; bound default ${samplerKind} sampler`
            );
          }
        }
      }
      if (!entryResource) {
        this._packetError = `bind group ${id}: binding ${binding} resource ${resourceId} unavailable`;
        return false;
      }
      entries.push({ binding, resource: entryResource });

      const descResource = realResource || entryResource;
      const descIsDummy = !realResource;
      let resourceKind = null;
      if (descIsDummy) {
        if (layoutEntry) {
          if (layoutEntry.buffer) {
            resourceKind = layoutEntry.buffer.type || 'uniform';
            bufferUsage = GPU_BUFFER_USAGE_UNIFORM_STORAGE_COPY_DST;
          } else if (layoutEntry.storageTexture) {
            resourceKind = 'storage-texture';
            textureUsage = GPU_TEXTURE_USAGE_STORAGE_BINDING;
          } else if (layoutEntry.texture) {
            resourceKind = 'texture';
            textureUsage = GPU_TEXTURE_USAGE_TEXTURE_BINDING;
          } else if (layoutEntry.sampler) {
            resourceKind = 'sampler';
            if (samplerComparison === null) {
              samplerComparison = layoutEntry.sampler.type === 'comparison';
            }
          }
        } else if (type === BIND_RESOURCE_TYPE.SAMPLER) {
          resourceKind = 'sampler';
          if (samplerComparison === null) {
            samplerComparison = false;
          }
        }
      } else if (type === BIND_RESOURCE_TYPE.BUFFER) {
        if (!kindMismatch && !entryUsedDummy && layoutEntry && layoutEntry.buffer) {
          resourceKind = layoutEntry.buffer.type || 'uniform';
        } else {
          resourceKind =
            bufferUsage !== null &&
            (bufferUsage & GPU_BUFFER_USAGE_STORAGE) !== 0 &&
            (bufferUsage & GPU_BUFFER_USAGE_UNIFORM) === 0
              ? 'storage'
              : 'uniform';
        }
      } else if (type === BIND_RESOURCE_TYPE.TEXTURE_VIEW) {
        if (!kindMismatch && !entryUsedDummy && layoutEntry && layoutEntry.storageTexture) {
          resourceKind = 'storage-texture';
        } else if (!kindMismatch && !entryUsedDummy && layoutEntry && layoutEntry.texture) {
          resourceKind = 'texture';
        } else {
          resourceKind =
            textureUsage !== null &&
            (textureUsage & GPU_TEXTURE_USAGE_STORAGE_BINDING) !== 0 &&
            (textureUsage & GPU_TEXTURE_USAGE_TEXTURE_BINDING) === 0
              ? 'storage-texture'
              : 'texture';
        }
      } else if (type === BIND_RESOURCE_TYPE.SAMPLER) {
        resourceKind = 'sampler';
        if (!kindMismatch && layoutEntry && layoutEntry.sampler && samplerComparison === null) {
          samplerComparison = layoutEntry.sampler.type === 'comparison';
        }
      }

      if (resourceKind) {
        const descEntry = { binding, kind: resourceKind, resource: descResource, isDummy: descIsDummy };
        if (bufferUsage !== null) descEntry.bufferUsage = bufferUsage;
        if (textureUsage !== null) descEntry.textureUsage = textureUsage;
        if (textureFormat !== null && !descIsDummy) descEntry.textureFormat = textureFormat;
        if (textureViewDimension !== null && !descIsDummy) descEntry.viewDimension = textureViewDimension;
        if (textureMultisampled !== null && !descIsDummy) descEntry.multisampled = textureMultisampled;
        if (type === BIND_RESOURCE_TYPE.TEXTURE_VIEW && !descIsDummy && resourceId !== 0) {
          descEntry.planeTextureId = resourceId;
          if (textureViewDescUsed) descEntry.viewDesc = textureViewDescUsed;
        }
        if (type === BIND_RESOURCE_TYPE.SAMPLER && !descIsDummy && resourceId !== 0) {
          descEntry.samplerId = resourceId;
        }
        if (samplerComparison !== null) descEntry.comparison = samplerComparison;
        if (type === BIND_RESOURCE_TYPE.BUFFER && !descIsDummy) {
          if (bufferSizeKnown !== null) descEntry.bufferSize = bufferSizeKnown;
          // W101: remember which setBindGroup dynamicOffsets slot (root-constant ring offset)
          // applies to this buffer so layout:'auto' remapping can fold it into the static offset.
          const dynIndex =
            record.dynamicOffsetIndices && layoutExpectsBuffer
              ? record.dynamicOffsetIndices.get(binding)
              : undefined;
          if (dynIndex !== undefined) {
            descEntry.hasDynamicOffset = true;
            descEntry.dynamicOffsetIndex = dynIndex;
          }
        }
        descriptorResources.push(descEntry);
      }
    }

    // W167/W186: if the layout declares sampler slots omitted from CreateBindGroup2 (e.g. static
    // samplers baked into a root signature) and pipelineCache.getOrCreateDefaultSampler is available,
    // synthesize safe default samplers so the bind group creation succeeds instead of failing.
    if (Array.isArray(record.entries) && this._getPipelineCache()) {
      for (let i = 0; i < record.entries.length; i += 1) {
        const le = record.entries[i];
        if (!le || !le.sampler || seenBindings.has(le.binding)) continue;
        const samplerKind = le.sampler.type || 'filtering';
        const fallbackSampler = this._resolveDefaultSamplerFromCache(samplerKind);
        if (!fallbackSampler) continue;
        seenBindings.add(le.binding);
        entries.push({ binding: le.binding, resource: fallbackSampler });
        descriptorResources.push({
          binding: le.binding,
          kind: 'sampler',
          resource: fallbackSampler,
          isDummy: true,
          comparison: samplerKind === 'comparison',
        });
        this.telemetry.fallbackSamplersBound += 1;
        this._recordTelemetryWarningOnce(
          `bg-omitted-sampler:${layoutId}:${le.binding}`,
          `bind group ${id}: omitted sampler binding ${le.binding}; bound default ${samplerKind} sampler`
        );
      }
    }

    // W186: hot-path PresentX multi-plane bind-group cache for 2-entry (2D float texture + sampler)
    // plane-blit bind groups, keyed by (planeTextureId, viewFormat, sampler, layout) and invalidated
    // on texture/sampler/layout/device changes.
    let presentxCacheKey = null;
    let presentxCacheMeta = null;
    if (
      entries.length === 2 &&
      Array.isArray(record.entries) &&
      record.entries.length === 2 &&
      descriptorResources.length === 2
    ) {
      const texLayout = record.entries.find(
        (le) =>
          le &&
          le.texture &&
          !le.texture.multisampled &&
          le.texture.viewDimension === '2d' &&
          le.texture.sampleType === 'float'
      );
      const smpLayout = record.entries.find((le) => le && le.sampler);
      if (texLayout && smpLayout) {
        const texDesc = descriptorResources.find(
          (r) =>
            r &&
            r.binding === texLayout.binding &&
            r.kind === 'texture' &&
            !r.isDummy &&
            Number.isInteger(r.planeTextureId) &&
            r.planeTextureId > 0
        );
        const smpDesc = descriptorResources.find(
          (r) => r && r.binding === smpLayout.binding && r.kind === 'sampler' && r.resource
        );
        if (texDesc && smpDesc) {
          const liveTexture = this.processor.textures.get(texDesc.planeTextureId) || null;
          if (liveTexture) {
            const texGen = this._getTextureGeneration(texDesc.planeTextureId, liveTexture);
            const viewFormat = texDesc.textureFormat || 'default';
            const viewDescKey = buildTextureViewKey(texDesc.viewDesc);
            presentxCacheKey = `px:${texDesc.planeTextureId}:${texGen}:${viewFormat}:${viewDescKey}:s=${gpuObjectId(
              smpDesc.resource
            )}:p=0:l=${gpuObjectId(record.layout)}:b=${texDesc.binding},${smpDesc.binding}`;
            presentxCacheMeta = {
              view: texDesc.resource,
              planeTextureId: texDesc.planeTextureId,
              texture: liveTexture,
              textureGen: texGen,
              viewFormat,
              viewDescKey,
              samplerId: Number.isInteger(smpDesc.samplerId) ? smpDesc.samplerId : 0,
              sampler: smpDesc.resource,
              layoutId,
              layout: record.layout,
              pipelineId: 0,
              pipeline: null,
              device: this.device,
              deviceGeneration: this._cachedDeviceGeneration,
            };
          }
        }
      }
    }

    let bindGroup = null;
    if (presentxCacheKey && presentxCacheMeta) {
      const cached = this.presentxBindGroupCache.get(presentxCacheKey);
      if (
        cached &&
        cached.device === this.device &&
        cached.deviceGeneration === this._cachedDeviceGeneration &&
        cached.texture === presentxCacheMeta.texture &&
        cached.textureGen === presentxCacheMeta.textureGen &&
        cached.sampler === presentxCacheMeta.sampler &&
        cached.layout === presentxCacheMeta.layout
      ) {
        this._cachePresentXBindGroup(presentxCacheKey, cached);
        this.telemetry.presentxBindGroupCacheHits += 1;
        bindGroup = cached.bindGroup;
      }
    }

    if (!bindGroup) {
      try {
        bindGroup = this.device.createBindGroup({ layout: record.layout, entries });
      } catch (err) {
        this._packetError = `bind group ${id}: ${String(err && err.message ? err.message : err)}`;
        return false;
      }
      if (presentxCacheKey && presentxCacheMeta) {
        this.telemetry.presentxBindGroupCacheMisses += 1;
        this._cachePresentXBindGroup(presentxCacheKey, {
          ...presentxCacheMeta,
          bindGroup,
        });
      }
    }
    this.processor.bindGroups.set(id, bindGroup);
    this.bindGroupDescriptors.set(id, { id, layoutId, resources: descriptorResources });
    this._invalidateRemappedBindGroupsForBindGroup(id);
    return true;
  }

  _setBindGroup(view, offset, payloadBytes) {
    if (payloadBytes < 16) return false;
    const index = view.getUint32(offset, true);
    const bindGroupId = view.getUint32(offset + 4, true);
    const count = view.getUint32(offset + 8, true);
    if (16 + count * 4 > payloadBytes) return false;
    const offsets = [];
    for (let i = 0; i < count; i++) offsets.push(view.getUint32(offset + 16 + i * 4, true));
    // Resolved now: a later DestroyResource (bind group retired by the runtime) must not affect it.
    const bindGroup = this.processor.bindGroups.get(bindGroupId);
    if (!bindGroup) {
      this._packetError = `unknown bind group ${bindGroupId}`;
      return false;
    }
    const bindGroupDesc = this.bindGroupDescriptors.get(bindGroupId) || null;
    this.processor._passCommands.push({
      type: 'renderState',
      _rsSetBindGroup: {
        groupIndex: index,
        bindGroupId,
        bindGroup,
        bindGroupDesc,
        offsets,
      },
      _wgpuMeta: {
        kind: 'setBindGroup2',
        groupIndex: index,
        bindGroupId,
      },
      run: (ctl) => {
        if (ctl.pass) ctl.pass.setBindGroup(index, bindGroup, offsets);
      },
    });
    return true;
  }

  _createSampler(view, offset) {
    const id = view.getUint32(offset, true);
    const desc = {
      addressModeU: nameOf(ADDRESS_MODE_NAMES, view.getUint8(offset + 4)) || 'clamp-to-edge',
      addressModeV: nameOf(ADDRESS_MODE_NAMES, view.getUint8(offset + 5)) || 'clamp-to-edge',
      addressModeW: nameOf(ADDRESS_MODE_NAMES, view.getUint8(offset + 6)) || 'clamp-to-edge',
      magFilter: nameOf(FILTER_MODE_NAMES, view.getUint8(offset + 7)) || 'nearest',
      minFilter: nameOf(FILTER_MODE_NAMES, view.getUint8(offset + 8)) || 'nearest',
      mipmapFilter: nameOf(FILTER_MODE_NAMES, view.getUint8(offset + 9)) || 'nearest',
      lodMinClamp: view.getFloat32(offset + 16, true),
      lodMaxClamp: view.getFloat32(offset + 20, true),
      maxAnisotropy: Math.max(1, view.getUint16(offset + 12, true)),
    };
    const compare = view.getUint8(offset + 10);
    if (compare !== 0) desc.compare = nameOf(COMPARE_FUNCTION_NAMES, compare);
    let sampler;
    try {
      sampler = this.device.createSampler(desc);
    } catch (err) {
      this._packetError = `sampler ${id}: ${String(err && err.message ? err.message : err)}`;
      return false;
    }
    this.processor.samplers.set(id, sampler);
    this._invalidatePresentXCacheForResource(id);
    return true;
  }
}

function installRenderStatePackets(processor, options = {}) {
  const handler = new RenderStatePackets(processor, options);
  processor.extension = handler;

  if (processor && typeof processor === 'object') {
    processor[HOOK_ACTIVE_HANDLER] = handler;

    if (typeof processor._flushPresent === 'function' && !processor[HOOK_ORIG_FLUSH_PRESENT]) {
      const origFlushPresent = processor._flushPresent;
      processor[HOOK_ORIG_FLUSH_PRESENT] = origFlushPresent;
      processor._flushPresent = function hardenedFlushPresent() {
        const activeHandler = this[HOOK_ACTIVE_HANDLER];
        if (activeHandler && typeof activeHandler._preparePassCommandsForFlush === 'function') {
          activeHandler._preparePassCommandsForFlush();
        }
        return origFlushPresent.call(this);
      };
    }

    if (
      typeof processor._handleCreateComputePipeline === 'function' &&
      !processor[HOOK_ORIG_CREATE_COMPUTE_PIPELINE]
    ) {
      const origCreateComputePipeline = processor._handleCreateComputePipeline;
      processor[HOOK_ORIG_CREATE_COMPUTE_PIPELINE] = origCreateComputePipeline;
      processor._handleCreateComputePipeline = function hardenedHandleCreateComputePipeline(
        view,
        payloadOffset
      ) {
        const activeHandler = this[HOOK_ACTIVE_HANDLER];
        if (activeHandler && typeof activeHandler._handleV1CreateComputePipeline === 'function') {
          activeHandler._handleV1CreateComputePipeline(origCreateComputePipeline, view, payloadOffset);
          return;
        }
        origCreateComputePipeline.call(this, view, payloadOffset);
      };
    }
  }

  return handler;
}

const exportedApi = {
  RenderStatePackets,
  installRenderStatePackets,
  clampViewportToTarget,
  clampScissorToTarget,
  decodeRenderPipelineKey,
  decodeBeginRenderPass,
  decodeTextureViewDesc,
  decodeBindGroupLayoutEntry,
  decodeCreateTexture2,
  sanitizeTextureDescriptor,
  sanitizeColorTargetBlendState,
  sanitizeRenderPassAttachments,
  validateComputeDispatch,
  getMaxComputeWorkgroupsPerDimension,
  deviceHasFeature,
  isBcCompressedFormat,
  isSrgbTextureFormat,
  isDepthOrStencilTextureFormat,
  isFloat32TextureFormat,
  isIntegerTextureFormat,
  isBlendableTextureFormat,
  isStorageBindingSupported,
  resolveStorageTextureFormat,
  isRenderAttachmentSupported,
  isMultisampleSupported,
  resolveTextureSampleType,
  computeMaxMipLevels,
  resolveShaderEntryPoint,
  findWgslStageEntryPoints,
  parseWgslGroup0Bindings,
  isExplicitLayoutCompatibleWithWgsl,
  validateVertexBufferLayouts,
  DEFAULT_ENTRY_POINTS,
  DEPTH_TEXTURE_FORMATS,
  BLENDABLE_CORE_FORMATS,
  VERTEX_FORMAT_BYTE_SIZES,
  V2_OPCODES,
  SWAPCHAIN_TEXTURE_ID,
  TEXTURE_FORMAT_NAMES,
  TEXTURE_DIMENSION_NAMES,
  VERTEX_FORMAT_NAMES,
  BLEND_FACTOR_NAMES,
  BLEND_OPERATION_NAMES,
  COMPARE_FUNCTION_NAMES,
  STENCIL_OPERATION_NAMES,
  CULL_MODE_NAMES,
  FRONT_FACE_NAMES,
  PRIMITIVE_TOPOLOGY_NAMES,
  INDEX_FORMAT_NAMES,
  VERTEX_STEP_MODE_NAMES,
  LOAD_OP_NAMES,
  STORE_OP_NAMES,
  TEXTURE_ASPECT_NAMES,
  VIEW_DIMENSION_NAMES,
};

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6RenderState = exportedApi;
  } else if (globalScope) {
    globalScope.PlayGta6RenderState = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
