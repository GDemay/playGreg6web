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
const GPU_BUFFER_USAGE_UNIFORM_STORAGE_COPY_DST = 0x0040 | 0x0080 | 0x0008;
const GPU_TEXTURE_USAGE_COPY_SRC = 0x0001;
const GPU_TEXTURE_USAGE_COPY_DST = 0x0002;
const GPU_TEXTURE_USAGE_TEXTURE_BINDING = 0x0004;
const GPU_TEXTURE_USAGE_STORAGE_BINDING = 0x0008;
const GPU_TEXTURE_USAGE_RENDER_ATTACHMENT = 0x0010;
const EMPTY_FRAGMENT_WGSL = '@fragment fn fs_main() {}\n';
const GPU_SHADER_STAGE_VERTEX = 0x1;
const GPU_SHADER_STAGE_ALL_MASK = 0x7;
const MAX_TELEMETRY_WARNINGS = 32;
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

function isStorageBindingSupported(format, device) {
  if (typeof format !== 'string' || format.length === 0 || format === 'undefined') return false;
  if (
    isSrgbTextureFormat(format) ||
    isDepthOrStencilTextureFormat(format) ||
    isBcCompressedFormat(format) ||
    format === 'rgb9e5ufloat'
  ) {
    return false;
  }
  if (CORE_STORAGE_TEXTURE_FORMATS.has(format)) return true;
  if (format === 'bgra8unorm') return deviceHasFeature(device, 'bgra8unorm-storage');
  if (TIER1_STORAGE_TEXTURE_FORMATS.has(format)) {
    return deviceHasFeature(device, 'texture-formats-tier1');
  }
  return false;
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
function sanitizeTextureDescriptor(rawDescriptor, device) {
  const originalFormat = rawDescriptor.format;
  const wasBc = isBcCompressedFormat(originalFormat);
  let effectiveFormat = originalFormat;
  let usage = rawDescriptor.usage >>> 0;
  let sampleCount = rawDescriptor.sampleCount || 1;
  let bcFallback = false;
  let formatFallback = false;
  const strippedUsages = [];
  const warnings = [];

  if (wasBc && !deviceHasFeature(device, 'texture-compression-bc')) {
    bcFallback = true;
    formatFallback = true;
    effectiveFormat = isSrgbTextureFormat(originalFormat) ? 'rgba8unorm-srgb' : 'rgba8unorm';
    warnings.push({
      key: `bc-unsupported:${originalFormat}`,
      message: `texture format ${originalFormat} unavailable without texture-compression-bc; falling back to ${effectiveFormat}`,
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
    (wasBc || !isStorageBindingSupported(effectiveFormat, device))
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
    return {
      x: Math.max(0, rawX) || 0,
      y: Math.max(0, rawY) || 0,
      width: Math.max(0, rawW) || 0,
      height: Math.max(0, rawH) || 0,
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
    rawY + rawH <= targetHeight
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

  const x = Math.min(Math.max(0, rawX), targetWidth) || 0;
  const y = Math.min(Math.max(0, rawY), targetHeight) || 0;
  let width = Math.min(Math.max(0, rawW), targetWidth - x) || 0;
  let height = Math.min(Math.max(0, rawH), targetHeight - y) || 0;
  if (x + width > targetWidth) width = Math.max(0, targetWidth - x) || 0;
  if (y + height > targetHeight) height = Math.max(0, targetHeight - y) || 0;
  return { x, y, width, height, minDepth, maxDepth };
}

// Clamps a guest integer scissor rect to [0..targetWidth] x [0..targetHeight].
function clampScissorToTarget(scissor, targetSize) {
  const rawX = Math.max(0, Math.floor(finiteOr(scissor && scissor.x, 0))) || 0;
  const rawY = Math.max(0, Math.floor(finiteOr(scissor && scissor.y, 0))) || 0;
  const rawW = Math.max(0, Math.floor(finiteOr(scissor && scissor.width, 0))) || 0;
  const rawH = Math.max(0, Math.floor(finiteOr(scissor && scissor.height, 0))) || 0;

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
    // Optional: (kind 'bindGroupLayout' | 'pipelineLayout', contentId: bigint) => GPU object, for
    // ManifestReference layouts loaded from layout-manifest.json (e.g. by reflection_loader.js).
    this.resolveManifestLayout =
      typeof options.resolveManifestLayout === 'function' ? options.resolveManifestLayout : null;
    this.bindGroupLayouts = new Map(); // id -> { layout, entries, sequentialBindings }
    this.pipelineLayouts = new Map(); // id -> GPUPipelineLayout
    this.pipelineLayoutMeta = new Map(); // id -> { groupCount, isSingleGroup0, sequentialGroup0 }
    this.computePipelineMeta = new Map(); // id -> { id, csId, layoutId, entryPoint, hasComputeAttr }
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
      computePipelineCount: 0,
      computeDispatchCount: 0,
      skippedComputeDispatchCount: 0,
      clampedComputeDispatchCount: 0,
      texture2Count: 0,
      bcFormatFallbacks: 0,
      usageStrips: 0,
      unfilterableFloatFallbacks: 0,
      warnings: [],
    };
    this._seenWarningKeys = new Set();
    this.lastError = null;
    this._packetError = null;
    this._activePass = null;
    this._activePassTargetSize = null;
  }

  getStats() {
    return {
      packets: this.stats.packets,
      errors: this.stats.errors,
      legacyEntryPointRemaps: this.telemetry.legacyEntryPointRemaps,
      depthCompareDefaults: this.telemetry.depthCompareDefaults,
      rootSigGroup0Layouts: this.telemetry.rootSigGroup0Layouts,
      autoLayoutFallbacks: this.telemetry.autoLayoutFallbacks,
      computePipelineCount: this.telemetry.computePipelineCount,
      computeDispatchCount: this.telemetry.computeDispatchCount,
      skippedComputeDispatchCount: this.telemetry.skippedComputeDispatchCount,
      clampedComputeDispatchCount: this.telemetry.clampedComputeDispatchCount,
      texture2Count: this.telemetry.texture2Count,
      bcFormatFallbacks: this.telemetry.bcFormatFallbacks,
      usageStrips: this.telemetry.usageStrips,
      unfilterableFloatFallbacks: this.telemetry.unfilterableFloatFallbacks,
    };
  }

  getTelemetry() {
    return {
      legacyEntryPointRemaps: this.telemetry.legacyEntryPointRemaps,
      depthCompareDefaults: this.telemetry.depthCompareDefaults,
      rootSigGroup0Layouts: this.telemetry.rootSigGroup0Layouts,
      autoLayoutFallbacks: this.telemetry.autoLayoutFallbacks,
      computePipelineCount: this.telemetry.computePipelineCount,
      computeDispatchCount: this.telemetry.computeDispatchCount,
      skippedComputeDispatchCount: this.telemetry.skippedComputeDispatchCount,
      clampedComputeDispatchCount: this.telemetry.clampedComputeDispatchCount,
      texture2Count: this.telemetry.texture2Count,
      bcFormatFallbacks: this.telemetry.bcFormatFallbacks,
      usageStrips: this.telemetry.usageStrips,
      unfilterableFloatFallbacks: this.telemetry.unfilterableFloatFallbacks,
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
    this._packetError = null;
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
      case V2_OPCODES.END_RENDER_PASS:
        this._queue((ctl) => {
          if (ctl.passKind === 'render') ctl.endPass();
          this._activePass = null;
          this._activePassTargetSize = null;
        });
        return this._count(true);
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
          const vp = clampViewportToTarget(rawViewport, this._getActivePassTargetSize(ctl));
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
          const sc = clampScissorToTarget(rawScissor, this._getActivePassTargetSize(ctl));
          ctl.pass.setScissorRect(sc.x, sc.y, sc.width, sc.height);
        });
        return this._count(true);
      }
      case V2_OPCODES.SET_BLEND_CONSTANT: {
        if (payloadBytes < 16) return this._count(false);
        const c = [0, 4, 8, 12].map((o) => view.getFloat32(offset + o, true));
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
    if (!ok) {
      this.stats.errors += 1;
      this.lastError = this._packetError || 'malformed or truncated v2 packet';
    }
    return true;
  }

  // DestroyResource hook: forget layouts, metadata and cached views of `id`.
  destroyResource(id) {
    this.bindGroupLayouts.delete(id);
    this.pipelineLayouts.delete(id);
    this.pipelineLayoutMeta.delete(id);
    this.computePipelineMeta.delete(id);
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
        if (!isStorageBindingSupported(entry.storageTexture.format, this.device)) {
          const origFmt = entry.storageTexture.format;
          entry.storageTexture.format = 'rgba8unorm';
          this._recordTelemetryWarningOnce(
            `bgl-storage-format:${origFmt}`,
            `bind group layout storageTexture format ${origFmt} unsupported; falling back to rgba8unorm`
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
    this.bindGroupLayouts.set(id, { layout, entries, sequentialBindings });
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

  // The exact GPURenderPipelineDescriptor for a CreateRenderPipeline2 payload at `offset`.
  buildRenderPipelineDescriptor(view, offset) {
    const vsId = view.getUint32(offset + 4, true);
    const fsId = view.getUint32(offset + 8, true);
    const layoutId = view.getUint32(offset + 12, true);
    const layout = this._layoutFor(layoutId);
    const vs = this.processor.shaderModules.get(vsId);
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
      fs = this.processor.shaderModules.get(fsId);
      if (!fs) {
        this._packetError = `unknown fragment shader module ${fsId}`;
        return null;
      }
      fragmentModule = fs.module;
    } else if (key.colorTargetCount > 0) {
      // No pixel shader but color targets: an empty fragment stage that writes nothing.
      if (!this.emptyFragment) this.emptyFragment = this.device.createShaderModule({ code: EMPTY_FRAGMENT_WGSL });
      fragmentModule = this.emptyFragment;
      targets = targets.map((t) => (t ? { format: t.format, writeMask: 0 } : null));
    }

    if (layoutId === 0) {
      this.telemetry.autoLayoutFallbacks += 1;
    } else {
      const layoutMeta = this.pipelineLayoutMeta.get(layoutId);
      if (layoutMeta && layoutMeta.isSingleGroup0) {
        this.telemetry.rootSigGroup0Layouts += 1;
      }
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
    const id = view.getUint32(offset, true);
    const vsId = view.getUint32(offset + 4, true);
    const fsId = view.getUint32(offset + 8, true);
    const layoutId = view.getUint32(offset + 12, true);
    let pipeline;
    try {
      pipeline = this.device.createRenderPipeline(descriptor);
    } catch (err) {
      this._packetError = `render pipeline ${id}: ${String(
        err && err.message ? err.message : err
      )}`;
      return false;
    }
    this.processor.pipelines.set(id, pipeline);
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
    return true;
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
    const layout = this._layoutFor(layoutId);
    const cs = this.processor.shaderModules.get(csId);
    if (!cs || !layout) {
      if (!cs) this._packetError = `unknown compute shader module ${csId}`;
      return false;
    }
    const { entryPoint: csEntryPoint, hasComputeAttr } = this._inspectComputeShaderEntryPoint(cs, csId);
    let pipeline;
    try {
      pipeline = this.device.createComputePipeline({
        layout,
        compute: { module: cs.module, entryPoint: csEntryPoint },
      });
    } catch (err) {
      this._packetError = `compute pipeline ${id}: ${String(err && err.message ? err.message : err)}`;
      return false;
    }
    if (layoutId === 0) {
      this.telemetry.autoLayoutFallbacks += 1;
    } else {
      const layoutMeta = this.pipelineLayoutMeta.get(layoutId);
      if (layoutMeta && layoutMeta.isSingleGroup0) {
        this.telemetry.rootSigGroup0Layouts += 1;
      }
    }
    this.processor.computePipelines.set(id, pipeline);
    this.computePipelineMeta.set(id, {
      id,
      csId,
      layoutId,
      entryPoint: csEntryPoint,
      hasComputeAttr,
    });
    this.telemetry.computePipelineCount += 1;
    return true;
  }

  _handleV1CreateComputePipeline(origHandler, view, payloadOffset) {
    const id = view.getUint32(payloadOffset, true);
    const csId = view.getUint32(payloadOffset + 4, true);
    const cs = this.processor && this.processor.shaderModules ? this.processor.shaderModules.get(csId) : null;
    let inspected = null;
    if (cs) {
      inspected = this._inspectComputeShaderEntryPoint(cs, csId);
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
      this.computePipelineMeta.set(id, {
        id,
        csId,
        layoutId: 0,
        entryPoint: inspected ? inspected.entryPoint : DEFAULT_ENTRY_POINTS.compute,
        hasComputeAttr: Boolean(inspected && inspected.hasComputeAttr),
      });
      this.telemetry.computePipelineCount += 1;
    }
  }

  // Guards queued compute dispatches against zero/oversized workgroup dimensions and missing
  // compute pipelines immediately before WgpuCommandProcessor._flushPresent() executes them.
  _preparePassCommandsForFlush() {
    const cmds = this.processor && this.processor._passCommands;
    if (!Array.isArray(cmds) || cmds.length === 0) return;

    let activeComputePipelineId = null;
    let activeComputePipelineValid = false;
    let hadExplicitSetComputePipeline = false;
    let modified = false;
    const filtered = [];

    for (let i = 0; i < cmds.length; i += 1) {
      const cmd = cmds[i];
      if (!cmd) continue;

      if (
        cmd.type === 'setPipeline' ||
        cmd.type === 'draw' ||
        cmd.type === 'drawIndexed' ||
        (cmd._wgpuMeta &&
          (cmd._wgpuMeta.kind === 'beginRenderPass' || cmd._wgpuMeta.kind === 'endRenderPass'))
      ) {
        activeComputePipelineId = null;
        activeComputePipelineValid = false;
        hadExplicitSetComputePipeline = false;
        filtered.push(cmd);
        continue;
      }

      if (cmd.type === 'setComputePipeline') {
        hadExplicitSetComputePipeline = true;
        activeComputePipelineId = cmd.pipelineId;
        activeComputePipelineValid = Boolean(
          this.processor.computePipelines && this.processor.computePipelines.has(cmd.pipelineId)
        );
        if (!activeComputePipelineValid) {
          this._recordTelemetryWarningOnce(
            `missing-compute-pipeline:${cmd.pipelineId}`,
            `compute pipeline ${cmd.pipelineId} unavailable; subsequent dispatches will be skipped`
          );
          modified = true;
          continue;
        }
        filtered.push(cmd);
        continue;
      }

      if (cmd.type === 'dispatchCompute') {
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
    const sanitized = sanitizeTextureDescriptor(decoded.descriptor, this.device);
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
      width: sanitized.descriptor.size.width,
      height: sanitized.descriptor.size.height,
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
    const texture = this.processor.textures.get(textureId);
    if (!texture || typeof texture.createView !== 'function') return null;
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
    const key = JSON.stringify(effectiveDesc);
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
    if (textureId === SWAPCHAIN_TEXTURE_ID) return (ctl) => ctl.swapchainView;
    const textureView = this._textureView(textureId, desc);
    return textureView ? () => textureView : null;
  }

  _textureAttachmentSize(textureId, viewDesc) {
    const textures = this.processor && this.processor.textures;
    const texture = textures && typeof textures.get === 'function' ? textures.get(textureId) : null;
    if (!texture) return null;
    const base = validTargetSize(texture.width, texture.height);
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
      const fromCtl =
        validTargetSize(ctl.canvasWidth, ctl.canvasHeight) ||
        validTargetSize(ctl.swapchainWidth, ctl.swapchainHeight);
      if (fromCtl) return fromCtl;
      if (ctl.swapchainTexture) {
        const fromTex = validTargetSize(ctl.swapchainTexture.width, ctl.swapchainTexture.height);
        if (fromTex) return fromTex;
      }
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
    const colorAttachments = decoded.colors.map((c, i) => {
      if (!c) return null;
      const attachment = { view: resolvers.colors[i](ctl), loadOp: c.loadOp, storeOp: c.storeOp };
      if (c.loadOp === 'clear') attachment.clearValue = c.clearValue;
      return attachment;
    });
    const descriptor = { colorAttachments };
    const d = decoded.depthStencil;
    if (d) {
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
    return descriptor;
  }

  _beginRenderPass(view, offset) {
    const decoded = decodeBeginRenderPass(view, offset);
    const resolvers = { colors: [], depthStencil: null };
    for (const c of decoded.colors) {
      const resolver = c ? this._attachmentView(c.textureId, c.view) : null;
      if (c && !resolver) {
        this._packetError = `render pass: unknown texture ${c.textureId}`;
        return false;
      }
      resolvers.colors.push(resolver);
    }
    if (decoded.depthStencil) {
      resolvers.depthStencil = this._attachmentView(decoded.depthStencil.textureId, decoded.depthStencil.view);
      if (!resolvers.depthStencil) {
        this._packetError = `render pass: unknown texture ${decoded.depthStencil.textureId}`;
        return false;
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
    this._queue((ctl) => {
      const pass = ctl.beginRenderPass(this.buildRenderPassDescriptor(decoded, resolvers, ctl));
      this._activePass = pass || ctl.pass || null;
      this._activePassTargetSize = resolveTargetSize ? resolveTargetSize(ctl) : null;
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
      format = { float: 'rgba8unorm', 'unfilterable-float': 'r32float', depth: 'depth32float', sint: 'rgba8sint', uint: 'rgba8uint' }[
        tex.sampleType
      ];
    }
    if (layoutEntry.storageTexture && !isStorageBindingSupported(format, this.device)) {
      format = 'rgba8unorm';
    }
    const multisampled = Boolean(layoutEntry.texture && layoutEntry.texture.multisampled);
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
      let resource = null;
      if (resourceId === 0) {
        resource = layoutEntry ? this._dummyResource(layoutEntry) : null;
      } else if (type === BIND_RESOURCE_TYPE.BUFFER) {
        const buffer = this.processor.buffers.get(resourceId);
        if (buffer) {
          const bufOffset = Number(view.getBigUint64(o + 16, true));
          const requestedSize = Number(view.getBigUint64(o + 24, true));
          const bufferSize =
            typeof buffer.size === 'number' && Number.isFinite(buffer.size) && buffer.size > 0
              ? buffer.size
              : null;
          const minBindingSize =
            layoutEntry &&
            layoutEntry.buffer &&
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
              resource = this._dummyResource(layoutEntry);
            }
          } else {
            let effectiveSize = requestedSize;
            if (bufferSize !== null && effectiveSize > 0 && bufOffset + effectiveSize > bufferSize) {
              effectiveSize = bufferSize - bufOffset;
            }
            const remaining = bufferSize !== null ? bufferSize - bufOffset : null;
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
                  resource = this._dummyResource(layoutEntry);
                }
              }
            }
            if (!resource) {
              resource = { buffer, offset: bufOffset };
              if (effectiveSize > 0) resource.size = effectiveSize;
            }
          }
        }
      } else if (type === BIND_RESOURCE_TYPE.TEXTURE_VIEW) {
        const viewDesc = decodeTextureViewDesc(view, o + 32);
        const texFormat = this._resolveTextureFormat(resourceId, viewDesc);
        if (
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
          resource = this._dummyResource(layoutEntry);
        } else {
          resource = this._textureView(resourceId, viewDesc);
        }
      } else if (type === BIND_RESOURCE_TYPE.SAMPLER) {
        resource = this.processor.samplers.get(resourceId) || null;
      }
      if (!resource) {
        this._packetError = `bind group ${id}: binding ${binding} resource ${resourceId} unavailable`;
        return false;
      }
      entries.push({ binding, resource });
    }
    let bindGroup;
    try {
      bindGroup = this.device.createBindGroup({ layout: record.layout, entries });
    } catch (err) {
      this._packetError = `bind group ${id}: ${String(err && err.message ? err.message : err)}`;
      return false;
    }
    this.processor.bindGroups.set(id, bindGroup);
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
    this._queue((ctl) => {
      if (ctl.pass) ctl.pass.setBindGroup(index, bindGroup, offsets);
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
  validateComputeDispatch,
  getMaxComputeWorkgroupsPerDimension,
  deviceHasFeature,
  isBcCompressedFormat,
  isSrgbTextureFormat,
  isDepthOrStencilTextureFormat,
  isFloat32TextureFormat,
  isStorageBindingSupported,
  isRenderAttachmentSupported,
  isMultisampleSupported,
  resolveTextureSampleType,
  computeMaxMipLevels,
  resolveShaderEntryPoint,
  findWgslStageEntryPoints,
  validateVertexBufferLayouts,
  DEFAULT_ENTRY_POINTS,
  DEPTH_TEXTURE_FORMATS,
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
