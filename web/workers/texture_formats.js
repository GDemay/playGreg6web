// DXGI / Xbox Series texture format mapper, WebGPU 256-byte mip-chain layout &
// upload alignment helpers, and CPU fallback BC1/BC3/BC4 block decoders.
// Zero external dependencies; runs in Web Workers (importScripts), browsers, and Node.js.
// Designed for runtime/gpu <-> web/workers/wgpu_worker.js (docs/gpu_commands.md).

(function (globalScope) {
  'use strict';

  const WEBGPU_BYTES_PER_ROW_ALIGNMENT = 256;
  const DEFAULT_MAX_SCRATCH_BYTES = 16 * 1024 * 1024; // 16 MiB upper cap for persistent scratch pool
  const A8_UNORM_EXPANDED_WEBGPU_FORMAT = 'rgba8unorm';
  const TEXTURE_COMPRESSION_BC_FEATURE = 'texture-compression-bc';
  const TEXTURE_COMPRESSION_BC_SLICED_3D_FEATURE = 'texture-compression-bc-sliced-3d';
  const TEXTURE_FORMATS_TIER1_FEATURE = 'texture-formats-tier1';
  const CHROMIUM_UNORM16_FEATURE = 'chromium-experimental-unorm16-texture-formats';
  const DEPTH32FLOAT_STENCIL8_FEATURE = 'depth32float-stencil8';
  const BGRA8UNORM_STORAGE_FEATURE = 'bgra8unorm-storage';

  const SRGB_SIBLING_FORMATS = Object.freeze({
    rgba8unorm: 'rgba8unorm-srgb',
    'rgba8unorm-srgb': 'rgba8unorm',
    bgra8unorm: 'bgra8unorm-srgb',
    'bgra8unorm-srgb': 'bgra8unorm',
    'bc1-rgba-unorm': 'bc1-rgba-unorm-srgb',
    'bc1-rgba-unorm-srgb': 'bc1-rgba-unorm',
    'bc2-rgba-unorm': 'bc2-rgba-unorm-srgb',
    'bc2-rgba-unorm-srgb': 'bc2-rgba-unorm',
    'bc3-rgba-unorm': 'bc3-rgba-unorm-srgb',
    'bc3-rgba-unorm-srgb': 'bc3-rgba-unorm',
    'bc7-rgba-unorm': 'bc7-rgba-unorm-srgb',
    'bc7-rgba-unorm-srgb': 'bc7-rgba-unorm',
  });

  const NORM16_TEXTURE_FORMATS = new Set([
    'r16unorm',
    'rg16unorm',
    'rgba16unorm',
    'r16snorm',
    'rg16snorm',
    'rgba16snorm',
  ]);

  const DEPTH_STENCIL_TEXTURE_FORMATS = new Set([
    'depth16unorm',
    'depth24plus',
    'depth24plus-stencil8',
    'depth32float',
    'depth32float-stencil8',
    'stencil8',
  ]);

  // Core WebGPU storage-compatible texture formats (WebGPU "Texture Format Capabilities").
  const CORE_STORAGE_TEXTURE_FORMATS = new Set([
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
  ]);

  // Storage-compatible formats enabled by optional 'texture-formats-tier1'.
  const TIER1_STORAGE_TEXTURE_FORMATS = new Set([
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
  ]);

  const BC_FORMAT_PATTERN = /^bc[1-7]/;
  const COMPRESSED_FORMAT_PATTERN = /^(bc[1-7]|etc2-|eac-|astc-)/;

  // Canonical WebGPU GPUTextureFormat block geometry (blockWidth, blockHeight, bytesPerBlock).
  const WEBGPU_FORMAT_INFO = Object.freeze({
    r8unorm: Object.freeze({ webgpuFormat: 'r8unorm', blockWidth: 1, blockHeight: 1, bytesPerBlock: 1, isCompressed: false, isDepth: false, isSrgb: false }),
    r8snorm: Object.freeze({ webgpuFormat: 'r8snorm', blockWidth: 1, blockHeight: 1, bytesPerBlock: 1, isCompressed: false, isDepth: false, isSrgb: false }),
    r8uint: Object.freeze({ webgpuFormat: 'r8uint', blockWidth: 1, blockHeight: 1, bytesPerBlock: 1, isCompressed: false, isDepth: false, isSrgb: false }),
    r8sint: Object.freeze({ webgpuFormat: 'r8sint', blockWidth: 1, blockHeight: 1, bytesPerBlock: 1, isCompressed: false, isDepth: false, isSrgb: false }),
    r16unorm: Object.freeze({ webgpuFormat: 'r16unorm', blockWidth: 1, blockHeight: 1, bytesPerBlock: 2, isCompressed: false, isDepth: false, isSrgb: false }),
    r16snorm: Object.freeze({ webgpuFormat: 'r16snorm', blockWidth: 1, blockHeight: 1, bytesPerBlock: 2, isCompressed: false, isDepth: false, isSrgb: false }),
    r16uint: Object.freeze({ webgpuFormat: 'r16uint', blockWidth: 1, blockHeight: 1, bytesPerBlock: 2, isCompressed: false, isDepth: false, isSrgb: false }),
    r16sint: Object.freeze({ webgpuFormat: 'r16sint', blockWidth: 1, blockHeight: 1, bytesPerBlock: 2, isCompressed: false, isDepth: false, isSrgb: false }),
    r16float: Object.freeze({ webgpuFormat: 'r16float', blockWidth: 1, blockHeight: 1, bytesPerBlock: 2, isCompressed: false, isDepth: false, isSrgb: false }),
    rg8unorm: Object.freeze({ webgpuFormat: 'rg8unorm', blockWidth: 1, blockHeight: 1, bytesPerBlock: 2, isCompressed: false, isDepth: false, isSrgb: false }),
    rg8snorm: Object.freeze({ webgpuFormat: 'rg8snorm', blockWidth: 1, blockHeight: 1, bytesPerBlock: 2, isCompressed: false, isDepth: false, isSrgb: false }),
    rg8uint: Object.freeze({ webgpuFormat: 'rg8uint', blockWidth: 1, blockHeight: 1, bytesPerBlock: 2, isCompressed: false, isDepth: false, isSrgb: false }),
    rg8sint: Object.freeze({ webgpuFormat: 'rg8sint', blockWidth: 1, blockHeight: 1, bytesPerBlock: 2, isCompressed: false, isDepth: false, isSrgb: false }),
    r32uint: Object.freeze({ webgpuFormat: 'r32uint', blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false, isDepth: false, isSrgb: false }),
    r32sint: Object.freeze({ webgpuFormat: 'r32sint', blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false, isDepth: false, isSrgb: false }),
    r32float: Object.freeze({ webgpuFormat: 'r32float', blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false, isDepth: false, isSrgb: false }),
    rg16unorm: Object.freeze({ webgpuFormat: 'rg16unorm', blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false, isDepth: false, isSrgb: false }),
    rg16snorm: Object.freeze({ webgpuFormat: 'rg16snorm', blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false, isDepth: false, isSrgb: false }),
    rg16uint: Object.freeze({ webgpuFormat: 'rg16uint', blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false, isDepth: false, isSrgb: false }),
    rg16sint: Object.freeze({ webgpuFormat: 'rg16sint', blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false, isDepth: false, isSrgb: false }),
    rg16float: Object.freeze({ webgpuFormat: 'rg16float', blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false, isDepth: false, isSrgb: false }),
    rgba8unorm: Object.freeze({ webgpuFormat: 'rgba8unorm', blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false, isDepth: false, isSrgb: false }),
    'rgba8unorm-srgb': Object.freeze({ webgpuFormat: 'rgba8unorm-srgb', blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false, isDepth: false, isSrgb: true }),
    rgba8snorm: Object.freeze({ webgpuFormat: 'rgba8snorm', blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false, isDepth: false, isSrgb: false }),
    rgba8uint: Object.freeze({ webgpuFormat: 'rgba8uint', blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false, isDepth: false, isSrgb: false }),
    rgba8sint: Object.freeze({ webgpuFormat: 'rgba8sint', blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false, isDepth: false, isSrgb: false }),
    bgra8unorm: Object.freeze({ webgpuFormat: 'bgra8unorm', blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false, isDepth: false, isSrgb: false }),
    'bgra8unorm-srgb': Object.freeze({ webgpuFormat: 'bgra8unorm-srgb', blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false, isDepth: false, isSrgb: true }),
    rgb9e5ufloat: Object.freeze({ webgpuFormat: 'rgb9e5ufloat', blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false, isDepth: false, isSrgb: false }),
    rgb10a2uint: Object.freeze({ webgpuFormat: 'rgb10a2uint', blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false, isDepth: false, isSrgb: false }),
    rgb10a2unorm: Object.freeze({ webgpuFormat: 'rgb10a2unorm', blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false, isDepth: false, isSrgb: false }),
    rg11b10ufloat: Object.freeze({ webgpuFormat: 'rg11b10ufloat', blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false, isDepth: false, isSrgb: false }),
    rg32uint: Object.freeze({ webgpuFormat: 'rg32uint', blockWidth: 1, blockHeight: 1, bytesPerBlock: 8, isCompressed: false, isDepth: false, isSrgb: false }),
    rg32sint: Object.freeze({ webgpuFormat: 'rg32sint', blockWidth: 1, blockHeight: 1, bytesPerBlock: 8, isCompressed: false, isDepth: false, isSrgb: false }),
    rg32float: Object.freeze({ webgpuFormat: 'rg32float', blockWidth: 1, blockHeight: 1, bytesPerBlock: 8, isCompressed: false, isDepth: false, isSrgb: false }),
    rgba16unorm: Object.freeze({ webgpuFormat: 'rgba16unorm', blockWidth: 1, blockHeight: 1, bytesPerBlock: 8, isCompressed: false, isDepth: false, isSrgb: false }),
    rgba16snorm: Object.freeze({ webgpuFormat: 'rgba16snorm', blockWidth: 1, blockHeight: 1, bytesPerBlock: 8, isCompressed: false, isDepth: false, isSrgb: false }),
    rgba16uint: Object.freeze({ webgpuFormat: 'rgba16uint', blockWidth: 1, blockHeight: 1, bytesPerBlock: 8, isCompressed: false, isDepth: false, isSrgb: false }),
    rgba16sint: Object.freeze({ webgpuFormat: 'rgba16sint', blockWidth: 1, blockHeight: 1, bytesPerBlock: 8, isCompressed: false, isDepth: false, isSrgb: false }),
    rgba16float: Object.freeze({ webgpuFormat: 'rgba16float', blockWidth: 1, blockHeight: 1, bytesPerBlock: 8, isCompressed: false, isDepth: false, isSrgb: false }),
    rgba32uint: Object.freeze({ webgpuFormat: 'rgba32uint', blockWidth: 1, blockHeight: 1, bytesPerBlock: 16, isCompressed: false, isDepth: false, isSrgb: false }),
    rgba32sint: Object.freeze({ webgpuFormat: 'rgba32sint', blockWidth: 1, blockHeight: 1, bytesPerBlock: 16, isCompressed: false, isDepth: false, isSrgb: false }),
    rgba32float: Object.freeze({ webgpuFormat: 'rgba32float', blockWidth: 1, blockHeight: 1, bytesPerBlock: 16, isCompressed: false, isDepth: false, isSrgb: false }),
    stencil8: Object.freeze({ webgpuFormat: 'stencil8', blockWidth: 1, blockHeight: 1, bytesPerBlock: 1, isCompressed: false, isDepth: true, isSrgb: false }),
    depth16unorm: Object.freeze({ webgpuFormat: 'depth16unorm', blockWidth: 1, blockHeight: 1, bytesPerBlock: 2, isCompressed: false, isDepth: true, isSrgb: false }),
    depth24plus: Object.freeze({ webgpuFormat: 'depth24plus', blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false, isDepth: true, isSrgb: false }),
    'depth24plus-stencil8': Object.freeze({ webgpuFormat: 'depth24plus-stencil8', blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false, isDepth: true, isSrgb: false }),
    depth32float: Object.freeze({ webgpuFormat: 'depth32float', blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false, isDepth: true, isSrgb: false }),
    'depth32float-stencil8': Object.freeze({ webgpuFormat: 'depth32float-stencil8', blockWidth: 1, blockHeight: 1, bytesPerBlock: 8, isCompressed: false, isDepth: true, isSrgb: false }),
    'bc1-rgba-unorm': Object.freeze({ webgpuFormat: 'bc1-rgba-unorm', blockWidth: 4, blockHeight: 4, bytesPerBlock: 8, isCompressed: true, isDepth: false, isSrgb: false }),
    'bc1-rgba-unorm-srgb': Object.freeze({ webgpuFormat: 'bc1-rgba-unorm-srgb', blockWidth: 4, blockHeight: 4, bytesPerBlock: 8, isCompressed: true, isDepth: false, isSrgb: true }),
    'bc2-rgba-unorm': Object.freeze({ webgpuFormat: 'bc2-rgba-unorm', blockWidth: 4, blockHeight: 4, bytesPerBlock: 16, isCompressed: true, isDepth: false, isSrgb: false }),
    'bc2-rgba-unorm-srgb': Object.freeze({ webgpuFormat: 'bc2-rgba-unorm-srgb', blockWidth: 4, blockHeight: 4, bytesPerBlock: 16, isCompressed: true, isDepth: false, isSrgb: true }),
    'bc3-rgba-unorm': Object.freeze({ webgpuFormat: 'bc3-rgba-unorm', blockWidth: 4, blockHeight: 4, bytesPerBlock: 16, isCompressed: true, isDepth: false, isSrgb: false }),
    'bc3-rgba-unorm-srgb': Object.freeze({ webgpuFormat: 'bc3-rgba-unorm-srgb', blockWidth: 4, blockHeight: 4, bytesPerBlock: 16, isCompressed: true, isDepth: false, isSrgb: true }),
    'bc4-r-unorm': Object.freeze({ webgpuFormat: 'bc4-r-unorm', blockWidth: 4, blockHeight: 4, bytesPerBlock: 8, isCompressed: true, isDepth: false, isSrgb: false }),
    'bc4-r-snorm': Object.freeze({ webgpuFormat: 'bc4-r-snorm', blockWidth: 4, blockHeight: 4, bytesPerBlock: 8, isCompressed: true, isDepth: false, isSrgb: false }),
    'bc5-rg-unorm': Object.freeze({ webgpuFormat: 'bc5-rg-unorm', blockWidth: 4, blockHeight: 4, bytesPerBlock: 16, isCompressed: true, isDepth: false, isSrgb: false }),
    'bc5-rg-snorm': Object.freeze({ webgpuFormat: 'bc5-rg-snorm', blockWidth: 4, blockHeight: 4, bytesPerBlock: 16, isCompressed: true, isDepth: false, isSrgb: false }),
    'bc6h-rgb-ufloat': Object.freeze({ webgpuFormat: 'bc6h-rgb-ufloat', blockWidth: 4, blockHeight: 4, bytesPerBlock: 16, isCompressed: true, isDepth: false, isSrgb: false }),
    'bc6h-rgb-float': Object.freeze({ webgpuFormat: 'bc6h-rgb-float', blockWidth: 4, blockHeight: 4, bytesPerBlock: 16, isCompressed: true, isDepth: false, isSrgb: false }),
    'bc7-rgba-unorm': Object.freeze({ webgpuFormat: 'bc7-rgba-unorm', blockWidth: 4, blockHeight: 4, bytesPerBlock: 16, isCompressed: true, isDepth: false, isSrgb: false }),
    'bc7-rgba-unorm-srgb': Object.freeze({ webgpuFormat: 'bc7-rgba-unorm-srgb', blockWidth: 4, blockHeight: 4, bytesPerBlock: 16, isCompressed: true, isDepth: false, isSrgb: true }),
  });

  /**
   * Standard DirectX / Xbox Series DXGI_FORMAT numeric enum values (dxgiformat.h).
   */
  const DXGI_FORMAT = Object.freeze({
    UNKNOWN: 0,
    R32G32B32A32_FLOAT: 2,
    R32G32B32A32_UINT: 3,
    R16G16B16A16_FLOAT: 10,
    R16G16B16A16_UNORM: 11,
    R16G16B16A16_UINT: 12,
    R16G16B16A16_SNORM: 13,
    R32G32_FLOAT: 16,
    R32G32_UINT: 17,
    D32_FLOAT_S8X24_UINT: 20,
    R10G10B10A2_UNORM: 24,
    R11G11B10_FLOAT: 26,
    R8G8B8A8_UNORM: 28,
    R8G8B8A8_UNORM_SRGB: 29,
    R8G8B8A8_UINT: 30,
    R8G8B8A8_SNORM: 31,
    R8G8B8A8_SINT: 32,
    R16G16_FLOAT: 34,
    R16G16_UNORM: 35,
    R16G16_UINT: 36,
    D32_FLOAT: 40,
    R32_FLOAT: 41,
    R32_UINT: 42,
    R32_SINT: 43,
    D24_UNORM_S8_UINT: 45,
    R8G8_UNORM: 49,
    R8G8_UINT: 50,
    R16_FLOAT: 54,
    D16_UNORM: 55,
    R16_UNORM: 56,
    R16_UINT: 57,
    R8_UNORM: 61,
    R8_UINT: 62,
    R8_SNORM: 63,
    A8_UNORM: 65,
    BC1_UNORM: 71,
    BC1_UNORM_SRGB: 72,
    BC2_UNORM: 74,
    BC2_UNORM_SRGB: 75,
    BC3_UNORM: 77,
    BC3_UNORM_SRGB: 78,
    BC4_UNORM: 80,
    BC4_SNORM: 81,
    BC5_UNORM: 83,
    BC5_SNORM: 84,
    B8G8R8A8_UNORM: 87,
    B8G8R8A8_UNORM_SRGB: 91,
    BC6H_UF16: 95,
    BC6H_SF16: 96,
    BC7_UNORM: 98,
    BC7_UNORM_SRGB: 99,
  });

  const DXGI_FORMAT_NAMES = Object.freeze(
    Object.fromEntries(
      Object.entries(DXGI_FORMAT).map(([name, code]) => [code, name])
    )
  );

  function makeEntry(
    dxgiFormat,
    webgpuFormat,
    blockWidth,
    blockHeight,
    bytesPerBlock,
    options = {}
  ) {
    return Object.freeze({
      dxgiFormat,
      name: DXGI_FORMAT_NAMES[dxgiFormat],
      webgpuFormat,
      blockWidth,
      blockHeight,
      bytesPerBlock,
      isCompressed: Boolean(options.isCompressed),
      isDepth: Boolean(options.isDepth),
      isSrgb: Boolean(options.isSrgb),
      isAlphaOnly: Boolean(options.isAlphaOnly),
      swizzle: options.swizzle || null,
      requiresFeature: options.requiresFeature || null,
    });
  }

  // Canonical DXGI_FORMAT -> WebGPU GPUTextureFormat descriptor table.
  const FORMAT_TABLE = Object.freeze({
    [DXGI_FORMAT.R32G32B32A32_FLOAT]: makeEntry(DXGI_FORMAT.R32G32B32A32_FLOAT, 'rgba32float', 1, 1, 16),
    [DXGI_FORMAT.R32G32B32A32_UINT]: makeEntry(DXGI_FORMAT.R32G32B32A32_UINT, 'rgba32uint', 1, 1, 16),
    [DXGI_FORMAT.R16G16B16A16_FLOAT]: makeEntry(DXGI_FORMAT.R16G16B16A16_FLOAT, 'rgba16float', 1, 1, 8),
    [DXGI_FORMAT.R16G16B16A16_UNORM]: makeEntry(
      DXGI_FORMAT.R16G16B16A16_UNORM,
      'rgba16unorm',
      1,
      1,
      8,
      { requiresFeature: TEXTURE_FORMATS_TIER1_FEATURE }
    ),
    [DXGI_FORMAT.R16G16B16A16_UINT]: makeEntry(DXGI_FORMAT.R16G16B16A16_UINT, 'rgba16uint', 1, 1, 8),
    [DXGI_FORMAT.R16G16B16A16_SNORM]: makeEntry(
      DXGI_FORMAT.R16G16B16A16_SNORM,
      'rgba16snorm',
      1,
      1,
      8,
      { requiresFeature: TEXTURE_FORMATS_TIER1_FEATURE }
    ),
    [DXGI_FORMAT.R32G32_FLOAT]: makeEntry(DXGI_FORMAT.R32G32_FLOAT, 'rg32float', 1, 1, 8),
    [DXGI_FORMAT.R32G32_UINT]: makeEntry(DXGI_FORMAT.R32G32_UINT, 'rg32uint', 1, 1, 8),
    [DXGI_FORMAT.D32_FLOAT_S8X24_UINT]: makeEntry(
      DXGI_FORMAT.D32_FLOAT_S8X24_UINT,
      'depth32float-stencil8',
      1,
      1,
      8,
      { isDepth: true, requiresFeature: DEPTH32FLOAT_STENCIL8_FEATURE }
    ),
    [DXGI_FORMAT.R10G10B10A2_UNORM]: makeEntry(DXGI_FORMAT.R10G10B10A2_UNORM, 'rgb10a2unorm', 1, 1, 4),
    [DXGI_FORMAT.R11G11B10_FLOAT]: makeEntry(DXGI_FORMAT.R11G11B10_FLOAT, 'rg11b10ufloat', 1, 1, 4),
    [DXGI_FORMAT.R8G8B8A8_UNORM]: makeEntry(DXGI_FORMAT.R8G8B8A8_UNORM, 'rgba8unorm', 1, 1, 4),
    [DXGI_FORMAT.R8G8B8A8_UNORM_SRGB]: makeEntry(
      DXGI_FORMAT.R8G8B8A8_UNORM_SRGB,
      'rgba8unorm-srgb',
      1,
      1,
      4,
      { isSrgb: true }
    ),
    [DXGI_FORMAT.R8G8B8A8_UINT]: makeEntry(DXGI_FORMAT.R8G8B8A8_UINT, 'rgba8uint', 1, 1, 4),
    [DXGI_FORMAT.R8G8B8A8_SNORM]: makeEntry(DXGI_FORMAT.R8G8B8A8_SNORM, 'rgba8snorm', 1, 1, 4),
    [DXGI_FORMAT.R8G8B8A8_SINT]: makeEntry(DXGI_FORMAT.R8G8B8A8_SINT, 'rgba8sint', 1, 1, 4),
    [DXGI_FORMAT.R16G16_FLOAT]: makeEntry(DXGI_FORMAT.R16G16_FLOAT, 'rg16float', 1, 1, 4),
    [DXGI_FORMAT.R16G16_UNORM]: makeEntry(
      DXGI_FORMAT.R16G16_UNORM,
      'rg16unorm',
      1,
      1,
      4,
      { requiresFeature: TEXTURE_FORMATS_TIER1_FEATURE }
    ),
    [DXGI_FORMAT.R16G16_UINT]: makeEntry(DXGI_FORMAT.R16G16_UINT, 'rg16uint', 1, 1, 4),
    [DXGI_FORMAT.D32_FLOAT]: makeEntry(DXGI_FORMAT.D32_FLOAT, 'depth32float', 1, 1, 4, { isDepth: true }),
    [DXGI_FORMAT.R32_FLOAT]: makeEntry(DXGI_FORMAT.R32_FLOAT, 'r32float', 1, 1, 4),
    [DXGI_FORMAT.R32_UINT]: makeEntry(DXGI_FORMAT.R32_UINT, 'r32uint', 1, 1, 4),
    [DXGI_FORMAT.R32_SINT]: makeEntry(DXGI_FORMAT.R32_SINT, 'r32sint', 1, 1, 4),
    [DXGI_FORMAT.D24_UNORM_S8_UINT]: makeEntry(
      DXGI_FORMAT.D24_UNORM_S8_UINT,
      'depth24plus-stencil8',
      1,
      1,
      4,
      { isDepth: true }
    ),
    [DXGI_FORMAT.R8G8_UNORM]: makeEntry(DXGI_FORMAT.R8G8_UNORM, 'rg8unorm', 1, 1, 2),
    [DXGI_FORMAT.R8G8_UINT]: makeEntry(DXGI_FORMAT.R8G8_UINT, 'rg8uint', 1, 1, 2),
    [DXGI_FORMAT.R16_FLOAT]: makeEntry(DXGI_FORMAT.R16_FLOAT, 'r16float', 1, 1, 2),
    [DXGI_FORMAT.D16_UNORM]: makeEntry(DXGI_FORMAT.D16_UNORM, 'depth16unorm', 1, 1, 2, { isDepth: true }),
    [DXGI_FORMAT.R16_UNORM]: makeEntry(
      DXGI_FORMAT.R16_UNORM,
      'r16unorm',
      1,
      1,
      2,
      { requiresFeature: TEXTURE_FORMATS_TIER1_FEATURE }
    ),
    [DXGI_FORMAT.R16_UINT]: makeEntry(DXGI_FORMAT.R16_UINT, 'r16uint', 1, 1, 2),
    [DXGI_FORMAT.R8_UNORM]: makeEntry(DXGI_FORMAT.R8_UNORM, 'r8unorm', 1, 1, 1),
    [DXGI_FORMAT.R8_UINT]: makeEntry(DXGI_FORMAT.R8_UINT, 'r8uint', 1, 1, 1),
    [DXGI_FORMAT.R8_SNORM]: makeEntry(DXGI_FORMAT.R8_SNORM, 'r8snorm', 1, 1, 1),
    [DXGI_FORMAT.A8_UNORM]: makeEntry(DXGI_FORMAT.A8_UNORM, 'r8unorm', 1, 1, 1, {
      isAlphaOnly: true,
      swizzle: '000r',
    }),
    [DXGI_FORMAT.B8G8R8A8_UNORM]: makeEntry(DXGI_FORMAT.B8G8R8A8_UNORM, 'bgra8unorm', 1, 1, 4),
    [DXGI_FORMAT.B8G8R8A8_UNORM_SRGB]: makeEntry(
      DXGI_FORMAT.B8G8R8A8_UNORM_SRGB,
      'bgra8unorm-srgb',
      1,
      1,
      4,
      { isSrgb: true }
    ),

    // BC1..BC7 4x4 block-compressed formats (require 'texture-compression-bc' in WebGPU).
    [DXGI_FORMAT.BC1_UNORM]: makeEntry(DXGI_FORMAT.BC1_UNORM, 'bc1-rgba-unorm', 4, 4, 8, {
      isCompressed: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC1_UNORM_SRGB]: makeEntry(DXGI_FORMAT.BC1_UNORM_SRGB, 'bc1-rgba-unorm-srgb', 4, 4, 8, {
      isCompressed: true,
      isSrgb: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC2_UNORM]: makeEntry(DXGI_FORMAT.BC2_UNORM, 'bc2-rgba-unorm', 4, 4, 16, {
      isCompressed: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC2_UNORM_SRGB]: makeEntry(DXGI_FORMAT.BC2_UNORM_SRGB, 'bc2-rgba-unorm-srgb', 4, 4, 16, {
      isCompressed: true,
      isSrgb: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC3_UNORM]: makeEntry(DXGI_FORMAT.BC3_UNORM, 'bc3-rgba-unorm', 4, 4, 16, {
      isCompressed: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC3_UNORM_SRGB]: makeEntry(DXGI_FORMAT.BC3_UNORM_SRGB, 'bc3-rgba-unorm-srgb', 4, 4, 16, {
      isCompressed: true,
      isSrgb: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC4_UNORM]: makeEntry(DXGI_FORMAT.BC4_UNORM, 'bc4-r-unorm', 4, 4, 8, {
      isCompressed: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC4_SNORM]: makeEntry(DXGI_FORMAT.BC4_SNORM, 'bc4-r-snorm', 4, 4, 8, {
      isCompressed: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC5_UNORM]: makeEntry(DXGI_FORMAT.BC5_UNORM, 'bc5-rg-unorm', 4, 4, 16, {
      isCompressed: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC5_SNORM]: makeEntry(DXGI_FORMAT.BC5_SNORM, 'bc5-rg-snorm', 4, 4, 16, {
      isCompressed: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC6H_UF16]: makeEntry(DXGI_FORMAT.BC6H_UF16, 'bc6h-rgb-ufloat', 4, 4, 16, {
      isCompressed: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC6H_SF16]: makeEntry(DXGI_FORMAT.BC6H_SF16, 'bc6h-rgb-float', 4, 4, 16, {
      isCompressed: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC7_UNORM]: makeEntry(DXGI_FORMAT.BC7_UNORM, 'bc7-rgba-unorm', 4, 4, 16, {
      isCompressed: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC7_UNORM_SRGB]: makeEntry(DXGI_FORMAT.BC7_UNORM_SRGB, 'bc7-rgba-unorm-srgb', 4, 4, 16, {
      isCompressed: true,
      isSrgb: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
  });

  function resolveDxgiFormatCode(dxgiFormat) {
    if (typeof dxgiFormat === 'number') {
      if (Number.isInteger(dxgiFormat) && Object.prototype.hasOwnProperty.call(FORMAT_TABLE, dxgiFormat)) {
        return dxgiFormat;
      }
      throw new RangeError(`Unsupported DXGI_FORMAT value: ${dxgiFormat}`);
    }
    if (typeof dxgiFormat === 'string') {
      const normalized = dxgiFormat.trim().toUpperCase().replace(/^DXGI_FORMAT_/, '');
      if (Object.prototype.hasOwnProperty.call(DXGI_FORMAT, normalized)) {
        const code = DXGI_FORMAT[normalized];
        if (Object.prototype.hasOwnProperty.call(FORMAT_TABLE, code)) {
          return code;
        }
      }
      throw new RangeError(`Unsupported DXGI_FORMAT name: ${JSON.stringify(dxgiFormat)}`);
    }
    throw new TypeError(`dxgiFormat must be a DXGI_FORMAT integer or name string`);
  }

  function hasSupportedFeature(supportedFeatures, featureName) {
    if (!supportedFeatures || !featureName) {
      return false;
    }
    if (typeof supportedFeatures.has === 'function') {
      return Boolean(supportedFeatures.has(featureName));
    }
    if (Array.isArray(supportedFeatures)) {
      return supportedFeatures.includes(featureName);
    }
    if (typeof supportedFeatures[Symbol.iterator] === 'function') {
      for (const item of supportedFeatures) {
        if (item === featureName) {
          return true;
        }
      }
      return false;
    }
    if (typeof supportedFeatures === 'object') {
      return Boolean(supportedFeatures[featureName]);
    }
    return false;
  }

  function isSharedBuffer(buf) {
    if (!buf || typeof buf !== 'object') {
      return false;
    }
    if (typeof SharedArrayBuffer !== 'undefined' && buf instanceof SharedArrayBuffer) {
      return true;
    }
    return Object.prototype.toString.call(buf) === '[object SharedArrayBuffer]';
  }

  function toUint8Array(input, label) {
    if (input instanceof Uint8Array) {
      return input;
    }
    if (ArrayBuffer.isView(input)) {
      return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
    }
    if (input instanceof ArrayBuffer || isSharedBuffer(input)) {
      return new Uint8Array(input);
    }
    if (
      input &&
      typeof input === 'object' &&
      typeof input.byteLength === 'number' &&
      Object.prototype.toString.call(input) === '[object ArrayBuffer]'
    ) {
      return new Uint8Array(input);
    }
    throw new TypeError(
      `${label} expects a Uint8Array, ArrayBufferView, ArrayBuffer, or SharedArrayBuffer`
    );
  }

  function resolveFormatBlockInfo(formatOrInfo) {
    if (formatOrInfo && typeof formatOrInfo === 'object') {
      return formatOrInfo;
    }
    if (typeof formatOrInfo === 'string') {
      const lowered = formatOrInfo.trim().toLowerCase();
      if (Object.prototype.hasOwnProperty.call(WEBGPU_FORMAT_INFO, lowered)) {
        return WEBGPU_FORMAT_INFO[lowered];
      }
      return FORMAT_TABLE[resolveDxgiFormatCode(formatOrInfo)];
    }
    if (typeof formatOrInfo === 'number') {
      return FORMAT_TABLE[resolveDxgiFormatCode(formatOrInfo)];
    }
    throw new TypeError('formatInfo must be a format descriptor object, WebGPU format name, or DXGI_FORMAT');
  }

  /**
   * Look up WebGPU block geometry (`blockWidth`, `blockHeight`, `bytesPerBlock`,
   * `isCompressed`, `isDepth`, `isSrgb`) by WebGPU format string, DXGI_FORMAT code/name,
   * or format descriptor object. Returns `null` for unknown/invalid values.
   *
   * @param {string | number | object} formatOrInfo
   */
  function getWebGpuFormatInfo(formatOrInfo) {
    if (typeof formatOrInfo === 'string') {
      const lowered = formatOrInfo.trim().toLowerCase();
      if (Object.prototype.hasOwnProperty.call(WEBGPU_FORMAT_INFO, lowered)) {
        return WEBGPU_FORMAT_INFO[lowered];
      }
      try {
        return FORMAT_TABLE[resolveDxgiFormatCode(formatOrInfo)];
      } catch {
        return null;
      }
    }
    if (typeof formatOrInfo === 'number') {
      return Number.isInteger(formatOrInfo) &&
        Object.prototype.hasOwnProperty.call(FORMAT_TABLE, formatOrInfo)
        ? FORMAT_TABLE[formatOrInfo]
        : null;
    }
    if (
      formatOrInfo &&
      typeof formatOrInfo === 'object' &&
      Number.isInteger(formatOrInfo.blockWidth) &&
      formatOrInfo.blockWidth >= 1 &&
      Number.isInteger(formatOrInfo.blockHeight) &&
      formatOrInfo.blockHeight >= 1 &&
      Number.isInteger(formatOrInfo.bytesPerBlock) &&
      formatOrInfo.bytesPerBlock >= 1
    ) {
      return formatOrInfo;
    }
    return null;
  }

  /**
   * Bounded reusable non-shared `Uint8Array` scratch buffer pool for row-pitch
   * repacking and `A8_UNORM -> rgba8unorm` expansion.
   *
   * - Enforces an upper cap (`maxBytes`, default 16 MiB) so oversized one-off
   *   uploads allocate a standalone buffer and never pin giant RAM.
   * - Supports `copy: true` (or `snapshot(view)`) for callers that retain slices
   *   across frames so subsequent scratch acquisitions cannot alias stored data.
   */
  class TextureScratchPool {
    constructor(options = {}) {
      const opts = options && typeof options === 'object' ? options : {};
      const rawMaxBytes =
        opts.maxBytes !== undefined ? opts.maxBytes : opts.maxScratchBytes;
      this.maxBytes =
        Number.isInteger(rawMaxBytes) && rawMaxBytes >= 0
          ? rawMaxBytes
          : DEFAULT_MAX_SCRATCH_BYTES;
      const rawMaxSlots = opts.maxSlots;
      this.maxSlots =
        Number.isInteger(rawMaxSlots) && rawMaxSlots >= 1 ? rawMaxSlots : 1;
      this._slots = new Array(this.maxSlots).fill(null);
      this._allocations = 0;
      this._reuses = 0;
      this._oversizedFallbacks = 0;
    }

    get currentBuffer() {
      return this._slots[0] || null;
    }

    get retainedBytes() {
      let total = 0;
      for (let i = 0; i < this._slots.length; i += 1) {
        if (this._slots[i]) {
          total += this._slots[i].byteLength;
        }
      }
      return total;
    }

    /**
     * Acquire a non-shared `Uint8Array` of length `requiredBytes`.
     *
     * @param {number} requiredBytes
     * @param {{ slot?: number, copy?: boolean }} [options]
     * @returns {{
     *   buffer: Uint8Array,
     *   view: Uint8Array,
     *   reused: boolean,
     *   pooled: boolean,
     *   oversized: boolean,
     *   copied: boolean,
     * }}
     */
    acquire(requiredBytes, options = {}) {
      assertNonNegativeInteger(requiredBytes, 'requiredBytes');
      const opts = options && typeof options === 'object' ? options : {};
      if (requiredBytes === 0) {
        const empty = new Uint8Array(0);
        return {
          buffer: empty,
          view: empty,
          reused: false,
          pooled: false,
          oversized: false,
          copied: Boolean(opts.copy),
        };
      }
      if (opts.copy === true) {
        const standalone = new Uint8Array(requiredBytes);
        return {
          buffer: standalone,
          view: standalone,
          reused: false,
          pooled: false,
          oversized: false,
          copied: true,
        };
      }

      const slotIndex =
        Number.isInteger(opts.slot) && opts.slot >= 0 && opts.slot < this.maxSlots
          ? opts.slot
          : 0;
      const existing = this._slots[slotIndex];
      if (existing && existing.byteLength >= requiredBytes) {
        this._reuses += 1;
        return {
          buffer: existing,
          view: existing.subarray(0, requiredBytes),
          reused: true,
          pooled: true,
          oversized: false,
          copied: false,
        };
      }

      const otherSlotsBytes = this.retainedBytes - (existing ? existing.byteLength : 0);
      if (requiredBytes > this.maxBytes || otherSlotsBytes + requiredBytes > this.maxBytes) {
        this._oversizedFallbacks += 1;
        const standalone = new Uint8Array(requiredBytes);
        return {
          buffer: standalone,
          view: standalone,
          reused: false,
          pooled: false,
          oversized: true,
          copied: false,
        };
      }

      const allocated = new Uint8Array(requiredBytes);
      this._slots[slotIndex] = allocated;
      this._allocations += 1;
      return {
        buffer: allocated,
        view: allocated,
        reused: false,
        pooled: true,
        oversized: false,
        copied: false,
      };
    }

    /**
     * Create an independent owned copy of a scratch slice so it cannot be aliased
     * by subsequent pool acquisitions.
     *
     * @param {Uint8Array | ArrayBufferView} slice
     * @returns {Uint8Array}
     */
    snapshot(slice) {
      const src = toUint8Array(slice, 'TextureScratchPool.snapshot');
      const copy = new Uint8Array(src.byteLength);
      copy.set(src, 0);
      return copy;
    }

    getStats() {
      let activeSlots = 0;
      let retainedBytes = 0;
      for (let i = 0; i < this._slots.length; i += 1) {
        if (this._slots[i]) {
          activeSlots += 1;
          retainedBytes += this._slots[i].byteLength;
        }
      }
      return {
        allocations: this._allocations,
        reuses: this._reuses,
        oversizedFallbacks: this._oversizedFallbacks,
        retainedBytes,
        maxBytes: this.maxBytes,
        maxSlots: this.maxSlots,
        activeSlots,
      };
    }

    reset() {
      this._slots.fill(null);
      this._allocations = 0;
      this._reuses = 0;
      this._oversizedFallbacks = 0;
    }

    destroy() {
      this.reset();
    }
  }

  function resolveScratchDestination(requiredBytes, scratchOrOptions) {
    if (requiredBytes === 0) {
      return {
        view: new Uint8Array(0),
        usingScratch: false,
        reusedScratch: false,
        pooledScratch: false,
        oversized: false,
      };
    }
    if (!scratchOrOptions) {
      return {
        view: new Uint8Array(requiredBytes),
        usingScratch: false,
        reusedScratch: false,
        pooledScratch: false,
        oversized: false,
      };
    }

    // Raw Uint8Array scratch buffer
    if (scratchOrOptions instanceof Uint8Array) {
      if (!isSharedBuffer(scratchOrOptions.buffer) && scratchOrOptions.byteLength >= requiredBytes) {
        return {
          view: scratchOrOptions.subarray(0, requiredBytes),
          usingScratch: true,
          reusedScratch: true,
          pooledScratch: false,
          oversized: false,
        };
      }
      return {
        view: new Uint8Array(requiredBytes),
        usingScratch: false,
        reusedScratch: false,
        pooledScratch: false,
        oversized: false,
      };
    }

    // TextureScratchPool instance
    if (typeof scratchOrOptions === 'object' && typeof scratchOrOptions.acquire === 'function') {
      const acquired = scratchOrOptions.acquire(requiredBytes);
      return {
        view: acquired.view,
        usingScratch: acquired.pooled,
        reusedScratch: acquired.reused,
        pooledScratch: acquired.pooled,
        oversized: acquired.oversized,
      };
    }

    // Options object: { scratchBuffer, scratchPool, slot, copyResult }
    if (typeof scratchOrOptions === 'object') {
      if (scratchOrOptions.copyResult === true || scratchOrOptions.copy === true) {
        return {
          view: new Uint8Array(requiredBytes),
          usingScratch: false,
          reusedScratch: false,
          pooledScratch: false,
          oversized: false,
        };
      }
      if (
        scratchOrOptions.scratchPool &&
        typeof scratchOrOptions.scratchPool.acquire === 'function'
      ) {
        const acquired = scratchOrOptions.scratchPool.acquire(requiredBytes, {
          slot: scratchOrOptions.slot,
        });
        return {
          view: acquired.view,
          usingScratch: acquired.pooled,
          reusedScratch: acquired.reused,
          pooledScratch: acquired.pooled,
          oversized: acquired.oversized,
        };
      }
      const buf = scratchOrOptions.scratchBuffer;
      if (buf instanceof Uint8Array && !isSharedBuffer(buf.buffer) && buf.byteLength >= requiredBytes) {
        return {
          view: buf.subarray(0, requiredBytes),
          usingScratch: true,
          reusedScratch: true,
          pooledScratch: false,
          oversized: false,
        };
      }
    }

    return {
      view: new Uint8Array(requiredBytes),
      usingScratch: false,
      reusedScratch: false,
      pooledScratch: false,
      oversized: false,
    };
  }

  function assertPositiveInteger(value, name) {
    if (!Number.isInteger(value) || value < 1) {
      throw new RangeError(`${name} must be a positive integer, got ${value}`);
    }
  }

  function assertNonNegativeInteger(value, name) {
    if (!Number.isInteger(value) || value < 0) {
      throw new RangeError(`${name} must be a non-negative integer, got ${value}`);
    }
  }

  /**
   * Map a DXGI / Xbox texture format to WebGPU GPUTextureFormat metadata.
   *
   * When a BCn format is requested and `supportedFeatures` does not contain
   * `'texture-compression-bc'`, `needsSoftwareDecode` is set to `true` and
   * `webgpuFormat` falls back to `'rgba8unorm'` or `'rgba8unorm-srgb'`.
   *
   * @param {number | string} dxgiFormat
   * @param {Set<string> | string[] | Iterable<string> | Record<string, boolean> | null} [supportedFeatures]
   */
  function mapDxgiFormatToWebGpu(dxgiFormat, supportedFeatures) {
    const code = resolveDxgiFormatCode(dxgiFormat);
    const entry = FORMAT_TABLE[code];

    let webgpuFormat = entry.webgpuFormat;
    let needsSoftwareDecode = false;

    if (entry.requiresFeature === TEXTURE_COMPRESSION_BC_FEATURE) {
      const hasBc = hasSupportedFeature(supportedFeatures, TEXTURE_COMPRESSION_BC_FEATURE);
      if (!hasBc) {
        needsSoftwareDecode = true;
        webgpuFormat = entry.isSrgb ? 'rgba8unorm-srgb' : 'rgba8unorm';
      }
    } else if (
      entry.requiresFeature === DEPTH32FLOAT_STENCIL8_FEATURE &&
      !hasSupportedFeature(supportedFeatures, DEPTH32FLOAT_STENCIL8_FEATURE)
    ) {
      webgpuFormat = 'depth24plus-stencil8';
    } else if (
      entry.requiresFeature === TEXTURE_FORMATS_TIER1_FEATURE &&
      NORM16_TEXTURE_FORMATS.has(entry.webgpuFormat) &&
      !hasSupportedFeature(supportedFeatures, TEXTURE_FORMATS_TIER1_FEATURE) &&
      !hasSupportedFeature(supportedFeatures, CHROMIUM_UNORM16_FEATURE)
    ) {
      webgpuFormat = entry.webgpuFormat.startsWith('rgba')
        ? 'rgba16float'
        : entry.webgpuFormat.startsWith('rg')
          ? 'rg16float'
          : 'r16float';
    }

    return {
      dxgiFormat: entry.dxgiFormat,
      name: entry.name,
      webgpuFormat,
      nativeWebgpuFormat: entry.webgpuFormat,
      blockWidth: entry.blockWidth,
      blockHeight: entry.blockHeight,
      bytesPerBlock: entry.bytesPerBlock,
      isCompressed: entry.isCompressed,
      isDepth: entry.isDepth,
      isSrgb: entry.isSrgb,
      isAlphaOnly: entry.isAlphaOnly,
      swizzle: entry.swizzle,
      requiresFeature: entry.requiresFeature,
      needsSoftwareDecode,
    };
  }

  /**
   * Return the sRGB <-> linear sibling for WebGPU texture formats that have one
   * (`rgba8unorm`, `bgra8unorm`, `bc1/bc2/bc3/bc7` unorm <-> `-srgb`), or `null`
   * if the format has no sRGB sibling or the input is invalid.
   *
   * @param {string} format
   * @returns {string | null}
   */
  function getSrgbSiblingFormat(format) {
    if (typeof format !== 'string') {
      return null;
    }
    const normalized = format.trim().toLowerCase();
    if (Object.prototype.hasOwnProperty.call(SRGB_SIBLING_FORMATS, normalized)) {
      return SRGB_SIBLING_FORMATS[normalized];
    }
    return null;
  }

  /**
   * Classify whether a WebGPU texture format is supported for a given dimension
   * and optional device feature set, returning the required feature, fallback
   * format, and machine-readable reason when unsupported.
   *
   * @param {string} format
   * @param {{
   *   dimension?: '1d' | '2d' | '3d',
   *   deviceFeatures?: Set<string> | string[] | Iterable<string> | Record<string, boolean> | null,
   *   usage?: number
   * }} [options]
   * @returns {{
   *   supported: boolean,
   *   requiredFeature: string | null,
   *   fallbackFormat: string | null,
   *   reason: string | null
   * }}
   */
  function classifyTextureFormatSupport(format, options = {}) {
    if (typeof format !== 'string') {
      return {
        supported: false,
        requiredFeature: null,
        fallbackFormat: null,
        reason: 'invalid-format',
      };
    }

    const normalized = format.trim().toLowerCase();
    if (!normalized) {
      return {
        supported: false,
        requiredFeature: null,
        fallbackFormat: null,
        reason: 'invalid-format',
      };
    }

    const opts = options && typeof options === 'object' ? options : {};
    const dimension =
      typeof opts.dimension === 'string' && opts.dimension.trim()
        ? opts.dimension.trim().toLowerCase()
        : '2d';
    const deviceFeatures = opts.deviceFeatures !== undefined ? opts.deviceFeatures : null;

    if (BC_FORMAT_PATTERN.test(normalized)) {
      if (dimension === '3d') {
        if (hasSupportedFeature(deviceFeatures, TEXTURE_COMPRESSION_BC_SLICED_3D_FEATURE)) {
          return {
            supported: true,
            requiredFeature: TEXTURE_COMPRESSION_BC_SLICED_3D_FEATURE,
            fallbackFormat: null,
            reason: null,
          };
        }
        return {
          supported: false,
          requiredFeature: TEXTURE_COMPRESSION_BC_SLICED_3D_FEATURE,
          fallbackFormat: 'rgba8unorm',
          reason: 'missing-bc-sliced-3d',
        };
      }

      if (hasSupportedFeature(deviceFeatures, TEXTURE_COMPRESSION_BC_FEATURE)) {
        return {
          supported: true,
          requiredFeature: TEXTURE_COMPRESSION_BC_FEATURE,
          fallbackFormat: null,
          reason: null,
        };
      }
      return {
        supported: false,
        requiredFeature: TEXTURE_COMPRESSION_BC_FEATURE,
        fallbackFormat: 'rgba8unorm',
        reason: 'missing-bc',
      };
    }

    if (NORM16_TEXTURE_FORMATS.has(normalized)) {
      if (hasSupportedFeature(deviceFeatures, TEXTURE_FORMATS_TIER1_FEATURE)) {
        return {
          supported: true,
          requiredFeature: TEXTURE_FORMATS_TIER1_FEATURE,
          fallbackFormat: null,
          reason: null,
        };
      }
      if (hasSupportedFeature(deviceFeatures, CHROMIUM_UNORM16_FEATURE)) {
        return {
          supported: true,
          requiredFeature: CHROMIUM_UNORM16_FEATURE,
          fallbackFormat: null,
          reason: null,
        };
      }
      const fallbackFormat = normalized.startsWith('rgba')
        ? 'rgba16float'
        : normalized.startsWith('rg')
          ? 'rg16float'
          : 'r16float';
      return {
        supported: false,
        requiredFeature: TEXTURE_FORMATS_TIER1_FEATURE,
        fallbackFormat,
        reason: 'missing-unorm16',
      };
    }

    if (normalized === 'depth32float-stencil8') {
      if (hasSupportedFeature(deviceFeatures, DEPTH32FLOAT_STENCIL8_FEATURE)) {
        return {
          supported: true,
          requiredFeature: DEPTH32FLOAT_STENCIL8_FEATURE,
          fallbackFormat: null,
          reason: null,
        };
      }
      return {
        supported: false,
        requiredFeature: DEPTH32FLOAT_STENCIL8_FEATURE,
        fallbackFormat: 'depth24plus-stencil8',
        reason: 'missing-depth32float-stencil8',
      };
    }

    return {
      supported: true,
      requiredFeature: null,
      fallbackFormat: null,
      reason: null,
    };
  }

  function extractDeviceFeatures(optionsOrFeatures) {
    if (!optionsOrFeatures) {
      return null;
    }
    if (
      typeof optionsOrFeatures === 'object' &&
      !Array.isArray(optionsOrFeatures) &&
      typeof optionsOrFeatures.has !== 'function' &&
      typeof optionsOrFeatures[Symbol.iterator] !== 'function'
    ) {
      if (Object.prototype.hasOwnProperty.call(optionsOrFeatures, 'deviceFeatures')) {
        return optionsOrFeatures.deviceFeatures || null;
      }
      if (Object.prototype.hasOwnProperty.call(optionsOrFeatures, 'supportedFeatures')) {
        return optionsOrFeatures.supportedFeatures || null;
      }
    }
    return optionsOrFeatures;
  }

  /**
   * Resolve whether a WebGPU texture format can be bound as a UAV storage texture
   * (`GPUTextureUsage.STORAGE_BINDING`), and if an sRGB format requires a linear
   * sibling texture view (e.g. `rgba8unorm-srgb` -> `rgba8unorm`).
   *
   * Depth/stencil, block-compressed, and unsupported sRGB/optional formats return
   * `storageFormat: null` and `supportsStorageBinding: false` with a machine-readable
   * `reason` so callers can avoid WebGPU validation errors cleanly.
   *
   * @param {string} webgpuFormat
   * @param {{
   *   deviceFeatures?: Set<string> | string[] | Iterable<string> | Record<string, boolean> | null,
   *   supportedFeatures?: Set<string> | string[] | Iterable<string> | Record<string, boolean> | null
   * } | Set<string> | string[] | Iterable<string> | Record<string, boolean> | null} [optionsOrFeatures]
   * @returns {{
   *   storageFormat: string | null,
   *   supportsStorageBinding: boolean,
   *   needsLinearSiblingView: boolean,
   *   reason: string | null
   * }}
   */
  function resolveStorageCompatibleFormat(webgpuFormat, optionsOrFeatures) {
    if (typeof webgpuFormat !== 'string') {
      return {
        storageFormat: null,
        supportsStorageBinding: false,
        needsLinearSiblingView: false,
        reason: 'invalid-format',
      };
    }

    const normalized = webgpuFormat.trim().toLowerCase();
    if (!normalized) {
      return {
        storageFormat: null,
        supportsStorageBinding: false,
        needsLinearSiblingView: false,
        reason: 'invalid-format',
      };
    }

    const deviceFeatures = extractDeviceFeatures(optionsOrFeatures);

    if (DEPTH_STENCIL_TEXTURE_FORMATS.has(normalized)) {
      return {
        storageFormat: null,
        supportsStorageBinding: false,
        needsLinearSiblingView: false,
        reason: 'depth-stencil-not-storage-compatible',
      };
    }

    if (COMPRESSED_FORMAT_PATTERN.test(normalized)) {
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
      if (hasSupportedFeature(deviceFeatures, BGRA8UNORM_STORAGE_FEATURE)) {
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
      if (hasSupportedFeature(deviceFeatures, TEXTURE_FORMATS_TIER1_FEATURE)) {
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

    if (normalized.endsWith('-srgb')) {
      const linearSibling = getSrgbSiblingFormat(normalized);
      if (linearSibling === 'rgba8unorm') {
        return {
          storageFormat: 'rgba8unorm',
          supportsStorageBinding: true,
          needsLinearSiblingView: true,
          reason: 'srgb-requires-linear-view',
        };
      }
      if (linearSibling === 'bgra8unorm') {
        if (hasSupportedFeature(deviceFeatures, BGRA8UNORM_STORAGE_FEATURE)) {
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
        reason: 'srgb-not-storage-compatible',
      };
    }

    return {
      storageFormat: null,
      supportsStorageBinding: false,
      needsLinearSiblingView: false,
      reason: 'unsupported-storage-format',
    };
  }

  /**
   * Compute mip-level dimensions clamped down to 1x1.
   * By default `depthOrLayers` is treated as array layers (unchanged across mips);
   * pass `is3D = true` for 3D volume textures whose depth halves per mip level.
   *
   * @param {number} width
   * @param {number} height
   * @param {number} [depthOrLayers=1]
   * @param {number} [mipLevel=0]
   * @param {boolean} [is3D=false]
   * @returns {{ width: number, height: number, depthOrLayers: number }}
   */
  function calcMipSize(width, height, depthOrLayers = 1, mipLevel = 0, is3D = false) {
    assertPositiveInteger(width, 'width');
    assertPositiveInteger(height, 'height');
    assertPositiveInteger(depthOrLayers, 'depthOrLayers');
    assertNonNegativeInteger(mipLevel, 'mipLevel');

    const scale = mipLevel >= 31 ? Infinity : Math.pow(2, mipLevel);
    const mipWidth = scale === Infinity ? 1 : Math.max(1, Math.floor(width / scale));
    const mipHeight = scale === Infinity ? 1 : Math.max(1, Math.floor(height / scale));
    const mipDepth = is3D
      ? (scale === Infinity ? 1 : Math.max(1, Math.floor(depthOrLayers / scale)))
      : depthOrLayers;

    return {
      width: mipWidth,
      height: mipHeight,
      depthOrLayers: mipDepth,
    };
  }

  /**
   * Compute block grid counts, unaligned row stride, WebGPU-aligned `bytesPerRow`
   * (default 256-byte alignment), `rowsPerImage`, and `totalBytes` for a mip level.
   *
   * @param {object | number | string} formatInfo
   * @param {number} width
   * @param {number} height
   * @param {number} [depthOrLayers=1]
   * @param {number} [alignBytesPerRow=256]
   */
  function calcMipLayout(
    formatInfo,
    width,
    height,
    depthOrLayers = 1,
    alignBytesPerRow = WEBGPU_BYTES_PER_ROW_ALIGNMENT
  ) {
    const info = resolveFormatBlockInfo(formatInfo);

    assertPositiveInteger(info.blockWidth, 'formatInfo.blockWidth');
    assertPositiveInteger(info.blockHeight, 'formatInfo.blockHeight');
    assertPositiveInteger(info.bytesPerBlock, 'formatInfo.bytesPerBlock');
    assertPositiveInteger(width, 'width');
    assertPositiveInteger(height, 'height');
    assertPositiveInteger(depthOrLayers, 'depthOrLayers');
    assertPositiveInteger(alignBytesPerRow, 'alignBytesPerRow');

    const blocksX = Math.ceil(width / info.blockWidth);
    const blocksY = Math.ceil(height / info.blockHeight);
    const unalignedBytesPerRow = blocksX * info.bytesPerBlock;
    const bytesPerRow = Math.ceil(unalignedBytesPerRow / alignBytesPerRow) * alignBytesPerRow;
    const rowsPerImage = blocksY;
    const totalBytes = bytesPerRow * rowsPerImage * depthOrLayers;

    return {
      width,
      height,
      depthOrLayers,
      blocksX,
      blocksY,
      unalignedBytesPerRow,
      bytesPerRow,
      rowsPerImage,
      totalBytes,
    };
  }

  /**
   * Align a WebGPU texture copy extent `(width, height, depthOrLayers)` up to
   * the format's block size (`4x4` for BC1..BC7) so small tail mips (`1x1`,
   * `2x2`, `3x3`) and non-multiple-of-4 sub-mips satisfy WebGPU validation.
   *
   * @param {object | number | string} formatInfo
   * @param {number} width
   * @param {number} height
   * @param {number} [depthOrLayers=1]
   */
  function alignCopyExtentForFormat(formatInfo, width, height, depthOrLayers = 1) {
    const info = resolveFormatBlockInfo(formatInfo);
    assertPositiveInteger(info.blockWidth, 'formatInfo.blockWidth');
    assertPositiveInteger(info.blockHeight, 'formatInfo.blockHeight');
    assertPositiveInteger(width, 'width');
    assertPositiveInteger(height, 'height');
    assertPositiveInteger(depthOrLayers, 'depthOrLayers');

    const blocksX = Math.ceil(width / info.blockWidth);
    const blocksY = Math.ceil(height / info.blockHeight);
    const alignedWidth = blocksX * info.blockWidth;
    const alignedHeight = blocksY * info.blockHeight;

    return {
      width: alignedWidth,
      height: alignedHeight,
      depthOrLayers,
      blocksX,
      blocksY,
      blockWidth: info.blockWidth,
      blockHeight: info.blockHeight,
      wasClamped: alignedWidth !== width || alignedHeight !== height,
    };
  }

  /**
   * Compute full WebGPU upload layout for a mip level, including block-clamped
   * copy extents (`copyWidth`, `copyHeight`) for BC1..BC7 tail mips (`1x1`,
   * `2x2`, `3x3` -> `4x4`), tight vs 256-byte aligned row strides, and
   * multi-row/multi-layer padding requirements.
   *
   * @param {object | number | string} formatInfo
   * @param {number} width
   * @param {number} height
   * @param {number} [depthOrLayers=1]
   * @param {number} [alignBytesPerRow=256]
   */
  function calcMipUploadLayout(
    formatInfo,
    width,
    height,
    depthOrLayers = 1,
    alignBytesPerRow = WEBGPU_BYTES_PER_ROW_ALIGNMENT
  ) {
    const info = resolveFormatBlockInfo(formatInfo);
    const base = calcMipLayout(info, width, height, depthOrLayers, alignBytesPerRow);
    const copyWidth = base.blocksX * info.blockWidth;
    const copyHeight = base.blocksY * info.blockHeight;
    const isCompressed = Boolean(info.isCompressed || info.blockWidth > 1 || info.blockHeight > 1);
    const tightTotalBytes = base.unalignedBytesPerRow * base.rowsPerImage * depthOrLayers;

    return {
      width: base.width,
      height: base.height,
      copyWidth,
      copyHeight,
      depthOrLayers: base.depthOrLayers,
      blocksX: base.blocksX,
      blocksY: base.blocksY,
      blockWidth: info.blockWidth,
      blockHeight: info.blockHeight,
      bytesPerBlock: info.bytesPerBlock,
      isCompressed,
      isTailMip: isCompressed && (width < info.blockWidth || height < info.blockHeight),
      wasBlockClamped: copyWidth !== width || copyHeight !== height,
      unalignedBytesPerRow: base.unalignedBytesPerRow,
      bytesPerRow: base.bytesPerRow,
      rowsPerImage: base.rowsPerImage,
      tightTotalBytes,
      totalBytes: base.totalBytes,
      needsRowPadding: base.bytesPerRow !== base.unalignedBytesPerRow,
    };
  }

  /**
   * Pad (or tightly copy) texture rows so every row starts on an `alignedBytesPerRow`
   * boundary (256 bytes for WebGPU `copyBufferToTexture` / staging uploads).
   *
   * By default (when `scratchOrOptions` is omitted), returns a freshly allocated,
   * non-shared `Uint8Array` so callers retaining slices are never aliased.
   * Pass a non-shared `Uint8Array`, a `TextureScratchPool`, or
   * `{ scratchBuffer, scratchPool, copyResult }` to opt into bounded zero-churn
   * scratch reuse with strict trailing-padding zeroing.
   *
   * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} srcBytes
   * @param {number} unalignedBytesPerRow
   * @param {number} alignedBytesPerRow
   * @param {number} rowCount
   * @param {Uint8Array | TextureScratchPool | { scratchBuffer?: Uint8Array, scratchPool?: TextureScratchPool, slot?: number, copyResult?: boolean } | null} [scratchOrOptions=null]
   * @returns {Uint8Array}
   */
  function padTextureRowsForWriteTexture(
    srcBytes,
    unalignedBytesPerRow,
    alignedBytesPerRow,
    rowCount,
    scratchOrOptions = null
  ) {
    const src = toUint8Array(srcBytes, 'padTextureRowsForWriteTexture');
    assertNonNegativeInteger(unalignedBytesPerRow, 'unalignedBytesPerRow');
    assertNonNegativeInteger(alignedBytesPerRow, 'alignedBytesPerRow');
    assertNonNegativeInteger(rowCount, 'rowCount');

    if (alignedBytesPerRow < unalignedBytesPerRow) {
      throw new RangeError(
        `alignedBytesPerRow (${alignedBytesPerRow}) must be >= unalignedBytesPerRow (${unalignedBytesPerRow})`
      );
    }

    const requiredSrcBytes = unalignedBytesPerRow * rowCount;
    if (src.byteLength < requiredSrcBytes) {
      throw new RangeError(
        `srcBytes too small (${src.byteLength} bytes; need at least ${requiredSrcBytes} bytes for ${rowCount} rows)`
      );
    }

    const dstBytes = alignedBytesPerRow * rowCount;
    const dest = resolveScratchDestination(dstBytes, scratchOrOptions);
    const dst = dest.view;

    if (alignedBytesPerRow === unalignedBytesPerRow) {
      dst.set(src.subarray(0, dstBytes), 0);
      return dst;
    }

    for (let row = 0; row < rowCount; row += 1) {
      const srcOffset = row * unalignedBytesPerRow;
      const dstOffset = row * alignedBytesPerRow;
      dst.set(src.subarray(srcOffset, srcOffset + unalignedBytesPerRow), dstOffset);
      if (dest.usingScratch) {
        dst.fill(0, dstOffset + unalignedBytesPerRow, dstOffset + alignedBytesPerRow);
      }
    }
    return dst;
  }

  /**
   * Expand `DXGI_FORMAT_A8_UNORM` (1 Bpp alpha-only font/mask atlas) into
   * `#973` `rgba8unorm` texels `(0, 0, 0, a)` with optional 256-byte row-pitch
   * alignment and bounded reusable scratch buffer support.
   *
   * Returns a metadata + `data` object (default `alignBytesPerRow = 256`).
   * By default (when no `scratchBuffer`/`scratchPool` is provided, or when
   * `copyResult: true`), returns an independent non-shared `Uint8Array`.
   *
   * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} srcBytes
   * @param {number} width
   * @param {number} height
   * @param {{
   *   depthOrLayers?: number,
   *   srcBytesPerRow?: number,
   *   srcRowsPerImage?: number,
   *   alignBytesPerRow?: number,
   *   scratchBuffer?: Uint8Array | null,
   *   scratchPool?: TextureScratchPool | null,
   *   slot?: number,
   *   copyResult?: boolean,
   * }} [options]
   */
  function expandA8ToRgba8Layout(srcBytes, width, height, options = {}) {
    const src = toUint8Array(srcBytes, 'expandA8ToRgba8Layout');
    assertPositiveInteger(width, 'width');
    assertPositiveInteger(height, 'height');

    const opts = options && typeof options === 'object' ? options : {};
    const depthOrLayers = opts.depthOrLayers !== undefined ? opts.depthOrLayers : 1;
    const srcBytesPerRow = opts.srcBytesPerRow !== undefined ? opts.srcBytesPerRow : width;
    const srcRowsPerImage = opts.srcRowsPerImage !== undefined ? opts.srcRowsPerImage : height;
    const alignBytesPerRow =
      opts.alignBytesPerRow !== undefined
        ? opts.alignBytesPerRow
        : WEBGPU_BYTES_PER_ROW_ALIGNMENT;

    assertPositiveInteger(depthOrLayers, 'depthOrLayers');
    assertPositiveInteger(srcBytesPerRow, 'srcBytesPerRow');
    assertPositiveInteger(srcRowsPerImage, 'srcRowsPerImage');
    assertPositiveInteger(alignBytesPerRow, 'alignBytesPerRow');

    if (srcBytesPerRow < width) {
      throw new RangeError(`srcBytesPerRow (${srcBytesPerRow}) must be >= width (${width})`);
    }
    if (srcRowsPerImage < height) {
      throw new RangeError(`srcRowsPerImage (${srcRowsPerImage}) must be >= height (${height})`);
    }

    const srcSliceBytes = srcBytesPerRow * srcRowsPerImage;
    const minRequiredSrcBytes =
      (depthOrLayers - 1) * srcSliceBytes + (height - 1) * srcBytesPerRow + width;
    if (src.byteLength < minRequiredSrcBytes) {
      throw new RangeError(
        `A8_UNORM input too small (${src.byteLength} bytes; need at least ${minRequiredSrcBytes} bytes for ${width}x${height}x${depthOrLayers})`
      );
    }

    const unalignedBytesPerRow = width * 4;
    const bytesPerRow = Math.ceil(unalignedBytesPerRow / alignBytesPerRow) * alignBytesPerRow;
    const rowsPerImage = height;
    const totalBytes = bytesPerRow * rowsPerImage * depthOrLayers;
    const dest = resolveScratchDestination(totalBytes, opts);
    const dst = dest.view;
    const hasTrailingPad = bytesPerRow > unalignedBytesPerRow;

    for (let z = 0; z < depthOrLayers; z += 1) {
      const srcSliceBase = z * srcSliceBytes;
      const dstSliceBase = z * rowsPerImage * bytesPerRow;
      for (let y = 0; y < height; y += 1) {
        const srcRow = srcSliceBase + y * srcBytesPerRow;
        const dstRow = dstSliceBase + y * bytesPerRow;
        for (let x = 0; x < width; x += 1) {
          const outOff = dstRow + (x << 2);
          dst[outOff] = 0;
          dst[outOff + 1] = 0;
          dst[outOff + 2] = 0;
          dst[outOff + 3] = src[srcRow + x];
        }
        if (dest.usingScratch && hasTrailingPad) {
          dst.fill(0, dstRow + unalignedBytesPerRow, dstRow + bytesPerRow);
        }
      }
    }

    return {
      data: dst,
      width,
      height,
      depthOrLayers,
      unalignedBytesPerRow,
      bytesPerRow,
      rowsPerImage,
      totalBytes,
      copiedFromSab: isSharedBuffer(src.buffer),
      reusedScratch: dest.reusedScratch,
      pooledScratch: dest.pooledScratch,
      oversized: dest.oversized,
    };
  }

  /**
   * Expand `DXGI_FORMAT_A8_UNORM` alpha texels into `#973` `rgba8unorm`
   * `(0, 0, 0, a)` bytes, returning a `Uint8Array`.
   * Defaults to tightly-packed rows (`alignBytesPerRow = 1`); pass
   * `{ alignBytesPerRow: 256 }` (or `alignTo256: true`) for 256-byte row padding,
   * and `{ scratchBuffer }` or `{ scratchPool }` for opt-in scratch reuse.
   *
   * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} srcBytes
   * @param {number} width
   * @param {number} height
   * @param {object} [options]
   * @returns {Uint8Array}
   */
  function expandA8ToRgba8(srcBytes, width, height, options = {}) {
    const opts = options && typeof options === 'object' ? options : {};
    const alignBytesPerRow =
      opts.alignBytesPerRow !== undefined
        ? opts.alignBytesPerRow
        : opts.alignTo256
          ? WEBGPU_BYTES_PER_ROW_ALIGNMENT
          : 1;
    return expandA8ToRgba8Layout(srcBytes, width, height, {
      ...opts,
      alignBytesPerRow,
    }).data;
  }

  function decodeRgb565(c, outPalette, baseOffset) {
    const r5 = (c >>> 11) & 0x1f;
    const g6 = (c >>> 5) & 0x3f;
    const b5 = c & 0x1f;
    outPalette[baseOffset] = (r5 << 3) | (r5 >>> 2);
    outPalette[baseOffset + 1] = (g6 << 2) | (g6 >>> 4);
    outPalette[baseOffset + 2] = (b5 << 3) | (b5 >>> 2);
    outPalette[baseOffset + 3] = 255;
  }

  // Decode an 8-byte BC4 / BC3-alpha interpolated endpoint table into `outTable` (length 8).
  function decodeBc4Palette(endpoint0, endpoint1, outTable) {
    outTable[0] = endpoint0;
    outTable[1] = endpoint1;
    if (endpoint0 > endpoint1) {
      outTable[2] = Math.floor((6 * endpoint0 + 1 * endpoint1 + 3) / 7);
      outTable[3] = Math.floor((5 * endpoint0 + 2 * endpoint1 + 3) / 7);
      outTable[4] = Math.floor((4 * endpoint0 + 3 * endpoint1 + 3) / 7);
      outTable[5] = Math.floor((3 * endpoint0 + 4 * endpoint1 + 3) / 7);
      outTable[6] = Math.floor((2 * endpoint0 + 5 * endpoint1 + 3) / 7);
      outTable[7] = Math.floor((1 * endpoint0 + 6 * endpoint1 + 3) / 7);
    } else {
      outTable[2] = Math.floor((4 * endpoint0 + 1 * endpoint1 + 2) / 5);
      outTable[3] = Math.floor((3 * endpoint0 + 2 * endpoint1 + 2) / 5);
      outTable[4] = Math.floor((2 * endpoint0 + 3 * endpoint1 + 2) / 5);
      outTable[5] = Math.floor((1 * endpoint0 + 4 * endpoint1 + 2) / 5);
      outTable[6] = 0;
      outTable[7] = 255;
    }
  }

  /**
   * CPU fallback BC1 / DXT1 decoder (4x4 RGB565 4-color opaque and
   * 3-color + 1-bit alpha punchthrough modes) -> tightly packed RGBA8 Uint8Array.
   *
   * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} srcBytes
   * @param {number} width
   * @param {number} height
   * @returns {Uint8Array}
   */
  function decodeBc1ToRgba8(srcBytes, width, height) {
    const src = toUint8Array(srcBytes, 'decodeBc1ToRgba8');
    assertPositiveInteger(width, 'width');
    assertPositiveInteger(height, 'height');

    const blocksX = Math.ceil(width / 4);
    const blocksY = Math.ceil(height / 4);
    const requiredBytes = blocksX * blocksY * 8;
    if (src.byteLength < requiredBytes) {
      throw new RangeError(
        `BC1 input too small (${src.byteLength} bytes; need ${requiredBytes} bytes for ${width}x${height})`
      );
    }

    const out = new Uint8Array(width * height * 4);
    const palette = new Uint8Array(16);

    let blockOffset = 0;
    for (let by = 0; by < blocksY; by += 1) {
      const baseY = by * 4;
      for (let bx = 0; bx < blocksX; bx += 1) {
        const baseX = bx * 4;
        const color0 = src[blockOffset] | (src[blockOffset + 1] << 8);
        const color1 = src[blockOffset + 2] | (src[blockOffset + 3] << 8);
        const indices =
          (src[blockOffset + 4] |
            (src[blockOffset + 5] << 8) |
            (src[blockOffset + 6] << 16) |
            (src[blockOffset + 7] << 24)) >>> 0;
        blockOffset += 8;

        decodeRgb565(color0, palette, 0);
        decodeRgb565(color1, palette, 4);

        const r0 = palette[0];
        const g0 = palette[1];
        const b0 = palette[2];
        const r1 = palette[4];
        const g1 = palette[5];
        const b1 = palette[6];

        if (color0 > color1) {
          // 4-color opaque interpolation mode
          palette[8] = Math.floor((2 * r0 + r1 + 1) / 3);
          palette[9] = Math.floor((2 * g0 + g1 + 1) / 3);
          palette[10] = Math.floor((2 * b0 + b1 + 1) / 3);
          palette[11] = 255;

          palette[12] = Math.floor((r0 + 2 * r1 + 1) / 3);
          palette[13] = Math.floor((g0 + 2 * g1 + 1) / 3);
          palette[14] = Math.floor((b0 + 2 * b1 + 1) / 3);
          palette[15] = 255;
        } else {
          // 3-color + 1-bit transparent black punchthrough mode
          palette[8] = (r0 + r1) >>> 1;
          palette[9] = (g0 + g1) >>> 1;
          palette[10] = (b0 + b1) >>> 1;
          palette[11] = 255;

          palette[12] = 0;
          palette[13] = 0;
          palette[14] = 0;
          palette[15] = 0;
        }

        for (let py = 0; py < 4; py += 1) {
          const y = baseY + py;
          if (y >= height) {
            break;
          }
          const rowOutBase = y * width * 4;
          const rowShiftBase = py * 8;
          for (let px = 0; px < 4; px += 1) {
            const x = baseX + px;
            if (x >= width) {
              break;
            }
            const palIdx = ((indices >>> (rowShiftBase + px * 2)) & 0x3) * 4;
            const dstOff = rowOutBase + x * 4;
            out[dstOff] = palette[palIdx];
            out[dstOff + 1] = palette[palIdx + 1];
            out[dstOff + 2] = palette[palIdx + 2];
            out[dstOff + 3] = palette[palIdx + 3];
          }
        }
      }
    }

    return out;
  }

  /**
   * CPU fallback BC3 / DXT5 decoder (8-level / 6-level interpolated alpha block
   * + 4-color RGB565 block) -> tightly packed RGBA8 Uint8Array.
   *
   * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} srcBytes
   * @param {number} width
   * @param {number} height
   * @returns {Uint8Array}
   */
  function decodeBc3ToRgba8(srcBytes, width, height) {
    const src = toUint8Array(srcBytes, 'decodeBc3ToRgba8');
    assertPositiveInteger(width, 'width');
    assertPositiveInteger(height, 'height');

    const blocksX = Math.ceil(width / 4);
    const blocksY = Math.ceil(height / 4);
    const requiredBytes = blocksX * blocksY * 16;
    if (src.byteLength < requiredBytes) {
      throw new RangeError(
        `BC3 input too small (${src.byteLength} bytes; need ${requiredBytes} bytes for ${width}x${height})`
      );
    }

    const out = new Uint8Array(width * height * 4);
    const alphaTable = new Uint8Array(8);
    const colorPalette = new Uint8Array(16);

    let blockOffset = 0;
    for (let by = 0; by < blocksY; by += 1) {
      const baseY = by * 4;
      for (let bx = 0; bx < blocksX; bx += 1) {
        const baseX = bx * 4;

        // Bytes 0..7: interpolated alpha block (identical layout to BC4)
        const alpha0 = src[blockOffset];
        const alpha1 = src[blockOffset + 1];
        const alphaBitsLo =
          src[blockOffset + 2] | (src[blockOffset + 3] << 8) | (src[blockOffset + 4] << 16);
        const alphaBitsHi =
          src[blockOffset + 5] | (src[blockOffset + 6] << 8) | (src[blockOffset + 7] << 16);
        decodeBc4Palette(alpha0, alpha1, alphaTable);

        // Bytes 8..15: RGB565 color block (always 4-color mode in BC3)
        const color0 = src[blockOffset + 8] | (src[blockOffset + 9] << 8);
        const color1 = src[blockOffset + 10] | (src[blockOffset + 11] << 8);
        const colorIndices =
          (src[blockOffset + 12] |
            (src[blockOffset + 13] << 8) |
            (src[blockOffset + 14] << 16) |
            (src[blockOffset + 15] << 24)) >>> 0;
        blockOffset += 16;

        decodeRgb565(color0, colorPalette, 0);
        decodeRgb565(color1, colorPalette, 4);

        const r0 = colorPalette[0];
        const g0 = colorPalette[1];
        const b0 = colorPalette[2];
        const r1 = colorPalette[4];
        const g1 = colorPalette[5];
        const b1 = colorPalette[6];

        colorPalette[8] = Math.floor((2 * r0 + r1 + 1) / 3);
        colorPalette[9] = Math.floor((2 * g0 + g1 + 1) / 3);
        colorPalette[10] = Math.floor((2 * b0 + b1 + 1) / 3);

        colorPalette[12] = Math.floor((r0 + 2 * r1 + 1) / 3);
        colorPalette[13] = Math.floor((g0 + 2 * g1 + 1) / 3);
        colorPalette[14] = Math.floor((b0 + 2 * b1 + 1) / 3);

        for (let py = 0; py < 4; py += 1) {
          const y = baseY + py;
          if (y >= height) {
            break;
          }
          const rowOutBase = y * width * 4;
          const pixelRowBase = py * 4;
          for (let px = 0; px < 4; px += 1) {
            const x = baseX + px;
            if (x >= width) {
              break;
            }
            const pixelIndex = pixelRowBase + px;
            const colorPalOff = ((colorIndices >>> (pixelIndex * 2)) & 0x3) * 4;
            const alphaIdx =
              pixelIndex < 8
                ? (alphaBitsLo >>> (pixelIndex * 3)) & 0x7
                : (alphaBitsHi >>> ((pixelIndex - 8) * 3)) & 0x7;

            const dstOff = rowOutBase + x * 4;
            out[dstOff] = colorPalette[colorPalOff];
            out[dstOff + 1] = colorPalette[colorPalOff + 1];
            out[dstOff + 2] = colorPalette[colorPalOff + 2];
            out[dstOff + 3] = alphaTable[alphaIdx];
          }
        }
      }
    }

    return out;
  }

  /**
   * CPU fallback BC4 UNORM single-channel red block decoder (used for roughness,
   * ambient occlusion, and height maps) -> tightly packed R8 Uint8Array.
   *
   * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} srcBytes
   * @param {number} width
   * @param {number} height
   * @returns {Uint8Array}
   */
  function decodeBc4ToR8(srcBytes, width, height) {
    const src = toUint8Array(srcBytes, 'decodeBc4ToR8');
    assertPositiveInteger(width, 'width');
    assertPositiveInteger(height, 'height');

    const blocksX = Math.ceil(width / 4);
    const blocksY = Math.ceil(height / 4);
    const requiredBytes = blocksX * blocksY * 8;
    if (src.byteLength < requiredBytes) {
      throw new RangeError(
        `BC4 input too small (${src.byteLength} bytes; need ${requiredBytes} bytes for ${width}x${height})`
      );
    }

    const out = new Uint8Array(width * height);
    const redTable = new Uint8Array(8);

    let blockOffset = 0;
    for (let by = 0; by < blocksY; by += 1) {
      const baseY = by * 4;
      for (let bx = 0; bx < blocksX; bx += 1) {
        const baseX = bx * 4;

        const red0 = src[blockOffset];
        const red1 = src[blockOffset + 1];
        const bitsLo =
          src[blockOffset + 2] | (src[blockOffset + 3] << 8) | (src[blockOffset + 4] << 16);
        const bitsHi =
          src[blockOffset + 5] | (src[blockOffset + 6] << 8) | (src[blockOffset + 7] << 16);
        blockOffset += 8;

        decodeBc4Palette(red0, red1, redTable);

        for (let py = 0; py < 4; py += 1) {
          const y = baseY + py;
          if (y >= height) {
            break;
          }
          const rowOutBase = y * width;
          const pixelRowBase = py * 4;
          for (let px = 0; px < 4; px += 1) {
            const x = baseX + px;
            if (x >= width) {
              break;
            }
            const pixelIndex = pixelRowBase + px;
            const redIdx =
              pixelIndex < 8
                ? (bitsLo >>> (pixelIndex * 3)) & 0x7
                : (bitsHi >>> ((pixelIndex - 8) * 3)) & 0x7;
            out[rowOutBase + x] = redTable[redIdx];
          }
        }
      }
    }

    return out;
  }

  const exportsObj = Object.freeze({
    DXGI_FORMAT,
    DXGI_FORMAT_NAMES,
    WEBGPU_FORMAT_INFO,
    WEBGPU_BYTES_PER_ROW_ALIGNMENT,
    DEFAULT_MAX_SCRATCH_BYTES,
    A8_UNORM_EXPANDED_WEBGPU_FORMAT,
    TEXTURE_COMPRESSION_BC_FEATURE,
    TEXTURE_COMPRESSION_BC_SLICED_3D_FEATURE,
    TEXTURE_FORMATS_TIER1_FEATURE,
    CHROMIUM_UNORM16_FEATURE,
    DEPTH32FLOAT_STENCIL8_FEATURE,
    BGRA8UNORM_STORAGE_FEATURE,
    TextureScratchPool,
    getWebGpuFormatInfo,
    mapDxgiFormatToWebGpu,
    getSrgbSiblingFormat,
    classifyTextureFormatSupport,
    resolveStorageCompatibleFormat,
    calcMipSize,
    calcMipLayout,
    alignCopyExtentForFormat,
    calcMipUploadLayout,
    padTextureRowsForWriteTexture,
    expandA8ToRgba8,
    expandA8ToRgba8Layout,
    decodeBc1ToRgba8,
    decodeBc3ToRgba8,
    decodeBc4ToR8,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6TextureFormats = exportsObj;
  } else if (globalScope) {
    globalScope.PlayGta6TextureFormats = exportsObj;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportsObj;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
