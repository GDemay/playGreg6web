// DXGI / Xbox Series texture format mapper, WebGPU 256-byte mip-chain layout &
// upload alignment helpers, and CPU fallback BC1/BC3/BC4 block decoders.
// Zero external dependencies; runs in Web Workers (importScripts), browsers, and Node.js.
// Designed for runtime/gpu <-> web/workers/wgpu_worker.js (docs/gpu_commands.md).

(function (globalScope) {
  'use strict';

  const WEBGPU_BYTES_PER_ROW_ALIGNMENT = 256;
  const DXGI_FORMAT_MAX_ID = 132;
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

  function makeWebGpuInfo(
    webgpuFormat,
    blockWidth,
    blockHeight,
    bytesPerBlock,
    options = {}
  ) {
    const isDepth = Boolean(options.isDepth);
    const hasStencil = Boolean(options.hasStencil);
    const aspect =
      options.aspect ||
      (hasStencil && !isDepth
        ? 'stencil-only'
        : isDepth && !hasStencil
          ? 'depth-only'
          : 'all');
    return Object.freeze({
      webgpuFormat,
      blockWidth,
      blockHeight,
      bytesPerBlock,
      aspect,
      isCompressed: Boolean(options.isCompressed),
      isDepth,
      hasStencil,
      isSrgb: Boolean(options.isSrgb),
      isHdr: Boolean(options.isHdr),
      requiresFeature: options.requiresFeature || null,
    });
  }

  // Canonical WebGPU GPUTextureFormat block geometry, aspect, sRGB, HDR, and feature metadata.
  const WEBGPU_FORMAT_INFO = Object.freeze({
    r8unorm: makeWebGpuInfo('r8unorm', 1, 1, 1),
    r8snorm: makeWebGpuInfo('r8snorm', 1, 1, 1),
    r8uint: makeWebGpuInfo('r8uint', 1, 1, 1),
    r8sint: makeWebGpuInfo('r8sint', 1, 1, 1),
    r16unorm: makeWebGpuInfo('r16unorm', 1, 1, 2, { requiresFeature: TEXTURE_FORMATS_TIER1_FEATURE }),
    r16snorm: makeWebGpuInfo('r16snorm', 1, 1, 2, { requiresFeature: TEXTURE_FORMATS_TIER1_FEATURE }),
    r16uint: makeWebGpuInfo('r16uint', 1, 1, 2),
    r16sint: makeWebGpuInfo('r16sint', 1, 1, 2),
    r16float: makeWebGpuInfo('r16float', 1, 1, 2, { isHdr: true }),
    rg8unorm: makeWebGpuInfo('rg8unorm', 1, 1, 2),
    rg8snorm: makeWebGpuInfo('rg8snorm', 1, 1, 2),
    rg8uint: makeWebGpuInfo('rg8uint', 1, 1, 2),
    rg8sint: makeWebGpuInfo('rg8sint', 1, 1, 2),
    r32uint: makeWebGpuInfo('r32uint', 1, 1, 4),
    r32sint: makeWebGpuInfo('r32sint', 1, 1, 4),
    r32float: makeWebGpuInfo('r32float', 1, 1, 4, { isHdr: true }),
    rg16unorm: makeWebGpuInfo('rg16unorm', 1, 1, 4, { requiresFeature: TEXTURE_FORMATS_TIER1_FEATURE }),
    rg16snorm: makeWebGpuInfo('rg16snorm', 1, 1, 4, { requiresFeature: TEXTURE_FORMATS_TIER1_FEATURE }),
    rg16uint: makeWebGpuInfo('rg16uint', 1, 1, 4),
    rg16sint: makeWebGpuInfo('rg16sint', 1, 1, 4),
    rg16float: makeWebGpuInfo('rg16float', 1, 1, 4, { isHdr: true }),
    rgba8unorm: makeWebGpuInfo('rgba8unorm', 1, 1, 4),
    'rgba8unorm-srgb': makeWebGpuInfo('rgba8unorm-srgb', 1, 1, 4, { isSrgb: true }),
    rgba8snorm: makeWebGpuInfo('rgba8snorm', 1, 1, 4),
    rgba8uint: makeWebGpuInfo('rgba8uint', 1, 1, 4),
    rgba8sint: makeWebGpuInfo('rgba8sint', 1, 1, 4),
    bgra8unorm: makeWebGpuInfo('bgra8unorm', 1, 1, 4),
    'bgra8unorm-srgb': makeWebGpuInfo('bgra8unorm-srgb', 1, 1, 4, { isSrgb: true }),
    rgb9e5ufloat: makeWebGpuInfo('rgb9e5ufloat', 1, 1, 4, { isHdr: true }),
    rgb10a2uint: makeWebGpuInfo('rgb10a2uint', 1, 1, 4),
    rgb10a2unorm: makeWebGpuInfo('rgb10a2unorm', 1, 1, 4, { isHdr: true }),
    rg11b10ufloat: makeWebGpuInfo('rg11b10ufloat', 1, 1, 4, { isHdr: true }),
    rg32uint: makeWebGpuInfo('rg32uint', 1, 1, 8),
    rg32sint: makeWebGpuInfo('rg32sint', 1, 1, 8),
    rg32float: makeWebGpuInfo('rg32float', 1, 1, 8, { isHdr: true }),
    rgba16unorm: makeWebGpuInfo('rgba16unorm', 1, 1, 8, { requiresFeature: TEXTURE_FORMATS_TIER1_FEATURE }),
    rgba16snorm: makeWebGpuInfo('rgba16snorm', 1, 1, 8, { requiresFeature: TEXTURE_FORMATS_TIER1_FEATURE }),
    rgba16uint: makeWebGpuInfo('rgba16uint', 1, 1, 8),
    rgba16sint: makeWebGpuInfo('rgba16sint', 1, 1, 8),
    rgba16float: makeWebGpuInfo('rgba16float', 1, 1, 8, { isHdr: true }),
    rgba32uint: makeWebGpuInfo('rgba32uint', 1, 1, 16),
    rgba32sint: makeWebGpuInfo('rgba32sint', 1, 1, 16),
    rgba32float: makeWebGpuInfo('rgba32float', 1, 1, 16, { isHdr: true }),
    stencil8: makeWebGpuInfo('stencil8', 1, 1, 1, { isDepth: true, hasStencil: true, aspect: 'stencil-only' }),
    depth16unorm: makeWebGpuInfo('depth16unorm', 1, 1, 2, { isDepth: true, aspect: 'depth-only' }),
    depth24plus: makeWebGpuInfo('depth24plus', 1, 1, 4, { isDepth: true, aspect: 'depth-only' }),
    'depth24plus-stencil8': makeWebGpuInfo('depth24plus-stencil8', 1, 1, 4, { isDepth: true, hasStencil: true, aspect: 'all' }),
    depth32float: makeWebGpuInfo('depth32float', 1, 1, 4, { isDepth: true, aspect: 'depth-only' }),
    'depth32float-stencil8': makeWebGpuInfo('depth32float-stencil8', 1, 1, 8, {
      isDepth: true,
      hasStencil: true,
      aspect: 'all',
      requiresFeature: DEPTH32FLOAT_STENCIL8_FEATURE,
    }),
    'bc1-rgba-unorm': makeWebGpuInfo('bc1-rgba-unorm', 4, 4, 8, { isCompressed: true, requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE }),
    'bc1-rgba-unorm-srgb': makeWebGpuInfo('bc1-rgba-unorm-srgb', 4, 4, 8, { isCompressed: true, isSrgb: true, requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE }),
    'bc2-rgba-unorm': makeWebGpuInfo('bc2-rgba-unorm', 4, 4, 16, { isCompressed: true, requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE }),
    'bc2-rgba-unorm-srgb': makeWebGpuInfo('bc2-rgba-unorm-srgb', 4, 4, 16, { isCompressed: true, isSrgb: true, requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE }),
    'bc3-rgba-unorm': makeWebGpuInfo('bc3-rgba-unorm', 4, 4, 16, { isCompressed: true, requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE }),
    'bc3-rgba-unorm-srgb': makeWebGpuInfo('bc3-rgba-unorm-srgb', 4, 4, 16, { isCompressed: true, isSrgb: true, requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE }),
    'bc4-r-unorm': makeWebGpuInfo('bc4-r-unorm', 4, 4, 8, { isCompressed: true, requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE }),
    'bc4-r-snorm': makeWebGpuInfo('bc4-r-snorm', 4, 4, 8, { isCompressed: true, requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE }),
    'bc5-rg-unorm': makeWebGpuInfo('bc5-rg-unorm', 4, 4, 16, { isCompressed: true, requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE }),
    'bc5-rg-snorm': makeWebGpuInfo('bc5-rg-snorm', 4, 4, 16, { isCompressed: true, requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE }),
    'bc6h-rgb-ufloat': makeWebGpuInfo('bc6h-rgb-ufloat', 4, 4, 16, { isCompressed: true, isHdr: true, requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE }),
    'bc6h-rgb-float': makeWebGpuInfo('bc6h-rgb-float', 4, 4, 16, { isCompressed: true, isHdr: true, requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE }),
    'bc7-rgba-unorm': makeWebGpuInfo('bc7-rgba-unorm', 4, 4, 16, { isCompressed: true, requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE }),
    'bc7-rgba-unorm-srgb': makeWebGpuInfo('bc7-rgba-unorm-srgb', 4, 4, 16, { isCompressed: true, isSrgb: true, requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE }),
  });

  /**
   * Standard DirectX / Xbox Series DXGI_FORMAT numeric enum values (dxgiformat.h).
   */
  const DXGI_FORMAT = Object.freeze({
    UNKNOWN: 0,
    R32G32B32A32_TYPELESS: 1,
    R32G32B32A32_FLOAT: 2,
    R32G32B32A32_UINT: 3,
    R32G32B32A32_SINT: 4,
    R16G16B16A16_TYPELESS: 9,
    R16G16B16A16_FLOAT: 10,
    R16G16B16A16_UNORM: 11,
    R16G16B16A16_UINT: 12,
    R16G16B16A16_SNORM: 13,
    R16G16B16A16_SINT: 14,
    R32G32_TYPELESS: 15,
    R32G32_FLOAT: 16,
    R32G32_UINT: 17,
    R32G32_SINT: 18,
    R32G8X24_TYPELESS: 19,
    D32_FLOAT_S8X24_UINT: 20,
    R32_FLOAT_X8X24_TYPELESS: 21,
    X32_TYPELESS_G8X24_UINT: 22,
    R10G10B10A2_TYPELESS: 23,
    R10G10B10A2_UNORM: 24,
    R10G10B10A2_UINT: 25,
    R11G11B10_FLOAT: 26,
    R8G8B8A8_TYPELESS: 27,
    R8G8B8A8_UNORM: 28,
    R8G8B8A8_UNORM_SRGB: 29,
    R8G8B8A8_UINT: 30,
    R8G8B8A8_SNORM: 31,
    R8G8B8A8_SINT: 32,
    R16G16_TYPELESS: 33,
    R16G16_FLOAT: 34,
    R16G16_UNORM: 35,
    R16G16_UINT: 36,
    R16G16_SNORM: 37,
    R16G16_SINT: 38,
    R32_TYPELESS: 39,
    D32_FLOAT: 40,
    R32_FLOAT: 41,
    R32_UINT: 42,
    R32_SINT: 43,
    R24G8_TYPELESS: 44,
    D24_UNORM_S8_UINT: 45,
    R24_UNORM_X8_TYPELESS: 46,
    X24_TYPELESS_G8_UINT: 47,
    R8G8_TYPELESS: 48,
    R8G8_UNORM: 49,
    R8G8_UINT: 50,
    R8G8_SNORM: 51,
    R8G8_SINT: 52,
    R16_TYPELESS: 53,
    R16_FLOAT: 54,
    D16_UNORM: 55,
    R16_UNORM: 56,
    R16_UINT: 57,
    R16_SNORM: 58,
    R16_SINT: 59,
    R8_TYPELESS: 60,
    R8_UNORM: 61,
    R8_UINT: 62,
    R8_SNORM: 63,
    R8_SINT: 64,
    A8_UNORM: 65,
    R9G9B9E5_SHAREDEXP: 67,
    BC1_TYPELESS: 70,
    BC1_UNORM: 71,
    BC1_UNORM_SRGB: 72,
    BC2_TYPELESS: 73,
    BC2_UNORM: 74,
    BC2_UNORM_SRGB: 75,
    BC3_TYPELESS: 76,
    BC3_UNORM: 77,
    BC3_UNORM_SRGB: 78,
    BC4_TYPELESS: 79,
    BC4_UNORM: 80,
    BC4_SNORM: 81,
    BC5_TYPELESS: 82,
    BC5_UNORM: 83,
    BC5_SNORM: 84,
    B8G8R8A8_UNORM: 87,
    B8G8R8X8_UNORM: 88,
    B8G8R8A8_TYPELESS: 90,
    B8G8R8A8_UNORM_SRGB: 91,
    B8G8R8X8_TYPELESS: 92,
    B8G8R8X8_UNORM_SRGB: 93,
    BC6H_TYPELESS: 94,
    BC6H_UF16: 95,
    BC6H_SF16: 96,
    BC7_TYPELESS: 97,
    BC7_UNORM: 98,
    BC7_UNORM_SRGB: 99,
  });

  const DXGI_FORMAT_NAMES = Object.freeze(
    Object.fromEntries(
      Object.entries(DXGI_FORMAT).map(([name, code]) => [code, name])
    )
  );

  function makeDescriptor(
    dxgiFormat,
    webgpuFormat,
    nativeWebgpuFormat,
    blockWidth,
    blockHeight,
    bytesPerBlock,
    options = {}
  ) {
    const isDepth = Boolean(options.isDepth);
    const hasStencil = Boolean(options.hasStencil);
    const aspect =
      options.aspect ||
      (isDepth && !hasStencil ? 'depth-only' : 'all');
    return Object.freeze({
      dxgiFormat,
      name: DXGI_FORMAT_NAMES[dxgiFormat],
      webgpuFormat,
      nativeWebgpuFormat,
      blockWidth,
      blockHeight,
      bytesPerBlock,
      aspect,
      isCompressed: Boolean(options.isCompressed),
      isDepth,
      hasStencil,
      isSrgb: Boolean(options.isSrgb),
      isHdr: Boolean(options.isHdr),
      isTypeless: Boolean(options.isTypeless),
      isAlphaOnly: Boolean(options.isAlphaOnly),
      swizzle: options.swizzle || null,
      requiresFeature: options.requiresFeature || null,
      needsSoftwareDecode: Boolean(options.needsSoftwareDecode),
    });
  }

  function makeEntry(
    dxgiFormat,
    webgpuFormat,
    blockWidth,
    blockHeight,
    bytesPerBlock,
    options = {}
  ) {
    return makeDescriptor(
      dxgiFormat,
      webgpuFormat,
      webgpuFormat,
      blockWidth,
      blockHeight,
      bytesPerBlock,
      options
    );
  }

  // Canonical DXGI_FORMAT -> WebGPU GPUTextureFormat descriptor table.
  const FORMAT_TABLE = Object.freeze({
    [DXGI_FORMAT.R32G32B32A32_TYPELESS]: makeEntry(DXGI_FORMAT.R32G32B32A32_TYPELESS, 'rgba32float', 1, 1, 16, { isHdr: true, isTypeless: true }),
    [DXGI_FORMAT.R32G32B32A32_FLOAT]: makeEntry(DXGI_FORMAT.R32G32B32A32_FLOAT, 'rgba32float', 1, 1, 16, { isHdr: true }),
    [DXGI_FORMAT.R32G32B32A32_UINT]: makeEntry(DXGI_FORMAT.R32G32B32A32_UINT, 'rgba32uint', 1, 1, 16),
    [DXGI_FORMAT.R32G32B32A32_SINT]: makeEntry(DXGI_FORMAT.R32G32B32A32_SINT, 'rgba32sint', 1, 1, 16),
    [DXGI_FORMAT.R16G16B16A16_TYPELESS]: makeEntry(DXGI_FORMAT.R16G16B16A16_TYPELESS, 'rgba16float', 1, 1, 8, { isHdr: true, isTypeless: true }),
    [DXGI_FORMAT.R16G16B16A16_FLOAT]: makeEntry(DXGI_FORMAT.R16G16B16A16_FLOAT, 'rgba16float', 1, 1, 8, { isHdr: true }),
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
    [DXGI_FORMAT.R16G16B16A16_SINT]: makeEntry(DXGI_FORMAT.R16G16B16A16_SINT, 'rgba16sint', 1, 1, 8),
    [DXGI_FORMAT.R32G32_TYPELESS]: makeEntry(DXGI_FORMAT.R32G32_TYPELESS, 'rg32float', 1, 1, 8, { isHdr: true, isTypeless: true }),
    [DXGI_FORMAT.R32G32_FLOAT]: makeEntry(DXGI_FORMAT.R32G32_FLOAT, 'rg32float', 1, 1, 8, { isHdr: true }),
    [DXGI_FORMAT.R32G32_UINT]: makeEntry(DXGI_FORMAT.R32G32_UINT, 'rg32uint', 1, 1, 8),
    [DXGI_FORMAT.R32G32_SINT]: makeEntry(DXGI_FORMAT.R32G32_SINT, 'rg32sint', 1, 1, 8),
    [DXGI_FORMAT.R32G8X24_TYPELESS]: makeEntry(
      DXGI_FORMAT.R32G8X24_TYPELESS,
      'depth32float-stencil8',
      1,
      1,
      8,
      { isDepth: true, hasStencil: true, aspect: 'all', isTypeless: true, requiresFeature: DEPTH32FLOAT_STENCIL8_FEATURE }
    ),
    [DXGI_FORMAT.D32_FLOAT_S8X24_UINT]: makeEntry(
      DXGI_FORMAT.D32_FLOAT_S8X24_UINT,
      'depth32float-stencil8',
      1,
      1,
      8,
      { isDepth: true, hasStencil: true, aspect: 'all', requiresFeature: DEPTH32FLOAT_STENCIL8_FEATURE }
    ),
    [DXGI_FORMAT.R32_FLOAT_X8X24_TYPELESS]: makeEntry(
      DXGI_FORMAT.R32_FLOAT_X8X24_TYPELESS,
      'depth32float-stencil8',
      1,
      1,
      8,
      { isDepth: true, hasStencil: true, aspect: 'depth-only', isTypeless: true, requiresFeature: DEPTH32FLOAT_STENCIL8_FEATURE }
    ),
    [DXGI_FORMAT.X32_TYPELESS_G8X24_UINT]: makeEntry(
      DXGI_FORMAT.X32_TYPELESS_G8X24_UINT,
      'depth32float-stencil8',
      1,
      1,
      8,
      { isDepth: true, hasStencil: true, aspect: 'stencil-only', isTypeless: true, requiresFeature: DEPTH32FLOAT_STENCIL8_FEATURE }
    ),
    [DXGI_FORMAT.R10G10B10A2_TYPELESS]: makeEntry(DXGI_FORMAT.R10G10B10A2_TYPELESS, 'rgb10a2unorm', 1, 1, 4, { isHdr: true, isTypeless: true }),
    [DXGI_FORMAT.R10G10B10A2_UNORM]: makeEntry(DXGI_FORMAT.R10G10B10A2_UNORM, 'rgb10a2unorm', 1, 1, 4, { isHdr: true }),
    [DXGI_FORMAT.R10G10B10A2_UINT]: makeEntry(DXGI_FORMAT.R10G10B10A2_UINT, 'rgb10a2uint', 1, 1, 4),
    [DXGI_FORMAT.R11G11B10_FLOAT]: makeEntry(DXGI_FORMAT.R11G11B10_FLOAT, 'rg11b10ufloat', 1, 1, 4, { isHdr: true }),
    [DXGI_FORMAT.R8G8B8A8_TYPELESS]: makeEntry(DXGI_FORMAT.R8G8B8A8_TYPELESS, 'rgba8unorm', 1, 1, 4, { isTypeless: true }),
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
    [DXGI_FORMAT.R16G16_TYPELESS]: makeEntry(DXGI_FORMAT.R16G16_TYPELESS, 'rg16float', 1, 1, 4, { isHdr: true, isTypeless: true }),
    [DXGI_FORMAT.R16G16_FLOAT]: makeEntry(DXGI_FORMAT.R16G16_FLOAT, 'rg16float', 1, 1, 4, { isHdr: true }),
    [DXGI_FORMAT.R16G16_UNORM]: makeEntry(
      DXGI_FORMAT.R16G16_UNORM,
      'rg16unorm',
      1,
      1,
      4,
      { requiresFeature: TEXTURE_FORMATS_TIER1_FEATURE }
    ),
    [DXGI_FORMAT.R16G16_UINT]: makeEntry(DXGI_FORMAT.R16G16_UINT, 'rg16uint', 1, 1, 4),
    [DXGI_FORMAT.R16G16_SNORM]: makeEntry(
      DXGI_FORMAT.R16G16_SNORM,
      'rg16snorm',
      1,
      1,
      4,
      { requiresFeature: TEXTURE_FORMATS_TIER1_FEATURE }
    ),
    [DXGI_FORMAT.R16G16_SINT]: makeEntry(DXGI_FORMAT.R16G16_SINT, 'rg16sint', 1, 1, 4),
    [DXGI_FORMAT.R32_TYPELESS]: makeEntry(DXGI_FORMAT.R32_TYPELESS, 'r32float', 1, 1, 4, { isHdr: true, isTypeless: true }),
    [DXGI_FORMAT.D32_FLOAT]: makeEntry(DXGI_FORMAT.D32_FLOAT, 'depth32float', 1, 1, 4, { isDepth: true, aspect: 'depth-only' }),
    [DXGI_FORMAT.R32_FLOAT]: makeEntry(DXGI_FORMAT.R32_FLOAT, 'r32float', 1, 1, 4, { isHdr: true }),
    [DXGI_FORMAT.R32_UINT]: makeEntry(DXGI_FORMAT.R32_UINT, 'r32uint', 1, 1, 4),
    [DXGI_FORMAT.R32_SINT]: makeEntry(DXGI_FORMAT.R32_SINT, 'r32sint', 1, 1, 4),
    [DXGI_FORMAT.R24G8_TYPELESS]: makeEntry(
      DXGI_FORMAT.R24G8_TYPELESS,
      'depth24plus-stencil8',
      1,
      1,
      4,
      { isDepth: true, hasStencil: true, aspect: 'all', isTypeless: true }
    ),
    [DXGI_FORMAT.D24_UNORM_S8_UINT]: makeEntry(
      DXGI_FORMAT.D24_UNORM_S8_UINT,
      'depth24plus-stencil8',
      1,
      1,
      4,
      { isDepth: true, hasStencil: true, aspect: 'all' }
    ),
    [DXGI_FORMAT.R24_UNORM_X8_TYPELESS]: makeEntry(
      DXGI_FORMAT.R24_UNORM_X8_TYPELESS,
      'depth24plus-stencil8',
      1,
      1,
      4,
      { isDepth: true, hasStencil: true, aspect: 'depth-only', isTypeless: true }
    ),
    [DXGI_FORMAT.X24_TYPELESS_G8_UINT]: makeEntry(
      DXGI_FORMAT.X24_TYPELESS_G8_UINT,
      'depth24plus-stencil8',
      1,
      1,
      4,
      { isDepth: true, hasStencil: true, aspect: 'stencil-only', isTypeless: true }
    ),
    [DXGI_FORMAT.R8G8_TYPELESS]: makeEntry(DXGI_FORMAT.R8G8_TYPELESS, 'rg8unorm', 1, 1, 2, { isTypeless: true }),
    [DXGI_FORMAT.R8G8_UNORM]: makeEntry(DXGI_FORMAT.R8G8_UNORM, 'rg8unorm', 1, 1, 2),
    [DXGI_FORMAT.R8G8_UINT]: makeEntry(DXGI_FORMAT.R8G8_UINT, 'rg8uint', 1, 1, 2),
    [DXGI_FORMAT.R8G8_SNORM]: makeEntry(DXGI_FORMAT.R8G8_SNORM, 'rg8snorm', 1, 1, 2),
    [DXGI_FORMAT.R8G8_SINT]: makeEntry(DXGI_FORMAT.R8G8_SINT, 'rg8sint', 1, 1, 2),
    [DXGI_FORMAT.R16_TYPELESS]: makeEntry(DXGI_FORMAT.R16_TYPELESS, 'r16float', 1, 1, 2, { isHdr: true, isTypeless: true }),
    [DXGI_FORMAT.R16_FLOAT]: makeEntry(DXGI_FORMAT.R16_FLOAT, 'r16float', 1, 1, 2, { isHdr: true }),
    [DXGI_FORMAT.D16_UNORM]: makeEntry(DXGI_FORMAT.D16_UNORM, 'depth16unorm', 1, 1, 2, { isDepth: true, aspect: 'depth-only' }),
    [DXGI_FORMAT.R16_UNORM]: makeEntry(
      DXGI_FORMAT.R16_UNORM,
      'r16unorm',
      1,
      1,
      2,
      { requiresFeature: TEXTURE_FORMATS_TIER1_FEATURE }
    ),
    [DXGI_FORMAT.R16_UINT]: makeEntry(DXGI_FORMAT.R16_UINT, 'r16uint', 1, 1, 2),
    [DXGI_FORMAT.R16_SNORM]: makeEntry(
      DXGI_FORMAT.R16_SNORM,
      'r16snorm',
      1,
      1,
      2,
      { requiresFeature: TEXTURE_FORMATS_TIER1_FEATURE }
    ),
    [DXGI_FORMAT.R16_SINT]: makeEntry(DXGI_FORMAT.R16_SINT, 'r16sint', 1, 1, 2),
    [DXGI_FORMAT.R8_TYPELESS]: makeEntry(DXGI_FORMAT.R8_TYPELESS, 'r8unorm', 1, 1, 1, { isTypeless: true }),
    [DXGI_FORMAT.R8_UNORM]: makeEntry(DXGI_FORMAT.R8_UNORM, 'r8unorm', 1, 1, 1),
    [DXGI_FORMAT.R8_UINT]: makeEntry(DXGI_FORMAT.R8_UINT, 'r8uint', 1, 1, 1),
    [DXGI_FORMAT.R8_SNORM]: makeEntry(DXGI_FORMAT.R8_SNORM, 'r8snorm', 1, 1, 1),
    [DXGI_FORMAT.R8_SINT]: makeEntry(DXGI_FORMAT.R8_SINT, 'r8sint', 1, 1, 1),
    [DXGI_FORMAT.A8_UNORM]: makeEntry(DXGI_FORMAT.A8_UNORM, 'r8unorm', 1, 1, 1, {
      isAlphaOnly: true,
      swizzle: '000r',
    }),
    [DXGI_FORMAT.R9G9B9E5_SHAREDEXP]: makeEntry(DXGI_FORMAT.R9G9B9E5_SHAREDEXP, 'rgb9e5ufloat', 1, 1, 4, { isHdr: true }),
    [DXGI_FORMAT.B8G8R8A8_UNORM]: makeEntry(DXGI_FORMAT.B8G8R8A8_UNORM, 'bgra8unorm', 1, 1, 4),
    [DXGI_FORMAT.B8G8R8X8_UNORM]: makeEntry(DXGI_FORMAT.B8G8R8X8_UNORM, 'bgra8unorm', 1, 1, 4, { swizzle: 'rgb1' }),
    [DXGI_FORMAT.B8G8R8A8_TYPELESS]: makeEntry(DXGI_FORMAT.B8G8R8A8_TYPELESS, 'bgra8unorm', 1, 1, 4, { isTypeless: true }),
    [DXGI_FORMAT.B8G8R8A8_UNORM_SRGB]: makeEntry(
      DXGI_FORMAT.B8G8R8A8_UNORM_SRGB,
      'bgra8unorm-srgb',
      1,
      1,
      4,
      { isSrgb: true }
    ),
    [DXGI_FORMAT.B8G8R8X8_TYPELESS]: makeEntry(DXGI_FORMAT.B8G8R8X8_TYPELESS, 'bgra8unorm', 1, 1, 4, { isTypeless: true, swizzle: 'rgb1' }),
    [DXGI_FORMAT.B8G8R8X8_UNORM_SRGB]: makeEntry(
      DXGI_FORMAT.B8G8R8X8_UNORM_SRGB,
      'bgra8unorm-srgb',
      1,
      1,
      4,
      { isSrgb: true, swizzle: 'rgb1' }
    ),

    // BC1..BC7 4x4 block-compressed formats (require 'texture-compression-bc' in WebGPU).
    [DXGI_FORMAT.BC1_TYPELESS]: makeEntry(DXGI_FORMAT.BC1_TYPELESS, 'bc1-rgba-unorm', 4, 4, 8, {
      isCompressed: true,
      isTypeless: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC1_UNORM]: makeEntry(DXGI_FORMAT.BC1_UNORM, 'bc1-rgba-unorm', 4, 4, 8, {
      isCompressed: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC1_UNORM_SRGB]: makeEntry(DXGI_FORMAT.BC1_UNORM_SRGB, 'bc1-rgba-unorm-srgb', 4, 4, 8, {
      isCompressed: true,
      isSrgb: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC2_TYPELESS]: makeEntry(DXGI_FORMAT.BC2_TYPELESS, 'bc2-rgba-unorm', 4, 4, 16, {
      isCompressed: true,
      isTypeless: true,
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
    [DXGI_FORMAT.BC3_TYPELESS]: makeEntry(DXGI_FORMAT.BC3_TYPELESS, 'bc3-rgba-unorm', 4, 4, 16, {
      isCompressed: true,
      isTypeless: true,
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
    [DXGI_FORMAT.BC4_TYPELESS]: makeEntry(DXGI_FORMAT.BC4_TYPELESS, 'bc4-r-unorm', 4, 4, 8, {
      isCompressed: true,
      isTypeless: true,
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
    [DXGI_FORMAT.BC5_TYPELESS]: makeEntry(DXGI_FORMAT.BC5_TYPELESS, 'bc5-rg-unorm', 4, 4, 16, {
      isCompressed: true,
      isTypeless: true,
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
    [DXGI_FORMAT.BC6H_TYPELESS]: makeEntry(DXGI_FORMAT.BC6H_TYPELESS, 'bc6h-rgb-ufloat', 4, 4, 16, {
      isCompressed: true,
      isHdr: true,
      isTypeless: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC6H_UF16]: makeEntry(DXGI_FORMAT.BC6H_UF16, 'bc6h-rgb-ufloat', 4, 4, 16, {
      isCompressed: true,
      isHdr: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC6H_SF16]: makeEntry(DXGI_FORMAT.BC6H_SF16, 'bc6h-rgb-float', 4, 4, 16, {
      isCompressed: true,
      isHdr: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC7_TYPELESS]: makeEntry(DXGI_FORMAT.BC7_TYPELESS, 'bc7-rgba-unorm', 4, 4, 16, {
      isCompressed: true,
      isTypeless: true,
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

  // Precomputed O(1) dense lookup tables indexed by DXGI_FORMAT numeric ID 0..132.
  // Every valid slot holds a frozen descriptor; unsupported slots hold null (no holes).
  // Both native and feature-fallback variants are prebuilt so hot lookups allocate 0 objects.
  const _dxgiTable = new Array(DXGI_FORMAT_MAX_ID + 1).fill(null);
  const _dxgiFallbackTable = new Array(DXGI_FORMAT_MAX_ID + 1).fill(null);
  for (let code = 0; code <= DXGI_FORMAT_MAX_ID; code += 1) {
    if (!Object.prototype.hasOwnProperty.call(FORMAT_TABLE, code)) {
      continue;
    }
    const entry = FORMAT_TABLE[code];
    _dxgiTable[code] = entry;
    if (entry.requiresFeature === TEXTURE_COMPRESSION_BC_FEATURE) {
      _dxgiFallbackTable[code] = makeDescriptor(
        entry.dxgiFormat,
        entry.isSrgb ? 'rgba8unorm-srgb' : 'rgba8unorm',
        entry.nativeWebgpuFormat,
        entry.blockWidth,
        entry.blockHeight,
        entry.bytesPerBlock,
        {
          aspect: entry.aspect,
          isCompressed: entry.isCompressed,
          isDepth: entry.isDepth,
          hasStencil: entry.hasStencil,
          isSrgb: entry.isSrgb,
          isHdr: entry.isHdr,
          isTypeless: entry.isTypeless,
          isAlphaOnly: entry.isAlphaOnly,
          swizzle: entry.swizzle,
          requiresFeature: entry.requiresFeature,
          needsSoftwareDecode: true,
        }
      );
    } else if (entry.requiresFeature === DEPTH32FLOAT_STENCIL8_FEATURE) {
      _dxgiFallbackTable[code] = makeDescriptor(
        entry.dxgiFormat,
        'depth24plus-stencil8',
        entry.nativeWebgpuFormat,
        entry.blockWidth,
        entry.blockHeight,
        entry.bytesPerBlock,
        {
          aspect: entry.aspect,
          isCompressed: entry.isCompressed,
          isDepth: entry.isDepth,
          hasStencil: entry.hasStencil,
          isSrgb: entry.isSrgb,
          isHdr: entry.isHdr,
          isTypeless: entry.isTypeless,
          isAlphaOnly: entry.isAlphaOnly,
          swizzle: entry.swizzle,
          requiresFeature: entry.requiresFeature,
          needsSoftwareDecode: false,
        }
      );
    } else if (
      entry.requiresFeature === TEXTURE_FORMATS_TIER1_FEATURE &&
      NORM16_TEXTURE_FORMATS.has(entry.webgpuFormat)
    ) {
      const fallbackFmt = entry.webgpuFormat.startsWith('rgba')
        ? 'rgba16float'
        : entry.webgpuFormat.startsWith('rg')
          ? 'rg16float'
          : 'r16float';
      _dxgiFallbackTable[code] = makeDescriptor(
        entry.dxgiFormat,
        fallbackFmt,
        entry.nativeWebgpuFormat,
        entry.blockWidth,
        entry.blockHeight,
        entry.bytesPerBlock,
        {
          aspect: entry.aspect,
          isCompressed: entry.isCompressed,
          isDepth: entry.isDepth,
          hasStencil: entry.hasStencil,
          isSrgb: entry.isSrgb,
          isHdr: entry.isHdr,
          isTypeless: entry.isTypeless,
          isAlphaOnly: entry.isAlphaOnly,
          swizzle: entry.swizzle,
          requiresFeature: entry.requiresFeature,
          needsSoftwareDecode: false,
        }
      );
    } else {
      _dxgiFallbackTable[code] = entry;
    }
  }
  const DXGI_FORMAT_TABLE = Object.freeze(_dxgiTable);
  const DXGI_FALLBACK_TABLE = Object.freeze(_dxgiFallbackTable);

  // Precomputed O(1) null-prototype name -> DXGI numeric ID map (bare, DXGI_FORMAT_-prefixed,
  // and common HDR/RT shorthand aliases) for zero-alloc string lookups.
  const DXGI_NAME_TO_CODE = Object.create(null);
  for (const [name, code] of Object.entries(DXGI_FORMAT)) {
    if (code > 0 && code <= DXGI_FORMAT_MAX_ID && DXGI_FORMAT_TABLE[code] !== null) {
      DXGI_NAME_TO_CODE[name] = code;
      DXGI_NAME_TO_CODE['DXGI_FORMAT_' + name] = code;
    }
  }
  const DXGI_SHORTHAND_ALIASES = [
    ['RGBA32_FLOAT', DXGI_FORMAT.R32G32B32A32_FLOAT],
    ['RGBA32F', DXGI_FORMAT.R32G32B32A32_FLOAT],
    ['RGBA16_FLOAT', DXGI_FORMAT.R16G16B16A16_FLOAT],
    ['RGBA16F', DXGI_FORMAT.R16G16B16A16_FLOAT],
    ['RG32_FLOAT', DXGI_FORMAT.R32G32_FLOAT],
    ['RG16_FLOAT', DXGI_FORMAT.R16G16_FLOAT],
    ['RG16F', DXGI_FORMAT.R16G16_FLOAT],
    ['R16F', DXGI_FORMAT.R16_FLOAT],
    ['RGB10A2_UNORM', DXGI_FORMAT.R10G10B10A2_UNORM],
    ['RG11B10_FLOAT', DXGI_FORMAT.R11G11B10_FLOAT],
    ['RGBA8_UNORM', DXGI_FORMAT.R8G8B8A8_UNORM],
    ['RGBA8_UNORM_SRGB', DXGI_FORMAT.R8G8B8A8_UNORM_SRGB],
    ['BGRA8_UNORM', DXGI_FORMAT.B8G8R8A8_UNORM],
    ['BGRA8_UNORM_SRGB', DXGI_FORMAT.B8G8R8A8_UNORM_SRGB],
  ];
  for (let i = 0; i < DXGI_SHORTHAND_ALIASES.length; i += 1) {
    const aliasName = DXGI_SHORTHAND_ALIASES[i][0];
    const aliasCode = DXGI_SHORTHAND_ALIASES[i][1];
    DXGI_NAME_TO_CODE[aliasName] = aliasCode;
    DXGI_NAME_TO_CODE['DXGI_FORMAT_' + aliasName] = aliasCode;
  }
  Object.freeze(DXGI_NAME_TO_CODE);

  // Frozen classification & storage result singletons to eliminate hot-path allocations.
  const SUPPORT_CORE = Object.freeze({
    supported: true,
    requiredFeature: null,
    fallbackFormat: null,
    reason: null,
  });
  const SUPPORT_INVALID = Object.freeze({
    supported: false,
    requiredFeature: null,
    fallbackFormat: null,
    reason: 'invalid-format',
  });
  const SUPPORT_BC_OK = Object.freeze({
    supported: true,
    requiredFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    fallbackFormat: null,
    reason: null,
  });
  const SUPPORT_BC_MISSING = Object.freeze({
    supported: false,
    requiredFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    fallbackFormat: 'rgba8unorm',
    reason: 'missing-bc',
  });
  const SUPPORT_BC3D_OK = Object.freeze({
    supported: true,
    requiredFeature: TEXTURE_COMPRESSION_BC_SLICED_3D_FEATURE,
    fallbackFormat: null,
    reason: null,
  });
  const SUPPORT_BC3D_MISSING = Object.freeze({
    supported: false,
    requiredFeature: TEXTURE_COMPRESSION_BC_SLICED_3D_FEATURE,
    fallbackFormat: 'rgba8unorm',
    reason: 'missing-bc-sliced-3d',
  });
  const SUPPORT_NORM16_TIER1_OK = Object.freeze({
    supported: true,
    requiredFeature: TEXTURE_FORMATS_TIER1_FEATURE,
    fallbackFormat: null,
    reason: null,
  });
  const SUPPORT_NORM16_CHROMIUM_OK = Object.freeze({
    supported: true,
    requiredFeature: CHROMIUM_UNORM16_FEATURE,
    fallbackFormat: null,
    reason: null,
  });
  const SUPPORT_NORM16_MISSING_R = Object.freeze({
    supported: false,
    requiredFeature: TEXTURE_FORMATS_TIER1_FEATURE,
    fallbackFormat: 'r16float',
    reason: 'missing-unorm16',
  });
  const SUPPORT_NORM16_MISSING_RG = Object.freeze({
    supported: false,
    requiredFeature: TEXTURE_FORMATS_TIER1_FEATURE,
    fallbackFormat: 'rg16float',
    reason: 'missing-unorm16',
  });
  const SUPPORT_NORM16_MISSING_RGBA = Object.freeze({
    supported: false,
    requiredFeature: TEXTURE_FORMATS_TIER1_FEATURE,
    fallbackFormat: 'rgba16float',
    reason: 'missing-unorm16',
  });
  const SUPPORT_D32S8_OK = Object.freeze({
    supported: true,
    requiredFeature: DEPTH32FLOAT_STENCIL8_FEATURE,
    fallbackFormat: null,
    reason: null,
  });
  const SUPPORT_D32S8_MISSING = Object.freeze({
    supported: false,
    requiredFeature: DEPTH32FLOAT_STENCIL8_FEATURE,
    fallbackFormat: 'depth24plus-stencil8',
    reason: 'missing-depth32float-stencil8',
  });

  const STORAGE_OK_BY_FORMAT = Object.create(null);
  for (const fmt of CORE_STORAGE_TEXTURE_FORMATS) {
    STORAGE_OK_BY_FORMAT[fmt] = Object.freeze({
      storageFormat: fmt,
      supportsStorageBinding: true,
      needsLinearSiblingView: false,
      reason: null,
    });
  }
  for (const fmt of TIER1_STORAGE_TEXTURE_FORMATS) {
    if (!STORAGE_OK_BY_FORMAT[fmt]) {
      STORAGE_OK_BY_FORMAT[fmt] = Object.freeze({
        storageFormat: fmt,
        supportsStorageBinding: true,
        needsLinearSiblingView: false,
        reason: null,
      });
    }
  }
  STORAGE_OK_BY_FORMAT.bgra8unorm = Object.freeze({
    storageFormat: 'bgra8unorm',
    supportsStorageBinding: true,
    needsLinearSiblingView: false,
    reason: null,
  });
  Object.freeze(STORAGE_OK_BY_FORMAT);

  const STORAGE_INVALID = Object.freeze({
    storageFormat: null,
    supportsStorageBinding: false,
    needsLinearSiblingView: false,
    reason: 'invalid-format',
  });
  const STORAGE_DEPTH_STENCIL = Object.freeze({
    storageFormat: null,
    supportsStorageBinding: false,
    needsLinearSiblingView: false,
    reason: 'depth-stencil-not-storage-compatible',
  });
  const STORAGE_COMPRESSED = Object.freeze({
    storageFormat: null,
    supportsStorageBinding: false,
    needsLinearSiblingView: false,
    reason: 'compressed-not-storage-compatible',
  });
  const STORAGE_MISSING_BGRA = Object.freeze({
    storageFormat: null,
    supportsStorageBinding: false,
    needsLinearSiblingView: false,
    reason: 'missing-bgra8unorm-storage',
  });
  const STORAGE_MISSING_TIER1 = Object.freeze({
    storageFormat: null,
    supportsStorageBinding: false,
    needsLinearSiblingView: false,
    reason: 'missing-texture-formats-tier1',
  });
  const STORAGE_SRGB_RGBA8 = Object.freeze({
    storageFormat: 'rgba8unorm',
    supportsStorageBinding: true,
    needsLinearSiblingView: true,
    reason: 'srgb-requires-linear-view',
  });
  const STORAGE_SRGB_BGRA8 = Object.freeze({
    storageFormat: 'bgra8unorm',
    supportsStorageBinding: true,
    needsLinearSiblingView: true,
    reason: 'srgb-requires-linear-view',
  });
  const STORAGE_SRGB_UNSUPPORTED = Object.freeze({
    storageFormat: null,
    supportsStorageBinding: false,
    needsLinearSiblingView: false,
    reason: 'srgb-not-storage-compatible',
  });
  const STORAGE_UNSUPPORTED = Object.freeze({
    storageFormat: null,
    supportsStorageBinding: false,
    needsLinearSiblingView: false,
    reason: 'unsupported-storage-format',
  });

  const ZERO_MIP_LAYOUT = Object.freeze({
    width: 0,
    height: 0,
    mipLevel: 0,
    copyWidth: 0,
    copyHeight: 0,
    blocksX: 0,
    blocksY: 0,
    blockWidth: 0,
    blockHeight: 0,
    bytesPerBlock: 0,
    unalignedBytesPerRow: 0,
    bytesPerRow: 0,
    rowsPerImage: 0,
    tightBytes: 0,
    totalBytes: 0,
    byteSize: 0,
  });

  function resolveDxgiCodeFast(dxgiFormat) {
    if (typeof dxgiFormat === 'number') {
      if (dxgiFormat >= 0 && dxgiFormat <= DXGI_FORMAT_MAX_ID && (dxgiFormat | 0) === dxgiFormat) {
        return DXGI_FORMAT_TABLE[dxgiFormat] !== null ? dxgiFormat : -1;
      }
      return -1;
    }
    if (typeof dxgiFormat === 'string') {
      const direct = DXGI_NAME_TO_CODE[dxgiFormat];
      if (direct !== undefined) {
        return direct;
      }
      const normalized = dxgiFormat.trim().toUpperCase();
      const hit = DXGI_NAME_TO_CODE[normalized];
      if (hit !== undefined) {
        return hit;
      }
      if (normalized.startsWith('DXGI_FORMAT_')) {
        const stripped = normalized.slice(12);
        const strippedHit = DXGI_NAME_TO_CODE[stripped];
        if (strippedHit !== undefined) {
          return strippedHit;
        }
      }
      return -1;
    }
    return -2;
  }

  function resolveDxgiFormatCode(dxgiFormat) {
    const code = resolveDxgiCodeFast(dxgiFormat);
    if (code >= 0) {
      return code;
    }
    if (code === -2) {
      throw new TypeError(`dxgiFormat must be a DXGI_FORMAT integer or name string`);
    }
    if (typeof dxgiFormat === 'number') {
      throw new RangeError(`Unsupported DXGI_FORMAT value: ${dxgiFormat}`);
    }
    throw new RangeError(`Unsupported DXGI_FORMAT name: ${JSON.stringify(dxgiFormat)}`);
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
    if (typeof formatOrInfo === 'number') {
      if (
        formatOrInfo >= 0 &&
        formatOrInfo <= DXGI_FORMAT_MAX_ID &&
        (formatOrInfo | 0) === formatOrInfo
      ) {
        const entry = DXGI_FORMAT_TABLE[formatOrInfo];
        if (entry !== null) {
          return entry;
        }
      }
      throw new RangeError(`Unsupported DXGI_FORMAT value: ${formatOrInfo}`);
    }
    if (typeof formatOrInfo === 'string') {
      if (Object.prototype.hasOwnProperty.call(WEBGPU_FORMAT_INFO, formatOrInfo)) {
        return WEBGPU_FORMAT_INFO[formatOrInfo];
      }
      const directCode = DXGI_NAME_TO_CODE[formatOrInfo];
      if (directCode !== undefined) {
        return DXGI_FORMAT_TABLE[directCode];
      }
      const lowered = formatOrInfo.trim().toLowerCase();
      if (Object.prototype.hasOwnProperty.call(WEBGPU_FORMAT_INFO, lowered)) {
        return WEBGPU_FORMAT_INFO[lowered];
      }
      return DXGI_FORMAT_TABLE[resolveDxgiFormatCode(formatOrInfo)];
    }
    if (formatOrInfo && typeof formatOrInfo === 'object') {
      return formatOrInfo;
    }
    throw new TypeError('formatInfo must be a format descriptor object, WebGPU format name, or DXGI_FORMAT');
  }

  /**
   * Look up WebGPU block geometry (`blockWidth`, `blockHeight`, `bytesPerBlock`,
   * `aspect`, `isCompressed`, `isDepth`, `hasStencil`, `isSrgb`, `isHdr`,
   * `requiresFeature`) by WebGPU format string, DXGI_FORMAT code/name,
   * or format descriptor object. Returns `null` for unknown/invalid values.
   *
   * @param {string | number | object} formatOrInfo
   */
  function getWebGpuFormatInfo(formatOrInfo) {
    if (typeof formatOrInfo === 'number') {
      return formatOrInfo >= 0 &&
        formatOrInfo <= DXGI_FORMAT_MAX_ID &&
        (formatOrInfo | 0) === formatOrInfo
        ? DXGI_FORMAT_TABLE[formatOrInfo]
        : null;
    }
    if (typeof formatOrInfo === 'string') {
      if (Object.prototype.hasOwnProperty.call(WEBGPU_FORMAT_INFO, formatOrInfo)) {
        return WEBGPU_FORMAT_INFO[formatOrInfo];
      }
      const directCode = DXGI_NAME_TO_CODE[formatOrInfo];
      if (directCode !== undefined) {
        return DXGI_FORMAT_TABLE[directCode];
      }
      const lowered = formatOrInfo.trim().toLowerCase();
      if (Object.prototype.hasOwnProperty.call(WEBGPU_FORMAT_INFO, lowered)) {
        return WEBGPU_FORMAT_INFO[lowered];
      }
      const code = resolveDxgiCodeFast(formatOrInfo);
      return code >= 0 ? DXGI_FORMAT_TABLE[code] : null;
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

  function selectDxgiDescriptorForFeatures(code, supportedFeatures) {
    const entry = DXGI_FORMAT_TABLE[code];
    const req = entry.requiresFeature;
    if (req === null) {
      return entry;
    }
    if (req === TEXTURE_COMPRESSION_BC_FEATURE) {
      return hasSupportedFeature(supportedFeatures, TEXTURE_COMPRESSION_BC_FEATURE)
        ? entry
        : DXGI_FALLBACK_TABLE[code];
    }
    if (req === DEPTH32FLOAT_STENCIL8_FEATURE) {
      return hasSupportedFeature(supportedFeatures, DEPTH32FLOAT_STENCIL8_FEATURE)
        ? entry
        : DXGI_FALLBACK_TABLE[code];
    }
    if (req === TEXTURE_FORMATS_TIER1_FEATURE) {
      return hasSupportedFeature(supportedFeatures, TEXTURE_FORMATS_TIER1_FEATURE) ||
        hasSupportedFeature(supportedFeatures, CHROMIUM_UNORM16_FEATURE)
        ? entry
        : DXGI_FALLBACK_TABLE[code];
    }
    return entry;
  }

  /**
   * Non-throwing O(1) zero-allocation lookup into the precomputed dense DXGI
   * format table (`0..132`). Returns a frozen descriptor record (or `null` for
   * unsupported/out-of-range/invalid inputs).
   *
   * @param {number | string} dxgiFormat
   * @param {Set<string> | string[] | Iterable<string> | Record<string, boolean> | null} [supportedFeatures]
   */
  function lookupDxgiFormat(dxgiFormat, supportedFeatures) {
    const code = resolveDxgiCodeFast(dxgiFormat);
    if (code < 0) {
      return null;
    }
    return selectDxgiDescriptorForFeatures(code, supportedFeatures);
  }

  /**
   * Map a DXGI / Xbox texture format to WebGPU GPUTextureFormat metadata.
   *
   * Uses the precomputed O(1) dense lookup table (`0..132`) of frozen descriptor
   * records so hot texture-creation and upload calls allocate zero objects.
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
    return selectDxgiDescriptorForFeatures(code, supportedFeatures);
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
    if (Object.prototype.hasOwnProperty.call(SRGB_SIBLING_FORMATS, format)) {
      return SRGB_SIBLING_FORMATS[format];
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
  function classifyTextureFormatSupport(format, options) {
    if (typeof format !== 'string') {
      return SUPPORT_INVALID;
    }

    const normalized = Object.prototype.hasOwnProperty.call(WEBGPU_FORMAT_INFO, format)
      ? format
      : format.trim().toLowerCase();
    if (!normalized) {
      return SUPPORT_INVALID;
    }

    const opts = options && typeof options === 'object' ? options : null;
    const rawDim = opts && typeof opts.dimension === 'string' ? opts.dimension : '2d';
    const dimension =
      rawDim === '2d' || rawDim === '3d' || rawDim === '1d'
        ? rawDim
        : rawDim.trim().toLowerCase() || '2d';
    const deviceFeatures = opts && opts.deviceFeatures !== undefined ? opts.deviceFeatures : null;

    const info = Object.prototype.hasOwnProperty.call(WEBGPU_FORMAT_INFO, normalized)
      ? WEBGPU_FORMAT_INFO[normalized]
      : null;

    if ((info && info.requiresFeature === TEXTURE_COMPRESSION_BC_FEATURE) || (!info && BC_FORMAT_PATTERN.test(normalized))) {
      if (dimension === '3d') {
        return hasSupportedFeature(deviceFeatures, TEXTURE_COMPRESSION_BC_SLICED_3D_FEATURE)
          ? SUPPORT_BC3D_OK
          : SUPPORT_BC3D_MISSING;
      }
      return hasSupportedFeature(deviceFeatures, TEXTURE_COMPRESSION_BC_FEATURE)
        ? SUPPORT_BC_OK
        : SUPPORT_BC_MISSING;
    }

    if (NORM16_TEXTURE_FORMATS.has(normalized)) {
      if (hasSupportedFeature(deviceFeatures, TEXTURE_FORMATS_TIER1_FEATURE)) {
        return SUPPORT_NORM16_TIER1_OK;
      }
      if (hasSupportedFeature(deviceFeatures, CHROMIUM_UNORM16_FEATURE)) {
        return SUPPORT_NORM16_CHROMIUM_OK;
      }
      return normalized.startsWith('rgba')
        ? SUPPORT_NORM16_MISSING_RGBA
        : normalized.startsWith('rg')
          ? SUPPORT_NORM16_MISSING_RG
          : SUPPORT_NORM16_MISSING_R;
    }

    if (normalized === 'depth32float-stencil8') {
      return hasSupportedFeature(deviceFeatures, DEPTH32FLOAT_STENCIL8_FEATURE)
        ? SUPPORT_D32S8_OK
        : SUPPORT_D32S8_MISSING;
    }

    return SUPPORT_CORE;
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
      return STORAGE_INVALID;
    }

    const normalized = Object.prototype.hasOwnProperty.call(WEBGPU_FORMAT_INFO, webgpuFormat)
      ? webgpuFormat
      : webgpuFormat.trim().toLowerCase();
    if (!normalized) {
      return STORAGE_INVALID;
    }

    if (DEPTH_STENCIL_TEXTURE_FORMATS.has(normalized)) {
      return STORAGE_DEPTH_STENCIL;
    }

    const info = Object.prototype.hasOwnProperty.call(WEBGPU_FORMAT_INFO, normalized)
      ? WEBGPU_FORMAT_INFO[normalized]
      : null;
    if ((info && info.isCompressed) || (!info && COMPRESSED_FORMAT_PATTERN.test(normalized))) {
      return STORAGE_COMPRESSED;
    }

    if (CORE_STORAGE_TEXTURE_FORMATS.has(normalized)) {
      return STORAGE_OK_BY_FORMAT[normalized];
    }

    const deviceFeatures = extractDeviceFeatures(optionsOrFeatures);

    if (normalized === 'bgra8unorm') {
      return hasSupportedFeature(deviceFeatures, BGRA8UNORM_STORAGE_FEATURE)
        ? STORAGE_OK_BY_FORMAT.bgra8unorm
        : STORAGE_MISSING_BGRA;
    }

    if (TIER1_STORAGE_TEXTURE_FORMATS.has(normalized)) {
      return hasSupportedFeature(deviceFeatures, TEXTURE_FORMATS_TIER1_FEATURE)
        ? STORAGE_OK_BY_FORMAT[normalized]
        : STORAGE_MISSING_TIER1;
    }

    if (normalized.endsWith('-srgb')) {
      const linearSibling = getSrgbSiblingFormat(normalized);
      if (linearSibling === 'rgba8unorm') {
        return STORAGE_SRGB_RGBA8;
      }
      if (linearSibling === 'bgra8unorm') {
        return hasSupportedFeature(deviceFeatures, BGRA8UNORM_STORAGE_FEATURE)
          ? STORAGE_SRGB_BGRA8
          : STORAGE_MISSING_BGRA;
      }
      return STORAGE_SRGB_UNSUPPORTED;
    }

    return STORAGE_UNSUPPORTED;
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

  function sanitizeMipLevel(mipLevel) {
    if (mipLevel === undefined || mipLevel === null) {
      return 0;
    }
    if (typeof mipLevel !== 'number') {
      return 0;
    }
    if (mipLevel === Infinity) {
      return 31;
    }
    if (!Number.isFinite(mipLevel) || mipLevel <= 0) {
      return 0;
    }
    return Math.floor(mipLevel);
  }

  function shrinkDimForMip(dim, mip) {
    if (mip <= 0) {
      return dim;
    }
    if (mip >= 31) {
      return 1;
    }
    if (dim <= 0x7fffffff) {
      const shifted = dim >>> mip;
      return shifted > 1 ? shifted : 1;
    }
    const shrunk = Math.floor(dim / Math.pow(2, mip));
    return shrunk > 1 ? shrunk : 1;
  }

  function alignBytesPerRowValue(unalignedBytesPerRow, alignBytesPerRow) {
    if (unalignedBytesPerRow <= 0) {
      return 0;
    }
    if (alignBytesPerRow === WEBGPU_BYTES_PER_ROW_ALIGNMENT && unalignedBytesPerRow <= 0x7fffff00) {
      return (unalignedBytesPerRow + 255) & ~255;
    }
    if (alignBytesPerRow <= 1) {
      return unalignedBytesPerRow;
    }
    return Math.ceil(unalignedBytesPerRow / alignBytesPerRow) * alignBytesPerRow;
  }

  function writeZeroMipLayout(out, mip, blockWidth, blockHeight, bytesPerBlock) {
    out.width = 0;
    out.height = 0;
    out.mipLevel = mip;
    out.copyWidth = 0;
    out.copyHeight = 0;
    out.blocksX = 0;
    out.blocksY = 0;
    out.blockWidth = blockWidth;
    out.blockHeight = blockHeight;
    out.bytesPerBlock = bytesPerBlock;
    out.unalignedBytesPerRow = 0;
    out.bytesPerRow = 0;
    out.rowsPerImage = 0;
    out.tightBytes = 0;
    out.totalBytes = 0;
    out.byteSize = 0;
    return out;
  }

  /**
   * Pure, non-throwing mip-level layout helper handling BC 4x4 block rounding
   * and WebGPU 256-byte `bytesPerRow` alignment.
   *
   * - Never throws on degenerate (`0`, negative, `NaN`, `Infinity`) dimensions,
   *   negative/`NaN` `mipLevel`, or unknown formats; returns zeroed metrics.
   * - Pass a reusable `out` object to populate it in-place with ZERO heap
   *   allocations on hot 30 FPS texture upload loops.
   *
   * @param {number | string | object} dxgiOrWebGpuFormat
   * @param {number} width
   * @param {number} height
   * @param {number} [mipLevel=0]
   * @param {object | null} [out=null]
   * @param {number} [alignBytesPerRow=256]
   */
  function computeMipLayout(
    dxgiOrWebGpuFormat,
    width,
    height,
    mipLevel = 0,
    out = null,
    alignBytesPerRow = WEBGPU_BYTES_PER_ROW_ALIGNMENT
  ) {
    const target = out && typeof out === 'object' ? out : null;
    const info = getWebGpuFormatInfo(dxgiOrWebGpuFormat);
    const mip = sanitizeMipLevel(mipLevel);

    if (
      !info ||
      typeof width !== 'number' ||
      typeof height !== 'number' ||
      !Number.isFinite(width) ||
      !Number.isFinite(height) ||
      width < 1 ||
      height < 1
    ) {
      if (target) {
        return writeZeroMipLayout(
          target,
          mip,
          info ? info.blockWidth : 0,
          info ? info.blockHeight : 0,
          info ? info.bytesPerBlock : 0
        );
      }
      return ZERO_MIP_LAYOUT;
    }

    const baseWidth = Math.floor(width);
    const baseHeight = Math.floor(height);
    if (baseWidth < 1 || baseHeight < 1) {
      if (target) {
        return writeZeroMipLayout(target, mip, info.blockWidth, info.blockHeight, info.bytesPerBlock);
      }
      return ZERO_MIP_LAYOUT;
    }

    const align =
      typeof alignBytesPerRow === 'number' &&
      Number.isFinite(alignBytesPerRow) &&
      alignBytesPerRow >= 1
        ? Math.floor(alignBytesPerRow)
        : WEBGPU_BYTES_PER_ROW_ALIGNMENT;

    const mipWidth = shrinkDimForMip(baseWidth, mip);
    const mipHeight = shrinkDimForMip(baseHeight, mip);
    const blockWidth = info.blockWidth;
    const blockHeight = info.blockHeight;
    const bytesPerBlock = info.bytesPerBlock;

    let blocksX;
    let blocksY;
    if (blockWidth === 1 && blockHeight === 1) {
      blocksX = mipWidth;
      blocksY = mipHeight;
    } else if (blockWidth === 4 && blockHeight === 4 && mipWidth <= 0x7ffffffc && mipHeight <= 0x7ffffffc) {
      blocksX = (mipWidth + 3) >>> 2;
      blocksY = (mipHeight + 3) >>> 2;
    } else {
      blocksX = Math.ceil(mipWidth / blockWidth);
      blocksY = Math.ceil(mipHeight / blockHeight);
    }

    const copyWidth = blocksX * blockWidth;
    const copyHeight = blocksY * blockHeight;
    const unalignedBytesPerRow = blocksX * bytesPerBlock;
    const bytesPerRow = alignBytesPerRowValue(unalignedBytesPerRow, align);
    const rowsPerImage = blocksY;
    const tightBytes = unalignedBytesPerRow * rowsPerImage;
    const totalBytes = bytesPerRow * rowsPerImage;

    if (target) {
      target.width = mipWidth;
      target.height = mipHeight;
      target.mipLevel = mip;
      target.copyWidth = copyWidth;
      target.copyHeight = copyHeight;
      target.blocksX = blocksX;
      target.blocksY = blocksY;
      target.blockWidth = blockWidth;
      target.blockHeight = blockHeight;
      target.bytesPerBlock = bytesPerBlock;
      target.unalignedBytesPerRow = unalignedBytesPerRow;
      target.bytesPerRow = bytesPerRow;
      target.rowsPerImage = rowsPerImage;
      target.tightBytes = tightBytes;
      target.totalBytes = totalBytes;
      target.byteSize = totalBytes;
      return target;
    }

    return {
      width: mipWidth,
      height: mipHeight,
      mipLevel: mip,
      copyWidth,
      copyHeight,
      blocksX,
      blocksY,
      blockWidth,
      blockHeight,
      bytesPerBlock,
      unalignedBytesPerRow,
      bytesPerRow,
      rowsPerImage,
      tightBytes,
      totalBytes,
      byteSize: totalBytes,
    };
  }

  /**
   * Pure, guaranteed zero-allocation mip byte-size helper returning a primitive
   * number (0 on degenerate/invalid dimensions or unknown format).
   *
   * Handles BC 4x4 block rounding and 256-byte `bytesPerRow` alignment by default
   * (pass `alignedOrBytesPerRow = false` or `1` for tightly-packed byte size).
   *
   * @param {number | string | object} dxgiOrWebGpuFormat
   * @param {number} width
   * @param {number} height
   * @param {number} [mipLevel=0]
   * @param {number} [depthOrLayers=1]
   * @param {boolean | number} [alignedOrBytesPerRow=true]
   * @returns {number}
   */
  function computeMipByteSize(
    dxgiOrWebGpuFormat,
    width,
    height,
    mipLevel = 0,
    depthOrLayers = 1,
    alignedOrBytesPerRow = true
  ) {
    const info = getWebGpuFormatInfo(dxgiOrWebGpuFormat);
    if (
      !info ||
      typeof width !== 'number' ||
      typeof height !== 'number' ||
      !Number.isFinite(width) ||
      !Number.isFinite(height) ||
      width < 1 ||
      height < 1
    ) {
      return 0;
    }
    if (
      typeof depthOrLayers !== 'number' ||
      !Number.isFinite(depthOrLayers) ||
      depthOrLayers < 1
    ) {
      return 0;
    }

    const baseWidth = Math.floor(width);
    const baseHeight = Math.floor(height);
    const layers = Math.floor(depthOrLayers);
    if (baseWidth < 1 || baseHeight < 1 || layers < 1) {
      return 0;
    }

    const mip = sanitizeMipLevel(mipLevel);
    const mipWidth = shrinkDimForMip(baseWidth, mip);
    const mipHeight = shrinkDimForMip(baseHeight, mip);
    const blockWidth = info.blockWidth;
    const blockHeight = info.blockHeight;

    let blocksX;
    let blocksY;
    if (blockWidth === 1 && blockHeight === 1) {
      blocksX = mipWidth;
      blocksY = mipHeight;
    } else if (blockWidth === 4 && blockHeight === 4 && mipWidth <= 0x7ffffffc && mipHeight <= 0x7ffffffc) {
      blocksX = (mipWidth + 3) >>> 2;
      blocksY = (mipHeight + 3) >>> 2;
    } else {
      blocksX = Math.ceil(mipWidth / blockWidth);
      blocksY = Math.ceil(mipHeight / blockHeight);
    }

    const unalignedBytesPerRow = blocksX * info.bytesPerBlock;
    let align = WEBGPU_BYTES_PER_ROW_ALIGNMENT;
    if (alignedOrBytesPerRow === false) {
      align = 1;
    } else if (
      typeof alignedOrBytesPerRow === 'number' &&
      Number.isFinite(alignedOrBytesPerRow) &&
      alignedOrBytesPerRow >= 1
    ) {
      align = Math.floor(alignedOrBytesPerRow);
    }

    const bytesPerRow = alignBytesPerRowValue(unalignedBytesPerRow, align);
    return bytesPerRow * blocksY * layers;
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
    DXGI_FORMAT_MAX_ID,
    DXGI_FORMAT_TABLE,
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
    lookupDxgiFormat,
    mapDxgiFormatToWebGpu,
    getSrgbSiblingFormat,
    classifyTextureFormatSupport,
    resolveStorageCompatibleFormat,
    calcMipSize,
    calcMipLayout,
    alignCopyExtentForFormat,
    calcMipUploadLayout,
    computeMipLayout,
    computeMipByteSize,
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
