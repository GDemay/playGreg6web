'use strict';

// WebGPU transient render-target texture pool and per-frame attachment aliasing manager.
// Zero external dependencies; UMD export for Web Workers (importScripts / globalThis.PlayGta6RenderTargetPool),
// browsers, and Node.js (module.exports).
// Designed for RAGE deferred color/depth/shadow/post-FX passes in web/workers/wgpu_worker.js.

(function (root) {
  const DEFAULT_MAX_POOL_BYTES = 512 * 1024 * 1024; // 512 MiB (accommodates simultaneous 4K scene plane 1 + 1080p UI plane 0)
  const DEFAULT_IDLE_FRAME_TTL = 8;
  const DEFAULT_MAX_TEXTURE_DIMENSION_2D = 8192;
  const DEFAULT_MAX_TEXTURE_ARRAY_LAYERS = 256;

  // WebGPU GPUTextureUsage bitmask values with headless/Node fallbacks.
  const TEXTURE_USAGE = Object.freeze({
    COPY_SRC:
      typeof GPUTextureUsage !== 'undefined' && typeof GPUTextureUsage.COPY_SRC === 'number'
        ? GPUTextureUsage.COPY_SRC
        : 0x01,
    COPY_DST:
      typeof GPUTextureUsage !== 'undefined' && typeof GPUTextureUsage.COPY_DST === 'number'
        ? GPUTextureUsage.COPY_DST
        : 0x02,
    TEXTURE_BINDING:
      typeof GPUTextureUsage !== 'undefined' && typeof GPUTextureUsage.TEXTURE_BINDING === 'number'
        ? GPUTextureUsage.TEXTURE_BINDING
        : 0x04,
    STORAGE_BINDING:
      typeof GPUTextureUsage !== 'undefined' && typeof GPUTextureUsage.STORAGE_BINDING === 'number'
        ? GPUTextureUsage.STORAGE_BINDING
        : 0x08,
    RENDER_ATTACHMENT:
      typeof GPUTextureUsage !== 'undefined' && typeof GPUTextureUsage.RENDER_ATTACHMENT === 'number'
        ? GPUTextureUsage.RENDER_ATTACHMENT
        : 0x10,
  });

  const DEFAULT_TEXTURE_USAGE = TEXTURE_USAGE.RENDER_ATTACHMENT | TEXTURE_USAGE.TEXTURE_BINDING;

  const VALID_DIMENSIONS = Object.freeze(['1d', '2d', '3d']);

  /**
   * Common DXGI / Xbox Series render-target and depth-stencil format aliases mapped
   * to canonical WebGPU `GPUTextureFormat` strings.
   */
  const DXGI_FORMAT_ALIASES = Object.freeze({
    // 8-bit single-channel
    r8_unorm: 'r8unorm',
    r8_snorm: 'r8snorm',
    r8_uint: 'r8uint',
    r8_sint: 'r8sint',

    // 16-bit formats
    r16_float: 'r16float',
    r16_unorm: 'r16unorm',
    r16_snorm: 'r16snorm',
    r16_uint: 'r16uint',
    r16_sint: 'r16sint',
    r8g8_unorm: 'rg8unorm',
    r8g8_snorm: 'rg8snorm',
    r8g8_uint: 'rg8uint',
    r8g8_sint: 'rg8sint',
    d16_unorm: 'depth16unorm',

    // 32-bit color & depth-stencil formats
    r32_float: 'r32float',
    r32_uint: 'r32uint',
    r32_sint: 'r32sint',
    r16g16_float: 'rg16float',
    r16g16_unorm: 'rg16unorm',
    r16g16_snorm: 'rg16snorm',
    r16g16_uint: 'rg16uint',
    r16g16_sint: 'rg16sint',
    r8g8b8a8_unorm: 'rgba8unorm',
    r8g8b8a8_unorm_srgb: 'rgba8unorm-srgb',
    r8g8b8a8_snorm: 'rgba8snorm',
    r8g8b8a8_uint: 'rgba8uint',
    r8g8b8a8_sint: 'rgba8sint',
    b8g8r8a8_unorm: 'bgra8unorm',
    b8g8r8a8_unorm_srgb: 'bgra8unorm-srgb',
    r10g10b10a2_unorm: 'rgb10a2unorm',
    r10g10b10a2_uint: 'rgb10a2uint',
    r11g11b10_float: 'rg11b10ufloat',
    r9g9b9e5_sharedexp: 'rgb9e5ufloat',
    d32_float: 'depth32float',
    d24_unorm_s8_uint: 'depth24plus-stencil8',
    r24g8_typeless: 'depth24plus-stencil8',
    x24_typeless_g8_uint: 'depth24plus-stencil8',

    // 64-bit color & depth-stencil formats
    r32g32_float: 'rg32float',
    r32g32_uint: 'rg32uint',
    r32g32_sint: 'rg32sint',
    r16g16b16a16_float: 'rgba16float',
    r16g16b16a16_unorm: 'rgba16unorm',
    r16g16b16a16_snorm: 'rgba16snorm',
    r16g16b16a16_uint: 'rgba16uint',
    r16g16b16a16_sint: 'rgba16sint',
    d32_float_s8x24_uint: 'depth32float-stencil8',
    r32g8x24_typeless: 'depth32float-stencil8',
    x32_typeless_g8x24_uint: 'depth32float-stencil8',

    // 128-bit color formats
    r32g32b32a32_float: 'rgba32float',
    r32g32b32a32_uint: 'rgba32uint',
    r32g32b32a32_sint: 'rgba32sint',
  });

  /**
   * Bytes-per-texel lookup table for common WebGPU color, depth, and stencil render-target formats.
   * Note: `depth24plus` occupies 4 bytes/texel in hardware backing storage (D24X8 / D32F);
   * `depth32float-stencil8` occupies 8 bytes/texel (32-bit depth + 8-bit stencil + 24-bit pad).
   */
  const FORMAT_BYTES_PER_PIXEL = Object.freeze({
    // 8-bit / 1-byte formats
    r8unorm: 1,
    r8snorm: 1,
    r8uint: 1,
    r8sint: 1,
    stencil8: 1,

    // 16-bit / 2-byte formats
    r16float: 2,
    r16uint: 2,
    r16sint: 2,
    r16unorm: 2,
    r16snorm: 2,
    rg8unorm: 2,
    rg8snorm: 2,
    rg8uint: 2,
    rg8sint: 2,
    depth16unorm: 2,

    // 32-bit / 4-byte formats
    r32float: 4,
    r32uint: 4,
    r32sint: 4,
    rg16float: 4,
    rg16uint: 4,
    rg16sint: 4,
    rg16unorm: 4,
    rg16snorm: 4,
    rgba8unorm: 4,
    'rgba8unorm-srgb': 4,
    rgba8snorm: 4,
    rgba8uint: 4,
    rgba8sint: 4,
    bgra8unorm: 4,
    'bgra8unorm-srgb': 4,
    rgb10a2unorm: 4,
    rgb10a2uint: 4,
    rg11b10ufloat: 4,
    rgb9e5ufloat: 4,
    depth24plus: 4,
    'depth24plus-stencil8': 4,
    depth32float: 4,

    // 64-bit / 8-byte formats
    rg32float: 8,
    rg32uint: 8,
    rg32sint: 8,
    rgba16float: 8,
    rgba16uint: 8,
    rgba16sint: 8,
    rgba16unorm: 8,
    rgba16snorm: 8,
    'depth32float-stencil8': 8,

    // 128-bit / 16-byte formats
    rgba32float: 16,
    rgba32uint: 16,
    rgba32sint: 16,
  });

  /**
   * WebGPU compatible alternate `viewFormats` for render-target formats (sRGB / linear pairs).
   * A texture can always be viewed in its own base `format` without listing it in `viewFormats`.
   */
  const COMPATIBLE_VIEW_FORMATS = Object.freeze({
    rgba8unorm: Object.freeze(['rgba8unorm-srgb']),
    'rgba8unorm-srgb': Object.freeze(['rgba8unorm']),
    bgra8unorm: Object.freeze(['bgra8unorm-srgb']),
    'bgra8unorm-srgb': Object.freeze(['bgra8unorm']),
  });

  const EMPTY_VIEW_FORMATS = Object.freeze([]);

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
   * Resolve the effective 2D texture dimension limit from a WebGPU device,
   * defaulting to `8192` when `device.limits.maxTextureDimension2D` is absent.
   *
   * @param {object | null | undefined} device
   * @returns {number}
   */
  function resolveMaxTextureDimension2D(device) {
    const raw =
      device &&
      device.limits &&
      typeof device.limits.maxTextureDimension2D === 'number' &&
      Number.isFinite(device.limits.maxTextureDimension2D) &&
      device.limits.maxTextureDimension2D >= 1
        ? Math.floor(device.limits.maxTextureDimension2D)
        : DEFAULT_MAX_TEXTURE_DIMENSION_2D;
    return raw;
  }

  function resolveDeviceTextureLimits(device, overrideMaxDim2D) {
    const maxDim2D =
      typeof overrideMaxDim2D === 'number' &&
      Number.isFinite(overrideMaxDim2D) &&
      overrideMaxDim2D >= 1
        ? Math.floor(overrideMaxDim2D)
        : resolveMaxTextureDimension2D(device);

    const maxDim1D =
      device &&
      device.limits &&
      typeof device.limits.maxTextureDimension1D === 'number' &&
      Number.isFinite(device.limits.maxTextureDimension1D) &&
      device.limits.maxTextureDimension1D >= 1
        ? Math.floor(device.limits.maxTextureDimension1D)
        : maxDim2D;

    const maxDim3D =
      device &&
      device.limits &&
      typeof device.limits.maxTextureDimension3D === 'number' &&
      Number.isFinite(device.limits.maxTextureDimension3D) &&
      device.limits.maxTextureDimension3D >= 1
        ? Math.floor(device.limits.maxTextureDimension3D)
        : maxDim2D;

    const maxArrayLayers =
      device &&
      device.limits &&
      typeof device.limits.maxTextureArrayLayers === 'number' &&
      Number.isFinite(device.limits.maxTextureArrayLayers) &&
      device.limits.maxTextureArrayLayers >= 1
        ? Math.floor(device.limits.maxTextureArrayLayers)
        : DEFAULT_MAX_TEXTURE_ARRAY_LAYERS;

    return { maxDim1D, maxDim2D, maxDim3D, maxArrayLayers };
  }

  function clampDimensionValue(raw, maxDim) {
    if (typeof raw !== 'number' || !Number.isFinite(raw)) {
      return { value: 1, clamped: true };
    }
    const floored = Math.floor(raw);
    if (floored < 1) {
      return { value: 1, clamped: true };
    }
    if (floored > maxDim) {
      return { value: maxDim, clamped: true };
    }
    return { value: floored, clamped: floored !== raw };
  }

  function _normalizeFormatWithMeta(format) {
    if (typeof format !== 'string' || format.trim().length === 0) {
      throw new TypeError('desc.format must be a non-empty WebGPU GPUTextureFormat string');
    }
    const lower = format.trim().toLowerCase();
    if (Object.prototype.hasOwnProperty.call(FORMAT_BYTES_PER_PIXEL, lower)) {
      return { format: lower, normalized: false };
    }
    const aliasKey = lower.startsWith('dxgi_format_')
      ? lower.slice('dxgi_format_'.length)
      : lower;
    if (Object.prototype.hasOwnProperty.call(DXGI_FORMAT_ALIASES, aliasKey)) {
      return { format: DXGI_FORMAT_ALIASES[aliasKey], normalized: true };
    }
    return { format: lower, normalized: false };
  }

  /**
   * Normalize a DXGI / Xbox Series render-target or depth-stencil format string
   * to its canonical WebGPU `GPUTextureFormat`, while preserving valid WebGPU
   * format strings untouched.
   *
   * @param {string} format
   * @returns {string}
   */
  function normalizeRenderTargetFormat(format) {
    return _normalizeFormatWithMeta(format).format;
  }

  /**
   * Return true if `viewFormat` is a WebGPU-compatible texture view format for `baseFormat`
   * (either the same canonical format or its compatible sRGB/linear counterpart).
   *
   * @param {string} baseFormat
   * @param {string} viewFormat
   * @returns {boolean}
   */
  function isCompatibleViewFormat(baseFormat, viewFormat) {
    const base = _normalizeFormatWithMeta(baseFormat).format;
    const vf = _normalizeFormatWithMeta(viewFormat).format;
    if (vf === base) {
      return true;
    }
    const compatible = COMPATIBLE_VIEW_FORMATS[base];
    return Array.isArray(compatible) && compatible.includes(vf);
  }

  function _normalizeViewFormatsWithMeta(rawViewFormats, baseFormat, options = {}) {
    if (rawViewFormats === undefined || rawViewFormats === null) {
      return {
        viewFormats: EMPTY_VIEW_FORMATS,
        formatNormalized: false,
        viewFormatsNormalized: false,
      };
    }
    if (
      typeof rawViewFormats === 'string' ||
      typeof rawViewFormats[Symbol.iterator] !== 'function'
    ) {
      throw new TypeError('desc.viewFormats must be an iterable of WebGPU GPUTextureFormat strings');
    }

    const filterIncompatible = Boolean(options.filterIncompatible);
    const compatibleList = COMPATIBLE_VIEW_FORMATS[baseFormat] || EMPTY_VIEW_FORMATS;
    const unique = new Set();
    let formatNormalized = false;
    let viewFormatsNormalized = false;
    let seenCount = 0;

    for (const rawFmt of rawViewFormats) {
      seenCount += 1;
      const fmtRes = _normalizeFormatWithMeta(rawFmt);
      if (fmtRes.normalized) {
        formatNormalized = true;
      }
      const vf = fmtRes.format;
      if (vf === baseFormat) {
        // WebGPU always allows creating a view with the texture's own base format.
        viewFormatsNormalized = true;
        continue;
      }
      if (!compatibleList.includes(vf)) {
        if (filterIncompatible) {
          viewFormatsNormalized = true;
          continue;
        }
        throw new RangeError(
          `Incompatible viewFormat "${vf}" for base texture format "${baseFormat}"`
        );
      }
      if (unique.has(vf)) {
        viewFormatsNormalized = true;
        continue;
      }
      unique.add(vf);
    }

    if (unique.size === 0) {
      return {
        viewFormats: EMPTY_VIEW_FORMATS,
        formatNormalized,
        viewFormatsNormalized: viewFormatsNormalized || seenCount > 0,
      };
    }

    const sorted = Array.from(unique).sort();
    return {
      viewFormats: Object.freeze(sorted),
      formatNormalized,
      viewFormatsNormalized,
    };
  }

  /**
   * Normalize, deduplicate, and canonically sort `viewFormats` for a base texture format.
   *
   * @param {Iterable<string> | null | undefined} viewFormats
   * @param {string} baseFormat
   * @param {{ filterIncompatible?: boolean }} [options]
   * @returns {readonly string[]}
   */
  function normalizeViewFormats(viewFormats, baseFormat, options = {}) {
    const canonicalBase = _normalizeFormatWithMeta(baseFormat).format;
    return _normalizeViewFormatsWithMeta(viewFormats, canonicalBase, options).viewFormats;
  }

  /**
   * Normalize a requested `sampleCount` to WebGPU's valid set `{1, 4}`, falling
   * back to `1` when texture constraints disallow multisampling.
   *
   * @param {number | undefined | null} requestedSampleCount
   * @param {{
   *   mipLevelCount?: number,
   *   dimension?: string,
   *   depthOrArrayLayers?: number,
   *   usage?: number,
   * }} [context]
   * @returns {{ sampleCount: number, normalized: boolean }}
   */
  function normalizeSampleCount(requestedSampleCount, context = {}) {
    if (requestedSampleCount === undefined || requestedSampleCount === null) {
      return { sampleCount: 1, normalized: false };
    }
    const numeric = Number(requestedSampleCount);
    if (!Number.isFinite(numeric)) {
      return { sampleCount: 1, normalized: true };
    }

    const mipLevelCount = context.mipLevelCount !== undefined ? context.mipLevelCount : 1;
    const dimension = context.dimension !== undefined ? context.dimension : '2d';
    const depthOrArrayLayers =
      context.depthOrArrayLayers !== undefined ? context.depthOrArrayLayers : 1;
    const usage = context.usage !== undefined ? context.usage : 0;

    const multisampleAllowed =
      mipLevelCount <= 1 &&
      dimension === '2d' &&
      depthOrArrayLayers === 1 &&
      (usage & TEXTURE_USAGE.STORAGE_BINDING) === 0;

    const resolved = numeric >= 4 && multisampleAllowed ? 4 : 1;
    return {
      sampleCount: resolved,
      normalized: resolved !== requestedSampleCount,
    };
  }

  function extractRawDimensions(desc) {
    let width = desc.width;
    let height = desc.height;
    let depthOrArrayLayers = desc.depthOrArrayLayers;
    if (depthOrArrayLayers === undefined && desc.depthOrLayers !== undefined) {
      depthOrArrayLayers = desc.depthOrLayers;
    }

    if (desc.size !== undefined && desc.size !== null) {
      if (Array.isArray(desc.size)) {
        if (width === undefined) {
          width = desc.size[0];
        }
        if (height === undefined) {
          height = desc.size[1];
        }
        if (depthOrArrayLayers === undefined && desc.size[2] !== undefined) {
          depthOrArrayLayers = desc.size[2];
        }
      } else if (typeof desc.size === 'object') {
        if (width === undefined) {
          width = desc.size.width;
        }
        if (height === undefined) {
          height = desc.size.height;
        }
        if (depthOrArrayLayers === undefined && desc.size.depthOrArrayLayers !== undefined) {
          depthOrArrayLayers = desc.size.depthOrArrayLayers;
        }
      } else {
        throw new TypeError('desc.size must be an object { width, height, depthOrArrayLayers } or array');
      }
    }

    return { width, height, depthOrArrayLayers };
  }

  /**
   * Sanitize a render-target descriptor for WebGPU device safety:
   * - Clamps dimensions to `[1, maxTextureDimension2D]` (default `8192`).
   * - Normalizes DXGI / Xbox render-target & depth-stencil formats to canonical WebGPU strings.
   * - Clamps `mipLevelCount` to the valid chain depth for the clamped dimensions.
   * - Normalizes `sampleCount` to `{1, 4}` respecting WebGPU multisampling constraints.
   *
   * @param {object} desc
   * @param {{ device?: object | null, maxTextureDimension2D?: number }} [options]
   * @returns {{
   *   desc: Readonly<{
   *     width: number,
   *     height: number,
   *     format: string,
   *     usage: number,
   *     sampleCount: number,
   *     mipLevelCount: number,
   *     depthOrArrayLayers: number,
   *     dimension: '1d' | '2d' | '3d',
   *   }>,
   *   dimensionClamped: boolean,
   *   formatNormalized: boolean,
   *   sampleCountNormalized: boolean,
   * }}
   */
  function sanitizeRenderTargetDesc(desc, options = {}) {
    if (!desc || typeof desc !== 'object') {
      throw new TypeError('Render target descriptor must be an object');
    }

    const rawDims = extractRawDimensions(desc);

    const dimension = desc.dimension === undefined ? '2d' : String(desc.dimension).trim().toLowerCase();
    if (!VALID_DIMENSIONS.includes(dimension)) {
      throw new RangeError(
        `Unsupported texture dimension "${desc.dimension}"; expected one of: ${VALID_DIMENSIONS.join(', ')}`
      );
    }

    const limits = resolveDeviceTextureLimits(options.device, options.maxTextureDimension2D);
    const maxWidth =
      dimension === '1d' ? limits.maxDim1D : dimension === '3d' ? limits.maxDim3D : limits.maxDim2D;
    const maxHeight =
      dimension === '1d' ? 1 : dimension === '3d' ? limits.maxDim3D : limits.maxDim2D;
    const maxDepth =
      dimension === '1d' ? 1 : dimension === '3d' ? limits.maxDim3D : limits.maxArrayLayers;

    const widthResult = clampDimensionValue(rawDims.width, maxWidth);

    let heightResult;
    if (dimension === '1d' && rawDims.height === undefined) {
      heightResult = { value: 1, clamped: false };
    } else {
      heightResult = clampDimensionValue(rawDims.height, maxHeight);
    }

    let depthResult;
    if (rawDims.depthOrArrayLayers === undefined) {
      depthResult = { value: 1, clamped: false };
    } else {
      depthResult = clampDimensionValue(rawDims.depthOrArrayLayers, maxDepth);
    }

    const dimensionClamped =
      widthResult.clamped || heightResult.clamped || depthResult.clamped;

    const formatResult = _normalizeFormatWithMeta(desc.format);
    const viewFormatsResult = _normalizeViewFormatsWithMeta(
      desc.viewFormats,
      formatResult.format,
      { filterIncompatible: true }
    );

    const usage = desc.usage === undefined ? DEFAULT_TEXTURE_USAGE : desc.usage;
    assertPositiveInteger(usage, 'usage');

    const maxExtent =
      dimension === '1d'
        ? widthResult.value
        : dimension === '3d'
          ? Math.max(widthResult.value, heightResult.value, depthResult.value)
          : Math.max(widthResult.value, heightResult.value);
    const maxMips = Math.floor(Math.log2(maxExtent)) + 1;

    let mipLevelCount;
    if (desc.mipLevelCount === undefined) {
      mipLevelCount = 1;
    } else if (
      typeof desc.mipLevelCount !== 'number' ||
      !Number.isFinite(desc.mipLevelCount) ||
      Math.floor(desc.mipLevelCount) < 1
    ) {
      mipLevelCount = 1;
    } else {
      mipLevelCount = Math.min(Math.floor(desc.mipLevelCount), maxMips);
    }

    const sampleResult = normalizeSampleCount(desc.sampleCount, {
      mipLevelCount,
      dimension,
      depthOrArrayLayers: depthResult.value,
      usage,
    });

    return {
      desc: Object.freeze({
        width: widthResult.value,
        height: heightResult.value,
        format: formatResult.format,
        usage,
        sampleCount: sampleResult.sampleCount,
        mipLevelCount,
        depthOrArrayLayers: depthResult.value,
        dimension,
        viewFormats: viewFormatsResult.viewFormats,
      }),
      dimensionClamped,
      formatNormalized: formatResult.normalized || viewFormatsResult.formatNormalized,
      sampleCountNormalized: sampleResult.normalized,
      viewFormatsNormalized: viewFormatsResult.viewFormatsNormalized,
    };
  }

  /**
   * Normalize a render-target descriptor (accepting flat `{ width, height, ... }` or
   * WebGPU `{ size: { width, height, depthOrArrayLayers } | [w, h, d], ... }` shapes)
   * into a canonical frozen descriptor object.
   *
   * When `options.sanitize` (or `options.clampDimensions` / `options.normalizeSampleCount`)
   * is truthy, delegates to `sanitizeRenderTargetDesc(desc, options)`.
   *
   * @param {object} desc
   * @param {{ sanitize?: boolean, clampDimensions?: boolean, normalizeSampleCount?: boolean, device?: object | null, maxTextureDimension2D?: number }} [options]
   * @returns {{
   *   width: number,
   *   height: number,
   *   format: string,
   *   usage: number,
   *   sampleCount: number,
   *   mipLevelCount: number,
   *   depthOrArrayLayers: number,
   *   dimension: '1d' | '2d' | '3d',
   * }}
   */
  function normalizeRenderTargetDesc(desc, options) {
    if (
      options &&
      typeof options === 'object' &&
      (options.sanitize || options.clampDimensions || options.normalizeSampleCount)
    ) {
      return sanitizeRenderTargetDesc(desc, options).desc;
    }

    if (!desc || typeof desc !== 'object') {
      throw new TypeError('Render target descriptor must be an object');
    }

    let { width, height, depthOrArrayLayers } = extractRawDimensions(desc);

    const dimension = desc.dimension === undefined ? '2d' : String(desc.dimension).trim().toLowerCase();
    if (!VALID_DIMENSIONS.includes(dimension)) {
      throw new RangeError(
        `Unsupported texture dimension "${desc.dimension}"; expected one of: ${VALID_DIMENSIONS.join(', ')}`
      );
    }

    if (dimension === '1d' && height === undefined) {
      height = 1;
    }
    if (depthOrArrayLayers === undefined) {
      depthOrArrayLayers = 1;
    }

    assertPositiveInteger(width, 'width');
    assertPositiveInteger(height, 'height');
    assertPositiveInteger(depthOrArrayLayers, 'depthOrArrayLayers');

    const format = _normalizeFormatWithMeta(desc.format).format;
    const viewFormats = _normalizeViewFormatsWithMeta(desc.viewFormats, format, {
      filterIncompatible: false,
    }).viewFormats;

    const usage = desc.usage === undefined ? DEFAULT_TEXTURE_USAGE : desc.usage;
    assertPositiveInteger(usage, 'usage');

    const sampleCount = desc.sampleCount === undefined ? 1 : desc.sampleCount;
    if (sampleCount !== 1 && sampleCount !== 4) {
      throw new RangeError(`sampleCount must be 1 or 4, got ${sampleCount}`);
    }

    const mipLevelCount = desc.mipLevelCount === undefined ? 1 : desc.mipLevelCount;
    assertPositiveInteger(mipLevelCount, 'mipLevelCount');

    return Object.freeze({
      width,
      height,
      format,
      usage,
      sampleCount,
      mipLevelCount,
      depthOrArrayLayers,
      dimension,
      viewFormats,
    });
  }

  /**
   * Build a canonical deterministic pool key from a render-target descriptor.
   *
   * @param {object} desc
   * @returns {string}
   */
  function makeRenderTargetKey(desc) {
    const n = normalizeRenderTargetDesc(desc);
    const base =
      `${n.dimension}|${n.width}x${n.height}x${n.depthOrArrayLayers}|` +
      `${n.format}|u${n.usage}|s${n.sampleCount}|m${n.mipLevelCount}`;
    return n.viewFormats.length > 0 ? `${base}|vf${n.viewFormats.join(',')}` : base;
  }

  /**
   * Estimate VRAM footprint in bytes for a WebGPU render target descriptor,
   * accounting for format texel size, full mip chain, array layers / 3D depth,
   * and MSAA `sampleCount`.
   *
   * @param {object} desc
   * @returns {number}
   */
  function estimateRenderTargetBytes(desc) {
    const n = normalizeRenderTargetDesc(desc);
    if (!Object.prototype.hasOwnProperty.call(FORMAT_BYTES_PER_PIXEL, n.format)) {
      throw new RangeError(`Unsupported render target format for byte estimation: "${n.format}"`);
    }

    const bytesPerPixel = FORMAT_BYTES_PER_PIXEL[n.format];
    const is1D = n.dimension === '1d';
    const is3D = n.dimension === '3d';

    let totalBytes = 0;
    for (let mip = 0; mip < n.mipLevelCount; mip += 1) {
      const scale = mip >= 31 ? Infinity : Math.pow(2, mip);
      const mipWidth = scale === Infinity ? 1 : Math.max(1, Math.floor(n.width / scale));
      const mipHeight = is1D
        ? 1
        : scale === Infinity
          ? 1
          : Math.max(1, Math.floor(n.height / scale));
      const mipDepth = is3D
        ? scale === Infinity
          ? 1
          : Math.max(1, Math.floor(n.depthOrArrayLayers / scale))
        : n.depthOrArrayLayers;

      totalBytes += mipWidth * mipHeight * mipDepth * bytesPerPixel * n.sampleCount;
    }

    return totalBytes;
  }

  /**
   * WebGPU transient render-target texture pool & per-frame attachment aliasing manager.
   *
   * - Pools `GPUTexture` + cached default `GPUTextureView` objects keyed by normalized
   *   `(width, height, format, usage, sampleCount, mipLevelCount, depthOrArrayLayers, dimension)`.
   * - Supports immediate same-frame attachment aliasing once `release()` is called, plus
   *   cross-frame reuse without `device.createTexture()` churn.
   * - Enforces `maxPoolBytes` via global LRU eviction of idle targets and expires unused
   *   targets after `idleFrameTtl` frames in `endFrame()`.
   */
  class RenderTargetPool {
    constructor(options = {}) {
      if (!options || typeof options !== 'object') {
        throw new TypeError('RenderTargetPool constructor expects an options object');
      }

      const {
        device,
        maxPoolBytes = DEFAULT_MAX_POOL_BYTES,
        idleFrameTtl = DEFAULT_IDLE_FRAME_TTL,
        strictBudget = true,
        autoReleaseLeaked = true,
        onLeak = null,
        warn = null,
      } = options;

      if (!device || typeof device.createTexture !== 'function') {
        throw new TypeError('RenderTargetPool requires a WebGPU device with createTexture()');
      }

      if (maxPoolBytes !== Infinity && (!Number.isInteger(maxPoolBytes) || maxPoolBytes < 1)) {
        throw new RangeError(
          `maxPoolBytes must be a positive integer or Infinity, got ${maxPoolBytes}`
        );
      }
      assertNonNegativeInteger(idleFrameTtl, 'idleFrameTtl');

      if (onLeak !== null && typeof onLeak !== 'function') {
        throw new TypeError('onLeak must be a function when provided');
      }
      if (warn !== null && typeof warn !== 'function') {
        throw new TypeError('warn must be a function when provided');
      }

      this._device = device;
      this._maxPoolBytes = maxPoolBytes;
      this._idleFrameTtl = idleFrameTtl;
      this._strictBudget = Boolean(strictBudget);
      this._autoReleaseLeaked = Boolean(autoReleaseLeaked);
      this._onLeak = onLeak;
      this._warn = warn;

      this._destroyed = false;
      this._currentFrame = 0;
      this._nextId = 1;
      this._nextLeaseToken = 1;

      /** @type {Map<number, object>} All live (active + idle) records by id */
      this._allEntries = new Map();
      /** @type {Map<number, object>} Active in-use records by id */
      this._activeEntries = new Map();
      /** @type {Map<string, object[]>} Idle records grouped by normalized key (MRU stack) */
      this._idleBuckets = new Map();
      /** @type {Map<number, object>} Idle records in insertion order (oldest = LRU) */
      this._idleLru = new Map();
      /** @type {WeakMap<object, { id: number, leaseToken: number }>} Per-acquire handle tokens */
      this._handleTokens = new WeakMap();
      /** @type {WeakSet<object>} Tracks GPUTexture instances already destroyed to guarantee no double-destroy */
      this._destroyedTextures = new WeakSet();

      // Resolved device dimension cap (cached for post-destroy telemetry safety)
      this._maxTextureDimension2D = resolveMaxTextureDimension2D(device);

      // Telemetry counters
      this._hits = 0;
      this._misses = 0;
      this._activeBytes = 0;
      this._idleBytes = 0;
      this._peakBytes = 0;
      this._evictions = 0;
      this._budgetEvictions = 0;
      this._idleEvictions = 0;
      this._leakedReleases = 0;
      this._dimensionClampCount = 0;
      this._formatNormalizeCount = 0;
      this._sampleCountNormalizeCount = 0;
    }

    get isDestroyed() {
      return this._destroyed;
    }

    get maxPoolBytes() {
      return this._maxPoolBytes;
    }

    get idleFrameTtl() {
      return this._idleFrameTtl;
    }

    get currentFrame() {
      return this._currentFrame;
    }

    get maxTextureDimension2D() {
      return this._effectiveMaxTextureDimension2D();
    }

    _effectiveMaxTextureDimension2D() {
      if (this._device) {
        this._maxTextureDimension2D = resolveMaxTextureDimension2D(this._device);
      }
      return this._maxTextureDimension2D;
    }

    _assertNotDestroyed() {
      if (this._destroyed) {
        throw new Error('RenderTargetPool has been destroyed');
      }
    }

    _removeFromIdleBucket(record) {
      const bucket = this._idleBuckets.get(record.key);
      if (!bucket) {
        return;
      }
      const idx = bucket.indexOf(record);
      if (idx !== -1) {
        bucket.splice(idx, 1);
      }
      if (bucket.length === 0) {
        this._idleBuckets.delete(record.key);
      }
    }

    _destroyTextureSafely(texture) {
      if (!texture || typeof texture.destroy !== 'function') {
        return false;
      }
      if (texture.destroyed === true || this._destroyedTextures.has(texture)) {
        return false;
      }
      this._destroyedTextures.add(texture);
      try {
        texture.destroy();
      } catch (_) {
        // Ignore device-lost / already-destroyed WebGPU errors during teardown.
      }
      return true;
    }

    _isRecordTextureAlive(record) {
      if (!record || record.destroyed || !record.texture) {
        return false;
      }
      if (record.texture.destroyed === true || this._destroyedTextures.has(record.texture)) {
        return false;
      }
      if (record.hasView && !record.view) {
        return false;
      }
      return true;
    }

    _isRecordReusable(record) {
      return this._isRecordTextureAlive(record) && !record.retireOnRelease;
    }

    _destroyRecord(record) {
      if (!record || record.destroyed) {
        return false;
      }
      record.destroyed = true;
      record.active = false;
      record.retireOnRelease = false;
      record.leaseToken = 0;
      this._allEntries.delete(record.id);

      const tex = record.texture;
      record.texture = null;
      record.view = null;
      this._destroyTextureSafely(tex);
      return true;
    }

    _evictIdleRecord(record, reason = null) {
      this._idleLru.delete(record.id);
      this._removeFromIdleBucket(record);
      this._idleBytes = Math.max(0, this._idleBytes - record.byteSize);
      if (this._destroyRecord(record)) {
        this._evictions += 1;
        if (reason === 'budget') {
          this._budgetEvictions += 1;
        } else if (reason === 'idle') {
          this._idleEvictions += 1;
        }
        return true;
      }
      return false;
    }

    _evictIdleToBudget(targetBytes = this._maxPoolBytes, reserveBytes = 0) {
      let evictedCount = 0;
      let evictedBytes = 0;
      while (
        this._activeBytes + this._idleBytes + reserveBytes > targetBytes &&
        this._idleLru.size > 0
      ) {
        const oldest = this._idleLru.values().next().value;
        const bytes = oldest.byteSize;
        if (this._evictIdleRecord(oldest, 'budget')) {
          evictedCount += 1;
          evictedBytes += bytes;
        }
      }
      return { evictedCount, evictedBytes };
    }

    /**
     * Reconfigure the pool's GPU memory budget (`maxPoolBytes`) at runtime and
     * LRU-evict excess idle targets if current pooled bytes exceed the new budget.
     * Never destroys currently checked-out/active targets.
     *
     * @param {number} maxPoolBytes Positive integer or Infinity.
     * @returns {{ evictedCount: number, evictedBytes: number }}
     */
    setMaxPoolBytes(maxPoolBytes) {
      this._assertNotDestroyed();
      if (maxPoolBytes !== Infinity && (!Number.isInteger(maxPoolBytes) || maxPoolBytes < 1)) {
        throw new RangeError(
          `maxPoolBytes must be a positive integer or Infinity, got ${maxPoolBytes}`
        );
      }
      this._maxPoolBytes = maxPoolBytes;
      return this._evictIdleToBudget(this._maxPoolBytes, 0);
    }

    _retireActiveRecord(record) {
      record.active = false;
      record.leaseToken = 0;
      this._activeEntries.delete(record.id);
      this._activeBytes = Math.max(0, this._activeBytes - record.byteSize);
      if (this._destroyRecord(record)) {
        this._evictions += 1;
        return true;
      }
      return false;
    }

    _moveActiveToIdle(record, frameIndex) {
      record.active = false;
      record.leaseToken = 0;
      record.lastUsedFrame = frameIndex;
      this._activeEntries.delete(record.id);
      this._activeBytes = Math.max(0, this._activeBytes - record.byteSize);
      this._idleBytes += record.byteSize;

      let bucket = this._idleBuckets.get(record.key);
      if (!bucket) {
        bucket = [];
        this._idleBuckets.set(record.key, bucket);
      }
      bucket.push(record);
      this._idleLru.set(record.id, record);
    }

    _makeHandle(record, reused) {
      const leaseToken = this._nextLeaseToken++;
      record.leaseToken = leaseToken;
      const textureRef = record.texture;
      const pool = this;
      const handle = {
        id: record.id,
        key: record.key,
        desc: record.desc,
        texture: textureRef,
        get view() {
          return pool._isRecordTextureAlive(record) ? record.view : null;
        },
        get destroyed() {
          return !pool._isRecordTextureAlive(record);
        },
        byteSize: record.byteSize,
        reused,
      };
      this._handleTokens.set(handle, { id: record.id, leaseToken });
      return handle;
    }

    /**
     * Acquire a transient render-target texture matching `desc`. Reuses an idle pooled
     * texture (including same-frame targets released earlier in the same frame) when
     * available; otherwise allocates a new `GPUTexture` and cached default `GPUTextureView`,
     * LRU-evicting idle textures as needed to stay within `maxPoolBytes`.
     *
     * Sanitizes requested dimensions into `[1, device.limits.maxTextureDimension2D]`
     * (default `8192`), normalizes common DXGI/Xbox color & depth-stencil formats, and
     * normalizes `sampleCount` to `{1, 4}` to prevent WebGPU validation errors.
     *
     * @param {object} desc
     * @param {string} [label]
     * @returns {{
     *   id: number,
     *   key: string,
     *   desc: Readonly<object>,
     *   texture: GPUTexture | null,
     *   view: GPUTextureView | null,
     *   destroyed: boolean,
     *   byteSize: number,
     *   reused: boolean,
     * }}
     */
    acquire(desc, label) {
      this._assertNotDestroyed();

      const maxDim2D = this._effectiveMaxTextureDimension2D();
      const sanitized = sanitizeRenderTargetDesc(desc, {
        device: this._device,
        maxTextureDimension2D: maxDim2D,
      });
      const normalizedDesc = sanitized.desc;
      const key = makeRenderTargetKey(normalizedDesc);
      const byteSize = estimateRenderTargetBytes(normalizedDesc);

      // 1. Check idle bucket for an exact descriptor match (MRU pop lets unused duplicates age out).
      //    Skip and clean up any invalidated or externally-destroyed records so stale views are never returned.
      const bucket = this._idleBuckets.get(key);
      if (bucket && bucket.length > 0) {
        while (bucket.length > 0) {
          const record = bucket.pop();
          if (bucket.length === 0) {
            this._idleBuckets.delete(key);
          }
          this._idleLru.delete(record.id);

          if (!this._isRecordReusable(record)) {
            this._idleBytes = Math.max(0, this._idleBytes - record.byteSize);
            if (this._destroyRecord(record)) {
              this._evictions += 1;
            }
            continue;
          }

          record.active = true;
          record.lastUsedFrame = this._currentFrame;
          if (typeof label === 'string' && label.length > 0) {
            record.label = label;
          }

          this._activeEntries.set(record.id, record);
          this._idleBytes = Math.max(0, this._idleBytes - record.byteSize);
          this._activeBytes += record.byteSize;
          this._hits += 1;

          if (sanitized.dimensionClamped) {
            this._dimensionClampCount += 1;
          }
          if (sanitized.formatNormalized) {
            this._formatNormalizeCount += 1;
          }
          if (sanitized.sampleCountNormalized) {
            this._sampleCountNormalizeCount += 1;
          }

          return this._makeHandle(record, true);
        }
      }

      // 2. Miss: check strict memory budget before evicting idle targets or allocating.
      if (this._strictBudget && this._activeBytes + byteSize > this._maxPoolBytes) {
        throw new RangeError(
          `RenderTargetPool budget exceeded: cannot allocate ${byteSize} bytes for ` +
            `${normalizedDesc.width}x${normalizedDesc.height} ${normalizedDesc.format} ` +
            `(activeBytes=${this._activeBytes}, maxPoolBytes=${this._maxPoolBytes})`
        );
      }

      // 3. LRU-evict idle targets until totalBytes + byteSize <= maxPoolBytes (or no idle left).
      //    Checked-out / active targets are never destroyed by budget eviction.
      this._evictIdleToBudget(this._maxPoolBytes, byteSize);

      // 4. Allocate new GPUTexture + cached default GPUTextureView.
      const id = this._nextId++;
      const resolvedLabel =
        typeof label === 'string' && label.length > 0
          ? label
          : `playgta6-rt-${id}-${normalizedDesc.width}x${normalizedDesc.height}-${normalizedDesc.format}`;

      const textureDesc = {
        label: resolvedLabel,
        size: {
          width: normalizedDesc.width,
          height: normalizedDesc.height,
          depthOrArrayLayers: normalizedDesc.depthOrArrayLayers,
        },
        mipLevelCount: normalizedDesc.mipLevelCount,
        sampleCount: normalizedDesc.sampleCount,
        dimension: normalizedDesc.dimension,
        format: normalizedDesc.format,
        usage: normalizedDesc.usage,
      };
      if (normalizedDesc.viewFormats.length > 0) {
        textureDesc.viewFormats = normalizedDesc.viewFormats.slice();
      }
      const texture = this._device.createTexture(textureDesc);

      let view = null;
      try {
        view = texture && typeof texture.createView === 'function' ? texture.createView() : null;
      } catch (err) {
        this._destroyTextureSafely(texture);
        throw err;
      }

      const record = {
        id,
        key,
        desc: normalizedDesc,
        label: resolvedLabel,
        texture,
        view,
        hasView: view !== null,
        byteSize,
        active: true,
        destroyed: false,
        retireOnRelease: false,
        lastUsedFrame: this._currentFrame,
        leaseToken: 0,
      };

      this._allEntries.set(id, record);
      this._activeEntries.set(id, record);
      this._activeBytes += byteSize;
      this._misses += 1;

      if (sanitized.dimensionClamped) {
        this._dimensionClampCount += 1;
      }
      if (sanitized.formatNormalized) {
        this._formatNormalizeCount += 1;
      }
      if (sanitized.sampleCountNormalized) {
        this._sampleCountNormalizeCount += 1;
      }

      const totalBytes = this._activeBytes + this._idleBytes;
      if (totalBytes > this._peakBytes) {
        this._peakBytes = totalBytes;
      }

      return this._makeHandle(record, false);
    }

    /**
     * Mark an acquired render target available for immediate aliasing within the same frame
     * (or subsequent frames). If the target was invalidated during a resize while in flight
     * (or externally destroyed), it is retired and destroyed cleanly instead of re-entering
     * the idle pool. Safe against duplicate / stale releases.
     *
     * @param {object | number} entryOrId
     * @returns {boolean} True if an active target was released; false if already idle/unknown/stale.
     */
    release(entryOrId) {
      this._assertNotDestroyed();

      if (entryOrId === null || entryOrId === undefined) {
        throw new TypeError('release() requires an acquired entry object or numeric id');
      }

      let id;
      let expectedLeaseToken = null;

      if (typeof entryOrId === 'number') {
        if (!Number.isInteger(entryOrId) || entryOrId < 1) {
          throw new TypeError(`release() id must be a positive integer, got ${entryOrId}`);
        }
        id = entryOrId;
      } else if (typeof entryOrId === 'object') {
        const tokenInfo = this._handleTokens.get(entryOrId);
        if (tokenInfo) {
          id = tokenInfo.id;
          expectedLeaseToken = tokenInfo.leaseToken;
        } else if (Number.isInteger(entryOrId.id) && entryOrId.id >= 1) {
          id = entryOrId.id;
        } else {
          throw new TypeError('release() entry object must have a valid positive integer id');
        }
      } else {
        throw new TypeError('release() requires an acquired entry object or numeric id');
      }

      const record = this._allEntries.get(id);
      if (!record || !record.active) {
        return false;
      }

      // Guard against stale handle double-release stealing a newer pass's active lease.
      if (expectedLeaseToken !== null && record.leaseToken !== expectedLeaseToken) {
        return false;
      }

      if (!this._isRecordReusable(record)) {
        this._retireActiveRecord(record);
        return true;
      }

      this._moveActiveToIdle(record, this._currentFrame);
      return true;
    }

    /**
     * End the current frame epoch:
     * 1. Handle any leaked (still-active) render targets: auto-release back into the idle pool
     *    (or retire/destroy if invalidated during resize) when `autoReleaseLeaked` is enabled,
     *    or warn/notify via `onLeak` / `warn` when disabled.
     * 2. Increment the frame epoch (`currentFrame += 1`).
     * 3. Destroy idle textures unused for more than `idleFrameTtl` frames (or externally destroyed).
     *
     * @returns {{
     *   frame: number,
     *   evictedCount: number,
     *   leakedCount: number,
     *   autoReleasedCount: number,
     *   activeCount: number,
     *   idleCount: number,
     * }}
     */
    endFrame() {
      this._assertNotDestroyed();

      const endedFrame = this._currentFrame;
      this._currentFrame += 1;

      let leakedCount = 0;
      let autoReleasedCount = 0;
      let evictedCount = 0;

      if (this._activeEntries.size > 0) {
        const leakedRecords = Array.from(this._activeEntries.values());
        leakedCount = leakedRecords.length;

        for (let i = 0; i < leakedRecords.length; i += 1) {
          const record = leakedRecords[i];
          const leakInfo = {
            id: record.id,
            key: record.key,
            label: record.label,
            byteSize: record.byteSize,
            frame: endedFrame,
            autoReleased: this._autoReleaseLeaked,
          };

          if (this._autoReleaseLeaked) {
            if (!this._isRecordReusable(record)) {
              if (this._retireActiveRecord(record)) {
                evictedCount += 1;
              }
            } else {
              this._moveActiveToIdle(record, endedFrame);
            }
            this._leakedReleases += 1;
            autoReleasedCount += 1;
          }

          if (this._onLeak) {
            this._onLeak(leakInfo);
          }

          if (!this._autoReleaseLeaked) {
            const msg =
              `[RenderTargetPool] Leaked active render target #${record.id} ` +
              `(${record.label}, ${record.byteSize} bytes) at end of frame ${endedFrame}`;
            if (this._warn) {
              this._warn(msg, leakInfo);
            } else if (!this._onLeak && typeof console !== 'undefined' && typeof console.warn === 'function') {
              console.warn(msg);
            }
          } else if (this._warn) {
            const msg =
              `[RenderTargetPool] Auto-released leaked render target #${record.id} ` +
              `(${record.label}, ${record.byteSize} bytes) at end of frame ${endedFrame}`;
            this._warn(msg, leakInfo);
          }
        }
      }

      // Evict idle textures unused for more than idleFrameTtl frames or no longer reusable.
      const idleRecords = Array.from(this._idleLru.values());
      for (let i = 0; i < idleRecords.length; i += 1) {
        const record = idleRecords[i];
        const expired = this._currentFrame - record.lastUsedFrame > this._idleFrameTtl;
        if (expired || !this._isRecordReusable(record)) {
          if (this._evictIdleRecord(record, expired ? 'idle' : null)) {
            evictedCount += 1;
          }
        }
      }

      // Trim any excess idle targets back within maxPoolBytes (e.g. after a strictBudget=false burst)
      // without ever destroying checked-out/active targets.
      if (this._activeBytes + this._idleBytes > this._maxPoolBytes && this._idleLru.size > 0) {
        const trimmed = this._evictIdleToBudget(this._maxPoolBytes, 0);
        evictedCount += trimmed.evictedCount;
      }

      return {
        frame: this._currentFrame,
        evictedCount,
        leakedCount,
        autoReleasedCount,
        activeCount: this._activeEntries.size,
        idleCount: this._idleLru.size,
      };
    }

    /**
     * Explicitly destroy and remove non-acquired (idle) pooled render targets whose
     * idle age satisfies `currentFrame - entry.lastUsedFrame >= maxIdleFrames`.
     * Never evicts currently acquired/in-use targets.
     *
     * @param {number} [maxIdleFrames=this._idleFrameTtl] Non-negative integer frame threshold.
     * @returns {{ evictedCount: number, evictedBytes: number }}
     */
    evictIdle(maxIdleFrames = this._idleFrameTtl) {
      this._assertNotDestroyed();
      assertNonNegativeInteger(maxIdleFrames, 'maxIdleFrames');

      let evictedCount = 0;
      let evictedBytes = 0;
      const idleRecords = Array.from(this._idleLru.values());
      for (let i = 0; i < idleRecords.length; i += 1) {
        const record = idleRecords[i];
        if (this._currentFrame - record.lastUsedFrame >= maxIdleFrames) {
          if (this._evictIdleRecord(record, 'idle')) {
            evictedCount += 1;
            evictedBytes += record.byteSize;
          }
        }
      }

      return { evictedCount, evictedBytes };
    }

    /**
     * Destroy all currently idle pooled render targets (e.g., on viewport/swapchain resize)
     * while leaving any currently active in-flight targets intact and flagging them for
     * immediate destruction on release so stale pre-resize targets are never reused.
     *
     * @returns {number} Number of idle textures destroyed immediately.
     */
    invalidateAll() {
      this._assertNotDestroyed();

      let destroyedCount = 0;
      const idleRecords = Array.from(this._idleLru.values());
      for (let i = 0; i < idleRecords.length; i += 1) {
        const record = idleRecords[i];
        if (this._evictIdleRecord(record)) {
          destroyedCount += 1;
        }
      }

      this._idleBuckets.clear();
      this._idleLru.clear();
      this._idleBytes = 0;

      // Mark in-flight active records so they are retired/destroyed cleanly on release
      // instead of being returned to post-resize passes.
      this._activeEntries.forEach((record) => {
        record.retireOnRelease = true;
      });

      return destroyedCount;
    }

    /**
     * Convenience resize hook: invalidates all idle targets and flags in-flight active
     * targets for retirement on release.
     *
     * @param {number} [_width]
     * @param {number} [_height]
     * @returns {number} Number of idle textures destroyed immediately.
     */
    resize(_width, _height) {
      return this.invalidateAll();
    }

    /**
     * Return pool hit-rate, occupancy, memory, sanitization, and lifecycle telemetry.
     */
    getStats() {
      const totalRequests = this._hits + this._misses;
      const totalBytes = this._activeBytes + this._idleBytes;
      const hitRate = totalRequests > 0 ? this._hits / totalRequests : 0;
      return {
        hits: this._hits,
        misses: this._misses,
        poolHits: this._hits,
        poolMisses: this._misses,
        hitRate,
        reuseRatio: hitRate,
        activeCount: this._activeEntries.size,
        idleCount: this._idleLru.size,
        activeTargets: this._activeEntries.size,
        idleTargets: this._idleLru.size,
        activeBytes: this._activeBytes,
        idleBytes: this._idleBytes,
        totalBytes,
        estimatedBytes: totalBytes,
        peakBytes: this._peakBytes,
        peakPooledBytes: this._peakBytes,
        peakEstimatedBytes: this._peakBytes,
        maxPoolBytes: this._maxPoolBytes,
        evictions: this._evictions,
        budgetEvictions: this._budgetEvictions,
        idleEvictions: this._idleEvictions,
        leakedReleases: this._leakedReleases,
        dimensionClampCount: this._dimensionClampCount,
        formatNormalizeCount: this._formatNormalizeCount,
        sampleCountNormalizeCount: this._sampleCountNormalizeCount,
        maxTextureDimension2D: this._effectiveMaxTextureDimension2D(),
        currentFrame: this._currentFrame,
      };
    }

    /**
     * Destroy all active and idle GPU textures owned by the pool and release references.
     * Safe to call multiple times (idempotent).
     */
    destroy() {
      if (this._destroyed) {
        return;
      }

      this._destroyed = true;

      const allRecords = Array.from(this._allEntries.values());
      for (let i = 0; i < allRecords.length; i += 1) {
        this._destroyRecord(allRecords[i]);
      }

      this._allEntries.clear();
      this._activeEntries.clear();
      this._idleBuckets.clear();
      this._idleLru.clear();
      this._activeBytes = 0;
      this._idleBytes = 0;
      this._device = null;
    }
  }

  const exportsObj = Object.freeze({
    DEFAULT_MAX_POOL_BYTES,
    DEFAULT_IDLE_FRAME_TTL,
    DEFAULT_MAX_TEXTURE_DIMENSION_2D,
    TEXTURE_USAGE,
    DEFAULT_TEXTURE_USAGE,
    FORMAT_BYTES_PER_PIXEL,
    DXGI_FORMAT_ALIASES,
    COMPATIBLE_VIEW_FORMATS,
    normalizeRenderTargetFormat,
    isCompatibleViewFormat,
    normalizeViewFormats,
    normalizeSampleCount,
    sanitizeRenderTargetDesc,
    normalizeRenderTargetDesc,
    makeRenderTargetKey,
    estimateRenderTargetBytes,
    RenderTargetPool,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6RenderTargetPool = exportsObj;
  } else if (root) {
    root.PlayGta6RenderTargetPool = exportsObj;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportsObj;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
