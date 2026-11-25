'use strict';

// Hardened WebGPU ResourceTable for CreateTexture2 (ABI v2 opcode 35),
// BC1..BC7 feature fallback & 4x4 dimension alignment, and WebGPU buffer
// size/write 4-byte alignment padding.
// Zero external dependencies; UMD/CommonJS/ESM compatible for classic Web Workers
// (importScripts -> globalThis.PlayGta6ResourceTable), browsers, and Node.js.

(function (globalScope) {
  const CREATE_TEXTURE2_OPCODE = 35;
  const CREATE_TEXTURE2_PAYLOAD_BYTES = 40;
  const MAX_TEXTURE_VIEW_FORMATS = 8;

  const TEXTURE_COMPRESSION_BC_FEATURE = 'texture-compression-bc';
  const BC_FALLBACK_FORMAT = 'rgba8unorm';
  const BC_FALLBACK_SRGB_FORMAT = 'rgba8unorm-srgb';

  const BUFFER_ALIGNMENT_BYTES = 4;
  const UNIFORM_BUFFER_MIN_BYTES = 16;

  // WebGPU-compatible GPUBufferUsage bitmask values (runtime/abi/playgta6_abi.h).
  const GPU_BUFFER_USAGE = Object.freeze({
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

  // WebGPU-compatible GPUTextureUsage bitmask values (runtime/abi/playgta6_abi.h).
  const GPU_TEXTURE_USAGE = Object.freeze({
    COPY_SRC: 0x0001,
    COPY_DST: 0x0002,
    TEXTURE_BINDING: 0x0004,
    STORAGE_BINDING: 0x0008,
    RENDER_ATTACHMENT: 0x0010,
  });

  // v1 GpuTextureFormat enum names (runtime/abi/playgta6_abi.h).
  const V1_TEXTURE_FORMAT_NAMES = Object.freeze([
    'rgba8unorm',
    'bgra8unorm',
    'rgba16float',
    'depth24plus',
    'depth32float',
    'bc7-rgba-unorm',
  ]);

  // v2 texture format codes ("WebGPU enum codes" in docs/gpu_commands.md & texture_packets.js).
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

  // GpuTextureDimension enum names (runtime/abi/playgta6_abi.h).
  const TEXTURE_DIMENSION_NAMES = Object.freeze(['1d', '2d', '3d']);

  // All WebGPU BC1..BC7 4x4 block-compressed texture formats.
  const BC_TEXTURE_FORMATS = Object.freeze([
    'bc1-rgba-unorm',
    'bc1-rgba-unorm-srgb',
    'bc2-rgba-unorm',
    'bc2-rgba-unorm-srgb',
    'bc3-rgba-unorm',
    'bc3-rgba-unorm-srgb',
    'bc4-r-unorm',
    'bc4-r-snorm',
    'bc5-rg-unorm',
    'bc5-rg-snorm',
    'bc6h-rgb-ufloat',
    'bc6h-rgb-float',
    'bc7-rgba-unorm',
    'bc7-rgba-unorm-srgb',
  ]);

  const BC_FORMAT_SET = new Set(BC_TEXTURE_FORMATS);

  function isBcFormat(format) {
    return typeof format === 'string' && BC_FORMAT_SET.has(format.toLowerCase());
  }

  /**
   * Check whether a WebGPU device supports `texture-compression-bc`.
   * Returns `false` only when `device.features` exists and does not contain
   * `'texture-compression-bc'`.
   *
   * @param {object | null | undefined} device
   * @returns {boolean}
   */
  function deviceSupportsBc(device) {
    if (!device || !device.features) {
      return true;
    }
    const features = device.features;
    if (typeof features.has === 'function') {
      return Boolean(features.has(TEXTURE_COMPRESSION_BC_FEATURE));
    }
    if (Array.isArray(features)) {
      return features.includes(TEXTURE_COMPRESSION_BC_FEATURE);
    }
    if (typeof features[Symbol.iterator] === 'function') {
      for (const item of features) {
        if (item === TEXTURE_COMPRESSION_BC_FEATURE) {
          return true;
        }
      }
      return false;
    }
    if (typeof features === 'object') {
      return Boolean(features[TEXTURE_COMPRESSION_BC_FEATURE]);
    }
    return false;
  }

  function resolveTextureFormat(format, useV1Table = false) {
    if (typeof format === 'string' && format.length > 0 && format !== 'undefined') {
      return format.toLowerCase();
    }
    if (typeof format === 'number' && Number.isInteger(format)) {
      if (useV1Table) {
        return V1_TEXTURE_FORMAT_NAMES[format] || BC_FALLBACK_FORMAT;
      }
      const name = TEXTURE_FORMAT_NAMES[format];
      if (format > 0 && name && name !== 'undefined') {
        return name;
      }
    }
    return BC_FALLBACK_FORMAT;
  }

  function resolveTextureDimension(dimension) {
    if (typeof dimension === 'string') {
      const lower = dimension.toLowerCase();
      if (lower === '1d' || lower === '2d' || lower === '3d') {
        return lower;
      }
    }
    if (typeof dimension === 'number' && Number.isInteger(dimension)) {
      const name = TEXTURE_DIMENSION_NAMES[dimension];
      if (name) {
        return name;
      }
    }
    return '2d';
  }

  /**
   * Compute maximum valid WebGPU mip level count for texture dimensions.
   * For 1D/2D/2D-array textures this is `floor(log2(max(width, height))) + 1`;
   * for 3D textures this is `floor(log2(max(width, height, depth))) + 1`.
   *
   * @param {number} width
   * @param {number} height
   * @param {number} [depthOrArrayLayers=1]
   * @param {string} [dimension='2d']
   * @returns {number}
   */
  function computeMaxMipLevels(width, height, depthOrArrayLayers = 1, dimension = '2d') {
    const w = Math.max(1, Math.floor(Number(width) || 1));
    const h = dimension === '1d' ? 1 : Math.max(1, Math.floor(Number(height) || 1));
    const d = Math.max(1, Math.floor(Number(depthOrArrayLayers) || 1));
    const maxDim = dimension === '3d' ? Math.max(w, h, d) : Math.max(w, h);
    return Math.floor(Math.log2(maxDim)) + 1;
  }

  /**
   * Clamp a requested mip level count into `[1, maxMipLevels]`.
   *
   * @param {number | undefined} requestedMipLevels
   * @param {number} width
   * @param {number} height
   * @param {number} [depthOrArrayLayers=1]
   * @param {string} [dimension='2d']
   * @returns {{ mipLevelCount: number, maxMipLevels: number, clamped: boolean }}
   */
  function clampMipLevelCount(
    requestedMipLevels,
    width,
    height,
    depthOrArrayLayers = 1,
    dimension = '2d'
  ) {
    const maxMipLevels = computeMaxMipLevels(width, height, depthOrArrayLayers, dimension);
    if (requestedMipLevels === undefined || requestedMipLevels === null) {
      return { mipLevelCount: 1, maxMipLevels, clamped: false };
    }
    const numeric = Number(requestedMipLevels);
    if (!Number.isFinite(numeric)) {
      return { mipLevelCount: 1, maxMipLevels, clamped: true };
    }
    const truncated = Math.floor(numeric);
    const clampedValue = Math.min(maxMipLevels, Math.max(1, truncated));
    return {
      mipLevelCount: clampedValue,
      maxMipLevels,
      clamped: clampedValue !== numeric,
    };
  }

  /**
   * Clamp a texture `sampleCount` to WebGPU's valid set `{1, 4}`, falling back
   * to `1` whenever `mipLevelCount > 1` or multisampling is unsupported for the
   * texture's format/dimension/layers/usage.
   *
   * @param {number | undefined} requestedSampleCount
   * @param {{
   *   mipLevelCount?: number,
   *   format?: string,
   *   dimension?: string,
   *   depthOrArrayLayers?: number,
   *   usage?: number,
   * }} [context]
   * @returns {{ sampleCount: number, clamped: boolean }}
   */
  function clampSampleCount(requestedSampleCount, context = {}) {
    if (requestedSampleCount === undefined || requestedSampleCount === null) {
      return { sampleCount: 1, clamped: false };
    }
    const numeric = Number(requestedSampleCount);
    if (!Number.isFinite(numeric)) {
      return { sampleCount: 1, clamped: true };
    }

    const mipLevelCount = context.mipLevelCount !== undefined ? context.mipLevelCount : 1;
    const format = context.format || BC_FALLBACK_FORMAT;
    const dimension = context.dimension || '2d';
    const depthOrArrayLayers =
      context.depthOrArrayLayers !== undefined ? context.depthOrArrayLayers : 1;
    const usage = context.usage !== undefined ? context.usage >>> 0 : 0;

    const multisampleSupported =
      mipLevelCount <= 1 &&
      dimension === '2d' &&
      depthOrArrayLayers === 1 &&
      !isBcFormat(format) &&
      (usage & GPU_TEXTURE_USAGE.STORAGE_BINDING) === 0;

    let resolved;
    if (numeric <= 1) {
      resolved = 1;
    } else if (numeric >= 4 && multisampleSupported) {
      resolved = 4;
    } else {
      // Unsupported MSAA count (e.g. 2x) or texture constraints disallow MSAA.
      resolved = 1;
    }

    return {
      sampleCount: resolved,
      clamped: resolved !== numeric,
    };
  }

  /**
   * Round a single texture dimension up to a multiple of 4 (minimum 4) for BCn blocks.
   *
   * @param {number} dim
   * @returns {number}
   */
  function alignBcDimension(dim) {
    const n = Math.max(1, Math.floor(Number(dim) || 1));
    return Math.ceil(n / 4) * 4;
  }

  /**
   * Align a WebGPU `createBuffer` byte size to 4 bytes, enforcing a minimum of
   * 4 bytes (or 16 bytes when `GPUBufferUsage.UNIFORM` is present).
   * Uses floating-point ceiling rather than 32-bit bitwise ops so sizes > 2 GiB
   * never overflow.
   *
   * @param {number | bigint} requestedSize
   * @param {number} [usage=0]
   * @returns {{ requestedSize: number, alignedSize: number, padded: boolean, isUniform: boolean }}
   */
  function alignBufferSize(requestedSize, usage = 0) {
    const numeric = typeof requestedSize === 'bigint' ? Number(requestedSize) : Number(requestedSize);
    const rawSize = Number.isFinite(numeric) && numeric > 0 ? Math.floor(numeric) : 0;
    const usageFlags = Number(usage) >>> 0;
    const isUniform = (usageFlags & GPU_BUFFER_USAGE.UNIFORM) !== 0;
    const minSize = isUniform ? UNIFORM_BUFFER_MIN_BYTES : BUFFER_ALIGNMENT_BYTES;
    const aligned4 = rawSize <= 0 ? BUFFER_ALIGNMENT_BYTES : Math.ceil(rawSize / 4) * 4;
    const alignedSize = Math.max(minSize, aligned4);
    return {
      requestedSize: rawSize,
      alignedSize,
      padded: alignedSize !== rawSize,
      isUniform,
    };
  }

  /**
   * Normalize and 4-byte-align a `queue.writeBuffer` payload so odd-length
   * CBV/VB uploads never fail WebGPU validation.
   *
   * @param {ArrayBufferView | ArrayBuffer | SharedArrayBuffer} data
   * @param {number} [dataOffset]
   * @param {number} [size]
   * @returns {{ bytes: Uint8Array, originalByteLength: number, alignedByteLength: number, padded: boolean }}
   */
  function padBufferUploadData(data, dataOffset, size) {
    let byteView;
    if (data instanceof Uint8Array) {
      const start = dataOffset !== undefined ? Math.max(0, Math.floor(dataOffset)) : 0;
      const maxLen = Math.max(0, data.byteLength - start);
      const len = size !== undefined ? Math.min(maxLen, Math.max(0, Math.floor(size))) : maxLen;
      byteView = data.subarray(start, start + len);
    } else if (ArrayBuffer.isView(data)) {
      const elemSize = data.BYTES_PER_ELEMENT || 1;
      const startBytes =
        dataOffset !== undefined ? Math.max(0, Math.floor(dataOffset) * elemSize) : 0;
      const maxBytes = Math.max(0, data.byteLength - startBytes);
      const lenBytes =
        size !== undefined
          ? Math.min(maxBytes, Math.max(0, Math.floor(size) * elemSize))
          : maxBytes;
      byteView = new Uint8Array(data.buffer, data.byteOffset + startBytes, lenBytes);
    } else if (
      data instanceof ArrayBuffer ||
      (typeof SharedArrayBuffer !== 'undefined' && data instanceof SharedArrayBuffer)
    ) {
      const startBytes = dataOffset !== undefined ? Math.max(0, Math.floor(dataOffset)) : 0;
      const maxBytes = Math.max(0, data.byteLength - startBytes);
      const lenBytes =
        size !== undefined ? Math.min(maxBytes, Math.max(0, Math.floor(size))) : maxBytes;
      byteView = new Uint8Array(data, startBytes, lenBytes);
    } else {
      throw new TypeError('writeBuffer data must be an ArrayBufferView or ArrayBuffer');
    }

    const originalByteLength = byteView.byteLength;
    if (originalByteLength === 0) {
      return {
        bytes: new Uint8Array(0),
        originalByteLength: 0,
        alignedByteLength: 0,
        padded: false,
      };
    }

    const remainder = originalByteLength % BUFFER_ALIGNMENT_BYTES;
    if (remainder !== 0) {
      const alignedByteLength = originalByteLength + (BUFFER_ALIGNMENT_BYTES - remainder);
      const paddedCopy = new Uint8Array(alignedByteLength);
      paddedCopy.set(byteView, 0);
      return {
        bytes: paddedCopy,
        originalByteLength,
        alignedByteLength,
        padded: true,
      };
    }

    const isShared =
      typeof SharedArrayBuffer !== 'undefined' && byteView.buffer instanceof SharedArrayBuffer;
    if (isShared) {
      const copy = new Uint8Array(originalByteLength);
      copy.set(byteView, 0);
      return {
        bytes: copy,
        originalByteLength,
        alignedByteLength: originalByteLength,
        padded: false,
      };
    }

    return {
      bytes: byteView,
      originalByteLength,
      alignedByteLength: originalByteLength,
      padded: false,
    };
  }

  /**
   * Decode a binary ABI v2 `CreateTexture2` payload (opcode 35, 40 bytes,
   * `GpuCreateTexture2Payload` in `runtime/abi/playgta6_abi.h`).
   *
   * Preserves raw `mipLevelCount` and `sampleCount` so `sanitizeTextureDescriptor`
   * can clamp them accurately and update telemetry counters.
   *
   * @param {DataView} view
   * @param {number} [offset=0]
   * @param {number} [payloadBytes]
   * @returns {{ id: number, descriptor?: object, error?: string }}
   */
  function decodeCreateTexture2(view, offset = 0, payloadBytes) {
    if (!(view instanceof DataView)) {
      throw new TypeError('decodeCreateTexture2 expects a DataView');
    }
    const available =
      payloadBytes !== undefined ? payloadBytes : view.byteLength - offset;
    if (available < CREATE_TEXTURE2_PAYLOAD_BYTES || offset + CREATE_TEXTURE2_PAYLOAD_BYTES > view.byteLength) {
      return {
        id: 0,
        error: `CreateTexture2: payload ${available} < ${CREATE_TEXTURE2_PAYLOAD_BYTES} bytes`,
      };
    }

    const id = view.getUint32(offset, true);
    const width = view.getUint32(offset + 4, true);
    const height = view.getUint32(offset + 8, true);
    const depthOrArrayLayers = Math.max(1, view.getUint32(offset + 12, true));
    const mipLevelCount = view.getUint32(offset + 16, true);
    const sampleCount = view.getUint32(offset + 20, true);
    const usage = view.getUint32(offset + 24, true);
    const dimensionCode = view.getUint8(offset + 28);
    const formatCode = view.getUint8(offset + 29);
    const viewFormatCount = view.getUint8(offset + 30);

    const format = TEXTURE_FORMAT_NAMES[formatCode];
    if (formatCode === 0 || format === undefined) {
      return {
        id,
        error: `CreateTexture2 ${id}: invalid texture format code ${formatCode}`,
      };
    }

    const dimension = TEXTURE_DIMENSION_NAMES[dimensionCode];
    if (dimension === undefined) {
      return {
        id,
        error: `CreateTexture2 ${id}: invalid dimension ${dimensionCode}`,
      };
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
      const name = TEXTURE_FORMAT_NAMES[code];
      if (code === 0 || name === undefined) {
        return {
          id,
          error: `CreateTexture2 ${id}: invalid view format code ${code}`,
        };
      }
      viewFormats.push(name);
    }

    const descriptor = {
      size: { width, height, depthOrArrayLayers },
      mipLevelCount,
      sampleCount,
      dimension,
      format,
      usage,
    };
    if (viewFormats.length > 0) {
      descriptor.viewFormats = viewFormats;
    }

    return { id, descriptor };
  }

  function extractTextureSize(descriptor, dimension) {
    let rawWidth = 1;
    let rawHeight = 1;
    let rawDepth = 1;

    if (descriptor.size !== undefined && descriptor.size !== null) {
      if (Array.isArray(descriptor.size)) {
        rawWidth = descriptor.size[0] !== undefined ? descriptor.size[0] : 1;
        rawHeight = descriptor.size[1] !== undefined ? descriptor.size[1] : 1;
        rawDepth = descriptor.size[2] !== undefined ? descriptor.size[2] : 1;
      } else if (typeof descriptor.size === 'object') {
        rawWidth = descriptor.size.width !== undefined ? descriptor.size.width : 1;
        rawHeight = descriptor.size.height !== undefined ? descriptor.size.height : 1;
        rawDepth =
          descriptor.size.depthOrArrayLayers !== undefined
            ? descriptor.size.depthOrArrayLayers
            : descriptor.size.depth !== undefined
              ? descriptor.size.depth
              : 1;
      }
    } else {
      rawWidth = descriptor.width !== undefined ? descriptor.width : 1;
      rawHeight = descriptor.height !== undefined ? descriptor.height : 1;
      rawDepth =
        descriptor.depthOrArrayLayers !== undefined
          ? descriptor.depthOrArrayLayers
          : descriptor.depth !== undefined
            ? descriptor.depth
            : descriptor.arrayLayers !== undefined
              ? descriptor.arrayLayers
              : 1;
    }

    const width = Math.max(1, Math.floor(Number(rawWidth) || 1));
    const height = dimension === '1d' ? 1 : Math.max(1, Math.floor(Number(rawHeight) || 1));
    const depthOrArrayLayers =
      dimension === '1d' ? 1 : Math.max(1, Math.floor(Number(rawDepth) || 1));

    return { width, height, depthOrArrayLayers };
  }

  /**
   * Sanitize a WebGPU texture descriptor (including `CreateTexture2` metadata):
   * - Falls back `BC1..BC7` formats to `'rgba8unorm'` when `device.features`
   *   exists and lacks `'texture-compression-bc'`.
   * - Aligns `BC1..BC7` width/height up to multiples of 4 when BC is supported.
   * - Clamps `mipLevelCount` into `[1, floor(log2(max(width, height))) + 1]`
   *   (or including `depth` for `'3d'`).
   * - Clamps `sampleCount` to `{1, 4}` (falling back to `1` if `mipLevelCount > 1`
   *   or multisampling is unsupported).
   * - Normalizes `viewFormats` (remapping BC view formats on BC fallback).
   *
   * @param {object} rawDescriptor
   * @param {object | null | undefined} [device]
   * @param {{ useV1FormatTable?: boolean }} [options]
   * @returns {{
   *   descriptor: object,
   *   originalFormat: string,
   *   effectiveFormat: string,
   *   bcFormatFallback: boolean,
   *   bcDimensionAligned: boolean,
   *   mipClamped: boolean,
   *   sampleClamped: boolean,
   * }}
   */
  function sanitizeTextureDescriptor(rawDescriptor, device, options = {}) {
    if (!rawDescriptor || typeof rawDescriptor !== 'object') {
      throw new TypeError('sanitizeTextureDescriptor expects a descriptor object');
    }

    const dimension = resolveTextureDimension(rawDescriptor.dimension);
    const originalFormat = resolveTextureFormat(
      rawDescriptor.format,
      Boolean(options.useV1FormatTable)
    );
    const usage =
      rawDescriptor.usage !== undefined
        ? Number(rawDescriptor.usage) >>> 0
        : rawDescriptor.usageFlags !== undefined
          ? Number(rawDescriptor.usageFlags) >>> 0
          : GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.COPY_DST;

    let { width, height, depthOrArrayLayers } = extractTextureSize(rawDescriptor, dimension);

    let effectiveFormat = originalFormat;
    let bcFormatFallback = false;
    let bcDimensionAligned = false;

    if (isBcFormat(originalFormat)) {
      if (!deviceSupportsBc(device)) {
        effectiveFormat = BC_FALLBACK_FORMAT;
        bcFormatFallback = true;
      } else {
        const alignedWidth = alignBcDimension(width);
        const alignedHeight = dimension === '1d' ? height : alignBcDimension(height);
        if (alignedWidth !== width || alignedHeight !== height) {
          width = alignedWidth;
          height = alignedHeight;
          bcDimensionAligned = true;
        }
      }
    }

    const requestedMipLevels =
      rawDescriptor.mipLevelCount !== undefined
        ? rawDescriptor.mipLevelCount
        : rawDescriptor.mipLevels;
    const mipResult = clampMipLevelCount(
      requestedMipLevels,
      width,
      height,
      depthOrArrayLayers,
      dimension
    );

    const requestedSampleCount =
      rawDescriptor.sampleCount !== undefined
        ? rawDescriptor.sampleCount
        : rawDescriptor.samples;
    const sampleResult = clampSampleCount(requestedSampleCount, {
      mipLevelCount: mipResult.mipLevelCount,
      format: effectiveFormat,
      dimension,
      depthOrArrayLayers,
      usage,
    });

    const sanitizedDesc = {
      size: { width, height, depthOrArrayLayers },
      mipLevelCount: mipResult.mipLevelCount,
      sampleCount: sampleResult.sampleCount,
      dimension,
      format: effectiveFormat,
      usage,
    };

    if (typeof rawDescriptor.label === 'string' && rawDescriptor.label.length > 0) {
      sanitizedDesc.label = rawDescriptor.label;
    }

    if (Array.isArray(rawDescriptor.viewFormats) && rawDescriptor.viewFormats.length > 0) {
      const normalizedViews = [];
      const seen = new Set();
      for (let i = 0; i < rawDescriptor.viewFormats.length; i += 1) {
        const rawVf = rawDescriptor.viewFormats[i];
        let vf =
          typeof rawVf === 'string'
            ? rawVf.toLowerCase()
            : typeof rawVf === 'number' && rawVf > 0
              ? TEXTURE_FORMAT_NAMES[rawVf]
              : undefined;
        if (!vf || vf === 'undefined') {
          continue;
        }
        if (bcFormatFallback && isBcFormat(vf)) {
          vf = vf.endsWith('-srgb') ? BC_FALLBACK_SRGB_FORMAT : BC_FALLBACK_FORMAT;
        }
        if (!seen.has(vf)) {
          seen.add(vf);
          normalizedViews.push(vf);
        }
      }
      if (normalizedViews.length > 0) {
        sanitizedDesc.viewFormats = normalizedViews;
      }
    }

    return {
      descriptor: sanitizedDesc,
      originalFormat,
      effectiveFormat,
      bcFormatFallback,
      bcDimensionAligned,
      mipClamped: mipResult.clamped,
      sampleClamped: sampleResult.clamped,
    };
  }

  /**
   * Hardened WebGPU resource table managing buffers, textures (v1 & ABI v2
   * CreateTexture2), texture views, samplers, and shader modules with
   * automatic BC fallback/alignment, mip/sample clamping, and 4-byte buffer
   * alignment/padding.
   */
  class ResourceTable {
    /**
     * @param {object} device WebGPU GPUDevice
     * @param {{
     *   resourceRegistry?: object | null,
     *   registry?: object | null,
     *   onWarning?: ((event: { kind: string, id: number, detail: string }) => void) | null,
     * }} [options]
     */
    constructor(device, options = {}) {
      if (!device || typeof device !== 'object') {
        throw new TypeError('ResourceTable requires a valid WebGPU device object');
      }
      if (options === null || typeof options !== 'object') {
        throw new TypeError('ResourceTable options must be an object');
      }

      this.device = device;
      this.resourceRegistry = options.resourceRegistry || options.registry || null;
      this._onWarning = typeof options.onWarning === 'function' ? options.onWarning : null;

      /** @type {Map<number, object>} */
      this.buffers = new Map();
      /** @type {Map<number, object>} */
      this._bufferMeta = new Map();
      /** @type {Map<number, object>} */
      this.textures = new Map();
      /** @type {Map<number, object>} */
      this.textureViews = new Map();
      /** @type {Map<number, object>} */
      this._textureMeta = new Map();
      /** @type {Map<number, object>} */
      this.samplers = new Map();
      /** @type {Map<number, object>} */
      this.shaderModules = new Map();

      // Sanitized hardening counters exposed in getStats().
      this._bcFormatFallbackCount = 0;
      this._bcDimensionAlignCount = 0;
      this._bufferAlignPadCount = 0;
      this._mipClampCount = 0;
      this._sampleClampCount = 0;

      // Sanitized lifecycle counters.
      this._buffersCreated = 0;
      this._texturesCreated = 0;
      this._texture2Created = 0;
      this._samplersCreated = 0;
      this._shaderModulesCreated = 0;
      this._bufferWrites = 0;
      this._destroyCount = 0;
      this._errors = 0;
      this._lastError = null;

      // Sanitized occupancy, overwrite & lookup telemetry counters.
      this._peakLiveHandles = 0;
      this._overwrites = 0;
      this._missingLookups = 0;
      this._totalInserts = 0;
      this._totalDeletes = 0;
    }

    _getLiveHandleCount() {
      return (
        this.buffers.size +
        this.textures.size +
        this.samplers.size +
        this.shaderModules.size
      );
    }

    _updatePeakLiveHandles() {
      const live = this._getLiveHandleCount();
      if (live > this._peakLiveHandles) {
        this._peakLiveHandles = live;
      }
    }

    _recordInsert(wasOverwrite) {
      this._totalInserts += 1;
      if (wasOverwrite) {
        this._overwrites += 1;
      }
      this._updatePeakLiveHandles();
    }

    _warn(kind, id, detail) {
      if (this._onWarning) {
        this._onWarning({ kind, id, detail });
      }
    }

    _recordError(message) {
      this._errors += 1;
      this._lastError = String(message);
    }

    /**
     * Create and track a WebGPU buffer with 4-byte size alignment (minimum 4
     * bytes, or 16 bytes when `GPUBufferUsage.UNIFORM` is set).
     *
     * Accepts either:
     * - `createBuffer(id, { size, usage, mappedAtCreation, label })`
     * - `createBuffer(id, byteSize, usage)`
     *
     * @param {number} id
     * @param {object | number | bigint} descOrSize
     * @param {number} [usageArg]
     * @returns {object} GPUBuffer
     */
    createBuffer(id, descOrSize, usageArg) {
      let rawSize;
      let usage;
      let mappedAtCreation = false;
      let label = `buffer:${id}`;

      if (descOrSize !== null && typeof descOrSize === 'object') {
        rawSize = descOrSize.size !== undefined ? descOrSize.size : descOrSize.byteSize;
        usage = descOrSize.usage !== undefined ? descOrSize.usage : descOrSize.usageFlags;
        mappedAtCreation = Boolean(descOrSize.mappedAtCreation);
        if (typeof descOrSize.label === 'string' && descOrSize.label.length > 0) {
          label = descOrSize.label;
        }
      } else {
        rawSize = descOrSize;
        usage = usageArg;
      }

      const usageFlags =
        usage !== undefined
          ? Number(usage) >>> 0
          : GPU_BUFFER_USAGE.COPY_DST | GPU_BUFFER_USAGE.VERTEX;

      const aligned = alignBufferSize(rawSize, usageFlags);
      if (aligned.padded) {
        this._bufferAlignPadCount += 1;
        this._warn(
          'buffer_size_align',
          id,
          `aligned buffer size ${aligned.requestedSize} -> ${aligned.alignedSize}`
        );
      }

      const existed = this.buffers.has(id);
      if (existed) {
        this._destroyBufferInternal(id, false);
      }

      const createDesc = {
        size: aligned.alignedSize,
        usage: usageFlags,
      };
      if (mappedAtCreation) {
        createDesc.mappedAtCreation = true;
      }
      if (label) {
        createDesc.label = label;
      }

      let buffer;
      try {
        buffer = this.device.createBuffer(createDesc);
      } catch (err) {
        if (existed) {
          this._totalDeletes += 1;
        }
        throw err;
      }
      this.buffers.set(id, buffer);
      this._bufferMeta.set(id, {
        id,
        requestedSize: aligned.requestedSize,
        alignedSize: aligned.alignedSize,
        usage: usageFlags,
        isUniform: aligned.isUniform,
        padded: aligned.padded,
        label,
      });

      if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
        this.resourceRegistry.register('buffer', id, buffer, {
          byteSize: aligned.alignedSize,
          label,
        });
      }

      this._buffersCreated += 1;
      this._recordInsert(existed);
      return buffer;
    }

    /**
     * Upload bytes to a tracked buffer (or direct GPUBuffer), automatically
     * padding non-4-byte-aligned payloads into a zero-padded 4-byte-aligned copy.
     *
     * @param {number | object} idOrBuffer
     * @param {number} bufferOffset
     * @param {ArrayBufferView | ArrayBuffer | SharedArrayBuffer} data
     * @param {number} [dataOffset]
     * @param {number} [size]
     * @returns {boolean}
     */
    writeBuffer(idOrBuffer, bufferOffset, data, dataOffset, size) {
      let buffer = null;
      let resourceId = -1;
      if (typeof idOrBuffer === 'number') {
        resourceId = idOrBuffer;
        buffer = this.buffers.get(idOrBuffer) || null;
        if (!buffer) {
          this._missingLookups += 1;
        }
      } else if (idOrBuffer && typeof idOrBuffer === 'object') {
        buffer = idOrBuffer;
      }
      if (!buffer) {
        return false;
      }

      const dstOffset = Math.max(0, Math.floor(Number(bufferOffset) || 0));
      const upload = padBufferUploadData(data, dataOffset, size);
      if (upload.alignedByteLength === 0) {
        return true;
      }

      if (upload.padded) {
        this._bufferAlignPadCount += 1;
        this._warn(
          'buffer_write_pad',
          resourceId,
          `padded writeBuffer ${upload.originalByteLength} -> ${upload.alignedByteLength} bytes`
        );
      }

      this.device.queue.writeBuffer(buffer, dstOffset, upload.bytes);
      this._bufferWrites += 1;
      return true;
    }

    /**
     * Create and track a WebGPU texture with full mip/sample clamping and BC
     * feature fallback or 4x4 dimension alignment.
     *
     * @param {number} id
     * @param {object} descriptor
     * @param {{ useV1FormatTable?: boolean, isTexture2?: boolean }} [options]
     * @returns {object} GPUTexture
     */
    createTexture(id, descriptor, options = {}) {
      const sanitized = sanitizeTextureDescriptor(descriptor, this.device, {
        useV1FormatTable: Boolean(options.useV1FormatTable),
      });

      if (sanitized.bcFormatFallback) {
        this._bcFormatFallbackCount += 1;
        this._warn(
          'bc_format_fallback',
          id,
          `${sanitized.originalFormat} -> ${sanitized.effectiveFormat}`
        );
      }
      if (sanitized.bcDimensionAligned) {
        this._bcDimensionAlignCount += 1;
        this._warn(
          'bc_dimension_align',
          id,
          `aligned BC dimensions to ${sanitized.descriptor.size.width}x${sanitized.descriptor.size.height}`
        );
      }
      if (sanitized.mipClamped) {
        this._mipClampCount += 1;
        this._warn(
          'mip_clamp',
          id,
          `clamped mipLevelCount to ${sanitized.descriptor.mipLevelCount}`
        );
      }
      if (sanitized.sampleClamped) {
        this._sampleClampCount += 1;
        this._warn(
          'sample_clamp',
          id,
          `clamped sampleCount to ${sanitized.descriptor.sampleCount}`
        );
      }

      const existed = this.textures.has(id);
      this._destroyTextureInternal(id, false);

      let texture;
      try {
        texture = this.device.createTexture(sanitized.descriptor);
      } catch (err) {
        if (existed) {
          this._totalDeletes += 1;
        }
        throw err;
      }
      this.textures.set(id, texture);

      let textureView = null;
      if (texture && typeof texture.createView === 'function') {
        textureView = texture.createView();
        this.textureViews.set(id, textureView);
      }

      const meta = {
        id,
        width: sanitized.descriptor.size.width,
        height: sanitized.descriptor.size.height,
        depthOrArrayLayers: sanitized.descriptor.size.depthOrArrayLayers,
        mipLevelCount: sanitized.descriptor.mipLevelCount,
        sampleCount: sanitized.descriptor.sampleCount,
        dimension: sanitized.descriptor.dimension,
        format: sanitized.effectiveFormat,
        originalFormat: sanitized.originalFormat,
        usage: sanitized.descriptor.usage,
        viewFormats: sanitized.descriptor.viewFormats
          ? sanitized.descriptor.viewFormats.slice()
          : [],
        bcFormatFallback: sanitized.bcFormatFallback,
        bcDimensionAligned: sanitized.bcDimensionAligned,
        mipClamped: sanitized.mipClamped,
        sampleClamped: sanitized.sampleClamped,
      };
      this._textureMeta.set(id, meta);

      if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
        const label = sanitized.descriptor.label || `texture:${id}`;
        this.resourceRegistry.register('texture', id, texture, {
          byteSize: 0,
          label,
        });
        if (textureView) {
          this.resourceRegistry.register('textureView', id, textureView, {
            byteSize: 0,
            label: `textureView:${id}`,
          });
        }
      }

      this._texturesCreated += 1;
      if (options.isTexture2) {
        this._texture2Created += 1;
      }
      this._recordInsert(existed);
      return texture;
    }

    /**
     * Create a texture using ABI v2 `CreateTexture2` metadata (opcode 35).
     * Accepts either:
     * - `createTexture2(id, descriptor)` (structured descriptor with v2 format code/name)
     * - `createTexture2(dataView, payloadOffset, payloadBytes)` (raw 40-byte payload)
     *
     * @param {number | DataView} idOrView
     * @param {object | number} [descriptorOrOffset]
     * @param {number} [payloadBytes]
     * @returns {object}
     */
    createTexture2(idOrView, descriptorOrOffset, payloadBytes) {
      if (idOrView instanceof DataView) {
        return this.createTexture2FromPacket(
          idOrView,
          typeof descriptorOrOffset === 'number' ? descriptorOrOffset : 0,
          payloadBytes
        );
      }
      return this.createTexture(idOrView, descriptorOrOffset, {
        useV1FormatTable: false,
        isTexture2: true,
      });
    }

    /**
     * Decode and create a texture from a binary ABI v2 `CreateTexture2` payload
     * (40 bytes, `GpuCreateTexture2Payload`).
     *
     * @param {DataView} view
     * @param {number} [payloadOffset=0]
     * @param {number} [payloadBytes]
     * @returns {{ id: number, texture?: object, view?: object | null, meta?: object, error?: string }}
     */
    createTexture2FromPacket(view, payloadOffset = 0, payloadBytes) {
      const decoded = decodeCreateTexture2(view, payloadOffset, payloadBytes);
      if (decoded.error) {
        this._recordError(decoded.error);
        return { id: decoded.id, error: decoded.error };
      }

      try {
        const texture = this.createTexture(decoded.id, decoded.descriptor, {
          useV1FormatTable: false,
          isTexture2: true,
        });
        return {
          id: decoded.id,
          texture,
          view: this.textureViews.get(decoded.id) || null,
          meta: this._textureMeta.get(decoded.id) || null,
        };
      } catch (err) {
        const msg = `CreateTexture2 ${decoded.id}: ${String(err && err.message ? err.message : err)}`;
        this._recordError(msg);
        return { id: decoded.id, error: msg };
      }
    }

    createSampler(id, descriptor = {}) {
      const existed = this.samplers.has(id);
      const sampler = this.device.createSampler(descriptor);
      this.samplers.set(id, sampler);
      if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
        this.resourceRegistry.register('sampler', id, sampler, {
          byteSize: 0,
          label: descriptor.label || `sampler:${id}`,
        });
      }
      this._samplersCreated += 1;
      this._recordInsert(existed);
      return sampler;
    }

    createShaderModule(id, descriptor) {
      const existed = this.shaderModules.has(id);
      const mod = this.device.createShaderModule(descriptor);
      this.shaderModules.set(id, mod);
      if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
        this.resourceRegistry.register('shaderModule', id, mod, {
          byteSize: 0,
          label: (descriptor && descriptor.label) || `shaderModule:${id}`,
        });
      }
      this._shaderModulesCreated += 1;
      this._recordInsert(existed);
      return mod;
    }

    getBuffer(id) {
      const value = this.buffers.get(id);
      if (value === undefined) {
        this._missingLookups += 1;
      }
      return value;
    }

    getBufferMeta(id) {
      const meta = this._bufferMeta.get(id);
      if (meta === undefined) {
        this._missingLookups += 1;
        return undefined;
      }
      return { ...meta };
    }

    hasBuffer(id) {
      return this.buffers.has(id);
    }

    getTexture(id) {
      const value = this.textures.get(id);
      if (value === undefined) {
        this._missingLookups += 1;
      }
      return value;
    }

    getTextureView(id) {
      const value = this.textureViews.get(id);
      if (value === undefined) {
        this._missingLookups += 1;
      }
      return value;
    }

    getTextureMeta(id) {
      const meta = this._textureMeta.get(id);
      if (meta === undefined) {
        this._missingLookups += 1;
        return undefined;
      }
      return { ...meta, viewFormats: meta.viewFormats.slice() };
    }

    hasTexture(id) {
      return this.textures.has(id);
    }

    getSampler(id) {
      const value = this.samplers.get(id);
      if (value === undefined) {
        this._missingLookups += 1;
      }
      return value;
    }

    getShaderModule(id) {
      const value = this.shaderModules.get(id);
      if (value === undefined) {
        this._missingLookups += 1;
      }
      return value;
    }

    _destroyBufferInternal(id, countDelete) {
      const existing = this.buffers.get(id);
      if (!existing) {
        return false;
      }
      if (this.resourceRegistry && typeof this.resourceRegistry.retire === 'function') {
        const retired = this.resourceRegistry.retire('buffer', id);
        if (!retired && typeof existing.destroy === 'function') {
          existing.destroy();
        }
      } else if (typeof existing.destroy === 'function') {
        existing.destroy();
      }
      this.buffers.delete(id);
      this._bufferMeta.delete(id);
      this._destroyCount += 1;
      if (countDelete) {
        this._totalDeletes += 1;
      }
      return true;
    }

    destroyBuffer(id) {
      return this._destroyBufferInternal(id, true);
    }

    _destroyTextureInternal(id, countDelete) {
      const existing = this.textures.get(id);
      if (!existing) {
        this.textureViews.delete(id);
        this._textureMeta.delete(id);
        return false;
      }
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
      this.textureViews.delete(id);
      this._textureMeta.delete(id);
      this._destroyCount += 1;
      if (countDelete) {
        this._totalDeletes += 1;
      }
      return true;
    }

    destroyTexture(id) {
      return this._destroyTextureInternal(id, true);
    }

    destroy(id) {
      let count = 0;
      if (this.destroyBuffer(id)) count += 1;
      if (this.destroyTexture(id)) count += 1;
      if (this.samplers.delete(id)) {
        if (this.resourceRegistry && typeof this.resourceRegistry.retire === 'function') {
          this.resourceRegistry.retire('sampler', id);
        }
        this._destroyCount += 1;
        this._totalDeletes += 1;
        count += 1;
      }
      if (this.shaderModules.delete(id)) {
        if (this.resourceRegistry && typeof this.resourceRegistry.retire === 'function') {
          this.resourceRegistry.retire('shaderModule', id);
        }
        this._destroyCount += 1;
        this._totalDeletes += 1;
        count += 1;
      }
      return count;
    }

    destroyAll() {
      for (const id of Array.from(this.buffers.keys())) {
        this.destroyBuffer(id);
      }
      for (const id of Array.from(this.textures.keys())) {
        this.destroyTexture(id);
      }
      for (const id of Array.from(this.samplers.keys())) {
        this.destroy(id);
      }
      for (const id of Array.from(this.shaderModules.keys())) {
        this.destroy(id);
      }
    }

    /**
     * Destroy all tracked GPU resources and clear internal handle/metadata maps.
     * Pass `{ resetStats: true }` to also reset telemetry counters.
     *
     * @param {{ resetStats?: boolean }} [options]
     */
    clear(options = {}) {
      this.destroyAll();
      this._bufferMeta.clear();
      this.textureViews.clear();
      this._textureMeta.clear();
      if (options && options.resetStats) {
        this.resetStats();
      }
    }

    /**
     * Return sanitized diagnostic counters (no host paths or raw payload data).
     */
    getStats() {
      return {
        bcFormatFallbackCount: this._bcFormatFallbackCount,
        bcDimensionAlignCount: this._bcDimensionAlignCount,
        bufferAlignPadCount: this._bufferAlignPadCount,
        mipClampCount: this._mipClampCount,
        sampleClampCount: this._sampleClampCount,
        buffersCreated: this._buffersCreated,
        texturesCreated: this._texturesCreated,
        texture2Created: this._texture2Created,
        samplersCreated: this._samplersCreated,
        shaderModulesCreated: this._shaderModulesCreated,
        bufferWrites: this._bufferWrites,
        buffersActive: this.buffers.size,
        texturesActive: this.textures.size,
        textureViewsActive: this.textureViews.size,
        samplersActive: this.samplers.size,
        shaderModulesActive: this.shaderModules.size,
        destroyCount: this._destroyCount,
        errors: this._errors,
        liveHandles: this._getLiveHandleCount(),
        peakLiveHandles: this._peakLiveHandles,
        overwrites: this._overwrites,
        missingLookups: this._missingLookups,
        totalInserts: this._totalInserts,
        totalDeletes: this._totalDeletes,
      };
    }

    resetStats() {
      this._bcFormatFallbackCount = 0;
      this._bcDimensionAlignCount = 0;
      this._bufferAlignPadCount = 0;
      this._mipClampCount = 0;
      this._sampleClampCount = 0;
      this._buffersCreated = 0;
      this._texturesCreated = 0;
      this._texture2Created = 0;
      this._samplersCreated = 0;
      this._shaderModulesCreated = 0;
      this._bufferWrites = 0;
      this._destroyCount = 0;
      this._errors = 0;
      this._lastError = null;
      this._peakLiveHandles = this._getLiveHandleCount();
      this._overwrites = 0;
      this._missingLookups = 0;
      this._totalInserts = 0;
      this._totalDeletes = 0;
    }
  }

  const exportsObj = Object.freeze({
    CREATE_TEXTURE2_OPCODE,
    CREATE_TEXTURE2_PAYLOAD_BYTES,
    MAX_TEXTURE_VIEW_FORMATS,
    GPU_BUFFER_USAGE,
    GPU_TEXTURE_USAGE,
    V1_TEXTURE_FORMAT_NAMES,
    TEXTURE_FORMAT_NAMES,
    TEXTURE_DIMENSION_NAMES,
    BC_TEXTURE_FORMATS,
    BC_FALLBACK_FORMAT,
    TEXTURE_COMPRESSION_BC_FEATURE,
    BUFFER_ALIGNMENT_BYTES,
    UNIFORM_BUFFER_MIN_BYTES,
    isBcFormat,
    deviceSupportsBc,
    computeMaxMipLevels,
    clampMipLevelCount,
    clampSampleCount,
    alignBcDimension,
    alignBufferSize,
    padBufferUploadData,
    decodeCreateTexture2,
    sanitizeTextureDescriptor,
    ResourceTable,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6ResourceTable = exportsObj;
  } else if (globalScope) {
    globalScope.PlayGta6ResourceTable = exportsObj;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      CREATE_TEXTURE2_OPCODE,
      CREATE_TEXTURE2_PAYLOAD_BYTES,
      MAX_TEXTURE_VIEW_FORMATS,
      GPU_BUFFER_USAGE,
      GPU_TEXTURE_USAGE,
      V1_TEXTURE_FORMAT_NAMES,
      TEXTURE_FORMAT_NAMES,
      TEXTURE_DIMENSION_NAMES,
      BC_TEXTURE_FORMATS,
      BC_FALLBACK_FORMAT,
      TEXTURE_COMPRESSION_BC_FEATURE,
      BUFFER_ALIGNMENT_BYTES,
      UNIFORM_BUFFER_MIN_BYTES,
      isBcFormat,
      deviceSupportsBc,
      computeMaxMipLevels,
      clampMipLevelCount,
      clampSampleCount,
      alignBcDimension,
      alignBufferSize,
      padBufferUploadData,
      decodeCreateTexture2,
      sanitizeTextureDescriptor,
      ResourceTable,
    };
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
