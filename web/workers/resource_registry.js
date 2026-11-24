'use strict';

// WebGPU resource handle registry, frame-deferred garbage collector, and
// Frame-1 hardened resource creation/upload helpers (CreateTexture2 ABI v2
// opcode 35, BC1..BC7 fallback & 4x4 alignment, mip/sampleCount clamping,
// and 4-byte WebGPU buffer size/write alignment).
// Zero external dependencies; UMD/CommonJS/ESM compatible for Web Workers
// (importScripts -> globalThis.PlayGta6ResourceRegistry), browsers, and
// Node.js (require() / import()).

(function (root) {
  const DEFAULT_FRAMES_IN_FLIGHT = 3;
  const DEFAULT_SOFT_BUDGET_BYTES = 512 * 1024 * 1024; // 512 MiB

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

  const BC_8BYTE_BLOCK_FORMATS = new Set([
    'bc1-rgba-unorm',
    'bc1-rgba-unorm-srgb',
    'bc4-r-unorm',
    'bc4-r-snorm',
  ]);

  // Core WebGPU formats that support multisampling (sampleCount === 4).
  const MULTISAMPLE_SUPPORTED_FORMATS = new Set([
    'r8unorm',
    'r8uint',
    'r8sint',
    'rg8unorm',
    'rg8uint',
    'rg8sint',
    'rgba8unorm',
    'rgba8unorm-srgb',
    'rgba8uint',
    'rgba8sint',
    'bgra8unorm',
    'bgra8unorm-srgb',
    'r16uint',
    'r16sint',
    'r16float',
    'rg16uint',
    'rg16sint',
    'rg16float',
    'rgba16uint',
    'rgba16sint',
    'rgba16float',
    'rgb10a2uint',
    'rgb10a2unorm',
    'rg11b10ufloat',
    'r32float',
    'stencil8',
    'depth16unorm',
    'depth24plus',
    'depth24plus-stencil8',
    'depth32float',
    'depth32float-stencil8',
  ]);

  const UNCOMPRESSED_FORMAT_BPP = Object.freeze({
    r8unorm: 1,
    r8snorm: 1,
    r8uint: 1,
    r8sint: 1,
    stencil8: 1,
    r16unorm: 2,
    r16snorm: 2,
    r16uint: 2,
    r16sint: 2,
    r16float: 2,
    rg8unorm: 2,
    rg8snorm: 2,
    rg8uint: 2,
    rg8sint: 2,
    depth16unorm: 2,
    r32uint: 4,
    r32sint: 4,
    r32float: 4,
    rg16unorm: 4,
    rg16snorm: 4,
    rg16uint: 4,
    rg16sint: 4,
    rg16float: 4,
    rgba8unorm: 4,
    'rgba8unorm-srgb': 4,
    rgba8snorm: 4,
    rgba8uint: 4,
    rgba8sint: 4,
    bgra8unorm: 4,
    'bgra8unorm-srgb': 4,
    rgb9e5ufloat: 4,
    rgb10a2uint: 4,
    rgb10a2unorm: 4,
    rg11b10ufloat: 4,
    depth24plus: 4,
    'depth24plus-stencil8': 4,
    depth32float: 4,
    rg32uint: 8,
    rg32sint: 8,
    rg32float: 8,
    rgba16unorm: 8,
    rgba16snorm: 8,
    rgba16uint: 8,
    rgba16sint: 8,
    rgba16float: 8,
    'depth32float-stencil8': 8,
    rgba32uint: 16,
    rgba32sint: 16,
    rgba32float: 16,
  });

  const RESOURCE_KIND = Object.freeze({
    BUFFER: 'buffer',
    TEXTURE: 'texture',
    TEXTURE_VIEW: 'textureView',
    SAMPLER: 'sampler',
    BIND_GROUP: 'bindGroup',
    BIND_GROUP_LAYOUT: 'bindGroupLayout',
    PIPELINE_LAYOUT: 'pipelineLayout',
    SHADER_MODULE: 'shaderModule',
    RENDER_PIPELINE: 'renderPipeline',
    COMPUTE_PIPELINE: 'computePipeline',
  });

  const RESOURCE_KINDS = Object.freeze(Object.values(RESOURCE_KIND));
  const VALID_KIND_SET = new Set(RESOURCE_KINDS);

  function assertValidKind(kind) {
    if (typeof kind !== 'string' || !VALID_KIND_SET.has(kind)) {
      throw new RangeError(
        `Invalid WebGPU resource kind: ${JSON.stringify(kind)} (expected one of ${RESOURCE_KINDS.join(', ')})`
      );
    }
  }

  function assertValidHandle(handle) {
    if (typeof handle !== 'number' || !Number.isSafeInteger(handle) || handle < 0) {
      throw new RangeError(
        `Invalid resource handle: ${String(handle)} (expected a non-negative safe integer)`
      );
    }
  }

  function assertPositiveInteger(value, name) {
    if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1) {
      throw new RangeError(`${name} must be a positive safe integer, got ${String(value)}`);
    }
  }

  function assertNonNegativeInteger(value, name) {
    if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) {
      throw new RangeError(`${name} must be a non-negative safe integer, got ${String(value)}`);
    }
  }

  function snapshotEntry(entry) {
    return {
      kind: entry.kind,
      handle: entry.handle,
      resource: entry.resource,
      byteSize: entry.byteSize,
      label: entry.label,
      tag: entry.tag,
      registeredAtFrame: entry.registeredAtFrame,
    };
  }

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
   * Compute the maximum valid WebGPU mip level count for texture dimensions:
   * `floor(log2(max(width, height))) + 1` (or `max(w, h, d)` for `'3d'`).
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
   * @param {number | undefined | null} requestedMipLevels
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
   * to `1` when `mipLevelCount > 1` or multisampling is invalid for the
   * format/dimension/layers/usage.
   *
   * @param {number | undefined | null} requestedSampleCount
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
    const format = (context.format || BC_FALLBACK_FORMAT).toLowerCase();
    const dimension = context.dimension || '2d';
    const depthOrArrayLayers =
      context.depthOrArrayLayers !== undefined ? context.depthOrArrayLayers : 1;
    const usage = context.usage !== undefined ? Number(context.usage) >>> 0 : 0;

    const multisampleSupported =
      mipLevelCount <= 1 &&
      dimension === '2d' &&
      depthOrArrayLayers === 1 &&
      MULTISAMPLE_SUPPORTED_FORMATS.has(format) &&
      (usage & GPU_TEXTURE_USAGE.STORAGE_BINDING) === 0;

    let resolved;
    if (numeric <= 1) {
      resolved = 1;
    } else if (numeric >= 4 && multisampleSupported) {
      resolved = 4;
    } else {
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
   * Align a WebGPU `createBuffer` byte size up to a multiple of 4 bytes:
   * `Math.max(4, Math.ceil(size / 4) * 4)`, with a minimum of 16 bytes when
   * `GPUBufferUsage.UNIFORM` is set.
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
    const aligned4 = Math.max(BUFFER_ALIGNMENT_BYTES, Math.ceil(rawSize / 4) * 4);
    const alignedSize = isUniform ? Math.max(UNIFORM_BUFFER_MIN_BYTES, aligned4) : aligned4;
    return {
      requestedSize: rawSize,
      alignedSize,
      padded: alignedSize !== rawSize,
      isUniform,
    };
  }

  /**
   * Normalize and 4-byte-align a `queue.writeBuffer` payload, zero-padding
   * non-4-byte-aligned inputs and safely copying `SharedArrayBuffer`-backed views
   * into a non-shared `Uint8Array`.
   *
   * @param {ArrayBufferView | ArrayBuffer | SharedArrayBuffer} data
   * @param {number} [dataOffset]
   * @param {number} [size]
   * @returns {{
   *   bytes: Uint8Array,
   *   originalByteLength: number,
   *   alignedByteLength: number,
   *   padded: boolean,
   *   copiedFromShared: boolean,
   * }}
   */
  function padBufferUploadData(data, dataOffset, size) {
    let byteView;
    if (data instanceof Uint8Array) {
      if (dataOffset === undefined && size === undefined) {
        byteView = data;
      } else {
        const start = dataOffset !== undefined ? Math.max(0, Math.floor(dataOffset)) : 0;
        const maxLen = Math.max(0, data.byteLength - start);
        const len = size !== undefined ? Math.min(maxLen, Math.max(0, Math.floor(size))) : maxLen;
        byteView = data.subarray(start, start + len);
      }
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
      throw new TypeError('writeBuffer data must be an ArrayBufferView, ArrayBuffer, or SharedArrayBuffer');
    }

    const originalByteLength = byteView.byteLength;
    const isShared =
      typeof SharedArrayBuffer !== 'undefined' && byteView.buffer instanceof SharedArrayBuffer;

    if (originalByteLength === 0) {
      return {
        bytes: new Uint8Array(0),
        originalByteLength: 0,
        alignedByteLength: 0,
        padded: false,
        copiedFromShared: isShared,
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
        copiedFromShared: isShared,
      };
    }

    if (isShared) {
      const copy = new Uint8Array(originalByteLength);
      copy.set(byteView, 0);
      return {
        bytes: copy,
        originalByteLength,
        alignedByteLength: originalByteLength,
        padded: false,
        copiedFromShared: true,
      };
    }

    return {
      bytes: byteView,
      originalByteLength,
      alignedByteLength: originalByteLength,
      padded: false,
      copiedFromShared: false,
    };
  }

  /**
   * Estimate VRAM byte size for a WebGPU texture descriptor (including mip chain,
   * BC block sizes, array layers / 3D depth, and MSAA sample count).
   *
   * @param {number} width
   * @param {number} height
   * @param {number} [depthOrArrayLayers=1]
   * @param {number} [mipLevelCount=1]
   * @param {string} [format='rgba8unorm']
   * @param {string} [dimension='2d']
   * @param {number} [sampleCount=1]
   * @returns {number}
   */
  function estimateTextureBytes(
    width,
    height,
    depthOrArrayLayers = 1,
    mipLevelCount = 1,
    format = BC_FALLBACK_FORMAT,
    dimension = '2d',
    sampleCount = 1
  ) {
    const w0 = Math.max(1, Math.floor(Number(width) || 1));
    const h0 = Math.max(1, Math.floor(Number(height) || 1));
    const d0 = Math.max(1, Math.floor(Number(depthOrArrayLayers) || 1));
    const mips = Math.max(1, Math.floor(Number(mipLevelCount) || 1));
    const samples = Math.max(1, Math.floor(Number(sampleCount) || 1));
    const fmt = typeof format === 'string' ? format.toLowerCase() : BC_FALLBACK_FORMAT;
    const is1D = dimension === '1d';
    const is3D = dimension === '3d';
    const compressed = isBcFormat(fmt);
    const bytesPerBlock = compressed ? (BC_8BYTE_BLOCK_FORMATS.has(fmt) ? 8 : 16) : 0;
    const bpp = UNCOMPRESSED_FORMAT_BPP[fmt] || 4;

    let total = 0;
    for (let mip = 0; mip < mips; mip += 1) {
      const scale = mip >= 31 ? Infinity : Math.pow(2, mip);
      const mw = scale === Infinity ? 1 : Math.max(1, Math.floor(w0 / scale));
      const mh = is1D ? 1 : scale === Infinity ? 1 : Math.max(1, Math.floor(h0 / scale));
      const md = is3D ? (scale === Infinity ? 1 : Math.max(1, Math.floor(d0 / scale))) : d0;
      if (compressed) {
        total += Math.ceil(mw / 4) * Math.ceil(mh / 4) * bytesPerBlock * md;
      } else {
        total += mw * mh * md * bpp * samples;
      }
    }
    return total;
  }

  /**
   * Decode a binary ABI v2 `CreateTexture2` payload (opcode 35, 40 bytes,
   * `GpuCreateTexture2Payload` in `runtime/abi/playgta6_abi.h`).
   *
   * Preserves raw `mipLevelCount` and `sampleCount` so `sanitizeTextureDescriptor`
   * can clamp them accurately and track telemetry counters.
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
    const available = payloadBytes !== undefined ? payloadBytes : view.byteLength - offset;
    if (
      available < CREATE_TEXTURE2_PAYLOAD_BYTES ||
      offset < 0 ||
      offset + CREATE_TEXTURE2_PAYLOAD_BYTES > view.byteLength
    ) {
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
              : descriptor.size.arrayLayers !== undefined
                ? descriptor.size.arrayLayers
                : 1;
      }
    } else {
      rawWidth = descriptor.width !== undefined ? descriptor.width : 1;
      rawHeight = descriptor.height !== undefined ? descriptor.height : 1;
      rawDepth =
        descriptor.depthOrArrayLayers !== undefined
          ? descriptor.depthOrArrayLayers
          : descriptor.depth_or_layers !== undefined
            ? descriptor.depth_or_layers
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
   * Sanitize a WebGPU texture descriptor (including ABI v2 `CreateTexture2`
   * fields `arrayLayers`/`depth`, `mipLevels`, `sampleCount`, `dimension`):
   * - For `bc1..bc7` formats: if `device.features` exists and lacks
   *   `'texture-compression-bc'`, falls back cleanly to `'rgba8unorm'`; when BC
   *   is supported, aligns width/height up to multiples of 4 when needed.
   * - Clamps `mipLevelCount` to `[1, floor(log2(max(width, height))) + 1]`
   *   (or `max(w, h, d)` for `'3d'`).
   * - Clamps `sampleCount` to `{1, 4}`, falling back to `1` when
   *   `mipLevelCount > 1` or multisampling is invalid for the format/dimension.
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
          : rawDescriptor.usage_flags !== undefined
            ? Number(rawDescriptor.usage_flags) >>> 0
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
        : rawDescriptor.mipLevels !== undefined
          ? rawDescriptor.mipLevels
          : rawDescriptor.mip_levels;
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
        : rawDescriptor.sample_count !== undefined
          ? rawDescriptor.sample_count
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

    const rawViewFormats =
      rawDescriptor.viewFormats !== undefined
        ? rawDescriptor.viewFormats
        : rawDescriptor.view_formats;
    if (Array.isArray(rawViewFormats) && rawViewFormats.length > 0) {
      const normalizedViews = [];
      const seen = new Set();
      for (let i = 0; i < rawViewFormats.length; i += 1) {
        const rawVf = rawViewFormats[i];
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
   * Tracks WebGPU resources by `(kind, handle)` with VRAM byte accounting,
   * frame-epoch / GPU-fence deferred destruction, and hardened Frame-1
   * texture/buffer creation & upload helpers.
   */
  class GpuResourceRegistry {
    /**
     * @param {{
     *   framesInFlight?: number,
     *   softBudgetBytes?: number | null,
     *   device?: object | null,
     *   onBudgetWarning?: ((info: {
     *     activeBytes: number,
     *     pendingRetireBytes: number,
     *     totalBytes: number,
     *     softBudgetBytes: number,
     *     currentFrame: number,
     *     kind: string,
     *     handle: number,
     *     label: string,
     *   }) => void) | null,
     *   onDestroyError?: ((err: unknown, entry: {
     *     kind: string,
     *     handle: number,
     *     byteSize: number,
     *     label: string,
     *   }) => void) | null,
     *   onSanitizeWarning?: ((event: {
     *     kind: string,
     *     handle: number,
     *     detail: string,
     *   }) => void) | null,
     * }} [options]
     */
    constructor(options = {}) {
      if (options === null || typeof options !== 'object') {
        throw new TypeError('GpuResourceRegistry constructor expects an options object');
      }

      const {
        framesInFlight = DEFAULT_FRAMES_IN_FLIGHT,
        softBudgetBytes = DEFAULT_SOFT_BUDGET_BYTES,
        device = null,
        onBudgetWarning = null,
        onDestroyError = null,
        onSanitizeWarning = null,
      } = options;

      assertPositiveInteger(framesInFlight, 'framesInFlight');

      let resolvedBudget = null;
      if (softBudgetBytes !== null && softBudgetBytes !== Infinity) {
        assertPositiveInteger(softBudgetBytes, 'softBudgetBytes');
        resolvedBudget = softBudgetBytes;
      }

      if (device !== null && typeof device !== 'object') {
        throw new TypeError('device must be an object or null when provided');
      }
      if (onBudgetWarning !== null && typeof onBudgetWarning !== 'function') {
        throw new TypeError('onBudgetWarning must be a function when provided');
      }
      if (onDestroyError !== null && typeof onDestroyError !== 'function') {
        throw new TypeError('onDestroyError must be a function when provided');
      }
      if (onSanitizeWarning !== null && typeof onSanitizeWarning !== 'function') {
        throw new TypeError('onSanitizeWarning must be a function when provided');
      }

      this.framesInFlight = framesInFlight;
      this.softBudgetBytes = resolvedBudget;
      this.device = device;
      this._onBudgetWarning = onBudgetWarning;
      this._onDestroyError = onDestroyError;
      this._onSanitizeWarning = onSanitizeWarning;

      /** @type {Map<string, Map<number, object>>} */
      this._tables = new Map();
      for (let i = 0; i < RESOURCE_KINDS.length; i += 1) {
        this._tables.set(RESOURCE_KINDS[i], new Map());
      }

      /** @type {Map<number, object>} */
      this._bufferMeta = new Map();
      /** @type {Map<number, object>} */
      this._textureMeta = new Map();

      /** @type {Array<object>} */
      this._pendingRetire = [];
      /** @type {Set<Promise<void>>} */
      this._pendingFences = new Set();
      /** @type {WeakSet<object>} */
      this._destroyedResources = new WeakSet();

      this._currentFrame = 0;
      this._generation = 0;

      this._activeBytes = 0;
      this._pendingRetireBytes = 0;
      this._peakBytes = 0;

      this._budgetExceeded = false;
      this._budgetWarningCount = 0;

      this._totalRegistered = 0;
      this._totalRetired = 0;
      this._totalDestroyed = 0;
      this._destroyErrors = 0;

      // Sanitized Frame-1 hardening counters exposed on getStats().
      this._bcFormatFallbackCount = 0;
      this._bcDimensionAlignCount = 0;
      this._bufferAlignPadCount = 0;
      this._mipClampCount = 0;
      this._sampleClampCount = 0;
    }

    /**
     * Attach or replace the WebGPU device used by `createTexture`,
     * `createTexture2`, `createBuffer`, and `writeBuffer`.
     *
     * @param {object | null} device
     */
    setDevice(device) {
      if (device !== null && typeof device !== 'object') {
        throw new TypeError('setDevice expects a WebGPU device object or null');
      }
      this.device = device;
    }

    _resolveDevice(overrideDevice, methodName) {
      const target = overrideDevice !== undefined && overrideDevice !== null ? overrideDevice : this.device;
      if (!target || typeof target !== 'object') {
        throw new TypeError(`GpuResourceRegistry.${methodName} requires a WebGPU device`);
      }
      return target;
    }

    _emitSanitizeWarning(kind, handle, detail) {
      if (this._onSanitizeWarning) {
        this._onSanitizeWarning({ kind, handle, detail });
      }
    }

    _updatePeakBytes() {
      const totalBytes = this._activeBytes + this._pendingRetireBytes;
      if (totalBytes > this._peakBytes) {
        this._peakBytes = totalBytes;
      }
      return totalBytes;
    }

    _checkBudget(context) {
      const totalBytes = this._activeBytes + this._pendingRetireBytes;
      if (this.softBudgetBytes === null) {
        this._budgetExceeded = false;
        return;
      }

      if (totalBytes > this.softBudgetBytes) {
        if (!this._budgetExceeded) {
          this._budgetExceeded = true;
          this._budgetWarningCount += 1;
          if (this._onBudgetWarning) {
            this._onBudgetWarning({
              activeBytes: this._activeBytes,
              pendingRetireBytes: this._pendingRetireBytes,
              totalBytes,
              softBudgetBytes: this.softBudgetBytes,
              currentFrame: this._currentFrame,
              kind: context ? context.kind : '',
              handle: context ? context.handle : -1,
              label: context ? context.label : '',
            });
          }
        }
      } else {
        this._budgetExceeded = false;
      }
    }

    _sealUnsealedRetirements() {
      const batch = [];
      for (let i = 0; i < this._pendingRetire.length; i += 1) {
        const entry = this._pendingRetire[i];
        if (!entry.sealed) {
          entry.sealed = true;
          batch.push(entry);
        }
      }
      return batch;
    }

    _destroyEntry(entry) {
      if (entry.destroyed) {
        return false;
      }
      entry.destroyed = true;

      const resource = entry.resource;
      entry.resource = null;

      if (resource && typeof resource === 'object' && !this._destroyedResources.has(resource)) {
        this._destroyedResources.add(resource);
        if (typeof resource.destroy === 'function') {
          try {
            resource.destroy();
          } catch (err) {
            this._destroyErrors += 1;
            if (this._onDestroyError) {
              this._onDestroyError(err, {
                kind: entry.kind,
                handle: entry.handle,
                byteSize: entry.byteSize,
                label: entry.label,
              });
            }
          }
        }
      }

      this._totalDestroyed += 1;
      return true;
    }

    /**
     * Register a WebGPU resource under `(kind, handle)`.
     * If a different resource already occupies `(kind, handle)`, the prior resource
     * is retired for deferred destruction rather than leaked.
     *
     * @param {string} kind
     * @param {number} handle
     * @param {object} resource
     * @param {{ byteSize?: number, label?: string, tag?: unknown }} [meta]
     * @returns {object} The registered resource
     */
    register(kind, handle, resource, meta = {}) {
      assertValidKind(kind);
      assertValidHandle(handle);

      if (resource === null || typeof resource !== 'object') {
        throw new TypeError('register requires a non-null resource object');
      }
      if (meta === null || typeof meta !== 'object') {
        throw new TypeError('register metadata must be an object when provided');
      }

      const { byteSize = 0, label = '', tag = null } = meta;
      assertNonNegativeInteger(byteSize, 'byteSize');
      if (typeof label !== 'string') {
        throw new TypeError('register metadata.label must be a string');
      }

      const table = this._tables.get(kind);
      const existing = table.get(handle);

      if (existing) {
        if (existing.resource === resource) {
          this._activeBytes = Math.max(0, this._activeBytes - existing.byteSize) + byteSize;
          existing.byteSize = byteSize;
          existing.label = label;
          existing.tag = tag;
          this._updatePeakBytes();
          this._checkBudget({ kind, handle, label });
          return resource;
        }
        // Cleanly retire the previous resource on handle overwrite so it is never leaked.
        this.retire(kind, handle);
      }

      const entry = {
        kind,
        handle,
        resource,
        byteSize,
        label,
        tag,
        registeredAtFrame: this._currentFrame,
        retiredAtFrame: -1,
        sealed: false,
        hasFence: false,
        fenceDone: false,
        fenceFailed: false,
        destroyed: false,
      };

      table.set(handle, entry);
      this._activeBytes += byteSize;
      this._totalRegistered += 1;

      this._updatePeakBytes();
      this._checkBudget({ kind, handle, label });

      return resource;
    }

    /**
     * Create, align, and register a WebGPU buffer under `(RESOURCE_KIND.BUFFER, handle)`.
     * Aligns `size` up to a multiple of 4 bytes (`Math.max(4, Math.ceil(size / 4) * 4)`,
     * minimum 16 bytes for `UNIFORM` usage) and tracks `bufferAlignPadCount`.
     *
     * Accepts either:
     * - `createBuffer(handle, { size, usage, mappedAtCreation, label }, options)`
     * - `createBuffer(handle, byteSize, usage, options)`
     *
     * @param {number} handle
     * @param {object | number | bigint} descOrSize
     * @param {number | object} [usageOrOptions]
     * @param {object} [maybeOptions]
     * @returns {object} GPUBuffer
     */
    createBuffer(handle, descOrSize, usageOrOptions, maybeOptions) {
      assertValidHandle(handle);

      let rawSize;
      let usage;
      let mappedAtCreation = false;
      let label = `buffer:${handle}`;
      let opts = {};

      if (descOrSize !== null && typeof descOrSize === 'object') {
        rawSize = descOrSize.size !== undefined ? descOrSize.size : descOrSize.byteSize;
        usage =
          descOrSize.usage !== undefined
            ? descOrSize.usage
            : descOrSize.usageFlags !== undefined
              ? descOrSize.usageFlags
              : descOrSize.usage_flags;
        mappedAtCreation = Boolean(descOrSize.mappedAtCreation);
        if (typeof descOrSize.label === 'string' && descOrSize.label.length > 0) {
          label = descOrSize.label;
        }
        if (usageOrOptions && typeof usageOrOptions === 'object') {
          opts = usageOrOptions;
        }
      } else {
        rawSize = descOrSize;
        usage = usageOrOptions;
        if (maybeOptions && typeof maybeOptions === 'object') {
          opts = maybeOptions;
        }
      }

      if (typeof opts.label === 'string' && opts.label.length > 0) {
        label = opts.label;
      }

      const device = this._resolveDevice(opts.device, 'createBuffer');
      const usageFlags =
        usage !== undefined
          ? Number(usage) >>> 0
          : GPU_BUFFER_USAGE.COPY_DST | GPU_BUFFER_USAGE.VERTEX;

      const aligned = alignBufferSize(rawSize, usageFlags);
      if (aligned.padded) {
        this._bufferAlignPadCount += 1;
        this._emitSanitizeWarning(
          'buffer_size_align',
          handle,
          `aligned buffer size ${aligned.requestedSize} -> ${aligned.alignedSize}`
        );
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

      const buffer = device.createBuffer(createDesc);
      this.register(RESOURCE_KIND.BUFFER, handle, buffer, {
        byteSize: aligned.alignedSize,
        label,
        tag: opts.tag !== undefined ? opts.tag : null,
      });

      this._bufferMeta.set(handle, {
        handle,
        requestedSize: aligned.requestedSize,
        alignedSize: aligned.alignedSize,
        usage: usageFlags,
        isUniform: aligned.isUniform,
        padded: aligned.padded,
        label,
      });

      return buffer;
    }

    /**
     * Upload bytes to a registered buffer handle (or direct `GPUBuffer`) via
     * `device.queue.writeBuffer`, zero-padding non-4-byte-aligned payloads into
     * a 4-byte-aligned copy and safely copying `SharedArrayBuffer` views.
     *
     * @param {number | object} handleOrBuffer
     * @param {number} bufferOffset
     * @param {ArrayBufferView | ArrayBuffer | SharedArrayBuffer} data
     * @param {number} [dataOffset]
     * @param {number} [size]
     * @param {{ device?: object }} [options]
     * @returns {boolean}
     */
    writeBuffer(handleOrBuffer, bufferOffset, data, dataOffset, size, options = {}) {
      let buffer = null;
      let handle = -1;
      if (typeof handleOrBuffer === 'number') {
        assertValidHandle(handleOrBuffer);
        handle = handleOrBuffer;
        buffer = this.get(RESOURCE_KIND.BUFFER, handleOrBuffer) || null;
      } else if (handleOrBuffer && typeof handleOrBuffer === 'object') {
        buffer = handleOrBuffer;
      }
      if (!buffer) {
        return false;
      }

      const device = this._resolveDevice(options && options.device, 'writeBuffer');
      const dstOffset = Math.max(0, Math.floor(Number(bufferOffset) || 0));
      const upload = padBufferUploadData(data, dataOffset, size);
      if (upload.alignedByteLength === 0) {
        return true;
      }

      if (upload.padded) {
        this._bufferAlignPadCount += 1;
        this._emitSanitizeWarning(
          'buffer_write_pad',
          handle,
          `padded writeBuffer ${upload.originalByteLength} -> ${upload.alignedByteLength} bytes`
        );
      }

      device.queue.writeBuffer(buffer, dstOffset, upload.bytes);
      return true;
    }

    /**
     * Create, sanitize, and register a WebGPU texture under
     * `(RESOURCE_KIND.TEXTURE, handle)` (and its default view under
     * `(RESOURCE_KIND.TEXTURE_VIEW, handle)` when `texture.createView` exists).
     *
     * @param {number} handle
     * @param {object} descriptor
     * @param {{
     *   device?: object,
     *   useV1FormatTable?: boolean,
     *   isTexture2?: boolean,
     *   createView?: boolean,
     *   byteSize?: number,
     *   label?: string,
     *   tag?: unknown,
     * }} [options]
     * @returns {object} GPUTexture
     */
    createTexture(handle, descriptor, options = {}) {
      assertValidHandle(handle);
      const device = this._resolveDevice(options && options.device, 'createTexture');
      const sanitized = sanitizeTextureDescriptor(descriptor, device, {
        useV1FormatTable: Boolean(options && options.useV1FormatTable),
      });

      if (sanitized.bcFormatFallback) {
        this._bcFormatFallbackCount += 1;
        this._emitSanitizeWarning(
          'bc_format_fallback',
          handle,
          `${sanitized.originalFormat} -> ${sanitized.effectiveFormat}`
        );
      }
      if (sanitized.bcDimensionAligned) {
        this._bcDimensionAlignCount += 1;
        this._emitSanitizeWarning(
          'bc_dimension_align',
          handle,
          `aligned BC dimensions to ${sanitized.descriptor.size.width}x${sanitized.descriptor.size.height}`
        );
      }
      if (sanitized.mipClamped) {
        this._mipClampCount += 1;
        this._emitSanitizeWarning(
          'mip_clamp',
          handle,
          `clamped mipLevelCount to ${sanitized.descriptor.mipLevelCount}`
        );
      }
      if (sanitized.sampleClamped) {
        this._sampleClampCount += 1;
        this._emitSanitizeWarning(
          'sample_clamp',
          handle,
          `clamped sampleCount to ${sanitized.descriptor.sampleCount}`
        );
      }

      const texture = device.createTexture(sanitized.descriptor);
      const label =
        options && typeof options.label === 'string' && options.label.length > 0
          ? options.label
          : sanitized.descriptor.label || `texture:${handle}`;
      const tag = options && options.tag !== undefined ? options.tag : null;
      const estimatedBytes =
        options && options.byteSize !== undefined
          ? options.byteSize
          : estimateTextureBytes(
              sanitized.descriptor.size.width,
              sanitized.descriptor.size.height,
              sanitized.descriptor.size.depthOrArrayLayers,
              sanitized.descriptor.mipLevelCount,
              sanitized.effectiveFormat,
              sanitized.descriptor.dimension,
              sanitized.descriptor.sampleCount
            );

      // Registering on an existing handle automatically retires the prior texture safely.
      this.register(RESOURCE_KIND.TEXTURE, handle, texture, {
        byteSize: estimatedBytes,
        label,
        tag,
      });

      if (
        (!options || options.createView !== false) &&
        texture &&
        typeof texture.createView === 'function'
      ) {
        const textureView = texture.createView();
        this.register(RESOURCE_KIND.TEXTURE_VIEW, handle, textureView, {
          byteSize: 0,
          label: `textureView:${handle}`,
          tag,
        });
      } else {
        this.retire(RESOURCE_KIND.TEXTURE_VIEW, handle);
      }

      this._textureMeta.set(handle, {
        handle,
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
        isTexture2: Boolean(options && options.isTexture2),
        byteSize: estimatedBytes,
      });

      return texture;
    }

    /**
     * Create a texture using ABI v2 `CreateTexture2` metadata (opcode 35).
     * Accepts either:
     * - `createTexture2(handle, descriptor, options)` (structured descriptor with v2 format code/name)
     * - `createTexture2(dataView, payloadOffset, payloadBytes, options)` (raw 40-byte payload)
     *
     * @param {number | DataView} handleOrView
     * @param {object | number} [descriptorOrOffset]
     * @param {object | number} [optionsOrPayloadBytes]
     * @param {object} [maybeOptions]
     * @returns {object}
     */
    createTexture2(handleOrView, descriptorOrOffset, optionsOrPayloadBytes, maybeOptions) {
      if (handleOrView instanceof DataView) {
        const offset = typeof descriptorOrOffset === 'number' ? descriptorOrOffset : 0;
        const payloadBytes =
          typeof optionsOrPayloadBytes === 'number' ? optionsOrPayloadBytes : undefined;
        const opts =
          typeof optionsOrPayloadBytes === 'object' && optionsOrPayloadBytes !== null
            ? optionsOrPayloadBytes
            : maybeOptions || {};
        return this.createTexture2FromPacket(handleOrView, offset, payloadBytes, opts);
      }

      const opts =
        optionsOrPayloadBytes && typeof optionsOrPayloadBytes === 'object'
          ? optionsOrPayloadBytes
          : {};
      return this.createTexture(handleOrView, descriptorOrOffset, {
        ...opts,
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
     * @param {object} [options]
     * @returns {{ id: number, texture?: object, view?: object | null, meta?: object | null, error?: string }}
     */
    createTexture2FromPacket(view, payloadOffset = 0, payloadBytes, options = {}) {
      const decoded = decodeCreateTexture2(view, payloadOffset, payloadBytes);
      if (decoded.error) {
        return { id: decoded.id, error: decoded.error };
      }

      try {
        const texture = this.createTexture(decoded.id, decoded.descriptor, {
          ...(options || {}),
          useV1FormatTable: false,
          isTexture2: true,
        });
        return {
          id: decoded.id,
          texture,
          view: this.get(RESOURCE_KIND.TEXTURE_VIEW, decoded.id) || null,
          meta: this.getTextureMeta(decoded.id) || null,
        };
      } catch (err) {
        return {
          id: decoded.id,
          error: `CreateTexture2 ${decoded.id}: ${String(err && err.message ? err.message : err)}`,
        };
      }
    }

    /**
     * Look up sanitized creation metadata for a buffer created via `createBuffer`.
     *
     * @param {number} handle
     * @returns {object | undefined}
     */
    getBufferMeta(handle) {
      assertValidHandle(handle);
      const meta = this._bufferMeta.get(handle);
      return meta ? { ...meta } : undefined;
    }

    /**
     * Look up sanitized creation metadata for a texture created via `createTexture` / `createTexture2`.
     *
     * @param {number} handle
     * @returns {object | undefined}
     */
    getTextureMeta(handle) {
      assertValidHandle(handle);
      const meta = this._textureMeta.get(handle);
      return meta ? { ...meta, viewFormats: meta.viewFormats.slice() } : undefined;
    }

    /**
     * Look up an active resource by `(kind, handle)`.
     *
     * @param {string} kind
     * @param {number} handle
     * @returns {object | undefined}
     */
    get(kind, handle) {
      assertValidKind(kind);
      assertValidHandle(handle);
      const entry = this._tables.get(kind).get(handle);
      return entry ? entry.resource : undefined;
    }

    /**
     * Look up an active resource entry snapshot (including metadata) by `(kind, handle)`.
     *
     * @param {string} kind
     * @param {number} handle
     * @returns {{ kind: string, handle: number, resource: object, byteSize: number, label: string, tag: unknown, registeredAtFrame: number } | undefined}
     */
    getEntry(kind, handle) {
      assertValidKind(kind);
      assertValidHandle(handle);
      const entry = this._tables.get(kind).get(handle);
      return entry ? snapshotEntry(entry) : undefined;
    }

    /**
     * Return whether an active resource exists for `(kind, handle)`.
     *
     * @param {string} kind
     * @param {number} handle
     * @returns {boolean}
     */
    has(kind, handle) {
      assertValidKind(kind);
      assertValidHandle(handle);
      return this._tables.get(kind).has(handle);
    }

    /**
     * List active resource entry snapshots, optionally filtered to a single `kind`.
     *
     * @param {string} [kind]
     * @returns {Array<{ kind: string, handle: number, resource: object, byteSize: number, label: string, tag: unknown, registeredAtFrame: number }>}
     */
    list(kind) {
      if (kind !== undefined) {
        assertValidKind(kind);
        const table = this._tables.get(kind);
        const result = [];
        for (const entry of table.values()) {
          result.push(snapshotEntry(entry));
        }
        return result;
      }

      const result = [];
      for (let i = 0; i < RESOURCE_KINDS.length; i += 1) {
        const table = this._tables.get(RESOURCE_KINDS[i]);
        for (const entry of table.values()) {
          result.push(snapshotEntry(entry));
        }
      }
      return result;
    }

    /**
     * Remove `(kind, handle)` from the active table immediately and enqueue it
     * for deferred destruction after `framesInFlight` frames or once its frame's
     * GPU fence resolves.
     *
     * @param {string} kind
     * @param {number} handle
     * @returns {boolean} True if an active resource was retired, false otherwise
     */
    retire(kind, handle) {
      assertValidKind(kind);
      assertValidHandle(handle);

      const table = this._tables.get(kind);
      const entry = table.get(handle);
      if (!entry) {
        return false;
      }

      table.delete(handle);
      if (kind === RESOURCE_KIND.BUFFER) {
        this._bufferMeta.delete(handle);
      } else if (kind === RESOURCE_KIND.TEXTURE) {
        this._textureMeta.delete(handle);
      }

      this._activeBytes = Math.max(0, this._activeBytes - entry.byteSize);
      this._pendingRetireBytes += entry.byteSize;

      entry.retiredAtFrame = this._currentFrame;
      entry.sealed = false;
      entry.hasFence = false;
      entry.fenceDone = false;
      entry.fenceFailed = false;

      this._pendingRetire.push(entry);
      this._totalRetired += 1;
      return true;
    }

    /**
     * Convenience helper that retires a numeric `handle` from every resource kind
     * where it is currently registered (mirrors opcode DESTROY_RESOURCE).
     *
     * @param {number} handle
     * @returns {number} Number of active resources retired
     */
    retireHandle(handle) {
      assertValidHandle(handle);
      let count = 0;
      for (let i = 0; i < RESOURCE_KINDS.length; i += 1) {
        if (this.retire(RESOURCE_KINDS[i], handle)) {
          count += 1;
        }
      }
      return count;
    }

    /**
     * Advance the frame epoch and collect any deferred resources whose frame
     * latency or GPU fence has completed.
     *
     * @param {number} [frameIndex]
     * @returns {{ frameIndex: number, destroyedCount: number, freedBytes: number, remainingCount: number, pendingRetireBytes: number }}
     */
    beginFrame(frameIndex) {
      // Seal any unsealed retirements from the prior frame as epoch-based so a
      // future endFrame(fence) in this frame does not retroactively attach to them.
      this._sealUnsealedRetirements();

      let nextFrame;
      if (frameIndex === undefined) {
        nextFrame = this._currentFrame + 1;
      } else {
        assertNonNegativeInteger(frameIndex, 'frameIndex');
        if (frameIndex < this._currentFrame) {
          throw new RangeError(
            `frameIndex (${frameIndex}) cannot move backwards from currentFrame (${this._currentFrame})`
          );
        }
        nextFrame = frameIndex;
      }

      this._currentFrame = nextFrame;
      const collected = this.collect(false);

      return {
        frameIndex: this._currentFrame,
        destroyedCount: collected.destroyedCount,
        freedBytes: collected.freedBytes,
        remainingCount: collected.remainingCount,
        pendingRetireBytes: collected.pendingRetireBytes,
      };
    }

    /**
     * Seal all resources retired during the current frame and optionally bind
     * them to a GPU work-done fence promise (e.g. `device.queue.onSubmittedWorkDone()`).
     *
     * When a fence promise is supplied:
     *  - Resolving the fence allows early collection on the next `collect()` /
     *    `beginFrame()`, even before `framesInFlight` frames elapse.
     *  - While the fence remains pending, collection of those entries is gated
     *    until the fence resolves (unless `collect(true)` or `destroyAll()` is called).
     *  - If the fence rejects (e.g. device lost), those entries fall back to
     *    frame-epoch expiration instead of leaking or causing unhandled rejections.
     *
     * @param {Promise<unknown> | null} [fencePromise]
     * @returns {{ frameIndex: number, sealedCount: number, pendingCount: number, pendingRetireBytes: number }}
     */
    endFrame(fencePromise) {
      if (
        fencePromise !== undefined &&
        fencePromise !== null &&
        typeof fencePromise.then !== 'function'
      ) {
        throw new TypeError('endFrame expects fencePromise to be a Promise/thenable, null, or undefined');
      }

      const batch = this._sealUnsealedRetirements();

      if (fencePromise && typeof fencePromise.then === 'function') {
        const generation = this._generation;
        for (let i = 0; i < batch.length; i += 1) {
          batch[i].hasFence = true;
          batch[i].fenceDone = false;
          batch[i].fenceFailed = false;
        }

        const tracked = Promise.resolve(fencePromise).then(
          () => {
            if (generation !== this._generation) {
              return;
            }
            for (let i = 0; i < batch.length; i += 1) {
              if (!batch[i].destroyed) {
                batch[i].fenceDone = true;
              }
            }
          },
          () => {
            if (generation !== this._generation) {
              return;
            }
            for (let i = 0; i < batch.length; i += 1) {
              if (!batch[i].destroyed) {
                batch[i].fenceFailed = true;
              }
            }
          }
        );

        this._pendingFences.add(tracked);
        tracked.finally(() => {
          this._pendingFences.delete(tracked);
        });
      }

      return {
        frameIndex: this._currentFrame,
        sealedCount: batch.length,
        pendingCount: this._pendingRetire.length,
        pendingRetireBytes: this._pendingRetireBytes,
      };
    }

    /**
     * Destroy eligible retired resources and reclaim their pending-retirement VRAM accounting.
     *
     * @param {boolean} [force=false] When true, destroys all pending retired resources immediately regardless of frame epoch or pending fences.
     * @returns {{ destroyedCount: number, freedBytes: number, remainingCount: number, pendingRetireBytes: number }}
     */
    collect(force = false) {
      const forceAll = Boolean(force);
      let destroyedCount = 0;
      let freedBytes = 0;
      const remaining = [];

      for (let i = 0; i < this._pendingRetire.length; i += 1) {
        const entry = this._pendingRetire[i];
        const epochElapsed = this._currentFrame - entry.retiredAtFrame >= this.framesInFlight;

        let eligible;
        if (forceAll) {
          eligible = true;
        } else if (entry.hasFence) {
          eligible = entry.fenceDone || (entry.fenceFailed && epochElapsed);
        } else {
          eligible = epochElapsed;
        }

        if (eligible) {
          this._pendingRetireBytes = Math.max(0, this._pendingRetireBytes - entry.byteSize);
          freedBytes += entry.byteSize;
          if (this._destroyEntry(entry)) {
            destroyedCount += 1;
          }
        } else {
          remaining.push(entry);
        }
      }

      this._pendingRetire = remaining;
      this._checkBudget(null);

      return {
        destroyedCount,
        freedBytes,
        remainingCount: this._pendingRetire.length,
        pendingRetireBytes: this._pendingRetireBytes,
      };
    }

    /**
     * Alias for `collect(force)`.
     *
     * @param {boolean} [force=false]
     */
    flush(force = false) {
      return this.collect(force);
    }

    /**
     * Return a sanitized snapshot of per-kind counts, VRAM byte accounting,
     * lifecycle stats, and Frame-1 hardening counters.
     */
    getStats() {
      const countsByKind = {};
      let activeCount = 0;
      for (let i = 0; i < RESOURCE_KINDS.length; i += 1) {
        const kind = RESOURCE_KINDS[i];
        const count = this._tables.get(kind).size;
        countsByKind[kind] = count;
        activeCount += count;
      }

      const pendingCountsByKind = {};
      for (let i = 0; i < RESOURCE_KINDS.length; i += 1) {
        pendingCountsByKind[RESOURCE_KINDS[i]] = 0;
      }
      for (let i = 0; i < this._pendingRetire.length; i += 1) {
        pendingCountsByKind[this._pendingRetire[i].kind] += 1;
      }

      return {
        countsByKind,
        pendingCountsByKind,
        activeCount,
        pendingRetireCount: this._pendingRetire.length,
        activeBytes: this._activeBytes,
        pendingRetireBytes: this._pendingRetireBytes,
        totalBytes: this._activeBytes + this._pendingRetireBytes,
        peakBytes: this._peakBytes,
        softBudgetBytes: this.softBudgetBytes,
        budgetExceeded: this._budgetExceeded,
        budgetWarningCount: this._budgetWarningCount,
        totalRegistered: this._totalRegistered,
        totalRetired: this._totalRetired,
        totalDestroyed: this._totalDestroyed,
        destroyErrors: this._destroyErrors,
        currentFrame: this._currentFrame,
        framesInFlight: this.framesInFlight,
        bcFormatFallbackCount: this._bcFormatFallbackCount,
        bcDimensionAlignCount: this._bcDimensionAlignCount,
        bufferAlignPadCount: this._bufferAlignPadCount,
        mipClampCount: this._mipClampCount,
        sampleClampCount: this._sampleClampCount,
      };
    }

    /**
     * Idempotently destroy all active and pending-retirement resources
     * (e.g. for `device.lost` recovery or session shutdown).
     *
     * @returns {{ destroyedCount: number, freedBytes: number }}
     */
    destroyAll() {
      this._generation += 1;
      this._pendingFences.clear();

      const toDestroy = [];
      for (let i = 0; i < RESOURCE_KINDS.length; i += 1) {
        const table = this._tables.get(RESOURCE_KINDS[i]);
        for (const entry of table.values()) {
          toDestroy.push(entry);
        }
        table.clear();
      }
      this._bufferMeta.clear();
      this._textureMeta.clear();

      for (let i = 0; i < this._pendingRetire.length; i += 1) {
        toDestroy.push(this._pendingRetire[i]);
      }
      this._pendingRetire = [];

      this._activeBytes = 0;
      this._pendingRetireBytes = 0;
      this._budgetExceeded = false;

      let destroyedCount = 0;
      let freedBytes = 0;

      for (let i = 0; i < toDestroy.length; i += 1) {
        const entry = toDestroy[i];
        freedBytes += entry.byteSize;
        if (this._destroyEntry(entry)) {
          destroyedCount += 1;
        }
      }

      return {
        destroyedCount,
        freedBytes,
      };
    }
  }

  const exportsObj = Object.freeze({
    DEFAULT_FRAMES_IN_FLIGHT,
    DEFAULT_SOFT_BUDGET_BYTES,
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
    RESOURCE_KIND,
    RESOURCE_KINDS,
    isBcFormat,
    deviceSupportsBc,
    computeMaxMipLevels,
    clampMipLevelCount,
    clampSampleCount,
    alignBcDimension,
    alignBufferSize,
    padBufferUploadData,
    estimateTextureBytes,
    decodeCreateTexture2,
    sanitizeTextureDescriptor,
    GpuResourceRegistry,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6ResourceRegistry = exportsObj;
  } else if (root) {
    root.PlayGta6ResourceRegistry = exportsObj;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      DEFAULT_FRAMES_IN_FLIGHT,
      DEFAULT_SOFT_BUDGET_BYTES,
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
      RESOURCE_KIND,
      RESOURCE_KINDS,
      isBcFormat,
      deviceSupportsBc,
      computeMaxMipLevels,
      clampMipLevelCount,
      clampSampleCount,
      alignBcDimension,
      alignBufferSize,
      padBufferUploadData,
      estimateTextureBytes,
      decodeCreateTexture2,
      sanitizeTextureDescriptor,
      GpuResourceRegistry,
    };
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
