// GPU packet ABI v1/v2 texture handling for WgpuCommandProcessor:
// - ABI v2 CreateTexture2 (opcode 35, 40-byte payload, docs/gpu_commands.md "CreateTexture2")
//   with safe mipLevels and sampleCount clamping alongside ABI v1 CreateTexture (opcode 12),
//   automatically attaching compatible sRGB/linear sibling `viewFormats` (`rgba8unorm`,
//   `bgra8unorm`, `bc1..bc7`).
// - Zero-copy direct `queue.writeTexture` fast path when source row pitch is already 256-byte
//   aligned (e.g., 4K Bink2 `3840x2160` `r8unorm` Y plane).
// - WebGPU 256-byte bytesPerRow alignment & row-pitch repacking helpers (`alignBytesPerRow256`,
//   `repackRowsTo256`) backed by a bounded persistent per-handler scratch `Uint8Array` (e.g.,
//   reused across 4K Bink2 `1920x1080` `r8unorm` Cr/Cb planes where `1920 -> 2048`).
// - Safe `SharedArrayBuffer`-backed view copying before `queue.writeTexture` /
//   `copyBufferToTexture`.
// - Sanitized telemetry counters (`createTexture2Count`, `rowPitchRepackCount`, `sabCopyCount`,
//   `uploadCount`, `uploadBytes`, `zeroCopyUploads`, `scratchRepackUploads`,
//   `directZeroCopyUploads`, `repackedUploads`, `scratchBufferReuses`, `r8unormUploads`,
//   `bytesUploaded`, `srgbSiblingViewFormatsAdded`) exposed via `getStats()`, `getUploadStats()`,
//   and the `uploadStats` getter.
//
// Install with installTexturePackets(processor) after the v2 handler (render_state_packets.js):
// it chains in front of processor.extension and forwards every other opcode to it.

(function (globalScope) {
  'use strict';

  const CREATE_TEXTURE = 12;
  const CREATE_TEXTURE_BYTES = 32;
  const WRITE_TEXTURE = 13;
  const WRITE_TEXTURE_HEADER_BYTES = 48;
  const CREATE_TEXTURE2 = 35;
  const CREATE_TEXTURE2_BYTES = 40;
  const MAX_VIEW_FORMATS = 8;
  const WEBGPU_BYTES_PER_ROW_ALIGNMENT = 256;
  const DEFAULT_MAX_SCRATCH_BYTES = 16 * 1024 * 1024; // 16 MiB cap for persistent row-repack / SAB scratch
  const GPU_BUFFER_USAGE_COPY_SRC_DST = 0x0004 | 0x0008;

  // Compatible WebGPU sRGB <-> linear viewFormat sibling pairs.
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

  // v1 GpuTextureFormat enum names (runtime/abi/playgta6_abi.h).
  const V1_TEXTURE_FORMAT_NAMES = Object.freeze([
    'rgba8unorm',
    'bgra8unorm',
    'rgba16float',
    'depth24plus',
    'depth32float',
    'bc7-rgba-unorm',
  ]);

  // v2 texture format codes ("WebGPU enum codes" in docs/gpu_commands.md).
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

  const TEXTURE_DIMENSION_NAMES = Object.freeze(['1d', '2d', '3d']);
  const BC_SLICED_3D_FEATURE = 'texture-compression-bc-sliced-3d';

  const HOOK_ORIG_CREATE_TEXTURE = '__playgta6OrigHandleCreateTexture';
  const HOOK_ORIG_WRITE_TEXTURE = '__playgta6OrigHandleWriteTexture';
  const HOOK_ORIG_DESTROY = '__playgta6OrigTexturePacketsDestroy';

  function isBcCompressedFormat(format) {
    return typeof format === 'string' && format.startsWith('bc');
  }

  /**
   * Return the WebGPU-compatible sRGB <-> linear sibling format for `format`
   * (`rgba8unorm`, `bgra8unorm`, `bc1/bc2/bc3/bc7` unorm <-> `-srgb`), or `null`
   * if `format` has no sRGB sibling.
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
   * Build a deduplicated, WebGPU-valid `viewFormats` array for a texture of `format`:
   * - Filters any caller-supplied `requestedViewFormats` down to formats that are
   *   view-format-compatible with `format` (`format` itself or its sRGB/linear sibling).
   * - Automatically appends the compatible sRGB/linear sibling when one exists so
   *   sRGB/linear texture views never fail WebGPU validation.
   *
   * @param {string} format
   * @param {string[] | null | undefined} [requestedViewFormats]
   * @returns {{ viewFormats: string[] | undefined, siblingAdded: boolean }}
   */
  function resolveCompatibleViewFormats(format, requestedViewFormats) {
    if (typeof format !== 'string' || !format) {
      return { viewFormats: undefined, siblingAdded: false };
    }
    const baseFormat = format.trim().toLowerCase();
    const sibling = getSrgbSiblingFormat(baseFormat);
    const out = [];
    const seen = new Set();

    if (Array.isArray(requestedViewFormats)) {
      for (let i = 0; i < requestedViewFormats.length; i += 1) {
        const raw = requestedViewFormats[i];
        if (typeof raw !== 'string') {
          continue;
        }
        const vf = raw.trim().toLowerCase();
        if (!vf || seen.has(vf)) {
          continue;
        }
        if (vf === baseFormat || (sibling !== null && vf === sibling)) {
          seen.add(vf);
          out.push(vf);
        }
      }
    }

    let siblingAdded = false;
    if (sibling !== null && !seen.has(sibling)) {
      seen.add(sibling);
      out.push(sibling);
      siblingAdded = true;
    }

    return {
      viewFormats: out.length > 0 ? out : undefined,
      siblingAdded,
    };
  }

  function deviceSupportsBcSliced3d(device) {
    const features = device && device.features;
    if (!features) {
      return false;
    }
    if (typeof features.has === 'function') {
      return Boolean(features.has(BC_SLICED_3D_FEATURE));
    }
    if (Array.isArray(features)) {
      return features.includes(BC_SLICED_3D_FEATURE);
    }
    if (typeof features[Symbol.iterator] === 'function') {
      for (const item of features) {
        if (item === BC_SLICED_3D_FEATURE) {
          return true;
        }
      }
      return false;
    }
    if (typeof features === 'object') {
      return Boolean(features[BC_SLICED_3D_FEATURE]);
    }
    return false;
  }

  function toNonNegativeInt(value) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
      return 0;
    }
    return Math.floor(value);
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

  /**
   * Returns `true` if `input` is a `SharedArrayBuffer` or an `ArrayBufferView`
   * backed by a `SharedArrayBuffer` (cross-realm safe).
   *
   * @param {unknown} input
   * @returns {boolean}
   */
  function isSharedArrayBufferView(input) {
    if (!input || typeof input !== 'object') {
      return false;
    }
    if (isSharedBuffer(input)) {
      return true;
    }
    if (ArrayBuffer.isView(input)) {
      return isSharedBuffer(input.buffer);
    }
    return false;
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

  /**
   * Safely converts `input` to a `Uint8Array` backed by a non-shared `ArrayBuffer`.
   * If `input` is backed by a `SharedArrayBuffer`, copies into an optional reusable
   * non-shared `scratchBuffer` (when large enough) or allocates a standalone copy so
   * WebGPU `queue.writeTexture` / `queue.writeBuffer` / `copyBufferToTexture` calls
   * never reject shared buffers or race with concurrent guest writes.
   *
   * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} input
   * @param {Uint8Array | null} [scratchBuffer=null]
   * @returns {{ data: Uint8Array, copiedFromSab: boolean, reusedScratch: boolean }}
   */
  function copyFromSharedIfNeeded(input, scratchBuffer = null) {
    const src = toUint8Array(input, 'copyFromSharedIfNeeded');
    if (isSharedBuffer(src.buffer)) {
      const canReuseScratch =
        scratchBuffer instanceof Uint8Array &&
        !isSharedBuffer(scratchBuffer.buffer) &&
        scratchBuffer.byteLength >= src.byteLength;
      const copy = canReuseScratch
        ? scratchBuffer.subarray(0, src.byteLength)
        : new Uint8Array(src.byteLength);
      copy.set(src, 0);
      return { data: copy, copiedFromSab: true, reusedScratch: canReuseScratch };
    }
    return { data: src, copiedFromSab: false, reusedScratch: false };
  }

  /**
   * Align a texture row stride (`bytesPerRow`) up to WebGPU's required 256-byte
   * `copyBufferToTexture` boundary without 32-bit bitwise truncation.
   *
   * @param {number} bytesPerRow
   * @returns {number}
   */
  function alignBytesPerRow256(bytesPerRow) {
    if (!Number.isInteger(bytesPerRow) || bytesPerRow < 0) {
      throw new RangeError(
        `alignBytesPerRow256 expects a non-negative integer, got ${bytesPerRow}`
      );
    }
    if (bytesPerRow === 0) {
      return 0;
    }
    const remainder = bytesPerRow % WEBGPU_BYTES_PER_ROW_ALIGNMENT;
    return remainder === 0
      ? bytesPerRow
      : bytesPerRow + (WEBGPU_BYTES_PER_ROW_ALIGNMENT - remainder);
  }

  /**
   * Repack texture rows into a 256-byte-aligned `Uint8Array` suitable for WebGPU
   * `copyBufferToTexture`, zero-padding each row's trailing alignment bytes and
   * safely copying out of `SharedArrayBuffer`-backed views.
   *
   * When `bytesPerRow` is already a multiple of 256 and `srcBytes` is not backed
   * by a `SharedArrayBuffer`, returns a zero-copy view with `repacked: false`.
   *
   * When an optional non-shared `scratchBuffer` (`Uint8Array`) of sufficient
   * `byteLength` is provided and copying/repacking is required, writes into
   * `scratchBuffer.subarray(0, totalBytes)` instead of allocating a new buffer
   * and zeroes each row's trailing padding bytes.
   *
   * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} srcBytes
   * @param {number} bytesPerRow
   * @param {number} rowCount
   * @param {number} [depthOrLayers=1]
   * @param {Uint8Array | null} [scratchBuffer=null]
   * @returns {{
   *   data: Uint8Array,
   *   bytesPerRow: number,
   *   originalBytesPerRow: number,
   *   rowCount: number,
   *   repacked: boolean,
   *   copiedFromSab: boolean,
   *   reusedScratch: boolean,
   * }}
   */
  function repackRowsTo256(
    srcBytes,
    bytesPerRow,
    rowCount,
    depthOrLayers = 1,
    scratchBuffer = null
  ) {
    const src = toUint8Array(srcBytes, 'repackRowsTo256');
    if (!Number.isInteger(bytesPerRow) || bytesPerRow < 1) {
      throw new RangeError(`repackRowsTo256: bytesPerRow must be a positive integer, got ${bytesPerRow}`);
    }
    if (!Number.isInteger(rowCount) || rowCount < 0) {
      throw new RangeError(`repackRowsTo256: rowCount must be a non-negative integer, got ${rowCount}`);
    }
    if (!Number.isInteger(depthOrLayers) || depthOrLayers < 1) {
      throw new RangeError(
        `repackRowsTo256: depthOrLayers must be a positive integer, got ${depthOrLayers}`
      );
    }

    const totalRows = rowCount * depthOrLayers;
    const alignedBytesPerRow = alignBytesPerRow256(bytesPerRow);
    const fromSab = isSharedBuffer(src.buffer);

    if (totalRows === 0) {
      return {
        data: new Uint8Array(0),
        bytesPerRow: alignedBytesPerRow,
        originalBytesPerRow: bytesPerRow,
        rowCount: 0,
        repacked: false,
        copiedFromSab: false,
        reusedScratch: false,
      };
    }

    const requiredSrcBytes = bytesPerRow * totalRows;
    if (src.byteLength < requiredSrcBytes) {
      throw new RangeError(
        `repackRowsTo256: srcBytes too small (${src.byteLength} bytes; need at least ${requiredSrcBytes} bytes for ${totalRows} rows of ${bytesPerRow} bytes)`
      );
    }

    if (alignedBytesPerRow === bytesPerRow) {
      const slice = src.byteLength === requiredSrcBytes ? src : src.subarray(0, requiredSrcBytes);
      if (fromSab) {
        const canReuseScratch =
          scratchBuffer instanceof Uint8Array &&
          !isSharedBuffer(scratchBuffer.buffer) &&
          scratchBuffer.byteLength >= requiredSrcBytes;
        const copy = canReuseScratch
          ? scratchBuffer.subarray(0, requiredSrcBytes)
          : new Uint8Array(requiredSrcBytes);
        copy.set(slice, 0);
        return {
          data: copy,
          bytesPerRow: alignedBytesPerRow,
          originalBytesPerRow: bytesPerRow,
          rowCount: totalRows,
          repacked: false,
          copiedFromSab: true,
          reusedScratch: canReuseScratch,
        };
      }
      return {
        data: slice,
        bytesPerRow: alignedBytesPerRow,
        originalBytesPerRow: bytesPerRow,
        rowCount: totalRows,
        repacked: false,
        copiedFromSab: false,
        reusedScratch: false,
      };
    }

    const totalDstBytes = alignedBytesPerRow * totalRows;
    const canReuseScratch =
      scratchBuffer instanceof Uint8Array &&
      !isSharedBuffer(scratchBuffer.buffer) &&
      scratchBuffer.byteLength >= totalDstBytes;
    const dst = canReuseScratch
      ? scratchBuffer.subarray(0, totalDstBytes)
      : new Uint8Array(totalDstBytes);

    for (let row = 0; row < totalRows; row += 1) {
      const srcOffset = row * bytesPerRow;
      const dstOffset = row * alignedBytesPerRow;
      dst.set(src.subarray(srcOffset, srcOffset + bytesPerRow), dstOffset);
      if (canReuseScratch) {
        dst.fill(0, dstOffset + bytesPerRow, dstOffset + alignedBytesPerRow);
      }
    }

    return {
      data: dst,
      bytesPerRow: alignedBytesPerRow,
      originalBytesPerRow: bytesPerRow,
      rowCount: totalRows,
      repacked: true,
      copiedFromSab: fromSab,
      reusedScratch: canReuseScratch,
    };
  }

  /**
   * Compute the maximum valid WebGPU mip level count for texture dimensions.
   * Per WebGPU spec:
   * - `'1d'` textures only allow 1 mip level.
   * - `'2d'` textures allow `floor(log2(max(width, height))) + 1`.
   * - `'3d'` textures allow `floor(log2(max(width, height, depth))) + 1`.
   *
   * @param {number} width
   * @param {number} height
   * @param {number} [depthOrArrayLayers=1]
   * @param {string} [dimension='2d']
   * @returns {number}
   */
  function computeMaxMipLevels(width, height, depthOrArrayLayers = 1, dimension = '2d') {
    if (dimension === '1d') {
      return 1;
    }
    const w = Math.max(1, Math.floor(Number(width) || 1));
    const h = Math.max(1, Math.floor(Number(height) || 1));
    const d = Math.max(1, Math.floor(Number(depthOrArrayLayers) || 1));
    const maxDim = dimension === '3d' ? Math.max(w, h, d) : Math.max(w, h);
    return Math.floor(Math.log2(maxDim)) + 1;
  }

  /**
   * Clamp a requested `mipLevelCount` safely into `[1, maxMipLevels]`.
   *
   * @param {number} requestedMipLevels
   * @param {number} width
   * @param {number} height
   * @param {number} [depthOrArrayLayers=1]
   * @param {string} [dimension='2d']
   * @returns {number}
   */
  function clampMipLevels(
    requestedMipLevels,
    width,
    height,
    depthOrArrayLayers = 1,
    dimension = '2d'
  ) {
    const maxMips = computeMaxMipLevels(width, height, depthOrArrayLayers, dimension);
    const n = Number(requestedMipLevels);
    if (!Number.isFinite(n) || n < 1) {
      return 1;
    }
    return Math.min(maxMips, Math.max(1, Math.floor(n)));
  }

  /**
   * Clamp a requested `sampleCount` safely to WebGPU's legal `{1, 4}` set.
   * Forces `1` whenever multisampling is disallowed by WebGPU texture geometry
   * (`dimension !== '2d'`, `mipLevelCount > 1`, or `depthOrArrayLayers > 1`).
   *
   * @param {number} requestedSampleCount
   * @param {{ mipLevelCount?: number, dimension?: string, depthOrArrayLayers?: number }} [context]
   * @returns {number}
   */
  function clampSampleCount(requestedSampleCount, context = {}) {
    const n = Number(requestedSampleCount);
    if (!Number.isFinite(n) || n <= 1) {
      return 1;
    }
    const mipLevelCount = context.mipLevelCount !== undefined ? context.mipLevelCount : 1;
    const dimension = context.dimension || '2d';
    const depthOrArrayLayers =
      context.depthOrArrayLayers !== undefined ? context.depthOrArrayLayers : 1;

    const geometryAllowsMsaa =
      dimension === '2d' && mipLevelCount <= 1 && depthOrArrayLayers === 1;

    if (n >= 4 && geometryAllowsMsaa) {
      return 4;
    }
    return 1;
  }

  /**
   * Decodes a v1 `CreateTexture` (opcode 12, 32-byte `GpuCreateTexturePayload`)
   * into `{ id, descriptor }` with safely clamped dimensions and `mipLevelCount`,
   * or `{ id, error }` when truncated.
   *
   * @param {DataView} view
   * @param {number} [offset=0]
   * @param {number} [payloadBytes]
   */
  function decodeCreateTexture(view, offset = 0, payloadBytes) {
    const available = payloadBytes !== undefined ? payloadBytes : view.byteLength - offset;
    if (offset < 0 || available < CREATE_TEXTURE_BYTES || offset + CREATE_TEXTURE_BYTES > view.byteLength) {
      return {
        id: 0,
        error: `CreateTexture: payload ${available} < ${CREATE_TEXTURE_BYTES} bytes`,
      };
    }

    const id = view.getUint32(offset, true);
    const rawWidth = view.getUint32(offset + 4, true);
    const rawHeight = view.getUint32(offset + 8, true);
    const rawDepth = view.getUint32(offset + 12, true);
    const rawMipLevels = view.getUint32(offset + 16, true);
    const formatCode = view.getUint32(offset + 20, true);
    const usage = view.getUint32(offset + 24, true);
    const dimensionCode = view.getUint32(offset + 28, true);

    const format = V1_TEXTURE_FORMAT_NAMES[formatCode] || 'rgba8unorm';
    const dimension = TEXTURE_DIMENSION_NAMES[dimensionCode] || '2d';
    const width = Math.max(1, rawWidth);
    const height = dimension === '1d' ? 1 : Math.max(1, rawHeight);
    const depthOrArrayLayers = dimension === '1d' ? 1 : Math.max(1, rawDepth);
    const mipLevelCount = clampMipLevels(
      rawMipLevels,
      width,
      height,
      depthOrArrayLayers,
      dimension
    );

    const descriptor = {
      size: { width, height, depthOrArrayLayers },
      mipLevelCount,
      format,
      usage,
      dimension,
    };
    const sibling = getSrgbSiblingFormat(format);
    if (sibling !== null) {
      descriptor.viewFormats = [sibling];
    }

    return {
      id,
      descriptor,
    };
  }

  /**
   * Decodes an ABI v2 `CreateTexture2` payload (opcode 35, 40-byte
   * `GpuCreateTexture2Payload`) into `{ id, descriptor }` (`GPUTextureDescriptor`)
   * with safely clamped `mipLevelCount` and `sampleCount`, or `{ id, error }` when
   * the payload is truncated or contains invalid codes.
   *
   * @param {DataView} view
   * @param {number} [offset=0]
   * @param {number} [payloadBytes]
   */
  function decodeCreateTexture2(view, offset = 0, payloadBytes) {
    const available = payloadBytes !== undefined ? payloadBytes : view.byteLength - offset;
    if (
      offset < 0 ||
      available < CREATE_TEXTURE2_BYTES ||
      offset + CREATE_TEXTURE2_BYTES > view.byteLength
    ) {
      return {
        id: 0,
        error: `CreateTexture2: payload ${available} < ${CREATE_TEXTURE2_BYTES} bytes`,
      };
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

    const format = TEXTURE_FORMAT_NAMES[formatCode];
    if (formatCode === 0 || format === undefined) {
      return { id, error: `CreateTexture2 ${id}: invalid texture format code ${formatCode}` };
    }
    const dimension = TEXTURE_DIMENSION_NAMES[dimensionCode];
    if (dimension === undefined) {
      return { id, error: `CreateTexture2 ${id}: invalid dimension ${dimensionCode}` };
    }
    if (viewFormatCount > MAX_VIEW_FORMATS) {
      return {
        id,
        error: `CreateTexture2 ${id}: view_format_count ${viewFormatCount} > ${MAX_VIEW_FORMATS}`,
      };
    }

    const viewFormats = [];
    for (let i = 0; i < viewFormatCount; i += 1) {
      const code = view.getUint8(offset + 32 + i);
      const name = TEXTURE_FORMAT_NAMES[code];
      if (code === 0 || name === undefined) {
        return { id, error: `CreateTexture2 ${id}: invalid view format code ${code}` };
      }
      viewFormats.push(name);
    }

    const width = Math.max(1, rawWidth);
    const height = dimension === '1d' ? 1 : Math.max(1, rawHeight);
    const depthOrArrayLayers = dimension === '1d' ? 1 : Math.max(1, rawDepth);
    const mipLevelCount = clampMipLevels(
      rawMipLevels,
      width,
      height,
      depthOrArrayLayers,
      dimension
    );
    const sampleCount = clampSampleCount(rawSampleCount, {
      mipLevelCount,
      dimension,
      depthOrArrayLayers,
    });

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

  function normalizeCopySize(copySize) {
    if (Array.isArray(copySize)) {
      return {
        width: Math.max(1, Math.floor(Number(copySize[0]) || 1)),
        height: Math.max(1, Math.floor(Number(copySize[1] !== undefined ? copySize[1] : 1) || 1)),
        depthOrArrayLayers: Math.max(
          1,
          Math.floor(Number(copySize[2] !== undefined ? copySize[2] : 1) || 1)
        ),
      };
    }
    if (copySize && typeof copySize === 'object') {
      return {
        width: Math.max(1, Math.floor(Number(copySize.width) || 1)),
        height: Math.max(1, Math.floor(Number(copySize.height !== undefined ? copySize.height : 1) || 1)),
        depthOrArrayLayers: Math.max(
          1,
          Math.floor(
            Number(
              copySize.depthOrArrayLayers !== undefined
                ? copySize.depthOrArrayLayers
                : copySize.depthOrLayers !== undefined
                  ? copySize.depthOrLayers
                  : 1
            ) || 1
          )
        ),
      };
    }
    throw new TypeError('copySize must be a GPUExtent3D object or [width, height, depth] array');
  }

  class TexturePackets {
    constructor(processor, next, options = {}) {
      this.processor = processor;
      this.next = next || null;
      this._bc3dFallbackTextures = new Set();
      this._bc3dFallbackTextureObjs = new WeakSet();
      this._textureFormats = new Map();
      this._textureObjFormats = new WeakMap();
      const rawMaxScratch =
        options && typeof options === 'object' ? options.maxScratchBytes : undefined;
      this._maxScratchBytes =
        Number.isInteger(rawMaxScratch) && rawMaxScratch >= 0
          ? rawMaxScratch
          : DEFAULT_MAX_SCRATCH_BYTES;
      this._scratchBuffer = null;
      this.stats = this._createEmptyStats();
      this.lastError = null;
    }

    _createEmptyStats() {
      return {
        created: 0,
        errors: 0,
        createTexture2Count: 0,
        rowPitchRepackCount: 0,
        sabCopyCount: 0,
        bc3dFallbacks: 0,
        bc3dFallbackUploadsSkipped: 0,
        uploadCount: 0,
        uploadBytes: 0,
        zeroCopyUploads: 0,
        scratchRepackUploads: 0,
        directZeroCopyUploads: 0,
        repackedUploads: 0,
        scratchBufferReuses: 0,
        scratchBufferAllocations: 0,
        r8unormUploads: 0,
        bytesUploaded: 0,
        srgbSiblingViewFormatsAdded: 0,
      };
    }

    _recordUploadBytes(byteLength) {
      const bytes = toNonNegativeInt(byteLength);
      this.stats.bytesUploaded += bytes;
      this.stats.uploadBytes += bytes;
    }

    /**
     * Return a sanitized snapshot of texture packet counters (finite non-negative
     * integers only; no host paths, raw buffers, or mutable internal references).
     */
    getStats() {
      return {
        created: toNonNegativeInt(this.stats.created),
        errors: toNonNegativeInt(this.stats.errors),
        createTexture2Count: toNonNegativeInt(this.stats.createTexture2Count),
        rowPitchRepackCount: toNonNegativeInt(this.stats.rowPitchRepackCount),
        sabCopyCount: toNonNegativeInt(this.stats.sabCopyCount),
        bc3dFallbacks: toNonNegativeInt(this.stats.bc3dFallbacks),
        bc3dFallbackUploadsSkipped: toNonNegativeInt(this.stats.bc3dFallbackUploadsSkipped),
        uploadCount: toNonNegativeInt(this.stats.uploadCount),
        uploadBytes: toNonNegativeInt(this.stats.uploadBytes),
        zeroCopyUploads: toNonNegativeInt(this.stats.zeroCopyUploads),
        scratchRepackUploads: toNonNegativeInt(this.stats.scratchRepackUploads),
        directZeroCopyUploads: toNonNegativeInt(this.stats.directZeroCopyUploads),
        repackedUploads: toNonNegativeInt(this.stats.repackedUploads),
        scratchBufferReuses: toNonNegativeInt(this.stats.scratchBufferReuses),
        scratchBufferAllocations: toNonNegativeInt(this.stats.scratchBufferAllocations),
        scratchBufferBytes: toNonNegativeInt(
          this._scratchBuffer ? this._scratchBuffer.byteLength : 0
        ),
        r8unormUploads: toNonNegativeInt(this.stats.r8unormUploads),
        bytesUploaded: toNonNegativeInt(this.stats.bytesUploaded),
        srgbSiblingViewFormatsAdded: toNonNegativeInt(this.stats.srgbSiblingViewFormatsAdded),
      };
    }

    /**
     * Return a lightweight sanitized snapshot of WriteTexture / texture upload
     * telemetry (`uploadCount`, `uploadBytes`, `zeroCopyUploads`,
     * `scratchRepackUploads`, and scratch/SAB counters).
     */
    getUploadStats() {
      return {
        uploadCount: toNonNegativeInt(this.stats.uploadCount),
        uploadBytes: toNonNegativeInt(this.stats.uploadBytes),
        zeroCopyUploads: toNonNegativeInt(this.stats.zeroCopyUploads),
        scratchRepackUploads: toNonNegativeInt(this.stats.scratchRepackUploads),
        scratchBufferAllocations: toNonNegativeInt(this.stats.scratchBufferAllocations),
        scratchBufferReuses: toNonNegativeInt(this.stats.scratchBufferReuses),
        scratchBufferBytes: toNonNegativeInt(
          this._scratchBuffer ? this._scratchBuffer.byteLength : 0
        ),
        sabCopyCount: toNonNegativeInt(this.stats.sabCopyCount),
      };
    }

    get uploadStats() {
      return this.getUploadStats();
    }

    /**
     * Reset all telemetry counters, cached texture format/fallback tracking, and
     * release the persistent scratch buffer. Handler remains usable after reset().
     */
    reset() {
      this._bc3dFallbackTextures.clear();
      this._bc3dFallbackTextureObjs = new WeakSet();
      this._textureFormats.clear();
      this._textureObjFormats = new WeakMap();
      this._scratchBuffer = null;
      this.stats = this._createEmptyStats();
      this.lastError = null;
      if (this.next && typeof this.next.reset === 'function') {
        this.next.reset();
      }
    }

    /**
     * Release internal resources (including the persistent scratch buffer) and
     * reset all telemetry counters. Idempotent.
     */
    destroy() {
      this._bc3dFallbackTextures.clear();
      this._bc3dFallbackTextureObjs = new WeakSet();
      this._textureFormats.clear();
      this._textureObjFormats = new WeakMap();
      this._scratchBuffer = null;
      this.stats = this._createEmptyStats();
      this.lastError = null;
      if (this.next && typeof this.next.destroy === 'function') {
        this.next.destroy();
      }
    }

    _acquireScratchBuffer(requiredBytes) {
      if (
        !Number.isInteger(requiredBytes) ||
        requiredBytes <= 0 ||
        requiredBytes > this._maxScratchBytes
      ) {
        return { buffer: null, reused: false };
      }
      if (this._scratchBuffer && this._scratchBuffer.byteLength >= requiredBytes) {
        this.stats.scratchBufferReuses += 1;
        return { buffer: this._scratchBuffer, reused: true };
      }
      this._scratchBuffer = new Uint8Array(requiredBytes);
      this.stats.scratchBufferAllocations += 1;
      return { buffer: this._scratchBuffer, reused: false };
    }

    _prepareDirectWriteData(input) {
      const src = toUint8Array(input, 'writeTexture');
      if (!isSharedBuffer(src.buffer)) {
        return { data: src, copiedFromSab: false, reusedScratch: false };
      }
      if (src.byteLength === 0) {
        return { data: new Uint8Array(0), copiedFromSab: true, reusedScratch: false };
      }
      const scratch = this._acquireScratchBuffer(src.byteLength);
      if (scratch.buffer) {
        const dst = scratch.buffer.subarray(0, src.byteLength);
        dst.set(src, 0);
        return { data: dst, copiedFromSab: true, reusedScratch: scratch.reused };
      }
      const copy = new Uint8Array(src.byteLength);
      copy.set(src, 0);
      return { data: copy, copiedFromSab: true, reusedScratch: false };
    }

    _repackRowsWithScratch(srcBytes, bytesPerRow, rowCount, depthOrLayers = 1) {
      const src = toUint8Array(srcBytes, 'repackRowsTo256');
      const totalRows =
        Number.isInteger(rowCount) && rowCount > 0 && Number.isInteger(depthOrLayers) && depthOrLayers > 0
          ? rowCount * depthOrLayers
          : 0;
      const alignedBytesPerRow =
        Number.isInteger(bytesPerRow) && bytesPerRow > 0 ? alignBytesPerRow256(bytesPerRow) : 0;
      const fromSab = isSharedBuffer(src.buffer);
      const needsCopyOrRepack =
        totalRows > 0 && alignedBytesPerRow > 0 && (alignedBytesPerRow !== bytesPerRow || fromSab);

      let scratchBuffer = null;
      if (needsCopyOrRepack && src.byteLength >= bytesPerRow * totalRows) {
        const neededBytes = alignedBytesPerRow * totalRows;
        const acquired = this._acquireScratchBuffer(neededBytes);
        scratchBuffer = acquired.buffer;
      }

      return repackRowsTo256(src, bytesPerRow, rowCount, depthOrLayers, scratchBuffer);
    }

    _isR8UnormTexture(texture, textureId) {
      if (textureId !== undefined && this._textureFormats.get(textureId) === 'r8unorm') {
        return true;
      }
      if (texture && typeof texture === 'object') {
        if (this._textureObjFormats.get(texture) === 'r8unorm') {
          return true;
        }
        if (texture.format === 'r8unorm') {
          return true;
        }
        if (texture.desc && texture.desc.format === 'r8unorm') {
          return true;
        }
      }
      return false;
    }

    handlePacket(opcode, view, payloadOffset, payloadBytes) {
      if (opcode === CREATE_TEXTURE2) {
        if (payloadBytes < CREATE_TEXTURE2_BYTES) {
          this._error(`CreateTexture2: payload ${payloadBytes} < ${CREATE_TEXTURE2_BYTES} bytes`);
          return true;
        }
        const decoded = decodeCreateTexture2(view, payloadOffset, payloadBytes);
        if (decoded.error) {
          this._error(decoded.error);
          return true;
        }
        this._create(decoded.id, decoded.descriptor, true);
        return true;
      }

      if (opcode === CREATE_TEXTURE) {
        if (payloadBytes < CREATE_TEXTURE_BYTES) {
          this._error(`CreateTexture: payload ${payloadBytes} < ${CREATE_TEXTURE_BYTES} bytes`);
          return true;
        }
        const decoded = decodeCreateTexture(view, payloadOffset, payloadBytes);
        if (decoded.error) {
          this._error(decoded.error);
          return true;
        }
        this._create(decoded.id, decoded.descriptor, false);
        return true;
      }

      if (opcode === WRITE_TEXTURE) {
        if (payloadBytes < WRITE_TEXTURE_HEADER_BYTES) {
          this._error(
            `WriteTexture: payload ${payloadBytes} < ${WRITE_TEXTURE_HEADER_BYTES} bytes`
          );
          return true;
        }
        this.handleWriteTexture(view, payloadOffset, payloadBytes);
        return true;
      }

      return this.next ? this.next.handlePacket(opcode, view, payloadOffset, payloadBytes) : false;
    }

    destroyResource(id) {
      this._bc3dFallbackTextures.delete(id);
      this._textureFormats.delete(id);
      if (this.next && typeof this.next.destroyResource === 'function') {
        this.next.destroyResource(id);
      }
    }

    /**
     * Safe `queue.writeTexture` wrapper that takes a zero-copy direct upload path
     * for non-shared views and copies `SharedArrayBuffer`-backed views into a
     * reusable non-shared scratch `Uint8Array` before uploading.
     */
    writeTexture(destination, data, dataLayout, size, textureId) {
      if (
        destination &&
        destination.texture &&
        this._bc3dFallbackTextureObjs.has(destination.texture)
      ) {
        this.stats.bc3dFallbackUploadsSkipped += 1;
        return { uploaded: false, copiedFromSab: false, skippedBc3dFallback: true };
      }

      const p = this.processor;
      if (!p || !p.device || !p.device.queue || typeof p.device.queue.writeTexture !== 'function') {
        this._error('writeTexture: device.queue.writeTexture is unavailable');
        return { uploaded: false, copiedFromSab: false };
      }

      const prepared = this._prepareDirectWriteData(data);
      if (prepared.copiedFromSab) {
        this.stats.sabCopyCount += 1;
      }

      p.device.queue.writeTexture(destination, prepared.data, dataLayout, size);

      this.stats.uploadCount += 1;
      this._recordUploadBytes(prepared.data.byteLength);
      if (!prepared.copiedFromSab) {
        this.stats.directZeroCopyUploads += 1;
        this.stats.zeroCopyUploads += 1;
      }
      if (this._isR8UnormTexture(destination && destination.texture, textureId)) {
        this.stats.r8unormUploads += 1;
      }

      return {
        uploaded: true,
        copiedFromSab: prepared.copiedFromSab,
        zeroCopy: !prepared.copiedFromSab,
        reusedScratch: prepared.reusedScratch,
      };
    }

    /**
     * Hardened `encoder.copyBufferToTexture` wrapper:
     * - Repacks CPU texture data (`sourceBytes` or `source.data`) via `repackRowsTo256`
     *   whenever `bytesPerRow` is not a multiple of 256 bytes, incrementing
     *   `stats.rowPitchRepackCount`.
     * - Safely copies `SharedArrayBuffer`-backed views out before staging/uploading,
     *   incrementing `stats.sabCopyCount`.
     * - When copying from an existing `GPUBuffer` with an unaligned 4-byte-multiple
     *   `bytesPerRow`, performs a GPU-side row-by-row repack via `copyBufferToBuffer`
     *   into a 256-byte-aligned staging buffer before `copyBufferToTexture`.
     */
    copyBufferToTexture(encoder, source, destination, copySize, sourceBytes) {
      if (!encoder || typeof encoder.copyBufferToTexture !== 'function') {
        throw new TypeError('copyBufferToTexture requires an encoder with copyBufferToTexture()');
      }
      if (!source || typeof source !== 'object') {
        throw new TypeError('copyBufferToTexture requires a source descriptor object');
      }
      if (!destination || typeof destination !== 'object' || !destination.texture) {
        throw new TypeError('copyBufferToTexture requires a destination with a texture');
      }

      if (this._bc3dFallbackTextureObjs.has(destination.texture)) {
        this.stats.bc3dFallbackUploadsSkipped += 1;
        return {
          repacked: false,
          copiedFromSab: false,
          bytesPerRow: 0,
          originalBytesPerRow: Number.isInteger(source.bytesPerRow) ? source.bytesPerRow : 0,
          staged: false,
          skippedBc3dFallback: true,
        };
      }

      const normSize = normalizeCopySize(copySize);
      const rowsPerImage =
        Number.isInteger(source.rowsPerImage) && source.rowsPerImage > 0
          ? source.rowsPerImage
          : normSize.height;
      const totalRows = rowsPerImage * normSize.depthOrArrayLayers;
      const cpuInput = sourceBytes !== undefined ? sourceBytes : source.data;
      const p = this.processor;
      const device = p && p.device ? p.device : null;

      if (cpuInput !== undefined && cpuInput !== null) {
        const rawBytes = toUint8Array(cpuInput, 'copyBufferToTexture');
        let rawBytesPerRow = source.bytesPerRow;
        if (!Number.isInteger(rawBytesPerRow) || rawBytesPerRow <= 0) {
          if (totalRows > 0 && rawBytes.byteLength > 0 && rawBytes.byteLength % totalRows === 0) {
            rawBytesPerRow = rawBytes.byteLength / totalRows;
          } else {
            this._error('copyBufferToTexture: unable to infer positive bytesPerRow');
            return {
              repacked: false,
              copiedFromSab: false,
              bytesPerRow: 0,
              originalBytesPerRow: 0,
              error: this.lastError,
            };
          }
        }

        const packed = this._repackRowsWithScratch(
          rawBytes,
          rawBytesPerRow,
          rowsPerImage,
          normSize.depthOrArrayLayers
        );
        if (packed.repacked) {
          this.stats.rowPitchRepackCount += 1;
          this.stats.repackedUploads += 1;
          if (packed.reusedScratch) {
            this.stats.scratchRepackUploads += 1;
          }
        }
        if (packed.copiedFromSab) {
          this.stats.sabCopyCount += 1;
        }
        const isR8 = this._isR8UnormTexture(destination.texture);

        if (
          p &&
          p.stagingPool &&
          !p.stagingPool._destroyed &&
          typeof p.stagingPool.stageTextureUpload === 'function'
        ) {
          const stageLayout = {
            width: normSize.width,
            height: normSize.height,
            depthOrArrayLayers: normSize.depthOrArrayLayers,
            mipLevel: destination.mipLevel || 0,
            origin: destination.origin || { x: 0, y: 0, z: 0 },
            bytesPerRow: packed.bytesPerRow,
            rowsPerImage,
          };
          if (destination.aspect !== undefined) {
            stageLayout.aspect = destination.aspect;
          }
          p.stagingPool.stageTextureUpload(encoder, destination.texture, packed.data, stageLayout);
          this.stats.uploadCount += 1;
          this._recordUploadBytes(packed.data.byteLength);
          if (!packed.repacked && !packed.copiedFromSab) {
            this.stats.zeroCopyUploads += 1;
          }
          if (isR8) {
            this.stats.r8unormUploads += 1;
          }
          return {
            repacked: packed.repacked,
            copiedFromSab: packed.copiedFromSab,
            reusedScratch: packed.reusedScratch,
            bytesPerRow: packed.bytesPerRow,
            originalBytesPerRow: rawBytesPerRow,
            staged: true,
          };
        }

        if (!packed.repacked && source.buffer) {
          const offset = source.offset || 0;
          if (device && device.queue && typeof device.queue.writeBuffer === 'function') {
            device.queue.writeBuffer(source.buffer, offset, packed.data);
          }
          encoder.copyBufferToTexture(
            {
              buffer: source.buffer,
              offset,
              bytesPerRow: packed.bytesPerRow,
              rowsPerImage,
            },
            destination,
            normSize
          );
          this.stats.uploadCount += 1;
          this._recordUploadBytes(packed.data.byteLength);
          if (!packed.copiedFromSab) {
            this.stats.zeroCopyUploads += 1;
          }
          if (isR8) {
            this.stats.r8unormUploads += 1;
          }
          return {
            repacked: false,
            copiedFromSab: packed.copiedFromSab,
            reusedScratch: packed.reusedScratch,
            bytesPerRow: packed.bytesPerRow,
            originalBytesPerRow: rawBytesPerRow,
            staged: false,
          };
        }

        if (
          !device ||
          typeof device.createBuffer !== 'function' ||
          !device.queue ||
          typeof device.queue.writeBuffer !== 'function'
        ) {
          this._error(
            'copyBufferToTexture: device.createBuffer/queue.writeBuffer required for repacking'
          );
          return {
            repacked: packed.repacked,
            copiedFromSab: packed.copiedFromSab,
            bytesPerRow: packed.bytesPerRow,
            originalBytesPerRow: rawBytesPerRow,
            error: this.lastError,
          };
        }

        const stagingBuffer = device.createBuffer({
          label: 'playgta6-texture-repack-staging',
          size: Math.max(4, packed.data.byteLength),
          usage: GPU_BUFFER_USAGE_COPY_SRC_DST,
        });
        device.queue.writeBuffer(stagingBuffer, 0, packed.data);
        encoder.copyBufferToTexture(
          {
            buffer: stagingBuffer,
            offset: 0,
            bytesPerRow: packed.bytesPerRow,
            rowsPerImage,
          },
          destination,
          normSize
        );
        this.stats.uploadCount += 1;
        this._recordUploadBytes(packed.data.byteLength);
        if (!packed.repacked && !packed.copiedFromSab) {
          this.stats.zeroCopyUploads += 1;
        }
        if (isR8) {
          this.stats.r8unormUploads += 1;
        }
        return {
          repacked: packed.repacked,
          copiedFromSab: packed.copiedFromSab,
          reusedScratch: packed.reusedScratch,
          bytesPerRow: packed.bytesPerRow,
          originalBytesPerRow: rawBytesPerRow,
          stagingBuffer,
          staged: true,
        };
      }

      // GPU-buffer-only copy path (no CPU bytes supplied).
      const rawBytesPerRow = Number.isInteger(source.bytesPerRow) ? source.bytesPerRow : 0;
      const isR8 = this._isR8UnormTexture(destination.texture);
      if (rawBytesPerRow === 0 || rawBytesPerRow % WEBGPU_BYTES_PER_ROW_ALIGNMENT === 0) {
        encoder.copyBufferToTexture(source, destination, normSize);
        this.stats.uploadCount += 1;
        if (rawBytesPerRow > 0) {
          this._recordUploadBytes(rawBytesPerRow * totalRows);
        }
        if (isR8) {
          this.stats.r8unormUploads += 1;
        }
        return {
          repacked: false,
          copiedFromSab: false,
          bytesPerRow: rawBytesPerRow,
          originalBytesPerRow: rawBytesPerRow,
          staged: false,
        };
      }

      const alignedBytesPerRow = alignBytesPerRow256(rawBytesPerRow);
      const repackedTotalBytes = alignedBytesPerRow * totalRows;
      if (
        source.buffer &&
        rawBytesPerRow % 4 === 0 &&
        typeof encoder.copyBufferToBuffer === 'function' &&
        device &&
        typeof device.createBuffer === 'function'
      ) {
        let dstBuffer;
        let dstOffset = 0;
        if (
          p &&
          p.stagingPool &&
          !p.stagingPool._destroyed &&
          typeof p.stagingPool.allocate === 'function'
        ) {
          const slice = p.stagingPool.allocate(
            Math.max(4, repackedTotalBytes),
            WEBGPU_BYTES_PER_ROW_ALIGNMENT
          );
          dstBuffer = slice.buffer;
          dstOffset = slice.offset;
        } else {
          dstBuffer = device.createBuffer({
            label: 'playgta6-texture-repack-gpu',
            size: Math.max(4, repackedTotalBytes),
            usage: GPU_BUFFER_USAGE_COPY_SRC_DST,
          });
        }

        const srcBaseOffset = source.offset || 0;
        for (let row = 0; row < totalRows; row += 1) {
          encoder.copyBufferToBuffer(
            source.buffer,
            srcBaseOffset + row * rawBytesPerRow,
            dstBuffer,
            dstOffset + row * alignedBytesPerRow,
            rawBytesPerRow
          );
        }
        encoder.copyBufferToTexture(
          {
            buffer: dstBuffer,
            offset: dstOffset,
            bytesPerRow: alignedBytesPerRow,
            rowsPerImage,
          },
          destination,
          normSize
        );
        this.stats.uploadCount += 1;
        this.stats.rowPitchRepackCount += 1;
        this.stats.repackedUploads += 1;
        this._recordUploadBytes(repackedTotalBytes);
        if (isR8) {
          this.stats.r8unormUploads += 1;
        }
        return {
          repacked: true,
          copiedFromSab: false,
          bytesPerRow: alignedBytesPerRow,
          originalBytesPerRow: rawBytesPerRow,
          stagingBuffer: dstBuffer,
          staged: true,
        };
      }

      this._error(
        `copyBufferToTexture: unaligned bytesPerRow ${rawBytesPerRow} cannot be repacked without CPU bytes or 4-byte-aligned copyBufferToBuffer`
      );
      return {
        repacked: false,
        copiedFromSab: false,
        bytesPerRow: rawBytesPerRow,
        originalBytesPerRow: rawBytesPerRow,
        error: this.lastError,
      };
    }

    /**
     * Hardened `WriteTexture` (opcode 13) handler:
     * - When `processor.stagingPool` is active, repacks unaligned rows to 256-byte
     *   `bytesPerRow` via `repackRowsTo256` (incrementing `rowPitchRepackCount`) and
     *   copies `SharedArrayBuffer`-backed views safely (incrementing `sabCopyCount`)
     *   before recording `copyBufferToTexture`.
     * - When uploading directly via `device.queue.writeTexture`, copies
     *   `SharedArrayBuffer`-backed views safely (incrementing `sabCopyCount`).
     */
    handleWriteTexture(view, payloadOffset, payloadBytes) {
      const p = this.processor;
      if (!p || !p.device) {
        return;
      }

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
      if (WRITE_TEXTURE_HEADER_BYTES + dataLen > payloadBytes) {
        return;
      }

      const texture = p.textures.get(textureId);
      if (!texture) {
        return;
      }

      if (this._bc3dFallbackTextures.has(textureId)) {
        this.stats.bc3dFallbackUploadsSkipped += 1;
        return;
      }

      const rawData = new Uint8Array(
        view.buffer,
        view.byteOffset + payloadOffset + WRITE_TEXTURE_HEADER_BYTES,
        dataLen
      );

      const effectiveRowsPerImage = rowsPerImage > 0 ? rowsPerImage : height;
      const totalRows = effectiveRowsPerImage * depthOrArrayLayers;
      const effectiveBytesPerRow =
        bytesPerRow > 0
          ? bytesPerRow
          : totalRows > 0 && dataLen % totalRows === 0
            ? dataLen / totalRows
            : 0;
      const isRowAligned256 =
        effectiveBytesPerRow > 0 &&
        effectiveBytesPerRow % WEBGPU_BYTES_PER_ROW_ALIGNMENT === 0;
      const hasDirectQueueWrite = Boolean(
        p.device.queue && typeof p.device.queue.writeTexture === 'function'
      );

      // Fast path: when source row pitch is already 256-byte aligned (e.g., 4K Bink2
      // 3840x2160 r8unorm Y plane where 3840 % 256 === 0), upload directly via
      // queue.writeTexture without staging pool churn or scratch buffer repacking.
      if (isRowAligned256 && hasDirectQueueWrite && width > 0 && height > 0 && dataLen > 0) {
        const directLayout = { offset: 0, bytesPerRow: effectiveBytesPerRow };
        if (rowsPerImage > 0) {
          directLayout.rowsPerImage = rowsPerImage;
        } else if (totalRows > 1) {
          directLayout.rowsPerImage = effectiveRowsPerImage;
        }
        this.writeTexture(
          {
            texture,
            mipLevel,
            origin: { x: originX, y: originY, z: originZ },
          },
          rawData,
          directLayout,
          {
            width,
            height,
            depthOrArrayLayers,
          },
          textureId
        );
        return;
      }

      if (
        p.stagingPool &&
        !p.stagingPool._destroyed &&
        typeof p.stagingPool.stageTextureUpload === 'function' &&
        width > 0 &&
        height > 0 &&
        dataLen > 0
      ) {
        const canStage =
          effectiveBytesPerRow > 0 && dataLen >= effectiveBytesPerRow * totalRows;

        if (canStage) {
          const encoder =
            typeof p._ensurePendingEncoder === 'function' ? p._ensurePendingEncoder() : null;
          if (encoder && typeof encoder.copyBufferToTexture === 'function') {
            const packed = this._repackRowsWithScratch(
              rawData,
              effectiveBytesPerRow,
              effectiveRowsPerImage,
              depthOrArrayLayers
            );
            if (packed.repacked) {
              this.stats.rowPitchRepackCount += 1;
              this.stats.repackedUploads += 1;
              if (packed.reusedScratch) {
                this.stats.scratchRepackUploads += 1;
              }
            }
            if (packed.copiedFromSab) {
              this.stats.sabCopyCount += 1;
            }

            const layout = {
              width,
              height,
              depthOrArrayLayers,
              mipLevel,
              origin: { x: originX, y: originY, z: originZ },
              bytesPerRow: packed.bytesPerRow,
              rowsPerImage: effectiveRowsPerImage,
            };
            p.stagingPool.stageTextureUpload(encoder, texture, packed.data, layout);
            p._hasPendingUploadCommands = true;
            this.stats.uploadCount += 1;
            this._recordUploadBytes(packed.data.byteLength);
            if (!packed.repacked && !packed.copiedFromSab) {
              this.stats.zeroCopyUploads += 1;
            }
            if (this._isR8UnormTexture(texture, textureId)) {
              this.stats.r8unormUploads += 1;
            }
            return;
          }
          if (!p._hasPendingUploadCommands) {
            p._pendingEncoder = null;
          }
        }
      }

      const dataLayout = { offset: 0 };
      if (bytesPerRow > 0) {
        dataLayout.bytesPerRow = bytesPerRow;
      } else if (effectiveBytesPerRow > 0 && totalRows > 1) {
        dataLayout.bytesPerRow = effectiveBytesPerRow;
      }
      if (rowsPerImage > 0) {
        dataLayout.rowsPerImage = rowsPerImage;
      } else if (depthOrArrayLayers > 1 && effectiveRowsPerImage > 0) {
        dataLayout.rowsPerImage = effectiveRowsPerImage;
      }

      this.writeTexture(
        {
          texture,
          mipLevel,
          origin: { x: originX, y: originY, z: originZ },
        },
        rawData,
        dataLayout,
        {
          width,
          height,
          depthOrArrayLayers,
        },
        textureId
      );
    }

    _create(id, descriptor, isV2 = true) {
      const p = this.processor;
      this._bc3dFallbackTextures.delete(id);
      this._textureFormats.delete(id);
      if (typeof p._destroyTextureIfPresent === 'function') {
        p._destroyTextureIfPresent(id);
      }
      // Cached v2 views of a previous texture with this id must not survive.
      if (this.next && typeof this.next.destroyResource === 'function') {
        this.next.destroyResource(id);
      }

      let effectiveDescriptor = descriptor;
      let bc3dFallback = false;
      let srgbSiblingAdded = false;
      if (
        descriptor &&
        descriptor.dimension === '3d' &&
        isBcCompressedFormat(descriptor.format) &&
        !deviceSupportsBcSliced3d(p && p.device)
      ) {
        // W108 (#734): WebGPU rejects 3D BC-compressed textures without texture-compression-bc-sliced-3d.
        bc3dFallback = true;
        effectiveDescriptor = {
          ...descriptor,
          size: descriptor.size ? { ...descriptor.size } : descriptor.size,
          format: 'rgba8unorm',
        };
        if (Array.isArray(descriptor.viewFormats)) {
          const filteredViewFormats = descriptor.viewFormats.filter(
            (vf) => !isBcCompressedFormat(vf)
          );
          if (filteredViewFormats.length > 0) {
            effectiveDescriptor.viewFormats = filteredViewFormats;
          } else {
            delete effectiveDescriptor.viewFormats;
          }
        }
      } else if (descriptor && typeof descriptor.format === 'string') {
        const resolved = resolveCompatibleViewFormats(descriptor.format, descriptor.viewFormats);
        const origViewFormats = Array.isArray(descriptor.viewFormats) ? descriptor.viewFormats : null;
        const resolvedViewFormats = resolved.viewFormats || null;
        const changed =
          resolved.siblingAdded ||
          (origViewFormats !== null &&
            (resolvedViewFormats === null ||
              origViewFormats.length !== resolvedViewFormats.length ||
              origViewFormats.some((vf, idx) => vf !== resolvedViewFormats[idx])));
        if (changed) {
          effectiveDescriptor = {
            ...descriptor,
            size: descriptor.size ? { ...descriptor.size } : descriptor.size,
          };
          if (resolvedViewFormats && resolvedViewFormats.length > 0) {
            effectiveDescriptor.viewFormats = resolvedViewFormats;
          } else {
            delete effectiveDescriptor.viewFormats;
          }
          srgbSiblingAdded = resolved.siblingAdded;
        } else if (!isV2 && resolvedViewFormats && resolvedViewFormats.length > 0) {
          // decodeCreateTexture (v1) already attached the compatible sibling viewFormat.
          srgbSiblingAdded = true;
        }
      }

      let texture;
      try {
        texture = p.device.createTexture(effectiveDescriptor);
      } catch (err) {
        const label = isV2 ? 'CreateTexture2' : 'CreateTexture';
        this._error(`${label} ${id}: ${String(err && err.message ? err.message : err)}`);
        return;
      }
      p.textures.set(id, texture);
      if (effectiveDescriptor && typeof effectiveDescriptor.format === 'string') {
        this._textureFormats.set(id, effectiveDescriptor.format);
        if (texture && typeof texture === 'object') {
          this._textureObjFormats.set(texture, effectiveDescriptor.format);
        }
      }
      if (srgbSiblingAdded) {
        this.stats.srgbSiblingViewFormatsAdded += 1;
      }
      if (bc3dFallback) {
        this._bc3dFallbackTextures.add(id);
        if (texture && typeof texture === 'object') {
          this._bc3dFallbackTextureObjs.add(texture);
        }
        this.stats.bc3dFallbacks += 1;
        // Propagate fallback metadata to chained RenderStatePackets so CreateBindGroup2 /
        // BeginRenderPass views never pass a bc* view format against the rgba8unorm texture.
        if (this.next && this.next.textureMeta && typeof this.next.textureMeta.set === 'function') {
          this.next.textureMeta.set(id, {
            id,
            requestedFormat: descriptor.format,
            format: effectiveDescriptor.format,
            bcFallback: true,
            formatFallback: true,
            usage: effectiveDescriptor.usage,
            dimension: effectiveDescriptor.dimension,
            sampleCount: effectiveDescriptor.sampleCount || 1,
            width: effectiveDescriptor.size ? effectiveDescriptor.size.width : 1,
            height: effectiveDescriptor.size ? effectiveDescriptor.size.height : 1,
            depthOrArrayLayers: effectiveDescriptor.size
              ? effectiveDescriptor.size.depthOrArrayLayers
              : 1,
          });
        }
      }
      let textureView = null;
      if (texture && typeof texture.createView === 'function') {
        textureView = texture.createView();
        p.textureViews.set(id, textureView);
      }
      if (p.resourceRegistry && typeof p.resourceRegistry.register === 'function') {
        p.resourceRegistry.register('texture', id, texture, {
          byteSize: 0,
          label: `texture:${id}`,
        });
        if (textureView) {
          p.resourceRegistry.register('textureView', id, textureView, {
            byteSize: 0,
            label: `textureView:${id}`,
          });
        }
      }
      this.stats.created += 1;
      if (isV2) {
        this.stats.createTexture2Count += 1;
      }
    }

    _error(message) {
      this.stats.errors += 1;
      this.lastError = message;
    }
  }

  function installTexturePackets(processor, options = {}) {
    const handler = new TexturePackets(processor, processor.extension || null, options);
    processor.extension = handler;

    if (processor && typeof processor === 'object') {
      if (
        typeof processor._handleCreateTexture === 'function' &&
        !processor[HOOK_ORIG_CREATE_TEXTURE]
      ) {
        processor[HOOK_ORIG_CREATE_TEXTURE] = processor._handleCreateTexture;
      }
      if (typeof processor._handleCreateTexture === 'function') {
        processor._handleCreateTexture = function hardenedHandleCreateTexture(view, payloadOffset) {
          const decoded = decodeCreateTexture(view, payloadOffset);
          if (decoded.error) {
            handler._error(decoded.error);
            return;
          }
          handler._create(decoded.id, decoded.descriptor, false);
        };
      }

      if (
        typeof processor._handleWriteTexture === 'function' &&
        !processor[HOOK_ORIG_WRITE_TEXTURE]
      ) {
        processor[HOOK_ORIG_WRITE_TEXTURE] = processor._handleWriteTexture;
      }
      if (typeof processor._handleWriteTexture === 'function') {
        processor._handleWriteTexture = function hardenedHandleWriteTexture(
          view,
          payloadOffset,
          payloadBytes
        ) {
          handler.handleWriteTexture(view, payloadOffset, payloadBytes);
        };
      }

      if (typeof processor.destroy === 'function' && !processor[HOOK_ORIG_DESTROY]) {
        processor[HOOK_ORIG_DESTROY] = processor.destroy;
      }
      if (typeof processor[HOOK_ORIG_DESTROY] === 'function') {
        const origDestroy = processor[HOOK_ORIG_DESTROY];
        processor.destroy = function hardenedDestroy() {
          const result = origDestroy.apply(this, arguments);
          handler.destroy();
          return result;
        };
      }
    }

    return handler;
  }

  const exportedApi = {
    TexturePackets,
    installTexturePackets,
    decodeCreateTexture,
    decodeCreateTexture2,
    computeMaxMipLevels,
    clampMipLevels,
    clampSampleCount,
    alignBytesPerRow256,
    repackRowsTo256,
    copyFromSharedIfNeeded,
    isSharedArrayBufferView,
    isBcCompressedFormat,
    deviceSupportsBcSliced3d,
    getSrgbSiblingFormat,
    resolveCompatibleViewFormats,
    SRGB_SIBLING_FORMATS,
    DEFAULT_MAX_SCRATCH_BYTES,
    BC_SLICED_3D_FEATURE,
    CREATE_TEXTURE,
    CREATE_TEXTURE_BYTES,
    WRITE_TEXTURE,
    WRITE_TEXTURE_HEADER_BYTES,
    CREATE_TEXTURE2,
    CREATE_TEXTURE2_BYTES,
    MAX_VIEW_FORMATS,
    WEBGPU_BYTES_PER_ROW_ALIGNMENT,
    TEXTURE_FORMAT_NAMES,
    V1_TEXTURE_FORMAT_NAMES,
    TEXTURE_DIMENSION_NAMES,
  };

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6TexturePackets = exportedApi;
  } else if (globalScope) {
    globalScope.PlayGta6TexturePackets = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
