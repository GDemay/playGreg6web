// GPU packet ABI v1/v2 texture handling for WgpuCommandProcessor:
// - ABI v2 CreateTexture2 (opcode 35, 40-byte payload, docs/gpu_commands.md "CreateTexture2")
//   with safe mipLevels and sampleCount clamping alongside ABI v1 CreateTexture (opcode 12).
// - WebGPU 256-byte bytesPerRow alignment & row-pitch repacking helpers (`alignBytesPerRow256`,
//   `repackRowsTo256`) applied on buffer-to-texture copies (`copyBufferToTexture` / staged
//   `WriteTexture`) so unaligned row strides never fail WebGPU validation.
// - Safe `SharedArrayBuffer`-backed view copying before `queue.writeTexture` /
//   `copyBufferToTexture`.
// - Sanitized telemetry counters (`createTexture2Count`, `rowPitchRepackCount`, `sabCopyCount`).
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
  const GPU_BUFFER_USAGE_COPY_SRC_DST = 0x0004 | 0x0008;

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

  function isBcCompressedFormat(format) {
    return typeof format === 'string' && format.startsWith('bc');
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
   * If `input` is backed by a `SharedArrayBuffer`, allocates a standalone copy so
   * WebGPU `queue.writeTexture` / `queue.writeBuffer` / `copyBufferToTexture` calls
   * never reject shared buffers or race with concurrent guest writes.
   *
   * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} input
   * @returns {{ data: Uint8Array, copiedFromSab: boolean }}
   */
  function copyFromSharedIfNeeded(input) {
    const src = toUint8Array(input, 'copyFromSharedIfNeeded');
    if (isSharedBuffer(src.buffer)) {
      const copy = new Uint8Array(src.byteLength);
      copy.set(src, 0);
      return { data: copy, copiedFromSab: true };
    }
    return { data: src, copiedFromSab: false };
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
   * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} srcBytes
   * @param {number} bytesPerRow
   * @param {number} rowCount
   * @param {number} [depthOrLayers=1]
   * @returns {{
   *   data: Uint8Array,
   *   bytesPerRow: number,
   *   originalBytesPerRow: number,
   *   rowCount: number,
   *   repacked: boolean,
   *   copiedFromSab: boolean,
   * }}
   */
  function repackRowsTo256(srcBytes, bytesPerRow, rowCount, depthOrLayers = 1) {
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
        const copy = new Uint8Array(requiredSrcBytes);
        copy.set(slice, 0);
        return {
          data: copy,
          bytesPerRow: alignedBytesPerRow,
          originalBytesPerRow: bytesPerRow,
          rowCount: totalRows,
          repacked: false,
          copiedFromSab: true,
        };
      }
      return {
        data: slice,
        bytesPerRow: alignedBytesPerRow,
        originalBytesPerRow: bytesPerRow,
        rowCount: totalRows,
        repacked: false,
        copiedFromSab: false,
      };
    }

    const dst = new Uint8Array(alignedBytesPerRow * totalRows);
    for (let row = 0; row < totalRows; row += 1) {
      const srcOffset = row * bytesPerRow;
      dst.set(src.subarray(srcOffset, srcOffset + bytesPerRow), row * alignedBytesPerRow);
    }

    return {
      data: dst,
      bytesPerRow: alignedBytesPerRow,
      originalBytesPerRow: bytesPerRow,
      rowCount: totalRows,
      repacked: true,
      copiedFromSab: fromSab,
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

    return {
      id,
      descriptor: {
        size: { width, height, depthOrArrayLayers },
        mipLevelCount,
        format,
        usage,
        dimension,
      },
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
    constructor(processor, next) {
      this.processor = processor;
      this.next = next || null;
      this._bc3dFallbackTextures = new Set();
      this._bc3dFallbackTextureObjs = new WeakSet();
      this.stats = {
        created: 0,
        errors: 0,
        createTexture2Count: 0,
        rowPitchRepackCount: 0,
        sabCopyCount: 0,
        bc3dFallbacks: 0,
        bc3dFallbackUploadsSkipped: 0,
      };
      this.lastError = null;
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
      };
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
      if (this.next && typeof this.next.destroyResource === 'function') {
        this.next.destroyResource(id);
      }
    }

    /**
     * Safe `queue.writeTexture` wrapper that copies `SharedArrayBuffer`-backed views
     * into a standalone `ArrayBuffer`-backed `Uint8Array` before uploading.
     */
    writeTexture(destination, data, dataLayout, size) {
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

      const prepared = copyFromSharedIfNeeded(data);
      if (prepared.copiedFromSab) {
        this.stats.sabCopyCount += 1;
      }

      p.device.queue.writeTexture(destination, prepared.data, dataLayout, size);
      return { uploaded: true, copiedFromSab: prepared.copiedFromSab };
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

        const packed = repackRowsTo256(
          rawBytes,
          rawBytesPerRow,
          rowsPerImage,
          normSize.depthOrArrayLayers
        );
        if (packed.repacked) {
          this.stats.rowPitchRepackCount += 1;
        }
        if (packed.copiedFromSab) {
          this.stats.sabCopyCount += 1;
        }

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
          return {
            repacked: packed.repacked,
            copiedFromSab: packed.copiedFromSab,
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
          return {
            repacked: false,
            copiedFromSab: packed.copiedFromSab,
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
        return {
          repacked: packed.repacked,
          copiedFromSab: packed.copiedFromSab,
          bytesPerRow: packed.bytesPerRow,
          originalBytesPerRow: rawBytesPerRow,
          stagingBuffer,
          staged: true,
        };
      }

      // GPU-buffer-only copy path (no CPU bytes supplied).
      const rawBytesPerRow = Number.isInteger(source.bytesPerRow) ? source.bytesPerRow : 0;
      if (rawBytesPerRow === 0 || rawBytesPerRow % WEBGPU_BYTES_PER_ROW_ALIGNMENT === 0) {
        encoder.copyBufferToTexture(source, destination, normSize);
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
        this.stats.rowPitchRepackCount += 1;
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

      if (
        p.stagingPool &&
        !p.stagingPool._destroyed &&
        typeof p.stagingPool.stageTextureUpload === 'function' &&
        width > 0 &&
        height > 0 &&
        dataLen > 0
      ) {
        const effectiveRowsPerImage = rowsPerImage > 0 ? rowsPerImage : height;
        const totalRows = effectiveRowsPerImage * depthOrArrayLayers;
        const effectiveBytesPerRow =
          bytesPerRow > 0
            ? bytesPerRow
            : totalRows > 0 && dataLen % totalRows === 0
              ? dataLen / totalRows
              : 0;
        const canStage =
          effectiveBytesPerRow > 0 && dataLen >= effectiveBytesPerRow * totalRows;

        if (canStage) {
          const encoder =
            typeof p._ensurePendingEncoder === 'function' ? p._ensurePendingEncoder() : null;
          if (encoder && typeof encoder.copyBufferToTexture === 'function') {
            const packed = repackRowsTo256(
              rawData,
              effectiveBytesPerRow,
              effectiveRowsPerImage,
              depthOrArrayLayers
            );
            if (packed.repacked) {
              this.stats.rowPitchRepackCount += 1;
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
      }
      if (rowsPerImage > 0) {
        dataLayout.rowsPerImage = rowsPerImage;
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
        }
      );
    }

    _create(id, descriptor, isV2 = true) {
      const p = this.processor;
      this._bc3dFallbackTextures.delete(id);
      if (typeof p._destroyTextureIfPresent === 'function') {
        p._destroyTextureIfPresent(id);
      }
      // Cached v2 views of a previous texture with this id must not survive.
      if (this.next && typeof this.next.destroyResource === 'function') {
        this.next.destroyResource(id);
      }

      let effectiveDescriptor = descriptor;
      let bc3dFallback = false;
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

  function installTexturePackets(processor) {
    const handler = new TexturePackets(processor, processor.extension || null);
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
