'use strict';

// Splash / Bink2 frame presentation bridge.
// Zero external dependencies; UMD export for Web Workers (globalThis.PlayGta6SplashFrameBridge)
// and Node.js (module.exports). Queues decoded RGBA8 / YUV420P / YUVA420P / NV12 / ImageBitmap
// frames into BinkVideoPresenter (W38/W115/W165) for letterboxed WebGPU presentation with
// Bink2Scarlett YVU (Y/Cr/Cb) plane-order forwarding, underlay loadOp control, optional
// zero-copy 4K typed-array handoff, and FIRST_PRESENT telemetry.

(function (root) {
  const DEFAULT_MAX_QUEUE_DEPTH = 3;
  const DEFAULT_PRESENTATION_FORMAT = 'bgra8unorm';
  const DEFAULT_TARGET_FPS = 30;
  const DEFAULT_MIN_VISIBLE_LUMA = 8;
  const DEFAULT_MIN_VISIBLE_PIXELS = 4;
  const DEFAULT_NONBLACK_GRID_MAX_DIM = 64;

  const SPLASH_FRAME_SOURCE = Object.freeze({
    RAW: 'raw',
    EXTERNAL: 'external',
    DIAGNOSTIC: 'diagnostic',
  });

  const PIXEL_FORMAT = Object.freeze({
    RGBA8: 'rgba8',
    BGRA8: 'bgra8',
    YUV420P: 'yuv420p',
    YUVA420P: 'yuva420p',
    NV12: 'nv12',
  });

  // Matches BinkVideoPresenter PIXEL_FORMAT_MODE numeric IDs.
  const PIXEL_FORMAT_ID = Object.freeze({
    0: PIXEL_FORMAT.RGBA8,
    1: PIXEL_FORMAT.YUV420P,
    2: PIXEL_FORMAT.YUVA420P,
    3: PIXEL_FORMAT.NV12,
  });

  // Strict allowlist for sanitized `getStats().lastFrameFormat` telemetry tokens.
  const ALLOWED_FRAME_FORMATS = Object.freeze({
    RGBA8: 'rgba8',
    YUV420P: 'yuv420p',
    NV12: 'nv12',
    IMAGEBITMAP: 'imagebitmap',
  });

  const COLOR_SPACE = Object.freeze({
    BT709: 'bt709',
    BT601: 'bt601',
  });

  const COLOR_RANGE = Object.freeze({
    LIMITED: 'limited',
    FULL: 'full',
  });

  const CHROMA_ORDER = Object.freeze({
    YUV: 'yuv',
    YVU: 'yvu',
  });

  const LOAD_OP = Object.freeze({
    CLEAR: 'clear',
    LOAD: 'load',
  });

  const TEXTURE_USAGE = Object.freeze({
    COPY_DST:
      typeof GPUTextureUsage !== 'undefined' && typeof GPUTextureUsage.COPY_DST === 'number'
        ? GPUTextureUsage.COPY_DST
        : 0x02,
    TEXTURE_BINDING:
      typeof GPUTextureUsage !== 'undefined' &&
      typeof GPUTextureUsage.TEXTURE_BINDING === 'number'
        ? GPUTextureUsage.TEXTURE_BINDING
        : 0x04,
    RENDER_ATTACHMENT:
      typeof GPUTextureUsage !== 'undefined' &&
      typeof GPUTextureUsage.RENDER_ATTACHMENT === 'number'
        ? GPUTextureUsage.RENDER_ATTACHMENT
        : 0x10,
  });

  // Original procedural color-bar palette (not derived from any copyrighted media).
  const DIAGNOSTIC_BARS = Object.freeze([
    Object.freeze([235, 235, 235]),
    Object.freeze([235, 235, 16]),
    Object.freeze([16, 235, 235]),
    Object.freeze([16, 235, 16]),
    Object.freeze([235, 16, 235]),
    Object.freeze([235, 16, 16]),
    Object.freeze([16, 16, 235]),
    Object.freeze([16, 16, 16]),
  ]);

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

  function toUint8Array(input, label) {
    if (input instanceof Uint8Array) {
      return input;
    }
    if (ArrayBuffer.isView(input)) {
      return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
    }
    if (input instanceof ArrayBuffer) {
      return new Uint8Array(input);
    }
    if (typeof SharedArrayBuffer !== 'undefined' && input instanceof SharedArrayBuffer) {
      return new Uint8Array(input);
    }
    throw new TypeError(
      `${label} expects a Uint8Array, ArrayBufferView, ArrayBuffer, or SharedArrayBuffer`
    );
  }

  function normalizePixelFormat(format) {
    if (typeof format === 'number') {
      if (Object.prototype.hasOwnProperty.call(PIXEL_FORMAT_ID, format)) {
        return PIXEL_FORMAT_ID[format];
      }
      throw new RangeError(`Unsupported pixel format "${format}"`);
    }
    if (typeof format === 'string') {
      const normalized = format.trim().toLowerCase();
      if (normalized === 'bgra8unorm') {
        return PIXEL_FORMAT.BGRA8;
      }
      if (
        normalized === PIXEL_FORMAT.RGBA8 ||
        normalized === PIXEL_FORMAT.BGRA8 ||
        normalized === PIXEL_FORMAT.YUV420P ||
        normalized === PIXEL_FORMAT.YUVA420P ||
        normalized === PIXEL_FORMAT.NV12
      ) {
        return normalized;
      }
    }
    throw new RangeError(`Unsupported pixel format "${format}"`);
  }

  function normalizeColorSpace(colorSpace) {
    if (colorSpace === undefined || colorSpace === null || colorSpace === '') {
      return null;
    }
    if (typeof colorSpace === 'string') {
      const key = colorSpace.trim().toLowerCase();
      if (key === COLOR_SPACE.BT709 || key === '709' || key === 'rec709' || key === 'bt.709') {
        return COLOR_SPACE.BT709;
      }
      if (key === COLOR_SPACE.BT601 || key === '601' || key === 'rec601' || key === 'bt.601') {
        return COLOR_SPACE.BT601;
      }
    }
    throw new RangeError(
      `Unsupported colorSpace "${colorSpace}"; expected "${COLOR_SPACE.BT709}" or "${COLOR_SPACE.BT601}"`
    );
  }

  function normalizeColorRange(colorRange) {
    if (colorRange === undefined || colorRange === null || colorRange === '') {
      return null;
    }
    if (typeof colorRange === 'string') {
      const key = colorRange.trim().toLowerCase();
      if (key === COLOR_RANGE.LIMITED || key === 'tv' || key === 'studio' || key === 'mpeg') {
        return COLOR_RANGE.LIMITED;
      }
      if (key === COLOR_RANGE.FULL || key === 'pc' || key === 'jpeg') {
        return COLOR_RANGE.FULL;
      }
    }
    throw new RangeError(
      `Unsupported colorRange "${colorRange}"; expected "${COLOR_RANGE.LIMITED}" or "${COLOR_RANGE.FULL}"`
    );
  }

  function normalizeChromaOrder(chromaOrder) {
    if (chromaOrder === undefined || chromaOrder === null || chromaOrder === '') {
      return null;
    }
    if (typeof chromaOrder === 'string') {
      const key = chromaOrder.trim().toLowerCase();
      if (key === CHROMA_ORDER.YUV) {
        return CHROMA_ORDER.YUV;
      }
      if (key === CHROMA_ORDER.YVU) {
        return CHROMA_ORDER.YVU;
      }
    }
    throw new RangeError(
      `Unsupported chromaOrder "${chromaOrder}"; expected "${CHROMA_ORDER.YUV}" or "${CHROMA_ORDER.YVU}"`
    );
  }

  function normalizeLoadOp(loadOp) {
    if (loadOp === undefined || loadOp === null || loadOp === '') {
      return null;
    }
    if (typeof loadOp === 'string') {
      const key = loadOp.trim().toLowerCase();
      if (key === LOAD_OP.CLEAR) {
        return LOAD_OP.CLEAR;
      }
      if (key === LOAD_OP.LOAD) {
        return LOAD_OP.LOAD;
      }
    }
    throw new RangeError(
      `Unsupported loadOp "${loadOp}"; expected "${LOAD_OP.CLEAR}" or "${LOAD_OP.LOAD}"`
    );
  }

  function sanitizeChromaOrder(chromaOrder) {
    if (typeof chromaOrder !== 'string') {
      return null;
    }
    const token = chromaOrder.trim().toLowerCase();
    if (token === CHROMA_ORDER.YUV || token === CHROMA_ORDER.YVU) {
      return token;
    }
    return null;
  }

  function sanitizeLoadOp(loadOp) {
    if (typeof loadOp !== 'string') {
      return null;
    }
    const token = loadOp.trim().toLowerCase();
    if (token === LOAD_OP.CLEAR || token === LOAD_OP.LOAD) {
      return token;
    }
    return null;
  }

  function sanitizeCounter(value) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
      return 0;
    }
    return Math.floor(value);
  }

  function sanitizeLuma(value) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
      return 0;
    }
    if (value >= 255) {
      return 255;
    }
    return Math.floor(value);
  }

  function normalizeMinVisibleLuma(value, fallback = DEFAULT_MIN_VISIBLE_LUMA) {
    if (value === undefined || value === null) {
      return fallback;
    }
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 255) {
      throw new RangeError(`minVisibleLuma must be a finite number between 0 and 255, got ${value}`);
    }
    return value;
  }

  function normalizePositiveIntOption(value, fallback, name) {
    if (value === undefined || value === null) {
      return fallback;
    }
    assertPositiveInteger(value, name);
    return value;
  }

  function sanitizeSource(source) {
    if (
      source === SPLASH_FRAME_SOURCE.RAW ||
      source === SPLASH_FRAME_SOURCE.EXTERNAL ||
      source === SPLASH_FRAME_SOURCE.DIAGNOSTIC
    ) {
      return source;
    }
    return null;
  }

  /**
   * Sanitize a frame format value into the strict allowlisted telemetry token set:
   * `'rgba8' | 'yuv420p' | 'nv12' | 'imagebitmap' | null`.
   *
   * @param {unknown} format
   * @returns {'rgba8' | 'yuv420p' | 'nv12' | 'imagebitmap' | null}
   */
  function sanitizeFrameFormat(format) {
    if (format === null || format === undefined) {
      return null;
    }
    if (typeof format === 'number') {
      if (format === 0) return ALLOWED_FRAME_FORMATS.RGBA8;
      if (format === 1 || format === 2) return ALLOWED_FRAME_FORMATS.YUV420P;
      if (format === 3) return ALLOWED_FRAME_FORMATS.NV12;
      return null;
    }
    if (typeof format !== 'string') {
      return null;
    }
    const token = format.trim().toLowerCase();
    if (
      token === 'rgba8' ||
      token === 'rgba' ||
      token === 'rgba8unorm' ||
      token === 'bgra8' ||
      token === 'bgra' ||
      token === 'bgra8unorm'
    ) {
      return ALLOWED_FRAME_FORMATS.RGBA8;
    }
    if (token === 'yuv420p' || token === 'yuv420' || token === 'yuva420p' || token === 'yuva420') {
      return ALLOWED_FRAME_FORMATS.YUV420P;
    }
    if (token === 'nv12') {
      return ALLOWED_FRAME_FORMATS.NV12;
    }
    if (token === 'imagebitmap' || token === 'image_bitmap' || token === 'image-bitmap') {
      return ALLOWED_FRAME_FORMATS.IMAGEBITMAP;
    }
    return null;
  }

  function resolveOptionalStride(primaryValue, fallbackValue, name) {
    const val = primaryValue !== undefined ? primaryValue : fallbackValue;
    if (val === undefined || val === null) {
      return null;
    }
    assertPositiveInteger(val, name);
    return val;
  }

  function resolveZeroCopyOption(frame, defaultZeroCopy) {
    if (frame && frame.zeroCopy !== undefined && frame.zeroCopy !== null) {
      return Boolean(frame.zeroCopy);
    }
    if (frame && frame.copyPlanes !== undefined && frame.copyPlanes !== null) {
      return !Boolean(frame.copyPlanes);
    }
    if (frame && frame.clonePlanes !== undefined && frame.clonePlanes !== null) {
      return !Boolean(frame.clonePlanes);
    }
    return defaultZeroCopy;
  }

  function acquirePlaneBytes(input, label, zeroCopy) {
    const view = toUint8Array(input, label);
    return zeroCopy ? view : new Uint8Array(view);
  }

  function resolveBinkVideoPresenterCtor(injected) {
    if (typeof injected === 'function') {
      return injected;
    }
    if (
      typeof globalThis !== 'undefined' &&
      globalThis.PlayGta6BinkPresenter &&
      typeof globalThis.PlayGta6BinkPresenter.BinkVideoPresenter === 'function'
    ) {
      return globalThis.PlayGta6BinkPresenter.BinkVideoPresenter;
    }
    if (typeof importScripts === 'function') {
      try {
        importScripts('./bink_presenter.js');
      } catch (_) {
        // Optional helper unavailable in this worker environment.
      }
      if (
        typeof globalThis !== 'undefined' &&
        globalThis.PlayGta6BinkPresenter &&
        typeof globalThis.PlayGta6BinkPresenter.BinkVideoPresenter === 'function'
      ) {
        return globalThis.PlayGta6BinkPresenter.BinkVideoPresenter;
      }
    }
    if (typeof require === 'function') {
      try {
        const mod = require('./bink_presenter.js');
        if (mod && typeof mod.BinkVideoPresenter === 'function') {
          return mod.BinkVideoPresenter;
        }
      } catch (_) {
        // Optional helper module unavailable.
      }
    }
    throw new Error(
      'BinkVideoPresenter unavailable; load ./bink_presenter.js or inject presenter / BinkVideoPresenter'
    );
  }

  /**
   * Generate a pure original RGBA8 diagnostic splash frame (color bars + animated gradient).
   * No game media, copyrighted assets, or external images are used.
   *
   * @param {number} width
   * @param {number} height
   * @param {number} [frameIndex=0]
   * @returns {{
   *   width: number,
   *   height: number,
   *   format: string,
   *   planes: { rgba: Uint8Array },
   *   frameIndex: number,
   *   source: string,
   * }}
   */
  function generateDiagnosticSplashFrame(width, height, frameIndex = 0) {
    assertPositiveInteger(width, 'width');
    assertPositiveInteger(height, 'height');
    assertNonNegativeInteger(frameIndex, 'frameIndex');

    const rgba = new Uint8Array(width * height * 4);
    const barHeight = Math.max(1, Math.floor((height * 2) / 3));
    const barCount = DIAGNOSTIC_BARS.length;
    const phase = frameIndex % 256;

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const offset = (y * width + x) * 4;
        let r;
        let g;
        let b;

        if (y < barHeight) {
          const barIndex = Math.min(barCount - 1, Math.floor((x * barCount) / width));
          const bar = DIAGNOSTIC_BARS[barIndex];
          r = bar[0];
          g = bar[1];
          b = bar[2];
        } else {
          const t = width <= 1 ? 0 : x / (width - 1);
          const wave = ((y + phase) * 3) & 0xff;
          r = Math.min(255, Math.floor(16 + t * 200 + (wave & 31)));
          g = Math.min(255, Math.floor(32 + (1 - t) * 180));
          b = Math.min(255, Math.floor(48 + ((phase + x) & 0x7f)));
        }

        rgba[offset] = r;
        rgba[offset + 1] = g;
        rgba[offset + 2] = b;
        rgba[offset + 3] = 255;
      }
    }

    return {
      width,
      height,
      format: PIXEL_FORMAT.RGBA8,
      planes: { rgba },
      frameIndex,
      source: SPLASH_FRAME_SOURCE.DIAGNOSTIC,
    };
  }

  function extractRgbaFromExternal(image, width, height) {
    if (!image || typeof image !== 'object') {
      throw new TypeError('pushExternalImage expects an ImageBitmap-like object');
    }

    if (image.data !== undefined && image.data !== null) {
      const bytes = toUint8Array(image.data, 'externalImage.data');
      const needed = width * height * 4;
      if (bytes.byteLength < needed) {
        throw new RangeError(
          `external image data too small (${bytes.byteLength} bytes; need ${needed})`
        );
      }
      return bytes.byteLength === needed ? new Uint8Array(bytes) : new Uint8Array(bytes.subarray(0, needed));
    }

    if (
      typeof OffscreenCanvas !== 'undefined' &&
      typeof OffscreenCanvas === 'function' &&
      typeof image.width === 'number'
    ) {
      try {
        const canvas = new OffscreenCanvas(width, height);
        const ctx = canvas.getContext('2d');
        if (ctx && typeof ctx.drawImage === 'function' && typeof ctx.getImageData === 'function') {
          ctx.drawImage(image, 0, 0, width, height);
          const imageData = ctx.getImageData(0, 0, width, height);
          return new Uint8Array(imageData.data);
        }
      } catch (_) {
        // Fall through to empty buffer; copyExternalImageToTexture may still succeed.
      }
    }

    return null;
  }

  /**
   * Fast, zero-allocation deterministic strided pixel-luminance check (W307).
   * Samples up to `maxGridDim x maxGridDim` (default 64x64 = 4096) grid points in O(1)
   * time without allocating temporary buffers.
   *
   * Supports RGBA8 (`Uint8Array` / `Uint8ClampedArray` / `ImageData`), optional BGRA8
   * (`bgra: true` or `format: 'bgra8'`), and single-channel Y8 luma planes (`bytesPerPixel: 1`
   * with optional `colorRange: 'limited'` studio-range expansion).
   *
   * @param {Uint8Array|Uint8ClampedArray|ArrayBufferView|ArrayBuffer|{ data: Uint8Array|Uint8ClampedArray, width?: number, height?: number }|null|undefined} rgbaOrImageData
   * @param {number|object} [width]
   * @param {number} [height]
   * @param {{
   *   minVisibleLuma?: number,
   *   minVisiblePixels?: number,
   *   maxGridDim?: number,
   *   strideBytes?: number|null,
   *   bytesPerPixel?: number,
   *   format?: string,
   *   bgra?: boolean,
   *   colorRange?: 'limited'|'full'|null,
   *   respectAlpha?: boolean,
   *   earlyExit?: boolean,
   *   out?: { visible: boolean, maxLuma: number, visiblePixels: number, sampledPixels: number }|null,
   * }} [options]
   * @returns {boolean}
   */
  function hasVisibleNonBlackPixels(rgbaOrImageData, width, height, options) {
    let pixels = null;
    let w = width;
    let h = height;
    let opts = options;

    if (
      rgbaOrImageData !== null &&
      rgbaOrImageData !== undefined &&
      typeof rgbaOrImageData === 'object' &&
      !ArrayBuffer.isView(rgbaOrImageData) &&
      !(rgbaOrImageData instanceof ArrayBuffer) &&
      (typeof SharedArrayBuffer === 'undefined' || !(rgbaOrImageData instanceof SharedArrayBuffer))
    ) {
      // ImageData-like object `{ data, width, height }`
      if (typeof width === 'object' && width !== null && height === undefined) {
        opts = width;
        w = rgbaOrImageData.width;
        h = rgbaOrImageData.height;
      } else {
        if (w === undefined || w === null) {
          w = rgbaOrImageData.width;
        }
        if (h === undefined || h === null) {
          h = rgbaOrImageData.height;
        }
      }
      pixels = rgbaOrImageData.data || null;
    } else {
      pixels = rgbaOrImageData;
    }

    const out = opts && typeof opts === 'object' && opts.out && typeof opts.out === 'object'
      ? opts.out
      : null;

    if (out) {
      out.visible = false;
      out.maxLuma = 0;
      out.visiblePixels = 0;
      out.sampledPixels = 0;
    }

    const minVisibleLuma = normalizeMinVisibleLuma(
      opts && opts.minVisibleLuma,
      DEFAULT_MIN_VISIBLE_LUMA
    );
    const minVisiblePixels = normalizePositiveIntOption(
      opts && opts.minVisiblePixels,
      DEFAULT_MIN_VISIBLE_PIXELS,
      'minVisiblePixels'
    );
    const maxGridDim = normalizePositiveIntOption(
      opts && opts.maxGridDim,
      DEFAULT_NONBLACK_GRID_MAX_DIM,
      'maxGridDim'
    );

    if (
      !pixels ||
      typeof w !== 'number' ||
      typeof h !== 'number' ||
      !Number.isFinite(w) ||
      !Number.isFinite(h)
    ) {
      return false;
    }

    const frameWidth = Math.floor(w);
    const frameHeight = Math.floor(h);
    if (frameWidth < 1 || frameHeight < 1) {
      return false;
    }

    // Zero-alloc fast path for Uint8Array / Uint8ClampedArray (ImageData.data).
    let byteView;
    if (pixels instanceof Uint8Array || pixels instanceof Uint8ClampedArray) {
      byteView = pixels;
    } else if (
      ArrayBuffer.isView(pixels) ||
      pixels instanceof ArrayBuffer ||
      (typeof SharedArrayBuffer !== 'undefined' && pixels instanceof SharedArrayBuffer)
    ) {
      byteView = toUint8Array(pixels, 'rgbaOrImageData');
    } else {
      return false;
    }

    const len = byteView.length;
    if (len === 0) {
      return false;
    }

    const bpp =
      opts && (opts.bytesPerPixel === 1 || opts.format === 'y8' || opts.format === 'luma')
        ? 1
        : 4;
    const isBgra = Boolean(
      bpp === 4 &&
        opts &&
        (opts.bgra === true ||
          (typeof opts.format === 'string' &&
            (opts.format === 'bgra8' || opts.format === 'bgra8unorm')))
    );
    const respectAlpha = Boolean(bpp === 4 && opts && opts.respectAlpha === true);
    const limitedRange = Boolean(
      bpp === 1 && opts && opts.colorRange === COLOR_RANGE.LIMITED
    );

    const activeRowBytes = frameWidth * bpp;
    let rowStride = activeRowBytes;
    if (opts && opts.strideBytes !== undefined && opts.strideBytes !== null) {
      assertPositiveInteger(opts.strideBytes, 'strideBytes');
      if (opts.strideBytes < activeRowBytes) {
        throw new RangeError(
          `strideBytes (${opts.strideBytes}) must be >= width * bytesPerPixel (${activeRowBytes})`
        );
      }
      rowStride = opts.strideBytes;
    }

    const earlyExit =
      opts && opts.earlyExit !== undefined ? Boolean(opts.earlyExit) : !out;

    const stepX = frameWidth <= maxGridDim ? 1 : Math.ceil(frameWidth / maxGridDim);
    const stepY = frameHeight <= maxGridDim ? 1 : Math.ceil(frameHeight / maxGridDim);
    const stepBytes = stepX * bpp;

    let maxLuma = 0;
    let visiblePixels = 0;
    let sampledPixels = 0;
    let visible = false;

    if (bpp === 1) {
      for (let y = 0; y < frameHeight; y += stepY) {
        const rowStart = y * rowStride;
        if (rowStart >= len) {
          break;
        }
        const maxRowEnd = rowStart + activeRowBytes;
        const rowEnd = maxRowEnd <= len ? maxRowEnd : len;
        for (let offset = rowStart; offset < rowEnd; offset += stepBytes) {
          const rawY = byteView[offset];
          const luma = limitedRange
            ? rawY <= 16
              ? 0
              : rawY >= 235
                ? 255
                : ((rawY - 16) * 298 + 128) >> 8
            : rawY;
          sampledPixels += 1;
          if (luma > maxLuma) {
            maxLuma = luma;
          }
          if (luma >= minVisibleLuma) {
            visiblePixels += 1;
            if (visiblePixels >= minVisiblePixels) {
              visible = true;
              if (earlyExit) {
                if (out) {
                  out.visible = true;
                  out.maxLuma = maxLuma;
                  out.visiblePixels = visiblePixels;
                  out.sampledPixels = sampledPixels;
                }
                return true;
              }
            }
          }
        }
      }
    } else {
      const tailPad = respectAlpha ? 3 : 2;
      for (let y = 0; y < frameHeight; y += stepY) {
        const rowStart = y * rowStride;
        if (rowStart >= len) {
          break;
        }
        const maxRowEnd = rowStart + activeRowBytes;
        const rowEnd = maxRowEnd <= len ? maxRowEnd : len;
        const limit = rowEnd - tailPad;
        for (let offset = rowStart; offset < limit; offset += stepBytes) {
          const c0 = byteView[offset];
          const g = byteView[offset + 1];
          const c2 = byteView[offset + 2];
          const r = isBgra ? c2 : c0;
          const b = isBgra ? c0 : c2;
          // Exact integer Rec.709 luma (54 + 183 + 19 = 256 -> uniform gray (v,v,v) maps to exact v).
          let luma = (54 * r + 183 * g + 19 * b + 128) >> 8;
          if (respectAlpha) {
            const a = byteView[offset + 3];
            if (a === 0) {
              luma = 0;
            } else if (a < 255) {
              luma = ((luma * a + 127) / 255) | 0;
            }
          }
          sampledPixels += 1;
          if (luma > maxLuma) {
            maxLuma = luma;
          }
          if (luma >= minVisibleLuma) {
            visiblePixels += 1;
            if (visiblePixels >= minVisiblePixels) {
              visible = true;
              if (earlyExit) {
                if (out) {
                  out.visible = true;
                  out.maxLuma = maxLuma;
                  out.visiblePixels = visiblePixels;
                  out.sampledPixels = sampledPixels;
                }
                return true;
              }
            }
          }
        }
      }
    }

    if (out) {
      out.visible = visible;
      out.maxLuma = maxLuma;
      out.visiblePixels = visiblePixels;
      out.sampledPixels = sampledPixels;
    }
    return visible;
  }

  /**
   * Bounded splash/Bink2 frame queue that feeds `BinkVideoPresenter` and encodes letterboxed presents.
   */
  class SplashFrameBridge {
    constructor(options = {}) {
      if (options !== undefined && options !== null && typeof options !== 'object') {
        throw new TypeError('SplashFrameBridge constructor expects an options object');
      }
      const opts = options || {};

      const maxQueueDepth =
        opts.maxQueueDepth !== undefined ? opts.maxQueueDepth : DEFAULT_MAX_QUEUE_DEPTH;
      assertPositiveInteger(maxQueueDepth, 'maxQueueDepth');

      if (opts.onFirstPresent !== undefined && opts.onFirstPresent !== null) {
        if (typeof opts.onFirstPresent !== 'function') {
          throw new TypeError('onFirstPresent must be a function when provided');
        }
      }
      if (opts.onFirstVisibleFrame !== undefined && opts.onFirstVisibleFrame !== null) {
        if (typeof opts.onFirstVisibleFrame !== 'function') {
          throw new TypeError('onFirstVisibleFrame must be a function when provided');
        }
      }
      const rawProbe =
        opts.pixelProbe !== undefined ? opts.pixelProbe : opts.canvasProbe;
      if (rawProbe !== undefined && rawProbe !== null && typeof rawProbe !== 'function') {
        throw new TypeError('pixelProbe / canvasProbe must be a function when provided');
      }

      this._maxQueueDepth = maxQueueDepth;
      this._onFirstPresent = opts.onFirstPresent || null;
      this._onFirstVisibleFrame = opts.onFirstVisibleFrame || null;
      this._pixelProbe = rawProbe || null;
      this._requireNonBlackPixels =
        opts.requireNonBlackPixels !== undefined
          ? Boolean(opts.requireNonBlackPixels)
          : Boolean(this._pixelProbe);
      this._minVisibleLuma = normalizeMinVisibleLuma(
        opts.minVisibleLuma,
        DEFAULT_MIN_VISIBLE_LUMA
      );
      this._minVisiblePixels = normalizePositiveIntOption(
        opts.minVisiblePixels,
        DEFAULT_MIN_VISIBLE_PIXELS,
        'minVisiblePixels'
      );
      this._maxGridDim = normalizePositiveIntOption(
        opts.maxGridDim,
        DEFAULT_NONBLACK_GRID_MAX_DIM,
        'maxGridDim'
      );
      this._presentationFormat = opts.presentationFormat || DEFAULT_PRESENTATION_FORMAT;
      this._stagingPool = opts.stagingPool || null;
      this._colorSpace =
        opts.colorSpace !== undefined && opts.colorSpace !== null
          ? normalizeColorSpace(opts.colorSpace)
          : COLOR_SPACE.BT709;
      this._colorRange =
        opts.colorRange !== undefined && opts.colorRange !== null
          ? normalizeColorRange(opts.colorRange)
          : COLOR_RANGE.LIMITED;
      const normalizedChromaOrder = normalizeChromaOrder(opts.chromaOrder);
      const presenterChromaOrder =
        opts.presenter && typeof opts.presenter === 'object'
          ? sanitizeChromaOrder(opts.presenter.chromaOrder)
          : null;
      this._chromaOrder = normalizedChromaOrder || presenterChromaOrder || CHROMA_ORDER.YUV;
      const normalizedLoadOp = normalizeLoadOp(opts.loadOp);
      const presenterLoadOp =
        opts.presenter && typeof opts.presenter === 'object'
          ? sanitizeLoadOp(opts.presenter.loadOp)
          : null;
      this._loadOp = normalizedLoadOp || presenterLoadOp || LOAD_OP.CLEAR;
      this._zeroCopy = resolveZeroCopyOption(opts, false);
      this._targetFps = opts.targetFps || DEFAULT_TARGET_FPS;
      this._BinkVideoPresenterCtor = opts.BinkVideoPresenter || null;
      this._presenterOptions = opts.presenterOptions || null;
      this._suppressSplashAfterFirstPresent =
        opts.suppressSplashAfterFirstPresent !== undefined
          ? Boolean(opts.suppressSplashAfterFirstPresent)
          : opts.suppressAfterFirstPresent !== undefined
            ? Boolean(opts.suppressAfterFirstPresent)
            : true;

      this._queue = [];
      this._framesPresented = 0;
      this._framesBridged = 0;
      this._droppedFrames = 0;
      this._droppedStaleFrames = 0;
      this._bitmapsClosed = 0;
      this._releaseCount = 0;
      this._firstPresentSent = false;
      this._firstVisibleFrameSent = false;
      this._guestFirstPresent = false;
      this._splashSuppressed = Boolean(opts.suppressSplash);
      this._framesBeforeFirstPresent = 0;
      this._lateSplashFramesIgnored = 0;
      this._suppressedBlackFrames = 0;
      this._lastSampledMaxLuma = 0;
      this._lumaProbeScratch = {
        visible: false,
        maxLuma: 0,
        visiblePixels: 0,
        sampledPixels: 0,
      };
      this._lumaProbeOpts = {
        minVisibleLuma: this._minVisibleLuma,
        minVisiblePixels: this._minVisiblePixels,
        maxGridDim: this._maxGridDim,
        strideBytes: null,
        bytesPerPixel: 4,
        format: null,
        bgra: false,
        colorRange: null,
        respectAlpha: false,
        earlyExit: false,
        out: this._lumaProbeScratch,
      };
      this._nv12FramesForwarded = 0;
      this._yvuFramesForwarded = 0;
      this._underlayLoadFramesPresented = 0;
      this._zeroCopyFramesForwarded = 0;
      this._lastFrameFormat = null;
      this._lastChromaOrder = null;
      this._lastLoadOp = null;
      this._lastFrameWidth = 0;
      this._lastFrameHeight = 0;
      this._lastSource = null;
      this._nextFrameIndex = 0;
      this._destroyed = false;
      this._resourcesReleased = false;
      this._externalScratchTextures = [];
      this._retainedBitmaps = new Set();
      this._closedBitmaps = typeof WeakSet === 'function' ? new WeakSet() : new Set();

      if (opts.presenter) {
        if (
          typeof opts.presenter !== 'object' ||
          (typeof opts.presenter.uploadRgbaFrame !== 'function' &&
            typeof opts.presenter.uploadFrame !== 'function')
        ) {
          throw new TypeError(
            'presenter must provide uploadRgbaFrame / uploadFrame / presentFrame APIs'
          );
        }
        this._presenter = opts.presenter;
        this._ownsPresenter = false;
      } else {
        this._presenter = null;
        this._ownsPresenter = true;
        this._pendingDevice = opts.device || null;
      }
    }

    get maxQueueDepth() {
      return this._maxQueueDepth;
    }

    get isDestroyed() {
      return this._destroyed;
    }

    get presenter() {
      return this._presenter;
    }

    get guestFirstPresent() {
      return this._guestFirstPresent;
    }

    get splashSuppressed() {
      return this._splashSuppressed;
    }

    get chromaOrder() {
      return this._chromaOrder;
    }

    get loadOp() {
      return this._loadOp;
    }

    get zeroCopy() {
      return this._zeroCopy;
    }

    get requireNonBlackPixels() {
      return this._requireNonBlackPixels;
    }

    get minVisibleLuma() {
      return this._minVisibleLuma;
    }

    get minVisiblePixels() {
      return this._minVisiblePixels;
    }

    get firstVisibleFrameSent() {
      return this._firstVisibleFrameSent;
    }

    get suppressedBlackFrames() {
      return sanitizeCounter(this._suppressedBlackFrames);
    }

    get lastSampledMaxLuma() {
      return sanitizeLuma(this._lastSampledMaxLuma);
    }

    /**
     * Switch default chroma plane ordering (`'yuv'` for Y/Cb/Cr or `'yvu'` for Bink2Scarlett Y/Cr/Cb).
     *
     * @param {'yuv' | 'yvu'} chromaOrder
     */
    setChromaOrder(chromaOrder) {
      this._assertNotDestroyed();
      const normalized = normalizeChromaOrder(chromaOrder);
      if (!normalized) {
        throw new RangeError(
          `Unsupported chromaOrder "${chromaOrder}"; expected "${CHROMA_ORDER.YUV}" or "${CHROMA_ORDER.YVU}"`
        );
      }
      this._chromaOrder = normalized;
      if (this._presenter && typeof this._presenter.setChromaOrder === 'function') {
        this._presenter.setChromaOrder(normalized);
      }
      return this._chromaOrder;
    }

    /**
     * Switch default render-pass `loadOp` (`'clear'` or `'load'` for menu/UI underlay composition).
     *
     * @param {'clear' | 'load'} loadOp
     */
    setLoadOp(loadOp) {
      this._assertNotDestroyed();
      const normalized = normalizeLoadOp(loadOp);
      if (!normalized) {
        throw new RangeError(
          `Unsupported loadOp "${loadOp}"; expected "${LOAD_OP.CLEAR}" or "${LOAD_OP.LOAD}"`
        );
      }
      this._loadOp = normalized;
      if (this._presenter && typeof this._presenter.setLoadOp === 'function') {
        this._presenter.setLoadOp(normalized);
      }
      return this._loadOp;
    }

    /**
     * Switch presentation swapchain format (e.g. `bgra8unorm`, `bgra8unorm-srgb`, `rgba16float`)
     * and propagate it to the underlying presenter so sRGB/linear output stays exact.
     *
     * @param {string} presentationFormat
     */
    setPresentationFormat(presentationFormat) {
      this._assertNotDestroyed();
      if (typeof presentationFormat !== 'string' || presentationFormat.trim().length === 0) {
        throw new TypeError('presentationFormat must be a non-empty string');
      }
      const normalized = presentationFormat.trim();
      this._presentationFormat = normalized;
      if (this._presenter && typeof this._presenter.setPresentationFormat === 'function') {
        this._presenter.setPresentationFormat(normalized);
      }
      return this._presentationFormat;
    }

    _assertNotDestroyed() {
      if (this._destroyed) {
        throw new Error('SplashFrameBridge has been destroyed');
      }
    }

    _ensurePresenter(device) {
      if (this._presenter) {
        return this._presenter;
      }
      const resolvedDevice = device || this._pendingDevice;
      if (!resolvedDevice || typeof resolvedDevice.createTexture !== 'function') {
        throw new TypeError(
          'SplashFrameBridge requires a WebGPU device to construct BinkVideoPresenter'
        );
      }
      const Ctor = resolveBinkVideoPresenterCtor(this._BinkVideoPresenterCtor);
      const ctorOptions = {
        device: resolvedDevice,
        presentationFormat: this._presentationFormat,
        stagingPool: this._stagingPool,
        colorSpace: this._colorSpace,
        colorRange: this._colorRange,
        chromaOrder: this._chromaOrder,
        loadOp: this._loadOp,
        targetFps: this._targetFps,
        ...(this._presenterOptions || {}),
      };
      this._presenter = new Ctor(ctorOptions);
      this._ownsPresenter = true;
      this._pendingDevice = resolvedDevice;
      return this._presenter;
    }

    _drainQueueAsIgnored() {
      let drained = 0;
      while (this._queue.length > 0) {
        const dropped = this._queue.shift();
        this._releaseEntryResources(dropped);
        this._lateSplashFramesIgnored += 1;
        drained += 1;
      }
      return drained;
    }

    _ignoreLateSplashFrame(frameIndex) {
      this._lateSplashFramesIgnored += 1;
      return {
        queueDepth: this._queue.length,
        droppedFrames: this._droppedFrames,
        frameIndex,
        ignored: true,
        reason: 'suppressed',
        lateSplashFramesIgnored: this._lateSplashFramesIgnored,
      };
    }

    _resetLumaScratch() {
      const s = this._lumaProbeScratch;
      s.visible = false;
      s.maxLuma = 0;
      s.visiblePixels = 0;
      s.sampledPixels = 0;
      return s;
    }

    _configureLumaProbeOpts(overrideOpts, defaults) {
      const p = this._lumaProbeOpts;
      p.minVisibleLuma =
        overrideOpts && overrideOpts.minVisibleLuma !== undefined
          ? normalizeMinVisibleLuma(overrideOpts.minVisibleLuma, this._minVisibleLuma)
          : this._minVisibleLuma;
      p.minVisiblePixels =
        overrideOpts && overrideOpts.minVisiblePixels !== undefined
          ? normalizePositiveIntOption(
              overrideOpts.minVisiblePixels,
              this._minVisiblePixels,
              'minVisiblePixels'
            )
          : this._minVisiblePixels;
      p.maxGridDim =
        overrideOpts && overrideOpts.maxGridDim !== undefined
          ? normalizePositiveIntOption(
              overrideOpts.maxGridDim,
              this._maxGridDim,
              'maxGridDim'
            )
          : this._maxGridDim;
      p.strideBytes =
        overrideOpts && overrideOpts.strideBytes !== undefined
          ? overrideOpts.strideBytes
          : defaults && defaults.strideBytes !== undefined
            ? defaults.strideBytes
            : null;
      p.bytesPerPixel =
        defaults && defaults.bytesPerPixel !== undefined ? defaults.bytesPerPixel : 4;
      p.format =
        overrideOpts && overrideOpts.format !== undefined
          ? overrideOpts.format
          : defaults && defaults.format !== undefined
            ? defaults.format
            : null;
      p.bgra = Boolean(overrideOpts && overrideOpts.bgra === true);
      p.colorRange =
        defaults && defaults.colorRange !== undefined ? defaults.colorRange : null;
      p.respectAlpha = Boolean(overrideOpts && overrideOpts.respectAlpha === true);
      p.earlyExit = false;
      p.out = this._lumaProbeScratch;
      return p;
    }

    _evaluateProbeOutput(result, fallbackWidth, fallbackHeight, overrideOpts) {
      const scratch = this._resetLumaScratch();
      if (result === null || result === undefined) {
        return null;
      }
      if (typeof result === 'boolean') {
        scratch.visible = result;
        scratch.maxLuma = result ? 255 : 0;
        scratch.visiblePixels = result ? this._minVisiblePixels : 0;
        scratch.sampledPixels = 1;
        return scratch;
      }
      if (typeof result === 'object') {
        if (typeof result.visible === 'boolean' || typeof result.hasVisiblePixels === 'boolean') {
          const vis =
            typeof result.visible === 'boolean' ? result.visible : Boolean(result.hasVisiblePixels);
          scratch.visible = vis;
          scratch.maxLuma = sanitizeLuma(
            result.maxLuma !== undefined ? result.maxLuma : vis ? 255 : 0
          );
          scratch.visiblePixels = sanitizeCounter(
            result.visiblePixels !== undefined
              ? result.visiblePixels
              : vis
                ? this._minVisiblePixels
                : 0
          );
          scratch.sampledPixels = sanitizeCounter(
            result.sampledPixels !== undefined ? result.sampledPixels : 1
          );
          return scratch;
        }
        if (
          result instanceof Uint8Array ||
          result instanceof Uint8ClampedArray ||
          ArrayBuffer.isView(result) ||
          result instanceof ArrayBuffer ||
          (typeof SharedArrayBuffer !== 'undefined' && result instanceof SharedArrayBuffer)
        ) {
          const probeOpts = this._configureLumaProbeOpts(overrideOpts, null);
          hasVisibleNonBlackPixels(result, fallbackWidth, fallbackHeight, probeOpts);
          return scratch;
        }
        if (result.data !== undefined && result.data !== null) {
          const w = result.width !== undefined ? result.width : fallbackWidth;
          const h = result.height !== undefined ? result.height : fallbackHeight;
          const probeOpts = this._configureLumaProbeOpts(overrideOpts, {
            strideBytes: result.strideBytes !== undefined ? result.strideBytes : null,
          });
          hasVisibleNonBlackPixels(result.data, w, h, probeOpts);
          return scratch;
        }
      }
      return null;
    }

    _probeFrameDescriptorNonBlack(frame, overrideOpts) {
      const scratch = this._resetLumaScratch();
      if (!frame || typeof frame !== 'object') {
        return scratch;
      }
      const w = frame.width;
      const h = frame.height;
      const planes = frame.planes || null;
      const fmt = frame.format;

      if (
        fmt === PIXEL_FORMAT.RGBA8 ||
        fmt === PIXEL_FORMAT.BGRA8 ||
        fmt === 0 ||
        (!fmt && (frame.rgba || frame.bgra || (planes && (planes.rgba || planes.bgra))))
      ) {
        const isBgra = fmt === PIXEL_FORMAT.BGRA8 || Boolean(frame.bgra || (planes && planes.bgra));
        const rgba =
          (planes && (planes.bgra || planes.rgba)) || frame.bgra || frame.rgba || frame.data || null;
        if (rgba) {
          const probeOpts = this._configureLumaProbeOpts(overrideOpts, {
            bytesPerPixel: 4,
            strideBytes: frame.strideBytes !== undefined ? frame.strideBytes : null,
            format: isBgra ? 'bgra8' : 'rgba8',
          });
          hasVisibleNonBlackPixels(rgba, w, h, probeOpts);
          return scratch;
        }
      }

      if (
        fmt === PIXEL_FORMAT.NV12 ||
        fmt === PIXEL_FORMAT.YUV420P ||
        fmt === PIXEL_FORMAT.YUVA420P ||
        fmt === 1 ||
        fmt === 2 ||
        fmt === 3
      ) {
        const yPlane = (planes && planes.y) || frame.yPlane || frame.y || null;
        if (yPlane) {
          const probeOpts = this._configureLumaProbeOpts(overrideOpts, {
            bytesPerPixel: 1,
            strideBytes: frame.yStride !== undefined ? frame.yStride : null,
            colorRange: frame.colorRange || this._colorRange,
          });
          hasVisibleNonBlackPixels(yPlane, w, h, probeOpts);
          return scratch;
        }
      }

      return scratch;
    }

    _probeFrameNonBlack(frame, opts) {
      const scratch = this._resetLumaScratch();

      // 1. Explicit per-call pixel/ImageData probe input takes highest precedence.
      if (opts) {
        const explicitPixels =
          opts.pixelData !== undefined
            ? opts.pixelData
            : opts.imageData !== undefined
              ? opts.imageData
              : opts.rgba !== undefined
                ? opts.rgba
                : opts.pixels;
        if (explicitPixels !== undefined && explicitPixels !== null) {
          const w = opts.width !== undefined ? opts.width : frame.width;
          const h = opts.height !== undefined ? opts.height : frame.height;
          const evaluated = this._evaluateProbeOutput(explicitPixels, w, h, opts);
          if (evaluated) {
            return evaluated;
          }
        }
      }

      // 2. Caller-provided pixelProbe / canvasProbe callback (per-call or constructor).
      const probeFn =
        (opts &&
          (typeof opts.pixelProbe === 'function'
            ? opts.pixelProbe
            : typeof opts.canvasProbe === 'function'
              ? opts.canvasProbe
              : null)) ||
        this._pixelProbe;
      if (typeof probeFn === 'function') {
        try {
          const probeOutput = probeFn(frame, opts || null);
          const evaluated = this._evaluateProbeOutput(
            probeOutput,
            frame.width,
            frame.height,
            opts
          );
          if (evaluated) {
            return evaluated;
          }
        } catch (_) {
          return this._resetLumaScratch();
        }
      }

      // 3. Direct zero-alloc probe of the frame's own RGBA or Y luma plane.
      return this._probeFrameDescriptorNonBlack(frame, opts);
    }

    _probeHandoffNonBlack(opts) {
      const scratch = this._resetLumaScratch();

      const explicitPixels =
        opts.pixelData !== undefined
          ? opts.pixelData
          : opts.imageData !== undefined
            ? opts.imageData
            : opts.rgba !== undefined
              ? opts.rgba
              : opts.pixels;
      if (explicitPixels !== undefined && explicitPixels !== null) {
        const w =
          opts.width !== undefined
            ? opts.width
            : explicitPixels.width !== undefined
              ? explicitPixels.width
              : this._lastFrameWidth;
        const h =
          opts.height !== undefined
            ? opts.height
            : explicitPixels.height !== undefined
              ? explicitPixels.height
              : this._lastFrameHeight;
        const evaluated = this._evaluateProbeOutput(explicitPixels, w, h, opts);
        if (evaluated) {
          return evaluated;
        }
      }

      if (opts.frame && typeof opts.frame === 'object') {
        return this._probeFrameDescriptorNonBlack(opts.frame, opts);
      }

      const probeFn =
        (typeof opts.pixelProbe === 'function'
          ? opts.pixelProbe
          : typeof opts.canvasProbe === 'function'
            ? opts.canvasProbe
            : null) || this._pixelProbe;
      if (typeof probeFn === 'function') {
        try {
          const probeOutput = probeFn(opts, null);
          const w = opts.width !== undefined ? opts.width : this._lastFrameWidth;
          const h = opts.height !== undefined ? opts.height : this._lastFrameHeight;
          const evaluated = this._evaluateProbeOutput(probeOutput, w, h, opts);
          if (evaluated) {
            return evaluated;
          }
        } catch (_) {
          return this._resetLumaScratch();
        }
      }

      return scratch;
    }

    /**
     * Signal that the live guest engine has reached `FIRST_PRESENT` handoff.
     * Freezes `framesBeforeFirstPresent` and activates splash suppression unless disabled.
     * When suppression activates, also releases retained splash `ImageBitmap`s, transient
     * pixel buffers, and external scratch textures so splash GPU/CPU memory cannot leak.
     *
     * When `requireNonBlackPixels` is active (via constructor, `options.requireNonBlackPixels`,
     * or supplied pixel/probe data), suppresses premature splash dismissal and `first-visible-frame`
     * emission on all-black frames until a frame passes `hasVisibleNonBlackPixels`.
     *
     * @param {{
     *   suppress?: boolean,
     *   requireNonBlackPixels?: boolean,
     *   rgba?: Uint8Array|Uint8ClampedArray,
     *   pixels?: Uint8Array|Uint8ClampedArray,
     *   pixelData?: Uint8Array|Uint8ClampedArray|object,
     *   imageData?: { data: Uint8Array|Uint8ClampedArray, width: number, height: number },
     *   frame?: object,
     *   width?: number,
     *   height?: number,
     *   minVisibleLuma?: number,
     *   minVisiblePixels?: number,
     *   pixelProbe?: Function,
     *   canvasProbe?: Function,
     * }} [options]
     */
    notifyGuestFirstPresent(options = {}) {
      this._assertNotDestroyed();
      if (options !== undefined && options !== null && typeof options !== 'object') {
        throw new TypeError('notifyGuestFirstPresent options must be an object');
      }
      const opts = options || {};
      const hasExplicitProbe = Boolean(
        opts.rgba !== undefined ||
          opts.pixels !== undefined ||
          opts.pixelData !== undefined ||
          opts.imageData !== undefined ||
          opts.frame !== undefined ||
          typeof opts.pixelProbe === 'function' ||
          typeof opts.canvasProbe === 'function'
      );
      const checkNonBlack =
        opts.requireNonBlackPixels !== undefined
          ? Boolean(opts.requireNonBlackPixels)
          : this._requireNonBlackPixels || hasExplicitProbe;

      let sampledMaxLuma = 0;
      let sampledVisiblePixels = 0;
      let sampledTotalPixels = 0;

      if (checkNonBlack && !this._guestFirstPresent) {
        const probe = this._probeHandoffNonBlack(opts);
        sampledMaxLuma = probe.maxLuma;
        sampledVisiblePixels = probe.visiblePixels;
        sampledTotalPixels = probe.sampledPixels;
        this._lastSampledMaxLuma = sampledMaxLuma;

        if (!probe.visible) {
          this._suppressedBlackFrames += 1;
          return {
            guestFirstPresent: this._guestFirstPresent,
            splashSuppressed: this._splashSuppressed,
            firstVisibleFrame: false,
            firstVisibleFrameEmitted: false,
            firstVisibleFrameSent: this._firstVisibleFrameSent,
            suppressedBlackFrame: true,
            suppressedBlackFrames: sanitizeCounter(this._suppressedBlackFrames),
            lastSampledMaxLuma: sanitizeLuma(this._lastSampledMaxLuma),
            reason: 'black-frame-suppressed',
            framesBeforeFirstPresent: this._framesBeforeFirstPresent,
            lateSplashFramesIgnored: this._lateSplashFramesIgnored,
            drainedFrames: 0,
            bitmapsClosed: this._bitmapsClosed,
            releaseCount: this._releaseCount,
            chromaOrder: sanitizeChromaOrder(this._chromaOrder) || CHROMA_ORDER.YUV,
            loadOp: sanitizeLoadOp(this._loadOp) || LOAD_OP.CLEAR,
            lastChromaOrder: sanitizeChromaOrder(this._lastChromaOrder),
            lastLoadOp: sanitizeLoadOp(this._lastLoadOp),
          };
        }
      }

      this._guestFirstPresent = true;
      const shouldSuppress =
        opts.suppress !== undefined
          ? Boolean(opts.suppress)
          : this._suppressSplashAfterFirstPresent;
      let drainedFrames = 0;
      if (shouldSuppress) {
        this._splashSuppressed = true;
        drainedFrames = this._drainQueueAsIgnored();
        this._releaseSplashResourcesInternal();
      }

      let firstVisibleFrameEmitted = false;
      if (!this._firstVisibleFrameSent) {
        this._firstVisibleFrameSent = true;
        firstVisibleFrameEmitted = true;
        if (typeof this._onFirstVisibleFrame === 'function') {
          this._onFirstVisibleFrame({
            type: 'first-visible-frame',
            source: 'guest-handoff',
            width: sanitizeCounter(
              opts.width ||
                (opts.imageData && opts.imageData.width) ||
                (opts.frame && opts.frame.width) ||
                this._lastFrameWidth
            ),
            height: sanitizeCounter(
              opts.height ||
                (opts.imageData && opts.imageData.height) ||
                (opts.frame && opts.frame.height) ||
                this._lastFrameHeight
            ),
            firstVisibleFrame: true,
            guestFirstPresent: true,
            splashSuppressed: this._splashSuppressed,
            suppressedBlackFrames: sanitizeCounter(this._suppressedBlackFrames),
            lastSampledMaxLuma: sanitizeLuma(this._lastSampledMaxLuma),
            sampledMaxLuma,
            visiblePixels: sampledVisiblePixels,
            sampledPixels: sampledTotalPixels,
          });
        }
      }

      return {
        guestFirstPresent: this._guestFirstPresent,
        splashSuppressed: this._splashSuppressed,
        firstVisibleFrame: this._firstVisibleFrameSent,
        firstVisibleFrameEmitted,
        firstVisibleFrameSent: this._firstVisibleFrameSent,
        suppressedBlackFrame: false,
        suppressedBlackFrames: sanitizeCounter(this._suppressedBlackFrames),
        lastSampledMaxLuma: sanitizeLuma(this._lastSampledMaxLuma),
        reason: 'ok',
        framesBeforeFirstPresent: this._framesBeforeFirstPresent,
        lateSplashFramesIgnored: this._lateSplashFramesIgnored,
        drainedFrames,
        bitmapsClosed: this._bitmapsClosed,
        releaseCount: this._releaseCount,
        chromaOrder: sanitizeChromaOrder(this._chromaOrder) || CHROMA_ORDER.YUV,
        loadOp: sanitizeLoadOp(this._loadOp) || LOAD_OP.CLEAR,
        lastChromaOrder: sanitizeChromaOrder(this._lastChromaOrder),
        lastLoadOp: sanitizeLoadOp(this._lastLoadOp),
      };
    }

    /**
     * Explicitly enable or disable post-handoff splash frame suppression.
     *
     * @param {boolean} suppressed
     */
    setSplashSuppressed(suppressed) {
      this._assertNotDestroyed();
      const next = Boolean(suppressed);
      this._splashSuppressed = next;
      const drainedFrames = next ? this._drainQueueAsIgnored() : 0;
      if (next) {
        this._closeRemainingRetainedBitmaps();
        this._destroyExternalScratchTextures();
      }
      return {
        splashSuppressed: this._splashSuppressed,
        lateSplashFramesIgnored: this._lateSplashFramesIgnored,
        drainedFrames,
        bitmapsClosed: this._bitmapsClosed,
      };
    }

    _enqueue(entry) {
      this._resourcesReleased = false;
      while (this._queue.length >= this._maxQueueDepth) {
        const dropped = this._queue.shift();
        this._droppedFrames += 1;
        this._droppedStaleFrames += 1;
        this._releaseEntryResources(dropped);
      }
      this._queue.push(entry);
      return {
        queueDepth: this._queue.length,
        droppedFrames: this._droppedFrames,
        droppedStaleFrames: this._droppedStaleFrames,
        frameIndex: entry.frameIndex,
        chromaOrder: entry.chromaOrder || this._chromaOrder,
        loadOp: entry.loadOp || this._loadOp,
        zeroCopy: Boolean(entry.zeroCopy),
      };
    }

    _closeBitmap(bmp) {
      if (!bmp || typeof bmp !== 'object') {
        return false;
      }
      this._retainedBitmaps.delete(bmp);
      if (this._closedBitmaps.has(bmp)) {
        return false;
      }
      if (typeof bmp.close !== 'function') {
        return false;
      }
      this._closedBitmaps.add(bmp);
      try {
        bmp.close();
        this._bitmapsClosed += 1;
        return true;
      } catch (_) {
        // ImageBitmap may already be closed or detached; swallow safely.
        return false;
      }
    }

    _closeRemainingRetainedBitmaps() {
      let closed = 0;
      if (this._retainedBitmaps.size === 0) {
        return 0;
      }
      const snapshot = Array.from(this._retainedBitmaps);
      this._retainedBitmaps.clear();
      for (let i = 0; i < snapshot.length; i += 1) {
        if (this._closeBitmap(snapshot[i])) {
          closed += 1;
        }
      }
      return closed;
    }

    _destroyExternalScratchTextures() {
      if (this._externalScratchTextures.length === 0) {
        return 0;
      }
      const textures = this._externalScratchTextures;
      this._externalScratchTextures = [];
      let destroyed = 0;
      for (let i = 0; i < textures.length; i += 1) {
        const tex = textures[i];
        textures[i] = null;
        if (tex && typeof tex.destroy === 'function') {
          try {
            tex.destroy();
            destroyed += 1;
          } catch (_) {
            // Scratch texture may already be destroyed.
          }
        }
      }
      return destroyed;
    }

    _releaseEntryResources(entry, options = {}) {
      if (!entry || typeof entry !== 'object') {
        return { bitmapsClosed: 0, buffersReleased: 0 };
      }
      const closeExternalSource =
        options && options.closeExternalSource !== undefined
          ? Boolean(options.closeExternalSource)
          : true;
      let bitmapsClosed = 0;
      let buffersReleased = 0;

      const img = entry.image;
      const ext = entry.externalSource;
      entry.image = null;
      entry.externalSource = null;

      if (img && this._closeBitmap(img)) {
        bitmapsClosed += 1;
      }
      if (ext && ext !== img) {
        if (closeExternalSource) {
          if (this._closeBitmap(ext)) {
            bitmapsClosed += 1;
          }
        } else {
          this._retainedBitmaps.delete(ext);
        }
      }

      if (entry.planes && typeof entry.planes === 'object') {
        const keys = Object.keys(entry.planes);
        for (let i = 0; i < keys.length; i += 1) {
          const key = keys[i];
          if (entry.planes[key] !== null && entry.planes[key] !== undefined) {
            entry.planes[key] = null;
            buffersReleased += 1;
          }
        }
        entry.planes = null;
      }

      return { bitmapsClosed, buffersReleased };
    }

    _releaseQueuedExternal(entry) {
      return this._releaseEntryResources(entry);
    }

    _releaseSplashResourcesInternal() {
      const hasPendingWork =
        this._queue.length > 0 ||
        this._retainedBitmaps.size > 0 ||
        this._externalScratchTextures.length > 0;
      if (this._resourcesReleased && !hasPendingWork) {
        return {
          released: false,
          framesReleased: 0,
          bitmapsClosed: 0,
          buffersReleased: 0,
          texturesDestroyed: 0,
          releaseCount: this._releaseCount,
          queueDepth: 0,
        };
      }

      let framesReleased = 0;
      let bitmapsClosed = 0;
      let buffersReleased = 0;

      while (this._queue.length > 0) {
        const entry = this._queue.shift();
        const res = this._releaseEntryResources(entry);
        framesReleased += 1;
        bitmapsClosed += res.bitmapsClosed;
        buffersReleased += res.buffersReleased;
      }

      bitmapsClosed += this._closeRemainingRetainedBitmaps();
      const texturesDestroyed = this._destroyExternalScratchTextures();

      this._resourcesReleased = true;
      this._releaseCount += 1;

      return {
        released: true,
        framesReleased,
        bitmapsClosed,
        buffersReleased,
        texturesDestroyed,
        releaseCount: this._releaseCount,
        queueDepth: this._queue.length,
      };
    }

    /**
     * Explicitly close all retained splash `ImageBitmap`s, null transient splash pixel buffers,
     * and destroy external scratch GPU textures. Safe and idempotent across repeated calls or
     * after `dispose()` / `destroy()`.
     */
    releaseSplashResources() {
      return this._releaseSplashResourcesInternal();
    }

    /**
     * Push a raw planar/semi-planar/RGBA frame into the bounded queue.
     * Supports RGBA8 (`'rgba8' | 0`), YUV420P (`'yuv420p' | 1`), YUVA420P (`'yuva420p' | 2`),
     * and NV12 (`'nv12' | 3`, with `yPlane`/`uvPlane` or `planes: { y, uv }`).
     * Supports Bink2Scarlett `chromaOrder: 'yvu'` (Y/Cr/Cb) via either `{ y, u: cr, v: cb }` or
     * `{ y, cr, cb }`, per-frame `loadOp` (`'clear' | 'load'`), and `zeroCopy: true` to preserve
     * caller typed-array views without copying 4K payloads.
     *
     * @param {{
     *   width: number,
     *   height: number,
     *   format: string|number,
     *   planes?: {
     *     rgba?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *     y?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *     u?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *     v?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *     cr?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *     cb?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *     uv?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *     yPlane?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *     uPlane?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *     vPlane?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *     crPlane?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *     cbPlane?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *     uvPlane?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *     a?: Uint8Array|ArrayBufferView|ArrayBuffer|null,
     *   },
     *   yPlane?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *   uPlane?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *   vPlane?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *   crPlane?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *   cbPlane?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *   uvPlane?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *   y?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *   u?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *   v?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *   cr?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *   cb?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *   uv?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *   a?: Uint8Array|ArrayBufferView|ArrayBuffer|null,
     *   rgba?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *   yStride?: number,
     *   uvStride?: number,
     *   uStride?: number,
     *   vStride?: number,
     *   crStride?: number,
     *   cbStride?: number,
     *   aStride?: number,
     *   strideBytes?: number,
     *   timestampMs?: number,
     *   colorSpace?: string,
     *   colorRange?: string,
     *   chromaOrder?: 'yuv'|'yvu',
     *   loadOp?: 'clear'|'load',
     *   zeroCopy?: boolean,
     *   copyPlanes?: boolean,
     *   clonePlanes?: boolean,
     *   source?: string,
     *   frameIndex?: number,
     * }} frame
     */
    pushRawFrame(frame) {
      this._assertNotDestroyed();
      if (!frame || typeof frame !== 'object') {
        throw new TypeError('pushRawFrame expects a frame descriptor object');
      }

      const { width, height, timestampMs } = frame;
      assertPositiveInteger(width, 'width');
      assertPositiveInteger(height, 'height');
      const format = normalizePixelFormat(frame.format);
      const colorSpace = normalizeColorSpace(frame.colorSpace);
      const colorRange = normalizeColorRange(frame.colorRange);
      const explicitChromaOrder = normalizeChromaOrder(frame.chromaOrder);
      const chromaOrder = explicitChromaOrder || this._chromaOrder;
      const loadOp = normalizeLoadOp(frame.loadOp);
      const zeroCopy = resolveZeroCopyOption(frame, this._zeroCopy);

      if (frame.planes !== undefined && frame.planes !== null && typeof frame.planes !== 'object') {
        throw new TypeError('pushRawFrame requires a planes object when planes is provided');
      }
      const planes = frame.planes || null;

      const frameIndex =
        frame.frameIndex !== undefined ? frame.frameIndex : this._nextFrameIndex;
      assertNonNegativeInteger(frameIndex, 'frameIndex');
      this._nextFrameIndex = Math.max(this._nextFrameIndex, frameIndex + 1);

      if (this._splashSuppressed) {
        return this._ignoreLateSplashFrame(frameIndex);
      }

      const source =
        frame.source === SPLASH_FRAME_SOURCE.DIAGNOSTIC
          ? SPLASH_FRAME_SOURCE.DIAGNOSTIC
          : SPLASH_FRAME_SOURCE.RAW;

      let ownedPlanes;
      let yStride = null;
      let uvStride = null;
      let uStride = null;
      let vStride = null;
      let aStride = null;
      let strideBytes = null;

      if (format === PIXEL_FORMAT.RGBA8 || format === PIXEL_FORMAT.BGRA8) {
        const isBgra = format === PIXEL_FORMAT.BGRA8;
        const pixelInput =
          planes && planes.bgra !== undefined
            ? planes.bgra
            : planes && planes.rgba !== undefined
              ? planes.rgba
              : frame.bgra !== undefined
                ? frame.bgra
                : frame.rgba !== undefined
                  ? frame.rgba
                  : frame.data;
        if (pixelInput === undefined || pixelInput === null) {
          throw new TypeError(
            isBgra ? 'BGRA8 frames require planes.bgra or planes.rgba' : 'RGBA8 frames require planes.rgba'
          );
        }
        ownedPlanes = isBgra
          ? { bgra: acquirePlaneBytes(pixelInput, 'planes.bgra', zeroCopy) }
          : { rgba: acquirePlaneBytes(pixelInput, 'planes.rgba', zeroCopy) };
        strideBytes = resolveOptionalStride(
          frame.strideBytes,
          planes && planes.strideBytes,
          'strideBytes'
        );
      } else if (format === PIXEL_FORMAT.NV12) {
        const yInput =
          planes && planes.y !== undefined
            ? planes.y
            : planes && planes.yPlane !== undefined
              ? planes.yPlane
              : frame.yPlane !== undefined
                ? frame.yPlane
                : frame.y;
        const uvInput =
          planes && planes.uv !== undefined
            ? planes.uv
            : planes && planes.uvPlane !== undefined
              ? planes.uvPlane
              : frame.uvPlane !== undefined
                ? frame.uvPlane
                : frame.uv;
        if (yInput === undefined || yInput === null || uvInput === undefined || uvInput === null) {
          throw new TypeError('NV12 frames require yPlane/planes.y and uvPlane/planes.uv');
        }
        ownedPlanes = {
          y: acquirePlaneBytes(yInput, 'yPlane', zeroCopy),
          uv: acquirePlaneBytes(uvInput, 'uvPlane', zeroCopy),
        };
        yStride = resolveOptionalStride(frame.yStride, planes && planes.yStride, 'yStride');
        uvStride = resolveOptionalStride(frame.uvStride, planes && planes.uvStride, 'uvStride');
      } else {
        const yInput =
          planes && planes.y !== undefined
            ? planes.y
            : planes && planes.yPlane !== undefined
              ? planes.yPlane
              : frame.yPlane !== undefined
                ? frame.yPlane
                : frame.y;
        const rawUInput =
          planes && planes.u !== undefined
            ? planes.u
            : planes && planes.uPlane !== undefined
              ? planes.uPlane
              : frame.uPlane !== undefined
                ? frame.uPlane
                : frame.u;
        const rawVInput =
          planes && planes.v !== undefined
            ? planes.v
            : planes && planes.vPlane !== undefined
              ? planes.vPlane
              : frame.vPlane !== undefined
                ? frame.vPlane
                : frame.v;
        const rawCrInput =
          planes && planes.cr !== undefined
            ? planes.cr
            : planes && planes.crPlane !== undefined
              ? planes.crPlane
              : frame.crPlane !== undefined
                ? frame.crPlane
                : frame.cr;
        const rawCbInput =
          planes && planes.cb !== undefined
            ? planes.cb
            : planes && planes.cbPlane !== undefined
              ? planes.cbPlane
              : frame.cbPlane !== undefined
                ? frame.cbPlane
                : frame.cb;

        let uInput = rawUInput;
        let vInput = rawVInput;
        let usedCrCbAliases = false;
        if (
          (uInput === undefined || uInput === null || vInput === undefined || vInput === null) &&
          rawCrInput !== undefined &&
          rawCrInput !== null &&
          rawCbInput !== undefined &&
          rawCbInput !== null
        ) {
          usedCrCbAliases = true;
          if (chromaOrder === CHROMA_ORDER.YVU) {
            // Bink2Scarlett native Y/Cr/Cb: first chroma slot (u) holds Cr, second (v) holds Cb.
            uInput = rawCrInput;
            vInput = rawCbInput;
          } else {
            uInput = rawCbInput;
            vInput = rawCrInput;
          }
        }

        if (
          yInput === undefined ||
          yInput === null ||
          uInput === undefined ||
          uInput === null ||
          vInput === undefined ||
          vInput === null
        ) {
          throw new TypeError('YUV frames require planes.y, planes.u, and planes.v');
        }
        ownedPlanes = {
          y: acquirePlaneBytes(yInput, 'planes.y', zeroCopy),
          u: acquirePlaneBytes(uInput, 'planes.u', zeroCopy),
          v: acquirePlaneBytes(vInput, 'planes.v', zeroCopy),
        };
        if (format === PIXEL_FORMAT.YUVA420P) {
          const aInput =
            planes && planes.a !== undefined
              ? planes.a
              : planes && planes.aPlane !== undefined
                ? planes.aPlane
                : frame.aPlane !== undefined
                  ? frame.aPlane
                  : frame.a;
          if (aInput === undefined || aInput === null) {
            throw new TypeError('YUVA420P frames require planes.a');
          }
          ownedPlanes.a = acquirePlaneBytes(aInput, 'planes.a', zeroCopy);
          aStride = resolveOptionalStride(frame.aStride, planes && planes.aStride, 'aStride');
        }
        yStride = resolveOptionalStride(frame.yStride, planes && planes.yStride, 'yStride');
        uvStride = resolveOptionalStride(frame.uvStride, planes && planes.uvStride, 'uvStride');
        uStride = resolveOptionalStride(frame.uStride, planes && planes.uStride, 'uStride');
        vStride = resolveOptionalStride(frame.vStride, planes && planes.vStride, 'vStride');
        if (usedCrCbAliases) {
          const crStride = resolveOptionalStride(
            frame.crStride,
            planes && planes.crStride,
            'crStride'
          );
          const cbStride = resolveOptionalStride(
            frame.cbStride,
            planes && planes.cbStride,
            'cbStride'
          );
          if (uStride === null) {
            uStride = chromaOrder === CHROMA_ORDER.YVU ? crStride : cbStride;
          }
          if (vStride === null) {
            vStride = chromaOrder === CHROMA_ORDER.YVU ? cbStride : crStride;
          }
        }
      }

      return this._enqueue({
        source,
        width,
        height,
        format,
        telemetryFormat: sanitizeFrameFormat(format),
        planes: ownedPlanes,
        yStride,
        uvStride,
        uStride,
        vStride,
        aStride,
        strideBytes,
        timestampMs:
          typeof timestampMs === 'number' && Number.isFinite(timestampMs) ? timestampMs : null,
        colorSpace,
        colorRange,
        chromaOrder,
        loadOp,
        zeroCopy,
        frameIndex,
        image: null,
      });
    }

    /**
     * Convenience wrapper to push a semi-planar NV12 frame into the bounded queue.
     *
     * @param {object} frame
     */
    pushNv12Frame(frame) {
      this._assertNotDestroyed();
      if (!frame || typeof frame !== 'object') {
        throw new TypeError('pushNv12Frame expects a frame descriptor object');
      }
      return this.pushRawFrame({
        ...frame,
        format: PIXEL_FORMAT.NV12,
      });
    }

    /**
     * Queue an external ImageBitmap / canvas-like image for presentation.
     * Uses `queue.copyExternalImageToTexture` when available, otherwise RGBA extraction.
     *
     * @param {ImageBitmap|object} imageBitmap
     * @param {{ width?: number, height?: number, timestampMs?: number, frameIndex?: number }} [options]
     */
    pushExternalImage(imageBitmap, options = {}) {
      this._assertNotDestroyed();
      if (!imageBitmap || typeof imageBitmap !== 'object') {
        throw new TypeError('pushExternalImage expects an ImageBitmap-like object');
      }
      if (options !== undefined && options !== null && typeof options !== 'object') {
        throw new TypeError('pushExternalImage options must be an object');
      }

      const width =
        options && options.width !== undefined
          ? options.width
          : imageBitmap.width !== undefined
            ? imageBitmap.width
            : null;
      const height =
        options && options.height !== undefined
          ? options.height
          : imageBitmap.height !== undefined
            ? imageBitmap.height
            : null;
      assertPositiveInteger(width, 'width');
      assertPositiveInteger(height, 'height');

      const loadOp = normalizeLoadOp(options && options.loadOp);

      const frameIndex =
        options && options.frameIndex !== undefined
          ? options.frameIndex
          : this._nextFrameIndex;
      assertNonNegativeInteger(frameIndex, 'frameIndex');
      this._nextFrameIndex = Math.max(this._nextFrameIndex, frameIndex + 1);

      if (this._splashSuppressed) {
        this._closeBitmap(imageBitmap);
        return this._ignoreLateSplashFrame(frameIndex);
      }

      const timestampMs =
        options && typeof options.timestampMs === 'number' && Number.isFinite(options.timestampMs)
          ? options.timestampMs
          : null;

      // Prefer immediate RGBA extraction so queue ownership does not depend on live bitmaps.
      const extracted = extractRgbaFromExternal(imageBitmap, width, height);
      const takeOwnership = Boolean(
        options && (options.closeOnRelease === true || options.takeOwnership === true)
      );
      this._retainedBitmaps.add(imageBitmap);
      const entry = {
        source: SPLASH_FRAME_SOURCE.EXTERNAL,
        width,
        height,
        format: PIXEL_FORMAT.RGBA8,
        telemetryFormat: ALLOWED_FRAME_FORMATS.IMAGEBITMAP,
        planes: extracted ? { rgba: extracted } : { rgba: null },
        yStride: null,
        uvStride: null,
        uStride: null,
        vStride: null,
        aStride: null,
        strideBytes: null,
        timestampMs,
        colorSpace: null,
        colorRange: null,
        chromaOrder: this._chromaOrder,
        loadOp,
        zeroCopy: false,
        frameIndex,
        image: extracted && !takeOwnership ? null : imageBitmap,
        preferCopyExternal: true,
        externalSource: imageBitmap,
      };

      return this._enqueue(entry);
    }

    /**
     * Generate and enqueue a diagnostic splash frame.
     *
     * @param {number} width
     * @param {number} height
     * @param {number} [frameIndex]
     * @param {{ timestampMs?: number, loadOp?: 'clear'|'load' }} [options]
     */
    pushDiagnosticFrame(width, height, frameIndex, options = {}) {
      this._assertNotDestroyed();
      assertPositiveInteger(width, 'width');
      assertPositiveInteger(height, 'height');
      const loadOp = normalizeLoadOp(options && options.loadOp);
      const index = frameIndex !== undefined ? frameIndex : this._nextFrameIndex;
      assertNonNegativeInteger(index, 'frameIndex');
      if (this._splashSuppressed) {
        this._nextFrameIndex = Math.max(this._nextFrameIndex, index + 1);
        return this._ignoreLateSplashFrame(index);
      }
      const generated = generateDiagnosticSplashFrame(width, height, index);
      return this.pushRawFrame({
        width: generated.width,
        height: generated.height,
        format: generated.format,
        planes: generated.planes,
        timestampMs: options && options.timestampMs,
        loadOp,
        source: SPLASH_FRAME_SOURCE.DIAGNOSTIC,
        frameIndex: generated.frameIndex,
      });
    }

    _findLatestDueIndex(nowMs) {
      if (this._queue.length === 0) {
        return -1;
      }

      const hasTiming = typeof nowMs === 'number' && Number.isFinite(nowMs);
      if (!hasTiming) {
        return 0;
      }

      let latestDueIndex = -1;
      for (let i = 0; i < this._queue.length; i += 1) {
        const ts = this._queue[i].timestampMs;
        if (ts === null || ts === undefined || ts <= nowMs) {
          latestDueIndex = i;
        }
      }
      return latestDueIndex;
    }

    _dequeueLatestDue(nowMs) {
      const latestDueIndex = this._findLatestDueIndex(nowMs);
      if (latestDueIndex < 0) {
        return null;
      }

      // Drop older due frames so playback stays on the freshest ready sample.
      for (let i = 0; i < latestDueIndex; i += 1) {
        const dropped = this._queue.shift();
        this._droppedFrames += 1;
        this._droppedStaleFrames += 1;
        this._releaseEntryResources(dropped);
      }
      return this._queue.shift();
    }

    _uploadExternalViaCopy(device, frame) {
      if (
        !device ||
        !device.queue ||
        typeof device.queue.copyExternalImageToTexture !== 'function' ||
        !frame.externalSource
      ) {
        return false;
      }
      if (typeof device.createTexture !== 'function') {
        return false;
      }

      // Release prior external scratch textures before allocating a replacement.
      this._destroyExternalScratchTextures();

      const texture = device.createTexture({
        label: `playgta6-splash-external-${frame.frameIndex}`,
        size: { width: frame.width, height: frame.height, depthOrArrayLayers: 1 },
        format: 'rgba8unorm',
        usage:
          TEXTURE_USAGE.COPY_DST |
          TEXTURE_USAGE.TEXTURE_BINDING |
          TEXTURE_USAGE.RENDER_ATTACHMENT,
      });
      device.queue.copyExternalImageToTexture(
        { source: frame.externalSource },
        { texture },
        { width: frame.width, height: frame.height, depthOrArrayLayers: 1 }
      );
      this._resourcesReleased = false;
      this._externalScratchTextures.push(texture);
      return texture;
    }

    _uploadFrame(presenter, device, frame) {
      if (frame.format === PIXEL_FORMAT.BGRA8) {
        const bgra = frame.planes && (frame.planes.bgra || frame.planes.rgba);
        if (typeof presenter.uploadBgraFrame === 'function') {
          if (frame.strideBytes !== null && frame.strideBytes !== undefined) {
            return presenter.uploadBgraFrame(bgra, frame.width, frame.height, {
              strideBytes: frame.strideBytes,
            });
          }
          return presenter.uploadBgraFrame(bgra, frame.width, frame.height);
        }
        if (typeof presenter.uploadFrame === 'function') {
          const bgraDesc = {
            format: PIXEL_FORMAT.BGRA8,
            bgra,
            width: frame.width,
            height: frame.height,
          };
          if (frame.strideBytes !== null && frame.strideBytes !== undefined) {
            bgraDesc.strideBytes = frame.strideBytes;
          }
          return presenter.uploadFrame(bgraDesc);
        }
        throw new TypeError('presenter does not support BGRA8 uploads');
      }

      if (frame.format === PIXEL_FORMAT.RGBA8) {
        let externalTexture = null;
        if (frame.source === SPLASH_FRAME_SOURCE.EXTERNAL) {
          externalTexture = this._uploadExternalViaCopy(device, frame) || null;
        }
        let rgba = frame.planes && frame.planes.rgba;
        if (!rgba && frame.externalSource) {
          rgba = extractRgbaFromExternal(frame.externalSource, frame.width, frame.height);
        }
        if (!rgba && externalTexture && typeof presenter.bindExternalTexture === 'function') {
          // Bind the GPU-copied external texture directly so GPU-only ImageBitmaps render
          // their actual pixels instead of a synthetic solid-white fallback.
          return presenter.bindExternalTexture(externalTexture, frame.width, frame.height);
        }
        if (!rgba) {
          // Last-resort solid frame when neither CPU extraction nor presenter.uploadExternalImage is available.
          rgba = new Uint8Array(frame.width * frame.height * 4);
          rgba.fill(255);
        }
        if (typeof presenter.uploadRgbaFrame === 'function') {
          if (frame.strideBytes !== null && frame.strideBytes !== undefined) {
            return presenter.uploadRgbaFrame(rgba, frame.width, frame.height, {
              strideBytes: frame.strideBytes,
            });
          }
          return presenter.uploadRgbaFrame(rgba, frame.width, frame.height);
        }
        if (typeof presenter.uploadFrame === 'function') {
          const rgbaDesc = {
            format: PIXEL_FORMAT.RGBA8,
            rgba,
            width: frame.width,
            height: frame.height,
          };
          if (frame.strideBytes !== null && frame.strideBytes !== undefined) {
            rgbaDesc.strideBytes = frame.strideBytes;
          }
          return presenter.uploadFrame(rgbaDesc);
        }
        throw new TypeError('presenter does not support RGBA8 uploads');
      }

      if (frame.format === PIXEL_FORMAT.NV12) {
        const nv12 = {
          format: PIXEL_FORMAT.NV12,
          y: frame.planes.y,
          uv: frame.planes.uv,
          width: frame.width,
          height: frame.height,
          chromaOrder: frame.chromaOrder || this._chromaOrder,
          colorSpace: frame.colorSpace || this._colorSpace,
          colorRange: frame.colorRange || this._colorRange,
        };
        if (frame.yStride !== null && frame.yStride !== undefined) {
          nv12.yStride = frame.yStride;
        }
        if (frame.uvStride !== null && frame.uvStride !== undefined) {
          nv12.uvStride = frame.uvStride;
        }
        if (typeof presenter.uploadNv12Frame === 'function') {
          return presenter.uploadNv12Frame(nv12);
        }
        if (typeof presenter.uploadFrame === 'function') {
          return presenter.uploadFrame(nv12);
        }
        throw new TypeError(
          'presenter does not support NV12 uploads (missing uploadNv12Frame / uploadFrame)'
        );
      }

      const yuv = {
        format: frame.format,
        y: frame.planes.y,
        u: frame.planes.u,
        v: frame.planes.v,
        width: frame.width,
        height: frame.height,
        chromaOrder: frame.chromaOrder || this._chromaOrder,
        colorSpace: frame.colorSpace || this._colorSpace,
        colorRange: frame.colorRange || this._colorRange,
      };
      if (frame.format === PIXEL_FORMAT.YUVA420P) {
        yuv.a = frame.planes.a;
      }
      if (frame.yStride !== null && frame.yStride !== undefined) {
        yuv.yStride = frame.yStride;
      }
      if (frame.uvStride !== null && frame.uvStride !== undefined) {
        yuv.uvStride = frame.uvStride;
      }
      if (frame.uStride !== null && frame.uStride !== undefined) {
        yuv.uStride = frame.uStride;
      }
      if (frame.vStride !== null && frame.vStride !== undefined) {
        yuv.vStride = frame.vStride;
      }
      if (frame.aStride !== null && frame.aStride !== undefined) {
        yuv.aStride = frame.aStride;
      }
      if (typeof presenter.uploadYuv420Frame === 'function') {
        return presenter.uploadYuv420Frame(yuv);
      }
      if (typeof presenter.uploadFrame === 'function') {
        return presenter.uploadFrame(yuv);
      }
      throw new TypeError(
        'presenter does not support YUV420 uploads (missing uploadYuv420Frame / uploadFrame)'
      );
    }

    /**
     * Dequeue the latest due frame (honoring presenter FPS pacing), upload via the presenter,
     * encode the letterboxed WebGPU render pass, and fire `onFirstPresent` exactly once.
     *
     * @param {GPUDevice} device
     * @param {GPUTextureView} targetView
     * @param {number} targetWidth
     * @param {number} targetHeight
     * @param {number|object} [nowMs]
     * @param {{
     *   loadOp?: 'clear'|'load',
     *   clearColor?: { r: number, g: number, b: number, a: number }|number[],
     *   storeOp?: 'store'|'discard',
     *   opacity?: number,
     * }} [renderOptions]
     */
    renderFrame(device, targetView, targetWidth, targetHeight, nowMs, renderOptions) {
      this._assertNotDestroyed();
      if (!device || typeof device !== 'object') {
        throw new TypeError('renderFrame requires a WebGPU device');
      }
      if (!targetView || typeof targetView !== 'object') {
        throw new TypeError('renderFrame requires a target GPUTextureView');
      }
      assertPositiveInteger(Math.floor(targetWidth), 'targetWidth');
      assertPositiveInteger(Math.floor(targetHeight), 'targetHeight');

      let resolvedNowMs = nowMs;
      let opts = renderOptions;
      if (
        nowMs !== undefined &&
        nowMs !== null &&
        typeof nowMs === 'object' &&
        renderOptions === undefined
      ) {
        opts = nowMs;
        resolvedNowMs = opts.nowMs;
      }
      if (opts !== undefined && opts !== null && typeof opts !== 'object') {
        throw new TypeError('renderFrame options must be an object');
      }
      const renderLoadOp = opts ? normalizeLoadOp(opts.loadOp) : null;

      if (this._splashSuppressed) {
        if (this._queue.length > 0) {
          this._drainQueueAsIgnored();
        }
        return {
          presented: false,
          reason: 'suppressed',
          queueDepth: this._queue.length,
          framesPresented: this._framesPresented,
          framesBeforeFirstPresent: this._framesBeforeFirstPresent,
          lateSplashFramesIgnored: this._lateSplashFramesIgnored,
        };
      }

      const presenter = this._ensurePresenter(device);
      const timestamp =
        typeof resolvedNowMs === 'number' && Number.isFinite(resolvedNowMs)
          ? resolvedNowMs
          : typeof performance !== 'undefined' && typeof performance.now === 'function'
            ? performance.now()
            : Date.now();

      // Peek first so empty queues never consume presenter.shouldAdvance timing slots.
      if (this._findLatestDueIndex(timestamp) < 0) {
        return {
          presented: false,
          reason: 'empty',
          queueDepth: this._queue.length,
          framesPresented: this._framesPresented,
        };
      }

      if (typeof presenter.shouldAdvance === 'function' && !presenter.shouldAdvance(timestamp)) {
        return {
          presented: false,
          reason: 'pacing',
          queueDepth: this._queue.length,
          framesPresented: this._framesPresented,
        };
      }

      const frame = this._dequeueLatestDue(timestamp);
      if (!frame) {
        return {
          presented: false,
          reason: 'empty',
          queueDepth: this._queue.length,
          framesPresented: this._framesPresented,
        };
      }

      const effectiveChromaOrder =
        sanitizeChromaOrder(frame.chromaOrder) ||
        sanitizeChromaOrder(this._chromaOrder) ||
        CHROMA_ORDER.YUV;
      const effectiveLoadOp =
        renderLoadOp ||
        sanitizeLoadOp(frame.loadOp) ||
        sanitizeLoadOp(this._loadOp) ||
        LOAD_OP.CLEAR;

      const hasExplicitProbe = Boolean(
        opts &&
          (opts.pixelData !== undefined ||
            opts.imageData !== undefined ||
            opts.rgba !== undefined ||
            opts.pixels !== undefined ||
            typeof opts.pixelProbe === 'function' ||
            typeof opts.canvasProbe === 'function')
      );
      const checkNonBlack =
        opts && opts.requireNonBlackPixels !== undefined
          ? Boolean(opts.requireNonBlackPixels)
          : this._requireNonBlackPixels || hasExplicitProbe;

      let uploadResult;
      let presentResult;
      let sanitizedFormat;
      let probeChecked = false;
      let frameVisible = true;
      let sampledMaxLuma = 0;
      let sampledVisiblePixels = 0;
      let sampledTotalPixels = 0;
      try {
        // W307: Run non-black visibility probe BEFORE finally releases planes / closes bitmaps.
        if (checkNonBlack) {
          probeChecked = true;
          const probe = this._probeFrameNonBlack(frame, opts);
          frameVisible = probe.visible;
          sampledMaxLuma = probe.maxLuma;
          sampledVisiblePixels = probe.visiblePixels;
          sampledTotalPixels = probe.sampledPixels;
          this._lastSampledMaxLuma = sampledMaxLuma;
        }

        uploadResult = this._uploadFrame(presenter, device, frame);
        this._lastSource = frame.source;
        sanitizedFormat = sanitizeFrameFormat(frame.telemetryFormat || frame.format);
        this._lastFrameFormat = sanitizedFormat;
        this._lastChromaOrder = effectiveChromaOrder;
        this._lastLoadOp = effectiveLoadOp;
        this._lastFrameWidth = frame.width;
        this._lastFrameHeight = frame.height;
        if (frame.format === PIXEL_FORMAT.NV12) {
          this._nv12FramesForwarded += 1;
        }
        if (effectiveChromaOrder === CHROMA_ORDER.YVU) {
          this._yvuFramesForwarded += 1;
        }
        if (frame.zeroCopy) {
          this._zeroCopyFramesForwarded += 1;
        }

        if (typeof device.createCommandEncoder !== 'function') {
          throw new TypeError('renderFrame requires device.createCommandEncoder()');
        }
        const encoder = device.createCommandEncoder({
          label: 'playgta6-splash-frame-bridge-encoder',
        });
        const presentOptions = {
          loadOp: effectiveLoadOp,
        };
        if (opts) {
          if (opts.clearColor !== undefined) {
            presentOptions.clearColor = opts.clearColor;
          }
          if (opts.storeOp !== undefined) {
            presentOptions.storeOp = opts.storeOp;
          }
          if (opts.opacity !== undefined) {
            presentOptions.opacity = opts.opacity;
          }
        }
        presentResult = presenter.presentFrame(
          encoder,
          targetView,
          targetWidth,
          targetHeight,
          presentOptions
        );

        if (device.queue && typeof device.queue.submit === 'function') {
          const commandBuffer =
            encoder && typeof encoder.finish === 'function' ? encoder.finish() : encoder;
          device.queue.submit([commandBuffer]);
        }

        this._framesPresented += 1;
        this._framesBridged += 1;
        if (effectiveLoadOp === LOAD_OP.LOAD) {
          this._underlayLoadFramesPresented += 1;
        }
        if (!this._guestFirstPresent) {
          this._framesBeforeFirstPresent += 1;
        }
      } finally {
        this._releaseEntryResources(frame, { closeExternalSource: false });
      }

      const resolvedChromaOrder =
        sanitizeChromaOrder(effectiveChromaOrder) ||
        (uploadResult && sanitizeChromaOrder(uploadResult.chromaOrder)) ||
        (presentResult && sanitizeChromaOrder(presentResult.chromaOrder)) ||
        CHROMA_ORDER.YUV;
      const resolvedLoadOp =
        sanitizeLoadOp(effectiveLoadOp) ||
        (presentResult && sanitizeLoadOp(presentResult.loadOp)) ||
        LOAD_OP.CLEAR;

      const presentInfo = {
        frameIndex: frame.frameIndex,
        width: frame.width,
        height: frame.height,
        source: frame.source,
        format: sanitizedFormat,
        colorSpace:
          frame.colorSpace || (presentResult && presentResult.colorSpace) || null,
        colorRange:
          frame.colorRange || (presentResult && presentResult.colorRange) || null,
        chromaOrder: resolvedChromaOrder,
        loadOp: resolvedLoadOp,
        zeroCopy: Boolean(frame.zeroCopy),
        timestampMs:
          frame.timestampMs !== null && frame.timestampMs !== undefined
            ? frame.timestampMs
            : timestamp,
      };

      if (!this._firstPresentSent) {
        this._firstPresentSent = true;
        if (typeof this._onFirstPresent === 'function') {
          this._onFirstPresent(presentInfo);
        }
      }

      let firstVisibleFrameEmitted = false;
      let suppressedBlackFrame = false;
      if (!this._firstVisibleFrameSent) {
        if (probeChecked && !frameVisible) {
          this._suppressedBlackFrames += 1;
          suppressedBlackFrame = true;
        } else {
          this._firstVisibleFrameSent = true;
          firstVisibleFrameEmitted = true;
          if (typeof this._onFirstVisibleFrame === 'function') {
            this._onFirstVisibleFrame({
              type: 'first-visible-frame',
              ...presentInfo,
              firstVisibleFrame: true,
              suppressedBlackFrames: sanitizeCounter(this._suppressedBlackFrames),
              lastSampledMaxLuma: sanitizeLuma(this._lastSampledMaxLuma),
              sampledMaxLuma,
              visiblePixels: sampledVisiblePixels,
              sampledPixels: sampledTotalPixels,
            });
          }
        }
      }

      return {
        presented: Boolean(presentResult && presentResult.presented),
        reason: 'ok',
        ...presentInfo,
        viewport: presentResult ? presentResult.viewport : null,
        queueDepth: this._queue.length,
        framesPresented: this._framesPresented,
        framesBridged: this._framesBridged,
        bitmapsClosed: this._bitmapsClosed,
        framesBeforeFirstPresent: this._framesBeforeFirstPresent,
        nv12FramesForwarded: this._nv12FramesForwarded,
        yvuFramesForwarded: this._yvuFramesForwarded,
        underlayLoadFramesPresented: this._underlayLoadFramesPresented,
        zeroCopyFramesForwarded: this._zeroCopyFramesForwarded,
        firstPresentSent: this._firstPresentSent,
        firstVisibleFrame: this._firstVisibleFrameSent,
        firstVisibleFrameEmitted,
        firstVisibleFrameSent: this._firstVisibleFrameSent,
        suppressedBlackFrame,
        suppressedBlackFrames: sanitizeCounter(this._suppressedBlackFrames),
        lastSampledMaxLuma: sanitizeLuma(this._lastSampledMaxLuma),
      };
    }

    /**
     * Telemetry snapshot for HUD / BootTracer FIRST_PRESENT wiring.
     */
    getStats() {
      const chromaOrder = sanitizeChromaOrder(this._chromaOrder) || CHROMA_ORDER.YUV;
      const loadOp = sanitizeLoadOp(this._loadOp) || LOAD_OP.CLEAR;
      return {
        framesPresented: sanitizeCounter(this._framesPresented),
        framesBridged: sanitizeCounter(this._framesBridged),
        droppedFrames: sanitizeCounter(this._droppedFrames),
        droppedStaleFrames: sanitizeCounter(this._droppedStaleFrames),
        bitmapsClosed: sanitizeCounter(this._bitmapsClosed),
        releaseCount: sanitizeCounter(this._releaseCount),
        retainedBitmaps: this._retainedBitmaps.size,
        queueDepth: this._queue.length,
        firstPresentSent: Boolean(this._firstPresentSent),
        firstVisibleFrameSent: Boolean(this._firstVisibleFrameSent),
        guestFirstPresent: Boolean(this._guestFirstPresent),
        splashSuppressed: Boolean(this._splashSuppressed),
        requireNonBlackPixels: Boolean(this._requireNonBlackPixels),
        minVisibleLuma: sanitizeLuma(this._minVisibleLuma),
        minVisiblePixels: sanitizeCounter(this._minVisiblePixels),
        suppressedBlackFrames: sanitizeCounter(this._suppressedBlackFrames),
        lastSampledMaxLuma: sanitizeLuma(this._lastSampledMaxLuma),
        framesBeforeFirstPresent: sanitizeCounter(this._framesBeforeFirstPresent),
        lateSplashFramesIgnored: sanitizeCounter(this._lateSplashFramesIgnored),
        nv12FramesForwarded: sanitizeCounter(this._nv12FramesForwarded),
        yvuFramesForwarded: sanitizeCounter(this._yvuFramesForwarded),
        underlayLoadFramesPresented: sanitizeCounter(this._underlayLoadFramesPresented),
        zeroCopyFramesForwarded: sanitizeCounter(this._zeroCopyFramesForwarded),
        chromaOrder,
        loadOp,
        lastChromaOrder: sanitizeChromaOrder(this._lastChromaOrder),
        lastLoadOp: sanitizeLoadOp(this._lastLoadOp),
        lastFrameWidth: sanitizeCounter(this._lastFrameWidth),
        lastFrameHeight: sanitizeCounter(this._lastFrameHeight),
        lastFrameFormat: sanitizeFrameFormat(this._lastFrameFormat),
        lastSource: sanitizeSource(this._lastSource),
        maxQueueDepth: this._maxQueueDepth,
        zeroCopy: Boolean(this._zeroCopy),
        destroyed: Boolean(this._destroyed),
        presenterStats:
          this._presenter && typeof this._presenter.getStats === 'function'
            ? this._presenter.getStats()
            : null,
      };
    }

    /**
     * Configuration and splash-to-guest handoff state snapshot.
     */
    getState() {
      const chromaOrder = sanitizeChromaOrder(this._chromaOrder) || CHROMA_ORDER.YUV;
      const loadOp = sanitizeLoadOp(this._loadOp) || LOAD_OP.CLEAR;
      return {
        chromaOrder,
        loadOp,
        lastChromaOrder: sanitizeChromaOrder(this._lastChromaOrder),
        lastLoadOp: sanitizeLoadOp(this._lastLoadOp),
        colorSpace: this._colorSpace,
        colorRange: this._colorRange,
        zeroCopy: Boolean(this._zeroCopy),
        maxQueueDepth: this._maxQueueDepth,
        queueDepth: this._queue.length,
        framesPresented: sanitizeCounter(this._framesPresented),
        framesBridged: sanitizeCounter(this._framesBridged),
        droppedFrames: sanitizeCounter(this._droppedFrames),
        droppedStaleFrames: sanitizeCounter(this._droppedStaleFrames),
        bitmapsClosed: sanitizeCounter(this._bitmapsClosed),
        releaseCount: sanitizeCounter(this._releaseCount),
        retainedBitmaps: this._retainedBitmaps.size,
        firstPresentSent: Boolean(this._firstPresentSent),
        firstVisibleFrameSent: Boolean(this._firstVisibleFrameSent),
        guestFirstPresent: Boolean(this._guestFirstPresent),
        splashSuppressed: Boolean(this._splashSuppressed),
        requireNonBlackPixels: Boolean(this._requireNonBlackPixels),
        minVisibleLuma: sanitizeLuma(this._minVisibleLuma),
        minVisiblePixels: sanitizeCounter(this._minVisiblePixels),
        suppressedBlackFrames: sanitizeCounter(this._suppressedBlackFrames),
        lastSampledMaxLuma: sanitizeLuma(this._lastSampledMaxLuma),
        framesBeforeFirstPresent: sanitizeCounter(this._framesBeforeFirstPresent),
        lateSplashFramesIgnored: sanitizeCounter(this._lateSplashFramesIgnored),
        lastFrameFormat: sanitizeFrameFormat(this._lastFrameFormat),
        lastFrameWidth: sanitizeCounter(this._lastFrameWidth),
        lastFrameHeight: sanitizeCounter(this._lastFrameHeight),
        lastSource: sanitizeSource(this._lastSource),
        destroyed: Boolean(this._destroyed),
      };
    }

    /**
     * Reset telemetry and lifecycle counters without clearing queued frames,
     * destroying the presenter, or altering bridge configuration.
     */
    resetStats() {
      this._assertNotDestroyed();
      this._framesPresented = 0;
      this._framesBridged = 0;
      this._droppedFrames = 0;
      this._droppedStaleFrames = 0;
      this._bitmapsClosed = 0;
      this._releaseCount = 0;
      this._framesBeforeFirstPresent = 0;
      this._lateSplashFramesIgnored = 0;
      this._suppressedBlackFrames = 0;
      this._lastSampledMaxLuma = 0;
      this._nv12FramesForwarded = 0;
      this._yvuFramesForwarded = 0;
      this._underlayLoadFramesPresented = 0;
      this._zeroCopyFramesForwarded = 0;
      this._lastFrameFormat = null;
      this._lastChromaOrder = null;
      this._lastLoadOp = null;
      this._lastFrameWidth = 0;
      this._lastFrameHeight = 0;
      this._lastSource = null;
      return this.getStats();
    }

    /**
     * Clear the queue and presentation telemetry so the next presented frame can fire
     * `onFirstPresent` again. Does not destroy the underlying presenter.
     */
    reset() {
      this._assertNotDestroyed();
      while (this._queue.length > 0) {
        this._releaseEntryResources(this._queue.shift());
      }
      this._closeRemainingRetainedBitmaps();
      this._destroyExternalScratchTextures();

      this._framesPresented = 0;
      this._framesBridged = 0;
      this._droppedFrames = 0;
      this._droppedStaleFrames = 0;
      this._bitmapsClosed = 0;
      this._releaseCount = 0;
      this._firstPresentSent = false;
      this._firstVisibleFrameSent = false;
      this._guestFirstPresent = false;
      this._splashSuppressed = false;
      this._resourcesReleased = false;
      this._framesBeforeFirstPresent = 0;
      this._lateSplashFramesIgnored = 0;
      this._suppressedBlackFrames = 0;
      this._lastSampledMaxLuma = 0;
      this._nv12FramesForwarded = 0;
      this._yvuFramesForwarded = 0;
      this._underlayLoadFramesPresented = 0;
      this._zeroCopyFramesForwarded = 0;
      this._lastFrameFormat = null;
      this._lastChromaOrder = null;
      this._lastLoadOp = null;
      this._lastFrameWidth = 0;
      this._lastFrameHeight = 0;
      this._lastSource = null;
      this._nextFrameIndex = 0;

      if (this._presenter) {
        if (typeof this._presenter.setSkipped === 'function') {
          this._presenter.setSkipped(false);
        }
        // Pause→resume clears the presenter's shouldAdvance timing anchor.
        if (typeof this._presenter.setPaused === 'function') {
          this._presenter.setPaused(true);
          this._presenter.setPaused(false);
        }
      }
    }

    /**
     * Idempotently destroy the bridge, queue contents, retained ImageBitmaps,
     * scratch textures, and owned presenter.
     */
    destroy() {
      if (this._destroyed) {
        return;
      }
      this._destroyed = true;

      this._releaseSplashResourcesInternal();

      if (this._ownsPresenter && this._presenter && typeof this._presenter.destroy === 'function') {
        this._presenter.destroy();
      }
      this._presenter = null;
      this._pendingDevice = null;
      this._onFirstPresent = null;
      this._onFirstVisibleFrame = null;
      this._pixelProbe = null;
      this._stagingPool = null;
    }

    /**
     * Alias for `destroy()` for callers expecting a `dispose()` lifecycle hook.
     */
    dispose() {
      this.destroy();
    }
  }

  const exportedApi = Object.freeze({
    DEFAULT_MAX_QUEUE_DEPTH,
    DEFAULT_PRESENTATION_FORMAT,
    DEFAULT_TARGET_FPS,
    DEFAULT_MIN_VISIBLE_LUMA,
    DEFAULT_MIN_VISIBLE_PIXELS,
    DEFAULT_NONBLACK_GRID_MAX_DIM,
    SPLASH_FRAME_SOURCE,
    PIXEL_FORMAT,
    PIXEL_FORMAT_ID,
    ALLOWED_FRAME_FORMATS,
    COLOR_SPACE,
    COLOR_RANGE,
    CHROMA_ORDER,
    LOAD_OP,
    sanitizeFrameFormat,
    sanitizeChromaOrder,
    sanitizeLoadOp,
    hasVisibleNonBlackPixels,
    generateDiagnosticSplashFrame,
    SplashFrameBridge,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6SplashFrameBridge = exportedApi;
  } else if (root) {
    root.PlayGta6SplashFrameBridge = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
