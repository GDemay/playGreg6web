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

  const SPLASH_FRAME_SOURCE = Object.freeze({
    RAW: 'raw',
    EXTERNAL: 'external',
    DIAGNOSTIC: 'diagnostic',
  });

  const PIXEL_FORMAT = Object.freeze({
    RGBA8: 'rgba8',
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
      if (
        normalized === PIXEL_FORMAT.RGBA8 ||
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
    if (token === 'rgba8' || token === 'rgba' || token === 'rgba8unorm') {
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

      this._maxQueueDepth = maxQueueDepth;
      this._onFirstPresent = opts.onFirstPresent || null;
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
      this._droppedFrames = 0;
      this._firstPresentSent = false;
      this._guestFirstPresent = false;
      this._splashSuppressed = Boolean(opts.suppressSplash);
      this._framesBeforeFirstPresent = 0;
      this._lateSplashFramesIgnored = 0;
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
      this._externalScratchTextures = [];

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
        this._releaseQueuedExternal(dropped);
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

    /**
     * Signal that the live guest engine has reached `FIRST_PRESENT` handoff.
     * Freezes `framesBeforeFirstPresent` and activates splash suppression unless disabled.
     *
     * @param {{ suppress?: boolean }} [options]
     */
    notifyGuestFirstPresent(options = {}) {
      this._assertNotDestroyed();
      if (options !== undefined && options !== null && typeof options !== 'object') {
        throw new TypeError('notifyGuestFirstPresent options must be an object');
      }
      this._guestFirstPresent = true;
      const shouldSuppress =
        options && options.suppress !== undefined
          ? Boolean(options.suppress)
          : this._suppressSplashAfterFirstPresent;
      let drainedFrames = 0;
      if (shouldSuppress) {
        this._splashSuppressed = true;
        drainedFrames = this._drainQueueAsIgnored();
      }
      return {
        guestFirstPresent: this._guestFirstPresent,
        splashSuppressed: this._splashSuppressed,
        framesBeforeFirstPresent: this._framesBeforeFirstPresent,
        lateSplashFramesIgnored: this._lateSplashFramesIgnored,
        drainedFrames,
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
      return {
        splashSuppressed: this._splashSuppressed,
        lateSplashFramesIgnored: this._lateSplashFramesIgnored,
        drainedFrames,
      };
    }

    _enqueue(entry) {
      while (this._queue.length >= this._maxQueueDepth) {
        const dropped = this._queue.shift();
        this._droppedFrames += 1;
        this._releaseQueuedExternal(dropped);
      }
      this._queue.push(entry);
      return {
        queueDepth: this._queue.length,
        droppedFrames: this._droppedFrames,
        frameIndex: entry.frameIndex,
        chromaOrder: entry.chromaOrder || this._chromaOrder,
        loadOp: entry.loadOp || this._loadOp,
        zeroCopy: Boolean(entry.zeroCopy),
      };
    }

    _releaseQueuedExternal(entry) {
      if (!entry) {
        return;
      }
      if (entry.image && typeof entry.image.close === 'function') {
        try {
          entry.image.close();
        } catch (_) {
          // ImageBitmap may already be closed.
        }
      }
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

      if (format === PIXEL_FORMAT.RGBA8) {
        const rgbaInput =
          planes && planes.rgba !== undefined
            ? planes.rgba
            : frame.rgba !== undefined
              ? frame.rgba
              : frame.data;
        if (rgbaInput === undefined || rgbaInput === null) {
          throw new TypeError('RGBA8 frames require planes.rgba');
        }
        ownedPlanes = { rgba: acquirePlaneBytes(rgbaInput, 'planes.rgba', zeroCopy) };
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
        return this._ignoreLateSplashFrame(frameIndex);
      }

      const timestampMs =
        options && typeof options.timestampMs === 'number' && Number.isFinite(options.timestampMs)
          ? options.timestampMs
          : null;

      // Prefer immediate RGBA extraction so queue ownership does not depend on live bitmaps.
      const extracted = extractRgbaFromExternal(imageBitmap, width, height);
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
        image: extracted ? null : imageBitmap,
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
        this._releaseQueuedExternal(dropped);
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

      const texture = device.createTexture({
        label: `playgta6-splash-external-${frame.frameIndex}`,
        size: { width: frame.width, height: frame.height, depthOrArrayLayers: 1 },
        format: 'rgba8unorm',
        usage: TEXTURE_USAGE.COPY_DST | TEXTURE_USAGE.TEXTURE_BINDING,
      });
      device.queue.copyExternalImageToTexture(
        { source: frame.externalSource },
        { texture },
        { width: frame.width, height: frame.height, depthOrArrayLayers: 1 }
      );
      this._externalScratchTextures.push(texture);
      return true;
    }

    _uploadFrame(presenter, device, frame) {
      if (frame.source === SPLASH_FRAME_SOURCE.EXTERNAL) {
        this._uploadExternalViaCopy(device, frame);
      }

      if (frame.format === PIXEL_FORMAT.RGBA8) {
        let rgba = frame.planes && frame.planes.rgba;
        if (!rgba && frame.externalSource) {
          rgba = extractRgbaFromExternal(frame.externalSource, frame.width, frame.height);
        }
        if (!rgba) {
          // Last-resort solid frame so presentation never stalls when only GPU copy succeeded.
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
        };
        if (frame.yStride !== null && frame.yStride !== undefined) {
          nv12.yStride = frame.yStride;
        }
        if (frame.uvStride !== null && frame.uvStride !== undefined) {
          nv12.uvStride = frame.uvStride;
        }
        if (frame.colorSpace) {
          nv12.colorSpace = frame.colorSpace;
        }
        if (frame.colorRange) {
          nv12.colorRange = frame.colorRange;
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
      if (frame.colorSpace) {
        yuv.colorSpace = frame.colorSpace;
      }
      if (frame.colorRange) {
        yuv.colorRange = frame.colorRange;
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

      const uploadResult = this._uploadFrame(presenter, device, frame);
      this._lastSource = frame.source;
      const sanitizedFormat = sanitizeFrameFormat(frame.telemetryFormat || frame.format);
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
      const presentResult = presenter.presentFrame(
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
      if (effectiveLoadOp === LOAD_OP.LOAD) {
        this._underlayLoadFramesPresented += 1;
      }
      if (!this._guestFirstPresent) {
        this._framesBeforeFirstPresent += 1;
      }
      this._releaseQueuedExternal(frame);

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

      return {
        presented: Boolean(presentResult && presentResult.presented),
        reason: 'ok',
        ...presentInfo,
        viewport: presentResult ? presentResult.viewport : null,
        queueDepth: this._queue.length,
        framesPresented: this._framesPresented,
        framesBeforeFirstPresent: this._framesBeforeFirstPresent,
        nv12FramesForwarded: this._nv12FramesForwarded,
        yvuFramesForwarded: this._yvuFramesForwarded,
        underlayLoadFramesPresented: this._underlayLoadFramesPresented,
        zeroCopyFramesForwarded: this._zeroCopyFramesForwarded,
        firstPresentSent: this._firstPresentSent,
      };
    }

    /**
     * Telemetry snapshot for HUD / BootTracer FIRST_PRESENT wiring.
     */
    getStats() {
      const chromaOrder = sanitizeChromaOrder(this._chromaOrder) || CHROMA_ORDER.YUV;
      const loadOp = sanitizeLoadOp(this._loadOp) || LOAD_OP.CLEAR;
      return {
        framesPresented: this._framesPresented,
        droppedFrames: this._droppedFrames,
        queueDepth: this._queue.length,
        firstPresentSent: this._firstPresentSent,
        guestFirstPresent: this._guestFirstPresent,
        splashSuppressed: this._splashSuppressed,
        framesBeforeFirstPresent: this._framesBeforeFirstPresent,
        lateSplashFramesIgnored: this._lateSplashFramesIgnored,
        nv12FramesForwarded: this._nv12FramesForwarded,
        yvuFramesForwarded: this._yvuFramesForwarded,
        underlayLoadFramesPresented: this._underlayLoadFramesPresented,
        zeroCopyFramesForwarded: this._zeroCopyFramesForwarded,
        chromaOrder,
        loadOp,
        lastChromaOrder: sanitizeChromaOrder(this._lastChromaOrder),
        lastLoadOp: sanitizeLoadOp(this._lastLoadOp),
        lastFrameWidth: this._lastFrameWidth,
        lastFrameHeight: this._lastFrameHeight,
        lastFrameFormat: sanitizeFrameFormat(this._lastFrameFormat),
        lastSource: this._lastSource,
        maxQueueDepth: this._maxQueueDepth,
        zeroCopy: this._zeroCopy,
        destroyed: this._destroyed,
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
        zeroCopy: this._zeroCopy,
        maxQueueDepth: this._maxQueueDepth,
        queueDepth: this._queue.length,
        framesPresented: this._framesPresented,
        droppedFrames: this._droppedFrames,
        firstPresentSent: this._firstPresentSent,
        guestFirstPresent: this._guestFirstPresent,
        splashSuppressed: this._splashSuppressed,
        framesBeforeFirstPresent: this._framesBeforeFirstPresent,
        lateSplashFramesIgnored: this._lateSplashFramesIgnored,
        lastFrameFormat: sanitizeFrameFormat(this._lastFrameFormat),
        lastFrameWidth: this._lastFrameWidth,
        lastFrameHeight: this._lastFrameHeight,
        lastSource: this._lastSource,
        destroyed: this._destroyed,
      };
    }

    /**
     * Clear the queue and presentation telemetry so the next presented frame can fire
     * `onFirstPresent` again. Does not destroy the underlying presenter.
     */
    reset() {
      this._assertNotDestroyed();
      while (this._queue.length > 0) {
        this._releaseQueuedExternal(this._queue.shift());
      }
      this._framesPresented = 0;
      this._droppedFrames = 0;
      this._firstPresentSent = false;
      this._guestFirstPresent = false;
      this._splashSuppressed = false;
      this._framesBeforeFirstPresent = 0;
      this._lateSplashFramesIgnored = 0;
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

      for (let i = 0; i < this._externalScratchTextures.length; i += 1) {
        const tex = this._externalScratchTextures[i];
        if (tex && typeof tex.destroy === 'function') {
          tex.destroy();
        }
      }
      this._externalScratchTextures = [];
    }

    /**
     * Idempotently destroy the bridge, queue contents, scratch textures, and owned presenter.
     */
    destroy() {
      if (this._destroyed) {
        return;
      }
      this._destroyed = true;

      while (this._queue.length > 0) {
        this._releaseQueuedExternal(this._queue.shift());
      }

      for (let i = 0; i < this._externalScratchTextures.length; i += 1) {
        const tex = this._externalScratchTextures[i];
        if (tex && typeof tex.destroy === 'function') {
          tex.destroy();
        }
      }
      this._externalScratchTextures = [];

      if (this._ownsPresenter && this._presenter && typeof this._presenter.destroy === 'function') {
        this._presenter.destroy();
      }
      this._presenter = null;
      this._pendingDevice = null;
      this._onFirstPresent = null;
    }
  }

  const exportedApi = Object.freeze({
    DEFAULT_MAX_QUEUE_DEPTH,
    DEFAULT_PRESENTATION_FORMAT,
    DEFAULT_TARGET_FPS,
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
