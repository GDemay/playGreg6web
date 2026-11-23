'use strict';

// Splash / Bink2 frame presentation bridge.
// Zero external dependencies; UMD export for Web Workers (globalThis.PlayGta6SplashFrameBridge)
// and Node.js (module.exports). Queues decoded RGBA8 / YUV420P / YUVA420P / ImageBitmap frames
// into BinkVideoPresenter (W38) for letterboxed WebGPU presentation with FIRST_PRESENT telemetry.

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

  function validatePixelFormat(format) {
    if (
      format !== PIXEL_FORMAT.RGBA8 &&
      format !== PIXEL_FORMAT.YUV420P &&
      format !== PIXEL_FORMAT.YUVA420P
    ) {
      throw new RangeError(`Unsupported pixel format "${format}"`);
    }
  }

  function clonePlaneBytes(input, label) {
    const src = toUint8Array(input, label);
    return new Uint8Array(src);
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
      this._colorSpace = opts.colorSpace || 'bt709';
      this._targetFps = opts.targetFps || DEFAULT_TARGET_FPS;
      this._BinkVideoPresenterCtor = opts.BinkVideoPresenter || null;
      this._presenterOptions = opts.presenterOptions || null;

      this._queue = [];
      this._framesPresented = 0;
      this._droppedFrames = 0;
      this._firstPresentSent = false;
      this._lastSource = null;
      this._nextFrameIndex = 0;
      this._destroyed = false;
      this._externalScratchTextures = [];

      if (opts.presenter) {
        if (typeof opts.presenter !== 'object' || typeof opts.presenter.uploadRgbaFrame !== 'function') {
          throw new TypeError('presenter must provide uploadRgbaFrame / presentFrame APIs');
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
        targetFps: this._targetFps,
        ...(this._presenterOptions || {}),
      };
      this._presenter = new Ctor(ctorOptions);
      this._ownsPresenter = true;
      this._pendingDevice = resolvedDevice;
      return this._presenter;
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
     * Push a raw planar/RGBA frame into the bounded queue.
     *
     * @param {{
     *   width: number,
     *   height: number,
     *   format: string,
     *   planes: {
     *     rgba?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *     y?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *     u?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *     v?: Uint8Array|ArrayBufferView|ArrayBuffer,
     *     a?: Uint8Array|ArrayBufferView|ArrayBuffer|null,
     *   },
     *   timestampMs?: number,
     *   colorSpace?: string,
     *   source?: string,
     *   frameIndex?: number,
     * }} frame
     */
    pushRawFrame(frame) {
      this._assertNotDestroyed();
      if (!frame || typeof frame !== 'object') {
        throw new TypeError('pushRawFrame expects a frame descriptor object');
      }

      const { width, height, format, planes, timestampMs, colorSpace } = frame;
      assertPositiveInteger(width, 'width');
      assertPositiveInteger(height, 'height');
      validatePixelFormat(format);
      if (!planes || typeof planes !== 'object') {
        throw new TypeError('pushRawFrame requires a planes object');
      }

      const frameIndex =
        frame.frameIndex !== undefined ? frame.frameIndex : this._nextFrameIndex;
      assertNonNegativeInteger(frameIndex, 'frameIndex');
      this._nextFrameIndex = Math.max(this._nextFrameIndex, frameIndex + 1);

      const source =
        frame.source === SPLASH_FRAME_SOURCE.DIAGNOSTIC
          ? SPLASH_FRAME_SOURCE.DIAGNOSTIC
          : SPLASH_FRAME_SOURCE.RAW;

      let ownedPlanes;
      if (format === PIXEL_FORMAT.RGBA8) {
        if (planes.rgba === undefined || planes.rgba === null) {
          throw new TypeError('RGBA8 frames require planes.rgba');
        }
        ownedPlanes = { rgba: clonePlaneBytes(planes.rgba, 'planes.rgba') };
      } else {
        if (planes.y === undefined || planes.u === undefined || planes.v === undefined) {
          throw new TypeError('YUV frames require planes.y, planes.u, and planes.v');
        }
        ownedPlanes = {
          y: clonePlaneBytes(planes.y, 'planes.y'),
          u: clonePlaneBytes(planes.u, 'planes.u'),
          v: clonePlaneBytes(planes.v, 'planes.v'),
        };
        if (format === PIXEL_FORMAT.YUVA420P) {
          if (planes.a === undefined || planes.a === null) {
            throw new TypeError('YUVA420P frames require planes.a');
          }
          ownedPlanes.a = clonePlaneBytes(planes.a, 'planes.a');
        }
      }

      return this._enqueue({
        source,
        width,
        height,
        format,
        planes: ownedPlanes,
        timestampMs:
          typeof timestampMs === 'number' && Number.isFinite(timestampMs) ? timestampMs : null,
        colorSpace: colorSpace || null,
        frameIndex,
        image: null,
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

      const frameIndex =
        options && options.frameIndex !== undefined
          ? options.frameIndex
          : this._nextFrameIndex;
      assertNonNegativeInteger(frameIndex, 'frameIndex');
      this._nextFrameIndex = Math.max(this._nextFrameIndex, frameIndex + 1);

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
        planes: extracted ? { rgba: extracted } : { rgba: null },
        timestampMs,
        colorSpace: null,
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
     * @param {{ timestampMs?: number }} [options]
     */
    pushDiagnosticFrame(width, height, frameIndex, options = {}) {
      this._assertNotDestroyed();
      const index = frameIndex !== undefined ? frameIndex : this._nextFrameIndex;
      const generated = generateDiagnosticSplashFrame(width, height, index);
      return this.pushRawFrame({
        width: generated.width,
        height: generated.height,
        format: generated.format,
        planes: generated.planes,
        timestampMs: options && options.timestampMs,
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
        return presenter.uploadRgbaFrame(rgba, frame.width, frame.height);
      }

      const yuv = {
        y: frame.planes.y,
        u: frame.planes.u,
        v: frame.planes.v,
        width: frame.width,
        height: frame.height,
      };
      if (frame.format === PIXEL_FORMAT.YUVA420P) {
        yuv.a = frame.planes.a;
      }
      if (frame.colorSpace) {
        yuv.colorSpace = frame.colorSpace;
      }
      return presenter.uploadYuv420Frame(yuv);
    }

    /**
     * Dequeue the latest due frame (honoring presenter FPS pacing), upload via the presenter,
     * encode the letterboxed WebGPU render pass, and fire `onFirstPresent` exactly once.
     *
     * @param {GPUDevice} device
     * @param {GPUTextureView} targetView
     * @param {number} targetWidth
     * @param {number} targetHeight
     * @param {number} [nowMs]
     */
    renderFrame(device, targetView, targetWidth, targetHeight, nowMs) {
      this._assertNotDestroyed();
      if (!device || typeof device !== 'object') {
        throw new TypeError('renderFrame requires a WebGPU device');
      }
      if (!targetView || typeof targetView !== 'object') {
        throw new TypeError('renderFrame requires a target GPUTextureView');
      }
      assertPositiveInteger(Math.floor(targetWidth), 'targetWidth');
      assertPositiveInteger(Math.floor(targetHeight), 'targetHeight');

      const presenter = this._ensurePresenter(device);
      const timestamp =
        typeof nowMs === 'number' && Number.isFinite(nowMs)
          ? nowMs
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

      this._uploadFrame(presenter, device, frame);
      this._lastSource = frame.source;

      if (typeof device.createCommandEncoder !== 'function') {
        throw new TypeError('renderFrame requires device.createCommandEncoder()');
      }
      const encoder = device.createCommandEncoder({
        label: 'playgta6-splash-frame-bridge-encoder',
      });
      const presentResult = presenter.presentFrame(
        encoder,
        targetView,
        targetWidth,
        targetHeight
      );

      if (device.queue && typeof device.queue.submit === 'function') {
        const commandBuffer =
          encoder && typeof encoder.finish === 'function' ? encoder.finish() : encoder;
        device.queue.submit([commandBuffer]);
      }

      this._framesPresented += 1;
      this._releaseQueuedExternal(frame);

      const presentInfo = {
        frameIndex: frame.frameIndex,
        width: frame.width,
        height: frame.height,
        source: frame.source,
        timestampMs: frame.timestampMs !== null && frame.timestampMs !== undefined
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
        firstPresentSent: this._firstPresentSent,
      };
    }

    /**
     * Telemetry snapshot for HUD / BootTracer FIRST_PRESENT wiring.
     */
    getStats() {
      return {
        framesPresented: this._framesPresented,
        droppedFrames: this._droppedFrames,
        queueDepth: this._queue.length,
        firstPresentSent: this._firstPresentSent,
        lastSource: this._lastSource,
        maxQueueDepth: this._maxQueueDepth,
        destroyed: this._destroyed,
        presenterStats:
          this._presenter && typeof this._presenter.getStats === 'function'
            ? this._presenter.getStats()
            : null,
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
