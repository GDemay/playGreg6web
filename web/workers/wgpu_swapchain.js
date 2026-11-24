'use strict';

(function (root) {
  const DEFAULT_DEPTH_FORMAT = 'depth24plus';

  const SUPPORTED_DEPTH_FORMATS = Object.freeze([
    'depth24plus',
    'depth24plus-stencil8',
    'depth32float',
    'depth32float-stencil8',
    'depth16unorm',
  ]);

  const STENCIL_DEPTH_FORMATS = new Set([
    'depth24plus-stencil8',
    'depth32float-stencil8',
  ]);

  const SUPPORTED_ALPHA_MODES = Object.freeze(['opaque', 'premultiplied']);

  const DEFAULT_CLEAR_COLOR = Object.freeze({ r: 0.05, g: 0.07, b: 0.11, a: 1.0 });
  const DEFAULT_FORMAT = 'bgra8unorm';
  const DEFAULT_ALPHA_MODE = 'opaque';
  const DEFAULT_MAX_DIMENSION = 8192;
  const DEFAULT_FRAME_WINDOW = 60;
  const DEFAULT_MAX_ERROR_HISTORY = 32;
  const MAX_VALIDATION_ERROR_MESSAGE_LENGTH = 512;

  // WebGPU GPUTextureUsage.RENDER_ATTACHMENT === 0x10 (fallback for Node/headless tests).
  const TEXTURE_USAGE_RENDER_ATTACHMENT =
    typeof GPUTextureUsage !== 'undefined' &&
    typeof GPUTextureUsage.RENDER_ATTACHMENT === 'number'
      ? GPUTextureUsage.RENDER_ATTACHMENT
      : 0x10;

  function defaultNow() {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
      return performance.now();
    }
    return Date.now();
  }

  function isStencilFormat(format) {
    return STENCIL_DEPTH_FORMATS.has(format);
  }

  function validateDepthFormat(depthFormat) {
    if (typeof depthFormat !== 'string' || !SUPPORTED_DEPTH_FORMATS.includes(depthFormat)) {
      throw new RangeError(
        `Unsupported depthFormat "${depthFormat}"; expected one of: ${SUPPORTED_DEPTH_FORMATS.join(', ')}`
      );
    }
  }

  function validateAlphaMode(alphaMode) {
    if (typeof alphaMode !== 'string' || !SUPPORTED_ALPHA_MODES.includes(alphaMode)) {
      throw new RangeError(
        `Unsupported alphaMode "${alphaMode}"; expected one of: ${SUPPORTED_ALPHA_MODES.join(', ')}`
      );
    }
  }

  function clampDimension(cssSize, devicePixelRatio, maxDimension) {
    if (cssSize === Infinity) {
      return maxDimension;
    }
    if (cssSize === -Infinity || typeof cssSize !== 'number' || Number.isNaN(cssSize)) {
      return 1;
    }

    let dpr = 1;
    if (devicePixelRatio !== undefined && devicePixelRatio !== null) {
      if (devicePixelRatio === Infinity) {
        return cssSize > 0 ? maxDimension : 1;
      }
      if (typeof devicePixelRatio === 'number' && !Number.isNaN(devicePixelRatio)) {
        dpr = devicePixelRatio;
      }
    }

    const scaled = Math.round(cssSize * dpr);
    if (!Number.isFinite(scaled)) {
      return scaled > 0 ? maxDimension : 1;
    }
    if (scaled < 1) {
      return 1;
    }
    if (scaled > maxDimension) {
      return maxDimension;
    }
    return scaled;
  }

  function normalizeClearColor(clearColor) {
    if (!clearColor) {
      return { ...DEFAULT_CLEAR_COLOR };
    }
    if (Array.isArray(clearColor) && clearColor.length >= 4) {
      return {
        r: clearColor[0],
        g: clearColor[1],
        b: clearColor[2],
        a: clearColor[3],
      };
    }
    return {
      r: typeof clearColor.r === 'number' ? clearColor.r : DEFAULT_CLEAR_COLOR.r,
      g: typeof clearColor.g === 'number' ? clearColor.g : DEFAULT_CLEAR_COLOR.g,
      b: typeof clearColor.b === 'number' ? clearColor.b : DEFAULT_CLEAR_COLOR.b,
      a: typeof clearColor.a === 'number' ? clearColor.a : DEFAULT_CLEAR_COLOR.a,
    };
  }

  function classifyGpuErrorType(raw) {
    if (typeof raw !== 'string') {
      return null;
    }
    const normalized = raw.trim().toLowerCase();
    if (!normalized) {
      return null;
    }
    if (
      normalized === 'out-of-memory' ||
      normalized === 'out_of_memory' ||
      normalized === 'outofmemory' ||
      normalized === 'oom' ||
      normalized.includes('outofmemory') ||
      normalized.includes('out-of-memory') ||
      normalized.includes('out_of_memory')
    ) {
      return 'out-of-memory';
    }
    if (normalized === 'validation' || normalized.includes('validation')) {
      return 'validation';
    }
    if (normalized === 'internal' || normalized.includes('internal')) {
      return 'internal';
    }
    return null;
  }

  function normalizeValidationErrorType(error, event) {
    if (error && typeof error === 'object') {
      if (typeof GPUOutOfMemoryError === 'function' && error instanceof GPUOutOfMemoryError) {
        return 'out-of-memory';
      }
      if (typeof GPUValidationError === 'function' && error instanceof GPUValidationError) {
        return 'validation';
      }
      if (typeof GPUInternalError === 'function' && error instanceof GPUInternalError) {
        return 'internal';
      }
      const fromType = classifyGpuErrorType(error.type);
      if (fromType) {
        return fromType;
      }
      const fromName = classifyGpuErrorType(error.name);
      if (fromName) {
        return fromName;
      }
      if (error.constructor && typeof error.constructor.name === 'string') {
        const fromCtor = classifyGpuErrorType(error.constructor.name);
        if (fromCtor) {
          return fromCtor;
        }
      }
    }
    if (event && typeof event === 'object') {
      const fromEventType = classifyGpuErrorType(event.errorType);
      if (fromEventType) {
        return fromEventType;
      }
    }
    return 'unknown';
  }

  function sanitizeValidationErrorMessage(rawMessage) {
    let msg;
    if (typeof rawMessage === 'string') {
      msg = rawMessage;
    } else if (rawMessage === null || rawMessage === undefined) {
      return '';
    } else {
      msg = String(rawMessage);
    }

    // Normalize line breaks/tabs to spaces and strip control characters.
    msg = msg.replace(/[\r\n\t]+/g, ' ').replace(/[\x00-\x1F\x7F]+/g, '');

    // Strip file:// URLs and host filesystem paths (Windows drive, UNC, and Unix paths).
    msg = msg.replace(/\bfile:\/\/[^\s"'`<>)\]]+/gi, '<path>');
    msg = msg.replace(/\\\\[^\s\\"'`<>)\]]+\\[^\s"'`<>)\]]+/g, '<path>');
    msg = msg.replace(/(["'`])[A-Za-z]:[\\\/][^"'`\r\n]+\1/g, '$1<path>$1');
    msg = msg.replace(/\b[A-Za-z]:[\\\/][^\s"'`<>)\],;]+/g, '<path>');
    msg = msg.replace(
      /(["'`])\/(?:Users|home|var|tmp|private|etc|opt|usr|Volumes|mnt|root|Library|Applications|System)\/[^"'`\r\n]+\1/gi,
      '$1<path>$1'
    );
    msg = msg.replace(
      /(^|[\s"'`(\[=,:])\/(?:[A-Za-z0-9._-]+\/)+[A-Za-z0-9._~+-]+/g,
      '$1<path>'
    );

    // Strip long embedded buffers (data URIs, byte arrays, hex dumps, base64 blobs).
    msg = msg.replace(/\bdata:[^\s;,'"`<>]+;base64,[A-Za-z0-9+/=]{16,}/gi, '<buffer>');
    msg = msg.replace(
      /(?:\b(?:0x)?[0-9a-fA-F]{1,4}\s*,\s*){16,}(?:0x)?[0-9a-fA-F]{1,4}\b/g,
      '<buffer>'
    );
    msg = msg.replace(/(?:\b[0-9a-fA-F]{2}\s+){16,}[0-9a-fA-F]{2}\b/g, '<buffer>');
    msg = msg.replace(/\b(?:0x)?[0-9a-fA-F]{48,}\b/g, '<buffer>');
    msg = msg.replace(
      /(^|[\s"'`(\[=,:])[A-Za-z0-9+/]{64,}={0,2}(?=$|[\s"'`)\]=,.;:])/g,
      '$1<buffer>'
    );

    msg = msg.replace(/\s{2,}/g, ' ').trim();

    if (msg.length > MAX_VALIDATION_ERROR_MESSAGE_LENGTH) {
      msg = `${msg.slice(0, MAX_VALIDATION_ERROR_MESSAGE_LENGTH - 3)}...`;
    }
    return msg;
  }

  /**
   * Manages an OffscreenCanvas WebGPU swapchain:
   * - DPR-aware configuration and dynamic resize with safe clamping
   * - Lazy depth-stencil GPUTexture allocation and leak-free recreation
   * - GPURenderPassDescriptor acquisition compatible with WgpuCommandProcessor
   * - device.lost handling and live GPUDevice recovery
   * - Rolling frame-pacing telemetry (FPS, avg/max frame time, resize count)
   */
  class WgpuSwapchain {
    constructor(options = {}) {
      if (!options || typeof options !== 'object') {
        throw new TypeError('WgpuSwapchain requires an options object');
      }

      const {
        canvas,
        device,
        context,
        format,
        depthFormat = DEFAULT_DEPTH_FORMAT,
        alphaMode = DEFAULT_ALPHA_MODE,
        maxDimension = DEFAULT_MAX_DIMENSION,
        enableDepth = true,
        now = defaultNow,
        frameWindowSize = DEFAULT_FRAME_WINDOW,
        onValidationError = null,
        maxErrorHistory = DEFAULT_MAX_ERROR_HISTORY,
      } = options;

      if (!canvas || typeof canvas !== 'object') {
        throw new TypeError('WgpuSwapchain requires a canvas (OffscreenCanvas or canvas-like object)');
      }
      if (!device || typeof device !== 'object') {
        throw new TypeError('WgpuSwapchain requires a GPUDevice');
      }

      const resolvedContext =
        context || (typeof canvas.getContext === 'function' ? canvas.getContext('webgpu') : null);
      if (!resolvedContext || typeof resolvedContext.configure !== 'function') {
        throw new TypeError('WgpuSwapchain requires a WebGPU GPUCanvasContext');
      }

      validateDepthFormat(depthFormat);
      validateAlphaMode(alphaMode);

      if (!Number.isInteger(maxDimension) || maxDimension < 1) {
        throw new RangeError(`maxDimension must be a positive integer, got ${maxDimension}`);
      }
      if (typeof now !== 'function') {
        throw new TypeError('now must be a function returning a timestamp in milliseconds');
      }
      if (!Number.isInteger(frameWindowSize) || frameWindowSize < 1) {
        throw new RangeError(`frameWindowSize must be a positive integer, got ${frameWindowSize}`);
      }
      if (
        onValidationError !== null &&
        onValidationError !== undefined &&
        typeof onValidationError !== 'function'
      ) {
        throw new TypeError('onValidationError must be a function when provided');
      }
      if (!Number.isInteger(maxErrorHistory) || maxErrorHistory < 1) {
        throw new RangeError(`maxErrorHistory must be a positive integer, got ${maxErrorHistory}`);
      }

      const resolvedFormat =
        typeof format === 'string' && format.length > 0
          ? format
          : typeof navigator !== 'undefined' &&
              navigator.gpu &&
              typeof navigator.gpu.getPreferredCanvasFormat === 'function'
            ? navigator.gpu.getPreferredCanvasFormat()
            : DEFAULT_FORMAT;

      this._canvas = canvas;
      this._device = device;
      this._context = resolvedContext;
      this._format = resolvedFormat;
      this._depthFormat = depthFormat;
      this._alphaMode = alphaMode;
      this._maxDimension = maxDimension;
      this._enableDepth = Boolean(enableDepth);
      this._now = now;
      this._frameWindowSize = frameWindowSize;
      this._onValidationError = typeof onValidationError === 'function' ? onValidationError : null;
      this._maxErrorHistory = maxErrorHistory;

      this._width = 0;
      this._height = 0;
      this._devicePixelRatio = 1;
      this._configured = false;
      this._destroyed = false;

      this._depthTexture = null;
      this._depthTextureView = null;
      this._depthTextureMeta = null;

      this._isDeviceLost = false;
      this._lastDeviceLostInfo = null;
      this._deviceLostCallbacks = new Set();
      this._deviceLostGeneration = 0;
      this._hookedDevice = null;

      this._errorHookedDevice = null;
      this._boundUncapturedErrorHandler = null;
      this._validationErrors = [];
      this._validationErrorCount = 0;
      this._validationErrorSeq = 0;

      this._totalFrames = 0;
      this._resizeCount = 0;
      this._lastFrameTimestamp = null;
      this._frameDeltas = [];

      this._hookDeviceLost(this._device);
      this._hookUncapturedError(this._device);
    }

    get width() {
      return this._width;
    }

    get height() {
      return this._height;
    }

    get format() {
      return this._format;
    }

    get depthFormat() {
      return this._enableDepth ? this._depthFormat : null;
    }

    get alphaMode() {
      return this._alphaMode;
    }

    get enableDepth() {
      return this._enableDepth;
    }

    get device() {
      return this._device;
    }

    get context() {
      return this._context;
    }

    get canvas() {
      return this._canvas;
    }

    get depthTexture() {
      return this._depthTexture;
    }

    get depthTextureView() {
      return this._depthTextureView;
    }

    get isConfigured() {
      return this._configured;
    }

    get isDeviceLost() {
      return this._isDeviceLost;
    }

    get isDestroyed() {
      return this._destroyed;
    }

    _effectiveMaxDimension() {
      const deviceLimit =
        this._device &&
        this._device.limits &&
        Number.isInteger(this._device.limits.maxTextureDimension2D) &&
        this._device.limits.maxTextureDimension2D > 0
          ? this._device.limits.maxTextureDimension2D
          : this._maxDimension;
      return Math.min(this._maxDimension, deviceLimit);
    }

    _destroyDepthTexture() {
      if (this._depthTexture) {
        if (typeof this._depthTexture.destroy === 'function') {
          this._depthTexture.destroy();
        }
        this._depthTexture = null;
      }
      this._depthTextureView = null;
      this._depthTextureMeta = null;
    }

    _ensureDepthTexture() {
      if (!this._enableDepth) {
        this._destroyDepthTexture();
        return null;
      }
      if (this._width < 1 || this._height < 1 || !this._device) {
        return null;
      }

      const meta = this._depthTextureMeta;
      if (
        this._depthTexture &&
        this._depthTextureView &&
        meta &&
        meta.width === this._width &&
        meta.height === this._height &&
        meta.format === this._depthFormat &&
        meta.device === this._device
      ) {
        return this._depthTexture;
      }

      // Destroy previous depth texture BEFORE allocating the new one to avoid VRAM leaks.
      this._destroyDepthTexture();

      const texture = this._device.createTexture({
        label: 'playgta6-swapchain-depth',
        size: {
          width: this._width,
          height: this._height,
          depthOrArrayLayers: 1,
        },
        format: this._depthFormat,
        usage: TEXTURE_USAGE_RENDER_ATTACHMENT,
      });

      this._depthTexture = texture;
      this._depthTextureView =
        texture && typeof texture.createView === 'function' ? texture.createView() : null;
      this._depthTextureMeta = {
        width: this._width,
        height: this._height,
        format: this._depthFormat,
        device: this._device,
      };
      return texture;
    }

    _hookDeviceLost(device) {
      if (!device || device === this._hookedDevice) {
        return;
      }
      if (!device.lost || typeof device.lost.then !== 'function') {
        this._hookedDevice = device;
        return;
      }

      this._hookedDevice = device;
      const generation = this._deviceLostGeneration;

      device.lost.then((info) => {
        if (
          this._destroyed ||
          generation !== this._deviceLostGeneration ||
          this._device !== device
        ) {
          return;
        }
        this._isDeviceLost = true;
        this._lastDeviceLostInfo = info || { reason: 'unknown', message: '' };
        this._destroyDepthTexture();

        for (const callback of Array.from(this._deviceLostCallbacks)) {
          callback(this._lastDeviceLostInfo, this);
        }
      });
    }

    _unhookUncapturedError() {
      const prevDevice = this._errorHookedDevice;
      const prevHandler = this._boundUncapturedErrorHandler;
      this._errorHookedDevice = null;
      this._boundUncapturedErrorHandler = null;

      if (
        prevDevice &&
        prevHandler &&
        typeof prevDevice.removeEventListener === 'function'
      ) {
        try {
          prevDevice.removeEventListener('uncapturederror', prevHandler);
        } catch {
          // Ignore removal failures on torn-down or mock devices.
        }
      }
    }

    _hookUncapturedError(device) {
      if (!device) {
        this._unhookUncapturedError();
        return;
      }
      if (device === this._errorHookedDevice && this._boundUncapturedErrorHandler) {
        return;
      }

      this._unhookUncapturedError();

      if (typeof device.addEventListener !== 'function') {
        return;
      }

      const handler = (event) => {
        if (this._destroyed || this._device !== device) {
          return;
        }
        this._handleUncapturedError(event);
      };

      this._errorHookedDevice = device;
      this._boundUncapturedErrorHandler = handler;
      device.addEventListener('uncapturederror', handler);
    }

    _handleUncapturedError(event) {
      const rawError =
        event && typeof event === 'object' && 'error' in event ? event.error : event;
      const type = normalizeValidationErrorType(rawError, event);

      let rawMessage = '';
      if (rawError && typeof rawError === 'object' && typeof rawError.message === 'string') {
        rawMessage = rawError.message;
      } else if (event && typeof event === 'object' && typeof event.message === 'string') {
        rawMessage = event.message;
      } else if (typeof rawError === 'string') {
        rawMessage = rawError;
      } else if (rawError !== null && rawError !== undefined) {
        rawMessage = String(rawError);
      }

      const message = sanitizeValidationErrorMessage(rawMessage);
      const nowVal = this._now();
      const timestampMs =
        typeof nowVal === 'number' && Number.isFinite(nowVal) ? nowVal : Date.now();

      this._validationErrorSeq += 1;
      this._validationErrorCount += 1;

      const record = {
        seq: this._validationErrorSeq,
        type,
        message,
        timestampMs,
      };

      this._validationErrors.push(record);
      if (this._validationErrors.length > this._maxErrorHistory) {
        this._validationErrors.shift();
      }

      if (typeof this._onValidationError === 'function') {
        try {
          this._onValidationError({ ...record });
        } catch {
          // Isolate callback exceptions so validation error capture never crashes the worker.
        }
      }

      return record;
    }

    _applyContextConfiguration() {
      this._canvas.width = this._width;
      this._canvas.height = this._height;
      this._context.configure({
        device: this._device,
        format: this._format,
        alphaMode: this._alphaMode,
      });
      this._configured = true;
      this._ensureDepthTexture();
    }

    /**
     * Configure canvas pixel dimensions, WebGPU canvas context, and depth-stencil attachment.
     *
     * @param {{
     *   width: number,
     *   height: number,
     *   devicePixelRatio?: number,
     *   format?: string,
     *   depthFormat?: string,
     *   alphaMode?: string,
     *   enableDepth?: boolean,
     * }} config
     */
    configure(config = {}) {
      if (this._destroyed) {
        throw new Error('Cannot configure a destroyed WgpuSwapchain');
      }
      if (this._isDeviceLost) {
        throw new Error('Cannot configure WgpuSwapchain while GPUDevice is lost; call recover() first');
      }
      if (!config || typeof config !== 'object') {
        throw new TypeError('configure requires a configuration object');
      }

      if (config.format !== undefined) {
        if (typeof config.format !== 'string' || config.format.length === 0) {
          throw new TypeError('format must be a non-empty string');
        }
        this._format = config.format;
      }
      if (config.depthFormat !== undefined) {
        validateDepthFormat(config.depthFormat);
        this._depthFormat = config.depthFormat;
      }
      if (config.alphaMode !== undefined) {
        validateAlphaMode(config.alphaMode);
        this._alphaMode = config.alphaMode;
      }
      if (config.enableDepth !== undefined) {
        this._enableDepth = Boolean(config.enableDepth);
      }

      const dpr = config.devicePixelRatio !== undefined ? config.devicePixelRatio : 1;
      const maxDim = this._effectiveMaxDimension();
      const nextWidth = clampDimension(config.width, dpr, maxDim);
      const nextHeight = clampDimension(config.height, dpr, maxDim);

      const wasConfigured = this._configured;
      const dimensionsChanged =
        wasConfigured && (nextWidth !== this._width || nextHeight !== this._height);

      this._width = nextWidth;
      this._height = nextHeight;
      this._devicePixelRatio =
        typeof dpr === 'number' && Number.isFinite(dpr) && dpr > 0 ? dpr : 1;

      this._applyContextConfiguration();

      if (dimensionsChanged) {
        this._resizeCount += 1;
      }

      return {
        width: this._width,
        height: this._height,
        format: this._format,
        depthFormat: this._enableDepth ? this._depthFormat : null,
        devicePixelRatio: this._devicePixelRatio,
      };
    }

    /**
     * Dynamically resize the swapchain. No-ops (returning false) when clamped pixel dimensions
     * match the currently configured dimensions.
     *
     * @param {number} cssWidth
     * @param {number} cssHeight
     * @param {number} [devicePixelRatio=1]
     * @returns {boolean} True if swapchain was reconfigured, false on no-op.
     */
    resize(cssWidth, cssHeight, devicePixelRatio = 1) {
      if (this._destroyed) {
        throw new Error('Cannot resize a destroyed WgpuSwapchain');
      }
      if (this._isDeviceLost) {
        throw new Error('Cannot resize WgpuSwapchain while GPUDevice is lost; call recover() first');
      }

      const maxDim = this._effectiveMaxDimension();
      const nextWidth = clampDimension(cssWidth, devicePixelRatio, maxDim);
      const nextHeight = clampDimension(cssHeight, devicePixelRatio, maxDim);

      if (this._configured && nextWidth === this._width && nextHeight === this._height) {
        return false;
      }

      this.configure({
        width: cssWidth,
        height: cssHeight,
        devicePixelRatio,
      });
      return true;
    }

    /**
     * Acquire the current swapchain color texture view (and matching depth-stencil view when
     * enabled) packaged as a GPURenderPassDescriptor ready for encoder.beginRenderPass().
     *
     * @param {{
     *   clearColor?: { r: number, g: number, b: number, a: number } | number[],
     *   loadOp?: 'clear' | 'load',
     *   storeOp?: 'store' | 'discard',
     *   depthClearValue?: number,
     *   depthLoadOp?: 'clear' | 'load',
     *   depthStoreOp?: 'store' | 'discard',
     *   stencilClearValue?: number,
     *   stencilLoadOp?: 'clear' | 'load',
     *   stencilStoreOp?: 'store' | 'discard',
     *   label?: string,
     * }} [options]
     * @returns {GPURenderPassDescriptor}
     */
    acquireRenderPassDescriptor(options = {}) {
      if (this._destroyed) {
        throw new Error('Cannot acquire render pass descriptor from a destroyed WgpuSwapchain');
      }
      if (this._isDeviceLost) {
        throw new Error('Cannot acquire render pass descriptor while GPUDevice is lost');
      }
      if (!this._configured) {
        throw new Error('WgpuSwapchain must be configured before calling acquireRenderPassDescriptor()');
      }

      const {
        clearColor,
        loadOp = 'clear',
        storeOp = 'store',
        depthClearValue = 1.0,
        depthLoadOp = 'clear',
        depthStoreOp = 'store',
        stencilClearValue = 0,
        stencilLoadOp = 'clear',
        stencilStoreOp = 'store',
        label,
      } = options || {};

      const currentTexture = this._context.getCurrentTexture();
      const colorView = currentTexture.createView();

      const colorAttachment = {
        view: colorView,
        loadOp,
        storeOp,
      };
      if (loadOp === 'clear') {
        colorAttachment.clearValue = normalizeClearColor(clearColor);
      }

      const descriptor = {
        colorAttachments: [colorAttachment],
      };

      if (this._enableDepth) {
        this._ensureDepthTexture();
        const depthStencilAttachment = {
          view: this._depthTextureView,
          depthLoadOp,
          depthStoreOp,
        };
        if (depthLoadOp === 'clear') {
          depthStencilAttachment.depthClearValue = depthClearValue;
        }
        if (isStencilFormat(this._depthFormat)) {
          depthStencilAttachment.stencilLoadOp = stencilLoadOp;
          depthStencilAttachment.stencilStoreOp = stencilStoreOp;
          if (stencilLoadOp === 'clear') {
            depthStencilAttachment.stencilClearValue = stencilClearValue;
          }
        }
        descriptor.depthStencilAttachment = depthStencilAttachment;
      }

      if (label !== undefined) {
        descriptor.label = label;
      }

      return descriptor;
    }

    /**
     * Record a presented frame timestamp into the rolling frame-pacing window.
     *
     * @param {number} [timestampMs] Optional explicit timestamp override; defaults to `now()`.
     */
    recordFramePresented(timestampMs) {
      const nowMs =
        typeof timestampMs === 'number' && Number.isFinite(timestampMs)
          ? timestampMs
          : this._now();

      this._totalFrames += 1;

      if (this._lastFrameTimestamp !== null) {
        const deltaMs = Math.max(0, nowMs - this._lastFrameTimestamp);
        this._frameDeltas.push(deltaMs);
        if (this._frameDeltas.length > this._frameWindowSize) {
          this._frameDeltas.shift();
        }
      }

      this._lastFrameTimestamp = nowMs;
      return this.getTelemetry();
    }

    /**
     * Return rolling frame-pacing and swapchain state telemetry.
     */
    getTelemetry() {
      const count = this._frameDeltas.length;
      let avgFrameTimeMs = 0;
      let maxFrameTimeMs = 0;

      if (count > 0) {
        let sum = 0;
        for (let i = 0; i < count; i += 1) {
          const dt = this._frameDeltas[i];
          sum += dt;
          if (dt > maxFrameTimeMs) {
            maxFrameTimeMs = dt;
          }
        }
        avgFrameTimeMs = sum / count;
      }

      const fps = avgFrameTimeMs > 0 ? 1000 / avgFrameTimeMs : 0;

      return {
        fps,
        avgFrameTimeMs,
        maxFrameTimeMs,
        totalFrames: this._totalFrames,
        resizeCount: this._resizeCount,
        width: this._width,
        height: this._height,
        format: this._format,
        depthFormat: this._enableDepth ? this._depthFormat : null,
        isDeviceLost: this._isDeviceLost,
      };
    }

    /**
     * Return swapchain telemetry plus WebGPU validation-error counters.
     */
    getStats() {
      return {
        ...this.getTelemetry(),
        validationErrorCount: this._validationErrorCount,
        validationErrorHistorySize: this._validationErrors.length,
        maxErrorHistory: this._maxErrorHistory,
        isDestroyed: this._destroyed,
      };
    }

    /**
     * Return shallow copies of captured WebGPU uncaptured validation/OOM error records
     * retained in the bounded history ring.
     *
     * @returns {Array<{ seq: number, type: string, message: string, timestampMs: number }>}
     */
    getValidationErrors() {
      return this._validationErrors.map((entry) => ({ ...entry }));
    }

    /**
     * Clear the bounded WebGPU validation error history ring.
     */
    clearValidationErrors() {
      this._validationErrors.length = 0;
    }

    /**
     * Register a callback invoked when the active GPUDevice's `device.lost` promise resolves.
     * Returns an unsubscribe function.
     *
     * @param {(info: { reason?: string, message?: string }, swapchain: WgpuSwapchain) => void} onLost
     * @returns {() => void}
     */
    attachDeviceLostHandler(onLost) {
      if (this._destroyed) {
        throw new Error('Cannot attach device lost handler to a destroyed WgpuSwapchain');
      }
      if (typeof onLost !== 'function') {
        throw new TypeError('attachDeviceLostHandler requires a callback function');
      }

      this._deviceLostCallbacks.add(onLost);
      if (this._device && this._device !== this._hookedDevice) {
        this._hookDeviceLost(this._device);
      }

      return () => {
        this._deviceLostCallbacks.delete(onLost);
      };
    }

    /**
     * Recover from a lost GPUDevice by binding a new GPUDevice (and optional new canvas format),
     * destroying any stale depth texture, hooking the new device's `lost` promise, and
     * re-configuring the existing canvas/context bindings.
     *
     * @param {GPUDevice} newDevice
     * @param {string} [newFormat]
     */
    recover(newDevice, newFormat) {
      if (this._destroyed) {
        throw new Error('Cannot recover a destroyed WgpuSwapchain');
      }
      if (!newDevice || typeof newDevice !== 'object') {
        throw new TypeError('recover requires a new GPUDevice');
      }
      if (newFormat !== undefined && newFormat !== null) {
        if (typeof newFormat !== 'string' || newFormat.length === 0) {
          throw new TypeError('newFormat must be a non-empty string when provided');
        }
        this._format = newFormat;
      }

      this._deviceLostGeneration += 1;
      this._unhookUncapturedError();
      this._destroyDepthTexture();

      this._device = newDevice;
      this._isDeviceLost = false;
      this._lastDeviceLostInfo = null;
      this._hookedDevice = null;
      this._lastFrameTimestamp = null;

      this._hookDeviceLost(newDevice);
      this._hookUncapturedError(newDevice);

      if (this._configured && this._width > 0 && this._height > 0) {
        const maxDim = this._effectiveMaxDimension();
        this._width = Math.min(this._width, maxDim);
        this._height = Math.min(this._height, maxDim);
        this._applyContextConfiguration();
      }

      return {
        width: this._width,
        height: this._height,
        format: this._format,
        depthFormat: this._enableDepth ? this._depthFormat : null,
        isDeviceLost: this._isDeviceLost,
      };
    }

    /**
     * Destroy owned GPU resources (depth texture), unconfigure the canvas context if supported,
     * and release references. Safe to call multiple times (idempotent).
     */
    destroy() {
      if (this._destroyed) {
        return;
      }

      this._destroyed = true;
      this._deviceLostGeneration += 1;
      this._unhookUncapturedError();
      this._destroyDepthTexture();

      if (this._context && typeof this._context.unconfigure === 'function') {
        this._context.unconfigure();
      }

      this._deviceLostCallbacks.clear();
      this._onValidationError = null;
      this._device = null;
      this._context = null;
      this._canvas = null;
      this._hookedDevice = null;
      this._configured = false;
    }
  }

  const exportedApi = {
    DEFAULT_DEPTH_FORMAT,
    SUPPORTED_DEPTH_FORMATS,
    DEFAULT_CLEAR_COLOR,
    DEFAULT_ALPHA_MODE,
    DEFAULT_MAX_DIMENSION,
    DEFAULT_FRAME_WINDOW,
    DEFAULT_MAX_ERROR_HISTORY,
    MAX_VALIDATION_ERROR_MESSAGE_LENGTH,
    TEXTURE_USAGE_RENDER_ATTACHMENT,
    WgpuSwapchain,
  };

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6WgpuSwapchain = exportedApi;
  } else if (root) {
    root.PlayGta6WgpuSwapchain = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
