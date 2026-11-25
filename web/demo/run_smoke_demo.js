// Turnkey WebGPU smoke demo runner wiring the G5 binary command-stream scene,
// SharedArrayBuffer GpuRingBuffer, WgpuSwapchain, WgpuCommandProcessor,
// optional GpuFrameRecorder (.pg6trace capture), and optional PerfMonitor + HUD.
// Zero external dependencies; ES module compatible with both Browser and Node.js.

import {
  GPU_OPCODES,
  SmokeSceneController,
  submitToRing,
} from './smoke_scene.js';
import { PerfMonitor } from '../telemetry/perf_monitor.js';

// Side-effect imports register globalThis.PlayGta6* exports in both Browser ESM and Node.js ESM.
import '../workers/gpu_ring.js';
import '../workers/wgpu_swapchain.js';
import '../workers/wgpu_worker.js';
import '../workers/gpu_recorder.js';

const DEFAULT_WIDTH = 640;
const DEFAULT_HEIGHT = 360;
const DEFAULT_FORMAT = 'bgra8unorm';
const DEFAULT_RING_CAPACITY_BYTES = 256 * 1024;
const PACKET_HEADER_BYTES = 8;
const DEFAULT_BUDGET_WINDOW_SIZE = 300;

export const DEFAULT_TARGET_FPS = 30;
export const DEFAULT_FRAME_BUDGET_MS = 33.333;

function defaultNow() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

function defaultRaf(callback) {
  if (typeof globalThis !== 'undefined' && typeof globalThis.requestAnimationFrame === 'function') {
    return globalThis.requestAnimationFrame(callback);
  }
  return globalThis.setTimeout(() => callback(defaultNow()), 16);
}

function defaultCancelRaf(handle) {
  if (typeof globalThis !== 'undefined' && typeof globalThis.cancelAnimationFrame === 'function') {
    globalThis.cancelAnimationFrame(handle);
    return;
  }
  globalThis.clearTimeout(handle);
}

function defaultSetTimeout(callback, ms) {
  return globalThis.setTimeout(callback, ms);
}

function defaultClearTimeout(handle) {
  globalThis.clearTimeout(handle);
}

function round3(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return 0;
  }
  return Math.round(value * 1000) / 1000;
}

function roundPct(withinCount, totalCount) {
  if (
    typeof withinCount !== 'number' ||
    typeof totalCount !== 'number' ||
    !Number.isFinite(withinCount) ||
    !Number.isFinite(totalCount) ||
    totalCount <= 0
  ) {
    return 0;
  }
  const pct = Math.round((withinCount / totalCount) * 10000) / 100;
  return pct < 0 ? 0 : pct > 100 ? 100 : pct;
}

function sanitizeNonNegativeInt(value, fallback = 0) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    return fallback;
  }
  return Math.floor(value);
}

function validateAbortSignalOption(signal, label = 'signal') {
  if (signal === null || signal === undefined) {
    return null;
  }
  if (typeof signal !== 'object' || typeof signal.aborted !== 'boolean') {
    throw new TypeError(`${label} must be an AbortSignal-compatible object or null`);
  }
  return signal;
}

function validateTimeoutMsOption(timeoutMs, label = 'timeoutMs') {
  if (timeoutMs === null || timeoutMs === undefined) {
    return null;
  }
  if (typeof timeoutMs !== 'number' || !Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    throw new RangeError(`${label} must be a positive finite number, got ${timeoutMs}`);
  }
  return timeoutMs;
}

function resolveGpuRingBuffer() {
  const ctor = globalThis.PlayGta6GpuRing && globalThis.PlayGta6GpuRing.GpuRingBuffer;
  if (typeof ctor !== 'function') {
    throw new Error('GpuRingBuffer is unavailable; ensure web/workers/gpu_ring.js is loaded');
  }
  return ctor;
}

function resolveWgpuSwapchain() {
  const ctor = globalThis.PlayGta6WgpuSwapchain && globalThis.PlayGta6WgpuSwapchain.WgpuSwapchain;
  if (typeof ctor !== 'function') {
    throw new Error('WgpuSwapchain is unavailable; ensure web/workers/wgpu_swapchain.js is loaded');
  }
  return ctor;
}

function resolveWgpuCommandProcessor() {
  const ctor = globalThis.PlayGta6Wgpu && globalThis.PlayGta6Wgpu.WgpuCommandProcessor;
  if (typeof ctor !== 'function') {
    throw new Error('WgpuCommandProcessor is unavailable; ensure web/workers/wgpu_worker.js is loaded');
  }
  return ctor;
}

function resolveGpuFrameRecorder() {
  const ctor = globalThis.PlayGta6GpuRecorder && globalThis.PlayGta6GpuRecorder.GpuFrameRecorder;
  if (typeof ctor !== 'function') {
    throw new Error('GpuFrameRecorder is unavailable; ensure web/workers/gpu_recorder.js is loaded');
  }
  return ctor;
}

/**
 * Inspects a binary GpuPacketHeader stream without heap allocations beyond the
 * returned summary object, counting packets, draw calls, and compute dispatches.
 *
 * @param {Uint8Array} bytes
 * @returns {{ packets: number, drawCalls: number, computeDispatches: number, bytes: number }}
 */
export function inspectCommandBuffer(bytes) {
  const byteLength = bytes ? bytes.byteLength : 0;
  if (byteLength < PACKET_HEADER_BYTES) {
    return { packets: 0, drawCalls: 0, computeDispatches: 0, bytes: byteLength };
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, byteLength);
  let offset = 0;
  let packets = 0;
  let drawCalls = 0;
  let computeDispatches = 0;

  while (offset + PACKET_HEADER_BYTES <= byteLength) {
    const opcode = view.getUint16(offset, true);
    const payloadBytes = view.getUint32(offset + 4, true);
    const nextOffset = offset + PACKET_HEADER_BYTES + payloadBytes;
    if (nextOffset > byteLength) {
      break;
    }

    packets += 1;
    if (opcode === GPU_OPCODES.DRAW || opcode === GPU_OPCODES.DRAW_INDEXED) {
      drawCalls += 1;
    } else if (opcode === GPU_OPCODES.DISPATCH_COMPUTE) {
      computeDispatches += 1;
    }

    offset = nextOffset;
  }

  return { packets, drawCalls, computeDispatches, bytes: byteLength };
}

/**
 * Deterministically computes 30 FPS frame-budget summary telemetry from per-frame
 * delta durations in milliseconds, with explicit zero-frame and invalid-sample guards.
 *
 * @param {ArrayLike<number>|Iterable<number>|null|undefined} frameDeltasMs
 * @param {object} [options]
 * @param {number} [options.frameBudgetMs=33.333] Target per-frame budget in milliseconds.
 * @param {number} [options.totalFrames] Total presented frames (defaults to valid delta count).
 * @param {number} [options.nonBlackFrames=0] Count of verified non-black presented frames.
 * @param {boolean} [options.nonBlackFrameVerified] Explicit non-black verification flag.
 * @returns {{
 *   frameBudgetMs: number,
 *   targetFps: number,
 *   totalFrames: number,
 *   sampleCount: number,
 *   presentFpsAvg: number,
 *   avgFrameMs: number,
 *   minFrameMs: number,
 *   maxFrameMs: number,
 *   p95FrameMs: number,
 *   framesWithinBudget: number,
 *   framesOverBudget: number,
 *   budgetCompliancePct: number,
 *   nonBlackFrames: number,
 *   nonBlackFrameVerified: boolean,
 * }}
 */
export function computeFrameBudgetSummary(frameDeltasMs, options = {}) {
  const opts = options && typeof options === 'object' ? options : {};
  const effectiveBudgetMs =
    typeof opts.frameBudgetMs === 'number' &&
    Number.isFinite(opts.frameBudgetMs) &&
    opts.frameBudgetMs > 0
      ? opts.frameBudgetMs
      : DEFAULT_FRAME_BUDGET_MS;
  const targetFps =
    effectiveBudgetMs === DEFAULT_FRAME_BUDGET_MS
      ? DEFAULT_TARGET_FPS
      : round3(1000 / effectiveBudgetMs);

  const validDeltas = [];
  if (frameDeltasMs !== null && frameDeltasMs !== undefined && typeof frameDeltasMs !== 'string') {
    if (Array.isArray(frameDeltasMs) || ArrayBuffer.isView(frameDeltasMs)) {
      for (let i = 0; i < frameDeltasMs.length; i += 1) {
        const dt = frameDeltasMs[i];
        if (typeof dt === 'number' && Number.isFinite(dt) && dt >= 0) {
          validDeltas.push(dt);
        }
      }
    } else if (typeof frameDeltasMs[Symbol.iterator] === 'function') {
      for (const dt of frameDeltasMs) {
        if (typeof dt === 'number' && Number.isFinite(dt) && dt >= 0) {
          validDeltas.push(dt);
        }
      }
    }
  }

  const sampleCount = validDeltas.length;
  const totalFrames =
    opts.totalFrames !== undefined
      ? sanitizeNonNegativeInt(opts.totalFrames, sampleCount)
      : sampleCount;
  const nonBlackFrames = sanitizeNonNegativeInt(opts.nonBlackFrames, 0);
  const nonBlackFrameVerified =
    typeof opts.nonBlackFrameVerified === 'boolean'
      ? opts.nonBlackFrameVerified
      : nonBlackFrames > 0;

  if (sampleCount === 0) {
    return {
      frameBudgetMs: effectiveBudgetMs,
      targetFps,
      totalFrames,
      sampleCount: 0,
      presentFpsAvg: 0,
      avgFrameMs: 0,
      minFrameMs: 0,
      maxFrameMs: 0,
      p95FrameMs: 0,
      framesWithinBudget: 0,
      framesOverBudget: 0,
      budgetCompliancePct: 0,
      nonBlackFrames,
      nonBlackFrameVerified,
    };
  }

  let sumMs = 0;
  let framesWithinBudget = 0;
  let framesOverBudget = 0;
  for (let i = 0; i < sampleCount; i += 1) {
    const dt = validDeltas[i];
    sumMs += dt;
    if (dt <= effectiveBudgetMs) {
      framesWithinBudget += 1;
    } else {
      framesOverBudget += 1;
    }
  }

  validDeltas.sort((a, b) => a - b);
  const p95Index = Math.min(sampleCount - 1, Math.max(0, Math.ceil(0.95 * sampleCount) - 1));
  const rawAvgFrameMs = sumMs / sampleCount;
  const presentFpsAvg = rawAvgFrameMs > 0 ? round3(1000 / rawAvgFrameMs) : 0;

  return {
    frameBudgetMs: effectiveBudgetMs,
    targetFps,
    totalFrames,
    sampleCount,
    presentFpsAvg,
    avgFrameMs: round3(rawAvgFrameMs),
    minFrameMs: round3(validDeltas[0]),
    maxFrameMs: round3(validDeltas[sampleCount - 1]),
    p95FrameMs: round3(validDeltas[p95Index]),
    framesWithinBudget,
    framesOverBudget,
    budgetCompliancePct: roundPct(framesWithinBudget, sampleCount),
    nonBlackFrames,
    nonBlackFrameVerified,
  };
}

/**
 * Turnkey WebGPU smoke demo driver connecting:
 * - SmokeSceneController (G5 binary command encoder)
 * - GpuRingBuffer (SharedArrayBuffer SPSC command ring)
 * - WgpuSwapchain (DPR-aware canvas/depth-stencil lifecycle + pacing)
 * - WgpuCommandProcessor (binary opcode -> WebGPU command executor)
 * - Optional GpuFrameRecorder (live .pg6trace frame capture)
 * - Optional PerfMonitor + HUD metrics bridge
 */
export class SmokeDemoRunner {
  /**
   * @param {object} options
   * @param {GPUDevice} options.device
   * @param {GPUCanvasContext} [options.context]
   * @param {HTMLCanvasElement|OffscreenCanvas|object} [options.canvas]
   * @param {string} [options.format='bgra8unorm']
   * @param {number} [options.width=640]
   * @param {number} [options.height=360]
   * @param {number} [options.devicePixelRatio=1]
   * @param {number} [options.ringCapacityBytes=262144]
   * @param {SharedArrayBuffer|ArrayBuffer|ArrayBufferView|object|null} [options.inputSab=null]
   * @param {PerfMonitor|boolean|null} [options.perfMonitor=null]
   * @param {{ updateMetrics: (metrics: object) => void }|null} [options.hud=null]
   * @param {boolean} [options.enableRecorder=false]
   * @param {object} [options.recorderOptions]
   * @param {number} [options.textureSize]
   * @param {boolean} [options.enableDepth=true]
   * @param {string} [options.depthFormat]
   * @param {string} [options.alphaMode]
   * @param {(cb: (ts: number) => void) => any} [options.raf]
   * @param {(handle: any) => void} [options.cancelRaf]
   * @param {() => number} [options.now]
   * @param {number} [options.frameBudgetMs=33.333]
   * @param {number} [options.budgetWindowSize=300]
   * @param {AbortSignal|null} [options.signal=null]
   * @param {number|null} [options.timeoutMs=null]
   * @param {(cb: () => void, ms: number) => any} [options.setTimeout]
   * @param {(handle: any) => void} [options.clearTimeout]
   * @param {(info: object) => void|null} [options.onTeardown=null]
   * @param {(frameInfo: object) => boolean|null} [options.verifyNonBlackFrame=null]
   */
  constructor(options = {}) {
    if (!options || typeof options !== 'object') {
      throw new TypeError('SmokeDemoRunner requires an options object');
    }

    const {
      device,
      context = null,
      canvas = null,
      format = DEFAULT_FORMAT,
      width = DEFAULT_WIDTH,
      height = DEFAULT_HEIGHT,
      devicePixelRatio = 1,
      ringCapacityBytes = DEFAULT_RING_CAPACITY_BYTES,
      inputSab = null,
      perfMonitor = null,
      hud = null,
      enableRecorder = false,
      recorderOptions = {},
      textureSize,
      enableDepth = true,
      depthFormat,
      alphaMode,
      raf = defaultRaf,
      cancelRaf = defaultCancelRaf,
      now = defaultNow,
      frameBudgetMs = DEFAULT_FRAME_BUDGET_MS,
      budgetWindowSize = DEFAULT_BUDGET_WINDOW_SIZE,
      signal = null,
      timeoutMs = null,
      setTimeout: setTimeoutImpl = defaultSetTimeout,
      clearTimeout: clearTimeoutImpl = defaultClearTimeout,
      onTeardown = null,
      verifyNonBlackFrame = null,
    } = options;

    if (!device || typeof device !== 'object') {
      throw new TypeError('SmokeDemoRunner requires a GPUDevice');
    }
    if (typeof width !== 'number' || !Number.isFinite(width) || width <= 0) {
      throw new RangeError(`width must be a positive finite number, got ${width}`);
    }
    if (typeof height !== 'number' || !Number.isFinite(height) || height <= 0) {
      throw new RangeError(`height must be a positive finite number, got ${height}`);
    }
    if (
      typeof devicePixelRatio !== 'number' ||
      !Number.isFinite(devicePixelRatio) ||
      devicePixelRatio <= 0
    ) {
      throw new RangeError(
        `devicePixelRatio must be a positive finite number, got ${devicePixelRatio}`
      );
    }
    if (typeof raf !== 'function') {
      throw new TypeError('raf must be a function');
    }
    if (typeof cancelRaf !== 'function') {
      throw new TypeError('cancelRaf must be a function');
    }
    if (typeof now !== 'function') {
      throw new TypeError('now must be a function returning milliseconds');
    }
    if (
      typeof frameBudgetMs !== 'number' ||
      !Number.isFinite(frameBudgetMs) ||
      frameBudgetMs <= 0
    ) {
      throw new RangeError(`frameBudgetMs must be a positive finite number, got ${frameBudgetMs}`);
    }
    if (!Number.isInteger(budgetWindowSize) || budgetWindowSize < 1) {
      throw new RangeError(
        `budgetWindowSize must be a positive integer, got ${budgetWindowSize}`
      );
    }
    const resolvedSignal = validateAbortSignalOption(signal, 'signal');
    const resolvedTimeoutMs = validateTimeoutMsOption(timeoutMs, 'timeoutMs');
    if (typeof setTimeoutImpl !== 'function') {
      throw new TypeError('setTimeout must be a function');
    }
    if (typeof clearTimeoutImpl !== 'function') {
      throw new TypeError('clearTimeout must be a function');
    }
    if (onTeardown !== null && onTeardown !== undefined && typeof onTeardown !== 'function') {
      throw new TypeError('onTeardown must be a function or null');
    }
    if (
      verifyNonBlackFrame !== null &&
      verifyNonBlackFrame !== undefined &&
      typeof verifyNonBlackFrame !== 'function'
    ) {
      throw new TypeError('verifyNonBlackFrame must be a function or null');
    }
    if (hud !== null && hud !== undefined) {
      if (typeof hud !== 'object' || typeof hud.updateMetrics !== 'function') {
        throw new TypeError('hud must be an object with an updateMetrics(metrics) function');
      }
    }

    let resolvedPerfMonitor = null;
    if (perfMonitor === true) {
      resolvedPerfMonitor = new PerfMonitor({ now });
    } else if (perfMonitor !== null && perfMonitor !== undefined && perfMonitor !== false) {
      if (typeof perfMonitor !== 'object') {
        throw new TypeError('perfMonitor must be a PerfMonitor instance, boolean, or null');
      }
      resolvedPerfMonitor = perfMonitor;
    }

    let resolvedCanvas = canvas;
    if (!resolvedCanvas && context && typeof context === 'object') {
      resolvedCanvas =
        context.canvas && typeof context.canvas === 'object'
          ? context.canvas
          : {
              width: Math.max(1, Math.round(width * devicePixelRatio)),
              height: Math.max(1, Math.round(height * devicePixelRatio)),
              getContext() {
                return context;
              },
            };
    }
    if (!resolvedCanvas || typeof resolvedCanvas !== 'object') {
      throw new TypeError('SmokeDemoRunner requires a canvas or WebGPU context');
    }

    const resolvedContext =
      context || (typeof resolvedCanvas.getContext === 'function' ? resolvedCanvas.getContext('webgpu') : null);
    if (!resolvedContext || typeof resolvedContext.configure !== 'function') {
      throw new TypeError('SmokeDemoRunner requires a WebGPU GPUCanvasContext');
    }

    const GpuRingBuffer = resolveGpuRingBuffer();
    const WgpuSwapchain = resolveWgpuSwapchain();
    const WgpuCommandProcessor = resolveWgpuCommandProcessor();

    this._device = device;
    this._raf = raf;
    this._cancelRaf = cancelRaf;
    this._now = now;
    this._setTimeout = setTimeoutImpl;
    this._clearTimeout = clearTimeoutImpl;
    this._inputSab = inputSab;
    this._perfMonitor = resolvedPerfMonitor;
    this._hud = hud || null;
    this._verifyNonBlackFrame = verifyNonBlackFrame || null;

    this._width = width;
    this._height = height;
    this._devicePixelRatio = devicePixelRatio;

    this._frameBudgetMs = frameBudgetMs;
    this._frameDeltasRing = new Float64Array(budgetWindowSize);
    this._frameDeltaHead = 0;
    this._frameDeltaRingCount = 0;
    this._totalDeltaSamples = 0;
    this._totalDeltaSumMs = 0;
    this._minDeltaMs = Infinity;
    this._maxDeltaMs = 0;
    this._framesWithinBudget = 0;
    this._framesOverBudget = 0;
    this._nonBlackFrames = 0;
    this._nonBlackFrameVerified = false;

    this._signal = resolvedSignal;
    this._runSignal = null;
    this._timeoutMs = resolvedTimeoutMs;
    this._timeoutHandle = null;
    this._aborted = false;
    this._timedOut = false;
    this._teardownReason = null;
    this._teardownError = null;
    this._teardownNotified = false;
    this._onTeardown = onTeardown || null;
    this._boundAbortHandler = () => {
      this._handleAbort();
    };

    this._ring = GpuRingBuffer.create(ringCapacityBytes);

    const swapchainConfig = {
      canvas: resolvedCanvas,
      device,
      context: resolvedContext,
      format,
      enableDepth,
      now,
    };
    if (depthFormat !== undefined) {
      swapchainConfig.depthFormat = depthFormat;
    }
    if (alphaMode !== undefined) {
      swapchainConfig.alphaMode = alphaMode;
    }
    this._swapchain = new WgpuSwapchain(swapchainConfig);

    this._processor = new WgpuCommandProcessor(device, this._swapchain.context, {
      format: this._swapchain.format,
      GpuRingBuffer,
    });
    this._processor.attachRing(this._ring);

    const sceneOpts = { aspect: width / height };
    if (textureSize !== undefined) {
      sceneOpts.textureSize = textureSize;
    }
    this._scene = new SmokeSceneController(sceneOpts);

    this._recorderOptions =
      recorderOptions && typeof recorderOptions === 'object' ? { ...recorderOptions } : {};
    this._recorder = null;
    this._recorderHandle = null;

    this._initialized = false;
    this._running = false;
    this._destroyed = false;
    this._rafHandle = null;
    this._frameIndex = 0;
    this._startTimestampMs = null;
    this._lastTimestampMs = null;
    this._totalGpuPackets = 0;
    this._totalGpuBytes = 0;

    if (enableRecorder) {
      this._ensureRecorder(true);
    }

    if (this._signal) {
      if (this._signal.aborted) {
        this._handleAbort();
      } else if (typeof this._signal.addEventListener === 'function') {
        this._signal.addEventListener('abort', this._boundAbortHandler, { once: true });
      }
    }
  }

  get initialized() {
    return this._initialized;
  }

  get running() {
    return this._running;
  }

  get destroyed() {
    return this._destroyed;
  }

  get aborted() {
    return this._aborted;
  }

  get timedOut() {
    return this._timedOut;
  }

  get teardownReason() {
    return this._teardownReason;
  }

  get frameBudgetMs() {
    return this._frameBudgetMs;
  }

  get frameIndex() {
    return this._frameIndex;
  }

  get ring() {
    return this._ring;
  }

  get swapchain() {
    return this._swapchain;
  }

  get processor() {
    return this._processor;
  }

  get scene() {
    return this._scene;
  }

  get recorder() {
    return this._recorder;
  }

  get perfMonitor() {
    return this._perfMonitor;
  }

  get inputSab() {
    return this._inputSab;
  }

  set inputSab(value) {
    this._inputSab = value;
  }

  _assertNotDestroyed(action) {
    if (this._destroyed) {
      throw new Error(`Cannot ${action} a destroyed SmokeDemoRunner`);
    }
  }

  _clearTimeoutHandle() {
    if (this._timeoutHandle !== null) {
      const handle = this._timeoutHandle;
      this._timeoutHandle = null;
      this._clearTimeout(handle);
    }
  }

  _detachConstructorSignal() {
    if (this._signal && typeof this._signal.removeEventListener === 'function') {
      this._signal.removeEventListener('abort', this._boundAbortHandler);
    }
    this._signal = null;
  }

  _detachRunSignal() {
    if (this._runSignal && typeof this._runSignal.removeEventListener === 'function') {
      this._runSignal.removeEventListener('abort', this._boundAbortHandler);
    }
    this._runSignal = null;
  }

  _isAnySignalAborted() {
    return Boolean(
      (this._signal && this._signal.aborted) || (this._runSignal && this._runSignal.aborted)
    );
  }

  _handleAbort() {
    if (this._destroyed) {
      return;
    }
    this._aborted = true;
    this.destroy('abort');
  }

  _handleTimeout() {
    this._timeoutHandle = null;
    if (this._destroyed) {
      return;
    }
    this._timedOut = true;
    this.destroy('timeout');
  }

  _recordFrameDelta(deltaMs) {
    const safeDt = typeof deltaMs === 'number' && Number.isFinite(deltaMs) && deltaMs >= 0 ? deltaMs : 0;
    const ring = this._frameDeltasRing;
    ring[this._frameDeltaHead] = safeDt;
    this._frameDeltaHead = (this._frameDeltaHead + 1) % ring.length;
    if (this._frameDeltaRingCount < ring.length) {
      this._frameDeltaRingCount += 1;
    }
    this._totalDeltaSamples += 1;
    this._totalDeltaSumMs += safeDt;
    if (safeDt < this._minDeltaMs) {
      this._minDeltaMs = safeDt;
    }
    if (safeDt > this._maxDeltaMs) {
      this._maxDeltaMs = safeDt;
    }
    if (safeDt <= this._frameBudgetMs) {
      this._framesWithinBudget += 1;
    } else {
      this._framesOverBudget += 1;
    }
  }

  _ensureRecorder(autoStart = false) {
    this._assertNotDestroyed('attach recorder to');
    if (!this._recorder) {
      const GpuFrameRecorder = resolveGpuFrameRecorder();
      this._recorder = new GpuFrameRecorder({
        ...this._recorderOptions,
        autoStart: Boolean(autoStart),
      });
      this._recorderHandle = this._recorder.createInterceptor(this._processor);
    }
    return this._recorder;
  }

  /**
   * Configures the swapchain, submits the G5 scene init command buffer through
   * the SharedArrayBuffer GpuRingBuffer, and drains it into WgpuCommandProcessor.
   * Safe to call multiple times (idempotent).
   */
  init() {
    this._assertNotDestroyed('initialize');
    if (this._initialized) {
      return {
        initialized: true,
        alreadyInitialized: true,
        bytesSubmitted: 0,
        bytesDrained: 0,
        swapchain: this._swapchain.getTelemetry(),
      };
    }

    const swapConfig = this._swapchain.configure({
      width: this._width,
      height: this._height,
      devicePixelRatio: this._devicePixelRatio,
      format: this._swapchain.format,
    });

    if (swapConfig.width > 0 && swapConfig.height > 0) {
      this._scene.aspect = swapConfig.width / swapConfig.height;
    }

    const initBytes = this._scene.buildInitCommandBuffer();
    const committed = submitToRing(this._ring, initBytes);
    if (!committed) {
      throw new Error(
        `GpuRingBuffer overflow during init: failed to write ${initBytes.byteLength} bytes (capacity ${this._ring.capacityBytes})`
      );
    }

    const drained = this._processor.drainRing();
    const initBatch = inspectCommandBuffer(initBytes);
    this._totalGpuPackets += initBatch.packets;
    this._totalGpuBytes += initBatch.bytes;
    this._initialized = true;

    return {
      initialized: true,
      alreadyInitialized: false,
      bytesSubmitted: initBytes.byteLength,
      bytesDrained: drained.bytesDrained,
      swapchain: swapConfig,
    };
  }

  /**
   * Encodes and executes a single animated G5 frame through the ring buffer into
   * WgpuCommandProcessor, recording swapchain pacing, PerfMonitor telemetry, and
   * optional HUD metrics.
   *
   * @param {number} [timestampMs] Optional explicit monotonic timestamp in ms.
   */
  stepFrame(timestampMs) {
    this._assertNotDestroyed('step frame on');
    if (!this._initialized) {
      this.init();
    }

    const ts =
      typeof timestampMs === 'number' && Number.isFinite(timestampMs)
        ? timestampMs
        : this._now();

    if (this._startTimestampMs === null) {
      this._startTimestampMs = ts;
    }
    const timeSec = Math.max(0, (ts - this._startTimestampMs) / 1000);
    const currentFrameIndex = this._frameIndex;

    const frameBytes = this._scene.buildFrameCommandBuffer(
      currentFrameIndex,
      timeSec,
      this._inputSab
    );
    const committed = submitToRing(this._ring, frameBytes);
    if (!committed) {
      throw new Error(
        `GpuRingBuffer overflow on frame ${currentFrameIndex}: failed to write ${frameBytes.byteLength} bytes (capacity ${this._ring.capacityBytes})`
      );
    }

    const drained = this._processor.drainRing();
    let frameDeltaMs = null;
    if (this._lastTimestampMs !== null) {
      const rawDelta = ts - this._lastTimestampMs;
      frameDeltaMs = Number.isFinite(rawDelta) && rawDelta > 0 ? rawDelta : 0;
      this._recordFrameDelta(frameDeltaMs);
    }
    this._frameIndex += 1;
    this._lastTimestampMs = ts;

    const batch = inspectCommandBuffer(frameBytes);
    this._totalGpuPackets += batch.packets;
    this._totalGpuBytes += batch.bytes;

    const frameNonBlack = this._verifyNonBlackFrame
      ? Boolean(
          this._verifyNonBlackFrame({
            frameIndex: currentFrameIndex,
            timestampMs: ts,
            presents: drained.presents,
            drawCalls: batch.drawCalls,
            computeDispatches: batch.computeDispatches,
            packets: batch.packets,
            bytes: frameBytes,
          })
        )
      : drained.presents > 0 && batch.drawCalls > 0;

    if (frameNonBlack) {
      this._nonBlackFrames += 1;
      this._nonBlackFrameVerified = true;
    }

    const swapTelemetry = this._swapchain.recordFramePresented(ts, {
      hasVisibleContent: frameNonBlack,
      hasDraws: batch.drawCalls > 0,
    });

    if (this._perfMonitor) {
      if (typeof this._perfMonitor.recordGpuBatch === 'function') {
        this._perfMonitor.recordGpuBatch({
          packets: batch.packets,
          drawCalls: batch.drawCalls,
          computeDispatches: batch.computeDispatches,
          bytes: batch.bytes,
          timestampMs: ts,
        });
      }
      if (typeof this._perfMonitor.recordFrame === 'function') {
        this._perfMonitor.recordFrame(ts);
      }
    }

    if (this._hud && typeof this._hud.updateMetrics === 'function') {
      const metrics =
        this._perfMonitor && typeof this._perfMonitor.toHudMetrics === 'function'
          ? this._perfMonitor.toHudMetrics()
          : {
              fps: swapTelemetry.fps,
              gpuPackets: this._totalGpuPackets,
              avgFrameTimeMs: swapTelemetry.avgFrameTimeMs,
              maxFrameTimeMs: swapTelemetry.maxFrameTimeMs,
              totalFrames: swapTelemetry.totalFrames,
            };
      this._hud.updateMetrics(metrics);
    }

    return {
      frameIndex: currentFrameIndex,
      timestampMs: ts,
      timeSec,
      frameDeltaMs,
      bytesSubmitted: frameBytes.byteLength,
      bytesDrained: drained.bytesDrained,
      presents: drained.presents,
      packets: batch.packets,
      drawCalls: batch.drawCalls,
      computeDispatches: batch.computeDispatches,
      nonBlackFrameVerified: frameNonBlack,
    };
  }

  /**
   * Starts the requestAnimationFrame render loop. Safe against double-start.
   * Optionally accepts per-run `{ signal, timeoutMs }` abort/timeout teardown guards.
   *
   * @param {{ signal?: AbortSignal|null, timeoutMs?: number|null }|null} [startOptions]
   * @returns {boolean} True if loop transitioned from stopped to running.
   */
  start(startOptions = null) {
    if (startOptions !== null && startOptions !== undefined && typeof startOptions !== 'object') {
      throw new TypeError('start options must be an object or null');
    }
    const runSignal =
      startOptions && startOptions.signal !== undefined
        ? validateAbortSignalOption(startOptions.signal, 'startOptions.signal')
        : null;
    const startTimeoutMs =
      startOptions && startOptions.timeoutMs !== undefined
        ? validateTimeoutMsOption(startOptions.timeoutMs, 'startOptions.timeoutMs')
        : null;

    this._assertNotDestroyed('start');
    if (this._running) {
      return false;
    }
    if ((this._signal && this._signal.aborted) || (runSignal && runSignal.aborted)) {
      this._handleAbort();
      return false;
    }
    if (!this._initialized) {
      this.init();
    }

    this._detachRunSignal();
    if (runSignal && runSignal !== this._signal) {
      this._runSignal = runSignal;
      if (typeof runSignal.addEventListener === 'function') {
        runSignal.addEventListener('abort', this._boundAbortHandler, { once: true });
      }
    }

    this._clearTimeoutHandle();
    const effectiveTimeoutMs = startTimeoutMs !== null ? startTimeoutMs : this._timeoutMs;
    if (effectiveTimeoutMs !== null) {
      this._timeoutHandle = this._setTimeout(() => {
        this._handleTimeout();
      }, effectiveTimeoutMs);
    }

    this._running = true;

    const tick = (rafTimestamp) => {
      if (!this._running || this._destroyed) {
        return;
      }
      if (this._isAnySignalAborted()) {
        this._handleAbort();
        return;
      }
      this._rafHandle = null;
      const ts =
        typeof rafTimestamp === 'number' && Number.isFinite(rafTimestamp)
          ? rafTimestamp
          : this._now();
      try {
        this.stepFrame(ts);
      } catch (err) {
        this.stop();
        throw err;
      }
      if (this._running && !this._destroyed) {
        this._rafHandle = this._raf(tick);
      }
    };

    this._rafHandle = this._raf(tick);
    return true;
  }

  /**
   * Stops the requestAnimationFrame render loop and clears any pending timeout
   * or per-run abort subscription. Safe against double-stop.
   *
   * @returns {boolean} True if loop transitioned from running to stopped.
   */
  stop() {
    const wasRunning = this._running;
    this._running = false;
    if (this._rafHandle !== null) {
      const handle = this._rafHandle;
      this._rafHandle = null;
      this._cancelRaf(handle);
    }
    this._clearTimeoutHandle();
    this._detachRunSignal();
    return wasRunning;
  }

  /**
   * Resizes the swapchain and updates scene aspect ratio.
   *
   * @param {number} cssWidth
   * @param {number} cssHeight
   * @param {number} [dpr]
   * @returns {boolean} True if swapchain reconfigured, false on no-op.
   */
  resize(cssWidth, cssHeight, dpr) {
    this._assertNotDestroyed('resize');
    const nextDpr = dpr !== undefined ? dpr : this._devicePixelRatio;
    const resized = this._swapchain.resize(cssWidth, cssHeight, nextDpr);

    if (typeof cssWidth === 'number' && Number.isFinite(cssWidth) && cssWidth > 0) {
      this._width = cssWidth;
    }
    if (typeof cssHeight === 'number' && Number.isFinite(cssHeight) && cssHeight > 0) {
      this._height = cssHeight;
    }
    if (typeof nextDpr === 'number' && Number.isFinite(nextDpr) && nextDpr > 0) {
      this._devicePixelRatio = nextDpr;
    }

    if (this._swapchain.width > 0 && this._swapchain.height > 0) {
      this._scene.aspect = this._swapchain.width / this._swapchain.height;
    }
    return resized;
  }

  /**
   * Begins capturing rendered frames via GpuFrameRecorder. Lazily attaches the
   * recorder interceptor to WgpuCommandProcessor if not already attached.
   */
  startRecording() {
    const recorder = this._ensureRecorder(false);
    recorder.start();
    return this;
  }

  /**
   * Pauses frame capture in GpuFrameRecorder while retaining recorded frames.
   */
  stopRecording() {
    if (this._recorder) {
      this._recorder.stop();
    }
    return this;
  }

  /**
   * Exports captured frames as a `.pg6trace` Uint8Array directly readable by
   * `tools/gpu_trace.py`.
   *
   * @param {{ frameStart?: number, frameCount?: number }} [options]
   * @returns {Uint8Array}
   */
  exportTraceBytes(options) {
    if (!this._recorder) {
      return new Uint8Array(0);
    }
    return this._recorder.exportTraceBytes(options);
  }

  /**
   * Returns JSON-safe trace capture metadata from GpuFrameRecorder, or null if
   * recording has not been enabled.
   */
  exportTraceMetadata() {
    if (!this._recorder) {
      return null;
    }
    return this._recorder.exportMetadata();
  }

  /**
   * Stops the render loop, clears timeout/abort guards, submits and drains scene
   * teardown commands to free GPU resources, destroys the swapchain, and detaches
   * the frame recorder. Safe to call multiple times (idempotent).
   *
   * @param {string} [reason='manual'] Teardown reason (e.g. `'manual'`, `'abort'`, `'timeout'`).
   */
  destroy(reason = 'manual') {
    if (this._destroyed) {
      return;
    }

    const normalizedReason = typeof reason === 'string' && reason.length > 0 ? reason : 'manual';
    this._teardownReason = normalizedReason;
    if (normalizedReason === 'abort') {
      this._aborted = true;
    } else if (normalizedReason === 'timeout') {
      this._timedOut = true;
    }

    this.stop();
    this._detachConstructorSignal();
    this._destroyed = true;

    let teardownError = null;
    if (this._initialized) {
      try {
        const teardownBytes = this._scene.buildTeardownCommandBuffer();
        if (this._ring && this._processor) {
          if (this._ring.availableBytes() > 0) {
            this._processor.drainRing();
          }
          if (submitToRing(this._ring, teardownBytes)) {
            this._processor.drainRing();
          } else {
            this._processor.executeCommandBuffer(teardownBytes);
          }
        }
      } catch (err) {
        teardownError = err;
      } finally {
        this._initialized = false;
      }
    }

    if (this._recorder) {
      try {
        this._recorder.stop();
      } catch (err) {
        if (!teardownError) {
          teardownError = err;
        }
      }
    }
    if (this._recorderHandle) {
      const handle = this._recorderHandle;
      this._recorderHandle = null;
      try {
        handle.detach();
      } catch (err) {
        if (!teardownError) {
          teardownError = err;
        }
      }
    }

    if (this._swapchain) {
      try {
        this._swapchain.destroy();
      } catch (err) {
        if (!teardownError) {
          teardownError = err;
        }
      }
    }

    this._teardownError = teardownError;

    if (this._onTeardown && !this._teardownNotified) {
      this._teardownNotified = true;
      const cb = this._onTeardown;
      this._onTeardown = null;
      try {
        cb({
          reason: this._teardownReason,
          aborted: this._aborted,
          timedOut: this._timedOut,
          destroyed: this._destroyed,
          summary: this.getBudgetSummary(),
          teardownError,
        });
      } catch {
        // Isolate observer exceptions so teardown stays deterministic.
      }
    }
  }

  /**
   * Returns a deterministic 30 FPS frame-budget summary across all stepped frames,
   * safe to call before init (zero-frame guard), during execution, or after teardown.
   */
  getBudgetSummary() {
    if (this._totalDeltaSamples === 0) {
      return computeFrameBudgetSummary(null, {
        frameBudgetMs: this._frameBudgetMs,
        totalFrames: this._frameIndex,
        nonBlackFrames: this._nonBlackFrames,
        nonBlackFrameVerified: this._nonBlackFrameVerified,
      });
    }

    const ringCount = this._frameDeltaRingCount;
    if (this._totalDeltaSamples <= ringCount) {
      return computeFrameBudgetSummary(this._frameDeltasRing.subarray(0, ringCount), {
        frameBudgetMs: this._frameBudgetMs,
        totalFrames: this._frameIndex,
        nonBlackFrames: this._nonBlackFrames,
        nonBlackFrameVerified: this._nonBlackFrameVerified,
      });
    }

    const rollingSummary = computeFrameBudgetSummary(this._frameDeltasRing.subarray(0, ringCount), {
      frameBudgetMs: this._frameBudgetMs,
      totalFrames: this._frameIndex,
      nonBlackFrames: this._nonBlackFrames,
      nonBlackFrameVerified: this._nonBlackFrameVerified,
    });
    const rawAvgFrameMs = this._totalDeltaSumMs / this._totalDeltaSamples;
    return {
      ...rollingSummary,
      sampleCount: this._totalDeltaSamples,
      presentFpsAvg: rawAvgFrameMs > 0 ? round3(1000 / rawAvgFrameMs) : 0,
      avgFrameMs: round3(rawAvgFrameMs),
      minFrameMs: Number.isFinite(this._minDeltaMs) ? round3(this._minDeltaMs) : 0,
      maxFrameMs: round3(this._maxDeltaMs),
      framesWithinBudget: this._framesWithinBudget,
      framesOverBudget: this._framesOverBudget,
      budgetCompliancePct: roundPct(this._framesWithinBudget, this._totalDeltaSamples),
    };
  }

  /**
   * Returns a snapshot of runner lifecycle state and subsystem telemetry.
   */
  getStats() {
    return {
      initialized: this._initialized,
      running: this._running,
      destroyed: this._destroyed,
      aborted: this._aborted,
      timedOut: this._timedOut,
      teardownReason: this._teardownReason,
      frameIndex: this._frameIndex,
      budgetSummary: this.getBudgetSummary(),
      swapchain: this._swapchain ? this._swapchain.getTelemetry() : null,
      perf: this._perfMonitor ? this._perfMonitor.getSnapshot() : null,
      recorder: this._recorder ? this._recorder.exportMetadata() : null,
    };
  }
}

export default SmokeDemoRunner;
