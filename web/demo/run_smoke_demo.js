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
    this._inputSab = inputSab;
    this._perfMonitor = resolvedPerfMonitor;
    this._hud = hud || null;

    this._width = width;
    this._height = height;
    this._devicePixelRatio = devicePixelRatio;

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
    this._frameIndex += 1;
    this._lastTimestampMs = ts;

    const batch = inspectCommandBuffer(frameBytes);
    this._totalGpuPackets += batch.packets;
    this._totalGpuBytes += batch.bytes;

    const swapTelemetry = this._swapchain.recordFramePresented(ts);

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
      bytesSubmitted: frameBytes.byteLength,
      bytesDrained: drained.bytesDrained,
      presents: drained.presents,
      packets: batch.packets,
      drawCalls: batch.drawCalls,
      computeDispatches: batch.computeDispatches,
    };
  }

  /**
   * Starts the requestAnimationFrame render loop. Safe against double-start.
   *
   * @returns {boolean} True if loop transitioned from stopped to running.
   */
  start() {
    this._assertNotDestroyed('start');
    if (this._running) {
      return false;
    }
    if (!this._initialized) {
      this.init();
    }

    this._running = true;

    const tick = (rafTimestamp) => {
      if (!this._running || this._destroyed) {
        return;
      }
      this._rafHandle = null;
      const ts =
        typeof rafTimestamp === 'number' && Number.isFinite(rafTimestamp)
          ? rafTimestamp
          : this._now();
      this.stepFrame(ts);
      if (this._running && !this._destroyed) {
        this._rafHandle = this._raf(tick);
      }
    };

    this._rafHandle = this._raf(tick);
    return true;
  }

  /**
   * Stops the requestAnimationFrame render loop. Safe against double-stop.
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
   * Stops the render loop, submits and drains scene teardown commands to free
   * GPU resources, destroys the swapchain, and detaches the frame recorder.
   * Safe to call multiple times (idempotent).
   */
  destroy() {
    if (this._destroyed) {
      return;
    }

    this.stop();
    this._destroyed = true;

    if (this._initialized) {
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
      this._initialized = false;
    }

    if (this._recorder) {
      this._recorder.stop();
    }
    if (this._recorderHandle) {
      this._recorderHandle.detach();
      this._recorderHandle = null;
    }

    if (this._swapchain) {
      this._swapchain.destroy();
    }
  }

  /**
   * Returns a snapshot of runner lifecycle state and subsystem telemetry.
   */
  getStats() {
    return {
      initialized: this._initialized,
      running: this._running,
      frameIndex: this._frameIndex,
      swapchain: this._swapchain ? this._swapchain.getTelemetry() : null,
      perf: this._perfMonitor ? this._perfMonitor.getSnapshot() : null,
      recorder: this._recorder ? this._recorder.exportMetadata() : null,
    };
  }
}

export default SmokeDemoRunner;
