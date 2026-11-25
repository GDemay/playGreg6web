// Rolling frame telemetry ring and HUD performance profiler for playgta6web.
// Zero external dependencies; clean ES module compatible with <script type="module">
// and Node.js ESM imports. Designed for zero heap allocations on hot record*() paths.

export const DEFAULT_WINDOW_SIZE = 120;
export const DEFAULT_JANK_THRESHOLD_MS = 33.34;
export const DEFAULT_HUD_INTERVAL_MS = 250;
export const BYTES_PER_MB = 1024 * 1024;

function defaultNow() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

function defaultSetInterval(fn, ms) {
  if (typeof globalThis !== 'undefined' && typeof globalThis.setInterval === 'function') {
    return globalThis.setInterval(fn, ms);
  }
  throw new Error('setInterval is unavailable in this environment');
}

function defaultClearInterval(id) {
  if (typeof globalThis !== 'undefined' && typeof globalThis.clearInterval === 'function') {
    globalThis.clearInterval(id);
  }
}

function sanitizeNonNegativeNumber(value, fallback = 0) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    return fallback;
  }
  return value;
}

function percentileFromSorted(sorted, count, fraction) {
  if (count <= 0) {
    return 0;
  }
  const idx = Math.min(count - 1, Math.max(0, Math.ceil(fraction * count) - 1));
  return sanitizeNonNegativeNumber(sorted[idx], 0);
}

/**
 * Rolling performance monitor aggregating:
 * - Frame pacing (FPS, 1% low FPS, avg/min/p50/p95/p99/max frame time, jank frames) via fixed-capacity Float64Array ring
 * - GPU worker batch throughput (packets/sec, draw calls/frame, cumulative packet/draw/compute/byte totals)
 * - HTTP Range / OPFS I/O streaming throughput (MB/s, cache hit ratio, memory & OPFS hit counts)
 * - AudioWorklet ring telemetry (buffered frames, underruns, underrun frames, active voices)
 * - Optional periodic push to `window.playgta6Hud.updateMetrics(...)`
 */
export class PerfMonitor {
  /**
   * @param {object} [options]
   * @param {number} [options.windowSize=120] Fixed ring capacity for rolling frame statistics.
   * @param {number} [options.jankThresholdMs=33.34] Frame delta threshold above which a frame counts as jank.
   * @param {number} [options.gpuWindowSize] Fixed ring capacity for GPU batch rate aggregation (defaults to windowSize).
   * @param {number} [options.ioWindowSize] Fixed ring capacity for I/O throughput aggregation (defaults to windowSize).
   * @param {() => number} [options.now] Monotonic clock returning milliseconds.
   * @param {(fn: () => void, ms: number) => any} [options.setInterval] Injectable interval scheduler.
   * @param {(id: any) => void} [options.clearInterval] Injectable interval canceller.
   */
  constructor(options = {}) {
    if (options !== undefined && (options === null || typeof options !== 'object')) {
      throw new TypeError('PerfMonitor options must be an object');
    }

    const {
      windowSize = DEFAULT_WINDOW_SIZE,
      jankThresholdMs = DEFAULT_JANK_THRESHOLD_MS,
      gpuWindowSize = windowSize,
      ioWindowSize = windowSize,
      now = defaultNow,
      setInterval: setIntervalImpl = defaultSetInterval,
      clearInterval: clearIntervalImpl = defaultClearInterval,
    } = options;

    if (!Number.isInteger(windowSize) || windowSize < 1) {
      throw new RangeError(`windowSize must be a positive integer, got ${windowSize}`);
    }
    if (typeof jankThresholdMs !== 'number' || !Number.isFinite(jankThresholdMs) || jankThresholdMs <= 0) {
      throw new RangeError(`jankThresholdMs must be a positive finite number, got ${jankThresholdMs}`);
    }
    if (!Number.isInteger(gpuWindowSize) || gpuWindowSize < 1) {
      throw new RangeError(`gpuWindowSize must be a positive integer, got ${gpuWindowSize}`);
    }
    if (!Number.isInteger(ioWindowSize) || ioWindowSize < 1) {
      throw new RangeError(`ioWindowSize must be a positive integer, got ${ioWindowSize}`);
    }
    if (typeof now !== 'function') {
      throw new TypeError('now must be a function returning milliseconds');
    }
    if (typeof setIntervalImpl !== 'function') {
      throw new TypeError('setInterval must be a function');
    }
    if (typeof clearIntervalImpl !== 'function') {
      throw new TypeError('clearInterval must be a function');
    }

    this._windowSize = windowSize;
    this._jankThresholdMs = jankThresholdMs;
    this._gpuWindowSize = gpuWindowSize;
    this._ioWindowSize = ioWindowSize;
    this._now = now;
    this._setInterval = setIntervalImpl;
    this._clearInterval = clearIntervalImpl;

    // Frame timing ring buffers (pre-allocated, zero allocations on hot path).
    this._frameDeltas = new Float64Array(windowSize);
    this._scratchSorted = new Float64Array(windowSize);
    this._frameDrawCalls = new Float64Array(windowSize);

    // GPU batch rate ring buffers (pre-allocated).
    this._gpuPacketsRing = new Float64Array(gpuWindowSize);
    this._gpuDurationsRing = new Float64Array(gpuWindowSize);

    // I/O read throughput ring buffers (pre-allocated).
    this._ioBytesRing = new Float64Array(ioWindowSize);
    this._ioDurationsRing = new Float64Array(ioWindowSize);

    // HUD attachment state (preserved across reset()).
    this._hud = null;
    this._hudTimerId = null;

    this.reset();
  }

  get windowSize() {
    return this._windowSize;
  }

  get jankThresholdMs() {
    return this._jankThresholdMs;
  }

  get isHudAttached() {
    return this._hudTimerId !== null;
  }

  /**
   * Resets all frame, GPU, I/O, and audio telemetry counters and rolling rings.
   * Keeps any attached HUD timer active so subsequent ticks push fresh values.
   */
  reset() {
    const nowMs = this._now();

    this._frameDeltas.fill(0);
    this._scratchSorted.fill(0);
    this._frameDrawCalls.fill(0);
    this._frameWriteIdx = 0;
    this._frameCount = 0;
    this._frameDrawWriteIdx = 0;
    this._frameDrawCount = 0;
    this._frameDrawSum = 0;
    this._pendingFrameDrawCalls = 0;
    this._totalFrames = 0;
    this._jankFrames = 0;
    this._lastFrameTimestamp = null;

    this._gpuPacketsRing.fill(0);
    this._gpuDurationsRing.fill(0);
    this._gpuWriteIdx = 0;
    this._gpuCount = 0;
    this._gpuEpochMs = nowMs;
    this._gpuLastTimestamp = null;
    this._totalGpuBatches = 0;
    this._totalGpuPackets = 0;
    this._totalDrawCalls = 0;
    this._totalComputeDispatches = 0;
    this._totalGpuBytes = 0;

    this._ioBytesRing.fill(0);
    this._ioDurationsRing.fill(0);
    this._ioWriteIdx = 0;
    this._ioCount = 0;
    this._ioEpochMs = nowMs;
    this._ioLastTimestamp = null;
    this._totalBytesRead = 0;
    this._cacheHits = 0;
    this._opfsCacheHits = 0;
    this._cacheMisses = 0;

    this._audioBufferedFrames = 0;
    this._audioUnderruns = 0;
    this._audioUnderrunFrames = 0;
    this._audioActiveVoices = 0;
  }

  /**
   * Internal zero-allocation helper to commit a frame delta and flush pending draw calls.
   * @param {number} dtMs
   */
  _pushFrameDelta(dtMs) {
    const safeDtMs = sanitizeNonNegativeNumber(dtMs, 0);
    this._frameDeltas[this._frameWriteIdx] = safeDtMs;
    this._frameWriteIdx += 1;
    if (this._frameWriteIdx === this._windowSize) {
      this._frameWriteIdx = 0;
    }
    if (this._frameCount < this._windowSize) {
      this._frameCount += 1;
    }
    if (safeDtMs > this._jankThresholdMs) {
      this._jankFrames += 1;
    }
  }

  /**
   * Internal zero-allocation helper to flush per-frame draw calls into the rolling window.
   */
  _commitFrameDrawCalls() {
    const idx = this._frameDrawWriteIdx;
    if (this._frameDrawCount === this._windowSize) {
      this._frameDrawSum -= this._frameDrawCalls[idx];
    } else {
      this._frameDrawCount += 1;
    }
    const drawCalls = this._pendingFrameDrawCalls;
    this._frameDrawCalls[idx] = drawCalls;
    this._frameDrawSum += drawCalls;
    this._pendingFrameDrawCalls = 0;

    this._frameDrawWriteIdx = idx + 1;
    if (this._frameDrawWriteIdx === this._windowSize) {
      this._frameDrawWriteIdx = 0;
    }
  }

  /**
   * Records a frame presentation timestamp in milliseconds.
   * The first call establishes the baseline timestamp; subsequent calls record deltas
   * into the fixed-capacity Float64Array ring without heap allocations.
   *
   * @param {number} [timestampMs] Optional explicit timestamp (defaults to `now()`).
   * @returns {number|null} Recorded frame delta in ms, or null on the initial baseline frame.
   */
  recordFrame(timestampMs) {
    const ts =
      typeof timestampMs === 'number' && Number.isFinite(timestampMs)
        ? timestampMs
        : this._now();

    this._totalFrames += 1;
    this._commitFrameDrawCalls();

    let deltaMs = null;
    if (this._lastFrameTimestamp !== null) {
      const rawDelta = ts - this._lastFrameTimestamp;
      deltaMs = Number.isFinite(rawDelta) && rawDelta > 0 ? rawDelta : 0;
      this._pushFrameDelta(deltaMs);
    }

    this._lastFrameTimestamp = ts;
    return deltaMs;
  }

  /**
   * Records an explicit frame delta duration in milliseconds directly into the ring buffer.
   * Zero heap allocations on the hot path.
   *
   * @param {number} dtMs Non-negative finite frame duration in milliseconds.
   * @returns {number} Recorded frame delta in ms.
   */
  recordFrameTimeMs(dtMs) {
    if (typeof dtMs !== 'number' || !Number.isFinite(dtMs) || dtMs < 0) {
      throw new RangeError(`dtMs must be a non-negative finite number, got ${dtMs}`);
    }

    this._totalFrames += 1;
    this._commitFrameDrawCalls();
    this._pushFrameDelta(dtMs);

    if (this._lastFrameTimestamp !== null) {
      this._lastFrameTimestamp += dtMs;
    }
    return dtMs;
  }

  /**
   * Records a GPU worker command batch. Zero heap allocations on the hot path.
   *
   * @param {object} [batch]
   * @param {number} [batch.packets=0]
   * @param {number} [batch.drawCalls=0]
   * @param {number} [batch.computeDispatches=0]
   * @param {number} [batch.bytes=0]
   * @param {number} [batch.durationMs] Optional explicit batch time span in ms.
   * @param {number} [batch.timestampMs] Optional explicit timestamp in ms.
   */
  recordGpuBatch(batch = {}) {
    if (!batch || typeof batch !== 'object') {
      return;
    }

    const packets = sanitizeNonNegativeNumber(batch.packets, 0);
    const drawCalls = sanitizeNonNegativeNumber(batch.drawCalls, 0);
    const computeDispatches = sanitizeNonNegativeNumber(batch.computeDispatches, 0);
    const bytes = sanitizeNonNegativeNumber(batch.bytes, 0);

    const hasExplicitTimestamp =
      typeof batch.timestampMs === 'number' && Number.isFinite(batch.timestampMs);
    const ts = hasExplicitTimestamp ? batch.timestampMs : this._now();

    let dtMs = 0;
    if (typeof batch.durationMs === 'number' && Number.isFinite(batch.durationMs) && batch.durationMs > 0) {
      dtMs = batch.durationMs;
    } else if (this._gpuLastTimestamp !== null) {
      const diff = ts - this._gpuLastTimestamp;
      dtMs = diff > 0 ? diff : 0;
    } else if (!hasExplicitTimestamp) {
      const diff = ts - this._gpuEpochMs;
      dtMs = diff > 0 ? diff : 0;
    }
    this._gpuLastTimestamp = ts;

    this._totalGpuBatches += 1;
    this._totalGpuPackets += packets;
    this._totalDrawCalls += drawCalls;
    this._totalComputeDispatches += computeDispatches;
    this._totalGpuBytes += bytes;
    this._pendingFrameDrawCalls += drawCalls;

    this._gpuPacketsRing[this._gpuWriteIdx] = packets;
    this._gpuDurationsRing[this._gpuWriteIdx] = dtMs;
    this._gpuWriteIdx += 1;
    if (this._gpuWriteIdx === this._gpuWindowSize) {
      this._gpuWriteIdx = 0;
    }
    if (this._gpuCount < this._gpuWindowSize) {
      this._gpuCount += 1;
    }
  }

  /**
   * Records an HTTP Range / OPFS I/O read operation. Zero heap allocations on the hot path.
   *
   * @param {object} [read]
   * @param {number} [read.bytes=0]
   * @param {boolean} [read.cacheHit]
   * @param {boolean} [read.opfsHit=false]
   * @param {number} [read.durationMs] Optional transfer/read duration in ms.
   * @param {number} [read.timestampMs] Optional explicit timestamp in ms.
   */
  recordIoRead(read = {}) {
    if (!read || typeof read !== 'object') {
      return;
    }

    const bytes = sanitizeNonNegativeNumber(read.bytes, 0);
    const opfsHit = Boolean(read.opfsHit);
    const cacheHit = read.cacheHit !== undefined ? Boolean(read.cacheHit) || opfsHit : opfsHit;

    const hasExplicitTimestamp =
      typeof read.timestampMs === 'number' && Number.isFinite(read.timestampMs);
    const ts = hasExplicitTimestamp ? read.timestampMs : this._now();

    let dtMs = 0;
    if (typeof read.durationMs === 'number' && Number.isFinite(read.durationMs) && read.durationMs > 0) {
      dtMs = read.durationMs;
    } else if (this._ioLastTimestamp !== null) {
      const diff = ts - this._ioLastTimestamp;
      dtMs = diff > 0 ? diff : 0;
    } else if (!hasExplicitTimestamp) {
      const diff = ts - this._ioEpochMs;
      dtMs = diff > 0 ? diff : 0;
    }
    this._ioLastTimestamp = ts;

    this._totalBytesRead += bytes;
    if (cacheHit) {
      this._cacheHits += 1;
    } else {
      this._cacheMisses += 1;
    }
    if (opfsHit) {
      this._opfsCacheHits += 1;
    }

    this._ioBytesRing[this._ioWriteIdx] = bytes;
    this._ioDurationsRing[this._ioWriteIdx] = dtMs;
    this._ioWriteIdx += 1;
    if (this._ioWriteIdx === this._ioWindowSize) {
      this._ioWriteIdx = 0;
    }
    if (this._ioCount < this._ioWindowSize) {
      this._ioCount += 1;
    }
  }

  /**
   * Updates the latest AudioWorklet / SAB ring buffer telemetry values.
   * Accepts partial updates or full `AudioEngine.getTelemetry()` snapshots.
   * Zero heap allocations on the hot path.
   *
   * @param {object} [telemetry]
   * @param {number} [telemetry.bufferedFrames]
   * @param {number} [telemetry.underruns]
   * @param {number} [telemetry.underrunFrames]
   * @param {number} [telemetry.activeVoices]
   */
  updateAudioTelemetry(telemetry = {}) {
    if (!telemetry || typeof telemetry !== 'object') {
      return;
    }
    if (telemetry.bufferedFrames !== undefined) {
      this._audioBufferedFrames = sanitizeNonNegativeNumber(
        telemetry.bufferedFrames,
        this._audioBufferedFrames
      );
    }
    if (telemetry.underruns !== undefined) {
      this._audioUnderruns = sanitizeNonNegativeNumber(
        telemetry.underruns,
        this._audioUnderruns
      );
    }
    if (telemetry.underrunFrames !== undefined) {
      this._audioUnderrunFrames = sanitizeNonNegativeNumber(
        telemetry.underrunFrames,
        this._audioUnderrunFrames
      );
    }
    if (telemetry.activeVoices !== undefined) {
      this._audioActiveVoices = sanitizeNonNegativeNumber(
        telemetry.activeVoices,
        this._audioActiveVoices
      );
    }
  }

  /**
   * Computes a structured telemetry snapshot across frames, GPU, I/O, and audio.
   */
  getSnapshot() {
    const count = this._frameCount;
    let avgFrameTimeMs = 0;
    let minFrameTimeMs = 0;
    let p50FrameTimeMs = 0;
    let p95FrameTimeMs = 0;
    let p99FrameTimeMs = 0;
    let maxFrameTimeMs = 0;
    let windowJankFrames = 0;

    if (count > 0) {
      let sum = 0;
      for (let i = 0; i < count; i += 1) {
        const dt = sanitizeNonNegativeNumber(this._frameDeltas[i], 0);
        this._scratchSorted[i] = dt;
        sum += dt;
        if (dt > this._jankThresholdMs) {
          windowJankFrames += 1;
        }
      }

      const sorted = this._scratchSorted.subarray(0, count);
      sorted.sort();

      avgFrameTimeMs = sanitizeNonNegativeNumber(sum / count, 0);
      minFrameTimeMs = sanitizeNonNegativeNumber(sorted[0], 0);
      p50FrameTimeMs = percentileFromSorted(sorted, count, 0.5);
      p95FrameTimeMs = percentileFromSorted(sorted, count, 0.95);
      p99FrameTimeMs = percentileFromSorted(sorted, count, 0.99);
      maxFrameTimeMs = sanitizeNonNegativeNumber(sorted[count - 1], 0);
    }

    const fps = avgFrameTimeMs > 0 ? sanitizeNonNegativeNumber(1000 / avgFrameTimeMs, 0) : 0;
    const onePercentLowFps =
      p99FrameTimeMs > 0 ? sanitizeNonNegativeNumber(1000 / p99FrameTimeMs, 0) : 0;

    // Rolling GPU packet throughput & draw calls per frame.
    let gpuPacketsSum = 0;
    let gpuDurationSumMs = 0;
    for (let i = 0; i < this._gpuCount; i += 1) {
      gpuPacketsSum += this._gpuPacketsRing[i];
      gpuDurationSumMs += this._gpuDurationsRing[i];
    }

    let gpuPacketsPerSec = 0;
    if (gpuDurationSumMs > 0) {
      gpuPacketsPerSec = (gpuPacketsSum * 1000) / gpuDurationSumMs;
    } else if (this._totalGpuPackets > 0 && fps > 0 && this._totalFrames > 0) {
      gpuPacketsPerSec = (this._totalGpuPackets / this._totalFrames) * fps;
    }

    let drawCallsPerFrame = 0;
    if (this._frameDrawCount > 0) {
      if (this._pendingFrameDrawCalls > 0) {
        if (this._frameDrawCount === this._windowSize) {
          const oldest = this._frameDrawCalls[this._frameDrawWriteIdx];
          drawCallsPerFrame =
            (this._frameDrawSum - oldest + this._pendingFrameDrawCalls) / this._frameDrawCount;
        } else {
          drawCallsPerFrame =
            (this._frameDrawSum + this._pendingFrameDrawCalls) / this._frameDrawCount;
        }
      } else {
        drawCallsPerFrame = this._frameDrawSum / this._frameDrawCount;
      }
    }

    // Rolling I/O throughput (MB/s) & cumulative cache hit ratio.
    let ioBytesSum = 0;
    let ioDurationSumMs = 0;
    for (let i = 0; i < this._ioCount; i += 1) {
      ioBytesSum += this._ioBytesRing[i];
      ioDurationSumMs += this._ioDurationsRing[i];
    }

    const ioThroughputMBps =
      ioDurationSumMs > 0 && ioBytesSum > 0
        ? (ioBytesSum * 1000) / (ioDurationSumMs * BYTES_PER_MB)
        : 0;

    const totalReads = this._cacheHits + this._cacheMisses;
    const cacheHitRate = totalReads > 0 ? this._cacheHits / totalReads : 0;

    return {
      timestampMs: this._now(),
      windowSize: this._windowSize,
      jankThresholdMs: this._jankThresholdMs,
      frames: {
        fps,
        onePercentLowFps,
        fps1PctLow: onePercentLowFps,
        avgFrameTimeMs,
        minFrameTimeMs,
        p50FrameTimeMs,
        p95FrameTimeMs,
        p99FrameTimeMs,
        p50FrameMs: p50FrameTimeMs,
        p95FrameMs: p95FrameTimeMs,
        p99FrameMs: p99FrameTimeMs,
        p50Ms: p50FrameTimeMs,
        p95Ms: p95FrameTimeMs,
        p99Ms: p99FrameTimeMs,
        maxFrameTimeMs,
        totalFrames: this._totalFrames,
        jankFrames: this._jankFrames,
        jankFrameCount: this._jankFrames,
        windowJankFrames,
        sampleCount: count,
      },
      gpu: {
        gpuPacketsPerSec,
        drawCallsPerFrame,
        totalGpuPackets: this._totalGpuPackets,
        totalDrawCalls: this._totalDrawCalls,
        totalComputeDispatches: this._totalComputeDispatches,
        totalGpuBytes: this._totalGpuBytes,
        totalBatches: this._totalGpuBatches,
      },
      io: {
        cacheHitRate,
        ioThroughputMBps,
        totalBytesRead: this._totalBytesRead,
        cacheHits: this._cacheHits,
        opfsCacheHits: this._opfsCacheHits,
        cacheMisses: this._cacheMisses,
        totalReads,
      },
      audio: {
        bufferedFrames: this._audioBufferedFrames,
        underruns: this._audioUnderruns,
        underrunFrames: this._audioUnderrunFrames,
        activeVoices: this._audioActiveVoices,
      },
    };
  }

  /**
   * Returns a flat frame-pacing summary including rolling P50/P95/P99 frame times,
   * average and 1% low FPS, and jank counters.
   */
  getSummary() {
    const snap = this.getSnapshot();
    return {
      timestampMs: snap.timestampMs,
      windowSize: snap.windowSize,
      jankThresholdMs: snap.jankThresholdMs,
      sampleCount: snap.frames.sampleCount,
      totalFrames: snap.frames.totalFrames,
      fps: snap.frames.fps,
      onePercentLowFps: snap.frames.onePercentLowFps,
      fps1PctLow: snap.frames.fps1PctLow,
      avgFrameTimeMs: snap.frames.avgFrameTimeMs,
      minFrameTimeMs: snap.frames.minFrameTimeMs,
      p50FrameTimeMs: snap.frames.p50FrameTimeMs,
      p95FrameTimeMs: snap.frames.p95FrameTimeMs,
      p99FrameTimeMs: snap.frames.p99FrameTimeMs,
      p50FrameMs: snap.frames.p50FrameMs,
      p95FrameMs: snap.frames.p95FrameMs,
      p99FrameMs: snap.frames.p99FrameMs,
      p50Ms: snap.frames.p50Ms,
      p95Ms: snap.frames.p95Ms,
      p99Ms: snap.frames.p99Ms,
      maxFrameTimeMs: snap.frames.maxFrameTimeMs,
      jankFrames: snap.frames.jankFrames,
      jankFrameCount: snap.frames.jankFrameCount,
      windowJankFrames: snap.frames.windowJankFrames,
    };
  }

  /**
   * Returns a flat metrics payload matching the `window.playgta6Hud.updateMetrics(...)`
   * contract in `web/index.html` without clobbering unrelated HUD fields
   * (`pointerLock`, `gamepad`, `status`).
   */
  toHudMetrics() {
    const snap = this.getSnapshot();
    return {
      fps: snap.frames.fps,
      onePercentLowFps: snap.frames.onePercentLowFps,
      fps1PctLow: snap.frames.fps1PctLow,
      gpuPackets: snap.gpu.totalGpuPackets,
      opfsCacheHits: snap.io.opfsCacheHits,
      cacheHits: snap.io.cacheHits,
      avgFrameTimeMs: snap.frames.avgFrameTimeMs,
      minFrameTimeMs: snap.frames.minFrameTimeMs,
      p50FrameTimeMs: snap.frames.p50FrameTimeMs,
      p95FrameTimeMs: snap.frames.p95FrameTimeMs,
      p99FrameTimeMs: snap.frames.p99FrameTimeMs,
      p50FrameMs: snap.frames.p50FrameMs,
      p95FrameMs: snap.frames.p95FrameMs,
      p99FrameMs: snap.frames.p99FrameMs,
      p50Ms: snap.frames.p50Ms,
      p95Ms: snap.frames.p95Ms,
      p99Ms: snap.frames.p99Ms,
      maxFrameTimeMs: snap.frames.maxFrameTimeMs,
      totalFrames: snap.frames.totalFrames,
      jankFrames: snap.frames.jankFrames,
      jankFrameCount: snap.frames.jankFrameCount,
      windowJankFrames: snap.frames.windowJankFrames,
      gpuPacketsPerSec: snap.gpu.gpuPacketsPerSec,
      drawCallsPerFrame: snap.gpu.drawCallsPerFrame,
      cacheHitRate: snap.io.cacheHitRate,
      ioThroughputMBps: snap.io.ioThroughputMBps,
      totalBytesRead: snap.io.totalBytesRead,
      cacheMisses: snap.io.cacheMisses,
      audioBufferedFrames: snap.audio.bufferedFrames,
      audioUnderruns: snap.audio.underruns,
      audioUnderrunFrames: snap.audio.underrunFrames,
      audioActiveVoices: snap.audio.activeVoices,
    };
  }

  /**
   * Immediately pushes the current `toHudMetrics()` payload to the attached HUD, if any.
   * @returns {object|null} The pushed metrics payload, or null if no HUD is attached.
   */
  pushHudMetrics() {
    if (!this._hud || typeof this._hud.updateMetrics !== 'function') {
      return null;
    }
    const metrics = this.toHudMetrics();
    this._hud.updateMetrics(metrics);
    return metrics;
  }

  /**
   * Starts periodically pushing `toHudMetrics()` to `hud.updateMetrics(...)`.
   * Replaces any previously attached HUD timer to avoid leaks.
   *
   * @param {{ updateMetrics: (metrics: object) => void }} hud
   * @param {number} [intervalMs=250]
   * @returns {() => void} Detach function.
   */
  attachHud(hud, intervalMs = DEFAULT_HUD_INTERVAL_MS) {
    if (!hud || typeof hud !== 'object' || typeof hud.updateMetrics !== 'function') {
      throw new TypeError('attachHud requires a HUD object with an updateMetrics(metrics) function');
    }
    if (typeof intervalMs !== 'number' || !Number.isFinite(intervalMs) || intervalMs <= 0) {
      throw new RangeError(`intervalMs must be a positive finite number, got ${intervalMs}`);
    }

    this.detachHud();

    this._hud = hud;
    this._hudTimerId = this._setInterval(() => {
      this.pushHudMetrics();
    }, intervalMs);

    return () => {
      this.detachHud();
    };
  }

  /**
   * Stops periodic HUD metric updates and releases the HUD reference. Idempotent.
   */
  detachHud() {
    if (this._hudTimerId !== null) {
      const id = this._hudTimerId;
      this._hudTimerId = null;
      this._clearInterval(id);
    }
    this._hud = null;
  }
}

export default PerfMonitor;
