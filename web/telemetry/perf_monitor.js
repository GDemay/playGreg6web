// Rolling frame telemetry ring and HUD performance profiler for playgta6web.
// Zero external dependencies; clean ES module compatible with <script type="module">
// and Node.js ESM imports. Designed for zero heap allocations on hot record*() paths.

export const DEFAULT_WINDOW_SIZE = 120;
export const DEFAULT_JANK_THRESHOLD_MS = 33.34;
export const DEFAULT_TARGET_FPS = 30;
export const DEFAULT_FRAME_BUDGET_MS = 33.333;
export const DEFAULT_HUD_INTERVAL_MS = 250;
export const BYTES_PER_MB = 1024 * 1024;
export const PERF_PHASES = Object.freeze({
  BOOT: 'boot',
  LOADING_ASSETS: 'loading_assets',
  GAMEPLAY: 'gameplay',
  MENU: 'menu',
});
export const FRAME_BOTTLENECKS = Object.freeze({
  WITHIN_BUDGET: 'within_budget',
  IO_BOUND: 'io_bound',
  SHADER_COMPILE_BOUND: 'shader_compile_bound',
  GPU_BOUND: 'gpu_bound',
  CPU_BOUND: 'cpu_bound',
});

const PERF_PHASE_VALUES = Object.freeze(Object.values(PERF_PHASES));
const PERF_PHASE_SET = new Set(PERF_PHASE_VALUES);

function sanitizePhase(phase, fallback = PERF_PHASES.BOOT) {
  if (typeof phase !== 'string') {
    return fallback;
  }
  if (PERF_PHASE_SET.has(phase)) {
    return phase;
  }
  const normalized = phase.trim().toLowerCase().replace(/[\s-]+/g, '_');
  return PERF_PHASE_SET.has(normalized) ? normalized : fallback;
}

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

function classifyFrameBottleneckComponents(
  frameMs,
  ioWaitMs,
  shaderCompileMs,
  gpuMs,
  cpuMs,
  budgetMs
) {
  if (frameMs <= budgetMs) {
    return FRAME_BOTTLENECKS.WITHIN_BUDGET;
  }

  // Deterministic tie-breaking priority:
  // io_bound > shader_compile_bound > gpu_bound > cpu_bound.
  // Over-budget frames with no positive stage breakdown fall back to cpu_bound.
  let winner = FRAME_BOTTLENECKS.CPU_BOUND;
  let bestMs = cpuMs;

  if (gpuMs > 0 && gpuMs >= bestMs) {
    winner = FRAME_BOTTLENECKS.GPU_BOUND;
    bestMs = gpuMs;
  }
  if (shaderCompileMs > 0 && shaderCompileMs >= bestMs) {
    winner = FRAME_BOTTLENECKS.SHADER_COMPILE_BOUND;
    bestMs = shaderCompileMs;
  }
  if (ioWaitMs > 0 && ioWaitMs >= bestMs) {
    winner = FRAME_BOTTLENECKS.IO_BOUND;
    bestMs = ioWaitMs;
  }

  return winner;
}

/**
 * Deterministically classifies why a frame exceeded the target frame budget (default 33.333 ms / 30 FPS).
 * Sanitizes NaN, Infinity, negative, or missing fields safely to 0.
 *
 * @param {object} [sample]
 * @param {number} [sample.frameMs] Total wall-clock frame duration in milliseconds.
 * @param {number} [sample.ioWaitMs] Time spent waiting on HTTP Range / OPFS / VFS I/O in ms.
 * @param {number} [sample.shaderCompileMs] Time spent compiling WGSL shaders or WebGPU pipelines in ms.
 * @param {number} [sample.gpuMs] GPU execution / submission time in ms.
 * @param {number} [sample.cpuMs] CPU / Wasm simulation & command encoding time in ms.
 * @param {number} [budgetMs=33.333] Target frame budget in milliseconds (defaults to 33.333 ms for 30 FPS).
 * @returns {'within_budget'|'io_bound'|'shader_compile_bound'|'gpu_bound'|'cpu_bound'}
 */
export function classifyFrameBottleneck(sample, budgetMs = DEFAULT_FRAME_BUDGET_MS) {
  const effectiveBudgetMs =
    typeof budgetMs === 'number' && Number.isFinite(budgetMs) && budgetMs > 0
      ? budgetMs
      : DEFAULT_FRAME_BUDGET_MS;

  if (!sample || typeof sample !== 'object') {
    return FRAME_BOTTLENECKS.WITHIN_BUDGET;
  }

  const ioWaitMs = sanitizeNonNegativeNumber(sample.ioWaitMs, 0);
  const shaderCompileMs = sanitizeNonNegativeNumber(sample.shaderCompileMs, 0);
  const gpuMs = sanitizeNonNegativeNumber(sample.gpuMs, 0);
  const cpuMs = sanitizeNonNegativeNumber(sample.cpuMs, 0);
  const frameMs =
    sample.frameMs !== undefined
      ? sanitizeNonNegativeNumber(sample.frameMs, 0)
      : ioWaitMs + shaderCompileMs + gpuMs + cpuMs;

  return classifyFrameBottleneckComponents(
    frameMs,
    ioWaitMs,
    shaderCompileMs,
    gpuMs,
    cpuMs,
    effectiveBudgetMs
  );
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
   * @param {number} [options.targetFps=30] Target frame rate for budget compliance tracking.
   * @param {number} [options.budgetMs=33.333] Target per-frame budget in milliseconds.
   * @param {number} [options.gpuWindowSize] Fixed ring capacity for GPU batch rate aggregation (defaults to windowSize).
   * @param {number} [options.ioWindowSize] Fixed ring capacity for I/O throughput aggregation (defaults to windowSize).
   * @param {string} [options.initialPhase='boot'] Initial telemetry phase (`PERF_PHASES`).
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
      targetFps,
      budgetMs,
      gpuWindowSize = windowSize,
      ioWindowSize = windowSize,
      initialPhase = PERF_PHASES.BOOT,
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
    let resolvedTargetFps = DEFAULT_TARGET_FPS;
    let resolvedBudgetMs = DEFAULT_FRAME_BUDGET_MS;
    if (targetFps !== undefined) {
      if (typeof targetFps !== 'number' || !Number.isFinite(targetFps) || targetFps <= 0) {
        throw new RangeError(`targetFps must be a positive finite number, got ${targetFps}`);
      }
      resolvedTargetFps = targetFps;
      resolvedBudgetMs = Math.round((1000 / resolvedTargetFps) * 1000) / 1000;
    }
    if (budgetMs !== undefined) {
      if (typeof budgetMs !== 'number' || !Number.isFinite(budgetMs) || budgetMs <= 0) {
        throw new RangeError(`budgetMs must be a positive finite number, got ${budgetMs}`);
      }
      resolvedBudgetMs = budgetMs;
      if (targetFps === undefined) {
        resolvedTargetFps =
          resolvedBudgetMs === DEFAULT_FRAME_BUDGET_MS
            ? DEFAULT_TARGET_FPS
            : 1000 / resolvedBudgetMs;
      }
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
    this._targetFps = resolvedTargetFps;
    this._budgetMs = resolvedBudgetMs;
    this._gpuWindowSize = gpuWindowSize;
    this._ioWindowSize = ioWindowSize;
    this._initialPhase = sanitizePhase(initialPhase, PERF_PHASES.BOOT);
    this._now = now;
    this._setInterval = setIntervalImpl;
    this._clearInterval = clearIntervalImpl;

    // Frame timing ring buffers (pre-allocated, zero allocations on hot path).
    this._frameDeltas = new Float64Array(windowSize);
    this._scratchSorted = new Float64Array(windowSize);
    this._frameDrawCalls = new Float64Array(windowSize);

    // Pre-allocated bottleneck counter map (mutated in place on hot path).
    this._bottleneckCounts = {
      within_budget: 0,
      io_bound: 0,
      shader_compile_bound: 0,
      gpu_bound: 0,
      cpu_bound: 0,
    };

    // Per-phase frame timing ring buffers (pre-allocated, zero allocations on hot path).
    this._phaseBuckets = Object.create(null);
    for (let i = 0; i < PERF_PHASE_VALUES.length; i += 1) {
      const phase = PERF_PHASE_VALUES[i];
      this._phaseBuckets[phase] = {
        deltas: new Float64Array(windowSize),
        writeIdx: 0,
        sampleCount: 0,
        frameCount: 0,
      };
    }

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

  get targetFps() {
    return this._targetFps;
  }

  get budgetMs() {
    return this._budgetMs;
  }

  get isHudAttached() {
    return this._hudTimerId !== null;
  }

  get currentPhase() {
    return this._currentPhase;
  }

  /**
   * Sets the active telemetry phase for phase-segmented frame-time histograms.
   * Invalid or unrecognized values are safely sanitized to `'boot'`.
   *
   * @param {string} [phase] One of `PERF_PHASES` (`'boot'`, `'loading_assets'`, `'gameplay'`, `'menu'`).
   * @returns {string} The sanitized active phase.
   */
  setPhase(phase) {
    const sanitized = sanitizePhase(phase, PERF_PHASES.BOOT);
    this._currentPhase = sanitized;
    return sanitized;
  }

  /**
   * Resets all frame, phase, GPU, I/O, and audio telemetry counters and rolling rings.
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

    this._framesWithin30FpsBudget = 0;
    this._framesOver30FpsBudget = 0;
    this._bottleneckCounts.within_budget = 0;
    this._bottleneckCounts.io_bound = 0;
    this._bottleneckCounts.shader_compile_bound = 0;
    this._bottleneckCounts.gpu_bound = 0;
    this._bottleneckCounts.cpu_bound = 0;
    this._lastBottleneck = FRAME_BOTTLENECKS.WITHIN_BUDGET;

    this._currentPhase = this._initialPhase;
    for (let i = 0; i < PERF_PHASE_VALUES.length; i += 1) {
      const bucket = this._phaseBuckets[PERF_PHASE_VALUES[i]];
      bucket.deltas.fill(0);
      bucket.writeIdx = 0;
      bucket.sampleCount = 0;
      bucket.frameCount = 0;
    }

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
   * Internal zero-allocation helper to commit a frame delta to global, active-phase,
   * and 30 FPS budget / bottleneck counters.
   * @param {number} dtMs
   * @param {object|null} [breakdown]
   * @returns {'within_budget'|'io_bound'|'shader_compile_bound'|'gpu_bound'|'cpu_bound'}
   */
  _pushFrameDelta(dtMs, breakdown = null) {
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

    const ioWaitMs =
      breakdown && typeof breakdown === 'object'
        ? sanitizeNonNegativeNumber(breakdown.ioWaitMs, 0)
        : 0;
    const shaderCompileMs =
      breakdown && typeof breakdown === 'object'
        ? sanitizeNonNegativeNumber(breakdown.shaderCompileMs, 0)
        : 0;
    const gpuMs =
      breakdown && typeof breakdown === 'object'
        ? sanitizeNonNegativeNumber(breakdown.gpuMs, 0)
        : 0;
    const cpuMs =
      breakdown && typeof breakdown === 'object'
        ? sanitizeNonNegativeNumber(breakdown.cpuMs, 0)
        : 0;

    const bottleneck = classifyFrameBottleneckComponents(
      safeDtMs,
      ioWaitMs,
      shaderCompileMs,
      gpuMs,
      cpuMs,
      this._budgetMs
    );

    if (bottleneck === FRAME_BOTTLENECKS.WITHIN_BUDGET) {
      this._framesWithin30FpsBudget += 1;
    } else {
      this._framesOver30FpsBudget += 1;
    }
    this._bottleneckCounts[bottleneck] += 1;
    this._lastBottleneck = bottleneck;

    const phase = sanitizePhase(this._currentPhase, PERF_PHASES.BOOT);
    const bucket = this._phaseBuckets[phase];
    if (bucket) {
      bucket.deltas[bucket.writeIdx] = safeDtMs;
      bucket.writeIdx += 1;
      if (bucket.writeIdx === this._windowSize) {
        bucket.writeIdx = 0;
      }
      if (bucket.sampleCount < this._windowSize) {
        bucket.sampleCount += 1;
      }
      bucket.frameCount += 1;
    }

    return bottleneck;
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
   * @param {object} [breakdown] Optional stage breakdown `{ ioWaitMs, shaderCompileMs, gpuMs, cpuMs }` for bottleneck classification.
   * @returns {number|null} Recorded frame delta in ms, or null on the initial baseline frame.
   */
  recordFrame(timestampMs, breakdown = null) {
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
      this._pushFrameDelta(deltaMs, breakdown);
    }

    this._lastFrameTimestamp = ts;
    return deltaMs;
  }

  /**
   * Records an explicit frame delta duration in milliseconds directly into the ring buffer.
   * Zero heap allocations on the hot path.
   *
   * @param {number} dtMs Non-negative finite frame duration in milliseconds.
   * @param {object} [breakdown] Optional stage breakdown `{ ioWaitMs, shaderCompileMs, gpuMs, cpuMs }` for bottleneck classification.
   * @returns {number} Recorded frame delta in ms.
   */
  recordFrameTimeMs(dtMs, breakdown = null) {
    if (typeof dtMs !== 'number' || !Number.isFinite(dtMs) || dtMs < 0) {
      throw new RangeError(`dtMs must be a non-negative finite number, got ${dtMs}`);
    }

    this._totalFrames += 1;
    this._commitFrameDrawCalls();
    this._pushFrameDelta(dtMs, breakdown);

    if (this._lastFrameTimestamp !== null) {
      this._lastFrameTimestamp += dtMs;
    }
    return dtMs;
  }

  /**
   * Records a frame along with per-stage timing breakdown (`ioWaitMs`, `shaderCompileMs`,
   * `gpuMs`, `cpuMs`) for 30 FPS budget compliance tracking and automatic bottleneck classification.
   * Zero heap allocations on the hot path; NaN/negative values are safely sanitized to 0.
   *
   * @param {object} sample
   * @param {number} [sample.frameMs] Total frame duration in ms.
   * @param {number} [sample.timestampMs] Optional monotonic frame timestamp when `frameMs` is omitted.
   * @param {number} [sample.ioWaitMs] Time spent waiting on I/O in ms.
   * @param {number} [sample.shaderCompileMs] Time spent compiling shaders/pipelines in ms.
   * @param {number} [sample.gpuMs] GPU time in ms.
   * @param {number} [sample.cpuMs] CPU/Wasm time in ms.
   * @returns {'within_budget'|'io_bound'|'shader_compile_bound'|'gpu_bound'|'cpu_bound'|null}
   */
  recordDetailedFrame(sample) {
    if (!sample || typeof sample !== 'object') {
      return null;
    }

    if (
      sample.frameMs === undefined &&
      typeof sample.timestampMs === 'number' &&
      Number.isFinite(sample.timestampMs)
    ) {
      const deltaMs = this.recordFrame(sample.timestampMs, sample);
      return deltaMs === null ? null : this._lastBottleneck;
    }

    const ioWaitMs = sanitizeNonNegativeNumber(sample.ioWaitMs, 0);
    const shaderCompileMs = sanitizeNonNegativeNumber(sample.shaderCompileMs, 0);
    const gpuMs = sanitizeNonNegativeNumber(sample.gpuMs, 0);
    const cpuMs = sanitizeNonNegativeNumber(sample.cpuMs, 0);
    const frameMs =
      sample.frameMs !== undefined
        ? sanitizeNonNegativeNumber(sample.frameMs, 0)
        : ioWaitMs + shaderCompileMs + gpuMs + cpuMs;

    this._totalFrames += 1;
    this._commitFrameDrawCalls();
    const bottleneck = this._pushFrameDelta(frameMs, sample);

    if (this._lastFrameTimestamp !== null) {
      this._lastFrameTimestamp += frameMs;
    }
    return bottleneck;
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
   * Computes sanitized rolling histogram stats for a single phase bucket using
   * the pre-allocated `_scratchSorted` buffer.
   *
   * @param {{ deltas: Float64Array, sampleCount: number, frameCount: number }} bucket
   */
  _computePhaseStats(bucket) {
    const count = Math.min(bucket.sampleCount, bucket.deltas.length);
    const frameCount = sanitizeNonNegativeNumber(bucket.frameCount, 0);

    if (count <= 0) {
      return {
        frameCount,
        avgFps: 0,
        onePercentLowFps: 0,
        avgFrameMs: 0,
        p95FrameMs: 0,
        p99FrameMs: 0,
        maxFrameMs: 0,
      };
    }

    let sum = 0;
    for (let i = 0; i < count; i += 1) {
      const dt = sanitizeNonNegativeNumber(bucket.deltas[i], 0);
      this._scratchSorted[i] = dt;
      sum += dt;
    }

    const sorted = this._scratchSorted.subarray(0, count);
    sorted.sort();

    const avgFrameMs = sanitizeNonNegativeNumber(sum / count, 0);
    const p95FrameMs = percentileFromSorted(sorted, count, 0.95);
    const p99FrameMs = percentileFromSorted(sorted, count, 0.99);
    const maxFrameMs = sanitizeNonNegativeNumber(sorted[count - 1], 0);
    const avgFps = avgFrameMs > 0 ? sanitizeNonNegativeNumber(1000 / avgFrameMs, 0) : 0;
    const onePercentLowFps =
      p99FrameMs > 0 ? sanitizeNonNegativeNumber(1000 / p99FrameMs, 0) : 0;

    return {
      frameCount,
      avgFps,
      onePercentLowFps,
      avgFrameMs,
      p95FrameMs,
      p99FrameMs,
      maxFrameMs,
    };
  }

  /**
   * Returns a fresh sanitized copy of cumulative bottleneck counts.
   */
  _copyBottleneckCounts() {
    return {
      within_budget: sanitizeNonNegativeNumber(this._bottleneckCounts.within_budget, 0),
      io_bound: sanitizeNonNegativeNumber(this._bottleneckCounts.io_bound, 0),
      shader_compile_bound: sanitizeNonNegativeNumber(this._bottleneckCounts.shader_compile_bound, 0),
      gpu_bound: sanitizeNonNegativeNumber(this._bottleneckCounts.gpu_bound, 0),
      cpu_bound: sanitizeNonNegativeNumber(this._bottleneckCounts.cpu_bound, 0),
    };
  }

  /**
   * Computes the dominant over-budget bottleneck, or `'within_budget'` when no
   * over-budget frames have been observed.
   *
   * @param {{ io_bound: number, shader_compile_bound: number, gpu_bound: number, cpu_bound: number }} counts
   * @returns {'within_budget'|'io_bound'|'shader_compile_bound'|'gpu_bound'|'cpu_bound'}
   */
  _computeDominantBottleneck(counts) {
    const io = counts.io_bound;
    const shader = counts.shader_compile_bound;
    const gpu = counts.gpu_bound;
    const cpu = counts.cpu_bound;

    if (io === 0 && shader === 0 && gpu === 0 && cpu === 0) {
      return FRAME_BOTTLENECKS.WITHIN_BUDGET;
    }

    // Deterministic tie-breaking matching classifyFrameBottleneck priority:
    // io_bound > shader_compile_bound > gpu_bound > cpu_bound.
    let winner = FRAME_BOTTLENECKS.CPU_BOUND;
    let best = cpu;

    if (gpu > 0 && gpu >= best) {
      winner = FRAME_BOTTLENECKS.GPU_BOUND;
      best = gpu;
    }
    if (shader > 0 && shader >= best) {
      winner = FRAME_BOTTLENECKS.SHADER_COMPILE_BOUND;
      best = shader;
    }
    if (io > 0 && io >= best) {
      winner = FRAME_BOTTLENECKS.IO_BOUND;
      best = io;
    }

    return winner;
  }

  /**
   * Computes a structured telemetry snapshot across frames, 30 FPS budget/bottlenecks,
   * phases, GPU, I/O, and audio.
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

    const currentPhase = sanitizePhase(this._currentPhase, PERF_PHASES.BOOT);
    const phases = {};
    for (let i = 0; i < PERF_PHASE_VALUES.length; i += 1) {
      const phase = PERF_PHASE_VALUES[i];
      phases[phase] = this._computePhaseStats(this._phaseBuckets[phase]);
    }

    const bottleneckCounts = this._copyBottleneckCounts();
    const framesWithin30FpsBudget = sanitizeNonNegativeNumber(this._framesWithin30FpsBudget, 0);
    const framesOver30FpsBudget = sanitizeNonNegativeNumber(this._framesOver30FpsBudget, 0);
    const totalBudgetFrames = framesWithin30FpsBudget + framesOver30FpsBudget;
    const rawComplianceRate =
      totalBudgetFrames > 0 ? framesWithin30FpsBudget / totalBudgetFrames : 0;
    const fps30ComplianceRate = Math.min(
      1,
      Math.max(0, sanitizeNonNegativeNumber(rawComplianceRate, 0))
    );
    const dominantBottleneck = this._computeDominantBottleneck(bottleneckCounts);

    return {
      timestampMs: this._now(),
      windowSize: this._windowSize,
      jankThresholdMs: this._jankThresholdMs,
      targetFps: this._targetFps,
      budgetMs: this._budgetMs,
      framesWithin30FpsBudget,
      framesOver30FpsBudget,
      fps30ComplianceRate,
      dominantBottleneck,
      bottleneckCounts,
      currentPhase,
      phases,
      budget: {
        targetFps: this._targetFps,
        budgetMs: this._budgetMs,
        framesWithin30FpsBudget,
        framesOver30FpsBudget,
        fps30ComplianceRate,
        dominantBottleneck,
        lastBottleneck: this._lastBottleneck,
        bottleneckCounts: { ...bottleneckCounts },
      },
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
        targetFps: this._targetFps,
        budgetMs: this._budgetMs,
        framesWithin30FpsBudget,
        framesOver30FpsBudget,
        fps30ComplianceRate,
        dominantBottleneck,
        bottleneckCounts: { ...bottleneckCounts },
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
   * Alias for `getSnapshot()`, returning the complete sanitized telemetry & 30 FPS budget stats.
   */
  getStats() {
    return this.getSnapshot();
  }

  /**
   * Returns a flat frame-pacing summary including rolling P50/P95/P99 frame times,
   * average and 1% low FPS, jank counters, and 30 FPS budget / bottleneck metrics.
   */
  getSummary() {
    const snap = this.getSnapshot();
    return {
      timestampMs: snap.timestampMs,
      windowSize: snap.windowSize,
      jankThresholdMs: snap.jankThresholdMs,
      targetFps: snap.targetFps,
      budgetMs: snap.budgetMs,
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
      framesWithin30FpsBudget: snap.framesWithin30FpsBudget,
      framesOver30FpsBudget: snap.framesOver30FpsBudget,
      fps30ComplianceRate: snap.fps30ComplianceRate,
      dominantBottleneck: snap.dominantBottleneck,
      bottleneckCounts: { ...snap.bottleneckCounts },
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
      targetFps: snap.targetFps,
      budgetMs: snap.budgetMs,
      framesWithin30FpsBudget: snap.framesWithin30FpsBudget,
      framesOver30FpsBudget: snap.framesOver30FpsBudget,
      fps30ComplianceRate: snap.fps30ComplianceRate,
      dominantBottleneck: snap.dominantBottleneck,
      bottleneckCounts: { ...snap.bottleneckCounts },
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
