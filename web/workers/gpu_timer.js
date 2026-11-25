'use strict';

(function (root) {
  const DEFAULT_MAX_PASSES_PER_FRAME = 32;
  const DEFAULT_READBACK_SLOTS = 3;
  const DEFAULT_HISTORY_SIZE = 120;
  const DEFAULT_FRAME_BUDGET_MS = 16.67;
  const TIMESTAMP_BYTE_SIZE = 8;
  const NS_PER_MS = 1e6;

  // WebGPU GPUBufferUsage & GPUMapMode constants with headless/Node fallbacks.
  const BUFFER_USAGE = Object.freeze({
    MAP_READ:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.MAP_READ === 'number'
        ? GPUBufferUsage.MAP_READ
        : 0x0001,
    COPY_SRC:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.COPY_SRC === 'number'
        ? GPUBufferUsage.COPY_SRC
        : 0x0004,
    COPY_DST:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.COPY_DST === 'number'
        ? GPUBufferUsage.COPY_DST
        : 0x0008,
    QUERY_RESOLVE:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.QUERY_RESOLVE === 'number'
        ? GPUBufferUsage.QUERY_RESOLVE
        : 0x0200,
  });

  const MAP_MODE = Object.freeze({
    READ:
      typeof GPUMapMode !== 'undefined' && typeof GPUMapMode.READ === 'number'
        ? GPUMapMode.READ
        : 0x0001,
  });

  function normalizePositiveInt(value, fallback, name) {
    if (value === undefined) {
      return fallback;
    }
    if (!Number.isInteger(value) || value < 1) {
      throw new RangeError(`${name} must be a positive integer, got ${value}`);
    }
    return value;
  }

  function normalizePositiveNumber(value, fallback, name) {
    if (value === undefined) {
      return fallback;
    }
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
      throw new RangeError(`${name} must be a positive finite number, got ${value}`);
    }
    return value;
  }

  function percentileFromSorted(sorted, count, fraction) {
    if (count <= 0) {
      return 0;
    }
    const idx = Math.min(count - 1, Math.max(0, Math.ceil(fraction * count) - 1));
    return sorted[idx];
  }

  /**
   * Pure helper computing deterministic nearest-rank percentile statistics over
   * numeric duration samples (ms) without mutating the input collection.
   *
   * @param {ArrayLike<number> | Iterable<number> | null | undefined} samples
   * @returns {{ count: number, avgMs: number, minMs: number, p50Ms: number, p95Ms: number, p99Ms: number, maxMs: number }}
   */
  function computePercentiles(samples) {
    if (!samples || typeof samples !== 'object') {
      return {
        count: 0,
        avgMs: 0,
        minMs: 0,
        p50Ms: 0,
        p95Ms: 0,
        p99Ms: 0,
        maxMs: 0,
      };
    }

    const values = [];
    if (typeof samples.length === 'number') {
      for (let i = 0; i < samples.length; i += 1) {
        const v = samples[i];
        if (typeof v === 'number' && Number.isFinite(v)) {
          values.push(v);
        }
      }
    } else if (typeof samples[Symbol.iterator] === 'function') {
      for (const v of samples) {
        if (typeof v === 'number' && Number.isFinite(v)) {
          values.push(v);
        }
      }
    }

    const count = values.length;
    if (count === 0) {
      return {
        count: 0,
        avgMs: 0,
        minMs: 0,
        p50Ms: 0,
        p95Ms: 0,
        p99Ms: 0,
        maxMs: 0,
      };
    }

    values.sort((a, b) => a - b);

    let sum = 0;
    for (let i = 0; i < count; i += 1) {
      sum += values[i];
    }

    return {
      count,
      avgMs: sum / count,
      minMs: values[0],
      p50Ms: percentileFromSorted(values, count, 0.5),
      p95Ms: percentileFromSorted(values, count, 0.95),
      p99Ms: percentileFromSorted(values, count, 0.99),
      maxMs: values[count - 1],
    };
  }

  function hasTimestampQueryFeature(device) {
    if (!device || typeof device !== 'object') {
      return false;
    }
    const features = device.features;
    if (!features || typeof features.has !== 'function') {
      return false;
    }
    try {
      if (!features.has('timestamp-query')) {
        return false;
      }
    } catch (_) {
      return false;
    }
    return typeof device.createQuerySet === 'function' && typeof device.createBuffer === 'function';
  }

  function isSlotAvailable(slot) {
    if (!slot || slot.state !== 'idle' || !slot.buffer || slot.buffer.destroyed) {
      return false;
    }
    const mapState = slot.buffer.mapState;
    return mapState === undefined || mapState === 'unmapped';
  }

  function readTimestampsView(mappedRange, expectedCount) {
    if (mappedRange instanceof BigUint64Array) {
      return mappedRange;
    }
    if (ArrayBuffer.isView(mappedRange)) {
      const count = Math.min(
        expectedCount,
        Math.floor(mappedRange.byteLength / TIMESTAMP_BYTE_SIZE)
      );
      return new BigUint64Array(mappedRange.buffer, mappedRange.byteOffset, count);
    }
    if (mappedRange && typeof mappedRange.byteLength === 'number') {
      const count = Math.min(
        expectedCount,
        Math.floor(mappedRange.byteLength / TIMESTAMP_BYTE_SIZE)
      );
      return new BigUint64Array(mappedRange, 0, count);
    }
    return new BigUint64Array(0);
  }

  function computeDurationMs(startNs, endNs) {
    if (typeof startNs !== 'bigint' || typeof endNs !== 'bigint') {
      return 0;
    }
    if (endNs <= startNs) {
      return 0;
    }
    const durationMs = Number(endNs - startNs) / NS_PER_MS;
    if (!Number.isFinite(durationMs) || durationMs < 0) {
      return 0;
    }
    return durationMs;
  }

  function defaultNowMs() {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
      return performance.now();
    }
    return Date.now();
  }

  /**
   * Non-blocking WebGPU per-pass GPU profiler backed by optional `'timestamp-query'`
   * `GPUQuerySet` resolution and multi-slot staging buffer readback.
   *
   * Supports named pass profiling via `beginPass(label)` / `endPass()` and
   * `getTopPasses(limit)` bottleneck summaries. When WebGPU `'timestamp-query'`
   * is unavailable, `beginPass(label)` / `endPass()` gracefully degrade to CPU
   * `performance.now()` timing while keeping GPU-only methods (`getPassTimestampWrites`,
   * `resolveFrame`, `collectCompletedReadback`) as safe no-ops.
   */
  class GpuPassTimer {
    /**
     * @param {object} [options]
     * @param {GPUDevice} [options.device]
     * @param {number} [options.maxPassesPerFrame=32]
     * @param {number} [options.readbackSlots=3]
     * @param {number} [options.historySize=120]
     * @param {number} [options.frameBudgetMs=16.67]
     * @param {() => number} [options.now] Optional clock returning milliseconds (defaults to `performance.now()`).
     */
    constructor(options = {}) {
      const opts = options && typeof options === 'object' ? options : {};
      const device = opts.device || null;

      this._maxPassesPerFrame = normalizePositiveInt(
        opts.maxPassesPerFrame,
        DEFAULT_MAX_PASSES_PER_FRAME,
        'maxPassesPerFrame'
      );
      this._readbackSlotCount = normalizePositiveInt(
        opts.readbackSlots,
        DEFAULT_READBACK_SLOTS,
        'readbackSlots'
      );
      this._historySize = normalizePositiveInt(
        opts.historySize,
        DEFAULT_HISTORY_SIZE,
        'historySize'
      );
      this._frameBudgetMs = normalizePositiveNumber(
        opts.frameBudgetMs,
        DEFAULT_FRAME_BUDGET_MS,
        'frameBudgetMs'
      );
      if (opts.now !== undefined && typeof opts.now !== 'function') {
        throw new TypeError('now must be a function');
      }
      this._now = opts.now || defaultNowMs;

      this._device = device;
      this._maxQueries = this._maxPassesPerFrame * 2;
      this._bufferByteSize = this._maxQueries * TIMESTAMP_BYTE_SIZE;

      this._supported = false;
      this._destroyed = false;
      this._deviceLost = false;

      this._querySet = null;
      this._resolveBuffer = null;
      this._readbackSlots = [];
      this._timestampWriteDescriptors = [];
      this._pendingRing = new Array(this._readbackSlotCount).fill(null);
      this._pendingHead = 0;
      this._pendingTail = 0;
      this._pendingCount = 0;

      // Pre-allocated per-frame pass recording storage (zero hot-path allocations).
      this._framePassNames = new Array(this._maxPassesPerFrame);
      this._framePassBeginIdx = new Uint32Array(this._maxPassesPerFrame);
      this._framePassEndIdx = new Uint32Array(this._maxPassesPerFrame);
      this._framePassCount = 0;
      this._queryCursor = 0;

      // Pre-allocated active pass stack for beginPass(label) / endPass().
      this._activePassLabels = new Array(this._maxPassesPerFrame);
      this._activePassStarts = new Float64Array(this._maxPassesPerFrame);
      this._activePassModes = new Uint8Array(this._maxPassesPerFrame); // 0 = GPU, 1 = CPU
      this._activePassDepth = 0;

      // Pre-allocated rolling frame total ring for share-of-frame % calculations.
      this._frameTotals = new Float64Array(this._historySize);
      this._frameTotalSeqs = new Float64Array(this._historySize);
      this._frameTotalWriteIdx = 0;
      this._frameTotalWindowCount = 0;
      this._frameSeq = 0;

      // Pre-allocated rolling pass duration ring for global p50PassMs / p95PassMs stats.
      this._passDurations = new Float64Array(this._historySize);
      this._passDurationWriteIdx = 0;
      this._passDurationWindowCount = 0;

      // Open CPU fallback frame accumulators (active only when !supported and beginPass/endPass are used).
      this._cpuFrameOpen = false;
      this._cpuFrameSeq = 0;
      this._cpuFrameTotalMs = 0;
      this._cpuFramePassCount = 0;

      this._resolvedFrames = 0;
      this._skippedFrames = 0;
      this._droppedPasses = 0;
      this._droppedQueryFrames = 0;
      this._mapErrors = 0;
      this._lastFrameGpuMs = 0;
      this._resetGeneration = 0;
      this._passHistories = new Map();

      if (hasTimestampQueryFeature(device)) {
        this._initGpuResources(device);
      }
    }

    _initGpuResources(device) {
      let querySet = null;
      let resolveBuffer = null;
      const readbackSlots = [];

      try {
        querySet = device.createQuerySet({
          label: 'playgta6-gpu-timer-queries',
          type: 'timestamp',
          count: this._maxQueries,
        });
        if (!querySet) {
          throw new Error('createQuerySet returned empty query set');
        }

        resolveBuffer = device.createBuffer({
          label: 'playgta6-gpu-timer-resolve',
          size: this._bufferByteSize,
          usage: BUFFER_USAGE.QUERY_RESOLVE | BUFFER_USAGE.COPY_SRC,
        });
        if (!resolveBuffer) {
          throw new Error('createBuffer returned empty resolve buffer');
        }

        for (let i = 0; i < this._readbackSlotCount; i += 1) {
          const buffer = device.createBuffer({
            label: `playgta6-gpu-timer-readback-${i}`,
            size: this._bufferByteSize,
            usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
          });
          if (!buffer) {
            throw new Error(`createBuffer returned empty readback buffer ${i}`);
          }
          readbackSlots.push({
            index: i,
            buffer,
            state: 'idle',
            passNames: new Array(this._maxPassesPerFrame),
            passBeginIdx: new Uint32Array(this._maxPassesPerFrame),
            passEndIdx: new Uint32Array(this._maxPassesPerFrame),
            passCount: 0,
            queryCount: 0,
            copyBytes: 0,
            generation: 0,
          });
        }

        const timestampWriteDescriptors = new Array(this._maxPassesPerFrame);
        for (let i = 0; i < this._maxPassesPerFrame; i += 1) {
          timestampWriteDescriptors[i] = Object.freeze({
            querySet,
            beginningOfPassWriteIndex: i * 2,
            endOfPassWriteIndex: i * 2 + 1,
          });
        }

        this._querySet = querySet;
        this._resolveBuffer = resolveBuffer;
        this._readbackSlots = readbackSlots;
        this._timestampWriteDescriptors = timestampWriteDescriptors;
        this._supported = true;

        if (device && device.lost && typeof device.lost.then === 'function') {
          device.lost.then(
            () => {
              this._deviceLost = true;
            },
            () => {
              this._deviceLost = true;
            }
          );
        }
      } catch (_) {
        if (querySet && typeof querySet.destroy === 'function') {
          try {
            querySet.destroy();
          } catch (__) {
            // ignore cleanup errors
          }
        }
        if (resolveBuffer && typeof resolveBuffer.destroy === 'function') {
          try {
            resolveBuffer.destroy();
          } catch (__) {
            // ignore cleanup errors
          }
        }
        for (const slot of readbackSlots) {
          if (slot.buffer && typeof slot.buffer.destroy === 'function') {
            try {
              slot.buffer.destroy();
            } catch (__) {
              // ignore cleanup errors
            }
          }
        }
        this._querySet = null;
        this._resolveBuffer = null;
        this._readbackSlots = [];
        this._timestampWriteDescriptors = [];
        this._supported = false;
      }
    }

    get supported() {
      return this._supported;
    }

    get mode() {
      return this._supported ? 'gpu' : 'cpu';
    }

    get maxPassesPerFrame() {
      return this._maxPassesPerFrame;
    }

    get readbackSlots() {
      return this._readbackSlotCount;
    }

    get historySize() {
      return this._historySize;
    }

    get frameBudgetMs() {
      return this._frameBudgetMs;
    }

    get isDestroyed() {
      return this._destroyed;
    }

    _recordFrameTotal(totalMs, frameSeq) {
      const safeTotalMs =
        typeof totalMs === 'number' && Number.isFinite(totalMs) && totalMs >= 0 ? totalMs : 0;
      this._frameTotals[this._frameTotalWriteIdx] = safeTotalMs;
      this._frameTotalSeqs[this._frameTotalWriteIdx] = frameSeq;
      this._frameTotalWriteIdx = (this._frameTotalWriteIdx + 1) % this._historySize;
      if (this._frameTotalWindowCount < this._historySize) {
        this._frameTotalWindowCount += 1;
      }
    }

    _finalizeCpuFrame() {
      if (this._cpuFramePassCount === 0) {
        this._cpuFrameOpen = false;
        this._cpuFrameSeq = 0;
        this._cpuFrameTotalMs = 0;
        return false;
      }
      const frameSeq = this._cpuFrameSeq;
      const totalMs = this._cpuFrameTotalMs;
      this._frameSeq = frameSeq;
      this._recordFrameTotal(totalMs, frameSeq);
      this._resolvedFrames += 1;
      this._lastFrameGpuMs = totalMs;

      this._cpuFrameOpen = false;
      this._cpuFrameSeq = 0;
      this._cpuFrameTotalMs = 0;
      this._cpuFramePassCount = 0;
      return true;
    }

    /**
     * Resets per-frame pass metadata and query cursor for a new frame.
     * In CPU fallback mode, commits any pending passes recorded during the previous frame.
     */
    beginFrame() {
      if (this._destroyed) {
        return;
      }
      if (this._supported) {
        this._framePassCount = 0;
        this._queryCursor = 0;
        this._activePassDepth = 0;
        return;
      }
      if (this._cpuFramePassCount > 0) {
        this._finalizeCpuFrame();
      }
      this._activePassDepth = 0;
      this._cpuFrameOpen = true;
      this._cpuFrameSeq = 0;
      this._cpuFrameTotalMs = 0;
      this._cpuFramePassCount = 0;
    }

    /**
     * Explicitly finalizes the current frame in CPU fallback mode.
     * In GPU mode this is a safe no-op (frames finalize when `collectCompletedReadback()` resolves).
     *
     * @returns {boolean} True if a CPU fallback frame with recorded passes was committed.
     */
    endFrame() {
      if (this._destroyed) {
        return false;
      }
      this._activePassDepth = 0;
      if (this._supported) {
        return false;
      }
      return this._finalizeCpuFrame();
    }

    /**
     * Begins a named pass scope without hot-path heap allocations.
     *
     * - When WebGPU `'timestamp-query'` is supported: allocates a query pair and returns
     *   the pre-allocated `timestampWrites` descriptor for `beginRenderPass` / `beginComputePass`.
     * - When `'timestamp-query'` is unavailable: captures a CPU `performance.now()` start
     *   timestamp and returns `undefined`.
     *
     * @param {string} [label]
     * @returns {{ querySet: GPUQuerySet, beginningOfPassWriteIndex: number, endOfPassWriteIndex: number } | undefined}
     */
    beginPass(label) {
      if (this._destroyed) {
        return undefined;
      }

      if (this._supported) {
        const descriptor = this.getPassTimestampWrites(label);
        if (descriptor !== undefined && this._activePassDepth < this._maxPassesPerFrame) {
          const depth = this._activePassDepth;
          this._activePassLabels[depth] = this._framePassNames[this._framePassCount - 1];
          this._activePassStarts[depth] = 0;
          this._activePassModes[depth] = 0;
          this._activePassDepth = depth + 1;
        }
        return descriptor;
      }

      if (this._activePassDepth >= this._maxPassesPerFrame) {
        this._droppedPasses += 1;
        return undefined;
      }

      const depth = this._activePassDepth;
      const name =
        typeof label === 'string' && label.length > 0
          ? label
          : `pass_${this._cpuFramePassCount}`;
      this._activePassLabels[depth] = name;
      this._activePassStarts[depth] = this._now();
      this._activePassModes[depth] = 1;
      this._activePassDepth = depth + 1;
      return undefined;
    }

    /**
     * Ends the most recently started pass from `beginPass(label)`.
     *
     * - In CPU fallback mode: records the elapsed `performance.now()` duration into the
     *   pass's rolling history and returns the duration in milliseconds.
     * - In GPU timestamp-query mode: closes the active pass scope and returns `null`
     *   (GPU timings are recorded asynchronously when `collectCompletedReadback()` resolves).
     *
     * @returns {number | null}
     */
    endPass() {
      if (this._destroyed || this._activePassDepth === 0) {
        return null;
      }

      const depth = this._activePassDepth - 1;
      this._activePassDepth = depth;
      const mode = this._activePassModes[depth];
      const label = this._activePassLabels[depth];
      this._activePassLabels[depth] = undefined;

      if (mode === 0) {
        return null;
      }

      const endMs = this._now();
      const rawDelta = endMs - this._activePassStarts[depth];
      const durationMs =
        typeof rawDelta === 'number' && Number.isFinite(rawDelta) && rawDelta > 0 ? rawDelta : 0;

      if (this._cpuFramePassCount === 0) {
        this._cpuFrameSeq = this._frameSeq + 1;
        this._cpuFrameOpen = true;
      }
      this._cpuFrameTotalMs += durationMs;
      this._cpuFramePassCount += 1;
      this._recordPassSample(label, durationMs, this._cpuFrameSeq);
      return durationMs;
    }

    /**
     * Allocates a `(beginningOfPassWriteIndex, endOfPassWriteIndex)` pair for a render
     * or compute pass descriptor's `timestampWrites` field.
     *
     * Returns `undefined` when `'timestamp-query'` is unsupported, the timer is destroyed,
     * or the per-frame query budget is exhausted.
     *
     * @param {string} [passName]
     * @returns {{ querySet: GPUQuerySet, beginningOfPassWriteIndex: number, endOfPassWriteIndex: number } | undefined}
     */
    getPassTimestampWrites(passName) {
      if (!this._supported || this._destroyed || !this._querySet) {
        return undefined;
      }
      if (this._queryCursor + 2 > this._maxQueries) {
        this._droppedPasses += 1;
        return undefined;
      }

      const passIdx = this._framePassCount;
      const beginningOfPassWriteIndex = this._queryCursor;
      const endOfPassWriteIndex = this._queryCursor + 1;
      this._queryCursor += 2;
      this._framePassCount = passIdx + 1;

      const name =
        typeof passName === 'string' && passName.length > 0 ? passName : `pass_${passIdx}`;

      this._framePassNames[passIdx] = name;
      this._framePassBeginIdx[passIdx] = beginningOfPassWriteIndex;
      this._framePassEndIdx[passIdx] = endOfPassWriteIndex;

      return this._timestampWriteDescriptors[passIdx];
    }

    /**
     * Encodes `resolveQuerySet` + `copyBufferToBuffer` into an available readback slot.
     * Skips cleanly (incrementing `skippedFrames`) if all readback slots are currently
     * in-flight or mapped so rendering never stalls.
     *
     * In CPU fallback mode, commits any passes recorded via `beginPass()`/`endPass()`
     * for the current frame and returns `false` (no GPU resolve commands encoded).
     *
     * @param {GPUCommandEncoder} encoder
     * @returns {boolean} True if resolve commands were encoded, false if skipped or no-op.
     */
    resolveFrame(encoder) {
      if (this._destroyed) {
        return false;
      }
      if (!this._supported) {
        if (this._cpuFramePassCount > 0) {
          this._finalizeCpuFrame();
        }
        this._activePassDepth = 0;
        return false;
      }
      if (!this._querySet || !this._resolveBuffer) {
        return false;
      }
      const passCount = this._framePassCount;
      if (passCount === 0) {
        return false;
      }
      if (
        !encoder ||
        typeof encoder.resolveQuerySet !== 'function' ||
        typeof encoder.copyBufferToBuffer !== 'function'
      ) {
        return false;
      }

      if (this._deviceLost) {
        this._droppedQueryFrames += 1;
        this._framePassCount = 0;
        this._queryCursor = 0;
        this._activePassDepth = 0;
        return false;
      }

      let freeSlot = null;
      for (let i = 0; i < this._readbackSlots.length; i += 1) {
        if (isSlotAvailable(this._readbackSlots[i])) {
          freeSlot = this._readbackSlots[i];
          break;
        }
      }

      if (!freeSlot) {
        this._skippedFrames += 1;
        this._droppedQueryFrames += 1;
        this._framePassCount = 0;
        this._queryCursor = 0;
        this._activePassDepth = 0;
        return false;
      }

      const queryCount = this._queryCursor;
      const copyBytes = queryCount * TIMESTAMP_BYTE_SIZE;

      try {
        encoder.resolveQuerySet(this._querySet, 0, queryCount, this._resolveBuffer, 0);
        encoder.copyBufferToBuffer(this._resolveBuffer, 0, freeSlot.buffer, 0, copyBytes);
      } catch (_) {
        this._droppedQueryFrames += 1;
        this._framePassCount = 0;
        this._queryCursor = 0;
        this._activePassDepth = 0;
        return false;
      }

      freeSlot.state = 'pending';
      for (let i = 0; i < passCount; i += 1) {
        freeSlot.passNames[i] = this._framePassNames[i];
        freeSlot.passBeginIdx[i] = this._framePassBeginIdx[i];
        freeSlot.passEndIdx[i] = this._framePassEndIdx[i];
      }
      freeSlot.passCount = passCount;
      freeSlot.queryCount = queryCount;
      freeSlot.copyBytes = copyBytes;
      freeSlot.generation = this._resetGeneration;

      this._pendingRing[this._pendingTail] = freeSlot;
      this._pendingTail = (this._pendingTail + 1) % this._readbackSlotCount;
      this._pendingCount += 1;

      this._framePassCount = 0;
      this._queryCursor = 0;
      this._activePassDepth = 0;
      return true;
    }

    _recordPassSample(name, durationMs, frameSeq) {
      const safeDurationMs =
        typeof durationMs === 'number' && Number.isFinite(durationMs) && durationMs >= 0
          ? durationMs
          : 0;

      this._passDurations[this._passDurationWriteIdx] = safeDurationMs;
      this._passDurationWriteIdx = (this._passDurationWriteIdx + 1) % this._historySize;
      if (this._passDurationWindowCount < this._historySize) {
        this._passDurationWindowCount += 1;
      }

      let entry = this._passHistories.get(name);
      if (!entry) {
        entry = {
          samples: new Float64Array(this._historySize),
          writeIdx: 0,
          windowCount: 0,
          count: 0,
          lastMs: 0,
          budgetExceedCount: 0,
          frameSamples: new Float64Array(this._historySize),
          frameSeqs: new Float64Array(this._historySize),
          frameWriteIdx: 0,
          frameWindowCount: 0,
          lastFrameSeq: 0,
          lastFrameSlotIdx: -1,
        };
        this._passHistories.set(name, entry);
      }

      entry.samples[entry.writeIdx] = safeDurationMs;
      entry.writeIdx = (entry.writeIdx + 1) % this._historySize;
      if (entry.windowCount < this._historySize) {
        entry.windowCount += 1;
      }
      entry.count += 1;
      entry.lastMs = safeDurationMs;
      if (safeDurationMs > this._frameBudgetMs) {
        entry.budgetExceedCount += 1;
      }

      if (typeof frameSeq === 'number' && frameSeq > 0) {
        if (entry.lastFrameSeq === frameSeq && entry.lastFrameSlotIdx >= 0) {
          entry.frameSamples[entry.lastFrameSlotIdx] += safeDurationMs;
        } else {
          const slotIdx = entry.frameWriteIdx;
          entry.frameSamples[slotIdx] = safeDurationMs;
          entry.frameSeqs[slotIdx] = frameSeq;
          entry.lastFrameSeq = frameSeq;
          entry.lastFrameSlotIdx = slotIdx;
          entry.frameWriteIdx = (slotIdx + 1) % this._historySize;
          if (entry.frameWindowCount < this._historySize) {
            entry.frameWindowCount += 1;
          }
        }
      }
    }

    /**
     * Asynchronously maps the oldest submitted readback buffer, converts `BigUint64Array`
     * nanosecond timestamps to milliseconds, updates per-pass rolling stats, unmaps the
     * buffer, and returns the frame's pass timings.
     *
     * @returns {Promise<{ totalGpuMs: number, passes: Array<{ name: string, durationMs: number }> } | null>}
     */
    async collectCompletedReadback() {
      if (!this._supported || this._destroyed) {
        return null;
      }
      if (this._pendingCount === 0) {
        return null;
      }

      const slot = this._pendingRing[this._pendingHead];
      this._pendingRing[this._pendingHead] = null;
      this._pendingHead = (this._pendingHead + 1) % this._readbackSlotCount;
      this._pendingCount -= 1;

      if (!slot || !slot.buffer || slot.state !== 'pending') {
        if (!this._destroyed) {
          this._droppedQueryFrames += 1;
        }
        return null;
      }

      const buffer = slot.buffer;
      const passCount = slot.passCount;
      const queryCount = slot.queryCount;
      const copyBytes = slot.copyBytes;
      const slotGeneration = slot.generation;

      slot.state = 'mapping';

      let mapped = false;
      let unmapAttempted = false;
      try {
        if (buffer.destroyed) {
          throw new Error('Readback buffer was destroyed before mapAsync');
        }
        if (typeof buffer.mapAsync !== 'function') {
          throw new TypeError('Readback buffer does not implement mapAsync');
        }

        await buffer.mapAsync(MAP_MODE.READ);
        mapped = true;

        if (
          this._destroyed ||
          slot.buffer !== buffer ||
          slotGeneration !== this._resetGeneration
        ) {
          return null;
        }

        if (typeof buffer.getMappedRange !== 'function') {
          throw new TypeError('Readback buffer does not implement getMappedRange');
        }

        const range = buffer.getMappedRange(0, copyBytes);
        if (!range) {
          throw new Error('Readback buffer getMappedRange returned empty range');
        }
        const timestamps = readTimestampsView(range, queryCount);

        let totalGpuMs = 0;
        const passes = new Array(passCount);

        for (let i = 0; i < passCount; i += 1) {
          const passName = slot.passNames[i];
          const beginIdx = slot.passBeginIdx[i];
          const endIdx = slot.passEndIdx[i];
          const startNs = beginIdx < timestamps.length ? timestamps[beginIdx] : 0n;
          const endNs = endIdx < timestamps.length ? timestamps[endIdx] : 0n;
          const durationMs = computeDurationMs(startNs, endNs);
          totalGpuMs += durationMs;
          passes[i] = { name: passName, durationMs };
        }

        if (typeof buffer.unmap === 'function') {
          unmapAttempted = true;
          mapped = false;
          buffer.unmap();
        } else {
          mapped = false;
        }

        const frameSeq = this._frameSeq + 1;
        this._frameSeq = frameSeq;

        for (let i = 0; i < passCount; i += 1) {
          this._recordPassSample(passes[i].name, passes[i].durationMs, frameSeq);
        }

        this._recordFrameTotal(totalGpuMs, frameSeq);
        this._resolvedFrames += 1;
        this._lastFrameGpuMs = totalGpuMs;

        return { totalGpuMs, passes };
      } catch (_) {
        if (!this._destroyed && slotGeneration === this._resetGeneration) {
          this._mapErrors += 1;
          this._droppedQueryFrames += 1;
        }
        return null;
      } finally {
        const stillMapped =
          !unmapAttempted &&
          buffer &&
          !buffer.destroyed &&
          (mapped || buffer.mapState === 'mapped');
        if (stillMapped && typeof buffer.unmap === 'function') {
          try {
            buffer.unmap();
          } catch (_) {
            // ignore secondary unmap errors if buffer was destroyed or device was lost
          }
        }
        if (!this._destroyed && slot.buffer === buffer) {
          slot.state = 'idle';
          slot.passCount = 0;
          slot.passNames.fill(undefined);
          slot.queryCount = 0;
          slot.copyBytes = 0;
          slot.generation = this._resetGeneration;
        }
      }
    }

    /**
     * Returns rolling percentile statistics and budget-exceedance spike count for
     * a single pass label, or `null` if no samples have been recorded for it.
     *
     * @param {string} label
     * @returns {{
     *   count: number,
     *   totalCount: number,
     *   lastMs: number,
     *   avgMs: number,
     *   minMs: number,
     *   p50Ms: number,
     *   p95Ms: number,
     *   p99Ms: number,
     *   maxMs: number,
     *   budgetExceedCount: number
     * } | null}
     */
    getPassPercentiles(label) {
      if (typeof label !== 'string' || label.length === 0) {
        return null;
      }
      const entry = this._passHistories.get(label);
      if (!entry || entry.windowCount === 0) {
        return null;
      }
      const windowSamples = entry.samples.subarray(0, entry.windowCount);
      const percentiles = computePercentiles(windowSamples);
      return {
        count: percentiles.count,
        totalCount: entry.count,
        lastMs: entry.lastMs,
        avgMs: percentiles.avgMs,
        minMs: percentiles.minMs,
        p50Ms: percentiles.p50Ms,
        p95Ms: percentiles.p95Ms,
        p99Ms: percentiles.p99Ms,
        maxMs: percentiles.maxMs,
        budgetExceedCount: entry.budgetExceedCount,
      };
    }

    /**
     * Returns a sanitized numeric telemetry summary including budget configuration,
     * rolling window capacity, per-pass percentile breakdowns, and total spike count.
     *
     * @returns {{
     *   supported: boolean,
     *   frameBudgetMs: number,
     *   historySize: number,
     *   resolvedFrames: number,
     *   skippedFrames: number,
     *   droppedPasses: number,
     *   lastFrameGpuMs: number,
     *   budgetExceedCount: number,
     *   passes: Record<string, {
     *     count: number,
     *     totalCount: number,
     *     lastMs: number,
     *     avgMs: number,
     *     minMs: number,
     *     p50Ms: number,
     *     p95Ms: number,
     *     p99Ms: number,
     *     maxMs: number,
     *     budgetExceedCount: number
     *   }>
     * }}
     */
    getSummary() {
      const passes = {};
      let budgetExceedCount = 0;

      for (const name of this._passHistories.keys()) {
        const passStats = this.getPassPercentiles(name);
        if (passStats) {
          passes[name] = passStats;
          budgetExceedCount += passStats.budgetExceedCount;
        }
      }

      return {
        supported: this._supported,
        frameBudgetMs: this._frameBudgetMs,
        historySize: this._historySize,
        resolvedFrames: this._resolvedFrames,
        skippedFrames: this._skippedFrames,
        droppedPasses: this._droppedPasses,
        lastFrameGpuMs: this._lastFrameGpuMs,
        budgetExceedCount,
        passes,
      };
    }

    /**
     * Returns the slowest passes ranked by rolling window frame contribution (highest
     * share-of-frame % first), computing rolling per-pass average ms, max ms, and
     * share-of-frame % over the active `historySize` rolling window.
     *
     * @param {number} [limit] Maximum number of passes to return (defaults to all active passes).
     * @returns {Array<{
     *   label: string,
     *   name: string,
     *   count: number,
     *   totalCount: number,
     *   avgMs: number,
     *   maxMs: number,
     *   lastMs: number,
     *   totalMs: number,
     *   shareOfFramePct: number,
     *   mode: 'gpu' | 'cpu'
     * }>}
     */
    getTopPasses(limit) {
      let maxResults = Infinity;
      if (limit !== undefined && limit !== null) {
        if (typeof limit !== 'number' || Number.isNaN(limit) || limit <= 0) {
          return [];
        }
        maxResults = Number.isFinite(limit) ? Math.floor(limit) : Infinity;
        if (maxResults <= 0) {
          return [];
        }
      }

      const hasOpenCpuFrame = !this._supported && this._cpuFramePassCount > 0;
      const finalizedCount = this._frameTotalWindowCount;
      if (finalizedCount === 0 && !hasOpenCpuFrame) {
        return [];
      }

      let maxSeq = 0;
      let minSeq = 0;
      let rollingFrameTotalMs = 0;

      if (hasOpenCpuFrame) {
        const activeFrameCount = Math.min(this._historySize, finalizedCount + 1);
        maxSeq = this._cpuFrameSeq;
        minSeq = maxSeq - activeFrameCount + 1;
        rollingFrameTotalMs = this._cpuFrameTotalMs;
        for (let i = 0; i < finalizedCount; i += 1) {
          const seq = this._frameTotalSeqs[i];
          if (seq >= minSeq && seq <= maxSeq) {
            rollingFrameTotalMs += this._frameTotals[i];
          }
        }
      } else {
        maxSeq = this._frameSeq;
        minSeq = maxSeq - finalizedCount + 1;
        for (let i = 0; i < finalizedCount; i += 1) {
          rollingFrameTotalMs += this._frameTotals[i];
        }
      }

      const mode = this._supported ? 'gpu' : 'cpu';
      const ranked = [];

      for (const [label, entry] of this._passHistories.entries()) {
        let activeCount = 0;
        let sum = 0;
        let maxMs = 0;
        let lastMs = 0;
        let latestSeq = -1;

        for (let i = 0; i < entry.frameWindowCount; i += 1) {
          const seq = entry.frameSeqs[i];
          if (seq >= minSeq && seq <= maxSeq) {
            const v = entry.frameSamples[i];
            activeCount += 1;
            sum += v;
            if (v > maxMs) {
              maxMs = v;
            }
            if (seq >= latestSeq) {
              latestSeq = seq;
              lastMs = v;
            }
          }
        }

        if (activeCount === 0) {
          continue;
        }

        const avgMs = sum / activeCount;
        const shareOfFramePct =
          rollingFrameTotalMs > 0 ? (sum / rollingFrameTotalMs) * 100 : 0;

        ranked.push({
          label,
          name: label,
          count: activeCount,
          totalCount: entry.count,
          avgMs,
          maxMs,
          lastMs,
          totalMs: sum,
          shareOfFramePct,
          mode,
        });
      }

      ranked.sort((a, b) => {
        if (b.totalMs !== a.totalMs) {
          return b.totalMs - a.totalMs;
        }
        if (b.avgMs !== a.avgMs) {
          return b.avgMs - a.avgMs;
        }
        if (b.maxMs !== a.maxMs) {
          return b.maxMs - a.maxMs;
        }
        if (a.label < b.label) {
          return -1;
        }
        if (a.label > b.label) {
          return 1;
        }
        return 0;
      });

      if (ranked.length > maxResults) {
        ranked.length = maxResults;
      }
      return ranked;
    }

    /**
     * Clears rolling per-pass histories, percentile samples, and budget spike counters
     * while keeping GPU resources intact for subsequent frames.
     */
    reset() {
      this._resetGeneration += 1;
      this._resolvedFrames = 0;
      this._skippedFrames = 0;
      this._droppedPasses = 0;
      this._droppedQueryFrames = 0;
      this._mapErrors = 0;
      this._lastFrameGpuMs = 0;
      this._passHistories.clear();
      this._passDurations.fill(0);
      this._passDurationWriteIdx = 0;
      this._passDurationWindowCount = 0;
      this._framePassCount = 0;
      this._framePassNames.fill(undefined);
      this._queryCursor = 0;
      this._activePassDepth = 0;
      this._activePassLabels.fill(undefined);
      this._frameTotals.fill(0);
      this._frameTotalSeqs.fill(0);
      this._frameTotalWriteIdx = 0;
      this._frameTotalWindowCount = 0;
      this._frameSeq = 0;
      this._cpuFrameOpen = false;
      this._cpuFrameSeq = 0;
      this._cpuFrameTotalMs = 0;
      this._cpuFramePassCount = 0;

      for (let i = 0; i < this._pendingCount; i += 1) {
        const idx = (this._pendingHead + i) % this._readbackSlotCount;
        const slot = this._pendingRing[idx];
        this._pendingRing[idx] = null;
        if (slot && slot.state === 'pending') {
          slot.state = 'idle';
          slot.passCount = 0;
          slot.passNames.fill(undefined);
          slot.queryCount = 0;
          slot.copyBytes = 0;
          slot.generation = this._resetGeneration;
        }
      }
      this._pendingHead = 0;
      this._pendingTail = 0;
      this._pendingCount = 0;
    }

    /**
     * Resets profiler counters, error counts, and rolling pass-duration windows
     * while keeping GPU resources intact, returning the fresh `getStats()` snapshot.
     */
    resetStats() {
      this.reset();
      return this.getStats();
    }

    /**
     * Returns a snapshot of overall profiler counters and per-pass rolling stats.
     *
     * @returns {{
     *   supported: boolean,
     *   resolvedFrames: number,
     *   skippedFrames: number,
     *   droppedPasses: number,
     *   droppedQueryFrames: number,
     *   mapErrors: number,
     *   lastFrameGpuMs: number,
     *   p50PassMs: number,
     *   p95PassMs: number,
     *   passStats: Record<string, { count: number, lastMs: number, avgMs: number, maxMs: number }>
     * }}
     */
    getStats() {
      const passStats = {};
      for (const [name, entry] of this._passHistories.entries()) {
        let sum = 0;
        let maxMs = 0;
        for (let i = 0; i < entry.windowCount; i += 1) {
          const v = entry.samples[i];
          sum += v;
          if (v > maxMs) {
            maxMs = v;
          }
        }
        const avgMs = entry.windowCount > 0 ? sum / entry.windowCount : 0;
        passStats[name] = {
          count: entry.count,
          lastMs: entry.lastMs,
          avgMs,
          maxMs,
        };
      }

      const rollingPassPercentiles = computePercentiles(
        this._passDurations.subarray(0, this._passDurationWindowCount)
      );

      return {
        supported: this._supported,
        resolvedFrames: this._resolvedFrames,
        skippedFrames: this._skippedFrames,
        droppedPasses: this._droppedPasses,
        droppedQueryFrames: this._droppedQueryFrames,
        mapErrors: this._mapErrors,
        lastFrameGpuMs: this._lastFrameGpuMs,
        p50PassMs: rollingPassPercentiles.p50Ms,
        p95PassMs: rollingPassPercentiles.p95Ms,
        passStats,
      };
    }

    /**
     * Idempotently destroys the GPUQuerySet and all resolve/readback GPUBuffers.
     */
    destroy() {
      if (this._destroyed) {
        return;
      }
      this._destroyed = true;

      if (this._querySet && typeof this._querySet.destroy === 'function') {
        try {
          this._querySet.destroy();
        } catch (_) {
          // ignore cleanup errors
        }
      }
      this._querySet = null;

      if (this._resolveBuffer && typeof this._resolveBuffer.destroy === 'function') {
        try {
          this._resolveBuffer.destroy();
        } catch (_) {
          // ignore cleanup errors
        }
      }
      this._resolveBuffer = null;

      for (const slot of this._readbackSlots) {
        if (slot.buffer && typeof slot.buffer.destroy === 'function') {
          try {
            slot.buffer.destroy();
          } catch (_) {
            // ignore cleanup errors
          }
        }
        slot.buffer = null;
        slot.state = 'destroyed';
        slot.passCount = 0;
        slot.passNames.fill(undefined);
        slot.queryCount = 0;
        slot.copyBytes = 0;
      }
      this._readbackSlots = [];
      this._timestampWriteDescriptors = [];
      this._pendingRing.fill(null);
      this._pendingHead = 0;
      this._pendingTail = 0;
      this._pendingCount = 0;
      this.reset();
      this._device = null;
    }
  }

  const exportedApi = {
    DEFAULT_MAX_PASSES_PER_FRAME,
    DEFAULT_READBACK_SLOTS,
    DEFAULT_HISTORY_SIZE,
    DEFAULT_FRAME_BUDGET_MS,
    TIMESTAMP_BYTE_SIZE,
    BUFFER_USAGE,
    MAP_MODE,
    computePercentiles,
    GpuPassTimer,
  };

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6GpuTimer = exportedApi;
  } else if (root) {
    root.PlayGta6GpuTimer = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
