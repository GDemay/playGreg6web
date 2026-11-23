'use strict';

(function (root) {
  const DEFAULT_MAX_PASSES_PER_FRAME = 32;
  const DEFAULT_READBACK_SLOTS = 3;
  const DEFAULT_HISTORY_SIZE = 60;
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
    if (!slot || slot.state !== 'idle' || !slot.buffer) {
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

  /**
   * Non-blocking WebGPU per-pass GPU profiler backed by optional `'timestamp-query'`
   * `GPUQuerySet` resolution and multi-slot staging buffer readback.
   *
   * Degrades cleanly to a zero-overhead no-op whenever the GPUDevice does not support
   * `'timestamp-query'` (or after `destroy()`).
   */
  class GpuPassTimer {
    /**
     * @param {object} [options]
     * @param {GPUDevice} [options.device]
     * @param {number} [options.maxPassesPerFrame=32]
     * @param {number} [options.readbackSlots=3]
     * @param {number} [options.historySize=60]
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

      this._device = device;
      this._maxQueries = this._maxPassesPerFrame * 2;
      this._bufferByteSize = this._maxQueries * TIMESTAMP_BYTE_SIZE;

      this._supported = false;
      this._destroyed = false;

      this._querySet = null;
      this._resolveBuffer = null;
      this._readbackSlots = [];
      this._pendingQueue = [];

      this._framePasses = [];
      this._queryCursor = 0;

      this._resolvedFrames = 0;
      this._skippedFrames = 0;
      this._droppedPasses = 0;
      this._lastFrameGpuMs = 0;
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
            passes: null,
            queryCount: 0,
            copyBytes: 0,
          });
        }

        this._querySet = querySet;
        this._resolveBuffer = resolveBuffer;
        this._readbackSlots = readbackSlots;
        this._supported = true;
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
        this._supported = false;
      }
    }

    get supported() {
      return this._supported;
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

    get isDestroyed() {
      return this._destroyed;
    }

    /**
     * Resets per-frame pass metadata and query cursor for a new frame.
     */
    beginFrame() {
      if (!this._supported || this._destroyed) {
        return;
      }
      this._framePasses = [];
      this._queryCursor = 0;
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

      const beginningOfPassWriteIndex = this._queryCursor;
      const endOfPassWriteIndex = this._queryCursor + 1;
      this._queryCursor += 2;

      const name =
        typeof passName === 'string' && passName.length > 0
          ? passName
          : `pass_${this._framePasses.length}`;

      this._framePasses.push({
        name,
        beginIndex: beginningOfPassWriteIndex,
        endIndex: endOfPassWriteIndex,
      });

      return {
        querySet: this._querySet,
        beginningOfPassWriteIndex,
        endOfPassWriteIndex,
      };
    }

    /**
     * Encodes `resolveQuerySet` + `copyBufferToBuffer` into an available readback slot.
     * Skips cleanly (incrementing `skippedFrames`) if all readback slots are currently
     * in-flight or mapped so rendering never stalls.
     *
     * @param {GPUCommandEncoder} encoder
     * @returns {boolean} True if resolve commands were encoded, false if skipped or no-op.
     */
    resolveFrame(encoder) {
      if (!this._supported || this._destroyed || !this._querySet || !this._resolveBuffer) {
        return false;
      }
      const passCount = this._framePasses.length;
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

      let freeSlot = null;
      for (let i = 0; i < this._readbackSlots.length; i += 1) {
        if (isSlotAvailable(this._readbackSlots[i])) {
          freeSlot = this._readbackSlots[i];
          break;
        }
      }

      if (!freeSlot) {
        this._skippedFrames += 1;
        this._framePasses = [];
        this._queryCursor = 0;
        return false;
      }

      const queryCount = this._queryCursor;
      const copyBytes = queryCount * TIMESTAMP_BYTE_SIZE;

      encoder.resolveQuerySet(this._querySet, 0, queryCount, this._resolveBuffer, 0);
      encoder.copyBufferToBuffer(this._resolveBuffer, 0, freeSlot.buffer, 0, copyBytes);

      freeSlot.state = 'pending';
      freeSlot.passes = this._framePasses.slice();
      freeSlot.queryCount = queryCount;
      freeSlot.copyBytes = copyBytes;
      this._pendingQueue.push(freeSlot);

      this._framePasses = [];
      this._queryCursor = 0;
      return true;
    }

    _recordPassSample(name, durationMs) {
      let entry = this._passHistories.get(name);
      if (!entry) {
        entry = {
          samples: new Float64Array(this._historySize),
          writeIdx: 0,
          windowCount: 0,
          count: 0,
          lastMs: 0,
        };
        this._passHistories.set(name, entry);
      }

      entry.samples[entry.writeIdx] = durationMs;
      entry.writeIdx = (entry.writeIdx + 1) % this._historySize;
      if (entry.windowCount < this._historySize) {
        entry.windowCount += 1;
      }
      entry.count += 1;
      entry.lastMs = durationMs;
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
      if (this._pendingQueue.length === 0) {
        return null;
      }

      const slot = this._pendingQueue.shift();
      if (!slot || !slot.buffer || slot.state !== 'pending') {
        return null;
      }

      const buffer = slot.buffer;
      const recordedPasses = slot.passes || [];
      const queryCount = slot.queryCount;
      const copyBytes = slot.copyBytes;

      slot.state = 'mapping';

      let mapped = false;
      try {
        await buffer.mapAsync(MAP_MODE.READ);
        mapped = true;

        if (this._destroyed || slot.buffer !== buffer) {
          return null;
        }

        const range = buffer.getMappedRange(0, copyBytes);
        const timestamps = readTimestampsView(range, queryCount);

        let totalGpuMs = 0;
        const passes = new Array(recordedPasses.length);

        for (let i = 0; i < recordedPasses.length; i += 1) {
          const pass = recordedPasses[i];
          const startNs = pass.beginIndex < timestamps.length ? timestamps[pass.beginIndex] : 0n;
          const endNs = pass.endIndex < timestamps.length ? timestamps[pass.endIndex] : 0n;
          const durationMs = computeDurationMs(startNs, endNs);
          totalGpuMs += durationMs;
          passes[i] = { name: pass.name, durationMs };
          this._recordPassSample(pass.name, durationMs);
        }

        this._resolvedFrames += 1;
        this._lastFrameGpuMs = totalGpuMs;

        return { totalGpuMs, passes };
      } catch (_) {
        return null;
      } finally {
        if (mapped && !this._destroyed && typeof buffer.unmap === 'function') {
          try {
            buffer.unmap();
          } catch (_) {
            // ignore unmap errors if buffer was destroyed or device was lost
          }
        }
        if (!this._destroyed && slot.buffer === buffer) {
          slot.state = 'idle';
          slot.passes = null;
          slot.queryCount = 0;
          slot.copyBytes = 0;
        }
      }
    }

    /**
     * Returns a snapshot of overall profiler counters and per-pass rolling stats.
     *
     * @returns {{
     *   supported: boolean,
     *   resolvedFrames: number,
     *   skippedFrames: number,
     *   droppedPasses: number,
     *   lastFrameGpuMs: number,
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

      return {
        supported: this._supported,
        resolvedFrames: this._resolvedFrames,
        skippedFrames: this._skippedFrames,
        droppedPasses: this._droppedPasses,
        lastFrameGpuMs: this._lastFrameGpuMs,
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
        slot.passes = null;
        slot.queryCount = 0;
        slot.copyBytes = 0;
      }
      this._readbackSlots = [];
      this._pendingQueue = [];
      this._framePasses = [];
      this._queryCursor = 0;
      this._device = null;
    }
  }

  const exportedApi = {
    DEFAULT_MAX_PASSES_PER_FRAME,
    DEFAULT_READBACK_SLOTS,
    DEFAULT_HISTORY_SIZE,
    TIMESTAMP_BYTE_SIZE,
    BUFFER_USAGE,
    MAP_MODE,
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
