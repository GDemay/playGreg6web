'use strict';

// 256-byte-aligned WebGPU dynamic uniform buffer ring & per-frame single-flush draw constant batcher.
// Zero external dependencies; UMD export for Web Workers (globalThis.PlayGta6UniformRing)
// and Node.js (module.exports). Aligns cleanly with web/workers/wgpu_worker.js and
// web/workers/staging_pool.js.

(function (root) {
  const DEFAULT_UNIFORM_ALIGNMENT = 256;
  const DEFAULT_MIN_UNIFORM_BINDING_SIZE = 16;
  const DEFAULT_BYTE_SIZE = 1024 * 1024; // 1 MiB
  const DEFAULT_MAX_BYTE_SIZE = 16 * 1024 * 1024; // 16 MiB
  const DEFAULT_FRAMES_IN_FLIGHT = 3;
  const DEFAULT_MAX_BINDING_SIZE = 65536; // 64 KiB WebGPU default maxUniformBufferBindingSize
  const WEBGPU_BUFFER_COPY_ALIGNMENT = 4;

  // WebGPU GPUBufferUsage bitmask values with headless/Node fallbacks.
  const BUFFER_USAGE = Object.freeze({
    COPY_DST:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.COPY_DST === 'number'
        ? GPUBufferUsage.COPY_DST
        : 0x0008,
    UNIFORM:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.UNIFORM === 'number'
        ? GPUBufferUsage.UNIFORM
        : 0x0040,
  });

  // WebGPU GPUShaderStage bitmask values with headless/Node fallbacks.
  const SHADER_STAGE = Object.freeze({
    VERTEX:
      typeof GPUShaderStage !== 'undefined' && typeof GPUShaderStage.VERTEX === 'number'
        ? GPUShaderStage.VERTEX
        : 0x0001,
    FRAGMENT:
      typeof GPUShaderStage !== 'undefined' && typeof GPUShaderStage.FRAGMENT === 'number'
        ? GPUShaderStage.FRAGMENT
        : 0x0002,
    COMPUTE:
      typeof GPUShaderStage !== 'undefined' && typeof GPUShaderStage.COMPUTE === 'number'
        ? GPUShaderStage.COMPUTE
        : 0x0004,
  });

  const DEFAULT_SHADER_VISIBILITY = SHADER_STAGE.VERTEX | SHADER_STAGE.FRAGMENT;

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

  function isPowerOfTwo(value) {
    if (!Number.isInteger(value) || value < 1) {
      return false;
    }
    let v = value;
    while (v > 1) {
      if (v % 2 !== 0) {
        return false;
      }
      v /= 2;
    }
    return true;
  }

  function nextPowerOfTwo(value) {
    let p = WEBGPU_BUFFER_COPY_ALIGNMENT;
    while (p < value) {
      p *= 2;
    }
    return p;
  }

  function isArrayBuffer(value) {
    if (value instanceof ArrayBuffer) {
      return true;
    }
    return Object.prototype.toString.call(value) === '[object ArrayBuffer]';
  }

  function isSharedArrayBuffer(value) {
    if (typeof SharedArrayBuffer !== 'undefined' && value instanceof SharedArrayBuffer) {
      return true;
    }
    return Object.prototype.toString.call(value) === '[object SharedArrayBuffer]';
  }

  /**
   * Align a non-negative integer up to the nearest multiple of `alignment`.
   * Safe across the full IEEE-754 safe integer range.
   *
   * @param {number} value
   * @param {number} alignment
   * @returns {number}
   */
  function alignUp(value, alignment) {
    assertNonNegativeInteger(value, 'value');
    assertPositiveInteger(alignment, 'alignment');
    const remainder = value % alignment;
    return remainder === 0 ? value : value + (alignment - remainder);
  }

  /**
   * Pad a uniform slice byte length to a 4-byte multiple and guarantee at least
   * `minBindingSize` bytes (default 16B) so empty or tiny CBVs satisfy WebGPU
   * bind-group validation.
   *
   * @param {number} byteLength
   * @param {number} [minBindingSize=16]
   * @returns {number}
   */
  function alignUniformSliceSize(byteLength, minBindingSize = DEFAULT_MIN_UNIFORM_BINDING_SIZE) {
    assertNonNegativeInteger(byteLength, 'byteLength');
    assertPositiveInteger(minBindingSize, 'minBindingSize');
    const minAligned = Math.max(
      DEFAULT_MIN_UNIFORM_BINDING_SIZE,
      alignUp(minBindingSize, WEBGPU_BUFFER_COPY_ALIGNMENT)
    );
    const aligned4 = alignUp(byteLength, WEBGPU_BUFFER_COPY_ALIGNMENT);
    return Math.max(minAligned, aligned4);
  }

  function toUint8Array(input, label) {
    if (input instanceof Uint8Array) {
      return input;
    }
    if (ArrayBuffer.isView(input)) {
      return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
    }
    if (isArrayBuffer(input) || isSharedArrayBuffer(input)) {
      return new Uint8Array(input);
    }
    throw new TypeError(
      `${label} expects a Uint8Array, ArrayBufferView, ArrayBuffer, or SharedArrayBuffer`
    );
  }

  /**
   * Prepare a byte view for `GPUQueue.writeBuffer`:
   * - Always copies `SharedArrayBuffer`-backed data into a regular `ArrayBuffer`-backed
   *   `Uint8Array` (Chrome WebGPU rejects `SharedArrayBuffer` directly in `writeBuffer`).
   * - Copies non-4-byte-aligned views (odd `byteOffset` or `byteLength`) into a fresh
   *   zero-padded, 4-byte-aligned `Uint8Array`.
   * - Zero-copy fast path when already a 4-byte-aligned regular `ArrayBuffer` `Uint8Array`.
   *
   * @param {ArrayBufferView | ArrayBuffer | SharedArrayBuffer} input
   * @param {number} [minBytes=0]
   * @returns {{ bytes: Uint8Array, copiedFromSab: boolean, copiedForAlignment: boolean }}
   */
  function copyForWriteBuffer(input, minBytes = 0) {
    assertNonNegativeInteger(minBytes, 'minBytes');
    const view = toUint8Array(input, 'copyForWriteBuffer');
    const fromSab = isSharedArrayBuffer(view.buffer);
    const targetLen = Math.max(
      alignUp(view.byteLength, WEBGPU_BUFFER_COPY_ALIGNMENT),
      alignUp(minBytes, WEBGPU_BUFFER_COPY_ALIGNMENT)
    );
    const unalignedOffset = view.byteOffset % WEBGPU_BUFFER_COPY_ALIGNMENT !== 0;
    const unalignedLength = view.byteLength !== targetLen;
    const needsAlignCopy = unalignedOffset || unalignedLength;

    if (fromSab || needsAlignCopy) {
      const copy = new Uint8Array(targetLen);
      if (view.byteLength > 0) {
        copy.set(view, 0);
      }
      return {
        bytes: copy,
        copiedFromSab: fromSab,
        copiedForAlignment: needsAlignCopy,
      };
    }

    return {
      bytes: view,
      copiedFromSab: false,
      copiedForAlignment: false,
    };
  }

  function resolveDeviceUniformAlignment(device) {
    const raw =
      device &&
      device.limits &&
      typeof device.limits.minUniformBufferOffsetAlignment === 'number' &&
      Number.isFinite(device.limits.minUniformBufferOffsetAlignment) &&
      device.limits.minUniformBufferOffsetAlignment > 0
        ? Math.floor(device.limits.minUniformBufferOffsetAlignment)
        : 0;

    if (raw < WEBGPU_BUFFER_COPY_ALIGNMENT) {
      return DEFAULT_UNIFORM_ALIGNMENT;
    }
    return isPowerOfTwo(raw) ? raw : nextPowerOfTwo(raw);
  }

  function resolveDeviceMaxBindingSize(device) {
    const raw =
      device &&
      device.limits &&
      typeof device.limits.maxUniformBufferBindingSize === 'number' &&
      Number.isInteger(device.limits.maxUniformBufferBindingSize) &&
      device.limits.maxUniformBufferBindingSize > 0
        ? device.limits.maxUniformBufferBindingSize
        : 0;
    return raw > 0 ? raw : DEFAULT_MAX_BINDING_SIZE;
  }

  /**
   * Insert `[start, end)` into a sorted, disjoint array of `{ start, end, writeCount }`
   * dirty byte ranges, coalescing adjacent/overlapping intervals (and intervals separated
   * by at most `maxGapBytes`).
   *
   * O(1) zero-allocation fast path for sequential appends (the dominant per-frame draw path).
   *
   * @param {Array<{ start: number, end: number, writeCount: number }>} ranges
   * @param {number} start
   * @param {number} end
   * @param {number} [writeCount=1]
   * @param {number} [maxGapBytes=0]
   */
  function insertDirtyRange(ranges, start, end, writeCount = 1, maxGapBytes = 0) {
    if (end <= start) {
      return;
    }

    const len = ranges.length;
    if (len === 0) {
      ranges.push({ start, end, writeCount });
      return;
    }

    // O(1) fast path: sequential append or overlap with the last range.
    const last = ranges[len - 1];
    if (start >= last.start) {
      if (start <= last.end + maxGapBytes) {
        if (end > last.end) {
          last.end = end;
        }
        last.writeCount += writeCount;
        return;
      }
      ranges.push({ start, end, writeCount });
      return;
    }

    // General sorted insertion + coalescing for wrap-around or out-of-order staged writes.
    let insertIdx = 0;
    while (insertIdx < len && ranges[insertIdx].end + maxGapBytes < start) {
      insertIdx += 1;
    }

    let mergedStart = start;
    let mergedEnd = end;
    let mergedWrites = writeCount;
    let mergeEndIdx = insertIdx;

    while (mergeEndIdx < len && ranges[mergeEndIdx].start <= mergedEnd + maxGapBytes) {
      const existing = ranges[mergeEndIdx];
      if (existing.start < mergedStart) {
        mergedStart = existing.start;
      }
      if (existing.end > mergedEnd) {
        mergedEnd = existing.end;
      }
      mergedWrites += existing.writeCount;
      mergeEndIdx += 1;
    }

    const merged = {
      start: mergedStart,
      end: mergedEnd,
      writeCount: mergedWrites,
    };

    if (mergeEndIdx === insertIdx) {
      ranges.splice(insertIdx, 0, merged);
    } else {
      ranges.splice(insertIdx, mergeEndIdx - insertIdx, merged);
    }
  }

  /**
   * Pure helper: sort and coalesce an array of `{ start, end, writeCount? }` byte ranges,
   * merging adjacent/overlapping intervals (and optionally bridging gaps <= `maxGapBytes`).
   *
   * @param {Array<{ start: number, end: number, writeCount?: number }>} inputRanges
   * @param {number} [maxGapBytes=0]
   * @returns {Array<{ start: number, end: number, byteLength: number, writeCount: number }>}
   */
  function coalesceDirtyRanges(inputRanges, maxGapBytes = 0) {
    if (!Array.isArray(inputRanges)) {
      throw new TypeError('coalesceDirtyRanges expects an array of { start, end } ranges');
    }
    assertNonNegativeInteger(maxGapBytes, 'maxGapBytes');
    const merged = [];
    for (let i = 0; i < inputRanges.length; i += 1) {
      const r = inputRanges[i];
      if (!r || typeof r !== 'object') {
        throw new TypeError(`Dirty range at index ${i} must be an object`);
      }
      assertNonNegativeInteger(r.start, 'range.start');
      assertNonNegativeInteger(r.end, 'range.end');
      if (r.end < r.start) {
        throw new RangeError(`Dirty range end (${r.end}) must be >= start (${r.start})`);
      }
      const writes = r.writeCount === undefined ? 1 : r.writeCount;
      assertPositiveInteger(writes, 'range.writeCount');
      insertDirtyRange(merged, r.start, r.end, writes, maxGapBytes);
    }
    return merged.map((r) => ({
      start: r.start,
      end: r.end,
      byteLength: r.end - r.start,
      writeCount: r.writeCount,
    }));
  }

  /**
   * Multi-buffered WebGPU dynamic uniform buffer ring for per-draw D3D12 root constants / CBVs.
   *
   * Accumulates per-draw uniform payloads into an aligned CPU shadow buffer and uploads the
   * dirty `[0, usedBytes)` prefix in a single `device.queue.writeBuffer(...)` call on `flush()`.
   * Rotates across `framesInFlight` dedicated `GPUBuffer` slots so frame N+1 never overwrites
   * uniform data still being read by the GPU for frame N.
   */
  class DynamicUniformRing {
    constructor(options = {}) {
      if (!options || typeof options !== 'object') {
        throw new TypeError('DynamicUniformRing constructor expects an options object');
      }

      const {
        device,
        byteSize,
        capacityBytes,
        alignment,
        minUniformBufferOffsetAlignment,
        maxByteSize,
        maxCapacityBytes,
        framesInFlight = DEFAULT_FRAMES_IN_FLIGHT,
        maxBindingSize,
        minUniformBindingSize = DEFAULT_MIN_UNIFORM_BINDING_SIZE,
        allowEmptyPayloads = false,
        allowOverflow = false,
        allowSpill,
        maxRingBytes,
        multiSlice,
        preserveBuffersOnGrow,
        allowWrap = false,
        wrapOnOverflow = false,
        immediateFlush = false,
        maxCoalesceGapBytes = 0,
      } = options;

      assertNonNegativeInteger(maxCoalesceGapBytes, 'maxCoalesceGapBytes');

      if (!device || typeof device.createBuffer !== 'function') {
        throw new TypeError('DynamicUniformRing requires a WebGPU device with createBuffer()');
      }

      const deviceAlignment = resolveDeviceUniformAlignment(device);
      const rawAlignment =
        alignment !== undefined ? alignment : minUniformBufferOffsetAlignment;
      let resolvedAlignment;
      if (rawAlignment === undefined) {
        resolvedAlignment = deviceAlignment;
      } else {
        if (!isPowerOfTwo(rawAlignment) || rawAlignment < WEBGPU_BUFFER_COPY_ALIGNMENT) {
          throw new RangeError(
            `alignment must be a power-of-two integer >= ${WEBGPU_BUFFER_COPY_ALIGNMENT}, got ${rawAlignment}`
          );
        }
        // Never allow dynamic offsets below the device's minUniformBufferOffsetAlignment.
        resolvedAlignment = Math.max(rawAlignment, deviceAlignment);
      }

      const resolvedMaxBindingSize =
        maxBindingSize === undefined ? resolveDeviceMaxBindingSize(device) : maxBindingSize;

      const rawMaxByteSize =
        maxByteSize !== undefined ? maxByteSize : maxCapacityBytes;
      const hasExplicitMaxByteSize = rawMaxByteSize !== undefined;
      const hasExplicitMaxRingBytes = maxRingBytes !== undefined;
      const rawByteSize = byteSize !== undefined ? byteSize : capacityBytes;
      const resolvedMultiSlice =
        multiSlice !== undefined
          ? Boolean(multiSlice)
          : preserveBuffersOnGrow !== undefined
            ? Boolean(preserveBuffersOnGrow)
            : false;

      if (hasExplicitMaxRingBytes) {
        assertPositiveInteger(maxRingBytes, 'maxRingBytes');
        if (maxRingBytes < resolvedAlignment) {
          throw new RangeError(
            `maxRingBytes (${maxRingBytes}) must be at least alignment (${resolvedAlignment})`
          );
        }
      }

      assertPositiveInteger(framesInFlight, 'framesInFlight');
      assertPositiveInteger(resolvedMaxBindingSize, 'maxBindingSize');
      assertPositiveInteger(minUniformBindingSize, 'minUniformBindingSize');

      const resolvedMinUniformBindingSize = Math.max(
        DEFAULT_MIN_UNIFORM_BINDING_SIZE,
        alignUp(minUniformBindingSize, WEBGPU_BUFFER_COPY_ALIGNMENT)
      );

      if (resolvedMinUniformBindingSize > resolvedMaxBindingSize) {
        throw new RangeError(
          `minUniformBindingSize (${resolvedMinUniformBindingSize}) exceeds maxBindingSize (${resolvedMaxBindingSize})`
        );
      }

      const spillRequestedByOptions =
        allowSpill === true || (allowSpill === undefined && hasExplicitMaxRingBytes);

      let resolvedMaxByteSize;
      if (hasExplicitMaxByteSize) {
        assertPositiveInteger(rawMaxByteSize, 'maxByteSize');
        if (rawMaxByteSize < resolvedAlignment) {
          throw new RangeError(
            `maxByteSize (${rawMaxByteSize}) must be at least alignment (${resolvedAlignment})`
          );
        }
        resolvedMaxByteSize = rawMaxByteSize;
      } else if (spillRequestedByOptions) {
        resolvedMaxByteSize = null;
      } else if (hasExplicitMaxRingBytes) {
        resolvedMaxByteSize = Math.min(DEFAULT_MAX_BYTE_SIZE, maxRingBytes);
      } else {
        resolvedMaxByteSize = DEFAULT_MAX_BYTE_SIZE;
      }

      let resolvedByteSize;
      if (rawByteSize === undefined) {
        const capBasis =
          resolvedMaxByteSize !== null
            ? resolvedMaxByteSize
            : hasExplicitMaxRingBytes
              ? maxRingBytes
              : DEFAULT_MAX_BYTE_SIZE;
        const clampedMax =
          Math.floor(capBasis / resolvedAlignment) * resolvedAlignment;
        resolvedByteSize = Math.min(
          alignUp(DEFAULT_BYTE_SIZE, resolvedAlignment),
          clampedMax
        );
      } else {
        assertPositiveInteger(rawByteSize, 'byteSize');
        resolvedByteSize = alignUp(rawByteSize, resolvedAlignment);
        if (resolvedMaxByteSize !== null && resolvedByteSize > resolvedMaxByteSize) {
          throw new RangeError(
            `Aligned byteSize (${resolvedByteSize}) exceeds maxByteSize (${resolvedMaxByteSize})`
          );
        }
      }

      if (resolvedMaxByteSize === null) {
        resolvedMaxByteSize = resolvedByteSize;
      }

      if (hasExplicitMaxRingBytes) {
        if (maxRingBytes < resolvedByteSize) {
          throw new RangeError(
            `maxRingBytes (${maxRingBytes}) must be at least aligned byteSize (${resolvedByteSize})`
          );
        }
        if (hasExplicitMaxByteSize && maxRingBytes < resolvedMaxByteSize) {
          throw new RangeError(
            `maxRingBytes (${maxRingBytes}) must be at least maxByteSize (${resolvedMaxByteSize})`
          );
        }
      }

      let resolvedAllowSpill;
      if (allowSpill !== undefined) {
        resolvedAllowSpill = Boolean(allowSpill);
      } else if (hasExplicitMaxRingBytes && maxRingBytes > resolvedMaxByteSize) {
        resolvedAllowSpill = true;
      } else {
        resolvedAllowSpill = false;
      }

      let resolvedMaxRingBytes;
      if (hasExplicitMaxRingBytes) {
        resolvedMaxRingBytes = maxRingBytes;
      } else if (resolvedAllowSpill) {
        resolvedMaxRingBytes = Math.max(DEFAULT_MAX_BYTE_SIZE, resolvedMaxByteSize);
      } else if (allowOverflow) {
        resolvedMaxRingBytes = null;
      } else {
        resolvedMaxRingBytes = resolvedMaxByteSize;
      }

      this.device = device;
      this.alignment = resolvedAlignment;
      this.deviceMinUniformBufferOffsetAlignment = deviceAlignment;
      this.maxByteSize = resolvedMaxByteSize;
      this.maxCapacityBytes = resolvedMaxByteSize;
      this.framesInFlight = framesInFlight;
      this.maxBindingSize = resolvedMaxBindingSize;
      this.minUniformBindingSize = resolvedMinUniformBindingSize;
      this.allowEmptyPayloads = Boolean(allowEmptyPayloads);
      this.allowOverflow = Boolean(allowOverflow);
      this.allowSpill = resolvedAllowSpill;
      this.maxRingBytes = resolvedMaxRingBytes;
      this.multiSlice = resolvedMultiSlice;
      this.allowWrap = Boolean(allowWrap || wrapOnOverflow);
      this.immediateFlush = Boolean(immediateFlush);
      this.maxCoalesceGapBytes = maxCoalesceGapBytes;

      this._capacity = resolvedByteSize;
      this._shadow = new Uint8Array(resolvedByteSize);
      this._dirtyRanges = [];
      this._slots = [];
      this._destroyed = false;

      this._frameIndex = -1;
      this._slotIndex = 0;
      this._cursor = 0;
      this._writtenHighWater = 0;
      this._framePrimaryBytes = 0;
      this._flushedCursor = 0;
      this._flushedDrawCount = 0;
      this._hasFlushedActiveSlot = false;
      this._primaryFrameDrawCount = 0;
      this._frameSliceBytes = 0;
      this._activeSliceIndex = 0;
      this._frameOverflowBytes = 0;
      this._batchDepth = 0;
      this._dirty = false;

      // Telemetry counters.
      this._allocatedGpuBytes = 0;
      this._peakAllocatedGpuBytes = 0;
      this._peakUsedBytes = 0;
      this._peakFrameBytes = 0;
      this._frameDrawCount = 0;
      this._totalDrawCount = 0;
      this._totalBytesPushed = 0;
      this._totalBytesUploaded = 0;
      this._alignedBytesWasted = 0;
      this._flushCount = 0;
      this._writeBufferCalls = 0;
      this._coalescedWrites = 0;
      this._stagedWritesFlushed = 0;
      this._growCount = 0;
      this._overflowAllocCount = 0;
      this._spillAllocations = 0;
      this._spillPagesCreated = 0;
      this._sabCopyCount = 0;
      this._unalignedCopyCount = 0;
      this._wrapCount = 0;

      for (let i = 0; i < framesInFlight; i += 1) {
        const buffer = this._createSlotBuffer(i, resolvedByteSize);
        this._slots.push({
          index: i,
          buffer,
          byteSize: resolvedByteSize,
          generation: 1,
          used: false,
          retiredBuffers: [],
          retiredSlices: [],
          overflowChunks: [],
        });
      }
    }

    get currentBuffer() {
      if (this._destroyed || this._slots.length === 0) {
        return null;
      }
      return this._slots[this._slotIndex].buffer;
    }

    get capacity() {
      return this._capacity;
    }

    get capacityBytes() {
      return this._capacity;
    }

    get currentCapacityBytes() {
      return this._capacity;
    }

    get usedBytes() {
      return this._frameSliceBytes + this._framePrimaryBytes + this._frameOverflowBytes;
    }

    get cursor() {
      return this._cursor;
    }

    get batchDepth() {
      return this._batchDepth;
    }

    get isBatchActive() {
      return this._batchDepth > 0;
    }

    get slotIndex() {
      return this._slotIndex;
    }

    get frameIndex() {
      return this._frameIndex;
    }

    _assertNotDestroyed() {
      if (this._destroyed) {
        throw new Error('DynamicUniformRing has been destroyed');
      }
    }

    _createGpuUniformBuffer(label, byteSize) {
      const buffer = this.device.createBuffer({
        label,
        size: byteSize,
        usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST,
      });
      this._allocatedGpuBytes += byteSize;
      if (this._allocatedGpuBytes > this._peakAllocatedGpuBytes) {
        this._peakAllocatedGpuBytes = this._allocatedGpuBytes;
      }
      return buffer;
    }

    _createSlotBuffer(slotIndex, byteSize) {
      return this._createGpuUniformBuffer(`playgta6-uniform-ring-slot-${slotIndex}`, byteSize);
    }

    _destroyRawBuffer(buffer, byteSize) {
      if (!buffer) {
        return;
      }
      this._allocatedGpuBytes = Math.max(0, this._allocatedGpuBytes - byteSize);
      if (typeof buffer.destroy === 'function') {
        buffer.destroy();
      }
    }

    _destroySlotBuffer(slot) {
      if (!slot || !slot.buffer) {
        return;
      }
      const oldBuffer = slot.buffer;
      const oldSize = slot.byteSize;
      slot.buffer = null;
      slot.byteSize = 0;
      this._destroyRawBuffer(oldBuffer, oldSize);
    }

    _releaseSlotTransientBuffers(slot) {
      if (!slot) {
        return;
      }
      if (slot.retiredBuffers && slot.retiredBuffers.length > 0) {
        for (let i = 0; i < slot.retiredBuffers.length; i += 1) {
          const entry = slot.retiredBuffers[i];
          this._destroyRawBuffer(entry.buffer, entry.byteSize);
        }
        slot.retiredBuffers.length = 0;
      }
      if (slot.retiredSlices && slot.retiredSlices.length > 0) {
        for (let i = 0; i < slot.retiredSlices.length; i += 1) {
          const slice = slot.retiredSlices[i];
          this._destroyRawBuffer(slice.buffer, slice.byteSize);
          slice.buffer = null;
          slice.shadow = null;
        }
        slot.retiredSlices.length = 0;
      }
      if (slot.overflowChunks && slot.overflowChunks.length > 0) {
        for (let i = 0; i < slot.overflowChunks.length; i += 1) {
          const chunk = slot.overflowChunks[i];
          this._destroyRawBuffer(chunk.buffer, chunk.byteSize);
          chunk.buffer = null;
          chunk.shadow = null;
        }
        slot.overflowChunks.length = 0;
      }
    }

    _resizeSlot(slot, targetByteSize, deferOldDestroy = false) {
      if (slot.byteSize === targetByteSize && slot.buffer) {
        return;
      }
      if (deferOldDestroy && slot.buffer) {
        // Keep previously flushed buffer alive until this slot is next reset by beginFrame()
        // or destroyed, preventing WebGPU use-after-destroy for earlier draws in the frame.
        slot.retiredBuffers.push({
          buffer: slot.buffer,
          byteSize: slot.byteSize,
        });
        slot.buffer = null;
        slot.byteSize = 0;
      } else {
        this._destroySlotBuffer(slot);
      }
      slot.buffer = this._createSlotBuffer(slot.index, targetByteSize);
      slot.byteSize = targetByteSize;
      slot.generation += 1;
    }

    _grow(requiredBytes) {
      if (requiredBytes > this.maxByteSize) {
        throw new RangeError(
          `DynamicUniformRing capacity exceeded: need ${requiredBytes} bytes, ` +
            `maxByteSize is ${this.maxByteSize} bytes`
        );
      }

      let nextCapacity = this._capacity;
      while (nextCapacity < requiredBytes) {
        nextCapacity *= 2;
      }
      if (nextCapacity > this.maxByteSize) {
        const alignedMax =
          Math.floor(this.maxByteSize / this.alignment) * this.alignment;
        const copyAlignedMax =
          Math.floor(this.maxByteSize / WEBGPU_BUFFER_COPY_ALIGNMENT) *
          WEBGPU_BUFFER_COPY_ALIGNMENT;
        nextCapacity = Math.max(requiredBytes, alignedMax >= requiredBytes ? alignedMax : copyAlignedMax);
      }

      const nextShadow = new Uint8Array(nextCapacity);
      const copyLen = Math.max(this._cursor, this._writtenHighWater);
      if (copyLen > 0) {
        nextShadow.set(this._shadow.subarray(0, copyLen), 0);
      }
      this._shadow = nextShadow;
      this._capacity = nextCapacity;

      // Resize only the active slot now; other slots may still be in flight on the GPU
      // and will resize cleanly when beginFrame() rotates to them.
      // If draws were already flushed into the active slot's previous buffer in this frame,
      // retain the old buffer until the slot rotates back in a later frame.
      const activeSlot = this._slots[this._slotIndex];
      const deferOldDestroy = this._flushedCursor > 0 || this._hasFlushedActiveSlot;
      this._resizeSlot(activeSlot, nextCapacity, deferOldDestroy);

      // The active slot now has a fresh GPUBuffer, so any prior partial flush in this frame
      // must be re-uploaded from offset 0 on the next flush().
      this._dirtyRanges = [];
      if (copyLen > 0) {
        insertDirtyRange(
          this._dirtyRanges,
          0,
          copyLen,
          Math.max(1, this._primaryFrameDrawCount),
          this.maxCoalesceGapBytes
        );
      }
      this._flushedCursor = 0;
      this._flushedDrawCount = 0;
      this._hasFlushedActiveSlot = false;
      this._growCount += 1;
      this._overflowAllocCount += 1;
    }

    _getSlotRetiredSliceBytes(slot) {
      let bytes = 0;
      if (slot && slot.retiredSlices) {
        for (let i = 0; i < slot.retiredSlices.length; i += 1) {
          bytes += slot.retiredSlices[i].byteSize;
        }
      }
      return bytes;
    }

    _getSlotPrimaryAndRetiredBytes(slot) {
      if (!slot) {
        return 0;
      }
      return slot.byteSize + this._getSlotRetiredSliceBytes(slot);
    }

    _getSlotSpillBytes(slot) {
      let bytes = 0;
      if (slot && slot.overflowChunks) {
        for (let i = 0; i < slot.overflowChunks.length; i += 1) {
          bytes += slot.overflowChunks[i].byteSize;
        }
      }
      return bytes;
    }

    _getSlotLiveBytes(slot) {
      if (!slot) {
        return 0;
      }
      return (
        slot.byteSize +
        this._getSlotRetiredSliceBytes(slot) +
        this._getSlotSpillBytes(slot)
      );
    }

    _recordFramePeak() {
      const totalUsed = this._frameSliceBytes + this._framePrimaryBytes + this._frameOverflowBytes;
      if (totalUsed > this._peakUsedBytes) {
        this._peakUsedBytes = totalUsed;
      }
      if (totalUsed > this._peakFrameBytes) {
        this._peakFrameBytes = totalUsed;
      }
    }

    /**
     * Non-destructive multi-slice dynamic growth for heavy single-pass frames:
     * preserves the currently-filled slice GPUBuffer (and its dirty CPU shadow range)
     * in `slot.retiredSlices` so earlier draw allocations bound to `alloc.buffer` remain
     * intact and get flushed before GPU submission, then allocates a fresh geometrically
     * larger slice bounded by `maxCapacityBytes` (`maxByteSize`).
     *
     * Returns `false` (without mutating state) if the remaining budget under `maxByteSize`
     * cannot accommodate `alignedSize`.
     *
     * @param {object} slot
     * @param {number} alignedSize
     * @returns {boolean}
     */
    _growMultiSlice(slot, alignedSize) {
      const alignedMax =
        Math.floor(this.maxByteSize / this.alignment) * this.alignment;
      const retiredBytes = this._getSlotRetiredSliceBytes(slot);

      // Fast path when the active slice has zero allocations yet (e.g., first draw of
      // the frame is larger than the current slice size): resize in place cleanly.
      if (this._cursor === 0) {
        const remaining = this.maxByteSize - retiredBytes;
        const maxAlignedRemaining =
          remaining > 0 ? Math.floor(remaining / this.alignment) * this.alignment : 0;
        if (maxAlignedRemaining < alignedSize) {
          return false;
        }
        let nextSize = Math.max(slot.byteSize, this.alignment);
        while (nextSize < alignedSize) {
          nextSize *= 2;
        }
        nextSize = Math.min(alignUp(nextSize, this.alignment), maxAlignedRemaining);
        this._resizeSlot(slot, nextSize, false);
        this._shadow = new Uint8Array(nextSize);
        if (nextSize > this._capacity) {
          this._capacity = Math.min(alignedMax, nextSize);
        }
        this._growCount += 1;
        this._overflowAllocCount += 1;
        return true;
      }

      const committedBytes = retiredBytes + slot.byteSize;
      const remaining = this.maxByteSize - committedBytes;
      const maxAlignedRemaining =
        remaining > 0 ? Math.floor(remaining / this.alignment) * this.alignment : 0;
      if (maxAlignedRemaining < alignedSize) {
        return false;
      }

      let nextSliceSize = Math.max(slot.byteSize * 2, alignedSize);
      nextSliceSize = alignUp(nextSliceSize, this.alignment);
      if (nextSliceSize > maxAlignedRemaining) {
        nextSliceSize = maxAlignedRemaining;
      }

      const sliceDirty = this._dirtyRanges.length > 0;
      slot.retiredSlices.push({
        buffer: slot.buffer,
        byteSize: slot.byteSize,
        shadow: sliceDirty ? this._shadow : null,
        dirtyRanges: sliceDirty ? this._dirtyRanges : [],
        cursor: this._cursor,
        flushedCursor: this._flushedCursor,
        drawCount: this._primaryFrameDrawCount,
        flushedDrawCount: this._flushedDrawCount,
        dirty: sliceDirty,
      });

      this._frameSliceBytes += this._framePrimaryBytes;
      this._activeSliceIndex += 1;

      const nextBuffer = this._createGpuUniformBuffer(
        `playgta6-uniform-ring-slot-${slot.index}-slice-${this._activeSliceIndex}`,
        nextSliceSize
      );
      slot.buffer = nextBuffer;
      slot.byteSize = nextSliceSize;
      slot.generation += 1;

      this._shadow = new Uint8Array(nextSliceSize);
      this._dirtyRanges = [];
      this._cursor = 0;
      this._writtenHighWater = 0;
      this._framePrimaryBytes = 0;
      this._flushedCursor = 0;
      this._flushedDrawCount = 0;
      this._primaryFrameDrawCount = 0;
      this._hasFlushedActiveSlot = false;

      // Raise steady-state per-slot capacity target so subsequent frames fit in a single buffer.
      const totalFrameCapacity = this._getSlotPrimaryAndRetiredBytes(slot);
      let targetCap = this._capacity;
      while (targetCap < totalFrameCapacity && targetCap < alignedMax) {
        targetCap *= 2;
      }
      targetCap = Math.min(alignedMax, Math.max(targetCap, totalFrameCapacity));
      if (targetCap > this._capacity) {
        this._capacity = targetCap;
      }

      this._growCount += 1;
      this._overflowAllocCount += 1;
      return true;
    }

    _allocateOverflowSlice(slot, alignedSize) {
      let chunk =
        slot.overflowChunks.length > 0
          ? slot.overflowChunks[slot.overflowChunks.length - 1]
          : null;

      if (!chunk || alignUp(chunk.cursor, this.alignment) + alignedSize > chunk.byteSize) {
        const chunkIndex = slot.overflowChunks.length;
        const desiredPageSize = alignUp(
          Math.max(alignedSize, this._capacity),
          this.alignment
        );
        let chunkByteSize = desiredPageSize;
        if (this.maxRingBytes !== null) {
          const liveBytes = this._getSlotLiveBytes(slot);
          const remainingBytes = this.maxRingBytes - liveBytes;
          const maxAlignedRemaining =
            remainingBytes > 0
              ? Math.floor(remainingBytes / this.alignment) * this.alignment
              : 0;
          if (maxAlignedRemaining < alignedSize) {
            throw new RangeError(
              `DynamicUniformRing maxRingBytes budget exceeded: need ${liveBytes + alignedSize} bytes ` +
                `(live ${liveBytes} + ${alignedSize}), maxRingBytes is ${this.maxRingBytes} bytes`
            );
          }
          chunkByteSize = Math.min(desiredPageSize, maxAlignedRemaining);
        }

        const buffer = this._createGpuUniformBuffer(
          `playgta6-uniform-ring-slot-${slot.index}-overflow-${chunkIndex}`,
          chunkByteSize
        );
        chunk = {
          buffer,
          byteSize: chunkByteSize,
          shadow: new Uint8Array(chunkByteSize),
          dirtyRanges: [],
          cursor: 0,
          flushedCursor: 0,
          drawCount: 0,
          flushedDrawCount: 0,
          dirty: false,
        };
        slot.overflowChunks.push(chunk);
        this._overflowAllocCount += 1;
        this._spillPagesCreated += 1;
      }

      const dynamicOffset = alignUp(chunk.cursor, this.alignment);
      return { chunk, dynamicOffset };
    }

    _safeWriteBuffer(buffer, bufferOffset, data) {
      if (!this.device.queue || typeof this.device.queue.writeBuffer !== 'function') {
        throw new TypeError(
          'DynamicUniformRing.flush() requires device.queue.writeBuffer()'
        );
      }
      const prepared = copyForWriteBuffer(data);
      if (prepared.copiedFromSab) {
        this._sabCopyCount += 1;
      }
      if (prepared.copiedForAlignment) {
        this._unalignedCopyCount += 1;
      }
      this.device.queue.writeBuffer(buffer, bufferOffset, prepared.bytes);
      return prepared.bytes.byteLength;
    }

    _countLiveOverflowBuffers() {
      let count = 0;
      for (let i = 0; i < this._slots.length; i += 1) {
        const slot = this._slots[i];
        count +=
          slot.overflowChunks.length +
          slot.retiredBuffers.length +
          slot.retiredSlices.length;
      }
      return count;
    }

    _countLiveRetiredSlices() {
      let count = 0;
      for (let i = 0; i < this._slots.length; i += 1) {
        count += this._slots[i].retiredSlices.length;
      }
      return count;
    }

    _countLiveSpillPages() {
      let count = 0;
      for (let i = 0; i < this._slots.length; i += 1) {
        count += this._slots[i].overflowChunks.length;
      }
      return count;
    }

    _countLiveSpillBytes() {
      let bytes = 0;
      for (let i = 0; i < this._slots.length; i += 1) {
        bytes += this._getSlotSpillBytes(this._slots[i]);
      }
      return bytes;
    }

    _countPendingDirtyRanges() {
      let count = this._dirtyRanges ? this._dirtyRanges.length : 0;
      const slot =
        !this._destroyed && this._slots.length > 0
          ? this._slots[this._slotIndex]
          : null;
      if (slot) {
        for (let i = 0; i < slot.retiredSlices.length; i += 1) {
          const slice = slot.retiredSlices[i];
          if (slice.dirty && slice.dirtyRanges) {
            count += slice.dirtyRanges.length;
          }
        }
        for (let i = 0; i < slot.overflowChunks.length; i += 1) {
          const chunk = slot.overflowChunks[i];
          if (chunk.dirty && chunk.dirtyRanges) {
            count += chunk.dirtyRanges.length;
          }
        }
      }
      return count;
    }

    /**
     * Pad a uniform slice byte length to a 4-byte multiple and at least
     * `this.minUniformBindingSize` (default 16B).
     *
     * @param {number} byteLength
     * @returns {number}
     */
    normalizeSliceSize(byteLength) {
      return alignUniformSliceSize(byteLength, this.minUniformBindingSize);
    }

    /**
     * Begin a new frame: rotate to the next frame-in-flight slot, release transient
     * overflow/retired buffers from that recycled slot, resize the slot's GPUBuffer if
     * the ring grew on an earlier frame, and reset the write cursor to 0.
     *
     * @param {number} [frameIndex]
     * @returns {{ frameIndex: number, slotIndex: number, buffer: GPUBuffer, byteSize: number, generation: number }}
     */
    beginFrame(frameIndex) {
      this._assertNotDestroyed();

      this._recordFramePeak();

      let nextFrame;
      if (frameIndex === undefined) {
        nextFrame = this._frameIndex + 1;
      } else {
        assertNonNegativeInteger(frameIndex, 'frameIndex');
        nextFrame = frameIndex;
      }

      this._frameIndex = nextFrame;
      this._slotIndex = nextFrame % this.framesInFlight;
      this._cursor = 0;
      this._writtenHighWater = 0;
      this._framePrimaryBytes = 0;
      this._flushedCursor = 0;
      this._flushedDrawCount = 0;
      this._hasFlushedActiveSlot = false;
      this._primaryFrameDrawCount = 0;
      this._frameSliceBytes = 0;
      this._activeSliceIndex = 0;
      this._frameOverflowBytes = 0;
      this._frameDrawCount = 0;
      this._batchDepth = 0;
      this._dirtyRanges = [];
      this._dirty = false;

      const slot = this._slots[this._slotIndex];
      if (slot.used) {
        this._wrapCount += 1;
      }
      slot.used = true;

      // This slot is now safe to recycle: destroy any overflow/retired buffers from its
      // previous frame while leaving all other in-flight slots untouched.
      this._releaseSlotTransientBuffers(slot);

      if (slot.byteSize < this._capacity) {
        this._resizeSlot(slot, this._capacity, false);
      }
      if (!this._shadow || this._shadow.byteLength !== slot.byteSize) {
        this._shadow = new Uint8Array(slot.byteSize);
      }

      return {
        frameIndex: this._frameIndex,
        slotIndex: this._slotIndex,
        buffer: slot.buffer,
        byteSize: slot.byteSize,
        generation: slot.generation,
      };
    }

    /**
     * Begin a coalesced write batch. Multiple `pushConstants()`, `pushCbv()`, or
     * `writeStaged()` calls inside a batch accumulate in the CPU shadow staging slab and
     * merge adjacent/overlapping dirty ranges so `endBatch()`, `flush()`, or `flushDirty()`
     * emits the minimum number of `device.queue.writeBuffer()` calls.
     *
     * Supports nesting: nested `beginBatch()` / `endBatch()` pairs defer auto-flushing
     * until the outermost batch completes.
     *
     * @returns {{ batchDepth: number, frameIndex: number, slotIndex: number, cursor: number }}
     */
    beginBatch() {
      this._assertNotDestroyed();
      if (this._frameIndex < 0) {
        this._frameIndex = 0;
        this._slots[this._slotIndex].used = true;
      }
      this._batchDepth += 1;
      return {
        batchDepth: this._batchDepth,
        frameIndex: this._frameIndex,
        slotIndex: this._slotIndex,
        cursor: this._cursor,
      };
    }

    /**
     * End the current write batch. When the outermost batch completes (`batchDepth` returns
     * to 0), flushes all coalesced dirty ranges in the CPU shadow staging slab unless
     * `{ autoFlush: false }` is passed.
     *
     * @param {{ autoFlush?: boolean }} [options]
     * @returns {{ uploadedBytes: number, bytesUploaded: number, drawCount: number, stagedWrites: number, writeBufferCalls: number, coalescedWrites: number, dirtyRangesFlushed: number, batchDepth: number }}
     */
    endBatch(options = {}) {
      this._assertNotDestroyed();
      if (this._batchDepth === 0) {
        return {
          uploadedBytes: 0,
          bytesUploaded: 0,
          drawCount: 0,
          stagedWrites: 0,
          writeBufferCalls: 0,
          coalescedWrites: 0,
          dirtyRangesFlushed: 0,
          batchDepth: 0,
        };
      }
      this._batchDepth -= 1;
      const shouldFlush = !(options && options.autoFlush === false);
      if (this._batchDepth === 0 && shouldFlush) {
        const res = this._flushDirtyInternal();
        return {
          uploadedBytes: res.uploadedBytes,
          bytesUploaded: res.bytesUploaded,
          drawCount: res.drawCount,
          stagedWrites: res.stagedWrites,
          writeBufferCalls: res.writeBufferCalls,
          coalescedWrites: res.coalescedWrites,
          dirtyRangesFlushed: res.dirtyRangesFlushed,
          batchDepth: 0,
        };
      }
      return {
        uploadedBytes: 0,
        bytesUploaded: 0,
        drawCount: 0,
        stagedWrites: 0,
        writeBufferCalls: 0,
        coalescedWrites: 0,
        dirtyRangesFlushed: 0,
        batchDepth: this._batchDepth,
      };
    }

    /**
     * Return a defensive snapshot of currently pending coalesced dirty byte ranges in the
     * active slot's CPU shadow staging slab.
     *
     * @returns {Array<{ start: number, end: number, byteLength: number, writeCount: number }>}
     */
    getDirtyRanges() {
      this._assertNotDestroyed();
      return this._dirtyRanges.map((r) => ({
        start: r.start,
        end: r.end,
        byteLength: r.end - r.start,
        writeCount: r.writeCount,
      }));
    }

    /**
     * Append a per-draw constant payload into the CPU shadow ring aligned to `alignment`
     * (at least `device.limits.minUniformBufferOffsetAlignment`, default 256B).
     * Auto-grows capacity (doubling up to `maxByteSize`) when needed, and optionally
     * allocates transient per-frame overflow buffers when `allowOverflow` is enabled.
     *
     * @param {ArrayBufferView | ArrayBuffer | SharedArrayBuffer | null | undefined} data
     * @param {{ allowEmpty?: boolean }} [pushOptions]
     * @returns {{ buffer: GPUBuffer, dynamicOffset: number, size: number, paddedSize: number, alignedSize: number, overflow: boolean }}
     */
    pushConstants(data, pushOptions) {
      this._assertNotDestroyed();

      const allowEmpty =
        (pushOptions && pushOptions.allowEmpty === true) || this.allowEmptyPayloads;
      const allowWrap =
        (pushOptions && pushOptions.wrap === true) || this.allowWrap;
      const deferImmediateFlush = Boolean(
        pushOptions && pushOptions._deferImmediateFlush === true
      );

      let src;
      if ((data === null || data === undefined) && allowEmpty) {
        src = new Uint8Array(0);
      } else {
        src = toUint8Array(data, 'pushConstants');
      }

      const size = src.byteLength;
      if (size === 0 && !allowEmpty) {
        throw new RangeError('pushConstants data must not be empty (got 0 bytes)');
      }
      if (size > this.maxBindingSize) {
        throw new RangeError(
          `pushConstants size (${size} bytes) exceeds maxBindingSize (${this.maxBindingSize} bytes)`
        );
      }

      const paddedSize = Math.max(
        this.minUniformBindingSize,
        alignUp(size, WEBGPU_BUFFER_COPY_ALIGNMENT)
      );
      if (paddedSize > this.maxBindingSize) {
        throw new RangeError(
          `pushConstants padded size (${paddedSize} bytes) exceeds maxBindingSize (${this.maxBindingSize} bytes)`
        );
      }

      const alignedSize = alignUp(paddedSize, this.alignment);

      if (this._frameIndex < 0) {
        this._frameIndex = 0;
        this._slots[this._slotIndex].used = true;
      }

      const slot = this._slots[this._slotIndex];
      const fromSab = size > 0 && isSharedArrayBuffer(src.buffer);
      const isUnalignedInput =
        size > 0 &&
        (src.byteOffset % WEBGPU_BUFFER_COPY_ALIGNMENT !== 0 ||
          size % WEBGPU_BUFFER_COPY_ALIGNMENT !== 0);

      // Copy SAB-backed source bytes out of shared memory into a regular ArrayBuffer
      // snapshot before writing into the ring shadow buffer.
      let uploadBytes = src;
      if (fromSab) {
        const snapshot = new Uint8Array(size);
        snapshot.set(src, 0);
        uploadBytes = snapshot;
      }

      const primaryDynamicOffset = alignUp(this._cursor, this.alignment);
      const primaryEndOffset = primaryDynamicOffset + alignedSize;

      let targetBuffer;
      let dynamicOffset;
      let isOverflow = false;
      let wrapped = false;
      let allocationSliceIndex = this._activeSliceIndex;
      let wastedBytes = 0;

      if (primaryEndOffset <= slot.byteSize) {
        wastedBytes = (primaryDynamicOffset - this._cursor) + (alignedSize - size);
        this._shadow.set(uploadBytes, primaryDynamicOffset);
        if (alignedSize > size) {
          this._shadow.fill(0, primaryDynamicOffset + size, primaryEndOffset);
        }
        insertDirtyRange(
          this._dirtyRanges,
          primaryDynamicOffset,
          primaryEndOffset,
          1,
          this.maxCoalesceGapBytes
        );
        this._cursor = primaryEndOffset;
        if (primaryEndOffset > this._writtenHighWater) {
          this._writtenHighWater = primaryEndOffset;
        }
        this._framePrimaryBytes += alignedSize;
        this._primaryFrameDrawCount += 1;
        targetBuffer = slot.buffer;
        dynamicOffset = primaryDynamicOffset;
      } else if (allowWrap && alignedSize <= slot.byteSize) {
        // Wrap cursor cleanly to offset 0 while preserving 256-byte alignment and
        // retaining any unflushed tail dirty range for coalesced multi-range flush.
        wastedBytes = (slot.byteSize - this._cursor) + (alignedSize - size);
        this._shadow.set(uploadBytes, 0);
        if (alignedSize > size) {
          this._shadow.fill(0, size, alignedSize);
        }
        insertDirtyRange(
          this._dirtyRanges,
          0,
          alignedSize,
          1,
          this.maxCoalesceGapBytes
        );
        this._cursor = alignedSize;
        if (alignedSize > this._writtenHighWater) {
          this._writtenHighWater = alignedSize;
        }
        this._framePrimaryBytes += alignedSize;
        this._primaryFrameDrawCount += 1;
        this._wrapCount += 1;
        targetBuffer = slot.buffer;
        dynamicOffset = 0;
        wrapped = true;
      } else if (this.multiSlice && this._growMultiSlice(slot, alignedSize)) {
        wastedBytes = alignedSize - size;
        this._shadow.set(uploadBytes, 0);
        if (alignedSize > size) {
          this._shadow.fill(0, size, alignedSize);
        }
        insertDirtyRange(
          this._dirtyRanges,
          0,
          alignedSize,
          1,
          this.maxCoalesceGapBytes
        );
        this._cursor = alignedSize;
        if (alignedSize > this._writtenHighWater) {
          this._writtenHighWater = alignedSize;
        }
        this._framePrimaryBytes += alignedSize;
        this._primaryFrameDrawCount += 1;
        targetBuffer = slot.buffer;
        dynamicOffset = 0;
        allocationSliceIndex = this._activeSliceIndex;
      } else if (!this.multiSlice && primaryEndOffset <= this.maxByteSize) {
        wastedBytes = (primaryDynamicOffset - this._cursor) + (alignedSize - size);
        this._grow(primaryEndOffset);
        this._shadow.set(uploadBytes, primaryDynamicOffset);
        if (alignedSize > size) {
          this._shadow.fill(0, primaryDynamicOffset + size, primaryEndOffset);
        }
        insertDirtyRange(
          this._dirtyRanges,
          primaryDynamicOffset,
          primaryEndOffset,
          1,
          this.maxCoalesceGapBytes
        );
        this._cursor = primaryEndOffset;
        if (primaryEndOffset > this._writtenHighWater) {
          this._writtenHighWater = primaryEndOffset;
        }
        this._framePrimaryBytes += alignedSize;
        this._primaryFrameDrawCount += 1;
        targetBuffer = slot.buffer;
        dynamicOffset = primaryDynamicOffset;
      } else if (allowWrap && !this.multiSlice && alignedSize <= this.maxByteSize) {
        wastedBytes = alignedSize - size;
        this._grow(alignedSize);
        this._shadow.set(uploadBytes, 0);
        if (alignedSize > size) {
          this._shadow.fill(0, size, alignedSize);
        }
        insertDirtyRange(
          this._dirtyRanges,
          0,
          alignedSize,
          1,
          this.maxCoalesceGapBytes
        );
        this._cursor = alignedSize;
        if (alignedSize > this._writtenHighWater) {
          this._writtenHighWater = alignedSize;
        }
        this._framePrimaryBytes += alignedSize;
        this._primaryFrameDrawCount += 1;
        this._wrapCount += 1;
        targetBuffer = slot.buffer;
        dynamicOffset = 0;
        wrapped = true;
      } else if (this.allowSpill || this.allowOverflow) {
        const overflowAlloc = this._allocateOverflowSlice(slot, alignedSize);
        const chunk = overflowAlloc.chunk;
        const prevChunkCursor = chunk.cursor;
        dynamicOffset = overflowAlloc.dynamicOffset;
        const chunkEndOffset = dynamicOffset + alignedSize;
        wastedBytes = (dynamicOffset - prevChunkCursor) + (alignedSize - size);

        chunk.shadow.set(uploadBytes, dynamicOffset);
        if (alignedSize > size) {
          chunk.shadow.fill(0, dynamicOffset + size, chunkEndOffset);
        }
        insertDirtyRange(
          chunk.dirtyRanges,
          dynamicOffset,
          chunkEndOffset,
          1,
          this.maxCoalesceGapBytes
        );
        chunk.cursor = chunkEndOffset;
        chunk.drawCount += 1;
        chunk.dirty = true;
        this._frameOverflowBytes += alignedSize;
        this._spillAllocations += 1;

        targetBuffer = chunk.buffer;
        isOverflow = true;
        allocationSliceIndex = this._activeSliceIndex + slot.overflowChunks.length;
      } else {
        const neededBytes = this.multiSlice
          ? this._getSlotPrimaryAndRetiredBytes(slot) + alignedSize
          : primaryEndOffset;
        throw new RangeError(
          `DynamicUniformRing capacity exceeded: need ${neededBytes} bytes, ` +
            `maxByteSize is ${this.maxByteSize} bytes`
        );
      }

      if (fromSab) {
        this._sabCopyCount += 1;
      }
      if (isUnalignedInput) {
        this._unalignedCopyCount += 1;
      }

      this._dirty = true;
      this._frameDrawCount += 1;
      this._totalDrawCount += 1;
      this._totalBytesPushed += size;
      this._alignedBytesWasted += wastedBytes;
      this._recordFramePeak();

      if (this.immediateFlush && this._batchDepth === 0 && !deferImmediateFlush) {
        this._flushDirtyInternal();
      }

      return {
        buffer: targetBuffer,
        offset: dynamicOffset,
        dynamicOffset,
        size,
        paddedSize,
        alignedSize,
        overflow: isOverflow,
        isSpill: isOverflow,
        wrapped,
        sliceIndex: allocationSliceIndex,
      };
    }

    /**
     * Stage uniform constant bytes into the CPU shadow slab and record their dirty byte
     * range for coalesced upload on `flush()` / `flushDirty()` / `endBatch()`.
     *
     * Supports two call forms:
     * 1. `writeStaged(data, options?)` — allocates the next `alignment`-aligned (default 256B)
     *    slice in the ring (wrapping to offset 0 when `allowWrap` or `options.wrap` is true).
     * 2. `writeStaged(offset, data, options?)` — writes directly at an explicit
     *    `alignment`-aligned (or 4B-aligned when `options.allowSubAlignment` is true) byte
     *    offset in the active slot's CPU shadow slab.
     *
     * Unlike `pushConstants()` under `immediateFlush: true`, `writeStaged()` never auto-flushes
     * before `flushDirty()` / `flush()` / `endBatch()`.
     *
     * @param {number | ArrayBufferView | ArrayBuffer | SharedArrayBuffer | null | undefined} offsetOrData
     * @param {ArrayBufferView | ArrayBuffer | SharedArrayBuffer | object | null | undefined} [dataOrOptions]
     * @param {{ allowEmpty?: boolean, wrap?: boolean, allowSubAlignment?: boolean }} [maybeOptions]
     */
    writeStaged(offsetOrData, dataOrOptions, maybeOptions) {
      this._assertNotDestroyed();

      if (typeof offsetOrData !== 'number') {
        const pushOpts =
          dataOrOptions && typeof dataOrOptions === 'object'
            ? Object.assign({}, dataOrOptions, { _deferImmediateFlush: true })
            : { _deferImmediateFlush: true };
        const alloc = this.pushConstants(offsetOrData, pushOpts);
        alloc.staged = true;
        return alloc;
      }

      const offset = offsetOrData;
      const data = dataOrOptions;
      const options = maybeOptions && typeof maybeOptions === 'object' ? maybeOptions : {};
      assertNonNegativeInteger(offset, 'offset');

      const allowSubAlignment = options.allowSubAlignment === true;
      const requiredOffsetAlign = allowSubAlignment
        ? WEBGPU_BUFFER_COPY_ALIGNMENT
        : this.alignment;
      if (offset % requiredOffsetAlign !== 0) {
        throw new RangeError(
          `writeStaged offset (${offset}) must be aligned to ${requiredOffsetAlign} bytes`
        );
      }

      const allowEmpty =
        options.allowEmpty === true || this.allowEmptyPayloads;

      let src;
      if ((data === null || data === undefined) && allowEmpty) {
        src = new Uint8Array(0);
      } else {
        src = toUint8Array(data, 'writeStaged');
      }

      const size = src.byteLength;
      if (size === 0 && !allowEmpty) {
        throw new RangeError('writeStaged data must not be empty (got 0 bytes)');
      }
      if (size > this.maxBindingSize) {
        throw new RangeError(
          `writeStaged size (${size} bytes) exceeds maxBindingSize (${this.maxBindingSize} bytes)`
        );
      }

      const paddedSize = Math.max(
        this.minUniformBindingSize,
        alignUp(size, WEBGPU_BUFFER_COPY_ALIGNMENT)
      );
      if (paddedSize > this.maxBindingSize) {
        throw new RangeError(
          `writeStaged padded size (${paddedSize} bytes) exceeds maxBindingSize (${this.maxBindingSize} bytes)`
        );
      }

      const alignedSize = alignUp(paddedSize, this.alignment);
      const writeSpan = allowSubAlignment ? paddedSize : alignedSize;

      if (this._frameIndex < 0) {
        this._frameIndex = 0;
        this._slots[this._slotIndex].used = true;
      }

      const slot = this._slots[this._slotIndex];
      const endOffset = offset + writeSpan;
      if (endOffset > slot.byteSize) {
        throw new RangeError(
          `writeStaged range [${offset}, ${endOffset}) exceeds active slot capacity (${slot.byteSize} bytes)`
        );
      }

      const fromSab = size > 0 && isSharedArrayBuffer(src.buffer);
      const isUnalignedInput =
        size > 0 &&
        (src.byteOffset % WEBGPU_BUFFER_COPY_ALIGNMENT !== 0 ||
          size % WEBGPU_BUFFER_COPY_ALIGNMENT !== 0);

      let uploadBytes = src;
      if (fromSab) {
        const snapshot = new Uint8Array(size);
        snapshot.set(src, 0);
        uploadBytes = snapshot;
      }

      this._shadow.set(uploadBytes, offset);
      if (writeSpan > size) {
        this._shadow.fill(0, offset + size, endOffset);
      }

      insertDirtyRange(
        this._dirtyRanges,
        offset,
        endOffset,
        1,
        this.maxCoalesceGapBytes
      );

      if (endOffset > this._cursor) {
        this._cursor = endOffset;
      }
      if (endOffset > this._writtenHighWater) {
        this._writtenHighWater = endOffset;
      }
      this._framePrimaryBytes += writeSpan;
      this._primaryFrameDrawCount += 1;

      if (fromSab) {
        this._sabCopyCount += 1;
      }
      if (isUnalignedInput) {
        this._unalignedCopyCount += 1;
      }

      this._dirty = true;
      this._frameDrawCount += 1;
      this._totalDrawCount += 1;
      this._totalBytesPushed += size;
      this._alignedBytesWasted += writeSpan - size;
      this._recordFramePeak();

      return {
        buffer: slot.buffer,
        offset,
        dynamicOffset: offset,
        size,
        paddedSize,
        alignedSize,
        overflow: false,
        isSpill: false,
        wrapped: false,
        staged: true,
        sliceIndex: this._activeSliceIndex,
      };
    }

    /**
     * Convenience wrapper for CBV uploads (including empty or <16B CBVs), automatically
     * padding to a 4-byte multiple of at least `minUniformBindingSize` (16B) and
     * aligning dynamic offsets to `alignment` (default 256B).
     *
     * @param {ArrayBufferView | ArrayBuffer | SharedArrayBuffer | null | undefined} data
     * @returns {{ buffer: GPUBuffer, dynamicOffset: number, size: number, paddedSize: number, alignedSize: number, overflow: boolean }}
     */
    pushCbv(data) {
      return this.pushConstants(data, { allowEmpty: true });
    }

    _flushDirtyInternal() {
      this._assertNotDestroyed();

      const slot = this._slots[this._slotIndex];
      let hasRetiredSlicesDirty = false;
      for (let i = 0; i < slot.retiredSlices.length; i += 1) {
        const slice = slot.retiredSlices[i];
        if (slice.dirty && slice.dirtyRanges && slice.dirtyRanges.length > 0) {
          hasRetiredSlicesDirty = true;
          break;
        }
      }
      const hasPrimaryDirty = this._dirtyRanges.length > 0;
      let hasOverflowDirty = false;
      for (let i = 0; i < slot.overflowChunks.length; i += 1) {
        const chunk = slot.overflowChunks[i];
        if (chunk.dirty && chunk.dirtyRanges && chunk.dirtyRanges.length > 0) {
          hasOverflowDirty = true;
          break;
        }
      }

      if (!this._dirty || (!hasRetiredSlicesDirty && !hasPrimaryDirty && !hasOverflowDirty)) {
        return {
          uploadedBytes: 0,
          bytesUploaded: 0,
          drawCount: 0,
          stagedWrites: 0,
          writeBufferCalls: 0,
          coalescedWrites: 0,
          dirtyRangesFlushed: 0,
        };
      }

      if (!this.device.queue || typeof this.device.queue.writeBuffer !== 'function') {
        throw new TypeError(
          'DynamicUniformRing.flush() requires device.queue.writeBuffer()'
        );
      }

      let uploadedBytes = 0;
      let stagedWrites = 0;
      let writeBufferCalls = 0;
      let coalescedWrites = 0;

      for (let i = 0; i < slot.retiredSlices.length; i += 1) {
        const slice = slot.retiredSlices[i];
        if (!slice.dirty || !slice.dirtyRanges || slice.dirtyRanges.length === 0) {
          continue;
        }
        for (let rIdx = 0; rIdx < slice.dirtyRanges.length; rIdx += 1) {
          const range = slice.dirtyRanges[rIdx];
          const rangeBytes = range.end - range.start;
          const sliceView = slice.shadow.subarray(range.start, range.end);

          this._safeWriteBuffer(slice.buffer, range.start, sliceView);

          writeBufferCalls += 1;
          stagedWrites += range.writeCount;
          coalescedWrites += Math.max(0, range.writeCount - 1);
          uploadedBytes += rangeBytes;
        }
        slice.dirtyRanges.length = 0;
        slice.flushedCursor = slice.cursor;
        slice.flushedDrawCount = slice.drawCount;
        slice.dirty = false;
        slice.shadow = null;
      }

      if (hasPrimaryDirty) {
        for (let rIdx = 0; rIdx < this._dirtyRanges.length; rIdx += 1) {
          const range = this._dirtyRanges[rIdx];
          const rangeBytes = range.end - range.start;
          const shadowSlice = this._shadow.subarray(range.start, range.end);

          this._safeWriteBuffer(slot.buffer, range.start, shadowSlice);

          writeBufferCalls += 1;
          stagedWrites += range.writeCount;
          coalescedWrites += Math.max(0, range.writeCount - 1);
          uploadedBytes += rangeBytes;
        }
        this._dirtyRanges.length = 0;
        this._flushedCursor = this._cursor;
        this._flushedDrawCount = this._primaryFrameDrawCount;
        this._hasFlushedActiveSlot = true;
      }

      for (let i = 0; i < slot.overflowChunks.length; i += 1) {
        const chunk = slot.overflowChunks[i];
        if (!chunk.dirty || !chunk.dirtyRanges || chunk.dirtyRanges.length === 0) {
          continue;
        }
        for (let rIdx = 0; rIdx < chunk.dirtyRanges.length; rIdx += 1) {
          const range = chunk.dirtyRanges[rIdx];
          const rangeBytes = range.end - range.start;
          const chunkSlice = chunk.shadow.subarray(range.start, range.end);

          this._safeWriteBuffer(chunk.buffer, range.start, chunkSlice);

          writeBufferCalls += 1;
          stagedWrites += range.writeCount;
          coalescedWrites += Math.max(0, range.writeCount - 1);
          uploadedBytes += rangeBytes;
        }
        chunk.dirtyRanges.length = 0;
        chunk.flushedCursor = chunk.cursor;
        chunk.flushedDrawCount = chunk.drawCount;
        chunk.dirty = false;
      }

      this._dirty = false;
      this._flushCount += 1;
      this._totalBytesUploaded += uploadedBytes;
      this._writeBufferCalls += writeBufferCalls;
      this._coalescedWrites += coalescedWrites;
      this._stagedWritesFlushed += stagedWrites;

      return {
        uploadedBytes,
        bytesUploaded: uploadedBytes,
        drawCount: stagedWrites,
        stagedWrites,
        writeBufferCalls,
        coalescedWrites,
        dirtyRangesFlushed: writeBufferCalls,
      };
    }

    /**
     * Upload all coalesced dirty ranges in the CPU shadow staging slab (plus any dirty
     * retired slices or per-frame overflow buffers) to GPU memory using the minimum number
     * of regular `ArrayBuffer`-backed, 4-byte-aligned `device.queue.writeBuffer(...)` calls,
     * returning full coalescing telemetry for this flush.
     *
     * @returns {{ uploadedBytes: number, bytesUploaded: number, drawCount: number, stagedWrites: number, writeBufferCalls: number, coalescedWrites: number, dirtyRangesFlushed: number }}
     */
    flushDirty() {
      return this._flushDirtyInternal();
    }

    /**
     * Upload all coalesced dirty CPU shadow ranges for the current frame (plus any dirty
     * retired slices or per-frame overflow buffers) to GPU memory using regular
     * `ArrayBuffer`-backed, 4-byte-aligned `device.queue.writeBuffer(...)` calls.
     *
     * Preserves the legacy `{ uploadedBytes, drawCount }` return contract for existing
     * callers; call `flushDirty()` or `getStats()` for full coalescing stats.
     *
     * @returns {{ uploadedBytes: number, drawCount: number }}
     */
    flush() {
      const res = this._flushDirtyInternal();
      return {
        uploadedBytes: res.uploadedBytes,
        drawCount: res.drawCount,
      };
    }

    /**
     * Reclaim and destroy auxiliary spill-page GPUBuffers across all slots (or a single
     * specified `slotIndex`). Safe and idempotent to call on quiet frames or after GPU work
     * completes.
     *
     * @param {number} [slotIndex]
     * @returns {{ reclaimedPages: number, reclaimedBytes: number }}
     */
    reclaimSpillPages(slotIndex) {
      this._assertNotDestroyed();
      if (slotIndex !== undefined) {
        assertNonNegativeInteger(slotIndex, 'slotIndex');
        if (slotIndex >= this.framesInFlight) {
          throw new RangeError(
            `slotIndex (${slotIndex}) out of range [0, ${this.framesInFlight - 1}]`
          );
        }
      }

      let reclaimedPages = 0;
      let reclaimedBytes = 0;
      const start = slotIndex !== undefined ? slotIndex : 0;
      const end = slotIndex !== undefined ? slotIndex + 1 : this._slots.length;

      for (let i = start; i < end; i += 1) {
        const slot = this._slots[i];
        if (!slot || !slot.overflowChunks || slot.overflowChunks.length === 0) {
          continue;
        }
        for (let j = 0; j < slot.overflowChunks.length; j += 1) {
          const chunk = slot.overflowChunks[j];
          reclaimedPages += 1;
          reclaimedBytes += chunk.byteSize;
          this._destroyRawBuffer(chunk.buffer, chunk.byteSize);
          chunk.buffer = null;
          chunk.shadow = null;
        }
        slot.overflowChunks.length = 0;
      }

      const activeSlot = this._slots[this._slotIndex];
      let activeHasDirtySpill = false;
      if (activeSlot && activeSlot.overflowChunks) {
        for (let i = 0; i < activeSlot.overflowChunks.length; i += 1) {
          const chunk = activeSlot.overflowChunks[i];
          if (chunk.dirty && chunk.dirtyRanges && chunk.dirtyRanges.length > 0) {
            activeHasDirtySpill = true;
            break;
          }
        }
      }
      let activeHasDirtyRetiredSlice = false;
      if (activeSlot && activeSlot.retiredSlices) {
        for (let i = 0; i < activeSlot.retiredSlices.length; i += 1) {
          const slice = activeSlot.retiredSlices[i];
          if (slice.dirty && slice.dirtyRanges && slice.dirtyRanges.length > 0) {
            activeHasDirtyRetiredSlice = true;
            break;
          }
        }
      }
      if (
        this._dirtyRanges.length === 0 &&
        !activeHasDirtySpill &&
        !activeHasDirtyRetiredSlice
      ) {
        this._dirty = false;
      }

      return { reclaimedPages, reclaimedBytes };
    }

    /**
     * Complete the active frame, updating peak frame utilization telemetry and reclaiming
     * the active slot's spill pages unless `{ deferReclaim: true }` or
     * `{ reclaimSpillPages: false }` is passed.
     *
     * @param {{ deferReclaim?: boolean, reclaimSpillPages?: boolean }} [options]
     * @returns {{ frameIndex: number, slotIndex: number, frameBytes: number, reclaimedPages: number, reclaimedBytes: number }}
     */
    endFrame(options = {}) {
      this._assertNotDestroyed();
      this._recordFramePeak();

      const frameBytes = this._frameSliceBytes + this._framePrimaryBytes + this._frameOverflowBytes;
      const shouldReclaim =
        !(options && (options.deferReclaim === true || options.reclaimSpillPages === false));

      let reclaimedPages = 0;
      let reclaimedBytes = 0;
      if (shouldReclaim) {
        const res = this.reclaimSpillPages(this._slotIndex);
        reclaimedPages = res.reclaimedPages;
        reclaimedBytes = res.reclaimedBytes;
      }

      return {
        frameIndex: this._frameIndex,
        slotIndex: this._slotIndex,
        frameBytes,
        reclaimedPages,
        reclaimedBytes,
      };
    }

    /**
     * Reset per-frame write state and destroy all live spill pages and retired buffers across
     * all slots while preserving primary slot GPUBuffers and lifetime telemetry counters.
     *
     * @returns {{ reclaimedPages: number, reclaimedBytes: number }}
     */
    reset() {
      this._assertNotDestroyed();
      this._recordFramePeak();

      const { reclaimedPages, reclaimedBytes } = this.reclaimSpillPages();
      for (let i = 0; i < this._slots.length; i += 1) {
        this._releaseSlotTransientBuffers(this._slots[i]);
      }

      this._cursor = 0;
      this._writtenHighWater = 0;
      this._framePrimaryBytes = 0;
      this._flushedCursor = 0;
      this._flushedDrawCount = 0;
      this._hasFlushedActiveSlot = false;
      this._primaryFrameDrawCount = 0;
      this._frameSliceBytes = 0;
      this._activeSliceIndex = 0;
      this._frameOverflowBytes = 0;
      this._frameDrawCount = 0;
      this._batchDepth = 0;
      this._dirtyRanges = [];
      this._dirty = false;

      return { reclaimedPages, reclaimedBytes };
    }

    /**
     * Build a WebGPU `GPUBindGroupLayoutEntry` for this dynamic uniform buffer
     * (`type: 'uniform'`, `hasDynamicOffset: true`).
     *
     * @param {number} binding
     * @param {number} [visibility]
     * @param {number} [minBindingSize=0]
     */
    createBindGroupLayoutEntry(
      binding,
      visibility = DEFAULT_SHADER_VISIBILITY,
      minBindingSize = 0
    ) {
      this._assertNotDestroyed();
      assertNonNegativeInteger(binding, 'binding');
      assertPositiveInteger(visibility, 'visibility');
      assertNonNegativeInteger(minBindingSize, 'minBindingSize');
      if (minBindingSize > this.maxBindingSize) {
        throw new RangeError(
          `minBindingSize (${minBindingSize}) exceeds maxBindingSize (${this.maxBindingSize})`
        );
      }

      return {
        binding,
        visibility,
        buffer: {
          type: 'uniform',
          hasDynamicOffset: true,
          minBindingSize,
        },
      };
    }

    /**
     * Build a WebGPU `GPUBindGroupEntry` referencing the active (or specified) frame slot's
     * uniform buffer at `offset: 0` with a 4-byte-aligned dynamic window size of at least
     * `minUniformBindingSize` (16B).
     *
     * @param {number} binding
     * @param {number} [sliceSize]
     * @param {number} [slotIndex]
     */
    createBindGroupEntry(binding, sliceSize, slotIndex = this._slotIndex) {
      this._assertNotDestroyed();
      assertNonNegativeInteger(binding, 'binding');
      assertNonNegativeInteger(slotIndex, 'slotIndex');
      if (slotIndex >= this.framesInFlight) {
        throw new RangeError(
          `slotIndex (${slotIndex}) out of range [0, ${this.framesInFlight - 1}]`
        );
      }

      const slot = this._slots[slotIndex];
      let resolvedSize;
      if (sliceSize === undefined) {
        resolvedSize = Math.min(this.maxBindingSize, slot.byteSize);
      } else {
        assertPositiveInteger(sliceSize, 'sliceSize');
        if (sliceSize > this.maxBindingSize) {
          throw new RangeError(
            `sliceSize (${sliceSize}) exceeds maxBindingSize (${this.maxBindingSize})`
          );
        }
        if (sliceSize > slot.byteSize) {
          throw new RangeError(
            `sliceSize (${sliceSize}) exceeds slot buffer size (${slot.byteSize})`
          );
        }
        resolvedSize = Math.max(
          this.minUniformBindingSize,
          alignUp(sliceSize, WEBGPU_BUFFER_COPY_ALIGNMENT)
        );
        if (resolvedSize > this.maxBindingSize) {
          throw new RangeError(
            `Padded sliceSize (${resolvedSize}) exceeds maxBindingSize (${this.maxBindingSize})`
          );
        }
        if (resolvedSize > slot.byteSize) {
          throw new RangeError(
            `Padded sliceSize (${resolvedSize}) exceeds slot buffer size (${slot.byteSize})`
          );
        }
      }

      return {
        binding,
        resource: {
          buffer: slot.buffer,
          offset: 0,
          size: resolvedSize,
        },
      };
    }

    /**
     * Build a WebGPU `GPUBindGroupEntry` directly for a `pushConstants()` / `pushCbv()`
     * allocation (works transparently for both primary ring slots and overflow buffers).
     *
     * @param {number} binding
     * @param {{ buffer: GPUBuffer, size?: number, paddedSize?: number }} allocation
     */
    createBindGroupEntryForAllocation(binding, allocation) {
      this._assertNotDestroyed();
      assertNonNegativeInteger(binding, 'binding');
      if (!allocation || typeof allocation !== 'object' || !allocation.buffer) {
        throw new TypeError(
          'createBindGroupEntryForAllocation expects an allocation with a GPUBuffer'
        );
      }
      const rawSize =
        allocation.paddedSize !== undefined
          ? allocation.paddedSize
          : allocation.size !== undefined
            ? allocation.size
            : 0;
      const resolvedSize = this.normalizeSliceSize(rawSize);
      if (resolvedSize > this.maxBindingSize) {
        throw new RangeError(
          `Allocation binding size (${resolvedSize}) exceeds maxBindingSize (${this.maxBindingSize})`
        );
      }
      return {
        binding,
        resource: {
          buffer: allocation.buffer,
          offset: 0,
          size: resolvedSize,
        },
      };
    }

    /**
     * Return ring configuration, usage, alignment, SAB copy, and overflow diagnostics.
     */
    getStats() {
      const activeSlot =
        !this._destroyed && this._slots.length > 0
          ? this._slots[this._slotIndex]
          : null;
      return {
        frameIndex: this._frameIndex,
        slotIndex: this._slotIndex,
        framesInFlight: this.framesInFlight,
        alignment: this.alignment,
        minUniformBindingSize: this.minUniformBindingSize,
        byteSize: this._capacity,
        currentCapacityBytes: this._capacity,
        maxByteSize: this.maxByteSize,
        maxCapacityBytes: this.maxByteSize,
        maxRingBytes: this.maxRingBytes,
        maxBindingSize: this.maxBindingSize,
        allowOverflow: this.allowOverflow,
        allowSpill: this.allowSpill,
        multiSlice: this.multiSlice,
        allowWrap: this.allowWrap,
        immediateFlush: this.immediateFlush,
        maxCoalesceGapBytes: this.maxCoalesceGapBytes,
        usedBytes: this._frameSliceBytes + this._framePrimaryBytes + this._frameOverflowBytes,
        peakUsedBytes: this._peakUsedBytes,
        peakFrameBytes: this._peakFrameBytes,
        peakUtilizationRatio: this._capacity > 0 ? this._peakFrameBytes / this._capacity : 0,
        allocatedGpuBytes: this._allocatedGpuBytes,
        peakAllocatedGpuBytes: this._peakAllocatedGpuBytes,
        frameDrawCount: this._frameDrawCount,
        totalDrawCount: this._totalDrawCount,
        totalAllocations: this._totalDrawCount,
        totalBytesPushed: this._totalBytesPushed,
        totalBytesUploaded: this._totalBytesUploaded,
        bytesUploaded: this._totalBytesUploaded,
        alignedBytesWasted: this._alignedBytesWasted,
        flushCount: this._flushCount,
        writeBufferCalls: this._writeBufferCalls,
        coalescedWrites: this._coalescedWrites,
        stagedWritesFlushed: this._stagedWritesFlushed,
        dirtyRangeCount: this._destroyed ? 0 : this._countPendingDirtyRanges(),
        batchDepth: this._batchDepth,
        batchActive: this._batchDepth > 0,
        growCount: this._growCount,
        growthCount: this._growCount,
        overflowAllocCount: this._overflowAllocCount,
        overflowBufferCount: this._destroyed ? 0 : this._countLiveOverflowBuffers(),
        liveRetiredSlices: this._destroyed ? 0 : this._countLiveRetiredSlices(),
        spillAllocations: this._spillAllocations,
        spillPagesCreated: this._spillPagesCreated,
        liveSpillPages: this._destroyed ? 0 : this._countLiveSpillPages(),
        liveSpillBytes: this._destroyed ? 0 : this._countLiveSpillBytes(),
        sabCopyCount: this._sabCopyCount,
        unalignedCopyCount: this._unalignedCopyCount,
        wrapCount: this._wrapCount,
        bufferGeneration: activeSlot ? activeSlot.generation : 0,
        destroyed: this._destroyed,
      };
    }

    /**
     * Destroy all allocated GPUBuffers (primary, retired, and overflow) idempotently and
     * release CPU shadow memory.
     */
    destroy() {
      if (this._destroyed) {
        return;
      }
      this._destroyed = true;

      for (let i = 0; i < this._slots.length; i += 1) {
        this._releaseSlotTransientBuffers(this._slots[i]);
        this._destroySlotBuffer(this._slots[i]);
      }

      this._slots = [];
      this._shadow = null;
      this._dirtyRanges = [];
      this._allocatedGpuBytes = 0;
      this._cursor = 0;
      this._writtenHighWater = 0;
      this._framePrimaryBytes = 0;
      this._flushedCursor = 0;
      this._flushedDrawCount = 0;
      this._hasFlushedActiveSlot = false;
      this._primaryFrameDrawCount = 0;
      this._frameSliceBytes = 0;
      this._activeSliceIndex = 0;
      this._frameOverflowBytes = 0;
      this._frameDrawCount = 0;
      this._batchDepth = 0;
      this._dirty = false;
    }
  }

  const exportsObj = Object.freeze({
    DEFAULT_UNIFORM_ALIGNMENT,
    DEFAULT_MIN_UNIFORM_BINDING_SIZE,
    DEFAULT_BYTE_SIZE,
    DEFAULT_MAX_BYTE_SIZE,
    DEFAULT_FRAMES_IN_FLIGHT,
    DEFAULT_MAX_BINDING_SIZE,
    WEBGPU_BUFFER_COPY_ALIGNMENT,
    BUFFER_USAGE,
    SHADER_STAGE,
    alignUp,
    alignUniformSliceSize,
    copyForWriteBuffer,
    coalesceDirtyRanges,
    DynamicUniformRing,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6UniformRing = exportsObj;
  } else if (root) {
    root.PlayGta6UniformRing = exportsObj;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportsObj;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
