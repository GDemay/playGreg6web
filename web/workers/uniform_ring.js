'use strict';

// 256-byte-aligned WebGPU dynamic uniform buffer ring & per-frame single-flush draw constant batcher.
// Zero external dependencies; UMD export for Web Workers (globalThis.PlayGta6UniformRing)
// and Node.js (module.exports). Aligns cleanly with web/workers/wgpu_worker.js and
// web/workers/staging_pool.js.

(function (root) {
  const DEFAULT_UNIFORM_ALIGNMENT = 256;
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
        alignment,
        maxByteSize = DEFAULT_MAX_BYTE_SIZE,
        framesInFlight = DEFAULT_FRAMES_IN_FLIGHT,
        maxBindingSize = DEFAULT_MAX_BINDING_SIZE,
      } = options;

      if (!device || typeof device.createBuffer !== 'function') {
        throw new TypeError('DynamicUniformRing requires a WebGPU device with createBuffer()');
      }

      const deviceAlignment =
        device.limits &&
        typeof device.limits.minUniformBufferOffsetAlignment === 'number' &&
        device.limits.minUniformBufferOffsetAlignment > 0
          ? device.limits.minUniformBufferOffsetAlignment
          : DEFAULT_UNIFORM_ALIGNMENT;

      const resolvedAlignment = alignment === undefined ? deviceAlignment : alignment;
      if (
        !isPowerOfTwo(resolvedAlignment) ||
        resolvedAlignment < WEBGPU_BUFFER_COPY_ALIGNMENT
      ) {
        throw new RangeError(
          `alignment must be a power-of-two integer >= ${WEBGPU_BUFFER_COPY_ALIGNMENT}, got ${resolvedAlignment}`
        );
      }

      assertPositiveInteger(maxByteSize, 'maxByteSize');
      assertPositiveInteger(framesInFlight, 'framesInFlight');
      assertPositiveInteger(maxBindingSize, 'maxBindingSize');

      if (maxByteSize < resolvedAlignment) {
        throw new RangeError(
          `maxByteSize (${maxByteSize}) must be at least alignment (${resolvedAlignment})`
        );
      }

      let resolvedByteSize;
      if (byteSize === undefined) {
        const clampedMax =
          Math.floor(maxByteSize / resolvedAlignment) * resolvedAlignment;
        resolvedByteSize = Math.min(
          alignUp(DEFAULT_BYTE_SIZE, resolvedAlignment),
          clampedMax
        );
      } else {
        assertPositiveInteger(byteSize, 'byteSize');
        resolvedByteSize = alignUp(byteSize, resolvedAlignment);
        if (resolvedByteSize > maxByteSize) {
          throw new RangeError(
            `Aligned byteSize (${resolvedByteSize}) exceeds maxByteSize (${maxByteSize})`
          );
        }
      }

      this.device = device;
      this.alignment = resolvedAlignment;
      this.maxByteSize = maxByteSize;
      this.framesInFlight = framesInFlight;
      this.maxBindingSize = maxBindingSize;

      this._capacity = resolvedByteSize;
      this._shadow = new Uint8Array(resolvedByteSize);
      this._slots = [];
      this._destroyed = false;

      this._frameIndex = -1;
      this._slotIndex = 0;
      this._cursor = 0;
      this._flushedCursor = 0;
      this._flushedDrawCount = 0;
      this._dirty = false;

      // Telemetry counters.
      this._allocatedGpuBytes = 0;
      this._peakAllocatedGpuBytes = 0;
      this._peakUsedBytes = 0;
      this._frameDrawCount = 0;
      this._totalDrawCount = 0;
      this._totalBytesPushed = 0;
      this._totalBytesUploaded = 0;
      this._flushCount = 0;
      this._growCount = 0;

      for (let i = 0; i < framesInFlight; i += 1) {
        const buffer = this._createSlotBuffer(i, resolvedByteSize);
        this._slots.push({
          index: i,
          buffer,
          byteSize: resolvedByteSize,
          generation: 1,
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

    get usedBytes() {
      return this._cursor;
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

    _createSlotBuffer(slotIndex, byteSize) {
      const buffer = this.device.createBuffer({
        label: `playgta6-uniform-ring-slot-${slotIndex}`,
        size: byteSize,
        usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST,
      });
      this._allocatedGpuBytes += byteSize;
      if (this._allocatedGpuBytes > this._peakAllocatedGpuBytes) {
        this._peakAllocatedGpuBytes = this._allocatedGpuBytes;
      }
      return buffer;
    }

    _destroySlotBuffer(slot) {
      if (!slot || !slot.buffer) {
        return;
      }
      const oldBuffer = slot.buffer;
      slot.buffer = null;
      this._allocatedGpuBytes = Math.max(0, this._allocatedGpuBytes - slot.byteSize);
      slot.byteSize = 0;
      if (typeof oldBuffer.destroy === 'function') {
        oldBuffer.destroy();
      }
    }

    _resizeSlot(slot, targetByteSize) {
      if (slot.byteSize === targetByteSize && slot.buffer) {
        return;
      }
      this._destroySlotBuffer(slot);
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
          Math.floor(this.maxByteSize / WEBGPU_BUFFER_COPY_ALIGNMENT) *
          WEBGPU_BUFFER_COPY_ALIGNMENT;
        nextCapacity = Math.max(requiredBytes, alignedMax);
      }

      const nextShadow = new Uint8Array(nextCapacity);
      if (this._cursor > 0) {
        nextShadow.set(this._shadow.subarray(0, this._cursor), 0);
      }
      this._shadow = nextShadow;
      this._capacity = nextCapacity;

      // Resize only the active slot now; other slots may still be in flight on the GPU
      // and will resize cleanly when beginFrame() rotates to them.
      const activeSlot = this._slots[this._slotIndex];
      this._resizeSlot(activeSlot, nextCapacity);

      // The active slot now has a fresh GPUBuffer, so any prior partial flush in this frame
      // must be re-uploaded from offset 0 on the next flush().
      this._flushedCursor = 0;
      this._flushedDrawCount = 0;
      this._growCount += 1;
    }

    /**
     * Begin a new frame: rotate to the next frame-in-flight slot, resize the slot's
     * GPUBuffer if the ring grew on an earlier frame, and reset the write cursor to 0.
     *
     * @param {number} [frameIndex]
     * @returns {{ frameIndex: number, slotIndex: number, buffer: GPUBuffer, byteSize: number, generation: number }}
     */
    beginFrame(frameIndex) {
      this._assertNotDestroyed();

      if (this._cursor > this._peakUsedBytes) {
        this._peakUsedBytes = this._cursor;
      }

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
      this._flushedCursor = 0;
      this._flushedDrawCount = 0;
      this._frameDrawCount = 0;
      this._dirty = false;

      const slot = this._slots[this._slotIndex];
      if (slot.byteSize < this._capacity) {
        this._resizeSlot(slot, this._capacity);
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
     * Append a per-draw constant payload into the CPU shadow ring aligned to `alignment`.
     * Auto-grows capacity (doubling up to `maxByteSize`) when needed.
     *
     * @param {ArrayBufferView | ArrayBuffer | SharedArrayBuffer} data
     * @returns {{ buffer: GPUBuffer, dynamicOffset: number, size: number, alignedSize: number }}
     */
    pushConstants(data) {
      this._assertNotDestroyed();

      const src = toUint8Array(data, 'pushConstants');
      const size = src.byteLength;
      if (size === 0) {
        throw new RangeError('pushConstants data must not be empty (got 0 bytes)');
      }
      if (size > this.maxBindingSize) {
        throw new RangeError(
          `pushConstants size (${size} bytes) exceeds maxBindingSize (${this.maxBindingSize} bytes)`
        );
      }

      if (this._frameIndex < 0) {
        this._frameIndex = 0;
      }

      const dynamicOffset = alignUp(this._cursor, this.alignment);
      const alignedSize = alignUp(size, this.alignment);
      const endOffset = dynamicOffset + alignedSize;

      if (endOffset > this._capacity) {
        this._grow(endOffset);
      }

      this._shadow.set(src, dynamicOffset);
      if (alignedSize > size) {
        this._shadow.fill(0, dynamicOffset + size, endOffset);
      }

      this._cursor = endOffset;
      this._dirty = true;
      this._frameDrawCount += 1;
      this._totalDrawCount += 1;
      this._totalBytesPushed += size;
      if (this._cursor > this._peakUsedBytes) {
        this._peakUsedBytes = this._cursor;
      }

      const slot = this._slots[this._slotIndex];
      return {
        buffer: slot.buffer,
        dynamicOffset,
        size,
        alignedSize,
      };
    }

    /**
     * Upload the dirty CPU shadow prefix for the current frame to the active slot's GPUBuffer
     * in a single `device.queue.writeBuffer(...)` call.
     *
     * @returns {{ uploadedBytes: number, drawCount: number }}
     */
    flush() {
      this._assertNotDestroyed();

      if (!this._dirty || this._cursor <= this._flushedCursor) {
        return { uploadedBytes: 0, drawCount: 0 };
      }

      if (!this.device.queue || typeof this.device.queue.writeBuffer !== 'function') {
        throw new TypeError(
          'DynamicUniformRing.flush() requires device.queue.writeBuffer()'
        );
      }

      const slot = this._slots[this._slotIndex];
      const startOffset = this._flushedCursor;
      const uploadedBytes = this._cursor - startOffset;
      const drawCount = this._frameDrawCount - this._flushedDrawCount;
      const shadowSlice = this._shadow.subarray(startOffset, this._cursor);

      this.device.queue.writeBuffer(slot.buffer, startOffset, shadowSlice);

      this._flushedCursor = this._cursor;
      this._flushedDrawCount = this._frameDrawCount;
      this._dirty = false;
      this._flushCount += 1;
      this._totalBytesUploaded += uploadedBytes;

      return { uploadedBytes, drawCount };
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
     * uniform buffer at `offset: 0` with a dynamic window size `sliceSize`.
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
        resolvedSize = sliceSize;
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
     * Return ring configuration, usage, and upload diagnostics.
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
        byteSize: this._capacity,
        maxByteSize: this.maxByteSize,
        maxBindingSize: this.maxBindingSize,
        usedBytes: this._cursor,
        peakUsedBytes: this._peakUsedBytes,
        allocatedGpuBytes: this._allocatedGpuBytes,
        peakAllocatedGpuBytes: this._peakAllocatedGpuBytes,
        frameDrawCount: this._frameDrawCount,
        totalDrawCount: this._totalDrawCount,
        totalBytesPushed: this._totalBytesPushed,
        totalBytesUploaded: this._totalBytesUploaded,
        flushCount: this._flushCount,
        growCount: this._growCount,
        bufferGeneration: activeSlot ? activeSlot.generation : 0,
        destroyed: this._destroyed,
      };
    }

    /**
     * Destroy all allocated GPUBuffers idempotently and release CPU shadow memory.
     */
    destroy() {
      if (this._destroyed) {
        return;
      }
      this._destroyed = true;

      for (let i = 0; i < this._slots.length; i += 1) {
        this._destroySlotBuffer(this._slots[i]);
      }

      this._slots = [];
      this._shadow = null;
      this._allocatedGpuBytes = 0;
      this._cursor = 0;
      this._flushedCursor = 0;
      this._flushedDrawCount = 0;
      this._frameDrawCount = 0;
      this._dirty = false;
    }
  }

  const exportsObj = Object.freeze({
    DEFAULT_UNIFORM_ALIGNMENT,
    DEFAULT_BYTE_SIZE,
    DEFAULT_MAX_BYTE_SIZE,
    DEFAULT_FRAMES_IN_FLIGHT,
    DEFAULT_MAX_BINDING_SIZE,
    BUFFER_USAGE,
    SHADER_STAGE,
    alignUp,
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
