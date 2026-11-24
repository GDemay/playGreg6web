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
        maxBindingSize,
        minUniformBindingSize = DEFAULT_MIN_UNIFORM_BINDING_SIZE,
        allowEmptyPayloads = false,
        allowOverflow = false,
      } = options;

      if (!device || typeof device.createBuffer !== 'function') {
        throw new TypeError('DynamicUniformRing requires a WebGPU device with createBuffer()');
      }

      const deviceAlignment = resolveDeviceUniformAlignment(device);
      let resolvedAlignment;
      if (alignment === undefined) {
        resolvedAlignment = deviceAlignment;
      } else {
        if (!isPowerOfTwo(alignment) || alignment < WEBGPU_BUFFER_COPY_ALIGNMENT) {
          throw new RangeError(
            `alignment must be a power-of-two integer >= ${WEBGPU_BUFFER_COPY_ALIGNMENT}, got ${alignment}`
          );
        }
        // Never allow dynamic offsets below the device's minUniformBufferOffsetAlignment.
        resolvedAlignment = Math.max(alignment, deviceAlignment);
      }

      const resolvedMaxBindingSize =
        maxBindingSize === undefined ? resolveDeviceMaxBindingSize(device) : maxBindingSize;

      assertPositiveInteger(maxByteSize, 'maxByteSize');
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
      this.deviceMinUniformBufferOffsetAlignment = deviceAlignment;
      this.maxByteSize = maxByteSize;
      this.framesInFlight = framesInFlight;
      this.maxBindingSize = resolvedMaxBindingSize;
      this.minUniformBindingSize = resolvedMinUniformBindingSize;
      this.allowEmptyPayloads = Boolean(allowEmptyPayloads);
      this.allowOverflow = Boolean(allowOverflow);

      this._capacity = resolvedByteSize;
      this._shadow = new Uint8Array(resolvedByteSize);
      this._slots = [];
      this._destroyed = false;

      this._frameIndex = -1;
      this._slotIndex = 0;
      this._cursor = 0;
      this._flushedCursor = 0;
      this._flushedDrawCount = 0;
      this._frameOverflowBytes = 0;
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
      this._overflowAllocCount = 0;
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

    get usedBytes() {
      return this._cursor + this._frameOverflowBytes;
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
      if (this._cursor > 0) {
        nextShadow.set(this._shadow.subarray(0, this._cursor), 0);
      }
      this._shadow = nextShadow;
      this._capacity = nextCapacity;

      // Resize only the active slot now; other slots may still be in flight on the GPU
      // and will resize cleanly when beginFrame() rotates to them.
      // If draws were already flushed into the active slot's previous buffer in this frame,
      // retain the old buffer until the slot rotates back in a later frame.
      const activeSlot = this._slots[this._slotIndex];
      const deferOldDestroy = this._flushedCursor > 0;
      this._resizeSlot(activeSlot, nextCapacity, deferOldDestroy);

      // The active slot now has a fresh GPUBuffer, so any prior partial flush in this frame
      // must be re-uploaded from offset 0 on the next flush().
      this._flushedCursor = 0;
      this._flushedDrawCount = 0;
      this._growCount += 1;
      this._overflowAllocCount += 1;
    }

    _allocateOverflowSlice(slot, alignedSize) {
      let chunk =
        slot.overflowChunks.length > 0
          ? slot.overflowChunks[slot.overflowChunks.length - 1]
          : null;

      if (!chunk || alignUp(chunk.cursor, this.alignment) + alignedSize > chunk.byteSize) {
        const chunkIndex = slot.overflowChunks.length;
        const chunkByteSize = alignUp(
          Math.max(alignedSize, this._capacity),
          this.alignment
        );
        const buffer = this._createGpuUniformBuffer(
          `playgta6-uniform-ring-slot-${slot.index}-overflow-${chunkIndex}`,
          chunkByteSize
        );
        chunk = {
          buffer,
          byteSize: chunkByteSize,
          shadow: new Uint8Array(chunkByteSize),
          cursor: 0,
          flushedCursor: 0,
          drawCount: 0,
          flushedDrawCount: 0,
          dirty: false,
        };
        slot.overflowChunks.push(chunk);
        this._overflowAllocCount += 1;
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
        count += slot.overflowChunks.length + slot.retiredBuffers.length;
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

      const totalUsed = this._cursor + this._frameOverflowBytes;
      if (totalUsed > this._peakUsedBytes) {
        this._peakUsedBytes = totalUsed;
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
      this._frameOverflowBytes = 0;
      this._frameDrawCount = 0;
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

      return {
        frameIndex: this._frameIndex,
        slotIndex: this._slotIndex,
        buffer: slot.buffer,
        byteSize: slot.byteSize,
        generation: slot.generation,
      };
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

      if (primaryEndOffset <= this._capacity) {
        this._shadow.set(uploadBytes, primaryDynamicOffset);
        if (alignedSize > size) {
          this._shadow.fill(0, primaryDynamicOffset + size, primaryEndOffset);
        }
        this._cursor = primaryEndOffset;
        targetBuffer = slot.buffer;
        dynamicOffset = primaryDynamicOffset;
      } else if (primaryEndOffset <= this.maxByteSize) {
        this._grow(primaryEndOffset);
        this._shadow.set(uploadBytes, primaryDynamicOffset);
        if (alignedSize > size) {
          this._shadow.fill(0, primaryDynamicOffset + size, primaryEndOffset);
        }
        this._cursor = primaryEndOffset;
        targetBuffer = slot.buffer;
        dynamicOffset = primaryDynamicOffset;
      } else if (this.allowOverflow) {
        const overflowAlloc = this._allocateOverflowSlice(slot, alignedSize);
        const chunk = overflowAlloc.chunk;
        dynamicOffset = overflowAlloc.dynamicOffset;
        const chunkEndOffset = dynamicOffset + alignedSize;

        chunk.shadow.set(uploadBytes, dynamicOffset);
        if (alignedSize > size) {
          chunk.shadow.fill(0, dynamicOffset + size, chunkEndOffset);
        }
        chunk.cursor = chunkEndOffset;
        chunk.drawCount += 1;
        chunk.dirty = true;
        this._frameOverflowBytes += alignedSize;

        targetBuffer = chunk.buffer;
        isOverflow = true;
      } else {
        throw new RangeError(
          `DynamicUniformRing capacity exceeded: need ${primaryEndOffset} bytes, ` +
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
      const totalUsed = this._cursor + this._frameOverflowBytes;
      if (totalUsed > this._peakUsedBytes) {
        this._peakUsedBytes = totalUsed;
      }

      return {
        buffer: targetBuffer,
        dynamicOffset,
        size,
        paddedSize,
        alignedSize,
        overflow: isOverflow,
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

    /**
     * Upload the dirty CPU shadow prefix for the current frame (plus any dirty per-frame
     * overflow buffers) to GPU memory using regular `ArrayBuffer`-backed, 4-byte-aligned
     * `device.queue.writeBuffer(...)` calls.
     *
     * @returns {{ uploadedBytes: number, drawCount: number }}
     */
    flush() {
      this._assertNotDestroyed();

      const slot = this._slots[this._slotIndex];
      const hasPrimaryDirty = this._cursor > this._flushedCursor;
      let hasOverflowDirty = false;
      for (let i = 0; i < slot.overflowChunks.length; i += 1) {
        const chunk = slot.overflowChunks[i];
        if (chunk.dirty && chunk.cursor > chunk.flushedCursor) {
          hasOverflowDirty = true;
          break;
        }
      }

      if (!this._dirty || (!hasPrimaryDirty && !hasOverflowDirty)) {
        return { uploadedBytes: 0, drawCount: 0 };
      }

      if (!this.device.queue || typeof this.device.queue.writeBuffer !== 'function') {
        throw new TypeError(
          'DynamicUniformRing.flush() requires device.queue.writeBuffer()'
        );
      }

      let uploadedBytes = 0;
      let drawCount = 0;

      if (hasPrimaryDirty) {
        const startOffset = this._flushedCursor;
        const primaryBytes = this._cursor - startOffset;
        const overflowDrawsTotal = slot.overflowChunks.reduce(
          (acc, c) => acc + c.drawCount,
          0
        );
        const primaryDrawTotal = this._frameDrawCount - overflowDrawsTotal;
        const primaryDraws = primaryDrawTotal - this._flushedDrawCount;
        const shadowSlice = this._shadow.subarray(startOffset, this._cursor);

        this._safeWriteBuffer(slot.buffer, startOffset, shadowSlice);

        this._flushedCursor = this._cursor;
        this._flushedDrawCount = primaryDrawTotal;
        uploadedBytes += primaryBytes;
        drawCount += primaryDraws;
      }

      for (let i = 0; i < slot.overflowChunks.length; i += 1) {
        const chunk = slot.overflowChunks[i];
        if (!chunk.dirty || chunk.cursor <= chunk.flushedCursor) {
          continue;
        }
        const startOffset = chunk.flushedCursor;
        const chunkBytes = chunk.cursor - startOffset;
        const chunkDraws = chunk.drawCount - chunk.flushedDrawCount;
        const chunkSlice = chunk.shadow.subarray(startOffset, chunk.cursor);

        this._safeWriteBuffer(chunk.buffer, startOffset, chunkSlice);

        chunk.flushedCursor = chunk.cursor;
        chunk.flushedDrawCount = chunk.drawCount;
        chunk.dirty = false;
        uploadedBytes += chunkBytes;
        drawCount += chunkDraws;
      }

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
        maxByteSize: this.maxByteSize,
        maxBindingSize: this.maxBindingSize,
        allowOverflow: this.allowOverflow,
        usedBytes: this._cursor + this._frameOverflowBytes,
        peakUsedBytes: this._peakUsedBytes,
        allocatedGpuBytes: this._allocatedGpuBytes,
        peakAllocatedGpuBytes: this._peakAllocatedGpuBytes,
        frameDrawCount: this._frameDrawCount,
        totalDrawCount: this._totalDrawCount,
        totalBytesPushed: this._totalBytesPushed,
        totalBytesUploaded: this._totalBytesUploaded,
        flushCount: this._flushCount,
        growCount: this._growCount,
        overflowAllocCount: this._overflowAllocCount,
        overflowBufferCount: this._destroyed ? 0 : this._countLiveOverflowBuffers(),
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
      this._allocatedGpuBytes = 0;
      this._cursor = 0;
      this._flushedCursor = 0;
      this._flushedDrawCount = 0;
      this._frameOverflowBytes = 0;
      this._frameDrawCount = 0;
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
