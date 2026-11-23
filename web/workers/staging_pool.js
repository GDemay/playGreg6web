'use strict';

// Frame-recycled WebGPU staging buffer pool and upload ring.
// Zero external dependencies; UMD export for Web Workers (globalThis.PlayGta6StagingPool)
// and Node.js (module.exports). Aligns cleanly with web/workers/wgpu_worker.js and
// web/workers/texture_formats.js (4-byte copyBufferToBuffer & 256-byte copyBufferToTexture).

(function (root) {
  const DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024; // 4 MiB
  const DEFAULT_MAX_POOL_BYTES = 64 * 1024 * 1024; // 64 MiB
  const DEFAULT_FRAMES_IN_FLIGHT = 3;
  const WEBGPU_BUFFER_COPY_ALIGNMENT = 4;
  const WEBGPU_BYTES_PER_ROW_ALIGNMENT = 256;

  // WebGPU GPUBufferUsage bitmask values with headless/Node fallbacks.
  const BUFFER_USAGE = Object.freeze({
    MAP_WRITE:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.MAP_WRITE === 'number'
        ? GPUBufferUsage.MAP_WRITE
        : 0x0002,
    COPY_SRC:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.COPY_SRC === 'number'
        ? GPUBufferUsage.COPY_SRC
        : 0x0004,
    COPY_DST:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.COPY_DST === 'number'
        ? GPUBufferUsage.COPY_DST
        : 0x0008,
  });

  // WebGPU GPUMapMode bitmask values with headless/Node fallbacks.
  const MAP_MODE = Object.freeze({
    WRITE:
      typeof GPUMapMode !== 'undefined' && typeof GPUMapMode.WRITE === 'number'
        ? GPUMapMode.WRITE
        : 0x0002,
  });

  const STAGING_MODE = Object.freeze({
    WRITE_BUFFER: 'writeBuffer',
    MAPPED: 'mapped',
  });

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

  /**
   * Align a non-negative integer up to the nearest multiple of `alignment`.
   * Safe across the full IEEE-754 integer range (no 32-bit bitwise truncation).
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

  function normalizeOrigin(origin) {
    if (origin === undefined || origin === null) {
      return { x: 0, y: 0, z: 0 };
    }
    if (Array.isArray(origin)) {
      const x = origin[0] ?? 0;
      const y = origin[1] ?? 0;
      const z = origin[2] ?? 0;
      assertNonNegativeInteger(x, 'origin[0]');
      assertNonNegativeInteger(y, 'origin[1]');
      assertNonNegativeInteger(z, 'origin[2]');
      return { x, y, z };
    }
    if (typeof origin === 'object') {
      const x = origin.x ?? 0;
      const y = origin.y ?? 0;
      const z = origin.z ?? 0;
      assertNonNegativeInteger(x, 'origin.x');
      assertNonNegativeInteger(y, 'origin.y');
      assertNonNegativeInteger(z, 'origin.z');
      return { x, y, z };
    }
    throw new TypeError('origin must be an object { x, y, z } or [x, y, z] array');
  }

  /**
   * Sub-allocating WebGPU staging buffer pool with frame-epoch recycling.
   *
   * Supports both:
   *  - `'writeBuffer'` mode (default): chunks use `COPY_SRC | COPY_DST`, populated via
   *    `device.queue.writeBuffer` and copied via `encoder.copyBufferToBuffer` /
   *    `encoder.copyBufferToTexture`. Recycles synchronously by frame epoch or GPU fence.
   *  - `'mapped'` mode: chunks use `MAP_WRITE | COPY_SRC` (`mappedAtCreation: true`),
   *    exposing a direct `Uint8Array` `view` for zero-extra-copy staging, unmapped on
   *    `prepareSubmit()` / `endFrame()` and remapped via `mapAsync(GPUMapMode.WRITE)`.
   */
  class GpuStagingPool {
    constructor(options = {}) {
      if (!options || typeof options !== 'object') {
        throw new TypeError('GpuStagingPool constructor expects an options object');
      }
      const {
        device,
        chunkSize,
        maxPoolBytes = DEFAULT_MAX_POOL_BYTES,
        framesInFlight = DEFAULT_FRAMES_IN_FLIGHT,
        mode = STAGING_MODE.WRITE_BUFFER,
      } = options;

      if (!device || typeof device.createBuffer !== 'function') {
        throw new TypeError('GpuStagingPool requires a WebGPU device with createBuffer()');
      }
      assertPositiveInteger(maxPoolBytes, 'maxPoolBytes');
      assertPositiveInteger(framesInFlight, 'framesInFlight');

      if (mode !== STAGING_MODE.WRITE_BUFFER && mode !== STAGING_MODE.MAPPED) {
        throw new RangeError(
          `Unsupported staging mode: ${JSON.stringify(mode)} (expected 'writeBuffer' or 'mapped')`
        );
      }

      let resolvedChunkSize;
      if (chunkSize === undefined) {
        const clampedMax = Math.max(
          WEBGPU_BUFFER_COPY_ALIGNMENT,
          Math.floor(maxPoolBytes / WEBGPU_BUFFER_COPY_ALIGNMENT) * WEBGPU_BUFFER_COPY_ALIGNMENT
        );
        resolvedChunkSize = Math.min(DEFAULT_CHUNK_SIZE, clampedMax);
      } else {
        assertPositiveInteger(chunkSize, 'chunkSize');
        resolvedChunkSize = alignUp(chunkSize, WEBGPU_BUFFER_COPY_ALIGNMENT);
      }

      this.device = device;
      this.chunkSize = resolvedChunkSize;
      this.maxPoolBytes = maxPoolBytes;
      this.framesInFlight = framesInFlight;
      this.mode = mode;

      this._destroyed = false;
      this._currentFrame = -1;
      this._inFrame = false;
      this._nextChunkId = 1;

      this._allChunks = new Set();
      this._activeChunks = [];
      this._recycledChunks = [];
      this._inFlightChunks = [];
      this._currentChunk = null;
      this._pendingMapPromises = new Set();

      // Pool-wide telemetry.
      this._totalChunksAllocated = 0;
      this._allocatedBytes = 0;
      this._peakAllocatedBytes = 0;
      this._bytesUploaded = 0;
      this._allocationCount = 0;
      this._bufferReuseHits = 0;
      this._bufferReuseMisses = 0;

      // Per-frame telemetry (reset on beginFrame / endFrame).
      this._frameAllocations = 0;
      this._frameBytesAllocated = 0;
      this._frameBytesUploaded = 0;
    }

    _assertNotDestroyed() {
      if (this._destroyed) {
        throw new Error('GpuStagingPool has been destroyed');
      }
    }

    _ensureInFrame() {
      if (!this._inFrame) {
        if (this._currentFrame < 0) {
          this._currentFrame = 0;
        }
        this._inFrame = true;
      }
    }

    _createChunk(byteSize, oversized) {
      const allocSize = alignUp(byteSize, WEBGPU_BUFFER_COPY_ALIGNMENT);
      this._ensureBudget(allocSize, byteSize);

      const id = this._nextChunkId++;
      const isMappedMode = this.mode === STAGING_MODE.MAPPED;
      const descriptor = isMappedMode
        ? {
            label: `playgta6-staging-${id}`,
            size: allocSize,
            usage: BUFFER_USAGE.MAP_WRITE | BUFFER_USAGE.COPY_SRC,
            mappedAtCreation: true,
          }
        : {
            label: `playgta6-staging-${id}`,
            size: allocSize,
            usage: BUFFER_USAGE.COPY_SRC | BUFFER_USAGE.COPY_DST,
          };

      const buffer = this.device.createBuffer(descriptor);
      let mappedView = null;
      let isMapped = false;
      if (isMappedMode) {
        if (typeof buffer.getMappedRange !== 'function') {
          if (typeof buffer.destroy === 'function') {
            buffer.destroy();
          }
          throw new TypeError(
            'GpuStagingPool mapped mode requires GPUBuffer.getMappedRange()'
          );
        }
        mappedView = new Uint8Array(buffer.getMappedRange());
        isMapped = true;
      }

      const chunk = {
        id,
        buffer,
        size: allocSize,
        cursor: 0,
        oversized,
        retiredAtFrame: -1,
        gpuDone: false,
        readyToRecycle: false,
        isMapped,
        mapPending: false,
        mappedView,
        destroyed: false,
      };

      this._allChunks.add(chunk);
      this._totalChunksAllocated += 1;
      this._allocatedBytes += allocSize;
      if (this._allocatedBytes > this._peakAllocatedBytes) {
        this._peakAllocatedBytes = this._allocatedBytes;
      }

      return chunk;
    }

    _destroyChunk(chunk) {
      if (chunk.destroyed) {
        return;
      }
      chunk.destroyed = true;
      chunk.isMapped = false;
      chunk.mapPending = false;
      chunk.mappedView = null;
      this._allChunks.delete(chunk);
      this._allocatedBytes = Math.max(0, this._allocatedBytes - chunk.size);
      if (chunk.buffer && typeof chunk.buffer.destroy === 'function') {
        chunk.buffer.destroy();
      }
    }

    _ensureBudget(allocSize, requestedSize) {
      if (this._allocatedBytes + allocSize <= this.maxPoolBytes) {
        return;
      }

      // Check whether evicting idle recycled chunks can make room for this allocation.
      let reclaimable = 0;
      for (let i = 0; i < this._recycledChunks.length; i += 1) {
        reclaimable += this._recycledChunks[i].size;
      }

      if (this._allocatedBytes - reclaimable + allocSize <= this.maxPoolBytes) {
        while (
          this._recycledChunks.length > 0 &&
          this._allocatedBytes + allocSize > this.maxPoolBytes
        ) {
          const evicted = this._recycledChunks.shift();
          this._destroyChunk(evicted);
        }
        return;
      }

      throw new RangeError(
        `GpuStagingPool budget exceeded: cannot allocate ${allocSize}-byte chunk ` +
          `(requested ${requestedSize} bytes; allocated=${this._allocatedBytes}, ` +
          `maxPoolBytes=${this.maxPoolBytes})`
      );
    }

    _takeRecycledChunk(minSize, oversized) {
      if (this._recycledChunks.length === 0) {
        return null;
      }

      let bestIndex = -1;
      let bestSize = Infinity;

      // Prefer standard chunks for standard allocations; otherwise best-fit >= minSize.
      if (!oversized) {
        for (let i = 0; i < this._recycledChunks.length; i += 1) {
          const candidate = this._recycledChunks[i];
          if (!candidate.oversized && candidate.size >= minSize && candidate.size < bestSize) {
            bestIndex = i;
            bestSize = candidate.size;
          }
        }
      }

      if (bestIndex === -1) {
        for (let i = 0; i < this._recycledChunks.length; i += 1) {
          const candidate = this._recycledChunks[i];
          if (candidate.size >= minSize && candidate.size < bestSize) {
            bestIndex = i;
            bestSize = candidate.size;
          }
        }
      }

      if (bestIndex === -1) {
        return null;
      }

      const [chunk] = this._recycledChunks.splice(bestIndex, 1);
      chunk.cursor = 0;
      chunk.retiredAtFrame = -1;
      chunk.gpuDone = false;
      chunk.readyToRecycle = false;
      return chunk;
    }

    _acquireChunk(requiredSize, oversized) {
      const targetSize = oversized ? alignUp(requiredSize, WEBGPU_BUFFER_COPY_ALIGNMENT) : this.chunkSize;
      const recycled = this._takeRecycledChunk(
        oversized ? targetSize : requiredSize,
        oversized
      );

      if (recycled) {
        this._bufferReuseHits += 1;
        this._activeChunks.push(recycled);
        return recycled;
      }

      const created = this._createChunk(targetSize, oversized);
      this._bufferReuseMisses += 1;
      this._activeChunks.push(created);
      return created;
    }

    _moveInFlightToRecycled(chunk) {
      const idx = this._inFlightChunks.indexOf(chunk);
      if (idx === -1) {
        return false;
      }
      this._inFlightChunks.splice(idx, 1);
      chunk.cursor = 0;
      chunk.retiredAtFrame = -1;
      chunk.gpuDone = false;
      chunk.readyToRecycle = false;
      this._recycledChunks.push(chunk);
      return true;
    }

    _beginRemap(chunk) {
      if (
        this._destroyed ||
        chunk.destroyed ||
        this.mode !== STAGING_MODE.MAPPED ||
        chunk.isMapped ||
        chunk.mapPending
      ) {
        return;
      }

      if (typeof chunk.buffer.mapAsync === 'function') {
        chunk.mapPending = true;
        const mapPromise = Promise.resolve(chunk.buffer.mapAsync(MAP_MODE.WRITE)).then(
          () => {
            chunk.mapPending = false;
            if (this._destroyed || chunk.destroyed) {
              return;
            }
            chunk.mappedView = new Uint8Array(chunk.buffer.getMappedRange());
            chunk.isMapped = true;
            if (chunk.readyToRecycle) {
              this._moveInFlightToRecycled(chunk);
            }
          },
          () => {
            chunk.mapPending = false;
          }
        );
        this._pendingMapPromises.add(mapPromise);
        mapPromise.finally(() => {
          this._pendingMapPromises.delete(mapPromise);
        });
      } else if (typeof chunk.buffer.getMappedRange === 'function') {
        chunk.mappedView = new Uint8Array(chunk.buffer.getMappedRange());
        chunk.isMapped = true;
      }
    }

    _onWorkDone(retiredChunks) {
      if (this._destroyed) {
        return;
      }
      for (let i = 0; i < retiredChunks.length; i += 1) {
        const chunk = retiredChunks[i];
        if (chunk.destroyed || this._inFlightChunks.indexOf(chunk) === -1) {
          continue;
        }
        chunk.gpuDone = true;
        if (this.mode === STAGING_MODE.MAPPED && !chunk.isMapped) {
          this._beginRemap(chunk);
        }
      }
    }

    /**
     * Begin a new frame epoch and recycle any in-flight chunks whose retirement epoch
     * has elapsed (`frameIndex - retiredAtFrame >= framesInFlight`) or whose GPU
     * `onSubmittedWorkDone` fence has resolved.
     *
     * @param {number} [frameIndex]
     * @returns {{ frameIndex: number, recycledCount: number, recycledChunks: number, inFlightChunks: number }}
     */
    beginFrame(frameIndex) {
      this._assertNotDestroyed();

      if (this._inFrame) {
        this.endFrame();
      }

      let nextFrame;
      if (frameIndex === undefined) {
        nextFrame = this._currentFrame + 1;
      } else {
        assertNonNegativeInteger(frameIndex, 'frameIndex');
        nextFrame = frameIndex;
      }

      this._currentFrame = nextFrame;
      this._inFrame = true;
      this._frameAllocations = 0;
      this._frameBytesAllocated = 0;
      this._frameBytesUploaded = 0;

      let recycledCount = 0;
      const stillInFlight = [];

      for (let i = 0; i < this._inFlightChunks.length; i += 1) {
        const chunk = this._inFlightChunks[i];
        const epochElapsed = nextFrame - chunk.retiredAtFrame >= this.framesInFlight;
        if (chunk.gpuDone || epochElapsed) {
          if (this.mode === STAGING_MODE.MAPPED && !chunk.isMapped) {
            chunk.readyToRecycle = true;
            this._beginRemap(chunk);
            if (chunk.isMapped) {
              chunk.cursor = 0;
              chunk.retiredAtFrame = -1;
              chunk.gpuDone = false;
              chunk.readyToRecycle = false;
              this._recycledChunks.push(chunk);
              recycledCount += 1;
            } else {
              stillInFlight.push(chunk);
            }
          } else {
            chunk.cursor = 0;
            chunk.retiredAtFrame = -1;
            chunk.gpuDone = false;
            chunk.readyToRecycle = false;
            this._recycledChunks.push(chunk);
            recycledCount += 1;
          }
        } else {
          stillInFlight.push(chunk);
        }
      }

      this._inFlightChunks = stillInFlight;

      return {
        frameIndex: this._currentFrame,
        recycledCount,
        recycledChunks: this._recycledChunks.length,
        inFlightChunks: this._inFlightChunks.length,
      };
    }

    /**
     * Unmap any active mapped staging chunks prior to `device.queue.submit()`.
     * Called automatically by `endFrame()` if not invoked explicitly beforehand.
     */
    prepareSubmit() {
      this._assertNotDestroyed();
      if (this.mode !== STAGING_MODE.MAPPED) {
        return;
      }
      for (let i = 0; i < this._activeChunks.length; i += 1) {
        const chunk = this._activeChunks[i];
        if (chunk.isMapped) {
          if (typeof chunk.buffer.unmap === 'function') {
            chunk.buffer.unmap();
          }
          chunk.isMapped = false;
          chunk.mappedView = null;
        }
      }
    }

    /**
     * Await any in-progress `GPUBuffer.mapAsync()` operations (useful in `'mapped'` mode).
     */
    async waitForPendingMaps() {
      while (this._pendingMapPromises.size > 0) {
        await Promise.all(Array.from(this._pendingMapPromises));
      }
    }

    /**
     * End the current frame, unmap any active mapped chunks, retire active chunks
     * into the in-flight queue for recycling after `framesInFlight` frames (or once
     * `device.queue.onSubmittedWorkDone` resolves), and return per-frame upload stats.
     *
     * @returns {{ frameIndex: number, chunksRetired: number, allocations: number, bytesAllocated: number, bytesUploaded: number, inFlightChunks: number, recycledChunks: number }}
     */
    endFrame() {
      this._assertNotDestroyed();

      if (this._currentFrame < 0) {
        this._currentFrame = 0;
      }

      this.prepareSubmit();

      const retiredChunks = this._activeChunks;
      this._activeChunks = [];
      this._currentChunk = null;

      for (let i = 0; i < retiredChunks.length; i += 1) {
        const chunk = retiredChunks[i];
        chunk.retiredAtFrame = this._currentFrame;
        chunk.gpuDone = false;
        chunk.readyToRecycle = false;
        this._inFlightChunks.push(chunk);
      }

      if (
        retiredChunks.length > 0 &&
        this.device.queue &&
        typeof this.device.queue.onSubmittedWorkDone === 'function'
      ) {
        const workDone = this.device.queue.onSubmittedWorkDone();
        if (workDone && typeof workDone.then === 'function') {
          workDone.then(
            () => {
              this._onWorkDone(retiredChunks);
            },
            () => {
              // Ignore rejected fence (e.g., lost/destroyed device).
            }
          );
        }
      }

      const frameStats = {
        frameIndex: this._currentFrame,
        chunksRetired: retiredChunks.length,
        allocations: this._frameAllocations,
        bytesAllocated: this._frameBytesAllocated,
        bytesUploaded: this._frameBytesUploaded,
        inFlightChunks: this._inFlightChunks.length,
        recycledChunks: this._recycledChunks.length,
      };

      this._inFrame = false;
      this._frameAllocations = 0;
      this._frameBytesAllocated = 0;
      this._frameBytesUploaded = 0;

      return frameStats;
    }

    /**
     * Sub-allocate an aligned staging slice from the current active chunk, rolling
     * over to a recycled or newly allocated chunk (or a dedicated oversized chunk
     * when `byteSize > chunkSize`) within `maxPoolBytes`.
     *
     * @param {number} byteSize
     * @param {number} [alignment=4]
     * @returns {{ buffer: GPUBuffer, offset: number, size: number, view?: Uint8Array }}
     */
    allocate(byteSize, alignment = WEBGPU_BUFFER_COPY_ALIGNMENT) {
      this._assertNotDestroyed();
      assertPositiveInteger(byteSize, 'byteSize');
      assertPositiveInteger(alignment, 'alignment');
      this._ensureInFrame();

      let chunk;
      let offset;

      if (byteSize > this.chunkSize) {
        // Dedicated oversized chunk; preserve _currentChunk for smaller allocations.
        chunk = this._acquireChunk(byteSize, true);
        offset = 0;
        chunk.cursor = byteSize;
      } else {
        if (this._currentChunk) {
          const candidateOffset = alignUp(this._currentChunk.cursor, alignment);
          if (candidateOffset + byteSize <= this._currentChunk.size) {
            chunk = this._currentChunk;
            offset = candidateOffset;
            chunk.cursor = offset + byteSize;
          }
        }

        if (!chunk) {
          chunk = this._acquireChunk(byteSize, false);
          offset = 0;
          chunk.cursor = byteSize;
          this._currentChunk = chunk;
        }
      }

      this._allocationCount += 1;
      this._frameAllocations += 1;
      this._frameBytesAllocated += byteSize;

      const slice = {
        buffer: chunk.buffer,
        offset,
        size: byteSize,
      };
      if (chunk.isMapped && chunk.mappedView) {
        slice.view = chunk.mappedView.subarray(offset, offset + byteSize);
      }
      return slice;
    }

    /**
     * Stage `srcBytes` into the pool (with 4-byte aligned length per WebGPU
     * `copyBufferToBuffer` requirements) and record `encoder.copyBufferToBuffer(...)`.
     *
     * Supports both `device.queue.writeBuffer` staging-ring mode and mapped
     * `copyBufferToBuffer` mode transparently.
     *
     * @param {GPUCommandEncoder} encoder
     * @param {GPUBuffer} dstBuffer
     * @param {number} dstOffset
     * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} srcBytes
     */
    stageBufferUpload(encoder, dstBuffer, dstOffset, srcBytes) {
      this._assertNotDestroyed();
      if (!encoder || typeof encoder.copyBufferToBuffer !== 'function') {
        throw new TypeError('stageBufferUpload requires an encoder with copyBufferToBuffer()');
      }
      if (!dstBuffer || typeof dstBuffer !== 'object') {
        throw new TypeError('stageBufferUpload requires a destination GPUBuffer');
      }
      assertNonNegativeInteger(dstOffset, 'dstOffset');
      if (dstOffset % WEBGPU_BUFFER_COPY_ALIGNMENT !== 0) {
        throw new RangeError(
          `dstOffset (${dstOffset}) must be a multiple of ${WEBGPU_BUFFER_COPY_ALIGNMENT}`
        );
      }

      const src = toUint8Array(srcBytes, 'stageBufferUpload');
      if (src.byteLength === 0) {
        return {
          buffer: null,
          offset: 0,
          size: 0,
          srcByteLength: 0,
        };
      }

      const copySize = alignUp(src.byteLength, WEBGPU_BUFFER_COPY_ALIGNMENT);
      const slice = this.allocate(copySize, WEBGPU_BUFFER_COPY_ALIGNMENT);

      if (slice.view) {
        slice.view.set(src, 0);
        if (copySize > src.byteLength) {
          slice.view.fill(0, src.byteLength, copySize);
        }
      } else {
        if (!this.device.queue || typeof this.device.queue.writeBuffer !== 'function') {
          throw new TypeError(
            'stageBufferUpload in writeBuffer mode requires device.queue.writeBuffer()'
          );
        }
        let payload;
        if (copySize === src.byteLength) {
          payload = src;
        } else {
          payload = new Uint8Array(copySize);
          payload.set(src, 0);
        }
        this.device.queue.writeBuffer(slice.buffer, slice.offset, payload);
      }

      encoder.copyBufferToBuffer(slice.buffer, slice.offset, dstBuffer, dstOffset, copySize);

      this._bytesUploaded += copySize;
      this._frameBytesUploaded += copySize;

      return {
        buffer: slice.buffer,
        offset: slice.offset,
        size: copySize,
        srcByteLength: src.byteLength,
      };
    }

    /**
     * Stage texture texel/block rows into the pool, enforcing WebGPU's 256-byte
     * `bytesPerRow` alignment, and record `encoder.copyBufferToTexture(...)`.
     *
     * @param {GPUCommandEncoder} encoder
     * @param {GPUTexture} dstTexture
     * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} srcBytes
     * @param {{ width: number, height: number, depthOrLayers?: number, depthOrArrayLayers?: number, bytesPerRow?: number, rowsPerImage?: number, mipLevel?: number, origin?: { x?: number, y?: number, z?: number } | number[], aspect?: string }} layout
     */
    stageTextureUpload(encoder, dstTexture, srcBytes, layout) {
      this._assertNotDestroyed();
      if (!encoder || typeof encoder.copyBufferToTexture !== 'function') {
        throw new TypeError('stageTextureUpload requires an encoder with copyBufferToTexture()');
      }
      if (!dstTexture || typeof dstTexture !== 'object') {
        throw new TypeError('stageTextureUpload requires a destination GPUTexture');
      }
      if (!layout || typeof layout !== 'object') {
        throw new TypeError('stageTextureUpload requires a layout options object');
      }

      const src = toUint8Array(srcBytes, 'stageTextureUpload');
      const { width, height, mipLevel = 0, origin, aspect } = layout;
      const depthOrLayers = layout.depthOrLayers ?? layout.depthOrArrayLayers ?? 1;
      const rowsPerImage = layout.rowsPerImage ?? height;

      assertPositiveInteger(width, 'width');
      assertPositiveInteger(height, 'height');
      assertPositiveInteger(depthOrLayers, 'depthOrLayers');
      assertPositiveInteger(rowsPerImage, 'rowsPerImage');
      assertNonNegativeInteger(mipLevel, 'mipLevel');
      const normOrigin = normalizeOrigin(origin);

      const totalRows = rowsPerImage * depthOrLayers;
      let unalignedBytesPerRow = layout.bytesPerRow;
      if (unalignedBytesPerRow === undefined) {
        if (src.byteLength > 0 && src.byteLength % totalRows === 0) {
          unalignedBytesPerRow = src.byteLength / totalRows;
        } else {
          throw new RangeError(
            'bytesPerRow is required when srcBytes.byteLength is not evenly divisible by rowsPerImage * depthOrLayers'
          );
        }
      }
      assertPositiveInteger(unalignedBytesPerRow, 'bytesPerRow');

      const requiredSrcBytes = unalignedBytesPerRow * totalRows;
      if (src.byteLength < requiredSrcBytes) {
        throw new RangeError(
          `srcBytes too small (${src.byteLength} bytes; need at least ${requiredSrcBytes} bytes ` +
            `for ${totalRows} rows of ${unalignedBytesPerRow} bytes)`
        );
      }

      const alignedBytesPerRow = alignUp(unalignedBytesPerRow, WEBGPU_BYTES_PER_ROW_ALIGNMENT);
      const stagedBytes = alignedBytesPerRow * totalRows;
      const slice = this.allocate(stagedBytes, WEBGPU_BYTES_PER_ROW_ALIGNMENT);

      if (slice.view) {
        if (alignedBytesPerRow === unalignedBytesPerRow) {
          slice.view.set(src.subarray(0, stagedBytes), 0);
        } else {
          for (let row = 0; row < totalRows; row += 1) {
            const srcRowStart = row * unalignedBytesPerRow;
            const dstRowStart = row * alignedBytesPerRow;
            slice.view.set(
              src.subarray(srcRowStart, srcRowStart + unalignedBytesPerRow),
              dstRowStart
            );
            slice.view.fill(0, dstRowStart + unalignedBytesPerRow, dstRowStart + alignedBytesPerRow);
          }
        }
      } else {
        if (!this.device.queue || typeof this.device.queue.writeBuffer !== 'function') {
          throw new TypeError(
            'stageTextureUpload in writeBuffer mode requires device.queue.writeBuffer()'
          );
        }
        let payload;
        if (alignedBytesPerRow === unalignedBytesPerRow) {
          payload = src.byteLength === stagedBytes ? src : src.subarray(0, stagedBytes);
        } else {
          payload = new Uint8Array(stagedBytes);
          for (let row = 0; row < totalRows; row += 1) {
            const srcRowStart = row * unalignedBytesPerRow;
            payload.set(
              src.subarray(srcRowStart, srcRowStart + unalignedBytesPerRow),
              row * alignedBytesPerRow
            );
          }
        }
        this.device.queue.writeBuffer(slice.buffer, slice.offset, payload);
      }

      const source = {
        buffer: slice.buffer,
        offset: slice.offset,
        bytesPerRow: alignedBytesPerRow,
        rowsPerImage,
      };
      const destination = {
        texture: dstTexture,
        mipLevel,
        origin: normOrigin,
      };
      if (aspect !== undefined) {
        destination.aspect = aspect;
      }
      const copySize = {
        width,
        height,
        depthOrArrayLayers: depthOrLayers,
      };

      encoder.copyBufferToTexture(source, destination, copySize);

      this._bytesUploaded += stagedBytes;
      this._frameBytesUploaded += stagedBytes;

      return {
        buffer: slice.buffer,
        offset: slice.offset,
        size: stagedBytes,
        bytesPerRow: alignedBytesPerRow,
        unalignedBytesPerRow,
        rowsPerImage,
        width,
        height,
        depthOrArrayLayers: depthOrLayers,
      };
    }

    /**
     * Alias for `stageBufferUpload`.
     */
    writeBuffer(encoder, dstBuffer, dstOffset, srcBytes) {
      return this.stageBufferUpload(encoder, dstBuffer, dstOffset, srcBytes);
    }

    /**
     * Alias for `stageTextureUpload`.
     */
    writeTexture(encoder, dstTexture, srcBytes, layout) {
      return this.stageTextureUpload(encoder, dstTexture, srcBytes, layout);
    }

    /**
     * Return pool-wide allocation, reuse, and upload statistics.
     */
    getStats() {
      return {
        totalChunksAllocated: this._totalChunksAllocated,
        activeChunks: this._activeChunks.length,
        recycledChunks: this._recycledChunks.length,
        inFlightChunks: this._inFlightChunks.length,
        allocatedBytes: this._allocatedBytes,
        peakAllocatedBytes: this._peakAllocatedBytes,
        bytesUploaded: this._bytesUploaded,
        allocationCount: this._allocationCount,
        bufferReuseHits: this._bufferReuseHits,
        bufferReuseMisses: this._bufferReuseMisses,
      };
    }

    /**
     * Destroy all allocated GPUBuffers idempotently and release pool references.
     */
    destroy() {
      if (this._destroyed) {
        return;
      }
      this._destroyed = true;

      for (const chunk of Array.from(this._allChunks)) {
        this._destroyChunk(chunk);
      }

      this._allChunks.clear();
      this._activeChunks = [];
      this._recycledChunks = [];
      this._inFlightChunks = [];
      this._currentChunk = null;
      this._allocatedBytes = 0;
      this._inFrame = false;
    }
  }

  const exportsObj = Object.freeze({
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_POOL_BYTES,
    DEFAULT_FRAMES_IN_FLIGHT,
    WEBGPU_BUFFER_COPY_ALIGNMENT,
    WEBGPU_BYTES_PER_ROW_ALIGNMENT,
    BUFFER_USAGE,
    MAP_MODE,
    STAGING_MODE,
    alignUp,
    GpuStagingPool,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6StagingPool = exportsObj;
  } else if (root) {
    root.PlayGta6StagingPool = exportsObj;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportsObj;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
