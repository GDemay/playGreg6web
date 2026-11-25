'use strict';

// Frame-recycled WebGPU staging buffer pool and upload ring.
// Zero external dependencies; UMD export for Web Workers (globalThis.PlayGta6StagingPool)
// and Node.js (module.exports). Aligns cleanly with web/workers/wgpu_worker.js and
// web/workers/texture_formats.js (4-byte copyBufferToBuffer & 256-byte copyBufferToTexture).

(function (root) {
  const DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024; // 4 MiB
  const DEFAULT_MAX_POOL_BYTES = 64 * 1024 * 1024; // 64 MiB
  const DEFAULT_FRAMES_IN_FLIGHT = 3;
  const DEFAULT_MIN_WARM_CHUNKS = 1;
  const DEFAULT_IDLE_EVICT_FRAMES = 0;
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

  /**
   * Align a requested staging buffer size to a WebGPU 4-byte multiple with a
   * minimum size of 4 bytes (`Math.max(4, Math.ceil(size / 4) * 4)`), safe across
   * the full IEEE-754 integer range.
   *
   * @param {number} size
   * @returns {number}
   */
  function alignBufferSize(size) {
    assertNonNegativeInteger(size, 'size');
    if (size === 0) {
      return WEBGPU_BUFFER_COPY_ALIGNMENT;
    }
    return alignUp(size, WEBGPU_BUFFER_COPY_ALIGNMENT);
  }

  function lcmWithBufferAlignment(alignment) {
    if (alignment % WEBGPU_BUFFER_COPY_ALIGNMENT === 0) {
      return alignment;
    }
    if (alignment % 2 === 0) {
      return alignment * 2;
    }
    return alignment * WEBGPU_BUFFER_COPY_ALIGNMENT;
  }

  function isSharedBuffer(buffer) {
    return typeof SharedArrayBuffer !== 'undefined' && buffer instanceof SharedArrayBuffer;
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
    if (isSharedBuffer(input)) {
      return new Uint8Array(input);
    }
    throw new TypeError(
      `${label} expects a Uint8Array, ArrayBufferView, ArrayBuffer, or SharedArrayBuffer`
    );
  }

  /**
   * Sanitize a buffer upload payload for `GPUQueue.writeBuffer()`:
   *  - Copies `SharedArrayBuffer`-backed views into a regular `ArrayBuffer`-backed `Uint8Array`.
   *  - Zero-pads non-4-byte lengths up to a 4-byte multiple (`alignBufferSize`).
   *
   * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} srcBytes
   * @returns {{ bytes: Uint8Array, originalByteLength: number, alignedByteLength: number, padded: boolean, copiedFromSab: boolean }}
   */
  function sanitizeWriteBufferPayload(srcBytes) {
    const src = toUint8Array(srcBytes, 'sanitizeWriteBufferPayload');
    const originalByteLength = src.byteLength;
    if (originalByteLength === 0) {
      return {
        bytes: new Uint8Array(0),
        originalByteLength: 0,
        alignedByteLength: 0,
        padded: false,
        copiedFromSab: false,
      };
    }

    const alignedByteLength = alignBufferSize(originalByteLength);
    const padded = alignedByteLength !== originalByteLength;
    const copiedFromSab = isSharedBuffer(src.buffer);

    if (padded || copiedFromSab) {
      const copy = new Uint8Array(alignedByteLength);
      copy.set(src, 0);
      return {
        bytes: copy,
        originalByteLength,
        alignedByteLength,
        padded,
        copiedFromSab,
      };
    }

    return {
      bytes: src,
      originalByteLength,
      alignedByteLength,
      padded: false,
      copiedFromSab: false,
    };
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
        minWarmChunks = DEFAULT_MIN_WARM_CHUNKS,
        idleEvictFrames = DEFAULT_IDLE_EVICT_FRAMES,
      } = options;

      if (!device || typeof device.createBuffer !== 'function') {
        throw new TypeError('GpuStagingPool requires a WebGPU device with createBuffer()');
      }
      assertPositiveInteger(maxPoolBytes, 'maxPoolBytes');
      assertPositiveInteger(framesInFlight, 'framesInFlight');
      assertNonNegativeInteger(minWarmChunks, 'minWarmChunks');
      assertNonNegativeInteger(idleEvictFrames, 'idleEvictFrames');

      if (mode !== STAGING_MODE.WRITE_BUFFER && mode !== STAGING_MODE.MAPPED) {
        throw new RangeError(
          `Unsupported staging mode: ${JSON.stringify(mode)} (expected 'writeBuffer' or 'mapped')`
        );
      }

      let resolvedChunkSize;
      let initialAlignPads = 0;
      if (chunkSize === undefined) {
        const clampedMax = Math.max(
          WEBGPU_BUFFER_COPY_ALIGNMENT,
          Math.floor(maxPoolBytes / WEBGPU_BUFFER_COPY_ALIGNMENT) * WEBGPU_BUFFER_COPY_ALIGNMENT
        );
        resolvedChunkSize = Math.min(DEFAULT_CHUNK_SIZE, clampedMax);
      } else {
        assertPositiveInteger(chunkSize, 'chunkSize');
        resolvedChunkSize = alignBufferSize(chunkSize);
        if (resolvedChunkSize !== chunkSize) {
          initialAlignPads = 1;
        }
      }

      this.device = device;
      this.chunkSize = resolvedChunkSize;
      this.maxPoolBytes = maxPoolBytes;
      this.framesInFlight = framesInFlight;
      this.mode = mode;
      this.minWarmChunks = minWarmChunks;
      this.idleEvictFrames = idleEvictFrames;

      this._destroyed = false;
      this._currentFrame = -1;
      this._inFrame = false;
      this._frameStartedAtLeastOnce = false;
      this._nextChunkId = 1;

      this._allChunks = new Set();
      this._chunkByBuffer = new Map();
      this._activeChunks = [];
      this._recycledChunks = [];
      this._inFlightChunks = [];
      this._currentChunk = null;
      this._pendingMapPromises = new Set();

      // Pool-wide telemetry.
      this._totalChunksAllocated = 0;
      this._allocatedBytes = 0;
      this._peakAllocatedBytes = 0;
      this._reclaimedChunks = 0;
      this._reclaimedBytes = 0;
      this._oversizeAllocations = 0;
      this._bytesUploaded = 0;
      this._allocationCount = 0;
      this._bufferReuseHits = 0;
      this._bufferReuseMisses = 0;
      this._alignPadCount = initialAlignPads;
      this._sabCopyCount = 0;
      this._doubleReleaseIgnored = 0;

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
        this._frameStartedAtLeastOnce = true;
      }
    }

    _createChunk(byteSize, oversized) {
      const allocSize = alignBufferSize(byteSize);
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
        state: 'active',
        lastUsedFrame: this._currentFrame < 0 ? 0 : this._currentFrame,
        retiredAtFrame: -1,
        gpuDone: false,
        readyToRecycle: false,
        isMapped,
        mapPending: false,
        mappedView,
        destroyed: false,
      };

      this._allChunks.add(chunk);
      this._chunkByBuffer.set(buffer, chunk);
      this._totalChunksAllocated += 1;
      this._allocatedBytes += allocSize;
      if (this._allocatedBytes > this._peakAllocatedBytes) {
        this._peakAllocatedBytes = this._allocatedBytes;
      }

      return chunk;
    }

    _destroyChunk(chunk) {
      if (!chunk || chunk.destroyed || chunk.state === 'destroyed') {
        return;
      }
      chunk.destroyed = true;
      chunk.state = 'destroyed';
      chunk.isMapped = false;
      chunk.mapPending = false;
      chunk.mappedView = null;
      this._allChunks.delete(chunk);
      if (chunk.buffer) {
        this._chunkByBuffer.delete(chunk.buffer);
      }
      if (this._currentChunk === chunk) {
        this._currentChunk = null;
      }
      const activeIdx = this._activeChunks.indexOf(chunk);
      if (activeIdx !== -1) {
        this._activeChunks.splice(activeIdx, 1);
      }
      const recycledIdx = this._recycledChunks.indexOf(chunk);
      if (recycledIdx !== -1) {
        this._recycledChunks.splice(recycledIdx, 1);
      }
      const inFlightIdx = this._inFlightChunks.indexOf(chunk);
      if (inFlightIdx !== -1) {
        this._inFlightChunks.splice(inFlightIdx, 1);
      }
      this._allocatedBytes = Math.max(0, this._allocatedBytes - chunk.size);
      if (chunk.buffer && typeof chunk.buffer.destroy === 'function') {
        chunk.buffer.destroy();
      }
    }

    _pickEvictionIndex() {
      // Prefer evicting idle oversized burst chunks (largest first) to keep
      // steady-state standard chunks warm in the recycled pool.
      let oversizedIdx = -1;
      let oversizedMax = -1;
      for (let i = 0; i < this._recycledChunks.length; i += 1) {
        const candidate = this._recycledChunks[i];
        if (candidate.oversized && candidate.size > oversizedMax) {
          oversizedIdx = i;
          oversizedMax = candidate.size;
        }
      }
      return oversizedIdx !== -1 ? oversizedIdx : 0;
    }

    _reclaimCompletedInFlight() {
      if (this._destroyed || this._inFlightChunks.length === 0) {
        return 0;
      }
      let reclaimed = 0;
      const stillInFlight = [];
      for (let i = 0; i < this._inFlightChunks.length; i += 1) {
        const chunk = this._inFlightChunks[i];
        if (!chunk || chunk.destroyed || chunk.state !== 'inFlight') {
          continue;
        }
        if (chunk.gpuDone) {
          if (this.mode === STAGING_MODE.MAPPED && !chunk.isMapped) {
            chunk.readyToRecycle = true;
            this._beginRemap(chunk);
            if (chunk.isMapped) {
              chunk.cursor = 0;
              chunk.retiredAtFrame = -1;
              chunk.gpuDone = false;
              chunk.readyToRecycle = false;
              chunk.state = 'recycled';
              this._recycledChunks.push(chunk);
              reclaimed += 1;
            } else {
              stillInFlight.push(chunk);
            }
          } else {
            chunk.cursor = 0;
            chunk.retiredAtFrame = -1;
            chunk.gpuDone = false;
            chunk.readyToRecycle = false;
            chunk.state = 'recycled';
            this._recycledChunks.push(chunk);
            reclaimed += 1;
          }
        } else {
          stillInFlight.push(chunk);
        }
      }
      this._inFlightChunks = stillInFlight;
      return reclaimed;
    }

    /**
     * Synchronously move any GPU-completed (`onSubmittedWorkDone`) in-flight chunks
     * into the recycled pool so burst allocations can reuse them immediately.
     *
     * @returns {number} Number of chunks reclaimed into the recycled pool.
     */
    recycleCompleted() {
      this._assertNotDestroyed();
      const recycled = this._reclaimCompletedInFlight();
      if (this.idleEvictFrames > 0) {
        this.reclaimIdleChunks(this.idleEvictFrames);
      }
      return recycled;
    }

    /**
     * Destroy and remove idle recycled (not active, not in-flight) staging chunks
     * whose `currentFrame - chunk.lastUsedFrame >= maxIdleFrames`, while keeping
     * at least `minWarmChunks` warm chunks in the recycled pool.
     *
     * Prefers evicting oversized burst chunks (largest first) and oldest chunks
     * first so steady-state standard chunks stay warm.
     *
     * @param {number} [maxIdleFrames]
     * @returns {{ reclaimedChunks: number, reclaimedBytes: number }}
     */
    reclaimIdleChunks(maxIdleFrames) {
      this._assertNotDestroyed();

      let threshold;
      if (maxIdleFrames === undefined) {
        if (this.idleEvictFrames <= 0) {
          return { reclaimedChunks: 0, reclaimedBytes: 0 };
        }
        threshold = this.idleEvictFrames;
      } else {
        assertNonNegativeInteger(maxIdleFrames, 'maxIdleFrames');
        threshold = maxIdleFrames;
      }

      if (this._recycledChunks.length <= this.minWarmChunks) {
        return { reclaimedChunks: 0, reclaimedBytes: 0 };
      }

      const currentFrame = this._currentFrame < 0 ? 0 : this._currentFrame;
      const candidates = [];
      for (let i = 0; i < this._recycledChunks.length; i += 1) {
        const chunk = this._recycledChunks[i];
        if (
          chunk &&
          !chunk.destroyed &&
          chunk.state === 'recycled' &&
          currentFrame - chunk.lastUsedFrame >= threshold
        ) {
          candidates.push(chunk);
        }
      }

      if (candidates.length === 0) {
        return { reclaimedChunks: 0, reclaimedBytes: 0 };
      }

      candidates.sort((a, b) => {
        if (a.oversized !== b.oversized) {
          return a.oversized ? -1 : 1;
        }
        if (a.oversized && a.size !== b.size) {
          return b.size - a.size;
        }
        if (a.lastUsedFrame !== b.lastUsedFrame) {
          return a.lastUsedFrame - b.lastUsedFrame;
        }
        return a.id - b.id;
      });

      let reclaimedChunks = 0;
      let reclaimedBytes = 0;
      for (let i = 0; i < candidates.length; i += 1) {
        if (this._recycledChunks.length <= this.minWarmChunks) {
          break;
        }
        const chunk = candidates[i];
        if (chunk.destroyed || chunk.state !== 'recycled') {
          continue;
        }
        const byteSize = chunk.size;
        this._destroyChunk(chunk);
        reclaimedChunks += 1;
        reclaimedBytes += byteSize;
      }

      if (reclaimedChunks > 0) {
        this._reclaimedChunks += reclaimedChunks;
        this._reclaimedBytes += reclaimedBytes;
      }

      return { reclaimedChunks, reclaimedBytes };
    }

    _ensureBudget(allocSize, requestedSize) {
      if (this._allocatedBytes + allocSize <= this.maxPoolBytes) {
        return;
      }

      // First reclaim any GPU-completed in-flight chunks so burst eviction can reuse/free them.
      this._reclaimCompletedInFlight();

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
          const evictIdx = this._pickEvictionIndex();
          const [evicted] = this._recycledChunks.splice(evictIdx, 1);
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
          if (
            !candidate.destroyed &&
            candidate.state === 'recycled' &&
            !candidate.oversized &&
            candidate.size >= minSize &&
            candidate.size < bestSize
          ) {
            bestIndex = i;
            bestSize = candidate.size;
          }
        }
      }

      if (bestIndex === -1) {
        for (let i = 0; i < this._recycledChunks.length; i += 1) {
          const candidate = this._recycledChunks[i];
          if (
            !candidate.destroyed &&
            candidate.state === 'recycled' &&
            candidate.size >= minSize &&
            candidate.size < bestSize
          ) {
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
      chunk.lastUsedFrame = this._currentFrame < 0 ? 0 : this._currentFrame;
      chunk.retiredAtFrame = -1;
      chunk.gpuDone = false;
      chunk.readyToRecycle = false;
      chunk.state = 'active';
      return chunk;
    }

    _acquireChunk(requiredSize, oversized) {
      const targetSize = oversized ? alignBufferSize(requiredSize) : this.chunkSize;
      const minRequired = oversized ? targetSize : requiredSize;
      let recycled = this._takeRecycledChunk(minRequired, oversized);

      // During Frame-1 or mid-frame bursts, reclaim GPU-finished chunks on demand
      // before allocating fresh GPUBuffers.
      if (!recycled && this._reclaimCompletedInFlight() > 0) {
        recycled = this._takeRecycledChunk(minRequired, oversized);
      }

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
      if (this._destroyed || !chunk || chunk.destroyed || chunk.state !== 'inFlight') {
        return false;
      }
      const idx = this._inFlightChunks.indexOf(chunk);
      if (idx === -1) {
        return false;
      }
      this._inFlightChunks.splice(idx, 1);
      chunk.cursor = 0;
      chunk.retiredAtFrame = -1;
      chunk.gpuDone = false;
      chunk.readyToRecycle = false;
      chunk.state = 'recycled';
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
        if (!chunk || chunk.destroyed || chunk.state !== 'inFlight' || this._inFlightChunks.indexOf(chunk) === -1) {
          continue;
        }
        chunk.gpuDone = true;
        if (this.mode === STAGING_MODE.MAPPED && !chunk.isMapped) {
          this._beginRemap(chunk);
        }
      }
    }

    _attachWorkDoneFence(retiredChunks) {
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
      this._frameStartedAtLeastOnce = true;
      this._frameAllocations = 0;
      this._frameBytesAllocated = 0;
      this._frameBytesUploaded = 0;

      let recycledCount = 0;
      const stillInFlight = [];

      for (let i = 0; i < this._inFlightChunks.length; i += 1) {
        const chunk = this._inFlightChunks[i];
        if (!chunk || chunk.destroyed || chunk.state !== 'inFlight') {
          continue;
        }
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
              chunk.state = 'recycled';
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
            chunk.state = 'recycled';
            this._recycledChunks.push(chunk);
            recycledCount += 1;
          }
        } else {
          stillInFlight.push(chunk);
        }
      }

      this._inFlightChunks = stillInFlight;

      let idleReclaim = { reclaimedChunks: 0, reclaimedBytes: 0 };
      if (this.idleEvictFrames > 0) {
        idleReclaim = this.reclaimIdleChunks(this.idleEvictFrames);
      }

      return {
        frameIndex: this._currentFrame,
        recycledCount,
        recycledChunks: this._recycledChunks.length,
        inFlightChunks: this._inFlightChunks.length,
        reclaimedChunks: idleReclaim.reclaimedChunks,
        reclaimedBytes: idleReclaim.reclaimedBytes,
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
        if (!chunk.destroyed && chunk.isMapped) {
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
      while (!this._destroyed && this._pendingMapPromises.size > 0) {
        await Promise.all(Array.from(this._pendingMapPromises));
      }
    }

    /**
     * Explicitly retire an active staging slice or GPUBuffer into the in-flight queue.
     * Guarded against double-release: releasing an already-retired, recycled, destroyed,
     * or unknown buffer is a safe no-op returning `false`.
     *
     * @param {{ buffer: GPUBuffer } | GPUBuffer} sliceOrBuffer
     * @returns {boolean} `true` if the active chunk was retired, `false` if ignored.
     */
    release(sliceOrBuffer) {
      this._assertNotDestroyed();
      if (!sliceOrBuffer || typeof sliceOrBuffer !== 'object') {
        return false;
      }
      const buffer = sliceOrBuffer.buffer !== undefined ? sliceOrBuffer.buffer : sliceOrBuffer;
      if (!buffer || typeof buffer !== 'object') {
        return false;
      }
      const chunk = this._chunkByBuffer.get(buffer);
      if (!chunk) {
        return false;
      }
      if (chunk.destroyed || chunk.state !== 'active') {
        this._doubleReleaseIgnored += 1;
        return false;
      }

      const activeIdx = this._activeChunks.indexOf(chunk);
      if (activeIdx !== -1) {
        this._activeChunks.splice(activeIdx, 1);
      }
      if (this._currentChunk === chunk) {
        this._currentChunk = null;
      }

      if (this.mode === STAGING_MODE.MAPPED && chunk.isMapped) {
        if (typeof chunk.buffer.unmap === 'function') {
          chunk.buffer.unmap();
        }
        chunk.isMapped = false;
        chunk.mappedView = null;
      }

      chunk.state = 'inFlight';
      chunk.retiredAtFrame = this._currentFrame < 0 ? 0 : this._currentFrame;
      chunk.gpuDone = false;
      chunk.readyToRecycle = false;
      this._inFlightChunks.push(chunk);
      this._attachWorkDoneFence([chunk]);
      return true;
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

      const wasNotInFrame = !this._inFrame && this._frameStartedAtLeastOnce;

      this.prepareSubmit();

      const rawActive = this._activeChunks;
      this._activeChunks = [];
      this._currentChunk = null;

      const retiredChunks = [];
      for (let i = 0; i < rawActive.length; i += 1) {
        const chunk = rawActive[i];
        if (!chunk || chunk.destroyed || chunk.state !== 'active') {
          this._doubleReleaseIgnored += 1;
          continue;
        }
        chunk.state = 'inFlight';
        chunk.retiredAtFrame = this._currentFrame;
        chunk.gpuDone = false;
        chunk.readyToRecycle = false;
        this._inFlightChunks.push(chunk);
        retiredChunks.push(chunk);
      }

      if (wasNotInFrame && retiredChunks.length === 0) {
        this._doubleReleaseIgnored += 1;
      }

      this._attachWorkDoneFence(retiredChunks);

      let idleReclaim = { reclaimedChunks: 0, reclaimedBytes: 0 };
      if (this.idleEvictFrames > 0) {
        idleReclaim = this.reclaimIdleChunks(this.idleEvictFrames);
      }

      const frameStats = {
        frameIndex: this._currentFrame,
        chunksRetired: retiredChunks.length,
        allocations: this._frameAllocations,
        bytesAllocated: this._frameBytesAllocated,
        bytesUploaded: this._frameBytesUploaded,
        inFlightChunks: this._inFlightChunks.length,
        recycledChunks: this._recycledChunks.length,
        reclaimedChunks: idleReclaim.reclaimedChunks,
        reclaimedBytes: idleReclaim.reclaimedBytes,
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
     * Enforces WebGPU 4-byte minimum alignment for buffer reservation and offsets
     * while preserving `slice.size === byteSize` for backwards compatibility.
     *
     * @param {number} byteSize
     * @param {number} [alignment=4]
     * @returns {{ buffer: GPUBuffer, offset: number, size: number, alignedSize: number, view?: Uint8Array }}
     */
    allocate(byteSize, alignment = WEBGPU_BUFFER_COPY_ALIGNMENT) {
      this._assertNotDestroyed();
      assertPositiveInteger(byteSize, 'byteSize');
      assertPositiveInteger(alignment, 'alignment');
      this._ensureInFrame();

      const alignedSize = alignBufferSize(byteSize);
      const effectiveAlignment = lcmWithBufferAlignment(alignment);

      const isOversized = alignedSize > this.chunkSize;
      let chunk;
      let offset;

      if (isOversized) {
        // Dedicated oversized chunk; preserve _currentChunk for smaller allocations.
        chunk = this._acquireChunk(alignedSize, true);
        offset = 0;
        chunk.cursor = alignedSize;
      } else {
        if (this._currentChunk && !this._currentChunk.destroyed && this._currentChunk.state === 'active') {
          const candidateOffset = alignUp(this._currentChunk.cursor, effectiveAlignment);
          if (candidateOffset + alignedSize <= this._currentChunk.size) {
            chunk = this._currentChunk;
            offset = candidateOffset;
            chunk.cursor = offset + alignedSize;
          }
        }

        if (!chunk) {
          chunk = this._acquireChunk(alignedSize, false);
          offset = 0;
          chunk.cursor = alignedSize;
          this._currentChunk = chunk;
        }
      }

      chunk.lastUsedFrame = this._currentFrame;

      if (isOversized) {
        this._oversizeAllocations += 1;
      }

      if (alignedSize !== byteSize) {
        this._alignPadCount += 1;
      }

      this._allocationCount += 1;
      this._frameAllocations += 1;
      this._frameBytesAllocated += byteSize;

      const slice = {
        buffer: chunk.buffer,
        offset,
        size: byteSize,
        alignedSize,
      };
      if (chunk.isMapped && chunk.mappedView) {
        slice.view = chunk.mappedView.subarray(offset, offset + byteSize);
        if (alignedSize > byteSize) {
          chunk.mappedView.fill(0, offset + byteSize, offset + alignedSize);
        }
      }
      return slice;
    }

    /**
     * Stage `srcBytes` into the pool (with 4-byte aligned length per WebGPU
     * `copyBufferToBuffer` requirements and SharedArrayBuffer-safe copies) and
     * record `encoder.copyBufferToBuffer(...)`.
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

      const copySize = alignBufferSize(src.byteLength);
      const needsPad = copySize !== src.byteLength;
      const slice = this.allocate(copySize, WEBGPU_BUFFER_COPY_ALIGNMENT);

      if (slice.view) {
        slice.view.set(src, 0);
        if (needsPad) {
          slice.view.fill(0, src.byteLength, copySize);
          this._alignPadCount += 1;
        }
      } else {
        if (!this.device.queue || typeof this.device.queue.writeBuffer !== 'function') {
          throw new TypeError(
            'stageBufferUpload in writeBuffer mode requires device.queue.writeBuffer()'
          );
        }
        const prepared = sanitizeWriteBufferPayload(src);
        this.device.queue.writeBuffer(slice.buffer, slice.offset, prepared.bytes);
        if (prepared.padded) {
          this._alignPadCount += 1;
        }
        if (prepared.copiedFromSab) {
          this._sabCopyCount += 1;
        }
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
        const fromSab = isSharedBuffer(src.buffer);
        let payload;
        if (alignedBytesPerRow === unalignedBytesPerRow) {
          if (fromSab) {
            payload = new Uint8Array(stagedBytes);
            payload.set(src.subarray(0, stagedBytes), 0);
          } else {
            payload = src.byteLength === stagedBytes ? src : src.subarray(0, stagedBytes);
          }
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
        if (fromSab) {
          this._sabCopyCount += 1;
        }
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
      const activeCount = this._activeChunks.length;
      const recycledCount = this._recycledChunks.length;
      return {
        totalChunksAllocated: this._totalChunksAllocated,
        activeChunks: activeCount,
        recycledChunks: recycledCount,
        inFlightChunks: this._inFlightChunks.length,
        activeBuffers: activeCount,
        recycledBuffers: recycledCount,
        allocatedBytes: this._allocatedBytes,
        peakAllocatedBytes: this._peakAllocatedBytes,
        peakBytesAllocated: this._peakAllocatedBytes,
        reclaimedChunks: this._reclaimedChunks,
        reclaimedBytes: this._reclaimedBytes,
        oversizeAllocations: this._oversizeAllocations,
        bytesUploaded: this._bytesUploaded,
        allocationCount: this._allocationCount,
        bufferReuseHits: this._bufferReuseHits,
        bufferReuseMisses: this._bufferReuseMisses,
        alignPadCount: this._alignPadCount,
        sabCopyCount: this._sabCopyCount,
        doubleReleaseIgnored: this._doubleReleaseIgnored,
        destroyed: this._destroyed,
      };
    }

    /**
     * Destroy all allocated GPUBuffers idempotently and release pool references.
     */
    destroy() {
      if (this._destroyed) {
        this._doubleReleaseIgnored += 1;
        return;
      }
      this._destroyed = true;

      for (const chunk of Array.from(this._allChunks)) {
        this._destroyChunk(chunk);
      }

      this._allChunks.clear();
      this._chunkByBuffer.clear();
      this._pendingMapPromises.clear();
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
    DEFAULT_MIN_WARM_CHUNKS,
    DEFAULT_IDLE_EVICT_FRAMES,
    WEBGPU_BUFFER_COPY_ALIGNMENT,
    WEBGPU_BYTES_PER_ROW_ALIGNMENT,
    BUFFER_USAGE,
    MAP_MODE,
    STAGING_MODE,
    alignUp,
    alignBufferSize,
    isSharedBuffer,
    sanitizeWriteBufferPayload,
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
