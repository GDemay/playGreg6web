'use strict';

// Frame-recycled WebGPU staging buffer pool and upload ring.
// Zero external dependencies; UMD export for Web Workers (globalThis.PlayGta6StagingPool)
// and Node.js (module.exports). Aligns cleanly with web/workers/wgpu_worker.js and
// web/workers/texture_formats.js (4-byte copyBufferToBuffer & 256-byte copyBufferToTexture).

(function (root) {
  const DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024; // 4 MiB
  const DEFAULT_MAX_POOL_BYTES = 64 * 1024 * 1024; // 64 MiB
  const DEFAULT_FRAMES_IN_FLIGHT = 3;
  const DEFAULT_MAX_FRAMES_IN_FLIGHT = DEFAULT_FRAMES_IN_FLIGHT;
  const DEFAULT_MIN_WARM_CHUNKS = 1;
  const DEFAULT_IDLE_EVICT_FRAMES = 0;
  const DEFAULT_IDLE_SHRINK_FRAMES = 120; // ~4 seconds at 30 FPS for transient oversized slabs
  const DEFAULT_LARGE_SLAB_THRESHOLD = 2 * 1024 * 1024; // 2 MiB
  const DEFAULT_MAX_LARGE_SLABS = 12;
  const DEFAULT_POW2_MIN_SLAB_BYTES = 1 * 1024 * 1024; // 1 MiB
  const DEFAULT_POW2_MAX_BUCKET_BYTES = 16 * 1024 * 1024; // 16 MiB
  const DEFAULT_LARGE_SLAB_BUCKETS = Object.freeze([
    2 * 1024 * 1024, // 2 MiB
    3 * 1024 * 1024, // 3 MiB (fits 4K 1920x1080 256B-aligned chroma = 2.11 MiB)
    4 * 1024 * 1024, // 4 MiB (fits 4K NV12 UV = 3.96 MiB)
    6 * 1024 * 1024, // 6 MiB
    8 * 1024 * 1024, // 8 MiB (fits 4K 3840x2160 Y plane = 7.91 MiB)
    12 * 1024 * 1024, // 12 MiB (fits 8.3–12 MiB video / texture slabs)
    16 * 1024 * 1024, // 16 MiB
  ]);
  const POW2_LARGE_SLAB_BUCKETS = Object.freeze([
    1 * 1024 * 1024, // 1 MiB
    2 * 1024 * 1024, // 2 MiB (fits 1920x1080 = 2.07 MB raw chroma)
    4 * 1024 * 1024, // 4 MiB (fits 2.07 MiB / 2048x1080 256B-aligned chroma & 4K NV12 UV)
    8 * 1024 * 1024, // 8 MiB (fits 3840x2160 = 7.91 MiB / 8.29 MB 4K Y plane)
    16 * 1024 * 1024, // 16 MiB (fits 8.3 MiB / 4096x2160 4K Y & 16 MiB RGBA slabs)
  ]);
  const BUCKET_STRATEGY = Object.freeze({
    DEFAULT: 'default',
    POW2: 'pow2',
  });
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

  function sanitizeNonNegativeCounter(value) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
      return 0;
    }
    return Math.floor(value);
  }

  function clampUnitRate(numerator, denominator) {
    if (
      typeof numerator !== 'number' ||
      typeof denominator !== 'number' ||
      !Number.isFinite(numerator) ||
      !Number.isFinite(denominator) ||
      denominator <= 0 ||
      numerator <= 0
    ) {
      return 0;
    }
    const rate = numerator / denominator;
    if (!Number.isFinite(rate) || rate <= 0) {
      return 0;
    }
    return rate >= 1 ? 1 : rate;
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

  function normalizeLargeSlabBuckets(buckets) {
    if (!Array.isArray(buckets)) {
      throw new TypeError('largeSlabBuckets must be an array of positive integers');
    }
    const seen = new Set();
    const normalized = [];
    for (let i = 0; i < buckets.length; i += 1) {
      assertPositiveInteger(buckets[i], `largeSlabBuckets[${i}]`);
      const aligned = alignBufferSize(buckets[i]);
      if (!seen.has(aligned)) {
        seen.add(aligned);
        normalized.push(aligned);
      }
    }
    normalized.sort((a, b) => a - b);
    return Object.freeze(normalized);
  }

  /**
   * Compute the smallest power of two `>= value`, safe across the full IEEE-754
   * safe-integer range (no 32-bit bitwise truncation).
   *
   * @param {number} value
   * @returns {number}
   */
  function nextPowerOfTwo(value) {
    assertPositiveInteger(value, 'value');
    if (value <= 1) {
      return 1;
    }
    let pow2 = 1;
    while (pow2 < value) {
      pow2 *= 2;
    }
    return pow2;
  }

  /**
   * Build a sorted, frozen power-of-two bucket ladder from `minBytes` through
   * `maxBytes` (default 1 MiB .. 16 MiB for 4K Bink2 YUV / RGBA planes).
   *
   * @param {number} [minBytes=DEFAULT_POW2_MIN_SLAB_BYTES]
   * @param {number} [maxBytes=DEFAULT_POW2_MAX_BUCKET_BYTES]
   * @returns {readonly number[]}
   */
  function buildPowerOfTwoBuckets(
    minBytes = DEFAULT_POW2_MIN_SLAB_BYTES,
    maxBytes = DEFAULT_POW2_MAX_BUCKET_BYTES
  ) {
    assertPositiveInteger(minBytes, 'minBytes');
    assertPositiveInteger(maxBytes, 'maxBytes');
    if (minBytes > maxBytes) {
      throw new RangeError(`minBytes (${minBytes}) must be <= maxBytes (${maxBytes})`);
    }
    const start = Math.max(WEBGPU_BUFFER_COPY_ALIGNMENT, nextPowerOfTwo(minBytes));
    const buckets = [];
    for (let size = start; size <= maxBytes; size *= 2) {
      buckets.push(size);
    }
    if (buckets.length === 0) {
      buckets.push(alignBufferSize(maxBytes));
    }
    return Object.freeze(buckets);
  }

  /**
   * Select a power-of-two bucket size `>= alignBufferSize(byteSize)` clamped between
   * `minBucketBytes` and `maxBucketBytes` (default 16 MiB). Requests above
   * `maxBucketBytes` fall back to exact 4-byte-aligned size to avoid doubling huge slabs.
   *
   * @param {number} byteSize
   * @param {number} [minBucketBytes=WEBGPU_BUFFER_COPY_ALIGNMENT]
   * @param {number} [maxBucketBytes=DEFAULT_POW2_MAX_BUCKET_BYTES]
   * @returns {number}
   */
  function selectPowerOfTwoBucket(
    byteSize,
    minBucketBytes = WEBGPU_BUFFER_COPY_ALIGNMENT,
    maxBucketBytes = DEFAULT_POW2_MAX_BUCKET_BYTES
  ) {
    assertPositiveInteger(byteSize, 'byteSize');
    assertPositiveInteger(minBucketBytes, 'minBucketBytes');
    assertPositiveInteger(maxBucketBytes, 'maxBucketBytes');
    if (minBucketBytes > maxBucketBytes) {
      throw new RangeError(
        `minBucketBytes (${minBucketBytes}) must be <= maxBucketBytes (${maxBucketBytes})`
      );
    }
    const aligned = alignBufferSize(byteSize);
    if (aligned > maxBucketBytes) {
      return aligned;
    }
    const floor = Math.max(WEBGPU_BUFFER_COPY_ALIGNMENT, nextPowerOfTwo(minBucketBytes));
    const pow2 = Math.max(floor, nextPowerOfTwo(aligned));
    return pow2 <= maxBucketBytes ? pow2 : aligned;
  }

  /**
   * Deterministic LRU eviction comparator for idle buffers/slabs:
   *  1. Oldest `lastUsedFrame` first (LRU across frames),
   *  2. Largest `size` first on same-frame tie (reclaims oversized VRAM fastest),
   *  3. Oldest monotonic `lastUsedSeq` first on same-frame, same-size tie (LRU within bucket),
   *  4. Lowest `id` as final deterministic tie-breaker.
   */
  function compareEvictionLru(a, b) {
    if (a.lastUsedFrame !== b.lastUsedFrame) {
      return a.lastUsedFrame - b.lastUsedFrame;
    }
    if (a.size !== b.size) {
      return b.size - a.size;
    }
    if (a.lastUsedSeq !== b.lastUsedSeq) {
      return a.lastUsedSeq - b.lastUsedSeq;
    }
    return a.id - b.id;
  }

  /**
   * Select the canonical large-slab bucket byte size for a requested allocation.
   * Returns the smallest bucket `>= alignBufferSize(byteSize)`, or `alignBufferSize(byteSize)`
   * if the request exceeds the largest configured bucket.
   *
   * @param {number} byteSize
   * @param {readonly number[]} [buckets=DEFAULT_LARGE_SLAB_BUCKETS]
   * @returns {number}
   */
  function selectLargeSlabBucket(byteSize, buckets = DEFAULT_LARGE_SLAB_BUCKETS) {
    assertPositiveInteger(byteSize, 'byteSize');
    const aligned = alignBufferSize(byteSize);
    for (let i = 0; i < buckets.length; i += 1) {
      if (buckets[i] >= aligned) {
        return buckets[i];
      }
    }
    return aligned;
  }

  function defaultNowMs() {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
      return performance.now();
    }
    if (typeof Date !== 'undefined' && typeof Date.now === 'function') {
      return Date.now();
    }
    return 0;
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
        maxPoolBytes,
        maxPooledBytes,
        framesInFlight,
        maxFramesInFlight,
        mode = STAGING_MODE.WRITE_BUFFER,
        minWarmChunks = DEFAULT_MIN_WARM_CHUNKS,
        idleEvictFrames = DEFAULT_IDLE_EVICT_FRAMES,
        idleShrinkFrames = DEFAULT_IDLE_SHRINK_FRAMES,
        maxLargeSlabs = DEFAULT_MAX_LARGE_SLABS,
        maxSlabsPerBucket,
        maxPerBucket,
        bucketStrategy,
        powerOfTwoBuckets,
        maxBucketBytes = DEFAULT_POW2_MAX_BUCKET_BYTES,
        largeSlabMinBytes,
        largeSlabBuckets,
        now,
      } = options;

      if (!device || typeof device.createBuffer !== 'function') {
        throw new TypeError('GpuStagingPool requires a WebGPU device with createBuffer()');
      }

      if (maxPoolBytes !== undefined) {
        assertPositiveInteger(maxPoolBytes, 'maxPoolBytes');
      }
      if (maxPooledBytes !== undefined) {
        assertPositiveInteger(maxPooledBytes, 'maxPooledBytes');
      }
      if (
        maxPoolBytes !== undefined &&
        maxPooledBytes !== undefined &&
        maxPoolBytes !== maxPooledBytes
      ) {
        throw new RangeError(
          `Conflicting maxPoolBytes (${maxPoolBytes}) and maxPooledBytes (${maxPooledBytes})`
        );
      }
      const resolvedMaxPoolBytes =
        maxPoolBytes !== undefined
          ? maxPoolBytes
          : maxPooledBytes !== undefined
            ? maxPooledBytes
            : DEFAULT_MAX_POOL_BYTES;

      if (framesInFlight !== undefined) {
        assertPositiveInteger(framesInFlight, 'framesInFlight');
      }
      if (maxFramesInFlight !== undefined) {
        assertPositiveInteger(maxFramesInFlight, 'maxFramesInFlight');
      }
      if (
        framesInFlight !== undefined &&
        maxFramesInFlight !== undefined &&
        framesInFlight !== maxFramesInFlight
      ) {
        throw new RangeError(
          `Conflicting framesInFlight (${framesInFlight}) and maxFramesInFlight (${maxFramesInFlight})`
        );
      }
      const resolvedFramesInFlight =
        maxFramesInFlight !== undefined
          ? maxFramesInFlight
          : framesInFlight !== undefined
            ? framesInFlight
            : DEFAULT_FRAMES_IN_FLIGHT;

      assertNonNegativeInteger(minWarmChunks, 'minWarmChunks');
      assertNonNegativeInteger(idleEvictFrames, 'idleEvictFrames');
      assertNonNegativeInteger(idleShrinkFrames, 'idleShrinkFrames');
      assertNonNegativeInteger(maxLargeSlabs, 'maxLargeSlabs');
      assertPositiveInteger(maxBucketBytes, 'maxBucketBytes');

      if (maxSlabsPerBucket !== undefined && maxSlabsPerBucket !== null) {
        assertNonNegativeInteger(maxSlabsPerBucket, 'maxSlabsPerBucket');
      }
      if (maxPerBucket !== undefined && maxPerBucket !== null) {
        assertNonNegativeInteger(maxPerBucket, 'maxPerBucket');
      }
      if (
        maxSlabsPerBucket !== undefined &&
        maxSlabsPerBucket !== null &&
        maxPerBucket !== undefined &&
        maxPerBucket !== null &&
        maxSlabsPerBucket !== maxPerBucket
      ) {
        throw new RangeError(
          `Conflicting maxSlabsPerBucket (${maxSlabsPerBucket}) and maxPerBucket (${maxPerBucket})`
        );
      }
      const resolvedMaxSlabsPerBucket =
        maxSlabsPerBucket !== undefined && maxSlabsPerBucket !== null
          ? maxSlabsPerBucket
          : maxPerBucket !== undefined && maxPerBucket !== null
            ? maxPerBucket
            : null;

      if (powerOfTwoBuckets !== undefined && typeof powerOfTwoBuckets !== 'boolean') {
        throw new TypeError('powerOfTwoBuckets must be a boolean');
      }
      let resolvedBucketStrategy;
      if (bucketStrategy !== undefined) {
        if (
          bucketStrategy !== BUCKET_STRATEGY.DEFAULT &&
          bucketStrategy !== BUCKET_STRATEGY.POW2
        ) {
          throw new RangeError(
            `Unsupported bucketStrategy: ${JSON.stringify(bucketStrategy)} (expected 'default' or 'pow2')`
          );
        }
        if (powerOfTwoBuckets === true && bucketStrategy !== BUCKET_STRATEGY.POW2) {
          throw new RangeError(
            `Conflicting powerOfTwoBuckets=true and bucketStrategy=${JSON.stringify(bucketStrategy)}`
          );
        }
        if (powerOfTwoBuckets === false && bucketStrategy === BUCKET_STRATEGY.POW2) {
          throw new RangeError(
            `Conflicting powerOfTwoBuckets=false and bucketStrategy=${JSON.stringify(bucketStrategy)}`
          );
        }
        resolvedBucketStrategy = bucketStrategy;
      } else {
        resolvedBucketStrategy =
          powerOfTwoBuckets === true ? BUCKET_STRATEGY.POW2 : BUCKET_STRATEGY.DEFAULT;
      }

      if (mode !== STAGING_MODE.WRITE_BUFFER && mode !== STAGING_MODE.MAPPED) {
        throw new RangeError(
          `Unsupported staging mode: ${JSON.stringify(mode)} (expected 'writeBuffer' or 'mapped')`
        );
      }

      if (now !== undefined && typeof now !== 'function') {
        throw new TypeError('now must be a function returning a timestamp in milliseconds');
      }

      let resolvedLargeSlabBuckets;
      let resolvedLargeSlabMinBytes;
      if (largeSlabBuckets !== undefined) {
        resolvedLargeSlabBuckets = normalizeLargeSlabBuckets(largeSlabBuckets);
        if (largeSlabMinBytes !== undefined) {
          assertPositiveInteger(largeSlabMinBytes, 'largeSlabMinBytes');
          resolvedLargeSlabMinBytes = largeSlabMinBytes;
        } else if (resolvedLargeSlabBuckets.length > 0) {
          resolvedLargeSlabMinBytes = resolvedLargeSlabBuckets[0];
        } else {
          resolvedLargeSlabMinBytes = DEFAULT_LARGE_SLAB_THRESHOLD;
        }
      } else if (resolvedBucketStrategy === BUCKET_STRATEGY.POW2) {
        if (largeSlabMinBytes !== undefined) {
          assertPositiveInteger(largeSlabMinBytes, 'largeSlabMinBytes');
          resolvedLargeSlabMinBytes = largeSlabMinBytes;
        } else {
          resolvedLargeSlabMinBytes = Math.min(DEFAULT_POW2_MIN_SLAB_BYTES, maxBucketBytes);
        }
        resolvedLargeSlabBuckets = buildPowerOfTwoBuckets(
          resolvedLargeSlabMinBytes,
          maxBucketBytes
        );
      } else {
        resolvedLargeSlabBuckets = DEFAULT_LARGE_SLAB_BUCKETS;
        if (largeSlabMinBytes !== undefined) {
          assertPositiveInteger(largeSlabMinBytes, 'largeSlabMinBytes');
          resolvedLargeSlabMinBytes = largeSlabMinBytes;
        } else {
          resolvedLargeSlabMinBytes = DEFAULT_LARGE_SLAB_THRESHOLD;
        }
      }

      let resolvedChunkSize;
      let initialAlignPads = 0;
      if (chunkSize === undefined) {
        const clampedMax = Math.max(
          WEBGPU_BUFFER_COPY_ALIGNMENT,
          Math.floor(resolvedMaxPoolBytes / WEBGPU_BUFFER_COPY_ALIGNMENT) *
            WEBGPU_BUFFER_COPY_ALIGNMENT
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
      this.maxPoolBytes = resolvedMaxPoolBytes;
      this.maxPooledBytes = resolvedMaxPoolBytes;
      this.framesInFlight = resolvedFramesInFlight;
      this.maxFramesInFlight = resolvedFramesInFlight;
      this.mode = mode;
      this.minWarmChunks = minWarmChunks;
      this.idleEvictFrames = idleEvictFrames;
      this.idleShrinkFrames = idleShrinkFrames;
      this.maxLargeSlabs = maxLargeSlabs;
      this.maxSlabsPerBucket = resolvedMaxSlabsPerBucket;
      this.bucketStrategy = resolvedBucketStrategy;
      this.maxBucketBytes = maxBucketBytes;
      this.largeSlabMinBytes = resolvedLargeSlabMinBytes;
      this.largeSlabBuckets = resolvedLargeSlabBuckets;
      this._now = now || defaultNowMs;

      this._destroyed = false;
      this._currentFrame = -1;
      this._inFrame = false;
      this._frameStartedAtLeastOnce = false;
      this._nextChunkId = 1;
      this._lruSeq = 0;

      this._allChunks = new Set();
      this._chunkByBuffer = new Map();
      this._activeChunks = [];
      this._recycledChunks = [];
      this._largeSlabBuckets = new Map();
      this._idleLargeSlabCount = 0;
      this._inFlightChunks = [];
      this._currentChunk = null;
      this._pendingMapPromises = new Set();
      this._lastAcquireFresh = false;

      // Pool-wide telemetry.
      this._totalChunksAllocated = 0;
      this._totalBytesAllocated = 0;
      this._allocatedBytes = 0;
      this._peakAllocatedBytes = 0;
      this._peakSingleAllocationBytes = 0;
      this._largeSlabBytes = 0;
      this._reclaimedChunks = 0;
      this._reclaimedBytes = 0;
      this._evictedCount = 0;
      this._evictedBytes = 0;
      this._oversizeAllocations = 0;
      this._bytesUploaded = 0;
      this._totalBytesStaged = 0;
      this._bytesReused = 0;
      this._allocationCount = 0;
      this._bufferReuseHits = 0;
      this._bufferReuseMisses = 0;
      this._largeSlabHits = 0;
      this._largeSlabMisses = 0;
      this._largeSlabEvictions = 0;
      this._reusedAllocations = 0;
      this._freshAllocations = 0;
      this._alignPadCount = initialAlignPads;
      this._sabCopyCount = 0;
      this._doubleReleaseIgnored = 0;
      this._subAllocCoalesced = 0;
      this._idleSlabShrinks = 0;
      this._stallAvoidedCount = 0;

      // Upload throughput & per-frame telemetry.
      this._framesCompleted = 0;
      this._lastFrameBytesUploaded = 0;
      this._peakFrameBytesUploaded = 0;
      this._firstActivityTimeMs = null;
      this._lastActivityTimeMs = null;
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

    _recordActivityTime() {
      const t = this._now();
      if (typeof t === 'number' && Number.isFinite(t)) {
        if (this._firstActivityTimeMs === null) {
          this._firstActivityTimeMs = t;
        }
        if (this._lastActivityTimeMs === null || t >= this._lastActivityTimeMs) {
          this._lastActivityTimeMs = t;
        }
      }
    }

    _getRecycledCount() {
      return this._recycledChunks.length + this._idleLargeSlabCount;
    }

    _resolveLargeSlabBucket(alignedSize) {
      if (this.largeSlabBuckets.length === 0 || alignedSize < this.largeSlabMinBytes) {
        return 0;
      }
      let bucketSize = selectLargeSlabBucket(alignedSize, this.largeSlabBuckets);
      if (bucketSize > this.maxPoolBytes && alignedSize <= this.maxPoolBytes) {
        bucketSize = alignedSize;
      }
      return bucketSize;
    }

    _putIntoRecycled(chunk) {
      chunk.cursor = 0;
      chunk.subAllocCount = 0;
      chunk.retiredAtFrame = -1;
      chunk.gpuDone = false;
      chunk.readyToRecycle = false;
      chunk.state = 'recycled';
      if (chunk.isLargeSlab) {
        let bucket = this._largeSlabBuckets.get(chunk.slabBucketSize);
        if (!bucket) {
          bucket = [];
          this._largeSlabBuckets.set(chunk.slabBucketSize, bucket);
        }
        bucket.push(chunk);
        this._idleLargeSlabCount += 1;
      } else {
        this._recycledChunks.push(chunk);
      }
    }

    _enforceLargeSlabCap() {
      let reclaimedChunks = 0;
      let reclaimedBytes = 0;

      // 1. Enforce optional per-bucket cap via deterministic LRU eviction within each bucket.
      if (this.maxSlabsPerBucket !== null) {
        for (const bucket of Array.from(this._largeSlabBuckets.values())) {
          if (bucket.length <= this.maxSlabsPerBucket) {
            continue;
          }
          const bucketSlabs = [];
          for (let i = 0; i < bucket.length; i += 1) {
            const slab = bucket[i];
            if (slab && !slab.destroyed && slab.state === 'recycled') {
              bucketSlabs.push(slab);
            }
          }
          if (bucketSlabs.length <= this.maxSlabsPerBucket) {
            continue;
          }
          bucketSlabs.sort(compareEvictionLru);
          const toEvict = bucketSlabs.length - this.maxSlabsPerBucket;
          for (let i = 0; i < toEvict; i += 1) {
            const slab = bucketSlabs[i];
            if (slab.destroyed || slab.state !== 'recycled') {
              continue;
            }
            const byteSize = slab.size;
            this._destroyChunk(slab);
            reclaimedChunks += 1;
            reclaimedBytes += byteSize;
            this._largeSlabEvictions += 1;
            this._evictedCount += 1;
            this._evictedBytes += byteSize;
          }
        }
      }

      // 2. Enforce global idle large-slab cap via deterministic LRU (largest first on same-frame tie).
      if (this._idleLargeSlabCount > this.maxLargeSlabs) {
        const idleSlabs = [];
        for (const bucket of this._largeSlabBuckets.values()) {
          for (let i = 0; i < bucket.length; i += 1) {
            const slab = bucket[i];
            if (slab && !slab.destroyed && slab.state === 'recycled') {
              idleSlabs.push(slab);
            }
          }
        }

        idleSlabs.sort(compareEvictionLru);

        for (
          let i = 0;
          i < idleSlabs.length && this._idleLargeSlabCount > this.maxLargeSlabs;
          i += 1
        ) {
          const slab = idleSlabs[i];
          if (slab.destroyed || slab.state !== 'recycled') {
            continue;
          }
          const byteSize = slab.size;
          this._destroyChunk(slab);
          reclaimedChunks += 1;
          reclaimedBytes += byteSize;
          this._largeSlabEvictions += 1;
          this._evictedCount += 1;
          this._evictedBytes += byteSize;
        }
      }

      if (reclaimedChunks > 0) {
        this._reclaimedChunks += reclaimedChunks;
        this._reclaimedBytes += reclaimedBytes;
      }

      return { reclaimedChunks, reclaimedBytes };
    }

    _createChunk(byteSize, oversized, slabBucketSize = 0) {
      const allocSize = alignBufferSize(byteSize);
      this._ensureBudget(allocSize, byteSize);

      const id = this._nextChunkId++;
      const isLargeSlab = slabBucketSize > 0;
      const isMappedMode = this.mode === STAGING_MODE.MAPPED;
      const label = isLargeSlab ? `playgta6-staging-slab-${id}` : `playgta6-staging-${id}`;
      const descriptor = isMappedMode
        ? {
            label,
            size: allocSize,
            usage: BUFFER_USAGE.MAP_WRITE | BUFFER_USAGE.COPY_SRC,
            mappedAtCreation: true,
          }
        : {
            label,
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
        subAllocCount: 0,
        oversized,
        isLargeSlab,
        slabBucketSize,
        state: 'active',
        lastUsedFrame: this._currentFrame < 0 ? 0 : this._currentFrame,
        lastUsedSeq: ++this._lruSeq,
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
      this._totalBytesAllocated += allocSize;
      this._allocatedBytes += allocSize;
      if (isLargeSlab) {
        this._largeSlabBytes += allocSize;
      }
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
      if (chunk.isLargeSlab) {
        const bucket = this._largeSlabBuckets.get(chunk.slabBucketSize);
        if (bucket) {
          const bucketIdx = bucket.indexOf(chunk);
          if (bucketIdx !== -1) {
            bucket.splice(bucketIdx, 1);
            this._idleLargeSlabCount = Math.max(0, this._idleLargeSlabCount - 1);
            if (bucket.length === 0) {
              this._largeSlabBuckets.delete(chunk.slabBucketSize);
            }
          }
        }
        this._largeSlabBytes = Math.max(0, this._largeSlabBytes - chunk.size);
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

    _pickBudgetEvictionCandidate() {
      // 1. Prefer evicting idle non-bucketed oversized burst chunks via deterministic LRU
      //    (oldest frame first, largest first on tie) so steady-state standard chunks stay warm.
      let bestOversized = null;
      for (let i = 0; i < this._recycledChunks.length; i += 1) {
        const candidate = this._recycledChunks[i];
        if (!candidate || candidate.destroyed || candidate.state !== 'recycled' || !candidate.oversized) {
          continue;
        }
        if (!bestOversized || compareEvictionLru(candidate, bestOversized) < 0) {
          bestOversized = candidate;
        }
      }
      if (bestOversized) {
        return bestOversized;
      }

      // 2. Next evict idle large slabs via deterministic LRU (oldest used first, then largest first).
      let bestSlab = null;
      for (const bucket of this._largeSlabBuckets.values()) {
        for (let i = 0; i < bucket.length; i += 1) {
          const slab = bucket[i];
          if (!slab || slab.destroyed || slab.state !== 'recycled') {
            continue;
          }
          if (!bestSlab || compareEvictionLru(slab, bestSlab) < 0) {
            bestSlab = slab;
          }
        }
      }
      if (bestSlab) {
        return bestSlab;
      }

      // 3. Finally evict the LRU idle standard chunk.
      let bestStd = null;
      for (let i = 0; i < this._recycledChunks.length; i += 1) {
        const candidate = this._recycledChunks[i];
        if (!candidate || candidate.destroyed || candidate.state !== 'recycled') {
          continue;
        }
        if (!bestStd || compareEvictionLru(candidate, bestStd) < 0) {
          bestStd = candidate;
        }
      }
      return bestStd;
    }

    _reclaimCompletedInFlight(enforceCap = true) {
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
              this._putIntoRecycled(chunk);
              reclaimed += 1;
            } else {
              stillInFlight.push(chunk);
            }
          } else {
            this._putIntoRecycled(chunk);
            reclaimed += 1;
          }
        } else {
          stillInFlight.push(chunk);
        }
      }
      this._inFlightChunks = stillInFlight;
      if (enforceCap && reclaimed > 0) {
        this._enforceLargeSlabCap();
      }
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
      const recycled = this._reclaimCompletedInFlight(true);
      if (this.idleShrinkFrames > 0) {
        this.shrinkIdleOversizedSlabs(this.idleShrinkFrames);
      }
      if (this.idleEvictFrames > 0) {
        this.reclaimIdleChunks(this.idleEvictFrames);
      }
      return recycled;
    }

    /**
     * Destroy and remove idle recycled oversized / large staging slabs
     * (`chunk.oversized || chunk.isLargeSlab`) whose
     * `currentFrame - chunk.lastUsedFrame >= maxIdleFrames`.
     *
     * Keeps standard steady-state bump-allocation chunks warm while releasing
     * transient boot-time or cutscene oversized slabs to keep GPU process RAM bounded.
     * Never touches active or in-flight slabs.
     *
     * @param {number} [maxIdleFrames]
     * @returns {{ shrunkSlabs: number, shrunkBytes: number }}
     */
    shrinkIdleOversizedSlabs(maxIdleFrames) {
      this._assertNotDestroyed();

      let threshold;
      if (maxIdleFrames === undefined) {
        if (this.idleShrinkFrames <= 0) {
          return { shrunkSlabs: 0, shrunkBytes: 0 };
        }
        threshold = this.idleShrinkFrames;
      } else {
        assertNonNegativeInteger(maxIdleFrames, 'maxIdleFrames');
        threshold = maxIdleFrames;
      }

      const currentFrame = this._currentFrame < 0 ? 0 : this._currentFrame;
      const candidates = [];

      for (let i = 0; i < this._recycledChunks.length; i += 1) {
        const chunk = this._recycledChunks[i];
        if (
          chunk &&
          !chunk.destroyed &&
          chunk.state === 'recycled' &&
          (chunk.oversized || chunk.isLargeSlab) &&
          currentFrame - chunk.lastUsedFrame >= threshold
        ) {
          candidates.push(chunk);
        }
      }

      for (const bucket of this._largeSlabBuckets.values()) {
        for (let i = 0; i < bucket.length; i += 1) {
          const slab = bucket[i];
          if (
            slab &&
            !slab.destroyed &&
            slab.state === 'recycled' &&
            (slab.oversized || slab.isLargeSlab) &&
            currentFrame - slab.lastUsedFrame >= threshold
          ) {
            candidates.push(slab);
          }
        }
      }

      if (candidates.length === 0) {
        return { shrunkSlabs: 0, shrunkBytes: 0 };
      }

      candidates.sort(compareEvictionLru);

      let shrunkSlabs = 0;
      let shrunkBytes = 0;
      for (let i = 0; i < candidates.length; i += 1) {
        const chunk = candidates[i];
        if (chunk.destroyed || chunk.state !== 'recycled') {
          continue;
        }
        const byteSize = chunk.size;
        const wasLargeSlab = chunk.isLargeSlab;
        this._destroyChunk(chunk);
        if (wasLargeSlab) {
          this._largeSlabEvictions += 1;
        }
        this._evictedCount += 1;
        this._evictedBytes += byteSize;
        this._idleSlabShrinks += 1;
        shrunkSlabs += 1;
        shrunkBytes += byteSize;
      }

      if (shrunkSlabs > 0) {
        this._reclaimedChunks += shrunkSlabs;
        this._reclaimedBytes += shrunkBytes;
      }

      return { shrunkSlabs, shrunkBytes };
    }

    /**
     * Destroy and remove idle recycled (not active, not in-flight) staging chunks
     * whose `currentFrame - chunk.lastUsedFrame >= maxIdleFrames`, while keeping
     * at least `minWarmChunks` warm chunks in the recycled pool.
     *
     * Prefers evicting oversized burst chunks / large slabs (largest first) and
     * oldest chunks first so steady-state standard chunks stay warm.
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

      if (this._getRecycledCount() <= this.minWarmChunks) {
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
      for (const bucket of this._largeSlabBuckets.values()) {
        for (let i = 0; i < bucket.length; i += 1) {
          const chunk = bucket[i];
          if (
            chunk &&
            !chunk.destroyed &&
            chunk.state === 'recycled' &&
            currentFrame - chunk.lastUsedFrame >= threshold
          ) {
            candidates.push(chunk);
          }
        }
      }

      if (candidates.length === 0) {
        return { reclaimedChunks: 0, reclaimedBytes: 0 };
      }

      candidates.sort((a, b) => {
        const aLarge = a.oversized || a.isLargeSlab;
        const bLarge = b.oversized || b.isLargeSlab;
        if (aLarge !== bLarge) {
          return aLarge ? -1 : 1;
        }
        if (aLarge && a.size !== b.size) {
          return b.size - a.size;
        }
        if (a.lastUsedFrame !== b.lastUsedFrame) {
          return a.lastUsedFrame - b.lastUsedFrame;
        }
        if (a.lastUsedSeq !== b.lastUsedSeq) {
          return a.lastUsedSeq - b.lastUsedSeq;
        }
        return a.id - b.id;
      });

      let reclaimedChunks = 0;
      let reclaimedBytes = 0;
      for (let i = 0; i < candidates.length; i += 1) {
        if (this._getRecycledCount() <= this.minWarmChunks) {
          break;
        }
        const chunk = candidates[i];
        if (chunk.destroyed || chunk.state !== 'recycled') {
          continue;
        }
        const byteSize = chunk.size;
        const wasLargeSlab = chunk.isLargeSlab;
        this._destroyChunk(chunk);
        if (wasLargeSlab) {
          this._largeSlabEvictions += 1;
        }
        this._evictedCount += 1;
        this._evictedBytes += byteSize;
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
      const reclaimedInFlight = this._reclaimCompletedInFlight(false);

      // Check whether evicting idle recycled chunks / idle large slabs can make room.
      let reclaimable = 0;
      for (let i = 0; i < this._recycledChunks.length; i += 1) {
        reclaimable += this._recycledChunks[i].size;
      }
      for (const bucket of this._largeSlabBuckets.values()) {
        for (let i = 0; i < bucket.length; i += 1) {
          reclaimable += bucket[i].size;
        }
      }

      if (this._allocatedBytes - reclaimable + allocSize <= this.maxPoolBytes) {
        while (this._allocatedBytes + allocSize > this.maxPoolBytes) {
          const evicted = this._pickBudgetEvictionCandidate();
          if (!evicted) {
            break;
          }
          const wasLargeSlab = evicted.isLargeSlab;
          const byteSize = evicted.size;
          this._destroyChunk(evicted);
          this._evictedCount += 1;
          this._evictedBytes += byteSize;
          if (wasLargeSlab) {
            this._largeSlabEvictions += 1;
          }
        }
        if (reclaimedInFlight > 0) {
          this._enforceLargeSlabCap();
        }
        return;
      }

      if (reclaimedInFlight > 0) {
        this._enforceLargeSlabCap();
      }

      throw new RangeError(
        `GpuStagingPool budget exceeded: cannot allocate ${allocSize}-byte chunk ` +
          `(requested ${requestedSize} bytes; allocated=${this._allocatedBytes}, ` +
          `maxPoolBytes=${this.maxPoolBytes})`
      );
    }

    _takeLargeSlab(bucketSize) {
      const bucket = this._largeSlabBuckets.get(bucketSize);
      if (!bucket || bucket.length === 0) {
        return null;
      }
      while (bucket.length > 0) {
        const slab = bucket.pop();
        this._idleLargeSlabCount = Math.max(0, this._idleLargeSlabCount - 1);
        if (bucket.length === 0) {
          this._largeSlabBuckets.delete(bucketSize);
        }
        if (slab && !slab.destroyed && slab.state === 'recycled') {
          slab.cursor = 0;
          slab.subAllocCount = 0;
          slab.lastUsedFrame = this._currentFrame < 0 ? 0 : this._currentFrame;
          slab.lastUsedSeq = ++this._lruSeq;
          slab.retiredAtFrame = -1;
          slab.gpuDone = false;
          slab.readyToRecycle = false;
          slab.state = 'active';
          return slab;
        }
      }
      return null;
    }

    _acquireLargeSlab(bucketSize, oversized) {
      let recycled = this._takeLargeSlab(bucketSize);

      if (!recycled && this._reclaimCompletedInFlight(false) > 0) {
        recycled = this._takeLargeSlab(bucketSize);
        this._enforceLargeSlabCap();
      }

      if (recycled) {
        this._bufferReuseHits += 1;
        this._largeSlabHits += 1;
        this._bytesReused += recycled.size;
        this._activeChunks.push(recycled);
        this._lastAcquireFresh = false;
        if (this._inFlightChunks.length > 0) {
          this._stallAvoidedCount += 1;
        }
        return recycled;
      }

      const created = this._createChunk(bucketSize, oversized, bucketSize);
      this._bufferReuseMisses += 1;
      this._largeSlabMisses += 1;
      this._activeChunks.push(created);
      this._lastAcquireFresh = true;
      if (this._inFlightChunks.length > 0) {
        this._stallAvoidedCount += 1;
      }
      return created;
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

      const chunk = this._recycledChunks[bestIndex];
      for (let i = bestIndex + 1; i < this._recycledChunks.length; i += 1) {
        this._recycledChunks[i - 1] = this._recycledChunks[i];
      }
      this._recycledChunks.pop();
      chunk.cursor = 0;
      chunk.subAllocCount = 0;
      chunk.lastUsedFrame = this._currentFrame < 0 ? 0 : this._currentFrame;
      chunk.lastUsedSeq = ++this._lruSeq;
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
      if (!recycled && this._reclaimCompletedInFlight(false) > 0) {
        recycled = this._takeRecycledChunk(minRequired, oversized);
        this._enforceLargeSlabCap();
      }

      if (recycled) {
        this._bufferReuseHits += 1;
        this._bytesReused += recycled.size;
        this._activeChunks.push(recycled);
        this._lastAcquireFresh = false;
        if (this._inFlightChunks.length > 0) {
          this._stallAvoidedCount += 1;
        }
        return recycled;
      }

      const created = this._createChunk(targetSize, oversized, 0);
      this._bufferReuseMisses += 1;
      this._activeChunks.push(created);
      this._lastAcquireFresh = true;
      if (this._inFlightChunks.length > 0) {
        this._stallAvoidedCount += 1;
      }
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
      this._putIntoRecycled(chunk);
      if (chunk.isLargeSlab) {
        this._enforceLargeSlabCap();
      }
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
      this._recordActivityTime();
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
              this._putIntoRecycled(chunk);
              recycledCount += 1;
            } else {
              stillInFlight.push(chunk);
            }
          } else {
            this._putIntoRecycled(chunk);
            recycledCount += 1;
          }
        } else {
          stillInFlight.push(chunk);
        }
      }

      this._inFlightChunks = stillInFlight;

      const capReclaim = this._enforceLargeSlabCap();
      let shrinkReclaim = { shrunkSlabs: 0, shrunkBytes: 0 };
      if (this.idleShrinkFrames > 0) {
        shrinkReclaim = this.shrinkIdleOversizedSlabs(this.idleShrinkFrames);
      }
      let idleReclaim = { reclaimedChunks: 0, reclaimedBytes: 0 };
      if (this.idleEvictFrames > 0) {
        idleReclaim = this.reclaimIdleChunks(this.idleEvictFrames);
      }

      return {
        frameIndex: this._currentFrame,
        recycledCount,
        recycledChunks: this._getRecycledCount(),
        inFlightChunks: this._inFlightChunks.length,
        reclaimedChunks:
          capReclaim.reclaimedChunks + shrinkReclaim.shrunkSlabs + idleReclaim.reclaimedChunks,
        reclaimedBytes:
          capReclaim.reclaimedBytes + shrinkReclaim.shrunkBytes + idleReclaim.reclaimedBytes,
        shrunkSlabs: shrinkReclaim.shrunkSlabs,
        shrunkBytes: shrinkReclaim.shrunkBytes,
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

      const isDuplicateEnd = wasNotInFrame && retiredChunks.length === 0;
      if (isDuplicateEnd) {
        this._doubleReleaseIgnored += 1;
      } else {
        this._framesCompleted += 1;
        this._lastFrameBytesUploaded = this._frameBytesUploaded;
        if (this._frameBytesUploaded > this._peakFrameBytesUploaded) {
          this._peakFrameBytesUploaded = this._frameBytesUploaded;
        }
        this._recordActivityTime();
      }

      this._attachWorkDoneFence(retiredChunks);

      const capReclaim = this._enforceLargeSlabCap();
      let shrinkReclaim = { shrunkSlabs: 0, shrunkBytes: 0 };
      if (this.idleShrinkFrames > 0) {
        shrinkReclaim = this.shrinkIdleOversizedSlabs(this.idleShrinkFrames);
      }
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
        recycledChunks: this._getRecycledCount(),
        reclaimedChunks:
          capReclaim.reclaimedChunks + shrinkReclaim.shrunkSlabs + idleReclaim.reclaimedChunks,
        reclaimedBytes:
          capReclaim.reclaimedBytes + shrinkReclaim.shrunkBytes + idleReclaim.reclaimedBytes,
        shrunkSlabs: shrinkReclaim.shrunkSlabs,
        shrunkBytes: shrinkReclaim.shrunkBytes,
      };

      this._inFrame = false;
      this._frameAllocations = 0;
      this._frameBytesAllocated = 0;
      this._frameBytesUploaded = 0;

      return frameStats;
    }

    /**
     * Sub-allocate an aligned staging slice from the current active chunk, rolling
     * over to a recycled or newly allocated chunk (or a persistent large-slab bucket /
     * dedicated oversized chunk) within `maxPoolBytes`.
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
      const slabBucketSize = this._resolveLargeSlabBucket(alignedSize);
      const isLargeSlab = slabBucketSize > 0;
      let chunk;
      let offset;
      let isFreshBuffer = false;

      if (isLargeSlab) {
        // Persistent large-slab bucket (e.g., 2–16 MiB 4K Bink2 YUV planes);
        // preserves _currentChunk for smaller sub-allocations.
        chunk = this._acquireLargeSlab(slabBucketSize, isOversized);
        isFreshBuffer = this._lastAcquireFresh;
        offset = 0;
        chunk.cursor = alignedSize;
        chunk.subAllocCount = 1;
      } else if (isOversized) {
        // Dedicated oversized chunk; preserve _currentChunk for smaller allocations.
        chunk = this._acquireChunk(alignedSize, true);
        isFreshBuffer = this._lastAcquireFresh;
        offset = 0;
        chunk.cursor = alignedSize;
        chunk.subAllocCount = 1;
      } else {
        if (this._currentChunk && !this._currentChunk.destroyed && this._currentChunk.state === 'active') {
          const candidateOffset = alignUp(this._currentChunk.cursor, effectiveAlignment);
          if (candidateOffset + alignedSize <= this._currentChunk.size) {
            chunk = this._currentChunk;
            offset = candidateOffset;
            chunk.cursor = offset + alignedSize;
            chunk.subAllocCount += 1;
            isFreshBuffer = false;
            this._subAllocCoalesced += 1;
          }
        }

        if (!chunk) {
          chunk = this._acquireChunk(alignedSize, false);
          isFreshBuffer = this._lastAcquireFresh;
          offset = 0;
          chunk.cursor = alignedSize;
          chunk.subAllocCount = 1;
          this._currentChunk = chunk;
        }
      }

      chunk.lastUsedFrame = this._currentFrame;
      chunk.lastUsedSeq = ++this._lruSeq;

      if (isOversized) {
        this._oversizeAllocations += 1;
      }

      if (alignedSize !== byteSize) {
        this._alignPadCount += 1;
      }

      if (isFreshBuffer) {
        this._freshAllocations += 1;
      } else {
        this._reusedAllocations += 1;
      }

      if (byteSize > this._peakSingleAllocationBytes) {
        this._peakSingleAllocationBytes = byteSize;
      }

      this._allocationCount += 1;
      this._totalBytesStaged += byteSize;
      this._frameAllocations += 1;
      this._frameBytesAllocated += byteSize;

      const slice = {
        buffer: chunk.buffer,
        offset,
        size: byteSize,
        alignedSize,
      };
      if (isLargeSlab) {
        slice.isLargeSlab = true;
        slice.slabBucketSize = chunk.slabBucketSize;
      }
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
      this._recordActivityTime();

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
      this._recordActivityTime();

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
     * Return pool-wide allocation, large-slab reuse, and upload bandwidth statistics.
     */
    getStats() {
      const activeCount = this._activeChunks.length;
      const recycledCount = this._getRecycledCount();
      let activeLargeSlabs = 0;
      for (let i = 0; i < this._activeChunks.length; i += 1) {
        if (this._activeChunks[i].isLargeSlab) {
          activeLargeSlabs += 1;
        }
      }
      let inFlightLargeSlabs = 0;
      for (let i = 0; i < this._inFlightChunks.length; i += 1) {
        if (this._inFlightChunks[i].isLargeSlab) {
          inFlightLargeSlabs += 1;
        }
      }
      const largeSlabBucketCounts = {};
      for (const [bucketSize, bucket] of this._largeSlabBuckets.entries()) {
        if (bucket.length > 0) {
          largeSlabBucketCounts[bucketSize] = bucket.length;
        }
      }

      const poolHits = sanitizeNonNegativeCounter(this._bufferReuseHits);
      const poolMisses = sanitizeNonNegativeCounter(this._bufferReuseMisses);
      const poolLookups = poolHits + poolMisses;
      const hitRate = clampUnitRate(poolHits, poolLookups);
      const largeSlabHits = sanitizeNonNegativeCounter(this._largeSlabHits);
      const largeSlabMisses = sanitizeNonNegativeCounter(this._largeSlabMisses);
      const largeSlabTotal = largeSlabHits + largeSlabMisses;
      const peakAllocatedBytes = sanitizeNonNegativeCounter(this._peakAllocatedBytes);
      const lruEvictions = sanitizeNonNegativeCounter(this._evictedCount);
      const oversizedAllocations = sanitizeNonNegativeCounter(this._oversizeAllocations);
      const totalBytesStaged = sanitizeNonNegativeCounter(this._totalBytesStaged);
      const bytesUploaded = sanitizeNonNegativeCounter(this._bytesUploaded);
      const allocationCount = sanitizeNonNegativeCounter(this._allocationCount);
      const reusedAllocations = sanitizeNonNegativeCounter(this._reusedAllocations);
      const freshAllocations = sanitizeNonNegativeCounter(this._freshAllocations);
      const elapsedMs =
        this._firstActivityTimeMs !== null && this._lastActivityTimeMs !== null
          ? Math.max(0, this._lastActivityTimeMs - this._firstActivityTimeMs)
          : 0;
      const uploadBytesPerSec =
        elapsedMs > 0 && bytesUploaded > 0 ? (bytesUploaded * 1000) / elapsedMs : 0;

      return {
        totalChunksAllocated: sanitizeNonNegativeCounter(this._totalChunksAllocated),
        activeChunks: activeCount,
        recycledChunks: recycledCount,
        inFlightChunks: this._inFlightChunks.length,
        activeBuffers: activeCount,
        recycledBuffers: recycledCount,
        activeCount,
        idleCount: recycledCount,
        allocatedBytes: sanitizeNonNegativeCounter(this._allocatedBytes),
        bytesAllocated: sanitizeNonNegativeCounter(this._totalBytesAllocated),
        bytesReused: sanitizeNonNegativeCounter(this._bytesReused),
        peakBytes: peakAllocatedBytes,
        peakAllocatedBytes,
        peakBytesAllocated: peakAllocatedBytes,
        peakAllocationBytes: peakAllocatedBytes,
        peakSingleAllocationBytes: sanitizeNonNegativeCounter(this._peakSingleAllocationBytes),
        maxPoolBytes: this.maxPoolBytes,
        maxPooledBytes: this.maxPoolBytes,
        reclaimedChunks: sanitizeNonNegativeCounter(this._reclaimedChunks),
        reclaimedBytes: sanitizeNonNegativeCounter(this._reclaimedBytes),
        evictedCount: lruEvictions,
        evictedBytes: sanitizeNonNegativeCounter(this._evictedBytes),
        lruEvictions,
        oversizeAllocations: oversizedAllocations,
        oversizedAllocations,
        bytesUploaded,
        totalBytesUploaded: bytesUploaded,
        totalBytesStaged,
        allocationCount,
        poolHits,
        poolMisses,
        poolHitRate: hitRate,
        hitRate,
        bufferReuseHits: poolHits,
        bufferReuseMisses: poolMisses,
        reusedAllocations,
        freshAllocations,
        reuseRate: clampUnitRate(reusedAllocations, allocationCount),
        largeSlabHits,
        largeSlabMisses,
        largeSlabEvictions: sanitizeNonNegativeCounter(this._largeSlabEvictions),
        largeSlabReuseRate: clampUnitRate(largeSlabHits, largeSlabTotal),
        activeLargeSlabs,
        inFlightLargeSlabs,
        idleLargeSlabs: sanitizeNonNegativeCounter(this._idleLargeSlabCount),
        largeSlabBytes: sanitizeNonNegativeCounter(this._largeSlabBytes),
        maxLargeSlabs: this.maxLargeSlabs,
        maxSlabsPerBucket: this.maxSlabsPerBucket,
        bucketStrategy: this.bucketStrategy,
        maxBucketBytes: this.maxBucketBytes,
        largeSlabBucketCounts,
        framesCompleted: sanitizeNonNegativeCounter(this._framesCompleted),
        lastFrameBytesUploaded: sanitizeNonNegativeCounter(this._lastFrameBytesUploaded),
        peakFrameBytesUploaded: sanitizeNonNegativeCounter(this._peakFrameBytesUploaded),
        avgBytesPerFrame:
          this._framesCompleted > 0 ? bytesUploaded / this._framesCompleted : 0,
        uploadBytesPerSec,
        uploadMiBPerSec: uploadBytesPerSec / (1024 * 1024),
        alignPadCount: sanitizeNonNegativeCounter(this._alignPadCount),
        sabCopyCount: sanitizeNonNegativeCounter(this._sabCopyCount),
        doubleReleaseIgnored: sanitizeNonNegativeCounter(this._doubleReleaseIgnored),
        maxFramesInFlight: this.maxFramesInFlight,
        idleShrinkFrames: this.idleShrinkFrames,
        subAllocCoalesced: sanitizeNonNegativeCounter(this._subAllocCoalesced),
        idleSlabShrinks: sanitizeNonNegativeCounter(this._idleSlabShrinks),
        stallAvoidedCount: sanitizeNonNegativeCounter(this._stallAvoidedCount),
        destroyed: this._destroyed,
      };
    }

    /**
     * Reset cumulative telemetry counters while preserving live pooled GPUBuffers,
     * active/in-flight/recycled queues, and frame/LRU ordering state.
     * Re-baselines `peakAllocatedBytes` to the current live `allocatedBytes` so
     * `peakAllocatedBytes >= allocatedBytes` always holds.
     */
    resetStats() {
      this._totalChunksAllocated = 0;
      this._totalBytesAllocated = 0;
      this._peakAllocatedBytes = this._allocatedBytes;
      this._peakSingleAllocationBytes = 0;
      this._reclaimedChunks = 0;
      this._reclaimedBytes = 0;
      this._evictedCount = 0;
      this._evictedBytes = 0;
      this._oversizeAllocations = 0;
      this._bytesUploaded = 0;
      this._totalBytesStaged = 0;
      this._bytesReused = 0;
      this._allocationCount = 0;
      this._bufferReuseHits = 0;
      this._bufferReuseMisses = 0;
      this._largeSlabHits = 0;
      this._largeSlabMisses = 0;
      this._largeSlabEvictions = 0;
      this._reusedAllocations = 0;
      this._freshAllocations = 0;
      this._alignPadCount = 0;
      this._sabCopyCount = 0;
      this._doubleReleaseIgnored = 0;
      this._subAllocCoalesced = 0;
      this._idleSlabShrinks = 0;
      this._stallAvoidedCount = 0;
      this._framesCompleted = 0;
      this._lastFrameBytesUploaded = 0;
      this._peakFrameBytesUploaded = 0;
      this._firstActivityTimeMs = null;
      this._lastActivityTimeMs = null;
      this._frameAllocations = 0;
      this._frameBytesAllocated = 0;
      this._frameBytesUploaded = 0;
      return this.getStats();
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
      this._largeSlabBuckets.clear();
      this._idleLargeSlabCount = 0;
      this._largeSlabBytes = 0;
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
    DEFAULT_MAX_FRAMES_IN_FLIGHT,
    DEFAULT_MIN_WARM_CHUNKS,
    DEFAULT_IDLE_EVICT_FRAMES,
    DEFAULT_IDLE_SHRINK_FRAMES,
    DEFAULT_LARGE_SLAB_THRESHOLD,
    DEFAULT_MAX_LARGE_SLABS,
    DEFAULT_POW2_MIN_SLAB_BYTES,
    DEFAULT_POW2_MAX_BUCKET_BYTES,
    DEFAULT_LARGE_SLAB_BUCKETS,
    POW2_LARGE_SLAB_BUCKETS,
    BUCKET_STRATEGY,
    WEBGPU_BUFFER_COPY_ALIGNMENT,
    WEBGPU_BYTES_PER_ROW_ALIGNMENT,
    BUFFER_USAGE,
    MAP_MODE,
    STAGING_MODE,
    alignUp,
    alignBufferSize,
    nextPowerOfTwo,
    buildPowerOfTwoBuckets,
    selectPowerOfTwoBucket,
    selectLargeSlabBucket,
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
