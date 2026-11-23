'use strict';

// WebGPU transient render-target texture pool and per-frame attachment aliasing manager.
// Zero external dependencies; UMD export for Web Workers (importScripts / globalThis.PlayGta6RenderTargetPool),
// browsers, and Node.js (module.exports).
// Designed for RAGE deferred color/depth/shadow/post-FX passes in web/workers/wgpu_worker.js.

(function (root) {
  const DEFAULT_MAX_POOL_BYTES = 256 * 1024 * 1024; // 256 MiB
  const DEFAULT_IDLE_FRAME_TTL = 8;

  // WebGPU GPUTextureUsage bitmask values with headless/Node fallbacks.
  const TEXTURE_USAGE = Object.freeze({
    COPY_SRC:
      typeof GPUTextureUsage !== 'undefined' && typeof GPUTextureUsage.COPY_SRC === 'number'
        ? GPUTextureUsage.COPY_SRC
        : 0x01,
    COPY_DST:
      typeof GPUTextureUsage !== 'undefined' && typeof GPUTextureUsage.COPY_DST === 'number'
        ? GPUTextureUsage.COPY_DST
        : 0x02,
    TEXTURE_BINDING:
      typeof GPUTextureUsage !== 'undefined' && typeof GPUTextureUsage.TEXTURE_BINDING === 'number'
        ? GPUTextureUsage.TEXTURE_BINDING
        : 0x04,
    STORAGE_BINDING:
      typeof GPUTextureUsage !== 'undefined' && typeof GPUTextureUsage.STORAGE_BINDING === 'number'
        ? GPUTextureUsage.STORAGE_BINDING
        : 0x08,
    RENDER_ATTACHMENT:
      typeof GPUTextureUsage !== 'undefined' && typeof GPUTextureUsage.RENDER_ATTACHMENT === 'number'
        ? GPUTextureUsage.RENDER_ATTACHMENT
        : 0x10,
  });

  const DEFAULT_TEXTURE_USAGE = TEXTURE_USAGE.RENDER_ATTACHMENT | TEXTURE_USAGE.TEXTURE_BINDING;

  const VALID_DIMENSIONS = Object.freeze(['1d', '2d', '3d']);

  /**
   * Bytes-per-texel lookup table for common WebGPU color, depth, and stencil render-target formats.
   * Note: `depth24plus` occupies 4 bytes/texel in hardware backing storage (D24X8 / D32F);
   * `depth32float-stencil8` occupies 8 bytes/texel (32-bit depth + 8-bit stencil + 24-bit pad).
   */
  const FORMAT_BYTES_PER_PIXEL = Object.freeze({
    // 8-bit / 1-byte formats
    r8unorm: 1,
    r8snorm: 1,
    r8uint: 1,
    r8sint: 1,
    stencil8: 1,

    // 16-bit / 2-byte formats
    r16float: 2,
    r16uint: 2,
    r16sint: 2,
    r16unorm: 2,
    r16snorm: 2,
    rg8unorm: 2,
    rg8snorm: 2,
    rg8uint: 2,
    rg8sint: 2,
    depth16unorm: 2,

    // 32-bit / 4-byte formats
    r32float: 4,
    r32uint: 4,
    r32sint: 4,
    rg16float: 4,
    rg16uint: 4,
    rg16sint: 4,
    rg16unorm: 4,
    rg16snorm: 4,
    rgba8unorm: 4,
    'rgba8unorm-srgb': 4,
    rgba8snorm: 4,
    rgba8uint: 4,
    rgba8sint: 4,
    bgra8unorm: 4,
    'bgra8unorm-srgb': 4,
    rgb10a2unorm: 4,
    rgb10a2uint: 4,
    rg11b10ufloat: 4,
    rgb9e5ufloat: 4,
    depth24plus: 4,
    'depth24plus-stencil8': 4,
    depth32float: 4,

    // 64-bit / 8-byte formats
    rg32float: 8,
    rg32uint: 8,
    rg32sint: 8,
    rgba16float: 8,
    rgba16uint: 8,
    rgba16sint: 8,
    rgba16unorm: 8,
    rgba16snorm: 8,
    'depth32float-stencil8': 8,

    // 128-bit / 16-byte formats
    rgba32float: 16,
    rgba32uint: 16,
    rgba32sint: 16,
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

  function isPowerOfTwo(n) {
    return Number.isInteger(n) && n >= 1 && (n & (n - 1)) === 0;
  }

  /**
   * Normalize a render-target descriptor (accepting flat `{ width, height, ... }` or
   * WebGPU `{ size: { width, height, depthOrArrayLayers } | [w, h, d], ... }` shapes)
   * into a canonical frozen descriptor object.
   *
   * @param {object} desc
   * @returns {{
   *   width: number,
   *   height: number,
   *   format: string,
   *   usage: number,
   *   sampleCount: number,
   *   mipLevelCount: number,
   *   depthOrArrayLayers: number,
   *   dimension: '1d' | '2d' | '3d',
   * }}
   */
  function normalizeRenderTargetDesc(desc) {
    if (!desc || typeof desc !== 'object') {
      throw new TypeError('Render target descriptor must be an object');
    }

    let width = desc.width;
    let height = desc.height;
    let depthOrArrayLayers = desc.depthOrArrayLayers;
    if (depthOrArrayLayers === undefined && desc.depthOrLayers !== undefined) {
      depthOrArrayLayers = desc.depthOrLayers;
    }

    if (desc.size !== undefined && desc.size !== null) {
      if (Array.isArray(desc.size)) {
        if (width === undefined) {
          width = desc.size[0];
        }
        if (height === undefined) {
          height = desc.size[1];
        }
        if (depthOrArrayLayers === undefined && desc.size[2] !== undefined) {
          depthOrArrayLayers = desc.size[2];
        }
      } else if (typeof desc.size === 'object') {
        if (width === undefined) {
          width = desc.size.width;
        }
        if (height === undefined) {
          height = desc.size.height;
        }
        if (depthOrArrayLayers === undefined && desc.size.depthOrArrayLayers !== undefined) {
          depthOrArrayLayers = desc.size.depthOrArrayLayers;
        }
      } else {
        throw new TypeError('desc.size must be an object { width, height, depthOrArrayLayers } or array');
      }
    }

    const dimension = desc.dimension === undefined ? '2d' : String(desc.dimension).trim().toLowerCase();
    if (!VALID_DIMENSIONS.includes(dimension)) {
      throw new RangeError(
        `Unsupported texture dimension "${desc.dimension}"; expected one of: ${VALID_DIMENSIONS.join(', ')}`
      );
    }

    if (dimension === '1d' && height === undefined) {
      height = 1;
    }
    if (depthOrArrayLayers === undefined) {
      depthOrArrayLayers = 1;
    }

    assertPositiveInteger(width, 'width');
    assertPositiveInteger(height, 'height');
    assertPositiveInteger(depthOrArrayLayers, 'depthOrArrayLayers');

    if (typeof desc.format !== 'string' || desc.format.trim().length === 0) {
      throw new TypeError('desc.format must be a non-empty WebGPU GPUTextureFormat string');
    }
    const format = desc.format.trim().toLowerCase();

    const usage = desc.usage === undefined ? DEFAULT_TEXTURE_USAGE : desc.usage;
    assertPositiveInteger(usage, 'usage');

    const sampleCount = desc.sampleCount === undefined ? 1 : desc.sampleCount;
    if (!isPowerOfTwo(sampleCount)) {
      throw new RangeError(`sampleCount must be a positive power-of-two integer, got ${sampleCount}`);
    }

    const mipLevelCount = desc.mipLevelCount === undefined ? 1 : desc.mipLevelCount;
    assertPositiveInteger(mipLevelCount, 'mipLevelCount');

    return Object.freeze({
      width,
      height,
      format,
      usage,
      sampleCount,
      mipLevelCount,
      depthOrArrayLayers,
      dimension,
    });
  }

  /**
   * Build a canonical deterministic pool key from a render-target descriptor.
   *
   * @param {object} desc
   * @returns {string}
   */
  function makeRenderTargetKey(desc) {
    const n = normalizeRenderTargetDesc(desc);
    return (
      `${n.dimension}|${n.width}x${n.height}x${n.depthOrArrayLayers}|` +
      `${n.format}|u${n.usage}|s${n.sampleCount}|m${n.mipLevelCount}`
    );
  }

  /**
   * Estimate VRAM footprint in bytes for a WebGPU render target descriptor,
   * accounting for format texel size, full mip chain, array layers / 3D depth,
   * and MSAA `sampleCount`.
   *
   * @param {object} desc
   * @returns {number}
   */
  function estimateRenderTargetBytes(desc) {
    const n = normalizeRenderTargetDesc(desc);
    if (!Object.prototype.hasOwnProperty.call(FORMAT_BYTES_PER_PIXEL, n.format)) {
      throw new RangeError(`Unsupported render target format for byte estimation: "${n.format}"`);
    }

    const bytesPerPixel = FORMAT_BYTES_PER_PIXEL[n.format];
    const is1D = n.dimension === '1d';
    const is3D = n.dimension === '3d';

    let totalBytes = 0;
    for (let mip = 0; mip < n.mipLevelCount; mip += 1) {
      const scale = mip >= 31 ? Infinity : Math.pow(2, mip);
      const mipWidth = scale === Infinity ? 1 : Math.max(1, Math.floor(n.width / scale));
      const mipHeight = is1D
        ? 1
        : scale === Infinity
          ? 1
          : Math.max(1, Math.floor(n.height / scale));
      const mipDepth = is3D
        ? scale === Infinity
          ? 1
          : Math.max(1, Math.floor(n.depthOrArrayLayers / scale))
        : n.depthOrArrayLayers;

      totalBytes += mipWidth * mipHeight * mipDepth * bytesPerPixel * n.sampleCount;
    }

    return totalBytes;
  }

  /**
   * WebGPU transient render-target texture pool & per-frame attachment aliasing manager.
   *
   * - Pools `GPUTexture` + cached default `GPUTextureView` objects keyed by normalized
   *   `(width, height, format, usage, sampleCount, mipLevelCount, depthOrArrayLayers, dimension)`.
   * - Supports immediate same-frame attachment aliasing once `release()` is called, plus
   *   cross-frame reuse without `device.createTexture()` churn.
   * - Enforces `maxPoolBytes` via global LRU eviction of idle targets and expires unused
   *   targets after `idleFrameTtl` frames in `endFrame()`.
   */
  class RenderTargetPool {
    constructor(options = {}) {
      if (!options || typeof options !== 'object') {
        throw new TypeError('RenderTargetPool constructor expects an options object');
      }

      const {
        device,
        maxPoolBytes = DEFAULT_MAX_POOL_BYTES,
        idleFrameTtl = DEFAULT_IDLE_FRAME_TTL,
        strictBudget = true,
        autoReleaseLeaked = true,
        onLeak = null,
        warn = null,
      } = options;

      if (!device || typeof device.createTexture !== 'function') {
        throw new TypeError('RenderTargetPool requires a WebGPU device with createTexture()');
      }

      if (maxPoolBytes !== Infinity && (!Number.isInteger(maxPoolBytes) || maxPoolBytes < 1)) {
        throw new RangeError(
          `maxPoolBytes must be a positive integer or Infinity, got ${maxPoolBytes}`
        );
      }
      assertNonNegativeInteger(idleFrameTtl, 'idleFrameTtl');

      if (onLeak !== null && typeof onLeak !== 'function') {
        throw new TypeError('onLeak must be a function when provided');
      }
      if (warn !== null && typeof warn !== 'function') {
        throw new TypeError('warn must be a function when provided');
      }

      this._device = device;
      this._maxPoolBytes = maxPoolBytes;
      this._idleFrameTtl = idleFrameTtl;
      this._strictBudget = Boolean(strictBudget);
      this._autoReleaseLeaked = Boolean(autoReleaseLeaked);
      this._onLeak = onLeak;
      this._warn = warn;

      this._destroyed = false;
      this._currentFrame = 0;
      this._nextId = 1;
      this._nextLeaseToken = 1;

      /** @type {Map<number, object>} All live (active + idle) records by id */
      this._allEntries = new Map();
      /** @type {Map<number, object>} Active in-use records by id */
      this._activeEntries = new Map();
      /** @type {Map<string, object[]>} Idle records grouped by normalized key (MRU stack) */
      this._idleBuckets = new Map();
      /** @type {Map<number, object>} Idle records in insertion order (oldest = LRU) */
      this._idleLru = new Map();
      /** @type {WeakMap<object, { id: number, leaseToken: number }>} Per-acquire handle tokens */
      this._handleTokens = new WeakMap();

      // Telemetry counters
      this._hits = 0;
      this._misses = 0;
      this._activeBytes = 0;
      this._idleBytes = 0;
      this._peakBytes = 0;
      this._evictions = 0;
      this._leakedReleases = 0;
    }

    get isDestroyed() {
      return this._destroyed;
    }

    get maxPoolBytes() {
      return this._maxPoolBytes;
    }

    get idleFrameTtl() {
      return this._idleFrameTtl;
    }

    get currentFrame() {
      return this._currentFrame;
    }

    _assertNotDestroyed() {
      if (this._destroyed) {
        throw new Error('RenderTargetPool has been destroyed');
      }
    }

    _removeFromIdleBucket(record) {
      const bucket = this._idleBuckets.get(record.key);
      if (!bucket) {
        return;
      }
      const idx = bucket.indexOf(record);
      if (idx !== -1) {
        bucket.splice(idx, 1);
      }
      if (bucket.length === 0) {
        this._idleBuckets.delete(record.key);
      }
    }

    _destroyRecord(record) {
      if (record.destroyed) {
        return;
      }
      record.destroyed = true;
      record.active = false;
      record.leaseToken = 0;
      this._allEntries.delete(record.id);

      if (record.texture && typeof record.texture.destroy === 'function') {
        record.texture.destroy();
      }
      record.texture = null;
      record.view = null;
    }

    _evictIdleRecord(record) {
      this._idleLru.delete(record.id);
      this._removeFromIdleBucket(record);
      this._idleBytes = Math.max(0, this._idleBytes - record.byteSize);
      this._destroyRecord(record);
      this._evictions += 1;
    }

    _moveActiveToIdle(record, frameIndex) {
      record.active = false;
      record.leaseToken = 0;
      record.lastUsedFrame = frameIndex;
      this._activeEntries.delete(record.id);
      this._activeBytes = Math.max(0, this._activeBytes - record.byteSize);
      this._idleBytes += record.byteSize;

      let bucket = this._idleBuckets.get(record.key);
      if (!bucket) {
        bucket = [];
        this._idleBuckets.set(record.key, bucket);
      }
      bucket.push(record);
      this._idleLru.set(record.id, record);
    }

    _makeHandle(record, reused) {
      const leaseToken = this._nextLeaseToken++;
      record.leaseToken = leaseToken;
      const handle = {
        id: record.id,
        key: record.key,
        desc: record.desc,
        texture: record.texture,
        view: record.view,
        byteSize: record.byteSize,
        reused,
      };
      this._handleTokens.set(handle, { id: record.id, leaseToken });
      return handle;
    }

    /**
     * Acquire a transient render-target texture matching `desc`. Reuses an idle pooled
     * texture (including same-frame targets released earlier in the same frame) when
     * available; otherwise allocates a new `GPUTexture` and cached default `GPUTextureView`,
     * LRU-evicting idle textures as needed to stay within `maxPoolBytes`.
     *
     * @param {object} desc
     * @param {string} [label]
     * @returns {{
     *   id: number,
     *   key: string,
     *   desc: Readonly<object>,
     *   texture: GPUTexture,
     *   view: GPUTextureView | null,
     *   byteSize: number,
     *   reused: boolean,
     * }}
     */
    acquire(desc, label) {
      this._assertNotDestroyed();

      const normalizedDesc = normalizeRenderTargetDesc(desc);
      const key = makeRenderTargetKey(normalizedDesc);
      const byteSize = estimateRenderTargetBytes(normalizedDesc);

      // 1. Check idle bucket for an exact descriptor match (MRU pop lets unused duplicates age out).
      const bucket = this._idleBuckets.get(key);
      if (bucket && bucket.length > 0) {
        const record = bucket.pop();
        if (bucket.length === 0) {
          this._idleBuckets.delete(key);
        }
        this._idleLru.delete(record.id);

        record.active = true;
        record.lastUsedFrame = this._currentFrame;
        if (typeof label === 'string' && label.length > 0) {
          record.label = label;
        }

        this._activeEntries.set(record.id, record);
        this._idleBytes = Math.max(0, this._idleBytes - record.byteSize);
        this._activeBytes += record.byteSize;
        this._hits += 1;

        return this._makeHandle(record, true);
      }

      // 2. Miss: check strict memory budget before evicting idle targets or allocating.
      if (this._strictBudget && this._activeBytes + byteSize > this._maxPoolBytes) {
        throw new RangeError(
          `RenderTargetPool budget exceeded: cannot allocate ${byteSize} bytes for ` +
            `${normalizedDesc.width}x${normalizedDesc.height} ${normalizedDesc.format} ` +
            `(activeBytes=${this._activeBytes}, maxPoolBytes=${this._maxPoolBytes})`
        );
      }

      // 3. LRU-evict idle targets until totalBytes + byteSize <= maxPoolBytes (or no idle left).
      while (
        this._activeBytes + this._idleBytes + byteSize > this._maxPoolBytes &&
        this._idleLru.size > 0
      ) {
        const oldest = this._idleLru.values().next().value;
        this._evictIdleRecord(oldest);
      }

      // 4. Allocate new GPUTexture + cached default GPUTextureView.
      const id = this._nextId++;
      const resolvedLabel =
        typeof label === 'string' && label.length > 0
          ? label
          : `playgta6-rt-${id}-${normalizedDesc.width}x${normalizedDesc.height}-${normalizedDesc.format}`;

      const texture = this._device.createTexture({
        label: resolvedLabel,
        size: {
          width: normalizedDesc.width,
          height: normalizedDesc.height,
          depthOrArrayLayers: normalizedDesc.depthOrArrayLayers,
        },
        mipLevelCount: normalizedDesc.mipLevelCount,
        sampleCount: normalizedDesc.sampleCount,
        dimension: normalizedDesc.dimension,
        format: normalizedDesc.format,
        usage: normalizedDesc.usage,
      });

      let view = null;
      try {
        view = texture && typeof texture.createView === 'function' ? texture.createView() : null;
      } catch (err) {
        if (texture && typeof texture.destroy === 'function') {
          texture.destroy();
        }
        throw err;
      }

      const record = {
        id,
        key,
        desc: normalizedDesc,
        label: resolvedLabel,
        texture,
        view,
        byteSize,
        active: true,
        destroyed: false,
        lastUsedFrame: this._currentFrame,
        leaseToken: 0,
      };

      this._allEntries.set(id, record);
      this._activeEntries.set(id, record);
      this._activeBytes += byteSize;
      this._misses += 1;

      const totalBytes = this._activeBytes + this._idleBytes;
      if (totalBytes > this._peakBytes) {
        this._peakBytes = totalBytes;
      }

      return this._makeHandle(record, false);
    }

    /**
     * Mark an acquired render target available for immediate aliasing within the same frame
     * (or subsequent frames). Safe against duplicate / stale releases (returns `false` instead
     * of throwing or corrupting pool state).
     *
     * @param {object | number} entryOrId
     * @returns {boolean} True if an active target was released; false if already idle/unknown/stale.
     */
    release(entryOrId) {
      this._assertNotDestroyed();

      if (entryOrId === null || entryOrId === undefined) {
        throw new TypeError('release() requires an acquired entry object or numeric id');
      }

      let id;
      let expectedLeaseToken = null;

      if (typeof entryOrId === 'number') {
        if (!Number.isInteger(entryOrId) || entryOrId < 1) {
          throw new TypeError(`release() id must be a positive integer, got ${entryOrId}`);
        }
        id = entryOrId;
      } else if (typeof entryOrId === 'object') {
        const tokenInfo = this._handleTokens.get(entryOrId);
        if (tokenInfo) {
          id = tokenInfo.id;
          expectedLeaseToken = tokenInfo.leaseToken;
        } else if (Number.isInteger(entryOrId.id) && entryOrId.id >= 1) {
          id = entryOrId.id;
        } else {
          throw new TypeError('release() entry object must have a valid positive integer id');
        }
      } else {
        throw new TypeError('release() requires an acquired entry object or numeric id');
      }

      const record = this._allEntries.get(id);
      if (!record || !record.active) {
        return false;
      }

      // Guard against stale handle double-release stealing a newer pass's active lease.
      if (expectedLeaseToken !== null && record.leaseToken !== expectedLeaseToken) {
        return false;
      }

      this._moveActiveToIdle(record, this._currentFrame);
      return true;
    }

    /**
     * End the current frame epoch:
     * 1. Handle any leaked (still-active) render targets: auto-release back into the idle pool
     *    when `autoReleaseLeaked` is enabled, or warn/notify via `onLeak` / `warn` when disabled.
     * 2. Increment the frame epoch (`currentFrame += 1`).
     * 3. Destroy idle textures unused for more than `idleFrameTtl` frames.
     *
     * @returns {{
     *   frame: number,
     *   evictedCount: number,
     *   leakedCount: number,
     *   autoReleasedCount: number,
     *   activeCount: number,
     *   idleCount: number,
     * }}
     */
    endFrame() {
      this._assertNotDestroyed();

      const endedFrame = this._currentFrame;
      this._currentFrame += 1;

      let leakedCount = 0;
      let autoReleasedCount = 0;

      if (this._activeEntries.size > 0) {
        const leakedRecords = Array.from(this._activeEntries.values());
        leakedCount = leakedRecords.length;

        for (let i = 0; i < leakedRecords.length; i += 1) {
          const record = leakedRecords[i];
          const leakInfo = {
            id: record.id,
            key: record.key,
            label: record.label,
            byteSize: record.byteSize,
            frame: endedFrame,
            autoReleased: this._autoReleaseLeaked,
          };

          if (this._autoReleaseLeaked) {
            this._moveActiveToIdle(record, endedFrame);
            this._leakedReleases += 1;
            autoReleasedCount += 1;
          }

          if (this._onLeak) {
            this._onLeak(leakInfo);
          }

          if (!this._autoReleaseLeaked) {
            const msg =
              `[RenderTargetPool] Leaked active render target #${record.id} ` +
              `(${record.label}, ${record.byteSize} bytes) at end of frame ${endedFrame}`;
            if (this._warn) {
              this._warn(msg, leakInfo);
            } else if (!this._onLeak && typeof console !== 'undefined' && typeof console.warn === 'function') {
              console.warn(msg);
            }
          } else if (this._warn) {
            const msg =
              `[RenderTargetPool] Auto-released leaked render target #${record.id} ` +
              `(${record.label}, ${record.byteSize} bytes) at end of frame ${endedFrame}`;
            this._warn(msg, leakInfo);
          }
        }
      }

      // Evict idle textures unused for more than idleFrameTtl frames.
      let evictedCount = 0;
      const idleRecords = Array.from(this._idleLru.values());
      for (let i = 0; i < idleRecords.length; i += 1) {
        const record = idleRecords[i];
        if (this._currentFrame - record.lastUsedFrame > this._idleFrameTtl) {
          this._evictIdleRecord(record);
          evictedCount += 1;
        }
      }

      return {
        frame: this._currentFrame,
        evictedCount,
        leakedCount,
        autoReleasedCount,
        activeCount: this._activeEntries.size,
        idleCount: this._idleLru.size,
      };
    }

    /**
     * Destroy all currently idle pooled render targets (e.g., on viewport/swapchain resize)
     * while leaving any currently active targets intact.
     *
     * @returns {number} Number of idle textures destroyed.
     */
    invalidateAll() {
      this._assertNotDestroyed();

      let destroyedCount = 0;
      const idleRecords = Array.from(this._idleLru.values());
      for (let i = 0; i < idleRecords.length; i += 1) {
        const record = idleRecords[i];
        this._idleLru.delete(record.id);
        this._removeFromIdleBucket(record);
        this._idleBytes = Math.max(0, this._idleBytes - record.byteSize);
        this._destroyRecord(record);
        destroyedCount += 1;
      }

      this._idleBuckets.clear();
      this._idleLru.clear();
      this._idleBytes = 0;
      return destroyedCount;
    }

    /**
     * Return pool hit-rate, occupancy, memory, and lifecycle telemetry.
     */
    getStats() {
      const totalRequests = this._hits + this._misses;
      const totalBytes = this._activeBytes + this._idleBytes;
      return {
        hits: this._hits,
        misses: this._misses,
        hitRate: totalRequests > 0 ? this._hits / totalRequests : 0,
        activeCount: this._activeEntries.size,
        idleCount: this._idleLru.size,
        activeBytes: this._activeBytes,
        idleBytes: this._idleBytes,
        totalBytes,
        peakBytes: this._peakBytes,
        evictions: this._evictions,
        leakedReleases: this._leakedReleases,
        currentFrame: this._currentFrame,
      };
    }

    /**
     * Destroy all active and idle GPU textures owned by the pool and release references.
     * Safe to call multiple times (idempotent).
     */
    destroy() {
      if (this._destroyed) {
        return;
      }

      this._destroyed = true;

      const allRecords = Array.from(this._allEntries.values());
      for (let i = 0; i < allRecords.length; i += 1) {
        this._destroyRecord(allRecords[i]);
      }

      this._allEntries.clear();
      this._activeEntries.clear();
      this._idleBuckets.clear();
      this._idleLru.clear();
      this._activeBytes = 0;
      this._idleBytes = 0;
      this._device = null;
    }
  }

  const exportsObj = Object.freeze({
    DEFAULT_MAX_POOL_BYTES,
    DEFAULT_IDLE_FRAME_TTL,
    TEXTURE_USAGE,
    DEFAULT_TEXTURE_USAGE,
    FORMAT_BYTES_PER_PIXEL,
    normalizeRenderTargetDesc,
    makeRenderTargetKey,
    estimateRenderTargetBytes,
    RenderTargetPool,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6RenderTargetPool = exportsObj;
  } else if (root) {
    root.PlayGta6RenderTargetPool = exportsObj;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportsObj;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
