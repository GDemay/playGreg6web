'use strict';

// WebGPU resource handle registry and frame-deferred garbage collector.
// Zero external dependencies; UMD/CommonJS/ESM compatible for Web Workers
// (importScripts -> globalThis.PlayGta6ResourceRegistry), browsers, and
// Node.js (require() / import()).

(function (root) {
  const DEFAULT_FRAMES_IN_FLIGHT = 3;
  const DEFAULT_SOFT_BUDGET_BYTES = 512 * 1024 * 1024; // 512 MiB

  const RESOURCE_KIND = Object.freeze({
    BUFFER: 'buffer',
    TEXTURE: 'texture',
    TEXTURE_VIEW: 'textureView',
    SAMPLER: 'sampler',
    BIND_GROUP: 'bindGroup',
    BIND_GROUP_LAYOUT: 'bindGroupLayout',
    PIPELINE_LAYOUT: 'pipelineLayout',
    SHADER_MODULE: 'shaderModule',
    RENDER_PIPELINE: 'renderPipeline',
    COMPUTE_PIPELINE: 'computePipeline',
  });

  const RESOURCE_KINDS = Object.freeze(Object.values(RESOURCE_KIND));
  const VALID_KIND_SET = new Set(RESOURCE_KINDS);

  function assertValidKind(kind) {
    if (typeof kind !== 'string' || !VALID_KIND_SET.has(kind)) {
      throw new RangeError(
        `Invalid WebGPU resource kind: ${JSON.stringify(kind)} (expected one of ${RESOURCE_KINDS.join(', ')})`
      );
    }
  }

  function assertValidHandle(handle) {
    if (typeof handle !== 'number' || !Number.isSafeInteger(handle) || handle < 0) {
      throw new RangeError(
        `Invalid resource handle: ${String(handle)} (expected a non-negative safe integer)`
      );
    }
  }

  function assertPositiveInteger(value, name) {
    if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1) {
      throw new RangeError(`${name} must be a positive safe integer, got ${String(value)}`);
    }
  }

  function assertNonNegativeInteger(value, name) {
    if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) {
      throw new RangeError(`${name} must be a non-negative safe integer, got ${String(value)}`);
    }
  }

  function snapshotEntry(entry) {
    return {
      kind: entry.kind,
      handle: entry.handle,
      resource: entry.resource,
      byteSize: entry.byteSize,
      label: entry.label,
      tag: entry.tag,
      registeredAtFrame: entry.registeredAtFrame,
    };
  }

  /**
   * Tracks WebGPU resources by `(kind, handle)` with VRAM byte accounting and
   * frame-epoch / GPU-fence deferred destruction so in-flight GPU command buffers
   * never hit use-after-destroy validation errors.
   */
  class GpuResourceRegistry {
    /**
     * @param {{
     *   framesInFlight?: number,
     *   softBudgetBytes?: number | null,
     *   onBudgetWarning?: ((info: {
     *     activeBytes: number,
     *     pendingRetireBytes: number,
     *     totalBytes: number,
     *     softBudgetBytes: number,
     *     currentFrame: number,
     *     kind: string,
     *     handle: number,
     *     label: string,
     *   }) => void) | null,
     *   onDestroyError?: ((err: unknown, entry: {
     *     kind: string,
     *     handle: number,
     *     byteSize: number,
     *     label: string,
     *   }) => void) | null,
     * }} [options]
     */
    constructor(options = {}) {
      if (options === null || typeof options !== 'object') {
        throw new TypeError('GpuResourceRegistry constructor expects an options object');
      }

      const {
        framesInFlight = DEFAULT_FRAMES_IN_FLIGHT,
        softBudgetBytes = DEFAULT_SOFT_BUDGET_BYTES,
        onBudgetWarning = null,
        onDestroyError = null,
      } = options;

      assertPositiveInteger(framesInFlight, 'framesInFlight');

      let resolvedBudget = null;
      if (softBudgetBytes !== null && softBudgetBytes !== Infinity) {
        assertPositiveInteger(softBudgetBytes, 'softBudgetBytes');
        resolvedBudget = softBudgetBytes;
      }

      if (onBudgetWarning !== null && typeof onBudgetWarning !== 'function') {
        throw new TypeError('onBudgetWarning must be a function when provided');
      }
      if (onDestroyError !== null && typeof onDestroyError !== 'function') {
        throw new TypeError('onDestroyError must be a function when provided');
      }

      this.framesInFlight = framesInFlight;
      this.softBudgetBytes = resolvedBudget;
      this._onBudgetWarning = onBudgetWarning;
      this._onDestroyError = onDestroyError;

      /** @type {Map<string, Map<number, object>>} */
      this._tables = new Map();
      for (let i = 0; i < RESOURCE_KINDS.length; i += 1) {
        this._tables.set(RESOURCE_KINDS[i], new Map());
      }

      /** @type {Array<object>} */
      this._pendingRetire = [];
      /** @type {Set<Promise<void>>} */
      this._pendingFences = new Set();
      /** @type {WeakSet<object>} */
      this._destroyedResources = new WeakSet();

      this._currentFrame = 0;
      this._generation = 0;

      this._activeBytes = 0;
      this._pendingRetireBytes = 0;
      this._peakBytes = 0;

      this._budgetExceeded = false;
      this._budgetWarningCount = 0;

      this._totalRegistered = 0;
      this._totalRetired = 0;
      this._totalDestroyed = 0;
      this._destroyErrors = 0;
    }

    _updatePeakBytes() {
      const totalBytes = this._activeBytes + this._pendingRetireBytes;
      if (totalBytes > this._peakBytes) {
        this._peakBytes = totalBytes;
      }
      return totalBytes;
    }

    _checkBudget(context) {
      const totalBytes = this._activeBytes + this._pendingRetireBytes;
      if (this.softBudgetBytes === null) {
        this._budgetExceeded = false;
        return;
      }

      if (totalBytes > this.softBudgetBytes) {
        if (!this._budgetExceeded) {
          this._budgetExceeded = true;
          this._budgetWarningCount += 1;
          if (this._onBudgetWarning) {
            this._onBudgetWarning({
              activeBytes: this._activeBytes,
              pendingRetireBytes: this._pendingRetireBytes,
              totalBytes,
              softBudgetBytes: this.softBudgetBytes,
              currentFrame: this._currentFrame,
              kind: context ? context.kind : '',
              handle: context ? context.handle : -1,
              label: context ? context.label : '',
            });
          }
        }
      } else {
        this._budgetExceeded = false;
      }
    }

    _sealUnsealedRetirements() {
      const batch = [];
      for (let i = 0; i < this._pendingRetire.length; i += 1) {
        const entry = this._pendingRetire[i];
        if (!entry.sealed) {
          entry.sealed = true;
          batch.push(entry);
        }
      }
      return batch;
    }

    _destroyEntry(entry) {
      if (entry.destroyed) {
        return false;
      }
      entry.destroyed = true;

      const resource = entry.resource;
      entry.resource = null;

      if (resource && typeof resource === 'object' && !this._destroyedResources.has(resource)) {
        this._destroyedResources.add(resource);
        if (typeof resource.destroy === 'function') {
          try {
            resource.destroy();
          } catch (err) {
            this._destroyErrors += 1;
            if (this._onDestroyError) {
              this._onDestroyError(err, {
                kind: entry.kind,
                handle: entry.handle,
                byteSize: entry.byteSize,
                label: entry.label,
              });
            }
          }
        }
      }

      this._totalDestroyed += 1;
      return true;
    }

    /**
     * Register a WebGPU resource under `(kind, handle)`.
     * If a different resource already occupies `(kind, handle)`, the prior resource
     * is retired for deferred destruction rather than leaked.
     *
     * @param {string} kind
     * @param {number} handle
     * @param {object} resource
     * @param {{ byteSize?: number, label?: string, tag?: unknown }} [meta]
     * @returns {object} The registered resource
     */
    register(kind, handle, resource, meta = {}) {
      assertValidKind(kind);
      assertValidHandle(handle);

      if (resource === null || typeof resource !== 'object') {
        throw new TypeError('register requires a non-null resource object');
      }
      if (meta === null || typeof meta !== 'object') {
        throw new TypeError('register metadata must be an object when provided');
      }

      const { byteSize = 0, label = '', tag = null } = meta;
      assertNonNegativeInteger(byteSize, 'byteSize');
      if (typeof label !== 'string') {
        throw new TypeError('register metadata.label must be a string');
      }

      const table = this._tables.get(kind);
      const existing = table.get(handle);

      if (existing) {
        if (existing.resource === resource) {
          this._activeBytes = Math.max(0, this._activeBytes - existing.byteSize) + byteSize;
          existing.byteSize = byteSize;
          existing.label = label;
          existing.tag = tag;
          this._updatePeakBytes();
          this._checkBudget({ kind, handle, label });
          return resource;
        }
        // Cleanly retire the previous resource on handle overwrite so it is never leaked.
        this.retire(kind, handle);
      }

      const entry = {
        kind,
        handle,
        resource,
        byteSize,
        label,
        tag,
        registeredAtFrame: this._currentFrame,
        retiredAtFrame: -1,
        sealed: false,
        hasFence: false,
        fenceDone: false,
        fenceFailed: false,
        destroyed: false,
      };

      table.set(handle, entry);
      this._activeBytes += byteSize;
      this._totalRegistered += 1;

      this._updatePeakBytes();
      this._checkBudget({ kind, handle, label });

      return resource;
    }

    /**
     * Look up an active resource by `(kind, handle)`.
     *
     * @param {string} kind
     * @param {number} handle
     * @returns {object | undefined}
     */
    get(kind, handle) {
      assertValidKind(kind);
      assertValidHandle(handle);
      const entry = this._tables.get(kind).get(handle);
      return entry ? entry.resource : undefined;
    }

    /**
     * Look up an active resource entry snapshot (including metadata) by `(kind, handle)`.
     *
     * @param {string} kind
     * @param {number} handle
     * @returns {{ kind: string, handle: number, resource: object, byteSize: number, label: string, tag: unknown, registeredAtFrame: number } | undefined}
     */
    getEntry(kind, handle) {
      assertValidKind(kind);
      assertValidHandle(handle);
      const entry = this._tables.get(kind).get(handle);
      return entry ? snapshotEntry(entry) : undefined;
    }

    /**
     * Return whether an active resource exists for `(kind, handle)`.
     *
     * @param {string} kind
     * @param {number} handle
     * @returns {boolean}
     */
    has(kind, handle) {
      assertValidKind(kind);
      assertValidHandle(handle);
      return this._tables.get(kind).has(handle);
    }

    /**
     * List active resource entry snapshots, optionally filtered to a single `kind`.
     *
     * @param {string} [kind]
     * @returns {Array<{ kind: string, handle: number, resource: object, byteSize: number, label: string, tag: unknown, registeredAtFrame: number }>}
     */
    list(kind) {
      if (kind !== undefined) {
        assertValidKind(kind);
        const table = this._tables.get(kind);
        const result = [];
        for (const entry of table.values()) {
          result.push(snapshotEntry(entry));
        }
        return result;
      }

      const result = [];
      for (let i = 0; i < RESOURCE_KINDS.length; i += 1) {
        const table = this._tables.get(RESOURCE_KINDS[i]);
        for (const entry of table.values()) {
          result.push(snapshotEntry(entry));
        }
      }
      return result;
    }

    /**
     * Remove `(kind, handle)` from the active table immediately and enqueue it
     * for deferred destruction after `framesInFlight` frames or once its frame's
     * GPU fence resolves.
     *
     * @param {string} kind
     * @param {number} handle
     * @returns {boolean} True if an active resource was retired, false otherwise
     */
    retire(kind, handle) {
      assertValidKind(kind);
      assertValidHandle(handle);

      const table = this._tables.get(kind);
      const entry = table.get(handle);
      if (!entry) {
        return false;
      }

      table.delete(handle);
      this._activeBytes = Math.max(0, this._activeBytes - entry.byteSize);
      this._pendingRetireBytes += entry.byteSize;

      entry.retiredAtFrame = this._currentFrame;
      entry.sealed = false;
      entry.hasFence = false;
      entry.fenceDone = false;
      entry.fenceFailed = false;

      this._pendingRetire.push(entry);
      this._totalRetired += 1;
      return true;
    }

    /**
     * Convenience helper that retires a numeric `handle` from every resource kind
     * where it is currently registered (mirrors opcode DESTROY_RESOURCE).
     *
     * @param {number} handle
     * @returns {number} Number of active resources retired
     */
    retireHandle(handle) {
      assertValidHandle(handle);
      let count = 0;
      for (let i = 0; i < RESOURCE_KINDS.length; i += 1) {
        if (this.retire(RESOURCE_KINDS[i], handle)) {
          count += 1;
        }
      }
      return count;
    }

    /**
     * Advance the frame epoch and collect any deferred resources whose frame
     * latency or GPU fence has completed.
     *
     * @param {number} [frameIndex]
     * @returns {{ frameIndex: number, destroyedCount: number, freedBytes: number, remainingCount: number, pendingRetireBytes: number }}
     */
    beginFrame(frameIndex) {
      // Seal any unsealed retirements from the prior frame as epoch-based so a
      // future endFrame(fence) in this frame does not retroactively attach to them.
      this._sealUnsealedRetirements();

      let nextFrame;
      if (frameIndex === undefined) {
        nextFrame = this._currentFrame + 1;
      } else {
        assertNonNegativeInteger(frameIndex, 'frameIndex');
        if (frameIndex < this._currentFrame) {
          throw new RangeError(
            `frameIndex (${frameIndex}) cannot move backwards from currentFrame (${this._currentFrame})`
          );
        }
        nextFrame = frameIndex;
      }

      this._currentFrame = nextFrame;
      const collected = this.collect(false);

      return {
        frameIndex: this._currentFrame,
        destroyedCount: collected.destroyedCount,
        freedBytes: collected.freedBytes,
        remainingCount: collected.remainingCount,
        pendingRetireBytes: collected.pendingRetireBytes,
      };
    }

    /**
     * Seal all resources retired during the current frame and optionally bind
     * them to a GPU work-done fence promise (e.g. `device.queue.onSubmittedWorkDone()`).
     *
     * When a fence promise is supplied:
     *  - Resolving the fence allows early collection on the next `collect()` /
     *    `beginFrame()`, even before `framesInFlight` frames elapse.
     *  - While the fence remains pending, collection of those entries is gated
     *    until the fence resolves (unless `collect(true)` or `destroyAll()` is called).
     *  - If the fence rejects (e.g. device lost), those entries fall back to
     *    frame-epoch expiration instead of leaking or causing unhandled rejections.
     *
     * @param {Promise<unknown> | null} [fencePromise]
     * @returns {{ frameIndex: number, sealedCount: number, pendingCount: number, pendingRetireBytes: number }}
     */
    endFrame(fencePromise) {
      if (
        fencePromise !== undefined &&
        fencePromise !== null &&
        typeof fencePromise.then !== 'function'
      ) {
        throw new TypeError('endFrame expects fencePromise to be a Promise/thenable, null, or undefined');
      }

      const batch = this._sealUnsealedRetirements();

      if (fencePromise && typeof fencePromise.then === 'function') {
        const generation = this._generation;
        for (let i = 0; i < batch.length; i += 1) {
          batch[i].hasFence = true;
          batch[i].fenceDone = false;
          batch[i].fenceFailed = false;
        }

        const tracked = Promise.resolve(fencePromise).then(
          () => {
            if (generation !== this._generation) {
              return;
            }
            for (let i = 0; i < batch.length; i += 1) {
              if (!batch[i].destroyed) {
                batch[i].fenceDone = true;
              }
            }
          },
          () => {
            if (generation !== this._generation) {
              return;
            }
            for (let i = 0; i < batch.length; i += 1) {
              if (!batch[i].destroyed) {
                batch[i].fenceFailed = true;
              }
            }
          }
        );

        this._pendingFences.add(tracked);
        tracked.finally(() => {
          this._pendingFences.delete(tracked);
        });
      }

      return {
        frameIndex: this._currentFrame,
        sealedCount: batch.length,
        pendingCount: this._pendingRetire.length,
        pendingRetireBytes: this._pendingRetireBytes,
      };
    }

    /**
     * Destroy eligible retired resources and reclaim their pending-retirement VRAM accounting.
     *
     * @param {boolean} [force=false] When true, destroys all pending retired resources immediately regardless of frame epoch or pending fences.
     * @returns {{ destroyedCount: number, freedBytes: number, remainingCount: number, pendingRetireBytes: number }}
     */
    collect(force = false) {
      const forceAll = Boolean(force);
      let destroyedCount = 0;
      let freedBytes = 0;
      const remaining = [];

      for (let i = 0; i < this._pendingRetire.length; i += 1) {
        const entry = this._pendingRetire[i];
        const epochElapsed = this._currentFrame - entry.retiredAtFrame >= this.framesInFlight;

        let eligible;
        if (forceAll) {
          eligible = true;
        } else if (entry.hasFence) {
          eligible = entry.fenceDone || (entry.fenceFailed && epochElapsed);
        } else {
          eligible = epochElapsed;
        }

        if (eligible) {
          this._pendingRetireBytes = Math.max(0, this._pendingRetireBytes - entry.byteSize);
          freedBytes += entry.byteSize;
          if (this._destroyEntry(entry)) {
            destroyedCount += 1;
          }
        } else {
          remaining.push(entry);
        }
      }

      this._pendingRetire = remaining;
      this._checkBudget(null);

      return {
        destroyedCount,
        freedBytes,
        remainingCount: this._pendingRetire.length,
        pendingRetireBytes: this._pendingRetireBytes,
      };
    }

    /**
     * Alias for `collect(force)`.
     *
     * @param {boolean} [force=false]
     */
    flush(force = false) {
      return this.collect(force);
    }

    /**
     * Return a snapshot of per-kind counts, VRAM byte accounting, and lifecycle stats.
     */
    getStats() {
      const countsByKind = {};
      let activeCount = 0;
      for (let i = 0; i < RESOURCE_KINDS.length; i += 1) {
        const kind = RESOURCE_KINDS[i];
        const count = this._tables.get(kind).size;
        countsByKind[kind] = count;
        activeCount += count;
      }

      const pendingCountsByKind = {};
      for (let i = 0; i < RESOURCE_KINDS.length; i += 1) {
        pendingCountsByKind[RESOURCE_KINDS[i]] = 0;
      }
      for (let i = 0; i < this._pendingRetire.length; i += 1) {
        pendingCountsByKind[this._pendingRetire[i].kind] += 1;
      }

      return {
        countsByKind,
        pendingCountsByKind,
        activeCount,
        pendingRetireCount: this._pendingRetire.length,
        activeBytes: this._activeBytes,
        pendingRetireBytes: this._pendingRetireBytes,
        totalBytes: this._activeBytes + this._pendingRetireBytes,
        peakBytes: this._peakBytes,
        softBudgetBytes: this.softBudgetBytes,
        budgetExceeded: this._budgetExceeded,
        budgetWarningCount: this._budgetWarningCount,
        totalRegistered: this._totalRegistered,
        totalRetired: this._totalRetired,
        totalDestroyed: this._totalDestroyed,
        destroyErrors: this._destroyErrors,
        currentFrame: this._currentFrame,
        framesInFlight: this.framesInFlight,
      };
    }

    /**
     * Idempotently destroy all active and pending-retirement resources
     * (e.g. for `device.lost` recovery or session shutdown).
     *
     * @returns {{ destroyedCount: number, freedBytes: number }}
     */
    destroyAll() {
      this._generation += 1;
      this._pendingFences.clear();

      const toDestroy = [];
      for (let i = 0; i < RESOURCE_KINDS.length; i += 1) {
        const table = this._tables.get(RESOURCE_KINDS[i]);
        for (const entry of table.values()) {
          toDestroy.push(entry);
        }
        table.clear();
      }

      for (let i = 0; i < this._pendingRetire.length; i += 1) {
        toDestroy.push(this._pendingRetire[i]);
      }
      this._pendingRetire = [];

      this._activeBytes = 0;
      this._pendingRetireBytes = 0;
      this._budgetExceeded = false;

      let destroyedCount = 0;
      let freedBytes = 0;

      for (let i = 0; i < toDestroy.length; i += 1) {
        const entry = toDestroy[i];
        freedBytes += entry.byteSize;
        if (this._destroyEntry(entry)) {
          destroyedCount += 1;
        }
      }

      return {
        destroyedCount,
        freedBytes,
      };
    }
  }

  const exportsObj = Object.freeze({
    DEFAULT_FRAMES_IN_FLIGHT,
    DEFAULT_SOFT_BUDGET_BYTES,
    RESOURCE_KIND,
    RESOURCE_KINDS,
    GpuResourceRegistry,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6ResourceRegistry = exportsObj;
  } else if (root) {
    root.PlayGta6ResourceRegistry = exportsObj;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      DEFAULT_FRAMES_IN_FLIGHT,
      DEFAULT_SOFT_BUDGET_BYTES,
      RESOURCE_KIND,
      RESOURCE_KINDS,
      GpuResourceRegistry,
    };
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
