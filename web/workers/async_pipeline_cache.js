// W102 / W138 / W168 / W203: priority-scheduled, bounded-concurrency async GPU pipeline compilation.
//
// A burst of ~200 createRenderPipeline() calls on one frame blocks the GPU process long enough to
// trip a Windows WDDM TDR and lose the Dawn instance. AsyncPipelineCache routes every compile
// through create{Render,Compute}PipelineAsync with at most `maxConcurrentCompiles` (default 4,
// 0 = unlimited, dynamically adjustable via setMaxConcurrentCompiles) compilations in flight,
// prioritizing graphics (render) pipelines ahead of background compute pipelines (or explicit
// numeric priority / named priority lanes, FIFO within equal priority), coalesces identical
// pipelines by `id` or `(kind, dedupeKey)` (promoting queued jobs when a higher-priority caller
// arrives or via promote()), retries once with a `fallbackDescriptor` (e.g. layout: 'auto') when
// the primary descriptor rejects, tracks compile-stall and queue-wait telemetry via getTelemetry()
// and getStats(), and drops results that resolve after reset(newDevice) (device loss) instead of
// writing them into the new device.
//
// Zero external dependencies; UMD export for classic Web Workers loaded through importScripts
// (globalThis.PlayGta6AsyncPipelineCache) and Node.js (module.exports), like render_state_packets.js.
// compile*() never rejects: it resolves { id, status: 'ready'|'failed'|'stale'|'deleted', ... }.

(function (globalScope) {
  'use strict';

const DEFAULT_MAX_CONCURRENT_PIPELINES = 4;
const DEFAULT_COMPILE_STALL_THRESHOLD_MS = 16;

const PIPELINE_KIND = Object.freeze({ RENDER: 'render', COMPUTE: 'compute' });

const DEFAULT_PIPELINE_PRIORITY = Object.freeze({
  render: 10,
  compute: 0,
});

const PIPELINE_PRIORITY_LANE = Object.freeze({
  BACKGROUND: -10,
  PREWARM: -10,
  LOW: 0,
  COMPUTE: 0,
  NORMAL: 10,
  RENDER: 10,
  HIGH: 20,
  URGENT: 100,
});

const PIPELINE_STATUS = Object.freeze({
  PENDING: 'pending',
  READY: 'ready',
  FAILED: 'failed',
  // The entry was dropped by reset(newDevice) before its compilation finished.
  STALE: 'stale',
  // The entry was removed with delete(id) before its compilation finished.
  DELETED: 'deleted',
});

const DEVICE_METHODS = Object.freeze({
  render: Object.freeze({ async: 'createRenderPipelineAsync', sync: 'createRenderPipeline' }),
  compute: Object.freeze({ async: 'createComputePipelineAsync', sync: 'createComputePipeline' }),
});

/**
 * @param {unknown} value
 * @returns {number}
 */
function normalizeMaxConcurrent(value) {
  const n = Number(value);
  return Number.isInteger(n) && n >= 1 ? n : DEFAULT_MAX_CONCURRENT_PIPELINES;
}

/**
 * @param {unknown} value
 * @returns {number}
 */
function normalizeMaxConcurrentCompiles(value) {
  if (value === 0) return 0;
  const n = Number(value);
  return Number.isInteger(n) && n >= 1 ? n : DEFAULT_MAX_CONCURRENT_PIPELINES;
}

/**
 * @param {unknown} value
 * @returns {number}
 */
function normalizeStallThresholdMs(value) {
  if (value === 0) return 0;
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? n : DEFAULT_COMPILE_STALL_THRESHOLD_MS;
}

/**
 * @param {unknown} value
 * @returns {boolean}
 */
function hasFinitePriority(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

/**
 * Parse a numeric priority or named priority lane (e.g. 'urgent', 'high', 'normal', 'low', 'background', 'prewarm').
 * @param {unknown} value
 * @returns {number | null}
 */
function parsePriorityValue(value) {
  if (hasFinitePriority(value)) return value;
  if (typeof value === 'string') {
    const key = value.trim().toUpperCase();
    if (key && Object.prototype.hasOwnProperty.call(PIPELINE_PRIORITY_LANE, key)) {
      return PIPELINE_PRIORITY_LANE[key];
    }
  }
  return null;
}

/**
 * @param {'render' | 'compute'} kind
 * @param {unknown} priority
 * @param {boolean} prioritizeGraphics
 * @param {unknown} [lane]
 * @returns {number}
 */
function resolvePipelinePriority(kind, priority, prioritizeGraphics, lane) {
  const parsedPriority = parsePriorityValue(priority);
  if (parsedPriority !== null) return parsedPriority;
  const parsedLane = parsePriorityValue(lane);
  if (parsedLane !== null) return parsedLane;
  if (!prioritizeGraphics) return 0;
  return kind === PIPELINE_KIND.RENDER
    ? DEFAULT_PIPELINE_PRIORITY.render
    : DEFAULT_PIPELINE_PRIORITY.compute;
}

/**
 * @param {number} priority
 * @returns {'urgent' | 'high' | 'normal' | 'low' | 'background'}
 */
function classifyPriorityLane(priority) {
  if (priority >= PIPELINE_PRIORITY_LANE.URGENT) return 'urgent';
  if (priority >= PIPELINE_PRIORITY_LANE.HIGH) return 'high';
  if (priority >= PIPELINE_PRIORITY_LANE.NORMAL) return 'normal';
  if (priority >= PIPELINE_PRIORITY_LANE.LOW) return 'low';
  return 'background';
}

/**
 * @param {'render' | 'compute'} kind
 * @param {string | null} dedupeKey
 * @returns {string | null}
 */
function makeDedupeMapKey(kind, dedupeKey) {
  return dedupeKey !== null ? `${kind}\0${dedupeKey}` : null;
}

function defaultNow() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

/**
 * @param {unknown} err
 * @returns {Error}
 */
function toError(err) {
  if (err instanceof Error) return err;
  if (err && typeof err === 'object' && typeof err.message === 'string') {
    const wrapped = new Error(err.message);
    wrapped.cause = err;
    return wrapped;
  }
  return new Error(String(err));
}

function createDeferred() {
  let resolve;
  const promise = new Promise((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

class AsyncPipelineCache {
  /**
   * @param {{
   *   device?: GPUDevice | null,
   *   maxConcurrent?: number,
   *   maxConcurrentCompiles?: number,
   *   prioritizeGraphics?: boolean,
   *   stallThresholdMs?: number,
   *   nowFn?: (() => number) | null,
   *   onReady?: ((id: unknown, pipeline: object, meta: object) => void) | null,
   *   onError?: ((id: unknown, err: Error, meta: object) => void) | null,
   *   onStall?: ((detail: object) => void) | null,
   *   logger?: { warn?: Function, debug?: Function } | null,
   * }} [options]
   */
  constructor({
    device = null,
    maxConcurrent,
    maxConcurrentCompiles,
    prioritizeGraphics = true,
    stallThresholdMs = DEFAULT_COMPILE_STALL_THRESHOLD_MS,
    nowFn = null,
    onReady = null,
    onError = null,
    onStall = null,
    logger = null,
  } = {}) {
    this.device = device;
    const hasMaxConcurrent = maxConcurrent !== undefined;
    const hasMaxConcurrentCompiles = maxConcurrentCompiles !== undefined;
    if (hasMaxConcurrentCompiles) {
      this.maxConcurrentCompiles = normalizeMaxConcurrentCompiles(maxConcurrentCompiles);
      if (hasMaxConcurrent) {
        const legacyMax = normalizeMaxConcurrent(maxConcurrent);
        this.maxConcurrent = this.maxConcurrentCompiles > 0
          ? Math.min(legacyMax, this.maxConcurrentCompiles)
          : legacyMax;
      } else {
        this.maxConcurrent = this.maxConcurrentCompiles;
      }
    } else {
      this.maxConcurrent = normalizeMaxConcurrent(maxConcurrent);
      this.maxConcurrentCompiles = this.maxConcurrent;
    }
    this._concurrencyLimit = this.maxConcurrent > 0 ? this.maxConcurrent : Infinity;
    this.prioritizeGraphics = prioritizeGraphics !== false;
    this.stallThresholdMs = normalizeStallThresholdMs(stallThresholdMs);
    this._nowFn = typeof nowFn === 'function' ? nowFn : defaultNow;
    this.onReady = typeof onReady === 'function' ? onReady : null;
    this.onError = typeof onError === 'function' ? onError : null;
    this.onStall = typeof onStall === 'function' ? onStall : null;
    this.logger = logger;
    this.generation = 0;

    /** id -> { id, kind, state, pipeline, error, job, dedupeKey, deferred } */
    this._entries = new Map();
    /** `${kind}\0${dedupeKey}` -> pending job, or { kind, done: true, pipeline } once compiled */
    this._dedupe = new Map();
    this._queue = [];
    this._nextSeq = 0;
    this._inFlight = 0;
    this._idleWaiters = [];

    this._stats = {
      peakQueueDepth: 0,
      compiled: 0,
      failed: 0,
      dedupeHits: 0,
      syncFallbacks: 0,
      autoLayoutRetries: 0,
      discardedStale: 0,
    };
    this._peakInFlightCompiles = 0;
    this._failedAsyncCompiles = 0;
    this._asyncCompileCount = 0;
    this._asyncCompileTotalMs = 0;
    this._maxCompileMs = 0;
    this._lastCompileMs = 0;
    this._priorityPromotions = 0;
    this._queuedPromotions = 0;
    this._dedupedCount = 0;
    this._inFlightDedupedCount = 0;
    this._queuedDedupedCount = 0;
    this._startedJobCount = 0;
    this._totalQueueWaitMs = 0;
    this._maxQueueWaitMs = 0;
    this._stallCount = 0;
  }

  /**
   * Dynamically adjust the maximum number of concurrent async pipeline compiles.
   * Raising the limit immediately drains eligible queued compiles.
   * @param {number} limit positive integer, or 0 for unlimited.
   * @returns {number} the active maxConcurrentCompiles limit.
   */
  setMaxConcurrentCompiles(limit) {
    const normalized = normalizeMaxConcurrentCompiles(limit);
    this.maxConcurrentCompiles = normalized;
    this.maxConcurrent = normalized;
    this._concurrencyLimit = normalized > 0 ? normalized : Infinity;
    this._pump();
    return this.maxConcurrentCompiles;
  }

  /**
   * Promote a queued pipeline compilation by entry `id` or `dedupeKey` to a higher priority
   * (numeric or named lane such as 'urgent', 'high', 'normal').
   * @param {unknown} idOrKey
   * @param {number | string} [priority]
   * @param {'render' | 'compute' | null} [kind]
   * @returns {boolean} whether any queued job was promoted.
   */
  promote(idOrKey, priority = PIPELINE_PRIORITY_LANE.URGENT, kind = null) {
    const targetPriority = parsePriorityValue(priority);
    if (targetPriority === null) return false;

    const entry = this._entries.get(idOrKey);
    if (entry && entry.state === PIPELINE_STATUS.PENDING && entry.job && !entry.job.started) {
      if (kind !== null && entry.kind !== kind) return false;
      return this._promoteQueuedJob(entry.job, targetPriority);
    }

    if (typeof idOrKey === 'string' && idOrKey !== '') {
      const kinds = kind === PIPELINE_KIND.RENDER || kind === PIPELINE_KIND.COMPUTE
        ? [kind]
        : [PIPELINE_KIND.RENDER, PIPELINE_KIND.COMPUTE];
      let promoted = false;
      for (const candidateKind of kinds) {
        const dedupeMapKey = makeDedupeMapKey(candidateKind, idOrKey);
        const shared = this._dedupe.get(dedupeMapKey);
        if (shared && !shared.done && !shared.started) {
          if (this._promoteQueuedJob(shared, targetPriority)) {
            promoted = true;
          }
        }
      }
      return promoted;
    }

    return false;
  }

  /**
   * Compile a render pipeline without blocking the GPU process on a burst.
   * @param {unknown} id
   * @param {GPURenderPipelineDescriptor} descriptor
   * @param {{ dedupeKey?: string, priority?: number, allowSyncFallback?: boolean,
   *   fallbackDescriptor?: GPURenderPipelineDescriptor | null }} [options]
   * @returns {Promise<{ id: unknown, status: string, pipeline?: GPURenderPipeline, error?: Error }>}
   */
  compileRenderPipeline(id, descriptor, options = {}) {
    return this._compile(PIPELINE_KIND.RENDER, id, descriptor, options);
  }

  /**
   * Compute counterpart of compileRenderPipeline().
   * @param {unknown} id
   * @param {GPUComputePipelineDescriptor} descriptor
   * @param {{ dedupeKey?: string, priority?: number, allowSyncFallback?: boolean,
   *   fallbackDescriptor?: GPUComputePipelineDescriptor | null }} [options]
   * @returns {Promise<{ id: unknown, status: string, pipeline?: GPUComputePipeline, error?: Error }>}
   */
  compileComputePipeline(id, descriptor, options = {}) {
    return this._compile(PIPELINE_KIND.COMPUTE, id, descriptor, options);
  }

  /** @returns {object | null} the ready pipeline for `id`, else null. */
  get(id) {
    const entry = this._entries.get(id);
    return entry && entry.state === PIPELINE_STATUS.READY ? entry.pipeline : null;
  }

  isPending(id) {
    const entry = this._entries.get(id);
    return !!entry && entry.state === PIPELINE_STATUS.PENDING;
  }

  hasFailed(id) {
    const entry = this._entries.get(id);
    return !!entry && entry.state === PIPELINE_STATUS.FAILED;
  }

  /**
   * Forget `id`. A queued compilation that no other id is waiting on is dropped from the queue;
   * an in-flight one still finishes (it holds a GPU slot) but no longer writes into `id`.
   * @returns {boolean} whether an entry existed.
   */
  delete(id) {
    const entry = this._entries.get(id);
    if (!entry) return false;
    this._entries.delete(id);
    const job = entry.job;
    if (job) {
      job.ids.delete(id);
      if (job.ids.size === 0 && !job.started) {
        const index = this._queue.indexOf(job);
        if (index >= 0) this._queue.splice(index, 1);
        if (job.dedupeMapKey !== null && this._dedupe.get(job.dedupeMapKey) === job) {
          this._dedupe.delete(job.dedupeMapKey);
        }
        this._checkIdle();
      }
    }
    if (entry.state === PIPELINE_STATUS.PENDING) entry.deferred.resolve({ id, status: PIPELINE_STATUS.DELETED });
    return true;
  }

  /**
   * Switch to a new device (device loss / re-init). Clears every table and the queue; anything the
   * old device resolves or rejects later is counted in `discardedStale` and otherwise ignored.
   * Callers still awaiting a compile from the old generation get { status: 'stale' }.
   * @param {GPUDevice | null} newDevice
   */
  reset(newDevice = null) {
    this.generation += 1;
    const pending = [];
    for (const entry of this._entries.values()) {
      if (entry.state === PIPELINE_STATUS.PENDING) pending.push(entry);
    }
    this._entries.clear();
    this._dedupe.clear();
    this._queue = [];
    this._inFlight = 0;
    this.device = newDevice;
    for (const entry of pending) {
      entry.deferred.resolve({ id: entry.id, status: PIPELINE_STATUS.STALE });
    }
    this._checkIdle();
  }

  /** @returns {Promise<void>} resolves once nothing is queued or in flight for this generation. */
  whenIdle() {
    if (this._isIdle()) return Promise.resolve();
    return new Promise((resolve) => this._idleWaiters.push(resolve));
  }

  flush() {
    return this.whenIdle();
  }

  getStats() {
    const avgCompileMs = this._asyncCompileCount > 0
      ? this._asyncCompileTotalMs / this._asyncCompileCount
      : 0;
    const avgQueueWaitMs = this._startedJobCount > 0
      ? this._totalQueueWaitMs / this._startedJobCount
      : 0;
    const stats = {
      generation: this.generation,
      maxConcurrent: this.maxConcurrent,
      queued: this._queue.length,
      inFlight: this._inFlight,
      ...this._stats,
      inFlightCompiles: this._inFlight,
      peakInFlightCompiles: this._peakInFlightCompiles,
      activeCompiles: this._inFlight,
      peakConcurrentCompiles: this._peakInFlightCompiles,
      queuedCompiles: this._queue.length,
      failedAsyncCompiles: this._failedAsyncCompiles,
      avgAsyncCompileMs: avgCompileMs,
      avgCompileMs,
      maxCompileMs: this._maxCompileMs,
      priorityPromotions: this._priorityPromotions,
    };
    // W203 canonical telemetry aliases exposed on getStats() as non-enumerable properties
    // so legacy deepEqual snapshots keep their exact own-enumerable shape while callers reading
    // stats.inFlightCount / queuedCount / promotedCount / dedupedCount / completedCount / failedCount
    // work without needing a separate call. Use getTelemetry() for an all-enumerable snapshot.
    Object.defineProperties(stats, {
      inFlightCount: { value: this._inFlight, enumerable: false, configurable: true },
      queuedCount: { value: this._queue.length, enumerable: false, configurable: true },
      promotedCount: { value: this._queuedPromotions, enumerable: false, configurable: true },
      dedupedCount: { value: this._dedupedCount, enumerable: false, configurable: true },
      completedCount: { value: this._stats.compiled, enumerable: false, configurable: true },
      failedCount: { value: this._stats.failed, enumerable: false, configurable: true },
      stallCount: { value: this._stallCount, enumerable: false, configurable: true },
      maxQueueWaitMs: { value: this._maxQueueWaitMs, enumerable: false, configurable: true },
      avgQueueWaitMs: { value: avgQueueWaitMs, enumerable: false, configurable: true },
    });
    return stats;
  }

  /**
   * W203: Return an all-enumerable compile queue & stall telemetry snapshot.
   */
  getTelemetry() {
    const avgCompileMs = this._asyncCompileCount > 0
      ? this._asyncCompileTotalMs / this._asyncCompileCount
      : 0;
    const avgQueueWaitMs = this._startedJobCount > 0
      ? this._totalQueueWaitMs / this._startedJobCount
      : 0;
    return {
      generation: this.generation,
      maxConcurrentCompiles: this.maxConcurrentCompiles,
      inFlightCount: this._inFlight,
      peakInFlightCount: this._peakInFlightCompiles,
      queuedCount: this._queue.length,
      peakQueuedCount: this._stats.peakQueueDepth,
      promotedCount: this._queuedPromotions,
      priorityPromotions: this._priorityPromotions,
      dedupedCount: this._dedupedCount,
      inFlightDedupedCount: this._inFlightDedupedCount,
      queuedDedupedCount: this._queuedDedupedCount,
      completedCount: this._stats.compiled,
      failedCount: this._stats.failed,
      failedAsyncCompiles: this._failedAsyncCompiles,
      maxCompileMs: this._maxCompileMs,
      avgCompileMs,
      lastCompileMs: this._lastCompileMs,
      maxQueueWaitMs: this._maxQueueWaitMs,
      avgQueueWaitMs,
      stallThresholdMs: this.stallThresholdMs,
      stallCount: this._stallCount,
      queueByLane: this._getQueuedLaneBreakdown(),
    };
  }

  // -------------------------------------------------------------------------------------------

  _compile(kind, id, descriptor, options) {
    const opts = options && typeof options === 'object' ? options : {};
    const dedupeKey = typeof opts.dedupeKey === 'string' && opts.dedupeKey !== '' ? opts.dedupeKey : null;
    const dedupeMapKey = makeDedupeMapKey(kind, dedupeKey);
    const parsedPriority = parsePriorityValue(opts.priority);
    const parsedLane = parsedPriority === null ? parsePriorityValue(opts.lane) : null;
    const hasExplicitPriority = parsedPriority !== null || parsedLane !== null;
    const priority = resolvePipelinePriority(
      kind,
      parsedPriority !== null ? parsedPriority : parsedLane,
      this.prioritizeGraphics,
    );
    const allowSyncFallback = opts.allowSyncFallback !== false;
    const fallbackDescriptor = opts.fallbackDescriptor || null;

    const existing = this._entries.get(id);
    if (existing && existing.state !== PIPELINE_STATUS.FAILED) {
      if (existing.state === PIPELINE_STATUS.PENDING) {
        this._dedupedCount += 1;
        if (existing.job && existing.job.started) {
          this._inFlightDedupedCount += 1;
        } else {
          this._queuedDedupedCount += 1;
          if (existing.job) {
            if (!existing.job.fallbackDescriptor && fallbackDescriptor) {
              existing.job.fallbackDescriptor = fallbackDescriptor;
            }
            if (existing.kind === kind || hasExplicitPriority) {
              this._promoteQueuedJob(existing.job, priority);
            }
          }
        }
        return existing.deferred.promise;
      }
      return Promise.resolve({ id, status: PIPELINE_STATUS.READY, pipeline: existing.pipeline });
    }

    const entry = {
      id,
      kind,
      state: PIPELINE_STATUS.PENDING,
      pipeline: null,
      error: null,
      job: null,
      dedupeKey,
      deferred: createDeferred(),
    };
    this._entries.set(id, entry);

    if (dedupeMapKey !== null) {
      const shared = this._dedupe.get(dedupeMapKey);
      if (shared && shared.kind === kind) {
        this._stats.dedupeHits += 1;
        this._dedupedCount += 1;
        if (shared.done) {
          this._finishReady(entry, shared.pipeline, { dedupe: true, syncFallback: false, retried: false });
        } else {
          entry.job = shared;
          shared.ids.add(id);
          if (shared.started) {
            this._inFlightDedupedCount += 1;
          } else {
            this._queuedDedupedCount += 1;
            if (!shared.fallbackDescriptor && fallbackDescriptor) {
              shared.fallbackDescriptor = fallbackDescriptor;
            }
            this._promoteQueuedJob(shared, priority);
          }
        }
        return entry.deferred.promise;
      }
    }

    const job = {
      kind,
      descriptor,
      fallbackDescriptor,
      dedupeKey,
      dedupeMapKey,
      priority,
      seq: (this._nextSeq += 1),
      generation: this.generation,
      ownerId: id,
      ids: new Set([id]),
      started: false,
      enqueuedAtMs: this._now(),
      startedAtMs: 0,
      retried: false,
    };
    entry.job = job;

    const methods = DEVICE_METHODS[kind];
    const device = this.device;
    if (!device || typeof device[methods.async] !== 'function') {
      this._compileSync(job, device, methods, allowSyncFallback);
      return entry.deferred.promise;
    }

    if (dedupeMapKey !== null) this._dedupe.set(dedupeMapKey, job);
    this._enqueueJob(job);
    this._pump();
    return entry.deferred.promise;
  }

  _enqueueJob(job) {
    let insertIndex = this._queue.length;
    for (let i = 0; i < this._queue.length; i += 1) {
      const queued = this._queue[i];
      if (job.priority > queued.priority || (job.priority === queued.priority && job.seq < queued.seq)) {
        insertIndex = i;
        break;
      }
    }
    const promotedAhead = insertIndex < this._queue.length;
    this._queue.splice(insertIndex, 0, job);
    if (promotedAhead) {
      this._priorityPromotions += 1;
    }
    if (this._queue.length > this._stats.peakQueueDepth) {
      this._stats.peakQueueDepth = this._queue.length;
    }
  }

  _promoteQueuedJob(job, newPriority) {
    if (!job || job.started || !hasFinitePriority(newPriority) || newPriority <= job.priority) return false;
    const currentIndex = this._queue.indexOf(job);
    if (currentIndex < 0) return false;
    this._queue.splice(currentIndex, 1);
    job.priority = newPriority;
    let insertIndex = this._queue.length;
    for (let i = 0; i < this._queue.length; i += 1) {
      const queued = this._queue[i];
      if (job.priority > queued.priority || (job.priority === queued.priority && job.seq < queued.seq)) {
        insertIndex = i;
        break;
      }
    }
    this._queue.splice(insertIndex, 0, job);
    this._priorityPromotions += 1;
    this._queuedPromotions += 1;
    return true;
  }

  _getQueuedLaneBreakdown() {
    const lanes = { urgent: 0, high: 0, normal: 0, low: 0, background: 0 };
    for (let i = 0; i < this._queue.length; i += 1) {
      lanes[classifyPriorityLane(this._queue[i].priority)] += 1;
    }
    return lanes;
  }

  _compileSync(job, device, methods, allowSyncFallback) {
    if (!device || !allowSyncFallback || typeof device[methods.sync] !== 'function') {
      const reason = !device
        ? 'AsyncPipelineCache: no GPU device'
        : `AsyncPipelineCache: device has no ${methods.async}` +
          (allowSyncFallback ? ` or ${methods.sync}` : ' and allowSyncFallback is false');
      this._completeJob(job, null, new Error(reason), false);
      return;
    }
    this._stats.syncFallbacks += 1;
    let pipeline = null;
    let error = null;
    try {
      pipeline = device[methods.sync](job.descriptor);
    } catch (err) {
      error = toError(err);
      if (job.fallbackDescriptor) {
        job.retried = true;
        this._stats.autoLayoutRetries += 1;
        try {
          pipeline = device[methods.sync](job.fallbackDescriptor);
          error = null;
        } catch (retryErr) {
          error = toError(retryErr);
        }
      }
    }
    if (!error && (pipeline === null || pipeline === undefined)) {
      error = new Error(`AsyncPipelineCache: ${methods.sync} returned no pipeline`);
    }
    if (!error && job.dedupeMapKey !== null) {
      this._dedupe.set(job.dedupeMapKey, { kind: job.kind, done: true, pipeline });
    }
    this._completeJob(job, pipeline, error, true);
  }

  _pump() {
    while (this._inFlight < this._concurrencyLimit && this._queue.length > 0) {
      const job = this._queue.shift();
      job.started = true;
      job.startedAtMs = this._now();
      const queueWaitMs = Math.max(0, job.startedAtMs - job.enqueuedAtMs);
      this._startedJobCount += 1;
      this._totalQueueWaitMs += queueWaitMs;
      if (queueWaitMs > this._maxQueueWaitMs) {
        this._maxQueueWaitMs = queueWaitMs;
      }
      this._inFlight += 1;
      if (this._inFlight > this._peakInFlightCompiles) {
        this._peakInFlightCompiles = this._inFlight;
      }
      this._issue(job, job.descriptor);
    }
  }

  _issue(job, descriptor) {
    const device = this.device;
    const method = DEVICE_METHODS[job.kind].async;
    let promise;
    try {
      promise = Promise.resolve(device[method](descriptor));
    } catch (err) {
      promise = Promise.reject(err);
    }
    promise.then(
      (pipeline) => this._onSettled(job, pipeline, null),
      (err) => this._onSettled(job, null, toError(err)),
    );
  }

  _onSettled(job, pipeline, error) {
    if (job.generation !== this.generation) {
      this._stats.discardedStale += 1;
      this._log('debug', 'discarded pipeline compile from a previous device generation', {
        kind: job.kind,
        generation: job.generation,
        failed: !!error,
      });
      return;
    }

    if (!error && (pipeline === null || pipeline === undefined)) {
      error = new Error(`AsyncPipelineCache: ${DEVICE_METHODS[job.kind].async} resolved no pipeline`);
    }

    if (error && job.fallbackDescriptor && !job.retried) {
      job.retried = true;
      this._stats.autoLayoutRetries += 1;
      this._log('warn', 'pipeline compile rejected; retrying with fallbackDescriptor', {
        kind: job.kind,
        message: error.message,
      });
      this._issue(job, job.fallbackDescriptor);
      return; // keeps its concurrency slot
    }

    let slotReleased = false;
    const releaseSlot = () => {
      if (slotReleased) return;
      slotReleased = true;
      if (job.generation === this.generation && this._inFlight > 0) {
        this._inFlight -= 1;
      }
    };

    try {
      const elapsedMs = Math.max(0, this._now() - job.startedAtMs);
      this._lastCompileMs = elapsedMs;
      this._asyncCompileCount += 1;
      this._asyncCompileTotalMs += elapsedMs;
      if (elapsedMs > this._maxCompileMs) {
        this._maxCompileMs = elapsedMs;
      }
      if (this.stallThresholdMs > 0 && elapsedMs >= this.stallThresholdMs) {
        this._stallCount += 1;
        this._invokeStall({
          id: job.ownerId,
          ids: Array.from(job.ids),
          kind: job.kind,
          dedupeKey: job.dedupeKey,
          elapsedMs,
          thresholdMs: this.stallThresholdMs,
          retried: job.retried,
          failed: !!error,
          generation: this.generation,
        });
      }
      if (error) {
        this._failedAsyncCompiles += 1;
      }
      releaseSlot();
      if (job.dedupeMapKey !== null && this._dedupe.get(job.dedupeMapKey) === job) {
        if (error) this._dedupe.delete(job.dedupeMapKey);
        else this._dedupe.set(job.dedupeMapKey, { kind: job.kind, done: true, pipeline });
      }
      this._completeJob(job, pipeline, error, false);
    } catch (err) {
      this._log('warn', 'AsyncPipelineCache internal error', { message: toError(err).message });
    } finally {
      releaseSlot();
      if (job.generation === this.generation) {
        this._pump();
        this._checkIdle();
      }
    }
  }

  _completeJob(job, pipeline, error, syncFallback) {
    if (error) this._stats.failed += 1;
    else this._stats.compiled += 1;
    for (const id of job.ids) {
      const entry = this._entries.get(id);
      if (!entry || entry.job !== job) continue;
      const meta = { dedupe: id !== job.ownerId, syncFallback, retried: job.retried };
      if (error) this._finishFailed(entry, error, meta);
      else this._finishReady(entry, pipeline, meta);
    }
  }

  _finishReady(entry, pipeline, meta) {
    entry.state = PIPELINE_STATUS.READY;
    entry.pipeline = pipeline;
    entry.job = null;
    const fullMeta = this._meta(entry, meta);
    entry.deferred.resolve({ id: entry.id, status: PIPELINE_STATUS.READY, pipeline, ...fullMeta });
    this._invoke(this.onReady, 'onReady', entry.id, pipeline, fullMeta);
  }

  _finishFailed(entry, error, meta) {
    entry.state = PIPELINE_STATUS.FAILED;
    entry.error = error;
    entry.job = null;
    const fullMeta = this._meta(entry, meta);
    entry.deferred.resolve({ id: entry.id, status: PIPELINE_STATUS.FAILED, error, ...fullMeta });
    this._invoke(this.onError, 'onError', entry.id, error, fullMeta);
  }

  _meta(entry, meta) {
    return {
      kind: entry.kind,
      generation: this.generation,
      dedupeKey: entry.dedupeKey,
      dedupe: !!meta.dedupe,
      syncFallback: !!meta.syncFallback,
      retried: !!meta.retried,
    };
  }

  _invoke(callback, name, id, value, meta) {
    if (!callback) return;
    try {
      const result = callback(id, value, meta);
      if (result && typeof result.then === 'function') {
        result.then(null, (err) => {
          this._log('warn', `AsyncPipelineCache ${name} callback rejected`, { message: toError(err).message });
        });
      }
    } catch (err) {
      this._log('warn', `AsyncPipelineCache ${name} callback threw`, { message: toError(err).message });
    }
  }

  _invokeStall(detail) {
    if (!this.onStall) return;
    try {
      const result = this.onStall(detail);
      if (result && typeof result.then === 'function') {
        result.then(null, (err) => {
          this._log('warn', 'AsyncPipelineCache onStall callback rejected', { message: toError(err).message });
        });
      }
    } catch (err) {
      this._log('warn', 'AsyncPipelineCache onStall callback threw', { message: toError(err).message });
    }
  }

  _isIdle() {
    return this._inFlight === 0 && this._queue.length === 0;
  }

  _checkIdle() {
    if (!this._isIdle() || this._idleWaiters.length === 0) return;
    const waiters = this._idleWaiters;
    this._idleWaiters = [];
    for (const resolve of waiters) resolve();
  }

  _now() {
    try {
      const value = Number(this._nowFn());
      return Number.isFinite(value) ? value : 0;
    } catch {
      return 0;
    }
  }

  _log(level, message, detail) {
    const logger = this.logger;
    if (!logger || typeof logger[level] !== 'function') return;
    try {
      logger[level](message, detail);
    } catch {
      // A broken logger must never break pipeline compilation.
    }
  }
}

const exportedApi = {
  AsyncPipelineCache,
  DEFAULT_MAX_CONCURRENT_PIPELINES,
  DEFAULT_PIPELINE_PRIORITY,
  DEFAULT_COMPILE_STALL_THRESHOLD_MS,
  PIPELINE_KIND,
  PIPELINE_PRIORITY_LANE,
  PIPELINE_STATUS,
};

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6AsyncPipelineCache = exportedApi;
  } else if (globalScope) {
    globalScope.PlayGta6AsyncPipelineCache = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
