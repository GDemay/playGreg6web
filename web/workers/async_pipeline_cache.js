// W102: bounded-concurrency async GPU pipeline compilation.
//
// A burst of ~200 createRenderPipeline() calls on one frame blocks the GPU process long enough to
// trip a Windows WDDM TDR and lose the Dawn instance. AsyncPipelineCache routes every compile
// through create{Render,Compute}PipelineAsync with at most `maxConcurrent` compilations in flight
// (FIFO queue for the rest), dedupes identical pipelines by `dedupeKey`, retries once with a
// `fallbackDescriptor` (e.g. layout: 'auto') when the primary descriptor rejects, and drops results
// that resolve after reset(newDevice) (device loss) instead of writing them into the new device.
//
// Zero external dependencies; UMD export for classic Web Workers loaded through importScripts
// (globalThis.PlayGta6AsyncPipelineCache) and Node.js (module.exports), like render_state_packets.js.
// compile*() never rejects: it resolves { id, status: 'ready'|'failed'|'stale'|'deleted', ... }.

(function (globalScope) {
  'use strict';

const DEFAULT_MAX_CONCURRENT_PIPELINES = 4;

const PIPELINE_KIND = Object.freeze({ RENDER: 'render', COMPUTE: 'compute' });

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
   *   onReady?: ((id: unknown, pipeline: object, meta: object) => void) | null,
   *   onError?: ((id: unknown, err: Error, meta: object) => void) | null,
   *   logger?: { warn?: Function, debug?: Function } | null,
   * }} [options]
   */
  constructor({ device = null, maxConcurrent = DEFAULT_MAX_CONCURRENT_PIPELINES, onReady = null,
    onError = null, logger = null } = {}) {
    this.device = device;
    this.maxConcurrent = normalizeMaxConcurrent(maxConcurrent);
    this.onReady = typeof onReady === 'function' ? onReady : null;
    this.onError = typeof onError === 'function' ? onError : null;
    this.logger = logger;
    this.generation = 0;

    /** id -> { id, kind, state, pipeline, error, job, dedupeKey, deferred } */
    this._entries = new Map();
    /** dedupeKey -> pending job, or { done: true, pipeline } once it compiled */
    this._dedupe = new Map();
    this._queue = [];
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
  }

  /**
   * Compile a render pipeline without blocking the GPU process on a burst.
   * @param {unknown} id
   * @param {GPURenderPipelineDescriptor} descriptor
   * @param {{ dedupeKey?: string, allowSyncFallback?: boolean,
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
   * @param {{ dedupeKey?: string, allowSyncFallback?: boolean,
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
        if (job.dedupeKey !== null && this._dedupe.get(job.dedupeKey) === job) {
          this._dedupe.delete(job.dedupeKey);
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
    return {
      generation: this.generation,
      maxConcurrent: this.maxConcurrent,
      queued: this._queue.length,
      inFlight: this._inFlight,
      ...this._stats,
    };
  }

  // -------------------------------------------------------------------------------------------

  _compile(kind, id, descriptor, options) {
    const opts = options && typeof options === 'object' ? options : {};
    const dedupeKey = typeof opts.dedupeKey === 'string' && opts.dedupeKey !== '' ? opts.dedupeKey : null;
    const allowSyncFallback = opts.allowSyncFallback !== false;
    const fallbackDescriptor = opts.fallbackDescriptor || null;

    const existing = this._entries.get(id);
    if (existing && existing.state !== PIPELINE_STATUS.FAILED) {
      return existing.state === PIPELINE_STATUS.PENDING
        ? existing.deferred.promise
        : Promise.resolve({ id, status: PIPELINE_STATUS.READY, pipeline: existing.pipeline });
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

    if (dedupeKey !== null) {
      const shared = this._dedupe.get(dedupeKey);
      if (shared && shared.kind === kind) {
        this._stats.dedupeHits += 1;
        if (shared.done) {
          this._finishReady(entry, shared.pipeline, { dedupe: true, syncFallback: false, retried: false });
        } else {
          entry.job = shared;
          shared.ids.add(id);
        }
        return entry.deferred.promise;
      }
    }

    const job = {
      kind,
      descriptor,
      fallbackDescriptor,
      dedupeKey,
      generation: this.generation,
      ownerId: id,
      ids: new Set([id]),
      started: false,
      retried: false,
    };
    entry.job = job;

    const methods = DEVICE_METHODS[kind];
    const device = this.device;
    if (!device || typeof device[methods.async] !== 'function') {
      this._compileSync(job, device, methods, allowSyncFallback);
      return entry.deferred.promise;
    }

    if (dedupeKey !== null) this._dedupe.set(dedupeKey, job);
    this._queue.push(job);
    if (this._queue.length > this._stats.peakQueueDepth) this._stats.peakQueueDepth = this._queue.length;
    this._pump();
    return entry.deferred.promise;
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
    if (!error && job.dedupeKey !== null) {
      this._dedupe.set(job.dedupeKey, { kind: job.kind, done: true, pipeline });
    }
    this._completeJob(job, pipeline, error, true);
  }

  _pump() {
    while (this._inFlight < this.maxConcurrent && this._queue.length > 0) {
      const job = this._queue.shift();
      job.started = true;
      this._inFlight += 1;
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
    try {
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
      this._inFlight -= 1;
      if (job.dedupeKey !== null && this._dedupe.get(job.dedupeKey) === job) {
        if (error) this._dedupe.delete(job.dedupeKey);
        else this._dedupe.set(job.dedupeKey, { kind: job.kind, done: true, pipeline });
      }
      this._completeJob(job, pipeline, error, false);
      this._pump();
      this._checkIdle();
    } catch (err) {
      this._log('warn', 'AsyncPipelineCache internal error', { message: toError(err).message });
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

  _isIdle() {
    return this._inFlight === 0 && this._queue.length === 0;
  }

  _checkIdle() {
    if (!this._isIdle() || this._idleWaiters.length === 0) return;
    const waiters = this._idleWaiters;
    this._idleWaiters = [];
    for (const resolve of waiters) resolve();
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
  PIPELINE_KIND,
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
