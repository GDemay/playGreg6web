// Bounded Web Worker pool and offscreen task dispatcher.
//
// Intended for background decompression and shader pre-parsing: jobs carry
// transferable ArrayBuffers, can be prioritised, timed out and cancelled, and a
// crashing worker script is contained by replacement, retry and a circuit breaker.
//
// Pure ES module, no top-level side effects. Works on the main thread and inside
// a worker, and under Node with an injected worker factory (see tests/test_worker_pool.mjs).
//
// ---------------------------------------------------------------------------
// Message protocol (structured-clone friendly plain objects)
//
//   pool -> worker
//     { id, type, payload }        run job `id` with handler `type`
//     { id, cancel: true }         cooperative cancel request for job `id`
//     { __watchdogPing: true, pingId }  health watchdog ping
//
//   worker -> pool
//     { id, ok: true,  result }    job finished
//     { id, ok: false, error }     job threw; error = { name, message, stack, code? }
//     { id, progress }             optional progress value (any cloneable)
//     { id, cancelled: true }      acknowledgement of a cancel request
//     { __watchdogPong: true, pingId }  health watchdog pong
//
//   Messages with an unknown, finished or duplicate id are ignored.
//
// Ownership rule for transferables
//   Buffers listed in `submit(..., { transfer })` are TRANSFERRED to the worker and
//   are detached (byteLength 0) in the caller as soon as `submit` returns. Never
//   reuse them. Results may come back with transferables (see `serveWorkerPool`);
//   the receiving side then owns them. Because a transferred buffer cannot be sent
//   twice, a job submitted with a non-empty transfer list is never retried after a
//   worker crash.
// ---------------------------------------------------------------------------

/** Job priorities. Higher numeric value runs first; FIFO within equal priority. */
export const PRIORITY = Object.freeze({ HIGH: 10, NORMAL: 0, LOW: -10 });

const PRIORITY_NAMES = Object.freeze({
  high: PRIORITY.HIGH,
  normal: PRIORITY.NORMAL,
  low: PRIORITY.LOW,
});

export const DEFAULT_CANCEL_GRACE_MS = 1000;
export const DEFAULT_BREAKER_MAX_CRASHES = 5;
export const DEFAULT_BREAKER_WINDOW_MS = 10000;
export const DEFAULT_MAX_RESPAWNS_PER_SLOT = 16;
export const DEFAULT_RESPAWN_BACKOFF_BASE_MS = 0;
export const DEFAULT_RESPAWN_BACKOFF_MAX_MS = 5000;
export const DEFAULT_MAX_BOOT_RETRIES = 3;
export const DEFAULT_BOOT_RETRY_BASE_DELAY_MS = 50;
export const DEFAULT_BOOT_RETRY_MAX_DELAY_MS = 500;
export const DEFAULT_WATCHDOG_INTERVAL_MS = 0;
export const DEFAULT_STALL_TIMEOUT_MS = 5000;

/** Base class for all errors raised by the pool itself. */
export class WorkerPoolError extends Error {
  constructor(message, code, options) {
    super(message, options);
    this.name = 'WorkerPoolError';
    this.code = code;
  }
}

/** The pool was terminated (or is terminating) and cannot accept or finish the job. */
export class PoolTerminatedError extends WorkerPoolError {
  constructor(message = 'Worker pool terminated') {
    super(message, 'POOL_TERMINATED');
    this.name = 'PoolTerminatedError';
  }
}

/** The bounded queue is full and the overflow policy is `'reject'`. */
export class QueueFullError extends WorkerPoolError {
  constructor(limit) {
    super(`Worker pool queue is full (limit ${limit})`, 'QUEUE_FULL');
    this.name = 'QueueFullError';
    this.limit = limit;
  }
}

/** The job was cancelled through its AbortSignal. Named `AbortError` like DOM aborts. */
export class JobAbortedError extends WorkerPoolError {
  constructor(reason) {
    super('Worker job aborted', 'ABORTED', { cause: reason });
    this.name = 'AbortError';
    this.reason = reason;
  }
}

/** The job exceeded its per-job timeout. */
export class JobTimeoutError extends WorkerPoolError {
  constructor(timeoutMs) {
    super(`Worker job timed out after ${timeoutMs} ms`, 'TASK_TIMEOUT');
    this.name = 'JobTimeoutError';
    this.timeoutMs = timeoutMs;
  }
}

/** The worker running the job crashed (error, messageerror, exit or spawn failure). */
export class WorkerCrashError extends WorkerPoolError {
  constructor(reason, cause) {
    super(`Worker crashed: ${reason}`, 'WORKER_CRASH', cause === undefined ? undefined : { cause });
    this.name = 'WorkerCrashError';
    this.reason = reason;
  }
}

/** A worker failed to respond to health watchdog heartbeats or report progress within the stall timeout. */
export class WorkerStallError extends WorkerPoolError {
  constructor(stallTimeoutMs, slotIndex, missedPings = 1) {
    const slotLabel = Number.isInteger(slotIndex) ? `slot ${slotIndex}` : 'worker';
    super(
      `Worker ${slotLabel} stalled (no heartbeat or activity for ${stallTimeoutMs} ms)`,
      'WORKER_STALL'
    );
    this.name = 'WorkerStallError';
    this.stallTimeoutMs = stallTimeoutMs;
    this.slotIndex = slotIndex;
    this.missedPings = missedPings;
  }
}

/** Too many crashes within the breaker window; the pool refuses further work. */
export class CircuitOpenError extends WorkerPoolError {
  constructor(crashes, windowMs, cause) {
    super(
      `Worker pool circuit open: ${crashes} worker crashes within ${windowMs} ms`,
      'CIRCUIT_OPEN',
      cause === undefined ? undefined : { cause }
    );
    this.name = 'CircuitOpenError';
    this.crashes = crashes;
    this.windowMs = windowMs;
  }
}

const STANDARD_ERRORS = {
  Error,
  TypeError,
  RangeError,
  SyntaxError,
  ReferenceError,
  EvalError,
  URIError,
};

/**
 * Serialise any thrown value into a structured-clone safe envelope.
 * @param {unknown} err
 * @returns {{ name: string, message: string, stack?: string, code?: string|number }}
 */
export function serializeError(err) {
  if (err !== null && typeof err === 'object') {
    const out = {
      name: typeof err.name === 'string' ? err.name : 'Error',
      message: typeof err.message === 'string' ? err.message : String(err.message ?? ''),
    };
    if (typeof err.stack === 'string') out.stack = err.stack;
    if (typeof err.code === 'string' || typeof err.code === 'number') out.code = err.code;
    return out;
  }
  return { name: 'Error', message: String(err) };
}

/**
 * Rebuild a real `Error` from a serialised envelope. The result is flagged
 * `remote = true` so callers can tell worker-side failures from pool errors.
 * @param {{ name?: string, message?: string, stack?: string, code?: string|number }} data
 * @returns {Error}
 */
export function deserializeError(data) {
  const src = data !== null && typeof data === 'object' ? data : { message: String(data) };
  const name = typeof src.name === 'string' ? src.name : 'Error';
  const Ctor = Object.prototype.hasOwnProperty.call(STANDARD_ERRORS, name)
    ? STANDARD_ERRORS[name]
    : Error;
  const err = new Ctor(typeof src.message === 'string' ? src.message : '');
  err.name = name;
  if (typeof src.stack === 'string') err.stack = src.stack;
  if (src.code !== undefined) err.code = src.code;
  err.remote = true;
  return err;
}

/**
 * Suggested pool size: hardware threads minus headroom for the game's own
 * pthreads, clamped to `[min, max]`.
 * @param {{ hardwareConcurrency?: number, reserve?: number, min?: number, max?: number }} [opts]
 * @returns {number}
 */
export function defaultPoolSize(opts = {}) {
  const { reserve = 2, min = 1, max = 8 } = opts;
  let hc = opts.hardwareConcurrency;
  if (hc === undefined) {
    hc = typeof navigator !== 'undefined' && navigator ? navigator.hardwareConcurrency : undefined;
  }
  if (!Number.isFinite(hc) || hc < 1) hc = 4;
  return Math.max(min, Math.min(max, Math.floor(hc) - reserve));
}

function normalisePriority(priority) {
  if (priority === undefined || priority === null) return PRIORITY.NORMAL;
  if (typeof priority === 'string' && Object.prototype.hasOwnProperty.call(PRIORITY_NAMES, priority)) {
    return PRIORITY_NAMES[priority];
  }
  if (typeof priority === 'number' && Number.isFinite(priority)) {
    return priority === 0 ? 0 : priority;
  }
  throw new RangeError(`Invalid priority: ${String(priority)}`);
}

function jobHigherPriority(a, b) {
  if (a.priority !== b.priority) return a.priority > b.priority;
  return a.seq < b.seq;
}

/** Attach a listener to either a DOM-style or a Node-style worker and return an idempotent detach function. */
function listen(target, event, handler) {
  let active = true;
  if (typeof target.addEventListener === 'function') {
    target.addEventListener(event, handler);
    return () => {
      if (!active) return;
      active = false;
      if (typeof target.removeEventListener === 'function') {
        try {
          target.removeEventListener(event, handler);
        } catch (_) {
          // Ignore teardown listener removal errors.
        }
      }
    };
  } else if (typeof target.on === 'function') {
    // Node worker_threads: message/error/exit hand over the raw value.
    const wrapped = (value) => {
      if (event === 'message') handler({ data: value });
      else if (event === 'error') handler({ error: value, message: value && value.message });
      else handler({ code: value });
    };
    target.on(event, wrapped);
    return () => {
      if (!active) return;
      active = false;
      try {
        if (typeof target.off === 'function') target.off(event, wrapped);
        else if (typeof target.removeListener === 'function') target.removeListener(event, wrapped);
      } catch (_) {
        // Ignore teardown listener removal errors.
      }
    };
  }
  return () => {};
}

function describeErrorEvent(ev) {
  if (!ev) return 'unknown error';
  const msg = ev.message || (ev.error && ev.error.message);
  return msg ? String(msg) : 'unknown error';
}

const TRANSIENT_BOOT_ERROR_NAMES = new Set(['NetworkError']);
const TRANSIENT_BOOT_ERROR_CODES = new Set([
  'NETWORK_ERROR',
  'ERR_NETWORK',
  'BOOT_TRANSIENT',
  'ECONNRESET',
  'ETIMEDOUT',
  'ECONNREFUSED',
  'EAGAIN',
]);
const TRANSIENT_BOOT_MESSAGE_RE =
  /\b(?:networkerror|network\s+error|importscripts|failed\s+to\s+(?:fetch|load\s+script)|transient|econnreset|etimedout|econnrefused|eagain)\b|net::err_/i;

function isTransientBootError(err, rawEvent) {
  const candidates = [];
  if (err && typeof err === 'object') {
    candidates.push(err);
    if (err.cause && typeof err.cause === 'object') candidates.push(err.cause);
  }
  if (rawEvent && typeof rawEvent === 'object') {
    candidates.push(rawEvent);
    if (rawEvent.error && typeof rawEvent.error === 'object') candidates.push(rawEvent.error);
  }
  for (const obj of candidates) {
    if (obj.transient === true || obj.isTransient === true) return true;
    if (typeof obj.name === 'string' && TRANSIENT_BOOT_ERROR_NAMES.has(obj.name)) return true;
    if (typeof obj.code === 'string' && TRANSIENT_BOOT_ERROR_CODES.has(obj.code.toUpperCase())) return true;
    const msg =
      typeof obj.reason === 'string'
        ? obj.reason
        : typeof obj.message === 'string'
          ? obj.message
          : '';
    if (msg && TRANSIENT_BOOT_MESSAGE_RE.test(msg)) return true;
  }
  return false;
}

/**
 * Bounded pool of dedicated workers speaking the protocol above.
 */
export class WorkerPool {
  /**
   * @param {object} options
   * @param {() => object} [options.workerFactory] Returns a Worker-like object.
   * @param {string|URL} [options.url] Used when no factory is given: `new Worker(url, workerOptions)`.
   * @param {object} [options.workerOptions] Default `{ type: 'module' }`.
   * @param {number} [options.maxWorkers] Default `defaultPoolSize()`.
   * @param {number} [options.minWorkers=0] Idle shrink never goes below this.
   * @param {number} [options.minWarmWorkers=0] Warm idle worker reserve floor (clamped to `[0, maxWorkers]`); idle trimming never evicts below this.
   * @param {number} [options.idleTimeoutMs=0] Terminate surplus idle workers after this long (0 = never).
   * @param {number} [options.maxQueue=Infinity] Maximum queued (not yet running) jobs.
   * @param {'reject'|'wait'} [options.overflow='reject'] Policy when the queue is full.
   * @param {number} [options.cancelGraceMs=1000] Grace before a cancelled running job's worker is terminated.
   * @param {number} [options.retries=0] Retries per job after a worker crash (never after a job error).
   * @param {number} [options.defaultTimeoutMs=0] Per-job run timeout default (0 = none).
   * @param {number} [options.taskTimeoutMs] Alias for `defaultTimeoutMs` (0 = disabled unless > 0).
   * @param {number} [options.maxRespawnsPerSlot=16] Max worker respawns per slot before capping (Infinity = unlimited).
   * @param {{ maxCrashes?: number, windowMs?: number }|false} [options.breaker] Circuit breaker; `false` disables.
   * @param {number} [options.maxConsecutiveCrashes] Crash threshold within `crashWindowMs` before opening circuit (`0` disables; overrides `breaker.maxCrashes`).
   * @param {number} [options.crashWindowMs] Sliding crash-rate window in ms (overrides `breaker.windowMs`).
   * @param {number} [options.respawnBackoffBaseMs=0] Initial exponential backoff delay in ms before respawning a crashed worker (`0` = immediate).
   * @param {number} [options.respawnBackoffMaxMs=5000] Upper bound in ms for exponential respawn backoff.
   * @param {number} [options.maxBootRetries=3] Max exponential-backoff retries on transient worker boot/startup failures (`0` disables).
   * @param {number} [options.bootRetryBaseDelayMs=50] Initial exponential backoff delay in ms for transient worker boot retries.
   * @param {number} [options.bootRetryMaxDelayMs=500] Upper bound in ms for transient worker boot retry backoff.
   * @param {number} [options.watchdogIntervalMs=0] Health watchdog ping interval in ms (`0` disables).
   * @param {number} [options.stallTimeoutMs=5000] Max time in ms without heartbeat pong or job activity before a worker is classified as stalled.
   * @param {(info: object) => void} [options.onWorkerStall] Optional callback invoked when a worker stalls and is recycled.
   * @param {(info: object) => void} [options.onWorkerCrash] Optional callback invoked when a worker crashes.
   * @param {() => number} [options.now] Clock in ms.
   * @param {Function} [options.setTimeout]
   * @param {Function} [options.clearTimeout]
   * @param {Function} [options.setInterval]
   * @param {Function} [options.clearInterval]
   */
  constructor(options = {}) {
    const {
      workerFactory,
      url,
      workerOptions = { type: 'module' },
      maxWorkers = defaultPoolSize(),
      minWorkers = 0,
      minWarmWorkers = 0,
      idleTimeoutMs = 0,
      maxQueue = Infinity,
      overflow = 'reject',
      cancelGraceMs = DEFAULT_CANCEL_GRACE_MS,
      retries = 0,
      defaultTimeoutMs = 0,
      taskTimeoutMs,
      maxRespawnsPerSlot = DEFAULT_MAX_RESPAWNS_PER_SLOT,
      breaker = {},
      maxConsecutiveCrashes,
      crashWindowMs,
      respawnBackoffBaseMs = DEFAULT_RESPAWN_BACKOFF_BASE_MS,
      respawnBackoffMaxMs = DEFAULT_RESPAWN_BACKOFF_MAX_MS,
      maxBootRetries = DEFAULT_MAX_BOOT_RETRIES,
      bootRetryBaseDelayMs = DEFAULT_BOOT_RETRY_BASE_DELAY_MS,
      bootRetryMaxDelayMs = DEFAULT_BOOT_RETRY_MAX_DELAY_MS,
      watchdogIntervalMs = DEFAULT_WATCHDOG_INTERVAL_MS,
      stallTimeoutMs = DEFAULT_STALL_TIMEOUT_MS,
      onWorkerStall = null,
      onWorkerCrash = null,
      now,
      setTimeout: setTimeoutFn,
      clearTimeout: clearTimeoutFn,
      setInterval: setIntervalFn,
      clearInterval: clearIntervalFn,
    } = options;

    if (typeof workerFactory !== 'function' && url === undefined) {
      throw new TypeError('WorkerPool requires a workerFactory function or a url');
    }
    if (!Number.isInteger(maxWorkers) || maxWorkers < 1) {
      throw new RangeError(`maxWorkers must be a positive integer, got ${maxWorkers}`);
    }
    if (!Number.isInteger(minWorkers) || minWorkers < 0 || minWorkers > maxWorkers) {
      throw new RangeError(`minWorkers must be an integer in [0, maxWorkers], got ${minWorkers}`);
    }
    if (
      typeof minWarmWorkers !== 'number' ||
      Number.isNaN(minWarmWorkers) ||
      !Number.isFinite(minWarmWorkers)
    ) {
      throw new RangeError(`minWarmWorkers must be a finite number, got ${minWarmWorkers}`);
    }
    const clampedMinWarmWorkers = Math.max(0, Math.min(maxWorkers, Math.trunc(minWarmWorkers))) || 0;
    if (overflow !== 'reject' && overflow !== 'wait') {
      throw new RangeError(`overflow must be 'reject' or 'wait', got ${overflow}`);
    }
    if (!(maxQueue >= 0)) throw new RangeError(`maxQueue must be >= 0, got ${maxQueue}`);
    if (!Number.isInteger(retries) || retries < 0) {
      throw new RangeError(`retries must be a non-negative integer, got ${retries}`);
    }
    const resolvedTimeoutMs = taskTimeoutMs !== undefined ? taskTimeoutMs : defaultTimeoutMs;
    if (typeof resolvedTimeoutMs !== 'number' || Number.isNaN(resolvedTimeoutMs) || resolvedTimeoutMs < 0) {
      throw new RangeError(`taskTimeoutMs must be a non-negative number, got ${resolvedTimeoutMs}`);
    }
    if (
      maxRespawnsPerSlot !== Infinity &&
      (!Number.isInteger(maxRespawnsPerSlot) || maxRespawnsPerSlot < 0)
    ) {
      throw new RangeError(
        `maxRespawnsPerSlot must be a non-negative integer or Infinity, got ${maxRespawnsPerSlot}`
      );
    }
    if (
      maxConsecutiveCrashes !== undefined &&
      (!Number.isInteger(maxConsecutiveCrashes) || maxConsecutiveCrashes < 0)
    ) {
      throw new RangeError(
        `maxConsecutiveCrashes must be a non-negative integer, got ${maxConsecutiveCrashes}`
      );
    }
    if (
      crashWindowMs !== undefined &&
      (typeof crashWindowMs !== 'number' || Number.isNaN(crashWindowMs) || !Number.isFinite(crashWindowMs) || crashWindowMs < 0)
    ) {
      throw new RangeError(`crashWindowMs must be a finite non-negative number, got ${crashWindowMs}`);
    }
    if (
      typeof respawnBackoffBaseMs !== 'number' ||
      Number.isNaN(respawnBackoffBaseMs) ||
      !Number.isFinite(respawnBackoffBaseMs) ||
      respawnBackoffBaseMs < 0
    ) {
      throw new RangeError(
        `respawnBackoffBaseMs must be a finite non-negative number, got ${respawnBackoffBaseMs}`
      );
    }
    if (
      typeof respawnBackoffMaxMs !== 'number' ||
      Number.isNaN(respawnBackoffMaxMs) ||
      !Number.isFinite(respawnBackoffMaxMs) ||
      respawnBackoffMaxMs < 0
    ) {
      throw new RangeError(
        `respawnBackoffMaxMs must be a finite non-negative number, got ${respawnBackoffMaxMs}`
      );
    }
    if (!Number.isInteger(maxBootRetries) || maxBootRetries < 0) {
      throw new RangeError(`maxBootRetries must be a non-negative integer, got ${maxBootRetries}`);
    }
    if (
      typeof bootRetryBaseDelayMs !== 'number' ||
      Number.isNaN(bootRetryBaseDelayMs) ||
      !Number.isFinite(bootRetryBaseDelayMs) ||
      bootRetryBaseDelayMs < 0
    ) {
      throw new RangeError(
        `bootRetryBaseDelayMs must be a finite non-negative number, got ${bootRetryBaseDelayMs}`
      );
    }
    if (
      typeof bootRetryMaxDelayMs !== 'number' ||
      Number.isNaN(bootRetryMaxDelayMs) ||
      !Number.isFinite(bootRetryMaxDelayMs) ||
      bootRetryMaxDelayMs < 0
    ) {
      throw new RangeError(
        `bootRetryMaxDelayMs must be a finite non-negative number, got ${bootRetryMaxDelayMs}`
      );
    }
    if (
      typeof watchdogIntervalMs !== 'number' ||
      Number.isNaN(watchdogIntervalMs) ||
      !Number.isFinite(watchdogIntervalMs) ||
      watchdogIntervalMs < 0
    ) {
      throw new RangeError(
        `watchdogIntervalMs must be a finite non-negative number, got ${watchdogIntervalMs}`
      );
    }
    if (
      typeof stallTimeoutMs !== 'number' ||
      Number.isNaN(stallTimeoutMs) ||
      !Number.isFinite(stallTimeoutMs) ||
      stallTimeoutMs <= 0
    ) {
      throw new RangeError(
        `stallTimeoutMs must be a finite positive number, got ${stallTimeoutMs}`
      );
    }
    if (onWorkerStall !== null && onWorkerStall !== undefined && typeof onWorkerStall !== 'function') {
      throw new TypeError('onWorkerStall must be a function or null');
    }
    if (onWorkerCrash !== null && onWorkerCrash !== undefined && typeof onWorkerCrash !== 'function') {
      throw new TypeError('onWorkerCrash must be a function or null');
    }
    if (setIntervalFn !== undefined && typeof setIntervalFn !== 'function') {
      throw new TypeError('setInterval must be a function');
    }
    if (clearIntervalFn !== undefined && typeof clearIntervalFn !== 'function') {
      throw new TypeError('clearInterval must be a function');
    }

    const breakerObj = breaker && typeof breaker === 'object' ? breaker : null;
    const breakerDisabled = breaker === false && maxConsecutiveCrashes === undefined;

    this._factory =
      typeof workerFactory === 'function' ? workerFactory : () => new Worker(url, workerOptions);
    this._maxWorkers = maxWorkers;
    this._minWorkers = minWorkers;
    this._minWarmWorkers = clampedMinWarmWorkers;
    this._idleTimeoutMs = idleTimeoutMs;
    this._maxQueue = maxQueue;
    this._overflow = overflow;
    this._cancelGraceMs = cancelGraceMs;
    this._retries = retries;
    this._defaultTimeoutMs = resolvedTimeoutMs;
    this._maxRespawnsPerSlot = maxRespawnsPerSlot;
    this._breakerMax = breakerDisabled
      ? 0
      : (maxConsecutiveCrashes ?? (breakerObj ? breakerObj.maxCrashes : undefined) ?? DEFAULT_BREAKER_MAX_CRASHES);
    this._breakerWindow = breakerDisabled
      ? 0
      : (crashWindowMs ?? (breakerObj ? breakerObj.windowMs : undefined) ?? DEFAULT_BREAKER_WINDOW_MS);
    this._respawnBackoffBaseMs = respawnBackoffBaseMs;
    this._respawnBackoffMaxMs = respawnBackoffMaxMs;
    this._maxBootRetries = maxBootRetries;
    this._bootRetryBaseDelayMs = bootRetryBaseDelayMs;
    this._bootRetryMaxDelayMs = bootRetryMaxDelayMs;
    this._watchdogIntervalMs = watchdogIntervalMs;
    this._stallTimeoutMs = stallTimeoutMs;
    this._onWorkerStallCallback = onWorkerStall || null;
    this._onWorkerCrashCallback = onWorkerCrash || null;
    this._now =
      now ||
      (() => (typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now()));
    this._setTimeout = setTimeoutFn || ((fn, ms) => globalThis.setTimeout(fn, ms));
    this._clearTimeout = clearTimeoutFn || ((t) => globalThis.clearTimeout(t));
    this._useTimeoutForWatchdog = !setIntervalFn && Boolean(setTimeoutFn);
    this._setInterval = setIntervalFn || ((fn, ms) => globalThis.setInterval(fn, ms));
    this._clearInterval = clearIntervalFn || ((t) => globalThis.clearInterval(t));
    this._watchdogTimer = null;
    this._nextPingId = 0;

    this._slots = new Set();
    this._slotRespawns = new Array(maxWorkers).fill(0);
    this._slotNeedsRespawn = new Array(maxWorkers).fill(false);
    this._slotCrashStreak = new Array(maxWorkers).fill(0);
    this._slotBackoffUntil = new Array(maxWorkers).fill(0);
    this._slotBackoffTimers = new Array(maxWorkers).fill(null);
    this._slotBootRetries = new Array(maxWorkers).fill(0);
    this._slotBootRetryTimers = new Array(maxWorkers).fill(null);
    this._slotBootRetryPending = new Array(maxWorkers).fill(false);
    this._slotBootRetryForWarm = new Array(maxWorkers).fill(false);
    this._lastSpawnDeferredByBootRetry = false;
    this._queue = [];
    this._blocked = [];
    this._running = 0;
    this._nextId = 1;
    this._nextSeq = 1;
    this._idleSeq = 0;
    this._crashTimes = [];
    this._lastCrash = null;
    this._circuitOpen = false;
    this._closed = false;
    this._terminatePromise = null;
    this._terminateState = null;
    this._drainWaiters = [];

    this._counters = {
      submitted: 0,
      completed: 0,
      failed: 0,
      cancelled: 0,
      rejected: 0,
      crashed: 0,
      retried: 0,
      hardKills: 0,
      spawned: 0,
      warmReuses: 0,
      timedOutTasks: 0,
      respawnedWorkers: 0,
      trimmedWorkers: 0,
      circuitBreakerTrips: 0,
      peakQueueDepth: 0,
      bootRetryCount: 0,
      bootRetrySuccessCount: 0,
      bootRetryExhaustedCount: 0,
      stalledWorkers: 0,
      watchdogRecycled: 0,
      watchdogPingsSent: 0,
      watchdogPongsReceived: 0,
    };
    this._waitTotal = 0;
    this._waitCount = 0;
    this._runTotal = 0;
    this._runCount = 0;

    if (this._watchdogIntervalMs > 0) {
      this._startWatchdog();
    }
  }

  /**
   * Submit a job.
   * @param {string} type Handler name registered with `serveWorkerPool` in the worker.
   * @param {*} payload Structured-cloneable job input.
   * @param {object} [opts]
   * @param {Transferable[]} [opts.transfer] Transferred (detached in the caller) with the payload.
   * @param {'high'|'normal'|'low'|number} [opts.priority='normal'] Higher numeric priority runs first; FIFO within equal priority.
   * @param {AbortSignal} [opts.signal]
   * @param {number} [opts.timeoutMs] Run-time limit, measured from when the job starts running.
   * @param {(value: *) => void} [opts.onProgress]
   * @returns {Promise<*>} Never throws synchronously; always settles exactly once.
   */
  submit(type, payload, opts = {}) {
    return new Promise((resolve, reject) => {
      let priority;
      let timeoutMs;
      try {
        if (typeof type !== 'string' || type === '') throw new TypeError('job type must be a non-empty string');
        priority = normalisePriority(opts.priority);
        timeoutMs =
          opts.timeoutMs === undefined || opts.timeoutMs === null
            ? this._defaultTimeoutMs
            : opts.timeoutMs;
        if (typeof timeoutMs !== 'number' || Number.isNaN(timeoutMs) || timeoutMs < 0) {
          throw new RangeError(`timeoutMs must be a non-negative number, got ${String(timeoutMs)}`);
        }
      } catch (err) {
        reject(err);
        return;
      }
      if (this._closed) {
        this._counters.rejected++;
        reject(new PoolTerminatedError('Worker pool is terminated'));
        return;
      }
      if (this._circuitOpen) {
        this._counters.rejected++;
        reject(this._circuitError());
        return;
      }
      const signal = opts.signal;
      if (signal && signal.aborted) {
        this._counters.cancelled++;
        reject(new JobAbortedError(signal.reason));
        return;
      }

      const job = {
        id: this._nextId++,
        seq: this._nextSeq++,
        type,
        payload,
        transfer: Array.isArray(opts.transfer) ? opts.transfer : [],
        priority,
        timeoutMs,
        onProgress: typeof opts.onProgress === 'function' ? opts.onProgress : null,
        signal: signal || null,
        resolve,
        reject,
        state: 'queued',
        slot: null,
        settled: false,
        submittedAt: this._now(),
        startedAt: 0,
        crashRetries: 0,
        timeoutTimer: null,
        abortListener: null,
        _heapIndex: -1,
      };
      this._counters.submitted++;

      if (signal) {
        job.abortListener = () => this._onAbort(job);
        signal.addEventListener('abort', job.abortListener, { once: true });
      }

      this._enqueue(job);
      this._pump();
      if (job.state === 'queued' && this._queuedCount() > this._maxQueue) {
        if (this._overflow === 'reject') {
          this._removeFromQueue(job);
          this._counters.rejected++;
          this._detach(job);
          job.settled = true;
          reject(new QueueFullError(this._maxQueue));
        } else {
          this._removeFromQueue(job);
          job.state = 'blocked';
          this._blocked.push(job);
        }
      }
      this._recordPeakQueueDepth();
    });
  }

  /** Resolves once no job is queued, blocked or running. */
  drain() {
    if (this._isIdle()) return Promise.resolve();
    return new Promise((resolve) => this._drainWaiters.push(resolve));
  }

  /**
   * Shut the pool down. Idempotent; calling it again with `graceful: false`
   * upgrades a pending graceful shutdown to an immediate one.
   * @param {{ graceful?: boolean }} [opts]
   * @returns {Promise<void>}
   */
  terminate({ graceful = false } = {}) {
    if (!this._terminatePromise) {
      this._closed = true;
      this._terminatePromise = new Promise((resolve) => {
        this._terminateState = { resolve, graceful };
      });
      if (graceful) this._checkIdle();
      else this._finishImmediate();
    } else if (!graceful && this._terminateState && this._terminateState.graceful) {
      this._terminateState.graceful = false;
      this._finishImmediate();
    }
    return this._terminatePromise;
  }

  /** Alias for `terminate(opts)`. */
  destroy(opts) {
    return this.terminate(opts);
  }

  /** Alias for `terminate(opts)`. */
  terminateAll(opts) {
    return this.terminate(opts);
  }

  /**
   * Pre-spawn idle warm workers up to `count` (clamped to `[0, maxWorkers]`).
   * No-op if the pool is closed or the circuit breaker is open.
   * @param {number} [count=Math.max(this._minWorkers, this._minWarmWorkers)]
   * @returns {number} Current live worker count.
   */
  warmUp(count = this._effectiveMinWorkers()) {
    if (typeof count !== 'number' || !Number.isInteger(count) || count < 0) {
      throw new RangeError(`warmUp count must be a non-negative integer, got ${count}`);
    }
    if (this._closed || this._circuitOpen) return this._slots.size;
    const target = Math.min(count, this._maxWorkers);
    while (this._slots.size < target) {
      const slot = this._spawn({ forWarm: true });
      if (!slot) {
        if (this._lastSpawnDeferredByBootRetry) break;
        this._checkBreaker();
        break;
      }
      this._armIdle(slot);
    }
    return this._slots.size;
  }

  /**
   * Terminate excess idle workers cleanly down to `Math.max(minWarm, minWarmWorkers)`
   * (least-recently-used first) to release renderer memory without disrupting active tasks.
   * @param {number} [minWarm=0]
   * @returns {{ trimmed: number, remainingIdle: number, active: number }}
   */
  trimIdleWorkers(minWarm = 0) {
    if (typeof minWarm !== 'number' || !Number.isInteger(minWarm) || minWarm < 0) {
      throw new RangeError(`trimIdleWorkers minWarm must be a non-negative integer, got ${minWarm}`);
    }
    const effectiveMinWarm = Math.max(minWarm, this._minWarmWorkers);
    const idleSlots = [];
    let active = 0;
    for (const slot of this._slots) {
      if (slot.dead) continue;
      if (slot.state === 'idle') idleSlots.push(slot);
      else active++;
    }
    let trimmed = 0;
    if (idleSlots.length > effectiveMinWarm) {
      idleSlots.sort((a, b) => a.lastIdleSeq - b.lastIdleSeq);
      const toTrim = idleSlots.length - effectiveMinWarm;
      for (let i = 0; i < toTrim; i++) {
        this._destroySlot(idleSlots[i]);
        this._counters.trimmedWorkers++;
        trimmed++;
      }
    }
    return {
      trimmed,
      remainingIdle: idleSlots.length - trimmed,
      active,
    };
  }

  /** Re-close an open circuit breaker and reset per-slot respawn budgets and backoffs so new work is accepted again. */
  resetCircuitBreaker() {
    this._circuitOpen = false;
    this._crashTimes = [];
    this._lastCrash = null;
    this._slotRespawns.fill(0);
    this._slotNeedsRespawn.fill(false);
    this._slotCrashStreak.fill(0);
    this._clearAllBackoffTimers();
    this._clearAllBootRetryTimers(true);
  }

  /** Alias for `resetCircuitBreaker()`. */
  resetCircuit() {
    this.resetCircuitBreaker();
  }

  /** Reset cumulative telemetry counters and latency accumulators while keeping live pool state intact. */
  resetStats() {
    this._counters = {
      submitted: 0,
      completed: 0,
      failed: 0,
      cancelled: 0,
      rejected: 0,
      crashed: 0,
      retried: 0,
      hardKills: 0,
      spawned: 0,
      warmReuses: 0,
      timedOutTasks: 0,
      respawnedWorkers: 0,
      trimmedWorkers: 0,
      circuitBreakerTrips: 0,
      peakQueueDepth: 0,
      bootRetryCount: 0,
      bootRetrySuccessCount: 0,
      bootRetryExhaustedCount: 0,
      stalledWorkers: 0,
      watchdogRecycled: 0,
      watchdogPingsSent: 0,
      watchdogPongsReceived: 0,
    };
    this._waitTotal = 0;
    this._waitCount = 0;
    this._runTotal = 0;
    this._runCount = 0;
  }

  /** Snapshot of pool counters. Times are in clock milliseconds. */
  stats() {
    let busy = 0;
    for (const slot of this._slots) if (slot.state !== 'idle') busy++;
    const workers = this._slots.size;
    const idleWorkers = workers - busy;
    const queued = this._queuedCount();
    const blocked = this._blocked.length;
    const queuedTasks = queued + blocked;
    const avgRunMs = this._runCount ? this._runTotal / this._runCount : 0;
    return {
      workers,
      busy,
      activeWorkers: busy,
      idleWorkers,
      warmWorkersReserved: Math.min(idleWorkers, this._minWarmWorkers),
      queued,
      blocked,
      queuedTasks,
      running: this._running,
      submitted: this._counters.submitted,
      completed: this._counters.completed,
      completedTasks: this._counters.completed,
      failed: this._counters.failed,
      failedTasks: this._counters.failed,
      cancelled: this._counters.cancelled,
      rejected: this._counters.rejected,
      crashed: this._counters.crashed,
      retried: this._counters.retried,
      hardKills: this._counters.hardKills,
      spawned: this._counters.spawned,
      warmReuses: this._counters.warmReuses,
      timedOutTasks: this._counters.timedOutTasks,
      watchdogTimeouts: this._counters.timedOutTasks,
      respawnedWorkers: this._counters.respawnedWorkers,
      respawnCount: this._counters.respawnedWorkers,
      trimmedWorkers: this._counters.trimmedWorkers,
      circuitBreakerTrips: this._counters.circuitBreakerTrips,
      peakQueueDepth: this._counters.peakQueueDepth,
      peakQueuedTasks: this._counters.peakQueueDepth,
      bootRetryCount: this._counters.bootRetryCount,
      bootRetrySuccessCount: this._counters.bootRetrySuccessCount,
      bootRetryExhaustedCount: this._counters.bootRetryExhaustedCount,
      stalledWorkers: this._counters.stalledWorkers,
      watchdogRecycled: this._counters.watchdogRecycled,
      watchdogPingsSent: this._counters.watchdogPingsSent,
      watchdogPongsReceived: this._counters.watchdogPongsReceived,
      watchdogEnabled: this._watchdogIntervalMs > 0,
      avgWaitMs: this._waitCount ? this._waitTotal / this._waitCount : 0,
      avgRunMs,
      avgTaskMs: avgRunMs,
      circuitOpen: this._circuitOpen,
      terminated: this._closed,
    };
  }

  /** Alias for `stats()`. */
  getStats() {
    return this.stats();
  }

  /**
   * Returns a sanitized health snapshot with per-slot heartbeat/activity state,
   * stall and crash counters, circuit-breaker state, active timer counts, and
   * watchdog status. Never exposes raw worker handles or job payloads.
   */
  getHealthSnapshot() {
    const now = this._now();
    const slots = [];
    for (const slot of this._slots) {
      if (slot.dead) continue;
      slots.push({
        index: slot.index,
        state: slot.state,
        booted: slot.booted,
        busy: slot.state !== 'idle',
        jobId: slot.job ? slot.job.id : null,
        jobsHandled: slot.jobsHandled,
        spawnedAt: slot.spawnedAt,
        lastActivityAt: slot.lastActivityAt,
        idleForMs: Math.max(0, now - slot.lastActivityAt),
        lastPingSentAt: slot.lastPingSentAt,
        pendingPingId: slot.pendingPingId,
        consecutiveMissedPings: slot.consecutiveMissedPings,
        respawns: this._slotRespawns[slot.index],
        crashStreak: this._slotCrashStreak[slot.index],
      });
    }
    slots.sort((a, b) => a.index - b.index);
    const activeTimers = this._countActiveTimers();
    return {
      healthy: !this._closed && !this._circuitOpen,
      terminated: this._closed,
      circuitOpen: this._circuitOpen,
      circuitBreakerTrips: this._counters.circuitBreakerTrips,
      crashed: this._counters.crashed,
      stalledWorkers: this._counters.stalledWorkers,
      watchdogRecycled: this._counters.watchdogRecycled,
      watchdogPingsSent: this._counters.watchdogPingsSent,
      watchdogPongsReceived: this._counters.watchdogPongsReceived,
      watchdog: {
        enabled: this._watchdogIntervalMs > 0,
        active: this._watchdogTimer !== null,
        intervalMs: this._watchdogIntervalMs,
        stallTimeoutMs: this._stallTimeoutMs,
        pingsSent: this._counters.watchdogPingsSent,
        pongsReceived: this._counters.watchdogPongsReceived,
        stalledWorkers: this._counters.stalledWorkers,
        recycled: this._counters.watchdogRecycled,
      },
      activeTimers,
      workers: slots.length,
      maxWorkers: this._maxWorkers,
      queuedTasks: this._queuedCount() + this._blocked.length,
      running: this._running,
      slots,
      timestampMs: now,
    };
  }

  // ---- internals ---------------------------------------------------------

  _queuedCount() {
    return this._queue.length;
  }

  _recordPeakQueueDepth() {
    const depth = this._queuedCount() + this._blocked.length;
    if (depth > this._counters.peakQueueDepth) {
      this._counters.peakQueueDepth = depth;
    }
  }

  _isIdle() {
    return this._queuedCount() === 0 && this._blocked.length === 0 && this._running === 0;
  }

  _enqueue(job) {
    const q = this._queue;
    const idx = q.length;
    job._heapIndex = idx;
    q.push(job);
    this._siftUp(idx);
  }

  _removeFromQueue(job) {
    const q = this._queue;
    const idx = job._heapIndex;
    if (idx < 0 || idx >= q.length || q[idx] !== job) return false;
    job._heapIndex = -1;
    const last = q.pop();
    if (idx < q.length) {
      q[idx] = last;
      last._heapIndex = idx;
      if (!this._siftUp(idx)) {
        this._siftDown(idx);
      }
    }
    return true;
  }

  _nextQueued() {
    const q = this._queue;
    if (q.length === 0) return null;
    const top = q[0];
    top._heapIndex = -1;
    const last = q.pop();
    if (q.length > 0) {
      q[0] = last;
      last._heapIndex = 0;
      this._siftDown(0);
    }
    return top;
  }

  _drainQueued() {
    const q = this._queue;
    this._queue = [];
    for (let i = 0; i < q.length; i++) {
      q[i]._heapIndex = -1;
    }
    return q;
  }

  _siftUp(idx) {
    const q = this._queue;
    const job = q[idx];
    let moved = false;
    while (idx > 0) {
      const parentIdx = (idx - 1) >> 1;
      const parent = q[parentIdx];
      if (!jobHigherPriority(job, parent)) break;
      q[idx] = parent;
      parent._heapIndex = idx;
      idx = parentIdx;
      moved = true;
    }
    q[idx] = job;
    job._heapIndex = idx;
    return moved;
  }

  _siftDown(idx) {
    const q = this._queue;
    const len = q.length;
    const job = q[idx];
    for (;;) {
      const left = (idx << 1) + 1;
      if (left >= len) break;
      let best = left;
      const right = left + 1;
      if (right < len && jobHigherPriority(q[right], q[left])) {
        best = right;
      }
      if (!jobHigherPriority(q[best], job)) break;
      q[idx] = q[best];
      q[idx]._heapIndex = idx;
      idx = best;
    }
    q[idx] = job;
    job._heapIndex = idx;
  }

  _circuitError() {
    return new CircuitOpenError(this._crashTimes.length, this._breakerWindow, this._lastCrash || undefined);
  }

  /** Remove timers and abort listener from a job. */
  _detach(job) {
    if (job.timeoutTimer !== null) {
      this._clearTimeout(job.timeoutTimer);
      job.timeoutTimer = null;
    }
    if (job.abortListener && job.signal) {
      job.signal.removeEventListener('abort', job.abortListener);
      job.abortListener = null;
    }
    job.signal = null;
  }

  /**
   * Settle a job exactly once.
   * @param {object} job
   * @param {'ok'|'fail'|'cancel'} kind
   * @param {*} value
   */
  _settle(job, kind, value) {
    if (job.settled) return;
    job.settled = true;
    this._detach(job);
    const wasRunning = job.state === 'running';
    if (wasRunning) {
      this._running--;
      this._runTotal += Math.max(0, this._now() - job.startedAt);
      this._runCount++;
    }
    job.state = 'done';
    if (kind === 'ok') this._counters.completed++;
    else if (kind === 'cancel') this._counters.cancelled++;
    else this._counters.failed++;
    if (kind === 'ok') job.resolve(value);
    else job.reject(value);
  }

  _onAbort(job) {
    if (job.settled) return;
    const error = new JobAbortedError(job.signal ? job.signal.reason : undefined);
    if (job.state === 'running') {
      this._cancelRunning(job, error, 'cancel');
    } else {
      if (job.state === 'queued') this._removeFromQueue(job);
      else if (job.state === 'blocked') {
        const idx = this._blocked.indexOf(job);
        if (idx >= 0) this._blocked.splice(idx, 1);
      }
      this._settle(job, 'cancel', error);
    }
    this._pump();
    this._ensureMinWorkers();
    this._checkIdle();
  }

  _onTimeout(job) {
    job.timeoutTimer = null;
    if (job.settled || job.state !== 'running') return;
    this._counters.timedOutTasks++;
    this._cancelRunning(job, new JobTimeoutError(job.timeoutMs), 'fail');
    this._pump();
    this._ensureMinWorkers();
    this._checkIdle();
  }

  /** Reject a running job now, then try a cooperative cancel and fall back to killing the worker. */
  _cancelRunning(job, error, kind) {
    const slot = job.slot;
    this._settle(job, kind, error);
    slot.job = null;
    slot.state = 'cancelling';
    slot.cancelId = job.id;
    let postFailed = false;
    try {
      slot.worker.postMessage({ id: job.id, cancel: true });
    } catch (_) {
      postFailed = true;
    }
    if (postFailed || !(this._cancelGraceMs > 0)) {
      this._hardKill(slot);
    } else if (Number.isFinite(this._cancelGraceMs)) {
      slot.graceTimer = this._setTimeout(() => {
        slot.graceTimer = null;
        if (!slot.dead && slot.state === 'cancelling') {
          this._hardKill(slot);
          this._pump();
          this._ensureMinWorkers();
          this._checkIdle();
        }
      }, this._cancelGraceMs);
    }
  }

  _hardKill(slot) {
    this._counters.hardKills++;
    this._slotNeedsRespawn[slot.index] = true;
    this._destroySlot(slot);
  }

  _destroySlot(slot) {
    if (slot.dead) return;
    slot.dead = true;
    if (slot.idleTimer !== null) this._clearTimeout(slot.idleTimer);
    if (slot.graceTimer !== null) this._clearTimeout(slot.graceTimer);
    slot.idleTimer = null;
    slot.graceTimer = null;
    slot.pendingPingId = null;
    slot.job = null;
    if (Array.isArray(slot._detachListeners)) {
      const detachers = slot._detachListeners;
      slot._detachListeners = [];
      for (const detach of detachers) {
        try {
          detach();
        } catch (_) {
          // Ignore listener removal errors during teardown.
        }
      }
    }
    this._slots.delete(slot);
    const worker = slot.worker;
    slot.worker = null;
    if (worker) {
      try {
        worker.terminate();
      } catch (_) {
        // Already gone.
      }
    }
  }

  _countActiveTimers() {
    let total = this._watchdogTimer !== null ? 1 : 0;
    for (const slot of this._slots) {
      if (slot.dead) continue;
      if (slot.idleTimer !== null) total++;
      if (slot.graceTimer !== null) total++;
      if (slot.job && slot.job.timeoutTimer !== null) total++;
    }
    for (let i = 0; i < this._maxWorkers; i++) {
      if (this._slotBackoffTimers[i] !== null) total++;
      if (this._slotBootRetryTimers[i] !== null) total++;
    }
    return total;
  }

  _startWatchdog() {
    if (this._watchdogIntervalMs <= 0 || this._watchdogTimer !== null || this._closed) return;
    if (this._useTimeoutForWatchdog) {
      const scheduleNext = () => {
        if (this._closed && !(this._terminateState && this._terminateState.graceful)) {
          this._watchdogTimer = null;
          return;
        }
        this._watchdogTimer = this._setTimeout(() => {
          this._watchdogTimer = null;
          this._onWatchdogTick();
          if (
            this._watchdogIntervalMs > 0 &&
            (!this._closed || (this._terminateState && this._terminateState.graceful))
          ) {
            scheduleNext();
          }
        }, this._watchdogIntervalMs);
      };
      scheduleNext();
    } else {
      this._watchdogTimer = this._setInterval(() => {
        this._onWatchdogTick();
      }, this._watchdogIntervalMs);
    }
  }

  _stopWatchdog() {
    if (this._watchdogTimer === null) return;
    const timer = this._watchdogTimer;
    this._watchdogTimer = null;
    if (this._useTimeoutForWatchdog) {
      this._clearTimeout(timer);
    } else {
      this._clearInterval(timer);
    }
  }

  _onWatchdogTick() {
    const canRunWhileClosed = Boolean(this._terminateState && this._terminateState.graceful);
    if (this._closed && !canRunWhileClosed) {
      this._stopWatchdog();
      return;
    }
    const now = this._now();
    for (const slot of [...this._slots]) {
      if (slot.dead || !slot.worker) continue;
      if (slot.pendingPingId !== null) {
        slot.consecutiveMissedPings++;
        const silentMs = Math.max(0, now - slot.lastActivityAt);
        const pingWaitMs = Math.max(0, now - slot.lastPingSentAt);
        if (silentMs >= this._stallTimeoutMs || pingWaitMs >= this._stallTimeoutMs) {
          this._onWorkerStall(slot);
        }
      } else {
        const pingId = ++this._nextPingId;
        slot.pendingPingId = pingId;
        slot.lastPingSentAt = now;
        this._counters.watchdogPingsSent++;
        try {
          slot.worker.postMessage({ __watchdogPing: true, pingId });
        } catch (_) {
          slot.consecutiveMissedPings++;
          this._onWorkerStall(slot);
        }
      }
    }
  }

  _onWorkerStall(slot) {
    if (slot.dead) return;
    const now = this._now();
    const slotIndex = slot.index;
    const job = slot.job;
    const state = slot.state;
    const silentMs = Math.max(0, now - slot.lastActivityAt);
    const missedPings = Math.max(1, slot.consecutiveMissedPings);
    const stallError = new WorkerStallError(this._stallTimeoutMs, slotIndex, missedPings);

    this._counters.stalledWorkers++;
    this._counters.watchdogRecycled++;
    this._counters.hardKills++;

    if (this._onWorkerStallCallback) {
      try {
        this._onWorkerStallCallback({
          slotIndex,
          state,
          jobId: job ? job.id : null,
          jobType: job ? job.type : null,
          stallTimeoutMs: this._stallTimeoutMs,
          silentMs,
          missedPings,
          timestampMs: now,
          error: stallError,
        });
      } catch (_) {
        // Callback errors must never disrupt stall recovery.
      }
    }

    this._slotNeedsRespawn[slotIndex] = true;
    this._destroySlot(slot);

    if (job && !job.settled) {
      job.slot = null;
      if (job.crashRetries < this._retries && job.transfer.length === 0 && !this._closed) {
        this._running--;
        job.state = 'queued';
        if (job.timeoutTimer !== null) {
          this._clearTimeout(job.timeoutTimer);
          job.timeoutTimer = null;
        }
        job.crashRetries++;
        this._counters.retried++;
        this._enqueue(job);
      } else {
        this._settle(job, 'fail', stallError);
      }
    }

    this._pump();
    this._recordPeakQueueDepth();
    this._ensureMinWorkers();
    this._checkIdle();
  }

  _computeRespawnBackoffMs(streak) {
    if (!(this._respawnBackoffBaseMs > 0) || streak <= 0) return 0;
    const exp = Math.min(streak - 1, 30);
    const delay = this._respawnBackoffBaseMs * Math.pow(2, exp);
    return Math.min(this._respawnBackoffMaxMs, delay);
  }

  _clearSlotBackoffTimer(index) {
    const timer = this._slotBackoffTimers[index];
    if (timer !== null) {
      this._clearTimeout(timer);
      this._slotBackoffTimers[index] = null;
    }
    this._slotBackoffUntil[index] = 0;
  }

  _effectiveMinWorkers() {
    return Math.max(this._minWorkers, this._minWarmWorkers);
  }

  _computeBootRetryDelayMs(attempt) {
    if (!(this._bootRetryBaseDelayMs > 0) || attempt <= 0) return 0;
    const exp = Math.min(attempt - 1, 30);
    const delay = this._bootRetryBaseDelayMs * Math.pow(2, exp);
    return Math.min(this._bootRetryMaxDelayMs, delay);
  }

  _clearSlotBootRetryTimer(index, resetCount = true) {
    const timer = this._slotBootRetryTimers[index];
    if (timer !== null) {
      this._clearTimeout(timer);
      this._slotBootRetryTimers[index] = null;
    }
    this._slotBootRetryPending[index] = false;
    this._slotBootRetryForWarm[index] = false;
    if (resetCount) this._slotBootRetries[index] = 0;
  }

  _clearAllBootRetryTimers(resetCount = true) {
    for (let i = 0; i < this._maxWorkers; i++) {
      this._clearSlotBootRetryTimer(i, resetCount);
    }
  }

  _hasPendingBootRetry() {
    for (let i = 0; i < this._maxWorkers; i++) {
      if (this._slotBootRetryPending[i]) return true;
    }
    return false;
  }

  _scheduleBootRetry(index, { forWarm = false } = {}) {
    const canRunWhileClosed = Boolean(this._terminateState && this._terminateState.graceful);
    if ((this._closed && !canRunWhileClosed) || this._circuitOpen) return false;
    if (this._slotBootRetryTimers[index] !== null) {
      this._clearTimeout(this._slotBootRetryTimers[index]);
      this._slotBootRetryTimers[index] = null;
    }
    this._slotBootRetries[index]++;
    this._counters.bootRetryCount++;
    this._slotBootRetryPending[index] = true;
    if (forWarm) this._slotBootRetryForWarm[index] = true;
    const delay = this._computeBootRetryDelayMs(this._slotBootRetries[index]);
    this._slotBootRetryTimers[index] = this._setTimeout(() => {
      this._slotBootRetryTimers[index] = null;
      this._slotBootRetryPending[index] = false;
      const graceful = Boolean(this._terminateState && this._terminateState.graceful);
      if (this._circuitOpen || (this._closed && !graceful)) {
        this._slotBootRetryForWarm[index] = false;
        return;
      }
      const warmTarget = this._slotBootRetryForWarm[index];
      this._slotBootRetryForWarm[index] = false;
      if (this._queuedCount() > 0 || this._blocked.length > 0) {
        this._pump();
      } else if (warmTarget && !this._closed && this._slots.size < this._maxWorkers) {
        const slot = this._spawn({ preferredIndex: index, forWarm: true });
        if (slot) {
          this._armIdle(slot);
        } else if (!this._lastSpawnDeferredByBootRetry) {
          this._checkBreaker();
        }
      }
      if (!this._closed) this._ensureMinWorkers();
      this._checkIdle();
    }, delay);
    return true;
  }

  _clearAllBackoffTimers() {
    for (let i = 0; i < this._maxWorkers; i++) {
      this._clearSlotBackoffTimer(i);
    }
  }

  _isSlotInBackoff(index, now = this._now()) {
    const until = this._slotBackoffUntil[index];
    if (until <= 0) return false;
    if (now >= until) {
      this._clearSlotBackoffTimer(index);
      return false;
    }
    return true;
  }

  _scheduleSlotBackoff(index) {
    const canRunWhileClosed = Boolean(this._terminateState && this._terminateState.graceful);
    if ((this._closed && !canRunWhileClosed) || this._circuitOpen) return;
    if (this._slotBackoffTimers[index] !== null) return;
    const remaining = this._slotBackoffUntil[index] - this._now();
    if (remaining <= 0) {
      this._slotBackoffUntil[index] = 0;
      return;
    }
    this._slotBackoffTimers[index] = this._setTimeout(() => {
      this._slotBackoffTimers[index] = null;
      this._slotBackoffUntil[index] = 0;
      const graceful = Boolean(this._terminateState && this._terminateState.graceful);
      if (this._circuitOpen || (this._closed && !graceful)) return;
      this._pump();
      if (!this._closed) this._ensureMinWorkers();
      this._checkIdle();
    }, remaining);
  }

  _scheduleAnyBackoffSlots() {
    const occupied = new Set();
    for (const slot of this._slots) occupied.add(slot.index);
    const now = this._now();
    let anyInBackoff = false;
    for (let i = 0; i < this._maxWorkers; i++) {
      if (occupied.has(i) || this._slotBootRetryPending[i]) continue;
      if (
        this._slotNeedsRespawn[i] &&
        this._slotRespawns[i] < this._maxRespawnsPerSlot &&
        this._isSlotInBackoff(i, now)
      ) {
        this._scheduleSlotBackoff(i);
        anyInBackoff = true;
      }
    }
    return anyInBackoff;
  }

  _pickFreeSlotIndex(preferredIndex = -1) {
    const occupied = new Set();
    for (const slot of this._slots) occupied.add(slot.index);
    const now = this._now();
    if (
      preferredIndex >= 0 &&
      preferredIndex < this._maxWorkers &&
      !occupied.has(preferredIndex) &&
      !this._slotBootRetryPending[preferredIndex]
    ) {
      if (!this._slotNeedsRespawn[preferredIndex]) {
        return { index: preferredIndex, isRespawn: false };
      }
      if (
        this._slotRespawns[preferredIndex] < this._maxRespawnsPerSlot &&
        !this._isSlotInBackoff(preferredIndex, now)
      ) {
        return { index: preferredIndex, isRespawn: true };
      }
    }
    // Prioritize a slot resuming from a completed boot-retry backoff timer.
    for (let i = 0; i < this._maxWorkers; i++) {
      if (occupied.has(i) || this._slotBootRetryPending[i]) continue;
      if (this._slotBootRetries[i] > 0) {
        const isRespawn = this._slotNeedsRespawn[i];
        if (!isRespawn || (this._slotRespawns[i] < this._maxRespawnsPerSlot && !this._isSlotInBackoff(i, now))) {
          return { index: i, isRespawn };
        }
      }
    }
    let freshCandidate = null;
    for (let i = 0; i < this._maxWorkers; i++) {
      if (occupied.has(i) || this._slotBootRetryPending[i]) continue;
      if (this._slotNeedsRespawn[i]) {
        if (this._slotRespawns[i] < this._maxRespawnsPerSlot) {
          if (!this._isSlotInBackoff(i, now)) {
            return { index: i, isRespawn: true };
          }
        }
      } else if (freshCandidate === null) {
        freshCandidate = { index: i, isRespawn: false };
      }
    }
    return freshCandidate;
  }

  _spawn({ preferredIndex = -1, forWarm = false } = {}) {
    this._lastSpawnDeferredByBootRetry = false;
    const pick = this._pickFreeSlotIndex(preferredIndex);
    if (!pick) return null;
    const { index, isRespawn } = pick;
    let worker;
    try {
      worker = this._factory();
    } catch (err) {
      this._counters.spawned++;
      const crashErr = new WorkerCrashError('spawn failed: ' + (err && err.message), err);
      const canRunWhileClosed = Boolean(this._terminateState && this._terminateState.graceful);
      if (
        this._maxBootRetries > 0 &&
        (!this._closed || canRunWhileClosed) &&
        !this._circuitOpen &&
        isTransientBootError(err)
      ) {
        if (this._slotBootRetries[index] < this._maxBootRetries) {
          this._scheduleBootRetry(index, { forWarm });
          this._lastSpawnDeferredByBootRetry = true;
          return null;
        }
        this._counters.bootRetryExhaustedCount++;
        this._slotBootRetries[index] = 0;
      } else if (this._slotBootRetries[index] > 0) {
        this._slotBootRetries[index] = 0;
      }
      if (isRespawn) {
        this._slotRespawns[index]++;
      }
      this._recordCrash(crashErr);
      if (this._onWorkerCrashCallback) {
        try {
          this._onWorkerCrashCallback({
            slotIndex: index,
            jobId: null,
            jobType: null,
            reason: crashErr.reason,
            crashed: this._counters.crashed,
            circuitOpen: this._circuitOpen,
            timestampMs: this._now(),
            error: crashErr,
          });
        } catch (_) {
          // Callback errors must never disrupt pool recovery.
        }
      }
      if (isRespawn && this._slotRespawns[index] < this._maxRespawnsPerSlot) {
        this._slotCrashStreak[index]++;
        const delay = this._computeRespawnBackoffMs(this._slotCrashStreak[index]);
        if (delay > 0) {
          this._slotBackoffUntil[index] = this._now() + delay;
        }
      }
      return null;
    }
    this._counters.spawned++;
    if (isRespawn) {
      this._slotRespawns[index]++;
      this._slotNeedsRespawn[index] = false;
      this._counters.respawnedWorkers++;
    }
    const bootRetryAttempt = this._slotBootRetries[index];
    const spawnedAt = this._now();
    const slot = {
      index,
      worker,
      state: 'idle',
      job: null,
      cancelId: 0,
      dead: false,
      idleTimer: null,
      graceTimer: null,
      jobsHandled: 0,
      lastIdleSeq: 0,
      booted: false,
      bootRetryAttempt,
      forWarm,
      spawnedAt,
      lastActivityAt: spawnedAt,
      lastPingSentAt: 0,
      pendingPingId: null,
      consecutiveMissedPings: 0,
      _detachListeners: [],
    };
    this._slots.add(slot);
    slot._detachListeners.push(
      listen(worker, 'message', (ev) => this._onMessage(slot, ev && ev.data)),
      listen(worker, 'error', (ev) =>
        this._onCrash(slot, new WorkerCrashError(describeErrorEvent(ev), ev && ev.error), ev)
      ),
      listen(worker, 'messageerror', () =>
        this._onCrash(slot, new WorkerCrashError('messageerror: message could not be deserialised'))
      ),
      listen(worker, 'exit', (ev) =>
        this._onCrash(slot, new WorkerCrashError(`worker exited unexpectedly (code ${ev && ev.code})`))
      )
    );
    return slot;
  }

  _ensureMinWorkers() {
    if (this._closed || this._circuitOpen) return;
    const target = this._effectiveMinWorkers();
    while (this._slots.size < target) {
      const slot = this._spawn({ forWarm: true });
      if (!slot) {
        if (this._lastSpawnDeferredByBootRetry) break;
        this._checkBreaker();
        if (!this._circuitOpen) {
          this._scheduleAnyBackoffSlots();
        }
        break;
      }
      this._armIdle(slot);
    }
  }

  /** Pick the most recently used idle warm worker to maximise locality and allow surplus workers to shrink. */
  _idleSlot() {
    let best = null;
    for (const slot of this._slots) {
      if (slot.state !== 'idle' || slot.dead) continue;
      if (!best || slot.lastIdleSeq > best.lastIdleSeq) {
        best = slot;
      }
    }
    return best;
  }

  /** Dispatch queued jobs only onto already-idle healthy workers without spawning fresh slots. */
  _pumpIdleOnly() {
    for (;;) {
      if (this._circuitOpen) return;
      this._admitBlocked();
      if (this._queuedCount() === 0) return;
      const slot = this._idleSlot();
      if (!slot) return;
      const job = this._nextQueued();
      this._dispatch(slot, job);
    }
  }

  /** Move jobs onto workers while capacity and breaker allow. */
  _pump() {
    for (;;) {
      if (this._circuitOpen) return;
      this._admitBlocked();
      if (this._queuedCount() === 0) return;
      let slot = this._idleSlot();
      if (!slot) {
        if (this._slots.size >= this._maxWorkers) return;
        if (!this._pickFreeSlotIndex()) {
          if (this._scheduleAnyBackoffSlots() || this._hasPendingBootRetry()) {
            return;
          }
          if (this._slots.size === 0) {
            const err = new WorkerCrashError(
              `Worker respawn limit exceeded (${this._maxRespawnsPerSlot} per slot)`,
              this._lastCrash || undefined
            );
            const pending = [...this._drainQueued(), ...this._blocked];
            this._blocked = [];
            for (const job of pending) this._settle(job, 'fail', err);
          }
          return;
        }
        slot = this._spawn();
        if (!slot) {
          if (this._circuitOpen) return;
          if (this._lastSpawnDeferredByBootRetry) return;
          // Spawn failed but the breaker is still closed: fail one job rather than spin.
          this._settle(this._nextQueued(), 'fail', this._lastCrash);
          this._checkBreaker();
          continue;
        }
      }
      const job = this._nextQueued();
      this._dispatch(slot, job);
    }
  }

  _admitBlocked() {
    while (this._blocked.length > 0 && this._queuedCount() < this._maxQueue) {
      const job = this._blocked.shift();
      job.state = 'queued';
      this._enqueue(job);
    }
  }

  _dispatch(slot, job) {
    if (slot.idleTimer !== null) {
      this._clearTimeout(slot.idleTimer);
      slot.idleTimer = null;
    }
    job.state = 'running';
    job.slot = slot;
    job.startedAt = this._now();
    this._running++;
    this._waitTotal += Math.max(0, job.startedAt - job.submittedAt);
    this._waitCount++;
    slot.job = job;
    slot.state = 'busy';
    if (job.timeoutMs > 0 && Number.isFinite(job.timeoutMs)) {
      job.timeoutTimer = this._setTimeout(() => this._onTimeout(job), job.timeoutMs);
    }
    try {
      slot.worker.postMessage({ id: job.id, type: job.type, payload: job.payload }, job.transfer);
      if (slot.jobsHandled > 0) this._counters.warmReuses++;
      slot.jobsHandled++;
    } catch (err) {
      // Job-level failure (e.g. DataCloneError): the worker itself is fine.
      slot.job = null;
      slot.state = 'idle';
      slot.lastIdleSeq = ++this._idleSeq;
      this._settle(job, 'fail', err);
      if (this._queuedCount() === 0 && this._blocked.length === 0) {
        this._armIdle(slot);
      }
      this._checkIdle();
    }
  }

  _armIdle(slot) {
    if (
      this._idleTimeoutMs > 0 &&
      slot.idleTimer === null &&
      !slot.dead &&
      slot.state === 'idle'
    ) {
      slot.idleTimer = this._setTimeout(() => {
        slot.idleTimer = null;
        if (!slot.dead && slot.state === 'idle') {
          let idleCount = 0;
          for (const s of this._slots) {
            if (!s.dead && s.state === 'idle') idleCount++;
          }
          if (this._slots.size > this._minWorkers && idleCount > this._minWarmWorkers) {
            this._counters.trimmedWorkers++;
            this._destroySlot(slot);
          }
        }
      }, this._idleTimeoutMs);
    }
  }

  _releaseSlot(slot) {
    slot.job = null;
    slot.state = 'idle';
    slot.lastIdleSeq = ++this._idleSeq;
    this._pump();
    if (!slot.dead && slot.state === 'idle') {
      this._armIdle(slot);
    }
    this._checkIdle();
  }

  _markSlotBooted(slot) {
    if (slot.booted) return;
    slot.booted = true;
    if (slot.bootRetryAttempt > 0) {
      this._counters.bootRetrySuccessCount++;
      this._slotBootRetries[slot.index] = 0;
      slot.bootRetryAttempt = 0;
    }
  }

  _onMessage(slot, data) {
    if (slot.dead || data === null || typeof data !== 'object') return;
    if (data.__watchdogPong === true) {
      this._counters.watchdogPongsReceived++;
      slot.lastActivityAt = this._now();
      this._markSlotBooted(slot);
      if (slot.pendingPingId !== null && (data.pingId === slot.pendingPingId || data.pingId === undefined)) {
        slot.pendingPingId = null;
        slot.consecutiveMissedPings = 0;
      }
      return;
    }
    if (data.ready === true) {
      slot.lastActivityAt = this._now();
      slot.consecutiveMissedPings = 0;
      this._markSlotBooted(slot);
      return;
    }
    const id = data.id;
    if (slot.state === 'cancelling') {
      if (id === slot.cancelId && (data.cancelled === true || typeof data.ok === 'boolean')) {
        slot.lastActivityAt = this._now();
        slot.pendingPingId = null;
        slot.consecutiveMissedPings = 0;
        this._markSlotBooted(slot);
        if (slot.graceTimer !== null) {
          this._clearTimeout(slot.graceTimer);
          slot.graceTimer = null;
        }
        this._releaseSlot(slot);
      }
      return;
    }
    const job = slot.job;
    if (!job || job.settled || job.id !== id) return;
    if (data.ok === true) {
      slot.lastActivityAt = this._now();
      slot.pendingPingId = null;
      slot.consecutiveMissedPings = 0;
      this._markSlotBooted(slot);
      this._slotCrashStreak[slot.index] = 0;
      this._settle(job, 'ok', data.result);
      this._releaseSlot(slot);
    } else if (data.ok === false) {
      slot.lastActivityAt = this._now();
      slot.pendingPingId = null;
      slot.consecutiveMissedPings = 0;
      this._markSlotBooted(slot);
      this._settle(job, 'fail', deserializeError(data.error));
      this._releaseSlot(slot);
    } else if (data.cancelled === true) {
      slot.lastActivityAt = this._now();
      slot.pendingPingId = null;
      slot.consecutiveMissedPings = 0;
      this._markSlotBooted(slot);
      this._settle(job, 'cancel', new JobAbortedError('cancelled by worker'));
      this._releaseSlot(slot);
    } else if ('progress' in data) {
      slot.lastActivityAt = this._now();
      slot.pendingPingId = null;
      slot.consecutiveMissedPings = 0;
      this._markSlotBooted(slot);
      if (job.onProgress) {
        try {
          job.onProgress(data.progress);
        } catch (_) {
          // A throwing progress callback must not break the pool.
        }
      }
    }
  }

  _recordCrash(error) {
    const t = this._now();
    this._counters.crashed++;
    this._lastCrash = error;
    if (this._breakerMax > 0) {
      this._crashTimes.push(t);
      const cutoff = t - this._breakerWindow;
      this._crashTimes = this._crashTimes.filter((c) => (this._breakerWindow === 0 ? c === t : c > cutoff));
    }
  }

  _onCrash(slot, error, rawEvent = null) {
    if (slot.dead) return;
    const job = slot.job;
    const slotIndex = slot.index;
    const canRunWhileClosed = Boolean(this._terminateState && this._terminateState.graceful);

    if (
      !slot.booted &&
      this._maxBootRetries > 0 &&
      isTransientBootError(error, rawEvent)
    ) {
      if (
        (!this._closed || canRunWhileClosed) &&
        !this._circuitOpen &&
        this._slotBootRetries[slotIndex] < this._maxBootRetries
      ) {
        const wasForWarm = slot.forWarm || (job === null && this._slots.size <= this._effectiveMinWorkers());
        this._destroySlot(slot);
        if (job && !job.settled) {
          job.slot = null;
          if (job.transfer.length === 0) {
            this._running--;
            job.state = 'queued';
            if (job.timeoutTimer !== null) {
              this._clearTimeout(job.timeoutTimer);
              job.timeoutTimer = null;
            }
            this._enqueue(job);
          } else {
            this._settle(job, 'fail', error);
          }
        }
        this._scheduleBootRetry(slotIndex, { forWarm: wasForWarm });
        this._pumpIdleOnly();
        this._recordPeakQueueDepth();
        this._checkIdle();
        return;
      }
      if (this._slotBootRetries[slotIndex] >= this._maxBootRetries) {
        this._counters.bootRetryExhaustedCount++;
        this._slotBootRetries[slotIndex] = 0;
      }
    } else if (!slot.booted && this._slotBootRetries[slotIndex] > 0) {
      this._slotBootRetries[slotIndex] = 0;
    }

    this._slotNeedsRespawn[slotIndex] = true;
    this._destroySlot(slot);
    this._recordCrash(error);
    this._slotCrashStreak[slotIndex]++;
    if (this._slotRespawns[slotIndex] < this._maxRespawnsPerSlot) {
      const delay = this._computeRespawnBackoffMs(this._slotCrashStreak[slotIndex]);
      if (delay > 0) {
        this._slotBackoffUntil[slotIndex] = this._now() + delay;
      }
    }
    if (job && !job.settled) {
      job.slot = null;
      if (job.crashRetries < this._retries && job.transfer.length === 0 && !this._closed) {
        // Requeue at the front of its level; the failed attempt's timers are dropped.
        this._running--;
        job.state = 'queued';
        if (job.timeoutTimer !== null) {
          this._clearTimeout(job.timeoutTimer);
          job.timeoutTimer = null;
        }
        job.crashRetries++;
        this._counters.retried++;
        this._enqueue(job);
      } else {
        this._settle(job, 'fail', error);
      }
    }
    this._checkBreaker();
    if (this._onWorkerCrashCallback) {
      try {
        this._onWorkerCrashCallback({
          slotIndex,
          jobId: job ? job.id : null,
          jobType: job ? job.type : null,
          reason: error.reason || error.message,
          crashed: this._counters.crashed,
          circuitOpen: this._circuitOpen,
          timestampMs: this._now(),
          error,
        });
      } catch (_) {
        // Callback errors must never disrupt pool recovery.
      }
    }
    this._pump();
    this._recordPeakQueueDepth();
    this._ensureMinWorkers();
    this._checkIdle();
  }

  _checkBreaker() {
    if (this._circuitOpen || this._breakerMax <= 0) return;
    if (this._crashTimes.length >= this._breakerMax) {
      this._circuitOpen = true;
      this._counters.circuitBreakerTrips++;
      this._clearAllBackoffTimers();
      this._clearAllBootRetryTimers(true);
      const pending = [...this._drainQueued(), ...this._blocked];
      this._blocked = [];
      for (const job of pending) this._settle(job, 'fail', this._circuitError());
    }
  }

  _checkIdle() {
    if (!this._isIdle()) return;
    if (this._slots.size >= this._effectiveMinWorkers()) {
      for (let i = 0; i < this._maxWorkers; i++) {
        if (this._slotBackoffTimers[i] !== null) {
          this._clearTimeout(this._slotBackoffTimers[i]);
          this._slotBackoffTimers[i] = null;
        }
        if (this._slotBootRetryTimers[i] !== null && !this._slotBootRetryForWarm[i]) {
          this._clearSlotBootRetryTimer(i, true);
        }
      }
    }
    const waiters = this._drainWaiters;
    this._drainWaiters = [];
    for (const resolve of waiters) resolve();
    const st = this._terminateState;
    if (st && st.graceful) this._finishImmediate();
  }

  _finishImmediate() {
    const st = this._terminateState;
    this._stopWatchdog();
    this._clearAllBackoffTimers();
    this._clearAllBootRetryTimers(true);
    const pending = [...this._drainQueued(), ...this._blocked];
    this._blocked = [];
    const running = [];
    for (const slot of this._slots) if (slot.job) running.push(slot.job);
    for (const job of pending) this._settle(job, 'cancel', new PoolTerminatedError());
    for (const job of running) this._settle(job, 'cancel', new PoolTerminatedError());
    for (const slot of [...this._slots]) this._destroySlot(slot);
    const waiters = this._drainWaiters;
    this._drainWaiters = [];
    for (const resolve of waiters) resolve();
    if (st) st.resolve();
  }
}

/** Result wrapper carrying a transfer list; see `withTransfer`. */
export class TransferableResult {
  constructor(result, transfer) {
    this.result = result;
    this.transfer = transfer;
  }
}

/**
 * Return a handler result together with buffers to transfer back to the pool.
 * @param {*} result
 * @param {Transferable[]} transfer
 */
export function withTransfer(result, transfer) {
  return new TransferableResult(result, transfer);
}

function unwrapResult(value) {
  if (value instanceof TransferableResult) return { result: value.result, transfer: value.transfer };
  if (
    value !== null &&
    typeof value === 'object' &&
    Object.getPrototypeOf(value) === Object.prototype &&
    Array.isArray(value.transfer) &&
    Object.keys(value).length === 2 &&
    'result' in value
  ) {
    return { result: value.result, transfer: value.transfer };
  }
  return { result: value, transfer: [] };
}

/**
 * Worker-side half of the protocol. Call once inside a worker script.
 *
 * Handlers receive `(payload, ctx)` where `ctx` is
 * `{ id, type, signal, progress(value), transfer(...buffers) }`:
 *  - `ctx.signal` aborts when the pool sends a cancel request (cooperative cancel);
 *  - `ctx.progress(value)` posts a progress message;
 *  - `ctx.transfer(...)` adds buffers to the result's transfer list; alternatively
 *    return `withTransfer(result, list)` or a plain `{ result, transfer }`.
 * A cancelled job is acknowledged with `{ id, cancelled: true }` once its handler
 * settles; its result is discarded.
 *
 * @param {Record<string, (payload: *, ctx: object) => *>} handlers
 * @param {{ scope?: { addEventListener: Function, postMessage: Function } }} [options]
 * @returns {() => void} Function that detaches the message listener.
 */
export function serveWorkerPool(handlers, options = {}) {
  const scope = options.scope || globalThis;
  const active = new Map();

  function post(message, transfer) {
    try {
      scope.postMessage(message, transfer && transfer.length ? transfer : []);
    } catch (err) {
      // e.g. DataCloneError on the result: report it as a job error instead.
      if (message && message.id !== undefined) {
        try {
          scope.postMessage({ id: message.id, ok: false, error: serializeError(err) }, []);
        } catch (_) {
          // Nothing more can be done.
        }
      }
    }
  }

  async function run(id, type, payload) {
    const controller = new AbortController();
    const extra = [];
    active.set(id, controller);
    const ctx = {
      id,
      type,
      signal: controller.signal,
      progress(value) {
        if (!controller.signal.aborted) post({ id, progress: value });
      },
      transfer(...buffers) {
        extra.push(...buffers.flat());
      },
    };
    let outcome;
    try {
      const handler = Object.prototype.hasOwnProperty.call(handlers, type) ? handlers[type] : null;
      if (typeof handler !== 'function') throw new TypeError(`Unknown job type: ${type}`);
      const value = unwrapResult(await handler(payload, ctx));
      outcome = { ok: true, result: value.result, transfer: [...extra, ...value.transfer] };
    } catch (err) {
      outcome = { ok: false, error: serializeError(err) };
    }
    active.delete(id);
    if (controller.signal.aborted) {
      post({ id, cancelled: true });
    } else if (outcome.ok) {
      post({ id, ok: true, result: outcome.result }, outcome.transfer);
    } else {
      post({ id, ok: false, error: outcome.error });
    }
  }

  function onMessage(ev) {
    const data = ev && ev.data;
    if (data === null || typeof data !== 'object') return;
    if (data.__watchdogPing === true) {
      post({ __watchdogPong: true, pingId: data.pingId });
      return;
    }
    if (data.id === undefined) return;
    if (data.cancel === true) {
      const controller = active.get(data.id);
      if (controller) controller.abort();
      else post({ id: data.id, cancelled: true });
      return;
    }
    if (typeof data.type !== 'string' || active.has(data.id)) return;
    run(data.id, data.type, data.payload);
  }

  scope.addEventListener('message', onMessage);
  return () => {
    if (typeof scope.removeEventListener === 'function') scope.removeEventListener('message', onMessage);
  };
}
