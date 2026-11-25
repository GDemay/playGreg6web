(function (globalScope) {
  'use strict';

  // Priority HTTPFS streaming scheduler with bounded concurrency, AbortSignal
  // cancellation, priority aging (starvation prevention), in-flight range
  // deduplication, latency telemetry, and exponential retry backoff sitting
  // in front of RangeCache.
  // Compatible with classic Web Workers (importScripts) and Node CommonJS/ESM tests.

  const IO_PRIORITY = Object.freeze({
    CRITICAL: 0,
    HIGH: 1,
    NORMAL: 2,
    PREFETCH: 3,
  });

  const PRIORITY_LEVELS = Object.freeze([
    IO_PRIORITY.CRITICAL,
    IO_PRIORITY.HIGH,
    IO_PRIORITY.NORMAL,
    IO_PRIORITY.PREFETCH,
  ]);

  const PRIORITY_NAMES = Object.freeze({
    [IO_PRIORITY.CRITICAL]: 'CRITICAL',
    [IO_PRIORITY.HIGH]: 'HIGH',
    [IO_PRIORITY.NORMAL]: 'NORMAL',
    [IO_PRIORITY.PREFETCH]: 'PREFETCH',
  });

  const DEFAULT_MAX_CONCURRENCY = 6;
  const DEFAULT_MAX_RETRIES = 2;
  const DEFAULT_BASE_RETRY_DELAY_MS = 50;
  const DEFAULT_AGING_THRESHOLD_MS = 0;
  const DEFAULT_AGING_INTERVAL_MS = 0;
  const DEFAULT_DEDUPE_IN_FLIGHT = true;

  const RETRYABLE_ERROR_CODES = new Set([
    'ECONNRESET',
    'ECONNREFUSED',
    'ETIMEDOUT',
    'EAI_AGAIN',
    'ENOTFOUND',
    'ENETUNREACH',
    'EHOSTUNREACH',
    'EPIPE',
    'ESOCKETTIMEDOUT',
    'UND_ERR_CONNECT_TIMEOUT',
    'UND_ERR_HEADERS_TIMEOUT',
    'UND_ERR_BODY_TIMEOUT',
    'UND_ERR_SOCKET',
    'ERR_NETWORK',
    'NETWORK_ERR',
  ]);

  function defaultSleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function defaultNow() {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
      return performance.now();
    }
    return Date.now();
  }

  function isValidPriority(priority) {
    return (
      priority === IO_PRIORITY.CRITICAL ||
      priority === IO_PRIORITY.HIGH ||
      priority === IO_PRIORITY.NORMAL ||
      priority === IO_PRIORITY.PREFETCH
    );
  }

  const SPECULATIVE_PREFETCH_RE = /(?:^|\/)(?:prefetch|speculative)(?:\/|$)|\.prefetch$/i;
  const UI_FONT_PATH_RE =
    /(?:^|[/._-])(?:ui|hud|fonts?|scaleform|frontend|menu|splash)(?:[/._-]|$)/i;
  const BOOT_CRITICAL_RPF_NAMES = new Set([
    'movies.rpf',
    'shaders.rpf',
    'textures.rpf',
    'levels.rpf',
  ]);

  function normalizePathForClassification(pathOrRelPath) {
    if (typeof pathOrRelPath !== 'string') {
      return '';
    }
    let s = pathOrRelPath.trim();
    if (!s) {
      return '';
    }
    s = s.replace(/\\/g, '/');
    // Strip Win32 extended-length / device prefixes (e.g. `\\?\G:\...`, `\\.\G:\...`)
    // before stripping URL query strings so the `?` in `\\?\` is not mistaken for a query.
    if (/^\/\/[?.]\//.test(s)) {
      s = s.slice(4);
    }
    const qIdx = s.indexOf('?');
    if (qIdx !== -1) {
      s = s.slice(0, qIdx);
    }
    const hIdx = s.indexOf('#');
    if (hIdx !== -1) {
      s = s.slice(0, hIdx);
    }
    s = s.toLowerCase();
    if (/^rpf:\/\//.test(s)) {
      s = s.slice(6);
    } else if (/^[a-z][a-z0-9+.-]*:\/\//.test(s)) {
      const afterScheme = s.indexOf('://') + 3;
      const pathStart = s.indexOf('/', afterScheme);
      s = pathStart === -1 ? '' : s.slice(pathStart);
    }
    return s;
  }

  function isBootCriticalRpfPath(pathOrUrl) {
    const norm = normalizePathForClassification(pathOrUrl);
    if (!norm) {
      return false;
    }
    const parts = norm.split('/');
    const hasNestedRpf = norm.includes('.rpf/');
    for (let i = parts.length - 1; i >= 0; i -= 1) {
      const seg = parts[i];
      if (!seg) {
        continue;
      }
      if (BOOT_CRITICAL_RPF_NAMES.has(seg)) {
        return true;
      }
      if (i === parts.length - 1 && !hasNestedRpf) {
        return false;
      }
    }
    return false;
  }

  function classifyBootArchivePriority(pathOrUrl) {
    const norm = normalizePathForClassification(pathOrUrl);
    if (!norm) {
      return IO_PRIORITY.NORMAL;
    }
    if (SPECULATIVE_PREFETCH_RE.test(norm)) {
      return IO_PRIORITY.PREFETCH;
    }
    if (
      isBootCriticalRpfPath(norm) ||
      norm.endsWith('.wgsl.pack') ||
      norm.endsWith('.pgim')
    ) {
      return IO_PRIORITY.HIGH;
    }
    return IO_PRIORITY.NORMAL;
  }

  function isBinkVideoPath(pathOrRelPath) {
    const norm = normalizePathForClassification(pathOrRelPath);
    return norm.endsWith('.bk2') || norm.endsWith('.bik');
  }

  function isShaderPackPath(pathOrRelPath) {
    const norm = normalizePathForClassification(pathOrRelPath);
    return norm.endsWith('.wgsl.pack');
  }

  function isUiFontRpfPath(pathOrRelPath) {
    const norm = normalizePathForClassification(pathOrRelPath);
    if (!norm) {
      return false;
    }
    if (norm.endsWith('.rpf') || norm.endsWith('.ttpf') || norm.includes('.rpf/')) {
      return UI_FONT_PATH_RE.test(norm);
    }
    return false;
  }

  function isDirectFontOrUiAssetPath(norm) {
    return (
      norm.endsWith('.ttf') ||
      norm.endsWith('.otf') ||
      norm.endsWith('.woff') ||
      norm.endsWith('.woff2') ||
      norm.endsWith('.gfx')
    );
  }

  function classifyIoPriority(pathOrRelPath, explicitPriority) {
    if (isValidPriority(explicitPriority)) {
      return explicitPriority;
    }

    const norm = normalizePathForClassification(pathOrRelPath);
    if (!norm) {
      return IO_PRIORITY.NORMAL;
    }

    if (SPECULATIVE_PREFETCH_RE.test(norm)) {
      return IO_PRIORITY.PREFETCH;
    }

    const lastSlash = norm.lastIndexOf('/');
    const basename = lastSlash === -1 ? norm : norm.slice(lastSlash + 1);

    if (
      norm.endsWith('.wgsl.pack') ||
      norm.endsWith('.reflect.json') ||
      basename === 'boot-options.json' ||
      norm.endsWith('.pgim')
    ) {
      return IO_PRIORITY.CRITICAL;
    }

    if (
      norm.endsWith('.bk2') ||
      norm.endsWith('.bik') ||
      isBootCriticalRpfPath(norm) ||
      isUiFontRpfPath(norm) ||
      isDirectFontOrUiAssetPath(norm)
    ) {
      return IO_PRIORITY.HIGH;
    }

    return IO_PRIORITY.NORMAL;
  }

  function isAbortError(err) {
    return Boolean(
      err &&
      typeof err === 'object' &&
      (err.name === 'AbortError' || err.code === 'ABORT_ERR')
    );
  }

  function createAbortError(reason) {
    if (reason instanceof Error && reason.name === 'AbortError') {
      return reason;
    }
    const message =
      typeof reason === 'string' && reason.length > 0
        ? reason
        : reason instanceof Error && reason.message
          ? reason.message
          : 'The operation was aborted';
    const err = new Error(message);
    err.name = 'AbortError';
    err.code = 'ABORT_ERR';
    if (reason !== undefined && reason !== message) {
      err.cause = reason;
    }
    return err;
  }

  function extractHttpStatus(err) {
    if (!err || typeof err !== 'object') {
      return null;
    }
    const direct =
      typeof err.status === 'number'
        ? err.status
        : typeof err.statusCode === 'number'
          ? err.statusCode
          : null;
    if (direct !== null && Number.isInteger(direct) && direct >= 100 && direct <= 599) {
      return direct;
    }
    const message = typeof err.message === 'string' ? err.message : '';
    if (!message) {
      return null;
    }
    const match = /\b(?:HTTP|status(?:\s+code)?[:\s]+)\s*(\d{3})\b/i.exec(message);
    if (match) {
      const parsed = Number(match[1]);
      if (Number.isInteger(parsed) && parsed >= 100 && parsed <= 599) {
        return parsed;
      }
    }
    return null;
  }

  function isRetryableError(err) {
    if (!err || typeof err !== 'object') {
      return false;
    }
    if (isAbortError(err)) {
      return false;
    }
    if (typeof err.retryable === 'boolean') {
      return err.retryable;
    }

    const status = extractHttpStatus(err);
    if (status !== null) {
      return status === 429 || status === 408 || (status >= 500 && status <= 599);
    }

    if (
      err instanceof RangeError ||
      err instanceof SyntaxError ||
      err instanceof ReferenceError ||
      err instanceof URIError
    ) {
      return false;
    }

    const code = typeof err.code === 'string' ? err.code.toUpperCase() : '';
    if (code && RETRYABLE_ERROR_CODES.has(code)) {
      return true;
    }

    if (err.name === 'NetworkError' || err.name === 'TimeoutError') {
      return true;
    }

    const message = typeof err.message === 'string' ? err.message : '';
    if (
      /\binvalid\b/i.test(message) ||
      /must be a non-empty string/i.test(message) ||
      /no fetch implementation/i.test(message) ||
      /\bnot found\b/i.test(message) ||
      /\bENOENT\b/.test(message)
    ) {
      return false;
    }

    if (err instanceof TypeError) {
      return /fetch|network|load failed|connection|timeout|socket|reset|offline|dns/i.test(
        message
      );
    }

    return /network|fetch|timeout|timed out|econn|socket|temporar|transient|rate limit|too many requests|service unavailable|bad gateway|gateway timeout|overloaded|connection|reset|offline/i.test(
      message
    );
  }

  function validateReadArgs(path, offset, length) {
    if (typeof path !== 'string' || path.length === 0) {
      throw new TypeError('IoPriorityScheduler path must be a non-empty string');
    }
    if (path.includes('\\') || path.includes('\0')) {
      throw new RangeError(`Invalid IoPriorityScheduler path: ${path}`);
    }
    const stripped = path.replace(/^\/+/, '');
    if (!stripped) {
      throw new RangeError(`Invalid IoPriorityScheduler path: ${path}`);
    }
    const parts = stripped.split('/');
    for (let i = 0; i < parts.length; i += 1) {
      if (!parts[i] || parts[i] === '.' || parts[i] === '..') {
        throw new RangeError(`Invalid IoPriorityScheduler path segment in: ${path}`);
      }
    }
    if (!Number.isInteger(offset) || offset < 0) {
      throw new RangeError(`Invalid read offset: ${offset}`);
    }
    if (!Number.isInteger(length) || length < 0) {
      throw new RangeError(`Invalid read length: ${length}`);
    }
  }

  function makeReadKey(path, offset, length) {
    return `${path.replace(/^\/+/, '')}\0${offset}\0${length}`;
  }

  class IoPriorityScheduler {
    constructor(options = {}) {
      if (!options || !options.rangeCache || typeof options.rangeCache.read !== 'function') {
        throw new TypeError(
          'IoPriorityScheduler requires options.rangeCache with a read(path, offset, length) method'
        );
      }

      this.rangeCache = options.rangeCache;
      this.maxConcurrency =
        Number.isInteger(options.maxConcurrency) && options.maxConcurrency > 0
          ? options.maxConcurrency
          : DEFAULT_MAX_CONCURRENCY;
      this.maxRetries =
        Number.isInteger(options.maxRetries) && options.maxRetries >= 0
          ? options.maxRetries
          : DEFAULT_MAX_RETRIES;
      this.baseRetryDelayMs =
        Number.isFinite(options.baseRetryDelayMs) && options.baseRetryDelayMs >= 0
          ? options.baseRetryDelayMs
          : DEFAULT_BASE_RETRY_DELAY_MS;
      this.agingThresholdMs =
        Number.isFinite(options.agingThresholdMs) && options.agingThresholdMs > 0
          ? options.agingThresholdMs
          : DEFAULT_AGING_THRESHOLD_MS;
      this.agingIntervalMs =
        Number.isFinite(options.agingIntervalMs) && options.agingIntervalMs > 0
          ? options.agingIntervalMs
          : DEFAULT_AGING_INTERVAL_MS;
      this.dedupeInFlight =
        options.dedupeInFlight !== undefined
          ? Boolean(options.dedupeInFlight)
          : options.dedupe !== undefined
            ? Boolean(options.dedupe)
            : DEFAULT_DEDUPE_IN_FLIGHT;
      this.sleep = typeof options.sleep === 'function' ? options.sleep : defaultSleep;
      this.nowFn = typeof options.nowFn === 'function' ? options.nowFn : defaultNow;

      this._queues = PRIORITY_LEVELS.map(() => []);
      this._activeByKey = new Map();
      this._inFlight = 0;
      this._nextSeq = 1;

      this._stats = {
        enqueued: 0,
        dispatched: 0,
        completed: 0,
        failed: 0,
        cancelled: 0,
        retries: 0,
        deduped: 0,
        agedPromotions: 0,
        totalQueueWaitMs: 0,
        maxQueueWaitMs: 0,
        peakQueueDepth: 0,
        highPriorityCount: 0,
        preemptedCount: 0,
        binkVideoReads: 0,
        shaderPackReads: 0,
        uiFontReads: 0,
      };

      this._byPriority = PRIORITY_LEVELS.map(() => ({
        scheduled: 0,
        enqueued: 0,
        inFlight: 0,
        completed: 0,
        failed: 0,
        cancelled: 0,
        retries: 0,
        deduped: 0,
        dispatched: 0,
        totalQueueWaitMs: 0,
        maxQueueWaitMs: 0,
        agedPromotions: 0,
      }));
    }

    scheduleRead(path, offset, length, options = {}) {
      const opts = options && typeof options === 'object' ? options : {};
      const hasExplicitPriority = opts.priority !== undefined;
      const signal = opts.signal || null;

      if (hasExplicitPriority && !isValidPriority(opts.priority)) {
        this._stats.failed += 1;
        return Promise.reject(new RangeError(`Invalid IO priority: ${opts.priority}`));
      }

      const priority = hasExplicitPriority
        ? opts.priority
        : opts.prefetch === true
          ? IO_PRIORITY.PREFETCH
          : classifyIoPriority(path);

      try {
        validateReadArgs(path, offset, length);
      } catch (err) {
        this._stats.failed += 1;
        this._byPriority[priority].failed += 1;
        return Promise.reject(err);
      }

      if (signal && signal.aborted) {
        this._stats.cancelled += 1;
        this._byPriority[priority].cancelled += 1;
        return Promise.reject(createAbortError(signal.reason));
      }

      this._byPriority[priority].scheduled += 1;
      if (priority <= IO_PRIORITY.HIGH) {
        this._stats.highPriorityCount += 1;
      }
      if (isBinkVideoPath(path)) {
        this._stats.binkVideoReads += 1;
      } else if (isShaderPackPath(path)) {
        this._stats.shaderPackReads += 1;
      } else if (isUiFontRpfPath(path)) {
        this._stats.uiFontReads += 1;
      }

      const key = makeReadKey(path, offset, length);

      if (this.dedupeInFlight) {
        const existingTask = this._activeByKey.get(key);
        if (existingTask && existingTask.state !== 'settled' && !existingTask.allWaitersCancelled) {
          this._stats.deduped += 1;
          this._byPriority[priority].deduped += 1;

          return new Promise((resolve, reject) => {
            const waiter = {
              priority,
              signal,
              resolve,
              reject,
              settled: false,
              deduped: true,
              onAbort: null,
            };
            this._attachWaiterAbortListener(existingTask, waiter);
            existingTask.waiters.push(waiter);

            // Prevent priority inversion when a higher-priority caller joins a queued read.
            if (existingTask.state === 'queued' && priority < existingTask.priority) {
              this._moveQueuedTask(existingTask, priority);
            }
            this._drain();
          });
        }
      }

      this._stats.enqueued += 1;
      this._byPriority[priority].enqueued += 1;

      return new Promise((resolve, reject) => {
        const task = {
          seq: this._nextSeq++,
          key,
          path,
          offset,
          length,
          priority,
          enqueuedAtMs: this._now(),
          state: 'queued',
          waiters: [],
          allWaitersCancelled: false,
          cancelError: null,
          onAllCancelled: null,
        };

        const waiter = {
          priority,
          signal,
          resolve,
          reject,
          settled: false,
          deduped: false,
          onAbort: null,
        };
        this._attachWaiterAbortListener(task, waiter);
        task.waiters.push(waiter);

        if (this.dedupeInFlight) {
          this._activeByKey.set(key, task);
        }

        this._queues[priority].push(task);
        this._drain();
        this._updatePeakQueueDepth();
      });
    }

    schedulePrefetchList(entries, options = {}) {
      const opts = options && typeof options === 'object' ? options : {};
      const signal = opts.signal || null;

      const rawList = Array.isArray(entries)
        ? entries
        : entries && Array.isArray(entries.entries)
          ? entries.entries
          : [];

      const promises = [];
      for (let i = 0; i < rawList.length; i += 1) {
        const entry = rawList[i];
        if (!entry || typeof entry.path !== 'string' || entry.path.length === 0) {
          continue;
        }
        const offset = Number.isInteger(entry.offset) && entry.offset >= 0 ? entry.offset : 0;
        const length = Number.isInteger(entry.length) && entry.length >= 0 ? entry.length : 0;
        if (length === 0) {
          continue;
        }
        promises.push(
          this.scheduleRead(entry.path, offset, length, {
            priority: IO_PRIORITY.PREFETCH,
            signal,
          })
        );
      }

      return Promise.allSettled(promises).then((outcomes) => {
        let completed = 0;
        let cancelled = 0;
        let failed = 0;
        for (let i = 0; i < outcomes.length; i += 1) {
          const outcome = outcomes[i];
          if (outcome.status === 'fulfilled') {
            completed += 1;
          } else if (isAbortError(outcome.reason)) {
            cancelled += 1;
          } else {
            failed += 1;
          }
        }
        return {
          requested: rawList.length,
          scheduled: promises.length,
          completed,
          cancelled,
          failed,
        };
      });
    }

    cancelPrefetch() {
      const queue = this._queues[IO_PRIORITY.PREFETCH];
      if (queue.length === 0) {
        return 0;
      }
      const queuedTasks = queue.splice(0, queue.length);
      let count = 0;
      for (let i = 0; i < queuedTasks.length; i += 1) {
        const task = queuedTasks[i];
        if (task.state === 'settled') {
          continue;
        }
        count += this._cancelAllWaiters(task, createAbortError('Prefetch cancelled'));
      }
      return count;
    }

    clear(reason = 'Scheduler cleared') {
      let cancelledCount = 0;
      for (let i = 0; i < PRIORITY_LEVELS.length; i += 1) {
        const queue = this._queues[PRIORITY_LEVELS[i]];
        if (queue.length === 0) {
          continue;
        }
        const queuedTasks = queue.splice(0, queue.length);
        for (let j = 0; j < queuedTasks.length; j += 1) {
          const task = queuedTasks[j];
          this._deleteActiveKey(task);
          if (task.state === 'settled') {
            continue;
          }
          task.state = 'settled';
          const waiters = task.waiters.splice(0, task.waiters.length);
          for (let k = 0; k < waiters.length; k += 1) {
            const waiter = waiters[k];
            this._detachWaiterAbortListener(waiter);
            if (!waiter.settled) {
              waiter.settled = true;
              waiter.reject(createAbortError(reason));
              cancelledCount += 1;
            }
          }
        }
      }
      this._activeByKey.clear();
      this._resetCounters();
      if (this._inFlight === 0) {
        this._nextSeq = 1;
      }
      return cancelledCount;
    }

    reset() {
      return this.clear('Scheduler reset');
    }

    resetStats() {
      this._resetCounters();
    }

    getStats() {
      const byPriority = {};
      const avgQueueWaitMsByPriority = {};
      const priorityCounts = {};
      let queued = 0;

      for (let i = 0; i < PRIORITY_LEVELS.length; i += 1) {
        const level = PRIORITY_LEVELS[i];
        const name = PRIORITY_NAMES[level];
        const levelQueued = this._queues[level].length;
        const counters = this._byPriority[level];
        const avgQueueWaitMs =
          counters.dispatched > 0 ? counters.totalQueueWaitMs / counters.dispatched : 0;
        queued += levelQueued;
        avgQueueWaitMsByPriority[name] = avgQueueWaitMs;
        priorityCounts[name] = counters.scheduled;
        byPriority[name] = {
          priority: level,
          scheduled: counters.scheduled,
          enqueued: counters.enqueued,
          queued: levelQueued,
          inFlight: counters.inFlight,
          completed: counters.completed,
          failed: counters.failed,
          cancelled: counters.cancelled,
          retries: counters.retries,
          deduped: counters.deduped,
          dispatched: counters.dispatched,
          avgQueueWaitMs,
          maxQueueWaitMs: counters.maxQueueWaitMs,
          agedPromotions: counters.agedPromotions,
          promotions: counters.agedPromotions,
        };
      }

      const avgQueueWaitMs =
        this._stats.dispatched > 0 ? this._stats.totalQueueWaitMs / this._stats.dispatched : 0;

      return {
        queued,
        inFlight: this._inFlight,
        enqueued: this._stats.enqueued,
        dispatched: this._stats.dispatched,
        completed: this._stats.completed,
        failed: this._stats.failed,
        cancelled: this._stats.cancelled,
        retries: this._stats.retries,
        deduped: this._stats.deduped,
        agedPromotions: this._stats.agedPromotions,
        promotions: this._stats.agedPromotions,
        avgQueueWaitMs,
        maxQueueWaitMs: this._stats.maxQueueWaitMs,
        peakQueueDepth: this._stats.peakQueueDepth,
        avgWaitMs: avgQueueWaitMs,
        maxWaitMs: this._stats.maxQueueWaitMs,
        highPriorityCount: this._stats.highPriorityCount,
        preemptedCount: this._stats.preemptedCount,
        binkVideoReads: this._stats.binkVideoReads,
        shaderPackReads: this._stats.shaderPackReads,
        uiFontReads: this._stats.uiFontReads,
        priorityCounts,
        avgQueueWaitMsByPriority,
        byPriority,
      };
    }

    _now() {
      const value = this.nowFn();
      return Number.isFinite(value) ? value : 0;
    }

    _getQueuedCount() {
      let total = 0;
      for (let i = 0; i < PRIORITY_LEVELS.length; i += 1) {
        total += this._queues[PRIORITY_LEVELS[i]].length;
      }
      return total;
    }

    _updatePeakQueueDepth() {
      const queued = this._getQueuedCount();
      if (queued > this._stats.peakQueueDepth) {
        this._stats.peakQueueDepth = queued;
      }
    }

    _hasEarlierLowerPriorityQueued(task) {
      for (let i = 0; i < PRIORITY_LEVELS.length; i += 1) {
        const level = PRIORITY_LEVELS[i];
        if (level <= task.priority) {
          continue;
        }
        const queue = this._queues[level];
        for (let j = 0; j < queue.length; j += 1) {
          const other = queue[j];
          if (other.state === 'queued' && other.seq < task.seq) {
            return true;
          }
        }
      }
      return false;
    }

    _resetCounters() {
      this._stats.enqueued = 0;
      this._stats.dispatched = 0;
      this._stats.completed = 0;
      this._stats.failed = 0;
      this._stats.cancelled = 0;
      this._stats.retries = 0;
      this._stats.deduped = 0;
      this._stats.agedPromotions = 0;
      this._stats.totalQueueWaitMs = 0;
      this._stats.maxQueueWaitMs = 0;
      this._stats.peakQueueDepth = 0;
      this._stats.highPriorityCount = 0;
      this._stats.preemptedCount = 0;
      this._stats.binkVideoReads = 0;
      this._stats.shaderPackReads = 0;
      this._stats.uiFontReads = 0;

      for (let i = 0; i < PRIORITY_LEVELS.length; i += 1) {
        const counters = this._byPriority[PRIORITY_LEVELS[i]];
        counters.scheduled = 0;
        counters.enqueued = 0;
        counters.completed = 0;
        counters.failed = 0;
        counters.cancelled = 0;
        counters.retries = 0;
        counters.deduped = 0;
        counters.dispatched = 0;
        counters.totalQueueWaitMs = 0;
        counters.maxQueueWaitMs = 0;
        counters.agedPromotions = 0;
      }
    }

    _deleteActiveKey(task) {
      if (task && task.key && this._activeByKey.get(task.key) === task) {
        this._activeByKey.delete(task.key);
      }
    }

    _detachWaiterAbortListener(waiter) {
      if (
        waiter.signal &&
        waiter.onAbort &&
        typeof waiter.signal.removeEventListener === 'function'
      ) {
        waiter.signal.removeEventListener('abort', waiter.onAbort);
      }
      waiter.onAbort = null;
    }

    _attachWaiterAbortListener(task, waiter) {
      if (!waiter.signal || typeof waiter.signal.addEventListener !== 'function') {
        return;
      }
      const onAbort = () => {
        this._abortWaiter(task, waiter, createAbortError(waiter.signal.reason));
      };
      waiter.onAbort = onAbort;
      waiter.signal.addEventListener('abort', onAbort, { once: true });
    }

    _removeQueuedTask(task) {
      const queue = this._queues[task.priority];
      const idx = queue.indexOf(task);
      if (idx !== -1) {
        queue.splice(idx, 1);
      }
    }

    _moveQueuedTask(task, newPriority) {
      if (task.state !== 'queued' || task.priority === newPriority) {
        return;
      }
      this._removeQueuedTask(task);
      const targetQueue = this._queues[newPriority];
      task.priority = newPriority;
      // Preserve (enqueuedAtMs, seq) order so targetQueue[0] stays the oldest task
      // for accurate priority-aging checks.
      let insertIdx = targetQueue.length;
      while (insertIdx > 0) {
        const prev = targetQueue[insertIdx - 1];
        if (
          prev.enqueuedAtMs < task.enqueuedAtMs ||
          (prev.enqueuedAtMs === task.enqueuedAtMs && prev.seq < task.seq)
        ) {
          break;
        }
        insertIdx -= 1;
      }
      targetQueue.splice(insertIdx, 0, task);
    }

    _abortWaiter(task, waiter, abortErr) {
      if (waiter.settled) {
        return;
      }
      waiter.settled = true;
      this._detachWaiterAbortListener(waiter);
      const idx = task.waiters.indexOf(waiter);
      if (idx !== -1) {
        task.waiters.splice(idx, 1);
      }
      this._stats.cancelled += 1;
      this._byPriority[waiter.priority].cancelled += 1;
      waiter.reject(abortErr);

      if (task.state === 'settled') {
        return;
      }

      if (task.waiters.length === 0) {
        if (task.state === 'queued') {
          this._removeQueuedTask(task);
          this._deleteActiveKey(task);
          task.state = 'settled';
        } else if (task.state === 'in_flight') {
          task.allWaitersCancelled = true;
          task.cancelError = abortErr;
          this._deleteActiveKey(task);
          if (task.onAllCancelled) {
            const onCancel = task.onAllCancelled;
            task.onAllCancelled = null;
            onCancel(abortErr);
          }
        }
        return;
      }

      if (task.state === 'queued') {
        let bestPriority = IO_PRIORITY.PREFETCH;
        for (let i = 0; i < task.waiters.length; i += 1) {
          if (task.waiters[i].priority < bestPriority) {
            bestPriority = task.waiters[i].priority;
          }
        }
        if (bestPriority !== task.priority) {
          this._moveQueuedTask(task, bestPriority);
        }
      }
    }

    _completeTask(task, value) {
      if (task.state === 'settled') return;
      task.state = 'settled';
      this._deleteActiveKey(task);
      const waiters = task.waiters.splice(0, task.waiters.length);
      let resolvedAny = false;
      for (let i = 0; i < waiters.length; i += 1) {
        const waiter = waiters[i];
        this._detachWaiterAbortListener(waiter);
        if (waiter.settled) continue;
        if (waiter.signal && waiter.signal.aborted) {
          waiter.settled = true;
          this._stats.cancelled += 1;
          this._byPriority[waiter.priority].cancelled += 1;
          waiter.reject(createAbortError(waiter.signal.reason));
          continue;
        }
        waiter.settled = true;
        resolvedAny = true;
        waiter.resolve(value);
      }
      if (resolvedAny) {
        this._stats.completed += 1;
        this._byPriority[task.priority].completed += 1;
      }
    }

    _failTask(task, err) {
      if (task.state === 'settled') return;
      task.state = 'settled';
      this._deleteActiveKey(task);
      const waiters = task.waiters.splice(0, task.waiters.length);
      let failedAny = false;
      for (let i = 0; i < waiters.length; i += 1) {
        const waiter = waiters[i];
        this._detachWaiterAbortListener(waiter);
        if (waiter.settled) continue;
        waiter.settled = true;
        failedAny = true;
        waiter.reject(err);
      }
      if (failedAny) {
        this._stats.failed += 1;
        this._byPriority[task.priority].failed += 1;
      }
    }

    _cancelAllWaiters(task, err) {
      if (task.state === 'settled') return 0;
      task.state = 'settled';
      this._deleteActiveKey(task);
      const waiters = task.waiters.splice(0, task.waiters.length);
      let count = 0;
      for (let i = 0; i < waiters.length; i += 1) {
        const waiter = waiters[i];
        this._detachWaiterAbortListener(waiter);
        if (waiter.settled) continue;
        waiter.settled = true;
        this._stats.cancelled += 1;
        this._byPriority[waiter.priority].cancelled += 1;
        waiter.reject(err);
        count += 1;
      }
      return count;
    }

    _dequeueNext(nowMs) {
      let strictLevel = -1;
      for (let i = 0; i < PRIORITY_LEVELS.length; i += 1) {
        const level = PRIORITY_LEVELS[i];
        if (this._queues[level].length > 0) {
          strictLevel = level;
          break;
        }
      }
      if (strictLevel === -1) {
        return null;
      }

      if (this.agingThresholdMs > 0) {
        let bestStarvedLevel = -1;
        let bestStarvedEnqueuedAtMs = Infinity;
        let bestStarvedSeq = Infinity;

        for (let i = 0; i < PRIORITY_LEVELS.length; i += 1) {
          const level = PRIORITY_LEVELS[i];
          const queue = this._queues[level];
          if (queue.length === 0) {
            continue;
          }
          const head = queue[0];
          const waitMs = nowMs - head.enqueuedAtMs;
          if (waitMs >= this.agingThresholdMs) {
            if (
              bestStarvedLevel === -1 ||
              head.enqueuedAtMs < bestStarvedEnqueuedAtMs ||
              (head.enqueuedAtMs === bestStarvedEnqueuedAtMs &&
                (level < bestStarvedLevel ||
                  (level === bestStarvedLevel && head.seq < bestStarvedSeq)))
            ) {
              bestStarvedLevel = level;
              bestStarvedEnqueuedAtMs = head.enqueuedAtMs;
              bestStarvedSeq = head.seq;
            }
          }
        }

        if (bestStarvedLevel !== -1) {
          return {
            task: this._queues[bestStarvedLevel].shift(),
            promoted: bestStarvedLevel > strictLevel,
          };
        }
      }

      if (this.agingIntervalMs > 0) {
        let bestLevel = -1;
        let bestEffectivePriority = Infinity;
        let bestEnqueuedAtMs = Infinity;
        let bestSeq = Infinity;

        for (let i = 0; i < PRIORITY_LEVELS.length; i += 1) {
          const level = PRIORITY_LEVELS[i];
          const queue = this._queues[level];
          if (queue.length === 0) {
            continue;
          }
          const head = queue[0];
          const waitMs = Math.max(0, nowMs - head.enqueuedAtMs);
          const steps = Math.floor(waitMs / this.agingIntervalMs);
          const effectivePriority = Math.max(IO_PRIORITY.CRITICAL, level - steps);

          if (
            bestLevel === -1 ||
            effectivePriority < bestEffectivePriority ||
            (effectivePriority === bestEffectivePriority &&
              (head.enqueuedAtMs < bestEnqueuedAtMs ||
                (head.enqueuedAtMs === bestEnqueuedAtMs &&
                  (level < bestLevel || (level === bestLevel && head.seq < bestSeq)))))
          ) {
            bestLevel = level;
            bestEffectivePriority = effectivePriority;
            bestEnqueuedAtMs = head.enqueuedAtMs;
            bestSeq = head.seq;
          }
        }

        if (bestLevel !== -1) {
          return {
            task: this._queues[bestLevel].shift(),
            promoted: bestLevel > strictLevel,
          };
        }
      }

      return {
        task: this._queues[strictLevel].shift(),
        promoted: false,
      };
    }

    _drain() {
      while (this._inFlight < this.maxConcurrency) {
        const nowMs = this._now();
        const dequeued = this._dequeueNext(nowMs);
        if (!dequeued) {
          return;
        }
        const { task, promoted } = dequeued;
        if (task.state === 'settled') {
          continue;
        }

        // Guard against signals marked aborted without firing an event.
        for (let i = task.waiters.length - 1; i >= 0; i -= 1) {
          const waiter = task.waiters[i];
          if (waiter.signal && waiter.signal.aborted) {
            task.waiters.splice(i, 1);
            this._detachWaiterAbortListener(waiter);
            if (!waiter.settled) {
              waiter.settled = true;
              this._stats.cancelled += 1;
              this._byPriority[waiter.priority].cancelled += 1;
              waiter.reject(createAbortError(waiter.signal.reason));
            }
          }
        }
        if (task.waiters.length === 0) {
          this._deleteActiveKey(task);
          task.state = 'settled';
          continue;
        }

        task.state = 'in_flight';

        const waitMs = Math.max(0, nowMs - task.enqueuedAtMs);
        const prioStats = this._byPriority[task.priority];
        this._stats.dispatched += 1;
        this._stats.totalQueueWaitMs += waitMs;
        prioStats.dispatched += 1;
        prioStats.totalQueueWaitMs += waitMs;
        if (waitMs > prioStats.maxQueueWaitMs) {
          prioStats.maxQueueWaitMs = waitMs;
        }
        if (waitMs > this._stats.maxQueueWaitMs) {
          this._stats.maxQueueWaitMs = waitMs;
        }
        if (promoted) {
          this._stats.agedPromotions += 1;
          prioStats.agedPromotions += 1;
        } else if (this._hasEarlierLowerPriorityQueued(task)) {
          this._stats.preemptedCount += 1;
        }

        this._inFlight += 1;
        prioStats.inFlight += 1;
        this._runTask(task);
      }
    }

    _awaitTaskOrAbort(promise, task) {
      if (task.allWaitersCancelled) {
        promise.catch(() => {});
        return Promise.reject(task.cancelError || createAbortError('The operation was aborted'));
      }
      return new Promise((resolve, reject) => {
        let done = false;
        const onTaskCancel = (err) => {
          if (done) return;
          done = true;
          promise.catch(() => {});
          reject(err);
        };
        task.onAllCancelled = onTaskCancel;
        promise.then(
          (val) => {
            if (done) return;
            done = true;
            if (task.onAllCancelled === onTaskCancel) {
              task.onAllCancelled = null;
            }
            resolve(val);
          },
          (err) => {
            if (done) return;
            done = true;
            if (task.onAllCancelled === onTaskCancel) {
              task.onAllCancelled = null;
            }
            reject(err);
          }
        );
      });
    }

    async _runTask(task) {
      const { path, offset, length, priority } = task;
      const prioStats = this._byPriority[priority];
      let attempt = 0;
      let pendingReadPromise = null;

      const releaseSlot = () => {
        this._inFlight -= 1;
        prioStats.inFlight -= 1;
        this._drain();
      };

      try {
        while (true) {
          if (task.allWaitersCancelled || task.waiters.length === 0) {
            this._deleteActiveKey(task);
            task.state = 'settled';
            return;
          }

          if (attempt > 0) {
            this._stats.retries += 1;
            prioStats.retries += 1;
          }

          let readPromise;
          try {
            readPromise = Promise.resolve(this.rangeCache.read(path, offset, length));
          } catch (syncErr) {
            readPromise = Promise.reject(syncErr);
          }

          let readSettled = false;
          const trackedRead = readPromise.then(
            (val) => {
              readSettled = true;
              return val;
            },
            (err) => {
              readSettled = true;
              throw err;
            }
          );

          try {
            const bytes = await this._awaitTaskOrAbort(trackedRead, task);
            if (task.allWaitersCancelled || task.waiters.length === 0) {
              this._deleteActiveKey(task);
              task.state = 'settled';
              return;
            }
            this._completeTask(task, bytes);
            return;
          } catch (err) {
            if (!readSettled) {
              // All waiters aborted while underlying rangeCache.read() is still running.
              // Waiters were already rejected immediately; hold the concurrency slot until
              // the underlying read actually settles so maxConcurrency is preserved.
              pendingReadPromise = trackedRead;
              this._deleteActiveKey(task);
              task.state = 'settled';
              return;
            }

            if (task.allWaitersCancelled || task.waiters.length === 0) {
              this._deleteActiveKey(task);
              task.state = 'settled';
              return;
            }

            if (isAbortError(err)) {
              this._cancelAllWaiters(task, err);
              return;
            }

            if (attempt >= this.maxRetries || !isRetryableError(err)) {
              this._failTask(task, err);
              return;
            }

            const delayMs = this.baseRetryDelayMs * Math.pow(2, attempt);
            attempt += 1;

            let sleepPromise;
            try {
              sleepPromise = Promise.resolve(this.sleep(delayMs));
            } catch (syncSleepErr) {
              sleepPromise = Promise.reject(syncSleepErr);
            }

            try {
              await this._awaitTaskOrAbort(sleepPromise, task);
            } catch (sleepErr) {
              if (task.allWaitersCancelled || task.waiters.length === 0) {
                this._deleteActiveKey(task);
                task.state = 'settled';
                return;
              }
              if (isAbortError(sleepErr)) {
                this._cancelAllWaiters(task, sleepErr);
                return;
              }
              this._failTask(task, sleepErr);
              return;
            }

            if (task.allWaitersCancelled || task.waiters.length === 0) {
              this._deleteActiveKey(task);
              task.state = 'settled';
              return;
            }
          }
        }
      } finally {
        task.onAllCancelled = null;
        this._deleteActiveKey(task);
        if (pendingReadPromise) {
          pendingReadPromise.then(releaseSlot, releaseSlot);
        } else {
          releaseSlot();
        }
      }
    }
  }

  const exportedApi = {
    IO_PRIORITY,
    IoPriorityScheduler,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_BASE_RETRY_DELAY_MS,
    DEFAULT_AGING_THRESHOLD_MS,
    DEFAULT_AGING_INTERVAL_MS,
    DEFAULT_DEDUPE_IN_FLIGHT,
    makeReadKey,
    classifyIoPriority,
    classifyBootArchivePriority,
    isBootCriticalRpfPath,
    isBinkVideoPath,
    isShaderPackPath,
    isUiFontRpfPath,
    isRetryableError,
    isAbortError,
    createAbortError,
  };

  if (globalScope) {
    globalScope.PlayGta6IoScheduler = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
