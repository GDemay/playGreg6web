(function (globalScope) {
  'use strict';

  // Priority HTTPFS streaming scheduler with bounded concurrency, AbortSignal
  // cancellation, and exponential retry backoff sitting in front of RangeCache.
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

  function normalizePathForClassification(pathOrRelPath) {
    if (typeof pathOrRelPath !== 'string') {
      return '';
    }
    let s = pathOrRelPath.trim();
    if (!s) {
      return '';
    }
    const qIdx = s.indexOf('?');
    if (qIdx !== -1) {
      s = s.slice(0, qIdx);
    }
    const hIdx = s.indexOf('#');
    if (hIdx !== -1) {
      s = s.slice(0, hIdx);
    }
    return s.replace(/\\/g, '/').toLowerCase();
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
      this.sleep = typeof options.sleep === 'function' ? options.sleep : defaultSleep;
      this.nowFn = typeof options.nowFn === 'function' ? options.nowFn : defaultNow;

      this._queues = PRIORITY_LEVELS.map(() => []);
      this._inFlight = 0;
      this._nextSeq = 1;

      this._stats = {
        completed: 0,
        failed: 0,
        cancelled: 0,
        retries: 0,
        agedPromotions: 0,
        maxQueueWaitMs: 0,
        binkVideoReads: 0,
        shaderPackReads: 0,
        uiFontReads: 0,
      };

      this._byPriority = PRIORITY_LEVELS.map(() => ({
        scheduled: 0,
        inFlight: 0,
        completed: 0,
        failed: 0,
        cancelled: 0,
        retries: 0,
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
      if (isBinkVideoPath(path)) {
        this._stats.binkVideoReads += 1;
      } else if (isShaderPackPath(path)) {
        this._stats.shaderPackReads += 1;
      } else if (isUiFontRpfPath(path)) {
        this._stats.uiFontReads += 1;
      }

      return new Promise((resolve, reject) => {
        const task = {
          seq: this._nextSeq++,
          path,
          offset,
          length,
          priority,
          signal,
          enqueuedAtMs: this._now(),
          resolve,
          reject,
          settled: false,
          onQueuedAbort: null,
        };

        if (signal && typeof signal.addEventListener === 'function') {
          const onQueuedAbort = () => {
            this._removeQueuedTask(task);
            this._cancelTask(task, createAbortError(signal.reason));
          };
          task.onQueuedAbort = onQueuedAbort;
          signal.addEventListener('abort', onQueuedAbort, { once: true });
        }

        this._queues[priority].push(task);
        this._drain();
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
        this._detachQueuedAbortListener(task);
        if (!task.settled) {
          this._cancelTask(task, createAbortError('Prefetch cancelled'));
          count += 1;
        }
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
          this._detachQueuedAbortListener(task);
          if (!task.settled) {
            task.settled = true;
            task.reject(createAbortError(reason));
            cancelledCount += 1;
          }
        }
      }
      this._resetCounters();
      if (this._inFlight === 0) {
        this._nextSeq = 1;
      }
      return cancelledCount;
    }

    reset() {
      return this.clear('Scheduler reset');
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
          queued: levelQueued,
          inFlight: counters.inFlight,
          completed: counters.completed,
          failed: counters.failed,
          cancelled: counters.cancelled,
          retries: counters.retries,
          dispatched: counters.dispatched,
          avgQueueWaitMs,
          maxQueueWaitMs: counters.maxQueueWaitMs,
          agedPromotions: counters.agedPromotions,
          promotions: counters.agedPromotions,
        };
      }

      return {
        queued,
        inFlight: this._inFlight,
        completed: this._stats.completed,
        failed: this._stats.failed,
        cancelled: this._stats.cancelled,
        retries: this._stats.retries,
        agedPromotions: this._stats.agedPromotions,
        promotions: this._stats.agedPromotions,
        maxQueueWaitMs: this._stats.maxQueueWaitMs,
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

    _resetCounters() {
      this._stats.completed = 0;
      this._stats.failed = 0;
      this._stats.cancelled = 0;
      this._stats.retries = 0;
      this._stats.agedPromotions = 0;
      this._stats.maxQueueWaitMs = 0;
      this._stats.binkVideoReads = 0;
      this._stats.shaderPackReads = 0;
      this._stats.uiFontReads = 0;

      for (let i = 0; i < PRIORITY_LEVELS.length; i += 1) {
        const counters = this._byPriority[PRIORITY_LEVELS[i]];
        counters.scheduled = 0;
        counters.completed = 0;
        counters.failed = 0;
        counters.cancelled = 0;
        counters.retries = 0;
        counters.dispatched = 0;
        counters.totalQueueWaitMs = 0;
        counters.maxQueueWaitMs = 0;
        counters.agedPromotions = 0;
      }
    }

    _detachQueuedAbortListener(task) {
      if (
        task.signal &&
        task.onQueuedAbort &&
        typeof task.signal.removeEventListener === 'function'
      ) {
        task.signal.removeEventListener('abort', task.onQueuedAbort);
      }
      task.onQueuedAbort = null;
    }

    _removeQueuedTask(task) {
      const queue = this._queues[task.priority];
      const idx = queue.indexOf(task);
      if (idx !== -1) {
        queue.splice(idx, 1);
      }
      this._detachQueuedAbortListener(task);
    }

    _completeTask(task, value) {
      if (task.settled) return;
      task.settled = true;
      this._stats.completed += 1;
      this._byPriority[task.priority].completed += 1;
      task.resolve(value);
    }

    _failTask(task, err) {
      if (task.settled) return;
      task.settled = true;
      this._stats.failed += 1;
      this._byPriority[task.priority].failed += 1;
      task.reject(err);
    }

    _cancelTask(task, err) {
      if (task.settled) return;
      task.settled = true;
      this._stats.cancelled += 1;
      this._byPriority[task.priority].cancelled += 1;
      task.reject(err);
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
        this._detachQueuedAbortListener(task);
        if (task.settled) {
          continue;
        }
        if (task.signal && task.signal.aborted) {
          this._cancelTask(task, createAbortError(task.signal.reason));
          continue;
        }

        const waitMs = Math.max(0, nowMs - task.enqueuedAtMs);
        const prioStats = this._byPriority[task.priority];
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
        }

        this._inFlight += 1;
        prioStats.inFlight += 1;
        this._runTask(task);
      }
    }

    _awaitWithAbort(promise, signal) {
      if (!signal || typeof signal.addEventListener !== 'function') {
        return promise;
      }
      if (signal.aborted) {
        promise.catch(() => {});
        return Promise.reject(createAbortError(signal.reason));
      }
      return new Promise((resolve, reject) => {
        const onAbort = () => {
          promise.catch(() => {});
          reject(createAbortError(signal.reason));
        };
        signal.addEventListener('abort', onAbort, { once: true });
        promise.then(
          (val) => {
            signal.removeEventListener('abort', onAbort);
            resolve(val);
          },
          (err) => {
            signal.removeEventListener('abort', onAbort);
            reject(err);
          }
        );
      });
    }

    async _runTask(task) {
      const { path, offset, length, priority, signal } = task;
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
          if (signal && signal.aborted) {
            this._cancelTask(task, createAbortError(signal.reason));
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
            const bytes = await this._awaitWithAbort(trackedRead, signal);
            if (signal && signal.aborted) {
              this._cancelTask(task, createAbortError(signal.reason));
              return;
            }
            this._completeTask(task, bytes);
            return;
          } catch (err) {
            if (!readSettled) {
              // Aborted while underlying rangeCache.read() is still running.
              // Reject the caller immediately, but hold the concurrency slot until
              // the underlying read actually settles so maxConcurrency is preserved.
              pendingReadPromise = trackedRead;
              this._cancelTask(task, createAbortError(signal ? signal.reason : err));
              return;
            }

            if ((signal && signal.aborted) || isAbortError(err)) {
              const abortErr =
                signal && signal.aborted ? createAbortError(signal.reason) : err;
              this._cancelTask(task, abortErr);
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
              await this._awaitWithAbort(sleepPromise, signal);
            } catch (sleepErr) {
              if ((signal && signal.aborted) || isAbortError(sleepErr)) {
                const abortErr =
                  signal && signal.aborted ? createAbortError(signal.reason) : sleepErr;
                this._cancelTask(task, abortErr);
                return;
              }
              this._failTask(task, sleepErr);
              return;
            }

            if (signal && signal.aborted) {
              this._cancelTask(task, createAbortError(signal.reason));
              return;
            }
          }
        }
      } finally {
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
    classifyIoPriority,
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
