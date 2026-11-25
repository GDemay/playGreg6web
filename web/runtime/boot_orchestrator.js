'use strict';

// Browser boot pipeline orchestrator for playgta6web.
// Discovers local boot artifacts via `/api/status`, drives SplashOverlay + PgimPreloader,
// feeds BootTracer milestones, launches Wasm64 runtime or falls back to the smoke demo,
// and dismisses splash on FIRST_PRESENT.
// Zero external dependencies; UMD export for Browser (globalThis.PlayGta6BootOrchestrator)
// and Node.js (module.exports).

(function (root) {
  const DEFAULT_STATUS_URL = '/api/status';
  const DEFAULT_STAGE_TIMEOUT_MS = 0;
  const DEFAULT_PARALLEL_PRELOAD = true;
  const DEFAULT_RELEASE_PRELOAD_BUFFERS = true;
  const MAX_DIAGNOSTIC_TEXT_LENGTH = 512;
  const FORBIDDEN_TASK_NAMES = Object.freeze(new Set(['__proto__', 'constructor', 'prototype']));

  /** Ordered high-level boot pipeline phases. */
  const BOOT_PHASE = Object.freeze({
    IDLE: 'IDLE',
    DISCOVERING: 'DISCOVERING',
    PRELOADING: 'PRELOADING',
    LAUNCHING_RUNTIME: 'LAUNCHING_RUNTIME',
    FALLBACK_SMOKE_DEMO: 'FALLBACK_SMOKE_DEMO',
    PRESENTING: 'PRESENTING',
    ERROR: 'ERROR',
    CANCELLED: 'CANCELLED',
    DESTROYED: 'DESTROYED',
  });

  /** Active boot pipeline phases tracked for duration and stall watchdog detection. */
  const WATCHED_BOOT_PHASES = Object.freeze(
    new Set([
      BOOT_PHASE.DISCOVERING,
      BOOT_PHASE.PRELOADING,
      BOOT_PHASE.LAUNCHING_RUNTIME,
      BOOT_PHASE.FALLBACK_SMOKE_DEMO,
    ])
  );

  /** Explicit boot mode distinguishing real Wasm64 guest boot vs smoke-demo fallback. */
  const BOOT_MODE = Object.freeze({
    PENDING: 'pending',
    WASM_GUEST: 'wasm-guest',
    SMOKE_DEMO: 'smoke-demo',
  });

  function defaultNow() {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
      return performance.now();
    }
    return Date.now();
  }

  function clampNonNegativeFinite(value) {
    return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : 0;
  }

  function normalizeStageTimeoutMs(value) {
    if (value === undefined || value === null) return DEFAULT_STAGE_TIMEOUT_MS;
    if (typeof value !== 'number') {
      throw new TypeError('stageTimeoutMs must be a finite non-negative number or null');
    }
    if (!Number.isFinite(value) || value < 0) {
      throw new RangeError('stageTimeoutMs must be a finite non-negative number');
    }
    return value === 0 ? 0 : value;
  }

  function normalizeParallelPreload(value) {
    if (value === undefined) return DEFAULT_PARALLEL_PRELOAD;
    if (typeof value !== 'boolean') {
      throw new TypeError('parallelPreload must be a boolean');
    }
    return value;
  }

  function normalizeReleasePreloadBuffers(value) {
    if (value === undefined) return DEFAULT_RELEASE_PRELOAD_BUFFERS;
    if (typeof value !== 'boolean') {
      throw new TypeError('releasePreloadBuffers must be a boolean');
    }
    return value;
  }

  function getBinaryBufferInfo(value) {
    if (value instanceof ArrayBuffer) {
      return { backingBuffer: value, byteLength: value.byteLength };
    }
    if (typeof SharedArrayBuffer !== 'undefined' && value instanceof SharedArrayBuffer) {
      return { backingBuffer: value, byteLength: value.byteLength };
    }
    if (ArrayBuffer.isView(value)) {
      return { backingBuffer: value.buffer, byteLength: value.byteLength };
    }
    return null;
  }

  function isWritableOwnProperty(obj, key) {
    if (!obj || typeof obj !== 'object') return false;
    if (!Object.prototype.hasOwnProperty.call(obj, key)) return false;
    const desc = Object.getOwnPropertyDescriptor(obj, key);
    if (!desc) return false;
    if ('get' in desc || 'set' in desc) {
      return typeof desc.set === 'function';
    }
    return desc.writable === true;
  }

  /**
   * Release large binary payload references (`ArrayBuffer`, `SharedArrayBuffer`, or TypedArray views)
   * held on a preloaded boot bundle (or single-asset preload result) while preserving lightweight
   * metadata (`bytesLoaded`, `totalBytes`, `entryCount`, `manifest`, `header`, `timingMs`,
   * `durationMs`, `transport`). Defensive and idempotent: never throws on null, primitive, or
   * frozen inputs.
   *
   * @param {unknown} bundle
   * @returns {{ releasedFields: string[], releasedBytes: number }}
   */
  function releaseBootBundleBytes(bundle) {
    const releasedFields = [];
    let releasedBytes = 0;
    if (!bundle || typeof bundle !== 'object') {
      return { releasedFields, releasedBytes };
    }

    const countedByBuffer = new Map();
    const visitedObjects = new Set();

    const recordBufferBytes = (info) => {
      if (!info || !info.backingBuffer) return;
      const len =
        typeof info.byteLength === 'number' && Number.isFinite(info.byteLength) && info.byteLength > 0
          ? info.byteLength
          : 0;
      const prev = countedByBuffer.get(info.backingBuffer) || 0;
      if (len > prev) {
        releasedBytes += len - prev;
        countedByBuffer.set(info.backingBuffer, len);
      }
    };

    const nullBinaryProp = (obj, key, path) => {
      if (!isWritableOwnProperty(obj, key)) return false;
      let val;
      try {
        val = obj[key];
      } catch {
        return false;
      }
      const info = getBinaryBufferInfo(val);
      if (!info) return false;
      try {
        obj[key] = null;
      } catch {
        return false;
      }
      releasedFields.push(path);
      recordBufferBytes(info);
      return true;
    };

    const releasePayloadContainer = (obj, prefix) => {
      if (!obj || typeof obj !== 'object' || visitedObjects.has(obj)) return;
      visitedObjects.add(obj);
      nullBinaryProp(obj, 'bytes', `${prefix}.bytes`);
      nullBinaryProp(obj, 'buffer', `${prefix}.buffer`);
    };

    const releaseAssetSlot = (parent, key, prefix) => {
      if (!parent || typeof parent !== 'object' || !Object.prototype.hasOwnProperty.call(parent, key)) {
        return;
      }
      let val;
      try {
        val = parent[key];
      } catch {
        return;
      }
      if (!val || typeof val !== 'object') return;
      if (getBinaryBufferInfo(val)) {
        nullBinaryProp(parent, key, prefix);
        return;
      }
      releasePayloadContainer(val, prefix);
    };

    const releaseModuleList = (list, prefix) => {
      if (!Array.isArray(list) || visitedObjects.has(list)) return;
      visitedObjects.add(list);
      for (let i = 0; i < list.length; i += 1) {
        const item = list[i];
        if (!item || typeof item !== 'object') continue;
        if (getBinaryBufferInfo(item)) {
          nullBinaryProp(list, String(i), `${prefix}[${i}]`);
        } else {
          releasePayloadContainer(item, `${prefix}[${i}]`);
        }
      }
    };

    const releaseModuleMap = (map, prefix) => {
      if (!map || typeof map !== 'object' || Array.isArray(map) || visitedObjects.has(map)) return;
      visitedObjects.add(map);
      for (const key of Object.keys(map)) {
        releaseAssetSlot(map, key, `${prefix}.${key}`);
      }
    };

    visitedObjects.add(bundle);

    nullBinaryProp(bundle, 'buffer', 'buffer');
    nullBinaryProp(bundle, 'bytes', 'bytes');
    nullBinaryProp(bundle, 'pgimBytes', 'pgimBytes');

    releaseAssetSlot(bundle, 'shaderPackBytes', 'shaderPackBytes');
    releaseAssetSlot(bundle, 'pgim', 'pgim');
    releaseAssetSlot(bundle, 'shaderPackIndex', 'shaderPackIndex');
    releaseAssetSlot(bundle, 'reflectionManifest', 'reflectionManifest');
    releaseAssetSlot(bundle, 'reflectionSidecar', 'reflectionSidecar');

    if (Object.prototype.hasOwnProperty.call(bundle, 'modules')) {
      releaseModuleList(bundle.modules, 'modules');
    }
    if (Object.prototype.hasOwnProperty.call(bundle, 'modulesByName')) {
      releaseModuleMap(bundle.modulesByName, 'modulesByName');
    }
    if (Object.prototype.hasOwnProperty.call(bundle, 'mainModule')) {
      releaseAssetSlot(bundle, 'mainModule', 'mainModule');
    }
    if (
      Object.prototype.hasOwnProperty.call(bundle, 'program') &&
      bundle.program &&
      typeof bundle.program === 'object' &&
      !visitedObjects.has(bundle.program)
    ) {
      const program = bundle.program;
      visitedObjects.add(program);
      nullBinaryProp(program, 'bytes', 'program.bytes');
      nullBinaryProp(program, 'buffer', 'program.buffer');
      if (Object.prototype.hasOwnProperty.call(program, 'mainModule')) {
        releaseAssetSlot(program, 'mainModule', 'program.mainModule');
      }
      if (Object.prototype.hasOwnProperty.call(program, 'modules')) {
        releaseModuleList(program.modules, 'program.modules');
      }
      if (Object.prototype.hasOwnProperty.call(program, 'modulesByName')) {
        releaseModuleMap(program.modulesByName, 'program.modulesByName');
      }
    }

    if (
      Object.prototype.hasOwnProperty.call(bundle, 'sidecars') &&
      bundle.sidecars &&
      typeof bundle.sidecars === 'object' &&
      !Array.isArray(bundle.sidecars) &&
      !visitedObjects.has(bundle.sidecars)
    ) {
      const sidecars = bundle.sidecars;
      visitedObjects.add(sidecars);
      for (const key of Object.keys(sidecars)) {
        releaseAssetSlot(sidecars, key, `sidecars.${key}`);
      }
    }

    return { releasedFields, releasedBytes };
  }

  function normalizePreloadTasks(value) {
    if (value === undefined || value === null) return [];
    if (!Array.isArray(value)) {
      throw new TypeError('preloadTasks must be an array or null');
    }
    const seen = new Set();
    const out = [];
    for (let i = 0; i < value.length; i += 1) {
      const entry = value[i];
      if (!entry || typeof entry !== 'object') {
        throw new TypeError(`preloadTasks[${i}] must be an object`);
      }
      const name = typeof entry.name === 'string' ? entry.name.trim() : '';
      if (!name || FORBIDDEN_TASK_NAMES.has(name)) {
        throw new TypeError(`preloadTasks[${i}].name must be a valid non-empty string`);
      }
      if (seen.has(name)) {
        throw new TypeError(`preloadTasks contains duplicate task name "${name}"`);
      }
      if (typeof entry.run !== 'function') {
        throw new TypeError(`preloadTasks[${i}].run must be a function`);
      }
      if (entry.optional !== undefined && typeof entry.optional !== 'boolean') {
        throw new TypeError(`preloadTasks[${i}].optional must be a boolean`);
      }
      const asset =
        typeof entry.asset === 'string' && entry.asset.trim().length > 0 ? entry.asset.trim() : name;
      seen.add(name);
      out.push(
        Object.freeze({
          name,
          asset,
          run: entry.run,
          optional: entry.optional !== undefined ? entry.optional : true,
        })
      );
    }
    return out;
  }

  function emptyPreloadStats(parallelEnabled) {
    return {
      parallelEnabled: Boolean(parallelEnabled),
      taskCount: 0,
      completedCount: 0,
      optionalMissingCount: 0,
      failedCount: 0,
      abortedCount: 0,
      maxConcurrency: 0,
      wallTimeMs: 0,
      sequentialTimeMs: 0,
      parallelPreloadSavedMs: 0,
      taskTimingsMs: {},
      tasks: {},
    };
  }

  function clonePreloadStats(stats) {
    const src = stats || emptyPreloadStats(DEFAULT_PARALLEL_PRELOAD);
    const taskTimingsMs = {};
    if (src.taskTimingsMs && typeof src.taskTimingsMs === 'object') {
      for (const key of Object.keys(src.taskTimingsMs)) {
        taskTimingsMs[key] = clampNonNegativeFinite(src.taskTimingsMs[key]);
      }
    }
    const tasks = {};
    if (src.tasks && typeof src.tasks === 'object') {
      for (const key of Object.keys(src.tasks)) {
        const entry = src.tasks[key];
        if (entry && typeof entry === 'object') {
          tasks[key] = {
            status: entry.status || 'completed',
            optional: Boolean(entry.optional),
            durationMs: clampNonNegativeFinite(entry.durationMs),
            error: entry.error ? sanitizeDiagnosticText(entry.error) : null,
          };
        }
      }
    }
    return {
      parallelEnabled: Boolean(src.parallelEnabled),
      taskCount: Number.isInteger(src.taskCount) && src.taskCount >= 0 ? src.taskCount : 0,
      completedCount: Number.isInteger(src.completedCount) && src.completedCount >= 0 ? src.completedCount : 0,
      optionalMissingCount:
        Number.isInteger(src.optionalMissingCount) && src.optionalMissingCount >= 0 ? src.optionalMissingCount : 0,
      failedCount: Number.isInteger(src.failedCount) && src.failedCount >= 0 ? src.failedCount : 0,
      abortedCount: Number.isInteger(src.abortedCount) && src.abortedCount >= 0 ? src.abortedCount : 0,
      maxConcurrency: Number.isInteger(src.maxConcurrency) && src.maxConcurrency >= 0 ? src.maxConcurrency : 0,
      wallTimeMs: clampNonNegativeFinite(src.wallTimeMs),
      sequentialTimeMs: clampNonNegativeFinite(src.sequentialTimeMs),
      parallelPreloadSavedMs: clampNonNegativeFinite(src.parallelPreloadSavedMs),
      taskTimingsMs,
      tasks,
    };
  }

  function inferBytesLoaded(value) {
    if (!value || typeof value !== 'object') return 0;
    if (typeof value.bytesLoaded === 'number' && Number.isFinite(value.bytesLoaded) && value.bytesLoaded > 0) {
      return value.bytesLoaded;
    }
    if (typeof value.totalBytes === 'number' && Number.isFinite(value.totalBytes) && value.totalBytes > 0) {
      return value.totalBytes;
    }
    if (typeof value.byteLength === 'number' && Number.isFinite(value.byteLength) && value.byteLength > 0) {
      return value.byteLength;
    }
    if (
      value.bytes &&
      typeof value.bytes.byteLength === 'number' &&
      Number.isFinite(value.bytes.byteLength) &&
      value.bytes.byteLength > 0
    ) {
      return value.bytes.byteLength;
    }
    return 0;
  }

  function isMissingOptionalError(err) {
    if (!err || typeof err !== 'object') return false;
    if (err.status === 404 || err.status === 410) return true;
    if (err.code === 'NOT_FOUND' || err.code === 'ENOENT' || err.code === 'OPTIONAL_MISSING') return true;
    if (typeof err.message === 'string' && /\b(404|410|not\s+found|missing)\b/i.test(err.message)) {
      return true;
    }
    return false;
  }

  function isValidationError(err) {
    return Boolean(err && typeof err === 'object' && typeof err.code === 'string' && err.code.startsWith('BAD_'));
  }

  function raceAbort(promise, signal) {
    if (!signal) return promise;
    if (signal.aborted) {
      Promise.resolve(promise).catch(() => {});
      return Promise.reject(abortReason(signal));
    }
    return new Promise((resolve, reject) => {
      const onAbort = () => reject(abortReason(signal));
      signal.addEventListener('abort', onAbort, { once: true });
      Promise.resolve(promise).then(
        (value) => {
          signal.removeEventListener('abort', onAbort);
          resolve(value);
        },
        (err) => {
          signal.removeEventListener('abort', onAbort);
          reject(err);
        }
      );
    });
  }

  function sumStageDurations(stageDurationsMs, currentStageElapsedMs) {
    let total = clampNonNegativeFinite(currentStageElapsedMs);
    for (const key of Object.keys(stageDurationsMs)) {
      total += clampNonNegativeFinite(stageDurationsMs[key]);
    }
    return clampNonNegativeFinite(total);
  }

  const FILE_URL_RE = /\bfile:\/\/[^\s"'`<>)\]}]+/gi;
  const URL_USERINFO_RE = /\b((?:https?|wss?|ftp):\/\/)([^\s/@"'`<>]+):([^\s/@"'`<>]*)@/gi;
  const URL_QUERY_RE = /((?:https?|wss?):\/\/[^\s"'`<>?#]+)\?(?!\[REDACTED\])[^\s"'`<>#)\]}]+/gi;
  const AUTH_SCHEME_RE = /\b(Bearer|Basic)\s+(?!\[REDACTED\])[A-Za-z0-9\-._~+/=:]+/gi;
  const SECRET_KV_RE =
    /\b(authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|id[_-]?token|client[_-]?secret|x-api-key|x-auth-token|cookie|set-cookie|password|passwd|secret|token)(\s*[:=]\s*)(?!(?:(?:Bearer|Basic)\s+)?\[REDACTED\])(?:"[^"]*"|'[^']*'|[^\s,;}\])]+)/gi;
  const WIN_DRIVE_PATH_RE = /(?<![A-Za-z0-9_])[A-Za-z]:(?:\\|\/(?!\/))[^\s"'`<>|*?)\]},;]+/g;
  const WIN_UNC_PATH_RE = /(^|[\s"'`(\[={<,:;])\\\\[^\s\\"'`<>|*?)\]},;]+\\[^\s"'`<>|*?)\]},;]+/g;
  const POSIX_ABS_PATH_RE =
    /(^|[\s(\[={<,:;`"'])\/(?:home|Users|root|tmp|var|etc|opt|usr|mnt|media|Volumes|private|srv)\/[^\s"'`<>)\]},;]+/g;

  function sanitizeDiagnosticText(value, maxLen = MAX_DIAGNOSTIC_TEXT_LENGTH) {
    if (value == null) return null;
    let text = String(value);
    text = text.replace(FILE_URL_RE, '[REDACTED_PATH]');
    text = text.replace(URL_USERINFO_RE, '$1[REDACTED]@');
    text = text.replace(URL_QUERY_RE, '$1?[REDACTED]');
    text = text.replace(AUTH_SCHEME_RE, '$1 [REDACTED]');
    text = text.replace(SECRET_KV_RE, '$1$2[REDACTED]');
    text = text.replace(WIN_DRIVE_PATH_RE, '[REDACTED_PATH]');
    text = text.replace(WIN_UNC_PATH_RE, '$1[REDACTED_PATH]');
    text = text.replace(POSIX_ABS_PATH_RE, '$1[REDACTED_PATH]');
    if (text.length > maxLen) {
      return `${text.slice(0, maxLen - 3)}...`;
    }
    return text;
  }

  function defaultFetchImpl() {
    if (typeof globalThis !== 'undefined' && typeof globalThis.fetch === 'function') {
      return globalThis.fetch.bind(globalThis);
    }
    return null;
  }

  function abortReason(signal) {
    if (signal && signal.reason !== undefined) return signal.reason;
    const Err = typeof DOMException === 'function' ? DOMException : Error;
    try {
      return new Err('The operation was aborted.', 'AbortError');
    } catch {
      const err = new Error('The operation was aborted.');
      err.name = 'AbortError';
      return err;
    }
  }

  function isAbortError(err) {
    if (!err || typeof err !== 'object') return false;
    if (err.name === 'AbortError') return true;
    if (typeof DOMException === 'function' && err instanceof DOMException && err.name === 'AbortError') {
      return true;
    }
    return false;
  }

  function throwIfAborted(signal) {
    if (signal && signal.aborted) {
      throw abortReason(signal);
    }
  }

  function toError(err, fallbackMessage = 'Unknown boot orchestrator error') {
    if (err instanceof Error) return err;
    if (typeof err === 'string' && err.length > 0) return new Error(err);
    if (err && typeof err === 'object' && typeof err.message === 'string') return new Error(err.message);
    return new Error(fallbackMessage);
  }

  function nonEmptyString(value) {
    return typeof value === 'string' && value.trim().length > 0 ? value.trim() : null;
  }

  function optionalFn(value, label) {
    if (value == null) return null;
    if (typeof value !== 'function') {
      throw new TypeError(`${label} must be a function or null`);
    }
    return value;
  }

  function optionalObject(value, label) {
    if (value == null) return null;
    if (typeof value !== 'object') {
      throw new TypeError(`${label} must be an object or null`);
    }
    return value;
  }

  function callQuiet(target, method, ...args) {
    if (!target || typeof target[method] !== 'function') return undefined;
    try {
      return target[method](...args);
    } catch {
      return undefined;
    }
  }

  function invokeQuiet(fn, ...args) {
    if (typeof fn !== 'function') return undefined;
    try {
      return fn(...args);
    } catch {
      return undefined;
    }
  }

  async function invokeQuietAsync(fn, ...args) {
    if (typeof fn !== 'function') return undefined;
    try {
      return await fn(...args);
    } catch {
      return undefined;
    }
  }

  function pickUrl(...candidates) {
    for (const candidate of candidates) {
      const url = nonEmptyString(candidate);
      if (url) return url;
    }
    return null;
  }

  function presentFlag(value) {
    if (value === true) return true;
    if (value && typeof value === 'object') {
      if (value.present === true || value.available === true || value.ok === true || value.nonEmpty === true) {
        return true;
      }
      if (nonEmptyString(value.url) || nonEmptyString(value.path) || nonEmptyString(value.href)) {
        return true;
      }
    }
    return typeof value === 'string' && value.trim().length > 0;
  }

  function urlFromArtifact(value) {
    if (typeof value === 'string') return nonEmptyString(value);
    if (value && typeof value === 'object') {
      return pickUrl(value.url, value.path, value.href);
    }
    return null;
  }

  function normalizeFirstPresentSource(value) {
    if (typeof value !== 'string') return null;
    const normalized = value.trim().toLowerCase();
    if (normalized === 'wasm-guest' || normalized === 'wasm_guest') {
      return BOOT_MODE.WASM_GUEST;
    }
    if (normalized === 'smoke-demo' || normalized === 'smoke_demo') {
      return BOOT_MODE.SMOKE_DEMO;
    }
    return null;
  }

  function resolveFirstPresentSource(meta, bootMode) {
    const explicit =
      meta && typeof meta === 'object'
        ? normalizeFirstPresentSource(meta.firstPresentSource) || normalizeFirstPresentSource(meta.source)
        : null;
    if (explicit) return explicit;
    if (bootMode === BOOT_MODE.WASM_GUEST) return BOOT_MODE.WASM_GUEST;
    if (bootMode === BOOT_MODE.SMOKE_DEMO) return BOOT_MODE.SMOKE_DEMO;
    return null;
  }

  /**
   * Returns true ONLY when the supplied orchestrator, state snapshot, assets object, or
   * mode string represents a real Wasm64 guest boot (`bootMode === 'wasm-guest'`).
   *
   * @param {unknown} target
   * @returns {boolean}
   */
  function isRealGuestBoot(target) {
    if (typeof target === 'string') {
      return target === BOOT_MODE.WASM_GUEST;
    }
    if (target && typeof target === 'object') {
      if (typeof target.isRealGuestBoot === 'function') {
        return target.isRealGuestBoot() === true;
      }
      if (typeof target.getState === 'function') {
        const state = target.getState();
        return Boolean(state && state.bootMode === BOOT_MODE.WASM_GUEST);
      }
      return target.bootMode === BOOT_MODE.WASM_GUEST;
    }
    return false;
  }

  /**
   * Normalize `/api/status` JSON into the boot-discovery fields used by this orchestrator.
   * Accepts both the current server payload (`hasWasmBinary`, `shaderPackCount`) and the
   * richer artifact URLs (`splashImage`, `pgimUrl`, `programManifestUrl`, `gameBase`,
   * `shaderPackUrl`, `reflectionManifestUrl`, `runtimeWasm`) that tests and later server
   * revisions expose.
   */
  function discoverBootAssets(status) {
    const src = status && typeof status === 'object' ? status : {};
    const splashImage = pickUrl(
      src.splashImage,
      src.splashUrl,
      src.splashScreenImage,
      urlFromArtifact(src.splash)
    );
    const pgimUrl = pickUrl(src.pgimUrl, src.pgim, urlFromArtifact(src.pgimImage));
    const programManifestUrl = pickUrl(
      src.programManifestUrl,
      urlFromArtifact(src.programManifest)
    );
    const gameBase = pickUrl(src.gameBase, src.gameBaseUrl);
    const shaderPackUrl = pickUrl(src.shaderPackUrl, src.shaderPack, urlFromArtifact(src.wgslPack));
    const shaderPackIndexUrl = pickUrl(
      src.shaderPackIndexUrl,
      src.shaderPackHeaderUrl,
      urlFromArtifact(src.shaderPackIndex),
      urlFromArtifact(src.shaderPackHeader)
    );
    const reflectionManifestUrl = pickUrl(
      src.reflectionManifestUrl,
      src.reflectionUrl,
      urlFromArtifact(src.reflectionManifest)
    );
    const reflectionSidecarUrl = pickUrl(
      src.reflectionSidecarUrl,
      src.stageReflectionSidecarUrl,
      src.sidecarUrl,
      urlFromArtifact(src.reflectionSidecar),
      urlFromArtifact(src.sidecar)
    );

    const runtimeArtifact = src.runtimeWasm !== undefined ? src.runtimeWasm : src.hasWasmBinary;
    const runtimeWasmUrl = urlFromArtifact(src.runtimeWasm) || (src.hasWasmBinary ? '/dist/runtime.wasm' : null);
    const runtimeWasmAvailable = presentFlag(runtimeArtifact) || src.hasWasmBinary === true;
    const canLaunchGuest = runtimeWasmAvailable && Boolean(pgimUrl || programManifestUrl);
    const bootMode = canLaunchGuest ? BOOT_MODE.WASM_GUEST : BOOT_MODE.SMOKE_DEMO;

    return {
      splashImage,
      pgimUrl,
      programManifestUrl,
      gameBase,
      shaderPackUrl,
      shaderPackIndexUrl,
      reflectionManifestUrl,
      reflectionSidecarUrl,
      hasWasmBinary: runtimeWasmAvailable,
      runtimeWasm: runtimeWasmAvailable ? runtimeWasmUrl || true : null,
      runtimeWasmUrl: runtimeWasmAvailable ? runtimeWasmUrl : null,
      runtimeWasmAvailable,
      bootMode,
    };
  }

  function tracerSnapshot(bootTracer) {
    if (!bootTracer || typeof bootTracer.getSummary !== 'function') return null;
    try {
      return bootTracer.getSummary();
    } catch {
      return null;
    }
  }

  function linkSignal(outer) {
    const controller = new AbortController();
    if (!outer) return { controller, unlink: () => {} };
    if (outer.aborted) {
      controller.abort(abortReason(outer));
      return { controller, unlink: () => {} };
    }
    const onAbort = () => controller.abort(abortReason(outer));
    outer.addEventListener('abort', onAbort, { once: true });
    return { controller, unlink: () => outer.removeEventListener('abort', onAbort) };
  }

  /**
   * Coordinates `/api/status` discovery, splash overlay, PGIM preload, runtime launch
   * or smoke-demo fallback, and FIRST_PRESENT splash dismissal.
   */
  class BootOrchestrator {
    /**
     * @param {object} [options]
     * @param {typeof fetch | null} [options.fetchImpl]
     * @param {string} [options.statusUrl='/api/status']
     * @param {object|null} [options.splashOverlay]
     * @param {object|null} [options.preloader]
     * @param {object|null} [options.bootTracer]
     * @param {Function|null} [options.onLaunchRuntime]
     * @param {Function|null} [options.onFallbackSmokeDemo]
     * @param {Function|null} [options.onPhaseChange]
     * @param {Function|null} [options.onFirstPresent]
     * @param {Function|null} [options.onStageStall]
     * @param {(() => number)|null} [options.nowFn]
     * @param {number|null} [options.stageTimeoutMs=0]
     * @param {boolean} [options.parallelPreload=true]
     * @param {boolean} [options.releasePreloadBuffers=true]
     * @param {Array<{name: string, run: Function, optional?: boolean, asset?: string}>|null} [options.preloadTasks=null]
     */
    constructor(options = {}) {
      if (options !== undefined && (options === null || typeof options !== 'object')) {
        throw new TypeError('BootOrchestrator options must be an object');
      }

      const {
        fetchImpl = defaultFetchImpl(),
        statusUrl = DEFAULT_STATUS_URL,
        splashOverlay = null,
        preloader = null,
        bootTracer = null,
        onLaunchRuntime = null,
        onFallbackSmokeDemo = null,
        onPhaseChange = null,
        onFirstPresent = null,
        onStageStall = null,
        nowFn = null,
        stageTimeoutMs = DEFAULT_STAGE_TIMEOUT_MS,
        parallelPreload = DEFAULT_PARALLEL_PRELOAD,
        releasePreloadBuffers = DEFAULT_RELEASE_PRELOAD_BUFFERS,
        preloadTasks = null,
      } = options;

      if (fetchImpl !== null && fetchImpl !== undefined && typeof fetchImpl !== 'function') {
        throw new TypeError('fetchImpl must be a function or null');
      }
      if (typeof statusUrl !== 'string' || statusUrl.length === 0) {
        throw new TypeError('statusUrl must be a non-empty string');
      }
      const resolvedNowFn = optionalFn(nowFn, 'nowFn') || defaultNow;
      const resolvedStageTimeoutMs = normalizeStageTimeoutMs(stageTimeoutMs);
      const resolvedParallelPreload = normalizeParallelPreload(parallelPreload);
      const resolvedReleasePreloadBuffers = normalizeReleasePreloadBuffers(releasePreloadBuffers);
      const resolvedPreloadTasks = normalizePreloadTasks(preloadTasks);

      this._fetchImpl = fetchImpl || null;
      this._statusUrl = statusUrl;
      this._splashOverlay = optionalObject(splashOverlay, 'splashOverlay');
      this._preloader = optionalObject(preloader, 'preloader');
      this._bootTracer = optionalObject(bootTracer, 'bootTracer');
      this._onLaunchRuntime = optionalFn(onLaunchRuntime, 'onLaunchRuntime');
      this._onFallbackSmokeDemo = optionalFn(onFallbackSmokeDemo, 'onFallbackSmokeDemo');
      this._onPhaseChange = optionalFn(onPhaseChange, 'onPhaseChange');
      this._onFirstPresent = optionalFn(onFirstPresent, 'onFirstPresent');
      this._onStageStall = optionalFn(onStageStall, 'onStageStall');
      this._nowFn = resolvedNowFn;
      this._stageTimeoutMs = resolvedStageTimeoutMs;
      this._parallelPreload = resolvedParallelPreload;
      this._releasePreloadBuffers = resolvedReleasePreloadBuffers;
      this._preloadTasks = resolvedPreloadTasks;
      this._lastClockMs = 0;

      this._phase = BOOT_PHASE.IDLE;
      this._bootMode = BOOT_MODE.PENDING;
      this._status = null;
      this._assets = null;
      this._bundle = null;
      this._progress = null;
      this._fallbackReason = null;
      this._error = null;
      this._firstPresent = false;
      this._firstPresentSource = null;
      this._destroyed = false;
      this._aborted = false;
      this._pgimLoadMarked = false;
      this._startPromise = null;
      this._sessionToken = 0;
      this._abortUnlink = null;
      this._controller = new AbortController();

      this._stageDurationsMs = Object.create(null);
      this._stageEnteredAt = this._readNow();
      this._stalled = false;
      this._stalledStage = null;
      this._stalledMs = 0;
      this._stallCount = 0;
      this._preloadStats = emptyPreloadStats(this._parallelPreload);
    }

    _readNow() {
      try {
        const value = this._nowFn();
        if (typeof value === 'number' && Number.isFinite(value)) {
          this._lastClockMs = value;
          return value;
        }
      } catch {
        // Fall back to last known finite clock reading.
      }
      return this._lastClockMs;
    }

    _cloneStageDurations() {
      const out = {};
      for (const phase of Object.keys(this._stageDurationsMs)) {
        out[phase] = clampNonNegativeFinite(this._stageDurationsMs[phase]);
      }
      return out;
    }

    _resetTimingTelemetry(now = this._readNow()) {
      this._stageDurationsMs = Object.create(null);
      this._stageEnteredAt = now;
      this._stalled = false;
      this._stalledStage = null;
      this._stalledMs = 0;
      this._stallCount = 0;
      this._preloadStats = emptyPreloadStats(this._parallelPreload);
    }

    _buildTimingSnapshot(now = this._readNow()) {
      const watchdog = this._evaluateWatchdog(now);
      const stageDurationsMs = this._cloneStageDurations();
      const phaseTimingsMs = this._cloneStageDurations();
      const totalStageDurationMs = sumStageDurations(stageDurationsMs, watchdog.currentStageElapsedMs);
      const totalBootMs = totalStageDurationMs;
      const preloadStats = clonePreloadStats(this._preloadStats);
      const preloadTaskTimingsMs = { ...preloadStats.taskTimingsMs };
      return {
        watchdog,
        stageDurationsMs,
        phaseTimingsMs,
        totalStageDurationMs,
        totalBootMs,
        parallelPreloadSavedMs: preloadStats.parallelPreloadSavedMs,
        preloadTaskTimingsMs,
        preloadStats,
      };
    }

    /**
     * Return structured phase-duration and parallel-preload timing telemetry.
     */
    getTimingStats() {
      const timing = this._buildTimingSnapshot();
      return {
        phase: this._phase,
        bootMode: this._bootMode,
        phaseTimingsMs: timing.phaseTimingsMs,
        stageDurationsMs: timing.stageDurationsMs,
        currentStageElapsedMs: timing.watchdog.currentStageElapsedMs,
        totalBootMs: timing.totalBootMs,
        totalStageDurationMs: timing.totalStageDurationMs,
        parallelPreloadEnabled: this._parallelPreload,
        parallelPreloadSavedMs: timing.parallelPreloadSavedMs,
        preloadTaskTimingsMs: timing.preloadTaskTimingsMs,
        preloadStats: timing.preloadStats,
      };
    }

    /**
     * Return a cloned snapshot of parallel preload timing and task status statistics.
     */
    getPreloadStats() {
      return clonePreloadStats(this._preloadStats);
    }

    /**
     * Release any retained large binary preload buffers (`shaderPackBytes`, `pgim.bytes`/`buffer`,
     * `reflectionManifest.bytes`, `reflectionSidecar.bytes`, `sidecars[k].bytes`/`buffer`) and
     * drop internal bundle references while preserving `getPreloadStats()`, `getState()`, and
     * `assets`. Idempotent: calling again returns `{ releasedFields: [], releasedBytes: 0 }`.
     *
     * @returns {{ releasedFields: string[], releasedBytes: number }}
     */
    releasePreloadBuffers() {
      const bundle = this._bundle;
      this._bundle = null;
      if (!bundle) {
        return { releasedFields: [], releasedBytes: 0 };
      }
      return releaseBootBundleBytes(bundle);
    }

    _evaluateWatchdog(now = this._readNow()) {
      const active = WATCHED_BOOT_PHASES.has(this._phase);
      const currentStageElapsedMs = active ? clampNonNegativeFinite(now - this._stageEnteredAt) : 0;
      const enabled = this._stageTimeoutMs > 0;
      const isStalled = active && enabled && currentStageElapsedMs >= this._stageTimeoutMs;
      return {
        currentStageElapsedMs,
        isStalled,
        stalledStage: isStalled ? this._phase : null,
        stalledMs: isStalled ? currentStageElapsedMs : 0,
      };
    }

    /**
     * Evaluate whether the current active boot stage has exceeded `stageTimeoutMs`.
     * Non-destructive: never aborts or mutates phase transitions.
     *
     * @returns {{
     *   isStalled: boolean,
     *   stalledStage: string | null,
     *   stalledMs: number,
     *   stageTimeoutMs: number,
     *   phase: string,
     *   currentStageElapsedMs: number,
     * }}
     */
    checkStageStall() {
      const now = this._readNow();
      const watchdog = this._evaluateWatchdog(now);

      if (watchdog.isStalled) {
        const newlyStalled = !this._stalled || this._stalledStage !== watchdog.stalledStage;
        this._stalled = true;
        this._stalledStage = watchdog.stalledStage;
        this._stalledMs = watchdog.stalledMs;
        if (newlyStalled) {
          this._stallCount += 1;
          invokeQuiet(this._onStageStall, {
            phase: this._phase,
            bootMode: this._bootMode,
            stalledStage: watchdog.stalledStage,
            stalledMs: watchdog.stalledMs,
            stageTimeoutMs: this._stageTimeoutMs,
            currentStageElapsedMs: watchdog.currentStageElapsedMs,
            stageDurationsMs: this._cloneStageDurations(),
            stallCount: this._stallCount,
          });
        }
      } else {
        this._stalled = false;
        this._stalledStage = null;
        this._stalledMs = 0;
      }

      return {
        isStalled: watchdog.isStalled,
        stalledStage: watchdog.stalledStage,
        stalledMs: watchdog.stalledMs,
        stageTimeoutMs: this._stageTimeoutMs,
        phase: this._phase,
        currentStageElapsedMs: watchdog.currentStageElapsedMs,
      };
    }

    /**
     * Returns true ONLY when the orchestrator is running a real Wasm64 guest boot
     * (`bootMode === 'wasm-guest'`).
     */
    isRealGuestBoot() {
      return this._bootMode === BOOT_MODE.WASM_GUEST;
    }

    getState() {
      const timing = this._buildTimingSnapshot();
      const { watchdog } = timing;
      return {
        phase: this._phase,
        bootMode: this._bootMode,
        programManifestUrl: this._assets ? this._assets.programManifestUrl : null,
        gameBase: this._assets ? this._assets.gameBase : null,
        status: this._status ? { ...this._status } : null,
        assets: this._assets ? { ...this._assets } : null,
        bundle: this._bundle,
        progress: this._progress ? { ...this._progress } : null,
        tracer: tracerSnapshot(this._bootTracer),
        fallbackReason: this._fallbackReason,
        error: this._error ? this._error.message : null,
        firstPresent: this._firstPresent,
        firstPresentSource: this._firstPresentSource,
        destroyed: this._destroyed,
        aborted: this._aborted,
        parallelPreload: this._parallelPreload,
        stageDurationsMs: timing.stageDurationsMs,
        phaseTimingsMs: timing.phaseTimingsMs,
        currentStageElapsedMs: watchdog.currentStageElapsedMs,
        totalStageDurationMs: timing.totalStageDurationMs,
        totalBootMs: timing.totalBootMs,
        parallelPreloadSavedMs: timing.parallelPreloadSavedMs,
        preloadTaskTimingsMs: timing.preloadTaskTimingsMs,
        preloadStats: timing.preloadStats,
        stageTimeoutMs: this._stageTimeoutMs,
        isStalled: watchdog.isStalled,
        stalledStage: watchdog.stalledStage,
        stalledMs: watchdog.stalledMs,
        stallCount: this._stallCount,
      };
    }

    /**
     * Export a JSON-safe, sanitized diagnostic snapshot with host paths, query strings,
     * and auth tokens redacted and binary bundle payloads omitted.
     */
    getDiagnosticSnapshot() {
      const timing = this._buildTimingSnapshot();
      const { watchdog } = timing;
      return {
        phase: this._phase,
        bootMode: this._bootMode,
        firstPresent: this._firstPresent,
        firstPresentSource: this._firstPresentSource,
        destroyed: this._destroyed,
        aborted: this._aborted,
        parallelPreload: this._parallelPreload,
        fallbackReason: this._fallbackReason ? sanitizeDiagnosticText(this._fallbackReason) : null,
        error: this._error ? sanitizeDiagnosticText(this._error.message) : null,
        stageDurationsMs: timing.stageDurationsMs,
        phaseTimingsMs: timing.phaseTimingsMs,
        currentStageElapsedMs: watchdog.currentStageElapsedMs,
        totalStageDurationMs: timing.totalStageDurationMs,
        totalBootMs: timing.totalBootMs,
        parallelPreloadSavedMs: timing.parallelPreloadSavedMs,
        preloadTaskTimingsMs: timing.preloadTaskTimingsMs,
        preloadStats: timing.preloadStats,
        stageTimeoutMs: this._stageTimeoutMs,
        isStalled: watchdog.isStalled,
        stalledStage: watchdog.stalledStage,
        stalledMs: watchdog.stalledMs,
        stallCount: this._stallCount,
        tracer: tracerSnapshot(this._bootTracer),
      };
    }

    _setPhase(nextPhase, detail = null) {
      if (this._phase === nextPhase) return;
      const now = this._readNow();
      const previousPhase = this._phase;
      if (WATCHED_BOOT_PHASES.has(previousPhase)) {
        const elapsedMs = clampNonNegativeFinite(now - this._stageEnteredAt);
        const prevAccum = clampNonNegativeFinite(this._stageDurationsMs[previousPhase]);
        this._stageDurationsMs[previousPhase] = clampNonNegativeFinite(prevAccum + elapsedMs);
      }
      this._phase = nextPhase;
      this._stageEnteredAt = now;
      this._stalled = false;
      this._stalledStage = null;
      this._stalledMs = 0;

      const stageDurationsMs = this._cloneStageDurations();
      const phaseTimingsMs = this._cloneStageDurations();
      const totalBootMs = sumStageDurations(stageDurationsMs, 0);
      const parallelPreloadSavedMs = clampNonNegativeFinite(this._preloadStats.parallelPreloadSavedMs);

      const event = {
        phase: nextPhase,
        previousPhase,
        bootMode: this._bootMode,
        programManifestUrl: this._assets ? this._assets.programManifestUrl : null,
        gameBase: this._assets ? this._assets.gameBase : null,
        firstPresentSource: this._firstPresentSource,
        status: this._status,
        assets: this._assets,
        bundle: this._bundle,
        progress: this._progress,
        tracer: tracerSnapshot(this._bootTracer),
        fallbackReason: this._fallbackReason,
        error: this._error,
        stageDurationsMs,
        phaseTimingsMs,
        totalBootMs,
        parallelPreloadSavedMs,
        ...(detail && typeof detail === 'object' ? detail : {}),
      };
      invokeQuiet(this._onPhaseChange, event);
      this._pushOverlay({ phase: nextPhase, ...event });
    }

    _pushOverlay(payload) {
      const overlay = this._splashOverlay;
      if (!overlay) return;
      callQuiet(overlay, 'update', payload);
      if (payload && payload.progress) {
        callQuiet(overlay, 'updateProgress', payload.progress);
      }
      if (payload && payload.tracer) {
        callQuiet(overlay, 'updateTracer', payload.tracer);
        callQuiet(overlay, 'setTracerSummary', payload.tracer);
      }
    }

    _initOverlay(splashImage) {
      const overlay = this._splashOverlay;
      if (!overlay) return;
      const tracer = tracerSnapshot(this._bootTracer);
      const payload = {
        splashImage,
        bootMode: this._bootMode,
        programManifestUrl: this._assets ? this._assets.programManifestUrl : null,
        gameBase: this._assets ? this._assets.gameBase : null,
        status: this._status,
        assets: this._assets,
        tracer,
        phase: this._phase,
      };
      if (typeof overlay.init === 'function') {
        callQuiet(overlay, 'init', payload);
      } else {
        callQuiet(overlay, 'show', payload);
      }
      if (splashImage) {
        callQuiet(overlay, 'setSplashImage', splashImage);
        callQuiet(overlay, 'setImage', splashImage);
      }
      this._pushOverlay(payload);
    }

    _forwardProgress(event) {
      const progress = event && typeof event === 'object' ? event : { loadedBytes: 0, totalBytes: null, pct: 0 };
      this._progress = {
        phase: progress.phase || 'download',
        asset: progress.asset || 'bundle',
        loadedBytes: Number.isFinite(progress.loadedBytes) ? progress.loadedBytes : 0,
        totalBytes: Number.isFinite(progress.totalBytes) ? progress.totalBytes : null,
        pct: Number.isFinite(progress.pct) ? progress.pct : 0,
      };
      callQuiet(this._bootTracer, 'recordProgress', this._progress);
      callQuiet(this._bootTracer, 'updateProgress', this._progress);

      const completedPgim =
        (this._progress.asset === 'pgim' || this._progress.asset === 'bundle') &&
        this._progress.phase === 'complete';
      if (completedPgim && !this._pgimLoadMarked && this._bootTracer && typeof this._bootTracer.markMilestone === 'function') {
        this._pgimLoadMarked = true;
        try {
          this._bootTracer.markMilestone('PGIM_LOAD', {
            loadedBytes: this._progress.loadedBytes,
            totalBytes: this._progress.totalBytes,
          });
        } catch {
          // Milestone names are fixed; ignore tracer validation errors.
        }
      }

      this._pushOverlay({
        phase: this._phase,
        bootMode: this._bootMode,
        progress: this._progress,
        tracer: tracerSnapshot(this._bootTracer),
        status: this._status,
      });
    }

    async _fetchStatus(signal) {
      if (!this._fetchImpl) {
        throw new Error(`Unable to fetch ${this._statusUrl}: fetch API is unavailable`);
      }
      throwIfAborted(signal);
      const response = await this._fetchImpl(this._statusUrl, { cache: 'no-store', signal });
      throwIfAborted(signal);
      if (!response || typeof response.ok !== 'boolean') {
        throw new Error(`Invalid response from ${this._statusUrl}`);
      }
      if (!response.ok) {
        const err = new Error(`Status probe ${this._statusUrl} failed with HTTP ${response.status ?? 'unknown'}`);
        err.status = response.status ?? null;
        throw err;
      }
      const payload = await response.json();
      throwIfAborted(signal);
      if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
        throw new Error(`Status probe ${this._statusUrl} returned non-object JSON`);
      }
      return payload;
    }

    async _preloadBundle(assets, signal) {
      const preloader = this._preloader;
      const needsPreloader = Boolean(
        assets.pgimUrl || assets.shaderPackUrl || assets.shaderPackIndexUrl || assets.reflectionSidecarUrl
      );
      if (needsPreloader && !preloader) {
        throw new Error('Boot artifacts are present but no preloader is configured');
      }

      const onProgress = (event) => this._forwardProgress(event);
      const forwardFor = (asset) => (event) => {
        const safeEvent = event && typeof event === 'object' ? event : {};
        this._forwardProgress({ ...safeEvent, asset: safeEvent.asset || asset });
      };

      const hasBootBundle = Boolean(preloader && typeof preloader.preloadBootBundle === 'function');
      const pgimLoader =
        preloader && typeof preloader.preloadPgim === 'function'
          ? preloader.preloadPgim.bind(preloader)
          : preloader && typeof preloader.preloadGuestImage === 'function'
            ? preloader.preloadGuestImage.bind(preloader)
            : null;
      const shaderPackIndexLoader =
        preloader && typeof preloader.preloadShaderPackHeader === 'function'
          ? preloader.preloadShaderPackHeader.bind(preloader)
          : preloader && typeof preloader.preloadShaderPackIndex === 'function'
            ? preloader.preloadShaderPackIndex.bind(preloader)
            : null;
      const reflectionSidecarLoader =
        preloader && typeof preloader.preloadReflectionSidecar === 'function'
          ? preloader.preloadReflectionSidecar.bind(preloader)
          : preloader && typeof preloader.preloadSidecar === 'function'
            ? preloader.preloadSidecar.bind(preloader)
            : null;

      if (assets.pgimUrl && !hasBootBundle && !pgimLoader) {
        throw new TypeError('preloader.preloadBootBundle must be a function');
      }

      const useBootBundle = Boolean(assets.pgimUrl && hasBootBundle && (this._parallelPreload || !pgimLoader));
      const tasks = [];

      if (useBootBundle) {
        tasks.push({
          name: 'bootBundle',
          asset: 'bundle',
          kind: 'bootBundle',
          optional: false,
          run: (taskSignal) =>
            preloader.preloadBootBundle({
              pgimUrl: assets.pgimUrl,
              shaderPackUrl: assets.shaderPackUrl,
              reflectionUrl: assets.reflectionManifestUrl,
              onProgress,
              signal: taskSignal,
            }),
        });
      } else {
        if (assets.pgimUrl && pgimLoader) {
          tasks.push({
            name: 'pgim',
            asset: 'pgim',
            kind: 'pgim',
            optional: false,
            run: (taskSignal) => pgimLoader(assets.pgimUrl, { onProgress: forwardFor('pgim'), signal: taskSignal }),
          });
        }
        if (assets.shaderPackUrl && preloader && typeof preloader.preloadShaderPack === 'function') {
          tasks.push({
            name: 'shaderPack',
            asset: 'shaderPack',
            kind: 'shaderPack',
            optional: Boolean(assets.pgimUrl),
            run: (taskSignal) =>
              preloader.preloadShaderPack(assets.shaderPackUrl, {
                onProgress: forwardFor('shaderPack'),
                signal: taskSignal,
              }),
          });
        }
        if (assets.reflectionManifestUrl && preloader && typeof preloader.preloadReflectionManifest === 'function') {
          tasks.push({
            name: 'reflection',
            asset: 'reflection',
            kind: 'reflection',
            optional: true,
            run: (taskSignal) =>
              preloader.preloadReflectionManifest(assets.reflectionManifestUrl, {
                onProgress: forwardFor('reflection'),
                signal: taskSignal,
              }),
          });
        }
      }

      if (shaderPackIndexLoader && (assets.shaderPackIndexUrl || assets.shaderPackUrl)) {
        const indexUrl = assets.shaderPackIndexUrl || assets.shaderPackUrl;
        tasks.push({
          name: 'shaderPackIndex',
          asset: 'shaderPackIndex',
          kind: 'shaderPackIndex',
          optional: true,
          run: (taskSignal) =>
            shaderPackIndexLoader(indexUrl, {
              onProgress: forwardFor('shaderPackIndex'),
              signal: taskSignal,
            }),
        });
      }

      if (assets.reflectionSidecarUrl) {
        const sidecarFn =
          reflectionSidecarLoader ||
          (preloader && typeof preloader.preloadReflectionManifest === 'function'
            ? preloader.preloadReflectionManifest.bind(preloader)
            : null);
        if (sidecarFn) {
          tasks.push({
            name: 'reflectionSidecar',
            asset: 'reflectionSidecar',
            kind: 'reflectionSidecar',
            optional: true,
            run: (taskSignal) =>
              sidecarFn(assets.reflectionSidecarUrl, {
                onProgress: forwardFor('reflectionSidecar'),
                signal: taskSignal,
              }),
          });
        }
      }

      for (const customTask of this._preloadTasks) {
        tasks.push({
          name: customTask.name,
          asset: customTask.asset,
          kind: 'custom',
          optional: customTask.optional,
          run: (taskSignal) =>
            customTask.run({
              assets: { ...assets },
              status: this._status ? { ...this._status } : null,
              signal: taskSignal,
              onProgress: forwardFor(customTask.asset),
            }),
        });
      }

      if (tasks.length === 0) {
        this._preloadStats = emptyPreloadStats(this._parallelPreload);
        return {
          pgim: null,
          shaderPackBytes: null,
          reflectionManifest: null,
          bytesLoaded: 0,
        };
      }

      const batchStartedAt = this._readNow();
      const { controller: batchController, unlink: unlinkBatch } = linkSignal(signal);
      const batchSignal = batchController.signal;

      let inFlight = 0;
      let maxConcurrency = 0;
      let firstFatalError = null;
      let completedCount = 0;
      let optionalMissingCount = 0;
      let failedCount = 0;
      let abortedCount = 0;
      let sequentialTimeMs = 0;
      let effectiveTaskCount = 0;
      let reportedBootBundleTotalMs = 0;
      const results = new Map();
      const taskTimingsMs = Object.create(null);
      const taskRecords = Object.create(null);

      const runSingleTask = async (task) => {
        const startedAt = this._readNow();
        inFlight += 1;
        if (inFlight > maxConcurrency) {
          maxConcurrency = inFlight;
        }
        try {
          throwIfAborted(batchSignal);
          const value = await raceAbort(Promise.resolve().then(() => task.run(batchSignal)), batchSignal);
          const durationMs = clampNonNegativeFinite(this._readNow() - startedAt);
          results.set(task.name, value);

          const subTiming =
            task.kind === 'bootBundle' &&
            value &&
            typeof value === 'object' &&
            value.timingMs &&
            typeof value.timingMs === 'object'
              ? value.timingMs
              : null;
          const subKeys = subTiming
            ? Object.keys(subTiming).filter(
                (key) =>
                  key !== 'total' &&
                  typeof subTiming[key] === 'number' &&
                  Number.isFinite(subTiming[key]) &&
                  subTiming[key] >= 0
              )
            : [];

          if (subKeys.length > 0) {
            for (const key of subKeys) {
              const subDur = clampNonNegativeFinite(subTiming[key]);
              taskTimingsMs[key] = subDur;
              taskRecords[key] = {
                status: 'completed',
                optional: key !== 'pgim',
                durationMs: subDur,
                error: null,
              };
              sequentialTimeMs += subDur;
              effectiveTaskCount += 1;
              completedCount += 1;
            }
            if (typeof subTiming.total === 'number' && Number.isFinite(subTiming.total) && subTiming.total > 0) {
              reportedBootBundleTotalMs = clampNonNegativeFinite(subTiming.total);
            }
            if (subKeys.length > maxConcurrency) {
              maxConcurrency = subKeys.length;
            }
          } else {
            taskTimingsMs[task.name] = durationMs;
            taskRecords[task.name] = {
              status: 'completed',
              optional: Boolean(task.optional),
              durationMs,
              error: null,
            };
            sequentialTimeMs += durationMs;
            effectiveTaskCount += 1;
            completedCount += 1;
          }
        } catch (err) {
          const durationMs = clampNonNegativeFinite(this._readNow() - startedAt);
          const abortedByOuter = Boolean(signal && signal.aborted);
          const abortedBySibling = Boolean(!abortedByOuter && batchSignal.aborted && firstFatalError !== null);
          const explicitAbort = isAbortError(err);

          if (abortedByOuter || abortedBySibling || explicitAbort) {
            abortedCount += 1;
            taskTimingsMs[task.name] = durationMs;
            taskRecords[task.name] = {
              status: 'aborted',
              optional: Boolean(task.optional),
              durationMs,
              error: sanitizeDiagnosticText(toError(err, 'Preload task aborted').message),
            };
            if (!abortedBySibling && firstFatalError === null) {
              firstFatalError = err;
              if (!batchSignal.aborted) {
                batchController.abort(err);
              }
            }
            return;
          }

          if (task.optional && !isValidationError(err)) {
            optionalMissingCount += 1;
            results.set(task.name, null);
            const statusLabel = isMissingOptionalError(err) ? 'missing' : 'fallback';
            taskTimingsMs[task.name] = durationMs;
            taskRecords[task.name] = {
              status: statusLabel,
              optional: true,
              durationMs,
              error: sanitizeDiagnosticText(toError(err, `Optional preload task ${task.name} failed`).message),
            };
            sequentialTimeMs += durationMs;
            effectiveTaskCount += 1;
            this._forwardProgress({
              phase: 'optional-missing',
              asset: task.asset,
              loadedBytes: 0,
              totalBytes: 0,
              pct: 0,
            });
            return;
          }

          failedCount += 1;
          taskTimingsMs[task.name] = durationMs;
          taskRecords[task.name] = {
            status: 'failed',
            optional: Boolean(task.optional),
            durationMs,
            error: sanitizeDiagnosticText(toError(err, `Preload task ${task.name} failed`).message),
          };
          if (firstFatalError === null) {
            firstFatalError = err;
            if (!batchSignal.aborted) {
              batchController.abort(err);
            }
          }
        } finally {
          inFlight -= 1;
        }
      };

      try {
        if (this._parallelPreload && tasks.length > 1) {
          await Promise.all(tasks.map((task) => runSingleTask(task)));
        } else {
          for (const task of tasks) {
            throwIfAborted(batchSignal);
            await runSingleTask(task);
            if (firstFatalError !== null) break;
          }
        }
      } finally {
        unlinkBatch();
        const measuredWallMs = clampNonNegativeFinite(this._readNow() - batchStartedAt);
        const effectiveWallMs = Math.max(measuredWallMs, reportedBootBundleTotalMs);
        const canSave = this._parallelPreload && effectiveTaskCount > 1 && failedCount === 0 && abortedCount === 0;
        const parallelPreloadSavedMs = canSave
          ? clampNonNegativeFinite(sequentialTimeMs - effectiveWallMs)
          : 0;
        this._preloadStats = {
          parallelEnabled: this._parallelPreload,
          taskCount: effectiveTaskCount > 0 ? effectiveTaskCount : tasks.length,
          completedCount,
          optionalMissingCount,
          failedCount,
          abortedCount,
          maxConcurrency,
          wallTimeMs: effectiveWallMs,
          sequentialTimeMs: clampNonNegativeFinite(sequentialTimeMs),
          parallelPreloadSavedMs,
          taskTimingsMs,
          tasks: taskRecords,
        };
      }

      throwIfAborted(signal);
      if (firstFatalError !== null) {
        throw firstFatalError;
      }

      const customSidecars = this._preloadTasks.length > 0 ? {} : null;
      if (customSidecars) {
        for (const customTask of this._preloadTasks) {
          customSidecars[customTask.name] = results.has(customTask.name) ? results.get(customTask.name) : null;
        }
      }

      if (useBootBundle) {
        const baseBundle = results.get('bootBundle') || null;
        if (tasks.length === 1) {
          return baseBundle;
        }
        const merged = baseBundle && typeof baseBundle === 'object' ? { ...baseBundle } : { pgim: null };
        let extraBytes = 0;
        if (results.has('shaderPackIndex')) {
          merged.shaderPackIndex = results.get('shaderPackIndex');
          extraBytes += inferBytesLoaded(merged.shaderPackIndex);
        }
        if (results.has('reflectionSidecar')) {
          merged.reflectionSidecar = results.get('reflectionSidecar');
          extraBytes += inferBytesLoaded(merged.reflectionSidecar);
        }
        if (customSidecars) {
          merged.sidecars = customSidecars;
          for (const key of Object.keys(customSidecars)) {
            extraBytes += inferBytesLoaded(customSidecars[key]);
          }
        }
        const baseBytes = inferBytesLoaded(baseBundle);
        merged.bytesLoaded = baseBytes + extraBytes;
        return merged;
      }

      const pgimResult = results.has('pgim') ? results.get('pgim') : null;
      const shaderPackResult = results.has('shaderPack') ? results.get('shaderPack') : null;
      const shaderPackIndexResult = results.has('shaderPackIndex') ? results.get('shaderPackIndex') : null;
      const reflectionResult = results.has('reflection') ? results.get('reflection') : null;
      const reflectionSidecarResult = results.has('reflectionSidecar') ? results.get('reflectionSidecar') : null;

      let bytesLoaded =
        inferBytesLoaded(pgimResult) +
        inferBytesLoaded(shaderPackResult) +
        inferBytesLoaded(shaderPackIndexResult) +
        inferBytesLoaded(reflectionResult) +
        inferBytesLoaded(reflectionSidecarResult);
      if (customSidecars) {
        for (const key of Object.keys(customSidecars)) {
          bytesLoaded += inferBytesLoaded(customSidecars[key]);
        }
      }

      const bundle = {
        pgim: pgimResult,
        shaderPackBytes: shaderPackResult,
        reflectionManifest: reflectionResult,
        bytesLoaded,
      };
      if (results.has('shaderPackIndex')) {
        bundle.shaderPackIndex = shaderPackIndexResult;
      }
      if (results.has('reflectionSidecar')) {
        bundle.reflectionSidecar = reflectionSidecarResult;
      }
      if (customSidecars) {
        bundle.sidecars = customSidecars;
      }
      return bundle;
    }

    _fallbackReasonFor(assets) {
      if (!assets.runtimeWasmAvailable) {
        return 'No Wasm64 runtime binary available (runtimeWasm/hasWasmBinary); falling back to WebGPU G5 smoke demo.';
      }
      if (!assets.pgimUrl && !assets.programManifestUrl) {
        return 'No PGIM guest image URL (pgimUrl) or programManifestUrl discovered; falling back to WebGPU G5 smoke demo.';
      }
      return 'Runtime launch requirements were not met; falling back to WebGPU G5 smoke demo.';
    }

    /**
     * Run discovery → optional preload → runtime launch or smoke-demo fallback.
     *
     * @param {{ signal?: AbortSignal }} [options]
     */
    async start(options = {}) {
      if (this._destroyed) {
        return {
          ok: false,
          destroyed: true,
          phase: BOOT_PHASE.DESTROYED,
          bootMode: this._bootMode,
          error: new Error('BootOrchestrator has been destroyed'),
        };
      }
      if (this._startPromise) {
        return this._startPromise;
      }
      if (
        this._phase === BOOT_PHASE.LAUNCHING_RUNTIME ||
        this._phase === BOOT_PHASE.FALLBACK_SMOKE_DEMO ||
        this._phase === BOOT_PHASE.PRESENTING
      ) {
        const timing = this._buildTimingSnapshot();
        return {
          ok: true,
          alreadyStarted: true,
          phase: this._phase,
          bootMode: this._bootMode,
          programManifestUrl: this._assets ? this._assets.programManifestUrl : null,
          gameBase: this._assets ? this._assets.gameBase : null,
          status: this._status,
          assets: this._assets,
          bundle: this._bundle,
          fallbackReason: this._fallbackReason,
          phaseTimingsMs: timing.phaseTimingsMs,
          parallelPreloadSavedMs: timing.parallelPreloadSavedMs,
          totalBootMs: timing.totalBootMs,
        };
      }

      const outerSignal = options && options.signal ? options.signal : null;
      if (outerSignal && typeof outerSignal.aborted !== 'boolean') {
        throw new TypeError('start({ signal }) requires an AbortSignal');
      }

      const token = ++this._sessionToken;
      const linked = linkSignal(outerSignal);
      this._controller = linked.controller;
      this._abortUnlink = linked.unlink;
      const signal = this._controller.signal;

      const promise = this._startInternal(signal, token).finally(() => {
        if (this._startPromise === promise) {
          this._startPromise = null;
        }
        if (typeof this._abortUnlink === 'function') {
          this._abortUnlink();
          this._abortUnlink = null;
        }
      });
      this._startPromise = promise;
      return promise;
    }

    async _startInternal(signal, token) {
      this._bootMode = BOOT_MODE.PENDING;
      this._error = null;
      this._fallbackReason = null;
      this._bundle = null;
      this._progress = null;
      this._aborted = false;
      this._pgimLoadMarked = false;
      this._resetTimingTelemetry();

      try {
        throwIfAborted(signal);
        this._setPhase(BOOT_PHASE.DISCOVERING);
        const status = await this._fetchStatus(signal);
        if (token !== this._sessionToken) {
          return { ok: false, aborted: true, phase: this._phase, bootMode: this._bootMode };
        }
        this._assets = discoverBootAssets(status);
        this._status = {
          ...status,
          programManifestUrl: this._assets.programManifestUrl,
          gameBase: this._assets.gameBase,
          bootMode: this._assets.bootMode,
        };
        this._initOverlay(this._assets.splashImage);
        this._pushOverlay({
          phase: this._phase,
          bootMode: this._bootMode,
          programManifestUrl: this._assets.programManifestUrl,
          gameBase: this._assets.gameBase,
          status: this._status,
          assets: this._assets,
          tracer: tracerSnapshot(this._bootTracer),
          splashImage: this._assets.splashImage,
        });

        throwIfAborted(signal);
        const shouldPreload = Boolean(
          this._assets.pgimUrl ||
          this._assets.shaderPackUrl ||
          this._assets.shaderPackIndexUrl ||
          this._assets.reflectionSidecarUrl ||
          this._preloadTasks.length > 0
        );
        if (shouldPreload) {
          this._setPhase(BOOT_PHASE.PRELOADING);
          this._bundle = (await this._preloadBundle(this._assets, signal)) || null;
        }

        throwIfAborted(signal);
        if (token !== this._sessionToken) {
          return { ok: false, aborted: true, phase: this._phase, bootMode: this._bootMode };
        }

        const canLaunch =
          this._assets.runtimeWasmAvailable &&
          Boolean(this._assets.pgimUrl || this._assets.programManifestUrl);

        if (canLaunch) {
          this._bootMode = BOOT_MODE.WASM_GUEST;
          this._status = { ...this._status, bootMode: this._bootMode };
          const launchPayload = {
            bootMode: this._bootMode,
            programManifestUrl: this._assets.programManifestUrl,
            gameBase: this._assets.gameBase,
            status: this._status,
            assets: this._assets,
            bundle: this._bundle,
            bootTracer: this._bootTracer,
          };
          this._setPhase(BOOT_PHASE.LAUNCHING_RUNTIME);
          await invokeQuietAsync(this._onLaunchRuntime, launchPayload);
          if (this._releasePreloadBuffers && this._onLaunchRuntime) {
            this.releasePreloadBuffers();
          }
          const timing = this._buildTimingSnapshot();
          return {
            ok: true,
            phase: this._phase,
            ...launchPayload,
            phaseTimingsMs: timing.phaseTimingsMs,
            parallelPreloadSavedMs: timing.parallelPreloadSavedMs,
            totalBootMs: timing.totalBootMs,
          };
        }

        this._bootMode = BOOT_MODE.SMOKE_DEMO;
        this._status = { ...this._status, bootMode: this._bootMode };
        this._fallbackReason = this._fallbackReasonFor(this._assets);
        this._setPhase(BOOT_PHASE.FALLBACK_SMOKE_DEMO, { reason: this._fallbackReason });
        const fallbackPayload = {
          bootMode: this._bootMode,
          programManifestUrl: this._assets.programManifestUrl,
          gameBase: this._assets.gameBase,
          status: this._status,
          bundle: this._bundle,
          reason: this._fallbackReason,
          assets: this._assets,
          bootTracer: this._bootTracer,
        };
        await invokeQuietAsync(this._onFallbackSmokeDemo, fallbackPayload);
        if (this._releasePreloadBuffers && this._onFallbackSmokeDemo) {
          this.releasePreloadBuffers();
        }
        const timing = this._buildTimingSnapshot();
        return {
          ok: true,
          fallback: true,
          phase: this._phase,
          bootMode: this._bootMode,
          programManifestUrl: this._assets.programManifestUrl,
          gameBase: this._assets.gameBase,
          status: this._status,
          bundle: fallbackPayload.bundle,
          assets: this._assets,
          reason: this._fallbackReason,
          fallbackReason: this._fallbackReason,
          phaseTimingsMs: timing.phaseTimingsMs,
          parallelPreloadSavedMs: timing.parallelPreloadSavedMs,
          totalBootMs: timing.totalBootMs,
        };
      } catch (err) {
        if (this._destroyed) {
          this.releasePreloadBuffers();
          return { ok: false, destroyed: true, phase: BOOT_PHASE.DESTROYED, bootMode: this._bootMode };
        }
        if (token !== this._sessionToken) {
          return {
            ok: false,
            aborted: true,
            phase: this._phase,
            bootMode: this._bootMode,
            error: toError(err, 'Boot pipeline aborted'),
          };
        }
        this.releasePreloadBuffers();
        if (signal.aborted || isAbortError(err)) {
          this._aborted = true;
          this._setPhase(BOOT_PHASE.CANCELLED);
          const timing = this._buildTimingSnapshot();
          return {
            ok: false,
            aborted: true,
            phase: this._phase,
            bootMode: this._bootMode,
            error: toError(err, 'Boot pipeline aborted'),
            phaseTimingsMs: timing.phaseTimingsMs,
            parallelPreloadSavedMs: timing.parallelPreloadSavedMs,
            totalBootMs: timing.totalBootMs,
          };
        }
        const normalized = toError(err, 'Boot pipeline failed');
        this._error = normalized;
        this._setPhase(BOOT_PHASE.ERROR, { error: normalized });
        const timing = this._buildTimingSnapshot();
        return {
          ok: false,
          phase: this._phase,
          bootMode: this._bootMode,
          error: normalized,
          phaseTimingsMs: timing.phaseTimingsMs,
          parallelPreloadSavedMs: timing.parallelPreloadSavedMs,
          totalBootMs: timing.totalBootMs,
        };
      }
    }

    /**
     * Forward a runtime log line into BootTracer, refresh the splash overlay, and
     * auto-dismiss on FIRST_PRESENT (`readyForSplash === true`).
     *
     * @param {unknown} line
     */
    feedRuntimeLog(line) {
      if (this._destroyed) return null;
      let ingested = null;
      if (this._bootTracer && typeof this._bootTracer.ingestLogLine === 'function') {
        try {
          ingested = this._bootTracer.ingestLogLine(line);
        } catch {
          ingested = null;
        }
      }
      const tracer = tracerSnapshot(this._bootTracer);
      this._pushOverlay({
        phase: this._phase,
        bootMode: this._bootMode,
        firstPresentSource: this._firstPresentSource,
        tracer,
        status: this._status,
        logLine: typeof line === 'string' ? line : null,
        ingested,
      });
      if (tracer && tracer.readyForSplash === true) {
        this.notifyFirstPresent({ source: 'bootTracer', tracer, ingested });
      }
      return ingested;
    }

    /**
     * Transition to PRESENTING, dismiss splash, record `firstPresentSource`
     * (`'wasm-guest'` vs `'smoke-demo'`), and fire `onFirstPresent` once.
     *
     * @param {object} [meta]
     */
    notifyFirstPresent(meta = {}) {
      if (this._destroyed || this._firstPresent) {
        return {
          ok: true,
          alreadyPresented: true,
          phase: this._phase,
          bootMode: this._bootMode,
          firstPresentSource: this._firstPresentSource,
        };
      }
      const safeMeta = meta && typeof meta === 'object' ? meta : {};
      const firstPresentSource = resolveFirstPresentSource(safeMeta, this._bootMode);
      const enrichedMeta = {
        ...safeMeta,
        bootMode: this._bootMode,
        firstPresentSource,
        isRealGuestBoot: this.isRealGuestBoot(),
      };

      this._firstPresent = true;
      this._firstPresentSource = firstPresentSource;
      if (this._releasePreloadBuffers && this._bundle) {
        this.releasePreloadBuffers();
      }
      if (this._bootTracer && typeof this._bootTracer.markMilestone === 'function') {
        const tracer = tracerSnapshot(this._bootTracer);
        if (!tracer || tracer.readyForSplash !== true) {
          try {
            this._bootTracer.markMilestone('FIRST_PRESENT', enrichedMeta);
          } catch {
            // Ignore unknown-milestone errors from stub tracers.
          }
        }
      }
      this._setPhase(BOOT_PHASE.PRESENTING, {
        meta: enrichedMeta,
        firstPresentSource: this._firstPresentSource,
      });
      callQuiet(this._splashOverlay, 'dismiss', enrichedMeta);
      callQuiet(this._splashOverlay, 'hide', enrichedMeta);
      invokeQuiet(this._onFirstPresent, {
        ...enrichedMeta,
        status: this._status,
        assets: this._assets,
        bundle: this._bundle,
        tracer: tracerSnapshot(this._bootTracer),
      });
      return {
        ok: true,
        alreadyPresented: false,
        phase: this._phase,
        bootMode: this._bootMode,
        firstPresentSource: this._firstPresentSource,
      };
    }

    /**
     * Alias for `notifyFirstPresent(meta)` to mark FIRST_PRESENT cleanly for guest or smoke frames.
     *
     * @param {object} [meta]
     */
    markFirstPresent(meta = {}) {
      return this.notifyFirstPresent(meta);
    }

    /**
     * Reset boot state, stage timing breakdown, and stalled-stage watchdog back to IDLE
     * so the orchestrator can be started cleanly again. No-op once destroyed.
     */
    reset() {
      if (this._destroyed) {
        return {
          ok: false,
          destroyed: true,
          phase: BOOT_PHASE.DESTROYED,
          bootMode: this._bootMode,
        };
      }
      this._sessionToken += 1;
      this._startPromise = null;
      try {
        if (this._controller && typeof this._controller.abort === 'function' && !this._controller.signal.aborted) {
          this._controller.abort(abortReason(this._controller.signal));
        }
      } catch {
        // Ignore abort errors during reset.
      }
      if (typeof this._abortUnlink === 'function') {
        try {
          this._abortUnlink();
        } catch {
          // Ignore.
        }
        this._abortUnlink = null;
      }
      this._controller = new AbortController();
      this.releasePreloadBuffers();
      this._phase = BOOT_PHASE.IDLE;
      this._bootMode = BOOT_MODE.PENDING;
      this._status = null;
      this._assets = null;
      this._progress = null;
      this._fallbackReason = null;
      this._error = null;
      this._firstPresent = false;
      this._firstPresentSource = null;
      this._aborted = false;
      this._pgimLoadMarked = false;
      this._resetTimingTelemetry();
      return {
        ok: true,
        phase: this._phase,
        bootMode: this._bootMode,
      };
    }

    /**
     * Abort in-flight work, release retained preload buffers, destroy the overlay, and freeze
     * the orchestrator. Safe to call more than once.
     */
    destroy() {
      if (this._destroyed) {
        return { ok: true, alreadyDestroyed: true, phase: this._phase };
      }
      this._destroyed = true;
      this._sessionToken += 1;
      this._startPromise = null;
      this.releasePreloadBuffers();
      try {
        if (this._controller && typeof this._controller.abort === 'function' && !this._controller.signal.aborted) {
          this._controller.abort(abortReason(this._controller.signal));
        }
      } catch {
        // Ignore abort errors during teardown.
      }
      if (typeof this._abortUnlink === 'function') {
        try {
          this._abortUnlink();
        } catch {
          // Ignore.
        }
        this._abortUnlink = null;
      }
      callQuiet(this._splashOverlay, 'destroy');
      this._setPhase(BOOT_PHASE.DESTROYED);
      return { ok: true, alreadyDestroyed: false, phase: this._phase };
    }

    /**
     * Alias for `destroy()` that releases retained preload buffers and tears down the orchestrator.
     */
    dispose() {
      return this.destroy();
    }
  }

  const exportedApi = Object.freeze({
    BOOT_PHASE,
    BOOT_MODE,
    DEFAULT_STATUS_URL,
    DEFAULT_STAGE_TIMEOUT_MS,
    DEFAULT_PARALLEL_PRELOAD,
    DEFAULT_RELEASE_PRELOAD_BUFFERS,
    BootOrchestrator,
    discoverBootAssets,
    isRealGuestBoot,
    releaseBootBundleBytes,
    default: BootOrchestrator,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6BootOrchestrator = exportedApi;
  } else if (root) {
    root.PlayGta6BootOrchestrator = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
