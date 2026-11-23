// Sanitized crash reporter and runtime diagnostic bundle exporter for playgta6web.
// Zero external dependencies; clean ES module compatible with <script type="module">
// and Node.js ESM imports. Redacts local host paths, file:// URLs, query strings,
// and auth tokens so exported diagnostic bundles never leak host usernames,
// local filesystem paths, secrets, or game data.

export const SCHEMA_VERSION = 1;
export const DEFAULT_MAX_LOGS = 200;
export const DEFAULT_MAX_ERRORS = 32;
export const DEFAULT_APP_VERSION = '0.1.0';

export const ERROR_CATEGORIES = Object.freeze([
  'wasm_trap',
  'webgpu_validation',
  'webgpu_device_lost',
  'worker_error',
  'io_error',
  'unhandled_rejection',
  'runtime',
]);

export const LOG_LEVELS = Object.freeze(['debug', 'info', 'ok', 'warn', 'error']);

const ERROR_CATEGORY_SET = new Set(ERROR_CATEGORIES);
const LOG_LEVEL_SET = new Set(LOG_LEVELS);
const MAX_SANITIZE_DEPTH = 16;

const FILE_URL_RE =
  /\bfile:\/\/[^\s"'`<>)\]}]+?(?=:\d+:\d+(?:[)\s"'`<>\]};,]|$)|[\s"'`<>)\]}]|$)/gi;

const URL_USERINFO_RE = /\b((?:https?|wss?|ftp):\/\/)([^\s/@"'`<>]+):([^\s/@"'`<>]*)@/gi;

const URL_QUERY_RE =
  /((?:https?|wss?):\/\/[^\s"'`<>?#]+)\?(?!\[REDACTED\])[^\s"'`<>#)\]}]+?(?=:\d+:\d+(?:[)\s"'`<>\]}]|$)|[\s"'`<>#)\]}]|$)/gi;

const STANDALONE_QUERY_RE =
  /\?[A-Za-z0-9_.~%-]+=[^\s"'`<>#)\]}]+?(?=:\d+:\d+(?:[)\s"'`<>\]}]|$)|[\s"'`<>#)\]}]|$)/g;

const AUTH_SCHEME_RE = /\b(Bearer|Basic)\s+(?!\[REDACTED\])[A-Za-z0-9\-._~+/=:]+/gi;

const SECRET_KV_RE =
  /\b(authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|id[_-]?token|client[_-]?secret|x-api-key|x-auth-token|cookie|set-cookie|password|passwd|secret|token)(\s*[:=]\s*)(?!(?:(?:Bearer|Basic)\s+)?\[REDACTED\])(?:"[^"]*"|'[^']*'|[^\s,;}\])]+)/gi;

const QUOTED_WIN_PATH_RE = /(["'`])([A-Za-z]:[\\/][^"'`\r\n]+?)(:\d+:\d+)?\1/g;

const WIN_DRIVE_PATH_RE =
  /(?<![A-Za-z0-9_])([A-Za-z]:(?:\\|\/(?!\/))[^\s"'`<>|*?)\]},;]+?)(?=:\d+:\d+(?:[)\s"'`<>\]};,]|$)|[\s"'`<>|*?)\]},;]|$)/g;

const WIN_UNC_PATH_RE =
  /(^|[\s"'`(\[={<,:;])(\\\\[^\s\\"'`<>|*?)\]},;]+\\[^\s"'`<>|*?)\]},;]+?)(?=:\d+:\d+(?:[)\s"'`<>\]};,]|$)|[\s"'`<>|*?)\]},;]|$)/g;

const POSIX_ROOTS =
  'home|Users|root|tmp|var|etc|opt|usr|mnt|media|Volumes|private|srv|proc|sys|dev|run|Library|Applications';

const QUOTED_POSIX_PATH_RE = new RegExp(
  `(["'\`])(\\/(?:${POSIX_ROOTS})(?:\\/[^"'\`\\r\\n]*?)?)(:\\d+:\\d+)?\\1`,
  'g'
);

const POSIX_ABS_PATH_RE = new RegExp(
  `(^|[\\s(\\[={<,:;\`])(\\/(?:${POSIX_ROOTS})(?:\\/[^\\s"'\`<>\\)\\]},;]+?)?)(?=:\\d+:\\d+(?:[)\\s"'\`<>\\]};,]|$)|[\\s"'\`<>\\)\\]},;]|$)`,
  'g'
);

const TILDE_HOME_PATH_RE =
  /(^|[\s"'`(\[={<,:;])(~[A-Za-z0-9._-]*\/[^\s"'`<>)\]},;]+?)(?=:\d+:\d+(?:[)\s"'`<>\]};,]|$)|[\s"'`<>)\]},;]|$)/g;

const SENSITIVE_KEY_RE =
  /^(?:authorization|cookie|set[_-]?cookie|password|passwd|secret|client[_-]?secret|api[_-]?key|apikey|access[_-]?token|refresh[_-]?token|auth[_-]?token|id[_-]?token|bearer[_-]?token|session[_-]?id|token)$/i;

/**
 * Strips Windows paths (`C:\Users\...`), POSIX host paths (`/home/...`, `/Users/...`),
 * `file://` URLs, query strings (`?token=...`), URL credentials, and bearer/auth tokens
 * so diagnostic bundles never leak host usernames, local paths, or secrets.
 *
 * @param {unknown} text
 * @returns {string}
 */
export function sanitizeString(text) {
  if (text === null || text === undefined) {
    return '';
  }

  let out = typeof text === 'string' ? text : String(text);
  if (out.length === 0) {
    return out;
  }

  // 1. Redact file:// URLs completely (preserving optional :line:col in stack frames).
  out = out.replace(FILE_URL_RE, '[REDACTED_FILE_URL]');

  // 2. Redact embedded URL credentials (scheme://user:pass@host).
  out = out.replace(URL_USERINFO_RE, '$1[REDACTED]@');

  // 3. Redact query strings on HTTP/WS URLs and standalone ?key=value tokens.
  out = out.replace(URL_QUERY_RE, '$1?[REDACTED]');
  out = out.replace(STANDALONE_QUERY_RE, '?[REDACTED]');

  // 4. Redact Bearer/Basic tokens and inline credential key=value / header pairs.
  out = out.replace(AUTH_SCHEME_RE, '$1 [REDACTED]');
  out = out.replace(SECRET_KV_RE, '$1$2[REDACTED]');

  // 5. Redact Windows absolute drive and UNC paths (quoted first to catch spaces).
  out = out.replace(QUOTED_WIN_PATH_RE, (_, quote, _path, lineCol) => {
    return `${quote}[REDACTED_PATH]${lineCol || ''}${quote}`;
  });
  out = out.replace(WIN_DRIVE_PATH_RE, '[REDACTED_PATH]');
  out = out.replace(WIN_UNC_PATH_RE, '$1[REDACTED_PATH]');

  // 6. Redact POSIX host absolute paths and ~ home paths (preserving app routes like /api/status).
  out = out.replace(QUOTED_POSIX_PATH_RE, (_, quote, _path, lineCol) => {
    return `${quote}[REDACTED_PATH]${lineCol || ''}${quote}`;
  });
  out = out.replace(POSIX_ABS_PATH_RE, '$1[REDACTED_PATH]');
  out = out.replace(TILDE_HOME_PATH_RE, '$1[REDACTED_PATH]');

  return out;
}

function isBinaryBuffer(value) {
  if (!value || typeof value !== 'object') {
    return false;
  }
  if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) {
    return true;
  }
  if (typeof SharedArrayBuffer === 'function' && value instanceof SharedArrayBuffer) {
    return true;
  }
  return false;
}

function isErrorLike(value) {
  if (!value || typeof value !== 'object') {
    return false;
  }
  if (value instanceof Error) {
    return true;
  }
  const tag = Object.prototype.toString.call(value);
  return tag === '[object Error]' || tag === '[object DOMException]';
}

/**
 * Recursively sanitizes any value for safe JSON diagnostic serialization.
 * Strips paths/secrets from all strings (and object keys), summarizes binary buffers
 * without dumping raw bytes, handles BigInt/Error/Date/Map/Set, and guards against cycles.
 *
 * @param {unknown} value
 * @param {WeakSet<object>} [seen]
 * @param {number} [depth]
 * @returns {any}
 */
export function sanitizeValue(value, seen = new WeakSet(), depth = 0) {
  if (value === null || value === undefined) {
    return null;
  }

  const kind = typeof value;
  if (kind === 'string') {
    return sanitizeString(value);
  }
  if (kind === 'number') {
    return Number.isFinite(value) ? value : null;
  }
  if (kind === 'boolean') {
    return value;
  }
  if (kind === 'bigint') {
    return value.toString();
  }
  if (kind === 'symbol' || kind === 'function') {
    return undefined;
  }

  if (depth >= MAX_SANITIZE_DEPTH) {
    return '[MaxDepth]';
  }

  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? null : value.toISOString();
  }

  // Never dump raw binary buffers (e.g. game assets or Wasm memory slices) into crash reports.
  if (isBinaryBuffer(value)) {
    return {
      type: value.constructor ? sanitizeString(value.constructor.name) : 'BinaryBuffer',
      byteLength: typeof value.byteLength === 'number' ? value.byteLength : 0,
    };
  }

  if (seen.has(value)) {
    return '[Circular]';
  }
  seen.add(value);

  try {
    if (Array.isArray(value)) {
      const out = [];
      for (let i = 0; i < value.length; i += 1) {
        const sanitized = sanitizeValue(value[i], seen, depth + 1);
        out.push(sanitized === undefined ? null : sanitized);
      }
      return out;
    }

    if (value instanceof Set) {
      const out = [];
      for (const item of value) {
        const sanitized = sanitizeValue(item, seen, depth + 1);
        out.push(sanitized === undefined ? null : sanitized);
      }
      return out;
    }

    if (value instanceof Map) {
      const out = {};
      for (const [k, v] of value.entries()) {
        const rawKey = typeof k === 'string' ? k : String(k);
        const safeKey = sanitizeString(rawKey);
        if (SENSITIVE_KEY_RE.test(rawKey) && typeof v === 'string' && v.length > 0) {
          out[safeKey] = '[REDACTED]';
        } else {
          const sanitized = sanitizeValue(v, seen, depth + 1);
          if (sanitized !== undefined) {
            out[safeKey] = sanitized;
          }
        }
      }
      return out;
    }

    if (isErrorLike(value)) {
      const errOut = {
        name: sanitizeString(value.name || 'Error'),
        message: sanitizeString(value.message || ''),
        stack: typeof value.stack === 'string' ? sanitizeString(value.stack) : null,
      };
      if (value.cause !== undefined) {
        errOut.cause = sanitizeValue(value.cause, seen, depth + 1);
      }
      if (typeof value.code === 'string' || typeof value.code === 'number') {
        errOut.code = sanitizeValue(value.code, seen, depth + 1);
      }
      if (typeof value.reason === 'string') {
        errOut.reason = sanitizeString(value.reason);
      }
      return errOut;
    }

    const out = {};
    for (const key of Object.keys(value)) {
      const raw = value[key];
      if (raw === undefined || typeof raw === 'function' || typeof raw === 'symbol') {
        continue;
      }
      const safeKey = sanitizeString(key);
      if (SENSITIVE_KEY_RE.test(key) && typeof raw === 'string' && raw.length > 0) {
        out[safeKey] = '[REDACTED]';
        continue;
      }
      const sanitized = sanitizeValue(raw, seen, depth + 1);
      if (sanitized !== undefined) {
        out[safeKey] = sanitized;
      }
    }
    return out;
  } finally {
    seen.delete(value);
  }
}

function normalizeLogLevel(level) {
  if (typeof level !== 'string') {
    return 'info';
  }
  const lower = level.trim().toLowerCase();
  if (LOG_LEVEL_SET.has(lower)) {
    return lower;
  }
  if (lower === 'warning') {
    return 'warn';
  }
  if (lower === 'err' || lower === 'fatal') {
    return 'error';
  }
  return 'info';
}

function formatIsoTimestamp(ms) {
  if (typeof ms === 'number' && Number.isFinite(ms)) {
    const date = new Date(ms);
    if (!Number.isNaN(date.getTime())) {
      return date.toISOString();
    }
  }
  return new Date(0).toISOString();
}

/**
 * Classifies an error into one of the canonical diagnostic categories.
 * If `preferredCategory` is already a canonical category, it is returned directly.
 *
 * @param {unknown} errorOrMessage
 * @param {string|null} [preferredCategory]
 * @returns {string}
 */
export function classifyError(errorOrMessage, preferredCategory = null) {
  if (typeof preferredCategory === 'string') {
    const normalized = preferredCategory.trim().toLowerCase();
    if (ERROR_CATEGORY_SET.has(normalized)) {
      return normalized;
    }
    if (normalized === 'unhandledrejection') {
      return 'unhandled_rejection';
    }
  }

  if (
    typeof WebAssembly !== 'undefined' &&
    errorOrMessage &&
    typeof errorOrMessage === 'object' &&
    ((typeof WebAssembly.RuntimeError === 'function' && errorOrMessage instanceof WebAssembly.RuntimeError) ||
      (typeof WebAssembly.CompileError === 'function' && errorOrMessage instanceof WebAssembly.CompileError) ||
      (typeof WebAssembly.LinkError === 'function' && errorOrMessage instanceof WebAssembly.LinkError))
  ) {
    return 'wasm_trap';
  }

  const name =
    errorOrMessage && typeof errorOrMessage === 'object' && typeof errorOrMessage.name === 'string'
      ? errorOrMessage.name
      : '';
  const message =
    typeof errorOrMessage === 'string'
      ? errorOrMessage
      : errorOrMessage && typeof errorOrMessage === 'object' && typeof errorOrMessage.message === 'string'
        ? errorOrMessage.message
        : '';
  const stack =
    errorOrMessage && typeof errorOrMessage === 'object' && typeof errorOrMessage.stack === 'string'
      ? errorOrMessage.stack
      : '';
  const combined = `${name} ${message} ${stack}`;

  if (/\b(?:device[\s_-]*lost|gpudevicelostinfo|webgpu.*lost)\b/i.test(combined)) {
    return 'webgpu_device_lost';
  }
  if (
    /\b(?:gpuvalidationerror|gpuoutofmemoryerror|gpuinternalerror|webgpu|wgpu|wgsl|validation\s+error)\b/i.test(
      combined
    )
  ) {
    return 'webgpu_validation';
  }
  if (
    /\b(?:wasm|webassembly|unreachable|memory\s+access\s+out\s+of\s+bounds|out\s+of\s+bounds\s+memory|indirect\s+call|integer\s+divide\s+by\s+zero|wasm_trap)\b/i.test(
      combined
    )
  ) {
    return 'wasm_trap';
  }
  if (/\b(?:worker|importscripts|postmessage|offscreencanvas)\b/i.test(combined)) {
    return 'worker_error';
  }
  if (/\b(?:opfs|httpfs|range\s+request|storage|quota|fetch|network|manifest|bootset|io_error)\b/i.test(combined)) {
    return 'io_error';
  }
  if (/\bunhandled\s*rejection\b/i.test(combined)) {
    return 'unhandled_rejection';
  }

  return 'runtime';
}

function extractErrorFields(errorOrMessage) {
  if (typeof errorOrMessage === 'string') {
    return {
      name: 'Error',
      message: sanitizeString(errorOrMessage),
      stack: null,
    };
  }

  if (errorOrMessage && typeof errorOrMessage === 'object') {
    const rawName =
      typeof errorOrMessage.name === 'string' && errorOrMessage.name.length > 0
        ? errorOrMessage.name
        : typeof errorOrMessage.reason === 'string' && !('stack' in errorOrMessage)
          ? 'GPUDeviceLostInfo'
          : errorOrMessage.constructor &&
              typeof errorOrMessage.constructor.name === 'string' &&
              errorOrMessage.constructor.name !== 'Object'
            ? errorOrMessage.constructor.name
            : 'Error';

    const rawMessage =
      typeof errorOrMessage.message === 'string'
        ? errorOrMessage.message
        : typeof errorOrMessage.reason === 'string'
          ? errorOrMessage.reason
          : String(errorOrMessage);

    const rawStack = typeof errorOrMessage.stack === 'string' ? errorOrMessage.stack : null;

    return {
      name: sanitizeString(rawName),
      message: sanitizeString(rawMessage),
      stack: rawStack !== null ? sanitizeString(rawStack) : null,
    };
  }

  return {
    name: 'Error',
    message: sanitizeString(errorOrMessage == null ? '' : String(errorOrMessage)),
    stack: null,
  };
}

class BoundedRing {
  constructor(capacity) {
    this._capacity = capacity;
    this._items = new Array(capacity);
    this._writeIdx = 0;
    this._count = 0;
    this._totalRecorded = 0;
  }

  get capacity() {
    return this._capacity;
  }

  get length() {
    return this._count;
  }

  get totalRecorded() {
    return this._totalRecorded;
  }

  get droppedCount() {
    return Math.max(0, this._totalRecorded - this._count);
  }

  push(item) {
    this._items[this._writeIdx] = item;
    this._writeIdx = (this._writeIdx + 1) % this._capacity;
    if (this._count < this._capacity) {
      this._count += 1;
    }
    this._totalRecorded += 1;
    return item;
  }

  toArray() {
    const out = new Array(this._count);
    const start = this._count < this._capacity ? 0 : this._writeIdx;
    for (let i = 0; i < this._count; i += 1) {
      out[i] = this._items[(start + i) % this._capacity];
    }
    return out;
  }

  clear() {
    this._items = new Array(this._capacity);
    this._writeIdx = 0;
    this._count = 0;
    this._totalRecorded = 0;
  }
}

/**
 * Captures Wasm traps, WebGPU validation/device-lost errors, worker/IO faults,
 * bounded recent runtime log lines, perf snapshots, and safe browser capability
 * fingerprints into a versioned, sanitized JSON diagnostic bundle.
 */
export class CrashReporter {
  /**
   * @param {object} [options]
   * @param {number} [options.maxLogs=200]
   * @param {number} [options.maxErrors=32]
   * @param {string} [options.appVersion='0.1.0']
   * @param {() => number} [options.now]
   * @param {object} [options.navigatorObj]
   * @param {object} [options.globalObj]
   */
  constructor(options = {}) {
    if (options !== undefined && (options === null || typeof options !== 'object')) {
      throw new TypeError('CrashReporter options must be an object');
    }

    const {
      maxLogs = DEFAULT_MAX_LOGS,
      maxErrors = DEFAULT_MAX_ERRORS,
      appVersion = DEFAULT_APP_VERSION,
      now = () => Date.now(),
      navigatorObj = typeof navigator !== 'undefined' ? navigator : undefined,
      globalObj = typeof globalThis !== 'undefined' ? globalThis : undefined,
    } = options;

    if (!Number.isInteger(maxLogs) || maxLogs < 1) {
      throw new RangeError(`maxLogs must be a positive integer, got ${maxLogs}`);
    }
    if (!Number.isInteger(maxErrors) || maxErrors < 1) {
      throw new RangeError(`maxErrors must be a positive integer, got ${maxErrors}`);
    }
    if (typeof now !== 'function') {
      throw new TypeError('now must be a function returning epoch milliseconds');
    }

    this._maxLogs = maxLogs;
    this._maxErrors = maxErrors;
    this._appVersion = sanitizeString(appVersion || DEFAULT_APP_VERSION);
    this._now = now;
    this._navigatorObj = navigatorObj;
    this._globalObj = globalObj;

    this._logs = new BoundedRing(maxLogs);
    this._errors = new BoundedRing(maxErrors);

    this._attachedTarget = null;
    this._boundOnError = null;
    this._boundOnRejection = null;
  }

  get maxLogs() {
    return this._maxLogs;
  }

  get maxErrors() {
    return this._maxErrors;
  }

  get appVersion() {
    return this._appVersion;
  }

  get logCount() {
    return this._logs.length;
  }

  get errorCount() {
    return this._errors.length;
  }

  get isGlobalHandlersAttached() {
    return this._attachedTarget !== null;
  }

  /**
   * Clears all buffered logs and errors without detaching global handlers.
   */
  clear() {
    this._logs.clear();
    this._errors.clear();
  }

  /**
   * Appends a sanitized log entry into the bounded ring buffer.
   *
   * @param {string} level
   * @param {string} source
   * @param {unknown} message
   * @returns {{ timestamp: string, timestampMs: number, level: string, source: string, message: string }}
   */
  recordLog(level, source, message) {
    const ts = this._now();
    const entry = {
      timestamp: formatIsoTimestamp(ts),
      timestampMs: Number.isFinite(ts) ? ts : 0,
      level: normalizeLogLevel(level),
      source: sanitizeString(source || 'runtime'),
      message: sanitizeString(message),
    };
    this._logs.push(entry);
    return entry;
  }

  /**
   * Records and classifies a runtime error into the bounded error ring buffer.
   *
   * @param {string|null|undefined} category
   * @param {unknown} errorOrMessage
   * @param {unknown} [context]
   * @returns {object}
   */
  recordError(category, errorOrMessage, context) {
    const ts = this._now();
    const resolvedCategory = classifyError(errorOrMessage, category);
    const fields = extractErrorFields(errorOrMessage);

    let sanitizedContext = context !== undefined ? sanitizeValue(context) : null;
    if (
      errorOrMessage &&
      typeof errorOrMessage === 'object' &&
      typeof errorOrMessage.reason === 'string' &&
      !Array.isArray(sanitizedContext)
    ) {
      const reason = sanitizeString(errorOrMessage.reason);
      if (sanitizedContext && typeof sanitizedContext === 'object') {
        if (sanitizedContext.reason === undefined) {
          sanitizedContext = { ...sanitizedContext, reason };
        }
      } else {
        sanitizedContext = { reason };
      }
    }

    const entry = {
      timestamp: formatIsoTimestamp(ts),
      timestampMs: Number.isFinite(ts) ? ts : 0,
      category: resolvedCategory,
      name: fields.name,
      message: fields.message,
      stack: fields.stack,
      context: sanitizedContext,
    };

    this._errors.push(entry);
    return entry;
  }

  /**
   * Returns a shallow copy of buffered sanitized log entries (oldest to newest).
   */
  getLogs() {
    return this._logs.toArray();
  }

  /**
   * Returns a shallow copy of buffered sanitized error entries (oldest to newest).
   */
  getErrors() {
    return this._errors.toArray();
  }

  /**
   * Safely fingerprints browser/runtime capabilities without leaking raw
   * user-agent strings, host paths, or cookies.
   *
   * @returns {object}
   */
  collectCapabilities() {
    const g = this._globalObj;
    const nav = this._navigatorObj;

    let crossOriginIsolated = false;
    try {
      crossOriginIsolated = Boolean(g && g.crossOriginIsolated === true);
    } catch {
      crossOriginIsolated = false;
    }

    let sharedArrayBuffer = false;
    try {
      sharedArrayBuffer = Boolean(g && typeof g.SharedArrayBuffer === 'function');
    } catch {
      sharedArrayBuffer = false;
    }

    let webgpu = false;
    try {
      webgpu = Boolean(nav && 'gpu' in nav && nav.gpu != null);
    } catch {
      webgpu = false;
    }

    let wasm64 = false;
    try {
      if (g && g.WebAssembly && typeof g.WebAssembly.Memory === 'function') {
        new g.WebAssembly.Memory({ initial: 1n, maximum: 2n, index: 'i64' });
        wasm64 = true;
      }
    } catch {
      wasm64 = false;
    }

    let opfs = false;
    try {
      opfs = Boolean(nav && nav.storage && typeof nav.storage.getDirectory === 'function');
    } catch {
      opfs = false;
    }

    let gamepad = false;
    try {
      gamepad = Boolean(nav && typeof nav.getGamepads === 'function');
    } catch {
      gamepad = false;
    }

    let audioWorklet = false;
    try {
      audioWorklet = Boolean(
        g &&
          (typeof g.AudioWorkletNode === 'function' ||
            typeof g.AudioWorklet === 'function' ||
            (typeof g.AudioContext === 'function' &&
              g.AudioContext.prototype &&
              'audioWorklet' in g.AudioContext.prototype))
      );
    } catch {
      audioWorklet = false;
    }

    let hardwareConcurrency = null;
    try {
      if (
        nav &&
        typeof nav.hardwareConcurrency === 'number' &&
        Number.isFinite(nav.hardwareConcurrency) &&
        nav.hardwareConcurrency > 0
      ) {
        hardwareConcurrency = Math.floor(nav.hardwareConcurrency);
      }
    } catch {
      hardwareConcurrency = null;
    }

    let deviceMemory = null;
    try {
      if (
        nav &&
        typeof nav.deviceMemory === 'number' &&
        Number.isFinite(nav.deviceMemory) &&
        nav.deviceMemory > 0
      ) {
        deviceMemory = nav.deviceMemory;
      }
    } catch {
      deviceMemory = null;
    }

    return {
      crossOriginIsolated,
      sharedArrayBuffer,
      webgpu,
      wasm64,
      opfs,
      gamepad,
      audioWorklet,
      hardwareConcurrency,
      deviceMemory,
    };
  }

  /**
   * Builds a versioned, recursively sanitized diagnostic report suitable for
   * JSON export or bug reports.
   *
   * @param {object} [options]
   * @param {object} [options.perfSnapshot] A `PerfMonitor.getSnapshot()` object or `PerfMonitor` instance.
   * @param {object} [options.storageSummary]
   * @param {object} [options.serverStatus]
   * @param {unknown} [options.extra]
   * @returns {object}
   */
  buildReport({ perfSnapshot, storageSummary, serverStatus, extra } = {}) {
    const ts = this._now();

    let resolvedPerf = null;
    if (perfSnapshot && typeof perfSnapshot === 'object') {
      if (typeof perfSnapshot.getSnapshot === 'function') {
        resolvedPerf = perfSnapshot.getSnapshot();
      } else {
        resolvedPerf = perfSnapshot;
      }
    }

    const report = {
      schemaVersion: SCHEMA_VERSION,
      generatedAt: formatIsoTimestamp(ts),
      appVersion: this._appVersion,
      capabilities: this.collectCapabilities(),
      errors: this._errors.toArray(),
      recentLogs: this._logs.toArray(),
      perf: resolvedPerf,
      storage: storageSummary !== undefined ? storageSummary : null,
      serverStatus: serverStatus !== undefined ? serverStatus : null,
    };

    if (extra !== undefined) {
      report.extra = extra;
    }

    return sanitizeValue(report);
  }

  /**
   * Serializes `buildReport(options)` into a sanitized JSON string.
   *
   * @param {object} [options]
   * @param {number|string} [space]
   * @returns {string}
   */
  toJson(options = {}, space) {
    const opts = options && typeof options === 'object' ? options : {};
    const indent =
      space !== undefined
        ? space
        : opts.pretty === false
          ? 0
          : typeof opts.space === 'number' || typeof opts.space === 'string'
            ? opts.space
            : 2;

    return JSON.stringify(this.buildReport(opts), null, indent);
  }

  /**
   * Hooks global `error` and `unhandledrejection` events on an EventTarget (e.g. `window` or `self`).
   * Replaces any previously attached target to avoid duplicate handlers.
   *
   * @param {EventTarget} [target]
   * @returns {() => void} Detach function.
   */
  attachGlobalHandlers(target = this._globalObj) {
    if (
      !target ||
      typeof target.addEventListener !== 'function' ||
      typeof target.removeEventListener !== 'function'
    ) {
      throw new TypeError('attachGlobalHandlers requires an EventTarget with addEventListener/removeEventListener');
    }

    this.detachGlobalHandlers();

    const onError = (event) => {
      const errObj =
        event && event.error !== undefined && event.error !== null
          ? event.error
          : {
              name: 'Error',
              message: event && typeof event.message === 'string' ? event.message : 'Unhandled error event',
            };

      const context = {};
      if (event && typeof event.filename === 'string' && event.filename.length > 0) {
        context.filename = event.filename;
      }
      if (event && typeof event.lineno === 'number' && Number.isFinite(event.lineno)) {
        context.lineno = event.lineno;
      }
      if (event && typeof event.colno === 'number' && Number.isFinite(event.colno)) {
        context.colno = event.colno;
      }

      const hasContext = Object.keys(context).length > 0;
      const category = classifyError(errObj, null);
      this.recordError(category, errObj, hasContext ? context : undefined);
    };

    const onRejection = (event) => {
      const reason =
        event && event.reason !== undefined && event.reason !== null
          ? event.reason
          : 'Unhandled promise rejection';
      this.recordError('unhandled_rejection', reason);
    };

    target.addEventListener('error', onError);
    target.addEventListener('unhandledrejection', onRejection);

    this._attachedTarget = target;
    this._boundOnError = onError;
    this._boundOnRejection = onRejection;

    return () => {
      this.detachGlobalHandlers();
    };
  }

  /**
   * Removes previously attached global `error` and `unhandledrejection` listeners. Idempotent.
   */
  detachGlobalHandlers() {
    if (this._attachedTarget) {
      if (this._boundOnError) {
        this._attachedTarget.removeEventListener('error', this._boundOnError);
      }
      if (this._boundOnRejection) {
        this._attachedTarget.removeEventListener('unhandledrejection', this._boundOnRejection);
      }
    }
    this._attachedTarget = null;
    this._boundOnError = null;
    this._boundOnRejection = null;
  }
}

export default CrashReporter;
