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

export const BOOT_ERROR_CODES = Object.freeze({
  WASM_MODULE_TOO_LARGE: 'WASM_MODULE_TOO_LARGE',
  WASM64_UNSUPPORTED: 'WASM64_UNSUPPORTED',
  COI_SAB_MISSING: 'COI_SAB_MISSING',
  WORKER_SCRIPT_COLLISION: 'WORKER_SCRIPT_COLLISION',
  HTTPFS_PGIM_ERROR: 'HTTPFS_PGIM_ERROR',
  UNKNOWN_BOOT_ERROR: 'UNKNOWN_BOOT_ERROR',
});

export const LOG_LEVELS = Object.freeze(['debug', 'info', 'ok', 'warn', 'error']);

const ERROR_CATEGORY_SET = new Set(ERROR_CATEGORIES);
const LOG_LEVEL_SET = new Set(LOG_LEVELS);
const MAX_SANITIZE_DEPTH = 16;
const MAX_BOOT_MILESTONES = 32;
const MAX_BOOT_TRAPS = 32;
const MAX_BOOT_UNRESOLVED_IMPORTS = 64;
const MAX_GPU_MISSING_SHADERS = 64;

const WASM_MODULE_TOO_LARGE_RE =
  /\b(?:exceeds\s+maximum\s+size\s+of\s+1073741824|buffer\s+source\s+exceeds\s+maximum\s+size|(?:wasm\s+)?module\s+(?:size\s+)?exceeds\s+(?:the\s+)?(?:1\s*gib|1073741824|maximum\s+size))\b/i;

const WASM64_UNSUPPORTED_RE =
  /\b(?:wasm64|memory64)\b[^.\n]{0,80}\b(?:unsupported|not\s+supported|not\s+enabled|unavailable|disabled|missing|required|failed)\b|\b(?:unsupported|not\s+supported|does\s+not\s+support|unavailable|disabled|missing|no|lacks?)\b[^.\n]{0,80}\b(?:wasm64|memory64)\b|\b(?:invalid|unsupported|unknown)\s+(?:memory\s+)?(?:address|index)(?:\s+type)?\s*[:=]?\s*['"]?i64['"]?|\bi64\s+memory\s+(?:address|index)\b[^.\n]{0,40}\b(?:unsupported|not\s+supported|disabled|invalid)\b/i;

const COI_SAB_MISSING_RE =
  /\bsharedarraybuffer\b[^.\n]{0,80}\b(?:not\s+defined|undefined|missing|unavailable|disabled|unsupported|not\s+supported|requires?\s+cross[\s-]*origin|cannot\s+be\s+(?:created|used|transferred|cloned))\b|\b(?:missing|no|lacks?|requires?|without|not)\b[^.\n]{0,80}\b(?:sharedarraybuffer|cross[\s-]*origin[\s-]*isolat(?:ed|ion)|coop\s*\/\s*coep)\b|\bcross[\s-]*origin[\s-]*isolat(?:ed|ion)\b[^.\n]{0,80}\b(?:false|missing|required|disabled|unavailable|not\s+enabled|not\s+active)\b|\b(?:coop\s*\/\s*coep|cross-origin-(?:opener|embedder)-policy)\b/i;

const WORKER_SCRIPT_COLLISION_RE =
  /\bhas\s+already\s+been\s+declared\b|\bredeclaration\s+of\s+(?:const|let|var|class|function)\b/i;

const HTTPFS_PGIM_RE = /\b(?:httpfs|pgim)\b|\.pgim\b/i;

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

function extractRawBootErrorText(errorOrMessage) {
  if (errorOrMessage === null || errorOrMessage === undefined) {
    return { rawName: '', rawMessage: '' };
  }
  if (typeof errorOrMessage === 'string') {
    return { rawName: '', rawMessage: errorOrMessage };
  }
  if (typeof errorOrMessage === 'object') {
    const rawName = typeof errorOrMessage.name === 'string' ? errorOrMessage.name : '';
    const rawMessage =
      typeof errorOrMessage.message === 'string'
        ? errorOrMessage.message
        : typeof errorOrMessage.reason === 'string'
          ? errorOrMessage.reason
          : String(errorOrMessage);
    return { rawName, rawMessage };
  }
  return { rawName: '', rawMessage: String(errorOrMessage) };
}

/**
 * Classifies common browser Wasm64 boot failures into a structured diagnostic
 * `{ code, category, hint, message }` with a sanitized message (zero host paths,
 * drive letters, usernames, or auth tokens).
 *
 * @param {unknown} errorOrMessage
 * @returns {{ code: string, category: string, hint: string, message: string }}
 */
export function classifyBootError(errorOrMessage) {
  const { rawName, rawMessage } = extractRawBootErrorText(errorOrMessage);
  const combined = rawName ? `${rawName}: ${rawMessage}` : rawMessage;
  const message = sanitizeString(rawMessage);

  if (WASM_MODULE_TOO_LARGE_RE.test(combined)) {
    return {
      code: BOOT_ERROR_CODES.WASM_MODULE_TOO_LARGE,
      category: 'wasm_compile',
      hint: 'Guest Wasm module exceeds V8 1 GiB compile limit (1073741824 bytes). Rebuild with --opt Oz --no-link-opt --strip to shrink the binary.',
      message,
    };
  }

  if (WASM64_UNSUPPORTED_RE.test(combined)) {
    return {
      code: BOOT_ERROR_CODES.WASM64_UNSUPPORTED,
      category: 'capability',
      hint: 'Browser lacks WebAssembly Memory64 (Wasm64) support. Use Chrome/Edge 133+ or Firefox 134+ with memory64 enabled.',
      message,
    };
  }

  if (COI_SAB_MISSING_RE.test(combined)) {
    return {
      code: BOOT_ERROR_CODES.COI_SAB_MISSING,
      category: 'capability',
      hint: 'SharedArrayBuffer requires cross-origin isolation. Serve with Cross-Origin-Opener-Policy: same-origin and Cross-Origin-Embedder-Policy: require-corp headers.',
      message,
    };
  }

  if (WORKER_SCRIPT_COLLISION_RE.test(combined)) {
    return {
      code: BOOT_ERROR_CODES.WORKER_SCRIPT_COLLISION,
      category: 'worker',
      hint: 'Classic worker importScripts identifier collision detected. Wrap helper scripts in UMD IIFEs so top-level const/let declarations do not collide.',
      message,
    };
  }

  if (HTTPFS_PGIM_RE.test(combined)) {
    return {
      code: BOOT_ERROR_CODES.HTTPFS_PGIM_ERROR,
      category: 'io',
      hint: 'HTTPFS/PGIM image load failed. Verify the local Range-capable HTTP server is running and /game/boot.pgim (or PGIM segments) are staged.',
      message,
    };
  }

  return {
    code: BOOT_ERROR_CODES.UNKNOWN_BOOT_ERROR,
    category: 'runtime',
    hint: 'Inspect the sanitized error message, stack trace, and bootDiagnostics snapshot for the failing boot stage.',
    message,
  };
}

function isPlainObject(value) {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value) && !isBinaryBuffer(value));
}

function clipSanitizedString(value, maxLen = 256) {
  if (typeof value !== 'string') {
    return undefined;
  }
  const safe = sanitizeString(value);
  return safe.length > maxLen ? `${safe.slice(0, maxLen - 3)}...` : safe;
}

function pickNonNegInt(obj, key) {
  const val = obj[key];
  if (typeof val === 'number' && Number.isFinite(val) && val >= 0) {
    return Math.floor(val);
  }
  return undefined;
}

function pickNonNegNumber(obj, key) {
  const val = obj[key];
  if (typeof val === 'number' && Number.isFinite(val) && val >= 0) {
    return val;
  }
  return undefined;
}

function pickPct(obj, key) {
  const val = obj[key];
  if (typeof val === 'number' && Number.isFinite(val)) {
    return Math.min(100, Math.max(0, val));
  }
  return undefined;
}

function sanitizeTrapRecord(raw) {
  if (!isPlainObject(raw)) {
    return null;
  }
  const out = {};
  const seq = pickNonNegInt(raw, 'seq');
  if (seq !== undefined) out.seq = seq;
  const kind = clipSanitizedString(raw.kind, 64);
  if (kind !== undefined) out.kind = kind;
  if (raw.guestVa === null) {
    out.guestVa = null;
  } else {
    const guestVa = clipSanitizedString(raw.guestVa, 32);
    if (guestVa !== undefined) out.guestVa = guestVa;
  }
  if (raw.dll === null) {
    out.dll = null;
  } else {
    const dll = clipSanitizedString(raw.dll, 128);
    if (dll !== undefined) out.dll = dll;
  }
  if (raw.symbol === null) {
    out.symbol = null;
  } else {
    const symbol = clipSanitizedString(raw.symbol, 256);
    if (symbol !== undefined) out.symbol = symbol;
  }
  if (raw.import === null) {
    out.import = null;
  } else {
    const imp = clipSanitizedString(raw.import, 256);
    if (imp !== undefined) out.import = imp;
  }
  const message = clipSanitizedString(raw.message, 512);
  if (message !== undefined) out.message = message;
  if (typeof raw.fatal === 'boolean') out.fatal = raw.fatal;
  const elapsedMs = pickNonNegNumber(raw, 'elapsedMs');
  if (elapsedMs !== undefined) out.elapsedMs = elapsedMs;
  const stage = clipSanitizedString(raw.stage, 64);
  if (stage !== undefined) out.stage = stage;
  const count = pickNonNegInt(raw, 'count');
  if (count !== undefined) out.count = count;
  return out;
}

function sanitizeUnresolvedImportRecord(raw) {
  if (!isPlainObject(raw)) {
    return null;
  }
  const out = {};
  const name = clipSanitizedString(raw.name, 256);
  if (name !== undefined) out.name = name;
  if (raw.dll === null) {
    out.dll = null;
  } else {
    const dll = clipSanitizedString(raw.dll, 128);
    if (dll !== undefined) out.dll = dll;
  }
  if (raw.symbol === null) {
    out.symbol = null;
  } else {
    const symbol = clipSanitizedString(raw.symbol, 256);
    if (symbol !== undefined) out.symbol = symbol;
  }
  if (raw.ordinal === null) {
    out.ordinal = null;
  } else {
    const ordinal = pickNonNegInt(raw, 'ordinal');
    if (ordinal !== undefined) out.ordinal = ordinal;
  }
  if (raw.guestVa === null) {
    out.guestVa = null;
  } else {
    const guestVa = clipSanitizedString(raw.guestVa, 32);
    if (guestVa !== undefined) out.guestVa = guestVa;
  }
  const count = pickNonNegInt(raw, 'count');
  if (count !== undefined) out.count = count;
  const firstElapsedMs = pickNonNegNumber(raw, 'firstElapsedMs');
  if (firstElapsedMs !== undefined) out.firstElapsedMs = firstElapsedMs;
  const lastElapsedMs = pickNonNegNumber(raw, 'lastElapsedMs');
  if (lastElapsedMs !== undefined) out.lastElapsedMs = lastElapsedMs;
  return out;
}

function sanitizeMilestoneRecord(raw) {
  if (!isPlainObject(raw)) {
    return null;
  }
  const out = {};
  const milestone = clipSanitizedString(raw.milestone, 64);
  if (milestone !== undefined) out.milestone = milestone;
  const index = pickNonNegInt(raw, 'index');
  if (index !== undefined) out.index = index;
  if (typeof raw.reached === 'boolean') out.reached = raw.reached;
  if (raw.elapsedMs === null) {
    out.elapsedMs = null;
  } else {
    const elapsedMs = pickNonNegNumber(raw, 'elapsedMs');
    if (elapsedMs !== undefined) out.elapsedMs = elapsedMs;
  }
  if (raw.lastElapsedMs === null) {
    out.lastElapsedMs = null;
  } else {
    const lastElapsedMs = pickNonNegNumber(raw, 'lastElapsedMs');
    if (lastElapsedMs !== undefined) out.lastElapsedMs = lastElapsedMs;
  }
  const count = pickNonNegInt(raw, 'count');
  if (count !== undefined) out.count = count;
  if (raw.detail === null) {
    out.detail = null;
  } else if (isPlainObject(raw.detail)) {
    const detail = {};
    const source = clipSanitizedString(raw.detail.source, 64);
    if (source !== undefined) detail.source = source;
    const line = clipSanitizedString(raw.detail.line, 256);
    if (line !== undefined) detail.line = line;
    if (Object.keys(detail).length > 0) {
      out.detail = detail;
    }
  }
  return out;
}

/**
 * Strictly allowlists and sanitizes a boot summary snapshot (or BootTracer instance).
 * Drops unknown keys, binary buffers, and redacts any host paths or secrets in strings.
 *
 * @param {unknown} raw
 * @returns {object|null}
 */
function sanitizeBootSnapshot(raw) {
  if (raw === null || raw === undefined) {
    return null;
  }
  let source = raw;
  if (typeof source === 'object' && typeof source.getSummary === 'function') {
    try {
      source = source.getSummary();
    } catch {
      return null;
    }
  }
  if (!isPlainObject(source)) {
    return null;
  }

  const out = {};

  const stage = clipSanitizedString(source.stage, 64);
  if (stage !== undefined) out.stage = stage;
  const mode = clipSanitizedString(source.mode, 64);
  if (mode !== undefined) out.mode = mode;
  const bootMode = clipSanitizedString(source.bootMode, 64);
  if (bootMode !== undefined) out.bootMode = bootMode;
  const status = clipSanitizedString(source.status, 64);
  if (status !== undefined) out.status = status;

  const progressPct = pickPct(source, 'progressPct');
  if (progressPct !== undefined) out.progressPct = progressPct;
  if (typeof source.readyForSplash === 'boolean') out.readyForSplash = source.readyForSplash;

  const elapsedMs = pickNonNegNumber(source, 'elapsedMs');
  if (elapsedMs !== undefined) out.elapsedMs = elapsedMs;
  const milestonesReached = pickNonNegInt(source, 'milestonesReached');
  if (milestonesReached !== undefined) out.milestonesReached = milestonesReached;

  if (Array.isArray(source.milestones)) {
    const milestones = [];
    for (let i = 0; i < source.milestones.length && milestones.length < MAX_BOOT_MILESTONES; i += 1) {
      const clean = sanitizeMilestoneRecord(source.milestones[i]);
      if (clean && Object.keys(clean).length > 0) {
        milestones.push(clean);
      }
    }
    out.milestones = milestones;
  }

  const trapCount = pickNonNegInt(source, 'trapCount');
  if (trapCount !== undefined) out.trapCount = trapCount;
  const uniqueTrapCount = pickNonNegInt(source, 'uniqueTrapCount');
  if (uniqueTrapCount !== undefined) out.uniqueTrapCount = uniqueTrapCount;

  if (source.lastTrap === null) {
    out.lastTrap = null;
  } else if (source.lastTrap !== undefined) {
    const cleanTrap = sanitizeTrapRecord(source.lastTrap);
    if (cleanTrap) {
      out.lastTrap = cleanTrap;
    }
  }

  if (Array.isArray(source.unresolvedImports)) {
    const unresolvedImports = [];
    for (
      let i = 0;
      i < source.unresolvedImports.length && unresolvedImports.length < MAX_BOOT_UNRESOLVED_IMPORTS;
      i += 1
    ) {
      const clean = sanitizeUnresolvedImportRecord(source.unresolvedImports[i]);
      if (clean && Object.keys(clean).length > 0) {
        unresolvedImports.push(clean);
      }
    }
    out.unresolvedImports = unresolvedImports;
  }

  if (Array.isArray(source.recentTraps)) {
    const recentTraps = [];
    for (let i = 0; i < source.recentTraps.length && recentTraps.length < MAX_BOOT_TRAPS; i += 1) {
      const clean = sanitizeTrapRecord(source.recentTraps[i]);
      if (clean && Object.keys(clean).length > 0) {
        recentTraps.push(clean);
      }
    }
    out.recentTraps = recentTraps;
  }

  const wasmModuleBytes = pickNonNegNumber(source, 'wasmModuleBytes');
  if (wasmModuleBytes !== undefined) out.wasmModuleBytes = wasmModuleBytes;
  const wasmSizeBytes = pickNonNegNumber(source, 'wasmSizeBytes');
  if (wasmSizeBytes !== undefined) out.wasmSizeBytes = wasmSizeBytes;
  const pgimBytes = pickNonNegNumber(source, 'pgimBytes');
  if (pgimBytes !== undefined) out.pgimBytes = pgimBytes;
  const pgimSegments = pickNonNegInt(source, 'pgimSegments');
  if (pgimSegments !== undefined) out.pgimSegments = pgimSegments;
  const shaderPackCount = pickNonNegInt(source, 'shaderPackCount');
  if (shaderPackCount !== undefined) out.shaderPackCount = shaderPackCount;

  if (typeof source.updatedAt === 'string') {
    const updatedAt = clipSanitizedString(source.updatedAt, 64);
    if (updatedAt !== undefined) out.updatedAt = updatedAt;
  } else {
    const updatedAtNum = pickNonNegNumber(source, 'updatedAt');
    if (updatedAtNum !== undefined) out.updatedAt = updatedAtNum;
  }

  return out;
}

function sanitizeMissingShaderEntry(raw) {
  if (!isPlainObject(raw)) {
    return null;
  }
  let hash;
  if (typeof raw.hash === 'bigint' && raw.hash >= 0n) {
    hash = `0x${raw.hash.toString(16).padStart(16, '0')}`;
  } else if (typeof raw.hash === 'string') {
    hash = clipSanitizedString(raw.hash, 64);
  } else if (typeof raw.hashHex === 'string') {
    hash = clipSanitizedString(raw.hashHex, 64);
  }
  const stage = clipSanitizedString(raw.stage, 32);
  const hits = pickNonNegInt(raw, 'hits') ?? pickNonNegInt(raw, 'count');
  const firstSeenFrame = pickNonNegInt(raw, 'firstSeenFrame');

  if (hash === undefined && stage === undefined && hits === undefined) {
    return null;
  }
  const out = {};
  if (hash !== undefined) out.hash = hash;
  if (stage !== undefined) out.stage = stage;
  if (hits !== undefined) out.hits = hits;
  if (firstSeenFrame !== undefined) out.firstSeenFrame = firstSeenFrame;
  return out;
}

/**
 * Strictly allowlists and sanitizes a GPU telemetry snapshot.
 * Drops unknown keys, shader sources, and raw binary buffers; redacts host paths/tokens.
 *
 * @param {unknown} raw
 * @returns {object|null}
 */
function sanitizeGpuTelemetry(raw) {
  if (raw === null || raw === undefined) {
    return null;
  }
  if (!isPlainObject(raw)) {
    return null;
  }

  const out = {};

  const timestamp = pickNonNegNumber(raw, 'timestamp');
  if (timestamp !== undefined) out.timestamp = timestamp;
  const adapter = clipSanitizedString(raw.adapter, 128);
  if (adapter !== undefined) out.adapter = adapter;
  const backend = clipSanitizedString(raw.backend, 64);
  if (backend !== undefined) out.backend = backend;
  if (typeof raw.deviceLost === 'boolean') out.deviceLost = raw.deviceLost;

  for (const intKey of [
    'validationErrors',
    'shaderPackCount',
    'loadedShaders',
    'pipelineCount',
    'bufferCount',
    'textureCount',
    'framesPresented',
    'droppedFrames',
    'drawCalls',
    'computeDispatches',
    'packetsProcessed',
  ]) {
    const val = pickNonNegInt(raw, intKey);
    if (val !== undefined) out[intKey] = val;
  }

  if (isPlainObject(raw.frameStats)) {
    const fs = {};
    for (const intKey of [
      'framesPresented',
      'droppedFrames',
      'drawCalls',
      'computeDispatches',
      'packetsProcessed',
      'ringBytesDrained',
      'width',
      'height',
    ]) {
      const val = pickNonNegInt(raw.frameStats, intKey);
      if (val !== undefined) fs[intKey] = val;
    }
    for (const numKey of ['lastFrameMs', 'avgFrameMs']) {
      const val = pickNonNegNumber(raw.frameStats, numKey);
      if (val !== undefined) fs[numKey] = val;
    }
    out.frameStats = fs;
  }

  const rawFallback = isPlainObject(raw.fallbackStats)
    ? raw.fallbackStats
    : isPlainObject(raw.shaderFallbackStats)
      ? raw.shaderFallbackStats
      : null;
  if (rawFallback) {
    const fb = {};
    for (const intKey of [
      'packHits',
      'fallbackHits',
      'builtinHits',
      'totalLookups',
      'uniqueMissing',
      'overflowMissingCount',
    ]) {
      const val = pickNonNegInt(rawFallback, intKey);
      if (val !== undefined) fb[intKey] = val;
    }
    const hitRatePct = pickPct(rawFallback, 'hitRatePct');
    if (hitRatePct !== undefined) fb.hitRatePct = hitRatePct;
    out.fallbackStats = fb;
  }

  if (Array.isArray(raw.missingShaders)) {
    const missingShaders = [];
    for (
      let i = 0;
      i < raw.missingShaders.length && missingShaders.length < MAX_GPU_MISSING_SHADERS;
      i += 1
    ) {
      const clean = sanitizeMissingShaderEntry(raw.missingShaders[i]);
      if (clean) {
        missingShaders.push(clean);
      }
    }
    out.missingShaders = missingShaders;
  }

  return out;
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

function makeErrorCoalesceKey(category, message) {
  return `${category}\u0000${message}`;
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
    this.pushWithEvicted(item);
    return item;
  }

  pushWithEvicted(item) {
    const evicted = this._count === this._capacity ? this._items[this._writeIdx] : undefined;
    this._items[this._writeIdx] = item;
    this._writeIdx = (this._writeIdx + 1) % this._capacity;
    if (this._count < this._capacity) {
      this._count += 1;
    }
    this._totalRecorded += 1;
    return evicted;
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
   * @param {boolean} [options.coalesceDuplicates=true]
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
      coalesceDuplicates = true,
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
    if (typeof coalesceDuplicates !== 'boolean') {
      throw new TypeError('coalesceDuplicates must be a boolean');
    }

    this._maxLogs = maxLogs;
    this._maxErrors = maxErrors;
    this._appVersion = sanitizeString(appVersion || DEFAULT_APP_VERSION);
    this._now = now;
    this._coalesceDuplicates = coalesceDuplicates;
    this._navigatorObj = navigatorObj;
    this._globalObj = globalObj;

    this._logs = new BoundedRing(maxLogs);
    this._errors = new BoundedRing(maxErrors);
    this._errorIndex = new Map();
    this._recordedErrorCount = 0;
    this._coalescedErrorCount = 0;
    this._categoryCounts = Object.create(null);
    this._errorClassifications = new WeakMap();
    this._bootSnapshot = null;
    this._gpuTelemetry = null;

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

  get recordedErrorCount() {
    return this._recordedErrorCount;
  }

  get coalescedCount() {
    return this._coalescedErrorCount;
  }

  get categoryCounts() {
    return { ...this._categoryCounts };
  }

  get isGlobalHandlersAttached() {
    return this._attachedTarget !== null;
  }

  /**
   * Returns additive, sanitized error recording and coalescing statistics.
   *
   * @returns {{
   *   recordedCount: number,
   *   bufferedCount: number,
   *   coalescedCount: number,
   *   droppedCount: number,
   *   maxErrors: number,
   *   categoryCounts: Record<string, number>
   * }}
   */
  getErrorStats() {
    return {
      recordedCount: this._recordedErrorCount,
      bufferedCount: this._errors.length,
      coalescedCount: this._coalescedErrorCount,
      droppedCount: this._errors.droppedCount,
      maxErrors: this._maxErrors,
      categoryCounts: { ...this._categoryCounts },
    };
  }

  /**
   * Clears all buffered logs, errors, and attached boot/GPU snapshots without detaching global handlers.
   */
  clear() {
    this._logs.clear();
    this._errors.clear();
    this._errorIndex.clear();
    this._recordedErrorCount = 0;
    this._coalescedErrorCount = 0;
    this._categoryCounts = Object.create(null);
    this._bootSnapshot = null;
    this._gpuTelemetry = null;
  }

  /**
   * Stores an allowlisted, sanitized boot summary snapshot (or `BootTracer` instance)
   * to include under `bootDiagnostics.bootSnapshot` in exported diagnostic bundles.
   * Pass `null` or `undefined` to clear.
   *
   * @param {object|null|undefined} summary
   * @returns {object|null}
   */
  setBootSnapshot(summary) {
    this._bootSnapshot = sanitizeBootSnapshot(summary);
    return this.getBootSnapshot();
  }

  /**
   * Returns a sanitized clone of the attached boot snapshot, or `null` when unset.
   *
   * @returns {object|null}
   */
  getBootSnapshot() {
    return this._bootSnapshot ? sanitizeValue(this._bootSnapshot) : null;
  }

  /**
   * Stores an allowlisted, sanitized GPU telemetry snapshot to include under
   * `bootDiagnostics.gpuTelemetry` in exported diagnostic bundles.
   * Pass `null` or `undefined` to clear.
   *
   * @param {object|null|undefined} snapshot
   * @returns {object|null}
   */
  setGpuTelemetry(snapshot) {
    this._gpuTelemetry = sanitizeGpuTelemetry(snapshot);
    return this.getGpuTelemetry();
  }

  /**
   * Returns a sanitized clone of the attached GPU telemetry snapshot, or `null` when unset.
   *
   * @returns {object|null}
   */
  getGpuTelemetry() {
    return this._gpuTelemetry ? sanitizeValue(this._gpuTelemetry) : null;
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
   * When duplicate coalescing is enabled (default), repeated identical
   * `(category, message)` errors increment `occurrences` and update `lastSeenAtMs`
   * on the existing buffered entry instead of evicting earlier root-cause errors.
   *
   * @param {string|null|undefined} category
   * @param {unknown} errorOrMessage
   * @param {unknown} [context]
   * @returns {object}
   */
  recordError(category, errorOrMessage, context) {
    const ts = this._now();
    const safeTs = Number.isFinite(ts) ? ts : 0;
    const resolvedCategory = classifyError(errorOrMessage, category);
    const fields = extractErrorFields(errorOrMessage);

    this._recordedErrorCount += 1;
    this._categoryCounts[resolvedCategory] = (this._categoryCounts[resolvedCategory] || 0) + 1;

    const coalesceKey = makeErrorCoalesceKey(resolvedCategory, fields.message);
    if (this._coalesceDuplicates) {
      const existing = this._errorIndex.get(coalesceKey);
      if (existing) {
        existing.occurrences += 1;
        existing.lastSeenAtMs = Math.max(existing.lastSeenAtMs, safeTs);
        this._coalescedErrorCount += 1;
        return { ...existing };
      }
    }

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
      timestampMs: safeTs,
      firstSeenAtMs: safeTs,
      lastSeenAtMs: safeTs,
      occurrences: 1,
      category: resolvedCategory,
      name: fields.name,
      message: fields.message,
      stack: fields.stack,
      context: sanitizedContext,
    };

    this._errorClassifications.set(entry, classifyBootError(errorOrMessage));
    const evicted = this._errors.pushWithEvicted(entry);
    if (evicted !== undefined) {
      const evictedKey = makeErrorCoalesceKey(evicted.category, evicted.message);
      if (this._errorIndex.get(evictedKey) === evicted) {
        this._errorIndex.delete(evictedKey);
      }
    }
    if (this._coalesceDuplicates) {
      this._errorIndex.set(coalesceKey, entry);
    }
    return { ...entry };
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
    return this._errors.toArray().map((entry) => ({ ...entry }));
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
   * @param {object} [options.bootSnapshot] Optional per-report boot summary override.
   * @param {object} [options.bootSummary] Alias for `bootSnapshot`.
   * @param {object} [options.gpuTelemetry] Optional per-report GPU telemetry override.
   * @param {unknown} [options.extra]
   * @returns {object}
   */
  buildReport({
    perfSnapshot,
    storageSummary,
    serverStatus,
    bootSnapshot,
    bootSummary,
    gpuTelemetry,
    extra,
  } = {}) {
    const ts = this._now();

    let resolvedPerf = null;
    if (perfSnapshot && typeof perfSnapshot === 'object') {
      if (typeof perfSnapshot.getSnapshot === 'function') {
        resolvedPerf = perfSnapshot.getSnapshot();
      } else {
        resolvedPerf = perfSnapshot;
      }
    }

    const recentErrors = this._errors.toArray();
    const classifiedErrors = [];
    const errorCodeCounts = {};
    for (let i = 0; i < recentErrors.length; i += 1) {
      const errEntry = recentErrors[i];
      const classified = this._errorClassifications.get(errEntry) || classifyBootError(errEntry);
      const occurrences = errEntry.occurrences || 1;
      classifiedErrors.push({
        timestamp: errEntry.timestamp,
        timestampMs: errEntry.timestampMs,
        firstSeenAtMs: errEntry.firstSeenAtMs,
        lastSeenAtMs: errEntry.lastSeenAtMs,
        occurrences,
        errorCategory: errEntry.category,
        code: classified.code,
        category: classified.category,
        hint: classified.hint,
        message: classified.message,
      });
      errorCodeCounts[classified.code] = (errorCodeCounts[classified.code] || 0) + occurrences;
    }

    const rawBootOverride = bootSnapshot !== undefined ? bootSnapshot : bootSummary;
    const resolvedBootSnapshot =
      rawBootOverride !== undefined ? sanitizeBootSnapshot(rawBootOverride) : this._bootSnapshot;
    const resolvedGpuTelemetry =
      gpuTelemetry !== undefined ? sanitizeGpuTelemetry(gpuTelemetry) : this._gpuTelemetry;

    const report = {
      schemaVersion: SCHEMA_VERSION,
      generatedAt: formatIsoTimestamp(ts),
      appVersion: this._appVersion,
      capabilities: this.collectCapabilities(),
      errors: recentErrors,
      errorStats: this.getErrorStats(),
      recentLogs: this._logs.toArray(),
      perf: resolvedPerf,
      storage: storageSummary !== undefined ? storageSummary : null,
      serverStatus: serverStatus !== undefined ? serverStatus : null,
      bootDiagnostics: {
        classifiedErrors,
        errorCodeCounts,
        bootSnapshot: resolvedBootSnapshot,
        gpuTelemetry: resolvedGpuTelemetry,
      },
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
