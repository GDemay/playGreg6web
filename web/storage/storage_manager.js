// Browser storage, quota, OPFS range-cache, and IndexedDB /userdata save manager.
// Matches OPFS directory ('playgta6-cache') from web/workers/range_cache.js and
// IndexedDB database/store ('playgta6-userdata' / 'files') + /userdata/ path rules
// from web/workers/userdata.js. Zero external dependencies; browser & Node testable.

export const DEFAULT_OPFS_CACHE_DIR = 'playgta6-cache';
export const DEFAULT_CACHE_INDEX_FILENAME = '.playgta6-cache-index.json';
export const CACHE_INDEX_VERSION = 1;
export const DEFAULT_USERDATA_DB_NAME = 'playgta6-userdata';
export const DEFAULT_USERDATA_STORE_NAME = 'files';
export const DEFAULT_USERDATA_DB_VERSION = 1;
export const USERDATA_PREFIX = '/userdata/';
export const SNAPSHOT_FORMAT = 'playgta6-userdata-snapshot';
export const SNAPSHOT_VERSION = 1;
export const DEFAULT_WARN_THRESHOLD_RATIO = 0.75;
export const DEFAULT_CRITICAL_THRESHOLD_RATIO = 0.9;
export const DEFAULT_QUOTA_BUDGET_RATIO = 0.6;
export const DEFAULT_WARM_BOOT_MIN_HEADROOM_RATIO = 0.1;
export const DEFAULT_WARM_BOOT_MIN_RPF_CHUNKS = 1;
export const DEFAULT_FAST_BOOT_SLAB_MAX_BYTES = 192 * 1024 * 1024; // 192 MiB healthy slab cap (#1077 / W255)
export const MIN_FAST_BOOT_SLAB_MAX_BYTES = 32 * 1024 * 1024; // 32 MiB floor under critical pressure
export const DEFAULT_FAST_BOOT_OPFS_BUDGET_BYTES = 512 * 1024 * 1024; // 512 MiB default OPFS cache cap
export const DEFAULT_FAST_BOOT_MEDIUM_THRESHOLD_RATIO = 0.5;
export const FAST_BOOT_PRESSURE_LEVELS = Object.freeze({
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high',
  CRITICAL: 'critical',
});
export const QUOTA_PRESSURE_LEVELS = Object.freeze({
  OK: 'ok',
  WARN: 'warn',
  CRITICAL: 'critical',
  UNKNOWN: 'unknown',
});
export const CACHE_ENTRY_KINDS = Object.freeze({
  SHADER_PACK: 'shader-pack',
  PGIM: 'pgim',
  RANGE_CHUNK: 'range-chunk',
  GENERIC: 'generic',
});
export const WARM_BOOT_RECOMMENDATIONS = Object.freeze({
  READY: 'ready',
  OPFS_UNSUPPORTED: 'opfs-unsupported',
  LOW_QUOTA_HEADROOM: 'low-quota-headroom',
  REQUEST_PERSISTENCE: 'request-persistence',
  WARM_CACHE_EMPTY: 'warm-cache-empty',
  CACHE_SHADER_PACK: 'cache-shader-pack',
  CACHE_RPF_CHUNKS: 'cache-rpf-chunks',
});

const BYTE_UNITS = Object.freeze(['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']);
const BASE64_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const BASE64_LOOKUP = (() => {
  const table = new Int16Array(128).fill(-1);
  for (let i = 0; i < BASE64_ALPHABET.length; i += 1) {
    table[BASE64_ALPHABET.charCodeAt(i)] = i;
  }
  return table;
})();

/**
 * Formats a byte count into a human-readable IEC string (B, KiB, MiB, GiB, ...).
 * Non-finite or negative values safely format as '0 B'.
 *
 * @param {number} bytes
 * @param {number} [decimals=2]
 * @returns {string}
 */
export function formatBytes(bytes, decimals = 2) {
  if (typeof bytes !== 'number' || !Number.isFinite(bytes) || bytes <= 0) {
    return '0 B';
  }
  const precision = Number.isInteger(decimals) && decimals >= 0 ? decimals : 2;
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < BYTE_UNITS.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  if (unitIndex === 0) {
    return `${Math.round(value)} B`;
  }
  const formatted = Number(value.toFixed(precision)).toString();
  return `${formatted} ${BYTE_UNITS[unitIndex]}`;
}

/**
 * Validates that a save path stays strictly confined within /userdata/.
 * Mirrors validateUserdataPath() in web/workers/userdata.js exactly.
 *
 * @param {string} path
 * @returns {string}
 */
export function validateUserdataPath(path) {
  if (typeof path !== 'string' || path.length === 0) {
    throw new Error('Userdata path must be a non-empty string');
  }
  if (path.includes('\\') || path.includes('\0')) {
    throw new Error(`Invalid userdata path (forbidden character): ${path}`);
  }
  if (!path.startsWith(USERDATA_PREFIX)) {
    throw new Error(`Userdata path must start with ${USERDATA_PREFIX}: ${path}`);
  }
  if (path.length === USERDATA_PREFIX.length || path.endsWith('/')) {
    throw new Error(`Userdata path must reference a file inside ${USERDATA_PREFIX}: ${path}`);
  }

  const rel = path.slice(USERDATA_PREFIX.length);
  const segments = rel.split('/');
  for (const seg of segments) {
    if (!seg || seg === '.' || seg === '..') {
      throw new Error(`Invalid userdata path segment in: ${path}`);
    }
  }
  return path;
}

function toUint8Array(data) {
  if (data instanceof Uint8Array) {
    return data.slice();
  }
  if (ArrayBuffer.isView(data)) {
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
  }
  if (data instanceof ArrayBuffer) {
    return new Uint8Array(data.slice(0));
  }
  if (typeof data === 'string') {
    return new TextEncoder().encode(data);
  }
  throw new TypeError('Userdata payload must be a Uint8Array, ArrayBufferView, ArrayBuffer, or string');
}

/**
 * Pure RFC 4648 base64 encoder safe for arbitrarily large Uint8Arrays in
 * browsers, Web Workers, and Node without relying on Node Buffer.
 *
 * @param {Uint8Array|ArrayBufferView|ArrayBuffer} input
 * @returns {string}
 */
export function encodeBase64(input) {
  const bytes = toUint8Array(input);
  const len = bytes.byteLength;
  if (len === 0) {
    return '';
  }

  const chunks = [];
  let i = 0;
  for (; i + 2 < len; i += 3) {
    const triplet = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2];
    chunks.push(
      BASE64_ALPHABET[(triplet >>> 18) & 0x3f] +
        BASE64_ALPHABET[(triplet >>> 12) & 0x3f] +
        BASE64_ALPHABET[(triplet >>> 6) & 0x3f] +
        BASE64_ALPHABET[triplet & 0x3f]
    );
  }

  const rem = len - i;
  if (rem === 1) {
    const b0 = bytes[i];
    chunks.push(
      BASE64_ALPHABET[(b0 >>> 2) & 0x3f] +
        BASE64_ALPHABET[(b0 << 4) & 0x3f] +
        '=='
    );
  } else if (rem === 2) {
    const b0 = bytes[i];
    const b1 = bytes[i + 1];
    chunks.push(
      BASE64_ALPHABET[(b0 >>> 2) & 0x3f] +
        BASE64_ALPHABET[((b0 << 4) | (b1 >>> 4)) & 0x3f] +
        BASE64_ALPHABET[(b1 << 2) & 0x3f] +
        '='
    );
  }

  return chunks.join('');
}

/**
 * Pure RFC 4648 base64 decoder with strict character and padding validation.
 *
 * @param {string} base64
 * @returns {Uint8Array}
 */
export function decodeBase64(base64) {
  if (typeof base64 !== 'string') {
    throw new TypeError('Base64 payload must be a string');
  }
  const clean = base64.replace(/[\r\n\t ]+/g, '');
  if (clean.length === 0) {
    return new Uint8Array(0);
  }
  if (clean.length % 4 !== 0) {
    throw new Error('Invalid base64 payload length');
  }

  let pad = 0;
  if (clean.endsWith('==')) {
    pad = 2;
  } else if (clean.endsWith('=')) {
    pad = 1;
  }

  const outLen = (clean.length / 4) * 3 - pad;
  const out = new Uint8Array(outLen);
  let outPos = 0;

  for (let i = 0; i < clean.length; i += 4) {
    const isLast = i + 4 === clean.length;
    const c0 = clean.charCodeAt(i);
    const c1 = clean.charCodeAt(i + 1);
    const c2 = clean.charCodeAt(i + 2);
    const c3 = clean.charCodeAt(i + 3);

    const v0 = c0 < 128 ? BASE64_LOOKUP[c0] : -1;
    const v1 = c1 < 128 ? BASE64_LOOKUP[c1] : -1;
    const v2 = isLast && pad === 2 && c2 === 61 ? 0 : c2 < 128 ? BASE64_LOOKUP[c2] : -1;
    const v3 = isLast && pad >= 1 && c3 === 61 ? 0 : c3 < 128 ? BASE64_LOOKUP[c3] : -1;

    if (v0 < 0 || v1 < 0 || v2 < 0 || v3 < 0) {
      throw new Error('Invalid base64 character in snapshot payload');
    }
    if (!isLast && (c2 === 61 || c3 === 61)) {
      throw new Error('Unexpected base64 padding before end of payload');
    }

    const triplet = (v0 << 18) | (v1 << 12) | (v2 << 6) | v3;
    out[outPos] = (triplet >>> 16) & 0xff;
    outPos += 1;
    if (outPos < outLen) {
      out[outPos] = (triplet >>> 8) & 0xff;
      outPos += 1;
    }
    if (outPos < outLen) {
      out[outPos] = triplet & 0xff;
      outPos += 1;
    }
  }

  return out;
}

function compareAscii(a, b) {
  return a < b ? -1 : a > b ? 1 : 0;
}

function clampUnitRatio(value, fallback) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return fallback;
  }
  if (value <= 0) {
    return 0;
  }
  if (value >= 1) {
    return 1;
  }
  return value;
}

function sanitizeNonNegativeInt(value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    return 0;
  }
  return Math.floor(value);
}

function sanitizeTimestampMs(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return null;
  }
  return Math.max(0, Math.floor(value));
}

function normalizeQuotaThresholds(options = {}) {
  const opts = options && typeof options === 'object' ? options : {};
  const hasExplicitCritical =
    typeof opts.criticalThresholdRatio === 'number' && Number.isFinite(opts.criticalThresholdRatio);
  let warnThresholdRatio = clampUnitRatio(opts.warnThresholdRatio, DEFAULT_WARN_THRESHOLD_RATIO);
  let criticalThresholdRatio = clampUnitRatio(
    opts.criticalThresholdRatio,
    DEFAULT_CRITICAL_THRESHOLD_RATIO
  );

  if (warnThresholdRatio > criticalThresholdRatio) {
    if (!hasExplicitCritical) {
      criticalThresholdRatio = warnThresholdRatio;
    } else {
      warnThresholdRatio = criticalThresholdRatio;
    }
  }

  return {
    warnThresholdRatio,
    criticalThresholdRatio,
  };
}

/**
 * Classifies browser storage quota pressure into 'ok', 'warn', 'critical', or 'unknown'.
 *
 * Accepts either:
 * - A quota estimate object (`{ supported?, usageBytes, quotaBytes }`, `{ usage, quota }`,
 *   or `{ supported?, usageRatio }`) and optional threshold overrides, or
 * - Positional numeric arguments `(usageBytes, quotaBytes, options?)`.
 *
 * Non-finite, non-positive quota, or unsupported inputs safely resolve to 'unknown'.
 * Negative usage values clamp to 0 and over-quota ratios clamp to 1.
 *
 * @param {object|number|null|undefined} estimateOrUsage
 * @param {number|object} [quotaOrOptions]
 * @param {object} [maybeOptions]
 * @returns {'ok'|'warn'|'critical'|'unknown'}
 */
export function classifyQuotaPressure(estimateOrUsage, quotaOrOptions, maybeOptions) {
  if (estimateOrUsage === null || estimateOrUsage === undefined) {
    return QUOTA_PRESSURE_LEVELS.UNKNOWN;
  }

  let ratio = null;
  let thresholdSource = null;

  if (typeof estimateOrUsage === 'object') {
    if (estimateOrUsage.supported === false) {
      return QUOTA_PRESSURE_LEVELS.UNKNOWN;
    }

    const usage =
      estimateOrUsage.usageBytes !== undefined ? estimateOrUsage.usageBytes : estimateOrUsage.usage;
    const quota =
      estimateOrUsage.quotaBytes !== undefined ? estimateOrUsage.quotaBytes : estimateOrUsage.quota;

    if (usage === undefined && quota === undefined && typeof estimateOrUsage.usageRatio === 'number') {
      if (!Number.isFinite(estimateOrUsage.usageRatio)) {
        return QUOTA_PRESSURE_LEVELS.UNKNOWN;
      }
      ratio = clampUnitRatio(estimateOrUsage.usageRatio, 0);
    } else {
      if (
        typeof usage !== 'number' ||
        !Number.isFinite(usage) ||
        typeof quota !== 'number' ||
        !Number.isFinite(quota) ||
        quota <= 0
      ) {
        return QUOTA_PRESSURE_LEVELS.UNKNOWN;
      }
      const clampedUsage = Math.max(0, usage);
      ratio = clampUnitRatio(clampedUsage / quota, 0);
    }

    thresholdSource = {};
    if (estimateOrUsage.warnThresholdRatio !== undefined) {
      thresholdSource.warnThresholdRatio = estimateOrUsage.warnThresholdRatio;
    }
    if (estimateOrUsage.criticalThresholdRatio !== undefined) {
      thresholdSource.criticalThresholdRatio = estimateOrUsage.criticalThresholdRatio;
    }
    if (quotaOrOptions && typeof quotaOrOptions === 'object') {
      if (quotaOrOptions.warnThresholdRatio !== undefined) {
        thresholdSource.warnThresholdRatio = quotaOrOptions.warnThresholdRatio;
      }
      if (quotaOrOptions.criticalThresholdRatio !== undefined) {
        thresholdSource.criticalThresholdRatio = quotaOrOptions.criticalThresholdRatio;
      }
    }
  } else {
    if (
      typeof estimateOrUsage !== 'number' ||
      !Number.isFinite(estimateOrUsage) ||
      typeof quotaOrOptions !== 'number' ||
      !Number.isFinite(quotaOrOptions) ||
      quotaOrOptions <= 0
    ) {
      return QUOTA_PRESSURE_LEVELS.UNKNOWN;
    }
    const clampedUsage = Math.max(0, estimateOrUsage);
    ratio = clampUnitRatio(clampedUsage / quotaOrOptions, 0);
    thresholdSource = maybeOptions && typeof maybeOptions === 'object' ? maybeOptions : {};
  }

  const { warnThresholdRatio, criticalThresholdRatio } = normalizeQuotaThresholds(thresholdSource);
  if (ratio >= criticalThresholdRatio) {
    return QUOTA_PRESSURE_LEVELS.CRITICAL;
  }
  if (ratio >= warnThresholdRatio) {
    return QUOTA_PRESSURE_LEVELS.WARN;
  }
  return QUOTA_PRESSURE_LEVELS.OK;
}

function isNotFoundError(err) {
  if (!err) return false;
  return err.name === 'NotFoundError' || err.code === 8;
}

function isQuotaExceededError(err) {
  if (!err) return false;
  return (
    err.name === 'QuotaExceededError' ||
    err.code === 22 ||
    err.code === 1014 ||
    /quota/i.test(String(err.message || ''))
  );
}

/**
 * Validates that an OPFS cache entry key is a clean, relative, traversal-free path
 * inside the cache directory and does not collide with the internal metadata index file.
 *
 * @param {string} key
 * @param {string} [indexFileName=DEFAULT_CACHE_INDEX_FILENAME]
 * @returns {string}
 */
export function validateCacheKey(key, indexFileName = DEFAULT_CACHE_INDEX_FILENAME) {
  if (typeof key !== 'string' || key.length === 0) {
    throw new Error('Cache key must be a non-empty string');
  }
  if (key.includes('\\') || key.includes('\0')) {
    throw new Error(`Invalid cache key (forbidden character): ${key}`);
  }
  if (key.startsWith('/') || key.endsWith('/')) {
    throw new Error(`Cache key must be a relative file path without leading/trailing slash: ${key}`);
  }
  const segments = key.split('/');
  for (const seg of segments) {
    if (!seg || seg === '.' || seg === '..') {
      throw new Error(`Invalid cache key segment in: ${key}`);
    }
  }
  if (key === indexFileName || key === DEFAULT_CACHE_INDEX_FILENAME) {
    throw new Error(`Cache key "${key}" is reserved for the internal cache metadata index`);
  }
  return key;
}

/**
 * Infers a coarse cache entry kind ('pgim', 'shader-pack', 'range-chunk', or 'generic')
 * from a cache key.
 *
 * @param {string} key
 * @returns {'pgim'|'shader-pack'|'range-chunk'|'generic'}
 */
export function inferCacheEntryKind(key) {
  if (typeof key !== 'string' || key.length === 0) {
    return CACHE_ENTRY_KINDS.GENERIC;
  }
  const lower = key.toLowerCase();
  if (lower.endsWith('.pgim')) {
    return CACHE_ENTRY_KINDS.PGIM;
  }
  if (
    lower.endsWith('.wgsl.pack') ||
    lower.endsWith('.wgsl') ||
    lower.endsWith('.spv') ||
    lower.endsWith('.spirv') ||
    lower.endsWith('.shaderpack') ||
    lower.startsWith('shaders/') ||
    lower.includes('/shaders/')
  ) {
    return CACHE_ENTRY_KINDS.SHADER_PACK;
  }
  const leaf = key.includes('/') ? key.slice(key.lastIndexOf('/') + 1) : key;
  if (/^c_\d+_\d+_.*\.bin$/.test(leaf)) {
    return CACHE_ENTRY_KINDS.RANGE_CHUNK;
  }
  return CACHE_ENTRY_KINDS.GENERIC;
}

function normalizeOptionalTag(value) {
  if (typeof value !== 'string') {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function normalizeCacheKind(kind, key) {
  if (typeof kind === 'string') {
    const trimmed = kind.trim();
    if (
      trimmed === CACHE_ENTRY_KINDS.PGIM ||
      trimmed === CACHE_ENTRY_KINDS.SHADER_PACK ||
      trimmed === CACHE_ENTRY_KINDS.RANGE_CHUNK ||
      trimmed === CACHE_ENTRY_KINDS.GENERIC
    ) {
      return trimmed;
    }
  }
  return inferCacheEntryKind(key);
}

function cloneCacheEntryMeta(entry) {
  if (!entry) return null;
  return {
    key: entry.key,
    sizeBytes: entry.sizeBytes,
    lastAccessedMs: entry.lastAccessedMs,
    createdAtMs: entry.createdAtMs,
    etag: entry.etag,
    versionKey: entry.versionKey,
    pinned: entry.pinned,
    kind: entry.kind,
  };
}

async function readFileHandleBytes(handle) {
  if (!handle || typeof handle.getFile !== 'function') {
    return null;
  }
  const file = await handle.getFile();
  if (!file) {
    return null;
  }
  if (typeof file.arrayBuffer === 'function') {
    const buf = await file.arrayBuffer();
    return new Uint8Array(buf);
  }
  if (typeof file.text === 'function') {
    return new TextEncoder().encode(await file.text());
  }
  return new Uint8Array(0);
}

async function readFileHandleMeta(handle) {
  if (!handle || typeof handle.getFile !== 'function') {
    return { size: 0, lastModified: 0 };
  }
  const file = await handle.getFile();
  let size = 0;
  if (file && Number.isFinite(file.size) && file.size >= 0) {
    size = Math.floor(file.size);
  } else if (file && typeof file.arrayBuffer === 'function') {
    const buf = await file.arrayBuffer();
    size = buf ? buf.byteLength : 0;
  }
  const lastModified =
    file && typeof file.lastModified === 'number' && Number.isFinite(file.lastModified)
      ? Math.max(0, Math.floor(file.lastModified))
      : 0;
  return { size, lastModified };
}

async function* iterateDirEntries(dirHandle) {
  if (!dirHandle) return;
  if (typeof dirHandle.entries === 'function') {
    for await (const pair of dirHandle.entries()) {
      yield pair;
    }
    return;
  }
  if (typeof dirHandle[Symbol.asyncIterator] === 'function') {
    for await (const pair of dirHandle) {
      yield pair;
    }
    return;
  }
  if (typeof dirHandle.values === 'function') {
    for await (const handle of dirHandle.values()) {
      if (handle && typeof handle.name === 'string') {
        yield [handle.name, handle];
      }
    }
  }
}

async function readFileHandleSize(handle) {
  if (!handle || typeof handle.getFile !== 'function') {
    return 0;
  }
  const file = await handle.getFile();
  if (file && Number.isFinite(file.size) && file.size >= 0) {
    return Math.floor(file.size);
  }
  if (file && typeof file.arrayBuffer === 'function') {
    const buf = await file.arrayBuffer();
    return buf ? buf.byteLength : 0;
  }
  return 0;
}

const FAST_BOOT_PRESSURE_RANK = Object.freeze({
  [FAST_BOOT_PRESSURE_LEVELS.LOW]: 0,
  [FAST_BOOT_PRESSURE_LEVELS.MEDIUM]: 1,
  [FAST_BOOT_PRESSURE_LEVELS.HIGH]: 2,
  [FAST_BOOT_PRESSURE_LEVELS.CRITICAL]: 3,
});

const FAST_BOOT_PRESSURE_BY_RANK = Object.freeze([
  FAST_BOOT_PRESSURE_LEVELS.LOW,
  FAST_BOOT_PRESSURE_LEVELS.MEDIUM,
  FAST_BOOT_PRESSURE_LEVELS.HIGH,
  FAST_BOOT_PRESSURE_LEVELS.CRITICAL,
]);

function normalizeFastBootPressureLevel(level) {
  if (typeof level !== 'string') {
    return null;
  }
  const lower = level.trim().toLowerCase();
  if (lower === 'ok' || lower === 'low' || lower === 'normal') {
    return FAST_BOOT_PRESSURE_LEVELS.LOW;
  }
  if (lower === 'medium' || lower === 'moderate') {
    return FAST_BOOT_PRESSURE_LEVELS.MEDIUM;
  }
  if (lower === 'warn' || lower === 'warning' || lower === 'high') {
    return FAST_BOOT_PRESSURE_LEVELS.HIGH;
  }
  if (lower === 'critical') {
    return FAST_BOOT_PRESSURE_LEVELS.CRITICAL;
  }
  return null;
}

function maxFastBootPressure(a, b) {
  const rankA = FAST_BOOT_PRESSURE_RANK[a] ?? 0;
  const rankB = FAST_BOOT_PRESSURE_RANK[b] ?? 0;
  return FAST_BOOT_PRESSURE_BY_RANK[Math.max(rankA, rankB)];
}

function classifyMemoryPressureForFastBoot(options = {}) {
  let level = FAST_BOOT_PRESSURE_LEVELS.LOW;
  if (!options || typeof options !== 'object') {
    return level;
  }

  const explicit = normalizeFastBootPressureLevel(options.memoryPressure);
  if (explicit) {
    level = maxFastBootPressure(level, explicit);
  }

  const memObj = options.memory && typeof options.memory === 'object' ? options.memory : null;
  const usedRaw =
    options.jsHeapUsedBytes !== undefined
      ? options.jsHeapUsedBytes
      : memObj
        ? memObj.usedJSHeapSize
        : undefined;
  const limitRaw =
    options.jsHeapLimitBytes !== undefined
      ? options.jsHeapLimitBytes
      : memObj
        ? memObj.jsHeapSizeLimit
        : undefined;

  if (
    typeof usedRaw === 'number' &&
    Number.isFinite(usedRaw) &&
    typeof limitRaw === 'number' &&
    Number.isFinite(limitRaw) &&
    limitRaw > 0
  ) {
    const heapRatio = clampUnitRatio(Math.max(0, usedRaw) / limitRaw, 0);
    if (heapRatio >= 0.9) {
      level = maxFastBootPressure(level, FAST_BOOT_PRESSURE_LEVELS.CRITICAL);
    } else if (heapRatio >= 0.75) {
      level = maxFastBootPressure(level, FAST_BOOT_PRESSURE_LEVELS.HIGH);
    } else if (heapRatio >= 0.55) {
      level = maxFastBootPressure(level, FAST_BOOT_PRESSURE_LEVELS.MEDIUM);
    }
  }

  if (
    typeof options.deviceMemoryGb === 'number' &&
    Number.isFinite(options.deviceMemoryGb) &&
    options.deviceMemoryGb > 0
  ) {
    if (options.deviceMemoryGb <= 2) {
      level = maxFastBootPressure(level, FAST_BOOT_PRESSURE_LEVELS.CRITICAL);
    } else if (options.deviceMemoryGb <= 4) {
      level = maxFastBootPressure(level, FAST_BOOT_PRESSURE_LEVELS.HIGH);
    } else if (options.deviceMemoryGb <= 8) {
      level = maxFastBootPressure(level, FAST_BOOT_PRESSURE_LEVELS.MEDIUM);
    }
  }

  return level;
}

function normalizeDelegatedEvictionResult(rawResult) {
  if (typeof rawResult === 'number') {
    const evictedBytes = sanitizeNonNegativeInt(rawResult);
    return {
      evictedBytes,
      evictedEntries: evictedBytes > 0 ? 1 : 0,
      ok: true,
    };
  }
  if (!rawResult || typeof rawResult !== 'object') {
    return {
      evictedBytes: 0,
      evictedEntries: 0,
      ok: true,
    };
  }

  const evictedBytes = sanitizeNonNegativeInt(
    rawResult.evictedBytes !== undefined
      ? rawResult.evictedBytes
      : rawResult.freedBytes !== undefined
        ? rawResult.freedBytes
        : rawResult.bytesEvicted !== undefined
          ? rawResult.bytesEvicted
          : rawResult.bytesFreed !== undefined
            ? rawResult.bytesFreed
            : rawResult.bytes
  );

  const evictedEntries = sanitizeNonNegativeInt(
    rawResult.evictedEntries !== undefined
      ? rawResult.evictedEntries
      : rawResult.removedCount !== undefined
        ? rawResult.removedCount
        : rawResult.evictedCount !== undefined
          ? rawResult.evictedCount
          : rawResult.entriesEvicted !== undefined
            ? rawResult.entriesEvicted
            : rawResult.count !== undefined
              ? rawResult.count
              : rawResult.entries
  );

  let ok = true;
  if (rawResult.ok !== undefined) {
    ok = Boolean(rawResult.ok);
  } else if (rawResult.supported === false) {
    ok = false;
  }

  return {
    evictedBytes,
    evictedEntries,
    ok,
  };
}

export class BrowserStorageManager {
  /**
   * @param {object} [options]
   * @param {StorageManager|null} [options.storageObj] Injectable navigator.storage equivalent.
   * @param {FileSystemDirectoryHandle|null} [options.opfsRoot] Injectable OPFS root handle.
   * @param {string} [options.opfsDirName='playgta6-cache'] OPFS cache subdirectory name.
   * @param {IDBFactory|null} [options.indexedDBFactory] Injectable indexedDB factory.
   * @param {string} [options.dbName='playgta6-userdata'] IndexedDB database name.
   * @param {string} [options.storeName='files'] IndexedDB object store name.
   * @param {number} [options.dbVersion=1] IndexedDB schema version.
   * @param {() => number} [options.now] Injectable clock function returning ms epoch.
   * @param {() => number} [options.nowFn] Injectable clock function returning ms epoch (preferred over `now`).
   * @param {number} [options.warnThresholdRatio=0.75] Quota ratio in [0, 1] at/above which pressure is 'warn'.
   * @param {number} [options.criticalThresholdRatio=0.9] Quota ratio in [0, 1] at/above which pressure is 'critical'.
   */
  constructor(options = {}) {
    this.storageObj =
      options.storageObj !== undefined
        ? options.storageObj
        : typeof navigator !== 'undefined' && navigator && navigator.storage
          ? navigator.storage
          : null;

    this._opfsRootExplicitlyNull = options.opfsRoot === null;
    this.opfsRoot = options.opfsRoot !== undefined ? options.opfsRoot : null;

    this.opfsDirName =
      typeof options.opfsDirName === 'string' && options.opfsDirName.length > 0
        ? options.opfsDirName
        : DEFAULT_OPFS_CACHE_DIR;

    this.indexedDBFactory =
      options.indexedDBFactory !== undefined
        ? options.indexedDBFactory
        : options.indexedDB !== undefined
          ? options.indexedDB
          : typeof globalThis !== 'undefined' && globalThis.indexedDB
            ? globalThis.indexedDB
            : null;

    this.dbName =
      typeof options.dbName === 'string' && options.dbName.length > 0
        ? options.dbName
        : DEFAULT_USERDATA_DB_NAME;

    this.storeName =
      typeof options.storeName === 'string' && options.storeName.length > 0
        ? options.storeName
        : DEFAULT_USERDATA_STORE_NAME;

    this.dbVersion =
      Number.isInteger(options.dbVersion) && options.dbVersion > 0
        ? options.dbVersion
        : DEFAULT_USERDATA_DB_VERSION;

    this._now =
      typeof options.nowFn === 'function'
        ? options.nowFn
        : typeof options.now === 'function'
          ? options.now
          : () => Date.now();

    const thresholds = normalizeQuotaThresholds(options);
    this.warnThresholdRatio = thresholds.warnThresholdRatio;
    this.criticalThresholdRatio = thresholds.criticalThresholdRatio;

    this.maxCacheBytes =
      typeof options.maxCacheBytes === 'number' &&
      Number.isFinite(options.maxCacheBytes) &&
      options.maxCacheBytes >= 0
        ? Math.floor(options.maxCacheBytes)
        : null;

    this.quotaBudgetRatio =
      typeof options.quotaBudgetRatio === 'number' && Number.isFinite(options.quotaBudgetRatio)
        ? clampUnitRatio(options.quotaBudgetRatio, DEFAULT_QUOTA_BUDGET_RATIO)
        : null;

    this.cacheIndexFileName =
      typeof options.cacheIndexFileName === 'string' &&
      options.cacheIndexFileName.length > 0 &&
      !options.cacheIndexFileName.includes('/') &&
      !options.cacheIndexFileName.includes('\\') &&
      !options.cacheIndexFileName.includes('\0')
        ? options.cacheIndexFileName
        : DEFAULT_CACHE_INDEX_FILENAME;

    this.persistCacheIndex = options.persistCacheIndex !== false;
    this.opfsCache =
      options.opfsCache && typeof options.opfsCache === 'object' ? options.opfsCache : null;

    this._evictedEntries = 0;
    this._evictedBytes = 0;
    this._cleanupCount = 0;
    this._quotaEvictionRuns = 0;
    this._lastCleanupAtMs = null;
    this._lastQuotaPressure = QUOTA_PRESSURE_LEVELS.UNKNOWN;
    this._lastQuotaUsageRatio = null;
    this._lastQuotaSampleAtMs = null;
    this._lastQuotaBytes = 0;
    this._lastQuotaUsageBytes = 0;

    this._cacheEntries = new Map();
    this._cacheIndexLoaded = false;
    this._cacheAccessSeq = 0;
    this._cacheEvictedEntries = 0;
    this._cacheEvictedBytes = 0;
    this._cacheHits = 0;
    this._cacheMisses = 0;
    this._cacheInvalidations = 0;
  }

  _nextAccessOrder() {
    this._cacheAccessSeq += 1;
    return this._cacheAccessSeq;
  }

  _readNowMs() {
    try {
      return sanitizeTimestampMs(this._now());
    } catch {
      return null;
    }
  }

  _recordCleanup(entries, bytes) {
    this._evictedEntries += sanitizeNonNegativeInt(entries);
    this._evictedBytes += sanitizeNonNegativeInt(bytes);
    this._cleanupCount += 1;
    this._lastCleanupAtMs = this._readNowMs();
  }

  _recordQuotaSample(estimate) {
    const pressure = this.classifyQuotaPressure(estimate);
    const usageRatio =
      estimate &&
      estimate.supported !== false &&
      typeof estimate.quotaBytes === 'number' &&
      Number.isFinite(estimate.quotaBytes) &&
      estimate.quotaBytes > 0 &&
      typeof estimate.usageBytes === 'number' &&
      Number.isFinite(estimate.usageBytes)
        ? clampUnitRatio(Math.max(0, estimate.usageBytes) / estimate.quotaBytes, 0)
        : null;

    this._lastQuotaPressure = pressure;
    this._lastQuotaUsageRatio = usageRatio;
    this._lastQuotaSampleAtMs = this._readNowMs();
    if (
      estimate &&
      estimate.supported !== false &&
      typeof estimate.quotaBytes === 'number' &&
      Number.isFinite(estimate.quotaBytes) &&
      estimate.quotaBytes > 0
    ) {
      this._lastQuotaBytes = Math.floor(estimate.quotaBytes);
      this._lastQuotaUsageBytes = sanitizeNonNegativeInt(estimate.usageBytes);
    } else {
      this._lastQuotaBytes = 0;
      this._lastQuotaUsageBytes = 0;
    }
  }

  /**
   * Classifies quota pressure using this manager's configured thresholds by default,
   * while allowing per-call threshold overrides.
   *
   * @param {object|number|null|undefined} estimateOrUsage
   * @param {number|object} [quotaOrOptions]
   * @param {object} [maybeOptions]
   * @returns {'ok'|'warn'|'critical'|'unknown'}
   */
  classifyQuotaPressure(estimateOrUsage, quotaOrOptions, maybeOptions) {
    const defaultThresholds = {
      warnThresholdRatio: this.warnThresholdRatio,
      criticalThresholdRatio: this.criticalThresholdRatio,
    };

    if (estimateOrUsage !== null && typeof estimateOrUsage === 'object') {
      const mergedOptions = { ...defaultThresholds };
      if (quotaOrOptions && typeof quotaOrOptions === 'object') {
        if (quotaOrOptions.warnThresholdRatio !== undefined) {
          mergedOptions.warnThresholdRatio = quotaOrOptions.warnThresholdRatio;
        }
        if (quotaOrOptions.criticalThresholdRatio !== undefined) {
          mergedOptions.criticalThresholdRatio = quotaOrOptions.criticalThresholdRatio;
        }
      }
      return classifyQuotaPressure(estimateOrUsage, mergedOptions);
    }

    const mergedOptions = { ...defaultThresholds };
    if (maybeOptions && typeof maybeOptions === 'object') {
      if (maybeOptions.warnThresholdRatio !== undefined) {
        mergedOptions.warnThresholdRatio = maybeOptions.warnThresholdRatio;
      }
      if (maybeOptions.criticalThresholdRatio !== undefined) {
        mergedOptions.criticalThresholdRatio = maybeOptions.criticalThresholdRatio;
      }
    }
    return classifyQuotaPressure(estimateOrUsage, quotaOrOptions, mergedOptions);
  }

  /**
   * Returns a fresh, sanitized snapshot of eviction & quota telemetry counters.
   */
  getStats() {
    return {
      evictedEntries: this._evictedEntries,
      evictedBytes: this._evictedBytes,
      evictedFormatted: formatBytes(this._evictedBytes),
      cleanupCount: this._cleanupCount,
      quotaEvictionRuns: this._quotaEvictionRuns,
      lastCleanupAtMs: this._lastCleanupAtMs,
      warnThresholdRatio: this.warnThresholdRatio,
      criticalThresholdRatio: this.criticalThresholdRatio,
      lastQuotaPressure: this._lastQuotaPressure,
      lastQuotaUsageRatio: this._lastQuotaUsageRatio,
      lastQuotaSampleAtMs: this._lastQuotaSampleAtMs,
    };
  }

  /**
   * Resets eviction and last-observed quota telemetry counters and returns the cleared stats.
   */
  resetStats() {
    this._evictedEntries = 0;
    this._evictedBytes = 0;
    this._cleanupCount = 0;
    this._quotaEvictionRuns = 0;
    this._lastCleanupAtMs = null;
    this._lastQuotaPressure = QUOTA_PRESSURE_LEVELS.UNKNOWN;
    this._lastQuotaUsageRatio = null;
    this._lastQuotaSampleAtMs = null;
    this._lastQuotaBytes = 0;
    this._lastQuotaUsageBytes = 0;
    this._cacheEvictedEntries = 0;
    this._cacheEvictedBytes = 0;
    this._cacheHits = 0;
    this._cacheMisses = 0;
    this._cacheInvalidations = 0;
    return this.getStats();
  }

  // --- Storage Quota & Persistence ---

  async getQuotaEstimate() {
    const fallback = {
      supported: false,
      usageBytes: 0,
      quotaBytes: 0,
      availableBytes: 0,
      percentUsed: 0,
      persisted: false,
    };

    if (!this.storageObj || typeof this.storageObj.estimate !== 'function') {
      this._recordQuotaSample(fallback);
      return fallback;
    }

    try {
      const estimate = await this.storageObj.estimate();
      const usageBytes =
        estimate && Number.isFinite(estimate.usage) && estimate.usage >= 0
          ? Math.floor(estimate.usage)
          : 0;
      const quotaBytes =
        estimate && Number.isFinite(estimate.quota) && estimate.quota >= 0
          ? Math.floor(estimate.quota)
          : 0;
      const availableBytes = Math.max(0, quotaBytes - usageBytes);
      const percentUsed =
        quotaBytes > 0
          ? Math.min(100, Math.round((usageBytes / quotaBytes) * 10000) / 100)
          : 0;

      let persisted = false;
      if (typeof this.storageObj.persisted === 'function') {
        try {
          persisted = Boolean(await this.storageObj.persisted());
        } catch {
          persisted = false;
        }
      }

      const result = {
        supported: true,
        usageBytes,
        quotaBytes,
        availableBytes,
        percentUsed,
        persisted,
      };
      this._recordQuotaSample(result);
      return result;
    } catch {
      this._recordQuotaSample(fallback);
      return fallback;
    }
  }

  async requestPersistentStorage() {
    if (!this.storageObj || typeof this.storageObj.persist !== 'function') {
      return { supported: false, persisted: false };
    }
    try {
      const persisted = Boolean(await this.storageObj.persist());
      return { supported: true, persisted };
    } catch {
      return { supported: true, persisted: false };
    }
  }

  // --- OPFS Range Cache ('playgta6-cache') ---

  async _resolveOpfsRoot() {
    if (this._opfsRootExplicitlyNull) {
      return null;
    }
    if (this.opfsRoot && typeof this.opfsRoot.getDirectoryHandle === 'function') {
      return this.opfsRoot;
    }
    if (this.storageObj && typeof this.storageObj.getDirectory === 'function') {
      try {
        const root = await this.storageObj.getDirectory();
        if (root && typeof root.getDirectoryHandle === 'function') {
          return root;
        }
      } catch {
        return null;
      }
    }
    return null;
  }

  async _collectOpfsEntries(dirHandle, prefix = '', out = []) {
    for await (const [name, handle] of iterateDirEntries(dirHandle)) {
      if (!handle || typeof name !== 'string') continue;
      if (prefix === '' && name === this.cacheIndexFileName && handle.kind !== 'directory') {
        continue;
      }
      const relName = prefix ? `${prefix}/${name}` : name;
      if (handle.kind === 'directory') {
        await this._collectOpfsEntries(handle, relName, out);
      } else if (handle.kind === 'file' || typeof handle.getFile === 'function') {
        const size = await readFileHandleSize(handle);
        out.push({ name: relName, size });
      }
    }
    return out;
  }

  async _collectOpfsEntriesWithMeta(dirHandle, prefix = '', out = []) {
    for await (const [name, handle] of iterateDirEntries(dirHandle)) {
      if (!handle || typeof name !== 'string') continue;
      if (prefix === '' && name === this.cacheIndexFileName && handle.kind !== 'directory') {
        continue;
      }
      const relName = prefix ? `${prefix}/${name}` : name;
      if (handle.kind === 'directory') {
        await this._collectOpfsEntriesWithMeta(handle, relName, out);
      } else if (handle.kind === 'file' || typeof handle.getFile === 'function') {
        const meta = await readFileHandleMeta(handle);
        out.push({ name: relName, size: meta.size, lastModified: meta.lastModified });
      }
    }
    return out;
  }

  async _ensureOpfsCacheDir({ create = false } = {}) {
    const root = await this._resolveOpfsRoot();
    if (!root) {
      return { supported: false, dir: null };
    }
    try {
      const dir = await root.getDirectoryHandle(this.opfsDirName, { create });
      return { supported: true, dir };
    } catch (err) {
      if (!create && isNotFoundError(err)) {
        return { supported: true, dir: null };
      }
      return { supported: false, dir: null };
    }
  }

  async _getOpfsFileHandle(cacheDir, key, { create = false } = {}) {
    const segments = key.split('/');
    let dir = cacheDir;
    for (let i = 0; i < segments.length - 1; i += 1) {
      dir = await dir.getDirectoryHandle(segments[i], { create });
    }
    const leafName = segments[segments.length - 1];
    return dir.getFileHandle(leafName, { create });
  }

  async _writeOpfsFile(cacheDir, key, bytes) {
    const handle = await this._getOpfsFileHandle(cacheDir, key, { create: true });
    if (!handle || typeof handle.createWritable !== 'function') {
      throw new Error(`OPFS file handle for "${key}" does not support createWritable()`);
    }
    const writable = await handle.createWritable();
    try {
      await writable.write(bytes);
      await writable.close();
    } catch (err) {
      if (typeof writable.abort === 'function') {
        try {
          await writable.abort();
        } catch {
          // Ignore abort errors.
        }
      }
      throw err;
    }
  }

  async _readOpfsFile(cacheDir, key) {
    let handle;
    try {
      handle = await this._getOpfsFileHandle(cacheDir, key, { create: false });
    } catch (err) {
      if (isNotFoundError(err)) {
        return null;
      }
      throw err;
    }
    return readFileHandleBytes(handle);
  }

  async _removeOpfsFile(cacheDir, key) {
    const segments = key.split('/');
    const dirChain = [{ dir: cacheDir, name: null }];
    let dir = cacheDir;
    for (let i = 0; i < segments.length - 1; i += 1) {
      try {
        dir = await dir.getDirectoryHandle(segments[i], { create: false });
        dirChain.push({ dir, name: segments[i] });
      } catch (err) {
        if (isNotFoundError(err)) {
          return false;
        }
        throw err;
      }
    }
    const leafName = segments[segments.length - 1];
    if (typeof dir.removeEntry !== 'function') {
      return false;
    }
    let removed = false;
    try {
      await dir.removeEntry(leafName);
      removed = true;
    } catch (err) {
      if (!isNotFoundError(err)) {
        throw err;
      }
    }

    // Best-effort prune of now-empty nested subdirectories (never removes cacheDir itself).
    for (let i = dirChain.length - 1; i >= 1; i -= 1) {
      const child = dirChain[i];
      const parent = dirChain[i - 1].dir;
      try {
        let hasAny = false;
        for await (const _ of iterateDirEntries(child.dir)) {
          hasAny = true;
          break;
        }
        if (!hasAny && typeof parent.removeEntry === 'function') {
          await parent.removeEntry(child.name);
        } else {
          break;
        }
      } catch {
        break;
      }
    }

    return removed;
  }

  async _readCacheIndexFile(cacheDir) {
    if (!this.persistCacheIndex || !cacheDir || typeof cacheDir.getFileHandle !== 'function') {
      return null;
    }
    try {
      const handle = await cacheDir.getFileHandle(this.cacheIndexFileName, { create: false });
      const bytes = await readFileHandleBytes(handle);
      if (!bytes || bytes.byteLength === 0) {
        return null;
      }
      const parsed = JSON.parse(new TextDecoder().decode(bytes));
      if (!parsed || typeof parsed !== 'object' || parsed.version !== CACHE_INDEX_VERSION) {
        return null;
      }
      if (!Array.isArray(parsed.entries)) {
        return null;
      }
      return parsed.entries;
    } catch {
      return null;
    }
  }

  async _persistCacheIndex(cacheDir) {
    if (!this.persistCacheIndex || !cacheDir || typeof cacheDir.getFileHandle !== 'function') {
      return false;
    }
    try {
      const entries = Array.from(this._cacheEntries.values())
        .sort((a, b) => compareAscii(a.key, b.key))
        .map((e) => ({
          key: e.key,
          sizeBytes: e.sizeBytes,
          lastAccessedMs: e.lastAccessedMs,
          createdAtMs: e.createdAtMs,
          etag: e.etag,
          versionKey: e.versionKey,
          pinned: e.pinned,
          kind: e.kind,
          accessOrder: e.accessOrder,
        }));
      const payload = new TextEncoder().encode(
        JSON.stringify({
          version: CACHE_INDEX_VERSION,
          entries,
        })
      );
      await this._writeOpfsFile(cacheDir, this.cacheIndexFileName, payload);
      return true;
    } catch {
      return false;
    }
  }

  _computeCacheTotals() {
    let totalBytes = 0;
    let pinnedCount = 0;
    let pinnedBytes = 0;
    for (const entry of this._cacheEntries.values()) {
      totalBytes += entry.sizeBytes;
      if (entry.pinned) {
        pinnedCount += 1;
        pinnedBytes += entry.sizeBytes;
      }
    }
    return {
      entryCount: this._cacheEntries.size,
      totalBytes,
      pinnedCount,
      pinnedBytes,
    };
  }

  async _refreshQuotaEstimateIfAvailable() {
    if (this.storageObj && typeof this.storageObj.estimate === 'function') {
      return this.getQuotaEstimate();
    }
    return null;
  }

  async inspectOpfsCache() {
    const unsupported = {
      supported: false,
      dirName: this.opfsDirName,
      entryCount: 0,
      totalBytes: 0,
      entries: [],
    };

    const root = await this._resolveOpfsRoot();
    if (!root) {
      return unsupported;
    }

    let cacheDir;
    try {
      cacheDir = await root.getDirectoryHandle(this.opfsDirName, { create: false });
    } catch (err) {
      if (isNotFoundError(err)) {
        return {
          supported: true,
          dirName: this.opfsDirName,
          entryCount: 0,
          totalBytes: 0,
          entries: [],
        };
      }
      return unsupported;
    }

    try {
      const entries = await this._collectOpfsEntries(cacheDir);
      entries.sort((a, b) => compareAscii(a.name, b.name));
      let totalBytes = 0;
      for (const entry of entries) {
        totalBytes += entry.size;
      }
      return {
        supported: true,
        dirName: this.opfsDirName,
        entryCount: entries.length,
        totalBytes,
        entries,
      };
    } catch {
      return unsupported;
    }
  }

  async clearOpfsCache() {
    const unsupported = {
      supported: false,
      removedCount: 0,
      freedBytes: 0,
    };

    const root = await this._resolveOpfsRoot();
    if (!root) {
      return unsupported;
    }

    let cacheDir;
    try {
      cacheDir = await root.getDirectoryHandle(this.opfsDirName, { create: false });
    } catch (err) {
      if (isNotFoundError(err)) {
        this._cacheEntries.clear();
        this._cacheIndexLoaded = true;
        this._recordCleanup(0, 0);
        return { supported: true, removedCount: 0, freedBytes: 0 };
      }
      return unsupported;
    }

    if (typeof cacheDir.removeEntry !== 'function') {
      return unsupported;
    }

    try {
      const topLevel = [];
      let hasIndexFile = false;
      for await (const [name, handle] of iterateDirEntries(cacheDir)) {
        if (!handle || typeof name !== 'string') continue;
        if (name === this.cacheIndexFileName && handle.kind !== 'directory') {
          hasIndexFile = true;
          continue;
        }
        if (handle.kind === 'directory') {
          const nested = await this._collectOpfsEntries(handle);
          let nestedBytes = 0;
          for (const item of nested) {
            nestedBytes += item.size;
          }
          topLevel.push({ name, isDir: true, fileCount: nested.length, bytes: nestedBytes });
        } else if (handle.kind === 'file' || typeof handle.getFile === 'function') {
          const size = await readFileHandleSize(handle);
          topLevel.push({ name, isDir: false, fileCount: 1, bytes: size });
        }
      }

      let removedCount = 0;
      let freedBytes = 0;
      for (const item of topLevel) {
        if (item.isDir) {
          await cacheDir.removeEntry(item.name, { recursive: true });
        } else {
          await cacheDir.removeEntry(item.name);
        }
        removedCount += item.fileCount;
        freedBytes += item.bytes;
      }

      if (hasIndexFile) {
        try {
          await cacheDir.removeEntry(this.cacheIndexFileName);
        } catch {
          // Ignore index removal error.
        }
      }

      this._cacheEntries.clear();
      this._cacheIndexLoaded = true;
      this._cacheEvictedEntries += removedCount;
      this._cacheEvictedBytes += freedBytes;
      this._recordCleanup(removedCount, freedBytes);
      return {
        supported: true,
        removedCount,
        freedBytes,
      };
    } catch {
      return unsupported;
    }
  }

  // --- OPFS Quota-Aware LRU & Integrity-Keyed Shader/PGIM Cache Management ---

  /**
   * Hydrates persisted cache metadata from OPFS and reconciles it against actual
   * files in the cache directory (discovering untracked files and pruning missing ones).
   *
   * @param {object} [options]
   * @param {boolean} [options.force=false] Re-read persisted index file even if already loaded.
   */
  async syncCacheIndex(options = {}) {
    const { force = false } = options;
    const { supported, dir: cacheDir } = await this._ensureOpfsCacheDir({ create: false });
    if (!supported) {
      return {
        supported: false,
        entryCount: 0,
        totalBytes: 0,
        discoveredCount: 0,
        prunedCount: 0,
      };
    }

    if (!cacheDir) {
      const prunedCount = this._cacheEntries.size;
      this._cacheEntries.clear();
      this._cacheIndexLoaded = true;
      return {
        supported: true,
        entryCount: 0,
        totalBytes: 0,
        discoveredCount: 0,
        prunedCount,
      };
    }

    if (!this._cacheIndexLoaded || force) {
      const storedEntries = await this._readCacheIndexFile(cacheDir);
      if (storedEntries) {
        if (force) {
          this._cacheEntries.clear();
        }
        for (const raw of storedEntries) {
          if (!raw || typeof raw !== 'object' || typeof raw.key !== 'string') continue;
          let validKey;
          try {
            validKey = validateCacheKey(raw.key, this.cacheIndexFileName);
          } catch {
            continue;
          }
          if (this._cacheEntries.has(validKey) && !force) {
            continue;
          }
          const accessOrder =
            Number.isInteger(raw.accessOrder) && raw.accessOrder > 0
              ? raw.accessOrder
              : this._nextAccessOrder();
          if (accessOrder > this._cacheAccessSeq) {
            this._cacheAccessSeq = accessOrder;
          }
          const lastAccessedMs = sanitizeTimestampMs(raw.lastAccessedMs) ?? 0;
          const createdAtMs = sanitizeTimestampMs(raw.createdAtMs) ?? lastAccessedMs;
          this._cacheEntries.set(validKey, {
            key: validKey,
            sizeBytes: sanitizeNonNegativeInt(raw.sizeBytes),
            lastAccessedMs,
            createdAtMs,
            etag: normalizeOptionalTag(raw.etag),
            versionKey: normalizeOptionalTag(raw.versionKey),
            pinned: Boolean(raw.pinned),
            kind: normalizeCacheKind(raw.kind, validKey),
            accessOrder,
          });
        }
      }
    }

    const diskEntries = await this._collectOpfsEntriesWithMeta(cacheDir);
    diskEntries.sort((a, b) => compareAscii(a.name, b.name));

    const presentKeys = new Set();
    let discoveredCount = 0;

    for (const file of diskEntries) {
      let validKey;
      try {
        validKey = validateCacheKey(file.name, this.cacheIndexFileName);
      } catch {
        continue;
      }
      presentKeys.add(validKey);
      const existing = this._cacheEntries.get(validKey);
      if (existing) {
        existing.sizeBytes = file.size;
      } else {
        const ts = sanitizeTimestampMs(file.lastModified) ?? 0;
        this._cacheEntries.set(validKey, {
          key: validKey,
          sizeBytes: file.size,
          lastAccessedMs: ts,
          createdAtMs: ts,
          etag: null,
          versionKey: null,
          pinned: false,
          kind: inferCacheEntryKind(validKey),
          accessOrder: this._nextAccessOrder(),
        });
        discoveredCount += 1;
      }
    }

    let prunedCount = 0;
    for (const key of Array.from(this._cacheEntries.keys())) {
      if (!presentKeys.has(key)) {
        this._cacheEntries.delete(key);
        prunedCount += 1;
      }
    }

    this._cacheIndexLoaded = true;
    const totals = this._computeCacheTotals();
    return {
      supported: true,
      entryCount: totals.entryCount,
      totalBytes: totals.totalBytes,
      discoveredCount,
      prunedCount,
    };
  }

  /**
   * Stores an entry (such as a `.pgim` boot image, `.wgsl.pack` shader pack, or chunk)
   * into OPFS with integrity metadata (`etag`, `versionKey`, `pinned`, `sizeBytes`, `lastAccessedMs`).
   * Automatically recovers from OPFS `QuotaExceededError` via LRU eviction and enforces
   * configured cache/quota budgets when enabled.
   *
   * @param {string} key Relative OPFS cache key (e.g. 'boot.pgim', 'shaders/main.wgsl.pack').
   * @param {Uint8Array|ArrayBufferView|ArrayBuffer|string} data Payload bytes.
   * @param {object} [options]
   * @param {string|null} [options.etag] Optional HTTP ETag / integrity digest.
   * @param {string|null} [options.versionKey] Optional content/build version key.
   * @param {boolean} [options.pinned=false] When true, protects entry from LRU eviction.
   * @param {string} [options.kind] Optional cache kind ('pgim'|'shader-pack'|'range-chunk'|'generic').
   * @param {number} [options.maxBytes] Optional budget override enforced after storing.
   * @param {boolean} [options.autoEvict=true] Whether to enforce configured budget after write.
   * @param {boolean} [options.enforceQuota=false] Whether to also enforce browser quota budget.
   * @param {number} [options.targetUsageRatio] Optional target quota usage ratio when enforceQuota is active.
   */
  async putCacheEntry(key, data, options = {}) {
    const validKey = validateCacheKey(key, this.cacheIndexFileName);
    const bytes = toUint8Array(data);

    const { supported, dir: cacheDir } = await this._ensureOpfsCacheDir({ create: true });
    if (!supported || !cacheDir) {
      return {
        supported: false,
        stored: false,
        key: validKey,
        sizeBytes: 0,
        replaced: false,
        versionChanged: false,
        quotaRecovered: false,
        evictedEntries: 0,
        evictedBytes: 0,
        meta: null,
      };
    }

    if (!this._cacheIndexLoaded) {
      await this.syncCacheIndex();
    }

    const existing = this._cacheEntries.get(validKey) || null;
    const replaced = existing !== null;
    const etag = normalizeOptionalTag(options.etag);
    const versionKey = normalizeOptionalTag(options.versionKey);
    const pinned = options.pinned !== undefined ? Boolean(options.pinned) : false;
    const kind = normalizeCacheKind(options.kind, validKey);

    const versionChanged =
      replaced &&
      ((existing.etag !== etag && (existing.etag !== null || etag !== null)) ||
        (existing.versionKey !== versionKey &&
          (existing.versionKey !== null || versionKey !== null)));

    let evictedEntries = 0;
    let evictedBytes = 0;
    let quotaRecovered = false;

    try {
      await this._writeOpfsFile(cacheDir, validKey, bytes);
    } catch (err) {
      if (!isQuotaExceededError(err)) {
        throw err;
      }
      const totals = this._computeCacheTotals();
      const targetBudget = Math.max(0, totals.totalBytes - Math.max(bytes.byteLength, 1));
      const recovery = await this.evictToBudget(targetBudget, { excludeKeys: [validKey] });
      evictedEntries += recovery.evictedEntries;
      evictedBytes += recovery.evictedBytes;
      await this._writeOpfsFile(cacheDir, validKey, bytes);
      quotaRecovered = true;
    }

    if (versionChanged) {
      this._cacheInvalidations += 1;
    }

    const nowMs = this._readNowMs() ?? 0;
    const meta = {
      key: validKey,
      sizeBytes: bytes.byteLength,
      lastAccessedMs: nowMs,
      createdAtMs: existing && !versionChanged ? existing.createdAtMs : nowMs,
      etag,
      versionKey,
      pinned,
      kind,
      accessOrder: this._nextAccessOrder(),
    };
    this._cacheEntries.set(validKey, meta);
    await this._persistCacheIndex(cacheDir);

    if (options.autoEvict !== false) {
      let effectiveBudget =
        typeof options.maxBytes === 'number' &&
        Number.isFinite(options.maxBytes) &&
        options.maxBytes >= 0
          ? Math.floor(options.maxBytes)
          : this.maxCacheBytes;

      if (options.enforceQuota === true || this.quotaBudgetRatio !== null) {
        const quota = await this.getQuotaEstimate();
        const quotaBudget = this.computeQuotaBudget(quota, {
          targetUsageRatio: options.targetUsageRatio,
          maxBytes: effectiveBudget,
        });
        if (quotaBudget !== null) {
          effectiveBudget =
            effectiveBudget === null ? quotaBudget : Math.min(effectiveBudget, quotaBudget);
        }
      }

      if (effectiveBudget !== null) {
        const ev = await this.evictToBudget(effectiveBudget, { excludeKeys: [validKey] });
        evictedEntries += ev.evictedEntries;
        evictedBytes += ev.evictedBytes;
      } else {
        await this._refreshQuotaEstimateIfAvailable();
      }
    } else {
      await this._refreshQuotaEstimateIfAvailable();
    }

    return {
      supported: true,
      stored: true,
      key: validKey,
      sizeBytes: bytes.byteLength,
      replaced,
      versionChanged,
      quotaRecovered,
      evictedEntries,
      evictedBytes,
      meta: cloneCacheEntryMeta(this._cacheEntries.get(validKey) || meta),
    };
  }

  /**
   * Reads a cached entry from OPFS, validating optional `etag`, `versionKey`, and `expectedSizeBytes`.
   * On version/ETag/size mismatch, automatically invalidates and removes the stale entry from OPFS.
   * On hit, updates `lastAccessedMs` and LRU order (unless `touch: false`).
   *
   * @param {string} key
   * @param {object} [options]
   * @param {string|null} [options.etag] Expected ETag; mismatches trigger invalidation.
   * @param {string|null} [options.expectedEtag] Alias for `etag`.
   * @param {string|null} [options.versionKey] Expected version key; mismatches trigger invalidation.
   * @param {string|null} [options.expectedVersionKey] Alias for `versionKey`.
   * @param {number} [options.expectedSizeBytes] Optional expected byte length for integrity validation.
   * @param {boolean} [options.touch=true] Update `lastAccessedMs` and LRU recency on hit.
   */
  async getCacheEntry(key, options = {}) {
    const validKey = validateCacheKey(key, this.cacheIndexFileName);
    const { supported, dir: cacheDir } = await this._ensureOpfsCacheDir({ create: false });
    if (!supported) {
      return {
        supported: false,
        hit: false,
        invalidated: false,
        reason: 'unsupported',
        key: validKey,
        data: null,
        meta: null,
      };
    }

    if (!cacheDir) {
      this._cacheMisses += 1;
      return {
        supported: true,
        hit: false,
        invalidated: false,
        reason: 'not-found',
        key: validKey,
        data: null,
        meta: null,
      };
    }

    if (!this._cacheIndexLoaded) {
      await this.syncCacheIndex();
    }

    const expectedEtag =
      options.etag !== undefined
        ? normalizeOptionalTag(options.etag)
        : options.expectedEtag !== undefined
          ? normalizeOptionalTag(options.expectedEtag)
          : null;
    const expectedVersionKey =
      options.versionKey !== undefined
        ? normalizeOptionalTag(options.versionKey)
        : options.expectedVersionKey !== undefined
          ? normalizeOptionalTag(options.expectedVersionKey)
          : null;

    const existing = this._cacheEntries.get(validKey) || null;

    if (existing) {
      if (expectedEtag !== null && existing.etag !== expectedEtag) {
        await this._removeOpfsFile(cacheDir, validKey);
        this._cacheEntries.delete(validKey);
        await this._persistCacheIndex(cacheDir);
        this._cacheInvalidations += 1;
        this._cacheMisses += 1;
        return {
          supported: true,
          hit: false,
          invalidated: true,
          reason: 'etag-mismatch',
          key: validKey,
          data: null,
          meta: null,
        };
      }
      if (expectedVersionKey !== null && existing.versionKey !== expectedVersionKey) {
        await this._removeOpfsFile(cacheDir, validKey);
        this._cacheEntries.delete(validKey);
        await this._persistCacheIndex(cacheDir);
        this._cacheInvalidations += 1;
        this._cacheMisses += 1;
        return {
          supported: true,
          hit: false,
          invalidated: true,
          reason: 'version-mismatch',
          key: validKey,
          data: null,
          meta: null,
        };
      }
    }

    const bytes = await this._readOpfsFile(cacheDir, validKey);
    if (bytes === null) {
      if (existing) {
        this._cacheEntries.delete(validKey);
        await this._persistCacheIndex(cacheDir);
      }
      this._cacheMisses += 1;
      return {
        supported: true,
        hit: false,
        invalidated: false,
        reason: 'not-found',
        key: validKey,
        data: null,
        meta: null,
      };
    }

    if (!existing && (expectedEtag !== null || expectedVersionKey !== null)) {
      await this._removeOpfsFile(cacheDir, validKey);
      await this._persistCacheIndex(cacheDir);
      this._cacheInvalidations += 1;
      this._cacheMisses += 1;
      return {
        supported: true,
        hit: false,
        invalidated: true,
        reason: expectedEtag !== null ? 'etag-mismatch' : 'version-mismatch',
        key: validKey,
        data: null,
        meta: null,
      };
    }

    if (
      typeof options.expectedSizeBytes === 'number' &&
      Number.isFinite(options.expectedSizeBytes) &&
      bytes.byteLength !== Math.floor(options.expectedSizeBytes)
    ) {
      await this._removeOpfsFile(cacheDir, validKey);
      this._cacheEntries.delete(validKey);
      await this._persistCacheIndex(cacheDir);
      this._cacheInvalidations += 1;
      this._cacheMisses += 1;
      return {
        supported: true,
        hit: false,
        invalidated: true,
        reason: 'size-mismatch',
        key: validKey,
        data: null,
        meta: null,
      };
    }

    const nowMs = this._readNowMs() ?? 0;
    let meta = existing;
    if (!meta) {
      meta = {
        key: validKey,
        sizeBytes: bytes.byteLength,
        lastAccessedMs: nowMs,
        createdAtMs: nowMs,
        etag: null,
        versionKey: null,
        pinned: false,
        kind: inferCacheEntryKind(validKey),
        accessOrder: this._nextAccessOrder(),
      };
      this._cacheEntries.set(validKey, meta);
    } else {
      meta.sizeBytes = bytes.byteLength;
      if (options.touch !== false) {
        meta.lastAccessedMs = nowMs;
        meta.accessOrder = this._nextAccessOrder();
      }
    }

    this._cacheHits += 1;
    if (options.touch !== false || !existing) {
      await this._persistCacheIndex(cacheDir);
    }

    return {
      supported: true,
      hit: true,
      invalidated: false,
      reason: null,
      key: validKey,
      data: bytes,
      meta: cloneCacheEntryMeta(meta),
    };
  }

  /**
   * Registers or updates metadata for an existing OPFS cache entry without rewriting
   * its payload (useful for tagging or pinning files written by RangeCache or other workers).
   *
   * @param {string} key
   * @param {object} [options]
   */
  async registerCacheEntry(key, options = {}) {
    const validKey = validateCacheKey(key, this.cacheIndexFileName);
    const { supported, dir: cacheDir } = await this._ensureOpfsCacheDir({ create: false });
    if (!supported) {
      return { supported: false, found: false, key: validKey, meta: null };
    }
    if (!cacheDir) {
      return { supported: true, found: false, key: validKey, meta: null };
    }
    if (!this._cacheIndexLoaded) {
      await this.syncCacheIndex();
    }

    let sizeBytes = sanitizeNonNegativeInt(options.sizeBytes);
    const existing = this._cacheEntries.get(validKey) || null;

    try {
      const handle = await this._getOpfsFileHandle(cacheDir, validKey, { create: false });
      const fileMeta = await readFileHandleMeta(handle);
      sizeBytes = fileMeta.size;
    } catch (err) {
      if (isNotFoundError(err)) {
        if (existing) {
          this._cacheEntries.delete(validKey);
          await this._persistCacheIndex(cacheDir);
        }
        return { supported: true, found: false, key: validKey, meta: null };
      }
      throw err;
    }

    const nowMs = this._readNowMs() ?? 0;
    const lastAccessedMs =
      options.lastAccessedMs !== undefined
        ? (sanitizeTimestampMs(options.lastAccessedMs) ?? nowMs)
        : existing
          ? existing.lastAccessedMs
          : nowMs;
    const meta = {
      key: validKey,
      sizeBytes,
      lastAccessedMs,
      createdAtMs: existing ? existing.createdAtMs : nowMs,
      etag:
        options.etag !== undefined
          ? normalizeOptionalTag(options.etag)
          : existing
            ? existing.etag
            : null,
      versionKey:
        options.versionKey !== undefined
          ? normalizeOptionalTag(options.versionKey)
          : existing
            ? existing.versionKey
            : null,
      pinned:
        options.pinned !== undefined
          ? Boolean(options.pinned)
          : existing
            ? existing.pinned
            : false,
      kind:
        options.kind !== undefined
          ? normalizeCacheKind(options.kind, validKey)
          : existing
            ? existing.kind
            : inferCacheEntryKind(validKey),
      accessOrder: existing ? existing.accessOrder : this._nextAccessOrder(),
    };
    this._cacheEntries.set(validKey, meta);
    await this._persistCacheIndex(cacheDir);

    return {
      supported: true,
      found: true,
      key: validKey,
      meta: cloneCacheEntryMeta(meta),
    };
  }

  /**
   * Marks an existing cache entry as recently accessed without reading its payload.
   *
   * @param {string} key
   */
  async touchCacheEntry(key) {
    const validKey = validateCacheKey(key, this.cacheIndexFileName);
    const { supported, dir: cacheDir } = await this._ensureOpfsCacheDir({ create: false });
    if (!supported || !cacheDir) {
      return null;
    }
    if (!this._cacheIndexLoaded) {
      await this.syncCacheIndex();
    }
    const existing = this._cacheEntries.get(validKey);
    if (!existing) {
      return null;
    }
    existing.lastAccessedMs = this._readNowMs() ?? existing.lastAccessedMs;
    existing.accessOrder = this._nextAccessOrder();
    await this._persistCacheIndex(cacheDir);
    return cloneCacheEntryMeta(existing);
  }

  /**
   * Sets or clears the `pinned` flag on a cached entry (pinned entries are skipped by `evictToBudget`).
   *
   * @param {string} key
   * @param {boolean} pinned
   */
  async setCacheEntryPinned(key, pinned) {
    const validKey = validateCacheKey(key, this.cacheIndexFileName);
    const { supported, dir: cacheDir } = await this._ensureOpfsCacheDir({ create: false });
    if (!supported) {
      return { supported: false, found: false, key: validKey, pinned: false, meta: null };
    }
    if (!cacheDir) {
      return { supported: true, found: false, key: validKey, pinned: false, meta: null };
    }
    if (!this._cacheIndexLoaded) {
      await this.syncCacheIndex();
    }
    const existing = this._cacheEntries.get(validKey);
    if (!existing) {
      return { supported: true, found: false, key: validKey, pinned: false, meta: null };
    }
    existing.pinned = Boolean(pinned);
    await this._persistCacheIndex(cacheDir);
    return {
      supported: true,
      found: true,
      key: validKey,
      pinned: existing.pinned,
      meta: cloneCacheEntryMeta(existing),
    };
  }

  async pinCacheEntry(key) {
    return this.setCacheEntryPinned(key, true);
  }

  async unpinCacheEntry(key) {
    return this.setCacheEntryPinned(key, false);
  }

  /**
   * Deletes a single cache entry from OPFS and metadata tracking.
   *
   * @param {string} key
   * @param {object} [options]
   * @param {boolean} [options.respectPinned=false] Skip deletion when entry is pinned.
   * @param {boolean} [options.countInvalidation=false] Increment invalidation counter on removal.
   */
  async deleteCacheEntry(key, options = {}) {
    const validKey = validateCacheKey(key, this.cacheIndexFileName);
    const { respectPinned = false, countInvalidation = false } = options;
    const { supported, dir: cacheDir } = await this._ensureOpfsCacheDir({ create: false });
    if (!supported) {
      return {
        supported: false,
        removed: false,
        skippedPinned: false,
        key: validKey,
        freedBytes: 0,
      };
    }
    if (!cacheDir) {
      return {
        supported: true,
        removed: false,
        skippedPinned: false,
        key: validKey,
        freedBytes: 0,
      };
    }
    if (!this._cacheIndexLoaded) {
      await this.syncCacheIndex();
    }

    const existing = this._cacheEntries.get(validKey) || null;
    if (existing && respectPinned && existing.pinned) {
      return {
        supported: true,
        removed: false,
        skippedPinned: true,
        key: validKey,
        freedBytes: 0,
      };
    }

    const removedFromDisk = await this._removeOpfsFile(cacheDir, validKey);
    const freedBytes = existing ? existing.sizeBytes : 0;
    const existed = existing !== null || removedFromDisk;
    if (existing) {
      this._cacheEntries.delete(validKey);
    }
    if (existed) {
      await this._persistCacheIndex(cacheDir);
      if (countInvalidation) {
        this._cacheInvalidations += 1;
      }
    }

    return {
      supported: true,
      removed: existed,
      skippedPinned: false,
      key: validKey,
      freedBytes,
    };
  }

  async invalidateCacheEntry(key, options = {}) {
    return this.deleteCacheEntry(key, {
      ...options,
      countInvalidation: options.countInvalidation !== false,
    });
  }

  /**
   * Bulk-invalidates cached entries whose `versionKey` or `etag` no longer matches
   * the expected integrity values.
   *
   * Supports either:
   * - `{ expectedVersionKey?, expectedEtag?, versionKey?, etag?, kind?, prefix?, includePinned? }`
   * - `{ entries: { [key]: { versionKey?, etag? } }, includePinned? }`
   *
   * @param {object} [expectations]
   */
  async invalidateStaleCacheEntries(expectations = {}) {
    const { supported, dir: cacheDir } = await this._ensureOpfsCacheDir({ create: false });
    if (!supported) {
      return {
        supported: false,
        invalidatedCount: 0,
        freedBytes: 0,
        invalidatedKeys: [],
        skippedPinnedCount: 0,
      };
    }
    if (!cacheDir) {
      return {
        supported: true,
        invalidatedCount: 0,
        freedBytes: 0,
        invalidatedKeys: [],
        skippedPinnedCount: 0,
      };
    }
    if (!this._cacheIndexLoaded) {
      await this.syncCacheIndex();
    }

    const includePinned = expectations.includePinned !== false;
    const toRemove = [];
    let skippedPinnedCount = 0;

    if (expectations.entries && typeof expectations.entries === 'object') {
      const keys = Object.keys(expectations.entries).sort(compareAscii);
      for (const rawKey of keys) {
        let validKey;
        try {
          validKey = validateCacheKey(rawKey, this.cacheIndexFileName);
        } catch {
          continue;
        }
        const entry = this._cacheEntries.get(validKey);
        if (!entry) continue;
        const spec = expectations.entries[rawKey];
        if (!spec || typeof spec !== 'object') continue;
        const expVer = normalizeOptionalTag(spec.versionKey ?? spec.expectedVersionKey);
        const expEtag = normalizeOptionalTag(spec.etag ?? spec.expectedEtag);
        const mismatch =
          (expVer !== null && entry.versionKey !== expVer) ||
          (expEtag !== null && entry.etag !== expEtag);
        if (!mismatch) continue;
        if (entry.pinned && !includePinned) {
          skippedPinnedCount += 1;
          continue;
        }
        toRemove.push(entry);
      }
    } else {
      const expVer = normalizeOptionalTag(
        expectations.expectedVersionKey ?? expectations.versionKey
      );
      const expEtag = normalizeOptionalTag(expectations.expectedEtag ?? expectations.etag);
      const filterKind = typeof expectations.kind === 'string' ? expectations.kind : null;
      const filterPrefix = typeof expectations.prefix === 'string' ? expectations.prefix : null;

      const sortedEntries = Array.from(this._cacheEntries.values()).sort((a, b) =>
        compareAscii(a.key, b.key)
      );
      for (const entry of sortedEntries) {
        if (filterKind && entry.kind !== filterKind) continue;
        if (filterPrefix && !entry.key.startsWith(filterPrefix)) continue;
        const mismatch =
          (expVer !== null && entry.versionKey !== expVer) ||
          (expEtag !== null && entry.etag !== expEtag);
        if (!mismatch) continue;
        if (entry.pinned && !includePinned) {
          skippedPinnedCount += 1;
          continue;
        }
        toRemove.push(entry);
      }
    }

    let invalidatedCount = 0;
    let freedBytes = 0;
    const invalidatedKeys = [];

    for (const entry of toRemove) {
      await this._removeOpfsFile(cacheDir, entry.key);
      this._cacheEntries.delete(entry.key);
      invalidatedCount += 1;
      freedBytes += entry.sizeBytes;
      invalidatedKeys.push(entry.key);
    }

    if (invalidatedCount > 0) {
      this._cacheInvalidations += invalidatedCount;
      await this._persistCacheIndex(cacheDir);
    }

    return {
      supported: true,
      invalidatedCount,
      freedBytes,
      invalidatedKeys,
      skippedPinnedCount,
    };
  }

  /**
   * Deterministically evicts the least-recently-used unpinned OPFS cache entries
   * until total cache bytes are at or below `maxBytes`.
   *
   * Pinned entries (such as the active boot `.pgim` or active shader pack) are never evicted.
   * Ties in `lastAccessedMs` break deterministically by monotonic access sequence, then ASCII key.
   *
   * @param {number} [maxBytes=this.maxCacheBytes] Target maximum total cache bytes (>= 0).
   * @param {object} [options]
   * @param {Iterable<string>} [options.excludeKeys] Additional keys to skip during this eviction pass.
   * @param {boolean} [options.sync=false] Force re-sync with OPFS before evicting.
   */
  async evictToBudget(maxBytes = this.maxCacheBytes, options = {}) {
    const rawBudget = maxBytes === undefined || maxBytes === null ? this.maxCacheBytes : maxBytes;
    if (typeof rawBudget !== 'number' || !Number.isFinite(rawBudget) || rawBudget < 0) {
      throw new RangeError('Cache eviction budget maxBytes must be a finite non-negative number');
    }
    const budgetBytes = Math.floor(rawBudget);

    const { supported, dir: cacheDir } = await this._ensureOpfsCacheDir({ create: false });
    if (!supported) {
      return {
        supported: false,
        budgetBytes,
        beforeBytes: 0,
        afterBytes: 0,
        evictedEntries: 0,
        evictedBytes: 0,
        evictedKeys: [],
        skippedPinnedCount: 0,
        skippedPinnedBytes: 0,
        budgetMet: true,
      };
    }

    if (!cacheDir) {
      this._cacheEntries.clear();
      this._cacheIndexLoaded = true;
      await this._refreshQuotaEstimateIfAvailable();
      return {
        supported: true,
        budgetBytes,
        beforeBytes: 0,
        afterBytes: 0,
        evictedEntries: 0,
        evictedBytes: 0,
        evictedKeys: [],
        skippedPinnedCount: 0,
        skippedPinnedBytes: 0,
        budgetMet: true,
      };
    }

    if (!this._cacheIndexLoaded || options.sync === true) {
      await this.syncCacheIndex({ force: options.sync === true });
    }

    const excludeSet = new Set();
    if (options.excludeKeys && typeof options.excludeKeys[Symbol.iterator] === 'function') {
      for (const k of options.excludeKeys) {
        if (typeof k === 'string') {
          excludeSet.add(k);
        }
      }
    }

    let beforeBytes = 0;
    let skippedPinnedCount = 0;
    let skippedPinnedBytes = 0;
    const candidates = [];

    for (const entry of this._cacheEntries.values()) {
      beforeBytes += entry.sizeBytes;
      if (entry.pinned) {
        skippedPinnedCount += 1;
        skippedPinnedBytes += entry.sizeBytes;
        continue;
      }
      if (excludeSet.has(entry.key)) {
        continue;
      }
      candidates.push(entry);
    }

    if (beforeBytes <= budgetBytes) {
      await this._refreshQuotaEstimateIfAvailable();
      return {
        supported: true,
        budgetBytes,
        beforeBytes,
        afterBytes: beforeBytes,
        evictedEntries: 0,
        evictedBytes: 0,
        evictedKeys: [],
        skippedPinnedCount,
        skippedPinnedBytes,
        budgetMet: true,
      };
    }

    // Deterministic LRU sort: oldest lastAccessedMs -> oldest accessOrder -> ASCII key.
    candidates.sort((a, b) => {
      if (a.lastAccessedMs !== b.lastAccessedMs) {
        return a.lastAccessedMs - b.lastAccessedMs;
      }
      if (a.accessOrder !== b.accessOrder) {
        return a.accessOrder - b.accessOrder;
      }
      return compareAscii(a.key, b.key);
    });

    let currentBytes = beforeBytes;
    let evictedEntries = 0;
    let evictedBytes = 0;
    const evictedKeys = [];

    for (const candidate of candidates) {
      if (currentBytes <= budgetBytes) {
        break;
      }
      await this._removeOpfsFile(cacheDir, candidate.key);
      this._cacheEntries.delete(candidate.key);
      currentBytes = Math.max(0, currentBytes - candidate.sizeBytes);
      evictedEntries += 1;
      evictedBytes += candidate.sizeBytes;
      evictedKeys.push(candidate.key);
    }

    if (evictedEntries > 0) {
      this._cacheEvictedEntries += evictedEntries;
      this._cacheEvictedBytes += evictedBytes;
      this._recordCleanup(evictedEntries, evictedBytes);
      await this._persistCacheIndex(cacheDir);
    }

    await this._refreshQuotaEstimateIfAvailable();

    return {
      supported: true,
      budgetBytes,
      beforeBytes,
      afterBytes: currentBytes,
      evictedEntries,
      evictedBytes,
      evictedKeys,
      skippedPinnedCount,
      skippedPinnedBytes,
      budgetMet: currentBytes <= budgetBytes,
    };
  }

  /**
   * Computes an effective OPFS cache byte budget from a browser storage quota estimate
   * and optional explicit `maxBytes` / `targetUsageRatio` limits.
   *
   * @param {object|null} quotaEstimate
   * @param {object} [options]
   * @returns {number|null}
   */
  computeQuotaBudget(quotaEstimate, options = {}) {
    const explicitMax =
      typeof options.maxBytes === 'number' && Number.isFinite(options.maxBytes) && options.maxBytes >= 0
        ? Math.floor(options.maxBytes)
        : this.maxCacheBytes;

    let quotaBudget = null;
    if (
      quotaEstimate &&
      quotaEstimate.supported !== false &&
      typeof quotaEstimate.quotaBytes === 'number' &&
      Number.isFinite(quotaEstimate.quotaBytes) &&
      quotaEstimate.quotaBytes > 0
    ) {
      const targetRatio = clampUnitRatio(
        options.targetUsageRatio !== undefined
          ? options.targetUsageRatio
          : this.quotaBudgetRatio !== null
            ? this.quotaBudgetRatio
            : DEFAULT_QUOTA_BUDGET_RATIO,
        DEFAULT_QUOTA_BUDGET_RATIO
      );
      const targetTotalUsage = Math.floor(quotaEstimate.quotaBytes * targetRatio);
      const cacheBytes = this._computeCacheTotals().totalBytes;
      const reportedUsage = sanitizeNonNegativeInt(quotaEstimate.usageBytes);
      const nonCacheUsage = Math.max(0, reportedUsage - cacheBytes);
      quotaBudget = Math.max(0, targetTotalUsage - nonCacheUsage);
    }

    if (quotaBudget !== null && explicitMax !== null) {
      return Math.min(quotaBudget, explicitMax);
    }
    return quotaBudget !== null ? quotaBudget : explicitMax;
  }

  /**
   * Samples browser storage quota and evicts LRU OPFS cache entries to fit within
   * the quota-derived budget (capped by `maxCacheBytes` if configured).
   *
   * @param {object} [options]
   * @param {number} [options.targetUsageRatio] Target quota usage ratio in [0, 1].
   * @param {number} [options.maxBytes] Optional hard cap on cache bytes.
   * @param {boolean} [options.onlyUnderPressure=false] When true, only evict if pressure is 'warn' or 'critical'.
   */
  async enforceQuotaBudget(options = {}) {
    if (!this._cacheIndexLoaded) {
      await this.syncCacheIndex();
    }
    const quota = await this.getQuotaEstimate();
    const pressure = this.classifyQuotaPressure(quota, options);
    const targetBudgetBytes = this.computeQuotaBudget(quota, options);

    if (targetBudgetBytes === null) {
      return {
        supported: false,
        pressure,
        quota,
        targetBudgetBytes: null,
        evictedEntries: 0,
        evictedBytes: 0,
        evictedKeys: [],
        skippedPinnedCount: 0,
        skippedPinnedBytes: 0,
        budgetMet: true,
      };
    }

    if (
      options.onlyUnderPressure === true &&
      pressure !== QUOTA_PRESSURE_LEVELS.WARN &&
      pressure !== QUOTA_PRESSURE_LEVELS.CRITICAL
    ) {
      const totals = this._computeCacheTotals();
      return {
        supported: true,
        pressure,
        quota,
        targetBudgetBytes,
        beforeBytes: totals.totalBytes,
        afterBytes: totals.totalBytes,
        evictedEntries: 0,
        evictedBytes: 0,
        evictedKeys: [],
        skippedPinnedCount: totals.pinnedCount,
        skippedPinnedBytes: totals.pinnedBytes,
        budgetMet: totals.totalBytes <= targetBudgetBytes,
      };
    }

    const eviction = await this.evictToBudget(targetBudgetBytes, options);
    return {
      ...eviction,
      pressure,
      quota,
      targetBudgetBytes,
    };
  }

  /**
   * Returns a cloned metadata record for a tracked cache entry, or `null` if untracked.
   *
   * @param {string} key
   */
  getCacheEntryMeta(key) {
    const validKey = validateCacheKey(key, this.cacheIndexFileName);
    return cloneCacheEntryMeta(this._cacheEntries.get(validKey) || null);
  }

  /**
   * Lists cloned metadata records for all tracked cache entries.
   *
   * @param {object} [options]
   * @param {'key'|'lru'} [options.sortBy='key']
   */
  listCacheEntries(options = {}) {
    const entries = Array.from(this._cacheEntries.values());
    if (options.sortBy === 'lru') {
      entries.sort((a, b) => {
        if (a.lastAccessedMs !== b.lastAccessedMs) {
          return a.lastAccessedMs - b.lastAccessedMs;
        }
        if (a.accessOrder !== b.accessOrder) {
          return a.accessOrder - b.accessOrder;
        }
        return compareAscii(a.key, b.key);
      });
    } else {
      entries.sort((a, b) => compareAscii(a.key, b.key));
    }
    return entries.map(cloneCacheEntryMeta);
  }

  /**
   * Returns a sanitized snapshot of OPFS cache occupancy, eviction telemetry, and quota usage %.
   */
  getCacheStats() {
    const totals = this._computeCacheTotals();
    const budgetUsagePct =
      this.maxCacheBytes !== null && this.maxCacheBytes > 0
        ? Math.min(100, Math.round((totals.totalBytes / this.maxCacheBytes) * 10000) / 100)
        : null;

    let quotaUsagePct = 0;
    if (this._lastQuotaUsageRatio !== null) {
      quotaUsagePct = Math.min(100, Math.round(this._lastQuotaUsageRatio * 10000) / 100);
    } else if (budgetUsagePct !== null) {
      quotaUsagePct = budgetUsagePct;
    }

    return {
      entryCount: totals.entryCount,
      totalBytes: totals.totalBytes,
      totalFormatted: formatBytes(totals.totalBytes),
      pinnedCount: totals.pinnedCount,
      pinnedBytes: totals.pinnedBytes,
      evictedEntries: this._cacheEvictedEntries,
      evictedBytes: this._cacheEvictedBytes,
      evictedFormatted: formatBytes(this._cacheEvictedBytes),
      quotaEvictionRuns: this._quotaEvictionRuns,
      quotaUsagePct,
      budgetUsagePct,
      maxCacheBytes: this.maxCacheBytes,
      quotaPressure: this._lastQuotaPressure,
      hits: this._cacheHits,
      misses: this._cacheMisses,
      invalidations: this._cacheInvalidations,
    };
  }

  // --- IndexedDB /userdata Save Management ('playgta6-userdata' / 'files') ---

  _openUserdataDb() {
    if (!this.indexedDBFactory || typeof this.indexedDBFactory.open !== 'function') {
      return Promise.resolve(null);
    }

    return new Promise((resolve, reject) => {
      let req;
      try {
        req = this.indexedDBFactory.open(this.dbName, this.dbVersion);
      } catch (err) {
        reject(err);
        return;
      }

      req.onupgradeneeded = () => {
        const db = req.result;
        const hasStore =
          db.objectStoreNames &&
          (typeof db.objectStoreNames.contains === 'function'
            ? db.objectStoreNames.contains(this.storeName)
            : Array.from(db.objectStoreNames).includes(this.storeName));
        if (!hasStore) {
          db.createObjectStore(this.storeName, { keyPath: 'path' });
        }
      };

      req.onsuccess = () => {
        resolve(req.result);
      };

      req.onerror = () => {
        reject(req.error || new Error(`Failed to open IndexedDB database "${this.dbName}"`));
      };
    });
  }

  async _runUserdataTransaction(db, mode, operation) {
    return new Promise((resolve, reject) => {
      let settled = false;
      const finishResolve = (value) => {
        if (!settled) {
          settled = true;
          resolve(value);
        }
      };
      const finishReject = (err) => {
        if (!settled) {
          settled = true;
          reject(err);
        }
      };

      let tx;
      let req;
      try {
        tx = db.transaction(this.storeName, mode);
        const store = tx.objectStore(this.storeName);
        req = operation(store);
      } catch (err) {
        finishReject(err);
        return;
      }

      let reqResult;
      let reqDone = false;
      const waitForTxComplete = tx && 'oncomplete' in tx;

      req.onsuccess = () => {
        reqResult = req.result;
        reqDone = true;
        if (!waitForTxComplete) {
          finishResolve(reqResult);
        }
      };

      req.onerror = () => {
        finishReject(req.error || new Error(`IndexedDB request failed (${mode})`));
      };

      if (tx) {
        tx.oncomplete = () => {
          finishResolve(reqDone ? reqResult : req.result);
        };
        tx.onerror = () => {
          finishReject(tx.error || req.error || new Error(`IndexedDB transaction failed (${mode})`));
        };
        tx.onabort = () => {
          finishReject(tx.error || new Error(`IndexedDB transaction aborted (${mode})`));
        };
      }
    });
  }

  async _withUserdataDb(fn) {
    const db = await this._openUserdataDb();
    if (!db) {
      return { supported: false, value: null };
    }
    try {
      const value = await fn(db);
      return { supported: true, value };
    } finally {
      if (typeof db.close === 'function') {
        try {
          db.close();
        } catch {
          // Ignore close errors.
        }
      }
    }
  }

  async _readAllUserdataRecords(db) {
    const records = await this._runUserdataTransaction(db, 'readonly', (store) => store.getAll());
    return Array.isArray(records) ? records : [];
  }

  async listUserdataFiles() {
    const unsupported = {
      supported: false,
      dbName: this.dbName,
      storeName: this.storeName,
      fileCount: 0,
      totalBytes: 0,
      files: [],
    };

    let result;
    try {
      result = await this._withUserdataDb((db) => this._readAllUserdataRecords(db));
    } catch {
      return unsupported;
    }

    if (!result.supported) {
      return unsupported;
    }

    const files = [];
    let totalBytes = 0;
    for (const rec of result.value) {
      if (!rec || typeof rec.path !== 'string') continue;
      let validPath;
      let payload;
      try {
        validPath = validateUserdataPath(rec.path);
        payload = toUint8Array(rec.data);
      } catch {
        continue;
      }
      const updatedAt = Number.isFinite(rec.mtime)
        ? rec.mtime
        : Number.isFinite(rec.updatedAt)
          ? rec.updatedAt
          : 0;
      files.push({
        path: validPath,
        byteLength: payload.byteLength,
        updatedAt,
      });
      totalBytes += payload.byteLength;
    }

    files.sort((a, b) => compareAscii(a.path, b.path));

    return {
      supported: true,
      dbName: this.dbName,
      storeName: this.storeName,
      fileCount: files.length,
      totalBytes,
      files,
    };
  }

  async exportUserdataSnapshot() {
    const exportedAt = new Date(this._now()).toISOString();
    const unsupported = {
      format: SNAPSHOT_FORMAT,
      version: SNAPSHOT_VERSION,
      supported: false,
      exportedAt,
      dbName: this.dbName,
      storeName: this.storeName,
      fileCount: 0,
      totalBytes: 0,
      files: [],
    };

    let result;
    try {
      result = await this._withUserdataDb((db) => this._readAllUserdataRecords(db));
    } catch {
      return unsupported;
    }

    if (!result.supported) {
      return unsupported;
    }

    const entries = [];
    let totalBytes = 0;
    for (const rec of result.value) {
      if (!rec || typeof rec.path !== 'string') continue;
      let validPath;
      let payload;
      try {
        validPath = validateUserdataPath(rec.path);
        payload = toUint8Array(rec.data);
      } catch {
        continue;
      }
      const updatedAt = Number.isFinite(rec.mtime)
        ? rec.mtime
        : Number.isFinite(rec.updatedAt)
          ? rec.updatedAt
          : 0;
      entries.push({
        path: validPath,
        encoding: 'base64',
        dataBase64: encodeBase64(payload),
        byteLength: payload.byteLength,
        updatedAt,
      });
      totalBytes += payload.byteLength;
    }

    entries.sort((a, b) => compareAscii(a.path, b.path));

    return {
      format: SNAPSHOT_FORMAT,
      version: SNAPSHOT_VERSION,
      supported: true,
      exportedAt,
      dbName: this.dbName,
      storeName: this.storeName,
      fileCount: entries.length,
      totalBytes,
      files: entries,
    };
  }

  async importUserdataSnapshot(snapshot, options = {}) {
    const { overwrite = true } = options;

    let parsed = snapshot;
    if (typeof snapshot === 'string') {
      parsed = JSON.parse(snapshot);
    }

    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new TypeError('Userdata snapshot must be an object or JSON string');
    }
    if (parsed.format !== undefined && parsed.format !== SNAPSHOT_FORMAT) {
      throw new Error(`Unsupported userdata snapshot format: ${parsed.format}`);
    }
    if (parsed.version !== undefined && parsed.version !== SNAPSHOT_VERSION) {
      throw new Error(`Unsupported userdata snapshot version: ${parsed.version}`);
    }
    if (!Array.isArray(parsed.files)) {
      throw new TypeError('Userdata snapshot must contain a files array');
    }

    // Validate and decode every entry up front so invalid/traversal paths fail
    // atomically before mutating IndexedDB.
    const validatedRecords = [];
    const seenPaths = new Set();
    for (const entry of parsed.files) {
      if (!entry || typeof entry !== 'object') {
        throw new TypeError('Snapshot file entry must be an object');
      }
      const validPath = validateUserdataPath(entry.path);
      if (seenPaths.has(validPath)) {
        throw new Error(`Duplicate userdata path in snapshot: ${validPath}`);
      }
      seenPaths.add(validPath);

      if (entry.encoding !== undefined && entry.encoding !== 'base64') {
        throw new Error(`Unsupported snapshot entry encoding for ${validPath}: ${entry.encoding}`);
      }
      if (typeof entry.dataBase64 !== 'string') {
        throw new TypeError(`Snapshot entry for ${validPath} must include a base64 dataBase64 string`);
      }

      const data = decodeBase64(entry.dataBase64);
      if (
        entry.byteLength !== undefined &&
        (!Number.isInteger(entry.byteLength) || entry.byteLength !== data.byteLength)
      ) {
        throw new Error(
          `Snapshot byteLength mismatch for ${validPath}: expected ${entry.byteLength}, decoded ${data.byteLength}`
        );
      }

      const mtime = Number.isFinite(entry.updatedAt)
        ? entry.updatedAt
        : Number.isFinite(entry.mtime)
          ? entry.mtime
          : this._now();

      validatedRecords.push({ path: validPath, data, mtime });
    }

    const result = await this._withUserdataDb(async (db) => {
      let existingPaths = null;
      if (!overwrite) {
        const existing = await this._readAllUserdataRecords(db);
        existingPaths = new Set();
        for (const rec of existing) {
          if (rec && typeof rec.path === 'string') {
            existingPaths.add(rec.path);
          }
        }
      }

      let importedCount = 0;
      let skippedCount = 0;
      let importedBytes = 0;

      for (const record of validatedRecords) {
        if (existingPaths && existingPaths.has(record.path)) {
          skippedCount += 1;
          continue;
        }
        await this._runUserdataTransaction(db, 'readwrite', (store) => store.put(record));
        importedCount += 1;
        importedBytes += record.data.byteLength;
      }

      return { importedCount, skippedCount, importedBytes };
    });

    if (!result.supported) {
      return {
        supported: false,
        importedCount: 0,
        skippedCount: 0,
        importedBytes: 0,
      };
    }

    return {
      supported: true,
      ...result.value,
    };
  }

  async clearUserdata() {
    const unsupported = {
      supported: false,
      clearedCount: 0,
      freedBytes: 0,
    };

    let result;
    try {
      result = await this._withUserdataDb(async (db) => {
        const existing = await this._readAllUserdataRecords(db);
        let clearedCount = 0;
        let freedBytes = 0;
        const keys = [];

        for (const rec of existing) {
          if (!rec || typeof rec.path !== 'string') continue;
          keys.push(rec.path);
          clearedCount += 1;
          try {
            freedBytes += toUint8Array(rec.data).byteLength;
          } catch {
            // Ignore malformed payload sizes while clearing.
          }
        }

        if (keys.length > 0) {
          let usedClear = false;
          try {
            await this._runUserdataTransaction(db, 'readwrite', (store) => {
              if (typeof store.clear === 'function') {
                usedClear = true;
                return store.clear();
              }
              throw new Error('NO_STORE_CLEAR');
            });
          } catch (err) {
            if (usedClear || (err && err.message !== 'NO_STORE_CLEAR')) {
              throw err;
            }
            for (const key of keys) {
              await this._runUserdataTransaction(db, 'readwrite', (store) => store.delete(key));
            }
          }
        }

        return { clearedCount, freedBytes };
      });
    } catch {
      return unsupported;
    }

    if (!result.supported) {
      return unsupported;
    }

    this._recordCleanup(result.value.clearedCount, result.value.freedBytes);
    return {
      supported: true,
      ...result.value,
    };
  }

  // --- Combined HUD Summary & Sanitized Usage Snapshot ---

  async getSummary() {
    const [quota, opfsCache, userdata] = await Promise.all([
      this.getQuotaEstimate(),
      this.inspectOpfsCache(),
      this.listUserdataFiles(),
    ]);

    const totalManagedBytes = opfsCache.totalBytes + userdata.totalBytes;
    const pressure = this.classifyQuotaPressure(quota);
    const usageRatio =
      quota.supported && quota.quotaBytes > 0
        ? clampUnitRatio(Math.max(0, quota.usageBytes) / quota.quotaBytes, 0)
        : null;

    return {
      pressure,
      quota: {
        ...quota,
        usageRatio,
        pressure,
        warnThresholdRatio: this.warnThresholdRatio,
        criticalThresholdRatio: this.criticalThresholdRatio,
        usageFormatted: formatBytes(quota.usageBytes),
        quotaFormatted: formatBytes(quota.quotaBytes),
        availableFormatted: formatBytes(quota.availableBytes),
      },
      opfsCache: {
        supported: opfsCache.supported,
        dirName: opfsCache.dirName,
        entryCount: opfsCache.entryCount,
        totalBytes: opfsCache.totalBytes,
        totalFormatted: formatBytes(opfsCache.totalBytes),
      },
      userdata: {
        supported: userdata.supported,
        dbName: userdata.dbName,
        storeName: userdata.storeName,
        fileCount: userdata.fileCount,
        totalBytes: userdata.totalBytes,
        totalFormatted: formatBytes(userdata.totalBytes),
      },
      totalManagedBytes,
      totalManagedFormatted: formatBytes(totalManagedBytes),
      eviction: {
        evictedEntries: this._evictedEntries,
        evictedBytes: this._evictedBytes,
        evictedFormatted: formatBytes(this._evictedBytes),
        cleanupCount: this._cleanupCount,
        lastCleanupAtMs: this._lastCleanupAtMs,
      },
    };
  }

  /**
   * Returns a sanitized, JSON-serializable usage & telemetry snapshot suitable
   * for HUD badges, crash reports, and diagnostics (no raw file paths or payloads).
   */
  async getUsageSnapshot() {
    const summary = await this.getSummary();
    const stats = this.getStats();

    return {
      capturedAtMs: this._readNowMs(),
      pressure: summary.pressure,
      thresholds: {
        warnThresholdRatio: this.warnThresholdRatio,
        criticalThresholdRatio: this.criticalThresholdRatio,
      },
      quota: {
        supported: summary.quota.supported,
        usageBytes: summary.quota.usageBytes,
        quotaBytes: summary.quota.quotaBytes,
        availableBytes: summary.quota.availableBytes,
        usageRatio: summary.quota.usageRatio,
        percentUsed: summary.quota.percentUsed,
        persisted: summary.quota.persisted,
        pressure: summary.quota.pressure,
        usageFormatted: summary.quota.usageFormatted,
        quotaFormatted: summary.quota.quotaFormatted,
        availableFormatted: summary.quota.availableFormatted,
      },
      opfsCache: {
        supported: summary.opfsCache.supported,
        entryCount: summary.opfsCache.entryCount,
        totalBytes: summary.opfsCache.totalBytes,
        totalFormatted: summary.opfsCache.totalFormatted,
      },
      userdata: {
        supported: summary.userdata.supported,
        fileCount: summary.userdata.fileCount,
        totalBytes: summary.userdata.totalBytes,
        totalFormatted: summary.userdata.totalFormatted,
      },
      totalManagedBytes: summary.totalManagedBytes,
      totalManagedFormatted: summary.totalManagedFormatted,
      eviction: {
        evictedEntries: stats.evictedEntries,
        evictedBytes: stats.evictedBytes,
        evictedFormatted: stats.evictedFormatted,
        cleanupCount: stats.cleanupCount,
        lastCleanupAtMs: stats.lastCleanupAtMs,
      },
    };
  }

  /**
   * Evaluates sanitized warm-boot storage readiness by combining OPFS support,
   * persistence status, estimated quota headroom, and cached shader-pack / RPF
   * chunk coverage. Never exposes host paths, directory/database names, cache
   * keys, or raw file contents.
   *
   * @param {object} [options]
   * @param {object} [options.quota] Optional pre-fetched quota estimate override.
   * @param {object} [options.quotaEstimate] Alias for `options.quota`.
   * @param {object} [options.cacheStats] Optional cache stats/summary override.
   * @param {object} [options.cacheSummary] Alias for `options.cacheStats`.
   * @param {object} [options.opfsCache] Optional OPFS cache inspection override.
   * @param {object} [options.summary] Optional combined summary (`getSummary()` / `getUsageSnapshot()`).
   * @param {boolean} [options.opfsSupported] Optional explicit OPFS support override.
   * @param {boolean} [options.persisted] Optional explicit persistence override.
   * @param {boolean} [options.shaderPackCached] Optional explicit shader-pack coverage override.
   * @param {number} [options.rpfChunksCached] Optional explicit cached RPF chunk count override.
   * @param {number} [options.minHeadroomRatio=0.1] Minimum required free quota ratio in [0, 1] when quota is known.
   * @param {number} [options.minAvailableBytes=0] Minimum required available quota bytes.
   * @param {number} [options.minRpfChunks=1] Minimum cached RPF chunks required for warm-boot coverage.
   * @param {boolean} [options.requireShaderPack=true] Whether a cached shader pack is required for warm-boot readiness.
   * @param {boolean} [options.requirePersistence=false] Whether `persisted === true` is required for `warmBootReady`.
   */
  async getWarmBootReadiness(options = {}) {
    const opts = options && typeof options === 'object' ? options : {};
    const summaryInput = opts.summary && typeof opts.summary === 'object' ? opts.summary : null;

    const rawQuota =
      opts.quota && typeof opts.quota === 'object'
        ? opts.quota
        : opts.quotaEstimate && typeof opts.quotaEstimate === 'object'
          ? opts.quotaEstimate
          : summaryInput && summaryInput.quota && typeof summaryInput.quota === 'object'
            ? summaryInput.quota
            : null;

    let quotaSupported = false;
    let quotaBytes = 0;
    let usageBytes = 0;
    let availableBytes = 0;
    let persisted = false;

    if (rawQuota) {
      const usageRaw = rawQuota.usageBytes !== undefined ? rawQuota.usageBytes : rawQuota.usage;
      const quotaRaw = rawQuota.quotaBytes !== undefined ? rawQuota.quotaBytes : rawQuota.quota;
      const validQuotaNumber =
        typeof quotaRaw === 'number' && Number.isFinite(quotaRaw) && quotaRaw > 0;
      quotaSupported =
        rawQuota.supported === false ? false : rawQuota.supported === true || validQuotaNumber;
      quotaBytes = quotaSupported && validQuotaNumber ? Math.floor(quotaRaw) : 0;
      usageBytes = quotaSupported ? sanitizeNonNegativeInt(usageRaw) : 0;
      availableBytes = quotaSupported && quotaBytes > 0 ? Math.max(0, quotaBytes - usageBytes) : 0;
      persisted =
        opts.persisted !== undefined ? Boolean(opts.persisted) : Boolean(rawQuota.persisted);
    } else {
      const liveQuota = await this.getQuotaEstimate();
      quotaSupported = Boolean(liveQuota.supported);
      quotaBytes = quotaSupported ? sanitizeNonNegativeInt(liveQuota.quotaBytes) : 0;
      usageBytes = quotaSupported ? sanitizeNonNegativeInt(liveQuota.usageBytes) : 0;
      availableBytes = quotaSupported && quotaBytes > 0 ? Math.max(0, quotaBytes - usageBytes) : 0;
      persisted =
        opts.persisted !== undefined ? Boolean(opts.persisted) : Boolean(liveQuota.persisted);
    }

    const headroomRatio =
      quotaSupported && quotaBytes > 0
        ? clampUnitRatio(availableBytes / quotaBytes, 0)
        : null;

    const rawCache =
      opts.cacheSummary && typeof opts.cacheSummary === 'object'
        ? opts.cacheSummary
        : opts.cacheStats && typeof opts.cacheStats === 'object'
          ? opts.cacheStats
          : opts.opfsCache && typeof opts.opfsCache === 'object'
            ? opts.opfsCache
            : summaryInput && summaryInput.opfsCache && typeof summaryInput.opfsCache === 'object'
              ? summaryInput.opfsCache
              : null;

    let opfsSupported = false;
    let cachedEntries = 0;
    let cachedBytes = 0;
    let inferredShaderPacks = 0;
    let inferredRpfChunks = 0;
    let hasExplicitEntryCount = false;

    if (rawCache) {
      if (opts.opfsSupported !== undefined) {
        opfsSupported = Boolean(opts.opfsSupported);
      } else if (rawCache.opfsSupported !== undefined) {
        opfsSupported = Boolean(rawCache.opfsSupported);
      } else if (rawCache.supported !== undefined) {
        opfsSupported = Boolean(rawCache.supported);
      } else {
        opfsSupported = (await this._resolveOpfsRoot()) !== null;
      }

      if (rawCache.cachedEntries !== undefined) {
        cachedEntries = sanitizeNonNegativeInt(rawCache.cachedEntries);
        hasExplicitEntryCount = true;
      } else if (rawCache.entryCount !== undefined) {
        cachedEntries = sanitizeNonNegativeInt(rawCache.entryCount);
        hasExplicitEntryCount = true;
      } else if (rawCache.entriesCount !== undefined) {
        cachedEntries = sanitizeNonNegativeInt(rawCache.entriesCount);
        hasExplicitEntryCount = true;
      } else if (Array.isArray(rawCache.entries)) {
        cachedEntries = rawCache.entries.length;
        hasExplicitEntryCount = true;
      }

      if (rawCache.cachedBytes !== undefined) {
        cachedBytes = sanitizeNonNegativeInt(rawCache.cachedBytes);
      } else if (rawCache.totalBytes !== undefined) {
        cachedBytes = sanitizeNonNegativeInt(rawCache.totalBytes);
      } else if (rawCache.bytes !== undefined) {
        cachedBytes = sanitizeNonNegativeInt(rawCache.bytes);
      } else if (Array.isArray(rawCache.entries)) {
        for (const item of rawCache.entries) {
          if (!item || typeof item !== 'object') continue;
          cachedBytes += sanitizeNonNegativeInt(
            item.sizeBytes !== undefined ? item.sizeBytes : item.size
          );
        }
      }

      if (Array.isArray(rawCache.entries)) {
        for (const item of rawCache.entries) {
          if (!item || typeof item !== 'object') continue;
          const key = typeof item.key === 'string' ? item.key : typeof item.name === 'string' ? item.name : '';
          const tracked = key ? this._cacheEntries.get(key) : null;
          const kind =
            typeof item.kind === 'string'
              ? normalizeCacheKind(item.kind, key)
              : tracked
                ? tracked.kind
                : inferCacheEntryKind(key);
          if (kind === CACHE_ENTRY_KINDS.SHADER_PACK) {
            inferredShaderPacks += 1;
          } else if (kind === CACHE_ENTRY_KINDS.RANGE_CHUNK) {
            inferredRpfChunks += 1;
          }
        }
      } else if (
        this._cacheEntries.size > 0 &&
        cachedEntries === this._cacheEntries.size &&
        rawCache.shaderPackCached === undefined &&
        rawCache.shaderPackCount === undefined &&
        rawCache.shaderPacksCached === undefined &&
        rawCache.rpfChunksCached === undefined &&
        rawCache.rangeChunkCount === undefined &&
        rawCache.chunkCount === undefined
      ) {
        for (const entry of this._cacheEntries.values()) {
          if (entry.kind === CACHE_ENTRY_KINDS.SHADER_PACK) {
            inferredShaderPacks += 1;
          } else if (entry.kind === CACHE_ENTRY_KINDS.RANGE_CHUNK) {
            inferredRpfChunks += 1;
          }
        }
      }
    } else {
      const liveOpfsCache = await this.inspectOpfsCache();
      opfsSupported =
        opts.opfsSupported !== undefined
          ? Boolean(opts.opfsSupported)
          : Boolean(liveOpfsCache.supported);
      if (liveOpfsCache.supported) {
        cachedEntries = sanitizeNonNegativeInt(liveOpfsCache.entryCount);
        cachedBytes = sanitizeNonNegativeInt(liveOpfsCache.totalBytes);
        hasExplicitEntryCount = true;
        for (const entry of liveOpfsCache.entries) {
          const tracked = this._cacheEntries.get(entry.name);
          const kind = tracked ? tracked.kind : inferCacheEntryKind(entry.name);
          if (kind === CACHE_ENTRY_KINDS.SHADER_PACK) {
            inferredShaderPacks += 1;
          } else if (kind === CACHE_ENTRY_KINDS.RANGE_CHUNK) {
            inferredRpfChunks += 1;
          }
        }
      }
    }

    let shaderPackCached = inferredShaderPacks > 0;
    if (opts.shaderPackCached !== undefined) {
      shaderPackCached = Boolean(opts.shaderPackCached);
    } else if (rawCache) {
      if (rawCache.shaderPackCached !== undefined) {
        shaderPackCached = Boolean(rawCache.shaderPackCached);
      } else if (rawCache.shaderPackCount !== undefined) {
        shaderPackCached = sanitizeNonNegativeInt(rawCache.shaderPackCount) > 0;
      } else if (rawCache.shaderPacksCached !== undefined) {
        shaderPackCached = sanitizeNonNegativeInt(rawCache.shaderPacksCached) > 0;
      }
    }

    let rpfChunksCached = inferredRpfChunks;
    if (opts.rpfChunksCached !== undefined) {
      rpfChunksCached = sanitizeNonNegativeInt(opts.rpfChunksCached);
    } else if (rawCache) {
      if (rawCache.rpfChunksCached !== undefined) {
        rpfChunksCached = sanitizeNonNegativeInt(rawCache.rpfChunksCached);
      } else if (rawCache.rangeChunkCount !== undefined) {
        rpfChunksCached = sanitizeNonNegativeInt(rawCache.rangeChunkCount);
      } else if (rawCache.chunkCount !== undefined) {
        rpfChunksCached = sanitizeNonNegativeInt(rawCache.chunkCount);
      }
    }

    if (!hasExplicitEntryCount && (shaderPackCached || rpfChunksCached > 0)) {
      cachedEntries = rpfChunksCached + (shaderPackCached ? 1 : 0);
    }

    const minHeadroomRatio = clampUnitRatio(
      opts.minHeadroomRatio,
      DEFAULT_WARM_BOOT_MIN_HEADROOM_RATIO
    );
    const minAvailableBytes = sanitizeNonNegativeInt(opts.minAvailableBytes);
    const minRpfChunks =
      opts.minRpfChunks !== undefined
        ? sanitizeNonNegativeInt(opts.minRpfChunks)
        : DEFAULT_WARM_BOOT_MIN_RPF_CHUNKS;
    const requireShaderPack = opts.requireShaderPack !== false;
    const requirePersistence = opts.requirePersistence === true;

    const hasSufficientHeadroom =
      headroomRatio !== null
        ? headroomRatio >= minHeadroomRatio && availableBytes >= minAvailableBytes
        : minAvailableBytes === 0;

    let warmBootReady = false;
    let recommendation = WARM_BOOT_RECOMMENDATIONS.READY;

    if (!opfsSupported) {
      warmBootReady = false;
      recommendation = WARM_BOOT_RECOMMENDATIONS.OPFS_UNSUPPORTED;
    } else if (!hasSufficientHeadroom) {
      warmBootReady = false;
      recommendation = WARM_BOOT_RECOMMENDATIONS.LOW_QUOTA_HEADROOM;
    } else if (cachedEntries === 0 && !shaderPackCached && rpfChunksCached === 0) {
      warmBootReady = false;
      recommendation = WARM_BOOT_RECOMMENDATIONS.WARM_CACHE_EMPTY;
    } else if (requireShaderPack && !shaderPackCached) {
      warmBootReady = false;
      recommendation = WARM_BOOT_RECOMMENDATIONS.CACHE_SHADER_PACK;
    } else if (rpfChunksCached < minRpfChunks) {
      warmBootReady = false;
      recommendation = WARM_BOOT_RECOMMENDATIONS.CACHE_RPF_CHUNKS;
    } else if (!persisted) {
      warmBootReady = !requirePersistence;
      recommendation = WARM_BOOT_RECOMMENDATIONS.REQUEST_PERSISTENCE;
    } else {
      warmBootReady = true;
      recommendation = WARM_BOOT_RECOMMENDATIONS.READY;
    }

    return {
      opfsSupported,
      persisted,
      quotaBytes,
      usageBytes,
      availableBytes,
      headroomRatio,
      cachedEntries,
      cachedBytes,
      shaderPackCached,
      rpfChunksCached,
      warmBootReady,
      recommendation,
    };
  }

  /**
   * Attaches or detaches an external OPFS chunk cache instance (e.g. OpfsChunkCache)
   * for safe quota-pressure eviction delegation.
   *
   * @param {object|null} opfsCache
   * @returns {object|null}
   */
  attachOpfsCache(opfsCache) {
    this.opfsCache = opfsCache && typeof opfsCache === 'object' ? opfsCache : null;
    return this.opfsCache;
  }

  /**
   * Computes a sanitized, deterministic fast-boot budget and eviction plan combining
   * W243 OPFS warm-boot readiness, browser storage quota headroom (`quota - usage`),
   * and optional memory pressure signals (#1077).
   *
   * Recommends a 192 MiB slab LRU cap when headroom is healthy and scales down
   * gracefully under medium/high/critical storage or memory pressure.
   *
   * @param {object} [options]
   * @returns {Promise<{
   *   warmCacheReady: boolean,
   *   recommendedSlabMaxBytes: number,
   *   recommendedOpfsBudgetBytes: number,
   *   quotaHeadroomBytes: number,
   *   pressureLevel: 'low'|'medium'|'high'|'critical',
   *   shouldEvictColdChunks: boolean
   * }>}
   */
  async getFastBootPlan(options = {}) {
    const opts = options && typeof options === 'object' ? options : {};

    let readiness =
      opts.warmBootReadiness && typeof opts.warmBootReadiness === 'object'
        ? opts.warmBootReadiness
        : null;

    if (!readiness) {
      const readinessOpts = { ...opts };
      if (
        !readinessOpts.cacheSummary &&
        !readinessOpts.cacheStats &&
        !readinessOpts.opfsCache &&
        !readinessOpts.summary &&
        this.opfsCache &&
        typeof this.opfsCache.getStats === 'function'
      ) {
        const root = await this._resolveOpfsRoot();
        if (!root) {
          try {
            const attachedStats = this.opfsCache.getStats();
            if (attachedStats && typeof attachedStats === 'object') {
              readinessOpts.cacheStats = attachedStats;
            }
          } catch {
            // Ignore attached cache stats errors and fall back cleanly.
          }
        }
      }
      readiness = await this.getWarmBootReadiness(readinessOpts);
    }

    const opfsSupported = Boolean(readiness.opfsSupported);
    const warmCacheReady = Boolean(readiness.warmBootReady);
    const quotaBytes = sanitizeNonNegativeInt(readiness.quotaBytes);
    const usageBytes = sanitizeNonNegativeInt(readiness.usageBytes);
    const cachedBytes = sanitizeNonNegativeInt(readiness.cachedBytes);
    const quotaHeadroomBytes = quotaBytes > 0 ? Math.max(0, quotaBytes - usageBytes) : 0;

    const thresholds = normalizeQuotaThresholds({
      warnThresholdRatio:
        opts.warnThresholdRatio !== undefined ? opts.warnThresholdRatio : this.warnThresholdRatio,
      criticalThresholdRatio:
        opts.criticalThresholdRatio !== undefined
          ? opts.criticalThresholdRatio
          : this.criticalThresholdRatio,
    });
    const highThresholdRatio = thresholds.warnThresholdRatio;
    const criticalThresholdRatio = thresholds.criticalThresholdRatio;
    const mediumThresholdRatio = Math.min(
      clampUnitRatio(opts.mediumThresholdRatio, DEFAULT_FAST_BOOT_MEDIUM_THRESHOLD_RATIO),
      highThresholdRatio
    );

    let storagePressure = FAST_BOOT_PRESSURE_LEVELS.MEDIUM;
    if (quotaBytes > 0) {
      const usageRatio = clampUnitRatio(usageBytes / quotaBytes, 0);
      if (usageRatio >= criticalThresholdRatio) {
        storagePressure = FAST_BOOT_PRESSURE_LEVELS.CRITICAL;
      } else if (usageRatio >= highThresholdRatio) {
        storagePressure = FAST_BOOT_PRESSURE_LEVELS.HIGH;
      } else if (usageRatio >= mediumThresholdRatio) {
        storagePressure = FAST_BOOT_PRESSURE_LEVELS.MEDIUM;
      } else {
        storagePressure = FAST_BOOT_PRESSURE_LEVELS.LOW;
      }
    }

    const memoryPressure = classifyMemoryPressureForFastBoot(opts);
    const pressureLevel = maxFastBootPressure(storagePressure, memoryPressure);

    const healthySlabMaxBytes =
      typeof opts.healthySlabMaxBytes === 'number' &&
      Number.isFinite(opts.healthySlabMaxBytes) &&
      opts.healthySlabMaxBytes > 0
        ? Math.floor(opts.healthySlabMaxBytes)
        : DEFAULT_FAST_BOOT_SLAB_MAX_BYTES;
    const minSlabMaxBytes = Math.min(
      healthySlabMaxBytes,
      typeof opts.minSlabMaxBytes === 'number' &&
        Number.isFinite(opts.minSlabMaxBytes) &&
        opts.minSlabMaxBytes >= 0
        ? Math.floor(opts.minSlabMaxBytes)
        : MIN_FAST_BOOT_SLAB_MAX_BYTES
    );

    let slabScale = 1;
    if (pressureLevel === FAST_BOOT_PRESSURE_LEVELS.MEDIUM) {
      slabScale = 2 / 3;
    } else if (pressureLevel === FAST_BOOT_PRESSURE_LEVELS.HIGH) {
      slabScale = 0.5;
    } else if (pressureLevel === FAST_BOOT_PRESSURE_LEVELS.CRITICAL) {
      slabScale = 0.25;
    }
    const recommendedSlabMaxBytes = Math.max(
      minSlabMaxBytes,
      Math.min(healthySlabMaxBytes, Math.round(healthySlabMaxBytes * slabScale))
    );

    const maxOpfsBudgetBytes =
      typeof opts.maxOpfsBudgetBytes === 'number' &&
      Number.isFinite(opts.maxOpfsBudgetBytes) &&
      opts.maxOpfsBudgetBytes >= 0
        ? Math.floor(opts.maxOpfsBudgetBytes)
        : this.maxCacheBytes !== null
          ? this.maxCacheBytes
          : DEFAULT_FAST_BOOT_OPFS_BUDGET_BYTES;

    let recommendedOpfsBudgetBytes = 0;
    if (opfsSupported) {
      let opfsScale = 1;
      let headroomFactor = 0.5;
      let quotaCapRatio = DEFAULT_QUOTA_BUDGET_RATIO;
      let includeCachedInHeadroom = true;

      if (pressureLevel === FAST_BOOT_PRESSURE_LEVELS.MEDIUM) {
        opfsScale = 0.5;
        headroomFactor = 0.35;
        quotaCapRatio = 0.45;
        includeCachedInHeadroom = true;
      } else if (pressureLevel === FAST_BOOT_PRESSURE_LEVELS.HIGH) {
        opfsScale = 0.25;
        headroomFactor = 0.25;
        quotaCapRatio = 0.25;
        includeCachedInHeadroom = false;
      } else if (pressureLevel === FAST_BOOT_PRESSURE_LEVELS.CRITICAL) {
        opfsScale = 0.125;
        headroomFactor = 0.1;
        quotaCapRatio = 0.1;
        includeCachedInHeadroom = false;
      }

      const scaledMax = Math.floor(maxOpfsBudgetBytes * opfsScale);
      if (quotaBytes > 0) {
        const headroomBase = includeCachedInHeadroom
          ? quotaHeadroomBytes + cachedBytes
          : quotaHeadroomBytes;
        const headroomCap = Math.floor(headroomBase * headroomFactor);
        const quotaCap = Math.floor(quotaBytes * quotaCapRatio);
        recommendedOpfsBudgetBytes = Math.max(0, Math.min(scaledMax, headroomCap, quotaCap));
      } else {
        recommendedOpfsBudgetBytes = Math.max(0, scaledMax);
      }
    }

    const shouldEvictColdChunks =
      opfsSupported &&
      (pressureLevel === FAST_BOOT_PRESSURE_LEVELS.HIGH ||
        pressureLevel === FAST_BOOT_PRESSURE_LEVELS.CRITICAL ||
        (cachedBytes > recommendedOpfsBudgetBytes && cachedBytes > 0));

    return {
      warmCacheReady,
      recommendedSlabMaxBytes,
      recommendedOpfsBudgetBytes,
      quotaHeadroomBytes,
      pressureLevel,
      shouldEvictColdChunks,
    };
  }

  /**
   * Executes a safe quota-pressure eviction pass against any attached OPFS cache
   * (`evictToTarget` -> `prune` -> `evictOldest` -> `evictToBudget`), falling back
   * to this manager's built-in OPFS LRU cache (`evictToBudget`) when no external
   * cache is attached. Never touches IndexedDB `/userdata` player saves.
   *
   * Always increments `quotaEvictionRuns` in stats telemetry.
   *
   * @param {number} [targetFreeBytes] Bytes of cold OPFS cache to reclaim.
   * @param {object} [options]
   * @param {object|null} [options.opfsCache] Optional per-call OPFS cache override.
   * @returns {Promise<{ evictedBytes: number, evictedEntries: number, ok: boolean }>}
   */
  async runQuotaPressureEviction(targetFreeBytes, options = {}) {
    const opts = options && typeof options === 'object' ? options : {};
    const cache =
      opts.opfsCache && typeof opts.opfsCache === 'object' ? opts.opfsCache : this.opfsCache;

    this._quotaEvictionRuns += 1;

    let resolvedTargetBytes;
    if (targetFreeBytes === undefined) {
      const plan = await this.getFastBootPlan(opts);
      let currentBytes = 0;
      if (cache && typeof cache.getStats === 'function') {
        try {
          let stats = cache.getStats();
          if (stats && stats.initialized === false && typeof cache.init === 'function') {
            await cache.init();
            stats = cache.getStats();
          }
          currentBytes = sanitizeNonNegativeInt(
            stats && stats.totalBytes !== undefined ? stats.totalBytes : stats && stats.cachedBytes
          );
        } catch {
          currentBytes = 0;
        }
      } else {
        if (!this._cacheIndexLoaded) {
          await this.syncCacheIndex();
        }
        currentBytes = this._computeCacheTotals().totalBytes;
      }
      resolvedTargetBytes = Math.max(0, currentBytes - plan.recommendedOpfsBudgetBytes);
    } else {
      resolvedTargetBytes = sanitizeNonNegativeInt(targetFreeBytes);
    }

    if (resolvedTargetBytes === 0) {
      return {
        evictedBytes: 0,
        evictedEntries: 0,
        ok: true,
      };
    }

    // Delegate safely to attached OPFS cache without ever touching /userdata saves.
    if (cache) {
      try {
        let handled = false;
        let rawResult;
        if (typeof cache.evictToTarget === 'function') {
          handled = true;
          rawResult = await cache.evictToTarget(resolvedTargetBytes);
        } else if (typeof cache.prune === 'function') {
          handled = true;
          rawResult = await cache.prune(resolvedTargetBytes);
        } else if (typeof cache.evictOldest === 'function') {
          handled = true;
          rawResult = await cache.evictOldest(resolvedTargetBytes);
        } else if (typeof cache.evictToBudget === 'function') {
          handled = true;
          let currentBytes = 0;
          if (typeof cache.getStats === 'function') {
            let stats = cache.getStats();
            if (stats && stats.initialized === false && typeof cache.init === 'function') {
              await cache.init();
              stats = cache.getStats();
            }
            currentBytes = sanitizeNonNegativeInt(
              stats && stats.totalBytes !== undefined
                ? stats.totalBytes
                : stats && stats.cachedBytes
            );
          }
          const targetBudget = Math.max(0, currentBytes - resolvedTargetBytes);
          rawResult = await cache.evictToBudget(targetBudget);
        }

        if (handled) {
          const normalized = normalizeDelegatedEvictionResult(rawResult);
          if (normalized.evictedEntries > 0 || normalized.evictedBytes > 0) {
            this._cacheEvictedEntries += normalized.evictedEntries;
            this._cacheEvictedBytes += normalized.evictedBytes;
            this._recordCleanup(normalized.evictedEntries, normalized.evictedBytes);
          }
          return normalized;
        }
      } catch {
        return {
          evictedBytes: 0,
          evictedEntries: 0,
          ok: false,
        };
      }
    }

    // Fallback: evict cold unpinned OPFS entries managed directly by BrowserStorageManager.
    try {
      if (!this._cacheIndexLoaded) {
        const synced = await this.syncCacheIndex();
        if (!synced.supported) {
          return {
            evictedBytes: 0,
            evictedEntries: 0,
            ok: false,
          };
        }
      }
      const currentBytes = this._computeCacheTotals().totalBytes;
      const targetBudget = Math.max(0, currentBytes - resolvedTargetBytes);
      const ev = await this.evictToBudget(targetBudget);
      return {
        evictedBytes: sanitizeNonNegativeInt(ev.evictedBytes),
        evictedEntries: sanitizeNonNegativeInt(ev.evictedEntries),
        ok: Boolean(ev.supported),
      };
    } catch {
      return {
        evictedBytes: 0,
        evictedEntries: 0,
        ok: false,
      };
    }
  }
}
