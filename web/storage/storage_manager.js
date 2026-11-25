// Browser storage, quota, OPFS range-cache, and IndexedDB /userdata save manager.
// Matches OPFS directory ('playgta6-cache') from web/workers/range_cache.js and
// IndexedDB database/store ('playgta6-userdata' / 'files') + /userdata/ path rules
// from web/workers/userdata.js. Zero external dependencies; browser & Node testable.

export const DEFAULT_OPFS_CACHE_DIR = 'playgta6-cache';
export const DEFAULT_USERDATA_DB_NAME = 'playgta6-userdata';
export const DEFAULT_USERDATA_STORE_NAME = 'files';
export const DEFAULT_USERDATA_DB_VERSION = 1;
export const USERDATA_PREFIX = '/userdata/';
export const SNAPSHOT_FORMAT = 'playgta6-userdata-snapshot';
export const SNAPSHOT_VERSION = 1;
export const DEFAULT_WARN_THRESHOLD_RATIO = 0.75;
export const DEFAULT_CRITICAL_THRESHOLD_RATIO = 0.9;
export const QUOTA_PRESSURE_LEVELS = Object.freeze({
  OK: 'ok',
  WARN: 'warn',
  CRITICAL: 'critical',
  UNKNOWN: 'unknown',
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

    this._evictedEntries = 0;
    this._evictedBytes = 0;
    this._cleanupCount = 0;
    this._lastCleanupAtMs = null;
    this._lastQuotaPressure = QUOTA_PRESSURE_LEVELS.UNKNOWN;
    this._lastQuotaUsageRatio = null;
    this._lastQuotaSampleAtMs = null;
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
    this._lastCleanupAtMs = null;
    this._lastQuotaPressure = QUOTA_PRESSURE_LEVELS.UNKNOWN;
    this._lastQuotaUsageRatio = null;
    this._lastQuotaSampleAtMs = null;
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
      for await (const [name, handle] of iterateDirEntries(cacheDir)) {
        if (!handle || typeof name !== 'string') continue;
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
}
