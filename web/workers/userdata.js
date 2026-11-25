'use strict';

// IndexedDB persistence for Wasm /userdata/ saves and settings.
// Enforces strict /userdata/ path confinement and safe restore timeouts so a
// broken or hung IndexedDB never blocks runtime boot.
// Compatible with classic Web Workers (importScripts) and Node CommonJS/ESM tests.

const DEFAULT_DB_NAME = 'playgta6-userdata';
const DEFAULT_STORE_NAME = 'files';
const DEFAULT_RESTORE_TIMEOUT_MS = 8000;
const USERDATA_PREFIX = '/userdata/';
const ARCHIVE_FORMAT = 'playgta6-userdata-archive-v1';
const ARCHIVE_VERSION = 1;
const DEFAULT_MAX_FILE_BYTES = 16 * 1024 * 1024;
const DEFAULT_MAX_TOTAL_BYTES = 256 * 1024 * 1024;
const DEFAULT_MAX_FILE_COUNT = 4096;
const TEMP_SUFFIX = '.tmp';
const QUARANTINE_PREFIX = '__quarantine__/';
const DEFAULT_MAX_QUARANTINE_FILES = 32;
const DEFAULT_MAX_QUARANTINE_BYTES = 8 * 1024 * 1024;
const DEFAULT_MAX_QUOTA_CLEANUP_STEPS = 64;
const MAX_QUARANTINE_RECORD_BYTES = 256 * 1024;
const DEFAULT_MAX_COALESCED_PER_KEY = 256;
const DEFAULT_MAX_PENDING_COALESCED_KEYS = 256;
const MAX_CONTAINER_LIST_CACHE_ENTRIES = 64;

const BASE64_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const BASE64_LOOKUP = (() => {
  const table = new Int16Array(128).fill(-1);
  for (let i = 0; i < BASE64_ALPHABET.length; i += 1) {
    table[BASE64_ALPHABET.charCodeAt(i)] = i;
  }
  return table;
})();

function compareAscii(a, b) {
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}

function validateUserdataPath(path) {
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

function validateUserdataPrefix(prefix) {
  if (typeof prefix !== 'string' || prefix.length === 0) {
    throw new Error('Userdata prefix must be a non-empty string');
  }
  if (prefix.includes('\\') || prefix.includes('\0')) {
    throw new Error(`Invalid userdata prefix (forbidden character): ${prefix}`);
  }
  if (!prefix.startsWith(USERDATA_PREFIX)) {
    throw new Error(`Userdata prefix must start with ${USERDATA_PREFIX}: ${prefix}`);
  }
  if (prefix === USERDATA_PREFIX) {
    return USERDATA_PREFIX;
  }

  const trimmed = prefix.endsWith('/') ? prefix.slice(0, -1) : prefix;
  const rel = trimmed.slice(USERDATA_PREFIX.length);
  const segments = rel.split('/');
  for (const seg of segments) {
    if (!seg || seg === '.' || seg === '..') {
      throw new Error(`Invalid userdata prefix segment in: ${prefix}`);
    }
  }
  return `${USERDATA_PREFIX}${segments.join('/')}/`;
}

function toUint8Array(bytes) {
  if (bytes instanceof Uint8Array) {
    return bytes.slice();
  }
  if (ArrayBuffer.isView(bytes)) {
    return new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength).slice();
  }
  if (bytes instanceof ArrayBuffer) {
    return new Uint8Array(bytes.slice(0));
  }
  if (typeof bytes === 'string') {
    return new TextEncoder().encode(bytes);
  }
  throw new TypeError('Userdata file payload must be a Uint8Array, ArrayBufferView, ArrayBuffer, or string');
}

function encodeBase64(input) {
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
        BASE64_ALPHABET[triplet & 0x3f],
    );
  }

  const rem = len - i;
  if (rem === 1) {
    const b0 = bytes[i];
    chunks.push(
      BASE64_ALPHABET[(b0 >>> 2) & 0x3f] +
        BASE64_ALPHABET[(b0 << 4) & 0x3f] +
        '==',
    );
  } else if (rem === 2) {
    const b0 = bytes[i];
    const b1 = bytes[i + 1];
    chunks.push(
      BASE64_ALPHABET[(b0 >>> 2) & 0x3f] +
        BASE64_ALPHABET[((b0 << 4) | (b1 >>> 4)) & 0x3f] +
        BASE64_ALPHABET[(b1 << 2) & 0x3f] +
        '=',
    );
  }

  return chunks.join('');
}

function decodeBase64(base64) {
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
      throw new Error('Invalid base64 character');
    }

    const triplet = (v0 << 18) | (v1 << 12) | (v2 << 6) | v3;
    if (outPos < outLen) out[outPos++] = (triplet >>> 16) & 0xff;
    if (outPos < outLen) out[outPos++] = (triplet >>> 8) & 0xff;
    if (outPos < outLen) out[outPos++] = triplet & 0xff;
  }

  return out;
}

async function ensureParentDirs(fsAdapter, filePath) {
  if (!fsAdapter) return;
  const lastSlash = filePath.lastIndexOf('/');
  if (lastSlash <= 0) return;
  const parentDir = filePath.slice(0, lastSlash);

  if (typeof fsAdapter.mkdirTree === 'function') {
    await fsAdapter.mkdirTree(parentDir);
    return;
  }

  if (typeof fsAdapter.mkdir === 'function') {
    const parts = parentDir.split('/').filter(Boolean);
    let current = '';
    for (const part of parts) {
      current += `/${part}`;
      try {
        await fsAdapter.mkdir(current);
      } catch {
        // Ignore already-existing directory errors (e.g. Emscripten FS EEXIST).
      }
    }
  }
}

function nowMs() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

function parseQuotaLimit(value, fallback, label) {
  if (value === undefined) {
    return fallback;
  }
  if (value === Infinity) {
    return Infinity;
  }
  if (typeof value === 'number' && Number.isFinite(value) && value >= 0) {
    return Math.floor(value);
  }
  throw new RangeError(`${label} must be a non-negative finite number or Infinity`);
}

function createQuotaError(message, details) {
  const err = new Error(message);
  err.name = 'QuotaExceededError';
  err.code = 'QUOTA_EXCEEDED';
  if (details && typeof details === 'object') {
    err.details = details;
  }
  return err;
}

function createWriteVerifyError(message, details) {
  const err = new Error(message);
  err.name = 'WriteVerifyError';
  err.code = 'WRITE_VERIFY_FAILED';
  if (details && typeof details === 'object') {
    err.details = details;
  }
  return err;
}

function isQuotaExceededError(err) {
  if (!err || typeof err !== 'object') {
    return false;
  }
  if (
    err.name === 'QuotaExceededError' ||
    err.name === 'NS_ERROR_DOM_QUOTA_REACHED' ||
    err.code === 'QUOTA_EXCEEDED' ||
    err.code === 22 ||
    err.code === 1014
  ) {
    return true;
  }
  if (typeof err.message === 'string' && /quota\s*exceeded|storage\s*quota/i.test(err.message)) {
    return true;
  }
  return false;
}

function normalizeQuotaError(err, fallbackMessage) {
  if (err && typeof err === 'object') {
    try {
      if (err.code === undefined) {
        err.code = 'QUOTA_EXCEEDED';
      }
      if (err.name !== 'QuotaExceededError') {
        err.name = 'QuotaExceededError';
      }
    } catch {
      // Ignore read-only properties on host DOMException instances.
    }
    return err;
  }
  return createQuotaError(fallbackMessage || 'Storage quota exceeded');
}

function computeChecksum(input) {
  const bytes = input instanceof Uint8Array ? input : toUint8Array(input);
  let hash = 0x811c9dc5;
  for (let i = 0; i < bytes.byteLength; i += 1) {
    hash ^= bytes[i];
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

function isTempPath(path) {
  return typeof path === 'string' && path.length > TEMP_SUFFIX.length && path.endsWith(TEMP_SUFFIX);
}

function makeTempPath(validPath) {
  return `${validPath}${TEMP_SUFFIX}`;
}

function isQuarantinePath(path) {
  return typeof path === 'string' && path.startsWith(QUARANTINE_PREFIX);
}

function sanitizeReasonToken(reason) {
  const raw = typeof reason === 'string' && reason.length > 0 ? reason : 'corrupt';
  return raw.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 40) || 'corrupt';
}

function buildQuarantineKey(validPath, reason, ts, seq) {
  const rel = validPath.startsWith(USERDATA_PREFIX)
    ? validPath.slice(USERDATA_PREFIX.length)
    : validPath.replace(/^\/+/, '');
  const safeTs = Number.isFinite(ts) && ts >= 0 ? Math.floor(ts) : 0;
  const safeSeq = Number.isInteger(seq) && seq >= 0 ? seq : 0;
  return `${QUARANTINE_PREFIX}${safeTs}-${safeSeq}-${sanitizeReasonToken(reason)}/${rel}`;
}

function parseQuarantineMetadata(rec) {
  const key = typeof rec.path === 'string' ? rec.path : '';
  const body = key.startsWith(QUARANTINE_PREFIX) ? key.slice(QUARANTINE_PREFIX.length) : key;
  const slashIdx = body.indexOf('/');
  const header = slashIdx === -1 ? body : body.slice(0, slashIdx);
  const rel = slashIdx === -1 ? '' : body.slice(slashIdx + 1);
  const headerParts = header.split('-');
  const parsedTs = headerParts.length > 0 ? Number(headerParts[0]) : 0;
  const parsedReason = headerParts.length >= 3 ? headerParts.slice(2).join('-') : 'corrupt';

  const originalPath =
    typeof rec.originalPath === 'string' && rec.originalPath.length > 0
      ? rec.originalPath
      : rel.length > 0
        ? `${USERDATA_PREFIX}${rel}`
        : USERDATA_PREFIX;
  const reason =
    typeof rec.reason === 'string' && rec.reason.length > 0 ? rec.reason : parsedReason;
  const quarantinedAt = Number.isFinite(rec.quarantinedAt)
    ? rec.quarantinedAt
    : Number.isFinite(rec.mtime)
      ? rec.mtime
      : Number.isFinite(parsedTs)
        ? parsedTs
        : 0;
  let size = 0;
  try {
    size = getRecordByteLength(rec.data);
  } catch {
    size = 0;
  }
  return {
    quarantineKey: key,
    originalPath,
    reason,
    quarantinedAt,
    size,
  };
}

function validateStoredRecord(rec) {
  if (!rec || typeof rec !== 'object') {
    return { ok: false, reason: 'invalid_record' };
  }
  if (typeof rec.path !== 'string') {
    return { ok: false, reason: 'invalid_path' };
  }
  let validPath;
  try {
    validPath = validateUserdataPath(rec.path);
  } catch {
    return { ok: false, reason: 'invalid_path' };
  }
  if (
    rec.mtime !== undefined &&
    (typeof rec.mtime !== 'number' || !Number.isFinite(rec.mtime) || rec.mtime < 0)
  ) {
    return { ok: false, reason: 'invalid_metadata', path: validPath };
  }
  let data;
  try {
    data = toUint8Array(rec.data);
  } catch {
    return { ok: false, reason: 'invalid_payload', path: validPath };
  }
  if (rec.size !== undefined) {
    if (!Number.isInteger(rec.size) || rec.size < 0 || rec.size !== data.byteLength) {
      return { ok: false, reason: 'size_mismatch', path: validPath, data };
    }
  }
  if (rec.checksum !== undefined) {
    if (
      typeof rec.checksum !== 'number' ||
      !Number.isFinite(rec.checksum) ||
      (rec.checksum >>> 0) !== computeChecksum(data)
    ) {
      return { ok: false, reason: 'checksum_mismatch', path: validPath, data };
    }
  }
  const mtime = typeof rec.mtime === 'number' && Number.isFinite(rec.mtime) ? rec.mtime : 0;
  return {
    ok: true,
    path: validPath,
    data,
    mtime,
  };
}

function getRecordByteLength(data) {
  if (data instanceof Uint8Array) {
    return data.byteLength;
  }
  if (ArrayBuffer.isView(data)) {
    return data.byteLength;
  }
  if (data instanceof ArrayBuffer) {
    return data.byteLength;
  }
  if (typeof data === 'string') {
    return new TextEncoder().encode(data).byteLength;
  }
  throw new TypeError('Userdata file payload must be a Uint8Array, ArrayBufferView, ArrayBuffer, or string');
}

function buildMetadataEntry(validPath, size, mtime) {
  const rel = validPath.slice(USERDATA_PREFIX.length);
  const firstSlash = rel.indexOf('/');
  const containerPrefix =
    firstSlash === -1 ? USERDATA_PREFIX : `${USERDATA_PREFIX}${rel.slice(0, firstSlash)}/`;
  const lastSlash = validPath.lastIndexOf('/');
  const parentDir = validPath.slice(0, lastSlash + 1);
  const name = validPath.slice(lastSlash + 1);
  return {
    path: validPath,
    size,
    mtime: Number.isFinite(mtime) ? mtime : 0,
    containerPrefix,
    parentDir,
    name,
  };
}

function ancestorPrefixesOf(validPath) {
  const prefixes = [USERDATA_PREFIX];
  const rel = validPath.slice(USERDATA_PREFIX.length);
  const segments = rel.split('/');
  let current = USERDATA_PREFIX;
  for (let i = 0; i < segments.length - 1; i += 1) {
    current += `${segments[i]}/`;
    prefixes.push(current);
  }
  return prefixes;
}

function cloneMetadataEntry(meta) {
  return {
    path: meta.path,
    size: meta.size,
    mtime: meta.mtime,
    containerPrefix: meta.containerPrefix,
    parentDir: meta.parentDir,
    name: meta.name,
  };
}

class UserdataStore {
  constructor(options = {}) {
    this.dbName = typeof options.dbName === 'string' && options.dbName ? options.dbName : DEFAULT_DB_NAME;
    this.storeName =
      typeof options.storeName === 'string' && options.storeName ? options.storeName : DEFAULT_STORE_NAME;
    this.dbVersion =
      Number.isInteger(options.dbVersion) && options.dbVersion > 0 ? options.dbVersion : 1;
    this.indexedDB =
      options.indexedDB !== undefined
        ? options.indexedDB
        : typeof globalThis !== 'undefined' && globalThis.indexedDB
          ? globalThis.indexedDB
          : null;
    this.backend = options.backend || null;

    this.maxFileBytes = parseQuotaLimit(options.maxFileBytes, DEFAULT_MAX_FILE_BYTES, 'maxFileBytes');
    this.maxTotalBytes = parseQuotaLimit(options.maxTotalBytes, DEFAULT_MAX_TOTAL_BYTES, 'maxTotalBytes');
    this.maxFileCount = parseQuotaLimit(options.maxFileCount, DEFAULT_MAX_FILE_COUNT, 'maxFileCount');
    this.maxQuarantineFiles = parseQuotaLimit(
      options.maxQuarantineFiles,
      DEFAULT_MAX_QUARANTINE_FILES,
      'maxQuarantineFiles',
    );
    this.maxQuarantineBytes = parseQuotaLimit(
      options.maxQuarantineBytes,
      DEFAULT_MAX_QUARANTINE_BYTES,
      'maxQuarantineBytes',
    );
    this.maxQuotaCleanupSteps = parseQuotaLimit(
      options.maxQuotaCleanupSteps,
      DEFAULT_MAX_QUOTA_CLEANUP_STEPS,
      'maxQuotaCleanupSteps',
    );
    this.coalescePendingWrites = options.coalescePendingWrites !== false;
    this.maxCoalescedPerKey = parseQuotaLimit(
      options.maxCoalescedPerKey,
      DEFAULT_MAX_COALESCED_PER_KEY,
      'maxCoalescedPerKey',
    );
    this.maxPendingCoalescedKeys = parseQuotaLimit(
      options.maxPendingCoalescedKeys,
      DEFAULT_MAX_PENDING_COALESCED_KEYS,
      'maxPendingCoalescedKeys',
    );
    this.storageEstimate = typeof options.storageEstimate === 'function' ? options.storageEstimate : null;

    this._db = null;
    this._dbPromise = null;

    // In-memory container/blob metadata index (never stores raw Uint8Array payloads).
    this._metaIndex = new Map();
    this._dirIndex = new Map();
    this._sortedDirCache = new Map();
    this._containerListCache = new Map();
    this._containerBuckets = new Map();
    this._indexedTotalBytes = 0;
    this._fallbackStoredBytes = 0;
    this._fallbackSizes = new Map();
    this._indexReady = false;
    this._indexEpoch = 0;
    this._indexLoadPromise = null;
    this._quarantineSeq = 0;

    // Serializes mutations so quota validation + persistence + index updates never race,
    // while allowing bounded latest-wins coalescing for back-to-back writes to the same slot.
    this._mutationTail = Promise.resolve();
    this._activeMutations = 0;
    this._pendingWrites = new Map();

    // Lightweight operational telemetry.
    this._stats = {
      reads: 0,
      writes: 0,
      deletes: 0,
      indexHits: 0,
      indexMisses: 0,
      indexCacheHits: 0,
      indexCacheMisses: 0,
      coalescedWrites: 0,
      bytesWritten: 0,
      bytesRead: 0,
      quotaRejections: 0,
      quotaErrors: 0,
      quotaCleanups: 0,
      invalidations: 0,
      atomicCommits: 0,
      atomicWrites: 0,
      writeVerifyFailures: 0,
      corruptRecoveries: 0,
      lastWriteDurationMs: 0,
    };
  }

  _withMutationLock(fn) {
    this._activeMutations += 1;
    const exec = async () => {
      try {
        return await fn();
      } finally {
        this._activeMutations = Math.max(0, this._activeMutations - 1);
      }
    };
    const run = this._mutationTail.then(exec, exec);
    this._mutationTail = run.then(
      () => undefined,
      () => undefined,
    );
    return run;
  }

  _sealPendingWrite(validPath) {
    this._pendingWrites.delete(validPath);
  }

  _sealPendingWritesByPrefix(normalizedPrefix) {
    if (this._pendingWrites.size === 0) {
      return;
    }
    for (const key of this._pendingWrites.keys()) {
      if (key.startsWith(normalizedPrefix)) {
        this._pendingWrites.delete(key);
      }
    }
  }

  _sealAllPendingWrites() {
    this._pendingWrites.clear();
  }

  async _flushPendingWrite(entry) {
    if (this._pendingWrites.get(entry.path) === entry) {
      this._pendingWrites.delete(entry.path);
    }
    const { path: validPath, data, ts, waiters } = entry;
    entry.data = null;
    entry.waiters = [];
    try {
      const result = await this._saveFileUnlocked(validPath, data, ts);
      for (let i = 0; i < waiters.length; i += 1) {
        waiters[i].resolve({
          path: result.path,
          size: result.size,
          mtime: result.mtime,
        });
      }
    } catch (err) {
      for (let i = 0; i < waiters.length; i += 1) {
        waiters[i].reject(err);
      }
    }
  }

  async flushPendingWrites() {
    while (this._activeMutations > 0 || this._pendingWrites.size > 0) {
      await this._mutationTail;
    }
  }

  _clearIndexMaps() {
    this._metaIndex.clear();
    this._dirIndex.clear();
    this._sortedDirCache.clear();
    this._containerListCache.clear();
    this._containerBuckets.clear();
    this._indexedTotalBytes = 0;
    this._fallbackStoredBytes = 0;
    this._fallbackSizes.clear();
    this._indexReady = false;
  }

  _getSortedPathsForPrefix(normalizedPrefix) {
    const cached = this._sortedDirCache.get(normalizedPrefix);
    if (cached) {
      return cached.slice();
    }
    const set = this._dirIndex.get(normalizedPrefix);
    if (!set || set.size === 0) {
      return [];
    }
    const sorted = Array.from(set).sort(compareAscii);
    this._sortedDirCache.set(normalizedPrefix, sorted);
    return sorted.slice();
  }

  _indexPut(validPath, size, mtime) {
    const nextMeta = buildMetadataEntry(validPath, size, mtime);
    const prev = this._metaIndex.get(validPath);
    this._containerListCache.clear();

    if (prev) {
      const delta = size - prev.size;
      this._indexedTotalBytes = Math.max(0, this._indexedTotalBytes + delta);
      const bucket = this._containerBuckets.get(prev.containerPrefix);
      if (bucket) {
        bucket.totalBytes = Math.max(0, bucket.totalBytes + delta);
      }
      this._metaIndex.set(validPath, nextMeta);
      return;
    }

    this._metaIndex.set(validPath, nextMeta);
    this._indexedTotalBytes += size;

    const prefixes = ancestorPrefixesOf(validPath);
    for (let i = 0; i < prefixes.length; i += 1) {
      const prefix = prefixes[i];
      let set = this._dirIndex.get(prefix);
      if (!set) {
        set = new Set();
        this._dirIndex.set(prefix, set);
      }
      set.add(validPath);
      this._sortedDirCache.delete(prefix);
    }

    let bucket = this._containerBuckets.get(nextMeta.containerPrefix);
    if (!bucket) {
      bucket = { fileCount: 0, totalBytes: 0 };
      this._containerBuckets.set(nextMeta.containerPrefix, bucket);
    }
    bucket.fileCount += 1;
    bucket.totalBytes += size;
  }

  _indexDelete(validPath) {
    const prev = this._metaIndex.get(validPath);
    if (!prev) {
      return false;
    }

    this._metaIndex.delete(validPath);
    this._containerListCache.clear();
    this._indexedTotalBytes = Math.max(0, this._indexedTotalBytes - prev.size);

    const prefixes = ancestorPrefixesOf(validPath);
    for (let i = 0; i < prefixes.length; i += 1) {
      const prefix = prefixes[i];
      const set = this._dirIndex.get(prefix);
      if (set) {
        set.delete(validPath);
        if (set.size === 0) {
          this._dirIndex.delete(prefix);
        }
      }
      this._sortedDirCache.delete(prefix);
    }

    const bucket = this._containerBuckets.get(prev.containerPrefix);
    if (bucket) {
      bucket.fileCount -= 1;
      bucket.totalBytes = Math.max(0, bucket.totalBytes - prev.size);
      if (bucket.fileCount <= 0) {
        this._containerBuckets.delete(prev.containerPrefix);
      }
    }

    return true;
  }

  _populateIndexFromEntries(entries) {
    this._clearIndexMaps();
    for (let i = 0; i < entries.length; i += 1) {
      const entry = entries[i];
      this._indexPut(entry.path, entry.size, entry.mtime);
    }
    this._indexReady = true;
  }

  async _populateIndexFromRecords(records) {
    const entries = [];
    for (let i = 0; i < records.length; i += 1) {
      const rec = records[i];
      if (!rec || typeof rec.path !== 'string') continue;
      if (isQuarantinePath(rec.path)) continue;
      if (isTempPath(rec.path)) {
        await this._cleanupTempPath(rec.path);
        this._stats.corruptRecoveries += 1;
        continue;
      }
      const checked = validateStoredRecord(rec);
      if (!checked.ok) {
        if (checked.reason !== 'invalid_path' && checked.path) {
          await this._quarantineCorruptRecord(checked.path, rec, checked.reason);
        }
        continue;
      }
      entries.push({
        path: checked.path,
        size: checked.data.byteLength,
        mtime: checked.mtime,
      });
    }
    this._populateIndexFromEntries(entries);
  }

  invalidateIndex() {
    this._indexEpoch += 1;
    this._clearIndexMaps();
    this._indexLoadPromise = null;
    this._stats.invalidations += 1;
  }

  async _ensureIndex(options = {}) {
    const countHit = Boolean(options && options.countHit);
    if (this._indexReady) {
      if (countHit) {
        this._stats.indexHits += 1;
        this._stats.indexCacheHits += 1;
      }
      return true;
    }

    if (this.backend && typeof this.backend.getAll !== 'function') {
      if (countHit) {
        this._stats.indexMisses += 1;
        this._stats.indexCacheMisses += 1;
      }
      return false;
    }

    if (!this._indexLoadPromise) {
      this._indexLoadPromise = (async () => {
        while (!this._indexReady) {
          const startEpoch = this._indexEpoch;
          this._stats.indexMisses += 1;
          this._stats.indexCacheMisses += 1;
          const records = await this._getAllRecords();
          if (this._indexEpoch === startEpoch) {
            await this._populateIndexFromRecords(records);
            return true;
          }
        }
        return true;
      })().finally(() => {
        this._indexLoadPromise = null;
      });
    }

    await this._indexLoadPromise;
    return this._indexReady;
  }

  async _validateQuotaForWrites(writeSpecs, deletePathsSet) {
    for (let i = 0; i < writeSpecs.length; i += 1) {
      const spec = writeSpecs[i];
      if (spec.size > this.maxFileBytes) {
        this._stats.quotaRejections += 1;
        this._stats.quotaErrors += 1;
        throw createQuotaError(
          `Userdata file "${spec.path}" (${spec.size} bytes) exceeds maxFileBytes (${this.maxFileBytes})`,
          { path: spec.path, size: spec.size, maxFileBytes: this.maxFileBytes },
        );
      }
    }

    const needsTotalCheck =
      this.maxTotalBytes < Infinity ||
      this.maxFileCount < Infinity ||
      typeof this.storageEstimate === 'function';

    if (!needsTotalCheck) {
      return;
    }

    const indexed = await this._ensureIndex({ countHit: false });
    const baseBytes = indexed ? this._indexedTotalBytes : 0;
    const baseCount = indexed ? this._metaIndex.size : 0;

    let projectedBytes = baseBytes;
    let projectedCount = baseCount;

    if (indexed && deletePathsSet && deletePathsSet.size > 0) {
      for (const delPath of deletePathsSet) {
        const existing = this._metaIndex.get(delPath);
        if (existing) {
          projectedBytes = Math.max(0, projectedBytes - existing.size);
          projectedCount = Math.max(0, projectedCount - 1);
        }
      }
    }

    for (let i = 0; i < writeSpecs.length; i += 1) {
      const spec = writeSpecs[i];
      const existing =
        indexed && (!deletePathsSet || !deletePathsSet.has(spec.path))
          ? this._metaIndex.get(spec.path)
          : null;
      if (existing) {
        projectedBytes = Math.max(0, projectedBytes - existing.size + spec.size);
      } else {
        projectedBytes += spec.size;
        projectedCount += 1;
      }
    }

    if (projectedBytes > this.maxTotalBytes) {
      this._stats.quotaRejections += 1;
      this._stats.quotaErrors += 1;
      throw createQuotaError(
        `Userdata write would reach ${projectedBytes} bytes, exceeding maxTotalBytes (${this.maxTotalBytes})`,
        {
          currentBytes: baseBytes,
          projectedBytes,
          maxTotalBytes: this.maxTotalBytes,
        },
      );
    }

    if (projectedCount > this.maxFileCount) {
      this._stats.quotaRejections += 1;
      this._stats.quotaErrors += 1;
      throw createQuotaError(
        `Userdata write would reach ${projectedCount} files, exceeding maxFileCount (${this.maxFileCount})`,
        {
          currentCount: baseCount,
          projectedCount,
          maxFileCount: this.maxFileCount,
        },
      );
    }

    const deltaBytes = projectedBytes - baseBytes;
    if (deltaBytes > 0 && typeof this.storageEstimate === 'function') {
      const estimate = await this.storageEstimate();
      if (
        estimate &&
        typeof estimate === 'object' &&
        Number.isFinite(estimate.quota) &&
        estimate.quota > 0 &&
        Number.isFinite(estimate.usage) &&
        estimate.usage >= 0
      ) {
        if (estimate.usage + deltaBytes > estimate.quota) {
          this._stats.quotaRejections += 1;
          this._stats.quotaErrors += 1;
          throw createQuotaError(
            `Browser storage quota exceeded: usage ${estimate.usage} + ${deltaBytes} > quota ${estimate.quota}`,
            {
              usage: estimate.usage,
              quota: estimate.quota,
              deltaBytes,
            },
          );
        }
      }
    }
  }

  _openDb() {
    if (this._db) {
      return Promise.resolve(this._db);
    }
    if (this._dbPromise) {
      return this._dbPromise;
    }
    if (!this.indexedDB || typeof this.indexedDB.open !== 'function') {
      return Promise.resolve(null);
    }

    this._dbPromise = new Promise((resolve, reject) => {
      let req;
      try {
        req = this.indexedDB.open(this.dbName, this.dbVersion);
      } catch (err) {
        this._dbPromise = null;
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
        const db = req.result;
        this._db = db;
        if (db && typeof db === 'object') {
          db.onversionchange = () => {
            try {
              db.close();
            } catch {
              // Ignore close errors during version change.
            }
            if (this._db === db) {
              this._db = null;
              this._dbPromise = null;
              this._indexEpoch += 1;
              this._clearIndexMaps();
            }
          };
          db.onclose = () => {
            if (this._db === db) {
              this._db = null;
              this._dbPromise = null;
              this._indexEpoch += 1;
              this._clearIndexMaps();
            }
          };
        }
        resolve(db);
      };

      req.onerror = () => {
        this._dbPromise = null;
        reject(req.error || new Error(`Failed to open IndexedDB database "${this.dbName}"`));
      };
    });

    return this._dbPromise;
  }

  async _runTransaction(mode, operation) {
    const db = await this._openDb();
    if (!db) {
      throw new Error('IndexedDB is unavailable');
    }

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

  async _getAllRecords() {
    if (this.backend) {
      const records = await this.backend.getAll();
      return Array.isArray(records) ? records : [];
    }
    const records = await this._runTransaction('readonly', (store) => store.getAll());
    return Array.isArray(records) ? records : [];
  }

  async _putRawRecord(record) {
    if (this.backend) {
      await this.backend.put(record);
      return;
    }
    await this._runTransaction('readwrite', (store) => store.put(record));
  }

  async _getRawRecord(key) {
    if (this.backend) {
      if (typeof this.backend.get === 'function') {
        return await this.backend.get(key);
      }
      if (typeof this.backend.getAll === 'function') {
        const records = await this._getAllRecords();
        return records.find((r) => r && r.path === key) || null;
      }
      return null;
    }
    const rec = await this._runTransaction('readonly', (store) => {
      if (typeof store.get === 'function') {
        return store.get(key);
      }
      return store.getAll();
    });
    if (Array.isArray(rec)) {
      return rec.find((r) => r && r.path === key) || null;
    }
    return rec || null;
  }

  async _deleteRawRecord(key) {
    if (this.backend) {
      if (typeof this.backend.delete === 'function') {
        await this.backend.delete(key);
      }
      return;
    }
    await this._runTransaction('readwrite', (store) => store.delete(key));
  }

  async _cleanupTempPath(tmpPath) {
    try {
      await this._deleteRawRecord(tmpPath);
    } catch {
      // Best-effort cleanup of temp staging artifacts.
    }
  }

  _verifyStagedRecord(staged, expectedSize, expectedChecksum) {
    if (!staged || typeof staged !== 'object') {
      return { ok: false, reason: 'missing_staged_record' };
    }
    let verifyData;
    try {
      verifyData = toUint8Array(staged.data);
    } catch {
      return { ok: false, reason: 'invalid_staged_payload' };
    }
    if (verifyData.byteLength !== expectedSize) {
      return {
        ok: false,
        reason: 'staged_size_mismatch',
        expectedSize,
        actualSize: verifyData.byteLength,
      };
    }
    if (staged.size !== undefined && (!Number.isInteger(staged.size) || staged.size !== expectedSize)) {
      return {
        ok: false,
        reason: 'staged_metadata_size_mismatch',
        expectedSize,
        actualSize: staged.size,
      };
    }
    const actualChecksum = computeChecksum(verifyData);
    if (actualChecksum !== expectedChecksum) {
      return {
        ok: false,
        reason: 'staged_checksum_mismatch',
        expectedChecksum,
        actualChecksum,
      };
    }
    if (
      staged.checksum !== undefined &&
      (typeof staged.checksum !== 'number' || (staged.checksum >>> 0) !== expectedChecksum)
    ) {
      return {
        ok: false,
        reason: 'staged_metadata_checksum_mismatch',
        expectedChecksum,
        actualChecksum: staged.checksum,
      };
    }
    return { ok: true };
  }

  async _enforceQuarantineBounds() {
    if (this.maxQuarantineFiles === Infinity && this.maxQuarantineBytes === Infinity) {
      return { removed: 0, reclaimedBytes: 0 };
    }
    let records;
    try {
      records = await this._getAllRecords();
    } catch {
      return { removed: 0, reclaimedBytes: 0 };
    }
    const quarantined = [];
    for (let i = 0; i < records.length; i += 1) {
      const rec = records[i];
      if (!rec || typeof rec.path !== 'string' || !isQuarantinePath(rec.path)) continue;
      quarantined.push(parseQuarantineMetadata(rec));
    }
    if (quarantined.length === 0) {
      return { removed: 0, reclaimedBytes: 0 };
    }

    // Evict oldest quarantine entries first until both file count and byte budget are respected.
    quarantined.sort((a, b) => {
      if (a.quarantinedAt !== b.quarantinedAt) {
        return a.quarantinedAt - b.quarantinedAt;
      }
      return compareAscii(a.quarantineKey, b.quarantineKey);
    });

    let totalFiles = quarantined.length;
    let totalBytes = 0;
    for (let i = 0; i < quarantined.length; i += 1) {
      totalBytes += quarantined[i].size;
    }

    let removed = 0;
    let reclaimedBytes = 0;
    const stepCap =
      this.maxQuotaCleanupSteps === Infinity ? quarantined.length : this.maxQuotaCleanupSteps;
    for (let i = 0; i < quarantined.length && removed < stepCap; i += 1) {
      if (totalFiles <= this.maxQuarantineFiles && totalBytes <= this.maxQuarantineBytes) {
        break;
      }
      const entry = quarantined[i];
      try {
        await this._deleteRawRecord(entry.quarantineKey);
        totalFiles -= 1;
        totalBytes = Math.max(0, totalBytes - entry.size);
        removed += 1;
        reclaimedBytes += entry.size;
      } catch {
        // Best-effort bounded quarantine eviction.
      }
    }
    return { removed, reclaimedBytes };
  }

  async _quarantineCorruptRecord(validPath, rec, reason) {
    const quarantinedAt = Date.now();
    this._quarantineSeq += 1;
    const quarantineKey = buildQuarantineKey(validPath, reason, quarantinedAt, this._quarantineSeq);

    let safeBytes = new Uint8Array(0);
    if (rec && rec.data !== undefined && rec.data !== null) {
      try {
        const raw = toUint8Array(rec.data);
        const cap = Math.min(
          MAX_QUARANTINE_RECORD_BYTES,
          this.maxQuarantineBytes === Infinity ? MAX_QUARANTINE_RECORD_BYTES : this.maxQuarantineBytes,
        );
        safeBytes = raw.byteLength > cap ? raw.slice(0, cap) : raw;
      } catch {
        safeBytes = new Uint8Array(0);
      }
    }

    if (this.maxQuarantineFiles > 0 && this.maxQuarantineBytes >= safeBytes.byteLength) {
      try {
        await this._putRawRecord({
          path: quarantineKey,
          originalPath: validPath,
          reason,
          quarantinedAt,
          data: safeBytes,
          mtime: quarantinedAt,
          quarantined: true,
        });
        await this._enforceQuarantineBounds();
      } catch {
        // Best-effort quarantine snapshot; still remove corrupt live key below.
      }
    }

    try {
      await this._deleteRawRecord(validPath);
    } catch {
      // Best-effort removal of corrupt live entry.
    }

    if (this._indexReady) {
      this._indexDelete(validPath);
    } else if (this._fallbackSizes.has(validPath)) {
      const prevFallbackSize = this._fallbackSizes.get(validPath) || 0;
      this._fallbackSizes.delete(validPath);
      this._fallbackStoredBytes = Math.max(0, this._fallbackStoredBytes - prevFallbackSize);
    }
    this._stats.corruptRecoveries += 1;
    return quarantineKey;
  }

  async _reclaimStorageOnQuotaPressure(neededBytes = 0, excludePath = null) {
    let records;
    try {
      records = await this._getAllRecords();
    } catch {
      return { removed: 0, removedTemps: 0, removedQuarantined: 0, reclaimedBytes: 0 };
    }

    const temps = [];
    const quarantined = [];
    for (let i = 0; i < records.length; i += 1) {
      const rec = records[i];
      if (!rec || typeof rec.path !== 'string') continue;
      if (excludePath && rec.path === excludePath) continue;
      let size = 0;
      try {
        size = getRecordByteLength(rec.data);
      } catch {
        size = 0;
      }
      if (isTempPath(rec.path)) {
        temps.push({ path: rec.path, size });
      } else if (isQuarantinePath(rec.path)) {
        const meta = parseQuarantineMetadata(rec);
        quarantined.push({ path: meta.quarantineKey, size: meta.size, quarantinedAt: meta.quarantinedAt });
      }
    }

    temps.sort((a, b) => compareAscii(a.path, b.path));
    quarantined.sort((a, b) => {
      if (a.quarantinedAt !== b.quarantinedAt) {
        return a.quarantinedAt - b.quarantinedAt;
      }
      return compareAscii(a.path, b.path);
    });

    const stepCap = this.maxQuotaCleanupSteps === Infinity ? Infinity : this.maxQuotaCleanupSteps;
    let removed = 0;
    let removedTemps = 0;
    let removedQuarantined = 0;
    let reclaimedBytes = 0;

    // 1. Always reclaim orphaned .tmp artifacts first (never touches live user saves).
    for (let i = 0; i < temps.length && removed < stepCap; i += 1) {
      try {
        await this._deleteRawRecord(temps[i].path);
        if (this._indexReady && this._metaIndex.has(temps[i].path)) {
          this._indexDelete(temps[i].path);
        }
        removed += 1;
        removedTemps += 1;
        reclaimedBytes += temps[i].size;
      } catch {
        // Best-effort cleanup.
      }
    }

    // 2. Evict oldest quarantined corrupt records until neededBytes is satisfied or stepCap hit.
    for (let i = 0; i < quarantined.length && removed < stepCap; i += 1) {
      if (neededBytes > 0 && reclaimedBytes >= neededBytes) {
        break;
      }
      try {
        await this._deleteRawRecord(quarantined[i].path);
        if (this._indexReady && this._metaIndex.has(quarantined[i].path)) {
          this._indexDelete(quarantined[i].path);
        }
        removed += 1;
        removedQuarantined += 1;
        reclaimedBytes += quarantined[i].size;
      } catch {
        // Best-effort cleanup.
      }
    }

    if (removed > 0) {
      this._stats.quotaCleanups += 1;
    }

    return { removed, removedTemps, removedQuarantined, reclaimedBytes };
  }

  async restoreAll(fsAdapter, options = {}) {
    const timeoutMs =
      Number.isFinite(options.timeoutMs) && options.timeoutMs > 0
        ? options.timeoutMs
        : DEFAULT_RESTORE_TIMEOUT_MS;

    if (!fsAdapter || typeof fsAdapter.writeFile !== 'function') {
      throw new TypeError('restoreAll requires an fsAdapter with a writeFile(path, bytes) method');
    }

    if (!this.backend && (!this.indexedDB || typeof this.indexedDB.open !== 'function')) {
      return { restored: 0, skipped: 0, timedOut: false, error: 'IndexedDB unavailable' };
    }

    let timer = null;
    let timedOut = false;

    const timeoutPromise = new Promise((resolve) => {
      timer = setTimeout(() => {
        timedOut = true;
        if (!this._db) {
          this._dbPromise = null;
        }
        resolve({
          restored: 0,
          skipped: 0,
          timedOut: true,
          error: `IndexedDB restore timed out after ${timeoutMs}ms`,
        });
      }, timeoutMs);
    });

    const restorePromise = (async () => {
      let restored = 0;
      let skipped = 0;
      const restoredEntries = [];
      const startEpoch = this._indexEpoch;
      try {
        const records = await this._getAllRecords();
        if (timedOut) {
          return {
            restored: 0,
            skipped: 0,
            timedOut: true,
            error: `IndexedDB restore timed out after ${timeoutMs}ms`,
          };
        }

        if (typeof fsAdapter.mkdirTree === 'function') {
          await fsAdapter.mkdirTree('/userdata');
        } else if (typeof fsAdapter.mkdir === 'function') {
          try {
            await fsAdapter.mkdir('/userdata');
          } catch {
            // Ignore already-existing /userdata directory.
          }
        }

        for (const rec of records) {
          if (timedOut) break;
          if (!rec || typeof rec.path !== 'string') {
            skipped += 1;
            continue;
          }
          if (isQuarantinePath(rec.path)) {
            continue;
          }
          if (isTempPath(rec.path)) {
            await this._cleanupTempPath(rec.path);
            this._stats.corruptRecoveries += 1;
            continue;
          }

          const checked = validateStoredRecord(rec);
          if (!checked.ok) {
            if (checked.reason !== 'invalid_path' && checked.path) {
              await this._quarantineCorruptRecord(checked.path, rec, checked.reason);
            }
            skipped += 1;
            continue;
          }

          await ensureParentDirs(fsAdapter, checked.path);
          await fsAdapter.writeFile(checked.path, checked.data, checked.mtime);
          if (typeof fsAdapter.utime === 'function' && Number.isFinite(checked.mtime)) {
            try {
              await fsAdapter.utime(checked.path, checked.mtime, checked.mtime);
            } catch {
              // Setting mtime is best-effort.
            }
          }
          this._stats.reads += 1;
          this._stats.bytesRead += checked.data.byteLength;
          restoredEntries.push({
            path: checked.path,
            size: checked.data.byteLength,
            mtime: checked.mtime,
          });
          restored += 1;
        }

        if (!timedOut && this._indexEpoch === startEpoch) {
          this._populateIndexFromEntries(restoredEntries);
        }

        return { restored, skipped, timedOut: false, error: null };
      } catch (err) {
        return { restored, skipped, timedOut: false, error: String(err) };
      }
    })();

    try {
      return await Promise.race([restorePromise, timeoutPromise]);
    } finally {
      if (timer !== null) {
        clearTimeout(timer);
      }
    }
  }

  async _saveFileUnlocked(validPath, data, ts, options = {}) {
    const startMs = nowMs();
    const expectedSize = data.byteLength;
    const expectedChecksum = computeChecksum(data);

    if (!options.skipQuotaCheck) {
      await this._validateQuotaForWrites([{ path: validPath, size: expectedSize }], null);
    }

    const tmpPath = makeTempPath(validPath);
    const stagedRecord = {
      path: tmpPath,
      data,
      mtime: ts,
      size: expectedSize,
      checksum: expectedChecksum,
      staging: true,
      targetPath: validPath,
    };
    const liveRecord = {
      path: validPath,
      data,
      mtime: ts,
      size: expectedSize,
      checksum: expectedChecksum,
    };

    // Phase 1: Stage write into sibling .tmp entry without touching live entry.
    try {
      await this._putRawRecord(stagedRecord);
    } catch (err) {
      if (isQuotaExceededError(err)) {
        const reclaimed = await this._reclaimStorageOnQuotaPressure(expectedSize, tmpPath);
        if (reclaimed.removed > 0) {
          try {
            await this._putRawRecord(stagedRecord);
          } catch (retryErr) {
            await this._cleanupTempPath(tmpPath);
            if (isQuotaExceededError(retryErr)) {
              this._stats.quotaRejections += 1;
              this._stats.quotaErrors += 1;
              throw normalizeQuotaError(retryErr);
            }
            throw retryErr;
          }
        } else {
          await this._cleanupTempPath(tmpPath);
          this._stats.quotaRejections += 1;
          this._stats.quotaErrors += 1;
          throw normalizeQuotaError(err);
        }
      } else {
        await this._cleanupTempPath(tmpPath);
        throw err;
      }
    }

    // Phase 2: Read back and verify staged byte length + checksum before touching live entry.
    let stagedReadback;
    try {
      stagedReadback = await this._getRawRecord(tmpPath);
    } catch (err) {
      this._stats.writeVerifyFailures += 1;
      await this._cleanupTempPath(tmpPath);
      throw createWriteVerifyError(
        `Failed to read back staged userdata record for "${validPath}": ${String(err)}`,
        { path: validPath, tmpPath, cause: err },
      );
    }

    const verification = this._verifyStagedRecord(stagedReadback, expectedSize, expectedChecksum);
    if (!verification.ok) {
      this._stats.writeVerifyFailures += 1;
      await this._cleanupTempPath(tmpPath);
      throw createWriteVerifyError(
        `Staged userdata write verification failed for "${validPath}" (${verification.reason})`,
        { path: validPath, tmpPath, ...verification },
      );
    }

    // Phase 3: Commit verified record to live key, then remove .tmp sibling.
    try {
      await this._putRawRecord(liveRecord);
    } catch (err) {
      await this._cleanupTempPath(tmpPath);
      if (isQuotaExceededError(err)) {
        const reclaimed = await this._reclaimStorageOnQuotaPressure(expectedSize, null);
        if (reclaimed.removed > 0) {
          try {
            await this._putRawRecord(liveRecord);
          } catch (retryErr) {
            if (isQuotaExceededError(retryErr)) {
              this._stats.quotaRejections += 1;
              this._stats.quotaErrors += 1;
              throw normalizeQuotaError(retryErr);
            }
            throw retryErr;
          }
        } else {
          this._stats.quotaRejections += 1;
          this._stats.quotaErrors += 1;
          throw normalizeQuotaError(err);
        }
      } else {
        throw err;
      }
    }

    await this._cleanupTempPath(tmpPath);

    this._indexEpoch += 1;
    if (this._indexReady) {
      this._indexPut(validPath, expectedSize, ts);
    } else {
      const prevFallbackSize = this._fallbackSizes.get(validPath) || 0;
      this._fallbackSizes.set(validPath, expectedSize);
      this._fallbackStoredBytes = Math.max(0, this._fallbackStoredBytes - prevFallbackSize + expectedSize);
    }
    this._stats.writes += 1;
    this._stats.atomicWrites += 1;
    this._stats.bytesWritten += expectedSize;
    this._stats.lastWriteDurationMs = Math.max(0, nowMs() - startMs);

    return { path: validPath, size: expectedSize, mtime: ts };
  }

  async saveFile(path, bytes, mtime = Date.now()) {
    const validPath = validateUserdataPath(path);
    const data = toUint8Array(bytes);
    const ts = Number.isFinite(mtime) ? mtime : Date.now();

    if (!this.coalescePendingWrites) {
      return this._withMutationLock(() => this._saveFileUnlocked(validPath, data, ts));
    }

    // Reject single-file quota violations immediately before coalescing so an invalid
    // oversized write never overwrites a valid pending coalesced save for the same slot.
    if (data.byteLength > this.maxFileBytes) {
      this._stats.quotaRejections += 1;
      this._stats.quotaErrors += 1;
      throw createQuotaError(
        `Userdata file "${validPath}" (${data.byteLength} bytes) exceeds maxFileBytes (${this.maxFileBytes})`,
        { path: validPath, size: data.byteLength, maxFileBytes: this.maxFileBytes },
      );
    }

    const existing = this._pendingWrites.get(validPath);
    if (existing && existing.coalescedCount < this.maxCoalescedPerKey) {
      existing.data = data;
      existing.ts = ts;
      existing.coalescedCount += 1;
      this._stats.coalescedWrites += 1;
      return new Promise((resolve, reject) => {
        existing.waiters.push({ resolve, reject });
      });
    }

    // If this path hit its per-key coalescing bound, seal the full wave so the
    // new write begins a fresh bounded flush wave after it.
    if (existing) {
      this._pendingWrites.delete(validPath);
    }

    // If distinct pending keys hit the global bound, fall back to a direct serialized write.
    if (this._pendingWrites.size >= this.maxPendingCoalescedKeys) {
      return this._withMutationLock(() => this._saveFileUnlocked(validPath, data, ts));
    }

    return new Promise((resolve, reject) => {
      const entry = {
        path: validPath,
        data,
        ts,
        coalescedCount: 0,
        waiters: [{ resolve, reject }],
      };
      this._pendingWrites.set(validPath, entry);
      this._withMutationLock(() => this._flushPendingWrite(entry));
    });
  }

  async getFile(path) {
    const validPath = validateUserdataPath(path);
    this._stats.reads += 1;

    if (this._indexReady && !this._metaIndex.has(validPath)) {
      this._stats.indexHits += 1;
      this._stats.indexCacheHits += 1;
      return null;
    }

    const rec = await this._getRawRecord(validPath);

    if (!rec || typeof rec !== 'object') {
      if (this._indexReady) {
        this._indexDelete(validPath);
      }
      return null;
    }

    const checked = validateStoredRecord(rec);
    if (!checked.ok) {
      await this._quarantineCorruptRecord(validPath, rec, checked.reason);
      return null;
    }

    this._stats.bytesRead += checked.data.byteLength;

    return {
      path: validPath,
      data: checked.data,
      mtime: checked.mtime,
    };
  }

  async _deleteFileUnlocked(validPath) {
    const prevSize = this._indexReady && this._metaIndex.has(validPath)
      ? this._metaIndex.get(validPath).size
      : 0;
    await this._deleteRawRecord(validPath);
    await this._cleanupTempPath(makeTempPath(validPath));

    this._indexEpoch += 1;
    if (this._indexReady) {
      this._indexDelete(validPath);
    } else if (this._fallbackSizes.has(validPath)) {
      const prevFallbackSize = this._fallbackSizes.get(validPath) || 0;
      this._fallbackSizes.delete(validPath);
      this._fallbackStoredBytes = Math.max(0, this._fallbackStoredBytes - prevFallbackSize);
    } else if (prevSize > 0) {
      this._fallbackStoredBytes = Math.max(0, this._fallbackStoredBytes - prevSize);
    }
    this._stats.deletes += 1;

    return { path: validPath, deleted: true };
  }

  async deleteFile(path) {
    const validPath = validateUserdataPath(path);
    this._sealPendingWrite(validPath);
    return this._withMutationLock(() => this._deleteFileUnlocked(validPath));
  }

  async listFiles() {
    const startEpoch = this._indexEpoch;
    const records = await this._getAllRecords();
    const out = [];
    const indexEntries = [];
    for (const rec of records) {
      if (!rec || typeof rec.path !== 'string') continue;
      if (isQuarantinePath(rec.path)) continue;
      if (isTempPath(rec.path)) {
        await this._cleanupTempPath(rec.path);
        this._stats.corruptRecoveries += 1;
        continue;
      }
      const checked = validateStoredRecord(rec);
      if (!checked.ok) {
        if (checked.reason !== 'invalid_path' && checked.path) {
          await this._quarantineCorruptRecord(checked.path, rec, checked.reason);
        }
        continue;
      }
      out.push({
        path: checked.path,
        data: checked.data,
        mtime: checked.mtime,
      });
      indexEntries.push({
        path: checked.path,
        size: checked.data.byteLength,
        mtime: checked.mtime,
      });
      this._stats.reads += 1;
      this._stats.bytesRead += checked.data.byteLength;
    }

    if (this._indexEpoch === startEpoch) {
      this._populateIndexFromEntries(indexEntries);
    }
    return out;
  }

  async _listByPrefixUnlocked(normalizedPrefix) {
    const indexed = await this._ensureIndex({ countHit: true });
    if (indexed) {
      return this._getSortedPathsForPrefix(normalizedPrefix);
    }
    const files = await this.listFiles();
    const paths = [];
    for (const file of files) {
      if (file.path.startsWith(normalizedPrefix)) {
        paths.push(file.path);
      }
    }
    paths.sort(compareAscii);
    return paths;
  }

  async listByPrefix(prefix) {
    const normalizedPrefix = validateUserdataPrefix(prefix);
    return this._listByPrefixUnlocked(normalizedPrefix);
  }

  async listKeys(prefix = USERDATA_PREFIX) {
    return this.listByPrefix(prefix);
  }

  async _deletePrefixUnlocked(normalizedPrefix) {
    const paths = await this._listByPrefixUnlocked(normalizedPrefix);
    for (const filePath of paths) {
      await this._deleteFileUnlocked(filePath);
    }
    return paths.length;
  }

  async deletePrefix(prefix) {
    const normalizedPrefix = validateUserdataPrefix(prefix);
    this._sealPendingWritesByPrefix(normalizedPrefix);
    return this._withMutationLock(() => this._deletePrefixUnlocked(normalizedPrefix));
  }

  async clear(options = {}) {
    const includeQuarantine = Boolean(options && options.includeQuarantine === true);
    this._sealAllPendingWrites();
    return this._withMutationLock(async () => {
      const deleted = await this._deletePrefixUnlocked(USERDATA_PREFIX);
      let quarantinedRemoved = 0;
      if (includeQuarantine) {
        const entries = await this.listQuarantined();
        for (let i = 0; i < entries.length; i += 1) {
          await this._deleteRawRecord(entries[i].quarantineKey);
          quarantinedRemoved += 1;
        }
      }
      this._indexEpoch += 1;
      this._clearIndexMaps();
      this._indexReady = true;
      return includeQuarantine
        ? { deleted, quarantinedRemoved, cleared: true }
        : { deleted, cleared: true };
    });
  }

  async getMetadata(path) {
    const validPath = validateUserdataPath(path);
    const indexed = await this._ensureIndex({ countHit: true });
    if (indexed) {
      const meta = this._metaIndex.get(validPath);
      return meta ? cloneMetadataEntry(meta) : null;
    }
    const file = await this.getFile(validPath);
    if (!file) return null;
    return buildMetadataEntry(file.path, file.data.byteLength, file.mtime);
  }

  async stat(path) {
    return this.getMetadata(path);
  }

  async hasFile(path) {
    const meta = await this.getMetadata(path);
    return meta !== null;
  }

  async listMetadataByPrefix(prefix) {
    const normalizedPrefix = validateUserdataPrefix(prefix);
    const indexed = await this._ensureIndex({ countHit: true });
    if (indexed) {
      const sortedPaths = this._getSortedPathsForPrefix(normalizedPrefix);
      if (sortedPaths.length === 0) {
        return [];
      }
      const out = [];
      for (let i = 0; i < sortedPaths.length; i += 1) {
        const meta = this._metaIndex.get(sortedPaths[i]);
        if (meta) {
          out.push(cloneMetadataEntry(meta));
        }
      }
      return out;
    }

    const files = await this.listFiles();
    const out = [];
    for (const file of files) {
      if (file.path.startsWith(normalizedPrefix)) {
        out.push(buildMetadataEntry(file.path, file.data.byteLength, file.mtime));
      }
    }
    out.sort((a, b) => compareAscii(a.path, b.path));
    return out;
  }

  async listContainers(rootPrefix = USERDATA_PREFIX, options = {}) {
    const normalizedRoot = validateUserdataPrefix(rootPrefix);
    let namePrefix = '';
    if (options && options.namePrefix !== undefined && options.namePrefix !== null) {
      if (typeof options.namePrefix !== 'string') {
        throw new TypeError('Container namePrefix must be a string');
      }
      if (
        options.namePrefix.includes('/') ||
        options.namePrefix.includes('\\') ||
        options.namePrefix.includes('\0')
      ) {
        throw new Error(`Invalid container namePrefix: ${options.namePrefix}`);
      }
      namePrefix = options.namePrefix;
    }

    const indexed = await this._ensureIndex({ countHit: true });
    const cacheKey = `${normalizedRoot}\0${namePrefix}`;
    if (indexed) {
      const cached = this._containerListCache.get(cacheKey);
      if (cached) {
        const cloned = new Array(cached.length);
        for (let i = 0; i < cached.length; i += 1) {
          cloned[i] = { ...cached[i] };
        }
        return cloned;
      }
    }

    const containerMap = new Map();

    const accumulate = (filePath, size, mtime) => {
      const rel = filePath.slice(normalizedRoot.length);
      const slashIdx = rel.indexOf('/');
      if (slashIdx <= 0) {
        return;
      }
      const name = rel.slice(0, slashIdx);
      if (namePrefix && !name.startsWith(namePrefix)) {
        return;
      }
      let bucket = containerMap.get(name);
      if (!bucket) {
        bucket = {
          name,
          prefix: `${normalizedRoot}${name}/`,
          blobCount: 0,
          totalBytes: 0,
          lastModified: 0,
        };
        containerMap.set(name, bucket);
      }
      bucket.blobCount += 1;
      bucket.totalBytes += size;
      if (mtime > bucket.lastModified) {
        bucket.lastModified = mtime;
      }
    };

    if (indexed) {
      const set = this._dirIndex.get(normalizedRoot);
      if (set && set.size > 0) {
        for (const filePath of set) {
          const meta = this._metaIndex.get(filePath);
          if (meta) {
            accumulate(meta.path, meta.size, meta.mtime);
          }
        }
      }
    } else {
      const files = await this.listFiles();
      for (const file of files) {
        if (file.path.startsWith(normalizedRoot)) {
          accumulate(file.path, file.data.byteLength, file.mtime);
        }
      }
    }

    const out = Array.from(containerMap.values());
    out.sort((a, b) => compareAscii(a.name, b.name));

    if (indexed) {
      if (this._containerListCache.size >= MAX_CONTAINER_LIST_CACHE_ENTRIES) {
        this._containerListCache.clear();
      }
      const cachedCopy = new Array(out.length);
      for (let i = 0; i < out.length; i += 1) {
        cachedCopy[i] = { ...out[i] };
      }
      this._containerListCache.set(cacheKey, cachedCopy);
    }

    return out;
  }

  async commitUpdate(update = {}) {
    if (!update || typeof update !== 'object' || Array.isArray(update)) {
      throw new TypeError('commitUpdate requires an update object');
    }
    const rawWrites = update.writes === undefined ? [] : update.writes;
    const rawDeletes = update.deletes === undefined ? [] : update.deletes;
    if (!Array.isArray(rawWrites)) {
      throw new TypeError('commitUpdate writes must be an array');
    }
    if (!Array.isArray(rawDeletes)) {
      throw new TypeError('commitUpdate deletes must be an array');
    }

    const defaultMtime = Number.isFinite(update.mtime) ? update.mtime : Date.now();
    const deleteSet = new Set();
    const validatedDeletes = [];
    for (let i = 0; i < rawDeletes.length; i += 1) {
      const validPath = validateUserdataPath(rawDeletes[i]);
      if (!deleteSet.has(validPath)) {
        deleteSet.add(validPath);
        validatedDeletes.push(validPath);
      }
    }

    const seenWrites = new Set();
    const validatedWrites = [];
    const writeSpecs = [];
    for (let i = 0; i < rawWrites.length; i += 1) {
      const entry = rawWrites[i];
      if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
        throw new TypeError('commitUpdate write entry must be an object');
      }
      const validPath = validateUserdataPath(entry.path);
      if (seenWrites.has(validPath)) {
        throw new Error(`Duplicate write path in commitUpdate: ${validPath}`);
      }
      if (deleteSet.has(validPath)) {
        throw new Error(`Conflicting write and delete path in commitUpdate: ${validPath}`);
      }
      seenWrites.add(validPath);

      const data = toUint8Array(entry.data);
      const mtime = Number.isFinite(entry.mtime) ? entry.mtime : defaultMtime;
      validatedWrites.push({ path: validPath, data, mtime });
      writeSpecs.push({ path: validPath, size: data.byteLength });
    }

    this._sealAllPendingWrites();
    return this._withMutationLock(async () => {
      const startMs = nowMs();
      await this._validateQuotaForWrites(writeSpecs, deleteSet);

      for (let i = 0; i < validatedDeletes.length; i += 1) {
        await this._deleteFileUnlocked(validatedDeletes[i]);
      }

      let bytesWritten = 0;
      for (let i = 0; i < validatedWrites.length; i += 1) {
        const rec = validatedWrites[i];
        await this._saveFileUnlocked(rec.path, rec.data, rec.mtime, { skipQuotaCheck: true });
        bytesWritten += rec.data.byteLength;
      }

      this._stats.atomicCommits += 1;
      this._stats.lastWriteDurationMs = Math.max(0, nowMs() - startMs);

      return {
        written: validatedWrites.length,
        deleted: validatedDeletes.length,
        bytesWritten,
      };
    });
  }

  async getQuotaInfo() {
    const indexed = await this._ensureIndex({ countHit: true });
    let usedBytes = 0;
    let fileCount = 0;
    if (indexed) {
      usedBytes = this._indexedTotalBytes;
      fileCount = this._metaIndex.size;
    } else {
      const usage = await this.getUsageStats();
      usedBytes = usage.totalBytes;
      fileCount = usage.fileCount;
    }

    const remainingBytes =
      this.maxTotalBytes === Infinity ? Infinity : Math.max(0, this.maxTotalBytes - usedBytes);
    return {
      usedBytes,
      maxTotalBytes: this.maxTotalBytes,
      remainingBytes,
      maxFileBytes: this.maxFileBytes,
      fileCount,
      maxFileCount: this.maxFileCount,
    };
  }

  async listQuarantined() {
    const records = await this._getAllRecords();
    const out = [];
    for (let i = 0; i < records.length; i += 1) {
      const rec = records[i];
      if (!rec || typeof rec.path !== 'string' || !isQuarantinePath(rec.path)) continue;
      out.push(parseQuarantineMetadata(rec));
    }
    out.sort((a, b) => {
      if (a.quarantinedAt !== b.quarantinedAt) {
        return a.quarantinedAt - b.quarantinedAt;
      }
      return compareAscii(a.quarantineKey, b.quarantineKey);
    });
    return out;
  }

  async purgeQuarantine() {
    return this._withMutationLock(async () => {
      const entries = await this.listQuarantined();
      let removed = 0;
      let reclaimedBytes = 0;
      for (let i = 0; i < entries.length; i += 1) {
        await this._deleteRawRecord(entries[i].quarantineKey);
        removed += 1;
        reclaimedBytes += entries[i].size;
      }
      return { removed, reclaimedBytes };
    });
  }

  async cleanupArtifacts(options = {}) {
    const neededBytes =
      options && Number.isFinite(options.neededBytes) && options.neededBytes > 0
        ? Math.floor(options.neededBytes)
        : Infinity;
    return this._withMutationLock(() => this._reclaimStorageOnQuotaPressure(neededBytes, null));
  }

  getStats() {
    const storedBytes = this._indexReady ? this._indexedTotalBytes : this._fallbackStoredBytes;
    return {
      reads: this._stats.reads,
      writes: this._stats.writes,
      deletes: this._stats.deletes,
      indexHits: this._stats.indexHits,
      indexMisses: this._stats.indexMisses,
      indexCacheHits: this._stats.indexCacheHits,
      indexCacheMisses: this._stats.indexCacheMisses,
      coalescedWrites: this._stats.coalescedWrites,
      bytesWritten: this._stats.bytesWritten,
      bytesRead: this._stats.bytesRead,
      quotaRejections: this._stats.quotaRejections,
      quotaErrors: this._stats.quotaErrors,
      quotaCleanups: this._stats.quotaCleanups,
      invalidations: this._stats.invalidations,
      atomicCommits: this._stats.atomicCommits,
      atomicWrites: this._stats.atomicWrites,
      writeVerifyFailures: this._stats.writeVerifyFailures,
      corruptRecoveries: this._stats.corruptRecoveries,
      lastWriteDurationMs: this._stats.lastWriteDurationMs,
      indexReady: this._indexReady,
      indexedFiles: this._metaIndex.size,
      indexedBytes: this._indexedTotalBytes,
      storedBytes,
    };
  }

  resetStats() {
    this._stats.reads = 0;
    this._stats.writes = 0;
    this._stats.deletes = 0;
    this._stats.indexHits = 0;
    this._stats.indexMisses = 0;
    this._stats.indexCacheHits = 0;
    this._stats.indexCacheMisses = 0;
    this._stats.coalescedWrites = 0;
    this._stats.bytesWritten = 0;
    this._stats.bytesRead = 0;
    this._stats.quotaRejections = 0;
    this._stats.quotaErrors = 0;
    this._stats.quotaCleanups = 0;
    this._stats.invalidations = 0;
    this._stats.atomicCommits = 0;
    this._stats.atomicWrites = 0;
    this._stats.writeVerifyFailures = 0;
    this._stats.corruptRecoveries = 0;
    this._stats.lastWriteDurationMs = 0;
  }

  async getUsageStats() {
    const indexed = await this._ensureIndex({ countHit: true });
    if (indexed) {
      const sortedKeys = Array.from(this._containerBuckets.keys()).sort(compareAscii);
      const prefixes = {};
      for (const key of sortedKeys) {
        const bucket = this._containerBuckets.get(key);
        prefixes[key] = {
          fileCount: bucket.fileCount,
          totalBytes: bucket.totalBytes,
        };
      }
      return {
        fileCount: this._metaIndex.size,
        totalBytes: this._indexedTotalBytes,
        prefixes,
      };
    }

    const files = await this.listFiles();
    let totalBytes = 0;
    const prefixMap = new Map();

    for (const file of files) {
      const size = file.data.byteLength;
      totalBytes += size;
      const rel = file.path.slice(USERDATA_PREFIX.length);
      const slashIdx = rel.indexOf('/');
      const containerPrefix =
        slashIdx === -1 ? USERDATA_PREFIX : `${USERDATA_PREFIX}${rel.slice(0, slashIdx)}/`;

      let bucket = prefixMap.get(containerPrefix);
      if (!bucket) {
        bucket = { fileCount: 0, totalBytes: 0 };
        prefixMap.set(containerPrefix, bucket);
      }
      bucket.fileCount += 1;
      bucket.totalBytes += size;
    }

    const sortedKeys = Array.from(prefixMap.keys()).sort(compareAscii);
    const prefixes = {};
    for (const key of sortedKeys) {
      prefixes[key] = prefixMap.get(key);
    }

    return {
      fileCount: files.length,
      totalBytes,
      prefixes,
    };
  }

  async exportArchive(options = {}) {
    const filterPrefix =
      options && options.prefix !== undefined ? validateUserdataPrefix(options.prefix) : null;
    const entries = [];

    if (filterPrefix && this._indexReady) {
      const paths = await this._listByPrefixUnlocked(filterPrefix);
      for (let i = 0; i < paths.length; i += 1) {
        const file = await this.getFile(paths[i]);
        if (!file) continue;
        entries.push({
          path: file.path,
          size: file.data.byteLength,
          mtime: Number.isFinite(file.mtime) ? file.mtime : 0,
          base64: encodeBase64(file.data),
        });
      }
    } else {
      const files = await this.listFiles();
      for (const file of files) {
        const validPath = validateUserdataPath(file.path);
        if (filterPrefix && !validPath.startsWith(filterPrefix)) {
          continue;
        }
        entries.push({
          path: validPath,
          size: file.data.byteLength,
          mtime: Number.isFinite(file.mtime) ? file.mtime : 0,
          base64: encodeBase64(file.data),
        });
      }
    }

    entries.sort((a, b) => compareAscii(a.path, b.path));

    return {
      format: ARCHIVE_FORMAT,
      version: ARCHIVE_VERSION,
      exportedAt: new Date().toISOString(),
      files: entries,
    };
  }

  async importArchive(archive, options = {}) {
    let parsed = archive;
    if (typeof archive === 'string') {
      parsed = JSON.parse(archive);
    }

    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new TypeError('Userdata archive must be an object or JSON string');
    }
    if (parsed.format !== ARCHIVE_FORMAT) {
      throw new Error(`Unsupported userdata archive format: ${parsed.format}`);
    }
    if (parsed.version !== ARCHIVE_VERSION) {
      throw new Error(`Unsupported userdata archive version: ${parsed.version}`);
    }
    if (!Array.isArray(parsed.files)) {
      throw new TypeError('Userdata archive must contain a files array');
    }

    const validated = [];
    const writeSpecs = [];
    const seenPaths = new Set();
    for (const entry of parsed.files) {
      if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
        throw new TypeError('Userdata archive file entry must be an object');
      }

      const validPath = validateUserdataPath(entry.path);
      if (seenPaths.has(validPath)) {
        throw new Error(`Duplicate userdata path in archive: ${validPath}`);
      }
      seenPaths.add(validPath);

      let data;
      if (typeof entry.base64 === 'string') {
        data = decodeBase64(entry.base64);
      } else if (
        entry.data instanceof Uint8Array ||
        ArrayBuffer.isView(entry.data) ||
        entry.data instanceof ArrayBuffer
      ) {
        data = toUint8Array(entry.data);
      } else {
        throw new TypeError(`Archive entry for ${validPath} must include a base64 string or binary data`);
      }

      if (
        entry.size !== undefined &&
        (!Number.isInteger(entry.size) || entry.size !== data.byteLength)
      ) {
        throw new Error(
          `Archive entry size mismatch for ${validPath}: expected ${entry.size}, decoded ${data.byteLength}`,
        );
      }

      const mtime = Number.isFinite(entry.mtime) ? entry.mtime : Date.now();
      validated.push({ path: validPath, data, mtime });
      writeSpecs.push({ path: validPath, size: data.byteLength });
    }

    const replace = Boolean(options && options.replace === true);

    this._sealAllPendingWrites();
    return this._withMutationLock(async () => {
      const startMs = nowMs();
      await this._ensureIndex({ countHit: false });
      const deleteSet = replace ? new Set(this._metaIndex.keys()) : null;
      await this._validateQuotaForWrites(writeSpecs, deleteSet);

      if (replace) {
        await this._deletePrefixUnlocked(USERDATA_PREFIX);
      }

      let totalBytes = 0;
      for (const rec of validated) {
        await this._saveFileUnlocked(rec.path, rec.data, rec.mtime, { skipQuotaCheck: true });
        totalBytes += rec.data.byteLength;
      }

      this._stats.atomicCommits += 1;
      this._stats.lastWriteDurationMs = Math.max(0, nowMs() - startMs);

      return {
        importedCount: validated.length,
        totalBytes,
      };
    });
  }

  close() {
    this._sealAllPendingWrites();
    if (this._db && typeof this._db.close === 'function') {
      try {
        this._db.close();
      } catch {
        // Ignore close errors.
      }
    }
    this._db = null;
    this._dbPromise = null;
    this._indexEpoch += 1;
    this._clearIndexMaps();
    this._indexLoadPromise = null;
    if (this.backend && typeof this.backend.close === 'function') {
      this.backend.close();
    }
  }
}

const exportedApi = {
  UserdataStore,
  DEFAULT_DB_NAME,
  DEFAULT_STORE_NAME,
  DEFAULT_RESTORE_TIMEOUT_MS,
  DEFAULT_MAX_FILE_BYTES,
  DEFAULT_MAX_TOTAL_BYTES,
  DEFAULT_MAX_FILE_COUNT,
  DEFAULT_MAX_QUARANTINE_FILES,
  DEFAULT_MAX_QUARANTINE_BYTES,
  DEFAULT_MAX_QUOTA_CLEANUP_STEPS,
  DEFAULT_MAX_COALESCED_PER_KEY,
  DEFAULT_MAX_PENDING_COALESCED_KEYS,
  TEMP_SUFFIX,
  QUARANTINE_PREFIX,
  USERDATA_PREFIX,
  ARCHIVE_FORMAT,
  ARCHIVE_VERSION,
  validateUserdataPath,
  validateUserdataPrefix,
  encodeBase64,
  decodeBase64,
  computeChecksum,
  isQuotaExceededError,
};

if (typeof globalThis !== 'undefined') {
  globalThis.PlayGta6Userdata = exportedApi;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = exportedApi;
}
