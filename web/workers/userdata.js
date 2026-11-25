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
    this.storageEstimate = typeof options.storageEstimate === 'function' ? options.storageEstimate : null;

    this._db = null;
    this._dbPromise = null;

    // In-memory container/blob metadata index (never stores raw Uint8Array payloads).
    this._metaIndex = new Map();
    this._dirIndex = new Map();
    this._containerBuckets = new Map();
    this._indexedTotalBytes = 0;
    this._indexReady = false;
    this._indexEpoch = 0;
    this._indexLoadPromise = null;

    // Serializes mutations so quota validation + persistence + index updates never race.
    this._mutationTail = Promise.resolve();

    // Lightweight operational telemetry.
    this._stats = {
      reads: 0,
      writes: 0,
      deletes: 0,
      indexHits: 0,
      indexMisses: 0,
      bytesWritten: 0,
      bytesRead: 0,
      quotaRejections: 0,
      invalidations: 0,
      atomicCommits: 0,
      lastWriteDurationMs: 0,
    };
  }

  _withMutationLock(fn) {
    const run = this._mutationTail.then(fn, fn);
    this._mutationTail = run.then(
      () => undefined,
      () => undefined,
    );
    return run;
  }

  _clearIndexMaps() {
    this._metaIndex.clear();
    this._dirIndex.clear();
    this._containerBuckets.clear();
    this._indexedTotalBytes = 0;
    this._indexReady = false;
  }

  _indexPut(validPath, size, mtime) {
    const nextMeta = buildMetadataEntry(validPath, size, mtime);
    const prev = this._metaIndex.get(validPath);

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

  _populateIndexFromRecords(records) {
    const entries = [];
    for (let i = 0; i < records.length; i += 1) {
      const rec = records[i];
      if (!rec || typeof rec.path !== 'string') continue;
      let validPath;
      try {
        validPath = validateUserdataPath(rec.path);
      } catch {
        continue;
      }
      let size;
      try {
        size = getRecordByteLength(rec.data);
      } catch {
        continue;
      }
      entries.push({
        path: validPath,
        size,
        mtime: Number.isFinite(rec.mtime) ? rec.mtime : 0,
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
      }
      return true;
    }

    if (this.backend && typeof this.backend.getAll !== 'function') {
      return false;
    }

    if (!this._indexLoadPromise) {
      this._indexLoadPromise = (async () => {
        while (!this._indexReady) {
          const startEpoch = this._indexEpoch;
          this._stats.indexMisses += 1;
          const records = await this._getAllRecords();
          if (this._indexEpoch === startEpoch) {
            this._populateIndexFromRecords(records);
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
          let validPath;
          try {
            validPath = validateUserdataPath(rec.path);
          } catch {
            skipped += 1;
            continue;
          }

          let payload;
          try {
            payload = toUint8Array(rec.data);
          } catch {
            skipped += 1;
            continue;
          }

          await ensureParentDirs(fsAdapter, validPath);
          await fsAdapter.writeFile(validPath, payload, rec.mtime);
          if (typeof fsAdapter.utime === 'function' && Number.isFinite(rec.mtime)) {
            try {
              await fsAdapter.utime(validPath, rec.mtime, rec.mtime);
            } catch {
              // Setting mtime is best-effort.
            }
          }
          this._stats.reads += 1;
          this._stats.bytesRead += payload.byteLength;
          restoredEntries.push({
            path: validPath,
            size: payload.byteLength,
            mtime: Number.isFinite(rec.mtime) ? rec.mtime : 0,
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
    if (!options.skipQuotaCheck) {
      await this._validateQuotaForWrites([{ path: validPath, size: data.byteLength }], null);
    }

    const record = { path: validPath, data, mtime: ts };
    try {
      if (this.backend) {
        await this.backend.put(record);
      } else {
        await this._runTransaction('readwrite', (store) => store.put(record));
      }
    } catch (err) {
      if (err && (err.name === 'QuotaExceededError' || err.code === 'QUOTA_EXCEEDED')) {
        this._stats.quotaRejections += 1;
      }
      throw err;
    }

    this._indexEpoch += 1;
    if (this._indexReady) {
      this._indexPut(validPath, data.byteLength, ts);
    }
    this._stats.writes += 1;
    this._stats.bytesWritten += data.byteLength;
    this._stats.lastWriteDurationMs = Math.max(0, nowMs() - startMs);

    return { path: validPath, size: data.byteLength, mtime: ts };
  }

  async saveFile(path, bytes, mtime = Date.now()) {
    const validPath = validateUserdataPath(path);
    const data = toUint8Array(bytes);
    const ts = Number.isFinite(mtime) ? mtime : Date.now();
    return this._withMutationLock(() => this._saveFileUnlocked(validPath, data, ts));
  }

  async getFile(path) {
    const validPath = validateUserdataPath(path);
    this._stats.reads += 1;

    if (this._indexReady && !this._metaIndex.has(validPath)) {
      this._stats.indexHits += 1;
      return null;
    }

    let rec = null;

    if (this.backend) {
      if (typeof this.backend.get === 'function') {
        rec = await this.backend.get(validPath);
      } else {
        const records = await this._getAllRecords();
        rec = records.find((r) => r && r.path === validPath) || null;
      }
    } else {
      rec = await this._runTransaction('readonly', (store) => {
        if (typeof store.get === 'function') {
          return store.get(validPath);
        }
        return store.getAll();
      });
      if (Array.isArray(rec)) {
        rec = rec.find((r) => r && r.path === validPath) || null;
      }
    }

    if (!rec || typeof rec !== 'object') {
      if (this._indexReady) {
        this._indexDelete(validPath);
      }
      return null;
    }

    const data = toUint8Array(rec.data);
    const mtime = Number.isFinite(rec.mtime) ? rec.mtime : 0;
    this._stats.bytesRead += data.byteLength;

    return {
      path: validPath,
      data,
      mtime,
    };
  }

  async _deleteFileUnlocked(validPath) {
    if (this.backend) {
      await this.backend.delete(validPath);
    } else {
      await this._runTransaction('readwrite', (store) => store.delete(validPath));
    }

    this._indexEpoch += 1;
    if (this._indexReady) {
      this._indexDelete(validPath);
    }
    this._stats.deletes += 1;

    return { path: validPath, deleted: true };
  }

  async deleteFile(path) {
    const validPath = validateUserdataPath(path);
    return this._withMutationLock(() => this._deleteFileUnlocked(validPath));
  }

  async listFiles() {
    const startEpoch = this._indexEpoch;
    const records = await this._getAllRecords();
    const out = [];
    const indexEntries = [];
    for (const rec of records) {
      if (!rec || typeof rec.path !== 'string') continue;
      try {
        const validPath = validateUserdataPath(rec.path);
        const data = toUint8Array(rec.data);
        const mtime = Number.isFinite(rec.mtime) ? rec.mtime : 0;
        out.push({
          path: validPath,
          data,
          mtime,
        });
        indexEntries.push({
          path: validPath,
          size: data.byteLength,
          mtime,
        });
        this._stats.reads += 1;
        this._stats.bytesRead += data.byteLength;
      } catch {
        // Skip invalid entries.
      }
    }

    if (this._indexEpoch === startEpoch) {
      this._populateIndexFromEntries(indexEntries);
    }
    return out;
  }

  async _listByPrefixUnlocked(normalizedPrefix) {
    const indexed = await this._ensureIndex({ countHit: true });
    if (indexed) {
      const set = this._dirIndex.get(normalizedPrefix);
      if (!set || set.size === 0) {
        return [];
      }
      return Array.from(set).sort(compareAscii);
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

  async _deletePrefixUnlocked(normalizedPrefix) {
    const paths = await this._listByPrefixUnlocked(normalizedPrefix);
    for (const filePath of paths) {
      await this._deleteFileUnlocked(filePath);
    }
    return paths.length;
  }

  async deletePrefix(prefix) {
    const normalizedPrefix = validateUserdataPrefix(prefix);
    return this._withMutationLock(() => this._deletePrefixUnlocked(normalizedPrefix));
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

  async hasFile(path) {
    const meta = await this.getMetadata(path);
    return meta !== null;
  }

  async listMetadataByPrefix(prefix) {
    const normalizedPrefix = validateUserdataPrefix(prefix);
    const indexed = await this._ensureIndex({ countHit: true });
    if (indexed) {
      const set = this._dirIndex.get(normalizedPrefix);
      if (!set || set.size === 0) {
        return [];
      }
      const sortedPaths = Array.from(set).sort(compareAscii);
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

  getStats() {
    return {
      reads: this._stats.reads,
      writes: this._stats.writes,
      deletes: this._stats.deletes,
      indexHits: this._stats.indexHits,
      indexMisses: this._stats.indexMisses,
      bytesWritten: this._stats.bytesWritten,
      bytesRead: this._stats.bytesRead,
      quotaRejections: this._stats.quotaRejections,
      invalidations: this._stats.invalidations,
      atomicCommits: this._stats.atomicCommits,
      lastWriteDurationMs: this._stats.lastWriteDurationMs,
      indexReady: this._indexReady,
      indexedFiles: this._metaIndex.size,
      indexedBytes: this._indexedTotalBytes,
    };
  }

  resetStats() {
    this._stats.reads = 0;
    this._stats.writes = 0;
    this._stats.deletes = 0;
    this._stats.indexHits = 0;
    this._stats.indexMisses = 0;
    this._stats.bytesWritten = 0;
    this._stats.bytesRead = 0;
    this._stats.quotaRejections = 0;
    this._stats.invalidations = 0;
    this._stats.atomicCommits = 0;
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
  USERDATA_PREFIX,
  ARCHIVE_FORMAT,
  ARCHIVE_VERSION,
  validateUserdataPath,
  validateUserdataPrefix,
  encodeBase64,
  decodeBase64,
};

if (typeof globalThis !== 'undefined') {
  globalThis.PlayGta6Userdata = exportedApi;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = exportedApi;
}
