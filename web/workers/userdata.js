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

    this._db = null;
    this._dbPromise = null;
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
            }
          };
          db.onclose = () => {
            if (this._db === db) {
              this._db = null;
              this._dbPromise = null;
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
          restored += 1;
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

  async saveFile(path, bytes, mtime = Date.now()) {
    const validPath = validateUserdataPath(path);
    const data = toUint8Array(bytes);
    const ts = Number.isFinite(mtime) ? mtime : Date.now();
    const record = { path: validPath, data, mtime: ts };

    if (this.backend) {
      await this.backend.put(record);
    } else {
      await this._runTransaction('readwrite', (store) => store.put(record));
    }

    return { path: validPath, size: data.byteLength, mtime: ts };
  }

  async getFile(path) {
    const validPath = validateUserdataPath(path);
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
      return null;
    }

    return {
      path: validPath,
      data: toUint8Array(rec.data),
      mtime: Number.isFinite(rec.mtime) ? rec.mtime : 0,
    };
  }

  async deleteFile(path) {
    const validPath = validateUserdataPath(path);

    if (this.backend) {
      await this.backend.delete(validPath);
    } else {
      await this._runTransaction('readwrite', (store) => store.delete(validPath));
    }

    return { path: validPath, deleted: true };
  }

  async listFiles() {
    const records = await this._getAllRecords();
    const out = [];
    for (const rec of records) {
      if (!rec || typeof rec.path !== 'string') continue;
      try {
        validateUserdataPath(rec.path);
        out.push({
          path: rec.path,
          data: toUint8Array(rec.data),
          mtime: Number.isFinite(rec.mtime) ? rec.mtime : 0,
        });
      } catch {
        // Skip invalid entries.
      }
    }
    return out;
  }

  async listByPrefix(prefix) {
    const normalizedPrefix = validateUserdataPrefix(prefix);
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

  async deletePrefix(prefix) {
    const normalizedPrefix = validateUserdataPrefix(prefix);
    const paths = await this.listByPrefix(normalizedPrefix);
    for (const filePath of paths) {
      await this.deleteFile(filePath);
    }
    return paths.length;
  }

  async getUsageStats() {
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
    const files = await this.listFiles();
    const entries = [];

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
    }

    if (options && options.replace === true) {
      await this.deletePrefix(USERDATA_PREFIX);
    }

    let totalBytes = 0;
    for (const rec of validated) {
      await this.saveFile(rec.path, rec.data, rec.mtime);
      totalBytes += rec.data.byteLength;
    }

    return {
      importedCount: validated.length,
      totalBytes,
    };
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
