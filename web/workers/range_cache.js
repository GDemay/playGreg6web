'use strict';

// Fixed-size HTTP 206 Range chunk cache with memory LRU, optional OPFS L2 persistence,
// concurrent in-flight deduplication, and bootset.json prefetching.
// Compatible with classic Web Workers (importScripts) and Node CommonJS/ESM tests.

const DEFAULT_CHUNK_SIZE = 256 * 1024; // 256 KiB
const DEFAULT_MAX_BYTES = 64 * 1024 * 1024; // 64 MiB
const DEFAULT_PREFETCH_CONCURRENCY = 4;
const MAX_TRACKED_CHUNK_LENGTHS = 65536;
const MAX_TRACKED_FILE_SIZES = 4096;

function isQuotaExceededError(err) {
  return Boolean(
    err &&
      (err.name === 'QuotaExceededError' ||
        err.code === 22 ||
        err.code === 1014 ||
        /quota/i.test(String(err.message || ''))),
  );
}

function isNotFoundError(err) {
  return Boolean(
    err &&
      (err.name === 'NotFoundError' ||
        err.code === 'ENOENT' ||
        err.code === 8 ||
        /not\s*found/i.test(String(err.message || ''))),
  );
}

function normalizeDataBase(base) {
  const raw = typeof base === 'string' && base.length > 0 ? base : '/data/';
  return raw.endsWith('/') ? raw : `${raw}/`;
}

function normalizeRelPath(relPath) {
  if (typeof relPath !== 'string' || relPath.length === 0) {
    throw new Error('RangeCache path must be a non-empty string');
  }
  if (relPath.includes('\\') || relPath.includes('\0')) {
    throw new Error(`Invalid RangeCache path: ${relPath}`);
  }
  const stripped = relPath.replace(/^\/+/, '');
  if (!stripped) {
    throw new Error(`Invalid RangeCache path: ${relPath}`);
  }
  const parts = stripped.split('/');
  for (const part of parts) {
    if (!part || part === '.' || part === '..') {
      throw new Error(`Invalid RangeCache path segment in: ${relPath}`);
    }
  }
  return stripped;
}

function utf8ToHex(str) {
  const bytes = new TextEncoder().encode(str);
  let hex = '';
  for (let i = 0; i < bytes.length; i += 1) {
    hex += bytes[i].toString(16).padStart(2, '0');
  }
  return hex;
}

function fnv1a64Hex(str) {
  // Deterministic two-lane 32-bit FNV-1a hash (128-bit hex) for long path keys.
  const bytes = new TextEncoder().encode(str);
  let h1 = 0x811c9dc5;
  let h2 = 0x811c9dc5 ^ 0x9e3779b9;
  let h3 = 0x811c9dc5 ^ 0x85ebca6b;
  let h4 = 0x811c9dc5 ^ 0xc2b2ae35;
  for (let i = 0; i < bytes.length; i += 1) {
    const b = bytes[i];
    h1 = Math.imul(h1 ^ b, 0x01000193) >>> 0;
    h2 = Math.imul(h2 ^ ((b + i) & 0xff), 0x01000193) >>> 0;
    h3 = Math.imul(h3 ^ (b ^ 0xa5), 0x01000193) >>> 0;
    h4 = Math.imul(h4 ^ ((b + (i * 31)) & 0xff), 0x01000193) >>> 0;
  }
  return (
    h1.toString(16).padStart(8, '0') +
    h2.toString(16).padStart(8, '0') +
    h3.toString(16).padStart(8, '0') +
    h4.toString(16).padStart(8, '0')
  );
}

function opfsChunkFileName(relPath, chunkIndex, chunkSize) {
  const encoded = utf8ToHex(relPath);
  const pathToken = encoded.length <= 128 ? encoded : `${encoded.slice(0, 64)}_${fnv1a64Hex(relPath)}`;
  return `c_${chunkSize}_${chunkIndex}_${pathToken}.bin`;
}

function parseContentRange(header) {
  if (typeof header !== 'string') return null;
  const match = /^bytes\s+(\d+)-(\d+)\/(\d+|\*)$/i.exec(header.trim());
  if (!match) return null;
  const start = Number(match[1]);
  const end = Number(match[2]);
  const total = match[3] === '*' ? null : Number(match[3]);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) {
    return null;
  }
  return { start, end, total };
}

async function openOpfsCacheDir(subdir = 'playgta6-cache', storageObj = null) {
  const storage =
    storageObj ||
    (typeof navigator !== 'undefined' && navigator && navigator.storage ? navigator.storage : null);
  if (!storage || typeof storage.getDirectory !== 'function') {
    return null;
  }
  try {
    const root = await storage.getDirectory();
    if (!root || typeof root.getDirectoryHandle !== 'function') {
      return null;
    }
    return await root.getDirectoryHandle(subdir, { create: true });
  } catch {
    return null;
  }
}

class RangeCache {
  constructor(options = {}) {
    this.dataBase = normalizeDataBase(options.dataBase);
    this.chunkSize =
      Number.isInteger(options.chunkSize) && options.chunkSize > 0
        ? options.chunkSize
        : DEFAULT_CHUNK_SIZE;
    this.maxBytes =
      Number.isFinite(options.maxBytes) && options.maxBytes >= 0
        ? Math.floor(options.maxBytes)
        : DEFAULT_MAX_BYTES;
    this.maxPrefetchConcurrency =
      Number.isInteger(options.maxPrefetchConcurrency) && options.maxPrefetchConcurrency > 0
        ? options.maxPrefetchConcurrency
        : DEFAULT_PREFETCH_CONCURRENCY;
    this.fetchImpl = options.fetchImpl || (typeof fetch === 'function' ? fetch.bind(globalThis) : null);
    this.opfsDir = options.opfsDir || null;

    this._memCache = new Map();
    this._inFlight = new Map();
    this._opfsEntries = new Map();
    this._keyToOpfsFile = new Map();
    this._expectedChunkLengths = new Map();
    this._fileSizes = new Map();
    this._bytesCached = 0;
    this._stats = {
      hits: 0,
      misses: 0,
      opfsHits: 0,
      evictions: 0,
      quotaEvictions: 0,
      quotaRetrySuccesses: 0,
      corruptEvictions: 0,
    };
  }

  setOpfsDir(opfsDir) {
    const nextDir = opfsDir || null;
    if (nextDir !== this.opfsDir) {
      this._opfsEntries.clear();
      this._keyToOpfsFile.clear();
    }
    this.opfsDir = nextDir;
  }

  _chunkKey(relPath, chunkIndex) {
    return `${relPath}::${chunkIndex}`;
  }

  _recordOpfsEntry(fileName, key, relPath, chunkIndex, byteLength) {
    if (!fileName) return;
    if (this._opfsEntries.has(fileName)) {
      this._opfsEntries.delete(fileName);
    }
    this._opfsEntries.set(fileName, { key, relPath, chunkIndex, byteLength });
    if (key) {
      this._keyToOpfsFile.set(key, fileName);
    }
  }

  _touchOpfsEntryByKey(key) {
    const fileName = this._keyToOpfsFile.get(key);
    if (!fileName || !this._opfsEntries.has(fileName)) return;
    const meta = this._opfsEntries.get(fileName);
    this._opfsEntries.delete(fileName);
    this._opfsEntries.set(fileName, meta);
  }

  _recordExpectedChunkLength(key, byteLength) {
    if (!Number.isInteger(byteLength) || byteLength <= 0) return;
    if (this._expectedChunkLengths.has(key)) {
      this._expectedChunkLengths.delete(key);
    } else if (this._expectedChunkLengths.size >= MAX_TRACKED_CHUNK_LENGTHS) {
      const oldest = this._expectedChunkLengths.keys().next().value;
      if (oldest !== undefined) {
        this._expectedChunkLengths.delete(oldest);
      }
    }
    this._expectedChunkLengths.set(key, byteLength);
  }

  _recordFileSize(relPath, totalSize) {
    if (!Number.isSafeInteger(totalSize) || totalSize < 0) return;
    if (this._fileSizes.has(relPath)) {
      this._fileSizes.delete(relPath);
    } else if (this._fileSizes.size >= MAX_TRACKED_FILE_SIZES) {
      const oldest = this._fileSizes.keys().next().value;
      if (oldest !== undefined) {
        this._fileSizes.delete(oldest);
      }
    }
    this._fileSizes.set(relPath, totalSize);
  }

  _getExpectedChunkLength(key, relPath, chunkIndex) {
    if (this._expectedChunkLengths.has(key)) {
      return this._expectedChunkLengths.get(key);
    }
    if (this._fileSizes.has(relPath)) {
      const total = this._fileSizes.get(relPath);
      const chunkStart = chunkIndex * this.chunkSize;
      return Math.max(0, Math.min(this.chunkSize, total - chunkStart));
    }
    return null;
  }

  _validateCachedChunk(key, relPath, chunkIndex, bytes, requiredEndInChunk) {
    if (!(bytes instanceof Uint8Array)) {
      return 'corrupt';
    }
    if (bytes.byteLength === 0 || bytes.byteLength > this.chunkSize) {
      return 'corrupt';
    }
    const expected = this._getExpectedChunkLength(key, relPath, chunkIndex);
    if (expected !== null) {
      return bytes.byteLength === expected ? 'valid' : 'corrupt';
    }
    if (
      Number.isInteger(requiredEndInChunk) &&
      requiredEndInChunk > 0 &&
      bytes.byteLength < requiredEndInChunk
    ) {
      return 'suspect';
    }
    return 'valid';
  }

  _touchMemory(key) {
    if (!this._memCache.has(key)) return null;
    const bytes = this._memCache.get(key);
    this._memCache.delete(key);
    this._memCache.set(key, bytes);
    this._touchOpfsEntryByKey(key);
    return bytes;
  }

  _putMemory(key, bytes) {
    if (!(bytes instanceof Uint8Array)) return;
    if (this._memCache.has(key)) {
      const existing = this._memCache.get(key);
      if (existing && Number.isFinite(existing.byteLength)) {
        this._bytesCached = Math.max(0, this._bytesCached - existing.byteLength);
      }
      this._memCache.delete(key);
    }

    if (this.maxBytes <= 0 || bytes.byteLength > this.maxBytes) {
      return;
    }

    while (this._bytesCached + bytes.byteLength > this.maxBytes && this._memCache.size > 0) {
      const oldestKey = this._memCache.keys().next().value;
      const oldestBytes = this._memCache.get(oldestKey);
      this._memCache.delete(oldestKey);
      if (oldestBytes && Number.isFinite(oldestBytes.byteLength)) {
        this._bytesCached = Math.max(0, this._bytesCached - oldestBytes.byteLength);
      }
      this._stats.evictions += 1;
    }

    this._memCache.set(key, bytes);
    this._bytesCached += bytes.byteLength;
  }

  async _deleteOpfsFile(fileName) {
    if (!this.opfsDir || !fileName) return false;
    try {
      if (typeof this.opfsDir.removeEntry === 'function') {
        await this.opfsDir.removeEntry(fileName);
        return true;
      }
      if (typeof this.opfsDir.delete === 'function') {
        return Boolean(await this.opfsDir.delete(fileName));
      }
      if (typeof this.opfsDir.remove === 'function') {
        await this.opfsDir.remove(fileName);
        return true;
      }
    } catch {
      // Ignore NotFoundError or unsupported delete failures.
    }
    return false;
  }

  async _listOpfsEntries() {
    if (!this.opfsDir) return [];
    const results = [];
    try {
      if (typeof this.opfsDir.entries === 'function') {
        for await (const pair of this.opfsDir.entries()) {
          if (!Array.isArray(pair) || typeof pair[0] !== 'string') continue;
          const name = pair[0];
          const val = pair[1];
          let byteLength = 0;
          if (val instanceof Uint8Array || val instanceof ArrayBuffer) {
            byteLength = val.byteLength;
          } else if (val && typeof val.getFile === 'function') {
            try {
              const file = await val.getFile();
              if (file && Number.isFinite(file.size)) {
                byteLength = file.size;
              }
            } catch {
              // Fall back to chunkSize estimate.
            }
          }
          results.push({ name, byteLength });
        }
        return results;
      }
      if (typeof this.opfsDir.keys === 'function') {
        for await (const name of this.opfsDir.keys()) {
          if (typeof name === 'string') {
            results.push({ name, byteLength: 0 });
          }
        }
      }
    } catch {
      // Ignore directory enumeration failures.
    }
    return results;
  }

  async _evictCorruptEntry(key, relPath, chunkIndex, countEviction = true) {
    if (this._memCache.has(key)) {
      const existing = this._memCache.get(key);
      this._memCache.delete(key);
      if (existing && Number.isFinite(existing.byteLength)) {
        this._bytesCached = Math.max(0, this._bytesCached - existing.byteLength);
      }
    }
    const fileName = opfsChunkFileName(relPath, chunkIndex, this.chunkSize);
    this._opfsEntries.delete(fileName);
    this._keyToOpfsFile.delete(key);
    await this._deleteOpfsFile(fileName);
    if (countEviction) {
      this._stats.corruptEvictions += 1;
    }
  }

  async _evictForQuota(excludeKey, excludeFileName, targetBytes) {
    let freedBytes = 0;
    let evictedEntries = 0;
    const evictedFileNames = new Set();
    if (excludeFileName) {
      evictedFileNames.add(excludeFileName);
    }

    // 1. Evict oldest tracked OPFS/backing-store entries first (LRU order).
    if (this._opfsEntries.size > 0) {
      const opfsSnapshot = Array.from(this._opfsEntries.entries());
      for (let i = 0; i < opfsSnapshot.length && freedBytes < targetBytes; i += 1) {
        const [fileName, meta] = opfsSnapshot[i];
        if (fileName === excludeFileName || meta.key === excludeKey) {
          continue;
        }
        if (!this._opfsEntries.has(fileName)) {
          continue;
        }
        this._opfsEntries.delete(fileName);
        if (meta.key && this._keyToOpfsFile.get(meta.key) === fileName) {
          this._keyToOpfsFile.delete(meta.key);
        }
        evictedFileNames.add(fileName);

        const entryBytes = Number.isFinite(meta.byteLength) ? meta.byteLength : 0;
        await this._deleteOpfsFile(fileName);
        freedBytes += entryBytes > 0 ? entryBytes : this.chunkSize;
        evictedEntries += 1;
        this._stats.quotaEvictions += 1;
      }
    }

    // 2. If target still not reached, enumerate untracked OPFS cache files (e.g. from prior sessions).
    if (freedBytes < targetBytes && this.opfsDir) {
      const existingEntries = await this._listOpfsEntries();
      for (let i = 0; i < existingEntries.length && freedBytes < targetBytes; i += 1) {
        const entry = existingEntries[i];
        if (evictedFileNames.has(entry.name)) {
          continue;
        }
        if (!/^c_\d+_\d+_.*\.bin$/.test(entry.name)) {
          continue;
        }
        evictedFileNames.add(entry.name);
        this._opfsEntries.delete(entry.name);
        await this._deleteOpfsFile(entry.name);

        freedBytes += entry.byteLength > 0 ? entry.byteLength : this.chunkSize;
        evictedEntries += 1;
        this._stats.quotaEvictions += 1;
      }
    }

    return { freedBytes, evictedEntries };
  }

  async _readOpfsDetailed(relPath, chunkIndex) {
    if (!this.opfsDir) return { bytes: null, corrupt: false };
    const fileName = opfsChunkFileName(relPath, chunkIndex, this.chunkSize);
    if (typeof this.opfsDir.getFileHandle === 'function') {
      let handle;
      try {
        handle = await this.opfsDir.getFileHandle(fileName, { create: false });
      } catch (err) {
        if (isNotFoundError(err)) {
          return { bytes: null, corrupt: false };
        }
        return { bytes: null, corrupt: true };
      }
      try {
        const file = await handle.getFile();
        const buf = await file.arrayBuffer();
        return { bytes: new Uint8Array(buf), corrupt: false };
      } catch {
        return { bytes: null, corrupt: true };
      }
    }
    if (typeof this.opfsDir.get === 'function') {
      try {
        const val = await this.opfsDir.get(fileName);
        if (val === undefined || val === null) {
          return { bytes: null, corrupt: false };
        }
        if (val instanceof Uint8Array) {
          return { bytes: val.slice(), corrupt: false };
        }
        if (val instanceof ArrayBuffer) {
          return { bytes: new Uint8Array(val.slice(0)), corrupt: false };
        }
        return { bytes: null, corrupt: true };
      } catch (err) {
        if (isNotFoundError(err)) {
          return { bytes: null, corrupt: false };
        }
        return { bytes: null, corrupt: true };
      }
    }
    return { bytes: null, corrupt: false };
  }

  async _readOpfs(relPath, chunkIndex) {
    const res = await this._readOpfsDetailed(relPath, chunkIndex);
    return res.bytes;
  }

  async _writeOpfsOnce(fileName, bytes) {
    if (typeof this.opfsDir.getFileHandle === 'function') {
      const handle = await this.opfsDir.getFileHandle(fileName, { create: true });
      if (typeof handle.createWritable === 'function') {
        const writable = await handle.createWritable();
        try {
          await writable.write(bytes);
          await writable.close();
        } catch (err) {
          if (typeof writable.abort === 'function') {
            try {
              await writable.abort();
            } catch {
              // Ignore abort failures during error cleanup.
            }
          }
          throw err;
        }
        return true;
      }
      if (typeof handle.createSyncAccessHandle === 'function') {
        const syncHandle = await handle.createSyncAccessHandle();
        try {
          syncHandle.truncate(0);
          syncHandle.write(bytes, { at: 0 });
          syncHandle.flush();
        } finally {
          syncHandle.close();
        }
        return true;
      }
      return false;
    }
    if (typeof this.opfsDir.set === 'function') {
      await this.opfsDir.set(fileName, bytes.slice());
      return true;
    }
    return false;
  }

  async _writeOpfs(relPath, chunkIndex, bytes) {
    if (!this.opfsDir || !(bytes instanceof Uint8Array) || bytes.byteLength === 0) {
      return false;
    }
    const key = this._chunkKey(relPath, chunkIndex);
    const fileName = opfsChunkFileName(relPath, chunkIndex, this.chunkSize);

    try {
      const wrote = await this._writeOpfsOnce(fileName, bytes);
      if (wrote) {
        this._recordOpfsEntry(fileName, key, relPath, chunkIndex, bytes.byteLength);
      }
      return wrote;
    } catch (err) {
      // Remove any partial/0-byte file left behind by the failed write.
      this._opfsEntries.delete(fileName);
      if (this._keyToOpfsFile.get(key) === fileName) {
        this._keyToOpfsFile.delete(key);
      }
      await this._deleteOpfsFile(fileName);

      if (!isQuotaExceededError(err)) {
        // Ignore non-quota OPFS write failures and continue memory-only.
        return false;
      }

      // QuotaExceededError: evict oldest entries until freeing at least
      // max(entryBytes, ceil(maxBytes * 0.25)) (or all evictable entries), then retry ONCE.
      const targetBytes = Math.max(bytes.byteLength, Math.ceil(this.maxBytes * 0.25));
      await this._evictForQuota(key, fileName, targetBytes);

      try {
        const retryWrote = await this._writeOpfsOnce(fileName, bytes);
        if (retryWrote) {
          this._recordOpfsEntry(fileName, key, relPath, chunkIndex, bytes.byteLength);
          this._stats.quotaRetrySuccesses += 1;
        }
        return retryWrote;
      } catch {
        // Quota remains exhausted: clean up any partial file and degrade gracefully.
        this._opfsEntries.delete(fileName);
        if (this._keyToOpfsFile.get(key) === fileName) {
          this._keyToOpfsFile.delete(key);
        }
        await this._deleteOpfsFile(fileName);
        return false;
      }
    }
  }

  async _fetchChunkFromNetwork(relPath, chunkIndex) {
    if (typeof this.fetchImpl !== 'function') {
      throw new Error('RangeCache has no fetch implementation available');
    }
    const chunkStart = chunkIndex * this.chunkSize;
    const chunkEnd = chunkStart + this.chunkSize - 1;
    const url = `${this.dataBase}${relPath}`;

    const res = await this.fetchImpl(url, {
      headers: { Range: `bytes=${chunkStart}-${chunkEnd}` },
    });

    if (res.status === 416) {
      return new Uint8Array(0);
    }

    if (!res.ok && res.status !== 206) {
      throw new Error(`HTTP ${res.status} reading ${relPath}[${chunkStart}..${chunkEnd}]`);
    }

    const raw = new Uint8Array(await res.arrayBuffer());

    if (res.status === 206) {
      const contentRangeHeader =
        res.headers && typeof res.headers.get === 'function'
          ? res.headers.get('Content-Range') || res.headers.get('content-range')
          : null;
      const parsed = parseContentRange(contentRangeHeader);
      if (parsed) {
        if (Number.isSafeInteger(parsed.total) && parsed.total >= 0) {
          this._recordFileSize(relPath, parsed.total);
        }
        const expectedLen = parsed.end - parsed.start + 1;
        const relOffset = chunkStart >= parsed.start ? chunkStart - parsed.start : 0;
        const available = Math.min(raw.byteLength, expectedLen);
        if (relOffset >= available) {
          return new Uint8Array(0);
        }
        const sliceLen = Math.min(this.chunkSize, available - relOffset);
        return raw.slice(relOffset, relOffset + sliceLen);
      }
      if (raw.byteLength <= this.chunkSize) {
        return raw;
      }
      return raw.slice(0, this.chunkSize);
    }

    // 200 OK fallback (server returned full resource instead of 206 Partial Content).
    this._recordFileSize(relPath, raw.byteLength);
    if (chunkStart >= raw.byteLength) {
      return new Uint8Array(0);
    }
    return raw.slice(chunkStart, Math.min(raw.byteLength, chunkStart + this.chunkSize));
  }

  async _getChunk(relPath, chunkIndex, requiredEndInChunk = 0) {
    const key = this._chunkKey(relPath, chunkIndex);
    let suspectByteLength = null;

    const memHit = this._touchMemory(key);
    if (memHit !== null) {
      const verdict = this._validateCachedChunk(key, relPath, chunkIndex, memHit, requiredEndInChunk);
      if (verdict === 'valid') {
        this._stats.hits += 1;
        return memHit;
      }
      if (verdict === 'corrupt') {
        await this._evictCorruptEntry(key, relPath, chunkIndex, true);
      } else {
        suspectByteLength = memHit.byteLength;
        await this._evictCorruptEntry(key, relPath, chunkIndex, false);
      }
    }

    const existingFlight = this._inFlight.get(key);
    if (existingFlight) {
      const inflightBytes = await existingFlight;
      if (inflightBytes instanceof Uint8Array && inflightBytes.byteLength === 0) {
        return inflightBytes;
      }
      const verdict = this._validateCachedChunk(
        key,
        relPath,
        chunkIndex,
        inflightBytes,
        requiredEndInChunk,
      );
      if (verdict === 'valid') {
        return inflightBytes;
      }
      if (verdict === 'corrupt') {
        await this._evictCorruptEntry(key, relPath, chunkIndex, true);
      } else {
        suspectByteLength = inflightBytes.byteLength;
        await this._evictCorruptEntry(key, relPath, chunkIndex, false);
      }
    }

    const loadPromise = (async () => {
      if (suspectByteLength === null) {
        const opfsResult = await this._readOpfsDetailed(relPath, chunkIndex);
        if (opfsResult.corrupt) {
          await this._evictCorruptEntry(key, relPath, chunkIndex, true);
        } else if (opfsResult.bytes !== null) {
          const verdict = this._validateCachedChunk(
            key,
            relPath,
            chunkIndex,
            opfsResult.bytes,
            requiredEndInChunk,
          );
          if (verdict === 'valid') {
            const fileName = opfsChunkFileName(relPath, chunkIndex, this.chunkSize);
            this._recordOpfsEntry(fileName, key, relPath, chunkIndex, opfsResult.bytes.byteLength);
            this._stats.opfsHits += 1;
            this._stats.hits += 1;
            this._putMemory(key, opfsResult.bytes);
            return opfsResult.bytes;
          }
          if (verdict === 'corrupt') {
            await this._evictCorruptEntry(key, relPath, chunkIndex, true);
          } else {
            suspectByteLength = opfsResult.bytes.byteLength;
            await this._evictCorruptEntry(key, relPath, chunkIndex, false);
          }
        }
      }

      this._stats.misses += 1;
      const netBytes = await this._fetchChunkFromNetwork(relPath, chunkIndex);
      if (suspectByteLength !== null && netBytes.byteLength !== suspectByteLength) {
        this._stats.corruptEvictions += 1;
      }
      if (netBytes.byteLength > 0) {
        this._recordExpectedChunkLength(key, netBytes.byteLength);
        this._putMemory(key, netBytes);
        await this._writeOpfs(relPath, chunkIndex, netBytes);
      }
      return netBytes;
    })();

    this._inFlight.set(key, loadPromise);
    try {
      return await loadPromise;
    } finally {
      if (this._inFlight.get(key) === loadPromise) {
        this._inFlight.delete(key);
      }
    }
  }

  async read(relPath, offset, length) {
    const cleanPath = normalizeRelPath(relPath);
    if (!Number.isInteger(offset) || offset < 0) {
      throw new Error(`Invalid read offset: ${offset}`);
    }
    if (!Number.isInteger(length) || length < 0) {
      throw new Error(`Invalid read length: ${length}`);
    }
    if (length === 0) {
      return new Uint8Array(0);
    }

    let effectiveLength = length;
    if (this._fileSizes.has(cleanPath)) {
      const knownTotal = this._fileSizes.get(cleanPath);
      if (offset >= knownTotal) {
        return new Uint8Array(0);
      }
      effectiveLength = Math.min(length, knownTotal - offset);
    }

    const firstChunk = Math.floor(offset / this.chunkSize);
    const lastChunk = Math.floor((offset + effectiveLength - 1) / this.chunkSize);

    const chunkPromises = [];
    for (let idx = firstChunk; idx <= lastChunk; idx += 1) {
      const chunkStart = idx * this.chunkSize;
      const requiredEndInChunk = Math.min(this.chunkSize, offset + effectiveLength - chunkStart);
      chunkPromises.push(this._getChunk(cleanPath, idx, requiredEndInChunk));
    }
    const chunks = await Promise.all(chunkPromises);

    const slices = [];
    let totalBytes = 0;

    for (let i = 0; i < chunks.length; i += 1) {
      const chunkIndex = firstChunk + i;
      const chunk = chunks[i];
      if (chunk.byteLength === 0) {
        break;
      }
      const chunkStart = chunkIndex * this.chunkSize;
      const sliceStart = Math.max(0, offset - chunkStart);
      const sliceEnd = Math.min(chunk.byteLength, offset + length - chunkStart);
      if (sliceEnd > sliceStart) {
        const part = chunk.subarray(sliceStart, sliceEnd);
        slices.push(part);
        totalBytes += part.byteLength;
      }
      if (chunk.byteLength < this.chunkSize) {
        // Reached tail chunk (EOF).
        break;
      }
    }

    // Always return a fresh owned buffer so callers (e.g. postMessage transferables)
    // can transfer bytes.buffer without detaching cached Uint8Array instances.
    const out = new Uint8Array(totalBytes);
    let cursor = 0;
    for (const part of slices) {
      out.set(part, cursor);
      cursor += part.byteLength;
    }
    return out;
  }

  async prefetchBootset(entries, options = {}) {
    const rawList = Array.isArray(entries)
      ? entries
      : entries && Array.isArray(entries.entries)
        ? entries.entries
        : [];

    const concurrency =
      Number.isInteger(options.concurrency) && options.concurrency > 0
        ? options.concurrency
        : this.maxPrefetchConcurrency;

    const tasks = [];
    const taskIndexByKey = new Map();

    for (const entry of rawList) {
      if (!entry || typeof entry.path !== 'string') continue;
      const offset = Number.isInteger(entry.offset) && entry.offset >= 0 ? entry.offset : 0;
      const length = Number.isInteger(entry.length) && entry.length > 0 ? entry.length : 0;
      if (length === 0) continue;

      let cleanPath;
      try {
        cleanPath = normalizeRelPath(entry.path);
      } catch {
        continue;
      }

      const firstChunk = Math.floor(offset / this.chunkSize);
      const lastChunk = Math.floor((offset + length - 1) / this.chunkSize);
      for (let idx = firstChunk; idx <= lastChunk; idx += 1) {
        const chunkStart = idx * this.chunkSize;
        const requiredEndInChunk = Math.min(this.chunkSize, offset + length - chunkStart);
        const key = this._chunkKey(cleanPath, idx);
        if (!taskIndexByKey.has(key)) {
          taskIndexByKey.set(key, tasks.length);
          tasks.push({ path: cleanPath, chunkIndex: idx, requiredEndInChunk });
        } else {
          const existingTask = tasks[taskIndexByKey.get(key)];
          if (requiredEndInChunk > existingTask.requiredEndInChunk) {
            existingTask.requiredEndInChunk = requiredEndInChunk;
          }
        }
      }
    }

    let nextIndex = 0;
    let prefetched = 0;
    let failed = 0;

    const worker = async () => {
      while (true) {
        const current = nextIndex;
        if (current >= tasks.length) return;
        nextIndex += 1;
        const task = tasks[current];
        try {
          await this._getChunk(task.path, task.chunkIndex, task.requiredEndInChunk);
          prefetched += 1;
        } catch {
          failed += 1;
        }
      }
    };

    const workerCount = Math.min(concurrency, tasks.length);
    const workers = [];
    for (let i = 0; i < workerCount; i += 1) {
      workers.push(worker());
    }
    await Promise.all(workers);

    return {
      requestedEntries: rawList.length,
      uniqueChunks: tasks.length,
      prefetched,
      failed,
    };
  }

  clear() {
    this._memCache.clear();
    this._bytesCached = 0;
  }

  getStats() {
    return {
      hits: this._stats.hits,
      misses: this._stats.misses,
      opfsHits: this._stats.opfsHits,
      evictions: this._stats.evictions,
      quotaEvictions: this._stats.quotaEvictions,
      quotaRetrySuccesses: this._stats.quotaRetrySuccesses,
      corruptEvictions: this._stats.corruptEvictions,
      bytesCached: this._bytesCached,
      entries: this._memCache.size,
    };
  }
}

const exportedApi = {
  RangeCache,
  DEFAULT_CHUNK_SIZE,
  DEFAULT_MAX_BYTES,
  DEFAULT_PREFETCH_CONCURRENCY,
  normalizeRelPath,
  opfsChunkFileName,
  parseContentRange,
  openOpfsCacheDir,
  isQuotaExceededError,
};

if (typeof globalThis !== 'undefined') {
  globalThis.PlayGta6RangeCache = exportedApi;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = exportedApi;
}
