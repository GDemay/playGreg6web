// OPFS chunk cache with batched/coalesced index persistence, O(1) doubly-linked-list
// LRU touch tracking, bounded parallel warm-boot prefetch/hydration with in-flight
// deduplication, and sanitized telemetry.
// Shares the OPFS directory ('playgta6-cache') and JSON metadata index schema
// ('.playgta6-cache-index.json', version 1) with web/storage/storage_manager.js and
// web/workers/range_cache.js. Zero external dependencies; browser & Node testable.

export const DEFAULT_OPFS_CACHE_DIR = 'playgta6-cache';
export const DEFAULT_INDEX_FILENAME = '.playgta6-cache-index.json';
export const OPFS_CACHE_INDEX_VERSION = 1;
export const DEFAULT_MAX_BYTES = 512 * 1024 * 1024; // 512 MiB
export const DEFAULT_INDEX_FLUSH_DELAY_MS = 0;
export const DEFAULT_PREFETCH_CONCURRENCY = 4;

export const CHUNK_KINDS = Object.freeze({
  SHADER_PACK: 'shader-pack',
  PGIM: 'pgim',
  RANGE_CHUNK: 'range-chunk',
  GENERIC: 'generic',
});

function compareAscii(a, b) {
  return a < b ? -1 : a > b ? 1 : 0;
}

function isNotFoundError(err) {
  if (!err) return false;
  return (
    err.name === 'NotFoundError' ||
    err.code === 'ENOENT' ||
    err.code === 8 ||
    /not\s*found/i.test(String(err.message || ''))
  );
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

function sanitizeNonNegativeInt(value, fallback = 0) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    return fallback;
  }
  return Math.floor(value);
}

function sanitizePositiveInt(value, fallback) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 1) {
    return fallback;
  }
  return Math.floor(value);
}

function normalizeOptionalTag(value) {
  if (typeof value !== 'string') {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function sanitizeErrorCode(err) {
  if (!err) return 'unknown_error';
  if (typeof err.name === 'string' && err.name.length > 0) {
    return err.name;
  }
  return 'Error';
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
  throw new TypeError('Chunk payload must be a Uint8Array, ArrayBufferView, ArrayBuffer, or string');
}

function utf8ToHex(str) {
  const bytes = new TextEncoder().encode(str);
  let hex = '';
  for (let i = 0; i < bytes.length; i += 1) {
    hex += bytes[i].toString(16).padStart(2, '0');
  }
  return hex;
}

function fnv1a128Hex(str) {
  // Deterministic 128-bit four-lane FNV-1a hash matching web/workers/range_cache.js.
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
    h4 = Math.imul(h4 ^ ((b + i * 31) & 0xff), 0x01000193) >>> 0;
  }
  return (
    h1.toString(16).padStart(8, '0') +
    h2.toString(16).padStart(8, '0') +
    h3.toString(16).padStart(8, '0') +
    h4.toString(16).padStart(8, '0')
  );
}

function normalizeRelPath(relPath) {
  if (typeof relPath !== 'string' || relPath.length === 0) {
    throw new Error('Chunk path must be a non-empty string');
  }
  if (relPath.includes('\\') || relPath.includes('\0')) {
    throw new Error(`Invalid chunk path: ${relPath}`);
  }
  const stripped = relPath.replace(/^\/+/, '');
  if (!stripped || stripped.endsWith('/')) {
    throw new Error(`Invalid chunk path: ${relPath}`);
  }
  const parts = stripped.split('/');
  for (let i = 0; i < parts.length; i += 1) {
    const part = parts[i];
    if (!part || part === '.' || part === '..') {
      throw new Error(`Invalid chunk path segment in: ${relPath}`);
    }
  }
  return stripped;
}

/**
 * Validates that a chunk key is a clean, relative, traversal-free OPFS path
 * that does not collide with the internal index filename.
 *
 * @param {string} key
 * @param {string} [indexFileName=DEFAULT_INDEX_FILENAME]
 * @returns {string}
 */
export function validateChunkKey(key, indexFileName = DEFAULT_INDEX_FILENAME) {
  if (typeof key !== 'string' || key.length === 0) {
    throw new Error('Chunk key must be a non-empty string');
  }
  if (key.includes('\\') || key.includes('\0')) {
    throw new Error(`Invalid chunk key (forbidden character): ${key}`);
  }
  if (key.startsWith('/') || key.endsWith('/')) {
    throw new Error(`Chunk key must be a relative path without leading/trailing slash: ${key}`);
  }
  const segments = key.split('/');
  for (let i = 0; i < segments.length; i += 1) {
    const seg = segments[i];
    if (!seg || seg === '.' || seg === '..') {
      throw new Error(`Invalid chunk key segment in: ${key}`);
    }
  }
  if (key === indexFileName || key === DEFAULT_INDEX_FILENAME) {
    throw new Error(`Chunk key "${key}" is reserved for the internal cache index`);
  }
  return key;
}

/**
 * Infers coarse cache entry kind from a chunk key, matching BrowserStorageManager.
 *
 * @param {string} key
 * @returns {'pgim'|'shader-pack'|'range-chunk'|'generic'}
 */
export function inferChunkKind(key) {
  if (typeof key !== 'string' || key.length === 0) {
    return CHUNK_KINDS.GENERIC;
  }
  const lower = key.toLowerCase();
  if (lower.endsWith('.pgim')) {
    return CHUNK_KINDS.PGIM;
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
    return CHUNK_KINDS.SHADER_PACK;
  }
  const leaf = key.includes('/') ? key.slice(key.lastIndexOf('/') + 1) : key;
  if (/^[cr]_\d+_\d+_.*\.bin$/.test(leaf)) {
    return CHUNK_KINDS.RANGE_CHUNK;
  }
  return CHUNK_KINDS.GENERIC;
}

function normalizeChunkKind(kind, key) {
  if (typeof kind === 'string') {
    const trimmed = kind.trim();
    if (
      trimmed === CHUNK_KINDS.PGIM ||
      trimmed === CHUNK_KINDS.SHADER_PACK ||
      trimmed === CHUNK_KINDS.RANGE_CHUNK ||
      trimmed === CHUNK_KINDS.GENERIC
    ) {
      return trimmed;
    }
  }
  return inferChunkKind(key);
}

/**
 * Normalizes a string key or chunk descriptor (`{ key }`, `{ path, chunkIndex, chunkSize }`,
 * or `{ path, offset, length }`) into a deterministic validated OPFS cache key.
 *
 * @param {string|object} descriptor
 * @param {string} [indexFileName=DEFAULT_INDEX_FILENAME]
 * @returns {string}
 */
export function makeChunkKey(descriptor, indexFileName = DEFAULT_INDEX_FILENAME) {
  if (typeof descriptor === 'string') {
    return validateChunkKey(descriptor, indexFileName);
  }
  if (!descriptor || typeof descriptor !== 'object') {
    throw new Error('Chunk descriptor must be a non-empty string or object');
  }
  if (typeof descriptor.key === 'string' && descriptor.key.length > 0) {
    return validateChunkKey(descriptor.key, indexFileName);
  }
  if (typeof descriptor.path === 'string' && descriptor.path.length > 0) {
    const normPath = normalizeRelPath(descriptor.path);
    const encoded = utf8ToHex(normPath);
    const pathToken =
      encoded.length <= 128 ? encoded : `${encoded.slice(0, 64)}_${fnv1a128Hex(normPath)}`;

    if (Number.isInteger(descriptor.chunkIndex) && descriptor.chunkIndex >= 0) {
      const chunkSize =
        Number.isInteger(descriptor.chunkSize) && descriptor.chunkSize > 0
          ? descriptor.chunkSize
          : 0;
      return validateChunkKey(
        `c_${chunkSize}_${descriptor.chunkIndex}_${pathToken}.bin`,
        indexFileName
      );
    }

    if (
      Number.isInteger(descriptor.offset) &&
      descriptor.offset >= 0 &&
      Number.isInteger(descriptor.length) &&
      descriptor.length > 0
    ) {
      return validateChunkKey(
        `r_${descriptor.offset}_${descriptor.length}_${pathToken}.bin`,
        indexFileName
      );
    }
  }
  throw new Error('Chunk descriptor must include a valid "key" or "path" with chunkIndex/range');
}

function cloneNodeMeta(node) {
  if (!node) return null;
  return {
    key: node.key,
    sizeBytes: node.sizeBytes,
    lastAccessedMs: node.lastAccessedMs,
    createdAtMs: node.createdAtMs,
    etag: node.etag,
    versionKey: node.versionKey,
    pinned: node.pinned,
    kind: node.kind,
    accessOrder: node.accessOrder,
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

export class OpfsChunkCache {
  /**
   * @param {object} [options]
   * @param {FileSystemDirectoryHandle|null} [options.opfsRoot] Injectable OPFS root directory handle.
   * @param {FileSystemDirectoryHandle|null} [options.cacheDir] Optional direct handle to the cache directory.
   * @param {StorageManager|null} [options.storageObj] Injectable navigator.storage equivalent.
   * @param {string} [options.dirName='playgta6-cache'] OPFS cache subdirectory name.
   * @param {string} [options.indexFileName='.playgta6-cache-index.json'] Metadata index filename.
   * @param {number|null} [options.maxBytes=536870912] Byte budget for LRU eviction (`null` disables automatic eviction).
   * @param {number} [options.indexFlushDelayMs=0] `0` for immediate backwards-compatible flush; `> 0` for debounced batching.
   * @param {boolean} [options.persistIndex=true] Persist chunk index metadata JSON to OPFS.
   * @param {boolean} [options.reconcileOnInit=true] Reconcile persisted index against OPFS files on init.
   * @param {() => number} [options.nowFn] Injectable clock function returning epoch ms.
   * @param {() => number} [options.now] Fallback clock alias.
   * @param {(fn: Function, delayMs: number) => any} [options.setTimeoutFn] Injectable setTimeout.
   * @param {(id: any) => void} [options.clearTimeoutFn] Injectable clearTimeout.
   */
  constructor(options = {}) {
    const opts = options && typeof options === 'object' ? options : {};

    this.storageObj =
      opts.storageObj !== undefined
        ? opts.storageObj
        : typeof navigator !== 'undefined' && navigator && navigator.storage
          ? navigator.storage
          : null;

    this._opfsRootExplicitlyNull = opts.opfsRoot === null && opts.cacheDir === undefined;
    this.opfsRoot = opts.opfsRoot !== undefined ? opts.opfsRoot : null;
    this._directCacheDir = opts.cacheDir !== undefined ? opts.cacheDir : null;

    this.dirName =
      typeof opts.dirName === 'string' && opts.dirName.length > 0
        ? opts.dirName
        : DEFAULT_OPFS_CACHE_DIR;

    this.indexFileName =
      typeof opts.indexFileName === 'string' && opts.indexFileName.length > 0
        ? opts.indexFileName
        : DEFAULT_INDEX_FILENAME;

    if (opts.maxBytes === null) {
      this.maxBytes = null;
    } else if (typeof opts.maxBytes === 'number' && Number.isFinite(opts.maxBytes) && opts.maxBytes >= 0) {
      this.maxBytes = Math.floor(opts.maxBytes);
    } else {
      this.maxBytes = DEFAULT_MAX_BYTES;
    }

    this.indexFlushDelayMs = sanitizeNonNegativeInt(
      opts.indexFlushDelayMs,
      DEFAULT_INDEX_FLUSH_DELAY_MS
    );
    this.persistIndex = opts.persistIndex !== false;
    this.reconcileOnInit = opts.reconcileOnInit !== false;

    this._nowFn =
      typeof opts.nowFn === 'function'
        ? opts.nowFn
        : typeof opts.now === 'function'
          ? opts.now
          : () => Date.now();

    this._setTimeout =
      typeof opts.setTimeoutFn === 'function'
        ? opts.setTimeoutFn
        : (fn, delay) => setTimeout(fn, delay);
    this._clearTimeout =
      typeof opts.clearTimeoutFn === 'function'
        ? opts.clearTimeoutFn
        : (id) => clearTimeout(id);

    // O(1) LRU index: Map<key, node> + intrusive doubly-linked list (_head = LRU, _tail = MRU).
    this._entries = new Map();
    this._head = null;
    this._tail = null;
    this._totalBytes = 0;
    this._pinnedCount = 0;
    this._pinnedBytes = 0;
    this._accessSeq = 0;

    // Lifecycle & batched index persistence state.
    this._initialized = false;
    this._supported = false;
    this._initPromise = null;
    this._indexDirty = false;
    this._flushTimer = null;
    this._activeFlushPromise = null;

    // In-flight prefetch deduplication map: Map<key, Promise<{ok, bytes, error}>>.
    this._inFlightPrefetches = new Map();

    // Sanitized warm-boot telemetry counters.
    this._hits = 0;
    this._misses = 0;
    this._puts = 0;
    this._deletes = 0;
    this._evictions = 0;
    this._evictedBytes = 0;
    this._invalidations = 0;
    this._indexFlushes = 0;
    this._coalescedFlushes = 0;
    this._flushErrors = 0;
    this._prefetchRequested = 0;
    this._prefetchHits = 0;
    this._prefetchFetched = 0;
    this._prefetchInFlightDeduped = 0;
    this._prefetchFailed = 0;
    this._prefetchBytes = 0;
  }

  // --- O(1) Doubly-Linked-List LRU Primitives ---

  _readNowMs() {
    const v = this._nowFn();
    return typeof v === 'number' && Number.isFinite(v) ? Math.max(0, Math.floor(v)) : 0;
  }

  _nextAccessOrder() {
    this._accessSeq += 1;
    return this._accessSeq;
  }

  _linkTail(node) {
    node.prev = this._tail;
    node.next = null;
    if (this._tail !== null) {
      this._tail.next = node;
    } else {
      this._head = node;
    }
    this._tail = node;
  }

  _unlinkNode(node) {
    if (node.prev !== null) {
      node.prev.next = node.next;
    } else {
      this._head = node.next;
    }
    if (node.next !== null) {
      node.next.prev = node.prev;
    } else {
      this._tail = node.prev;
    }
    node.prev = null;
    node.next = null;
  }

  _moveToTail(node) {
    if (node === this._tail) {
      return;
    }
    this._unlinkNode(node);
    this._linkTail(node);
  }

  _upsertNode(meta) {
    const existing = this._entries.get(meta.key);
    if (existing) {
      this._totalBytes -= existing.sizeBytes;
      if (existing.pinned) {
        this._pinnedCount -= 1;
        this._pinnedBytes -= existing.sizeBytes;
      }
      existing.sizeBytes = meta.sizeBytes;
      existing.lastAccessedMs = meta.lastAccessedMs;
      existing.createdAtMs = meta.createdAtMs;
      existing.etag = meta.etag;
      existing.versionKey = meta.versionKey;
      existing.pinned = meta.pinned;
      existing.kind = meta.kind;
      existing.accessOrder = meta.accessOrder;

      this._totalBytes += existing.sizeBytes;
      if (existing.pinned) {
        this._pinnedCount += 1;
        this._pinnedBytes += existing.sizeBytes;
      }
      this._moveToTail(existing);
      return existing;
    }

    const node = {
      key: meta.key,
      sizeBytes: meta.sizeBytes,
      lastAccessedMs: meta.lastAccessedMs,
      createdAtMs: meta.createdAtMs,
      etag: meta.etag,
      versionKey: meta.versionKey,
      pinned: meta.pinned,
      kind: meta.kind,
      accessOrder: meta.accessOrder,
      prev: null,
      next: null,
    };
    this._entries.set(node.key, node);
    this._linkTail(node);
    this._totalBytes += node.sizeBytes;
    if (node.pinned) {
      this._pinnedCount += 1;
      this._pinnedBytes += node.sizeBytes;
    }
    return node;
  }

  _deleteNode(node) {
    if (!node || !this._entries.has(node.key)) {
      return false;
    }
    this._entries.delete(node.key);
    this._unlinkNode(node);
    this._totalBytes = Math.max(0, this._totalBytes - node.sizeBytes);
    if (node.pinned) {
      this._pinnedCount = Math.max(0, this._pinnedCount - 1);
      this._pinnedBytes = Math.max(0, this._pinnedBytes - node.sizeBytes);
    }
    return true;
  }

  _clearList() {
    this._entries.clear();
    this._head = null;
    this._tail = null;
    this._totalBytes = 0;
    this._pinnedCount = 0;
    this._pinnedBytes = 0;
  }

  _touchNode(node, nowMs = this._readNowMs()) {
    node.lastAccessedMs = nowMs;
    node.accessOrder = this._nextAccessOrder();
    this._moveToTail(node);
    return node;
  }

  // --- OPFS Directory & File Helpers ---

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

  async _ensureCacheDir({ create = false } = {}) {
    if (this._directCacheDir) {
      this._supported = true;
      return { supported: true, dir: this._directCacheDir };
    }
    const root = await this._resolveOpfsRoot();
    if (!root) {
      this._supported = false;
      return { supported: false, dir: null };
    }
    try {
      const dir = await root.getDirectoryHandle(this.dirName, { create });
      this._supported = true;
      return { supported: true, dir };
    } catch (err) {
      if (!create && isNotFoundError(err)) {
        this._supported = true;
        return { supported: true, dir: null };
      }
      this._supported = false;
      return { supported: false, dir: null };
    }
  }

  async _getFileHandle(cacheDir, key, { create = false } = {}) {
    const segments = key.split('/');
    let dir = cacheDir;
    for (let i = 0; i < segments.length - 1; i += 1) {
      dir = await dir.getDirectoryHandle(segments[i], { create });
    }
    const leaf = segments[segments.length - 1];
    return dir.getFileHandle(leaf, { create });
  }

  async _writeFile(cacheDir, key, bytes) {
    const handle = await this._getFileHandle(cacheDir, key, { create: true });
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

  async _readFile(cacheDir, key) {
    let handle;
    try {
      handle = await this._getFileHandle(cacheDir, key, { create: false });
    } catch (err) {
      if (isNotFoundError(err)) {
        return null;
      }
      throw err;
    }
    return readFileHandleBytes(handle);
  }

  async _removeFile(cacheDir, key) {
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
    const leaf = segments[segments.length - 1];
    if (typeof dir.removeEntry !== 'function') {
      return false;
    }
    let removed = false;
    try {
      await dir.removeEntry(leaf);
      removed = true;
    } catch (err) {
      if (!isNotFoundError(err)) {
        throw err;
      }
    }

    // Best-effort prune of empty nested directories.
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

  async _collectDiskEntries(dirHandle, prefix = '', out = []) {
    for await (const [name, handle] of iterateDirEntries(dirHandle)) {
      if (!handle || typeof name !== 'string') continue;
      if (prefix === '' && name === this.indexFileName && handle.kind !== 'directory') {
        continue;
      }
      const rel = prefix ? `${prefix}/${name}` : name;
      if (handle.kind === 'directory') {
        await this._collectDiskEntries(handle, rel, out);
      } else if (handle.kind === 'file' || typeof handle.getFile === 'function') {
        const meta = await readFileHandleMeta(handle);
        out.push({ name: rel, size: meta.size, lastModified: meta.lastModified });
      }
    }
    return out;
  }

  // --- Batched / Coalesced Index Persistence ---

  async _readIndexFile(cacheDir) {
    if (!this.persistIndex || !cacheDir || typeof cacheDir.getFileHandle !== 'function') {
      return null;
    }
    try {
      const handle = await cacheDir.getFileHandle(this.indexFileName, { create: false });
      const bytes = await readFileHandleBytes(handle);
      if (!bytes || bytes.byteLength === 0) {
        return null;
      }
      const parsed = JSON.parse(new TextDecoder().decode(bytes));
      if (
        !parsed ||
        typeof parsed !== 'object' ||
        parsed.version !== OPFS_CACHE_INDEX_VERSION ||
        !Array.isArray(parsed.entries)
      ) {
        return null;
      }
      return parsed.entries;
    } catch {
      return null;
    }
  }

  _serializeIndexPayload() {
    const entries = [];
    for (let node = this._head; node !== null; node = node.next) {
      entries.push({
        key: node.key,
        sizeBytes: node.sizeBytes,
        lastAccessedMs: node.lastAccessedMs,
        createdAtMs: node.createdAtMs,
        etag: node.etag,
        versionKey: node.versionKey,
        pinned: node.pinned,
        kind: node.kind,
        accessOrder: node.accessOrder,
      });
    }
    return new TextEncoder().encode(
      JSON.stringify({
        version: OPFS_CACHE_INDEX_VERSION,
        entries,
      })
    );
  }

  _cancelFlushTimer() {
    if (this._flushTimer !== null) {
      this._clearTimeout(this._flushTimer);
      this._flushTimer = null;
    }
  }

  _markIndexDirty({ syncOnly = false } = {}) {
    if (!this.persistIndex) {
      this._indexDirty = true;
      return null;
    }

    const hadPending =
      this._indexDirty || this._flushTimer !== null || this._activeFlushPromise !== null;
    if (hadPending) {
      this._coalescedFlushes += 1;
    }
    this._indexDirty = true;

    if (this.indexFlushDelayMs > 0) {
      this._cancelFlushTimer();
      this._flushTimer = this._setTimeout(() => {
        this._flushTimer = null;
        this.flushIndex().catch(() => {
          // Recorded in flushErrors inside flushIndex().
        });
      }, this.indexFlushDelayMs);
      return null;
    }

    if (syncOnly) {
      const p = this.flushIndex().catch(() => {
        // Recorded in flushErrors inside flushIndex().
      });
      return p;
    }

    return this.flushIndex();
  }

  /**
   * Immediately persists any pending chunk-index metadata changes to OPFS,
   * cancelling any scheduled debounce timer and coalescing concurrent flushes.
   *
   * @param {object} [options]
   * @param {boolean} [options.force=false] Force an OPFS index write even if not dirty.
   * @returns {Promise<{flushed: boolean, supported: boolean, entryCount: number, totalBytes: number}>}
   */
  async flushIndex(options = {}) {
    const force = Boolean(options && options.force);
    this._cancelFlushTimer();

    if (!this.persistIndex) {
      return {
        flushed: false,
        supported: this._supported,
        entryCount: this._entries.size,
        totalBytes: this._totalBytes,
      };
    }

    if (force) {
      this._indexDirty = true;
    }

    if (!this._indexDirty && !this._activeFlushPromise) {
      return {
        flushed: false,
        supported: this._supported,
        entryCount: this._entries.size,
        totalBytes: this._totalBytes,
      };
    }

    while (this._activeFlushPromise) {
      await this._activeFlushPromise;
      if (!this._indexDirty) {
        return {
          flushed: true,
          supported: this._supported,
          entryCount: this._entries.size,
          totalBytes: this._totalBytes,
        };
      }
    }

    let didFlush = false;
    const runFlushLoop = async () => {
      while (this._indexDirty) {
        this._cancelFlushTimer();
        // Clear dirty flag before awaiting OPFS write so mutations arriving during
        // the write re-mark dirty and trigger one clean follow-up pass.
        this._indexDirty = false;
        const { supported, dir: cacheDir } = await this._ensureCacheDir({
          create: this._entries.size > 0 || force,
        });
        if (!supported || !cacheDir) {
          return false;
        }
        try {
          const payload = this._serializeIndexPayload();
          await this._writeFile(cacheDir, this.indexFileName, payload);
          this._indexFlushes += 1;
          didFlush = true;
        } catch {
          this._flushErrors += 1;
          this._indexDirty = true;
          return false;
        }
      }
      return didFlush;
    };

    this._activeFlushPromise = runFlushLoop();
    try {
      await this._activeFlushPromise;
    } finally {
      this._activeFlushPromise = null;
    }

    return {
      flushed: didFlush,
      supported: this._supported,
      entryCount: this._entries.size,
      totalBytes: this._totalBytes,
    };
  }

  // --- Initialization & Hydration ---

  /**
   * Loads persisted index metadata from OPFS and optionally reconciles it against
   * actual files in the OPFS cache directory.
   *
   * @param {object} [options]
   * @param {boolean} [options.force=false] Re-read index even if already initialized.
   * @param {boolean} [options.reconcile] Override `reconcileOnInit` for this call.
   */
  async init(options = {}) {
    const force = Boolean(options && options.force);
    const reconcile =
      options && options.reconcile !== undefined
        ? Boolean(options.reconcile)
        : this.reconcileOnInit;

    if (this._initialized && !force) {
      return {
        supported: this._supported,
        initialized: true,
        entryCount: this._entries.size,
        totalBytes: this._totalBytes,
        loadedFromIndex: false,
        discoveredCount: 0,
        prunedCount: 0,
      };
    }

    if (this._initPromise && !force) {
      return this._initPromise;
    }

    const doInit = async () => {
      const { supported, dir: cacheDir } = await this._ensureCacheDir({ create: false });
      if (!supported) {
        this._initialized = true;
        return {
          supported: false,
          initialized: true,
          entryCount: 0,
          totalBytes: 0,
          loadedFromIndex: false,
          discoveredCount: 0,
          prunedCount: 0,
        };
      }

      if (!cacheDir) {
        const prunedCount = this._entries.size;
        this._clearList();
        this._initialized = true;
        return {
          supported: true,
          initialized: true,
          entryCount: 0,
          totalBytes: 0,
          loadedFromIndex: false,
          discoveredCount: 0,
          prunedCount,
        };
      }

      if (force) {
        this._clearList();
      }

      let loadedFromIndex = false;
      const storedEntries = await this._readIndexFile(cacheDir);
      if (storedEntries) {
        loadedFromIndex = true;
        const normalized = [];
        for (let i = 0; i < storedEntries.length; i += 1) {
          const raw = storedEntries[i];
          if (!raw || typeof raw !== 'object' || typeof raw.key !== 'string') continue;
          let validKey;
          try {
            validKey = validateChunkKey(raw.key, this.indexFileName);
          } catch {
            continue;
          }
          const lastAccessedMs = sanitizeNonNegativeInt(raw.lastAccessedMs, 0);
          const createdAtMs = sanitizeNonNegativeInt(raw.createdAtMs, lastAccessedMs);
          const accessOrder =
            Number.isInteger(raw.accessOrder) && raw.accessOrder > 0 ? raw.accessOrder : 0;
          normalized.push({
            key: validKey,
            sizeBytes: sanitizeNonNegativeInt(raw.sizeBytes, 0),
            lastAccessedMs,
            createdAtMs,
            etag: normalizeOptionalTag(raw.etag),
            versionKey: normalizeOptionalTag(raw.versionKey),
            pinned: Boolean(raw.pinned),
            kind: normalizeChunkKind(raw.kind, validKey),
            accessOrder,
          });
        }

        // Sort ascending by LRU order (oldest first -> _head, newest last -> _tail).
        normalized.sort((a, b) => {
          if (a.accessOrder !== b.accessOrder) {
            return a.accessOrder - b.accessOrder;
          }
          if (a.lastAccessedMs !== b.lastAccessedMs) {
            return a.lastAccessedMs - b.lastAccessedMs;
          }
          return compareAscii(a.key, b.key);
        });

        for (let i = 0; i < normalized.length; i += 1) {
          const item = normalized[i];
          if (item.accessOrder <= 0) {
            item.accessOrder = this._nextAccessOrder();
          } else if (item.accessOrder > this._accessSeq) {
            this._accessSeq = item.accessOrder;
          }
          this._upsertNode(item);
        }
      }

      let discoveredCount = 0;
      let prunedCount = 0;

      if (reconcile) {
        const diskEntries = await this._collectDiskEntries(cacheDir);
        diskEntries.sort((a, b) => compareAscii(a.name, b.name));

        const presentKeys = new Set();
        for (let i = 0; i < diskEntries.length; i += 1) {
          const file = diskEntries[i];
          let validKey;
          try {
            validKey = validateChunkKey(file.name, this.indexFileName);
          } catch {
            continue;
          }
          presentKeys.add(validKey);
          const existing = this._entries.get(validKey);
          if (existing) {
            if (existing.sizeBytes !== file.size) {
              this._totalBytes = Math.max(0, this._totalBytes - existing.sizeBytes) + file.size;
              if (existing.pinned) {
                this._pinnedBytes = Math.max(0, this._pinnedBytes - existing.sizeBytes) + file.size;
              }
              existing.sizeBytes = file.size;
            }
          } else {
            const ts = sanitizeNonNegativeInt(file.lastModified, 0);
            this._upsertNode({
              key: validKey,
              sizeBytes: file.size,
              lastAccessedMs: ts,
              createdAtMs: ts,
              etag: null,
              versionKey: null,
              pinned: false,
              kind: inferChunkKind(validKey),
              accessOrder: this._nextAccessOrder(),
            });
            discoveredCount += 1;
          }
        }

        for (const [key, node] of Array.from(this._entries.entries())) {
          if (!presentKeys.has(key)) {
            this._deleteNode(node);
            prunedCount += 1;
          }
        }
      }

      this._initialized = true;
      if (discoveredCount > 0 || prunedCount > 0) {
        await this._markIndexDirty();
      }

      return {
        supported: true,
        initialized: true,
        entryCount: this._entries.size,
        totalBytes: this._totalBytes,
        loadedFromIndex,
        discoveredCount,
        prunedCount,
      };
    };

    this._initPromise = doInit();
    try {
      return await this._initPromise;
    } finally {
      this._initPromise = null;
    }
  }

  // --- Synchronous O(1) Inspection & Touch Helpers ---

  /**
   * O(1) in-memory presence check (call `init()` first to hydrate persisted entries).
   *
   * @param {string|object} keyOrDescriptor
   * @returns {boolean}
   */
  has(keyOrDescriptor) {
    const key = makeChunkKey(keyOrDescriptor, this.indexFileName);
    return this._entries.has(key);
  }

  /**
   * Returns cloned metadata for a cached chunk in O(1), or `null` if absent.
   *
   * @param {string|object} keyOrDescriptor
   */
  getMeta(keyOrDescriptor) {
    const key = makeChunkKey(keyOrDescriptor, this.indexFileName);
    return cloneNodeMeta(this._entries.get(key) || null);
  }

  /**
   * Returns all tracked keys ordered from least-recently-used to most-recently-used.
   *
   * @returns {string[]}
   */
  getLruKeys() {
    const keys = [];
    for (let node = this._head; node !== null; node = node.next) {
      keys.push(node.key);
    }
    return keys;
  }

  /**
   * O(1) LRU touch promoting an existing chunk to the MRU tail and marking the index dirty.
   *
   * @param {string|object} keyOrDescriptor
   * @param {object} [options]
   * @param {boolean} [options.persist=true] Mark index dirty / schedule persistence.
   * @returns {boolean}
   */
  touch(keyOrDescriptor, options = {}) {
    const key = makeChunkKey(keyOrDescriptor, this.indexFileName);
    const node = this._entries.get(key);
    if (!node) {
      return false;
    }
    this._touchNode(node, this._readNowMs());
    if (options.persist !== false) {
      this._markIndexDirty({ syncOnly: true });
    }
    return true;
  }

  // --- Core Cache Operations ---

  /**
   * Stores a chunk in OPFS, updates O(1) LRU metadata, enforces `maxBytes` via LRU eviction
   * (with automatic `QuotaExceededError` recovery), and persists/batches index metadata.
   *
   * @param {string|object} keyOrDescriptor
   * @param {Uint8Array|ArrayBufferView|ArrayBuffer|string} data
   * @param {object} [options]
   */
  async put(keyOrDescriptor, data, options = {}) {
    const key = makeChunkKey(keyOrDescriptor, this.indexFileName);
    const bytes = toUint8Array(data);
    const opts = options && typeof options === 'object' ? options : {};

    if (!this._initialized) {
      await this.init();
    }

    const { supported, dir: cacheDir } = await this._ensureCacheDir({ create: true });
    if (!supported || !cacheDir) {
      return {
        supported: false,
        stored: false,
        key,
        sizeBytes: 0,
        replaced: false,
        quotaRecovered: false,
        evictedEntries: 0,
        evictedBytes: 0,
        meta: null,
      };
    }

    const existing = this._entries.get(key) || null;
    const replaced = existing !== null;
    const etag = normalizeOptionalTag(
      opts.etag !== undefined
        ? opts.etag
        : typeof keyOrDescriptor === 'object' && keyOrDescriptor
          ? keyOrDescriptor.etag
          : null
    );
    const versionKey = normalizeOptionalTag(
      opts.versionKey !== undefined
        ? opts.versionKey
        : typeof keyOrDescriptor === 'object' && keyOrDescriptor
          ? keyOrDescriptor.versionKey
          : null
    );
    const pinned =
      opts.pinned !== undefined
        ? Boolean(opts.pinned)
        : typeof keyOrDescriptor === 'object' && keyOrDescriptor && keyOrDescriptor.pinned !== undefined
          ? Boolean(keyOrDescriptor.pinned)
          : false;
    const kind = normalizeChunkKind(
      opts.kind !== undefined
        ? opts.kind
        : typeof keyOrDescriptor === 'object' && keyOrDescriptor
          ? keyOrDescriptor.kind
          : undefined,
      key
    );

    const versionChanged =
      replaced &&
      ((existing.etag !== etag && (existing.etag !== null || etag !== null)) ||
        (existing.versionKey !== versionKey &&
          (existing.versionKey !== null || versionKey !== null)));

    let evictedEntries = 0;
    let evictedBytes = 0;
    let quotaRecovered = false;

    try {
      await this._writeFile(cacheDir, key, bytes);
    } catch (err) {
      if (!isQuotaExceededError(err)) {
        throw err;
      }
      const targetBudget = Math.max(0, this._totalBytes - Math.max(bytes.byteLength, 1));
      const recovery = await this.evictToBudget(targetBudget, {
        excludeKeys: [key],
        deferFlush: true,
      });
      evictedEntries += recovery.evictedEntries;
      evictedBytes += recovery.evictedBytes;
      await this._writeFile(cacheDir, key, bytes);
      quotaRecovered = true;
    }

    if (versionChanged) {
      this._invalidations += 1;
    }

    const nowMs = this._readNowMs();
    const node = this._upsertNode({
      key,
      sizeBytes: bytes.byteLength,
      lastAccessedMs: nowMs,
      createdAtMs: existing && !versionChanged ? existing.createdAtMs : nowMs,
      etag,
      versionKey,
      pinned,
      kind,
      accessOrder: this._nextAccessOrder(),
    });
    this._puts += 1;

    if (opts.autoEvict !== false) {
      const effectiveBudget =
        opts.maxBytes === null
          ? null
          : typeof opts.maxBytes === 'number' && Number.isFinite(opts.maxBytes) && opts.maxBytes >= 0
            ? Math.floor(opts.maxBytes)
            : this.maxBytes;

      if (effectiveBudget !== null && this._totalBytes > effectiveBudget) {
        const ev = await this.evictToBudget(effectiveBudget, {
          excludeKeys: [key],
          deferFlush: true,
        });
        evictedEntries += ev.evictedEntries;
        evictedBytes += ev.evictedBytes;
      }
    }

    await this._markIndexDirty();

    return {
      supported: true,
      stored: true,
      key,
      sizeBytes: bytes.byteLength,
      replaced,
      quotaRecovered,
      evictedEntries,
      evictedBytes,
      meta: cloneNodeMeta(node),
    };
  }

  /**
   * Reads a cached chunk with rich hit/invalidation metadata and O(1) LRU touch on hit.
   *
   * @param {string|object} keyOrDescriptor
   * @param {object} [options]
   */
  async getEntry(keyOrDescriptor, options = {}) {
    const key = makeChunkKey(keyOrDescriptor, this.indexFileName);
    const opts = options && typeof options === 'object' ? options : {};

    if (!this._initialized) {
      await this.init();
    }

    const { supported, dir: cacheDir } = await this._ensureCacheDir({ create: false });
    if (!supported) {
      return {
        supported: false,
        hit: false,
        invalidated: false,
        reason: 'unsupported',
        key,
        data: null,
        meta: null,
      };
    }

    if (!cacheDir) {
      this._misses += 1;
      return {
        supported: true,
        hit: false,
        invalidated: false,
        reason: 'not-found',
        key,
        data: null,
        meta: null,
      };
    }

    const expectedEtag = normalizeOptionalTag(
      opts.etag !== undefined ? opts.etag : opts.expectedEtag
    );
    const expectedVersionKey = normalizeOptionalTag(
      opts.versionKey !== undefined ? opts.versionKey : opts.expectedVersionKey
    );

    const existing = this._entries.get(key) || null;

    if (existing) {
      if (expectedEtag !== null && existing.etag !== expectedEtag) {
        await this._removeFile(cacheDir, key);
        this._deleteNode(existing);
        this._invalidations += 1;
        this._misses += 1;
        await this._markIndexDirty();
        return {
          supported: true,
          hit: false,
          invalidated: true,
          reason: 'etag-mismatch',
          key,
          data: null,
          meta: null,
        };
      }
      if (expectedVersionKey !== null && existing.versionKey !== expectedVersionKey) {
        await this._removeFile(cacheDir, key);
        this._deleteNode(existing);
        this._invalidations += 1;
        this._misses += 1;
        await this._markIndexDirty();
        return {
          supported: true,
          hit: false,
          invalidated: true,
          reason: 'version-mismatch',
          key,
          data: null,
          meta: null,
        };
      }
    }

    const bytes = await this._readFile(cacheDir, key);
    if (bytes === null) {
      if (existing) {
        this._deleteNode(existing);
        await this._markIndexDirty();
      }
      this._misses += 1;
      return {
        supported: true,
        hit: false,
        invalidated: false,
        reason: 'not-found',
        key,
        data: null,
        meta: null,
      };
    }

    if (!existing && (expectedEtag !== null || expectedVersionKey !== null)) {
      await this._removeFile(cacheDir, key);
      this._invalidations += 1;
      this._misses += 1;
      await this._markIndexDirty();
      return {
        supported: true,
        hit: false,
        invalidated: true,
        reason: expectedEtag !== null ? 'etag-mismatch' : 'version-mismatch',
        key,
        data: null,
        meta: null,
      };
    }

    if (
      typeof opts.expectedSizeBytes === 'number' &&
      Number.isFinite(opts.expectedSizeBytes) &&
      bytes.byteLength !== Math.floor(opts.expectedSizeBytes)
    ) {
      await this._removeFile(cacheDir, key);
      if (existing) {
        this._deleteNode(existing);
      }
      this._invalidations += 1;
      this._misses += 1;
      await this._markIndexDirty();
      return {
        supported: true,
        hit: false,
        invalidated: true,
        reason: 'size-mismatch',
        key,
        data: null,
        meta: null,
      };
    }

    const nowMs = this._readNowMs();
    let node = existing;
    if (!node) {
      node = this._upsertNode({
        key,
        sizeBytes: bytes.byteLength,
        lastAccessedMs: nowMs,
        createdAtMs: nowMs,
        etag: null,
        versionKey: null,
        pinned: false,
        kind: inferChunkKind(key),
        accessOrder: this._nextAccessOrder(),
      });
    } else {
      if (node.sizeBytes !== bytes.byteLength) {
        this._totalBytes = Math.max(0, this._totalBytes - node.sizeBytes) + bytes.byteLength;
        if (node.pinned) {
          this._pinnedBytes = Math.max(0, this._pinnedBytes - node.sizeBytes) + bytes.byteLength;
        }
        node.sizeBytes = bytes.byteLength;
      }
      if (opts.touch !== false) {
        this._touchNode(node, nowMs);
      }
    }

    this._hits += 1;
    if (opts.touch !== false || !existing) {
      await this._markIndexDirty();
    }

    return {
      supported: true,
      hit: true,
      invalidated: false,
      reason: null,
      key,
      data: bytes,
      meta: cloneNodeMeta(node),
    };
  }

  /**
   * Reads a cached chunk's bytes (`Uint8Array`) or `null` on cache miss.
   * Performs O(1) LRU touch on hit unless `options.touch === false`.
   *
   * @param {string|object} keyOrDescriptor
   * @param {object} [options]
   * @returns {Promise<Uint8Array|null>}
   */
  async get(keyOrDescriptor, options = {}) {
    const result = await this.getEntry(keyOrDescriptor, options);
    return result.hit ? result.data : null;
  }

  /**
   * Deletes a cached chunk from OPFS and the O(1) LRU index.
   *
   * @param {string|object} keyOrDescriptor
   * @returns {Promise<boolean>}
   */
  async delete(keyOrDescriptor) {
    const key = makeChunkKey(keyOrDescriptor, this.indexFileName);
    if (!this._initialized) {
      await this.init();
    }

    const { supported, dir: cacheDir } = await this._ensureCacheDir({ create: false });
    const node = this._entries.get(key) || null;
    let removedFile = false;
    if (supported && cacheDir) {
      removedFile = await this._removeFile(cacheDir, key);
    }
    const removedMeta = node ? this._deleteNode(node) : false;

    if (removedFile || removedMeta) {
      this._deletes += 1;
      await this._markIndexDirty();
      return true;
    }
    return false;
  }

  /**
   * Evicts unpinned chunks in O(1)-per-victim LRU order (from `_head` toward `_tail`)
   * until `totalBytes <= targetBytes`.
   *
   * @param {number} targetBytes
   * @param {object} [options]
   * @param {Iterable<string>} [options.excludeKeys]
   * @param {boolean} [options.deferFlush=false]
   */
  async evictToBudget(targetBytes, options = {}) {
    const budget = sanitizeNonNegativeInt(targetBytes, 0);
    const opts = options && typeof options === 'object' ? options : {};
    const excludeSet = new Set(opts.excludeKeys || []);

    if (!this._initialized) {
      await this.init();
    }

    const { supported, dir: cacheDir } = await this._ensureCacheDir({ create: false });
    if (!supported || !cacheDir || this._totalBytes <= budget) {
      return {
        evictedEntries: 0,
        evictedBytes: 0,
        remainingEntries: this._entries.size,
        remainingBytes: this._totalBytes,
      };
    }

    let evictedEntries = 0;
    let evictedBytes = 0;
    let node = this._head;

    while (node !== null && this._totalBytes > budget) {
      const next = node.next;
      if (node.pinned || excludeSet.has(node.key)) {
        node = next;
        continue;
      }
      const key = node.key;
      const sizeBytes = node.sizeBytes;
      await this._removeFile(cacheDir, key);
      this._deleteNode(node);
      evictedEntries += 1;
      evictedBytes += sizeBytes;
      this._evictions += 1;
      this._evictedBytes += sizeBytes;
      node = next;
    }

    if (evictedEntries > 0 && !opts.deferFlush) {
      await this._markIndexDirty();
    }

    return {
      evictedEntries,
      evictedBytes,
      remainingEntries: this._entries.size,
      remainingBytes: this._totalBytes,
    };
  }

  /**
   * Clears all cached chunks and the persisted index file from OPFS.
   */
  async clear() {
    this._cancelFlushTimer();
    const { supported, dir: cacheDir } = await this._ensureCacheDir({ create: false });
    if (!supported) {
      return { supported: false, removedCount: 0, freedBytes: 0 };
    }

    if (!cacheDir) {
      this._clearList();
      this._initialized = true;
      this._indexDirty = false;
      return { supported: true, removedCount: 0, freedBytes: 0 };
    }

    let removedCount = 0;
    let freedBytes = 0;
    const topLevel = [];
    let hasIndexFile = false;

    for await (const [name, handle] of iterateDirEntries(cacheDir)) {
      if (!handle || typeof name !== 'string') continue;
      if (name === this.indexFileName && handle.kind !== 'directory') {
        hasIndexFile = true;
        continue;
      }
      if (handle.kind === 'directory') {
        const nested = await this._collectDiskEntries(handle);
        let nestedBytes = 0;
        for (let i = 0; i < nested.length; i += 1) {
          nestedBytes += nested[i].size;
        }
        topLevel.push({ name, isDir: true, count: nested.length, bytes: nestedBytes });
      } else if (handle.kind === 'file' || typeof handle.getFile === 'function') {
        const meta = await readFileHandleMeta(handle);
        topLevel.push({ name, isDir: false, count: 1, bytes: meta.size });
      }
    }

    for (let i = 0; i < topLevel.length; i += 1) {
      const item = topLevel[i];
      if (typeof cacheDir.removeEntry === 'function') {
        if (item.isDir) {
          await cacheDir.removeEntry(item.name, { recursive: true });
        } else {
          await cacheDir.removeEntry(item.name);
        }
      }
      removedCount += item.count;
      freedBytes += item.bytes;
    }

    if (hasIndexFile && typeof cacheDir.removeEntry === 'function') {
      try {
        await cacheDir.removeEntry(this.indexFileName);
      } catch {
        // Ignore missing index file.
      }
    }

    this._clearList();
    this._initialized = true;
    this._indexDirty = false;
    return { supported: true, removedCount, freedBytes };
  }

  // --- Bounded Parallel Warm-Boot Prefetch / Hydration ---

  /**
   * Hydrates a set of chunk descriptors into OPFS with bounded concurrency,
   * skipping already-cached chunks (with O(1) LRU touch) and deduplicating
   * concurrent in-flight fetches for identical keys.
   *
   * @param {Iterable<string|object>} descriptors
   * @param {(descriptor: string|object, ctx: {key: string, signal: AbortSignal|null}) => Promise<Uint8Array|ArrayBufferView|ArrayBuffer|string|{data: any, etag?: string, versionKey?: string, pinned?: boolean, kind?: string}|null>} fetcher
   * @param {object} [options]
   * @param {number} [options.concurrency=4] Maximum concurrent `fetcher` invocations.
   * @param {AbortSignal|null} [options.signal=null] Optional cancellation signal.
   * @param {boolean} [options.flushOnComplete=false] Force an index flush when the batch finishes.
   */
  async prefetchChunks(descriptors, fetcher, options = {}) {
    if (typeof fetcher !== 'function') {
      throw new TypeError('prefetchChunks requires a fetcher function');
    }

    const opts = options && typeof options === 'object' ? options : {};
    const concurrency = sanitizePositiveInt(opts.concurrency, DEFAULT_PREFETCH_CONCURRENCY);
    const signal = opts.signal || null;
    const flushOnComplete = Boolean(opts.flushOnComplete);

    const list = descriptors ? Array.from(descriptors) : [];
    if (list.length === 0) {
      return {
        requested: 0,
        hits: 0,
        fetched: 0,
        deduplicatedInFlight: 0,
        failed: 0,
        aborted: Boolean(signal && signal.aborted),
        bytesFetched: 0,
        results: [],
      };
    }

    if (!this._initialized) {
      await this.init();
    }

    this._prefetchRequested += list.length;

    let nextIndex = 0;
    let hits = 0;
    let fetched = 0;
    let deduplicatedInFlight = 0;
    let failed = 0;
    let bytesFetched = 0;
    let aborted = false;
    const results = new Array(list.length);

    const runOne = async (idx) => {
      const desc = list[idx];
      let key;
      try {
        key = makeChunkKey(desc, this.indexFileName);
      } catch (err) {
        failed += 1;
        this._prefetchFailed += 1;
        results[idx] = { key: null, status: 'failed', bytes: 0, error: sanitizeErrorCode(err) };
        return;
      }

      const expectedEtag =
        desc && typeof desc === 'object' ? normalizeOptionalTag(desc.etag) : null;
      const expectedVersionKey =
        desc && typeof desc === 'object' ? normalizeOptionalTag(desc.versionKey) : null;

      // 1. Check already-cached O(1) index entry (skip fetch on warm hit).
      const existing = this._entries.get(key);
      if (existing) {
        const etagOk = expectedEtag === null || existing.etag === expectedEtag;
        const versionOk = expectedVersionKey === null || existing.versionKey === expectedVersionKey;
        if (etagOk && versionOk) {
          this._touchNode(existing, this._readNowMs());
          this._hits += 1;
          this._prefetchHits += 1;
          hits += 1;
          await this._markIndexDirty();
          results[idx] = { key, status: 'hit', bytes: existing.sizeBytes, error: null };
          return;
        }
      }

      // 2. Check concurrent in-flight fetches for the exact same key.
      const existingInFlight = this._inFlightPrefetches.get(key);
      if (existingInFlight) {
        this._prefetchInFlightDeduped += 1;
        deduplicatedInFlight += 1;
        const shared = await existingInFlight;
        if (!shared.ok) {
          failed += 1;
          results[idx] = { key, status: 'failed', bytes: 0, error: shared.error };
        } else {
          results[idx] = { key, status: 'deduped', bytes: shared.bytes, error: null };
        }
        return;
      }

      // 3. Register in-flight promise synchronously BEFORE awaiting fetcher.
      const fetchAndStore = (async () => {
        this._misses += 1;
        try {
          const response = await fetcher(desc, { key, signal });
          if (response === null || response === undefined) {
            this._prefetchFailed += 1;
            return { ok: false, bytes: 0, error: 'EmptyFetchResult' };
          }

          let rawBytes = response;
          let putOpts = {};
          if (
            typeof response === 'object' &&
            !(response instanceof Uint8Array) &&
            !(response instanceof ArrayBuffer) &&
            !ArrayBuffer.isView(response) &&
            response.data !== undefined
          ) {
            rawBytes = response.data;
            putOpts = {
              etag: response.etag !== undefined ? response.etag : expectedEtag,
              versionKey:
                response.versionKey !== undefined ? response.versionKey : expectedVersionKey,
              pinned:
                response.pinned !== undefined
                  ? response.pinned
                  : desc && typeof desc === 'object'
                    ? desc.pinned
                    : undefined,
              kind:
                response.kind !== undefined
                  ? response.kind
                  : desc && typeof desc === 'object'
                    ? desc.kind
                    : undefined,
            };
          } else {
            putOpts = {
              etag: expectedEtag,
              versionKey: expectedVersionKey,
              pinned: desc && typeof desc === 'object' ? desc.pinned : undefined,
              kind: desc && typeof desc === 'object' ? desc.kind : undefined,
            };
          }

          const bytes = toUint8Array(rawBytes);
          const putRes = await this.put(key, bytes, putOpts);
          if (!putRes.stored) {
            this._prefetchFailed += 1;
            return { ok: false, bytes: 0, error: 'StorageUnavailable' };
          }
          this._prefetchFetched += 1;
          this._prefetchBytes += bytes.byteLength;
          return { ok: true, bytes: bytes.byteLength, error: null };
        } catch (err) {
          this._prefetchFailed += 1;
          return { ok: false, bytes: 0, error: sanitizeErrorCode(err) };
        } finally {
          this._inFlightPrefetches.delete(key);
        }
      })();

      this._inFlightPrefetches.set(key, fetchAndStore);
      const outcome = await fetchAndStore;
      if (outcome.ok) {
        fetched += 1;
        bytesFetched += outcome.bytes;
        results[idx] = { key, status: 'fetched', bytes: outcome.bytes, error: null };
      } else {
        failed += 1;
        results[idx] = { key, status: 'failed', bytes: 0, error: outcome.error };
      }
    };

    const workerCount = Math.min(concurrency, list.length);
    const workers = [];
    for (let w = 0; w < workerCount; w += 1) {
      workers.push(
        (async () => {
          while (true) {
            if (signal && signal.aborted) {
              aborted = true;
              return;
            }
            const idx = nextIndex;
            if (idx >= list.length) {
              return;
            }
            nextIndex += 1;
            await runOne(idx);
          }
        })()
      );
    }

    await Promise.all(workers);

    if (flushOnComplete) {
      await this.flushIndex();
    }

    return {
      requested: list.length,
      hits,
      fetched,
      deduplicatedInFlight,
      failed,
      aborted,
      bytesFetched,
      results: results.filter((r) => r !== undefined),
    };
  }

  // --- Sanitized Warm-Boot Telemetry & Lifecycle ---

  /**
   * Returns a sanitized snapshot of warm-boot cache telemetry (numeric/boolean metrics only;
   * no file paths, keys, or game bytes).
   */
  getStats() {
    const totalLookups = this._hits + this._misses;
    const hitRate = totalLookups > 0 ? this._hits / totalLookups : 0;
    const utilizationRatio =
      this.maxBytes !== null && this.maxBytes > 0
        ? Math.min(1, this._totalBytes / this.maxBytes)
        : 0;

    return {
      initialized: this._initialized,
      supported: this._supported,
      entryCount: this._entries.size,
      totalBytes: this._totalBytes,
      maxBytes: this.maxBytes,
      pinnedCount: this._pinnedCount,
      pinnedBytes: this._pinnedBytes,
      utilizationRatio,
      hits: this._hits,
      misses: this._misses,
      hitRate,
      puts: this._puts,
      deletes: this._deletes,
      evictions: this._evictions,
      evictedBytes: this._evictedBytes,
      invalidations: this._invalidations,
      indexFlushes: this._indexFlushes,
      coalescedFlushes: this._coalescedFlushes,
      flushErrors: this._flushErrors,
      indexDirty: this._indexDirty,
      indexFlushDelayMs: this.indexFlushDelayMs,
      prefetchRequested: this._prefetchRequested,
      prefetchHits: this._prefetchHits,
      prefetchFetched: this._prefetchFetched,
      prefetchInFlightDeduped: this._prefetchInFlightDeduped,
      prefetchFailed: this._prefetchFailed,
      prefetchBytes: this._prefetchBytes,
      inFlightCount: this._inFlightPrefetches.size,
    };
  }

  /**
   * Resets telemetry counters while keeping live cache entries and dirty state intact.
   */
  resetStats() {
    this._hits = 0;
    this._misses = 0;
    this._puts = 0;
    this._deletes = 0;
    this._evictions = 0;
    this._evictedBytes = 0;
    this._invalidations = 0;
    this._indexFlushes = 0;
    this._coalescedFlushes = 0;
    this._flushErrors = 0;
    this._prefetchRequested = 0;
    this._prefetchHits = 0;
    this._prefetchFetched = 0;
    this._prefetchInFlightDeduped = 0;
    this._prefetchFailed = 0;
    this._prefetchBytes = 0;
    return this.getStats();
  }

  /**
   * Flushes any pending batched index metadata and clears pending timers.
   */
  async close() {
    this._cancelFlushTimer();
    return this.flushIndex();
  }
}

export { OpfsChunkCache as OpfsCache };
export default OpfsChunkCache;
