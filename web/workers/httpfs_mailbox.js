'use strict';

// SharedArrayBuffer HTTPFS mailbox protocol matching runtime/os/httpfs_mailbox.h.
// Zero external dependencies; compatible with classic Web Workers (importScripts) and Node.
(function initPlayGta6HttpfsMailbox() {
const HTTPFS_MAILBOX_VERSION = 1;
const HTTPFS_HEADER_BYTES = 40;
const HTTPFS_PAYLOAD_BYTES = 64 * 1024; // 64 KiB fixed transfer buffer
const HTTPFS_MAILBOX_BYTES = HTTPFS_HEADER_BYTES + HTTPFS_PAYLOAD_BYTES;

// Mailbox state machine transitions for Atomics.wait / Atomics.notify:
//   Idle -> RequestReady (client) -> ResponseReady / Error (worker) -> Idle (client)
//   Any  -> Shutdown (teardown to wake blocked responder/client threads)
const HttpfsStatus = Object.freeze({
  Idle: 0,
  RequestReady: 1,
  ResponseReady: 2,
  Error: -1,
  Shutdown: -2,
  // Uppercase convenience aliases
  IDLE: 0,
  REQUEST_READY: 1,
  PENDING: 1,
  RESPONSE_READY: 2,
  DONE: 2,
  ERROR: -1,
  SHUTDOWN: -2,
});

const HttpfsOp = Object.freeze({
  None: 0,
  Open: 1,
  Read: 2,
  Close: 3,
  // Uppercase convenience aliases
  NONE: 0,
  OPEN: 1,
  READ: 2,
  CLOSE: 3,
});

// Exact byte offsets matching static_assert(offsetof(HttpfsMailbox, ...)) in runtime/os/httpfs_mailbox.h.
const HttpfsOffsets = Object.freeze({
  STATUS: 0,           // int32_t  (0..4)
  OP: 4,               // uint32_t (4..8)
  HANDLE: 8,           // int32_t  (8..12)
  PAYLOAD_CAPACITY: 12,// uint32_t (12..16)
  OFFSET: 16,          // uint64_t (16..24)
  LENGTH: 24,          // uint64_t (24..32)
  RESULT: 32,          // int64_t  (32..40)
  PAYLOAD: 40,         // uint8_t[payload_capacity]
});

const HTTPFS_DIRECTSTORAGE_CHUNK_BYTES = 64 * 1024; // 64 KiB DirectStorage streaming burst chunk
const HTTPFS_READAHEAD_BASE_BYTES = 256 * 1024;     // 256 KiB initial sequential window
const HTTPFS_READAHEAD_MID_BYTES = 1024 * 1024;     // 1 MiB sustained sequential burst window
const HTTPFS_READAHEAD_MAX_BYTES = 2 * 1024 * 1024; // 2 MiB default read-ahead upper cap
const HTTPFS_DEFAULT_PIPELINE_SLOTS = 4;            // Default multi-slot HTTPFS pipeline depth
const HTTPFS_MAX_PIPELINE_SLOTS = 16;               // Bounded upper cap for multi-slot pool
const MAX_RETAINED_CLOSED_TELEMETRY = 64;           // Bounded closed-handle telemetry history
const VIDEO_READAHEAD_EXT_RE = /\.(?:bk2|bik)$/i;

const HTTPFS_DEFAULT_CACHE_MAX_BYTES = 32 * 1024 * 1024; // 32 MiB default bounded LRU chunk cache
const HTTPFS_DEFAULT_CACHE_CHUNK_SIZE = 256 * 1024;      // 256 KiB default HTTP Range cache chunk
const HTTPFS_DEFAULT_READAHEAD_CHUNKS = 2;               // 2 chunks sequential read-ahead prefetch
const HTTPFS_MAX_READAHEAD_CHUNKS = 8;                   // Upper bound for read-ahead chunk count
const HTTPFS_MAX_TRACKED_URLS = 1024;                    // Bounded metadata maps per worker
const IMMUTABLE_GAME_RPF_RE = /^\/game\/.+\.rpf$/i;
const IMMUTABLE_SHADERS_RE = /^\/shaders\/.+$/i;

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder('utf-8', { fatal: false });

function defaultNowMs() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

function isVideoReadAheadPath(relPath) {
  return typeof relPath === 'string' && relPath.length > 0 && VIDEO_READAHEAD_EXT_RE.test(relPath);
}

function toNonNegFiniteNumber(value) {
  if (typeof value === 'bigint') {
    if (value < 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) {
      return NaN;
    }
    return Number(value);
  }
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : NaN;
  }
  return NaN;
}

function positiveIntOr(value, fallback) {
  if (value === undefined || value === null) {
    return fallback;
  }
  const n = Math.floor(Number(value));
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

/**
 * Compute adaptive read-ahead window size for sequential 64 KiB DirectStorage bursts
 * and 4K Bink2 (.bk2/.bik) streaming reads.
 *
 * Ramps cleanly across tiers: 256 KiB -> 1 MiB -> 2 MiB, capped by `maxReadAheadBytes`.
 * Returns 0 for non-sequential (sequentialCount <= 0), invalid request sizes, or maxReadAheadBytes <= 0.
 */
function computeReadAheadWindowSize(requestBytes, sequentialCount, options = {}) {
  const req = toNonNegFiniteNumber(requestBytes);
  if (!Number.isFinite(req) || req <= 0) {
    return 0;
  }
  const seq = Math.floor(toNonNegFiniteNumber(sequentialCount));
  if (!Number.isFinite(seq) || seq <= 0) {
    return 0;
  }

  const opts = options && typeof options === 'object' ? options : {};
  const maxCap =
    opts.maxReadAheadBytes !== undefined && opts.maxReadAheadBytes !== null
      ? Math.floor(Number(opts.maxReadAheadBytes))
      : HTTPFS_READAHEAD_MAX_BYTES;
  if (!Number.isFinite(maxCap) || maxCap <= 0) {
    return 0;
  }

  const isVideo =
    typeof opts.isVideoStream === 'boolean'
      ? opts.isVideoStream
      : isVideoReadAheadPath(opts.path);

  const baseBytes = positiveIntOr(
    opts.baseReadAheadBytes !== undefined ? opts.baseReadAheadBytes : opts.minReadAheadBytes,
    HTTPFS_READAHEAD_BASE_BYTES,
  );
  const midBytes = Math.max(
    baseBytes,
    positiveIntOr(opts.midReadAheadBytes, HTTPFS_READAHEAD_MID_BYTES),
  );
  const highBytes = Math.max(
    midBytes,
    positiveIntOr(opts.highReadAheadBytes, HTTPFS_READAHEAD_MAX_BYTES),
  );

  // 4K Bink2 (.bk2/.bik) streams ramp faster (seq 1 -> 256 KiB, 2 -> 1 MiB, 4 -> 2 MiB)
  // than general 64 KiB DirectStorage streaming bursts (seq 1 -> 256 KiB, 4 -> 1 MiB, 8 -> 2 MiB).
  const midThreshold = positiveIntOr(opts.rampMidThreshold, isVideo ? 2 : 4);
  const highThreshold = Math.max(
    midThreshold + 1,
    positiveIntOr(opts.rampHighThreshold, isVideo ? 4 : 8),
  );

  let targetBytes;
  if (seq >= highThreshold) {
    targetBytes = highBytes;
  } else if (seq >= midThreshold) {
    targetBytes = midBytes;
  } else {
    targetBytes = baseBytes;
  }

  const desired = Math.max(targetBytes, Math.ceil(req));
  return Math.min(maxCap, desired);
}

function normalizeReadAheadConfig(options = {}, defaultEnabled = false) {
  const opts = options && typeof options === 'object' ? options : {};
  const nested =
    opts.readAhead && typeof opts.readAhead === 'object' ? opts.readAhead : null;

  let enabled;
  if (typeof opts.readAhead === 'boolean') {
    enabled = opts.readAhead;
  } else if (typeof opts.readAheadEnabled === 'boolean') {
    enabled = opts.readAheadEnabled;
  } else if (nested && typeof nested.enabled === 'boolean') {
    enabled = nested.enabled;
  } else if (
    nested !== null ||
    opts.maxReadAheadBytes !== undefined ||
    opts.baseReadAheadBytes !== undefined ||
    opts.midReadAheadBytes !== undefined ||
    opts.highReadAheadBytes !== undefined
  ) {
    enabled = true;
  } else {
    enabled = Boolean(defaultEnabled);
  }

  const rawMax =
    nested && nested.maxReadAheadBytes !== undefined
      ? nested.maxReadAheadBytes
      : opts.maxReadAheadBytes;
  const maxReadAheadBytes =
    rawMax !== undefined && rawMax !== null
      ? Math.max(0, Math.floor(Number(rawMax)))
      : HTTPFS_READAHEAD_MAX_BYTES;
  if (!Number.isFinite(maxReadAheadBytes) || maxReadAheadBytes <= 0) {
    enabled = false;
  }

  const pickOpt = (key) => (nested && nested[key] !== undefined ? nested[key] : opts[key]);
  return {
    enabled,
    maxReadAheadBytes: Number.isFinite(maxReadAheadBytes) ? maxReadAheadBytes : 0,
    baseReadAheadBytes: pickOpt('baseReadAheadBytes'),
    minReadAheadBytes: pickOpt('minReadAheadBytes'),
    midReadAheadBytes: pickOpt('midReadAheadBytes'),
    highReadAheadBytes: pickOpt('highReadAheadBytes'),
    rampMidThreshold: pickOpt('rampMidThreshold'),
    rampHighThreshold: pickOpt('rampHighThreshold'),
  };
}

function createHandleTelemetryRecord(handle, isVideoStream = false) {
  return {
    handle,
    reads: 0,
    sequentialCount: 0,
    maxSequentialCount: 0,
    burstReads: 0,
    expectedNextOffset: -1,
    lastOffset: -1,
    totalWaitMs: 0,
    waitSamples: 0,
    maxWaitMs: 0,
    activePendingSlots: 0,
    peakPendingSlots: 0,
    currentWindowBytes: 0,
    readAheadBytesServed: 0,
    readAheadHits: 0,
    readAheadFetches: 0,
    coalescedSlotReads: 0,
    isVideoStream: Boolean(isVideoStream),
    closed: false,
    raBuffer: null,
    inFlightReadAhead: null,
  };
}

function snapshotHandleTelemetry(rec) {
  return {
    handle: rec.handle,
    reads: rec.reads,
    sequentialCount: rec.sequentialCount,
    maxSequentialCount: rec.maxSequentialCount,
    burstReads: rec.burstReads,
    avgWaitMs: rec.waitSamples > 0 ? rec.totalWaitMs / rec.waitSamples : 0,
    totalWaitMs: rec.totalWaitMs,
    maxWaitMs: rec.maxWaitMs,
    peakPendingSlots: rec.peakPendingSlots,
    currentWindowBytes: rec.currentWindowBytes,
    readAheadBytesServed: rec.readAheadBytesServed,
    readAheadHits: rec.readAheadHits,
    readAheadFetches: rec.readAheadFetches,
    coalescedSlotReads: rec.coalescedSlotReads,
    isVideoStream: rec.isVideoStream,
    closed: rec.closed,
  };
}

function extractUrlPathname(url) {
  if (typeof url !== 'string' || url.length === 0) {
    return '';
  }
  const trimmed = url.trim();
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(trimmed)) {
    try {
      return new URL(trimmed).pathname;
    } catch {
      return '';
    }
  }
  const withoutQuery = trimmed.split(/[?#]/, 1)[0];
  if (!withoutQuery) {
    return '';
  }
  return withoutQuery.startsWith('/') ? withoutQuery : `/${withoutQuery}`;
}

/**
 * Returns true if the URL points to an immutable `/game/*.rpf` archive or
 * `/shaders/*` shader asset eligible for bounded LRU chunk caching and
 * sequential read-ahead prefetching.
 */
function isCacheableImmutableUrl(url) {
  const pathname = extractUrlPathname(url);
  if (!pathname) {
    return false;
  }
  return IMMUTABLE_GAME_RPF_RE.test(pathname) || IMMUTABLE_SHADERS_RE.test(pathname);
}

function normalizeUrlBasePrefix(base, fallback) {
  const raw = typeof base === 'string' && base.length > 0 ? base : fallback;
  if (!raw) {
    return '/';
  }
  return raw.endsWith('/') ? raw : `${raw}/`;
}

function joinOriginAndPath(baseUrl, pathStr) {
  if (typeof baseUrl !== 'string' || baseUrl.length === 0) {
    return pathStr;
  }
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(pathStr)) {
    return pathStr;
  }
  const origin = baseUrl.replace(/\/+$/, '');
  const rel = pathStr.startsWith('/') ? pathStr : `/${pathStr}`;
  return `${origin}${rel}`;
}

function resolveHttpfsUrl(cleanPath, options = {}) {
  const opts = options && typeof options === 'object' ? options : {};
  if (typeof opts.urlResolver === 'function') {
    return opts.urlResolver(cleanPath);
  }
  const baseUrl = typeof opts.baseUrl === 'string' ? opts.baseUrl : '';
  const gameBase = joinOriginAndPath(baseUrl, normalizeUrlBasePrefix(opts.gameBase, '/game/'));
  const shadersBase = joinOriginAndPath(
    baseUrl,
    normalizeUrlBasePrefix(opts.shadersBase, '/shaders/'),
  );
  const dataBase = joinOriginAndPath(baseUrl, normalizeUrlBasePrefix(opts.dataBase, '/'));

  const lower = cleanPath.toLowerCase();
  if (lower.startsWith('game/')) {
    return `${gameBase}${cleanPath.slice(5)}`;
  }
  if (lower.startsWith('shaders/')) {
    return `${shadersBase}${cleanPath.slice(8)}`;
  }
  if (lower.endsWith('.rpf')) {
    return `${gameBase}${cleanPath}`;
  }
  return `${dataBase}${cleanPath}`;
}

function getHeaderCaseInsensitive(headers, name) {
  if (!headers) {
    return null;
  }
  const lower = name.toLowerCase();
  if (typeof headers.get === 'function') {
    const direct = headers.get(name);
    if (direct !== undefined && direct !== null) {
      return String(direct);
    }
    const lowerVal = headers.get(lower);
    if (lowerVal !== undefined && lowerVal !== null) {
      return String(lowerVal);
    }
    if (typeof headers.entries === 'function') {
      for (const [k, v] of headers.entries()) {
        if (String(k).toLowerCase() === lower && v !== undefined && v !== null) {
          return String(v);
        }
      }
    }
    return null;
  }
  if (typeof headers === 'object') {
    const keys = Object.keys(headers);
    for (let i = 0; i < keys.length; i += 1) {
      const k = keys[i];
      if (k.toLowerCase() === lower && headers[k] !== undefined && headers[k] !== null) {
        return String(headers[k]);
      }
    }
  }
  return null;
}

function parseContentRangeHeader(header) {
  if (typeof header !== 'string') {
    return null;
  }
  const match = /^bytes\s+(\d+)-(\d+)\/(\d+|\*)$/i.exec(header.trim());
  if (!match) {
    return null;
  }
  const start = Number(match[1]);
  const end = Number(match[2]);
  const total = match[3] === '*' ? null : Number(match[3]);
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || end < start) {
    return null;
  }
  if (total !== null && (!Number.isSafeInteger(total) || total < 0)) {
    return null;
  }
  return { start, end, total };
}

function parseUnsatisfiedRangeTotal(header) {
  if (typeof header !== 'string') {
    return null;
  }
  const match = /^bytes\s+\*\/(\d+)$/i.exec(header.trim());
  if (!match) {
    return null;
  }
  const total = Number(match[1]);
  return Number.isSafeInteger(total) && total >= 0 ? total : null;
}

function setBoundedMapEntry(map, key, value, maxEntries) {
  if (map.has(key)) {
    map.delete(key);
  } else if (map.size >= maxEntries) {
    const oldest = map.keys().next().value;
    if (oldest !== undefined) {
      map.delete(oldest);
    }
  }
  map.set(key, value);
}

function makeHttpfsChunkKey(url, chunkIndex) {
  return `${url}\0${chunkIndex}`;
}

function createEmptyCacheStats() {
  return {
    cacheHits: 0,
    cacheMisses: 0,
    cacheBytes: 0,
    cacheEvictions: 0,
    prefetchIssued: 0,
    prefetchHits: 0,
    coalescedReads: 0,
    bytesServedFromCache: 0,
    bytesFetchedFromNetwork: 0,
  };
}

/**
 * Bounded in-worker LRU chunk cache + sequential read-ahead prefetcher for
 * immutable `/game/*.rpf` and `/shaders/*` HTTP Range reads.
 *
 * - Safe opt-out when `cacheMaxBytes === 0` (direct uncached HTTP Range reads).
 * - Coalesces duplicate in-flight fetches for the same `(url, chunkIndex)`.
 * - Clamps cleanly at range boundaries and EOF (`206`, `200`, and `416`).
 * - Exposes sanitized telemetry via `getStats()` with zero paths or URLs.
 */
class HttpfsLruChunkCache {
  constructor(options = {}) {
    const opts = options && typeof options === 'object' ? options : {};

    const fetchFn =
      typeof opts.fetchImpl === 'function'
        ? opts.fetchImpl
        : typeof opts.fetch === 'function'
          ? opts.fetch
          : typeof fetch === 'function'
            ? fetch.bind(globalThis)
            : null;
    this._fetchImpl = fetchFn;

    if (opts.cacheMaxBytes === undefined || opts.cacheMaxBytes === null) {
      this._cacheMaxBytes = HTTPFS_DEFAULT_CACHE_MAX_BYTES;
    } else {
      const parsedMax = Math.floor(Number(opts.cacheMaxBytes));
      this._cacheMaxBytes = Number.isFinite(parsedMax) && parsedMax > 0 ? parsedMax : 0;
    }

    if (opts.chunkSize === undefined || opts.chunkSize === null) {
      this._chunkSize = HTTPFS_DEFAULT_CACHE_CHUNK_SIZE;
    } else {
      const parsedChunk = Math.floor(Number(opts.chunkSize));
      if (!Number.isFinite(parsedChunk) || parsedChunk <= 0) {
        throw new RangeError(`HttpfsLruChunkCache chunkSize must be a positive integer: ${opts.chunkSize}`);
      }
      this._chunkSize = parsedChunk;
    }

    if (opts.readAheadChunks === undefined || opts.readAheadChunks === null) {
      this._readAheadChunks = HTTPFS_DEFAULT_READAHEAD_CHUNKS;
    } else {
      const parsedRa = Math.floor(Number(opts.readAheadChunks));
      this._readAheadChunks =
        Number.isFinite(parsedRa) && parsedRa > 0
          ? Math.min(parsedRa, HTTPFS_MAX_READAHEAD_CHUNKS)
          : 0;
    }

    this._sequentialThreshold = positiveIntOr(opts.sequentialThreshold, 1);
    this._isCacheableFn =
      typeof opts.isCacheableUrl === 'function' ? opts.isCacheableUrl : isCacheableImmutableUrl;

    this._cache = new Map();
    this._cacheBytes = 0;
    this._inFlight = new Map();
    this._pendingPrefetches = new Set();
    this._fileSizes = new Map();
    this._seqState = new Map();

    this._stats = {
      cacheHits: 0,
      cacheMisses: 0,
      cacheEvictions: 0,
      prefetchIssued: 0,
      prefetchHits: 0,
      coalescedReads: 0,
      bytesServedFromCache: 0,
      bytesFetchedFromNetwork: 0,
    };
  }

  get cacheMaxBytes() {
    return this._cacheMaxBytes;
  }

  get chunkSize() {
    return this._chunkSize;
  }

  get readAheadChunks() {
    return this._readAheadChunks;
  }

  get cacheBytes() {
    return this._cacheBytes;
  }

  isCacheableUrl(url) {
    return Boolean(this._isCacheableFn(url));
  }

  getKnownSize(url) {
    return this._fileSizes.has(url) ? this._fileSizes.get(url) : null;
  }

  getStats() {
    return {
      enabled: this._cacheMaxBytes > 0,
      cacheMaxBytes: this._cacheMaxBytes,
      chunkSize: this._chunkSize,
      readAheadChunks: this._readAheadChunks,
      cacheEntries: this._cache.size,
      inFlightRequests: this._inFlight.size,
      cacheHits: this._stats.cacheHits,
      cacheMisses: this._stats.cacheMisses,
      cacheBytes: this._cacheBytes,
      cacheEvictions: this._stats.cacheEvictions,
      prefetchIssued: this._stats.prefetchIssued,
      prefetchHits: this._stats.prefetchHits,
      coalescedReads: this._stats.coalescedReads,
      bytesServedFromCache: this._stats.bytesServedFromCache,
      bytesFetchedFromNetwork: this._stats.bytesFetchedFromNetwork,
    };
  }

  clear(options = {}) {
    this._cache.clear();
    this._cacheBytes = 0;
    this._seqState.clear();
    if (options && options.clearFileSizes) {
      this._fileSizes.clear();
    }
    if (options && options.resetStats) {
      this._stats.cacheHits = 0;
      this._stats.cacheMisses = 0;
      this._stats.cacheEvictions = 0;
      this._stats.prefetchIssued = 0;
      this._stats.prefetchHits = 0;
      this._stats.coalescedReads = 0;
      this._stats.bytesServedFromCache = 0;
      this._stats.bytesFetchedFromNetwork = 0;
    }
  }

  async flushPrefetches() {
    while (this._pendingPrefetches.size > 0) {
      const batch = Array.from(this._pendingPrefetches);
      await Promise.allSettled(batch);
    }
  }

  _requireFetch() {
    const fn = this._fetchImpl || (typeof fetch === 'function' ? fetch.bind(globalThis) : null);
    if (typeof fn !== 'function') {
      throw new Error('HttpfsLruChunkCache requires a fetch implementation');
    }
    return fn;
  }

  _recordKnownSize(url, totalSize) {
    if (Number.isSafeInteger(totalSize) && totalSize >= 0) {
      setBoundedMapEntry(this._fileSizes, url, totalSize, HTTPFS_MAX_TRACKED_URLS);
    }
  }

  _touchEntry(key, entry) {
    this._cache.delete(key);
    this._cache.set(key, entry);
  }

  _insertEntry(key, url, chunkIndex, bytes, isPrefetch, consumedPrefetch) {
    if (this._cacheMaxBytes <= 0 || !(bytes instanceof Uint8Array) || bytes.byteLength === 0) {
      return;
    }
    if (bytes.byteLength > this._cacheMaxBytes) {
      return;
    }

    const existing = this._cache.get(key);
    if (existing) {
      this._cacheBytes -= existing.byteLength;
      this._cache.delete(key);
    }

    while (this._cacheBytes + bytes.byteLength > this._cacheMaxBytes && this._cache.size > 0) {
      const oldestKey = this._cache.keys().next().value;
      const oldest = this._cache.get(oldestKey);
      this._cache.delete(oldestKey);
      if (oldest) {
        this._cacheBytes -= oldest.byteLength;
      }
      this._stats.cacheEvictions += 1;
    }

    const stored = new Uint8Array(bytes.byteLength);
    stored.set(bytes);
    this._cache.set(key, {
      key,
      url,
      chunkIndex,
      bytes: stored,
      byteLength: stored.byteLength,
      prefetched: Boolean(isPrefetch),
      consumedPrefetch: Boolean(consumedPrefetch),
    });
    this._cacheBytes += stored.byteLength;
  }

  async _fetchRawRange(url, start, end) {
    const fetchFn = this._requireFetch();
    const res = await fetchFn(url, {
      method: 'GET',
      headers: {
        Range: `bytes=${start}-${end}`,
      },
    });

    if (res.status === 416) {
      const cr = getHeaderCaseInsensitive(res.headers, 'Content-Range');
      const total = parseUnsatisfiedRangeTotal(cr);
      if (total !== null) {
        this._recordKnownSize(url, total);
      }
      return { bytes: new Uint8Array(0), eof: true, status: 416 };
    }

    if (res.status === 206) {
      const cr = parseContentRangeHeader(getHeaderCaseInsensitive(res.headers, 'Content-Range'));
      if (cr && cr.total !== null) {
        this._recordKnownSize(url, cr.total);
      }
      const rawBuf = await res.arrayBuffer();
      const body = new Uint8Array(rawBuf);
      this._stats.bytesFetchedFromNetwork += body.byteLength;

      const knownTotal = this.getKnownSize(url);
      const reachedKnownEof = knownTotal !== null && start + body.byteLength >= knownTotal;
      const unknownTotalShort =
        knownTotal === null && body.byteLength < end - start + 1;
      return {
        bytes: body,
        eof: reachedKnownEof || unknownTotalShort,
        status: 206,
      };
    }

    if (res.status === 200) {
      const rawBuf = await res.arrayBuffer();
      const full = new Uint8Array(rawBuf);
      this._stats.bytesFetchedFromNetwork += full.byteLength;
      this._recordKnownSize(url, full.byteLength);
      if (start >= full.byteLength) {
        return { bytes: new Uint8Array(0), eof: true, status: 200 };
      }
      const sliceEnd = Math.min(end + 1, full.byteLength);
      return {
        bytes: full.slice(start, sliceEnd),
        eof: true,
        status: 200,
      };
    }

    const err = new Error(`HTTPFS range request failed (HTTP ${res.status})`);
    err.status = res.status;
    throw err;
  }

  async _fetchExactRange(url, start, length) {
    if (length <= 0) {
      return new Uint8Array(0);
    }
    const knownTotal = this.getKnownSize(url);
    if (knownTotal !== null && start >= knownTotal) {
      return new Uint8Array(0);
    }
    const targetLen =
      knownTotal !== null ? Math.min(length, knownTotal - start) : length;
    if (targetLen <= 0) {
      return new Uint8Array(0);
    }

    const parts = [];
    let collected = 0;
    while (collected < targetLen) {
      const curStart = start + collected;
      const curKnown = this.getKnownSize(url);
      if (curKnown !== null && curStart >= curKnown) {
        break;
      }
      const remaining =
        curKnown !== null
          ? Math.min(targetLen - collected, curKnown - curStart)
          : targetLen - collected;
      if (remaining <= 0) {
        break;
      }
      const curEnd = curStart + remaining - 1;
      const resp = await this._fetchRawRange(url, curStart, curEnd);
      if (resp.bytes.byteLength > 0) {
        const take =
          resp.bytes.byteLength > remaining
            ? resp.bytes.subarray(0, remaining)
            : resp.bytes;
        parts.push(take);
        collected += take.byteLength;
      }
      if (resp.eof || resp.bytes.byteLength === 0) {
        break;
      }
    }

    if (parts.length === 0) {
      return new Uint8Array(0);
    }
    if (parts.length === 1) {
      return parts[0];
    }
    const merged = new Uint8Array(collected);
    let cursor = 0;
    for (let i = 0; i < parts.length; i += 1) {
      merged.set(parts[i], cursor);
      cursor += parts[i].byteLength;
    }
    return merged;
  }

  async _fetchChunkNetwork(url, chunkIndex) {
    const chunkStart = chunkIndex * this._chunkSize;
    const chunkBytes = await this._fetchExactRange(url, chunkStart, this._chunkSize);
    if (chunkBytes.byteLength < this._chunkSize && !this._fileSizes.has(url)) {
      this._recordKnownSize(url, chunkStart + chunkBytes.byteLength);
    }
    return chunkBytes;
  }

  _startChunkFetch(url, chunkIndex, isPrefetch) {
    const key = makeHttpfsChunkKey(url, chunkIndex);
    const inFlightEntry = {
      promise: null,
      isPrefetch: Boolean(isPrefetch),
      prefetchHitCounted: false,
    };

    const promise = this._fetchChunkNetwork(url, chunkIndex).then(
      (chunkBytes) => {
        if (this._inFlight.get(key) === inFlightEntry) {
          this._inFlight.delete(key);
        }
        if (chunkBytes.byteLength > 0 && this._cacheMaxBytes > 0) {
          this._insertEntry(
            key,
            url,
            chunkIndex,
            chunkBytes,
            inFlightEntry.isPrefetch,
            inFlightEntry.prefetchHitCounted,
          );
        }
        return chunkBytes;
      },
      (err) => {
        if (this._inFlight.get(key) === inFlightEntry) {
          this._inFlight.delete(key);
        }
        throw err;
      },
    );

    inFlightEntry.promise = promise;
    this._inFlight.set(key, inFlightEntry);

    if (isPrefetch) {
      this._stats.prefetchIssued += 1;
      const tracked = promise.catch(() => {});
      this._pendingPrefetches.add(tracked);
      tracked.finally(() => {
        this._pendingPrefetches.delete(tracked);
      });
    } else {
      this._stats.cacheMisses += 1;
    }

    return promise;
  }

  _updateSequentialState(url, offset, length) {
    const startChunk = Math.floor(offset / this._chunkSize);
    const endChunk = Math.floor((offset + length - 1) / this._chunkSize);
    const prev = this._seqState.get(url) || null;

    let isSequential = false;
    if (prev !== null) {
      if (offset === prev.lastEndOffset) {
        isSequential = true;
      } else if (startChunk === prev.lastChunkIndex + 1 && offset > prev.lastOffset) {
        isSequential = true;
      }
    }

    const sequentialCount = isSequential ? prev.sequentialCount + 1 : 0;
    const nextState = {
      lastOffset: offset,
      lastLength: length,
      lastEndOffset: offset + length,
      lastChunkIndex: endChunk,
      sequentialCount,
    };
    setBoundedMapEntry(this._seqState, url, nextState, HTTPFS_MAX_TRACKED_URLS);
    return { isSequential, sequentialCount, endChunk };
  }

  _schedulePrefetches(url, endChunk, sequentialCount) {
    if (
      this._cacheMaxBytes <= 0 ||
      this._readAheadChunks <= 0 ||
      sequentialCount < this._sequentialThreshold
    ) {
      return;
    }
    const maxBudgetChunks = Math.floor(this._cacheMaxBytes / this._chunkSize);
    if (maxBudgetChunks < 2) {
      return;
    }
    const prefetchCount = Math.min(this._readAheadChunks, maxBudgetChunks - 1);
    const knownTotal = this.getKnownSize(url);

    for (let i = 1; i <= prefetchCount; i += 1) {
      const targetChunk = endChunk + i;
      const targetStart = targetChunk * this._chunkSize;
      if (knownTotal !== null && targetStart >= knownTotal) {
        break;
      }
      const key = makeHttpfsChunkKey(url, targetChunk);
      if (this._cache.has(key) || this._inFlight.has(key)) {
        continue;
      }
      this._startChunkFetch(url, targetChunk, true);
    }
  }

  async probe(url) {
    if (typeof url !== 'string' || url.length === 0) {
      throw new Error('HTTPFS probe requires a non-empty URL');
    }
    if (this._fileSizes.has(url)) {
      return { size: this._fileSizes.get(url) };
    }
    const key0 = makeHttpfsChunkKey(url, 0);
    if (this._cache.has(key0)) {
      return { size: this.getKnownSize(url) };
    }
    await this._fetchRawRange(url, 0, 0);
    return { size: this.getKnownSize(url) };
  }

  tryReadSync(url, offset, length) {
    const off = toSafeNonNegNumber(offset, 'offset');
    const len = toSafeNonNegNumber(length, 'length');
    if (len === 0) {
      return new Uint8Array(0);
    }
    if (this._cacheMaxBytes <= 0 || !this.isCacheableUrl(url)) {
      return null;
    }

    const knownTotal = this.getKnownSize(url);
    if (knownTotal !== null && off >= knownTotal) {
      return new Uint8Array(0);
    }

    const effectiveEnd = knownTotal !== null ? Math.min(off + len, knownTotal) : off + len;
    const effectiveLen = effectiveEnd - off;
    if (effectiveLen <= 0) {
      return new Uint8Array(0);
    }

    const firstChunk = Math.floor(off / this._chunkSize);
    const lastChunk = Math.floor((effectiveEnd - 1) / this._chunkSize);

    const entries = [];
    for (let idx = firstChunk; idx <= lastChunk; idx += 1) {
      const key = makeHttpfsChunkKey(url, idx);
      const entry = this._cache.get(key);
      if (!entry) {
        return null;
      }
      entries.push(entry);
    }

    const out = new Uint8Array(effectiveLen);
    let cursor = 0;
    for (let i = 0; i < entries.length; i += 1) {
      const entry = entries[i];
      this._touchEntry(entry.key, entry);
      this._stats.cacheHits += 1;
      if (entry.prefetched && !entry.consumedPrefetch) {
        entry.consumedPrefetch = true;
        this._stats.prefetchHits += 1;
      }

      const chunkStart = entry.chunkIndex * this._chunkSize;
      const sliceStart = Math.max(0, off - chunkStart);
      const remaining = effectiveLen - cursor;
      const available = Math.max(0, entry.byteLength - sliceStart);
      const copyLen = Math.min(remaining, available);
      if (copyLen > 0) {
        out.set(entry.bytes.subarray(sliceStart, sliceStart + copyLen), cursor);
        cursor += copyLen;
        this._stats.bytesServedFromCache += copyLen;
      }
      if (entry.byteLength < this._chunkSize) {
        break;
      }
    }

    const result = cursor === out.byteLength ? out : out.slice(0, cursor);
    if (result.byteLength > 0) {
      const seq = this._updateSequentialState(url, off, result.byteLength);
      if (seq.isSequential) {
        this._schedulePrefetches(url, seq.endChunk, seq.sequentialCount);
      }
    }
    return result;
  }

  async read(url, offset, length, options = {}) {
    const off = toSafeNonNegNumber(offset, 'offset');
    const len = toSafeNonNegNumber(length, 'length');
    if (len === 0) {
      return new Uint8Array(0);
    }

    const trackSequential = !(options && options.trackSequential === false);
    const knownTotalBefore = this.getKnownSize(url);
    if (knownTotalBefore !== null && off >= knownTotalBefore) {
      return new Uint8Array(0);
    }

    // Opt-out (cacheMaxBytes === 0) or non-immutable URL: direct HTTP Range read without caching.
    if (this._cacheMaxBytes <= 0 || !this.isCacheableUrl(url)) {
      this._stats.cacheMisses += 1;
      return this._fetchExactRange(url, off, len);
    }

    const effectiveEndBefore =
      knownTotalBefore !== null ? Math.min(off + len, knownTotalBefore) : off + len;
    const effectiveLenBefore = effectiveEndBefore - off;
    if (effectiveLenBefore <= 0) {
      return new Uint8Array(0);
    }

    const firstChunk = Math.floor(off / this._chunkSize);
    let lastChunk = Math.floor((effectiveEndBefore - 1) / this._chunkSize);

    const resolveChunk = (chunkIdx) => {
      const key = makeHttpfsChunkKey(url, chunkIdx);
      const cached = this._cache.get(key);
      if (cached) {
        this._touchEntry(key, cached);
        this._stats.cacheHits += 1;
        if (cached.prefetched && !cached.consumedPrefetch) {
          cached.consumedPrefetch = true;
          this._stats.prefetchHits += 1;
        }
        return { fromCache: true, bytes: cached.bytes, promise: null };
      }

      const existingInFlight = this._inFlight.get(key);
      if (existingInFlight) {
        this._stats.coalescedReads += 1;
        if (existingInFlight.isPrefetch && !existingInFlight.prefetchHitCounted) {
          existingInFlight.prefetchHitCounted = true;
          this._stats.prefetchHits += 1;
        }
        return { fromCache: false, bytes: null, promise: existingInFlight.promise };
      }

      const promise = this._startChunkFetch(url, chunkIdx, false);
      return { fromCache: false, bytes: null, promise };
    };

    let seqInfo = null;
    let prefetchScheduledEarly = false;
    if (trackSequential) {
      seqInfo = this._updateSequentialState(url, off, effectiveLenBefore);
      if (seqInfo.isSequential && knownTotalBefore !== null) {
        this._schedulePrefetches(url, seqInfo.endChunk, seqInfo.sequentialCount);
        prefetchScheduledEarly = true;
      }
    }

    const chunkResults = new Map();

    // If total file size is unknown and the read spans multiple chunks, fetch the first chunk
    // first so we learn Content-Range total / EOF and avoid issuing wasted past-EOF requests.
    if (knownTotalBefore === null && firstChunk < lastChunk) {
      const firstHandle = resolveChunk(firstChunk);
      const firstBytes = firstHandle.fromCache ? firstHandle.bytes : await firstHandle.promise;
      chunkResults.set(firstChunk, { bytes: firstBytes, fromCache: firstHandle.fromCache });

      if (firstBytes.byteLength < this._chunkSize) {
        lastChunk = firstChunk;
      } else {
        const knownAfterFirst = this.getKnownSize(url);
        if (knownAfterFirst !== null) {
          const clampedEnd = Math.min(off + len, knownAfterFirst);
          lastChunk =
            clampedEnd <= off ? firstChunk : Math.floor((clampedEnd - 1) / this._chunkSize);
        }
      }

      if (lastChunk > firstChunk) {
        const restPromises = [];
        for (let idx = firstChunk + 1; idx <= lastChunk; idx += 1) {
          const h = resolveChunk(idx);
          if (h.fromCache) {
            chunkResults.set(idx, { bytes: h.bytes, fromCache: true });
          } else {
            restPromises.push(
              h.promise.then((bytes) => {
                chunkResults.set(idx, { bytes, fromCache: false });
              }),
            );
          }
        }
        if (restPromises.length > 0) {
          await Promise.all(restPromises);
        }
      }
    } else {
      const pending = [];
      for (let idx = firstChunk; idx <= lastChunk; idx += 1) {
        const h = resolveChunk(idx);
        if (h.fromCache) {
          chunkResults.set(idx, { bytes: h.bytes, fromCache: true });
        } else {
          pending.push(
            h.promise.then((bytes) => {
              chunkResults.set(idx, { bytes, fromCache: false });
            }),
          );
        }
      }
      if (pending.length > 0) {
        await Promise.all(pending);
      }
    }

    const out = new Uint8Array(effectiveLenBefore);
    let cursor = 0;
    for (let idx = firstChunk; idx <= lastChunk; idx += 1) {
      const item = chunkResults.get(idx);
      if (!item || !item.bytes || item.bytes.byteLength === 0) {
        break;
      }
      const chunkStart = idx * this._chunkSize;
      const sliceStart = Math.max(0, off - chunkStart);
      const remaining = effectiveLenBefore - cursor;
      const available = Math.max(0, item.bytes.byteLength - sliceStart);
      const copyLen = Math.min(remaining, available);
      if (copyLen > 0) {
        out.set(item.bytes.subarray(sliceStart, sliceStart + copyLen), cursor);
        cursor += copyLen;
        if (item.fromCache) {
          this._stats.bytesServedFromCache += copyLen;
        }
      }
      if (item.bytes.byteLength < this._chunkSize) {
        break;
      }
    }

    const result = cursor === out.byteLength ? out : out.slice(0, cursor);

    if (trackSequential && seqInfo) {
      // Adjust recorded end offset on clamped EOF reads so subsequent reads stay accurate.
      if (result.byteLength < effectiveLenBefore) {
        const state = this._seqState.get(url);
        if (state && state.lastOffset === off) {
          state.lastLength = result.byteLength;
          state.lastEndOffset = off + result.byteLength;
          state.lastChunkIndex =
            result.byteLength > 0
              ? Math.floor((off + result.byteLength - 1) / this._chunkSize)
              : firstChunk;
        }
      }
      if (seqInfo.isSequential && !prefetchScheduledEarly && result.byteLength > 0) {
        const finalEndChunk = Math.floor((off + result.byteLength - 1) / this._chunkSize);
        this._schedulePrefetches(url, finalEndChunk, seqInfo.sequentialCount);
      }
    }

    return result;
  }
}

function hasHttpRangeCacheOptions(opts) {
  return Boolean(
    opts &&
      (typeof opts.fetchImpl === 'function' ||
        typeof opts.fetch === 'function' ||
        opts.chunkCache !== undefined ||
        opts.baseUrl !== undefined ||
        opts.gameBase !== undefined ||
        opts.shadersBase !== undefined ||
        opts.dataBase !== undefined ||
        typeof opts.urlResolver === 'function' ||
        opts.cacheMaxBytes !== undefined ||
        opts.chunkSize !== undefined ||
        opts.readAheadChunks !== undefined ||
        opts.enableHttpRange === true),
  );
}

function createSharedMailboxHostState(options = {}, defaultReadAheadEnabled = false, slotCount = 1) {
  const opts = options && typeof options === 'object' ? options : {};
  const backend = opts.backend || null;
  const hasExternalBackend = Boolean(
    opts.rangeCache ||
      typeof opts.getRangeCache === 'function' ||
      typeof opts.open === 'function' ||
      typeof opts.read === 'function' ||
      (backend && (typeof backend.open === 'function' || typeof backend.read === 'function')),
  );

  let chunkCache = null;
  if (opts.chunkCache && typeof opts.chunkCache.read === 'function') {
    chunkCache = opts.chunkCache;
  } else if (!hasExternalBackend && hasHttpRangeCacheOptions(opts)) {
    chunkCache = new HttpfsLruChunkCache({
      fetchImpl: opts.fetchImpl,
      fetch: opts.fetch,
      cacheMaxBytes: opts.cacheMaxBytes,
      chunkSize: opts.chunkSize,
      readAheadChunks: opts.readAheadChunks,
      sequentialThreshold: opts.sequentialThreshold,
      isCacheableUrl: opts.isCacheableUrl,
    });
  }

  return {
    slotCount: Math.max(1, Math.floor(Number(slotCount) || 1)),
    handles: new Map(),
    nextHandle: 1,
    handleRecords: new Map(),
    closedOrder: [],
    now: typeof opts.now === 'function' ? opts.now : defaultNowMs,
    readAhead: normalizeReadAheadConfig(opts, defaultReadAheadEnabled),
    chunkCache,
    urlConfig: {
      baseUrl: opts.baseUrl,
      gameBase: opts.gameBase,
      shadersBase: opts.shadersBase,
      dataBase: opts.dataBase,
      urlResolver: opts.urlResolver,
    },
    stats: {
      requests: 0,
      syncFastPathHits: 0,
      asyncPathHits: 0,
      burstReads: 0,
      totalWaitMs: 0,
      waitSamples: 0,
      maxWaitMs: 0,
      activePendingSlots: 0,
      peakPendingSlots: 0,
      readAheadBytesServed: 0,
      readAheadHits: 0,
      readAheadFetches: 0,
      coalescedSlotReads: 0,
    },
  };
}

function isSharedOrArrayBuffer(buf) {
  if (buf instanceof ArrayBuffer) return true;
  if (typeof SharedArrayBuffer !== 'undefined' && buf instanceof SharedArrayBuffer) return true;
  return false;
}

function assertValidMailboxBuffer(sab) {
  if (!isSharedOrArrayBuffer(sab)) {
    throw new TypeError('HttpfsMailbox requires a SharedArrayBuffer (or ArrayBuffer)');
  }
  if (sab.byteLength < HTTPFS_HEADER_BYTES) {
    throw new RangeError(
      `HttpfsMailbox buffer too small: ${sab.byteLength} < ${HTTPFS_HEADER_BYTES}`,
    );
  }
}

function normalizeRelPath(relPath) {
  if (typeof relPath !== 'string' || relPath.length === 0) {
    throw new Error('HTTPFS path must be a non-empty string');
  }
  if (relPath.includes('\\') || relPath.includes('\0')) {
    throw new Error(`Invalid HTTPFS path: ${relPath}`);
  }
  const stripped = relPath.replace(/^\/+/, '');
  if (!stripped) {
    throw new Error(`Invalid HTTPFS path: ${relPath}`);
  }
  const parts = stripped.split('/');
  for (const part of parts) {
    if (!part || part === '.' || part === '..') {
      throw new Error(`Invalid HTTPFS path segment in: ${relPath}`);
    }
  }
  return stripped;
}

function toSafeUint64BigInt(value, label) {
  if (typeof value === 'bigint') {
    if (value < 0n) {
      throw new RangeError(`${label} must be non-negative`);
    }
    return value;
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value) || value < 0 || !Number.isInteger(value)) {
      throw new RangeError(`${label} must be a non-negative integer`);
    }
    return BigInt(value);
  }
  throw new TypeError(`${label} must be a Number or BigInt`);
}

function toSafeNonNegNumber(value, label) {
  if (typeof value === 'bigint') {
    if (value < 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new RangeError(`${label} out of safe integer range`);
    }
    return Number(value);
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value) || value < 0 || !Number.isInteger(value)) {
      throw new RangeError(`${label} must be a non-negative integer`);
    }
    return value;
  }
  throw new TypeError(`${label} must be a Number or BigInt`);
}

function initHttpfsMailbox(sab, payloadCapacity) {
  assertValidMailboxBuffer(sab);
  const available = sab.byteLength - HTTPFS_HEADER_BYTES;
  const maxCap = Math.min(available, HTTPFS_PAYLOAD_BYTES);
  const requested =
    payloadCapacity === undefined || payloadCapacity === null || payloadCapacity === 0
      ? maxCap
      : Math.min(Math.floor(payloadCapacity), maxCap);

  if (!Number.isInteger(requested) || requested < 0) {
    throw new RangeError(`Invalid payloadCapacity: ${payloadCapacity}`);
  }

  const header = new DataView(sab, 0, HTTPFS_HEADER_BYTES);
  const statusView = new Int32Array(sab, 0, 1);

  header.setUint32(HttpfsOffsets.OP, HttpfsOp.None, true);
  header.setInt32(HttpfsOffsets.HANDLE, -1, true);
  header.setUint32(HttpfsOffsets.PAYLOAD_CAPACITY, requested, true);
  header.setBigUint64(HttpfsOffsets.OFFSET, 0n, true);
  header.setBigUint64(HttpfsOffsets.LENGTH, 0n, true);
  header.setBigInt64(HttpfsOffsets.RESULT, 0n, true);
  Atomics.store(statusView, 0, HttpfsStatus.Idle);

  return sab;
}

function createHttpfsMailboxBuffer(payloadCapacity = HTTPFS_PAYLOAD_BYTES) {
  if (typeof SharedArrayBuffer === 'undefined') {
    throw new Error('SharedArrayBuffer is unavailable in this environment');
  }
  const cap =
    payloadCapacity === undefined || payloadCapacity === null || payloadCapacity === 0
      ? HTTPFS_PAYLOAD_BYTES
      : Math.min(Math.floor(payloadCapacity), HTTPFS_PAYLOAD_BYTES);
  if (!Number.isInteger(cap) || cap <= 0) {
    throw new RangeError(`Invalid payloadCapacity: ${payloadCapacity}`);
  }
  const sab = new SharedArrayBuffer(HTTPFS_HEADER_BYTES + cap);
  return initHttpfsMailbox(sab, cap);
}

function effectivePayloadCapacity(headerView, payloadByteLength) {
  const configured = headerView.getUint32(HttpfsOffsets.PAYLOAD_CAPACITY, true);
  const limit = Math.min(payloadByteLength, HTTPFS_PAYLOAD_BYTES);
  if (configured === 0) {
    return limit;
  }
  return Math.min(configured, limit);
}

function decodeSabUtf8(payloadView, byteLength) {
  // TextDecoder.decode rejects views backed by SharedArrayBuffer; copy into a local ArrayBuffer first.
  const copy = new Uint8Array(byteLength);
  copy.set(payloadView.subarray(0, byteLength));
  return textDecoder.decode(copy);
}

function isThenable(value) {
  return (
    value !== null &&
    (typeof value === 'object' || typeof value === 'function') &&
    typeof value.then === 'function'
  );
}

function extractReadBytes(res) {
  if (res === null || res === undefined) {
    return undefined;
  }
  const raw =
    typeof res === 'object' &&
    !ArrayBuffer.isView(res) &&
    !(res instanceof ArrayBuffer) &&
    res.bytes !== undefined
      ? res.bytes
      : res;
  if (raw === null || raw === undefined) {
    return undefined;
  }
  if (raw instanceof Uint8Array) {
    return raw;
  }
  if (raw instanceof ArrayBuffer) {
    return new Uint8Array(raw);
  }
  if (ArrayBuffer.isView(raw)) {
    return new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength);
  }
  return null;
}

class HttpfsMailboxHost {
  constructor(sab, options = {}) {
    assertValidMailboxBuffer(sab);
    this.sab = sab;
    this._statusView = new Int32Array(sab, 0, 1);
    this._headerView = new DataView(sab, 0, HTTPFS_HEADER_BYTES);
    this._payloadView = new Uint8Array(
      sab,
      HTTPFS_HEADER_BYTES,
      sab.byteLength - HTTPFS_HEADER_BYTES,
    );

    const backend = options.backend || null;
    this._rangeCache = options.rangeCache || null;
    this._getRangeCache = typeof options.getRangeCache === 'function' ? options.getRangeCache : null;
    this._manifest = options.manifest || null;
    this._getManifest = typeof options.getManifest === 'function' ? options.getManifest : null;
    this._probeOnOpen = options.probeOnOpen !== false;
    this._forcePoll = Boolean(options.forcePoll);
    this._fallbackPollMs =
      Number.isInteger(options.fallbackPollMs) && options.fallbackPollMs > 0
        ? options.fallbackPollMs
        : 2;

    this._customOpen =
      typeof options.open === 'function'
        ? options.open
        : backend && typeof backend.open === 'function'
          ? backend.open.bind(backend)
          : null;
    this._customRead =
      typeof options.read === 'function'
        ? options.read
        : backend && typeof backend.read === 'function'
          ? backend.read.bind(backend)
          : null;
    this._customClose =
      typeof options.close === 'function'
        ? options.close
        : backend && typeof backend.close === 'function'
          ? backend.close.bind(backend)
          : null;

    this._shared =
      options._sharedState && typeof options._sharedState === 'object'
        ? options._sharedState
        : createSharedMailboxHostState(options, false);
    this._handles = this._shared.handles;
    this._stats = this._shared.stats;
    this._now = this._shared.now;
    this._slotRequests = 0;
    this._slotSyncFastPathHits = 0;
    this._slotAsyncPathHits = 0;
    this._pendingFinalize = null;
    this._running = false;
    this._loopPromise = null;
    this._pollTimer = null;
    this._pollResolve = null;
  }

  get payloadCapacity() {
    return effectivePayloadCapacity(this._headerView, this._payloadView.byteLength);
  }

  get isRunning() {
    return this._running;
  }

  get syncFastPathHits() {
    return this._stats.syncFastPathHits;
  }

  get chunkCache() {
    return this._shared.chunkCache || null;
  }

  flushPrefetches() {
    const cc = this._shared.chunkCache;
    if (cc && typeof cc.flushPrefetches === 'function') {
      return cc.flushPrefetches();
    }
    return Promise.resolve();
  }

  _getOrCreateHandleRecord(handle, isVideoStream = false) {
    let rec = this._shared.handleRecords.get(handle);
    if (!rec) {
      rec = createHandleTelemetryRecord(handle, isVideoStream);
      this._shared.handleRecords.set(handle, rec);
    }
    return rec;
  }

  _initHandleRecordOnOpen(handle, cleanPath) {
    const isVideo = isVideoReadAheadPath(cleanPath);
    const rec = createHandleTelemetryRecord(handle, isVideo);
    this._shared.handleRecords.set(handle, rec);
    const closedIdx = this._shared.closedOrder.indexOf(handle);
    if (closedIdx >= 0) {
      this._shared.closedOrder.splice(closedIdx, 1);
    }
    return rec;
  }

  _markHandleClosed(handle) {
    const rec = this._shared.handleRecords.get(handle);
    if (!rec) {
      return;
    }
    rec.raBuffer = null;
    rec.inFlightReadAhead = null;
    rec.closed = true;
    const closedIdx = this._shared.closedOrder.indexOf(handle);
    if (closedIdx >= 0) {
      this._shared.closedOrder.splice(closedIdx, 1);
    }
    this._shared.closedOrder.push(handle);
    while (this._shared.closedOrder.length > MAX_RETAINED_CLOSED_TELEMETRY) {
      const evicted = this._shared.closedOrder.shift();
      const evictedRec = this._shared.handleRecords.get(evicted);
      if (evictedRec && evictedRec.closed) {
        this._shared.handleRecords.delete(evicted);
      }
    }
  }

  getHandleTelemetry(handle) {
    const rec = this._shared.handleRecords.get(handle);
    return rec ? snapshotHandleTelemetry(rec) : null;
  }

  getStats() {
    const handles = {};
    for (const [handle, rec] of this._shared.handleRecords.entries()) {
      handles[String(handle)] = snapshotHandleTelemetry(rec);
    }
    const cc = this._shared.chunkCache;
    const cacheStats =
      cc && typeof cc.getStats === 'function' ? cc.getStats() : createEmptyCacheStats();
    return {
      requests: this._stats.requests,
      syncFastPathHits: this._stats.syncFastPathHits,
      asyncPathHits: this._stats.asyncPathHits,
      burstReads: this._stats.burstReads,
      avgWaitMs:
        this._stats.waitSamples > 0 ? this._stats.totalWaitMs / this._stats.waitSamples : 0,
      totalWaitMs: this._stats.totalWaitMs,
      maxWaitMs: this._stats.maxWaitMs,
      activePendingSlots: this._stats.activePendingSlots,
      peakPendingSlots: this._stats.peakPendingSlots,
      readAheadEnabled: this._shared.readAhead.enabled,
      maxReadAheadBytes: this._shared.readAhead.maxReadAheadBytes,
      readAheadBytesServed: this._stats.readAheadBytesServed,
      readAheadHits: this._stats.readAheadHits,
      readAheadFetches: this._stats.readAheadFetches,
      coalescedSlotReads: this._stats.coalescedSlotReads,
      cacheHits: cacheStats.cacheHits,
      cacheMisses: cacheStats.cacheMisses,
      cacheBytes: cacheStats.cacheBytes,
      cacheEvictions: cacheStats.cacheEvictions,
      prefetchIssued: cacheStats.prefetchIssued,
      prefetchHits: cacheStats.prefetchHits,
      coalescedReads: cacheStats.coalescedReads,
      bytesServedFromCache: cacheStats.bytesServedFromCache,
      bytesFetchedFromNetwork: cacheStats.bytesFetchedFromNetwork,
      handles,
    };
  }

  _resolveRangeCache() {
    if (this._getRangeCache) {
      return this._getRangeCache();
    }
    return this._rangeCache;
  }

  _resolveManifest() {
    if (this._getManifest) {
      return this._getManifest();
    }
    return this._manifest;
  }

  _manifestHasFile(cleanPath) {
    const manifest = this._resolveManifest();
    if (!manifest || !Array.isArray(manifest.files)) {
      return null;
    }
    for (const entry of manifest.files) {
      if (!entry || typeof entry.path !== 'string') continue;
      try {
        if (normalizeRelPath(entry.path) === cleanPath) {
          return true;
        }
      } catch {
        // Ignore malformed manifest entries.
      }
    }
    return false;
  }

  completeResponse(result, status = HttpfsStatus.ResponseReady) {
    if (this._pendingFinalize) {
      const fin = this._pendingFinalize;
      this._pendingFinalize = null;
      fin();
    }
    this._headerView.setBigInt64(HttpfsOffsets.RESULT, BigInt(result), true);
    Atomics.store(this._statusView, 0, status);
    Atomics.notify(this._statusView, 0, 1);
  }

  _finishOpenResult(handle, cleanPath) {
    if (typeof handle === 'number' && Number.isInteger(handle) && handle >= 0) {
      this._initHandleRecordOnOpen(handle, cleanPath);
      this.completeResponse(BigInt(handle), HttpfsStatus.ResponseReady);
    } else {
      const errCode = typeof handle === 'number' && handle < 0 ? Math.trunc(handle) : -2;
      this.completeResponse(BigInt(errCode), HttpfsStatus.Error);
    }
  }

  _finishReadResult(res, reqLen) {
    const bytes = extractReadBytes(res);
    if (bytes === undefined) {
      this.completeResponse(-9n, HttpfsStatus.Error);
      return;
    }
    if (bytes === null) {
      this.completeResponse(-5n, HttpfsStatus.Error);
      return;
    }
    const toCopy = Math.min(bytes.byteLength, reqLen);
    if (toCopy > 0) {
      this._payloadView.set(bytes.subarray(0, toCopy), 0);
    }
    this.completeResponse(BigInt(toCopy), HttpfsStatus.ResponseReady);
  }

  _finishCloseResult(res, handle) {
    if (res === false || (typeof res === 'number' && res < 0)) {
      const errCode = typeof res === 'number' && res < 0 ? Math.trunc(res) : -9;
      this.completeResponse(BigInt(errCode), HttpfsStatus.Error);
    } else {
      if (typeof handle === 'number' && handle >= 0) {
        this._markHandleClosed(handle);
      }
      this.completeResponse(0n, HttpfsStatus.ResponseReady);
    }
  }

  _tryServeFromReadAheadBuffer(rec, offset, reqLen) {
    const buf = rec && rec.raBuffer;
    if (!buf || !(buf.bytes instanceof Uint8Array)) {
      return null;
    }
    const rel = offset - buf.startOffset;
    if (rel < 0 || rel >= buf.bytes.byteLength) {
      return null;
    }
    const avail = buf.bytes.byteLength - rel;
    if (avail >= reqLen) {
      return buf.bytes.subarray(rel, rel + reqLen);
    }
    if (buf.eof && avail > 0) {
      return buf.bytes.subarray(rel, rel + avail);
    }
    return null;
  }

  _handleOpen(lengthBig, cap) {
    if (lengthBig === 0n || lengthBig > BigInt(cap)) {
      this.completeResponse(-22n, HttpfsStatus.Error);
      return;
    }

    const pathLen = Number(lengthBig);
    const rawPath = decodeSabUtf8(this._payloadView, pathLen);
    let cleanPath;
    try {
      cleanPath = normalizeRelPath(rawPath);
    } catch {
      this.completeResponse(-22n, HttpfsStatus.Error);
      return;
    }

    if (this._customOpen) {
      const handle = this._customOpen(cleanPath);
      if (isThenable(handle)) {
        return handle.then((resolved) => {
          this._finishOpenResult(resolved, cleanPath);
        });
      }
      this._finishOpenResult(handle, cleanPath);
      return;
    }

    const inManifest = this._manifestHasFile(cleanPath);
    if (inManifest === false) {
      this.completeResponse(-2n, HttpfsStatus.Error);
      return;
    }

    const cache = this._resolveRangeCache();
    const chunkCache = !cache ? this._shared.chunkCache : null;
    if ((!cache || typeof cache.read !== 'function') && !chunkCache) {
      this.completeResponse(-19n, HttpfsStatus.Error);
      return;
    }

    if (chunkCache) {
      const url = resolveHttpfsUrl(cleanPath, this._shared.urlConfig);
      if (inManifest === null && this._probeOnOpen) {
        const probe = chunkCache.probe(url);
        if (isThenable(probe)) {
          return probe.then(
            () => {
              const handle = this._shared.nextHandle;
              this._shared.nextHandle += 1;
              this._handles.set(handle, { path: cleanPath, url });
              this._initHandleRecordOnOpen(handle, cleanPath);
              this.completeResponse(BigInt(handle), HttpfsStatus.ResponseReady);
            },
            () => {
              this.completeResponse(-2n, HttpfsStatus.Error);
            },
          );
        }
      }

      const handle = this._shared.nextHandle;
      this._shared.nextHandle += 1;
      this._handles.set(handle, { path: cleanPath, url });
      this._initHandleRecordOnOpen(handle, cleanPath);
      this.completeResponse(BigInt(handle), HttpfsStatus.ResponseReady);
      return;
    }

    if (inManifest === null && this._probeOnOpen) {
      const probe = cache.read(cleanPath, 0, 1);
      if (isThenable(probe)) {
        return probe.then(
          () => {
            const handle = this._shared.nextHandle;
            this._shared.nextHandle += 1;
            this._handles.set(handle, { path: cleanPath });
            this._initHandleRecordOnOpen(handle, cleanPath);
            this.completeResponse(BigInt(handle), HttpfsStatus.ResponseReady);
          },
          () => {
            this.completeResponse(-2n, HttpfsStatus.Error);
          },
        );
      }
    }

    const handle = this._shared.nextHandle;
    this._shared.nextHandle += 1;
    this._handles.set(handle, { path: cleanPath });
    this._initHandleRecordOnOpen(handle, cleanPath);
    this.completeResponse(BigInt(handle), HttpfsStatus.ResponseReady);
  }

  _handleRead(handle, offsetBig, lengthBig, cap) {
    if (offsetBig > BigInt(Number.MAX_SAFE_INTEGER)) {
      this.completeResponse(-22n, HttpfsStatus.Error);
      return;
    }

    const offset = Number(offsetBig);
    const reqLen = lengthBig > BigInt(cap) ? cap : Number(lengthBig);

    let entryPath = null;
    let entryUrl = null;
    let cache = null;
    let chunkCache = null;
    if (!this._customRead) {
      const entry = this._handles.get(handle);
      if (!entry) {
        this.completeResponse(-9n, HttpfsStatus.Error);
        return;
      }
      entryPath = entry.path;
      entryUrl = entry.url || null;
      if (reqLen === 0) {
        this.completeResponse(0n, HttpfsStatus.ResponseReady);
        return;
      }
      cache = this._resolveRangeCache();
      chunkCache = !cache ? this._shared.chunkCache : null;
      if ((!cache || typeof cache.read !== 'function') && !chunkCache) {
        this.completeResponse(-19n, HttpfsStatus.Error);
        return;
      }
    } else if (reqLen === 0) {
      this.completeResponse(0n, HttpfsStatus.ResponseReady);
      return;
    }

    const rec = this._getOrCreateHandleRecord(handle, isVideoReadAheadPath(entryPath));
    const slotCount = this._shared.slotCount || 1;
    let wasExpected = false;
    if (rec.expectedNextOffset >= 0) {
      if (slotCount <= 1) {
        wasExpected = offset === rec.expectedNextOffset;
      } else {
        // In multi-slot pipelining, sibling slots from the same sequential burst may wake up
        // slightly out of strict offset order within +/- slotCount chunks of the stream frontier.
        const forwardSlack = (slotCount - 1) * reqLen;
        const backwardSlack = slotCount * reqLen;
        wasExpected =
          offset !== rec.lastOffset &&
          offset >= rec.expectedNextOffset - backwardSlack &&
          offset <= rec.expectedNextOffset + forwardSlack;
      }
    }
    rec.reads += 1;
    rec.lastOffset = offset;
    if (wasExpected) {
      rec.sequentialCount += 1;
      if (rec.sequentialCount > rec.maxSequentialCount) {
        rec.maxSequentialCount = rec.sequentialCount;
      }
      rec.burstReads += 1;
      this._stats.burstReads += 1;
    } else {
      rec.sequentialCount = 0;
    }
    const provisionalNextOffset = offset + reqLen;
    if (wasExpected && slotCount > 1) {
      rec.expectedNextOffset = Math.max(rec.expectedNextOffset, provisionalNextOffset);
    } else {
      rec.expectedNextOffset = provisionalNextOffset;
    }

    const raCfg = this._shared.readAhead;
    const windowBytes = computeReadAheadWindowSize(reqLen, rec.sequentialCount, {
      path: entryPath || undefined,
      isVideoStream: rec.isVideoStream,
      maxReadAheadBytes: raCfg.maxReadAheadBytes,
      baseReadAheadBytes: raCfg.baseReadAheadBytes,
      minReadAheadBytes: raCfg.minReadAheadBytes,
      midReadAheadBytes: raCfg.midReadAheadBytes,
      highReadAheadBytes: raCfg.highReadAheadBytes,
      rampMidThreshold: raCfg.rampMidThreshold,
      rampHighThreshold: raCfg.rampHighThreshold,
    });
    rec.currentWindowBytes = windowBytes;

    const invokeBackendRead = (readOffset, readLen) => {
      if (this._customRead) {
        return this._customRead(handle, readOffset, readLen);
      }
      return cache.read(entryPath, readOffset, readLen);
    };

    const finishWithTelemetry = (res, expectedReqLen, isReadAheadHit) => {
      const bytes = extractReadBytes(res);
      if (bytes === undefined) {
        this.completeResponse(-9n, HttpfsStatus.Error);
        return;
      }
      if (bytes === null) {
        this.completeResponse(-5n, HttpfsStatus.Error);
        return;
      }
      const toCopy = Math.min(bytes.byteLength, expectedReqLen);
      if (toCopy < expectedReqLen && rec.expectedNextOffset === provisionalNextOffset) {
        rec.expectedNextOffset = offset + toCopy;
      }
      if (isReadAheadHit && toCopy > 0) {
        rec.readAheadHits += 1;
        rec.readAheadBytesServed += toCopy;
        this._stats.readAheadHits += 1;
        this._stats.readAheadBytesServed += toCopy;
      }
      if (toCopy > 0) {
        this._payloadView.set(bytes.subarray(0, toCopy), 0);
      }
      this.completeResponse(BigInt(toCopy), HttpfsStatus.ResponseReady);
    };

    // Bounded in-worker LRU chunk cache + sequential read-ahead mode (no external RangeCache/custom read).
    if (chunkCache) {
      const url = entryUrl || resolveHttpfsUrl(entryPath, this._shared.urlConfig);
      const syncBytes = chunkCache.tryReadSync(url, offset, reqLen);
      if (syncBytes !== null) {
        finishWithTelemetry(syncBytes, reqLen, false);
        return;
      }
      return chunkCache.read(url, offset, reqLen).then((resolved) => {
        finishWithTelemetry(resolved, reqLen, false);
      });
    }

    // Preserve exact legacy direct pass-through when host-side read-ahead buffering is disabled.
    if (!raCfg.enabled) {
      const res = invokeBackendRead(offset, reqLen);
      if (isThenable(res)) {
        return res.then((resolved) => {
          finishWithTelemetry(resolved, reqLen, false);
        });
      }
      finishWithTelemetry(res, reqLen, false);
      return;
    }

    // 1. Synchronous hit from this handle's adaptive read-ahead window buffer.
    const raHit = this._tryServeFromReadAheadBuffer(rec, offset, reqLen);
    if (raHit !== null) {
      finishWithTelemetry(raHit, reqLen, true);
      return;
    }

    // 2. Coalesce onto an in-flight read-ahead fetch issued by a sibling pipeline slot.
    const inflight = rec.inFlightReadAhead;
    if (inflight && offset >= inflight.startOffset && offset + reqLen <= inflight.endOffset) {
      return inflight.promise.then((bufBytes) => {
        if (bufBytes instanceof Uint8Array) {
          const rel = offset - inflight.startOffset;
          if (rel >= 0 && rel + reqLen <= bufBytes.byteLength) {
            rec.coalescedSlotReads += 1;
            this._stats.coalescedSlotReads += 1;
            finishWithTelemetry(bufBytes.subarray(rel, rel + reqLen), reqLen, true);
            return;
          }
        }
        const fallbackRes = invokeBackendRead(offset, reqLen);
        if (isThenable(fallbackRes)) {
          return fallbackRes.then((resolved) => {
            finishWithTelemetry(resolved, reqLen, false);
          });
        }
        finishWithTelemetry(fallbackRes, reqLen, false);
      });
    }

    // 3. Miss: issue a single backend read sized to max(reqLen, adaptiveWindowBytes).
    const fetchLen = windowBytes > reqLen ? windowBytes : reqLen;
    const isWindowFetch = fetchLen > reqLen;
    if (isWindowFetch) {
      rec.readAheadFetches += 1;
      this._stats.readAheadFetches += 1;
    }

    const populateAndFinish = (res) => {
      const bytes = extractReadBytes(res);
      if (bytes === undefined || bytes === null) {
        finishWithTelemetry(res, reqLen, false);
        return null;
      }
      if (isWindowFetch && bytes.byteLength > reqLen) {
        const stored = new Uint8Array(bytes.byteLength);
        stored.set(bytes);
        const existing = rec.raBuffer;
        const existingCoversNew =
          existing &&
          existing.bytes &&
          existing.startOffset <= offset &&
          existing.startOffset + existing.bytes.byteLength >= offset + stored.byteLength;
        if (!existingCoversNew) {
          rec.raBuffer = {
            startOffset: offset,
            bytes: stored,
            eof: bytes.byteLength < reqLen,
          };
        }
        finishWithTelemetry(stored, reqLen, false);
        return stored;
      }
      if (!wasExpected) {
        rec.raBuffer = null;
      }
      finishWithTelemetry(bytes, reqLen, false);
      return bytes;
    };

    const res = invokeBackendRead(offset, fetchLen);
    if (isThenable(res)) {
      let inflightEntry = null;
      const wrapped = res.then(
        (resolved) => {
          const stored = populateAndFinish(resolved);
          if (rec.inFlightReadAhead === inflightEntry) {
            rec.inFlightReadAhead = null;
          }
          return stored;
        },
        (err) => {
          if (rec.inFlightReadAhead === inflightEntry) {
            rec.inFlightReadAhead = null;
          }
          throw err;
        },
      );
      if (isWindowFetch) {
        const sharedPromise = wrapped.then(
          (stored) => stored,
          () => null,
        );
        inflightEntry = {
          startOffset: offset,
          endOffset: offset + fetchLen,
          promise: sharedPromise,
        };
        rec.inFlightReadAhead = inflightEntry;
      }
      return wrapped;
    }

    populateAndFinish(res);
  }

  _handleClose(handle) {
    if (this._customClose) {
      const res = this._customClose(handle);
      if (isThenable(res)) {
        return res.then((resolved) => {
          this._finishCloseResult(resolved, handle);
        });
      }
      this._finishCloseResult(res, handle);
      return;
    }

    if (this._handles.delete(handle)) {
      this._markHandleClosed(handle);
      this.completeResponse(0n, HttpfsStatus.ResponseReady);
    } else {
      this.completeResponse(-9n, HttpfsStatus.Error);
    }
  }

  _dispatchOnce() {
    const status = Atomics.load(this._statusView, 0);
    if (status !== HttpfsStatus.RequestReady) {
      return false;
    }

    this._stats.requests += 1;
    this._slotRequests += 1;
    const op = this._headerView.getUint32(HttpfsOffsets.OP, true);
    const handle = this._headerView.getInt32(HttpfsOffsets.HANDLE, true);
    const offsetBig = this._headerView.getBigUint64(HttpfsOffsets.OFFSET, true);
    const lengthBig = this._headerView.getBigUint64(HttpfsOffsets.LENGTH, true);
    const cap = this.payloadCapacity;

    const startMs = this._now();
    const trackHandle = (op === HttpfsOp.Read || op === HttpfsOp.Close) && handle >= 0 ? handle : null;
    const handleRec = trackHandle !== null ? this._getOrCreateHandleRecord(trackHandle) : null;

    this._stats.activePendingSlots += 1;
    if (this._stats.activePendingSlots > this._stats.peakPendingSlots) {
      this._stats.peakPendingSlots = this._stats.activePendingSlots;
    }
    if (handleRec) {
      handleRec.activePendingSlots += 1;
      if (handleRec.activePendingSlots > handleRec.peakPendingSlots) {
        handleRec.peakPendingSlots = handleRec.activePendingSlots;
      }
    }

    let finalized = false;
    const finalizeTiming = () => {
      if (finalized) {
        return;
      }
      finalized = true;
      const elapsed = Math.max(0, this._now() - startMs);
      this._stats.activePendingSlots = Math.max(0, this._stats.activePendingSlots - 1);
      this._stats.totalWaitMs += elapsed;
      this._stats.waitSamples += 1;
      if (elapsed > this._stats.maxWaitMs) {
        this._stats.maxWaitMs = elapsed;
      }
      if (handleRec) {
        handleRec.activePendingSlots = Math.max(0, handleRec.activePendingSlots - 1);
        handleRec.totalWaitMs += elapsed;
        handleRec.waitSamples += 1;
        if (elapsed > handleRec.maxWaitMs) {
          handleRec.maxWaitMs = elapsed;
        }
      }
    };
    this._pendingFinalize = finalizeTiming;

    let pending;
    try {
      if (op === HttpfsOp.Open) {
        pending = this._handleOpen(lengthBig, cap);
      } else if (op === HttpfsOp.Read) {
        pending = this._handleRead(handle, offsetBig, lengthBig, cap);
      } else if (op === HttpfsOp.Close) {
        pending = this._handleClose(handle);
      } else {
        this.completeResponse(-1n, HttpfsStatus.Error);
      }
    } catch {
      const errCode = op === HttpfsOp.Open ? -2n : -5n;
      this.completeResponse(errCode, HttpfsStatus.Error);
      finalizeTiming();
      this._pendingFinalize = null;
      this._stats.syncFastPathHits += 1;
      this._slotSyncFastPathHits += 1;
      return true;
    }

    if (isThenable(pending)) {
      this._stats.asyncPathHits += 1;
      this._slotAsyncPathHits += 1;
      return pending.then(
        () => {
          finalizeTiming();
          this._pendingFinalize = null;
          return true;
        },
        () => {
          const errCode = op === HttpfsOp.Open ? -2n : -5n;
          this.completeResponse(errCode, HttpfsStatus.Error);
          finalizeTiming();
          this._pendingFinalize = null;
          return true;
        },
      );
    }

    finalizeTiming();
    this._pendingFinalize = null;
    this._stats.syncFastPathHits += 1;
    this._slotSyncFastPathHits += 1;
    return true;
  }

  processOnce() {
    const dispatched = this._dispatchOnce();
    return isThenable(dispatched) ? dispatched : Promise.resolve(dispatched);
  }

  _waitForChange(expectedStatus) {
    if (!this._forcePoll && typeof Atomics.waitAsync === 'function') {
      return Atomics.waitAsync(this._statusView, 0, expectedStatus).value;
    }
    return new Promise((resolve) => {
      if (typeof setTimeout !== 'function') {
        resolve('ok');
        return;
      }
      this._pollResolve = resolve;
      this._pollTimer = setTimeout(() => {
        this._pollTimer = null;
        this._pollResolve = null;
        resolve('timed-out');
      }, this._fallbackPollMs);
    });
  }

  start() {
    if (this._running && this._loopPromise) {
      return this._loopPromise;
    }
    this._running = true;
    this._loopPromise = (async () => {
      let syncBurst = 0;
      while (this._running) {
        const status = Atomics.load(this._statusView, 0);
        if (status === HttpfsStatus.Shutdown) {
          break;
        }
        if (status === HttpfsStatus.RequestReady) {
          const dispatched = this._dispatchOnce();
          if (isThenable(dispatched)) {
            syncBurst = 0;
            await dispatched;
          } else {
            syncBurst += 1;
            if (syncBurst >= 128) {
              syncBurst = 0;
              await Promise.resolve();
            }
          }
          continue;
        }
        const waitResult = this._waitForChange(status);
        if (isThenable(waitResult)) {
          syncBurst = 0;
          await waitResult;
        } else {
          syncBurst += 1;
          if (syncBurst >= 128) {
            syncBurst = 0;
            await Promise.resolve();
          }
        }
      }
      this._running = false;
    })();
    return this._loopPromise;
  }

  shutdown() {
    Atomics.store(this._statusView, 0, HttpfsStatus.Shutdown);
    Atomics.notify(this._statusView, 0);
  }

  stop() {
    this._running = false;
    if (this._pollTimer !== null && typeof clearTimeout === 'function') {
      clearTimeout(this._pollTimer);
      this._pollTimer = null;
    }
    if (this._pollResolve) {
      const fn = this._pollResolve;
      this._pollResolve = null;
      fn('stopped');
    }
    this.shutdown();
    return this._loopPromise || Promise.resolve();
  }
}

class HttpfsMailboxClient {
  constructor(sab, optionsOrResponder = {}) {
    assertValidMailboxBuffer(sab);
    const options =
      typeof optionsOrResponder === 'function'
        ? { responder: optionsOrResponder }
        : optionsOrResponder || {};
    this.sab = sab;
    this._statusView = new Int32Array(sab, 0, 1);
    this._headerView = new DataView(sab, 0, HTTPFS_HEADER_BYTES);
    this._payloadView = new Uint8Array(
      sab,
      HTTPFS_HEADER_BYTES,
      sab.byteLength - HTTPFS_HEADER_BYTES,
    );
    this._responder = typeof options.responder === 'function' ? options.responder : null;
    this._timeoutMs =
      typeof options.timeoutMs === 'number' && options.timeoutMs > 0
        ? options.timeoutMs
        : 30000;
  }

  setResponder(responder) {
    this._responder = typeof responder === 'function' ? responder : null;
  }

  get payloadCapacity() {
    return effectivePayloadCapacity(this._headerView, this._payloadView.byteLength);
  }

  static shutdownWaiter(sab) {
    assertValidMailboxBuffer(sab);
    const statusView = new Int32Array(sab, 0, 1);
    Atomics.store(statusView, 0, HttpfsStatus.Shutdown);
    Atomics.notify(statusView, 0);
  }

  _resetToIdle(status) {
    if (status !== HttpfsStatus.Shutdown) {
      Atomics.store(this._statusView, 0, HttpfsStatus.Idle);
      Atomics.notify(this._statusView, 0, 1);
    }
  }

  _executeRoundtripSync() {
    Atomics.store(this._statusView, 0, HttpfsStatus.RequestReady);
    Atomics.notify(this._statusView, 0, 1);

    if (this._responder) {
      this._responder(this.sab);
    }

    let cur = Atomics.load(this._statusView, 0);
    while (cur === HttpfsStatus.RequestReady) {
      const waitResult = Atomics.wait(
        this._statusView,
        0,
        HttpfsStatus.RequestReady,
        this._timeoutMs,
      );
      cur = Atomics.load(this._statusView, 0);
      if (waitResult === 'timed-out' && cur === HttpfsStatus.RequestReady) {
        return { status: HttpfsStatus.Error, result: -110n };
      }
    }

    return {
      status: cur,
      result: this._headerView.getBigInt64(HttpfsOffsets.RESULT, true),
    };
  }

  async _executeRoundtripAsync() {
    Atomics.store(this._statusView, 0, HttpfsStatus.RequestReady);
    Atomics.notify(this._statusView, 0, 1);

    if (this._responder) {
      await this._responder(this.sab);
    }

    let cur = Atomics.load(this._statusView, 0);
    while (cur === HttpfsStatus.RequestReady) {
      if (typeof Atomics.waitAsync === 'function') {
        const waitObj = Atomics.waitAsync(
          this._statusView,
          0,
          HttpfsStatus.RequestReady,
          this._timeoutMs,
        );
        const waitResult = waitObj.async ? await waitObj.value : waitObj.value;
        cur = Atomics.load(this._statusView, 0);
        if (waitResult === 'timed-out' && cur === HttpfsStatus.RequestReady) {
          return { status: HttpfsStatus.Error, result: -110n };
        }
      } else {
        await new Promise((r) => setTimeout(r, 1));
        cur = Atomics.load(this._statusView, 0);
      }
    }

    return {
      status: cur,
      result: this._headerView.getBigInt64(HttpfsOffsets.RESULT, true),
    };
  }

  _prepareOpen(relPath) {
    if (typeof relPath !== 'string' || relPath.length === 0) {
      return false;
    }
    const encoded = textEncoder.encode(relPath);
    const cap = this.payloadCapacity;
    if (encoded.byteLength === 0 || encoded.byteLength > cap) {
      return false;
    }
    this._headerView.setUint32(HttpfsOffsets.OP, HttpfsOp.Open, true);
    this._headerView.setInt32(HttpfsOffsets.HANDLE, -1, true);
    this._headerView.setBigUint64(HttpfsOffsets.OFFSET, 0n, true);
    this._headerView.setBigUint64(HttpfsOffsets.LENGTH, BigInt(encoded.byteLength), true);
    this._headerView.setBigInt64(HttpfsOffsets.RESULT, -1n, true);
    this._payloadView.set(encoded, 0);
    return true;
  }

  open(relPath) {
    if (!this._prepareOpen(relPath)) {
      return -1;
    }
    const { status, result } = this._executeRoundtripSync();
    this._resetToIdle(status);
    if (status !== HttpfsStatus.ResponseReady || result < 0n) {
      return -1;
    }
    return Number(result) | 0;
  }

  async openAsync(relPath) {
    if (!this._prepareOpen(relPath)) {
      return -1;
    }
    const { status, result } = await this._executeRoundtripAsync();
    this._resetToIdle(status);
    if (status !== HttpfsStatus.ResponseReady || result < 0n) {
      return -1;
    }
    return Number(result) | 0;
  }

  read(handle, offset, dst, len) {
    const h = Number(handle) | 0;
    if (h < 0) return -1;
    if (!(dst instanceof Uint8Array)) return -1;
    const reqLen = len === undefined ? dst.byteLength : toSafeNonNegNumber(len, 'len');
    if (reqLen === 0) return 0;
    if (reqLen > dst.byteLength) return -1;

    const cap = this.payloadCapacity;
    if (cap === 0) return -1;

    let curOffset = toSafeUint64BigInt(offset, 'offset');
    let remaining = reqLen;
    let outCursor = 0;
    let totalRead = 0;

    while (remaining > 0) {
      const chunk = Math.min(remaining, cap);
      this._headerView.setUint32(HttpfsOffsets.OP, HttpfsOp.Read, true);
      this._headerView.setInt32(HttpfsOffsets.HANDLE, h, true);
      this._headerView.setBigUint64(HttpfsOffsets.OFFSET, curOffset, true);
      this._headerView.setBigUint64(HttpfsOffsets.LENGTH, BigInt(chunk), true);
      this._headerView.setBigInt64(HttpfsOffsets.RESULT, -1n, true);

      const { status, result } = this._executeRoundtripSync();
      if (status !== HttpfsStatus.ResponseReady || result < 0n || result > BigInt(chunk)) {
        this._resetToIdle(status);
        return result < 0n ? Number(result) : -1;
      }

      const got = Number(result);
      if (got > 0) {
        dst.set(this._payloadView.subarray(0, got), outCursor);
      }
      this._resetToIdle(status);

      outCursor += got;
      curOffset += BigInt(got);
      remaining -= got;
      totalRead += got;

      if (got < chunk) {
        break;
      }
    }

    return totalRead;
  }

  async readAsync(handle, offset, dst, len) {
    const h = Number(handle) | 0;
    if (h < 0) return -1;
    if (!(dst instanceof Uint8Array)) return -1;
    const reqLen = len === undefined ? dst.byteLength : toSafeNonNegNumber(len, 'len');
    if (reqLen === 0) return 0;
    if (reqLen > dst.byteLength) return -1;

    const cap = this.payloadCapacity;
    if (cap === 0) return -1;

    let curOffset = toSafeUint64BigInt(offset, 'offset');
    let remaining = reqLen;
    let outCursor = 0;
    let totalRead = 0;

    while (remaining > 0) {
      const chunk = Math.min(remaining, cap);
      this._headerView.setUint32(HttpfsOffsets.OP, HttpfsOp.Read, true);
      this._headerView.setInt32(HttpfsOffsets.HANDLE, h, true);
      this._headerView.setBigUint64(HttpfsOffsets.OFFSET, curOffset, true);
      this._headerView.setBigUint64(HttpfsOffsets.LENGTH, BigInt(chunk), true);
      this._headerView.setBigInt64(HttpfsOffsets.RESULT, -1n, true);

      const { status, result } = await this._executeRoundtripAsync();
      if (status !== HttpfsStatus.ResponseReady || result < 0n || result > BigInt(chunk)) {
        this._resetToIdle(status);
        return result < 0n ? Number(result) : -1;
      }

      const got = Number(result);
      if (got > 0) {
        dst.set(this._payloadView.subarray(0, got), outCursor);
      }
      this._resetToIdle(status);

      outCursor += got;
      curOffset += BigInt(got);
      remaining -= got;
      totalRead += got;

      if (got < chunk) {
        break;
      }
    }

    return totalRead;
  }

  close(handle) {
    const h = Number(handle) | 0;
    if (h < 0) return false;
    this._headerView.setUint32(HttpfsOffsets.OP, HttpfsOp.Close, true);
    this._headerView.setInt32(HttpfsOffsets.HANDLE, h, true);
    this._headerView.setBigUint64(HttpfsOffsets.OFFSET, 0n, true);
    this._headerView.setBigUint64(HttpfsOffsets.LENGTH, 0n, true);
    this._headerView.setBigInt64(HttpfsOffsets.RESULT, -1n, true);

    const { status, result } = this._executeRoundtripSync();
    this._resetToIdle(status);
    return status === HttpfsStatus.ResponseReady && result >= 0n;
  }

  async closeAsync(handle) {
    const h = Number(handle) | 0;
    if (h < 0) return false;
    this._headerView.setUint32(HttpfsOffsets.OP, HttpfsOp.Close, true);
    this._headerView.setInt32(HttpfsOffsets.HANDLE, h, true);
    this._headerView.setBigUint64(HttpfsOffsets.OFFSET, 0n, true);
    this._headerView.setBigUint64(HttpfsOffsets.LENGTH, 0n, true);
    this._headerView.setBigInt64(HttpfsOffsets.RESULT, -1n, true);

    const { status, result } = await this._executeRoundtripAsync();
    this._resetToIdle(status);
    return status === HttpfsStatus.ResponseReady && result >= 0n;
  }
}

function createHttpfsMailboxPool(
  slotCount = HTTPFS_DEFAULT_PIPELINE_SLOTS,
  payloadCapacity = HTTPFS_PAYLOAD_BYTES,
) {
  const count = Math.floor(Number(slotCount));
  if (!Number.isInteger(count) || count < 1 || count > HTTPFS_MAX_PIPELINE_SLOTS) {
    throw new RangeError(
      `Invalid HTTPFS mailbox slotCount: ${slotCount} (expected 1..${HTTPFS_MAX_PIPELINE_SLOTS})`,
    );
  }
  const slots = [];
  for (let i = 0; i < count; i += 1) {
    slots.push(createHttpfsMailboxBuffer(payloadCapacity));
  }
  return slots;
}

class HttpfsMailboxPoolHost {
  constructor(slotBuffers, options = {}) {
    if (
      !Array.isArray(slotBuffers) ||
      slotBuffers.length < 1 ||
      slotBuffers.length > HTTPFS_MAX_PIPELINE_SLOTS
    ) {
      throw new RangeError(
        `HttpfsMailboxPoolHost requires 1..${HTTPFS_MAX_PIPELINE_SLOTS} SharedArrayBuffer slots`,
      );
    }
    this._shared = createSharedMailboxHostState(options, true, slotBuffers.length);
    this._slots = slotBuffers.map(
      (sab) => new HttpfsMailboxHost(sab, { ...options, _sharedState: this._shared }),
    );
  }

  get slotCount() {
    return this._slots.length;
  }

  get isRunning() {
    return this._slots.some((slot) => slot.isRunning);
  }

  get chunkCache() {
    return this._shared.chunkCache || null;
  }

  flushPrefetches() {
    return this._slots[0].flushPrefetches();
  }

  getHandleTelemetry(handle) {
    return this._slots[0].getHandleTelemetry(handle);
  }

  getStats() {
    const base = this._slots[0].getStats();
    return {
      ...base,
      slotCount: this._slots.length,
      perSlot: this._slots.map((slot, index) => ({
        slotIndex: index,
        requests: slot._slotRequests,
        syncFastPathHits: slot._slotSyncFastPathHits,
        asyncPathHits: slot._slotAsyncPathHits,
      })),
    };
  }

  processOnce() {
    return Promise.all(this._slots.map((slot) => slot.processOnce()));
  }

  start() {
    return Promise.all(this._slots.map((slot) => slot.start()));
  }

  shutdown() {
    for (const slot of this._slots) {
      slot.shutdown();
    }
  }

  stop() {
    return Promise.all(this._slots.map((slot) => slot.stop()));
  }
}

class HttpfsMailboxPoolClient {
  constructor(slotBuffers, options = {}) {
    if (
      !Array.isArray(slotBuffers) ||
      slotBuffers.length < 1 ||
      slotBuffers.length > HTTPFS_MAX_PIPELINE_SLOTS
    ) {
      throw new RangeError(
        `HttpfsMailboxPoolClient requires 1..${HTTPFS_MAX_PIPELINE_SLOTS} SharedArrayBuffer slots`,
      );
    }
    this._clients = slotBuffers.map((sab) => new HttpfsMailboxClient(sab, options));
    this._freeSlots = slotBuffers.map((_, idx) => idx);
    this._waiters = [];
    this._now = typeof options.now === 'function' ? options.now : defaultNowMs;
    this._stats = {
      requests: 0,
      activePendingSlots: 0,
      peakPendingSlots: 0,
      queuedAcquires: 0,
      totalWaitMs: 0,
      waitSamples: 0,
      maxWaitMs: 0,
    };
  }

  get slotCount() {
    return this._clients.length;
  }

  get availableSlots() {
    return this._freeSlots.length;
  }

  _acquireSlot() {
    if (this._freeSlots.length > 0) {
      return Promise.resolve(this._freeSlots.shift());
    }
    this._stats.queuedAcquires += 1;
    return new Promise((resolve) => {
      this._waiters.push(resolve);
    });
  }

  _releaseSlot(slotIdx) {
    if (this._waiters.length > 0) {
      const next = this._waiters.shift();
      next(slotIdx);
    } else {
      this._freeSlots.push(slotIdx);
    }
  }

  async _withSlot(fn) {
    const startMs = this._now();
    this._stats.requests += 1;
    const slotIdx = await this._acquireSlot();
    this._stats.activePendingSlots += 1;
    if (this._stats.activePendingSlots > this._stats.peakPendingSlots) {
      this._stats.peakPendingSlots = this._stats.activePendingSlots;
    }
    try {
      return await fn(this._clients[slotIdx], slotIdx);
    } finally {
      const elapsed = Math.max(0, this._now() - startMs);
      this._stats.activePendingSlots = Math.max(0, this._stats.activePendingSlots - 1);
      this._stats.totalWaitMs += elapsed;
      this._stats.waitSamples += 1;
      if (elapsed > this._stats.maxWaitMs) {
        this._stats.maxWaitMs = elapsed;
      }
      this._releaseSlot(slotIdx);
    }
  }

  openAsync(relPath) {
    return this._withSlot((client) => client.openAsync(relPath));
  }

  readAsync(handle, offset, dst, len) {
    return this._withSlot((client) => client.readAsync(handle, offset, dst, len));
  }

  closeAsync(handle) {
    return this._withSlot((client) => client.closeAsync(handle));
  }

  readBatchAsync(requests) {
    if (!Array.isArray(requests)) {
      return Promise.reject(new TypeError('readBatchAsync requires an array of read descriptors'));
    }
    return Promise.all(
      requests.map((req) => {
        const handle = req.handle;
        const offset = req.offset;
        const reqLen = req.len !== undefined ? req.len : req.length;
        const dst =
          req.dst instanceof Uint8Array
            ? req.dst
            : new Uint8Array(toSafeNonNegNumber(reqLen, 'len'));
        const len = reqLen !== undefined ? reqLen : dst.byteLength;
        return this.readAsync(handle, offset, dst, len).then((bytesRead) => ({
          handle,
          offset,
          bytesRead,
          dst,
        }));
      }),
    );
  }

  getStats() {
    return {
      slotCount: this._clients.length,
      availableSlots: this._freeSlots.length,
      requests: this._stats.requests,
      activePendingSlots: this._stats.activePendingSlots,
      peakPendingSlots: this._stats.peakPendingSlots,
      queuedAcquires: this._stats.queuedAcquires,
      avgWaitMs:
        this._stats.waitSamples > 0 ? this._stats.totalWaitMs / this._stats.waitSamples : 0,
      totalWaitMs: this._stats.totalWaitMs,
      maxWaitMs: this._stats.maxWaitMs,
    };
  }
}

const exportedApi = {
  HTTPFS_MAILBOX_VERSION,
  HTTPFS_HEADER_BYTES,
  HTTPFS_PAYLOAD_BYTES,
  HTTPFS_MAILBOX_BYTES,
  HTTPFS_DIRECTSTORAGE_CHUNK_BYTES,
  HTTPFS_READAHEAD_BASE_BYTES,
  HTTPFS_READAHEAD_MID_BYTES,
  HTTPFS_READAHEAD_MAX_BYTES,
  HTTPFS_DEFAULT_PIPELINE_SLOTS,
  HTTPFS_MAX_PIPELINE_SLOTS,
  HTTPFS_DEFAULT_CACHE_MAX_BYTES,
  HTTPFS_DEFAULT_CACHE_CHUNK_SIZE,
  HTTPFS_DEFAULT_READAHEAD_CHUNKS,
  HTTPFS_MAX_READAHEAD_CHUNKS,
  HttpfsStatus,
  HttpfsOp,
  HttpfsOffsets,
  normalizeRelPath,
  isVideoReadAheadPath,
  isCacheableImmutableUrl,
  resolveHttpfsUrl,
  computeReadAheadWindowSize,
  HttpfsLruChunkCache,
  initHttpfsMailbox,
  createHttpfsMailboxBuffer,
  createHttpfsMailboxPool,
  HttpfsMailboxHost,
  HttpfsMailboxClient,
  HttpfsMailboxPoolHost,
  HttpfsMailboxPoolClient,
};

if (typeof globalThis !== 'undefined') {
  globalThis.PlayGta6HttpfsMailbox = exportedApi;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = exportedApi;
}
})();
