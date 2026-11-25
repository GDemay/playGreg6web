'use strict';

// Fixed-size HTTP 206 Range chunk cache with memory LRU, optional OPFS L2 persistence,
// concurrent in-flight deduplication, coalesced adjacent range merging, adaptive
// sequential read-ahead prefetching, and bootset.json prefetching.
// Compatible with classic Web Workers (importScripts) and Node CommonJS/ESM tests.

const DEFAULT_CHUNK_SIZE = 256 * 1024; // 256 KiB
const DEFAULT_MAX_BYTES = 64 * 1024 * 1024; // 64 MiB
const DEFAULT_PREFETCH_CONCURRENCY = 4;
const DEFAULT_COALESCE_MAX_GAP_BYTES = 64 * 1024; // 64 KiB
const DEFAULT_COALESCE_MAX_MERGED_BYTES = 2 * 1024 * 1024; // 2 MiB
const DEFAULT_MAX_READ_AHEAD_BYTES = 2 * 1024 * 1024; // 2 MiB (legacy W181 constant)
const DEFAULT_MIN_READAHEAD_BYTES = 256 * 1024; // 256 KiB (W312 adaptive floor)
const DEFAULT_MAX_READAHEAD_BYTES = 4 * 1024 * 1024; // 4 MiB (W312 adaptive ceiling)
const DEFAULT_SEQUENTIAL_STREAK_THRESHOLD = 2;
const DEFAULT_MAX_PENDING_OPFS_WRITES = 64;
const DEFAULT_MAX_OPFS_WRITE_CONCURRENCY = 4;
const ESTIMATED_HTTP_RANGE_OVERHEAD_BYTES = 256;
const MIN_AUTO_OPTIMIZE_CHUNK_SIZE = 16 * 1024; // 16 KiB
const MIN_AUTO_OPTIMIZE_CACHE_BYTES = 1024 * 1024; // 1 MiB
const MAX_TRACKED_CHUNK_LENGTHS = 65536;
const MAX_TRACKED_FILE_SIZES = 4096;
const MAX_TRACKED_SEQUENTIAL_KEYS = 1024;

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

function buildFileFingerprint(sizeOrMeta, etag = null) {
  if (sizeOrMeta === null || sizeOrMeta === undefined) {
    if (typeof etag === 'string' && etag.trim().length > 0) {
      return `e_${fnv1a64Hex(etag.trim()).slice(0, 16)}`;
    }
    return null;
  }
  if (typeof sizeOrMeta === 'string') {
    const trimmed = sizeOrMeta.trim();
    if (!trimmed) return null;
    if (/^[0-9a-zA-Z_-]{1,64}$/.test(trimmed)) {
      return trimmed;
    }
    return `h_${fnv1a64Hex(trimmed).slice(0, 16)}`;
  }

  let size = null;
  let tag = etag;
  if (typeof sizeOrMeta === 'object') {
    if (typeof sizeOrMeta.fileFingerprint === 'string' && sizeOrMeta.fileFingerprint.trim()) {
      return buildFileFingerprint(sizeOrMeta.fileFingerprint);
    }
    if (sizeOrMeta.fileSize !== undefined) {
      size = sizeOrMeta.fileSize;
    } else if (sizeOrMeta.size !== undefined) {
      size = sizeOrMeta.size;
    }
    if (tag === null && sizeOrMeta.etag !== undefined) {
      tag = sizeOrMeta.etag;
    }
  } else {
    size = sizeOrMeta;
  }

  const validSize = Number.isSafeInteger(size) && size >= 0 ? size : null;
  const validTag = typeof tag === 'string' && tag.trim().length > 0 ? tag.trim() : null;

  if (validSize !== null && validTag !== null) {
    return `${validSize}_${fnv1a64Hex(validTag).slice(0, 16)}`;
  }
  if (validSize !== null) {
    return `${validSize}`;
  }
  if (validTag !== null) {
    return `e_${fnv1a64Hex(validTag).slice(0, 16)}`;
  }
  return null;
}

function readResponseHeader(headers, name) {
  if (!headers) return null;
  if (typeof headers.get === 'function') {
    return headers.get(name) || headers.get(name.toLowerCase()) || null;
  }
  if (typeof headers === 'object') {
    return headers[name] || headers[name.toLowerCase()] || null;
  }
  return null;
}

function opfsChunkFilePrefix(relPath, chunkIndex, chunkSize) {
  const encoded = utf8ToHex(relPath);
  const pathToken = encoded.length <= 128 ? encoded : `${encoded.slice(0, 64)}_${fnv1a64Hex(relPath)}`;
  return `c_${chunkSize}_${chunkIndex}_${pathToken}`;
}

function opfsChunkFileName(relPath, chunkIndex, chunkSize, fileFingerprint = null) {
  const prefix = opfsChunkFilePrefix(relPath, chunkIndex, chunkSize);
  const fp = buildFileFingerprint(fileFingerprint);
  if (fp) {
    return `${prefix}_fp_${fp}.bin`;
  }
  return `${prefix}.bin`;
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

function parseUnsatisfiedContentRangeTotal(header) {
  if (typeof header !== 'string') return null;
  const match = /^bytes\s+\*\/(\d+)$/i.exec(header.trim());
  if (!match) return null;
  const total = Number(match[1]);
  return Number.isSafeInteger(total) && total >= 0 ? total : null;
}

function normalizeInputRange(rawRange) {
  if (!rawRange || typeof rawRange !== 'object') return null;
  let offset;
  let length;

  if (Number.isFinite(rawRange.offset)) {
    offset = Math.floor(rawRange.offset);
  } else if (Number.isFinite(rawRange.start)) {
    offset = Math.floor(rawRange.start);
  } else {
    return null;
  }

  if (Number.isFinite(rawRange.length)) {
    length = Math.floor(rawRange.length);
  } else if (Number.isFinite(rawRange.end) && rawRange.end > offset) {
    length = Math.floor(rawRange.end) - offset;
  } else {
    return null;
  }

  if (offset < 0 || length <= 0) {
    return null;
  }

  const hasPath = typeof rawRange.path === 'string' && rawRange.path.length > 0;
  return {
    hasPath,
    path: hasPath ? rawRange.path : null,
    offset,
    length,
  };
}

function coalesceAdjacentRanges(
  ranges,
  maxGapBytes = DEFAULT_COALESCE_MAX_GAP_BYTES,
  maxMergedBytes = DEFAULT_COALESCE_MAX_MERGED_BYTES,
) {
  if (!Array.isArray(ranges) || ranges.length === 0) {
    return [];
  }

  const maxGap =
    Number.isFinite(maxGapBytes) && maxGapBytes >= 0
      ? Math.floor(maxGapBytes)
      : DEFAULT_COALESCE_MAX_GAP_BYTES;
  const maxMerged =
    Number.isFinite(maxMergedBytes) && maxMergedBytes > 0
      ? Math.floor(maxMergedBytes)
      : DEFAULT_COALESCE_MAX_MERGED_BYTES;

  const groups = new Map();
  for (let i = 0; i < ranges.length; i += 1) {
    const normalized = normalizeInputRange(ranges[i]);
    if (!normalized) continue;
    const groupKey = normalized.hasPath ? `p:${normalized.path}` : '__pathless__';
    let list = groups.get(groupKey);
    if (!list) {
      list = [];
      groups.set(groupKey, list);
    }
    list.push(normalized);
  }

  const merged = [];
  for (const list of groups.values()) {
    list.sort((a, b) => (a.offset !== b.offset ? a.offset - b.offset : a.length - b.length));

    let current = {
      hasPath: list[0].hasPath,
      path: list[0].path,
      offset: list[0].offset,
      length: list[0].length,
      mergedCount: 1,
      gapBytes: 0,
      originalBytes: list[0].length,
    };

    const pushCurrent = () => {
      const unionBytes = current.length - current.gapBytes;
      const overlapBytes = Math.max(0, current.originalBytes - unionBytes);
      const out = {
        offset: current.offset,
        length: current.length,
        end: current.offset + current.length,
        mergedCount: current.mergedCount,
        gapBytes: current.gapBytes,
        originalBytes: current.originalBytes,
        overlapBytes,
      };
      if (current.hasPath) {
        out.path = current.path;
      }
      merged.push(out);
    };

    for (let i = 1; i < list.length; i += 1) {
      const item = list[i];
      const curEnd = current.offset + current.length;
      const itemEnd = item.offset + item.length;
      const gap = item.offset > curEnd ? item.offset - curEnd : 0;
      const candidateEnd = Math.max(curEnd, itemEnd);
      const candidateLength = candidateEnd - current.offset;
      const isContained = item.offset >= current.offset && itemEnd <= curEnd;

      if (gap <= maxGap && (candidateLength <= maxMerged || isContained)) {
        current.length = candidateLength;
        current.mergedCount += 1;
        current.gapBytes += gap;
        current.originalBytes += item.length;
      } else {
        pushCurrent();
        current = {
          hasPath: item.hasPath,
          path: item.path,
          offset: item.offset,
          length: item.length,
          mergedCount: 1,
          gapBytes: 0,
          originalBytes: item.length,
        };
      }
    }

    pushCurrent();
  }

  return merged;
}

function detectSequentialRead(prevState, offset, length, options = {}) {
  const chunkSize =
    Number.isInteger(options.chunkSize) && options.chunkSize > 0
      ? options.chunkSize
      : DEFAULT_CHUNK_SIZE;
  const rawMaxReadAhead =
    options.maxReadaheadBytes !== undefined
      ? options.maxReadaheadBytes
      : options.maxReadAheadBytes;
  const maxReadAheadBytes =
    Number.isFinite(rawMaxReadAhead) && rawMaxReadAhead >= 0
      ? Math.floor(rawMaxReadAhead)
      : DEFAULT_MAX_READ_AHEAD_BYTES;
  const rawMinReadAhead =
    options.minReadaheadBytes !== undefined
      ? options.minReadaheadBytes
      : options.minReadAheadBytes !== undefined
        ? options.minReadAheadBytes
        : options.initialReadAheadBytes;
  const minReadaheadBytes =
    maxReadAheadBytes > 0
      ? Math.min(
          maxReadAheadBytes,
          Number.isFinite(rawMinReadAhead) && rawMinReadAhead > 0
            ? Math.floor(rawMinReadAhead)
            : Math.max(chunkSize, length),
        )
      : 0;
  const maxGapBytes =
    Number.isFinite(options.maxGapBytes) && options.maxGapBytes >= 0
      ? Math.floor(options.maxGapBytes)
      : Math.min(DEFAULT_COALESCE_MAX_GAP_BYTES, chunkSize);
  const maxOverlapBytes =
    Number.isFinite(options.maxOverlapBytes) && options.maxOverlapBytes >= 0
      ? Math.floor(options.maxOverlapBytes)
      : Math.min(DEFAULT_COALESCE_MAX_GAP_BYTES, Math.floor(chunkSize / 2));
  const streakThreshold =
    Number.isInteger(options.streakThreshold) && options.streakThreshold >= 1
      ? options.streakThreshold
      : DEFAULT_SEQUENTIAL_STREAK_THRESHOLD;
  const growthFactor =
    Number.isFinite(options.growthFactor) && options.growthFactor >= 1
      ? options.growthFactor
      : 2;
  const shrinkFactor =
    Number.isFinite(options.shrinkFactor) && options.shrinkFactor > 0 && options.shrinkFactor < 1
      ? options.shrinkFactor
      : 0.5;

  const end = offset + length;
  let isSequential = false;
  let isRandomSeek = false;
  let gapBytes = 0;
  let stride = length;
  let streak = 0;
  let readAheadBytes = 0;
  let adaptiveWindowBytes = minReadaheadBytes;

  if (
    prevState &&
    Number.isFinite(prevState.lastOffset) &&
    Number.isFinite(prevState.lastEnd)
  ) {
    const prevOffset = prevState.lastOffset;
    const prevEnd = prevState.lastEnd;
    const prevStreak = Number.isInteger(prevState.streak) && prevState.streak >= 0 ? prevState.streak : 0;
    const prevAdaptiveWindow =
      maxReadAheadBytes > 0
        ? Math.min(
            maxReadAheadBytes,
            Math.max(
              minReadaheadBytes,
              Number.isFinite(prevState.adaptiveWindowBytes) && prevState.adaptiveWindowBytes > 0
                ? Math.floor(prevState.adaptiveWindowBytes)
                : Number.isFinite(prevState.readAheadBytes) && prevState.readAheadBytes > 0
                  ? Math.floor(prevState.readAheadBytes)
                  : minReadaheadBytes,
            ),
          )
        : 0;

    if (offset === prevEnd) {
      isSequential = true;
      gapBytes = 0;
      stride = offset - prevOffset;
      streak = prevStreak + 1;
    } else if (offset > prevEnd && offset - prevEnd <= maxGapBytes) {
      isSequential = true;
      gapBytes = offset - prevEnd;
      stride = offset - prevOffset;
      streak = prevStreak + 1;
    } else if (
      offset > prevOffset &&
      offset < prevEnd &&
      end > prevEnd &&
      prevEnd - offset <= maxOverlapBytes
    ) {
      isSequential = true;
      gapBytes = 0;
      stride = offset - prevOffset;
      streak = prevStreak + 1;
    } else {
      isSequential = false;
      isRandomSeek = true;
      gapBytes = 0;
      stride = length;
      streak = 0;
    }

    if (isRandomSeek) {
      adaptiveWindowBytes =
        maxReadAheadBytes > 0
          ? Math.max(minReadaheadBytes, Math.floor(prevAdaptiveWindow * shrinkFactor))
          : 0;
    } else if (isSequential && streak >= streakThreshold && maxReadAheadBytes > 0) {
      const prevWindow =
        Number.isFinite(prevState.readAheadBytes) && prevState.readAheadBytes > 0
          ? prevState.readAheadBytes
          : 0;
      if (prevWindow > 0) {
        readAheadBytes = Math.min(
          maxReadAheadBytes,
          Math.max(minReadaheadBytes, Math.floor(prevWindow * growthFactor)),
        );
      } else {
        const baseWindow = Math.min(maxReadAheadBytes, Math.max(minReadaheadBytes, prevAdaptiveWindow));
        const stepsAboveThreshold = streak - streakThreshold;
        const scaled = Math.floor(baseWindow * Math.pow(growthFactor, stepsAboveThreshold));
        readAheadBytes = Math.min(maxReadAheadBytes, Math.max(baseWindow, scaled));
      }
      adaptiveWindowBytes = readAheadBytes;
    } else {
      adaptiveWindowBytes = prevAdaptiveWindow;
    }
  }

  return {
    isSequential,
    isRandomSeek,
    streak,
    gapBytes,
    stride,
    readAheadBytes,
    adaptiveWindowBytes,
    nextState: {
      lastOffset: offset,
      lastLength: length,
      lastEnd: end,
      stride,
      streak,
      readAheadBytes,
      adaptiveWindowBytes,
    },
  };
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
    this.opfsCache = options.opfsCache || null;
    this.backgroundOpfsWrites = options.backgroundOpfsWrites !== false;
    this.maxPendingOpfsWrites =
      Number.isInteger(options.maxPendingOpfsWrites) && options.maxPendingOpfsWrites > 0
        ? options.maxPendingOpfsWrites
        : DEFAULT_MAX_PENDING_OPFS_WRITES;
    this.maxOpfsWriteConcurrency =
      Number.isInteger(options.maxOpfsWriteConcurrency) && options.maxOpfsWriteConcurrency > 0
        ? options.maxOpfsWriteConcurrency
        : DEFAULT_MAX_OPFS_WRITE_CONCURRENCY;

    this._explicitCoalesceMaxMergedBytes = options.coalesceMaxMergedBytes !== undefined;
    this.coalesceMaxGapBytes =
      Number.isFinite(options.coalesceMaxGapBytes) && options.coalesceMaxGapBytes >= 0
        ? Math.floor(options.coalesceMaxGapBytes)
        : DEFAULT_COALESCE_MAX_GAP_BYTES;
    this.coalesceMaxMergedBytes =
      Number.isFinite(options.coalesceMaxMergedBytes) && options.coalesceMaxMergedBytes > 0
        ? Math.floor(options.coalesceMaxMergedBytes)
        : DEFAULT_COALESCE_MAX_MERGED_BYTES;
    const rawMaxReadahead =
      options.maxReadaheadBytes !== undefined
        ? options.maxReadaheadBytes
        : options.maxReadAheadBytes;
    this.maxReadaheadBytes =
      Number.isFinite(rawMaxReadahead) && rawMaxReadahead >= 0
        ? Math.floor(rawMaxReadahead)
        : DEFAULT_MAX_READAHEAD_BYTES;
    this.maxReadAheadBytes = this.maxReadaheadBytes;
    const rawMinReadahead =
      options.minReadaheadBytes !== undefined
        ? options.minReadaheadBytes
        : options.minReadAheadBytes;
    const requestedMinReadahead =
      Number.isFinite(rawMinReadahead) && rawMinReadahead > 0
        ? Math.floor(rawMinReadahead)
        : DEFAULT_MIN_READAHEAD_BYTES;
    this.minReadaheadBytes =
      this.maxReadaheadBytes > 0 ? Math.min(requestedMinReadahead, this.maxReadaheadBytes) : 0;
    this.sequentialStreakThreshold =
      Number.isInteger(options.sequentialStreakThreshold) && options.sequentialStreakThreshold >= 1
        ? options.sequentialStreakThreshold
        : DEFAULT_SEQUENTIAL_STREAK_THRESHOLD;
    this.sequentialMaxGapBytes =
      Number.isFinite(options.sequentialMaxGapBytes) && options.sequentialMaxGapBytes >= 0
        ? Math.floor(options.sequentialMaxGapBytes)
        : Math.min(this.coalesceMaxGapBytes, this.chunkSize);

    const autoOptimize =
      this.chunkSize >= MIN_AUTO_OPTIMIZE_CHUNK_SIZE &&
      this.maxBytes >= MIN_AUTO_OPTIMIZE_CACHE_BYTES;

    if (typeof options.coalesceAdjacent === 'boolean') {
      this.coalesceAdjacent = options.coalesceAdjacent;
    } else if (
      options.coalesceMaxGapBytes !== undefined ||
      options.coalesceMaxMergedBytes !== undefined
    ) {
      this.coalesceAdjacent = true;
    } else {
      this.coalesceAdjacent = autoOptimize;
    }

    const explicitReadAhead =
      typeof options.enableReadAhead === 'boolean'
        ? options.enableReadAhead
        : typeof options.readAhead === 'boolean'
          ? options.readAhead
          : null;
    if (explicitReadAhead !== null) {
      this.enableReadAhead = explicitReadAhead && this.maxReadaheadBytes > 0;
    } else if (
      options.maxReadaheadBytes !== undefined ||
      options.maxReadAheadBytes !== undefined ||
      options.minReadaheadBytes !== undefined ||
      options.minReadAheadBytes !== undefined
    ) {
      this.enableReadAhead = this.maxReadaheadBytes > 0;
    } else {
      this.enableReadAhead = autoOptimize && this.maxReadaheadBytes > 0;
    }

    this._memCache = new Map();
    this._memChunkFingerprints = new Map();
    this._inFlight = new Map();
    this._opfsEntries = new Map();
    this._keyToOpfsFile = new Map();
    this._fileFingerprints = new Map();
    this._observedHttpMeta = new Map();
    this._opfsWriteQueue = new Map();
    this._activeOpfsWrites = new Set();
    this._opfsWriteWaiters = [];
    this._quotaEvictChain = Promise.resolve();
    this._expectedChunkLengths = new Map();
    this._fileSizes = new Map();
    this._seqTrackers = new Map();
    this._prefetchedKeys = new Set();
    this._readaheadKeys = new Map();
    this._inFlightPrefetchKeys = new Set();
    this._inFlightReadaheadKeys = new Set();
    this._activePrefetches = new Set();
    this._activeReadAheadBatches = 0;
    this._bytesCached = 0;
    this._stats = this._createEmptyStats();

    if (options.fileFingerprints instanceof Map) {
      for (const [p, fp] of options.fileFingerprints.entries()) {
        this.setFileFingerprint(p, fp);
      }
    } else if (options.fileFingerprints && typeof options.fileFingerprints === 'object') {
      for (const [p, fp] of Object.entries(options.fileFingerprints)) {
        this.setFileFingerprint(p, fp);
      }
    }
  }

  _createEmptyStats() {
    return {
      hits: 0,
      misses: 0,
      opfsHits: 0,
      opfsMisses: 0,
      opfsWrites: 0,
      opfsWriteErrors: 0,
      opfsStaleInvalidations: 0,
      evictions: 0,
      quotaEvictions: 0,
      quotaRetrySuccesses: 0,
      corruptEvictions: 0,
      coalescedRangeCount: 0,
      sequentialHits: 0,
      prefetchHits: 0,
      prefetchBytes: 0,
      bytesSavedByCoalescing: 0,
      adaptiveReadaheadWindowBytes: 0,
      sequentialStreakPeak: 0,
      readaheadHits: 0,
      readaheadWastedBytes: 0,
      zeroAllocCopyHits: 0,
    };
  }

  _normalizePathOrUrl(urlOrPath) {
    if (typeof urlOrPath !== 'string' || urlOrPath.length === 0) {
      throw new Error('RangeCache path must be a non-empty string');
    }
    let candidate = urlOrPath;
    if (candidate.startsWith(this.dataBase)) {
      candidate = candidate.slice(this.dataBase.length);
    } else if (/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(candidate)) {
      candidate = candidate.replace(/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\/[^/]*/, '');
      candidate = candidate.replace(/[?#].*$/, '');
      const basePath = this.dataBase.replace(/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\/[^/]*/, '');
      if (basePath && candidate.startsWith(basePath)) {
        candidate = candidate.slice(basePath.length);
      }
    }
    return normalizeRelPath(candidate);
  }

  _recordReadaheadWastedKey(key) {
    const wastedBytes = this._readaheadKeys.get(key);
    if (wastedBytes !== undefined) {
      this._readaheadKeys.delete(key);
      if (wastedBytes > 0) {
        this._stats.readaheadWastedBytes += wastedBytes;
      }
    }
  }

  _accountAbandonedReadaheadOnSeek(relPath, keepFirstChunk, keepLastChunk) {
    if (this._readaheadKeys.size === 0) {
      return;
    }
    const prefix = `${relPath}::`;
    for (const [key, byteLength] of Array.from(this._readaheadKeys.entries())) {
      if (!key.startsWith(prefix)) {
        continue;
      }
      const chunkIndex = Number(key.slice(prefix.length));
      if (
        Number.isInteger(chunkIndex) &&
        chunkIndex >= keepFirstChunk &&
        chunkIndex <= keepLastChunk
      ) {
        continue;
      }
      this._readaheadKeys.delete(key);
      if (byteLength > 0) {
        this._stats.readaheadWastedBytes += byteLength;
      }
    }
  }

  setOpfsDir(opfsDir) {
    const nextDir = opfsDir || null;
    if (nextDir !== this.opfsDir) {
      this._opfsEntries.clear();
      this._keyToOpfsFile.clear();
      this._opfsWriteQueue.clear();
    }
    this.opfsDir = nextDir;
  }

  setOpfsCache(opfsCache) {
    const nextCache = opfsCache || null;
    if (nextCache !== this.opfsCache) {
      this._opfsWriteQueue.clear();
    }
    this.opfsCache = nextCache;
  }

  setFileFingerprint(relPath, fingerprintOrMeta) {
    const cleanPath = normalizeRelPath(relPath);
    const nextFp = buildFileFingerprint(fingerprintOrMeta);
    const prevFp = this._fileFingerprints.get(cleanPath) || null;
    if (nextFp === null) {
      this._fileFingerprints.delete(cleanPath);
    } else {
      this._fileFingerprints.set(cleanPath, nextFp);
    }
    if (prevFp !== null && prevFp !== nextFp) {
      this._evictPathFromMemory(cleanPath);
    }
    return nextFp;
  }

  getFileFingerprint(relPath) {
    let cleanPath;
    try {
      cleanPath = normalizeRelPath(relPath);
    } catch {
      return null;
    }
    return this._fileFingerprints.get(cleanPath) || null;
  }

  clearFileFingerprint(relPath) {
    let cleanPath;
    try {
      cleanPath = normalizeRelPath(relPath);
    } catch {
      return false;
    }
    return this._fileFingerprints.delete(cleanPath);
  }

  _resolveEffectiveFingerprint(relPath, overrideFingerprint = undefined) {
    if (overrideFingerprint !== undefined && overrideFingerprint !== null) {
      return buildFileFingerprint(overrideFingerprint);
    }
    return this._fileFingerprints.get(relPath) || null;
  }

  _evictPathFromMemory(relPath) {
    const prefix = `${relPath}::`;
    for (const [key, bytes] of Array.from(this._memCache.entries())) {
      if (key.startsWith(prefix)) {
        this._memCache.delete(key);
        this._memChunkFingerprints.delete(key);
        this._prefetchedKeys.delete(key);
        this._recordReadaheadWastedKey(key);
        this._opfsWriteQueue.delete(key);
        if (bytes && Number.isFinite(bytes.byteLength)) {
          this._bytesCached = Math.max(0, this._bytesCached - bytes.byteLength);
        }
      }
    }
    for (const key of Array.from(this._expectedChunkLengths.keys())) {
      if (key.startsWith(prefix)) {
        this._expectedChunkLengths.delete(key);
      }
    }
    this._fileSizes.delete(relPath);
  }

  _recordObservedHttpMeta(relPath, totalSize, etag) {
    const validSize = Number.isSafeInteger(totalSize) && totalSize >= 0 ? totalSize : null;
    const validTag = typeof etag === 'string' && etag.trim().length > 0 ? etag.trim() : null;
    if (validSize === null && validTag === null) {
      return;
    }
    const prev = this._observedHttpMeta.get(relPath) || null;
    if (prev) {
      const sizeChanged = validSize !== null && prev.totalSize !== null && validSize !== prev.totalSize;
      const etagChanged = validTag !== null && prev.etag !== null && validTag !== prev.etag;
      if (sizeChanged || etagChanged) {
        this._evictPathFromMemory(relPath);
        if (this._fileFingerprints.has(relPath)) {
          const updatedFp = buildFileFingerprint(
            validSize !== null ? validSize : prev.totalSize,
            validTag !== null ? validTag : prev.etag,
          );
          if (updatedFp) {
            this._fileFingerprints.set(relPath, updatedFp);
          }
        }
      }
    }
    this._observedHttpMeta.set(relPath, {
      totalSize: validSize !== null ? validSize : prev ? prev.totalSize : null,
      etag: validTag !== null ? validTag : prev ? prev.etag : null,
    });
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

  _putMemory(key, bytes, fileFingerprint = null) {
    if (!(bytes instanceof Uint8Array)) return;
    if (this._memCache.has(key)) {
      const existing = this._memCache.get(key);
      if (existing && Number.isFinite(existing.byteLength)) {
        this._bytesCached = Math.max(0, this._bytesCached - existing.byteLength);
      }
      this._memCache.delete(key);
      this._memChunkFingerprints.delete(key);
    }

    if (this.maxBytes <= 0 || bytes.byteLength > this.maxBytes) {
      return;
    }

    while (this._bytesCached + bytes.byteLength > this.maxBytes && this._memCache.size > 0) {
      const oldestKey = this._memCache.keys().next().value;
      const oldestBytes = this._memCache.get(oldestKey);
      this._memCache.delete(oldestKey);
      this._memChunkFingerprints.delete(oldestKey);
      this._prefetchedKeys.delete(oldestKey);
      this._recordReadaheadWastedKey(oldestKey);
      if (oldestBytes && Number.isFinite(oldestBytes.byteLength)) {
        this._bytesCached = Math.max(0, this._bytesCached - oldestBytes.byteLength);
      }
      this._stats.evictions += 1;
    }

    this._memCache.set(key, bytes);
    if (fileFingerprint !== null && fileFingerprint !== undefined) {
      this._memChunkFingerprints.set(key, fileFingerprint);
    } else {
      this._memChunkFingerprints.delete(key);
    }
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

  async _listOpfsEntryNames() {
    if (!this.opfsDir) return [];
    const names = [];
    try {
      if (typeof this.opfsDir.keys === 'function') {
        for await (const name of this.opfsDir.keys()) {
          if (typeof name === 'string') {
            names.push(name);
          }
        }
        return names;
      }
      if (typeof this.opfsDir.entries === 'function') {
        for await (const pair of this.opfsDir.entries()) {
          if (Array.isArray(pair) && typeof pair[0] === 'string') {
            names.push(pair[0]);
          }
        }
        return names;
      }
    } catch {
      // Ignore directory enumeration failures.
    }
    return names;
  }

  async _invalidateStaleOpfsChunkEntries(relPath, chunkIndex, activeFileName) {
    if (!this.opfsDir) return 0;
    const prefix = opfsChunkFilePrefix(relPath, chunkIndex, this.chunkSize);
    const legacyName = `${prefix}.bin`;
    const fpPrefix = `${prefix}_fp_`;
    const isStaleMatch = (name) =>
      typeof name === 'string' &&
      name !== activeFileName &&
      (name === legacyName || (name.startsWith(fpPrefix) && name.endsWith('.bin')));

    const staleNames = new Set();
    for (const fileName of this._opfsEntries.keys()) {
      if (isStaleMatch(fileName)) {
        staleNames.add(fileName);
      }
    }

    const dirNames = await this._listOpfsEntryNames();
    for (let i = 0; i < dirNames.length; i += 1) {
      if (isStaleMatch(dirNames[i])) {
        staleNames.add(dirNames[i]);
      }
    }

    // Fallback for minimal KV stores lacking keys()/entries(): probe legacy un-fingerprinted key.
    if (
      staleNames.size === 0 &&
      dirNames.length === 0 &&
      legacyName !== activeFileName &&
      typeof this.opfsDir.get === 'function'
    ) {
      try {
        const legacyVal = await this.opfsDir.get(legacyName);
        if (legacyVal !== undefined && legacyVal !== null) {
          staleNames.add(legacyName);
        }
      } catch {
        // Ignore probe failures.
      }
    }

    let invalidated = 0;
    const key = this._chunkKey(relPath, chunkIndex);
    for (const staleName of staleNames) {
      this._opfsEntries.delete(staleName);
      if (this._keyToOpfsFile.get(key) === staleName) {
        this._keyToOpfsFile.delete(key);
      }
      const deleted = await this._deleteOpfsFile(staleName);
      if (deleted) {
        invalidated += 1;
      }
    }

    if (invalidated > 0) {
      this._stats.opfsStaleInvalidations += invalidated;
    }
    return invalidated;
  }

  async _evictCorruptEntry(key, relPath, chunkIndex, countEviction = true, fileFingerprint = null) {
    if (this._memCache.has(key)) {
      const existing = this._memCache.get(key);
      this._memCache.delete(key);
      if (existing && Number.isFinite(existing.byteLength)) {
        this._bytesCached = Math.max(0, this._bytesCached - existing.byteLength);
      }
    }
    this._memChunkFingerprints.delete(key);
    this._prefetchedKeys.delete(key);
    this._recordReadaheadWastedKey(key);
    this._opfsWriteQueue.delete(key);
    const fp = this._resolveEffectiveFingerprint(relPath, fileFingerprint);
    if (this.opfsCache && typeof this.opfsCache.delete === 'function') {
      try {
        await this.opfsCache.delete(opfsChunkFileName(relPath, chunkIndex, this.chunkSize));
      } catch {
        // Ignore backing store delete errors.
      }
    }
    const fileName = opfsChunkFileName(relPath, chunkIndex, this.chunkSize, fp);
    this._opfsEntries.delete(fileName);
    this._keyToOpfsFile.delete(key);
    await this._deleteOpfsFile(fileName);
    if (countEviction) {
      this._stats.corruptEvictions += 1;
    }
  }

  async _evictForQuotaSerialized(excludeKey, excludeFileName, targetBytes) {
    const run = async () => this._evictForQuota(excludeKey, excludeFileName, targetBytes);
    const next = this._quotaEvictChain.then(run, run);
    this._quotaEvictChain = next.then(
      () => {},
      () => {},
    );
    return next;
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

  async _readOpfsDetailed(relPath, chunkIndex, fileFingerprint = null) {
    if (!this.opfsDir && !this.opfsCache) {
      return { bytes: null, corrupt: false, staleInvalidated: false, checkedOpfs: false, fileName: null };
    }
    const fp = this._resolveEffectiveFingerprint(relPath, fileFingerprint);

    if (this.opfsCache && typeof this.opfsCache.getEntry === 'function') {
      const baseKey = opfsChunkFileName(relPath, chunkIndex, this.chunkSize);
      try {
        const entry = await this.opfsCache.getEntry(baseKey, fp ? { fileFingerprint: fp } : {});
        if (!entry || entry.supported === false) {
          return { bytes: null, corrupt: false, staleInvalidated: false, checkedOpfs: false, fileName: baseKey };
        }
        if (entry.hit && entry.data instanceof Uint8Array) {
          return { bytes: entry.data, corrupt: false, staleInvalidated: false, checkedOpfs: true, fileName: baseKey };
        }
        if (entry.invalidated) {
          this._stats.opfsStaleInvalidations += 1;
          return { bytes: null, corrupt: false, staleInvalidated: true, checkedOpfs: true, fileName: baseKey };
        }
        return { bytes: null, corrupt: false, staleInvalidated: false, checkedOpfs: true, fileName: baseKey };
      } catch {
        return { bytes: null, corrupt: true, staleInvalidated: false, checkedOpfs: true, fileName: baseKey };
      }
    }

    const fileName = opfsChunkFileName(relPath, chunkIndex, this.chunkSize, fp);
    if (typeof this.opfsDir.getFileHandle === 'function') {
      let handle;
      try {
        handle = await this.opfsDir.getFileHandle(fileName, { create: false });
      } catch (err) {
        if (isNotFoundError(err)) {
          let staleInvalidated = false;
          if (fp !== null) {
            staleInvalidated = (await this._invalidateStaleOpfsChunkEntries(relPath, chunkIndex, fileName)) > 0;
          }
          return { bytes: null, corrupt: false, staleInvalidated, checkedOpfs: true, fileName };
        }
        return { bytes: null, corrupt: true, staleInvalidated: false, checkedOpfs: true, fileName };
      }
      try {
        const file = await handle.getFile();
        const buf = await file.arrayBuffer();
        return { bytes: new Uint8Array(buf), corrupt: false, staleInvalidated: false, checkedOpfs: true, fileName };
      } catch {
        return { bytes: null, corrupt: true, staleInvalidated: false, checkedOpfs: true, fileName };
      }
    }
    if (typeof this.opfsDir.get === 'function') {
      try {
        const val = await this.opfsDir.get(fileName);
        if (val === undefined || val === null) {
          let staleInvalidated = false;
          if (fp !== null) {
            staleInvalidated = (await this._invalidateStaleOpfsChunkEntries(relPath, chunkIndex, fileName)) > 0;
          }
          return { bytes: null, corrupt: false, staleInvalidated, checkedOpfs: true, fileName };
        }
        if (val instanceof Uint8Array) {
          return { bytes: val.slice(), corrupt: false, staleInvalidated: false, checkedOpfs: true, fileName };
        }
        if (val instanceof ArrayBuffer) {
          return { bytes: new Uint8Array(val.slice(0)), corrupt: false, staleInvalidated: false, checkedOpfs: true, fileName };
        }
        return { bytes: null, corrupt: true, staleInvalidated: false, checkedOpfs: true, fileName };
      } catch (err) {
        if (isNotFoundError(err)) {
          let staleInvalidated = false;
          if (fp !== null) {
            staleInvalidated = (await this._invalidateStaleOpfsChunkEntries(relPath, chunkIndex, fileName)) > 0;
          }
          return { bytes: null, corrupt: false, staleInvalidated, checkedOpfs: true, fileName };
        }
        return { bytes: null, corrupt: true, staleInvalidated: false, checkedOpfs: true, fileName };
      }
    }
    return { bytes: null, corrupt: false, staleInvalidated: false, checkedOpfs: false, fileName };
  }

  async _readOpfs(relPath, chunkIndex, fileFingerprint = null) {
    const res = await this._readOpfsDetailed(relPath, chunkIndex, fileFingerprint);
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

  async _writeOpfs(relPath, chunkIndex, bytes, fileFingerprint = null) {
    if ((!this.opfsDir && !this.opfsCache) || !(bytes instanceof Uint8Array) || bytes.byteLength === 0) {
      return false;
    }
    const key = this._chunkKey(relPath, chunkIndex);
    const fp = this._resolveEffectiveFingerprint(relPath, fileFingerprint);

    if (this.opfsCache && typeof this.opfsCache.put === 'function') {
      const baseKey = opfsChunkFileName(relPath, chunkIndex, this.chunkSize);
      try {
        const res = await this.opfsCache.put(baseKey, bytes, fp ? { fileFingerprint: fp } : {});
        if (res && res.stored) {
          this._stats.opfsWrites += 1;
          return true;
        }
        this._stats.opfsWriteErrors += 1;
        return false;
      } catch {
        this._stats.opfsWriteErrors += 1;
        return false;
      }
    }

    const fileName = opfsChunkFileName(relPath, chunkIndex, this.chunkSize, fp);

    try {
      const wrote = await this._writeOpfsOnce(fileName, bytes);
      if (wrote) {
        this._recordOpfsEntry(fileName, key, relPath, chunkIndex, bytes.byteLength);
        this._stats.opfsWrites += 1;
      } else {
        this._stats.opfsWriteErrors += 1;
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
        this._stats.opfsWriteErrors += 1;
        return false;
      }

      // QuotaExceededError: evict oldest entries until freeing at least
      // max(entryBytes, ceil(maxBytes * 0.25)) (or all evictable entries), then retry ONCE.
      const targetBytes = Math.max(bytes.byteLength, Math.ceil(this.maxBytes * 0.25));
      await this._evictForQuotaSerialized(key, fileName, targetBytes);

      try {
        const retryWrote = await this._writeOpfsOnce(fileName, bytes);
        if (retryWrote) {
          this._recordOpfsEntry(fileName, key, relPath, chunkIndex, bytes.byteLength);
          this._stats.quotaRetrySuccesses += 1;
          this._stats.opfsWrites += 1;
        } else {
          this._stats.opfsWriteErrors += 1;
        }
        return retryWrote;
      } catch {
        // Quota remains exhausted: clean up any partial file and degrade gracefully.
        this._opfsEntries.delete(fileName);
        if (this._keyToOpfsFile.get(key) === fileName) {
          this._keyToOpfsFile.delete(key);
        }
        await this._deleteOpfsFile(fileName);
        this._stats.opfsWriteErrors += 1;
        return false;
      }
    }
  }

  _enqueueOpfsWrite(relPath, chunkIndex, bytes, fileFingerprint = null) {
    if ((!this.opfsDir && !this.opfsCache) || !(bytes instanceof Uint8Array) || bytes.byteLength === 0) {
      return;
    }
    const key = this._chunkKey(relPath, chunkIndex);
    const fp = this._resolveEffectiveFingerprint(relPath, fileFingerprint);
    if (this._opfsWriteQueue.has(key)) {
      this._opfsWriteQueue.delete(key);
    } else {
      while (this._opfsWriteQueue.size >= this.maxPendingOpfsWrites) {
        const oldestKey = this._opfsWriteQueue.keys().next().value;
        if (oldestKey === undefined) break;
        this._opfsWriteQueue.delete(oldestKey);
      }
    }
    this._opfsWriteQueue.set(key, { relPath, chunkIndex, bytes, fileFingerprint: fp });
    this._pumpOpfsWriteQueue();
  }

  _pumpOpfsWriteQueue() {
    while (
      this._activeOpfsWrites.size < this.maxOpfsWriteConcurrency &&
      this._opfsWriteQueue.size > 0
    ) {
      const nextEntry = this._opfsWriteQueue.entries().next().value;
      if (!nextEntry) break;
      const [key, task] = nextEntry;
      this._opfsWriteQueue.delete(key);

      const writePromise = (async () => {
        try {
          await this._writeOpfs(task.relPath, task.chunkIndex, task.bytes, task.fileFingerprint);
        } catch {
          this._stats.opfsWriteErrors += 1;
        }
      })();

      this._activeOpfsWrites.add(writePromise);
      writePromise.finally(() => {
        this._activeOpfsWrites.delete(writePromise);
        this._pumpOpfsWriteQueue();
        if (this._activeOpfsWrites.size === 0 && this._opfsWriteQueue.size === 0) {
          const waiters = this._opfsWriteWaiters.splice(0, this._opfsWriteWaiters.length);
          for (let i = 0; i < waiters.length; i += 1) {
            waiters[i]();
          }
        }
      });
    }
  }

  async flushOpfsWrites() {
    while (this._activeOpfsWrites.size > 0 || this._opfsWriteQueue.size > 0) {
      this._pumpOpfsWriteQueue();
      await new Promise((resolve) => {
        if (this._activeOpfsWrites.size === 0 && this._opfsWriteQueue.size === 0) {
          resolve();
          return;
        }
        this._opfsWriteWaiters.push(resolve);
      });
    }
    if (this.opfsCache && typeof this.opfsCache.flushBackgroundWrites === 'function') {
      await this.opfsCache.flushBackgroundWrites();
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
      const contentRangeHeader =
        res.headers && typeof res.headers.get === 'function'
          ? res.headers.get('Content-Range') || res.headers.get('content-range')
          : null;
      const total = parseUnsatisfiedContentRangeTotal(contentRangeHeader);
      if (total !== null) {
        this._recordFileSize(relPath, total);
      }
      return new Uint8Array(0);
    }

    if (!res.ok && res.status !== 206) {
      throw new Error(`HTTP ${res.status} reading ${relPath}[${chunkStart}..${chunkEnd}]`);
    }

    const raw = new Uint8Array(await res.arrayBuffer());
    const etagHeader = readResponseHeader(res.headers, 'ETag');

    if (res.status === 206) {
      const contentRangeHeader = readResponseHeader(res.headers, 'Content-Range');
      const parsed = parseContentRange(contentRangeHeader);
      if (parsed) {
        if (Number.isSafeInteger(parsed.total) && parsed.total >= 0) {
          this._recordObservedHttpMeta(relPath, parsed.total, etagHeader);
          this._recordFileSize(relPath, parsed.total);
        } else if (etagHeader) {
          this._recordObservedHttpMeta(relPath, null, etagHeader);
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
      if (etagHeader) {
        this._recordObservedHttpMeta(relPath, null, etagHeader);
      }
      if (raw.byteLength <= this.chunkSize) {
        return raw;
      }
      return raw.slice(0, this.chunkSize);
    }

    // 200 OK fallback (server returned full resource instead of 206 Partial Content).
    this._recordObservedHttpMeta(relPath, raw.byteLength, etagHeader);
    this._recordFileSize(relPath, raw.byteLength);
    if (chunkStart >= raw.byteLength) {
      return new Uint8Array(0);
    }
    return raw.slice(chunkStart, Math.min(raw.byteLength, chunkStart + this.chunkSize));
  }

  async _fetchChunkRunFromNetwork(relPath, startChunk, endChunk) {
    if (typeof this.fetchImpl !== 'function') {
      throw new Error('RangeCache has no fetch implementation available');
    }

    const resultMap = new Map();
    const uncovered = [];
    const runStart = startChunk * this.chunkSize;
    const knownTotalBefore = this._fileSizes.has(relPath) ? this._fileSizes.get(relPath) : null;

    if (knownTotalBefore !== null && runStart >= knownTotalBefore) {
      for (let idx = startChunk; idx <= endChunk; idx += 1) {
        resultMap.set(idx, new Uint8Array(0));
      }
      return { chunks: resultMap, uncovered };
    }

    const naturalEnd = (endChunk + 1) * this.chunkSize - 1;
    const reqEnd =
      knownTotalBefore !== null && knownTotalBefore > 0
        ? Math.min(naturalEnd, knownTotalBefore - 1)
        : naturalEnd;
    const url = `${this.dataBase}${relPath}`;

    const res = await this.fetchImpl(url, {
      headers: { Range: `bytes=${runStart}-${reqEnd}` },
    });

    if (res.status === 416) {
      const contentRangeHeader =
        res.headers && typeof res.headers.get === 'function'
          ? res.headers.get('Content-Range') || res.headers.get('content-range')
          : null;
      const total = parseUnsatisfiedContentRangeTotal(contentRangeHeader);
      if (total !== null) {
        this._recordFileSize(relPath, total);
      }
      for (let idx = startChunk; idx <= endChunk; idx += 1) {
        resultMap.set(idx, new Uint8Array(0));
      }
      return { chunks: resultMap, uncovered };
    }

    if (!res.ok && res.status !== 206) {
      throw new Error(`HTTP ${res.status} reading ${relPath}[${runStart}..${reqEnd}]`);
    }

    const raw = new Uint8Array(await res.arrayBuffer());
    const etagHeader = readResponseHeader(res.headers, 'ETag');

    let actualStart = runStart;
    let actualEnd = raw.byteLength > 0 ? runStart + raw.byteLength - 1 : runStart - 1;
    let knownTotal = knownTotalBefore;
    let reachedEof = false;

    if (res.status === 206) {
      const contentRangeHeader = readResponseHeader(res.headers, 'Content-Range');
      const parsed = parseContentRange(contentRangeHeader);
      if (parsed) {
        actualStart = parsed.start;
        const headerSpan = parsed.end - parsed.start + 1;
        const usableLen = Math.min(raw.byteLength, headerSpan);
        actualEnd = usableLen > 0 ? actualStart + usableLen - 1 : actualStart - 1;
        if (Number.isSafeInteger(parsed.total) && parsed.total >= 0) {
          this._recordObservedHttpMeta(relPath, parsed.total, etagHeader);
          this._recordFileSize(relPath, parsed.total);
          knownTotal = parsed.total;
          if (actualEnd + 1 >= parsed.total) {
            reachedEof = true;
          }
        } else if (etagHeader) {
          this._recordObservedHttpMeta(relPath, null, etagHeader);
        }
      } else if (etagHeader) {
        this._recordObservedHttpMeta(relPath, null, etagHeader);
      }
    } else {
      // 200 OK fallback (server returned full resource instead of 206 Partial Content).
      this._recordObservedHttpMeta(relPath, raw.byteLength, etagHeader);
      this._recordFileSize(relPath, raw.byteLength);
      knownTotal = raw.byteLength;
      actualStart = 0;
      actualEnd = raw.byteLength - 1;
      reachedEof = true;
    }

    for (let idx = startChunk; idx <= endChunk; idx += 1) {
      const cStart = idx * this.chunkSize;
      const cMaxEnd = cStart + this.chunkSize - 1;

      if (knownTotal !== null && cStart >= knownTotal) {
        resultMap.set(idx, new Uint8Array(0));
        continue;
      }

      const expectedEnd = knownTotal !== null ? Math.min(cMaxEnd, knownTotal - 1) : cMaxEnd;
      const startsInResponse =
        raw.byteLength > 0 && cStart >= actualStart && cStart <= actualEnd;

      if (startsInResponse) {
        const relStart = cStart - actualStart;
        const availFromChunkStart = actualEnd - cStart + 1;
        const neededLen = expectedEnd - cStart + 1;
        const coversCompleteChunk =
          actualEnd >= expectedEnd || (reachedEof && availFromChunkStart > 0);
        if (coversCompleteChunk) {
          const takeLen = Math.min(this.chunkSize, neededLen, availFromChunkStart);
          resultMap.set(idx, raw.slice(relStart, relStart + takeLen));
          continue;
        }
      } else if (reachedEof && cStart > actualEnd) {
        resultMap.set(idx, new Uint8Array(0));
        continue;
      }

      uncovered.push(idx);
    }

    return { chunks: resultMap, uncovered };
  }

  _consumePrefetchHit(key) {
    const wasPrefetch = this._prefetchedKeys.delete(key);
    if (wasPrefetch) {
      this._stats.prefetchHits += 1;
    }
    if (this._readaheadKeys.delete(key)) {
      this._stats.readaheadHits += 1;
    }
    return wasPrefetch;
  }

  _updateSequentialTracker(relPath, offset, length) {
    const prevState = this._seqTrackers.get(relPath) || null;
    const effectiveMaxReadahead = this.enableReadAhead ? this.maxReadaheadBytes : 0;
    const effectiveMinReadahead = this.enableReadAhead ? this.minReadaheadBytes : 0;
    const detection = detectSequentialRead(prevState, offset, length, {
      chunkSize: this.chunkSize,
      maxGapBytes: this.sequentialMaxGapBytes,
      minReadaheadBytes: effectiveMinReadahead,
      maxReadaheadBytes: effectiveMaxReadahead,
      streakThreshold: this.sequentialStreakThreshold,
    });

    if (this._seqTrackers.has(relPath)) {
      this._seqTrackers.delete(relPath);
    } else if (this._seqTrackers.size >= MAX_TRACKED_SEQUENTIAL_KEYS) {
      const oldest = this._seqTrackers.keys().next().value;
      if (oldest !== undefined) {
        this._seqTrackers.delete(oldest);
      }
    }
    this._seqTrackers.set(relPath, detection.nextState);

    if (detection.isSequential) {
      this._stats.sequentialHits += 1;
      if (detection.streak > this._stats.sequentialStreakPeak) {
        this._stats.sequentialStreakPeak = detection.streak;
      }
    } else if (detection.isRandomSeek && length > 0) {
      const firstChunk = Math.floor(offset / this.chunkSize);
      const lastChunk = Math.floor((offset + length - 1) / this.chunkSize);
      this._accountAbandonedReadaheadOnSeek(relPath, firstChunk, lastChunk);
    }

    if (this.enableReadAhead) {
      this._stats.adaptiveReadaheadWindowBytes = detection.adaptiveWindowBytes;
    }

    return detection;
  }

  getSequentialState(relPath) {
    let cleanPath;
    try {
      cleanPath = this._normalizePathOrUrl(relPath);
    } catch {
      return null;
    }
    const state = this._seqTrackers.get(cleanPath);
    return state ? { ...state } : null;
  }

  async _getChunk(relPath, chunkIndex, requiredEndInChunk = 0, options = {}) {
    const isPrefetch = Boolean(options.isPrefetch);
    const isReadahead = Boolean(options.isReadahead);
    const countMiss = options.countMiss !== false;
    const fp = this._resolveEffectiveFingerprint(relPath, options.fileFingerprint);
    const key = this._chunkKey(relPath, chunkIndex);
    let suspectByteLength = null;

    const memHit = this._touchMemory(key);
    if (memHit !== null) {
      const memFp = this._memChunkFingerprints.get(key) || null;
      if (fp !== null && memFp !== fp) {
        this._memCache.delete(key);
        this._memChunkFingerprints.delete(key);
        this._prefetchedKeys.delete(key);
        this._recordReadaheadWastedKey(key);
        if (Number.isFinite(memHit.byteLength)) {
          this._bytesCached = Math.max(0, this._bytesCached - memHit.byteLength);
        }
      } else {
        const verdict = this._validateCachedChunk(key, relPath, chunkIndex, memHit, requiredEndInChunk);
        if (verdict === 'valid') {
          if (!isPrefetch) {
            this._stats.hits += 1;
            this._consumePrefetchHit(key);
          }
          return memHit;
        }
        if (verdict === 'corrupt') {
          await this._evictCorruptEntry(key, relPath, chunkIndex, true, fp);
        } else {
          suspectByteLength = memHit.byteLength;
          await this._evictCorruptEntry(key, relPath, chunkIndex, false, fp);
        }
      }
    }

    const existingFlight = this._inFlight.get(key);
    if (existingFlight) {
      const joinedPrefetch = !isPrefetch && this._inFlightPrefetchKeys.delete(key);
      const joinedReadahead = !isPrefetch && this._inFlightReadaheadKeys.delete(key);
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
        if (joinedPrefetch) {
          this._prefetchedKeys.delete(key);
          this._stats.prefetchHits += 1;
        }
        if (joinedReadahead) {
          this._readaheadKeys.delete(key);
          this._stats.readaheadHits += 1;
        }
        return inflightBytes;
      }
      if (verdict === 'corrupt') {
        await this._evictCorruptEntry(key, relPath, chunkIndex, true, fp);
      } else {
        suspectByteLength = inflightBytes.byteLength;
        await this._evictCorruptEntry(key, relPath, chunkIndex, false, fp);
      }
    }

    if (isPrefetch) {
      this._inFlightPrefetchKeys.add(key);
      if (isReadahead) {
        this._inFlightReadaheadKeys.add(key);
      }
    }

    const loadPromise = (async () => {
      if (suspectByteLength === null) {
        const opfsResult = await this._readOpfsDetailed(relPath, chunkIndex, fp);
        if (opfsResult.corrupt) {
          if (!isPrefetch && opfsResult.checkedOpfs) {
            this._stats.opfsMisses += 1;
          }
          await this._evictCorruptEntry(key, relPath, chunkIndex, true, fp);
        } else if (opfsResult.bytes !== null) {
          const verdict = this._validateCachedChunk(
            key,
            relPath,
            chunkIndex,
            opfsResult.bytes,
            requiredEndInChunk,
          );
          if (verdict === 'valid') {
            const fileName = opfsResult.fileName || opfsChunkFileName(relPath, chunkIndex, this.chunkSize, fp);
            this._recordOpfsEntry(fileName, key, relPath, chunkIndex, opfsResult.bytes.byteLength);
            if (!isPrefetch) {
              this._stats.opfsHits += 1;
              this._stats.hits += 1;
            }
            this._putMemory(key, opfsResult.bytes, fp);
            if (isPrefetch && this._inFlightPrefetchKeys.has(key) && this._memCache.has(key)) {
              this._prefetchedKeys.add(key);
              if (isReadahead && this._inFlightReadaheadKeys.has(key)) {
                this._readaheadKeys.set(key, opfsResult.bytes.byteLength);
              }
            }
            return opfsResult.bytes;
          }
          if (!isPrefetch && opfsResult.checkedOpfs) {
            this._stats.opfsMisses += 1;
          }
          if (verdict === 'corrupt') {
            await this._evictCorruptEntry(key, relPath, chunkIndex, true, fp);
          } else {
            suspectByteLength = opfsResult.bytes.byteLength;
            await this._evictCorruptEntry(key, relPath, chunkIndex, false, fp);
          }
        } else if (!isPrefetch && opfsResult.checkedOpfs) {
          this._stats.opfsMisses += 1;
        }
      }

      if (countMiss) {
        this._stats.misses += 1;
      }
      const netBytes = await this._fetchChunkFromNetwork(relPath, chunkIndex);
      if (suspectByteLength !== null && netBytes.byteLength !== suspectByteLength) {
        this._stats.corruptEvictions += 1;
      }
      if (netBytes.byteLength > 0) {
        const effectiveFp = this._resolveEffectiveFingerprint(relPath, options.fileFingerprint);
        this._recordExpectedChunkLength(key, netBytes.byteLength);
        this._putMemory(key, netBytes, effectiveFp);
        if (isPrefetch) {
          this._stats.prefetchBytes += netBytes.byteLength;
          if (this._inFlightPrefetchKeys.has(key) && this._memCache.has(key)) {
            this._prefetchedKeys.add(key);
            if (isReadahead && this._inFlightReadaheadKeys.has(key)) {
              this._readaheadKeys.set(key, netBytes.byteLength);
            }
          }
        }
        if (this.backgroundOpfsWrites) {
          this._enqueueOpfsWrite(relPath, chunkIndex, netBytes, effectiveFp);
        } else {
          await this._writeOpfs(relPath, chunkIndex, netBytes, effectiveFp);
        }
      }
      return netBytes;
    })();

    this._inFlight.set(key, loadPromise);
    try {
      return await loadPromise;
    } finally {
      this._inFlightPrefetchKeys.delete(key);
      this._inFlightReadaheadKeys.delete(key);
      if (this._inFlight.get(key) === loadPromise) {
        this._inFlight.delete(key);
      }
    }
  }

  async _getChunksCoalesced(relPath, chunkSpecs, options = {}) {
    const isPrefetch = Boolean(options.isPrefetch);
    const isReadahead = Boolean(options.isReadahead);
    const countMiss = options.countMiss !== false;
    const countCoalesce = options.countCoalesce !== false;
    const fp = this._resolveEffectiveFingerprint(relPath, options.fileFingerprint);
    const maxMergedBytes =
      Number.isFinite(options.maxMergedBytes) && options.maxMergedBytes > 0
        ? Math.floor(options.maxMergedBytes)
        : this.coalesceMaxMergedBytes;

    const total = chunkSpecs.length;
    if (total === 0) {
      return [];
    }

    const results = new Array(total);
    const unresolved = [];

    // 1. Resolve memory hits and concurrent in-flight promises.
    for (let i = 0; i < total; i += 1) {
      const spec = chunkSpecs[i];
      const key = this._chunkKey(relPath, spec.chunkIndex);
      let suspectByteLength = null;

      const memHit = this._touchMemory(key);
      if (memHit !== null) {
        const memFp = this._memChunkFingerprints.get(key) || null;
        if (fp !== null && memFp !== fp) {
          this._memCache.delete(key);
          this._memChunkFingerprints.delete(key);
          this._prefetchedKeys.delete(key);
          this._recordReadaheadWastedKey(key);
          if (Number.isFinite(memHit.byteLength)) {
            this._bytesCached = Math.max(0, this._bytesCached - memHit.byteLength);
          }
        } else {
          const verdict = this._validateCachedChunk(
            key,
            relPath,
            spec.chunkIndex,
            memHit,
            spec.requiredEndInChunk,
          );
          if (verdict === 'valid') {
            if (!isPrefetch) {
              this._stats.hits += 1;
              this._consumePrefetchHit(key);
            }
            results[i] = memHit;
            continue;
          }
          if (verdict === 'corrupt') {
            await this._evictCorruptEntry(key, relPath, spec.chunkIndex, true, fp);
          } else {
            suspectByteLength = memHit.byteLength;
            await this._evictCorruptEntry(key, relPath, spec.chunkIndex, false, fp);
          }
        }
      }

      const existingFlight = this._inFlight.get(key);
      if (existingFlight) {
        unresolved.push({
          pos: i,
          spec,
          key,
          suspectByteLength,
          existingFlight,
          joinedPrefetch: !isPrefetch && this._inFlightPrefetchKeys.delete(key),
          joinedReadahead: !isPrefetch && this._inFlightReadaheadKeys.delete(key),
        });
        continue;
      }

      unresolved.push({
        pos: i,
        spec,
        key,
        suspectByteLength,
        existingFlight: null,
        joinedPrefetch: false,
        joinedReadahead: false,
      });
    }

    // 2. Await in-flight joins and check OPFS for non-in-flight entries.
    const missing = [];
    await Promise.all(
      unresolved.map(async (item) => {
        if (item.existingFlight) {
          const inflightBytes = await item.existingFlight;
          if (inflightBytes instanceof Uint8Array && inflightBytes.byteLength === 0) {
            results[item.pos] = inflightBytes;
            return;
          }
          const verdict = this._validateCachedChunk(
            item.key,
            relPath,
            item.spec.chunkIndex,
            inflightBytes,
            item.spec.requiredEndInChunk,
          );
          if (verdict === 'valid') {
            if (item.joinedPrefetch) {
              this._prefetchedKeys.delete(item.key);
              this._stats.prefetchHits += 1;
            }
            if (item.joinedReadahead) {
              this._readaheadKeys.delete(item.key);
              this._stats.readaheadHits += 1;
            }
            results[item.pos] = inflightBytes;
            return;
          }
          if (verdict === 'corrupt') {
            await this._evictCorruptEntry(item.key, relPath, item.spec.chunkIndex, true, fp);
          } else {
            item.suspectByteLength = inflightBytes.byteLength;
            await this._evictCorruptEntry(item.key, relPath, item.spec.chunkIndex, false, fp);
          }
        }

        if (item.suspectByteLength === null && (this.opfsDir || this.opfsCache)) {
          const opfsResult = await this._readOpfsDetailed(relPath, item.spec.chunkIndex, fp);
          if (opfsResult.corrupt) {
            if (!isPrefetch && opfsResult.checkedOpfs) {
              this._stats.opfsMisses += 1;
            }
            await this._evictCorruptEntry(item.key, relPath, item.spec.chunkIndex, true, fp);
          } else if (opfsResult.bytes !== null) {
            const verdict = this._validateCachedChunk(
              item.key,
              relPath,
              item.spec.chunkIndex,
              opfsResult.bytes,
              item.spec.requiredEndInChunk,
            );
            if (verdict === 'valid') {
              const fileName =
                opfsResult.fileName || opfsChunkFileName(relPath, item.spec.chunkIndex, this.chunkSize, fp);
              this._recordOpfsEntry(
                fileName,
                item.key,
                relPath,
                item.spec.chunkIndex,
                opfsResult.bytes.byteLength,
              );
              if (!isPrefetch) {
                this._stats.opfsHits += 1;
                this._stats.hits += 1;
              }
              this._putMemory(item.key, opfsResult.bytes, fp);
              if (isPrefetch && this._memCache.has(item.key)) {
                this._prefetchedKeys.add(item.key);
                if (isReadahead) {
                  this._readaheadKeys.set(item.key, opfsResult.bytes.byteLength);
                }
              }
              results[item.pos] = opfsResult.bytes;
              return;
            }
            if (!isPrefetch && opfsResult.checkedOpfs) {
              this._stats.opfsMisses += 1;
            }
            if (verdict === 'corrupt') {
              await this._evictCorruptEntry(item.key, relPath, item.spec.chunkIndex, true, fp);
            } else {
              item.suspectByteLength = opfsResult.bytes.byteLength;
              await this._evictCorruptEntry(item.key, relPath, item.spec.chunkIndex, false, fp);
            }
          } else if (!isPrefetch && opfsResult.checkedOpfs) {
            this._stats.opfsMisses += 1;
          }
        }

        missing.push(item);
      }),
    );

    if (missing.length === 0) {
      return results;
    }

    // Re-check in-flight / memory after async OPFS phase and sort missing ascending by chunkIndex.
    missing.sort((a, b) => a.spec.chunkIndex - b.spec.chunkIndex);
    const toFetch = [];
    const lateJoins = [];

    for (let i = 0; i < missing.length; i += 1) {
      const item = missing[i];
      const memHit = this._touchMemory(item.key);
      if (memHit !== null && item.suspectByteLength === null) {
        const memFp = this._memChunkFingerprints.get(item.key) || null;
        if (fp === null || memFp === fp) {
          const verdict = this._validateCachedChunk(
            item.key,
            relPath,
            item.spec.chunkIndex,
            memHit,
            item.spec.requiredEndInChunk,
          );
          if (verdict === 'valid') {
            if (!isPrefetch) {
              this._stats.hits += 1;
              this._consumePrefetchHit(item.key);
            }
            results[item.pos] = memHit;
            continue;
          }
        }
      }
      const lateFlight = this._inFlight.get(item.key);
      if (lateFlight) {
        const joinedPrefetch = !isPrefetch && this._inFlightPrefetchKeys.delete(item.key);
        const joinedReadahead = !isPrefetch && this._inFlightReadaheadKeys.delete(item.key);
        lateJoins.push(
          lateFlight.then((bytes) => {
            if (bytes instanceof Uint8Array && bytes.byteLength > 0) {
              if (joinedPrefetch) {
                this._prefetchedKeys.delete(item.key);
                this._stats.prefetchHits += 1;
              }
              if (joinedReadahead) {
                this._readaheadKeys.delete(item.key);
                this._stats.readaheadHits += 1;
              }
            }
            results[item.pos] = bytes;
          }),
        );
        continue;
      }
      toFetch.push(item);
    }

    // 3. Group contiguous missing chunks into coalesced runs bounded by maxMergedBytes.
    const maxRunChunks = Math.max(1, Math.floor(maxMergedBytes / this.chunkSize));
    const runs = [];
    let currentRun = [];

    for (let i = 0; i < toFetch.length; i += 1) {
      const item = toFetch[i];
      if (
        currentRun.length > 0 &&
        item.spec.chunkIndex === currentRun[currentRun.length - 1].spec.chunkIndex + 1 &&
        currentRun.length < maxRunChunks
      ) {
        currentRun.push(item);
      } else {
        if (currentRun.length > 0) {
          runs.push(currentRun);
        }
        currentRun = [item];
      }
    }
    if (currentRun.length > 0) {
      runs.push(currentRun);
    }

    const storeFetchedChunk = async (item, bytes) => {
      if (item.suspectByteLength !== null && bytes.byteLength !== item.suspectByteLength) {
        this._stats.corruptEvictions += 1;
      }
      if (bytes.byteLength > 0) {
        const effectiveFp = this._resolveEffectiveFingerprint(relPath, options.fileFingerprint);
        this._recordExpectedChunkLength(item.key, bytes.byteLength);
        this._putMemory(item.key, bytes, effectiveFp);
        if (isPrefetch) {
          this._stats.prefetchBytes += bytes.byteLength;
          if (this._inFlightPrefetchKeys.has(item.key) && this._memCache.has(item.key)) {
            this._prefetchedKeys.add(item.key);
            if (isReadahead && this._inFlightReadaheadKeys.has(item.key)) {
              this._readaheadKeys.set(item.key, bytes.byteLength);
            }
          }
        }
        if (this.backgroundOpfsWrites) {
          this._enqueueOpfsWrite(relPath, item.spec.chunkIndex, bytes, effectiveFp);
        } else {
          await this._writeOpfs(relPath, item.spec.chunkIndex, bytes, effectiveFp);
        }
      }
    };

    const runPromises = runs.map((run) => {
      const deferreds = run.map((item) => {
        let resolve;
        let reject;
        const promise = new Promise((res, rej) => {
          resolve = res;
          reject = rej;
        });
        this._inFlight.set(item.key, promise);
        if (isPrefetch) {
          this._inFlightPrefetchKeys.add(item.key);
          if (isReadahead) {
            this._inFlightReadaheadKeys.add(item.key);
          }
        }
        return { item, promise, resolve, reject };
      });

      const executeRun = async () => {
        try {
          if (run.length === 1) {
            const d = deferreds[0];
            if (countMiss) {
              this._stats.misses += 1;
            }
            const netBytes = await this._fetchChunkFromNetwork(relPath, d.item.spec.chunkIndex);
            await storeFetchedChunk(d.item, netBytes);
            results[d.item.pos] = netBytes;
            d.resolve(netBytes);
            return;
          }

          if (countCoalesce) {
            this._stats.coalescedRangeCount += 1;
            this._stats.bytesSavedByCoalescing +=
              (run.length - 1) * ESTIMATED_HTTP_RANGE_OVERHEAD_BYTES;
          }
          if (countMiss) {
            this._stats.misses += run.length;
          }

          const startChunk = run[0].spec.chunkIndex;
          const endChunk = run[run.length - 1].spec.chunkIndex;
          const { chunks: fetchedMap, uncovered } = await this._fetchChunkRunFromNetwork(
            relPath,
            startChunk,
            endChunk,
          );
          const uncoveredSet = new Set(uncovered);

          for (let k = 0; k < deferreds.length; k += 1) {
            const d = deferreds[k];
            const idx = d.item.spec.chunkIndex;
            if (uncoveredSet.has(idx)) {
              try {
                const fallbackBytes = await this._fetchChunkFromNetwork(relPath, idx);
                await storeFetchedChunk(d.item, fallbackBytes);
                results[d.item.pos] = fallbackBytes;
                d.resolve(fallbackBytes);
              } catch (err) {
                d.reject(err);
              }
              continue;
            }
            const bytes = fetchedMap.get(idx) || new Uint8Array(0);
            await storeFetchedChunk(d.item, bytes);
            results[d.item.pos] = bytes;
            d.resolve(bytes);
          }
        } catch (err) {
          for (let k = 0; k < deferreds.length; k += 1) {
            deferreds[k].reject(err);
          }
        } finally {
          for (let k = 0; k < deferreds.length; k += 1) {
            const d = deferreds[k];
            this._inFlightPrefetchKeys.delete(d.item.key);
            this._inFlightReadaheadKeys.delete(d.item.key);
            if (this._inFlight.get(d.item.key) === d.promise) {
              this._inFlight.delete(d.item.key);
            }
          }
        }
      };

      executeRun();
      return Promise.all(deferreds.map((d) => d.promise));
    });

    await Promise.all([...lateJoins, ...runPromises]);
    return results;
  }

  _scheduleReadAhead(relPath, lastChunk, detection) {
    if (
      !this.enableReadAhead ||
      !detection ||
      detection.readAheadBytes <= 0 ||
      this.maxBytes < this.chunkSize * 2 ||
      this._activeReadAheadBatches >= this.maxPrefetchConcurrency ||
      (typeof this.fetchImpl !== 'function' && !this.opfsDir && !this.opfsCache)
    ) {
      return;
    }

    const safeBudget = Math.min(
      detection.readAheadBytes,
      this.maxReadaheadBytes,
      Math.max(0, this.maxBytes - this.chunkSize),
    );
    if (safeBudget < this.chunkSize) {
      return;
    }

    const aheadCount = Math.max(1, Math.floor(safeBudget / this.chunkSize));
    const aheadFirstChunk = lastChunk + 1;
    let aheadLastChunk = aheadFirstChunk + aheadCount - 1;

    if (this._fileSizes.has(relPath)) {
      const knownTotal = this._fileSizes.get(relPath);
      if (aheadFirstChunk * this.chunkSize >= knownTotal) {
        return;
      }
      const maxChunk = Math.floor((knownTotal - 1) / this.chunkSize);
      aheadLastChunk = Math.min(aheadLastChunk, maxChunk);
      if (aheadLastChunk < aheadFirstChunk) {
        return;
      }
    }

    // Watermark hysteresis: avoid issuing tiny 1-chunk sliding-edge fetches while at least
    // half of the adaptive read-ahead window is already cached or in-flight immediately ahead.
    let contiguousReady = 0;
    for (let idx = aheadFirstChunk; idx <= aheadLastChunk; idx += 1) {
      const key = this._chunkKey(relPath, idx);
      if (this._memCache.has(key) || this._inFlight.has(key)) {
        contiguousReady += 1;
      } else {
        break;
      }
    }

    const windowChunks = aheadLastChunk - aheadFirstChunk + 1;
    if (contiguousReady >= windowChunks) {
      return;
    }
    const refillWatermark = Math.ceil(aheadCount / 2);
    if (contiguousReady >= refillWatermark) {
      return;
    }

    const missingSpecs = [];
    for (let idx = aheadFirstChunk + contiguousReady; idx <= aheadLastChunk; idx += 1) {
      const key = this._chunkKey(relPath, idx);
      if (this._memCache.has(key) || this._inFlight.has(key)) {
        continue;
      }
      missingSpecs.push({ chunkIndex: idx, requiredEndInChunk: 0 });
    }

    if (missingSpecs.length === 0) {
      return;
    }

    const readAheadMaxMergedBytes = this._explicitCoalesceMaxMergedBytes
      ? this.coalesceMaxMergedBytes
      : Math.max(this.coalesceMaxMergedBytes, Math.min(safeBudget, this.maxReadaheadBytes));

    this._activeReadAheadBatches += 1;
    const prefetchPromise = (async () => {
      try {
        if (this.coalesceAdjacent || missingSpecs.length > 1) {
          await this._getChunksCoalesced(relPath, missingSpecs, {
            isPrefetch: true,
            isReadahead: true,
            countMiss: false,
            countCoalesce: true,
            maxMergedBytes: readAheadMaxMergedBytes,
          });
        } else {
          await this._getChunk(relPath, missingSpecs[0].chunkIndex, 0, {
            isPrefetch: true,
            isReadahead: true,
            countMiss: false,
          });
        }
      } catch {
        // Speculative read-ahead errors must never disrupt foreground reads.
      } finally {
        this._activeReadAheadBatches = Math.max(0, this._activeReadAheadBatches - 1);
      }
    })();

    this._activePrefetches.add(prefetchPromise);
    prefetchPromise.finally(() => {
      this._activePrefetches.delete(prefetchPromise);
    });
  }

  async flushPrefetches() {
    while (this._activePrefetches.size > 0) {
      const pending = Array.from(this._activePrefetches);
      await Promise.allSettled(pending);
    }
  }

  /**
   * Zero-alloc synchronous L1 cache-hit fast path: copies cached bytes for [offset, offset + length)
   * directly into caller-provided `targetBytes` without intermediate buffer/array allocations.
   * Returns exact bytes written (>= 0) on full L1 memory hit, or -1 on cache miss (leaving
   * `targetBytes` unmodified so the caller can fall back to async `read()`).
   */
  readCachedRangeInto(url, offset, length, targetBytes, options = {}) {
    const cleanPath = this._normalizePathOrUrl(url);
    if (!Number.isInteger(offset) || offset < 0) {
      throw new Error(`Invalid read offset: ${offset}`);
    }
    if (!Number.isInteger(length) || length < 0) {
      throw new Error(`Invalid read length: ${length}`);
    }
    if (!(targetBytes instanceof Uint8Array)) {
      throw new TypeError('readCachedRangeInto requires a Uint8Array targetBytes');
    }

    const readOpts = options && typeof options === 'object' ? options : {};
    const targetOffset =
      Number.isInteger(readOpts.targetOffset) && readOpts.targetOffset >= 0
        ? readOpts.targetOffset
        : 0;
    if (targetOffset > targetBytes.byteLength) {
      throw new RangeError(
        `targetOffset (${targetOffset}) exceeds targetBytes length (${targetBytes.byteLength})`,
      );
    }

    if (length === 0) {
      return 0;
    }

    let effectiveLength = length;
    if (this._fileSizes.has(cleanPath)) {
      const knownTotal = this._fileSizes.get(cleanPath);
      if (offset >= knownTotal) {
        return 0;
      }
      effectiveLength = Math.min(length, knownTotal - offset);
    }

    const fp = this._resolveEffectiveFingerprint(cleanPath, readOpts.fileFingerprint);
    const firstChunk = Math.floor(offset / this.chunkSize);
    const lastChunk = Math.floor((offset + effectiveLength - 1) / this.chunkSize);

    // Pass 1: verify all required chunks are present & valid in L1 RAM before touching targetBytes.
    let totalBytes = 0;
    let actualLastChunk = firstChunk;
    let reachedEof = false;

    for (let idx = firstChunk; idx <= lastChunk; idx += 1) {
      const key = this._chunkKey(cleanPath, idx);
      const chunk = this._memCache.get(key);
      if (!chunk) {
        return -1;
      }
      if (fp !== null) {
        const memFp = this._memChunkFingerprints.get(key) || null;
        if (memFp !== fp) {
          return -1;
        }
      }
      const chunkStart = idx * this.chunkSize;
      const requiredEndInChunk = Math.min(this.chunkSize, offset + effectiveLength - chunkStart);
      if (this._validateCachedChunk(key, cleanPath, idx, chunk, requiredEndInChunk) !== 'valid') {
        return -1;
      }
      const sliceStart = offset > chunkStart ? offset - chunkStart : 0;
      const maxEnd = offset + effectiveLength - chunkStart;
      const sliceEnd = chunk.byteLength < maxEnd ? chunk.byteLength : maxEnd;
      if (sliceEnd > sliceStart) {
        totalBytes += sliceEnd - sliceStart;
      }
      actualLastChunk = idx;
      if (chunk.byteLength < this.chunkSize) {
        reachedEof = true;
        break;
      }
    }

    if (targetBytes.byteLength - targetOffset < totalBytes) {
      throw new RangeError(
        `targetBytes capacity (${targetBytes.byteLength - targetOffset}) is smaller than required bytes (${totalBytes})`,
      );
    }

    // Pass 2: touch LRU, consume prefetch/readahead hits, and copy directly into targetBytes.
    let cursor = targetOffset;
    for (let idx = firstChunk; idx <= actualLastChunk; idx += 1) {
      const key = this._chunkKey(cleanPath, idx);
      const chunk = this._touchMemory(key);
      this._stats.hits += 1;
      this._consumePrefetchHit(key);

      const chunkStart = idx * this.chunkSize;
      const sliceStart = offset > chunkStart ? offset - chunkStart : 0;
      const maxEnd = offset + effectiveLength - chunkStart;
      const sliceEnd = chunk.byteLength < maxEnd ? chunk.byteLength : maxEnd;
      const copyLen = sliceEnd - sliceStart;
      if (copyLen > 0) {
        if (sliceStart === 0 && sliceEnd === chunk.byteLength) {
          targetBytes.set(chunk, cursor);
        } else if (copyLen <= 64) {
          for (let j = 0; j < copyLen; j += 1) {
            targetBytes[cursor + j] = chunk[sliceStart + j];
          }
        } else {
          targetBytes.set(chunk.subarray(sliceStart, sliceEnd), cursor);
        }
        cursor += copyLen;
      }
    }

    if (totalBytes > 0) {
      this._stats.zeroAllocCopyHits += 1;
      const seqDetection = this._updateSequentialTracker(cleanPath, offset, totalBytes);
      if (!reachedEof && seqDetection.readAheadBytes > 0) {
        this._scheduleReadAhead(cleanPath, actualLastChunk, seqDetection);
      }
    }

    return totalBytes;
  }

  async read(relPath, offset, length, options = {}) {
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

    const readOpts = options && typeof options === 'object' ? options : {};
    const fileFingerprint =
      readOpts.fileFingerprint !== undefined ? readOpts.fileFingerprint : undefined;

    let effectiveLength = length;
    if (this._fileSizes.has(cleanPath)) {
      const knownTotal = this._fileSizes.get(cleanPath);
      if (offset >= knownTotal) {
        return new Uint8Array(0);
      }
      effectiveLength = Math.min(length, knownTotal - offset);
    }

    const seqDetection = this._updateSequentialTracker(cleanPath, offset, effectiveLength);

    const firstChunk = Math.floor(offset / this.chunkSize);
    const lastChunk = Math.floor((offset + effectiveLength - 1) / this.chunkSize);

    let chunks;
    if (this.coalesceAdjacent && lastChunk > firstChunk) {
      const specs = [];
      for (let idx = firstChunk; idx <= lastChunk; idx += 1) {
        const chunkStart = idx * this.chunkSize;
        const requiredEndInChunk = Math.min(this.chunkSize, offset + effectiveLength - chunkStart);
        specs.push({ chunkIndex: idx, requiredEndInChunk });
      }
      chunks = await this._getChunksCoalesced(cleanPath, specs, {
        isPrefetch: false,
        countMiss: true,
        countCoalesce: true,
        fileFingerprint,
      });
    } else {
      const chunkPromises = [];
      for (let idx = firstChunk; idx <= lastChunk; idx += 1) {
        const chunkStart = idx * this.chunkSize;
        const requiredEndInChunk = Math.min(this.chunkSize, offset + effectiveLength - chunkStart);
        chunkPromises.push(
          this._getChunk(cleanPath, idx, requiredEndInChunk, { fileFingerprint }),
        );
      }
      chunks = await Promise.all(chunkPromises);
    }

    const slices = [];
    let totalBytes = 0;
    let reachedEof = false;

    for (let i = 0; i < chunks.length; i += 1) {
      const chunkIndex = firstChunk + i;
      const chunk = chunks[i];
      if (chunk.byteLength === 0) {
        reachedEof = true;
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
        reachedEof = true;
        break;
      }
    }

    if (!reachedEof && seqDetection.readAheadBytes > 0) {
      this._scheduleReadAhead(cleanPath, lastChunk, seqDetection);
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

  _chunkAlignedSpanBytes(offset, length) {
    if (length <= 0) return 0;
    const firstChunk = Math.floor(offset / this.chunkSize);
    const lastChunk = Math.floor((offset + length - 1) / this.chunkSize);
    return (lastChunk - firstChunk + 1) * this.chunkSize;
  }

  _planCoalescedSpans(rawEntries, maxGapBytes, maxMergedBytes) {
    const normalizedEntries = [];
    for (let i = 0; i < rawEntries.length; i += 1) {
      const entry = rawEntries[i];
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
      normalizedEntries.push({ path: cleanPath, offset, length });
    }

    const coalesced = coalesceAdjacentRanges(normalizedEntries, maxGapBytes, maxMergedBytes);

    // Track per-span individual chunk-aligned bytes so bytesSavedByCoalescing captures
    // both overlap reduction and shared-chunk deduplication accurately.
    const byPath = new Map();
    for (let i = 0; i < normalizedEntries.length; i += 1) {
      const e = normalizedEntries[i];
      let list = byPath.get(e.path);
      if (!list) {
        list = [];
        byPath.set(e.path, list);
      }
      list.push(e);
    }

    return coalesced.map((span) => {
      const pathEntries = byPath.get(span.path) || [];
      let sumIndividualAligned = 0;
      for (let i = 0; i < pathEntries.length; i += 1) {
        const e = pathEntries[i];
        if (e.offset >= span.offset && e.offset + e.length <= span.end) {
          sumIndividualAligned += this._chunkAlignedSpanBytes(e.offset, e.length);
        }
      }
      const mergedAligned = this._chunkAlignedSpanBytes(span.offset, span.length);
      const chunkRoundingSavings = Math.max(0, sumIndividualAligned - mergedAligned);
      const headerSavings =
        span.mergedCount > 1 ? (span.mergedCount - 1) * ESTIMATED_HTTP_RANGE_OVERHEAD_BYTES : 0;
      const savedBytes =
        span.mergedCount > 1
          ? Math.max(
              0,
              Math.max(span.overlapBytes, chunkRoundingSavings) + headerSavings - span.gapBytes,
            )
          : 0;
      return { ...span, savedBytes };
    });
  }

  async prefetchRanges(ranges, options = {}) {
    const rawList = Array.isArray(ranges) ? ranges : [];
    const concurrency =
      Number.isInteger(options.concurrency) && options.concurrency > 0
        ? options.concurrency
        : this.maxPrefetchConcurrency;
    const maxGapBytes =
      Number.isFinite(options.maxGapBytes) && options.maxGapBytes >= 0
        ? Math.floor(options.maxGapBytes)
        : this.coalesceMaxGapBytes;
    const maxMergedBytes =
      Number.isFinite(options.maxMergedBytes) && options.maxMergedBytes > 0
        ? Math.floor(options.maxMergedBytes)
        : this.coalesceMaxMergedBytes;

    const spans = this._planCoalescedSpans(rawList, maxGapBytes, maxMergedBytes);
    const uniqueChunkKeys = new Set();
    const spanSpecs = spans.map((span) => {
      const firstChunk = Math.floor(span.offset / this.chunkSize);
      const lastChunk = Math.floor((span.offset + span.length - 1) / this.chunkSize);
      const specs = [];
      for (let idx = firstChunk; idx <= lastChunk; idx += 1) {
        const chunkStart = idx * this.chunkSize;
        const requiredEndInChunk = Math.min(this.chunkSize, span.offset + span.length - chunkStart);
        uniqueChunkKeys.add(this._chunkKey(span.path, idx));
        specs.push({ chunkIndex: idx, requiredEndInChunk });
      }
      return { span, specs };
    });

    let nextIndex = 0;
    let prefetched = 0;
    let failed = 0;
    let batchBytesSaved = 0;
    const bytesBefore = this._stats.prefetchBytes;

    const worker = async () => {
      while (true) {
        const current = nextIndex;
        if (current >= spanSpecs.length) return;
        nextIndex += 1;
        const { span, specs } = spanSpecs[current];
        if (span.mergedCount > 1) {
          this._stats.coalescedRangeCount += 1;
          this._stats.bytesSavedByCoalescing += span.savedBytes;
          batchBytesSaved += span.savedBytes;
        }
        try {
          await this._getChunksCoalesced(span.path, specs, {
            isPrefetch: true,
            countMiss: false,
            countCoalesce: span.mergedCount <= 1,
            maxMergedBytes,
          });
          prefetched += specs.length;
        } catch {
          failed += specs.length;
        }
      }
    };

    const workerCount = Math.min(concurrency, spanSpecs.length);
    const workers = [];
    for (let i = 0; i < workerCount; i += 1) {
      workers.push(worker());
    }
    await Promise.all(workers);

    return {
      requestedRanges: rawList.length,
      coalescedRanges: spans.length,
      uniqueChunks: uniqueChunkKeys.size,
      prefetched,
      failed,
      bytesPrefetched: this._stats.prefetchBytes - bytesBefore,
      bytesSavedByCoalescing: batchBytesSaved,
    };
  }

  async readRanges(ranges, options = {}) {
    if (!Array.isArray(ranges) || ranges.length === 0) {
      return [];
    }
    const maxGapBytes =
      Number.isFinite(options.maxGapBytes) && options.maxGapBytes >= 0
        ? Math.floor(options.maxGapBytes)
        : this.coalesceMaxGapBytes;
    const maxMergedBytes =
      Number.isFinite(options.maxMergedBytes) && options.maxMergedBytes > 0
        ? Math.floor(options.maxMergedBytes)
        : this.coalesceMaxMergedBytes;
    const concurrency =
      Number.isInteger(options.concurrency) && options.concurrency > 0
        ? options.concurrency
        : this.maxPrefetchConcurrency;

    // Validate all requested ranges up front.
    const validated = ranges.map((entry) => {
      if (!entry || typeof entry !== 'object') {
        throw new Error('RangeCache.readRanges entry must be an object');
      }
      const cleanPath = normalizeRelPath(entry.path);
      const offset = entry.offset;
      const length = entry.length;
      if (!Number.isInteger(offset) || offset < 0) {
        throw new Error(`Invalid read offset: ${offset}`);
      }
      if (!Number.isInteger(length) || length < 0) {
        throw new Error(`Invalid read length: ${length}`);
      }
      return {
        path: cleanPath,
        offset,
        length,
        fileFingerprint:
          entry.fileFingerprint !== undefined ? entry.fileFingerprint : options.fileFingerprint,
      };
    });

    const nonZero = validated.filter((r) => r.length > 0);
    if (nonZero.length > 0) {
      const spans = this._planCoalescedSpans(nonZero, maxGapBytes, maxMergedBytes);
      let nextIndex = 0;

      const worker = async () => {
        while (true) {
          const current = nextIndex;
          if (current >= spans.length) return;
          nextIndex += 1;
          const span = spans[current];
          if (span.mergedCount > 1) {
            this._stats.coalescedRangeCount += 1;
            this._stats.bytesSavedByCoalescing += span.savedBytes;
          }
          const firstChunk = Math.floor(span.offset / this.chunkSize);
          const lastChunk = Math.floor((span.offset + span.length - 1) / this.chunkSize);
          const specs = [];
          for (let idx = firstChunk; idx <= lastChunk; idx += 1) {
            const chunkStart = idx * this.chunkSize;
            const requiredEndInChunk = Math.min(
              this.chunkSize,
              span.offset + span.length - chunkStart,
            );
            specs.push({ chunkIndex: idx, requiredEndInChunk });
          }
          await this._getChunksCoalesced(span.path, specs, {
            isPrefetch: false,
            countMiss: true,
            countCoalesce: span.mergedCount <= 1,
            maxMergedBytes,
            fileFingerprint: options.fileFingerprint,
          });
        }
      };

      const workerCount = Math.min(concurrency, spans.length);
      const workers = [];
      for (let i = 0; i < workerCount; i += 1) {
        workers.push(worker());
      }
      await Promise.all(workers);
    }

    // Slice exact outputs from the warmed cache (or direct fallback if evicted under tiny maxBytes).
    const outputs = new Array(validated.length);
    for (let i = 0; i < validated.length; i += 1) {
      const req = validated[i];
      if (req.length === 0) {
        outputs[i] = new Uint8Array(0);
        continue;
      }
      outputs[i] = await this.read(req.path, req.offset, req.length, {
        fileFingerprint: req.fileFingerprint,
      });
    }
    return outputs;
  }

  async prefetchBootset(entries, options = {}) {
    const rawList = Array.isArray(entries)
      ? entries
      : entries && Array.isArray(entries.entries)
        ? entries.entries
        : [];

    const useCoalesce =
      options.coalesce === true || (this.coalesceAdjacent && options.coalesce !== false);

    if (useCoalesce) {
      const summary = await this.prefetchRanges(rawList, options);
      return {
        requestedEntries: rawList.length,
        uniqueChunks: summary.uniqueChunks,
        prefetched: summary.prefetched,
        failed: summary.failed,
        coalescedRanges: summary.coalescedRanges,
      };
    }

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
          await this._getChunk(task.path, task.chunkIndex, task.requiredEndInChunk, {
            isPrefetch: true,
            countMiss: true,
          });
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
    this._memChunkFingerprints.clear();
    this._seqTrackers.clear();
    this._prefetchedKeys.clear();
    this._readaheadKeys.clear();
    this._bytesCached = 0;
    this._stats.adaptiveReadaheadWindowBytes = 0;
  }

  resetStats() {
    this._stats = this._createEmptyStats();
  }

  getStats() {
    const opfsLookups = this._stats.opfsHits + this._stats.opfsMisses;
    const opfsHitRate =
      opfsLookups > 0 ? Math.max(0, Math.min(1, this._stats.opfsHits / opfsLookups)) : 0;
    return {
      hits: this._stats.hits,
      misses: this._stats.misses,
      opfsHits: this._stats.opfsHits,
      opfsMisses: this._stats.opfsMisses,
      opfsWrites: this._stats.opfsWrites,
      opfsWriteErrors: this._stats.opfsWriteErrors,
      opfsStaleInvalidations: this._stats.opfsStaleInvalidations,
      opfsHitRate,
      evictions: this._stats.evictions,
      quotaEvictions: this._stats.quotaEvictions,
      quotaRetrySuccesses: this._stats.quotaRetrySuccesses,
      corruptEvictions: this._stats.corruptEvictions,
      coalescedRangeCount: this._stats.coalescedRangeCount,
      sequentialHits: this._stats.sequentialHits,
      prefetchHits: this._stats.prefetchHits,
      prefetchBytes: this._stats.prefetchBytes,
      bytesSavedByCoalescing: this._stats.bytesSavedByCoalescing,
      adaptiveReadaheadWindowBytes: this._stats.adaptiveReadaheadWindowBytes,
      sequentialStreakPeak: this._stats.sequentialStreakPeak,
      readaheadHits: this._stats.readaheadHits,
      readaheadWastedBytes: this._stats.readaheadWastedBytes,
      zeroAllocCopyHits: this._stats.zeroAllocCopyHits,
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
  DEFAULT_COALESCE_MAX_GAP_BYTES,
  DEFAULT_COALESCE_MAX_MERGED_BYTES,
  DEFAULT_MAX_READ_AHEAD_BYTES,
  DEFAULT_MIN_READAHEAD_BYTES,
  DEFAULT_MAX_READAHEAD_BYTES,
  DEFAULT_SEQUENTIAL_STREAK_THRESHOLD,
  DEFAULT_MAX_PENDING_OPFS_WRITES,
  DEFAULT_MAX_OPFS_WRITE_CONCURRENCY,
  normalizeRelPath,
  buildFileFingerprint,
  opfsChunkFileName,
  parseContentRange,
  openOpfsCacheDir,
  isQuotaExceededError,
  coalesceAdjacentRanges,
  detectSequentialRead,
};

if (typeof globalThis !== 'undefined') {
  globalThis.PlayGta6RangeCache = exportedApi;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = exportedApi;
}
