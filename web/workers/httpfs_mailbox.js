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

function createSharedMailboxHostState(options = {}, defaultReadAheadEnabled = false, slotCount = 1) {
  const opts = options && typeof options === 'object' ? options : {};
  return {
    slotCount: Math.max(1, Math.floor(Number(slotCount) || 1)),
    handles: new Map(),
    nextHandle: 1,
    handleRecords: new Map(),
    closedOrder: [],
    now: typeof opts.now === 'function' ? opts.now : defaultNowMs,
    readAhead: normalizeReadAheadConfig(opts, defaultReadAheadEnabled),
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
    if (!cache || typeof cache.read !== 'function') {
      this.completeResponse(-19n, HttpfsStatus.Error);
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
    let cache = null;
    if (!this._customRead) {
      const entry = this._handles.get(handle);
      if (!entry) {
        this.completeResponse(-9n, HttpfsStatus.Error);
        return;
      }
      entryPath = entry.path;
      if (reqLen === 0) {
        this.completeResponse(0n, HttpfsStatus.ResponseReady);
        return;
      }
      cache = this._resolveRangeCache();
      if (!cache || typeof cache.read !== 'function') {
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
  HttpfsStatus,
  HttpfsOp,
  HttpfsOffsets,
  normalizeRelPath,
  isVideoReadAheadPath,
  computeReadAheadWindowSize,
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
