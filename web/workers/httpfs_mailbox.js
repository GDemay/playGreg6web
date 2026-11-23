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

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder('utf-8', { fatal: false });

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

    this._handles = new Map();
    this._nextHandle = 1;
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
    this._headerView.setBigInt64(HttpfsOffsets.RESULT, BigInt(result), true);
    Atomics.store(this._statusView, 0, status);
    Atomics.notify(this._statusView, 0, 1);
  }

  async _handleOpen(lengthBig, cap) {
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
      const handle = await this._customOpen(cleanPath);
      if (typeof handle === 'number' && Number.isInteger(handle) && handle >= 0) {
        this.completeResponse(BigInt(handle), HttpfsStatus.ResponseReady);
      } else {
        const errCode = typeof handle === 'number' && handle < 0 ? Math.trunc(handle) : -2;
        this.completeResponse(BigInt(errCode), HttpfsStatus.Error);
      }
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
      try {
        await cache.read(cleanPath, 0, 1);
      } catch {
        this.completeResponse(-2n, HttpfsStatus.Error);
        return;
      }
    }

    const handle = this._nextHandle;
    this._nextHandle += 1;
    this._handles.set(handle, { path: cleanPath });
    this.completeResponse(BigInt(handle), HttpfsStatus.ResponseReady);
  }

  async _handleRead(handle, offsetBig, lengthBig, cap) {
    if (offsetBig > BigInt(Number.MAX_SAFE_INTEGER)) {
      this.completeResponse(-22n, HttpfsStatus.Error);
      return;
    }

    const offset = Number(offsetBig);
    const reqLen = lengthBig > BigInt(cap) ? cap : Number(lengthBig);

    if (this._customRead) {
      if (reqLen === 0) {
        this.completeResponse(0n, HttpfsStatus.ResponseReady);
        return;
      }
      const res = await this._customRead(handle, offset, reqLen);
      if (res === null || res === undefined) {
        this.completeResponse(-9n, HttpfsStatus.Error);
        return;
      }
      const bytes =
        res instanceof Uint8Array
          ? res
          : res instanceof ArrayBuffer
            ? new Uint8Array(res)
            : ArrayBuffer.isView(res)
              ? new Uint8Array(res.buffer, res.byteOffset, res.byteLength)
              : null;
      if (!bytes) {
        this.completeResponse(-5n, HttpfsStatus.Error);
        return;
      }
      const toCopy = Math.min(bytes.byteLength, reqLen);
      if (toCopy > 0) {
        this._payloadView.set(bytes.subarray(0, toCopy), 0);
      }
      this.completeResponse(BigInt(toCopy), HttpfsStatus.ResponseReady);
      return;
    }

    const entry = this._handles.get(handle);
    if (!entry) {
      this.completeResponse(-9n, HttpfsStatus.Error);
      return;
    }

    if (reqLen === 0) {
      this.completeResponse(0n, HttpfsStatus.ResponseReady);
      return;
    }

    const cache = this._resolveRangeCache();
    if (!cache || typeof cache.read !== 'function') {
      this.completeResponse(-19n, HttpfsStatus.Error);
      return;
    }

    const bytes = await cache.read(entry.path, offset, reqLen);
    const toCopy = Math.min(bytes.byteLength, reqLen);
    if (toCopy > 0) {
      this._payloadView.set(bytes.subarray(0, toCopy), 0);
    }
    this.completeResponse(BigInt(toCopy), HttpfsStatus.ResponseReady);
  }

  async _handleClose(handle) {
    if (this._customClose) {
      const res = await this._customClose(handle);
      if (res === false || (typeof res === 'number' && res < 0)) {
        const errCode = typeof res === 'number' && res < 0 ? Math.trunc(res) : -9;
        this.completeResponse(BigInt(errCode), HttpfsStatus.Error);
      } else {
        this.completeResponse(0n, HttpfsStatus.ResponseReady);
      }
      return;
    }

    if (this._handles.delete(handle)) {
      this.completeResponse(0n, HttpfsStatus.ResponseReady);
    } else {
      this.completeResponse(-9n, HttpfsStatus.Error);
    }
  }

  async processOnce() {
    const status = Atomics.load(this._statusView, 0);
    if (status !== HttpfsStatus.RequestReady) {
      return false;
    }

    const op = this._headerView.getUint32(HttpfsOffsets.OP, true);
    const handle = this._headerView.getInt32(HttpfsOffsets.HANDLE, true);
    const offsetBig = this._headerView.getBigUint64(HttpfsOffsets.OFFSET, true);
    const lengthBig = this._headerView.getBigUint64(HttpfsOffsets.LENGTH, true);
    const cap = this.payloadCapacity;

    try {
      if (op === HttpfsOp.Open) {
        await this._handleOpen(lengthBig, cap);
      } else if (op === HttpfsOp.Read) {
        await this._handleRead(handle, offsetBig, lengthBig, cap);
      } else if (op === HttpfsOp.Close) {
        await this._handleClose(handle);
      } else {
        this.completeResponse(-1n, HttpfsStatus.Error);
      }
    } catch {
      const errCode = op === HttpfsOp.Open ? -2n : -5n;
      this.completeResponse(errCode, HttpfsStatus.Error);
    }

    return true;
  }

  _waitForChange(expectedStatus) {
    if (!this._forcePoll && typeof Atomics.waitAsync === 'function') {
      const res = Atomics.waitAsync(this._statusView, 0, expectedStatus);
      return res.async ? res.value : Promise.resolve(res.value);
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
      while (this._running) {
        const status = Atomics.load(this._statusView, 0);
        if (status === HttpfsStatus.Shutdown) {
          break;
        }
        if (status === HttpfsStatus.RequestReady) {
          await this.processOnce();
          continue;
        }
        await this._waitForChange(status);
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

const exportedApi = {
  HTTPFS_MAILBOX_VERSION,
  HTTPFS_HEADER_BYTES,
  HTTPFS_PAYLOAD_BYTES,
  HTTPFS_MAILBOX_BYTES,
  HttpfsStatus,
  HttpfsOp,
  HttpfsOffsets,
  normalizeRelPath,
  initHttpfsMailbox,
  createHttpfsMailboxBuffer,
  HttpfsMailboxHost,
  HttpfsMailboxClient,
};

if (typeof globalThis !== 'undefined') {
  globalThis.PlayGta6HttpfsMailbox = exportedApi;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = exportedApi;
}
})();
