// Binary WGPK (.wgsl.pack) parser and WebGPU GPUShaderModule cache.
// Pure ES module with zero external dependencies — runs in Web Workers and Node.js.
// Binary layout matches shaders/src/pack.rs and shaders/src/convert.rs byte-for-byte.

/** Binary pack file magic ("WGPK"). */
export const WGPK_MAGIC = 'WGPK';

/** Binary pack magic as raw ASCII bytes. */
export const WGPK_MAGIC_BYTES = Object.freeze([0x57, 0x47, 0x50, 0x4b]);

/** Supported WGPK format version (matches PACK_VERSION in shaders/src/pack.rs). */
export const WGPK_VERSION = 1;

/** Fixed header size in bytes: magic(4) + version(4) + entry_count(4) + flags(4). */
export const WGPK_HEADER_BYTES = 16;

/** Fixed entry table record size in bytes: hash(8) + stage(1) + reserved(3) + offset(4) + length(4). */
export const WGPK_ENTRY_BYTES = 20;

/** Default initial byte window for HTTP 206 Range index loading (64 KiB: header + ~3,276 entries in 1 roundtrip). */
export const DEFAULT_INITIAL_RANGE_BYTES = 64 * 1024;

/** Default bounded in-memory LRU cache budget for on-demand WGSL strings in ranged mode (32 MiB). */
export const DEFAULT_MAX_CACHED_WGSL_BYTES = 32 * 1024 * 1024;

/** Default maximum concurrent HTTP 206 Range payload fetches per ranged pack (`0` = unlimited). */
export const DEFAULT_MAX_CONCURRENT_RANGE_FETCHES = 8;

/** Default maximum retries on transient HTTP 206 Range fetch failures (`0` = no retries). */
export const DEFAULT_MAX_RANGE_FETCH_RETRIES = 2;

/** Default base backoff delay in milliseconds before retrying a transient Range fetch failure. */
export const DEFAULT_RANGE_RETRY_DELAY_MS = 25;

/** Default maximum byte gap between nearby shaders to coalesce into a single prefetch HTTP 206 Range request (16 KiB). */
export const DEFAULT_PREFETCH_MAX_GAP_BYTES = 16 * 1024;

/** Default maximum byte span for a single coalesced prefetch HTTP 206 Range request (512 KiB). */
export const DEFAULT_PREFETCH_MAX_SPAN_BYTES = 512 * 1024;

/** Shader index ABI version matching shaders/src/pack.rs and docs/gpu_commands.md. */
export const SHADER_INDEX_ABI_VERSION = 1;

/**
 * WebGPU shader stage enum matching `ShaderStage` in `shaders/src/convert.rs`.
 * Note: `.wgsl.pack` encodes Vertex=1, Fragment=2, Compute=3.
 */
export const SHADER_STAGE = Object.freeze({
  VERTEX: 1,
  FRAGMENT: 2,
  COMPUTE: 3,
});

/** Reverse lookup from WGPK numeric stage code to canonical stage name. */
export const STAGE_NAMES = Object.freeze({
  [SHADER_STAGE.VERTEX]: 'vertex',
  [SHADER_STAGE.FRAGMENT]: 'fragment',
  [SHADER_STAGE.COMPUTE]: 'compute',
});

const STAGE_NAME_TO_CODE = Object.freeze({
  vertex: SHADER_STAGE.VERTEX,
  fragment: SHADER_STAGE.FRAGMENT,
  compute: SHADER_STAGE.COMPUTE,
});

const FNV1A_64_OFFSET_BASIS = 0xcbf29ce484222325n;
const FNV1A_64_PRIME = 0x00000100000001b3n;
const U64_MAX = 0xffffffffffffffffn;
const U32_MAX = 0xffffffff;
const U32_RANGE_LIMIT = 0x100000000;

function sanitizeNonNegativeInt(value, fallback = 0) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    return fallback;
  }
  const floored = Math.floor(value);
  return Number.isSafeInteger(floored) ? floored : Number.MAX_SAFE_INTEGER;
}

const textEncoder = new TextEncoder();
const textDecoderStrict = new TextDecoder('utf-8', { fatal: true });

function isSharedArrayBuffer(buf) {
  return typeof SharedArrayBuffer !== 'undefined' && buf instanceof SharedArrayBuffer;
}

function toUint8Array(input, label = 'buffer') {
  if (typeof input === 'string') {
    return textEncoder.encode(input);
  }
  if (input instanceof Uint8Array) {
    return input;
  }
  if (ArrayBuffer.isView(input)) {
    return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  }
  if (input instanceof ArrayBuffer || isSharedArrayBuffer(input)) {
    return new Uint8Array(input);
  }
  throw new TypeError(`Expected ${label} to be an ArrayBuffer, TypedArray, DataView, or string`);
}

function toBinaryView(buffer) {
  if (buffer instanceof Uint8Array) {
    return buffer;
  }
  if (ArrayBuffer.isView(buffer)) {
    return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  }
  if (buffer instanceof ArrayBuffer || isSharedArrayBuffer(buffer)) {
    return new Uint8Array(buffer);
  }
  throw new TypeError('Expected WGPK buffer to be an ArrayBuffer, SharedArrayBuffer, or ArrayBufferView');
}

function decodeUtf8Strict(slice, entryIndex) {
  const safeSlice = isSharedArrayBuffer(slice.buffer) ? slice.slice() : slice;
  try {
    return textDecoderStrict.decode(safeSlice);
  } catch (err) {
    throw new Error(
      `WGPK entry ${entryIndex} payload is not valid UTF-8: ${err instanceof Error ? err.message : String(err)}`
    );
  }
}

function getResponseHeader(response, headerName) {
  if (!response || !response.headers) {
    return null;
  }
  const headers = response.headers;
  if (typeof headers.get === 'function') {
    const val = headers.get(headerName);
    return val === undefined ? null : val;
  }
  const target = headerName.toLowerCase();
  for (const [k, v] of Object.entries(headers)) {
    if (k.toLowerCase() === target) {
      return v === undefined || v === null ? null : String(v);
    }
  }
  return null;
}

function parseNonNegativeInt(raw) {
  if (raw === null || raw === undefined) {
    return null;
  }
  const trimmed = String(raw).trim();
  if (!/^\d+$/.test(trimmed)) {
    return null;
  }
  const value = Number(trimmed);
  if (!Number.isSafeInteger(value) || value < 0) {
    return null;
  }
  return value;
}

function parseContentRange(headerValue) {
  if (typeof headerValue !== 'string') {
    return null;
  }
  const match = /^\s*bytes\s+(\d+)-(\d+)\/(\d+|\*)\s*$/i.exec(headerValue);
  if (!match) {
    return null;
  }
  const start = Number(match[1]);
  const end = Number(match[2]);
  const total = match[3] === '*' ? null : Number(match[3]);
  if (
    !Number.isSafeInteger(start) ||
    !Number.isSafeInteger(end) ||
    start < 0 ||
    end < start ||
    (total !== null && (!Number.isSafeInteger(total) || total <= end))
  ) {
    return null;
  }
  return { start, end, total };
}

function mergeRequestInit(baseInit, overrides = {}) {
  const init = baseInit && typeof baseInit === 'object' ? { ...baseInit } : {};
  const mergedHeaders = {};

  if (init.headers) {
    if (typeof init.headers.forEach === 'function') {
      init.headers.forEach((value, key) => {
        mergedHeaders[key] = value;
      });
    } else if (Array.isArray(init.headers)) {
      for (const pair of init.headers) {
        if (Array.isArray(pair) && pair.length >= 2) {
          mergedHeaders[String(pair[0])] = String(pair[1]);
        }
      }
    } else if (typeof init.headers === 'object') {
      for (const [key, value] of Object.entries(init.headers)) {
        mergedHeaders[key] = value;
      }
    }
  }

  if (overrides.headers) {
    for (const [key, value] of Object.entries(overrides.headers)) {
      // Remove any existing header with the same case-insensitive name before setting.
      const lower = key.toLowerCase();
      for (const existingKey of Object.keys(mergedHeaders)) {
        if (existingKey.toLowerCase() === lower) {
          delete mergedHeaders[existingKey];
        }
      }
      mergedHeaders[key] = value;
    }
  }

  if (overrides.method !== undefined) {
    init.method = overrides.method;
  }
  if (Object.keys(mergedHeaders).length > 0) {
    init.headers = mergedHeaders;
  }
  return init;
}

function resolveFetchAndOptions(fetchImplOrOptions, maybeOptions, fallbackFetch) {
  let options;
  let fetcher;
  if (typeof fetchImplOrOptions === 'function') {
    fetcher = fetchImplOrOptions;
    options = maybeOptions && typeof maybeOptions === 'object' ? maybeOptions : {};
  } else {
    options = fetchImplOrOptions && typeof fetchImplOrOptions === 'object' ? fetchImplOrOptions : {};
    fetcher = options.fetch ?? fallbackFetch ?? globalThis.fetch;
  }
  return { fetcher, options };
}

function validateMaxCachedWgslBytes(value, defaultValue = DEFAULT_MAX_CACHED_WGSL_BYTES) {
  if (value === undefined || value === null) {
    return defaultValue;
  }
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new RangeError(`maxCachedWgslBytes must be a non-negative integer, got: ${value}`);
  }
  return value;
}

function validateInitialRangeBytes(value, defaultValue = DEFAULT_INITIAL_RANGE_BYTES) {
  if (value === undefined || value === null) {
    return defaultValue;
  }
  if (typeof value !== 'number' || !Number.isInteger(value) || value < WGPK_HEADER_BYTES) {
    throw new RangeError(
      `initialRangeBytes must be an integer >= ${WGPK_HEADER_BYTES}, got: ${value}`
    );
  }
  return value;
}

function validateStreamThresholdBytes(value, defaultValue = 0) {
  if (value === undefined || value === null) {
    return defaultValue;
  }
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new RangeError(`streamThresholdBytes must be a non-negative integer, got: ${value}`);
  }
  return value;
}

function validateMaxConcurrentRangeFetches(
  value,
  defaultValue = DEFAULT_MAX_CONCURRENT_RANGE_FETCHES
) {
  if (value === undefined || value === null) {
    return defaultValue;
  }
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new RangeError(
      `maxConcurrentRangeFetches must be a non-negative integer, got: ${value}`
    );
  }
  return value;
}

function validateMaxRangeFetchRetries(
  value,
  defaultValue = DEFAULT_MAX_RANGE_FETCH_RETRIES
) {
  if (value === undefined || value === null) {
    return defaultValue;
  }
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new RangeError(`maxRangeFetchRetries must be a non-negative integer, got: ${value}`);
  }
  return value;
}

function validateRangeRetryDelayMs(value, defaultValue = DEFAULT_RANGE_RETRY_DELAY_MS) {
  if (value === undefined || value === null) {
    return defaultValue;
  }
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new RangeError(
      `rangeRetryDelayMs must be a non-negative finite number, got: ${value}`
    );
  }
  return value;
}

function validateSleepFn(value, defaultValue = null) {
  if (value === undefined || value === null) {
    return defaultValue;
  }
  if (typeof value !== 'function') {
    throw new TypeError('sleepFn must be a function');
  }
  return value;
}

function validatePrefetchMaxGapBytes(value, defaultValue = DEFAULT_PREFETCH_MAX_GAP_BYTES) {
  if (value === undefined || value === null) {
    return defaultValue;
  }
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new RangeError(`prefetchMaxGapBytes must be a non-negative integer, got: ${value}`);
  }
  return value;
}

function validatePrefetchMaxSpanBytes(value, defaultValue = DEFAULT_PREFETCH_MAX_SPAN_BYTES) {
  if (value === undefined || value === null) {
    return defaultValue;
  }
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 1) {
    throw new RangeError(`prefetchMaxSpanBytes must be a positive integer, got: ${value}`);
  }
  return value;
}

function defaultSleep(ms) {
  if (ms <= 0) {
    return Promise.resolve();
  }
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function computeRangeRetryDelayMs(baseDelayMs, attemptIndex) {
  if (baseDelayMs <= 0) {
    return 0;
  }
  const factor = 1 << Math.min(attemptIndex, 4);
  return baseDelayMs * factor;
}

function isRetryableRangeFetchError(err, fetchInit) {
  if (fetchInit && fetchInit.signal && fetchInit.signal.aborted) {
    return false;
  }
  if (!err) {
    return false;
  }
  if (err.name === 'AbortError' || err.code === 'ABORT_ERR') {
    return false;
  }
  if (err.name === 'AssertionError' || err.code === 'ERR_ASSERTION') {
    return false;
  }
  if (err instanceof RangeError) {
    return false;
  }
  if (typeof err.httpStatus === 'number') {
    const status = err.httpStatus;
    return status === 429 || status === 408 || (status >= 500 && status <= 599);
  }
  // Network / transport errors thrown by fetch() or response.arrayBuffer()
  // (e.g. TypeError: Failed to fetch, ECONNRESET, premature connection close).
  return true;
}

function parseWgpkHeader(bytes) {
  if (bytes.byteLength < WGPK_HEADER_BYTES) {
    throw new Error(
      `WGPK pack truncated: ${bytes.byteLength} bytes (minimum required for header: ${WGPK_HEADER_BYTES})`
    );
  }

  if (
    bytes[0] !== WGPK_MAGIC_BYTES[0] ||
    bytes[1] !== WGPK_MAGIC_BYTES[1] ||
    bytes[2] !== WGPK_MAGIC_BYTES[2] ||
    bytes[3] !== WGPK_MAGIC_BYTES[3]
  ) {
    const actualHex = Array.from(bytes.subarray(0, 4))
      .map((b) => `0x${b.toString(16).padStart(2, '0')}`)
      .join(' ');
    throw new Error(`Invalid WGPK pack magic [${actualHex}] (expected "${WGPK_MAGIC}")`);
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = view.getUint32(4, true) >>> 0;
  if (version !== WGPK_VERSION) {
    throw new Error(`Unsupported WGPK pack version ${version} (expected ${WGPK_VERSION})`);
  }

  const entryCount = view.getUint32(8, true) >>> 0;
  const flags = view.getUint32(12, true) >>> 0;
  const tableBytes = entryCount * WGPK_ENTRY_BYTES;
  const dataStart = WGPK_HEADER_BYTES + tableBytes;

  return { version, entryCount, flags, tableBytes, dataStart };
}

function parseWgpkEntryTable(bytes, header, totalFileBytes) {
  const { entryCount, dataStart } = header;
  if (bytes.byteLength < dataStart) {
    throw new Error(
      `WGPK pack truncated: ${bytes.byteLength} bytes (header + ${entryCount} entry table records require at least ${dataStart} bytes)`
    );
  }
  if (totalFileBytes !== null && totalFileBytes !== undefined && totalFileBytes < dataStart) {
    throw new Error(
      `WGPK pack truncated: ${totalFileBytes} total bytes (header + ${entryCount} entry table records require at least ${dataStart} bytes)`
    );
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const rawEntries = new Array(entryCount);
  const ranges = [];
  const stageCounts = { vertex: 0, fragment: 0, compute: 0 };
  let totalPayloadBytes = 0;
  let maxEnd = dataStart;

  // When totalFileBytes is unknown (e.g. Content-Range: bytes 0-N/*), allow full unsigned 32-bit
  // byte indices [0..0xFFFFFFFF] up to the exact 4 GiB exclusive end (0x100000000).
  const upperBound =
    totalFileBytes !== null && totalFileBytes !== undefined ? totalFileBytes : U32_RANGE_LIMIT;

  for (let idx = 0; idx < entryCount; idx += 1) {
    const base = WGPK_HEADER_BYTES + idx * WGPK_ENTRY_BYTES;
    const hash = view.getBigUint64(base, true);
    const stageCode = bytes[base + 8];
    const stage = STAGE_NAMES[stageCode];
    if (stage === undefined) {
      throw new Error(`Invalid WGPK shader stage value ${stageCode} at entry ${idx}`);
    }
    // bytes[base + 9 .. base + 12] are reserved for forward compatibility.
    const offset = view.getUint32(base + 12, true) >>> 0;
    const length = view.getUint32(base + 16, true) >>> 0;
    const end = offset + length;

    if (!Number.isSafeInteger(end) || offset < dataStart || end > upperBound) {
      throw new Error(
        `WGPK entry ${idx} payload [${offset}..+${length}] is out of bounds (valid payload range: [${dataStart}..${upperBound}])`
      );
    }

    if (length > 0) {
      ranges.push({ start: offset, end, index: idx });
    }

    if (end > maxEnd) {
      maxEnd = end;
    }

    totalPayloadBytes += length;
    stageCounts[stage] += 1;

    rawEntries[idx] = {
      index: idx,
      hash,
      hashHex: formatHashHex(hash),
      stage,
      stageCode,
      offset,
      length,
      end,
    };
  }

  // Reject overlapping non-empty payload ranges across entries.
  if (ranges.length > 1) {
    ranges.sort((a, b) => a.start - b.start || a.end - b.end);
    for (let i = 1; i < ranges.length; i += 1) {
      const prev = ranges[i - 1];
      const curr = ranges[i];
      if (curr.start < prev.end) {
        throw new Error(
          `WGPK overlapping payload ranges detected between entry ${prev.index} [${prev.start}..${prev.end}) and entry ${curr.index} [${curr.start}..${curr.end})`
        );
      }
    }
  }

  return { rawEntries, stageCounts, totalPayloadBytes, maxEnd };
}

async function fetchRangeChunk(fetcher, url, start, end, fetchInit) {
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start) {
    throw new RangeError(`Invalid HTTP byte range: bytes=${start}-${end}`);
  }
  const rangeHeader = `bytes=${start}-${end}`;
  const response = await fetcher(url, mergeRequestInit(fetchInit, { headers: { Range: rangeHeader } }));
  if (!response || (!response.ok && response.status !== 206)) {
    const status = response ? response.status : 'unknown';
    const statusText = response && response.statusText ? ` ${response.statusText}` : '';
    const err = new Error(
      `Failed to fetch WGSL pack range (${rangeHeader}) from "${url}": HTTP ${status}${statusText}`
    );
    err.httpStatus = response && typeof response.status === 'number' ? response.status : 0;
    throw err;
  }

  const buffer = await response.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const contentRange = parseContentRange(getResponseHeader(response, 'content-range'));
  const contentLength = parseNonNegativeInt(getResponseHeader(response, 'content-length'));

  return {
    status: response.status,
    bytes,
    contentRange,
    contentLength,
  };
}

async function fetchRangeChunkWithRetry(fetcher, url, start, end, fetchInit, retryOptions = {}) {
  const maxRetries = validateMaxRangeFetchRetries(retryOptions.maxRangeFetchRetries);
  const baseDelayMs = validateRangeRetryDelayMs(retryOptions.rangeRetryDelayMs);
  const sleep = validateSleepFn(retryOptions.sleepFn) || defaultSleep;
  const onAttempt = typeof retryOptions.onAttempt === 'function' ? retryOptions.onAttempt : null;
  const onRetry = typeof retryOptions.onRetry === 'function' ? retryOptions.onRetry : null;

  let attempt = 0;
  for (;;) {
    if (onAttempt) {
      onAttempt();
    }
    try {
      return await fetchRangeChunk(fetcher, url, start, end, fetchInit);
    } catch (err) {
      if (attempt >= maxRetries || !isRetryableRangeFetchError(err, fetchInit)) {
        throw err;
      }
      if (onRetry) {
        onRetry(err, attempt);
      }
      const delayMs = computeRangeRetryDelayMs(baseDelayMs, attempt);
      attempt += 1;
      await sleep(delayMs);
      if (fetchInit && fetchInit.signal && fetchInit.signal.aborted) {
        throw err;
      }
    }
  }
}

/**
 * Compute deterministic 64-bit FNV-1a hash matching `fnv1a_64` in `shaders/src/convert.rs`.
 *
 * @param {Uint8Array | ArrayBuffer | ArrayBufferView | string} input
 * @returns {bigint} Unsigned 64-bit hash as BigInt
 */
export function fnv1a64(input) {
  const bytes = toUint8Array(input, 'input');
  let hash = FNV1A_64_OFFSET_BASIS;
  for (let i = 0; i < bytes.length; i += 1) {
    hash ^= BigInt(bytes[i]);
    hash = (hash * FNV1A_64_PRIME) & U64_MAX;
  }
  return hash;
}

/**
 * Normalize a shader hash (`BigInt`, hex `string`, or safe non-negative integer `number`)
 * into an unsigned 64-bit `BigInt`.
 *
 * @param {bigint | string | number} hash
 * @returns {bigint}
 */
export function normalizeHash(hash) {
  if (typeof hash === 'bigint') {
    if (hash < 0n || hash > U64_MAX) {
      throw new RangeError(`Shader hash out of u64 range: ${hash.toString()}`);
    }
    return hash;
  }

  if (typeof hash === 'number') {
    if (!Number.isInteger(hash) || !Number.isSafeInteger(hash) || hash < 0) {
      throw new RangeError(`Numeric shader hash must be a non-negative safe integer, got: ${hash}`);
    }
    return BigInt(hash);
  }

  if (typeof hash === 'string') {
    const trimmed = hash.trim();
    const rawHex = trimmed.startsWith('0x') || trimmed.startsWith('0X') ? trimmed.slice(2) : trimmed;
    if (!/^[0-9a-fA-F]{1,16}$/.test(rawHex)) {
      throw new RangeError(`Invalid hex shader hash: ${JSON.stringify(hash)}`);
    }
    return BigInt(`0x${rawHex}`);
  }

  throw new TypeError(`Shader hash must be a BigInt, hex string, or non-negative safe integer`);
}

/**
 * Format a 64-bit shader hash as canonical `0x` + 16 lowercase hex digits,
 * matching `format_shader_hash` in `shaders/src/convert.rs`.
 *
 * @param {bigint | string | number} hash
 * @returns {string}
 */
export function formatHashHex(hash) {
  const value = normalizeHash(hash);
  return `0x${value.toString(16).padStart(16, '0')}`;
}

function resolveStageCode(stage) {
  if (typeof stage === 'number') {
    if (Object.prototype.hasOwnProperty.call(STAGE_NAMES, stage)) {
      return stage;
    }
    throw new RangeError(`Invalid shader stage code: ${stage}`);
  }
  if (typeof stage === 'string') {
    const code = STAGE_NAME_TO_CODE[stage.toLowerCase()];
    if (code !== undefined) {
      return code;
    }
    throw new RangeError(`Invalid shader stage name: ${JSON.stringify(stage)}`);
  }
  throw new TypeError(`Shader stage must be a stage code (1..3) or stage name`);
}

function createDynamicShaderEntry(hash, stage, wgslSource, meta, prevEntry) {
  const key = normalizeHash(hash);
  const hashHex = formatHashHex(key);
  const stageCode = resolveStageCode(stage);
  const stageName = STAGE_NAMES[stageCode];

  if (typeof wgslSource !== 'string') {
    throw new TypeError('wgslSource must be a non-empty string');
  }
  if (wgslSource.trim().length === 0) {
    throw new RangeError('wgslSource must be a non-empty WGSL string');
  }

  let opts = {};
  if (meta !== undefined && meta !== null) {
    if (typeof meta !== 'object' || Array.isArray(meta)) {
      throw new TypeError('meta must be an object');
    }
    opts = meta;
  }

  let label = null;
  if (opts.label !== undefined && opts.label !== null) {
    if (typeof opts.label !== 'string') {
      throw new TypeError('meta.label must be a string');
    }
    label = opts.label;
  }

  let entryPoint = null;
  if (opts.entryPoint !== undefined && opts.entryPoint !== null) {
    if (typeof opts.entryPoint !== 'string') {
      throw new TypeError('meta.entryPoint must be a string');
    }
    entryPoint = opts.entryPoint;
  }

  let source = 'dynamic';
  if (opts.source !== undefined && opts.source !== null) {
    if (typeof opts.source !== 'string' || opts.source.trim().length === 0) {
      throw new TypeError('meta.source must be a non-empty string');
    }
    source = opts.source;
  }

  let tags = [];
  if (opts.tags !== undefined && opts.tags !== null) {
    if (!Array.isArray(opts.tags) || opts.tags.some((t) => typeof t !== 'string')) {
      throw new TypeError('meta.tags must be an array of strings');
    }
    tags = opts.tags.slice();
  }

  const byteLength = textEncoder.encode(wgslSource).byteLength;
  const revision = prevEntry ? prevEntry.revision + 1 : 1;

  return {
    hash: key,
    hashHex,
    stage: stageName,
    stageCode,
    wgsl: wgslSource,
    length: byteLength,
    dynamic: true,
    override: Boolean(opts.override),
    source,
    label,
    entryPoint,
    tags,
    revision,
    registeredAt: Date.now(),
  };
}

function formatDynamicLookupRecord(entry, overridesStatic) {
  return {
    hash: entry.hash,
    hashHex: entry.hashHex,
    stage: entry.stage,
    stageCode: entry.stageCode,
    wgsl: entry.wgsl,
    length: entry.length,
    dynamic: true,
    overridesStatic,
    packName: null,
    source: entry.source,
    label: entry.label,
    entryPoint: entry.entryPoint,
    tags: entry.tags.slice(),
    revision: entry.revision,
    registeredAt: entry.registeredAt,
  };
}

/**
 * Serialize shader entries into a binary `.wgsl.pack` `ArrayBuffer` matching
 * `write_pack` / `write_pack_with_layout` in `shaders/src/pack.rs` byte-for-byte.
 *
 * @param {Array<{ hash: bigint | string | number, stage: number | string, wgsl: string | Uint8Array }>} entries
 * @param {{ flags?: number }} [options]
 * @returns {ArrayBuffer}
 */
export function buildWgslPackBuffer(entries, options = {}) {
  if (!Array.isArray(entries)) {
    throw new TypeError('entries must be an array');
  }
  const flags = options.flags === undefined ? 0 : options.flags;
  if (!Number.isInteger(flags) || flags < 0 || flags > U32_MAX) {
    throw new RangeError(`flags must be a uint32 integer, got: ${flags}`);
  }
  if (entries.length > U32_MAX) {
    throw new RangeError('entry count exceeds u32::MAX');
  }

  const encodedEntries = new Array(entries.length);
  const dataStart = WGPK_HEADER_BYTES + entries.length * WGPK_ENTRY_BYTES;
  let totalPayloadBytes = 0;

  for (let i = 0; i < entries.length; i += 1) {
    const item = entries[i];
    if (!item || typeof item !== 'object') {
      throw new TypeError(`Entry ${i} must be an object`);
    }
    const hash = normalizeHash(item.hash);
    const stageCode = resolveStageCode(item.stage);
    const payload =
      typeof item.wgsl === 'string'
        ? textEncoder.encode(item.wgsl)
        : toBinaryView(item.wgsl);

    if (payload.byteLength > U32_MAX) {
      throw new RangeError(`Entry ${i} payload exceeds u32::MAX bytes`);
    }
    totalPayloadBytes += payload.byteLength;
    encodedEntries[i] = { hash, stageCode, payload };
  }

  const totalSize = dataStart + totalPayloadBytes;
  if (totalSize > U32_MAX) {
    throw new RangeError('Total WGPK pack size exceeds u32::MAX byte offsets');
  }

  const buffer = new ArrayBuffer(totalSize);
  const out = new Uint8Array(buffer);
  const view = new DataView(buffer);

  // Header (16 bytes)
  out[0] = WGPK_MAGIC_BYTES[0];
  out[1] = WGPK_MAGIC_BYTES[1];
  out[2] = WGPK_MAGIC_BYTES[2];
  out[3] = WGPK_MAGIC_BYTES[3];
  view.setUint32(4, WGPK_VERSION >>> 0, true);
  view.setUint32(8, encodedEntries.length >>> 0, true);
  view.setUint32(12, flags >>> 0, true);

  // Entry table (20 bytes each) + payloads
  let cursor = dataStart;
  for (let i = 0; i < encodedEntries.length; i += 1) {
    const { hash, stageCode, payload } = encodedEntries[i];
    const base = WGPK_HEADER_BYTES + i * WGPK_ENTRY_BYTES;

    view.setBigUint64(base, hash, true);
    out[base + 8] = stageCode;
    out[base + 9] = 0;
    out[base + 10] = 0;
    out[base + 11] = 0;
    view.setUint32(base + 12, cursor >>> 0, true);
    view.setUint32(base + 16, payload.byteLength >>> 0, true);

    out.set(payload, cursor);
    cursor += payload.byteLength;
  }

  return buffer;
}

/**
 * Parsed view over a single `.wgsl.pack` archive with optional live dynamic WGSL overlays.
 */
export class WgslPack {
  /**
   * @param {object} init
   */
  constructor({
    version,
    flags,
    byteLength,
    dataStart,
    payloadBytes,
    lazyDecode,
    entryList,
    byHash,
    byHex,
    stageCounts,
    getDecodedCount,
    mode = 'buffer',
    url = null,
    fetchImpl = null,
    fetchInit = undefined,
    maxCachedWgslBytes = DEFAULT_MAX_CACHED_WGSL_BYTES,
    maxConcurrentRangeFetches = DEFAULT_MAX_CONCURRENT_RANGE_FETCHES,
    maxRangeFetchRetries = DEFAULT_MAX_RANGE_FETCH_RETRIES,
    rangeRetryDelayMs = DEFAULT_RANGE_RETRY_DELAY_MS,
    sleepFn = null,
    prefetchMaxGapBytes = DEFAULT_PREFETCH_MAX_GAP_BYTES,
    prefetchMaxSpanBytes = DEFAULT_PREFETCH_MAX_SPAN_BYTES,
    initialRangeFetches = 0,
    initialRangeBytesFetched = 0,
    initialRangeFetchRetries = 0,
    wgslCache = null,
    rangedDecodeCounter = null,
  }) {
    this.version = version;
    this.flags = flags;
    this.byteLength = byteLength;
    this.dataStart = dataStart;
    this.payloadBytes = payloadBytes;
    this.lazyDecode = lazyDecode;
    this.mode = mode;
    this.url = url;
    this.entries = byHash;
    this._byHex = byHex;
    this._entryList = entryList;
    this._stageCounts = stageCounts;
    this._getDecodedCount = getDecodedCount;
    this._dynamicEntries = new Map();
    this._dynamicHits = 0;

    // Ranged streaming state (bounded LRU cache, FIFO concurrency semaphore, transient retry,
    // coalesced range prefetch, and concurrent in-flight deduplication).
    this._fetch = fetchImpl;
    this._fetchInit = fetchInit;
    this._maxCachedWgslBytes =
      mode === 'ranged' ? validateMaxCachedWgslBytes(maxCachedWgslBytes) : null;
    this._maxConcurrentRangeFetches =
      mode === 'ranged' ? validateMaxConcurrentRangeFetches(maxConcurrentRangeFetches) : null;
    this._maxRangeFetchRetries =
      mode === 'ranged' ? validateMaxRangeFetchRetries(maxRangeFetchRetries) : null;
    this._rangeRetryDelayMs =
      mode === 'ranged' ? validateRangeRetryDelayMs(rangeRetryDelayMs) : null;
    this._sleepFn = mode === 'ranged' ? validateSleepFn(sleepFn) : null;
    this._prefetchMaxGapBytes =
      mode === 'ranged' ? validatePrefetchMaxGapBytes(prefetchMaxGapBytes) : null;
    this._prefetchMaxSpanBytes =
      mode === 'ranged' ? validatePrefetchMaxSpanBytes(prefetchMaxSpanBytes) : null;
    this._wgslCache = wgslCache || new Map();
    this._cachedWgslBytes = 0;
    this._mruKey = undefined;
    this._inFlightFetches = new Map();
    this._rangeFetchWaiters = [];
    this._activeRangeFetches = 0;
    this._peakConcurrentRangeFetches = 0;
    this._rangeFetches = sanitizeNonNegativeInt(initialRangeFetches, 0);
    this._rangeBytesFetched = sanitizeNonNegativeInt(initialRangeBytesFetched, 0);
    this._rangeFetchRetries = sanitizeNonNegativeInt(initialRangeFetchRetries, 0);
    this._cacheHits = 0;
    this._cacheMisses = 0;
    this._lruHits = 0;
    this._lruMisses = 0;
    this._wgslCacheEvictions = 0;
    this._prefetchBatches = 0;
    this._prefetchShaders = 0;
    this._inFlightCoalesced = 0;
    this._rangedDecodeCounter = rangedDecodeCounter;
  }

  /**
   * Parse a full in-memory `.wgsl.pack` buffer (alias for `parseWgslPack`).
   *
   * @param {ArrayBuffer | SharedArrayBuffer | ArrayBufferView} buffer
   * @param {{ lazyDecode?: boolean }} [options]
   * @returns {WgslPack}
   */
  static loadFromBuffer(buffer, options = {}) {
    return parseWgslPack(buffer, options);
  }

  /**
   * Load only the WGPK header + entry table via HTTP `206 Range` requests.
   * Seamlessly falls back to full in-memory buffer mode if the server responds `200 OK`.
   *
   * @param {string} url
   * @param {typeof fetch | object} [fetchImplOrOptions]
   * @param {object} [maybeOptions]
   * @returns {Promise<WgslPack>}
   */
  static loadIndexFromUrl(url, fetchImplOrOptions, maybeOptions) {
    return loadIndexFromUrl(url, fetchImplOrOptions, maybeOptions);
  }

  /**
   * Load a `.wgsl.pack` from a URL. When `options.streamThresholdBytes > 0`, automatically
   * switches to HTTP Range index-only mode when the remote pack exceeds the threshold.
   *
   * @param {string} url
   * @param {typeof fetch | object} [fetchImplOrOptions]
   * @param {object} [maybeOptions]
   * @returns {Promise<WgslPack>}
   */
  static loadFromUrl(url, fetchImplOrOptions, maybeOptions) {
    return loadFromUrl(url, fetchImplOrOptions, maybeOptions);
  }

  /** Number of distinct resolvable shader entries (static + non-overriding dynamic). */
  get size() {
    let extraDynamic = 0;
    for (const key of this._dynamicEntries.keys()) {
      if (!this.entries.has(key)) {
        extraDynamic += 1;
      }
    }
    return this._entryList.length + extraDynamic;
  }

  /**
   * Register a live dynamic WGSL shader on this pack instance.
   * Pass `{ override: true }` in `meta` to supersede an existing static pack entry
   * while preserving the static entry for `clearDynamicShaders()`.
   *
   * @param {bigint | string | number} hash
   * @param {number | string} stage
   * @param {string} wgslSource
   * @param {{ override?: boolean, label?: string, entryPoint?: string, source?: string, tags?: string[] }} [meta]
   */
  registerDynamicShader(hash, stage, wgslSource, meta) {
    const key = normalizeHash(hash);
    const hasStatic = this.entries.has(key);
    const prevDynamic = this._dynamicEntries.get(key);
    const entry = createDynamicShaderEntry(hash, stage, wgslSource, meta, prevDynamic);

    if (hasStatic && !entry.override) {
      throw new Error(
        `Shader ${entry.hashHex} already exists in static WGSL pack; pass { override: true } to supersede it`
      );
    }

    this._dynamicEntries.set(key, entry);
    return formatDynamicLookupRecord(entry, hasStatic);
  }

  /**
   * Check whether a dynamic shader is registered for `hash` (and optional `stage`).
   *
   * @param {bigint | string | number} hash
   * @param {number | string} [stage]
   * @returns {boolean}
   */
  hasDynamicShader(hash, stage) {
    const key = normalizeHash(hash);
    const entry = this._dynamicEntries.get(key);
    if (!entry) {
      return false;
    }
    if (stage !== undefined && stage !== null) {
      return entry.stageCode === resolveStageCode(stage);
    }
    return true;
  }

  /**
   * Remove a single dynamic shader by hash, restoring any underlying static pack entry.
   *
   * @param {bigint | string | number} hash
   * @returns {boolean}
   */
  unregisterDynamicShader(hash) {
    const key = normalizeHash(hash);
    return this._dynamicEntries.delete(key);
  }

  /**
   * Remove all registered dynamic shaders while preserving static pack entries.
   *
   * @returns {number} Number of dynamic shaders cleared
   */
  clearDynamicShaders() {
    const count = this._dynamicEntries.size;
    this._dynamicEntries.clear();
    return count;
  }

  /**
   * Look up a shader entry by hash (and optional stage), checking dynamic overrides first
   * and falling back to static pack entries.
   *
   * @param {bigint | string | number} hash
   * @param {number | string} [stage]
   */
  lookup(hash, stage) {
    const key = normalizeHash(hash);
    const expectedStage =
      stage !== undefined && stage !== null ? resolveStageCode(stage) : null;

    const dynamicEntry = this._dynamicEntries.get(key);
    if (dynamicEntry) {
      if (expectedStage !== null && dynamicEntry.stageCode !== expectedStage) {
        return undefined;
      }
      this._dynamicHits += 1;
      return formatDynamicLookupRecord(dynamicEntry, this.entries.has(key));
    }

    const staticEntry = this.entries.get(key);
    if (!staticEntry) {
      return undefined;
    }
    if (expectedStage !== null && staticEntry.stageCode !== expectedStage) {
      return undefined;
    }
    return {
      hash: staticEntry.hash,
      hashHex: staticEntry.hashHex,
      stage: staticEntry.stage,
      stageCode: staticEntry.stageCode,
      wgsl: staticEntry.wgsl,
      length: staticEntry.length,
      offset: staticEntry.offset,
      dynamic: false,
      overridesStatic: false,
      packName: null,
      source: 'pack',
      label: null,
      entryPoint: null,
      tags: [],
      revision: 0,
      registeredAt: null,
    };
  }

  /**
   * Alias for `lookup(hash, stage)`.
   *
   * @param {bigint | string | number} hash
   * @param {number | string} [stage]
   */
  get(hash, stage) {
    return this.lookup(hash, stage);
  }

  /**
   * Look up a shader entry by `BigInt` hash, hex string, or safe integer.
   * Checks dynamic overrides first, then static pack entries.
   *
   * @param {bigint | string | number} hash
   */
  getByHash(hash) {
    if (this._dynamicEntries.size > 0) {
      const key = normalizeHash(hash);
      const dynamicEntry = this._dynamicEntries.get(key);
      if (dynamicEntry) {
        this._dynamicHits += 1;
        return formatDynamicLookupRecord(dynamicEntry, this.entries.has(key));
      }
      return this.entries.get(key);
    }
    if (typeof hash === 'string') {
      const fast = this._byHex.get(hash);
      if (fast !== undefined) {
        return fast;
      }
    }
    const key = normalizeHash(hash);
    return this.entries.get(key);
  }

  /**
   * Retrieve the UTF-8 WGSL source string for a shader hash.
   * Checks dynamic overrides first; in `lazyDecode` mode, decodes static entries on first access.
   *
   * @param {bigint | string | number} hash
   * @returns {string | undefined}
   */
  getSource(hash) {
    const entry = this.getByHash(hash);
    return entry ? entry.wgsl : undefined;
  }

  /**
   * Alias for `getSource(hash)`.
   *
   * @param {bigint | string | number} hash
   * @returns {string | undefined}
   */
  getWgsl(hash) {
    return this.getSource(hash);
  }

  _resolveHashKey(hash) {
    if (typeof hash === 'bigint') {
      if (hash < 0n || hash > U64_MAX) {
        throw new RangeError(`Shader hash out of u64 range: ${hash.toString()}`);
      }
      return hash;
    }
    if (typeof hash === 'string') {
      const fast = this._byHex.get(hash);
      if (fast !== undefined) {
        return fast.hash;
      }
    }
    return normalizeHash(hash);
  }

  _touchWgslCache(key) {
    const cached = this._wgslCache.get(key);
    if (cached === undefined) {
      return undefined;
    }
    if (this._mruKey !== key && this._wgslCache.size > 1) {
      this._wgslCache.delete(key);
      this._wgslCache.set(key, cached);
    }
    this._mruKey = key;
    return cached;
  }

  _putWgslCache(key, wgsl, byteLength) {
    const existing = this._wgslCache.get(key);
    if (existing !== undefined) {
      this._wgslCache.delete(key);
      this._cachedWgslBytes -= existing.byteLength;
      if (this._mruKey === key) {
        this._mruKey = undefined;
      }
    }

    const budget = this._maxCachedWgslBytes;
    if (budget === null || budget === 0 || byteLength > budget) {
      return false;
    }

    while (this._wgslCache.size > 0 && this._cachedWgslBytes + byteLength > budget) {
      const oldestKey = this._wgslCache.keys().next().value;
      const oldest = this._wgslCache.get(oldestKey);
      this._wgslCache.delete(oldestKey);
      this._cachedWgslBytes -= oldest.byteLength;
      this._wgslCacheEvictions += 1;
      if (this._mruKey === oldestKey) {
        this._mruKey = undefined;
      }
    }

    this._wgslCache.set(key, { wgsl, byteLength });
    this._cachedWgslBytes += byteLength;
    this._mruKey = key;
    return true;
  }

  /**
   * Clear all decoded WGSL strings cached in the ranged-mode LRU cache.
   * Does not unload the entry table or dynamic shaders.
   *
   * @returns {number} Number of cached WGSL entries cleared
   */
  clearWgslCache() {
    const count = this._wgslCache.size;
    this._wgslCache.clear();
    this._cachedWgslBytes = 0;
    this._mruKey = undefined;
    return count;
  }

  _acquireRangeFetchSlot() {
    const limit = this._maxConcurrentRangeFetches;
    if (limit === null || limit === 0 || this._activeRangeFetches < limit) {
      this._activeRangeFetches += 1;
      if (this._activeRangeFetches > this._peakConcurrentRangeFetches) {
        this._peakConcurrentRangeFetches = this._activeRangeFetches;
      }
      return null;
    }
    return new Promise((resolve) => {
      this._rangeFetchWaiters.push(resolve);
    });
  }

  _releaseRangeFetchSlot() {
    const limit = this._maxConcurrentRangeFetches;
    if (limit !== null && limit > 0 && this._rangeFetchWaiters.length > 0) {
      // Transfer the slot directly to the next FIFO waiter so _activeRangeFetches never
      // exceeds the configured limit and new arrivals cannot jump ahead in the queue.
      const next = this._rangeFetchWaiters.shift();
      next();
      return;
    }
    if (this._activeRangeFetches > 0) {
      this._activeRangeFetches -= 1;
    }
  }

  async _withRangeFetchSlot(fn) {
    const waitPromise = this._acquireRangeFetchSlot();
    if (waitPromise !== null) {
      await waitPromise;
    }
    try {
      return await fn();
    } finally {
      this._releaseRangeFetchSlot();
    }
  }

  async _fetchRangeChunkBounded(fetcher, start, end) {
    const maxRetries = this._maxRangeFetchRetries ?? 0;
    const baseDelayMs = this._rangeRetryDelayMs ?? DEFAULT_RANGE_RETRY_DELAY_MS;
    const sleep = this._sleepFn || defaultSleep;

    let attempt = 0;
    for (;;) {
      try {
        return await this._withRangeFetchSlot(() => {
          this._rangeFetches += 1;
          return fetchRangeChunk(fetcher, this.url, start, end, this._fetchInit);
        });
      } catch (err) {
        if (attempt >= maxRetries || !isRetryableRangeFetchError(err, this._fetchInit)) {
          throw err;
        }
        this._rangeFetchRetries += 1;
        const delayMs = computeRangeRetryDelayMs(baseDelayMs, attempt);
        attempt += 1;
        await sleep(delayMs);
        if (this._fetchInit && this._fetchInit.signal && this._fetchInit.signal.aborted) {
          throw err;
        }
      }
    }
  }

  /**
   * Asynchronously resolve the WGSL source string for a shader hash.
   * Works in both full-buffer mode (returns immediately) and ranged index mode
   * (checks bounded LRU cache, deduplicates concurrent in-flight requests, enforces
   * bounded HTTP Range concurrency + transient retry, and fetches only the exact
   * `[offset, offset + length - 1]` byte slice via HTTP `206 Range`).
   *
   * @param {bigint | string | number} hash
   * @returns {Promise<string | null>}
   */
  async getWgslAsync(hash) {
    let key;
    let staticEntry;
    if (this._dynamicEntries.size === 0 && typeof hash === 'string') {
      staticEntry = this._byHex.get(hash);
      if (staticEntry !== undefined) {
        key = staticEntry.hash;
      } else {
        key = normalizeHash(hash);
        staticEntry = this.entries.get(key);
      }
    } else {
      key = this._resolveHashKey(hash);
      const dynamicEntry = this._dynamicEntries.get(key);
      if (dynamicEntry) {
        this._dynamicHits += 1;
        return dynamicEntry.wgsl;
      }
      staticEntry = this.entries.get(key);
    }

    if (!staticEntry) {
      return null;
    }

    if (this.mode !== 'ranged') {
      this._cacheHits += 1;
      return staticEntry.wgsl;
    }

    // Check bounded in-memory hot-hash LRU cache first.
    const cached = this._touchWgslCache(key);
    if (cached !== undefined) {
      this._cacheHits += 1;
      this._lruHits += 1;
      return cached.wgsl;
    }

    // Zero-length payload needs no HTTP Range request.
    const length = staticEntry.length >>> 0;
    if (length === 0) {
      this._cacheMisses += 1;
      this._lruMisses += 1;
      this._putWgslCache(key, '', 0);
      return '';
    }

    // Deduplicate concurrent in-flight fetches (including active coalesced prefetch batches)
    // before acquiring a range fetch slot.
    const existingPromise = this._inFlightFetches.get(key);
    if (existingPromise !== undefined) {
      this._inFlightCoalesced += 1;
      return existingPromise;
    }

    this._cacheMisses += 1;
    this._lruMisses += 1;
    const fetcher = this._fetch ?? globalThis.fetch;
    if (typeof fetcher !== 'function') {
      throw new Error('No fetch implementation available for ranged getWgslAsync');
    }

    // Use unsigned 32-bit offsets + safe IEEE-754 float64 addition so offsets past 2 GiB
    // (0x80000000..0xFFFFFFFF) never wrap negative.
    const start = staticEntry.offset >>> 0;
    const end = start + length - 1;

    let settled = false;
    let fetchPromise;
    fetchPromise = (async () => {
      try {
        const chunk = await this._fetchRangeChunkBounded(fetcher, start, end);
        this._rangeBytesFetched += sanitizeNonNegativeInt(chunk.bytes.byteLength, 0);

        let payloadBytes;
        if (chunk.status === 206) {
          if (chunk.contentRange && chunk.contentRange.start !== start) {
            throw new Error(
              `WGPK ranged fetch mismatch for ${staticEntry.hashHex}: expected start byte ${start}, got ${chunk.contentRange.start}`
            );
          }
          if (chunk.bytes.byteLength < length) {
            throw new Error(
              `WGPK ranged payload truncated for ${staticEntry.hashHex}: expected ${length} bytes, received ${chunk.bytes.byteLength}`
            );
          }
          payloadBytes = chunk.bytes.subarray(0, length);
        } else {
          // Server returned 200 OK (entire pack) instead of 206 Partial Content.
          const sliceEnd = start + length;
          if (chunk.bytes.byteLength < sliceEnd) {
            throw new Error(
              `WGPK full response truncated for ${staticEntry.hashHex}: need byte range [${start}..${sliceEnd}), received ${chunk.bytes.byteLength} bytes`
            );
          }
          payloadBytes = chunk.bytes.subarray(start, sliceEnd);
        }

        const entryIdx =
          staticEntry._index !== undefined ? staticEntry._index : staticEntry.hashHex;
        const wgsl = decodeUtf8Strict(payloadBytes, entryIdx);
        if (this._rangedDecodeCounter) {
          this._rangedDecodeCounter.count += 1;
        }
        this._putWgslCache(key, wgsl, length);
        return wgsl;
      } finally {
        settled = true;
        if (this._inFlightFetches.get(key) === fetchPromise) {
          this._inFlightFetches.delete(key);
        }
      }
    })();

    if (!settled) {
      this._inFlightFetches.set(key, fetchPromise);
    }
    return fetchPromise;
  }

  /**
   * Prefetch a set of shader hashes into the bounded hot-hash WGSL LRU cache by coalescing
   * nearby byte ranges (within `maxGapBytes` and `maxSpanBytes`) into batched HTTP `206 Range`
   * fetches.
   *
   * Already-cached entries (and dynamic overrides) are refreshed/skipped with zero HTTP traffic;
   * hashes already in flight via `getWgslAsync()` or a concurrent `prefetchHashes()` call are
   * deduplicated onto their existing promises.
   *
   * @param {Iterable<bigint | string | number>} hashes
   * @param {{
   *   maxGapBytes?: number,
   *   maxSpanBytes?: number,
   *   prefetchMaxGapBytes?: number,
   *   prefetchMaxSpanBytes?: number,
   * }} [options]
   * @returns {Promise<{
   *   requested: number,
   *   prefetched: number,
   *   cached: number,
   *   inFlight: number,
   *   missing: number,
   *   batches: number,
   *   bytesFetched: number,
   * }>}
   */
  async prefetchHashes(hashes, options = {}) {
    const opts = options && typeof options === 'object' ? options : {};
    const maxGapBytes = validatePrefetchMaxGapBytes(
      opts.maxGapBytes !== undefined ? opts.maxGapBytes : opts.prefetchMaxGapBytes,
      this._prefetchMaxGapBytes !== null
        ? this._prefetchMaxGapBytes
        : DEFAULT_PREFETCH_MAX_GAP_BYTES
    );
    const maxSpanBytes = validatePrefetchMaxSpanBytes(
      opts.maxSpanBytes !== undefined ? opts.maxSpanBytes : opts.prefetchMaxSpanBytes,
      this._prefetchMaxSpanBytes !== null
        ? this._prefetchMaxSpanBytes
        : DEFAULT_PREFETCH_MAX_SPAN_BYTES
    );

    if (hashes === undefined || hashes === null) {
      return {
        requested: 0,
        prefetched: 0,
        cached: 0,
        inFlight: 0,
        missing: 0,
        batches: 0,
        bytesFetched: 0,
      };
    }
    if (typeof hashes[Symbol.iterator] !== 'function') {
      throw new TypeError('hashes must be an iterable of shader hashes');
    }

    const seen = new Set();
    let requested = 0;
    let cached = 0;
    let inFlight = 0;
    let missing = 0;
    const toFetch = [];
    const inFlightPromises = [];

    for (const rawHash of hashes) {
      const key = this._resolveHashKey(rawHash);
      if (seen.has(key)) {
        continue;
      }
      seen.add(key);
      requested += 1;

      if (this._dynamicEntries.has(key)) {
        cached += 1;
        continue;
      }

      const staticEntry = this.entries.get(key);
      if (!staticEntry) {
        missing += 1;
        continue;
      }

      if (this.mode !== 'ranged') {
        void staticEntry.wgsl;
        cached += 1;
        continue;
      }

      if (this._touchWgslCache(key) !== undefined) {
        cached += 1;
        continue;
      }

      const length = staticEntry.length >>> 0;
      if (length === 0) {
        this._putWgslCache(key, '', 0);
        cached += 1;
        continue;
      }

      const existingPromise = this._inFlightFetches.get(key);
      if (existingPromise !== undefined) {
        this._inFlightCoalesced += 1;
        inFlight += 1;
        inFlightPromises.push(existingPromise);
        continue;
      }

      const start = staticEntry.offset >>> 0;
      toFetch.push({
        key,
        entry: staticEntry,
        start,
        length,
        end: start + length,
      });
    }

    if (toFetch.length === 0) {
      if (inFlightPromises.length > 0) {
        await Promise.all(inFlightPromises);
      }
      return {
        requested,
        prefetched: 0,
        cached,
        inFlight,
        missing,
        batches: 0,
        bytesFetched: 0,
      };
    }

    const fetcher = this._fetch ?? globalThis.fetch;
    if (typeof fetcher !== 'function') {
      throw new Error('No fetch implementation available for ranged prefetchHashes');
    }

    // Sort by ascending byte offset and coalesce nearby ranges within maxGapBytes & maxSpanBytes.
    toFetch.sort((a, b) => a.start - b.start);
    const batches = [];
    let current = {
      start: toFetch[0].start,
      end: toFetch[0].end,
      items: [toFetch[0]],
    };
    for (let i = 1; i < toFetch.length; i += 1) {
      const item = toFetch[i];
      const gap = item.start - current.end;
      const proposedSpan = item.end - current.start;
      if (gap <= maxGapBytes && proposedSpan <= maxSpanBytes) {
        current.end = item.end;
        current.items.push(item);
      } else {
        batches.push(current);
        current = {
          start: item.start,
          end: item.end,
          items: [item],
        };
      }
    }
    batches.push(current);

    let prefetched = 0;
    let completedBatches = 0;
    let bytesFetched = 0;

    // Register per-hash in-flight promises synchronously BEFORE awaiting any batch so concurrent
    // getWgslAsync() or prefetchHashes() calls deduplicate cleanly onto the coalesced batch.
    const batchPromises = batches.map((batch) => {
      const deferreds = batch.items.map((item) => {
        let resolve;
        let reject;
        const promise = new Promise((res, rej) => {
          resolve = res;
          reject = rej;
        });
        // Prevent unhandled rejection noise if the batch fails and only prefetchHashes() catches.
        promise.catch(() => {});
        this._inFlightFetches.set(item.key, promise);
        return { item, promise, resolve, reject };
      });

      const clearInFlight = () => {
        for (let i = 0; i < deferreds.length; i += 1) {
          const { item, promise } = deferreds[i];
          if (this._inFlightFetches.get(item.key) === promise) {
            this._inFlightFetches.delete(item.key);
          }
        }
      };

      return (async () => {
        let chunk;
        try {
          chunk = await this._fetchRangeChunkBounded(fetcher, batch.start, batch.end - 1);
        } catch (err) {
          clearInFlight();
          for (let i = 0; i < deferreds.length; i += 1) {
            deferreds[i].reject(err);
          }
          throw err;
        }

        const chunkBytes = sanitizeNonNegativeInt(chunk.bytes.byteLength, 0);
        this._rangeBytesFetched += chunkBytes;
        bytesFetched += chunkBytes;

        const expectedSpan = batch.end - batch.start;
        try {
          if (chunk.status === 206) {
            if (chunk.contentRange && chunk.contentRange.start !== batch.start) {
              throw new Error(
                `WGPK ranged prefetch mismatch: expected start byte ${batch.start}, got ${chunk.contentRange.start}`
              );
            }
            if (chunk.bytes.byteLength < expectedSpan) {
              throw new Error(
                `WGPK ranged prefetch truncated: expected ${expectedSpan} bytes for [${batch.start}..${batch.end}), received ${chunk.bytes.byteLength}`
              );
            }
          } else if (chunk.bytes.byteLength < batch.end) {
            throw new Error(
              `WGPK full prefetch response truncated: need byte range [${batch.start}..${batch.end}), received ${chunk.bytes.byteLength} bytes`
            );
          }
        } catch (err) {
          clearInFlight();
          for (let i = 0; i < deferreds.length; i += 1) {
            deferreds[i].reject(err);
          }
          throw err;
        }

        const decodedResults = new Array(deferreds.length);
        let firstDecodeError = null;
        for (let i = 0; i < deferreds.length; i += 1) {
          const { item } = deferreds[i];
          const relStart = chunk.status === 206 ? item.start - batch.start : item.start;
          const payloadBytes = chunk.bytes.subarray(relStart, relStart + item.length);
          const entryIdx =
            item.entry._index !== undefined ? item.entry._index : item.entry.hashHex;
          try {
            const wgsl = decodeUtf8Strict(payloadBytes, entryIdx);
            if (this._rangedDecodeCounter) {
              this._rangedDecodeCounter.count += 1;
            }
            this._putWgslCache(item.key, wgsl, item.length);
            prefetched += 1;
            this._prefetchShaders += 1;
            decodedResults[i] = { ok: true, wgsl };
          } catch (decodeErr) {
            if (firstDecodeError === null) {
              firstDecodeError = decodeErr;
            }
            decodedResults[i] = { ok: false, err: decodeErr };
          }
        }

        this._prefetchBatches += 1;
        completedBatches += 1;
        clearInFlight();

        for (let i = 0; i < deferreds.length; i += 1) {
          const res = decodedResults[i];
          if (res.ok) {
            deferreds[i].resolve(res.wgsl);
          } else {
            deferreds[i].reject(res.err);
          }
        }

        if (firstDecodeError !== null) {
          throw firstDecodeError;
        }
      })();
    });

    await Promise.all([...batchPromises, ...inFlightPromises]);

    return {
      requested,
      prefetched,
      cached,
      inFlight,
      missing,
      batches: completedBatches,
      bytesFetched,
    };
  }

  /**
   * Check whether a shader hash exists in dynamic overrides or this static pack
   * without triggering lazy UTF-8 decode.
   *
   * @param {bigint | string | number} hash
   * @returns {boolean}
   */
  has(hash) {
    if (this._dynamicEntries.size > 0) {
      const key = normalizeHash(hash);
      if (this._dynamicEntries.has(key)) {
        return true;
      }
      return this.entries.has(key);
    }
    if (typeof hash === 'string') {
      if (this._byHex.has(hash)) {
        return true;
      }
    }
    const key = normalizeHash(hash);
    return this.entries.has(key);
  }

  /**
   * Return all static shader entries in their original pack table order.
   * Does not trigger lazy UTF-8 decoding until `.wgsl` is accessed on an entry.
   */
  listEntries() {
    return this._entryList.slice();
  }

  /**
   * Return metadata summaries for all dynamically registered shaders on this pack.
   *
   * @param {{ stage?: number | string, includeSource?: boolean }} [options]
   */
  listDynamicShaders(options = {}) {
    const expectedStage =
      options && options.stage !== undefined && options.stage !== null
        ? resolveStageCode(options.stage)
        : null;
    const includeSource = Boolean(options && options.includeSource);

    const results = [];
    for (const entry of this._dynamicEntries.values()) {
      if (expectedStage !== null && entry.stageCode !== expectedStage) {
        continue;
      }
      const item = {
        hash: entry.hash,
        hashHex: entry.hashHex,
        stage: entry.stage,
        stageCode: entry.stageCode,
        length: entry.length,
        dynamic: true,
        overridesStatic: this.entries.has(entry.hash),
        source: entry.source,
        label: entry.label,
        entryPoint: entry.entryPoint,
        tags: entry.tags.slice(),
        revision: entry.revision,
        registeredAt: entry.registeredAt,
      };
      if (includeSource) {
        item.wgsl = entry.wgsl;
      }
      results.push(item);
    }
    return results;
  }

  /**
   * Return structural, decode, and dynamic overlay statistics for this pack.
   */
  stats() {
    const entryCount = this._entryList.length;
    let dynamicOverrideCount = 0;
    let dynamicPayloadBytes = 0;
    const dynamicStageCounts = { vertex: 0, fragment: 0, compute: 0 };

    for (const entry of this._dynamicEntries.values()) {
      if (this.entries.has(entry.hash)) {
        dynamicOverrideCount += 1;
      }
      dynamicPayloadBytes += entry.length;
      dynamicStageCounts[entry.stage] += 1;
    }

    const rangeFetchCount = sanitizeNonNegativeInt(this._rangeFetches, 0);
    const rangeBytesFetched = sanitizeNonNegativeInt(this._rangeBytesFetched, 0);
    const rangeFetchRetries = sanitizeNonNegativeInt(this._rangeFetchRetries, 0);
    const activeRangeFetches = sanitizeNonNegativeInt(this._activeRangeFetches, 0);
    const peakConcurrentRangeFetches = sanitizeNonNegativeInt(this._peakConcurrentRangeFetches, 0);
    const cacheHits = sanitizeNonNegativeInt(this._cacheHits, 0);
    const cacheMisses = sanitizeNonNegativeInt(this._cacheMisses, 0);
    const lruHits = sanitizeNonNegativeInt(this._lruHits, 0);
    const lruMisses = sanitizeNonNegativeInt(this._lruMisses, 0);
    const cachedWgslBytes = sanitizeNonNegativeInt(this._cachedWgslBytes, 0);
    const lruBytes = cachedWgslBytes;
    const cachedWgslEntries = sanitizeNonNegativeInt(this._wgslCache.size, 0);
    const lruEvictions = sanitizeNonNegativeInt(this._wgslCacheEvictions, 0);
    const prefetchBatches = sanitizeNonNegativeInt(this._prefetchBatches, 0);
    const prefetchShaders = sanitizeNonNegativeInt(this._prefetchShaders, 0);
    const inFlightCoalesced = sanitizeNonNegativeInt(this._inFlightCoalesced, 0);
    const inFlightCount = sanitizeNonNegativeInt(this._inFlightFetches.size, 0);

    return {
      version: this.version,
      flags: this.flags,
      mode: this.mode,
      entryCount,
      byteLength: this.byteLength,
      headerBytes: WGPK_HEADER_BYTES,
      tableBytes: entryCount * WGPK_ENTRY_BYTES,
      payloadBytes: this.payloadBytes,
      decodedCount: this._getDecodedCount(),
      lazyDecode: this.lazyDecode,
      stageCounts: { ...this._stageCounts },
      dynamicCount: this._dynamicEntries.size,
      dynamicOverrideCount,
      dynamicPayloadBytes,
      dynamicHits: this._dynamicHits,
      dynamicStageCounts,
      rangeFetches: rangeFetchCount,
      rangeFetchCount,
      rangeBytesFetched,
      rangeFetchRetries,
      activeRangeFetches,
      peakConcurrentRangeFetches,
      maxConcurrentRangeFetches: this._maxConcurrentRangeFetches,
      maxRangeFetchRetries: this._maxRangeFetchRetries,
      cacheHits,
      cacheMisses,
      lruHits,
      lruMisses,
      lruBytes,
      cachedWgslBytes,
      cachedWgslEntries,
      maxCachedWgslBytes: this._maxCachedWgslBytes,
      wgslCacheEvictions: lruEvictions,
      lruEvictions,
      prefetchBatches,
      prefetchShaders,
      prefetchMaxGapBytes: this._prefetchMaxGapBytes,
      prefetchMaxSpanBytes: this._prefetchMaxSpanBytes,
      inFlightDeduplications: inFlightCoalesced,
      inFlightCoalesced,
      inFlightCount,
    };
  }

  /**
   * Alias for `stats()`.
   */
  getStats() {
    return this.stats();
  }
}

/**
 * Parse and validate a binary `.wgsl.pack` (`WGPK`) buffer.
 *
 * @param {ArrayBuffer | SharedArrayBuffer | ArrayBufferView} buffer
 * @param {{ lazyDecode?: boolean }} [options]
 * @returns {WgslPack}
 */
export function parseWgslPack(buffer, options = {}) {
  const bytes = toBinaryView(buffer);
  const lazyDecode = Boolean(options && options.lazyDecode);

  const header = parseWgpkHeader(bytes);
  const { rawEntries, stageCounts, totalPayloadBytes } = parseWgpkEntryTable(
    bytes,
    header,
    bytes.byteLength
  );

  const entryList = new Array(header.entryCount);
  const byHash = new Map();
  const byHex = new Map();
  let decodedCount = 0;

  for (let idx = 0; idx < rawEntries.length; idx += 1) {
    const raw = rawEntries[idx];
    const slice = bytes.subarray(raw.offset, raw.end);

    let entry;
    if (lazyDecode) {
      let cachedWgsl = null;
      entry = {
        hash: raw.hash,
        hashHex: raw.hashHex,
        stage: raw.stage,
        stageCode: raw.stageCode,
        offset: raw.offset,
        length: raw.length,
      };
      Object.defineProperty(entry, 'wgsl', {
        enumerable: true,
        configurable: true,
        get() {
          if (cachedWgsl === null) {
            cachedWgsl = decodeUtf8Strict(slice, idx);
            decodedCount += 1;
          }
          return cachedWgsl;
        },
      });
    } else {
      const wgsl = decodeUtf8Strict(slice, idx);
      decodedCount += 1;
      entry = {
        hash: raw.hash,
        hashHex: raw.hashHex,
        stage: raw.stage,
        stageCode: raw.stageCode,
        offset: raw.offset,
        length: raw.length,
        wgsl,
      };
    }

    entryList[idx] = entry;
    byHash.set(raw.hash, entry);
    byHex.set(raw.hashHex, entry);
  }

  return new WgslPack({
    version: header.version,
    flags: header.flags,
    byteLength: bytes.byteLength,
    dataStart: header.dataStart,
    payloadBytes: totalPayloadBytes,
    lazyDecode,
    entryList,
    byHash,
    byHex,
    stageCounts,
    getDecodedCount: () => decodedCount,
    mode: 'buffer',
  });
}

/**
 * Parse only the WGPK header + entry table into a ranged-mode `WgslPack` without
 * holding payload bytes in memory. Payloads are fetched on demand via `getWgslAsync(hash)`.
 *
 * @param {ArrayBuffer | SharedArrayBuffer | ArrayBufferView} indexBuffer
 * @param {{
 *   url?: string | null,
 *   totalBytes?: number | null,
 *   fetch?: typeof fetch,
 *   fetchInit?: RequestInit,
 *   maxCachedWgslBytes?: number,
 *   initialRangeFetches?: number,
 * }} [options]
 * @returns {WgslPack}
 */
export function parseWgslPackIndex(indexBuffer, options = {}) {
  const bytes = toBinaryView(indexBuffer);
  const totalBytes =
    options && options.totalBytes !== undefined && options.totalBytes !== null
      ? options.totalBytes
      : null;
  if (totalBytes !== null && (!Number.isSafeInteger(totalBytes) || totalBytes < 0)) {
    throw new RangeError(`totalBytes must be a non-negative safe integer, got: ${totalBytes}`);
  }

  const header = parseWgpkHeader(bytes);
  const { rawEntries, stageCounts, totalPayloadBytes, maxEnd } = parseWgpkEntryTable(
    bytes,
    header,
    totalBytes
  );

  const wgslCache = new Map();
  const rangedDecodeCounter = { count: 0 };
  const entryList = new Array(header.entryCount);
  const byHash = new Map();
  const byHex = new Map();

  for (let idx = 0; idx < rawEntries.length; idx += 1) {
    const raw = rawEntries[idx];
    const entry = {
      hash: raw.hash,
      hashHex: raw.hashHex,
      stage: raw.stage,
      stageCode: raw.stageCode,
      offset: raw.offset,
      length: raw.length,
    };
    Object.defineProperty(entry, '_index', {
      value: idx,
      enumerable: false,
      configurable: true,
    });
    Object.defineProperty(entry, 'wgsl', {
      enumerable: true,
      configurable: true,
      get() {
        const cached = wgslCache.get(raw.hash);
        if (cached !== undefined) {
          return cached.wgsl;
        }
        if (raw.length === 0) {
          return '';
        }
        throw new Error(
          `WGPK shader ${raw.hashHex} is not loaded locally in ranged mode; call getWgslAsync(hash) first`
        );
      },
    });

    entryList[idx] = entry;
    byHash.set(raw.hash, entry);
    byHex.set(raw.hashHex, entry);
  }

  const maxCachedWgslBytes = validateMaxCachedWgslBytes(
    options ? options.maxCachedWgslBytes : undefined
  );
  const maxConcurrentRangeFetches = validateMaxConcurrentRangeFetches(
    options ? options.maxConcurrentRangeFetches : undefined
  );
  const maxRangeFetchRetries = validateMaxRangeFetchRetries(
    options ? options.maxRangeFetchRetries : undefined
  );
  const rangeRetryDelayMs = validateRangeRetryDelayMs(
    options ? options.rangeRetryDelayMs : undefined
  );
  const sleepFn = validateSleepFn(options ? options.sleepFn : undefined);
  const prefetchMaxGapBytes = validatePrefetchMaxGapBytes(
    options ? options.prefetchMaxGapBytes : undefined
  );
  const prefetchMaxSpanBytes = validatePrefetchMaxSpanBytes(
    options ? options.prefetchMaxSpanBytes : undefined
  );
  const initialRangeFetches = sanitizeNonNegativeInt(
    options ? options.initialRangeFetches : 0,
    0
  );
  const initialRangeBytesFetched = sanitizeNonNegativeInt(
    options ? options.initialRangeBytesFetched : 0,
    0
  );
  const initialRangeFetchRetries = sanitizeNonNegativeInt(
    options ? options.initialRangeFetchRetries : 0,
    0
  );

  return new WgslPack({
    version: header.version,
    flags: header.flags,
    byteLength: totalBytes !== null ? totalBytes : Math.max(header.dataStart, maxEnd),
    dataStart: header.dataStart,
    payloadBytes: totalPayloadBytes,
    lazyDecode: true,
    entryList,
    byHash,
    byHex,
    stageCounts,
    getDecodedCount: () => rangedDecodeCounter.count,
    mode: 'ranged',
    url: options && options.url ? String(options.url) : null,
    fetchImpl: options && typeof options.fetch === 'function' ? options.fetch : null,
    fetchInit: options ? options.fetchInit : undefined,
    maxCachedWgslBytes,
    maxConcurrentRangeFetches,
    maxRangeFetchRetries,
    rangeRetryDelayMs,
    sleepFn,
    prefetchMaxGapBytes,
    prefetchMaxSpanBytes,
    initialRangeFetches,
    initialRangeBytesFetched,
    initialRangeFetchRetries,
    wgslCache,
    rangedDecodeCounter,
  });
}

async function finishPackIndexFromInitialChunk(url, fetcher, initialChunk, options = {}) {
  const lazyDecode = options && options.lazyDecode !== undefined ? Boolean(options.lazyDecode) : true;
  const maxCachedWgslBytes = validateMaxCachedWgslBytes(
    options ? options.maxCachedWgslBytes : undefined
  );
  const maxConcurrentRangeFetches = validateMaxConcurrentRangeFetches(
    options ? options.maxConcurrentRangeFetches : undefined
  );
  const maxRangeFetchRetries = validateMaxRangeFetchRetries(
    options ? options.maxRangeFetchRetries : undefined
  );
  const rangeRetryDelayMs = validateRangeRetryDelayMs(
    options ? options.rangeRetryDelayMs : undefined
  );
  const sleepFn = validateSleepFn(options ? options.sleepFn : undefined);
  const prefetchMaxGapBytes = validatePrefetchMaxGapBytes(
    options ? options.prefetchMaxGapBytes : undefined
  );
  const prefetchMaxSpanBytes = validatePrefetchMaxSpanBytes(
    options ? options.prefetchMaxSpanBytes : undefined
  );

  // If the server returned 200 OK instead of 206 Partial Content, fall back seamlessly to buffer mode.
  if (initialChunk.status !== 206) {
    return parseWgslPack(initialChunk.bytes, { lazyDecode });
  }

  if (initialChunk.contentRange && initialChunk.contentRange.start !== 0) {
    throw new Error(
      `WGPK initial range response from "${url}" started at byte ${initialChunk.contentRange.start} (expected 0)`
    );
  }

  let totalBytes = initialChunk.contentRange ? initialChunk.contentRange.total : null;
  let rangeFetches = sanitizeNonNegativeInt(
    options && options.initialRangeFetches !== undefined ? options.initialRangeFetches : 1,
    1
  );
  let rangeFetchRetries = sanitizeNonNegativeInt(
    options && options.initialRangeFetchRetries !== undefined
      ? options.initialRangeFetchRetries
      : 0,
    0
  );
  let rangeBytesFetched = sanitizeNonNegativeInt(initialChunk.bytes.byteLength, 0);

  const header = parseWgpkHeader(initialChunk.bytes);
  const tableEnd = header.dataStart;

  if (totalBytes !== null && totalBytes < tableEnd) {
    throw new Error(
      `WGPK pack truncated: ${totalBytes} total bytes (header + ${header.entryCount} entry table records require at least ${tableEnd} bytes)`
    );
  }

  let indexBytes;
  if (initialChunk.bytes.byteLength >= tableEnd) {
    // Entire header + entry table fit in the initial range chunk (single roundtrip!).
    // Copy only the exact index prefix so any trailing bytes are not retained in memory.
    indexBytes = initialChunk.bytes.slice(0, tableEnd);
  } else {
    // Table extends beyond the initial range chunk: fetch only the remaining table bytes.
    indexBytes = new Uint8Array(tableEnd);
    indexBytes.set(initialChunk.bytes, 0);
    let have = initialChunk.bytes.byteLength;

    while (have < tableEnd) {
      const remainder = await fetchRangeChunkWithRetry(
        fetcher,
        url,
        have,
        tableEnd - 1,
        options.fetchInit,
        {
          maxRangeFetchRetries,
          rangeRetryDelayMs,
          sleepFn,
          onAttempt: () => {
            rangeFetches += 1;
          },
          onRetry: () => {
            rangeFetchRetries += 1;
          },
        }
      );

      // Server fell back to 200 OK on the continuation request: switch seamlessly to full buffer mode.
      if (remainder.status !== 206) {
        return parseWgslPack(remainder.bytes, { lazyDecode });
      }

      rangeBytesFetched += sanitizeNonNegativeInt(remainder.bytes.byteLength, 0);

      if (remainder.contentRange) {
        if (remainder.contentRange.start !== have) {
          throw new Error(
            `WGPK table continuation range from "${url}" started at byte ${remainder.contentRange.start} (expected ${have})`
          );
        }
        if (remainder.contentRange.total !== null) {
          totalBytes = remainder.contentRange.total;
        }
      }

      if (remainder.bytes.byteLength === 0) {
        throw new Error(
          `WGPK pack truncated while fetching entry table from "${url}": received 0 bytes at offset ${have}/${tableEnd}`
        );
      }

      const copyLen = Math.min(remainder.bytes.byteLength, tableEnd - have);
      indexBytes.set(remainder.bytes.subarray(0, copyLen), have);
      have += copyLen;
    }
  }

  return parseWgslPackIndex(indexBytes, {
    url,
    totalBytes,
    fetch: fetcher,
    fetchInit: options.fetchInit,
    maxCachedWgslBytes,
    maxConcurrentRangeFetches,
    maxRangeFetchRetries,
    rangeRetryDelayMs,
    sleepFn,
    prefetchMaxGapBytes,
    prefetchMaxSpanBytes,
    initialRangeFetches: rangeFetches,
    initialRangeBytesFetched: rangeBytesFetched,
    initialRangeFetchRetries: rangeFetchRetries,
  });
}

/**
 * Fetch only the WGPK header + entry table from a URL using HTTP `206 Range` requests.
 * Falls back seamlessly to full in-memory buffer mode if the server responds `200 OK`.
 *
 * @param {string} url
 * @param {typeof fetch | {
 *   fetch?: typeof fetch,
 *   fetchInit?: RequestInit,
 *   initialRangeBytes?: number,
 *   maxCachedWgslBytes?: number,
 *   maxConcurrentRangeFetches?: number,
 *   maxRangeFetchRetries?: number,
 *   rangeRetryDelayMs?: number,
 *   sleepFn?: (ms: number) => Promise<void>,
 *   prefetchMaxGapBytes?: number,
 *   prefetchMaxSpanBytes?: number,
 *   lazyDecode?: boolean,
 * }} [fetchImplOrOptions]
 * @param {object} [maybeOptions]
 * @returns {Promise<WgslPack>}
 */
export async function loadIndexFromUrl(url, fetchImplOrOptions, maybeOptions) {
  if (typeof url !== 'string' || url.length === 0) {
    throw new TypeError('url must be a non-empty string');
  }
  const { fetcher, options } = resolveFetchAndOptions(fetchImplOrOptions, maybeOptions, null);
  if (typeof fetcher !== 'function') {
    throw new Error('No fetch implementation available for loadIndexFromUrl');
  }

  const initialRangeBytes = validateInitialRangeBytes(options.initialRangeBytes);
  validateMaxCachedWgslBytes(options.maxCachedWgslBytes);
  validateMaxConcurrentRangeFetches(options.maxConcurrentRangeFetches);
  const maxRangeFetchRetries = validateMaxRangeFetchRetries(options.maxRangeFetchRetries);
  const rangeRetryDelayMs = validateRangeRetryDelayMs(options.rangeRetryDelayMs);
  const sleepFn = validateSleepFn(options.sleepFn);
  validatePrefetchMaxGapBytes(options.prefetchMaxGapBytes);
  validatePrefetchMaxSpanBytes(options.prefetchMaxSpanBytes);

  let initialRangeFetches = 0;
  let initialRangeFetchRetries = 0;
  const initialChunk = await fetchRangeChunkWithRetry(
    fetcher,
    url,
    0,
    initialRangeBytes - 1,
    options.fetchInit,
    {
      maxRangeFetchRetries,
      rangeRetryDelayMs,
      sleepFn,
      onAttempt: () => {
        initialRangeFetches += 1;
      },
      onRetry: () => {
        initialRangeFetchRetries += 1;
      },
    }
  );
  return finishPackIndexFromInitialChunk(url, fetcher, initialChunk, {
    ...options,
    initialRangeFetches,
    initialRangeFetchRetries,
  });
}

/**
 * Fetch a `.wgsl.pack` from a URL into a `WgslPack`.
 * When `options.streamThresholdBytes > 0`, probes the remote pack size (via `HEAD` with
 * Range-probe fallback) and automatically switches to ranged index-only mode if
 * `totalSize > options.streamThresholdBytes`. When `streamThresholdBytes` is omitted or `0`,
 * fetches the full buffer via a single `GET` request.
 *
 * @param {string} url
 * @param {typeof fetch | {
 *   fetch?: typeof fetch,
 *   fetchInit?: RequestInit,
 *   lazyDecode?: boolean,
 *   streamThresholdBytes?: number,
 *   initialRangeBytes?: number,
 *   maxCachedWgslBytes?: number,
 * }} [fetchImplOrOptions]
 * @param {object} [maybeOptions]
 * @returns {Promise<WgslPack>}
 */
export async function loadFromUrl(url, fetchImplOrOptions, maybeOptions) {
  if (typeof url !== 'string' || url.length === 0) {
    throw new TypeError('url must be a non-empty string');
  }
  const { fetcher, options } = resolveFetchAndOptions(fetchImplOrOptions, maybeOptions, null);
  if (typeof fetcher !== 'function') {
    throw new Error('No fetch implementation available for loadFromUrl');
  }

  const lazyDecode = options.lazyDecode !== undefined ? Boolean(options.lazyDecode) : true;
  const streamThresholdBytes = validateStreamThresholdBytes(options.streamThresholdBytes, 0);

  if (streamThresholdBytes > 0) {
    const initialRangeBytes = validateInitialRangeBytes(options.initialRangeBytes);
    validateMaxCachedWgslBytes(options.maxCachedWgslBytes);
    validateMaxConcurrentRangeFetches(options.maxConcurrentRangeFetches);
    const maxRangeFetchRetries = validateMaxRangeFetchRetries(options.maxRangeFetchRetries);
    const rangeRetryDelayMs = validateRangeRetryDelayMs(options.rangeRetryDelayMs);
    const sleepFn = validateSleepFn(options.sleepFn);
    validatePrefetchMaxGapBytes(options.prefetchMaxGapBytes);
    validatePrefetchMaxSpanBytes(options.prefetchMaxSpanBytes);

    // 1. Prefer lightweight HEAD probe for Content-Length & Accept-Ranges.
    let headResolved = false;
    let useFullGet = false;
    let useRanged = false;

    const headResponse = await fetcher(url, mergeRequestInit(options.fetchInit, { method: 'HEAD' }));
    if (headResponse && headResponse.ok) {
      const acceptRanges = getResponseHeader(headResponse, 'accept-ranges');
      if (acceptRanges && acceptRanges.trim().toLowerCase() === 'none') {
        headResolved = true;
        useFullGet = true;
      } else {
        const contentLength = parseNonNegativeInt(getResponseHeader(headResponse, 'content-length'));
        if (contentLength !== null) {
          headResolved = true;
          if (contentLength > streamThresholdBytes) {
            useRanged = true;
          } else {
            useFullGet = true;
          }
        }
      }
    } else if (headResponse && headResponse.status !== 405 && headResponse.status !== 501) {
      const status = headResponse.status;
      const statusText = headResponse.statusText ? ` ${headResponse.statusText}` : '';
      throw new Error(`Failed to fetch WGSL pack from "${url}": HTTP ${status}${statusText}`);
    }

    if (headResolved && useRanged) {
      return loadIndexFromUrl(url, fetcher, options);
    }

    if (!headResolved && !useFullGet) {
      // 2. Fallback probe when HEAD is unsupported or lacks Content-Length: initial byte range.
      let probeFetches = 0;
      let probeRetries = 0;
      const probe = await fetchRangeChunkWithRetry(
        fetcher,
        url,
        0,
        initialRangeBytes - 1,
        options.fetchInit,
        {
          maxRangeFetchRetries,
          rangeRetryDelayMs,
          sleepFn,
          onAttempt: () => {
            probeFetches += 1;
          },
          onRetry: () => {
            probeRetries += 1;
          },
        }
      );
      if (probe.status !== 206) {
        return parseWgslPack(probe.bytes, { lazyDecode });
      }
      const probeTotal = probe.contentRange ? probe.contentRange.total : null;
      if (probeTotal !== null && probeTotal <= streamThresholdBytes) {
        if (
          probe.bytes.byteLength >= probeTotal &&
          (!probe.contentRange || probe.contentRange.start === 0)
        ) {
          return parseWgslPack(probe.bytes.subarray(0, probeTotal), { lazyDecode });
        }
        // Small pack larger than initialRangeBytes: fall through to full GET below.
      } else {
        return finishPackIndexFromInitialChunk(url, fetcher, probe, {
          ...options,
          initialRangeFetches: probeFetches,
          initialRangeFetchRetries: probeRetries,
        });
      }
    }
  }

  const response = await fetcher(url, options.fetchInit);
  if (!response || !response.ok) {
    const status = response ? response.status : 'unknown';
    const statusText = response && response.statusText ? ` ${response.statusText}` : '';
    throw new Error(`Failed to fetch WGSL pack from "${url}": HTTP ${status}${statusText}`);
  }

  const buffer = await response.arrayBuffer();
  return parseWgslPack(buffer, { lazyDecode });
}

/**
 * Alias for `parseWgslPack(buffer, options)`.
 *
 * @param {ArrayBuffer | SharedArrayBuffer | ArrayBufferView} buffer
 * @param {{ lazyDecode?: boolean }} [options]
 * @returns {WgslPack}
 */
export function loadFromBuffer(buffer, options = {}) {
  return parseWgslPack(buffer, options);
}

/**
 * Multi-pack WGSL loader and compiled WebGPU `GPUShaderModule` cache.
 */
export class WgslPackCache {
  /**
   * @param {{
   *   lazyDecode?: boolean,
   *   labelPrefix?: string,
   *   prewarmConcurrency?: number,
   *   fetch?: typeof fetch,
   * }} [options]
   */
  constructor(options = {}) {
    this._lazyDecode = options.lazyDecode !== undefined ? Boolean(options.lazyDecode) : true;
    this._labelPrefix = options.labelPrefix !== undefined ? String(options.labelPrefix) : 'wgpk';
    this._prewarmConcurrency =
      options.prewarmConcurrency !== undefined ? options.prewarmConcurrency : 8;
    if (!Number.isInteger(this._prewarmConcurrency) || this._prewarmConcurrency < 1) {
      throw new RangeError('prewarmConcurrency must be a positive integer');
    }
    this._fetch = typeof options.fetch === 'function' ? options.fetch : null;
    this._maxCachedWgslBytes = validateMaxCachedWgslBytes(options.maxCachedWgslBytes);
    this._initialRangeBytes = validateInitialRangeBytes(options.initialRangeBytes);
    this._streamThresholdBytes = validateStreamThresholdBytes(options.streamThresholdBytes, 0);
    this._maxConcurrentRangeFetches = validateMaxConcurrentRangeFetches(
      options.maxConcurrentRangeFetches
    );
    this._maxRangeFetchRetries = validateMaxRangeFetchRetries(options.maxRangeFetchRetries);
    this._rangeRetryDelayMs = validateRangeRetryDelayMs(options.rangeRetryDelayMs);
    this._sleepFn = validateSleepFn(options.sleepFn);
    this._prefetchMaxGapBytes = validatePrefetchMaxGapBytes(options.prefetchMaxGapBytes);
    this._prefetchMaxSpanBytes = validatePrefetchMaxSpanBytes(options.prefetchMaxSpanBytes);

    this._packs = new Map();
    this._shaderEntries = new Map();
    this._dynamicEntries = new Map();
    this._deviceModules = new Map();
    this._indexMeta = new Map();
    this._shaderIndexAbiVersion = null;
    this._shaderIndexPackPath = null;
    this._nextPackId = 0;

    this._stats = {
      hits: 0,
      misses: 0,
      compiles: 0,
      evictions: 0,
      dynamicHits: 0,
    };
  }

  _registerPack(pack, packName) {
    const name =
      typeof packName === 'string' && packName.length > 0
        ? packName
        : `pack-${this._nextPackId++}`;

    // If replacing an existing pack with the same name, detach its old entries first.
    if (this._packs.has(name)) {
      for (const [hash, record] of this._shaderEntries.entries()) {
        if (record.packName === name) {
          this._shaderEntries.delete(hash);
          if (!this._dynamicEntries.has(hash)) {
            this._evictModuleOnly(hash);
          }
        }
      }
    }

    this._packs.set(name, pack);
    for (const entry of pack.listEntries()) {
      if (this._shaderEntries.has(entry.hash) && !this._dynamicEntries.has(entry.hash)) {
        this._evictModuleOnly(entry.hash);
      }
      this._shaderEntries.set(entry.hash, { entry, pack, packName: name });
    }

    return pack;
  }

  /**
   * Parse and register a `.wgsl.pack` binary buffer.
   * If a pack with the same `packName` is already loaded, its entries are replaced.
   * Active dynamic shader overrides are preserved across static pack loads.
   *
   * @param {ArrayBuffer | SharedArrayBuffer | ArrayBufferView} buffer
   * @param {string} [packName]
   * @param {{ lazyDecode?: boolean }} [options]
   * @returns {WgslPack}
   */
  loadPackBuffer(buffer, packName, options = {}) {
    const lazyDecode =
      options && options.lazyDecode !== undefined ? Boolean(options.lazyDecode) : this._lazyDecode;
    const pack = parseWgslPack(buffer, { lazyDecode });
    return this._registerPack(pack, packName);
  }

  /**
   * Alias for `loadPackBuffer` accepting either `(buffer, packName, options)` or `(buffer, options)`.
   *
   * @param {ArrayBuffer | SharedArrayBuffer | ArrayBufferView} buffer
   * @param {string | { packName?: string, lazyDecode?: boolean }} [packNameOrOptions]
   * @param {{ lazyDecode?: boolean }} [maybeOptions]
   * @returns {WgslPack}
   */
  loadFromBuffer(buffer, packNameOrOptions, maybeOptions) {
    if (packNameOrOptions && typeof packNameOrOptions === 'object') {
      return this.loadPackBuffer(buffer, packNameOrOptions.packName, packNameOrOptions);
    }
    return this.loadPackBuffer(buffer, packNameOrOptions, maybeOptions);
  }

  /**
   * Fetch only the WGPK header + entry table from a URL via HTTP `206 Range` and register
   * the resulting ranged pack (or buffer fallback on `200 OK`).
   *
   * @param {string} url
   * @param {typeof fetch | {
   *   packName?: string,
   *   fetch?: typeof fetch,
   *   fetchInit?: RequestInit,
   *   initialRangeBytes?: number,
   *   maxCachedWgslBytes?: number,
   *   lazyDecode?: boolean,
   * }} [fetchImplOrOptions]
   * @param {object} [maybeOptions]
   * @returns {Promise<WgslPack>}
   */
  async loadIndexFromUrl(url, fetchImplOrOptions, maybeOptions) {
    if (typeof url !== 'string' || url.length === 0) {
      throw new TypeError('url must be a non-empty string');
    }
    const { fetcher, options } = resolveFetchAndOptions(
      fetchImplOrOptions,
      maybeOptions,
      this._fetch
    );
    if (typeof fetcher !== 'function') {
      throw new Error('No fetch implementation available for loadIndexFromUrl');
    }

    const pack = await loadIndexFromUrl(url, fetcher, {
      fetchInit: options.fetchInit,
      initialRangeBytes: options.initialRangeBytes ?? this._initialRangeBytes,
      maxCachedWgslBytes: options.maxCachedWgslBytes ?? this._maxCachedWgslBytes,
      maxConcurrentRangeFetches:
        options.maxConcurrentRangeFetches ?? this._maxConcurrentRangeFetches,
      maxRangeFetchRetries: options.maxRangeFetchRetries ?? this._maxRangeFetchRetries,
      rangeRetryDelayMs: options.rangeRetryDelayMs ?? this._rangeRetryDelayMs,
      sleepFn: options.sleepFn ?? this._sleepFn,
      prefetchMaxGapBytes: options.prefetchMaxGapBytes ?? this._prefetchMaxGapBytes,
      prefetchMaxSpanBytes: options.prefetchMaxSpanBytes ?? this._prefetchMaxSpanBytes,
      lazyDecode: options.lazyDecode !== undefined ? options.lazyDecode : this._lazyDecode,
    });
    const name = options.packName ?? url;
    return this._registerPack(pack, name);
  }

  /**
   * Alias for `loadIndexFromUrl`.
   */
  loadPackIndexUrl(url, fetchImplOrOptions, maybeOptions) {
    return this.loadIndexFromUrl(url, fetchImplOrOptions, maybeOptions);
  }

  /**
   * Fetch a `.wgsl.pack` bundle from a URL and register it in the cache.
   * If `options.streamThresholdBytes > 0` (or configured on the cache), automatically uses
   * HTTP `206 Range` index-only mode when the remote pack exceeds the threshold.
   *
   * @param {string} url
   * @param {typeof fetch | {
   *   packName?: string,
   *   lazyDecode?: boolean,
   *   fetch?: typeof fetch,
   *   fetchInit?: RequestInit,
   *   streamThresholdBytes?: number,
   *   initialRangeBytes?: number,
   *   maxCachedWgslBytes?: number,
   * }} [fetchImplOrOptions]
   * @param {object} [maybeOptions]
   * @returns {Promise<WgslPack>}
   */
  async loadPackUrl(url, fetchImplOrOptions, maybeOptions) {
    if (typeof url !== 'string' || url.length === 0) {
      throw new TypeError('url must be a non-empty string');
    }
    const { fetcher, options } = resolveFetchAndOptions(
      fetchImplOrOptions,
      maybeOptions,
      this._fetch
    );
    if (typeof fetcher !== 'function') {
      throw new Error('No fetch implementation available for loadPackUrl');
    }

    const streamThresholdBytes = validateStreamThresholdBytes(
      options.streamThresholdBytes !== undefined
        ? options.streamThresholdBytes
        : this._streamThresholdBytes,
      0
    );
    const lazyDecode = options.lazyDecode !== undefined ? Boolean(options.lazyDecode) : this._lazyDecode;
    const name = options.packName ?? url;

    if (streamThresholdBytes > 0) {
      const pack = await loadFromUrl(url, fetcher, {
        fetchInit: options.fetchInit,
        lazyDecode,
        streamThresholdBytes,
        initialRangeBytes: options.initialRangeBytes ?? this._initialRangeBytes,
        maxCachedWgslBytes: options.maxCachedWgslBytes ?? this._maxCachedWgslBytes,
        maxConcurrentRangeFetches:
          options.maxConcurrentRangeFetches ?? this._maxConcurrentRangeFetches,
        maxRangeFetchRetries: options.maxRangeFetchRetries ?? this._maxRangeFetchRetries,
        rangeRetryDelayMs: options.rangeRetryDelayMs ?? this._rangeRetryDelayMs,
        sleepFn: options.sleepFn ?? this._sleepFn,
        prefetchMaxGapBytes: options.prefetchMaxGapBytes ?? this._prefetchMaxGapBytes,
        prefetchMaxSpanBytes: options.prefetchMaxSpanBytes ?? this._prefetchMaxSpanBytes,
      });
      return this._registerPack(pack, name);
    }

    const response = await fetcher(url, options.fetchInit);
    if (!response || !response.ok) {
      const status = response ? response.status : 'unknown';
      const statusText = response && response.statusText ? ` ${response.statusText}` : '';
      throw new Error(`Failed to fetch WGSL pack from "${url}": HTTP ${status}${statusText}`);
    }

    const buffer = await response.arrayBuffer();
    return this.loadPackBuffer(buffer, name, { lazyDecode });
  }

  /**
   * Alias for `loadPackUrl`.
   */
  loadFromUrl(url, fetchImplOrOptions, maybeOptions) {
    return this.loadPackUrl(url, fetchImplOrOptions, maybeOptions);
  }

  /**
   * Load optional `shader-index.json` metadata generated by `playgta6-shaders`.
   *
   * @param {object | string} indexData
   * @returns {{ abiVersion: number, pack: string | null, count: number }}
   */
  loadShaderIndex(indexData) {
    const parsed = typeof indexData === 'string' ? JSON.parse(indexData) : indexData;
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new TypeError('shader-index.json payload must be a JSON object');
    }

    const abiVersion = parsed.abi_version;
    if (abiVersion !== SHADER_INDEX_ABI_VERSION) {
      throw new Error(
        `Unsupported shader-index.json abi_version ${abiVersion} (expected ${SHADER_INDEX_ABI_VERSION})`
      );
    }

    if (!parsed.shaders || typeof parsed.shaders !== 'object' || Array.isArray(parsed.shaders)) {
      throw new TypeError('shader-index.json must contain a "shaders" object');
    }

    this._shaderIndexAbiVersion = abiVersion;
    this._shaderIndexPackPath = typeof parsed.pack === 'string' ? parsed.pack : null;

    let count = 0;
    for (const [hashKey, raw] of Object.entries(parsed.shaders)) {
      if (!raw || typeof raw !== 'object') {
        throw new TypeError(`Invalid shader-index.json entry for "${hashKey}"`);
      }
      const hash = normalizeHash(hashKey);
      const hashHex = formatHashHex(hash);
      this._indexMeta.set(hash, {
        hash,
        hashHex,
        stage: typeof raw.stage === 'string' ? raw.stage : null,
        wgslPath: typeof raw.wgsl === 'string' ? raw.wgsl : null,
        entryPoint:
          typeof raw.entry_point === 'string'
            ? raw.entry_point
            : typeof raw.entryPoint === 'string'
              ? raw.entryPoint
              : null,
        packOffset:
          typeof raw.pack_offset === 'number'
            ? raw.pack_offset
            : typeof raw.packOffset === 'number'
              ? raw.packOffset
              : null,
        packLength:
          typeof raw.pack_length === 'number'
            ? raw.pack_length
            : typeof raw.packLength === 'number'
              ? raw.packLength
              : null,
      });
      count += 1;
    }

    return {
      abiVersion,
      pack: this._shaderIndexPackPath,
      count,
    };
  }

  /**
   * Fetch and register `shader-index.json` metadata from a URL.
   *
   * @param {string} url
   * @param {typeof fetch} [fetchImpl]
   */
  async loadShaderIndexUrl(url, fetchImpl) {
    if (typeof url !== 'string' || url.length === 0) {
      throw new TypeError('url must be a non-empty string');
    }
    const fetcher = fetchImpl ?? this._fetch ?? globalThis.fetch;
    if (typeof fetcher !== 'function') {
      throw new Error('No fetch implementation available for loadShaderIndexUrl');
    }
    const response = await fetcher(url);
    if (!response || !response.ok) {
      const status = response ? response.status : 'unknown';
      const statusText = response && response.statusText ? ` ${response.statusText}` : '';
      throw new Error(`Failed to fetch shader-index.json from "${url}": HTTP ${status}${statusText}`);
    }
    const json = await response.json();
    return this.loadShaderIndex(json);
  }

  /**
   * Register a live dynamic WGSL shader at runtime (e.g., freshly translated RDNA2->WGSL
   * or hot-patched shader). Pass `{ override: true }` in `meta` to supersede an existing
   * static pack entry while preserving the static entry for `clearDynamicShaders()`.
   * Automatically evicts any stale compiled `GPUShaderModule` cached for `hash`.
   *
   * @param {bigint | string | number} hash
   * @param {number | string} stage
   * @param {string} wgslSource
   * @param {{ override?: boolean, label?: string, entryPoint?: string, source?: string, tags?: string[] }} [meta]
   */
  registerDynamicShader(hash, stage, wgslSource, meta) {
    const key = normalizeHash(hash);
    const staticRecord = this._shaderEntries.get(key);
    const prevDynamic = this._dynamicEntries.get(key);
    const entry = createDynamicShaderEntry(hash, stage, wgslSource, meta, prevDynamic);

    if (staticRecord && !entry.override) {
      throw new Error(
        `Shader ${entry.hashHex} already exists in static pack "${staticRecord.packName}"; pass { override: true } to supersede it`
      );
    }

    this._dynamicEntries.set(key, entry);
    this._evictModuleOnly(key);
    return formatDynamicLookupRecord(entry, Boolean(staticRecord));
  }

  /**
   * Check whether a dynamic shader is registered for `hash` (and optional `stage`).
   *
   * @param {bigint | string | number} hash
   * @param {number | string} [stage]
   * @returns {boolean}
   */
  hasDynamicShader(hash, stage) {
    const key = normalizeHash(hash);
    const entry = this._dynamicEntries.get(key);
    if (!entry) {
      return false;
    }
    if (stage !== undefined && stage !== null) {
      return entry.stageCode === resolveStageCode(stage);
    }
    return true;
  }

  /**
   * Remove a single dynamic shader by hash and evict its compiled module(s),
   * restoring any underlying static pack entry.
   *
   * @param {bigint | string | number} hash
   * @returns {boolean}
   */
  unregisterDynamicShader(hash) {
    const key = normalizeHash(hash);
    const removed = this._dynamicEntries.delete(key);
    if (removed) {
      this._evictModuleOnly(key);
    }
    return removed;
  }

  /**
   * Remove all dynamically registered shaders and evict their compiled `GPUShaderModule`s
   * while preserving all static packs, static entries, and shader-index metadata.
   *
   * @returns {number} Number of dynamic shaders cleared
   */
  clearDynamicShaders() {
    const count = this._dynamicEntries.size;
    for (const key of this._dynamicEntries.keys()) {
      this._evictModuleOnly(key);
    }
    this._dynamicEntries.clear();
    return count;
  }

  /**
   * Look up a shader entry by hash (and optional stage), checking dynamic overrides first
   * and then static pack entries. Increments `dynamicHits` when resolved dynamically.
   *
   * @param {bigint | string | number} hash
   * @param {number | string} [stage]
   */
  lookup(hash, stage) {
    const key = normalizeHash(hash);
    const expectedStage =
      stage !== undefined && stage !== null ? resolveStageCode(stage) : null;

    const dynamicEntry = this._dynamicEntries.get(key);
    if (dynamicEntry) {
      if (expectedStage !== null && dynamicEntry.stageCode !== expectedStage) {
        return undefined;
      }
      this._stats.dynamicHits += 1;
      return formatDynamicLookupRecord(dynamicEntry, this._shaderEntries.has(key));
    }

    const record = this._shaderEntries.get(key);
    if (!record) {
      return undefined;
    }
    if (expectedStage !== null && record.entry.stageCode !== expectedStage) {
      return undefined;
    }
    const indexMeta = this._indexMeta.get(key);
    return {
      hash: record.entry.hash,
      hashHex: record.entry.hashHex,
      stage: record.entry.stage,
      stageCode: record.entry.stageCode,
      wgsl: record.entry.wgsl,
      length: record.entry.length,
      offset: record.entry.offset,
      dynamic: false,
      overridesStatic: false,
      packName: record.packName,
      source: 'pack',
      label: null,
      entryPoint: indexMeta ? indexMeta.entryPoint : null,
      tags: [],
      revision: 0,
      registeredAt: null,
    };
  }

  /**
   * Alias for `lookup(hash, stage)`.
   *
   * @param {bigint | string | number} hash
   * @param {number | string} [stage]
   */
  get(hash, stage) {
    return this.lookup(hash, stage);
  }

  /**
   * Return whether any dynamic override or loaded static pack contains the given shader hash
   * (optionally matching `stage`).
   *
   * @param {bigint | string | number} hash
   * @param {number | string} [stage]
   * @returns {boolean}
   */
  hasShader(hash, stage) {
    const key = normalizeHash(hash);
    const expectedStage =
      stage !== undefined && stage !== null ? resolveStageCode(stage) : null;

    const dynamicEntry = this._dynamicEntries.get(key);
    if (dynamicEntry) {
      return expectedStage === null || dynamicEntry.stageCode === expectedStage;
    }

    const record = this._shaderEntries.get(key);
    if (!record) {
      return false;
    }
    return expectedStage === null || record.entry.stageCode === expectedStage;
  }

  /**
   * Alias for `hasShader(hash, stage)`.
   *
   * @param {bigint | string | number} hash
   * @param {number | string} [stage]
   * @returns {boolean}
   */
  has(hash, stage) {
    return this.hasShader(hash, stage);
  }

  /**
   * Get the WGSL source string for a shader hash, checking dynamic overrides first
   * and then loaded static packs.
   *
   * @param {bigint | string | number} hash
   * @param {number | string} [stage]
   * @returns {string | undefined}
   */
  getShaderSource(hash, stage) {
    const found = this.lookup(hash, stage);
    return found ? found.wgsl : undefined;
  }

  /**
   * Alias for `getShaderSource(hash, stage)`.
   *
   * @param {bigint | string | number} hash
   * @param {number | string} [stage]
   * @returns {string | undefined}
   */
  getWgsl(hash, stage) {
    return this.getShaderSource(hash, stage);
  }

  /**
   * Asynchronously retrieve the WGSL source string for a shader hash across both
   * full-buffer packs and HTTP-ranged packs (checking dynamic overrides first).
   *
   * @param {bigint | string | number} hash
   * @param {number | string} [stage]
   * @returns {Promise<string | null>}
   */
  async getWgslAsync(hash, stage) {
    const key = normalizeHash(hash);
    const expectedStage =
      stage !== undefined && stage !== null ? resolveStageCode(stage) : null;

    const dynamicEntry = this._dynamicEntries.get(key);
    if (dynamicEntry) {
      if (expectedStage !== null && dynamicEntry.stageCode !== expectedStage) {
        return null;
      }
      this._stats.dynamicHits += 1;
      return dynamicEntry.wgsl;
    }

    const record = this._shaderEntries.get(key);
    if (!record) {
      return null;
    }
    if (expectedStage !== null && record.entry.stageCode !== expectedStage) {
      return null;
    }
    return record.pack.getWgslAsync(key);
  }

  /**
   * Alias for `getWgslAsync(hash, stage)`.
   *
   * @param {bigint | string | number} hash
   * @param {number | string} [stage]
   * @returns {Promise<string | null>}
   */
  getShaderSourceAsync(hash, stage) {
    return this.getWgslAsync(hash, stage);
  }

  /**
   * Prefetch a set of shader hashes across all loaded packs, coalescing nearby byte ranges
   * per ranged pack into batched HTTP `206 Range` fetches and populating each pack's bounded
   * hot-hash WGSL LRU cache.
   *
   * @param {Iterable<bigint | string | number>} hashes
   * @param {{
   *   maxGapBytes?: number,
   *   maxSpanBytes?: number,
   *   prefetchMaxGapBytes?: number,
   *   prefetchMaxSpanBytes?: number,
   * }} [options]
   * @returns {Promise<{
   *   requested: number,
   *   prefetched: number,
   *   cached: number,
   *   inFlight: number,
   *   missing: number,
   *   batches: number,
   *   bytesFetched: number,
   * }>}
   */
  async prefetchHashes(hashes, options = {}) {
    const opts = options && typeof options === 'object' ? options : {};
    const maxGapBytes =
      opts.maxGapBytes !== undefined || opts.prefetchMaxGapBytes !== undefined
        ? validatePrefetchMaxGapBytes(
            opts.maxGapBytes !== undefined ? opts.maxGapBytes : opts.prefetchMaxGapBytes,
            this._prefetchMaxGapBytes
          )
        : undefined;
    const maxSpanBytes =
      opts.maxSpanBytes !== undefined || opts.prefetchMaxSpanBytes !== undefined
        ? validatePrefetchMaxSpanBytes(
            opts.maxSpanBytes !== undefined ? opts.maxSpanBytes : opts.prefetchMaxSpanBytes,
            this._prefetchMaxSpanBytes
          )
        : undefined;

    if (hashes === undefined || hashes === null) {
      return {
        requested: 0,
        prefetched: 0,
        cached: 0,
        inFlight: 0,
        missing: 0,
        batches: 0,
        bytesFetched: 0,
      };
    }
    if (typeof hashes[Symbol.iterator] !== 'function') {
      throw new TypeError('hashes must be an iterable of shader hashes');
    }

    const seen = new Set();
    let requested = 0;
    let cached = 0;
    let missing = 0;
    const byPack = new Map();

    for (const rawHash of hashes) {
      const key = normalizeHash(rawHash);
      if (seen.has(key)) {
        continue;
      }
      seen.add(key);
      requested += 1;

      if (this._dynamicEntries.has(key)) {
        cached += 1;
        continue;
      }

      const record = this._shaderEntries.get(key);
      if (!record) {
        missing += 1;
        continue;
      }

      let list = byPack.get(record.pack);
      if (!list) {
        list = [];
        byPack.set(record.pack, list);
      }
      list.push(key);
    }

    if (byPack.size === 0) {
      return {
        requested,
        prefetched: 0,
        cached,
        inFlight: 0,
        missing,
        batches: 0,
        bytesFetched: 0,
      };
    }

    const forwardOptions = {};
    if (maxGapBytes !== undefined) {
      forwardOptions.maxGapBytes = maxGapBytes;
    }
    if (maxSpanBytes !== undefined) {
      forwardOptions.maxSpanBytes = maxSpanBytes;
    }

    const results = await Promise.all(
      Array.from(byPack.entries(), ([pack, keys]) => pack.prefetchHashes(keys, forwardOptions))
    );

    let prefetched = 0;
    let inFlight = 0;
    let batches = 0;
    let bytesFetched = 0;
    for (let i = 0; i < results.length; i += 1) {
      const r = results[i];
      prefetched += r.prefetched;
      cached += r.cached;
      inFlight += r.inFlight;
      missing += r.missing;
      batches += r.batches;
      bytesFetched += r.bytesFetched;
    }

    return {
      requested,
      prefetched,
      cached,
      inFlight,
      missing,
      batches,
      bytesFetched,
    };
  }

  /**
   * Get combined dynamic + static pack + optional `shader-index.json` metadata for a shader hash.
   *
   * @param {bigint | string | number} hash
   */
  getShaderMetadata(hash) {
    const key = normalizeHash(hash);
    const dynamicEntry = this._dynamicEntries.get(key);
    const record = this._shaderEntries.get(key);
    const indexMeta = this._indexMeta.get(key);
    if (!dynamicEntry && !record && !indexMeta) {
      return undefined;
    }
    if (dynamicEntry) {
      return {
        hash: key,
        hashHex: formatHashHex(key),
        stage: dynamicEntry.stage,
        stageCode: dynamicEntry.stageCode,
        packName: record ? record.packName : null,
        offset: record ? record.entry.offset : indexMeta ? indexMeta.packOffset : null,
        length: dynamicEntry.length,
        wgslPath: indexMeta ? indexMeta.wgslPath : null,
        entryPoint:
          dynamicEntry.entryPoint !== null
            ? dynamicEntry.entryPoint
            : indexMeta
              ? indexMeta.entryPoint
              : null,
        dynamic: true,
        overridesStatic: Boolean(record),
        source: dynamicEntry.source,
        label: dynamicEntry.label,
        tags: dynamicEntry.tags.slice(),
        revision: dynamicEntry.revision,
        registeredAt: dynamicEntry.registeredAt,
      };
    }
    return {
      hash: key,
      hashHex: formatHashHex(key),
      stage: record ? record.entry.stage : indexMeta.stage,
      stageCode: record ? record.entry.stageCode : null,
      packName: record ? record.packName : null,
      offset: record ? record.entry.offset : indexMeta.packOffset,
      length: record ? record.entry.length : indexMeta.packLength,
      wgslPath: indexMeta ? indexMeta.wgslPath : null,
      entryPoint: indexMeta ? indexMeta.entryPoint : null,
      dynamic: false,
      overridesStatic: false,
    };
  }

  _getDeviceMap(device, createIfMissing) {
    let map = this._deviceModules.get(device);
    if (!map && createIfMissing) {
      map = new Map();
      this._deviceModules.set(device, map);
    }
    return map;
  }

  _hasCachedModule(device, hashBigInt) {
    const map = this._deviceModules.get(device);
    return map ? map.has(hashBigInt) : false;
  }

  /**
   * Return a cached `GPUShaderModule` for `(device, hash)` or compile and cache it.
   * Resolves dynamic shader overrides ahead of static pack entries.
   *
   * @param {GPUDevice} device
   * @param {bigint | string | number} hash
   * @param {string} [labelPrefix]
   * @returns {GPUShaderModule}
   */
  getOrCreateShaderModule(device, hash, labelPrefix) {
    if (!device || typeof device.createShaderModule !== 'function') {
      throw new TypeError('device must provide a createShaderModule(descriptor) method');
    }

    const key = normalizeHash(hash);
    const deviceMap = this._getDeviceMap(device, true);
    const cached = deviceMap.get(key);
    if (cached !== undefined) {
      this._stats.hits += 1;
      if (this._dynamicEntries.has(key)) {
        this._stats.dynamicHits += 1;
      }
      return cached;
    }

    this._stats.misses += 1;
    const dynamicEntry = this._dynamicEntries.get(key);
    let entry;
    if (dynamicEntry) {
      this._stats.dynamicHits += 1;
      entry = dynamicEntry;
    } else {
      const record = this._shaderEntries.get(key);
      if (!record) {
        throw new Error(`Shader ${formatHashHex(key)} not found in any loaded WGSL pack`);
      }
      entry = record.entry;
    }

    const prefix = labelPrefix !== undefined ? String(labelPrefix) : this._labelPrefix;
    const baseLabel = prefix
      ? `${prefix}:${entry.stage}:${entry.hashHex}`
      : `${entry.stage}:${entry.hashHex}`;
    const label =
      dynamicEntry && dynamicEntry.label ? `${baseLabel}:${dynamicEntry.label}` : baseLabel;

    const shaderModule = device.createShaderModule({
      label,
      code: entry.wgsl,
    });

    deviceMap.set(key, shaderModule);
    this._stats.compiles += 1;
    return shaderModule;
  }

  /**
   * Async companion to `getOrCreateShaderModule` supporting both full-buffer and ranged packs.
   *
   * @param {GPUDevice} device
   * @param {bigint | string | number} hash
   * @param {string} [labelPrefix]
   * @returns {Promise<GPUShaderModule>}
   */
  async getOrCreateShaderModuleAsync(device, hash, labelPrefix) {
    if (!device || typeof device.createShaderModule !== 'function') {
      throw new TypeError('device must provide a createShaderModule(descriptor) method');
    }

    const key = normalizeHash(hash);
    const deviceMap = this._getDeviceMap(device, true);
    const cached = deviceMap.get(key);
    if (cached !== undefined) {
      this._stats.hits += 1;
      if (this._dynamicEntries.has(key)) {
        this._stats.dynamicHits += 1;
      }
      return cached;
    }

    const dynamicEntry = this._dynamicEntries.get(key);
    if (dynamicEntry) {
      return this.getOrCreateShaderModule(device, key, labelPrefix);
    }

    const record = this._shaderEntries.get(key);
    if (!record) {
      this._stats.misses += 1;
      throw new Error(`Shader ${formatHashHex(key)} not found in any loaded WGSL pack`);
    }

    if (record.pack.mode !== 'ranged') {
      return this.getOrCreateShaderModule(device, key, labelPrefix);
    }

    const code = await record.pack.getWgslAsync(key);
    if (code === null) {
      this._stats.misses += 1;
      throw new Error(`Shader ${formatHashHex(key)} not found in any loaded WGSL pack`);
    }

    // Re-check device cache in case a concurrent caller compiled while we awaited.
    const cachedAfterFetch = deviceMap.get(key);
    if (cachedAfterFetch !== undefined) {
      this._stats.hits += 1;
      return cachedAfterFetch;
    }

    this._stats.misses += 1;
    const prefix = labelPrefix !== undefined ? String(labelPrefix) : this._labelPrefix;
    const label = prefix
      ? `${prefix}:${record.entry.stage}:${record.entry.hashHex}`
      : `${record.entry.stage}:${record.entry.hashHex}`;

    const shaderModule = device.createShaderModule({
      label,
      code,
    });

    deviceMap.set(key, shaderModule);
    this._stats.compiles += 1;
    return shaderModule;
  }

  /**
   * Batch-compile a set of shader hashes on `device` with bounded concurrency.
   * If `hashes` is omitted, prewarms every shader loaded across all packs.
   *
   * @param {GPUDevice} device
   * @param {Iterable<bigint | string | number>} [hashes]
   * @param {{
   *   concurrency?: number,
   *   labelPrefix?: string,
   *   awaitCompilationInfo?: boolean,
   * }} [options]
   * @returns {Promise<{
   *   total: number,
   *   compiled: number,
   *   cached: number,
   *   modules: Map<bigint, GPUShaderModule>,
   * }>}
   */
  async prewarm(device, hashes, options = {}) {
    if (!device || typeof device.createShaderModule !== 'function') {
      throw new TypeError('device must provide a createShaderModule(descriptor) method');
    }

    const concurrency =
      options && options.concurrency !== undefined
        ? options.concurrency
        : this._prewarmConcurrency;
    if (!Number.isInteger(concurrency) || concurrency < 1) {
      throw new RangeError('prewarm concurrency must be a positive integer');
    }

    const labelPrefix = options ? options.labelPrefix : undefined;
    const awaitCompilationInfo =
      options && options.awaitCompilationInfo !== undefined
        ? Boolean(options.awaitCompilationInfo)
        : true;

    const rawList =
      hashes === undefined || hashes === null
        ? [...this._shaderEntries.keys(), ...this._dynamicEntries.keys()]
        : Array.from(hashes, (h) => normalizeHash(h));

    // Deduplicate while preserving submission order.
    const uniqueHashes = [];
    const seen = new Set();
    for (const h of rawList) {
      if (!seen.has(h)) {
        seen.add(h);
        uniqueHashes.push(h);
      }
    }

    const modules = new Map();
    let compiled = 0;
    let cached = 0;
    let cursor = 0;

    const worker = async () => {
      while (cursor < uniqueHashes.length) {
        const hash = uniqueHashes[cursor++];
        const alreadyCached = this._hasCachedModule(device, hash);
        const record = this._shaderEntries.get(hash);
        const useAsync =
          !alreadyCached &&
          !this._dynamicEntries.has(hash) &&
          record &&
          record.pack.mode === 'ranged';
        const mod = useAsync
          ? await this.getOrCreateShaderModuleAsync(device, hash, labelPrefix)
          : this.getOrCreateShaderModule(device, hash, labelPrefix);
        modules.set(hash, mod);
        if (alreadyCached) {
          cached += 1;
        } else {
          compiled += 1;
          if (awaitCompilationInfo && mod && typeof mod.getCompilationInfo === 'function') {
            await mod.getCompilationInfo();
          }
        }
      }
    };

    const workerCount = Math.min(concurrency, uniqueHashes.length);
    const workers = [];
    for (let i = 0; i < workerCount; i += 1) {
      workers.push(worker());
    }
    await Promise.all(workers);

    return {
      total: uniqueHashes.length,
      compiled,
      cached,
      modules,
    };
  }

  _evictModuleOnly(key) {
    let removed = 0;
    for (const [device, map] of this._deviceModules.entries()) {
      if (map.delete(key)) {
        removed += 1;
        if (map.size === 0) {
          this._deviceModules.delete(device);
        }
      }
    }
    if (removed > 0) {
      this._stats.evictions += removed;
    }
    return removed > 0;
  }

  /**
   * Evict a compiled `GPUShaderModule` for `hash` from the cache.
   * Pass `{ removeSource: true }` to also drop its loaded pack / dynamic entry.
   *
   * @param {bigint | string | number} hash
   * @param {{ removeSource?: boolean }} [options]
   * @returns {boolean} True if a cached module (or source, when requested) was evicted
   */
  evict(hash, options = {}) {
    const key = normalizeHash(hash);
    const evictedModule = this._evictModuleOnly(key);
    let evictedSource = false;
    if (options && options.removeSource) {
      const removedStatic = this._shaderEntries.delete(key);
      const removedDynamic = this._dynamicEntries.delete(key);
      evictedSource = removedStatic || removedDynamic;
    }
    return evictedModule || evictedSource;
  }

  /**
   * Clear all loaded packs, dynamic shaders, cached shader modules, and index metadata.
   *
   * @param {{ resetStats?: boolean }} [options]
   */
  clear(options = {}) {
    this._packs.clear();
    this._shaderEntries.clear();
    this._dynamicEntries.clear();
    this._deviceModules.clear();
    this._indexMeta.clear();
    this._shaderIndexAbiVersion = null;
    this._shaderIndexPackPath = null;
    if (options && options.resetStats) {
      this._stats.hits = 0;
      this._stats.misses = 0;
      this._stats.compiles = 0;
      this._stats.evictions = 0;
      this._stats.dynamicHits = 0;
    }
  }

  /**
   * List metadata summaries for all dynamically registered shaders.
   *
   * @param {{ stage?: number | string, includeSource?: boolean }} [options]
   */
  listDynamicShaders(options = {}) {
    const expectedStage =
      options && options.stage !== undefined && options.stage !== null
        ? resolveStageCode(options.stage)
        : null;
    const includeSource = Boolean(options && options.includeSource);

    const results = [];
    for (const entry of this._dynamicEntries.values()) {
      if (expectedStage !== null && entry.stageCode !== expectedStage) {
        continue;
      }
      const item = {
        hash: entry.hash,
        hashHex: entry.hashHex,
        stage: entry.stage,
        stageCode: entry.stageCode,
        length: entry.length,
        dynamic: true,
        overridesStatic: this._shaderEntries.has(entry.hash),
        source: entry.source,
        label: entry.label,
        entryPoint: entry.entryPoint,
        tags: entry.tags.slice(),
        revision: entry.revision,
        registeredAt: entry.registeredAt,
      };
      if (includeSource) {
        item.wgsl = entry.wgsl;
      }
      results.push(item);
    }
    return results;
  }

  /**
   * List active resolvable shaders across static packs and dynamic overrides
   * without triggering lazy UTF-8 decoding unless `includeSource: true`.
   *
   * @param {{
   *   stage?: number | string,
   *   dynamicOnly?: boolean,
   *   staticOnly?: boolean,
   *   includeSource?: boolean,
   * }} [options]
   */
  listShaders(options = {}) {
    const expectedStage =
      options && options.stage !== undefined && options.stage !== null
        ? resolveStageCode(options.stage)
        : null;
    const dynamicOnly = Boolean(options && options.dynamicOnly);
    const staticOnly = Boolean(options && options.staticOnly);
    const includeSource = Boolean(options && options.includeSource);

    const results = [];
    const seen = new Set();

    if (!staticOnly) {
      for (const entry of this._dynamicEntries.values()) {
        seen.add(entry.hash);
        if (expectedStage !== null && entry.stageCode !== expectedStage) {
          continue;
        }
        const item = {
          hash: entry.hash,
          hashHex: entry.hashHex,
          stage: entry.stage,
          stageCode: entry.stageCode,
          length: entry.length,
          dynamic: true,
          overridesStatic: this._shaderEntries.has(entry.hash),
          packName: null,
          source: entry.source,
          label: entry.label,
          entryPoint: entry.entryPoint,
          tags: entry.tags.slice(),
          revision: entry.revision,
          registeredAt: entry.registeredAt,
        };
        if (includeSource) {
          item.wgsl = entry.wgsl;
        }
        results.push(item);
      }
    }

    if (!dynamicOnly) {
      for (const [hash, record] of this._shaderEntries.entries()) {
        if (!staticOnly && seen.has(hash)) {
          continue;
        }
        if (expectedStage !== null && record.entry.stageCode !== expectedStage) {
          continue;
        }
        const indexMeta = this._indexMeta.get(hash);
        const item = {
          hash,
          hashHex: record.entry.hashHex,
          stage: record.entry.stage,
          stageCode: record.entry.stageCode,
          length: record.entry.length,
          offset: record.entry.offset,
          dynamic: false,
          overridesStatic: false,
          overriddenByDynamic: this._dynamicEntries.has(hash),
          packName: record.packName,
          source: 'pack',
          label: null,
          entryPoint: indexMeta ? indexMeta.entryPoint : null,
          tags: [],
          revision: 0,
          registeredAt: null,
        };
        if (includeSource) {
          item.wgsl = record.entry.wgsl;
        }
        results.push(item);
      }
    }

    return results;
  }

  /**
   * Return a JSON-serializable introspection snapshot of loaded packs, index metadata,
   * dynamic overrides, and cache statistics.
   */
  inspect() {
    const stats = this.getStats();
    const dynamicShaders = [];
    for (const entry of this._dynamicEntries.values()) {
      dynamicShaders.push({
        hashHex: entry.hashHex,
        stage: entry.stage,
        stageCode: entry.stageCode,
        length: entry.length,
        overridesStatic: this._shaderEntries.has(entry.hash),
        source: entry.source,
        label: entry.label,
        entryPoint: entry.entryPoint,
        tags: entry.tags.slice(),
        revision: entry.revision,
        registeredAt: entry.registeredAt,
      });
    }

    return {
      stats,
      packNames: Array.from(this._packs.keys()),
      shaderIndex: {
        abiVersion: this._shaderIndexAbiVersion,
        packPath: this._shaderIndexPackPath,
        count: this._indexMeta.size,
      },
      dynamicShaders,
    };
  }

  /**
   * Return cache statistics across all loaded packs, dynamic shaders, and compiled modules.
   */
  getStats() {
    let cachedModules = 0;
    for (const map of this._deviceModules.values()) {
      cachedModules += map.size;
    }

    let totalPackBytes = 0;
    let rangeFetches = 0;
    let rangeBytesFetched = 0;
    let rangeFetchRetries = 0;
    let activeRangeFetches = 0;
    let peakConcurrentRangeFetches = 0;
    let cacheHits = 0;
    let cacheMisses = 0;
    let lruHits = 0;
    let lruMisses = 0;
    let lruBytes = 0;
    let cachedWgslBytes = 0;
    let cachedWgslEntries = 0;
    let wgslCacheEvictions = 0;
    let prefetchBatches = 0;
    let prefetchShaders = 0;
    let inFlightDeduplications = 0;
    let inFlightCount = 0;
    let hasBufferPack = false;
    let hasRangedPack = false;
    const packs = {};
    for (const [name, pack] of this._packs.entries()) {
      const s = pack.stats();
      totalPackBytes += sanitizeNonNegativeInt(s.byteLength, 0);
      rangeFetches += sanitizeNonNegativeInt(s.rangeFetchCount, 0);
      rangeBytesFetched += sanitizeNonNegativeInt(s.rangeBytesFetched, 0);
      rangeFetchRetries += sanitizeNonNegativeInt(s.rangeFetchRetries, 0);
      activeRangeFetches += sanitizeNonNegativeInt(s.activeRangeFetches, 0);
      const packPeak = sanitizeNonNegativeInt(s.peakConcurrentRangeFetches, 0);
      if (packPeak > peakConcurrentRangeFetches) {
        peakConcurrentRangeFetches = packPeak;
      }
      cacheHits += sanitizeNonNegativeInt(s.cacheHits, 0);
      cacheMisses += sanitizeNonNegativeInt(s.cacheMisses, 0);
      lruHits += sanitizeNonNegativeInt(s.lruHits, 0);
      lruMisses += sanitizeNonNegativeInt(s.lruMisses, 0);
      lruBytes += sanitizeNonNegativeInt(s.lruBytes, 0);
      cachedWgslBytes += sanitizeNonNegativeInt(s.cachedWgslBytes, 0);
      cachedWgslEntries += sanitizeNonNegativeInt(s.cachedWgslEntries, 0);
      wgslCacheEvictions += sanitizeNonNegativeInt(s.lruEvictions, 0);
      prefetchBatches += sanitizeNonNegativeInt(s.prefetchBatches, 0);
      prefetchShaders += sanitizeNonNegativeInt(s.prefetchShaders, 0);
      inFlightDeduplications += sanitizeNonNegativeInt(s.inFlightCoalesced, 0);
      inFlightCount += sanitizeNonNegativeInt(s.inFlightCount, 0);
      if (s.mode === 'ranged') {
        hasRangedPack = true;
      } else {
        hasBufferPack = true;
      }
      packs[name] = s;
    }
    if (activeRangeFetches > peakConcurrentRangeFetches) {
      peakConcurrentRangeFetches = activeRangeFetches;
    }

    const mode =
      hasRangedPack && hasBufferPack
        ? 'mixed'
        : hasRangedPack
          ? 'ranged'
          : 'buffer';

    const stageCounts = { vertex: 0, fragment: 0, compute: 0 };
    const dynamicStageCounts = { vertex: 0, fragment: 0, compute: 0 };
    let dynamicOverrideCount = 0;
    let dynamicPayloadBytes = 0;

    for (const [hash, entry] of this._dynamicEntries.entries()) {
      if (this._shaderEntries.has(hash)) {
        dynamicOverrideCount += 1;
      }
      dynamicPayloadBytes += entry.length;
      dynamicStageCounts[entry.stage] += 1;
      stageCounts[entry.stage] += 1;
    }

    for (const [hash, record] of this._shaderEntries.entries()) {
      if (!this._dynamicEntries.has(hash)) {
        stageCounts[record.entry.stage] += 1;
      }
    }

    const staticShaderCount = this._shaderEntries.size;
    const dynamicCount = this._dynamicEntries.size;
    const shaderCount = staticShaderCount + (dynamicCount - dynamicOverrideCount);

    const totalLookups = this._stats.hits + this._stats.misses;
    return {
      mode,
      packCount: this._packs.size,
      shaderCount,
      staticShaderCount,
      dynamicCount,
      dynamicOverrideCount,
      dynamicPayloadBytes,
      indexedMetadataCount: this._indexMeta.size,
      cachedModules,
      totalPackBytes,
      hits: this._stats.hits,
      misses: this._stats.misses,
      compiles: this._stats.compiles,
      evictions: this._stats.evictions,
      dynamicHits: this._stats.dynamicHits,
      hitRate: totalLookups > 0 ? this._stats.hits / totalLookups : 0,
      rangeFetches,
      rangeFetchCount: rangeFetches,
      rangeBytesFetched,
      rangeFetchRetries,
      activeRangeFetches,
      peakConcurrentRangeFetches,
      cacheHits,
      cacheMisses,
      lruHits,
      lruMisses,
      lruBytes,
      cachedWgslBytes,
      cachedWgslEntries,
      wgslCacheEvictions,
      lruEvictions: wgslCacheEvictions,
      prefetchBatches,
      prefetchShaders,
      inFlightDeduplications,
      inFlightCoalesced: inFlightDeduplications,
      inFlightCount,
      stageCounts,
      dynamicStageCounts,
      packs,
    };
  }
}
