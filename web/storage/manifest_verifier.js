// Browser & Node streaming manifest (`data-manifest.json`) and bootset (`bootset.json`)
// schema validator, partial-download readiness calculator, and live HTTP Range/ETag verifier.
// Matches `docs/io_manifest.md`, `tools/src/main.rs`, `tools/src/rpf9.rs`, `tools/src/scan.rs`,
// and `server/serve_local.py`. Zero external dependencies.

export const MANIFEST_ABI_VERSION = 1;
export const DEFAULT_SAMPLE_COUNT = 5;
export const DEFAULT_RANGE_PROBE_BYTES = 1;
export const DEFAULT_MAX_CONCURRENCY = 4;
export const MIN_MAX_CONCURRENCY = 1;
export const MAX_MAX_CONCURRENCY = 16;

const INTERNAL_SKIP_STATS = Symbol('manifestVerifierSkipModuleStats');

const moduleStats = {
  filesVerified: 0,
  bytesHashed: 0,
  mismatchCount: 0,
  abortedCount: 0,
  peakConcurrentHashes: 0,
  activeHashes: 0,
};

const moduleFingerprintCacheStats = {
  fingerprintSkippedCount: 0,
  bytesSkippedByFingerprint: 0,
};

function normalizeMaxConcurrency(raw, fallback = DEFAULT_MAX_CONCURRENCY) {
  if (raw === undefined || raw === null || typeof raw === 'boolean' || raw === '') {
    return fallback;
  }
  const num = Number(raw);
  if (Number.isNaN(num)) {
    return fallback;
  }
  if (num === Infinity) {
    return MAX_MAX_CONCURRENCY;
  }
  if (num === -Infinity) {
    return MIN_MAX_CONCURRENCY;
  }
  const floored = Math.floor(num);
  return Math.max(MIN_MAX_CONCURRENCY, Math.min(MAX_MAX_CONCURRENCY, floored));
}

function enterModuleHash() {
  moduleStats.activeHashes += 1;
  if (moduleStats.activeHashes > moduleStats.peakConcurrentHashes) {
    moduleStats.peakConcurrentHashes = moduleStats.activeHashes;
  }
}

function leaveModuleHash() {
  if (moduleStats.activeHashes > 0) {
    moduleStats.activeHashes -= 1;
  }
}

/**
 * Returns a sanitized snapshot of cumulative manifest & stream verification telemetry.
 * Contains only numeric counters (never host filesystem paths, URLs, or raw file bytes).
 *
 * @returns {{
 *   filesVerified: number,
 *   bytesHashed: number,
 *   mismatchCount: number,
 *   abortedCount: number,
 *   peakConcurrentHashes: number,
 *   activeHashes: number,
 * }}
 */
export function getStats() {
  return {
    filesVerified: moduleStats.filesVerified,
    bytesHashed: moduleStats.bytesHashed,
    mismatchCount: moduleStats.mismatchCount,
    abortedCount: moduleStats.abortedCount,
    peakConcurrentHashes: moduleStats.peakConcurrentHashes,
    activeHashes: moduleStats.activeHashes,
  };
}

/**
 * Returns a sanitized snapshot of cumulative warm-boot fingerprint cache skip counters.
 * Contains only non-negative integer counters (never paths, URLs, ETags, or file bytes).
 *
 * @returns {{
 *   fingerprintSkippedCount: number,
 *   bytesSkippedByFingerprint: number,
 * }}
 */
export function getFingerprintCacheStats() {
  return {
    fingerprintSkippedCount: moduleFingerprintCacheStats.fingerprintSkippedCount,
    bytesSkippedByFingerprint: moduleFingerprintCacheStats.bytesSkippedByFingerprint,
  };
}

/**
 * Resets cumulative manifest & stream verification telemetry counters.
 * Preserves any currently active hash count so in-flight workers cannot underflow counters.
 */
export function resetStats() {
  moduleStats.filesVerified = 0;
  moduleStats.bytesHashed = 0;
  moduleStats.mismatchCount = 0;
  moduleStats.abortedCount = 0;
  moduleStats.peakConcurrentHashes = moduleStats.activeHashes;
  moduleFingerprintCacheStats.fingerprintSkippedCount = 0;
  moduleFingerprintCacheStats.bytesSkippedByFingerprint = 0;
}

export const MISMATCH_CATEGORIES = Object.freeze({
  MISSING: 'missing',
  SIZE_MISMATCH: 'size_mismatch',
  HASH_MISMATCH: 'hash_mismatch',
});

export const STREAM_HASH_ALGORITHMS = Object.freeze({
  SHA256: 'sha256',
  CRC32: 'crc32',
  FNV1A32: 'fnv1a32',
  FNV1A64: 'fnv1a64',
});

export const SCHEMA_DIAGNOSTIC_CODES = Object.freeze({
  JSON_PARSE_ERROR: 'JSON_PARSE_ERROR',
  INVALID_ROOT: 'INVALID_ROOT',
  MISSING_ABI_VERSION: 'MISSING_ABI_VERSION',
  INVALID_ABI_VERSION: 'INVALID_ABI_VERSION',
  UNSUPPORTED_ABI_VERSION: 'UNSUPPORTED_ABI_VERSION',
  LEGACY_VERSION_FIELD: 'LEGACY_VERSION_FIELD',
  VERSION_MISMATCH: 'VERSION_MISMATCH',
  INVALID_FILE_COUNT: 'INVALID_FILE_COUNT',
  FILE_COUNT_MISMATCH: 'FILE_COUNT_MISMATCH',
  INVALID_TOTAL_BYTES: 'INVALID_TOTAL_BYTES',
  TOTAL_BYTES_MISMATCH: 'TOTAL_BYTES_MISMATCH',
  INVALID_FILES_ARRAY: 'INVALID_FILES_ARRAY',
  EMPTY_FILES_ARRAY: 'EMPTY_FILES_ARRAY',
  INVALID_ENTRY: 'INVALID_ENTRY',
  INVALID_PATH: 'INVALID_PATH',
  INVALID_SIZE: 'INVALID_SIZE',
  ZERO_BYTE_FILE: 'ZERO_BYTE_FILE',
  INVALID_SHA256: 'INVALID_SHA256',
  INVALID_CRC32: 'INVALID_CRC32',
  INVALID_FNV1A32: 'INVALID_FNV1A32',
  INVALID_FNV1A64: 'INVALID_FNV1A64',
  INVALID_CHUNK_SIZE: 'INVALID_CHUNK_SIZE',
  INVALID_CHUNK_HASHES: 'INVALID_CHUNK_HASHES',
  CHUNK_HASH_COUNT_MISMATCH: 'CHUNK_HASH_COUNT_MISMATCH',
  INVALID_ETAG: 'INVALID_ETAG',
  UNQUOTED_ETAG: 'UNQUOTED_ETAG',
  INVALID_INCOMPLETE_FLAG: 'INVALID_INCOMPLETE_FLAG',
  INVALID_STATUS: 'INVALID_STATUS',
  UNRECOGNIZED_STATUS: 'UNRECOGNIZED_STATUS',
  CONFLICTING_STATUS: 'CONFLICTING_STATUS',
  DUPLICATE_PATH: 'DUPLICATE_PATH',
  CASE_COLLISION: 'CASE_COLLISION',
  INCOMPLETE_SUFFIX: 'INCOMPLETE_SUFFIX',
  INVALID_BOOTSET_ROOT: 'INVALID_BOOTSET_ROOT',
  INVALID_OFFSET: 'INVALID_OFFSET',
  INVALID_LENGTH: 'INVALID_LENGTH',
  RANGE_OVERFLOW: 'RANGE_OVERFLOW',
  MISSING_MANIFEST_FILE: 'MISSING_MANIFEST_FILE',
  INCOMPLETE_MANIFEST_FILE: 'INCOMPLETE_MANIFEST_FILE',
  RANGE_OUT_OF_BOUNDS: 'RANGE_OUT_OF_BOUNDS',
  DUPLICATE_RANGE: 'DUPLICATE_RANGE',
  OVERLAPPING_RANGE: 'OVERLAPPING_RANGE',
});

// Case-insensitive temporary download suffixes matching tools/src/scan.rs (INCOMPLETE_SUFFIXES).
export const INCOMPLETE_DOWNLOAD_SUFFIXES = Object.freeze([
  '.!qb',
  '.part',
  '.crdownload',
  '.tmp',
  '.download',
  '.aria2',
  '.bc!',
]);

const COMPLETE_STATUS_VALUES = new Set(['complete', 'completed', 'ready', 'ok']);
const INCOMPLETE_STATUS_VALUES = new Set([
  'incomplete',
  'partial',
  'downloading',
  'pending',
  'missing',
  'unreadable',
  'sparse',
  'empty',
  'corrupt',
  'error',
]);

const SHA256_HEX_RE = /^[0-9a-fA-F]{64}$/;
const HEX32_RE = /^[0-9a-fA-F]{8}$/;
const HEX64_RE = /^[0-9a-fA-F]{16}$/;
const WINDOWS_DRIVE_RE = /^[A-Za-z]:/;
const CONTROL_CHAR_RE = /[\x00-\x1f\x7f]/;
const CONTENT_RANGE_RE = /^bytes\s+(\d+)-(\d+)\/(\d+|\*)$/i;
const QUOTED_ETAG_RE = /^(?:W\/)?"[^"]*"$/;
const STRONG_ETAG_RE = /^"[^"\x00-\x1f\x7f]+"$/;

function normalizeStrongEtag(raw) {
  if (typeof raw !== 'string') {
    return null;
  }
  const trimmed = raw.trim();
  if (!STRONG_ETAG_RE.test(trimmed)) {
    return null;
  }
  return trimmed;
}

/**
 * Checks whether an HTTP `ETag` header value is a valid, non-empty strong ETag per RFC 9110
 * (quoted, non-empty, no control characters, and NOT a weak `W/"..."` validator).
 *
 * @param {unknown} rawEtag
 * @returns {boolean}
 */
export function isStrongEtag(rawEtag) {
  return normalizeStrongEtag(rawEtag) !== null;
}

function normalizeFingerprintEntry(rawPath, rawRec) {
  const pathCheck = validateManifestPath(rawPath);
  if (!pathCheck.ok || !isPlainObject(rawRec)) {
    return null;
  }
  const etag = normalizeStrongEtag(rawRec.etag);
  if (etag === null) {
    return null;
  }
  if (!isNonNegativeSafeInt(rawRec.size)) {
    return null;
  }
  let sha256 = null;
  if (rawRec.sha256 !== undefined && rawRec.sha256 !== null) {
    if (typeof rawRec.sha256 !== 'string' || !SHA256_HEX_RE.test(rawRec.sha256.trim())) {
      return null;
    }
    sha256 = rawRec.sha256.trim().toLowerCase();
  }
  return {
    path: pathCheck.normalized,
    record: {
      etag,
      size: rawRec.size,
      sha256,
    },
  };
}

function ingestFingerprintSource(targetMap, source) {
  if (!source) {
    return;
  }
  if (source instanceof Map) {
    for (const [rawPath, rawRec] of source.entries()) {
      const normalized = normalizeFingerprintEntry(rawPath, rawRec);
      if (normalized) {
        targetMap.set(normalized.path, normalized.record);
      }
    }
    return;
  }
  if (!isPlainObject(source)) {
    return;
  }

  const looksLikeDirectRecord = 'etag' in source && 'size' in source;
  if (!looksLikeDirectRecord) {
    if (
      source.verifiedFingerprints !== undefined &&
      (source.verifiedFingerprints instanceof Map || isPlainObject(source.verifiedFingerprints))
    ) {
      ingestFingerprintSource(targetMap, source.verifiedFingerprints);
      return;
    }
    if (
      source.fingerprints !== undefined &&
      (source.fingerprints instanceof Map || isPlainObject(source.fingerprints))
    ) {
      ingestFingerprintSource(targetMap, source.fingerprints);
      return;
    }
    if (Array.isArray(source.results)) {
      for (const item of source.results) {
        if (!isPlainObject(item) || item.ok !== true) {
          continue;
        }
        const verifiedSha =
          isPlainObject(item.hash) && item.hash.verified === true
            ? item.hash.actual || item.hash.expected || null
            : null;
        const normalized = normalizeFingerprintEntry(item.path, {
          etag: item.etag ?? item.head?.etag ?? null,
          size: item.expectedSize ?? item.head?.contentLength ?? null,
          sha256: verifiedSha,
        });
        if (normalized) {
          targetMap.set(normalized.path, normalized.record);
        }
      }
      return;
    }
  }

  for (const [rawPath, rawRec] of Object.entries(source)) {
    const normalized = normalizeFingerprintEntry(rawPath, rawRec);
    if (normalized) {
      targetMap.set(normalized.path, normalized.record);
    }
  }
}

function buildFingerprintCacheMap(previousSnapshot, verifiedFingerprints) {
  const cache = new Map();
  if (previousSnapshot) {
    ingestFingerprintSource(cache, previousSnapshot);
  }
  if (verifiedFingerprints) {
    ingestFingerprintSource(cache, verifiedFingerprints);
  }
  return cache;
}

const FILE_URI_RE = /file:\/\/[^\s'")\]]+/gi;
const WINDOWS_UNC_PATH_RE = /\\\\[^\s'")\]]+/g;
const QUOTED_WINDOWS_ABS_PATH_RE = /(?<=['"])[A-Za-z]:[\\/][^'"\r\n]+(?=['"])/g;
const WINDOWS_ABS_PATH_RE = /\b[A-Za-z]:[\\/][^\s'")\]]*/g;
const QUOTED_POSIX_HOST_PATH_RE =
  /(?<=['"])\/(?:Users|home|tmp|var|etc|opt|mnt|Volumes|private|root|proc|sys|dev|usr)\/[^'"\r\n]+(?=['"])/g;
const POSIX_HOST_PATH_RE =
  /(?:^|(?<=[\s'"(=:\[,]))\/(?:Users|home|tmp|var|etc|opt|mnt|Volumes|private|root|proc|sys|dev|usr)\/[^\s'")\]]*/g;
const MAX_DIAGNOSTIC_TEXT_LEN = 256;

function defaultNow() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

function resolveNowFn(nowFn) {
  return typeof nowFn === 'function' ? nowFn : defaultNow;
}

function computeElapsedMs(startTime, endTime) {
  if (
    typeof startTime !== 'number' ||
    typeof endTime !== 'number' ||
    !Number.isFinite(startTime) ||
    !Number.isFinite(endTime)
  ) {
    return 0;
  }
  const delta = endTime - startTime;
  if (delta <= 0) {
    return 0;
  }
  return Math.round(delta * 1000) / 1000;
}

function computeBytesPerSecond(bytesVerified, elapsedMs) {
  if (
    typeof bytesVerified !== 'number' ||
    !Number.isFinite(bytesVerified) ||
    bytesVerified <= 0
  ) {
    return 0;
  }
  if (typeof elapsedMs !== 'number' || !Number.isFinite(elapsedMs) || elapsedMs <= 0) {
    return 0;
  }
  return Math.round((bytesVerified / (elapsedMs / 1000)) * 100) / 100;
}

function createMismatchesByCategory() {
  return {
    [MISMATCH_CATEGORIES.MISSING]: 0,
    [MISMATCH_CATEGORIES.SIZE_MISMATCH]: 0,
    [MISMATCH_CATEGORIES.HASH_MISMATCH]: 0,
  };
}

function countMismatches(mismatchesByCategory) {
  return (
    (mismatchesByCategory[MISMATCH_CATEGORIES.MISSING] || 0) +
    (mismatchesByCategory[MISMATCH_CATEGORIES.SIZE_MISMATCH] || 0) +
    (mismatchesByCategory[MISMATCH_CATEGORIES.HASH_MISMATCH] || 0)
  );
}

function buildTelemetry({
  elapsedMs,
  bytesVerified,
  filesVerified,
  mismatchesByCategory,
  aborted,
}) {
  const categories = {
    [MISMATCH_CATEGORIES.MISSING]: mismatchesByCategory[MISMATCH_CATEGORIES.MISSING] || 0,
    [MISMATCH_CATEGORIES.SIZE_MISMATCH]:
      mismatchesByCategory[MISMATCH_CATEGORIES.SIZE_MISMATCH] || 0,
    [MISMATCH_CATEGORIES.HASH_MISMATCH]:
      mismatchesByCategory[MISMATCH_CATEGORIES.HASH_MISMATCH] || 0,
  };
  const mismatchCount = countMismatches(categories);
  const safeBytes =
    typeof bytesVerified === 'number' && Number.isFinite(bytesVerified) && bytesVerified > 0
      ? Math.floor(bytesVerified)
      : 0;
  const safeFiles =
    typeof filesVerified === 'number' && Number.isFinite(filesVerified) && filesVerified > 0
      ? Math.floor(filesVerified)
      : 0;
  const safeElapsed =
    typeof elapsedMs === 'number' && Number.isFinite(elapsedMs) && elapsedMs > 0 ? elapsedMs : 0;
  return {
    elapsedMs: safeElapsed,
    bytesVerified: safeBytes,
    filesVerified: safeFiles,
    bytesPerSecond: computeBytesPerSecond(safeBytes, safeElapsed),
    mismatchCount,
    mismatchesByCategory: categories,
    aborted: Boolean(aborted),
  };
}

function sanitizeDiagnosticText(value, maxLen = MAX_DIAGNOSTIC_TEXT_LEN) {
  if (value === null || value === undefined) {
    return '';
  }
  let text = String(value);
  text = text
    .replace(FILE_URI_RE, '<redacted-path>')
    .replace(WINDOWS_UNC_PATH_RE, '<redacted-path>')
    .replace(QUOTED_WINDOWS_ABS_PATH_RE, '<redacted-path>')
    .replace(WINDOWS_ABS_PATH_RE, '<redacted-path>')
    .replace(QUOTED_POSIX_HOST_PATH_RE, '<redacted-path>')
    .replace(POSIX_HOST_PATH_RE, '<redacted-path>')
    .replace(/[\x00-\x1f\x7f]+/g, ' ')
    .trim();
  if (text.length > maxLen) {
    return `${text.slice(0, Math.max(0, maxLen - 3))}...`;
  }
  return text;
}

function sanitizeDiagnosticValue(value) {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'string') {
    return sanitizeDiagnosticText(value, MAX_DIAGNOSTIC_TEXT_LEN);
  }
  try {
    return sanitizeDiagnosticText(JSON.stringify(value), MAX_DIAGNOSTIC_TEXT_LEN);
  } catch {
    return sanitizeDiagnosticText(String(value), MAX_DIAGNOSTIC_TEXT_LEN);
  }
}

function createSchemaDiagnostic({
  severity,
  code,
  field,
  message,
  path = null,
  expected = null,
  actual = null,
}) {
  return {
    severity,
    code,
    field,
    path: path === null || path === undefined ? null : sanitizeDiagnosticText(path),
    expected: sanitizeDiagnosticValue(expected),
    actual: sanitizeDiagnosticValue(actual),
    message: sanitizeDiagnosticText(message),
  };
}

// ---------------------------------------------------------------------------
// Zero-dependency O(1)-memory incremental hashers (SHA-256, CRC32, FNV-1a 32/64)
// ---------------------------------------------------------------------------

const SHA256_INIT = new Uint32Array([
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
]);

const SHA256_K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

const CRC32_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i += 1) {
    let c = i;
    for (let k = 0; k < 8; k += 1) {
      c = (c & 1) !== 0 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[i] = c >>> 0;
  }
  return table;
})();

const FNV1A32_OFFSET_BASIS = 0x811c9dc5;
const FNV1A32_PRIME = 0x01000193;
const FNV1A64_OFFSET_LOW = 0x84222325;
const FNV1A64_OFFSET_HIGH = 0xcbf29ce4;

function createSha256State() {
  const h = new Uint32Array(SHA256_INIT);
  const w = new Uint32Array(64);
  const block = new Uint8Array(64);
  let blockLen = 0;
  let totalBytes = 0;

  const compress = (bytes, offset) => {
    for (let i = 0; i < 16; i += 1) {
      const o = offset + i * 4;
      w[i] =
        ((bytes[o] << 24) |
          (bytes[o + 1] << 16) |
          (bytes[o + 2] << 8) |
          bytes[o + 3]) >>>
        0;
    }
    for (let i = 16; i < 64; i += 1) {
      const w15 = w[i - 15];
      const w2 = w[i - 2];
      const s0 = ((w15 >>> 7) | (w15 << 25)) ^ ((w15 >>> 18) | (w15 << 14)) ^ (w15 >>> 3);
      const s1 = ((w2 >>> 17) | (w2 << 15)) ^ ((w2 >>> 19) | (w2 << 13)) ^ (w2 >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }

    let a = h[0];
    let b = h[1];
    let c = h[2];
    let d = h[3];
    let e = h[4];
    let f = h[5];
    let g = h[6];
    let hh = h[7];

    for (let i = 0; i < 64; i += 1) {
      const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const ch = (e & f) ^ (~e & g);
      const temp1 = (hh + S1 + ch + SHA256_K[i] + w[i]) >>> 0;
      const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (S0 + maj) >>> 0;

      hh = g;
      g = f;
      f = e;
      e = (d + temp1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temp1 + temp2) >>> 0;
    }

    h[0] = (h[0] + a) >>> 0;
    h[1] = (h[1] + b) >>> 0;
    h[2] = (h[2] + c) >>> 0;
    h[3] = (h[3] + d) >>> 0;
    h[4] = (h[4] + e) >>> 0;
    h[5] = (h[5] + f) >>> 0;
    h[6] = (h[6] + g) >>> 0;
    h[7] = (h[7] + hh) >>> 0;
  };

  return {
    reset() {
      h.set(SHA256_INIT);
      blockLen = 0;
      totalBytes = 0;
    },
    update(bytes) {
      const len = bytes.byteLength;
      if (len === 0) return;
      totalBytes += len;
      let offset = 0;

      if (blockLen > 0) {
        const take = Math.min(64 - blockLen, len);
        block.set(bytes.subarray(0, take), blockLen);
        blockLen += take;
        offset += take;
        if (blockLen === 64) {
          compress(block, 0);
          blockLen = 0;
        }
      }

      while (offset + 64 <= len) {
        compress(bytes, offset);
        offset += 64;
      }

      if (offset < len) {
        const rem = len - offset;
        block.set(bytes.subarray(offset, len), 0);
        blockLen = rem;
      }
    },
    digestHex() {
      block[blockLen] = 0x80;
      blockLen += 1;
      if (blockLen > 56) {
        block.fill(0, blockLen, 64);
        compress(block, 0);
        blockLen = 0;
      }
      block.fill(0, blockLen, 56);

      const totalBits = BigInt(totalBytes) * 8n;
      const hi = Number((totalBits >> 32n) & 0xffffffffn);
      const lo = Number(totalBits & 0xffffffffn);
      block[56] = (hi >>> 24) & 0xff;
      block[57] = (hi >>> 16) & 0xff;
      block[58] = (hi >>> 8) & 0xff;
      block[59] = hi & 0xff;
      block[60] = (lo >>> 24) & 0xff;
      block[61] = (lo >>> 16) & 0xff;
      block[62] = (lo >>> 8) & 0xff;
      block[63] = lo & 0xff;

      compress(block, 0);
      blockLen = 0;

      let hex = '';
      for (let i = 0; i < 8; i += 1) {
        hex += (h[i] >>> 0).toString(16).padStart(8, '0');
      }
      return hex;
    },
  };
}

function createCrc32State() {
  let crc = 0xffffffff;
  return {
    reset() {
      crc = 0xffffffff;
    },
    update(bytes) {
      let c = crc;
      for (let i = 0; i < bytes.byteLength; i += 1) {
        c = (c >>> 8) ^ CRC32_TABLE[(c ^ bytes[i]) & 0xff];
      }
      crc = c;
    },
    digestHex() {
      return ((crc ^ 0xffffffff) >>> 0).toString(16).padStart(8, '0');
    },
  };
}

function createFnv1a32State() {
  let hash = FNV1A32_OFFSET_BASIS;
  return {
    reset() {
      hash = FNV1A32_OFFSET_BASIS;
    },
    update(bytes) {
      let h = hash;
      for (let i = 0; i < bytes.byteLength; i += 1) {
        h ^= bytes[i];
        h = Math.imul(h, FNV1A32_PRIME) >>> 0;
      }
      hash = h;
    },
    digestHex() {
      return (hash >>> 0).toString(16).padStart(8, '0');
    },
  };
}

function createFnv1a64State() {
  let low = FNV1A64_OFFSET_LOW;
  let high = FNV1A64_OFFSET_HIGH;
  return {
    reset() {
      low = FNV1A64_OFFSET_LOW;
      high = FNV1A64_OFFSET_HIGH;
    },
    update(bytes) {
      let lo = low;
      let hi = high;
      for (let i = 0; i < bytes.byteLength; i += 1) {
        lo = (lo ^ bytes[i]) >>> 0;
        // FNV prime = 2^40 + 435. lo * 435 is exact in a Number
        // (< 2^41); carry its upper word into the high half modulo 2^32.
        const product = lo * 435;
        hi = (Math.imul(hi, 435) + Math.imul(lo, 256) + Math.floor(product / 0x100000000)) >>> 0;
        lo = product >>> 0;
      }
      low = lo;
      high = hi;
    },
    digestHex() {
      return high.toString(16).padStart(8, '0') + low.toString(16).padStart(8, '0');
    },
  };
}

function normalizeHashAlgorithm(raw) {
  if (raw === undefined || raw === null || raw === '') {
    return STREAM_HASH_ALGORITHMS.SHA256;
  }
  if (typeof raw !== 'string') {
    throw new RangeError(`Unsupported hash algorithm: ${String(raw)}`);
  }
  const key = raw.trim().toLowerCase().replace(/[\s_-]+/g, '');
  if (key === 'sha256') return STREAM_HASH_ALGORITHMS.SHA256;
  if (key === 'crc32') return STREAM_HASH_ALGORITHMS.CRC32;
  if (key === 'fnv1a' || key === 'fnv1a32') return STREAM_HASH_ALGORITHMS.FNV1A32;
  if (key === 'fnv1a64') return STREAM_HASH_ALGORITHMS.FNV1A64;
  throw new RangeError(`Unsupported hash algorithm: ${sanitizeDiagnosticText(raw, 64)}`);
}

function digestPatternForAlgorithm(algorithm) {
  switch (algorithm) {
    case STREAM_HASH_ALGORITHMS.SHA256:
      return SHA256_HEX_RE;
    case STREAM_HASH_ALGORITHMS.CRC32:
    case STREAM_HASH_ALGORITHMS.FNV1A32:
      return HEX32_RE;
    case STREAM_HASH_ALGORITHMS.FNV1A64:
      return HEX64_RE;
    default:
      return SHA256_HEX_RE;
  }
}

function expectedHexLengthForAlgorithm(algorithm) {
  switch (algorithm) {
    case STREAM_HASH_ALGORITHMS.SHA256:
      return 64;
    case STREAM_HASH_ALGORITHMS.CRC32:
    case STREAM_HASH_ALGORITHMS.FNV1A32:
      return 8;
    case STREAM_HASH_ALGORITHMS.FNV1A64:
      return 16;
    default:
      return 64;
  }
}

function createHasherState(algorithm) {
  switch (algorithm) {
    case STREAM_HASH_ALGORITHMS.SHA256:
      return createSha256State();
    case STREAM_HASH_ALGORITHMS.CRC32:
      return createCrc32State();
    case STREAM_HASH_ALGORITHMS.FNV1A32:
      return createFnv1a32State();
    case STREAM_HASH_ALGORITHMS.FNV1A64:
      return createFnv1a64State();
    default:
      throw new RangeError(`Unsupported hash algorithm: ${String(algorithm)}`);
  }
}

function toUint8View(chunk) {
  if (chunk instanceof Uint8Array) {
    return chunk;
  }
  if (chunk instanceof ArrayBuffer) {
    return new Uint8Array(chunk);
  }
  if (typeof SharedArrayBuffer !== 'undefined' && chunk instanceof SharedArrayBuffer) {
    return new Uint8Array(chunk);
  }
  if (ArrayBuffer.isView(chunk)) {
    return new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength);
  }
  throw new TypeError(
    'Chunk must be a Uint8Array, ArrayBuffer, SharedArrayBuffer, or ArrayBufferView',
  );
}

function normalizeExpectedDigest(raw, algorithm, label = 'expectedDigest') {
  if (raw === undefined || raw === null) {
    return null;
  }
  if (typeof raw !== 'string') {
    throw new TypeError(`${label} must be a hexadecimal string`);
  }
  const trimmed = raw.trim().toLowerCase();
  const pattern = digestPatternForAlgorithm(algorithm);
  if (!pattern.test(trimmed)) {
    const hexLen = expectedHexLengthForAlgorithm(algorithm);
    throw new RangeError(
      `${label} for ${algorithm} must be a ${hexLen}-character hexadecimal string`,
    );
  }
  return trimmed;
}

function createAbortError(signal) {
  if (signal && signal.reason instanceof Error) {
    return signal.reason;
  }
  const err = new Error('Verification aborted');
  err.name = 'AbortError';
  return err;
}

function isAbortError(err) {
  return Boolean(
    err &&
      typeof err === 'object' &&
      (err.name === 'AbortError' || err.code === 'ABORT_ERR'),
  );
}

async function raceWithAbort(promise, signal) {
  if (!signal) {
    return promise;
  }
  if (signal.aborted) {
    Promise.resolve(promise).catch(() => {});
    throw createAbortError(signal);
  }
  return new Promise((resolve, reject) => {
    let settled = false;
    const onAbort = () => {
      if (settled) return;
      settled = true;
      Promise.resolve(promise).catch(() => {});
      reject(createAbortError(signal));
    };
    signal.addEventListener('abort', onAbort, { once: true });
    Promise.resolve(promise).then(
      (value) => {
        if (settled) return;
        settled = true;
        signal.removeEventListener('abort', onAbort);
        resolve(value);
      },
      (err) => {
        if (settled) return;
        settled = true;
        signal.removeEventListener('abort', onAbort);
        reject(err);
      },
    );
  });
}

async function computeSha256Hex(bytes) {
  const view =
    bytes instanceof Uint8Array
      ? bytes
      : bytes instanceof ArrayBuffer
        ? new Uint8Array(bytes)
        : new Uint8Array(0);
  const subtle = globalThis.crypto?.subtle;
  if (!subtle || typeof subtle.digest !== 'function') {
    return computeDigest(view, STREAM_HASH_ALGORITHMS.SHA256);
  }
  const digest = await subtle.digest('SHA-256', view);
  const out = new Uint8Array(digest);
  let hex = '';
  for (let i = 0; i < out.length; i += 1) {
    hex += out[i].toString(16).padStart(2, '0');
  }
  return hex;
}

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isNonNegativeSafeInt(value) {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;
}

function isPositiveSafeInt(value) {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0;
}

function compareAscii(a, b) {
  return a < b ? -1 : a > b ? 1 : 0;
}

function getHeader(headers, name) {
  if (!headers) return null;
  if (typeof headers.get === 'function') {
    const val = headers.get(name);
    return val === undefined ? null : val;
  }
  const target = name.toLowerCase();
  for (const [k, v] of Object.entries(headers)) {
    if (k.toLowerCase() === target) {
      return v == null ? null : String(v);
    }
  }
  return null;
}

function normalizeDataBase(base) {
  const raw = typeof base === 'string' && base.length > 0 ? base : '/data/';
  return raw.endsWith('/') ? raw : `${raw}/`;
}

function encodeRelPathForUrl(relPath) {
  return relPath
    .split('/')
    .map((segment) => encodeURIComponent(segment))
    .join('/');
}

/**
 * Checks whether a relative path ends with a known incomplete-download suffix
 * (case-insensitive, matching tools/src/scan.rs).
 *
 * @param {string} filePath
 * @returns {boolean}
 */
export function hasIncompleteSuffix(filePath) {
  if (typeof filePath !== 'string' || filePath.length === 0) {
    return false;
  }
  const lower = filePath.toLowerCase();
  for (const suffix of INCOMPLETE_DOWNLOAD_SUFFIXES) {
    if (lower.endsWith(suffix)) {
      return true;
    }
  }
  return false;
}

/**
 * Validates and normalizes a manifest/bootset relative POSIX path.
 * Rejects path traversal (`..`, URL-encoded `..`), leading/trailing slashes,
 * backslashes, Windows drive letters (`C:`), `.` or empty segments, and control chars.
 *
 * @param {unknown} rawPath
 * @returns {{ ok: boolean, normalized: string | null, error: string | null }}
 */
export function validateManifestPath(rawPath) {
  if (typeof rawPath !== 'string' || rawPath.length === 0) {
    return { ok: false, normalized: null, error: 'Path must be a non-empty string' };
  }
  if (CONTROL_CHAR_RE.test(rawPath)) {
    return {
      ok: false,
      normalized: null,
      error: `Path contains forbidden control character or NUL byte: ${JSON.stringify(rawPath)}`,
    };
  }
  if (rawPath.includes('\\')) {
    return {
      ok: false,
      normalized: null,
      error: `Path must use POSIX forward slashes and not backslashes: "${rawPath}"`,
    };
  }
  if (WINDOWS_DRIVE_RE.test(rawPath)) {
    return {
      ok: false,
      normalized: null,
      error: `Path must not contain a drive letter: "${rawPath}"`,
    };
  }
  if (rawPath.startsWith('/')) {
    return {
      ok: false,
      normalized: null,
      error: `Path must be relative (leading slash forbidden): "${rawPath}"`,
    };
  }
  if (rawPath.endsWith('/')) {
    return {
      ok: false,
      normalized: null,
      error: `Path must reference a file (trailing slash forbidden): "${rawPath}"`,
    };
  }

  const segments = rawPath.split('/');
  for (const seg of segments) {
    if (seg.length === 0) {
      return {
        ok: false,
        normalized: null,
        error: `Path contains empty segment ("//"): "${rawPath}"`,
      };
    }
    if (seg === '.' || seg === '..') {
      return {
        ok: false,
        normalized: null,
        error: `Path traversal or dot segment ("${seg}") forbidden: "${rawPath}"`,
      };
    }
    if (WINDOWS_DRIVE_RE.test(seg)) {
      return {
        ok: false,
        normalized: null,
        error: `Path segment must not contain a drive letter: "${rawPath}"`,
      };
    }
    // Defense-in-depth: reject percent-encoded traversal or slash injection.
    if (seg.includes('%')) {
      try {
        const decoded = decodeURIComponent(seg);
        if (
          decoded === '.' ||
          decoded === '..' ||
          decoded.includes('/') ||
          decoded.includes('\\') ||
          CONTROL_CHAR_RE.test(decoded)
        ) {
          return {
            ok: false,
            normalized: null,
            error: `Path contains encoded traversal or separator in segment "${seg}": "${rawPath}"`,
          };
        }
      } catch {
        // Literal '%' that is not a valid percent-escape is allowed as a filename character.
      }
    }
  }

  return { ok: true, normalized: rawPath, error: null };
}

/**
 * Parses an HTTP `Content-Range` header of the form `bytes <start>-<end>/<total|*>`.
 *
 * @param {string | null | undefined} header
 * @returns {{ start: number, end: number, total: number | null } | null}
 */
export function parseContentRange(header) {
  if (typeof header !== 'string') {
    return null;
  }
  const match = CONTENT_RANGE_RE.exec(header.trim());
  if (!match) {
    return null;
  }
  const start = Number(match[1]);
  const end = Number(match[2]);
  const total = match[3] === '*' ? null : Number(match[3]);
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start) {
    return null;
  }
  if (total !== null) {
    if (!Number.isSafeInteger(total) || total <= 0 || end >= total) {
      return null;
    }
  }
  return { start, end, total };
}

function emptyManifestSummary() {
  return {
    abiVersion: null,
    fileCount: 0,
    declaredFileCount: null,
    totalBytes: 0,
    declaredTotalBytes: null,
    completeFiles: 0,
    incompleteFiles: 0,
    completeBytes: 0,
    incompleteBytes: 0,
    hashedFiles: 0,
    etagFiles: 0,
    duplicateCount: 0,
  };
}

/**
 * Validates a `data-manifest.json` object (or JSON string) against the schema defined in
 * `docs/io_manifest.md` and `tools/src/main.rs`.
 *
 * @param {unknown} manifestInput
 * @param {{ signal?: AbortSignal | null, nowFn?: (() => number) | null }} [options]
 * @returns {{
 *   ok: boolean,
 *   errors: string[],
 *   warnings: string[],
 *   fileMap: Map<string, {
 *     path: string,
 *     size: number,
 *     sha256: string | null,
 *     etag: string | null,
 *     incomplete: boolean,
 *     status: string,
 *     incompleteReason: string | null,
 *   }>,
 *   summary: {
 *     abiVersion: number | null,
 *     fileCount: number,
 *     declaredFileCount: number | null,
 *     totalBytes: number,
 *     declaredTotalBytes: number | null,
 *     completeFiles: number,
 *     incompleteFiles: number,
 *     completeBytes: number,
 *     incompleteBytes: number,
 *     hashedFiles: number,
 *     etagFiles: number,
 *     duplicateCount: number,
 *   },
 *   aborted: boolean,
 *   elapsedMs: number,
 *   bytesVerified: number,
 *   filesVerified: number,
 *   mismatchCount: number,
 *   mismatchesByCategory: { missing: number, size_mismatch: number, hash_mismatch: number },
 *   telemetry: {
 *     elapsedMs: number,
 *     bytesVerified: number,
 *     filesVerified: number,
 *     bytesPerSecond: number,
 *     mismatchCount: number,
 *     mismatchesByCategory: { missing: number, size_mismatch: number, hash_mismatch: number },
 *     aborted: boolean,
 *   },
 * }}
 */
export function validateDataManifest(manifestInput, options = {}) {
  const { signal = null, nowFn = null } = isPlainObject(options) ? options : {};
  const now = resolveNowFn(nowFn);
  const startTime = now();

  const errors = [];
  const warnings = [];
  const schemaDiagnostics = [];
  const fileMap = new Map();
  const summary = emptyManifestSummary();
  const mismatchesByCategory = createMismatchesByCategory();

  const recordSchemaError = (code, field, message, extra = {}) => {
    errors.push(message);
    schemaDiagnostics.push(
      createSchemaDiagnostic({
        severity: 'error',
        code,
        field,
        message,
        path: extra.path ?? null,
        expected: extra.expected ?? null,
        actual: extra.actual ?? null,
      }),
    );
  };

  const recordSchemaWarning = (code, field, message, extra = {}) => {
    warnings.push(message);
    schemaDiagnostics.push(
      createSchemaDiagnostic({
        severity: 'warning',
        code,
        field,
        message,
        path: extra.path ?? null,
        expected: extra.expected ?? null,
        actual: extra.actual ?? null,
      }),
    );
  };

  const finalize = (aborted = false) => {
    const elapsedMs = computeElapsedMs(startTime, now());
    const telemetry = buildTelemetry({
      elapsedMs,
      bytesVerified: summary.totalBytes,
      filesVerified: summary.fileCount,
      mismatchesByCategory,
      aborted,
    });
    return {
      ok: !aborted && errors.length === 0,
      errors,
      warnings,
      schemaDiagnostics,
      fileMap,
      summary,
      aborted: telemetry.aborted,
      elapsedMs: telemetry.elapsedMs,
      bytesVerified: telemetry.bytesVerified,
      filesVerified: telemetry.filesVerified,
      mismatchCount: telemetry.mismatchCount,
      mismatchesByCategory: { ...telemetry.mismatchesByCategory },
      telemetry,
    };
  };

  if (signal?.aborted) {
    return finalize(true);
  }

  let manifest = manifestInput;
  if (typeof manifestInput === 'string') {
    try {
      manifest = JSON.parse(manifestInput);
    } catch (err) {
      recordSchemaError(
        SCHEMA_DIAGNOSTIC_CODES.JSON_PARSE_ERROR,
        '$',
        `Manifest JSON parse error: ${sanitizeDiagnosticText(err instanceof Error ? err.message : String(err))}`,
        { expected: 'valid JSON object', actual: 'malformed JSON' },
      );
      return finalize(false);
    }
  }

  if (!isPlainObject(manifest)) {
    recordSchemaError(
      SCHEMA_DIAGNOSTIC_CODES.INVALID_ROOT,
      '$',
      'Manifest must be a JSON object',
      { expected: 'object', actual: manifest === null ? 'null' : Array.isArray(manifest) ? 'array' : typeof manifest },
    );
    return finalize(false);
  }

  // Version validation: canonical field is `abi_version`, with optional fallback to `version`.
  let resolvedVersion = null;
  if (manifest.abi_version !== undefined) {
    if (!isPositiveSafeInt(manifest.abi_version)) {
      recordSchemaError(
        SCHEMA_DIAGNOSTIC_CODES.INVALID_ABI_VERSION,
        'abi_version',
        `Manifest "abi_version" must be a positive integer (got ${JSON.stringify(manifest.abi_version)})`,
        { expected: MANIFEST_ABI_VERSION, actual: manifest.abi_version },
      );
    } else if (manifest.abi_version !== MANIFEST_ABI_VERSION) {
      recordSchemaError(
        SCHEMA_DIAGNOSTIC_CODES.UNSUPPORTED_ABI_VERSION,
        'abi_version',
        `Unsupported manifest "abi_version": ${manifest.abi_version} (expected ${MANIFEST_ABI_VERSION})`,
        { expected: MANIFEST_ABI_VERSION, actual: manifest.abi_version },
      );
    } else {
      resolvedVersion = manifest.abi_version;
    }
    if (manifest.version !== undefined && manifest.version !== manifest.abi_version) {
      recordSchemaWarning(
        SCHEMA_DIAGNOSTIC_CODES.VERSION_MISMATCH,
        'version',
        `Manifest "version" (${JSON.stringify(manifest.version)}) differs from "abi_version" (${JSON.stringify(manifest.abi_version)})`,
        { expected: manifest.abi_version, actual: manifest.version },
      );
    }
  } else if (manifest.version !== undefined) {
    if (!isPositiveSafeInt(manifest.version)) {
      recordSchemaError(
        SCHEMA_DIAGNOSTIC_CODES.INVALID_ABI_VERSION,
        'version',
        `Manifest "version" must be a positive integer (got ${JSON.stringify(manifest.version)})`,
        { expected: MANIFEST_ABI_VERSION, actual: manifest.version },
      );
    } else if (manifest.version !== MANIFEST_ABI_VERSION) {
      recordSchemaError(
        SCHEMA_DIAGNOSTIC_CODES.UNSUPPORTED_ABI_VERSION,
        'version',
        `Unsupported manifest "version": ${manifest.version} (expected ${MANIFEST_ABI_VERSION})`,
        { expected: MANIFEST_ABI_VERSION, actual: manifest.version },
      );
    } else {
      resolvedVersion = manifest.version;
      recordSchemaWarning(
        SCHEMA_DIAGNOSTIC_CODES.LEGACY_VERSION_FIELD,
        'version',
        'Manifest uses "version" instead of canonical "abi_version"',
        { expected: 'abi_version', actual: 'version' },
      );
    }
  } else {
    recordSchemaError(
      SCHEMA_DIAGNOSTIC_CODES.MISSING_ABI_VERSION,
      'abi_version',
      `Manifest is missing required "abi_version" (expected ${MANIFEST_ABI_VERSION})`,
      { expected: MANIFEST_ABI_VERSION, actual: null },
    );
  }
  summary.abiVersion = resolvedVersion;

  if (manifest.file_count !== undefined) {
    if (!isNonNegativeSafeInt(manifest.file_count)) {
      recordSchemaError(
        SCHEMA_DIAGNOSTIC_CODES.INVALID_FILE_COUNT,
        'file_count',
        `Manifest "file_count" must be a non-negative integer (got ${JSON.stringify(manifest.file_count)})`,
        { expected: 'non-negative integer', actual: manifest.file_count },
      );
    } else {
      summary.declaredFileCount = manifest.file_count;
    }
  }

  if (manifest.total_bytes !== undefined) {
    if (!isNonNegativeSafeInt(manifest.total_bytes)) {
      recordSchemaError(
        SCHEMA_DIAGNOSTIC_CODES.INVALID_TOTAL_BYTES,
        'total_bytes',
        `Manifest "total_bytes" must be a non-negative integer (got ${JSON.stringify(manifest.total_bytes)})`,
        { expected: 'non-negative integer', actual: manifest.total_bytes },
      );
    } else {
      summary.declaredTotalBytes = manifest.total_bytes;
    }
  }

  if (!Array.isArray(manifest.files)) {
    recordSchemaError(
      SCHEMA_DIAGNOSTIC_CODES.INVALID_FILES_ARRAY,
      'files',
      'Manifest "files" must be an array',
      { expected: 'array', actual: manifest.files === null ? 'null' : typeof manifest.files },
    );
    return finalize(false);
  }

  if (manifest.files.length === 0) {
    recordSchemaWarning(
      SCHEMA_DIAGNOSTIC_CODES.EMPTY_FILES_ARRAY,
      'files',
      'Manifest "files" array is empty',
      { expected: '>= 1 entries', actual: 0 },
    );
  }

  if (summary.declaredFileCount !== null && summary.declaredFileCount !== manifest.files.length) {
    recordSchemaError(
      SCHEMA_DIAGNOSTIC_CODES.FILE_COUNT_MISMATCH,
      'file_count',
      `Manifest "file_count" (${summary.declaredFileCount}) does not match files.length (${manifest.files.length})`,
      { expected: manifest.files.length, actual: summary.declaredFileCount },
    );
  }

  const lowerPathMap = new Map();
  let allSizesValid = true;
  let sumAllValidSizes = 0;

  for (let i = 0; i < manifest.files.length; i += 1) {
    if (signal?.aborted) {
      return finalize(true);
    }
    const raw = manifest.files[i];
    const label = `files[${i}]`;

    if (!isPlainObject(raw)) {
      recordSchemaError(
        SCHEMA_DIAGNOSTIC_CODES.INVALID_ENTRY,
        label,
        `${label} must be an object`,
        { expected: 'object', actual: raw === null ? 'null' : Array.isArray(raw) ? 'array' : typeof raw },
      );
      allSizesValid = false;
      continue;
    }

    const pathCheck = validateManifestPath(raw.path);
    const diagPath = pathCheck.ok ? pathCheck.normalized : null;
    if (!pathCheck.ok) {
      recordSchemaError(
        SCHEMA_DIAGNOSTIC_CODES.INVALID_PATH,
        `${label}.path`,
        `${label}.path: ${pathCheck.error}`,
        { expected: 'clean relative POSIX path', actual: raw.path },
      );
    }

    const sizeValid = isNonNegativeSafeInt(raw.size);
    if (!sizeValid) {
      recordSchemaError(
        SCHEMA_DIAGNOSTIC_CODES.INVALID_SIZE,
        `${label}.size`,
        `${label}.size must be a non-negative safe integer (got ${JSON.stringify(raw.size)})`,
        { path: diagPath, expected: 'non-negative safe integer', actual: raw.size },
      );
      allSizesValid = false;
    } else {
      sumAllValidSizes += raw.size;
      if (raw.size === 0) {
        recordSchemaWarning(
          SCHEMA_DIAGNOSTIC_CODES.ZERO_BYTE_FILE,
          `${label}.size`,
          `${label} ("${pathCheck.normalized || raw.path}") has zero byte size`,
          { path: diagPath, expected: '> 0 bytes', actual: 0 },
        );
      }
    }

    // Optional sha256 validation (64 hex chars).
    let sha256 = null;
    if (raw.sha256 !== undefined && raw.sha256 !== null) {
      if (typeof raw.sha256 !== 'string' || !SHA256_HEX_RE.test(raw.sha256)) {
        recordSchemaError(
          SCHEMA_DIAGNOSTIC_CODES.INVALID_SHA256,
          `${label}.sha256`,
          `${label}.sha256 must be a 64-character hexadecimal string (got ${JSON.stringify(raw.sha256)})`,
          { path: diagPath, expected: '64 hex chars', actual: raw.sha256 },
        );
      } else {
        sha256 = raw.sha256.toLowerCase();
      }
    }

    // Optional crc32 validation (8 hex chars).
    let crc32 = undefined;
    if (raw.crc32 !== undefined && raw.crc32 !== null) {
      if (typeof raw.crc32 !== 'string' || !HEX32_RE.test(raw.crc32)) {
        recordSchemaError(
          SCHEMA_DIAGNOSTIC_CODES.INVALID_CRC32,
          `${label}.crc32`,
          `${label}.crc32 must be an 8-character hexadecimal string (got ${JSON.stringify(raw.crc32)})`,
          { path: diagPath, expected: '8 hex chars', actual: raw.crc32 },
        );
      } else {
        crc32 = raw.crc32.toLowerCase();
      }
    }

    // Optional fnv1a32 validation (8 hex chars).
    let fnv1a32 = undefined;
    if (raw.fnv1a32 !== undefined && raw.fnv1a32 !== null) {
      if (typeof raw.fnv1a32 !== 'string' || !HEX32_RE.test(raw.fnv1a32)) {
        recordSchemaError(
          SCHEMA_DIAGNOSTIC_CODES.INVALID_FNV1A32,
          `${label}.fnv1a32`,
          `${label}.fnv1a32 must be an 8-character hexadecimal string (got ${JSON.stringify(raw.fnv1a32)})`,
          { path: diagPath, expected: '8 hex chars', actual: raw.fnv1a32 },
        );
      } else {
        fnv1a32 = raw.fnv1a32.toLowerCase();
      }
    }

    // Optional fnv1a64 validation (16 hex chars).
    let fnv1a64 = undefined;
    if (raw.fnv1a64 !== undefined && raw.fnv1a64 !== null) {
      if (typeof raw.fnv1a64 !== 'string' || !HEX64_RE.test(raw.fnv1a64)) {
        recordSchemaError(
          SCHEMA_DIAGNOSTIC_CODES.INVALID_FNV1A64,
          `${label}.fnv1a64`,
          `${label}.fnv1a64 must be a 16-character hexadecimal string (got ${JSON.stringify(raw.fnv1a64)})`,
          { path: diagPath, expected: '16 hex chars', actual: raw.fnv1a64 },
        );
      } else {
        fnv1a64 = raw.fnv1a64.toLowerCase();
      }
    }

    // Optional chunk_size validation (positive safe integer).
    let chunkSize = undefined;
    if (raw.chunk_size !== undefined && raw.chunk_size !== null) {
      if (!isPositiveSafeInt(raw.chunk_size)) {
        recordSchemaError(
          SCHEMA_DIAGNOSTIC_CODES.INVALID_CHUNK_SIZE,
          `${label}.chunk_size`,
          `${label}.chunk_size must be a positive safe integer (got ${JSON.stringify(raw.chunk_size)})`,
          { path: diagPath, expected: 'positive safe integer', actual: raw.chunk_size },
        );
      } else {
        chunkSize = raw.chunk_size;
      }
    }

    // Optional chunk_hashes validation (array of consistent valid hex digests).
    let chunkHashes = undefined;
    if (raw.chunk_hashes !== undefined && raw.chunk_hashes !== null) {
      if (!Array.isArray(raw.chunk_hashes)) {
        recordSchemaError(
          SCHEMA_DIAGNOSTIC_CODES.INVALID_CHUNK_HASHES,
          `${label}.chunk_hashes`,
          `${label}.chunk_hashes must be an array of hexadecimal digest strings`,
          { path: diagPath, expected: 'array of hex digests', actual: typeof raw.chunk_hashes },
        );
      } else {
        const normalizedChunks = [];
        let expectedLen = null;
        let chunksValid = true;
        for (let c = 0; c < raw.chunk_hashes.length; c += 1) {
          const item = raw.chunk_hashes[c];
          if (
            typeof item !== 'string' ||
            !(SHA256_HEX_RE.test(item) || HEX64_RE.test(item) || HEX32_RE.test(item))
          ) {
            recordSchemaError(
              SCHEMA_DIAGNOSTIC_CODES.INVALID_CHUNK_HASHES,
              `${label}.chunk_hashes[${c}]`,
              `${label}.chunk_hashes[${c}] must be a valid 8-, 16-, or 64-character hexadecimal digest`,
              { path: diagPath, expected: '8/16/64 hex chars', actual: item },
            );
            chunksValid = false;
            continue;
          }
          if (expectedLen === null) {
            expectedLen = item.length;
          } else if (item.length !== expectedLen) {
            recordSchemaError(
              SCHEMA_DIAGNOSTIC_CODES.INVALID_CHUNK_HASHES,
              `${label}.chunk_hashes[${c}]`,
              `${label}.chunk_hashes[${c}] digest length (${item.length}) does not match earlier chunk digest length (${expectedLen})`,
              { path: diagPath, expected: expectedLen, actual: item.length },
            );
            chunksValid = false;
            continue;
          }
          normalizedChunks.push(item.toLowerCase());
        }
        if (chunksValid) {
          chunkHashes = normalizedChunks;
          if (sizeValid && chunkSize !== undefined) {
            const expectedCount = raw.size === 0 ? 0 : Math.ceil(raw.size / chunkSize);
            if (normalizedChunks.length !== expectedCount) {
              recordSchemaError(
                SCHEMA_DIAGNOSTIC_CODES.CHUNK_HASH_COUNT_MISMATCH,
                `${label}.chunk_hashes`,
                `${label}.chunk_hashes length (${normalizedChunks.length}) does not match expected chunk count (${expectedCount}) for size ${raw.size} and chunk_size ${chunkSize}`,
                { path: diagPath, expected: expectedCount, actual: normalizedChunks.length },
              );
            }
          }
        }
      }
    }

    // Optional etag validation.
    let etag = null;
    if (raw.etag !== undefined && raw.etag !== null) {
      if (typeof raw.etag !== 'string' || raw.etag.trim().length === 0) {
        recordSchemaError(
          SCHEMA_DIAGNOSTIC_CODES.INVALID_ETAG,
          `${label}.etag`,
          `${label}.etag must be a non-empty string`,
          { path: diagPath, expected: 'non-empty string', actual: raw.etag },
        );
      } else {
        etag = raw.etag.trim();
        if (!QUOTED_ETAG_RE.test(etag)) {
          recordSchemaWarning(
            SCHEMA_DIAGNOSTIC_CODES.UNQUOTED_ETAG,
            `${label}.etag`,
            `${label}.etag (${JSON.stringify(etag)}) is not RFC 9110 quoted`,
            { path: diagPath, expected: 'RFC 9110 quoted ETag', actual: etag },
          );
        }
      }
    }

    // Optional incomplete boolean flag.
    let explicitIncomplete = null;
    if (raw.incomplete !== undefined) {
      if (typeof raw.incomplete !== 'boolean') {
        recordSchemaError(
          SCHEMA_DIAGNOSTIC_CODES.INVALID_INCOMPLETE_FLAG,
          `${label}.incomplete`,
          `${label}.incomplete must be a boolean (got ${JSON.stringify(raw.incomplete)})`,
          { path: diagPath, expected: 'boolean', actual: raw.incomplete },
        );
      } else {
        explicitIncomplete = raw.incomplete;
      }
    }

    // Optional status string flag.
    let status = null;
    let statusMarksIncomplete = false;
    let statusMarksComplete = false;
    if (raw.status !== undefined && raw.status !== null) {
      if (typeof raw.status !== 'string' || raw.status.trim().length === 0) {
        recordSchemaError(
          SCHEMA_DIAGNOSTIC_CODES.INVALID_STATUS,
          `${label}.status`,
          `${label}.status must be a non-empty string`,
          { path: diagPath, expected: 'non-empty string', actual: raw.status },
        );
      } else {
        const lowerStatus = raw.status.trim().toLowerCase();
        status = lowerStatus;
        if (INCOMPLETE_STATUS_VALUES.has(lowerStatus)) {
          statusMarksIncomplete = true;
        } else if (COMPLETE_STATUS_VALUES.has(lowerStatus)) {
          statusMarksComplete = true;
        } else {
          recordSchemaWarning(
            SCHEMA_DIAGNOSTIC_CODES.UNRECOGNIZED_STATUS,
            `${label}.status`,
            `${label}.status has unrecognized value "${raw.status}"`,
            { path: diagPath, expected: 'known manifest status value', actual: raw.status },
          );
        }
      }
    }

    if (explicitIncomplete === true && statusMarksComplete) {
      recordSchemaWarning(
        SCHEMA_DIAGNOSTIC_CODES.CONFLICTING_STATUS,
        `${label}.status`,
        `${label} ("${pathCheck.normalized || raw.path}") has conflicting incomplete=true and status="${status}"`,
        { path: diagPath, expected: 'consistent incomplete/status flags', actual: status },
      );
    } else if (explicitIncomplete === false && statusMarksIncomplete) {
      recordSchemaWarning(
        SCHEMA_DIAGNOSTIC_CODES.CONFLICTING_STATUS,
        `${label}.status`,
        `${label} ("${pathCheck.normalized || raw.path}") has conflicting incomplete=false and status="${status}"`,
        { path: diagPath, expected: 'consistent incomplete/status flags', actual: status },
      );
    }

    if (!pathCheck.ok || !sizeValid) {
      continue;
    }

    const normalizedPath = pathCheck.normalized;
    if (fileMap.has(normalizedPath)) {
      summary.duplicateCount += 1;
      recordSchemaError(
        SCHEMA_DIAGNOSTIC_CODES.DUPLICATE_PATH,
        `${label}.path`,
        `${label}.path duplicates an earlier manifest entry: "${normalizedPath}"`,
        { path: normalizedPath, expected: 'unique path', actual: normalizedPath },
      );
      continue;
    }

    const lowerPath = normalizedPath.toLowerCase();
    if (lowerPathMap.has(lowerPath)) {
      recordSchemaWarning(
        SCHEMA_DIAGNOSTIC_CODES.CASE_COLLISION,
        `${label}.path`,
        `${label}.path ("${normalizedPath}") collides case-insensitively with "${lowerPathMap.get(lowerPath)}"`,
        { path: normalizedPath, expected: 'case-unique path', actual: normalizedPath },
      );
    } else {
      lowerPathMap.set(lowerPath, normalizedPath);
    }

    const suffixIncomplete = hasIncompleteSuffix(normalizedPath);
    let incompleteReason = null;
    if (typeof raw.reason === 'string' && raw.reason.trim().length > 0) {
      incompleteReason = raw.reason.trim();
    }

    const isIncomplete = Boolean(
      explicitIncomplete === true || statusMarksIncomplete || suffixIncomplete,
    );

    if (suffixIncomplete && explicitIncomplete !== true && !statusMarksIncomplete) {
      recordSchemaWarning(
        SCHEMA_DIAGNOSTIC_CODES.INCOMPLETE_SUFFIX,
        `${label}.path`,
        `${label} ("${normalizedPath}") has a temporary download suffix and is treated as incomplete`,
        { path: normalizedPath, expected: 'complete filename without temporary suffix', actual: normalizedPath },
      );
      if (!incompleteReason) {
        incompleteReason = 'temp_suffix';
      }
    } else if (isIncomplete && !incompleteReason) {
      incompleteReason = statusMarksIncomplete ? status : 'flagged_incomplete';
    }

    const normalizedEntry = {
      path: normalizedPath,
      size: raw.size,
      sha256,
      etag,
      incomplete: isIncomplete,
      status: status || (isIncomplete ? 'incomplete' : 'complete'),
      incompleteReason: isIncomplete ? incompleteReason : null,
    };
    if (crc32 !== undefined) normalizedEntry.crc32 = crc32;
    if (fnv1a32 !== undefined) normalizedEntry.fnv1a32 = fnv1a32;
    if (fnv1a64 !== undefined) normalizedEntry.fnv1a64 = fnv1a64;
    if (chunkSize !== undefined) normalizedEntry.chunkSize = chunkSize;
    if (chunkHashes !== undefined) normalizedEntry.chunkHashes = chunkHashes;

    fileMap.set(normalizedPath, normalizedEntry);
    summary.fileCount += 1;
    summary.totalBytes += raw.size;
    if (isIncomplete) {
      summary.incompleteFiles += 1;
      summary.incompleteBytes += raw.size;
    } else {
      summary.completeFiles += 1;
      summary.completeBytes += raw.size;
    }
    if (sha256 !== null) {
      summary.hashedFiles += 1;
    }
    if (etag !== null) {
      summary.etagFiles += 1;
    }
  }

  if (summary.declaredTotalBytes !== null && allSizesValid) {
    if (summary.declaredTotalBytes !== sumAllValidSizes) {
      mismatchesByCategory[MISMATCH_CATEGORIES.SIZE_MISMATCH] += 1;
      recordSchemaError(
        SCHEMA_DIAGNOSTIC_CODES.TOTAL_BYTES_MISMATCH,
        'total_bytes',
        `Manifest "total_bytes" (${summary.declaredTotalBytes}) does not match sum of file sizes (${sumAllValidSizes})`,
        { expected: sumAllValidSizes, actual: summary.declaredTotalBytes },
      );
    }
  }

  return finalize(Boolean(signal?.aborted));
}

function resolveManifestFileMap(manifest, options = {}) {
  if (manifest === null || manifest === undefined) {
    return null;
  }
  if (manifest instanceof Map) {
    return manifest;
  }
  if (isPlainObject(manifest) && manifest.fileMap instanceof Map) {
    return manifest.fileMap;
  }
  return validateDataManifest(manifest, options).fileMap;
}

/**
 * Validates a `bootset.json` prefetch range list (top-level array per `docs/io_manifest.md`
 * or `{ entries: [...] }` object wrapper supported by `io_worker.js`), optionally checking
 * bounds against a data manifest. Deduplicates exact duplicates, detects overlapping ranges,
 * and computes both raw and deduplicated/merged byte budgets.
 *
 * @param {unknown} bootsetInput
 * @param {unknown} [manifest=null]
 * @param {{ signal?: AbortSignal | null, nowFn?: (() => number) | null }} [options]
 * @returns {{
 *   ok: boolean,
 *   errors: string[],
 *   warnings: string[],
 *   entries: Array<{ path: string, offset: number, length: number, end: number }>,
 *   deduplicatedEntries: Array<{ path: string, offset: number, length: number, end: number }>,
 *   mergedRanges: Array<{ path: string, offset: number, length: number, end: number }>,
 *   duplicates: Array<{ path: string, offset: number, length: number, index: number }>,
 *   overlaps: Array<{
 *     path: string,
 *     first: { offset: number, length: number, end: number },
 *     second: { offset: number, length: number, end: number },
 *     overlapBytes: number,
 *   }>,
 *   missingFiles: string[],
 *   incompleteFiles: string[],
 *   totalBytes: number,
 *   uniqueBytes: number,
 *   overlapBytes: number,
 *   aborted: boolean,
 *   elapsedMs: number,
 *   bytesVerified: number,
 *   filesVerified: number,
 *   mismatchCount: number,
 *   mismatchesByCategory: { missing: number, size_mismatch: number, hash_mismatch: number },
 *   telemetry: {
 *     elapsedMs: number,
 *     bytesVerified: number,
 *     filesVerified: number,
 *     bytesPerSecond: number,
 *     mismatchCount: number,
 *     mismatchesByCategory: { missing: number, size_mismatch: number, hash_mismatch: number },
 *     aborted: boolean,
 *   },
 * }}
 */
export function validateBootset(bootsetInput, manifest = null, options = {}) {
  const { signal = null, nowFn = null } = isPlainObject(options) ? options : {};
  const now = resolveNowFn(nowFn);
  const startTime = now();

  const errors = [];
  const warnings = [];
  const schemaDiagnostics = [];
  const entries = [];
  const deduplicatedEntries = [];
  const mergedRanges = [];
  const duplicates = [];
  const overlaps = [];
  const missingSet = new Set();
  const incompleteSet = new Set();
  const rangesByPath = new Map();
  const mismatchesByCategory = createMismatchesByCategory();
  let totalBytes = 0;
  let uniqueBytes = 0;

  const recordBootError = (code, field, message, extra = {}) => {
    errors.push(message);
    schemaDiagnostics.push(
      createSchemaDiagnostic({
        severity: 'error',
        code,
        field,
        message,
        path: extra.path ?? null,
        expected: extra.expected ?? null,
        actual: extra.actual ?? null,
      }),
    );
  };

  const recordBootWarning = (code, field, message, extra = {}) => {
    warnings.push(message);
    schemaDiagnostics.push(
      createSchemaDiagnostic({
        severity: 'warning',
        code,
        field,
        message,
        path: extra.path ?? null,
        expected: extra.expected ?? null,
        actual: extra.actual ?? null,
      }),
    );
  };

  const finalize = (aborted = false) => {
    const missingFiles = Array.from(missingSet).sort(compareAscii);
    const incompleteFiles = Array.from(incompleteSet).sort(compareAscii);
    const overlapBytes = Math.max(0, totalBytes - uniqueBytes);
    const elapsedMs = computeElapsedMs(startTime, now());
    const telemetry = buildTelemetry({
      elapsedMs,
      bytesVerified: uniqueBytes,
      filesVerified: rangesByPath.size,
      mismatchesByCategory,
      aborted,
    });
    return {
      ok: !aborted && errors.length === 0,
      errors,
      warnings,
      schemaDiagnostics,
      entries,
      deduplicatedEntries,
      mergedRanges,
      duplicates,
      overlaps,
      missingFiles,
      incompleteFiles,
      totalBytes,
      uniqueBytes,
      overlapBytes,
      aborted: telemetry.aborted,
      elapsedMs: telemetry.elapsedMs,
      bytesVerified: telemetry.bytesVerified,
      filesVerified: telemetry.filesVerified,
      mismatchCount: telemetry.mismatchCount,
      mismatchesByCategory: { ...telemetry.mismatchesByCategory },
      telemetry,
    };
  };

  if (signal?.aborted) {
    return finalize(true);
  }

  let parsed = bootsetInput;
  if (typeof bootsetInput === 'string') {
    try {
      parsed = JSON.parse(bootsetInput);
    } catch (err) {
      recordBootError(
        SCHEMA_DIAGNOSTIC_CODES.JSON_PARSE_ERROR,
        '$',
        `Bootset JSON parse error: ${sanitizeDiagnosticText(err instanceof Error ? err.message : String(err))}`,
        { expected: 'valid JSON array or { entries } object', actual: 'malformed JSON' },
      );
      return finalize(false);
    }
  }

  let rawEntries = null;
  if (Array.isArray(parsed)) {
    rawEntries = parsed;
  } else if (isPlainObject(parsed) && Array.isArray(parsed.entries)) {
    rawEntries = parsed.entries;
  } else {
    recordBootError(
      SCHEMA_DIAGNOSTIC_CODES.INVALID_BOOTSET_ROOT,
      '$',
      'Bootset must be an array of range entries or an object with an "entries" array',
      { expected: 'array or { entries: [] }', actual: parsed === null ? 'null' : typeof parsed },
    );
    return finalize(false);
  }

  const fileMap = resolveManifestFileMap(manifest, { signal, nowFn });
  if (signal?.aborted) {
    return finalize(true);
  }
  const seenExact = new Set();

  for (let i = 0; i < rawEntries.length; i += 1) {
    if (signal?.aborted) {
      return finalize(true);
    }
    const raw = rawEntries[i];
    const label = `bootset[${i}]`;

    if (!isPlainObject(raw)) {
      recordBootError(
        SCHEMA_DIAGNOSTIC_CODES.INVALID_ENTRY,
        label,
        `${label} must be an object`,
        { expected: 'object', actual: raw === null ? 'null' : typeof raw },
      );
      continue;
    }

    const pathCheck = validateManifestPath(raw.path);
    const diagPath = pathCheck.ok ? pathCheck.normalized : null;
    if (!pathCheck.ok) {
      recordBootError(
        SCHEMA_DIAGNOSTIC_CODES.INVALID_PATH,
        `${label}.path`,
        `${label}.path: ${pathCheck.error}`,
        { expected: 'clean relative POSIX path', actual: raw.path },
      );
    }

    const offsetValid = isNonNegativeSafeInt(raw.offset);
    if (!offsetValid) {
      recordBootError(
        SCHEMA_DIAGNOSTIC_CODES.INVALID_OFFSET,
        `${label}.offset`,
        `${label}.offset must be a non-negative safe integer (got ${JSON.stringify(raw.offset)})`,
        { path: diagPath, expected: 'non-negative safe integer', actual: raw.offset },
      );
    }

    const lengthValid = isPositiveSafeInt(raw.length);
    if (!lengthValid) {
      recordBootError(
        SCHEMA_DIAGNOSTIC_CODES.INVALID_LENGTH,
        `${label}.length`,
        `${label}.length must be a positive safe integer (got ${JSON.stringify(raw.length)})`,
        { path: diagPath, expected: 'positive safe integer', actual: raw.length },
      );
    }

    if (!pathCheck.ok || !offsetValid || !lengthValid) {
      continue;
    }

    const end = raw.offset + raw.length;
    if (!Number.isSafeInteger(end)) {
      recordBootError(
        SCHEMA_DIAGNOSTIC_CODES.RANGE_OVERFLOW,
        label,
        `${label} range offset + length overflows safe integer bounds`,
        { path: diagPath, expected: `<= ${Number.MAX_SAFE_INTEGER}`, actual: end },
      );
      continue;
    }

    const relPath = pathCheck.normalized;

    if (fileMap !== null) {
      const manifestFile = fileMap.get(relPath);
      if (!manifestFile) {
        missingSet.add(relPath);
        mismatchesByCategory[MISMATCH_CATEGORIES.MISSING] += 1;
        recordBootError(
          SCHEMA_DIAGNOSTIC_CODES.MISSING_MANIFEST_FILE,
          `${label}.path`,
          `${label} references file missing from manifest: "${relPath}"`,
          { path: relPath, expected: 'file present in manifest', actual: relPath },
        );
        continue;
      }
      if (manifestFile.incomplete) {
        incompleteSet.add(relPath);
        recordBootWarning(
          SCHEMA_DIAGNOSTIC_CODES.INCOMPLETE_MANIFEST_FILE,
          `${label}.path`,
          `${label} references incomplete manifest file: "${relPath}"`,
          { path: relPath, expected: 'complete manifest file', actual: manifestFile.status },
        );
      }
      if (end > manifestFile.size) {
        mismatchesByCategory[MISMATCH_CATEGORIES.SIZE_MISMATCH] += 1;
        recordBootError(
          SCHEMA_DIAGNOSTIC_CODES.RANGE_OUT_OF_BOUNDS,
          label,
          `${label} range [${raw.offset}, ${end}) exceeds manifest file size ${manifestFile.size} for "${relPath}"`,
          { path: relPath, expected: manifestFile.size, actual: end },
        );
        continue;
      }
    }

    const entry = { path: relPath, offset: raw.offset, length: raw.length, end };
    entries.push(entry);
    totalBytes += raw.length;

    const exactKey = `${relPath}\0${raw.offset}\0${raw.length}`;
    if (seenExact.has(exactKey)) {
      duplicates.push({ path: relPath, offset: raw.offset, length: raw.length, index: i });
      recordBootWarning(
        SCHEMA_DIAGNOSTIC_CODES.DUPLICATE_RANGE,
        label,
        `${label} duplicates bootset range [${raw.offset}, ${end}) for "${relPath}"`,
        { path: relPath, expected: 'unique range', actual: `[${raw.offset}, ${end})` },
      );
    } else {
      seenExact.add(exactKey);
      deduplicatedEntries.push(entry);
    }

    let list = rangesByPath.get(relPath);
    if (!list) {
      list = [];
      rangesByPath.set(relPath, list);
    }
    list.push(entry);
  }

  // Detect overlapping ranges and merge intervals per path for exact unique byte budget.
  const sortedPaths = Array.from(rangesByPath.keys()).sort(compareAscii);

  for (const relPath of sortedPaths) {
    if (signal?.aborted) {
      return finalize(true);
    }
    const pathRanges = rangesByPath.get(relPath);
    pathRanges.sort((a, b) => a.offset - b.offset || a.end - b.end);

    let current = {
      path: relPath,
      offset: pathRanges[0].offset,
      end: pathRanges[0].end,
      length: pathRanges[0].length,
    };

    for (let i = 1; i < pathRanges.length; i += 1) {
      const next = pathRanges[i];
      if (next.offset < current.end) {
        const isExactDuplicate =
          next.offset === pathRanges[i - 1].offset && next.end === pathRanges[i - 1].end;
        const overlapBytes = Math.min(current.end, next.end) - next.offset;
        if (!isExactDuplicate) {
          overlaps.push({
            path: relPath,
            first: { offset: current.offset, length: current.end - current.offset, end: current.end },
            second: { offset: next.offset, length: next.length, end: next.end },
            overlapBytes,
          });
          recordBootWarning(
            SCHEMA_DIAGNOSTIC_CODES.OVERLAPPING_RANGE,
            'bootset',
            `Bootset ranges overlap by ${overlapBytes} bytes for "${relPath}" ([${current.offset}, ${current.end}) and [${next.offset}, ${next.end}))`,
            { path: relPath, expected: 'disjoint ranges', actual: overlapBytes },
          );
        }
        if (next.end > current.end) {
          current.end = next.end;
          current.length = current.end - current.offset;
        }
      } else {
        mergedRanges.push(current);
        uniqueBytes += current.length;
        current = {
          path: relPath,
          offset: next.offset,
          end: next.end,
          length: next.length,
        };
      }
    }

    mergedRanges.push(current);
    uniqueBytes += current.length;
  }

  return finalize(Boolean(signal?.aborted));
}

/**
 * Computes a structured streaming readiness summary from a manifest and optional bootset,
 * suitable for the browser HUD and pre-boot diagnostics.
 *
 * @param {unknown} manifest
 * @param {unknown} [bootset=null]
 * @param {{ signal?: AbortSignal | null, nowFn?: (() => number) | null }} [options]
 * @returns {{
 *   manifestValid: boolean,
 *   bootsetValid: boolean,
 *   totalFiles: number,
 *   completeFiles: number,
 *   incompleteFiles: number,
 *   totalBytes: number,
 *   readyBytes: number,
 *   incompleteBytes: number,
 *   readinessPct: number,
 *   bootsetReady: boolean,
 *   bootsetBytes: number,
 *   bootsetRawBytes: number,
 *   missingBootsetFiles: string[],
 *   incompleteBootsetFiles: string[],
 *   issues: Array<{ severity: 'error' | 'warning', source: 'manifest' | 'bootset', message: string }>,
 *   aborted: boolean,
 *   elapsedMs: number,
 *   bytesVerified: number,
 *   filesVerified: number,
 *   mismatchCount: number,
 *   mismatchesByCategory: { missing: number, size_mismatch: number, hash_mismatch: number },
 *   telemetry: {
 *     elapsedMs: number,
 *     bytesVerified: number,
 *     filesVerified: number,
 *     bytesPerSecond: number,
 *     mismatchCount: number,
 *     mismatchesByCategory: { missing: number, size_mismatch: number, hash_mismatch: number },
 *     aborted: boolean,
 *   },
 * }}
 */
export function computeStreamingReadiness(manifest, bootset = null, options = {}) {
  const { signal = null, nowFn = null } = isPlainObject(options) ? options : {};
  const now = resolveNowFn(nowFn);
  const startTime = now();
  const mismatchesByCategory = createMismatchesByCategory();

  if (signal?.aborted) {
    const elapsedMs = computeElapsedMs(startTime, now());
    const telemetry = buildTelemetry({
      elapsedMs,
      bytesVerified: 0,
      filesVerified: 0,
      mismatchesByCategory,
      aborted: true,
    });
    return {
      manifestValid: false,
      bootsetValid: false,
      totalFiles: 0,
      completeFiles: 0,
      incompleteFiles: 0,
      totalBytes: 0,
      readyBytes: 0,
      incompleteBytes: 0,
      readinessPct: 0,
      bootsetReady: false,
      bootsetBytes: 0,
      bootsetRawBytes: 0,
      missingBootsetFiles: [],
      incompleteBootsetFiles: [],
      issues: [],
      aborted: true,
      elapsedMs: telemetry.elapsedMs,
      bytesVerified: telemetry.bytesVerified,
      filesVerified: telemetry.filesVerified,
      mismatchCount: telemetry.mismatchCount,
      mismatchesByCategory: { ...telemetry.mismatchesByCategory },
      telemetry,
    };
  }

  const manifestResult =
    isPlainObject(manifest) &&
    manifest.fileMap instanceof Map &&
    isPlainObject(manifest.summary) &&
    Array.isArray(manifest.errors) &&
    Array.isArray(manifest.warnings)
      ? manifest
      : validateDataManifest(manifest, { signal, nowFn });

  if (isPlainObject(manifestResult.mismatchesByCategory)) {
    mismatchesByCategory[MISMATCH_CATEGORIES.MISSING] +=
      manifestResult.mismatchesByCategory[MISMATCH_CATEGORIES.MISSING] || 0;
    mismatchesByCategory[MISMATCH_CATEGORIES.SIZE_MISMATCH] +=
      manifestResult.mismatchesByCategory[MISMATCH_CATEGORIES.SIZE_MISMATCH] || 0;
    mismatchesByCategory[MISMATCH_CATEGORIES.HASH_MISMATCH] +=
      manifestResult.mismatchesByCategory[MISMATCH_CATEGORIES.HASH_MISMATCH] || 0;
  }

  const issues = [];
  for (const msg of manifestResult.errors) {
    issues.push({ severity: 'error', source: 'manifest', message: msg });
  }
  for (const msg of manifestResult.warnings) {
    issues.push({ severity: 'warning', source: 'manifest', message: msg });
  }

  const totalFiles = manifestResult.summary.fileCount;
  const completeFiles = manifestResult.summary.completeFiles;
  const incompleteFiles = manifestResult.summary.incompleteFiles;
  const totalBytes = manifestResult.summary.totalBytes;
  const readyBytes = manifestResult.summary.completeBytes;
  const incompleteBytes = manifestResult.summary.incompleteBytes;

  let readinessPct = 0;
  if (totalBytes > 0) {
    readinessPct = Math.round((readyBytes / totalBytes) * 10000) / 100;
  } else if (totalFiles > 0) {
    readinessPct = Math.round((completeFiles / totalFiles) * 10000) / 100;
  }

  let aborted = Boolean(signal?.aborted || manifestResult.aborted);
  let bootsetValid = !aborted;
  let bootsetReady = !aborted && manifestResult.ok && completeFiles > 0;
  let bootsetBytes = 0;
  let bootsetRawBytes = 0;
  let missingBootsetFiles = [];
  let incompleteBootsetFiles = [];

  if (!aborted && bootset !== null && bootset !== undefined) {
    const bootsetResult = validateBootset(bootset, manifestResult, { signal, nowFn });
    aborted = Boolean(signal?.aborted || bootsetResult.aborted);
    bootsetValid = bootsetResult.ok;
    bootsetBytes = bootsetResult.uniqueBytes;
    bootsetRawBytes = bootsetResult.totalBytes;
    missingBootsetFiles = bootsetResult.missingFiles;
    incompleteBootsetFiles = bootsetResult.incompleteFiles;

    if (isPlainObject(bootsetResult.mismatchesByCategory)) {
      mismatchesByCategory[MISMATCH_CATEGORIES.MISSING] +=
        bootsetResult.mismatchesByCategory[MISMATCH_CATEGORIES.MISSING] || 0;
      mismatchesByCategory[MISMATCH_CATEGORIES.SIZE_MISMATCH] +=
        bootsetResult.mismatchesByCategory[MISMATCH_CATEGORIES.SIZE_MISMATCH] || 0;
      mismatchesByCategory[MISMATCH_CATEGORIES.HASH_MISMATCH] +=
        bootsetResult.mismatchesByCategory[MISMATCH_CATEGORIES.HASH_MISMATCH] || 0;
    }

    for (const msg of bootsetResult.errors) {
      issues.push({ severity: 'error', source: 'bootset', message: msg });
    }
    for (const msg of bootsetResult.warnings) {
      issues.push({ severity: 'warning', source: 'bootset', message: msg });
    }

    bootsetReady =
      !aborted &&
      manifestResult.ok &&
      bootsetResult.ok &&
      missingBootsetFiles.length === 0 &&
      incompleteBootsetFiles.length === 0;
  }

  const elapsedMs = computeElapsedMs(startTime, now());
  const telemetry = buildTelemetry({
    elapsedMs,
    bytesVerified: readyBytes,
    filesVerified: completeFiles,
    mismatchesByCategory,
    aborted,
  });

  return {
    manifestValid: !aborted && manifestResult.ok,
    bootsetValid: !aborted && bootsetValid,
    totalFiles,
    completeFiles,
    incompleteFiles,
    totalBytes,
    readyBytes,
    incompleteBytes,
    readinessPct,
    bootsetReady: !aborted && bootsetReady,
    bootsetBytes,
    bootsetRawBytes,
    missingBootsetFiles,
    incompleteBootsetFiles,
    issues,
    aborted: telemetry.aborted,
    elapsedMs: telemetry.elapsedMs,
    bytesVerified: telemetry.bytesVerified,
    filesVerified: telemetry.filesVerified,
    mismatchCount: telemetry.mismatchCount,
    mismatchesByCategory: { ...telemetry.mismatchesByCategory },
    telemetry,
  };
}

async function cancelResponseBodyQuietly(res) {
  if (res?.body && typeof res.body.cancel === 'function') {
    try {
      await res.body.cancel();
    } catch {
      // Ignore stream cancellation errors.
    }
  }
}

async function consumeResponseStream(res, options = {}) {
  if (!res) {
    return { byteLength: null, bytes: null, streamed: false };
  }
  const { signal = null, needBytes = true } = options;

  if (res.body && typeof res.body.getReader === 'function') {
    const reader = res.body.getReader();
    const chunks = needBytes ? [] : null;
    let totalBytes = 0;
    let shouldCancel = false;
    try {
      while (true) {
        if (signal?.aborted) {
          shouldCancel = true;
          throw createAbortError(signal);
        }
        let step;
        try {
          step = await raceWithAbort(reader.read(), signal);
        } catch (err) {
          if (signal?.aborted || isAbortError(err)) {
            shouldCancel = true;
          }
          throw err;
        }
        if (!step || step.done) {
          break;
        }
        if (step.value !== undefined && step.value !== null) {
          const view = toUint8View(step.value);
          totalBytes += view.byteLength;
          if (chunks !== null && view.byteLength > 0) {
            chunks.push(view);
          }
        }
      }
    } finally {
      if (shouldCancel) {
        try {
          await reader.cancel();
        } catch {
          // Ignore stream cancel errors.
        }
      }
      try {
        reader.releaseLock();
      } catch {
        // Ignore lock release errors.
      }
    }

    if (!needBytes) {
      return { byteLength: totalBytes, bytes: null, streamed: true };
    }
    if (chunks.length === 0) {
      return { byteLength: 0, bytes: new Uint8Array(0), streamed: true };
    }
    if (chunks.length === 1) {
      return { byteLength: totalBytes, bytes: chunks[0], streamed: true };
    }
    const merged = new Uint8Array(totalBytes);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return { byteLength: totalBytes, bytes: merged, streamed: true };
  }

  if (typeof res.arrayBuffer === 'function') {
    const buf = await raceWithAbort(res.arrayBuffer(), signal);
    if (!buf) {
      return { byteLength: 0, bytes: needBytes ? new Uint8Array(0) : null, streamed: false };
    }
    const bytes = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
    return { byteLength: bytes.byteLength, bytes: needBytes ? bytes : null, streamed: false };
  }

  await cancelResponseBodyQuietly(res);
  return { byteLength: null, bytes: null, streamed: false };
}

function hasCoopCoepHeaders(headers) {
  const coop = getHeader(headers, 'cross-origin-opener-policy');
  const coep = getHeader(headers, 'cross-origin-embedder-policy');
  return (
    typeof coop === 'string' &&
    coop.trim().toLowerCase() === 'same-origin' &&
    typeof coep === 'string' &&
    coep.trim().toLowerCase() === 'require-corp'
  );
}

/**
 * Probes up to `sampleCount` complete manifest files against a live HTTP endpoint using
 * `HEAD` and `GET Range: bytes=0-<end>` requests with bounded concurrency (`maxConcurrency`,
 * default `4`, clamped `[1, 16]`). Verifies `206 Partial Content`, `Accept-Ranges: bytes`,
 * `Content-Range` total length matching the manifest size, response byte count, `ETag`
 * presence/consistency, and optional streaming SHA-256 content hashes.
 *
 * Supports optional `AbortSignal` cancellation (`signal`) propagated across all active workers
 * and injectable clock (`nowFn`), returning sanitized additive throughput/timing telemetry and
 * structured mismatch categories without leaking host paths or raw file bytes.
 *
 * @param {object} options
 * @param {typeof fetch} [options.fetchImpl] Injectable fetch implementation (defaults to globalThis.fetch).
 * @param {string} [options.dataBase='/data/'] Base URL serving manifest files.
 * @param {unknown} options.manifest Raw manifest object, JSON string, or validateDataManifest() result.
 * @param {number} [options.sampleCount=5] Maximum number of complete files to probe.
 * @param {number} [options.probeBytes=1] Number of prefix bytes to request in the Range probe (1 -> bytes=0-0).
 * @param {number} [options.maxConcurrency=4] Maximum concurrent verification workers, clamped to `[1, 16]`.
 * @param {boolean} [options.checkCoopCoep=false] When true, fail probes missing COOP/COEP headers.
 * @param {boolean} [options.verifyHashes=false] When true, verify SHA-256 digests for entries declaring `sha256`.
 * @param {((bytes: Uint8Array) => Promise<string> | string) | null} [options.hashFn=null] Optional SHA-256 hex digest provider.
 * @param {Record<string, { etag: string, size: number, sha256?: string | null }> | Map<string, { etag: string, size: number, sha256?: string | null }> | object | null} [options.verifiedFingerprints=null] Optional verified fingerprint cache keyed by sanitized entry path for warm-boot ETag+size skip.
 * @param {Record<string, { etag: string, size: number, sha256?: string | null }> | Map<string, { etag: string, size: number, sha256?: string | null }> | object | null} [options.previousSnapshot=null] Optional previous `verifyLiveEndpoints()` report or fingerprint snapshot map for warm-boot ETag+size skip.
 * @param {boolean} [options.forceReverify=false] When true, bypass any `verifiedFingerprints`/`previousSnapshot` skip cache and perform full streaming verification.
 * @param {AbortSignal | null} [options.signal=null] Optional AbortSignal to cancel in-flight verification.
 * @param {(() => number) | null} [options.nowFn=null] Optional injectable monotonic clock in milliseconds.
 * @returns {Promise<{
 *   ok: boolean,
 *   probed: number,
 *   passed: number,
 *   failed: number,
 *   skippedIncomplete: number,
 *   errors: string[],
 *   results: Array<object>,
 *   aborted: boolean,
 *   abortedCount: number,
 *   elapsedMs: number,
 *   bytesVerified: number,
 *   bytesHashed: number,
 *   filesVerified: number,
 *   peakConcurrentHashes: number,
 *   maxConcurrency: number,
 *   fingerprintSkippedCount: number,
 *   bytesSkippedByFingerprint: number,
 *   verifiedFingerprints: Record<string, { etag: string, size: number, sha256: string | null }>,
 *   mismatchCount: number,
 *   mismatchesByCategory: { missing: number, size_mismatch: number, hash_mismatch: number },
 *   mismatches: Array<{ path: string, category: 'missing' | 'size_mismatch' | 'hash_mismatch', detail: string }>,
 *   telemetry: {
 *     elapsedMs: number,
 *     bytesVerified: number,
 *     bytesHashed: number,
 *     filesVerified: number,
 *     bytesPerSecond: number,
 *     mismatchCount: number,
 *     mismatchesByCategory: { missing: number, size_mismatch: number, hash_mismatch: number },
 *     aborted: boolean,
 *     abortedCount: number,
 *     peakConcurrentHashes: number,
 *     maxConcurrency: number,
 *     fingerprintSkippedCount: number,
 *     bytesSkippedByFingerprint: number,
 *   },
 * }>}
 */
export async function verifyLiveEndpoints(options = {}) {
  const {
    fetchImpl = typeof fetch === 'function' ? fetch.bind(globalThis) : null,
    dataBase = '/data/',
    manifest,
    sampleCount = DEFAULT_SAMPLE_COUNT,
    probeBytes = DEFAULT_RANGE_PROBE_BYTES,
    maxConcurrency = DEFAULT_MAX_CONCURRENCY,
    checkCoopCoep = false,
    verifyHashes = false,
    hashFn = null,
    verifiedFingerprints = null,
    previousSnapshot = null,
    forceReverify = false,
    signal = null,
    nowFn = null,
  } = options || {};

  if (typeof fetchImpl !== 'function') {
    throw new TypeError('verifyLiveEndpoints requires a fetchImpl function');
  }

  const concurrency = normalizeMaxConcurrency(maxConcurrency, DEFAULT_MAX_CONCURRENCY);
  const fingerprintCache = buildFingerprintCacheMap(previousSnapshot, verifiedFingerprints);
  const skipCacheEnabled = !forceReverify && fingerprintCache.size > 0;
  const now = resolveNowFn(nowFn);
  const startTime = now();

  const mismatchesByCategory = createMismatchesByCategory();
  const mismatches = [];
  const results = [];
  const verifiedFingerprintsOut = {};
  let passed = 0;
  let failed = 0;
  let skippedIncomplete = 0;
  let bytesVerified = 0;
  let filesVerified = 0;
  let aborted = false;
  let runBytesHashed = 0;
  let runAbortedCount = 0;
  let runActiveHashes = 0;
  let runPeakConcurrentHashes = 0;
  let runFingerprintSkippedCount = 0;
  let runBytesSkippedByFingerprint = 0;

  const enterRunHash = () => {
    runActiveHashes += 1;
    if (runActiveHashes > runPeakConcurrentHashes) {
      runPeakConcurrentHashes = runActiveHashes;
    }
    enterModuleHash();
  };

  const leaveRunHash = () => {
    if (runActiveHashes > 0) {
      runActiveHashes -= 1;
    }
    leaveModuleHash();
  };

  const finalize = (manifestOk, topErrors) => {
    const elapsedMs = computeElapsedMs(startTime, now());
    const baseTelemetry = buildTelemetry({
      elapsedMs,
      bytesVerified,
      filesVerified,
      mismatchesByCategory,
      aborted,
    });
    const telemetry = {
      ...baseTelemetry,
      bytesHashed: runBytesHashed,
      abortedCount: runAbortedCount,
      peakConcurrentHashes: runPeakConcurrentHashes,
      maxConcurrency: concurrency,
      fingerprintSkippedCount: runFingerprintSkippedCount,
      bytesSkippedByFingerprint: runBytesSkippedByFingerprint,
    };
    return {
      ok: !aborted && manifestOk && failed === 0,
      probed: results.length,
      passed,
      failed,
      skippedIncomplete,
      errors: topErrors,
      results,
      aborted: telemetry.aborted,
      abortedCount: runAbortedCount,
      elapsedMs: telemetry.elapsedMs,
      bytesVerified: telemetry.bytesVerified,
      bytesHashed: runBytesHashed,
      filesVerified: telemetry.filesVerified,
      peakConcurrentHashes: runPeakConcurrentHashes,
      maxConcurrency: concurrency,
      fingerprintSkippedCount: runFingerprintSkippedCount,
      bytesSkippedByFingerprint: runBytesSkippedByFingerprint,
      verifiedFingerprints: verifiedFingerprintsOut,
      mismatchCount: telemetry.mismatchCount,
      mismatchesByCategory: { ...telemetry.mismatchesByCategory },
      mismatches,
      telemetry,
    };
  };

  if (signal?.aborted) {
    aborted = true;
    runAbortedCount = 1;
    moduleStats.abortedCount += 1;
    return finalize(false, []);
  }

  const manifestResult =
    isPlainObject(manifest) &&
    manifest.fileMap instanceof Map &&
    isPlainObject(manifest.summary) &&
    Array.isArray(manifest.errors)
      ? manifest
      : validateDataManifest(manifest, { signal, nowFn });

  const topErrors = [...manifestResult.errors];

  if (signal?.aborted || manifestResult.aborted) {
    aborted = true;
    runAbortedCount = 1;
    moduleStats.abortedCount += 1;
    return finalize(manifestResult.ok, topErrors);
  }

  const base = normalizeDataBase(dataBase);
  const maxSamples =
    typeof sampleCount === 'number' && Number.isFinite(sampleCount)
      ? Math.max(0, Math.floor(sampleCount))
      : DEFAULT_SAMPLE_COUNT;
  const requestedProbeBytes =
    typeof probeBytes === 'number' && Number.isFinite(probeBytes)
      ? Math.max(1, Math.floor(probeBytes))
      : DEFAULT_RANGE_PROBE_BYTES;

  const completeEntries = [];
  for (const entry of manifestResult.fileMap.values()) {
    if (entry.incomplete) {
      skippedIncomplete += 1;
    } else {
      completeEntries.push(entry);
    }
  }

  const selected = completeEntries.slice(0, maxSamples);

  const verifySingleEntry = async (entry) => {
    const fileStart = now();
    const safePath = validateManifestPath(entry.path).ok ? entry.path : '<invalid-path>';
    const url = `${base}${encodeRelPathForUrl(entry.path)}`;
    const fileErrors = [];
    const fileCategories = new Set();
    const fileMismatches = [];

    const recordFileMismatch = (category, detail) => {
      if (fileCategories.has(category)) {
        return;
      }
      fileCategories.add(category);
      fileMismatches.push({
        path: safePath,
        category,
        detail: sanitizeDiagnosticText(detail),
      });
    };

    const headInfo = {
      status: null,
      contentLength: null,
      acceptRanges: null,
      etag: null,
    };
    const rangeInfo = {
      requested: null,
      skipped: entry.size === 0,
      status: null,
      contentRange: null,
      parsedContentRange: null,
      contentLength: null,
      receivedBytes: null,
      acceptRanges: null,
      etag: null,
    };
    let hashInfo = null;

    let headCoopCoep = false;
    let rangeCoopCoep = false;
    let fileBytesVerified = 0;
    let fileBytesHashed = 0;
    let rangeBytes = null;
    let rangeStreamedHash = null;
    let rangeEndByte = -1;
    let entryAborted = false;

    // 1. HEAD probe
    try {
      const headInit = signal ? { method: 'HEAD', signal } : { method: 'HEAD' };
      const headRes = await raceWithAbort(fetchImpl(url, headInit), signal);
      if (signal?.aborted) {
        await cancelResponseBodyQuietly(headRes);
        entryAborted = true;
      } else {
        headInfo.status = headRes ? headRes.status : null;
        headInfo.acceptRanges = getHeader(headRes?.headers, 'accept-ranges');
        headInfo.etag = getHeader(headRes?.headers, 'etag');
        const headLenRaw = getHeader(headRes?.headers, 'content-length');
        headInfo.contentLength =
          headLenRaw !== null && /^\d+$/.test(headLenRaw.trim()) ? Number(headLenRaw.trim()) : null;
        headCoopCoep = hasCoopCoepHeaders(headRes?.headers);

        await cancelResponseBodyQuietly(headRes);

        if (headInfo.status !== 200) {
          const msg = `HEAD ${safePath} returned HTTP ${headInfo.status} (expected 200)`;
          fileErrors.push(msg);
          if (headInfo.status === 404 || headInfo.status === 410) {
            recordFileMismatch(MISMATCH_CATEGORIES.MISSING, msg);
          }
        } else {
          if (!headInfo.acceptRanges || !/\bbytes\b/i.test(headInfo.acceptRanges)) {
            fileErrors.push(
              `HEAD ${safePath} missing Accept-Ranges: bytes (got ${JSON.stringify(headInfo.acceptRanges)})`,
            );
          }
          if (headInfo.contentLength !== null && headInfo.contentLength !== entry.size) {
            const msg = `HEAD ${safePath} Content-Length (${headInfo.contentLength}) does not match manifest size (${entry.size})`;
            fileErrors.push(msg);
            recordFileMismatch(MISMATCH_CATEGORIES.SIZE_MISMATCH, msg);
          }
          if (!headInfo.etag || headInfo.etag.trim().length === 0) {
            fileErrors.push(`HEAD ${safePath} missing ETag header`);
          } else if (entry.etag && headInfo.etag.trim() !== entry.etag) {
            const msg = `HEAD ${safePath} ETag (${sanitizeDiagnosticText(headInfo.etag.trim(), 96)}) does not match manifest ETag (${sanitizeDiagnosticText(entry.etag, 96)})`;
            fileErrors.push(msg);
            recordFileMismatch(MISMATCH_CATEGORIES.HASH_MISMATCH, msg);
          }
          if (checkCoopCoep && !headCoopCoep) {
            fileErrors.push(`HEAD ${safePath} missing required COOP/COEP headers`);
          }
        }
      }
    } catch (err) {
      if (signal?.aborted || isAbortError(err)) {
        entryAborted = true;
      } else {
        const safeErr = sanitizeDiagnosticText(err instanceof Error ? err.message : String(err));
        fileErrors.push(`HEAD ${safePath} request failed: ${safeErr}`);
      }
    }

    if (entryAborted || signal?.aborted) {
      return {
        aborted: true,
        bytesHashed: fileBytesHashed,
        fingerprintSkipped: false,
        bytesSkippedByFingerprint: 0,
      };
    }

    // 1b. Warm-boot ETag + size fingerprint skip fast path:
    //     When HEAD returns HTTP 200 with a strong ETag and Content-Length matching both
    //     the manifest entry and the verified snapshot (and matching manifest sha256 when
    //     declared), short-circuit GET body fetches and full streaming SHA-256 hashing.
    let fingerprintSkipped = false;
    let fileBytesSkippedByFingerprint = 0;
    const headStrongEtag = normalizeStrongEtag(headInfo.etag);
    const cachedFingerprint =
      skipCacheEnabled && safePath !== '<invalid-path>' ? fingerprintCache.get(safePath) : null;

    if (
      cachedFingerprint &&
      headInfo.status === 200 &&
      fileErrors.length === 0 &&
      headStrongEtag !== null &&
      headInfo.contentLength !== null &&
      headInfo.contentLength === entry.size &&
      cachedFingerprint.etag === headStrongEtag &&
      cachedFingerprint.size === entry.size &&
      (entry.sha256 === null || cachedFingerprint.sha256 === entry.sha256)
    ) {
      fingerprintSkipped = true;
      fileBytesSkippedByFingerprint = entry.size;
      rangeInfo.skipped = true;
      if (verifyHashes && entry.sha256 !== null) {
        hashInfo = {
          expected: entry.sha256,
          actual: cachedFingerprint.sha256,
          verified: true,
          checkedBytes: 0,
          skippedByFingerprint: true,
          elapsedMs: 0,
        };
      }
    }

    // 2. Range GET probe (only when file has at least 1 byte and not skipped via fingerprint cache)
    if (entry.size > 0 && !fingerprintSkipped) {
      const endByte = Math.min(entry.size - 1, requestedProbeBytes - 1);
      const expectedLength = endByte + 1;
      const rangeHeader = `bytes=0-${endByte}`;
      rangeInfo.requested = rangeHeader;
      rangeEndByte = endByte;
      const canReuseRangeForHash =
        verifyHashes &&
        entry.sha256 !== null &&
        !fileCategories.has(MISMATCH_CATEGORIES.MISSING) &&
        endByte === entry.size - 1;

      try {
        const rangeInit = signal
          ? { method: 'GET', headers: { Range: rangeHeader }, signal }
          : { method: 'GET', headers: { Range: rangeHeader } };
        const rangeRes = await raceWithAbort(fetchImpl(url, rangeInit), signal);
        if (signal?.aborted) {
          await cancelResponseBodyQuietly(rangeRes);
          entryAborted = true;
        } else {
          rangeInfo.status = rangeRes ? rangeRes.status : null;
          rangeInfo.acceptRanges = getHeader(rangeRes?.headers, 'accept-ranges');
          rangeInfo.contentRange = getHeader(rangeRes?.headers, 'content-range');
          rangeInfo.etag = getHeader(rangeRes?.headers, 'etag');
          const rangeLenRaw = getHeader(rangeRes?.headers, 'content-length');
          rangeInfo.contentLength =
            rangeLenRaw !== null && /^\d+$/.test(rangeLenRaw.trim())
              ? Number(rangeLenRaw.trim())
              : null;
          rangeCoopCoep = hasCoopCoepHeaders(rangeRes?.headers);

          if (rangeInfo.status !== 206) {
            await cancelResponseBodyQuietly(rangeRes);
            const msg = `GET ${safePath} (${rangeHeader}) returned HTTP ${rangeInfo.status} (expected 206 Partial Content)`;
            fileErrors.push(msg);
            if (rangeInfo.status === 404 || rangeInfo.status === 410) {
              recordFileMismatch(MISMATCH_CATEGORIES.MISSING, msg);
            } else if (rangeInfo.status === 416) {
              recordFileMismatch(MISMATCH_CATEGORIES.SIZE_MISMATCH, msg);
            }
          } else if (
            canReuseRangeForHash &&
            typeof hashFn !== 'function' &&
            rangeRes?.body &&
            typeof rangeRes.body.getReader === 'function'
          ) {
            enterRunHash();
            try {
              rangeStreamedHash = await verifyStream(rangeRes.body, {
                algorithm: STREAM_HASH_ALGORITHMS.SHA256,
                expectedDigest: entry.sha256,
                expectedSize: entry.size,
                path: safePath,
                signal,
                nowFn,
                [INTERNAL_SKIP_STATS]: true,
              });
            } finally {
              leaveRunHash();
            }
            fileBytesHashed += rangeStreamedHash.checkedBytes;
            if (rangeStreamedHash.aborted || signal?.aborted) {
              entryAborted = true;
            } else {
              rangeInfo.receivedBytes = rangeStreamedHash.checkedBytes;
            }
          } else {
            const consumed = await consumeResponseStream(rangeRes, {
              signal,
              needBytes: canReuseRangeForHash,
            });
            if (signal?.aborted) {
              entryAborted = true;
            } else {
              rangeInfo.receivedBytes = consumed.byteLength;
              rangeBytes = consumed.bytes;
            }
          }

          if (!entryAborted && !signal?.aborted && rangeInfo.status === 206) {
            if (rangeInfo.receivedBytes !== null && rangeInfo.receivedBytes > 0) {
              fileBytesVerified += rangeInfo.receivedBytes;
            }
            if (!rangeInfo.acceptRanges || !/\bbytes\b/i.test(rangeInfo.acceptRanges)) {
              fileErrors.push(
                `GET ${safePath} (${rangeHeader}) missing Accept-Ranges: bytes (got ${JSON.stringify(rangeInfo.acceptRanges)})`,
              );
            }

            const parsedCr = parseContentRange(rangeInfo.contentRange);
            rangeInfo.parsedContentRange = parsedCr;
            if (!parsedCr) {
              fileErrors.push(
                `GET ${safePath} (${rangeHeader}) returned missing or invalid Content-Range: ${JSON.stringify(rangeInfo.contentRange)}`,
              );
            } else {
              if (parsedCr.start !== 0 || parsedCr.end !== endByte) {
                fileErrors.push(
                  `GET ${safePath} (${rangeHeader}) Content-Range span ${parsedCr.start}-${parsedCr.end} does not match requested 0-${endByte}`,
                );
              }
              if (parsedCr.total !== entry.size) {
                const msg = `GET ${safePath} (${rangeHeader}) Content-Range total (${parsedCr.total}) does not match manifest size (${entry.size})`;
                fileErrors.push(msg);
                recordFileMismatch(MISMATCH_CATEGORIES.SIZE_MISMATCH, msg);
              }
            }

            if (rangeInfo.contentLength !== null && rangeInfo.contentLength !== expectedLength) {
              const msg = `GET ${safePath} (${rangeHeader}) Content-Length (${rangeInfo.contentLength}) does not match expected ${expectedLength}`;
              fileErrors.push(msg);
              recordFileMismatch(MISMATCH_CATEGORIES.SIZE_MISMATCH, msg);
            }

            if (rangeInfo.receivedBytes !== null && rangeInfo.receivedBytes !== expectedLength) {
              const msg = `GET ${safePath} (${rangeHeader}) returned ${rangeInfo.receivedBytes} bytes (expected ${expectedLength})`;
              fileErrors.push(msg);
              recordFileMismatch(MISMATCH_CATEGORIES.SIZE_MISMATCH, msg);
            }

            if (!rangeInfo.etag || rangeInfo.etag.trim().length === 0) {
              fileErrors.push(`GET ${safePath} (${rangeHeader}) missing ETag header`);
            } else {
              if (headInfo.etag && rangeInfo.etag.trim() !== headInfo.etag.trim()) {
                const msg = `GET ${safePath} ETag (${sanitizeDiagnosticText(rangeInfo.etag.trim(), 96)}) differs from HEAD ETag (${sanitizeDiagnosticText(headInfo.etag.trim(), 96)})`;
                fileErrors.push(msg);
                recordFileMismatch(MISMATCH_CATEGORIES.HASH_MISMATCH, msg);
              }
              if (entry.etag && rangeInfo.etag.trim() !== entry.etag) {
                const msg = `GET ${safePath} ETag (${sanitizeDiagnosticText(rangeInfo.etag.trim(), 96)}) does not match manifest ETag (${sanitizeDiagnosticText(entry.etag, 96)})`;
                fileErrors.push(msg);
                recordFileMismatch(MISMATCH_CATEGORIES.HASH_MISMATCH, msg);
              }
            }

            if (checkCoopCoep && !rangeCoopCoep) {
              fileErrors.push(
                `GET ${safePath} (${rangeHeader}) missing required COOP/COEP headers`,
              );
            }
          }
        }
      } catch (err) {
        if (signal?.aborted || isAbortError(err)) {
          entryAborted = true;
        } else {
          const safeErr = sanitizeDiagnosticText(err instanceof Error ? err.message : String(err));
          fileErrors.push(`GET ${safePath} (${rangeHeader}) request failed: ${safeErr}`);
        }
      }
    }

    if (entryAborted || signal?.aborted) {
      rangeBytes = null;
      return {
        aborted: true,
        bytesHashed: fileBytesHashed,
        fingerprintSkipped: false,
        bytesSkippedByFingerprint: 0,
      };
    }

    // 3. Optional SHA-256 content hash verification (streams response body in O(1) RAM when
    //    available so multi-hundred-MB .pgim / .wgsl.pack files avoid duplicate full buffers).
    if (
      !fingerprintSkipped &&
      verifyHashes &&
      entry.sha256 !== null &&
      !fileCategories.has(MISMATCH_CATEGORIES.MISSING)
    ) {
      const hashStart = now();
      let bytesToHash = null;
      let streamedResult = rangeStreamedHash;
      try {
        if (streamedResult !== null) {
          fileBytesVerified = Math.max(fileBytesVerified, streamedResult.checkedBytes);
          const verified = Boolean(streamedResult.digestVerified);
          hashInfo = {
            expected: entry.sha256,
            actual: streamedResult.digest,
            verified,
            checkedBytes: streamedResult.checkedBytes,
            elapsedMs: streamedResult.elapsedMs,
          };
          if (!verified) {
            const msg = `SHA-256 ${safePath} digest (${streamedResult.digest || 'invalid'}) does not match manifest sha256 (${entry.sha256})`;
            fileErrors.push(msg);
            recordFileMismatch(MISMATCH_CATEGORIES.HASH_MISMATCH, msg);
          }
        } else if (entry.size === 0) {
          bytesToHash = new Uint8Array(0);
        } else if (
          rangeInfo.status === 206 &&
          rangeBytes !== null &&
          rangeEndByte === entry.size - 1 &&
          rangeBytes.byteLength === entry.size
        ) {
          bytesToHash = rangeBytes;
        } else {
          const fullInit = signal ? { method: 'GET', signal } : { method: 'GET' };
          const fullRes = await raceWithAbort(fetchImpl(url, fullInit), signal);
          if (signal?.aborted) {
            await cancelResponseBodyQuietly(fullRes);
            entryAborted = true;
          } else {
            const fullStatus = fullRes ? fullRes.status : null;
            if (fullStatus === 404 || fullStatus === 410) {
              await cancelResponseBodyQuietly(fullRes);
              const msg = `GET ${safePath} (sha256) returned HTTP ${fullStatus} (expected 200)`;
              fileErrors.push(msg);
              recordFileMismatch(MISMATCH_CATEGORIES.MISSING, msg);
            } else if (fullStatus !== 200) {
              await cancelResponseBodyQuietly(fullRes);
              fileErrors.push(
                `GET ${safePath} (sha256) returned HTTP ${fullStatus} (expected 200)`,
              );
            } else if (
              typeof hashFn !== 'function' &&
              fullRes?.body &&
              typeof fullRes.body.getReader === 'function'
            ) {
              enterRunHash();
              try {
                streamedResult = await verifyStream(fullRes.body, {
                  algorithm: STREAM_HASH_ALGORITHMS.SHA256,
                  expectedDigest: entry.sha256,
                  expectedSize: entry.size,
                  path: safePath,
                  signal,
                  nowFn,
                  [INTERNAL_SKIP_STATS]: true,
                });
              } finally {
                leaveRunHash();
              }
              fileBytesHashed += streamedResult.checkedBytes;
              if (streamedResult.aborted || signal?.aborted) {
                entryAborted = true;
              } else {
                fileBytesVerified = Math.max(fileBytesVerified, streamedResult.checkedBytes);
                if (streamedResult.checkedBytes !== entry.size) {
                  const msg = `GET ${safePath} (sha256) returned ${streamedResult.checkedBytes} bytes (expected ${entry.size})`;
                  fileErrors.push(msg);
                  recordFileMismatch(MISMATCH_CATEGORIES.SIZE_MISMATCH, msg);
                }
                const verified = Boolean(streamedResult.digestVerified);
                hashInfo = {
                  expected: entry.sha256,
                  actual: streamedResult.digest,
                  verified,
                  checkedBytes: streamedResult.checkedBytes,
                  elapsedMs: streamedResult.elapsedMs,
                };
                if (!verified) {
                  const msg = `SHA-256 ${safePath} digest (${streamedResult.digest || 'invalid'}) does not match manifest sha256 (${entry.sha256})`;
                  fileErrors.push(msg);
                  recordFileMismatch(MISMATCH_CATEGORIES.HASH_MISMATCH, msg);
                }
              }
            } else {
              const fullConsumed = await consumeResponseStream(fullRes, {
                signal,
                needBytes: true,
              });
              if (signal?.aborted) {
                entryAborted = true;
              } else if (fullConsumed.bytes !== null) {
                fileBytesVerified = Math.max(fileBytesVerified, fullConsumed.byteLength);
                if (fullConsumed.byteLength !== entry.size) {
                  const msg = `GET ${safePath} (sha256) returned ${fullConsumed.byteLength} bytes (expected ${entry.size})`;
                  fileErrors.push(msg);
                  recordFileMismatch(MISMATCH_CATEGORIES.SIZE_MISMATCH, msg);
                }
                bytesToHash = fullConsumed.bytes;
              }
            }
          }
        }

        if (!entryAborted && !signal?.aborted && streamedResult === null && bytesToHash !== null) {
          let rawDigest;
          enterRunHash();
          try {
            rawDigest = await raceWithAbort(
              Promise.resolve(
                typeof hashFn === 'function' ? hashFn(bytesToHash) : computeSha256Hex(bytesToHash),
              ),
              signal,
            );
          } finally {
            leaveRunHash();
          }
          if (signal?.aborted) {
            entryAborted = true;
          } else {
            fileBytesHashed += bytesToHash.byteLength;
            const actualHex =
              typeof rawDigest === 'string' && SHA256_HEX_RE.test(rawDigest.trim())
                ? rawDigest.trim().toLowerCase()
                : null;
            const verified = actualHex !== null && actualHex === entry.sha256;
            hashInfo = {
              expected: entry.sha256,
              actual: actualHex,
              verified,
              checkedBytes: bytesToHash.byteLength,
              elapsedMs: computeElapsedMs(hashStart, now()),
            };
            if (!verified) {
              const msg = `SHA-256 ${safePath} digest (${actualHex || 'invalid'}) does not match manifest sha256 (${entry.sha256})`;
              fileErrors.push(msg);
              recordFileMismatch(MISMATCH_CATEGORIES.HASH_MISMATCH, msg);
            }
          }
        }
      } catch (err) {
        if (signal?.aborted || isAbortError(err)) {
          entryAborted = true;
        } else {
          const safeErr = sanitizeDiagnosticText(err instanceof Error ? err.message : String(err));
          fileErrors.push(`SHA-256 ${safePath} verification failed: ${safeErr}`);
        }
      } finally {
        bytesToHash = null;
      }
    }

    // Drop any reference to raw file bytes before finalizing diagnostics.
    rangeBytes = null;

    if (entryAborted || signal?.aborted) {
      return {
        aborted: true,
        bytesHashed: fileBytesHashed,
        fingerprintSkipped: false,
        bytesSkippedByFingerprint: 0,
      };
    }

    const fileElapsedMs = computeElapsedMs(fileStart, now());
    const fileOk = fileErrors.length === 0;
    const resolvedEtag = rangeInfo.etag || headInfo.etag || null;
    const resolvedStrongEtag = normalizeStrongEtag(resolvedEtag);
    let verifiedFingerprint = null;
    if (fileOk && safePath !== '<invalid-path>' && resolvedStrongEtag !== null) {
      const verifiedSha256 =
        hashInfo && hashInfo.verified === true
          ? hashInfo.actual || hashInfo.expected || null
          : fingerprintSkipped && cachedFingerprint
            ? cachedFingerprint.sha256
            : null;
      verifiedFingerprint = {
        path: safePath,
        etag: resolvedStrongEtag,
        size: entry.size,
        sha256: verifiedSha256,
      };
    }

    return {
      aborted: false,
      bytesVerified: fileBytesVerified,
      bytesHashed: fileBytesHashed,
      fingerprintSkipped,
      bytesSkippedByFingerprint: fileBytesSkippedByFingerprint,
      verifiedFingerprint,
      mismatches: fileMismatches,
      result: {
        path: entry.path,
        url,
        expectedSize: entry.size,
        ok: fileOk,
        errors: fileErrors,
        head: headInfo,
        range: rangeInfo,
        hash: hashInfo,
        etag: resolvedEtag,
        coopCoep:
          entry.size > 0 && !fingerprintSkipped ? headCoopCoep && rangeCoopCoep : headCoopCoep,
        elapsedMs: fileElapsedMs,
        bytesVerified: fileBytesVerified,
        fingerprintSkipped,
        bytesSkippedByFingerprint: fileBytesSkippedByFingerprint,
        mismatchCategories: Array.from(fileCategories),
      },
    };
  };

  const orderedOutcomes = new Array(selected.length).fill(null);
  let nextIndex = 0;
  const workerCount = Math.min(concurrency, selected.length);

  const runWorker = async () => {
    while (true) {
      if (signal?.aborted) {
        aborted = true;
        return;
      }
      const idx = nextIndex;
      if (idx >= selected.length) {
        return;
      }
      nextIndex += 1;

      const outcome = await verifySingleEntry(selected[idx]);
      if (outcome.aborted) {
        aborted = true;
        runAbortedCount += 1;
        moduleStats.abortedCount += 1;
        if (outcome.bytesHashed > 0) {
          runBytesHashed += outcome.bytesHashed;
          moduleStats.bytesHashed += outcome.bytesHashed;
        }
        return;
      }

      orderedOutcomes[idx] = outcome;
      runBytesHashed += outcome.bytesHashed;
      if (outcome.fingerprintSkipped) {
        runFingerprintSkippedCount += 1;
        runBytesSkippedByFingerprint += outcome.bytesSkippedByFingerprint;
        moduleFingerprintCacheStats.fingerprintSkippedCount += 1;
        moduleFingerprintCacheStats.bytesSkippedByFingerprint += outcome.bytesSkippedByFingerprint;
      }
      moduleStats.filesVerified += 1;
      moduleStats.bytesHashed += outcome.bytesHashed;
      moduleStats.mismatchCount += outcome.mismatches.length;
    }
  };

  if (workerCount > 0) {
    const workers = [];
    for (let w = 0; w < workerCount; w += 1) {
      workers.push(runWorker());
    }
    await Promise.all(workers);
  }

  if (aborted && runAbortedCount === 0) {
    runAbortedCount = 1;
    moduleStats.abortedCount += 1;
  }

  for (let i = 0; i < orderedOutcomes.length; i += 1) {
    const outcome = orderedOutcomes[i];
    if (!outcome) continue;
    for (const m of outcome.mismatches) {
      mismatchesByCategory[m.category] += 1;
      mismatches.push(m);
    }
    filesVerified += 1;
    bytesVerified += outcome.bytesVerified;
    if (outcome.result.ok) {
      passed += 1;
    } else {
      failed += 1;
    }
    if (outcome.verifiedFingerprint) {
      verifiedFingerprintsOut[outcome.verifiedFingerprint.path] = {
        etag: outcome.verifiedFingerprint.etag,
        size: outcome.verifiedFingerprint.size,
        sha256: outcome.verifiedFingerprint.sha256,
      };
    }
    results.push(outcome.result);
  }

  return finalize(manifestResult.ok, topErrors);
}

/**
 * Computes a single-shot hex digest over a byte buffer using the zero-dependency
 * incremental hashers (`sha256`, `crc32`, `fnv1a32`, `fnv1a64`).
 *
 * @param {Uint8Array | ArrayBuffer | ArrayBufferView} data
 * @param {string} [algorithm='sha256']
 * @returns {string}
 */
export function computeDigest(data, algorithm = STREAM_HASH_ALGORITHMS.SHA256) {
  const normalizedAlgorithm = normalizeHashAlgorithm(algorithm);
  const view = toUint8View(data);
  const hasher = createHasherState(normalizedAlgorithm);
  hasher.update(view);
  return hasher.digestHex();
}

function resolveAlgorithmAndExpectations(options) {
  const opts = isPlainObject(options) ? options : {};
  const entry = isPlainObject(opts.entry) ? opts.entry : null;

  let algorithm;
  if (opts.algorithm !== undefined && opts.algorithm !== null && opts.algorithm !== '') {
    algorithm = normalizeHashAlgorithm(opts.algorithm);
  } else if (entry) {
    if (typeof entry.sha256 === 'string') {
      algorithm = STREAM_HASH_ALGORITHMS.SHA256;
    } else if (typeof entry.fnv1a64 === 'string') {
      algorithm = STREAM_HASH_ALGORITHMS.FNV1A64;
    } else if (typeof entry.fnv1a32 === 'string') {
      algorithm = STREAM_HASH_ALGORITHMS.FNV1A32;
    } else if (typeof entry.crc32 === 'string') {
      algorithm = STREAM_HASH_ALGORITHMS.CRC32;
    } else {
      algorithm = STREAM_HASH_ALGORITHMS.SHA256;
    }
  } else {
    algorithm = STREAM_HASH_ALGORITHMS.SHA256;
  }

  const rawExpectedDigest =
    opts.expectedDigest !== undefined
      ? opts.expectedDigest
      : opts.expectedHash !== undefined
        ? opts.expectedHash
        : opts.expected !== undefined
          ? opts.expected
          : entry && entry[algorithm] !== undefined
            ? entry[algorithm]
            : null;
  const expectedDigest = normalizeExpectedDigest(rawExpectedDigest, algorithm, 'expectedDigest');

  const rawExpectedSize =
    opts.expectedSize !== undefined
      ? opts.expectedSize
      : entry && entry.size !== undefined
        ? entry.size
        : null;
  let expectedSize = null;
  if (rawExpectedSize !== undefined && rawExpectedSize !== null) {
    if (!isNonNegativeSafeInt(rawExpectedSize)) {
      throw new RangeError('expectedSize must be a non-negative safe integer');
    }
    expectedSize = rawExpectedSize;
  }

  const rawChunkSize =
    opts.chunkSize !== undefined
      ? opts.chunkSize
      : entry && (entry.chunkSize !== undefined || entry.chunk_size !== undefined)
        ? (entry.chunkSize ?? entry.chunk_size)
        : null;
  let chunkSize = null;
  if (rawChunkSize !== undefined && rawChunkSize !== null) {
    if (!isPositiveSafeInt(rawChunkSize)) {
      throw new RangeError('chunkSize must be a positive safe integer');
    }
    chunkSize = rawChunkSize;
  }

  const rawChunkDigests =
    opts.expectedChunkDigests !== undefined
      ? opts.expectedChunkDigests
      : opts.chunkDigests !== undefined
        ? opts.chunkDigests
        : entry && (entry.chunkHashes !== undefined || entry.chunk_hashes !== undefined)
          ? (entry.chunkHashes ?? entry.chunk_hashes)
          : null;
  let expectedChunkDigests = null;
  if (rawChunkDigests !== undefined && rawChunkDigests !== null) {
    if (!Array.isArray(rawChunkDigests)) {
      throw new TypeError('expectedChunkDigests must be an array');
    }
    expectedChunkDigests = rawChunkDigests.map((item, idx) => {
      const digestRaw = isPlainObject(item) ? item.digest : item;
      return normalizeExpectedDigest(digestRaw, algorithm, `expectedChunkDigests[${idx}]`);
    });
  }

  const rawPath = opts.path !== undefined ? opts.path : entry?.path ?? null;
  let safePath = null;
  if (rawPath !== undefined && rawPath !== null && String(rawPath).length > 0) {
    const pathCheck = validateManifestPath(rawPath);
    safePath = pathCheck.ok ? pathCheck.normalized : sanitizeDiagnosticText(rawPath);
  }

  return {
    algorithm,
    expectedDigest,
    expectedSize,
    chunkSize,
    expectedChunkDigests,
    safePath,
    failFast: Boolean(opts.failFast),
    signal: opts.signal ?? null,
    nowFn: opts.nowFn ?? null,
    onChunk: typeof opts.onChunk === 'function' ? opts.onChunk : null,
  };
}

/**
 * Creates a zero-copy, O(1)-memory incremental streaming verifier for large binary assets
 * (such as multi-hundred-MB `.pgim` guest images or `.wgsl.pack` shader packs).
 *
 * Chunks are hashed incrementally as they arrive via `update(chunk)` without buffering
 * duplicate copies in RAM. Supports `sha256` (default), `crc32`, `fnv1a32`, and `fnv1a64`,
 * optional per-chunk/per-block expected digest lists for immediate corrupted-chunk detection,
 * `AbortSignal` cancellation, injectable clock (`nowFn`), and structured mismatch diagnostics
 * containing `{ expected, actual, checkedBytes, elapsedMs }`.
 *
 * @param {object} [options]
 * @param {string} [options.algorithm='sha256'] 'sha256' | 'crc32' | 'fnv1a32' | 'fnv1a64'
 * @param {string | null} [options.expectedDigest=null] Expected hex digest for the full stream.
 * @param {number | null} [options.expectedSize=null] Expected total byte size.
 * @param {number | null} [options.chunkSize=null] Fixed block size in bytes when verifying `expectedChunkDigests` across arbitrary network chunk boundaries.
 * @param {Array<string | { digest: string }> | null} [options.expectedChunkDigests=null] Optional expected per-chunk/per-block hex digests.
 * @param {object | null} [options.entry=null] Optional manifest file entry shorthand.
 * @param {string | null} [options.path=null] Optional manifest-relative path for sanitized diagnostics.
 * @param {boolean} [options.failFast=false] When true, stop hashing subsequent chunks after the first chunk/size mismatch.
 * @param {AbortSignal | null} [options.signal=null] Optional AbortSignal.
 * @param {(() => number) | null} [options.nowFn=null] Optional injectable monotonic clock in ms.
 * @param {((updateResult: object) => void) | null} [options.onChunk=null] Optional per-update progress callback.
 */
export function createIncrementalVerifier(options = {}) {
  const skipModuleStats = Boolean(isPlainObject(options) && options[INTERNAL_SKIP_STATS]);
  const config = resolveAlgorithmAndExpectations(options);
  const now = resolveNowFn(config.nowFn);

  const streamHasher = createHasherState(config.algorithm);
  const blockHasher =
    config.expectedChunkDigests !== null ? createHasherState(config.algorithm) : null;

  let startTime = now();
  let checkedBytes = 0;
  let chunkCount = 0;
  let blockIndex = 0;
  let blockBytesWritten = 0;
  let verifiedBlocks = 0;
  let aborted = Boolean(config.signal?.aborted);
  let finalized = false;
  let failedFast = false;
  let sizeOverflowRecorded = false;
  let cachedResult = null;
  let errors = [];
  let mismatches = [];
  let chunkMismatches = [];
  let mismatchesByCategory = createMismatchesByCategory();

  const label = config.safePath || '<stream>';

  const recordMismatch = ({
    category,
    kind,
    expected,
    actual,
    elapsedMs,
    chunkIndex = null,
    offset = null,
    length = null,
    detail,
  }) => {
    const safeDetail = sanitizeDiagnosticText(detail);
    const diag = {
      category,
      kind,
      path: config.safePath,
      algorithm: config.algorithm,
      expected,
      actual,
      checkedBytes,
      elapsedMs,
      chunkIndex,
      offset,
      length,
      detail: safeDetail,
    };
    mismatches.push(diag);
    mismatchesByCategory[category] = (mismatchesByCategory[category] || 0) + 1;
    errors.push(safeDetail);
    if (kind === 'chunk_digest_mismatch' || kind === 'chunk_count_mismatch') {
      chunkMismatches.push(diag);
    }
    return diag;
  };

  const evaluateBlockDigest = (actualDigest, bIndex, offset, length, elapsedMs) => {
    if (bIndex >= config.expectedChunkDigests.length) {
      return recordMismatch({
        category: MISMATCH_CATEGORIES.SIZE_MISMATCH,
        kind: 'chunk_count_mismatch',
        expected: config.expectedChunkDigests.length,
        actual: bIndex + 1,
        elapsedMs,
        chunkIndex: bIndex,
        offset,
        length,
        detail: `${label} chunk #${bIndex} at offset ${offset} (${length} bytes) exceeds expected chunk count (${config.expectedChunkDigests.length})`,
      });
    }
    const expected = config.expectedChunkDigests[bIndex];
    if (actualDigest !== expected) {
      return recordMismatch({
        category: MISMATCH_CATEGORIES.HASH_MISMATCH,
        kind: 'chunk_digest_mismatch',
        expected,
        actual: actualDigest,
        elapsedMs,
        chunkIndex: bIndex,
        offset,
        length,
        detail: `${label} ${config.algorithm} chunk #${bIndex} mismatch at offset ${offset} (${length} bytes): expected ${expected}, actual ${actualDigest}`,
      });
    }
    verifiedBlocks += 1;
    return null;
  };

  const reset = () => {
    streamHasher.reset();
    if (blockHasher) {
      blockHasher.reset();
    }
    startTime = now();
    checkedBytes = 0;
    chunkCount = 0;
    blockIndex = 0;
    blockBytesWritten = 0;
    verifiedBlocks = 0;
    aborted = Boolean(config.signal?.aborted);
    finalized = false;
    failedFast = false;
    sizeOverflowRecorded = false;
    cachedResult = null;
    errors = [];
    mismatches = [];
    chunkMismatches = [];
    mismatchesByCategory = createMismatchesByCategory();
  };

  const update = (chunk) => {
    if (finalized) {
      throw new Error('Incremental verifier has already been finalized; call reset() before reusing');
    }
    const view = toUint8View(chunk);

    if (aborted || config.signal?.aborted) {
      aborted = true;
      const elapsedMs = computeElapsedMs(startTime, now());
      const res = {
        ok: false,
        aborted: true,
        chunkIndex: chunkCount,
        chunkBytes: 0,
        checkedBytes,
        chunkCount,
        elapsedMs,
        chunkDigest: null,
        mismatch: null,
      };
      if (config.onChunk) config.onChunk(res);
      return res;
    }

    const currentChunkIndex = chunkCount;
    const chunkStartOffset = checkedBytes;
    const chunkBytes = view.byteLength;
    chunkCount += 1;

    if (failedFast) {
      const elapsedMs = computeElapsedMs(startTime, now());
      const res = {
        ok: false,
        aborted: false,
        chunkIndex: currentChunkIndex,
        chunkBytes: 0,
        checkedBytes,
        chunkCount,
        elapsedMs,
        chunkDigest: null,
        mismatch: mismatches[0] || null,
      };
      if (config.onChunk) config.onChunk(res);
      return res;
    }

    streamHasher.update(view);
    checkedBytes += chunkBytes;

    let chunkDigest = null;
    let firstChunkMismatch = null;

    if (blockHasher !== null) {
      if (config.chunkSize !== null) {
        let offsetInView = 0;
        while (offsetInView < chunkBytes) {
          const room = config.chunkSize - blockBytesWritten;
          const take = Math.min(room, chunkBytes - offsetInView);
          blockHasher.update(view.subarray(offsetInView, offsetInView + take));
          blockBytesWritten += take;
          offsetInView += take;

          if (blockBytesWritten === config.chunkSize) {
            const digest = blockHasher.digestHex();
            blockHasher.reset();
            chunkDigest = digest;
            const elapsedNow = computeElapsedMs(startTime, now());
            const bOffset = blockIndex * config.chunkSize;
            const mismatch = evaluateBlockDigest(
              digest,
              blockIndex,
              bOffset,
              config.chunkSize,
              elapsedNow,
            );
            blockIndex += 1;
            blockBytesWritten = 0;
            if (mismatch && !firstChunkMismatch) {
              firstChunkMismatch = mismatch;
              if (config.failFast) {
                failedFast = true;
                break;
              }
            }
          }
        }
      } else {
        blockHasher.reset();
        blockHasher.update(view);
        chunkDigest = blockHasher.digestHex();
        const elapsedNow = computeElapsedMs(startTime, now());
        const mismatch = evaluateBlockDigest(
          chunkDigest,
          currentChunkIndex,
          chunkStartOffset,
          chunkBytes,
          elapsedNow,
        );
        blockIndex += 1;
        if (mismatch) {
          firstChunkMismatch = mismatch;
          if (config.failFast) {
            failedFast = true;
          }
        }
      }
    }

    let sizeMismatch = null;
    if (
      !sizeOverflowRecorded &&
      config.expectedSize !== null &&
      checkedBytes > config.expectedSize
    ) {
      sizeOverflowRecorded = true;
      const elapsedNow = computeElapsedMs(startTime, now());
      sizeMismatch = recordMismatch({
        category: MISMATCH_CATEGORIES.SIZE_MISMATCH,
        kind: 'size_mismatch',
        expected: config.expectedSize,
        actual: checkedBytes,
        elapsedMs: elapsedNow,
        chunkIndex: currentChunkIndex,
        offset: chunkStartOffset,
        length: chunkBytes,
        detail: `${label} stream exceeded expected size (${config.expectedSize} bytes; received ${checkedBytes} bytes)`,
      });
      if (config.failFast) {
        failedFast = true;
      }
    }

    const elapsedMs = computeElapsedMs(startTime, now());
    const mismatch = firstChunkMismatch || sizeMismatch || null;
    const res = {
      ok: mismatch === null && mismatches.length === 0,
      aborted: false,
      chunkIndex: currentChunkIndex,
      chunkBytes,
      checkedBytes,
      chunkCount,
      elapsedMs,
      chunkDigest,
      mismatch,
    };
    if (config.onChunk) config.onChunk(res);
    return res;
  };

  const finalize = () => {
    if (finalized && cachedResult !== null) {
      return cachedResult;
    }
    finalized = true;

    if (aborted || config.signal?.aborted) {
      aborted = true;
      const elapsedMs = computeElapsedMs(startTime, now());
      const telemetry = buildTelemetry({
        elapsedMs,
        bytesVerified: checkedBytes,
        filesVerified: 0,
        mismatchesByCategory,
        aborted: true,
      });
      cachedResult = {
        ok: false,
        aborted: true,
        path: config.safePath,
        algorithm: config.algorithm,
        digest: null,
        expected: config.expectedDigest !== null ? config.expectedDigest : config.expectedSize,
        actual: null,
        expectedDigest: config.expectedDigest,
        actualDigest: null,
        expectedSize: config.expectedSize,
        actualSize: checkedBytes,
        checkedBytes,
        chunkCount,
        verifiedBlocks,
        sizeVerified: null,
        digestVerified: null,
        chunkMismatches,
        mismatches,
        mismatch: mismatches[0] || null,
        mismatchCount: telemetry.mismatchCount,
        mismatchesByCategory: { ...telemetry.mismatchesByCategory },
        errors,
        elapsedMs: telemetry.elapsedMs,
        bytesVerified: telemetry.bytesVerified,
        filesVerified: telemetry.filesVerified,
        bytesPerSecond: telemetry.bytesPerSecond,
        telemetry: {
          ...telemetry,
          checkedBytes,
          chunkCount,
          verifiedBlocks,
        },
      };
      if (!skipModuleStats) {
        moduleStats.bytesHashed += checkedBytes;
        moduleStats.mismatchCount += telemetry.mismatchCount;
        moduleStats.abortedCount += 1;
      }
      return cachedResult;
    }

    // Flush trailing partial fixed-size block if present.
    if (blockHasher !== null && config.chunkSize !== null && blockBytesWritten > 0 && !failedFast) {
      const trailingDigest = blockHasher.digestHex();
      blockHasher.reset();
      const elapsedNow = computeElapsedMs(startTime, now());
      evaluateBlockDigest(
        trailingDigest,
        blockIndex,
        blockIndex * config.chunkSize,
        blockBytesWritten,
        elapsedNow,
      );
      blockIndex += 1;
      blockBytesWritten = 0;
    }

    // Check for missing expected chunks when fewer chunks arrived than declared.
    if (
      blockHasher !== null &&
      !failedFast &&
      blockIndex < config.expectedChunkDigests.length
    ) {
      const elapsedNow = computeElapsedMs(startTime, now());
      recordMismatch({
        category: MISMATCH_CATEGORIES.SIZE_MISMATCH,
        kind: 'chunk_count_mismatch',
        expected: config.expectedChunkDigests.length,
        actual: blockIndex,
        elapsedMs: elapsedNow,
        chunkIndex: blockIndex,
        offset: checkedBytes,
        length: 0,
        detail: `${label} ended after ${blockIndex} chunks (expected ${config.expectedChunkDigests.length})`,
      });
    }

    const actualDigest = failedFast ? null : streamHasher.digestHex();
    const elapsedMs = computeElapsedMs(startTime, now());

    let sizeVerified = null;
    if (config.expectedSize !== null) {
      sizeVerified = checkedBytes === config.expectedSize;
      if (!sizeVerified && !sizeOverflowRecorded) {
        recordMismatch({
          category: MISMATCH_CATEGORIES.SIZE_MISMATCH,
          kind: 'size_mismatch',
          expected: config.expectedSize,
          actual: checkedBytes,
          elapsedMs,
          detail: `${label} size mismatch: expected ${config.expectedSize} bytes, actual ${checkedBytes} bytes`,
        });
      }
    }

    let digestVerified = null;
    if (config.expectedDigest !== null) {
      digestVerified = actualDigest !== null && actualDigest === config.expectedDigest;
      if (!digestVerified && !failedFast) {
        recordMismatch({
          category: MISMATCH_CATEGORIES.HASH_MISMATCH,
          kind: 'digest_mismatch',
          expected: config.expectedDigest,
          actual: actualDigest,
          elapsedMs,
          detail: `${label} ${config.algorithm} digest mismatch: expected ${config.expectedDigest}, actual ${actualDigest}`,
        });
      }
    }

    const telemetry = buildTelemetry({
      elapsedMs,
      bytesVerified: checkedBytes,
      filesVerified: 1,
      mismatchesByCategory,
      aborted: false,
    });

    const topExpected =
      config.expectedDigest !== null
        ? config.expectedDigest
        : config.expectedSize !== null
          ? config.expectedSize
          : null;
    const topActual =
      config.expectedDigest !== null
        ? actualDigest
        : config.expectedSize !== null
          ? checkedBytes
          : actualDigest;

    cachedResult = {
      ok: errors.length === 0 && mismatches.length === 0,
      aborted: false,
      path: config.safePath,
      algorithm: config.algorithm,
      digest: actualDigest,
      expected: topExpected,
      actual: topActual,
      expectedDigest: config.expectedDigest,
      actualDigest,
      expectedSize: config.expectedSize,
      actualSize: checkedBytes,
      checkedBytes,
      chunkCount,
      verifiedBlocks,
      sizeVerified,
      digestVerified,
      chunkMismatches,
      mismatches,
      mismatch: mismatches[0] || null,
      mismatchCount: telemetry.mismatchCount,
      mismatchesByCategory: { ...telemetry.mismatchesByCategory },
      errors,
      elapsedMs: telemetry.elapsedMs,
      bytesVerified: telemetry.bytesVerified,
      filesVerified: telemetry.filesVerified,
      bytesPerSecond: telemetry.bytesPerSecond,
      telemetry: {
        ...telemetry,
        checkedBytes,
        chunkCount,
        verifiedBlocks,
      },
    };
    if (!skipModuleStats) {
      moduleStats.filesVerified += 1;
      moduleStats.bytesHashed += checkedBytes;
      moduleStats.mismatchCount += telemetry.mismatchCount;
    }
    return cachedResult;
  };

  const getSnapshot = () => {
    const elapsedMs = computeElapsedMs(startTime, now());
    const telemetry = buildTelemetry({
      elapsedMs,
      bytesVerified: checkedBytes,
      filesVerified: finalized && !aborted ? 1 : 0,
      mismatchesByCategory,
      aborted: aborted || Boolean(config.signal?.aborted),
    });
    return {
      path: config.safePath,
      algorithm: config.algorithm,
      checkedBytes,
      chunkCount,
      verifiedBlocks,
      elapsedMs: telemetry.elapsedMs,
      bytesPerSecond: telemetry.bytesPerSecond,
      mismatchCount: telemetry.mismatchCount,
      mismatchesByCategory: { ...telemetry.mismatchesByCategory },
      aborted: telemetry.aborted,
      finalized,
    };
  };

  return {
    algorithm: config.algorithm,
    path: config.safePath,
    update,
    finalize,
    reset,
    getSnapshot,
    snapshot: getSnapshot,
  };
}

/**
 * Incrementally verifies a `ReadableStream`, `AsyncIterable`, `Iterable` of chunks, or a
 * single byte buffer using `createIncrementalVerifier(options)` in O(1) memory.
 *
 * @param {ReadableStream | AsyncIterable | Iterable | Uint8Array | ArrayBuffer | ArrayBufferView} source
 * @param {object} [options]
 * @returns {Promise<ReturnType<ReturnType<typeof createIncrementalVerifier>['finalize']>>}
 */
export async function verifyStream(source, options = {}) {
  const opts = isPlainObject(options) ? options : {};
  const skipModuleStats = Boolean(opts[INTERNAL_SKIP_STATS]);
  const verifier = createIncrementalVerifier({
    ...opts,
    [INTERNAL_SKIP_STATS]: true,
  });
  const signal = opts.signal ?? null;
  const failFast = Boolean(opts.failFast);

  if (signal?.aborted) {
    const result = verifier.finalize();
    if (!skipModuleStats) {
      moduleStats.abortedCount += 1;
    }
    return result;
  }

  const isBufferSource =
    source instanceof Uint8Array ||
    source instanceof ArrayBuffer ||
    (typeof SharedArrayBuffer !== 'undefined' && source instanceof SharedArrayBuffer) ||
    ArrayBuffer.isView(source);
  const isReaderSource = Boolean(source && typeof source.getReader === 'function');
  const isAsyncIterable = Boolean(source && typeof source[Symbol.asyncIterator] === 'function');
  const isSyncIterable = Boolean(source && typeof source[Symbol.iterator] === 'function');

  if (!isBufferSource && !isReaderSource && !isAsyncIterable && !isSyncIterable) {
    throw new TypeError(
      'verifyStream source must be a ReadableStream, AsyncIterable, Iterable of chunks, or byte buffer',
    );
  }

  const recordStreamStats = (result) => {
    if (skipModuleStats) return;
    moduleStats.bytesHashed += result.checkedBytes;
    moduleStats.mismatchCount += result.mismatchCount;
    if (result.aborted) {
      moduleStats.abortedCount += 1;
    } else {
      moduleStats.filesVerified += 1;
    }
  };

  if (!skipModuleStats) {
    enterModuleHash();
  }
  try {
    if (isBufferSource) {
      verifier.update(source);
      const result = verifier.finalize();
      recordStreamStats(result);
      return result;
    }

    if (isReaderSource) {
      const reader = source.getReader();
      let shouldCancel = false;
      try {
        while (true) {
          if (signal?.aborted) {
            shouldCancel = true;
            break;
          }
          let step;
          try {
            step = await raceWithAbort(reader.read(), signal);
          } catch (err) {
            if (signal?.aborted || isAbortError(err)) {
              shouldCancel = true;
              break;
            }
            throw err;
          }
          if (!step || step.done) {
            break;
          }
          if (step.value !== undefined && step.value !== null) {
            const updateRes = verifier.update(step.value);
            if (updateRes.aborted || (failFast && !updateRes.ok)) {
              shouldCancel = true;
              break;
            }
          }
        }
      } finally {
        if (shouldCancel) {
          try {
            await reader.cancel();
          } catch {
            // Ignore stream cancel errors.
          }
        }
        try {
          reader.releaseLock();
        } catch {
          // Ignore lock release errors.
        }
      }
      const result = verifier.finalize();
      recordStreamStats(result);
      return result;
    }

    if (isAsyncIterable) {
      const iterator = source[Symbol.asyncIterator]();
      let shouldClose = false;
      try {
        while (true) {
          if (signal?.aborted) {
            shouldClose = true;
            break;
          }
          let step;
          try {
            step = await raceWithAbort(iterator.next(), signal);
          } catch (err) {
            if (signal?.aborted || isAbortError(err)) {
              shouldClose = true;
              break;
            }
            throw err;
          }
          if (!step || step.done) {
            break;
          }
          const updateRes = verifier.update(step.value);
          if (updateRes.aborted || (failFast && !updateRes.ok)) {
            shouldClose = true;
            break;
          }
        }
      } finally {
        if (shouldClose && typeof iterator.return === 'function') {
          try {
            await iterator.return();
          } catch {
            // Ignore iterator close errors.
          }
        }
      }
      const result = verifier.finalize();
      recordStreamStats(result);
      return result;
    }

    for (const chunk of source) {
      if (signal?.aborted) {
        break;
      }
      const updateRes = verifier.update(chunk);
      if (updateRes.aborted || (failFast && !updateRes.ok)) {
        break;
      }
    }
    const result = verifier.finalize();
    recordStreamStats(result);
    return result;
  } finally {
    if (!skipModuleStats) {
      leaveModuleHash();
    }
  }
}
