// Browser & Node streaming manifest (`data-manifest.json`) and bootset (`bootset.json`)
// schema validator, partial-download readiness calculator, and live HTTP Range/ETag verifier.
// Matches `docs/io_manifest.md`, `tools/src/main.rs`, `tools/src/rpf9.rs`, `tools/src/scan.rs`,
// and `server/serve_local.py`. Zero external dependencies.

export const MANIFEST_ABI_VERSION = 1;
export const DEFAULT_SAMPLE_COUNT = 5;
export const DEFAULT_RANGE_PROBE_BYTES = 1;

export const MISMATCH_CATEGORIES = Object.freeze({
  MISSING: 'missing',
  SIZE_MISMATCH: 'size_mismatch',
  HASH_MISMATCH: 'hash_mismatch',
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
const WINDOWS_DRIVE_RE = /^[A-Za-z]:/;
const CONTROL_CHAR_RE = /[\x00-\x1f\x7f]/;
const CONTENT_RANGE_RE = /^bytes\s+(\d+)-(\d+)\/(\d+|\*)$/i;
const QUOTED_ETAG_RE = /^(?:W\/)?"[^"]*"$/;

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
  const subtle = globalThis.crypto?.subtle;
  if (!subtle || typeof subtle.digest !== 'function') {
    throw new Error('SHA-256 verification requires Web Crypto crypto.subtle.digest or a custom hashFn');
  }
  const view =
    bytes instanceof Uint8Array
      ? bytes
      : bytes instanceof ArrayBuffer
        ? new Uint8Array(bytes)
        : new Uint8Array(0);
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
  const fileMap = new Map();
  const summary = emptyManifestSummary();
  const mismatchesByCategory = createMismatchesByCategory();

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
      errors.push(
        `Manifest JSON parse error: ${sanitizeDiagnosticText(err instanceof Error ? err.message : String(err))}`,
      );
      return finalize(false);
    }
  }

  if (!isPlainObject(manifest)) {
    errors.push('Manifest must be a JSON object');
    return finalize(false);
  }

  // Version validation: canonical field is `abi_version`, with optional fallback to `version`.
  let resolvedVersion = null;
  if (manifest.abi_version !== undefined) {
    if (!isPositiveSafeInt(manifest.abi_version)) {
      errors.push(
        `Manifest "abi_version" must be a positive integer (got ${JSON.stringify(manifest.abi_version)})`,
      );
    } else if (manifest.abi_version !== MANIFEST_ABI_VERSION) {
      errors.push(
        `Unsupported manifest "abi_version": ${manifest.abi_version} (expected ${MANIFEST_ABI_VERSION})`,
      );
    } else {
      resolvedVersion = manifest.abi_version;
    }
    if (manifest.version !== undefined && manifest.version !== manifest.abi_version) {
      warnings.push(
        `Manifest "version" (${JSON.stringify(manifest.version)}) differs from "abi_version" (${JSON.stringify(manifest.abi_version)})`,
      );
    }
  } else if (manifest.version !== undefined) {
    if (!isPositiveSafeInt(manifest.version)) {
      errors.push(
        `Manifest "version" must be a positive integer (got ${JSON.stringify(manifest.version)})`,
      );
    } else if (manifest.version !== MANIFEST_ABI_VERSION) {
      errors.push(
        `Unsupported manifest "version": ${manifest.version} (expected ${MANIFEST_ABI_VERSION})`,
      );
    } else {
      resolvedVersion = manifest.version;
      warnings.push('Manifest uses "version" instead of canonical "abi_version"');
    }
  } else {
    errors.push(`Manifest is missing required "abi_version" (expected ${MANIFEST_ABI_VERSION})`);
  }
  summary.abiVersion = resolvedVersion;

  if (manifest.file_count !== undefined) {
    if (!isNonNegativeSafeInt(manifest.file_count)) {
      errors.push(
        `Manifest "file_count" must be a non-negative integer (got ${JSON.stringify(manifest.file_count)})`,
      );
    } else {
      summary.declaredFileCount = manifest.file_count;
    }
  }

  if (manifest.total_bytes !== undefined) {
    if (!isNonNegativeSafeInt(manifest.total_bytes)) {
      errors.push(
        `Manifest "total_bytes" must be a non-negative integer (got ${JSON.stringify(manifest.total_bytes)})`,
      );
    } else {
      summary.declaredTotalBytes = manifest.total_bytes;
    }
  }

  if (!Array.isArray(manifest.files)) {
    errors.push('Manifest "files" must be an array');
    return finalize(false);
  }

  if (manifest.files.length === 0) {
    warnings.push('Manifest "files" array is empty');
  }

  if (summary.declaredFileCount !== null && summary.declaredFileCount !== manifest.files.length) {
    errors.push(
      `Manifest "file_count" (${summary.declaredFileCount}) does not match files.length (${manifest.files.length})`,
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
      errors.push(`${label} must be an object`);
      allSizesValid = false;
      continue;
    }

    const pathCheck = validateManifestPath(raw.path);
    if (!pathCheck.ok) {
      errors.push(`${label}.path: ${pathCheck.error}`);
    }

    const sizeValid = isNonNegativeSafeInt(raw.size);
    if (!sizeValid) {
      errors.push(
        `${label}.size must be a non-negative safe integer (got ${JSON.stringify(raw.size)})`,
      );
      allSizesValid = false;
    } else {
      sumAllValidSizes += raw.size;
      if (raw.size === 0) {
        warnings.push(`${label} ("${pathCheck.normalized || raw.path}") has zero byte size`);
      }
    }

    // Optional sha256 validation (64 hex chars).
    let sha256 = null;
    if (raw.sha256 !== undefined && raw.sha256 !== null) {
      if (typeof raw.sha256 !== 'string' || !SHA256_HEX_RE.test(raw.sha256)) {
        errors.push(
          `${label}.sha256 must be a 64-character hexadecimal string (got ${JSON.stringify(raw.sha256)})`,
        );
      } else {
        sha256 = raw.sha256.toLowerCase();
      }
    }

    // Optional etag validation.
    let etag = null;
    if (raw.etag !== undefined && raw.etag !== null) {
      if (typeof raw.etag !== 'string' || raw.etag.trim().length === 0) {
        errors.push(`${label}.etag must be a non-empty string`);
      } else {
        etag = raw.etag.trim();
        if (!QUOTED_ETAG_RE.test(etag)) {
          warnings.push(`${label}.etag (${JSON.stringify(etag)}) is not RFC 9110 quoted`);
        }
      }
    }

    // Optional incomplete boolean flag.
    let explicitIncomplete = null;
    if (raw.incomplete !== undefined) {
      if (typeof raw.incomplete !== 'boolean') {
        errors.push(
          `${label}.incomplete must be a boolean (got ${JSON.stringify(raw.incomplete)})`,
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
        errors.push(`${label}.status must be a non-empty string`);
      } else {
        const lowerStatus = raw.status.trim().toLowerCase();
        status = lowerStatus;
        if (INCOMPLETE_STATUS_VALUES.has(lowerStatus)) {
          statusMarksIncomplete = true;
        } else if (COMPLETE_STATUS_VALUES.has(lowerStatus)) {
          statusMarksComplete = true;
        } else {
          warnings.push(`${label}.status has unrecognized value "${raw.status}"`);
        }
      }
    }

    if (explicitIncomplete === true && statusMarksComplete) {
      warnings.push(
        `${label} ("${pathCheck.normalized || raw.path}") has conflicting incomplete=true and status="${status}"`,
      );
    } else if (explicitIncomplete === false && statusMarksIncomplete) {
      warnings.push(
        `${label} ("${pathCheck.normalized || raw.path}") has conflicting incomplete=false and status="${status}"`,
      );
    }

    if (!pathCheck.ok || !sizeValid) {
      continue;
    }

    const normalizedPath = pathCheck.normalized;
    if (fileMap.has(normalizedPath)) {
      summary.duplicateCount += 1;
      errors.push(`${label}.path duplicates an earlier manifest entry: "${normalizedPath}"`);
      continue;
    }

    const lowerPath = normalizedPath.toLowerCase();
    if (lowerPathMap.has(lowerPath)) {
      warnings.push(
        `${label}.path ("${normalizedPath}") collides case-insensitively with "${lowerPathMap.get(lowerPath)}"`,
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
      warnings.push(
        `${label} ("${normalizedPath}") has a temporary download suffix and is treated as incomplete`,
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
      errors.push(
        `Manifest "total_bytes" (${summary.declaredTotalBytes}) does not match sum of file sizes (${sumAllValidSizes})`,
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
      errors.push(
        `Bootset JSON parse error: ${sanitizeDiagnosticText(err instanceof Error ? err.message : String(err))}`,
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
    errors.push('Bootset must be an array of range entries or an object with an "entries" array');
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
      errors.push(`${label} must be an object`);
      continue;
    }

    const pathCheck = validateManifestPath(raw.path);
    if (!pathCheck.ok) {
      errors.push(`${label}.path: ${pathCheck.error}`);
    }

    const offsetValid = isNonNegativeSafeInt(raw.offset);
    if (!offsetValid) {
      errors.push(
        `${label}.offset must be a non-negative safe integer (got ${JSON.stringify(raw.offset)})`,
      );
    }

    const lengthValid = isPositiveSafeInt(raw.length);
    if (!lengthValid) {
      errors.push(
        `${label}.length must be a positive safe integer (got ${JSON.stringify(raw.length)})`,
      );
    }

    if (!pathCheck.ok || !offsetValid || !lengthValid) {
      continue;
    }

    const end = raw.offset + raw.length;
    if (!Number.isSafeInteger(end)) {
      errors.push(`${label} range offset + length overflows safe integer bounds`);
      continue;
    }

    const relPath = pathCheck.normalized;

    if (fileMap !== null) {
      const manifestFile = fileMap.get(relPath);
      if (!manifestFile) {
        missingSet.add(relPath);
        mismatchesByCategory[MISMATCH_CATEGORIES.MISSING] += 1;
        errors.push(`${label} references file missing from manifest: "${relPath}"`);
        continue;
      }
      if (manifestFile.incomplete) {
        incompleteSet.add(relPath);
        warnings.push(`${label} references incomplete manifest file: "${relPath}"`);
      }
      if (end > manifestFile.size) {
        mismatchesByCategory[MISMATCH_CATEGORIES.SIZE_MISMATCH] += 1;
        errors.push(
          `${label} range [${raw.offset}, ${end}) exceeds manifest file size ${manifestFile.size} for "${relPath}"`,
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
      warnings.push(
        `${label} duplicates bootset range [${raw.offset}, ${end}) for "${relPath}"`,
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
          warnings.push(
            `Bootset ranges overlap by ${overlapBytes} bytes for "${relPath}" ([${current.offset}, ${current.end}) and [${next.offset}, ${next.end}))`,
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

async function consumeResponseBuffer(res) {
  if (!res) return { byteLength: null, bytes: null };
  if (typeof res.arrayBuffer === 'function') {
    const buf = await res.arrayBuffer();
    if (!buf) {
      return { byteLength: 0, bytes: new Uint8Array(0) };
    }
    const bytes = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
    return { byteLength: bytes.byteLength, bytes };
  }
  if (res.body && typeof res.body.cancel === 'function') {
    try {
      await res.body.cancel();
    } catch {
      // Ignore stream cancellation errors.
    }
  }
  return { byteLength: null, bytes: null };
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
 * `HEAD` and `GET Range: bytes=0-<end>` requests. Verifies `206 Partial Content`,
 * `Accept-Ranges: bytes`, `Content-Range` total length matching the manifest size,
 * response byte count, `ETag` presence/consistency, and optional SHA-256 content hashes.
 *
 * Supports optional `AbortSignal` cancellation (`signal`) and injectable clock (`nowFn`),
 * returning sanitized additive throughput/timing telemetry and structured mismatch categories
 * without leaking host paths or raw file bytes.
 *
 * @param {object} options
 * @param {typeof fetch} [options.fetchImpl] Injectable fetch implementation (defaults to globalThis.fetch).
 * @param {string} [options.dataBase='/data/'] Base URL serving manifest files.
 * @param {unknown} options.manifest Raw manifest object, JSON string, or validateDataManifest() result.
 * @param {number} [options.sampleCount=5] Maximum number of complete files to probe.
 * @param {number} [options.probeBytes=1] Number of prefix bytes to request in the Range probe (1 -> bytes=0-0).
 * @param {boolean} [options.checkCoopCoep=false] When true, fail probes missing COOP/COEP headers.
 * @param {boolean} [options.verifyHashes=false] When true, verify SHA-256 digests for entries declaring `sha256`.
 * @param {((bytes: Uint8Array) => Promise<string> | string) | null} [options.hashFn=null] Optional SHA-256 hex digest provider.
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
 *   elapsedMs: number,
 *   bytesVerified: number,
 *   filesVerified: number,
 *   mismatchCount: number,
 *   mismatchesByCategory: { missing: number, size_mismatch: number, hash_mismatch: number },
 *   mismatches: Array<{ path: string, category: 'missing' | 'size_mismatch' | 'hash_mismatch', detail: string }>,
 *   telemetry: {
 *     elapsedMs: number,
 *     bytesVerified: number,
 *     filesVerified: number,
 *     bytesPerSecond: number,
 *     mismatchCount: number,
 *     mismatchesByCategory: { missing: number, size_mismatch: number, hash_mismatch: number },
 *     aborted: boolean,
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
    checkCoopCoep = false,
    verifyHashes = false,
    hashFn = null,
    signal = null,
    nowFn = null,
  } = options || {};

  if (typeof fetchImpl !== 'function') {
    throw new TypeError('verifyLiveEndpoints requires a fetchImpl function');
  }

  const now = resolveNowFn(nowFn);
  const startTime = now();

  const mismatchesByCategory = createMismatchesByCategory();
  const mismatches = [];
  const results = [];
  let passed = 0;
  let failed = 0;
  let skippedIncomplete = 0;
  let bytesVerified = 0;
  let filesVerified = 0;
  let aborted = false;

  const finalize = (manifestOk, topErrors) => {
    const elapsedMs = computeElapsedMs(startTime, now());
    const telemetry = buildTelemetry({
      elapsedMs,
      bytesVerified,
      filesVerified,
      mismatchesByCategory,
      aborted,
    });
    return {
      ok: !aborted && manifestOk && failed === 0,
      probed: results.length,
      passed,
      failed,
      skippedIncomplete,
      errors: topErrors,
      results,
      aborted: telemetry.aborted,
      elapsedMs: telemetry.elapsedMs,
      bytesVerified: telemetry.bytesVerified,
      filesVerified: telemetry.filesVerified,
      mismatchCount: telemetry.mismatchCount,
      mismatchesByCategory: { ...telemetry.mismatchesByCategory },
      mismatches,
      telemetry,
    };
  };

  if (signal?.aborted) {
    aborted = true;
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

  for (const entry of selected) {
    if (signal?.aborted) {
      aborted = true;
      break;
    }

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
    let rangeBytes = null;
    let rangeEndByte = -1;
    let entryAborted = false;

    // 1. HEAD probe
    try {
      const headInit = signal ? { method: 'HEAD', signal } : { method: 'HEAD' };
      const headRes = await raceWithAbort(fetchImpl(url, headInit), signal);
      if (signal?.aborted) {
        entryAborted = true;
      } else {
        headInfo.status = headRes ? headRes.status : null;
        headInfo.acceptRanges = getHeader(headRes?.headers, 'accept-ranges');
        headInfo.etag = getHeader(headRes?.headers, 'etag');
        const headLenRaw = getHeader(headRes?.headers, 'content-length');
        headInfo.contentLength =
          headLenRaw !== null && /^\d+$/.test(headLenRaw.trim()) ? Number(headLenRaw.trim()) : null;
        headCoopCoep = hasCoopCoepHeaders(headRes?.headers);

        if (headRes?.body && typeof headRes.body.cancel === 'function') {
          try {
            await headRes.body.cancel();
          } catch {
            // Ignore body cancel errors on HEAD.
          }
        }

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
      aborted = true;
      break;
    }

    // 2. Range GET probe (only when file has at least 1 byte; 0-byte files cannot satisfy bytes=0-0)
    if (entry.size > 0) {
      const endByte = Math.min(entry.size - 1, requestedProbeBytes - 1);
      const expectedLength = endByte + 1;
      const rangeHeader = `bytes=0-${endByte}`;
      rangeInfo.requested = rangeHeader;
      rangeEndByte = endByte;

      try {
        const rangeInit = signal
          ? { method: 'GET', headers: { Range: rangeHeader }, signal }
          : { method: 'GET', headers: { Range: rangeHeader } };
        const rangeRes = await raceWithAbort(fetchImpl(url, rangeInit), signal);
        if (signal?.aborted) {
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

          const consumed = await raceWithAbort(consumeResponseBuffer(rangeRes), signal);
          if (signal?.aborted) {
            entryAborted = true;
          } else {
            rangeInfo.receivedBytes = consumed.byteLength;
            rangeBytes = consumed.bytes;

            if (rangeInfo.status !== 206) {
              const msg = `GET ${safePath} (${rangeHeader}) returned HTTP ${rangeInfo.status} (expected 206 Partial Content)`;
              fileErrors.push(msg);
              if (rangeInfo.status === 404 || rangeInfo.status === 410) {
                recordFileMismatch(MISMATCH_CATEGORIES.MISSING, msg);
              } else if (rangeInfo.status === 416) {
                recordFileMismatch(MISMATCH_CATEGORIES.SIZE_MISMATCH, msg);
              }
            } else {
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
      aborted = true;
      break;
    }

    // 3. Optional SHA-256 content hash verification (never stores or returns raw bytes)
    if (
      verifyHashes &&
      entry.sha256 !== null &&
      !fileCategories.has(MISMATCH_CATEGORIES.MISSING)
    ) {
      let bytesToHash = null;
      try {
        if (entry.size === 0) {
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
            entryAborted = true;
          } else {
            const fullStatus = fullRes ? fullRes.status : null;
            const fullConsumed = await raceWithAbort(consumeResponseBuffer(fullRes), signal);
            if (signal?.aborted) {
              entryAborted = true;
            } else if (fullStatus === 404 || fullStatus === 410) {
              const msg = `GET ${safePath} (sha256) returned HTTP ${fullStatus} (expected 200)`;
              fileErrors.push(msg);
              recordFileMismatch(MISMATCH_CATEGORIES.MISSING, msg);
            } else if (fullStatus !== 200) {
              fileErrors.push(
                `GET ${safePath} (sha256) returned HTTP ${fullStatus} (expected 200)`,
              );
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

        if (!entryAborted && !signal?.aborted && bytesToHash !== null) {
          const rawDigest = await raceWithAbort(
            Promise.resolve(
              typeof hashFn === 'function' ? hashFn(bytesToHash) : computeSha256Hex(bytesToHash),
            ),
            signal,
          );
          if (signal?.aborted) {
            entryAborted = true;
          } else {
            const actualHex =
              typeof rawDigest === 'string' && SHA256_HEX_RE.test(rawDigest.trim())
                ? rawDigest.trim().toLowerCase()
                : null;
            const verified = actualHex !== null && actualHex === entry.sha256;
            hashInfo = {
              expected: entry.sha256,
              actual: actualHex,
              verified,
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
      }
    }

    // Drop any reference to raw file bytes before finalizing diagnostics.
    rangeBytes = null;

    if (entryAborted || signal?.aborted) {
      aborted = true;
      break;
    }

    for (const m of fileMismatches) {
      mismatchesByCategory[m.category] += 1;
      mismatches.push(m);
    }

    const fileElapsedMs = computeElapsedMs(fileStart, now());
    filesVerified += 1;
    bytesVerified += fileBytesVerified;

    const fileOk = fileErrors.length === 0;
    if (fileOk) {
      passed += 1;
    } else {
      failed += 1;
    }

    results.push({
      path: entry.path,
      url,
      expectedSize: entry.size,
      ok: fileOk,
      errors: fileErrors,
      head: headInfo,
      range: rangeInfo,
      hash: hashInfo,
      etag: rangeInfo.etag || headInfo.etag || null,
      coopCoep: entry.size > 0 ? headCoopCoep && rangeCoopCoep : headCoopCoep,
      elapsedMs: fileElapsedMs,
      bytesVerified: fileBytesVerified,
      mismatchCategories: Array.from(fileCategories),
    });
  }

  return finalize(manifestResult.ok, topErrors);
}
