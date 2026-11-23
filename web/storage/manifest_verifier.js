// Browser & Node streaming manifest (`data-manifest.json`) and bootset (`bootset.json`)
// schema validator, partial-download readiness calculator, and live HTTP Range/ETag verifier.
// Matches `docs/io_manifest.md`, `tools/src/main.rs`, `tools/src/rpf9.rs`, `tools/src/scan.rs`,
// and `server/serve_local.py`. Zero external dependencies.

export const MANIFEST_ABI_VERSION = 1;
export const DEFAULT_SAMPLE_COUNT = 5;
export const DEFAULT_RANGE_PROBE_BYTES = 1;

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
 *   }
 * }}
 */
export function validateDataManifest(manifestInput) {
  const errors = [];
  const warnings = [];
  const fileMap = new Map();
  const summary = emptyManifestSummary();

  let manifest = manifestInput;
  if (typeof manifestInput === 'string') {
    try {
      manifest = JSON.parse(manifestInput);
    } catch (err) {
      errors.push(`Manifest JSON parse error: ${err instanceof Error ? err.message : String(err)}`);
      return { ok: false, errors, warnings, fileMap, summary };
    }
  }

  if (!isPlainObject(manifest)) {
    errors.push('Manifest must be a JSON object');
    return { ok: false, errors, warnings, fileMap, summary };
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
    return { ok: false, errors, warnings, fileMap, summary };
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
      errors.push(
        `Manifest "total_bytes" (${summary.declaredTotalBytes}) does not match sum of file sizes (${sumAllValidSizes})`,
      );
    }
  }

  return {
    ok: errors.length === 0,
    errors,
    warnings,
    fileMap,
    summary,
  };
}

function resolveManifestFileMap(manifest) {
  if (manifest === null || manifest === undefined) {
    return null;
  }
  if (manifest instanceof Map) {
    return manifest;
  }
  if (isPlainObject(manifest) && manifest.fileMap instanceof Map) {
    return manifest.fileMap;
  }
  return validateDataManifest(manifest).fileMap;
}

/**
 * Validates a `bootset.json` prefetch range list (top-level array per `docs/io_manifest.md`
 * or `{ entries: [...] }` object wrapper supported by `io_worker.js`), optionally checking
 * bounds against a data manifest. Deduplicates exact duplicates, detects overlapping ranges,
 * and computes both raw and deduplicated/merged byte budgets.
 *
 * @param {unknown} bootsetInput
 * @param {unknown} [manifest=null]
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
 * }}
 */
export function validateBootset(bootsetInput, manifest = null) {
  const errors = [];
  const warnings = [];
  const entries = [];
  const deduplicatedEntries = [];
  const mergedRanges = [];
  const duplicates = [];
  const overlaps = [];
  const missingSet = new Set();
  const incompleteSet = new Set();

  let parsed = bootsetInput;
  if (typeof bootsetInput === 'string') {
    try {
      parsed = JSON.parse(bootsetInput);
    } catch (err) {
      errors.push(`Bootset JSON parse error: ${err instanceof Error ? err.message : String(err)}`);
      return {
        ok: false,
        errors,
        warnings,
        entries,
        deduplicatedEntries,
        mergedRanges,
        duplicates,
        overlaps,
        missingFiles: [],
        incompleteFiles: [],
        totalBytes: 0,
        uniqueBytes: 0,
        overlapBytes: 0,
      };
    }
  }

  let rawEntries = null;
  if (Array.isArray(parsed)) {
    rawEntries = parsed;
  } else if (isPlainObject(parsed) && Array.isArray(parsed.entries)) {
    rawEntries = parsed.entries;
  } else {
    errors.push('Bootset must be an array of range entries or an object with an "entries" array');
    return {
      ok: false,
      errors,
      warnings,
      entries,
      deduplicatedEntries,
      mergedRanges,
      duplicates,
      overlaps,
      missingFiles: [],
      incompleteFiles: [],
      totalBytes: 0,
      uniqueBytes: 0,
      overlapBytes: 0,
    };
  }

  const fileMap = resolveManifestFileMap(manifest);
  const seenExact = new Set();
  const rangesByPath = new Map();
  let totalBytes = 0;

  for (let i = 0; i < rawEntries.length; i += 1) {
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
        errors.push(`${label} references file missing from manifest: "${relPath}"`);
        continue;
      }
      if (manifestFile.incomplete) {
        incompleteSet.add(relPath);
        warnings.push(`${label} references incomplete manifest file: "${relPath}"`);
      }
      if (end > manifestFile.size) {
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
  let uniqueBytes = 0;
  const sortedPaths = Array.from(rangesByPath.keys()).sort(compareAscii);

  for (const relPath of sortedPaths) {
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

  const missingFiles = Array.from(missingSet).sort(compareAscii);
  const incompleteFiles = Array.from(incompleteSet).sort(compareAscii);
  const overlapBytes = Math.max(0, totalBytes - uniqueBytes);

  return {
    ok: errors.length === 0,
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
  };
}

/**
 * Computes a structured streaming readiness summary from a manifest and optional bootset,
 * suitable for the browser HUD and pre-boot diagnostics.
 *
 * @param {unknown} manifest
 * @param {unknown} [bootset=null]
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
 * }}
 */
export function computeStreamingReadiness(manifest, bootset = null) {
  const manifestResult =
    isPlainObject(manifest) &&
    manifest.fileMap instanceof Map &&
    isPlainObject(manifest.summary) &&
    Array.isArray(manifest.errors) &&
    Array.isArray(manifest.warnings)
      ? manifest
      : validateDataManifest(manifest);

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

  let bootsetValid = true;
  let bootsetReady = manifestResult.ok && completeFiles > 0;
  let bootsetBytes = 0;
  let bootsetRawBytes = 0;
  let missingBootsetFiles = [];
  let incompleteBootsetFiles = [];

  if (bootset !== null && bootset !== undefined) {
    const bootsetResult = validateBootset(bootset, manifestResult);
    bootsetValid = bootsetResult.ok;
    bootsetBytes = bootsetResult.uniqueBytes;
    bootsetRawBytes = bootsetResult.totalBytes;
    missingBootsetFiles = bootsetResult.missingFiles;
    incompleteBootsetFiles = bootsetResult.incompleteFiles;

    for (const msg of bootsetResult.errors) {
      issues.push({ severity: 'error', source: 'bootset', message: msg });
    }
    for (const msg of bootsetResult.warnings) {
      issues.push({ severity: 'warning', source: 'bootset', message: msg });
    }

    bootsetReady =
      manifestResult.ok &&
      bootsetResult.ok &&
      missingBootsetFiles.length === 0 &&
      incompleteBootsetFiles.length === 0;
  }

  return {
    manifestValid: manifestResult.ok,
    bootsetValid,
    totalFiles,
    completeFiles,
    incompleteFiles,
    totalBytes,
    readyBytes,
    incompleteBytes,
    readinessPct,
    bootsetReady,
    bootsetBytes,
    bootsetRawBytes,
    missingBootsetFiles,
    incompleteBootsetFiles,
    issues,
  };
}

async function consumeResponseBytes(res) {
  if (!res) return null;
  if (typeof res.arrayBuffer === 'function') {
    const buf = await res.arrayBuffer();
    return buf ? buf.byteLength : 0;
  }
  if (res.body && typeof res.body.cancel === 'function') {
    try {
      await res.body.cancel();
    } catch {
      // Ignore stream cancellation errors.
    }
  }
  return null;
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
 * response byte count, and `ETag` presence/consistency.
 *
 * @param {object} options
 * @param {typeof fetch} [options.fetchImpl] Injectable fetch implementation (defaults to globalThis.fetch).
 * @param {string} [options.dataBase='/data/'] Base URL serving manifest files.
 * @param {unknown} options.manifest Raw manifest object, JSON string, or validateDataManifest() result.
 * @param {number} [options.sampleCount=5] Maximum number of complete files to probe.
 * @param {number} [options.probeBytes=1] Number of prefix bytes to request in the Range probe (1 -> bytes=0-0).
 * @param {boolean} [options.checkCoopCoep=false] When true, fail probes missing COOP/COEP headers.
 * @returns {Promise<{
 *   ok: boolean,
 *   probed: number,
 *   passed: number,
 *   failed: number,
 *   skippedIncomplete: number,
 *   errors: string[],
 *   results: Array<object>,
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
  } = options || {};

  if (typeof fetchImpl !== 'function') {
    throw new TypeError('verifyLiveEndpoints requires a fetchImpl function');
  }

  const manifestResult =
    isPlainObject(manifest) &&
    manifest.fileMap instanceof Map &&
    isPlainObject(manifest.summary) &&
    Array.isArray(manifest.errors)
      ? manifest
      : validateDataManifest(manifest);

  const topErrors = [...manifestResult.errors];
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
  let skippedIncomplete = 0;
  for (const entry of manifestResult.fileMap.values()) {
    if (entry.incomplete) {
      skippedIncomplete += 1;
    } else {
      completeEntries.push(entry);
    }
  }

  const selected = completeEntries.slice(0, maxSamples);
  const results = [];
  let passed = 0;
  let failed = 0;

  for (const entry of selected) {
    const url = `${base}${encodeRelPathForUrl(entry.path)}`;
    const fileErrors = [];

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

    let headCoopCoep = false;
    let rangeCoopCoep = false;

    // 1. HEAD probe
    try {
      const headRes = await fetchImpl(url, { method: 'HEAD' });
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
        fileErrors.push(`HEAD ${entry.path} returned HTTP ${headInfo.status} (expected 200)`);
      } else {
        if (!headInfo.acceptRanges || !/\bbytes\b/i.test(headInfo.acceptRanges)) {
          fileErrors.push(
            `HEAD ${entry.path} missing Accept-Ranges: bytes (got ${JSON.stringify(headInfo.acceptRanges)})`,
          );
        }
        if (headInfo.contentLength !== null && headInfo.contentLength !== entry.size) {
          fileErrors.push(
            `HEAD ${entry.path} Content-Length (${headInfo.contentLength}) does not match manifest size (${entry.size})`,
          );
        }
        if (!headInfo.etag || headInfo.etag.trim().length === 0) {
          fileErrors.push(`HEAD ${entry.path} missing ETag header`);
        } else if (entry.etag && headInfo.etag.trim() !== entry.etag) {
          fileErrors.push(
            `HEAD ${entry.path} ETag (${headInfo.etag.trim()}) does not match manifest ETag (${entry.etag})`,
          );
        }
        if (checkCoopCoep && !headCoopCoep) {
          fileErrors.push(`HEAD ${entry.path} missing required COOP/COEP headers`);
        }
      }
    } catch (err) {
      fileErrors.push(
        `HEAD ${entry.path} request failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    }

    // 2. Range GET probe (only when file has at least 1 byte; 0-byte files cannot satisfy bytes=0-0)
    if (entry.size > 0) {
      const endByte = Math.min(entry.size - 1, requestedProbeBytes - 1);
      const expectedLength = endByte + 1;
      const rangeHeader = `bytes=0-${endByte}`;
      rangeInfo.requested = rangeHeader;

      try {
        const rangeRes = await fetchImpl(url, {
          method: 'GET',
          headers: { Range: rangeHeader },
        });
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

        rangeInfo.receivedBytes = await consumeResponseBytes(rangeRes);

        if (rangeInfo.status !== 206) {
          fileErrors.push(
            `GET ${entry.path} (${rangeHeader}) returned HTTP ${rangeInfo.status} (expected 206 Partial Content)`,
          );
        } else {
          if (!rangeInfo.acceptRanges || !/\bbytes\b/i.test(rangeInfo.acceptRanges)) {
            fileErrors.push(
              `GET ${entry.path} (${rangeHeader}) missing Accept-Ranges: bytes (got ${JSON.stringify(rangeInfo.acceptRanges)})`,
            );
          }

          const parsedCr = parseContentRange(rangeInfo.contentRange);
          rangeInfo.parsedContentRange = parsedCr;
          if (!parsedCr) {
            fileErrors.push(
              `GET ${entry.path} (${rangeHeader}) returned missing or invalid Content-Range: ${JSON.stringify(rangeInfo.contentRange)}`,
            );
          } else {
            if (parsedCr.start !== 0 || parsedCr.end !== endByte) {
              fileErrors.push(
                `GET ${entry.path} (${rangeHeader}) Content-Range span ${parsedCr.start}-${parsedCr.end} does not match requested 0-${endByte}`,
              );
            }
            if (parsedCr.total !== entry.size) {
              fileErrors.push(
                `GET ${entry.path} (${rangeHeader}) Content-Range total (${parsedCr.total}) does not match manifest size (${entry.size})`,
              );
            }
          }

          if (rangeInfo.contentLength !== null && rangeInfo.contentLength !== expectedLength) {
            fileErrors.push(
              `GET ${entry.path} (${rangeHeader}) Content-Length (${rangeInfo.contentLength}) does not match expected ${expectedLength}`,
            );
          }

          if (rangeInfo.receivedBytes !== null && rangeInfo.receivedBytes !== expectedLength) {
            fileErrors.push(
              `GET ${entry.path} (${rangeHeader}) returned ${rangeInfo.receivedBytes} bytes (expected ${expectedLength})`,
            );
          }

          if (!rangeInfo.etag || rangeInfo.etag.trim().length === 0) {
            fileErrors.push(`GET ${entry.path} (${rangeHeader}) missing ETag header`);
          } else {
            if (headInfo.etag && rangeInfo.etag.trim() !== headInfo.etag.trim()) {
              fileErrors.push(
                `GET ${entry.path} ETag (${rangeInfo.etag.trim()}) differs from HEAD ETag (${headInfo.etag.trim()})`,
              );
            }
            if (entry.etag && rangeInfo.etag.trim() !== entry.etag) {
              fileErrors.push(
                `GET ${entry.path} ETag (${rangeInfo.etag.trim()}) does not match manifest ETag (${entry.etag})`,
              );
            }
          }

          if (checkCoopCoep && !rangeCoopCoep) {
            fileErrors.push(`GET ${entry.path} (${rangeHeader}) missing required COOP/COEP headers`);
          }
        }
      } catch (err) {
        fileErrors.push(
          `GET ${entry.path} (${rangeHeader}) request failed: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    }

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
      etag: rangeInfo.etag || headInfo.etag || null,
      coopCoep: entry.size > 0 ? headCoopCoep && rangeCoopCoep : headCoopCoep,
    });
  }

  return {
    ok: manifestResult.ok && failed === 0,
    probed: results.length,
    passed,
    failed,
    skippedIncomplete,
    errors: topErrors,
    results,
  };
}
