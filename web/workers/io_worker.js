'use strict';

// HTTP Range (206 Partial Content) VFS worker + OPFS/LRU RangeCache + IoPriorityScheduler
// + lazy Rpf9Vfs / RpfVfsMountTable (rpf://<archivePath>/<innerPath>) + SAB HttpfsMailbox.
(function initPlayGta6IoWorker() {
  const RPF_URI_PREFIX = 'rpf://';

  const FALLBACK_IO_PRIORITY = Object.freeze({
    CRITICAL: 0,
    HIGH: 1,
    NORMAL: 2,
    PREFETCH: 3,
  });

  function loadScriptIsolated(url) {
    // Classic Web Worker fallback when two sibling UMD scripts share a top-level lexical binding
    // (e.g. `const exportedApi` in range_cache.js and rpf9_vfs.js). Evaluating in a Function scope
    // isolates top-level const bindings while still attaching the global PlayGta6* namespace.
    if (typeof XMLHttpRequest !== 'function') {
      return false;
    }
    try {
      const xhr = new XMLHttpRequest();
      xhr.open('GET', url, false);
      xhr.send(null);
      if (xhr.status >= 200 && xhr.status < 300 && typeof xhr.responseText === 'string') {
        const exec = new Function(xhr.responseText);
        exec();
        return true;
      }
    } catch {
      // Ignore; module remains optional or can be injected programmatically.
    }
    return false;
  }

  function importWorkerScript(url, globalKey, required) {
    if (typeof globalThis !== 'undefined' && globalThis[globalKey]) {
      return;
    }
    if (typeof importScripts !== 'function') {
      return;
    }
    try {
      importScripts(url);
    } catch (err) {
      if (!loadScriptIsolated(url) && required) {
        throw err;
      }
    }
  }

  if (typeof importScripts === 'function') {
    importWorkerScript('./range_cache.js', 'PlayGta6RangeCache', true);
    importWorkerScript('./httpfs_mailbox.js', 'PlayGta6HttpfsMailbox', true);
    importWorkerScript('./io_scheduler.js', 'PlayGta6IoScheduler', false);
    importWorkerScript('./rpf9_vfs.js', 'PlayGta6Rpf9Vfs', false);
  }

  function tryRequire(modPath) {
    if (typeof require !== 'function') {
      return null;
    }
    try {
      return require(modPath);
    } catch {
      return null;
    }
  }

  function resolveRangeCacheModule(override) {
    return (
      override ||
      (typeof globalThis !== 'undefined' && globalThis.PlayGta6RangeCache) ||
      tryRequire('./range_cache.js')
    );
  }

  function resolveHttpfsMailboxModule(override) {
    return (
      override ||
      (typeof globalThis !== 'undefined' && globalThis.PlayGta6HttpfsMailbox) ||
      tryRequire('./httpfs_mailbox.js')
    );
  }

  function resolveIoSchedulerModule(override) {
    return (
      override ||
      (typeof globalThis !== 'undefined' && globalThis.PlayGta6IoScheduler) ||
      tryRequire('./io_scheduler.js')
    );
  }

  function resolveRpf9VfsModule(override) {
    return (
      override ||
      (typeof globalThis !== 'undefined' && globalThis.PlayGta6Rpf9Vfs) ||
      tryRequire('./rpf9_vfs.js')
    );
  }

  const defaultIoSchedulerMod = resolveIoSchedulerModule();
  const IO_PRIORITY =
    (defaultIoSchedulerMod && defaultIoSchedulerMod.IO_PRIORITY) || FALLBACK_IO_PRIORITY;

  const utf8Decoder = new TextDecoder('utf-8', { fatal: false });
  const utf8Encoder = new TextEncoder();

  const MOUNT_CACHE_PREFIX = '__pg6_mount__/';
  const DUMP_VIRTUAL_PREFIX = '/pg6_dump/';
  const DEFAULT_DUMP_ENDPOINT = '/__pg6_dump/';
  const DEFAULT_DUMP_MAX_BUFFER_BYTES = 16 * 1024 * 1024; // 16 MiB bounded per-handle dump buffer cap
  const DEFAULT_DUMP_FLUSH_CHUNK_BYTES = 1024 * 1024; // 1 MiB chunk size for sequential dump POSTs
  const HTTPFS_OP_WRITE = 4;
  const HTTPFS_OPEN_FLAG_WRITE = 1n;
  const HTTPFS_OPEN_FLAG_APPEND = 2n;

  const GAME_PACKAGE_EXT_RE = /\.(?:rpf|ttpf|dll|png|xvd|xvc|bin|pri|bk2|bik)$/i;
  const IMMUTABLE_HTTPFS_EXT_RE = /\.(?:rpf|pgim|dll|ttpf|pack|reflect\.json|bk2|bik)$/i;
  const VIDEO_STREAM_EXT_RE = /\.(?:bk2|bik)$/i;

  const DEFAULT_IMMUTABLE_CHUNK_SIZE = 64 * 1024; // 64 KiB aligned chunks
  const DEFAULT_IMMUTABLE_MAX_BYTES = 64 * 1024 * 1024; // 64 MiB RAM cap
  const DEFAULT_WHOLE_FILE_MAX_BYTES = 256 * 1024 * 1024; // 256 MiB safe cap for eager .pgim / manifest buffering
  const DEFAULT_SLAB_SIZE = 2 * 1024 * 1024; // 2 MiB aligned read-ahead slabs
  const DEFAULT_SLAB_MAX_BYTES = 32 * 1024 * 1024; // 32 MiB bounded LRU cap for read-ahead slabs
  const MAX_SLAB_SIZE = 16 * 1024 * 1024; // 16 MiB upper safety cap per slab
  const MAX_SLAB_CACHE_BYTES = 128 * 1024 * 1024; // 128 MiB upper safety cap for worker slab RAM
  const DEFAULT_SLAB_PREFETCH_AHEAD = 0; // Opt-in/tunable sequential read-ahead slab count
  const MAX_SLAB_PREFETCH_AHEAD = 8;
  const DEFAULT_MAX_CONCURRENT_SLAB_FETCHES = 4;
  const MAX_CONCURRENT_SLAB_FETCHES = 16;
  const DEFAULT_SEQUENTIAL_THRESHOLD_READS = 2;
  const MAX_STREAM_TRACKER_ENTRIES = 64;
  const DEFAULT_NEGATIVE_CACHE_TTL_MS = 30 * 1000; // 30s bounded TTL for missing optional files
  const DEFAULT_NEGATIVE_CACHE_MAX_ENTRIES = 512;
  const DEFAULT_METADATA_MAX_ENTRIES = 4096;
  const MAX_COALESCED_FETCH_BYTES = 4 * 1024 * 1024; // 4 MiB cap per single HTTP 206 fetch

  const SLAB_TUNING_PROFILES = Object.freeze({
    default: Object.freeze({
      slabSize: DEFAULT_SLAB_SIZE,
      slabMaxBytes: DEFAULT_SLAB_MAX_BYTES,
      prefetchAheadSlabs: DEFAULT_SLAB_PREFETCH_AHEAD,
      maxConcurrentSlabFetches: DEFAULT_MAX_CONCURRENT_SLAB_FETCHES,
      sequentialThresholdReads: DEFAULT_SEQUENTIAL_THRESHOLD_READS,
    }),
    video4k: Object.freeze({
      slabSize: 4 * 1024 * 1024, // 4 MiB slabs for smooth 4K Bink2 sequential reads
      slabMaxBytes: 48 * 1024 * 1024, // 48 MiB bounded LRU cap (12 slabs)
      prefetchAheadSlabs: 2, // Bounded 2-slab sequential read-ahead
      maxConcurrentSlabFetches: 3, // Leaves HTTP/1.1 connection headroom for live dump POSTs
      sequentialThresholdReads: 2,
    }),
    'low-memory': Object.freeze({
      slabSize: 1024 * 1024, // 1 MiB slabs
      slabMaxBytes: 16 * 1024 * 1024, // 16 MiB cap
      prefetchAheadSlabs: 1,
      maxConcurrentSlabFetches: 2,
      sequentialThresholdReads: 2,
    }),
  });

  const VIDEO_STREAM_SLAB_PRESET = SLAB_TUNING_PROFILES.video4k;

  function isVideoStreamPath(cleanPath) {
    if (typeof cleanPath !== 'string' || cleanPath.length === 0) {
      return false;
    }
    return VIDEO_STREAM_EXT_RE.test(cleanPath);
  }

  function isWholeFileBufferedPath(cleanPath) {
    if (typeof cleanPath !== 'string' || cleanPath.length === 0) {
      return false;
    }
    const lower = cleanPath.toLowerCase();
    const base = basenameOfPath(lower);
    return (
      base === 'program-manifest.json' ||
      base === 'boot-options.json' ||
      lower.endsWith('.pgim') ||
      lower.endsWith('.reflect.json')
    );
  }

  function isImmutableHttpfsPath(cleanPath) {
    if (typeof cleanPath !== 'string' || cleanPath.length === 0) {
      return false;
    }
    const base = basenameOfPath(cleanPath).toLowerCase();
    if (base === 'microsoftgame.config') {
      return true;
    }
    return IMMUTABLE_HTTPFS_EXT_RE.test(cleanPath);
  }

  function normalizeDataBase(base) {
    const raw = typeof base === 'string' && base.length > 0 ? base : '/data/';
    return raw.endsWith('/') ? raw : `${raw}/`;
  }

  function normalizeOptionalBase(base) {
    if (typeof base !== 'string') {
      return null;
    }
    const trimmed = base.trim();
    if (!trimmed) {
      return null;
    }
    return trimmed.endsWith('/') ? trimmed : `${trimmed}/`;
  }

  function basenameOfPath(posixPath) {
    const slash = posixPath.lastIndexOf('/');
    return slash < 0 ? posixPath : posixPath.slice(slash + 1);
  }

  function hasTraversalOrNulSegment(pathStr) {
    if (pathStr.includes('\0')) {
      return true;
    }
    const parts = pathStr.replace(/\\/g, '/').split('/');
    for (let i = 0; i < parts.length; i += 1) {
      const seg = parts[i];
      if (seg === '..') {
        return true;
      }
      if (seg.includes('%')) {
        try {
          const decoded = decodeURIComponent(seg);
          if (
            decoded === '..' ||
            decoded.includes('\0') ||
            decoded.includes('/') ||
            decoded.includes('\\')
          ) {
            return true;
          }
        } catch {
          // Ignore malformed percent-escape sequences.
        }
      }
    }
    return false;
  }

  function isDataPinnedPath(cleanPath) {
    const base = basenameOfPath(cleanPath).toLowerCase();
    return base === 'data-manifest.json' || base === 'bootset.json';
  }

  function isImplicitBootPath(cleanPath) {
    const lower = cleanPath.toLowerCase();
    return basenameOfPath(lower) === 'program-manifest.json' || lower.endsWith('.pgim');
  }

  function isGamePackagePath(cleanPath) {
    const base = basenameOfPath(cleanPath).toLowerCase();
    if (base === 'microsoftgame.config') {
      return true;
    }
    return GAME_PACKAGE_EXT_RE.test(cleanPath);
  }

  function hasMalformedPercentEscape(str) {
    if (typeof str !== 'string' || !str.includes('%')) {
      return false;
    }
    for (let i = 0; i < str.length; i += 1) {
      if (str[i] === '%') {
        const hex = str.slice(i + 1, i + 3);
        if (hex.length < 2 || !/^[0-9A-Fa-f]{2}$/.test(hex)) {
          return true;
        }
        i += 2;
      }
    }
    return false;
  }

  function normalizeDumpEndpoint(endpoint) {
    if (typeof endpoint !== 'string') {
      return null;
    }
    const trimmed = endpoint.trim();
    if (!trimmed || trimmed.includes('\0') || /[\x00-\x1f\x7f]/.test(trimmed)) {
      return null;
    }
    // Strip scheme + authority when validating path segments so URLs like
    // `http://127.0.0.1:8080/__pg6_dump/` pass cleanly while `..` segments are rejected.
    const pathPart = trimmed.replace(/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\/[^/]*/, '');
    if (hasTraversalOrNulSegment(pathPart)) {
      return null;
    }
    return trimmed.endsWith('/') ? trimmed : `${trimmed}/`;
  }

  function resolveDumpConfig(options = {}) {
    const opts = options && typeof options === 'object' ? options : {};
    if (opts.allowDumpWrites === false) {
      return { allowDumpWrites: false, dumpEndpoint: null };
    }
    if (opts.dumpEndpoint !== undefined && opts.dumpEndpoint !== null) {
      const normalized = normalizeDumpEndpoint(opts.dumpEndpoint);
      if (!normalized) {
        return { allowDumpWrites: false, dumpEndpoint: null };
      }
      return { allowDumpWrites: true, dumpEndpoint: normalized };
    }
    if (opts.allowDumpWrites === true) {
      return { allowDumpWrites: true, dumpEndpoint: DEFAULT_DUMP_ENDPOINT };
    }
    return { allowDumpWrites: false, dumpEndpoint: null };
  }

  function parseDumpVirtualPath(rawPath) {
    if (typeof rawPath !== 'string' || rawPath.length === 0) {
      return {
        ok: false,
        errno: -22,
        code: 'INVALID_PATH',
        error: 'Dump path must be a non-empty string',
      };
    }
    if (
      rawPath.includes('\0') ||
      hasTraversalOrNulSegment(rawPath) ||
      hasMalformedPercentEscape(rawPath)
    ) {
      return {
        ok: false,
        errno: -22,
        code: 'PATH_TRAVERSAL',
        error: 'path traversal blocked',
      };
    }

    // Reject URI schemes (e.g. rpf://, http://), Win32 device prefixes (\\?\), and drive prefixes (G:\).
    const slashNormalized = rawPath.replace(/\\/g, '/');
    if (
      /^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(slashNormalized) ||
      /^\/\/[?.]\//.test(slashNormalized) ||
      /^[A-Za-z]:/.test(slashNormalized)
    ) {
      return {
        ok: false,
        errno: -30,
        code: 'READ_ONLY',
        error: 'path is read-only (outside /pg6_dump/)',
      };
    }

    let s = slashNormalized;
    while (s.startsWith('/') || s.startsWith('./')) {
      if (s.startsWith('/')) {
        s = s.slice(1);
      } else {
        s = s.slice(2);
      }
    }

    const rawSegments = s.split('/');
    const cleanSegments = [];
    for (let i = 0; i < rawSegments.length; i += 1) {
      const seg = rawSegments[i];
      if (!seg || seg === '.') {
        continue;
      }
      if (seg === '..') {
        return {
          ok: false,
          errno: -22,
          code: 'PATH_TRAVERSAL',
          error: 'path traversal blocked',
        };
      }
      cleanSegments.push(seg);
    }

    if (cleanSegments.length === 0) {
      return {
        ok: false,
        errno: -22,
        code: 'INVALID_PATH',
        error: 'invalid dump path',
      };
    }

    if (cleanSegments[0].toLowerCase() !== 'pg6_dump') {
      return {
        ok: false,
        errno: -30,
        code: 'READ_ONLY',
        error: 'path is read-only (outside /pg6_dump/)',
      };
    }

    const relSegments = cleanSegments.slice(1);
    if (relSegments.length === 0) {
      return {
        ok: false,
        errno: -22,
        code: 'INVALID_PATH',
        error: 'Dump path must include a relative filename under /pg6_dump/',
      };
    }

    for (let i = 0; i < relSegments.length; i += 1) {
      const seg = relSegments[i];
      if (/[\x00-\x1f\x7f:*?"<>|]/.test(seg)) {
        return {
          ok: false,
          errno: -22,
          code: 'INVALID_PATH',
          error: `Invalid character in dump path segment: ${seg}`,
        };
      }
    }

    const relPath = relSegments.join('/');
    const encodedRelPath = relSegments.map((seg) => encodeURIComponent(seg)).join('/');
    return {
      ok: true,
      relPath,
      encodedRelPath,
    };
  }

  function isDumpVirtualPath(rawPath) {
    return parseDumpVirtualPath(rawPath).ok;
  }

  /**
   * Strictly validate a virtual `/pg6_dump/<relpath>` write target against opt-in dump config.
   * Rejects path traversal (`-22`), paths outside `/pg6_dump/` (`-30` EROFS), and disabled writes (`-30`).
   */
  function resolveDumpWriteRoute(rawPath, options = {}) {
    const parsed = parseDumpVirtualPath(rawPath);
    if (!parsed.ok) {
      return parsed;
    }

    const cfg = resolveDumpConfig(options);
    if (!cfg.allowDumpWrites || !cfg.dumpEndpoint) {
      return {
        ok: false,
        errno: -30,
        code: 'DUMP_WRITES_DISABLED',
        error: 'dump writes are disabled (opt-in required)',
      };
    }

    const url = `${cfg.dumpEndpoint}${parsed.encodedRelPath}`;
    return {
      ok: true,
      mount: 'dump',
      base: cfg.dumpEndpoint,
      relPath: parsed.relPath,
      encodedRelPath: parsed.encodedRelPath,
      url,
      appendUrl: `${url}?append=1`,
    };
  }

  function toUint8ArrayPayload(data) {
    if (data instanceof Uint8Array) {
      return data.byteOffset === 0 && data.byteLength === data.buffer.byteLength
        ? data
        : data.slice();
    }
    if (data instanceof ArrayBuffer) {
      return new Uint8Array(data).slice();
    }
    if (ArrayBuffer.isView(data)) {
      return new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
    }
    if (typeof data === 'string') {
      return utf8Encoder.encode(data);
    }
    return null;
  }

  function createSlabCacheTuningProfile(profileOrOptions = 'video4k') {
    const opts =
      typeof profileOrOptions === 'string'
        ? { profile: profileOrOptions }
        : profileOrOptions && typeof profileOrOptions === 'object'
          ? profileOrOptions
          : {};

    const rawProfile =
      typeof opts.profile === 'string'
        ? opts.profile.trim().toLowerCase()
        : typeof opts.slabProfile === 'string'
          ? opts.slabProfile.trim().toLowerCase()
          : null;

    let base = SLAB_TUNING_PROFILES.default;
    if (
      rawProfile === 'video4k' ||
      rawProfile === 'bink2_4k' ||
      rawProfile === 'bink2' ||
      rawProfile === 'video-streaming'
    ) {
      base = SLAB_TUNING_PROFILES.video4k;
    } else if (rawProfile === 'low-memory' || rawProfile === 'low_memory') {
      base = SLAB_TUNING_PROFILES['low-memory'];
    }

    const rawSlabSize =
      opts.slabSize !== undefined ? opts.slabSize : base.slabSize;
    const slabSize =
      Number.isInteger(rawSlabSize) && rawSlabSize >= 0
        ? Math.min(rawSlabSize, MAX_SLAB_SIZE)
        : base.slabSize;

    const rawSlabMaxBytes =
      opts.slabMaxBytes !== undefined ? opts.slabMaxBytes : base.slabMaxBytes;
    const slabMaxBytes =
      Number.isFinite(rawSlabMaxBytes) && rawSlabMaxBytes >= 0
        ? Math.min(Math.floor(rawSlabMaxBytes), MAX_SLAB_CACHE_BYTES)
        : base.slabMaxBytes;

    const rawPrefetch =
      opts.prefetchAheadSlabs !== undefined
        ? opts.prefetchAheadSlabs
        : base.prefetchAheadSlabs;
    const maxSafePrefetch =
      slabSize > 0 && slabMaxBytes >= slabSize * 2
        ? Math.min(MAX_SLAB_PREFETCH_AHEAD, Math.floor(slabMaxBytes / slabSize) - 1)
        : 0;
    const prefetchAheadSlabs =
      Number.isInteger(rawPrefetch) && rawPrefetch >= 0
        ? Math.min(rawPrefetch, maxSafePrefetch)
        : Math.min(base.prefetchAheadSlabs, maxSafePrefetch);

    const rawConc =
      opts.maxConcurrentSlabFetches !== undefined
        ? opts.maxConcurrentSlabFetches
        : base.maxConcurrentSlabFetches;
    const maxConcurrentSlabFetches =
      Number.isInteger(rawConc) && rawConc >= 1
        ? Math.min(rawConc, MAX_CONCURRENT_SLAB_FETCHES)
        : base.maxConcurrentSlabFetches;

    const rawSeqThresh =
      opts.sequentialThresholdReads !== undefined
        ? opts.sequentialThresholdReads
        : base.sequentialThresholdReads;
    const sequentialThresholdReads =
      Number.isInteger(rawSeqThresh) && rawSeqThresh >= 1
        ? Math.min(rawSeqThresh, 32)
        : base.sequentialThresholdReads;

    return {
      slabSize,
      slabMaxBytes,
      prefetchAheadSlabs,
      maxConcurrentSlabFetches,
      sequentialThresholdReads,
    };
  }

  /**
   * Normalize Xbox/Win32/POSIX HTTPFS paths and route them against `bootBase`, `gameBase`,
   * `shadersBase`, or `dataBase` while preserving `rpf://...` virtual URIs and blocking
   * `..` / NUL traversal.
   */
  function resolveHttpfsRoute(rawPath, options = {}) {
    if (typeof rawPath !== 'string' || rawPath.length === 0) {
      return { ok: false, error: 'HTTPFS path must be a non-empty string' };
    }
    if (rawPath.includes('\0')) {
      return { ok: false, error: 'path traversal blocked' };
    }

    const opts = options && typeof options === 'object' ? options : {};
    const dataBase = normalizeDataBase(opts.dataBase);
    const bootBase = normalizeOptionalBase(opts.bootBase);
    const gameBase = normalizeOptionalBase(opts.gameBase);
    const shadersBase = normalizeOptionalBase(opts.shadersBase);

    if (isRpfVirtualPath(rawPath)) {
      const rest = rawPath.slice(RPF_URI_PREFIX.length);
      if (hasTraversalOrNulSegment(rest)) {
        return { ok: false, error: 'path traversal blocked' };
      }
      return {
        ok: true,
        isRpf: true,
        mount: 'rpf',
        base: null,
        relPath: rawPath,
        url: rawPath,
      };
    }

    let s = rawPath.replace(/\\/g, '/');

    // Strip Win32 extended-length / device prefixes (e.g. `\\?\G:\...` or `\\.\G:\...`).
    if (/^\/\/[?.]\//.test(s)) {
      s = s.slice(4);
    }

    // Strip Xbox/Win32 single-letter drive prefixes (e.g. `G:/`, `T:/`, `D:/`, `C:/`, `G:`).
    let hadDrivePrefix = false;
    if (/^[A-Za-z]:/.test(s)) {
      hadDrivePrefix = true;
      s = s.slice(2);
    }

    // Strip leading `/` and `./` prefixes.
    while (s.startsWith('/') || s.startsWith('./')) {
      if (s.startsWith('/')) {
        s = s.slice(1);
      } else {
        s = s.slice(2);
      }
    }

    if (hasTraversalOrNulSegment(s)) {
      return { ok: false, error: 'path traversal blocked' };
    }

    const rawSegments = s.split('/');
    const cleanSegments = [];
    for (let i = 0; i < rawSegments.length; i += 1) {
      const seg = rawSegments[i];
      if (!seg || seg === '.') {
        continue;
      }
      cleanSegments.push(seg);
    }

    if (cleanSegments.length === 0) {
      return { ok: false, error: 'invalid HTTPFS path' };
    }

    const cleanPath = cleanSegments.join('/');
    const lower = cleanPath.toLowerCase();

    // 1. Data-pinned manifests always resolve against dataBase.
    if (isDataPinnedPath(cleanPath)) {
      return {
        ok: true,
        isRpf: false,
        mount: 'data',
        base: dataBase,
        relPath: cleanPath,
        url: `${dataBase}${cleanPath}`,
      };
    }

    // 2. Explicit `boot/...` prefix routes against bootBase (stripping `boot/`) when configured.
    if (bootBase && lower.startsWith('boot/')) {
      const stripped = cleanPath.slice('boot/'.length);
      if (!stripped) {
        return { ok: false, error: 'invalid HTTPFS path' };
      }
      return {
        ok: true,
        isRpf: false,
        mount: 'boot',
        base: bootBase,
        relPath: stripped,
        url: `${bootBase}${stripped}`,
      };
    }

    // 3. Explicit `game/...` mount prefix (without an Xbox/Win32 drive prefix) routes against
    //    gameBase (stripping leading `game/`) when configured.
    if (gameBase && !hadDrivePrefix && lower.startsWith('game/')) {
      const stripped = cleanPath.slice('game/'.length);
      if (!stripped) {
        return { ok: false, error: 'invalid HTTPFS path' };
      }
      return {
        ok: true,
        isRpf: false,
        mount: 'game',
        base: gameBase,
        relPath: stripped,
        url: `${gameBase}${stripped}`,
      };
    }

    // 3b. Explicit `shaders/...` mount prefix (without an Xbox/Win32 drive prefix) routes against
    //     shadersBase (stripping leading `shaders/`) when configured.
    if (shadersBase && !hadDrivePrefix && lower.startsWith('shaders/')) {
      const stripped = cleanPath.slice('shaders/'.length);
      if (!stripped) {
        return { ok: false, error: 'invalid HTTPFS path' };
      }
      return {
        ok: true,
        isRpf: false,
        mount: 'shaders',
        base: shadersBase,
        relPath: stripped,
        url: `${shadersBase}${stripped}`,
      };
    }

    // 4. Implicit boot files (`program-manifest.json`, `*.pgim`) route against bootBase when configured.
    if (bootBase && isImplicitBootPath(cleanPath)) {
      return {
        ok: true,
        isRpf: false,
        mount: 'boot',
        base: bootBase,
        relPath: cleanPath,
        url: `${bootBase}${cleanPath}`,
      };
    }

    // 5. Game package files route against gameBase when configured (falling back to dataBase otherwise).
    if (gameBase && isGamePackagePath(cleanPath)) {
      return {
        ok: true,
        isRpf: false,
        mount: 'game',
        base: gameBase,
        relPath: cleanPath,
        url: `${gameBase}${cleanPath}`,
      };
    }

    // 6. Default: resolve against dataBase.
    return {
      ok: true,
      isRpf: false,
      mount: 'data',
      base: dataBase,
      relPath: cleanPath,
      url: `${dataBase}${cleanPath}`,
    };
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
    for (let i = 0; i < parts.length; i += 1) {
      const part = parts[i];
      if (!part || part === '.' || part === '..') {
        throw new Error(`Invalid HTTPFS path segment in: ${relPath}`);
      }
    }
    return stripped;
  }

  function isRpfVirtualPath(pathStr) {
    return (
      typeof pathStr === 'string' &&
      pathStr.length >= RPF_URI_PREFIX.length &&
      pathStr.slice(0, RPF_URI_PREFIX.length).toLowerCase() === RPF_URI_PREFIX
    );
  }

  function createErrnoError(errno, message) {
    const err = new Error(message);
    err.errno = errno;
    return err;
  }

  function parseRpfVirtualPath(rawPath, options = {}) {
    if (!isRpfVirtualPath(rawPath)) {
      return null;
    }
    if (rawPath.includes('\0')) {
      throw createErrnoError(-22, `Invalid RPF virtual path (NUL byte): ${rawPath}`);
    }

    const rest = rawPath.slice(RPF_URI_PREFIX.length);
    if (!rest || rest.startsWith('/')) {
      throw createErrnoError(-22, `Invalid RPF virtual path: ${rawPath}`);
    }

    const rawSegments = rest.split('/');
    const allowEmptyInner = Boolean(options.allowEmptyInner);

    // Prefer the first segment ending in ".rpf" (case-insensitive) so both
    // `rpf://appdata0.rpf/config/settings.xml` and `rpf://update/update.rpf/common/data.xml`
    // split cleanly on the archive boundary.
    let archiveEndIdx = -1;
    for (let i = 0; i < rawSegments.length; i += 1) {
      if (/\.rpf$/i.test(rawSegments[i])) {
        archiveEndIdx = i;
        break;
      }
    }

    // Fallback: match against manifest paths if provided and no ".rpf" segment was found.
    const manifest = options.manifest || null;
    if (archiveEndIdx === -1 && manifest && Array.isArray(manifest.files)) {
      const manifestSet = new Set();
      for (let i = 0; i < manifest.files.length; i += 1) {
        const entry = manifest.files[i];
        if (entry && typeof entry.path === 'string') {
          try {
            manifestSet.add(normalizeRelPath(entry.path));
          } catch {
            // Ignore malformed manifest entries.
          }
        }
      }
      for (let i = rawSegments.length - 1; i >= 0; i -= 1) {
        const candidate = rawSegments.slice(0, i + 1).join('/');
        if (manifestSet.has(candidate)) {
          archiveEndIdx = i;
          break;
        }
      }
    }

    // Final fallback: first segment is archive path, remaining segments are inner path.
    if (archiveEndIdx === -1) {
      if (rawSegments.length >= 2) {
        archiveEndIdx = 0;
      } else if (allowEmptyInner && rawSegments.length === 1 && rawSegments[0]) {
        archiveEndIdx = 0;
      } else {
        throw createErrnoError(-22, `RPF virtual path missing inner path: ${rawPath}`);
      }
    }

    const archiveRaw = rawSegments.slice(0, archiveEndIdx + 1).join('/');
    let archivePath;
    try {
      archivePath = normalizeRelPath(archiveRaw);
    } catch {
      throw createErrnoError(-22, `Invalid RPF archive path in URI: ${rawPath}`);
    }

    const innerRaw = rawSegments.slice(archiveEndIdx + 1).join('/');
    const rpfMod = resolveRpf9VfsModule(options.rpf9VfsModule);
    let innerPath = '';
    if (innerRaw.length > 0) {
      if (rpfMod && typeof rpfMod.normalizeVirtualPath === 'function') {
        try {
          innerPath = rpfMod.normalizeVirtualPath(innerRaw).path;
        } catch {
          throw createErrnoError(-22, `Invalid RPF inner path in URI: ${rawPath}`);
        }
      } else {
        const parts = innerRaw.replace(/\\/g, '/').split('/');
        const clean = [];
        for (let i = 0; i < parts.length; i += 1) {
          const seg = parts[i];
          if (!seg || seg === '.') continue;
          if (seg === '..') {
            throw createErrnoError(-22, `Path traversal in RPF inner path: ${rawPath}`);
          }
          clean.push(seg);
        }
        innerPath = clean.join('/');
      }
    }

    if (!innerPath && !allowEmptyInner) {
      throw createErrnoError(-22, `RPF virtual path missing inner file path: ${rawPath}`);
    }

    const mountPath = innerPath ? `${archivePath}/${innerPath}` : archivePath;
    return {
      archivePath,
      innerPath,
      mountPath,
    };
  }

  function mapErrorToErrno(err, defaultErrno = -2) {
    if (!err || typeof err !== 'object') {
      return defaultErrno;
    }
    if (typeof err.errno === 'number' && Number.isInteger(err.errno) && err.errno < 0) {
      return err.errno;
    }
    if (err.code === 'ENCRYPTED_ARCHIVE' || err.name === 'EncryptedArchiveError') {
      return -13; // EACCES
    }
    if (err.code === 'PATH_NOT_FOUND') {
      return -2; // ENOENT
    }
    if (err.code === 'NOT_A_FILE') {
      return -21; // EISDIR
    }
    if (err.code === 'NOT_A_DIRECTORY') {
      return -20; // ENOTDIR
    }
    if (err.code === 'INVALID_PATH') {
      return -22; // EINVAL
    }
    if (err.status === 403 || err.statusCode === 403) {
      return -13; // EACCES
    }
    if (err.status === 404 || err.statusCode === 404) {
      return -2; // ENOENT
    }
    const msg = typeof err.message === 'string' ? err.message : '';
    if (/\bHTTP\s+403\b/i.test(msg)) {
      return -13; // EACCES
    }
    if (/\bHTTP\s+404\b/i.test(msg) || /\bnot found\b/i.test(msg)) {
      return -2; // ENOENT
    }
    if (err.name === 'RpfFormatError') {
      return -5; // EIO
    }
    return defaultErrno;
  }

  function getResponseHeader(headers, name) {
    if (!headers) {
      return null;
    }
    const lower = name.toLowerCase();
    if (typeof headers.get === 'function') {
      return headers.get(name) || headers.get(lower) || null;
    }
    if (typeof headers === 'object') {
      if (typeof headers[name] === 'string') return headers[name];
      if (typeof headers[lower] === 'string') return headers[lower];
    }
    return null;
  }

  function parseContentRangeHeader(header) {
    if (typeof header !== 'string') {
      return null;
    }
    const trimmed = header.trim();
    const unsatisfied = /^bytes\s+\*\/(\d+)$/i.exec(trimmed);
    if (unsatisfied) {
      const total = Number(unsatisfied[1]);
      if (!Number.isSafeInteger(total) || total < 0) {
        return null;
      }
      return { start: null, end: null, total, unsatisfied: true };
    }
    const match = /^bytes\s+(\d+)-(\d+)\/(\d+|\*)$/i.exec(trimmed);
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
      (total !== null && (!Number.isSafeInteger(total) || total < 0))
    ) {
      return null;
    }
    return { start, end, total, unsatisfied: false };
  }

  function resolveSlabConfig(options = {}) {
    const hasProfile = typeof options.slabProfile === 'string' && options.slabProfile.trim().length > 0;
    const profileDefaults = hasProfile
      ? createSlabCacheTuningProfile({ profile: options.slabProfile })
      : null;

    const explicitChunkOrMax =
      options.chunkSize !== undefined || options.maxBytes !== undefined;
    const slabExplicitlyConfigured =
      hasProfile ||
      options.slabSize !== undefined ||
      options.slabMaxBytes !== undefined ||
      options.slabCacheEnabled === true ||
      options.prefetchAheadSlabs !== undefined;

    const slabEnabled =
      options.slabCacheEnabled === false
        ? false
        : slabExplicitlyConfigured
          ? true
          : !explicitChunkOrMax;

    const rawSlabSize =
      options.slabSize !== undefined
        ? options.slabSize
        : profileDefaults
          ? profileDefaults.slabSize
          : DEFAULT_SLAB_SIZE;
    const slabSize = !slabEnabled
      ? 0
      : Number.isInteger(rawSlabSize) && rawSlabSize >= 0
        ? Math.min(rawSlabSize, MAX_SLAB_SIZE)
        : DEFAULT_SLAB_SIZE;

    const rawSlabMaxBytes =
      options.slabMaxBytes !== undefined
        ? options.slabMaxBytes
        : profileDefaults
          ? profileDefaults.slabMaxBytes
          : DEFAULT_SLAB_MAX_BYTES;
    const slabMaxBytes = !slabEnabled
      ? 0
      : Number.isFinite(rawSlabMaxBytes) && rawSlabMaxBytes >= 0
        ? Math.min(Math.floor(rawSlabMaxBytes), MAX_SLAB_CACHE_BYTES)
        : DEFAULT_SLAB_MAX_BYTES;

    const maxSafePrefetch =
      slabEnabled && slabSize > 0 && slabMaxBytes >= slabSize * 2
        ? Math.min(MAX_SLAB_PREFETCH_AHEAD, Math.floor(slabMaxBytes / slabSize) - 1)
        : 0;
    const rawPrefetch =
      options.prefetchAheadSlabs !== undefined
        ? options.prefetchAheadSlabs
        : profileDefaults
          ? profileDefaults.prefetchAheadSlabs
          : DEFAULT_SLAB_PREFETCH_AHEAD;
    const prefetchAheadSlabs = !slabEnabled
      ? 0
      : Number.isInteger(rawPrefetch) && rawPrefetch >= 0
        ? Math.min(rawPrefetch, maxSafePrefetch)
        : 0;

    const rawConc =
      options.maxConcurrentSlabFetches !== undefined
        ? options.maxConcurrentSlabFetches
        : profileDefaults
          ? profileDefaults.maxConcurrentSlabFetches
          : DEFAULT_MAX_CONCURRENT_SLAB_FETCHES;
    const maxConcurrentSlabFetches =
      Number.isInteger(rawConc) && rawConc >= 1
        ? Math.min(rawConc, MAX_CONCURRENT_SLAB_FETCHES)
        : DEFAULT_MAX_CONCURRENT_SLAB_FETCHES;

    const rawSeqThresh =
      options.sequentialThresholdReads !== undefined
        ? options.sequentialThresholdReads
        : profileDefaults
          ? profileDefaults.sequentialThresholdReads
          : DEFAULT_SEQUENTIAL_THRESHOLD_READS;
    const sequentialThresholdReads =
      Number.isInteger(rawSeqThresh) && rawSeqThresh >= 1
        ? Math.min(rawSeqThresh, 32)
        : DEFAULT_SEQUENTIAL_THRESHOLD_READS;

    return {
      slabSize,
      slabMaxBytes,
      prefetchAheadSlabs,
      maxConcurrentSlabFetches,
      sequentialThresholdReads,
    };
  }

  function emptyImmutableCacheStats(options = {}) {
    const {
      slabSize,
      slabMaxBytes,
      prefetchAheadSlabs,
      maxConcurrentSlabFetches,
    } = resolveSlabConfig(options);
    return {
      hits: 0,
      misses: 0,
      slabCacheHits: 0,
      slabCacheMisses: 0,
      slabEvictions: 0,
      slabNetworkFetches: 0,
      slabBytesCached: 0,
      slabMaxBytes,
      slabSize,
      slabEntries: 0,
      slabPrefetchesScheduled: 0,
      slabPrefetchesCompleted: 0,
      slabPrefetchErrors: 0,
      prefetchAheadSlabs,
      maxConcurrentSlabFetches,
      activeSlabFetches: 0,
      peakConcurrentSlabFetches: 0,
      metaHits: 0,
      metaMisses: 0,
      negativeHits: 0,
      negativeEntries: 0,
      inFlightCoalesced: 0,
      networkFetches: 0,
      evictions: 0,
      bytesCached: 0,
      maxBytes:
        Number.isFinite(options.maxBytes) && options.maxBytes >= 0
          ? Math.floor(options.maxBytes)
          : DEFAULT_IMMUTABLE_MAX_BYTES,
      chunkSize:
        Number.isInteger(options.chunkSize) && options.chunkSize > 0
          ? options.chunkSize
          : DEFAULT_IMMUTABLE_CHUNK_SIZE,
      entries: 0,
      filesIndexed: 0,
    };
  }

  /**
   * Bounded in-memory chunked LRU + sequential read-ahead slab LRU + file-length metadata
   * + short negative cache for immutable HTTPFS files (`*.rpf`, `*.pgim`, `*.dll`, `*.ttpf`,
   * `*.pack`, `*.reflect.json`, `*.bk2`, `*.bik`, `MicrosoftGame.Config`). Eliminates redundant
   * HTTP 206 round-trips across guest worker threads while keeping RAM strictly bounded and
   * returning 100% exact byte slices.
   */
  class ImmutableHttpfsCache {
    constructor(options = {}) {
      this.chunkSize =
        Number.isInteger(options.chunkSize) && options.chunkSize > 0
          ? options.chunkSize
          : DEFAULT_IMMUTABLE_CHUNK_SIZE;
      this.maxBytes =
        Number.isFinite(options.maxBytes) && options.maxBytes >= 0
          ? Math.floor(options.maxBytes)
          : DEFAULT_IMMUTABLE_MAX_BYTES;
      const slabCfg = resolveSlabConfig(options);
      this.slabSize = slabCfg.slabSize;
      this.slabMaxBytes = slabCfg.slabMaxBytes;
      this.prefetchAheadSlabs = slabCfg.prefetchAheadSlabs;
      this.maxConcurrentSlabFetches = slabCfg.maxConcurrentSlabFetches;
      this.sequentialThresholdReads = slabCfg.sequentialThresholdReads;
      this.negativeCacheTtlMs =
        Number.isFinite(options.negativeCacheTtlMs) && options.negativeCacheTtlMs >= 0
          ? Math.floor(options.negativeCacheTtlMs)
          : DEFAULT_NEGATIVE_CACHE_TTL_MS;
      this.negativeCacheMaxEntries =
        Number.isInteger(options.negativeCacheMaxEntries) && options.negativeCacheMaxEntries >= 0
          ? options.negativeCacheMaxEntries
          : DEFAULT_NEGATIVE_CACHE_MAX_ENTRIES;
      this.metadataMaxEntries =
        Number.isInteger(options.metadataMaxEntries) && options.metadataMaxEntries > 0
          ? options.metadataMaxEntries
          : DEFAULT_METADATA_MAX_ENTRIES;
      this.fetchImpl =
        typeof options.fetchImpl === 'function'
          ? options.fetchImpl
          : typeof fetch === 'function'
            ? fetch.bind(globalThis)
            : null;
      this._now = typeof options.now === 'function' ? options.now : () => Date.now();

      this._chunks = new Map(); // key (`${url}::${chunkIndex}`) -> owned Uint8Array (LRU order)
      this._slabs = new Map(); // key (`${url}::slab::${slabIndex}`) -> owned Uint8Array (LRU order)
      this._fileMeta = new Map(); // normalized url -> { size: number | null } (LRU order)
      this._negativeCache = new Map(); // normalized url -> { errno, status, expiresAt } (LRU order)
      this._inFlightChunks = new Map(); // chunkKey -> Promise<Uint8Array>
      this._inFlightSlabs = new Map(); // slabKey -> Promise<Uint8Array>
      this._inFlightOpen = new Map(); // url -> Promise<{ ok, size, errno, status }>
      this._streamTrackers = new Map(); // url -> { lastOffset, lastEndOffset, lastSlabIndex, sequentialCount }
      this._activeSlabFetches = 0;
      this._slabWaitQueue = []; // { resolve, priority } (demand=0 before prefetch=1)
      this._pendingPrefetches = new Set();
      this._bytesCached = 0;
      this._slabBytesCached = 0;

      this._stats = {
        chunkHits: 0,
        chunkMisses: 0,
        slabHits: 0,
        slabMisses: 0,
        slabEvictions: 0,
        slabNetworkFetches: 0,
        slabPrefetchesScheduled: 0,
        slabPrefetchesCompleted: 0,
        slabPrefetchErrors: 0,
        peakConcurrentSlabFetches: 0,
        metaHits: 0,
        metaMisses: 0,
        negativeHits: 0,
        inFlightCoalesced: 0,
        networkFetches: 0,
        evictions: 0,
      };
    }

    _chunkKey(url, chunkIndex) {
      return `${url}::${chunkIndex}`;
    }

    _slabKey(url, slabIndex) {
      return `${url}::slab::${slabIndex}`;
    }

    _touchMeta(url) {
      if (!this._fileMeta.has(url)) {
        return null;
      }
      const meta = this._fileMeta.get(url);
      this._fileMeta.delete(url);
      this._fileMeta.set(url, meta);
      return meta;
    }

    setFileMetadata(url, size) {
      if (typeof url !== 'string' || url.length === 0) {
        return;
      }
      const cleanSize = Number.isSafeInteger(size) && size >= 0 ? size : null;
      const existing = this._fileMeta.get(url);
      const nextSize = cleanSize !== null ? cleanSize : existing ? existing.size : null;
      if (existing) {
        this._fileMeta.delete(url);
      } else if (this._fileMeta.size >= this.metadataMaxEntries) {
        const oldestUrl = this._fileMeta.keys().next().value;
        if (oldestUrl !== undefined) {
          this._fileMeta.delete(oldestUrl);
        }
      }
      this._fileMeta.set(url, { size: nextSize });
      // A confirmed existing file clears any stale negative entry.
      this._negativeCache.delete(url);
    }

    getFileMetadata(url) {
      const meta = this._touchMeta(url);
      return meta ? { size: meta.size } : null;
    }

    getFileSize(url) {
      const meta = this._fileMeta.get(url);
      return meta && Number.isSafeInteger(meta.size) ? meta.size : null;
    }

    _getNegative(url) {
      const entry = this._negativeCache.get(url);
      if (!entry) {
        return null;
      }
      if (this.negativeCacheTtlMs <= 0 || this._now() >= entry.expiresAt) {
        this._negativeCache.delete(url);
        return null;
      }
      // Refresh LRU position on active negative hit.
      this._negativeCache.delete(url);
      this._negativeCache.set(url, entry);
      return entry;
    }

    _putNegative(url, errno, status) {
      if (this.negativeCacheTtlMs <= 0 || this.negativeCacheMaxEntries <= 0) {
        return;
      }
      if (this._negativeCache.has(url)) {
        this._negativeCache.delete(url);
      } else if (this._negativeCache.size >= this.negativeCacheMaxEntries) {
        const oldestUrl = this._negativeCache.keys().next().value;
        if (oldestUrl !== undefined) {
          this._negativeCache.delete(oldestUrl);
        }
      }
      this._negativeCache.set(url, {
        errno,
        status,
        expiresAt: this._now() + this.negativeCacheTtlMs,
      });
    }

    _touchChunk(key) {
      if (!this._chunks.has(key)) {
        return null;
      }
      const bytes = this._chunks.get(key);
      this._chunks.delete(key);
      this._chunks.set(key, bytes);
      return bytes;
    }

    _putChunk(key, bytes) {
      if (!(bytes instanceof Uint8Array) || bytes.byteLength === 0) {
        return;
      }
      // Always store a tightly-sized owned Uint8Array so evicting one chunk
      // never leaves a larger coalesced ArrayBuffer pinned in memory.
      const owned =
        bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
          ? bytes
          : bytes.slice();

      if (this._chunks.has(key)) {
        const existing = this._chunks.get(key);
        this._bytesCached -= existing.byteLength;
        this._chunks.delete(key);
      }

      if (this.maxBytes <= 0 || owned.byteLength > this.maxBytes) {
        return;
      }

      while (this._bytesCached + owned.byteLength > this.maxBytes && this._chunks.size > 0) {
        const oldestKey = this._chunks.keys().next().value;
        const oldestBytes = this._chunks.get(oldestKey);
        this._chunks.delete(oldestKey);
        this._bytesCached -= oldestBytes.byteLength;
        this._stats.evictions += 1;
      }

      this._chunks.set(key, owned);
      this._bytesCached += owned.byteLength;
    }

    _touchSlab(key) {
      if (!this._slabs.has(key)) {
        return null;
      }
      const bytes = this._slabs.get(key);
      this._slabs.delete(key);
      this._slabs.set(key, bytes);
      return bytes;
    }

    _putSlab(key, bytes) {
      if (!(bytes instanceof Uint8Array) || bytes.byteLength === 0) {
        return;
      }
      const owned =
        bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
          ? bytes
          : bytes.slice();

      if (this._slabs.has(key)) {
        const existing = this._slabs.get(key);
        this._slabBytesCached -= existing.byteLength;
        this._slabs.delete(key);
      }

      if (this.slabMaxBytes <= 0 || owned.byteLength > this.slabMaxBytes) {
        return;
      }

      while (
        this._slabBytesCached + owned.byteLength > this.slabMaxBytes &&
        this._slabs.size > 0
      ) {
        const oldestKey = this._slabs.keys().next().value;
        const oldestBytes = this._slabs.get(oldestKey);
        this._slabs.delete(oldestKey);
        this._slabBytesCached -= oldestBytes.byteLength;
        this._stats.slabEvictions += 1;
      }

      this._slabs.set(key, owned);
      this._slabBytesCached += owned.byteLength;
    }

    _makeHttpError(status, url, rangeLabel) {
      const errno = status === 403 ? -13 : status === 404 ? -2 : -5;
      const err = createErrnoError(errno, `HTTP ${status} reading ${url}[${rangeLabel}]`);
      err.status = status;
      return err;
    }

    async _fetchRangeBytes(url, reqStart, reqEnd) {
      if (typeof this.fetchImpl !== 'function') {
        throw createErrnoError(-19, 'ImmutableHttpfsCache has no fetch implementation available');
      }

      this._stats.networkFetches += 1;
      const res = await this.fetchImpl(url, {
        headers: { Range: `bytes=${reqStart}-${reqEnd}` },
      });

      const status = res && typeof res.status === 'number' ? res.status : 0;

      if (status === 416) {
        const cr = parseContentRangeHeader(getResponseHeader(res.headers, 'Content-Range'));
        if (cr && Number.isSafeInteger(cr.total)) {
          this.setFileMetadata(url, cr.total);
        }
        return {
          bytes: new Uint8Array(0),
          actualStart: reqStart,
          actualEnd: reqStart - 1,
          knownTotal: cr && Number.isSafeInteger(cr.total) ? cr.total : this.getFileSize(url),
          reachedEof: true,
        };
      }

      if (status === 404 || status === 403) {
        const errno = status === 403 ? -13 : -2;
        this._putNegative(url, errno, status);
        throw this._makeHttpError(status, url, `${reqStart}..${reqEnd}`);
      }

      if (!res || (!res.ok && status !== 206)) {
        throw this._makeHttpError(status || 500, url, `${reqStart}..${reqEnd}`);
      }

      const raw = new Uint8Array(await res.arrayBuffer());

      if (status === 206) {
        const cr = parseContentRangeHeader(getResponseHeader(res.headers, 'Content-Range'));
        if (cr && !cr.unsatisfied) {
          if (cr.total !== null) {
            this.setFileMetadata(url, cr.total);
          } else {
            this.setFileMetadata(url, null);
          }
          const expectedLen = cr.end - cr.start + 1;
          const availLen = Math.min(raw.byteLength, expectedLen);
          const actualEnd = cr.start + availLen - 1;
          const reachedEof =
            (cr.total !== null && actualEnd + 1 >= cr.total) || availLen === 0;
          return {
            bytes: availLen === raw.byteLength ? raw : raw.subarray(0, availLen),
            actualStart: cr.start,
            actualEnd,
            knownTotal: cr.total !== null ? cr.total : this.getFileSize(url),
            reachedEof,
          };
        }

        // 206 without parseable Content-Range header.
        const requestedLen = reqEnd - reqStart + 1;
        const reachedEof = raw.byteLength < requestedLen;
        if (reachedEof) {
          this.setFileMetadata(url, reqStart + raw.byteLength);
        } else {
          this.setFileMetadata(url, null);
        }
        return {
          bytes: raw,
          actualStart: reqStart,
          actualEnd: reqStart + raw.byteLength - 1,
          knownTotal: reachedEof ? reqStart + raw.byteLength : this.getFileSize(url),
          reachedEof,
        };
      }

      // 200 OK fallback (server returned full resource instead of 206 Partial Content).
      this.setFileMetadata(url, raw.byteLength);
      return {
        bytes: raw,
        actualStart: 0,
        actualEnd: raw.byteLength - 1,
        knownTotal: raw.byteLength,
        reachedEof: true,
      };
    }

    _fetchChunkRun(url, runStartIdx, runEndIdx) {
      const deferreds = [];
      for (let idx = runStartIdx; idx <= runEndIdx; idx += 1) {
        let resolve;
        let reject;
        const promise = new Promise((res, rej) => {
          resolve = res;
          reject = rej;
        });
        const key = this._chunkKey(url, idx);
        this._inFlightChunks.set(key, promise);
        deferreds.push({ idx, key, promise, resolve, reject });
      }

      const executeRun = async () => {
        try {
          const knownSize = this.getFileSize(url);
          const runStart = runStartIdx * this.chunkSize;
          if (knownSize !== null && runStart >= knownSize) {
            for (let i = 0; i < deferreds.length; i += 1) {
              deferreds[i].resolve(new Uint8Array(0));
            }
            return;
          }

          const naturalEnd = (runEndIdx + 1) * this.chunkSize - 1;
          const reqEnd =
            knownSize !== null && knownSize > 0
              ? Math.min(naturalEnd, knownSize - 1)
              : naturalEnd;

          const fetched = await this._fetchRangeBytes(url, runStart, reqEnd);
          const totalAfter =
            fetched.knownTotal !== null ? fetched.knownTotal : this.getFileSize(url);
          const uncovered = [];

          for (let i = 0; i < deferreds.length; i += 1) {
            const item = deferreds[i];
            const cStart = item.idx * this.chunkSize;
            const cMaxEnd = (item.idx + 1) * this.chunkSize - 1;

            if (totalAfter !== null && cStart >= totalAfter) {
              item.resolve(new Uint8Array(0));
              continue;
            }

            const expectedEnd =
              totalAfter !== null ? Math.min(cMaxEnd, totalAfter - 1) : cMaxEnd;

            const startsInResponse =
              fetched.bytes.byteLength > 0 &&
              cStart >= fetched.actualStart &&
              cStart <= fetched.actualEnd;

            if (startsInResponse) {
              const relStart = cStart - fetched.actualStart;
              const availFromChunkStart = fetched.bytes.byteLength - relStart;
              const neededLen = expectedEnd - cStart + 1;
              const coversCompleteChunk =
                fetched.actualEnd >= expectedEnd ||
                (fetched.reachedEof && availFromChunkStart > 0);

              if (coversCompleteChunk) {
                const takeLen = Math.min(this.chunkSize, neededLen, availFromChunkStart);
                const chunkBytes = fetched.bytes.slice(relStart, relStart + takeLen);
                if (chunkBytes.byteLength > 0) {
                  this._putChunk(item.key, chunkBytes);
                }
                item.resolve(chunkBytes);
                continue;
              }
            } else if (fetched.reachedEof && cStart > fetched.actualEnd) {
              item.resolve(new Uint8Array(0));
              continue;
            }

            uncovered.push(item);
          }

          // If the server returned a truncated 206 covering only a prefix of the
          // coalesced run, fetch each remaining uncovered chunk individually.
          for (let i = 0; i < uncovered.length; i += 1) {
            const item = uncovered[i];
            const latestSize = this.getFileSize(url);
            const cStart = item.idx * this.chunkSize;
            if (latestSize !== null && cStart >= latestSize) {
              item.resolve(new Uint8Array(0));
              continue;
            }
            const cNaturalEnd = (item.idx + 1) * this.chunkSize - 1;
            const cReqEnd =
              latestSize !== null && latestSize > 0
                ? Math.min(cNaturalEnd, latestSize - 1)
                : cNaturalEnd;
            try {
              const single = await this._fetchRangeBytes(url, cStart, cReqEnd);
              if (
                single.bytes.byteLength === 0 ||
                cStart < single.actualStart ||
                cStart > single.actualEnd
              ) {
                item.resolve(new Uint8Array(0));
              } else {
                const relStart = cStart - single.actualStart;
                const takeLen = Math.min(this.chunkSize, single.bytes.byteLength - relStart);
                const chunkBytes = single.bytes.slice(relStart, relStart + takeLen);
                if (chunkBytes.byteLength > 0) {
                  this._putChunk(item.key, chunkBytes);
                }
                item.resolve(chunkBytes);
              }
            } catch (err) {
              item.reject(err);
            }
          }
        } catch (err) {
          for (let i = 0; i < deferreds.length; i += 1) {
            deferreds[i].reject(err);
          }
        } finally {
          for (let i = 0; i < deferreds.length; i += 1) {
            const item = deferreds[i];
            if (this._inFlightChunks.get(item.key) === item.promise) {
              this._inFlightChunks.delete(item.key);
            }
          }
        }
      };

      executeRun();
      return deferreds.map((d) => d.promise);
    }

    _getChunkRange(url, firstChunk, lastChunk) {
      const totalChunks = lastChunk - firstChunk + 1;
      const chunkPromises = new Array(totalChunks);
      const maxRunChunks = Math.max(1, Math.floor(MAX_COALESCED_FETCH_BYTES / this.chunkSize));

      let runStartPos = -1;
      const flushMissingRun = (endPosExclusive) => {
        if (runStartPos < 0) {
          return;
        }
        let curPos = runStartPos;
        while (curPos < endPosExclusive) {
          const batchEndExclusive = Math.min(endPosExclusive, curPos + maxRunChunks);
          const runStartIdx = firstChunk + curPos;
          const runEndIdx = firstChunk + batchEndExclusive - 1;
          const promises = this._fetchChunkRun(url, runStartIdx, runEndIdx);
          for (let k = 0; k < promises.length; k += 1) {
            chunkPromises[curPos + k] = promises[k];
          }
          curPos = batchEndExclusive;
        }
        runStartPos = -1;
      };

      for (let i = 0; i < totalChunks; i += 1) {
        const chunkIndex = firstChunk + i;
        const key = this._chunkKey(url, chunkIndex);

        const memHit = this._touchChunk(key);
        if (memHit !== null) {
          flushMissingRun(i);
          this._stats.chunkHits += 1;
          chunkPromises[i] = Promise.resolve(memHit);
          continue;
        }

        const inFlight = this._inFlightChunks.get(key);
        if (inFlight) {
          flushMissingRun(i);
          this._stats.inFlightCoalesced += 1;
          chunkPromises[i] = inFlight;
          continue;
        }

        this._stats.chunkMisses += 1;
        if (runStartPos < 0) {
          runStartPos = i;
        }
      }
      flushMissingRun(totalChunks);

      return Promise.all(chunkPromises);
    }

    async open(url) {
      const existingMeta = this._touchMeta(url);
      if (existingMeta !== null) {
        this._stats.metaHits += 1;
        return { ok: true, size: existingMeta.size };
      }

      const neg = this._getNegative(url);
      if (neg !== null) {
        this._stats.negativeHits += 1;
        return { ok: false, errno: neg.errno, status: neg.status };
      }

      const existingOpen = this._inFlightOpen.get(url);
      if (existingOpen) {
        this._stats.inFlightCoalesced += 1;
        return existingOpen;
      }

      this._stats.metaMisses += 1;
      const openPromise = (async () => {
        try {
          await this._getChunkRange(url, 0, 0);
          const meta = this._fileMeta.get(url);
          if (!meta) {
            this.setFileMetadata(url, null);
          }
          return { ok: true, size: meta ? meta.size : null };
        } catch (err) {
          return {
            ok: false,
            errno: mapErrorToErrno(err, -2),
            status: err && typeof err.status === 'number' ? err.status : null,
          };
        }
      })();

      this._inFlightOpen.set(url, openPromise);
      try {
        return await openPromise;
      } finally {
        if (this._inFlightOpen.get(url) === openPromise) {
          this._inFlightOpen.delete(url);
        }
      }
    }

    _acquireSlabSlot(isPrefetch = false) {
      const limit = Math.max(1, this.maxConcurrentSlabFetches || DEFAULT_MAX_CONCURRENT_SLAB_FETCHES);
      if (this._activeSlabFetches < limit) {
        this._activeSlabFetches += 1;
        if (this._activeSlabFetches > this._stats.peakConcurrentSlabFetches) {
          this._stats.peakConcurrentSlabFetches = this._activeSlabFetches;
        }
        return Promise.resolve();
      }
      return new Promise((resolve) => {
        const waiter = { resolve, priority: isPrefetch ? 1 : 0 };
        if (waiter.priority === 0) {
          // Insert demand reads ahead of queued background prefetch waiters.
          let insertAt = this._slabWaitQueue.length;
          for (let i = 0; i < this._slabWaitQueue.length; i += 1) {
            if (this._slabWaitQueue[i].priority > 0) {
              insertAt = i;
              break;
            }
          }
          this._slabWaitQueue.splice(insertAt, 0, waiter);
        } else {
          this._slabWaitQueue.push(waiter);
        }
      });
    }

    _releaseSlabSlot() {
      const limit = Math.max(1, this.maxConcurrentSlabFetches || DEFAULT_MAX_CONCURRENT_SLAB_FETCHES);
      if (this._slabWaitQueue.length > 0 && this._activeSlabFetches <= limit) {
        const next = this._slabWaitQueue.shift();
        next.resolve();
        return;
      }
      if (this._activeSlabFetches > 0) {
        this._activeSlabFetches -= 1;
      }
      while (this._slabWaitQueue.length > 0 && this._activeSlabFetches < limit) {
        this._activeSlabFetches += 1;
        if (this._activeSlabFetches > this._stats.peakConcurrentSlabFetches) {
          this._stats.peakConcurrentSlabFetches = this._activeSlabFetches;
        }
        const next = this._slabWaitQueue.shift();
        next.resolve();
      }
    }

    _fetchSlab(url, slabIndex, options = {}) {
      const isPrefetch = Boolean(options && options.isPrefetch);
      const key = this._slabKey(url, slabIndex);
      const existing = this._inFlightSlabs.get(key);
      if (existing) {
        this._stats.inFlightCoalesced += 1;
        return existing;
      }

      const promise = (async () => {
        await this._acquireSlabSlot(isPrefetch);
        try {
          const slabStart = slabIndex * this.slabSize;
          const knownSize = this.getFileSize(url);
          if (knownSize !== null && slabStart >= knownSize) {
            return new Uint8Array(0);
          }

          const naturalEnd = (slabIndex + 1) * this.slabSize - 1;
          const reqEnd =
            knownSize !== null && knownSize > 0
              ? Math.min(naturalEnd, knownSize - 1)
              : naturalEnd;

          this._stats.slabNetworkFetches += 1;
          const first = await this._fetchRangeBytes(url, slabStart, reqEnd);
          const totalAfter =
            first.knownTotal !== null ? first.knownTotal : this.getFileSize(url);

          if (totalAfter !== null && slabStart >= totalAfter) {
            return new Uint8Array(0);
          }

          const expectedEnd =
            totalAfter !== null ? Math.min(naturalEnd, totalAfter - 1) : naturalEnd;
          const expectedLen = expectedEnd - slabStart + 1;

          if (
            first.bytes.byteLength > 0 &&
            slabStart >= first.actualStart &&
            slabStart <= first.actualEnd
          ) {
            const relStart = slabStart - first.actualStart;
            const avail = first.bytes.byteLength - relStart;
            const coversComplete =
              first.actualEnd >= expectedEnd || (first.reachedEof && avail > 0);
            if (coversComplete) {
              const takeLen = Math.min(this.slabSize, expectedLen, avail);
              const slabBytes =
                relStart === 0 &&
                takeLen === first.bytes.byteLength &&
                first.bytes.byteOffset === 0 &&
                first.bytes.byteLength === first.bytes.buffer.byteLength
                  ? first.bytes
                  : first.bytes.slice(relStart, relStart + takeLen);
              if (slabBytes.byteLength > 0) {
                this._putSlab(key, slabBytes);
              }
              return slabBytes;
            }

            // Server returned a truncated 206 prefix of the slab; continue fetching remaining tail.
            const parts = [first.bytes.slice(relStart, relStart + avail)];
            let filled = avail;
            let cursor = slabStart + avail;
            const maxContinuations = 64;
            for (let step = 0; step < maxContinuations && cursor <= expectedEnd; step += 1) {
              const latestSize = this.getFileSize(url);
              const curTargetEnd =
                latestSize !== null ? Math.min(naturalEnd, latestSize - 1) : expectedEnd;
              if (cursor > curTargetEnd) {
                break;
              }
              this._stats.slabNetworkFetches += 1;
              const next = await this._fetchRangeBytes(url, cursor, curTargetEnd);
              if (
                next.bytes.byteLength === 0 ||
                cursor < next.actualStart ||
                cursor > next.actualEnd
              ) {
                break;
              }
              const nRel = cursor - next.actualStart;
              const nAvail = Math.min(curTargetEnd - cursor + 1, next.bytes.byteLength - nRel);
              if (nAvail <= 0) {
                break;
              }
              parts.push(next.bytes.slice(nRel, nRel + nAvail));
              filled += nAvail;
              cursor += nAvail;
              if (next.reachedEof) {
                break;
              }
            }

            const combined = new Uint8Array(filled);
            let pos = 0;
            for (let i = 0; i < parts.length; i += 1) {
              combined.set(parts[i], pos);
              pos += parts[i].byteLength;
            }
            if (combined.byteLength > 0) {
              this._putSlab(key, combined);
            }
            return combined;
          }

          return new Uint8Array(0);
        } finally {
          this._releaseSlabSlot();
        }
      })();

      this._inFlightSlabs.set(key, promise);
      const cleanup = () => {
        if (this._inFlightSlabs.get(key) === promise) {
          this._inFlightSlabs.delete(key);
        }
      };
      promise.then(cleanup, cleanup);
      return promise;
    }

    _recordSequentialAccess(url, offset, length, lastSlab) {
      if (this.prefetchAheadSlabs <= 0 || this.slabSize <= 0 || this.slabMaxBytes < this.slabSize * 2) {
        return 0;
      }
      const endOffset = offset + length;
      let tracker = this._streamTrackers.get(url);
      if (tracker) {
        this._streamTrackers.delete(url);
        const isSequential =
          offset >= tracker.lastOffset &&
          offset <= tracker.lastEndOffset + this.slabSize &&
          (endOffset > tracker.lastEndOffset || lastSlab >= tracker.lastSlabIndex);
        if (isSequential) {
          tracker.sequentialCount = Math.min(tracker.sequentialCount + 1, 1024);
        } else {
          tracker.sequentialCount = 1;
        }
        tracker.lastOffset = offset;
        tracker.lastEndOffset = endOffset;
        tracker.lastSlabIndex = lastSlab;
      } else {
        if (this._streamTrackers.size >= MAX_STREAM_TRACKER_ENTRIES) {
          const oldest = this._streamTrackers.keys().next().value;
          if (oldest !== undefined) {
            this._streamTrackers.delete(oldest);
          }
        }
        tracker = {
          lastOffset: offset,
          lastEndOffset: endOffset,
          lastSlabIndex: lastSlab,
          sequentialCount: 1,
        };
      }
      this._streamTrackers.set(url, tracker);

      if (tracker.sequentialCount >= this.sequentialThresholdReads) {
        return this._scheduleSlabPrefetch(url, lastSlab + 1, this.prefetchAheadSlabs);
      }
      return 0;
    }

    _scheduleSlabPrefetch(url, startSlabIndex, requestedCount) {
      if (
        this.slabSize <= 0 ||
        this.slabMaxBytes < this.slabSize * 2 ||
        !Number.isInteger(startSlabIndex) ||
        startSlabIndex < 0 ||
        !Number.isInteger(requestedCount) ||
        requestedCount <= 0
      ) {
        return 0;
      }
      if (this._getNegative(url) !== null) {
        return 0;
      }

      const maxFitSlabs = Math.floor(this.slabMaxBytes / this.slabSize);
      const boundedCount = Math.min(requestedCount, MAX_SLAB_PREFETCH_AHEAD, Math.max(0, maxFitSlabs - 1));
      if (boundedCount <= 0) {
        return 0;
      }

      const knownSize = this.getFileSize(url);
      let scheduled = 0;

      for (let i = 0; i < boundedCount; i += 1) {
        const slabIdx = startSlabIndex + i;
        const slabStart = slabIdx * this.slabSize;
        if (knownSize !== null && slabStart >= knownSize) {
          break;
        }
        const key = this._slabKey(url, slabIdx);
        if (this._slabs.has(key) || this._inFlightSlabs.has(key)) {
          continue;
        }

        this._stats.slabPrefetchesScheduled += 1;
        scheduled += 1;
        const p = this._fetchSlab(url, slabIdx, { isPrefetch: true }).then(
          (bytes) => {
            if (bytes && bytes.byteLength > 0) {
              this._stats.slabPrefetchesCompleted += 1;
            }
          },
          () => {
            this._stats.slabPrefetchErrors += 1;
          },
        );
        this._pendingPrefetches.add(p);
        p.finally(() => {
          this._pendingPrefetches.delete(p);
        });
      }

      return scheduled;
    }

    async prefetchSlabs(url, startOffset = 0, count) {
      if (typeof url !== 'string' || url.length === 0) {
        return { scheduled: 0, alreadyCached: 0, skipped: 0, completed: 0, failed: 0 };
      }
      if (this.slabSize <= 0 || this.slabMaxBytes < this.slabSize) {
        return { scheduled: 0, alreadyCached: 0, skipped: 0, completed: 0, failed: 0 };
      }
      if (!Number.isInteger(startOffset) || startOffset < 0) {
        throw createErrnoError(-22, `Invalid prefetch startOffset: ${startOffset}`);
      }
      const reqCount =
        Number.isInteger(count) && count >= 0
          ? count
          : Math.max(1, this.prefetchAheadSlabs || 1);
      const maxFitSlabs = Math.floor(this.slabMaxBytes / this.slabSize);
      const boundedCount = Math.min(reqCount, MAX_SLAB_PREFETCH_AHEAD, Math.max(1, maxFitSlabs));
      const firstSlab = Math.floor(startOffset / this.slabSize);
      const knownSize = this.getFileSize(url);

      let scheduled = 0;
      let alreadyCached = 0;
      let skipped = 0;
      let completed = 0;
      let failed = 0;
      const tasks = [];

      for (let i = 0; i < boundedCount; i += 1) {
        const slabIdx = firstSlab + i;
        if (knownSize !== null && slabIdx * this.slabSize >= knownSize) {
          skipped += 1;
          break;
        }
        const key = this._slabKey(url, slabIdx);
        if (this._slabs.has(key)) {
          this._touchSlab(key);
          alreadyCached += 1;
          continue;
        }
        this._stats.slabPrefetchesScheduled += 1;
        scheduled += 1;
        const p = this._fetchSlab(url, slabIdx, { isPrefetch: true }).then(
          (bytes) => {
            if (bytes && bytes.byteLength > 0) {
              this._stats.slabPrefetchesCompleted += 1;
              completed += 1;
            }
          },
          () => {
            this._stats.slabPrefetchErrors += 1;
            failed += 1;
          },
        );
        this._pendingPrefetches.add(p);
        p.finally(() => {
          this._pendingPrefetches.delete(p);
        });
        tasks.push(p);
      }

      if (tasks.length > 0) {
        await Promise.all(tasks);
      }
      return { scheduled, alreadyCached, skipped, completed, failed };
    }

    async waitForPrefetches() {
      while (this._pendingPrefetches.size > 0) {
        await Promise.all(Array.from(this._pendingPrefetches));
      }
    }

    tuneSlabCache(profileOrOptions = 'video4k') {
      const merged =
        typeof profileOrOptions === 'string'
          ? createSlabCacheTuningProfile({ profile: profileOrOptions })
          : createSlabCacheTuningProfile(
              Object.assign(
                {
                  slabSize: this.slabSize || DEFAULT_SLAB_SIZE,
                  slabMaxBytes: this.slabMaxBytes || DEFAULT_SLAB_MAX_BYTES,
                  prefetchAheadSlabs: this.prefetchAheadSlabs,
                  maxConcurrentSlabFetches: this.maxConcurrentSlabFetches,
                  sequentialThresholdReads: this.sequentialThresholdReads,
                },
                profileOrOptions && typeof profileOrOptions === 'object' ? profileOrOptions : {},
              ),
            );

      const prevSlabSize = this.slabSize;
      this.slabSize = merged.slabSize;
      this.slabMaxBytes = merged.slabMaxBytes;
      this.prefetchAheadSlabs = merged.prefetchAheadSlabs;
      this.maxConcurrentSlabFetches = merged.maxConcurrentSlabFetches;
      this.sequentialThresholdReads = merged.sequentialThresholdReads;

      if (prevSlabSize !== this.slabSize || this.slabSize <= 0 || this.slabMaxBytes <= 0) {
        this._slabs.clear();
        this._slabBytesCached = 0;
        this._streamTrackers.clear();
      } else {
        while (this._slabBytesCached > this.slabMaxBytes && this._slabs.size > 0) {
          const oldestKey = this._slabs.keys().next().value;
          const oldestBytes = this._slabs.get(oldestKey);
          this._slabs.delete(oldestKey);
          this._slabBytesCached -= oldestBytes.byteLength;
          this._stats.slabEvictions += 1;
        }
      }

      return {
        slabSize: this.slabSize,
        slabMaxBytes: this.slabMaxBytes,
        prefetchAheadSlabs: this.prefetchAheadSlabs,
        maxConcurrentSlabFetches: this.maxConcurrentSlabFetches,
        sequentialThresholdReads: this.sequentialThresholdReads,
      };
    }

    _getSlabRange(url, firstSlab, lastSlab) {
      const totalSlabs = lastSlab - firstSlab + 1;
      const slabPromises = new Array(totalSlabs);
      for (let i = 0; i < totalSlabs; i += 1) {
        const slabIndex = firstSlab + i;
        const key = this._slabKey(url, slabIndex);

        const memHit = this._touchSlab(key);
        if (memHit !== null) {
          this._stats.slabHits += 1;
          slabPromises[i] = Promise.resolve(memHit);
          continue;
        }

        const inFlight = this._inFlightSlabs.get(key);
        if (inFlight) {
          this._stats.inFlightCoalesced += 1;
          slabPromises[i] = inFlight;
          continue;
        }

        this._stats.slabMisses += 1;
        slabPromises[i] = this._fetchSlab(url, slabIndex);
      }
      return Promise.all(slabPromises);
    }

    tryReadSync(url, offset, length, options = {}) {
      if (!Number.isInteger(offset) || offset < 0) {
        return null;
      }
      if (!Number.isInteger(length) || length < 0) {
        return null;
      }
      if (length === 0) {
        return new Uint8Array(0);
      }
      if (this._getNegative(url) !== null) {
        return null;
      }

      let effectiveLen = length;
      const meta = this._touchMeta(url);
      if (meta && Number.isSafeInteger(meta.size)) {
        if (offset >= meta.size) {
          return new Uint8Array(0);
        }
        effectiveLen = Math.min(length, meta.size - offset);
        if (effectiveLen <= 0) {
          return new Uint8Array(0);
        }
      }

      const copy = options.copy !== false;

      if (this.slabSize > 0 && this.slabMaxBytes > 0 && this._slabs.size > 0) {
        const firstSlab = Math.floor(offset / this.slabSize);
        const lastSlab = Math.floor((offset + effectiveLen - 1) / this.slabSize);
        let allSlabsPresent = true;
        for (let s = firstSlab; s <= lastSlab; s += 1) {
          if (!this._slabs.has(this._slabKey(url, s))) {
            allSlabsPresent = false;
            break;
          }
        }
        if (allSlabsPresent) {
          const slices = [];
          let totalBytes = 0;
          for (let s = firstSlab; s <= lastSlab; s += 1) {
            const slab = this._touchSlab(this._slabKey(url, s));
            this._stats.slabHits += 1;
            if (!slab || slab.byteLength === 0) {
              break;
            }
            const slabStart = s * this.slabSize;
            const sliceStart = Math.max(0, offset - slabStart);
            const sliceEnd = Math.min(slab.byteLength, offset + effectiveLen - slabStart);
            if (sliceEnd > sliceStart) {
              const part = slab.subarray(sliceStart, sliceEnd);
              slices.push(part);
              totalBytes += part.byteLength;
            }
            if (slab.byteLength < this.slabSize) {
              break;
            }
          }
          if (this.prefetchAheadSlabs > 0) {
            this._recordSequentialAccess(url, offset, effectiveLen, lastSlab);
          }
          if (!copy && slices.length === 1) {
            return slices[0];
          }
          const out = new Uint8Array(totalBytes);
          let cursor = 0;
          for (let i = 0; i < slices.length; i += 1) {
            out.set(slices[i], cursor);
            cursor += slices[i].byteLength;
          }
          return out;
        }
      }

      if (this._chunks.size > 0) {
        const firstChunk = Math.floor(offset / this.chunkSize);
        const lastChunk = Math.floor((offset + effectiveLen - 1) / this.chunkSize);
        let allChunksPresent = true;
        for (let c = firstChunk; c <= lastChunk; c += 1) {
          if (!this._chunks.has(this._chunkKey(url, c))) {
            allChunksPresent = false;
            break;
          }
        }
        if (allChunksPresent) {
          const slices = [];
          let totalBytes = 0;
          for (let c = firstChunk; c <= lastChunk; c += 1) {
            const chunk = this._touchChunk(this._chunkKey(url, c));
            this._stats.chunkHits += 1;
            if (!chunk || chunk.byteLength === 0) {
              break;
            }
            const chunkStart = c * this.chunkSize;
            const sliceStart = Math.max(0, offset - chunkStart);
            const sliceEnd = Math.min(chunk.byteLength, offset + effectiveLen - chunkStart);
            if (sliceEnd > sliceStart) {
              const part = chunk.subarray(sliceStart, sliceEnd);
              slices.push(part);
              totalBytes += part.byteLength;
            }
            if (chunk.byteLength < this.chunkSize) {
              break;
            }
          }
          if (!copy && slices.length === 1) {
            return slices[0];
          }
          const out = new Uint8Array(totalBytes);
          let cursor = 0;
          for (let i = 0; i < slices.length; i += 1) {
            out.set(slices[i], cursor);
            cursor += slices[i].byteLength;
          }
          return out;
        }
      }

      return null;
    }

    async read(url, offset, length) {
      if (!Number.isInteger(offset) || offset < 0) {
        throw createErrnoError(-22, `Invalid read offset: ${offset}`);
      }
      if (!Number.isInteger(length) || length < 0) {
        throw createErrnoError(-22, `Invalid read length: ${length}`);
      }
      if (length === 0) {
        return new Uint8Array(0);
      }

      const neg = this._getNegative(url);
      if (neg !== null) {
        this._stats.negativeHits += 1;
        throw this._makeHttpError(neg.status || 404, url, `${offset}..${offset + length - 1}`);
      }

      let effectiveLen = length;
      const meta = this._touchMeta(url);
      if (meta && Number.isSafeInteger(meta.size)) {
        if (offset >= meta.size) {
          return new Uint8Array(0);
        }
        effectiveLen = Math.min(length, meta.size - offset);
        if (effectiveLen <= 0) {
          return new Uint8Array(0);
        }
      }

      const syncHit = this.tryReadSync(url, offset, effectiveLen, { copy: true });
      if (syncHit !== null) {
        return syncHit;
      }

      if (this.slabSize > 0 && this.slabMaxBytes > 0) {
        const firstSlab = Math.floor(offset / this.slabSize);
        const lastSlab = Math.floor((offset + effectiveLen - 1) / this.slabSize);
        const slabs = await this._getSlabRange(url, firstSlab, lastSlab);

        const slices = [];
        let totalBytes = 0;
        for (let i = 0; i < slabs.length; i += 1) {
          const slabIndex = firstSlab + i;
          const slab = slabs[i];
          if (!slab || slab.byteLength === 0) {
            break;
          }
          const slabStart = slabIndex * this.slabSize;
          const sliceStart = Math.max(0, offset - slabStart);
          const sliceEnd = Math.min(slab.byteLength, offset + effectiveLen - slabStart);
          if (sliceEnd > sliceStart) {
            const part = slab.subarray(sliceStart, sliceEnd);
            slices.push(part);
            totalBytes += part.byteLength;
          }
          if (slab.byteLength < this.slabSize) {
            break;
          }
        }

        const out = new Uint8Array(totalBytes);
        let cursor = 0;
        for (let i = 0; i < slices.length; i += 1) {
          const part = slices[i];
          out.set(part, cursor);
          cursor += part.byteLength;
        }
        if (this.prefetchAheadSlabs > 0) {
          this._recordSequentialAccess(url, offset, effectiveLen, lastSlab);
        }
        return out;
      }

      const firstChunk = Math.floor(offset / this.chunkSize);
      const lastChunk = Math.floor((offset + effectiveLen - 1) / this.chunkSize);
      const chunks = await this._getChunkRange(url, firstChunk, lastChunk);

      const slices = [];
      let totalBytes = 0;

      for (let i = 0; i < chunks.length; i += 1) {
        const chunkIndex = firstChunk + i;
        const chunk = chunks[i];
        if (!chunk || chunk.byteLength === 0) {
          break;
        }
        const chunkStart = chunkIndex * this.chunkSize;
        const sliceStart = Math.max(0, offset - chunkStart);
        const sliceEnd = Math.min(chunk.byteLength, offset + effectiveLen - chunkStart);
        if (sliceEnd > sliceStart) {
          const part = chunk.subarray(sliceStart, sliceEnd);
          slices.push(part);
          totalBytes += part.byteLength;
        }
        if (chunk.byteLength < this.chunkSize) {
          break;
        }
      }

      // Always return a fresh owned buffer so callers cannot mutate cached chunks.
      const out = new Uint8Array(totalBytes);
      let cursor = 0;
      for (let i = 0; i < slices.length; i += 1) {
        const part = slices[i];
        out.set(part, cursor);
        cursor += part.byteLength;
      }
      return out;
    }

    clear() {
      this._chunks.clear();
      this._slabs.clear();
      this._fileMeta.clear();
      this._negativeCache.clear();
      this._streamTrackers.clear();
      this._bytesCached = 0;
      this._slabBytesCached = 0;
    }

    getStats() {
      return {
        hits: this._stats.chunkHits,
        misses: this._stats.chunkMisses,
        slabCacheHits: this._stats.slabHits,
        slabCacheMisses: this._stats.slabMisses,
        slabEvictions: this._stats.slabEvictions,
        slabNetworkFetches: this._stats.slabNetworkFetches,
        slabBytesCached: this._slabBytesCached,
        slabMaxBytes: this.slabMaxBytes,
        slabSize: this.slabSize,
        slabEntries: this._slabs.size,
        slabPrefetchesScheduled: this._stats.slabPrefetchesScheduled,
        slabPrefetchesCompleted: this._stats.slabPrefetchesCompleted,
        slabPrefetchErrors: this._stats.slabPrefetchErrors,
        prefetchAheadSlabs: this.prefetchAheadSlabs,
        maxConcurrentSlabFetches: this.maxConcurrentSlabFetches,
        activeSlabFetches: this._activeSlabFetches,
        peakConcurrentSlabFetches: this._stats.peakConcurrentSlabFetches,
        metaHits: this._stats.metaHits,
        metaMisses: this._stats.metaMisses,
        negativeHits: this._stats.negativeHits,
        negativeEntries: this._negativeCache.size,
        inFlightCoalesced: this._stats.inFlightCoalesced,
        networkFetches: this._stats.networkFetches,
        evictions: this._stats.evictions,
        bytesCached: this._bytesCached,
        maxBytes: this.maxBytes,
        chunkSize: this.chunkSize,
        entries: this._chunks.size,
        filesIndexed: this._fileMeta.size,
      };
    }
  }

  function emptyRangeCacheStats() {
    return {
      hits: 0,
      misses: 0,
      opfsHits: 0,
      evictions: 0,
      bytesCached: 0,
      entries: 0,
    };
  }

  function emptySchedulerStats() {
    const mk = (priority) => ({
      priority,
      queued: 0,
      inFlight: 0,
      completed: 0,
      failed: 0,
      cancelled: 0,
      retries: 0,
    });
    return {
      queued: 0,
      inFlight: 0,
      completed: 0,
      failed: 0,
      cancelled: 0,
      retries: 0,
      byPriority: {
        CRITICAL: mk(IO_PRIORITY.CRITICAL),
        HIGH: mk(IO_PRIORITY.HIGH),
        NORMAL: mk(IO_PRIORITY.NORMAL),
        PREFETCH: mk(IO_PRIORITY.PREFETCH),
      },
    };
  }

  function emptyWholeFileCacheStats(maxBytes = DEFAULT_WHOLE_FILE_MAX_BYTES) {
    return {
      pgimFullFetches: 0,
      wholeFileFetches: 0,
      hits: 0,
      fallbacks: 0,
      releases: 0,
      bytesFetched: 0,
      activeEntries: 0,
      activeBytes: 0,
      maxBytes:
        Number.isFinite(maxBytes) && maxBytes >= 0
          ? Math.floor(maxBytes)
          : DEFAULT_WHOLE_FILE_MAX_BYTES,
    };
  }

  function emptyMailboxStats() {
    return {
      requests: 0,
      syncFastPathHits: 0,
      asyncPathHits: 0,
    };
  }

  function emptyDumpWriteStats(enabled = false) {
    return {
      enabled: Boolean(enabled),
      dumpWritesCompleted: 0,
      dumpBytesWritten: 0,
      dumpWriteErrors: 0,
      dumpHandlesOpened: 0,
      dumpFlushesCompleted: 0,
      openDumpHandles: 0,
      bufferedBytes: 0,
    };
  }

  function emptyStats() {
    const cacheStats = emptyRangeCacheStats();
    const immStats = emptyImmutableCacheStats();
    const wholeStats = emptyWholeFileCacheStats();
    const mailboxStats = emptyMailboxStats();
    const dumpWriteStats = emptyDumpWriteStats(false);
    return {
      hits: cacheStats.hits,
      misses: cacheStats.misses,
      opfsHits: cacheStats.opfsHits,
      evictions: cacheStats.evictions,
      bytesCached: cacheStats.bytesCached,
      entries: cacheStats.entries,
      pgimFullFetches: wholeStats.pgimFullFetches,
      slabCacheHits: immStats.slabCacheHits,
      slabCacheMisses: immStats.slabCacheMisses,
      syncFastPathHits: mailboxStats.syncFastPathHits,
      dumpWritesCompleted: dumpWriteStats.dumpWritesCompleted,
      dumpBytesWritten: dumpWriteStats.dumpBytesWritten,
      dumpWriteErrors: dumpWriteStats.dumpWriteErrors,
      rangeCache: cacheStats,
      immutableCache: immStats,
      wholeFileCache: wholeStats,
      mailbox: mailboxStats,
      dumpWrites: dumpWriteStats,
      scheduler: emptySchedulerStats(),
      rpf: {
        mountedCount: 0,
        archives: [],
        mountErrors: 0,
      },
      mountedRpfCount: 0,
      openHandles: 0,
    };
  }

  function ensureTransferableBytes(bytes) {
    if (!(bytes instanceof Uint8Array)) {
      return new Uint8Array(0);
    }
    if (bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength) {
      return bytes;
    }
    return bytes.slice();
  }

  class IoWorkerController {
    constructor(options = {}) {
      this._rangeCacheModule = options.rangeCacheModule || null;
      this._httpfsMailboxModule = options.httpfsMailboxModule || null;
      this._ioSchedulerModule = options.ioSchedulerModule || null;
      this._rpf9VfsModule = options.rpf9VfsModule || null;

      this.dataBase = normalizeDataBase(options.dataBase);
      this.bootBase = normalizeOptionalBase(options.bootBase);
      this.gameBase = normalizeOptionalBase(options.gameBase);
      this.shadersBase =
        options.shadersBase !== undefined
          ? normalizeOptionalBase(options.shadersBase)
          : '/shaders/';
      const dumpCfg = resolveDumpConfig({
        allowDumpWrites: options.allowDumpWrites,
        dumpEndpoint: options.dumpEndpoint,
      });
      this.allowDumpWrites = dumpCfg.allowDumpWrites;
      this.dumpEndpoint = dumpCfg.dumpEndpoint;
      this._dumpMaxBufferBytes =
        Number.isFinite(options.dumpMaxBufferBytes) && options.dumpMaxBufferBytes > 0
          ? Math.floor(options.dumpMaxBufferBytes)
          : DEFAULT_DUMP_MAX_BUFFER_BYTES;
      this._dumpFlushChunkBytes =
        Number.isFinite(options.dumpFlushChunkBytes) && options.dumpFlushChunkBytes > 0
          ? Math.min(Math.floor(options.dumpFlushChunkBytes), this._dumpMaxBufferBytes)
          : Math.min(DEFAULT_DUMP_FLUSH_CHUNK_BYTES, this._dumpMaxBufferBytes);
      this.manifest = options.manifest || null;
      this.rangeCache = options.rangeCache || null;
      this._customRangeCacheProvided = Boolean(options.rangeCache);
      this._hasRoutedRangeCache = !options.rangeCache;
      this.immutableCache = options.immutableCache || null;
      this._immutableCacheEnabled = options.immutableCacheEnabled !== false;
      this.scheduler = options.scheduler || null;
      this.rpfMountTable = options.rpfMountTable || null;
      this.fetchImpl = typeof options.fetchImpl === 'function' ? options.fetchImpl : null;
      this.postMessage = typeof options.postMessage === 'function' ? options.postMessage : null;
      this.decryptBlock = typeof options.decryptBlock === 'function' ? options.decryptBlock : null;
      this.decompressEntry =
        typeof options.decompressEntry === 'function' ? options.decompressEntry : null;

      this._cacheOptions = {
        chunkSize: options.chunkSize,
        maxBytes: options.maxBytes,
        maxPrefetchConcurrency: options.maxPrefetchConcurrency,
        opfsDir: options.opfsDir !== undefined ? options.opfsDir : undefined,
      };
      this._immutableCacheOptions = {
        chunkSize: options.immutableCacheChunkSize,
        maxBytes: options.immutableCacheMaxBytes,
        slabSize:
          options.slabSize !== undefined ? options.slabSize : options.immutableSlabSize,
        slabMaxBytes:
          options.slabMaxBytes !== undefined
            ? options.slabMaxBytes
            : options.immutableSlabMaxBytes,
        slabCacheEnabled: options.slabCacheEnabled,
        slabProfile: options.slabProfile,
        prefetchAheadSlabs: options.prefetchAheadSlabs,
        maxConcurrentSlabFetches: options.maxConcurrentSlabFetches,
        sequentialThresholdReads: options.sequentialThresholdReads,
        negativeCacheTtlMs: options.negativeCacheTtlMs,
        negativeCacheMaxEntries: options.negativeCacheMaxEntries,
        metadataMaxEntries: options.metadataMaxEntries,
        now: typeof options.now === 'function' ? options.now : undefined,
      };
      this._wholeFileBufferEnabled = options.wholeFileBufferEnabled !== false;
      const rawWholeFileMax =
        options.wholeFileMaxBytes !== undefined
          ? options.wholeFileMaxBytes
          : options.pgimMaxBytes;
      this._wholeFileMaxBytes =
        Number.isFinite(rawWholeFileMax) && rawWholeFileMax >= 0
          ? Math.floor(rawWholeFileMax)
          : DEFAULT_WHOLE_FILE_MAX_BYTES;
      this._schedulerOptions = {
        maxConcurrency: options.maxConcurrency,
        maxRetries: options.maxRetries,
        baseRetryDelayMs: options.baseRetryDelayMs,
        sleep: options.sleep,
      };

      this._mailboxHost = null;
      this._handles = new Map();
      this._nextHandle = 1;
      this._activeWholeFiles = new Map(); // url -> { bytes, size, refCount }
      this._inFlightWholeFiles = new Map(); // url -> Promise<...>
      this._wholeFileStats = {
        pgimFullFetches: 0,
        wholeFileFetches: 0,
        hits: 0,
        fallbacks: 0,
        releases: 0,
        bytesFetched: 0,
      };
      this._dumpWriteStats = {
        dumpWritesCompleted: 0,
        dumpBytesWritten: 0,
        dumpWriteErrors: 0,
        dumpHandlesOpened: 0,
        dumpFlushesCompleted: 0,
      };
      this._mountedArchives = new Map();
      this._inFlightMounts = new Map();
      this._rpfMountErrors = 0;
      this._prefetchPromise = null;
      this._prefetchAbortController = null;
    }

    get prefetchPromise() {
      return this._prefetchPromise;
    }

    get mailboxHost() {
      return this._mailboxHost;
    }

    _getRangeCacheModule() {
      return resolveRangeCacheModule(this._rangeCacheModule);
    }

    _getHttpfsMailboxModule() {
      return resolveHttpfsMailboxModule(this._httpfsMailboxModule);
    }

    _getIoSchedulerModule() {
      return resolveIoSchedulerModule(this._ioSchedulerModule);
    }

    _getRpf9VfsModule() {
      return resolveRpf9VfsModule(this._rpf9VfsModule);
    }

    _getFetchImpl(override) {
      if (typeof override === 'function') return override;
      if (typeof this.fetchImpl === 'function') return this.fetchImpl;
      if (typeof fetch === 'function') return fetch.bind(globalThis);
      return null;
    }

    _getRouteBases() {
      return {
        dataBase: this.dataBase,
        bootBase: this.bootBase,
        gameBase: this.gameBase,
        shadersBase: this.shadersBase,
      };
    }

    _resolveRoute(rawPath) {
      return resolveHttpfsRoute(rawPath, this._getRouteBases());
    }

    _routeToCachePath(route) {
      if (
        this._hasRoutedRangeCache &&
        (route.mount === 'boot' || route.mount === 'game' || route.mount === 'shaders')
      ) {
        return `${MOUNT_CACHE_PREFIX}${route.mount}/${route.relPath}`;
      }
      return route.relPath;
    }

    _createRoutedFetch(baseFetch) {
      if (typeof baseFetch !== 'function') {
        return null;
      }
      const controller = this;
      return function routedFetch(url, init) {
        if (typeof url === 'string') {
          const dataPrefix = controller.dataBase;
          const bootMarker = `${dataPrefix}${MOUNT_CACHE_PREFIX}boot/`;
          if (controller.bootBase && url.startsWith(bootMarker)) {
            const rel = url.slice(bootMarker.length);
            return baseFetch(`${controller.bootBase}${rel}`, init);
          }
          const gameMarker = `${dataPrefix}${MOUNT_CACHE_PREFIX}game/`;
          if (controller.gameBase && url.startsWith(gameMarker)) {
            const rel = url.slice(gameMarker.length);
            return baseFetch(`${controller.gameBase}${rel}`, init);
          }
          const shadersMarker = `${dataPrefix}${MOUNT_CACHE_PREFIX}shaders/`;
          if (controller.shadersBase && url.startsWith(shadersMarker)) {
            const rel = url.slice(shadersMarker.length);
            return baseFetch(`${controller.shadersBase}${rel}`, init);
          }
        }
        return baseFetch(url, init);
      };
    }

    ensureRangeCache() {
      if (!this.rangeCache) {
        const mod = this._getRangeCacheModule();
        if (!mod || typeof mod.RangeCache !== 'function') {
          throw new Error('PlayGta6RangeCache unavailable in io_worker');
        }
        this._hasRoutedRangeCache = true;
        this.rangeCache = new mod.RangeCache({
          dataBase: this.dataBase,
          chunkSize: this._cacheOptions.chunkSize,
          maxBytes: this._cacheOptions.maxBytes,
          maxPrefetchConcurrency: this._cacheOptions.maxPrefetchConcurrency,
          opfsDir: this._cacheOptions.opfsDir || null,
          fetchImpl: this._createRoutedFetch(this._getFetchImpl()),
        });
      }
      return this.rangeCache;
    }

    ensureImmutableCache() {
      if (!this.immutableCache) {
        this.immutableCache = new ImmutableHttpfsCache({
          chunkSize: this._immutableCacheOptions.chunkSize,
          maxBytes: this._immutableCacheOptions.maxBytes,
          slabSize: this._immutableCacheOptions.slabSize,
          slabMaxBytes: this._immutableCacheOptions.slabMaxBytes,
          slabCacheEnabled: this._immutableCacheOptions.slabCacheEnabled,
          slabProfile: this._immutableCacheOptions.slabProfile,
          prefetchAheadSlabs: this._immutableCacheOptions.prefetchAheadSlabs,
          maxConcurrentSlabFetches: this._immutableCacheOptions.maxConcurrentSlabFetches,
          sequentialThresholdReads: this._immutableCacheOptions.sequentialThresholdReads,
          negativeCacheTtlMs: this._immutableCacheOptions.negativeCacheTtlMs,
          negativeCacheMaxEntries: this._immutableCacheOptions.negativeCacheMaxEntries,
          metadataMaxEntries: this._immutableCacheOptions.metadataMaxEntries,
          now: this._immutableCacheOptions.now,
          fetchImpl: this._getFetchImpl(),
        });
      }
      return this.immutableCache;
    }

    tuneSlabCache(profileOrOptions = 'video4k') {
      const immCache = this.ensureImmutableCache();
      const tuned = immCache.tuneSlabCache(profileOrOptions);
      this._immutableCacheOptions.slabSize = tuned.slabSize;
      this._immutableCacheOptions.slabMaxBytes = tuned.slabMaxBytes;
      this._immutableCacheOptions.slabCacheEnabled = tuned.slabSize > 0 && tuned.slabMaxBytes > 0;
      this._immutableCacheOptions.prefetchAheadSlabs = tuned.prefetchAheadSlabs;
      this._immutableCacheOptions.maxConcurrentSlabFetches = tuned.maxConcurrentSlabFetches;
      this._immutableCacheOptions.sequentialThresholdReads = tuned.sequentialThresholdReads;
      return tuned;
    }

    async prefetchVideoSlabs(rawPath, startOffset = 0, count) {
      const route = this._resolveRoute(rawPath);
      if (!route.ok || route.isRpf) {
        throw createErrnoError(-22, route.error || 'Invalid prefetch path');
      }
      const immCache = this.ensureImmutableCache();
      return immCache.prefetchSlabs(route.url, startOffset, count);
    }

    _shouldUseWholeFileBuffer(route) {
      if (
        !this._wholeFileBufferEnabled ||
        this._wholeFileMaxBytes <= 0 ||
        !route ||
        !route.ok ||
        route.isRpf
      ) {
        return false;
      }
      if (!isWholeFileBufferedPath(route.relPath)) {
        return false;
      }
      if (this._customRangeCacheProvided && !this.fetchImpl) {
        return false;
      }
      return Boolean(this._getFetchImpl());
    }

    _retainWholeFile(url, bytes, size) {
      let rec = this._activeWholeFiles.get(url);
      if (!rec) {
        rec = { bytes, size, refCount: 0 };
        this._activeWholeFiles.set(url, rec);
      }
      rec.refCount += 1;
      return rec;
    }

    _releaseWholeFile(url) {
      const rec = this._activeWholeFiles.get(url);
      if (!rec) {
        return false;
      }
      rec.refCount -= 1;
      if (rec.refCount <= 0) {
        this._activeWholeFiles.delete(url);
        this._wholeFileStats.releases += 1;
        return true;
      }
      return false;
    }

    async _fetchWholeFile(url, cleanPath, knownSize) {
      const cap = this._wholeFileMaxBytes;
      if (!this._wholeFileBufferEnabled || cap <= 0) {
        return { ok: true, buffered: false, size: knownSize };
      }

      const existing = this._activeWholeFiles.get(url);
      if (existing) {
        return {
          ok: true,
          buffered: true,
          bytes: existing.bytes,
          size: existing.size,
          reused: true,
        };
      }

      const immCache =
        this._immutableCacheEnabled && isImmutableHttpfsPath(cleanPath)
          ? this.ensureImmutableCache()
          : null;

      if (immCache) {
        const neg = immCache._getNegative(url);
        if (neg !== null) {
          immCache._stats.negativeHits += 1;
          return { ok: false, buffered: false, errno: neg.errno, status: neg.status };
        }
        if (!Number.isSafeInteger(knownSize)) {
          const cachedSize = immCache.getFileSize(url);
          if (Number.isSafeInteger(cachedSize)) {
            knownSize = cachedSize;
          }
        }
      }

      if (Number.isSafeInteger(knownSize) && knownSize > cap) {
        this._wholeFileStats.fallbacks += 1;
        return { ok: true, buffered: false, size: knownSize, probed: false };
      }

      const inFlight = this._inFlightWholeFiles.get(url);
      if (inFlight) {
        return inFlight;
      }

      const fetchFn = this._getFetchImpl();
      if (typeof fetchFn !== 'function') {
        return { ok: true, buffered: false, size: knownSize, probed: false };
      }

      const isPgim = cleanPath.toLowerCase().endsWith('.pgim');
      const rangeEnd =
        Number.isSafeInteger(knownSize) && knownSize > 0
          ? Math.min(knownSize - 1, cap - 1)
          : cap - 1;

      const promise = (async () => {
        try {
          const res = await fetchFn(url, {
            headers: { Range: `bytes=0-${rangeEnd}` },
          });
          const status = res && typeof res.status === 'number' ? res.status : 0;

          if (status === 416) {
            const cr = parseContentRangeHeader(getResponseHeader(res.headers, 'Content-Range'));
            if (cr && cr.total === 0) {
              if (immCache) {
                immCache.setFileMetadata(url, 0);
              }
              this._wholeFileStats.wholeFileFetches += 1;
              if (isPgim) {
                this._wholeFileStats.pgimFullFetches += 1;
              }
              return {
                ok: true,
                buffered: true,
                bytes: new Uint8Array(0),
                size: 0,
                reused: false,
                probed: true,
              };
            }
            return { ok: false, buffered: false, errno: -5, status: 416 };
          }

          if (status === 404 || status === 403) {
            const errno = status === 403 ? -13 : -2;
            if (immCache) {
              immCache._putNegative(url, errno, status);
            }
            return { ok: false, buffered: false, errno, status };
          }

          if (!res || (!res.ok && status !== 206)) {
            return { ok: false, buffered: false, errno: -5, status: status || 500 };
          }

          if (status === 206) {
            const cr = parseContentRangeHeader(getResponseHeader(res.headers, 'Content-Range'));
            if (cr && !cr.unsatisfied) {
              if (cr.total !== null && immCache) {
                immCache.setFileMetadata(url, cr.total);
              }
              if (cr.total !== null && cr.total > cap) {
                if (res.body && typeof res.body.cancel === 'function') {
                  try {
                    await res.body.cancel();
                  } catch {
                    // Ignore stream cancel errors.
                  }
                }
                this._wholeFileStats.fallbacks += 1;
                return { ok: true, buffered: false, size: cr.total, probed: true };
              }
            }

            const raw = new Uint8Array(await res.arrayBuffer());
            if (cr && !cr.unsatisfied) {
              const total = cr.total;
              const gotComplete =
                cr.start === 0 &&
                (total !== null
                  ? cr.end + 1 >= total && raw.byteLength >= total
                  : cr.end < rangeEnd && raw.byteLength <= cap);
              if (gotComplete) {
                const finalBytes =
                  total !== null && raw.byteLength > total ? raw.slice(0, total) : raw;
                const finalSize = total !== null ? total : finalBytes.byteLength;
                if (immCache) {
                  immCache.setFileMetadata(url, finalSize);
                }
                this._wholeFileStats.wholeFileFetches += 1;
                if (isPgim) {
                  this._wholeFileStats.pgimFullFetches += 1;
                }
                this._wholeFileStats.bytesFetched += finalBytes.byteLength;
                return {
                  ok: true,
                  buffered: true,
                  bytes: finalBytes,
                  size: finalSize,
                  reused: false,
                  probed: true,
                };
              }
              this._wholeFileStats.fallbacks += 1;
              return { ok: true, buffered: false, size: total, probed: true };
            }

            if (raw.byteLength < rangeEnd + 1 && raw.byteLength <= cap) {
              if (immCache) {
                immCache.setFileMetadata(url, raw.byteLength);
              }
              this._wholeFileStats.wholeFileFetches += 1;
              if (isPgim) {
                this._wholeFileStats.pgimFullFetches += 1;
              }
              this._wholeFileStats.bytesFetched += raw.byteLength;
              return {
                ok: true,
                buffered: true,
                bytes: raw,
                size: raw.byteLength,
                reused: false,
                probed: true,
              };
            }
            this._wholeFileStats.fallbacks += 1;
            return { ok: true, buffered: false, size: null, probed: true };
          }

          // 200 OK fallback (server ignored Range and sent full resource).
          const contentLenHeader = getResponseHeader(res.headers, 'Content-Length');
          const declaredLen = contentLenHeader !== null ? Number(contentLenHeader) : null;
          if (Number.isSafeInteger(declaredLen) && declaredLen > cap) {
            if (res.body && typeof res.body.cancel === 'function') {
              try {
                await res.body.cancel();
              } catch {
                // Ignore stream cancel errors.
              }
            }
            if (immCache) {
              immCache.setFileMetadata(url, declaredLen);
            }
            this._wholeFileStats.fallbacks += 1;
            return { ok: true, buffered: false, size: declaredLen, probed: true };
          }

          const raw = new Uint8Array(await res.arrayBuffer());
          if (immCache) {
            immCache.setFileMetadata(url, raw.byteLength);
          }
          if (raw.byteLength <= cap) {
            this._wholeFileStats.wholeFileFetches += 1;
            if (isPgim) {
              this._wholeFileStats.pgimFullFetches += 1;
            }
            this._wholeFileStats.bytesFetched += raw.byteLength;
            return {
              ok: true,
              buffered: true,
              bytes: raw,
              size: raw.byteLength,
              reused: false,
              probed: true,
            };
          }
          this._wholeFileStats.fallbacks += 1;
          return { ok: true, buffered: false, size: raw.byteLength, probed: true };
        } catch (err) {
          return {
            ok: false,
            buffered: false,
            errno: mapErrorToErrno(err, -2),
            status: err && typeof err.status === 'number' ? err.status : null,
          };
        }
      })();

      this._inFlightWholeFiles.set(url, promise);
      try {
        return await promise;
      } finally {
        if (this._inFlightWholeFiles.get(url) === promise) {
          this._inFlightWholeFiles.delete(url);
        }
      }
    }

    _shouldUseImmutableCache(route) {
      if (!this._immutableCacheEnabled || !route || !route.ok || route.isRpf) {
        return false;
      }
      if (!isImmutableHttpfsPath(route.relPath)) {
        return false;
      }
      if (this._customRangeCacheProvided && !this.immutableCache && !this.fetchImpl) {
        return false;
      }
      return Boolean(this.immutableCache || this._getFetchImpl());
    }

    ensureScheduler() {
      const cache = this.ensureRangeCache();
      if (!this.scheduler) {
        const mod = this._getIoSchedulerModule();
        if (mod && typeof mod.IoPriorityScheduler === 'function') {
          const sleepFn =
            typeof this._schedulerOptions.sleep === 'function'
              ? this._schedulerOptions.sleep
              : typeof setTimeout === 'function'
                ? undefined
                : () => Promise.resolve();
          this.scheduler = new mod.IoPriorityScheduler({
            rangeCache: cache,
            maxConcurrency: this._schedulerOptions.maxConcurrency,
            maxRetries: this._schedulerOptions.maxRetries,
            baseRetryDelayMs: this._schedulerOptions.baseRetryDelayMs,
            sleep: sleepFn,
          });
        }
      }
      return this.scheduler;
    }

    ensureRpfMountTable() {
      if (!this.rpfMountTable) {
        const mod = this._getRpf9VfsModule();
        if (mod && typeof mod.RpfVfsMountTable === 'function') {
          this.rpfMountTable = new mod.RpfVfsMountTable();
        }
      }
      return this.rpfMountTable;
    }

    _findManifestEntry(cleanPath) {
      if (!this.manifest || !Array.isArray(this.manifest.files)) {
        return { known: null, entry: null };
      }
      for (let i = 0; i < this.manifest.files.length; i += 1) {
        const entry = this.manifest.files[i];
        if (!entry || typeof entry.path !== 'string') continue;
        try {
          if (normalizeRelPath(entry.path) === cleanPath) {
            return { known: true, entry };
          }
        } catch {
          // Ignore malformed manifest entries.
        }
      }
      return { known: false, entry: null };
    }

    _scheduledRead(cleanPath, offset, length, options = {}) {
      const scheduler = this.ensureScheduler();
      const priority = options.priority === undefined ? IO_PRIORITY.NORMAL : options.priority;
      if (scheduler) {
        return scheduler.scheduleRead(cleanPath, offset, length, {
          priority,
          signal: options.signal || null,
        });
      }
      const cache = this.ensureRangeCache();
      return cache.read(cleanPath, offset, length);
    }

    prefetchBootset(entries, options = {}) {
      const rawList = Array.isArray(entries)
        ? entries
        : entries && Array.isArray(entries.entries)
          ? entries.entries
          : [];

      if (rawList.length === 0) {
        return Promise.resolve({
          requested: 0,
          scheduled: 0,
          completed: 0,
          cancelled: 0,
          failed: 0,
        });
      }

      const routedEntries = [];
      for (let i = 0; i < rawList.length; i += 1) {
        const entry = rawList[i];
        if (!entry || typeof entry.path !== 'string') {
          continue;
        }
        const route = this._resolveRoute(entry.path);
        if (!route.ok || route.isRpf) {
          continue;
        }
        routedEntries.push(
          Object.assign({}, entry, { path: this._routeToCachePath(route) }),
        );
      }

      if (routedEntries.length === 0) {
        return Promise.resolve({
          requested: rawList.length,
          scheduled: 0,
          completed: 0,
          cancelled: 0,
          failed: 0,
        });
      }

      const scheduler = this.ensureScheduler();
      if (scheduler && typeof scheduler.schedulePrefetchList === 'function') {
        let signal = options.signal || null;
        if (!signal && typeof AbortController === 'function') {
          this._prefetchAbortController = new AbortController();
          signal = this._prefetchAbortController.signal;
        }
        const p = scheduler.schedulePrefetchList(routedEntries, { signal });
        this._prefetchPromise = p;
        return p;
      }

      const cache = this.ensureRangeCache();
      const p = cache.prefetchBootset(routedEntries, options);
      this._prefetchPromise = p;
      return p;
    }

    cancelPrefetch(options = {}) {
      let cancelled = 0;
      if (this.scheduler && typeof this.scheduler.cancelPrefetch === 'function') {
        cancelled = this.scheduler.cancelPrefetch();
      }
      if (options.abortInFlight && this._prefetchAbortController) {
        try {
          this._prefetchAbortController.abort(options.reason || 'Prefetch cancelled');
        } catch {
          // Ignore already-aborted controller errors.
        }
      }
      return cancelled;
    }

    async ensureRpfMounted(archivePath, options = {}) {
      const route = this._resolveRoute(archivePath);
      if (!route.ok || route.isRpf) {
        throw createErrnoError(-22, route.error || `Invalid RPF archive path: ${archivePath}`);
      }
      const cleanArchive = route.relPath;
      const cacheArchivePath = this._routeToCachePath(route);
      const archiveKey = cleanArchive.toLowerCase();

      const existing = this._mountedArchives.get(archiveKey);
      if (existing) {
        return existing;
      }

      const inFlight = this._inFlightMounts.get(archiveKey);
      if (inFlight) {
        return inFlight;
      }

      const rpfMod = this._getRpf9VfsModule();
      if (!rpfMod || typeof rpfMod.RpfArchiveVfs !== 'function') {
        throw createErrnoError(-38, 'PlayGta6Rpf9Vfs unavailable in io_worker');
      }

      const manifestLookup =
        route.mount === 'data'
          ? this._findManifestEntry(cleanArchive)
          : { known: null, entry: null };
      if (manifestLookup.known === false) {
        throw createErrnoError(-2, `RPF archive not found in manifest: ${cleanArchive}`);
      }

      const totalSize =
        options.totalSize !== undefined && options.totalSize !== null
          ? options.totalSize
          : manifestLookup.entry && Number.isSafeInteger(manifestLookup.entry.size)
            ? manifestLookup.entry.size
            : null;

      const mountPriority =
        options.priority === undefined ? IO_PRIORITY.NORMAL : options.priority;

      const mountPromise = (async () => {
        const record = {
          archivePath: cleanArchive,
          archiveKey,
          vfs: null,
          mountInfo: null,
          mounting: true,
          openPriority: mountPriority,
          pendingPriority: null,
        };

        const rangeReader = (offset, length) => {
          const prio = record.mounting
            ? record.openPriority
            : record.pendingPriority !== null
              ? record.pendingPriority
              : IO_PRIORITY.NORMAL;
          return this._scheduledRead(cacheArchivePath, offset, length, { priority: prio });
        };

        try {
          const vfs = await rpfMod.RpfArchiveVfs.open(rangeReader, totalSize, {
            archivePath: cleanArchive,
            decryptBlock: options.decryptBlock || this.decryptBlock || undefined,
            decompressEntry: options.decompressEntry || this.decompressEntry || undefined,
          });
          record.mounting = false;
          record.vfs = vfs;

          const mountTable = this.ensureRpfMountTable();
          if (mountTable) {
            const mountPrefix =
              typeof options.mountPrefix === 'string' ? options.mountPrefix : cleanArchive;
            record.mountInfo = mountTable.mount(mountPrefix, vfs, {
              priority: options.mountPriority,
            });
          }

          this._mountedArchives.set(archiveKey, record);
          return record;
        } catch (err) {
          this._rpfMountErrors += 1;
          throw err;
        } finally {
          if (this._inFlightMounts.get(archiveKey) === mountPromise) {
            this._inFlightMounts.delete(archiveKey);
          }
        }
      })();

      this._inFlightMounts.set(archiveKey, mountPromise);
      return mountPromise;
    }

    _readFromMountedRpf(record, innerPath, offset, length, priority) {
      // RpfArchiveVfs.readFileRange invokes its _readRange callback synchronously before
      // awaiting, so scoping `pendingPriority` around the call attaches the exact caller
      // priority to the scheduled IoPriorityScheduler task without cross-request races.
      record.pendingPriority = priority === undefined ? IO_PRIORITY.NORMAL : priority;
      try {
        return record.vfs.readFileRange(innerPath, offset, length);
      } finally {
        record.pendingPriority = null;
      }
    }

    async read(pathStr, offset, length, options = {}) {
      const priority = options.priority === undefined ? IO_PRIORITY.NORMAL : options.priority;

      if (isRpfVirtualPath(pathStr)) {
        const parsed = parseRpfVirtualPath(pathStr, {
          manifest: this.manifest,
          rpf9VfsModule: this._getRpf9VfsModule(),
        });
        const record = await this.ensureRpfMounted(parsed.archivePath, { priority });
        const mountTable = this.ensureRpfMountTable();
        const resolved = mountTable ? mountTable.resolve(parsed.mountPath) : null;
        if (resolved) {
          if (!resolved.stat.isFile) {
            throw createErrnoError(-21, `Path is a directory, not a file: ${pathStr}`);
          }
          const targetRec =
            (resolved.vfs.archivePath &&
              this._mountedArchives.get(resolved.vfs.archivePath.toLowerCase())) ||
            record;
          return this._readFromMountedRpf(
            targetRec,
            resolved.archivePath,
            offset,
            length,
            priority,
          );
        }
        return this._readFromMountedRpf(record, parsed.innerPath, offset, length, priority);
      }

      const route = this._resolveRoute(pathStr);
      if (!route.ok) {
        throw createErrnoError(-22, route.error);
      }
      return this._scheduledRead(this._routeToCachePath(route), offset, length, {
        priority,
        signal: options.signal || null,
      });
    }

    async stat(pathStr, options = {}) {
      const priority = options.priority === undefined ? IO_PRIORITY.NORMAL : options.priority;
      if (isRpfVirtualPath(pathStr)) {
        const parsed = parseRpfVirtualPath(pathStr, {
          manifest: this.manifest,
          rpf9VfsModule: this._getRpf9VfsModule(),
          allowEmptyInner: true,
        });
        const record = await this.ensureRpfMounted(parsed.archivePath, { priority });
        const mountTable = this.ensureRpfMountTable();
        const st = mountTable
          ? mountTable.stat(parsed.mountPath)
          : record.vfs.stat(parsed.innerPath);
        if (!st) {
          return null;
        }
        return Object.assign({}, st, {
          archivePath: parsed.archivePath,
          innerPath: parsed.innerPath,
        });
      }

      const route = this._resolveRoute(pathStr);
      if (!route.ok) {
        throw createErrnoError(-22, route.error);
      }
      const cleanPath = route.relPath;
      const manifestLookup =
        route.mount === 'data'
          ? this._findManifestEntry(cleanPath)
          : { known: null, entry: null };
      if (manifestLookup.known === true) {
        return {
          path: cleanPath,
          kind: 'file',
          isFile: true,
          isDirectory: false,
          size: manifestLookup.entry.size ?? null,
        };
      }
      if (this.immutableCache && this._shouldUseImmutableCache(route)) {
        const cachedMeta = this.immutableCache.getFileMetadata(route.url);
        if (cachedMeta) {
          return {
            path: cleanPath,
            kind: 'file',
            isFile: true,
            isDirectory: false,
            size: cachedMeta.size,
          };
        }
      }
      return null;
    }

    async listDir(pathStr, options = {}) {
      const priority = options.priority === undefined ? IO_PRIORITY.NORMAL : options.priority;
      if (!isRpfVirtualPath(pathStr)) {
        throw createErrnoError(-22, `listDir requires an rpf:// virtual path: ${pathStr}`);
      }
      const parsed = parseRpfVirtualPath(pathStr, {
        manifest: this.manifest,
        rpf9VfsModule: this._getRpf9VfsModule(),
        allowEmptyInner: true,
      });
      const record = await this.ensureRpfMounted(parsed.archivePath, { priority });
      const mountTable = this.ensureRpfMountTable();
      if (mountTable) {
        return mountTable.listDir(parsed.mountPath);
      }
      return record.vfs.listDir(parsed.innerPath);
    }

    async openHandle(rawPath, options = {}) {
      const probeOnOpen = options.probeOnOpen !== false;
      const modeStr = typeof options.mode === 'string' ? options.mode.trim().toLowerCase() : '';
      const wantsWrite = Boolean(
        options.write ||
        options.create ||
        options.append ||
        modeStr === 'w' ||
        modeStr === 'wb' ||
        modeStr === 'a' ||
        modeStr === 'ab' ||
        modeStr === 'write' ||
        modeStr === 'create' ||
        modeStr === 'append' ||
        modeStr === 'w+' ||
        modeStr === 'r+' ||
        modeStr === 'a+',
      );

      if (wantsWrite) {
        const dumpRoute = resolveDumpWriteRoute(rawPath, {
          allowDumpWrites: this.allowDumpWrites,
          dumpEndpoint: this.dumpEndpoint,
        });
        if (!dumpRoute.ok) {
          this._dumpWriteStats.dumpWriteErrors += 1;
          return dumpRoute.errno;
        }
        const handle = this._nextHandle;
        this._nextHandle += 1;
        const appendMode = Boolean(
          options.append ||
          modeStr === 'a' ||
          modeStr === 'ab' ||
          modeStr === 'a+' ||
          modeStr === 'append',
        );
        this._handles.set(handle, {
          kind: 'dump_write',
          path: dumpRoute.relPath,
          mount: 'dump',
          url: dumpRoute.url,
          appendUrl: dumpRoute.appendUrl,
          writable: true,
          buffered: Boolean(options.buffered),
          appendOnFirstFlush: appendMode,
          hasFlushed: false,
          bytesWritten: 0,
          pendingChunks: [],
          pendingBytes: 0,
          writeQueue: Promise.resolve(),
        });
        this._dumpWriteStats.dumpHandlesOpened += 1;
        return handle;
      }

      if (isRpfVirtualPath(rawPath)) {
        let parsed;
        try {
          parsed = parseRpfVirtualPath(rawPath, {
            manifest: this.manifest,
            rpf9VfsModule: this._getRpf9VfsModule(),
          });
        } catch (err) {
          return mapErrorToErrno(err, -22);
        }

        let record;
        try {
          record = await this.ensureRpfMounted(parsed.archivePath, {
            priority: IO_PRIORITY.CRITICAL,
          });
        } catch (err) {
          return mapErrorToErrno(err, -2);
        }

        const mountTable = this.ensureRpfMountTable();
        const resolved = mountTable ? mountTable.resolve(parsed.mountPath) : null;
        const stat = resolved ? resolved.stat : record.vfs.stat(parsed.innerPath);

        if (!stat) {
          return -2; // ENOENT
        }
        if (!stat.isFile) {
          return -21; // EISDIR
        }

        const vfs = resolved ? resolved.vfs : record.vfs;
        const innerPath = resolved ? resolved.archivePath : stat.path;
        const mountRecord =
          (vfs.archivePath && this._mountedArchives.get(vfs.archivePath.toLowerCase())) || record;

        const handle = this._nextHandle;
        this._nextHandle += 1;
        this._handles.set(handle, {
          kind: 'rpf',
          rawPath,
          archivePath: parsed.archivePath,
          innerPath,
          mountPath: parsed.mountPath,
          vfs,
          mountRecord,
          size: stat.size,
        });
        return handle;
      }

      const route = this._resolveRoute(rawPath);
      if (!route.ok) {
        return -22; // EINVAL
      }

      const cleanPath = route.relPath;
      const cachePath = this._routeToCachePath(route);
      const manifestLookup =
        route.mount === 'data'
          ? this._findManifestEntry(cleanPath)
          : { known: null, entry: null };
      if (manifestLookup.known === false) {
        return -2; // ENOENT
      }

      const useImmutableCache = this._shouldUseImmutableCache(route);
      const useWholeFileBuffer = this._shouldUseWholeFileBuffer(route);
      let resolvedSize =
        manifestLookup.entry && Number.isSafeInteger(manifestLookup.entry.size)
          ? manifestLookup.entry.size
          : null;
      let wholeFileBytes = null;
      let wholeFilePending = false;
      let alreadyProbed = false;

      if (useWholeFileBuffer) {
        const existingWhole = this._activeWholeFiles.get(route.url);
        if (existingWhole) {
          const rec = this._retainWholeFile(route.url, existingWhole.bytes, existingWhole.size);
          wholeFileBytes = rec.bytes;
          resolvedSize = rec.size;
          alreadyProbed = true;
        } else if (probeOnOpen) {
          const wfRes = await this._fetchWholeFile(route.url, cleanPath, resolvedSize);
          if (!wfRes.ok) {
            return typeof wfRes.errno === 'number' && wfRes.errno < 0 ? wfRes.errno : -2;
          }
          if (wfRes.buffered) {
            const rec = this._retainWholeFile(route.url, wfRes.bytes, wfRes.size);
            wholeFileBytes = rec.bytes;
            resolvedSize = rec.size;
            alreadyProbed = true;
          } else {
            if (Number.isSafeInteger(wfRes.size)) {
              resolvedSize = wfRes.size;
            }
            alreadyProbed = Boolean(wfRes.probed);
          }
        } else {
          wholeFilePending = true;
        }
      }

      if (wholeFileBytes === null && !wholeFilePending) {
        if (useImmutableCache) {
          const immCache = this.ensureImmutableCache();
          if (manifestLookup.known === true) {
            immCache.setFileMetadata(route.url, resolvedSize);
          } else if (probeOnOpen && !alreadyProbed) {
            const openRes = await immCache.open(route.url);
            if (!openRes.ok) {
              return typeof openRes.errno === 'number' && openRes.errno < 0 ? openRes.errno : -2;
            }
            if (Number.isSafeInteger(openRes.size)) {
              resolvedSize = openRes.size;
            }
          } else if (resolvedSize === null) {
            const cachedSize = immCache.getFileSize(route.url);
            if (Number.isSafeInteger(cachedSize)) {
              resolvedSize = cachedSize;
            }
          }
        } else {
          try {
            this.ensureRangeCache();
          } catch {
            return -19; // ENODEV
          }

          if (manifestLookup.known === null && probeOnOpen && !alreadyProbed) {
            try {
              await this._scheduledRead(cachePath, 0, 1, { priority: IO_PRIORITY.CRITICAL });
            } catch (err) {
              return mapErrorToErrno(err, -2);
            }
          }
        }
      }

      const handle = this._nextHandle;
      this._nextHandle += 1;
      this._handles.set(handle, {
        kind: 'direct',
        path: cleanPath,
        cachePath,
        mount: route.mount,
        url: route.url,
        useImmutableCache,
        wholeFileBytes,
        wholeFilePending,
        size: resolvedSize,
      });
      return handle;
    }

    _readHandleFast(handle, offset, reqLen) {
      const entry = this._handles.get(handle);
      if (!entry) {
        return null;
      }
      if (entry.kind === 'dump_write') {
        return null;
      }
      if (reqLen === 0) {
        return new Uint8Array(0);
      }
      if (entry.wholeFileBytes !== null && entry.wholeFileBytes !== undefined) {
        this._wholeFileStats.hits += 1;
        const buf = entry.wholeFileBytes;
        if (offset >= buf.byteLength) {
          return new Uint8Array(0);
        }
        const end = Math.min(buf.byteLength, offset + reqLen);
        return buf.subarray(offset, end);
      }
      if (
        entry.kind === 'direct' &&
        entry.useImmutableCache &&
        !entry.wholeFilePending &&
        this.immutableCache
      ) {
        const syncBytes = this.immutableCache.tryReadSync(entry.url, offset, reqLen, {
          copy: false,
        });
        if (syncBytes !== null) {
          if (entry.size === null) {
            const learnedSize = this.immutableCache.getFileSize(entry.url);
            if (Number.isSafeInteger(learnedSize)) {
              entry.size = learnedSize;
            }
          }
          return syncBytes;
        }
      }
      return this.readHandle(handle, offset, reqLen);
    }

    async readHandle(handle, offset, reqLen) {
      const entry = this._handles.get(handle);
      if (!entry) {
        return null;
      }
      if (entry.kind === 'dump_write') {
        throw createErrnoError(-9, 'Cannot read from write-only /pg6_dump/ handle');
      }
      if (reqLen === 0) {
        return new Uint8Array(0);
      }
      if (entry.kind === 'rpf') {
        return this._readFromMountedRpf(
          entry.mountRecord,
          entry.innerPath,
          offset,
          reqLen,
          IO_PRIORITY.CRITICAL,
        );
      }
      if (entry.wholeFilePending) {
        entry.wholeFilePending = false;
        const wfRes = await this._fetchWholeFile(entry.url, entry.path, entry.size);
        if (!wfRes.ok) {
          throw createErrnoError(
            typeof wfRes.errno === 'number' && wfRes.errno < 0 ? wfRes.errno : -5,
            `Failed to fetch ${entry.url}`,
          );
        }
        if (wfRes.buffered) {
          const rec = this._retainWholeFile(entry.url, wfRes.bytes, wfRes.size);
          entry.wholeFileBytes = rec.bytes;
          entry.size = rec.size;
        } else if (Number.isSafeInteger(wfRes.size)) {
          entry.size = wfRes.size;
        }
      }
      if (entry.wholeFileBytes !== null && entry.wholeFileBytes !== undefined) {
        this._wholeFileStats.hits += 1;
        const buf = entry.wholeFileBytes;
        if (offset >= buf.byteLength) {
          return new Uint8Array(0);
        }
        const end = Math.min(buf.byteLength, offset + reqLen);
        return buf.slice(offset, end);
      }
      if (entry.useImmutableCache) {
        const immCache = this.ensureImmutableCache();
        const bytes = await immCache.read(entry.url, offset, reqLen);
        if (entry.size === null) {
          const learnedSize = immCache.getFileSize(entry.url);
          if (Number.isSafeInteger(learnedSize)) {
            entry.size = learnedSize;
          }
        }
        return bytes;
      }
      return this._scheduledRead(entry.cachePath || entry.path, offset, reqLen, {
        priority: IO_PRIORITY.CRITICAL,
      });
    }

    openWriteHandle(rawPath, options = {}) {
      return this.openHandle(rawPath, Object.assign({}, options, { write: true }));
    }

    async _postDumpBytes(url, appendUrl, bytes, useAppend) {
      const fetchFn = this._getFetchImpl();
      if (typeof fetchFn !== 'function') {
        this._dumpWriteStats.dumpWriteErrors += 1;
        throw createErrnoError(-19, 'No fetch implementation available for dump write');
      }
      const targetUrl = useAppend ? appendUrl : url;
      const body =
        bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
          ? bytes
          : bytes.slice();
      let res;
      try {
        res = await fetchFn(targetUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/octet-stream',
          },
          body,
        });
      } catch (err) {
        this._dumpWriteStats.dumpWriteErrors += 1;
        throw createErrnoError(mapErrorToErrno(err, -5), `Dump POST failed: ${err && err.message ? err.message : err}`);
      }

      const status = res && typeof res.status === 'number' ? res.status : 0;
      if (!res || (!res.ok && (status < 200 || status >= 300))) {
        this._dumpWriteStats.dumpWriteErrors += 1;
        const errno =
          status === 400
            ? -22
            : status === 403 || status === 405
              ? -13
              : status === 404
                ? -2
                : status === 413
                  ? -27
                  : -5;
        const err = createErrnoError(errno, `HTTP ${status || 500} posting dump chunk`);
        err.status = status || 500;
        throw err;
      }

      this._dumpWriteStats.dumpWritesCompleted += 1;
      this._dumpWriteStats.dumpBytesWritten += body.byteLength;
      return {
        ok: true,
        status: status || 200,
        bytesWritten: body.byteLength,
        append: Boolean(useAppend),
        url: targetUrl,
      };
    }

    async _flushDumpBytesChunked(entry, payload, forceAppend = false) {
      const chunkLimit = Math.max(1, this._dumpFlushChunkBytes || DEFAULT_DUMP_FLUSH_CHUNK_BYTES);
      if (payload.byteLength === 0) {
        const useAppend = Boolean(forceAppend || entry.hasFlushed || entry.appendOnFirstFlush);
        const res = await this._postDumpBytes(entry.url, entry.appendUrl, payload, useAppend);
        entry.hasFlushed = true;
        return {
          ok: true,
          bytesWritten: 0,
          totalBytesWritten: entry.bytesWritten,
          chunks: 1,
          append: useAppend,
          url: res.url,
          status: res.status,
        };
      }

      let offset = 0;
      let chunks = 0;
      let firstAppend = false;
      let lastUrl = entry.url;
      let lastStatus = 200;

      while (offset < payload.byteLength) {
        const end = Math.min(payload.byteLength, offset + chunkLimit);
        const slice = payload.subarray(offset, end);
        const useAppend = Boolean(
          forceAppend || chunks > 0 || entry.hasFlushed || entry.appendOnFirstFlush,
        );
        if (chunks === 0) {
          firstAppend = useAppend;
        }
        const res = await this._postDumpBytes(entry.url, entry.appendUrl, slice, useAppend);
        entry.hasFlushed = true;
        entry.bytesWritten += slice.byteLength;
        lastUrl = res.url;
        lastStatus = res.status;
        offset = end;
        chunks += 1;
      }

      return {
        ok: true,
        bytesWritten: payload.byteLength,
        totalBytesWritten: entry.bytesWritten,
        chunks,
        append: firstAppend,
        url: lastUrl,
        status: lastStatus,
      };
    }

    _drainPendingChunks(entry) {
      if (!entry.pendingChunks || entry.pendingChunks.length === 0) {
        entry.pendingBytes = 0;
        return new Uint8Array(0);
      }
      if (entry.pendingChunks.length === 1) {
        const single = entry.pendingChunks[0];
        entry.pendingChunks = [];
        entry.pendingBytes = 0;
        return single;
      }
      const combined = new Uint8Array(entry.pendingBytes);
      let cursor = 0;
      for (let i = 0; i < entry.pendingChunks.length; i += 1) {
        combined.set(entry.pendingChunks[i], cursor);
        cursor += entry.pendingChunks[i].byteLength;
      }
      entry.pendingChunks = [];
      entry.pendingBytes = 0;
      return combined;
    }

    _enqueueDumpOp(entry, fn) {
      const next = entry.writeQueue.then(fn, fn);
      entry.writeQueue = next.then(
        () => {},
        () => {},
      );
      return next;
    }

    async writeHandle(handle, data, options = {}) {
      const entry = this._handles.get(handle);
      if (!entry) {
        this._dumpWriteStats.dumpWriteErrors += 1;
        throw createErrnoError(-9, `Invalid file handle: ${handle}`);
      }
      if (entry.kind !== 'dump_write' || !entry.writable) {
        this._dumpWriteStats.dumpWriteErrors += 1;
        throw createErrnoError(-30, 'Read-only handle: writes are only permitted under /pg6_dump/');
      }
      if (!this.allowDumpWrites || !this.dumpEndpoint) {
        this._dumpWriteStats.dumpWriteErrors += 1;
        throw createErrnoError(-30, 'dump writes are disabled (opt-in required)');
      }

      const payload = toUint8ArrayPayload(data);
      if (payload === null) {
        this._dumpWriteStats.dumpWriteErrors += 1;
        throw createErrnoError(-22, 'Dump write data must be a Uint8Array, ArrayBuffer, typed array, or string');
      }

      const shouldBuffer =
        options.flush === false || (entry.buffered && options.flush !== true);
      const forceAppend = Boolean(options.append);

      return this._enqueueDumpOp(entry, async () => {
        if (shouldBuffer) {
          if (payload.byteLength > 0) {
            entry.pendingChunks.push(payload);
            entry.pendingBytes += payload.byteLength;
          }
          if (
            entry.pendingBytes >= this._dumpFlushChunkBytes ||
            entry.pendingBytes >= this._dumpMaxBufferBytes
          ) {
            const drained = this._drainPendingChunks(entry);
            const flushed = await this._flushDumpBytesChunked(entry, drained, forceAppend);
            this._dumpWriteStats.dumpFlushesCompleted += 1;
            return Object.assign({ buffered: false, autoFlushed: true }, flushed);
          }
          return {
            ok: true,
            buffered: true,
            bytesQueued: payload.byteLength,
            pendingBytes: entry.pendingBytes,
            bytesWritten: 0,
            totalBytesWritten: entry.bytesWritten,
          };
        }

        let toSend = payload;
        if (entry.pendingBytes > 0) {
          const prior = this._drainPendingChunks(entry);
          toSend = new Uint8Array(prior.byteLength + payload.byteLength);
          toSend.set(prior, 0);
          toSend.set(payload, prior.byteLength);
        }
        return this._flushDumpBytesChunked(entry, toSend, forceAppend);
      });
    }

    async flushHandle(handle) {
      const entry = this._handles.get(handle);
      if (!entry) {
        this._dumpWriteStats.dumpWriteErrors += 1;
        throw createErrnoError(-9, `Invalid file handle: ${handle}`);
      }
      if (entry.kind !== 'dump_write' || !entry.writable) {
        this._dumpWriteStats.dumpWriteErrors += 1;
        throw createErrnoError(-30, 'Read-only handle: flush is only permitted on /pg6_dump/ handles');
      }
      return this._enqueueDumpOp(entry, async () => {
        if (entry.pendingBytes === 0 && entry.pendingChunks.length === 0) {
          return {
            ok: true,
            bytesWritten: 0,
            totalBytesWritten: entry.bytesWritten,
            chunks: 0,
          };
        }
        const drained = this._drainPendingChunks(entry);
        const flushed = await this._flushDumpBytesChunked(entry, drained, false);
        this._dumpWriteStats.dumpFlushesCompleted += 1;
        return flushed;
      });
    }

    async closeDumpHandle(handle) {
      const entry = this._handles.get(handle);
      if (!entry) {
        return false;
      }
      if (entry.kind !== 'dump_write') {
        return this.closeHandle(handle);
      }
      try {
        await this.flushHandle(handle);
      } finally {
        this._handles.delete(handle);
      }
      return true;
    }

    async writeDump(rawPath, data, options = {}) {
      const dumpRoute = resolveDumpWriteRoute(rawPath, {
        allowDumpWrites: this.allowDumpWrites,
        dumpEndpoint: this.dumpEndpoint,
      });
      if (!dumpRoute.ok) {
        this._dumpWriteStats.dumpWriteErrors += 1;
        throw createErrnoError(dumpRoute.errno, dumpRoute.error);
      }
      const payload = toUint8ArrayPayload(data);
      if (payload === null) {
        this._dumpWriteStats.dumpWriteErrors += 1;
        throw createErrnoError(-22, 'Dump write data must be a Uint8Array, ArrayBuffer, typed array, or string');
      }
      const tempEntry = {
        url: dumpRoute.url,
        appendUrl: dumpRoute.appendUrl,
        appendOnFirstFlush: Boolean(options.append),
        hasFlushed: false,
        bytesWritten: 0,
      };
      const res = await this._flushDumpBytesChunked(tempEntry, payload, Boolean(options.append));
      return Object.assign({ relPath: dumpRoute.relPath }, res);
    }

    closeHandle(handle) {
      const entry = this._handles.get(handle);
      if (!entry) {
        return false;
      }
      if (entry.kind === 'dump_write' && entry.pendingBytes > 0) {
        const drained = this._drainPendingChunks(entry);
        this._handles.delete(handle);
        this._enqueueDumpOp(entry, () => this._flushDumpBytesChunked(entry, drained, false)).catch(
          () => {},
        );
        return true;
      }
      this._handles.delete(handle);
      if (entry.wholeFileBytes !== null && entry.wholeFileBytes !== undefined) {
        entry.wholeFileBytes = null;
        if (entry.url) {
          this._releaseWholeFile(entry.url);
        }
      }
      return true;
    }

    getHandleInfo(handle) {
      const entry = this._handles.get(handle);
      if (!entry) return null;
      if (entry.kind === 'rpf') {
        return {
          kind: 'rpf',
          archivePath: entry.archivePath,
          innerPath: entry.innerPath,
          size: entry.size,
        };
      }
      if (entry.kind === 'dump_write') {
        return {
          kind: 'dump_write',
          path: entry.path,
          mount: 'dump',
          url: entry.url,
          writable: true,
          bytesWritten: entry.bytesWritten,
          pendingBytes: entry.pendingBytes,
          hasFlushed: entry.hasFlushed,
        };
      }
      return {
        kind: 'direct',
        path: entry.path,
        mount: entry.mount || 'data',
        url: entry.url || null,
        size: entry.size,
        wholeFileBuffered: Boolean(entry.wholeFileBytes),
      };
    }

    async detachMailbox() {
      if (this._mailboxHost) {
        const prev = this._mailboxHost;
        this._mailboxHost = null;
        await prev.stop();
      }
      for (const entry of this._handles.values()) {
        if (entry && entry.wholeFileBytes) {
          entry.wholeFileBytes = null;
        }
      }
      this._handles.clear();
      this._activeWholeFiles.clear();
      this._inFlightWholeFiles.clear();
    }

    async attachMailbox(sab, options = {}) {
      const mailboxMod = this._getHttpfsMailboxModule();
      if (!mailboxMod || typeof mailboxMod.HttpfsMailboxHost !== 'function') {
        throw new Error('PlayGta6HttpfsMailbox unavailable in io_worker');
      }
      await this.detachMailbox();

      const probeOnOpen = options.probeOnOpen !== false;
      const BaseHost = mailboxMod.HttpfsMailboxHost;
      const HttpfsStatus = mailboxMod.HttpfsStatus;
      const controller = this;

      class IoWorkerMailboxHost extends BaseHost {
        _handleOpen(lengthBig, cap) {
          if (lengthBig === 0n || lengthBig > BigInt(cap)) {
            this.completeResponse(-22n, HttpfsStatus.Error);
            return;
          }

          const pathLen = Number(lengthBig);
          const copy = new Uint8Array(pathLen);
          copy.set(this._payloadView.subarray(0, pathLen));
          const rawPath = utf8Decoder.decode(copy);
          const openFlags = this._headerView.getBigUint64(mailboxMod.HttpfsOffsets.OFFSET, true);
          const writeFlag = (openFlags & HTTPFS_OPEN_FLAG_WRITE) !== 0n;
          const appendFlag = (openFlags & HTTPFS_OPEN_FLAG_APPEND) !== 0n;

          const pending = controller.openHandle(rawPath, {
            probeOnOpen,
            write: writeFlag,
            append: appendFlag,
          });
          if (pending && typeof pending.then === 'function') {
            return pending.then((handle) => {
              this._finishOpenResult(handle);
            });
          }
          this._finishOpenResult(pending);
        }

        _dispatchOnce() {
          const status = Atomics.load(this._statusView, 0);
          if (status !== HttpfsStatus.RequestReady) {
            return false;
          }
          const op = this._headerView.getUint32(mailboxMod.HttpfsOffsets.OP, true);
          if (op === HTTPFS_OP_WRITE) {
            this._stats.requests += 1;
            const handle = this._headerView.getInt32(mailboxMod.HttpfsOffsets.HANDLE, true);
            const offsetBig = this._headerView.getBigUint64(mailboxMod.HttpfsOffsets.OFFSET, true);
            const lengthBig = this._headerView.getBigUint64(mailboxMod.HttpfsOffsets.LENGTH, true);
            const cap = this.payloadCapacity;
            if (lengthBig > BigInt(cap)) {
              controller._dumpWriteStats.dumpWriteErrors += 1;
              this.completeResponse(-22n, HttpfsStatus.Error);
              this._stats.syncFastPathHits += 1;
              return true;
            }
            const writeLen = Number(lengthBig);
            const copy = new Uint8Array(writeLen);
            if (writeLen > 0) {
              copy.set(this._payloadView.subarray(0, writeLen));
            }
            const appendFlag = (offsetBig & HTTPFS_OPEN_FLAG_APPEND) !== 0n;
            this._stats.asyncPathHits += 1;
            return controller.writeHandle(handle, copy, { append: appendFlag }).then(
              (res) => {
                const written = res && typeof res.bytesWritten === 'number' ? res.bytesWritten : writeLen;
                this.completeResponse(BigInt(written), HttpfsStatus.ResponseReady);
                return true;
              },
              (err) => {
                this.completeResponse(BigInt(mapErrorToErrno(err, -5)), HttpfsStatus.Error);
                return true;
              },
            );
          }
          return super._dispatchOnce();
        }
      }

      this._mailboxHost = new IoWorkerMailboxHost(sab, {
        getRangeCache: () => this.ensureRangeCache(),
        getManifest: () => this.manifest,
        probeOnOpen,
        forcePoll: options.forcePoll,
        fallbackPollMs: options.fallbackPollMs,
        open: (relPath) => this.openHandle(relPath, { probeOnOpen }),
        read: (handle, offset, reqLen) => this._readHandleFast(handle, offset, reqLen),
        close: (handle) => {
          const entry = this._handles.get(handle);
          if (entry && entry.kind === 'dump_write') {
            return this.closeDumpHandle(handle);
          }
          return this.closeHandle(handle);
        },
      });

      this._mailboxHost.start().catch((err) => {
        if (this.postMessage) {
          this.postMessage({ error: `httpfs_mailbox: ${err}` });
        }
      });

      return this._mailboxHost;
    }

    getStats() {
      const cacheStats = this.rangeCache ? this.rangeCache.getStats() : emptyRangeCacheStats();
      const immutableCacheStats = this.immutableCache
        ? this.immutableCache.getStats()
        : emptyImmutableCacheStats({
            chunkSize: this._immutableCacheOptions.chunkSize,
            maxBytes: this._immutableCacheOptions.maxBytes,
            slabSize: this._immutableCacheOptions.slabSize,
            slabMaxBytes: this._immutableCacheOptions.slabMaxBytes,
            slabCacheEnabled: this._immutableCacheOptions.slabCacheEnabled,
            slabProfile: this._immutableCacheOptions.slabProfile,
            prefetchAheadSlabs: this._immutableCacheOptions.prefetchAheadSlabs,
            maxConcurrentSlabFetches: this._immutableCacheOptions.maxConcurrentSlabFetches,
            sequentialThresholdReads: this._immutableCacheOptions.sequentialThresholdReads,
          });
      const mailboxStats =
        this._mailboxHost && typeof this._mailboxHost.getStats === 'function'
          ? this._mailboxHost.getStats()
          : emptyMailboxStats();
      let openDumpHandles = 0;
      let dumpBufferedBytes = 0;
      for (const entry of this._handles.values()) {
        if (entry && entry.kind === 'dump_write') {
          openDumpHandles += 1;
          dumpBufferedBytes += entry.pendingBytes || 0;
        }
      }
      const dumpWriteStats = {
        enabled: Boolean(this.allowDumpWrites && this.dumpEndpoint),
        dumpWritesCompleted: this._dumpWriteStats.dumpWritesCompleted,
        dumpBytesWritten: this._dumpWriteStats.dumpBytesWritten,
        dumpWriteErrors: this._dumpWriteStats.dumpWriteErrors,
        dumpHandlesOpened: this._dumpWriteStats.dumpHandlesOpened,
        dumpFlushesCompleted: this._dumpWriteStats.dumpFlushesCompleted,
        openDumpHandles,
        bufferedBytes: dumpBufferedBytes,
      };
      let activeWholeFileBytes = 0;
      for (const rec of this._activeWholeFiles.values()) {
        if (rec && rec.bytes) {
          activeWholeFileBytes += rec.bytes.byteLength;
        }
      }
      const wholeFileStats = {
        pgimFullFetches: this._wholeFileStats.pgimFullFetches,
        wholeFileFetches: this._wholeFileStats.wholeFileFetches,
        hits: this._wholeFileStats.hits,
        fallbacks: this._wholeFileStats.fallbacks,
        releases: this._wholeFileStats.releases,
        bytesFetched: this._wholeFileStats.bytesFetched,
        activeEntries: this._activeWholeFiles.size,
        activeBytes: activeWholeFileBytes,
        maxBytes: this._wholeFileMaxBytes,
      };
      const schedulerStats = this.scheduler ? this.scheduler.getStats() : emptySchedulerStats();
      const mounts = this.rpfMountTable ? this.rpfMountTable.listMounts() : [];
      const mountedArchives = mounts.map((m) => {
        const rec = m.archivePath ? this._mountedArchives.get(m.archivePath.toLowerCase()) : null;
        const header = rec && rec.vfs && rec.vfs.header ? rec.vfs.header : null;
        return {
          id: m.id,
          prefix: m.prefix,
          archivePath: m.archivePath,
          priority: m.priority,
          order: m.order,
          magic: header ? header.magic : null,
          rawMagic: header ? header.rawMagic : null,
          magicEndian: header ? header.magicEndian : null,
          entryCount: header ? header.entryCount : null,
        };
      });

      return {
        // Backwards-compatible top-level RangeCache fields
        hits: cacheStats.hits,
        misses: cacheStats.misses,
        opfsHits: cacheStats.opfsHits,
        evictions: cacheStats.evictions,
        bytesCached: cacheStats.bytesCached,
        entries: cacheStats.entries,
        // Top-level HTTPFS acceleration counters
        pgimFullFetches: wholeFileStats.pgimFullFetches,
        slabCacheHits: immutableCacheStats.slabCacheHits,
        slabCacheMisses: immutableCacheStats.slabCacheMisses,
        syncFastPathHits: mailboxStats.syncFastPathHits,
        // Top-level opt-in /pg6_dump/* write forwarding telemetry
        dumpWritesCompleted: dumpWriteStats.dumpWritesCompleted,
        dumpBytesWritten: dumpWriteStats.dumpBytesWritten,
        dumpWriteErrors: dumpWriteStats.dumpWriteErrors,
        // Detailed subsystem telemetry
        rangeCache: cacheStats,
        immutableCache: immutableCacheStats,
        wholeFileCache: wholeFileStats,
        mailbox: mailboxStats,
        dumpWrites: dumpWriteStats,
        scheduler: schedulerStats,
        rpf: {
          mountedCount: mountedArchives.length,
          archives: mountedArchives,
          mountErrors: this._rpfMountErrors,
        },
        mountedRpfCount: mountedArchives.length,
        openHandles: this._handles.size,
      };
    }

    async init(options = {}) {
      const opts = typeof options === 'string' ? { dataBase: options } : options || {};
      this.cancelPrefetch({ abortInFlight: true });
      for (const entry of this._handles.values()) {
        if (entry && entry.wholeFileBytes) {
          entry.wholeFileBytes = null;
        }
      }
      this._handles.clear();
      this._activeWholeFiles.clear();
      this._inFlightWholeFiles.clear();
      this._wholeFileStats = {
        pgimFullFetches: 0,
        wholeFileFetches: 0,
        hits: 0,
        fallbacks: 0,
        releases: 0,
        bytesFetched: 0,
      };
      this._dumpWriteStats = {
        dumpWritesCompleted: 0,
        dumpBytesWritten: 0,
        dumpWriteErrors: 0,
        dumpHandlesOpened: 0,
        dumpFlushesCompleted: 0,
      };
      this._mountedArchives.clear();
      this._inFlightMounts.clear();
      this._rpfMountErrors = 0;
      this._prefetchPromise = null;
      this._prefetchAbortController = null;

      this.dataBase = normalizeDataBase(opts.dataBase !== undefined ? opts.dataBase : this.dataBase);
      if (opts.bootBase !== undefined) {
        this.bootBase = normalizeOptionalBase(opts.bootBase);
      }
      if (opts.gameBase !== undefined) {
        this.gameBase = normalizeOptionalBase(opts.gameBase);
      }
      if (opts.shadersBase !== undefined) {
        this.shadersBase = normalizeOptionalBase(opts.shadersBase);
      }
      if (opts.allowDumpWrites !== undefined || opts.dumpEndpoint !== undefined) {
        const dumpCfg = resolveDumpConfig({
          allowDumpWrites: opts.allowDumpWrites,
          dumpEndpoint: opts.dumpEndpoint,
        });
        this.allowDumpWrites = dumpCfg.allowDumpWrites;
        this.dumpEndpoint = dumpCfg.dumpEndpoint;
      }
      if (Number.isFinite(opts.dumpMaxBufferBytes) && opts.dumpMaxBufferBytes > 0) {
        this._dumpMaxBufferBytes = Math.floor(opts.dumpMaxBufferBytes);
      }
      if (Number.isFinite(opts.dumpFlushChunkBytes) && opts.dumpFlushChunkBytes > 0) {
        this._dumpFlushChunkBytes = Math.min(
          Math.floor(opts.dumpFlushChunkBytes),
          this._dumpMaxBufferBytes,
        );
      }
      if (opts.wholeFileBufferEnabled !== undefined) {
        this._wholeFileBufferEnabled = opts.wholeFileBufferEnabled !== false;
      }
      const nextWholeMax =
        opts.wholeFileMaxBytes !== undefined ? opts.wholeFileMaxBytes : opts.pgimMaxBytes;
      if (Number.isFinite(nextWholeMax) && nextWholeMax >= 0) {
        this._wholeFileMaxBytes = Math.floor(nextWholeMax);
      }
      if (typeof opts.fetchImpl === 'function') {
        this.fetchImpl = opts.fetchImpl;
      }
      if (typeof opts.decryptBlock === 'function') {
        this.decryptBlock = opts.decryptBlock;
      }
      if (typeof opts.decompressEntry === 'function') {
        this.decompressEntry = opts.decompressEntry;
      }

      this.manifest = opts.manifest || null;

      const rangeCacheMod = this._getRangeCacheModule();
      if (!rangeCacheMod || typeof rangeCacheMod.RangeCache !== 'function') {
        throw new Error('PlayGta6RangeCache unavailable in io_worker');
      }

      let opfsDir = null;
      if (opts.opfsDir !== undefined) {
        opfsDir = opts.opfsDir;
      } else if (this._cacheOptions.opfsDir !== undefined) {
        opfsDir = this._cacheOptions.opfsDir;
      } else if (typeof rangeCacheMod.openOpfsCacheDir === 'function') {
        opfsDir = await rangeCacheMod.openOpfsCacheDir('playgta6-cache');
      }

      const fetchFn = this._getFetchImpl(opts.fetchImpl);
      const routedFetchFn = this._createRoutedFetch(fetchFn);

      if (opts.immutableCacheEnabled !== undefined) {
        this._immutableCacheEnabled = opts.immutableCacheEnabled !== false;
      }
      this._customRangeCacheProvided = Boolean(opts.rangeCache);
      this._hasRoutedRangeCache = !opts.rangeCache;
      const resolvedSlabSize =
        opts.slabSize !== undefined
          ? opts.slabSize
          : opts.immutableSlabSize !== undefined
            ? opts.immutableSlabSize
            : this._immutableCacheOptions.slabSize;
      const resolvedSlabMaxBytes =
        opts.slabMaxBytes !== undefined
          ? opts.slabMaxBytes
          : opts.immutableSlabMaxBytes !== undefined
            ? opts.immutableSlabMaxBytes
            : this._immutableCacheOptions.slabMaxBytes;
      const resolvedSlabEnabled =
        opts.slabCacheEnabled !== undefined
          ? opts.slabCacheEnabled
          : this._immutableCacheOptions.slabCacheEnabled;
      const resolvedSlabProfile =
        opts.slabProfile !== undefined
          ? opts.slabProfile
          : this._immutableCacheOptions.slabProfile;
      const resolvedPrefetchAhead =
        opts.prefetchAheadSlabs !== undefined
          ? opts.prefetchAheadSlabs
          : this._immutableCacheOptions.prefetchAheadSlabs;
      const resolvedMaxConcurrentSlabFetches =
        opts.maxConcurrentSlabFetches !== undefined
          ? opts.maxConcurrentSlabFetches
          : this._immutableCacheOptions.maxConcurrentSlabFetches;
      const resolvedSequentialThreshold =
        opts.sequentialThresholdReads !== undefined
          ? opts.sequentialThresholdReads
          : this._immutableCacheOptions.sequentialThresholdReads;

      this.immutableCache =
        opts.immutableCache ||
        new ImmutableHttpfsCache({
          chunkSize:
            opts.immutableCacheChunkSize !== undefined
              ? opts.immutableCacheChunkSize
              : this._immutableCacheOptions.chunkSize,
          maxBytes:
            opts.immutableCacheMaxBytes !== undefined
              ? opts.immutableCacheMaxBytes
              : this._immutableCacheOptions.maxBytes,
          slabSize: resolvedSlabSize,
          slabMaxBytes: resolvedSlabMaxBytes,
          slabCacheEnabled: resolvedSlabEnabled,
          slabProfile: resolvedSlabProfile,
          prefetchAheadSlabs: resolvedPrefetchAhead,
          maxConcurrentSlabFetches: resolvedMaxConcurrentSlabFetches,
          sequentialThresholdReads: resolvedSequentialThreshold,
          negativeCacheTtlMs:
            opts.negativeCacheTtlMs !== undefined
              ? opts.negativeCacheTtlMs
              : this._immutableCacheOptions.negativeCacheTtlMs,
          negativeCacheMaxEntries:
            opts.negativeCacheMaxEntries !== undefined
              ? opts.negativeCacheMaxEntries
              : this._immutableCacheOptions.negativeCacheMaxEntries,
          metadataMaxEntries:
            opts.metadataMaxEntries !== undefined
              ? opts.metadataMaxEntries
              : this._immutableCacheOptions.metadataMaxEntries,
          now:
            typeof opts.now === 'function' ? opts.now : this._immutableCacheOptions.now,
          fetchImpl: fetchFn,
        });
      this.rangeCache =
        opts.rangeCache ||
        new rangeCacheMod.RangeCache({
          dataBase: this.dataBase,
          chunkSize:
            opts.chunkSize !== undefined ? opts.chunkSize : this._cacheOptions.chunkSize,
          maxBytes: opts.maxBytes !== undefined ? opts.maxBytes : this._cacheOptions.maxBytes,
          maxPrefetchConcurrency:
            opts.maxPrefetchConcurrency !== undefined
              ? opts.maxPrefetchConcurrency
              : this._cacheOptions.maxPrefetchConcurrency,
          opfsDir,
          fetchImpl: routedFetchFn,
        });

      const schedulerMod = this._getIoSchedulerModule();
      if (opts.scheduler) {
        this.scheduler = opts.scheduler;
      } else if (schedulerMod && typeof schedulerMod.IoPriorityScheduler === 'function') {
        const sleepFn =
          typeof opts.sleep === 'function'
            ? opts.sleep
            : typeof this._schedulerOptions.sleep === 'function'
              ? this._schedulerOptions.sleep
              : typeof setTimeout === 'function'
                ? undefined
                : () => Promise.resolve();
        this.scheduler = new schedulerMod.IoPriorityScheduler({
          rangeCache: this.rangeCache,
          maxConcurrency:
            opts.maxConcurrency !== undefined
              ? opts.maxConcurrency
              : this._schedulerOptions.maxConcurrency,
          maxRetries:
            opts.maxRetries !== undefined ? opts.maxRetries : this._schedulerOptions.maxRetries,
          baseRetryDelayMs:
            opts.baseRetryDelayMs !== undefined
              ? opts.baseRetryDelayMs
              : this._schedulerOptions.baseRetryDelayMs,
          sleep: sleepFn,
        });
      } else {
        this.scheduler = null;
      }

      const rpfMod = this._getRpf9VfsModule();
      if (opts.rpfMountTable) {
        this.rpfMountTable = opts.rpfMountTable;
      } else if (rpfMod && typeof rpfMod.RpfVfsMountTable === 'function') {
        this.rpfMountTable = new rpfMod.RpfVfsMountTable();
      } else {
        this.rpfMountTable = null;
      }

      let manifestPart = 'no data-manifest.json yet';
      if (this.manifest) {
        manifestPart = `${this.manifest.file_count || (Array.isArray(this.manifest.files) ? this.manifest.files.length : 0)} files indexed`;
      } else if (fetchFn) {
        try {
          const res = await fetchFn(`${this.dataBase}data-manifest.json`);
          if (res.ok) {
            this.manifest = await res.json();
            manifestPart = `${this.manifest.file_count || (Array.isArray(this.manifest.files) ? this.manifest.files.length : 0)} files indexed`;
          }
        } catch {
          // Manifest is optional until the user generates local-data/data-manifest.json.
        }
      }

      let bootsetCount = 0;
      let bootEntries = null;
      if (opts.bootset !== undefined) {
        bootEntries = Array.isArray(opts.bootset)
          ? opts.bootset
          : opts.bootset && Array.isArray(opts.bootset.entries)
            ? opts.bootset.entries
            : [];
      } else if (fetchFn) {
        try {
          const bootRes = await fetchFn(`${this.dataBase}bootset.json`);
          if (bootRes.ok) {
            const bootJson = await bootRes.json();
            bootEntries = Array.isArray(bootJson)
              ? bootJson
              : bootJson && Array.isArray(bootJson.entries)
                ? bootJson.entries
                : [];
          }
        } catch {
          // bootset.json is optional; absence must never fail boot.
        }
      }

      if (Array.isArray(bootEntries) && bootEntries.length > 0) {
        bootsetCount = bootEntries.length;
        // Prefetch hot initial ranges at lowest (PREFETCH) priority without blocking worker readiness.
        this.prefetchBootset(bootEntries).catch(() => {});
      }

      const cacheTier = opfsDir ? 'OPFS+LRU' : 'memory LRU';
      const bootPart = bootsetCount > 0 ? `, prefetching ${bootsetCount} bootset ranges` : '';
      const summary = `${manifestPart} (${cacheTier}${bootPart})`;

      const readyPayload = {
        ready: true,
        summary,
        opfs: Boolean(opfsDir),
        bootsetEntries: bootsetCount,
        scheduler: Boolean(this.scheduler),
        rpfVfs: Boolean(this.rpfMountTable),
        dataBase: this.dataBase,
        bootBase: this.bootBase,
        gameBase: this.gameBase,
        shadersBase: this.shadersBase,
        allowDumpWrites: this.allowDumpWrites,
        dumpEndpoint: this.dumpEndpoint,
      };

      if (this.postMessage) {
        this.postMessage(readyPayload);
      }

      return readyPayload;
    }

    async handleMessage(msg) {
      const m = msg || {};
      try {
        if (m.init) {
          if (m.mailboxSab) {
            await this.attachMailbox(m.mailboxSab, m);
          }
          await this.init(m);
        } else if (m.attachMailbox) {
          await this.attachMailbox(m.sab || m.mailboxSab, m);
          const reply = { mailboxAttached: true };
          if (m.id !== undefined) reply.id = m.id;
          if (this.postMessage) this.postMessage(reply);
        } else if (m.detachMailbox) {
          await this.detachMailbox();
          const reply = { mailboxDetached: true };
          if (m.id !== undefined) reply.id = m.id;
          if (this.postMessage) this.postMessage(reply);
        } else if (m.read) {
          const bytes = await this.read(m.path, m.offset, m.length, {
            priority: m.priority,
          });
          const transferable = ensureTransferableBytes(bytes);
          if (this.postMessage) {
            this.postMessage({ id: m.id, bytes: transferable }, [transferable.buffer]);
          }
        } else if (m.openWrite || m.openDump) {
          const handle = await this.openWriteHandle(m.path, {
            append: Boolean(m.append),
            buffered: Boolean(m.buffered),
            mode: m.mode,
          });
          if (this.postMessage) {
            if (handle < 0) {
              this.postMessage({
                id: m.id,
                ok: false,
                handle,
                errno: handle,
                error: handle === -22 ? 'path traversal or invalid dump path' : 'dump write refused (read-only or disabled)',
              });
            } else {
              this.postMessage({
                id: m.id,
                ok: true,
                handle,
                info: this.getHandleInfo(handle),
              });
            }
          }
        } else if (m.writeDump || m.write) {
          const payload = m.bytes !== undefined ? m.bytes : m.data;
          const result =
            m.handle !== undefined
              ? await this.writeHandle(m.handle, payload, {
                  append: m.append,
                  flush: m.flush,
                })
              : await this.writeDump(m.path, payload, { append: m.append });
          if (this.postMessage) {
            this.postMessage(Object.assign({ id: m.id }, result));
          }
        } else if (m.flushDump) {
          const result = await this.flushHandle(m.handle);
          if (this.postMessage) {
            this.postMessage(Object.assign({ id: m.id }, result));
          }
        } else if (m.closeDump) {
          const closed = await this.closeDumpHandle(m.handle);
          if (this.postMessage) {
            this.postMessage({ id: m.id, ok: closed, closed });
          }
        } else if (m.tuneSlabCache) {
          const target =
            typeof m.tuneSlabCache === 'string'
              ? m.tuneSlabCache
              : m.profile || m.config || m;
          const slabConfig = this.tuneSlabCache(target);
          if (this.postMessage) {
            this.postMessage({ id: m.id, ok: true, slabConfig });
          }
        } else if (m.prefetchSlabs) {
          const prefetch = await this.prefetchVideoSlabs(m.path, m.offset || 0, m.count);
          if (this.postMessage) {
            this.postMessage({ id: m.id, ok: true, prefetch });
          }
        } else if (m.stat) {
          const info = await this.stat(m.path, { priority: m.priority });
          if (this.postMessage) {
            this.postMessage({ id: m.id, stat: info });
          }
        } else if (m.listDir) {
          const entries = await this.listDir(m.path, { priority: m.priority });
          if (this.postMessage) {
            this.postMessage({ id: m.id, entries });
          }
        } else if (m.mountRpf) {
          const rec = await this.ensureRpfMounted(m.archivePath || m.path, {
            mountPrefix: m.mountPrefix,
            totalSize: m.totalSize,
            priority: m.priority,
            mountPriority: m.mountPriority,
          });
          if (this.postMessage) {
            this.postMessage({
              id: m.id,
              mounted: {
                archivePath: rec.archivePath,
                mountInfo: rec.mountInfo,
                magic: rec.vfs && rec.vfs.header ? rec.vfs.header.magic : null,
                rawMagic: rec.vfs && rec.vfs.header ? rec.vfs.header.rawMagic : null,
                entryCount: rec.vfs && rec.vfs.header ? rec.vfs.header.entryCount : 0,
              },
            });
          }
        } else if (m.cancelPrefetch) {
          const cancelled = this.cancelPrefetch({ abortInFlight: Boolean(m.abortInFlight) });
          if (this.postMessage) {
            this.postMessage({ id: m.id, cancelled });
          }
        } else if (m.stats) {
          const stats = this.getStats();
          if (this.postMessage) {
            this.postMessage({ id: m.id, stats });
          }
        }
      } catch (err) {
        if (this.postMessage) {
          const errPayload = { id: m.id, error: String(err) };
          if (err && typeof err.errno === 'number') {
            errPayload.errno = err.errno;
          }
          this.postMessage(errPayload);
        }
      }
    }
  }

  function createIoWorkerHandler(options = {}) {
    const controller = new IoWorkerController(options);
    const handler = (evOrMsg) => {
      const data =
        evOrMsg &&
        typeof evOrMsg === 'object' &&
        Object.prototype.hasOwnProperty.call(evOrMsg, 'data') &&
        evOrMsg.data &&
        typeof evOrMsg.data === 'object'
          ? evOrMsg.data
          : evOrMsg;
      return controller.handleMessage(data);
    };
    handler.controller = controller;
    return handler;
  }

  const defaultController = new IoWorkerController({
    postMessage:
      typeof self !== 'undefined' && typeof self.postMessage === 'function'
        ? (msg, transfer) =>
            transfer ? self.postMessage(msg, transfer) : self.postMessage(msg)
        : null,
  });

  if (typeof self !== 'undefined' && typeof self.postMessage === 'function') {
    self.onmessage = (ev) => defaultController.handleMessage(ev && ev.data);
  }

  const exportedApi = {
    IO_PRIORITY,
    RPF_URI_PREFIX,
    DUMP_VIRTUAL_PREFIX,
    DEFAULT_DUMP_ENDPOINT,
    DEFAULT_DUMP_MAX_BUFFER_BYTES,
    DEFAULT_DUMP_FLUSH_CHUNK_BYTES,
    HTTPFS_OP_WRITE,
    HTTPFS_OPEN_FLAG_WRITE,
    HTTPFS_OPEN_FLAG_APPEND,
    DEFAULT_IMMUTABLE_CHUNK_SIZE,
    DEFAULT_IMMUTABLE_MAX_BYTES,
    DEFAULT_WHOLE_FILE_MAX_BYTES,
    DEFAULT_SLAB_SIZE,
    DEFAULT_SLAB_MAX_BYTES,
    MAX_SLAB_SIZE,
    MAX_SLAB_CACHE_BYTES,
    DEFAULT_SLAB_PREFETCH_AHEAD,
    DEFAULT_MAX_CONCURRENT_SLAB_FETCHES,
    SLAB_TUNING_PROFILES,
    VIDEO_STREAM_SLAB_PRESET,
    DEFAULT_NEGATIVE_CACHE_TTL_MS,
    ImmutableHttpfsCache,
    IoWorkerController,
    createIoWorkerHandler,
    resolveHttpfsRoute,
    normalizeDumpEndpoint,
    resolveDumpConfig,
    isDumpVirtualPath,
    resolveDumpWriteRoute,
    createSlabCacheTuningProfile,
    isImmutableHttpfsPath,
    isWholeFileBufferedPath,
    isVideoStreamPath,
    parseContentRangeHeader,
    isRpfVirtualPath,
    parseRpfVirtualPath,
    emptyStats,
    emptyRangeCacheStats,
    emptyImmutableCacheStats,
    emptyWholeFileCacheStats,
    emptyMailboxStats,
    emptyDumpWriteStats,
    emptySchedulerStats,
    getDefaultController: () => defaultController,
  };

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6IoWorker = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})();
