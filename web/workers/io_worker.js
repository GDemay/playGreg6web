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

  function normalizeDataBase(base) {
    const raw = typeof base === 'string' && base.length > 0 ? base : '/data/';
    return raw.endsWith('/') ? raw : `${raw}/`;
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
    if (err.status === 404 || err.statusCode === 404) {
      return -2; // ENOENT
    }
    const msg = typeof err.message === 'string' ? err.message : '';
    if (/\bHTTP\s+404\b/i.test(msg) || /\bnot found\b/i.test(msg)) {
      return -2; // ENOENT
    }
    if (err.name === 'RpfFormatError') {
      return -5; // EIO
    }
    return defaultErrno;
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

  function emptyStats() {
    const cacheStats = emptyRangeCacheStats();
    return {
      hits: cacheStats.hits,
      misses: cacheStats.misses,
      opfsHits: cacheStats.opfsHits,
      evictions: cacheStats.evictions,
      bytesCached: cacheStats.bytesCached,
      entries: cacheStats.entries,
      rangeCache: cacheStats,
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
      this.manifest = options.manifest || null;
      this.rangeCache = options.rangeCache || null;
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
      this._schedulerOptions = {
        maxConcurrency: options.maxConcurrency,
        maxRetries: options.maxRetries,
        baseRetryDelayMs: options.baseRetryDelayMs,
        sleep: options.sleep,
      };

      this._mailboxHost = null;
      this._handles = new Map();
      this._nextHandle = 1;
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

    ensureRangeCache() {
      if (!this.rangeCache) {
        const mod = this._getRangeCacheModule();
        if (!mod || typeof mod.RangeCache !== 'function') {
          throw new Error('PlayGta6RangeCache unavailable in io_worker');
        }
        this.rangeCache = new mod.RangeCache({
          dataBase: this.dataBase,
          chunkSize: this._cacheOptions.chunkSize,
          maxBytes: this._cacheOptions.maxBytes,
          maxPrefetchConcurrency: this._cacheOptions.maxPrefetchConcurrency,
          opfsDir: this._cacheOptions.opfsDir || null,
          fetchImpl: this._getFetchImpl(),
        });
      }
      return this.rangeCache;
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

      const scheduler = this.ensureScheduler();
      if (scheduler && typeof scheduler.schedulePrefetchList === 'function') {
        let signal = options.signal || null;
        if (!signal && typeof AbortController === 'function') {
          this._prefetchAbortController = new AbortController();
          signal = this._prefetchAbortController.signal;
        }
        const p = scheduler.schedulePrefetchList(rawList, { signal });
        this._prefetchPromise = p;
        return p;
      }

      const cache = this.ensureRangeCache();
      const p = cache.prefetchBootset(rawList, options);
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
      const cleanArchive = normalizeRelPath(archivePath);
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

      const manifestLookup = this._findManifestEntry(cleanArchive);
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
          return this._scheduledRead(cleanArchive, offset, length, { priority: prio });
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

      const cleanPath = normalizeRelPath(pathStr);
      return this._scheduledRead(cleanPath, offset, length, {
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

      const cleanPath = normalizeRelPath(pathStr);
      const manifestLookup = this._findManifestEntry(cleanPath);
      if (manifestLookup.known === true) {
        return {
          path: cleanPath,
          kind: 'file',
          isFile: true,
          isDirectory: false,
          size: manifestLookup.entry.size ?? null,
        };
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

      let cleanPath;
      try {
        cleanPath = normalizeRelPath(rawPath);
      } catch {
        return -22; // EINVAL
      }

      const manifestLookup = this._findManifestEntry(cleanPath);
      if (manifestLookup.known === false) {
        return -2; // ENOENT
      }

      try {
        this.ensureRangeCache();
      } catch {
        return -19; // ENODEV
      }

      if (manifestLookup.known === null && probeOnOpen) {
        try {
          await this._scheduledRead(cleanPath, 0, 1, { priority: IO_PRIORITY.CRITICAL });
        } catch (err) {
          return mapErrorToErrno(err, -2);
        }
      }

      const handle = this._nextHandle;
      this._nextHandle += 1;
      this._handles.set(handle, {
        kind: 'direct',
        path: cleanPath,
        size:
          manifestLookup.entry && Number.isSafeInteger(manifestLookup.entry.size)
            ? manifestLookup.entry.size
            : null,
      });
      return handle;
    }

    async readHandle(handle, offset, reqLen) {
      const entry = this._handles.get(handle);
      if (!entry) {
        return null;
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
      return this._scheduledRead(entry.path, offset, reqLen, {
        priority: IO_PRIORITY.CRITICAL,
      });
    }

    closeHandle(handle) {
      return this._handles.delete(handle);
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
      return {
        kind: 'direct',
        path: entry.path,
        size: entry.size,
      };
    }

    async detachMailbox() {
      if (this._mailboxHost) {
        const prev = this._mailboxHost;
        this._mailboxHost = null;
        await prev.stop();
      }
      this._handles.clear();
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
        async _handleOpen(lengthBig, cap) {
          if (lengthBig === 0n || lengthBig > BigInt(cap)) {
            this.completeResponse(-22n, HttpfsStatus.Error);
            return;
          }

          const pathLen = Number(lengthBig);
          const copy = new Uint8Array(pathLen);
          copy.set(this._payloadView.subarray(0, pathLen));
          const rawPath = utf8Decoder.decode(copy);

          if (isRpfVirtualPath(rawPath)) {
            const handle = await controller.openHandle(rawPath, { probeOnOpen });
            if (typeof handle === 'number' && Number.isInteger(handle) && handle >= 0) {
              this.completeResponse(BigInt(handle), HttpfsStatus.ResponseReady);
            } else {
              const errCode = typeof handle === 'number' && handle < 0 ? Math.trunc(handle) : -2;
              this.completeResponse(BigInt(errCode), HttpfsStatus.Error);
            }
            return;
          }

          return super._handleOpen(lengthBig, cap);
        }
      }

      this._mailboxHost = new IoWorkerMailboxHost(sab, {
        getRangeCache: () => this.ensureRangeCache(),
        getManifest: () => this.manifest,
        probeOnOpen,
        forcePoll: options.forcePoll,
        fallbackPollMs: options.fallbackPollMs,
        open: (relPath) => this.openHandle(relPath, { probeOnOpen }),
        read: (handle, offset, reqLen) => this.readHandle(handle, offset, reqLen),
        close: (handle) => this.closeHandle(handle),
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
        // Detailed subsystem telemetry
        rangeCache: cacheStats,
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
      this._handles.clear();
      this._mountedArchives.clear();
      this._inFlightMounts.clear();
      this._rpfMountErrors = 0;
      this._prefetchPromise = null;
      this._prefetchAbortController = null;

      this.dataBase = normalizeDataBase(opts.dataBase !== undefined ? opts.dataBase : this.dataBase);
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
          fetchImpl: fetchFn,
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
          this.postMessage({ id: m.id, error: String(err) });
        }
      }
    }
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
    IoWorkerController,
    isRpfVirtualPath,
    parseRpfVirtualPath,
    emptyStats,
    emptyRangeCacheStats,
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
