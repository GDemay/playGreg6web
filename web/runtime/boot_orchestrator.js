'use strict';

// Browser boot pipeline orchestrator for playgta6web.
// Discovers local boot artifacts via `/api/status`, drives SplashOverlay + PgimPreloader,
// feeds BootTracer milestones, launches Wasm64 runtime or falls back to the smoke demo,
// and dismisses splash on FIRST_PRESENT.
// Zero external dependencies; UMD export for Browser (globalThis.PlayGta6BootOrchestrator)
// and Node.js (module.exports).

(function (root) {
  const DEFAULT_STATUS_URL = '/api/status';
  const DEFAULT_STAGE_TIMEOUT_MS = 0;
  const DEFAULT_PARALLEL_PRELOAD = true;
  const DEFAULT_PRELOAD_CONCURRENCY = 8;
  const DEFAULT_RELEASE_PRELOAD_BUFFERS = true;
  /** Explicit `?streamShaders=1` default threshold (matches web/workers/wgpu_worker.js). */
  const DEFAULT_STREAM_SHADERS_THRESHOLD_BYTES = 64 * 1024 * 1024;
  /** Default auto-stream threshold for plain URLs without `?streamShaders` (matches web/workers/wgpu_worker.js). */
  const DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES = 128 * 1024 * 1024;
  const MAX_DIAGNOSTIC_TEXT_LENGTH = 512;
  const FORBIDDEN_TASK_NAMES = Object.freeze(new Set(['__proto__', 'constructor', 'prototype']));

  /** Ordered high-level boot pipeline phases. */
  const BOOT_PHASE = Object.freeze({
    IDLE: 'IDLE',
    DISCOVERING: 'DISCOVERING',
    PRELOADING: 'PRELOADING',
    LAUNCHING_RUNTIME: 'LAUNCHING_RUNTIME',
    FALLBACK_SMOKE_DEMO: 'FALLBACK_SMOKE_DEMO',
    PRESENTING: 'PRESENTING',
    ERROR: 'ERROR',
    CANCELLED: 'CANCELLED',
    DESTROYED: 'DESTROYED',
  });

  /** Active boot pipeline phases tracked for duration and stall watchdog detection. */
  const WATCHED_BOOT_PHASES = Object.freeze(
    new Set([
      BOOT_PHASE.DISCOVERING,
      BOOT_PHASE.PRELOADING,
      BOOT_PHASE.LAUNCHING_RUNTIME,
      BOOT_PHASE.FALLBACK_SMOKE_DEMO,
    ])
  );

  /** Explicit boot mode distinguishing real Wasm64 guest boot vs smoke-demo fallback. */
  const BOOT_MODE = Object.freeze({
    PENDING: 'pending',
    WASM_GUEST: 'wasm-guest',
    SMOKE_DEMO: 'smoke-demo',
  });

  function defaultNow() {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
      return performance.now();
    }
    return Date.now();
  }

  function clampNonNegativeFinite(value) {
    return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : 0;
  }

  function clampNonNegativeInt(value) {
    return typeof value === 'number' && Number.isFinite(value) && value > 0 ? Math.floor(value) : 0;
  }

  function computeOpfsHitRatePct(hits, misses) {
    const h = clampNonNegativeInt(hits);
    const m = clampNonNegativeInt(misses);
    const total = h + m;
    if (total <= 0) return 0;
    const pct = (h / total) * 100;
    return Number.isFinite(pct) && pct > 0 ? Math.min(100, pct) : 0;
  }

  function normalizeStageTimeoutMs(value) {
    if (value === undefined || value === null) return DEFAULT_STAGE_TIMEOUT_MS;
    if (typeof value !== 'number') {
      throw new TypeError('stageTimeoutMs must be a finite non-negative number or null');
    }
    if (!Number.isFinite(value) || value < 0) {
      throw new RangeError('stageTimeoutMs must be a finite non-negative number');
    }
    return value === 0 ? 0 : value;
  }

  function normalizeParallelPreload(value) {
    if (value === undefined) return DEFAULT_PARALLEL_PRELOAD;
    if (typeof value !== 'boolean') {
      throw new TypeError('parallelPreload must be a boolean');
    }
    return value;
  }

  function normalizePreloadConcurrency(value, label = 'preloadConcurrency') {
    if (value === undefined || value === null) return DEFAULT_PRELOAD_CONCURRENCY;
    if (typeof value !== 'number') {
      throw new TypeError(`${label} must be a positive integer or null`);
    }
    if (!Number.isFinite(value) || !Number.isInteger(value) || value < 1) {
      throw new RangeError(`${label} must be a positive integer`);
    }
    return value;
  }

  function deriveReflectionSidecarUrl(shaderPackUrl) {
    const raw = nonEmptyString(shaderPackUrl);
    if (!raw) return null;
    const cutIdx = raw.search(/[?#]/);
    const basePath = cutIdx >= 0 ? raw.slice(0, cutIdx) : raw;
    if (!basePath) return null;
    if (basePath.endsWith('.reflect.json')) {
      return basePath;
    }
    return `${basePath}.reflect.json`;
  }

  function normalizeReleasePreloadBuffers(value) {
    if (value === undefined) return DEFAULT_RELEASE_PRELOAD_BUFFERS;
    if (typeof value !== 'boolean') {
      throw new TypeError('releasePreloadBuffers must be a boolean');
    }
    return value;
  }

  function normalizeOptionalStreamShaders(value) {
    if (value === undefined || value === null) return undefined;
    if (typeof value !== 'boolean') {
      throw new TypeError('streamShaders must be a boolean or null');
    }
    return value;
  }

  function normalizeOptionalStreamThresholdBytes(value) {
    if (value === undefined || value === null) return undefined;
    if (typeof value !== 'number') {
      throw new TypeError('streamThresholdBytes must be a non-negative safe integer or null');
    }
    if (!Number.isFinite(value) || !Number.isSafeInteger(value) || value < 0) {
      throw new RangeError('streamThresholdBytes must be a non-negative safe integer');
    }
    return value;
  }

  function resolveShaderPackStreamPolicy({ streamShaders, streamThresholdBytes } = {}) {
    const validThreshold =
      typeof streamThresholdBytes === 'number' &&
      Number.isSafeInteger(streamThresholdBytes) &&
      streamThresholdBytes >= 0
        ? streamThresholdBytes
        : undefined;
    if (streamShaders === true) {
      return {
        streamShaders: true,
        skipFullPreload: true,
        maxPreloadBytes:
          validThreshold !== undefined && validThreshold > 0
            ? validThreshold
            : DEFAULT_STREAM_SHADERS_THRESHOLD_BYTES,
      };
    }
    if (streamShaders === false) {
      return {
        streamShaders: false,
        skipFullPreload: false,
        maxPreloadBytes: 0,
      };
    }
    if (validThreshold !== undefined) {
      return {
        streamShaders: undefined,
        skipFullPreload: false,
        maxPreloadBytes: validThreshold,
      };
    }
    return {
      streamShaders: undefined,
      skipFullPreload: false,
      maxPreloadBytes: DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES,
    };
  }

  function readResponseHeader(response, name) {
    if (!response || !response.headers) return null;
    if (typeof response.headers.get === 'function') {
      const val = response.headers.get(name);
      return val !== null && val !== undefined ? String(val) : null;
    }
    if (typeof response.headers === 'object') {
      const lower = name.toLowerCase();
      for (const [k, v] of Object.entries(response.headers)) {
        if (String(k).toLowerCase() === lower && v !== null && v !== undefined) {
          return String(v);
        }
      }
    }
    return null;
  }

  function parseResponseContentLength(response) {
    const encoding = readResponseHeader(response, 'content-encoding');
    if (encoding && encoding.trim().toLowerCase() !== 'identity') {
      return null;
    }
    const raw = readResponseHeader(response, 'content-length');
    if (raw === null) return null;
    const trimmed = raw.trim();
    if (!/^\d+$/.test(trimmed)) return null;
    const n = Number(trimmed);
    return Number.isSafeInteger(n) && n >= 0 ? n : null;
  }

  function extractStatusShaderPackSizeBytes(status) {
    if (!status || typeof status !== 'object') return null;
    const candidates = [
      status.shaderPackSizeBytes,
      status.shaderPackBytes,
      status.shaderPackSize,
      status.splashShaderPackBytes,
      status.splashShaderPackSize,
      status.wgslPack && typeof status.wgslPack === 'object' ? status.wgslPack.size ?? status.wgslPack.bytes : null,
      status.splashShaderPack && typeof status.splashShaderPack === 'object'
        ? status.splashShaderPack.size ?? status.splashShaderPack.bytes
        : null,
    ];
    for (const candidate of candidates) {
      if (typeof candidate === 'number' && Number.isSafeInteger(candidate) && candidate > 0) {
        return candidate;
      }
    }
    return null;
  }

  const WASM_FIELD_PATH_RE = /(?:wasm|guest_?part|guestmodule|sidemodule|wirebytes)/i;

  function isCompiledWasmModule(value) {
    return Boolean(
      value &&
        typeof value === 'object' &&
        typeof WebAssembly !== 'undefined' &&
        typeof WebAssembly.Module === 'function' &&
        value instanceof WebAssembly.Module
    );
  }

  function isTransferableArrayBuffer(value) {
    if (!value || typeof value !== 'object') return false;
    if (typeof SharedArrayBuffer !== 'undefined' && value instanceof SharedArrayBuffer) {
      return false;
    }
    if (Object.prototype.toString.call(value) === '[object SharedArrayBuffer]') {
      return false;
    }
    return (
      value instanceof ArrayBuffer ||
      Object.prototype.toString.call(value) === '[object ArrayBuffer]'
    );
  }

  function getBinaryBufferInfo(value) {
    if (!value || typeof value !== 'object' || isCompiledWasmModule(value)) {
      return null;
    }
    if (value instanceof ArrayBuffer) {
      return { backingBuffer: value, byteLength: value.byteLength };
    }
    if (typeof SharedArrayBuffer !== 'undefined' && value instanceof SharedArrayBuffer) {
      return { backingBuffer: value, byteLength: value.byteLength };
    }
    if (ArrayBuffer.isView(value)) {
      return { backingBuffer: value.buffer, byteLength: value.byteLength };
    }
    return null;
  }

  /**
   * Collect unique standalone transferable non-SAB `ArrayBuffer`s reachable inside `payload`
   * so Window -> Worker `postMessage(msg, transferList)` moves large `.wasm` / `.pgim` /
   * `.wgsl.pack` wire buffers zero-copy instead of cloning them on the Window heap.
   *
   * Skips `SharedArrayBuffer`, `WebAssembly.Module`, already-detached (0-byte) buffers,
   * and subviews that do not span their entire backing `ArrayBuffer`.
   *
   * @param {unknown} payload
   * @returns {ArrayBuffer[]}
   */
  function collectTransferableBuffers(payload) {
    const unique = new Set();
    const visited = new Set();

    const visit = (value) => {
      if (!value || typeof value !== 'object' || isCompiledWasmModule(value)) {
        return;
      }
      if (isTransferableArrayBuffer(value)) {
        if (!value.detached && value.byteLength > 0) {
          unique.add(value);
        }
        return;
      }
      if (typeof SharedArrayBuffer !== 'undefined' && value instanceof SharedArrayBuffer) {
        return;
      }
      if (ArrayBuffer.isView(value)) {
        const buf = value.buffer;
        if (
          isTransferableArrayBuffer(buf) &&
          !buf.detached &&
          value.byteOffset === 0 &&
          value.byteLength === buf.byteLength &&
          buf.byteLength > 0
        ) {
          unique.add(buf);
        }
        return;
      }
      if (visited.has(value)) return;
      visited.add(value);

      if (Array.isArray(value)) {
        for (let i = 0; i < value.length; i += 1) {
          visit(value[i]);
        }
        return;
      }
      if (value instanceof Map) {
        for (const entryVal of value.values()) {
          visit(entryVal);
        }
        return;
      }
      if (value instanceof Set) {
        for (const entryVal of value.values()) {
          visit(entryVal);
        }
        return;
      }
      const keys = Object.keys(value);
      for (let i = 0; i < keys.length; i += 1) {
        let child;
        try {
          child = value[keys[i]];
        } catch {
          continue;
        }
        visit(child);
      }
    };

    visit(payload);
    return Array.from(unique);
  }

  function isWritableOwnProperty(obj, key) {
    if (!obj || typeof obj !== 'object') return false;
    if (!Object.prototype.hasOwnProperty.call(obj, key)) return false;
    const desc = Object.getOwnPropertyDescriptor(obj, key);
    if (!desc) return false;
    if ('get' in desc || 'set' in desc) {
      return typeof desc.set === 'function';
    }
    return desc.writable === true;
  }

  const BINARY_PAYLOAD_PROP_KEYS = Object.freeze([
    'bytes',
    'buffer',
    'wasmBinary',
    'wireBytes',
    'rawBytes',
    'arrayBuffer',
    'data',
    'wasmBytes',
    'wasmBuffer',
  ]);

  /**
   * Release large binary payload references (`ArrayBuffer`, `SharedArrayBuffer`, or TypedArray views)
   * held on a preloaded boot bundle (or single-asset preload result) while preserving lightweight
   * metadata (`bytesLoaded`, `totalBytes`, `entryCount`, `manifest`, `header`, `timingMs`,
   * `durationMs`, `transport`) and compiled `WebAssembly.Module` instances.
   * Defensive and idempotent: never throws on null, primitive, or frozen inputs.
   *
   * @param {unknown} bundle
   * @returns {{ releasedFields: string[], releasedBytes: number }}
   */
  function releaseBootBundleBytes(bundle) {
    const releasedFields = [];
    let releasedBytes = 0;
    if (!bundle || typeof bundle !== 'object') {
      return { releasedFields, releasedBytes };
    }

    const countedByBuffer = new Map();
    const visitedObjects = new Set();

    const recordBufferBytes = (info) => {
      if (!info || !info.backingBuffer) return;
      const len =
        typeof info.byteLength === 'number' && Number.isFinite(info.byteLength) && info.byteLength > 0
          ? info.byteLength
          : 0;
      const prev = countedByBuffer.get(info.backingBuffer) || 0;
      if (len > prev) {
        releasedBytes += len - prev;
        countedByBuffer.set(info.backingBuffer, len);
      }
    };

    const nullBinaryProp = (obj, key, path) => {
      if (!isWritableOwnProperty(obj, key)) return false;
      let val;
      try {
        val = obj[key];
      } catch {
        return false;
      }
      const info = getBinaryBufferInfo(val);
      if (!info) return false;
      try {
        obj[key] = null;
      } catch {
        return false;
      }
      releasedFields.push(path);
      recordBufferBytes(info);
      return true;
    };

    const releasePayloadContainer = (obj, prefix) => {
      if (!obj || typeof obj !== 'object' || isCompiledWasmModule(obj) || visitedObjects.has(obj)) return;
      visitedObjects.add(obj);
      for (let i = 0; i < BINARY_PAYLOAD_PROP_KEYS.length; i += 1) {
        const propKey = BINARY_PAYLOAD_PROP_KEYS[i];
        nullBinaryProp(obj, propKey, `${prefix}.${propKey}`);
      }
      if (Object.prototype.hasOwnProperty.call(obj, 'sideModules')) {
        releaseCollection(obj.sideModules, `${prefix}.sideModules`);
      }
      if (Object.prototype.hasOwnProperty.call(obj, 'modules')) {
        releaseCollection(obj.modules, `${prefix}.modules`);
      }
      if (Object.prototype.hasOwnProperty.call(obj, 'modulesByName')) {
        releaseCollection(obj.modulesByName, `${prefix}.modulesByName`);
      }
      if (Object.prototype.hasOwnProperty.call(obj, 'guestModuleBytes')) {
        releaseCollection(obj.guestModuleBytes, `${prefix}.guestModuleBytes`);
      }
      if (Object.prototype.hasOwnProperty.call(obj, 'preloadedBuffers')) {
        releaseCollection(obj.preloadedBuffers, `${prefix}.preloadedBuffers`);
      }
    };

    const releaseAssetSlot = (parent, key, prefix) => {
      if (!parent || typeof parent !== 'object' || !Object.prototype.hasOwnProperty.call(parent, key)) {
        return;
      }
      let val;
      try {
        val = parent[key];
      } catch {
        return;
      }
      if (!val || typeof val !== 'object' || isCompiledWasmModule(val)) return;
      if (getBinaryBufferInfo(val)) {
        nullBinaryProp(parent, key, prefix);
        return;
      }
      if (val instanceof Map || Array.isArray(val)) {
        releaseCollection(val, prefix);
        return;
      }
      releasePayloadContainer(val, prefix);
    };

    const releaseModuleList = (list, prefix) => {
      if (!Array.isArray(list) || visitedObjects.has(list)) return;
      visitedObjects.add(list);
      let allDirectBinary = list.length > 0;
      for (let i = 0; i < list.length; i += 1) {
        const item = list[i];
        if (!item || typeof item !== 'object' || isCompiledWasmModule(item)) {
          allDirectBinary = false;
          continue;
        }
        if (getBinaryBufferInfo(item)) {
          nullBinaryProp(list, String(i), `${prefix}[${i}]`);
        } else {
          allDirectBinary = false;
          releasePayloadContainer(item, `${prefix}[${i}]`);
        }
      }
      if (allDirectBinary && !Object.isFrozen(list)) {
        try {
          list.length = 0;
        } catch {
          // Ignore non-resizable arrays.
        }
      }
    };

    const releaseBinaryMap = (map, prefix) => {
      if (!(map instanceof Map) || visitedObjects.has(map)) return;
      visitedObjects.add(map);
      for (const [k, val] of Array.from(map.entries())) {
        if (!val || typeof val !== 'object' || isCompiledWasmModule(val)) continue;
        const info = getBinaryBufferInfo(val);
        const keyLabel = typeof k === 'string' || typeof k === 'number' ? String(k) : 'entry';
        if (info) {
          map.delete(k);
          releasedFields.push(`${prefix}.${keyLabel}`);
          recordBufferBytes(info);
        } else {
          releasePayloadContainer(val, `${prefix}.${keyLabel}`);
        }
      }
    };

    const releaseModuleMap = (map, prefix) => {
      if (!map || typeof map !== 'object' || Array.isArray(map) || visitedObjects.has(map)) return;
      if (map instanceof Map) {
        releaseBinaryMap(map, prefix);
        return;
      }
      visitedObjects.add(map);
      for (const key of Object.keys(map)) {
        releaseAssetSlot(map, key, `${prefix}.${key}`);
      }
    };

    const releaseCollection = (col, prefix) => {
      if (!col || typeof col !== 'object') return;
      if (Array.isArray(col)) {
        releaseModuleList(col, prefix);
      } else if (col instanceof Map) {
        releaseBinaryMap(col, prefix);
      } else {
        releaseModuleMap(col, prefix);
      }
    };

    visitedObjects.add(bundle);

    nullBinaryProp(bundle, 'buffer', 'buffer');
    nullBinaryProp(bundle, 'bytes', 'bytes');
    nullBinaryProp(bundle, 'pgimBytes', 'pgimBytes');
    nullBinaryProp(bundle, 'pgimBuffer', 'pgimBuffer');
    nullBinaryProp(bundle, 'shaderPackBuffer', 'shaderPackBuffer');
    nullBinaryProp(bundle, 'wasmBinary', 'wasmBinary');
    nullBinaryProp(bundle, 'wasmBytes', 'wasmBytes');
    nullBinaryProp(bundle, 'wasmBuffer', 'wasmBuffer');
    nullBinaryProp(bundle, 'wireBytes', 'wireBytes');
    nullBinaryProp(bundle, 'rawBytes', 'rawBytes');
    nullBinaryProp(bundle, 'arrayBuffer', 'arrayBuffer');

    releaseAssetSlot(bundle, 'shaderPackBytes', 'shaderPackBytes');
    releaseAssetSlot(bundle, 'pgim', 'pgim');
    releaseAssetSlot(bundle, 'shaderPackIndex', 'shaderPackIndex');
    releaseAssetSlot(bundle, 'reflectionManifest', 'reflectionManifest');
    releaseAssetSlot(bundle, 'reflectionSidecar', 'reflectionSidecar');
    releaseAssetSlot(bundle, 'runtimeWasm', 'runtimeWasm');
    releaseAssetSlot(bundle, 'opfsWarmBoot', 'opfsWarmBoot');

    if (Object.prototype.hasOwnProperty.call(bundle, 'modules')) {
      releaseCollection(bundle.modules, 'modules');
    }
    if (Object.prototype.hasOwnProperty.call(bundle, 'modulesByName')) {
      releaseCollection(bundle.modulesByName, 'modulesByName');
    }
    if (Object.prototype.hasOwnProperty.call(bundle, 'sideModules')) {
      releaseCollection(bundle.sideModules, 'sideModules');
    }
    if (Object.prototype.hasOwnProperty.call(bundle, 'sideModuleBytes')) {
      releaseCollection(bundle.sideModuleBytes, 'sideModuleBytes');
    }
    if (Object.prototype.hasOwnProperty.call(bundle, 'guestModules')) {
      releaseCollection(bundle.guestModules, 'guestModules');
    }
    if (Object.prototype.hasOwnProperty.call(bundle, 'guestModuleBytes')) {
      releaseCollection(bundle.guestModuleBytes, 'guestModuleBytes');
    }
    if (Object.prototype.hasOwnProperty.call(bundle, 'wasmModules')) {
      releaseCollection(bundle.wasmModules, 'wasmModules');
    }
    if (Object.prototype.hasOwnProperty.call(bundle, 'preloadedBuffers')) {
      releaseCollection(bundle.preloadedBuffers, 'preloadedBuffers');
    }
    if (Object.prototype.hasOwnProperty.call(bundle, 'mainModule')) {
      releaseAssetSlot(bundle, 'mainModule', 'mainModule');
    }
    if (
      Object.prototype.hasOwnProperty.call(bundle, 'program') &&
      bundle.program &&
      typeof bundle.program === 'object' &&
      !visitedObjects.has(bundle.program)
    ) {
      const program = bundle.program;
      visitedObjects.add(program);
      for (let i = 0; i < BINARY_PAYLOAD_PROP_KEYS.length; i += 1) {
        const propKey = BINARY_PAYLOAD_PROP_KEYS[i];
        nullBinaryProp(program, propKey, `program.${propKey}`);
      }
      if (Object.prototype.hasOwnProperty.call(program, 'mainModule')) {
        releaseAssetSlot(program, 'mainModule', 'program.mainModule');
      }
      if (Object.prototype.hasOwnProperty.call(program, 'modules')) {
        releaseCollection(program.modules, 'program.modules');
      }
      if (Object.prototype.hasOwnProperty.call(program, 'modulesByName')) {
        releaseCollection(program.modulesByName, 'program.modulesByName');
      }
    }

    if (
      Object.prototype.hasOwnProperty.call(bundle, 'sidecars') &&
      bundle.sidecars &&
      typeof bundle.sidecars === 'object' &&
      !Array.isArray(bundle.sidecars) &&
      !visitedObjects.has(bundle.sidecars)
    ) {
      const sidecars = bundle.sidecars;
      visitedObjects.add(sidecars);
      for (const key of Object.keys(sidecars)) {
        releaseAssetSlot(sidecars, key, `sidecars.${key}`);
      }
    }

    return { releasedFields, releasedBytes };
  }

  /**
   * Post `message` to `worker` with zero-copy `ArrayBuffer` transfer for every unique
   * standalone non-SAB `ArrayBuffer` found on `message` (and optional extra transferables
   * such as `OffscreenCanvas` or `MessagePort`), then immediately release raw binary
   * references on `releaseTarget` (if provided) so Window heap drops to ~0 bytes.
   *
   * @param {{ postMessage: Function }} worker
   * @param {object} message
   * @param {{ extraTransferables?: unknown[], releaseTarget?: unknown }} [options]
   * @returns {{ transferredBuffers: number, transferredBytes: number, releasedFields: string[], releasedBytes: number }}
   */
  function transferPreloadPayloadToWorker(worker, message, options = {}) {
    if (!worker || typeof worker.postMessage !== 'function') {
      throw new TypeError('worker must expose postMessage');
    }
    const safeOpts = options && typeof options === 'object' ? options : {};
    const arrayBuffers = collectTransferableBuffers(message);
    let transferredBytes = 0;
    for (let i = 0; i < arrayBuffers.length; i += 1) {
      transferredBytes += arrayBuffers[i].byteLength;
    }
    const transferList = arrayBuffers.slice();
    if (Array.isArray(safeOpts.extraTransferables)) {
      const seen = new Set(transferList);
      for (let i = 0; i < safeOpts.extraTransferables.length; i += 1) {
        const item = safeOpts.extraTransferables[i];
        if (item && !seen.has(item)) {
          seen.add(item);
          transferList.push(item);
        }
      }
    }
    if (transferList.length > 0) {
      worker.postMessage(message, transferList);
    } else {
      worker.postMessage(message);
    }
    const releaseTarget =
      safeOpts.releaseTarget !== undefined ? safeOpts.releaseTarget : message;
    const released = releaseTarget ? releaseBootBundleBytes(releaseTarget) : { releasedFields: [], releasedBytes: 0 };
    return {
      transferredBuffers: arrayBuffers.length,
      transferredBytes,
      releasedFields: released.releasedFields,
      releasedBytes: released.releasedBytes,
    };
  }

  function normalizePreloadTasks(value) {
    if (value === undefined || value === null) return [];
    if (!Array.isArray(value)) {
      throw new TypeError('preloadTasks must be an array or null');
    }
    const seen = new Set();
    const out = [];
    for (let i = 0; i < value.length; i += 1) {
      const entry = value[i];
      if (!entry || typeof entry !== 'object') {
        throw new TypeError(`preloadTasks[${i}] must be an object`);
      }
      const name = typeof entry.name === 'string' ? entry.name.trim() : '';
      if (!name || FORBIDDEN_TASK_NAMES.has(name)) {
        throw new TypeError(`preloadTasks[${i}].name must be a valid non-empty string`);
      }
      if (seen.has(name)) {
        throw new TypeError(`preloadTasks contains duplicate task name "${name}"`);
      }
      if (typeof entry.run !== 'function') {
        throw new TypeError(`preloadTasks[${i}].run must be a function`);
      }
      if (entry.optional !== undefined && typeof entry.optional !== 'boolean') {
        throw new TypeError(`preloadTasks[${i}].optional must be a boolean`);
      }
      const asset =
        typeof entry.asset === 'string' && entry.asset.trim().length > 0 ? entry.asset.trim() : name;
      seen.add(name);
      out.push(
        Object.freeze({
          name,
          asset,
          run: entry.run,
          optional: entry.optional !== undefined ? entry.optional : true,
        })
      );
    }
    return out;
  }

  function emptyPreloadStats(parallelEnabled, concurrencyLimit = DEFAULT_PRELOAD_CONCURRENCY) {
    const resolvedLimit =
      Number.isInteger(concurrencyLimit) && concurrencyLimit >= 1 ? concurrencyLimit : DEFAULT_PRELOAD_CONCURRENCY;
    return {
      parallelEnabled: Boolean(parallelEnabled),
      concurrencyLimit: resolvedLimit,
      taskCount: 0,
      completedCount: 0,
      optionalMissingCount: 0,
      failedCount: 0,
      abortedCount: 0,
      maxConcurrency: 0,
      parallelStagesCount: 0,
      wallTimeMs: 0,
      parallelWallMs: 0,
      sequentialTimeMs: 0,
      serialEstimateMs: 0,
      parallelPreloadSavedMs: 0,
      overlapSavedMs: 0,
      taskTimingsMs: {},
      tasks: {},
    };
  }

  function clonePreloadStats(stats) {
    const src = stats || emptyPreloadStats(DEFAULT_PARALLEL_PRELOAD, DEFAULT_PRELOAD_CONCURRENCY);
    const taskTimingsMs = {};
    if (src.taskTimingsMs && typeof src.taskTimingsMs === 'object') {
      for (const key of Object.keys(src.taskTimingsMs)) {
        taskTimingsMs[key] = clampNonNegativeFinite(src.taskTimingsMs[key]);
      }
    }
    const tasks = {};
    if (src.tasks && typeof src.tasks === 'object') {
      for (const key of Object.keys(src.tasks)) {
        const entry = src.tasks[key];
        if (entry && typeof entry === 'object') {
          tasks[key] = {
            status: entry.status || 'completed',
            optional: Boolean(entry.optional),
            durationMs: clampNonNegativeFinite(entry.durationMs),
            error: entry.error ? sanitizeDiagnosticText(entry.error) : null,
          };
        }
      }
    }
    const wallTimeMs = clampNonNegativeFinite(
      src.parallelWallMs !== undefined ? src.parallelWallMs : src.wallTimeMs
    );
    const sequentialTimeMs = clampNonNegativeFinite(
      src.serialEstimateMs !== undefined ? src.serialEstimateMs : src.sequentialTimeMs
    );
    const overlapSavedMs = clampNonNegativeFinite(
      src.overlapSavedMs !== undefined ? src.overlapSavedMs : src.parallelPreloadSavedMs
    );
    return {
      parallelEnabled: Boolean(src.parallelEnabled),
      concurrencyLimit:
        Number.isInteger(src.concurrencyLimit) && src.concurrencyLimit >= 1
          ? src.concurrencyLimit
          : DEFAULT_PRELOAD_CONCURRENCY,
      taskCount: Number.isInteger(src.taskCount) && src.taskCount >= 0 ? src.taskCount : 0,
      completedCount: Number.isInteger(src.completedCount) && src.completedCount >= 0 ? src.completedCount : 0,
      optionalMissingCount:
        Number.isInteger(src.optionalMissingCount) && src.optionalMissingCount >= 0 ? src.optionalMissingCount : 0,
      failedCount: Number.isInteger(src.failedCount) && src.failedCount >= 0 ? src.failedCount : 0,
      abortedCount: Number.isInteger(src.abortedCount) && src.abortedCount >= 0 ? src.abortedCount : 0,
      maxConcurrency: Number.isInteger(src.maxConcurrency) && src.maxConcurrency >= 0 ? src.maxConcurrency : 0,
      parallelStagesCount:
        Number.isInteger(src.parallelStagesCount) && src.parallelStagesCount >= 0 ? src.parallelStagesCount : 0,
      wallTimeMs,
      parallelWallMs: wallTimeMs,
      sequentialTimeMs,
      serialEstimateMs: sequentialTimeMs,
      parallelPreloadSavedMs: overlapSavedMs,
      overlapSavedMs,
      taskTimingsMs,
      tasks,
    };
  }

  /**
   * Execute independent preload tasks concurrently with bounded concurrency via `Promise.allSettled`,
   * deterministic fatal error propagation, optional-stage resilience, and sanitized overlap timing
   * savings telemetry (`serialEstimateMs`, `parallelWallMs`, `overlapSavedMs`, `parallelStagesCount`).
   *
   * @param {Array<{name: string, run: Function, optional?: boolean, asset?: string, kind?: string}>} tasks
   * @param {object} [options]
   * @param {boolean} [options.parallel=true]
   * @param {number|null} [options.maxConcurrency=8]
   * @param {AbortSignal|null} [options.signal=null]
   * @param {(() => number)|null} [options.nowFn]
   * @param {Function|null} [options.onOptionalMissing]
   * @param {Function|null} [options.onComplete]
   * @param {boolean} [options.throwOnError=true]
   */
  async function runConcurrentPreloads(tasks, options = {}) {
    if (!Array.isArray(tasks)) {
      throw new TypeError('runConcurrentPreloads tasks must be an array');
    }
    if (options !== undefined && (options === null || typeof options !== 'object')) {
      throw new TypeError('runConcurrentPreloads options must be an object');
    }

    const parallel = normalizeParallelPreload(options.parallel);
    const concurrencyOption =
      options.maxConcurrency !== undefined ? options.maxConcurrency : options.concurrency;
    const concurrencyLimit = normalizePreloadConcurrency(concurrencyOption, 'maxConcurrency');
    const signal = options.signal !== undefined ? options.signal : null;
    if (signal !== null && (typeof signal !== 'object' || typeof signal.aborted !== 'boolean')) {
      throw new TypeError('runConcurrentPreloads signal must be an AbortSignal or null');
    }
    const clockFn = optionalFn(options.nowFn, 'nowFn') || defaultNow;
    const onOptionalMissing = optionalFn(options.onOptionalMissing, 'onOptionalMissing');
    const onComplete = optionalFn(options.onComplete, 'onComplete');
    const throwOnError = options.throwOnError !== undefined ? Boolean(options.throwOnError) : true;

    const seenNames = new Set();
    const normalizedTasks = [];
    for (let i = 0; i < tasks.length; i += 1) {
      const entry = tasks[i];
      if (!entry || typeof entry !== 'object') {
        throw new TypeError(`tasks[${i}] must be an object`);
      }
      const name = typeof entry.name === 'string' ? entry.name.trim() : '';
      if (!name || FORBIDDEN_TASK_NAMES.has(name)) {
        throw new TypeError(`tasks[${i}].name must be a valid non-empty string`);
      }
      if (seenNames.has(name)) {
        throw new TypeError(`tasks contains duplicate task name "${name}"`);
      }
      if (typeof entry.run !== 'function') {
        throw new TypeError(`tasks[${i}].run must be a function`);
      }
      if (entry.optional !== undefined && typeof entry.optional !== 'boolean') {
        throw new TypeError(`tasks[${i}].optional must be a boolean`);
      }
      seenNames.add(name);
      normalizedTasks.push({
        name,
        asset: typeof entry.asset === 'string' && entry.asset.trim().length > 0 ? entry.asset.trim() : name,
        kind: typeof entry.kind === 'string' && entry.kind.length > 0 ? entry.kind : 'custom',
        optional: entry.optional !== undefined ? entry.optional : true,
        run: entry.run,
      });
    }

    if (normalizedTasks.length === 0) {
      const empty = emptyPreloadStats(parallel, concurrencyLimit);
      invokeQuiet(onComplete, clonePreloadStats(empty));
      return {
        ok: true,
        aborted: false,
        error: null,
        results: {},
        resultsMap: new Map(),
        settled: [],
        stats: clonePreloadStats(empty),
      };
    }

    let lastClockMs = 0;
    const readNow = () => {
      try {
        const value = clockFn();
        if (typeof value === 'number' && Number.isFinite(value)) {
          lastClockMs = value;
          return value;
        }
      } catch {
        // Fall back to last known finite reading.
      }
      return lastClockMs;
    };

    const batchStartedAt = readNow();
    const { controller: batchController, unlink: unlinkBatch } = linkSignal(signal);
    const batchSignal = batchController.signal;

    let inFlight = 0;
    let maxConcurrency = 0;
    let hasFatalError = false;
    let firstAbortError = null;
    const fatalErrorsByIndex = new Map();
    let completedCount = 0;
    let optionalMissingCount = 0;
    let failedCount = 0;
    let abortedCount = 0;
    let sequentialTimeMs = 0;
    let effectiveTaskCount = 0;
    let reportedBootBundleTotalMs = 0;
    const resultsMap = new Map();
    const taskTimingsMs = Object.create(null);
    const taskRecords = Object.create(null);
    const settled = new Array(normalizedTasks.length);

    const runSingleTask = async (task, taskIndex) => {
      const startedAt = readNow();
      inFlight += 1;
      if (inFlight > maxConcurrency) {
        maxConcurrency = inFlight;
      }
      try {
        throwIfAborted(batchSignal);
        const value = await raceAbort(Promise.resolve().then(() => task.run(batchSignal)), batchSignal);
        const durationMs = clampNonNegativeFinite(readNow() - startedAt);
        resultsMap.set(task.name, value);
        settled[taskIndex] = {
          name: task.name,
          asset: task.asset,
          optional: Boolean(task.optional),
          status: 'fulfilled',
          value,
          error: null,
          durationMs,
        };

        const subTiming =
          task.kind === 'bootBundle' &&
          value &&
          typeof value === 'object' &&
          value.timingMs &&
          typeof value.timingMs === 'object'
            ? value.timingMs
            : null;
        const subKeys = subTiming
          ? Object.keys(subTiming).filter(
              (key) =>
                key !== 'total' &&
                typeof subTiming[key] === 'number' &&
                Number.isFinite(subTiming[key]) &&
                subTiming[key] >= 0
            )
          : [];

        if (subKeys.length > 0) {
          for (const key of subKeys) {
            const subDur = clampNonNegativeFinite(subTiming[key]);
            taskTimingsMs[key] = subDur;
            taskRecords[key] = {
              status: 'completed',
              optional: key !== 'pgim',
              durationMs: subDur,
              error: null,
            };
            sequentialTimeMs += subDur;
            effectiveTaskCount += 1;
            completedCount += 1;
          }
          if (typeof subTiming.total === 'number' && Number.isFinite(subTiming.total) && subTiming.total > 0) {
            reportedBootBundleTotalMs = clampNonNegativeFinite(subTiming.total);
          }
          if (subKeys.length > maxConcurrency) {
            maxConcurrency = subKeys.length;
          }
        } else {
          taskTimingsMs[task.name] = durationMs;
          taskRecords[task.name] = {
            status: 'completed',
            optional: Boolean(task.optional),
            durationMs,
            error: null,
          };
          sequentialTimeMs += durationMs;
          effectiveTaskCount += 1;
          completedCount += 1;
        }
      } catch (err) {
        const durationMs = clampNonNegativeFinite(readNow() - startedAt);
        const abortedByOuter = Boolean(signal && signal.aborted);
        const explicitAbort = isAbortError(err);
        const abortedBySibling = Boolean(
          !abortedByOuter &&
          batchSignal.aborted &&
          hasFatalError &&
          (explicitAbort || err === batchSignal.reason)
        );

        if (abortedByOuter || abortedBySibling || explicitAbort) {
          abortedCount += 1;
          const sanitizedErr = sanitizeDiagnosticText(toError(err, 'Preload task aborted').message);
          taskTimingsMs[task.name] = durationMs;
          taskRecords[task.name] = {
            status: 'aborted',
            optional: Boolean(task.optional),
            durationMs,
            error: sanitizedErr,
          };
          settled[taskIndex] = {
            name: task.name,
            asset: task.asset,
            optional: Boolean(task.optional),
            status: 'aborted',
            value: null,
            error: sanitizedErr,
            durationMs,
          };
          if (!abortedBySibling && firstAbortError === null) {
            firstAbortError = err;
            if (!batchSignal.aborted) {
              batchController.abort(err);
            }
          }
          return;
        }

        if (task.optional && !isValidationError(err)) {
          optionalMissingCount += 1;
          resultsMap.set(task.name, null);
          const statusLabel = isMissingOptionalError(err) ? 'missing' : 'fallback';
          const sanitizedErr = sanitizeDiagnosticText(
            toError(err, `Optional preload task ${task.name} failed`).message
          );
          taskTimingsMs[task.name] = durationMs;
          taskRecords[task.name] = {
            status: statusLabel,
            optional: true,
            durationMs,
            error: sanitizedErr,
          };
          settled[taskIndex] = {
            name: task.name,
            asset: task.asset,
            optional: true,
            status: statusLabel,
            value: null,
            error: sanitizedErr,
            durationMs,
          };
          sequentialTimeMs += durationMs;
          effectiveTaskCount += 1;
          invokeQuiet(onOptionalMissing, { task, error: sanitizedErr, status: statusLabel, durationMs });
          return;
        }

        failedCount += 1;
        const sanitizedErr = sanitizeDiagnosticText(toError(err, `Preload task ${task.name} failed`).message);
        taskTimingsMs[task.name] = durationMs;
        taskRecords[task.name] = {
          status: 'failed',
          optional: Boolean(task.optional),
          durationMs,
          error: sanitizedErr,
        };
        settled[taskIndex] = {
          name: task.name,
          asset: task.asset,
          optional: Boolean(task.optional),
          status: 'rejected',
          value: null,
          error: sanitizedErr,
          durationMs,
        };
        fatalErrorsByIndex.set(taskIndex, err);
        hasFatalError = true;
        if (!batchSignal.aborted) {
          batchController.abort(err);
        }
      } finally {
        inFlight -= 1;
      }
    };

    const effectiveConcurrency = parallel ? Math.min(normalizedTasks.length, concurrencyLimit) : 1;
    let nextIndex = 0;

    const runWorker = async () => {
      while (nextIndex < normalizedTasks.length) {
        if (batchSignal.aborted || hasFatalError) {
          break;
        }
        const taskIndex = nextIndex;
        nextIndex += 1;
        await runSingleTask(normalizedTasks[taskIndex], taskIndex);
        if (!parallel && (batchSignal.aborted || hasFatalError)) {
          break;
        }
      }
    };

    let stats;
    try {
      const workers = [];
      for (let w = 0; w < effectiveConcurrency; w += 1) {
        workers.push(runWorker());
      }
      await Promise.allSettled(workers);
    } finally {
      unlinkBatch();
      const measuredWallMs = clampNonNegativeFinite(readNow() - batchStartedAt);
      const parallelWallMs = Math.max(measuredWallMs, reportedBootBundleTotalMs);
      const serialEstimateMs = clampNonNegativeFinite(sequentialTimeMs);
      const parallelStagesCount = parallel && effectiveTaskCount > 1 && maxConcurrency > 1 ? effectiveTaskCount : 0;
      const canSave = parallel && parallelStagesCount > 1 && failedCount === 0 && abortedCount === 0;
      const overlapSavedMs = canSave ? clampNonNegativeFinite(serialEstimateMs - parallelWallMs) : 0;
      stats = {
        parallelEnabled: parallel,
        concurrencyLimit,
        taskCount: effectiveTaskCount > 0 ? effectiveTaskCount : normalizedTasks.length,
        completedCount,
        optionalMissingCount,
        failedCount,
        abortedCount,
        maxConcurrency,
        parallelStagesCount,
        wallTimeMs: parallelWallMs,
        parallelWallMs,
        sequentialTimeMs: serialEstimateMs,
        serialEstimateMs,
        parallelPreloadSavedMs: overlapSavedMs,
        overlapSavedMs,
        taskTimingsMs,
        tasks: taskRecords,
      };
      invokeQuiet(onComplete, clonePreloadStats(stats));
    }

    let fatalError = null;
    if (signal && signal.aborted) {
      fatalError = abortReason(signal);
    } else if (fatalErrorsByIndex.size > 0) {
      let minIdx = Number.POSITIVE_INFINITY;
      for (const idx of fatalErrorsByIndex.keys()) {
        if (idx < minIdx) minIdx = idx;
      }
      fatalError = fatalErrorsByIndex.get(minIdx);
    } else if (firstAbortError !== null) {
      fatalError = firstAbortError;
    }

    const clonedStats = clonePreloadStats(stats);
    const results = Object.create(null);
    for (const [key, val] of resultsMap.entries()) {
      results[key] = val;
    }
    const plainResults = { ...results };
    const compactedSettled = settled.filter(Boolean);

    if (fatalError !== null) {
      if (throwOnError) {
        throw fatalError;
      }
      return {
        ok: false,
        aborted: Boolean((signal && signal.aborted) || isAbortError(fatalError)),
        error: fatalError,
        results: plainResults,
        resultsMap,
        settled: compactedSettled,
        stats: clonedStats,
      };
    }

    return {
      ok: true,
      aborted: false,
      error: null,
      results: plainResults,
      resultsMap,
      settled: compactedSettled,
      stats: clonedStats,
    };
  }

  function inferBytesLoaded(value) {
    if (!value || typeof value !== 'object') return 0;
    if (typeof value.bytesLoaded === 'number' && Number.isFinite(value.bytesLoaded) && value.bytesLoaded > 0) {
      return value.bytesLoaded;
    }
    if (typeof value.totalBytes === 'number' && Number.isFinite(value.totalBytes) && value.totalBytes > 0) {
      return value.totalBytes;
    }
    if (typeof value.byteLength === 'number' && Number.isFinite(value.byteLength) && value.byteLength > 0) {
      return value.byteLength;
    }
    if (
      value.bytes &&
      typeof value.bytes.byteLength === 'number' &&
      Number.isFinite(value.bytes.byteLength) &&
      value.bytes.byteLength > 0
    ) {
      return value.bytes.byteLength;
    }
    return 0;
  }

  function isMissingOptionalError(err) {
    if (!err || typeof err !== 'object') return false;
    if (err.status === 404 || err.status === 410) return true;
    if (err.code === 'NOT_FOUND' || err.code === 'ENOENT' || err.code === 'OPTIONAL_MISSING') return true;
    if (typeof err.message === 'string' && /\b(404|410|not\s+found|missing)\b/i.test(err.message)) {
      return true;
    }
    return false;
  }

  function isValidationError(err) {
    return Boolean(err && typeof err === 'object' && typeof err.code === 'string' && err.code.startsWith('BAD_'));
  }

  function raceAbort(promise, signal) {
    if (!signal) return promise;
    if (signal.aborted) {
      Promise.resolve(promise).catch(() => {});
      return Promise.reject(abortReason(signal));
    }
    return new Promise((resolve, reject) => {
      const onAbort = () => reject(abortReason(signal));
      signal.addEventListener('abort', onAbort, { once: true });
      Promise.resolve(promise).then(
        (value) => {
          signal.removeEventListener('abort', onAbort);
          resolve(value);
        },
        (err) => {
          signal.removeEventListener('abort', onAbort);
          reject(err);
        }
      );
    });
  }

  function sumStageDurations(stageDurationsMs, currentStageElapsedMs) {
    let total = clampNonNegativeFinite(currentStageElapsedMs);
    for (const key of Object.keys(stageDurationsMs)) {
      total += clampNonNegativeFinite(stageDurationsMs[key]);
    }
    return clampNonNegativeFinite(total);
  }

  const FILE_URL_RE = /\bfile:\/\/[^\s"'`<>)\]}]+/gi;
  const URL_USERINFO_RE = /\b((?:https?|wss?|ftp):\/\/)([^\s/@"'`<>]+):([^\s/@"'`<>]*)@/gi;
  const URL_QUERY_RE = /((?:https?|wss?):\/\/[^\s"'`<>?#]+)\?(?!\[REDACTED\])[^\s"'`<>#)\]}]+/gi;
  const AUTH_SCHEME_RE = /\b(Bearer|Basic)\s+(?!\[REDACTED\])[A-Za-z0-9\-._~+/=:]+/gi;
  const SECRET_KV_RE =
    /\b(authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|id[_-]?token|client[_-]?secret|x-api-key|x-auth-token|cookie|set-cookie|password|passwd|secret|token)(\s*[:=]\s*)(?!(?:(?:Bearer|Basic)\s+)?\[REDACTED\])(?:"[^"]*"|'[^']*'|[^\s,;}\])]+)/gi;
  const WIN_DRIVE_PATH_RE = /(?<![A-Za-z0-9_])[A-Za-z]:(?:\\|\/(?!\/))[^\s"'`<>|*?)\]},;]+/g;
  const WIN_UNC_PATH_RE = /(^|[\s"'`(\[={<,:;])\\\\[^\s\\"'`<>|*?)\]},;]+\\[^\s"'`<>|*?)\]},;]+/g;
  const POSIX_ABS_PATH_RE =
    /(^|[\s(\[={<,:;`"'])\/(?:home|Users|root|tmp|var|etc|opt|usr|mnt|media|Volumes|private|srv)\/[^\s"'`<>)\]},;]+/g;

  function sanitizeDiagnosticText(value, maxLen = MAX_DIAGNOSTIC_TEXT_LENGTH) {
    if (value == null) return null;
    let text = String(value);
    text = text.replace(FILE_URL_RE, '[REDACTED_PATH]');
    text = text.replace(URL_USERINFO_RE, '$1[REDACTED]@');
    text = text.replace(URL_QUERY_RE, '$1?[REDACTED]');
    text = text.replace(AUTH_SCHEME_RE, '$1 [REDACTED]');
    text = text.replace(SECRET_KV_RE, '$1$2[REDACTED]');
    text = text.replace(WIN_DRIVE_PATH_RE, '[REDACTED_PATH]');
    text = text.replace(WIN_UNC_PATH_RE, '$1[REDACTED_PATH]');
    text = text.replace(POSIX_ABS_PATH_RE, '$1[REDACTED_PATH]');
    if (text.length > maxLen) {
      return `${text.slice(0, maxLen - 3)}...`;
    }
    return text;
  }

  function defaultFetchImpl() {
    if (typeof globalThis !== 'undefined' && typeof globalThis.fetch === 'function') {
      return globalThis.fetch.bind(globalThis);
    }
    return null;
  }

  function abortReason(signal) {
    if (signal && signal.reason !== undefined) return signal.reason;
    const Err = typeof DOMException === 'function' ? DOMException : Error;
    try {
      return new Err('The operation was aborted.', 'AbortError');
    } catch {
      const err = new Error('The operation was aborted.');
      err.name = 'AbortError';
      return err;
    }
  }

  function isAbortError(err) {
    if (!err || typeof err !== 'object') return false;
    if (err.name === 'AbortError') return true;
    if (typeof DOMException === 'function' && err instanceof DOMException && err.name === 'AbortError') {
      return true;
    }
    return false;
  }

  function throwIfAborted(signal) {
    if (signal && signal.aborted) {
      throw abortReason(signal);
    }
  }

  function toError(err, fallbackMessage = 'Unknown boot orchestrator error') {
    if (err instanceof Error) return err;
    if (typeof err === 'string' && err.length > 0) return new Error(err);
    if (err && typeof err === 'object' && typeof err.message === 'string') return new Error(err.message);
    return new Error(fallbackMessage);
  }

  function nonEmptyString(value) {
    return typeof value === 'string' && value.trim().length > 0 ? value.trim() : null;
  }

  function optionalFn(value, label) {
    if (value == null) return null;
    if (typeof value !== 'function') {
      throw new TypeError(`${label} must be a function or null`);
    }
    return value;
  }

  function optionalObject(value, label) {
    if (value == null) return null;
    if (typeof value !== 'object') {
      throw new TypeError(`${label} must be an object or null`);
    }
    return value;
  }

  function callQuiet(target, method, ...args) {
    if (!target || typeof target[method] !== 'function') return undefined;
    try {
      return target[method](...args);
    } catch {
      return undefined;
    }
  }

  function invokeQuiet(fn, ...args) {
    if (typeof fn !== 'function') return undefined;
    try {
      return fn(...args);
    } catch {
      return undefined;
    }
  }

  async function invokeQuietAsync(fn, ...args) {
    if (typeof fn !== 'function') return undefined;
    try {
      return await fn(...args);
    } catch {
      return undefined;
    }
  }

  function pickUrl(...candidates) {
    for (const candidate of candidates) {
      const url = nonEmptyString(candidate);
      if (url) return url;
    }
    return null;
  }

  function presentFlag(value) {
    if (value === true) return true;
    if (value && typeof value === 'object') {
      if (value.present === true || value.available === true || value.ok === true || value.nonEmpty === true) {
        return true;
      }
      if (nonEmptyString(value.url) || nonEmptyString(value.path) || nonEmptyString(value.href)) {
        return true;
      }
    }
    return typeof value === 'string' && value.trim().length > 0;
  }

  function urlFromArtifact(value) {
    if (typeof value === 'string') return nonEmptyString(value);
    if (value && typeof value === 'object') {
      return pickUrl(value.url, value.path, value.href);
    }
    return null;
  }

  function normalizeFirstPresentSource(value) {
    if (typeof value !== 'string') return null;
    const normalized = value.trim().toLowerCase();
    if (normalized === 'wasm-guest' || normalized === 'wasm_guest') {
      return BOOT_MODE.WASM_GUEST;
    }
    if (normalized === 'smoke-demo' || normalized === 'smoke_demo') {
      return BOOT_MODE.SMOKE_DEMO;
    }
    return null;
  }

  function resolveFirstPresentSource(meta, bootMode) {
    const explicit =
      meta && typeof meta === 'object'
        ? normalizeFirstPresentSource(meta.firstPresentSource) || normalizeFirstPresentSource(meta.source)
        : null;
    if (explicit) return explicit;
    if (bootMode === BOOT_MODE.WASM_GUEST) return BOOT_MODE.WASM_GUEST;
    if (bootMode === BOOT_MODE.SMOKE_DEMO) return BOOT_MODE.SMOKE_DEMO;
    return null;
  }

  /**
   * Returns true ONLY when the supplied orchestrator, state snapshot, assets object, or
   * mode string represents a real Wasm64 guest boot (`bootMode === 'wasm-guest'`).
   *
   * @param {unknown} target
   * @returns {boolean}
   */
  function isRealGuestBoot(target) {
    if (typeof target === 'string') {
      return target === BOOT_MODE.WASM_GUEST;
    }
    if (target && typeof target === 'object') {
      if (typeof target.isRealGuestBoot === 'function') {
        return target.isRealGuestBoot() === true;
      }
      if (typeof target.getState === 'function') {
        const state = target.getState();
        return Boolean(state && state.bootMode === BOOT_MODE.WASM_GUEST);
      }
      return target.bootMode === BOOT_MODE.WASM_GUEST;
    }
    return false;
  }

  /**
   * Normalize `/api/status` JSON into the boot-discovery fields used by this orchestrator.
   * Accepts both the current server payload (`hasWasmBinary`, `shaderPackCount`) and the
   * richer artifact URLs (`splashImage`, `pgimUrl`, `programManifestUrl`, `gameBase`,
   * `shaderPackUrl`, `reflectionManifestUrl`, `runtimeWasm`) that tests and later server
   * revisions expose.
   */
  function discoverBootAssets(status) {
    const src = status && typeof status === 'object' ? status : {};
    const splashImage = pickUrl(
      src.splashImage,
      src.splashUrl,
      src.splashScreenImage,
      urlFromArtifact(src.splash)
    );
    const pgimUrl = pickUrl(src.pgimUrl, src.pgim, urlFromArtifact(src.pgimImage));
    const programManifestUrl = pickUrl(
      src.programManifestUrl,
      urlFromArtifact(src.programManifest)
    );
    const gameBase = pickUrl(src.gameBase, src.gameBaseUrl);
    const shaderPackUrl = pickUrl(
      src.shaderPackUrl,
      src.shaderPack,
      src.splashShaderPackUrl,
      src.splashPackUrl,
      urlFromArtifact(src.wgslPack),
      urlFromArtifact(src.splashShaderPack)
    );
    const shaderPackIndexUrl = pickUrl(
      src.shaderPackIndexUrl,
      src.shaderPackHeaderUrl,
      src.wgpkHeaderUrl,
      urlFromArtifact(src.shaderPackIndex),
      urlFromArtifact(src.shaderPackHeader),
      urlFromArtifact(src.wgpkHeader)
    );
    const reflectionManifestUrl = pickUrl(
      src.reflectionManifestUrl,
      src.reflectionUrl,
      urlFromArtifact(src.reflectionManifest)
    );
    const reflectionSidecarUrl = pickUrl(
      src.reflectionSidecarUrl,
      src.stageReflectionSidecarUrl,
      src.sidecarUrl,
      src.reflectJsonUrl,
      src.reflectionJsonUrl,
      urlFromArtifact(src.reflectionSidecar),
      urlFromArtifact(src.sidecar),
      urlFromArtifact(src.reflectJson)
    );

    const runtimeArtifact = src.runtimeWasm !== undefined ? src.runtimeWasm : src.hasWasmBinary;
    const runtimeWasmUrl = urlFromArtifact(src.runtimeWasm) || (src.hasWasmBinary ? '/dist/runtime.wasm' : null);
    const runtimeWasmAvailable = presentFlag(runtimeArtifact) || src.hasWasmBinary === true;
    const canLaunchGuest = runtimeWasmAvailable && Boolean(pgimUrl || programManifestUrl);
    const bootMode = canLaunchGuest ? BOOT_MODE.WASM_GUEST : BOOT_MODE.SMOKE_DEMO;

    return {
      splashImage,
      pgimUrl,
      programManifestUrl,
      gameBase,
      shaderPackUrl,
      shaderPackIndexUrl,
      reflectionManifestUrl,
      reflectionSidecarUrl,
      hasWasmBinary: runtimeWasmAvailable,
      runtimeWasm: runtimeWasmAvailable ? runtimeWasmUrl || true : null,
      runtimeWasmUrl: runtimeWasmAvailable ? runtimeWasmUrl : null,
      runtimeWasmAvailable,
      bootMode,
    };
  }

  function tracerSnapshot(bootTracer) {
    if (!bootTracer || typeof bootTracer.getSummary !== 'function') return null;
    try {
      return bootTracer.getSummary();
    } catch {
      return null;
    }
  }

  function linkSignal(outer) {
    const controller = new AbortController();
    if (!outer) return { controller, unlink: () => {} };
    if (outer.aborted) {
      controller.abort(abortReason(outer));
      return { controller, unlink: () => {} };
    }
    const onAbort = () => controller.abort(abortReason(outer));
    outer.addEventListener('abort', onAbort, { once: true });
    return { controller, unlink: () => outer.removeEventListener('abort', onAbort) };
  }

  /**
   * Coordinates `/api/status` discovery, splash overlay, PGIM preload, runtime launch
   * or smoke-demo fallback, and FIRST_PRESENT splash dismissal.
   */
  class BootOrchestrator {
    /**
     * @param {object} [options]
     * @param {typeof fetch | null} [options.fetchImpl]
     * @param {string} [options.statusUrl='/api/status']
     * @param {object|null} [options.splashOverlay]
     * @param {object|null} [options.preloader]
     * @param {object|null} [options.bootTracer]
     * @param {Function|null} [options.onLaunchRuntime]
     * @param {Function|null} [options.onFallbackSmokeDemo]
     * @param {Function|null} [options.onPhaseChange]
     * @param {Function|null} [options.onFirstPresent]
     * @param {Function|null} [options.onStageStall]
     * @param {(() => number)|null} [options.nowFn]
     * @param {number|null} [options.stageTimeoutMs=0]
     * @param {boolean} [options.parallelPreload=true]
     * @param {number|null} [options.preloadConcurrency=8]
     * @param {boolean} [options.releasePreloadBuffers=true]
     * @param {Function|null} [options.preloadRuntimeWasm=null]
     * @param {Function|null} [options.opfsProbe=null]
     * @param {Function|null} [options.preloadShaderPackHeader=null]
     * @param {Function|null} [options.preloadReflectionSidecar=null]
     * @param {Array<{name: string, run: Function, optional?: boolean, asset?: string}>|null} [options.preloadTasks=null]
     */
    constructor(options = {}) {
      if (options !== undefined && (options === null || typeof options !== 'object')) {
        throw new TypeError('BootOrchestrator options must be an object');
      }

      const {
        fetchImpl = defaultFetchImpl(),
        statusUrl = DEFAULT_STATUS_URL,
        splashOverlay = null,
        preloader = null,
        bootTracer = null,
        onLaunchRuntime = null,
        onFallbackSmokeDemo = null,
        onPhaseChange = null,
        onFirstPresent = null,
        onStageStall = null,
        nowFn = null,
        stageTimeoutMs = DEFAULT_STAGE_TIMEOUT_MS,
        parallelPreload = DEFAULT_PARALLEL_PRELOAD,
        preloadConcurrency = DEFAULT_PRELOAD_CONCURRENCY,
        releasePreloadBuffers = DEFAULT_RELEASE_PRELOAD_BUFFERS,
        preloadRuntimeWasm = null,
        opfsProbe = null,
        preloadShaderPackHeader = null,
        preloadReflectionSidecar = null,
        preloadTasks = null,
        streamShaders = undefined,
        streamThresholdBytes = undefined,
        getStreamConfig = null,
      } = options;

      if (fetchImpl !== null && fetchImpl !== undefined && typeof fetchImpl !== 'function') {
        throw new TypeError('fetchImpl must be a function or null');
      }
      if (typeof statusUrl !== 'string' || statusUrl.length === 0) {
        throw new TypeError('statusUrl must be a non-empty string');
      }
      const resolvedNowFn = optionalFn(nowFn, 'nowFn') || defaultNow;
      const resolvedStageTimeoutMs = normalizeStageTimeoutMs(stageTimeoutMs);
      const resolvedParallelPreload = normalizeParallelPreload(parallelPreload);
      const resolvedPreloadConcurrency = normalizePreloadConcurrency(preloadConcurrency, 'preloadConcurrency');
      const resolvedReleasePreloadBuffers = normalizeReleasePreloadBuffers(releasePreloadBuffers);
      const resolvedPreloadRuntimeWasm = optionalFn(preloadRuntimeWasm, 'preloadRuntimeWasm');
      const resolvedOpfsProbe = optionalFn(opfsProbe, 'opfsProbe');
      const resolvedPreloadShaderPackHeader = optionalFn(preloadShaderPackHeader, 'preloadShaderPackHeader');
      const resolvedPreloadReflectionSidecar = optionalFn(preloadReflectionSidecar, 'preloadReflectionSidecar');
      const resolvedPreloadTasks = normalizePreloadTasks(preloadTasks);
      const resolvedStreamShaders = normalizeOptionalStreamShaders(streamShaders);
      const resolvedStreamThresholdBytes = normalizeOptionalStreamThresholdBytes(streamThresholdBytes);
      const resolvedGetStreamConfig = optionalFn(getStreamConfig, 'getStreamConfig');

      this._fetchImpl = fetchImpl || null;
      this._statusUrl = statusUrl;
      this._splashOverlay = optionalObject(splashOverlay, 'splashOverlay');
      this._preloader = optionalObject(preloader, 'preloader');
      this._bootTracer = optionalObject(bootTracer, 'bootTracer');
      this._onLaunchRuntime = optionalFn(onLaunchRuntime, 'onLaunchRuntime');
      this._onFallbackSmokeDemo = optionalFn(onFallbackSmokeDemo, 'onFallbackSmokeDemo');
      this._onPhaseChange = optionalFn(onPhaseChange, 'onPhaseChange');
      this._onFirstPresent = optionalFn(onFirstPresent, 'onFirstPresent');
      this._onStageStall = optionalFn(onStageStall, 'onStageStall');
      this._nowFn = resolvedNowFn;
      this._stageTimeoutMs = resolvedStageTimeoutMs;
      this._parallelPreload = resolvedParallelPreload;
      this._preloadConcurrency = resolvedPreloadConcurrency;
      this._releasePreloadBuffers = resolvedReleasePreloadBuffers;
      this._preloadRuntimeWasm = resolvedPreloadRuntimeWasm;
      this._opfsProbe = resolvedOpfsProbe;
      this._preloadShaderPackHeader = resolvedPreloadShaderPackHeader;
      this._preloadReflectionSidecar = resolvedPreloadReflectionSidecar;
      this._preloadTasks = resolvedPreloadTasks;
      this._streamShaders = resolvedStreamShaders;
      this._streamThresholdBytes = resolvedStreamThresholdBytes;
      this._getStreamConfig = resolvedGetStreamConfig;
      this._activeStartStreamConfig = null;
      this._statusResponseHasHeaders = false;
      this._lastClockMs = 0;

      this._phase = BOOT_PHASE.IDLE;
      this._bootMode = BOOT_MODE.PENDING;
      this._status = null;
      this._assets = null;
      this._bundle = null;
      this._progress = null;
      this._fallbackReason = null;
      this._error = null;
      this._firstPresent = false;
      this._firstPresentSource = null;
      this._destroyed = false;
      this._aborted = false;
      this._pgimLoadMarked = false;
      this._startPromise = null;
      this._sessionToken = 0;
      this._abortUnlink = null;
      this._controller = new AbortController();

      this._stageDurationsMs = Object.create(null);
      this._stageEnteredAt = this._readNow();
      this._stalled = false;
      this._stalledStage = null;
      this._stalledMs = 0;
      this._stallCount = 0;
      this._preloadStats = emptyPreloadStats(this._parallelPreload, this._preloadConcurrency);
      this._wireBytesReleased = 0;
      this._droppedWasmBuffers = 0;
      this._droppedPreloadBuffers = 0;
      this._transferredArrayBuffers = 0;
    }

    _readNow() {
      try {
        const value = this._nowFn();
        if (typeof value === 'number' && Number.isFinite(value)) {
          this._lastClockMs = value;
          return value;
        }
      } catch {
        // Fall back to last known finite clock reading.
      }
      return this._lastClockMs;
    }

    _cloneStageDurations() {
      const out = {};
      for (const phase of Object.keys(this._stageDurationsMs)) {
        out[phase] = clampNonNegativeFinite(this._stageDurationsMs[phase]);
      }
      return out;
    }

    _resetTimingTelemetry(now = this._readNow()) {
      this._stageDurationsMs = Object.create(null);
      this._stageEnteredAt = now;
      this._stalled = false;
      this._stalledStage = null;
      this._stalledMs = 0;
      this._stallCount = 0;
      this._preloadStats = emptyPreloadStats(this._parallelPreload, this._preloadConcurrency);
      this._wireBytesReleased = 0;
      this._droppedWasmBuffers = 0;
      this._droppedPreloadBuffers = 0;
      this._transferredArrayBuffers = 0;
    }

    _buildTimingSnapshot(now = this._readNow()) {
      const watchdog = this._evaluateWatchdog(now);
      const stageDurationsMs = this._cloneStageDurations();
      const phaseTimingsMs = this._cloneStageDurations();
      const totalStageDurationMs = sumStageDurations(stageDurationsMs, watchdog.currentStageElapsedMs);
      const totalBootMs = totalStageDurationMs;
      const preloadStats = clonePreloadStats(this._preloadStats);
      const preloadTaskTimingsMs = { ...preloadStats.taskTimingsMs };
      return {
        watchdog,
        stageDurationsMs,
        phaseTimingsMs,
        totalStageDurationMs,
        totalBootMs,
        parallelPreloadSavedMs: preloadStats.parallelPreloadSavedMs,
        serialEstimateMs: preloadStats.serialEstimateMs,
        parallelWallMs: preloadStats.parallelWallMs,
        overlapSavedMs: preloadStats.overlapSavedMs,
        parallelStagesCount: preloadStats.parallelStagesCount,
        preloadTaskTimingsMs,
        preloadStats,
      };
    }

    /**
     * Return structured phase-duration and parallel-preload timing telemetry.
     */
    getTimingStats() {
      const timing = this._buildTimingSnapshot();
      return {
        phase: this._phase,
        bootMode: this._bootMode,
        phaseTimingsMs: timing.phaseTimingsMs,
        stageDurationsMs: timing.stageDurationsMs,
        currentStageElapsedMs: timing.watchdog.currentStageElapsedMs,
        totalBootMs: timing.totalBootMs,
        totalStageDurationMs: timing.totalStageDurationMs,
        parallelPreloadEnabled: this._parallelPreload,
        preloadConcurrency: this._preloadConcurrency,
        parallelPreloadSavedMs: timing.parallelPreloadSavedMs,
        serialEstimateMs: timing.serialEstimateMs,
        parallelWallMs: timing.parallelWallMs,
        overlapSavedMs: timing.overlapSavedMs,
        parallelStagesCount: timing.parallelStagesCount,
        preloadTaskTimingsMs: timing.preloadTaskTimingsMs,
        preloadStats: timing.preloadStats,
      };
    }

    _readOpfsTelemetry() {
      const preloader = this._preloader;
      let stats = null;
      if (preloader && typeof preloader.getStats === 'function') {
        try {
          stats = preloader.getStats();
        } catch {
          stats = null;
        }
      }
      const opfsHits = clampNonNegativeInt(stats && stats.opfsHits);
      const opfsMisses = clampNonNegativeInt(stats && stats.opfsMisses);
      const opfsWrites = clampNonNegativeInt(stats && stats.opfsWrites);
      const opfsInvalidations = clampNonNegativeInt(stats && stats.opfsInvalidations);
      const bytesFromOpfs = clampNonNegativeInt(stats && stats.bytesFromOpfs);
      const bytesFromNetwork = clampNonNegativeInt(stats && stats.bytesFromNetwork);
      const opfsHitRatePct =
        stats &&
        typeof stats.opfsHitRatePct === 'number' &&
        Number.isFinite(stats.opfsHitRatePct) &&
        stats.opfsHitRatePct >= 0
          ? Math.min(100, stats.opfsHitRatePct)
          : computeOpfsHitRatePct(opfsHits, opfsMisses);
      const opfsEnabled =
        stats && typeof stats.opfsEnabled === 'boolean'
          ? stats.opfsEnabled
          : Boolean(
              preloader &&
                (preloader.opfsEnabled === true ||
                  (preloader._opfsStore !== undefined && preloader._opfsStore !== null))
            );
      return {
        opfsEnabled,
        opfsHits,
        opfsMisses,
        opfsWrites,
        opfsInvalidations,
        bytesFromOpfs,
        bytesFromNetwork,
        opfsHitRatePct,
      };
    }

    /**
     * Wait for any background non-blocking OPFS write-through tasks owned by the preloader.
     */
    async flushOpfsWrites() {
      const preloader = this._preloader;
      if (preloader && typeof preloader.flushOpfsWrites === 'function') {
        await preloader.flushOpfsWrites();
      }
    }

    /**
     * Return sanitized boot orchestrator timing, concurrency, and overlap savings statistics
     * (`serialEstimateMs`, `parallelWallMs`, `overlapSavedMs`, `parallelStagesCount`) suitable
     * for telemetry panels and JSON diagnostics without binary bundle references.
     */
    getStats() {
      const timing = this._buildTimingSnapshot();
      const { watchdog } = timing;
      const opfs = this._readOpfsTelemetry();
      return {
        phase: this._phase,
        bootMode: this._bootMode,
        firstPresent: this._firstPresent,
        firstPresentSource: this._firstPresentSource,
        parallelPreload: this._parallelPreload,
        parallelPreloadEnabled: this._parallelPreload,
        preloadConcurrency: this._preloadConcurrency,
        phaseTimingsMs: timing.phaseTimingsMs,
        stageDurationsMs: timing.stageDurationsMs,
        currentStageElapsedMs: watchdog.currentStageElapsedMs,
        totalBootMs: timing.totalBootMs,
        totalStageDurationMs: timing.totalStageDurationMs,
        serialEstimateMs: timing.serialEstimateMs,
        parallelWallMs: timing.parallelWallMs,
        overlapSavedMs: timing.overlapSavedMs,
        parallelPreloadSavedMs: timing.parallelPreloadSavedMs,
        parallelStagesCount: timing.parallelStagesCount,
        preloadTaskTimingsMs: timing.preloadTaskTimingsMs,
        preloadStats: timing.preloadStats,
        isStalled: watchdog.isStalled,
        stalledStage: watchdog.stalledStage,
        stalledMs: watchdog.stalledMs,
        stallCount: this._stallCount,
        wireBytesReleased: this._wireBytesReleased,
        droppedWasmBuffers: this._droppedWasmBuffers,
        droppedPreloadBuffers: this._droppedPreloadBuffers,
        transferredArrayBuffers: this._transferredArrayBuffers,
        ...opfs,
      };
    }

    /**
     * Return sanitized numeric memory-drop and zero-copy handoff telemetry for Chrome heap verification.
     */
    getMemoryDropTelemetry() {
      return {
        wireBytesReleased: this._wireBytesReleased,
        droppedWasmBuffers: this._droppedWasmBuffers,
        droppedPreloadBuffers: this._droppedPreloadBuffers,
        transferredArrayBuffers: this._transferredArrayBuffers,
        bundleRetained: this._bundle !== null,
      };
    }

    /**
     * Run a batch of independent preload tasks using this orchestrator's clock, concurrency limit,
     * and `parallelPreload` configuration.
     */
    async runConcurrentPreloads(tasks, options = {}) {
      if (options !== undefined && (options === null || typeof options !== 'object')) {
        throw new TypeError('runConcurrentPreloads options must be an object');
      }
      const parallel =
        options.parallel !== undefined ? options.parallel : this._parallelPreload;
      const maxConcurrency =
        options.maxConcurrency !== undefined
          ? options.maxConcurrency
          : options.concurrency !== undefined
            ? options.concurrency
            : this._preloadConcurrency;
      const nowFn = options.nowFn !== undefined ? options.nowFn : () => this._readNow();
      return runConcurrentPreloads(tasks, {
        ...options,
        parallel,
        maxConcurrency,
        nowFn,
      });
    }

    /**
     * Return a cloned snapshot of parallel preload timing and task status statistics.
     */
    getPreloadStats() {
      return clonePreloadStats(this._preloadStats);
    }

    /**
     * Release any retained large binary preload buffers (`shaderPackBytes`, `pgim.bytes`/`buffer`,
     * `reflectionManifest.bytes`, `reflectionSidecar.bytes`, `sidecars[k].bytes`/`buffer`) and
     * drop internal bundle references while preserving `getPreloadStats()`, `getState()`, and
     * `assets`. Idempotent: calling again returns `{ releasedFields: [], releasedBytes: 0 }`.
     *
     * @returns {{ releasedFields: string[], releasedBytes: number }}
     */
    releasePreloadBuffers() {
      const bundle = this._bundle;
      this._bundle = null;
      if (!bundle) {
        return { releasedFields: [], releasedBytes: 0 };
      }
      const released = releaseBootBundleBytes(bundle);
      if (released.releasedBytes > 0) {
        this._wireBytesReleased += released.releasedBytes;
      }
      if (released.releasedFields.length > 0) {
        this._droppedPreloadBuffers += released.releasedFields.length;
        for (let i = 0; i < released.releasedFields.length; i += 1) {
          if (WASM_FIELD_PATH_RE.test(released.releasedFields[i])) {
            this._droppedWasmBuffers += 1;
          }
        }
      }
      return released;
    }

    /**
     * Transfer the currently held preload bundle's standalone non-SAB `ArrayBuffer`s to `worker`
     * via zero-copy `postMessage(msg, transferList)` and immediately drop all Window-side references.
     *
     * @param {{ postMessage: Function }} worker
     * @param {object} [message]
     * @param {{ extraTransferables?: unknown[] }} [options]
     */
    transferBundleToWorker(worker, message = null, options = {}) {
      const bundle = this._bundle;
      const payload =
        message && typeof message === 'object'
          ? { ...message, ...(bundle && message.bundle === undefined ? { bundle } : {}) }
          : { bundle };
      const result = transferPreloadPayloadToWorker(worker, payload, {
        ...(options && typeof options === 'object' ? options : {}),
        releaseTarget: bundle || payload,
      });
      this._bundle = null;
      if (result.transferredBuffers > 0) {
        this._transferredArrayBuffers += result.transferredBuffers;
      }
      if (result.releasedBytes > 0) {
        this._wireBytesReleased += result.releasedBytes;
      } else if (result.transferredBytes > 0) {
        this._wireBytesReleased += result.transferredBytes;
      }
      if (result.releasedFields.length > 0) {
        this._droppedPreloadBuffers += result.releasedFields.length;
        for (let i = 0; i < result.releasedFields.length; i += 1) {
          if (WASM_FIELD_PATH_RE.test(result.releasedFields[i])) {
            this._droppedWasmBuffers += 1;
          }
        }
      }
      return result;
    }

    _evaluateWatchdog(now = this._readNow()) {
      const active = WATCHED_BOOT_PHASES.has(this._phase);
      const currentStageElapsedMs = active ? clampNonNegativeFinite(now - this._stageEnteredAt) : 0;
      const enabled = this._stageTimeoutMs > 0;
      const isStalled = active && enabled && currentStageElapsedMs >= this._stageTimeoutMs;
      return {
        currentStageElapsedMs,
        isStalled,
        stalledStage: isStalled ? this._phase : null,
        stalledMs: isStalled ? currentStageElapsedMs : 0,
      };
    }

    /**
     * Evaluate whether the current active boot stage has exceeded `stageTimeoutMs`.
     * Non-destructive: never aborts or mutates phase transitions.
     *
     * @returns {{
     *   isStalled: boolean,
     *   stalledStage: string | null,
     *   stalledMs: number,
     *   stageTimeoutMs: number,
     *   phase: string,
     *   currentStageElapsedMs: number,
     * }}
     */
    checkStageStall() {
      const now = this._readNow();
      const watchdog = this._evaluateWatchdog(now);

      if (watchdog.isStalled) {
        const newlyStalled = !this._stalled || this._stalledStage !== watchdog.stalledStage;
        this._stalled = true;
        this._stalledStage = watchdog.stalledStage;
        this._stalledMs = watchdog.stalledMs;
        if (newlyStalled) {
          this._stallCount += 1;
          invokeQuiet(this._onStageStall, {
            phase: this._phase,
            bootMode: this._bootMode,
            stalledStage: watchdog.stalledStage,
            stalledMs: watchdog.stalledMs,
            stageTimeoutMs: this._stageTimeoutMs,
            currentStageElapsedMs: watchdog.currentStageElapsedMs,
            stageDurationsMs: this._cloneStageDurations(),
            stallCount: this._stallCount,
          });
        }
      } else {
        this._stalled = false;
        this._stalledStage = null;
        this._stalledMs = 0;
      }

      return {
        isStalled: watchdog.isStalled,
        stalledStage: watchdog.stalledStage,
        stalledMs: watchdog.stalledMs,
        stageTimeoutMs: this._stageTimeoutMs,
        phase: this._phase,
        currentStageElapsedMs: watchdog.currentStageElapsedMs,
      };
    }

    /**
     * Returns true ONLY when the orchestrator is running a real Wasm64 guest boot
     * (`bootMode === 'wasm-guest'`).
     */
    isRealGuestBoot() {
      return this._bootMode === BOOT_MODE.WASM_GUEST;
    }

    getState() {
      const timing = this._buildTimingSnapshot();
      const { watchdog } = timing;
      return {
        phase: this._phase,
        bootMode: this._bootMode,
        programManifestUrl: this._assets ? this._assets.programManifestUrl : null,
        gameBase: this._assets ? this._assets.gameBase : null,
        status: this._status ? { ...this._status } : null,
        assets: this._assets ? { ...this._assets } : null,
        bundle: this._bundle,
        progress: this._progress ? { ...this._progress } : null,
        tracer: tracerSnapshot(this._bootTracer),
        fallbackReason: this._fallbackReason,
        error: this._error ? this._error.message : null,
        firstPresent: this._firstPresent,
        firstPresentSource: this._firstPresentSource,
        destroyed: this._destroyed,
        aborted: this._aborted,
        parallelPreload: this._parallelPreload,
        preloadConcurrency: this._preloadConcurrency,
        stageDurationsMs: timing.stageDurationsMs,
        phaseTimingsMs: timing.phaseTimingsMs,
        currentStageElapsedMs: watchdog.currentStageElapsedMs,
        totalStageDurationMs: timing.totalStageDurationMs,
        totalBootMs: timing.totalBootMs,
        parallelPreloadSavedMs: timing.parallelPreloadSavedMs,
        serialEstimateMs: timing.serialEstimateMs,
        parallelWallMs: timing.parallelWallMs,
        overlapSavedMs: timing.overlapSavedMs,
        parallelStagesCount: timing.parallelStagesCount,
        preloadTaskTimingsMs: timing.preloadTaskTimingsMs,
        preloadStats: timing.preloadStats,
        stageTimeoutMs: this._stageTimeoutMs,
        isStalled: watchdog.isStalled,
        stalledStage: watchdog.stalledStage,
        stalledMs: watchdog.stalledMs,
        stallCount: this._stallCount,
        wireBytesReleased: this._wireBytesReleased,
        droppedWasmBuffers: this._droppedWasmBuffers,
        droppedPreloadBuffers: this._droppedPreloadBuffers,
        transferredArrayBuffers: this._transferredArrayBuffers,
      };
    }

    /**
     * Export a JSON-safe, sanitized diagnostic snapshot with host paths, query strings,
     * and auth tokens redacted and binary bundle payloads omitted.
     */
    getDiagnosticSnapshot() {
      const timing = this._buildTimingSnapshot();
      const { watchdog } = timing;
      const opfs = this._readOpfsTelemetry();
      return {
        phase: this._phase,
        bootMode: this._bootMode,
        firstPresent: this._firstPresent,
        firstPresentSource: this._firstPresentSource,
        destroyed: this._destroyed,
        aborted: this._aborted,
        parallelPreload: this._parallelPreload,
        preloadConcurrency: this._preloadConcurrency,
        fallbackReason: this._fallbackReason ? sanitizeDiagnosticText(this._fallbackReason) : null,
        error: this._error ? sanitizeDiagnosticText(this._error.message) : null,
        stageDurationsMs: timing.stageDurationsMs,
        phaseTimingsMs: timing.phaseTimingsMs,
        currentStageElapsedMs: watchdog.currentStageElapsedMs,
        totalStageDurationMs: timing.totalStageDurationMs,
        totalBootMs: timing.totalBootMs,
        parallelPreloadSavedMs: timing.parallelPreloadSavedMs,
        serialEstimateMs: timing.serialEstimateMs,
        parallelWallMs: timing.parallelWallMs,
        overlapSavedMs: timing.overlapSavedMs,
        parallelStagesCount: timing.parallelStagesCount,
        preloadTaskTimingsMs: timing.preloadTaskTimingsMs,
        preloadStats: timing.preloadStats,
        stageTimeoutMs: this._stageTimeoutMs,
        isStalled: watchdog.isStalled,
        stalledStage: watchdog.stalledStage,
        stalledMs: watchdog.stalledMs,
        stallCount: this._stallCount,
        wireBytesReleased: this._wireBytesReleased,
        droppedWasmBuffers: this._droppedWasmBuffers,
        droppedPreloadBuffers: this._droppedPreloadBuffers,
        transferredArrayBuffers: this._transferredArrayBuffers,
        ...opfs,
        tracer: tracerSnapshot(this._bootTracer),
      };
    }

    _setPhase(nextPhase, detail = null) {
      if (this._phase === nextPhase) return;
      const now = this._readNow();
      const previousPhase = this._phase;
      if (WATCHED_BOOT_PHASES.has(previousPhase)) {
        const elapsedMs = clampNonNegativeFinite(now - this._stageEnteredAt);
        const prevAccum = clampNonNegativeFinite(this._stageDurationsMs[previousPhase]);
        this._stageDurationsMs[previousPhase] = clampNonNegativeFinite(prevAccum + elapsedMs);
      }
      this._phase = nextPhase;
      this._stageEnteredAt = now;
      this._stalled = false;
      this._stalledStage = null;
      this._stalledMs = 0;

      const stageDurationsMs = this._cloneStageDurations();
      const phaseTimingsMs = this._cloneStageDurations();
      const totalBootMs = sumStageDurations(stageDurationsMs, 0);
      const parallelPreloadSavedMs = clampNonNegativeFinite(this._preloadStats.parallelPreloadSavedMs);
      const serialEstimateMs = clampNonNegativeFinite(this._preloadStats.serialEstimateMs);
      const parallelWallMs = clampNonNegativeFinite(this._preloadStats.parallelWallMs);
      const overlapSavedMs = clampNonNegativeFinite(this._preloadStats.overlapSavedMs);
      const parallelStagesCount =
        Number.isInteger(this._preloadStats.parallelStagesCount) && this._preloadStats.parallelStagesCount >= 0
          ? this._preloadStats.parallelStagesCount
          : 0;

      const event = {
        phase: nextPhase,
        previousPhase,
        bootMode: this._bootMode,
        programManifestUrl: this._assets ? this._assets.programManifestUrl : null,
        gameBase: this._assets ? this._assets.gameBase : null,
        firstPresentSource: this._firstPresentSource,
        status: this._status,
        assets: this._assets,
        bundle: this._bundle,
        progress: this._progress,
        tracer: tracerSnapshot(this._bootTracer),
        fallbackReason: this._fallbackReason,
        error: this._error,
        stageDurationsMs,
        phaseTimingsMs,
        totalBootMs,
        parallelPreloadSavedMs,
        serialEstimateMs,
        parallelWallMs,
        overlapSavedMs,
        parallelStagesCount,
        ...(detail && typeof detail === 'object' ? detail : {}),
      };
      invokeQuiet(this._onPhaseChange, event);
      // Never pass raw binary preload bundles into the splash overlay DOM controller.
      this._pushOverlay({ phase: nextPhase, ...event, bundle: null });
    }

    _pushOverlay(payload) {
      const overlay = this._splashOverlay;
      if (!overlay) return;
      callQuiet(overlay, 'update', payload);
      if (payload && payload.progress) {
        callQuiet(overlay, 'updateProgress', payload.progress);
      }
      if (payload && payload.tracer) {
        callQuiet(overlay, 'updateTracer', payload.tracer);
        callQuiet(overlay, 'setTracerSummary', payload.tracer);
      }
    }

    _initOverlay(splashImage) {
      const overlay = this._splashOverlay;
      if (!overlay) return;
      const tracer = tracerSnapshot(this._bootTracer);
      const payload = {
        splashImage,
        bootMode: this._bootMode,
        programManifestUrl: this._assets ? this._assets.programManifestUrl : null,
        gameBase: this._assets ? this._assets.gameBase : null,
        status: this._status,
        assets: this._assets,
        tracer,
        phase: this._phase,
      };
      if (typeof overlay.init === 'function') {
        callQuiet(overlay, 'init', payload);
      } else {
        callQuiet(overlay, 'show', payload);
      }
      if (splashImage) {
        callQuiet(overlay, 'setSplashImage', splashImage);
        callQuiet(overlay, 'setImage', splashImage);
      }
      this._pushOverlay(payload);
    }

    _forwardProgress(event) {
      const progress = event && typeof event === 'object' ? event : { loadedBytes: 0, totalBytes: null, pct: 0 };
      this._progress = {
        phase: progress.phase || 'download',
        asset: progress.asset || 'bundle',
        loadedBytes: Number.isFinite(progress.loadedBytes) ? progress.loadedBytes : 0,
        totalBytes: Number.isFinite(progress.totalBytes) ? progress.totalBytes : null,
        pct: Number.isFinite(progress.pct) ? progress.pct : 0,
        ...(typeof progress.reason === 'string' && progress.reason.length > 0
          ? { reason: progress.reason }
          : {}),
      };
      callQuiet(this._bootTracer, 'recordProgress', this._progress);
      callQuiet(this._bootTracer, 'updateProgress', this._progress);

      const completedPgim =
        (this._progress.asset === 'pgim' || this._progress.asset === 'bundle') &&
        this._progress.phase === 'complete';
      if (completedPgim && !this._pgimLoadMarked && this._bootTracer && typeof this._bootTracer.markMilestone === 'function') {
        this._pgimLoadMarked = true;
        try {
          this._bootTracer.markMilestone('PGIM_LOAD', {
            loadedBytes: this._progress.loadedBytes,
            totalBytes: this._progress.totalBytes,
          });
        } catch {
          // Milestone names are fixed; ignore tracer validation errors.
        }
      }

      this._pushOverlay({
        phase: this._phase,
        bootMode: this._bootMode,
        progress: this._progress,
        tracer: tracerSnapshot(this._bootTracer),
        status: this._status,
      });
    }

    async _fetchStatus(signal) {
      if (!this._fetchImpl) {
        throw new Error(`Unable to fetch ${this._statusUrl}: fetch API is unavailable`);
      }
      throwIfAborted(signal);
      const response = await this._fetchImpl(this._statusUrl, { cache: 'no-store', signal });
      throwIfAborted(signal);
      if (!response || typeof response.ok !== 'boolean') {
        throw new Error(`Invalid response from ${this._statusUrl}`);
      }
      if (!response.ok) {
        const err = new Error(`Status probe ${this._statusUrl} failed with HTTP ${response.status ?? 'unknown'}`);
        err.status = response.status ?? null;
        throw err;
      }
      this._statusResponseHasHeaders = Boolean(response.headers && typeof response.headers.get === 'function');
      const payload = await response.json();
      throwIfAborted(signal);
      if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
        throw new Error(`Status probe ${this._statusUrl} returned non-object JSON`);
      }
      return payload;
    }

    _resolveRuntimeWasmLoader() {
      if (this._preloadRuntimeWasm) return this._preloadRuntimeWasm;
      const preloader = this._preloader;
      if (!preloader) return null;
      if (typeof preloader.preloadRuntimeWasm === 'function') {
        return preloader.preloadRuntimeWasm.bind(preloader);
      }
      if (typeof preloader.compileRuntimeWasm === 'function') {
        return preloader.compileRuntimeWasm.bind(preloader);
      }
      if (typeof preloader.preloadWasm === 'function') {
        return preloader.preloadWasm.bind(preloader);
      }
      return null;
    }

    _resolveOpfsProbe() {
      if (this._opfsProbe) return this._opfsProbe;
      const preloader = this._preloader;
      if (!preloader) return null;
      if (typeof preloader.probeOpfsWarmBoot === 'function') {
        return preloader.probeOpfsWarmBoot.bind(preloader);
      }
      if (typeof preloader.probeOpfs === 'function') {
        return preloader.probeOpfs.bind(preloader);
      }
      return null;
    }

    _resolveShaderPackHeaderLoader() {
      if (this._preloadShaderPackHeader) return this._preloadShaderPackHeader;
      const preloader = this._preloader;
      if (!preloader) return null;
      if (typeof preloader.preloadShaderPackHeader === 'function') {
        return preloader.preloadShaderPackHeader.bind(preloader);
      }
      if (typeof preloader.preloadShaderPackIndex === 'function') {
        return preloader.preloadShaderPackIndex.bind(preloader);
      }
      return null;
    }

    _resolveReflectionSidecarLoader() {
      if (this._preloadReflectionSidecar) return this._preloadReflectionSidecar;
      const preloader = this._preloader;
      if (!preloader) return null;
      if (typeof preloader.preloadReflectionSidecar === 'function') {
        return preloader.preloadReflectionSidecar.bind(preloader);
      }
      if (typeof preloader.preloadSidecar === 'function') {
        return preloader.preloadSidecar.bind(preloader);
      }
      return null;
    }

    _resolveEffectiveShaderStreamPolicy() {
      let dynamicConfig = null;
      if (typeof this._getStreamConfig === 'function') {
        try {
          const raw = this._getStreamConfig();
          if (raw && typeof raw === 'object') {
            dynamicConfig = raw;
          }
        } catch {
          dynamicConfig = null;
        }
      }
      const startConfig = this._activeStartStreamConfig;
      const effectiveStreamShaders =
        startConfig && startConfig.streamShaders !== undefined
          ? startConfig.streamShaders
          : dynamicConfig && dynamicConfig.streamShaders !== undefined
            ? normalizeOptionalStreamShaders(dynamicConfig.streamShaders)
            : this._streamShaders;
      const effectiveStreamThresholdBytes =
        startConfig && startConfig.streamThresholdBytes !== undefined
          ? startConfig.streamThresholdBytes
          : dynamicConfig && dynamicConfig.streamThresholdBytes !== undefined
            ? normalizeOptionalStreamThresholdBytes(dynamicConfig.streamThresholdBytes)
            : this._streamThresholdBytes;
      return resolveShaderPackStreamPolicy({
        streamShaders: effectiveStreamShaders,
        streamThresholdBytes: effectiveStreamThresholdBytes,
      });
    }

    async _probeShaderPackSizeBytes(url, signal) {
      const fromStatus = extractStatusShaderPackSizeBytes(this._status);
      if (fromStatus !== null) {
        return fromStatus;
      }
      if (this._preloader && typeof this._preloader.probeShaderPackSize === 'function') {
        throwIfAborted(signal);
        try {
          const size = await this._preloader.probeShaderPackSize(url, { signal });
          throwIfAborted(signal);
          return typeof size === 'number' && Number.isSafeInteger(size) && size >= 0 ? size : null;
        } catch (err) {
          if ((signal && signal.aborted) || isAbortError(err)) {
            throw err;
          }
          return null;
        }
      }
      if (!this._statusResponseHasHeaders || typeof this._fetchImpl !== 'function') {
        return null;
      }
      throwIfAborted(signal);
      try {
        const response = await this._fetchImpl(url, { method: 'HEAD', signal });
        throwIfAborted(signal);
        if (!response || !response.ok) {
          return null;
        }
        return parseResponseContentLength(response);
      } catch (err) {
        if ((signal && signal.aborted) || isAbortError(err)) {
          throw err;
        }
        return null;
      }
    }

    async _preloadBundle(assets, signal) {
      const preloader = this._preloader;
      const shaderPackIndexLoader = this._resolveShaderPackHeaderLoader();
      const reflectionSidecarLoader = this._resolveReflectionSidecarLoader();
      const runtimeWasmLoader = this._resolveRuntimeWasmLoader();
      const opfsProbe = this._resolveOpfsProbe();

      const shaderStreamPolicy = this._resolveEffectiveShaderStreamPolicy();
      let skipShaderPackFullPreload = false;
      let shaderPackSkipReason = null;
      let shaderPackProbedBytes = null;

      if (assets.shaderPackUrl) {
        if (shaderStreamPolicy.skipFullPreload) {
          skipShaderPackFullPreload = true;
          shaderPackSkipReason = 'stream-shaders';
        } else if (shaderStreamPolicy.maxPreloadBytes > 0) {
          const probedBytes = await this._probeShaderPackSizeBytes(assets.shaderPackUrl, signal);
          if (probedBytes !== null && probedBytes >= shaderStreamPolicy.maxPreloadBytes) {
            skipShaderPackFullPreload = true;
            shaderPackSkipReason = 'exceeds-preload-threshold';
            shaderPackProbedBytes = probedBytes;
          }
        }
      }

      if (skipShaderPackFullPreload) {
        this._forwardProgress({
          phase: 'skipped',
          asset: 'shaderPack',
          reason: shaderPackSkipReason,
          loadedBytes: 0,
          totalBytes: shaderPackProbedBytes || 0,
          pct: 100,
        });
      }

      const effectiveShaderPackUrl = skipShaderPackFullPreload ? null : assets.shaderPackUrl;

      const needsPreloader = Boolean(
        assets.pgimUrl ||
        (effectiveShaderPackUrl && !this._preloadShaderPackHeader && !this._preloadReflectionSidecar) ||
        (assets.shaderPackIndexUrl && !this._preloadShaderPackHeader) ||
        (assets.reflectionSidecarUrl && !this._preloadReflectionSidecar)
      );
      if (needsPreloader && !preloader) {
        throw new Error('Boot artifacts are present but no preloader is configured');
      }

      const onProgress = (event) => this._forwardProgress(event);
      const forwardFor = (asset) => (event) => {
        const safeEvent = event && typeof event === 'object' ? event : {};
        this._forwardProgress({ ...safeEvent, asset: safeEvent.asset || asset });
      };

      const hasBootBundle = Boolean(preloader && typeof preloader.preloadBootBundle === 'function');
      const canPreloadProgramManifest = Boolean(
        assets.programManifestUrl && preloader && typeof preloader.preloadProgramModules === 'function'
      );
      const pgimLoader =
        preloader && typeof preloader.preloadPgim === 'function'
          ? preloader.preloadPgim.bind(preloader)
          : preloader && typeof preloader.preloadGuestImage === 'function'
            ? preloader.preloadGuestImage.bind(preloader)
            : null;

      if (assets.pgimUrl && !hasBootBundle && !pgimLoader) {
        throw new TypeError('preloader.preloadBootBundle must be a function');
      }

      const hasPrimaryGuestAsset = Boolean(assets.pgimUrl || canPreloadProgramManifest);
      const useBootBundle = Boolean(
        hasPrimaryGuestAsset &&
          hasBootBundle &&
          (this._parallelPreload || (!pgimLoader && !canPreloadProgramManifest))
      );
      const tasks = [];

      if (useBootBundle) {
        tasks.push({
          name: 'bootBundle',
          asset: 'bundle',
          kind: 'bootBundle',
          optional: false,
          run: (taskSignal) => {
            const bundleArgs = {
              pgimUrl: assets.pgimUrl,
              shaderPackUrl: effectiveShaderPackUrl,
              reflectionUrl: assets.reflectionManifestUrl,
              onProgress,
              signal: taskSignal,
            };
            if (assets.programManifestUrl) {
              bundleArgs.programManifestUrl = assets.programManifestUrl;
            }
            if (effectiveShaderPackUrl) {
              if (typeof shaderStreamPolicy.streamShaders === 'boolean') {
                bundleArgs.streamShaders = shaderStreamPolicy.streamShaders;
              }
              bundleArgs.streamThresholdBytes = shaderStreamPolicy.maxPreloadBytes;
            }
            return preloader.preloadBootBundle(bundleArgs);
          },
        });
      } else {
        if (canPreloadProgramManifest) {
          tasks.push({
            name: 'pgim',
            asset: 'pgim',
            kind: 'pgim',
            optional: false,
            run: (taskSignal) =>
              preloader.preloadProgramModules(assets.programManifestUrl, {
                fallbackPgimUrl: assets.pgimUrl || null,
                onProgress: forwardFor('pgim'),
                signal: taskSignal,
              }),
          });
        } else if (assets.pgimUrl && pgimLoader) {
          tasks.push({
            name: 'pgim',
            asset: 'pgim',
            kind: 'pgim',
            optional: false,
            run: (taskSignal) => pgimLoader(assets.pgimUrl, { onProgress: forwardFor('pgim'), signal: taskSignal }),
          });
        }
        if (effectiveShaderPackUrl && preloader && typeof preloader.preloadShaderPack === 'function') {
          tasks.push({
            name: 'shaderPack',
            asset: 'shaderPack',
            kind: 'shaderPack',
            optional: Boolean(assets.pgimUrl),
            run: (taskSignal) => {
              const shaderOpts = {
                onProgress: forwardFor('shaderPack'),
                signal: taskSignal,
                streamThresholdBytes: shaderStreamPolicy.maxPreloadBytes,
              };
              if (typeof shaderStreamPolicy.streamShaders === 'boolean') {
                shaderOpts.streamShaders = shaderStreamPolicy.streamShaders;
              }
              return preloader.preloadShaderPack(effectiveShaderPackUrl, shaderOpts);
            },
          });
        }
        if (assets.reflectionManifestUrl && preloader && typeof preloader.preloadReflectionManifest === 'function') {
          tasks.push({
            name: 'reflection',
            asset: 'reflection',
            kind: 'reflection',
            optional: true,
            run: (taskSignal) =>
              preloader.preloadReflectionManifest(assets.reflectionManifestUrl, {
                onProgress: forwardFor('reflection'),
                signal: taskSignal,
              }),
          });
        }
      }

      if (shaderPackIndexLoader && (assets.shaderPackIndexUrl || assets.shaderPackUrl)) {
        const indexUrl = assets.shaderPackIndexUrl || assets.shaderPackUrl;
        tasks.push({
          name: 'shaderPackIndex',
          asset: 'shaderPackIndex',
          kind: 'shaderPackIndex',
          optional: true,
          run: (taskSignal) =>
            shaderPackIndexLoader(indexUrl, {
              onProgress: forwardFor('shaderPackIndex'),
              signal: taskSignal,
            }),
        });
      }

      const resolvedReflectionSidecarUrl =
        assets.reflectionSidecarUrl ||
        (reflectionSidecarLoader && assets.shaderPackUrl ? deriveReflectionSidecarUrl(assets.shaderPackUrl) : null);
      if (resolvedReflectionSidecarUrl) {
        const sidecarFn =
          reflectionSidecarLoader ||
          (assets.reflectionSidecarUrl && preloader && typeof preloader.preloadReflectionManifest === 'function'
            ? preloader.preloadReflectionManifest.bind(preloader)
            : null);
        if (sidecarFn) {
          tasks.push({
            name: 'reflectionSidecar',
            asset: 'reflectionSidecar',
            kind: 'reflectionSidecar',
            optional: true,
            run: (taskSignal) =>
              sidecarFn(resolvedReflectionSidecarUrl, {
                onProgress: forwardFor('reflectionSidecar'),
                signal: taskSignal,
              }),
          });
        }
      }

      if (runtimeWasmLoader && assets.runtimeWasmAvailable && assets.runtimeWasmUrl) {
        tasks.push({
          name: 'runtimeWasm',
          asset: 'runtimeWasm',
          kind: 'runtimeWasm',
          optional: true,
          run: (taskSignal) =>
            runtimeWasmLoader(assets.runtimeWasmUrl, {
              assets: { ...assets },
              status: this._status ? { ...this._status } : null,
              onProgress: forwardFor('runtimeWasm'),
              signal: taskSignal,
            }),
        });
      }

      if (opfsProbe) {
        tasks.push({
          name: 'opfsWarmBoot',
          asset: 'opfsWarmBoot',
          kind: 'opfsWarmBoot',
          optional: true,
          run: (taskSignal) =>
            opfsProbe({
              assets: { ...assets },
              status: this._status ? { ...this._status } : null,
              onProgress: forwardFor('opfsWarmBoot'),
              signal: taskSignal,
            }),
        });
      }

      for (const customTask of this._preloadTasks) {
        const existingIdx = tasks.findIndex((entry) => entry.name === customTask.name);
        const wrappedTask = {
          name: customTask.name,
          asset: customTask.asset,
          kind: 'custom',
          optional: customTask.optional,
          run: (taskSignal) =>
            customTask.run({
              assets: { ...assets },
              status: this._status ? { ...this._status } : null,
              signal: taskSignal,
              onProgress: forwardFor(customTask.asset),
            }),
        };
        if (existingIdx >= 0) {
          tasks[existingIdx] = wrappedTask;
        } else {
          tasks.push(wrappedTask);
        }
      }

      if (tasks.length === 0) {
        this._preloadStats = emptyPreloadStats(this._parallelPreload, this._preloadConcurrency);
        return {
          pgim: null,
          shaderPackBytes: null,
          ...(skipShaderPackFullPreload
            ? {
                shaderPackSkipped: true,
                shaderPackSkipReason,
                shaderPackStreamedBytes: shaderPackProbedBytes,
              }
            : {}),
          reflectionManifest: null,
          bytesLoaded: 0,
        };
      }

      const execution = await runConcurrentPreloads(tasks, {
        parallel: this._parallelPreload,
        maxConcurrency: this._preloadConcurrency,
        signal,
        nowFn: () => this._readNow(),
        onOptionalMissing: ({ task }) => {
          this._forwardProgress({
            phase: 'optional-missing',
            asset: task.asset,
            loadedBytes: 0,
            totalBytes: 0,
            pct: 0,
          });
        },
        onComplete: (stats) => {
          this._preloadStats = stats;
        },
        throwOnError: true,
      });

      const results = execution.resultsMap;

      const customSidecars = this._preloadTasks.length > 0 ? {} : null;
      if (customSidecars) {
        for (const customTask of this._preloadTasks) {
          customSidecars[customTask.name] = results.has(customTask.name) ? results.get(customTask.name) : null;
        }
      }

      const cleanupExecutionContainers = () => {
        if (results && typeof results.clear === 'function') {
          results.clear();
        }
        if (Array.isArray(execution.settled)) {
          execution.settled.length = 0;
        }
      };

      if (useBootBundle) {
        const baseBundle = results.get('bootBundle') || null;
        if (tasks.length === 1 && !skipShaderPackFullPreload) {
          cleanupExecutionContainers();
          return baseBundle;
        }
        const merged = baseBundle && typeof baseBundle === 'object' ? { ...baseBundle } : { pgim: null };
        if (skipShaderPackFullPreload) {
          if (merged.shaderPackBytes === undefined) {
            merged.shaderPackBytes = null;
          }
          merged.shaderPackSkipped = true;
          merged.shaderPackSkipReason = shaderPackSkipReason;
          merged.shaderPackStreamedBytes = shaderPackProbedBytes;
        }
        let extraBytes = 0;
        if (results.has('shaderPackIndex')) {
          merged.shaderPackIndex = results.get('shaderPackIndex');
          extraBytes += inferBytesLoaded(merged.shaderPackIndex);
        }
        if (results.has('reflectionSidecar')) {
          merged.reflectionSidecar = results.get('reflectionSidecar');
          extraBytes += inferBytesLoaded(merged.reflectionSidecar);
        }
        if (results.has('runtimeWasm')) {
          merged.runtimeWasm = results.get('runtimeWasm');
          extraBytes += inferBytesLoaded(merged.runtimeWasm);
        }
        if (results.has('opfsWarmBoot')) {
          merged.opfsWarmBoot = results.get('opfsWarmBoot');
          extraBytes += inferBytesLoaded(merged.opfsWarmBoot);
        }
        if (customSidecars) {
          merged.sidecars = customSidecars;
          for (const key of Object.keys(customSidecars)) {
            extraBytes += inferBytesLoaded(customSidecars[key]);
          }
        }
        const baseBytes = inferBytesLoaded(baseBundle);
        merged.bytesLoaded = baseBytes + extraBytes;
        cleanupExecutionContainers();
        return merged;
      }

      const pgimResult = results.has('pgim') ? results.get('pgim') : null;
      const shaderPackResult = results.has('shaderPack') ? results.get('shaderPack') : null;
      const shaderPackIndexResult = results.has('shaderPackIndex') ? results.get('shaderPackIndex') : null;
      const reflectionResult = results.has('reflection') ? results.get('reflection') : null;
      const reflectionSidecarResult = results.has('reflectionSidecar') ? results.get('reflectionSidecar') : null;
      const runtimeWasmResult = results.has('runtimeWasm') ? results.get('runtimeWasm') : null;
      const opfsWarmBootResult = results.has('opfsWarmBoot') ? results.get('opfsWarmBoot') : null;

      let bytesLoaded =
        inferBytesLoaded(pgimResult) +
        inferBytesLoaded(shaderPackResult) +
        inferBytesLoaded(shaderPackIndexResult) +
        inferBytesLoaded(reflectionResult) +
        inferBytesLoaded(reflectionSidecarResult) +
        inferBytesLoaded(runtimeWasmResult) +
        inferBytesLoaded(opfsWarmBootResult);
      if (customSidecars) {
        for (const key of Object.keys(customSidecars)) {
          bytesLoaded += inferBytesLoaded(customSidecars[key]);
        }
      }

      const shaderPackSkippedByLoader = Boolean(
        shaderPackResult && typeof shaderPackResult === 'object' && shaderPackResult.skipped === true
      );
      const bundle = {
        pgim: pgimResult,
        shaderPackBytes: shaderPackSkippedByLoader ? null : shaderPackResult,
        reflectionManifest: reflectionResult,
        bytesLoaded,
      };
      if (skipShaderPackFullPreload || shaderPackSkippedByLoader) {
        bundle.shaderPackSkipped = true;
        bundle.shaderPackSkipReason =
          shaderPackSkipReason ||
          (shaderPackSkippedByLoader && shaderPackResult.skipReason) ||
          'exceeds-preload-threshold';
        bundle.shaderPackStreamedBytes =
          shaderPackProbedBytes !== null
            ? shaderPackProbedBytes
            : shaderPackSkippedByLoader && typeof shaderPackResult.totalBytes === 'number'
              ? shaderPackResult.totalBytes
              : null;
      }
      if (results.has('shaderPackIndex')) {
        bundle.shaderPackIndex = shaderPackIndexResult;
      }
      if (results.has('reflectionSidecar')) {
        bundle.reflectionSidecar = reflectionSidecarResult;
      }
      if (results.has('runtimeWasm')) {
        bundle.runtimeWasm = runtimeWasmResult;
      }
      if (results.has('opfsWarmBoot')) {
        bundle.opfsWarmBoot = opfsWarmBootResult;
      }
      if (customSidecars) {
        bundle.sidecars = customSidecars;
      }
      cleanupExecutionContainers();
      return bundle;
    }

    _fallbackReasonFor(assets) {
      if (!assets.runtimeWasmAvailable) {
        return 'No Wasm64 runtime binary available (runtimeWasm/hasWasmBinary); falling back to WebGPU G5 smoke demo.';
      }
      if (!assets.pgimUrl && !assets.programManifestUrl) {
        return 'No PGIM guest image URL (pgimUrl) or programManifestUrl discovered; falling back to WebGPU G5 smoke demo.';
      }
      return 'Runtime launch requirements were not met; falling back to WebGPU G5 smoke demo.';
    }

    /**
     * Run discovery → optional preload → runtime launch or smoke-demo fallback.
     *
     * @param {{ signal?: AbortSignal }} [options]
     */
    async start(options = {}) {
      if (this._destroyed) {
        return {
          ok: false,
          destroyed: true,
          phase: BOOT_PHASE.DESTROYED,
          bootMode: this._bootMode,
          error: new Error('BootOrchestrator has been destroyed'),
        };
      }
      if (this._startPromise) {
        return this._startPromise;
      }
      if (
        this._phase === BOOT_PHASE.LAUNCHING_RUNTIME ||
        this._phase === BOOT_PHASE.FALLBACK_SMOKE_DEMO ||
        this._phase === BOOT_PHASE.PRESENTING
      ) {
        const timing = this._buildTimingSnapshot();
        return {
          ok: true,
          alreadyStarted: true,
          phase: this._phase,
          bootMode: this._bootMode,
          programManifestUrl: this._assets ? this._assets.programManifestUrl : null,
          gameBase: this._assets ? this._assets.gameBase : null,
          status: this._status,
          assets: this._assets,
          bundle: this._bundle,
          fallbackReason: this._fallbackReason,
          phaseTimingsMs: timing.phaseTimingsMs,
          parallelPreloadSavedMs: timing.parallelPreloadSavedMs,
          serialEstimateMs: timing.serialEstimateMs,
          parallelWallMs: timing.parallelWallMs,
          overlapSavedMs: timing.overlapSavedMs,
          parallelStagesCount: timing.parallelStagesCount,
          totalBootMs: timing.totalBootMs,
        };
      }

      const outerSignal = options && options.signal ? options.signal : null;
      if (outerSignal && typeof outerSignal.aborted !== 'boolean') {
        throw new TypeError('start({ signal }) requires an AbortSignal');
      }
      const startStreamShaders =
        options && Object.prototype.hasOwnProperty.call(options, 'streamShaders')
          ? normalizeOptionalStreamShaders(options.streamShaders)
          : undefined;
      const startStreamThresholdBytes =
        options && Object.prototype.hasOwnProperty.call(options, 'streamThresholdBytes')
          ? normalizeOptionalStreamThresholdBytes(options.streamThresholdBytes)
          : undefined;
      this._activeStartStreamConfig =
        startStreamShaders !== undefined || startStreamThresholdBytes !== undefined
          ? { streamShaders: startStreamShaders, streamThresholdBytes: startStreamThresholdBytes }
          : null;

      const token = ++this._sessionToken;
      const linked = linkSignal(outerSignal);
      this._controller = linked.controller;
      this._abortUnlink = linked.unlink;
      const signal = this._controller.signal;

      const promise = this._startInternal(signal, token).finally(() => {
        if (this._startPromise === promise) {
          this._startPromise = null;
        }
        if (typeof this._abortUnlink === 'function') {
          this._abortUnlink();
          this._abortUnlink = null;
        }
      });
      this._startPromise = promise;
      return promise;
    }

    async _startInternal(signal, token) {
      this._bootMode = BOOT_MODE.PENDING;
      this._error = null;
      this._fallbackReason = null;
      this._bundle = null;
      this._progress = null;
      this._aborted = false;
      this._pgimLoadMarked = false;
      this._resetTimingTelemetry();

      try {
        throwIfAborted(signal);
        this._setPhase(BOOT_PHASE.DISCOVERING);
        const status = await this._fetchStatus(signal);
        if (token !== this._sessionToken) {
          return { ok: false, aborted: true, phase: this._phase, bootMode: this._bootMode };
        }
        this._assets = discoverBootAssets(status);
        this._status = {
          ...status,
          programManifestUrl: this._assets.programManifestUrl,
          gameBase: this._assets.gameBase,
          bootMode: this._assets.bootMode,
        };
        this._initOverlay(this._assets.splashImage);
        this._pushOverlay({
          phase: this._phase,
          bootMode: this._bootMode,
          programManifestUrl: this._assets.programManifestUrl,
          gameBase: this._assets.gameBase,
          status: this._status,
          assets: this._assets,
          tracer: tracerSnapshot(this._bootTracer),
          splashImage: this._assets.splashImage,
        });

        throwIfAborted(signal);
        const hasRuntimeWasmPreload = Boolean(
          this._resolveRuntimeWasmLoader() && this._assets.runtimeWasmAvailable && this._assets.runtimeWasmUrl
        );
        const hasOpfsProbe = Boolean(this._resolveOpfsProbe());
        const hasProgramManifestPreload = Boolean(
          this._assets.programManifestUrl &&
            this._preloader &&
            typeof this._preloader.preloadProgramModules === 'function'
        );
        const shouldPreload = Boolean(
          this._assets.pgimUrl ||
          hasProgramManifestPreload ||
          this._assets.shaderPackUrl ||
          this._assets.shaderPackIndexUrl ||
          this._assets.reflectionSidecarUrl ||
          hasRuntimeWasmPreload ||
          hasOpfsProbe ||
          this._preloadTasks.length > 0
        );
        if (shouldPreload) {
          this._setPhase(BOOT_PHASE.PRELOADING);
          this._bundle = (await this._preloadBundle(this._assets, signal)) || null;
        }

        throwIfAborted(signal);
        if (token !== this._sessionToken) {
          return { ok: false, aborted: true, phase: this._phase, bootMode: this._bootMode };
        }

        const canLaunch =
          this._assets.runtimeWasmAvailable &&
          Boolean(this._assets.pgimUrl || this._assets.programManifestUrl);

        if (canLaunch) {
          this._bootMode = BOOT_MODE.WASM_GUEST;
          this._status = { ...this._status, bootMode: this._bootMode };
          const launchPayload = {
            bootMode: this._bootMode,
            programManifestUrl: this._assets.programManifestUrl,
            gameBase: this._assets.gameBase,
            status: this._status,
            assets: this._assets,
            bundle: this._bundle,
            bootTracer: this._bootTracer,
          };
          this._setPhase(BOOT_PHASE.LAUNCHING_RUNTIME);
          await invokeQuietAsync(this._onLaunchRuntime, launchPayload);
          if (this._releasePreloadBuffers && this._onLaunchRuntime) {
            this.releasePreloadBuffers();
          }
          const timing = this._buildTimingSnapshot();
          return {
            ok: true,
            phase: this._phase,
            ...launchPayload,
            phaseTimingsMs: timing.phaseTimingsMs,
            parallelPreloadSavedMs: timing.parallelPreloadSavedMs,
            serialEstimateMs: timing.serialEstimateMs,
            parallelWallMs: timing.parallelWallMs,
            overlapSavedMs: timing.overlapSavedMs,
            parallelStagesCount: timing.parallelStagesCount,
            totalBootMs: timing.totalBootMs,
          };
        }

        this._bootMode = BOOT_MODE.SMOKE_DEMO;
        this._status = { ...this._status, bootMode: this._bootMode };
        this._fallbackReason = this._fallbackReasonFor(this._assets);
        this._setPhase(BOOT_PHASE.FALLBACK_SMOKE_DEMO, { reason: this._fallbackReason });
        const fallbackPayload = {
          bootMode: this._bootMode,
          programManifestUrl: this._assets.programManifestUrl,
          gameBase: this._assets.gameBase,
          status: this._status,
          bundle: this._bundle,
          reason: this._fallbackReason,
          assets: this._assets,
          bootTracer: this._bootTracer,
        };
        await invokeQuietAsync(this._onFallbackSmokeDemo, fallbackPayload);
        if (this._releasePreloadBuffers && this._onFallbackSmokeDemo) {
          this.releasePreloadBuffers();
        }
        const timing = this._buildTimingSnapshot();
        return {
          ok: true,
          fallback: true,
          phase: this._phase,
          bootMode: this._bootMode,
          programManifestUrl: this._assets.programManifestUrl,
          gameBase: this._assets.gameBase,
          status: this._status,
          bundle: fallbackPayload.bundle,
          assets: this._assets,
          reason: this._fallbackReason,
          fallbackReason: this._fallbackReason,
          phaseTimingsMs: timing.phaseTimingsMs,
          parallelPreloadSavedMs: timing.parallelPreloadSavedMs,
          serialEstimateMs: timing.serialEstimateMs,
          parallelWallMs: timing.parallelWallMs,
          overlapSavedMs: timing.overlapSavedMs,
          parallelStagesCount: timing.parallelStagesCount,
          totalBootMs: timing.totalBootMs,
        };
      } catch (err) {
        if (this._destroyed) {
          this.releasePreloadBuffers();
          return { ok: false, destroyed: true, phase: BOOT_PHASE.DESTROYED, bootMode: this._bootMode };
        }
        if (token !== this._sessionToken) {
          return {
            ok: false,
            aborted: true,
            phase: this._phase,
            bootMode: this._bootMode,
            error: toError(err, 'Boot pipeline aborted'),
          };
        }
        this.releasePreloadBuffers();
        if (signal.aborted || isAbortError(err)) {
          this._aborted = true;
          this._setPhase(BOOT_PHASE.CANCELLED);
          const timing = this._buildTimingSnapshot();
          return {
            ok: false,
            aborted: true,
            phase: this._phase,
            bootMode: this._bootMode,
            error: toError(err, 'Boot pipeline aborted'),
            phaseTimingsMs: timing.phaseTimingsMs,
            parallelPreloadSavedMs: timing.parallelPreloadSavedMs,
            serialEstimateMs: timing.serialEstimateMs,
            parallelWallMs: timing.parallelWallMs,
            overlapSavedMs: timing.overlapSavedMs,
            parallelStagesCount: timing.parallelStagesCount,
            totalBootMs: timing.totalBootMs,
          };
        }
        const normalized = toError(err, 'Boot pipeline failed');
        this._error = normalized;
        this._setPhase(BOOT_PHASE.ERROR, { error: normalized });
        const timing = this._buildTimingSnapshot();
        return {
          ok: false,
          phase: this._phase,
          bootMode: this._bootMode,
          error: normalized,
          phaseTimingsMs: timing.phaseTimingsMs,
          parallelPreloadSavedMs: timing.parallelPreloadSavedMs,
          serialEstimateMs: timing.serialEstimateMs,
          parallelWallMs: timing.parallelWallMs,
          overlapSavedMs: timing.overlapSavedMs,
          parallelStagesCount: timing.parallelStagesCount,
          totalBootMs: timing.totalBootMs,
        };
      }
    }

    /**
     * Forward a runtime log line into BootTracer, refresh the splash overlay, and
     * auto-dismiss on FIRST_PRESENT (`readyForSplash === true`).
     *
     * @param {unknown} line
     */
    feedRuntimeLog(line) {
      if (this._destroyed) return null;
      let ingested = null;
      if (this._bootTracer && typeof this._bootTracer.ingestLogLine === 'function') {
        try {
          ingested = this._bootTracer.ingestLogLine(line);
        } catch {
          ingested = null;
        }
      }
      const tracer = tracerSnapshot(this._bootTracer);
      this._pushOverlay({
        phase: this._phase,
        bootMode: this._bootMode,
        firstPresentSource: this._firstPresentSource,
        tracer,
        status: this._status,
        logLine: typeof line === 'string' ? line : null,
        ingested,
      });
      if (tracer && tracer.readyForSplash === true) {
        this.notifyFirstPresent({ source: 'bootTracer', tracer, ingested });
      }
      return ingested;
    }

    /**
     * Transition to PRESENTING, dismiss splash, record `firstPresentSource`
     * (`'wasm-guest'` vs `'smoke-demo'`), and fire `onFirstPresent` once.
     *
     * @param {object} [meta]
     */
    notifyFirstPresent(meta = {}) {
      if (this._destroyed || this._firstPresent) {
        return {
          ok: true,
          alreadyPresented: true,
          phase: this._phase,
          bootMode: this._bootMode,
          firstPresentSource: this._firstPresentSource,
        };
      }
      const safeMeta = meta && typeof meta === 'object' ? meta : {};
      const firstPresentSource = resolveFirstPresentSource(safeMeta, this._bootMode);
      const enrichedMeta = {
        ...safeMeta,
        bootMode: this._bootMode,
        firstPresentSource,
        isRealGuestBoot: this.isRealGuestBoot(),
      };

      this._firstPresent = true;
      this._firstPresentSource = firstPresentSource;
      if (this._releasePreloadBuffers && this._bundle) {
        this.releasePreloadBuffers();
      }
      if (this._bootTracer && typeof this._bootTracer.markMilestone === 'function') {
        const tracer = tracerSnapshot(this._bootTracer);
        if (!tracer || tracer.readyForSplash !== true) {
          try {
            this._bootTracer.markMilestone('FIRST_PRESENT', enrichedMeta);
          } catch {
            // Ignore unknown-milestone errors from stub tracers.
          }
        }
      }
      this._setPhase(BOOT_PHASE.PRESENTING, {
        meta: enrichedMeta,
        firstPresentSource: this._firstPresentSource,
      });
      if (this._splashOverlay && typeof this._splashOverlay.dismiss === 'function') {
        callQuiet(this._splashOverlay, 'dismiss', enrichedMeta);
      } else {
        callQuiet(this._splashOverlay, 'hide', enrichedMeta);
      }
      invokeQuiet(this._onFirstPresent, {
        ...enrichedMeta,
        status: this._status,
        assets: this._assets,
        bundle: this._bundle,
        tracer: tracerSnapshot(this._bootTracer),
      });
      return {
        ok: true,
        alreadyPresented: false,
        phase: this._phase,
        bootMode: this._bootMode,
        firstPresentSource: this._firstPresentSource,
      };
    }

    /**
     * Alias for `notifyFirstPresent(meta)` to mark FIRST_PRESENT cleanly for guest or smoke frames.
     *
     * @param {object} [meta]
     */
    markFirstPresent(meta = {}) {
      return this.notifyFirstPresent(meta);
    }

    /**
     * Reset boot state, stage timing breakdown, and stalled-stage watchdog back to IDLE
     * so the orchestrator can be started cleanly again. No-op once destroyed.
     */
    reset() {
      if (this._destroyed) {
        return {
          ok: false,
          destroyed: true,
          phase: BOOT_PHASE.DESTROYED,
          bootMode: this._bootMode,
        };
      }
      this._sessionToken += 1;
      this._startPromise = null;
      try {
        if (this._controller && typeof this._controller.abort === 'function' && !this._controller.signal.aborted) {
          this._controller.abort(abortReason(this._controller.signal));
        }
      } catch {
        // Ignore abort errors during reset.
      }
      if (typeof this._abortUnlink === 'function') {
        try {
          this._abortUnlink();
        } catch {
          // Ignore.
        }
        this._abortUnlink = null;
      }
      this._controller = new AbortController();
      this.releasePreloadBuffers();
      this._phase = BOOT_PHASE.IDLE;
      this._bootMode = BOOT_MODE.PENDING;
      this._status = null;
      this._assets = null;
      this._progress = null;
      this._fallbackReason = null;
      this._error = null;
      this._firstPresent = false;
      this._firstPresentSource = null;
      this._aborted = false;
      this._pgimLoadMarked = false;
      this._resetTimingTelemetry();
      return {
        ok: true,
        phase: this._phase,
        bootMode: this._bootMode,
      };
    }

    /**
     * Abort in-flight work, release retained preload buffers, destroy the overlay, and freeze
     * the orchestrator. Safe to call more than once.
     */
    destroy() {
      if (this._destroyed) {
        return { ok: true, alreadyDestroyed: true, phase: this._phase };
      }
      this._destroyed = true;
      this._sessionToken += 1;
      this._startPromise = null;
      this.releasePreloadBuffers();
      try {
        if (this._controller && typeof this._controller.abort === 'function' && !this._controller.signal.aborted) {
          this._controller.abort(abortReason(this._controller.signal));
        }
      } catch {
        // Ignore abort errors during teardown.
      }
      if (typeof this._abortUnlink === 'function') {
        try {
          this._abortUnlink();
        } catch {
          // Ignore.
        }
        this._abortUnlink = null;
      }
      callQuiet(this._splashOverlay, 'destroy');
      this._setPhase(BOOT_PHASE.DESTROYED);
      return { ok: true, alreadyDestroyed: false, phase: this._phase };
    }

    /**
     * Alias for `destroy()` that releases retained preload buffers and tears down the orchestrator.
     */
    dispose() {
      return this.destroy();
    }
  }

  const exportedApi = Object.freeze({
    BOOT_PHASE,
    BOOT_MODE,
    DEFAULT_STATUS_URL,
    DEFAULT_STAGE_TIMEOUT_MS,
    DEFAULT_PARALLEL_PRELOAD,
    DEFAULT_PRELOAD_CONCURRENCY,
    DEFAULT_RELEASE_PRELOAD_BUFFERS,
    DEFAULT_STREAM_SHADERS_THRESHOLD_BYTES,
    DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES,
    BootOrchestrator,
    discoverBootAssets,
    isRealGuestBoot,
    collectTransferableBuffers,
    transferPreloadPayloadToWorker,
    releaseBootBundleBytes,
    runConcurrentPreloads,
    default: BootOrchestrator,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6BootOrchestrator = exportedApi;
  } else if (root) {
    root.PlayGta6BootOrchestrator = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
