'use strict';

// Main runtime coordinator worker (keeps the browser DOM thread completely free).
(function initPlayGta6Loader() {
const EMSCRIPTEN_PTHREAD_WORKER_NAME = 'em-pthread';
const PTHREAD_WGPU_CHANNEL_NAME = 'playgta6-wgpu-pthread';
const PTHREAD_HTTPFS_SAB_CHANNEL_NAME = 'playgta6-httpfs-sab';
const HOST_ABI_IMPORT_NAMES = Object.freeze([
  'host_log',
  'host_httpfs_open',
  'host_httpfs_read',
  'host_httpfs_close',
  'host_wgpu_submit',
]);
const HOST_ABI_IMPORT_SET = new Set(HOST_ABI_IMPORT_NAMES);
const GUEST_SIDE_MODULE_BASENAME_RE = /^guest_part_\d+\.wasm$/i;
const DEFAULT_RUNTIME_SCRIPT_RELATIVE = './dist/runtime.js';
const DEFAULT_RUNTIME_SCRIPT_FALLBACK = 'dist/runtime.js';
const ABSOLUTE_URL_RE = /^[a-zA-Z][a-zA-Z0-9+.-]*:/;

// GpuPacketHeader / GpuOpcode from runtime/abi/playgta6_abi.h and docs/gpu_commands.md.
// Present is opcode 7 (ClearColor is 2). Detecting opcode 2 as Present would dismiss splash on clear.
const GPU_PACKET_HEADER_BYTES = 8;
const GPU_OPCODE_PRESENT = 7;
const utf8Decoder = new TextDecoder('utf-8', { fatal: false });

// Emscripten 7.0.1 `-pthread` spawns `new Worker(pthreadMainJs, { name: "em-pthread-" + id })`
// (or legacy `{ name: "em-pthread" }`), which our coordinator Worker Proxy redirects to `loader.js`.
// Detect pthread workers immediately so we bridge `env.host_*` imports and delegate straight
// to `./dist/runtime.js` without overwriting Emscripten's `self.onmessage = handleMessage`.
const initialWorkerScope = typeof self !== 'undefined' ? self : globalThis;
if (isEmscriptenPthreadWorker(initialWorkerScope) && typeof importScripts === 'function') {
  try {
    importScripts('./workers/httpfs_mailbox.js');
  } catch {
    // Degrade safely if httpfs_mailbox.js cannot be loaded inside a pthread worker.
  }
  const pthreadHttpfsMailboxApi =
    (initialWorkerScope && initialWorkerScope.PlayGta6HttpfsMailbox) ||
    (typeof globalThis !== 'undefined' && globalThis.PlayGta6HttpfsMailbox) ||
    null;
  bootstrapEmscriptenPthreadWorker({
    scope: initialWorkerScope,
    importScriptsFn: importScripts,
    runtimeScriptUrl: DEFAULT_RUNTIME_SCRIPT_RELATIVE,
    httpfsMailboxApi: pthreadHttpfsMailboxApi,
  });
  return;
}

if (typeof importScripts === 'function') {
  importScripts('./workers/userdata.js');
  importScripts('./workers/httpfs_mailbox.js');
}

const progress =
  typeof self !== 'undefined' && typeof BroadcastChannel === 'function'
    ? new BroadcastChannel('game-progress')
    : null;

const userdataApi =
  (typeof globalThis !== 'undefined' && globalThis.PlayGta6Userdata) ||
  (typeof require === 'function' ? require('./workers/userdata.js') : null);
const httpfsMailboxApi =
  (typeof globalThis !== 'undefined' && globalThis.PlayGta6HttpfsMailbox) ||
  (typeof require === 'function' ? require('./workers/httpfs_mailbox.js') : null);

const userdataStore = userdataApi ? new userdataApi.UserdataStore() : null;
const BOOT_INIT_FORWARD_KEYS = Object.freeze([
  'shaderPackUrl',
  'reflectionManifestUrl',
  'pgimUrl',
  'bootBase',
  'gameBase',
  'programManifestUrl',
]);
const GAME_PACKAGE_HTTPFS_EXT_RE = /\.(?:rpf|ttpf|dll|png|xvd|xvc|bin|pri)$/i;
const WIN32_DRIVE_PATH_RE = /^(?:\\\\[?.]\\|\/\/[?.]\/)?([A-Za-z]):(.*)$/;
const GPU_TELEMETRY_MESSAGE_TYPES = new Set([
  'gpu-stats',
  'gpu-telemetry',
  'missing-shaders',
  'frame-presented',
]);
const GPU_SHADER_STAGE_NAMES = Object.freeze({
  1: 'vertex',
  2: 'fragment',
  3: 'compute',
});
const GPU_SHADER_STAGE_ALIASES = Object.freeze({
  vertex: 'vertex',
  vs: 'vertex',
  vert: 'vertex',
  1: 'vertex',
  fragment: 'fragment',
  fs: 'fragment',
  frag: 'fragment',
  pixel: 'fragment',
  ps: 'fragment',
  2: 'fragment',
  compute: 'compute',
  cs: 'compute',
  comp: 'compute',
  3: 'compute',
});
const U64_MAX = 0xffffffffffffffffn;

let gpuWorker = null;
let ioWorker = null;
let audioWorker = null;
let inputSab = null;
let httpfsMailboxSab = null;
let httpfsClient = null;

function report(label) {
  if (progress) {
    progress.postMessage({ label });
  }
}

function ptrToNumber(value, label = 'pointer') {
  if (typeof value === 'bigint') {
    if (value < 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new RangeError(`${label} out of addressable range: ${value}`);
    }
    return Number(value);
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value) || value < 0 || !Number.isInteger(value)) {
      throw new RangeError(`${label} must be a non-negative integer: ${value}`);
    }
    return value;
  }
  throw new TypeError(`${label} must be a BigInt or Number`);
}

function resolveBuffer(memoryOrBuffer) {
  if (!memoryOrBuffer) return null;
  if (memoryOrBuffer.buffer) return memoryOrBuffer.buffer;
  if (memoryOrBuffer instanceof ArrayBuffer) return memoryOrBuffer;
  if (typeof SharedArrayBuffer !== 'undefined' && memoryOrBuffer instanceof SharedArrayBuffer) {
    return memoryOrBuffer;
  }
  return null;
}

function copyMemoryBytes(buf, ptr, len) {
  const byteOffset = ptrToNumber(ptr, 'ptr');
  const byteLength = ptrToNumber(len, 'len');
  if (byteOffset + byteLength > buf.byteLength) {
    throw new RangeError(
      `Wasm memory read out of bounds: [${byteOffset}..${byteOffset + byteLength}) > ${buf.byteLength}`,
    );
  }
  const copy = new Uint8Array(byteLength);
  if (byteLength > 0) {
    copy.set(new Uint8Array(buf, byteOffset, byteLength));
  }
  return copy;
}

function asDataView(bytes) {
  if (
    bytes instanceof ArrayBuffer ||
    (typeof SharedArrayBuffer !== 'undefined' && bytes instanceof SharedArrayBuffer)
  ) {
    return new DataView(bytes);
  }
  if (ArrayBuffer.isView(bytes)) {
    return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }
  return null;
}

/**
 * Walk a little-endian GpuPacketHeader stream (u16 opcode, u16 flags, u32 payloadSize)
 * and report whether it contains a Present packet. Never throws on truncated or malformed input.
 */
function inspectGpuPacketBatchForPresent(bytes) {
  const empty = {
    packetCount: 0,
    hasPresent: false,
    firstPresentOffset: -1,
    truncated: false,
  };
  let view;
  try {
    view = asDataView(bytes);
  } catch {
    return empty;
  }
  if (!view) {
    return empty;
  }

  const result = {
    packetCount: 0,
    hasPresent: false,
    firstPresentOffset: -1,
    truncated: false,
  };
  const total = view.byteLength;
  let offset = 0;

  while (offset < total) {
    if (offset + GPU_PACKET_HEADER_BYTES > total) {
      result.truncated = true;
      break;
    }
    let opcode;
    let payloadBytes;
    try {
      opcode = view.getUint16(offset, true);
      payloadBytes = view.getUint32(offset + 4, true);
    } catch {
      result.truncated = true;
      break;
    }
    const payloadOffset = offset + GPU_PACKET_HEADER_BYTES;
    const remaining = total - payloadOffset;
    if (payloadBytes > remaining) {
      result.truncated = true;
      break;
    }
    if (opcode === GPU_OPCODE_PRESENT && !result.hasPresent) {
      result.hasPresent = true;
      result.firstPresentOffset = offset;
    }
    result.packetCount += 1;
    offset = payloadOffset + payloadBytes;
  }

  return result;
}

function ensureTrailingSlash(base) {
  if (typeof base !== 'string' || base.length === 0) {
    return '';
  }
  return base.endsWith('/') ? base : `${base}/`;
}

function sanitizeHttpfsRelPath(relPath) {
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

function basenameOfPath(posixPath) {
  const slash = posixPath.lastIndexOf('/');
  return slash < 0 ? posixPath : posixPath.slice(slash + 1);
}

function isDataPinnedHttpfsPath(cleanPath) {
  const base = basenameOfPath(cleanPath).toLowerCase();
  if (base === 'data-manifest.json' || base === 'bootset.json') {
    return true;
  }
  return /\.rpf$/i.test(cleanPath);
}

function isManifestPinnedHttpfsPath(cleanPath) {
  const base = basenameOfPath(cleanPath).toLowerCase();
  return base === 'data-manifest.json' || base === 'bootset.json';
}

function isExplicitBootHttpfsPath(cleanPath) {
  const lower = cleanPath.toLowerCase();
  return lower === 'boot' || lower.startsWith('boot/');
}

function isBootRoutedHttpfsPath(cleanPath) {
  const lower = cleanPath.toLowerCase();
  if (isExplicitBootHttpfsPath(cleanPath)) {
    return true;
  }
  if (basenameOfPath(lower) === 'program-manifest.json') {
    return true;
  }
  return /\.pgim$/i.test(lower);
}

function isGamePrefixedHttpfsPath(cleanPath) {
  const lower = cleanPath.toLowerCase();
  return lower === 'game' || lower.startsWith('game/');
}

function isGamePackageHttpfsPath(cleanPath) {
  const base = basenameOfPath(cleanPath).toLowerCase();
  if (base === 'microsoftgame.config') {
    return true;
  }
  return GAME_PACKAGE_HTTPFS_EXT_RE.test(cleanPath);
}

function stripBootPrefix(cleanPath) {
  const lower = cleanPath.toLowerCase();
  if (lower === 'boot') {
    return '';
  }
  if (lower.startsWith('boot/')) {
    return cleanPath.slice('boot/'.length);
  }
  return cleanPath;
}

function stripGamePrefix(cleanPath) {
  const lower = cleanPath.toLowerCase();
  if (lower === 'game') {
    return '';
  }
  if (lower.startsWith('game/')) {
    return cleanPath.slice('game/'.length);
  }
  return cleanPath;
}

function joinHttpfsBase(base, rel) {
  const prefix = ensureTrailingSlash(base) || '/data/';
  const tail = String(rel || '').replace(/^\/+/, '');
  return tail ? `${prefix}${tail}` : prefix;
}

/**
 * Route guest HTTPFS paths: `boot/...`, `program-manifest.json`, and `*.pgim` go to
 * `bootBase` when configured; Xbox game package/drive paths (`G:\...`, `game/...`,
 * `*.rpf`, `*.dll`, `MicrosoftGame.Config`, etc.) go to `gameBase` when configured;
 * `data-manifest.json`, `bootset.json`, and fallback paths resolve on `dataBase`.
 */
function normalizeBootHttpfsPath(relPath, options = {}) {
  if (typeof relPath !== 'string' || relPath.length === 0) {
    throw new Error('HTTPFS path must be a non-empty string');
  }
  if (relPath.includes('\0')) {
    throw new Error(`Invalid HTTPFS path: ${relPath}`);
  }

  const dataBase = options && options.dataBase !== undefined ? options.dataBase : '/data/';
  const bootBase =
    options && typeof options.bootBase === 'string' ? options.bootBase.trim() : '';
  const gameBase =
    options && typeof options.gameBase === 'string' ? options.gameBase.trim() : '';

  let hadDrivePrefix = false;
  let pathToSanitize = relPath;
  if (gameBase) {
    const driveMatch = WIN32_DRIVE_PATH_RE.exec(relPath);
    if (driveMatch) {
      hadDrivePrefix = true;
      pathToSanitize = driveMatch[2].replace(/\\/g, '/');
    }
  }

  const clean = sanitizeHttpfsRelPath(pathToSanitize);

  if (isManifestPinnedHttpfsPath(clean)) {
    return joinHttpfsBase(dataBase, clean);
  }
  if (bootBase && isExplicitBootHttpfsPath(clean) && !isDataPinnedHttpfsPath(clean)) {
    return joinHttpfsBase(bootBase, stripBootPrefix(clean));
  }
  if (gameBase && !hadDrivePrefix && isGamePrefixedHttpfsPath(clean)) {
    return joinHttpfsBase(gameBase, stripGamePrefix(clean));
  }
  if (bootBase && isBootRoutedHttpfsPath(clean) && !isDataPinnedHttpfsPath(clean)) {
    return joinHttpfsBase(bootBase, stripBootPrefix(clean));
  }
  if (gameBase && (hadDrivePrefix || isGamePackageHttpfsPath(clean))) {
    return joinHttpfsBase(gameBase, clean);
  }
  return joinHttpfsBase(dataBase, clean);
}

function toNonNegativeInt(value, fallback = 0) {
  if (value === undefined || value === null || value === '') {
    return fallback;
  }
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n) || n < 0) {
    return fallback;
  }
  return Math.floor(n);
}

function toNonNegativeFinite(value) {
  if (value === undefined || value === null || value === '') {
    return null;
  }
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n) || n < 0) {
    return null;
  }
  return n;
}

function clampPct(value) {
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n) || n <= 0) return 0;
  if (n >= 100) return 100;
  return n;
}

function normalizeShaderHashHex(raw) {
  if (typeof raw === 'bigint') {
    if (raw < 0n || raw > U64_MAX) return null;
    return `0x${raw.toString(16).padStart(16, '0')}`;
  }
  if (typeof raw === 'number') {
    if (!Number.isSafeInteger(raw) || raw < 0) return null;
    return `0x${BigInt(raw).toString(16).padStart(16, '0')}`;
  }
  if (typeof raw === 'string') {
    const trimmed = raw.trim();
    const hex =
      trimmed.startsWith('0x') || trimmed.startsWith('0X') ? trimmed.slice(2) : trimmed;
    if (!/^[0-9a-fA-F]{1,16}$/.test(hex)) {
      return null;
    }
    return `0x${BigInt(`0x${hex}`).toString(16).padStart(16, '0')}`;
  }
  return null;
}

function normalizeShaderStageName(stage) {
  if (typeof stage === 'number') {
    return GPU_SHADER_STAGE_NAMES[stage] || 'unknown';
  }
  if (typeof stage === 'string') {
    const key = stage.trim().toLowerCase();
    if (Object.prototype.hasOwnProperty.call(GPU_SHADER_STAGE_ALIASES, key)) {
      return GPU_SHADER_STAGE_ALIASES[key];
    }
  }
  return 'unknown';
}

function sanitizeMissingShadersList(rawList) {
  if (!rawList || typeof rawList === 'string') {
    return [];
  }
  let items;
  if (Array.isArray(rawList)) {
    items = rawList;
  } else if (typeof rawList[Symbol.iterator] === 'function') {
    try {
      items = Array.from(rawList);
    } catch {
      return [];
    }
  } else {
    return [];
  }

  const out = [];
  for (let i = 0; i < items.length; i += 1) {
    const entry = items[i];
    if (!entry || typeof entry !== 'object') {
      continue;
    }
    const rawHash =
      entry.hash !== undefined
        ? entry.hash
        : entry.shaderHash !== undefined
          ? entry.shaderHash
          : entry.hashHex;
    const hash = normalizeShaderHashHex(rawHash);
    if (hash === null) {
      continue;
    }
    const stage = normalizeShaderStageName(entry.stage);
    const rawHits =
      entry.hits !== undefined
        ? entry.hits
        : entry.hitCount !== undefined
          ? entry.hitCount
          : entry.drawCount !== undefined
            ? entry.drawCount
            : entry.count;
    const hits = rawHits === undefined || rawHits === null ? 1 : toNonNegativeInt(rawHits, 0);
    const sanitized = { hash, stage, hits };
    const firstSeenMs = toNonNegativeFinite(entry.firstSeenMs);
    const lastSeenMs = toNonNegativeFinite(entry.lastSeenMs);
    if (firstSeenMs !== null) {
      sanitized.firstSeenMs = firstSeenMs;
    }
    if (lastSeenMs !== null) {
      sanitized.lastSeenMs = lastSeenMs;
    }
    out.push(sanitized);
  }
  return out;
}

function sanitizeFallbackStats(raw) {
  if (!raw || typeof raw !== 'object') {
    return null;
  }
  const packHits = toNonNegativeInt(raw.packHits, 0);
  const fallbackHits = toNonNegativeInt(raw.fallbackHits, 0);
  const builtinHits = toNonNegativeInt(raw.builtinHits, 0);
  const sumHits = packHits + fallbackHits + builtinHits;
  const totalLookups =
    raw.totalLookups !== undefined && raw.totalLookups !== null
      ? toNonNegativeInt(raw.totalLookups, sumHits)
      : sumHits;
  const overflowMissingCount = toNonNegativeInt(raw.overflowMissingCount, 0);
  let hitRatePct;
  if (
    raw.hitRatePct !== undefined &&
    raw.hitRatePct !== null &&
    Number.isFinite(Number(raw.hitRatePct))
  ) {
    hitRatePct = Math.round(clampPct(Number(raw.hitRatePct)) * 100) / 100;
  } else {
    hitRatePct = totalLookups > 0 ? Math.round((packHits / totalLookups) * 10000) / 100 : 0;
  }
  return {
    totalLookups,
    packHits,
    fallbackHits,
    builtinHits,
    hitRatePct,
    overflowMissingCount,
  };
}

function sanitizeFrameStats(eventData) {
  const rawFrameStats =
    eventData.frameStats && typeof eventData.frameStats === 'object' ? eventData.frameStats : null;
  const rawStats =
    eventData.stats && typeof eventData.stats === 'object' ? eventData.stats : null;

  const hasFrameField =
    Boolean(rawFrameStats) ||
    Boolean(rawStats) ||
    typeof eventData.framesPresented === 'number' ||
    typeof eventData.droppedFrames === 'number' ||
    typeof eventData.frame === 'number' ||
    typeof eventData.frameCount === 'number' ||
    eventData.type === 'frame-presented' ||
    eventData.type === 'gpu-stats';

  if (!hasFrameField) {
    return null;
  }

  const rawPresented =
    rawFrameStats && rawFrameStats.framesPresented !== undefined
      ? rawFrameStats.framesPresented
      : rawFrameStats && rawFrameStats.frameCount !== undefined
        ? rawFrameStats.frameCount
        : eventData.framesPresented !== undefined
          ? eventData.framesPresented
          : rawStats && rawStats.framesPresented !== undefined
            ? rawStats.framesPresented
            : rawStats && rawStats.frameCount !== undefined
              ? rawStats.frameCount
              : eventData.frameCount !== undefined
                ? eventData.frameCount
                : eventData.frame;

  const rawDropped =
    rawFrameStats && rawFrameStats.droppedFrames !== undefined
      ? rawFrameStats.droppedFrames
      : eventData.droppedFrames !== undefined
        ? eventData.droppedFrames
        : rawStats && rawStats.droppedFrames !== undefined
          ? rawStats.droppedFrames
          : 0;

  const frameStats = {
    framesPresented: toNonNegativeInt(rawPresented, 0),
    droppedFrames: toNonNegativeInt(rawDropped, 0),
  };

  const drawCallsSrc =
    rawFrameStats && rawFrameStats.drawCalls !== undefined
      ? rawFrameStats.drawCalls
      : rawStats && rawStats.drawCalls !== undefined
        ? rawStats.drawCalls
        : eventData.drawCalls;
  if (drawCallsSrc !== undefined && drawCallsSrc !== null) {
    frameStats.drawCalls = toNonNegativeInt(drawCallsSrc, 0);
  }

  const computeSrc =
    rawFrameStats && rawFrameStats.computeDispatches !== undefined
      ? rawFrameStats.computeDispatches
      : rawStats && rawStats.computeDispatches !== undefined
        ? rawStats.computeDispatches
        : eventData.computeDispatches;
  if (computeSrc !== undefined && computeSrc !== null) {
    frameStats.computeDispatches = toNonNegativeInt(computeSrc, 0);
  }

  const packetSrc =
    rawFrameStats && rawFrameStats.packetCount !== undefined
      ? rawFrameStats.packetCount
      : rawStats && rawStats.packetCount !== undefined
        ? rawStats.packetCount
        : eventData.packetCount;
  if (packetSrc !== undefined && packetSrc !== null) {
    frameStats.packetCount = toNonNegativeInt(packetSrc, 0);
  }

  const vramSrc =
    rawFrameStats && rawFrameStats.vramBytes !== undefined
      ? rawFrameStats.vramBytes
      : rawStats && rawStats.vramBytes !== undefined
        ? rawStats.vramBytes
        : eventData.vramBytes;
  if (vramSrc !== undefined && vramSrc !== null) {
    frameStats.vramBytes = toNonNegativeInt(vramSrc, 0);
  }

  return frameStats;
}

function isGpuTelemetryMessage(eventData) {
  if (!eventData || typeof eventData !== 'object' || Array.isArray(eventData)) {
    return false;
  }
  if (typeof eventData.type === 'string' && GPU_TELEMETRY_MESSAGE_TYPES.has(eventData.type)) {
    return true;
  }
  if (Array.isArray(eventData.missingShaders)) {
    return true;
  }
  if (
    (eventData.shaderFallback && typeof eventData.shaderFallback === 'object') ||
    (eventData.shaderFallbackStats && typeof eventData.shaderFallbackStats === 'object') ||
    (eventData.fallbackStats && typeof eventData.fallbackStats === 'object') ||
    (eventData.frameStats && typeof eventData.frameStats === 'object') ||
    (eventData.stats && typeof eventData.stats === 'object')
  ) {
    return true;
  }
  if (
    (typeof eventData.framesPresented === 'number' && Number.isFinite(eventData.framesPresented)) ||
    (typeof eventData.frame === 'number' && Number.isFinite(eventData.frame))
  ) {
    return true;
  }
  return false;
}

/**
 * Inspect a message emitted by `wgpu_worker.js`, sanitize/normalize GPU frame,
 * shader-fallback, and missing-shader telemetry, and forward `{ type: 'gpu-telemetry', ... }`
 * to the main thread via `postMessageFn`. Returns the forwarded payload or `null` when ignored.
 */
function forwardGpuWorkerMessage(eventData, postMessageFn) {
  if (!isGpuTelemetryMessage(eventData)) {
    return null;
  }

  const rawStats =
    eventData.stats && typeof eventData.stats === 'object' ? eventData.stats : null;
  const rawFallback =
    eventData.fallbackStats ||
    eventData.shaderFallbackStats ||
    eventData.shaderFallback ||
    (rawStats && (rawStats.shaderFallback || rawStats.fallbackStats || rawStats.shaderFallbackStats)) ||
    null;
  const rawMissing =
    eventData.missingShaders !== undefined
      ? eventData.missingShaders
      : rawStats && rawStats.missingShaders !== undefined
        ? rawStats.missingShaders
        : null;

  const timestamp =
    typeof eventData.timestamp === 'number' && Number.isFinite(eventData.timestamp)
      ? eventData.timestamp
      : Date.now();

  const telemetry = {
    type: 'gpu-telemetry',
    frameStats: sanitizeFrameStats(eventData),
    fallbackStats: sanitizeFallbackStats(rawFallback),
    missingShaders: sanitizeMissingShadersList(rawMissing),
    timestamp,
  };

  if (typeof postMessageFn === 'function') {
    try {
      postMessageFn(telemetry);
    } catch {
      // Telemetry forwarding must never disrupt the GPU worker loop.
    }
  }

  return telemetry;
}

function collectBootBridgeInitFields(src) {
  const extra = {};
  if (!src || typeof src !== 'object') {
    return extra;
  }
  for (let i = 0; i < BOOT_INIT_FORWARD_KEYS.length; i += 1) {
    const key = BOOT_INIT_FORWARD_KEYS[i];
    if (src[key] !== undefined) {
      extra[key] = src[key];
    }
  }
  return extra;
}

function createWorkerBootInitPayloads(msg = {}) {
  const bootFields = collectBootBridgeInitFields(msg);
  return {
    ioInit: Object.assign({ init: true, dataBase: msg.dataBase }, bootFields),
    gpuInit: Object.assign(
      { init: true, canvas: msg.canvas, width: msg.width, height: msg.height },
      bootFields,
    ),
  };
}

// Builds Wasm64/Wasm32 `env` host import functions matching docs/wasm_abi.md and runtime/os/os_shim.cpp.
function createHostImports(options = {}) {
  let currentMemory = options.memory || null;
  const getMemory = typeof options.getMemory === 'function' ? options.getMemory : null;
  const logFn = typeof options.log === 'function' ? options.log : (msg) => report(`[wasm] ${msg}`);
  const wgpuSubmitFn = typeof options.wgpuSubmit === 'function' ? options.wgpuSubmit : null;
  const onFirstPresent =
    typeof options.onFirstPresent === 'function' ? options.onFirstPresent : null;
  const reportFn = typeof options.report === 'function' ? options.report : report;
  const emitFirstPresentLog = options.emitFirstPresentLog === true;
  let currentHttpfsClient = options.httpfsClient || null;
  const getHttpfsClient =
    typeof options.getHttpfsClient === 'function' ? options.getHttpfsClient : null;
  const store = options.userdataStore || null;
  let firstPresentEmitted = false;

  function activeBuffer() {
    const raw = getMemory ? getMemory() || currentMemory : currentMemory;
    return resolveBuffer(raw);
  }

  function activeHttpfsClient() {
    return getHttpfsClient ? getHttpfsClient() || currentHttpfsClient : currentHttpfsClient;
  }

  function host_log(ptr, len) {
    const buf = activeBuffer();
    if (!buf) return;
    const bytes = copyMemoryBytes(buf, ptr, len);
    logFn(utf8Decoder.decode(bytes));
  }

  function emitFirstPresent(bytes, info) {
    if (firstPresentEmitted || !info || !info.hasPresent) {
      return;
    }
    firstPresentEmitted = true;
    const detail = {
      packetCount: info.packetCount,
      byteLength: bytes && typeof bytes.byteLength === 'number' ? bytes.byteLength : 0,
    };
    try {
      if (onFirstPresent) {
        onFirstPresent(detail);
      }
    } catch {
      // First-present hooks must never break the GPU submit path.
    }
    try {
      reportFn('[boot] FIRST_PRESENT');
    } catch {
      // Progress channel failures are non-fatal.
    }
    // Do not call host_log's logFn unless asked: existing Wasm64 tests treat `log` as host_log only.
    if (emitFirstPresentLog) {
      try {
        logFn('[boot] FIRST_PRESENT');
      } catch {
        // Boot log failures are non-fatal.
      }
    }
  }

  function host_wgpu_submit(ptr, len) {
    const buf = activeBuffer();
    if (!buf) return;
    const bytes = copyMemoryBytes(buf, ptr, len);
    if (!firstPresentEmitted) {
      emitFirstPresent(bytes, inspectGpuPacketBatchForPresent(bytes));
    }
    if (wgpuSubmitFn) {
      wgpuSubmitFn(bytes);
    }
  }

  function host_httpfs_open(ptr, len) {
    const fsClient = activeHttpfsClient();
    if (!fsClient || typeof fsClient.open !== 'function') {
      return -1;
    }
    const buf = activeBuffer();
    if (!buf) return -1;
    try {
      const bytes = copyMemoryBytes(buf, ptr, len);
      const relPath = utf8Decoder.decode(bytes);
      const handle = fsClient.open(relPath);
      return Number.isInteger(handle) ? handle | 0 : -1;
    } catch {
      return -1;
    }
  }

  function host_httpfs_read(handle, offset, dstPtr, len) {
    const fsClient = activeHttpfsClient();
    if (!fsClient || typeof fsClient.read !== 'function') {
      return -1n;
    }
    const buf = activeBuffer();
    if (!buf) return -1n;
    try {
      const h = Number(handle) | 0;
      if (h < 0) return -1n;
      const dstOffset = ptrToNumber(dstPtr, 'dstPtr');
      const byteLength = ptrToNumber(len, 'len');
      if (dstOffset + byteLength > buf.byteLength) {
        return -1n;
      }
      if (byteLength === 0) {
        return 0n;
      }
      const dstView = new Uint8Array(buf, dstOffset, byteLength);
      const n = fsClient.read(h, offset, dstView, byteLength);
      if (typeof n === 'bigint') {
        return n;
      }
      return Number.isFinite(n) ? BigInt(Math.trunc(n)) : -1n;
    } catch {
      return -1n;
    }
  }

  function host_httpfs_close(handle) {
    const fsClient = activeHttpfsClient();
    if (!fsClient || typeof fsClient.close !== 'function') {
      return;
    }
    try {
      fsClient.close(Number(handle) | 0);
    } catch {
      // Close errors are non-fatal for void ABI.
    }
  }

  const env = {
    host_log,
    host_wgpu_submit,
    host_httpfs_open,
    host_httpfs_read,
    host_httpfs_close,
  };

  const userdata = {
    restore: (fsAdapter, opts) => {
      if (!store) {
        return Promise.resolve({
          restored: 0,
          skipped: 0,
          timedOut: false,
          error: 'userdata store unavailable',
        });
      }
      return store.restoreAll(fsAdapter, opts);
    },
    save: (path, bytes, mtime) => {
      if (!store) return Promise.resolve(null);
      return store.saveFile(path, bytes, mtime).catch((err) => {
        report(`userdata save warning (${path}): ${err}`);
        return null;
      });
    },
    delete: (path) => {
      if (!store) return Promise.resolve(null);
      return store.deleteFile(path).catch((err) => {
        report(`userdata delete warning (${path}): ${err}`);
        return null;
      });
    },
  };

  return {
    env,
    userdata,
    setMemory(mem) {
      currentMemory = mem || null;
    },
    setHttpfsClient(client) {
      currentHttpfsClient = client || null;
    },
  };
}

function encodeTestFramePacket(r, g, b, a) {
  // Matches runtime/abi/playgta6_abi.h:
  // BeginFrame (8B) + ClearColor (8B + 16B) + Present (8B) = 40 bytes
  const buf = new ArrayBuffer(40);
  const view = new DataView(buf);
  // BeginFrame (opcode 1, payload 0)
  view.setUint16(0, 1, true);
  view.setUint16(2, 0, true);
  view.setUint32(4, 0, true);
  // ClearColor (opcode 2, payload 16)
  view.setUint16(8, 2, true);
  view.setUint16(10, 0, true);
  view.setUint32(12, 16, true);
  view.setFloat32(16, r, true);
  view.setFloat32(20, g, true);
  view.setFloat32(24, b, true);
  view.setFloat32(28, a, true);
  // Present (opcode 7, payload 0)
  view.setUint16(32, 7, true);
  view.setUint16(34, 0, true);
  view.setUint32(36, 0, true);
  return buf;
}

function tryOpenBroadcastChannel(BroadcastChannelCtor, channelName) {
  if (typeof BroadcastChannelCtor !== 'function') {
    return null;
  }
  try {
    return new BroadcastChannelCtor(channelName);
  } catch {
    return null;
  }
}

function isEmscriptenPthreadWorkerName(name) {
  if (typeof name !== 'string') {
    return false;
  }
  return (
    name === EMSCRIPTEN_PTHREAD_WORKER_NAME ||
    name.startsWith(`${EMSCRIPTEN_PTHREAD_WORKER_NAME}-`)
  );
}

function isEmscriptenPthreadWorker(scope) {
  return Boolean(scope && isEmscriptenPthreadWorkerName(scope.name));
}

function inspectRequiredHostAbiImports(wasmModule, wasmNs) {
  const ModuleCtor =
    (wasmNs && typeof wasmNs.Module === 'function' ? wasmNs.Module : null) ||
    (typeof WebAssembly !== 'undefined' && typeof WebAssembly.Module === 'function'
      ? WebAssembly.Module
      : null);
  if (!ModuleCtor || typeof ModuleCtor.imports !== 'function') {
    return null;
  }
  if (!(wasmModule instanceof ModuleCtor)) {
    return null;
  }
  try {
    const entries = ModuleCtor.imports(wasmModule);
    if (!Array.isArray(entries)) {
      return null;
    }
    const needed = new Set();
    for (let i = 0; i < entries.length; i += 1) {
      const imp = entries[i];
      if (imp && imp.module === 'env' && HOST_ABI_IMPORT_SET.has(imp.name)) {
        needed.add(imp.name);
      }
    }
    return needed;
  } catch {
    return null;
  }
}

function createPatchedPthreadEnv(baseEnv, hostEnv, requiredHostNames) {
  if (!hostEnv) {
    return baseEnv;
  }
  const targetNames = requiredHostNames ? Array.from(requiredHostNames) : HOST_ABI_IMPORT_NAMES;
  if (targetNames.length === 0) {
    return baseEnv;
  }

  // If baseEnv is frozen or has a non-configurable non-writable host_* stub,
  // a Proxy get-trap cannot return a replacement without throwing TypeError.
  let needsClone = Object.isFrozen(baseEnv);
  if (!needsClone) {
    for (let i = 0; i < targetNames.length; i += 1) {
      const desc = Reflect.getOwnPropertyDescriptor(baseEnv, targetNames[i]);
      if (desc && !desc.configurable && 'value' in desc && !desc.writable) {
        needsClone = true;
        break;
      }
    }
  }

  if (needsClone) {
    const cloned = Object.assign({}, baseEnv);
    for (let i = 0; i < targetNames.length; i += 1) {
      const name = targetNames[i];
      if (typeof hostEnv[name] !== 'function') {
        continue;
      }
      const existing = cloned[name];
      if (typeof existing !== 'function' || existing.stub === true) {
        cloned[name] = hostEnv[name];
      }
    }
    return cloned;
  }

  // Emscripten MAIN_MODULE=1 generates JS wrappers (`function _host_log(...args) {
  //   if (!wasmImports["host_log"] || wasmImports["host_log"].stub) abort(...);
  //   return wasmImports["host_log"](...args);
  // }`) that read the unwrapped `wasmImports` object directly rather than our Proxy.
  // Overwrite writable own `.stub === true` properties in-place so those wrappers resolve.
  for (let i = 0; i < targetNames.length; i += 1) {
    const name = targetNames[i];
    if (typeof hostEnv[name] !== 'function') {
      continue;
    }
    try {
      const desc = Reflect.getOwnPropertyDescriptor(baseEnv, name);
      if (
        desc &&
        'value' in desc &&
        desc.writable &&
        typeof desc.value === 'function' &&
        desc.value.stub === true
      ) {
        baseEnv[name] = hostEnv[name];
      }
    } catch {
      // Ignore non-writable or Proxy-protected targets; the returned Proxy still intercepts reads.
    }
  }

  // Wrap baseEnv in a transparent Proxy so Emscripten MAIN_MODULE=2 / SIDE_MODULE=1
  // dynamic-linking Proxies, __memory_base, __table_base, __indirect_function_table,
  // and all non-stub lifted/C functions remain completely untouched.
  return new Proxy(baseEnv, {
    get(target, prop, receiver) {
      const isTargetHost =
        typeof prop === 'string' &&
        HOST_ABI_IMPORT_SET.has(prop) &&
        (!requiredHostNames || requiredHostNames.has(prop));
      let existing;
      try {
        existing = Reflect.get(target, prop, receiver);
      } catch (err) {
        if (isTargetHost && typeof hostEnv[prop] === 'function') {
          return hostEnv[prop];
        }
        throw err;
      }
      if (isTargetHost && typeof hostEnv[prop] === 'function') {
        if (typeof existing !== 'function' || existing.stub === true) {
          return hostEnv[prop];
        }
      }
      return existing;
    },
  });
}

function patchPthreadWasmImports(wasmModuleOrBytes, rawImports, wasmNs, hostImports, hostEnv) {
  if (!rawImports || typeof rawImports !== 'object') {
    return rawImports;
  }

  const importedMemory = extractMemoryFromImports(rawImports);
  if (importedMemory && hostImports && typeof hostImports.setMemory === 'function') {
    hostImports.setMemory(importedMemory);
  }

  if (!hostEnv) {
    return rawImports;
  }

  const requiredHostNames = inspectRequiredHostAbiImports(wasmModuleOrBytes, wasmNs);
  if (requiredHostNames && requiredHostNames.size === 0) {
    // Side module imports none of the 5 host_* ABI symbols; leave imports & imports.env untouched.
    return rawImports;
  }

  const hasEnvObj = Boolean(rawImports.env && typeof rawImports.env === 'object');
  if (!hasEnvObj) {
    if (!requiredHostNames || requiredHostNames.size === 0) {
      return rawImports;
    }
    const synthesizedEnv = {};
    requiredHostNames.forEach((name) => {
      if (typeof hostEnv[name] === 'function') {
        synthesizedEnv[name] = hostEnv[name];
      }
    });
    const patchedImports = Object.assign({}, rawImports);
    patchedImports.env = synthesizedEnv;
    return patchedImports;
  }

  const patchedEnv = createPatchedPthreadEnv(rawImports.env, hostEnv, requiredHostNames);
  if (patchedEnv === rawImports.env) {
    return rawImports;
  }

  // Shallow-copy top-level namespace map so GOT.mem, GOT.func, wasi_snapshot_preview1, etc.
  // keep their exact object/Proxy references untouched.
  const patchedImports = Object.assign({}, rawImports);
  patchedImports.env = patchedEnv;
  return patchedImports;
}

/**
 * Wrap `globalScope.WebAssembly.Instance` (and `WebAssembly.instantiate` when present) so that
 * inside an Emscripten `em-pthread` worker, both the main module and `SIDE_MODULE=1` side modules
 * bind `imports.env.memory` (when present) onto `hostImports` and replace missing or `.stub === true`
 * entries for the 5 host ABI imports (`host_log`, `host_httpfs_*`, `host_wgpu_submit`) while
 * preserving `GOT.mem`, `GOT.func`, `env.__memory_base`, `env.__table_base`,
 * `env.__indirect_function_table`, dynamic-linking Proxies, non-stub functions, and
 * `WebAssembly.Instance.prototype` / `instanceof`.
 * Returns an idempotent restore function.
 */
function installPthreadHostImportShim(globalScope, hostImports) {
  if (
    !globalScope ||
    !globalScope.WebAssembly ||
    typeof globalScope.WebAssembly.Instance !== 'function'
  ) {
    return function noopRestore() {};
  }

  const wasmNs = globalScope.WebAssembly;
  const OriginalInstance = wasmNs.Instance;
  const OriginalInstantiate =
    typeof wasmNs.instantiate === 'function' ? wasmNs.instantiate : null;
  const hostEnv =
    hostImports && typeof hostImports === 'object'
      ? hostImports.env && typeof hostImports.env === 'object'
        ? hostImports.env
        : hostImports
      : null;

  const bindExportedMemoryIfNeeded = (rawImports, instance) => {
    if (
      (!rawImports || !extractMemoryFromImports(rawImports)) &&
      instance &&
      instance.exports &&
      instance.exports.memory &&
      hostImports &&
      typeof hostImports.setMemory === 'function'
    ) {
      hostImports.setMemory(instance.exports.memory);
    }
  };

  const WrappedInstance = new Proxy(OriginalInstance, {
    construct(target, args, newTarget) {
      const nextArgs = Array.prototype.slice.call(args);
      const rawImports = nextArgs.length >= 2 ? nextArgs[1] : undefined;

      if (rawImports && typeof rawImports === 'object') {
        nextArgs[1] = patchPthreadWasmImports(
          nextArgs[0],
          rawImports,
          wasmNs,
          hostImports,
          hostEnv,
        );
      }

      const targetNew = newTarget === WrappedInstance ? target : newTarget;
      const instance = Reflect.construct(target, nextArgs, targetNew);
      bindExportedMemoryIfNeeded(rawImports, instance);
      return instance;
    },
  });

  wasmNs.Instance = WrappedInstance;

  let WrappedInstantiate = null;
  if (OriginalInstantiate) {
    WrappedInstantiate = function wrappedWasmInstantiate(moduleOrBytes, imports) {
      const patchedImports =
        imports && typeof imports === 'object'
          ? patchPthreadWasmImports(moduleOrBytes, imports, wasmNs, hostImports, hostEnv)
          : imports;
      const res =
        arguments.length >= 2
          ? OriginalInstantiate.call(wasmNs, moduleOrBytes, patchedImports)
          : OriginalInstantiate.call(wasmNs, moduleOrBytes);
      if (res && typeof res.then === 'function') {
        return res.then((instantiated) => {
          const instance =
            instantiated && instantiated.instance ? instantiated.instance : instantiated;
          bindExportedMemoryIfNeeded(imports, instance);
          return instantiated;
        });
      }
      return res;
    };
    wasmNs.instantiate = WrappedInstantiate;
  }

  let restored = false;
  return function restorePthreadHostImportShim() {
    if (restored) {
      return;
    }
    restored = true;
    if (wasmNs.Instance === WrappedInstance) {
      wasmNs.Instance = OriginalInstance;
    }
    if (OriginalInstantiate && wasmNs.instantiate === WrappedInstantiate) {
      wasmNs.instantiate = OriginalInstantiate;
    }
  };
}

/**
 * Bootstrap an Emscripten `em-pthread` worker spawned via `new Worker(self.location.href, { name: "em-pthread" })`:
 * - Streams `host_log` over `BroadcastChannel('game-progress')`.
 * - Relays `host_wgpu_submit` batches over `BroadcastChannel('playgta6-wgpu-pthread')` to the coordinator worker.
 * - Acquires `httpfsMailboxSab` via `BroadcastChannel('playgta6-httpfs-sab')` so pthreads can call `host_httpfs_*`.
 * - Configures side-module `locateFile` resolution against `dist/` for `guest_part_*.wasm`.
 * - Installs the `WebAssembly.Instance` host-import shim BEFORE `importScripts(runtimeScriptUrl)`.
 */
function bootstrapEmscriptenPthreadWorker(options = {}) {
  const scope =
    options.scope ||
    (typeof self !== 'undefined' ? self : typeof globalThis !== 'undefined' ? globalThis : null);
  const importScriptsFn =
    typeof options.importScriptsFn === 'function'
      ? options.importScriptsFn
      : typeof importScripts === 'function'
        ? importScripts
        : null;
  const runtimeScriptUrl =
    typeof options.runtimeScriptUrl === 'string' && options.runtimeScriptUrl.trim().length > 0
      ? options.runtimeScriptUrl.trim()
      : DEFAULT_RUNTIME_SCRIPT_RELATIVE;
  const scopeLocationHref =
    typeof options.locationHref === 'string' && options.locationHref.trim().length > 0
      ? options.locationHref.trim()
      : scope && scope.location && typeof scope.location.href === 'string'
        ? scope.location.href
        : '';
  const resolvedRuntimeScriptUrl = resolveRuntimeScriptUrl(
    runtimeScriptUrl === DEFAULT_RUNTIME_SCRIPT_RELATIVE ? undefined : runtimeScriptUrl,
    scopeLocationHref,
  );
  const BroadcastChannelCtor =
    options.BroadcastChannelCtor !== undefined
      ? options.BroadcastChannelCtor
      : scope && typeof scope.BroadcastChannel === 'function'
        ? scope.BroadcastChannel
        : typeof BroadcastChannel === 'function'
          ? BroadcastChannel
          : null;
  const mailboxApi =
    options.httpfsMailboxApi !== undefined
      ? options.httpfsMailboxApi
      : (scope && scope.PlayGta6HttpfsMailbox) ||
        (typeof globalThis !== 'undefined' && globalThis.PlayGta6HttpfsMailbox) ||
        null;

  const progressChannel = tryOpenBroadcastChannel(BroadcastChannelCtor, 'game-progress');
  const wgpuChannel = tryOpenBroadcastChannel(BroadcastChannelCtor, PTHREAD_WGPU_CHANNEL_NAME);
  const httpfsChannel = tryOpenBroadcastChannel(
    BroadcastChannelCtor,
    PTHREAD_HTTPFS_SAB_CHANNEL_NAME,
  );

  const pthreadReport = (label) => {
    if (!progressChannel || typeof progressChannel.postMessage !== 'function') {
      return;
    }
    try {
      progressChannel.postMessage({ label });
    } catch {
      // Progress channel broadcast failures must never crash guest pthreads.
    }
  };

  const pthreadLocateFile = createRuntimeLocateFile(
    resolvedRuntimeScriptUrl,
    options.locateFileBase,
    {
      reportFn: pthreadReport,
      sideModuleNames: sanitizeSideModuleUrls(options.sideModuleUrls),
    },
  );

  // Ensure Emscripten's pthread bootstrap resolves side modules (`guest_part_*.wasm`)
  // relative to `dist/` even if the worker's `self.location.href` is `loader.js`.
  let installedScopeModule = false;
  let installedLocateFileOnModule = false;
  let installedMainScriptOnModule = false;
  if (scope && typeof scope === 'object') {
    if (!scope.Module || typeof scope.Module !== 'object') {
      scope.Module = {
        locateFile: pthreadLocateFile,
        mainScriptUrlOrBlob: resolvedRuntimeScriptUrl,
      };
      installedScopeModule = true;
    } else {
      if (typeof scope.Module.locateFile !== 'function') {
        scope.Module.locateFile = pthreadLocateFile;
        installedLocateFileOnModule = true;
      }
      if (!scope.Module.mainScriptUrlOrBlob) {
        scope.Module.mainScriptUrlOrBlob = resolvedRuntimeScriptUrl;
        installedMainScriptOnModule = true;
      }
    }
  }

  let pthreadHttpfsClient = null;
  const hostImports = createHostImports({
    log: (msg) => pthreadReport(`[wasm] ${msg}`),
    report: () => {},
    wgpuSubmit: (bytes) => {
      if (!wgpuChannel || typeof wgpuChannel.postMessage !== 'function') {
        return;
      }
      const copy = bytes.slice();
      try {
        wgpuChannel.postMessage({ type: 'wgpu-submit', cmdBuffer: copy.buffer });
      } catch {
        // Non-fatal GPU command relay failure.
      }
    },
    getHttpfsClient: () => pthreadHttpfsClient,
  });

  const attachHttpfsSab = (sab) => {
    if (!sab || !mailboxApi || typeof mailboxApi.HttpfsMailboxClient !== 'function') {
      return false;
    }
    try {
      pthreadHttpfsClient = new mailboxApi.HttpfsMailboxClient(sab);
      hostImports.setHttpfsClient(pthreadHttpfsClient);
      return true;
    } catch {
      return false;
    }
  };

  if (options.httpfsMailboxSab) {
    attachHttpfsSab(options.httpfsMailboxSab);
  }

  const onPthreadBootstrapMessage = (ev) => {
    const msg = ev && ev.data;
    if (msg && typeof msg === 'object' && msg.__playgta6_httpfs_sab) {
      attachHttpfsSab(msg.__playgta6_httpfs_sab);
      if (typeof ev.stopImmediatePropagation === 'function') {
        ev.stopImmediatePropagation();
      }
    }
  };

  let installedBootstrapMessageListener = false;
  if (scope && typeof scope.addEventListener === 'function') {
    scope.addEventListener('message', onPthreadBootstrapMessage);
    installedBootstrapMessageListener = true;
  }

  if (httpfsChannel) {
    httpfsChannel.onmessage = (ev) => {
      const msg = ev && ev.data;
      if (msg && msg.type === 'httpfs-sab' && msg.sab) {
        attachHttpfsSab(msg.sab);
      }
    };
    try {
      if (typeof httpfsChannel.postMessage === 'function') {
        httpfsChannel.postMessage({ type: 'request-httpfs-sab' });
      }
    } catch {
      // Ignore runtimes where BroadcastChannel messaging is restricted.
    }
  }

  const wasmTargetScope =
    scope && scope.WebAssembly
      ? scope
      : typeof globalThis !== 'undefined' && globalThis.WebAssembly
        ? globalThis
        : scope;
  const restoreWasmInstance = installPthreadHostImportShim(wasmTargetScope, hostImports);

  if (importScriptsFn) {
    importScriptsFn(runtimeScriptUrl);
  }

  return {
    hostImports,
    locateFile: pthreadLocateFile,
    runtimeScriptUrl: resolvedRuntimeScriptUrl,
    restoreWasmInstance,
    attachHttpfsSab,
    channels: {
      progress: progressChannel,
      wgpu: wgpuChannel,
      httpfs: httpfsChannel,
    },
    cleanup() {
      restoreWasmInstance();
      if (
        installedBootstrapMessageListener &&
        scope &&
        typeof scope.removeEventListener === 'function'
      ) {
        try {
          scope.removeEventListener('message', onPthreadBootstrapMessage);
        } catch {
          // Ignore removeEventListener errors during cleanup.
        }
        installedBootstrapMessageListener = false;
      }
      if (scope && typeof scope === 'object') {
        if (installedScopeModule) {
          delete scope.Module;
        } else if (scope.Module && typeof scope.Module === 'object') {
          if (installedLocateFileOnModule && scope.Module.locateFile === pthreadLocateFile) {
            delete scope.Module.locateFile;
          }
          if (
            installedMainScriptOnModule &&
            scope.Module.mainScriptUrlOrBlob === resolvedRuntimeScriptUrl
          ) {
            delete scope.Module.mainScriptUrlOrBlob;
          }
        }
      }
      const channels = [progressChannel, wgpuChannel, httpfsChannel];
      for (let i = 0; i < channels.length; i += 1) {
        const ch = channels[i];
        if (ch && typeof ch.close === 'function') {
          try {
            ch.close();
          } catch {
            // Ignore BroadcastChannel close errors.
          }
        }
      }
    },
  };
}

function isAbsoluteUrl(value) {
  return typeof value === 'string' && ABSOLUTE_URL_RE.test(value);
}

function dirnameWithSlash(relPath) {
  if (typeof relPath !== 'string' || relPath.length === 0) {
    return 'dist/';
  }
  const clean = relPath.split(/[?#]/, 1)[0];
  const slash = clean.lastIndexOf('/');
  if (slash < 0) {
    return 'dist/';
  }
  return clean.slice(0, slash + 1);
}

/**
 * Resolve the Emscripten main script URL (`dist/runtime.js`) relative to the
 * coordinator worker location so `-pthread` workers spawn `dist/runtime.js`
 * instead of recursively spawning `loader.js` via `self.location.href`.
 */
function resolveRuntimeScriptUrl(runtimeScriptUrl, locationHref) {
  const rawScript =
    typeof runtimeScriptUrl === 'string' && runtimeScriptUrl.trim().length > 0
      ? runtimeScriptUrl.trim()
      : '';
  if (rawScript && isAbsoluteUrl(rawScript)) {
    return rawScript;
  }

  const baseHref =
    typeof locationHref === 'string' && locationHref.trim().length > 0
      ? locationHref.trim()
      : typeof self !== 'undefined' &&
          self &&
          self.location &&
          typeof self.location.href === 'string'
        ? self.location.href
        : '';

  const relativeTarget = rawScript || DEFAULT_RUNTIME_SCRIPT_RELATIVE;
  if (baseHref && typeof URL === 'function') {
    try {
      return new URL(relativeTarget, baseHref).href;
    } catch {
      // Fall through to relative fallback when baseHref is not a valid URL.
    }
  }

  return rawScript || DEFAULT_RUNTIME_SCRIPT_FALLBACK;
}

function extractWasmBasename(value) {
  if (typeof value !== 'string' || value.length === 0) {
    return '';
  }
  const clean = value.split(/[?#]/, 1)[0];
  return basenameOfPath(clean);
}

function sanitizeSideModuleUrl(raw) {
  if (typeof raw !== 'string') {
    return null;
  }
  const trimmed = raw.trim();
  if (!trimmed || trimmed.includes('\0') || trimmed.includes('\\')) {
    return null;
  }
  const withoutQuery = trimmed.split(/[?#]/, 1)[0];
  if (!/\.wasm$/i.test(withoutQuery)) {
    return null;
  }
  if (isAbsoluteUrl(trimmed)) {
    if (!/^https?:/i.test(trimmed)) {
      return null;
    }
    if (typeof URL === 'function') {
      try {
        const parsed = new URL(trimmed);
        if (parsed.pathname.split('/').includes('..')) {
          return null;
        }
      } catch {
        return null;
      }
    }
    return trimmed;
  }
  const segments = withoutQuery.replace(/^\/+/, '').split('/');
  for (let i = 0; i < segments.length; i += 1) {
    if (segments[i] === '..') {
      return null;
    }
  }
  return trimmed;
}

function sanitizeSideModuleUrls(rawList) {
  if (!Array.isArray(rawList)) {
    return [];
  }
  const out = [];
  const seen = new Set();
  for (let i = 0; i < rawList.length; i += 1) {
    const clean = sanitizeSideModuleUrl(rawList[i]);
    if (clean && !seen.has(clean)) {
      seen.add(clean);
      out.push(clean);
    }
  }
  return out;
}

/**
 * Resolve a runtime asset or Emscripten `SIDE_MODULE=1` (`guest_part_*.wasm`)
 * filename/URL relative to `locateFileBase` or the `dist/` directory of
 * `runtimeScriptUrl`, avoiding duplicate `dist/dist/` prefixes.
 */
function resolveSideModuleUrl(file, runtimeScriptUrl, locateFileBase) {
  if (typeof file !== 'string') {
    return '';
  }
  const trimmed = file.trim();
  if (trimmed.length === 0) {
    return '';
  }
  if (isAbsoluteUrl(trimmed)) {
    return trimmed;
  }

  const explicitBase =
    typeof locateFileBase === 'string' && locateFileBase.trim().length > 0
      ? ensureTrailingSlash(locateFileBase.trim())
      : '';
  const effectiveScriptUrl =
    typeof runtimeScriptUrl === 'string' && runtimeScriptUrl.trim().length > 0
      ? runtimeScriptUrl.trim()
      : DEFAULT_RUNTIME_SCRIPT_FALLBACK;

  if (trimmed.startsWith('/')) {
    const absBase = isAbsoluteUrl(explicitBase)
      ? explicitBase
      : isAbsoluteUrl(effectiveScriptUrl)
        ? effectiveScriptUrl
        : '';
    if (absBase && typeof URL === 'function') {
      try {
        return new URL(trimmed, absBase).href;
      } catch {
        // Fall through to returning root-relative path.
      }
    }
    return trimmed;
  }

  const relNoDot = trimmed.replace(/^\.\//, '');

  if (explicitBase) {
    if (isAbsoluteUrl(explicitBase) && typeof URL === 'function') {
      try {
        return new URL(relNoDot, explicitBase).href;
      } catch {
        // Fall through to string join.
      }
    }
    return `${explicitBase}${relNoDot}`;
  }

  const scriptDir = dirnameWithSlash(effectiveScriptUrl);
  const effectiveRel =
    (scriptDir === 'dist/' || scriptDir.endsWith('/dist/')) && relNoDot.startsWith('dist/')
      ? relNoDot.slice('dist/'.length)
      : relNoDot;

  if (isAbsoluteUrl(effectiveScriptUrl) && typeof URL === 'function') {
    try {
      return new URL(effectiveRel, effectiveScriptUrl).href;
    } catch {
      // Fall through to relative dist/ join.
    }
  }
  return `${scriptDir}${effectiveRel}`;
}

function createRuntimeLocateFile(runtimeScriptUrl, locateFileBase, options = {}) {
  const reportFn = typeof options.reportFn === 'function' ? options.reportFn : null;
  const knownSideModules = new Set();
  if (Array.isArray(options.sideModuleNames)) {
    for (let i = 0; i < options.sideModuleNames.length; i += 1) {
      const base = extractWasmBasename(options.sideModuleNames[i]);
      if (base) {
        knownSideModules.add(base.toLowerCase());
      }
    }
  }
  const reportedSideModules = new Set();

  return function locateFile(file) {
    const resolved = resolveSideModuleUrl(file, runtimeScriptUrl, locateFileBase);
    if (!resolved) {
      return '';
    }
    if (reportFn) {
      const base = extractWasmBasename(resolved) || extractWasmBasename(file);
      const baseLower = base ? base.toLowerCase() : '';
      if (
        baseLower &&
        (GUEST_SIDE_MODULE_BASENAME_RE.test(baseLower) || knownSideModules.has(baseLower)) &&
        !reportedSideModules.has(baseLower)
      ) {
        reportedSideModules.add(baseLower);
        try {
          reportFn(`Loading Wasm side module (${base})...`);
        } catch {
          // Progress channel errors must never break locateFile.
        }
      }
    }
    return resolved;
  };
}

function extractMemoryFromImports(imports) {
  if (!imports || typeof imports !== 'object') {
    return null;
  }
  if (imports.env && imports.env.memory) {
    return imports.env.memory;
  }
  if (imports.a && imports.a.memory) {
    return imports.a.memory;
  }
  return null;
}

function mergeHostImportsIntoWasmImports(imports, hostEnv) {
  const merged = Object.assign({}, imports && typeof imports === 'object' ? imports : {});
  const baseEnv = merged.env && typeof merged.env === 'object' ? merged.env : {};
  merged.env = Object.assign({}, baseEnv, hostEnv || {});
  return merged;
}

function createInstantiateWasmHandler(options = {}) {
  const locateFile =
    typeof options.locateFile === 'function' ? options.locateFile : (f) => `dist/${f}`;
  const hostImports = options.hostImports || null;
  const reportFn = typeof options.reportFn === 'function' ? options.reportFn : report;
  const fetchFn =
    typeof options.fetchFn === 'function'
      ? options.fetchFn
      : typeof fetch === 'function'
        ? fetch
        : null;
  const wasmBinary = options.wasmBinary || null;
  const onError = typeof options.onError === 'function' ? options.onError : null;

  return function instantiateWasm(imports, receiveInstance) {
    const importedMemory = extractMemoryFromImports(imports);
    if (importedMemory && hostImports && typeof hostImports.setMemory === 'function') {
      hostImports.setMemory(importedMemory);
    }

    const mergedImports = mergeHostImportsIntoWasmImports(
      imports,
      hostImports && hostImports.env ? hostImports.env : null,
    );

    const finish = (result) => {
      const instance = result && result.instance ? result.instance : result;
      const mod =
        result && result.module
          ? result.module
          : typeof WebAssembly !== 'undefined' && wasmBinary instanceof WebAssembly.Module
            ? wasmBinary
            : undefined;
      if (
        instance &&
        instance.exports &&
        instance.exports.memory &&
        hostImports &&
        typeof hostImports.setMemory === 'function'
      ) {
        hostImports.setMemory(instance.exports.memory);
      }
      if (typeof receiveInstance === 'function') {
        receiveInstance(instance, mod);
      }
    };

    const handleFailure = (err) => {
      try {
        reportFn(`runtime wasm instantiation failed: ${err}`);
      } catch {
        // Progress channel failures must not mask instantiation errors.
      }
      if (onError) {
        try {
          onError(err);
        } catch {
          // Ignore secondary error hook exceptions.
        }
      }
    };

    const wasmUrl = locateFile('runtime.wasm');
    let instantiatePromise;
    if (wasmBinary) {
      instantiatePromise = WebAssembly.instantiate(wasmBinary, mergedImports);
    } else if (!fetchFn) {
      handleFailure(new Error(`fetch unavailable while loading ${wasmUrl}`));
      return {};
    } else {
      instantiatePromise = Promise.resolve()
        .then(() => fetchFn(wasmUrl, { credentials: 'same-origin' }))
        .then((resp) => {
          if (!resp || resp.ok === false) {
            const status = resp && resp.status !== undefined ? resp.status : 'unknown';
            throw new Error(`Failed to fetch ${wasmUrl} (HTTP ${status})`);
          }
          const contentType =
            resp.headers && typeof resp.headers.get === 'function'
              ? String(resp.headers.get('content-type') || '')
              : '';
          const canStream =
            typeof WebAssembly.instantiateStreaming === 'function' &&
            (typeof Response === 'undefined' || resp instanceof Response) &&
            contentType.toLowerCase().includes('application/wasm');
          if (canStream) {
            return WebAssembly.instantiateStreaming(Promise.resolve(resp), mergedImports);
          }
          return resp.arrayBuffer().then((bytes) => WebAssembly.instantiate(bytes, mergedImports));
        });
    }

    instantiatePromise.then(finish, handleFailure);
    return {};
  };
}

/**
 * Build the Emscripten `createRuntimeModule(...)` configuration object with:
 * - `mainScriptUrlOrBlob` pinned to `dist/runtime.js` (never `loader.js`),
 * - `locateFile` resolving `runtime.wasm` / `runtime.worker.js` cleanly under `dist/`,
 * - `env` + `instantiateWasm` merging `host_log`, `host_httpfs_*`, and `host_wgpu_submit`.
 */
function buildRuntimeModuleConfig(options = {}) {
  let boundModule = null;
  const activeReportFn =
    typeof options.reportFn === 'function'
      ? options.reportFn
      : typeof options.report === 'function'
        ? options.report
        : report;
  const activeHttpfsClient =
    options.httpfsClient !== undefined ? options.httpfsClient : httpfsClient;
  const activeUserdataStore =
    options.userdataStore !== undefined ? options.userdataStore : userdataStore;
  const activeGpuWorker = options.gpuWorker !== undefined ? options.gpuWorker : gpuWorker;

  const wgpuSubmit =
    typeof options.wgpuSubmit === 'function'
      ? options.wgpuSubmit
      : (bytes) => {
          if (!activeGpuWorker) return;
          const copy = bytes.slice().buffer;
          activeGpuWorker.postMessage({ cmdBuffer: copy }, [copy]);
        };

  const logFn =
    typeof options.log === 'function' ? options.log : (msg) => activeReportFn(`[wasm] ${msg}`);

  const activeHostImports =
    options.hostImports && typeof options.hostImports === 'object'
      ? options.hostImports
      : createHostImports({
          memory: options.memory || null,
          getMemory: () =>
            options.memory ||
            (boundModule &&
              (boundModule.wasmMemory || (boundModule.HEAPU8 && boundModule.HEAPU8.buffer))) ||
            null,
          log: logFn,
          report: activeReportFn,
          wgpuSubmit,
          httpfsClient: activeHttpfsClient,
          userdataStore: activeUserdataStore,
          onFirstPresent:
            typeof options.onFirstPresent === 'function' ? options.onFirstPresent : null,
          emitFirstPresentLog: options.emitFirstPresentLog === true,
        });

  const resolvedRuntimeScriptUrl = resolveRuntimeScriptUrl(
    options.runtimeScriptUrl,
    options.locationHref,
  );
  const rawDynamicLibraries = sanitizeSideModuleUrls(options.dynamicLibraries);
  const rawSideModuleUrls = sanitizeSideModuleUrls(options.sideModuleUrls);
  const combinedDynamicLibraries = [];
  const seenSideModules = new Set();
  const addSideModuleEntry = (entry) => {
    const key = entry.toLowerCase();
    const baseKey = extractWasmBasename(entry).toLowerCase();
    if (seenSideModules.has(key) || (baseKey && seenSideModules.has(baseKey))) {
      return;
    }
    seenSideModules.add(key);
    if (baseKey) {
      seenSideModules.add(baseKey);
    }
    combinedDynamicLibraries.push(entry);
  };
  for (let i = 0; i < rawDynamicLibraries.length; i += 1) {
    addSideModuleEntry(rawDynamicLibraries[i]);
  }
  for (let i = 0; i < rawSideModuleUrls.length; i += 1) {
    addSideModuleEntry(rawSideModuleUrls[i]);
  }
  const resolvedSideModuleUrls = combinedDynamicLibraries.map((entry) =>
    resolveSideModuleUrl(entry, resolvedRuntimeScriptUrl, options.locateFileBase),
  );

  const locateFile = createRuntimeLocateFile(resolvedRuntimeScriptUrl, options.locateFileBase, {
    reportFn: activeReportFn,
    sideModuleNames: combinedDynamicLibraries,
  });

  const preRun = Array.isArray(options.preRun)
    ? options.preRun
    : [
        (mod) => {
          boundModule = mod || null;
          if (
            mod &&
            (mod.wasmMemory || (mod.HEAPU8 && mod.HEAPU8.buffer)) &&
            typeof activeHostImports.setMemory === 'function'
          ) {
            activeHostImports.setMemory(mod.wasmMemory || mod.HEAPU8.buffer);
          }
          if (!activeUserdataStore || !mod || !mod.FS) return;
          const useRunDep =
            typeof mod.addRunDependency === 'function' &&
            typeof mod.removeRunDependency === 'function';
          if (useRunDep) {
            mod.addRunDependency('userdata-restore');
          }
          activeUserdataStore
            .restoreAll(mod.FS, { timeoutMs: 8000 })
            .then((res) => {
              if (res.timedOut) {
                activeReportFn(`userdata restore timed out (continuing boot): ${res.error}`);
              } else if (res.error) {
                activeReportFn(`userdata restore skipped: ${res.error}`);
              } else {
                activeReportFn(`userdata restored (${res.restored} files)`);
              }
            })
            .catch((err) => {
              activeReportFn(`userdata restore warning: ${err}`);
            })
            .finally(() => {
              if (useRunDep) {
                mod.removeRunDependency('userdata-restore');
              }
            });
        },
      ];

  const userdataApiObj = activeHostImports.userdata || {};
  const config = {
    mainScriptUrlOrBlob: resolvedRuntimeScriptUrl,
    locateFile,
    preRun,
    env: activeHostImports.env,
    hostImports: activeHostImports,
    httpfsMailboxSab:
      options.httpfsMailboxSab !== undefined ? options.httpfsMailboxSab : httpfsMailboxSab,
    inputSab: options.inputSab !== undefined ? options.inputSab : inputSab,
    hostCallbacks: options.hostCallbacks || {
      log: logFn,
      wgpuSubmit,
      httpfsOpen: (relPath) => (activeHttpfsClient ? activeHttpfsClient.open(relPath) : -1),
      httpfsRead: (handle, offset, dst, len) =>
        activeHttpfsClient ? activeHttpfsClient.read(handle, offset, dst, len) : -1,
      httpfsClose: (handle) => (activeHttpfsClient ? activeHttpfsClient.close(handle) : false),
      userdataRestore: userdataApiObj.restore,
      userdataSave: userdataApiObj.save,
      userdataDelete: userdataApiObj.delete,
    },
  };

  if (combinedDynamicLibraries.length > 0) {
    config.dynamicLibraries = combinedDynamicLibraries.slice();
    config.sideModuleUrls = resolvedSideModuleUrls.slice();
  }

  if (typeof options.instantiateWasm === 'function') {
    config.instantiateWasm = options.instantiateWasm;
  } else if (options.instantiateWasm !== null && options.instantiateWasm !== false) {
    config.instantiateWasm = createInstantiateWasmHandler({
      locateFile,
      hostImports: activeHostImports,
      reportFn: activeReportFn,
      fetchFn: options.fetchFn,
      wasmBinary: options.wasmBinary,
      onError: options.onWasmError,
    });
  }

  return config;
}

function createRuntimeModuleOptions(overrides = {}) {
  return buildRuntimeModuleConfig(overrides);
}

let coordinatorFirstPresentEmitted = false;
let configuredSideModuleUrls = null;
let configuredDynamicLibraries = null;

async function tryBootWasmRuntime() {
  try {
    const head = await fetch('./dist/runtime.js', { method: 'HEAD' });
    if (!head.ok) {
      report('No web/dist/runtime.js built yet (running WebGPU command-ring self-test).');
      const pkt = encodeTestFramePacket(0.08, 0.14, 0.24, 1.0);
      gpuWorker.postMessage({ cmdBuffer: pkt }, [pkt]);
      return;
    }
    report('Loading Emscripten Wasm64 runtime (dist/runtime.js)...');
    importScripts('./dist/runtime.js');
    if (typeof self.createRuntimeModule === 'function') {
      const runtimeOptions = {
        onFirstPresent: () => {
          coordinatorFirstPresentEmitted = true;
        },
        onWasmError: (err) => {
          if (progress) {
            progress.postMessage({ error: `runtime wasm: ${err}` });
          }
        },
      };
      if (configuredSideModuleUrls) {
        runtimeOptions.sideModuleUrls = configuredSideModuleUrls;
      }
      if (configuredDynamicLibraries) {
        runtimeOptions.dynamicLibraries = configuredDynamicLibraries;
      }
      await self.createRuntimeModule(createRuntimeModuleOptions(runtimeOptions));
    }
  } catch (err) {
    if (progress) {
      progress.postMessage({ error: `runtime boot: ${err}` });
    }
  }
}

if (typeof self !== 'undefined') {
  const OriginalWorker =
    typeof self.Worker === 'function'
      ? self.Worker
      : typeof Worker === 'function'
        ? Worker
        : null;
  if (OriginalWorker) {
    const WrappedWorker = new Proxy(OriginalWorker, {
      construct(target, args, newTarget) {
        const nextArgs = Array.prototype.slice.call(args);
        const workerOpts = nextArgs.length >= 2 ? nextArgs[1] : undefined;
        const isEmPthread = Boolean(
          workerOpts &&
            typeof workerOpts === 'object' &&
            isEmscriptenPthreadWorkerName(workerOpts.name),
        );
        if (
          isEmPthread &&
          self.location &&
          typeof self.location.href === 'string' &&
          self.location.href.length > 0
        ) {
          nextArgs[0] = self.location.href;
        }
        const targetNew = newTarget === WrappedWorker ? target : newTarget;
        const instance = Reflect.construct(target, nextArgs, targetNew);
        if (
          isEmPthread &&
          httpfsMailboxSab &&
          instance &&
          typeof instance.postMessage === 'function'
        ) {
          try {
            instance.postMessage({ __playgta6_httpfs_sab: httpfsMailboxSab });
          } catch {
            // Non-fatal: BroadcastChannel handshake remains as fallback.
          }
        }
        return instance;
      },
    });
    try {
      self.Worker = WrappedWorker;
    } catch {
      // Ignore if self.Worker is non-writable in a custom harness.
    }
    if (
      typeof globalThis !== 'undefined' &&
      globalThis !== self &&
      globalThis.Worker === OriginalWorker
    ) {
      try {
        globalThis.Worker = WrappedWorker;
      } catch {
        // Ignore if globalThis.Worker is non-writable.
      }
    }
  }

  const postToMain =
    typeof self.postMessage === 'function' ? (payload) => self.postMessage(payload) : null;

  const wgpuPthreadChannel =
    typeof BroadcastChannel === 'function'
      ? tryOpenBroadcastChannel(BroadcastChannel, PTHREAD_WGPU_CHANNEL_NAME)
      : null;
  const httpfsSabChannel =
    typeof BroadcastChannel === 'function'
      ? tryOpenBroadcastChannel(BroadcastChannel, PTHREAD_HTTPFS_SAB_CHANNEL_NAME)
      : null;

  const broadcastHttpfsSab = () => {
    if (
      !httpfsSabChannel ||
      !httpfsMailboxSab ||
      typeof httpfsSabChannel.postMessage !== 'function'
    ) {
      return;
    }
    try {
      httpfsSabChannel.postMessage({ type: 'httpfs-sab', sab: httpfsMailboxSab });
    } catch {
      // Some runtimes reject SharedArrayBuffer over BroadcastChannel; degrade safely.
    }
  };

  if (wgpuPthreadChannel) {
    wgpuPthreadChannel.onmessage = (ev) => {
      const data = ev && ev.data;
      const raw =
        data &&
        typeof data === 'object' &&
        !(data instanceof ArrayBuffer) &&
        !ArrayBuffer.isView(data)
          ? data.cmdBuffer || data.bytes || null
          : data;
      const view = asDataView(raw);
      if (!view || view.byteLength === 0) {
        return;
      }
      const copyBytes = new Uint8Array(view.byteLength);
      copyBytes.set(new Uint8Array(view.buffer, view.byteOffset, view.byteLength));
      if (!coordinatorFirstPresentEmitted) {
        const info = inspectGpuPacketBatchForPresent(copyBytes);
        if (info.hasPresent) {
          coordinatorFirstPresentEmitted = true;
          report('[boot] FIRST_PRESENT');
        }
      }
      if (gpuWorker && typeof gpuWorker.postMessage === 'function') {
        const copyBuf = copyBytes.buffer;
        gpuWorker.postMessage({ cmdBuffer: copyBuf }, [copyBuf]);
      }
    };
  }

  if (httpfsSabChannel) {
    httpfsSabChannel.onmessage = (ev) => {
      const msg = ev && ev.data;
      if (msg && msg.type === 'request-httpfs-sab') {
        broadcastHttpfsSab();
      }
    };
  }

  self.onmessage = (ev) => {
    const msg = ev.data || {};
    if (msg.type === 'request-gpu-telemetry') {
      if (gpuWorker && typeof gpuWorker.postMessage === 'function') {
        const statsReq = { stats: true };
        if (msg.id !== undefined) {
          statsReq.id = msg.id;
        }
        gpuWorker.postMessage(statsReq);
        if (msg.includeMissingShaders !== false) {
          const missingReq = { missingShaders: true };
          if (msg.id !== undefined) {
            missingReq.id = msg.id;
          }
          if (msg.limit !== undefined) {
            missingReq.limit = msg.limit;
          }
          gpuWorker.postMessage(missingReq);
        }
      }
      return;
    }
    if (msg.resize && gpuWorker) {
      gpuWorker.postMessage({ resize: true, width: msg.width, height: msg.height });
      return;
    }

    if (!msg.canvas) return;
    inputSab = msg.inputSab || null;
    configuredSideModuleUrls = Array.isArray(msg.sideModuleUrls) ? msg.sideModuleUrls : null;
    configuredDynamicLibraries = Array.isArray(msg.dynamicLibraries)
      ? msg.dynamicLibraries
      : null;

    if (
      typeof SharedArrayBuffer !== 'undefined' &&
      httpfsMailboxApi &&
      typeof httpfsMailboxApi.createHttpfsMailboxBuffer === 'function'
    ) {
      httpfsMailboxSab = httpfsMailboxApi.createHttpfsMailboxBuffer();
      httpfsClient = new httpfsMailboxApi.HttpfsMailboxClient(httpfsMailboxSab);
      broadcastHttpfsSab();
    } else {
      httpfsMailboxSab = null;
      httpfsClient = null;
    }

    ioWorker = new Worker('./workers/io_worker.js');
    gpuWorker = new Worker('./workers/wgpu_worker.js');
    audioWorker = new Worker('./workers/audio_worker.js');

    let readyCount = 0;
    const onWorkerReady = (name, detail) => {
      report(`${name} ready${detail ? ` (${detail})` : ''}`);
      if (++readyCount === 3) {
        tryBootWasmRuntime();
      }
    };

    ioWorker.onmessage = (e) => {
      if (e.data.ready) onWorkerReady('io_worker', e.data.summary);
      else if (e.data.error && progress) progress.postMessage({ error: `io_worker: ${e.data.error}` });
    };
    gpuWorker.onmessage = (e) => {
      const data = e && e.data;
      if (!data || typeof data !== 'object') return;
      if (data.ready) {
        onWorkerReady('wgpu_worker', data.adapter);
        return;
      }
      if (data.frame) {
        report(`WebGPU frame #${data.frame} presented`);
      } else if (data.error && progress) {
        progress.postMessage({ error: `wgpu_worker: ${data.error}` });
      }
      forwardGpuWorkerMessage(data, postToMain);
    };
    audioWorker.onmessage = (e) => {
      if (e.data.ready) onWorkerReady('audio_worker');
    };

    const workerInits = createWorkerBootInitPayloads(msg);
    ioWorker.postMessage(workerInits.ioInit);
    if (httpfsMailboxSab) {
      ioWorker.postMessage({ attachMailbox: true, sab: httpfsMailboxSab });
    }
    gpuWorker.postMessage(workerInits.gpuInit, [msg.canvas]);
    audioWorker.postMessage({ init: true });
  };
}

const exportedApi = {
  EMSCRIPTEN_PTHREAD_WORKER_NAME,
  PTHREAD_WGPU_CHANNEL_NAME,
  PTHREAD_HTTPFS_SAB_CHANNEL_NAME,
  HOST_ABI_IMPORT_NAMES,
  GPU_OPCODE_PRESENT,
  GPU_PACKET_HEADER_BYTES,
  BOOT_INIT_FORWARD_KEYS,
  isEmscriptenPthreadWorkerName,
  isEmscriptenPthreadWorker,
  createPatchedPthreadEnv,
  installPthreadHostImportShim,
  bootstrapEmscriptenPthreadWorker,
  inspectGpuPacketBatchForPresent,
  normalizeBootHttpfsPath,
  collectBootBridgeInitFields,
  createWorkerBootInitPayloads,
  forwardGpuWorkerMessage,
  createHostImports,
  resolveRuntimeScriptUrl,
  resolveSideModuleUrl,
  sanitizeSideModuleUrls,
  buildRuntimeModuleConfig,
  createRuntimeModuleOptions,
  encodeTestFramePacket,
  ptrToNumber,
};

if (typeof globalThis !== 'undefined') {
  globalThis.PlayGta6Loader = exportedApi;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = exportedApi;
}
})();
