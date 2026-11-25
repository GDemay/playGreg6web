'use strict';

// Main runtime coordinator worker (keeps the browser DOM thread completely free).
(function initPlayGta6Loader() {
const EMSCRIPTEN_PTHREAD_WORKER_NAME = 'em-pthread';
const PROGRESS_CHANNEL_NAME = 'game-progress';
const PTHREAD_WGPU_CHANNEL_NAME = 'playgta6-wgpu-pthread';
const PTHREAD_HTTPFS_SAB_CHANNEL_NAME = 'playgta6-httpfs-sab';
const PTHREAD_CMD_LOAD = 'load';
const PTHREAD_CMD_LOADED = 'loaded';
const PTHREAD_CMD_RUN = 'run';
const PTHREAD_CMD_OBJECT_TRANSFER = 'objectTransfer';
const DEFAULT_MAX_CONCURRENT_PTHREAD_LOADS = 2;
const DEFAULT_GUEST_MANIFEST_RELATIVE = './dist/guest_manifest.json';
const INDIRECT_DISPATCH_MODES = new Set(['shard', 'table']);
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

// playgta6::InputState (40 bytes = 10 x 32-bit words, runtime/abi/playgta6_abi.h & docs/wasm_abi.md).
const INPUT_STATE_BYTES = 40;
const INPUT_STATE_WORDS = 10;
const DEFAULT_INPUT_BRIDGE_POLL_MS = 16;
const INPUT_SNAPSHOT_MAX_RETRIES = 64;
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

function postProgress(payload) {
  if (typeof self === 'undefined') {
    return false;
  }
  const BroadcastChannelCtor =
    typeof self.BroadcastChannel === 'function'
      ? self.BroadcastChannel
      : typeof BroadcastChannel === 'function'
        ? BroadcastChannel
        : null;
  return postBroadcastOnce(BroadcastChannelCtor, PROGRESS_CHANNEL_NAME, payload);
}

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
// W101: render_state_packets.js counters surfaced from wgpu_worker.js `renderStateStats` (W99).
// The first four are always present (0 default); the rest are forwarded only when reported.
const RENDER_STATE_STATS_CORE_KEYS = Object.freeze([
  'computeDispatchCount',
  'computePipelineCount',
  'autoLayoutFallbacks',
  'bindGroupKindMismatches',
]);
const RENDER_STATE_STATS_OPTIONAL_KEYS = Object.freeze([
  'skippedComputeDispatchCount',
  'clampedComputeDispatchCount',
  'rootSigGroup0Layouts',
  'dynamicOffsetRemaps',
  'dynamicOffsetClamps',
  'remappedBindGroupEvictions',
]);
let lastRenderStateStats = null;
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
let activeCoordinatorInputBridge = null;
let httpfsMailboxSab = null;
let httpfsClient = null;

function report(label) {
  postProgress({ label });
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

// Returns a numeric-only copy of a renderStateStats object, or null when absent/invalid.
function sanitizeRenderStateStats(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return null;
  }
  const out = {};
  for (const key of RENDER_STATE_STATS_CORE_KEYS) {
    out[key] = toNonNegativeInt(raw[key], 0);
  }
  for (const key of RENDER_STATE_STATS_OPTIONAL_KEYS) {
    if (raw[key] !== undefined && raw[key] !== null) {
      out[key] = toNonNegativeInt(raw[key], 0);
    }
  }
  return out;
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

  const realPackDrawCountSrc =
    rawFrameStats &&
    rawFrameStats.realPackDrawCount !== undefined &&
    rawFrameStats.realPackDrawCount !== null
      ? rawFrameStats.realPackDrawCount
      : rawStats &&
          rawStats.realPackDrawCount !== undefined &&
          rawStats.realPackDrawCount !== null
        ? rawStats.realPackDrawCount
        : eventData.realPackDrawCount;
  if (realPackDrawCountSrc !== undefined && realPackDrawCountSrc !== null) {
    frameStats.realPackDrawCount = toNonNegativeInt(realPackDrawCountSrc, 0);
  }

  const fallbackDrawCountSrc =
    rawFrameStats &&
    rawFrameStats.fallbackDrawCount !== undefined &&
    rawFrameStats.fallbackDrawCount !== null
      ? rawFrameStats.fallbackDrawCount
      : rawStats &&
          rawStats.fallbackDrawCount !== undefined &&
          rawStats.fallbackDrawCount !== null
        ? rawStats.fallbackDrawCount
        : eventData.fallbackDrawCount;
  if (fallbackDrawCountSrc !== undefined && fallbackDrawCountSrc !== null) {
    frameStats.fallbackDrawCount = toNonNegativeInt(fallbackDrawCountSrc, 0);
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

  const packetCountSrc =
    rawFrameStats && rawFrameStats.packetCount !== undefined && rawFrameStats.packetCount !== null
      ? rawFrameStats.packetCount
      : rawStats && rawStats.packetCount !== undefined && rawStats.packetCount !== null
        ? rawStats.packetCount
        : eventData.packetCount !== undefined && eventData.packetCount !== null
          ? eventData.packetCount
          : undefined;
  const packetsProcessedSrc =
    rawFrameStats &&
    rawFrameStats.packetsProcessed !== undefined &&
    rawFrameStats.packetsProcessed !== null
      ? rawFrameStats.packetsProcessed
      : rawStats && rawStats.packetsProcessed !== undefined && rawStats.packetsProcessed !== null
        ? rawStats.packetsProcessed
        : eventData.packetsProcessed !== undefined && eventData.packetsProcessed !== null
          ? eventData.packetsProcessed
          : undefined;
  const packetSrc = packetCountSrc !== undefined ? packetCountSrc : packetsProcessedSrc;
  if (packetSrc !== undefined && packetSrc !== null) {
    frameStats.packetCount = toNonNegativeInt(packetSrc, 0);
  }
  if (packetsProcessedSrc !== undefined) {
    frameStats.packetsProcessed = toNonNegativeInt(packetsProcessedSrc, 0);
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

function formatGpuWorkerError(err) {
  if (err && typeof err === 'object') {
    const prefix = typeof err.type === 'string' && err.type.length > 0 ? `[${err.type}] ` : '';
    if (typeof err.message === 'string' && err.message.length > 0) {
      return `${prefix}${err.message}`;
    }
    try {
      return `${prefix}${JSON.stringify(err)}`;
    } catch {
      return `${prefix}${String(err)}`;
    }
  }
  return String(err);
}

/**
 * Inspect a message emitted by `wgpu_worker.js`, sanitize/normalize GPU frame,
 * shader-fallback, missing-shader, and validation-error telemetry, and forward
 * to the main thread via `postMessageFn`. Returns the forwarded payload or `null` when ignored.
 */
function forwardGpuWorkerMessage(eventData, postMessageFn) {
  if (
    eventData &&
    typeof eventData === 'object' &&
    !Array.isArray(eventData) &&
    eventData.type === 'gpu-validation-error'
  ) {
    const rawErr =
      eventData.error && typeof eventData.error === 'object' ? eventData.error : null;
    const rawTs = rawErr
      ? rawErr.timestampMs
      : eventData.timestampMs !== undefined
        ? eventData.timestampMs
        : eventData.timestamp;
    const timestampMs =
      typeof rawTs === 'number' && Number.isFinite(rawTs) ? rawTs : Date.now();
    const cleanErr = rawErr
      ? {
          seq: toNonNegativeInt(rawErr.seq, 0),
          type:
            typeof rawErr.type === 'string' && rawErr.type.length > 0
              ? rawErr.type
              : typeof rawErr.errorType === 'string' && rawErr.errorType.length > 0
                ? rawErr.errorType
                : 'validation',
          message:
            rawErr.message !== undefined && rawErr.message !== null
              ? String(rawErr.message)
              : '',
          timestampMs,
        }
      : {
          seq: toNonNegativeInt(eventData.seq, 0),
          type:
            typeof eventData.errorType === 'string' && eventData.errorType.length > 0
              ? eventData.errorType
              : 'validation',
          message:
            eventData.message !== undefined && eventData.message !== null
              ? String(eventData.message)
              : typeof eventData.error === 'string'
                ? eventData.error
                : '',
          timestampMs,
        };
    const payload = {
      type: 'gpu-validation-error',
      error: cleanErr,
      totalCount: toNonNegativeInt(eventData.totalCount, cleanErr.seq || 1),
    };
    if (typeof eventData.source === 'string' && eventData.source.length > 0) {
      payload.source = eventData.source;
    }
    if (typeof postMessageFn === 'function') {
      try {
        postMessageFn(payload);
      } catch {
        // Telemetry forwarding must never disrupt the GPU worker loop.
      }
    }
    return payload;
  }

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
  const renderStateStats = sanitizeRenderStateStats(
    eventData.renderStateStats ||
      (rawStats && (rawStats.renderStateStats || rawStats.renderState)) ||
      null
  );
  if (renderStateStats) {
    telemetry.renderStateStats = renderStateStats;
    lastRenderStateStats = renderStateStats;
  }

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

function closeBroadcastChannel(ch) {
  if (!ch || typeof ch.close !== 'function') {
    return;
  }
  try {
    ch.close();
  } catch {
    // Ignore BroadcastChannel close errors.
  }
}

/**
 * Send one payload on a short-lived BroadcastChannel and close it immediately in `finally`.
 * Keeps send-only threads (especially Emscripten pthreads running Wasm or `Atomics.wait`) from
 * holding open BroadcastChannel receive queues that never drain.
 */
function postBroadcastOnce(BroadcastChannelCtor, channelName, payload) {
  const ch = tryOpenBroadcastChannel(BroadcastChannelCtor, channelName);
  if (!ch) {
    return false;
  }
  try {
    if (typeof ch.postMessage !== 'function') {
      return false;
    }
    ch.postMessage(payload);
    return true;
  } catch {
    return false;
  } finally {
    closeBroadcastChannel(ch);
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
  if (!baseEnv || typeof baseEnv !== 'object' || !hostEnv) {
    return baseEnv;
  }
  const targetNames = requiredHostNames ? Array.from(requiredHostNames) : HOST_ABI_IMPORT_NAMES;
  if (targetNames.length === 0) {
    return baseEnv;
  }

  // If baseEnv is frozen or has a non-configurable non-writable host_* stub (or write-only
  // non-configurable accessor), a Proxy get-trap cannot return a replacement without throwing.
  let needsClone = false;
  try {
    needsClone = Object.isFrozen(baseEnv);
  } catch {
    needsClone = false;
  }
  if (!needsClone) {
    for (let i = 0; i < targetNames.length; i += 1) {
      let desc = null;
      try {
        desc = Reflect.getOwnPropertyDescriptor(baseEnv, targetNames[i]);
      } catch {
        // If baseEnv's getOwnPropertyDescriptor trap throws, a Proxy get-trap invariant check
        // will also fail when substituting a host ABI function; clone instead.
        needsClone = true;
        break;
      }
      if (
        desc &&
        !desc.configurable &&
        (('value' in desc && !desc.writable) || (!('value' in desc) && desc.get === undefined))
      ) {
        needsClone = true;
        break;
      }
    }
  }

  if (needsClone) {
    const cloned = {};
    try {
      const ownKeys = Reflect.ownKeys(baseEnv);
      for (let i = 0; i < ownKeys.length; i += 1) {
        const key = ownKeys[i];
        try {
          const desc = Reflect.getOwnPropertyDescriptor(baseEnv, key);
          if (desc) {
            if ('value' in desc) {
              Object.defineProperty(cloned, key, {
                value: desc.value,
                writable: true,
                configurable: true,
                enumerable: desc.enumerable !== false,
              });
            } else {
              Object.defineProperty(cloned, key, {
                get: desc.get,
                set: desc.set,
                configurable: true,
                enumerable: desc.enumerable !== false,
              });
            }
          }
        } catch {
          try {
            cloned[key] = baseEnv[key];
          } catch {
            // Ignore exotic keys that fail both descriptor inspection and property read.
          }
        }
      }
    } catch {
      try {
        Object.assign(cloned, baseEnv);
      } catch {
        // Ignore if baseEnv cannot be enumerated via Object.assign.
      }
    }
    for (let i = 0; i < targetNames.length; i += 1) {
      const name = targetNames[i];
      if (typeof hostEnv[name] !== 'function') {
        continue;
      }
      let existing;
      try {
        existing = cloned[name];
      } catch {
        existing = undefined;
      }
      if (typeof existing !== 'function' || existing.stub === true) {
        Object.defineProperty(cloned, name, {
          value: hostEnv[name],
          writable: true,
          configurable: true,
          enumerable: true,
        });
      }
    }
    return cloned;
  }

  // Emscripten MAIN_MODULE=1 generates JS wrappers (`function _host_log(...args) {
  //   if (!wasmImports["host_log"] || wasmImports["host_log"].stub) abort(...);
  //   return wasmImports["host_log"](...args);
  // }`) that read the unwrapped `wasmImports` object directly rather than our Proxy.
  // Overwrite writable (or configurable non-writable) own `.stub === true` properties in-place
  // so those wrappers resolve.
  for (let i = 0; i < targetNames.length; i += 1) {
    const name = targetNames[i];
    if (typeof hostEnv[name] !== 'function') {
      continue;
    }
    try {
      const desc = Reflect.getOwnPropertyDescriptor(baseEnv, name);
      if (desc && 'value' in desc && typeof desc.value === 'function' && desc.value.stub === true) {
        if (desc.writable) {
          baseEnv[name] = hostEnv[name];
        } else if (desc.configurable) {
          Object.defineProperty(baseEnv, name, {
            value: hostEnv[name],
            writable: true,
            configurable: true,
            enumerable: desc.enumerable !== false,
          });
        }
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
 * - Streams `host_log` over short-lived send-only `BroadcastChannel('game-progress')` instances.
 * - Relays `host_wgpu_submit` batches over short-lived send-only `BroadcastChannel('playgta6-wgpu-pthread')` instances to the coordinator worker.
 * - Acquires `httpfsMailboxSab` via `Worker.postMessage` or temporary `BroadcastChannel('playgta6-httpfs-sab')` (closed immediately once SAB arrives) so pthreads can call `host_httpfs_*` without keeping any open BroadcastChannels.
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

  const pthreadReport = (label) => {
    postBroadcastOnce(BroadcastChannelCtor, PROGRESS_CHANNEL_NAME, { label });
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
  let attachedHttpfsSab = null;
  let httpfsChannel = null;

  const closeHttpfsChannel = () => {
    const ch = httpfsChannel;
    httpfsChannel = null;
    if (!ch) {
      return;
    }
    try {
      ch.onmessage = null;
    } catch {
      // Ignore errors clearing BroadcastChannel message handler.
    }
    closeBroadcastChannel(ch);
  };

  const hostImports = createHostImports({
    log: (msg) => pthreadReport(`[wasm] ${msg}`),
    report: () => {},
    wgpuSubmit: (bytes) => {
      if (typeof BroadcastChannelCtor !== 'function') {
        return;
      }
      const copy = bytes.slice();
      postBroadcastOnce(BroadcastChannelCtor, PTHREAD_WGPU_CHANNEL_NAME, {
        type: 'wgpu-submit',
        cmdBuffer: copy.buffer,
      });
    },
    getHttpfsClient: () => pthreadHttpfsClient,
  });

  const attachHttpfsSab = (sab) => {
    if (!sab || !mailboxApi || typeof mailboxApi.HttpfsMailboxClient !== 'function') {
      return false;
    }
    if (attachedHttpfsSab === sab && pthreadHttpfsClient) {
      closeHttpfsChannel();
      return true;
    }
    try {
      pthreadHttpfsClient = new mailboxApi.HttpfsMailboxClient(sab);
      attachedHttpfsSab = sab;
      hostImports.setHttpfsClient(pthreadHttpfsClient);
      closeHttpfsChannel();
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

  const canAttachHttpfs = Boolean(
    mailboxApi && typeof mailboxApi.HttpfsMailboxClient === 'function',
  );
  if (!attachedHttpfsSab && canAttachHttpfs) {
    httpfsChannel = tryOpenBroadcastChannel(
      BroadcastChannelCtor,
      PTHREAD_HTTPFS_SAB_CHANNEL_NAME,
    );
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
    hasHttpfsSab: () => Boolean(pthreadHttpfsClient),
    channels: {
      progress: null,
      wgpu: null,
      get httpfs() {
        return httpfsChannel;
      },
    },
    cleanup() {
      restoreWasmInstance();
      attachedHttpfsSab = null;
      pthreadHttpfsClient = null;
      if (hostImports && typeof hostImports.setHttpfsClient === 'function') {
        hostImports.setHttpfsClient(null);
      }
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
      closeHttpfsChannel();
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

function normalizeIndirectDispatchMode(raw) {
  if (typeof raw === 'string') {
    const lower = raw.trim().toLowerCase();
    if (INDIRECT_DISPATCH_MODES.has(lower)) {
      return lower;
    }
  }
  return 'table';
}

function sanitizeManifestScriptUrl(raw) {
  if (typeof raw !== 'string') {
    return null;
  }
  const trimmed = raw.trim();
  if (!trimmed || trimmed.includes('\0') || trimmed.includes('\\')) {
    return null;
  }
  const withoutQuery = trimmed.split(/[?#]/, 1)[0];
  if (!/\.m?js$/i.test(withoutQuery)) {
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

function extractSideModuleEntryUrl(entry) {
  if (typeof entry === 'string') {
    return sanitizeSideModuleUrl(entry);
  }
  if (entry && typeof entry === 'object' && !Array.isArray(entry)) {
    const candidate =
      entry.url !== undefined
        ? entry.url
        : entry.path !== undefined
          ? entry.path
          : entry.name !== undefined
            ? entry.name
            : entry.file;
    return sanitizeSideModuleUrl(candidate);
  }
  return null;
}

function sanitizeManifestSideModules(rawList) {
  if (!Array.isArray(rawList)) {
    return [];
  }
  const out = [];
  const seen = new Set();
  for (let i = 0; i < rawList.length; i += 1) {
    const clean = extractSideModuleEntryUrl(rawList[i]);
    if (clean && !seen.has(clean)) {
      seen.add(clean);
      out.push(clean);
    }
  }
  return out;
}

function pickFirstDefined(obj, keys) {
  if (!obj || typeof obj !== 'object') {
    return undefined;
  }
  for (let i = 0; i < keys.length; i += 1) {
    const val = obj[keys[i]];
    if (val !== undefined && val !== null && val !== '') {
      return val;
    }
  }
  return undefined;
}

function toOptionalNonNegativeInt(value) {
  if (value === undefined || value === null || value === '' || typeof value === 'boolean') {
    return null;
  }
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n) || n < 0) {
    return null;
  }
  return Math.floor(n);
}

function sanitizeWasmMetrics(rawMetrics, fallbackSideModuleCount = 0, topLevelManifest = null) {
  const src =
    rawMetrics && typeof rawMetrics === 'object' && !Array.isArray(rawMetrics) ? rawMetrics : null;
  const top =
    topLevelManifest && typeof topLevelManifest === 'object' && !Array.isArray(topLevelManifest)
      ? topLevelManifest
      : null;

  const totalTableSize = toOptionalNonNegativeInt(
    pickFirstDefined(src, [
      'totalTableSize',
      'total_table_size',
      'tableSize',
      'table_size',
      'wasmTableSize',
      'wasm_table_size',
    ]) !== undefined
      ? pickFirstDefined(src, [
          'totalTableSize',
          'total_table_size',
          'tableSize',
          'table_size',
          'wasmTableSize',
          'wasm_table_size',
        ])
      : pickFirstDefined(top, [
          'totalTableSize',
          'total_table_size',
          'wasmTableSize',
          'wasm_table_size',
        ]),
  );
  const totalGotFuncImports = toOptionalNonNegativeInt(
    pickFirstDefined(src, [
      'totalGotFuncImports',
      'total_got_func_imports',
      'gotFuncImports',
      'got_func_imports',
    ]) !== undefined
      ? pickFirstDefined(src, [
          'totalGotFuncImports',
          'total_got_func_imports',
          'gotFuncImports',
          'got_func_imports',
        ])
      : pickFirstDefined(top, [
          'totalGotFuncImports',
          'total_got_func_imports',
          'gotFuncImports',
          'got_func_imports',
        ]),
  );
  const explicitSideModuleCount = toOptionalNonNegativeInt(
    pickFirstDefined(src, ['sideModuleCount', 'side_module_count']),
  );

  const optionalSpecs = [
    [
      'totalGotMemImports',
      ['totalGotMemImports', 'total_got_mem_imports', 'gotMemImports', 'got_mem_imports'],
    ],
    ['totalWasmBytes', ['totalWasmBytes', 'total_wasm_bytes', 'wasmBytes', 'wasm_bytes']],
    [
      'mainWasmBytes',
      ['mainWasmBytes', 'main_wasm_bytes', 'runtimeWasmBytes', 'runtime_wasm_bytes'],
    ],
    ['sideModulesTotalBytes', ['sideModulesTotalBytes', 'side_modules_total_bytes']],
    ['shardCount', ['shardCount', 'shard_count']],
    ['dispatchShardCount', ['dispatchShardCount', 'dispatch_shard_count']],
    ['functionCount', ['functionCount', 'function_count']],
    ['exportedFunctionCount', ['exportedFunctionCount', 'exported_function_count']],
  ];

  const extras = {};
  let hasAnyOptional = false;
  if (src) {
    for (let i = 0; i < optionalSpecs.length; i += 1) {
      const [outKey, aliases] = optionalSpecs[i];
      const val = toOptionalNonNegativeInt(pickFirstDefined(src, aliases));
      if (val !== null) {
        extras[outKey] = val;
        hasAnyOptional = true;
      }
    }
  }

  const hasMetricsSource =
    Boolean(src) || totalTableSize !== null || totalGotFuncImports !== null || hasAnyOptional;
  if (!hasMetricsSource) {
    return null;
  }

  const baseCount = toNonNegativeInt(fallbackSideModuleCount, 0);
  const sideModuleCount =
    explicitSideModuleCount !== null ? Math.max(explicitSideModuleCount, baseCount) : baseCount;

  const sanitized = {
    sideModuleCount,
  };
  if (totalTableSize !== null) {
    sanitized.totalTableSize = totalTableSize;
  }
  if (totalGotFuncImports !== null) {
    sanitized.totalGotFuncImports = totalGotFuncImports;
  }
  const extraKeys = Object.keys(extras);
  for (let i = 0; i < extraKeys.length; i += 1) {
    sanitized[extraKeys[i]] = extras[extraKeys[i]];
  }
  return sanitized;
}

/**
 * Parse and sanitize `guest_manifest.json` while preserving backwards compatibility with
 * legacy `table`-dispatch manifests and surfacing optional `indirectDispatch` (`"shard"` | `"table"`)
 * and numeric-only `wasmMetrics` (`totalTableSize`, `totalGotFuncImports`, `sideModuleCount`, ...).
 */
function parseGuestManifest(rawManifest) {
  let manifest = rawManifest;
  if (typeof manifest === 'string') {
    try {
      manifest = JSON.parse(manifest);
    } catch {
      return null;
    }
  }
  if (!manifest || typeof manifest !== 'object' || Array.isArray(manifest)) {
    return null;
  }

  const rawSideList =
    manifest.sideModules !== undefined
      ? manifest.sideModules
      : manifest.side_modules !== undefined
        ? manifest.side_modules
        : manifest.sideModuleUrls !== undefined
          ? manifest.sideModuleUrls
          : manifest.side_module_urls !== undefined
            ? manifest.side_module_urls
            : manifest.dynamicLibraries !== undefined
              ? manifest.dynamicLibraries
              : manifest.dynamic_libraries;
  const sideModules = sanitizeManifestSideModules(rawSideList);

  const explicitSideCount = toOptionalNonNegativeInt(
    pickFirstDefined(manifest, ['sideModuleCount', 'side_module_count']),
  );
  const sideModuleCount =
    explicitSideCount !== null ? Math.max(explicitSideCount, sideModules.length) : sideModules.length;

  const rawMetrics =
    manifest.wasmMetrics !== undefined ? manifest.wasmMetrics : manifest.wasm_metrics;
  const rawDispatch =
    pickFirstDefined(manifest, ['indirectDispatch', 'indirect_dispatch']) !== undefined
      ? pickFirstDefined(manifest, ['indirectDispatch', 'indirect_dispatch'])
      : pickFirstDefined(rawMetrics, ['indirectDispatch', 'indirect_dispatch']);
  const indirectDispatch = normalizeIndirectDispatchMode(rawDispatch);

  const wasmMetrics = sanitizeWasmMetrics(rawMetrics, sideModuleCount, manifest);
  const maxConcurrentPthreadLoads = toOptionalNonNegativeInt(
    pickFirstDefined(manifest, ['maxConcurrentPthreadLoads', 'max_concurrent_pthread_loads']),
  );
  const runtimeScriptUrl = sanitizeManifestScriptUrl(
    pickFirstDefined(manifest, ['runtimeScriptUrl', 'runtimeJs', 'runtime_js']),
  );

  return {
    indirectDispatch,
    sideModules,
    sideModuleCount,
    wasmMetrics,
    maxConcurrentPthreadLoads,
    runtimeScriptUrl,
  };
}

async function loadGuestManifest(options = {}) {
  const url =
    typeof options.url === 'string' && options.url.trim().length > 0
      ? options.url.trim()
      : DEFAULT_GUEST_MANIFEST_RELATIVE;
  const fetchFn =
    typeof options.fetchFn === 'function'
      ? options.fetchFn
      : typeof fetch === 'function'
        ? fetch
        : null;
  const onProgress = typeof options.onProgress === 'function' ? options.onProgress : null;
  const reportFn = typeof options.reportFn === 'function' ? options.reportFn : null;

  if (!fetchFn) {
    return null;
  }

  let parsed = null;
  try {
    const resp = await fetchFn(url, { credentials: 'same-origin' });
    if (!resp || resp.ok === false) {
      return null;
    }
    const raw =
      typeof resp.json === 'function'
        ? await resp.json()
        : typeof resp.text === 'function'
          ? await resp.text()
          : null;
    parsed = parseGuestManifest(raw);
  } catch {
    return null;
  }

  if (!parsed) {
    return null;
  }

  if (onProgress) {
    try {
      onProgress({
        phase: 'manifest',
        indirectDispatch: parsed.indirectDispatch,
        sideModules: parsed.sideModules.slice(),
        sideModuleCount: parsed.sideModuleCount,
        wasmMetrics: parsed.wasmMetrics ? Object.assign({}, parsed.wasmMetrics) : null,
      });
    } catch {
      // Progress hooks must never fail manifest loading.
    }
  }
  if (reportFn) {
    try {
      const tableSuffix =
        parsed.wasmMetrics && parsed.wasmMetrics.totalTableSize !== undefined
          ? `, table=${parsed.wasmMetrics.totalTableSize}`
          : '';
      reportFn(
        `Guest manifest loaded (dispatch=${parsed.indirectDispatch}, sideModules=${parsed.sideModuleCount}${tableSuffix})`,
      );
    } catch {
      // Progress reporting failures are non-fatal.
    }
  }

  return parsed;
}

function createRuntimeLocateFile(runtimeScriptUrl, locateFileBase, options = {}) {
  const reportFn = typeof options.reportFn === 'function' ? options.reportFn : null;
  const onProgress = typeof options.onProgress === 'function' ? options.onProgress : null;
  const indirectDispatch =
    options.indirectDispatch !== undefined
      ? normalizeIndirectDispatchMode(options.indirectDispatch)
      : null;
  const wasmMetrics =
    options.wasmMetrics && typeof options.wasmMetrics === 'object'
      ? Object.assign({}, options.wasmMetrics)
      : null;
  const knownSideModules = new Set();
  const expectedModuleNames = [];
  if (Array.isArray(options.sideModuleNames)) {
    for (let i = 0; i < options.sideModuleNames.length; i += 1) {
      const base = extractWasmBasename(options.sideModuleNames[i]);
      if (base) {
        const lower = base.toLowerCase();
        if (!knownSideModules.has(lower)) {
          knownSideModules.add(lower);
          expectedModuleNames.push(base);
        }
      }
    }
  }
  const explicitExpectedCount = toOptionalNonNegativeInt(options.expectedSideModuleCount);
  const reportedSideModules = new Set();
  const loadedModuleNames = [];

  function getProgress() {
    const loadedCount = reportedSideModules.size;
    const baseExpected = Math.max(
      knownSideModules.size,
      explicitExpectedCount !== null ? explicitExpectedCount : 0,
    );
    const totalCount = Math.max(baseExpected, loadedCount);
    const remainingCount = Math.max(0, totalCount - loadedCount);
    const complete = totalCount === 0 || loadedCount >= totalCount;
    const pct = totalCount > 0 ? Math.round((loadedCount / totalCount) * 10000) / 100 : 100;
    return {
      loadedCount,
      totalCount,
      remainingCount,
      complete,
      pct,
      loadedModules: loadedModuleNames.slice(),
      expectedModules: expectedModuleNames.slice(),
    };
  }

  function locateFile(file) {
    const resolved = resolveSideModuleUrl(file, runtimeScriptUrl, locateFileBase);
    if (!resolved) {
      return '';
    }
    const base = extractWasmBasename(resolved) || extractWasmBasename(file);
    const baseLower = base ? base.toLowerCase() : '';
    if (
      baseLower &&
      (GUEST_SIDE_MODULE_BASENAME_RE.test(baseLower) || knownSideModules.has(baseLower)) &&
      !reportedSideModules.has(baseLower)
    ) {
      reportedSideModules.add(baseLower);
      loadedModuleNames.push(base);
      const snapshot = getProgress();
      if (reportFn) {
        try {
          reportFn(`Loading Wasm side module (${base})...`);
        } catch {
          // Progress channel errors must never break locateFile.
        }
      }
      if (onProgress) {
        try {
          const payload = {
            phase: 'side-module',
            module: base,
            url: resolved,
            loadedCount: snapshot.loadedCount,
            totalCount: snapshot.totalCount,
            remainingCount: snapshot.remainingCount,
            complete: snapshot.complete,
            pct: snapshot.pct,
          };
          if (indirectDispatch) {
            payload.indirectDispatch = indirectDispatch;
          }
          if (wasmMetrics) {
            payload.wasmMetrics = Object.assign({}, wasmMetrics);
          }
          onProgress(payload);
        } catch {
          // Progress callback errors must never break locateFile.
        }
      }
    }
    return resolved;
  }

  locateFile.getProgress = getProgress;
  return locateFile;
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

function isSharedArrayBufferValue(buf) {
  return typeof SharedArrayBuffer !== 'undefined' && buf instanceof SharedArrayBuffer;
}

function isSupportedInputBuffer(buf) {
  if (!buf) {
    return false;
  }
  const validType = buf instanceof ArrayBuffer || isSharedArrayBufferValue(buf);
  return validType && buf.byteLength >= INPUT_STATE_BYTES;
}

function canUseAtomicsOnInt32View(atomics, view, isShared) {
  if (
    !atomics ||
    typeof atomics.load !== 'function' ||
    typeof atomics.store !== 'function' ||
    typeof atomics.add !== 'function'
  ) {
    return false;
  }
  if (isShared) {
    return true;
  }
  try {
    atomics.load(view, 0);
    return true;
  } catch {
    return false;
  }
}

function resolveInputStatePointerGetter(mod, explicitGetter) {
  if (typeof explicitGetter === 'function') {
    return explicitGetter;
  }
  if (!mod || typeof mod !== 'object') {
    return null;
  }
  if (typeof mod._playgta6_get_input_state_ptr === 'function') {
    return () => mod._playgta6_get_input_state_ptr();
  }
  if (mod.asm && typeof mod.asm.playgta6_get_input_state_ptr === 'function') {
    return () => mod.asm.playgta6_get_input_state_ptr();
  }
  if (mod.wasmExports && typeof mod.wasmExports.playgta6_get_input_state_ptr === 'function') {
    return () => mod.wasmExports.playgta6_get_input_state_ptr();
  }
  return null;
}

function makeInactiveInputBridgeHandle() {
  return {
    active: false,
    inputStatePtr: 0,
    syncNow() {
      return false;
    },
    stop() {},
    getStats() {
      return {
        active: false,
        stopped: true,
        inputStatePtr: 0,
        syncCount: 0,
        lastSrcSeq: 0,
        lastDstSeq: 0,
        bufferReacquires: 0,
        waitAsyncWakeups: 0,
        pollTicks: 0,
      };
    },
  };
}

/**
 * Bridge browser `InputBridge` (`inputSab`, 40-byte `playgta6::InputState` seqlock) into the
 * Wasm64 runtime's static linear-memory `InputState` exported via `_playgta6_get_input_state_ptr`.
 *
 * - Degrades cleanly (returns an inactive no-op handle) when `inputSab` is absent or the `.wasm`
 *   build does not export `_playgta6_get_input_state_ptr` (or returns 0 / 0n).
 * - Re-acquires `Int32Array(wasmMemory.buffer, ptr, 10)` automatically whenever linear memory
 *   grows or its backing buffer is replaced.
 * - Uses `Atomics.waitAsync` for instant wakeups when available, paired with a periodic fallback
 *   interval (`pollIntervalMs`, default 16 ms) so writers that omit `Atomics.notify` (such as
 *   `web/input/input_bridge.js`) never miss an edge.
 */
function startInputSabBridge(options = {}) {
  const srcBuf = options && options.inputSab ? options.inputSab : null;
  if (!isSupportedInputBuffer(srcBuf)) {
    return makeInactiveInputBridgeHandle();
  }

  const mod = options && options.module && typeof options.module === 'object' ? options.module : null;
  const ptrGetter = resolveInputStatePointerGetter(mod, options.getInputStatePtr);
  if (!ptrGetter) {
    return makeInactiveInputBridgeHandle();
  }

  let rawPtr;
  try {
    rawPtr = ptrGetter();
  } catch {
    return makeInactiveInputBridgeHandle();
  }
  if (rawPtr === null || rawPtr === undefined || rawPtr === 0 || rawPtr === 0n) {
    return makeInactiveInputBridgeHandle();
  }

  let inputStatePtr = 0;
  try {
    inputStatePtr = ptrToNumber(rawPtr, 'inputStatePtr');
  } catch {
    return makeInactiveInputBridgeHandle();
  }
  if (inputStatePtr <= 0 || (inputStatePtr & 3) !== 0) {
    return makeInactiveInputBridgeHandle();
  }

  const pollIntervalMs =
    typeof options.pollIntervalMs === 'number' &&
    Number.isFinite(options.pollIntervalMs) &&
    options.pollIntervalMs > 0
      ? Math.max(1, Math.floor(options.pollIntervalMs))
      : DEFAULT_INPUT_BRIDGE_POLL_MS;

  const atomics =
    options.atomicsObj !== undefined
      ? options.atomicsObj
      : typeof Atomics === 'object' && Atomics !== null
        ? Atomics
        : null;
  const setIntervalFn =
    options.setIntervalFn !== undefined
      ? options.setIntervalFn
      : typeof setInterval === 'function'
        ? setInterval
        : null;
  const clearIntervalFn =
    options.clearIntervalFn !== undefined
      ? options.clearIntervalFn
      : typeof clearInterval === 'function'
        ? clearInterval
        : null;

  const isSharedSrc = isSharedArrayBufferValue(srcBuf);
  const srcI32 = new Int32Array(srcBuf, 0, INPUT_STATE_WORDS);
  const useSrcAtomics = canUseAtomicsOnInt32View(atomics, srcI32, isSharedSrc);

  const loadSrc = (idx) => (useSrcAtomics ? atomics.load(srcI32, idx) : srcI32[idx]);
  const exchangeSrc = (idx, val) => {
    if (useSrcAtomics && typeof atomics.exchange === 'function') {
      return atomics.exchange(srcI32, idx, val);
    }
    const prev = srcI32[idx];
    srcI32[idx] = val | 0;
    return prev;
  };

  function resolveWasmBuffer() {
    const raw =
      typeof options.getMemory === 'function'
        ? options.getMemory()
        : options.memory ||
          (mod &&
            (mod.wasmMemory ||
              (mod.HEAPU8 && mod.HEAPU8.buffer) ||
              (mod.HEAP32 && mod.HEAP32.buffer) ||
              (mod.asm && mod.asm.memory) ||
              (mod.wasmExports && mod.wasmExports.memory))) ||
          null;
    const buf = resolveBuffer(raw);
    if (!buf || buf.detached || buf.byteLength < inputStatePtr + INPUT_STATE_BYTES) {
      return null;
    }
    return buf;
  }

  let cachedDstBuf = null;
  let cachedDstI32 = null;
  let cachedDstUseAtomics = false;
  let bufferReacquires = 0;

  function acquireDstView() {
    const buf = resolveWasmBuffer();
    if (!buf) {
      cachedDstBuf = null;
      cachedDstI32 = null;
      cachedDstUseAtomics = false;
      return null;
    }
    if (buf !== cachedDstBuf || !cachedDstI32) {
      if (cachedDstBuf !== null) {
        bufferReacquires += 1;
      }
      cachedDstBuf = buf;
      cachedDstI32 = new Int32Array(buf, inputStatePtr, INPUT_STATE_WORDS);
      cachedDstUseAtomics = canUseAtomicsOnInt32View(
        atomics,
        cachedDstI32,
        isSharedArrayBufferValue(buf),
      );
    }
    return cachedDstI32;
  }

  const loadDst = (dst, idx) => (cachedDstUseAtomics ? atomics.load(dst, idx) : dst[idx]);
  const storeDst = (dst, idx, val) => {
    if (cachedDstUseAtomics) {
      atomics.store(dst, idx, val | 0);
    } else {
      dst[idx] = val | 0;
    }
  };
  const addDst = (dst, idx, delta) => {
    if (cachedDstUseAtomics) {
      atomics.add(dst, idx, delta | 0);
    } else {
      dst[idx] = (dst[idx] + delta) | 0;
    }
  };

  // Pre-allocated 9-word payload scratch (buttons, mouse_dx, mouse_dy, 6 x f32 axes/triggers).
  const payloadWords = new Int32Array(INPUT_STATE_WORDS - 1);

  let stopped = false;
  let hasSynced = false;
  let lastSrcSeq = 0;
  let dstSeq = 0;
  let syncCount = 0;
  let waitAsyncWakeups = 0;
  let pollTicks = 0;
  let waitAsyncEnabled =
    options.useWaitAsync !== false &&
    isSharedSrc &&
    Boolean(atomics && typeof atomics.waitAsync === 'function');
  let waitAsyncInFlight = false;
  let intervalId = null;

  function readConsistentSnapshot() {
    for (let attempt = 0; attempt < INPUT_SNAPSHOT_MAX_RETRIES; attempt += 1) {
      const seqBefore = loadSrc(0) >>> 0;
      if ((seqBefore & 1) !== 0) {
        continue;
      }
      for (let w = 1; w < INPUT_STATE_WORDS; w += 1) {
        payloadWords[w - 1] = loadSrc(w);
      }
      const seqAfter = loadSrc(0) >>> 0;
      if (seqBefore === seqAfter) {
        return seqBefore;
      }
    }
    return -1;
  }

  function hasNonZeroPayload() {
    for (let i = 0; i < payloadWords.length; i += 1) {
      if (payloadWords[i] !== 0) {
        return true;
      }
    }
    return false;
  }

  function writeSnapshotToWasm() {
    const dst = acquireDstView();
    if (!dst) {
      return false;
    }

    const existingDstSeq = loadDst(dst, 0) >>> 0;
    const baseSeq = existingDstSeq > dstSeq ? existingDstSeq : dstSeq;
    const oddSeq = ((baseSeq & 1) !== 0 ? baseSeq : baseSeq + 1) >>> 0;
    const nextEvenSeq = (oddSeq + 1) >>> 0;

    // Drain accumulated mouse deltas atomically from inputSab so we neither drop nor double-count.
    const dx = payloadWords[1] !== 0 ? exchangeSrc(2, 0) : 0;
    const dy = payloadWords[2] !== 0 ? exchangeSrc(3, 0) : 0;

    // Seqlock write matching MonotonicSeq & EvenOddSeqlock in runtime/os/game_input.cpp:
    // odd seq -> write 36-byte payload -> even seq.
    storeDst(dst, 0, oddSeq | 0);
    storeDst(dst, 1, payloadWords[0]);
    if (dx !== 0) {
      addDst(dst, 2, dx);
    }
    if (dy !== 0) {
      addDst(dst, 3, dy);
    }
    for (let w = 4; w < INPUT_STATE_WORDS; w += 1) {
      storeDst(dst, w, payloadWords[w - 1]);
    }
    storeDst(dst, 0, nextEvenSeq | 0);

    dstSeq = nextEvenSeq;
    return true;
  }

  function syncNow(force = false) {
    if (stopped) {
      return false;
    }
    const curSeq = loadSrc(0) >>> 0;
    if (!force && hasSynced && (curSeq & 1) === 0 && curSeq === lastSrcSeq) {
      return false;
    }

    const snapSeq = readConsistentSnapshot();
    if (snapSeq < 0) {
      return false;
    }

    if (!hasSynced) {
      if (!force && snapSeq === 0 && !hasNonZeroPayload()) {
        hasSynced = true;
        lastSrcSeq = 0;
        return false;
      }
      if (!writeSnapshotToWasm()) {
        return false;
      }
      hasSynced = true;
      lastSrcSeq = snapSeq;
      syncCount += 1;
      return true;
    }

    if (!force && snapSeq === lastSrcSeq) {
      return false;
    }

    if (!writeSnapshotToWasm()) {
      return false;
    }
    lastSrcSeq = snapSeq;
    syncCount += 1;
    return true;
  }

  function armWaitAsync(syncDepth = 0) {
    if (stopped || !waitAsyncEnabled || waitAsyncInFlight) {
      return;
    }
    const expectedSeq = (hasSynced ? lastSrcSeq : 0) | 0;
    let res;
    try {
      res = atomics.waitAsync(srcI32, 0, expectedSeq, pollIntervalMs);
    } catch {
      waitAsyncEnabled = false;
      return;
    }
    if (!res) {
      return;
    }
    if (res.async === false) {
      if (res.value === 'not-equal') {
        const synced = syncNow(false);
        if (synced && !stopped && syncDepth < 4) {
          armWaitAsync(syncDepth + 1);
        }
      }
      return;
    }
    if (res.value && typeof res.value.then === 'function') {
      waitAsyncInFlight = true;
      res.value.then(
        (outcome) => {
          waitAsyncInFlight = false;
          if (stopped) {
            return;
          }
          if (outcome === 'ok') {
            waitAsyncWakeups += 1;
          }
          syncNow(false);
          armWaitAsync(0);
        },
        () => {
          waitAsyncInFlight = false;
          waitAsyncEnabled = false;
        },
      );
    }
  }

  // Immediate initial sync on attach, then reactive waitAsync + periodic fallback timer.
  syncNow(false);
  armWaitAsync(0);

  if (options.disableFallbackInterval !== true && typeof setIntervalFn === 'function') {
    intervalId = setIntervalFn(() => {
      if (stopped) {
        return;
      }
      pollTicks += 1;
      syncNow(false);
      if (!waitAsyncInFlight && waitAsyncEnabled) {
        armWaitAsync(0);
      }
    }, pollIntervalMs);
  }

  function stop() {
    if (stopped) {
      return;
    }
    stopped = true;
    if (intervalId !== null && typeof clearIntervalFn === 'function') {
      try {
        clearIntervalFn(intervalId);
      } catch {
        // Ignore timer cleanup errors.
      }
      intervalId = null;
    }
    if (waitAsyncInFlight && isSharedSrc && atomics && typeof atomics.notify === 'function') {
      try {
        atomics.notify(srcI32, 0, 1);
      } catch {
        // Ignore notify failures during teardown.
      }
    }
    cachedDstBuf = null;
    cachedDstI32 = null;
    cachedDstUseAtomics = false;
  }

  return {
    active: true,
    inputStatePtr,
    syncNow,
    stop,
    getStats() {
      return {
        active: !stopped,
        stopped,
        inputStatePtr,
        syncCount,
        lastSrcSeq,
        lastDstSeq: dstSeq,
        bufferReacquires,
        waitAsyncWakeups,
        pollTicks,
      };
    },
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

  const rawManifestInput =
    options.guestManifest !== undefined ? options.guestManifest : options.manifest;
  const manifestSummary =
    rawManifestInput !== undefined && rawManifestInput !== null
      ? parseGuestManifest(rawManifestInput)
      : null;

  const manifestRuntimeScriptUrl =
    manifestSummary && manifestSummary.runtimeScriptUrl
      ? resolveSideModuleUrl(
          manifestSummary.runtimeScriptUrl,
          DEFAULT_RUNTIME_SCRIPT_RELATIVE,
          options.locateFileBase,
        )
      : undefined;
  const resolvedRuntimeScriptUrl = resolveRuntimeScriptUrl(
    options.runtimeScriptUrl || manifestRuntimeScriptUrl || undefined,
    options.locationHref,
  );
  const rawDynamicLibraries = sanitizeSideModuleUrls(options.dynamicLibraries);
  const rawSideModuleUrls = sanitizeSideModuleUrls(options.sideModuleUrls);
  const manifestSideModules = manifestSummary ? manifestSummary.sideModules : [];
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
  if (combinedDynamicLibraries.length === 0 && manifestSideModules.length > 0) {
    for (let i = 0; i < manifestSideModules.length; i += 1) {
      addSideModuleEntry(manifestSideModules[i]);
    }
  }
  const resolvedSideModuleUrls = combinedDynamicLibraries.map((entry) =>
    resolveSideModuleUrl(entry, resolvedRuntimeScriptUrl, options.locateFileBase),
  );

  const locateFile = createRuntimeLocateFile(resolvedRuntimeScriptUrl, options.locateFileBase, {
    reportFn: activeReportFn,
    onProgress: typeof options.onProgress === 'function' ? options.onProgress : null,
    sideModuleNames: combinedDynamicLibraries,
    expectedSideModuleCount:
      manifestSummary && combinedDynamicLibraries.length === 0
        ? manifestSummary.sideModuleCount
        : combinedDynamicLibraries.length,
    indirectDispatch: manifestSummary ? manifestSummary.indirectDispatch : undefined,
    wasmMetrics: manifestSummary ? manifestSummary.wasmMetrics : null,
  });

  const preRun = Array.isArray(options.preRun)
    ? options.preRun
    : [
        (mod) => {
          boundModule = mod || null;
          if (typeof globalThis !== 'undefined') {
            globalThis.__playgta6RuntimeModule = boundModule;
          }
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

  const activeInputSab = options.inputSab !== undefined ? options.inputSab : inputSab;
  let activeInputBridge = null;

  const resolveRuntimeMemory = () =>
    options.memory ||
    (boundModule &&
      (boundModule.wasmMemory ||
        (boundModule.HEAPU8 && boundModule.HEAPU8.buffer) ||
        (boundModule.HEAP32 && boundModule.HEAP32.buffer) ||
        (boundModule.asm && boundModule.asm.memory) ||
        (boundModule.wasmExports && boundModule.wasmExports.memory))) ||
    null;

  const stopInputBridge = () => {
    if (activeInputBridge) {
      try {
        activeInputBridge.stop();
      } catch {
        // Ignore input bridge teardown errors.
      }
      activeInputBridge = null;
    }
  };

  const attachInputBridge = (mod) => {
    if (activeInputBridge && activeInputBridge.active) {
      return activeInputBridge;
    }
    const targetMod = mod && typeof mod === 'object' ? mod : boundModule;
    if (!activeInputSab || !targetMod) {
      return null;
    }
    const bridge = startInputSabBridge({
      inputSab: activeInputSab,
      module: targetMod,
      getMemory: () =>
        resolveRuntimeMemory() ||
        (targetMod &&
          (targetMod.wasmMemory ||
            (targetMod.HEAPU8 && targetMod.HEAPU8.buffer) ||
            (targetMod.HEAP32 && targetMod.HEAP32.buffer) ||
            (targetMod.asm && targetMod.asm.memory) ||
            (targetMod.wasmExports && targetMod.wasmExports.memory))) ||
        null,
      pollIntervalMs: options.inputPollIntervalMs,
    });
    if (bridge && bridge.active) {
      activeInputBridge = bridge;
      return bridge;
    }
    return null;
  };

  const userOnRuntimeInitialized =
    typeof options.onRuntimeInitialized === 'function' ? options.onRuntimeInitialized : null;
  const userOnExit = typeof options.onExit === 'function' ? options.onExit : null;

  const onRuntimeInitialized = function onPlayGta6RuntimeInitialized() {
    const mod = this && typeof this === 'object' ? this : boundModule;
    if (mod && !boundModule) {
      boundModule = mod;
    }
    attachInputBridge(mod);
    if (userOnRuntimeInitialized) {
      return userOnRuntimeInitialized.apply(this, arguments);
    }
    return undefined;
  };

  const onExit = function onPlayGta6RuntimeExit() {
    stopInputBridge();
    if (userOnExit) {
      return userOnExit.apply(this, arguments);
    }
    return undefined;
  };

  const userdataApiObj = activeHostImports.userdata || {};
  const config = {
    mainScriptUrlOrBlob: resolvedRuntimeScriptUrl,
    locateFile,
    preRun,
    onRuntimeInitialized,
    onExit,
    env: activeHostImports.env,
    hostImports: activeHostImports,
    httpfsMailboxSab:
      options.httpfsMailboxSab !== undefined ? options.httpfsMailboxSab : httpfsMailboxSab,
    inputSab: activeInputSab,
    attachInputBridge,
    stopInputBridge,
    getInputBridge: () => activeInputBridge,
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
  if (manifestSummary) {
    config.indirectDispatch = manifestSummary.indirectDispatch;
    config.wasmMetrics = manifestSummary.wasmMetrics
      ? Object.assign({}, manifestSummary.wasmMetrics)
      : null;
    config.manifestSummary = {
      indirectDispatch: manifestSummary.indirectDispatch,
      sideModules: manifestSummary.sideModules.slice(),
      sideModuleCount: manifestSummary.sideModuleCount,
      wasmMetrics: manifestSummary.wasmMetrics
        ? Object.assign({}, manifestSummary.wasmMetrics)
        : null,
    };
  }
  if (typeof locateFile.getProgress === 'function') {
    config.getSideModuleProgress = () => locateFile.getProgress();
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

function normalizeMaxConcurrentPthreadLoads(value, fallback = DEFAULT_MAX_CONCURRENT_PTHREAD_LOADS) {
  if (value === undefined || value === null || value === '' || typeof value === 'boolean') {
    return fallback;
  }
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n) || n < 0) {
    return fallback;
  }
  return Math.floor(n);
}

function isPthreadLoadMessage(msg) {
  return Boolean(
    msg &&
      typeof msg === 'object' &&
      (msg.cmd === PTHREAD_CMD_LOAD || msg.cmd === 'CMD_LOAD'),
  );
}

function isPthreadLoadedMessage(msg) {
  return Boolean(
    msg &&
      typeof msg === 'object' &&
      (msg.cmd === PTHREAD_CMD_LOADED || msg.cmd === 'CMD_LOADED'),
  );
}

function isPthreadPostLoadMessage(msg) {
  return Boolean(
    msg &&
      typeof msg === 'object' &&
      (msg.cmd === PTHREAD_CMD_RUN ||
        msg.cmd === 'CMD_RUN' ||
        msg.cmd === PTHREAD_CMD_OBJECT_TRANSFER ||
        msg.cmd === 'CMD_OBJECT_TRANSFER'),
  );
}

function isHttpfsSabControlMessage(msg) {
  return Boolean(msg && typeof msg === 'object' && msg.__playgta6_httpfs_sab);
}

function hookWorkerEventProperty(worker, propName, observer) {
  let protoDesc = null;
  let proto = Object.getPrototypeOf(worker);
  while (proto && !protoDesc) {
    try {
      protoDesc = Object.getOwnPropertyDescriptor(proto, propName) || null;
    } catch {
      protoDesc = null;
    }
    proto = Object.getPrototypeOf(proto);
  }
  let ownDesc = null;
  try {
    ownDesc = Object.getOwnPropertyDescriptor(worker, propName) || null;
  } catch {
    return null;
  }
  if (ownDesc && !ownDesc.configurable) {
    return null;
  }

  const hasProtoAccessor = Boolean(protoDesc && typeof protoDesc.set === 'function');
  let userHandler =
    !hasProtoAccessor && ownDesc && typeof ownDesc.value === 'function' ? ownDesc.value : null;
  let wrappedHandler = null;

  const makeWrapped = (fn) => {
    if (typeof fn !== 'function') {
      return null;
    }
    return function wrappedWorkerEventHandler(ev) {
      observer(ev);
      return fn.call(this, ev);
    };
  };

  try {
    if (hasProtoAccessor) {
      const existing = protoDesc.get ? protoDesc.get.call(worker) : null;
      if (typeof existing === 'function') {
        userHandler = existing;
        wrappedHandler = makeWrapped(existing);
        protoDesc.set.call(worker, wrappedHandler);
      }
      Object.defineProperty(worker, propName, {
        configurable: true,
        enumerable: protoDesc.enumerable !== false,
        get() {
          return userHandler;
        },
        set(fn) {
          userHandler = typeof fn === 'function' ? fn : null;
          wrappedHandler = makeWrapped(userHandler);
          protoDesc.set.call(worker, wrappedHandler);
        },
      });
    } else {
      wrappedHandler = makeWrapped(userHandler);
      Object.defineProperty(worker, propName, {
        configurable: true,
        enumerable: ownDesc ? ownDesc.enumerable !== false : true,
        get() {
          return wrappedHandler;
        },
        set(fn) {
          userHandler = typeof fn === 'function' ? fn : null;
          wrappedHandler = makeWrapped(userHandler);
        },
      });
    }
  } catch {
    return null;
  }

  let unhooked = false;
  return function unhookWorkerEventProperty() {
    if (unhooked) {
      return;
    }
    unhooked = true;
    const current = userHandler;
    userHandler = null;
    wrappedHandler = null;
    try {
      delete worker[propName];
      if (hasProtoAccessor) {
        if (typeof current === 'function') {
          protoDesc.set.call(worker, current);
        }
      } else {
        worker[propName] = current;
      }
    } catch {
      // Ignore unhook errors on frozen or exotic worker mocks.
    }
  };
}

/**
 * Coordinates 52+ concurrent Emscripten `em-pthread` / `em-pthread-<N>` workers:
 * - Delivers `{ __playgta6_httpfs_sab }` immediately before `CMD_LOAD` when available.
 * - Throttles concurrent in-flight `CMD_LOAD` instantiations to `maxConcurrentPthreadLoads`
 *   (0 = unthrottled) so burst spawns do not starve `io_worker.js` / `wgpu_worker.js` or blow V8 memory.
 * - Buffers any `CMD_RUN`, `CMD_OBJECT_TRANSFER`, or interleaved post-load messages (even if posted
 *   before `CMD_LOAD` is dispatched or while `CMD_LOAD` is in flight) and flushes them in strict FIFO
 *   order only AFTER `CMD_LOADED` completes.
 * - Cleans up all worker tracking references on `terminate()` or `'error'` and frees load slots
 *   immediately so queued workers never deadlock or leak memory.
 * - Tracks clean numeric-only telemetry (`pthreadSpawnedCount`, `pthreadActiveCount`,
 *   `pthreadPeakCount`, `pthreadLoadQueueDepth`, `pthreadFailedCount`).
 */
function createPthreadPoolCoordinator(options = {}) {
  let maxConcurrentLoads = normalizeMaxConcurrentPthreadLoads(
    options.maxConcurrentPthreadLoads,
    DEFAULT_MAX_CONCURRENT_PTHREAD_LOADS,
  );
  const getHttpfsMailboxSab =
    typeof options.getHttpfsMailboxSab === 'function'
      ? options.getHttpfsMailboxSab
      : () => options.httpfsMailboxSab || null;
  const onTelemetryChange =
    typeof options.onTelemetryChange === 'function' ? options.onTelemetryChange : null;

  const trackedWorkers = new Map();
  const loadQueue = [];
  let queueHead = 0;
  let queuedCount = 0;
  let activeLoads = 0;
  let draining = false;

  let pthreadSpawnedCount = 0;
  let pthreadActiveCount = 0;
  let pthreadPeakCount = 0;
  let pthreadFailedCount = 0;
  let pthreadLoadedCount = 0;
  let peakConcurrentLoads = 0;

  function getTelemetry() {
    return {
      pthreadSpawnedCount,
      pthreadActiveCount,
      pthreadPeakCount,
      pthreadLoadQueueDepth: queuedCount,
      pthreadFailedCount,
      pthreadLoadedCount,
      pthreadActiveLoads: activeLoads,
      pthreadPeakConcurrentLoads: peakConcurrentLoads,
      maxConcurrentPthreadLoads: maxConcurrentLoads,
    };
  }

  function emitTelemetry() {
    if (!onTelemetryChange) {
      return;
    }
    try {
      onTelemetryChange(getTelemetry());
    } catch {
      // Telemetry callbacks must never interfere with pthread scheduling.
    }
  }

  function compactQueueIfNeeded() {
    if (queuedCount === 0 || queueHead >= loadQueue.length) {
      loadQueue.length = 0;
      queueHead = 0;
      return;
    }
    if (queueHead >= 32 && queueHead * 2 >= loadQueue.length) {
      let writeIdx = 0;
      for (let i = queueHead; i < loadQueue.length; i += 1) {
        const item = loadQueue[i];
        if (item && !item.cleanedUp && item.state === 'queued') {
          loadQueue[writeIdx] = item;
          item.queueIndex = writeIdx;
          writeIdx += 1;
        }
      }
      loadQueue.length = writeIdx;
      queueHead = 0;
    }
  }

  function enqueueLoad(entry) {
    entry.state = 'queued';
    entry.queueIndex = loadQueue.length;
    loadQueue.push(entry);
    queuedCount += 1;
  }

  function removeFromLoadQueue(entry) {
    if (entry.queueIndex >= 0 && entry.queueIndex < loadQueue.length) {
      if (loadQueue[entry.queueIndex] === entry) {
        loadQueue[entry.queueIndex] = null;
        queuedCount = Math.max(0, queuedCount - 1);
      }
    }
    entry.queueIndex = -1;
    compactQueueIfNeeded();
  }

  function detachMessageObservers(entry) {
    if (!entry.messageCleanupHooks || entry.messageCleanupHooks.length === 0) {
      return;
    }
    const hooks = entry.messageCleanupHooks;
    entry.messageCleanupHooks = [];
    for (let i = 0; i < hooks.length; i += 1) {
      try {
        hooks[i]();
      } catch {
        // Ignore listener detach errors.
      }
    }
  }

  function cleanupEntry(entry, finalState) {
    if (!entry || entry.cleanedUp) {
      return;
    }
    entry.cleanedUp = true;

    const prevState = entry.state;
    if (prevState === 'queued') {
      removeFromLoadQueue(entry);
    } else if (prevState === 'loading') {
      activeLoads = Math.max(0, activeLoads - 1);
    }

    entry.state = finalState;
    if (finalState === 'failed') {
      pthreadFailedCount += 1;
    }
    pthreadActiveCount = Math.max(0, pthreadActiveCount - 1);

    entry.pendingMessages.length = 0;
    entry.loadArgs = null;

    detachMessageObservers(entry);
    const hooks = entry.cleanupHooks;
    entry.cleanupHooks = [];
    for (let i = 0; i < hooks.length; i += 1) {
      try {
        hooks[i]();
      } catch {
        // Ignore worker hook restoration errors.
      }
    }

    const worker = entry.worker;
    entry.worker = null;
    if (worker) {
      trackedWorkers.delete(worker);
    }

    emitTelemetry();
    drainLoadQueue();
  }

  function handleWorkerLoaded(entry) {
    if (!entry || entry.cleanedUp) {
      return;
    }
    if (entry.state !== 'loading' && entry.state !== 'queued' && entry.state !== 'spawned') {
      return;
    }
    if (entry.state === 'queued') {
      removeFromLoadQueue(entry);
    } else if (entry.state === 'loading') {
      activeLoads = Math.max(0, activeLoads - 1);
    }

    entry.state = 'loaded';
    entry.loadArgs = null;
    pthreadLoadedCount += 1;
    detachMessageObservers(entry);

    const worker = entry.worker;
    const origPost = entry.origPostMessage;
    const queued = entry.pendingMessages;
    entry.pendingMessages = [];

    if (worker && origPost) {
      for (let i = 0; i < queued.length; i += 1) {
        if (entry.cleanedUp || entry.state !== 'loaded') {
          break;
        }
        try {
          origPost.apply(worker, queued[i]);
        } catch {
          cleanupEntry(entry, 'failed');
          return;
        }
      }
    }

    emitTelemetry();
    drainLoadQueue();
  }

  function drainLoadQueue() {
    if (draining) {
      return;
    }
    draining = true;
    try {
      while (queuedCount > 0 && (maxConcurrentLoads === 0 || activeLoads < maxConcurrentLoads)) {
        let nextEntry = null;
        while (queueHead < loadQueue.length) {
          const candidate = loadQueue[queueHead];
          loadQueue[queueHead] = null;
          queueHead += 1;
          if (candidate && !candidate.cleanedUp && candidate.state === 'queued') {
            nextEntry = candidate;
            break;
          }
        }
        if (!nextEntry) {
          queuedCount = 0;
          compactQueueIfNeeded();
          break;
        }

        nextEntry.queueIndex = -1;
        queuedCount = Math.max(0, queuedCount - 1);
        compactQueueIfNeeded();

        const loadArgs = nextEntry.loadArgs;
        nextEntry.loadArgs = null;
        if (!loadArgs || !nextEntry.worker || !nextEntry.origPostMessage) {
          cleanupEntry(nextEntry, 'failed');
          continue;
        }

        nextEntry.state = 'loading';
        activeLoads += 1;
        if (activeLoads > peakConcurrentLoads) {
          peakConcurrentLoads = activeLoads;
        }
        emitTelemetry();

        try {
          nextEntry.origPostMessage.apply(nextEntry.worker, loadArgs);
        } catch {
          cleanupEntry(nextEntry, 'failed');
        }
      }
    } finally {
      draining = false;
    }
  }

  function registerWorker(worker, meta = {}) {
    if (!worker || typeof worker !== 'object') {
      return worker;
    }
    if (trackedWorkers.has(worker)) {
      return worker;
    }

    const origPostMessage = typeof worker.postMessage === 'function' ? worker.postMessage : null;
    const hasOwnPostMessage = Object.prototype.hasOwnProperty.call(worker, 'postMessage');
    const origTerminate = typeof worker.terminate === 'function' ? worker.terminate : null;
    const hasOwnTerminate = Object.prototype.hasOwnProperty.call(worker, 'terminate');

    const sab =
      meta && meta.httpfsMailboxSab !== undefined ? meta.httpfsMailboxSab : getHttpfsMailboxSab();
    if (sab && origPostMessage) {
      try {
        origPostMessage.call(worker, { __playgta6_httpfs_sab: sab });
      } catch {
        // Non-fatal: BroadcastChannel handshake remains as fallback.
      }
    }

    pthreadSpawnedCount += 1;
    pthreadActiveCount += 1;
    if (pthreadActiveCount > pthreadPeakCount) {
      pthreadPeakCount = pthreadActiveCount;
    }

    const entry = {
      worker,
      name: meta && typeof meta.name === 'string' ? meta.name : EMSCRIPTEN_PTHREAD_WORKER_NAME,
      state: 'spawned',
      cleanedUp: false,
      queueIndex: -1,
      loadArgs: null,
      pendingMessages: [],
      origPostMessage,
      origTerminate,
      messageCleanupHooks: [],
      cleanupHooks: [],
    };
    trackedWorkers.set(worker, entry);

    const onWorkerMessage = (ev) => {
      const data = ev && ev.data !== undefined ? ev.data : ev;
      if (isPthreadLoadedMessage(data)) {
        handleWorkerLoaded(entry);
      }
    };
    const onWorkerError = () => {
      cleanupEntry(entry, 'failed');
    };

    if (typeof worker.addEventListener === 'function') {
      try {
        worker.addEventListener('message', onWorkerMessage);
        entry.messageCleanupHooks.push(() => {
          if (typeof worker.removeEventListener === 'function') {
            worker.removeEventListener('message', onWorkerMessage);
          }
        });
      } catch {
        // Ignore non-standard addEventListener failures.
      }
      try {
        worker.addEventListener('error', onWorkerError);
        entry.cleanupHooks.push(() => {
          if (typeof worker.removeEventListener === 'function') {
            worker.removeEventListener('error', onWorkerError);
          }
        });
      } catch {
        // Ignore non-standard addEventListener failures.
      }
    }

    const unhookOnMessage = hookWorkerEventProperty(worker, 'onmessage', onWorkerMessage);
    if (unhookOnMessage) {
      entry.messageCleanupHooks.push(unhookOnMessage);
    }
    const unhookOnError = hookWorkerEventProperty(worker, 'onerror', onWorkerError);
    if (unhookOnError) {
      entry.cleanupHooks.push(unhookOnError);
    }

    if (origPostMessage) {
      const wrappedPostMessage = function wrappedPthreadPostMessage(msg) {
        const callArgs = Array.prototype.slice.call(arguments);
        if (entry.cleanedUp || entry.state === 'terminated' || entry.state === 'failed') {
          return undefined;
        }
        if (isHttpfsSabControlMessage(msg)) {
          return origPostMessage.apply(worker, callArgs);
        }
        if (entry.state === 'loaded') {
          return origPostMessage.apply(worker, callArgs);
        }
        if (isPthreadLoadMessage(msg)) {
          if (entry.state === 'spawned') {
            entry.loadArgs = callArgs;
            enqueueLoad(entry);
            emitTelemetry();
            drainLoadQueue();
          }
          return undefined;
        }
        if (
          entry.state === 'queued' ||
          entry.state === 'loading' ||
          isPthreadPostLoadMessage(msg)
        ) {
          entry.pendingMessages.push(callArgs);
          return undefined;
        }
        return origPostMessage.apply(worker, callArgs);
      };
      try {
        worker.postMessage = wrappedPostMessage;
        entry.cleanupHooks.push(() => {
          if (worker.postMessage === wrappedPostMessage) {
            if (hasOwnPostMessage) {
              worker.postMessage = origPostMessage;
            } else {
              delete worker.postMessage;
            }
          }
        });
      } catch {
        // Ignore if worker.postMessage is non-writable.
      }
    }

    if (origTerminate) {
      const wrappedTerminate = function wrappedPthreadTerminate() {
        cleanupEntry(entry, 'terminated');
        return origTerminate.apply(worker, arguments);
      };
      try {
        worker.terminate = wrappedTerminate;
        entry.cleanupHooks.push(() => {
          if (worker.terminate === wrappedTerminate) {
            if (hasOwnTerminate) {
              worker.terminate = origTerminate;
            } else {
              delete worker.terminate;
            }
          }
        });
      } catch {
        // Ignore if worker.terminate is non-writable.
      }
    }

    emitTelemetry();
    return worker;
  }

  return {
    registerWorker,
    getTelemetry,
    getStats: getTelemetry,
    getTrackedCount: () => trackedWorkers.size,
    getMaxConcurrentPthreadLoads: () => maxConcurrentLoads,
    setMaxConcurrentPthreadLoads(nextLimit) {
      maxConcurrentLoads = normalizeMaxConcurrentPthreadLoads(nextLimit, maxConcurrentLoads);
      emitTelemetry();
      drainLoadQueue();
      return maxConcurrentLoads;
    },
    notifyWorkerLoaded(worker) {
      const entry = trackedWorkers.get(worker);
      if (entry) {
        handleWorkerLoaded(entry);
      }
    },
    notifyWorkerError(worker) {
      const entry = trackedWorkers.get(worker);
      if (entry) {
        cleanupEntry(entry, 'failed');
      }
    },
    notifyWorkerTerminated(worker) {
      const entry = trackedWorkers.get(worker);
      if (entry) {
        cleanupEntry(entry, 'terminated');
      }
    },
  };
}

/**
 * Wrap a `Worker` constructor so `em-pthread` / `em-pthread-<N>` workers redirect to
 * `getRedirectHref()` and register with `coordinator` for `httpfsMailboxSab` delivery,
 * bounded `CMD_LOAD` throttling, strict `CMD_LOAD`-before-`CMD_RUN`/`CMD_OBJECT_TRANSFER`
 * ordering, cleanup on error/terminate, and numeric pthread telemetry.
 */
function installPthreadWorkerProxy(options = {}) {
  const OriginalWorker = options.WorkerCtor || null;
  if (typeof OriginalWorker !== 'function') {
    return null;
  }
  const coordinator = options.coordinator || createPthreadPoolCoordinator(options);
  const getRedirectHref =
    typeof options.getRedirectHref === 'function'
      ? options.getRedirectHref
      : () => (typeof options.redirectHref === 'string' ? options.redirectHref : '');

  const WrappedWorker = new Proxy(OriginalWorker, {
    construct(target, args, newTarget) {
      const nextArgs = Array.prototype.slice.call(args);
      const workerOpts = nextArgs.length >= 2 ? nextArgs[1] : undefined;
      const isEmPthread = Boolean(
        workerOpts &&
          typeof workerOpts === 'object' &&
          isEmscriptenPthreadWorkerName(workerOpts.name),
      );
      if (isEmPthread) {
        const redirectHref = getRedirectHref();
        if (typeof redirectHref === 'string' && redirectHref.length > 0) {
          nextArgs[0] = redirectHref;
        }
      }
      const targetNew = newTarget === WrappedWorker ? target : newTarget;
      const instance = Reflect.construct(target, nextArgs, targetNew);
      if (isEmPthread && instance) {
        coordinator.registerWorker(instance, { name: workerOpts.name });
      }
      return instance;
    },
  });

  return { WrappedWorker, coordinator };
}

let coordinatorFirstPresentEmitted = false;
let configuredSideModuleUrls = null;
let configuredDynamicLibraries = null;
let configuredGuestManifest = null;
let configuredGuestManifestUrl = null;
let configuredMaxConcurrentPthreadLoads = null;
let activeGuestManifestSummary = null;
let activeSideModuleProgressGetter = null;

const coordinatorPthreadPool = createPthreadPoolCoordinator({
  getHttpfsMailboxSab: () => httpfsMailboxSab,
});

function getPthreadTelemetry() {
  return coordinatorPthreadPool.getTelemetry();
}

function getLoaderStatus() {
  const pthread = coordinatorPthreadPool.getTelemetry();
  const sideModules =
    typeof activeSideModuleProgressGetter === 'function'
      ? activeSideModuleProgressGetter()
      : {
          loadedCount: 0,
          totalCount: activeGuestManifestSummary ? activeGuestManifestSummary.sideModuleCount : 0,
          remainingCount: activeGuestManifestSummary
            ? activeGuestManifestSummary.sideModuleCount
            : 0,
          complete: !activeGuestManifestSummary || activeGuestManifestSummary.sideModuleCount === 0,
          pct:
            !activeGuestManifestSummary || activeGuestManifestSummary.sideModuleCount === 0
              ? 100
              : 0,
          loadedModules: [],
          expectedModules: activeGuestManifestSummary
            ? activeGuestManifestSummary.sideModules.slice()
            : [],
        };
  return {
    pthreadSpawnedCount: pthread.pthreadSpawnedCount,
    pthreadActiveCount: pthread.pthreadActiveCount,
    pthreadPeakCount: pthread.pthreadPeakCount,
    pthreadLoadQueueDepth: pthread.pthreadLoadQueueDepth,
    pthreadFailedCount: pthread.pthreadFailedCount,
    pthreadLoadedCount: pthread.pthreadLoadedCount,
    pthreadActiveLoads: pthread.pthreadActiveLoads,
    pthreadPeakConcurrentLoads: pthread.pthreadPeakConcurrentLoads,
    maxConcurrentPthreadLoads: pthread.maxConcurrentPthreadLoads,
    indirectDispatch: activeGuestManifestSummary
      ? activeGuestManifestSummary.indirectDispatch
      : 'table',
    wasmMetrics:
      activeGuestManifestSummary && activeGuestManifestSummary.wasmMetrics
        ? Object.assign({}, activeGuestManifestSummary.wasmMetrics)
        : null,
    manifest: activeGuestManifestSummary
      ? {
          indirectDispatch: activeGuestManifestSummary.indirectDispatch,
          sideModules: activeGuestManifestSummary.sideModules.slice(),
          sideModuleCount: activeGuestManifestSummary.sideModuleCount,
          wasmMetrics: activeGuestManifestSummary.wasmMetrics
            ? Object.assign({}, activeGuestManifestSummary.wasmMetrics)
            : null,
        }
      : null,
    sideModules,
    firstPresentEmitted: coordinatorFirstPresentEmitted,
    renderStateStats: lastRenderStateStats ? Object.assign({}, lastRenderStateStats) : null,
  };
}

async function tryBootWasmRuntime() {
  try {
    const head = await fetch('./dist/runtime.js', { method: 'HEAD' });
    if (!head.ok) {
      report('No web/dist/runtime.js built yet (running WebGPU command-ring self-test).');
      const pkt = encodeTestFramePacket(0.08, 0.14, 0.24, 1.0);
      gpuWorker.postMessage({ cmdBuffer: pkt }, [pkt]);
      return;
    }

    let manifestSummary = null;
    if (configuredGuestManifest !== null) {
      manifestSummary = parseGuestManifest(configuredGuestManifest);
    } else if (typeof fetch === 'function') {
      manifestSummary = await loadGuestManifest({
        url: configuredGuestManifestUrl || DEFAULT_GUEST_MANIFEST_RELATIVE,
        fetchFn: fetch,
        reportFn: report,
      });
    }
    if (manifestSummary) {
      activeGuestManifestSummary = manifestSummary;
      if (
        configuredMaxConcurrentPthreadLoads === null &&
        manifestSummary.maxConcurrentPthreadLoads !== null
      ) {
        coordinatorPthreadPool.setMaxConcurrentPthreadLoads(
          manifestSummary.maxConcurrentPthreadLoads,
        );
      }
      postProgress({
        metrics: {
          indirectDispatch: manifestSummary.indirectDispatch,
          sideModuleCount: manifestSummary.sideModuleCount,
          sideModuleUrls: manifestSummary.sideModules.slice(),
          wasmMetrics: manifestSummary.wasmMetrics
            ? Object.assign({}, manifestSummary.wasmMetrics)
            : null,
        },
      });
    }

    report('Loading Emscripten Wasm64 runtime (dist/runtime.js)...');
    importScripts('./dist/runtime.js');
    if (typeof self.createRuntimeModule === 'function') {
      const runtimeOptions = {
        onFirstPresent: () => {
          coordinatorFirstPresentEmitted = true;
        },
        onWasmError: (err) => {
          postProgress({ error: `runtime wasm: ${err}` });
        },
      };
      if (manifestSummary) {
        runtimeOptions.guestManifest = manifestSummary;
      }
      if (configuredSideModuleUrls) {
        runtimeOptions.sideModuleUrls = configuredSideModuleUrls;
      }
      if (configuredDynamicLibraries) {
        runtimeOptions.dynamicLibraries = configuredDynamicLibraries;
      }
      const moduleConfig = createRuntimeModuleOptions(runtimeOptions);
      if (typeof moduleConfig.getSideModuleProgress === 'function') {
        activeSideModuleProgressGetter = moduleConfig.getSideModuleProgress;
      }
      if (activeCoordinatorInputBridge) {
        try {
          activeCoordinatorInputBridge.stop();
        } catch {
          // Ignore teardown errors from a previous runtime boot.
        }
        activeCoordinatorInputBridge = null;
      }
      const runtimeMod = await self.createRuntimeModule(moduleConfig);
      if (typeof moduleConfig.attachInputBridge === 'function') {
        const attached = moduleConfig.attachInputBridge(runtimeMod);
        if (attached && attached.active) {
          activeCoordinatorInputBridge = attached;
        }
      } else if (typeof moduleConfig.getInputBridge === 'function') {
        const existing = moduleConfig.getInputBridge();
        if (existing && existing.active) {
          activeCoordinatorInputBridge = existing;
        }
      }
    }
  } catch (err) {
    postProgress({ error: `runtime boot: ${err}` });
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
    const proxyBundle = installPthreadWorkerProxy({
      WorkerCtor: OriginalWorker,
      coordinator: coordinatorPthreadPool,
      getRedirectHref: () =>
        self.location && typeof self.location.href === 'string' ? self.location.href : '',
    });
    const WrappedWorker = proxyBundle ? proxyBundle.WrappedWorker : null;
    if (WrappedWorker) {
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
    if (msg.type === 'request-loader-status' || msg.type === 'request-pthread-telemetry') {
      if (postToMain) {
        const payload = Object.assign({ type: 'loader-status' }, getLoaderStatus(), {
          timestamp: Date.now(),
        });
        if (msg.id !== undefined) {
          payload.id = msg.id;
        }
        try {
          postToMain(payload);
        } catch {
          // Non-fatal status reply failure.
        }
      }
      return;
    }
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
    if (activeCoordinatorInputBridge) {
      try {
        activeCoordinatorInputBridge.stop();
      } catch {
        // Ignore teardown errors when re-initializing workers.
      }
      activeCoordinatorInputBridge = null;
    }
    inputSab = msg.inputSab || null;
    configuredSideModuleUrls = Array.isArray(msg.sideModuleUrls) ? msg.sideModuleUrls : null;
    configuredDynamicLibraries = Array.isArray(msg.dynamicLibraries)
      ? msg.dynamicLibraries
      : null;
    configuredGuestManifest =
      msg.guestManifest !== undefined ? msg.guestManifest : msg.manifest !== undefined ? msg.manifest : null;
    configuredGuestManifestUrl =
      typeof msg.guestManifestUrl === 'string' && msg.guestManifestUrl.trim().length > 0
        ? msg.guestManifestUrl.trim()
        : null;
    configuredMaxConcurrentPthreadLoads = toOptionalNonNegativeInt(msg.maxConcurrentPthreadLoads);
    if (configuredMaxConcurrentPthreadLoads !== null) {
      coordinatorPthreadPool.setMaxConcurrentPthreadLoads(configuredMaxConcurrentPthreadLoads);
    }

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
      else if (e.data.error) postProgress({ error: `io_worker: ${e.data.error}` });
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
      } else if (data.error) {
        postProgress({ error: `wgpu_worker: ${formatGpuWorkerError(data.error)}` });
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
  PROGRESS_CHANNEL_NAME,
  PTHREAD_WGPU_CHANNEL_NAME,
  PTHREAD_HTTPFS_SAB_CHANNEL_NAME,
  PTHREAD_CMD_LOAD,
  PTHREAD_CMD_LOADED,
  PTHREAD_CMD_RUN,
  PTHREAD_CMD_OBJECT_TRANSFER,
  DEFAULT_MAX_CONCURRENT_PTHREAD_LOADS,
  HOST_ABI_IMPORT_NAMES,
  GPU_OPCODE_PRESENT,
  GPU_PACKET_HEADER_BYTES,
  INPUT_STATE_BYTES,
  INPUT_STATE_WORDS,
  DEFAULT_INPUT_BRIDGE_POLL_MS,
  BOOT_INIT_FORWARD_KEYS,
  isEmscriptenPthreadWorkerName,
  isEmscriptenPthreadWorker,
  createPatchedPthreadEnv,
  installPthreadHostImportShim,
  bootstrapEmscriptenPthreadWorker,
  createPthreadPoolCoordinator,
  installPthreadWorkerProxy,
  getPthreadTelemetry,
  getLoaderStatus,
  parseGuestManifest,
  loadGuestManifest,
  inspectGpuPacketBatchForPresent,
  normalizeBootHttpfsPath,
  collectBootBridgeInitFields,
  createWorkerBootInitPayloads,
  forwardGpuWorkerMessage,
  sanitizeRenderStateStats,
  createHostImports,
  startInputSabBridge,
  resolveRuntimeScriptUrl,
  resolveSideModuleUrl,
  sanitizeSideModuleUrls,
  buildRuntimeModuleConfig,
  createRuntimeModuleOptions,
  encodeTestFramePacket,
  postBroadcastOnce,
  ptrToNumber,
};

if (typeof globalThis !== 'undefined') {
  globalThis.PlayGta6Loader = exportedApi;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = exportedApi;
}
})();
