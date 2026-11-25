// Unified browser runtime session controller for playgta6web.
// Orchestrates capability probing, /api/status readiness checks, Wasm64 runtime startup
// with automatic fallback to the WebGPU G5 smoke demo, and lifecycle wiring for
// HUD, InputBridge, AudioEngine, BrowserStorageManager, PerfMonitor, and CrashReporter.
// Zero external dependencies; ES module compatible with Browser and Node.js.

export const SESSION_STATE = Object.freeze({
  IDLE: 'idle',
  PROBING: 'probing',
  STARTING: 'starting',
  RUNNING_WASM: 'running_wasm',
  RUNNING_SMOKE: 'running_smoke',
  STOPPED: 'stopped',
  ERROR: 'error',
});

export const SESSION_MODE = Object.freeze({
  NONE: 'none',
  WASM: 'wasm',
  SMOKE: 'smoke',
});

export const BOOT_MODE = Object.freeze({
  IDLE: 'idle',
  REAL_WASM64: 'real-wasm64',
  SMOKE: 'smoke',
});

export const SESSION_PHASE = Object.freeze({
  BOOT: 'boot',
  START_SCREEN: 'start_screen',
  INTRO_VIDEO: 'intro_video',
  MAIN_MENU: 'main_menu',
  IN_GAME: 'in_game',
});

export const DEFAULT_STATUS_URL = '/api/status';
export const DEFAULT_MAX_VALIDATION_ERRORS = 32;
export const DEFAULT_MAX_TRANSITION_HISTORY = 64;

const MAX_VALIDATION_ERROR_MESSAGE_LENGTH = 512;
const MAX_TRANSITION_REASON_LENGTH = 256;
const MAX_SIDE_MODULE_URLS = 256;
const HOST_UNIX_PATH_PREFIX_RE =
  /^\/(?:Users|home|var|tmp|private|etc|opt|usr|Volumes|mnt|root|Library|Applications|System)(?:\/|$)/i;

const VALID_CAPABILITY_KEYS = Object.freeze([
  'crossOriginIsolated',
  'sharedArrayBuffer',
  'webgpu',
  'wasm64',
  'opfs',
  'gamepad',
  'audioWorklet',
]);

export function detectWasm64(wasmApi) {
  const api =
    arguments.length === 0
      ? typeof WebAssembly !== 'undefined'
        ? WebAssembly
        : null
      : wasmApi;
  if (!api || typeof api.Memory !== 'function') {
    return false;
  }
  // Probe spec-standard `{ address: 'i64' }` (Chrome 133+, Firefox 134+, Node 22+)
  // first, then fall back to legacy proposal `{ index: 'i64' }`. Passing BigInt
  // `initial`/`maximum` ensures non-Wasm64 engines (or mocks) that silently ignore
  // unknown descriptor keys still throw a TypeError on BigInt-to-Number coercion.
  for (const key of ['address', 'index']) {
    try {
      const mem = new api.Memory({ initial: 1n, maximum: 2n, [key]: 'i64' });
      if (!mem || typeof mem !== 'object') {
        continue;
      }
      if (typeof mem.type === 'function') {
        try {
          const info = mem.type();
          if (info && typeof info === 'object') {
            const addr = info.address ?? info.index;
            if (addr !== undefined && addr !== 'i64') {
              continue;
            }
          }
        } catch {
          // Ignore optional type-reflection errors.
        }
      }
      return true;
    } catch {
      // Try next descriptor key or report unsupported below.
    }
  }
  return false;
}

/**
 * Probes standard browser runtime capabilities without throwing in headless or Node environments.
 *
 * @returns {{
 *   crossOriginIsolated: boolean,
 *   sharedArrayBuffer: boolean,
 *   webgpu: boolean,
 *   wasm64: boolean,
 *   opfs: boolean,
 *   gamepad: boolean,
 *   audioWorklet: boolean,
 * }}
 */
export function probeBrowserCapabilities() {
  const g = typeof globalThis !== 'undefined' ? globalThis : {};
  const nav = typeof navigator !== 'undefined' && navigator ? navigator : null;

  const hasAudioContext =
    typeof g.AudioContext === 'function' || typeof g.webkitAudioContext === 'function';
  const hasAudioWorkletNode = typeof g.AudioWorkletNode === 'function';
  const wasmApi =
    'WebAssembly' in g
      ? g.WebAssembly
      : typeof WebAssembly !== 'undefined'
        ? WebAssembly
        : null;

  return {
    crossOriginIsolated: g.crossOriginIsolated === true,
    sharedArrayBuffer: typeof SharedArrayBuffer === 'function',
    webgpu: Boolean(nav && 'gpu' in nav && nav.gpu),
    wasm64: detectWasm64(wasmApi),
    opfs: Boolean(nav && nav.storage && typeof nav.storage.getDirectory === 'function'),
    gamepad: Boolean(nav && typeof nav.getGamepads === 'function'),
    audioWorklet: Boolean(hasAudioContext && hasAudioWorkletNode),
  };
}

function normalizeCapabilities(raw, fallback) {
  const base = fallback || {
    crossOriginIsolated: false,
    sharedArrayBuffer: false,
    webgpu: false,
    wasm64: false,
    opfs: false,
    gamepad: false,
    audioWorklet: false,
  };
  const out = {};
  for (const key of VALID_CAPABILITY_KEYS) {
    if (raw && typeof raw === 'object' && key in raw && raw[key] !== undefined) {
      out[key] = Boolean(raw[key]);
    } else {
      out[key] = Boolean(base[key]);
    }
  }
  return out;
}

function toError(err, fallbackMessage = 'Unknown session error') {
  if (err instanceof Error) {
    return err;
  }
  if (typeof err === 'string' && err.length > 0) {
    return new Error(err);
  }
  if (err && typeof err === 'object' && typeof err.message === 'string') {
    return new Error(err.message);
  }
  return new Error(fallbackMessage);
}

function isNonNegativeInt(value) {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0;
}

function normalizeBootMode(value) {
  if (typeof value !== 'string') {
    return null;
  }
  const normalized = value.trim().toLowerCase();
  if (!normalized) {
    return null;
  }
  if (
    normalized === 'real-wasm64' ||
    normalized === 'real_wasm64' ||
    normalized === 'wasm-guest' ||
    normalized === 'wasm_guest' ||
    normalized === 'wasm64' ||
    normalized === 'wasm'
  ) {
    return BOOT_MODE.REAL_WASM64;
  }
  if (normalized === 'smoke' || normalized === 'smoke-demo' || normalized === 'smoke_demo') {
    return BOOT_MODE.SMOKE;
  }
  if (normalized === 'idle' || normalized === 'none' || normalized === 'pending') {
    return BOOT_MODE.IDLE;
  }
  return null;
}

function isAbsoluteUrl(value) {
  return /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(value) || value.startsWith('//');
}

function sanitizeSideModuleUrl(raw) {
  if (typeof raw !== 'string') {
    return null;
  }
  const trimmed = raw.trim();
  if (
    !trimmed ||
    trimmed.includes('\0') ||
    trimmed.includes('\\') ||
    /[\x00-\x1F\x7F]/.test(trimmed) ||
    /^[A-Za-z]:\//.test(trimmed)
  ) {
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
  if (HOST_UNIX_PATH_PREFIX_RE.test(withoutQuery)) {
    return null;
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
  for (let i = 0; i < rawList.length && out.length < MAX_SIDE_MODULE_URLS; i += 1) {
    const clean = sanitizeSideModuleUrl(rawList[i]);
    if (clean && !seen.has(clean)) {
      seen.add(clean);
      out.push(clean);
    }
  }
  return out;
}

function extractSideModuleMetadata(source) {
  if (!source || typeof source !== 'object') {
    return {
      hasSideModuleUrls: false,
      hasSideModuleCount: false,
      sideModuleUrls: [],
      sideModuleCount: 0,
    };
  }

  const hasRawUrls = Array.isArray(source.sideModuleUrls) || Array.isArray(source.sideModules);
  const rawUrls = Array.isArray(source.sideModuleUrls)
    ? source.sideModuleUrls
    : Array.isArray(source.sideModules)
      ? source.sideModules
      : null;
  const hasRawCount = isNonNegativeInt(source.sideModuleCount);

  if (rawUrls && rawUrls.length > 0) {
    const sanitized = sanitizeSideModuleUrls(rawUrls);
    return {
      hasSideModuleUrls: true,
      hasSideModuleCount: true,
      sideModuleUrls: sanitized,
      sideModuleCount: sanitized.length,
    };
  }

  return {
    hasSideModuleUrls: hasRawUrls,
    hasSideModuleCount: hasRawCount,
    sideModuleUrls: [],
    sideModuleCount: hasRawCount ? source.sideModuleCount : 0,
  };
}

function classifyGpuErrorType(raw) {
  if (typeof raw !== 'string') {
    return null;
  }
  const normalized = raw.trim().toLowerCase();
  if (!normalized || normalized === 'gpu-validation-error') {
    return null;
  }
  if (
    normalized === 'out-of-memory' ||
    normalized === 'out_of_memory' ||
    normalized === 'outofmemory' ||
    normalized === 'oom' ||
    normalized.includes('outofmemory') ||
    normalized.includes('out-of-memory') ||
    normalized.includes('out_of_memory')
  ) {
    return 'out-of-memory';
  }
  if (normalized === 'validation' || normalized.includes('validation')) {
    return 'validation';
  }
  if (normalized === 'internal' || normalized.includes('internal')) {
    return 'internal';
  }
  return null;
}

function normalizeValidationErrorType(errorObj, envelope) {
  if (errorObj && typeof errorObj === 'object') {
    const fromType = classifyGpuErrorType(errorObj.type);
    if (fromType) return fromType;
    const fromErrorType = classifyGpuErrorType(errorObj.errorType || errorObj.kind);
    if (fromErrorType) return fromErrorType;
    const fromName = classifyGpuErrorType(errorObj.name);
    if (fromName) return fromName;
    if (errorObj.constructor && typeof errorObj.constructor.name === 'string') {
      const fromCtor = classifyGpuErrorType(errorObj.constructor.name);
      if (fromCtor) return fromCtor;
    }
  }
  if (envelope && typeof envelope === 'object') {
    const fromEnv = classifyGpuErrorType(
      envelope.errorType || envelope.kind || envelope.gpuErrorType
    );
    if (fromEnv) return fromEnv;
    const fromType = classifyGpuErrorType(envelope.type);
    if (fromType) return fromType;
  }
  if (envelope && typeof envelope === 'object' && envelope.type === 'gpu-validation-error') {
    return 'validation';
  }
  return 'unknown';
}

function sanitizeDiagnosticText(rawMessage, maxLength = MAX_VALIDATION_ERROR_MESSAGE_LENGTH) {
  let msg;
  if (typeof rawMessage === 'string') {
    msg = rawMessage;
  } else if (rawMessage === null || rawMessage === undefined) {
    return '';
  } else {
    msg = String(rawMessage);
  }

  // Normalize line breaks/tabs to spaces and strip control characters.
  msg = msg.replace(/[\r\n\t]+/g, ' ').replace(/[\x00-\x1F\x7F]+/g, '');

  // Strip file:// URLs and host filesystem paths (Windows drive, UNC, and Unix paths).
  msg = msg.replace(/\bfile:\/\/[^\s"'`<>)\]]+/gi, '<path>');
  msg = msg.replace(/\\\\[^\s\\"'`<>)\]]+\\[^\s"'`<>)\]]+/g, '<path>');
  msg = msg.replace(/(["'`])[A-Za-z]:[\\\/][^"'`\r\n]+\1/g, '$1<path>$1');
  msg = msg.replace(/\b[A-Za-z]:[\\\/][^\s"'`<>)\],;]+/g, '<path>');
  msg = msg.replace(
    /(["'`])\/(?:Users|home|var|tmp|private|etc|opt|usr|Volumes|mnt|root|Library|Applications|System)\/[^"'`\r\n]+\1/gi,
    '$1<path>$1'
  );
  msg = msg.replace(
    /(^|[\s"'`(\[=,:])\/(?:[A-Za-z0-9._-]+\/)+[A-Za-z0-9._~+-]+/g,
    '$1<path>'
  );

  // Strip long embedded buffers (data URIs, byte arrays, hex dumps, base64 blobs).
  msg = msg.replace(/\bdata:[^\s;,'"`<>]+;base64,[A-Za-z0-9+/=]{16,}/gi, '<buffer>');
  msg = msg.replace(
    /(?:\b(?:0x)?[0-9a-fA-F]{1,4}\s*,\s*){16,}(?:0x)?[0-9a-fA-F]{1,4}\b/g,
    '<buffer>'
  );
  msg = msg.replace(/(?:\b[0-9a-fA-F]{2}\s+){16,}[0-9a-fA-F]{2}\b/g, '<buffer>');
  msg = msg.replace(/\b(?:0x)?[0-9a-fA-F]{48,}\b/g, '<buffer>');
  msg = msg.replace(
    /(^|[\s"'`(\[=,:])[A-Za-z0-9+/]{64,}={0,2}(?=$|[\s"'`)\]=,.;:])/g,
    '$1<buffer>'
  );

  msg = msg.replace(/\s{2,}/g, ' ').trim();

  if (msg.length > maxLength) {
    msg = `${msg.slice(0, Math.max(0, maxLength - 3))}...`;
  }
  return msg;
}

function isEventTargetLike(target) {
  return Boolean(
    target &&
      typeof target === 'object' &&
      typeof target.addEventListener === 'function' &&
      typeof target.removeEventListener === 'function'
  );
}

function createOfflineServerStatus(errorMessage, httpStatus = null) {
  return {
    ok: false,
    offline: true,
    httpStatus,
    error: errorMessage,
    coopCoep: false,
    dataDirConfigured: false,
    hasManifest: false,
    hasBootset: false,
    hasWasmBinary: false,
    shaderPackCount: 0,
    bootMode: BOOT_MODE.IDLE,
    sideModuleCount: 0,
    sideModuleUrls: [],
  };
}

function normalizeServerStatus(data, httpStatus = 200) {
  const hasWasmBinary = Boolean(data.hasWasmBinary);
  const reportedBootMode = normalizeBootMode(data.bootMode);
  const sideMeta = extractSideModuleMetadata(data);

  return {
    ...data,
    ok: data.ok !== undefined ? Boolean(data.ok) : true,
    offline: false,
    httpStatus,
    error: null,
    coopCoep: Boolean(data.coopCoep),
    dataDirConfigured: Boolean(data.dataDirConfigured),
    hasManifest: Boolean(data.hasManifest),
    hasBootset: Boolean(data.hasBootset),
    hasWasmBinary,
    shaderPackCount: Number.isFinite(data.shaderPackCount) ? data.shaderPackCount : 0,
    bootMode: reportedBootMode || (hasWasmBinary ? BOOT_MODE.REAL_WASM64 : BOOT_MODE.SMOKE),
    sideModuleCount: sideMeta.sideModuleCount,
    sideModuleUrls: [...sideMeta.sideModuleUrls],
  };
}

function evaluateRequirements(capabilities) {
  const missing = [];
  const reasons = [];

  if (!capabilities.crossOriginIsolated) {
    missing.push('crossOriginIsolated');
    reasons.push(
      'Cross-Origin Isolation (COOP/COEP) is missing; serve via server/serve_local.py to enable SharedArrayBuffer and worker threads.'
    );
  }
  if (!capabilities.sharedArrayBuffer) {
    missing.push('sharedArrayBuffer');
    reasons.push(
      'SharedArrayBuffer is unavailable in this environment (requires Cross-Origin Isolation with COOP/COEP headers).'
    );
  }
  if (!capabilities.webgpu) {
    missing.push('webgpu');
    reasons.push('WebGPU is unavailable in this browser (requires a WebGPU-capable browser in a secure context).');
  }

  return { missing, reasons };
}

function createEmptyStateDurations() {
  return {
    [SESSION_STATE.IDLE]: 0,
    [SESSION_STATE.PROBING]: 0,
    [SESSION_STATE.STARTING]: 0,
    [SESSION_STATE.RUNNING_WASM]: 0,
    [SESSION_STATE.RUNNING_SMOKE]: 0,
    [SESSION_STATE.STOPPED]: 0,
    [SESSION_STATE.ERROR]: 0,
  };
}

const SESSION_PHASE_ALIASES = Object.freeze({
  boot: SESSION_PHASE.BOOT,
  booting: SESSION_PHASE.BOOT,
  startup: SESSION_PHASE.BOOT,
  start_screen: SESSION_PHASE.START_SCREEN,
  'start-screen': SESSION_PHASE.START_SCREEN,
  'start screen': SESSION_PHASE.START_SCREEN,
  startscreen: SESSION_PHASE.START_SCREEN,
  title_screen: SESSION_PHASE.START_SCREEN,
  'title-screen': SESSION_PHASE.START_SCREEN,
  'title screen': SESSION_PHASE.START_SCREEN,
  intro_video: SESSION_PHASE.INTRO_VIDEO,
  'intro-video': SESSION_PHASE.INTRO_VIDEO,
  'intro video': SESSION_PHASE.INTRO_VIDEO,
  introvideo: SESSION_PHASE.INTRO_VIDEO,
  intro: SESSION_PHASE.INTRO_VIDEO,
  main_menu: SESSION_PHASE.MAIN_MENU,
  'main-menu': SESSION_PHASE.MAIN_MENU,
  'main menu': SESSION_PHASE.MAIN_MENU,
  mainmenu: SESSION_PHASE.MAIN_MENU,
  menu: SESSION_PHASE.MAIN_MENU,
  in_game: SESSION_PHASE.IN_GAME,
  'in-game': SESSION_PHASE.IN_GAME,
  'in game': SESSION_PHASE.IN_GAME,
  ingame: SESSION_PHASE.IN_GAME,
  gameplay: SESSION_PHASE.IN_GAME,
});

function createEmptyPhaseDurations() {
  return {
    [SESSION_PHASE.BOOT]: 0,
    [SESSION_PHASE.START_SCREEN]: 0,
    [SESSION_PHASE.INTRO_VIDEO]: 0,
    [SESSION_PHASE.MAIN_MENU]: 0,
    [SESSION_PHASE.IN_GAME]: 0,
  };
}

function normalizeSessionPhase(raw) {
  if (typeof raw !== 'string') {
    return null;
  }
  const normalized = raw.trim().toLowerCase().replace(/\s+/g, ' ');
  if (!normalized || /[\x00-\x1F\x7F\\\/]/.test(normalized)) {
    return null;
  }
  return Object.prototype.hasOwnProperty.call(SESSION_PHASE_ALIASES, normalized)
    ? SESSION_PHASE_ALIASES[normalized]
    : null;
}

function sanitizeTransitionReason(rawReason) {
  if (typeof rawReason !== 'string') {
    return null;
  }
  const clean = sanitizeDiagnosticText(rawReason, MAX_TRANSITION_REASON_LENGTH);
  return clean.length > 0 ? clean : null;
}

function defaultFetchImpl() {
  if (typeof globalThis !== 'undefined' && typeof globalThis.fetch === 'function') {
    return globalThis.fetch.bind(globalThis);
  }
  return null;
}

export class RuntimeSessionController {
  /**
   * @param {object} [options]
   * @param {typeof fetch | null} [options.fetchImpl]
   * @param {string} [options.statusUrl='/api/status']
   * @param {object|null} [options.hud=null]
   * @param {object|null} [options.inputBridge=null]
   * @param {object|null} [options.audioEngine=null]
   * @param {object|null} [options.storageManager=null]
   * @param {object|null} [options.perfMonitor=null]
   * @param {object|Function|null} [options.crashReporter=null]
   * @param {object|null} [options.bootOrchestrator=null]
   * @param {object|null} [options.orchestrator=null]
   * @param {Function|null} [options.startWasmRuntime=null]
   * @param {Function|null} [options.startSmokeDemo=null]
   * @param {Function|null} [options.capabilityProbe=null]
   * @param {() => number} [options.now]
   * @param {() => number} [options.nowFn]
   * @param {number} [options.maxTransitionHistory=64]
   * @param {number} [options.hudIntervalMs]
   * @param {EventTarget} [options.audioGestureTarget]
   * @param {Function|null} [options.onDeviceLost=null]
   * @param {Function|null} [options.onDeviceRecovered=null]
   * @param {Function|null} [options.onMemoryReclaim=null]
   */
  constructor(options = {}) {
    if (options !== undefined && (options === null || typeof options !== 'object')) {
      throw new TypeError('RuntimeSessionController options must be an object');
    }

    const {
      fetchImpl = defaultFetchImpl(),
      statusUrl = DEFAULT_STATUS_URL,
      hud = null,
      inputBridge = null,
      audioEngine = null,
      storageManager = null,
      perfMonitor = null,
      crashReporter = null,
      bootOrchestrator = null,
      orchestrator = null,
      startWasmRuntime = null,
      startSmokeDemo = null,
      capabilityProbe = null,
      now,
      nowFn,
      hudIntervalMs,
      audioGestureTarget,
      wgpuWorker = null,
      maxValidationErrors = DEFAULT_MAX_VALIDATION_ERRORS,
      maxTransitionHistory = DEFAULT_MAX_TRANSITION_HISTORY,
      onStatusUpdate = null,
      onTelemetry = null,
      onValidationError = null,
      onSplashFrame = null,
      onDeviceLost = null,
      onDeviceRecovered = null,
      onMemoryReclaim = null,
    } = options;

    if (fetchImpl !== null && fetchImpl !== undefined && typeof fetchImpl !== 'function') {
      throw new TypeError('fetchImpl must be a function or null');
    }
    if (typeof statusUrl !== 'string' || statusUrl.length === 0) {
      throw new TypeError('statusUrl must be a non-empty string');
    }
    if (now !== undefined && typeof now !== 'function') {
      throw new TypeError('now must be a function returning milliseconds');
    }
    if (nowFn !== undefined && typeof nowFn !== 'function') {
      throw new TypeError('nowFn must be a function returning milliseconds');
    }
    const resolvedNow = now !== undefined ? now : nowFn !== undefined ? nowFn : () => Date.now();
    if (bootOrchestrator !== null && typeof bootOrchestrator !== 'object') {
      throw new TypeError('bootOrchestrator must be an object or null');
    }
    if (orchestrator !== null && typeof orchestrator !== 'object') {
      throw new TypeError('orchestrator must be an object or null');
    }
    if (startWasmRuntime !== null && typeof startWasmRuntime !== 'function') {
      throw new TypeError('startWasmRuntime must be a function or null');
    }
    if (startSmokeDemo !== null && typeof startSmokeDemo !== 'function') {
      throw new TypeError('startSmokeDemo must be a function or null');
    }
    if (capabilityProbe !== null && typeof capabilityProbe !== 'function') {
      throw new TypeError('capabilityProbe must be a function or null');
    }
    if (wgpuWorker !== null && !isEventTargetLike(wgpuWorker)) {
      throw new TypeError('wgpuWorker must implement addEventListener/removeEventListener or be null');
    }
    if (!Number.isInteger(maxValidationErrors) || maxValidationErrors < 1) {
      throw new RangeError(
        `maxValidationErrors must be a positive integer, got ${maxValidationErrors}`
      );
    }
    if (!Number.isInteger(maxTransitionHistory) || maxTransitionHistory < 1) {
      throw new RangeError(
        `maxTransitionHistory must be a positive integer, got ${maxTransitionHistory}`
      );
    }
    if (onStatusUpdate !== null && typeof onStatusUpdate !== 'function') {
      throw new TypeError('onStatusUpdate must be a function or null');
    }
    if (onTelemetry !== null && typeof onTelemetry !== 'function') {
      throw new TypeError('onTelemetry must be a function or null');
    }
    if (onValidationError !== null && typeof onValidationError !== 'function') {
      throw new TypeError('onValidationError must be a function or null');
    }
    if (onSplashFrame !== null && typeof onSplashFrame !== 'function') {
      throw new TypeError('onSplashFrame must be a function or null');
    }
    if (onDeviceLost !== null && typeof onDeviceLost !== 'function') {
      throw new TypeError('onDeviceLost must be a function or null');
    }
    if (onDeviceRecovered !== null && typeof onDeviceRecovered !== 'function') {
      throw new TypeError('onDeviceRecovered must be a function or null');
    }
    if (onMemoryReclaim !== null && typeof onMemoryReclaim !== 'function') {
      throw new TypeError('onMemoryReclaim must be a function or null');
    }

    this._fetchImpl = fetchImpl || null;
    this._statusUrl = statusUrl;
    this._hud = hud || null;
    this._inputBridge = inputBridge || null;
    this._audioEngine = audioEngine || null;
    this._storageManager = storageManager || null;
    this._perfMonitor = perfMonitor || null;
    this._crashReporter = crashReporter || null;
    this._bootOrchestrator = bootOrchestrator || orchestrator || null;
    this._startWasmRuntime = startWasmRuntime || null;
    this._startSmokeDemo = startSmokeDemo || null;
    this._capabilityProbe = capabilityProbe || null;
    this._now = resolvedNow;
    this._hudIntervalMs = hudIntervalMs;
    this._audioGestureTarget = audioGestureTarget;
    this._wgpuWorker = wgpuWorker || null;
    this._maxValidationErrors = maxValidationErrors;
    this._maxTransitionHistory = maxTransitionHistory;

    this._state = SESSION_STATE.IDLE;
    this._mode = SESSION_MODE.NONE;
    this._bootMode = BOOT_MODE.IDLE;
    this._sideModuleCount = 0;
    this._sideModuleUrls = [];
    this._capabilities = null;
    this._serverStatus = null;
    this._storageSummary = null;
    this._fallbackReason = null;
    this._diagnosticReasons = [];
    this._lastError = null;
    this._startedAtMs = null;

    this._transitionHistoryBuf = new Array(maxTransitionHistory);
    this._transitionHistoryHead = 0;
    this._transitionHistorySize = 0;
    this._transitionSeq = 0;
    this._transitionEvictedCount = 0;
    this._lastTransitionReason = null;
    this._restartCount = 0;
    this._coalescedRestarts = 0;
    this._recoveryCount = 0;
    this._deviceLossCount = 0;
    this._deviceRecoveryCount = 0;
    this._deviceLost = false;
    this._lastDeviceLossReason = null;
    this._lastDeviceLossAtMs = null;
    this._lastDeviceRecoveredAtMs = null;
    this._cleanShutdownCount = 0;
    this._preloadBufferReleaseCount = 0;
    this._stateDurationsMs = createEmptyStateDurations();
    this._stateEnteredAtMs = this._readNowMs();
    this._currentPhase = null;
    this._phaseEnteredAtMs = null;
    this._phaseDurationsMs = createEmptyPhaseDurations();
    this._phaseTransitionCount = 0;
    this._reclaimRuns = 0;
    this._reclaimFailures = 0;
    this._lastReclaimReason = null;
    this._lastReclaimAtMs = null;
    this._reclaimPromise = null;
    this._executingReclaimHook = false;

    this._validationErrorCount = 0;
    this._validationErrorSeq = 0;
    this._recentValidationErrors = [];
    this._splashFramesPresented = 0;
    this._splashFrameErrorCount = 0;
    this._firstSplashPresented = false;
    this._lastSplashFrame = null;

    this._activeHandle = null;
    this._perfDetach = null;
    this._audioGestureCleanup = null;
    this._inputStarted = false;
    this._attachedWorkers = new Map();

    this._destroyed = false;
    this._sessionToken = 0;
    this._externalStopToken = 0;
    this._startPromise = null;
    this._stopPromise = null;
    this._restartPromise = null;
    this._destroyPromise = null;
    this._stateListeners = new Set();
    this._statusListeners = new Set();
    this._telemetryListeners = new Set();
    this._validationErrorListeners = new Set();
    this._splashListeners = new Set();
    this._deviceLostListeners = new Set();
    this._deviceRecoveredListeners = new Set();
    this._reclaimHooks = new Set();
    this._sessionReclaimUnregisters = new Set();

    if (onStatusUpdate) {
      this._statusListeners.add(onStatusUpdate);
    }
    if (onTelemetry) {
      this._telemetryListeners.add(onTelemetry);
    }
    if (onValidationError) {
      this._validationErrorListeners.add(onValidationError);
    }
    if (onSplashFrame) {
      this._splashListeners.add(onSplashFrame);
    }
    if (onDeviceLost) {
      this._deviceLostListeners.add(onDeviceLost);
    }
    if (onDeviceRecovered) {
      this._deviceRecoveredListeners.add(onDeviceRecovered);
    }
    if (onMemoryReclaim) {
      this._reclaimHooks.add(onMemoryReclaim);
    }
  }

  // --- Observability & Diagnostics ---

  getState() {
    return this._state;
  }

  getMode() {
    return this._mode;
  }

  getBootMode() {
    return this._bootMode;
  }

  getSideModuleMetadata() {
    return {
      sideModuleCount: this._sideModuleCount,
      sideModuleUrls: [...this._sideModuleUrls],
    };
  }

  getValidationErrorCount() {
    return this._validationErrorCount;
  }

  getValidationErrors() {
    return this._recentValidationErrors.map((entry) => ({ ...entry }));
  }

  clearValidationErrors() {
    this._recentValidationErrors.length = 0;
  }

  getSplashStats() {
    return {
      framesPresented: this._splashFramesPresented,
      errorCount: this._splashFrameErrorCount,
      firstPresented: this._firstSplashPresented,
      lastFrame: this._lastSplashFrame ? { ...this._lastSplashFrame } : null,
    };
  }

  getTransitionHistory() {
    const out = new Array(this._transitionHistorySize);
    for (let i = 0; i < this._transitionHistorySize; i += 1) {
      const idx = (this._transitionHistoryHead + i) % this._maxTransitionHistory;
      out[i] = { ...this._transitionHistoryBuf[idx] };
    }
    return out;
  }

  clearTransitionHistory() {
    this._transitionHistoryBuf.fill(undefined);
    this._transitionHistoryHead = 0;
    this._transitionHistorySize = 0;
  }

  resetTelemetry() {
    this._transitionHistoryBuf.fill(undefined);
    this._transitionHistoryHead = 0;
    this._transitionHistorySize = 0;
    this._transitionSeq = 0;
    this._transitionEvictedCount = 0;
    this._lastTransitionReason = null;
    this._restartCount = 0;
    this._coalescedRestarts = 0;
    this._recoveryCount = 0;
    this._deviceLossCount = 0;
    this._deviceRecoveryCount = 0;
    this._deviceLost = false;
    this._lastDeviceLossReason = null;
    this._lastDeviceLossAtMs = null;
    this._lastDeviceRecoveredAtMs = null;
    this._cleanShutdownCount = 0;
    this._preloadBufferReleaseCount = 0;
    const nowMs = this._readNowMs();
    this._stateDurationsMs = createEmptyStateDurations();
    this._stateEnteredAtMs = nowMs;
    this._phaseDurationsMs = createEmptyPhaseDurations();
    this._phaseTransitionCount = 0;
    if (this._currentPhase !== null) {
      this._phaseEnteredAtMs = nowMs;
    } else {
      this._phaseEnteredAtMs = null;
    }
    this._reclaimRuns = 0;
    this._reclaimFailures = 0;
    this._lastReclaimReason = null;
    this._lastReclaimAtMs = null;
  }

  isDestroyed() {
    return this._destroyed;
  }

  getCurrentPhase() {
    return this._currentPhase;
  }

  getPhaseDurations(nowMs) {
    const durations = { ...this._phaseDurationsMs };
    if (this._currentPhase !== null && this._phaseEnteredAtMs !== null) {
      const currentNow = this._resolveNowMs(nowMs);
      const activeElapsedMs = Math.max(0, currentNow - this._phaseEnteredAtMs);
      durations[this._currentPhase] = (durations[this._currentPhase] || 0) + activeElapsedMs;
    }
    return durations;
  }

  getStats(nowMs) {
    const resolvedNow = this._resolveNowMs(nowMs);
    const uptimeMs = this._computeUptimeMs(resolvedNow);
    return {
      state: this._state,
      mode: this._mode,
      bootMode: this._bootMode,
      currentPhase: this._currentPhase,
      phaseDurationsMs: this.getPhaseDurations(resolvedNow),
      phaseTransitionCount: this._phaseTransitionCount,
      destroyed: this._destroyed,
      restartInFlight: Boolean(this._restartPromise),
      uptimeMs,
      sessionUptimeMs: uptimeMs,
      restartCount: this._restartCount,
      coalescedRestarts: this._coalescedRestarts,
      recoveryCount: this._recoveryCount,
      deviceLost: this._deviceLost,
      deviceLossCount: this._deviceLossCount,
      deviceRecoveryCount: this._deviceRecoveryCount,
      lastDeviceLossReason: this._lastDeviceLossReason,
      lastDeviceLossAtMs: this._lastDeviceLossAtMs,
      lastDeviceRecoveredAtMs: this._lastDeviceRecoveredAtMs,
      cleanShutdownCount: this._cleanShutdownCount,
      preloadBufferReleaseCount: this._preloadBufferReleaseCount,
      reclaimRuns: this._reclaimRuns,
      reclaimFailures: this._reclaimFailures,
      reclaimInFlight: Boolean(this._reclaimPromise),
      reclaimHookCount: this._reclaimHooks.size,
      lastReclaimReason: this._lastReclaimReason,
      lastReclaimAtMs: this._lastReclaimAtMs,
      validationErrorCount: this._validationErrorCount,
      splashFramesPresented: this._splashFramesPresented,
      splashFrameErrorCount: this._splashFrameErrorCount,
      transitionCount: this._transitionSeq,
      attachedWorkerCount: this._attachedWorkers.size,
      activeListenerCount:
        this._stateListeners.size +
        this._statusListeners.size +
        this._telemetryListeners.size +
        this._validationErrorListeners.size +
        this._splashListeners.size +
        this._deviceLostListeners.size +
        this._deviceRecoveredListeners.size,
    };
  }

  getSessionUptimeMs(nowMs) {
    return this._computeUptimeMs(nowMs);
  }

  getStateDurations(nowMs) {
    const durations = { ...this._stateDurationsMs };
    const currentNow = this._resolveNowMs(nowMs);
    const activeElapsedMs = Math.max(0, currentNow - this._stateEnteredAtMs);
    durations[this._state] = (durations[this._state] || 0) + activeElapsedMs;
    return durations;
  }

  getRestartCount() {
    return this._restartCount;
  }

  getLastTransitionReason() {
    return this._lastTransitionReason;
  }

  /**
   * Registers a state transition listener.
   *
   * @param {(event: { state: string, previousState: string, mode: string, bootMode: string, sideModuleCount: number, sideModuleUrls: string[], timestampMs: number }) => void} listener
   * @returns {() => void} Idempotent unsubscribe function.
   */
  onStateChange(listener) {
    if (typeof listener !== 'function') {
      throw new TypeError('onStateChange requires a listener function');
    }
    this._stateListeners.add(listener);
    let active = true;
    return () => {
      if (!active) return;
      active = false;
      this._stateListeners.delete(listener);
    };
  }

  /**
   * Registers a boot/split-module status listener notified on /api/status probes,
   * boot lifecycle updates, and session state changes.
   *
   * @param {(status: object) => void} listener
   * @returns {() => void} Idempotent unsubscribe function.
   */
  onStatusUpdate(listener) {
    if (typeof listener !== 'function') {
      throw new TypeError('onStatusUpdate requires a listener function');
    }
    this._statusListeners.add(listener);
    let active = true;
    return () => {
      if (!active) return;
      active = false;
      this._statusListeners.delete(listener);
    };
  }

  /**
   * Registers a live WebGPU validation error observer (`gpu-validation-error`).
   *
   * @param {(event: object) => void} listener
   * @returns {() => void} Idempotent unsubscribe function.
   */
  onValidationError(listener) {
    if (typeof listener !== 'function') {
      throw new TypeError('onValidationError requires a listener function');
    }
    this._validationErrorListeners.add(listener);
    let active = true;
    return () => {
      if (!active) return;
      active = false;
      this._validationErrorListeners.delete(listener);
    };
  }

  /**
   * Registers a splash/Bink frame presentation observer (`splash-frame-presented`).
   *
   * @param {(event: object) => void} listener
   * @returns {() => void} Idempotent unsubscribe function.
   */
  onSplashFrame(listener) {
    if (typeof listener !== 'function') {
      throw new TypeError('onSplashFrame requires a listener function');
    }
    this._splashListeners.add(listener);
    let active = true;
    return () => {
      if (!active) return;
      active = false;
      this._splashListeners.delete(listener);
    };
  }

  /**
   * Registers a unified telemetry/diagnostic listener notified on `gpu-validation-error`,
   * `splash-frame-presented`, and `boot-status` events.
   *
   * @param {(event: object) => void} listener
   * @returns {() => void} Idempotent unsubscribe function.
   */
  onTelemetry(listener) {
    if (typeof listener !== 'function') {
      throw new TypeError('onTelemetry requires a listener function');
    }
    this._telemetryListeners.add(listener);
    let active = true;
    return () => {
      if (!active) return;
      active = false;
      this._telemetryListeners.delete(listener);
    };
  }

  /**
   * Registers a WebGPU device-lost notification observer (`gpu-device-lost`).
   *
   * @param {(event: object) => void} listener
   * @returns {() => void} Idempotent unsubscribe function.
   */
  onDeviceLost(listener) {
    if (typeof listener !== 'function') {
      throw new TypeError('onDeviceLost requires a listener function');
    }
    this._deviceLostListeners.add(listener);
    let active = true;
    return () => {
      if (!active) return;
      active = false;
      this._deviceLostListeners.delete(listener);
    };
  }

  /**
   * Registers a WebGPU device-recovered notification observer (`gpu-device-recovered`).
   *
   * @param {(event: object) => void} listener
   * @returns {() => void} Idempotent unsubscribe function.
   */
  onDeviceRecovered(listener) {
    if (typeof listener !== 'function') {
      throw new TypeError('onDeviceRecovered requires a listener function');
    }
    this._deviceRecoveredListeners.add(listener);
    let active = true;
    return () => {
      if (!active) return;
      active = false;
      this._deviceRecoveredListeners.delete(listener);
    };
  }

  /**
   * Registers a soft memory-pressure reclaim callback (e.g. RT pool oversized reclaim,
   * staging slab trim, or OPFS quota eviction) invoked by `requestMemoryReclaim(reason)`
   * and automatically before full `restart()`.
   *
   * @param {(context: { reason: string, state: string, mode: string, bootMode: string, currentPhase: string | null, timestampMs: number }) => (void | Promise<void>)} hook
   * @returns {() => void} Idempotent unregister function.
   */
  registerReclaimHook(hook) {
    if (typeof hook !== 'function') {
      throw new TypeError('registerReclaimHook requires a hook function');
    }
    if (this._destroyed) {
      return () => {};
    }
    this._reclaimHooks.add(hook);
    let active = true;
    return () => {
      if (!active) return;
      active = false;
      this._reclaimHooks.delete(hook);
    };
  }

  /**
   * Alias for `registerReclaimHook(hook)` matching the `on*` observer convention.
   *
   * @param {Function} hook
   * @returns {() => void} Idempotent unregister function.
   */
  onMemoryReclaim(hook) {
    if (typeof hook !== 'function') {
      throw new TypeError('onMemoryReclaim requires a hook function');
    }
    return this.registerReclaimHook(hook);
  }

  _cloneServerStatus() {
    if (!this._serverStatus) {
      return null;
    }
    return {
      ...this._serverStatus,
      sideModuleUrls: Array.isArray(this._serverStatus.sideModuleUrls)
        ? [...this._serverStatus.sideModuleUrls]
        : [],
    };
  }

  _readNowMs() {
    const value = this._now();
    return typeof value === 'number' && Number.isFinite(value) ? value : Date.now();
  }

  _resolveNowMs(nowMs) {
    if (typeof nowMs === 'number' && Number.isFinite(nowMs)) {
      return nowMs;
    }
    return this._readNowMs();
  }

  _computeUptimeMs(nowMs) {
    const isRunning =
      this._state === SESSION_STATE.RUNNING_WASM || this._state === SESSION_STATE.RUNNING_SMOKE;
    if (!isRunning || this._startedAtMs === null) {
      return 0;
    }
    const currentNow = this._resolveNowMs(nowMs);
    return Math.max(0, currentNow - this._startedAtMs);
  }

  getSnapshot(nowMs) {
    const resolvedNow = this._resolveNowMs(nowMs);
    const uptimeMs = this._computeUptimeMs(resolvedNow);
    return {
      state: this._state,
      mode: this._mode,
      bootMode: this._bootMode,
      currentPhase: this._currentPhase,
      phaseDurationsMs: this.getPhaseDurations(resolvedNow),
      phaseTransitionCount: this._phaseTransitionCount,
      sideModuleCount: this._sideModuleCount,
      sideModuleUrls: [...this._sideModuleUrls],
      validationErrorCount: this._validationErrorCount,
      validationErrorHistorySize: this._recentValidationErrors.length,
      recentValidationErrors: this._recentValidationErrors.map((entry) => ({ ...entry })),
      splashFramesPresented: this._splashFramesPresented,
      splashFrameErrorCount: this._splashFrameErrorCount,
      firstSplashPresented: this._firstSplashPresented,
      lastSplashFrame: this._lastSplashFrame ? { ...this._lastSplashFrame } : null,
      capabilities: this._capabilities ? { ...this._capabilities } : null,
      serverStatus: this._cloneServerStatus(),
      storageSummary: this._storageSummary,
      uptimeMs,
      sessionUptimeMs: uptimeMs,
      stateDurations: this.getStateDurations(resolvedNow),
      transitionHistory: this.getTransitionHistory(),
      transitionCount: this._transitionSeq,
      transitionHistorySize: this._transitionHistorySize,
      transitionEvictedCount: this._transitionEvictedCount,
      maxTransitionHistory: this._maxTransitionHistory,
      lastTransitionReason: this._lastTransitionReason,
      destroyed: this._destroyed,
      restartInFlight: Boolean(this._restartPromise),
      restartCount: this._restartCount,
      coalescedRestarts: this._coalescedRestarts,
      recoveryCount: this._recoveryCount,
      deviceLost: this._deviceLost,
      deviceLossCount: this._deviceLossCount,
      deviceRecoveryCount: this._deviceRecoveryCount,
      lastDeviceLossReason: this._lastDeviceLossReason,
      cleanShutdownCount: this._cleanShutdownCount,
      preloadBufferReleaseCount: this._preloadBufferReleaseCount,
      reclaimRuns: this._reclaimRuns,
      reclaimFailures: this._reclaimFailures,
      reclaimInFlight: Boolean(this._reclaimPromise),
      reclaimHookCount: this._reclaimHooks.size,
      lastReclaimReason: this._lastReclaimReason,
      lastReclaimAtMs: this._lastReclaimAtMs,
      fallbackReason: this._fallbackReason,
      diagnosticReasons: [...this._diagnosticReasons],
      lastError: this._lastError ? this._lastError.message : null,
    };
  }

  getDiagnostics(nowMs) {
    let perfSnapshot = null;
    if (this._perfMonitor && typeof this._perfMonitor.getSnapshot === 'function') {
      try {
        perfSnapshot = this._perfMonitor.getSnapshot();
      } catch (err) {
        this._reportError(toError(err, 'Failed to read PerfMonitor snapshot'), {
          phase: 'diagnostics',
          source: 'perfMonitor',
        });
      }
    }

    return {
      ...this.getSnapshot(nowMs),
      perfSnapshot,
    };
  }

  // --- Internal Helpers ---

  _pushTransitionRecord(record) {
    if (this._transitionHistorySize < this._maxTransitionHistory) {
      const writeIdx =
        (this._transitionHistoryHead + this._transitionHistorySize) %
        this._maxTransitionHistory;
      this._transitionHistoryBuf[writeIdx] = record;
      this._transitionHistorySize += 1;
    } else {
      this._transitionHistoryBuf[this._transitionHistoryHead] = record;
      this._transitionHistoryHead =
        (this._transitionHistoryHead + 1) % this._maxTransitionHistory;
      this._transitionEvictedCount += 1;
    }
  }

  _extractTransitionReason(nextState, detail) {
    if (detail && typeof detail === 'object') {
      if (typeof detail.reason === 'string' && detail.reason.trim().length > 0) {
        return sanitizeTransitionReason(detail.reason);
      }
      if (typeof detail.fallbackReason === 'string' && detail.fallbackReason.trim().length > 0) {
        return sanitizeTransitionReason(detail.fallbackReason);
      }
      if (
        detail.error &&
        typeof detail.error === 'object' &&
        typeof detail.error.message === 'string'
      ) {
        return sanitizeTransitionReason(detail.error.message);
      }
      if (typeof detail.error === 'string' && detail.error.trim().length > 0) {
        return sanitizeTransitionReason(detail.error);
      }
      if (
        Array.isArray(detail.reasons) &&
        detail.reasons.length > 0 &&
        typeof detail.reasons[0] === 'string'
      ) {
        return sanitizeTransitionReason(detail.reasons[0]);
      }
      if (nextState === SESSION_STATE.STARTING && detail.targetMode) {
        return detail.targetMode === SESSION_MODE.WASM
          ? 'Starting Wasm64 runtime'
          : detail.targetMode === SESSION_MODE.SMOKE
            ? 'Starting smoke demo'
            : 'No runtime launcher available';
      }
      if (nextState === SESSION_STATE.RUNNING_WASM) {
        return 'Wasm64 runtime running';
      }
      if (nextState === SESSION_STATE.RUNNING_SMOKE) {
        return 'Smoke demo running';
      }
    }
    return null;
  }

  _transition(nextState, detail = null) {
    const previousState = this._state;
    if (previousState === nextState) {
      return;
    }

    const atMs = this._readNowMs();
    const durationInPrevMs = Math.max(0, atMs - this._stateEnteredAtMs);
    this._stateDurationsMs[previousState] =
      (this._stateDurationsMs[previousState] || 0) + durationInPrevMs;
    this._stateEnteredAtMs = atMs;
    this._state = nextState;

    if (
      nextState === SESSION_STATE.RUNNING_WASM ||
      nextState === SESSION_STATE.RUNNING_SMOKE
    ) {
      this._startedAtMs = atMs;
    }

    const reason = this._extractTransitionReason(nextState, detail);
    this._transitionSeq += 1;
    this._lastTransitionReason = reason;

    const record = {
      seq: this._transitionSeq,
      from: previousState,
      to: nextState,
      atMs,
      durationInPrevMs,
      reason,
    };
    this._pushTransitionRecord(record);

    const event = {
      state: nextState,
      previousState,
      mode: this._mode,
      bootMode: this._bootMode,
      sideModuleCount: this._sideModuleCount,
      sideModuleUrls: [...this._sideModuleUrls],
      timestampMs: atMs,
      durationInPrevMs,
      transitionSeq: record.seq,
      ...(detail && typeof detail === 'object' ? detail : {}),
    };

    const listeners = [...this._stateListeners];
    for (const listener of listeners) {
      try {
        listener(event);
      } catch (err) {
        this._reportError(toError(err, 'State change listener threw'), {
          phase: 'stateChange',
          state: nextState,
          previousState,
        });
      }
    }
  }

  _hudStatus(text, level = 'info') {
    if (this._hud && typeof this._hud.setStatus === 'function') {
      try {
        this._hud.setStatus(text, level);
      } catch {
        // Never allow HUD errors to disrupt session lifecycle.
      }
    }
  }

  _hudLog(message, level = 'info') {
    if (this._hud && typeof this._hud.appendLog === 'function') {
      try {
        this._hud.appendLog(message, level);
      } catch {
        // Never allow HUD errors to disrupt session lifecycle.
      }
    }
  }

  _buildStatusSnapshot(reason = null) {
    const nowMs = this._readNowMs();
    const sessionUptimeMs = this._computeUptimeMs(nowMs);
    return {
      state: this._state,
      mode: this._mode,
      bootMode: this._bootMode,
      currentPhase: this._currentPhase,
      phaseDurationsMs: this.getPhaseDurations(nowMs),
      sideModuleCount: this._sideModuleCount,
      sideModuleUrls: [...this._sideModuleUrls],
      serverStatus: this._cloneServerStatus(),
      fallbackReason: this._fallbackReason,
      validationErrorCount: this._validationErrorCount,
      splashFramesPresented: this._splashFramesPresented,
      firstSplashPresented: this._firstSplashPresented,
      sessionUptimeMs,
      stateDurations: this.getStateDurations(nowMs),
      transitionHistory: this.getTransitionHistory(),
      transitionCount: this._transitionSeq,
      lastTransitionReason: this._lastTransitionReason,
      restartCount: this._restartCount,
      coalescedRestarts: this._coalescedRestarts,
      recoveryCount: this._recoveryCount,
      deviceLost: this._deviceLost,
      deviceLossCount: this._deviceLossCount,
      deviceRecoveryCount: this._deviceRecoveryCount,
      cleanShutdownCount: this._cleanShutdownCount,
      reclaimRuns: this._reclaimRuns,
      reclaimFailures: this._reclaimFailures,
      reason,
      timestampMs: nowMs,
    };
  }

  _cloneStatusForObserver(snapshot) {
    return {
      ...snapshot,
      sideModuleUrls: [...snapshot.sideModuleUrls],
      stateDurations: { ...snapshot.stateDurations },
      phaseDurationsMs: { ...snapshot.phaseDurationsMs },
      transitionHistory: snapshot.transitionHistory.map((entry) => ({ ...entry })),
    };
  }

  _emitStatusUpdate(reason = null) {
    const snapshot = this._buildStatusSnapshot(reason);

    if (this._hud) {
      const hudStatusFns = ['updateBootStatus', 'setBootStatus', 'updateSessionStatus'];
      for (const fnName of hudStatusFns) {
        if (typeof this._hud[fnName] === 'function') {
          try {
            this._hud[fnName](this._cloneStatusForObserver(snapshot));
          } catch {
            // Isolate HUD callback errors.
          }
          break;
        }
      }
      if (typeof this._hud.setBootMode === 'function') {
        try {
          this._hud.setBootMode(this._bootMode, {
            sideModuleCount: this._sideModuleCount,
            sideModuleUrls: [...this._sideModuleUrls],
            state: this._state,
            mode: this._mode,
          });
        } catch {
          // Isolate HUD callback errors.
        }
      }
    }

    for (const listener of [...this._statusListeners]) {
      try {
        listener(this._cloneStatusForObserver(snapshot));
      } catch {
        // Never allow status observers to throw into session lifecycle.
      }
    }

    for (const listener of [...this._telemetryListeners]) {
      try {
        listener({
          type: 'boot-status',
          ...this._cloneStatusForObserver(snapshot),
        });
      } catch {
        // Never allow telemetry observers to throw into session lifecycle.
      }
    }
  }

  /**
   * Apply boot lifecycle metadata (`bootMode`, `sideModuleCount`, `sideModuleUrls`)
   * from `/api/status`, `BootOrchestrator`, `loader.js`, or runtime callbacks and
   * forward sanitized updates to HUD and subscribed status/telemetry observers.
   * Never throws.
   */
  notifyBootLifecycle(payload) {
    try {
      if (!payload || typeof payload !== 'object') {
        return {
          bootMode: this._bootMode,
          sideModuleCount: this._sideModuleCount,
          sideModuleUrls: [...this._sideModuleUrls],
        };
      }

      const nextBootMode =
        normalizeBootMode(payload.bootMode) ||
        normalizeBootMode(payload.mode);
      if (nextBootMode) {
        this._bootMode = nextBootMode;
      }

      const sideMeta = extractSideModuleMetadata(payload);
      if (sideMeta.hasSideModuleUrls) {
        this._sideModuleUrls = [...sideMeta.sideModuleUrls];
        this._sideModuleCount = sideMeta.sideModuleCount;
      } else if (sideMeta.hasSideModuleCount) {
        this._sideModuleCount = sideMeta.sideModuleCount;
      }

      const reason =
        typeof payload.reason === 'string' && payload.reason.trim().length > 0
          ? sanitizeDiagnosticText(payload.reason, 160)
          : typeof payload.stage === 'string' && payload.stage.trim().length > 0
            ? sanitizeDiagnosticText(payload.stage, 64)
            : typeof payload.phase === 'string' && payload.phase.trim().length > 0
              ? sanitizeDiagnosticText(payload.phase, 64)
              : 'boot-lifecycle';

      this._emitStatusUpdate(reason);
    } catch {
      // Boot lifecycle callbacks must never throw.
    }

    return {
      bootMode: this._bootMode,
      sideModuleCount: this._sideModuleCount,
      sideModuleUrls: [...this._sideModuleUrls],
    };
  }

  /**
   * Records a live WebGPU validation/OOM/internal error (e.g. `gpu-validation-error`
   * from `wgpu_worker.js` / `WgpuSwapchain` G35), maintains bounded sanitized history
   * and monotonic counters, and forwards sanitized updates to HUD, CrashReporter,
   * and subscribed listeners without ever throwing.
   */
  recordGpuValidationError(payload) {
    try {
      const envelope = payload && typeof payload === 'object' ? payload : null;
      const rawError =
        envelope && envelope.error !== undefined && envelope.error !== null
          ? envelope.error
          : payload;

      const errorType = normalizeValidationErrorType(rawError, envelope);

      let rawMessage = '';
      if (rawError && typeof rawError === 'object' && typeof rawError.message === 'string') {
        rawMessage = rawError.message;
      } else if (envelope && typeof envelope.message === 'string') {
        rawMessage = envelope.message;
      } else if (typeof rawError === 'string') {
        rawMessage = rawError;
      } else if (rawError !== null && rawError !== undefined) {
        rawMessage = String(rawError);
      }

      const message = sanitizeDiagnosticText(rawMessage) || 'WebGPU validation error';

      let rawTimestamp = null;
      if (rawError && typeof rawError === 'object' && typeof rawError.timestampMs === 'number' && Number.isFinite(rawError.timestampMs)) {
        rawTimestamp = rawError.timestampMs;
      } else if (envelope && typeof envelope.timestampMs === 'number' && Number.isFinite(envelope.timestampMs)) {
        rawTimestamp = envelope.timestampMs;
      }
      const nowVal = this._now();
      const timestampMs =
        rawTimestamp !== null
          ? rawTimestamp
          : typeof nowVal === 'number' && Number.isFinite(nowVal)
            ? nowVal
            : Date.now();

      const workerSeq =
        rawError && typeof rawError === 'object' && isNonNegativeInt(rawError.seq)
          ? rawError.seq
          : envelope && isNonNegativeInt(envelope.seq)
            ? envelope.seq
            : null;

      const rawSource =
        (envelope && typeof envelope.source === 'string' && envelope.source) ||
        (rawError && typeof rawError === 'object' && typeof rawError.source === 'string' && rawError.source) ||
        'wgpu_worker';
      const source = sanitizeDiagnosticText(rawSource, 64) || 'wgpu_worker';

      const rawLabel =
        (envelope && typeof envelope.label === 'string' && envelope.label) ||
        (rawError && typeof rawError === 'object' && typeof rawError.label === 'string' && rawError.label) ||
        null;
      const label = rawLabel ? sanitizeDiagnosticText(rawLabel, 128) : null;

      this._validationErrorSeq += 1;
      this._validationErrorCount += 1;

      const record = {
        seq: this._validationErrorSeq,
        type: errorType,
        message,
        source,
        timestampMs,
        ...(workerSeq !== null ? { workerSeq } : {}),
        ...(label ? { label } : {}),
      };

      this._recentValidationErrors.push(record);
      while (this._recentValidationErrors.length > this._maxValidationErrors) {
        this._recentValidationErrors.shift();
      }

      const level = errorType === 'out-of-memory' || errorType === 'internal' ? 'error' : 'warn';
      this._hudLog(`[WebGPU ${errorType}] ${message}`, level);

      const eventPayload = {
        type: 'gpu-validation-error',
        error: { ...record },
        record: { ...record },
        validationErrorCount: this._validationErrorCount,
        validationErrorHistorySize: this._recentValidationErrors.length,
        bootMode: this._bootMode,
        state: this._state,
        mode: this._mode,
      };

      if (this._hud) {
        const hudErrFns = ['recordValidationError', 'onValidationError', 'updateGpuValidation'];
        for (const fnName of hudErrFns) {
          if (typeof this._hud[fnName] === 'function') {
            try {
              this._hud[fnName]({ ...eventPayload, error: { ...record }, record: { ...record } });
            } catch {
              // Isolate HUD callback errors.
            }
            break;
          }
        }
        if (typeof this._hud.updateTelemetry === 'function') {
          try {
            this._hud.updateTelemetry({ ...eventPayload, error: { ...record }, record: { ...record } });
          } catch {
            // Isolate HUD callback errors.
          }
        }
      }

      this._reportError(new Error(`[WebGPU ${errorType}] ${message}`), {
        phase: 'runtime',
        source,
        category: 'webgpu_validation',
        errorType,
        seq: record.seq,
        validationErrorCount: this._validationErrorCount,
        bootMode: this._bootMode,
      });

      for (const listener of [...this._validationErrorListeners]) {
        try {
          listener({ ...eventPayload, error: { ...record }, record: { ...record } });
        } catch {
          // Never allow validation error observers to throw.
        }
      }

      for (const listener of [...this._telemetryListeners]) {
        try {
          listener({ ...eventPayload, error: { ...record }, record: { ...record } });
        } catch {
          // Never allow telemetry observers to throw.
        }
      }

      return {
        handled: true,
        type: 'gpu-validation-error',
        record: { ...record },
        validationErrorCount: this._validationErrorCount,
        validationErrorHistorySize: this._recentValidationErrors.length,
      };
    } catch {
      return {
        handled: false,
        type: 'gpu-validation-error',
        record: null,
        validationErrorCount: this._validationErrorCount,
        validationErrorHistorySize: this._recentValidationErrors.length,
      };
    }
  }

  /**
   * Records a `splash-frame-presented` message from `wgpu_worker.js` (W67),
   * updates splash presentation counters/state, and forwards sanitized updates
   * to HUD and subscribed listeners without ever throwing.
   */
  recordSplashFramePresented(payload) {
    try {
      const msg = payload && typeof payload === 'object' ? payload : {};
      const ok = msg.ok !== undefined ? Boolean(msg.ok) : !msg.error;
      const frameIndex = isNonNegativeInt(msg.frameIndex) ? msg.frameIndex : null;
      const workerFramesPresented = isNonNegativeInt(msg.framesPresented)
        ? msg.framesPresented
        : null;
      const rawSource = typeof msg.source === 'string' && msg.source ? msg.source : 'splash';
      const source = sanitizeDiagnosticText(rawSource, 64) || 'splash';
      const reason =
        typeof msg.reason === 'string' && msg.reason.trim().length > 0
          ? sanitizeDiagnosticText(msg.reason, 128)
          : null;
      const error = !ok
        ? sanitizeDiagnosticText(msg.error || msg.reason || 'Splash frame presentation failed')
        : msg.error
          ? sanitizeDiagnosticText(msg.error)
          : null;

      const nowVal = this._now();
      const timestampMs =
        typeof msg.timestampMs === 'number' && Number.isFinite(msg.timestampMs)
          ? msg.timestampMs
          : typeof nowVal === 'number' && Number.isFinite(nowVal)
            ? nowVal
            : Date.now();

      let isFirstPresent = false;
      if (ok) {
        this._splashFramesPresented += 1;
        isFirstPresent = Boolean(msg.firstPresent) || !this._firstSplashPresented;
        if (isFirstPresent) {
          this._firstSplashPresented = true;
        }
      } else {
        this._splashFrameErrorCount += 1;
      }

      const record = {
        ok,
        frameIndex,
        framesPresented: this._splashFramesPresented,
        workerFramesPresented:
          workerFramesPresented !== null ? workerFramesPresented : this._splashFramesPresented,
        firstPresent: isFirstPresent,
        source,
        reason,
        error,
        timestampMs,
      };
      this._lastSplashFrame = record;

      if (ok && isFirstPresent) {
        this._hudLog(
          `First splash frame presented (frame ${frameIndex ?? 0}, total ${this._splashFramesPresented}).`,
          'ok'
        );
      } else if (!ok) {
        this._hudLog(
          `Splash frame presentation warning: ${error || 'failed'}`,
          'warn'
        );
      }

      const eventPayload = {
        type: 'splash-frame-presented',
        ...record,
        splashFramesPresented: this._splashFramesPresented,
        splashFrameErrorCount: this._splashFrameErrorCount,
        firstSplashPresented: this._firstSplashPresented,
        bootMode: this._bootMode,
        state: this._state,
        mode: this._mode,
      };

      if (this._hud) {
        const hudSplashFns = ['onSplashFramePresented', 'recordSplashFrame', 'updateSplashStatus'];
        for (const fnName of hudSplashFns) {
          if (typeof this._hud[fnName] === 'function') {
            try {
              this._hud[fnName]({ ...eventPayload });
            } catch {
              // Isolate HUD callback errors.
            }
            break;
          }
        }
        if (typeof this._hud.updateTelemetry === 'function') {
          try {
            this._hud.updateTelemetry({ ...eventPayload });
          } catch {
            // Isolate HUD callback errors.
          }
        }
      }

      for (const listener of [...this._splashListeners]) {
        try {
          listener({ ...eventPayload });
        } catch {
          // Never allow splash observers to throw.
        }
      }

      for (const listener of [...this._telemetryListeners]) {
        try {
          listener({ ...eventPayload });
        } catch {
          // Never allow telemetry observers to throw.
        }
      }

      return {
        handled: true,
        type: 'splash-frame-presented',
        record: { ...record },
        splashFramesPresented: this._splashFramesPresented,
        splashFrameErrorCount: this._splashFrameErrorCount,
        firstSplashPresented: this._firstSplashPresented,
      };
    } catch {
      return {
        handled: false,
        type: 'splash-frame-presented',
        record: null,
        splashFramesPresented: this._splashFramesPresented,
        splashFrameErrorCount: this._splashFrameErrorCount,
        firstSplashPresented: this._firstSplashPresented,
      };
    }
  }

  /**
   * Records a WebGPU device-lost event (direct call or `gpu-device-lost` worker message),
   * sanitizes the reason, increments monotonic stability counters, reports to CrashReporter
   * and HUD, and notifies subscribed observers without ever throwing.
   *
   * @param {string|Error|object|null|undefined} [reason]
   */
  recordDeviceLost(reason) {
    try {
      const payload = reason && typeof reason === 'object' ? reason : null;
      let rawReason = '';
      if (typeof reason === 'string') {
        rawReason = reason;
      } else if (reason instanceof Error) {
        rawReason = reason.message || reason.name || 'device-lost';
      } else if (payload) {
        if (typeof payload.reason === 'string' && payload.reason.trim().length > 0) {
          rawReason = payload.reason;
          if (typeof payload.message === 'string' && payload.message.trim().length > 0 && payload.message.trim() !== payload.reason.trim()) {
            rawReason = `${payload.reason}: ${payload.message}`;
          }
        } else if (typeof payload.message === 'string' && payload.message.trim().length > 0) {
          rawReason = payload.message;
        } else if (payload.error && typeof payload.error === 'object' && typeof payload.error.message === 'string') {
          rawReason = payload.error.message;
        } else if (typeof payload.error === 'string') {
          rawReason = payload.error;
        }
      } else if (reason !== null && reason !== undefined) {
        rawReason = String(reason);
      }

      const sanitizedReason =
        sanitizeDiagnosticText(rawReason, MAX_TRANSITION_REASON_LENGTH) || 'unknown';
      const rawSource =
        (payload && typeof payload.source === 'string' && payload.source) || 'wgpu_worker';
      const source = sanitizeDiagnosticText(rawSource, 64) || 'wgpu_worker';
      const timestampMs =
        payload && typeof payload.timestampMs === 'number' && Number.isFinite(payload.timestampMs)
          ? payload.timestampMs
          : this._readNowMs();

      this._deviceLossCount += 1;
      this._deviceLost = true;
      this._lastDeviceLossReason = sanitizedReason;
      this._lastDeviceLossAtMs = timestampMs;

      this._hudLog(`[WebGPU device-lost] ${sanitizedReason}`, 'error');

      const eventPayload = {
        type: 'gpu-device-lost',
        reason: sanitizedReason,
        source,
        deviceLost: true,
        deviceLossCount: this._deviceLossCount,
        deviceRecoveryCount: this._deviceRecoveryCount,
        bootMode: this._bootMode,
        state: this._state,
        mode: this._mode,
        timestampMs,
      };

      if (this._hud) {
        const hudLostFns = ['onDeviceLost', 'recordDeviceLost'];
        for (const fnName of hudLostFns) {
          if (typeof this._hud[fnName] === 'function') {
            try {
              this._hud[fnName]({ ...eventPayload });
            } catch {
              // Isolate HUD callback errors.
            }
            break;
          }
        }
        if (typeof this._hud.updateTelemetry === 'function') {
          try {
            this._hud.updateTelemetry({ ...eventPayload });
          } catch {
            // Isolate HUD callback errors.
          }
        }
      }

      this._reportError(new Error(`[WebGPU device-lost] ${sanitizedReason}`), {
        phase: 'runtime',
        source,
        category: 'webgpu_device_lost',
        reason: sanitizedReason,
        deviceLossCount: this._deviceLossCount,
        bootMode: this._bootMode,
      });

      for (const listener of [...this._deviceLostListeners]) {
        try {
          listener({ ...eventPayload });
        } catch {
          // Never allow device-lost observers to throw.
        }
      }

      for (const listener of [...this._telemetryListeners]) {
        try {
          listener({ ...eventPayload });
        } catch {
          // Never allow telemetry observers to throw.
        }
      }

      this._emitStatusUpdate('gpu-device-lost');

      return {
        handled: true,
        type: 'gpu-device-lost',
        reason: sanitizedReason,
        deviceLost: true,
        deviceLossCount: this._deviceLossCount,
        deviceRecoveryCount: this._deviceRecoveryCount,
        timestampMs,
      };
    } catch {
      return {
        handled: false,
        type: 'gpu-device-lost',
        reason: this._lastDeviceLossReason,
        deviceLost: this._deviceLost,
        deviceLossCount: this._deviceLossCount,
        deviceRecoveryCount: this._deviceRecoveryCount,
        timestampMs: this._readNowMs(),
      };
    }
  }

  /**
   * Records a WebGPU device recovery event (direct call or `gpu-device-recovered` worker message),
   * increments recovery counters, updates HUD/observers, and clears the active device-lost flag
   * without ever throwing.
   *
   * @param {string|object|null|undefined} [detail]
   */
  recordDeviceRecovered(detail) {
    try {
      const payload = detail && typeof detail === 'object' ? detail : null;
      const rawNote =
        typeof detail === 'string'
          ? detail
          : payload && typeof payload.reason === 'string'
            ? payload.reason
            : payload && typeof payload.message === 'string'
              ? payload.message
              : null;
      const note = rawNote ? sanitizeDiagnosticText(rawNote, MAX_TRANSITION_REASON_LENGTH) : null;
      const rawSource =
        (payload && typeof payload.source === 'string' && payload.source) || 'wgpu_worker';
      const source = sanitizeDiagnosticText(rawSource, 64) || 'wgpu_worker';
      const timestampMs =
        payload && typeof payload.timestampMs === 'number' && Number.isFinite(payload.timestampMs)
          ? payload.timestampMs
          : this._readNowMs();
      const recoveryDurationMs =
        this._lastDeviceLossAtMs !== null
          ? Math.max(0, timestampMs - this._lastDeviceLossAtMs)
          : null;

      this._deviceRecoveryCount += 1;
      this._deviceLost = false;
      this._lastDeviceRecoveredAtMs = timestampMs;

      this._hudLog(
        `[WebGPU device-recovered] Device recovered (total recoveries: ${this._deviceRecoveryCount}).`,
        'ok'
      );

      const eventPayload = {
        type: 'gpu-device-recovered',
        source,
        deviceLost: false,
        deviceLossCount: this._deviceLossCount,
        deviceRecoveryCount: this._deviceRecoveryCount,
        recoveryDurationMs,
        ...(note ? { reason: note } : {}),
        bootMode: this._bootMode,
        state: this._state,
        mode: this._mode,
        timestampMs,
      };

      if (this._hud) {
        const hudRecFns = ['onDeviceRecovered', 'recordDeviceRecovered'];
        for (const fnName of hudRecFns) {
          if (typeof this._hud[fnName] === 'function') {
            try {
              this._hud[fnName]({ ...eventPayload });
            } catch {
              // Isolate HUD callback errors.
            }
            break;
          }
        }
        if (typeof this._hud.updateTelemetry === 'function') {
          try {
            this._hud.updateTelemetry({ ...eventPayload });
          } catch {
            // Isolate HUD callback errors.
          }
        }
      }

      for (const listener of [...this._deviceRecoveredListeners]) {
        try {
          listener({ ...eventPayload });
        } catch {
          // Never allow device-recovered observers to throw.
        }
      }

      for (const listener of [...this._telemetryListeners]) {
        try {
          listener({ ...eventPayload });
        } catch {
          // Never allow telemetry observers to throw.
        }
      }

      this._emitStatusUpdate('gpu-device-recovered');

      return {
        handled: true,
        type: 'gpu-device-recovered',
        deviceLost: false,
        deviceLossCount: this._deviceLossCount,
        deviceRecoveryCount: this._deviceRecoveryCount,
        recoveryDurationMs,
        timestampMs,
      };
    } catch {
      return {
        handled: false,
        type: 'gpu-device-recovered',
        deviceLost: this._deviceLost,
        deviceLossCount: this._deviceLossCount,
        deviceRecoveryCount: this._deviceRecoveryCount,
        recoveryDurationMs: null,
        timestampMs: this._readNowMs(),
      };
    }
  }

  _resetSessionPhases(enterBoot = false) {
    this._phaseDurationsMs = createEmptyPhaseDurations();
    this._phaseTransitionCount = 0;
    if (enterBoot) {
      this._currentPhase = SESSION_PHASE.BOOT;
      this._phaseEnteredAtMs = this._readNowMs();
      this._phaseTransitionCount = 1;
    } else {
      this._currentPhase = null;
      this._phaseEnteredAtMs = null;
    }
  }

  _finalizeCurrentPhase(atMs) {
    if (this._currentPhase !== null && this._phaseEnteredAtMs !== null) {
      const resolvedNow = this._resolveNowMs(atMs);
      const elapsedMs = Math.max(0, resolvedNow - this._phaseEnteredAtMs);
      this._phaseDurationsMs[this._currentPhase] =
        (this._phaseDurationsMs[this._currentPhase] || 0) + elapsedMs;
    }
    this._currentPhase = null;
    this._phaseEnteredAtMs = null;
  }

  /**
   * Records a sanitized session lifecycle phase transition
   * (`boot -> start_screen -> intro_video -> main_menu -> in_game`),
   * accumulating elapsed duration in the previous phase without ever throwing.
   *
   * @param {string|{ phase: string, timestampMs?: number }} phase
   * @param {number} [timestampMs]
   */
  recordPhase(phase, timestampMs) {
    try {
      let rawPhase = phase;
      let rawTimestamp = timestampMs;
      if (phase && typeof phase === 'object') {
        rawPhase = phase.phase;
        if (rawTimestamp === undefined) {
          rawTimestamp = phase.timestampMs;
        }
      }

      const normalizedPhase = normalizeSessionPhase(rawPhase);
      if (!normalizedPhase) {
        return {
          ok: false,
          changed: false,
          phase: null,
          previousPhase: this._currentPhase,
          currentPhase: this._currentPhase,
          durationInPrevMs: 0,
          phaseDurationsMs: this.getPhaseDurations(),
          timestampMs: this._readNowMs(),
        };
      }

      const atMs =
        typeof rawTimestamp === 'number' && Number.isFinite(rawTimestamp)
          ? rawTimestamp
          : this._readNowMs();

      const previousPhase = this._currentPhase;
      if (previousPhase === normalizedPhase) {
        return {
          ok: true,
          changed: false,
          phase: normalizedPhase,
          previousPhase,
          currentPhase: this._currentPhase,
          durationInPrevMs: 0,
          phaseDurationsMs: this.getPhaseDurations(atMs),
          timestampMs: atMs,
        };
      }

      let durationInPrevMs = 0;
      if (previousPhase !== null && this._phaseEnteredAtMs !== null) {
        durationInPrevMs = Math.max(0, atMs - this._phaseEnteredAtMs);
        this._phaseDurationsMs[previousPhase] =
          (this._phaseDurationsMs[previousPhase] || 0) + durationInPrevMs;
      }

      this._currentPhase = normalizedPhase;
      this._phaseEnteredAtMs = atMs;
      this._phaseTransitionCount += 1;

      const phaseDurationsMs = this.getPhaseDurations(atMs);
      this._emitStatusUpdate(`phase:${normalizedPhase}`);

      return {
        ok: true,
        changed: true,
        phase: normalizedPhase,
        previousPhase,
        currentPhase: this._currentPhase,
        durationInPrevMs,
        phaseDurationsMs,
        timestampMs: atMs,
      };
    } catch {
      return {
        ok: false,
        changed: false,
        phase: null,
        previousPhase: this._currentPhase,
        currentPhase: this._currentPhase,
        durationInPrevMs: 0,
        phaseDurationsMs: this.getPhaseDurations(),
        timestampMs: this._readNowMs(),
      };
    }
  }

  /**
   * Invokes registered soft memory-pressure reclaim hooks (e.g. RT pool oversized
   * reclaim, staging slab trim, OPFS quota eviction) before falling back to a full
   * session restart. Isolates per-hook errors, guards against overlapping/re-entrant
   * invocations, and tracks `reclaimRuns` and `reclaimFailures`. Never rejects.
   *
   * @param {string|{ reason?: string }} [reason='memory-pressure']
   */
  async requestMemoryReclaim(reason = 'memory-pressure') {
    const rawReason =
      typeof reason === 'string' && reason.trim().length > 0
        ? reason
        : reason &&
            typeof reason === 'object' &&
            typeof reason.reason === 'string' &&
            reason.reason.trim().length > 0
          ? reason.reason
          : 'memory-pressure';
    const sanitizedReason =
      sanitizeDiagnosticText(rawReason, MAX_TRANSITION_REASON_LENGTH) || 'memory-pressure';
    const nowMs = this._readNowMs();

    if (this._destroyed) {
      return {
        ok: false,
        destroyed: true,
        reason: sanitizedReason,
        hooksInvoked: 0,
        hooksSucceeded: 0,
        hooksFailed: 0,
        errors: [],
        reclaimRuns: this._reclaimRuns,
        reclaimFailures: this._reclaimFailures,
        timestampMs: nowMs,
      };
    }

    if (this._executingReclaimHook) {
      return {
        ok: false,
        skipped: true,
        reentrant: true,
        reason: sanitizedReason,
        hooksInvoked: 0,
        hooksSucceeded: 0,
        hooksFailed: 0,
        errors: [],
        reclaimRuns: this._reclaimRuns,
        reclaimFailures: this._reclaimFailures,
        timestampMs: nowMs,
      };
    }

    if (this._reclaimPromise) {
      const firstResult = await this._reclaimPromise;
      return {
        ...firstResult,
        errors: [...firstResult.errors],
        coalesced: true,
      };
    }

    const runReclaim = async () => {
      const timestampMs = this._readNowMs();
      this._reclaimRuns += 1;
      this._lastReclaimReason = sanitizedReason;
      this._lastReclaimAtMs = timestampMs;

      const hooks = [...this._reclaimHooks];
      const baseContext = {
        reason: sanitizedReason,
        state: this._state,
        mode: this._mode,
        bootMode: this._bootMode,
        currentPhase: this._currentPhase,
        timestampMs,
      };

      let hooksInvoked = 0;
      let hooksSucceeded = 0;
      let hooksFailed = 0;
      const errors = [];

      for (let i = 0; i < hooks.length; i += 1) {
        if (this._destroyed) {
          break;
        }
        const hook = hooks[i];
        if (!this._reclaimHooks.has(hook)) {
          continue;
        }
        hooksInvoked += 1;
        try {
          this._executingReclaimHook = true;
          let maybePromise;
          try {
            maybePromise = hook({ ...baseContext });
          } finally {
            this._executingReclaimHook = false;
          }
          await maybePromise;
          hooksSucceeded += 1;
        } catch (err) {
          hooksFailed += 1;
          this._reclaimFailures += 1;
          const normalized = toError(err, 'Memory reclaim hook failed');
          const sanitizedMessage =
            sanitizeDiagnosticText(normalized.message) || 'Memory reclaim hook failed';
          errors.push(sanitizedMessage);
          this._hudLog(
            `Memory reclaim hook warning (${sanitizedReason}): ${sanitizedMessage}`,
            'warn'
          );
          this._reportError(new Error(sanitizedMessage), {
            phase: 'reclaim',
            source: 'memoryReclaim',
            reason: sanitizedReason,
            hookIndex: i,
            reclaimRuns: this._reclaimRuns,
            reclaimFailures: this._reclaimFailures,
          });
        }
      }

      const eventPayload = {
        type: 'memory-reclaim',
        ok: hooksFailed === 0,
        reason: sanitizedReason,
        hooksInvoked,
        hooksSucceeded,
        hooksFailed,
        errors: [...errors],
        reclaimRuns: this._reclaimRuns,
        reclaimFailures: this._reclaimFailures,
        state: this._state,
        mode: this._mode,
        bootMode: this._bootMode,
        currentPhase: this._currentPhase,
        timestampMs,
      };

      for (const listener of [...this._telemetryListeners]) {
        try {
          listener({ ...eventPayload, errors: [...errors] });
        } catch {
          // Never allow telemetry observers to throw.
        }
      }

      this._emitStatusUpdate(`memory-reclaim:${sanitizedReason}`);

      return {
        ok: hooksFailed === 0,
        reason: sanitizedReason,
        hooksInvoked,
        hooksSucceeded,
        hooksFailed,
        errors,
        reclaimRuns: this._reclaimRuns,
        reclaimFailures: this._reclaimFailures,
        timestampMs,
      };
    };

    const promise = runReclaim().finally(() => {
      if (this._reclaimPromise === promise) {
        this._reclaimPromise = null;
      }
    });
    this._reclaimPromise = promise;
    return promise;
  }

  /**
   * Dispatches a message (or MessageEvent) from `wgpu_worker.js` / `loader.js`.
   * Handles `gpu-validation-error`, `splash-frame-presented`, device-lost/recovery,
   * `session-phase`, and optional boot lifecycle messages without ever throwing.
   */
  handleWorkerMessage(messageOrEvent) {
    try {
      const msg =
        messageOrEvent &&
        typeof messageOrEvent === 'object' &&
        'data' in messageOrEvent &&
        messageOrEvent.data &&
        typeof messageOrEvent.data === 'object'
          ? messageOrEvent.data
          : messageOrEvent;

      if (!msg || typeof msg !== 'object') {
        return { handled: false, type: null };
      }

      if (msg.type === 'gpu-validation-error') {
        return this.recordGpuValidationError(msg);
      }

      if (msg.type === 'splash-frame-presented') {
        return this.recordSplashFramePresented(msg);
      }

      if (
        msg.type === 'gpu-device-lost' ||
        msg.type === 'device-lost' ||
        msg.type === 'webgpu-device-lost'
      ) {
        return this.recordDeviceLost(msg);
      }

      if (
        msg.type === 'gpu-device-recovered' ||
        msg.type === 'device-recovered' ||
        msg.type === 'webgpu-device-recovered'
      ) {
        return this.recordDeviceRecovered(msg);
      }

      if (msg.type === 'session-phase' || msg.type === 'phase-transition') {
        const res = this.recordPhase(msg.phase, msg.timestampMs);
        return {
          handled: res.ok,
          type: msg.type,
          ...res,
        };
      }

      if (
        msg.type === 'boot-status' ||
        msg.type === 'boot-lifecycle' ||
        msg.type === 'runtime-boot-mode'
      ) {
        const meta = this.notifyBootLifecycle(msg);
        return {
          handled: true,
          type: msg.type,
          ...meta,
        };
      }

      return {
        handled: false,
        type: typeof msg.type === 'string' ? msg.type : null,
      };
    } catch {
      return { handled: false, type: null };
    }
  }

  /**
   * Subscribes to `message` events on a `wgpu_worker.js` (or compatible EventTarget)
   * instance so `gpu-validation-error` and `splash-frame-presented` notifications flow
   * automatically into session state and observers. Returns an idempotent detach function.
   */
  attachWorker(worker) {
    if (!isEventTargetLike(worker)) {
      throw new TypeError('attachWorker requires an EventTarget-like worker with addEventListener/removeEventListener');
    }
    const existing = this._attachedWorkers.get(worker);
    if (existing) {
      return existing;
    }

    const onMessage = (event) => {
      this.handleWorkerMessage(event);
    };
    worker.addEventListener('message', onMessage);

    let active = true;
    const detach = () => {
      if (!active) return;
      active = false;
      this._attachedWorkers.delete(worker);
      try {
        worker.removeEventListener('message', onMessage);
      } catch {
        // Ignore worker listener removal errors during teardown.
      }
    };

    this._attachedWorkers.set(worker, detach);
    return detach;
  }

  _reportError(err, context = {}) {
    const reporter = this._crashReporter;
    if (!reporter) {
      return;
    }
    const normalized = toError(err);
    try {
      if (typeof reporter === 'function') {
        reporter(normalized, context);
      } else if (typeof reporter.captureError === 'function') {
        reporter.captureError(normalized, context);
      } else if (typeof reporter.captureException === 'function') {
        reporter.captureException(normalized, context);
      } else if (typeof reporter.recordError === 'function') {
        reporter.recordError(normalized, context);
      } else if (typeof reporter.reportError === 'function') {
        reporter.reportError(normalized, context);
      } else if (typeof reporter.record === 'function') {
        reporter.record(normalized, context);
      } else if (typeof reporter.report === 'function') {
        reporter.report(normalized, context);
      }
    } catch {
      // Never allow crash reporter failures to throw into session lifecycle.
    }
  }

  async _stopHandle(handle) {
    if (!handle) {
      return true;
    }
    try {
      if (typeof handle === 'function') {
        await handle();
      } else if (typeof handle.stop === 'function') {
        await handle.stop();
      } else if (typeof handle.dispose === 'function') {
        await handle.dispose();
      } else if (typeof handle.destroy === 'function') {
        await handle.destroy();
      } else if (typeof handle.terminate === 'function') {
        await handle.terminate();
      }
      return true;
    } catch (err) {
      const normalized = toError(err, 'Failed to stop active runtime handle');
      this._reportError(normalized, { phase: 'stop', source: 'runtimeHandle' });
      this._hudLog(`Runtime stop warning: ${normalized.message}`, 'warn');
      return false;
    }
  }

  _releasePreloadBuffers(handle = null) {
    const targets = [];
    const seen = new Set();
    const addTarget = (candidate) => {
      if (candidate && typeof candidate === 'object' && !seen.has(candidate)) {
        seen.add(candidate);
        targets.push(candidate);
      }
    };

    addTarget(this._bootOrchestrator);
    if (handle && typeof handle === 'object') {
      addTarget(handle);
      addTarget(handle.bootOrchestrator);
      addTarget(handle.orchestrator);
    }

    let ok = true;
    let invoked = false;
    for (const target of targets) {
      const releaseFn =
        typeof target.releasePreloadBuffers === 'function'
          ? target.releasePreloadBuffers
          : typeof target.releaseBootBundleBytes === 'function'
            ? target.releaseBootBundleBytes
            : null;
      if (!releaseFn) {
        continue;
      }
      invoked = true;
      try {
        releaseFn.call(target);
      } catch (err) {
        ok = false;
        this._reportError(toError(err, 'Failed to release preload buffers'), {
          phase: 'stop',
          source: 'preloadBuffers',
        });
      }
    }

    if (invoked) {
      this._preloadBufferReleaseCount += 1;
    }
    return ok;
  }

  _teardownSubsystems() {
    let clean = true;
    if (this._perfDetach) {
      const detach = this._perfDetach;
      this._perfDetach = null;
      try {
        detach();
      } catch (err) {
        clean = false;
        this._reportError(toError(err, 'Failed to detach PerfMonitor from HUD'), {
          phase: 'stop',
          source: 'perfMonitor',
        });
      }
    } else if (this._perfMonitor && typeof this._perfMonitor.detachHud === 'function') {
      try {
        this._perfMonitor.detachHud();
      } catch (err) {
        clean = false;
        this._reportError(toError(err, 'Failed to detach PerfMonitor from HUD'), {
          phase: 'stop',
          source: 'perfMonitor',
        });
      }
    }

    if (this._inputBridge && typeof this._inputBridge.stop === 'function') {
      if (this._inputStarted || this._inputBridge.running) {
        this._inputStarted = false;
        try {
          this._inputBridge.stop();
        } catch (err) {
          clean = false;
          this._reportError(toError(err, 'Failed to stop InputBridge'), {
            phase: 'stop',
            source: 'inputBridge',
          });
        }
      }
    } else {
      this._inputStarted = false;
    }

    if (this._audioGestureCleanup) {
      const cleanup = this._audioGestureCleanup;
      this._audioGestureCleanup = null;
      try {
        cleanup();
      } catch (err) {
        clean = false;
        this._reportError(toError(err, 'Failed to clean up audio user-gesture listeners'), {
          phase: 'stop',
          source: 'audioEngine',
        });
      }
    }

    if (this._attachedWorkers.size > 0) {
      const detachers = [...this._attachedWorkers.values()];
      this._attachedWorkers.clear();
      for (const detach of detachers) {
        try {
          detach();
        } catch (err) {
          clean = false;
          this._reportError(toError(err, 'Failed to detach worker message listener'), {
            phase: 'stop',
            source: 'wgpuWorker',
          });
        }
      }
    }

    if (this._sessionReclaimUnregisters.size > 0) {
      const unregisters = [...this._sessionReclaimUnregisters];
      this._sessionReclaimUnregisters.clear();
      for (const unregister of unregisters) {
        try {
          unregister();
        } catch {
          // Isolate session-scoped reclaim hook unregister errors.
        }
      }
    }
    return clean;
  }

  _wireSubsystems() {
    // 1. Attach PerfMonitor to HUD (if both provided).
    if (
      this._perfMonitor &&
      this._hud &&
      typeof this._perfMonitor.attachHud === 'function' &&
      typeof this._hud.updateMetrics === 'function'
    ) {
      const detach =
        this._hudIntervalMs !== undefined
          ? this._perfMonitor.attachHud(this._hud, this._hudIntervalMs)
          : this._perfMonitor.attachHud(this._hud);
      this._perfDetach =
        typeof detach === 'function'
          ? detach
          : () => {
              if (typeof this._perfMonitor.detachHud === 'function') {
                this._perfMonitor.detachHud();
              }
            };
    }

    // 2. Start InputBridge (if provided).
    if (this._inputBridge && typeof this._inputBridge.start === 'function') {
      this._inputBridge.start();
      this._inputStarted = true;
    }

    // 3. Register AudioEngine user-gesture unlock (if provided).
    if (this._audioEngine && typeof this._audioEngine.resumeOnUserGesture === 'function') {
      if (this._audioGestureCleanup) {
        this._audioGestureCleanup();
        this._audioGestureCleanup = null;
      }
      const cleanup =
        this._audioGestureTarget !== undefined
          ? this._audioEngine.resumeOnUserGesture(this._audioGestureTarget)
          : this._audioEngine.resumeOnUserGesture();
      this._audioGestureCleanup = typeof cleanup === 'function' ? cleanup : null;
    }

    // 4. Attach optional explicit wgpuWorker (or handle.wgpuWorker) for live GPU telemetry.
    if (this._wgpuWorker) {
      this.attachWorker(this._wgpuWorker);
    }
    if (
      this._activeHandle &&
      typeof this._activeHandle === 'object' &&
      isEventTargetLike(this._activeHandle.wgpuWorker)
    ) {
      this.attachWorker(this._activeHandle.wgpuWorker);
    }
  }

  // --- Environment Probing ---

  /**
   * Probes browser capabilities, fetches `/api/status`, and inspects storage summary
   * if a storageManager is configured. Never throws unhandled exceptions.
   */
  async probeEnvironment() {
    // 1. Browser capabilities
    let capabilities;
    const defaultCaps = probeBrowserCapabilities();
    if (this._capabilityProbe) {
      try {
        const rawCaps = await this._capabilityProbe();
        capabilities = normalizeCapabilities(rawCaps, defaultCaps);
      } catch (err) {
        const normalized = toError(err, 'Capability probe failed');
        this._reportError(normalized, { phase: 'probe', source: 'capabilityProbe' });
        this._hudLog(`Capability probe error: ${normalized.message}`, 'warn');
        capabilities = normalizeCapabilities(null, null);
      }
    } else {
      capabilities = defaultCaps;
    }
    this._capabilities = capabilities;

    // 2. Server readiness (/api/status)
    let serverStatus;
    if (!this._fetchImpl) {
      const err = new Error(`Unable to fetch ${this._statusUrl}: fetch API is unavailable`);
      this._reportError(err, { phase: 'probe', source: 'status', url: this._statusUrl });
      serverStatus = createOfflineServerStatus(err.message, null);
    } else {
      try {
        const response = await this._fetchImpl(this._statusUrl, { cache: 'no-store' });
        if (!response || typeof response.ok !== 'boolean') {
          throw new Error(`Invalid response from ${this._statusUrl}`);
        }
        if (!response.ok) {
          const httpErr = new Error(
            `Status probe ${this._statusUrl} failed with HTTP ${response.status ?? 'unknown'}`
          );
          this._reportError(httpErr, {
            phase: 'probe',
            source: 'status',
            url: this._statusUrl,
            httpStatus: response.status ?? null,
          });
          this._hudLog(httpErr.message, 'warn');
          serverStatus = createOfflineServerStatus(httpErr.message, response.status ?? null);
        } else {
          const payload = await response.json();
          if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
            throw new Error(`Status probe ${this._statusUrl} returned non-object JSON`);
          }
          serverStatus = normalizeServerStatus(
            payload,
            typeof response.status === 'number' ? response.status : 200
          );
        }
      } catch (err) {
        const normalized = toError(err, `Failed to reach ${this._statusUrl}`);
        this._reportError(normalized, {
          phase: 'probe',
          source: 'status',
          url: this._statusUrl,
        });
        this._hudLog(`Server status probe failed (${this._statusUrl}): ${normalized.message}`, 'warn');
        serverStatus = createOfflineServerStatus(normalized.message, null);
      }
    }
    this._serverStatus = serverStatus;
    this._sideModuleCount = serverStatus.sideModuleCount;
    this._sideModuleUrls = [...serverStatus.sideModuleUrls];

    // 3. Optional BrowserStorageManager summary
    if (this._storageManager && typeof this._storageManager.getSummary === 'function') {
      try {
        this._storageSummary = await this._storageManager.getSummary();
      } catch (err) {
        const normalized = toError(err, 'Failed to read browser storage summary');
        this._reportError(normalized, { phase: 'probe', source: 'storageManager' });
        this._storageSummary = null;
      }
    }

    const { missing, reasons } = evaluateRequirements(capabilities);
    this._emitStatusUpdate('probe');

    return {
      capabilities: { ...capabilities },
      serverStatus: this._cloneServerStatus(),
      bootMode: this._bootMode,
      sideModuleCount: this._sideModuleCount,
      sideModuleUrls: [...this._sideModuleUrls],
      storageSummary: this._storageSummary,
      missingCapabilities: missing,
      diagnosticReasons: reasons,
      ready: missing.length === 0,
      probedAtMs: this._now(),
    };
  }

  // --- Lifecycle ---

  /**
   * Probes environment, validates critical browser requirements, launches Wasm64 or
   * falls back to the WebGPU G5 smoke demo, and wires HUD/Input/Audio/Telemetry.
   *
   * @param {{ preferSmoke?: boolean }} [options]
   */
  async start(options = {}) {
    if (this._destroyed) {
      return {
        ok: false,
        destroyed: true,
        state: this._state,
        mode: this._mode,
        bootMode: this._bootMode,
        sideModuleCount: this._sideModuleCount,
        sideModuleUrls: [...this._sideModuleUrls],
        error: new Error('RuntimeSessionController has been destroyed'),
        diagnostics: this.getDiagnostics(),
      };
    }

    if (this._restartPromise) {
      return this._restartPromise;
    }

    return this._startTracked(options);
  }

  async _startTracked(options = {}) {
    const preferSmoke = Boolean(options && options.preferSmoke);

    if (this._destroyed) {
      return {
        ok: false,
        destroyed: true,
        state: this._state,
        mode: this._mode,
        bootMode: this._bootMode,
        sideModuleCount: this._sideModuleCount,
        sideModuleUrls: [...this._sideModuleUrls],
        error: new Error('RuntimeSessionController has been destroyed'),
        diagnostics: this.getDiagnostics(),
      };
    }

    if (this._startPromise) {
      return this._startPromise;
    }

    if (
      this._state === SESSION_STATE.RUNNING_WASM ||
      this._state === SESSION_STATE.RUNNING_SMOKE
    ) {
      return {
        ok: true,
        alreadyRunning: true,
        state: this._state,
        mode: this._mode,
        bootMode: this._bootMode,
        sideModuleCount: this._sideModuleCount,
        sideModuleUrls: [...this._sideModuleUrls],
        fallbackReason: this._fallbackReason,
        capabilities: this._capabilities ? { ...this._capabilities } : null,
        serverStatus: this._cloneServerStatus(),
        handle: this._activeHandle,
        diagnostics: this.getDiagnostics(),
      };
    }

    const token = ++this._sessionToken;
    const promise = this._startInternal({ preferSmoke, token }).finally(() => {
      if (this._startPromise === promise) {
        this._startPromise = null;
      }
    });
    this._startPromise = promise;
    return promise;
  }

  async _startInternal({ preferSmoke, token }) {
    this._lastError = null;
    this._diagnosticReasons = [];
    this._fallbackReason = null;
    this._mode = SESSION_MODE.NONE;
    this._bootMode = BOOT_MODE.IDLE;
    this._startedAtMs = null;
    this._deviceLost = false;

    this._validationErrorCount = 0;
    this._validationErrorSeq = 0;
    this._recentValidationErrors = [];
    this._splashFramesPresented = 0;
    this._splashFrameErrorCount = 0;
    this._firstSplashPresented = false;
    this._lastSplashFrame = null;
    this._resetSessionPhases(true);

    if (this._state === SESSION_STATE.ERROR) {
      this._recoveryCount += 1;
    }

    this._transition(SESSION_STATE.PROBING, {
      reason: 'Probing browser capabilities and server readiness',
    });
    this._hudStatus('Probing environment...', 'info');

    const probe = await this.probeEnvironment();
    if (token !== this._sessionToken) {
      return {
        ok: false,
        aborted: true,
        state: this._state,
        mode: this._mode,
        bootMode: this._bootMode,
        sideModuleCount: this._sideModuleCount,
        sideModuleUrls: [...this._sideModuleUrls],
        diagnostics: this.getDiagnostics(),
      };
    }

    const { capabilities, serverStatus, missingCapabilities, diagnosticReasons } = probe;

    if (missingCapabilities.length > 0) {
      const error = new Error(`Critical browser requirements not met: ${diagnosticReasons.join(' ')}`);
      error.code = 'MISSING_BROWSER_CAPABILITIES';
      error.missingCapabilities = [...missingCapabilities];
      error.reasons = [...diagnosticReasons];

      this._lastError = error;
      this._diagnosticReasons = [...diagnosticReasons];
      this._mode = SESSION_MODE.NONE;
      this._bootMode = BOOT_MODE.IDLE;
      this._finalizeCurrentPhase();

      const shortStatus =
        missingCapabilities.includes('crossOriginIsolated') ||
        missingCapabilities.includes('sharedArrayBuffer')
          ? 'Missing COOP/COEP'
          : 'WebGPU unavailable';
      this._hudStatus(shortStatus, 'error');
      for (const reason of diagnosticReasons) {
        this._hudLog(reason, 'error');
      }

      this._reportError(error, {
        phase: 'probe',
        code: error.code,
        missingCapabilities: [...missingCapabilities],
        reasons: [...diagnosticReasons],
        capabilities: { ...capabilities },
      });

      this._transition(SESSION_STATE.ERROR, {
        error,
        reasons: [...diagnosticReasons],
        missingCapabilities: [...missingCapabilities],
      });
      this._emitStatusUpdate('missing-capabilities');

      return {
        ok: false,
        state: this._state,
        mode: this._mode,
        bootMode: this._bootMode,
        sideModuleCount: this._sideModuleCount,
        sideModuleUrls: [...this._sideModuleUrls],
        error,
        reasons: [...diagnosticReasons],
        capabilities: { ...capabilities },
        serverStatus: this._cloneServerStatus(),
        diagnostics: this.getDiagnostics(),
      };
    }

    // Decide runtime mode: Wasm64 when available & not overridden, else WebGPU G5 smoke demo.
    const canLaunchWasm =
      Boolean(serverStatus.hasWasmBinary) &&
      !preferSmoke &&
      typeof this._startWasmRuntime === 'function';
    const canLaunchSmoke = typeof this._startSmokeDemo === 'function';

    let targetMode = SESSION_MODE.NONE;
    let fallbackReason = null;

    if (canLaunchWasm) {
      targetMode = SESSION_MODE.WASM;
    } else if (canLaunchSmoke) {
      targetMode = SESSION_MODE.SMOKE;
      if (preferSmoke) {
        fallbackReason = 'Smoke demo selected (preferSmoke=true); starting WebGPU G5 smoke demo.';
      } else if (serverStatus.offline) {
        fallbackReason =
          'Server /api/status is unreachable; falling back to WebGPU G5 smoke demo.';
      } else if (!serverStatus.hasWasmBinary) {
        fallbackReason =
          'No Wasm64 runtime binary built yet (hasWasmBinary=false); falling back to WebGPU G5 smoke demo.';
      } else {
        fallbackReason =
          'Wasm64 runtime launcher not configured; falling back to WebGPU G5 smoke demo.';
      }
    }

    const targetBootMode =
      targetMode === SESSION_MODE.WASM
        ? BOOT_MODE.REAL_WASM64
        : targetMode === SESSION_MODE.SMOKE
          ? BOOT_MODE.SMOKE
          : BOOT_MODE.IDLE;

    this._fallbackReason = fallbackReason;

    this._transition(SESSION_STATE.STARTING, {
      targetMode,
      targetBootMode,
      fallbackReason,
    });

    if (targetMode === SESSION_MODE.NONE) {
      const error = new Error(
        preferSmoke
          ? 'Smoke demo requested (preferSmoke=true), but no startSmokeDemo launcher is configured.'
          : 'No runtime launcher available: Wasm64 binary/launcher unavailable and no startSmokeDemo fallback configured.'
      );
      error.code = 'NO_RUNTIME_LAUNCHER';
      this._lastError = error;
      this._diagnosticReasons = [error.message];
      this._bootMode = BOOT_MODE.IDLE;
      this._finalizeCurrentPhase();

      this._hudStatus('No runtime launcher', 'error');
      this._hudLog(error.message, 'error');
      this._reportError(error, {
        phase: 'starting',
        code: error.code,
        preferSmoke,
        serverStatus: this._cloneServerStatus(),
      });

      this._transition(SESSION_STATE.ERROR, { error, reasons: [error.message] });
      this._emitStatusUpdate('no-runtime-launcher');
      return {
        ok: false,
        state: this._state,
        mode: this._mode,
        bootMode: this._bootMode,
        sideModuleCount: this._sideModuleCount,
        sideModuleUrls: [...this._sideModuleUrls],
        error,
        reasons: [error.message],
        capabilities: { ...capabilities },
        serverStatus: this._cloneServerStatus(),
        diagnostics: this.getDiagnostics(),
      };
    }

    if (targetMode === SESSION_MODE.WASM) {
      this._hudStatus('Starting Wasm64 runtime...', 'info');
      this._hudLog('Server reports Wasm64 runtime binary ready; launching Wasm64 runtime.', 'info');
      if (this._sideModuleCount > 0) {
        this._hudLog(
          `Split-module Wasm64 layout detected (${this._sideModuleCount} side module${this._sideModuleCount === 1 ? '' : 's'}).`,
          'info'
        );
      }
    } else {
      this._hudStatus('Starting smoke demo...', 'info');
      this._hudLog(fallbackReason, preferSmoke ? 'info' : 'warn');
    }

    let callbackReportedBootMode = false;
    const launchContext = {
      mode: targetMode,
      bootMode: targetBootMode,
      sideModuleCount: this._sideModuleCount,
      sideModuleUrls: [...this._sideModuleUrls],
      capabilities: { ...capabilities },
      serverStatus: this._cloneServerStatus(),
      storageSummary: this._storageSummary,
      hud: this._hud,
      inputBridge: this._inputBridge,
      audioEngine: this._audioEngine,
      storageManager: this._storageManager,
      perfMonitor: this._perfMonitor,
      crashReporter: this._crashReporter,
      bootOrchestrator: this._bootOrchestrator,
      onBootLifecycle: (meta) => {
        if (token !== this._sessionToken) {
          return this.getSideModuleMetadata();
        }
        if (meta && typeof meta === 'object' && normalizeBootMode(meta.bootMode || meta.mode)) {
          callbackReportedBootMode = true;
        }
        return this.notifyBootLifecycle(meta);
      },
      onWorkerMessage: (msg) => {
        if (token !== this._sessionToken) {
          return { handled: false, type: null };
        }
        return this.handleWorkerMessage(msg);
      },
      onValidationError: (errPayload) => {
        if (token !== this._sessionToken) {
          return { handled: false, type: 'gpu-validation-error' };
        }
        return this.recordGpuValidationError(errPayload);
      },
      onSplashFramePresented: (splashPayload) => {
        if (token !== this._sessionToken) {
          return { handled: false, type: 'splash-frame-presented' };
        }
        return this.recordSplashFramePresented(splashPayload);
      },
      onDeviceLost: (reason) => {
        if (token !== this._sessionToken) {
          return { handled: false, type: 'gpu-device-lost' };
        }
        return this.recordDeviceLost(reason);
      },
      onDeviceRecovered: (detail) => {
        if (token !== this._sessionToken) {
          return { handled: false, type: 'gpu-device-recovered' };
        }
        return this.recordDeviceRecovered(detail);
      },
      recordPhase: (phase, timestampMs) => {
        if (token !== this._sessionToken) {
          return {
            ok: false,
            changed: false,
            phase: null,
            previousPhase: this._currentPhase,
            currentPhase: this._currentPhase,
            durationInPrevMs: 0,
            phaseDurationsMs: this.getPhaseDurations(),
            timestampMs: this._readNowMs(),
          };
        }
        return this.recordPhase(phase, timestampMs);
      },
      registerReclaimHook: (hook) => {
        if (typeof hook !== 'function') {
          throw new TypeError('registerReclaimHook requires a hook function');
        }
        if (token !== this._sessionToken || this._destroyed) {
          return () => {};
        }
        const unregister = this.registerReclaimHook(hook);
        this._sessionReclaimUnregisters.add(unregister);
        return () => {
          this._sessionReclaimUnregisters.delete(unregister);
          unregister();
        };
      },
      requestMemoryReclaim: (reason) => {
        if (token !== this._sessionToken || this._destroyed) {
          return Promise.resolve({
            ok: false,
            aborted: true,
            destroyed: this._destroyed,
            reason: sanitizeTransitionReason(reason) || 'memory-pressure',
            hooksInvoked: 0,
            hooksSucceeded: 0,
            hooksFailed: 0,
            errors: [],
            reclaimRuns: this._reclaimRuns,
            reclaimFailures: this._reclaimFailures,
            timestampMs: this._readNowMs(),
          });
        }
        return this.requestMemoryReclaim(reason);
      },
    };

    const launcher =
      targetMode === SESSION_MODE.WASM ? this._startWasmRuntime : this._startSmokeDemo;

    let handle = null;
    try {
      handle = await launcher(launchContext);
    } catch (err) {
      if (token !== this._sessionToken) {
        return {
          ok: false,
          aborted: true,
          state: this._state,
          mode: this._mode,
          bootMode: this._bootMode,
          sideModuleCount: this._sideModuleCount,
          sideModuleUrls: [...this._sideModuleUrls],
          diagnostics: this.getDiagnostics(),
        };
      }
      this._releasePreloadBuffers(null);
      const normalized = toError(err, `Failed to start ${targetMode} session`);
      this._lastError = normalized;
      this._diagnosticReasons = [normalized.message];
      this._mode = SESSION_MODE.NONE;
      this._bootMode = BOOT_MODE.IDLE;
      this._finalizeCurrentPhase();

      this._hudStatus('Runtime start error', 'error');
      this._hudLog(`Failed to start ${targetMode} runtime: ${normalized.message}`, 'error');
      this._reportError(normalized, {
        phase: 'starting',
        targetMode,
        targetBootMode,
        fallbackReason,
      });

      this._transition(SESSION_STATE.ERROR, {
        error: normalized,
        targetMode,
        reasons: [normalized.message],
      });
      this._emitStatusUpdate('launcher-error');

      return {
        ok: false,
        state: this._state,
        mode: this._mode,
        bootMode: this._bootMode,
        sideModuleCount: this._sideModuleCount,
        sideModuleUrls: [...this._sideModuleUrls],
        error: normalized,
        reasons: [normalized.message],
        capabilities: { ...capabilities },
        serverStatus: this._cloneServerStatus(),
        diagnostics: this.getDiagnostics(),
      };
    }

    // If stop() was called while awaiting launcher, dispose the late handle cleanly.
    if (token !== this._sessionToken) {
      await this._stopHandle(handle);
      this._releasePreloadBuffers(handle);
      return {
        ok: false,
        aborted: true,
        state: this._state,
        mode: this._mode,
        bootMode: this._bootMode,
        sideModuleCount: this._sideModuleCount,
        sideModuleUrls: [...this._sideModuleUrls],
        diagnostics: this.getDiagnostics(),
      };
    }

    this._activeHandle = handle ?? null;

    if (handle && typeof handle === 'object') {
      const handleSideMeta = extractSideModuleMetadata(handle);
      if (handleSideMeta.hasSideModuleUrls) {
        this._sideModuleUrls = [...handleSideMeta.sideModuleUrls];
        this._sideModuleCount = handleSideMeta.sideModuleCount;
      } else if (handleSideMeta.hasSideModuleCount) {
        this._sideModuleCount = handleSideMeta.sideModuleCount;
      }
      if (!callbackReportedBootMode) {
        const handleBootMode = normalizeBootMode(handle.bootMode);
        if (handleBootMode) {
          this._bootMode = handleBootMode;
          callbackReportedBootMode = true;
        }
      }
    }

    try {
      this._wireSubsystems();
    } catch (err) {
      const normalized = toError(err, 'Failed to wire runtime subsystems');
      this._teardownSubsystems();
      const handleToStop = this._activeHandle;
      this._activeHandle = null;
      await this._stopHandle(handleToStop);
      this._releasePreloadBuffers(handleToStop);

      this._lastError = normalized;
      this._diagnosticReasons = [normalized.message];
      this._mode = SESSION_MODE.NONE;
      this._bootMode = BOOT_MODE.IDLE;
      this._finalizeCurrentPhase();

      this._hudStatus('Subsystem error', 'error');
      this._hudLog(`Subsystem initialization failed: ${normalized.message}`, 'error');
      this._reportError(normalized, {
        phase: 'starting',
        source: 'subsystems',
        targetMode,
      });

      this._transition(SESSION_STATE.ERROR, {
        error: normalized,
        reasons: [normalized.message],
      });
      this._emitStatusUpdate('subsystem-error');

      return {
        ok: false,
        state: this._state,
        mode: this._mode,
        bootMode: this._bootMode,
        sideModuleCount: this._sideModuleCount,
        sideModuleUrls: [...this._sideModuleUrls],
        error: normalized,
        reasons: [normalized.message],
        capabilities: { ...capabilities },
        serverStatus: this._cloneServerStatus(),
        diagnostics: this.getDiagnostics(),
      };
    }

    this._mode = targetMode;
    if (!callbackReportedBootMode || this._bootMode === BOOT_MODE.IDLE) {
      this._bootMode = targetBootMode;
    }

    const runningState =
      targetMode === SESSION_MODE.WASM
        ? SESSION_STATE.RUNNING_WASM
        : SESSION_STATE.RUNNING_SMOKE;

    this._hudStatus(
      targetMode === SESSION_MODE.WASM ? 'Running (Wasm64)' : 'Running (Smoke Demo)',
      'ok'
    );
    this._transition(runningState, {
      mode: targetMode,
      bootMode: this._bootMode,
      fallbackReason,
    });
    this._emitStatusUpdate('running');

    return {
      ok: true,
      state: this._state,
      mode: this._mode,
      bootMode: this._bootMode,
      sideModuleCount: this._sideModuleCount,
      sideModuleUrls: [...this._sideModuleUrls],
      fallbackReason: this._fallbackReason,
      capabilities: { ...capabilities },
      serverStatus: this._cloneServerStatus(),
      handle: this._activeHandle,
      diagnostics: this.getDiagnostics(),
    };
  }

  /**
   * Idempotently stops the active runtime or smoke demo handle, detaches PerfMonitor,
   * stops InputBridge, removes AudioEngine user-gesture listeners, releases preloaded
   * boot buffers (if available on orchestrator/handle), and transitions to `stopped`.
   */
  async stop() {
    this._externalStopToken += 1;
    return this._stopInternal();
  }

  async _stopInternal() {
    if (this._stopPromise) {
      const firstResult = await this._stopPromise;
      return {
        ...firstResult,
        alreadyStopped: true,
        state: this._state,
        mode: this._mode,
        bootMode: this._bootMode,
        sideModuleCount: this._sideModuleCount,
        sideModuleUrls: [...this._sideModuleUrls],
        diagnostics: this.getDiagnostics(),
      };
    }

    // Invalidate any in-flight start() immediately so a subsequent start()/restart() never
    // coalesces onto an aborted start promise.
    const hadInFlightStart = Boolean(this._startPromise);
    this._sessionToken += 1;
    this._startPromise = null;

    const handleToStop = this._activeHandle;
    this._activeHandle = null;

    const wasAlreadyStopped =
      this._state === SESSION_STATE.STOPPED && !hadInFlightStart && !handleToStop;

    const runStop = async () => {
      const subsystemsClean = this._teardownSubsystems();
      const handleClean = await this._stopHandle(handleToStop);
      const buffersClean = wasAlreadyStopped ? true : this._releasePreloadBuffers(handleToStop);

      this._mode = SESSION_MODE.NONE;
      this._bootMode = BOOT_MODE.IDLE;
      this._startedAtMs = null;
      this._fallbackReason = null;
      this._finalizeCurrentPhase();

      if (this._state === SESSION_STATE.STOPPED) {
        return {
          ok: true,
          alreadyStopped: true,
          state: this._state,
          mode: this._mode,
          bootMode: this._bootMode,
          sideModuleCount: this._sideModuleCount,
          sideModuleUrls: [...this._sideModuleUrls],
          diagnostics: this.getDiagnostics(),
        };
      }

      if (subsystemsClean && handleClean && buffersClean) {
        this._cleanShutdownCount += 1;
      }

      this._hudStatus('Stopped', 'info');
      this._transition(SESSION_STATE.STOPPED, { reason: 'Session stopped' });
      this._emitStatusUpdate('stopped');

      return {
        ok: true,
        alreadyStopped: false,
        state: this._state,
        mode: this._mode,
        bootMode: this._bootMode,
        sideModuleCount: this._sideModuleCount,
        sideModuleUrls: [...this._sideModuleUrls],
        diagnostics: this.getDiagnostics(),
      };
    };

    const promise = runStop().finally(() => {
      if (this._stopPromise === promise) {
        this._stopPromise = null;
      }
    });
    this._stopPromise = promise;
    return promise;
  }

  /**
   * Cleanly stops any active session, releases preloaded boot buffers, and restarts
   * with the provided options without leaking listeners, timers, or runtime handles.
   * Overlapping concurrent calls are coalesced onto the in-flight restart so clicking
   * Restart rapidly never spawns duplicate workers or doubles renderer RAM.
   *
   * @param {{ preferSmoke?: boolean, skipReclaim?: boolean }} [options]
   */
  async restart(options = {}) {
    if (this._destroyed) {
      return {
        ok: false,
        destroyed: true,
        state: this._state,
        mode: this._mode,
        bootMode: this._bootMode,
        sideModuleCount: this._sideModuleCount,
        sideModuleUrls: [...this._sideModuleUrls],
        error: new Error('RuntimeSessionController has been destroyed'),
        diagnostics: this.getDiagnostics(),
      };
    }

    if (this._restartPromise) {
      this._coalescedRestarts += 1;
      return this._restartPromise;
    }

    this._restartCount += 1;
    const externalStopBefore = this._externalStopToken;

    const promise = this._restartInternal(options, externalStopBefore).finally(() => {
      if (this._restartPromise === promise) {
        this._restartPromise = null;
      }
    });
    this._restartPromise = promise;
    return promise;
  }

  async _restartInternal(options, externalStopBefore) {
    if (this._reclaimHooks.size > 0 && !(options && options.skipReclaim === true)) {
      await this.requestMemoryReclaim('restart');
    }

    if (this._destroyed || this._externalStopToken !== externalStopBefore) {
      return {
        ok: false,
        aborted: true,
        ...(this._destroyed ? { destroyed: true } : {}),
        state: this._state,
        mode: this._mode,
        bootMode: this._bootMode,
        sideModuleCount: this._sideModuleCount,
        sideModuleUrls: [...this._sideModuleUrls],
        diagnostics: this.getDiagnostics(),
      };
    }

    await this._stopInternal();

    if (this._destroyed || this._externalStopToken !== externalStopBefore) {
      return {
        ok: false,
        aborted: true,
        ...(this._destroyed ? { destroyed: true } : {}),
        state: this._state,
        mode: this._mode,
        bootMode: this._bootMode,
        sideModuleCount: this._sideModuleCount,
        sideModuleUrls: [...this._sideModuleUrls],
        diagnostics: this.getDiagnostics(),
      };
    }

    return this._startTracked(options);
  }

  /**
   * Permanently tears down the session controller: stops any active runtime handle,
   * detaches subsystems and worker listeners, releases preloaded boot buffers, clears
   * all observer references, and blocks subsequent start/restart calls. Idempotent.
   */
  async destroy() {
    if (this._destroyed) {
      if (this._destroyPromise) {
        await this._destroyPromise;
      }
      return {
        ok: true,
        destroyed: true,
        alreadyDestroyed: true,
        alreadyStopped: true,
        state: this._state,
        mode: this._mode,
        bootMode: this._bootMode,
        sideModuleCount: this._sideModuleCount,
        sideModuleUrls: [...this._sideModuleUrls],
        diagnostics: this.getDiagnostics(),
      };
    }

    this._destroyed = true;
    this._externalStopToken += 1;

    const runDestroy = async () => {
      const stopResult = await this._stopInternal();
      if (stopResult.alreadyStopped) {
        this._releasePreloadBuffers(null);
      }

      this._stateListeners.clear();
      this._statusListeners.clear();
      this._telemetryListeners.clear();
      this._validationErrorListeners.clear();
      this._splashListeners.clear();
      this._deviceLostListeners.clear();
      this._deviceRecoveredListeners.clear();
      this._reclaimHooks.clear();
      this._sessionReclaimUnregisters.clear();

      this._recentValidationErrors.length = 0;
      this._lastSplashFrame = null;
      this._bootOrchestrator = null;
      this._wgpuWorker = null;
      this._audioGestureTarget = undefined;

      return {
        ...stopResult,
        ok: true,
        destroyed: true,
        alreadyDestroyed: false,
      };
    };

    const promise = runDestroy().finally(() => {
      if (this._destroyPromise === promise) {
        this._destroyPromise = null;
      }
    });
    this._destroyPromise = promise;
    return promise;
  }

  /**
   * Alias for `destroy()` that releases retained preload buffers and tears down the controller.
   */
  async dispose() {
    return this.destroy();
  }
}

export default RuntimeSessionController;
