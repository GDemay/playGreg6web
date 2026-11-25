// Interactive browser HUD diagnostics & controls controller for playgta6web.
// Binds HUD actions (OPFS cache clear, userdata save backup/restore, GPU .pg6trace
// recording/export, sanitized crash report export, input profile switching, and
// smoke demo / runtime session controls) cleanly to BrowserStorageManager,
// GpuFrameRecorder / SmokeDemoRunner, CrashReporter, InputProfileManager,
// PerfMonitor, and RuntimeSessionController.
// Zero external dependencies; ES module compatible with Browser and Node.js.

import { formatBytes, SNAPSHOT_FORMAT } from '../storage/storage_manager.js';
import { SESSION_STATE } from '../runtime/session_controller.js';
import { sanitizeString } from '../telemetry/crash_report.js';

export const BOOT_MODE = Object.freeze({
  IDLE: 'idle',
  SMOKE: 'smoke',
  REAL_WASM64: 'real-wasm64',
});

export const AUDIO_OUTPUT_STATE = Object.freeze({
  ACTIVE: 'active',
  SUSPENDED: 'suspended',
  HEADLESS_DRAIN: 'headless-drain',
});

export const INDIRECT_DISPATCH_MODE = Object.freeze({
  SHARD: 'shard',
  TABLE: 'table',
  UNKNOWN: 'unknown',
});

export const DOMINANT_BOTTLENECK = Object.freeze({
  WITHIN_BUDGET: 'within_budget',
  IO_BOUND: 'io_bound',
  SHADER_COMPILE_BOUND: 'shader_compile_bound',
  GPU_BOUND: 'gpu_bound',
  CPU_BOUND: 'cpu_bound',
});

const DOMINANT_BOTTLENECK_SET = new Set(Object.values(DOMINANT_BOTTLENECK));

const BYTES_PER_MIB = 1024 * 1024;

const FRAME_BUDGET_TELEMETRY_FIELD_KEYS = Object.freeze([
  'fps30ComplianceRate',
  'fps30_compliance_rate',
  'complianceRate30Fps',
  'compliance_rate_30fps',
  'fps30Compliance',
  'fps30_compliance',
  'dominantBottleneck',
  'dominant_bottleneck',
  'bottleneck',
]);

const MEMORY_TELEMETRY_FIELD_KEYS = Object.freeze([
  'jsHeapUsedMb',
  'js_heap_used_mb',
  'heapUsedMb',
  'heap_used_mb',
  'jsHeapTotalMb',
  'js_heap_total_mb',
  'heapTotalMb',
  'heap_total_mb',
  'usedJSHeapSize',
  'totalJSHeapSize',
  'jsHeapUsedBytes',
  'js_heap_used_bytes',
  'jsHeapTotalBytes',
  'js_heap_total_bytes',
]);

const IO_TELEMETRY_FIELD_KEYS = Object.freeze([
  'mailboxLanes',
  'mailbox_lanes',
  'ioMailboxLanes',
  'io_mailbox_lanes',
]);

const GPU_TELEMETRY_FIELD_KEYS = Object.freeze([
  'realPackDrawCount',
  'real_pack_draw_count',
  'realDrawCount',
  'packDrawCount',
  'fallbackDrawCount',
  'fallback_draw_count',
  'presentCount',
  'present_count',
  'presents',
  'framesPresented',
  'frames_presented',
  'unboundBindGroupCount',
  'unbound_bind_group_count',
  'skippedDrawCount',
  'skipped_draw_count',
  'validationErrorCount',
  'validation_error_count',
  'onePercentLowFps',
  'one_percent_low_fps',
  'fps1PercentLow',
  'low1PercentFps',
  'p99FrameMs',
  'p99_frame_ms',
  'p99FrameTimeMs',
  'p99_frame_time_ms',
  'zeroCopyUploads',
  'zero_copy_uploads',
  'binkZeroCopyUploads',
  'bink_zero_copy_uploads',
  'zeroCopyFramesForwarded',
  'zero_copy_frames_forwarded',
  'zeroCopyFrames',
  'binkZeroCopyFrames',
  'gpuAdapter',
  'gpu_adapter',
  'adapterLabel',
  'adapter_label',
  'adapterInfo',
  'adapter_info',
  'blackLoadingMs',
  'black_loading_ms',
  'blackScreenMs',
  'black_screen_ms',
  'firstVisibleFrameMs',
  'first_visible_frame_ms',
  'firstVisibleMs',
  'first_visible_ms',
  'timeToFirstVisibleFrameMs',
  'time_to_first_visible_frame_ms',
  'loadingAssets',
  'loading_assets',
  'isLoadingAssets',
  'is_loading_assets',
]);

const GUEST_AUDIO_TELEMETRY_FIELD_KEYS = Object.freeze([
  'guestRingAttached',
  'guest_ring_attached',
  'ringAttached',
  'pcmRingAttached',
  'guestFramesDrained',
  'guest_frames_drained',
  'guestDrainedFrames',
  'guest_drained_frames',
  'guestUnderruns',
  'guest_underruns',
  'guestUnderrunCount',
  'guest_underrun_count',
  'guestPeakLevel',
  'guest_peak_level',
  'guestPeak',
  'guest_peak',
]);

const AUDIO_TELEMETRY_FIELD_KEYS = Object.freeze([
  'outputState',
  'output_state',
  'framesDrained',
  'frames_drained',
  'drainedFrames',
  'underruns',
  'underrunCount',
  'underrun_count',
  'overruns',
  'overrunCount',
  'overrun_count',
  'overflows',
  ...GUEST_AUDIO_TELEMETRY_FIELD_KEYS,
]);

const PTHREAD_TELEMETRY_FIELD_KEYS = Object.freeze([
  'pthreadSpawnedCount',
  'pthread_spawned_count',
  'pthreadActiveCount',
  'pthread_active_count',
  'pthreadPeakCount',
  'pthread_peak_count',
  'pthreadLoadQueueDepth',
  'pthread_load_queue_depth',
  'indirectDispatch',
  'indirect_dispatch',
  'dispatchMode',
  'dispatch_mode',
]);

export const DEFAULT_MAX_VALIDATION_ERRORS = 32;
export const MAX_VALIDATION_ERROR_MESSAGE_LENGTH = 256;
export const MAX_GPU_ADAPTER_LABEL_LENGTH = 96;
export const DEFAULT_TOGGLE_KEY = 'F3';
export const DEFAULT_MIN_RENDER_INTERVAL_MS = 0;
const SCRIPTED_INPUT_PRESET_RE = /^[a-z0-9][a-z0-9_-]{0,31}$/i;
const MAX_SIDE_MODULE_NAMES = 64;
const SIDE_MODULE_LAYOUT_RE =
  /\bsplit-module\b[^()\r\n]*?\(\s*(\d+)\s+side\s+modules?\s*\)/i;
const SIDE_MODULE_LOAD_RE =
  /\b(?:loading|loaded)\s+wasm\s+side\s+module\s*\(([^)\r\n]+)\)|\bwasm\s+side\s+module\s+loaded\s*\(([^)\r\n]+)\)/i;
const RUNTIME_STARTING_RE =
  /\bloading\s+emscripten\s+wasm64\s+runtime\b|\blaunching\s+wasm64\s+runtime\b/i;
const RUNTIME_INIT_RE =
  /\b(?:playgta6web\s+)?wasm64\s+runtime\s+initialized\b/i;
const POST_INIT_BOOT_STAGE_RE =
  /\[(?:boot|wasm)\][^\r\n]*\b(TLS_INIT|CRT_INITTERM|ENTRY_CALL|XMEM_INIT|GDK_INIT|D3D12_CREATE_DEVICE|BINK_OPEN|FIRST_PRESENT)\b/i;
const PROGRESS_LOG_FAILURE_RE = /\b(?:fail(?:ed|s|ure)?|error|abort(?:ed)?)\b/i;
const BOOT_MILESTONES = Object.freeze([
  'PGIM_LOAD',
  'TLS_INIT',
  'CRT_INITTERM',
  'ENTRY_CALL',
  'XMEM_INIT',
  'GDK_INIT',
  'D3D12_CREATE_DEVICE',
  'BINK_OPEN',
  'FIRST_PRESENT',
]);
const BOOT_MILESTONE_INDEX = new Map(BOOT_MILESTONES.map((name, idx) => [name, idx]));
const BOOT_STATUS_MILESTONE_RE = new RegExp(
  `\\bBoot\\s+(\\d+(?:\\.\\d+)?)%\\s*:\\s*(${BOOT_MILESTONES.join('|')})\\b`,
  'i'
);
const BOOT_MILESTONE_TOKEN_RE = new RegExp(`\\b(${BOOT_MILESTONES.join('|')})\\b`, 'gi');
const FPS_WINDOW_MS = 2000;
const MAX_FPS_SAMPLES = 32;
const MAX_FPS_CLOCK_JUMP_MS = 60000;

const POST_INIT_BOOT_STAGES = Object.freeze(
  new Set([
    'TLS_INIT',
    'CRT_INITTERM',
    'ENTRY_CALL',
    'XMEM_INIT',
    'GDK_INIT',
    'D3D12_CREATE_DEVICE',
    'BINK_OPEN',
    'FIRST_PRESENT',
  ])
);

const BOOT_MODE_META = Object.freeze({
  [BOOT_MODE.REAL_WASM64]: Object.freeze({
    badgeText: 'Real Wasm64',
    badgeState: 'ok',
    statusLevel: 'ok',
    baseStatusText: 'Real Wasm64 Guest',
  }),
  [BOOT_MODE.SMOKE]: Object.freeze({
    badgeText: 'Smoke Demo',
    badgeState: 'warn',
    statusLevel: 'warn',
    baseStatusText: 'Smoke Demo (Synthetic WebGPU — Not Real Game)',
  }),
  [BOOT_MODE.IDLE]: Object.freeze({
    badgeText: 'Idle',
    badgeState: 'info',
    statusLevel: 'info',
    baseStatusText: 'Idle',
  }),
});

const SUPPORTED_DISPLAY_ELEMENTS = Object.freeze([
  'hudRootEl',
  'rootEl',
  'hudRoot',
  'scriptedInputBadge',
  'scriptedInputBadgeEl',
  'scriptedInputStatusEl',
  'bootModeBadge',
  'bootStatusEl',
  'sideModuleStatusEl',
  'validationCountEl',
  'validationLatestEl',
  'validationBannerEl',
  'framesEl',
  'fpsEl',
  'pillFpsEl',
  'onePercentLowFpsEl',
  'p99FrameMsEl',
  'zeroCopyUploadsEl',
  'binkZeroCopyEl',
  'bootStageEl',
  'bootProgressBarEl',
  'bootProgressFillEl',
  'bootElapsedEl',
  'gpuAdapterEl',
  'gpuAdapterBadgeEl',
  'loadingAssetsBadgeEl',
  'loadingAssetsEl',
  'blackLoadingEl',
  'firstVisibleFrameEl',
  'audioStatusEl',
  'guestAudioStatusEl',
  'frameBudgetEl',
  'fps30BudgetEl',
  'dominantBottleneckEl',
  'bottleneckEl',
  'memoryStatsEl',
  'jsHeapEl',
  'ioMailboxLanesEl',
  'mailboxLanesEl',
  'diagnosticsSummaryEl',
  'diagnosticsPanelEl',
  'diagnosticsContainerEl',
]);

export const HUD_ACTIONS = Object.freeze({
  CLEAR_OPFS_CACHE: 'clear_opfs_cache',
  EXPORT_SAVES: 'export_saves',
  IMPORT_SAVES: 'import_saves',
  TOGGLE_GPU_RECORDING: 'toggle_gpu_recording',
  EXPORT_GPU_TRACE: 'export_gpu_trace',
  EXPORT_CRASH_REPORT: 'export_crash_report',
  EXPORT_MISSING_SHADERS: 'export_missing_shaders',
  SWITCH_INPUT_PROFILE: 'switch_input_profile',
  TOGGLE_SMOKE_DEMO: 'toggle_smoke_demo',
  RESTART_SESSION: 'restart_session',
});

export const DEFAULT_FILENAMES = Object.freeze({
  USERDATA_BACKUP: 'playgta6-userdata-backup.json',
  GPU_TRACE: 'playgta6-capture.pg6trace',
  CRASH_REPORT: 'playgta6-diagnostic-report.json',
  MISSING_SHADERS: 'playgta6-missing-shaders.json',
});

const SUPPORTED_BUTTON_BINDINGS = Object.freeze([
  ['clearCacheBtn', 'click'],
  ['exportSavesBtn', 'click'],
  ['recordTraceBtn', 'click'],
  ['exportTraceBtn', 'click'],
  ['exportCrashBtn', 'click'],
  ['exportMissingShadersBtn', 'click'],
  ['restartBtn', 'click'],
  ['smokeDemoBtn', 'click'],
  ['profileSelect', 'change'],
]);

function toErrorMessage(err, fallback = 'Unknown error') {
  if (err instanceof Error && typeof err.message === 'string' && err.message.length > 0) {
    return err.message;
  }
  if (typeof err === 'string' && err.length > 0) {
    return err;
  }
  if (err && typeof err === 'object' && typeof err.message === 'string' && err.message.length > 0) {
    return err.message;
  }
  return fallback;
}

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function toUint8Array(data) {
  if (data instanceof Uint8Array) {
    return data;
  }
  if (ArrayBuffer.isView(data)) {
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  }
  if (data instanceof ArrayBuffer) {
    return new Uint8Array(data);
  }
  throw new TypeError('Expected binary trace data (Uint8Array, ArrayBufferView, or ArrayBuffer)');
}

function createBrowserDownloadBlob() {
  if (
    typeof document === 'undefined' ||
    !document ||
    typeof document.createElement !== 'function' ||
    typeof Blob !== 'function' ||
    typeof URL === 'undefined' ||
    !URL ||
    typeof URL.createObjectURL !== 'function'
  ) {
    return null;
  }

  return function defaultBrowserDownloadBlob(filename, data, mimeType = 'application/octet-stream') {
    const blob = new Blob([data], { type: mimeType });
    const url = URL.createObjectURL(blob);
    try {
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = filename;
      anchor.style.display = 'none';
      if (document.body && typeof document.body.appendChild === 'function') {
        document.body.appendChild(anchor);
      }
      anchor.click();
      if (anchor.parentNode && typeof anchor.parentNode.removeChild === 'function') {
        anchor.parentNode.removeChild(anchor);
      }
    } finally {
      if (typeof URL.revokeObjectURL === 'function') {
        URL.revokeObjectURL(url);
      }
    }
  };
}

function createBrowserCopyToClipboard() {
  if (
    typeof navigator !== 'undefined' &&
    navigator &&
    navigator.clipboard &&
    typeof navigator.clipboard.writeText === 'function'
  ) {
    return (text) => navigator.clipboard.writeText(String(text));
  }
  return null;
}

function validateEventTarget(el, label) {
  if (
    !el ||
    typeof el !== 'object' ||
    typeof el.addEventListener !== 'function' ||
    typeof el.removeEventListener !== 'function'
  ) {
    throw new TypeError(
      `bindButtons("${label}") requires an EventTarget with addEventListener/removeEventListener`
    );
  }
}

/**
 * Normalizes a boot-mode string or alias into one of:
 * `'real-wasm64'`, `'smoke'`, or `'idle'`.
 * Unknown values always fall back to `'idle'` so smoke or unverified states are
 * never mislabeled as real recompiled game frames.
 *
 * @param {unknown} raw
 * @returns {'real-wasm64'|'smoke'|'idle'}
 */
export function normalizeBootMode(raw) {
  if (typeof raw !== 'string') {
    return BOOT_MODE.IDLE;
  }
  const normalized = raw.trim().toLowerCase();
  if (!normalized) {
    return BOOT_MODE.IDLE;
  }
  if (
    normalized === 'real-wasm64' ||
    normalized === 'real_wasm64' ||
    normalized === 'real-game' ||
    normalized === 'real_game' ||
    normalized === 'wasm-guest' ||
    normalized === 'wasm_guest' ||
    normalized === 'wasm64' ||
    normalized === 'wasm' ||
    normalized === 'running_wasm'
  ) {
    return BOOT_MODE.REAL_WASM64;
  }
  if (
    normalized === 'smoke' ||
    normalized === 'smoke-demo' ||
    normalized === 'smoke_demo' ||
    normalized === 'running_smoke' ||
    normalized === 'synthetic'
  ) {
    return BOOT_MODE.SMOKE;
  }
  return BOOT_MODE.IDLE;
}

function classifyGpuValidationErrorType(raw) {
  if (typeof raw !== 'string') {
    return null;
  }
  const normalized = raw.trim().toLowerCase();
  if (!normalized) {
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
  if (normalized === 'device-lost' || normalized === 'device_lost' || normalized.includes('lost')) {
    return 'device-lost';
  }
  if (normalized === 'internal' || normalized.includes('internal')) {
    return 'internal';
  }
  if (normalized === 'validation' || normalized.includes('validation')) {
    return 'validation';
  }
  return null;
}

function normalizeValidationErrorType(record) {
  if (!record || typeof record !== 'object') {
    return 'validation';
  }
  const errObj = record.error && typeof record.error === 'object' ? record.error : null;
  const candidates = [
    record.type,
    record.errorType,
    record.category,
    record.name,
    record.constructor && record.constructor.name,
    errObj && errObj.type,
    errObj && errObj.name,
    errObj && errObj.constructor && errObj.constructor.name,
  ];
  for (const candidate of candidates) {
    const classified = classifyGpuValidationErrorType(candidate);
    if (classified) {
      return classified;
    }
  }
  return 'validation';
}

/**
 * Sanitizes a live WebGPU validation error message for HUD display:
 * strips control chars, host filesystem paths, file:// URLs, query strings,
 * auth tokens, and embedded binary/hex/base64 dumps, then caps length.
 *
 * @param {unknown} raw
 * @param {number} [maxLength=MAX_VALIDATION_ERROR_MESSAGE_LENGTH]
 * @returns {string}
 */
export function sanitizeValidationErrorMessage(
  raw,
  maxLength = MAX_VALIDATION_ERROR_MESSAGE_LENGTH
) {
  const limit =
    typeof maxLength === 'number' && Number.isFinite(maxLength) && maxLength >= 1
      ? Math.floor(maxLength)
      : MAX_VALIDATION_ERROR_MESSAGE_LENGTH;

  let text = '';
  if (typeof raw === 'string') {
    text = raw;
  } else if (raw instanceof Error) {
    text = typeof raw.message === 'string' && raw.message.length > 0 ? raw.message : raw.name;
  } else if (raw && typeof raw === 'object') {
    if (typeof raw.message === 'string' && raw.message.length > 0) {
      text = raw.message;
    } else if (raw.error && typeof raw.error === 'object' && typeof raw.error.message === 'string') {
      text = raw.error.message;
    } else if (typeof raw.error === 'string' && raw.error.length > 0) {
      text = raw.error;
    } else if (typeof raw.reason === 'string' && raw.reason.length > 0) {
      text = raw.reason;
    }
  } else if (raw !== null && raw !== undefined) {
    text = String(raw);
  }

  // Normalize whitespace and remove control characters before path/token redaction.
  text = text.replace(/[\r\n\t]+/g, ' ').replace(/[\x00-\x1F\x7F]+/g, '');
  text = sanitizeString(text);

  // Strip embedded data URIs, byte arrays, long hex dumps, and base64 blobs.
  text = text.replace(/\bdata:[^\s;,'"`<>]+;base64,[A-Za-z0-9+/=]{16,}/gi, '[REDACTED_BUFFER]');
  text = text.replace(
    /(?:\b(?:0x)?[0-9a-fA-F]{1,4}\s*,\s*){16,}(?:0x)?[0-9a-fA-F]{1,4}\b/g,
    '[REDACTED_BUFFER]'
  );
  text = text.replace(/(?:\b[0-9a-fA-F]{2}\s+){16,}[0-9a-fA-F]{2}\b/g, '[REDACTED_BUFFER]');
  text = text.replace(/\b(?:0x)?[0-9a-fA-F]{48,}\b/g, '[REDACTED_BUFFER]');
  text = text.replace(
    /(^|[\s"'`(\[=,:])[A-Za-z0-9+/]{64,}={0,2}(?=$|[\s"'`)\]=,.;:])/g,
    '$1[REDACTED_BUFFER]'
  );

  // Also strip any remaining multi-segment POSIX host paths missed by known root lists.
  text = text.replace(
    /(^|[\s"'`(\[=,:])\/(?:[A-Za-z0-9._-]+\/){2,}[A-Za-z0-9._~+-]+/g,
    '$1[REDACTED_PATH]'
  );

  text = text.replace(/\s{2,}/g, ' ').trim();
  if (!text) {
    text = 'WebGPU validation error';
  }

  if (text.length > limit) {
    if (limit <= 3) {
      return text.slice(0, limit);
    }
    return `${text.slice(0, limit - 3)}...`;
  }
  return text;
}

function toNonNegativeInt(value, fallback = 0) {
  if (value === undefined || value === null || value === '') {
    return fallback;
  }
  if (typeof value === 'boolean' || typeof value === 'symbol' || typeof value === 'object') {
    return fallback;
  }
  if (typeof value === 'bigint') {
    if (value < 0n) {
      return fallback;
    }
    return value > BigInt(Number.MAX_SAFE_INTEGER) ? Number.MAX_SAFE_INTEGER : Number(value);
  }
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n) || n < 0) {
    return fallback;
  }
  return Math.min(Number.MAX_SAFE_INTEGER, Math.floor(n));
}

function toNonNegativeFiniteNumber(value, fallback = null) {
  if (value === undefined || value === null || value === '') {
    return fallback;
  }
  if (typeof value === 'boolean' || typeof value === 'symbol' || typeof value === 'object') {
    return fallback;
  }
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n) || n < 0) {
    return fallback;
  }
  return n;
}

function toUnitIntervalNumber(value, fallback = 0) {
  const n = toNonNegativeFiniteNumber(value, null);
  if (n === null) {
    return fallback;
  }
  return Math.min(1, n);
}

/**
 * Sanitizes a 30 FPS frame-budget compliance rate into a finite number clamped
 * to `[0, 1]`, returning `null` for non-finite, missing, or non-numeric inputs.
 *
 * @param {unknown} raw
 * @returns {number|null}
 */
export function sanitizeFps30ComplianceRate(raw) {
  if (raw === undefined || raw === null || raw === '') {
    return null;
  }
  if (
    typeof raw === 'boolean' ||
    typeof raw === 'symbol' ||
    typeof raw === 'object' ||
    typeof raw === 'bigint'
  ) {
    return null;
  }
  const n = typeof raw === 'number' ? raw : Number(raw);
  if (!Number.isFinite(n)) {
    return null;
  }
  const clamped = Math.min(1, Math.max(0, n));
  return clamped === 0 ? 0 : clamped;
}

/**
 * Strictly allowlists a dominant frame-budget bottleneck label to one of:
 * `'within_budget' | 'io_bound' | 'shader_compile_bound' | 'gpu_bound' | 'cpu_bound'`,
 * returning `null` for any unknown, empty, or hostile input.
 *
 * @param {unknown} raw
 * @returns {'within_budget'|'io_bound'|'shader_compile_bound'|'gpu_bound'|'cpu_bound'|null}
 */
export function normalizeDominantBottleneck(raw) {
  if (typeof raw !== 'string') {
    return null;
  }
  const normalized = raw.trim().toLowerCase().replace(/[\s-]+/g, '_');
  if (!normalized || !DOMINANT_BOTTLENECK_SET.has(normalized)) {
    return null;
  }
  return normalized;
}

/**
 * Sanitizes an HTTPFS I/O mailbox lane count to a positive integer (`>= 1`),
 * returning `null` for zero, negative, non-finite, or invalid inputs.
 *
 * @param {unknown} raw
 * @returns {number|null}
 */
export function sanitizeMailboxLanes(raw) {
  if (raw === undefined || raw === null || raw === '') {
    return null;
  }
  if (typeof raw === 'boolean' || typeof raw === 'symbol' || typeof raw === 'object') {
    return null;
  }
  if (typeof raw === 'bigint') {
    if (raw < 1n) {
      return null;
    }
    return raw > BigInt(Number.MAX_SAFE_INTEGER) ? Number.MAX_SAFE_INTEGER : Number(raw);
  }
  const n = typeof raw === 'number' ? raw : Number(raw);
  if (!Number.isFinite(n) || n < 1) {
    return null;
  }
  return Math.min(Number.MAX_SAFE_INTEGER, Math.floor(n));
}

function formatRoundedMbValue(mb) {
  const rounded = Math.round(mb * 10) / 10;
  return Number.isInteger(rounded) ? `${rounded}` : rounded.toFixed(1);
}

function formatCompliancePct(rate) {
  const pct = Math.round(rate * 1000) / 10;
  return Number.isInteger(pct) ? `${pct}%` : `${pct.toFixed(1)}%`;
}

function resolveFrameBudgetState(rate, bottleneck) {
  if (rate === null && bottleneck === null) {
    return '';
  }
  if (rate !== null && rate < 0.75) {
    return 'err';
  }
  if (
    (rate !== null && rate < 0.95) ||
    (bottleneck !== null && bottleneck !== DOMINANT_BOTTLENECK.WITHIN_BUDGET)
  ) {
    return 'warn';
  }
  return 'ok';
}

function resolveMemoryState(usedMb, totalMb) {
  if (usedMb === null && totalMb === null) {
    return '';
  }
  if (usedMb !== null && usedMb >= 1024) {
    return 'warn';
  }
  return 'ok';
}

function resolveIoMailboxState(lanes) {
  if (lanes === null) {
    return '';
  }
  return lanes > 1 ? 'ok' : 'warn';
}

function formatFrameBudgetValueText(rate, bottleneck) {
  if (rate !== null && bottleneck !== null) {
    return `${formatCompliancePct(rate)} (${bottleneck})`;
  }
  if (rate !== null) {
    return formatCompliancePct(rate);
  }
  if (bottleneck !== null) {
    return bottleneck;
  }
  return '';
}

function formatMemoryValueText(usedMb, totalMb) {
  if (usedMb !== null && totalMb !== null) {
    return `${formatRoundedMbValue(usedMb)} / ${formatRoundedMbValue(totalMb)} MB`;
  }
  if (usedMb !== null) {
    return `${formatRoundedMbValue(usedMb)} MB`;
  }
  if (totalMb !== null) {
    return `-- / ${formatRoundedMbValue(totalMb)} MB`;
  }
  return '';
}

/**
 * Formats 30 FPS frame-budget compliance and dominant bottleneck for Diagnostics
 * display (e.g. `'30 FPS budget: 96.4% (gpu_bound)'`). Returns `''` when unset.
 *
 * @param {object|null|undefined} input
 * @returns {string}
 */
export function formatFrameBudgetStatus(input) {
  if (!isPlainObject(input)) {
    return '';
  }
  const source = isPlainObject(safeGet(input, 'frameBudget'))
    ? safeGet(input, 'frameBudget')
    : input;
  const rate = sanitizeFps30ComplianceRate(
    pickFirstDefined(source, [
      'fps30ComplianceRate',
      'fps30_compliance_rate',
      'complianceRate30Fps',
      'compliance_rate_30fps',
      'fps30Compliance',
      'fps30_compliance',
    ])
  );
  const bottleneck = normalizeDominantBottleneck(
    pickFirstDefined(source, ['dominantBottleneck', 'dominant_bottleneck', 'bottleneck'])
  );
  const valueText = formatFrameBudgetValueText(rate, bottleneck);
  return valueText ? `30 FPS budget: ${valueText}` : '';
}

/**
 * Formats JS heap memory usage in MB for Diagnostics display
 * (e.g. `'JS heap: 256 / 512 MB'`, `'JS heap: 42.5 MB'`). Returns `''` when unset.
 *
 * @param {object|null|undefined} input
 * @returns {string}
 */
export function formatMemoryStats(input) {
  if (!isPlainObject(input)) {
    return '';
  }
  const source = isPlainObject(safeGet(input, 'memory')) ? safeGet(input, 'memory') : input;
  const usedMb = toNonNegativeFiniteNumber(
    pickFirstDefined(source, ['jsHeapUsedMb', 'js_heap_used_mb', 'heapUsedMb', 'heap_used_mb']),
    null
  );
  const totalMb = toNonNegativeFiniteNumber(
    pickFirstDefined(source, ['jsHeapTotalMb', 'js_heap_total_mb', 'heapTotalMb', 'heap_total_mb']),
    null
  );
  const valueText = formatMemoryValueText(usedMb, totalMb);
  return valueText ? `JS heap: ${valueText}` : '';
}

/**
 * Formats HTTPFS I/O mailbox lane count for Diagnostics display
 * (e.g. `'I/O lanes: 4'`). Returns `''` when unset or invalid.
 *
 * @param {unknown} lanes
 * @returns {string}
 */
export function formatIoMailboxLanes(lanes) {
  const raw = isPlainObject(lanes)
    ? pickFirstDefined(lanes, [
        'mailboxLanes',
        'mailbox_lanes',
        'ioMailboxLanes',
        'io_mailbox_lanes',
        'lanes',
      ])
    : lanes;
  const count = sanitizeMailboxLanes(raw);
  return count !== null ? `I/O lanes: ${count}` : '';
}

/**
 * Formats a concise combined Diagnostics line covering 30 FPS budget, JS heap MB,
 * and HTTPFS mailbox lanes, cleanly omitting any unset metrics.
 *
 * @param {object|null|undefined} input
 * @returns {string}
 */
export function formatDiagnosticsSummary(input) {
  if (!isPlainObject(input)) {
    return '';
  }
  const parts = [];
  const budgetText = formatFrameBudgetStatus(input);
  if (budgetText) {
    parts.push(budgetText);
  }
  const memoryText = formatMemoryStats(input);
  if (memoryText) {
    parts.push(memoryText);
  }
  const ioSource = isPlainObject(safeGet(input, 'io')) ? safeGet(input, 'io') : input;
  const lanesText = formatIoMailboxLanes(ioSource);
  if (lanesText) {
    parts.push(lanesText);
  }
  return parts.join(' · ');
}

/**
 * Sanitizes an active WebGPU adapter label (or adapter info object) into a safe,
 * lowercase vendor/architecture label (e.g. `'nvidia / turing'`, `'intel / gen-12lp'`),
 * capped at `maxLength` (default 96 chars) with zero host paths, URLs, scripts,
 * control characters, or secrets. Returns `null` when empty or wholly invalid.
 *
 * @param {unknown} raw
 * @param {number} [maxLength=MAX_GPU_ADAPTER_LABEL_LENGTH]
 * @returns {string|null}
 */
export function sanitizeGpuAdapterLabel(raw, maxLength = MAX_GPU_ADAPTER_LABEL_LENGTH) {
  const limit =
    typeof maxLength === 'number' && Number.isFinite(maxLength) && maxLength >= 1
      ? Math.min(MAX_GPU_ADAPTER_LABEL_LENGTH, Math.floor(maxLength))
      : MAX_GPU_ADAPTER_LABEL_LENGTH;

  let text = '';
  if (typeof raw === 'string') {
    text = raw;
  } else if (isPlainObject(raw)) {
    const nestedInfo = safeGet(raw, 'info');
    const source = isPlainObject(nestedInfo) ? nestedInfo : raw;
    const rawVendor = safeGet(source, 'vendor');
    const rawArch = safeGet(source, 'architecture') ?? safeGet(source, 'arch');
    const parts = [];
    if (typeof rawVendor === 'string' && rawVendor.trim().length > 0) {
      parts.push(rawVendor.trim());
    }
    if (typeof rawArch === 'string' && rawArch.trim().length > 0) {
      const archTrimmed = rawArch.trim();
      if (
        parts.length === 0 ||
        parts[0].toLowerCase() !== archTrimmed.toLowerCase()
      ) {
        parts.push(archTrimmed);
      }
    }
    if (parts.length > 0) {
      text = parts.join(' / ');
    } else {
      const fallback = pickFirstDefined(source, [
        'gpuAdapter',
        'gpu_adapter',
        'adapterLabel',
        'adapter_label',
        'label',
        'description',
        'device',
        'name',
      ]);
      if (typeof fallback === 'string') {
        text = fallback;
      } else {
        return null;
      }
    }
  } else {
    return null;
  }

  // Strip control characters and collapse whitespace before redaction.
  text = text.replace(/[\r\n\t]+/g, ' ').replace(/[\x00-\x1F\x7F]+/g, '');
  if (!text.trim()) {
    return null;
  }

  // Strip HTML/script blocks, URLs, and inline query/secret tokens.
  text = text.replace(/<script\b[\s\S]*?<\/script>/gi, ' ');
  text = text.replace(/<[^>]*>/g, ' ');
  text = text.replace(/\b(?:https?|file|data|javascript|ftp|wss?):\/\/[^\s"'`<>)\]},;]*/gi, ' ');
  text = text.replace(/\b(?:javascript|data):[^\s"'`<>)\]},;]*/gi, ' ');

  // Redact host filesystem paths, file:// URLs, tokens, and binary blobs.
  text = sanitizeString(text);
  text = text.replace(/\??\[REDACTED(?:_[A-Z0-9_]+)?\]/gi, ' ');

  // Strip any remaining Windows backslash paths, path traversal, or multi-segment POSIX paths.
  text = text.replace(/(?:^|\s)[^\s]*\\[^\s]*/g, ' ');
  text = text.replace(/\.\.(?:[\\/][^\s]*)?/g, ' ');
  text = text.replace(/(^|[\s"'`(\[=,:])\/(?:[A-Za-z0-9._-]+\/)+[A-Za-z0-9._~+-]*/g, '$1 ');
  text = text.replace(/(^|[\s"'`(\[=,:])\/[A-Za-z0-9._-]+\/[A-Za-z0-9._~+-]+/g, '$1 ');
  text = text.replace(/[?#][^\s]*/g, ' ');

  text = text.toLowerCase();
  text = text.replace(/\(\s*\)|\[\s*\]/g, ' ');
  text = text.replace(/[^a-z0-9 /._:+(),-]+/g, ' ');
  text = text.replace(/\s*\/\s*/g, ' / ');
  text = text.replace(/(?:\s*\/\s*){2,}/g, ' / ');
  text = text.replace(/\s{2,}/g, ' ').trim();
  text = text.replace(/^[\/\s,;:-]+|[\/\s,;:-]+$/g, '').trim();

  if (!text) {
    return null;
  }

  if (text.length > limit) {
    text = text
      .slice(0, limit)
      .replace(/[\/\s,;:-]+$/g, '')
      .trim();
  }

  return text || null;
}

function formatDurationShortMs(ms) {
  const value = toNonNegativeFiniteNumber(ms, null);
  if (value === null) {
    return '';
  }
  if (value >= 1000) {
    return `${(value / 1000).toFixed(1)}s`;
  }
  return `${Math.round(value)}ms`;
}

/**
 * Formats a sanitized WebGPU adapter label for HUD display (e.g. `'GPU: nvidia / turing'`).
 * Returns `''` when absent or invalid.
 *
 * @param {unknown} label
 * @returns {string}
 */
export function formatGpuAdapter(label) {
  const safe = sanitizeGpuAdapterLabel(label);
  return safe ? `GPU: ${safe}` : '';
}

/**
 * Formats a black-loading duration in milliseconds for HUD display
 * (e.g. `'Black loading: 3.5s'`, `'Black loading: 450ms'`). Returns `''` when absent.
 *
 * @param {unknown} ms
 * @returns {string}
 */
export function formatBlackLoadingMs(ms) {
  const formatted = formatDurationShortMs(ms);
  return formatted ? `Black loading: ${formatted}` : '';
}

/**
 * Formats time-to-first-visible-frame in milliseconds for HUD display
 * (e.g. `'First visible frame: 4.1s'`). Returns `''` when absent.
 *
 * @param {unknown} ms
 * @returns {string}
 */
export function formatFirstVisibleFrameMs(ms) {
  const formatted = formatDurationShortMs(ms);
  return formatted ? `First visible frame: ${formatted}` : '';
}

/**
 * Formats the combined loading-assets / black-loading / first-visible-frame
 * indicator text for HUD display. Returns `''` when none are set.
 *
 * @param {object|null|undefined} input
 * @returns {string}
 */
export function formatLoadingStatus(input) {
  if (!isPlainObject(input)) {
    return '';
  }
  const source = isPlainObject(safeGet(input, 'gpu')) ? safeGet(input, 'gpu') : input;
  const loadingAssets = safeGet(source, 'loadingAssets') === true;
  const blackMs = toNonNegativeFiniteNumber(safeGet(source, 'blackLoadingMs'), null);
  const firstMs = toNonNegativeFiniteNumber(safeGet(source, 'firstVisibleFrameMs'), null);

  if (loadingAssets) {
    const dur = formatDurationShortMs(blackMs);
    return dur ? `Loading assets (${dur})` : 'Loading assets';
  }
  if (firstMs !== null) {
    return `First visible frame: ${formatDurationShortMs(firstMs)}`;
  }
  if (blackMs !== null) {
    return `Black loading: ${formatDurationShortMs(blackMs)}`;
  }
  return '';
}

/**
 * Formats guest audio PCM ring status (attachment, drained frames, underruns,
 * and normalized peak level percentage) for HUD display.
 *
 * @param {object|null|undefined} input
 * @returns {string}
 */
export function formatGuestAudioStatus(input) {
  if (!isPlainObject(input)) {
    return '';
  }
  const source = isPlainObject(safeGet(input, 'audio')) ? safeGet(input, 'audio') : input;
  const attached = safeGet(source, 'guestRingAttached') === true;
  const drained = toNonNegativeInt(safeGet(source, 'guestFramesDrained'), 0);
  const underruns = toNonNegativeInt(safeGet(source, 'guestUnderruns'), 0);
  const peak = toUnitIntervalNumber(safeGet(source, 'guestPeakLevel'), 0);

  const details = [];
  if (drained > 0) {
    details.push(`${drained} drained`);
  }
  if (underruns > 0) {
    details.push(`${underruns} underrun${underruns === 1 ? '' : 's'}`);
  }
  if (peak > 0) {
    details.push(`peak ${Math.round(peak * 100)}%`);
  }

  const stateLabel = attached ? 'attached' : 'detached';
  return details.length > 0
    ? `Guest PCM: ${stateLabel} (${details.join(', ')})`
    : `Guest PCM: ${stateLabel}`;
}

/**
 * Formats an FPS number for HUD display matching `web/index.html` (`0.0` for 0,
 * one decimal below 10 FPS, rounded integer at or above 10 FPS).
 *
 * @param {unknown} fps
 * @returns {string}
 */
export function formatFpsDisplay(fps) {
  const value = toNonNegativeFiniteNumber(fps, 0);
  return value >= 10 ? Math.round(value).toString() : value.toFixed(1);
}

/**
 * Formats a 1%-low FPS telemetry value for HUD display (e.g. `'1% low: 54 fps'`,
 * `'1% low: 8.5 fps'`). Returns `''` when absent or invalid.
 *
 * @param {unknown} fps
 * @returns {string}
 */
export function formatOnePercentLowFps(fps) {
  const value = toNonNegativeFiniteNumber(fps, null);
  if (value === null) {
    return '';
  }
  return `1% low: ${formatFpsDisplay(value)} fps`;
}

/**
 * Formats a p99 frame time in milliseconds for HUD display (e.g. `'p99: 18.4 ms'`).
 * Returns `''` when absent or invalid.
 *
 * @param {unknown} frameMs
 * @returns {string}
 */
export function formatP99FrameMs(frameMs) {
  const value = toNonNegativeFiniteNumber(frameMs, null);
  if (value === null) {
    return '';
  }
  return `p99: ${value.toFixed(1)} ms`;
}

/**
 * Formats a Bink2 zero-copy texture upload counter for HUD display
 * (e.g. `'Bink2 zero-copy: 120'`). Returns `''` when absent or invalid.
 *
 * @param {unknown} uploads
 * @returns {string}
 */
export function formatZeroCopyUploads(uploads) {
  const value = toNonNegativeInt(uploads, null);
  if (value === null) {
    return '';
  }
  return `Bink2 zero-copy: ${value}`;
}

/**
 * Formats a combined single-line summary of optional frame-pacing & Bink2
 * zero-copy telemetry (`onePercentLowFps`, `p99FrameMs`, `zeroCopyUploads`),
 * cleanly omitting any unset or invalid metrics.
 *
 * @param {object|null|undefined} metrics
 * @returns {string}
 */
export function formatPerfMetricsSummary(metrics) {
  if (!isPlainObject(metrics)) {
    return '';
  }
  const source = isPlainObject(safeGet(metrics, 'gpu')) ? safeGet(metrics, 'gpu') : metrics;
  const parts = [];
  const lowText = formatOnePercentLowFps(
    pickFirstDefined(source, [
      'onePercentLowFps',
      'one_percent_low_fps',
      'fps1PercentLow',
      'low1PercentFps',
    ])
  );
  if (lowText) {
    parts.push(lowText);
  }
  const p99Text = formatP99FrameMs(
    pickFirstDefined(source, ['p99FrameMs', 'p99_frame_ms', 'p99FrameTimeMs', 'p99_frame_time_ms'])
  );
  if (p99Text) {
    parts.push(p99Text);
  }
  const zeroCopyText = formatZeroCopyUploads(
    pickFirstDefined(source, [
      'zeroCopyUploads',
      'zero_copy_uploads',
      'binkZeroCopyUploads',
      'bink_zero_copy_uploads',
      'zeroCopyFramesForwarded',
      'zero_copy_frames_forwarded',
      'zeroCopyFrames',
      'binkZeroCopyFrames',
    ])
  );
  if (zeroCopyText) {
    parts.push(zeroCopyText);
  }
  return parts.join(' · ');
}

function formatElapsedSeconds(ms) {
  const clamped = typeof ms === 'number' && Number.isFinite(ms) && ms > 0 ? ms : 0;
  return `${(clamped / 1000).toFixed(1)}s`;
}

function normalizeBootMilestone(raw) {
  if (typeof raw !== 'string') {
    return null;
  }
  const normalized = raw.trim().toUpperCase();
  if (!normalized || !BOOT_MILESTONE_INDEX.has(normalized)) {
    return null;
  }
  return normalized;
}

function defaultMilestoneProgressPct(stage) {
  const idx = BOOT_MILESTONE_INDEX.get(stage);
  if (idx === undefined) {
    return 0;
  }
  return Math.round(((idx + 1) / BOOT_MILESTONES.length) * 100);
}

function extractBootMilestoneFromText(raw) {
  if (typeof raw !== 'string') {
    return null;
  }
  const text = raw.trim();
  if (!text || PROGRESS_LOG_FAILURE_RE.test(text)) {
    return null;
  }
  const statusMatch = BOOT_STATUS_MILESTONE_RE.exec(text);
  if (statusMatch) {
    const stage = normalizeBootMilestone(statusMatch[2]);
    if (stage) {
      const pct = toNonNegativeFiniteNumber(statusMatch[1], null);
      return {
        stage,
        progressPct: pct !== null ? Math.min(100, pct) : defaultMilestoneProgressPct(stage),
      };
    }
  }
  let highestStage = null;
  let highestIdx = -1;
  BOOT_MILESTONE_TOKEN_RE.lastIndex = 0;
  let m;
  while ((m = BOOT_MILESTONE_TOKEN_RE.exec(text)) !== null) {
    const stage = normalizeBootMilestone(m[1]);
    if (stage) {
      const idx = BOOT_MILESTONE_INDEX.get(stage);
      if (idx > highestIdx) {
        highestIdx = idx;
        highestStage = stage;
      }
    }
  }
  if (!highestStage) {
    return null;
  }
  return {
    stage: highestStage,
    progressPct: defaultMilestoneProgressPct(highestStage),
  };
}

function safeGet(obj, key) {
  if (!obj || typeof obj !== 'object') {
    return undefined;
  }
  try {
    return obj[key];
  } catch {
    return undefined;
  }
}

function pickFirstDefined(obj, keys) {
  if (!obj || typeof obj !== 'object') {
    return undefined;
  }
  for (let i = 0; i < keys.length; i += 1) {
    const val = safeGet(obj, keys[i]);
    if (val !== undefined) {
      return val;
    }
  }
  return undefined;
}

function pickTelemetryField(primary, secondary, keys) {
  const fromPrimary = pickFirstDefined(primary, keys);
  if (fromPrimary !== undefined) {
    return fromPrimary;
  }
  if (secondary && secondary !== primary) {
    return pickFirstDefined(secondary, keys);
  }
  return undefined;
}

function hasAnyDefinedKey(obj, keys) {
  return pickFirstDefined(obj, keys) !== undefined;
}

/**
 * Normalizes an AudioContext/mixer output state into one of:
 * `'active'`, `'suspended'`, or `'headless-drain'`.
 * Unknown or hostile values safely fall back to `'suspended'`.
 *
 * @param {unknown} raw
 * @returns {'active'|'suspended'|'headless-drain'}
 */
export function normalizeAudioOutputState(raw) {
  if (typeof raw !== 'string') {
    return AUDIO_OUTPUT_STATE.SUSPENDED;
  }
  const normalized = raw.trim().toLowerCase();
  if (!normalized) {
    return AUDIO_OUTPUT_STATE.SUSPENDED;
  }
  if (normalized === 'active' || normalized === 'running' || normalized === 'playing') {
    return AUDIO_OUTPUT_STATE.ACTIVE;
  }
  if (
    normalized === 'headless-drain' ||
    normalized === 'headless_drain' ||
    normalized === 'headless' ||
    normalized === 'drain' ||
    normalized === 'draining'
  ) {
    return AUDIO_OUTPUT_STATE.HEADLESS_DRAIN;
  }
  if (
    normalized === 'suspended' ||
    normalized === 'paused' ||
    normalized === 'closed' ||
    normalized === 'idle' ||
    normalized === 'muted'
  ) {
    return AUDIO_OUTPUT_STATE.SUSPENDED;
  }
  return AUDIO_OUTPUT_STATE.SUSPENDED;
}

/**
 * Normalizes a Wasm64 indirect-call dispatch mode into one of:
 * `'shard'`, `'table'`, or `'unknown'`.
 * Unknown or hostile values safely fall back to `'unknown'`.
 *
 * @param {unknown} raw
 * @returns {'shard'|'table'|'unknown'}
 */
export function normalizeIndirectDispatchMode(raw) {
  if (typeof raw !== 'string') {
    return INDIRECT_DISPATCH_MODE.UNKNOWN;
  }
  const normalized = raw.trim().toLowerCase();
  if (!normalized) {
    return INDIRECT_DISPATCH_MODE.UNKNOWN;
  }
  if (
    normalized === 'shard' ||
    normalized === 'sharded' ||
    normalized === 'shard-dispatch' ||
    normalized === 'shard_dispatch' ||
    normalized === 'pg6-shard' ||
    normalized === 'pg6_shard'
  ) {
    return INDIRECT_DISPATCH_MODE.SHARD;
  }
  if (
    normalized === 'table' ||
    normalized === 'indirect-table' ||
    normalized === 'indirect_table' ||
    normalized === 'wasm-table' ||
    normalized === 'wasm_table'
  ) {
    return INDIRECT_DISPATCH_MODE.TABLE;
  }
  return INDIRECT_DISPATCH_MODE.UNKNOWN;
}

function createInitialGpuTelemetry() {
  return {
    realPackDrawCount: 0,
    fallbackDrawCount: 0,
    presentCount: 0,
    unboundBindGroupCount: 0,
    skippedDrawCount: 0,
    validationErrorCount: 0,
    onePercentLowFps: null,
    p99FrameMs: null,
    zeroCopyUploads: null,
    gpuAdapter: null,
    blackLoadingMs: null,
    firstVisibleFrameMs: null,
    loadingAssets: false,
    updatedAtMs: null,
  };
}

function createInitialAudioTelemetry() {
  return {
    outputState: AUDIO_OUTPUT_STATE.SUSPENDED,
    framesDrained: 0,
    underruns: 0,
    overruns: 0,
    guestRingAttached: false,
    guestFramesDrained: 0,
    guestUnderruns: 0,
    guestPeakLevel: 0,
    updatedAtMs: null,
  };
}

function createInitialPthreadTelemetry() {
  return {
    pthreadSpawnedCount: 0,
    pthreadActiveCount: 0,
    pthreadPeakCount: 0,
    pthreadLoadQueueDepth: 0,
    indirectDispatch: INDIRECT_DISPATCH_MODE.UNKNOWN,
    updatedAtMs: null,
  };
}

function createInitialFrameBudgetTelemetry() {
  return {
    fps30ComplianceRate: null,
    dominantBottleneck: null,
    updatedAtMs: null,
  };
}

function createInitialMemoryTelemetry() {
  return {
    jsHeapUsedMb: null,
    jsHeapTotalMb: null,
    updatedAtMs: null,
  };
}

function createInitialIoTelemetry() {
  return {
    mailboxLanes: null,
    updatedAtMs: null,
  };
}

function sanitizeSideModuleName(raw) {
  if (typeof raw !== 'string') {
    return null;
  }
  let candidate = raw.trim();
  if (!candidate) {
    return null;
  }
  // Strip query strings / fragments before extracting basename.
  const qIdx = candidate.search(/[?#]/);
  if (qIdx !== -1) {
    candidate = candidate.slice(0, qIdx);
  }
  // Extract only the final path segment (works for both POSIX and Windows separators).
  const lastSlash = Math.max(candidate.lastIndexOf('/'), candidate.lastIndexOf('\\'));
  if (lastSlash !== -1) {
    candidate = candidate.slice(lastSlash + 1);
  }
  candidate = candidate.trim();
  if (!candidate || candidate === '.' || candidate === '..') {
    return null;
  }
  if (!/^[A-Za-z0-9._-]{1,64}$/.test(candidate)) {
    return null;
  }
  return candidate;
}

/**
 * Parses loader/session progress log labels emitted on `BroadcastChannel('game-progress')`
 * or `hud.appendLog(...)` to detect split-module layout announcements, individual
 * `guest_part_*.wasm` loads, runtime restarts, and Wasm64 runtime initialization.
 *
 * @param {unknown} raw
 * @returns {
 *   | { kind: 'layout', sideModuleCount: number }
 *   | { kind: 'module', moduleName: string }
 *   | { kind: 'runtime_initialized' }
 *   | { kind: 'runtime_starting' }
 *   | null
 * }
 */
export function parseSideModuleProgressLabel(raw) {
  if (typeof raw !== 'string') {
    return null;
  }
  const text = raw.trim();
  if (!text) {
    return null;
  }

  const moduleMatch = SIDE_MODULE_LOAD_RE.exec(text);
  if (moduleMatch) {
    const rawName = moduleMatch[1] || moduleMatch[2] || '';
    const moduleName = sanitizeSideModuleName(rawName);
    if (moduleName) {
      return { kind: 'module', moduleName };
    }
    return null;
  }

  const layoutMatch = SIDE_MODULE_LAYOUT_RE.exec(text);
  if (layoutMatch) {
    return {
      kind: 'layout',
      sideModuleCount: toNonNegativeInt(layoutMatch[1], 0),
    };
  }

  if (!PROGRESS_LOG_FAILURE_RE.test(text)) {
    if (RUNTIME_INIT_RE.test(text) || POST_INIT_BOOT_STAGE_RE.test(text)) {
      return { kind: 'runtime_initialized' };
    }
  }

  if (RUNTIME_STARTING_RE.test(text)) {
    return { kind: 'runtime_starting' };
  }

  return null;
}

/**
 * Dirty-checked DOM text write helper. Skips writing `el.textContent` when the
 * element already holds the exact target string, preventing unnecessary browser
 * style/layout invalidation churn. Optionally updates `{ domWrites, domWritesElided }`.
 *
 * @param {object|null|undefined} el
 * @param {unknown} text
 * @param {{ domWrites: number, domWritesElided: number }|null} [stats=null]
 * @returns {boolean} `true` if a DOM write occurred, `false` if elided or unavailable.
 */
export function setTextIfChanged(el, text, stats = null) {
  if (!el || typeof el !== 'object') return false;
  const next = String(text ?? '');
  let hasCurrent = false;
  let current;
  try {
    current = el.textContent;
    hasCurrent = typeof current === 'string';
  } catch {
    hasCurrent = false;
  }
  if (hasCurrent && current === next) {
    if (stats) stats.domWritesElided += 1;
    return false;
  }
  try {
    el.textContent = next;
    if (stats) stats.domWrites += 1;
    return true;
  } catch {
    // Ignore read-only mock properties.
    return false;
  }
}

function safeSetText(el, text, stats = null) {
  return setTextIfChanged(el, text, stats);
}

function safeSetTitle(el, title, stats = null) {
  if (!el || typeof el !== 'object') return false;
  const next = String(title ?? '');
  let hasCurrent = false;
  let current;
  try {
    current = el.title;
    hasCurrent = typeof current === 'string';
  } catch {
    hasCurrent = false;
  }
  if (hasCurrent && current === next) {
    if (stats) stats.domWritesElided += 1;
    return false;
  }
  try {
    el.title = next;
    if (stats) stats.domWrites += 1;
    return true;
  } catch {
    // Ignore read-only mock properties.
    return false;
  }
}

function safeSetHidden(el, hidden, stats = null) {
  if (!el || typeof el !== 'object') return false;
  const next = Boolean(hidden);
  let hasCurrent = false;
  let current;
  try {
    current = el.hidden;
    hasCurrent = typeof current === 'boolean';
  } catch {
    hasCurrent = false;
  }
  if (hasCurrent && current === next) {
    if (stats) stats.domWritesElided += 1;
    return false;
  }
  try {
    el.hidden = next;
    if (stats) stats.domWrites += 1;
    return true;
  } catch {
    // Ignore read-only mock properties.
    return false;
  }
}

function hasDatasetKey(el, key) {
  if (!el || typeof el !== 'object') return false;
  try {
    return Boolean(
      el.dataset &&
        typeof el.dataset === 'object' &&
        Object.prototype.hasOwnProperty.call(el.dataset, key) &&
        el.dataset[key] !== undefined
    );
  } catch {
    return false;
  }
}

function safeSetDataset(el, key, value, stats = null) {
  if (!el || typeof el !== 'object') return false;
  try {
    if (el.dataset && typeof el.dataset === 'object') {
      if (value === '' || value === null || value === undefined) {
        if (
          !Object.prototype.hasOwnProperty.call(el.dataset, key) ||
          el.dataset[key] === undefined
        ) {
          if (stats) stats.domWritesElided += 1;
          return false;
        }
        delete el.dataset[key];
        if (stats) stats.domWrites += 1;
        return true;
      }
      const next = String(value);
      if (el.dataset[key] === next) {
        if (stats) stats.domWritesElided += 1;
        return false;
      }
      el.dataset[key] = next;
      if (stats) stats.domWrites += 1;
      return true;
    }
  } catch {
    // Ignore DOM dataset update errors on lightweight mocks.
  }
  return false;
}

function safeSetAttribute(el, name, value, stats = null) {
  if (!el || typeof el !== 'object' || typeof el.setAttribute !== 'function') return false;
  const next = String(value);
  if (typeof el.getAttribute === 'function') {
    try {
      const current = el.getAttribute(name);
      if (current === next) {
        if (stats) stats.domWritesElided += 1;
        return false;
      }
    } catch {
      // Ignore getAttribute errors and attempt write below.
    }
  }
  try {
    el.setAttribute(name, next);
    if (stats) stats.domWrites += 1;
    return true;
  } catch {
    // Ignore DOM attribute update errors on lightweight mocks.
    return false;
  }
}

function safeSetStyleWidth(el, width, stats = null) {
  if (!el || typeof el !== 'object') return false;
  const next = String(width);
  try {
    if (el.style && typeof el.style === 'object') {
      if (el.style.width === next) {
        if (stats) stats.domWritesElided += 1;
        return false;
      }
      el.style.width = next;
      if (stats) stats.domWrites += 1;
      return true;
    }
  } catch {
    // Ignore DOM style update errors on lightweight mocks.
  }
  return false;
}

/**
 * Strictly sanitizes a scripted-input preset identifier for HUD display.
 * Accepts only `/^[a-z0-9][a-z0-9_-]{0,31}$/i` (rejecting paths, URLs,
 * control characters, and arbitrary text to `null`).
 *
 * @param {unknown} raw
 * @returns {string|null}
 */
export function sanitizeScriptedInputPreset(raw) {
  if (typeof raw !== 'string') {
    return null;
  }
  const trimmed = raw.trim();
  if (!trimmed || /[\x00-\x1F\x7F]/.test(trimmed)) {
    return null;
  }
  if (!SCRIPTED_INPUT_PRESET_RE.test(trimmed)) {
    return null;
  }
  return trimmed;
}

function isEditableEventTarget(target) {
  if (!target || typeof target !== 'object') {
    return false;
  }
  const tagName = typeof target.tagName === 'string' ? target.tagName.toUpperCase() : '';
  if (tagName === 'INPUT' || tagName === 'TEXTAREA' || tagName === 'SELECT') {
    return true;
  }
  if (target.isContentEditable === true) {
    return true;
  }
  if (typeof target.getAttribute === 'function') {
    try {
      const ce = target.getAttribute('contenteditable');
      if (ce !== null && ce !== undefined && String(ce).toLowerCase() !== 'false') {
        return true;
      }
    } catch {
      // Ignore mock getAttribute errors.
    }
  }
  if (typeof target.contentEditable === 'string') {
    const ce = target.contentEditable.toLowerCase();
    if (ce === 'true' || ce === 'plaintext-only') {
      return true;
    }
  }
  return false;
}

export class HudController {
  /**
   * @param {object} [options]
   * @param {object|null} [options.hud=null]
   * @param {object|null} [options.app=null]
   * @param {object|null} [options.sessionController=null]
   * @param {object|null} [options.smokeDemoRunner=null]
   * @param {object|null} [options.storageManager=null]
   * @param {object|null} [options.gpuRecorder=null]
   * @param {object|null} [options.gpuBootPanel=null]
   * @param {object|null} [options.crashReporter=null]
   * @param {object|null} [options.inputProfiles=null]
   * @param {object|null} [options.perfMonitor=null]
   * @param {((filename: string, data: string|Uint8Array, mimeType: string) => any)|null} [options.downloadBlob=null]
   * @param {((text: string) => any)|null} [options.copyToClipboard=null]
   * @param {() => number} [options.now]
   */
  constructor(options = {}) {
    if (options !== undefined && (options === null || typeof options !== 'object')) {
      throw new TypeError('HudController options must be an object');
    }

    const {
      hud = null,
      app = null,
      sessionController = null,
      smokeDemoRunner = null,
      storageManager = null,
      gpuRecorder = null,
      gpuBootPanel = null,
      crashReporter = null,
      inputProfiles = null,
      perfMonitor = null,
      downloadBlob = undefined,
      copyToClipboard = undefined,
      now = () => Date.now(),
      bootMode = null,
      displayElements = null,
      elements = null,
      hudRootEl = null,
      rootEl = null,
      hudRoot = null,
      scriptedInputBadge = null,
      scriptedInputBadgeEl = null,
      scriptedInputStatusEl = null,
      visible = true,
      scriptedInputPreset = null,
      gpuAdapter = null,
      frameBudget = null,
      memory = null,
      mailboxLanes = undefined,
      enableToggleShortcut = false,
      toggleShortcutTarget = undefined,
      keyTarget = undefined,
      toggleShortcutKey = undefined,
      toggleKey = undefined,
      maxValidationErrors = DEFAULT_MAX_VALIDATION_ERRORS,
      maxValidationMessageLength = MAX_VALIDATION_ERROR_MESSAGE_LENGTH,
      minRenderIntervalMs = DEFAULT_MIN_RENDER_INTERVAL_MS,
    } = options;

    if (downloadBlob !== undefined && downloadBlob !== null && typeof downloadBlob !== 'function') {
      throw new TypeError('downloadBlob must be a function or null');
    }
    if (
      copyToClipboard !== undefined &&
      copyToClipboard !== null &&
      typeof copyToClipboard !== 'function'
    ) {
      throw new TypeError('copyToClipboard must be a function or null');
    }
    if (typeof now !== 'function') {
      throw new TypeError('now must be a function returning epoch milliseconds');
    }

    this._hud = hud || null;
    this._app = app || null;
    this._sessionController = sessionController || null;
    this._smokeDemoRunner = smokeDemoRunner || null;
    this._storageManager = storageManager || null;
    this._gpuRecorder = gpuRecorder || null;
    this._gpuBootPanel = gpuBootPanel || (app && app.gpuBootPanel) || null;
    this._crashReporter = crashReporter || null;
    this._inputProfiles = inputProfiles || null;
    this._perfMonitor = perfMonitor || null;
    this._downloadBlob =
      downloadBlob === undefined ? createBrowserDownloadBlob() : downloadBlob;
    this._copyToClipboard =
      copyToClipboard === undefined ? createBrowserCopyToClipboard() : copyToClipboard;
    this._now = now;
    this._hasCustomNow =
      options !== undefined &&
      options !== null &&
      Object.prototype.hasOwnProperty.call(options, 'now') &&
      options.now !== undefined;

    this._lastAction = null;
    this._boundListeners = [];
    this._boundElements = {};
    this._displayElements = {};
    this._hudRootEl =
      (hudRootEl && typeof hudRootEl === 'object' ? hudRootEl : null) ||
      (rootEl && typeof rootEl === 'object' ? rootEl : null) ||
      (hudRoot && typeof hudRoot === 'object' ? hudRoot : null) ||
      null;
    this._scriptedInputBadgeEl =
      (scriptedInputBadge && typeof scriptedInputBadge === 'object' ? scriptedInputBadge : null) ||
      (scriptedInputBadgeEl && typeof scriptedInputBadgeEl === 'object'
        ? scriptedInputBadgeEl
        : null) ||
      (scriptedInputStatusEl && typeof scriptedInputStatusEl === 'object'
        ? scriptedInputStatusEl
        : null) ||
      null;
    this._visible = visible !== undefined ? Boolean(visible) : true;
    this._scriptedInputPreset = sanitizeScriptedInputPreset(scriptedInputPreset);
    this._toggleKeyBinding = null;
    this._exportingMissingShaders = false;
    this._destroyed = false;

    this._explicitBootMode =
      typeof bootMode === 'string' && bootMode.trim().length > 0
        ? normalizeBootMode(bootMode)
        : null;

    this._sideModuleCount = 0;
    this._sideModuleTotalKnown = false;
    this._loadedSideModuleCount = 0;
    this._loadedSideModuleNames = [];
    this._currentSideModule = null;
    this._runtimeInitialized = false;
    this._inOwnHudLog = false;
    this._inOwnHudMetrics = false;
    this._inOwnHudStatus = false;
    this._inOwnUpdateGpuBootTelemetry = false;
    this._hudLogHook = null;
    this._hudMetricsHook = null;
    this._hudStatusHook = null;

    this._liveFramesPresented = 0;
    this._liveFps = 0;
    this._fpsSamples = [];
    this._bootStage = null;
    this._bootStageIndex = -1;
    this._bootProgressPct = 0;
    this._bootReadyForSplash = false;
    this._bootElapsedMs = null;
    this._lastPushedBootStage = null;
    this._lastPushedProgressPct = -1;
    this._lastSeenAppFrames = 0;
    this._ignoredStaleAppStage = null;
    this._lastExternalHudStatus = null;
    this._lastConciseHudStatus = null;

    const resolvedMaxErrors =
      typeof maxValidationErrors === 'number' &&
      Number.isFinite(maxValidationErrors) &&
      maxValidationErrors >= 1
        ? Math.min(256, Math.floor(maxValidationErrors))
        : DEFAULT_MAX_VALIDATION_ERRORS;
    const resolvedMaxMsgLen =
      typeof maxValidationMessageLength === 'number' &&
      Number.isFinite(maxValidationMessageLength) &&
      maxValidationMessageLength >= 16
        ? Math.min(2048, Math.floor(maxValidationMessageLength))
        : MAX_VALIDATION_ERROR_MESSAGE_LENGTH;

    this._maxValidationErrors = resolvedMaxErrors;
    this._maxValidationMessageLength = resolvedMaxMsgLen;
    this._validationErrors = [];
    this._validationErrorCount = 0;
    this._validationErrorSeq = 0;

    this._gpuTelemetry = createInitialGpuTelemetry();
    this._audioTelemetry = createInitialAudioTelemetry();
    this._pthreadTelemetry = createInitialPthreadTelemetry();
    this._frameBudgetTelemetry = createInitialFrameBudgetTelemetry();
    this._memoryTelemetry = createInitialMemoryTelemetry();
    this._ioTelemetry = createInitialIoTelemetry();
    this._hasGpuTelemetry = false;
    this._hasAudioTelemetry = false;
    this._hasGuestAudioTelemetry = false;
    this._hasPthreadTelemetry = false;
    this._hasFrameBudgetTelemetry = false;
    this._hasMemoryTelemetry = false;
    this._hasIoTelemetry = false;
    this._mountedDiagnosticsSection = null;

    const initialGpuAdapter = sanitizeGpuAdapterLabel(gpuAdapter);
    if (initialGpuAdapter !== null) {
      this._gpuTelemetry.gpuAdapter = initialGpuAdapter;
      this._hasGpuTelemetry = true;
    }
    if (isPlainObject(frameBudget)) {
      this._applyFrameBudgetTelemetry(frameBudget);
    }
    if (isPlainObject(memory)) {
      this._applyMemoryTelemetry(memory);
    }
    if (mailboxLanes !== undefined) {
      const initialLanes = sanitizeMailboxLanes(mailboxLanes);
      if (initialLanes !== null) {
        this._ioTelemetry.mailboxLanes = initialLanes;
        this._hasIoTelemetry = true;
      }
    }

    const resolvedMinRenderIntervalMs =
      typeof minRenderIntervalMs === 'number' &&
      Number.isFinite(minRenderIntervalMs) &&
      minRenderIntervalMs > 0
        ? minRenderIntervalMs
        : 0;
    this._minRenderIntervalMs = resolvedMinRenderIntervalMs;
    this._lastRenderTimestampMs = null;
    this._renderDirty = false;
    this._renderCount = 0;
    this._throttledRenders = 0;
    this._domWriteStats = {
      domWrites: 0,
      domWritesElided: 0,
    };

    const initialDisplayEls = isPlainObject(displayElements)
      ? displayElements
      : isPlainObject(elements)
        ? elements
        : null;
    if (initialDisplayEls) {
      this.bindDisplayElements(initialDisplayEls);
    } else {
      this._renderVisibility();
      if (this._scriptedInputBadgeEl) {
        this._renderScriptedInputBadge();
      }
    }
    if (enableToggleShortcut === true) {
      const resolvedShortcutTarget =
        toggleShortcutTarget !== undefined ? toggleShortcutTarget : keyTarget;
      const resolvedShortcutKey =
        toggleShortcutKey !== undefined ? toggleShortcutKey : toggleKey;
      this.bindToggleKey(resolvedShortcutTarget, resolvedShortcutKey);
    }
    this._attachHudLogObserver();
    this._attachHudMetricsObserver();
    this._attachHudStatusObserver();
  }

  get isExportingMissingShaders() {
    return this._exportingMissingShaders;
  }

  get domWrites() {
    return this._domWriteStats.domWrites;
  }

  get domWritesElided() {
    return this._domWriteStats.domWritesElided;
  }

  get minRenderIntervalMs() {
    return this._minRenderIntervalMs;
  }

  // --- Internal Helpers ---

  _getGpuBootPanel() {
    if (this._gpuBootPanel) {
      return this._gpuBootPanel;
    }
    if (this._app && this._app.gpuBootPanel) {
      return this._app.gpuBootPanel;
    }
    return null;
  }

  _attachHudLogObserver() {
    const hud = this._hud;
    if (!hud || typeof hud !== 'object' || typeof hud.appendLog !== 'function') {
      return;
    }
    const hasOwn = Object.prototype.hasOwnProperty.call(hud, 'appendLog');
    const desc = hasOwn ? Object.getOwnPropertyDescriptor(hud, 'appendLog') : null;
    if (desc && !desc.writable && !desc.configurable) {
      return;
    }
    if (desc && desc.get && !desc.set) {
      return;
    }
    if (!hasOwn && !Object.isExtensible(hud)) {
      return;
    }

    const original = hud.appendLog;
    const self = this;
    const wrapped = function hudControllerAppendLogObserver(message, level) {
      if (!self._destroyed && !self._inOwnHudLog) {
        try {
          self.ingestProgressLog(message, { updateHudStatus: false });
        } catch {
          // Progress log parsing must never disrupt HUD logging.
        }
      }
      return original.apply(this, arguments);
    };

    try {
      hud.appendLog = wrapped;
      this._hudLogHook = { hud, original, wrapped, hasOwn };
    } catch {
      // Ignore frozen or non-writable HUD objects.
      this._hudLogHook = null;
    }
  }

  _detachHudLogObserver() {
    const hook = this._hudLogHook;
    this._hudLogHook = null;
    if (!hook || !hook.hud) {
      return;
    }
    try {
      if (hook.hud.appendLog === hook.wrapped) {
        if (hook.hasOwn) {
          hook.hud.appendLog = hook.original;
        } else {
          delete hook.hud.appendLog;
        }
      }
    } catch {
      // Ignore teardown errors on frozen/detached HUD objects.
    }
  }

  _attachHudMetricsObserver() {
    const hud = this._hud;
    if (!hud || typeof hud !== 'object' || typeof hud.updateMetrics !== 'function') {
      return;
    }
    const hasOwn = Object.prototype.hasOwnProperty.call(hud, 'updateMetrics');
    const desc = hasOwn ? Object.getOwnPropertyDescriptor(hud, 'updateMetrics') : null;
    if (desc && !desc.writable && !desc.configurable) {
      return;
    }
    if (desc && desc.get && !desc.set) {
      return;
    }
    if (!hasOwn && !Object.isExtensible(hud)) {
      return;
    }

    const original = hud.updateMetrics;
    const self = this;
    const wrapped = function hudControllerUpdateMetricsObserver(metrics) {
      if (self._destroyed || self._inOwnHudMetrics) {
        return original.apply(this, arguments);
      }
      let forwarded = metrics;
      try {
        forwarded = self._interceptExternalHudMetrics(metrics);
      } catch {
        forwarded = metrics;
      }
      const prev = self._inOwnHudMetrics;
      self._inOwnHudMetrics = true;
      try {
        return original.call(this, forwarded);
      } finally {
        self._inOwnHudMetrics = prev;
      }
    };

    try {
      hud.updateMetrics = wrapped;
      this._hudMetricsHook = { hud, original, wrapped, hasOwn };
    } catch {
      this._hudMetricsHook = null;
    }
  }

  _detachHudMetricsObserver() {
    const hook = this._hudMetricsHook;
    this._hudMetricsHook = null;
    if (!hook || !hook.hud) {
      return;
    }
    try {
      if (hook.hud.updateMetrics === hook.wrapped) {
        if (hook.hasOwn) {
          hook.hud.updateMetrics = hook.original;
        } else {
          delete hook.hud.updateMetrics;
        }
      }
    } catch {
      // Ignore teardown errors on frozen/detached HUD objects.
    }
  }

  _attachHudStatusObserver() {
    const hud = this._hud;
    if (!hud || typeof hud !== 'object' || typeof hud.setStatus !== 'function') {
      return;
    }
    const hasOwn = Object.prototype.hasOwnProperty.call(hud, 'setStatus');
    const desc = hasOwn ? Object.getOwnPropertyDescriptor(hud, 'setStatus') : null;
    if (desc && !desc.writable && !desc.configurable) {
      return;
    }
    if (desc && desc.get && !desc.set) {
      return;
    }
    if (!hasOwn && !Object.isExtensible(hud)) {
      return;
    }

    const original = hud.setStatus;
    const self = this;
    const wrapped = function hudControllerSetStatusObserver(text, level) {
      const result = original.apply(this, arguments);
      if (!self._destroyed && !self._inOwnHudStatus) {
        try {
          self._observeExternalHudStatus(text, level);
        } catch {
          // Status observation must never disrupt HUD status updates.
        }
      }
      return result;
    };

    try {
      hud.setStatus = wrapped;
      this._hudStatusHook = { hud, original, wrapped, hasOwn };
    } catch {
      this._hudStatusHook = null;
    }
  }

  _detachHudStatusObserver() {
    const hook = this._hudStatusHook;
    this._hudStatusHook = null;
    if (!hook || !hook.hud) {
      return;
    }
    try {
      if (hook.hud.setStatus === hook.wrapped) {
        if (hook.hasOwn) {
          hook.hud.setStatus = hook.original;
        } else {
          delete hook.hud.setStatus;
        }
      }
    } catch {
      // Ignore teardown errors on frozen/detached HUD objects.
    }
  }

  _pushHudMetrics(payload) {
    if (
      this._destroyed ||
      this._inOwnHudMetrics ||
      !payload ||
      typeof payload !== 'object' ||
      !this._hud
    ) {
      return;
    }
    const fn =
      this._hudMetricsHook && typeof this._hudMetricsHook.original === 'function'
        ? this._hudMetricsHook.original
        : typeof this._hud.updateMetrics === 'function'
          ? this._hud.updateMetrics
          : null;
    if (!fn) {
      return;
    }
    const prev = this._inOwnHudMetrics;
    this._inOwnHudMetrics = true;
    try {
      fn.call(this._hud, payload);
    } catch {
      // Never allow HUD metric push errors to break controller methods.
    } finally {
      this._inOwnHudMetrics = prev;
    }
  }

  _setHudStatus(text, level = 'info') {
    if (this._destroyed || this._inOwnHudStatus || !this._hud) {
      return;
    }
    const fn =
      this._hudStatusHook && typeof this._hudStatusHook.original === 'function'
        ? this._hudStatusHook.original
        : typeof this._hud.setStatus === 'function'
          ? this._hud.setStatus
          : null;
    if (!fn) {
      return;
    }
    const prev = this._inOwnHudStatus;
    this._inOwnHudStatus = true;
    try {
      fn.call(this._hud, String(text ?? ''), level);
    } catch {
      // Never allow HUD status update errors to break controller methods.
    } finally {
      this._inOwnHudStatus = prev;
    }
  }

  _hudLog(message, level = 'info') {
    if (this._hud && typeof this._hud.appendLog === 'function') {
      const prev = this._inOwnHudLog;
      this._inOwnHudLog = true;
      try {
        this._hud.appendLog(String(message ?? ''), level);
      } catch {
        // Never allow HUD logging errors to break action execution.
      } finally {
        this._inOwnHudLog = prev;
      }
    }
  }

  _finishAction(action, ok, summary, extras = {}, logLevel = ok ? 'ok' : 'error') {
    const error = ok ? undefined : extras.error || summary;
    const timestampMs = this._now();

    this._lastAction = {
      ok: Boolean(ok),
      action,
      summary,
      error: ok ? null : error,
      timestampMs,
    };

    this._hudLog(summary, logLevel);

    const result = {
      ok: Boolean(ok),
      action,
      summary,
      ...extras,
    };
    if (!ok) {
      result.error = error;
    }
    return result;
  }

  async _triggerDownload(filename, data, mimeType) {
    if (typeof this._downloadBlob !== 'function') {
      return false;
    }
    await this._downloadBlob(filename, data, mimeType);
    return true;
  }

  async _triggerClipboardCopy(text) {
    if (typeof this._copyToClipboard !== 'function') {
      return false;
    }
    await this._copyToClipboard(text);
    return true;
  }

  _isRecording() {
    try {
      if (this._gpuRecorder && typeof this._gpuRecorder.isRecording === 'function') {
        return Boolean(this._gpuRecorder.isRecording());
      }
      if (
        this._smokeDemoRunner &&
        this._smokeDemoRunner.recorder &&
        typeof this._smokeDemoRunner.recorder.isRecording === 'function'
      ) {
        return Boolean(this._smokeDemoRunner.recorder.isRecording());
      }
    } catch {
      return false;
    }
    return false;
  }

  _getSessionState() {
    try {
      if (this._sessionController && typeof this._sessionController.getState === 'function') {
        return String(this._sessionController.getState());
      }
      if (this._smokeDemoRunner) {
        return this._smokeDemoRunner.running ? SESSION_STATE.RUNNING_SMOKE : SESSION_STATE.IDLE;
      }
    } catch {
      return SESSION_STATE.ERROR;
    }
    return SESSION_STATE.IDLE;
  }

  _getSessionMode() {
    try {
      if (this._sessionController && typeof this._sessionController.getMode === 'function') {
        return String(this._sessionController.getMode());
      }
      if (this._smokeDemoRunner && this._smokeDemoRunner.running) {
        return 'smoke';
      }
    } catch {
      return 'none';
    }
    return 'none';
  }

  _getActiveProfileId() {
    try {
      if (this._inputProfiles && typeof this._inputProfiles.getActiveProfile === 'function') {
        const active = this._inputProfiles.getActiveProfile();
        return active && typeof active.id === 'string' ? active.id : null;
      }
    } catch {
      return null;
    }
    return null;
  }

  _syncBoundControlState() {
    const domStats = this._domWriteStats;
    const recordBtn = this._boundElements.recordTraceBtn;
    if (recordBtn) {
      const recording = this._isRecording();
      safeSetDataset(recordBtn, 'recording', recording ? 'true' : 'false', domStats);
      safeSetAttribute(recordBtn, 'aria-pressed', recording ? 'true' : 'false', domStats);
    }

    const profileSelect = this._boundElements.profileSelect;
    if (profileSelect) {
      const activeId = this._getActiveProfileId();
      if (activeId !== null) {
        try {
          if (profileSelect.value !== activeId) {
            profileSelect.value = activeId;
            domStats.domWrites += 1;
          } else {
            domStats.domWritesElided += 1;
          }
        } catch {
          // Ignore read-only mock properties.
        }
      }
    }
  }

  // --- Interactive Actions ---

  /**
   * Clears the OPFS range cache via `storageManager.clearOpfsCache()` and logs
   * removed entry count + freed bytes to the HUD.
   */
  async clearOpfsCache() {
    const action = HUD_ACTIONS.CLEAR_OPFS_CACHE;
    if (!this._storageManager || typeof this._storageManager.clearOpfsCache !== 'function') {
      return this._finishAction(
        action,
        false,
        'Storage manager unavailable; cannot clear OPFS cache.',
        { error: 'Storage manager is not configured' },
        'warn'
      );
    }

    try {
      const result = await this._storageManager.clearOpfsCache();
      if (!result || !result.supported) {
        return this._finishAction(
          action,
          false,
          'OPFS cache is unavailable in this browser.',
          { error: 'OPFS cache is unsupported', result: result ?? null },
          'warn'
        );
      }

      const removedCount = Number.isFinite(result.removedCount) ? result.removedCount : 0;
      const freedBytes = Number.isFinite(result.freedBytes) ? result.freedBytes : 0;
      const freedFormatted = formatBytes(freedBytes);
      const summary = `Cleared OPFS cache: removed ${removedCount} file${removedCount === 1 ? '' : 's'}, freed ${freedFormatted} (${freedBytes} B).`;

      return this._finishAction(
        action,
        true,
        summary,
        { removedCount, freedBytes, freedFormatted, result },
        'ok'
      );
    } catch (err) {
      const error = toErrorMessage(err, 'Failed to clear OPFS cache');
      return this._finishAction(action, false, `Failed to clear OPFS cache: ${error}`, { error });
    }
  }

  /**
   * Exports `/userdata/` saves via `storageManager.exportUserdataSnapshot()`, triggers a
   * JSON download (`playgta6-userdata-backup.json`) if `downloadBlob` is available,
   * and returns the serialized snapshot JSON.
   */
  async exportSaves() {
    const action = HUD_ACTIONS.EXPORT_SAVES;
    if (!this._storageManager || typeof this._storageManager.exportUserdataSnapshot !== 'function') {
      return this._finishAction(
        action,
        false,
        'Storage manager unavailable; cannot export saves.',
        { error: 'Storage manager is not configured' },
        'warn'
      );
    }

    try {
      const snapshot = await this._storageManager.exportUserdataSnapshot();
      if (!snapshot || !snapshot.supported) {
        return this._finishAction(
          action,
          false,
          'IndexedDB userdata storage is unavailable; cannot export saves.',
          { error: 'IndexedDB userdata storage is unsupported', snapshot: snapshot ?? null },
          'warn'
        );
      }

      const json = JSON.stringify(snapshot, null, 2);
      const filename = DEFAULT_FILENAMES.USERDATA_BACKUP;
      const downloaded = await this._triggerDownload(filename, json, 'application/json');

      const fileCount = Number.isFinite(snapshot.fileCount) ? snapshot.fileCount : 0;
      const totalBytes = Number.isFinite(snapshot.totalBytes) ? snapshot.totalBytes : 0;
      const summary = `Exported ${fileCount} save file${fileCount === 1 ? '' : 's'} (${formatBytes(totalBytes)}) to ${filename}.`;

      return this._finishAction(
        action,
        true,
        summary,
        { filename, json, snapshot, fileCount, totalBytes, downloaded },
        'ok'
      );
    } catch (err) {
      const error = toErrorMessage(err, 'Failed to export saves');
      return this._finishAction(action, false, `Failed to export saves: ${error}`, { error });
    }
  }

  /**
   * Parses and validates a userdata backup JSON string (or snapshot object) and restores
   * it via `storageManager.importUserdataSnapshot(parsed, { overwrite })`.
   *
   * @param {string|object} jsonStr
   * @param {{ overwrite?: boolean }} [options]
   */
  async importSaves(jsonStr, options = {}) {
    const action = HUD_ACTIONS.IMPORT_SAVES;
    const overwrite = options && typeof options === 'object' && options.overwrite !== undefined
      ? Boolean(options.overwrite)
      : true;

    if (!this._storageManager || typeof this._storageManager.importUserdataSnapshot !== 'function') {
      return this._finishAction(
        action,
        false,
        'Storage manager unavailable; cannot import saves.',
        { error: 'Storage manager is not configured' },
        'warn'
      );
    }

    try {
      let parsed;
      if (typeof jsonStr === 'string') {
        if (jsonStr.trim().length === 0) {
          throw new SyntaxError('Save backup JSON string is empty');
        }
        parsed = JSON.parse(jsonStr);
      } else if (isPlainObject(jsonStr)) {
        parsed = jsonStr;
      } else {
        throw new TypeError('Save backup must be a JSON string or snapshot object');
      }

      if (!isPlainObject(parsed)) {
        throw new TypeError('Save backup JSON root must be an object');
      }
      if (parsed.format !== undefined && parsed.format !== SNAPSHOT_FORMAT) {
        throw new Error(`Unsupported save backup format: ${String(parsed.format)}`);
      }
      if (!Array.isArray(parsed.files)) {
        throw new TypeError('Save backup JSON must include a files array');
      }

      const result = await this._storageManager.importUserdataSnapshot(parsed, { overwrite });
      if (!result || !result.supported) {
        return this._finishAction(
          action,
          false,
          'IndexedDB userdata storage is unavailable; cannot import saves.',
          { error: 'IndexedDB userdata storage is unsupported', result: result ?? null },
          'warn'
        );
      }

      const importedCount = Number.isFinite(result.importedCount) ? result.importedCount : 0;
      const skippedCount = Number.isFinite(result.skippedCount) ? result.skippedCount : 0;
      const importedBytes = Number.isFinite(result.importedBytes) ? result.importedBytes : 0;
      const summary = `Imported ${importedCount} save file${importedCount === 1 ? '' : 's'} (${formatBytes(importedBytes)}, skipped ${skippedCount}).`;

      return this._finishAction(
        action,
        true,
        summary,
        { importedCount, skippedCount, importedBytes, overwrite, result },
        'ok'
      );
    } catch (err) {
      const error = toErrorMessage(err, 'Failed to import saves');
      return this._finishAction(action, false, `Failed to import saves: ${error}`, { error });
    }
  }

  /**
   * Toggles GPU frame recording on `gpuRecorder` or `smokeDemoRunner`.
   */
  async toggleGpuRecording() {
    const action = HUD_ACTIONS.TOGGLE_GPU_RECORDING;
    const hasGpuRecorder =
      this._gpuRecorder &&
      typeof this._gpuRecorder.start === 'function' &&
      typeof this._gpuRecorder.stop === 'function';
    const hasSmokeRunner =
      this._smokeDemoRunner &&
      typeof this._smokeDemoRunner.startRecording === 'function' &&
      typeof this._smokeDemoRunner.stopRecording === 'function';

    if (!hasGpuRecorder && !hasSmokeRunner) {
      return this._finishAction(
        action,
        false,
        'No GPU frame recorder or smoke demo runner configured.',
        { error: 'GPU recorder is not configured', isRecording: false },
        'warn'
      );
    }

    try {
      const wasRecording = this._isRecording();
      let metadata = null;

      if (hasGpuRecorder) {
        if (wasRecording) {
          this._gpuRecorder.stop();
        } else {
          this._gpuRecorder.start();
        }
        if (typeof this._gpuRecorder.exportMetadata === 'function') {
          metadata = this._gpuRecorder.exportMetadata();
        }
      } else {
        if (wasRecording) {
          this._smokeDemoRunner.stopRecording();
        } else {
          this._smokeDemoRunner.startRecording();
        }
        if (typeof this._smokeDemoRunner.exportTraceMetadata === 'function') {
          metadata = this._smokeDemoRunner.exportTraceMetadata();
        }
      }

      const isRecording = this._isRecording();
      this._syncBoundControlState();

      const frameCount = metadata && Number.isFinite(metadata.frameCount) ? metadata.frameCount : 0;
      const summary = isRecording
        ? 'GPU trace recording started.'
        : `GPU trace recording stopped (${frameCount} frame${frameCount === 1 ? '' : 's'} buffered).`;

      return this._finishAction(
        action,
        true,
        summary,
        { isRecording, metadata },
        isRecording ? 'info' : 'ok'
      );
    } catch (err) {
      const error = toErrorMessage(err, 'Failed to toggle GPU recording');
      return this._finishAction(action, false, `Failed to toggle GPU recording: ${error}`, {
        error,
        isRecording: this._isRecording(),
      });
    }
  }

  /**
   * Exports buffered GPU frames as a `.pg6trace` `Uint8Array` via
   * `downloadBlob('playgta6-capture.pg6trace', bytes, 'application/octet-stream')`.
   *
   * @param {{ frameStart?: number, frameCount?: number }} [options]
   */
  async exportGpuTrace(options) {
    const action = HUD_ACTIONS.EXPORT_GPU_TRACE;
    const hasGpuRecorder =
      this._gpuRecorder && typeof this._gpuRecorder.exportTraceBytes === 'function';
    const hasSmokeRunner =
      this._smokeDemoRunner && typeof this._smokeDemoRunner.exportTraceBytes === 'function';

    if (!hasGpuRecorder && !hasSmokeRunner) {
      return this._finishAction(
        action,
        false,
        'No GPU frame recorder or smoke demo runner configured.',
        { error: 'GPU recorder is not configured' },
        'warn'
      );
    }

    try {
      let rawBytes;
      let metadata = null;

      if (hasGpuRecorder) {
        rawBytes = this._gpuRecorder.exportTraceBytes(options);
        if (typeof this._gpuRecorder.exportMetadata === 'function') {
          metadata = this._gpuRecorder.exportMetadata();
        }
      } else {
        rawBytes = this._smokeDemoRunner.exportTraceBytes(options);
        if (typeof this._smokeDemoRunner.exportTraceMetadata === 'function') {
          metadata = this._smokeDemoRunner.exportTraceMetadata();
        }
      }

      const bytes = toUint8Array(rawBytes);
      if (bytes.byteLength === 0) {
        return this._finishAction(
          action,
          false,
          'No recorded GPU frames available to export.',
          {
            error: 'No recorded GPU frames available',
            bytes,
            byteLength: 0,
            frameCount: 0,
            metadata,
            downloaded: false,
          },
          'warn'
        );
      }

      const filename = DEFAULT_FILENAMES.GPU_TRACE;
      const downloaded = await this._triggerDownload(filename, bytes, 'application/octet-stream');
      const frameCount = metadata && Number.isFinite(metadata.frameCount) ? metadata.frameCount : null;
      const frameDetail = frameCount !== null ? ` (${frameCount} frame${frameCount === 1 ? '' : 's'})` : '';
      const summary = `Exported GPU trace${frameDetail} (${formatBytes(bytes.byteLength)}) to ${filename}.`;

      return this._finishAction(
        action,
        true,
        summary,
        {
          filename,
          bytes,
          byteLength: bytes.byteLength,
          frameCount,
          metadata,
          downloaded,
        },
        'ok'
      );
    } catch (err) {
      const error = toErrorMessage(err, 'Failed to export GPU trace');
      return this._finishAction(action, false, `Failed to export GPU trace: ${error}`, { error });
    }
  }

  /**
   * Builds a sanitized diagnostic report via `crashReporter.toJson({ perfSnapshot, storageSummary, serverStatus })`
   * and triggers a JSON download (`playgta6-diagnostic-report.json`).
   *
   * @param {{ perfSnapshot?: object, storageSummary?: object, serverStatus?: object, extra?: unknown, copyToClipboard?: boolean }} [options]
   */
  async exportCrashReport(options = {}) {
    const action = HUD_ACTIONS.EXPORT_CRASH_REPORT;
    if (!this._crashReporter || typeof this._crashReporter.toJson !== 'function') {
      return this._finishAction(
        action,
        false,
        'Crash reporter unavailable; cannot export diagnostic report.',
        { error: 'Crash reporter is not configured' },
        'warn'
      );
    }

    try {
      const opts = isPlainObject(options) ? options : {};

      let sessionDiagnostics = null;
      if (
        this._sessionController &&
        typeof this._sessionController.getDiagnostics === 'function'
      ) {
        try {
          sessionDiagnostics = this._sessionController.getDiagnostics();
        } catch {
          sessionDiagnostics = null;
        }
      }

      let perfSnapshot = opts.perfSnapshot;
      if (perfSnapshot === undefined) {
        if (this._perfMonitor && typeof this._perfMonitor.getSnapshot === 'function') {
          try {
            perfSnapshot = this._perfMonitor.getSnapshot();
          } catch {
            perfSnapshot = null;
          }
        } else if (sessionDiagnostics && sessionDiagnostics.perfSnapshot !== undefined) {
          perfSnapshot = sessionDiagnostics.perfSnapshot;
        }
      }

      let storageSummary = opts.storageSummary;
      if (storageSummary === undefined) {
        if (this._storageManager && typeof this._storageManager.getSummary === 'function') {
          try {
            storageSummary = await this._storageManager.getSummary();
          } catch {
            storageSummary = sessionDiagnostics ? sessionDiagnostics.storageSummary : null;
          }
        } else if (sessionDiagnostics && sessionDiagnostics.storageSummary !== undefined) {
          storageSummary = sessionDiagnostics.storageSummary;
        }
      }

      let serverStatus = opts.serverStatus;
      if (serverStatus === undefined && sessionDiagnostics && sessionDiagnostics.serverStatus !== undefined) {
        serverStatus = sessionDiagnostics.serverStatus;
      }

      const reportInput = {
        perfSnapshot,
        storageSummary,
        serverStatus,
      };
      if (opts.extra !== undefined) {
        reportInput.extra = opts.extra;
      }

      const json = this._crashReporter.toJson(reportInput);
      const filename = DEFAULT_FILENAMES.CRASH_REPORT;
      const downloaded = await this._triggerDownload(filename, json, 'application/json');
      const copied = opts.copyToClipboard === true
        ? await this._triggerClipboardCopy(json)
        : false;
      const byteLength = new TextEncoder().encode(json).byteLength;
      const summary = `Exported sanitized diagnostic report (${formatBytes(byteLength)}) to ${filename}.`;

      return this._finishAction(
        action,
        true,
        summary,
        { filename, json, byteLength, downloaded, copied },
        'ok'
      );
    } catch (err) {
      const error = toErrorMessage(err, 'Failed to export diagnostic report');
      return this._finishAction(
        action,
        false,
        `Failed to export diagnostic report: ${error}`,
        { error }
      );
    }
  }

  /**
   * Exports the sanitized missing-shader shopping list JSON via
   * `gpuBootPanel.exportMissingShadersJson()` and triggers a JSON download
   * (`playgta6-missing-shaders.json`) when `downloadBlob` is available.
   *
   * @param {{ copyToClipboard?: boolean }} [options]
   */
  async exportMissingShaders(options = {}) {
    const action = HUD_ACTIONS.EXPORT_MISSING_SHADERS;
    const panel = this._getGpuBootPanel();
    if (!panel || typeof panel.exportMissingShadersJson !== 'function') {
      return this._finishAction(
        action,
        false,
        'GPU boot panel unavailable; cannot export missing shaders.',
        { error: 'GPU boot panel is not configured' },
        'warn'
      );
    }

    try {
      const opts = isPlainObject(options) ? options : {};
      let json;
      this._exportingMissingShaders = true;
      try {
        json = panel.exportMissingShadersJson();
      } finally {
        this._exportingMissingShaders = false;
      }

      if (typeof json !== 'string' || json.length === 0) {
        throw new TypeError('GPU boot panel did not return a missing-shader JSON string');
      }

      let missingShaderCount = null;
      try {
        const parsed = JSON.parse(json);
        if (parsed && typeof parsed === 'object') {
          if (Number.isFinite(parsed.missingShaderCount)) {
            missingShaderCount = parsed.missingShaderCount;
          } else if (Array.isArray(parsed.missingShaders)) {
            missingShaderCount = parsed.missingShaders.length;
          }
        }
      } catch {
        // Leave missingShaderCount null if a custom mock returned non-JSON text.
      }

      const filename = DEFAULT_FILENAMES.MISSING_SHADERS;
      const downloaded = await this._triggerDownload(filename, json, 'application/json');
      const copied =
        opts.copyToClipboard === true ? await this._triggerClipboardCopy(json) : false;
      const byteLength = new TextEncoder().encode(json).byteLength;
      const countDetail =
        missingShaderCount !== null
          ? `${missingShaderCount} missing shader${missingShaderCount === 1 ? '' : 's'}, `
          : '';
      const summary = `Exported missing-shader report (${countDetail}${formatBytes(byteLength)}) to ${filename}.`;

      return this._finishAction(
        action,
        true,
        summary,
        { filename, json, byteLength, missingShaderCount, downloaded, copied },
        'ok'
      );
    } catch (err) {
      const error = toErrorMessage(err, 'Failed to export missing shaders');
      return this._finishAction(action, false, `Failed to export missing shaders: ${error}`, {
        error,
      });
    }
  }

  /**
   * Forwards GPU/boot diagnostics telemetry (`bootSummary`, `missingShaders`,
   * `fallbackStats`, `frameStats`) to `gpuBootPanel.update(payload)` when present,
   * and ingests optional `bootMode`, `firstPresentSource`, split-module progress,
   * and WebGPU validation error telemetry into HudController diagnostics state.
   * Safe no-op returning `null` when `gpuBootPanel` is absent or destroyed.
   *
   * @param {{
   *   bootSummary?: object|null,
   *   missingShaders?: Array|null,
   *   fallbackStats?: object|null,
   *   frameStats?: object|null,
   *   bootMode?: string|null,
   *   firstPresentSource?: string|null,
   *   sideModuleCount?: number|null,
   *   loadedSideModules?: number|null,
   *   sideModuleProgress?: object|null,
   *   validationError?: object|string|null,
   *   validationErrors?: Array|null,
   *   validationErrorCount?: number|null,
   * }|null|undefined} payload
   * @returns {object|null}
   */
  updateGpuBootTelemetry(payload, options = {}) {
    if (this._destroyed || this._inOwnUpdateGpuBootTelemetry) {
      return null;
    }

    const opts = isPlainObject(options) ? options : {};
    const prevReentry = this._inOwnUpdateGpuBootTelemetry;
    this._inOwnUpdateGpuBootTelemetry = true;
    try {
      if (payload && typeof payload === 'object') {
        this._ingestBootTelemetryPayload(payload, opts);
      }

      const panel = this._getGpuBootPanel();
      if (!panel || typeof panel.update !== 'function') {
        return null;
      }
      if (!payload || typeof payload !== 'object') {
        try {
          return typeof panel.getState === 'function' ? panel.getState() : null;
        } catch {
          return null;
        }
      }
      try {
        const state = panel.update(payload);
        return state !== undefined ? state : null;
      } catch {
        return null;
      }
    } finally {
      this._inOwnUpdateGpuBootTelemetry = prevReentry;
    }
  }

  _ingestBootTelemetryPayload(payload, opts = {}) {
    const force = Boolean(opts && opts.force === true);
    let changed = false;
    let liveMetricsDirty = false;
    let bootStageChanged = false;

    const summary =
      payload.bootSummary && typeof payload.bootSummary === 'object' ? payload.bootSummary : null;
    if (summary && this._applyBootStage(summary)) {
      bootStageChanged = true;
      changed = true;
    } else if (payload.milestone !== undefined || payload.stage !== undefined) {
      if (
        this._applyBootStage({
          stage: payload.milestone ?? payload.stage,
          progressPct: payload.progressPct,
          readyForSplash: payload.readyForSplash,
          elapsedMs: payload.elapsedMs,
        })
      ) {
        bootStageChanged = true;
        changed = true;
      }
    }

    const rawFirstPresentSource =
      payload.firstPresentSource !== undefined
        ? payload.firstPresentSource
        : summary && summary.firstPresentSource !== undefined
          ? summary.firstPresentSource
          : null;
    const rawBootMode =
      payload.bootMode !== undefined
        ? payload.bootMode
        : summary && summary.bootMode !== undefined
          ? summary.bootMode
          : null;

    if (rawFirstPresentSource !== null || rawBootMode !== null) {
      const sourceMode =
        rawFirstPresentSource !== null ? normalizeBootMode(rawFirstPresentSource) : null;
      const requestedMode = rawBootMode !== null ? normalizeBootMode(rawBootMode) : null;
      // Smoke frames must never be mislabeled as real Wasm64 frames.
      if (sourceMode === BOOT_MODE.SMOKE || requestedMode === BOOT_MODE.SMOKE) {
        this._explicitBootMode = BOOT_MODE.SMOKE;
        changed = true;
      } else if (sourceMode === BOOT_MODE.REAL_WASM64 || requestedMode === BOOT_MODE.REAL_WASM64) {
        if (!this._isActiveSmokeSession()) {
          this._explicitBootMode = BOOT_MODE.REAL_WASM64;
          changed = true;
        }
      } else if (rawBootMode !== null && requestedMode === BOOT_MODE.IDLE) {
        this._explicitBootMode = BOOT_MODE.IDLE;
        changed = true;
      }
    }

    const rawSideProgress =
      payload.sideModuleProgress && typeof payload.sideModuleProgress === 'object'
        ? payload.sideModuleProgress
        : null;
    const hasSideFields =
      rawSideProgress !== null ||
      payload.sideModuleCount !== undefined ||
      payload.side_module_count !== undefined ||
      payload.loadedSideModules !== undefined ||
      payload.loadedSideModuleCount !== undefined ||
      Array.isArray(payload.sideModules) ||
      (summary &&
        (summary.sideModuleCount !== undefined ||
          summary.side_module_count !== undefined ||
          summary.loadedSideModules !== undefined ||
          Array.isArray(summary.sideModules)));

    if (hasSideFields) {
      const progressInput = rawSideProgress || {
        sideModuleCount:
          payload.sideModuleCount ??
          payload.side_module_count ??
          (summary ? summary.sideModuleCount ?? summary.side_module_count : undefined),
        loadedCount:
          payload.loadedSideModules ??
          payload.loadedSideModuleCount ??
          (summary ? summary.loadedSideModules ?? summary.loadedCount : undefined),
        sideModules: payload.sideModules ?? (summary ? summary.sideModules : undefined),
        loadedModules: payload.loadedSideModuleNames ?? (summary ? summary.loadedModules : undefined),
      };
      this.setSideModuleProgress(progressInput, { updateHudStatus: false, force });
      changed = false; // setSideModuleProgress already rendered.
    } else if (
      summary &&
      typeof summary.stage === 'string' &&
      POST_INIT_BOOT_STAGES.has(summary.stage.trim().toUpperCase()) &&
      !this._isActiveSmokeSession() &&
      this.getBootMode() !== BOOT_MODE.SMOKE
    ) {
      this._runtimeInitialized = true;
      if (this._sideModuleCount > 0 && this._loadedSideModuleCount < this._sideModuleCount) {
        this._loadedSideModuleCount = this._sideModuleCount;
        changed = true;
      }
    }

    if (Array.isArray(payload.validationErrors)) {
      this.setValidationErrors(payload.validationErrors, {
        totalCount: payload.validationErrorCount,
        force,
      });
      changed = false;
    } else if (payload.validationError !== undefined && payload.validationError !== null) {
      this.recordValidationError(payload.validationError, { force });
      changed = false;
    }

    if (this._applyStructuredTelemetryPayload(payload, { includeFrameStatsPresent: true })) {
      changed = true;
      liveMetricsDirty = true;
    }

    if (changed) {
      this._renderBootDiagnostics({ updateHudStatus: false, force });
    }
    if (liveMetricsDirty || bootStageChanged) {
      this._syncLiveHudReadouts({
        includeFrames: liveMetricsDirty || this._liveFramesPresented > 0,
        includeBootSummary: bootStageChanged,
        updateHudStatus: false,
      });
    }
  }

  _safeNowMs() {
    try {
      const val = this._now();
      return typeof val === 'number' && Number.isFinite(val) ? val : Date.now();
    } catch {
      return Date.now();
    }
  }

  _recordLiveFrames(frameCount, opts = {}) {
    const parsedFrames = toNonNegativeInt(frameCount, null);
    const explicitFps = toNonNegativeFiniteNumber(opts.fps, null);
    let framesChanged = false;
    let fpsChanged = false;

    if (explicitFps !== null && explicitFps > 0) {
      const roundedExplicit = Math.round(explicitFps * 10) / 10;
      if (roundedExplicit !== this._liveFps) {
        this._liveFps = roundedExplicit;
        fpsChanged = true;
      }
    }

    if (parsedFrames === null) {
      return { framesChanged, fpsChanged };
    }

    const effectiveFrames = opts.allowReset
      ? parsedFrames
      : Math.max(this._liveFramesPresented, parsedFrames);
    const deltaFrames = effectiveFrames - this._liveFramesPresented;
    if (effectiveFrames !== this._liveFramesPresented) {
      this._liveFramesPresented = effectiveFrames;
      framesChanged = true;
    }
    if (effectiveFrames > this._gpuTelemetry.presentCount) {
      this._gpuTelemetry.presentCount = effectiveFrames;
    }

    if (opts.allowReset && effectiveFrames === 0) {
      this._fpsSamples = [];
      if (this._liveFps !== 0) {
        this._liveFps = 0;
        fpsChanged = true;
      }
      return { framesChanged, fpsChanged };
    }

    if (deltaFrames > 0 || (effectiveFrames > 0 && this._fpsSamples.length === 0)) {
      const nowMs = this._safeNowMs();
      const rawExplicitTs = toNonNegativeFiniteNumber(opts.timestampMs, null);
      let ts = nowMs;
      if (rawExplicitTs !== null) {
        if (!this._hasCustomNow) {
          ts = rawExplicitTs;
        } else if (
          this._fpsSamples.length > 0 &&
          nowMs === this._fpsSamples[this._fpsSamples.length - 1].timestampMs &&
          rawExplicitTs > this._fpsSamples[this._fpsSamples.length - 1].timestampMs
        ) {
          ts = rawExplicitTs;
        }
      }

      if (this._fpsSamples.length === 0) {
        this._fpsSamples.push({ frames: effectiveFrames, timestampMs: ts });
      } else {
        const last = this._fpsSamples[this._fpsSamples.length - 1];
        if (ts < last.timestampMs || ts - last.timestampMs > MAX_FPS_CLOCK_JUMP_MS) {
          this._fpsSamples = [{ frames: effectiveFrames, timestampMs: ts }];
        } else if (ts === last.timestampMs) {
          last.frames = Math.max(last.frames, effectiveFrames);
        } else {
          this._fpsSamples.push({ frames: effectiveFrames, timestampMs: ts });
          while (
            this._fpsSamples.length > 2 &&
            (this._fpsSamples.length > MAX_FPS_SAMPLES ||
              ts - this._fpsSamples[1].timestampMs > FPS_WINDOW_MS)
          ) {
            this._fpsSamples.shift();
          }
        }
      }

      if (explicitFps === null || explicitFps <= 0) {
        const oldest = this._fpsSamples[0];
        const newest = this._fpsSamples[this._fpsSamples.length - 1];
        const dtMs = newest.timestampMs - oldest.timestampMs;
        const df = newest.frames - oldest.frames;
        if (dtMs > 0 && df > 0) {
          const computed = (df * 1000) / dtMs;
          if (Number.isFinite(computed) && computed > 0) {
            const rounded = Math.round(computed * 10) / 10;
            if (rounded !== this._liveFps) {
              this._liveFps = rounded;
              fpsChanged = true;
            }
          }
        }
      }
    }

    return { framesChanged, fpsChanged };
  }

  _applyBootStage(input, opts = {}) {
    if (input === null || input === undefined) {
      return false;
    }
    let rawStage = null;
    let rawPct = null;
    let rawReady = null;
    let rawElapsed = null;

    if (typeof input === 'string') {
      rawStage = input;
    } else if (isPlainObject(input)) {
      rawStage = safeGet(input, 'stage') ?? safeGet(input, 'milestone') ?? null;
      const pctField = safeGet(input, 'progressPct');
      if (pctField !== undefined && pctField !== null) {
        rawPct = toNonNegativeFiniteNumber(pctField, null);
      } else {
        const reached = toNonNegativeFiniteNumber(safeGet(input, 'milestonesReached'), null);
        const total = toNonNegativeFiniteNumber(safeGet(input, 'totalMilestones'), null);
        if (reached !== null && total !== null && total > 0) {
          rawPct = Math.min(100, (reached / total) * 100);
        }
      }
      const readyField = safeGet(input, 'readyForSplash');
      if (typeof readyField === 'boolean') {
        rawReady = readyField;
      }
      const elapsedField = safeGet(input, 'elapsedMs');
      if (elapsedField !== undefined && elapsedField !== null) {
        rawElapsed = toNonNegativeFiniteNumber(elapsedField, null);
      }
    } else {
      return false;
    }

    const milestone = normalizeBootMilestone(rawStage);
    if (!milestone) {
      if (rawReady === true) {
        return this._applyBootStage(
          { stage: 'FIRST_PRESENT', progressPct: 100, readyForSplash: true, elapsedMs: rawElapsed },
          opts
        );
      }
      return false;
    }

    const nextIdx = BOOT_MILESTONE_INDEX.get(milestone);
    if (!opts.allowReset && nextIdx < this._bootStageIndex) {
      return false;
    }

    let changed = false;
    if (milestone !== this._bootStage) {
      this._bootStage = milestone;
      this._bootStageIndex = nextIdx;
      changed = true;
    }

    const computedPct =
      rawPct !== null ? Math.min(100, rawPct) : defaultMilestoneProgressPct(milestone);
    if (opts.allowReset || computedPct > this._bootProgressPct || changed) {
      const nextPct = opts.allowReset ? computedPct : Math.max(this._bootProgressPct, computedPct);
      if (nextPct !== this._bootProgressPct) {
        this._bootProgressPct = nextPct;
        changed = true;
      }
    }

    const nextReady = milestone === 'FIRST_PRESENT' || rawReady === true;
    if (nextReady && !this._bootReadyForSplash) {
      this._bootReadyForSplash = true;
      if (this._bootProgressPct < 100) {
        this._bootProgressPct = 100;
      }
      changed = true;
    }

    if (rawElapsed !== null && rawElapsed !== this._bootElapsedMs) {
      this._bootElapsedMs = rawElapsed;
    }

    if (
      POST_INIT_BOOT_STAGES.has(milestone) &&
      !this._isActiveSmokeSession() &&
      this.getBootMode() !== BOOT_MODE.SMOKE
    ) {
      this._runtimeInitialized = true;
      if (this._sideModuleCount > 0 && this._loadedSideModuleCount < this._sideModuleCount) {
        this._loadedSideModuleCount = this._sideModuleCount;
        changed = true;
      }
    }

    return changed;
  }

  _buildBootSummaryPayload(force = false) {
    if (!this._bootStage) {
      return null;
    }
    if (
      !force &&
      this._bootStage === this._lastPushedBootStage &&
      this._bootProgressPct === this._lastPushedProgressPct
    ) {
      return null;
    }
    this._lastPushedBootStage = this._bootStage;
    this._lastPushedProgressPct = this._bootProgressPct;
    const summary = {
      stage: this._bootStage,
      progressPct: this._bootProgressPct,
      readyForSplash: this._bootReadyForSplash,
    };
    if (this._bootElapsedMs !== null) {
      summary.elapsedMs = this._bootElapsedMs;
    }
    return summary;
  }

  _captureAppFramesBaseline() {
    let appFrames = 0;
    let appStage = null;
    const app = this._app;
    if (!app || typeof app !== 'object') {
      this._lastSeenAppFrames = 0;
      this._ignoredStaleAppStage = null;
      return;
    }
    try {
      if (app.bootTracer && typeof app.bootTracer.reset === 'function') {
        app.bootTracer.reset();
      }
    } catch {
      // Ignore optional bootTracer reset errors.
    }
    try {
      if (typeof app.getBootTelemetry === 'function') {
        const bt = app.getBootTelemetry();
        const p = toNonNegativeInt(bt && (bt.presentCount ?? bt.gpu?.presentCount), 0);
        if (p > appFrames) appFrames = p;
        const m = normalizeBootMilestone(bt && bt.milestone);
        if (m) appStage = m;
      }
    } catch {
      // Ignore optional app telemetry errors.
    }
    try {
      if (typeof app.getDiagnosticsSnapshot === 'function') {
        const snap = app.getDiagnosticsSnapshot();
        const f = toNonNegativeInt(snap && snap.framesPresented, 0);
        if (f > appFrames) appFrames = f;
        if (!appStage) {
          const m = normalizeBootMilestone(snap && snap.milestone);
          if (m) appStage = m;
        }
      }
    } catch {
      // Ignore optional diagnostics snapshot errors.
    }
    this._lastSeenAppFrames = appFrames;
    this._ignoredStaleAppStage = appStage;
  }

  _syncFromAppSources() {
    const app = this._app;
    if (!app || typeof app !== 'object') {
      return { bootStageChanged: false, telemetryChanged: false };
    }
    let bootStageChanged = false;
    let telemetryChanged = false;

    try {
      if (app.bootTracer && typeof app.bootTracer.getSummary === 'function') {
        const summary = app.bootTracer.getSummary();
        if (summary && typeof summary === 'object') {
          const tracerStage = normalizeBootMilestone(summary.stage);
          if (!tracerStage) {
            this._ignoredStaleAppStage = null;
          } else if (tracerStage !== this._ignoredStaleAppStage) {
            this._ignoredStaleAppStage = null;
            if (this._applyBootStage(summary)) {
              bootStageChanged = true;
            }
          }
        }
      }
    } catch {
      // Ignore optional bootTracer errors.
    }

    let appFrames = null;
    let appGpu = null;
    let appMilestone = null;

    try {
      if (typeof app.getBootTelemetry === 'function') {
        const bt = app.getBootTelemetry();
        if (bt && typeof bt === 'object') {
          if (bt.milestone) {
            appMilestone = bt.milestone;
          }
          const present = toNonNegativeInt(bt.presentCount ?? bt.gpu?.presentCount, null);
          if (present !== null) {
            appFrames = present;
          }
          appGpu = bt.gpu && typeof bt.gpu === 'object' ? bt.gpu : bt;
        }
      }
    } catch {
      // Ignore optional getBootTelemetry errors.
    }

    try {
      if (typeof app.getDiagnosticsSnapshot === 'function') {
        const snap = app.getDiagnosticsSnapshot();
        if (snap && typeof snap === 'object') {
          if (!appMilestone && snap.milestone) {
            appMilestone = snap.milestone;
          }
          const f = toNonNegativeInt(snap.framesPresented, null);
          if (f !== null && (appFrames === null || f > appFrames)) {
            appFrames = f;
          }
        }
      }
    } catch {
      // Ignore optional getDiagnosticsSnapshot errors.
    }

    const normalizedAppMilestone = normalizeBootMilestone(appMilestone);
    if (!normalizedAppMilestone) {
      this._ignoredStaleAppStage = null;
    } else if (normalizedAppMilestone !== this._ignoredStaleAppStage) {
      this._ignoredStaleAppStage = null;
      if (this._applyBootStage(normalizedAppMilestone)) {
        bootStageChanged = true;
      }
    }

    if (appFrames !== null) {
      if (appFrames < this._lastSeenAppFrames) {
        this._lastSeenAppFrames = appFrames;
      }
      if (appFrames > this._lastSeenAppFrames) {
        this._lastSeenAppFrames = appFrames;
        const { framesChanged, fpsChanged } = this._recordLiveFrames(appFrames);
        if (framesChanged || fpsChanged) {
          telemetryChanged = true;
        }
        if (appGpu && typeof appGpu === 'object') {
          const rp = toNonNegativeInt(safeGet(appGpu, 'realPackDrawCount'), null);
          const fb = toNonNegativeInt(safeGet(appGpu, 'fallbackDrawCount'), null);
          const ve = toNonNegativeInt(safeGet(appGpu, 'validationErrorCount'), null);
          if (rp !== null && rp !== this._gpuTelemetry.realPackDrawCount) {
            this._gpuTelemetry.realPackDrawCount = rp;
            this._hasGpuTelemetry = true;
            telemetryChanged = true;
          }
          if (fb !== null && fb !== this._gpuTelemetry.fallbackDrawCount) {
            this._gpuTelemetry.fallbackDrawCount = fb;
            this._hasGpuTelemetry = true;
            telemetryChanged = true;
          }
          if (ve !== null && ve > this._validationErrorCount) {
            this._gpuTelemetry.validationErrorCount = ve;
            this._validationErrorCount = ve;
            telemetryChanged = true;
          }
          const appLowFps = toNonNegativeFiniteNumber(
            pickFirstDefined(appGpu, ['onePercentLowFps', 'one_percent_low_fps']),
            null
          );
          if (appLowFps !== null && appLowFps !== this._gpuTelemetry.onePercentLowFps) {
            this._gpuTelemetry.onePercentLowFps = appLowFps;
            this._hasGpuTelemetry = true;
            telemetryChanged = true;
          }
          const appP99 = toNonNegativeFiniteNumber(
            pickFirstDefined(appGpu, ['p99FrameMs', 'p99_frame_ms']),
            null
          );
          if (appP99 !== null && appP99 !== this._gpuTelemetry.p99FrameMs) {
            this._gpuTelemetry.p99FrameMs = appP99;
            this._hasGpuTelemetry = true;
            telemetryChanged = true;
          }
          const appZeroCopy = toNonNegativeInt(
            pickFirstDefined(appGpu, [
              'zeroCopyUploads',
              'zero_copy_uploads',
              'binkZeroCopyUploads',
              'zeroCopyFramesForwarded',
            ]),
            null
          );
          if (appZeroCopy !== null && appZeroCopy !== this._gpuTelemetry.zeroCopyUploads) {
            this._gpuTelemetry.zeroCopyUploads = appZeroCopy;
            this._hasGpuTelemetry = true;
            telemetryChanged = true;
          }
        }
      }
    }

    return { bootStageChanged, telemetryChanged };
  }

  _buildConciseHudStatus() {
    const g = this._gpuTelemetry;
    const validation = this.getValidationSummary();
    const bootMode = this.getBootMode();
    const level = validation.hasErrors
      ? 'error'
      : bootMode === BOOT_MODE.SMOKE
        ? 'warn'
        : 'ok';

    if (g.realPackDrawCount > 0 || g.fallbackDrawCount > 0) {
      const text =
        g.fallbackDrawCount > 0
          ? `Draws: ${g.realPackDrawCount} real, ${g.fallbackDrawCount} fallback`
          : `Draws: ${g.realPackDrawCount} real`;
      return { text, level };
    }
    if (this._bootStage === 'FIRST_PRESENT') {
      const text = 'Boot 100%: FIRST_PRESENT';
      if (this._lastExternalHudStatus && /FIRST_PRESENT/i.test(this._lastExternalHudStatus)) {
        return null;
      }
      return { text, level };
    }
    return null;
  }

  _syncLiveHudReadouts(opts = {}) {
    if (this._destroyed) {
      return;
    }
    const payload = {};
    if (opts.includeFrames !== false && (this._liveFramesPresented > 0 || opts.forceFrames === true)) {
      payload.framesPresented = this._liveFramesPresented;
      payload.fps = this._liveFps;
    }
    const bootSummary = this._buildBootSummaryPayload(opts.forceBootSummary === true);
    if (bootSummary) {
      payload.bootSummary = bootSummary;
    }
    if (Object.keys(payload).length > 0) {
      this._pushHudMetrics(payload);
    }

    if (opts.updateHudStatus === undefined) {
      const concise = this._buildConciseHudStatus();
      if (concise && concise.text !== this._lastConciseHudStatus) {
        this._lastConciseHudStatus = concise.text;
        this._setHudStatus(concise.text, concise.level);
      }
    }
  }

  _observeExternalHudStatus(text, _level) {
    if (typeof text !== 'string') {
      return;
    }
    this._lastExternalHudStatus = text;
    let bootStageChanged = false;
    const parsed = extractBootMilestoneFromText(text);
    if (parsed && this._applyBootStage(parsed)) {
      bootStageChanged = true;
    }
    const appSync = this._syncFromAppSources();
    if (appSync.bootStageChanged) {
      bootStageChanged = true;
    }
    if (bootStageChanged || appSync.telemetryChanged) {
      this._renderBootDiagnostics({ updateHudStatus: false });
      const bootSummary = this._buildBootSummaryPayload(false);
      if (bootSummary || appSync.telemetryChanged) {
        const metricsPatch = {};
        if (bootSummary) {
          metricsPatch.bootSummary = bootSummary;
        }
        if (appSync.telemetryChanged && this._liveFramesPresented > 0) {
          metricsPatch.framesPresented = this._liveFramesPresented;
          metricsPatch.fps = this._liveFps;
        }
        if (Object.keys(metricsPatch).length > 0) {
          this._pushHudMetrics(metricsPatch);
        }
      }
    }
  }

  _interceptExternalHudMetrics(metrics, opts = {}) {
    if (!isPlainObject(metrics)) {
      return metrics;
    }
    const force = Boolean(opts && opts.force === true);
    let diagChanged = false;

    const appSync = this._syncFromAppSources();
    if (appSync.bootStageChanged || appSync.telemetryChanged) {
      diagChanged = true;
    }

    const incomingBootSummary = safeGet(metrics, 'bootSummary');
    if (isPlainObject(incomingBootSummary)) {
      if (this._applyBootStage(incomingBootSummary)) {
        diagChanged = true;
      }
    } else if (safeGet(metrics, 'milestone') !== undefined || safeGet(metrics, 'stage') !== undefined) {
      if (
        this._applyBootStage({
          stage: safeGet(metrics, 'milestone') ?? safeGet(metrics, 'stage'),
          progressPct: safeGet(metrics, 'progressPct'),
          readyForSplash: safeGet(metrics, 'readyForSplash'),
          elapsedMs: safeGet(metrics, 'elapsedMs'),
        })
      ) {
        diagChanged = true;
      }
    }

    const frameStats = safeGet(metrics, 'frameStats');
    const gpuStats = safeGet(metrics, 'gpuStats');
    const incomingFramesRaw = pickFirstDefined(metrics, [
      'framesPresented',
      'frames_presented',
      'presentCount',
      'present_count',
      'totalFrames',
    ]);
    const nestedFramesRaw =
      incomingFramesRaw !== undefined
        ? incomingFramesRaw
        : isPlainObject(frameStats)
          ? pickFirstDefined(frameStats, ['framesPresented', 'presentCount', 'totalFrames'])
          : isPlainObject(gpuStats)
            ? pickFirstDefined(gpuStats, ['presentCount', 'framesPresented'])
            : undefined;
    const incomingFrames = toNonNegativeInt(nestedFramesRaw, null);
    const incomingFps = toNonNegativeFiniteNumber(safeGet(metrics, 'fps'), null);
    const incomingTs = toNonNegativeFiniteNumber(
      safeGet(metrics, 'timestampMs') ?? safeGet(metrics, 'timestamp'),
      null
    );

    if (
      this._applyStructuredTelemetryPayload(metrics, { includeFrameStatsPresent: true })
    ) {
      diagChanged = true;
    }

    if (incomingFrames !== null || (incomingFps !== null && incomingFps > 0)) {
      const { framesChanged, fpsChanged } = this._recordLiveFrames(incomingFrames, {
        fps: incomingFps,
        timestampMs: incomingTs,
      });
      if (framesChanged || fpsChanged) {
        diagChanged = true;
      }
    }

    if (diagChanged) {
      this._renderBootDiagnostics({ updateHudStatus: false, force });
    }

    const patched = { ...metrics };
    if (this._liveFramesPresented > 0) {
      const effectiveFrames =
        incomingFrames !== null
          ? Math.max(incomingFrames, this._liveFramesPresented)
          : this._liveFramesPresented;
      patched.framesPresented = effectiveFrames;
      if (patched.totalFrames !== undefined) {
        patched.totalFrames = Math.max(
          toNonNegativeInt(patched.totalFrames, 0),
          effectiveFrames
        );
      }
      if (
        patched.fps === undefined ||
        patched.fps === null ||
        (typeof patched.fps === 'number' && (!Number.isFinite(patched.fps) || patched.fps <= 0))
      ) {
        patched.fps = this._liveFps;
      }
    } else if (incomingFrames !== null && patched.framesPresented === undefined) {
      patched.framesPresented = incomingFrames;
    }

    if (patched.bootSummary === undefined) {
      const pendingBootSummary = this._buildBootSummaryPayload(false);
      if (pendingBootSummary) {
        patched.bootSummary = pendingBootSummary;
      }
    } else if (isPlainObject(patched.bootSummary) && this._bootStage) {
      const incomingStage = normalizeBootMilestone(safeGet(patched.bootSummary, 'stage'));
      const incomingIdx = incomingStage ? BOOT_MILESTONE_INDEX.get(incomingStage) : -1;
      if (incomingIdx < this._bootStageIndex) {
        patched.bootSummary = {
          ...patched.bootSummary,
          stage: this._bootStage,
          progressPct: this._bootProgressPct,
          readyForSplash: this._bootReadyForSplash,
        };
      }
      this._lastPushedBootStage = this._bootStage;
      this._lastPushedProgressPct = this._bootProgressPct;
    }

    return patched;
  }

  _applyGpuTelemetry(input, outer = null) {
    if (!isPlainObject(input)) {
      return false;
    }
    const nestedStats = safeGet(input, 'stats');
    const primary = isPlainObject(nestedStats) ? nestedStats : input;
    const secondary = primary === input ? outer : input;

    let touched = false;

    const realPack = pickTelemetryField(primary, secondary, [
      'realPackDrawCount',
      'real_pack_draw_count',
      'realDrawCount',
      'packDrawCount',
    ]);
    if (realPack !== undefined) {
      this._gpuTelemetry.realPackDrawCount = toNonNegativeInt(realPack, 0);
      touched = true;
    }

    const fallback = pickTelemetryField(primary, secondary, [
      'fallbackDrawCount',
      'fallback_draw_count',
    ]);
    if (fallback !== undefined) {
      this._gpuTelemetry.fallbackDrawCount = toNonNegativeInt(fallback, 0);
      touched = true;
    }

    const present = pickTelemetryField(primary, secondary, [
      'presentCount',
      'present_count',
      'presents',
      'framesPresented',
      'frames_presented',
    ]);
    const explicitFps = pickTelemetryField(primary, secondary, ['fps']);
    const explicitTs = pickTelemetryField(primary, secondary, ['timestampMs', 'timestamp']);
    if (present !== undefined) {
      const parsedPresent = toNonNegativeInt(present, 0);
      this._gpuTelemetry.presentCount = parsedPresent;
      this._recordLiveFrames(parsedPresent, {
        fps: explicitFps,
        timestampMs: explicitTs,
      });
      touched = true;
    } else if (explicitFps !== undefined) {
      const { fpsChanged } = this._recordLiveFrames(null, { fps: explicitFps });
      if (fpsChanged) {
        touched = true;
      }
    }

    const unbound = pickTelemetryField(primary, secondary, [
      'unboundBindGroupCount',
      'unbound_bind_group_count',
    ]);
    if (unbound !== undefined) {
      this._gpuTelemetry.unboundBindGroupCount = toNonNegativeInt(unbound, 0);
      touched = true;
    }

    const skipped = pickTelemetryField(primary, secondary, [
      'skippedDrawCount',
      'skipped_draw_count',
    ]);
    if (skipped !== undefined) {
      this._gpuTelemetry.skippedDrawCount = toNonNegativeInt(skipped, 0);
      touched = true;
    }

    const validationErrs = pickTelemetryField(primary, secondary, [
      'validationErrorCount',
      'validation_error_count',
    ]);
    if (validationErrs !== undefined) {
      const sanitizedValErrs = toNonNegativeInt(validationErrs, 0);
      this._gpuTelemetry.validationErrorCount = sanitizedValErrs;
      if (sanitizedValErrs > this._validationErrorCount) {
        this._validationErrorCount = sanitizedValErrs;
      }
      touched = true;
    }

    const rawOnePercentLow = pickTelemetryField(primary, secondary, [
      'onePercentLowFps',
      'one_percent_low_fps',
      'fps1PercentLow',
      'low1PercentFps',
    ]);
    if (rawOnePercentLow !== undefined) {
      this._gpuTelemetry.onePercentLowFps = toNonNegativeFiniteNumber(rawOnePercentLow, null);
      touched = true;
    }

    const rawDirectP99 = pickTelemetryField(primary, secondary, ['p99FrameMs', 'p99_frame_ms']);
    if (rawDirectP99 !== undefined) {
      this._gpuTelemetry.p99FrameMs = toNonNegativeFiniteNumber(rawDirectP99, null);
      touched = true;
    } else {
      const rawPerfP99 = pickTelemetryField(primary, secondary, [
        'p99FrameTimeMs',
        'p99_frame_time_ms',
      ]);
      if (rawPerfP99 !== undefined) {
        const parsedPerfP99 = toNonNegativeFiniteNumber(rawPerfP99, null);
        if (parsedPerfP99 !== null && parsedPerfP99 > 0) {
          this._gpuTelemetry.p99FrameMs = parsedPerfP99;
          touched = true;
        }
      }
    }

    const rawZeroCopy = pickTelemetryField(primary, secondary, [
      'zeroCopyUploads',
      'zero_copy_uploads',
      'binkZeroCopyUploads',
      'bink_zero_copy_uploads',
      'zeroCopyFramesForwarded',
      'zero_copy_frames_forwarded',
      'zeroCopyFrames',
      'binkZeroCopyFrames',
    ]);
    if (rawZeroCopy !== undefined) {
      this._gpuTelemetry.zeroCopyUploads = toNonNegativeInt(rawZeroCopy, null);
      touched = true;
    }

    const rawGpuAdapter = pickTelemetryField(primary, secondary, [
      'gpuAdapter',
      'gpu_adapter',
      'adapterLabel',
      'adapter_label',
      'adapterInfo',
      'adapter_info',
    ]);
    if (rawGpuAdapter !== undefined) {
      this._gpuTelemetry.gpuAdapter = sanitizeGpuAdapterLabel(rawGpuAdapter);
      touched = true;
    }

    const rawBlackLoadingMs = pickTelemetryField(primary, secondary, [
      'blackLoadingMs',
      'black_loading_ms',
      'blackScreenMs',
      'black_screen_ms',
    ]);
    if (rawBlackLoadingMs !== undefined) {
      this._gpuTelemetry.blackLoadingMs = toNonNegativeFiniteNumber(rawBlackLoadingMs, null);
      touched = true;
    }

    const rawFirstVisibleFrameMs = pickTelemetryField(primary, secondary, [
      'firstVisibleFrameMs',
      'first_visible_frame_ms',
      'firstVisibleMs',
      'first_visible_ms',
      'timeToFirstVisibleFrameMs',
      'time_to_first_visible_frame_ms',
    ]);
    if (rawFirstVisibleFrameMs !== undefined) {
      this._gpuTelemetry.firstVisibleFrameMs = toNonNegativeFiniteNumber(
        rawFirstVisibleFrameMs,
        null
      );
      touched = true;
    }

    const rawLoadingAssets = pickTelemetryField(primary, secondary, [
      'loadingAssets',
      'loading_assets',
      'isLoadingAssets',
      'is_loading_assets',
    ]);
    if (rawLoadingAssets !== undefined) {
      this._gpuTelemetry.loadingAssets = rawLoadingAssets === true;
      touched = true;
    } else if (
      rawFirstVisibleFrameMs !== undefined &&
      this._gpuTelemetry.firstVisibleFrameMs !== null
    ) {
      this._gpuTelemetry.loadingAssets = false;
    }

    if (touched) {
      this._hasGpuTelemetry = true;
      this._gpuTelemetry.updatedAtMs = this._safeNowMs();
    }
    return touched;
  }

  _applyAudioTelemetry(input, outer = null) {
    if (!isPlainObject(input)) {
      return false;
    }
    const nestedStats = safeGet(input, 'stats');
    const primary = isPlainObject(nestedStats) ? nestedStats : input;
    const secondary = primary === input ? outer : input;

    let touched = false;
    let guestTouched = false;

    const outputState = pickTelemetryField(primary, secondary, [
      'outputState',
      'output_state',
      'state',
      'contextState',
    ]);
    if (outputState !== undefined) {
      this._audioTelemetry.outputState = normalizeAudioOutputState(outputState);
      touched = true;
    }

    const framesDrained = pickTelemetryField(primary, secondary, [
      'framesDrained',
      'frames_drained',
      'drainedFrames',
      'totalPulledFrames',
    ]);
    if (framesDrained !== undefined) {
      this._audioTelemetry.framesDrained = toNonNegativeInt(framesDrained, 0);
      touched = true;
    }

    const underruns = pickTelemetryField(primary, secondary, [
      'underruns',
      'underrunCount',
      'underrun_count',
    ]);
    if (underruns !== undefined) {
      this._audioTelemetry.underruns = toNonNegativeInt(underruns, 0);
      touched = true;
    }

    const overruns = pickTelemetryField(primary, secondary, [
      'overruns',
      'overrunCount',
      'overrun_count',
      'overflows',
    ]);
    if (overruns !== undefined) {
      this._audioTelemetry.overruns = toNonNegativeInt(overruns, 0);
      touched = true;
    }

    const guestRingAttached = pickTelemetryField(primary, secondary, [
      'guestRingAttached',
      'guest_ring_attached',
      'ringAttached',
      'pcmRingAttached',
    ]);
    if (guestRingAttached !== undefined) {
      this._audioTelemetry.guestRingAttached = guestRingAttached === true;
      guestTouched = true;
      touched = true;
    }

    const guestFramesDrained = pickTelemetryField(primary, secondary, [
      'guestFramesDrained',
      'guest_frames_drained',
      'guestDrainedFrames',
      'guest_drained_frames',
    ]);
    if (guestFramesDrained !== undefined) {
      this._audioTelemetry.guestFramesDrained = toNonNegativeInt(guestFramesDrained, 0);
      guestTouched = true;
      touched = true;
    }

    const guestUnderruns = pickTelemetryField(primary, secondary, [
      'guestUnderruns',
      'guest_underruns',
      'guestUnderrunCount',
      'guest_underrun_count',
    ]);
    if (guestUnderruns !== undefined) {
      this._audioTelemetry.guestUnderruns = toNonNegativeInt(guestUnderruns, 0);
      guestTouched = true;
      touched = true;
    }

    const guestPeakLevel = pickTelemetryField(primary, secondary, [
      'guestPeakLevel',
      'guest_peak_level',
      'guestPeak',
      'guest_peak',
    ]);
    if (guestPeakLevel !== undefined) {
      this._audioTelemetry.guestPeakLevel = toUnitIntervalNumber(guestPeakLevel, 0);
      guestTouched = true;
      touched = true;
    }

    if (guestTouched) {
      this._hasGuestAudioTelemetry = true;
    }
    if (touched) {
      this._hasAudioTelemetry = true;
      this._audioTelemetry.updatedAtMs = this._safeNowMs();
    }
    return touched;
  }

  _applyPthreadTelemetry(input, outer = null) {
    if (!isPlainObject(input)) {
      return false;
    }
    const nestedStats = safeGet(input, 'stats');
    const primary = isPlainObject(nestedStats) ? nestedStats : input;
    const secondary = primary === input ? outer : input;

    let touched = false;

    const spawned = pickTelemetryField(primary, secondary, [
      'pthreadSpawnedCount',
      'pthread_spawned_count',
      'spawnedCount',
      'spawned',
    ]);
    if (spawned !== undefined) {
      this._pthreadTelemetry.pthreadSpawnedCount = toNonNegativeInt(spawned, 0);
      touched = true;
    }

    const active = pickTelemetryField(primary, secondary, [
      'pthreadActiveCount',
      'pthread_active_count',
      'activeCount',
      'active',
    ]);
    if (active !== undefined) {
      this._pthreadTelemetry.pthreadActiveCount = toNonNegativeInt(active, 0);
      touched = true;
    }

    const peak = pickTelemetryField(primary, secondary, [
      'pthreadPeakCount',
      'pthread_peak_count',
      'peakCount',
      'peak',
    ]);
    if (peak !== undefined) {
      const sanitizedPeak = toNonNegativeInt(peak, 0);
      this._pthreadTelemetry.pthreadPeakCount = Math.max(
        this._pthreadTelemetry.pthreadPeakCount,
        sanitizedPeak,
        this._pthreadTelemetry.pthreadActiveCount
      );
      touched = true;
    } else if (active !== undefined) {
      this._pthreadTelemetry.pthreadPeakCount = Math.max(
        this._pthreadTelemetry.pthreadPeakCount,
        this._pthreadTelemetry.pthreadActiveCount
      );
    }

    const queueDepth = pickTelemetryField(primary, secondary, [
      'pthreadLoadQueueDepth',
      'pthread_load_queue_depth',
      'loadQueueDepth',
      'queueDepth',
    ]);
    if (queueDepth !== undefined) {
      this._pthreadTelemetry.pthreadLoadQueueDepth = toNonNegativeInt(queueDepth, 0);
      touched = true;
    }

    const dispatch = pickTelemetryField(primary, secondary, [
      'indirectDispatch',
      'indirect_dispatch',
      'dispatchMode',
      'dispatch_mode',
    ]);
    if (dispatch !== undefined) {
      this._pthreadTelemetry.indirectDispatch = normalizeIndirectDispatchMode(dispatch);
      touched = true;
    }

    if (touched) {
      this._hasPthreadTelemetry = true;
      this._pthreadTelemetry.updatedAtMs = this._safeNowMs();
    }
    return touched;
  }

  _applyFrameBudgetTelemetry(input, outer = null) {
    if (!isPlainObject(input)) {
      return false;
    }
    const nestedStats = safeGet(input, 'stats');
    const primary = isPlainObject(nestedStats) ? nestedStats : input;
    const secondary = primary === input ? outer : input;

    let touched = false;

    const rawRate = pickTelemetryField(primary, secondary, [
      'fps30ComplianceRate',
      'fps30_compliance_rate',
      'complianceRate30Fps',
      'compliance_rate_30fps',
      'fps30Compliance',
      'fps30_compliance',
    ]);
    if (rawRate !== undefined) {
      this._frameBudgetTelemetry.fps30ComplianceRate = sanitizeFps30ComplianceRate(rawRate);
      touched = true;
    }

    const rawBottleneck = pickTelemetryField(primary, secondary, [
      'dominantBottleneck',
      'dominant_bottleneck',
      'bottleneck',
    ]);
    if (rawBottleneck !== undefined) {
      this._frameBudgetTelemetry.dominantBottleneck = normalizeDominantBottleneck(rawBottleneck);
      touched = true;
    }

    if (touched) {
      this._hasFrameBudgetTelemetry =
        this._frameBudgetTelemetry.fps30ComplianceRate !== null ||
        this._frameBudgetTelemetry.dominantBottleneck !== null;
      this._frameBudgetTelemetry.updatedAtMs = this._safeNowMs();
    }
    return touched;
  }

  _applyMemoryTelemetry(input, outer = null) {
    if (!isPlainObject(input)) {
      return false;
    }
    const nestedStats = safeGet(input, 'stats');
    const primary = isPlainObject(nestedStats) ? nestedStats : input;
    const secondary = primary === input ? outer : input;

    let touched = false;

    const rawUsedMb = pickTelemetryField(primary, secondary, [
      'jsHeapUsedMb',
      'js_heap_used_mb',
      'heapUsedMb',
      'heap_used_mb',
    ]);
    if (rawUsedMb !== undefined) {
      this._memoryTelemetry.jsHeapUsedMb = toNonNegativeFiniteNumber(rawUsedMb, null);
      touched = true;
    } else {
      const rawUsedBytes = pickTelemetryField(primary, secondary, [
        'usedJSHeapSize',
        'jsHeapUsedBytes',
        'js_heap_used_bytes',
      ]);
      if (rawUsedBytes !== undefined) {
        const bytes = toNonNegativeFiniteNumber(rawUsedBytes, null);
        this._memoryTelemetry.jsHeapUsedMb =
          bytes !== null ? Math.round((bytes / BYTES_PER_MIB) * 100) / 100 : null;
        touched = true;
      }
    }

    const rawTotalMb = pickTelemetryField(primary, secondary, [
      'jsHeapTotalMb',
      'js_heap_total_mb',
      'heapTotalMb',
      'heap_total_mb',
    ]);
    if (rawTotalMb !== undefined) {
      this._memoryTelemetry.jsHeapTotalMb = toNonNegativeFiniteNumber(rawTotalMb, null);
      touched = true;
    } else {
      const rawTotalBytes = pickTelemetryField(primary, secondary, [
        'totalJSHeapSize',
        'jsHeapTotalBytes',
        'js_heap_total_bytes',
      ]);
      if (rawTotalBytes !== undefined) {
        const bytes = toNonNegativeFiniteNumber(rawTotalBytes, null);
        this._memoryTelemetry.jsHeapTotalMb =
          bytes !== null ? Math.round((bytes / BYTES_PER_MIB) * 100) / 100 : null;
        touched = true;
      }
    }

    if (touched) {
      this._hasMemoryTelemetry =
        this._memoryTelemetry.jsHeapUsedMb !== null ||
        this._memoryTelemetry.jsHeapTotalMb !== null;
      this._memoryTelemetry.updatedAtMs = this._safeNowMs();
    }
    return touched;
  }

  _applyIoTelemetry(input, outer = null, opts = {}) {
    if (!isPlainObject(input)) {
      return false;
    }
    const nestedStats = safeGet(input, 'stats');
    const primary = isPlainObject(nestedStats) ? nestedStats : input;
    const secondary = primary === input ? outer : input;

    const keys = opts.allowLanesAlias
      ? ['mailboxLanes', 'mailbox_lanes', 'ioMailboxLanes', 'io_mailbox_lanes', 'lanes']
      : ['mailboxLanes', 'mailbox_lanes', 'ioMailboxLanes', 'io_mailbox_lanes'];

    const rawLanes = pickTelemetryField(primary, secondary, keys);
    if (rawLanes === undefined) {
      return false;
    }
    this._ioTelemetry.mailboxLanes = sanitizeMailboxLanes(rawLanes);
    this._hasIoTelemetry = this._ioTelemetry.mailboxLanes !== null;
    this._ioTelemetry.updatedAtMs = this._safeNowMs();
    return true;
  }

  _applyStructuredTelemetryPayload(payload, opts = {}) {
    if (!isPlainObject(payload)) {
      return false;
    }
    let changed = false;

    const rawType = safeGet(payload, 'type');
    const msgType = typeof rawType === 'string' ? rawType.trim().toLowerCase() : '';

    const gpuNested =
      (isPlainObject(safeGet(payload, 'gpu')) && safeGet(payload, 'gpu')) ||
      (isPlainObject(safeGet(payload, 'gpuStats')) && safeGet(payload, 'gpuStats')) ||
      (isPlainObject(safeGet(payload, 'gpu-stats')) && safeGet(payload, 'gpu-stats')) ||
      null;

    if (gpuNested) {
      if (this._applyGpuTelemetry(gpuNested, payload)) {
        changed = true;
      }
    } else if (
      msgType === 'gpu-stats' ||
      msgType === 'gpu-telemetry' ||
      msgType === 'frame-presented' ||
      hasAnyDefinedKey(payload, GPU_TELEMETRY_FIELD_KEYS)
    ) {
      if (this._applyGpuTelemetry(payload)) {
        changed = true;
      }
    } else if (opts.includeFrameStatsPresent === true) {
      const frameStats = safeGet(payload, 'frameStats');
      if (isPlainObject(frameStats)) {
        const framePresent = pickFirstDefined(frameStats, [
          'presentCount',
          'present_count',
          'framesPresented',
          'frames_presented',
        ]);
        const frameFps = pickFirstDefined(frameStats, ['fps']);
        const frameTs = pickFirstDefined(frameStats, ['timestampMs', 'timestamp']);
        if (framePresent !== undefined) {
          const parsedPresent = toNonNegativeInt(framePresent, 0);
          this._gpuTelemetry.presentCount = parsedPresent;
          this._gpuTelemetry.updatedAtMs = this._safeNowMs();
          this._recordLiveFrames(parsedPresent, {
            fps: frameFps,
            timestampMs: frameTs,
          });
          changed = true;
        }
      }
    }

    const framesNested =
      (isPlainObject(safeGet(payload, 'frames')) && safeGet(payload, 'frames')) ||
      (isPlainObject(safeGet(payload, 'frameStats')) && safeGet(payload, 'frameStats')) ||
      (isPlainObject(safeGet(payload, 'perf')) && safeGet(payload, 'perf')) ||
      null;
    if (framesNested) {
      const nestedLow = pickFirstDefined(framesNested, [
        'onePercentLowFps',
        'one_percent_low_fps',
        'fps1PercentLow',
        'low1PercentFps',
      ]);
      if (nestedLow !== undefined) {
        this._gpuTelemetry.onePercentLowFps = toNonNegativeFiniteNumber(nestedLow, null);
        this._hasGpuTelemetry = true;
        this._gpuTelemetry.updatedAtMs = this._safeNowMs();
        changed = true;
      }
      const nestedDirectP99 = pickFirstDefined(framesNested, ['p99FrameMs', 'p99_frame_ms']);
      if (nestedDirectP99 !== undefined) {
        this._gpuTelemetry.p99FrameMs = toNonNegativeFiniteNumber(nestedDirectP99, null);
        this._hasGpuTelemetry = true;
        this._gpuTelemetry.updatedAtMs = this._safeNowMs();
        changed = true;
      } else {
        const nestedPerfP99 = pickFirstDefined(framesNested, [
          'p99FrameTimeMs',
          'p99_frame_time_ms',
        ]);
        if (nestedPerfP99 !== undefined) {
          const parsed = toNonNegativeFiniteNumber(nestedPerfP99, null);
          if (parsed !== null && parsed > 0) {
            this._gpuTelemetry.p99FrameMs = parsed;
            this._hasGpuTelemetry = true;
            this._gpuTelemetry.updatedAtMs = this._safeNowMs();
            changed = true;
          }
        }
      }
    }

    const binkNested =
      (isPlainObject(safeGet(payload, 'bink')) && safeGet(payload, 'bink')) ||
      (isPlainObject(safeGet(payload, 'binkStats')) && safeGet(payload, 'binkStats')) ||
      (isPlainObject(safeGet(payload, 'binkPresenter')) && safeGet(payload, 'binkPresenter')) ||
      null;
    if (binkNested) {
      const nestedZeroCopy = pickFirstDefined(binkNested, [
        'zeroCopyUploads',
        'zero_copy_uploads',
        'binkZeroCopyUploads',
        'bink_zero_copy_uploads',
        'zeroCopyFramesForwarded',
        'zero_copy_frames_forwarded',
        'zeroCopyFrames',
        'binkZeroCopyFrames',
      ]);
      if (nestedZeroCopy !== undefined) {
        this._gpuTelemetry.zeroCopyUploads = toNonNegativeInt(nestedZeroCopy, null);
        this._hasGpuTelemetry = true;
        this._gpuTelemetry.updatedAtMs = this._safeNowMs();
        changed = true;
      }
    }

    const audioNested =
      (isPlainObject(safeGet(payload, 'audio')) && safeGet(payload, 'audio')) ||
      (isPlainObject(safeGet(payload, 'audioStats')) && safeGet(payload, 'audioStats')) ||
      (isPlainObject(safeGet(payload, 'audio-stats')) && safeGet(payload, 'audio-stats')) ||
      null;

    if (audioNested) {
      if (this._applyAudioTelemetry(audioNested)) {
        changed = true;
      }
      if (hasAnyDefinedKey(payload, GUEST_AUDIO_TELEMETRY_FIELD_KEYS)) {
        if (this._applyAudioTelemetry(payload)) {
          changed = true;
        }
      }
    } else if (
      msgType === 'audio-stats' ||
      msgType === 'audio-telemetry' ||
      hasAnyDefinedKey(payload, AUDIO_TELEMETRY_FIELD_KEYS)
    ) {
      if (this._applyAudioTelemetry(payload)) {
        changed = true;
      }
    }

    const pthreadNested =
      (isPlainObject(safeGet(payload, 'pthread')) && safeGet(payload, 'pthread')) ||
      (isPlainObject(safeGet(payload, 'pthreads')) && safeGet(payload, 'pthreads')) ||
      (isPlainObject(safeGet(payload, 'pthreadStats')) && safeGet(payload, 'pthreadStats')) ||
      (isPlainObject(safeGet(payload, 'pthread-stats')) && safeGet(payload, 'pthread-stats')) ||
      (isPlainObject(safeGet(payload, 'wasmLoader')) && safeGet(payload, 'wasmLoader')) ||
      null;

    if (pthreadNested) {
      if (this._applyPthreadTelemetry(pthreadNested)) {
        changed = true;
      }
    } else if (
      msgType === 'pthread-stats' ||
      msgType === 'pthread-telemetry' ||
      msgType === 'wasm-stats' ||
      msgType === 'loader-stats' ||
      hasAnyDefinedKey(payload, PTHREAD_TELEMETRY_FIELD_KEYS)
    ) {
      if (this._applyPthreadTelemetry(payload)) {
        changed = true;
      }
    }

    const frameBudgetNested =
      (isPlainObject(safeGet(payload, 'frameBudget')) && safeGet(payload, 'frameBudget')) ||
      (isPlainObject(safeGet(payload, 'frame_budget')) && safeGet(payload, 'frame_budget')) ||
      (isPlainObject(safeGet(payload, 'budget')) && safeGet(payload, 'budget')) ||
      null;
    if (frameBudgetNested) {
      if (this._applyFrameBudgetTelemetry(frameBudgetNested, payload)) {
        changed = true;
      }
    } else {
      if (framesNested && hasAnyDefinedKey(framesNested, FRAME_BUDGET_TELEMETRY_FIELD_KEYS)) {
        if (this._applyFrameBudgetTelemetry(framesNested)) {
          changed = true;
        }
      }
      if (gpuNested && hasAnyDefinedKey(gpuNested, FRAME_BUDGET_TELEMETRY_FIELD_KEYS)) {
        if (this._applyFrameBudgetTelemetry(gpuNested)) {
          changed = true;
        }
      }
      if (
        msgType === 'frame-budget' ||
        msgType === 'perf-stats' ||
        hasAnyDefinedKey(payload, FRAME_BUDGET_TELEMETRY_FIELD_KEYS)
      ) {
        if (this._applyFrameBudgetTelemetry(payload)) {
          changed = true;
        }
      }
    }

    const memoryNested =
      (isPlainObject(safeGet(payload, 'memory')) && safeGet(payload, 'memory')) ||
      (isPlainObject(safeGet(payload, 'memoryStats')) && safeGet(payload, 'memoryStats')) ||
      (isPlainObject(safeGet(payload, 'memory_stats')) && safeGet(payload, 'memory_stats')) ||
      (isPlainObject(safeGet(payload, 'heap')) && safeGet(payload, 'heap')) ||
      null;
    if (memoryNested) {
      if (this._applyMemoryTelemetry(memoryNested, payload)) {
        changed = true;
      }
    } else {
      if (framesNested && hasAnyDefinedKey(framesNested, MEMORY_TELEMETRY_FIELD_KEYS)) {
        if (this._applyMemoryTelemetry(framesNested)) {
          changed = true;
        }
      }
      if (
        msgType === 'memory-stats' ||
        msgType === 'perf-stats' ||
        hasAnyDefinedKey(payload, MEMORY_TELEMETRY_FIELD_KEYS)
      ) {
        if (this._applyMemoryTelemetry(payload)) {
          changed = true;
        }
      }
    }

    const ioNested =
      (isPlainObject(safeGet(payload, 'io')) && safeGet(payload, 'io')) ||
      (isPlainObject(safeGet(payload, 'ioStats')) && safeGet(payload, 'ioStats')) ||
      (isPlainObject(safeGet(payload, 'io_stats')) && safeGet(payload, 'io_stats')) ||
      (isPlainObject(safeGet(payload, 'httpfs')) && safeGet(payload, 'httpfs')) ||
      (isPlainObject(safeGet(payload, 'httpfsStats')) && safeGet(payload, 'httpfsStats')) ||
      (isPlainObject(safeGet(payload, 'httpfs_stats')) && safeGet(payload, 'httpfs_stats')) ||
      null;
    if (ioNested) {
      if (this._applyIoTelemetry(ioNested, payload, { allowLanesAlias: true })) {
        changed = true;
      }
    } else {
      if (framesNested && hasAnyDefinedKey(framesNested, IO_TELEMETRY_FIELD_KEYS)) {
        if (this._applyIoTelemetry(framesNested)) {
          changed = true;
        }
      }
      if (
        msgType === 'io-stats' ||
        msgType === 'httpfs-stats' ||
        hasAnyDefinedKey(payload, IO_TELEMETRY_FIELD_KEYS)
      ) {
        if (this._applyIoTelemetry(payload)) {
          changed = true;
        }
      }
    }

    return changed;
  }

  /**
   * Returns the current sanitized WebGPU adapter label (`'nvidia / turing'`, etc.) or `null`.
   *
   * @returns {string|null}
   */
  getGpuAdapter() {
    return this._gpuTelemetry.gpuAdapter;
  }

  /**
   * Updates the active WebGPU adapter label (sanitized lowercase vendor/arch, max 96 chars,
   * no host paths or secrets) and re-renders HUD diagnostics.
   *
   * @param {string|object|null|undefined} label
   * @param {{ updateHudStatus?: boolean, force?: boolean }} [options]
   * @returns {string|null} Sanitized adapter label or `null`.
   */
  setGpuAdapter(label, options = {}) {
    if (this._destroyed) {
      return this._gpuTelemetry.gpuAdapter;
    }
    const opts = isPlainObject(options) ? options : {};
    const sanitized = sanitizeGpuAdapterLabel(label);
    this._gpuTelemetry.gpuAdapter = sanitized;
    if (sanitized !== null) {
      this._hasGpuTelemetry = true;
    }
    this._gpuTelemetry.updatedAtMs = this._safeNowMs();
    this._renderBootDiagnostics({
      updateHudStatus: opts.updateHudStatus === true,
      force: opts.force === true,
    });
    return sanitized;
  }

  /**
   * Updates honest GPU worker telemetry (`gpu-stats` / W83) keeping real-pack draws
   * strictly separate from fallback draws, sanitizing all counters to non-negative integers.
   *
   * @param {object|null|undefined} stats
   * @param {{ updateHudStatus?: boolean }} [options]
   */
  updateGpuStats(stats, options = {}) {
    if (this._destroyed) {
      return this.getTelemetry().gpu;
    }
    const opts = isPlainObject(options) ? options : {};
    let gpuChanged = this._applyGpuTelemetry(stats);
    if (isPlainObject(stats)) {
      const fbNested =
        (isPlainObject(safeGet(stats, 'frameBudget')) && safeGet(stats, 'frameBudget')) ||
        (isPlainObject(safeGet(stats, 'frame_budget')) && safeGet(stats, 'frame_budget')) ||
        null;
      if (fbNested) {
        if (this._applyFrameBudgetTelemetry(fbNested, stats)) {
          gpuChanged = true;
        }
      } else if (hasAnyDefinedKey(stats, FRAME_BUDGET_TELEMETRY_FIELD_KEYS)) {
        if (this._applyFrameBudgetTelemetry(stats)) {
          gpuChanged = true;
        }
      }

      const memNested =
        (isPlainObject(safeGet(stats, 'memory')) && safeGet(stats, 'memory')) ||
        (isPlainObject(safeGet(stats, 'memoryStats')) && safeGet(stats, 'memoryStats')) ||
        null;
      if (memNested) {
        if (this._applyMemoryTelemetry(memNested, stats)) {
          gpuChanged = true;
        }
      } else if (hasAnyDefinedKey(stats, MEMORY_TELEMETRY_FIELD_KEYS)) {
        if (this._applyMemoryTelemetry(stats)) {
          gpuChanged = true;
        }
      }

      const ioNested =
        (isPlainObject(safeGet(stats, 'io')) && safeGet(stats, 'io')) ||
        (isPlainObject(safeGet(stats, 'ioStats')) && safeGet(stats, 'ioStats')) ||
        null;
      if (ioNested) {
        if (this._applyIoTelemetry(ioNested, stats, { allowLanesAlias: true })) {
          gpuChanged = true;
        }
      } else if (hasAnyDefinedKey(stats, IO_TELEMETRY_FIELD_KEYS)) {
        if (this._applyIoTelemetry(stats)) {
          gpuChanged = true;
        }
      }
    }
    const appSync = this._syncFromAppSources();
    if (gpuChanged || appSync.bootStageChanged || appSync.telemetryChanged) {
      this._renderBootDiagnostics({
        updateHudStatus: opts.updateHudStatus === true,
        force: opts.force === true,
      });
      this._syncLiveHudReadouts({
        includeFrames: true,
        updateHudStatus: opts.updateHudStatus,
      });
    }
    return this.getTelemetry().gpu;
  }

  /**
   * Updates 30 FPS frame-budget diagnostics (`fps30ComplianceRate` clamped `[0, 1]`,
   * `dominantBottleneck` allowlisted) and re-renders the Diagnostics panel.
   *
   * @param {{ fps30ComplianceRate?: number|null, dominantBottleneck?: string|null }|null|undefined} stats
   * @param {{ updateHudStatus?: boolean, force?: boolean }} [options]
   */
  setFrameBudgetStats(stats, options = {}) {
    if (this._destroyed) {
      return this.getTelemetry().frameBudget;
    }
    const opts = isPlainObject(options) ? options : {};
    let changed = false;
    if (stats === null) {
      this._frameBudgetTelemetry = createInitialFrameBudgetTelemetry();
      this._hasFrameBudgetTelemetry = false;
      this._frameBudgetTelemetry.updatedAtMs = this._safeNowMs();
      changed = true;
    } else if (isPlainObject(stats)) {
      changed = this._applyFrameBudgetTelemetry(stats);
    }
    if (changed || opts.force === true) {
      this._renderBootDiagnostics({
        updateHudStatus: opts.updateHudStatus === true,
        force: opts.force === true,
      });
    }
    return this.getTelemetry().frameBudget;
  }

  /**
   * Updates JS heap memory diagnostics (`jsHeapUsedMb`, `jsHeapTotalMb`, non-negative
   * finite MB) and re-renders the Diagnostics panel.
   *
   * @param {{ jsHeapUsedMb?: number|null, jsHeapTotalMb?: number|null }|null|undefined} stats
   * @param {{ updateHudStatus?: boolean, force?: boolean }} [options]
   */
  setMemoryStats(stats, options = {}) {
    if (this._destroyed) {
      return this.getTelemetry().memory;
    }
    const opts = isPlainObject(options) ? options : {};
    let changed = false;
    if (stats === null) {
      this._memoryTelemetry = createInitialMemoryTelemetry();
      this._hasMemoryTelemetry = false;
      this._memoryTelemetry.updatedAtMs = this._safeNowMs();
      changed = true;
    } else if (isPlainObject(stats)) {
      changed = this._applyMemoryTelemetry(stats);
    }
    if (changed || opts.force === true) {
      this._renderBootDiagnostics({
        updateHudStatus: opts.updateHudStatus === true,
        force: opts.force === true,
      });
    }
    return this.getTelemetry().memory;
  }

  /**
   * Updates HTTPFS I/O mailbox lane count (`lanes` sanitized to a positive integer
   * or `null`) and re-renders the Diagnostics panel.
   *
   * @param {number|object|null|undefined} lanes
   * @param {{ updateHudStatus?: boolean, force?: boolean }} [options]
   */
  setIoMailboxLanes(lanes, options = {}) {
    if (this._destroyed) {
      return this.getTelemetry().io;
    }
    const opts = isPlainObject(options) ? options : {};
    if (isPlainObject(lanes)) {
      const applied = this._applyIoTelemetry(lanes, null, { allowLanesAlias: true });
      if (!applied) {
        this._ioTelemetry.mailboxLanes = null;
        this._hasIoTelemetry = false;
        this._ioTelemetry.updatedAtMs = this._safeNowMs();
      }
    } else {
      const sanitized = sanitizeMailboxLanes(lanes);
      this._ioTelemetry.mailboxLanes = sanitized;
      this._hasIoTelemetry = sanitized !== null;
      this._ioTelemetry.updatedAtMs = this._safeNowMs();
    }
    this._renderBootDiagnostics({
      updateHudStatus: opts.updateHudStatus === true,
      force: opts.force === true,
    });
    return this.getTelemetry().io;
  }

  /**
   * Ingests performance & frame-budget telemetry (30 FPS compliance rate, dominant
   * bottleneck, JS heap MB, HTTPFS mailbox lanes, plus 1%-low FPS / p99 frame time /
   * FPS) without bloating the compact primary HUD card.
   *
   * @param {object|null|undefined} stats
   * @param {{ updateHudStatus?: boolean, force?: boolean }} [options]
   */
  updatePerfStats(stats, options = {}) {
    if (this._destroyed) {
      return this.getTelemetry();
    }
    const opts = isPlainObject(options) ? options : {};
    let changed = false;
    if (isPlainObject(stats)) {
      if (this._applyStructuredTelemetryPayload(stats, { includeFrameStatsPresent: true })) {
        changed = true;
      }
      const explicitFps = toNonNegativeFiniteNumber(safeGet(stats, 'fps'), null);
      if (explicitFps !== null && explicitFps > 0) {
        const { fpsChanged } = this._recordLiveFrames(null, { fps: explicitFps });
        if (fpsChanged) {
          changed = true;
        }
      }
    }
    if (changed || opts.force === true) {
      this._renderBootDiagnostics({
        updateHudStatus: opts.updateHudStatus === true,
        force: opts.force === true,
      });
      if (this._liveFramesPresented > 0) {
        this._syncLiveHudReadouts({
          includeFrames: true,
          updateHudStatus: opts.updateHudStatus,
        });
      }
    }
    return this.getTelemetry();
  }

  /**
   * Handles `splash-frame-presented` notifications from `app_bootstrap.js` / `wgpu_worker`,
   * updating live Frames, rolling FPS, and advancing the boot milestone to `FIRST_PRESENT`.
   *
   * @param {object|null|undefined} event
   * @param {{ updateHudStatus?: boolean, force?: boolean }} [options]
   */
  onSplashFramePresented(event, options = {}) {
    if (this._destroyed) {
      return this.getTelemetry().gpu;
    }
    if (!isPlainObject(event)) {
      return this.getTelemetry().gpu;
    }
    if (safeGet(event, 'ok') === false) {
      return this.getTelemetry().gpu;
    }
    const opts = isPlainObject(options) ? options : {};
    const reportedFrames = toNonNegativeInt(
      pickFirstDefined(event, [
        'framesPresented',
        'frames_presented',
        'presentCount',
        'present_count',
        'frameCount',
      ]),
      null
    );
    const nextFrames =
      reportedFrames !== null
        ? Math.max(this._liveFramesPresented, reportedFrames)
        : this._liveFramesPresented + 1;
    const explicitFps = toNonNegativeFiniteNumber(safeGet(event, 'fps'), null);
    const explicitTs = toNonNegativeFiniteNumber(
      safeGet(event, 'timestampMs') ?? safeGet(event, 'timestamp'),
      null
    );
    const rawZeroCopy = pickFirstDefined(event, [
      'zeroCopyUploads',
      'zero_copy_uploads',
      'binkZeroCopyUploads',
      'zeroCopyFramesForwarded',
      'zero_copy_frames_forwarded',
    ]);
    if (rawZeroCopy !== undefined) {
      this._gpuTelemetry.zeroCopyUploads = toNonNegativeInt(rawZeroCopy, null);
    }

    const rawGpuAdapter = pickFirstDefined(event, [
      'gpuAdapter',
      'gpu_adapter',
      'adapterLabel',
      'adapter_label',
      'adapterInfo',
      'adapter_info',
    ]);
    if (rawGpuAdapter !== undefined) {
      this._gpuTelemetry.gpuAdapter = sanitizeGpuAdapterLabel(rawGpuAdapter);
    }

    const rawBlackLoadingMs = pickFirstDefined(event, [
      'blackLoadingMs',
      'black_loading_ms',
      'blackScreenMs',
      'black_screen_ms',
    ]);
    if (rawBlackLoadingMs !== undefined) {
      this._gpuTelemetry.blackLoadingMs = toNonNegativeFiniteNumber(rawBlackLoadingMs, null);
    }

    const rawFirstVisibleFrameMs = pickFirstDefined(event, [
      'firstVisibleFrameMs',
      'first_visible_frame_ms',
      'firstVisibleMs',
      'first_visible_ms',
      'timeToFirstVisibleFrameMs',
      'time_to_first_visible_frame_ms',
    ]);
    if (rawFirstVisibleFrameMs !== undefined) {
      this._gpuTelemetry.firstVisibleFrameMs = toNonNegativeFiniteNumber(
        rawFirstVisibleFrameMs,
        null
      );
    }

    const rawLoadingAssets = pickFirstDefined(event, [
      'loadingAssets',
      'loading_assets',
      'isLoadingAssets',
      'is_loading_assets',
    ]);
    if (rawLoadingAssets !== undefined) {
      this._gpuTelemetry.loadingAssets = rawLoadingAssets === true;
    } else if (
      rawFirstVisibleFrameMs !== undefined &&
      this._gpuTelemetry.firstVisibleFrameMs !== null
    ) {
      this._gpuTelemetry.loadingAssets = false;
    }

    this._recordLiveFrames(nextFrames, {
      fps: explicitFps,
      timestampMs: explicitTs,
    });
    this._hasGpuTelemetry = true;
    this._gpuTelemetry.updatedAtMs = this._safeNowMs();

    const firstPresent =
      safeGet(event, 'firstPresent') === true || nextFrames >= 1;
    if (firstPresent) {
      this._applyBootStage({
        stage: 'FIRST_PRESENT',
        progressPct: 100,
        readyForSplash: true,
      });
    }
    this._syncFromAppSources();
    this._renderBootDiagnostics({
      updateHudStatus: opts.updateHudStatus === true,
      force: opts.force === true,
    });
    this._syncLiveHudReadouts({
      includeFrames: true,
      updateHudStatus: opts.updateHudStatus,
    });
    return this.getTelemetry().gpu;
  }

  recordSplashFramePresented(event, options = {}) {
    return this.onSplashFramePresented(event, options);
  }

  handleSplashFramePresented(event, options = {}) {
    return this.onSplashFramePresented(event, options);
  }

  /**
   * Records a presented frame count (or increments by 1 when omitted) and updates
   * live Frames, rolling FPS, and optional `FIRST_PRESENT` milestone readouts.
   *
   * @param {number|object|null|undefined} [input]
   * @param {{ updateHudStatus?: boolean }} [options]
   */
  recordFramePresented(input, options = {}) {
    if (this._destroyed) {
      return this.getTelemetry().gpu;
    }
    if (typeof input === 'number' || typeof input === 'bigint') {
      const count = toNonNegativeInt(input, null);
      return this.onSplashFramePresented(
        {
          ok: true,
          framesPresented: count !== null ? count : this._liveFramesPresented + 1,
          firstPresent: count === null ? true : count >= 1,
        },
        options
      );
    }
    if (isPlainObject(input)) {
      return this.onSplashFramePresented(input, options);
    }
    return this.onSplashFramePresented(
      {
        ok: true,
        framesPresented: this._liveFramesPresented + 1,
        firstPresent: true,
      },
      options
    );
  }

  /**
   * Processes a HUD metrics update (e.g. from `PerfMonitor` or runtime telemetry),
   * protecting live `Frames` and rolling `FPS` from being overwritten back to `0`
   * by `PerfMonitor` ticks while `wgpu_worker` presents frames.
   *
   * @param {object|null|undefined} metrics
   * @param {{ updateHudStatus?: boolean }} [options]
   */
  updateMetrics(metrics, options = {}) {
    if (this._destroyed || !isPlainObject(metrics)) {
      return this.getTelemetry();
    }
    const opts = isPlainObject(options) ? options : {};
    const patched = this._interceptExternalHudMetrics(metrics, opts);
    if (opts.updateHudStatus === true) {
      this._renderBootDiagnostics({ updateHudStatus: true, force: opts.force === true });
    }
    this._pushHudMetrics(patched);
    return this.getTelemetry();
  }

  /**
   * Updates honest Audio worker telemetry (`audio-stats` / W84) with normalized
   * `outputState` (`'active' | 'suspended' | 'headless-drain'`) and sanitized counters.
   *
   * @param {object|null|undefined} stats
   * @param {{ updateHudStatus?: boolean, force?: boolean }} [options]
   */
  updateAudioStats(stats, options = {}) {
    if (this._destroyed) {
      return this.getTelemetry().audio;
    }
    const opts = isPlainObject(options) ? options : {};
    if (this._applyAudioTelemetry(stats)) {
      this._renderBootDiagnostics({
        updateHudStatus: opts.updateHudStatus === true,
        force: opts.force === true,
      });
    }
    return this.getTelemetry().audio;
  }

  /**
   * Updates honest Pthread / Wasm Loader telemetry (W86 / R37) including active/peak
   * thread counts, load queue depth, and normalized indirect dispatch mode.
   *
   * @param {object|null|undefined} stats
   * @param {{ updateHudStatus?: boolean, force?: boolean }} [options]
   */
  updatePthreadStats(stats, options = {}) {
    if (this._destroyed) {
      return this.getTelemetry().pthread;
    }
    const opts = isPlainObject(options) ? options : {};
    if (this._applyPthreadTelemetry(stats)) {
      this._renderBootDiagnostics({
        updateHudStatus: opts.updateHudStatus === true,
        force: opts.force === true,
      });
    }
    return this.getTelemetry().pthread;
  }

  /**
   * Ingests a unified telemetry update or worker stats message (`gpu-stats`,
   * `audio-stats`, `pthread-stats`, or a combined `{ gpu, audio, pthread }` object).
   *
   * @param {object|null|undefined} payload
   * @param {{ updateHudStatus?: boolean, force?: boolean }} [options]
   */
  updateTelemetry(payload, options = {}) {
    if (this._destroyed) {
      return this.getTelemetry();
    }
    const opts = isPlainObject(options) ? options : {};
    let bootStageChanged = false;
    if (isPlainObject(payload)) {
      const summary = safeGet(payload, 'bootSummary');
      if (isPlainObject(summary) && this._applyBootStage(summary)) {
        bootStageChanged = true;
      } else if (safeGet(payload, 'milestone') !== undefined || safeGet(payload, 'stage') !== undefined) {
        if (
          this._applyBootStage({
            stage: safeGet(payload, 'milestone') ?? safeGet(payload, 'stage'),
            progressPct: safeGet(payload, 'progressPct'),
            readyForSplash: safeGet(payload, 'readyForSplash'),
            elapsedMs: safeGet(payload, 'elapsedMs'),
          })
        ) {
          bootStageChanged = true;
        }
      }
    }
    const structuredChanged = this._applyStructuredTelemetryPayload(payload, {
      includeFrameStatsPresent: true,
    });
    const appSync = this._syncFromAppSources();
    if (structuredChanged || bootStageChanged || appSync.bootStageChanged || appSync.telemetryChanged) {
      this._renderBootDiagnostics({
        updateHudStatus: opts.updateHudStatus === true,
        force: opts.force === true,
      });
      this._syncLiveHudReadouts({
        includeFrames: this._liveFramesPresented > 0,
        updateHudStatus: opts.updateHudStatus,
      });
    } else if (opts.force === true) {
      this._renderBootDiagnostics({
        updateHudStatus: opts.updateHudStatus === true,
        force: true,
      });
    }
    return this.getTelemetry();
  }

  /**
   * Resets tracked GPU, Audio, and Pthread telemetry back to clean initial defaults.
   *
   * @param {{ updateHudStatus?: boolean, force?: boolean }} [options]
   */
  resetTelemetry(options = {}) {
    if (this._destroyed) {
      return this.getTelemetry();
    }
    const opts = isPlainObject(options) ? options : {};
    this._gpuTelemetry = createInitialGpuTelemetry();
    this._audioTelemetry = createInitialAudioTelemetry();
    this._pthreadTelemetry = createInitialPthreadTelemetry();
    this._frameBudgetTelemetry = createInitialFrameBudgetTelemetry();
    this._memoryTelemetry = createInitialMemoryTelemetry();
    this._ioTelemetry = createInitialIoTelemetry();
    this._hasGpuTelemetry = false;
    this._hasAudioTelemetry = false;
    this._hasGuestAudioTelemetry = false;
    this._hasPthreadTelemetry = false;
    this._hasFrameBudgetTelemetry = false;
    this._hasMemoryTelemetry = false;
    this._hasIoTelemetry = false;
    this._liveFramesPresented = 0;
    this._liveFps = 0;
    this._fpsSamples = [];
    this._lastConciseHudStatus = null;
    this._renderBootDiagnostics({
      updateHudStatus: opts.updateHudStatus === true,
      force: opts.force !== false,
    });
    return this.getTelemetry();
  }

  /**
   * Returns a defensive snapshot of sanitized GPU, Audio, and Pthread telemetry.
   */
  getTelemetry() {
    const g = this._gpuTelemetry;
    const a = this._audioTelemetry;
    const p = this._pthreadTelemetry;
    const fb = this._frameBudgetTelemetry;
    const m = this._memoryTelemetry;
    const io = this._ioTelemetry;

    const effectivePresentCount = Math.max(g.presentCount, this._liveFramesPresented);
    const totalDrawCount = g.realPackDrawCount + g.fallbackDrawCount;
    const hasRealPackDraws = g.realPackDrawCount > 0;
    const hasFallbackDraws = g.fallbackDrawCount > 0;

    let gpuStatusText = '';
    if (hasRealPackDraws || hasFallbackDraws) {
      const drawPart = hasFallbackDraws
        ? `Draws: ${g.realPackDrawCount} real, ${g.fallbackDrawCount} fallback`
        : `Draws: ${g.realPackDrawCount} real`;
      gpuStatusText =
        effectivePresentCount > 0
          ? `${drawPart} (${effectivePresentCount} present${effectivePresentCount === 1 ? '' : 's'})`
          : drawPart;
    } else if (this._hasGpuTelemetry && effectivePresentCount > 0) {
      gpuStatusText = `Presents: ${effectivePresentCount} (0 draws)`;
    }

    let audioStatusText = '';
    const guestAudioStatusText = this._hasGuestAudioTelemetry
      ? formatGuestAudioStatus(a)
      : '';
    if (this._hasAudioTelemetry) {
      const details = [];
      if (a.framesDrained > 0) {
        details.push(`${a.framesDrained} drained`);
      }
      if (a.underruns > 0) {
        details.push(`${a.underruns} underrun${a.underruns === 1 ? '' : 's'}`);
      }
      if (a.overruns > 0) {
        details.push(`${a.overruns} overrun${a.overruns === 1 ? '' : 's'}`);
      }
      const baseAudioText =
        details.length > 0
          ? `Audio: ${a.outputState} (${details.join(', ')})`
          : `Audio: ${a.outputState}`;
      audioStatusText = guestAudioStatusText
        ? `${baseAudioText} · ${guestAudioStatusText}`
        : baseAudioText;
    }

    let pthreadStatusText = '';
    if (p.pthreadActiveCount > 0 || p.pthreadPeakCount > 0 || p.pthreadSpawnedCount > 0) {
      const peak = Math.max(p.pthreadPeakCount, p.pthreadActiveCount);
      const base = `Threads: ${p.pthreadActiveCount}/${peak}`;
      pthreadStatusText =
        p.indirectDispatch !== INDIRECT_DISPATCH_MODE.UNKNOWN
          ? `${base} [${p.indirectDispatch}]`
          : base;
    }

    const onePercentLowFpsText = formatOnePercentLowFps(g.onePercentLowFps);
    const p99FrameMsText = formatP99FrameMs(g.p99FrameMs);
    const zeroCopyUploadsText = formatZeroCopyUploads(g.zeroCopyUploads);
    const perfSummaryText = formatPerfMetricsSummary(g);
    const gpuAdapterText = formatGpuAdapter(g.gpuAdapter);
    const blackLoadingText = formatBlackLoadingMs(g.blackLoadingMs);
    const firstVisibleFrameText = formatFirstVisibleFrameMs(g.firstVisibleFrameMs);
    const loadingStatusText = formatLoadingStatus(g);

    const frameBudgetStatusText = formatFrameBudgetStatus(fb);
    const frameBudgetValueText = formatFrameBudgetValueText(
      fb.fps30ComplianceRate,
      fb.dominantBottleneck
    );
    const frameBudgetState = resolveFrameBudgetState(
      fb.fps30ComplianceRate,
      fb.dominantBottleneck
    );

    const memoryStatusText = formatMemoryStats(m);
    const memoryValueText = formatMemoryValueText(m.jsHeapUsedMb, m.jsHeapTotalMb);
    const memoryState = resolveMemoryState(m.jsHeapUsedMb, m.jsHeapTotalMb);

    const ioMailboxStatusText = formatIoMailboxLanes(io.mailboxLanes);
    const ioMailboxState = resolveIoMailboxState(io.mailboxLanes);

    const diagnosticsSummaryText = formatDiagnosticsSummary({
      frameBudget: fb,
      memory: m,
      io,
    });

    return {
      gpu: {
        realPackDrawCount: g.realPackDrawCount,
        fallbackDrawCount: g.fallbackDrawCount,
        presentCount: effectivePresentCount,
        framesPresented: effectivePresentCount,
        fps: this._liveFps,
        onePercentLowFps: g.onePercentLowFps,
        p99FrameMs: g.p99FrameMs,
        zeroCopyUploads: g.zeroCopyUploads,
        onePercentLowFpsText,
        p99FrameMsText,
        zeroCopyUploadsText,
        perfSummaryText,
        gpuAdapter: g.gpuAdapter,
        gpuAdapterText,
        blackLoadingMs: g.blackLoadingMs,
        firstVisibleFrameMs: g.firstVisibleFrameMs,
        loadingAssets: g.loadingAssets,
        blackLoadingText,
        firstVisibleFrameText,
        loadingStatusText,
        unboundBindGroupCount: g.unboundBindGroupCount,
        skippedDrawCount: g.skippedDrawCount,
        validationErrorCount: g.validationErrorCount,
        totalDrawCount,
        hasRealPackDraws,
        hasFallbackDraws,
        updatedAtMs: g.updatedAtMs,
        statusText: gpuStatusText,
      },
      audio: {
        outputState: a.outputState,
        framesDrained: a.framesDrained,
        underruns: a.underruns,
        overruns: a.overruns,
        guestRingAttached: a.guestRingAttached,
        guestFramesDrained: a.guestFramesDrained,
        guestUnderruns: a.guestUnderruns,
        guestPeakLevel: a.guestPeakLevel,
        guestStatusText: guestAudioStatusText,
        updatedAtMs: a.updatedAtMs,
        statusText: audioStatusText,
      },
      pthread: {
        pthreadSpawnedCount: p.pthreadSpawnedCount,
        pthreadActiveCount: p.pthreadActiveCount,
        pthreadPeakCount: p.pthreadPeakCount,
        pthreadLoadQueueDepth: p.pthreadLoadQueueDepth,
        indirectDispatch: p.indirectDispatch,
        updatedAtMs: p.updatedAtMs,
        statusText: pthreadStatusText,
      },
      frameBudget: {
        fps30ComplianceRate: fb.fps30ComplianceRate,
        dominantBottleneck: fb.dominantBottleneck,
        valueText: frameBudgetValueText,
        statusText: frameBudgetStatusText,
        state: frameBudgetState,
        updatedAtMs: fb.updatedAtMs,
      },
      memory: {
        jsHeapUsedMb: m.jsHeapUsedMb,
        jsHeapTotalMb: m.jsHeapTotalMb,
        valueText: memoryValueText,
        statusText: memoryStatusText,
        state: memoryState,
        updatedAtMs: m.updatedAtMs,
      },
      io: {
        mailboxLanes: io.mailboxLanes,
        statusText: ioMailboxStatusText,
        state: ioMailboxState,
        updatedAtMs: io.updatedAtMs,
      },
      diagnosticsSummaryText,
    };
  }

  // --- Boot Mode, Split-Module Progress & Live WebGPU Validation Errors (W69) ---

  _isActiveSmokeSession() {
    const state = this._getSessionState();
    if (state === SESSION_STATE.RUNNING_SMOKE) {
      return true;
    }
    const mode = this._getSessionMode();
    if (mode === 'smoke') {
      return true;
    }
    return Boolean(this._smokeDemoRunner && this._smokeDemoRunner.running);
  }

  _isActiveWasmSession() {
    const state = this._getSessionState();
    if (state === SESSION_STATE.RUNNING_WASM) {
      return true;
    }
    return this._getSessionMode() === 'wasm';
  }

  /**
   * Returns the canonical active boot mode: `'real-wasm64'`, `'smoke'`, or `'idle'`.
   * Guarantees that an active smoke-demo session is NEVER reported as `'real-wasm64'`.
   *
   * @returns {'real-wasm64'|'smoke'|'idle'}
   */
  getBootMode() {
    if (this._isActiveSmokeSession()) {
      return BOOT_MODE.SMOKE;
    }
    if (this._explicitBootMode !== null) {
      return this._explicitBootMode;
    }
    if (this._isActiveWasmSession()) {
      return BOOT_MODE.REAL_WASM64;
    }
    try {
      const orchestrator = this._app && this._app.bootOrchestrator;
      if (orchestrator && typeof orchestrator.getState === 'function') {
        const orchState = orchestrator.getState();
        if (orchState && typeof orchState.bootMode === 'string') {
          return normalizeBootMode(orchState.bootMode);
        }
      }
    } catch {
      // Ignore optional orchestrator errors.
    }
    return BOOT_MODE.IDLE;
  }

  /**
   * Explicitly updates the tracked boot mode (`'real-wasm64'`, `'smoke'`, or `'idle'`)
   * and re-renders badge/status elements. If an active smoke-demo session is running
   * or `options.firstPresentSource` indicates smoke frames, the mode stays `'smoke'`
   * so smoke-demo frames are never mislabeled as real recompiled game frames.
   *
   * @param {string} mode
   * @param {{
   *   firstPresentSource?: string|null,
   *   isSmoke?: boolean,
   *   sideModuleCount?: number,
   *   loadedSideModules?: number,
   *   loadedModules?: Array<string>,
   *   updateHudStatus?: boolean,
   * }} [options]
   */
  setBootMode(mode, options = {}) {
    if (this._destroyed) {
      return this.getBootStatus();
    }
    const opts = isPlainObject(options) ? options : {};
    let normalized = normalizeBootMode(mode);

    const sourceMode =
      opts.firstPresentSource !== undefined && opts.firstPresentSource !== null
        ? normalizeBootMode(opts.firstPresentSource)
        : null;

    if (this._isActiveSmokeSession()) {
      // Active smoke sessions must never be mislabeled as real Wasm64.
      // Leave _explicitBootMode untouched when a live session/runner owns lifecycle
      // so stopping the smoke session transitions cleanly back to idle.
      if (!this._sessionController && !this._smokeDemoRunner) {
        this._explicitBootMode = BOOT_MODE.SMOKE;
      }
    } else if (opts.isSmoke === true || sourceMode === BOOT_MODE.SMOKE) {
      this._explicitBootMode = BOOT_MODE.SMOKE;
    } else {
      this._explicitBootMode = normalized;
    }

    if (
      opts.sideModuleCount !== undefined ||
      opts.loadedSideModules !== undefined ||
      Array.isArray(opts.loadedModules)
    ) {
      this._applySideModuleProgress({
        sideModuleCount: opts.sideModuleCount,
        loadedCount: opts.loadedSideModules,
        loadedModules: opts.loadedModules,
      });
    }

    return this._renderBootDiagnostics({
      updateHudStatus: opts.updateHudStatus !== false,
      force: opts.force === true,
    });
  }

  _applySideModuleProgress(progress) {
    if (!progress || typeof progress !== 'object') {
      return;
    }

    const rawList =
      Array.isArray(progress.sideModules)
        ? progress.sideModules
        : Array.isArray(progress.sideModuleUrls)
          ? progress.sideModuleUrls
          : Array.isArray(progress.sideModuleNames)
            ? progress.sideModuleNames
            : null;

    const rawLoadedList =
      Array.isArray(progress.loadedModules)
        ? progress.loadedModules
        : Array.isArray(progress.loadedNames)
          ? progress.loadedNames
          : null;

    const explicitTotal =
      progress.sideModuleCount !== undefined
        ? progress.sideModuleCount
        : progress.side_module_count !== undefined
          ? progress.side_module_count
          : progress.totalCount !== undefined
            ? progress.totalCount
            : progress.total !== undefined
              ? progress.total
              : undefined;

    if (explicitTotal !== undefined) {
      this._sideModuleCount = toNonNegativeInt(explicitTotal, 0);
      this._sideModuleTotalKnown = this._sideModuleCount > 0;
    } else if (rawList !== null) {
      this._sideModuleCount = rawList.length;
      this._sideModuleTotalKnown = this._sideModuleCount > 0;
    }

    if (rawLoadedList !== null) {
      const seen = new Set();
      const cleaned = [];
      for (const item of rawLoadedList) {
        const name = sanitizeSideModuleName(
          typeof item === 'string'
            ? item
            : item && typeof item === 'object'
              ? item.name || item.url || item.path
              : null
        );
        const key = name ? name.toLowerCase() : null;
        if (name && key && !seen.has(key)) {
          seen.add(key);
          if (cleaned.length < MAX_SIDE_MODULE_NAMES) {
            cleaned.push(name);
          }
        }
      }
      this._loadedSideModuleNames = cleaned;
      this._currentSideModule = cleaned.length > 0 ? cleaned[cleaned.length - 1] : null;
    }

    const explicitLoaded =
      progress.loadedCount !== undefined
        ? progress.loadedCount
        : progress.loadedSideModules !== undefined
          ? progress.loadedSideModules
          : progress.loadedSideModuleCount !== undefined
            ? progress.loadedSideModuleCount
            : progress.loaded !== undefined
              ? progress.loaded
              : undefined;

    if (explicitLoaded !== undefined) {
      this._loadedSideModuleCount = toNonNegativeInt(explicitLoaded, 0);
      if (this._loadedSideModuleCount === 0 && rawLoadedList === null) {
        this._loadedSideModuleNames = [];
        this._currentSideModule = null;
        this._runtimeInitialized = false;
      }
    } else if (rawLoadedList !== null) {
      this._loadedSideModuleCount = this._loadedSideModuleNames.length;
    } else if (
      this._runtimeInitialized &&
      this._sideModuleCount > 0 &&
      this._loadedSideModuleCount < this._sideModuleCount &&
      !this._isActiveSmokeSession() &&
      this.getBootMode() !== BOOT_MODE.SMOKE
    ) {
      this._loadedSideModuleCount = this._sideModuleCount;
    }

    if (typeof progress.currentModule === 'string') {
      const safeCurrent = sanitizeSideModuleName(progress.currentModule);
      if (safeCurrent) {
        this._currentSideModule = safeCurrent;
      }
    }

    if (this._sideModuleCount > 0 && this._loadedSideModuleCount > this._sideModuleCount) {
      this._loadedSideModuleCount = this._sideModuleCount;
    }
    if (this._sideModuleCount === 0 && this._loadedSideModuleCount === 0) {
      this._sideModuleTotalKnown = false;
      this._loadedSideModuleNames = [];
      this._currentSideModule = null;
    }
  }

  /**
   * Alias for `setSideModuleProgress()` matching app_bootstrap preflight hooks.
   */
  setSideModules(progress, options = {}) {
    return this.setSideModuleProgress(progress, options);
  }

  /**
   * Updates split-module (`guest_part_*.wasm`) progress counts/names and re-renders boot status.
   *
   * @param {{
   *   sideModuleCount?: number,
   *   side_module_count?: number,
   *   total?: number,
   *   totalCount?: number,
   *   loadedCount?: number,
   *   loadedSideModules?: number,
   *   loaded?: number,
   *   sideModules?: Array<string|object>,
   *   loadedModules?: Array<string|object>,
   *   currentModule?: string|null,
   * }} progress
   * @param {{ updateHudStatus?: boolean }} [options]
   */
  setSideModuleProgress(progress, options = {}) {
    if (this._destroyed) {
      return this.getSideModuleProgress();
    }
    const opts = isPlainObject(options) ? options : {};
    this._applySideModuleProgress(progress);
    this._renderBootDiagnostics({
      updateHudStatus: opts.updateHudStatus === true,
      force: opts.force === true,
    });
    return this.getSideModuleProgress();
  }

  /**
   * Records a single split side module (e.g. `guest_part_0.wasm`) as loaded,
   * sanitizing any host path or query string to its safe basename and deduplicating names.
   *
   * @param {string} moduleName
   * @param {{ sideModuleCount?: number, capToTotal?: boolean, updateHudStatus?: boolean }} [options]
   */
  recordSideModuleLoaded(moduleName, options = {}) {
    if (this._destroyed) {
      return this.getSideModuleProgress();
    }
    const opts = isPlainObject(options) ? options : {};
    if (opts.sideModuleCount !== undefined) {
      this._sideModuleCount = Math.max(
        this._sideModuleCount,
        toNonNegativeInt(opts.sideModuleCount, 0)
      );
      if (this._sideModuleCount > 0) {
        this._sideModuleTotalKnown = true;
      }
    }

    const safeName = sanitizeSideModuleName(moduleName);
    if (safeName) {
      const safeLower = safeName.toLowerCase();
      const alreadyLoaded = this._loadedSideModuleNames.some(
        (existing) => existing.toLowerCase() === safeLower
      );
      if (!alreadyLoaded) {
        if (this._loadedSideModuleNames.length < MAX_SIDE_MODULE_NAMES) {
          this._loadedSideModuleNames.push(safeName);
        }
        this._loadedSideModuleCount = Math.max(
          this._loadedSideModuleCount + 1,
          this._loadedSideModuleNames.length
        );
      }
      this._currentSideModule = safeName;
    } else if (moduleName !== undefined && moduleName !== null) {
      this._loadedSideModuleCount += 1;
    }

    if (opts.capToTotal === true && this._sideModuleCount > 0) {
      if (this._loadedSideModuleCount > this._sideModuleCount) {
        this._loadedSideModuleCount = this._sideModuleCount;
      }
    } else if (this._loadedSideModuleCount > this._sideModuleCount) {
      this._sideModuleCount = this._loadedSideModuleCount;
    }

    this._renderBootDiagnostics({
      updateHudStatus: opts.updateHudStatus === true,
      force: opts.force === true,
    });
    return this.getSideModuleProgress();
  }

  /**
   * Ingests a progress log label (e.g. `Split-module Wasm64 layout detected (5 side modules).`,
   * `Loading Wasm side module (guest_part_0.wasm)...`, or
   * `[wasm] playgta6web Wasm64 runtime initialized (ABI v1)`) and updates split-module
   * progress state and bound HUD display elements accordingly.
   *
   * @param {unknown} label
   * @param {{ updateHudStatus?: boolean }} [options]
   */
  ingestProgressLog(label, options = {}) {
    if (this._destroyed) {
      return null;
    }
    const opts = isPlainObject(options) ? options : {};
    const updateHudStatus = opts.updateHudStatus === true;

    const milestoneInfo = extractBootMilestoneFromText(label);
    let milestoneChanged = false;
    if (milestoneInfo && this._applyBootStage(milestoneInfo)) {
      milestoneChanged = true;
    }

    const parsed = parseSideModuleProgressLabel(label);
    if (!parsed) {
      if (milestoneChanged) {
        this._renderBootDiagnostics({ updateHudStatus });
        this._syncLiveHudReadouts({
          includeFrames: false,
          updateHudStatus: false,
        });
      }
      return null;
    }

    if (parsed.kind === 'runtime_starting') {
      if (this._runtimeInitialized) {
        this._runtimeInitialized = false;
        this._loadedSideModuleCount = 0;
        this._loadedSideModuleNames = [];
        this._currentSideModule = null;
        if (this._sideModuleCount > 0) {
          this._renderBootDiagnostics({ updateHudStatus });
        }
      }
      return {
        handled: true,
        kind: parsed.kind,
        progress: this.getSideModuleProgress(),
      };
    }

    if (parsed.kind === 'layout') {
      if (this._runtimeInitialized) {
        this._runtimeInitialized = false;
        this._loadedSideModuleCount = 0;
        this._loadedSideModuleNames = [];
        this._currentSideModule = null;
        this._sideModuleCount = parsed.sideModuleCount;
        this._sideModuleTotalKnown = parsed.sideModuleCount > 0;
      } else if (parsed.sideModuleCount > 0) {
        this._sideModuleTotalKnown = true;
        this._sideModuleCount = Math.max(this._sideModuleCount, parsed.sideModuleCount);
      } else if (this._loadedSideModuleCount === 0) {
        this._sideModuleTotalKnown = false;
        this._sideModuleCount = 0;
      }
      this._renderBootDiagnostics({ updateHudStatus });
      return {
        handled: true,
        kind: parsed.kind,
        sideModuleCount: parsed.sideModuleCount,
        progress: this.getSideModuleProgress(),
      };
    }

    if (parsed.kind === 'module') {
      const progress = this.recordSideModuleLoaded(parsed.moduleName, {
        updateHudStatus,
        capToTotal: this._sideModuleTotalKnown,
      });
      return {
        handled: true,
        kind: parsed.kind,
        moduleName: parsed.moduleName,
        progress,
      };
    }

    if (parsed.kind === 'runtime_initialized') {
      if (this._isActiveSmokeSession() || this.getBootMode() === BOOT_MODE.SMOKE) {
        return null;
      }
      this._runtimeInitialized = true;
      let rendered = false;
      if (this._sideModuleCount > 0 && this._loadedSideModuleCount < this._sideModuleCount) {
        this._loadedSideModuleCount = this._sideModuleCount;
        this._renderBootDiagnostics({ updateHudStatus });
        rendered = true;
      }
      if (milestoneChanged) {
        if (!rendered) {
          this._renderBootDiagnostics({ updateHudStatus });
        }
        this._syncLiveHudReadouts({
          includeFrames: false,
          updateHudStatus: false,
        });
      }
      return {
        handled: true,
        kind: parsed.kind,
        progress: this.getSideModuleProgress(),
      };
    }

    return null;
  }

  /**
   * Resets tracked split-module progress back to zero and re-renders boot status.
   *
   * @param {{ updateHudStatus?: boolean }} [options]
   */
  resetSideModuleProgress(options = {}) {
    if (this._destroyed) {
      return this.getSideModuleProgress();
    }
    const opts = isPlainObject(options) ? options : {};
    this._sideModuleCount = 0;
    this._sideModuleTotalKnown = false;
    this._loadedSideModuleCount = 0;
    this._loadedSideModuleNames = [];
    this._currentSideModule = null;
    this._runtimeInitialized = false;
    this._renderBootDiagnostics({
      updateHudStatus: opts.updateHudStatus === true,
      force: opts.force === true,
    });
    return this.getSideModuleProgress();
  }

  /**
   * Returns a defensive snapshot of tracked split-module (`guest_part_*.wasm`) progress.
   */
  getSideModuleProgress() {
    const sideModuleCount = this._sideModuleCount;
    const loadedCount =
      sideModuleCount > 0
        ? Math.min(this._loadedSideModuleCount, sideModuleCount)
        : this._loadedSideModuleCount;
    const active = sideModuleCount > 0 || loadedCount > 0;
    const remainingCount = Math.max(0, sideModuleCount - loadedCount);
    const complete = sideModuleCount > 0 && loadedCount >= sideModuleCount;
    const progressPct =
      sideModuleCount > 0
        ? Math.round((loadedCount / sideModuleCount) * 10000) / 100
        : 0;

    let statusText = '';
    if (active) {
      if (complete) {
        statusText = `Side modules: ${loadedCount}/${sideModuleCount} loaded`;
      } else {
        const pctLabel = Number.isInteger(progressPct)
          ? `${progressPct}%`
          : `${progressPct.toFixed(1)}%`;
        statusText = `Side modules: ${loadedCount}/${sideModuleCount} (${pctLabel})`;
      }
    }

    return {
      active,
      sideModuleCount,
      loadedCount,
      remainingCount,
      complete,
      progressPct,
      loadedModules: this._loadedSideModuleNames.slice(),
      currentModule: this._currentSideModule,
      statusText,
    };
  }

  _normalizeValidationErrorRecord(record, fallbackSeq) {
    if (record === null || record === undefined) {
      return null;
    }

    const isObj = typeof record === 'object';
    const type = normalizeValidationErrorType(record);
    const message = sanitizeValidationErrorMessage(record, this._maxValidationMessageLength);

    let seq = fallbackSeq;
    if (isObj && typeof record.seq === 'number' && Number.isFinite(record.seq) && record.seq > 0) {
      seq = Math.floor(record.seq);
    }

    let timestampMs;
    if (
      isObj &&
      typeof record.timestampMs === 'number' &&
      Number.isFinite(record.timestampMs) &&
      record.timestampMs >= 0
    ) {
      timestampMs = record.timestampMs;
    } else if (
      isObj &&
      typeof record.timestamp === 'number' &&
      Number.isFinite(record.timestamp) &&
      record.timestamp >= 0
    ) {
      timestampMs = record.timestamp;
    } else {
      const nowVal = this._now();
      timestampMs = typeof nowVal === 'number' && Number.isFinite(nowVal) ? nowVal : Date.now();
    }

    const normalized = {
      seq,
      type,
      message,
      timestampMs,
    };

    if (isObj && typeof record.source === 'string' && record.source.trim().length > 0) {
      normalized.source = sanitizeValidationErrorMessage(record.source, 64);
    }

    return normalized;
  }

  /**
   * Records a single live WebGPU validation/OOM/internal error into the bounded,
   * sanitized error ring, updates bound HUD display elements (without throwing when
   * optional DOM elements are absent), and returns the sanitized record.
   *
   * @param {object|Error|string|null|undefined} record
   * @param {{ silent?: boolean }} [options]
   * @returns {object|null}
   */
  recordValidationError(record, options = {}) {
    if (this._destroyed || record === null || record === undefined) {
      return null;
    }
    const opts = isPlainObject(options) ? options : {};
    this._validationErrorSeq += 1;
    const normalized = this._normalizeValidationErrorRecord(record, this._validationErrorSeq);
    if (!normalized) {
      this._validationErrorSeq -= 1;
      return null;
    }

    if (normalized.seq > this._validationErrorSeq) {
      this._validationErrorSeq = normalized.seq;
    }
    this._validationErrorCount += 1;
    this._validationErrors.push(normalized);
    if (this._validationErrors.length > this._maxValidationErrors) {
      this._validationErrors.shift();
    }

    if (opts.silent !== true) {
      this._hudLog(`[WebGPU ${normalized.type}] ${normalized.message}`, 'error');
    }

    this._renderBootDiagnostics({ updateHudStatus: false, force: opts.force === true });
    return { ...normalized };
  }

  /**
   * Replaces the bounded live WebGPU validation error list (e.g. from a worker snapshot
   * such as `WgpuSwapchain.getValidationErrors()`), sanitizing each entry, capping
   * history length to `maxValidationErrors`, and re-rendering optional DOM elements.
   *
   * @param {Array|Iterable|null|undefined} list
   * @param {{ totalCount?: number }} [options]
   */
  setValidationErrors(list, options = {}) {
    if (this._destroyed) {
      return this.getValidationSummary();
    }
    const opts = isPlainObject(options) ? options : {};

    let rawItems = [];
    if (Array.isArray(list)) {
      rawItems = list;
    } else if (list && typeof list !== 'string' && typeof list[Symbol.iterator] === 'function') {
      try {
        rawItems = Array.from(list);
      } catch {
        rawItems = [];
      }
    }

    const sanitizedAll = [];
    let maxSeq = 0;
    for (let i = 0; i < rawItems.length; i += 1) {
      const entry = this._normalizeValidationErrorRecord(rawItems[i], i + 1);
      if (entry) {
        if (entry.seq > maxSeq) {
          maxSeq = entry.seq;
        }
        sanitizedAll.push(entry);
      }
    }

    const bounded =
      sanitizedAll.length > this._maxValidationErrors
        ? sanitizedAll.slice(-this._maxValidationErrors)
        : sanitizedAll;

    const explicitTotal =
      typeof opts.totalCount === 'number' && Number.isFinite(opts.totalCount) && opts.totalCount >= 0
        ? Math.floor(opts.totalCount)
        : null;

    this._validationErrors = bounded;
    this._validationErrorCount =
      explicitTotal !== null
        ? Math.max(explicitTotal, sanitizedAll.length)
        : sanitizedAll.length;
    this._validationErrorSeq = Math.max(this._validationErrorSeq, maxSeq, this._validationErrorCount);

    this._renderBootDiagnostics({ updateHudStatus: false, force: opts.force === true });
    return this.getValidationSummary();
  }

  /**
   * Clears all recorded live WebGPU validation errors and resets the counter to zero.
   */
  clearValidationErrors(options = {}) {
    if (this._destroyed) {
      return this.getValidationSummary();
    }
    const opts = isPlainObject(options) ? options : {};
    this._validationErrors = [];
    this._validationErrorCount = 0;
    this._gpuTelemetry.validationErrorCount = 0;
    this._renderBootDiagnostics({ updateHudStatus: false, force: opts.force === true });
    return this.getValidationSummary();
  }

  /**
   * Returns a defensive summary of live WebGPU validation errors including total count,
   * bounded history size, latest sanitized error record/message, and formatted status text.
   */
  getValidationSummary() {
    const count = this._validationErrorCount;
    const historySize = this._validationErrors.length;
    const latest = historySize > 0 ? this._validationErrors[historySize - 1] : null;
    const latestMessage = latest ? latest.message : null;
    const hasErrors = count > 0;

    let statusText;
    if (!hasErrors) {
      statusText = '0 WebGPU validation errors';
    } else if (latestMessage) {
      statusText = `${count} WebGPU validation error${count === 1 ? '' : 's'}: ${latestMessage}`;
    } else {
      statusText = `${count} WebGPU validation error${count === 1 ? '' : 's'}`;
    }

    return {
      count,
      historySize,
      maxHistory: this._maxValidationErrors,
      hasErrors,
      latestError: latest ? { ...latest } : null,
      latestMessage,
      errors: this._validationErrors.map((entry) => ({ ...entry })),
      statusText,
    };
  }

  /**
   * Returns a complete snapshot of boot mode, badge/status labels, split-module progress,
   * and live WebGPU validation error state.
   */
  getBootStatus() {
    const bootMode = this.getBootMode();
    const meta = BOOT_MODE_META[bootMode] || BOOT_MODE_META[BOOT_MODE.IDLE];
    const sideModules = this.getSideModuleProgress();
    const validation = this.getValidationSummary();
    const telemetry = this.getTelemetry();

    const parts = [meta.baseStatusText];
    if (sideModules.active && sideModules.statusText) {
      parts.push(sideModules.statusText);
    }
    if (telemetry.gpu.gpuAdapterText) {
      parts.push(telemetry.gpu.gpuAdapterText);
    }
    if (telemetry.gpu.loadingStatusText) {
      parts.push(telemetry.gpu.loadingStatusText);
    }
    if (telemetry.gpu.statusText) {
      parts.push(telemetry.gpu.statusText);
    }
    if (telemetry.gpu.perfSummaryText) {
      parts.push(telemetry.gpu.perfSummaryText);
    }
    if (telemetry.pthread.statusText) {
      parts.push(telemetry.pthread.statusText);
    }
    if (telemetry.audio.statusText) {
      parts.push(telemetry.audio.statusText);
    }
    if (validation.hasErrors) {
      parts.push(
        `${validation.count} WebGPU validation error${validation.count === 1 ? '' : 's'}`
      );
    }
    if (this._scriptedInputPreset) {
      parts.push(`Scripted input: ${this._scriptedInputPreset}`);
    }

    const statusLevel = validation.hasErrors ? 'error' : meta.statusLevel;

    return {
      bootMode,
      badgeText: meta.badgeText,
      badgeState: meta.badgeState,
      statusText: parts.join(' — '),
      statusLevel,
      isRealWasm64: bootMode === BOOT_MODE.REAL_WASM64,
      isSmoke: bootMode === BOOT_MODE.SMOKE,
      isIdle: bootMode === BOOT_MODE.IDLE,
      visible: this._visible,
      scriptedInputPreset: this._scriptedInputPreset,
      milestone: this._bootStage,
      bootStage: this._bootStage || 'BOOTING',
      progressPct: this._bootProgressPct,
      readyForSplash: this._bootReadyForSplash,
      framesPresented: Math.max(telemetry.gpu.presentCount, this._liveFramesPresented),
      fps: this._liveFps,
      onePercentLowFps: telemetry.gpu.onePercentLowFps,
      p99FrameMs: telemetry.gpu.p99FrameMs,
      zeroCopyUploads: telemetry.gpu.zeroCopyUploads,
      perfSummaryText: telemetry.gpu.perfSummaryText,
      gpuAdapter: telemetry.gpu.gpuAdapter,
      blackLoadingMs: telemetry.gpu.blackLoadingMs,
      firstVisibleFrameMs: telemetry.gpu.firstVisibleFrameMs,
      loadingAssets: telemetry.gpu.loadingAssets,
      loadingStatusText: telemetry.gpu.loadingStatusText,
      guestRingAttached: telemetry.audio.guestRingAttached,
      guestFramesDrained: telemetry.audio.guestFramesDrained,
      guestUnderruns: telemetry.audio.guestUnderruns,
      guestPeakLevel: telemetry.audio.guestPeakLevel,
      guestAudioStatusText: telemetry.audio.guestStatusText,
      fps30ComplianceRate: telemetry.frameBudget.fps30ComplianceRate,
      dominantBottleneck: telemetry.frameBudget.dominantBottleneck,
      jsHeapUsedMb: telemetry.memory.jsHeapUsedMb,
      jsHeapTotalMb: telemetry.memory.jsHeapTotalMb,
      mailboxLanes: telemetry.io.mailboxLanes,
      diagnosticsSummaryText: telemetry.diagnosticsSummaryText,
      sideModules,
      validation,
      telemetry,
    };
  }

  /**
   * Binds optional display DOM elements for bootMode badge, boot status, split-module
   * progress, and live WebGPU validation errors. Never throws when elements are absent.
   *
   * @param {{
   *   bootModeBadge?: object|null,
   *   bootStatusEl?: object|null,
   *   sideModuleStatusEl?: object|null,
   *   validationCountEl?: object|null,
   *   validationLatestEl?: object|null,
   *   validationBannerEl?: object|null,
   * }} [elementMap]
   * @returns {() => void} Idempotent unbind function.
   */
  bindDisplayElements(elementMap = {}) {
    if (elementMap !== undefined && (elementMap === null || typeof elementMap !== 'object')) {
      throw new TypeError('bindDisplayElements expects an object mapping display keys to elements');
    }

    this._unmountDiagnosticsSection();
    this._displayElements = {};
    const map = elementMap || {};
    for (const key of SUPPORTED_DISPLAY_ELEMENTS) {
      const el = map[key];
      if (el !== undefined && el !== null && typeof el === 'object') {
        this._displayElements[key] = el;
      }
    }

    this._renderBootDiagnostics({
      updateHudStatus: false,
      force: true,
      recordRenderTimestamp: false,
    });

    let active = true;
    return () => {
      if (!active) return;
      active = false;
      this.unbindDisplayElements();
    };
  }

  /**
   * Detaches all bound display elements. Idempotent.
   */
  unbindDisplayElements() {
    this._unmountDiagnosticsSection();
    this._displayElements = {};
  }

  _unmountDiagnosticsSection() {
    const mounted = this._mountedDiagnosticsSection;
    this._mountedDiagnosticsSection = null;
    if (!mounted || !mounted.section) {
      return;
    }
    try {
      if (mounted.section.parentNode && typeof mounted.section.parentNode.removeChild === 'function') {
        mounted.section.parentNode.removeChild(mounted.section);
      }
    } catch {
      // Ignore teardown errors on lightweight or detached mock DOM nodes.
    }
  }

  _resolveDiagnosticsPanelTarget() {
    const els = this._displayElements;
    const explicit = els.diagnosticsPanelEl || els.diagnosticsContainerEl || null;
    if (explicit) {
      return { container: explicit, explicit: true };
    }
    const rootEl = this._getHudRootElement();
    if (rootEl && typeof rootEl.querySelector === 'function') {
      try {
        const collapsibleContent =
          rootEl.querySelector('#hud-diagnostics-disclosure .astryx-collapsible-content') ||
          rootEl.querySelector('#hud-diagnostics-disclosure');
        if (collapsibleContent && typeof collapsibleContent === 'object') {
          return { container: collapsibleContent, explicit: false };
        }
      } catch {
        // Ignore querySelector errors on synthetic mocks.
      }
    }
    return { container: null, explicit: false };
  }

  _ensureDiagnosticsSection(container) {
    if (!container || typeof container !== 'object' || typeof container.appendChild !== 'function') {
      return null;
    }
    if (this._mountedDiagnosticsSection && this._mountedDiagnosticsSection.container === container) {
      return this._mountedDiagnosticsSection;
    }

    const doc =
      (container.ownerDocument && typeof container.ownerDocument.createElement === 'function'
        ? container.ownerDocument
        : null) ||
      (typeof document !== 'undefined' && document && typeof document.createElement === 'function'
        ? document
        : null);
    if (!doc) {
      return null;
    }

    this._unmountDiagnosticsSection();

    try {
      const section = doc.createElement('section');
      section.className = 'diag-group';
      if (typeof section.setAttribute === 'function') {
        section.setAttribute('data-hud-section', 'w289-diagnostics');
        section.setAttribute('aria-label', '30 FPS budget, memory, and I/O');
      }

      const head = doc.createElement('div');
      head.className = 'diag-group-head';
      const headTitle = doc.createElement('span');
      headTitle.textContent = '30 FPS Budget & Memory';
      head.appendChild(headTitle);
      section.appendChild(head);

      const kvList = doc.createElement('div');
      kvList.className = 'kv-list';

      const makeRow = (labelText, valueKey) => {
        const row = doc.createElement('div');
        row.className = 'kv-row';
        const labelEl = doc.createElement('span');
        labelEl.className = 'kv-label';
        labelEl.textContent = labelText;
        const valueEl = doc.createElement('span');
        valueEl.className = 'kv-value';
        valueEl.textContent = '--';
        if (typeof valueEl.setAttribute === 'function') {
          valueEl.setAttribute('data-hud-metric', valueKey);
        }
        row.appendChild(labelEl);
        row.appendChild(valueEl);
        kvList.appendChild(row);
        return valueEl;
      };

      const budgetValueEl = makeRow('30 FPS Budget', 'frame-budget');
      const memoryValueEl = makeRow('JS Heap', 'js-heap');
      const lanesValueEl = makeRow('I/O Lanes', 'mailbox-lanes');

      section.appendChild(kvList);
      container.appendChild(section);

      this._mountedDiagnosticsSection = {
        container,
        section,
        budgetValueEl,
        memoryValueEl,
        lanesValueEl,
      };
      return this._mountedDiagnosticsSection;
    } catch {
      return null;
    }
  }

  /**
   * Configures the minimum interval (in ms) between non-forced DOM renders.
   * Pass `0` to disable throttling (default).
   *
   * @param {number} intervalMs
   * @returns {number} Effective `minRenderIntervalMs`.
   */
  setMinRenderIntervalMs(intervalMs) {
    this._minRenderIntervalMs =
      typeof intervalMs === 'number' && Number.isFinite(intervalMs) && intervalMs > 0
        ? intervalMs
        : 0;
    return this._minRenderIntervalMs;
  }

  /**
   * Returns dirty-check DOM write counters and render throttle telemetry.
   */
  getRenderStats() {
    return {
      domWrites: this._domWriteStats.domWrites,
      domWritesElided: this._domWriteStats.domWritesElided,
      renderCount: this._renderCount,
      throttledRenders: this._throttledRenders,
      minRenderIntervalMs: this._minRenderIntervalMs,
      lastRenderTimestampMs: this._lastRenderTimestampMs,
      dirty: this._renderDirty,
    };
  }

  /**
   * Resets dirty-check DOM write and throttle counters back to zero.
   */
  resetRenderStats() {
    this._domWriteStats.domWrites = 0;
    this._domWriteStats.domWritesElided = 0;
    this._renderCount = 0;
    this._throttledRenders = 0;
    return this.getRenderStats();
  }

  /**
   * Flushes any throttled/dirty DOM render immediately (or unconditionally when
   * `options.force === true`).
   *
   * @param {{ updateHudStatus?: boolean, force?: boolean }} [options]
   */
  flushRender(options = {}) {
    if (this._destroyed) {
      return this.getBootStatus();
    }
    const opts = isPlainObject(options) ? options : {};
    if (!this._renderDirty && opts.force !== true) {
      return this.getBootStatus();
    }
    return this._renderBootDiagnostics({
      updateHudStatus: opts.updateHudStatus === true,
      force: true,
    });
  }

  /**
   * Re-renders all bound boot mode, split-module, and validation-error display elements.
   *
   * @param {{ updateHudStatus?: boolean, force?: boolean }} [options]
   */
  renderBootStatus(options = {}) {
    if (this._destroyed) {
      return this.getBootStatus();
    }
    const opts = isPlainObject(options) ? options : {};
    return this._renderBootDiagnostics({
      updateHudStatus: opts.updateHudStatus === true,
      force: opts.force === true,
    });
  }

  _renderBootDiagnostics(options = {}) {
    const opts = isPlainObject(options) ? options : {};
    const force = opts.force === true;
    const recordRenderTimestamp = opts.recordRenderTimestamp !== false;
    const nowMs = this._safeNowMs();

    if (
      !force &&
      this._minRenderIntervalMs > 0 &&
      this._lastRenderTimestampMs !== null
    ) {
      const elapsedMs = nowMs - this._lastRenderTimestampMs;
      if (elapsedMs >= 0 && elapsedMs < this._minRenderIntervalMs) {
        this._renderDirty = true;
        this._throttledRenders += 1;
        return this.getBootStatus();
      }
    }

    const status = this.getBootStatus();
    if (recordRenderTimestamp) {
      this._lastRenderTimestampMs = nowMs;
    }
    this._renderDirty = false;
    this._renderCount += 1;

    const els = this._displayElements;
    const domStats = this._domWriteStats;

    if (els.bootModeBadge) {
      safeSetText(els.bootModeBadge, status.badgeText, domStats);
      safeSetDataset(els.bootModeBadge, 'bootMode', status.bootMode, domStats);
      safeSetDataset(els.bootModeBadge, 'state', status.badgeState, domStats);
      safeSetTitle(els.bootModeBadge, status.statusText, domStats);
      safeSetAttribute(els.bootModeBadge, 'aria-label', `Boot mode: ${status.badgeText}`, domStats);
    }

    if (els.bootStatusEl) {
      safeSetText(els.bootStatusEl, status.statusText, domStats);
      safeSetDataset(els.bootStatusEl, 'bootMode', status.bootMode, domStats);
      safeSetDataset(
        els.bootStatusEl,
        'state',
        status.statusLevel === 'error' ? 'err' : status.statusLevel,
        domStats
      );
      safeSetTitle(els.bootStatusEl, status.statusText, domStats);
    }

    if (els.sideModuleStatusEl) {
      const sm = status.sideModules;
      const pthreadText = status.telemetry && status.telemetry.pthread.statusText;
      const smText = sm.active
        ? pthreadText
          ? `${sm.statusText} · ${pthreadText}`
          : sm.statusText
        : 'No split modules';
      const smState = !sm.active ? '' : sm.complete ? 'ok' : 'info';
      safeSetText(els.sideModuleStatusEl, smText, domStats);
      safeSetDataset(els.sideModuleStatusEl, 'loaded', String(sm.loadedCount), domStats);
      safeSetDataset(els.sideModuleStatusEl, 'total', String(sm.sideModuleCount), domStats);
      safeSetDataset(els.sideModuleStatusEl, 'state', smState, domStats);
      safeSetTitle(els.sideModuleStatusEl, smText, domStats);
    }

    const validation = status.validation;
    if (els.validationCountEl) {
      safeSetText(els.validationCountEl, String(validation.count), domStats);
      safeSetDataset(els.validationCountEl, 'count', String(validation.count), domStats);
      safeSetDataset(els.validationCountEl, 'state', validation.hasErrors ? 'err' : 'ok', domStats);
    }

    if (els.validationLatestEl) {
      const latestText =
        validation.hasErrors && validation.latestMessage ? validation.latestMessage : 'None';
      safeSetText(els.validationLatestEl, latestText, domStats);
      safeSetDataset(els.validationLatestEl, 'state', validation.hasErrors ? 'err' : '', domStats);
      safeSetTitle(els.validationLatestEl, latestText, domStats);
    }

    if (els.validationBannerEl) {
      safeSetHidden(els.validationBannerEl, !validation.hasErrors, domStats);
      safeSetDataset(els.validationBannerEl, 'count', String(validation.count), domStats);
      safeSetDataset(
        els.validationBannerEl,
        'hasErrors',
        validation.hasErrors ? 'true' : 'false',
        domStats
      );
      safeSetDataset(els.validationBannerEl, 'state', validation.hasErrors ? 'err' : 'ok', domStats);
      const hasChildren =
        (Array.isArray(els.validationBannerEl.children) &&
          els.validationBannerEl.children.length > 0) ||
        (typeof els.validationBannerEl.childElementCount === 'number' &&
          els.validationBannerEl.childElementCount > 0);
      if (!hasChildren) {
        safeSetText(els.validationBannerEl, validation.statusText, domStats);
      }
    }

    const liveFrames = status.framesPresented;
    const liveFps = status.fps;
    const formattedFps = formatFpsDisplay(liveFps);
    const gpuTelemetry = status.telemetry && status.telemetry.gpu ? status.telemetry.gpu : null;
    const gpuStatus = gpuTelemetry ? gpuTelemetry.statusText : '';
    const onePercentLowText = gpuTelemetry ? gpuTelemetry.onePercentLowFpsText : '';
    const p99FrameMsText = gpuTelemetry ? gpuTelemetry.p99FrameMsText : '';
    const zeroCopyUploadsText = gpuTelemetry ? gpuTelemetry.zeroCopyUploadsText : '';
    const perfSummaryText = gpuTelemetry ? gpuTelemetry.perfSummaryText : '';

    if (els.framesEl) {
      safeSetText(els.framesEl, String(liveFrames), domStats);
      safeSetDataset(els.framesEl, 'frames', String(liveFrames), domStats);
      safeSetTitle(
        els.framesEl,
        gpuStatus || perfSummaryText || `${liveFrames} frame${liveFrames === 1 ? '' : 's'} presented`,
        domStats
      );
    }

    if (els.fpsEl) {
      safeSetText(els.fpsEl, formattedFps, domStats);
      safeSetDataset(els.fpsEl, 'fps', String(liveFps), domStats);
      safeSetDataset(els.fpsEl, 'state', liveFps >= 30 ? 'ok' : 'info', domStats);
      if (status.onePercentLowFps !== null) {
        safeSetDataset(els.fpsEl, 'onePercentLowFps', String(status.onePercentLowFps), domStats);
      } else if (hasDatasetKey(els.fpsEl, 'onePercentLowFps')) {
        safeSetDataset(els.fpsEl, 'onePercentLowFps', '', domStats);
      }
      if (status.p99FrameMs !== null) {
        safeSetDataset(els.fpsEl, 'p99FrameMs', String(status.p99FrameMs), domStats);
      } else if (hasDatasetKey(els.fpsEl, 'p99FrameMs')) {
        safeSetDataset(els.fpsEl, 'p99FrameMs', '', domStats);
      }
      if (perfSummaryText) {
        safeSetTitle(els.fpsEl, perfSummaryText, domStats);
      } else if (typeof els.fpsEl.title === 'string' && els.fpsEl.title.length > 0) {
        safeSetTitle(els.fpsEl, '', domStats);
      }
    }

    if (els.pillFpsEl) {
      safeSetText(els.pillFpsEl, `${formattedFps} fps`, domStats);
    }

    if (els.onePercentLowFpsEl) {
      safeSetText(els.onePercentLowFpsEl, onePercentLowText || '--', domStats);
      safeSetDataset(
        els.onePercentLowFpsEl,
        'onePercentLowFps',
        status.onePercentLowFps !== null ? String(status.onePercentLowFps) : '',
        domStats
      );
      safeSetTitle(els.onePercentLowFpsEl, onePercentLowText, domStats);
    }

    if (els.p99FrameMsEl) {
      safeSetText(els.p99FrameMsEl, p99FrameMsText || '--', domStats);
      safeSetDataset(
        els.p99FrameMsEl,
        'p99FrameMs',
        status.p99FrameMs !== null ? String(status.p99FrameMs) : '',
        domStats
      );
      safeSetTitle(els.p99FrameMsEl, p99FrameMsText, domStats);
    }

    const zeroCopyEl = els.zeroCopyUploadsEl || els.binkZeroCopyEl || null;
    if (zeroCopyEl) {
      safeSetText(zeroCopyEl, zeroCopyUploadsText || '--', domStats);
      safeSetDataset(
        zeroCopyEl,
        'zeroCopyUploads',
        status.zeroCopyUploads !== null ? String(status.zeroCopyUploads) : '',
        domStats
      );
      safeSetTitle(zeroCopyEl, zeroCopyUploadsText, domStats);
    }

    const gpuAdapterEl = els.gpuAdapterEl || els.gpuAdapterBadgeEl || null;
    if (gpuAdapterEl) {
      const adapterLabel = gpuTelemetry ? gpuTelemetry.gpuAdapter : null;
      const adapterTitle = gpuTelemetry ? gpuTelemetry.gpuAdapterText : '';
      safeSetText(gpuAdapterEl, adapterLabel || '--', domStats);
      safeSetDataset(gpuAdapterEl, 'gpuAdapter', adapterLabel || '', domStats);
      safeSetDataset(gpuAdapterEl, 'state', adapterLabel ? 'ok' : '', domStats);
      safeSetTitle(gpuAdapterEl, adapterTitle, domStats);
    }

    const loadingEl = els.loadingAssetsBadgeEl || els.loadingAssetsEl || els.blackLoadingEl || null;
    if (loadingEl) {
      const loadingAssets = gpuTelemetry ? gpuTelemetry.loadingAssets : false;
      const blackLoadingMs = gpuTelemetry ? gpuTelemetry.blackLoadingMs : null;
      const firstVisibleFrameMs = gpuTelemetry ? gpuTelemetry.firstVisibleFrameMs : null;
      const loadingStatusText = gpuTelemetry ? gpuTelemetry.loadingStatusText : '';
      const activeLoadingIndicator = loadingAssets || Boolean(loadingStatusText);
      safeSetHidden(loadingEl, !activeLoadingIndicator, domStats);
      safeSetAttribute(
        loadingEl,
        'aria-hidden',
        activeLoadingIndicator ? 'false' : 'true',
        domStats
      );
      safeSetText(loadingEl, loadingStatusText, domStats);
      safeSetDataset(loadingEl, 'loadingAssets', loadingAssets ? 'true' : 'false', domStats);
      safeSetDataset(
        loadingEl,
        'blackLoadingMs',
        blackLoadingMs !== null ? String(blackLoadingMs) : '',
        domStats
      );
      safeSetDataset(
        loadingEl,
        'firstVisibleFrameMs',
        firstVisibleFrameMs !== null ? String(firstVisibleFrameMs) : '',
        domStats
      );
      safeSetDataset(
        loadingEl,
        'state',
        loadingAssets ? 'info' : firstVisibleFrameMs !== null ? 'ok' : blackLoadingMs !== null ? 'info' : '',
        domStats
      );
      safeSetTitle(loadingEl, loadingStatusText, domStats);
    }

    if (els.firstVisibleFrameEl) {
      const firstVisibleFrameMs = gpuTelemetry ? gpuTelemetry.firstVisibleFrameMs : null;
      const firstVisibleText = gpuTelemetry ? gpuTelemetry.firstVisibleFrameText : '';
      safeSetText(els.firstVisibleFrameEl, firstVisibleText || '--', domStats);
      safeSetDataset(
        els.firstVisibleFrameEl,
        'firstVisibleFrameMs',
        firstVisibleFrameMs !== null ? String(firstVisibleFrameMs) : '',
        domStats
      );
      safeSetDataset(
        els.firstVisibleFrameEl,
        'state',
        firstVisibleFrameMs !== null ? 'ok' : '',
        domStats
      );
      safeSetTitle(els.firstVisibleFrameEl, firstVisibleText, domStats);
    }

    const audioTelemetry = status.telemetry && status.telemetry.audio ? status.telemetry.audio : null;
    if (els.audioStatusEl && audioTelemetry) {
      safeSetText(els.audioStatusEl, audioTelemetry.statusText || '--', domStats);
      safeSetDataset(els.audioStatusEl, 'outputState', audioTelemetry.outputState, domStats);
      safeSetDataset(
        els.audioStatusEl,
        'guestRingAttached',
        audioTelemetry.guestRingAttached ? 'true' : 'false',
        domStats
      );
      safeSetDataset(
        els.audioStatusEl,
        'guestFramesDrained',
        String(audioTelemetry.guestFramesDrained),
        domStats
      );
      safeSetDataset(
        els.audioStatusEl,
        'guestUnderruns',
        String(audioTelemetry.guestUnderruns),
        domStats
      );
      safeSetDataset(
        els.audioStatusEl,
        'guestPeakLevel',
        String(audioTelemetry.guestPeakLevel),
        domStats
      );
      safeSetDataset(
        els.audioStatusEl,
        'state',
        audioTelemetry.guestRingAttached || audioTelemetry.outputState === AUDIO_OUTPUT_STATE.ACTIVE
          ? 'ok'
          : 'info',
        domStats
      );
      safeSetTitle(els.audioStatusEl, audioTelemetry.statusText, domStats);
    }

    if (els.guestAudioStatusEl && audioTelemetry) {
      const guestText = audioTelemetry.guestStatusText || '';
      safeSetText(els.guestAudioStatusEl, guestText || '--', domStats);
      safeSetDataset(
        els.guestAudioStatusEl,
        'guestRingAttached',
        audioTelemetry.guestRingAttached ? 'true' : 'false',
        domStats
      );
      safeSetDataset(
        els.guestAudioStatusEl,
        'guestFramesDrained',
        String(audioTelemetry.guestFramesDrained),
        domStats
      );
      safeSetDataset(
        els.guestAudioStatusEl,
        'guestUnderruns',
        String(audioTelemetry.guestUnderruns),
        domStats
      );
      safeSetDataset(
        els.guestAudioStatusEl,
        'guestPeakLevel',
        String(audioTelemetry.guestPeakLevel),
        domStats
      );
      safeSetDataset(
        els.guestAudioStatusEl,
        'state',
        audioTelemetry.guestRingAttached
          ? audioTelemetry.guestUnderruns > 0
            ? 'warn'
            : 'ok'
          : '',
        domStats
      );
      safeSetTitle(els.guestAudioStatusEl, guestText, domStats);
    }

    if (els.bootStageEl) {
      safeSetText(els.bootStageEl, status.bootStage, domStats);
      safeSetDataset(els.bootStageEl, 'stage', status.bootStage, domStats);
      safeSetDataset(
        els.bootStageEl,
        'state',
        status.readyForSplash || status.progressPct >= 100 ? 'ok' : '',
        domStats
      );
    }

    if (els.bootProgressBarEl) {
      safeSetAttribute(
        els.bootProgressBarEl,
        'aria-valuenow',
        String(Math.round(status.progressPct)),
        domStats
      );
    }

    if (els.bootProgressFillEl) {
      safeSetStyleWidth(els.bootProgressFillEl, `${status.progressPct.toFixed(1)}%`, domStats);
      safeSetDataset(
        els.bootProgressFillEl,
        'state',
        status.readyForSplash || status.progressPct >= 100 ? 'ok' : '',
        domStats
      );
    }

    if (els.bootElapsedEl && this._bootElapsedMs !== null) {
      safeSetText(els.bootElapsedEl, formatElapsedSeconds(this._bootElapsedMs), domStats);
    }

    const frameBudgetTelemetry =
      status.telemetry && status.telemetry.frameBudget ? status.telemetry.frameBudget : null;
    const memoryTelemetry =
      status.telemetry && status.telemetry.memory ? status.telemetry.memory : null;
    const ioTelemetry = status.telemetry && status.telemetry.io ? status.telemetry.io : null;
    const diagSummaryText =
      status.telemetry && typeof status.telemetry.diagnosticsSummaryText === 'string'
        ? status.telemetry.diagnosticsSummaryText
        : '';

    const frameBudgetEl = els.frameBudgetEl || els.fps30BudgetEl || null;
    if (frameBudgetEl && frameBudgetTelemetry) {
      safeSetText(frameBudgetEl, frameBudgetTelemetry.statusText || '--', domStats);
      safeSetDataset(
        frameBudgetEl,
        'fps30ComplianceRate',
        frameBudgetTelemetry.fps30ComplianceRate !== null
          ? String(frameBudgetTelemetry.fps30ComplianceRate)
          : '',
        domStats
      );
      safeSetDataset(
        frameBudgetEl,
        'dominantBottleneck',
        frameBudgetTelemetry.dominantBottleneck || '',
        domStats
      );
      safeSetDataset(frameBudgetEl, 'state', frameBudgetTelemetry.state, domStats);
      safeSetTitle(frameBudgetEl, frameBudgetTelemetry.statusText, domStats);
    }

    const bottleneckEl = els.dominantBottleneckEl || els.bottleneckEl || null;
    if (bottleneckEl && frameBudgetTelemetry) {
      const bn = frameBudgetTelemetry.dominantBottleneck;
      safeSetText(bottleneckEl, bn || '--', domStats);
      safeSetDataset(bottleneckEl, 'dominantBottleneck', bn || '', domStats);
      safeSetDataset(
        bottleneckEl,
        'state',
        bn === null
          ? ''
          : bn === DOMINANT_BOTTLENECK.WITHIN_BUDGET
            ? 'ok'
            : 'warn',
        domStats
      );
      safeSetTitle(bottleneckEl, bn ? `Bottleneck: ${bn}` : '', domStats);
    }

    const memoryStatsEl = els.memoryStatsEl || els.jsHeapEl || null;
    if (memoryStatsEl && memoryTelemetry) {
      safeSetText(memoryStatsEl, memoryTelemetry.statusText || '--', domStats);
      safeSetDataset(
        memoryStatsEl,
        'jsHeapUsedMb',
        memoryTelemetry.jsHeapUsedMb !== null ? String(memoryTelemetry.jsHeapUsedMb) : '',
        domStats
      );
      safeSetDataset(
        memoryStatsEl,
        'jsHeapTotalMb',
        memoryTelemetry.jsHeapTotalMb !== null ? String(memoryTelemetry.jsHeapTotalMb) : '',
        domStats
      );
      safeSetDataset(memoryStatsEl, 'state', memoryTelemetry.state, domStats);
      safeSetTitle(memoryStatsEl, memoryTelemetry.statusText, domStats);
    }

    const ioMailboxLanesEl = els.ioMailboxLanesEl || els.mailboxLanesEl || null;
    if (ioMailboxLanesEl && ioTelemetry) {
      safeSetText(ioMailboxLanesEl, ioTelemetry.statusText || '--', domStats);
      safeSetDataset(
        ioMailboxLanesEl,
        'mailboxLanes',
        ioTelemetry.mailboxLanes !== null ? String(ioTelemetry.mailboxLanes) : '',
        domStats
      );
      safeSetDataset(ioMailboxLanesEl, 'state', ioTelemetry.state, domStats);
      safeSetTitle(ioMailboxLanesEl, ioTelemetry.statusText, domStats);
    }

    if (els.diagnosticsSummaryEl && frameBudgetTelemetry && memoryTelemetry && ioTelemetry) {
      safeSetText(els.diagnosticsSummaryEl, diagSummaryText || '--', domStats);
      safeSetDataset(
        els.diagnosticsSummaryEl,
        'fps30ComplianceRate',
        frameBudgetTelemetry.fps30ComplianceRate !== null
          ? String(frameBudgetTelemetry.fps30ComplianceRate)
          : '',
        domStats
      );
      safeSetDataset(
        els.diagnosticsSummaryEl,
        'dominantBottleneck',
        frameBudgetTelemetry.dominantBottleneck || '',
        domStats
      );
      safeSetDataset(
        els.diagnosticsSummaryEl,
        'jsHeapUsedMb',
        memoryTelemetry.jsHeapUsedMb !== null ? String(memoryTelemetry.jsHeapUsedMb) : '',
        domStats
      );
      safeSetDataset(
        els.diagnosticsSummaryEl,
        'mailboxLanes',
        ioTelemetry.mailboxLanes !== null ? String(ioTelemetry.mailboxLanes) : '',
        domStats
      );
      const summaryState =
        frameBudgetTelemetry.state === 'err'
          ? 'err'
          : frameBudgetTelemetry.state === 'warn' ||
              memoryTelemetry.state === 'warn' ||
              ioTelemetry.state === 'warn'
            ? 'warn'
            : diagSummaryText
              ? 'ok'
              : '';
      safeSetDataset(els.diagnosticsSummaryEl, 'state', summaryState, domStats);
      safeSetTitle(els.diagnosticsSummaryEl, diagSummaryText, domStats);
    }

    const diagTarget = this._resolveDiagnosticsPanelTarget();
    if (diagTarget.container && frameBudgetTelemetry && memoryTelemetry && ioTelemetry) {
      const hasW289Data =
        this._hasFrameBudgetTelemetry || this._hasMemoryTelemetry || this._hasIoTelemetry;
      if (diagTarget.explicit || hasW289Data || this._mountedDiagnosticsSection) {
        const mounted = this._ensureDiagnosticsSection(diagTarget.container);
        if (mounted) {
          safeSetText(
            mounted.budgetValueEl,
            frameBudgetTelemetry.valueText || '--',
            domStats
          );
          safeSetDataset(
            mounted.budgetValueEl,
            'fps30ComplianceRate',
            frameBudgetTelemetry.fps30ComplianceRate !== null
              ? String(frameBudgetTelemetry.fps30ComplianceRate)
              : '',
            domStats
          );
          safeSetDataset(
            mounted.budgetValueEl,
            'dominantBottleneck',
            frameBudgetTelemetry.dominantBottleneck || '',
            domStats
          );
          safeSetDataset(mounted.budgetValueEl, 'state', frameBudgetTelemetry.state, domStats);
          safeSetTitle(mounted.budgetValueEl, frameBudgetTelemetry.statusText, domStats);

          safeSetText(mounted.memoryValueEl, memoryTelemetry.valueText || '--', domStats);
          safeSetDataset(
            mounted.memoryValueEl,
            'jsHeapUsedMb',
            memoryTelemetry.jsHeapUsedMb !== null ? String(memoryTelemetry.jsHeapUsedMb) : '',
            domStats
          );
          safeSetDataset(
            mounted.memoryValueEl,
            'jsHeapTotalMb',
            memoryTelemetry.jsHeapTotalMb !== null ? String(memoryTelemetry.jsHeapTotalMb) : '',
            domStats
          );
          safeSetDataset(mounted.memoryValueEl, 'state', memoryTelemetry.state, domStats);
          safeSetTitle(mounted.memoryValueEl, memoryTelemetry.statusText, domStats);

          safeSetText(
            mounted.lanesValueEl,
            ioTelemetry.mailboxLanes !== null ? String(ioTelemetry.mailboxLanes) : '--',
            domStats
          );
          safeSetDataset(
            mounted.lanesValueEl,
            'mailboxLanes',
            ioTelemetry.mailboxLanes !== null ? String(ioTelemetry.mailboxLanes) : '',
            domStats
          );
          safeSetDataset(mounted.lanesValueEl, 'state', ioTelemetry.state, domStats);
          safeSetTitle(mounted.lanesValueEl, ioTelemetry.statusText, domStats);
        } else if (diagTarget.explicit) {
          safeSetText(diagTarget.container, diagSummaryText || '--', domStats);
          safeSetDataset(
            diagTarget.container,
            'fps30ComplianceRate',
            frameBudgetTelemetry.fps30ComplianceRate !== null
              ? String(frameBudgetTelemetry.fps30ComplianceRate)
              : '',
            domStats
          );
          safeSetDataset(
            diagTarget.container,
            'dominantBottleneck',
            frameBudgetTelemetry.dominantBottleneck || '',
            domStats
          );
          safeSetDataset(
            diagTarget.container,
            'jsHeapUsedMb',
            memoryTelemetry.jsHeapUsedMb !== null ? String(memoryTelemetry.jsHeapUsedMb) : '',
            domStats
          );
          safeSetDataset(
            diagTarget.container,
            'mailboxLanes',
            ioTelemetry.mailboxLanes !== null ? String(ioTelemetry.mailboxLanes) : '',
            domStats
          );
          safeSetTitle(diagTarget.container, diagSummaryText, domStats);
        }
      }
    }

    this._renderVisibility();
    this._renderScriptedInputBadge();

    if (
      opts.updateHudStatus === true &&
      this._hud &&
      typeof this._hud.setStatus === 'function'
    ) {
      this._setHudStatus(status.statusText, status.statusLevel);
    }

    return status;
  }

  // --- HUD Visibility Toggle & Scripted-Input Status Indicator (W143) ---

  _getHudRootElement() {
    const els = this._displayElements;
    return els.hudRootEl || els.rootEl || els.hudRoot || this._hudRootEl || null;
  }

  _getScriptedInputBadgeElement() {
    const els = this._displayElements;
    return (
      els.scriptedInputBadge ||
      els.scriptedInputBadgeEl ||
      els.scriptedInputStatusEl ||
      this._scriptedInputBadgeEl ||
      null
    );
  }

  _renderVisibility() {
    const rootEl = this._getHudRootElement();
    if (!rootEl) {
      return;
    }
    const domStats = this._domWriteStats;
    const hidden = !this._visible;
    safeSetHidden(rootEl, hidden, domStats);
    safeSetAttribute(rootEl, 'aria-hidden', hidden ? 'true' : 'false', domStats);
    safeSetDataset(rootEl, 'hudVisible', this._visible ? 'true' : 'false', domStats);
  }

  _renderScriptedInputBadge() {
    const badgeEl = this._getScriptedInputBadgeElement();
    if (!badgeEl) {
      return;
    }
    const domStats = this._domWriteStats;
    const preset = this._scriptedInputPreset;
    if (preset) {
      const label = `Scripted input: ${preset}`;
      safeSetHidden(badgeEl, false, domStats);
      safeSetAttribute(badgeEl, 'aria-hidden', 'false', domStats);
      safeSetText(badgeEl, label, domStats);
      safeSetDataset(badgeEl, 'preset', preset, domStats);
      safeSetDataset(badgeEl, 'active', 'true', domStats);
      safeSetDataset(badgeEl, 'state', 'info', domStats);
      safeSetTitle(badgeEl, label, domStats);
    } else {
      safeSetHidden(badgeEl, true, domStats);
      safeSetAttribute(badgeEl, 'aria-hidden', 'true', domStats);
      safeSetText(badgeEl, '', domStats);
      safeSetDataset(badgeEl, 'preset', '', domStats);
      safeSetDataset(badgeEl, 'active', 'false', domStats);
      safeSetDataset(badgeEl, 'state', '', domStats);
      safeSetTitle(badgeEl, '', domStats);
    }
  }

  /**
   * Returns whether the HUD root container is currently visible (default `true`).
   *
   * @returns {boolean}
   */
  isVisible() {
    return this._visible;
  }

  /**
   * Sets HUD visibility, updating the bound root HUD element's `hidden`,
   * `aria-hidden`, and `data-hud-visible` attributes cleanly. Safe no-op when
   * no DOM root element is bound or after `destroy()`.
   *
   * @param {boolean} visible
   * @returns {boolean} Current HUD visibility.
   */
  setVisible(visible) {
    if (this._destroyed) {
      return this._visible;
    }
    this._visible = Boolean(visible);
    this._renderVisibility();
    if (this._hud && typeof this._hud.setVisible === 'function') {
      try {
        this._hud.setVisible(this._visible);
      } catch {
        // Ignore optional HUD visibility callback errors.
      }
    }
    return this._visible;
  }

  /**
   * Toggles HUD visibility between visible and hidden.
   *
   * @returns {boolean} New HUD visibility.
   */
  toggleVisible() {
    if (this._destroyed) {
      return this._visible;
    }
    return this.setVisible(!this._visible);
  }

  /**
   * Alias for `toggleVisible()`.
   *
   * @returns {boolean} New HUD visibility.
   */
  toggleVisibility() {
    return this.toggleVisible();
  }

  /**
   * Attaches a `keydown` listener on `target` (defaulting to `window` or `document`
   * when available) that toggles HUD visibility on `key` (default `'F3'`) without
   * interfering with text inputs (`<input>`, `<textarea>`, `<select>`, or
   * `contenteditable` targets) or modifier chords. Replaces any previous toggle
   * key binding to prevent duplicate listeners.
   *
   * @param {EventTarget|null|undefined} [target]
   * @param {string} [key='F3']
   * @returns {() => void} Idempotent unbind function.
   */
  bindToggleKey(target, key = DEFAULT_TOGGLE_KEY) {
    if (this._destroyed) {
      return () => {};
    }

    let resolvedTarget = target;
    if (resolvedTarget === undefined) {
      if (typeof window !== 'undefined' && window) {
        resolvedTarget = window;
      } else if (typeof document !== 'undefined' && document) {
        resolvedTarget = document;
      } else {
        resolvedTarget = null;
      }
    }

    if (resolvedTarget === null) {
      this.unbindToggleKey();
      return () => {};
    }

    if (
      typeof resolvedTarget !== 'object' ||
      typeof resolvedTarget.addEventListener !== 'function' ||
      typeof resolvedTarget.removeEventListener !== 'function'
    ) {
      throw new TypeError(
        'bindToggleKey requires an EventTarget with addEventListener/removeEventListener'
      );
    }

    const normalizedKey =
      typeof key === 'string' && key.trim().length > 0 ? key.trim() : DEFAULT_TOGGLE_KEY;
    const expectedKeyLower = normalizedKey.toLowerCase();

    this.unbindToggleKey();

    const handler = (event) => {
      if (this._destroyed || !event || typeof event !== 'object') {
        return;
      }
      if (event.defaultPrevented || event.repeat) {
        return;
      }
      if (event.ctrlKey || event.altKey || event.metaKey) {
        return;
      }
      const eventKey = typeof event.key === 'string' ? event.key : '';
      const eventCode = typeof event.code === 'string' ? event.code : '';
      const matchesKey =
        eventKey.toLowerCase() === expectedKeyLower ||
        (expectedKeyLower === 'f3' && eventCode.toLowerCase() === 'f3');
      if (!matchesKey) {
        return;
      }
      if (isEditableEventTarget(event.target)) {
        return;
      }
      if (typeof event.preventDefault === 'function') {
        try {
          event.preventDefault();
        } catch {
          // Ignore preventDefault errors on synthetic events.
        }
      }
      this.toggleVisible();
    };

    resolvedTarget.addEventListener('keydown', handler);
    this._toggleKeyBinding = {
      target: resolvedTarget,
      key: normalizedKey,
      handler,
    };

    let active = true;
    return () => {
      if (!active) return;
      active = false;
      if (this._toggleKeyBinding && this._toggleKeyBinding.handler === handler) {
        this.unbindToggleKey();
      }
    };
  }

  /**
   * Removes any `keydown` shortcut listener previously attached via `bindToggleKey()`.
   * Idempotent.
   */
  unbindToggleKey() {
    const binding = this._toggleKeyBinding;
    this._toggleKeyBinding = null;
    if (!binding) {
      return;
    }
    try {
      binding.target.removeEventListener('keydown', binding.handler);
    } catch {
      // Ignore teardown errors from detached/mock targets.
    }
  }

  /**
   * Returns the currently active sanitized scripted-input preset name, or `null`
   * when no scripted-input playback is active.
   *
   * @returns {string|null}
   */
  getScriptedInputStatus() {
    return this._scriptedInputPreset;
  }

  /**
   * Updates the scripted-input status indicator. Strictly validates preset names
   * against `/^[a-z0-9][a-z0-9_-]{0,31}$/i` (rejecting paths, URLs, or arbitrary
   * text to `null`), stores `scriptedInputPreset`, and renders cleanly into any
   * bound badge element and HUD status line.
   *
   * @param {string|null|undefined} presetOrNull
   * @param {{ updateHudStatus?: boolean }} [options]
   * @returns {string|null} Sanitized active preset name or `null`.
   */
  setScriptedInputStatus(presetOrNull, options = {}) {
    if (this._destroyed) {
      return this._scriptedInputPreset;
    }
    const opts = isPlainObject(options) ? options : {};
    this._scriptedInputPreset = sanitizeScriptedInputPreset(presetOrNull);
    this._renderBootDiagnostics({
      updateHudStatus: opts.updateHudStatus !== false,
      force: opts.force === true,
    });
    return this._scriptedInputPreset;
  }

  /**
   * Switches the active input profile via `inputProfiles.setActiveProfile(profileId)`
   * and persists it via `inputProfiles.save()`.
   *
   * @param {string} profileId
   */
  async switchInputProfile(profileId) {
    const action = HUD_ACTIONS.SWITCH_INPUT_PROFILE;
    if (!this._inputProfiles || typeof this._inputProfiles.setActiveProfile !== 'function') {
      return this._finishAction(
        action,
        false,
        'Input profile manager unavailable; cannot switch profile.',
        { error: 'Input profile manager is not configured' },
        'warn'
      );
    }

    try {
      const profile = this._inputProfiles.setActiveProfile(profileId);
      let persisted = false;
      if (typeof this._inputProfiles.save === 'function') {
        persisted = Boolean(this._inputProfiles.save());
      }

      this._syncBoundControlState();

      const label = profile && profile.name ? `${profile.name} (${profile.id})` : String(profileId);
      const summary = `Switched input profile to ${label}${persisted ? ' (saved)' : ''}.`;

      return this._finishAction(
        action,
        true,
        summary,
        { profileId: profile ? profile.id : profileId, profile, persisted },
        'ok'
      );
    } catch (err) {
      const error = toErrorMessage(err, 'Failed to switch input profile');
      return this._finishAction(
        action,
        false,
        `Failed to switch input profile: ${error}`,
        { error, profileId }
      );
    }
  }

  /**
   * Toggles the WebGPU G5 smoke demo via `sessionController` (stopping if currently
   * running smoke demo, restarting with `{ preferSmoke: true }` if running Wasm64,
   * or starting with `{ preferSmoke: true }` if idle/stopped/error). Falls back to
   * `smokeDemoRunner.start()` / `stop()` when no `sessionController` is configured.
   */
  async toggleSmokeDemo() {
    const action = HUD_ACTIONS.TOGGLE_SMOKE_DEMO;
    const hasSession =
      this._sessionController &&
      typeof this._sessionController.start === 'function' &&
      typeof this._sessionController.stop === 'function';
    const hasSmokeRunner =
      this._smokeDemoRunner &&
      typeof this._smokeDemoRunner.start === 'function' &&
      typeof this._smokeDemoRunner.stop === 'function';

    if (!hasSession && !hasSmokeRunner) {
      return this._finishAction(
        action,
        false,
        'Session controller unavailable; cannot toggle smoke demo.',
        { error: 'Session controller is not configured' },
        'warn'
      );
    }

    try {
      if (hasSession) {
        const currentState = this._getSessionState();

        if (currentState === SESSION_STATE.RUNNING_SMOKE) {
          const sessionResult = await this._sessionController.stop();
          const state = this._getSessionState();
          this._explicitBootMode = null;
          this._renderBootDiagnostics({ updateHudStatus: false });
          const summary = 'Stopped smoke demo session.';
          return this._finishAction(
            action,
            true,
            summary,
            { running: false, state, sessionResult },
            'info'
          );
        }

        const sessionResult =
          currentState === SESSION_STATE.RUNNING_WASM &&
          typeof this._sessionController.restart === 'function'
            ? await this._sessionController.restart({ preferSmoke: true })
            : await this._sessionController.start({ preferSmoke: true });

        const state = this._getSessionState();
        if (sessionResult && sessionResult.ok === false) {
          this._renderBootDiagnostics({ updateHudStatus: false });
          const error = toErrorMessage(
            sessionResult.error ||
              (Array.isArray(sessionResult.reasons) ? sessionResult.reasons[0] : null),
            'Failed to start smoke demo'
          );
          return this._finishAction(
            action,
            false,
            `Failed to start smoke demo: ${error}`,
            { error, running: false, state, sessionResult }
          );
        }

        this._explicitBootMode = null;
        this._renderBootDiagnostics({ updateHudStatus: false });
        const summary = `Started smoke demo session (state: ${state}).`;
        return this._finishAction(
          action,
          true,
          summary,
          { running: state === SESSION_STATE.RUNNING_SMOKE, state, sessionResult },
          'ok'
        );
      }

      // Direct SmokeDemoRunner fallback when no RuntimeSessionController is supplied.
      if (this._smokeDemoRunner.running) {
        this._smokeDemoRunner.stop();
        this._explicitBootMode = null;
        this._renderBootDiagnostics({ updateHudStatus: false });
        return this._finishAction(
          action,
          true,
          'Stopped smoke demo runner.',
          { running: false, state: this._getSessionState() },
          'info'
        );
      }

      this._smokeDemoRunner.start();
      this._explicitBootMode = null;
      this._renderBootDiagnostics({ updateHudStatus: false });
      return this._finishAction(
        action,
        true,
        'Started smoke demo runner.',
        { running: Boolean(this._smokeDemoRunner.running), state: this._getSessionState() },
        'ok'
      );
    } catch (err) {
      const error = toErrorMessage(err, 'Failed to toggle smoke demo');
      return this._finishAction(action, false, `Failed to toggle smoke demo: ${error}`, {
        error,
        state: this._getSessionState(),
      });
    }
  }

  /**
   * Cleanly restarts the active runtime session via `sessionController.restart({ preferSmoke })`.
   *
   * @param {{ preferSmoke?: boolean }} [options]
   */
  async restartSession(options = {}) {
    const action = HUD_ACTIONS.RESTART_SESSION;
    const preferSmoke = Boolean(options && typeof options === 'object' && options.preferSmoke);

    if (!this._sessionController || typeof this._sessionController.restart !== 'function') {
      return this._finishAction(
        action,
        false,
        'Session controller unavailable; cannot restart session.',
        { error: 'Session controller is not configured' },
        'warn'
      );
    }

    try {
      this._captureAppFramesBaseline();
      this._loadedSideModuleCount = 0;
      this._loadedSideModuleNames = [];
      this._currentSideModule = null;
      this._runtimeInitialized = false;
      this._gpuTelemetry = createInitialGpuTelemetry();
      this._audioTelemetry = createInitialAudioTelemetry();
      this._pthreadTelemetry = createInitialPthreadTelemetry();
      this._frameBudgetTelemetry = createInitialFrameBudgetTelemetry();
      this._memoryTelemetry = createInitialMemoryTelemetry();
      this._ioTelemetry = createInitialIoTelemetry();
      this._hasGpuTelemetry = false;
      this._hasAudioTelemetry = false;
      this._hasGuestAudioTelemetry = false;
      this._hasPthreadTelemetry = false;
      this._hasFrameBudgetTelemetry = false;
      this._hasMemoryTelemetry = false;
      this._hasIoTelemetry = false;
      this._liveFramesPresented = 0;
      this._liveFps = 0;
      this._fpsSamples = [];
      this._bootStage = null;
      this._bootStageIndex = -1;
      this._bootProgressPct = 0;
      this._bootReadyForSplash = false;
      this._bootElapsedMs = 0;
      this._lastPushedBootStage = null;
      this._lastPushedProgressPct = -1;
      this._lastExternalHudStatus = null;
      this._lastConciseHudStatus = null;

      this._renderBootDiagnostics({ updateHudStatus: false, force: true });
      this._pushHudMetrics({
        framesPresented: 0,
        totalFrames: 0,
        fps: 0,
        bootSummary: {
          stage: 'BOOTING',
          progressPct: 0,
          readyForSplash: false,
          elapsedMs: 0,
        },
      });

      const sessionResult = await this._sessionController.restart({ preferSmoke });
      const state = this._getSessionState();
      const mode = this._getSessionMode();

      if (sessionResult && sessionResult.ok === false) {
        this._explicitBootMode = null;
        this._renderBootDiagnostics({ updateHudStatus: false, force: true });
        const error = toErrorMessage(
          sessionResult.error ||
            (Array.isArray(sessionResult.reasons) ? sessionResult.reasons[0] : null),
          'Session restart failed'
        );
        return this._finishAction(
          action,
          false,
          `Failed to restart session: ${error}`,
          { error, state, mode, preferSmoke, sessionResult }
        );
      }

      this._explicitBootMode = null;
      this._renderBootDiagnostics({ updateHudStatus: false, force: true });
      this._pushHudMetrics({
        framesPresented: this._liveFramesPresented,
        totalFrames: this._liveFramesPresented,
        fps: this._liveFps,
        ...(this._bootStage
          ? {}
          : {
              bootSummary: {
                stage: 'BOOTING',
                progressPct: 0,
                readyForSplash: false,
                elapsedMs: 0,
              },
            }),
      });

      const summary = `Restarted runtime session (mode: ${mode}, state: ${state}).`;
      return this._finishAction(
        action,
        true,
        summary,
        { state, mode, preferSmoke, sessionResult },
        'ok'
      );
    } catch (err) {
      const error = toErrorMessage(err, 'Failed to restart session');
      return this._finishAction(action, false, `Failed to restart session: ${error}`, {
        error,
        state: this._getSessionState(),
        mode: this._getSessionMode(),
        preferSmoke,
      });
    }
  }

  // --- DOM Control Wiring ---

  /**
   * Binds DOM click/change handlers for any provided HUD control elements.
   * Replaces any previously bound controls to avoid duplicate listeners.
   *
   * @param {{
   *   clearCacheBtn?: EventTarget|null,
   *   exportSavesBtn?: EventTarget|null,
   *   recordTraceBtn?: EventTarget|null,
   *   exportTraceBtn?: EventTarget|null,
   *   exportCrashBtn?: EventTarget|null,
   *   exportMissingShadersBtn?: EventTarget|null,
   *   restartBtn?: EventTarget|null,
   *   smokeDemoBtn?: EventTarget|null,
   *   profileSelect?: EventTarget|null,
   * }} [buttonMap]
   * @returns {() => void} Idempotent unbind function.
   */
  bindButtons(buttonMap = {}) {
    if (buttonMap !== undefined && (buttonMap === null || typeof buttonMap !== 'object')) {
      throw new TypeError('bindButtons expects an object mapping control names to DOM elements');
    }

    this.unbindButtons();

    const map = buttonMap || {};
    const handlersByKey = {
      clearCacheBtn: () => {
        void this.clearOpfsCache();
      },
      exportSavesBtn: () => {
        void this.exportSaves();
      },
      recordTraceBtn: () => {
        void this.toggleGpuRecording();
      },
      exportTraceBtn: () => {
        void this.exportGpuTrace();
      },
      exportCrashBtn: () => {
        void this.exportCrashReport();
      },
      exportMissingShadersBtn: () => {
        void this.exportMissingShaders();
      },
      restartBtn: () => {
        void this.restartSession();
      },
      smokeDemoBtn: () => {
        void this.toggleSmokeDemo();
      },
      profileSelect: (event) => {
        const el = map.profileSelect;
        const selected =
          event && event.target && typeof event.target.value === 'string'
            ? event.target.value
            : el && typeof el.value === 'string'
              ? el.value
              : '';
        void this.switchInputProfile(selected);
      },
    };

    for (const [key, eventType] of SUPPORTED_BUTTON_BINDINGS) {
      const element = map[key];
      if (element === undefined || element === null) {
        continue;
      }
      validateEventTarget(element, key);
      const handler = handlersByKey[key];
      element.addEventListener(eventType, handler);
      this._boundListeners.push({ key, element, eventType, handler });
      this._boundElements[key] = element;
    }

    let hasDisplayOverride = false;
    for (const key of SUPPORTED_DISPLAY_ELEMENTS) {
      const element = map[key];
      if (element !== undefined && element !== null && typeof element === 'object') {
        this._displayElements[key] = element;
        hasDisplayOverride = true;
      }
    }

    this._syncBoundControlState();
    if (hasDisplayOverride) {
      this._renderBootDiagnostics({
        updateHudStatus: false,
        force: true,
        recordRenderTimestamp: false,
      });
    }

    let active = true;
    return () => {
      if (!active) return;
      active = false;
      this.unbindButtons();
    };
  }

  /**
   * Removes all DOM event listeners previously attached via `bindButtons()`. Idempotent.
   */
  unbindButtons() {
    for (const entry of this._boundListeners) {
      try {
        entry.element.removeEventListener(entry.eventType, entry.handler);
      } catch {
        // Ignore teardown errors from detached/mock elements.
      }
    }
    this._boundListeners = [];
    this._boundElements = {};
  }

  /**
   * Tears down all bound DOM listeners and marks this controller destroyed. Idempotent.
   */
  destroy() {
    this._detachHudLogObserver();
    this._detachHudMetricsObserver();
    this._detachHudStatusObserver();
    this.unbindToggleKey();
    this.unbindButtons();
    this.unbindDisplayElements();
    this._hudRootEl = null;
    this._scriptedInputBadgeEl = null;
    this._renderDirty = false;
    this._destroyed = true;
  }

  // --- State Introspection ---

  /**
   * Returns current HUD control states across recording, session lifecycle,
   * active input profile, and last executed action.
   */
  getState() {
    return {
      isRecording: this._isRecording(),
      sessionState: this._getSessionState(),
      sessionMode: this._getSessionMode(),
      activeProfileId: this._getActiveProfileId(),
      lastAction: this._lastAction ? { ...this._lastAction } : null,
      boundControlCount: this._boundListeners.length,
      destroyed: this._destroyed,
    };
  }

  /**
   * Returns a complete defensive HUD snapshot including control state, visibility,
   * scripted-input status, boot diagnostics, render stats, and honest GPU/Audio/Pthread telemetry.
   */
  getSnapshot() {
    const bootStatus = this.getBootStatus();
    const telemetry = this.getTelemetry();
    return {
      ...this.getState(),
      visible: this._visible,
      scriptedInputPreset: this._scriptedInputPreset,
      gpuAdapter: telemetry.gpu.gpuAdapter,
      blackLoadingMs: telemetry.gpu.blackLoadingMs,
      firstVisibleFrameMs: telemetry.gpu.firstVisibleFrameMs,
      loadingAssets: telemetry.gpu.loadingAssets,
      guestRingAttached: telemetry.audio.guestRingAttached,
      guestFramesDrained: telemetry.audio.guestFramesDrained,
      guestUnderruns: telemetry.audio.guestUnderruns,
      guestPeakLevel: telemetry.audio.guestPeakLevel,
      fps30ComplianceRate: telemetry.frameBudget.fps30ComplianceRate,
      dominantBottleneck: telemetry.frameBudget.dominantBottleneck,
      jsHeapUsedMb: telemetry.memory.jsHeapUsedMb,
      jsHeapTotalMb: telemetry.memory.jsHeapTotalMb,
      mailboxLanes: telemetry.io.mailboxLanes,
      diagnosticsSummaryText: telemetry.diagnosticsSummaryText,
      frameBudget: telemetry.frameBudget,
      memory: telemetry.memory,
      io: telemetry.io,
      domWrites: this._domWriteStats.domWrites,
      domWritesElided: this._domWriteStats.domWritesElided,
      minRenderIntervalMs: this._minRenderIntervalMs,
      throttledRenders: this._throttledRenders,
      renderStats: this.getRenderStats(),
      bootStatus,
      telemetry,
    };
  }

  /**
   * Returns a defensive diagnostics snapshot surfacing 30 FPS frame-budget status,
   * dominant bottleneck, JS heap memory MB, HTTPFS I/O mailbox lanes, and full
   * HUD telemetry & boot status.
   */
  getDiagnosticsSnapshot() {
    const snapshot = this.getSnapshot();
    const { frameBudget, memory, io } = snapshot.telemetry;
    return {
      ...snapshot,
      fps30ComplianceRate: frameBudget.fps30ComplianceRate,
      dominantBottleneck: frameBudget.dominantBottleneck,
      frameBudgetStatusText: frameBudget.statusText,
      jsHeapUsedMb: memory.jsHeapUsedMb,
      jsHeapTotalMb: memory.jsHeapTotalMb,
      memoryStatusText: memory.statusText,
      mailboxLanes: io.mailboxLanes,
      ioMailboxStatusText: io.statusText,
      diagnosticsSummaryText: snapshot.telemetry.diagnosticsSummaryText,
      frameBudget: { ...frameBudget },
      memory: { ...memory },
      io: { ...io },
    };
  }
}

export default HudController;
