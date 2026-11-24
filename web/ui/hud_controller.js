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

export const DEFAULT_MAX_VALIDATION_ERRORS = 32;
export const MAX_VALIDATION_ERROR_MESSAGE_LENGTH = 256;
const MAX_SIDE_MODULE_NAMES = 64;

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
  'bootModeBadge',
  'bootStatusEl',
  'sideModuleStatusEl',
  'validationCountEl',
  'validationLatestEl',
  'validationBannerEl',
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
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n) || n < 0) {
    return fallback;
  }
  return Math.floor(n);
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

function safeSetText(el, text) {
  if (!el || typeof el !== 'object') return;
  try {
    el.textContent = String(text ?? '');
  } catch {
    // Ignore read-only mock properties.
  }
}

function safeSetTitle(el, title) {
  if (!el || typeof el !== 'object') return;
  try {
    el.title = String(title ?? '');
  } catch {
    // Ignore read-only mock properties.
  }
}

function safeSetHidden(el, hidden) {
  if (!el || typeof el !== 'object') return;
  try {
    el.hidden = Boolean(hidden);
  } catch {
    // Ignore read-only mock properties.
  }
}

function safeSetDataset(el, key, value) {
  if (!el || typeof el !== 'object') return;
  try {
    if (el.dataset && typeof el.dataset === 'object') {
      if (value === '' || value === null || value === undefined) {
        delete el.dataset[key];
      } else {
        el.dataset[key] = String(value);
      }
    }
  } catch {
    // Ignore DOM dataset update errors on lightweight mocks.
  }
}

function safeSetAttribute(el, name, value) {
  if (!el || typeof el !== 'object' || typeof el.setAttribute !== 'function') return;
  try {
    el.setAttribute(name, String(value));
  } catch {
    // Ignore DOM attribute update errors on lightweight mocks.
  }
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
      maxValidationErrors = DEFAULT_MAX_VALIDATION_ERRORS,
      maxValidationMessageLength = MAX_VALIDATION_ERROR_MESSAGE_LENGTH,
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

    this._lastAction = null;
    this._boundListeners = [];
    this._boundElements = {};
    this._displayElements = {};
    this._exportingMissingShaders = false;
    this._destroyed = false;

    this._explicitBootMode =
      typeof bootMode === 'string' && bootMode.trim().length > 0
        ? normalizeBootMode(bootMode)
        : null;

    this._sideModuleCount = 0;
    this._loadedSideModuleCount = 0;
    this._loadedSideModuleNames = [];
    this._currentSideModule = null;

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

    const initialDisplayEls = isPlainObject(displayElements)
      ? displayElements
      : isPlainObject(elements)
        ? elements
        : null;
    if (initialDisplayEls) {
      this.bindDisplayElements(initialDisplayEls);
    }
  }

  get isExportingMissingShaders() {
    return this._exportingMissingShaders;
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

  _hudLog(message, level = 'info') {
    if (this._hud && typeof this._hud.appendLog === 'function') {
      try {
        this._hud.appendLog(String(message ?? ''), level);
      } catch {
        // Never allow HUD logging errors to break action execution.
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
    const recordBtn = this._boundElements.recordTraceBtn;
    if (recordBtn) {
      const recording = this._isRecording();
      try {
        if (recordBtn.dataset && typeof recordBtn.dataset === 'object') {
          recordBtn.dataset.recording = recording ? 'true' : 'false';
        }
        if (typeof recordBtn.setAttribute === 'function') {
          recordBtn.setAttribute('aria-pressed', recording ? 'true' : 'false');
        }
      } catch {
        // Ignore DOM attribute update errors on lightweight mocks.
      }
    }

    const profileSelect = this._boundElements.profileSelect;
    if (profileSelect) {
      const activeId = this._getActiveProfileId();
      if (activeId !== null) {
        try {
          profileSelect.value = activeId;
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
  updateGpuBootTelemetry(payload) {
    if (this._destroyed) {
      return null;
    }

    if (payload && typeof payload === 'object') {
      this._ingestBootTelemetryPayload(payload);
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
  }

  _ingestBootTelemetryPayload(payload) {
    let changed = false;

    const summary =
      payload.bootSummary && typeof payload.bootSummary === 'object' ? payload.bootSummary : null;
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
      this.setSideModuleProgress(progressInput, { updateHudStatus: false });
      changed = false; // setSideModuleProgress already rendered.
    }

    if (Array.isArray(payload.validationErrors)) {
      this.setValidationErrors(payload.validationErrors, {
        totalCount: payload.validationErrorCount,
      });
      changed = false;
    } else if (payload.validationError !== undefined && payload.validationError !== null) {
      this.recordValidationError(payload.validationError);
      changed = false;
    }

    if (changed) {
      this._renderBootDiagnostics({ updateHudStatus: false });
    }
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
    } else if (rawList !== null) {
      this._sideModuleCount = rawList.length;
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
        if (name && !seen.has(name)) {
          seen.add(name);
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
    } else if (rawLoadedList !== null) {
      this._loadedSideModuleCount = this._loadedSideModuleNames.length;
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
      this._loadedSideModuleNames = [];
      this._currentSideModule = null;
    }
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
    });
    return this.getSideModuleProgress();
  }

  /**
   * Records a single split side module (e.g. `guest_part_0.wasm`) as loaded,
   * sanitizing any host path or query string to its safe basename and deduplicating names.
   *
   * @param {string} moduleName
   * @param {{ sideModuleCount?: number, updateHudStatus?: boolean }} [options]
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
    }

    const safeName = sanitizeSideModuleName(moduleName);
    if (safeName) {
      if (!this._loadedSideModuleNames.includes(safeName)) {
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

    if (this._loadedSideModuleCount > this._sideModuleCount) {
      this._sideModuleCount = this._loadedSideModuleCount;
    }

    this._renderBootDiagnostics({
      updateHudStatus: opts.updateHudStatus === true,
    });
    return this.getSideModuleProgress();
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
    this._loadedSideModuleCount = 0;
    this._loadedSideModuleNames = [];
    this._currentSideModule = null;
    this._renderBootDiagnostics({
      updateHudStatus: opts.updateHudStatus === true,
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

    this._renderBootDiagnostics({ updateHudStatus: false });
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

    this._renderBootDiagnostics({ updateHudStatus: false });
    return this.getValidationSummary();
  }

  /**
   * Clears all recorded live WebGPU validation errors and resets the counter to zero.
   */
  clearValidationErrors() {
    if (this._destroyed) {
      return this.getValidationSummary();
    }
    this._validationErrors = [];
    this._validationErrorCount = 0;
    this._renderBootDiagnostics({ updateHudStatus: false });
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

    const parts = [meta.baseStatusText];
    if (sideModules.active && sideModules.statusText) {
      parts.push(sideModules.statusText);
    }
    if (validation.hasErrors) {
      parts.push(
        `${validation.count} WebGPU validation error${validation.count === 1 ? '' : 's'}`
      );
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
      sideModules,
      validation,
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

    this._displayElements = {};
    const map = elementMap || {};
    for (const key of SUPPORTED_DISPLAY_ELEMENTS) {
      const el = map[key];
      if (el !== undefined && el !== null && typeof el === 'object') {
        this._displayElements[key] = el;
      }
    }

    this._renderBootDiagnostics({ updateHudStatus: false });

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
    this._displayElements = {};
  }

  /**
   * Re-renders all bound boot mode, split-module, and validation-error display elements.
   *
   * @param {{ updateHudStatus?: boolean }} [options]
   */
  renderBootStatus(options = {}) {
    const opts = isPlainObject(options) ? options : {};
    return this._renderBootDiagnostics({
      updateHudStatus: opts.updateHudStatus === true,
    });
  }

  _renderBootDiagnostics(options = {}) {
    const status = this.getBootStatus();
    const els = this._displayElements;

    if (els.bootModeBadge) {
      safeSetText(els.bootModeBadge, status.badgeText);
      safeSetDataset(els.bootModeBadge, 'bootMode', status.bootMode);
      safeSetDataset(els.bootModeBadge, 'state', status.badgeState);
      safeSetTitle(els.bootModeBadge, status.statusText);
      safeSetAttribute(els.bootModeBadge, 'aria-label', `Boot mode: ${status.badgeText}`);
    }

    if (els.bootStatusEl) {
      safeSetText(els.bootStatusEl, status.statusText);
      safeSetDataset(els.bootStatusEl, 'bootMode', status.bootMode);
      safeSetDataset(
        els.bootStatusEl,
        'state',
        status.statusLevel === 'error' ? 'err' : status.statusLevel
      );
      safeSetTitle(els.bootStatusEl, status.statusText);
    }

    if (els.sideModuleStatusEl) {
      const sm = status.sideModules;
      const smText = sm.active ? sm.statusText : 'No split modules';
      const smState = !sm.active ? '' : sm.complete ? 'ok' : 'info';
      safeSetText(els.sideModuleStatusEl, smText);
      safeSetDataset(els.sideModuleStatusEl, 'loaded', String(sm.loadedCount));
      safeSetDataset(els.sideModuleStatusEl, 'total', String(sm.sideModuleCount));
      safeSetDataset(els.sideModuleStatusEl, 'state', smState);
      safeSetTitle(els.sideModuleStatusEl, smText);
    }

    const validation = status.validation;
    if (els.validationCountEl) {
      safeSetText(els.validationCountEl, String(validation.count));
      safeSetDataset(els.validationCountEl, 'count', String(validation.count));
      safeSetDataset(els.validationCountEl, 'state', validation.hasErrors ? 'err' : 'ok');
    }

    if (els.validationLatestEl) {
      const latestText =
        validation.hasErrors && validation.latestMessage ? validation.latestMessage : 'None';
      safeSetText(els.validationLatestEl, latestText);
      safeSetDataset(els.validationLatestEl, 'state', validation.hasErrors ? 'err' : '');
      safeSetTitle(els.validationLatestEl, latestText);
    }

    if (els.validationBannerEl) {
      safeSetHidden(els.validationBannerEl, !validation.hasErrors);
      safeSetDataset(els.validationBannerEl, 'count', String(validation.count));
      safeSetDataset(els.validationBannerEl, 'hasErrors', validation.hasErrors ? 'true' : 'false');
      safeSetDataset(els.validationBannerEl, 'state', validation.hasErrors ? 'err' : 'ok');
      const hasChildren =
        (Array.isArray(els.validationBannerEl.children) &&
          els.validationBannerEl.children.length > 0) ||
        (typeof els.validationBannerEl.childElementCount === 'number' &&
          els.validationBannerEl.childElementCount > 0);
      if (!hasChildren) {
        safeSetText(els.validationBannerEl, validation.statusText);
      }
    }

    if (
      options &&
      options.updateHudStatus === true &&
      this._hud &&
      typeof this._hud.setStatus === 'function'
    ) {
      try {
        this._hud.setStatus(status.statusText, status.statusLevel);
      } catch {
        // Never allow HUD status update errors to break controller methods.
      }
    }

    return status;
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
      const sessionResult = await this._sessionController.restart({ preferSmoke });
      const state = this._getSessionState();
      const mode = this._getSessionMode();

      if (sessionResult && sessionResult.ok === false) {
        this._explicitBootMode = null;
        this._renderBootDiagnostics({ updateHudStatus: false });
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
      this._renderBootDiagnostics({ updateHudStatus: false });

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
      this._renderBootDiagnostics({ updateHudStatus: false });
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
    this.unbindButtons();
    this.unbindDisplayElements();
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
}

export default HudController;
