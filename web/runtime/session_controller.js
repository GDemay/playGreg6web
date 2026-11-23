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

export const DEFAULT_STATUS_URL = '/api/status';

const VALID_CAPABILITY_KEYS = Object.freeze([
  'crossOriginIsolated',
  'sharedArrayBuffer',
  'webgpu',
  'wasm64',
  'opfs',
  'gamepad',
  'audioWorklet',
]);

function detectWasm64() {
  if (typeof WebAssembly === 'undefined' || typeof WebAssembly.Memory !== 'function') {
    return false;
  }
  try {
    // eslint-disable-next-line no-new
    new WebAssembly.Memory({ initial: 1n, maximum: 2n, index: 'i64' });
    return true;
  } catch {
    return false;
  }
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

  return {
    crossOriginIsolated: g.crossOriginIsolated === true,
    sharedArrayBuffer: typeof SharedArrayBuffer === 'function',
    webgpu: Boolean(nav && 'gpu' in nav && nav.gpu),
    wasm64: detectWasm64(),
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
  };
}

function normalizeServerStatus(data, httpStatus = 200) {
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
    hasWasmBinary: Boolean(data.hasWasmBinary),
    shaderPackCount: Number.isFinite(data.shaderPackCount) ? data.shaderPackCount : 0,
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
   * @param {Function|null} [options.startWasmRuntime=null]
   * @param {Function|null} [options.startSmokeDemo=null]
   * @param {Function|null} [options.capabilityProbe=null]
   * @param {() => number} [options.now]
   * @param {number} [options.hudIntervalMs]
   * @param {EventTarget} [options.audioGestureTarget]
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
      startWasmRuntime = null,
      startSmokeDemo = null,
      capabilityProbe = null,
      now = () => Date.now(),
      hudIntervalMs,
      audioGestureTarget,
    } = options;

    if (fetchImpl !== null && fetchImpl !== undefined && typeof fetchImpl !== 'function') {
      throw new TypeError('fetchImpl must be a function or null');
    }
    if (typeof statusUrl !== 'string' || statusUrl.length === 0) {
      throw new TypeError('statusUrl must be a non-empty string');
    }
    if (typeof now !== 'function') {
      throw new TypeError('now must be a function returning milliseconds');
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

    this._fetchImpl = fetchImpl || null;
    this._statusUrl = statusUrl;
    this._hud = hud || null;
    this._inputBridge = inputBridge || null;
    this._audioEngine = audioEngine || null;
    this._storageManager = storageManager || null;
    this._perfMonitor = perfMonitor || null;
    this._crashReporter = crashReporter || null;
    this._startWasmRuntime = startWasmRuntime || null;
    this._startSmokeDemo = startSmokeDemo || null;
    this._capabilityProbe = capabilityProbe || null;
    this._now = now;
    this._hudIntervalMs = hudIntervalMs;
    this._audioGestureTarget = audioGestureTarget;

    this._state = SESSION_STATE.IDLE;
    this._mode = SESSION_MODE.NONE;
    this._capabilities = null;
    this._serverStatus = null;
    this._storageSummary = null;
    this._fallbackReason = null;
    this._diagnosticReasons = [];
    this._lastError = null;
    this._startedAtMs = null;

    this._activeHandle = null;
    this._perfDetach = null;
    this._audioGestureCleanup = null;
    this._inputStarted = false;

    this._sessionToken = 0;
    this._startPromise = null;
    this._stateListeners = new Set();
  }

  // --- Observability & Diagnostics ---

  getState() {
    return this._state;
  }

  getMode() {
    return this._mode;
  }

  /**
   * Registers a state transition listener.
   *
   * @param {(event: { state: string, previousState: string, mode: string, timestampMs: number }) => void} listener
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

  getDiagnostics() {
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

    const isRunning =
      this._state === SESSION_STATE.RUNNING_WASM || this._state === SESSION_STATE.RUNNING_SMOKE;
    const uptimeMs =
      isRunning && this._startedAtMs !== null ? Math.max(0, this._now() - this._startedAtMs) : 0;

    return {
      state: this._state,
      mode: this._mode,
      capabilities: this._capabilities ? { ...this._capabilities } : null,
      serverStatus: this._serverStatus ? { ...this._serverStatus } : null,
      storageSummary: this._storageSummary,
      perfSnapshot,
      uptimeMs,
      fallbackReason: this._fallbackReason,
      diagnosticReasons: [...this._diagnosticReasons],
      lastError: this._lastError ? this._lastError.message : null,
    };
  }

  // --- Internal Helpers ---

  _transition(nextState, detail = null) {
    const previousState = this._state;
    if (previousState === nextState) {
      return;
    }
    this._state = nextState;

    const event = {
      state: nextState,
      previousState,
      mode: this._mode,
      timestampMs: this._now(),
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
      return;
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
    } catch (err) {
      const normalized = toError(err, 'Failed to stop active runtime handle');
      this._reportError(normalized, { phase: 'stop', source: 'runtimeHandle' });
      this._hudLog(`Runtime stop warning: ${normalized.message}`, 'warn');
    }
  }

  _teardownSubsystems() {
    if (this._perfDetach) {
      const detach = this._perfDetach;
      this._perfDetach = null;
      try {
        detach();
      } catch (err) {
        this._reportError(toError(err, 'Failed to detach PerfMonitor from HUD'), {
          phase: 'stop',
          source: 'perfMonitor',
        });
      }
    } else if (this._perfMonitor && typeof this._perfMonitor.detachHud === 'function') {
      try {
        this._perfMonitor.detachHud();
      } catch (err) {
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
        this._reportError(toError(err, 'Failed to clean up audio user-gesture listeners'), {
          phase: 'stop',
          source: 'audioEngine',
        });
      }
    }
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

    return {
      capabilities: { ...capabilities },
      serverStatus: { ...serverStatus },
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
    const preferSmoke = Boolean(options && options.preferSmoke);

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
        fallbackReason: this._fallbackReason,
        capabilities: this._capabilities ? { ...this._capabilities } : null,
        serverStatus: this._serverStatus ? { ...this._serverStatus } : null,
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
    this._startedAtMs = null;

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

      return {
        ok: false,
        state: this._state,
        mode: this._mode,
        error,
        reasons: [...diagnosticReasons],
        capabilities: { ...capabilities },
        serverStatus: { ...serverStatus },
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

    this._fallbackReason = fallbackReason;

    this._transition(SESSION_STATE.STARTING, {
      targetMode,
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

      this._hudStatus('No runtime launcher', 'error');
      this._hudLog(error.message, 'error');
      this._reportError(error, {
        phase: 'starting',
        code: error.code,
        preferSmoke,
        serverStatus: { ...serverStatus },
      });

      this._transition(SESSION_STATE.ERROR, { error, reasons: [error.message] });
      return {
        ok: false,
        state: this._state,
        mode: this._mode,
        error,
        reasons: [error.message],
        capabilities: { ...capabilities },
        serverStatus: { ...serverStatus },
        diagnostics: this.getDiagnostics(),
      };
    }

    if (targetMode === SESSION_MODE.WASM) {
      this._hudStatus('Starting Wasm64 runtime...', 'info');
      this._hudLog('Server reports Wasm64 runtime binary ready; launching Wasm64 runtime.', 'info');
    } else {
      this._hudStatus('Starting smoke demo...', 'info');
      this._hudLog(fallbackReason, preferSmoke ? 'info' : 'warn');
    }

    const launchContext = {
      mode: targetMode,
      capabilities: { ...capabilities },
      serverStatus: { ...serverStatus },
      storageSummary: this._storageSummary,
      hud: this._hud,
      inputBridge: this._inputBridge,
      audioEngine: this._audioEngine,
      storageManager: this._storageManager,
      perfMonitor: this._perfMonitor,
      crashReporter: this._crashReporter,
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
          diagnostics: this.getDiagnostics(),
        };
      }
      const normalized = toError(err, `Failed to start ${targetMode} session`);
      this._lastError = normalized;
      this._diagnosticReasons = [normalized.message];
      this._mode = SESSION_MODE.NONE;

      this._hudStatus('Runtime start error', 'error');
      this._hudLog(`Failed to start ${targetMode} runtime: ${normalized.message}`, 'error');
      this._reportError(normalized, {
        phase: 'starting',
        targetMode,
        fallbackReason,
      });

      this._transition(SESSION_STATE.ERROR, {
        error: normalized,
        targetMode,
        reasons: [normalized.message],
      });

      return {
        ok: false,
        state: this._state,
        mode: this._mode,
        error: normalized,
        reasons: [normalized.message],
        capabilities: { ...capabilities },
        serverStatus: { ...serverStatus },
        diagnostics: this.getDiagnostics(),
      };
    }

    // If stop() was called while awaiting launcher, dispose the late handle cleanly.
    if (token !== this._sessionToken) {
      await this._stopHandle(handle);
      return {
        ok: false,
        aborted: true,
        state: this._state,
        mode: this._mode,
        diagnostics: this.getDiagnostics(),
      };
    }

    this._activeHandle = handle ?? null;

    try {
      this._wireSubsystems();
    } catch (err) {
      const normalized = toError(err, 'Failed to wire runtime subsystems');
      this._teardownSubsystems();
      const handleToStop = this._activeHandle;
      this._activeHandle = null;
      await this._stopHandle(handleToStop);

      this._lastError = normalized;
      this._diagnosticReasons = [normalized.message];
      this._mode = SESSION_MODE.NONE;

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

      return {
        ok: false,
        state: this._state,
        mode: this._mode,
        error: normalized,
        reasons: [normalized.message],
        capabilities: { ...capabilities },
        serverStatus: { ...serverStatus },
        diagnostics: this.getDiagnostics(),
      };
    }

    this._mode = targetMode;
    this._startedAtMs = this._now();

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
      fallbackReason,
    });

    return {
      ok: true,
      state: this._state,
      mode: this._mode,
      fallbackReason: this._fallbackReason,
      capabilities: { ...capabilities },
      serverStatus: { ...serverStatus },
      handle: this._activeHandle,
      diagnostics: this.getDiagnostics(),
    };
  }

  /**
   * Idempotently stops the active runtime or smoke demo handle, detaches PerfMonitor,
   * stops InputBridge, removes AudioEngine user-gesture listeners, and transitions to `stopped`.
   */
  async stop() {
    // Invalidate any in-flight start() immediately so a subsequent start()/restart() never
    // coalesces onto an aborted start promise.
    this._sessionToken += 1;
    this._startPromise = null;

    const handleToStop = this._activeHandle;
    this._activeHandle = null;

    this._teardownSubsystems();
    await this._stopHandle(handleToStop);

    this._mode = SESSION_MODE.NONE;
    this._startedAtMs = null;
    this._fallbackReason = null;

    if (this._state === SESSION_STATE.STOPPED) {
      return {
        ok: true,
        alreadyStopped: true,
        state: this._state,
        mode: this._mode,
        diagnostics: this.getDiagnostics(),
      };
    }

    this._hudStatus('Stopped', 'info');
    this._transition(SESSION_STATE.STOPPED, { reason: 'Session stopped' });

    return {
      ok: true,
      alreadyStopped: false,
      state: this._state,
      mode: this._mode,
      diagnostics: this.getDiagnostics(),
    };
  }

  /**
   * Cleanly stops any active session and restarts with the provided options
   * without leaking listeners, timers, or runtime handles.
   *
   * @param {{ preferSmoke?: boolean }} [options]
   */
  async restart(options = {}) {
    await this.stop();
    return this.start(options);
  }
}

export default RuntimeSessionController;
