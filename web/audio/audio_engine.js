// Main-thread WebAudio engine for PlayGta6Web.
// Manages AudioContext lifecycle, AudioWorklet registration & wiring, SharedArrayBuffer
// ring allocation/attachment, user-gesture unlock, volume/mute controls, and live telemetry.

export const PROCESSOR_NAME = 'playgta6-audio-processor';
export const DEFAULT_SAMPLE_RATE = 48000;
export const DEFAULT_CHANNELS = 2;
export const DEFAULT_CAPACITY_FRAMES = 4096;
export const DEFAULT_WORKLET_URL = new URL(
  '../workers/audio_worklet_processor.js',
  import.meta.url
).href;

// Atomic Int32 header layout at byte offset 0..32 of the SharedArrayBuffer.
// Matches web/workers/audio_worker.js and web/workers/audio_worklet_processor.js byte-for-byte.
export const HDR_WRITE_CURSOR = 0;
export const HDR_READ_CURSOR = 1;
export const HDR_CAPACITY_FRAMES = 2;
export const HDR_CHANNELS = 3;
export const HDR_UNDERRUNS = 4;
export const HDR_UNDERRUN_FRAMES = 5;
export const HDR_OVERFLOWS = 6;
export const HDR_DROPPED_FRAMES = 7;

export const RING_HEADER_INT32_COUNT = 8;
export const RING_HEADER_BYTES = RING_HEADER_INT32_COUNT * Int32Array.BYTES_PER_ELEMENT;

export const DEFAULT_MAX_RESUME_RETRIES = 5;
export const MAX_ERROR_MESSAGE_LENGTH = 256;

const KNOWN_OUTPUT_STATES = new Set([
  'idle',
  'initializing',
  'suspended',
  'running',
  'interrupted',
  'closed',
  'unsupported',
  'error',
]);

const CONTROL_CHARS_RE = /[\u0000-\u001f\u007f]+/g;
const FILE_URL_RE = /file:\/\/[^\s'")]+/gi;
const URL_USERINFO_RE = /\b([a-z][a-z0-9+.-]*:\/\/)[^/\s@]+@/gi;
const URL_QUERY_RE = /(\b(?:https?|wss?):\/\/[^\s?#]+)\?[^\s#'")]*/gi;
const AUTH_SCHEME_RE = /\b(Bearer|Basic)\s+[^\s'"]+/gi;
const SECRET_KV_RE =
  /\b(token|api[_-]?key|password|passwd|secret|authorization)(\s*[:=]\s*)[^\s,;'"&)]+/gi;
const WIN_DRIVE_PATH_RE = /\b[A-Za-z]:\\[^\s'")]+/g;
const WIN_UNC_PATH_RE = /(^|[\s('"])\\\\[^\s'")]+/g;
const POSIX_HOST_PATH_RE =
  /(^|[\s('"])\/(?:Users|home|var|etc|tmp|private|root|opt)\/[^\s'")]+/g;

function sanitizeFiniteNonNegInt(value, fallback = 0) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    return fallback;
  }
  const floored = Math.floor(value);
  return floored === 0 ? 0 : floored;
}

function sanitizePositiveFiniteNumber(value, fallback) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    return fallback;
  }
  return value;
}

function sanitizeOutputState(state) {
  if (typeof state !== 'string') {
    return 'unknown';
  }
  const normalized = state.trim().toLowerCase();
  if (KNOWN_OUTPUT_STATES.has(normalized)) {
    return normalized;
  }
  return 'unknown';
}

/**
 * Sanitizes an error or string for safe telemetry snapshots:
 * strips control characters, redacts host paths, file:// URLs, URL credentials/queries,
 * and auth tokens, and bounds length to MAX_ERROR_MESSAGE_LENGTH.
 *
 * @param {unknown} err
 * @returns {string|null}
 */
export function sanitizeResumeError(err) {
  if (err === null || err === undefined) {
    return null;
  }

  let raw = '';
  if (typeof err === 'string') {
    raw = err;
  } else if (typeof err === 'object') {
    let name = '';
    let message = '';
    try {
      if (typeof err.name === 'string') {
        name = err.name.trim();
      }
    } catch {
      // Ignore hostile property getters.
    }
    try {
      if (typeof err.message === 'string') {
        message = err.message.trim();
      }
    } catch {
      // Ignore hostile property getters.
    }

    if (name && message && name !== 'Error' && !message.startsWith(name)) {
      raw = `${name}: ${message}`;
    } else if (message) {
      raw = message;
    } else if (name) {
      raw = name;
    } else {
      try {
        raw = String(err);
      } catch {
        raw = '';
      }
    }
  } else {
    try {
      raw = String(err);
    } catch {
      raw = '';
    }
  }

  let out = raw
    .replace(CONTROL_CHARS_RE, ' ')
    .replace(FILE_URL_RE, '[REDACTED_FILE_URL]')
    .replace(URL_USERINFO_RE, '$1[REDACTED]@')
    .replace(URL_QUERY_RE, '$1?[REDACTED]')
    .replace(AUTH_SCHEME_RE, '$1 [REDACTED]')
    .replace(SECRET_KV_RE, '$1$2[REDACTED]')
    .replace(WIN_DRIVE_PATH_RE, '[REDACTED_PATH]')
    .replace(WIN_UNC_PATH_RE, '$1[REDACTED_PATH]')
    .replace(POSIX_HOST_PATH_RE, '$1[REDACTED_PATH]')
    .replace(/\s+/g, ' ')
    .trim();

  if (out.length === 0) {
    return 'AudioContext resume failed';
  }
  if (out.length > MAX_ERROR_MESSAGE_LENGTH) {
    return `${out.slice(0, MAX_ERROR_MESSAGE_LENGTH - 3)}...`;
  }
  return out;
}

/**
 * Clamps a volume/gain input to a finite number in [0, 1].
 * Non-number and non-finite inputs (NaN, ±Infinity) resolve to the sanitized fallback.
 *
 * @param {unknown} value
 * @param {number} [fallback=1.0]
 * @returns {number}
 */
export function clampVolume(value, fallback = 1.0) {
  const safeFallback =
    typeof fallback === 'number' && Number.isFinite(fallback)
      ? fallback <= 0.0
        ? 0.0
        : fallback >= 1.0
          ? 1.0
          : fallback
      : 1.0;

  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return safeFallback;
  }
  if (value <= 0.0) return 0.0;
  if (value >= 1.0) return 1.0;
  return value;
}

export const clampGain = clampVolume;

export function audioRingByteLength(
  capacityFrames = DEFAULT_CAPACITY_FRAMES,
  channels = DEFAULT_CHANNELS
) {
  if (!Number.isInteger(capacityFrames) || capacityFrames <= 0) {
    throw new RangeError(`capacityFrames must be a positive integer, got ${capacityFrames}`);
  }
  if (!Number.isInteger(channels) || channels <= 0) {
    throw new RangeError(`channels must be a positive integer, got ${channels}`);
  }
  return RING_HEADER_BYTES + capacityFrames * channels * Float32Array.BYTES_PER_ELEMENT;
}

/**
 * Allocates and atomically initializes a SharedArrayBuffer ring matching
 * AudioRingBuffer in web/workers/audio_worker.js.
 */
export function createAudioRingSab(
  capacityFrames = DEFAULT_CAPACITY_FRAMES,
  channels = DEFAULT_CHANNELS
) {
  if (typeof SharedArrayBuffer === 'undefined') {
    throw new Error(
      'SharedArrayBuffer is unavailable (requires Cross-Origin Isolation with COOP/COEP headers)'
    );
  }
  const sab = new SharedArrayBuffer(audioRingByteLength(capacityFrames, channels));
  const ctrl = new Int32Array(sab, 0, RING_HEADER_INT32_COUNT);
  Atomics.store(ctrl, HDR_WRITE_CURSOR, 0);
  Atomics.store(ctrl, HDR_READ_CURSOR, 0);
  Atomics.store(ctrl, HDR_CAPACITY_FRAMES, capacityFrames);
  Atomics.store(ctrl, HDR_CHANNELS, channels);
  Atomics.store(ctrl, HDR_UNDERRUNS, 0);
  Atomics.store(ctrl, HDR_UNDERRUN_FRAMES, 0);
  Atomics.store(ctrl, HDR_OVERFLOWS, 0);
  Atomics.store(ctrl, HDR_DROPPED_FRAMES, 0);
  return sab;
}

function resolveDefaultAudioContextCtor() {
  if (typeof globalThis === 'undefined') {
    return null;
  }
  return globalThis.AudioContext || globalThis.webkitAudioContext || null;
}

function resolveDefaultAudioWorkletNodeCtor() {
  if (typeof globalThis === 'undefined') {
    return null;
  }
  return globalThis.AudioWorkletNode || null;
}

export class AudioEngine {
  /**
   * @param {object} [options]
   * @param {number} [options.sampleRate=48000]
   * @param {number} [options.channels=2]
   * @param {number} [options.capacityFrames=4096]
   * @param {SharedArrayBuffer} [options.sab]
   * @param {{ sab: SharedArrayBuffer }} [options.ringBuffer]
   * @param {string} [options.workletUrl]
   * @param {string} [options.processorName='playgta6-audio-processor']
   * @param {number} [options.volume=1.0]
   * @param {number} [options.gain]
   * @param {boolean} [options.muted=false]
   * @param {number} [options.smoothingFrames]
   * @param {string|number} [options.latencyHint='interactive']
   * @param {number} [options.maxResumeRetries=5]
   * @param {Function} [options.AudioContextCtor]
   * @param {Function} [options.AudioWorkletNodeCtor]
   * @param {(err: Error) => void} [options.onError]
   * @param {(state: string) => void} [options.onStateChange]
   * @param {(data: any) => void} [options.onWorkletMessage]
   */
  constructor(options = {}) {
    const opts = options && typeof options === 'object' ? options : {};

    this._requestedSampleRate =
      typeof opts.sampleRate === 'number' && Number.isFinite(opts.sampleRate) && opts.sampleRate > 0
        ? opts.sampleRate
        : DEFAULT_SAMPLE_RATE;
    this._requestedChannels =
      Number.isInteger(opts.channels) && opts.channels > 0
        ? opts.channels
        : DEFAULT_CHANNELS;
    this._requestedCapacityFrames =
      Number.isInteger(opts.capacityFrames) && opts.capacityFrames > 0
        ? opts.capacityFrames
        : DEFAULT_CAPACITY_FRAMES;

    this._workletUrl =
      typeof opts.workletUrl === 'string' && opts.workletUrl.length > 0
        ? opts.workletUrl
        : DEFAULT_WORKLET_URL;
    this._processorName =
      typeof opts.processorName === 'string' && opts.processorName.length > 0
        ? opts.processorName
        : PROCESSOR_NAME;
    this._latencyHint = opts.latencyHint !== undefined ? opts.latencyHint : 'interactive';
    this._smoothingFrames = opts.smoothingFrames;
    this._maxResumeRetries = sanitizeFiniteNonNegInt(
      opts.maxResumeRetries,
      DEFAULT_MAX_RESUME_RETRIES
    );

    this._AudioContextCtor =
      opts.AudioContextCtor !== undefined
        ? opts.AudioContextCtor
        : resolveDefaultAudioContextCtor();
    this._AudioWorkletNodeCtor =
      opts.AudioWorkletNodeCtor !== undefined
        ? opts.AudioWorkletNodeCtor
        : resolveDefaultAudioWorkletNodeCtor();

    this._onError = typeof opts.onError === 'function' ? opts.onError : null;
    this._onStateChange = typeof opts.onStateChange === 'function' ? opts.onStateChange : null;
    this._onWorkletMessage =
      typeof opts.onWorkletMessage === 'function' ? opts.onWorkletMessage : null;

    const rawInitialVolume =
      opts.volume !== undefined ? opts.volume : opts.gain !== undefined ? opts.gain : 1.0;
    this._volume = clampVolume(rawInitialVolume, 1.0);
    this._muted = Boolean(opts.muted);

    this._sab = null;
    this._ctrl = null;
    this._capacityFrames = this._requestedCapacityFrames;
    this._channels = this._requestedChannels;

    this._context = null;
    this._node = null;
    this._status = 'idle';
    this._error = null;
    this._destroyed = false;
    this._startPromise = null;
    this._resumePromise = null;
    this._gestureCleanup = null;
    this._boundContextStateChange = null;

    this._resumeAttempts = 0;
    this._resumeSuccesses = 0;
    this._resumeFailures = 0;
    this._stateTransitions = 0;
    this._lastResumeError = null;
    this._lastTrackedState = 'idle';
    this._gainClamps = 0;
    this._invalidGainInputs = 0;

    const initialSab = opts.sab || (opts.ringBuffer && opts.ringBuffer.sab) || null;
    if (initialSab) {
      this.attachSab(initialSab);
    } else if (typeof SharedArrayBuffer !== 'undefined') {
      this.attachSab(
        createAudioRingSab(this._requestedCapacityFrames, this._requestedChannels)
      );
    }
  }

  get sab() {
    return this._sab;
  }

  get context() {
    return this._context;
  }

  get node() {
    return this._node;
  }

  get volume() {
    return this._volume;
  }

  get gain() {
    return this._volume;
  }

  get muted() {
    return this._muted;
  }

  get error() {
    return this._error;
  }

  get state() {
    if (this._destroyed) {
      return 'closed';
    }
    if (this._status === 'unsupported' || this._status === 'error') {
      return this._status;
    }
    if (
      this._context &&
      typeof this._context.state === 'string' &&
      this._context.state.length > 0
    ) {
      return this._context.state;
    }
    return this._status;
  }

  get outputState() {
    return sanitizeOutputState(this.state);
  }

  get sampleRate() {
    if (
      this._context &&
      typeof this._context.sampleRate === 'number' &&
      Number.isFinite(this._context.sampleRate) &&
      this._context.sampleRate > 0
    ) {
      return this._context.sampleRate;
    }
    return this._requestedSampleRate;
  }

  /**
   * Attaches an existing SharedArrayBuffer (e.g. from AudioRingBuffer) to the engine
   * and forwards it to the active AudioWorkletNode if already running.
   */
  attachSab(sab) {
    if (typeof SharedArrayBuffer === 'undefined' || !(sab instanceof SharedArrayBuffer)) {
      throw new TypeError('AudioEngine.attachSab requires a SharedArrayBuffer');
    }
    if (sab.byteLength < RING_HEADER_BYTES) {
      throw new RangeError(
        `SharedArrayBuffer too small (${sab.byteLength} bytes; need > ${RING_HEADER_BYTES})`
      );
    }

    const ctrl = new Int32Array(sab, 0, RING_HEADER_INT32_COUNT);
    let cap = Atomics.load(ctrl, HDR_CAPACITY_FRAMES);
    let ch = Atomics.load(ctrl, HDR_CHANNELS);

    if (cap === 0) {
      ch = ch > 0 ? ch : this._requestedChannels;
      const maxFrames = Math.floor(
        (sab.byteLength - RING_HEADER_BYTES) / (ch * Float32Array.BYTES_PER_ELEMENT)
      );
      if (maxFrames <= 0) {
        throw new RangeError('SharedArrayBuffer has zero frame capacity');
      }
      cap = maxFrames;
      Atomics.store(ctrl, HDR_WRITE_CURSOR, 0);
      Atomics.store(ctrl, HDR_READ_CURSOR, 0);
      Atomics.store(ctrl, HDR_CAPACITY_FRAMES, cap);
      Atomics.store(ctrl, HDR_CHANNELS, ch);
      Atomics.store(ctrl, HDR_UNDERRUNS, 0);
      Atomics.store(ctrl, HDR_UNDERRUN_FRAMES, 0);
      Atomics.store(ctrl, HDR_OVERFLOWS, 0);
      Atomics.store(ctrl, HDR_DROPPED_FRAMES, 0);
    } else {
      ch = ch > 0 ? ch : this._requestedChannels;
      const requiredBytes = audioRingByteLength(cap, ch);
      if (sab.byteLength < requiredBytes) {
        throw new RangeError(
          `SharedArrayBuffer (${sab.byteLength} bytes) is smaller than header requires (${requiredBytes} bytes)`
        );
      }
    }

    this._sab = sab;
    this._ctrl = ctrl;
    this._capacityFrames = cap;
    this._channels = ch;

    if (this._node && this._node.port && typeof this._node.port.postMessage === 'function') {
      this._node.port.postMessage({ type: 'attach', sab });
    }

    return sab;
  }

  _fail(status, message, cause = null) {
    const err = cause instanceof Error ? cause : new Error(message);
    if (cause instanceof Error && message && err.message !== message) {
      err.diagnostic = message;
    }
    this._status = status;
    this._error = err;
    if (this._onError) {
      try {
        this._onError(err);
      } catch {
        // Never allow user callbacks to break graceful audio error reporting.
      }
    }
    this._emitStateChange();
    return {
      ok: false,
      state: this.state,
      sampleRate: this.sampleRate,
      sab: this._sab,
      error: err,
    };
  }

  _recordStateTransition(nextState) {
    const normalized = sanitizeOutputState(nextState);
    if (normalized !== this._lastTrackedState) {
      this._lastTrackedState = normalized;
      this._stateTransitions += 1;
    }
  }

  _emitStateChange() {
    const currentState = this.state;
    this._recordStateTransition(currentState);
    if (this._onStateChange) {
      try {
        this._onStateChange(currentState);
      } catch {
        // Ignore callback exceptions.
      }
    }
  }

  async _safeCloseContext(ctx) {
    if (!ctx || typeof ctx.close !== 'function') {
      return;
    }
    if (ctx.state === 'closed') {
      return;
    }
    try {
      await ctx.close();
    } catch {
      // Ignore close errors during teardown / fallback.
    }
  }

  /**
   * Initializes the AudioContext, loads the AudioWorklet module, and connects
   * the AudioWorkletNode to `context.destination`.
   *
   * Never throws unhandled exceptions when WebAudio / AudioWorklet / SAB is
   * unsupported; returns `{ ok: false, state, error }` instead.
   */
  async start() {
    if (this._destroyed) {
      return {
        ok: false,
        state: 'closed',
        sampleRate: this.sampleRate,
        sab: this._sab,
        error: new Error('AudioEngine has been destroyed'),
      };
    }

    if (this._node && this._context) {
      return {
        ok: true,
        state: this.state,
        sampleRate: this.sampleRate,
        sab: this._sab,
        error: null,
      };
    }

    if (this._startPromise) {
      return this._startPromise;
    }

    this._startPromise = this._startInternal().finally(() => {
      this._startPromise = null;
    });
    return this._startPromise;
  }

  async _startInternal() {
    this._status = 'initializing';
    this._error = null;

    if (!this._sab) {
      if (typeof SharedArrayBuffer === 'undefined') {
        return this._fail(
          'unsupported',
          'SharedArrayBuffer is unavailable; Cross-Origin Isolation (COOP/COEP) is required for low-latency audio'
        );
      }
      try {
        this.attachSab(
          createAudioRingSab(this._requestedCapacityFrames, this._requestedChannels)
        );
      } catch (err) {
        return this._fail('error', 'Failed to allocate audio SharedArrayBuffer', err);
      }
    }

    const AudioContextCtor = this._AudioContextCtor;
    if (typeof AudioContextCtor !== 'function') {
      return this._fail(
        'unsupported',
        'WebAudio AudioContext is not supported in this browser/environment'
      );
    }

    const AudioWorkletNodeCtor = this._AudioWorkletNodeCtor;
    if (typeof AudioWorkletNodeCtor !== 'function') {
      return this._fail(
        'unsupported',
        'AudioWorkletNode is not supported in this browser/environment (requires a secure context)'
      );
    }

    let ctx;
    try {
      ctx = new AudioContextCtor({
        sampleRate: this._requestedSampleRate,
        latencyHint: this._latencyHint,
      });
    } catch (err) {
      return this._fail('error', 'Failed to construct AudioContext', err);
    }

    if (!ctx.audioWorklet || typeof ctx.audioWorklet.addModule !== 'function') {
      await this._safeCloseContext(ctx);
      return this._fail(
        'unsupported',
        'AudioContext.audioWorklet is unavailable in this browser (requires HTTPS or localhost secure context)'
      );
    }

    try {
      await ctx.audioWorklet.addModule(this._workletUrl);
    } catch (err) {
      await this._safeCloseContext(ctx);
      return this._fail(
        'error',
        `Failed to load AudioWorklet module from ${this._workletUrl}`,
        err
      );
    }

    if (this._destroyed) {
      await this._safeCloseContext(ctx);
      return {
        ok: false,
        state: 'closed',
        sampleRate: this._requestedSampleRate,
        sab: this._sab,
        error: new Error('AudioEngine was destroyed during initialization'),
      };
    }

    const processorOptions = {
      sab: this._sab,
      gain: this._volume,
      muted: this._muted,
    };
    if (this._smoothingFrames !== undefined) {
      processorOptions.smoothingFrames = this._smoothingFrames;
    }

    let node;
    try {
      node = new AudioWorkletNodeCtor(ctx, this._processorName, {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [this._channels],
        processorOptions,
      });
    } catch (err) {
      await this._safeCloseContext(ctx);
      return this._fail(
        'error',
        `Failed to instantiate AudioWorkletNode "${this._processorName}"`,
        err
      );
    }

    if (node.port) {
      node.port.onmessage = (event) => {
        const data = event ? event.data : null;
        if (data && data.type === 'error') {
          const workletErr = new Error(data.error || 'AudioWorklet processor error');
          this._error = workletErr;
          if (this._onError) {
            try {
              this._onError(workletErr);
            } catch {
              // Ignore callback errors.
            }
          }
        }
        if (this._onWorkletMessage) {
          try {
            this._onWorkletMessage(data);
          } catch {
            // Ignore callback errors.
          }
        }
      };
    }

    try {
      if (typeof node.connect === 'function' && ctx.destination) {
        node.connect(ctx.destination);
      }
    } catch (err) {
      try {
        if (typeof node.disconnect === 'function') node.disconnect();
      } catch {
        // Ignore.
      }
      await this._safeCloseContext(ctx);
      return this._fail('error', 'Failed to connect AudioWorkletNode to destination', err);
    }

    this._context = ctx;
    this._node = node;
    this._status = ctx.state || 'running';

    this._boundContextStateChange = () => {
      if (!this._destroyed && this._context) {
        this._status = this._context.state || this._status;
        this._emitStateChange();
      }
    };
    if (typeof ctx.addEventListener === 'function') {
      ctx.addEventListener('statechange', this._boundContextStateChange);
    } else if ('onstatechange' in ctx) {
      ctx.onstatechange = this._boundContextStateChange;
    }

    this._emitStateChange();

    return {
      ok: true,
      state: this.state,
      sampleRate: this.sampleRate,
      sab: this._sab,
      error: null,
    };
  }

  /**
   * Resumes a suspended AudioContext (typically triggered by a user gesture).
   * Optionally retries failed attempts when `options.maxRetries > 0`.
   *
   * @param {object} [options]
   * @param {number} [options.maxRetries=0] Additional retry attempts after an initial failure.
   * @param {number} [options.retryDelayMs=0] Delay in ms between retry attempts.
   * @returns {Promise<boolean>} True if context is running after resume.
   */
  resume(options = {}) {
    if (this._destroyed || !this._context) {
      return Promise.resolve(false);
    }
    if (this._context.state === 'running') {
      return Promise.resolve(true);
    }
    if (this._resumePromise) {
      return this._resumePromise;
    }

    const opts = options && typeof options === 'object' ? options : {};
    const maxRetries = sanitizeFiniteNonNegInt(opts.maxRetries, 0);
    const retryDelayMs = sanitizeFiniteNonNegInt(opts.retryDelayMs, 0);

    const promise = this._resumeInternal(maxRetries, retryDelayMs).finally(() => {
      if (this._resumePromise === promise) {
        this._resumePromise = null;
      }
    });
    this._resumePromise = promise;
    return promise;
  }

  async _resumeInternal(maxRetries, retryDelayMs) {
    const totalAttempts = 1 + maxRetries;

    for (let attempt = 0; attempt < totalAttempts; attempt++) {
      if (this._destroyed || !this._context) {
        return false;
      }
      if (this._context.state === 'running') {
        return true;
      }

      this._resumeAttempts += 1;

      if (typeof this._context.resume !== 'function') {
        const msg = 'AudioContext.resume is unavailable';
        this._resumeFailures += 1;
        this._lastResumeError = sanitizeResumeError(msg);
        this._error = new Error(msg);
        if (this._onError) {
          try {
            this._onError(this._error);
          } catch {
            // Ignore.
          }
        }
        return false;
      }

      try {
        await this._context.resume();
        if (this._destroyed || !this._context) {
          this._resumeFailures += 1;
          this._lastResumeError = sanitizeResumeError('AudioEngine was destroyed during resume');
          return false;
        }

        const nextState = sanitizeOutputState(this._context.state || 'running');
        this._status = nextState;
        this._emitStateChange();

        if (this._context.state === 'running') {
          this._resumeSuccesses += 1;
          this._lastResumeError = null;
          this._error = null;
          return true;
        }

        this._resumeFailures += 1;
        this._lastResumeError = sanitizeResumeError(
          `AudioContext remained in state "${nextState}" after resume`
        );
      } catch (err) {
        this._resumeFailures += 1;
        const sanitized = sanitizeResumeError(err);
        this._lastResumeError = sanitized;
        this._error =
          err instanceof Error ? err : new Error(sanitized || 'AudioContext resume failed');
        if (this._onError) {
          try {
            this._onError(this._error);
          } catch {
            // Ignore.
          }
        }
      }

      if (attempt + 1 < totalAttempts && !this._destroyed && this._context) {
        if (retryDelayMs > 0) {
          await new Promise((resolve) => setTimeout(resolve, retryDelayMs));
        }
      }
    }

    return false;
  }

  /**
   * Attaches `pointerdown` and `keydown` listeners to `target` to unlock a
   * suspended AudioContext on user gesture. Cleans up both listeners once
   * running (or when `destroy()` is called). If a gesture resume attempt fails,
   * listeners automatically re-arm up to `options.maxRetries` times.
   *
   * @param {EventTarget} [target] Defaults to `window` when available.
   * @param {object} [options]
   * @param {number} [options.maxRetries] Maximum gesture retry re-arms after an initial failure.
   * @returns {() => void} Manual cleanup function.
   */
  resumeOnUserGesture(target, options = {}) {
    const resolvedTarget =
      target ||
      (typeof window !== 'undefined'
        ? window
        : typeof globalThis !== 'undefined' && typeof globalThis.addEventListener === 'function'
          ? globalThis
          : null);

    if (
      this._destroyed ||
      this._status === 'unsupported' ||
      !resolvedTarget ||
      typeof resolvedTarget.addEventListener !== 'function'
    ) {
      return () => {};
    }

    if (this._context && this._context.state === 'running') {
      return () => {};
    }

    // Replace any previously armed unlock listeners to avoid leaks.
    if (this._gestureCleanup) {
      this._gestureCleanup();
      this._gestureCleanup = null;
    }

    const opts = options && typeof options === 'object' ? options : {};
    let retriesRemaining = sanitizeFiniteNonNegInt(opts.maxRetries, this._maxResumeRetries);
    let disposed = false;
    let activeCleanup = null;
    const events = ['pointerdown', 'keydown'];

    const removeListeners = () => {
      if (activeCleanup) {
        const fn = activeCleanup;
        activeCleanup = null;
        fn();
      }
    };

    const cleanup = () => {
      if (disposed) {
        return;
      }
      disposed = true;
      removeListeners();
      if (this._gestureCleanup === cleanup) {
        this._gestureCleanup = null;
      }
    };

    const arm = () => {
      if (disposed || this._destroyed) {
        return;
      }

      const onGesture = () => {
        removeListeners();
        Promise.resolve(this.resume()).then((resumed) => {
          if (
            !resumed &&
            !disposed &&
            !this._destroyed &&
            this._context &&
            this._context.state !== 'running' &&
            retriesRemaining > 0
          ) {
            retriesRemaining -= 1;
            arm();
          } else if (resumed || this._destroyed || retriesRemaining <= 0) {
            cleanup();
          }
        });
      };

      for (const evt of events) {
        resolvedTarget.addEventListener(evt, onGesture, { once: true, passive: true });
      }

      activeCleanup = () => {
        if (typeof resolvedTarget.removeEventListener === 'function') {
          for (const evt of events) {
            resolvedTarget.removeEventListener(evt, onGesture);
          }
        }
      };
    };

    arm();
    this._gestureCleanup = cleanup;
    return cleanup;
  }

  /**
   * Sets master output volume clamped to [0, 1] and forwards to the worklet.
   * Rejects non-finite (`NaN`, `±Infinity`) and non-number inputs without
   * mutating the active gain.
   *
   * @param {number} volume
   * @returns {number} Active sanitized volume in [0, 1].
   */
  setVolume(volume) {
    if (typeof volume !== 'number' || !Number.isFinite(volume)) {
      this._invalidGainInputs += 1;
      return this._volume;
    }

    const clamped = clampVolume(volume, this._volume);
    if (clamped !== volume || Object.is(volume, -0)) {
      this._gainClamps += 1;
    }

    this._volume = clamped;
    if (this._node && this._node.port && typeof this._node.port.postMessage === 'function') {
      this._node.port.postMessage({ type: 'setGain', gain: this._volume });
    }
    return this._volume;
  }

  /**
   * Alias for `setVolume(gain)` with identical `[0, 1]` finite clamping guardrails.
   *
   * @param {number} gain
   * @returns {number} Active sanitized gain in [0, 1].
   */
  setGain(gain) {
    return this.setVolume(gain);
  }

  /**
   * Sets mute state (strictly coerced to boolean) and forwards to the worklet.
   * @param {boolean} muted
   * @returns {boolean}
   */
  setMuted(muted) {
    this._muted = Boolean(muted);
    if (this._node && this._node.port && typeof this._node.port.postMessage === 'function') {
      this._node.port.postMessage({ type: 'setMuted', muted: this._muted });
    }
    return this._muted;
  }

  /**
   * Reads live WebAudio + SAB ring buffer telemetry without blocking or
   * round-tripping to the audio thread.
   */
  getTelemetry() {
    let bufferedFrames = 0;
    let underruns = 0;
    let underrunFrames = 0;
    let overflows = 0;
    let droppedFrames = 0;

    if (this._ctrl && this._capacityFrames > 0) {
      const write = Atomics.load(this._ctrl, HDR_WRITE_CURSOR);
      const read = Atomics.load(this._ctrl, HDR_READ_CURSOR);
      let used = write - read;
      if (used < 0) {
        used += this._capacityFrames * 2;
      }
      bufferedFrames = sanitizeFiniteNonNegInt(used, 0);
      underruns = sanitizeFiniteNonNegInt(Atomics.load(this._ctrl, HDR_UNDERRUNS), 0);
      underrunFrames = sanitizeFiniteNonNegInt(Atomics.load(this._ctrl, HDR_UNDERRUN_FRAMES), 0);
      overflows = sanitizeFiniteNonNegInt(Atomics.load(this._ctrl, HDR_OVERFLOWS), 0);
      droppedFrames = sanitizeFiniteNonNegInt(Atomics.load(this._ctrl, HDR_DROPPED_FRAMES), 0);
    }

    const outputState = sanitizeOutputState(this.state);
    const gain = clampVolume(this._volume, 1.0);
    const muted = Boolean(this._muted);
    const effectiveGain = muted ? 0.0 : gain;

    return {
      sampleRate: sanitizePositiveFiniteNumber(this.sampleRate, this._requestedSampleRate),
      state: this.state,
      outputState,
      channels: sanitizeFiniteNonNegInt(this._channels, this._requestedChannels),
      capacityFrames: sanitizeFiniteNonNegInt(this._capacityFrames, this._requestedCapacityFrames),
      bufferedFrames,
      underruns,
      underrunFrames,
      overflows,
      droppedFrames,
      volume: gain,
      gain,
      muted,
      effectiveGain,
      gainClamps: sanitizeFiniteNonNegInt(this._gainClamps, 0),
      invalidGainInputs: sanitizeFiniteNonNegInt(this._invalidGainInputs, 0),
      resumeAttempts: sanitizeFiniteNonNegInt(this._resumeAttempts, 0),
      resumeSuccesses: sanitizeFiniteNonNegInt(this._resumeSuccesses, 0),
      resumeFailures: sanitizeFiniteNonNegInt(this._resumeFailures, 0),
      stateTransitions: sanitizeFiniteNonNegInt(this._stateTransitions, 0),
      lastResumeError: this._lastResumeError,
      error: this._error ? sanitizeResumeError(this._error) : null,
    };
  }

  /**
   * Returns a sanitized, finite diagnostic snapshot of AudioContext resume/state
   * telemetry, clamped gain/mute controls, and live SAB ring counters.
   */
  getStats() {
    const telemetry = this.getTelemetry();
    return {
      outputState: telemetry.outputState,
      state: telemetry.outputState,
      sampleRate: telemetry.sampleRate,
      channels: telemetry.channels,
      capacityFrames: telemetry.capacityFrames,
      bufferedFrames: telemetry.bufferedFrames,
      underruns: telemetry.underruns,
      underrunFrames: telemetry.underrunFrames,
      overflows: telemetry.overflows,
      droppedFrames: telemetry.droppedFrames,
      gain: telemetry.gain,
      volume: telemetry.volume,
      muted: telemetry.muted,
      effectiveGain: telemetry.effectiveGain,
      gainClamps: telemetry.gainClamps,
      invalidGainInputs: telemetry.invalidGainInputs,
      resumeAttempts: telemetry.resumeAttempts,
      resumeSuccesses: telemetry.resumeSuccesses,
      resumeFailures: telemetry.resumeFailures,
      stateTransitions: telemetry.stateTransitions,
      lastResumeError: telemetry.lastResumeError,
      error: telemetry.error,
    };
  }

  /**
   * Stops the AudioWorkletProcessor, removes any pending user-gesture unlock
   * listeners, disconnects the node, and closes the AudioContext.
   */
  async destroy() {
    if (this._destroyed) {
      return;
    }

    this._destroyed = true;
    this._status = 'closed';

    if (this._gestureCleanup) {
      this._gestureCleanup();
      this._gestureCleanup = null;
    }

    const node = this._node;
    this._node = null;
    if (node) {
      try {
        if (node.port) {
          if (typeof node.port.postMessage === 'function') {
            node.port.postMessage({ type: 'stop' });
          }
          node.port.onmessage = null;
          if (typeof node.port.close === 'function') {
            node.port.close();
          }
        }
      } catch {
        // Ignore teardown errors on port.
      }
      try {
        if (typeof node.disconnect === 'function') {
          node.disconnect();
        }
      } catch {
        // Ignore teardown errors on disconnect.
      }
    }

    const ctx = this._context;
    this._context = null;
    if (ctx) {
      if (this._boundContextStateChange) {
        if (typeof ctx.removeEventListener === 'function') {
          ctx.removeEventListener('statechange', this._boundContextStateChange);
        } else if (ctx.onstatechange === this._boundContextStateChange) {
          ctx.onstatechange = null;
        }
        this._boundContextStateChange = null;
      }
      await this._safeCloseContext(ctx);
    }

    this._emitStateChange();
  }
}
