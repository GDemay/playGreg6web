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

export function clampVolume(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return 1.0;
  }
  if (value <= 0.0) return 0.0;
  if (value >= 1.0) return 1.0;
  return value;
}

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
   * @param {boolean} [options.muted=false]
   * @param {number} [options.smoothingFrames]
   * @param {string|number} [options.latencyHint='interactive']
   * @param {Function} [options.AudioContextCtor]
   * @param {Function} [options.AudioWorkletNodeCtor]
   * @param {(err: Error) => void} [options.onError]
   * @param {(state: string) => void} [options.onStateChange]
   * @param {(data: any) => void} [options.onWorkletMessage]
   */
  constructor(options = {}) {
    this._requestedSampleRate =
      typeof options.sampleRate === 'number' && Number.isFinite(options.sampleRate) && options.sampleRate > 0
        ? options.sampleRate
        : DEFAULT_SAMPLE_RATE;
    this._requestedChannels =
      Number.isInteger(options.channels) && options.channels > 0
        ? options.channels
        : DEFAULT_CHANNELS;
    this._requestedCapacityFrames =
      Number.isInteger(options.capacityFrames) && options.capacityFrames > 0
        ? options.capacityFrames
        : DEFAULT_CAPACITY_FRAMES;

    this._workletUrl =
      typeof options.workletUrl === 'string' && options.workletUrl.length > 0
        ? options.workletUrl
        : DEFAULT_WORKLET_URL;
    this._processorName =
      typeof options.processorName === 'string' && options.processorName.length > 0
        ? options.processorName
        : PROCESSOR_NAME;
    this._latencyHint = options.latencyHint !== undefined ? options.latencyHint : 'interactive';
    this._smoothingFrames = options.smoothingFrames;

    this._AudioContextCtor =
      options.AudioContextCtor !== undefined
        ? options.AudioContextCtor
        : resolveDefaultAudioContextCtor();
    this._AudioWorkletNodeCtor =
      options.AudioWorkletNodeCtor !== undefined
        ? options.AudioWorkletNodeCtor
        : resolveDefaultAudioWorkletNodeCtor();

    this._onError = typeof options.onError === 'function' ? options.onError : null;
    this._onStateChange = typeof options.onStateChange === 'function' ? options.onStateChange : null;
    this._onWorkletMessage =
      typeof options.onWorkletMessage === 'function' ? options.onWorkletMessage : null;

    this._volume = clampVolume(options.volume !== undefined ? options.volume : 1.0);
    this._muted = Boolean(options.muted);

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
    this._gestureCleanup = null;
    this._boundContextStateChange = null;

    const initialSab =
      options.sab || (options.ringBuffer && options.ringBuffer.sab) || null;
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
    if (this._context && typeof this._context.state === 'string') {
      return this._context.state;
    }
    return this._status;
  }

  get sampleRate() {
    if (this._context && typeof this._context.sampleRate === 'number') {
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

  _emitStateChange() {
    if (this._onStateChange) {
      try {
        this._onStateChange(this.state);
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
   * @returns {Promise<boolean>} True if context is running after resume.
   */
  async resume() {
    if (this._destroyed || !this._context) {
      return false;
    }
    if (this._context.state === 'running') {
      return true;
    }
    if (typeof this._context.resume !== 'function') {
      return false;
    }
    try {
      await this._context.resume();
      if (!this._destroyed && this._context) {
        this._status = this._context.state || 'running';
        this._emitStateChange();
        return this._context.state === 'running';
      }
      return false;
    } catch (err) {
      this._error = err instanceof Error ? err : new Error(String(err));
      if (this._onError) {
        try {
          this._onError(this._error);
        } catch {
          // Ignore.
        }
      }
      return false;
    }
  }

  /**
   * Attaches once-only `pointerdown` and `keydown` listeners to `target` to
   * unlock a suspended AudioContext on the first user gesture, and cleans up
   * both listeners as soon as either fires (or when `destroy()` is called).
   *
   * @param {EventTarget} [target] Defaults to `window` when available.
   * @returns {() => void} Manual cleanup function.
   */
  resumeOnUserGesture(target) {
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

    let cleaned = false;
    const events = ['pointerdown', 'keydown'];

    const cleanup = () => {
      if (cleaned) {
        return;
      }
      cleaned = true;
      if (typeof resolvedTarget.removeEventListener === 'function') {
        for (const evt of events) {
          resolvedTarget.removeEventListener(evt, onGesture);
        }
      }
      if (this._gestureCleanup === cleanup) {
        this._gestureCleanup = null;
      }
    };

    const onGesture = () => {
      cleanup();
      void this.resume();
    };

    for (const evt of events) {
      resolvedTarget.addEventListener(evt, onGesture, { once: true, passive: true });
    }

    this._gestureCleanup = cleanup;
    return cleanup;
  }

  /**
   * Sets master output volume clamped to [0, 1] and forwards to the worklet.
   * @param {number} volume
   * @returns {number} Clamped volume.
   */
  setVolume(volume) {
    this._volume = clampVolume(volume);
    if (this._node && this._node.port && typeof this._node.port.postMessage === 'function') {
      this._node.port.postMessage({ type: 'setGain', gain: this._volume });
    }
    return this._volume;
  }

  /**
   * Sets mute state and forwards to the worklet.
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
      bufferedFrames = used;
      underruns = Atomics.load(this._ctrl, HDR_UNDERRUNS);
      underrunFrames = Atomics.load(this._ctrl, HDR_UNDERRUN_FRAMES);
      overflows = Atomics.load(this._ctrl, HDR_OVERFLOWS);
      droppedFrames = Atomics.load(this._ctrl, HDR_DROPPED_FRAMES);
    }

    return {
      sampleRate: this.sampleRate,
      state: this.state,
      channels: this._channels,
      capacityFrames: this._capacityFrames,
      bufferedFrames,
      underruns,
      underrunFrames,
      overflows,
      droppedFrames,
      volume: this._volume,
      muted: this._muted,
      effectiveGain: this._muted ? 0.0 : this._volume,
      error: this._error ? this._error.message : null,
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
