'use strict';

(function (root) {
  const DEFAULT_SAMPLE_RATE = 48000;
  const MIN_SAMPLE_RATE = 8000;
  const MAX_SAMPLE_RATE = 384000;
  const DEFAULT_CHANNELS = 2;
  const MAX_CHANNELS = 8;
  const DEFAULT_CAPACITY_FRAMES = 4096;
  const MAX_CAPACITY_FRAMES = 1048576;

  // Bounded headless/suspended drain defaults (48 kHz real-time paced).
  const DEFAULT_DRAIN_INTERVAL_MS = 20;
  const MIN_DRAIN_INTERVAL_MS = 4;
  const MAX_DRAIN_INTERVAL_MS = 250;
  const MAX_DRAIN_ELAPSED_MS = 100;

  // Single-packet PCM safety bound (4 MiB = 524,288 stereo float32 frames ~= 10.9s @ 48 kHz).
  const DEFAULT_MAX_PCM_BYTES = 4 * 1024 * 1024;
  const MAX_ID_STRING_LEN = 128;

  const OUTPUT_STATES = Object.freeze({
    ACTIVE: 'active',
    SUSPENDED: 'suspended',
    HEADLESS_DRAIN: 'headless-drain',
  });

  // Atomic Int32 header layout at byte offset 0..32 of the SharedArrayBuffer.
  const HDR_WRITE_CURSOR = 0;
  const HDR_READ_CURSOR = 1;
  const HDR_CAPACITY_FRAMES = 2;
  const HDR_CHANNELS = 3;
  const HDR_UNDERRUNS = 4;
  const HDR_UNDERRUN_FRAMES = 5;
  const HDR_OVERFLOWS = 6;
  const HDR_DROPPED_FRAMES = 7;

  const RING_HEADER_INT32_COUNT = 8;
  const RING_HEADER_BYTES = RING_HEADER_INT32_COUNT * Int32Array.BYTES_PER_ELEMENT;

  function clampVolume(value) {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      return 1.0;
    }
    if (value < 0) return 0.0;
    if (value > 1) return 1.0;
    return value;
  }

  function sanitizeNonNegInt(value, fallback = 0) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
      return fallback;
    }
    return Math.floor(value);
  }

  function sanitizePositiveInt(value, fallback, minVal, maxVal) {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      return fallback;
    }
    const n = Math.floor(value);
    if (n < minVal) return fallback;
    if (maxVal !== undefined && n > maxVal) return maxVal;
    return n;
  }

  function sanitizeSampleRate(value) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
      return DEFAULT_SAMPLE_RATE;
    }
    if (value < MIN_SAMPLE_RATE) return MIN_SAMPLE_RATE;
    if (value > MAX_SAMPLE_RATE) return MAX_SAMPLE_RATE;
    return Math.floor(value);
  }

  function sanitizeMessageId(id) {
    if (typeof id === 'number' && Number.isFinite(id)) {
      return id;
    }
    if (typeof id === 'string') {
      return id.length > MAX_ID_STRING_LEN ? id.slice(0, MAX_ID_STRING_LEN) : id;
    }
    return undefined;
  }

  function isSharedArrayBuffer(buf) {
    return typeof SharedArrayBuffer !== 'undefined' && buf instanceof SharedArrayBuffer;
  }

  function isDetachedArrayBuffer(buf) {
    if (!(buf instanceof ArrayBuffer)) {
      return false;
    }
    if (buf.detached === true) {
      return true;
    }
    try {
      // Accessing byteLength never throws in ES2024, but guard older engines.
      void buf.byteLength;
      return false;
    } catch (_err) {
      return true;
    }
  }

  function normalizeOutputState(value, fallback = OUTPUT_STATES.HEADLESS_DRAIN) {
    if (typeof value !== 'string') {
      return fallback;
    }
    const normalized = value.trim().toLowerCase();
    if (normalized === OUTPUT_STATES.ACTIVE || normalized === 'running') {
      return OUTPUT_STATES.ACTIVE;
    }
    if (normalized === OUTPUT_STATES.SUSPENDED || normalized === 'interrupted') {
      return OUTPUT_STATES.SUSPENDED;
    }
    if (
      normalized === OUTPUT_STATES.HEADLESS_DRAIN ||
      normalized === 'headless' ||
      normalized === 'unavailable' ||
      normalized === 'unsupported' ||
      normalized === 'closed' ||
      normalized === 'error'
    ) {
      return OUTPUT_STATES.HEADLESS_DRAIN;
    }
    return fallback;
  }

  function detectDefaultOutputState() {
    const hasAudioCtx =
      typeof globalThis !== 'undefined' &&
      (typeof globalThis.AudioContext === 'function' ||
        typeof globalThis.webkitAudioContext === 'function');
    const hasWorklet =
      typeof globalThis !== 'undefined' &&
      typeof globalThis.AudioWorkletNode === 'function';
    if (hasAudioCtx && hasWorklet) {
      return OUTPUT_STATES.ACTIVE;
    }
    return OUTPUT_STATES.HEADLESS_DRAIN;
  }

  function sanitizeErrorMessage(err) {
    const raw = err && typeof err.message === 'string' ? err.message : String(err || 'Unknown audio worker error');
    // Strip control characters and clamp length to prevent noisy/leaky error strings.
    const clean = raw.replace(/[\r\n\t]+/g, ' ').trim();
    return clean.length > 200 ? clean.slice(0, 200) : clean;
  }

  /**
   * Filters a postMessage transfer list so SharedArrayBuffers, detached
   * ArrayBuffers, TypedArray views, and duplicates never trigger DataCloneError.
   *
   * @param {any[]} transfer
   * @returns {any[]}
   */
  function sanitizeTransferList(transfer) {
    if (!Array.isArray(transfer) || transfer.length === 0) {
      return [];
    }
    const out = [];
    const seen = new Set();
    for (let i = 0; i < transfer.length; i++) {
      const item = transfer[i];
      if (!item || typeof item !== 'object') {
        continue;
      }
      if (isSharedArrayBuffer(item)) {
        continue;
      }
      if (item instanceof ArrayBuffer && isDetachedArrayBuffer(item)) {
        continue;
      }
      if (ArrayBuffer.isView(item)) {
        continue;
      }
      if (!seen.has(item)) {
        seen.add(item);
        out.push(item);
      }
    }
    return out;
  }

  /**
   * Validates and extracts PCM Float32 samples safely from any supported payload.
   * - Rejects null, malformed types, detached ArrayBuffers, and oversized payloads.
   * - Copies SharedArrayBuffer-backed views (and unaligned TypedArray views) into
   *   a fresh standalone ArrayBuffer-backed Float32Array to prevent concurrent
   *   mutation races and DataCloneError on transfer.
   * - Sanitizes non-finite float samples (NaN / +-Infinity) to 0.0.
   *
   * @param {any} samples
   * @param {{ maxPcmBytes?: number }} [options]
   * @returns {{ ok: true, pcm: Float32Array, copiedFromSab: boolean, sanitizedSamples: number } | { ok: false, error: string }}
   */
  function extractSafePcmSamples(samples, options = {}) {
    const maxPcmBytes = sanitizePositiveInt(
      options && options.maxPcmBytes,
      DEFAULT_MAX_PCM_BYTES,
      Float32Array.BYTES_PER_ELEMENT,
      64 * 1024 * 1024
    );
    const maxSamples = Math.floor(maxPcmBytes / Float32Array.BYTES_PER_ELEMENT);

    if (samples === null || samples === undefined) {
      return { ok: false, error: 'PCM samples missing or null' };
    }

    // 1. Raw SharedArrayBuffer
    if (isSharedArrayBuffer(samples)) {
      const byteLen = samples.byteLength;
      if (byteLen > maxPcmBytes) {
        return {
          ok: false,
          error: `PCM payload (${byteLen} bytes) exceeds max allowed (${maxPcmBytes} bytes)`,
        };
      }
      const sampleCount = Math.floor(byteLen / Float32Array.BYTES_PER_ELEMENT);
      const view = new Float32Array(samples, 0, sampleCount);
      const copy = new Float32Array(sampleCount);
      let sanitizedSamples = 0;
      for (let i = 0; i < sampleCount; i++) {
        const v = view[i];
        if (Number.isFinite(v)) {
          copy[i] = v;
        } else {
          copy[i] = 0.0;
          sanitizedSamples += 1;
        }
      }
      return { ok: true, pcm: copy, copiedFromSab: true, sanitizedSamples };
    }

    // 2. Raw ArrayBuffer
    if (samples instanceof ArrayBuffer) {
      if (isDetachedArrayBuffer(samples)) {
        return { ok: false, error: 'PCM ArrayBuffer is detached' };
      }
      const byteLen = samples.byteLength;
      if (byteLen > maxPcmBytes) {
        return {
          ok: false,
          error: `PCM payload (${byteLen} bytes) exceeds max allowed (${maxPcmBytes} bytes)`,
        };
      }
      const sampleCount = Math.floor(byteLen / Float32Array.BYTES_PER_ELEMENT);
      const view = new Float32Array(samples, 0, sampleCount);
      let sanitizedSamples = 0;
      let copy = null;
      for (let i = 0; i < sampleCount; i++) {
        if (!Number.isFinite(view[i])) {
          if (!copy) {
            copy = new Float32Array(view);
          }
          copy[i] = 0.0;
          sanitizedSamples += 1;
        }
      }
      return { ok: true, pcm: copy || view, copiedFromSab: false, sanitizedSamples };
    }

    // 3. TypedArray or DataView (Float32Array, Uint8Array, etc.)
    if (ArrayBuffer.isView(samples)) {
      let backing;
      let byteOffset;
      let byteLength;
      try {
        backing = samples.buffer;
        byteOffset = samples.byteOffset;
        byteLength = samples.byteLength;
      } catch (_err) {
        return { ok: false, error: 'PCM TypedArray view is unreadable or detached' };
      }

      if (!backing) {
        return { ok: false, error: 'PCM view has no backing buffer' };
      }
      const fromSab = isSharedArrayBuffer(backing);
      if (!fromSab && isDetachedArrayBuffer(backing)) {
        return { ok: false, error: 'PCM TypedArray backing ArrayBuffer is detached' };
      }
      if (byteLength > maxPcmBytes) {
        return {
          ok: false,
          error: `PCM payload (${byteLength} bytes) exceeds max allowed (${maxPcmBytes} bytes)`,
        };
      }

      const alignedByteLength =
        Math.floor(byteLength / Float32Array.BYTES_PER_ELEMENT) * Float32Array.BYTES_PER_ELEMENT;
      const sampleCount = alignedByteLength / Float32Array.BYTES_PER_ELEMENT;

      if (fromSab || byteOffset % Float32Array.BYTES_PER_ELEMENT !== 0) {
        // Always copy SAB-backed or unaligned views into a fresh, aligned ArrayBuffer.
        let copy;
        if (samples instanceof Float32Array) {
          copy = new Float32Array(sampleCount);
          for (let i = 0; i < sampleCount; i++) {
            copy[i] = samples[i];
          }
        } else {
          const byteCopy = new Uint8Array(alignedByteLength);
          byteCopy.set(new Uint8Array(backing, byteOffset, alignedByteLength));
          copy = new Float32Array(byteCopy.buffer);
        }

        let sanitizedSamples = 0;
        for (let i = 0; i < sampleCount; i++) {
          if (!Number.isFinite(copy[i])) {
            copy[i] = 0.0;
            sanitizedSamples += 1;
          }
        }
        return { ok: true, pcm: copy, copiedFromSab: fromSab, sanitizedSamples };
      }

      const view =
        samples instanceof Float32Array
          ? samples
          : new Float32Array(backing, byteOffset, sampleCount);
      let sanitizedSamples = 0;
      let copy = null;
      for (let i = 0; i < view.length; i++) {
        if (!Number.isFinite(view[i])) {
          if (!copy) {
            copy = new Float32Array(view);
          }
          copy[i] = 0.0;
          sanitizedSamples += 1;
        }
      }
      return { ok: true, pcm: copy || view, copiedFromSab: false, sanitizedSamples };
    }

    // 4. Plain JS number[]
    if (Array.isArray(samples)) {
      if (samples.length > maxSamples) {
        return {
          ok: false,
          error: `PCM array (${samples.length} samples) exceeds max allowed (${maxSamples} samples)`,
        };
      }
      const out = new Float32Array(samples.length);
      let sanitizedSamples = 0;
      for (let i = 0; i < samples.length; i++) {
        const v = samples[i];
        if (typeof v !== 'number') {
          return { ok: false, error: `PCM array contains non-numeric element at index ${i}` };
        }
        if (Number.isFinite(v)) {
          out[i] = v;
        } else {
          out[i] = 0.0;
          sanitizedSamples += 1;
        }
      }
      return { ok: true, pcm: out, copiedFromSab: false, sanitizedSamples };
    }

    return {
      ok: false,
      error: 'PCM samples must be a Float32Array, ArrayBuffer, SharedArrayBuffer, TypedArray, or number[]',
    };
  }

  /**
   * Produces a standalone Float32Array + safe transfer list `[pcm.buffer]`
   * suitable for `postMessage(..., transfer)`. Always copies SAB-backed or
   * sliced views so transfer never throws DataCloneError or detaches guest memory.
   *
   * @param {any} samples
   * @param {{ maxPcmBytes?: number }} [options]
   */
  function clonePcmForTransfer(samples, options = {}) {
    const res = extractSafePcmSamples(samples, options);
    if (!res.ok) {
      return res;
    }
    const pcm = res.pcm;
    const needsOwnedCopy =
      isSharedArrayBuffer(pcm.buffer) ||
      pcm.byteOffset !== 0 ||
      pcm.byteLength !== pcm.buffer.byteLength;
    if (needsOwnedCopy) {
      const owned = new Float32Array(pcm.length);
      owned.set(pcm);
      return {
        ok: true,
        pcm: owned,
        transfer: [owned.buffer],
        copiedFromSab: res.copiedFromSab,
        sanitizedSamples: res.sanitizedSamples,
      };
    }
    return {
      ok: true,
      pcm,
      transfer: [pcm.buffer],
      copiedFromSab: res.copiedFromSab,
      sanitizedSamples: res.sanitizedSamples,
    };
  }

  function toFloat32Array(samples) {
    const extracted = extractSafePcmSamples(samples, { maxPcmBytes: 64 * 1024 * 1024 });
    if (!extracted.ok) {
      throw new TypeError(extracted.error);
    }
    return extracted.pcm;
  }

  /**
   * Builds a strictly sanitized `audio-stats` telemetry snapshot containing only
   * finite numeric counters, booleans, and whitelisted enum strings.
   * Strips raw buffers, SharedArrayBuffers, host paths, and unknown fields.
   *
   * @param {AudioMixerWorker|object|null} source
   * @param {{ now?: () => number, timestamp?: number }} [options]
   */
  function buildAudioTelemetrySnapshot(source, options = {}) {
    let raw = null;
    if (source && typeof source === 'object') {
      if (typeof source.getStats === 'function') {
        try {
          raw = source.getStats();
        } catch (_err) {
          raw = null;
        }
      } else {
        raw = source;
      }
    }

    const s = raw && typeof raw === 'object' ? raw : {};
    const nowFn = typeof options.now === 'function' ? options.now : () => Date.now();
    const rawTs = options.timestamp !== undefined ? options.timestamp : nowFn();
    const timestamp =
      typeof rawTs === 'number' && Number.isFinite(rawTs) && rawTs >= 0
        ? Math.floor(rawTs)
        : 0;

    const initialized = Boolean(s.initialized);
    const sampleRate = sanitizeSampleRate(s.sampleRate);
    const channels = sanitizePositiveInt(s.channels, DEFAULT_CHANNELS, 1, MAX_CHANNELS);
    const capacityFrames = sanitizeNonNegInt(s.capacityFrames, 0);
    const availableFrames = Math.min(
      sanitizeNonNegInt(s.availableFrames, 0),
      capacityFrames > 0 ? capacityFrames : sanitizeNonNegInt(s.availableFrames, 0)
    );
    const freeFrames =
      capacityFrames > 0
        ? Math.max(0, capacityFrames - availableFrames)
        : sanitizeNonNegInt(s.freeFrames, 0);
    const volume = clampVolume(s.volume !== undefined ? s.volume : 1.0);
    const muted = Boolean(s.muted);
    const effectiveGain = muted ? 0.0 : volume;
    const underruns = sanitizeNonNegInt(s.underruns, 0);
    const underrunFrames = sanitizeNonNegInt(s.underrunFrames, 0);
    const overflows = sanitizeNonNegInt(s.overflows, 0);
    const overruns = sanitizeNonNegInt(
      s.overruns !== undefined ? s.overruns : overflows,
      overflows
    );
    const droppedFrames = sanitizeNonNegInt(s.droppedFrames, 0);
    const totalPushedFrames = sanitizeNonNegInt(s.totalPushedFrames, 0);
    const totalPulledFrames = sanitizeNonNegInt(s.totalPulledFrames, 0);
    const framesDrained = sanitizeNonNegInt(s.framesDrained, 0);
    const drainTicks = sanitizeNonNegInt(s.drainTicks, 0);
    const rejectedPackets = sanitizeNonNegInt(s.rejectedPackets, 0);
    const sabCopiedPackets = sanitizeNonNegInt(s.sabCopiedPackets, 0);
    const sanitizedSamples = sanitizeNonNegInt(s.sanitizedSamples, 0);
    const outputState = normalizeOutputState(s.outputState, OUTPUT_STATES.HEADLESS_DRAIN);

    return {
      type: 'audio-stats',
      timestamp,
      stats: {
        initialized,
        outputState,
        sampleRate,
        channels,
        capacityFrames,
        availableFrames,
        freeFrames,
        volume,
        muted,
        effectiveGain,
        framesDrained,
        underruns,
        underrunFrames,
        overruns,
        overflows,
        droppedFrames,
        totalPushedFrames,
        totalPulledFrames,
        drainTicks,
        rejectedPackets,
        sabCopiedPackets,
        sanitizedSamples,
      },
    };
  }

  /**
   * Lock-free Single-Producer / Single-Consumer (SPSC) PCM audio ring buffer
   * backed by a SharedArrayBuffer.
   *
   * Uses double-modulo cursors in [0, 2 * capacityFrames) so 100% of
   * capacityFrames are usable without a wasted sentinel slot and for any
   * positive capacityFrames (power-of-two or arbitrary).
   */
  class AudioRingBuffer {
    static HEADER_INT32_COUNT = RING_HEADER_INT32_COUNT;
    static HEADER_BYTES = RING_HEADER_BYTES;

    static byteLength(capacityFrames, channels = DEFAULT_CHANNELS) {
      if (!Number.isInteger(capacityFrames) || capacityFrames <= 0) {
        throw new RangeError(`capacityFrames must be a positive integer, got ${capacityFrames}`);
      }
      if (!Number.isInteger(channels) || channels <= 0) {
        throw new RangeError(`channels must be a positive integer, got ${channels}`);
      }
      return (
        RING_HEADER_BYTES +
        capacityFrames * channels * Float32Array.BYTES_PER_ELEMENT
      );
    }

    static create(capacityFrames = DEFAULT_CAPACITY_FRAMES, channels = DEFAULT_CHANNELS) {
      const byteLen = AudioRingBuffer.byteLength(capacityFrames, channels);
      const sab = new SharedArrayBuffer(byteLen);
      return new AudioRingBuffer(sab, { capacityFrames, channels, initialize: true });
    }

    constructor(sab, options = {}) {
      if (typeof SharedArrayBuffer === 'undefined' || !(sab instanceof SharedArrayBuffer)) {
        throw new TypeError('AudioRingBuffer requires a SharedArrayBuffer');
      }
      if (sab.byteLength < RING_HEADER_BYTES) {
        throw new RangeError(
          `SharedArrayBuffer too small (${sab.byteLength} bytes; need > ${RING_HEADER_BYTES})`
        );
      }

      this._sab = sab;
      this._ctrl = new Int32Array(sab, 0, RING_HEADER_INT32_COUNT);

      const existingCap = Atomics.load(this._ctrl, HDR_CAPACITY_FRAMES);
      const existingCh = Atomics.load(this._ctrl, HDR_CHANNELS);
      const shouldInit = Boolean(options.initialize) || existingCap === 0;

      if (shouldInit) {
        const channels = options.channels !== undefined ? options.channels : DEFAULT_CHANNELS;
        if (!Number.isInteger(channels) || channels <= 0) {
          throw new RangeError(`channels must be a positive integer, got ${channels}`);
        }
        const maxFrames = Math.floor(
          (sab.byteLength - RING_HEADER_BYTES) /
            (channels * Float32Array.BYTES_PER_ELEMENT)
        );
        const capacityFrames =
          options.capacityFrames !== undefined ? options.capacityFrames : maxFrames;
        if (
          !Number.isInteger(capacityFrames) ||
          capacityFrames <= 0 ||
          capacityFrames > maxFrames
        ) {
          throw new RangeError(
            `capacityFrames (${capacityFrames}) exceeds SharedArrayBuffer capacity (${maxFrames})`
          );
        }

        Atomics.store(this._ctrl, HDR_WRITE_CURSOR, 0);
        Atomics.store(this._ctrl, HDR_READ_CURSOR, 0);
        Atomics.store(this._ctrl, HDR_CAPACITY_FRAMES, capacityFrames);
        Atomics.store(this._ctrl, HDR_CHANNELS, channels);
        Atomics.store(this._ctrl, HDR_UNDERRUNS, 0);
        Atomics.store(this._ctrl, HDR_UNDERRUN_FRAMES, 0);
        Atomics.store(this._ctrl, HDR_OVERFLOWS, 0);
        Atomics.store(this._ctrl, HDR_DROPPED_FRAMES, 0);

        this._capacityFrames = capacityFrames;
        this._channels = channels;
      } else {
        const channels = existingCh > 0 ? existingCh : (options.channels || DEFAULT_CHANNELS);
        const requiredBytes = AudioRingBuffer.byteLength(existingCap, channels);
        if (sab.byteLength < requiredBytes) {
          throw new RangeError(
            `SharedArrayBuffer (${sab.byteLength} bytes) is smaller than header requires (${requiredBytes} bytes)`
          );
        }
        this._capacityFrames = existingCap;
        this._channels = channels;
      }

      this._modulus = this._capacityFrames * 2;
      this._samples = new Float32Array(
        sab,
        RING_HEADER_BYTES,
        this._capacityFrames * this._channels
      );

      this._volume = clampVolume(options.volume !== undefined ? options.volume : 1.0);
      this._muted = Boolean(options.muted);
    }

    get sab() {
      return this._sab;
    }

    get capacityFrames() {
      return this._capacityFrames;
    }

    get channels() {
      return this._channels;
    }

    get volume() {
      return this._volume;
    }

    get muted() {
      return this._muted;
    }

    get underruns() {
      return Atomics.load(this._ctrl, HDR_UNDERRUNS);
    }

    get underrunFrames() {
      return Atomics.load(this._ctrl, HDR_UNDERRUN_FRAMES);
    }

    get overflows() {
      return Atomics.load(this._ctrl, HDR_OVERFLOWS);
    }

    get overruns() {
      return this.overflows;
    }

    get droppedFrames() {
      return Atomics.load(this._ctrl, HDR_DROPPED_FRAMES);
    }

    setVolume(volume) {
      this._volume = clampVolume(volume);
      return this._volume;
    }

    setMuted(muted) {
      this._muted = Boolean(muted);
      return this._muted;
    }

    effectiveGain() {
      return this._muted ? 0.0 : this._volume;
    }

    availableFrames() {
      const write = Atomics.load(this._ctrl, HDR_WRITE_CURSOR);
      const read = Atomics.load(this._ctrl, HDR_READ_CURSOR);
      let used = write - read;
      if (used < 0) {
        used += this._modulus;
      }
      if (used < 0) return 0;
      if (used > this._capacityFrames) return this._capacityFrames;
      return used;
    }

    freeFrames() {
      return this._capacityFrames - this.availableFrames();
    }

    /**
     * Pushes interleaved PCM float samples into the ring buffer.
     * Applies overflow backpressure: writes as many complete frames as fit,
     * never overwrites unread audio, and increments overflow counters on drop.
     *
     * @param {Float32Array|ArrayBuffer|SharedArrayBuffer|number[]} samples Interleaved PCM samples.
     * @returns {number} Number of complete frames written.
     */
    pushSamples(samples) {
      const pcm = toFloat32Array(samples);
      const channels = this._channels;
      const inputFrames = Math.floor(pcm.length / channels);
      if (inputFrames <= 0) {
        return 0;
      }

      const cap = this._capacityFrames;
      const write = Atomics.load(this._ctrl, HDR_WRITE_CURSOR);
      const read = Atomics.load(this._ctrl, HDR_READ_CURSOR);
      let used = write - read;
      if (used < 0) {
        used += this._modulus;
      }

      const free = Math.max(0, cap - used);
      const framesToWrite = inputFrames <= free ? inputFrames : free;

      if (framesToWrite > 0) {
        const writeSlot = write >= cap ? write - cap : write;
        const firstFrames = Math.min(framesToWrite, cap - writeSlot);
        const secondFrames = framesToWrite - firstFrames;

        const firstSampleCount = firstFrames * channels;
        this._samples.set(
          pcm.subarray(0, firstSampleCount),
          writeSlot * channels
        );

        if (secondFrames > 0) {
          this._samples.set(
            pcm.subarray(firstSampleCount, framesToWrite * channels),
            0
          );
        }

        let nextWrite = write + framesToWrite;
        if (nextWrite >= this._modulus) {
          nextWrite -= this._modulus;
        }
        Atomics.store(this._ctrl, HDR_WRITE_CURSOR, nextWrite);
      }

      if (framesToWrite < inputFrames) {
        Atomics.add(this._ctrl, HDR_OVERFLOWS, 1);
        Atomics.add(this._ctrl, HDR_DROPPED_FRAMES, inputFrames - framesToWrite);
      }

      return framesToWrite;
    }

    /**
     * Pulls interleaved PCM float samples out of the ring buffer into `output`.
     * Applies master volume / mute gain. On underrun, zero-fills remaining
     * frames and increments atomic underrun counters.
     *
     * @param {Float32Array} output Destination buffer (length = frames * channels).
     * @param {number} [gainOverride] Optional explicit gain override in [0, 1].
     * @returns {number} Number of real frames pulled before any zero-fill.
     */
    pullSamples(output, gainOverride) {
      if (!(output instanceof Float32Array)) {
        throw new TypeError('pullSamples target must be a Float32Array');
      }
      if (!isSharedArrayBuffer(output.buffer) && isDetachedArrayBuffer(output.buffer)) {
        throw new TypeError('pullSamples target buffer is detached');
      }

      const channels = this._channels;
      const requestedFrames = Math.floor(output.length / channels);
      if (requestedFrames <= 0) {
        return 0;
      }

      const cap = this._capacityFrames;
      const read = Atomics.load(this._ctrl, HDR_READ_CURSOR);
      const write = Atomics.load(this._ctrl, HDR_WRITE_CURSOR);
      let used = write - read;
      if (used < 0) {
        used += this._modulus;
      }

      const framesToRead = requestedFrames <= used ? requestedFrames : used;
      const gain =
        gainOverride !== undefined ? clampVolume(gainOverride) : this.effectiveGain();

      if (framesToRead > 0) {
        const readSlot = read >= cap ? read - cap : read;
        const firstFrames = Math.min(framesToRead, cap - readSlot);
        const secondFrames = framesToRead - firstFrames;
        const firstSamples = firstFrames * channels;
        const secondSamples = secondFrames * channels;

        if (gain === 0.0) {
          output.fill(0, 0, framesToRead * channels);
        } else if (gain === 1.0) {
          output.set(
            this._samples.subarray(readSlot * channels, readSlot * channels + firstSamples),
            0
          );
          if (secondSamples > 0) {
            output.set(this._samples.subarray(0, secondSamples), firstSamples);
          }
        } else {
          const srcBase = readSlot * channels;
          for (let i = 0; i < firstSamples; i++) {
            output[i] = this._samples[srcBase + i] * gain;
          }
          for (let i = 0; i < secondSamples; i++) {
            output[firstSamples + i] = this._samples[i] * gain;
          }
        }

        let nextRead = read + framesToRead;
        if (nextRead >= this._modulus) {
          nextRead -= this._modulus;
        }
        Atomics.store(this._ctrl, HDR_READ_CURSOR, nextRead);
      }

      if (framesToRead < requestedFrames) {
        output.fill(0, framesToRead * channels, requestedFrames * channels);
        Atomics.add(this._ctrl, HDR_UNDERRUNS, 1);
        Atomics.add(this._ctrl, HDR_UNDERRUN_FRAMES, requestedFrames - framesToRead);
      }

      return framesToRead;
    }

    /**
     * Atomically advances the read cursor by up to `maxFrames` available frames
     * without allocating or copying samples and without recording underruns.
     * Keeps guest audio producers unblocked when Web Audio is suspended or absent.
     *
     * @param {number} [maxFrames] Maximum frames to discard (defaults to all available).
     * @returns {number} Actual number of frames discarded.
     */
    discardFrames(maxFrames) {
      const cap = this._capacityFrames;
      if (cap <= 0) {
        return 0;
      }

      const read = Atomics.load(this._ctrl, HDR_READ_CURSOR);
      const write = Atomics.load(this._ctrl, HDR_WRITE_CURSOR);
      let used = write - read;
      if (used < 0) {
        used += this._modulus;
      }
      if (used <= 0) {
        return 0;
      }
      if (used > cap) {
        used = cap;
      }

      let toDiscard = used;
      if (maxFrames !== undefined && maxFrames !== null) {
        if (typeof maxFrames !== 'number' || !Number.isFinite(maxFrames) || maxFrames <= 0) {
          return 0;
        }
        const floored = Math.floor(maxFrames);
        if (floored <= 0) {
          return 0;
        }
        toDiscard = floored < used ? floored : used;
      }

      let nextRead = read + toDiscard;
      if (nextRead >= this._modulus) {
        nextRead -= this._modulus;
      }
      Atomics.store(this._ctrl, HDR_READ_CURSOR, nextRead);
      return toDiscard;
    }

    reset() {
      Atomics.store(this._ctrl, HDR_WRITE_CURSOR, 0);
      Atomics.store(this._ctrl, HDR_READ_CURSOR, 0);
      Atomics.store(this._ctrl, HDR_UNDERRUNS, 0);
      Atomics.store(this._ctrl, HDR_UNDERRUN_FRAMES, 0);
      Atomics.store(this._ctrl, HDR_OVERFLOWS, 0);
      Atomics.store(this._ctrl, HDR_DROPPED_FRAMES, 0);
    }

    getStats() {
      const overflows = this.overflows;
      return {
        capacityFrames: this._capacityFrames,
        channels: this._channels,
        availableFrames: this.availableFrames(),
        freeFrames: this.freeFrames(),
        volume: this._volume,
        muted: this._muted,
        effectiveGain: this.effectiveGain(),
        underruns: this.underruns,
        underrunFrames: this.underrunFrames,
        overflows,
        overruns: overflows,
        droppedFrames: this.droppedFrames,
      };
    }
  }

  /**
   * Audio mixer worker controller managing initialization, SAB ring buffer I/O,
   * bounded headless/autoplay-suspended draining, payload hardening, master
   * volume/mute state, and sanitized `audio-stats` telemetry.
   */
  class AudioMixerWorker {
    constructor(postMessageFn = null, options = {}) {
      this._postMessage =
        typeof postMessageFn === 'function' ? postMessageFn : () => {};
      const opts = options && typeof options === 'object' ? options : {};

      this._now = typeof opts.now === 'function' ? opts.now : () => Date.now();
      this._setInterval =
        typeof opts.setIntervalFn === 'function'
          ? opts.setIntervalFn
          : typeof setInterval === 'function'
            ? (fn, ms) => setInterval(fn, ms)
            : null;
      this._clearInterval =
        typeof opts.clearIntervalFn === 'function'
          ? opts.clearIntervalFn
          : typeof clearInterval === 'function'
            ? (id) => clearInterval(id)
            : null;

      this._maxPcmBytes = sanitizePositiveInt(
        opts.maxPcmBytes,
        DEFAULT_MAX_PCM_BYTES,
        Float32Array.BYTES_PER_ELEMENT,
        64 * 1024 * 1024
      );
      this._autoDrain = opts.autoDrain !== undefined ? Boolean(opts.autoDrain) : true;
      this._drainIntervalMs = sanitizePositiveInt(
        opts.drainIntervalMs,
        DEFAULT_DRAIN_INTERVAL_MS,
        MIN_DRAIN_INTERVAL_MS,
        MAX_DRAIN_INTERVAL_MS
      );
      this._explicitMaxDrainFramesPerTick =
        typeof opts.maxDrainFramesPerTick === 'number' &&
        Number.isFinite(opts.maxDrainFramesPerTick) &&
        opts.maxDrainFramesPerTick > 0
          ? Math.floor(opts.maxDrainFramesPerTick)
          : null;

      this._hasExplicitOutputState =
        opts.outputState !== undefined || opts.headless !== undefined;
      if (opts.outputState !== undefined) {
        this.outputState = normalizeOutputState(opts.outputState, OUTPUT_STATES.HEADLESS_DRAIN);
      } else if (opts.headless === true) {
        this.outputState = OUTPUT_STATES.HEADLESS_DRAIN;
      } else {
        this.outputState = detectDefaultOutputState();
      }

      this.initialized = false;
      this.sampleRate = DEFAULT_SAMPLE_RATE;
      this.channels = DEFAULT_CHANNELS;
      this.volume = 1.0;
      this.muted = false;
      this.ring = null;
      this.totalPushedFrames = 0;
      this.totalPulledFrames = 0;
      this.framesDrained = 0;
      this._drainTicks = 0;
      this._overruns = 0;
      this._rejectedPackets = 0;
      this._sabCopiedPackets = 0;
      this._sanitizedSamples = 0;
      this._drainTimer = null;
      this._lastDrainMs = null;
    }

    _safePostMessage(payload, transfer) {
      const safeTransfer = sanitizeTransferList(transfer);
      try {
        if (safeTransfer.length > 0) {
          this._postMessage(payload, safeTransfer);
        } else {
          this._postMessage(payload);
        }
      } catch (_err) {
        if (safeTransfer.length > 0) {
          try {
            this._postMessage(payload);
          } catch (_fallbackErr) {
            // Never allow postMessage errors to crash the audio worker.
          }
        }
      }
    }

    _defaultMaxDrainFramesPerTick() {
      return Math.max(128, Math.ceil((this.sampleRate * MAX_DRAIN_ELAPSED_MS) / 1000));
    }

    _effectiveMaxDrainFramesPerTick() {
      if (this._explicitMaxDrainFramesPerTick !== null) {
        return this._explicitMaxDrainFramesPerTick;
      }
      return this._defaultMaxDrainFramesPerTick();
    }

    _shouldDrain() {
      return (
        this.outputState === OUTPUT_STATES.HEADLESS_DRAIN ||
        this.outputState === OUTPUT_STATES.SUSPENDED
      );
    }

    _startDrainTimer() {
      this._stopDrainTimer();
      if (!this.initialized || !this.ring || !this._autoDrain || !this._shouldDrain()) {
        return;
      }
      if (!this._setInterval) {
        return;
      }
      this._lastDrainMs = this._now();
      const handle = this._setInterval(() => {
        try {
          this.drainTick();
        } catch (_err) {
          // Never throw from background drain timer.
        }
      }, this._drainIntervalMs);
      if (handle && typeof handle.unref === 'function') {
        handle.unref();
      }
      this._drainTimer = handle;
    }

    _stopDrainTimer() {
      if (this._drainTimer !== null && this._clearInterval) {
        try {
          this._clearInterval(this._drainTimer);
        } catch (_err) {
          // Ignore clearInterval failures during teardown.
        }
      }
      this._drainTimer = null;
    }

    setOutputState(state) {
      this.outputState = normalizeOutputState(state, OUTPUT_STATES.HEADLESS_DRAIN);
      if (this._shouldDrain()) {
        this._startDrainTimer();
      } else {
        this._stopDrainTimer();
      }
      return this.outputState;
    }

    /**
     * Executes one time-paced bounded drain step in headless or suspended mode.
     * Advances the SAB ring read cursor by at most the real-time frame budget
     * for the elapsed interval (capped by maxDrainFramesPerTick).
     *
     * @param {number} [nowMs] Optional monotonic timestamp in ms.
     * @returns {number} Number of frames drained.
     */
    drainTick(nowMs) {
      if (!this.ring || !this._shouldDrain()) {
        return 0;
      }
      const now =
        typeof nowMs === 'number' && Number.isFinite(nowMs) ? nowMs : this._now();
      let elapsedMs;
      if (this._lastDrainMs === null) {
        elapsedMs = this._drainIntervalMs;
      } else {
        elapsedMs = now - this._lastDrainMs;
        if (!Number.isFinite(elapsedMs) || elapsedMs < 0) {
          elapsedMs = this._drainIntervalMs;
        } else if (elapsedMs > MAX_DRAIN_ELAPSED_MS) {
          elapsedMs = MAX_DRAIN_ELAPSED_MS;
        }
      }
      this._lastDrainMs = now;

      let timeBudgetFrames = Math.floor((this.sampleRate * elapsedMs) / 1000);
      if (elapsedMs > 0 && timeBudgetFrames <= 0) {
        timeBudgetFrames = 1;
      }
      const maxFrames = Math.min(timeBudgetFrames, this._effectiveMaxDrainFramesPerTick());
      const drained = maxFrames > 0 ? this.ring.discardFrames(maxFrames) : 0;
      this._drainTicks += 1;
      this.framesDrained += drained;
      return drained;
    }

    /**
     * Drains up to `maxFrames` available frames from the SAB ring while in
     * headless-drain or suspended mode.
     *
     * @param {number} [maxFrames]
     * @param {{ force?: boolean }} [options]
     * @returns {number} Number of frames drained.
     */
    drain(maxFrames, options = {}) {
      if (!this.ring) {
        return 0;
      }
      const force = Boolean(options && options.force);
      if (!force && !this._shouldDrain()) {
        return 0;
      }

      const capBound = this._effectiveMaxDrainFramesPerTick();
      let limit;
      if (typeof maxFrames === 'number' && Number.isFinite(maxFrames) && maxFrames > 0) {
        const requested = Math.floor(maxFrames);
        const upperBound =
          this._explicitMaxDrainFramesPerTick !== null
            ? capBound
            : Math.max(capBound, this.ring.capacityFrames);
        limit = Math.min(requested, upperBound);
      } else {
        limit = capBound;
      }

      const drained = this.ring.discardFrames(limit);
      this._drainTicks += 1;
      this.framesDrained += drained;
      return drained;
    }

    stop() {
      this._stopDrainTimer();
    }

    destroy() {
      this._stopDrainTimer();
    }

    effectiveGain() {
      return this.muted ? 0.0 : this.volume;
    }

    pullSamples(output) {
      if (
        !(output instanceof Float32Array) ||
        (!isSharedArrayBuffer(output.buffer) && isDetachedArrayBuffer(output.buffer))
      ) {
        return 0;
      }
      if (!this.ring) {
        output.fill(0);
        return 0;
      }
      const framesRead = this.ring.pullSamples(output);
      this.totalPulledFrames += framesRead;
      return framesRead;
    }

    getStats() {
      const ringStats = this.ring
        ? this.ring.getStats()
        : {
            capacityFrames: 0,
            channels: this.channels,
            availableFrames: 0,
            freeFrames: 0,
            volume: this.volume,
            muted: this.muted,
            effectiveGain: this.effectiveGain(),
            underruns: 0,
            underrunFrames: 0,
            overflows: 0,
            overruns: 0,
            droppedFrames: 0,
          };

      const overruns = ringStats.overflows + this._overruns;

      return {
        initialized: this.initialized,
        outputState: this.outputState,
        sampleRate: this.sampleRate,
        channels: this.channels,
        capacityFrames: ringStats.capacityFrames,
        availableFrames: ringStats.availableFrames,
        freeFrames: ringStats.freeFrames,
        volume: this.volume,
        muted: this.muted,
        effectiveGain: this.effectiveGain(),
        framesDrained: this.framesDrained,
        underruns: ringStats.underruns,
        underrunFrames: ringStats.underrunFrames,
        overruns,
        overflows: ringStats.overflows,
        droppedFrames: ringStats.droppedFrames,
        totalPushedFrames: this.totalPushedFrames,
        totalPulledFrames: this.totalPulledFrames,
        drainTicks: this._drainTicks,
        rejectedPackets: this._rejectedPackets,
        sabCopiedPackets: this._sabCopiedPackets,
        sanitizedSamples: this._sanitizedSamples,
      };
    }

    getSanitizedStats() {
      return buildAudioTelemetrySnapshot(this, { now: this._now }).stats;
    }

    _resolveInitOutputState(msg) {
      if (typeof msg.outputState === 'string') {
        return normalizeOutputState(msg.outputState, OUTPUT_STATES.HEADLESS_DRAIN);
      }
      if (msg.headless === true || msg.audioAvailable === false || msg.workletAvailable === false) {
        return OUTPUT_STATES.HEADLESS_DRAIN;
      }
      if (typeof msg.audioContextState === 'string') {
        return normalizeOutputState(msg.audioContextState, OUTPUT_STATES.HEADLESS_DRAIN);
      }
      if (this._hasExplicitOutputState) {
        return this.outputState;
      }
      return detectDefaultOutputState();
    }

    handleMessage(msg) {
      if (!msg || typeof msg !== 'object' || Array.isArray(msg)) {
        this._rejectedPackets += 1;
        return;
      }

      let safeId;
      try {
        safeId = sanitizeMessageId(msg.id);
      } catch (_err) {
        this._rejectedPackets += 1;
        return;
      }

      try {
        if (msg.init) {
          this._stopDrainTimer();

          if (msg.autoDrain !== undefined) {
            this._autoDrain = Boolean(msg.autoDrain);
          }
          if (msg.drainIntervalMs !== undefined) {
            this._drainIntervalMs = sanitizePositiveInt(
              msg.drainIntervalMs,
              this._drainIntervalMs,
              MIN_DRAIN_INTERVAL_MS,
              MAX_DRAIN_INTERVAL_MS
            );
          }
          if (
            typeof msg.maxDrainFramesPerTick === 'number' &&
            Number.isFinite(msg.maxDrainFramesPerTick) &&
            msg.maxDrainFramesPerTick > 0
          ) {
            this._explicitMaxDrainFramesPerTick = Math.floor(msg.maxDrainFramesPerTick);
          }
          if (msg.maxPcmBytes !== undefined) {
            this._maxPcmBytes = sanitizePositiveInt(
              msg.maxPcmBytes,
              this._maxPcmBytes,
              Float32Array.BYTES_PER_ELEMENT,
              64 * 1024 * 1024
            );
          }

          const nextSampleRate = sanitizeSampleRate(msg.sampleRate);
          const nextChannels = sanitizePositiveInt(
            msg.channels,
            DEFAULT_CHANNELS,
            1,
            MAX_CHANNELS
          );

          let nextRing;
          if (typeof SharedArrayBuffer === 'undefined') {
            this._rejectedPackets += 1;
            const errReply = {
              ready: false,
              outputState: OUTPUT_STATES.HEADLESS_DRAIN,
              error: 'SharedArrayBuffer is unavailable',
            };
            if (safeId !== undefined) errReply.id = safeId;
            this._safePostMessage(errReply);
            return;
          }
          if (msg.ringSab !== undefined && msg.ringSab !== null) {
            if (!isSharedArrayBuffer(msg.ringSab)) {
              this._rejectedPackets += 1;
              const errReply = {
                ready: false,
                error: 'Invalid ringSab: expected SharedArrayBuffer',
              };
              if (safeId !== undefined) errReply.id = safeId;
              this._safePostMessage(errReply);
              return;
            }
            nextRing = new AudioRingBuffer(msg.ringSab, {
              channels: nextChannels,
              volume: this.volume,
              muted: this.muted,
            });
          } else {
            const cap = sanitizePositiveInt(
              msg.capacityFrames,
              DEFAULT_CAPACITY_FRAMES,
              1,
              MAX_CAPACITY_FRAMES
            );
            nextRing = AudioRingBuffer.create(cap, nextChannels);
            nextRing.setVolume(this.volume);
            nextRing.setMuted(this.muted);
          }

          this.sampleRate = nextSampleRate;
          this.ring = nextRing;
          this.channels = nextRing.channels;
          this.outputState = this._resolveInitOutputState(msg);
          this.initialized = true;
          this.totalPushedFrames = 0;
          this.totalPulledFrames = 0;
          this.framesDrained = 0;
          this._drainTicks = 0;
          this._overruns = 0;
          this._rejectedPackets = 0;
          this._sabCopiedPackets = 0;
          this._sanitizedSamples = 0;
          this._lastDrainMs = null;

          if (this._shouldDrain()) {
            this._startDrainTimer();
          }

          this._safePostMessage({
            ready: true,
            sampleRate: this.sampleRate,
            channels: this.channels,
            capacityFrames: this.ring.capacityFrames,
          });
          return;
        }

        if (msg.pushPcm) {
          if (!this.ring) {
            this._rejectedPackets += 1;
            this._safePostMessage({
              id: safeId,
              error: 'AudioMixerWorker not initialized',
            });
            return;
          }

          const extracted = extractSafePcmSamples(msg.samples, {
            maxPcmBytes: this._maxPcmBytes,
          });
          if (!extracted.ok) {
            this._rejectedPackets += 1;
            if (safeId !== undefined) {
              this._safePostMessage({
                id: safeId,
                error: extracted.error,
              });
            }
            return;
          }

          if (extracted.copiedFromSab) {
            this._sabCopiedPackets += 1;
          }
          if (extracted.sanitizedSamples > 0) {
            this._sanitizedSamples += extracted.sanitizedSamples;
          }

          const pcm = extracted.pcm;
          const inputFrames = Math.floor(pcm.length / this.channels);

          // In headless-drain or autoplay-suspended mode, drain oldest frames at
          // bounded rate when ring pressure builds so guest audio producers never
          // deadlock waiting for free ring space.
          if (inputFrames > 0 && this._shouldDrain()) {
            const free = this.ring.freeFrames();
            if (inputFrames > free) {
              const deficit = inputFrames - free;
              const drainCap = this._effectiveMaxDrainFramesPerTick();
              const drainBudget =
                this._explicitMaxDrainFramesPerTick !== null
                  ? Math.min(deficit, drainCap)
                  : Math.min(deficit, Math.max(drainCap, this.ring.capacityFrames));
              const preDrained = this.ring.discardFrames(drainBudget);
              if (preDrained > 0) {
                this.framesDrained += preDrained;
              }
            }
          }

          const pushedFrames = inputFrames > 0 ? this.ring.pushSamples(pcm) : 0;
          const droppedFrames = inputFrames - pushedFrames;
          this.totalPushedFrames += pushedFrames;

          if (safeId !== undefined) {
            this._safePostMessage({
              id: safeId,
              pushedFrames,
              droppedFrames,
              availableFrames: this.ring.availableFrames(),
            });
          }
          return;
        }

        if (msg.drain || msg.drainRing) {
          const drainedFrames = this.drain(msg.maxFrames, {
            force: Boolean(msg.force),
          });
          if (safeId !== undefined) {
            this._safePostMessage({
              id: safeId,
              drainedFrames,
              framesDrained: this.framesDrained,
              outputState: this.outputState,
              availableFrames: this.ring ? this.ring.availableFrames() : 0,
            });
          }
          return;
        }

        if (msg.setOutputState || msg.type === 'output-state' || msg.type === 'audio-context-state') {
          const nextState = this.setOutputState(
            msg.outputState !== undefined
              ? msg.outputState
              : msg.state !== undefined
                ? msg.state
                : msg.audioContextState
          );
          if (safeId !== undefined) {
            this._safePostMessage({
              id: safeId,
              outputState: nextState,
            });
          }
          return;
        }

        if (msg.setVolume) {
          if (typeof msg.volume === 'number' && Number.isFinite(msg.volume)) {
            this.volume = clampVolume(msg.volume);
          }
          if (typeof msg.muted === 'boolean') {
            this.muted = msg.muted;
          }
          if (this.ring) {
            this.ring.setVolume(this.volume);
            this.ring.setMuted(this.muted);
          }
          if (safeId !== undefined) {
            this._safePostMessage({
              id: safeId,
              volume: this.volume,
              muted: this.muted,
              effectiveGain: this.effectiveGain(),
            });
          }
          return;
        }

        if (msg.setMute) {
          this.muted = Boolean(msg.muted);
          if (this.ring) {
            this.ring.setMuted(this.muted);
          }
          if (safeId !== undefined) {
            this._safePostMessage({
              id: safeId,
              volume: this.volume,
              muted: this.muted,
              effectiveGain: this.effectiveGain(),
            });
          }
          return;
        }

        if (msg.stop || msg.destroy || msg.type === 'stop') {
          this.stop();
          if (safeId !== undefined) {
            this._safePostMessage({
              id: safeId,
              stopped: true,
              outputState: this.outputState,
            });
          }
          return;
        }

        if (
          msg.stats ||
          msg.type === 'audio-stats' ||
          msg.type === 'request-audio-stats'
        ) {
          const snapshot = buildAudioTelemetrySnapshot(this, { now: this._now });
          const reply = {
            type: 'audio-stats',
            id: safeId,
            stats: snapshot.stats,
            timestamp: snapshot.timestamp,
          };
          this._safePostMessage(reply);
          return;
        }
      } catch (err) {
        this._rejectedPackets += 1;
        if (safeId !== undefined) {
          this._safePostMessage({
            id: safeId,
            error: sanitizeErrorMessage(err),
          });
        }
      }
    }
  }

  const exportsObj = {
    AudioRingBuffer,
    AudioMixerWorker,
    OUTPUT_STATES,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_CHANNELS,
    DEFAULT_CAPACITY_FRAMES,
    MAX_CAPACITY_FRAMES,
    MAX_CHANNELS,
    DEFAULT_DRAIN_INTERVAL_MS,
    DEFAULT_MAX_PCM_BYTES,
    RING_HEADER_BYTES,
    extractSafePcmSamples,
    clonePcmForTransfer,
    sanitizeTransferList,
    buildAudioTelemetrySnapshot,
  };

  // UMD-style exports for browser globals and Node.js tests.
  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6Audio = exportsObj;
  } else if (root) {
    root.PlayGta6Audio = exportsObj;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportsObj;
  }

  // Classic Dedicated Worker execution path.
  const isDedicatedWorker =
    typeof self !== 'undefined' &&
    typeof self.postMessage === 'function' &&
    typeof window === 'undefined';

  if (isDedicatedWorker) {
    const worker = new AudioMixerWorker((payload, transfer) => {
      const safeTransfer = sanitizeTransferList(transfer);
      if (safeTransfer.length > 0) {
        self.postMessage(payload, safeTransfer);
      } else {
        self.postMessage(payload);
      }
    });
    self.playGta6AudioWorker = worker;
    self.onmessage = (ev) => {
      try {
        worker.handleMessage(ev ? ev.data : null);
      } catch (_err) {
        // Never allow an uncaught worker error to disrupt game boot.
      }
    };
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
