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

  // Underrun de-click micro-fade ramp defaults (32 frames ~= 0.67 ms @ 48 kHz).
  const DEFAULT_DECLICK_FRAMES = 32;
  const MAX_DECLICK_FRAMES = 4096;

  // Multi-channel ITU-R BS.775 downmix, soft-clip limiter, and resampling safeguards.
  const DEFAULT_SOFT_CLIP_KNEE = 0.95;
  const ITU_BS775_CENTER_GAIN = Math.SQRT1_2;
  const ITU_BS775_SURROUND_GAIN = Math.SQRT1_2;
  const ITU_BS775_LFE_GAIN = 0.0;
  const MAX_RESAMPLE_RATIO = 16;
  const DEFAULT_MAX_RESAMPLE_FRAMES = 524288;
  const SUPPORTED_DOWNMIX_CHANNELS = Object.freeze([1, 2, 4, 6, 8]);

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

  function sanitizeDeclickFrames(value, fallback) {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      return fallback;
    }
    const n = Math.floor(value);
    if (n < 0) return fallback;
    if (n > MAX_DECLICK_FRAMES) return MAX_DECLICK_FRAMES;
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

  function clampSample(value, min, max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
  }

  /**
   * Smooth C1-continuous soft-clip peak limiter backed by Math.tanh.
   * Non-finite inputs (NaN / +-Infinity) are sanitized to 0.0.
   *
   * @param {number} x
   * @param {number} [knee=0.95]
   * @returns {number}
   */
  function softClipSample(x, knee = DEFAULT_SOFT_CLIP_KNEE) {
    if (typeof knee !== 'number' || !Number.isFinite(knee) || knee < 0 || knee >= 1.0) {
      throw new RangeError(`knee must be in [0, 1), got ${knee}`);
    }
    if (!Number.isFinite(x)) {
      return 0.0;
    }
    const absX = Math.abs(x);
    if (absX <= knee) {
      return x;
    }
    const headroom = 1.0 - knee;
    const compressed = knee + headroom * Math.tanh((absX - knee) / headroom);
    const bounded = compressed > 1.0 ? 1.0 : compressed;
    return x < 0 ? -bounded : bounded;
  }

  function validateDownmixGain(value, fallback, name) {
    if (value === undefined) {
      return fallback;
    }
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
      throw new RangeError(`${name} must be a non-negative finite number, got ${value}`);
    }
    return value;
  }

  /**
   * ITU-R BS.775 multi-channel (1, 2, 4, 5.1 [6ch], 7.1 [8ch]) to stereo downmix
   * with per-channel non-finite sanitization and smooth soft-clip peak limiting.
   *
   * @param {Float32Array|ArrayBuffer|SharedArrayBuffer|ArrayBufferView|number[]} samples
   * @param {number} channels Source channel count (1, 2, 4, 6, or 8).
   * @param {object} [options]
   * @returns {Float32Array} Interleaved stereo Float32Array.
   */
  function downmixToStereo(samples, channels, options = {}) {
    if (!Number.isInteger(channels) || !SUPPORTED_DOWNMIX_CHANNELS.includes(channels)) {
      throw new RangeError(
        `downmixToStereo channels must be one of ${SUPPORTED_DOWNMIX_CHANNELS.join(', ')}, got ${channels}`
      );
    }

    const opts = options && typeof options === 'object' ? options : {};
    const centerGain = validateDownmixGain(
      opts.centerGain !== undefined ? opts.centerGain : opts.centerWeight,
      ITU_BS775_CENTER_GAIN,
      'centerGain'
    );
    const surroundGain = validateDownmixGain(
      opts.surroundGain !== undefined ? opts.surroundGain : opts.surroundWeight,
      ITU_BS775_SURROUND_GAIN,
      'surroundGain'
    );
    const lfeGain = validateDownmixGain(
      opts.lfeGain !== undefined ? opts.lfeGain : opts.lfeWeight,
      ITU_BS775_LFE_GAIN,
      'lfeGain'
    );
    const softClip = opts.softClip !== undefined ? Boolean(opts.softClip) : true;
    const knee =
      opts.limiterKnee !== undefined
        ? opts.limiterKnee
        : opts.knee !== undefined
          ? opts.knee
          : DEFAULT_SOFT_CLIP_KNEE;
    if (typeof knee !== 'number' || !Number.isFinite(knee) || knee < 0 || knee >= 1.0) {
      throw new RangeError(`limiterKnee must be in [0, 1), got ${knee}`);
    }

    let src;
    let nanSanitized = 0;
    if (samples instanceof Float32Array && !isSharedArrayBuffer(samples.buffer)) {
      if (isDetachedArrayBuffer(samples.buffer)) {
        throw new TypeError('PCM TypedArray backing ArrayBuffer is detached');
      }
      src = samples;
    } else {
      const extracted = extractSafePcmSamples(samples, { maxPcmBytes: 64 * 1024 * 1024 });
      if (!extracted.ok) {
        throw new TypeError(extracted.error);
      }
      src = extracted.pcm;
      nanSanitized += extracted.sanitizedSamples;
    }

    const frameCount = Math.floor(src.length / channels);
    const out = new Float32Array(frameCount * 2);
    let clipped = 0;
    let peak = 0.0;

    for (let f = 0; f < frameCount; f++) {
      const base = f * channels;
      let rawL;
      let rawR;

      if (channels === 6) {
        // 5.1: FL, FR, FC, LFE, BL, BR
        let fl = src[base];
        let fr = src[base + 1];
        let fc = src[base + 2];
        let lfe = src[base + 3];
        let bl = src[base + 4];
        let br = src[base + 5];

        if (!Number.isFinite(fl)) { fl = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(fr)) { fr = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(fc)) { fc = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(lfe)) { lfe = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(bl)) { bl = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(br)) { br = 0.0; nanSanitized += 1; }

        rawL = fl + fc * centerGain + bl * surroundGain + lfe * lfeGain;
        rawR = fr + fc * centerGain + br * surroundGain + lfe * lfeGain;
      } else if (channels === 8) {
        // 7.1: FL, FR, FC, LFE, BL, BR, SL, SR
        let fl = src[base];
        let fr = src[base + 1];
        let fc = src[base + 2];
        let lfe = src[base + 3];
        let bl = src[base + 4];
        let br = src[base + 5];
        let sl = src[base + 6];
        let sr = src[base + 7];

        if (!Number.isFinite(fl)) { fl = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(fr)) { fr = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(fc)) { fc = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(lfe)) { lfe = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(bl)) { bl = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(br)) { br = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(sl)) { sl = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(sr)) { sr = 0.0; nanSanitized += 1; }

        rawL = fl + fc * centerGain + (bl + sl) * surroundGain + lfe * lfeGain;
        rawR = fr + fc * centerGain + (br + sr) * surroundGain + lfe * lfeGain;
      } else if (channels === 4) {
        // Quad: FL, FR, BL, BR
        let fl = src[base];
        let fr = src[base + 1];
        let bl = src[base + 2];
        let br = src[base + 3];

        if (!Number.isFinite(fl)) { fl = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(fr)) { fr = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(bl)) { bl = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(br)) { br = 0.0; nanSanitized += 1; }

        rawL = fl + bl * surroundGain;
        rawR = fr + br * surroundGain;
      } else if (channels === 2) {
        let fl = src[base];
        let fr = src[base + 1];
        if (!Number.isFinite(fl)) { fl = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(fr)) { fr = 0.0; nanSanitized += 1; }
        rawL = fl;
        rawR = fr;
      } else {
        let m = src[base];
        if (!Number.isFinite(m)) { m = 0.0; nanSanitized += 1; }
        rawL = m;
        rawR = m;
      }

      if (!Number.isFinite(rawL)) { rawL = 0.0; nanSanitized += 1; }
      if (!Number.isFinite(rawR)) { rawR = 0.0; nanSanitized += 1; }

      let outL;
      let outR;
      if (softClip) {
        if (rawL > knee || rawL < -knee) clipped += 1;
        if (rawR > knee || rawR < -knee) clipped += 1;
        outL = softClipSample(rawL, knee);
        outR = softClipSample(rawR, knee);
      } else {
        if (rawL > 1.0 || rawL < -1.0) clipped += 1;
        if (rawR > 1.0 || rawR < -1.0) clipped += 1;
        outL = clampSample(rawL, -1.0, 1.0);
        outR = clampSample(rawR, -1.0, 1.0);
      }

      const absL = Math.abs(outL);
      const absR = Math.abs(outR);
      if (absL > peak) peak = absL;
      if (absR > peak) peak = absR;

      const outIdx = f * 2;
      out[outIdx] = outL;
      out[outIdx + 1] = outR;
    }

    if (opts.stats && typeof opts.stats === 'object') {
      const dmFrames = channels > 2 ? frameCount : 0;
      opts.stats.downmixedFrames = (opts.stats.downmixedFrames || 0) + dmFrames;
      opts.stats.clippedSamples = (opts.stats.clippedSamples || 0) + clipped;
      opts.stats.nanSanitizedSamples = (opts.stats.nanSanitizedSamples || 0) + nanSanitized;
      opts.stats.peakLevel = Math.max(opts.stats.peakLevel || 0.0, peak);
    }

    return out;
  }

  /**
   * Smooth linear PCM resampler with safeguards for 44.1 kHz <-> 48 kHz and
   * general Bink2 / XAudio2 sample-rate conversions.
   *
   * @param {Float32Array|ArrayBuffer|SharedArrayBuffer|ArrayBufferView|number[]} samples
   * @param {number} channels Interleaved channel count (1..8).
   * @param {number} srcSampleRate Source sample rate in Hz.
   * @param {number} dstSampleRate Destination sample rate in Hz.
   * @param {object} [options]
   * @returns {Float32Array}
   */
  function resampleLinear(samples, channels, srcSampleRate, dstSampleRate, options = {}) {
    if (!Number.isInteger(channels) || channels < 1 || channels > MAX_CHANNELS) {
      throw new RangeError(
        `resampleLinear channels must be an integer in [1, ${MAX_CHANNELS}], got ${channels}`
      );
    }
    if (
      typeof srcSampleRate !== 'number' ||
      !Number.isFinite(srcSampleRate) ||
      srcSampleRate < MIN_SAMPLE_RATE ||
      srcSampleRate > MAX_SAMPLE_RATE
    ) {
      throw new RangeError(
        `srcSampleRate must be in [${MIN_SAMPLE_RATE}, ${MAX_SAMPLE_RATE}], got ${srcSampleRate}`
      );
    }
    if (
      typeof dstSampleRate !== 'number' ||
      !Number.isFinite(dstSampleRate) ||
      dstSampleRate < MIN_SAMPLE_RATE ||
      dstSampleRate > MAX_SAMPLE_RATE
    ) {
      throw new RangeError(
        `dstSampleRate must be in [${MIN_SAMPLE_RATE}, ${MAX_SAMPLE_RATE}], got ${dstSampleRate}`
      );
    }

    const ratio = dstSampleRate / srcSampleRate;
    if (
      !Number.isFinite(ratio) ||
      ratio <= 0 ||
      ratio > MAX_RESAMPLE_RATIO ||
      1.0 / ratio > MAX_RESAMPLE_RATIO
    ) {
      throw new RangeError(
        `Resample ratio (${srcSampleRate} -> ${dstSampleRate}) exceeds max guard (${MAX_RESAMPLE_RATIO}x)`
      );
    }

    const opts = options && typeof options === 'object' ? options : {};
    const maxOutputFrames =
      typeof opts.maxOutputFrames === 'number' &&
      Number.isFinite(opts.maxOutputFrames) &&
      opts.maxOutputFrames > 0
        ? Math.floor(opts.maxOutputFrames)
        : DEFAULT_MAX_RESAMPLE_FRAMES;
    const softClip = Boolean(opts.softClip);
    const knee =
      opts.limiterKnee !== undefined
        ? opts.limiterKnee
        : opts.knee !== undefined
          ? opts.knee
          : DEFAULT_SOFT_CLIP_KNEE;
    if (softClip && (typeof knee !== 'number' || !Number.isFinite(knee) || knee < 0 || knee >= 1.0)) {
      throw new RangeError(`limiterKnee must be in [0, 1), got ${knee}`);
    }

    let src;
    let nanSanitized = 0;
    if (samples instanceof Float32Array && !isSharedArrayBuffer(samples.buffer)) {
      if (isDetachedArrayBuffer(samples.buffer)) {
        throw new TypeError('PCM TypedArray backing ArrayBuffer is detached');
      }
      src = samples;
    } else {
      const extracted = extractSafePcmSamples(samples, { maxPcmBytes: 64 * 1024 * 1024 });
      if (!extracted.ok) {
        throw new TypeError(extracted.error);
      }
      src = extracted.pcm;
      nanSanitized += extracted.sanitizedSamples;
    }

    const srcFrames = Math.floor(src.length / channels);
    if (srcFrames <= 0) {
      return new Float32Array(0);
    }

    const state = opts.state && typeof opts.state === 'object' ? opts.state : null;
    const EPS = 1e-12;
    let clipped = 0;

    const startPhase =
      state && typeof state.phase === 'number' && Number.isFinite(state.phase) && state.phase >= 0
        ? state.phase
        : 0.0;

    if (srcSampleRate === dstSampleRate && startPhase < EPS) {
      if (srcFrames > maxOutputFrames) {
        throw new RangeError(
          `Resampled output (${srcFrames} frames) exceeds maxOutputFrames (${maxOutputFrames})`
        );
      }
      const totalSamples = srcFrames * channels;
      const out = new Float32Array(totalSamples);
      for (let i = 0; i < totalSamples; i++) {
        let v = src[i];
        if (!Number.isFinite(v)) {
          v = 0.0;
          nanSanitized += 1;
        }
        if (softClip) {
          if (v > knee || v < -knee) clipped += 1;
          v = softClipSample(v, knee);
        }
        out[i] = v;
      }
      if (state) {
        state.phase = 0.0;
      }
      if (opts.stats && typeof opts.stats === 'object') {
        opts.stats.nanSanitizedSamples = (opts.stats.nanSanitizedSamples || 0) + nanSanitized;
        opts.stats.clippedSamples = (opts.stats.clippedSamples || 0) + clipped;
      }
      return out;
    }

    const step = srcSampleRate / dstSampleRate;
    if (srcFrames - startPhase <= EPS) {
      if (state) {
        const rem = startPhase - srcFrames;
        state.phase = rem > EPS ? rem : 0.0;
      }
      return new Float32Array(0);
    }

    const outFrames = Math.floor((srcFrames - startPhase - EPS) / step) + 1;
    if (outFrames > maxOutputFrames) {
      throw new RangeError(
        `Resampled output (${outFrames} frames) exceeds maxOutputFrames (${maxOutputFrames})`
      );
    }

    const out = new Float32Array(outFrames * channels);
    const lastFrame = srcFrames - 1;

    for (let f = 0; f < outFrames; f++) {
      const pos = startPhase + f * step;
      let i0 = Math.floor(pos);
      if (i0 < 0) i0 = 0;
      else if (i0 > lastFrame) i0 = lastFrame;
      const frac = pos - i0;
      const i1 = i0 < lastFrame ? i0 + 1 : lastFrame;
      const base0 = i0 * channels;
      const base1 = i1 * channels;
      const outBase = f * channels;

      for (let c = 0; c < channels; c++) {
        let s0 = src[base0 + c];
        if (!Number.isFinite(s0)) {
          s0 = 0.0;
          nanSanitized += 1;
        }
        let val;
        if (frac <= 0 || i1 === i0) {
          val = s0;
        } else {
          let s1 = src[base1 + c];
          if (!Number.isFinite(s1)) {
            s1 = 0.0;
            nanSanitized += 1;
          }
          val = s0 + (s1 - s0) * frac;
          if (!Number.isFinite(val)) {
            val = 0.0;
            nanSanitized += 1;
          }
        }
        if (softClip) {
          if (val > knee || val < -knee) clipped += 1;
          val = softClipSample(val, knee);
        }
        out[outBase + c] = val;
      }
    }

    if (state) {
      let nextPhase = startPhase + outFrames * step - srcFrames;
      if (!Number.isFinite(nextPhase) || nextPhase < EPS) {
        nextPhase = 0.0;
      }
      state.phase = nextPhase;
    }

    if (opts.stats && typeof opts.stats === 'object') {
      opts.stats.resampledFrames = (opts.stats.resampledFrames || 0) + outFrames;
      opts.stats.nanSanitizedSamples = (opts.stats.nanSanitizedSamples || 0) + nanSanitized;
      opts.stats.clippedSamples = (opts.stats.clippedSamples || 0) + clipped;
    }

    return out;
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
    const underruns = sanitizeNonNegInt(
      s.underruns !== undefined ? s.underruns : s.underrunCount,
      0
    );
    const underrunCount = sanitizeNonNegInt(
      s.underrunCount !== undefined ? s.underrunCount : underruns,
      underruns
    );
    const underrunFrames = sanitizeNonNegInt(s.underrunFrames, 0);
    const fadeOutCount = sanitizeNonNegInt(s.fadeOutCount, 0);
    const fadeInCount = sanitizeNonNegInt(s.fadeInCount, 0);
    const rawPeakFill = sanitizeNonNegInt(s.peakFillFrames, 0);
    const peakFillFrames =
      capacityFrames > 0 ? Math.min(rawPeakFill, capacityFrames) : rawPeakFill;
    const rawMinFill = sanitizeNonNegInt(s.minFillFrames, 0);
    const minFillFrames =
      capacityFrames > 0 ? Math.min(rawMinFill, capacityFrames) : rawMinFill;
    const rawAvgFillRatio =
      typeof s.avgFillRatio === 'number' && Number.isFinite(s.avgFillRatio)
        ? s.avgFillRatio
        : 0.0;
    const avgFillRatio =
      rawAvgFillRatio < 0 ? 0.0 : rawAvgFillRatio > 1 ? 1.0 : rawAvgFillRatio;
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
    const sanitizedSamples = sanitizeNonNegInt(
      s.sanitizedSamples !== undefined
        ? s.sanitizedSamples
        : s.nanSanitizedSamples !== undefined
          ? s.nanSanitizedSamples
          : s.sanitizedNonFiniteSamples,
      0
    );
    const nanSanitizedSamples = sanitizeNonNegInt(
      s.nanSanitizedSamples !== undefined ? s.nanSanitizedSamples : sanitizedSamples,
      sanitizedSamples
    );
    const downmixedFrames = sanitizeNonNegInt(s.downmixedFrames, 0);
    const resampledFrames = sanitizeNonNegInt(s.resampledFrames, 0);
    const clippedSamples = sanitizeNonNegInt(s.clippedSamples, 0);
    const activeVoicePeak = sanitizeNonNegInt(
      s.activeVoicePeak !== undefined ? s.activeVoicePeak : s.activeVoices,
      0
    );
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
        underrunCount,
        underrunFrames,
        fadeOutCount,
        fadeInCount,
        peakFillFrames,
        minFillFrames,
        avgFillRatio,
        overruns,
        overflows,
        droppedFrames,
        totalPushedFrames,
        totalPulledFrames,
        drainTicks,
        rejectedPackets,
        sabCopiedPackets,
        sanitizedSamples,
        nanSanitizedSamples,
        downmixedFrames,
        resampledFrames,
        clippedSamples,
        activeVoicePeak,
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

    static create(
      capacityFrames = DEFAULT_CAPACITY_FRAMES,
      channels = DEFAULT_CHANNELS,
      options = {}
    ) {
      const byteLen = AudioRingBuffer.byteLength(capacityFrames, channels);
      const sab = new SharedArrayBuffer(byteLen);
      const opts = options && typeof options === 'object' ? options : {};
      return new AudioRingBuffer(sab, {
        ...opts,
        capacityFrames,
        channels,
        initialize: true,
      });
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
      this._declickFrames = sanitizeDeclickFrames(options.declickFrames, 0);
      this._lastOutputFrame = new Float32Array(this._channels);
      this._hasActiveAudio = false;
      this._inUnderrun = false;
      this._fadeInPos = 0;
      this._fadeInTotal = 0;
      this._fadeOutCount = 0;
      this._fadeInCount = 0;
      this._peakFillFrames = 0;
      this._minFillFrames = Infinity;
      this._fillFramesSum = 0;
      this._fillSamples = 0;
      this._lastObservedFill = null;
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

    get declickFrames() {
      return this._declickFrames;
    }

    get underruns() {
      return Atomics.load(this._ctrl, HDR_UNDERRUNS);
    }

    get underrunCount() {
      return this.underruns;
    }

    get underrunFrames() {
      return Atomics.load(this._ctrl, HDR_UNDERRUN_FRAMES);
    }

    get fadeOutCount() {
      return this._fadeOutCount;
    }

    get fadeInCount() {
      return this._fadeInCount;
    }

    get peakFillFrames() {
      return this._fillSamples > 0 ? this._peakFillFrames : 0;
    }

    get minFillFrames() {
      return this._fillSamples > 0 ? this._minFillFrames : 0;
    }

    get avgFillRatio() {
      if (this._fillSamples <= 0 || this._capacityFrames <= 0) {
        return 0.0;
      }
      const ratio = this._fillFramesSum / this._fillSamples / this._capacityFrames;
      if (ratio < 0) return 0.0;
      if (ratio > 1) return 1.0;
      return ratio;
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

    setDeclickFrames(frames) {
      this._declickFrames = sanitizeDeclickFrames(frames, this._declickFrames);
      return this._declickFrames;
    }

    _recordFillSample(fillFrames) {
      const clamped =
        fillFrames < 0
          ? 0
          : fillFrames > this._capacityFrames
            ? this._capacityFrames
            : fillFrames;
      if (this._fillSamples === 0) {
        this._peakFillFrames = clamped;
        this._minFillFrames = clamped;
      } else {
        if (clamped > this._peakFillFrames) this._peakFillFrames = clamped;
        if (clamped < this._minFillFrames) this._minFillFrames = clamped;
      }
      this._fillFramesSum += clamped;
      this._fillSamples += 1;
      this._lastObservedFill = clamped;
    }

    _recordFillTransition(beforeFill, afterFill) {
      if (beforeFill !== afterFill && this._lastObservedFill !== beforeFill) {
        this._recordFillSample(beforeFill);
      }
      this._recordFillSample(afterFill);
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

      this._recordFillSample(used + framesToWrite);

      if (framesToWrite < inputFrames) {
        Atomics.add(this._ctrl, HDR_OVERFLOWS, 1);
        Atomics.add(this._ctrl, HDR_DROPPED_FRAMES, inputFrames - framesToWrite);
      }

      return framesToWrite;
    }

    /**
     * Pulls interleaved PCM float samples out of the ring buffer into `output`.
     * Applies master volume / mute gain. On underrun, applies a smooth linear
     * de-click fade-out when `declickFrames > 0` (or zero-fills when disabled)
     * and increments atomic underrun counters. On recovery after underrun,
     * applies a matching linear fade-in. Steady-state pulls with no underrun
     * remain bit-exact unchanged.
     *
     * @param {Float32Array} output Destination buffer (length = frames * channels).
     * @param {number} [gainOverride] Optional explicit gain override in [0, 1].
     * @returns {number} Number of real frames pulled before any underrun fill.
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
      if (used > cap) {
        used = cap;
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

      const afterFill = Math.max(0, used - framesToRead);
      this._recordFillTransition(used, afterFill);

      const declickFrames = this._declickFrames;

      // Apply smooth linear fade-in when valid audio resumes after an underrun.
      if (framesToRead > 0 && declickFrames > 0) {
        if (this._inUnderrun) {
          this._inUnderrun = false;
          this._fadeInPos = 0;
          this._fadeInTotal = declickFrames;
          this._fadeInCount += 1;
        }
        if (this._fadeInTotal > 0 && this._fadeInPos < this._fadeInTotal) {
          const remaining = this._fadeInTotal - this._fadeInPos;
          const rampCount = framesToRead < remaining ? framesToRead : remaining;
          const total = this._fadeInTotal;
          for (let f = 0; f < rampCount; f++) {
            const rampGain = (this._fadeInPos + f) / total;
            const base = f * channels;
            for (let c = 0; c < channels; c++) {
              output[base + c] *= rampGain;
            }
          }
          this._fadeInPos += rampCount;
          if (this._fadeInPos >= this._fadeInTotal) {
            this._fadeInPos = 0;
            this._fadeInTotal = 0;
          }
        }
      }

      if (framesToRead < requestedFrames) {
        const missingFrames = requestedFrames - framesToRead;

        if (declickFrames > 0) {
          // Entering underrun cancels any partial recovery fade-in.
          this._fadeInPos = 0;
          this._fadeInTotal = 0;

          if (framesToRead > 0 || this._hasActiveAudio) {
            if (
              framesToRead > 0 &&
              missingFrames < declickFrames &&
              framesToRead > missingFrames
            ) {
              // Fade out the last available frames toward 0.0, then zero-fill missing tail.
              const fadeFrames =
                declickFrames < framesToRead ? declickFrames : framesToRead;
              const fadeStart = framesToRead - fadeFrames;
              for (let i = 0; i < fadeFrames; i++) {
                const rampGain = (fadeFrames - 1 - i) / fadeFrames;
                const base = (fadeStart + i) * channels;
                for (let c = 0; c < channels; c++) {
                  output[base + c] *= rampGain;
                }
              }
              output.fill(0, framesToRead * channels, requestedFrames * channels);
            } else {
              // Ramp smoothly from the last output sample toward 0.0 over the missing tail.
              if (framesToRead > 0) {
                const lastBase = (framesToRead - 1) * channels;
                for (let c = 0; c < channels; c++) {
                  this._lastOutputFrame[c] = output[lastBase + c];
                }
              }
              const fadeFrames =
                declickFrames < missingFrames ? declickFrames : missingFrames;
              for (let i = 0; i < fadeFrames; i++) {
                const rampGain = (fadeFrames - 1 - i) / fadeFrames;
                const base = (framesToRead + i) * channels;
                for (let c = 0; c < channels; c++) {
                  output[base + c] = this._lastOutputFrame[c] * rampGain;
                }
              }
              if (framesToRead + fadeFrames < requestedFrames) {
                output.fill(
                  0,
                  (framesToRead + fadeFrames) * channels,
                  requestedFrames * channels
                );
              }
            }

            this._fadeOutCount += 1;
            this._hasActiveAudio = false;
            this._lastOutputFrame.fill(0);
          } else {
            output.fill(0, framesToRead * channels, requestedFrames * channels);
          }

          this._inUnderrun = true;
        } else {
          output.fill(0, framesToRead * channels, requestedFrames * channels);
        }

        Atomics.add(this._ctrl, HDR_UNDERRUNS, 1);
        Atomics.add(this._ctrl, HDR_UNDERRUN_FRAMES, missingFrames);
      } else {
        // Steady-state full pull: record trailing frame without mutating output.
        const lastBase = (requestedFrames - 1) * channels;
        for (let c = 0; c < channels; c++) {
          this._lastOutputFrame[c] = output[lastBase + c];
        }
        this._hasActiveAudio = true;
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
      this._recordFillTransition(used, Math.max(0, used - toDiscard));
      return toDiscard;
    }

    reset() {
      Atomics.store(this._ctrl, HDR_WRITE_CURSOR, 0);
      Atomics.store(this._ctrl, HDR_READ_CURSOR, 0);
      Atomics.store(this._ctrl, HDR_UNDERRUNS, 0);
      Atomics.store(this._ctrl, HDR_UNDERRUN_FRAMES, 0);
      Atomics.store(this._ctrl, HDR_OVERFLOWS, 0);
      Atomics.store(this._ctrl, HDR_DROPPED_FRAMES, 0);
      this._lastOutputFrame.fill(0);
      this._hasActiveAudio = false;
      this._inUnderrun = false;
      this._fadeInPos = 0;
      this._fadeInTotal = 0;
      this._fadeOutCount = 0;
      this._fadeInCount = 0;
      this._peakFillFrames = 0;
      this._minFillFrames = Infinity;
      this._fillFramesSum = 0;
      this._fillSamples = 0;
      this._lastObservedFill = null;
    }

    getStats() {
      const overflows = this.overflows;
      const underruns = this.underruns;
      return {
        capacityFrames: this._capacityFrames,
        channels: this._channels,
        availableFrames: this.availableFrames(),
        freeFrames: this.freeFrames(),
        volume: this._volume,
        muted: this._muted,
        effectiveGain: this.effectiveGain(),
        declickFrames: this._declickFrames,
        underruns,
        underrunCount: underruns,
        underrunFrames: this.underrunFrames,
        fadeOutCount: this._fadeOutCount,
        fadeInCount: this._fadeInCount,
        peakFillFrames: this.peakFillFrames,
        minFillFrames: this.minFillFrames,
        avgFillRatio: this.avgFillRatio,
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
      this._declickFrames = sanitizeDeclickFrames(
        opts.declickFrames,
        DEFAULT_DECLICK_FRAMES
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

      this._softClip = opts.softClip !== undefined ? Boolean(opts.softClip) : false;
      const rawKnee =
        opts.limiterKnee !== undefined
          ? opts.limiterKnee
          : opts.softClipKnee !== undefined
            ? opts.softClipKnee
            : DEFAULT_SOFT_CLIP_KNEE;
      this._limiterKnee =
        typeof rawKnee === 'number' && Number.isFinite(rawKnee) && rawKnee >= 0 && rawKnee < 1.0
          ? rawKnee
          : DEFAULT_SOFT_CLIP_KNEE;

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
      this._downmixedFrames = 0;
      this._resampledFrames = 0;
      this._clippedSamples = 0;
      this._activeVoicePeak = 0;
      this._resamplerState = { phase: 0.0 };
      this._lastResampleSrcRate = null;
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

    get declickFrames() {
      return this._declickFrames;
    }

    setDeclickFrames(frames) {
      this._declickFrames = sanitizeDeclickFrames(frames, this._declickFrames);
      if (this.ring) {
        this.ring.setDeclickFrames(this._declickFrames);
      }
      return this._declickFrames;
    }

    recordActiveVoices(count) {
      const sanitized = sanitizeNonNegInt(count, -1);
      if (sanitized > this._activeVoicePeak) {
        this._activeVoicePeak = sanitized;
      }
      return this._activeVoicePeak;
    }

    _resetTelemetryAndCounters() {
      this.totalPushedFrames = 0;
      this.totalPulledFrames = 0;
      this.framesDrained = 0;
      this._drainTicks = 0;
      this._overruns = 0;
      this._rejectedPackets = 0;
      this._sabCopiedPackets = 0;
      this._sanitizedSamples = 0;
      this._downmixedFrames = 0;
      this._resampledFrames = 0;
      this._clippedSamples = 0;
      this._activeVoicePeak = 0;
      this._resamplerState = { phase: 0.0 };
      this._lastResampleSrcRate = null;
      this._lastDrainMs = null;
    }

    reset() {
      if (this.ring) {
        this.ring.reset();
      }
      this._resetTelemetryAndCounters();
    }

    stop() {
      this._stopDrainTimer();
      this.reset();
    }

    destroy() {
      this._stopDrainTimer();
    }

    effectiveGain() {
      return this.muted ? 0.0 : this.volume;
    }

    pullSamples(output, gainOverride) {
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
      const framesRead = this.ring.pullSamples(output, gainOverride);
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
            declickFrames: this._declickFrames,
            underruns: 0,
            underrunCount: 0,
            underrunFrames: 0,
            fadeOutCount: 0,
            fadeInCount: 0,
            peakFillFrames: 0,
            minFillFrames: 0,
            avgFillRatio: 0.0,
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
        declickFrames: this._declickFrames,
        framesDrained: this.framesDrained,
        underruns: ringStats.underruns,
        underrunCount: ringStats.underrunCount,
        underrunFrames: ringStats.underrunFrames,
        fadeOutCount: ringStats.fadeOutCount,
        fadeInCount: ringStats.fadeInCount,
        peakFillFrames: ringStats.peakFillFrames,
        minFillFrames: ringStats.minFillFrames,
        avgFillRatio: ringStats.avgFillRatio,
        overruns,
        overflows: ringStats.overflows,
        droppedFrames: ringStats.droppedFrames,
        totalPushedFrames: this.totalPushedFrames,
        totalPulledFrames: this.totalPulledFrames,
        drainTicks: this._drainTicks,
        rejectedPackets: this._rejectedPackets,
        sabCopiedPackets: this._sabCopiedPackets,
        sanitizedSamples: this._sanitizedSamples,
        nanSanitizedSamples: this._sanitizedSamples,
        downmixedFrames: this._downmixedFrames,
        resampledFrames: this._resampledFrames,
        clippedSamples: this._clippedSamples,
        activeVoicePeak: this._activeVoicePeak,
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
          if (msg.declickFrames !== undefined) {
            this._declickFrames = sanitizeDeclickFrames(
              msg.declickFrames,
              this._declickFrames
            );
          }
          if (msg.softClip !== undefined) {
            this._softClip = Boolean(msg.softClip);
          }
          if (msg.limiterKnee !== undefined || msg.softClipKnee !== undefined) {
            const rawKnee =
              msg.limiterKnee !== undefined ? msg.limiterKnee : msg.softClipKnee;
            if (typeof rawKnee === 'number' && Number.isFinite(rawKnee) && rawKnee >= 0 && rawKnee < 1.0) {
              this._limiterKnee = rawKnee;
            }
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
            const reconfiguringSameSab = Boolean(
              this.ring && this.ring.sab === msg.ringSab
            );
            nextRing = new AudioRingBuffer(msg.ringSab, {
              channels: nextChannels,
              volume: this.volume,
              muted: this.muted,
              declickFrames: this._declickFrames,
            });
            if (reconfiguringSameSab) {
              nextRing.reset();
            }
          } else {
            const cap = sanitizePositiveInt(
              msg.capacityFrames,
              DEFAULT_CAPACITY_FRAMES,
              1,
              MAX_CAPACITY_FRAMES
            );
            nextRing = AudioRingBuffer.create(cap, nextChannels, {
              declickFrames: this._declickFrames,
            });
            nextRing.setVolume(this.volume);
            nextRing.setMuted(this.muted);
          }

          this.sampleRate = nextSampleRate;
          this.ring = nextRing;
          this.channels = nextRing.channels;
          this.outputState = this._resolveInitOutputState(msg);
          this.initialized = true;
          this._resetTelemetryAndCounters();

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

          const srcChannels = msg.channels !== undefined ? msg.channels : this.channels;
          if (!Number.isInteger(srcChannels) || srcChannels < 1 || srcChannels > MAX_CHANNELS) {
            this._rejectedPackets += 1;
            if (safeId !== undefined) {
              this._safePostMessage({
                id: safeId,
                error: `Unsupported PCM channel count: ${srcChannels}`,
              });
            }
            return;
          }
          if (
            srcChannels !== this.channels &&
            !(this.channels === 2 && SUPPORTED_DOWNMIX_CHANNELS.includes(srcChannels))
          ) {
            this._rejectedPackets += 1;
            if (safeId !== undefined) {
              this._safePostMessage({
                id: safeId,
                error: `Cannot convert ${srcChannels}-channel PCM into ${this.channels}-channel ring`,
              });
            }
            return;
          }

          const srcSampleRate = msg.sampleRate !== undefined ? msg.sampleRate : this.sampleRate;
          if (
            typeof srcSampleRate !== 'number' ||
            !Number.isFinite(srcSampleRate) ||
            srcSampleRate < MIN_SAMPLE_RATE ||
            srcSampleRate > MAX_SAMPLE_RATE
          ) {
            this._rejectedPackets += 1;
            if (safeId !== undefined) {
              this._safePostMessage({
                id: safeId,
                error: `Invalid PCM sampleRate: ${srcSampleRate}`,
              });
            }
            return;
          }
          const rateRatio = srcSampleRate / this.sampleRate;
          if (rateRatio > MAX_RESAMPLE_RATIO || 1.0 / rateRatio > MAX_RESAMPLE_RATIO) {
            this._rejectedPackets += 1;
            if (safeId !== undefined) {
              this._safePostMessage({
                id: safeId,
                error: `PCM sampleRate ratio (${srcSampleRate} -> ${this.sampleRate}) exceeds ${MAX_RESAMPLE_RATIO}x guard`,
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

          let pcm = extracted.pcm;

          // Step 1: ITU-R BS.775 multi-channel (5.1 / 7.1 / 4ch) or mono -> stereo downmix/adapt.
          if (srcChannels !== this.channels) {
            const dmStats = {
              downmixedFrames: 0,
              clippedSamples: 0,
              nanSanitizedSamples: 0,
              peakLevel: 0.0,
            };
            const dmSoftClip = msg.softClip !== undefined ? Boolean(msg.softClip) : true;
            const dmKnee =
              msg.limiterKnee !== undefined
                ? msg.limiterKnee
                : msg.softClipKnee !== undefined
                  ? msg.softClipKnee
                  : this._limiterKnee;
            pcm = downmixToStereo(pcm, srcChannels, {
              centerGain: msg.centerGain,
              surroundGain: msg.surroundGain,
              lfeGain: msg.lfeGain,
              softClip: dmSoftClip,
              limiterKnee: dmKnee,
              stats: dmStats,
            });
            this._downmixedFrames += dmStats.downmixedFrames;
            this._clippedSamples += dmStats.clippedSamples;
            this._sanitizedSamples += dmStats.nanSanitizedSamples;
          } else if (msg.softClip === true || (msg.softClip === undefined && this._softClip)) {
            const knee =
              msg.limiterKnee !== undefined
                ? msg.limiterKnee
                : msg.softClipKnee !== undefined
                  ? msg.softClipKnee
                  : this._limiterKnee;
            const limited = new Float32Array(pcm.length);
            let clipped = 0;
            for (let i = 0; i < pcm.length; i++) {
              const v = pcm[i];
              if (v > knee || v < -knee) {
                clipped += 1;
              }
              limited[i] = softClipSample(v, knee);
            }
            this._clippedSamples += clipped;
            pcm = limited;
          }

          // Step 2: Smooth 44.1 kHz <-> 48 kHz (and general) linear resampling with phase continuity.
          if (srcSampleRate !== this.sampleRate) {
            if (this._lastResampleSrcRate !== srcSampleRate) {
              this._resamplerState = { phase: 0.0 };
              this._lastResampleSrcRate = srcSampleRate;
            }
            const maxOutputFrames = Math.max(
              1,
              Math.floor(this._maxPcmBytes / (this.channels * Float32Array.BYTES_PER_ELEMENT))
            );
            const rsStats = {
              resampledFrames: 0,
              nanSanitizedSamples: 0,
              clippedSamples: 0,
            };
            pcm = resampleLinear(pcm, this.channels, srcSampleRate, this.sampleRate, {
              state: this._resamplerState,
              maxOutputFrames,
              stats: rsStats,
            });
            this._resampledFrames += rsStats.resampledFrames;
            this._sanitizedSamples += rsStats.nanSanitizedSamples;
            this._clippedSamples += rsStats.clippedSamples;
          } else if (this._lastResampleSrcRate !== null) {
            this._resamplerState = { phase: 0.0 };
            this._lastResampleSrcRate = null;
          }

          const inputFrames = Math.floor(pcm.length / this.channels);

          if (
            typeof msg.activeVoices === 'number' &&
            Number.isFinite(msg.activeVoices) &&
            msg.activeVoices >= 0
          ) {
            this.recordActiveVoices(msg.activeVoices);
          } else if (inputFrames > 0) {
            this.recordActiveVoices(1);
          }

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

        if (msg.reset || msg.type === 'reset') {
          this.reset();
          if (safeId !== undefined) {
            this._safePostMessage({
              id: safeId,
              reset: true,
              outputState: this.outputState,
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
    DEFAULT_DECLICK_FRAMES,
    MAX_DECLICK_FRAMES,
    DEFAULT_SOFT_CLIP_KNEE,
    ITU_BS775_CENTER_GAIN,
    ITU_BS775_SURROUND_GAIN,
    ITU_BS775_LFE_GAIN,
    MAX_RESAMPLE_RATIO,
    DEFAULT_MAX_RESAMPLE_FRAMES,
    SUPPORTED_DOWNMIX_CHANNELS,
    RING_HEADER_BYTES,
    extractSafePcmSamples,
    clonePcmForTransfer,
    sanitizeTransferList,
    softClipSample,
    downmixToStereo,
    resampleLinear,
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
