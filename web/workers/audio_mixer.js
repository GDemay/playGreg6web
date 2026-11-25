'use strict';

(function (root) {
  const DEFAULT_SAMPLE_RATE = 48000;
  const MIN_SAMPLE_RATE = 8000;
  const MAX_SAMPLE_RATE = 384000;
  const MAX_RESAMPLE_RATIO = 16;
  const DEFAULT_MAX_RESAMPLE_FRAMES = 524288;
  const DEFAULT_CHANNELS = 2;
  const DEFAULT_MAX_VOICES = 64;
  const DEFAULT_QUANTUM_FRAMES = 128;
  const DEFAULT_INITIAL_BUFFER_FRAMES = 1024;
  const DEFAULT_GAIN_RAMP_FRAMES = 64;
  const DEFAULT_UNDERRUN_RAMP_FRAMES = 64;
  const DEFAULT_SOFT_CLIP_KNEE = 0.95;
  const DEFAULT_LIMITER_KNEE = DEFAULT_SOFT_CLIP_KNEE;
  const DEFAULT_SOFT_LIMIT_KNEE = 0.9;

  // ITU-R BS.775-3 multi-channel to stereo downmix weights (-3 dB center & surround, LFE omitted by default).
  const ITU_BS775_CENTER_GAIN = Math.SQRT1_2;
  const ITU_BS775_SURROUND_GAIN = Math.SQRT1_2;
  const ITU_BS775_LFE_GAIN = 0.0;
  const SUPPORTED_VOICE_CHANNELS = Object.freeze([1, 2, 4, 6, 8]);

  // Standard IMA / DVI ADPCM step-index adjustment table (16 4-bit nibble values).
  const IMA_INDEX_TABLE = Object.freeze([
    -1, -1, -1, -1, 2, 4, 6, 8,
    -1, -1, -1, -1, 2, 4, 6, 8,
  ]);

  // Standard IMA / DVI ADPCM 89-entry step-size quantization table.
  const IMA_STEP_TABLE = Object.freeze([
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
    19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
    50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
    130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
    337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
    876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
    2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
    5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
    15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
  ]);

  function clamp(value, min, max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
  }

  function sanitizeFiniteNonNeg(value) {
    return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : 0;
  }

  function sanitizeCount(value) {
    return typeof value === 'number' && Number.isFinite(value) && value > 0
      ? Math.floor(value)
      : 0;
  }

  function sanitizeUnitLevel(value) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
      return 0;
    }
    return value > 1.0 ? 1.0 : value;
  }

  function toUint8Array(input) {
    if (input instanceof Uint8Array) {
      return input;
    }
    if (input instanceof ArrayBuffer) {
      return new Uint8Array(input);
    }
    if (ArrayBuffer.isView(input)) {
      return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
    }
    if (Array.isArray(input)) {
      return Uint8Array.from(input);
    }
    throw new TypeError('Expected Uint8Array, ArrayBuffer, TypedArray, or number[]');
  }

  function toFloat32Samples(samples) {
    if (samples instanceof Float32Array) {
      return samples;
    }
    if (Array.isArray(samples)) {
      return Float32Array.from(samples);
    }
    if (samples instanceof ArrayBuffer) {
      return new Float32Array(samples);
    }
    if (ArrayBuffer.isView(samples)) {
      return new Float32Array(
        samples.buffer,
        samples.byteOffset,
        Math.floor(samples.byteLength / Float32Array.BYTES_PER_ELEMENT)
      );
    }
    throw new TypeError('Voice samples must be a Float32Array, TypedArray, ArrayBuffer, or number[]');
  }

  /**
   * Decodes signed 16-bit little-endian PCM data into normalized Float32 samples in [-1, 1].
   *
   * @param {Int16Array|Uint8Array|ArrayBuffer|ArrayBufferView|number[]} bytesOrInt16
   * @param {number} [channels=1] Number of interleaved channels (used to align complete frames).
   * @returns {Float32Array}
   */
  function decodePcm16(bytesOrInt16, channels = 1) {
    if (!Number.isInteger(channels) || channels <= 0) {
      throw new RangeError(`channels must be a positive integer, got ${channels}`);
    }

    if (bytesOrInt16 instanceof Int16Array) {
      const sampleCount = Math.floor(bytesOrInt16.length / channels) * channels;
      const out = new Float32Array(sampleCount);
      for (let i = 0; i < sampleCount; i++) {
        out[i] = bytesOrInt16[i] / 32768.0;
      }
      return out;
    }

    if (Array.isArray(bytesOrInt16)) {
      const sampleCount = Math.floor(bytesOrInt16.length / channels) * channels;
      const out = new Float32Array(sampleCount);
      for (let i = 0; i < sampleCount; i++) {
        const clamped = clamp(bytesOrInt16[i] | 0, -32768, 32767);
        out[i] = clamped / 32768.0;
      }
      return out;
    }

    let view;
    if (bytesOrInt16 instanceof ArrayBuffer) {
      view = new DataView(bytesOrInt16);
    } else if (ArrayBuffer.isView(bytesOrInt16)) {
      view = new DataView(
        bytesOrInt16.buffer,
        bytesOrInt16.byteOffset,
        bytesOrInt16.byteLength
      );
    } else {
      throw new TypeError('decodePcm16 expects Int16Array, Uint8Array, ArrayBuffer, DataView, or number[]');
    }

    const totalSamples = Math.floor(view.byteLength / 2);
    const sampleCount = Math.floor(totalSamples / channels) * channels;
    const out = new Float32Array(sampleCount);
    for (let i = 0; i < sampleCount; i++) {
      out[i] = view.getInt16(i * 2, true) / 32768.0;
    }
    return out;
  }

  /**
   * Decodes one 4-bit IMA-ADPCM nibble and advances predictor + stepIndex in place.
   *
   * @param {number} nibble 4-bit value (0..15).
   * @param {{ predictor: number, stepIndex: number }} state
   * @returns {number} Decoded signed 16-bit PCM sample in [-32768, 32767].
   */
  function decodeImaNibble(nibble, state) {
    const step = IMA_STEP_TABLE[state.stepIndex];
    let diff = step >> 3;
    if (nibble & 1) diff += step >> 2;
    if (nibble & 2) diff += step >> 1;
    if (nibble & 4) diff += step;

    let predictor = (nibble & 8) ? state.predictor - diff : state.predictor + diff;
    if (predictor > 32767) predictor = 32767;
    else if (predictor < -32768) predictor = -32768;

    let stepIndex = state.stepIndex + IMA_INDEX_TABLE[nibble & 0x0f];
    if (stepIndex < 0) stepIndex = 0;
    else if (stepIndex > 88) stepIndex = 88;

    state.predictor = predictor;
    state.stepIndex = stepIndex;
    return predictor;
  }

  /**
   * Decodes standard IMA / DVI ADPCM 4-bit compressed audio into normalized Float32 PCM in [-1, 1].
   *
   * Supports two layouts:
   * - Block-framed (`blockAlign > 0`, standard WAV / XAudio2 / AWC IMA-ADPCM):
   *   Each block begins with a 4-byte header per channel (`int16le predictor`,
   *   `uint8 stepIndex`, `uint8 reserved`). For mono, each remaining byte holds
   *   two 4-bit samples (low nibble first). For stereo/multi-channel, remaining
   *   data is interleaved in 4-byte (8-sample) words per channel.
   * - Raw stream (`blockAlign` omitted or 0):
   *   Headerless stream starting at predictor=0, stepIndex=0 per channel;
   *   low nibble then high nibble per byte, interleaved across channels.
   *
   * @param {Uint8Array|ArrayBuffer|ArrayBufferView|number[]} bytes
   * @param {number} [channels=1]
   * @param {number} [blockAlign=0]
   * @returns {Float32Array}
   */
  function decodeImaAdpcm(bytes, channels = 1, blockAlign = 0) {
    if (!Number.isInteger(channels) || channels <= 0) {
      throw new RangeError(`channels must be a positive integer, got ${channels}`);
    }
    if (blockAlign !== undefined && (!Number.isInteger(blockAlign) || blockAlign < 0)) {
      throw new RangeError(`blockAlign must be a non-negative integer, got ${blockAlign}`);
    }

    const data = toUint8Array(bytes);
    if (data.byteLength === 0) {
      return new Float32Array(0);
    }

    const scale = 1.0 / 32768.0;

    // Mode 1: Block-framed IMA-ADPCM (WAV / XAudio2 / AWC block format).
    if (blockAlign > 0) {
      const headerBytes = 4 * channels;
      if (blockAlign < headerBytes) {
        throw new RangeError(
          `blockAlign (${blockAlign}) must be at least ${headerBytes} bytes for ${channels} channel(s)`
        );
      }

      const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
      const blocks = [];
      let totalFrames = 0;

      for (let offset = 0; offset < data.byteLength; offset += blockAlign) {
        const blockLen = Math.min(blockAlign, data.byteLength - offset);
        if (blockLen < headerBytes) {
          break;
        }
        const payloadBytes = blockLen - headerBytes;
        let blockFrames;
        if (channels === 1) {
          blockFrames = 1 + payloadBytes * 2;
        } else {
          // 4-byte (8-sample) word per channel per interleaved chunk.
          const chunkBytes = 4 * channels;
          const fullChunks = Math.floor(payloadBytes / chunkBytes);
          blockFrames = 1 + fullChunks * 8;
        }
        blocks.push({ offset, blockLen, blockFrames });
        totalFrames += blockFrames;
      }

      const out = new Float32Array(totalFrames * channels);
      const states = new Array(channels);
      for (let ch = 0; ch < channels; ch++) {
        states[ch] = { predictor: 0, stepIndex: 0 };
      }

      let outFrame = 0;
      for (let b = 0; b < blocks.length; b++) {
        const { offset, blockLen } = blocks[b];

        // Read 4-byte header per channel and emit initial predictor frame.
        for (let ch = 0; ch < channels; ch++) {
          const hdrOff = offset + ch * 4;
          const predictor = view.getInt16(hdrOff, true);
          const stepIndex = clamp(data[hdrOff + 2], 0, 88);
          states[ch].predictor = predictor;
          states[ch].stepIndex = stepIndex;
          out[outFrame * channels + ch] = predictor * scale;
        }
        outFrame += 1;

        let pos = offset + headerBytes;
        const blockEnd = offset + blockLen;

        if (channels === 1) {
          const st = states[0];
          while (pos < blockEnd) {
            const byte = data[pos++];
            out[outFrame++] = decodeImaNibble(byte & 0x0f, st) * scale;
            out[outFrame++] = decodeImaNibble((byte >> 4) & 0x0f, st) * scale;
          }
        } else {
          const chunkBytes = 4 * channels;
          while (pos + chunkBytes <= blockEnd) {
            for (let ch = 0; ch < channels; ch++) {
              const st = states[ch];
              let frameIdx = outFrame;
              for (let k = 0; k < 4; k++) {
                const byte = data[pos++];
                out[frameIdx * channels + ch] = decodeImaNibble(byte & 0x0f, st) * scale;
                frameIdx += 1;
                out[frameIdx * channels + ch] = decodeImaNibble((byte >> 4) & 0x0f, st) * scale;
                frameIdx += 1;
              }
            }
            outFrame += 8;
          }
        }
      }

      return out;
    }

    // Mode 2: Raw headerless IMA-ADPCM stream (nibble-interleaved across channels).
    const totalNibbles = data.byteLength * 2;
    const totalFrames = Math.floor(totalNibbles / channels);
    const totalSamples = totalFrames * channels;
    const out = new Float32Array(totalSamples);

    const states = new Array(channels);
    for (let ch = 0; ch < channels; ch++) {
      states[ch] = { predictor: 0, stepIndex: 0 };
    }

    let sampleIdx = 0;
    for (let i = 0; i < data.byteLength && sampleIdx < totalSamples; i++) {
      const byte = data[i];
      const ch0 = sampleIdx % channels;
      out[sampleIdx++] = decodeImaNibble(byte & 0x0f, states[ch0]) * scale;
      if (sampleIdx < totalSamples) {
        const ch1 = sampleIdx % channels;
        out[sampleIdx++] = decodeImaNibble((byte >> 4) & 0x0f, states[ch1]) * scale;
      }
    }

    return out;
  }

  /**
   * Computes equal-power stereo panning gains for mono voices.
   * pan = -1 => (1, 0), pan = 0 => (sqrt(0.5), sqrt(0.5)), pan = +1 => (0, 1).
   */
  function computeMonoPanGains(pan) {
    const p = clamp(pan, -1.0, 1.0);
    if (p <= -1.0) return { left: 1.0, right: 0.0 };
    if (p >= 1.0) return { left: 0.0, right: 1.0 };
    if (p === 0.0) return { left: Math.SQRT1_2, right: Math.SQRT1_2 };
    const angle = (p + 1.0) * (Math.PI * 0.25);
    return {
      left: Math.cos(angle),
      right: Math.sin(angle),
    };
  }

  /**
   * Precomputes and stores scalar mono and stereo/surround equal-power pan
   * coefficients directly on a voice object so the hot mix loop performs
   * zero trigonometric calls and zero object allocations per quantum.
   */
  function updateVoicePanGains(voice) {
    const p = voice.pan;
    if (p <= -1.0) {
      voice.panL = 1.0;
      voice.panR = 0.0;
      voice.panLL = 1.0;
      voice.panLR = 1.0;
      voice.panRL = 0.0;
      voice.panRR = 0.0;
    } else if (p >= 1.0) {
      voice.panL = 0.0;
      voice.panR = 1.0;
      voice.panLL = 0.0;
      voice.panLR = 0.0;
      voice.panRL = 1.0;
      voice.panRR = 1.0;
    } else if (p === 0.0) {
      voice.panL = Math.SQRT1_2;
      voice.panR = Math.SQRT1_2;
      voice.panLL = 1.0;
      voice.panLR = 0.0;
      voice.panRL = 0.0;
      voice.panRR = 1.0;
    } else {
      const monoAngle = (p + 1.0) * (Math.PI * 0.25);
      voice.panL = Math.cos(monoAngle);
      voice.panR = Math.sin(monoAngle);
      if (p < 0.0) {
        const x = (p + 1.0) * (Math.PI * 0.5);
        voice.panLL = 1.0;
        voice.panLR = Math.cos(x);
        voice.panRL = 0.0;
        voice.panRR = Math.sin(x);
      } else {
        const x = p * (Math.PI * 0.5);
        voice.panLL = Math.cos(x);
        voice.panLR = 0.0;
        voice.panRL = Math.sin(x);
        voice.panRR = 1.0;
      }
    }
  }

  /**
   * Resolves a ramp specification (non-negative frame count or options object
   * with `rampFrames`, `frames`, `durationMs`, or `durationSec`) into an integer
   * frame count without allocating in the numeric fast path.
   */
  function resolveRampFrames(rampSpec, sampleRate, fallbackFrames) {
    if (rampSpec === undefined || rampSpec === null) {
      return fallbackFrames;
    }
    if (typeof rampSpec === 'number') {
      if (!Number.isFinite(rampSpec) || rampSpec < 0) {
        throw new RangeError(`rampFrames must be a non-negative finite number, got ${rampSpec}`);
      }
      return Math.floor(rampSpec);
    }
    if (typeof rampSpec === 'object') {
      if (rampSpec.rampFrames !== undefined) {
        const v = rampSpec.rampFrames;
        if (typeof v !== 'number' || !Number.isFinite(v) || v < 0) {
          throw new RangeError(`rampFrames must be a non-negative finite number, got ${v}`);
        }
        return Math.floor(v);
      }
      if (rampSpec.frames !== undefined) {
        const v = rampSpec.frames;
        if (typeof v !== 'number' || !Number.isFinite(v) || v < 0) {
          throw new RangeError(`frames must be a non-negative finite number, got ${v}`);
        }
        return Math.floor(v);
      }
      if (rampSpec.durationMs !== undefined) {
        const ms = rampSpec.durationMs;
        if (typeof ms !== 'number' || !Number.isFinite(ms) || ms < 0) {
          throw new RangeError(`durationMs must be a non-negative finite number, got ${ms}`);
        }
        return Math.round((ms / 1000.0) * sampleRate);
      }
      if (rampSpec.durationSec !== undefined) {
        const sec = rampSpec.durationSec;
        if (typeof sec !== 'number' || !Number.isFinite(sec) || sec < 0) {
          throw new RangeError(`durationSec must be a non-negative finite number, got ${sec}`);
        }
        return Math.round(sec * sampleRate);
      }
      return fallbackFrames;
    }
    throw new RangeError(`rampFrames must be a non-negative number or options object, got ${rampSpec}`);
  }

  /**
   * Smooth C1-continuous soft-clip limiter backed by Math.tanh.
   * Non-finite inputs (NaN / +-Infinity) are sanitized to 0.0.
   * Completely bit-exact for |x| <= knee and smoothly saturates toward +-1.0 beyond knee.
   *
   * @param {number} x Input sample value.
   * @param {number} [knee=0.95] Linear passthrough knee in [0, 1).
   * @returns {number} Finite sample strictly bounded inside [-1.0, 1.0].
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

  /**
   * Sanitizes non-finite samples (`NaN` / `±Infinity`) to `0.0`, passes normal
   * audio `|x| <= knee` (default `0.9`) bit-exact linearly, and smoothly
   * soft-clips `|x| > knee` into `[-1.0, 1.0]` using a C1-continuous `tanh` knee.
   *
   * @param {number} x Input sample value.
   * @param {number} [knee=0.9] Linear passthrough threshold in `[0, 1)`.
   * @returns {number} Finite sample strictly bounded inside `[-1.0, 1.0]`.
   */
  function softLimitSample(x, knee = DEFAULT_SOFT_LIMIT_KNEE) {
    if (typeof knee !== 'number' || !Number.isFinite(knee) || knee < 0 || knee >= 1.0) {
      throw new RangeError(`knee must be in [0, 1), got ${knee}`);
    }
    if (typeof x !== 'number' || !Number.isFinite(x)) {
      return 0.0;
    }
    const absX = x < 0 ? -x : x;
    if (absX <= knee) {
      return x;
    }
    const headroom = 1.0 - knee;
    const compressed = knee + headroom * Math.tanh((absX - knee) / headroom);
    const bounded = compressed > 1.0 ? 1.0 : compressed;
    return x < 0 ? -bounded : bounded;
  }

  /**
   * Sanitizes non-finite samples to `0.0` and applies transparent `softLimitSample`
   * protection in place across `samples[0 .. count - 1]`.
   * Pass a reusable `outStats` object to avoid any heap allocation on hot paths.
   *
   * @param {Float32Array|number[]} samples
   * @param {number} [count] Number of samples to process (defaults to `samples.length`).
   * @param {number|object} [kneeOrOptions=0.9] Knee in `[0, 1)` or `{ knee, stats }`.
   * @param {{ softClippedSamples: number, nonFiniteSanitized: number }} [outStats]
   * @returns {{ softClippedSamples: number, nonFiniteSanitized: number }}
   */
  function sanitizeAndSoftLimitBuffer(samples, count, kneeOrOptions = DEFAULT_SOFT_LIMIT_KNEE, outStats = null) {
    if (!samples || typeof samples.length !== 'number') {
      throw new TypeError('sanitizeAndSoftLimitBuffer requires a Float32Array or array-like buffer');
    }

    let knee = DEFAULT_SOFT_LIMIT_KNEE;
    let statsTarget = outStats;
    if (typeof kneeOrOptions === 'object' && kneeOrOptions !== null) {
      if (kneeOrOptions.knee !== undefined) {
        knee = kneeOrOptions.knee;
      } else if (kneeOrOptions.softLimitKnee !== undefined) {
        knee = kneeOrOptions.softLimitKnee;
      }
      if (!statsTarget && kneeOrOptions.stats && typeof kneeOrOptions.stats === 'object') {
        statsTarget = kneeOrOptions.stats;
      } else if (
        !statsTarget &&
        kneeOrOptions.knee === undefined &&
        kneeOrOptions.softLimitKnee === undefined
      ) {
        statsTarget = kneeOrOptions;
      }
    } else if (kneeOrOptions !== undefined) {
      knee = kneeOrOptions;
    }

    if (typeof knee !== 'number' || !Number.isFinite(knee) || knee < 0 || knee >= 1.0) {
      throw new RangeError(`knee must be in [0, 1), got ${knee}`);
    }

    let n;
    if (count === undefined || count === null) {
      n = samples.length;
    } else {
      if (typeof count !== 'number' || !Number.isFinite(count) || count < 0) {
        throw new RangeError(`count must be a non-negative finite number, got ${count}`);
      }
      n = Math.floor(count);
      if (n > samples.length) n = samples.length;
    }

    const negKnee = -knee;
    const headroom = 1.0 - knee;
    let softClippedSamples = 0;
    let nonFiniteSanitized = 0;

    for (let i = 0; i < n; i++) {
      const s = samples[i];
      if (s - s !== 0) {
        samples[i] = 0.0;
        nonFiniteSanitized += 1;
      } else if (s > knee) {
        const compressed = knee + headroom * Math.tanh((s - knee) / headroom);
        samples[i] = compressed > 1.0 ? 1.0 : compressed;
        softClippedSamples += 1;
      } else if (s < negKnee) {
        const compressed = knee + headroom * Math.tanh((-s - knee) / headroom);
        samples[i] = compressed > 1.0 ? -1.0 : -compressed;
        softClippedSamples += 1;
      }
    }

    if (statsTarget) {
      statsTarget.softClippedSamples = (statsTarget.softClippedSamples || 0) + softClippedSamples;
      statsTarget.nonFiniteSanitized = (statsTarget.nonFiniteSanitized || 0) + nonFiniteSanitized;
      return statsTarget;
    }

    return { softClippedSamples, nonFiniteSanitized };
  }

  /**
   * Applies an in-place, zero-allocation half-cosine smooth gain ramp over
   * `frameCount` interleaved frames from `startGain` to `endGain`.
   *
   * @param {Float32Array|number[]} samples Interleaved PCM sample buffer modified in place.
   * @param {number} [channels=1] Number of interleaved channels (>= 1).
   * @param {number} [frameCount] Number of frames to ramp (clamped to buffer length).
   * @param {number} [startGain=1.0] Starting gain at frame 0.
   * @param {number} [endGain=0.0] Ending gain at frame `frameCount - 1`.
   * @returns {Float32Array|number[]} The mutated `samples` buffer.
   */
  function applyFadeRamp(samples, channels = 1, frameCount, startGain = 1.0, endGain = 0.0) {
    if (!samples || typeof samples.length !== 'number') {
      throw new TypeError('applyFadeRamp requires a Float32Array or array-like sample buffer');
    }
    if (!Number.isInteger(channels) || channels <= 0) {
      throw new RangeError(`channels must be a positive integer, got ${channels}`);
    }

    const maxFrames = Math.floor(samples.length / channels);
    let n;
    if (frameCount === undefined || frameCount === null) {
      n = maxFrames;
    } else {
      if (typeof frameCount !== 'number' || !Number.isFinite(frameCount) || frameCount < 0) {
        throw new RangeError(`frameCount must be a non-negative finite number, got ${frameCount}`);
      }
      n = Math.floor(frameCount);
      if (n > maxFrames) n = maxFrames;
    }
    if (n <= 0) {
      return samples;
    }

    const g0 = typeof startGain === 'number' && Number.isFinite(startGain) ? startGain : 0.0;
    const g1 = typeof endGain === 'number' && Number.isFinite(endGain) ? endGain : 0.0;

    if (n === 1) {
      for (let c = 0; c < channels; c++) {
        const s = samples[c];
        samples[c] = g1 === 0.0 || s - s !== 0 ? 0.0 : s * g1;
      }
      return samples;
    }

    const gDiff = g1 - g0;
    const lastIdx = n - 1;
    const invDenom = Math.PI / lastIdx;

    if (channels === 2) {
      for (let f = 0; f < n; f++) {
        const gain =
          f === 0
            ? g0
            : f === lastIdx
              ? g1
              : g0 + gDiff * (0.5 * (1.0 - Math.cos(f * invDenom)));
        const base = f * 2;
        if (gain === 0.0) {
          samples[base] = 0.0;
          samples[base + 1] = 0.0;
        } else {
          const s0 = samples[base];
          const s1 = samples[base + 1];
          samples[base] = s0 - s0 !== 0 ? 0.0 : s0 * gain;
          samples[base + 1] = s1 - s1 !== 0 ? 0.0 : s1 * gain;
        }
      }
    } else if (channels === 1) {
      for (let f = 0; f < n; f++) {
        const gain =
          f === 0
            ? g0
            : f === lastIdx
              ? g1
              : g0 + gDiff * (0.5 * (1.0 - Math.cos(f * invDenom)));
        const s = samples[f];
        samples[f] = gain === 0.0 || s - s !== 0 ? 0.0 : s * gain;
      }
    } else {
      for (let f = 0; f < n; f++) {
        const gain =
          f === 0
            ? g0
            : f === lastIdx
              ? g1
              : g0 + gDiff * (0.5 * (1.0 - Math.cos(f * invDenom)));
        const base = f * channels;
        if (gain === 0.0) {
          for (let c = 0; c < channels; c++) {
            samples[base + c] = 0.0;
          }
        } else {
          for (let c = 0; c < channels; c++) {
            const s = samples[base + c];
            samples[base + c] = s - s !== 0 ? 0.0 : s * gain;
          }
        }
      }
    }

    return samples;
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
   * Downmixes interleaved multi-channel PCM (1, 2, 4, 5.1 [6ch], or 7.1 [8ch])
   * into interleaved stereo Float32 PCM using ITU-R BS.775 weights:
   *   - 5.1 (FL, FR, FC, LFE, BL, BR):
   *       L = FL + 0.70710678 * FC + 0.70710678 * BL (+ lfeGain * LFE)
   *       R = FR + 0.70710678 * FC + 0.70710678 * BR (+ lfeGain * LFE)
   *   - 7.1 (FL, FR, FC, LFE, BL, BR, SL, SR):
   *       L = FL + 0.70710678 * FC + 0.70710678 * (BL + SL) (+ lfeGain * LFE)
   *       R = FR + 0.70710678 * FC + 0.70710678 * (BR + SR) (+ lfeGain * LFE)
   *
   * Sanitizes non-finite input samples per channel before summation so NaN/Infinity
   * in any channel never poisons the stereo mix, and applies the smooth tanh
   * soft-clip peak limiter by default so multi-channel Bink2/XAudio2 voices never
   * hard-clip.
   *
   * @param {Float32Array|ArrayBuffer|ArrayBufferView|number[]} samples
   * @param {number} channels Source interleaved channel count (1, 2, 4, 6, or 8).
   * @param {object} [options]
   * @returns {Float32Array} Interleaved stereo Float32Array (length = frameCount * 2).
   */
  function downmixToStereo(samples, channels, options = {}) {
    if (!Number.isInteger(channels) || !SUPPORTED_VOICE_CHANNELS.includes(channels)) {
      throw new RangeError(
        `downmixToStereo channels must be one of ${SUPPORTED_VOICE_CHANNELS.join(', ')}, got ${channels}`
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
          : DEFAULT_LIMITER_KNEE;
    if (typeof knee !== 'number' || !Number.isFinite(knee) || knee < 0 || knee >= 1.0) {
      throw new RangeError(`limiterKnee must be in [0, 1), got ${knee}`);
    }

    const src = toFloat32Samples(samples);
    const frameCount = Math.floor(src.length / channels);
    const out = new Float32Array(frameCount * 2);

    let nanSanitized = 0;
    let clipped = 0;
    let peak = 0.0;

    for (let f = 0; f < frameCount; f++) {
      const base = f * channels;
      let rawL;
      let rawR;

      if (channels === 6) {
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
        outL = clamp(rawL, -1.0, 1.0);
        outR = clamp(rawR, -1.0, 1.0);
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
   * general Xbox Series Bink2 / XAudio2 sample-rate conversions.
   *
   * Safeguards:
   * - Validates channel count, sample-rate range [8 kHz, 384 kHz], max conversion
   *   ratio (16x), and maximum output frame budget to prevent OOM allocations.
   * - Sanitizes non-finite input samples (NaN / +-Infinity) to 0.0 per channel
   *   before interpolation.
   * - Supports optional `options.state = { phase }` for drift-free fractional
   *   phase continuity across sequential audio chunks.
   *
   * @param {Float32Array|ArrayBuffer|ArrayBufferView|number[]} samples
   * @param {number} channels Interleaved channel count (1..8).
   * @param {number} srcSampleRate Source sample rate in Hz.
   * @param {number} dstSampleRate Destination sample rate in Hz.
   * @param {object} [options]
   * @returns {Float32Array}
   */
  function resampleLinear(samples, channels, srcSampleRate, dstSampleRate, options = {}) {
    if (!Number.isInteger(channels) || channels < 1 || channels > 8) {
      throw new RangeError(`resampleLinear channels must be an integer in [1, 8], got ${channels}`);
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
          : DEFAULT_LIMITER_KNEE;
    if (softClip && (typeof knee !== 'number' || !Number.isFinite(knee) || knee < 0 || knee >= 1.0)) {
      throw new RangeError(`limiterKnee must be in [0, 1), got ${knee}`);
    }

    const src = toFloat32Samples(samples);
    const srcFrames = Math.floor(src.length / channels);
    if (srcFrames <= 0) {
      return new Float32Array(0);
    }

    const state = opts.state && typeof opts.state === 'object' ? opts.state : null;
    const EPS = 1e-12;
    let nanSanitized = 0;
    let clipped = 0;

    // Fast same-rate passthrough when no fractional phase carry is active.
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
   * Lock-free multi-voice stereo audio mixer with zero-allocation hot mix
   * passes backed by pre-allocated Float32Array accumulation buffers, smooth
   * per-voice & master gain ramping, equal-power panning, linear resampling,
   * seamless looping, non-finite sample sanitization, configurable soft-clip
   * tanh limiter, and direct SharedArrayBuffer AudioRingBuffer pumping.
   */
  class AudioMixer {
    constructor(options = {}) {
      const sampleRate =
        options.sampleRate !== undefined ? options.sampleRate : DEFAULT_SAMPLE_RATE;
      if (typeof sampleRate !== 'number' || !Number.isFinite(sampleRate) || sampleRate <= 0) {
        throw new RangeError(`sampleRate must be a positive finite number, got ${sampleRate}`);
      }

      const maxVoices =
        options.maxVoices !== undefined ? options.maxVoices : DEFAULT_MAX_VOICES;
      if (!Number.isInteger(maxVoices) || maxVoices <= 0) {
        throw new RangeError(`maxVoices must be a positive integer, got ${maxVoices}`);
      }

      const masterGain =
        options.masterGain !== undefined
          ? options.masterGain
          : options.volume !== undefined
            ? options.volume
            : 1.0;
      if (typeof masterGain !== 'number' || !Number.isFinite(masterGain) || masterGain < 0) {
        throw new RangeError(`masterGain must be a non-negative finite number, got ${masterGain}`);
      }

      const muted = Boolean(options.muted);

      const softClip =
        options.softClip !== undefined ? Boolean(options.softClip) : true;

      const limiterKnee =
        options.limiterKnee !== undefined
          ? options.limiterKnee
          : options.softClipKnee !== undefined
            ? options.softClipKnee
            : options.softLimitKnee !== undefined
              ? options.softLimitKnee
              : DEFAULT_LIMITER_KNEE;
      if (
        typeof limiterKnee !== 'number' ||
        !Number.isFinite(limiterKnee) ||
        limiterKnee < 0 ||
        limiterKnee >= 1.0
      ) {
        throw new RangeError(`limiterKnee must be in [0, 1), got ${limiterKnee}`);
      }

      const initialBufferFrames =
        options.initialBufferFrames !== undefined
          ? options.initialBufferFrames
          : options.initialCapacityFrames !== undefined
            ? options.initialCapacityFrames
            : options.quantumFrames !== undefined
              ? options.quantumFrames
              : DEFAULT_INITIAL_BUFFER_FRAMES;
      if (!Number.isInteger(initialBufferFrames) || initialBufferFrames <= 0) {
        throw new RangeError(
          `initialBufferFrames must be a positive integer, got ${initialBufferFrames}`
        );
      }

      const defaultGainRampFrames =
        options.gainRampFrames !== undefined
          ? resolveRampFrames(options.gainRampFrames, sampleRate, 0)
          : options.rampFrames !== undefined
            ? resolveRampFrames(options.rampFrames, sampleRate, 0)
            : 0;

      const volumeRampFrames =
        options.volumeRampFrames !== undefined
          ? resolveRampFrames(options.volumeRampFrames, sampleRate, DEFAULT_GAIN_RAMP_FRAMES)
          : options.muteRampFrames !== undefined
            ? resolveRampFrames(options.muteRampFrames, sampleRate, DEFAULT_GAIN_RAMP_FRAMES)
            : defaultGainRampFrames > 0
              ? defaultGainRampFrames
              : DEFAULT_GAIN_RAMP_FRAMES;

      const underrunRampFrames =
        options.underrunRampFrames !== undefined
          ? resolveRampFrames(options.underrunRampFrames, sampleRate, DEFAULT_UNDERRUN_RAMP_FRAMES)
          : DEFAULT_UNDERRUN_RAMP_FRAMES;

      const reuseOutputBuffer = Boolean(options.reuseOutputBuffer);
      const initialEffectiveGain = muted ? 0.0 : masterGain;

      this.sampleRate = sampleRate;
      this.channels = DEFAULT_CHANNELS;
      this.maxVoices = maxVoices;
      this._muted = muted;
      this._masterGainUnmuted = masterGain;
      this._masterGain = initialEffectiveGain;
      this._masterGainTarget = initialEffectiveGain;
      this._masterGainRampRemaining = 0;
      this._masterGainRampStep = 0.0;
      this._defaultGainRampFrames = defaultGainRampFrames;
      this._volumeRampFrames = volumeRampFrames;
      this._underrunRampFrames = underrunRampFrames;
      this._reuseOutputBuffer = reuseOutputBuffer;
      this._softClip = softClip;
      this._limiterKnee = limiterKnee;

      this._voices = new Map();
      this._voiceList = [];
      this._finishedIds = [];
      this._nextVoiceId = 1;

      // Pre-allocated Float32Array stereo bus accumulation & scratch output buffers
      // so hot mix passes perform zero per-quantum heap allocations in steady state.
      const initialSamples = initialBufferFrames * DEFAULT_CHANNELS;
      this._mixScratch = new Float32Array(initialSamples);
      this._busLeftScratch = this._mixScratch.subarray(0, initialBufferFrames);
      this._busRightScratch = this._mixScratch.subarray(initialBufferFrames, initialSamples);
      this._outputScratch = new Float32Array(initialSamples);
      this._outputScratchView = this._outputScratch;
      this._outputScratchViewLen = initialSamples;
      this._monoScratch = new Float32Array(initialBufferFrames);
      this._monoScratchView = this._monoScratch;
      this._monoScratchViewLen = initialBufferFrames;
      this._emptyFloat32 = new Float32Array(0);
      this._scratchReallocations = 0;

      // Per-quantum scalar scratch accumulators (zero heap objects per quantum).
      this._quantumSanitized = 0;
      this._quantumDownmixed = 0;
      this._quantumResampled = 0;

      this._renderedFrames = 0;
      this._quantaMixed = 0;
      this._zeroAllocMixQuanta = 0;
      this._downmixedFrames = 0;
      this._resampledFrames = 0;
      this._ringUnderrunCount = 0;
      this._lastObservedRingUnderruns = 0;
      this._manualUnderrunQuanta = 0;
      this._underrunFadeOuts = 0;
      this._recoveryFadeIns = 0;
      this._recoveryFadeRemaining = 0;
      this._recoveryFadeTotal = 0;
      this._pumpRecoveryFadeRemaining = 0;
      this._pumpRecoveryFadeTotal = 0;
      this._lastOutL = 0.0;
      this._lastOutR = 0.0;
      this._peakLeft = 0.0;
      this._peakRight = 0.0;
      this._peakOutputLevel = 0.0;
      this._rmsLeftSumSq = 0.0;
      this._rmsRightSumSq = 0.0;
      this._rmsFrames = 0;
      this._clippedLeft = 0;
      this._clippedRight = 0;
      this._clippedSamples = 0;
      this._softClippedSamples = 0;
      this._sanitizedNonFiniteSamples = 0;
      this._gainRampsStarted = 0;
      this._activeVoicePeak = 0;
    }

    _ensureScratchCapacity(sampleCount) {
      if (sampleCount <= this._mixScratch.length) {
        return;
      }
      let nextSamples =
        this._mixScratch.length > 0
          ? this._mixScratch.length
          : DEFAULT_QUANTUM_FRAMES * DEFAULT_CHANNELS;
      while (nextSamples < sampleCount) {
        nextSamples *= 2;
      }
      const nextFrames = nextSamples >>> 1;
      this._mixScratch = new Float32Array(nextSamples);
      this._busLeftScratch = this._mixScratch.subarray(0, nextFrames);
      this._busRightScratch = this._mixScratch.subarray(nextFrames, nextSamples);
      this._outputScratch = new Float32Array(nextSamples);
      this._outputScratchView = this._outputScratch;
      this._outputScratchViewLen = nextSamples;
      this._scratchReallocations += 1;
    }

    _getScratchOutputView(sampleCount) {
      if (sampleCount === 0) {
        return this._emptyFloat32;
      }
      this._ensureScratchCapacity(sampleCount);
      if (
        this._outputScratchViewLen !== sampleCount ||
        this._outputScratchView.buffer !== this._outputScratch.buffer
      ) {
        this._outputScratchView =
          sampleCount === this._outputScratch.length
            ? this._outputScratch
            : this._outputScratch.subarray(0, sampleCount);
        this._outputScratchViewLen = sampleCount;
      }
      return this._outputScratchView;
    }

    _getMonoScratchView(frameCount) {
      if (frameCount === 0) {
        return this._emptyFloat32;
      }
      if (frameCount > this._monoScratch.length) {
        let nextFrames =
          this._monoScratch.length > 0 ? this._monoScratch.length : DEFAULT_QUANTUM_FRAMES;
        while (nextFrames < frameCount) {
          nextFrames *= 2;
        }
        this._monoScratch = new Float32Array(nextFrames);
        this._monoScratchView = this._monoScratch;
        this._monoScratchViewLen = nextFrames;
        this._scratchReallocations += 1;
      }
      if (
        this._monoScratchViewLen !== frameCount ||
        this._monoScratchView.buffer !== this._monoScratch.buffer
      ) {
        this._monoScratchView =
          frameCount === this._monoScratch.length
            ? this._monoScratch
            : this._monoScratch.subarray(0, frameCount);
        this._monoScratchViewLen = frameCount;
      }
      return this._monoScratchView;
    }

    _countActiveVoices() {
      let active = 0;
      const list = this._voiceList;
      for (let i = 0; i < list.length; i++) {
        const voice = list[i];
        if (!voice.stopped && !voice.finished) {
          active += 1;
        }
      }
      return active;
    }

    _updateActiveVoicePeak() {
      const active = this._countActiveVoices();
      if (active > this._activeVoicePeak) {
        this._activeVoicePeak = active;
      }
      return active;
    }

    get masterGain() {
      return this._masterGain;
    }

    get volume() {
      return this._masterGainUnmuted;
    }

    get muted() {
      return this._muted;
    }

    get effectiveGain() {
      return this._masterGain;
    }

    _scheduleMasterGainRamp(target, frames) {
      this._masterGainTarget = target;
      if (frames <= 0 || this._masterGain === target) {
        this._masterGain = target;
        this._masterGainRampRemaining = 0;
        this._masterGainRampStep = 0.0;
      } else {
        this._masterGainRampRemaining = frames;
        this._masterGainRampStep = (target - this._masterGain) / frames;
        this._gainRampsStarted += 1;
      }
    }

    setMasterGain(gain, rampFrames) {
      if (typeof gain !== 'number' || !Number.isFinite(gain)) {
        return this._masterGainTarget;
      }
      const target = gain < 0 ? 0.0 : gain;
      const frames = resolveRampFrames(
        rampFrames,
        this.sampleRate,
        this._defaultGainRampFrames
      );
      this._masterGainUnmuted = target;
      const effectiveTarget = this._muted ? 0.0 : target;
      this._scheduleMasterGainRamp(effectiveTarget, frames);
      return this._masterGainTarget;
    }

    /**
     * Smoothly ramps the master bus gain to `targetGain` over `rampFrames`
     * (defaults to `volumeRampFrames` / `DEFAULT_GAIN_RAMP_FRAMES`).
     *
     * @param {number} targetGain
     * @param {number|object} [rampFrames]
     * @returns {number}
     */
    rampMasterGain(targetGain, rampFrames) {
      const resolved = resolveRampFrames(
        rampFrames,
        this.sampleRate,
        this._volumeRampFrames
      );
      return this.setMasterGain(targetGain, resolved);
    }

    /**
     * Sets the logical master volume with smooth linear per-sample gain ramping
     * by default (`volumeRampFrames`, 64 frames @ 48 kHz) to prevent zipper noise
     * and clicks. Pass `rampFrames = 0` to snap immediately.
     *
     * @param {number} volume Non-negative finite volume multiplier.
     * @param {number|object} [rampFrames]
     * @returns {number} Updated logical master volume.
     */
    setVolume(volume, rampFrames) {
      if (typeof volume !== 'number' || !Number.isFinite(volume)) {
        return this._masterGainUnmuted;
      }
      const target = volume < 0 ? 0.0 : volume;
      const frames = resolveRampFrames(
        rampFrames,
        this.sampleRate,
        this._volumeRampFrames
      );
      this._masterGainUnmuted = target;
      if (!this._muted) {
        this._scheduleMasterGainRamp(target, frames);
      }
      return this._masterGainUnmuted;
    }

    /**
     * Mutes or unmutes the master bus using a smooth linear per-sample gain ramp
     * by default (`volumeRampFrames`, 64 frames @ 48 kHz) so toggling mute never
     * clicks or pops. Pass `rampFrames = 0` to snap immediately.
     *
     * @param {boolean} muted
     * @param {number|object} [rampFrames]
     * @returns {boolean} Updated mute state.
     */
    setMuted(muted, rampFrames) {
      const nextMuted = Boolean(muted);
      const frames = resolveRampFrames(
        rampFrames,
        this.sampleRate,
        this._volumeRampFrames
      );
      this._muted = nextMuted;
      const effectiveTarget = nextMuted ? 0.0 : this._masterGainUnmuted;
      this._scheduleMasterGainRamp(effectiveTarget, frames);
      return this._muted;
    }

    setMute(muted, rampFrames) {
      return this.setMuted(muted, rampFrames);
    }

    /**
     * Creates and starts a new mixer voice.
     *
     * @param {object} config
     * @param {Float32Array|ArrayBuffer|ArrayBufferView|number[]} config.samples
     * @param {number} [config.sampleRate]
     * @param {number} [config.channels=1] 1 (mono), 2 (stereo), 4 (quad), 6 (5.1), or 8 (7.1).
     * @param {number} [config.gain=1.0]
     * @param {number} [config.initialGain] Optional starting gain for fade-in ramps.
     * @param {number} [config.fadeInFrames=0] Optional fade-in ramp duration in output frames.
     * @param {number} [config.gainRampFrames] Optional default ramp duration for this voice.
     * @param {number} [config.pan=0.0] Equal-power pan in [-1, 1].
     * @param {boolean} [config.loop=false]
     * @param {number} [config.pitch=1.0] Playback rate multiplier (> 0).
     * @returns {number} Unique voice ID.
     */
    createVoice(config) {
      if (!config || typeof config !== 'object') {
        throw new TypeError('createVoice requires a configuration object');
      }
      if (this._voices.size >= this.maxVoices) {
        throw new RangeError(`Max voice limit (${this.maxVoices}) reached`);
      }

      const samples = toFloat32Samples(config.samples);
      const sampleRate =
        config.sampleRate !== undefined ? config.sampleRate : this.sampleRate;
      if (typeof sampleRate !== 'number' || !Number.isFinite(sampleRate) || sampleRate <= 0) {
        throw new RangeError(`voice sampleRate must be positive, got ${sampleRate}`);
      }

      const channels = config.channels !== undefined ? config.channels : 1;
      if (!Number.isInteger(channels) || !SUPPORTED_VOICE_CHANNELS.includes(channels)) {
        throw new RangeError(
          `voice channels must be one of ${SUPPORTED_VOICE_CHANNELS.join(', ')}, got ${channels}`
        );
      }

      const centerGain = validateDownmixGain(
        config.centerGain !== undefined ? config.centerGain : config.centerWeight,
        ITU_BS775_CENTER_GAIN,
        'voice centerGain'
      );
      const surroundGain = validateDownmixGain(
        config.surroundGain !== undefined ? config.surroundGain : config.surroundWeight,
        ITU_BS775_SURROUND_GAIN,
        'voice surroundGain'
      );
      const lfeGain = validateDownmixGain(
        config.lfeGain !== undefined ? config.lfeGain : config.lfeWeight,
        ITU_BS775_LFE_GAIN,
        'voice lfeGain'
      );

      const gain =
        typeof config.gain === 'number' && Number.isFinite(config.gain)
          ? Math.max(0.0, config.gain)
          : 1.0;

      const voiceGainRampFrames =
        config.gainRampFrames !== undefined
          ? resolveRampFrames(config.gainRampFrames, this.sampleRate, 0)
          : null;

      const fadeInFrames =
        config.fadeInFrames !== undefined
          ? resolveRampFrames(config.fadeInFrames, this.sampleRate, 0)
          : 0;

      const explicitInitialGain =
        config.initialGain !== undefined
          ? config.initialGain
          : config.startGain;

      let startGain = gain;
      let initRampFrames = 0;
      if (
        typeof explicitInitialGain === 'number' &&
        Number.isFinite(explicitInitialGain)
      ) {
        startGain = Math.max(0.0, explicitInitialGain);
        if (fadeInFrames > 0) {
          initRampFrames = fadeInFrames;
        } else if (voiceGainRampFrames !== null && voiceGainRampFrames > 0) {
          initRampFrames = voiceGainRampFrames;
        } else if (this._defaultGainRampFrames > 0) {
          initRampFrames = this._defaultGainRampFrames;
        } else if (startGain !== gain) {
          initRampFrames = DEFAULT_GAIN_RAMP_FRAMES;
        }
      } else if (fadeInFrames > 0) {
        startGain = 0.0;
        initRampFrames = fadeInFrames;
      }

      const initRampRemaining = startGain !== gain && initRampFrames > 0 ? initRampFrames : 0;
      const initRampStep = initRampRemaining > 0 ? (gain - startGain) / initRampRemaining : 0.0;

      const pan =
        typeof config.pan === 'number' && Number.isFinite(config.pan)
          ? clamp(config.pan, -1.0, 1.0)
          : 0.0;

      const pitch =
        config.pitch !== undefined ? config.pitch : 1.0;
      if (typeof pitch !== 'number' || !Number.isFinite(pitch) || pitch <= 0) {
        throw new RangeError(`voice pitch must be a positive finite number, got ${pitch}`);
      }
      const stepCheck = (sampleRate * pitch) / this.sampleRate;
      if (!Number.isFinite(stepCheck) || stepCheck <= 0) {
        throw new RangeError(
          `voice resampling step is non-finite or non-positive (sampleRate=${sampleRate}, pitch=${pitch})`
        );
      }

      const loop = Boolean(config.loop);
      const voiceMuted = Boolean(config.muted);
      const frameCount = Math.floor(samples.length / channels);
      const voiceId = this._nextVoiceId++;

      const effectiveStartGain = voiceMuted ? 0.0 : startGain;
      const effectiveTargetGain = voiceMuted ? 0.0 : gain;
      const effectiveRampRemaining = voiceMuted ? 0 : initRampRemaining;
      const effectiveRampStep = voiceMuted ? 0.0 : initRampStep;

      const voice = {
        id: voiceId,
        samples,
        frameCount,
        sampleRate,
        channels,
        centerGain,
        surroundGain,
        lfeGain,
        gain,
        unmutedGain: gain,
        muted: voiceMuted,
        currentGain: effectiveStartGain,
        targetGain: effectiveTargetGain,
        gainRampRemaining: effectiveRampRemaining,
        gainRampStep: effectiveRampStep,
        gainRampFrames: voiceGainRampFrames,
        pan,
        panL: Math.SQRT1_2,
        panR: Math.SQRT1_2,
        panLL: 1.0,
        panLR: 0.0,
        panRL: 0.0,
        panRR: 1.0,
        loop,
        pitch,
        position: 0.0,
        paused: false,
        stopped: false,
        finished: false,
      };
      updateVoicePanGains(voice);

      this._voices.set(voiceId, voice);
      this._voiceList.push(voice);

      this._updateActiveVoicePeak();
      return voiceId;
    }

    hasVoice(voiceId) {
      return this._voices.has(voiceId);
    }

    getVoiceGain(voiceId) {
      const voice = this._voices.get(voiceId);
      return voice ? voice.currentGain : null;
    }

    /**
     * Sets a voice's target gain with optional click-free linear ramping.
     *
     * @param {number} voiceId
     * @param {number} gain Non-negative target gain.
     * @param {number|object} [rampFrames] Optional ramp duration in output frames or `{ rampFrames | durationMs | durationSec }`.
     * @returns {boolean}
     */
    setVoiceGain(voiceId, gain, rampFrames) {
      const voice = this._voices.get(voiceId);
      if (!voice) return false;
      if (typeof gain !== 'number' || !Number.isFinite(gain)) {
        return false;
      }
      const target = gain < 0 ? 0.0 : gain;
      const defaultRamp =
        voice.gainRampFrames !== null
          ? voice.gainRampFrames
          : this._defaultGainRampFrames;
      const frames = resolveRampFrames(rampFrames, this.sampleRate, defaultRamp);

      voice.gain = target;
      voice.unmutedGain = target;
      if (voice.muted) {
        return true;
      }
      voice.targetGain = target;
      if (frames <= 0 || voice.currentGain === target) {
        voice.currentGain = target;
        voice.gainRampRemaining = 0;
        voice.gainRampStep = 0.0;
      } else {
        voice.gainRampRemaining = frames;
        voice.gainRampStep = (target - voice.currentGain) / frames;
        this._gainRampsStarted += 1;
      }
      return true;
    }

    /**
     * Smoothly mutes or unmutes a voice via linear per-sample gain ramping
     * (defaults to the voice's ramp length, mixer volumeRampFrames, or
     * `DEFAULT_GAIN_RAMP_FRAMES`). Pass `rampFrames = 0` to snap immediately.
     *
     * @param {number} voiceId
     * @param {boolean} muted
     * @param {number|object} [rampFrames]
     * @returns {boolean}
     */
    setVoiceMuted(voiceId, muted, rampFrames) {
      const voice = this._voices.get(voiceId);
      if (!voice) return false;
      const nextMuted = Boolean(muted);
      const fallback =
        voice.gainRampFrames !== null && voice.gainRampFrames > 0
          ? voice.gainRampFrames
          : this._volumeRampFrames;
      const frames = resolveRampFrames(rampFrames, this.sampleRate, fallback);

      voice.muted = nextMuted;
      const target = nextMuted ? 0.0 : voice.unmutedGain;
      voice.targetGain = target;
      if (frames <= 0 || voice.currentGain === target) {
        voice.currentGain = target;
        voice.gainRampRemaining = 0;
        voice.gainRampStep = 0.0;
      } else {
        voice.gainRampRemaining = frames;
        voice.gainRampStep = (target - voice.currentGain) / frames;
        this._gainRampsStarted += 1;
      }
      return true;
    }

    /**
     * Smoothly ramps a voice's gain from its current instantaneous gain to
     * `targetGain` over `rampFrames` output frames (defaults to voice/mixer
     * ramp length or `DEFAULT_GAIN_RAMP_FRAMES`).
     *
     * @param {number} voiceId
     * @param {number} targetGain
     * @param {number|object} [rampFrames]
     * @returns {boolean}
     */
    rampVoiceGain(voiceId, targetGain, rampFrames) {
      const voice = this._voices.get(voiceId);
      if (!voice) return false;
      const fallback =
        voice.gainRampFrames !== null && voice.gainRampFrames > 0
          ? voice.gainRampFrames
          : this._defaultGainRampFrames > 0
            ? this._defaultGainRampFrames
            : DEFAULT_GAIN_RAMP_FRAMES;
      const resolved = resolveRampFrames(rampFrames, this.sampleRate, fallback);
      return this.setVoiceGain(voiceId, targetGain, resolved);
    }

    setVoicePan(voiceId, pan) {
      const voice = this._voices.get(voiceId);
      if (!voice) return false;
      if (typeof pan !== 'number' || !Number.isFinite(pan)) {
        return false;
      }
      voice.pan = clamp(pan, -1.0, 1.0);
      updateVoicePanGains(voice);
      return true;
    }

    setVoicePitch(voiceId, pitch) {
      const voice = this._voices.get(voiceId);
      if (!voice) return false;
      if (typeof pitch !== 'number' || !Number.isFinite(pitch) || pitch <= 0) {
        throw new RangeError(`voice pitch must be a positive finite number, got ${pitch}`);
      }
      voice.pitch = pitch;
      return true;
    }

    pauseVoice(voiceId) {
      const voice = this._voices.get(voiceId);
      if (!voice) return false;
      voice.paused = true;
      return true;
    }

    resumeVoice(voiceId) {
      const voice = this._voices.get(voiceId);
      if (!voice) return false;
      voice.paused = false;
      voice.stopped = false;
      voice.finished = false;
      this._updateActiveVoicePeak();
      return true;
    }

    stopVoice(voiceId) {
      const voice = this._voices.get(voiceId);
      if (!voice) return false;
      voice.stopped = true;
      voice.paused = false;
      voice.position = 0.0;
      voice.currentGain = voice.targetGain;
      voice.gainRampRemaining = 0;
      voice.gainRampStep = 0.0;
      return true;
    }

    destroyVoice(voiceId) {
      const deleted = this._voices.delete(voiceId);
      if (deleted) {
        const list = this._voiceList;
        for (let i = 0; i < list.length; i++) {
          if (list[i].id === voiceId) {
            list.splice(i, 1);
            break;
          }
        }
      }
      return deleted;
    }

    /**
     * Records one or more underrun quanta in mixer telemetry (e.g. reported by
     * an AudioWorklet quantum callback or stream feeder).
     *
     * @param {number} [quanta=1]
     * @param {object} [options]
     * @param {boolean} [options.armRecovery=false] When true, arms a smooth cosine fade-in on recovery.
     * @returns {number} Updated total underrunQuanta count.
     */
    recordUnderrun(quanta = 1, options) {
      if (!Number.isInteger(quanta) || quanta < 0) {
        throw new RangeError(`quanta must be a non-negative integer, got ${quanta}`);
      }
      this._manualUnderrunQuanta += quanta;
      if (quanta > 0 && options && options.armRecovery) {
        this.armRecoveryFadeIn();
      }
      return this._ringUnderrunCount + this._manualUnderrunQuanta;
    }

    /**
     * Arms a smooth half-cosine fade-in (`0.0 -> 1.0`) for the next rendered
     * output frames after an underrun.
     *
     * @param {number|object} [rampFrames] Defaults to `underrunRampFrames` (64).
     * @returns {number} Armed fade-in frame count.
     */
    armRecoveryFadeIn(rampFrames) {
      const frames = resolveRampFrames(
        rampFrames,
        this.sampleRate,
        this._underrunRampFrames
      );
      if (frames > 0) {
        this._recoveryFadeTotal = frames;
        this._recoveryFadeRemaining = frames;
        this._recoveryFadeIns += 1;
      }
      return frames;
    }

    /**
     * Generates a click-free half-cosine fade-out from the last rendered stereo
     * sample `(lastOutL, lastOutR)` down to `0.0` over up to `underrunRampFrames`
     * frames and zero-fills any remaining frames, arming a smooth recovery fade-in.
     *
     * Pass a caller-supplied `Float32Array` as `targetBufferOrFrames` for zero
     * heap allocations, or a numeric frame count to return a new `Float32Array`.
     *
     * @param {Float32Array|number} targetBufferOrFrames
     * @param {number} [frameCount]
     * @returns {number|Float32Array} Frame count written (buffer mode) or new Float32Array (count mode).
     */
    renderUnderrunFadeOut(targetBufferOrFrames, frameCount) {
      let target;
      let frames;
      let returnBuffer = false;

      if (typeof targetBufferOrFrames === 'number') {
        if (!Number.isInteger(targetBufferOrFrames) || targetBufferOrFrames < 0) {
          throw new RangeError(`frameCount must be a non-negative integer, got ${targetBufferOrFrames}`);
        }
        frames = targetBufferOrFrames;
        target = new Float32Array(frames * 2);
        returnBuffer = true;
      } else if (targetBufferOrFrames instanceof Float32Array) {
        target = targetBufferOrFrames;
        frames = frameCount !== undefined ? frameCount : Math.floor(target.length / 2);
        if (!Number.isInteger(frames) || frames < 0) {
          throw new RangeError(`frameCount must be a non-negative integer, got ${frames}`);
        }
        if (target.length < frames * 2) {
          throw new RangeError(
            `targetBuffer length (${target.length}) is smaller than required ${frames * 2} samples`
          );
        }
      } else {
        throw new TypeError('renderUnderrunFadeOut expects a Float32Array or non-negative integer frameCount');
      }

      if (frames === 0) {
        return returnBuffer ? target : 0;
      }

      const lastL = this._lastOutL;
      const lastR = this._lastOutR;
      const fadeFrames =
        this._underrunRampFrames > 0
          ? frames < this._underrunRampFrames
            ? frames
            : this._underrunRampFrames
          : 0;

      if (fadeFrames > 0 && (lastL !== 0.0 || lastR !== 0.0)) {
        for (let f = 0; f < fadeFrames; f++) {
          const base = f * 2;
          target[base] = lastL;
          target[base + 1] = lastR;
        }
        applyFadeRamp(target, 2, fadeFrames, 1.0, 0.0);
        if (fadeFrames < frames) {
          target.fill(0.0, fadeFrames * 2, frames * 2);
        }
        this._underrunFadeOuts += 1;
        this._lastOutL = 0.0;
        this._lastOutR = 0.0;
      } else {
        target.fill(0.0, 0, frames * 2);
      }

      this.armRecoveryFadeIn();
      return returnBuffer ? target : frames;
    }

    /**
     * Applies an in-place zero-alloc half-cosine recovery fade-in chunk
     * spanning frames `[startPos .. startPos + rampCount - 1]` of `total`.
     *
     * @private
     */
    _applyCosineFadeInChunk(output, frameCount, startPos, total) {
      const remaining = total - startPos;
      const rampCount = frameCount < remaining ? frameCount : remaining;
      if (rampCount <= 0) {
        return 0;
      }
      if (startPos === 0 && rampCount === total) {
        applyFadeRamp(output, 2, total, 0.0, 1.0);
        return rampCount;
      }
      const lastIdx = total - 1;
      const invDenom = lastIdx > 0 ? Math.PI / lastIdx : 0.0;
      for (let f = 0; f < rampCount; f++) {
        const idx = startPos + f;
        const gain =
          idx === 0
            ? 0.0
            : idx >= lastIdx
              ? 1.0
              : 0.5 * (1.0 - Math.cos(idx * invDenom));
        const base = f * 2;
        if (gain === 0.0) {
          output[base] = 0.0;
          output[base + 1] = 0.0;
        } else if (gain !== 1.0) {
          output[base] *= gain;
          output[base + 1] *= gain;
        }
      }
      return rampCount;
    }

    /**
     * Applies an in-place zero-alloc half-cosine recovery fade-in chunk
     * spanning frames `[startPos .. startPos + rampCount - 1]` of `total`
     * onto planar stereo output buffers `(leftOut, rightOut)`.
     *
     * @private
     */
    _applyCosineFadeInChunkPlanar(leftOut, rightOut, frameCount, startPos, total) {
      const remaining = total - startPos;
      const rampCount = frameCount < remaining ? frameCount : remaining;
      if (rampCount <= 0) {
        return 0;
      }
      const lastIdx = total - 1;
      const invDenom = lastIdx > 0 ? Math.PI / lastIdx : 0.0;
      for (let f = 0; f < rampCount; f++) {
        const idx = startPos + f;
        const gain =
          total === 1
            ? 1.0
            : idx === 0
              ? 0.0
              : idx >= lastIdx
                ? 1.0
                : 0.5 * (1.0 - Math.cos(idx * invDenom));
        if (gain === 0.0) {
          leftOut[f] = 0.0;
          rightOut[f] = 0.0;
        } else if (gain !== 1.0) {
          leftOut[f] *= gain;
          rightOut[f] *= gain;
        }
      }
      return rampCount;
    }

    /**
     * Zero-allocation stereo bus voice accumulation pass. Clears and accumulates
     * all active voices into the pre-allocated `_busLeftScratch` and
     * `_busRightScratch` Float32Array buffers, reaps finished one-shot voices in
     * place, and records per-quantum scalar counters on `this`.
     *
     * @private
     */
    _accumulateVoicesToBus(frameCount) {
      const busL = this._busLeftScratch;
      const busR = this._busRightScratch;
      busL.fill(0.0, 0, frameCount);
      busR.fill(0.0, 0, frameCount);

      const finishedIds = this._finishedIds;
      finishedIds.length = 0;
      const outSampleRate = this.sampleRate;
      const voiceList = this._voiceList;
      const voiceCount = voiceList.length;
      let sanitized = 0;
      let downmixed = 0;
      let resampled = 0;

      this._updateActiveVoicePeak();

      for (let v = 0; v < voiceCount; v++) {
        const voice = voiceList[v];
        if (voice.paused || voice.stopped || voice.finished) {
          continue;
        }

        const srcFrames = voice.frameCount;
        if (srcFrames <= 0) {
          if (!voice.loop) {
            voice.finished = true;
            finishedIds.push(voice.id);
          }
          continue;
        }

        const src = voice.samples;
        const step = (voice.sampleRate * voice.pitch) / outSampleRate;
        const isResampling = voice.sampleRate !== outSampleRate || voice.pitch !== 1.0;
        const loop = voice.loop;
        let pos = voice.position;
        let curGain = voice.currentGain;
        const targetGain = voice.targetGain;
        let rampRem = voice.gainRampRemaining;
        const rampStep = voice.gainRampStep;
        let voiceRendered = 0;

        if (voice.channels === 1) {
          const panL = voice.panL;
          const panR = voice.panR;

          if (!isResampling && rampRem === 0 && pos === Math.floor(pos)) {
            // Fast 1:1 steady-gain mono path (no per-sample floor/frac/ramp checks).
            const gL = curGain * panL;
            const gR = curGain * panR;
            let f = 0;
            let idx = pos;
            while (f < frameCount) {
              if (idx >= srcFrames) {
                if (loop) {
                  idx %= srcFrames;
                } else {
                  voice.finished = true;
                  finishedIds.push(voice.id);
                  break;
                }
              }
              const avail = srcFrames - idx;
              const rem = frameCount - f;
              const chunk = rem < avail ? rem : avail;
              const fEnd = f + chunk;
              for (; f < fEnd; f++, idx++) {
                let sample = src[idx];
                if (sample - sample !== 0) {
                  sample = 0.0;
                  sanitized += 1;
                }
                busL[f] += sample * gL;
                busR[f] += sample * gR;
              }
              voiceRendered += chunk;
            }
            pos = idx;
          } else {
            for (let f = 0; f < frameCount; f++) {
              if (pos >= srcFrames) {
                if (loop) {
                  pos %= srcFrames;
                } else {
                  voice.finished = true;
                  finishedIds.push(voice.id);
                  break;
                }
              }

              if (rampRem > 0) {
                if (rampRem === 1) {
                  curGain = targetGain;
                  rampRem = 0;
                } else {
                  curGain += rampStep;
                  rampRem -= 1;
                }
              }

              const i0 = Math.floor(pos);
              const frac = pos - i0;
              let sample;
              if (frac === 0) {
                sample = src[i0];
              } else {
                let i1 = i0 + 1;
                if (i1 >= srcFrames) {
                  i1 = loop ? 0 : srcFrames - 1;
                }
                const s0 = src[i0];
                const s1 = src[i1];
                sample = s0 + (s1 - s0) * frac;
              }

              if (sample - sample !== 0) {
                sample = 0.0;
                sanitized += 1;
              }

              const scaled = sample * curGain;
              busL[f] += scaled * panL;
              busR[f] += scaled * panR;

              voiceRendered += 1;
              pos += step;
            }
          }

          if (!voice.finished && !loop && pos >= srcFrames) {
            voice.finished = true;
            finishedIds.push(voice.id);
          } else if (loop && pos >= srcFrames) {
            pos %= srcFrames;
          }
          voice.position = pos;
          voice.currentGain = curGain;
          voice.gainRampRemaining = rampRem;
          if (rampRem === 0) {
            voice.gainRampStep = 0.0;
          }
        } else if (voice.channels === 2) {
          // Stereo input with cached W3C Web Audio StereoPannerNode equal-power pan matrix.
          const ll = voice.panLL;
          const lr = voice.panLR;
          const rl = voice.panRL;
          const rr = voice.panRR;

          if (!isResampling && rampRem === 0 && pos === Math.floor(pos)) {
            let f = 0;
            let idx = pos;
            if (voice.pan === 0.0) {
              // Fast center-panned 1:1 stereo passthrough path.
              while (f < frameCount) {
                if (idx >= srcFrames) {
                  if (loop) {
                    idx %= srcFrames;
                  } else {
                    voice.finished = true;
                    finishedIds.push(voice.id);
                    break;
                  }
                }
                const avail = srcFrames - idx;
                const rem = frameCount - f;
                const chunk = rem < avail ? rem : avail;
                const fEnd = f + chunk;
                for (; f < fEnd; f++, idx++) {
                  const base0 = idx * 2;
                  let rawL = src[base0];
                  let rawR = src[base0 + 1];
                  if (rawL - rawL !== 0) {
                    rawL = 0.0;
                    sanitized += 1;
                  }
                  if (rawR - rawR !== 0) {
                    rawR = 0.0;
                    sanitized += 1;
                  }
                  busL[f] += rawL * curGain;
                  busR[f] += rawR * curGain;
                }
                voiceRendered += chunk;
              }
            } else {
              const gLL = curGain * ll;
              const gLR = curGain * lr;
              const gRL = curGain * rl;
              const gRR = curGain * rr;
              while (f < frameCount) {
                if (idx >= srcFrames) {
                  if (loop) {
                    idx %= srcFrames;
                  } else {
                    voice.finished = true;
                    finishedIds.push(voice.id);
                    break;
                  }
                }
                const avail = srcFrames - idx;
                const rem = frameCount - f;
                const chunk = rem < avail ? rem : avail;
                const fEnd = f + chunk;
                for (; f < fEnd; f++, idx++) {
                  const base0 = idx * 2;
                  let rawL = src[base0];
                  let rawR = src[base0 + 1];
                  if (rawL - rawL !== 0) {
                    rawL = 0.0;
                    sanitized += 1;
                  }
                  if (rawR - rawR !== 0) {
                    rawR = 0.0;
                    sanitized += 1;
                  }
                  busL[f] += rawL * gLL + rawR * gLR;
                  busR[f] += rawL * gRL + rawR * gRR;
                }
                voiceRendered += chunk;
              }
            }
            pos = idx;
          } else {
            for (let f = 0; f < frameCount; f++) {
              if (pos >= srcFrames) {
                if (loop) {
                  pos %= srcFrames;
                } else {
                  voice.finished = true;
                  finishedIds.push(voice.id);
                  break;
                }
              }

              if (rampRem > 0) {
                if (rampRem === 1) {
                  curGain = targetGain;
                  rampRem = 0;
                } else {
                  curGain += rampStep;
                  rampRem -= 1;
                }
              }

              const i0 = Math.floor(pos);
              const frac = pos - i0;
              const base0 = i0 * 2;
              let rawL;
              let rawR;
              if (frac === 0) {
                rawL = src[base0];
                rawR = src[base0 + 1];
              } else {
                let i1 = i0 + 1;
                if (i1 >= srcFrames) {
                  i1 = loop ? 0 : srcFrames - 1;
                }
                const base1 = i1 * 2;
                rawL = src[base0] + (src[base1] - src[base0]) * frac;
                rawR = src[base0 + 1] + (src[base1 + 1] - src[base0 + 1]) * frac;
              }

              if (rawL - rawL !== 0) {
                rawL = 0.0;
                sanitized += 1;
              }
              if (rawR - rawR !== 0) {
                rawR = 0.0;
                sanitized += 1;
              }

              const sL = rawL * curGain;
              const sR = rawR * curGain;

              busL[f] += sL * ll + sR * lr;
              busR[f] += sL * rl + sR * rr;

              voiceRendered += 1;
              pos += step;
            }
          }

          if (!voice.finished && !loop && pos >= srcFrames) {
            voice.finished = true;
            finishedIds.push(voice.id);
          } else if (loop && pos >= srcFrames) {
            pos %= srcFrames;
          }
          voice.position = pos;
          voice.currentGain = curGain;
          voice.gainRampRemaining = rampRem;
          if (rampRem === 0) {
            voice.gainRampStep = 0.0;
          }
        } else {
          // Multi-channel surround (4ch quad, 6ch 5.1, 8ch 7.1) ITU-R BS.775 downmix to stereo.
          const chCount = voice.channels;
          const centerGain = voice.centerGain;
          const surroundGain = voice.surroundGain;
          const lfeGain = voice.lfeGain;
          const ll = voice.panLL;
          const lr = voice.panLR;
          const rl = voice.panRL;
          const rr = voice.panRR;

          for (let f = 0; f < frameCount; f++) {
            if (pos >= srcFrames) {
              if (loop) {
                pos %= srcFrames;
              } else {
                voice.finished = true;
                finishedIds.push(voice.id);
                break;
              }
            }

            if (rampRem > 0) {
              if (rampRem === 1) {
                curGain = targetGain;
                rampRem = 0;
              } else {
                curGain += rampStep;
                rampRem -= 1;
              }
            }

            const i0 = Math.floor(pos);
            const frac = pos - i0;
            const base0 = i0 * chCount;
            let base1 = base0;
            if (frac !== 0) {
              let i1 = i0 + 1;
              if (i1 >= srcFrames) {
                i1 = loop ? 0 : srcFrames - 1;
              }
              base1 = i1 * chCount;
            }

            let rawL;
            let rawR;

            if (chCount === 6) {
              // 5.1: FL, FR, FC, LFE, BL, BR
              let fl = frac === 0 ? src[base0] : src[base0] + (src[base1] - src[base0]) * frac;
              let fr = frac === 0 ? src[base0 + 1] : src[base0 + 1] + (src[base1 + 1] - src[base0 + 1]) * frac;
              let fc = frac === 0 ? src[base0 + 2] : src[base0 + 2] + (src[base1 + 2] - src[base0 + 2]) * frac;
              let lfe = frac === 0 ? src[base0 + 3] : src[base0 + 3] + (src[base1 + 3] - src[base0 + 3]) * frac;
              let bl = frac === 0 ? src[base0 + 4] : src[base0 + 4] + (src[base1 + 4] - src[base0 + 4]) * frac;
              let br = frac === 0 ? src[base0 + 5] : src[base0 + 5] + (src[base1 + 5] - src[base0 + 5]) * frac;

              if (fl - fl !== 0) { fl = 0.0; sanitized += 1; }
              if (fr - fr !== 0) { fr = 0.0; sanitized += 1; }
              if (fc - fc !== 0) { fc = 0.0; sanitized += 1; }
              if (lfe - lfe !== 0) { lfe = 0.0; sanitized += 1; }
              if (bl - bl !== 0) { bl = 0.0; sanitized += 1; }
              if (br - br !== 0) { br = 0.0; sanitized += 1; }

              rawL = fl + fc * centerGain + bl * surroundGain + lfe * lfeGain;
              rawR = fr + fc * centerGain + br * surroundGain + lfe * lfeGain;
            } else if (chCount === 8) {
              // 7.1: FL, FR, FC, LFE, BL, BR, SL, SR
              let fl = frac === 0 ? src[base0] : src[base0] + (src[base1] - src[base0]) * frac;
              let fr = frac === 0 ? src[base0 + 1] : src[base0 + 1] + (src[base1 + 1] - src[base0 + 1]) * frac;
              let fc = frac === 0 ? src[base0 + 2] : src[base0 + 2] + (src[base1 + 2] - src[base0 + 2]) * frac;
              let lfe = frac === 0 ? src[base0 + 3] : src[base0 + 3] + (src[base1 + 3] - src[base0 + 3]) * frac;
              let bl = frac === 0 ? src[base0 + 4] : src[base0 + 4] + (src[base1 + 4] - src[base0 + 4]) * frac;
              let br = frac === 0 ? src[base0 + 5] : src[base0 + 5] + (src[base1 + 5] - src[base0 + 5]) * frac;
              let sl = frac === 0 ? src[base0 + 6] : src[base0 + 6] + (src[base1 + 6] - src[base0 + 6]) * frac;
              let sr = frac === 0 ? src[base0 + 7] : src[base0 + 7] + (src[base1 + 7] - src[base0 + 7]) * frac;

              if (fl - fl !== 0) { fl = 0.0; sanitized += 1; }
              if (fr - fr !== 0) { fr = 0.0; sanitized += 1; }
              if (fc - fc !== 0) { fc = 0.0; sanitized += 1; }
              if (lfe - lfe !== 0) { lfe = 0.0; sanitized += 1; }
              if (bl - bl !== 0) { bl = 0.0; sanitized += 1; }
              if (br - br !== 0) { br = 0.0; sanitized += 1; }
              if (sl - sl !== 0) { sl = 0.0; sanitized += 1; }
              if (sr - sr !== 0) { sr = 0.0; sanitized += 1; }

              rawL = fl + fc * centerGain + (bl + sl) * surroundGain + lfe * lfeGain;
              rawR = fr + fc * centerGain + (br + sr) * surroundGain + lfe * lfeGain;
            } else {
              // Quad (4ch): FL, FR, BL, BR
              let fl = frac === 0 ? src[base0] : src[base0] + (src[base1] - src[base0]) * frac;
              let fr = frac === 0 ? src[base0 + 1] : src[base0 + 1] + (src[base1 + 1] - src[base0 + 1]) * frac;
              let bl = frac === 0 ? src[base0 + 2] : src[base0 + 2] + (src[base1 + 2] - src[base0 + 2]) * frac;
              let br = frac === 0 ? src[base0 + 3] : src[base0 + 3] + (src[base1 + 3] - src[base0 + 3]) * frac;

              if (fl - fl !== 0) { fl = 0.0; sanitized += 1; }
              if (fr - fr !== 0) { fr = 0.0; sanitized += 1; }
              if (bl - bl !== 0) { bl = 0.0; sanitized += 1; }
              if (br - br !== 0) { br = 0.0; sanitized += 1; }

              rawL = fl + bl * surroundGain;
              rawR = fr + br * surroundGain;
            }

            if (rawL - rawL !== 0) { rawL = 0.0; sanitized += 1; }
            if (rawR - rawR !== 0) { rawR = 0.0; sanitized += 1; }

            const sL = rawL * curGain;
            const sR = rawR * curGain;

            busL[f] += sL * ll + sR * lr;
            busR[f] += sL * rl + sR * rr;

            downmixed += 1;
            voiceRendered += 1;
            pos += step;
          }

          if (!voice.finished && !loop && pos >= srcFrames) {
            voice.finished = true;
            finishedIds.push(voice.id);
          } else if (loop && pos >= srcFrames) {
            pos %= srcFrames;
          }
          voice.position = pos;
          voice.currentGain = curGain;
          voice.gainRampRemaining = rampRem;
          if (rampRem === 0) {
            voice.gainRampStep = 0.0;
          }
        }

        if (isResampling && voiceRendered > 0) {
          resampled += voiceRendered;
        }
      }

      // Automatically reap finished non-looping voices without allocating.
      if (finishedIds.length > 0) {
        for (let i = 0; i < finishedIds.length; i++) {
          this._voices.delete(finishedIds[i]);
        }
        let writeIdx = 0;
        for (let i = 0; i < voiceList.length; i++) {
          const voice = voiceList[i];
          if (!voice.finished) {
            voiceList[writeIdx++] = voice;
          }
        }
        voiceList.length = writeIdx;
        finishedIds.length = 0;
      }

      this._quantumSanitized = sanitized;
      this._quantumDownmixed = downmixed;
      this._quantumResampled = resampled;
    }

    /**
     * Zero-allocation core stereo mix pass supporting both interleaved stereo
     * output (`interleavedOut`) and planar stereo output (`leftOut`, `rightOut`).
     * Reuses pre-allocated `_busLeftScratch` and `_busRightScratch` Float32Array
     * buffers, applies per-voice and master gain/mute ramping, non-finite
     * sanitization, smooth soft-clip/hard-clip limiting, W272 half-cosine
     * recovery fade-in, and updates sanitized per-bus peak/RMS/clip telemetry.
     *
     * @private
     */
    _mixIntoCore(interleavedOut, leftOut, rightOut, frameCount, reallocsBefore) {
      if (frameCount === 0) {
        return 0;
      }

      const baselineReallocs =
        reallocsBefore !== undefined ? reallocsBefore : this._scratchReallocations;
      const sampleCount = frameCount * 2;
      this._ensureScratchCapacity(sampleCount);
      const zeroAllocQuantum = this._scratchReallocations === baselineReallocs;

      this._accumulateVoicesToBus(frameCount);

      const busL = this._busLeftScratch;
      const busR = this._busRightScratch;
      const softClip = this._softClip;
      const knee = this._limiterKnee;
      const negKnee = -knee;
      const headroom = 1.0 - knee;
      const hasRecoveryFade = this._recoveryFadeRemaining > 0;
      const isPlanar = leftOut !== null;

      let peakL = this._peakLeft;
      let peakR = this._peakRight;
      let sumSqL = 0.0;
      let sumSqR = 0.0;
      let clippedL = 0;
      let clippedR = 0;
      let sanitized = this._quantumSanitized;

      let mg = this._masterGain;
      const mgTarget = this._masterGainTarget;
      let mgRem = this._masterGainRampRemaining;
      const mgStep = this._masterGainRampStep;
      const rampingMaster = mgRem > 0;

      for (let f = 0; f < frameCount; f++) {
        if (rampingMaster && mgRem > 0) {
          if (mgRem === 1) {
            mg = mgTarget;
            mgRem = 0;
          } else {
            mg += mgStep;
            mgRem -= 1;
          }
        }

        let rawL = busL[f] * mg;
        if (rawL - rawL !== 0) {
          rawL = 0.0;
          sanitized += 1;
        } else if (mg === 0.0) {
          rawL = 0.0;
        }

        let rawR = busR[f] * mg;
        if (rawR - rawR !== 0) {
          rawR = 0.0;
          sanitized += 1;
        } else if (mg === 0.0) {
          rawR = 0.0;
        }

        let outL;
        let outR;
        if (softClip) {
          if (rawL > knee) {
            clippedL += 1;
            const compressed = knee + headroom * Math.tanh((rawL - knee) / headroom);
            outL = compressed > 1.0 ? 1.0 : compressed;
          } else if (rawL < negKnee) {
            clippedL += 1;
            const compressed = knee + headroom * Math.tanh((-rawL - knee) / headroom);
            outL = compressed > 1.0 ? -1.0 : -compressed;
          } else {
            outL = rawL;
          }

          if (rawR > knee) {
            clippedR += 1;
            const compressed = knee + headroom * Math.tanh((rawR - knee) / headroom);
            outR = compressed > 1.0 ? 1.0 : compressed;
          } else if (rawR < negKnee) {
            clippedR += 1;
            const compressed = knee + headroom * Math.tanh((-rawR - knee) / headroom);
            outR = compressed > 1.0 ? -1.0 : -compressed;
          } else {
            outR = rawR;
          }
        } else {
          if (rawL > 1.0) {
            clippedL += 1;
            outL = 1.0;
          } else if (rawL < -1.0) {
            clippedL += 1;
            outL = -1.0;
          } else {
            outL = rawL;
          }

          if (rawR > 1.0) {
            clippedR += 1;
            outR = 1.0;
          } else if (rawR < -1.0) {
            clippedR += 1;
            outR = -1.0;
          } else {
            outR = rawR;
          }
        }

        if (isPlanar) {
          leftOut[f] = outL;
          rightOut[f] = outR;
        } else {
          const outIdx = f * 2;
          interleavedOut[outIdx] = outL;
          interleavedOut[outIdx + 1] = outR;
        }

        if (!hasRecoveryFade) {
          const absL = outL < 0 ? -outL : outL;
          const absR = outR < 0 ? -outR : outR;
          if (absL > peakL) peakL = absL;
          if (absR > peakR) peakR = absR;
          sumSqL += outL * outL;
          sumSqR += outR * outR;
        }
      }

      if (rampingMaster) {
        this._masterGain = mg;
        this._masterGainRampRemaining = mgRem;
        if (mgRem === 0) {
          this._masterGainRampStep = 0.0;
        }
      }

      if (hasRecoveryFade) {
        const total = this._recoveryFadeTotal;
        const startPos = total - this._recoveryFadeRemaining;
        const applied = isPlanar
          ? this._applyCosineFadeInChunkPlanar(leftOut, rightOut, frameCount, startPos, total)
          : this._applyCosineFadeInChunk(interleavedOut, frameCount, startPos, total);
        this._recoveryFadeRemaining -= applied;
        if (this._recoveryFadeRemaining <= 0) {
          this._recoveryFadeRemaining = 0;
          this._recoveryFadeTotal = 0;
        }
        for (let f = 0; f < frameCount; f++) {
          const sL = isPlanar ? leftOut[f] : interleavedOut[f * 2];
          const sR = isPlanar ? rightOut[f] : interleavedOut[f * 2 + 1];
          const absL = sL < 0 ? -sL : sL;
          const absR = sR < 0 ? -sR : sR;
          if (absL > peakL) peakL = absL;
          if (absR > peakR) peakR = absR;
          sumSqL += sL * sL;
          sumSqR += sR * sR;
        }
      }

      if (isPlanar) {
        this._lastOutL = leftOut[frameCount - 1];
        this._lastOutR = rightOut[frameCount - 1];
      } else {
        const lastBase = (frameCount - 1) * 2;
        this._lastOutL = interleavedOut[lastBase];
        this._lastOutR = interleavedOut[lastBase + 1];
      }

      const clipped = clippedL + clippedR;
      this._peakLeft = peakL;
      this._peakRight = peakR;
      this._peakOutputLevel = peakL > peakR ? peakL : peakR;
      this._rmsLeftSumSq += sumSqL;
      this._rmsRightSumSq += sumSqR;
      this._rmsFrames += frameCount;
      this._clippedLeft += clippedL;
      this._clippedRight += clippedR;
      this._clippedSamples += clipped;
      if (softClip) {
        this._softClippedSamples += clipped;
      }
      this._sanitizedNonFiniteSamples += sanitized;
      this._downmixedFrames += this._quantumDownmixed;
      this._resampledFrames += this._quantumResampled;
      this._renderedFrames += frameCount;
      this._quantaMixed += 1;
      if (zeroAllocQuantum) {
        this._zeroAllocMixQuanta += 1;
      }

      return frameCount;
    }

    /**
     * Zero-allocation core stereo mix pass into an interleaved stereo Float32Array.
     *
     * @private
     */
    _mixIntoBuffer(output, frameCount, reallocsBefore) {
      return this._mixIntoCore(output, null, null, frameCount, reallocsBefore);
    }

    /**
     * Zero-allocation stereo mix pass directly into caller-supplied planar
     * left/right Float32Array buffers (`leftBuffer.length >= frameCount`,
     * `rightBuffer.length >= frameCount`), ideal for Web Audio AudioWorklet
     * stereo outputs.
     *
     * @param {Float32Array} leftBuffer Destination left channel buffer.
     * @param {Float32Array} rightBuffer Destination right channel buffer.
     * @param {number} [frameCount] Number of stereo frames to mix.
     * @returns {number} Number of frames mixed.
     */
    mixIntoPlanar(leftBuffer, rightBuffer, frameCount) {
      if (!(leftBuffer instanceof Float32Array) || !(rightBuffer instanceof Float32Array)) {
        throw new TypeError('leftBuffer and rightBuffer must be Float32Array instances');
      }
      const maxFrames =
        leftBuffer.length < rightBuffer.length ? leftBuffer.length : rightBuffer.length;
      const frames = frameCount !== undefined ? frameCount : maxFrames;
      if (!Number.isInteger(frames) || frames < 0) {
        throw new RangeError(`frameCount must be a non-negative integer, got ${frames}`);
      }
      if (leftBuffer.length < frames || rightBuffer.length < frames) {
        throw new RangeError(
          `Planar output buffers (${leftBuffer.length}, ${rightBuffer.length}) are smaller than required ${frames} frames`
        );
      }
      return this._mixIntoCore(null, leftBuffer, rightBuffer, frames, this._scratchReallocations);
    }

    /**
     * Zero-allocation stereo mix pass directly into a caller-supplied
     * interleaved stereo Float32Array (`targetBuffer.length >= frameCount * 2`).
     *
     * @param {Float32Array} targetBuffer Destination interleaved stereo buffer.
     * @param {number} [frameCount] Number of stereo frames to mix (defaults to `Math.floor(targetBuffer.length / 2)`).
     * @returns {number} Number of frames mixed.
     */
    mixInto(targetBuffer, frameCount) {
      if (!(targetBuffer instanceof Float32Array)) {
        throw new TypeError('targetBuffer must be a Float32Array');
      }
      const frames =
        frameCount !== undefined ? frameCount : Math.floor(targetBuffer.length / 2);
      if (!Number.isInteger(frames) || frames < 0) {
        throw new RangeError(`frameCount must be a non-negative integer, got ${frames}`);
      }
      const sampleCount = frames * 2;
      if (targetBuffer.length < sampleCount) {
        throw new RangeError(
          `targetBuffer length (${targetBuffer.length}) is smaller than required ${sampleCount} samples`
        );
      }
      return this._mixIntoBuffer(targetBuffer, frames);
    }

    /**
     * Alias for `mixInto(targetBuffer, frameCount)`.
     *
     * @param {Float32Array} targetBuffer
     * @param {number} [frameCount]
     * @returns {number}
     */
    renderFramesInto(targetBuffer, frameCount) {
      return this.mixInto(targetBuffer, frameCount);
    }

    /**
     * Renders `frameCount` interleaved stereo Float32 frames from all active voices,
     * applying linear resampling, per-voice gain ramping, equal-power panning,
     * master gain, and soft-clip Math.tanh limiting. Finished non-looping voices
     * are automatically reaped.
     *
     * Pass an optional pre-allocated `targetBuffer` (or construct with
     * `{ reuseOutputBuffer: true }`) for zero per-quantum output allocations.
     *
     * @param {number} frameCount
     * @param {Float32Array} [targetBuffer] Optional caller-owned destination buffer.
     * @returns {Float32Array} Interleaved stereo Float32Array of length frameCount * 2.
     */
    renderFrames(frameCount, targetBuffer) {
      if (!Number.isInteger(frameCount) || frameCount < 0) {
        throw new RangeError(`frameCount must be a non-negative integer, got ${frameCount}`);
      }

      const sampleCount = frameCount * 2;
      if (targetBuffer !== undefined) {
        if (!(targetBuffer instanceof Float32Array)) {
          throw new TypeError('targetBuffer must be a Float32Array');
        }
        if (targetBuffer.length < sampleCount) {
          throw new RangeError(
            `targetBuffer length (${targetBuffer.length}) is smaller than required ${sampleCount} samples`
          );
        }
        if (frameCount > 0) {
          this._mixIntoBuffer(targetBuffer, frameCount);
        }
        return targetBuffer.length === sampleCount
          ? targetBuffer
          : targetBuffer.subarray(0, sampleCount);
      }

      if (this._reuseOutputBuffer) {
        const reallocsBefore = this._scratchReallocations;
        const outView = this._getScratchOutputView(sampleCount);
        if (frameCount > 0) {
          this._mixIntoBuffer(outView, frameCount, reallocsBefore);
        }
        return outView;
      }

      const output = new Float32Array(sampleCount);
      if (frameCount === 0) {
        return output;
      }
      this._mixIntoBuffer(output, frameCount, this._scratchReallocations);
      return output;
    }

    /**
     * Renders frames using pre-allocated Float32Array scratch buffers (zero
     * per-quantum allocations) and writes them into an AudioRingBuffer (from
     * audio_worker.js or any compatible ring exposing pushSamples()/write()),
     * respecting available free ring space so it never overflows.
     *
     * @param {object} audioRingBuffer AudioRingBuffer instance.
     * @param {number} [maxFrames] Optional upper bound on frames to render and push.
     * @returns {number} Number of frames written to the ring buffer.
     */
    pumpToRing(audioRingBuffer, maxFrames) {
      if (!audioRingBuffer || typeof audioRingBuffer !== 'object') {
        throw new TypeError('pumpToRing requires an AudioRingBuffer instance');
      }

      const ringUnderruns =
        typeof audioRingBuffer.underrunCount === 'number' &&
        Number.isFinite(audioRingBuffer.underrunCount) &&
        audioRingBuffer.underrunCount >= 0
          ? Math.floor(audioRingBuffer.underrunCount)
          : typeof audioRingBuffer.underruns === 'number' &&
              Number.isFinite(audioRingBuffer.underruns) &&
              audioRingBuffer.underruns >= 0
            ? Math.floor(audioRingBuffer.underruns)
            : typeof audioRingBuffer.underrunQuanta === 'number' &&
                Number.isFinite(audioRingBuffer.underrunQuanta) &&
                audioRingBuffer.underrunQuanta >= 0
              ? Math.floor(audioRingBuffer.underrunQuanta)
              : null;
      let detectedRingUnderrun = false;
      if (ringUnderruns !== null) {
        if (ringUnderruns > this._lastObservedRingUnderruns) {
          this._ringUnderrunCount += ringUnderruns - this._lastObservedRingUnderruns;
          detectedRingUnderrun = true;
        }
        this._lastObservedRingUnderruns = ringUnderruns;
      }

      const hasPushSamples = typeof audioRingBuffer.pushSamples === 'function';
      const hasWrite = !hasPushSamples && typeof audioRingBuffer.write === 'function';
      if (!hasPushSamples && !hasWrite) {
        throw new TypeError('audioRingBuffer must implement pushSamples() or write()');
      }

      let free;
      if (typeof audioRingBuffer.freeFrames === 'function') {
        free = audioRingBuffer.freeFrames();
      } else if (typeof audioRingBuffer.availableWrite === 'function') {
        free = audioRingBuffer.availableWrite();
      } else if (
        typeof audioRingBuffer.capacityFrames === 'number' &&
        typeof audioRingBuffer.availableFrames === 'function'
      ) {
        free = Math.max(0, audioRingBuffer.capacityFrames - audioRingBuffer.availableFrames());
      } else if (typeof maxFrames === 'number') {
        free = maxFrames;
      } else {
        throw new TypeError('Cannot determine available free space on audioRingBuffer');
      }

      let framesToRender = Math.max(0, Math.floor(free));
      if (maxFrames !== undefined && maxFrames !== null) {
        if (!Number.isInteger(maxFrames) || maxFrames < 0) {
          throw new RangeError(`maxFrames must be a non-negative integer, got ${maxFrames}`);
        }
        framesToRender = Math.min(framesToRender, maxFrames);
      }

      if (detectedRingUnderrun && this._underrunRampFrames > 0) {
        this._pumpRecoveryFadeTotal = this._underrunRampFrames;
        this._pumpRecoveryFadeRemaining = this._underrunRampFrames;
        this._recoveryFadeIns += 1;
      }

      if (framesToRender <= 0) {
        return 0;
      }

      const ringChannels =
        Number.isInteger(audioRingBuffer.channels) && audioRingBuffer.channels > 0
          ? audioRingBuffer.channels
          : DEFAULT_CHANNELS;

      if (ringChannels !== 1 && ringChannels !== 2) {
        throw new RangeError(`Unsupported audioRingBuffer channel count: ${ringChannels}`);
      }

      const reallocsBefore = this._scratchReallocations;
      const outView = this._getScratchOutputView(framesToRender * 2);
      const monoView = ringChannels === 1 ? this._getMonoScratchView(framesToRender) : null;
      this._mixIntoBuffer(outView, framesToRender, reallocsBefore);

      if (this._pumpRecoveryFadeRemaining > 0) {
        const total = this._pumpRecoveryFadeTotal;
        const startPos = total - this._pumpRecoveryFadeRemaining;
        const applied = this._applyCosineFadeInChunk(outView, framesToRender, startPos, total);
        this._pumpRecoveryFadeRemaining -= applied;
        if (this._pumpRecoveryFadeRemaining <= 0) {
          this._pumpRecoveryFadeRemaining = 0;
          this._pumpRecoveryFadeTotal = 0;
        }
        const lastBase = (framesToRender - 1) * 2;
        this._lastOutL = outView[lastBase];
        this._lastOutR = outView[lastBase + 1];
      }

      if (ringChannels === 2) {
        return hasPushSamples
          ? audioRingBuffer.pushSamples(outView)
          : audioRingBuffer.write(outView);
      }

      for (let f = 0; f < framesToRender; f++) {
        const base = f * 2;
        monoView[f] = (outView[base] + outView[base + 1]) * 0.5;
      }
      return hasPushSamples
        ? audioRingBuffer.pushSamples(monoView)
        : audioRingBuffer.write(monoView);
    }

    /**
     * Clears all mixer telemetry and per-bus metering accumulators (`peakLeft`,
     * `peakRight`, `rmsLeft`, `rmsRight`, `clippedSamples`, `zeroAllocMixQuanta`,
     * underrun/fade counters, etc.) without disturbing active voices, gain/mute
     * state, or pre-allocated scratch buffers.
     */
    resetStats() {
      this._renderedFrames = 0;
      this._quantaMixed = 0;
      this._zeroAllocMixQuanta = 0;
      this._downmixedFrames = 0;
      this._resampledFrames = 0;
      this._ringUnderrunCount = 0;
      this._manualUnderrunQuanta = 0;
      this._underrunFadeOuts = 0;
      this._recoveryFadeIns = 0;
      this._peakLeft = 0.0;
      this._peakRight = 0.0;
      this._peakOutputLevel = 0.0;
      this._rmsLeftSumSq = 0.0;
      this._rmsRightSumSq = 0.0;
      this._rmsFrames = 0;
      this._clippedLeft = 0;
      this._clippedRight = 0;
      this._clippedSamples = 0;
      this._softClippedSamples = 0;
      this._sanitizedNonFiniteSamples = 0;
      this._scratchReallocations = 0;
      this._gainRampsStarted = 0;
      this._activeVoicePeak = this._countActiveVoices();
    }

    /**
     * Resets mixer voices (unless `{ clearVoices: false }`) and zeroes all mix
     * telemetry counters while retaining pre-allocated scratch buffers.
     *
     * @param {object} [options]
     * @param {boolean} [options.clearVoices=true] When true, clears all registered voices.
     */
    reset(options = {}) {
      const clearVoices =
        options && options.clearVoices !== undefined ? Boolean(options.clearVoices) : true;
      if (clearVoices) {
        this._voices.clear();
        this._voiceList.length = 0;
      }
      this._finishedIds.length = 0;
      if (this._mixScratch.length > 0) {
        this._mixScratch.fill(0.0);
      }
      if (this._outputScratch.length > 0) {
        this._outputScratch.fill(0.0);
      }
      if (this._monoScratch.length > 0) {
        this._monoScratch.fill(0.0);
      }
      const effectiveTarget = this._muted ? 0.0 : this._masterGainUnmuted;
      this._masterGain = effectiveTarget;
      this._masterGainTarget = effectiveTarget;
      this._masterGainRampRemaining = 0;
      this._masterGainRampStep = 0.0;
      this._lastObservedRingUnderruns = 0;
      this._recoveryFadeRemaining = 0;
      this._recoveryFadeTotal = 0;
      this._pumpRecoveryFadeRemaining = 0;
      this._pumpRecoveryFadeTotal = 0;
      this._lastOutL = 0.0;
      this._lastOutR = 0.0;
      this.resetStats();
    }

    /**
     * Returns sanitized mixer diagnostics and per-bus peak/RMS/clip telemetry.
     */
    getStats() {
      let activeVoices = 0;
      let playingVoices = 0;
      let pausedVoices = 0;
      const list = this._voiceList;
      for (let i = 0; i < list.length; i++) {
        const voice = list[i];
        if (!voice.stopped && !voice.finished) {
          activeVoices += 1;
          if (voice.paused) {
            pausedVoices += 1;
          } else {
            playingVoices += 1;
          }
        }
      }

      const totalUnderruns = sanitizeCount(this._ringUnderrunCount + this._manualUnderrunQuanta);
      const peakLeft = sanitizeUnitLevel(this._peakLeft);
      const peakRight = sanitizeUnitLevel(this._peakRight);
      const peakLevel = peakLeft > peakRight ? peakLeft : peakRight;
      const rmsFrames = sanitizeCount(this._rmsFrames);
      const rmsLeft =
        rmsFrames > 0
          ? sanitizeUnitLevel(Math.sqrt(sanitizeFiniteNonNeg(this._rmsLeftSumSq) / rmsFrames))
          : 0;
      const rmsRight =
        rmsFrames > 0
          ? sanitizeUnitLevel(Math.sqrt(sanitizeFiniteNonNeg(this._rmsRightSumSq) / rmsFrames))
          : 0;
      const clippedLeft = sanitizeCount(this._clippedLeft);
      const clippedRight = sanitizeCount(this._clippedRight);
      const clippedSamples = sanitizeCount(this._clippedSamples);
      const softClippedSamples = sanitizeCount(this._softClippedSamples);
      const sanitizedNonFiniteSamples = sanitizeCount(this._sanitizedNonFiniteSamples);
      const renderedFrames = sanitizeCount(this._renderedFrames);
      const activeVoicePeak = sanitizeCount(this._activeVoicePeak);
      const effectiveGain = sanitizeFiniteNonNeg(this._masterGain);

      return {
        activeVoices,
        playingVoices,
        pausedVoices,
        totalVoices: this._voices.size,
        maxVoices: this.maxVoices,
        sampleRate: this.sampleRate,
        channels: this.channels,
        masterGain: effectiveGain,
        masterGainTarget: sanitizeFiniteNonNeg(this._masterGainTarget),
        volume: sanitizeFiniteNonNeg(this._masterGainUnmuted),
        muted: Boolean(this._muted),
        effectiveGain,
        gainRampActive: this._masterGainRampRemaining > 0,
        gainRampsStarted: sanitizeCount(this._gainRampsStarted),
        softClip: this._softClip,
        limiterKnee: this._limiterKnee,
        gainRampFrames: this._defaultGainRampFrames,
        volumeRampFrames: this._volumeRampFrames,
        underrunRampFrames: this._underrunRampFrames,
        framesMixed: renderedFrames,
        renderedFrames,
        quantaMixed: sanitizeCount(this._quantaMixed),
        zeroAllocMixQuanta: sanitizeCount(this._zeroAllocMixQuanta),
        downmixedFrames: sanitizeCount(this._downmixedFrames),
        resampledFrames: sanitizeCount(this._resampledFrames),
        underrunQuanta: totalUnderruns,
        underrunCount: totalUnderruns,
        underrunFadeOuts: sanitizeCount(this._underrunFadeOuts),
        recoveryFadeIns: sanitizeCount(this._recoveryFadeIns),
        peakLeft,
        peakRight,
        rmsLeft,
        rmsRight,
        peakLevel,
        peakOutputLevel: peakLevel,
        clippedLeft,
        clippedRight,
        clippedSamples,
        softClippedSamples,
        sanitizedNonFiniteSamples,
        nanSanitizedSamples: sanitizedNonFiniteSamples,
        nonFiniteSanitized: sanitizedNonFiniteSamples,
        peakVoices: activeVoicePeak,
        activeVoicePeak,
        scratchCapacityFrames: this._mixScratch.length >>> 1,
        scratchReallocations: sanitizeCount(this._scratchReallocations),
      };
    }
  }

  const exportsObj = {
    AudioMixer,
    decodePcm16,
    decodeImaAdpcm,
    softClipSample,
    softLimitSample,
    sanitizeAndSoftLimitBuffer,
    applyFadeRamp,
    downmixToStereo,
    resampleLinear,
    DEFAULT_SAMPLE_RATE,
    MIN_SAMPLE_RATE,
    MAX_SAMPLE_RATE,
    MAX_RESAMPLE_RATIO,
    DEFAULT_MAX_RESAMPLE_FRAMES,
    DEFAULT_CHANNELS,
    DEFAULT_MAX_VOICES,
    DEFAULT_QUANTUM_FRAMES,
    DEFAULT_INITIAL_BUFFER_FRAMES,
    DEFAULT_GAIN_RAMP_FRAMES,
    DEFAULT_UNDERRUN_RAMP_FRAMES,
    DEFAULT_SOFT_CLIP_KNEE,
    DEFAULT_SOFT_LIMIT_KNEE,
    ITU_BS775_CENTER_GAIN,
    ITU_BS775_SURROUND_GAIN,
    ITU_BS775_LFE_GAIN,
    SUPPORTED_VOICE_CHANNELS,
    IMA_INDEX_TABLE,
    IMA_STEP_TABLE,
  };

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6AudioMixer = exportsObj;
  } else if (root) {
    root.PlayGta6AudioMixer = exportsObj;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportsObj;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
