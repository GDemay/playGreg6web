'use strict';

(function (root) {
  const DEFAULT_SAMPLE_RATE = 48000;
  const DEFAULT_CHANNELS = 2;
  const DEFAULT_MAX_VOICES = 64;
  const DEFAULT_SOFT_CLIP_KNEE = 0.95;
  const DEFAULT_LIMITER_KNEE = DEFAULT_SOFT_CLIP_KNEE;

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
    const angle = (p + 1.0) * (Math.PI * 0.25);
    return {
      left: Math.cos(angle),
      right: Math.sin(angle),
    };
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
   * Lock-free multi-voice stereo audio mixer with equal-power panning,
   * linear resampling, seamless looping, non-finite sample sanitization,
   * configurable soft-clip tanh limiter, and direct SharedArrayBuffer
   * AudioRingBuffer pumping.
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
        options.masterGain !== undefined ? options.masterGain : 1.0;
      if (typeof masterGain !== 'number' || !Number.isFinite(masterGain) || masterGain < 0) {
        throw new RangeError(`masterGain must be a non-negative finite number, got ${masterGain}`);
      }

      const softClip =
        options.softClip !== undefined ? Boolean(options.softClip) : true;

      const limiterKnee =
        options.limiterKnee !== undefined
          ? options.limiterKnee
          : options.softClipKnee !== undefined
            ? options.softClipKnee
            : DEFAULT_LIMITER_KNEE;
      if (
        typeof limiterKnee !== 'number' ||
        !Number.isFinite(limiterKnee) ||
        limiterKnee < 0 ||
        limiterKnee >= 1.0
      ) {
        throw new RangeError(`limiterKnee must be in [0, 1), got ${limiterKnee}`);
      }

      this.sampleRate = sampleRate;
      this.channels = DEFAULT_CHANNELS;
      this.maxVoices = maxVoices;
      this._masterGain = masterGain;
      this._softClip = softClip;
      this._limiterKnee = limiterKnee;

      this._voices = new Map();
      this._nextVoiceId = 1;
      this._mixScratch = new Float64Array(0);

      this._renderedFrames = 0;
      this._peakOutputLevel = 0.0;
      this._clippedSamples = 0;
      this._sanitizedNonFiniteSamples = 0;
      this._activeVoicePeak = 0;
    }

    _countActiveVoices() {
      let active = 0;
      for (const voice of this._voices.values()) {
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

    setMasterGain(gain) {
      if (typeof gain !== 'number' || !Number.isFinite(gain)) {
        return this._masterGain;
      }
      this._masterGain = gain < 0 ? 0.0 : gain;
      return this._masterGain;
    }

    /**
     * Creates and starts a new mixer voice.
     *
     * @param {object} config
     * @param {Float32Array|ArrayBuffer|ArrayBufferView|number[]} config.samples
     * @param {number} [config.sampleRate]
     * @param {number} [config.channels=1] 1 (mono) or 2 (interleaved stereo).
     * @param {number} [config.gain=1.0]
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
      if (channels !== 1 && channels !== 2) {
        throw new RangeError(`voice channels must be 1 or 2, got ${channels}`);
      }

      const gain =
        typeof config.gain === 'number' && Number.isFinite(config.gain)
          ? Math.max(0.0, config.gain)
          : 1.0;

      const pan =
        typeof config.pan === 'number' && Number.isFinite(config.pan)
          ? clamp(config.pan, -1.0, 1.0)
          : 0.0;

      const pitch =
        config.pitch !== undefined ? config.pitch : 1.0;
      if (typeof pitch !== 'number' || !Number.isFinite(pitch) || pitch <= 0) {
        throw new RangeError(`voice pitch must be a positive finite number, got ${pitch}`);
      }

      const loop = Boolean(config.loop);
      const frameCount = Math.floor(samples.length / channels);
      const voiceId = this._nextVoiceId++;

      this._voices.set(voiceId, {
        id: voiceId,
        samples,
        frameCount,
        sampleRate,
        channels,
        gain,
        pan,
        loop,
        pitch,
        position: 0.0,
        paused: false,
        stopped: false,
        finished: false,
      });

      this._updateActiveVoicePeak();
      return voiceId;
    }

    hasVoice(voiceId) {
      return this._voices.has(voiceId);
    }

    setVoiceGain(voiceId, gain) {
      const voice = this._voices.get(voiceId);
      if (!voice) return false;
      if (typeof gain !== 'number' || !Number.isFinite(gain)) {
        return false;
      }
      voice.gain = gain < 0 ? 0.0 : gain;
      return true;
    }

    setVoicePan(voiceId, pan) {
      const voice = this._voices.get(voiceId);
      if (!voice) return false;
      if (typeof pan !== 'number' || !Number.isFinite(pan)) {
        return false;
      }
      voice.pan = clamp(pan, -1.0, 1.0);
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
      return true;
    }

    destroyVoice(voiceId) {
      return this._voices.delete(voiceId);
    }

    /**
     * Renders `frameCount` interleaved stereo Float32 frames from all active voices,
     * applying linear resampling, equal-power panning, master gain, and soft-clip
     * Math.tanh limiting. Finished non-looping voices are automatically reaped.
     *
     * @param {number} frameCount
     * @returns {Float32Array} Interleaved stereo Float32Array of length frameCount * 2.
     */
    renderFrames(frameCount) {
      if (!Number.isInteger(frameCount) || frameCount < 0) {
        throw new RangeError(`frameCount must be a non-negative integer, got ${frameCount}`);
      }

      const sampleCount = frameCount * 2;
      const output = new Float32Array(sampleCount);
      if (frameCount === 0) {
        return output;
      }

      if (this._mixScratch.length < sampleCount) {
        this._mixScratch = new Float64Array(sampleCount);
      }
      const mix = this._mixScratch;
      mix.fill(0.0, 0, sampleCount);

      const finishedIds = [];
      const outSampleRate = this.sampleRate;
      let sanitized = 0;

      this._updateActiveVoicePeak();

      for (const voice of this._voices.values()) {
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
        const gain = voice.gain;
        const loop = voice.loop;
        let pos = voice.position;

        if (voice.channels === 1) {
          const panGains = computeMonoPanGains(voice.pan);
          const gL = gain * panGains.left;
          const gR = gain * panGains.right;

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

            if (!Number.isFinite(sample)) {
              sample = 0.0;
              sanitized += 1;
            }

            const outIdx = f * 2;
            mix[outIdx] += sample * gL;
            mix[outIdx + 1] += sample * gR;

            pos += step;
          }

          if (!voice.finished && !loop && pos >= srcFrames) {
            voice.finished = true;
            finishedIds.push(voice.id);
          } else if (loop && pos >= srcFrames) {
            pos %= srcFrames;
          }
          voice.position = pos;
        } else {
          // Stereo input with W3C Web Audio StereoPannerNode equal-power panning.
          const p = voice.pan;
          let ll, lr, rl, rr;
          if (p <= -1.0) {
            ll = 1.0; lr = 1.0; rl = 0.0; rr = 0.0;
          } else if (p >= 1.0) {
            ll = 0.0; lr = 0.0; rl = 1.0; rr = 1.0;
          } else if (p <= 0.0) {
            const x = (p + 1.0) * (Math.PI * 0.5);
            ll = 1.0;
            lr = Math.cos(x);
            rl = 0.0;
            rr = Math.sin(x);
          } else {
            const x = p * (Math.PI * 0.5);
            ll = Math.cos(x);
            lr = 0.0;
            rl = Math.sin(x);
            rr = 1.0;
          }

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

            if (!Number.isFinite(rawL)) {
              rawL = 0.0;
              sanitized += 1;
            }
            if (!Number.isFinite(rawR)) {
              rawR = 0.0;
              sanitized += 1;
            }

            const sL = rawL * gain;
            const sR = rawR * gain;

            const outIdx = f * 2;
            mix[outIdx] += sL * ll + sR * lr;
            mix[outIdx + 1] += sL * rl + sR * rr;

            pos += step;
          }

          if (!voice.finished && !loop && pos >= srcFrames) {
            voice.finished = true;
            finishedIds.push(voice.id);
          } else if (loop && pos >= srcFrames) {
            pos %= srcFrames;
          }
          voice.position = pos;
        }
      }

      // Automatically reap finished non-looping voices.
      for (let i = 0; i < finishedIds.length; i++) {
        this._voices.delete(finishedIds[i]);
      }

      // Apply master gain, non-finite sanitizer, soft-clip/hard-clip limiter, and update metering stats.
      const masterGain = this._masterGain;
      const softClip = this._softClip;
      const knee = this._limiterKnee;
      let peak = this._peakOutputLevel;
      let clipped = 0;

      for (let i = 0; i < sampleCount; i++) {
        let raw = mix[i] * masterGain;
        if (!Number.isFinite(raw)) {
          raw = 0.0;
          sanitized += 1;
        }

        let limited;
        if (softClip) {
          if (raw > knee || raw < -knee) {
            clipped += 1;
          }
          limited = softClipSample(raw, knee);
        } else {
          if (raw > 1.0 || raw < -1.0) {
            clipped += 1;
          }
          limited = clamp(raw, -1.0, 1.0);
        }

        const absVal = Math.abs(limited);
        if (absVal > peak) {
          peak = absVal;
        }
        output[i] = limited;
      }

      this._peakOutputLevel = peak;
      this._clippedSamples += clipped;
      this._sanitizedNonFiniteSamples += sanitized;
      this._renderedFrames += frameCount;

      return output;
    }

    /**
     * Renders frames and writes them into an AudioRingBuffer (from audio_worker.js
     * or any compatible ring exposing pushSamples()/write()), respecting available
     * free ring space so it never overflows.
     *
     * @param {object} audioRingBuffer AudioRingBuffer instance.
     * @param {number} [maxFrames] Optional upper bound on frames to render and push.
     * @returns {number} Number of frames written to the ring buffer.
     */
    pumpToRing(audioRingBuffer, maxFrames) {
      if (!audioRingBuffer || typeof audioRingBuffer !== 'object') {
        throw new TypeError('pumpToRing requires an AudioRingBuffer instance');
      }

      const writer =
        typeof audioRingBuffer.pushSamples === 'function'
          ? (buf) => audioRingBuffer.pushSamples(buf)
          : typeof audioRingBuffer.write === 'function'
            ? (buf) => audioRingBuffer.write(buf)
            : null;

      if (!writer) {
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

      if (framesToRender <= 0) {
        return 0;
      }

      const rendered = this.renderFrames(framesToRender);
      const ringChannels =
        Number.isInteger(audioRingBuffer.channels) && audioRingBuffer.channels > 0
          ? audioRingBuffer.channels
          : DEFAULT_CHANNELS;

      if (ringChannels === 2) {
        return writer(rendered);
      }

      if (ringChannels === 1) {
        const mono = new Float32Array(framesToRender);
        for (let f = 0; f < framesToRender; f++) {
          mono[f] = (rendered[f * 2] + rendered[f * 2 + 1]) * 0.5;
        }
        return writer(mono);
      }

      throw new RangeError(`Unsupported audioRingBuffer channel count: ${ringChannels}`);
    }

    /**
     * Resets mixer voices (unless `{ clearVoices: false }`) and zeroes all mix telemetry counters.
     *
     * @param {object} [options]
     * @param {boolean} [options.clearVoices=true] When true, clears all registered voices.
     */
    reset(options = {}) {
      const clearVoices = options && options.clearVoices !== undefined ? Boolean(options.clearVoices) : true;
      if (clearVoices) {
        this._voices.clear();
      }
      if (this._mixScratch.length > 0) {
        this._mixScratch.fill(0.0);
      }
      this._renderedFrames = 0;
      this._peakOutputLevel = 0.0;
      this._clippedSamples = 0;
      this._sanitizedNonFiniteSamples = 0;
      this._activeVoicePeak = clearVoices ? 0 : this._countActiveVoices();
    }

    /**
     * Returns mixer diagnostics and voice metering stats.
     */
    getStats() {
      let activeVoices = 0;
      let playingVoices = 0;
      let pausedVoices = 0;
      for (const voice of this._voices.values()) {
        if (!voice.stopped && !voice.finished) {
          activeVoices += 1;
          if (voice.paused) {
            pausedVoices += 1;
          } else {
            playingVoices += 1;
          }
        }
      }

      return {
        activeVoices,
        playingVoices,
        pausedVoices,
        totalVoices: this._voices.size,
        maxVoices: this.maxVoices,
        sampleRate: this.sampleRate,
        channels: this.channels,
        masterGain: this._masterGain,
        softClip: this._softClip,
        limiterKnee: this._limiterKnee,
        renderedFrames: this._renderedFrames,
        peakLevel: this._peakOutputLevel,
        peakOutputLevel: this._peakOutputLevel,
        clippedSamples: this._clippedSamples,
        sanitizedNonFiniteSamples: this._sanitizedNonFiniteSamples,
        activeVoicePeak: this._activeVoicePeak,
      };
    }
  }

  const exportsObj = {
    AudioMixer,
    decodePcm16,
    decodeImaAdpcm,
    softClipSample,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_CHANNELS,
    DEFAULT_MAX_VOICES,
    DEFAULT_SOFT_CLIP_KNEE,
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
