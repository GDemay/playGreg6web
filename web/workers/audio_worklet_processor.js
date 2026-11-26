'use strict';

(function (root) {
  const PROCESSOR_NAME = 'playgta6-audio-processor';
  const DEFAULT_CHANNELS = 2;
  const DEFAULT_SMOOTHING_FRAMES = 64;
  const DEFAULT_DECLICK_FRAMES = 32;
  const DEFAULT_FADE_IN_FRAMES = DEFAULT_DECLICK_FRAMES;
  const DEFAULT_UNDERRUN_RAMP_FRAMES = 64;
  const MAX_DECLICK_FRAMES = 4096;
  const DEFAULT_SAMPLE_CLAMP_LIMIT = 1.0;
  const DEFAULT_SOFT_LIMIT_KNEE = 0.9;
  const MAX_TRACKED_CHANNELS = 32;

  // W327: Overrun drift guard & lock-free watermark telemetry defaults (48 kHz stereo).
  const DEFAULT_MAX_BUFFERED_FRAMES = 0; // 0 = disabled unless explicitly configured
  const DEFAULT_HIGH_WATERMARK_FRAMES = 4800; // 100 ms @ 48 kHz (3 video frames @ 30 FPS)
  const DEFAULT_TARGET_WATERMARK_FRAMES = 1440; // 30 ms @ 48 kHz steady-state target
  const DEFAULT_TRIM_CROSSFADE_FRAMES = 32; // ~0.67 ms @ 48 kHz click-free splice
  const MAX_TRIM_CROSSFADE_FRAMES = 512;
  const DEFAULT_STATS_INTERVAL_QUANTA = 0; // 0 = periodic port stats disabled by default

  // Atomic Int32 header layout at byte offset 0..32 of the SharedArrayBuffer.
  // Must match web/workers/audio_worker.js byte-for-byte.
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

  function clampGain(value) {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      return 1.0;
    }
    if (value <= 0.0) return 0.0;
    if (value >= 1.0) return 1.0;
    return value;
  }

  function sanitizeSmoothingFrames(value) {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      return DEFAULT_SMOOTHING_FRAMES;
    }
    const n = Math.floor(value);
    return n < 0 ? 0 : n;
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

  function sanitizeClampLimit(value, fallback) {
    if (value === true) {
      return DEFAULT_SAMPLE_CLAMP_LIMIT;
    }
    if (value === false || value === null || value === 0) {
      return 0.0;
    }
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
      return fallback;
    }
    return value;
  }

  function sanitizeFadeCurve(value, fallback) {
    if (value === 'cosine' || value === 'linear') {
      return value;
    }
    return fallback;
  }

  function sanitizeSoftLimitKnee(value, fallback) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value >= 1.0) {
      return fallback;
    }
    return value;
  }

  function clampFillRatio(value) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0.0) {
      return 0.0;
    }
    if (value >= 1.0) {
      return 1.0;
    }
    return value;
  }

  function sanitizeNonNegInt(value, fallback) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
      return fallback;
    }
    return Math.floor(value);
  }

  function sanitizeTrimCrossfadeFrames(value, fallback) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
      return fallback;
    }
    const n = Math.floor(value);
    return n > MAX_TRIM_CROSSFADE_FRAMES ? MAX_TRIM_CROSSFADE_FRAMES : n;
  }

  function softLimitPositive(absVal, knee, headroom) {
    const compressed = knee + headroom * Math.tanh((absVal - knee) / headroom);
    return compressed > 1.0 ? 1.0 : compressed;
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

  /**
   * Sanitizes non-finite samples (`NaN` / `±Infinity`) to `0.0`, passes normal
   * audio `|x| <= knee` (default `0.9`) bit-exact linearly, and smoothly
   * soft-clips `|x| > knee` into `[-1.0, 1.0]` using a C1-continuous `tanh` knee.
   *
   * @param {number} x Input sample value.
   * @param {number} [knee=0.9] Linear passthrough threshold in `[0, 1)`.
   * @returns {number} Finite sample strictly bounded in `[-1.0, 1.0]`.
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
    const bounded = softLimitPositive(absX, knee, headroom);
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
        nonFiniteSanitized++;
      } else if (s > knee) {
        samples[i] = softLimitPositive(s, knee, headroom);
        softClippedSamples++;
      } else if (s < negKnee) {
        samples[i] = -softLimitPositive(-s, knee, headroom);
        softClippedSamples++;
      }
    }

    if (statsTarget) {
      statsTarget.softClippedSamples = (statsTarget.softClippedSamples || 0) + softClippedSamples;
      statsTarget.nonFiniteSanitized = (statsTarget.nonFiniteSanitized || 0) + nonFiniteSanitized;
      return statsTarget;
    }

    return { softClippedSamples, nonFiniteSanitized };
  }

  const BaseAudioWorkletProcessor =
    typeof AudioWorkletProcessor === 'function'
      ? AudioWorkletProcessor
      : class FallbackAudioWorkletProcessor {
          constructor() {
            this.port = {
              onmessage: null,
              postMessage() {},
            };
          }
        };

  /**
   * Zero-GC AudioWorkletProcessor that reads interleaved Float32 PCM frames
   * directly from the lock-free SharedArrayBuffer ring buffer defined in
   * web/workers/audio_worker.js and de-interleaves them into WebAudio planar
   * output channels.
   */
  class PlayGta6AudioWorkletProcessor extends BaseAudioWorkletProcessor {
    constructor(options = {}) {
      super(options);

      const procOpts =
        options && options.processorOptions && typeof options.processorOptions === 'object'
          ? options.processorOptions
          : {};

      this._sab = null;
      this._ctrl = null;
      this._samples = null;
      this._capacityFrames = 0;
      this._channels = DEFAULT_CHANNELS;
      this._modulus = 0;

      this._gain = clampGain(procOpts.gain !== undefined ? procOpts.gain : 1.0);
      this._muted = Boolean(procOpts.muted);
      this._smoothingFrames = sanitizeSmoothingFrames(procOpts.smoothingFrames);

      const explicitCosine =
        procOpts.cosineRamp === true ||
        procOpts.fadeCurve === 'cosine' ||
        procOpts.underrunRampFrames !== undefined;
      this._fadeCurve = sanitizeFadeCurve(
        procOpts.fadeCurve,
        explicitCosine ? 'cosine' : 'linear'
      );

      let rawDeclick;
      if (procOpts.declickFrames !== undefined) {
        rawDeclick = procOpts.declickFrames;
      } else if (procOpts.underrunFadeFrames !== undefined) {
        rawDeclick = procOpts.underrunFadeFrames;
      } else if (procOpts.underrunRampFrames !== undefined) {
        rawDeclick =
          procOpts.underrunRampFrames === true
            ? DEFAULT_UNDERRUN_RAMP_FRAMES
            : procOpts.underrunRampFrames;
      } else if (explicitCosine) {
        rawDeclick = DEFAULT_UNDERRUN_RAMP_FRAMES;
      }
      this._declickFrames = sanitizeDeclickFrames(rawDeclick, 0);

      const rawFadeIn =
        procOpts.fadeInFrames !== undefined
          ? procOpts.fadeInFrames
          : procOpts.recoveryFadeFrames;
      this._fadeInFramesOverride =
        rawFadeIn !== undefined && rawFadeIn !== null
          ? sanitizeDeclickFrames(rawFadeIn, -1)
          : -1;

      let clampLimit = 0.0;
      const rawClampLimit =
        procOpts.sampleClampLimit !== undefined
          ? procOpts.sampleClampLimit
          : procOpts.clampLimit;
      if (rawClampLimit !== undefined) {
        clampLimit = sanitizeClampLimit(rawClampLimit, 0.0);
      } else if (procOpts.clampSamples !== undefined) {
        clampLimit = sanitizeClampLimit(procOpts.clampSamples, 0.0);
      }
      this._clampLimit = clampLimit;

      let softLimitEnabled = false;
      let softLimitKnee = DEFAULT_SOFT_LIMIT_KNEE;
      const rawSoftKnee =
        procOpts.softLimitKnee !== undefined
          ? procOpts.softLimitKnee
          : procOpts.limiterKnee;
      if (rawSoftKnee !== undefined && rawSoftKnee !== null) {
        softLimitKnee = sanitizeSoftLimitKnee(rawSoftKnee, DEFAULT_SOFT_LIMIT_KNEE);
        softLimitEnabled = true;
      }
      const rawSoftLimit =
        procOpts.softLimit !== undefined ? procOpts.softLimit : procOpts.softClip;
      if (rawSoftLimit !== undefined) {
        if (typeof rawSoftLimit === 'number') {
          softLimitKnee = sanitizeSoftLimitKnee(rawSoftLimit, DEFAULT_SOFT_LIMIT_KNEE);
          softLimitEnabled = true;
        } else {
          softLimitEnabled = Boolean(rawSoftLimit);
        }
      }
      this._softLimitEnabled = softLimitEnabled;
      this._softLimitKnee = softLimitKnee;

      this._targetGain = this._muted ? 0.0 : this._gain;
      this._currentGain = this._targetGain;
      this._rampStep = 0.0;
      this._rampRemaining = 0;

      // Pre-allocated fixed-size per-channel sample history for zero-GC underrun fade-out & recovery fade-in.
      this._lastSamples = new Float32Array(MAX_TRACKED_CHANNELS);
      this._fadeInFrom = new Float32Array(MAX_TRACKED_CHANNELS);
      this._hasActiveAudio = false;
      this._inUnderrun = false;
      this._declickPos = 0;
      this._declickTotal = 0;
      this._declickFadeOuts = 0;
      this._fadeInPos = 0;
      this._fadeInTotal = 0;
      this._fadeInHasOffset = false;
      this._fadeInCount = 0;

      this._stopped = false;
      this._processedQuanta = 0;
      this._processedFrames = 0;
      this._sanitizedSamples = 0;
      this._clampedSamples = 0;
      this._softClippedSamples = 0;
      this._underruns = 0;
      this._underrunQuanta = 0;
      this._underrunFrames = 0;

      // W327: Overrun drift guard (high-watermark trim) & lock-free telemetry state.
      let maxBufferedFrames = DEFAULT_MAX_BUFFERED_FRAMES;
      let targetWatermarkOverride = -1;
      const rawMaxBuffered =
        procOpts.maxBufferedFrames !== undefined
          ? procOpts.maxBufferedFrames
          : procOpts.highWatermarkFrames;
      if (rawMaxBuffered === true) {
        maxBufferedFrames = DEFAULT_HIGH_WATERMARK_FRAMES;
        targetWatermarkOverride = DEFAULT_TARGET_WATERMARK_FRAMES;
      } else if (rawMaxBuffered !== undefined && rawMaxBuffered !== null && rawMaxBuffered !== false) {
        maxBufferedFrames = sanitizeNonNegInt(rawMaxBuffered, DEFAULT_MAX_BUFFERED_FRAMES);
      } else if (procOpts.overrunGuard === true) {
        maxBufferedFrames = DEFAULT_HIGH_WATERMARK_FRAMES;
        targetWatermarkOverride = DEFAULT_TARGET_WATERMARK_FRAMES;
      }

      const rawTargetWatermark =
        procOpts.targetWatermarkFrames !== undefined
          ? procOpts.targetWatermarkFrames
          : procOpts.targetBufferedFrames !== undefined
            ? procOpts.targetBufferedFrames
            : procOpts.lowWatermarkFrames;
      if (rawTargetWatermark !== undefined && rawTargetWatermark !== null) {
        targetWatermarkOverride = sanitizeNonNegInt(rawTargetWatermark, targetWatermarkOverride);
      }

      const rawTrimCrossfade =
        procOpts.trimCrossfadeFrames !== undefined
          ? procOpts.trimCrossfadeFrames
          : procOpts.overrunCrossfadeFrames;
      const trimCrossfadeFrames =
        rawTrimCrossfade !== undefined && rawTrimCrossfade !== null
          ? sanitizeTrimCrossfadeFrames(rawTrimCrossfade, DEFAULT_TRIM_CROSSFADE_FRAMES)
          : DEFAULT_TRIM_CROSSFADE_FRAMES;

      const rawStatsInterval =
        procOpts.statsIntervalQuanta !== undefined
          ? procOpts.statsIntervalQuanta
          : procOpts.telemetryIntervalQuanta;
      const statsIntervalQuanta =
        rawStatsInterval !== undefined && rawStatsInterval !== null
          ? sanitizeNonNegInt(rawStatsInterval, DEFAULT_STATS_INTERVAL_QUANTA)
          : DEFAULT_STATS_INTERVAL_QUANTA;

      this._maxBufferedFrames = maxBufferedFrames;
      this._targetWatermarkFramesOverride = targetWatermarkOverride;
      this._trimCrossfadeFrames = trimCrossfadeFrames;
      this._overrunTrimmedFrames = 0;
      this._overrunTrimCount = 0;
      this._lastFillRatio = 0.0;
      this._peakFillRatio = 0.0;
      this._statsIntervalQuanta = statsIntervalQuanta;
      this._statsQuantaCounter = 0;

      // Pre-allocated reusable telemetry object for zero-allocation periodic port stats in process().
      this._periodicStatsPayload = {
        type: 'stats',
        periodic: true,
        attached: false,
        capacityFrames: 0,
        channels: DEFAULT_CHANNELS,
        bufferedFrames: 0,
        fillRatio: 0.0,
        peakFillRatio: 0.0,
        maxBufferedFrames: this._maxBufferedFrames,
        targetWatermarkFrames: this.targetWatermarkFrames,
        trimCrossfadeFrames: this._trimCrossfadeFrames,
        overrunTrimmedFrames: 0,
        overrunTrimCount: 0,
        underruns: 0,
        underrunQuanta: 0,
        underrunFrames: 0,
        inUnderrun: false,
        fadeOutCount: 0,
        underrunFadeOuts: 0,
        fadeInCount: 0,
        recoveryFadeIns: 0,
        overflows: 0,
        overruns: 0,
        droppedFrames: 0,
        quantaProcessed: 0,
        processedQuanta: 0,
        processedFrames: 0,
        framesProcessed: 0,
        sanitizedSamples: 0,
        nonFiniteSanitized: 0,
        clampedSamples: 0,
        softClippedSamples: 0,
      };

      if (procOpts.sab !== undefined && procOpts.sab !== null) {
        this._attachSab(procOpts.sab);
      }

      if (this.port) {
        this.port.onmessage = (event) => {
          this._handleMessage(event ? event.data : null);
        };
      }
    }

    get framesProcessed() {
      return this._processedFrames;
    }

    get processedFrames() {
      return this._processedFrames;
    }

    get sanitizedSamples() {
      return this._sanitizedSamples;
    }

    get nonFiniteSanitized() {
      return this._sanitizedSamples;
    }

    get clampedSamples() {
      return this._clampedSamples;
    }

    get softClippedSamples() {
      return this._softClippedSamples;
    }

    get softLimit() {
      return this._softLimitEnabled;
    }

    get softLimitKnee() {
      return this._softLimitKnee;
    }

    get clampSamples() {
      return this._clampLimit > 0.0;
    }

    get clampLimit() {
      return this._clampLimit;
    }

    get underruns() {
      return this._ctrl ? Atomics.load(this._ctrl, HDR_UNDERRUNS) : this._underruns;
    }

    get underrunFrames() {
      return this._ctrl ? Atomics.load(this._ctrl, HDR_UNDERRUN_FRAMES) : this._underrunFrames;
    }

    get declickFrames() {
      return this._declickFrames;
    }

    get fadeCurve() {
      return this._fadeCurve;
    }

    get fadeInFrames() {
      return this._fadeInFramesOverride >= 0 ? this._fadeInFramesOverride : this._declickFrames;
    }

    get fadeOutCount() {
      return this._declickFadeOuts;
    }

    get underrunFadeOuts() {
      return this._declickFadeOuts;
    }

    get fadeInCount() {
      return this._fadeInCount;
    }

    get recoveryFadeIns() {
      return this._fadeInCount;
    }

    get inUnderrun() {
      return this._inUnderrun;
    }

    get quantaProcessed() {
      return this._processedQuanta;
    }

    get processedQuanta() {
      return this._processedQuanta;
    }

    get underrunQuanta() {
      return this._underrunQuanta;
    }

    get fillRatio() {
      if (!this._ctrl || this._capacityFrames <= 0) {
        return 0.0;
      }
      return clampFillRatio(this._computeBufferedFrames() / this._capacityFrames);
    }

    get peakFillRatio() {
      return this._peakFillRatio;
    }

    get overrunTrimmedFrames() {
      return this._overrunTrimmedFrames;
    }

    get overrunTrimCount() {
      return this._overrunTrimCount;
    }

    get maxBufferedFrames() {
      return this._maxBufferedFrames;
    }

    get targetWatermarkFrames() {
      if (this._maxBufferedFrames <= 0) {
        return 0;
      }
      const raw =
        this._targetWatermarkFramesOverride >= 0
          ? this._targetWatermarkFramesOverride
          : Math.floor(this._maxBufferedFrames / 2);
      return raw > this._maxBufferedFrames ? this._maxBufferedFrames : raw;
    }

    get trimCrossfadeFrames() {
      return this._trimCrossfadeFrames;
    }

    get statsIntervalQuanta() {
      return this._statsIntervalQuanta;
    }

    setMaxBufferedFrames(frames) {
      if (frames === false || frames === null || frames === 0) {
        this._maxBufferedFrames = 0;
      } else if (frames === true) {
        this._maxBufferedFrames = DEFAULT_HIGH_WATERMARK_FRAMES;
      } else {
        this._maxBufferedFrames = sanitizeNonNegInt(frames, this._maxBufferedFrames);
      }
      return this._maxBufferedFrames;
    }

    setTargetWatermarkFrames(frames) {
      if (frames === null || frames === undefined) {
        this._targetWatermarkFramesOverride = -1;
      } else {
        this._targetWatermarkFramesOverride = sanitizeNonNegInt(
          frames,
          this._targetWatermarkFramesOverride
        );
      }
      return this.targetWatermarkFrames;
    }

    setTrimCrossfadeFrames(frames) {
      this._trimCrossfadeFrames = sanitizeTrimCrossfadeFrames(frames, this._trimCrossfadeFrames);
      return this._trimCrossfadeFrames;
    }

    setStatsIntervalQuanta(quanta) {
      if (quanta === false || quanta === null || quanta === 0) {
        this._statsIntervalQuanta = 0;
        this._statsQuantaCounter = 0;
      } else {
        const next = sanitizeNonNegInt(quanta, this._statsIntervalQuanta);
        if (next !== this._statsIntervalQuanta) {
          this._statsIntervalQuanta = next;
          this._statsQuantaCounter = 0;
        }
      }
      return this._statsIntervalQuanta;
    }

    resetStats() {
      this._processedQuanta = 0;
      this._processedFrames = 0;
      this._sanitizedSamples = 0;
      this._clampedSamples = 0;
      this._softClippedSamples = 0;
      this._underruns = 0;
      this._underrunQuanta = 0;
      this._underrunFrames = 0;
      this._declickFadeOuts = 0;
      this._fadeInCount = 0;
      this._overrunTrimmedFrames = 0;
      this._overrunTrimCount = 0;
      this._lastFillRatio = 0.0;
      this._peakFillRatio = 0.0;
      this._statsQuantaCounter = 0;
      if (this._ctrl) {
        Atomics.store(this._ctrl, HDR_UNDERRUNS, 0);
        Atomics.store(this._ctrl, HDR_UNDERRUN_FRAMES, 0);
      }
      return this.getStats();
    }

    setFadeCurve(curve) {
      this._fadeCurve = sanitizeFadeCurve(curve, this._fadeCurve);
      return this._fadeCurve;
    }

    setSoftLimit(enabledOrKnee, optionalKnee) {
      if (typeof enabledOrKnee === 'number') {
        this._softLimitKnee = sanitizeSoftLimitKnee(enabledOrKnee, this._softLimitKnee);
        this._softLimitEnabled = true;
      } else if (enabledOrKnee !== undefined) {
        this._softLimitEnabled = Boolean(enabledOrKnee);
      }
      if (optionalKnee !== undefined && optionalKnee !== null) {
        this._softLimitKnee = sanitizeSoftLimitKnee(optionalKnee, this._softLimitKnee);
      }
      return this._softLimitEnabled;
    }

    setDeclickFrames(frames) {
      this._declickFrames = sanitizeDeclickFrames(frames, this._declickFrames);
      if (this._declickFrames === 0) {
        this._declickPos = 0;
        this._declickTotal = 0;
      }
      if (this.fadeInFrames === 0) {
        this._fadeInPos = 0;
        this._fadeInTotal = 0;
        this._fadeInHasOffset = false;
      }
      return this._declickFrames;
    }

    setFadeInFrames(frames) {
      if (frames === null || frames === undefined) {
        this._fadeInFramesOverride = -1;
      } else {
        this._fadeInFramesOverride = sanitizeDeclickFrames(frames, this._fadeInFramesOverride);
      }
      if (this.fadeInFrames === 0) {
        this._fadeInPos = 0;
        this._fadeInTotal = 0;
        this._fadeInHasOffset = false;
      }
      return this.fadeInFrames;
    }

    setClampSamples(enabledOrLimit) {
      this._clampLimit = sanitizeClampLimit(enabledOrLimit, this._clampLimit);
      return this._clampLimit;
    }

    _resetDeclickState() {
      this._lastSamples.fill(0);
      this._fadeInFrom.fill(0);
      this._hasActiveAudio = false;
      this._inUnderrun = false;
      this._declickPos = 0;
      this._declickTotal = 0;
      this._fadeInPos = 0;
      this._fadeInTotal = 0;
      this._fadeInHasOffset = false;
    }

    _attachSab(sab) {
      this._resetDeclickState();

      if (sab === null) {
        this._sab = null;
        this._ctrl = null;
        this._samples = null;
        this._capacityFrames = 0;
        this._channels = DEFAULT_CHANNELS;
        this._modulus = 0;
        return;
      }

      if (typeof SharedArrayBuffer === 'undefined' || !(sab instanceof SharedArrayBuffer)) {
        throw new TypeError('PlayGta6AudioWorkletProcessor requires a SharedArrayBuffer');
      }
      if (sab.byteLength < RING_HEADER_BYTES) {
        throw new RangeError(
          `SharedArrayBuffer too small (${sab.byteLength} bytes; need > ${RING_HEADER_BYTES})`
        );
      }

      const ctrl = new Int32Array(sab, 0, RING_HEADER_INT32_COUNT);
      const capacityFrames = Atomics.load(ctrl, HDR_CAPACITY_FRAMES);
      const channels = Atomics.load(ctrl, HDR_CHANNELS);

      if (!Number.isInteger(capacityFrames) || capacityFrames <= 0) {
        throw new RangeError(
          `Invalid AudioRingBuffer header capacityFrames (${capacityFrames})`
        );
      }
      if (!Number.isInteger(channels) || channels <= 0) {
        throw new RangeError(`Invalid AudioRingBuffer header channels (${channels})`);
      }

      const requiredBytes =
        RING_HEADER_BYTES + capacityFrames * channels * Float32Array.BYTES_PER_ELEMENT;
      if (sab.byteLength < requiredBytes) {
        throw new RangeError(
          `SharedArrayBuffer (${sab.byteLength} bytes) is smaller than header requires (${requiredBytes} bytes)`
        );
      }

      this._sab = sab;
      this._ctrl = ctrl;
      this._capacityFrames = capacityFrames;
      this._channels = channels;
      this._modulus = capacityFrames * 2;
      this._samples = new Float32Array(sab, RING_HEADER_BYTES, capacityFrames * channels);
    }

    _updateTargetGain(immediate = false, rampFramesOverride = undefined) {
      const nextTarget = this._muted ? 0.0 : this._gain;
      this._targetGain = nextTarget;

      const rampFrames =
        rampFramesOverride !== undefined && rampFramesOverride !== null
          ? sanitizeSmoothingFrames(rampFramesOverride)
          : this._smoothingFrames;

      if (immediate || rampFrames <= 0 || this._currentGain === nextTarget) {
        this._currentGain = nextTarget;
        this._rampStep = 0.0;
        this._rampRemaining = 0;
      } else {
        this._rampRemaining = rampFrames;
        this._rampStep = (nextTarget - this._currentGain) / rampFrames;
      }
    }

    _computeBufferedFrames() {
      if (!this._ctrl || this._capacityFrames <= 0) {
        return 0;
      }
      const modulus = this._modulus;
      let write = Atomics.load(this._ctrl, HDR_WRITE_CURSOR);
      let read = Atomics.load(this._ctrl, HDR_READ_CURSOR);
      if (write < 0 || write >= modulus) {
        write = ((write % modulus) + modulus) % modulus;
      }
      if (read < 0 || read >= modulus) {
        read = ((read % modulus) + modulus) % modulus;
      }
      let used = write - read;
      if (used < 0) {
        used += modulus;
      }
      return used > this._capacityFrames ? this._capacityFrames : used;
    }

    getStats() {
      const ctrl = this._ctrl;
      const overflows = ctrl ? Atomics.load(ctrl, HDR_OVERFLOWS) : 0;
      const bufferedFrames = this._computeBufferedFrames();
      const fillRatio =
        ctrl && this._capacityFrames > 0
          ? clampFillRatio(bufferedFrames / this._capacityFrames)
          : 0.0;
      return {
        attached: ctrl !== null,
        stopped: this._stopped,
        gain: this._gain,
        muted: this._muted,
        currentGain: this._currentGain,
        targetGain: this._targetGain,
        declickFrames: this._declickFrames,
        fadeCurve: this._fadeCurve,
        fadeInFrames: this.fadeInFrames,
        clampSamples: this._clampLimit > 0.0,
        clampLimit: this._clampLimit,
        softLimit: this._softLimitEnabled,
        softLimitKnee: this._softLimitKnee,
        capacityFrames: this._capacityFrames,
        channels: this._channels,
        bufferedFrames,
        fillRatio,
        peakFillRatio: this._peakFillRatio,
        maxBufferedFrames: this._maxBufferedFrames,
        targetWatermarkFrames: this.targetWatermarkFrames,
        trimCrossfadeFrames: this._trimCrossfadeFrames,
        overrunTrimmedFrames: this._overrunTrimmedFrames,
        overrunTrimCount: this._overrunTrimCount,
        statsIntervalQuanta: this._statsIntervalQuanta,
        underruns: ctrl ? Atomics.load(ctrl, HDR_UNDERRUNS) : this._underruns,
        underrunQuanta: this._underrunQuanta,
        underrunFrames: ctrl ? Atomics.load(ctrl, HDR_UNDERRUN_FRAMES) : this._underrunFrames,
        inUnderrun: this._inUnderrun,
        fadeOutCount: this._declickFadeOuts,
        underrunFadeOuts: this._declickFadeOuts,
        fadeInCount: this._fadeInCount,
        recoveryFadeIns: this._fadeInCount,
        overflows,
        overruns: overflows,
        droppedFrames: ctrl ? Atomics.load(ctrl, HDR_DROPPED_FRAMES) : 0,
        quantaProcessed: this._processedQuanta,
        processedQuanta: this._processedQuanta,
        processedFrames: this._processedFrames,
        framesProcessed: this._processedFrames,
        sanitizedSamples: this._sanitizedSamples,
        nonFiniteSanitized: this._sanitizedSamples,
        clampedSamples: this._clampedSamples,
        softClippedSamples: this._softClippedSamples,
      };
    }

    _emitPeriodicStats(bufferedFrames, fillRatio) {
      const ctrl = this._ctrl;
      const p = this._periodicStatsPayload;
      const overflows = ctrl ? Atomics.load(ctrl, HDR_OVERFLOWS) : 0;
      p.attached = ctrl !== null;
      p.capacityFrames = this._capacityFrames;
      p.channels = this._channels;
      p.bufferedFrames = bufferedFrames;
      p.fillRatio = fillRatio;
      p.peakFillRatio = this._peakFillRatio;
      p.maxBufferedFrames = this._maxBufferedFrames;
      p.targetWatermarkFrames = this.targetWatermarkFrames;
      p.trimCrossfadeFrames = this._trimCrossfadeFrames;
      p.overrunTrimmedFrames = this._overrunTrimmedFrames;
      p.overrunTrimCount = this._overrunTrimCount;
      p.underruns = ctrl ? Atomics.load(ctrl, HDR_UNDERRUNS) : this._underruns;
      p.underrunQuanta = this._underrunQuanta;
      p.underrunFrames = ctrl ? Atomics.load(ctrl, HDR_UNDERRUN_FRAMES) : this._underrunFrames;
      p.inUnderrun = this._inUnderrun;
      p.fadeOutCount = this._declickFadeOuts;
      p.underrunFadeOuts = this._declickFadeOuts;
      p.fadeInCount = this._fadeInCount;
      p.recoveryFadeIns = this._fadeInCount;
      p.overflows = overflows;
      p.overruns = overflows;
      p.droppedFrames = ctrl ? Atomics.load(ctrl, HDR_DROPPED_FRAMES) : 0;
      p.quantaProcessed = this._processedQuanta;
      p.processedQuanta = this._processedQuanta;
      p.processedFrames = this._processedFrames;
      p.framesProcessed = this._processedFrames;
      p.sanitizedSamples = this._sanitizedSamples;
      p.nonFiniteSanitized = this._sanitizedSamples;
      p.clampedSamples = this._clampedSamples;
      p.softClippedSamples = this._softClippedSamples;
      this._post(p);
    }

    _post(payload) {
      if (this.port && typeof this.port.postMessage === 'function') {
        this.port.postMessage(payload);
      }
    }

    _handleMessage(msg) {
      if (!msg || typeof msg !== 'object') {
        return;
      }

      switch (msg.type) {
        case 'attach': {
          try {
            this._attachSab(msg.sab !== undefined ? msg.sab : null);
            const reply = {
              type: 'attached',
              attached: this._ctrl !== null,
              capacityFrames: this._capacityFrames,
              channels: this._channels,
            };
            if (msg.id !== undefined) reply.id = msg.id;
            this._post(reply);
          } catch (err) {
            const reply = {
              type: 'error',
              error: err && err.message ? err.message : String(err),
            };
            if (msg.id !== undefined) reply.id = msg.id;
            this._post(reply);
          }
          break;
        }

        case 'setGain': {
          this._gain = clampGain(msg.gain);
          this._updateTargetGain(Boolean(msg.immediate), msg.rampFrames);
          break;
        }

        case 'setMuted': {
          this._muted = Boolean(msg.muted);
          this._updateTargetGain(Boolean(msg.immediate), msg.rampFrames);
          break;
        }

        case 'setDeclickFrames':
        case 'setDeclick': {
          const raw =
            msg.declickFrames !== undefined
              ? msg.declickFrames
              : msg.frames !== undefined
                ? msg.frames
                : msg.underrunFadeFrames;
          this.setDeclickFrames(raw);
          if (msg.id !== undefined) {
            this._post({
              type: 'declickFrames',
              declickFrames: this._declickFrames,
              id: msg.id,
            });
          }
          break;
        }

        case 'setFadeInFrames':
        case 'setRecoveryFadeFrames': {
          const raw =
            msg.fadeInFrames !== undefined
              ? msg.fadeInFrames
              : msg.recoveryFadeFrames !== undefined
                ? msg.recoveryFadeFrames
                : msg.frames;
          this.setFadeInFrames(raw);
          if (msg.id !== undefined) {
            this._post({
              type: 'fadeInFrames',
              fadeInFrames: this.fadeInFrames,
              id: msg.id,
            });
          }
          break;
        }

        case 'setClampSamples':
        case 'setSampleClamp': {
          const raw =
            msg.sampleClampLimit !== undefined
              ? msg.sampleClampLimit
              : msg.clampLimit !== undefined
                ? msg.clampLimit
                : msg.limit !== undefined
                  ? msg.limit
                  : msg.clampSamples !== undefined
                    ? msg.clampSamples
                    : msg.enabled;
          this.setClampSamples(raw);
          if (msg.id !== undefined) {
            this._post({
              type: 'clampSamples',
              clampSamples: this._clampLimit > 0.0,
              clampLimit: this._clampLimit,
              id: msg.id,
            });
          }
          break;
        }

        case 'setFadeCurve': {
          const raw = msg.fadeCurve !== undefined ? msg.fadeCurve : msg.curve;
          this.setFadeCurve(raw);
          if (msg.id !== undefined) {
            this._post({
              type: 'fadeCurve',
              fadeCurve: this._fadeCurve,
              id: msg.id,
            });
          }
          break;
        }

        case 'setSoftLimit':
        case 'setSoftClip': {
          const enabledOrKnee =
            msg.softLimit !== undefined
              ? msg.softLimit
              : msg.softClip !== undefined
                ? msg.softClip
                : msg.enabled;
          const knee =
            msg.softLimitKnee !== undefined
              ? msg.softLimitKnee
              : msg.knee !== undefined
                ? msg.knee
                : msg.limiterKnee;
          this.setSoftLimit(enabledOrKnee, knee);
          if (msg.id !== undefined) {
            this._post({
              type: 'softLimit',
              softLimit: this._softLimitEnabled,
              softLimitKnee: this._softLimitKnee,
              id: msg.id,
            });
          }
          break;
        }

        case 'setMaxBufferedFrames':
        case 'setWatermarkConfig':
        case 'setOverrunGuard': {
          const rawMax =
            msg.maxBufferedFrames !== undefined
              ? msg.maxBufferedFrames
              : msg.highWatermarkFrames !== undefined
                ? msg.highWatermarkFrames
                : msg.frames;
          if (rawMax !== undefined) {
            this.setMaxBufferedFrames(rawMax);
          }
          const rawTarget =
            msg.targetWatermarkFrames !== undefined
              ? msg.targetWatermarkFrames
              : msg.targetBufferedFrames !== undefined
                ? msg.targetBufferedFrames
                : msg.lowWatermarkFrames;
          if (rawTarget !== undefined) {
            this.setTargetWatermarkFrames(rawTarget);
          }
          const rawXfade =
            msg.trimCrossfadeFrames !== undefined
              ? msg.trimCrossfadeFrames
              : msg.overrunCrossfadeFrames;
          if (rawXfade !== undefined) {
            this.setTrimCrossfadeFrames(rawXfade);
          }
          if (msg.id !== undefined) {
            this._post({
              type: 'watermarkConfig',
              maxBufferedFrames: this._maxBufferedFrames,
              targetWatermarkFrames: this.targetWatermarkFrames,
              trimCrossfadeFrames: this._trimCrossfadeFrames,
              id: msg.id,
            });
          }
          break;
        }

        case 'setStatsInterval':
        case 'setStatsIntervalQuanta': {
          const raw =
            msg.statsIntervalQuanta !== undefined
              ? msg.statsIntervalQuanta
              : msg.intervalQuanta !== undefined
                ? msg.intervalQuanta
                : msg.quanta;
          this.setStatsIntervalQuanta(raw);
          if (msg.id !== undefined) {
            this._post({
              type: 'statsInterval',
              statsIntervalQuanta: this._statsIntervalQuanta,
              id: msg.id,
            });
          }
          break;
        }

        case 'resetStats': {
          this.resetStats();
          if (msg.id !== undefined) {
            this._post({
              type: 'statsReset',
              id: msg.id,
            });
          }
          break;
        }

        case 'stats': {
          const reply = Object.assign({ type: 'stats' }, this.getStats());
          if (msg.id !== undefined) reply.id = msg.id;
          this._post(reply);
          break;
        }

        case 'stop': {
          this._stopped = true;
          const reply = { type: 'stopped' };
          if (msg.id !== undefined) reply.id = msg.id;
          this._post(reply);
          break;
        }

        default:
          break;
      }
    }

    _applyTrimCrossfade(
      output,
      outChannels,
      samples,
      cap,
      inChannels,
      oldRead,
      framesToRead,
      preLoopGain,
      preLoopRampRemaining
    ) {
      const xfadeCount =
        this._trimCrossfadeFrames < framesToRead ? this._trimCrossfadeFrames : framesToRead;
      if (xfadeCount <= 0) {
        return;
      }
      if (preLoopRampRemaining === 0 && preLoopGain === 0.0) {
        return;
      }

      const clampLimit = this._clampLimit;
      const negClampLimit = -clampLimit;
      const softLimit = this._softLimitEnabled;
      const softKnee = this._softLimitKnee;
      const negSoftKnee = -softKnee;
      const softHeadroom = 1.0 - softKnee;
      const copyChannels = inChannels < outChannels ? inChannels : outChannels;

      let oldSlot = oldRead >= cap ? oldRead - cap : oldRead;
      let gain = preLoopGain;
      let rem = preLoopRampRemaining;
      const step = this._rampStep;
      const target = this._targetGain;
      const invXfade = xfadeCount > 1 ? Math.PI / xfadeCount : 0.0;

      if (inChannels === 2 && outChannels === 2) {
        const outL = output[0];
        const outR = output[1];
        for (let f = 0; f < xfadeCount; f++) {
          if (rem > 0) {
            gain += step;
            rem--;
            if (rem === 0) gain = target;
          }
          const alpha =
            xfadeCount === 1 ? 0.5 : f === 0 ? 0.0 : 0.5 * (1.0 - Math.cos(f * invXfade));
          const invAlpha = 1.0 - alpha;

          const src = oldSlot * 2;
          let sL = samples[src];
          let sR = samples[src + 1];
          if (sL - sL !== 0) {
            sL = 0.0;
          } else {
            if (softLimit) {
              if (sL > softKnee) sL = softLimitPositive(sL, softKnee, softHeadroom);
              else if (sL < negSoftKnee) sL = -softLimitPositive(-sL, softKnee, softHeadroom);
            }
            if (clampLimit > 0.0) {
              if (sL > clampLimit) sL = clampLimit;
              else if (sL < negClampLimit) sL = negClampLimit;
            }
          }
          if (sR - sR !== 0) {
            sR = 0.0;
          } else {
            if (softLimit) {
              if (sR > softKnee) sR = softLimitPositive(sR, softKnee, softHeadroom);
              else if (sR < negSoftKnee) sR = -softLimitPositive(-sR, softKnee, softHeadroom);
            }
            if (clampLimit > 0.0) {
              if (sR > clampLimit) sR = clampLimit;
              else if (sR < negClampLimit) sR = negClampLimit;
            }
          }
          if (gain === 0.0) {
            sL = 0.0;
            sR = 0.0;
          } else if (gain !== 1.0) {
            sL *= gain;
            sR *= gain;
          }

          outL[f] = sL * invAlpha + outL[f] * alpha;
          outR[f] = sR * invAlpha + outR[f] * alpha;

          oldSlot++;
          if (oldSlot >= cap) oldSlot = 0;
        }
      } else if (inChannels === 1) {
        for (let f = 0; f < xfadeCount; f++) {
          if (rem > 0) {
            gain += step;
            rem--;
            if (rem === 0) gain = target;
          }
          const alpha =
            xfadeCount === 1 ? 0.5 : f === 0 ? 0.0 : 0.5 * (1.0 - Math.cos(f * invXfade));
          const invAlpha = 1.0 - alpha;

          let s = samples[oldSlot];
          if (s - s !== 0) {
            s = 0.0;
          } else {
            if (softLimit) {
              if (s > softKnee) s = softLimitPositive(s, softKnee, softHeadroom);
              else if (s < negSoftKnee) s = -softLimitPositive(-s, softKnee, softHeadroom);
            }
            if (clampLimit > 0.0) {
              if (s > clampLimit) s = clampLimit;
              else if (s < negClampLimit) s = negClampLimit;
            }
          }
          const v = gain === 0.0 ? 0.0 : gain === 1.0 ? s : s * gain;
          for (let c = 0; c < outChannels; c++) {
            output[c][f] = v * invAlpha + output[c][f] * alpha;
          }

          oldSlot++;
          if (oldSlot >= cap) oldSlot = 0;
        }
      } else {
        for (let f = 0; f < xfadeCount; f++) {
          if (rem > 0) {
            gain += step;
            rem--;
            if (rem === 0) gain = target;
          }
          const alpha =
            xfadeCount === 1 ? 0.5 : f === 0 ? 0.0 : 0.5 * (1.0 - Math.cos(f * invXfade));
          const invAlpha = 1.0 - alpha;

          const srcBase = oldSlot * inChannels;
          for (let c = 0; c < copyChannels; c++) {
            let s = samples[srcBase + c];
            if (s - s !== 0) {
              s = 0.0;
            } else {
              if (softLimit) {
                if (s > softKnee) s = softLimitPositive(s, softKnee, softHeadroom);
                else if (s < negSoftKnee) s = -softLimitPositive(-s, softKnee, softHeadroom);
              }
              if (clampLimit > 0.0) {
                if (s > clampLimit) s = clampLimit;
                else if (s < negClampLimit) s = negClampLimit;
              }
            }
            const v = gain === 0.0 ? 0.0 : gain === 1.0 ? s : s * gain;
            output[c][f] = v * invAlpha + output[c][f] * alpha;
          }
          for (let c = copyChannels; c < outChannels; c++) {
            output[c][f] = 0.0;
          }

          oldSlot++;
          if (oldSlot >= cap) oldSlot = 0;
        }
      }
    }

    /**
     * WebAudio AudioWorklet render callback.
     * Strictly zero allocations and lock-free on the hot audio thread.
     * Sanitizes any non-finite (`NaN`/`±Infinity`) PCM sample to `0.0`,
     * optionally clamps out-of-range finite samples to `[-clampLimit, +clampLimit]`,
     * and applies short linear de-click fade-out on underrun and fade-in on recovery.
     *
     * @param {Float32Array[][]} _inputs
     * @param {Float32Array[][]} outputs
     * @returns {boolean} `true` to keep processor alive, `false` once stopped.
     */
    process(_inputs, outputs) {
      const hasOutput = outputs && outputs.length > 0 && outputs[0] && outputs[0].length > 0;

      if (this._stopped) {
        if (hasOutput) {
          const out = outputs[0];
          for (let c = 0; c < out.length; c++) {
            if (out[c]) out[c].fill(0);
          }
        }
        return false;
      }

      if (!hasOutput) {
        return true;
      }

      const output = outputs[0];
      const outChannels = output.length;
      const firstChannel = output[0];
      if (!firstChannel) {
        return true;
      }
      const quantumFrames = firstChannel.length;
      if (quantumFrames <= 0) {
        return true;
      }

      const ctrl = this._ctrl;
      const samples = this._samples;
      if (!ctrl || !samples) {
        for (let c = 0; c < outChannels; c++) {
          output[c].fill(0);
        }
        return true;
      }

      const cap = this._capacityFrames;
      const inChannels = this._channels;
      const modulus = this._modulus;

      let read = Atomics.load(ctrl, HDR_READ_CURSOR);
      let write = Atomics.load(ctrl, HDR_WRITE_CURSOR);
      if (read < 0 || read >= modulus) {
        read = ((read % modulus) + modulus) % modulus;
      }
      if (write < 0 || write >= modulus) {
        write = ((write % modulus) + modulus) % modulus;
      }
      let used = write - read;
      if (used < 0) {
        used += modulus;
      }
      if (used > cap) {
        used = cap;
      }

      // W327: Lock-free fill-ratio peak watermark tracking.
      const preFillRatio = cap > 0 ? clampFillRatio(used / cap) : 0.0;
      if (preFillRatio > this._peakFillRatio) {
        this._peakFillRatio = preFillRatio;
      }

      // W327: Bounded overrun drift guard (high-watermark trim).
      // Advances the read cursor down to target watermark without inducing an
      // artificial underrun, then crossfades from the discarded head to prevent pops.
      let trimFrames = 0;
      let trimOldRead = read;
      const maxBuffered = this._maxBufferedFrames;
      if (maxBuffered > 0 && used > maxBuffered && used > quantumFrames) {
        let target =
          this._targetWatermarkFramesOverride >= 0
            ? this._targetWatermarkFramesOverride
            : Math.floor(maxBuffered / 2);
        if (target > maxBuffered) {
          target = maxBuffered;
        }
        const keepFrames = target > quantumFrames ? target : quantumFrames;
        if (used > keepFrames) {
          trimFrames = used - keepFrames;
          trimOldRead = read;
          read += trimFrames;
          if (read >= modulus) {
            read -= modulus;
          }
          used = keepFrames;
          this._overrunTrimmedFrames += trimFrames;
          this._overrunTrimCount++;
        }
      }

      const framesToRead = quantumFrames <= used ? quantumFrames : used;
      const lastSamples = this._lastSamples;
      const trackedChannels =
        outChannels < lastSamples.length ? outChannels : lastSamples.length;
      const clampLimit = this._clampLimit;
      const negClampLimit = -clampLimit;
      const softLimit = this._softLimitEnabled;
      const softKnee = this._softLimitKnee;
      const negSoftKnee = -softKnee;
      const softHeadroom = 1.0 - softKnee;
      const useCosine = this._fadeCurve === 'cosine';

      if (framesToRead > 0) {
        const wasInUnderrun = this._inUnderrun;
        const wasMidFadeOut = this._declickTotal > 0;
        const fadeInFrames =
          this._fadeInFramesOverride >= 0 ? this._fadeInFramesOverride : this._declickFrames;

        if (wasInUnderrun) {
          this._inUnderrun = false;
          if (fadeInFrames > 0) {
            if (wasMidFadeOut) {
              const prevScale = useCosine
                ? 0.5 * (1.0 + Math.cos(Math.PI * (this._declickPos / this._declickTotal)))
                : (this._declickTotal - this._declickPos) / this._declickTotal;
              const fadeInFrom = this._fadeInFrom;
              let hasOffset = false;
              for (let c = 0; c < trackedChannels; c++) {
                const prevVal = lastSamples[c] * prevScale;
                fadeInFrom[c] = prevVal;
                if (prevVal !== 0.0) {
                  hasOffset = true;
                }
              }
              for (let c = trackedChannels; c < fadeInFrom.length; c++) {
                fadeInFrom[c] = 0.0;
              }
              this._fadeInHasOffset = hasOffset;
            } else {
              this._fadeInHasOffset = false;
            }
            this._fadeInPos = 0;
            this._fadeInTotal = fadeInFrames;
            this._fadeInCount++;
          } else {
            this._fadeInPos = 0;
            this._fadeInTotal = 0;
            this._fadeInHasOffset = false;
          }
        }

        // Fresh audio arrived: cancel any in-flight cross-quantum underrun fade-out.
        this._declickPos = 0;
        this._declickTotal = 0;

        const readSlot = read >= cap ? read - cap : read;
        const firstFrames = framesToRead <= cap - readSlot ? framesToRead : cap - readSlot;
        const secondFrames = framesToRead - firstFrames;
        const copyChannels = inChannels < outChannels ? inChannels : outChannels;
        const preLoopGain = this._currentGain;
        const preLoopRampRemaining = this._rampRemaining;
        let sanitized = 0;
        let clamped = 0;
        let softClipped = 0;

        if (this._rampRemaining > 0) {
          // Smooth linear gain ramp (click-free volume/mute transition).
          let gain = this._currentGain;
          let rem = this._rampRemaining;
          const step = this._rampStep;
          const target = this._targetGain;

          if (inChannels === 2 && outChannels === 2) {
            const outL = output[0];
            const outR = output[1];
            let src = readSlot * 2;
            for (let f = 0; f < firstFrames; f++) {
              if (rem > 0) {
                gain += step;
                rem--;
                if (rem === 0) gain = target;
              }
              let sL = samples[src];
              let sR = samples[src + 1];
              if (sL - sL !== 0) {
                sL = 0.0;
                sanitized++;
              } else {
                if (softLimit) {
                  if (sL > softKnee) {
                    sL = softLimitPositive(sL, softKnee, softHeadroom);
                    softClipped++;
                  } else if (sL < negSoftKnee) {
                    sL = -softLimitPositive(-sL, softKnee, softHeadroom);
                    softClipped++;
                  }
                }
                if (clampLimit > 0.0) {
                  if (sL > clampLimit) {
                    sL = clampLimit;
                    clamped++;
                  } else if (sL < negClampLimit) {
                    sL = negClampLimit;
                    clamped++;
                  }
                }
              }
              if (sR - sR !== 0) {
                sR = 0.0;
                sanitized++;
              } else {
                if (softLimit) {
                  if (sR > softKnee) {
                    sR = softLimitPositive(sR, softKnee, softHeadroom);
                    softClipped++;
                  } else if (sR < negSoftKnee) {
                    sR = -softLimitPositive(-sR, softKnee, softHeadroom);
                    softClipped++;
                  }
                }
                if (clampLimit > 0.0) {
                  if (sR > clampLimit) {
                    sR = clampLimit;
                    clamped++;
                  } else if (sR < negClampLimit) {
                    sR = negClampLimit;
                    clamped++;
                  }
                }
              }
              if (gain === 0.0) {
                outL[f] = 0.0;
                outR[f] = 0.0;
              } else {
                outL[f] = sL * gain;
                outR[f] = sR * gain;
              }
              src += 2;
            }
            if (secondFrames > 0) {
              src = 0;
              for (let f = 0; f < secondFrames; f++) {
                if (rem > 0) {
                  gain += step;
                  rem--;
                  if (rem === 0) gain = target;
                }
                const dst = firstFrames + f;
                let sL = samples[src];
                let sR = samples[src + 1];
                if (sL - sL !== 0) {
                  sL = 0.0;
                  sanitized++;
                } else {
                  if (softLimit) {
                    if (sL > softKnee) {
                      sL = softLimitPositive(sL, softKnee, softHeadroom);
                      softClipped++;
                    } else if (sL < negSoftKnee) {
                      sL = -softLimitPositive(-sL, softKnee, softHeadroom);
                      softClipped++;
                    }
                  }
                  if (clampLimit > 0.0) {
                    if (sL > clampLimit) {
                      sL = clampLimit;
                      clamped++;
                    } else if (sL < negClampLimit) {
                      sL = negClampLimit;
                      clamped++;
                    }
                  }
                }
                if (sR - sR !== 0) {
                  sR = 0.0;
                  sanitized++;
                } else {
                  if (softLimit) {
                    if (sR > softKnee) {
                      sR = softLimitPositive(sR, softKnee, softHeadroom);
                      softClipped++;
                    } else if (sR < negSoftKnee) {
                      sR = -softLimitPositive(-sR, softKnee, softHeadroom);
                      softClipped++;
                    }
                  }
                  if (clampLimit > 0.0) {
                    if (sR > clampLimit) {
                      sR = clampLimit;
                      clamped++;
                    } else if (sR < negClampLimit) {
                      sR = negClampLimit;
                      clamped++;
                    }
                  }
                }
                if (gain === 0.0) {
                  outL[dst] = 0.0;
                  outR[dst] = 0.0;
                } else {
                  outL[dst] = sL * gain;
                  outR[dst] = sR * gain;
                }
                src += 2;
              }
            }
          } else if (inChannels === 1) {
            let src = readSlot;
            for (let f = 0; f < firstFrames; f++) {
              if (rem > 0) {
                gain += step;
                rem--;
                if (rem === 0) gain = target;
              }
              let s = samples[src];
              if (s - s !== 0) {
                s = 0.0;
                sanitized++;
              } else {
                if (softLimit) {
                  if (s > softKnee) {
                    s = softLimitPositive(s, softKnee, softHeadroom);
                    softClipped++;
                  } else if (s < negSoftKnee) {
                    s = -softLimitPositive(-s, softKnee, softHeadroom);
                    softClipped++;
                  }
                }
                if (clampLimit > 0.0) {
                  if (s > clampLimit) {
                    s = clampLimit;
                    clamped++;
                  } else if (s < negClampLimit) {
                    s = negClampLimit;
                    clamped++;
                  }
                }
              }
              const v = gain === 0.0 ? 0.0 : s * gain;
              for (let c = 0; c < outChannels; c++) {
                output[c][f] = v;
              }
              src++;
            }
            if (secondFrames > 0) {
              src = 0;
              for (let f = 0; f < secondFrames; f++) {
                if (rem > 0) {
                  gain += step;
                  rem--;
                  if (rem === 0) gain = target;
                }
                const dst = firstFrames + f;
                let s = samples[src];
                if (s - s !== 0) {
                  s = 0.0;
                  sanitized++;
                } else {
                  if (softLimit) {
                    if (s > softKnee) {
                      s = softLimitPositive(s, softKnee, softHeadroom);
                      softClipped++;
                    } else if (s < negSoftKnee) {
                      s = -softLimitPositive(-s, softKnee, softHeadroom);
                      softClipped++;
                    }
                  }
                  if (clampLimit > 0.0) {
                    if (s > clampLimit) {
                      s = clampLimit;
                      clamped++;
                    } else if (s < negClampLimit) {
                      s = negClampLimit;
                      clamped++;
                    }
                  }
                }
                const v = gain === 0.0 ? 0.0 : s * gain;
                for (let c = 0; c < outChannels; c++) {
                  output[c][dst] = v;
                }
                src++;
              }
            }
          } else {
            let srcBase = readSlot * inChannels;
            for (let f = 0; f < firstFrames; f++) {
              if (rem > 0) {
                gain += step;
                rem--;
                if (rem === 0) gain = target;
              }
              for (let c = 0; c < copyChannels; c++) {
                let s = samples[srcBase + c];
                if (s - s !== 0) {
                  s = 0.0;
                  sanitized++;
                } else {
                  if (softLimit) {
                    if (s > softKnee) {
                      s = softLimitPositive(s, softKnee, softHeadroom);
                      softClipped++;
                    } else if (s < negSoftKnee) {
                      s = -softLimitPositive(-s, softKnee, softHeadroom);
                      softClipped++;
                    }
                  }
                  if (clampLimit > 0.0) {
                    if (s > clampLimit) {
                      s = clampLimit;
                      clamped++;
                    } else if (s < negClampLimit) {
                      s = negClampLimit;
                      clamped++;
                    }
                  }
                }
                output[c][f] = gain === 0.0 ? 0.0 : s * gain;
              }
              for (let c = copyChannels; c < outChannels; c++) {
                output[c][f] = 0.0;
              }
              srcBase += inChannels;
            }
            if (secondFrames > 0) {
              srcBase = 0;
              for (let f = 0; f < secondFrames; f++) {
                if (rem > 0) {
                  gain += step;
                  rem--;
                  if (rem === 0) gain = target;
                }
                const dst = firstFrames + f;
                for (let c = 0; c < copyChannels; c++) {
                  let s = samples[srcBase + c];
                  if (s - s !== 0) {
                    s = 0.0;
                    sanitized++;
                  } else {
                    if (softLimit) {
                      if (s > softKnee) {
                        s = softLimitPositive(s, softKnee, softHeadroom);
                        softClipped++;
                      } else if (s < negSoftKnee) {
                        s = -softLimitPositive(-s, softKnee, softHeadroom);
                        softClipped++;
                      }
                    }
                    if (clampLimit > 0.0) {
                      if (s > clampLimit) {
                        s = clampLimit;
                        clamped++;
                      } else if (s < negClampLimit) {
                        s = negClampLimit;
                        clamped++;
                      }
                    }
                  }
                  output[c][dst] = gain === 0.0 ? 0.0 : s * gain;
                }
                for (let c = copyChannels; c < outChannels; c++) {
                  output[c][dst] = 0.0;
                }
                srcBase += inChannels;
              }
            }
          }

          this._currentGain = gain;
          this._rampRemaining = rem;
        } else {
          // Steady-state gain fast paths (0, 1, or constant scalar).
          const gain = this._currentGain;

          if (gain === 0.0) {
            for (let c = 0; c < outChannels; c++) {
              output[c].fill(0, 0, framesToRead);
            }
            // Scan consumed ring samples so sanitized/clamped/softClipped telemetry stays exact while muted.
            const scanChannels = inChannels === 1 ? 1 : copyChannels;
            if (scanChannels === inChannels) {
              const firstCount = firstFrames * inChannels;
              let src = readSlot * inChannels;
              for (let i = 0; i < firstCount; i++) {
                const s = samples[src + i];
                if (s - s !== 0) {
                  sanitized++;
                } else {
                  if (softLimit && (s > softKnee || s < negSoftKnee)) {
                    softClipped++;
                  }
                  if (clampLimit > 0.0 && (s > clampLimit || s < negClampLimit)) {
                    clamped++;
                  }
                }
              }
              if (secondFrames > 0) {
                const secondCount = secondFrames * inChannels;
                for (let i = 0; i < secondCount; i++) {
                  const s = samples[i];
                  if (s - s !== 0) {
                    sanitized++;
                  } else {
                    if (softLimit && (s > softKnee || s < negSoftKnee)) {
                      softClipped++;
                    }
                    if (clampLimit > 0.0 && (s > clampLimit || s < negClampLimit)) {
                      clamped++;
                    }
                  }
                }
              }
            } else {
              let srcBase = readSlot * inChannels;
              for (let f = 0; f < firstFrames; f++) {
                for (let c = 0; c < scanChannels; c++) {
                  const s = samples[srcBase + c];
                  if (s - s !== 0) {
                    sanitized++;
                  } else {
                    if (softLimit && (s > softKnee || s < negSoftKnee)) {
                      softClipped++;
                    }
                    if (clampLimit > 0.0 && (s > clampLimit || s < negClampLimit)) {
                      clamped++;
                    }
                  }
                }
                srcBase += inChannels;
              }
              if (secondFrames > 0) {
                srcBase = 0;
                for (let f = 0; f < secondFrames; f++) {
                  for (let c = 0; c < scanChannels; c++) {
                    const s = samples[srcBase + c];
                    if (s - s !== 0) {
                      sanitized++;
                    } else {
                      if (softLimit && (s > softKnee || s < negSoftKnee)) {
                        softClipped++;
                      }
                      if (clampLimit > 0.0 && (s > clampLimit || s < negClampLimit)) {
                        clamped++;
                      }
                    }
                  }
                  srcBase += inChannels;
                }
              }
            }
          } else if (gain === 1.0) {
            if (inChannels === 2 && outChannels === 2) {
              const outL = output[0];
              const outR = output[1];
              let src = readSlot * 2;
              for (let f = 0; f < firstFrames; f++) {
                let sL = samples[src];
                let sR = samples[src + 1];
                if (sL - sL !== 0) {
                  sL = 0.0;
                  sanitized++;
                } else {
                  if (softLimit) {
                    if (sL > softKnee) {
                      sL = softLimitPositive(sL, softKnee, softHeadroom);
                      softClipped++;
                    } else if (sL < negSoftKnee) {
                      sL = -softLimitPositive(-sL, softKnee, softHeadroom);
                      softClipped++;
                    }
                  }
                  if (clampLimit > 0.0) {
                    if (sL > clampLimit) {
                      sL = clampLimit;
                      clamped++;
                    } else if (sL < negClampLimit) {
                      sL = negClampLimit;
                      clamped++;
                    }
                  }
                }
                if (sR - sR !== 0) {
                  sR = 0.0;
                  sanitized++;
                } else {
                  if (softLimit) {
                    if (sR > softKnee) {
                      sR = softLimitPositive(sR, softKnee, softHeadroom);
                      softClipped++;
                    } else if (sR < negSoftKnee) {
                      sR = -softLimitPositive(-sR, softKnee, softHeadroom);
                      softClipped++;
                    }
                  }
                  if (clampLimit > 0.0) {
                    if (sR > clampLimit) {
                      sR = clampLimit;
                      clamped++;
                    } else if (sR < negClampLimit) {
                      sR = negClampLimit;
                      clamped++;
                    }
                  }
                }
                outL[f] = sL;
                outR[f] = sR;
                src += 2;
              }
              if (secondFrames > 0) {
                src = 0;
                for (let f = 0; f < secondFrames; f++) {
                  const dst = firstFrames + f;
                  let sL = samples[src];
                  let sR = samples[src + 1];
                  if (sL - sL !== 0) {
                    sL = 0.0;
                    sanitized++;
                  } else {
                    if (softLimit) {
                      if (sL > softKnee) {
                        sL = softLimitPositive(sL, softKnee, softHeadroom);
                        softClipped++;
                      } else if (sL < negSoftKnee) {
                        sL = -softLimitPositive(-sL, softKnee, softHeadroom);
                        softClipped++;
                      }
                    }
                    if (clampLimit > 0.0) {
                      if (sL > clampLimit) {
                        sL = clampLimit;
                        clamped++;
                      } else if (sL < negClampLimit) {
                        sL = negClampLimit;
                        clamped++;
                      }
                    }
                  }
                  if (sR - sR !== 0) {
                    sR = 0.0;
                    sanitized++;
                  } else {
                    if (softLimit) {
                      if (sR > softKnee) {
                        sR = softLimitPositive(sR, softKnee, softHeadroom);
                        softClipped++;
                      } else if (sR < negSoftKnee) {
                        sR = -softLimitPositive(-sR, softKnee, softHeadroom);
                        softClipped++;
                      }
                    }
                    if (clampLimit > 0.0) {
                      if (sR > clampLimit) {
                        sR = clampLimit;
                        clamped++;
                      } else if (sR < negClampLimit) {
                        sR = negClampLimit;
                        clamped++;
                      }
                    }
                  }
                  outL[dst] = sL;
                  outR[dst] = sR;
                  src += 2;
                }
              }
            } else if (inChannels === 1) {
              let src = readSlot;
              for (let f = 0; f < firstFrames; f++) {
                let s = samples[src];
                if (s - s !== 0) {
                  s = 0.0;
                  sanitized++;
                } else {
                  if (softLimit) {
                    if (s > softKnee) {
                      s = softLimitPositive(s, softKnee, softHeadroom);
                      softClipped++;
                    } else if (s < negSoftKnee) {
                      s = -softLimitPositive(-s, softKnee, softHeadroom);
                      softClipped++;
                    }
                  }
                  if (clampLimit > 0.0) {
                    if (s > clampLimit) {
                      s = clampLimit;
                      clamped++;
                    } else if (s < negClampLimit) {
                      s = negClampLimit;
                      clamped++;
                    }
                  }
                }
                for (let c = 0; c < outChannels; c++) {
                  output[c][f] = s;
                }
                src++;
              }
              if (secondFrames > 0) {
                src = 0;
                for (let f = 0; f < secondFrames; f++) {
                  const dst = firstFrames + f;
                  let s = samples[src];
                  if (s - s !== 0) {
                    s = 0.0;
                    sanitized++;
                  } else {
                    if (softLimit) {
                      if (s > softKnee) {
                        s = softLimitPositive(s, softKnee, softHeadroom);
                        softClipped++;
                      } else if (s < negSoftKnee) {
                        s = -softLimitPositive(-s, softKnee, softHeadroom);
                        softClipped++;
                      }
                    }
                    if (clampLimit > 0.0) {
                      if (s > clampLimit) {
                        s = clampLimit;
                        clamped++;
                      } else if (s < negClampLimit) {
                        s = negClampLimit;
                        clamped++;
                      }
                    }
                  }
                  for (let c = 0; c < outChannels; c++) {
                    output[c][dst] = s;
                  }
                  src++;
                }
              }
            } else {
              let srcBase = readSlot * inChannels;
              for (let f = 0; f < firstFrames; f++) {
                for (let c = 0; c < copyChannels; c++) {
                  let s = samples[srcBase + c];
                  if (s - s !== 0) {
                    s = 0.0;
                    sanitized++;
                  } else {
                    if (softLimit) {
                      if (s > softKnee) {
                        s = softLimitPositive(s, softKnee, softHeadroom);
                        softClipped++;
                      } else if (s < negSoftKnee) {
                        s = -softLimitPositive(-s, softKnee, softHeadroom);
                        softClipped++;
                      }
                    }
                    if (clampLimit > 0.0) {
                      if (s > clampLimit) {
                        s = clampLimit;
                        clamped++;
                      } else if (s < negClampLimit) {
                        s = negClampLimit;
                        clamped++;
                      }
                    }
                  }
                  output[c][f] = s;
                }
                for (let c = copyChannels; c < outChannels; c++) {
                  output[c][f] = 0.0;
                }
                srcBase += inChannels;
              }
              if (secondFrames > 0) {
                srcBase = 0;
                for (let f = 0; f < secondFrames; f++) {
                  const dst = firstFrames + f;
                  for (let c = 0; c < copyChannels; c++) {
                    let s = samples[srcBase + c];
                    if (s - s !== 0) {
                      s = 0.0;
                      sanitized++;
                    } else {
                      if (softLimit) {
                        if (s > softKnee) {
                          s = softLimitPositive(s, softKnee, softHeadroom);
                          softClipped++;
                        } else if (s < negSoftKnee) {
                          s = -softLimitPositive(-s, softKnee, softHeadroom);
                          softClipped++;
                        }
                      }
                      if (clampLimit > 0.0) {
                        if (s > clampLimit) {
                          s = clampLimit;
                          clamped++;
                        } else if (s < negClampLimit) {
                          s = negClampLimit;
                          clamped++;
                        }
                      }
                    }
                    output[c][dst] = s;
                  }
                  for (let c = copyChannels; c < outChannels; c++) {
                    output[c][dst] = 0.0;
                  }
                  srcBase += inChannels;
                }
              }
            }
          } else {
            if (inChannels === 2 && outChannels === 2) {
              const outL = output[0];
              const outR = output[1];
              let src = readSlot * 2;
              for (let f = 0; f < firstFrames; f++) {
                let sL = samples[src];
                let sR = samples[src + 1];
                if (sL - sL !== 0) {
                  sL = 0.0;
                  sanitized++;
                } else {
                  if (softLimit) {
                    if (sL > softKnee) {
                      sL = softLimitPositive(sL, softKnee, softHeadroom);
                      softClipped++;
                    } else if (sL < negSoftKnee) {
                      sL = -softLimitPositive(-sL, softKnee, softHeadroom);
                      softClipped++;
                    }
                  }
                  if (clampLimit > 0.0) {
                    if (sL > clampLimit) {
                      sL = clampLimit;
                      clamped++;
                    } else if (sL < negClampLimit) {
                      sL = negClampLimit;
                      clamped++;
                    }
                  }
                }
                if (sR - sR !== 0) {
                  sR = 0.0;
                  sanitized++;
                } else {
                  if (softLimit) {
                    if (sR > softKnee) {
                      sR = softLimitPositive(sR, softKnee, softHeadroom);
                      softClipped++;
                    } else if (sR < negSoftKnee) {
                      sR = -softLimitPositive(-sR, softKnee, softHeadroom);
                      softClipped++;
                    }
                  }
                  if (clampLimit > 0.0) {
                    if (sR > clampLimit) {
                      sR = clampLimit;
                      clamped++;
                    } else if (sR < negClampLimit) {
                      sR = negClampLimit;
                      clamped++;
                    }
                  }
                }
                outL[f] = sL * gain;
                outR[f] = sR * gain;
                src += 2;
              }
              if (secondFrames > 0) {
                src = 0;
                for (let f = 0; f < secondFrames; f++) {
                  const dst = firstFrames + f;
                  let sL = samples[src];
                  let sR = samples[src + 1];
                  if (sL - sL !== 0) {
                    sL = 0.0;
                    sanitized++;
                  } else {
                    if (softLimit) {
                      if (sL > softKnee) {
                        sL = softLimitPositive(sL, softKnee, softHeadroom);
                        softClipped++;
                      } else if (sL < negSoftKnee) {
                        sL = -softLimitPositive(-sL, softKnee, softHeadroom);
                        softClipped++;
                      }
                    }
                    if (clampLimit > 0.0) {
                      if (sL > clampLimit) {
                        sL = clampLimit;
                        clamped++;
                      } else if (sL < negClampLimit) {
                        sL = negClampLimit;
                        clamped++;
                      }
                    }
                  }
                  if (sR - sR !== 0) {
                    sR = 0.0;
                    sanitized++;
                  } else {
                    if (softLimit) {
                      if (sR > softKnee) {
                        sR = softLimitPositive(sR, softKnee, softHeadroom);
                        softClipped++;
                      } else if (sR < negSoftKnee) {
                        sR = -softLimitPositive(-sR, softKnee, softHeadroom);
                        softClipped++;
                      }
                    }
                    if (clampLimit > 0.0) {
                      if (sR > clampLimit) {
                        sR = clampLimit;
                        clamped++;
                      } else if (sR < negClampLimit) {
                        sR = negClampLimit;
                        clamped++;
                      }
                    }
                  }
                  outL[dst] = sL * gain;
                  outR[dst] = sR * gain;
                  src += 2;
                }
              }
            } else if (inChannels === 1) {
              let src = readSlot;
              for (let f = 0; f < firstFrames; f++) {
                let s = samples[src];
                if (s - s !== 0) {
                  s = 0.0;
                  sanitized++;
                } else {
                  if (softLimit) {
                    if (s > softKnee) {
                      s = softLimitPositive(s, softKnee, softHeadroom);
                      softClipped++;
                    } else if (s < negSoftKnee) {
                      s = -softLimitPositive(-s, softKnee, softHeadroom);
                      softClipped++;
                    }
                  }
                  if (clampLimit > 0.0) {
                    if (s > clampLimit) {
                      s = clampLimit;
                      clamped++;
                    } else if (s < negClampLimit) {
                      s = negClampLimit;
                      clamped++;
                    }
                  }
                }
                const v = s * gain;
                for (let c = 0; c < outChannels; c++) {
                  output[c][f] = v;
                }
                src++;
              }
              if (secondFrames > 0) {
                src = 0;
                for (let f = 0; f < secondFrames; f++) {
                  const dst = firstFrames + f;
                  let s = samples[src];
                  if (s - s !== 0) {
                    s = 0.0;
                    sanitized++;
                  } else {
                    if (softLimit) {
                      if (s > softKnee) {
                        s = softLimitPositive(s, softKnee, softHeadroom);
                        softClipped++;
                      } else if (s < negSoftKnee) {
                        s = -softLimitPositive(-s, softKnee, softHeadroom);
                        softClipped++;
                      }
                    }
                    if (clampLimit > 0.0) {
                      if (s > clampLimit) {
                        s = clampLimit;
                        clamped++;
                      } else if (s < negClampLimit) {
                        s = negClampLimit;
                        clamped++;
                      }
                    }
                  }
                  const v = s * gain;
                  for (let c = 0; c < outChannels; c++) {
                    output[c][dst] = v;
                  }
                  src++;
                }
              }
            } else {
              let srcBase = readSlot * inChannels;
              for (let f = 0; f < firstFrames; f++) {
                for (let c = 0; c < copyChannels; c++) {
                  let s = samples[srcBase + c];
                  if (s - s !== 0) {
                    s = 0.0;
                    sanitized++;
                  } else {
                    if (softLimit) {
                      if (s > softKnee) {
                        s = softLimitPositive(s, softKnee, softHeadroom);
                        softClipped++;
                      } else if (s < negSoftKnee) {
                        s = -softLimitPositive(-s, softKnee, softHeadroom);
                        softClipped++;
                      }
                    }
                    if (clampLimit > 0.0) {
                      if (s > clampLimit) {
                        s = clampLimit;
                        clamped++;
                      } else if (s < negClampLimit) {
                        s = negClampLimit;
                        clamped++;
                      }
                    }
                  }
                  output[c][f] = s * gain;
                }
                for (let c = copyChannels; c < outChannels; c++) {
                  output[c][f] = 0.0;
                }
                srcBase += inChannels;
              }
              if (secondFrames > 0) {
                srcBase = 0;
                for (let f = 0; f < secondFrames; f++) {
                  const dst = firstFrames + f;
                  for (let c = 0; c < copyChannels; c++) {
                    let s = samples[srcBase + c];
                    if (s - s !== 0) {
                      s = 0.0;
                      sanitized++;
                    } else {
                      if (softLimit) {
                        if (s > softKnee) {
                          s = softLimitPositive(s, softKnee, softHeadroom);
                          softClipped++;
                        } else if (s < negSoftKnee) {
                          s = -softLimitPositive(-s, softKnee, softHeadroom);
                          softClipped++;
                        }
                      }
                      if (clampLimit > 0.0) {
                        if (s > clampLimit) {
                          s = clampLimit;
                          clamped++;
                        } else if (s < negClampLimit) {
                          s = negClampLimit;
                          clamped++;
                        }
                      }
                    }
                    output[c][dst] = s * gain;
                  }
                  for (let c = copyChannels; c < outChannels; c++) {
                    output[c][dst] = 0.0;
                  }
                  srcBase += inChannels;
                }
              }
            }
          }
        }

        if (sanitized > 0) {
          this._sanitizedSamples += sanitized;
        }
        if (clamped > 0) {
          this._clampedSamples += clamped;
        }
        if (softClipped > 0) {
          this._softClippedSamples += softClipped;
        }

        // W327: Click-free half-cosine splice across overrun trim BEFORE advancing
        // HDR_READ_CURSOR so the producer cannot overwrite the discarded head slots.
        if (trimFrames > 0 && !wasInUnderrun && this._trimCrossfadeFrames > 0) {
          this._applyTrimCrossfade(
            output,
            outChannels,
            samples,
            cap,
            inChannels,
            trimOldRead,
            framesToRead,
            preLoopGain,
            preLoopRampRemaining
          );
        }

        let nextRead = read + framesToRead;
        if (nextRead >= modulus) {
          nextRead -= modulus;
        }
        Atomics.store(ctrl, HDR_READ_CURSOR, nextRead);

        // Apply smooth linear or half-cosine fade-in (or crossfade from interrupted fade-out) on recovery.
        if (this._fadeInTotal > 0) {
          const total = this._fadeInTotal;
          const startPos = this._fadeInPos;
          const remaining = total - startPos;
          const rampCount = framesToRead < remaining ? framesToRead : remaining;

          if (this._fadeInHasOffset) {
            const fadeInFrom = this._fadeInFrom;
            for (let f = 0; f < rampCount; f++) {
              const stepIdx = startPos + f;
              const alpha = useCosine
                ? stepIdx === 0
                  ? 0.0
                  : 0.5 * (1.0 - Math.cos(Math.PI * (stepIdx / total)))
                : stepIdx / total;
              if (alpha === 0.0) {
                for (let c = 0; c < trackedChannels; c++) {
                  output[c][f] = fadeInFrom[c];
                }
                for (let c = trackedChannels; c < outChannels; c++) {
                  output[c][f] = 0.0;
                }
              } else {
                const invAlpha = 1.0 - alpha;
                for (let c = 0; c < trackedChannels; c++) {
                  output[c][f] = fadeInFrom[c] * invAlpha + output[c][f] * alpha;
                }
                for (let c = trackedChannels; c < outChannels; c++) {
                  output[c][f] *= alpha;
                }
              }
            }
          } else {
            for (let f = 0; f < rampCount; f++) {
              const stepIdx = startPos + f;
              const alpha = useCosine
                ? stepIdx === 0
                  ? 0.0
                  : 0.5 * (1.0 - Math.cos(Math.PI * (stepIdx / total)))
                : stepIdx / total;
              if (alpha === 0.0) {
                for (let c = 0; c < outChannels; c++) {
                  output[c][f] = 0.0;
                }
              } else {
                for (let c = 0; c < outChannels; c++) {
                  output[c][f] *= alpha;
                }
              }
            }
          }

          this._fadeInPos = startPos + rampCount;
          if (this._fadeInPos >= total) {
            this._fadeInPos = 0;
            this._fadeInTotal = 0;
            this._fadeInHasOffset = false;
          }
        }

        // Record the last emitted finite frame for smooth underrun de-click fade-out.
        const lastIdx = framesToRead - 1;
        let hasNonZero = false;
        for (let c = 0; c < trackedChannels; c++) {
          const v = output[c][lastIdx];
          lastSamples[c] = v;
          if (v !== 0.0) {
            hasNonZero = true;
          }
        }
        for (let c = trackedChannels; c < lastSamples.length; c++) {
          lastSamples[c] = 0.0;
        }
        this._hasActiveAudio = hasNonZero;
      }

      if (framesToRead < quantumFrames) {
        const missingFrames = quantumFrames - framesToRead;
        const declickFrames = this._declickFrames;

        // Entering underrun cancels any in-flight recovery fade-in cleanly.
        this._fadeInPos = 0;
        this._fadeInTotal = 0;
        this._fadeInHasOffset = false;
        this._inUnderrun = true;

        if (declickFrames > 0) {
          if (this._declickTotal === 0 && this._hasActiveAudio) {
            this._declickPos = 0;
            this._declickTotal = declickFrames;
            this._declickFadeOuts++;
            this._hasActiveAudio = false;
          }

          if (this._declickTotal > 0) {
            const total = this._declickTotal;
            const startPos = this._declickPos;
            const remainingFade = total - startPos;
            const fadeCount = missingFrames < remainingFade ? missingFrames : remainingFade;

            for (let i = 0; i < fadeCount; i++) {
              const stepIdx = startPos + i;
              const scale = useCosine
                ? stepIdx === total - 1
                  ? 0.0
                  : 0.5 * (1.0 + Math.cos(Math.PI * ((stepIdx + 1) / total)))
                : (total - 1 - stepIdx) / total;
              const dst = framesToRead + i;
              if (scale === 0.0) {
                for (let c = 0; c < outChannels; c++) {
                  output[c][dst] = 0.0;
                }
              } else {
                for (let c = 0; c < trackedChannels; c++) {
                  output[c][dst] = lastSamples[c] * scale;
                }
                for (let c = trackedChannels; c < outChannels; c++) {
                  output[c][dst] = 0.0;
                }
              }
            }

            this._declickPos = startPos + fadeCount;
            if (this._declickPos >= total) {
              this._declickPos = 0;
              this._declickTotal = 0;
              lastSamples.fill(0);
            }

            if (framesToRead + fadeCount < quantumFrames) {
              for (let c = 0; c < outChannels; c++) {
                output[c].fill(0, framesToRead + fadeCount, quantumFrames);
              }
            }
          } else {
            for (let c = 0; c < outChannels; c++) {
              output[c].fill(0, framesToRead, quantumFrames);
            }
          }
        } else {
          for (let c = 0; c < outChannels; c++) {
            output[c].fill(0, framesToRead, quantumFrames);
          }
          this._hasActiveAudio = false;
          this._declickPos = 0;
          this._declickTotal = 0;
          lastSamples.fill(0);
        }

        // Advance any in-flight gain ramp across the silent underrun tail so it
        // converges cleanly rather than stalling until samples resume.
        if (this._rampRemaining > 0) {
          const skip = this._rampRemaining < missingFrames ? this._rampRemaining : missingFrames;
          this._rampRemaining -= skip;
          if (this._rampRemaining === 0) {
            this._currentGain = this._targetGain;
          } else {
            this._currentGain += this._rampStep * skip;
          }
        }

        this._underruns++;
        this._underrunQuanta++;
        this._underrunFrames += missingFrames;
        Atomics.add(ctrl, HDR_UNDERRUNS, 1);
        Atomics.add(ctrl, HDR_UNDERRUN_FRAMES, missingFrames);
      }

      this._processedQuanta++;
      this._processedFrames += framesToRead;

      const remainingBuffered = used - framesToRead;
      const postFillRatio = cap > 0 ? clampFillRatio(remainingBuffered / cap) : 0.0;
      this._lastFillRatio = postFillRatio;

      if (this._statsIntervalQuanta > 0) {
        this._statsQuantaCounter++;
        if (this._statsQuantaCounter >= this._statsIntervalQuanta) {
          this._statsQuantaCounter = 0;
          this._emitPeriodicStats(remainingBuffered, postFillRatio);
        }
      }

      return true;
    }
  }

  if (typeof registerProcessor === 'function') {
    registerProcessor(PROCESSOR_NAME, PlayGta6AudioWorkletProcessor);
  }

  const exportsObj = {
    PlayGta6AudioWorkletProcessor,
    applyFadeRamp,
    softLimitSample,
    sanitizeAndSoftLimitBuffer,
    clampFillRatio,
    PROCESSOR_NAME,
    DEFAULT_CHANNELS,
    DEFAULT_SMOOTHING_FRAMES,
    DEFAULT_DECLICK_FRAMES,
    DEFAULT_FADE_IN_FRAMES,
    DEFAULT_UNDERRUN_RAMP_FRAMES,
    MAX_DECLICK_FRAMES,
    DEFAULT_SAMPLE_CLAMP_LIMIT,
    DEFAULT_SOFT_LIMIT_KNEE,
    DEFAULT_MAX_BUFFERED_FRAMES,
    DEFAULT_HIGH_WATERMARK_FRAMES,
    DEFAULT_TARGET_WATERMARK_FRAMES,
    DEFAULT_TRIM_CROSSFADE_FRAMES,
    MAX_TRIM_CROSSFADE_FRAMES,
    DEFAULT_STATS_INTERVAL_QUANTA,
    HDR_WRITE_CURSOR,
    HDR_READ_CURSOR,
    HDR_CAPACITY_FRAMES,
    HDR_CHANNELS,
    HDR_UNDERRUNS,
    HDR_UNDERRUN_FRAMES,
    HDR_OVERFLOWS,
    HDR_DROPPED_FRAMES,
    RING_HEADER_INT32_COUNT,
    RING_HEADER_BYTES,
  };

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6AudioWorklet = exportsObj;
  } else if (root) {
    root.PlayGta6AudioWorklet = exportsObj;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportsObj;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : undefined);
