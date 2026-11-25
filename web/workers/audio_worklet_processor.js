'use strict';

(function (root) {
  const PROCESSOR_NAME = 'playgta6-audio-processor';
  const DEFAULT_CHANNELS = 2;
  const DEFAULT_SMOOTHING_FRAMES = 64;
  const DEFAULT_DECLICK_FRAMES = 32;
  const DEFAULT_FADE_IN_FRAMES = DEFAULT_DECLICK_FRAMES;
  const MAX_DECLICK_FRAMES = 4096;
  const DEFAULT_SAMPLE_CLAMP_LIMIT = 1.0;
  const MAX_TRACKED_CHANNELS = 32;

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
      const rawDeclick =
        procOpts.declickFrames !== undefined
          ? procOpts.declickFrames
          : procOpts.underrunFadeFrames;
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
      this._underruns = 0;
      this._underrunFrames = 0;

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

    get clampedSamples() {
      return this._clampedSamples;
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

    get fadeInFrames() {
      return this._fadeInFramesOverride >= 0 ? this._fadeInFramesOverride : this._declickFrames;
    }

    get fadeOutCount() {
      return this._declickFadeOuts;
    }

    get fadeInCount() {
      return this._fadeInCount;
    }

    get inUnderrun() {
      return this._inUnderrun;
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

    _updateTargetGain(immediate = false) {
      const nextTarget = this._muted ? 0.0 : this._gain;
      this._targetGain = nextTarget;

      if (immediate || this._smoothingFrames <= 0 || this._currentGain === nextTarget) {
        this._currentGain = nextTarget;
        this._rampStep = 0.0;
        this._rampRemaining = 0;
      } else {
        this._rampRemaining = this._smoothingFrames;
        this._rampStep = (nextTarget - this._currentGain) / this._smoothingFrames;
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
      return {
        attached: ctrl !== null,
        stopped: this._stopped,
        gain: this._gain,
        muted: this._muted,
        currentGain: this._currentGain,
        targetGain: this._targetGain,
        declickFrames: this._declickFrames,
        fadeInFrames: this.fadeInFrames,
        clampSamples: this._clampLimit > 0.0,
        clampLimit: this._clampLimit,
        capacityFrames: this._capacityFrames,
        channels: this._channels,
        bufferedFrames: this._computeBufferedFrames(),
        underruns: ctrl ? Atomics.load(ctrl, HDR_UNDERRUNS) : this._underruns,
        underrunFrames: ctrl ? Atomics.load(ctrl, HDR_UNDERRUN_FRAMES) : this._underrunFrames,
        inUnderrun: this._inUnderrun,
        fadeOutCount: this._declickFadeOuts,
        fadeInCount: this._fadeInCount,
        overflows,
        overruns: overflows,
        droppedFrames: ctrl ? Atomics.load(ctrl, HDR_DROPPED_FRAMES) : 0,
        processedQuanta: this._processedQuanta,
        processedFrames: this._processedFrames,
        framesProcessed: this._processedFrames,
        sanitizedSamples: this._sanitizedSamples,
        clampedSamples: this._clampedSamples,
      };
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
          this._updateTargetGain(Boolean(msg.immediate));
          break;
        }

        case 'setMuted': {
          this._muted = Boolean(msg.muted);
          this._updateTargetGain(Boolean(msg.immediate));
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

      const framesToRead = quantumFrames <= used ? quantumFrames : used;
      const lastSamples = this._lastSamples;
      const trackedChannels =
        outChannels < lastSamples.length ? outChannels : lastSamples.length;
      const clampLimit = this._clampLimit;
      const negClampLimit = -clampLimit;

      if (framesToRead > 0) {
        const wasInUnderrun = this._inUnderrun;
        const wasMidFadeOut = this._declickTotal > 0;
        const fadeInFrames =
          this._fadeInFramesOverride >= 0 ? this._fadeInFramesOverride : this._declickFrames;

        if (wasInUnderrun) {
          this._inUnderrun = false;
          if (fadeInFrames > 0) {
            if (wasMidFadeOut) {
              const prevScale = (this._declickTotal - this._declickPos) / this._declickTotal;
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
        let sanitized = 0;
        let clamped = 0;

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
              } else if (clampLimit > 0.0) {
                if (sL > clampLimit) {
                  sL = clampLimit;
                  clamped++;
                } else if (sL < negClampLimit) {
                  sL = negClampLimit;
                  clamped++;
                }
              }
              if (sR - sR !== 0) {
                sR = 0.0;
                sanitized++;
              } else if (clampLimit > 0.0) {
                if (sR > clampLimit) {
                  sR = clampLimit;
                  clamped++;
                } else if (sR < negClampLimit) {
                  sR = negClampLimit;
                  clamped++;
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
                } else if (clampLimit > 0.0) {
                  if (sL > clampLimit) {
                    sL = clampLimit;
                    clamped++;
                  } else if (sL < negClampLimit) {
                    sL = negClampLimit;
                    clamped++;
                  }
                }
                if (sR - sR !== 0) {
                  sR = 0.0;
                  sanitized++;
                } else if (clampLimit > 0.0) {
                  if (sR > clampLimit) {
                    sR = clampLimit;
                    clamped++;
                  } else if (sR < negClampLimit) {
                    sR = negClampLimit;
                    clamped++;
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
              } else if (clampLimit > 0.0) {
                if (s > clampLimit) {
                  s = clampLimit;
                  clamped++;
                } else if (s < negClampLimit) {
                  s = negClampLimit;
                  clamped++;
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
                } else if (clampLimit > 0.0) {
                  if (s > clampLimit) {
                    s = clampLimit;
                    clamped++;
                  } else if (s < negClampLimit) {
                    s = negClampLimit;
                    clamped++;
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
                } else if (clampLimit > 0.0) {
                  if (s > clampLimit) {
                    s = clampLimit;
                    clamped++;
                  } else if (s < negClampLimit) {
                    s = negClampLimit;
                    clamped++;
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
                  } else if (clampLimit > 0.0) {
                    if (s > clampLimit) {
                      s = clampLimit;
                      clamped++;
                    } else if (s < negClampLimit) {
                      s = negClampLimit;
                      clamped++;
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
            // Scan consumed ring samples so sanitized/clamped telemetry stays exact while muted.
            const scanChannels = inChannels === 1 ? 1 : copyChannels;
            if (scanChannels === inChannels) {
              const firstCount = firstFrames * inChannels;
              let src = readSlot * inChannels;
              for (let i = 0; i < firstCount; i++) {
                const s = samples[src + i];
                if (s - s !== 0) {
                  sanitized++;
                } else if (clampLimit > 0.0 && (s > clampLimit || s < negClampLimit)) {
                  clamped++;
                }
              }
              if (secondFrames > 0) {
                const secondCount = secondFrames * inChannels;
                for (let i = 0; i < secondCount; i++) {
                  const s = samples[i];
                  if (s - s !== 0) {
                    sanitized++;
                  } else if (clampLimit > 0.0 && (s > clampLimit || s < negClampLimit)) {
                    clamped++;
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
                  } else if (clampLimit > 0.0 && (s > clampLimit || s < negClampLimit)) {
                    clamped++;
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
                    } else if (clampLimit > 0.0 && (s > clampLimit || s < negClampLimit)) {
                      clamped++;
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
                } else if (clampLimit > 0.0) {
                  if (sL > clampLimit) {
                    sL = clampLimit;
                    clamped++;
                  } else if (sL < negClampLimit) {
                    sL = negClampLimit;
                    clamped++;
                  }
                }
                if (sR - sR !== 0) {
                  sR = 0.0;
                  sanitized++;
                } else if (clampLimit > 0.0) {
                  if (sR > clampLimit) {
                    sR = clampLimit;
                    clamped++;
                  } else if (sR < negClampLimit) {
                    sR = negClampLimit;
                    clamped++;
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
                  } else if (clampLimit > 0.0) {
                    if (sL > clampLimit) {
                      sL = clampLimit;
                      clamped++;
                    } else if (sL < negClampLimit) {
                      sL = negClampLimit;
                      clamped++;
                    }
                  }
                  if (sR - sR !== 0) {
                    sR = 0.0;
                    sanitized++;
                  } else if (clampLimit > 0.0) {
                    if (sR > clampLimit) {
                      sR = clampLimit;
                      clamped++;
                    } else if (sR < negClampLimit) {
                      sR = negClampLimit;
                      clamped++;
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
                } else if (clampLimit > 0.0) {
                  if (s > clampLimit) {
                    s = clampLimit;
                    clamped++;
                  } else if (s < negClampLimit) {
                    s = negClampLimit;
                    clamped++;
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
                  } else if (clampLimit > 0.0) {
                    if (s > clampLimit) {
                      s = clampLimit;
                      clamped++;
                    } else if (s < negClampLimit) {
                      s = negClampLimit;
                      clamped++;
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
                  } else if (clampLimit > 0.0) {
                    if (s > clampLimit) {
                      s = clampLimit;
                      clamped++;
                    } else if (s < negClampLimit) {
                      s = negClampLimit;
                      clamped++;
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
                    } else if (clampLimit > 0.0) {
                      if (s > clampLimit) {
                        s = clampLimit;
                        clamped++;
                      } else if (s < negClampLimit) {
                        s = negClampLimit;
                        clamped++;
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
                } else if (clampLimit > 0.0) {
                  if (sL > clampLimit) {
                    sL = clampLimit;
                    clamped++;
                  } else if (sL < negClampLimit) {
                    sL = negClampLimit;
                    clamped++;
                  }
                }
                if (sR - sR !== 0) {
                  sR = 0.0;
                  sanitized++;
                } else if (clampLimit > 0.0) {
                  if (sR > clampLimit) {
                    sR = clampLimit;
                    clamped++;
                  } else if (sR < negClampLimit) {
                    sR = negClampLimit;
                    clamped++;
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
                  } else if (clampLimit > 0.0) {
                    if (sL > clampLimit) {
                      sL = clampLimit;
                      clamped++;
                    } else if (sL < negClampLimit) {
                      sL = negClampLimit;
                      clamped++;
                    }
                  }
                  if (sR - sR !== 0) {
                    sR = 0.0;
                    sanitized++;
                  } else if (clampLimit > 0.0) {
                    if (sR > clampLimit) {
                      sR = clampLimit;
                      clamped++;
                    } else if (sR < negClampLimit) {
                      sR = negClampLimit;
                      clamped++;
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
                } else if (clampLimit > 0.0) {
                  if (s > clampLimit) {
                    s = clampLimit;
                    clamped++;
                  } else if (s < negClampLimit) {
                    s = negClampLimit;
                    clamped++;
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
                  } else if (clampLimit > 0.0) {
                    if (s > clampLimit) {
                      s = clampLimit;
                      clamped++;
                    } else if (s < negClampLimit) {
                      s = negClampLimit;
                      clamped++;
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
                  } else if (clampLimit > 0.0) {
                    if (s > clampLimit) {
                      s = clampLimit;
                      clamped++;
                    } else if (s < negClampLimit) {
                      s = negClampLimit;
                      clamped++;
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
                    } else if (clampLimit > 0.0) {
                      if (s > clampLimit) {
                        s = clampLimit;
                        clamped++;
                      } else if (s < negClampLimit) {
                        s = negClampLimit;
                        clamped++;
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

        let nextRead = read + framesToRead;
        if (nextRead >= modulus) {
          nextRead -= modulus;
        }
        Atomics.store(ctrl, HDR_READ_CURSOR, nextRead);

        // Apply smooth linear fade-in (or crossfade from interrupted fade-out) on recovery.
        if (this._fadeInTotal > 0) {
          const total = this._fadeInTotal;
          const startPos = this._fadeInPos;
          const remaining = total - startPos;
          const rampCount = framesToRead < remaining ? framesToRead : remaining;

          if (this._fadeInHasOffset) {
            const fadeInFrom = this._fadeInFrom;
            for (let f = 0; f < rampCount; f++) {
              const alpha = (startPos + f) / total;
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
              const alpha = (startPos + f) / total;
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
              const scale = (total - 1 - (startPos + i)) / total;
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
        this._underrunFrames += missingFrames;
        Atomics.add(ctrl, HDR_UNDERRUNS, 1);
        Atomics.add(ctrl, HDR_UNDERRUN_FRAMES, missingFrames);
      }

      this._processedQuanta++;
      this._processedFrames += framesToRead;
      return true;
    }
  }

  if (typeof registerProcessor === 'function') {
    registerProcessor(PROCESSOR_NAME, PlayGta6AudioWorkletProcessor);
  }

  const exportsObj = {
    PlayGta6AudioWorkletProcessor,
    PROCESSOR_NAME,
    DEFAULT_CHANNELS,
    DEFAULT_SMOOTHING_FRAMES,
    DEFAULT_DECLICK_FRAMES,
    DEFAULT_FADE_IN_FRAMES,
    MAX_DECLICK_FRAMES,
    DEFAULT_SAMPLE_CLAMP_LIMIT,
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
