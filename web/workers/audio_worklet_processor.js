'use strict';

(function (root) {
  const PROCESSOR_NAME = 'playgta6-audio-processor';
  const DEFAULT_CHANNELS = 2;
  const DEFAULT_SMOOTHING_FRAMES = 64;
  const DEFAULT_DECLICK_FRAMES = 32;
  const MAX_DECLICK_FRAMES = 4096;
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

      this._targetGain = this._muted ? 0.0 : this._gain;
      this._currentGain = this._targetGain;
      this._rampStep = 0.0;
      this._rampRemaining = 0;

      // Pre-allocated fixed-size per-channel sample history for zero-GC underrun de-click fade-out.
      this._lastSamples = new Float32Array(MAX_TRACKED_CHANNELS);
      this._hasActiveAudio = false;
      this._declickPos = 0;
      this._declickTotal = 0;
      this._declickFadeOuts = 0;

      this._stopped = false;
      this._processedQuanta = 0;
      this._processedFrames = 0;
      this._sanitizedSamples = 0;
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

    get underruns() {
      return this._ctrl ? Atomics.load(this._ctrl, HDR_UNDERRUNS) : this._underruns;
    }

    get underrunFrames() {
      return this._ctrl ? Atomics.load(this._ctrl, HDR_UNDERRUN_FRAMES) : this._underrunFrames;
    }

    get declickFrames() {
      return this._declickFrames;
    }

    get fadeOutCount() {
      return this._declickFadeOuts;
    }

    setDeclickFrames(frames) {
      this._declickFrames = sanitizeDeclickFrames(frames, this._declickFrames);
      if (this._declickFrames === 0) {
        this._declickPos = 0;
        this._declickTotal = 0;
      }
      return this._declickFrames;
    }

    _resetDeclickState() {
      this._lastSamples.fill(0);
      this._hasActiveAudio = false;
      this._declickPos = 0;
      this._declickTotal = 0;
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
      const write = Atomics.load(this._ctrl, HDR_WRITE_CURSOR);
      const read = Atomics.load(this._ctrl, HDR_READ_CURSOR);
      let used = write - read;
      if (used < 0) {
        used += this._modulus;
      }
      return used;
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
        capacityFrames: this._capacityFrames,
        channels: this._channels,
        bufferedFrames: this._computeBufferedFrames(),
        underruns: ctrl ? Atomics.load(ctrl, HDR_UNDERRUNS) : this._underruns,
        underrunFrames: ctrl ? Atomics.load(ctrl, HDR_UNDERRUN_FRAMES) : this._underrunFrames,
        fadeOutCount: this._declickFadeOuts,
        overflows,
        overruns: overflows,
        droppedFrames: ctrl ? Atomics.load(ctrl, HDR_DROPPED_FRAMES) : 0,
        processedQuanta: this._processedQuanta,
        processedFrames: this._processedFrames,
        framesProcessed: this._processedFrames,
        sanitizedSamples: this._sanitizedSamples,
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
     * Sanitizes any non-finite (`NaN`/`±Infinity`) PCM sample to `0.0` and,
     * when `declickFrames > 0`, applies a short linear fade-out de-click ramp
     * from the last emitted sample when an underrun begins.
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
            out[c].fill(0);
          }
        }
        return false;
      }

      if (!hasOutput) {
        return true;
      }

      const output = outputs[0];
      const outChannels = output.length;
      const quantumFrames = output[0].length;
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

      const read = Atomics.load(ctrl, HDR_READ_CURSOR);
      const write = Atomics.load(ctrl, HDR_WRITE_CURSOR);
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

      if (framesToRead > 0) {
        // Fresh audio arrived: cancel any in-flight cross-quantum underrun fade-out.
        this._declickPos = 0;
        this._declickTotal = 0;

        const readSlot = read >= cap ? read - cap : read;
        const firstFrames = framesToRead <= cap - readSlot ? framesToRead : cap - readSlot;
        const secondFrames = framesToRead - firstFrames;
        const copyChannels = inChannels < outChannels ? inChannels : outChannels;
        let sanitized = 0;

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
              }
              if (sR - sR !== 0) {
                sR = 0.0;
                sanitized++;
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
                }
                if (sR - sR !== 0) {
                  sR = 0.0;
                  sanitized++;
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
            // Scan consumed ring samples so sanitizedSamples telemetry stays exact while muted.
            const scanChannels = inChannels === 1 ? 1 : copyChannels;
            if (scanChannels === inChannels) {
              const firstCount = firstFrames * inChannels;
              let src = readSlot * inChannels;
              for (let i = 0; i < firstCount; i++) {
                const s = samples[src + i];
                if (s - s !== 0) sanitized++;
              }
              if (secondFrames > 0) {
                const secondCount = secondFrames * inChannels;
                for (let i = 0; i < secondCount; i++) {
                  const s = samples[i];
                  if (s - s !== 0) sanitized++;
                }
              }
            } else {
              let srcBase = readSlot * inChannels;
              for (let f = 0; f < firstFrames; f++) {
                for (let c = 0; c < scanChannels; c++) {
                  const s = samples[srcBase + c];
                  if (s - s !== 0) sanitized++;
                }
                srcBase += inChannels;
              }
              if (secondFrames > 0) {
                srcBase = 0;
                for (let f = 0; f < secondFrames; f++) {
                  for (let c = 0; c < scanChannels; c++) {
                    const s = samples[srcBase + c];
                    if (s - s !== 0) sanitized++;
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
                }
                if (sR - sR !== 0) {
                  sR = 0.0;
                  sanitized++;
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
                  }
                  if (sR - sR !== 0) {
                    sR = 0.0;
                    sanitized++;
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
                }
                if (sR - sR !== 0) {
                  sR = 0.0;
                  sanitized++;
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
                  }
                  if (sR - sR !== 0) {
                    sR = 0.0;
                    sanitized++;
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

        let nextRead = read + framesToRead;
        if (nextRead >= modulus) {
          nextRead -= modulus;
        }
        Atomics.store(ctrl, HDR_READ_CURSOR, nextRead);

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
    MAX_DECLICK_FRAMES,
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
