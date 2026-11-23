'use strict';

(function (root) {
  const PROCESSOR_NAME = 'playgta6-audio-processor';
  const DEFAULT_CHANNELS = 2;
  const DEFAULT_SMOOTHING_FRAMES = 64;

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

      this._targetGain = this._muted ? 0.0 : this._gain;
      this._currentGain = this._targetGain;
      this._rampStep = 0.0;
      this._rampRemaining = 0;

      this._stopped = false;
      this._processedQuanta = 0;
      this._processedFrames = 0;

      if (procOpts.sab !== undefined && procOpts.sab !== null) {
        this._attachSab(procOpts.sab);
      }

      if (this.port) {
        this.port.onmessage = (event) => {
          this._handleMessage(event ? event.data : null);
        };
      }
    }

    _attachSab(sab) {
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
      return {
        attached: ctrl !== null,
        stopped: this._stopped,
        gain: this._gain,
        muted: this._muted,
        currentGain: this._currentGain,
        targetGain: this._targetGain,
        capacityFrames: this._capacityFrames,
        channels: this._channels,
        bufferedFrames: this._computeBufferedFrames(),
        underruns: ctrl ? Atomics.load(ctrl, HDR_UNDERRUNS) : 0,
        underrunFrames: ctrl ? Atomics.load(ctrl, HDR_UNDERRUN_FRAMES) : 0,
        overflows: ctrl ? Atomics.load(ctrl, HDR_OVERFLOWS) : 0,
        droppedFrames: ctrl ? Atomics.load(ctrl, HDR_DROPPED_FRAMES) : 0,
        processedQuanta: this._processedQuanta,
        processedFrames: this._processedFrames,
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
     * Strictly zero allocations on the hot audio thread.
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

      const framesToRead = quantumFrames <= used ? quantumFrames : used;

      if (framesToRead > 0) {
        const readSlot = read >= cap ? read - cap : read;
        const firstFrames = framesToRead <= cap - readSlot ? framesToRead : cap - readSlot;
        const secondFrames = framesToRead - firstFrames;

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
              if (gain === 0.0) {
                outL[f] = 0.0;
                outR[f] = 0.0;
              } else {
                outL[f] = samples[src] * gain;
                outR[f] = samples[src + 1] * gain;
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
                if (gain === 0.0) {
                  outL[dst] = 0.0;
                  outR[dst] = 0.0;
                } else {
                  outL[dst] = samples[src] * gain;
                  outR[dst] = samples[src + 1] * gain;
                }
                src += 2;
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
              for (let c = 0; c < outChannels; c++) {
                if (gain === 0.0) {
                  output[c][f] = 0.0;
                } else {
                  const s =
                    c < inChannels ? samples[srcBase + c] : inChannels === 1 ? samples[srcBase] : 0.0;
                  output[c][f] = s * gain;
                }
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
                for (let c = 0; c < outChannels; c++) {
                  if (gain === 0.0) {
                    output[c][dst] = 0.0;
                  } else {
                    const s =
                      c < inChannels
                        ? samples[srcBase + c]
                        : inChannels === 1
                          ? samples[srcBase]
                          : 0.0;
                    output[c][dst] = s * gain;
                  }
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
          } else if (gain === 1.0) {
            if (inChannels === 2 && outChannels === 2) {
              const outL = output[0];
              const outR = output[1];
              let src = readSlot * 2;
              for (let f = 0; f < firstFrames; f++) {
                outL[f] = samples[src];
                outR[f] = samples[src + 1];
                src += 2;
              }
              if (secondFrames > 0) {
                src = 0;
                for (let f = 0; f < secondFrames; f++) {
                  const dst = firstFrames + f;
                  outL[dst] = samples[src];
                  outR[dst] = samples[src + 1];
                  src += 2;
                }
              }
            } else {
              let srcBase = readSlot * inChannels;
              for (let f = 0; f < firstFrames; f++) {
                for (let c = 0; c < outChannels; c++) {
                  output[c][f] =
                    c < inChannels
                      ? samples[srcBase + c]
                      : inChannels === 1
                        ? samples[srcBase]
                        : 0.0;
                }
                srcBase += inChannels;
              }
              if (secondFrames > 0) {
                srcBase = 0;
                for (let f = 0; f < secondFrames; f++) {
                  const dst = firstFrames + f;
                  for (let c = 0; c < outChannels; c++) {
                    output[c][dst] =
                      c < inChannels
                        ? samples[srcBase + c]
                        : inChannels === 1
                          ? samples[srcBase]
                          : 0.0;
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
                outL[f] = samples[src] * gain;
                outR[f] = samples[src + 1] * gain;
                src += 2;
              }
              if (secondFrames > 0) {
                src = 0;
                for (let f = 0; f < secondFrames; f++) {
                  const dst = firstFrames + f;
                  outL[dst] = samples[src] * gain;
                  outR[dst] = samples[src + 1] * gain;
                  src += 2;
                }
              }
            } else {
              let srcBase = readSlot * inChannels;
              for (let f = 0; f < firstFrames; f++) {
                for (let c = 0; c < outChannels; c++) {
                  const s =
                    c < inChannels
                      ? samples[srcBase + c]
                      : inChannels === 1
                        ? samples[srcBase]
                        : 0.0;
                  output[c][f] = s * gain;
                }
                srcBase += inChannels;
              }
              if (secondFrames > 0) {
                srcBase = 0;
                for (let f = 0; f < secondFrames; f++) {
                  const dst = firstFrames + f;
                  for (let c = 0; c < outChannels; c++) {
                    const s =
                      c < inChannels
                        ? samples[srcBase + c]
                        : inChannels === 1
                          ? samples[srcBase]
                          : 0.0;
                    output[c][dst] = s * gain;
                  }
                  srcBase += inChannels;
                }
              }
            }
          }
        }

        let nextRead = read + framesToRead;
        if (nextRead >= modulus) {
          nextRead -= modulus;
        }
        Atomics.store(ctrl, HDR_READ_CURSOR, nextRead);
      }

      if (framesToRead < quantumFrames) {
        const missingFrames = quantumFrames - framesToRead;
        for (let c = 0; c < outChannels; c++) {
          output[c].fill(0, framesToRead, quantumFrames);
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
