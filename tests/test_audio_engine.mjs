import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

import {
  AudioEngine,
  PROCESSOR_NAME,
  DEFAULT_MAX_RESUME_RETRIES,
  DEFAULT_GESTURE_EVENTS,
  DEFAULT_GAIN_RAMP_MS,
  MAX_GAIN_RAMP_MS,
  DEFAULT_UNLOCK_FADE_IN_MS,
  MAX_ERROR_MESSAGE_LENGTH,
  MIN_TARGET_LATENCY_MS,
  MAX_TARGET_LATENCY_MS,
  DEFAULT_TARGET_LATENCY_MS,
  AUDIO_PREFS_STORAGE_KEY,
  clampVolume,
  clampGain,
  clampTargetLatencyMs,
  saveAudioPreferences,
  loadAudioPreferences,
  sanitizeResumeError,
  createAudioRingSab,
  HDR_UNDERRUNS,
  HDR_UNDERRUN_FRAMES,
  HDR_OVERFLOWS,
  HDR_DROPPED_FRAMES,
} from '../web/audio/audio_engine.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const require = createRequire(import.meta.url);
const { AudioRingBuffer } = require('../web/workers/audio_worker.js');
const workletMod = require('../web/workers/audio_worklet_processor.js');
const { PlayGta6AudioWorkletProcessor } = workletMod;

function makeStereoInterleaved(startIndex, frameCount) {
  const out = new Float32Array(frameCount * 2);
  for (let f = 0; f < frameCount; f++) {
    const idx = startIndex + f;
    out[f * 2] = Math.fround((idx + 1) * 0.003125);
    out[f * 2 + 1] = Math.fround(-(idx + 1) * 0.0015625);
  }
  return out;
}

function makePlanarOutput(channels = 2, frames = 128, fillValue = 0) {
  const out = [];
  for (let c = 0; c < channels; c++) {
    const ch = new Float32Array(frames);
    if (fillValue !== 0) {
      ch.fill(fillValue);
    }
    out.push(ch);
  }
  return [out];
}

function attachMessageRecorder(processor) {
  const messages = [];
  processor.port.postMessage = (msg) => {
    messages.push(msg);
  };
  return messages;
}

function createMockEventTarget() {
  const listeners = new Map();
  return {
    listeners,
    addEventListener(type, fn) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add(fn);
    },
    removeEventListener(type, fn) {
      const set = listeners.get(type);
      if (set) set.delete(fn);
    },
    dispatch(type, event = {}) {
      const set = listeners.get(type);
      if (!set) return;
      for (const fn of Array.from(set)) {
        fn(event);
      }
    },
    listenerCount(type) {
      return listeners.has(type) ? listeners.get(type).size : 0;
    },
  };
}

function createMockWebAudioEnv(options = {}) {
  const events = {
    contexts: [],
    nodes: [],
    addedModules: [],
  };

  class MockAudioContext {
    constructor(ctxOpts = {}) {
      this.sampleRate = options.actualSampleRate ?? ctxOpts.sampleRate ?? 48000;
      this.latencyHint = ctxOpts.latencyHint;
      this.state = options.initialState ?? 'suspended';
      this.destination = { kind: 'mock-destination' };
      this.resumeCalls = 0;
      this.closeCalls = 0;
      this._listeners = new Map();

      if (options.omitAudioWorklet) {
        this.audioWorklet = undefined;
      } else {
        this.audioWorklet = {
          addModule: async (url) => {
            events.addedModules.push(url);
            if (options.addModuleError) {
              throw options.addModuleError;
            }
          },
        };
      }

      events.contexts.push(this);
    }

    addEventListener(type, fn) {
      if (!this._listeners.has(type)) this._listeners.set(type, new Set());
      this._listeners.get(type).add(fn);
    }

    removeEventListener(type, fn) {
      const set = this._listeners.get(type);
      if (set) set.delete(fn);
    }

    _emit(type) {
      const set = this._listeners.get(type);
      if (!set) return;
      for (const fn of Array.from(set)) {
        fn({ target: this });
      }
    }

    async resume() {
      this.resumeCalls++;
      if (options.resumeError) {
        throw options.resumeError;
      }
      this.state = 'running';
      this._emit('statechange');
    }

    async close() {
      this.closeCalls++;
      this.state = 'closed';
      this._emit('statechange');
    }
  }

  class MockAudioWorkletNode {
    constructor(context, name, nodeOptions) {
      this.context = context;
      this.name = name;
      this.options = nodeOptions;
      this.connectedTo = null;
      this.disconnected = false;
      this.postedMessages = [];
      this.portClosed = false;
      this.port = {
        onmessage: null,
        postMessage: (msg) => {
          this.postedMessages.push(msg);
          if (this._processor && typeof this._processor.port.onmessage === 'function') {
            this._processor.port.onmessage({ data: msg });
          }
        },
        close: () => {
          this.portClosed = true;
        },
      };

      if (options.attachRealProcessor) {
        this._processor = new PlayGta6AudioWorkletProcessor(nodeOptions);
        this._processor.port.postMessage = (reply) => {
          if (typeof this.port.onmessage === 'function') {
            this.port.onmessage({ data: reply });
          }
        };
      }

      events.nodes.push(this);
    }

    connect(dest) {
      this.connectedTo = dest;
      return dest;
    }

    disconnect() {
      this.disconnected = true;
      this.connectedTo = null;
    }
  }

  return { MockAudioContext, MockAudioWorkletNode, events };
}

test('PlayGta6AudioWorkletProcessor exports UMD module and processor name', () => {
  assert.equal(typeof PlayGta6AudioWorkletProcessor, 'function');
  assert.equal(workletMod.PROCESSOR_NAME, 'playgta6-audio-processor');
  assert.equal(PROCESSOR_NAME, 'playgta6-audio-processor');
  assert.equal(
    globalThis.PlayGta6AudioWorklet.PlayGta6AudioWorkletProcessor,
    PlayGta6AudioWorkletProcessor
  );
});

test('PlayGta6AudioWorkletProcessor de-interleaves exact 128-frame stereo quanta from AudioRingBuffer SAB', () => {
  const ring = AudioRingBuffer.create(512, 2);
  const processor = new PlayGta6AudioWorkletProcessor({
    processorOptions: { sab: ring.sab },
  });

  // Push 256 stereo frames (2 full WebAudio quanta).
  const pcm = makeStereoInterleaved(0, 256);
  assert.equal(ring.pushSamples(pcm), 256);
  assert.equal(ring.availableFrames(), 256);

  for (let q = 0; q < 2; q++) {
    const outputs = makePlanarOutput(2, 128, -99.0);
    const keepAlive = processor.process([], outputs);
    assert.equal(keepAlive, true);

    const [outL, outR] = outputs[0];
    for (let f = 0; f < 128; f++) {
      const srcFrame = q * 128 + f;
      assert.equal(outL[f], pcm[srcFrame * 2]);
      assert.equal(outR[f], pcm[srcFrame * 2 + 1]);
    }
  }

  assert.equal(ring.availableFrames(), 0);
  assert.equal(ring.underruns, 0);
  assert.equal(ring.underrunFrames, 0);
});

test('PlayGta6AudioWorkletProcessor preserves exact sample order across SAB wrap-around and double-modulo wrap', () => {
  // Capacity 192 is intentionally not a multiple of 128 so every second 128-frame
  // quantum crosses the physical ring buffer end and wraps around to index 0.
  const ring = AudioRingBuffer.create(192, 2);
  const processor = new PlayGta6AudioWorkletProcessor({
    processorOptions: { sab: ring.sab },
  });

  let nextFrame = 0;
  for (let round = 0; round < 12; round++) {
    const chunk = makeStereoInterleaved(nextFrame, 128);
    assert.equal(ring.pushSamples(chunk), 128);

    const outputs = makePlanarOutput(2, 128, 7.0);
    assert.equal(processor.process([], outputs), true);

    const [outL, outR] = outputs[0];
    for (let f = 0; f < 128; f++) {
      assert.equal(outL[f], chunk[f * 2], `round ${round} frame ${f} L mismatch`);
      assert.equal(outR[f], chunk[f * 2 + 1], `round ${round} frame ${f} R mismatch`);
    }
    assert.equal(ring.availableFrames(), 0);
    nextFrame += 128;
  }

  assert.equal(ring.underruns, 0);
  assert.equal(ring.overflows, 0);
});

test('PlayGta6AudioWorkletProcessor handles N-channel (4ch) and mono-to-stereo upmix', () => {
  // 4-channel interleaved -> 4-channel planar
  const ring4 = AudioRingBuffer.create(256, 4);
  const pcm4 = new Float32Array(128 * 4);
  for (let f = 0; f < 128; f++) {
    pcm4[f * 4 + 0] = Math.fround(f * 0.01);
    pcm4[f * 4 + 1] = Math.fround(-f * 0.01);
    pcm4[f * 4 + 2] = Math.fround(0.5);
    pcm4[f * 4 + 3] = Math.fround(-0.25);
  }
  assert.equal(ring4.pushSamples(pcm4), 128);

  const proc4 = new PlayGta6AudioWorkletProcessor({
    processorOptions: { sab: ring4.sab },
  });
  const out4 = makePlanarOutput(4, 128);
  assert.equal(proc4.process([], out4), true);
  for (let f = 0; f < 128; f++) {
    assert.equal(out4[0][0][f], pcm4[f * 4 + 0]);
    assert.equal(out4[0][1][f], pcm4[f * 4 + 1]);
    assert.equal(out4[0][2][f], pcm4[f * 4 + 2]);
    assert.equal(out4[0][3][f], pcm4[f * 4 + 3]);
  }

  // Mono SAB -> stereo WebAudio output duplicates channel 0 into L and R.
  const ringMono = AudioRingBuffer.create(256, 1);
  const pcmMono = new Float32Array(128);
  for (let f = 0; f < 128; f++) pcmMono[f] = Math.fround((f + 1) * 0.005);
  assert.equal(ringMono.pushSamples(pcmMono), 128);

  const procMono = new PlayGta6AudioWorkletProcessor({
    processorOptions: { sab: ringMono.sab },
  });
  const outStereo = makePlanarOutput(2, 128);
  assert.equal(procMono.process([], outStereo), true);
  assert.deepEqual(Array.from(outStereo[0][0]), Array.from(pcmMono));
  assert.deepEqual(Array.from(outStereo[0][1]), Array.from(pcmMono));
});

test('PlayGta6AudioWorkletProcessor applies initial gain, smoothed gain ramping, and mute cleanly', () => {
  const ring = AudioRingBuffer.create(1024, 2);
  const processor = new PlayGta6AudioWorkletProcessor({
    processorOptions: {
      sab: ring.sab,
      gain: 0.5,
      muted: false,
      smoothingFrames: 32,
    },
  });

  // Push constant +1.0 / -1.0 stereo frames so output values directly reveal the applied gain.
  const ones = new Float32Array(128 * 2);
  for (let f = 0; f < 128; f++) {
    ones[f * 2] = 1.0;
    ones[f * 2 + 1] = -1.0;
  }

  // Quantum 1: initial gain=0.5 applies immediately from frame 0.
  assert.equal(ring.pushSamples(ones), 128);
  const out1 = makePlanarOutput(2, 128);
  assert.equal(processor.process([], out1), true);
  assert.equal(out1[0][0][0], 0.5);
  assert.equal(out1[0][1][0], -0.5);
  assert.equal(out1[0][0][127], 0.5);

  // Quantum 2: change gain to 1.0 -> ramps linearly over 32 frames, then holds exact 1.0.
  processor.port.onmessage({ data: { type: 'setGain', gain: 1.0 } });
  assert.equal(ring.pushSamples(ones), 128);
  const out2 = makePlanarOutput(2, 128);
  assert.equal(processor.process([], out2), true);
  assert.ok(out2[0][0][0] > 0.5 && out2[0][0][0] < 0.55);
  for (let f = 1; f < 32; f++) {
    assert.ok(out2[0][0][f] > out2[0][0][f - 1], `ramp should increase at frame ${f}`);
  }
  assert.equal(out2[0][0][31], 1.0);
  assert.equal(out2[0][0][127], 1.0);
  assert.equal(out2[0][1][127], -1.0);

  // Quantum 3: mute -> ramps down to 0 over 32 frames, then stays silent while consuming SAB frames.
  processor.port.onmessage({ data: { type: 'setMuted', muted: true } });
  assert.equal(ring.pushSamples(ones), 128);
  const out3 = makePlanarOutput(2, 128, 99.0);
  assert.equal(processor.process([], out3), true);
  assert.ok(out3[0][0][0] < 1.0 && out3[0][0][0] > 0.9);
  assert.equal(out3[0][0][31], 0.0);
  assert.equal(out3[0][0][127], 0.0);
  assert.equal(out3[0][1][127], 0.0);
  assert.equal(ring.availableFrames(), 0);

  // Quantum 4: steady-state muted fast path still consumes frames and outputs exact silence.
  assert.equal(ring.pushSamples(ones), 128);
  const out4 = makePlanarOutput(2, 128, 42.0);
  assert.equal(processor.process([], out4), true);
  assert.ok(out4[0][0].every((v) => v === 0));
  assert.ok(out4[0][1].every((v) => v === 0));
  assert.equal(ring.availableFrames(), 0);

  // Clamp out-of-range gain values.
  processor.port.onmessage({ data: { type: 'setGain', gain: 5.0, immediate: true } });
  processor.port.onmessage({ data: { type: 'setMuted', muted: false, immediate: true } });
  assert.equal(processor.getStats().currentGain, 1.0);
  processor.port.onmessage({ data: { type: 'setGain', gain: -2.0, immediate: true } });
  assert.equal(processor.getStats().currentGain, 0.0);
});

test('PlayGta6AudioWorkletProcessor zero-fills partial and full underruns and increments atomic counters', () => {
  const ring = AudioRingBuffer.create(256, 2);
  const processor = new PlayGta6AudioWorkletProcessor({
    processorOptions: { sab: ring.sab },
  });

  // Partial underrun: push 50 frames, request 128-frame quantum.
  const partial = makeStereoInterleaved(0, 50);
  assert.equal(ring.pushSamples(partial), 50);

  const outPartial = makePlanarOutput(2, 128, 88.5);
  assert.equal(processor.process([], outPartial), true);

  for (let f = 0; f < 50; f++) {
    assert.equal(outPartial[0][0][f], partial[f * 2]);
    assert.equal(outPartial[0][1][f], partial[f * 2 + 1]);
  }
  for (let f = 50; f < 128; f++) {
    assert.equal(outPartial[0][0][f], 0);
    assert.equal(outPartial[0][1][f], 0);
  }
  assert.equal(ring.underruns, 1);
  assert.equal(ring.underrunFrames, 78);

  // Full underrun: 0 frames available, request 128-frame quantum.
  const outEmpty = makePlanarOutput(2, 128, -12.25);
  assert.equal(processor.process([], outEmpty), true);
  assert.ok(outEmpty[0][0].every((v) => v === 0));
  assert.ok(outEmpty[0][1].every((v) => v === 0));
  assert.equal(ring.underruns, 2);
  assert.equal(ring.underrunFrames, 78 + 128);
});

test('PlayGta6AudioWorkletProcessor supports attach, stats, and stop returning false', () => {
  const processor = new PlayGta6AudioWorkletProcessor();
  const replies = attachMessageRecorder(processor);

  // Unattached processor outputs silence and keeps alive.
  const outUnattached = makePlanarOutput(2, 128, 3.5);
  assert.equal(processor.process([], outUnattached), true);
  assert.ok(outUnattached[0][0].every((v) => v === 0));

  // Attach a real SAB via control message.
  const ring = AudioRingBuffer.create(256, 2);
  processor.port.onmessage({ data: { type: 'attach', sab: ring.sab, id: 1 } });
  assert.equal(replies.length, 1);
  assert.deepEqual(replies[0], {
    type: 'attached',
    attached: true,
    capacityFrames: 256,
    channels: 2,
    id: 1,
  });

  const pcm = makeStereoInterleaved(10, 128);
  assert.equal(ring.pushSamples(pcm), 128);
  const outAttached = makePlanarOutput(2, 128);
  assert.equal(processor.process([], outAttached), true);
  assert.equal(outAttached[0][0][0], pcm[0]);
  assert.equal(outAttached[0][1][127], pcm[127 * 2 + 1]);

  // Stats query.
  processor.port.onmessage({ data: { type: 'stats', id: 2 } });
  const statsMsg = replies[replies.length - 1];
  assert.equal(statsMsg.type, 'stats');
  assert.equal(statsMsg.id, 2);
  assert.equal(statsMsg.attached, true);
  assert.equal(statsMsg.processedQuanta, 1);
  assert.equal(statsMsg.processedFrames, 128);
  assert.equal(statsMsg.underruns, 0);

  // Stop message causes process() to zero-fill and return false.
  processor.port.onmessage({ data: { type: 'stop', id: 3 } });
  assert.deepEqual(replies[replies.length - 1], { type: 'stopped', id: 3 });

  const outStopped = makePlanarOutput(2, 128, 5.0);
  assert.equal(processor.process([], outStopped), false);
  assert.ok(outStopped[0][0].every((v) => v === 0));
  assert.ok(outStopped[0][1].every((v) => v === 0));
});

test('AudioEngine full lifecycle: start, user-gesture unlock, volume/mute, live SAB telemetry, and destroy', async () => {
  const { MockAudioContext, MockAudioWorkletNode, events } = createMockWebAudioEnv({
    initialState: 'suspended',
    attachRealProcessor: true,
  });

  const ring = AudioRingBuffer.create(256, 2);
  const stateTransitions = [];

  const engine = new AudioEngine({
    sampleRate: 48000,
    sab: ring.sab,
    workletUrl: '/web/workers/audio_worklet_processor.js',
    volume: 0.75,
    muted: false,
    AudioContextCtor: MockAudioContext,
    AudioWorkletNodeCtor: MockAudioWorkletNode,
    onStateChange: (s) => stateTransitions.push(s),
  });

  assert.equal(engine.state, 'idle');
  assert.equal(engine.sab, ring.sab);

  const startRes = await engine.start();
  assert.equal(startRes.ok, true);
  assert.equal(startRes.state, 'suspended');
  assert.equal(startRes.sampleRate, 48000);
  assert.equal(events.addedModules.length, 1);
  assert.equal(events.addedModules[0], '/web/workers/audio_worklet_processor.js');
  assert.equal(events.nodes.length, 1);
  assert.equal(events.nodes[0].name, PROCESSOR_NAME);
  assert.equal(events.nodes[0].connectedTo, events.contexts[0].destination);
  assert.equal(events.nodes[0].options.processorOptions.sab, ring.sab);
  assert.equal(events.nodes[0].options.processorOptions.gain, 0.75);

  // Arm user-gesture unlock on mock target; firing pointerdown unlocks once and cleans up both listeners.
  const target = createMockEventTarget();
  engine.resumeOnUserGesture(target);
  assert.equal(target.listenerCount('pointerdown'), 1);
  assert.equal(target.listenerCount('keydown'), 1);

  target.dispatch('pointerdown');
  // Allow microtask queue for async resume() to settle.
  await Promise.resolve();
  assert.equal(events.contexts[0].resumeCalls, 1);
  assert.equal(engine.state, 'running');
  assert.equal(target.listenerCount('pointerdown'), 0);
  assert.equal(target.listenerCount('keydown'), 0);

  // Volume and mute controls clamp & post messages to the worklet port.
  assert.equal(engine.setVolume(0.4), 0.4);
  assert.equal(engine.setVolume(3.2), 1.0);
  assert.equal(engine.setVolume(-1.0), 0.0);
  assert.equal(engine.setMuted(true), true);
  assert.equal(engine.setMuted(false), false);

  assert.deepEqual(events.nodes[0].postedMessages, [
    { type: 'setGain', gain: 0.4 },
    { type: 'setGain', gain: 1.0 },
    { type: 'setGain', gain: 0.0 },
    { type: 'setMuted', muted: true },
    { type: 'setMuted', muted: false },
  ]);

  // Verify live getTelemetry() reads real atomic SAB counters.
  // Push 300 frames into a 256-frame ring -> 256 written, 1 overflow, 44 dropped frames.
  assert.equal(ring.pushSamples(makeStereoInterleaved(0, 300)), 256);

  let telemetry = engine.getTelemetry();
  assert.equal(telemetry.sampleRate, 48000);
  assert.equal(telemetry.state, 'running');
  assert.equal(telemetry.bufferedFrames, 256);
  assert.equal(telemetry.overflows, 1);
  assert.equal(telemetry.droppedFrames, 44);
  assert.equal(telemetry.underruns, 0);
  assert.equal(telemetry.underrunFrames, 0);

  // Drive the attached real processor through 3 quanta (2 full + 1 empty underrun).
  const proc = events.nodes[0]._processor;
  const out = makePlanarOutput(2, 128);
  assert.equal(proc.process([], out), true);
  assert.equal(proc.process([], out), true);
  assert.equal(proc.process([], out), true); // full underrun of 128 frames

  telemetry = engine.getTelemetry();
  assert.equal(telemetry.bufferedFrames, 0);
  assert.equal(telemetry.underruns, 1);
  assert.equal(telemetry.underrunFrames, 128);
  assert.equal(telemetry.overflows, 1);
  assert.equal(telemetry.droppedFrames, 44);

  // Destroy sends stop, disconnects node, closes context, and is idempotent.
  await engine.destroy();
  assert.equal(engine.state, 'closed');
  assert.equal(events.nodes[0].disconnected, true);
  assert.equal(events.nodes[0].portClosed, true);
  assert.equal(events.contexts[0].closeCalls, 1);
  assert.deepEqual(
    events.nodes[0].postedMessages[events.nodes[0].postedMessages.length - 1],
    { type: 'stop' }
  );
  assert.equal(proc.process([], out), false);
  assert.ok(stateTransitions.includes('closed'));

  // Second destroy() is a safe no-op.
  await engine.destroy();
  assert.equal(events.contexts[0].closeCalls, 1);
});

test('AudioEngine auto-allocates SAB compatible with AudioRingBuffer and cleans up unarmed gesture listeners on destroy', async () => {
  const { MockAudioContext, MockAudioWorkletNode } = createMockWebAudioEnv({
    initialState: 'suspended',
  });

  const engine = new AudioEngine({
    capacityFrames: 512,
    channels: 2,
    AudioContextCtor: MockAudioContext,
    AudioWorkletNodeCtor: MockAudioWorkletNode,
  });

  // AudioRingBuffer can wrap engine.sab directly and see exact capacity/channels.
  const producer = new AudioRingBuffer(engine.sab);
  assert.equal(producer.capacityFrames, 512);
  assert.equal(producer.channels, 2);

  await engine.start();

  const target = createMockEventTarget();
  engine.resumeOnUserGesture(target);
  assert.equal(target.listenerCount('pointerdown'), 1);
  assert.equal(target.listenerCount('keydown'), 1);

  // Destroying before a user gesture fires removes both listeners cleanly.
  await engine.destroy();
  assert.equal(target.listenerCount('pointerdown'), 0);
  assert.equal(target.listenerCount('keydown'), 0);
});

test('AudioEngine degrades gracefully without throwing when AudioContext or AudioWorklet is unavailable', async () => {
  const errors = [];

  // Case 1: No AudioContext constructor at all.
  const engineNoCtx = new AudioEngine({
    AudioContextCtor: null,
    AudioWorkletNodeCtor: null,
    onError: (err) => errors.push(err),
  });
  const res1 = await engineNoCtx.start();
  assert.equal(res1.ok, false);
  assert.equal(res1.state, 'unsupported');
  assert.ok(res1.error instanceof Error);
  assert.equal(engineNoCtx.getTelemetry().state, 'unsupported');
  assert.ok(engineNoCtx.getTelemetry().error.length > 0);
  assert.equal(errors.length, 1);

  // Case 2: AudioContext exists, but audioWorklet is absent (e.g. insecure context / older browser).
  const envNoWorklet = createMockWebAudioEnv({ omitAudioWorklet: true });
  const engineNoWorklet = new AudioEngine({
    AudioContextCtor: envNoWorklet.MockAudioContext,
    AudioWorkletNodeCtor: envNoWorklet.MockAudioWorkletNode,
    onError: (err) => errors.push(err),
  });
  const res2 = await engineNoWorklet.start();
  assert.equal(res2.ok, false);
  assert.equal(res2.state, 'unsupported');
  assert.equal(envNoWorklet.events.contexts[0].closeCalls, 1);
  assert.equal(engineNoWorklet.getTelemetry().state, 'unsupported');
  assert.equal(errors.length, 2);

  // Case 3: audioWorklet.addModule fails (e.g. network 404 or CSP error).
  const envBadModule = createMockWebAudioEnv({
    addModuleError: new Error('404 Not Found'),
  });
  const engineBadModule = new AudioEngine({
    AudioContextCtor: envBadModule.MockAudioContext,
    AudioWorkletNodeCtor: envBadModule.MockAudioWorkletNode,
    onError: (err) => errors.push(err),
  });
  const res3 = await engineBadModule.start();
  assert.equal(res3.ok, false);
  assert.equal(res3.state, 'error');
  assert.match(res3.error.message, /404 Not Found/);
  assert.equal(envBadModule.events.contexts[0].closeCalls, 1);
  assert.equal(engineBadModule.getTelemetry().state, 'error');
  assert.equal(errors.length, 3);
});

test('createAudioRingSab header layout matches AudioRingBuffer atomics', () => {
  const sab = createAudioRingSab(64, 2);
  const ctrl = new Int32Array(sab, 0, 8);
  const ring = new AudioRingBuffer(sab);

  assert.equal(ring.capacityFrames, 64);
  assert.equal(ring.channels, 2);

  // Trigger 1 overflow (6 dropped frames) and 1 underrun (10 underrun frames) via AudioRingBuffer.
  assert.equal(ring.pushSamples(makeStereoInterleaved(0, 70)), 64);
  const out = new Float32Array(74 * 2);
  assert.equal(ring.pullSamples(out), 64);

  assert.equal(Atomics.load(ctrl, HDR_OVERFLOWS), 1);
  assert.equal(Atomics.load(ctrl, HDR_DROPPED_FRAMES), 6);
  assert.equal(Atomics.load(ctrl, HDR_UNDERRUNS), 1);
  assert.equal(Atomics.load(ctrl, HDR_UNDERRUN_FRAMES), 10);
});

test('audio_worklet_processor.js registers playgta6-audio-processor in AudioWorkletGlobalScope', () => {
  const src = fs.readFileSync(
    path.resolve(__dirname, '../web/workers/audio_worklet_processor.js'),
    'utf8'
  );
  const registered = [];
  class MockBaseProcessor {
    constructor() {
      this.port = { onmessage: null, postMessage() {} };
    }
  }
  const sandbox = {
    AudioWorkletProcessor: MockBaseProcessor,
    registerProcessor(name, ctor) {
      registered.push({ name, ctor });
    },
    Int32Array,
    Float32Array,
    SharedArrayBuffer,
    Atomics,
  };
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;

  vm.runInNewContext(src, sandbox, { filename: 'audio_worklet_processor.js' });
  assert.equal(registered.length, 1);
  assert.equal(registered[0].name, 'playgta6-audio-processor');
  assert.ok(registered[0].ctor.prototype instanceof MockBaseProcessor);
});

test('AudioEngine.resumeOnUserGesture unlocks on keydown and removes both listeners', async () => {
  const { MockAudioContext, MockAudioWorkletNode, events } = createMockWebAudioEnv({
    initialState: 'suspended',
  });
  const engine = new AudioEngine({
    AudioContextCtor: MockAudioContext,
    AudioWorkletNodeCtor: MockAudioWorkletNode,
  });
  await engine.start();

  const target = createMockEventTarget();
  engine.resumeOnUserGesture(target);
  target.dispatch('keydown');
  await Promise.resolve();

  assert.equal(events.contexts[0].resumeCalls, 1);
  assert.equal(engine.state, 'running');
  assert.equal(target.listenerCount('pointerdown'), 0);
  assert.equal(target.listenerCount('keydown'), 0);
  await engine.destroy();
});

test('clampVolume / clampGain guard NaN, ±Infinity, -0, and out-of-range inputs cleanly', () => {
  assert.equal(clampGain, clampVolume);

  // Finite in-range values.
  assert.equal(clampVolume(0), 0);
  assert.equal(clampVolume(0.25), 0.25);
  assert.equal(clampVolume(1), 1);

  // Finite out-of-range values and negative zero.
  assert.equal(clampVolume(-5), 0);
  assert.equal(clampVolume(2.5), 1);
  assert.equal(Object.is(clampVolume(-0), 0), true);

  // Non-finite and non-number inputs resolve to default or sanitized fallback.
  assert.equal(clampVolume(Number.NaN), 1.0);
  assert.equal(clampVolume(Number.POSITIVE_INFINITY), 1.0);
  assert.equal(clampVolume(Number.NEGATIVE_INFINITY), 1.0);
  assert.equal(clampVolume(undefined), 1.0);
  assert.equal(clampVolume(null), 1.0);
  assert.equal(clampVolume('0.5'), 1.0);
  assert.equal(clampVolume({}), 1.0);

  assert.equal(clampVolume(Number.NaN, 0.4), 0.4);
  assert.equal(clampVolume(Number.POSITIVE_INFINITY, 0.4), 0.4);
  assert.equal(clampVolume(Number.NEGATIVE_INFINITY, 0.4), 0.4);
  assert.equal(clampVolume(Number.NaN, -2), 0.0);
  assert.equal(clampVolume(Number.NaN, 5), 1.0);
  assert.equal(clampVolume(Number.NaN, Number.NaN), 1.0);
  assert.equal(clampVolume(Number.NaN, Number.POSITIVE_INFINITY), 1.0);
});

test('AudioEngine guards volume/gain/mute setters against NaN, ±Infinity, and non-finite inputs', async () => {
  const { MockAudioContext, MockAudioWorkletNode, events } = createMockWebAudioEnv({
    initialState: 'running',
  });

  // Constructor guardrails: NaN / Infinity / out-of-range / gain alias / null options.
  assert.equal(new AudioEngine(null).volume, 1.0);
  assert.equal(new AudioEngine({ volume: Number.NaN }).gain, 1.0);
  assert.equal(new AudioEngine({ volume: Number.POSITIVE_INFINITY }).gain, 1.0);
  assert.equal(new AudioEngine({ volume: Number.NEGATIVE_INFINITY }).gain, 1.0);
  assert.equal(new AudioEngine({ volume: -3.5 }).gain, 0.0);
  assert.equal(new AudioEngine({ volume: 9.0 }).gain, 1.0);
  assert.equal(new AudioEngine({ gain: 0.35 }).volume, 0.35);

  const engine = new AudioEngine({
    volume: 0.6,
    muted: false,
    AudioContextCtor: MockAudioContext,
    AudioWorkletNodeCtor: MockAudioWorkletNode,
  });
  await engine.start();

  // Invalid non-finite / non-number inputs preserve active gain and never post to worklet.
  assert.equal(engine.setVolume(Number.NaN), 0.6);
  assert.equal(engine.setVolume(Number.POSITIVE_INFINITY), 0.6);
  assert.equal(engine.setVolume(Number.NEGATIVE_INFINITY), 0.6);
  assert.equal(engine.setGain(undefined), 0.6);
  assert.equal(engine.setGain(null), 0.6);
  assert.equal(engine.setGain('0.2'), 0.6);
  assert.equal(engine.volume, 0.6);
  assert.equal(engine.gain, 0.6);
  assert.equal(events.nodes[0].postedMessages.length, 0);

  // Finite out-of-range values clamp cleanly to [0, 1] and normalize -0.
  assert.equal(engine.setGain(4.2), 1.0);
  assert.equal(engine.setGain(-0.5), 0.0);
  const negZeroRes = engine.setVolume(-0);
  assert.equal(Object.is(negZeroRes, 0), true);
  assert.equal(engine.setGain(0.5), 0.5);

  assert.deepEqual(events.nodes[0].postedMessages, [
    { type: 'setGain', gain: 1.0 },
    { type: 'setGain', gain: 0.0 },
    { type: 'setGain', gain: 0.0 },
    { type: 'setGain', gain: 0.5 },
  ]);

  // Mute guardrail strictly coerces to boolean and updates effectiveGain.
  assert.equal(engine.setMuted(1), true);
  assert.equal(engine.muted, true);
  assert.equal(engine.getStats().muted, true);
  assert.equal(engine.getStats().gain, 0.5);
  assert.equal(engine.getStats().effectiveGain, 0.0);

  assert.equal(engine.setMuted(Number.NaN), false);
  assert.equal(engine.muted, false);
  assert.equal(engine.getStats().muted, false);
  assert.equal(engine.getStats().effectiveGain, 0.5);

  const stats = engine.getStats();
  assert.equal(stats.invalidGainInputs, 6);
  assert.equal(stats.gainClamps, 3);

  await engine.destroy();
});

test('sanitizeResumeError redacts host paths, URLs, credentials, tokens, control chars, and bounds length', () => {
  assert.equal(sanitizeResumeError(null), null);
  assert.equal(sanitizeResumeError(undefined), null);

  const winPathMsg = sanitizeResumeError(
    new Error('Failed at C:\\Users\\alice\\secrets\\audio.js:12\r\nnext line')
  );
  assert.ok(!winPathMsg.includes('alice'));
  assert.ok(!winPathMsg.includes('C:\\Users'));
  assert.ok(!/[\r\n]/.test(winPathMsg));
  assert.ok(winPathMsg.includes('[REDACTED_PATH]'));

  const posixMsg = sanitizeResumeError(
    new Error('Denied /Users/bob/project/worklet.js and file:///home/bob/index.html')
  );
  assert.ok(!posixMsg.includes('bob'));
  assert.ok(posixMsg.includes('[REDACTED_PATH]'));
  assert.ok(posixMsg.includes('[REDACTED_FILE_URL]'));

  const tokenMsg = sanitizeResumeError(
    new Error(
      'Auth Bearer super-secret-jwt failed for https://user:pass@example.com/audio?token=abc123'
    )
  );
  assert.ok(!tokenMsg.includes('super-secret-jwt'));
  assert.ok(!tokenMsg.includes('user:pass'));
  assert.ok(!tokenMsg.includes('abc123'));
  assert.ok(tokenMsg.includes('Bearer [REDACTED]'));
  assert.ok(tokenMsg.includes('https://[REDACTED]@example.com/audio?[REDACTED]'));

  const domLike = sanitizeResumeError({
    name: 'NotAllowedError',
    message: 'User gesture required',
  });
  assert.equal(domLike, 'NotAllowedError: User gesture required');

  // Hostile object with throwing getters never throws.
  const hostile = {
    get name() {
      throw new Error('boom');
    },
    get message() {
      throw new Error('boom');
    },
    toString() {
      throw new Error('boom');
    },
  };
  assert.equal(sanitizeResumeError(hostile), 'AudioContext resume failed');

  // Empty / whitespace string falls back safely.
  assert.equal(sanitizeResumeError('   \n\t  '), 'AudioContext resume failed');

  // Long message is cleanly truncated to MAX_ERROR_MESSAGE_LENGTH.
  const longMsg = sanitizeResumeError('X'.repeat(MAX_ERROR_MESSAGE_LENGTH + 100));
  assert.equal(longMsg.length, MAX_ERROR_MESSAGE_LENGTH);
  assert.ok(longMsg.endsWith('...'));
});

test('AudioEngine tracks resumeAttempts, resumeSuccesses, resumeFailures, sanitized lastResumeError, and retry recovery', async () => {
  let remainingFailures = 2;
  const errors = [];

  class FlakyResumeAudioContext {
    constructor(opts = {}) {
      this.sampleRate = opts.sampleRate ?? 48000;
      this.state = 'suspended';
      this.destination = { kind: 'mock-destination' };
      this.resumeCalls = 0;
      this._listeners = new Map();
      this.audioWorklet = { addModule: async () => {} };
    }
    addEventListener(type, fn) {
      if (!this._listeners.has(type)) this._listeners.set(type, new Set());
      this._listeners.get(type).add(fn);
    }
    removeEventListener(type, fn) {
      const set = this._listeners.get(type);
      if (set) set.delete(fn);
    }
    _emit(type) {
      const set = this._listeners.get(type);
      if (!set) return;
      for (const fn of Array.from(set)) fn({ target: this });
    }
    async resume() {
      this.resumeCalls += 1;
      if (remainingFailures === 2) {
        remainingFailures -= 1;
        throw new Error('NotAllowedError at C:\\Users\\dev\\autoplay.js?token=xyz');
      }
      if (remainingFailures === 1) {
        remainingFailures -= 1;
        // Silent refusal: resolves without throwing, but remains suspended.
        return;
      }
      this.state = 'running';
      this._emit('statechange');
    }
    async close() {
      this.state = 'closed';
      this._emit('statechange');
    }
  }

  const { MockAudioWorkletNode } = createMockWebAudioEnv();
  const engine = new AudioEngine({
    AudioContextCtor: FlakyResumeAudioContext,
    AudioWorkletNodeCtor: MockAudioWorkletNode,
    onError: (e) => errors.push(e),
  });

  // Before start(), resume() returns false and does not count as an AudioContext attempt.
  assert.equal(await engine.resume(), false);
  assert.equal(engine.getStats().resumeAttempts, 0);
  assert.equal(engine.getStats().outputState, 'idle');

  await engine.start();
  assert.equal(engine.getStats().outputState, 'suspended');
  assert.equal(engine.getStats().stateTransitions, 1);

  // Attempt 1: throws error with host path & query string -> sanitized lastResumeError.
  assert.equal(await engine.resume(), false);
  let stats = engine.getStats();
  assert.equal(stats.resumeAttempts, 1);
  assert.equal(stats.resumeSuccesses, 0);
  assert.equal(stats.resumeFailures, 1);
  assert.equal(stats.stateTransitions, 1);
  assert.equal(stats.outputState, 'suspended');
  assert.ok(typeof stats.lastResumeError === 'string' && stats.lastResumeError.length > 0);
  assert.ok(!stats.lastResumeError.includes('dev'));
  assert.ok(!stats.lastResumeError.includes('C:\\Users'));
  assert.equal(errors.length, 1);

  // Attempt 2: resolves without transitioning out of suspended -> counted as failure.
  assert.equal(await engine.resume(), false);
  stats = engine.getStats();
  assert.equal(stats.resumeAttempts, 2);
  assert.equal(stats.resumeSuccesses, 0);
  assert.equal(stats.resumeFailures, 2);
  assert.match(stats.lastResumeError, /remained in state "suspended"/);

  // Attempt 3: succeeds -> clears lastResumeError, updates stateTransitions once (no double count).
  assert.equal(await engine.resume(), true);
  stats = engine.getStats();
  assert.equal(stats.resumeAttempts, 3);
  assert.equal(stats.resumeSuccesses, 1);
  assert.equal(stats.resumeFailures, 2);
  assert.equal(stats.lastResumeError, null);
  assert.equal(stats.outputState, 'running');
  assert.equal(stats.stateTransitions, 2);

  // Already running -> returns true without incrementing resumeAttempts.
  assert.equal(await engine.resume(), true);
  assert.equal(engine.getStats().resumeAttempts, 3);
  assert.equal(engine.getStats().resumeSuccesses, 1);

  await engine.destroy();
  // Destroyed -> returns false without incrementing resumeAttempts.
  assert.equal(await engine.resume(), false);
  assert.equal(engine.getStats().resumeAttempts, 3);
  assert.equal(engine.getStats().outputState, 'closed');
  assert.equal(engine.getStats().stateTransitions, 3);
});

test('AudioEngine.resume supports explicit maxRetries and coalesces concurrent calls', async () => {
  let failTimes = 2;

  class RetryingContext {
    constructor() {
      this.sampleRate = 48000;
      this.state = 'suspended';
      this.destination = { kind: 'mock-destination' };
      this.resumeCalls = 0;
      this.audioWorklet = { addModule: async () => {} };
    }
    async resume() {
      this.resumeCalls += 1;
      await Promise.resolve();
      if (failTimes > 0) {
        failTimes -= 1;
        throw new Error('transient resume failure');
      }
      this.state = 'running';
    }
    async close() {
      this.state = 'closed';
    }
  }

  const { MockAudioWorkletNode } = createMockWebAudioEnv();
  const engine = new AudioEngine({
    AudioContextCtor: RetryingContext,
    AudioWorkletNodeCtor: MockAudioWorkletNode,
  });
  await engine.start();

  // Concurrent calls coalesce onto the single retry sequence (2 failures + 1 success = 3 attempts).
  const [r1, r2] = await Promise.all([
    engine.resume({ maxRetries: 3, retryDelayMs: 0 }),
    engine.resume({ maxRetries: 3, retryDelayMs: 0 }),
  ]);

  assert.equal(r1, true);
  assert.equal(r2, true);
  assert.equal(engine.context.resumeCalls, 3);

  const stats = engine.getStats();
  assert.equal(stats.resumeAttempts, 3);
  assert.equal(stats.resumeFailures, 2);
  assert.equal(stats.resumeSuccesses, 1);
  assert.equal(stats.lastResumeError, null);
  assert.equal(stats.outputState, 'running');
  assert.equal(stats.stateTransitions, 2);

  await engine.destroy();
});

test('AudioEngine.resumeOnUserGesture auto-rearms on failed resume and respects bounded maxRetries', async () => {
  // Part 1: Fails on first gesture, auto-rearms, and succeeds on second gesture.
  let failOnce = true;
  class RecoveringContext {
    constructor() {
      this.sampleRate = 48000;
      this.state = 'suspended';
      this.destination = { kind: 'mock-destination' };
      this.resumeCalls = 0;
      this.audioWorklet = { addModule: async () => {} };
    }
    async resume() {
      this.resumeCalls += 1;
      if (failOnce) {
        failOnce = false;
        throw new Error('autoplay blocked');
      }
      this.state = 'running';
    }
    async close() {
      this.state = 'closed';
    }
  }

  const { MockAudioWorkletNode } = createMockWebAudioEnv();
  const engine1 = new AudioEngine({
    AudioContextCtor: RecoveringContext,
    AudioWorkletNodeCtor: MockAudioWorkletNode,
  });
  await engine1.start();

  const target1 = createMockEventTarget();
  engine1.resumeOnUserGesture(target1);
  assert.equal(target1.listenerCount('pointerdown'), 1);
  assert.equal(target1.listenerCount('keydown'), 1);

  // First gesture fails -> microtasks settle -> listeners automatically re-arm.
  target1.dispatch('pointerdown');
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(engine1.getStats().resumeAttempts, 1);
  assert.equal(engine1.getStats().resumeFailures, 1);
  assert.equal(engine1.getStats().resumeSuccesses, 0);
  assert.equal(engine1.getStats().lastResumeError, 'autoplay blocked');
  assert.equal(target1.listenerCount('pointerdown'), 1);
  assert.equal(target1.listenerCount('keydown'), 1);

  // Second gesture succeeds -> listeners permanently cleaned up.
  target1.dispatch('keydown');
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(engine1.getStats().resumeAttempts, 2);
  assert.equal(engine1.getStats().resumeFailures, 1);
  assert.equal(engine1.getStats().resumeSuccesses, 1);
  assert.equal(engine1.getStats().lastResumeError, null);
  assert.equal(engine1.getStats().outputState, 'running');
  assert.equal(target1.listenerCount('pointerdown'), 0);
  assert.equal(target1.listenerCount('keydown'), 0);
  await engine1.destroy();

  // Part 2: Persistent failures stop re-arming once maxRetries is exhausted.
  const failingEnv = createMockWebAudioEnv({
    initialState: 'suspended',
    resumeError: new Error('persistent autoplay denial'),
  });
  const engine2 = new AudioEngine({
    AudioContextCtor: failingEnv.MockAudioContext,
    AudioWorkletNodeCtor: failingEnv.MockAudioWorkletNode,
  });
  await engine2.start();

  const target2 = createMockEventTarget();
  assert.equal(DEFAULT_MAX_RESUME_RETRIES, 5);
  engine2.resumeOnUserGesture(target2, { maxRetries: 2 });

  // 1 initial attempt + 2 retries = 3 total attempts, then disarmed.
  for (let i = 0; i < 3; i++) {
    assert.equal(target2.listenerCount('pointerdown'), 1);
    target2.dispatch('pointerdown');
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  assert.equal(target2.listenerCount('pointerdown'), 0);
  assert.equal(target2.listenerCount('keydown'), 0);
  assert.equal(engine2.getStats().resumeAttempts, 3);
  assert.equal(engine2.getStats().resumeFailures, 3);
  assert.equal(engine2.getStats().resumeSuccesses, 0);

  // Extra gesture after disarming does not invoke resume again.
  target2.dispatch('pointerdown');
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(engine2.getStats().resumeAttempts, 3);
  await engine2.destroy();
});

test('AudioEngine counts AudioContext stateTransitions accurately and returns sanitized defensive getStats() snapshots', async () => {
  const { MockAudioContext, MockAudioWorkletNode, events } = createMockWebAudioEnv({
    initialState: 'suspended',
  });

  const engine = new AudioEngine({
    sampleRate: 48000,
    volume: 0.8,
    muted: false,
    AudioContextCtor: MockAudioContext,
    AudioWorkletNodeCtor: MockAudioWorkletNode,
  });

  // Initial snapshot before start.
  const initialStats = engine.getStats();
  assert.equal(initialStats.outputState, 'idle');
  assert.equal(initialStats.state, 'idle');
  assert.equal(initialStats.gain, 0.8);
  assert.equal(initialStats.muted, false);
  assert.equal(initialStats.effectiveGain, 0.8);
  assert.equal(initialStats.resumeAttempts, 0);
  assert.equal(initialStats.resumeSuccesses, 0);
  assert.equal(initialStats.resumeFailures, 0);
  assert.equal(initialStats.stateTransitions, 0);
  assert.equal(initialStats.lastResumeError, null);
  assert.equal(initialStats.error, null);

  // Mutating the returned snapshot cannot corrupt engine internals.
  initialStats.gain = Number.NaN;
  initialStats.resumeAttempts = 999;
  initialStats.outputState = 'hacked';
  assert.equal(engine.getStats().gain, 0.8);
  assert.equal(engine.getStats().resumeAttempts, 0);
  assert.equal(engine.getStats().outputState, 'idle');

  // 1: idle -> suspended
  await engine.start();
  const ctx = events.contexts[0];
  assert.equal(engine.getStats().stateTransitions, 1);

  // 2: suspended -> running
  await engine.resume();
  assert.equal(engine.getStats().stateTransitions, 2);

  // Duplicate statechange with same state must not increment stateTransitions.
  ctx._emit('statechange');
  assert.equal(engine.getStats().stateTransitions, 2);

  // 3: running -> interrupted (e.g. OS audio interruption)
  ctx.state = 'interrupted';
  ctx._emit('statechange');
  assert.equal(engine.getStats().outputState, 'interrupted');
  assert.equal(engine.getStats().stateTransitions, 3);

  // 4: interrupted -> running via resume
  await engine.resume();
  assert.equal(engine.getStats().outputState, 'running');
  assert.equal(engine.getStats().stateTransitions, 4);
  assert.equal(engine.getStats().resumeAttempts, 2);
  assert.equal(engine.getStats().resumeSuccesses, 2);

  // Even if context exposes corrupted sampleRate or unknown state, getStats() stays finite & sanitized.
  ctx.sampleRate = Number.NaN;
  ctx.state = 'bogus-state-value';
  const corruptedStats = engine.getStats();
  assert.equal(corruptedStats.sampleRate, 48000);
  assert.equal(corruptedStats.outputState, 'unknown');
  for (const key of [
    'sampleRate',
    'channels',
    'capacityFrames',
    'bufferedFrames',
    'underruns',
    'underrunFrames',
    'overflows',
    'droppedFrames',
    'gain',
    'volume',
    'effectiveGain',
    'gainClamps',
    'invalidGainInputs',
    'resumeAttempts',
    'resumeSuccesses',
    'resumeFailures',
    'stateTransitions',
  ]) {
    assert.equal(
      Number.isFinite(corruptedStats[key]),
      true,
      `Expected finite numeric stat for ${key}, got ${corruptedStats[key]}`
    );
    assert.ok(corruptedStats[key] >= 0, `Expected non-negative stat for ${key}`);
  }

  // 5: -> closed
  await engine.destroy();
  assert.equal(engine.getStats().outputState, 'closed');
  assert.equal(engine.getStats().stateTransitions, 5);

  // Idempotent destroy does not increment stateTransitions again.
  await engine.destroy();
  assert.equal(engine.getStats().stateTransitions, 5);
});

test('W214: suspended-context autoplay guard and headless gesture resume never throw', async () => {
  // 1. Headless environment (no window / null target) and sync-throwing resume() never throw.
  let throwSync = true;
  class AutoplayBlockedContext {
    constructor(opts = {}) {
      this.sampleRate = opts.sampleRate ?? 48000;
      this.state = 'suspended';
      this.destination = { kind: 'mock-destination' };
      this.resumeCalls = 0;
      this.suspendCalls = 0;
      this.closeCalls = 0;
      this.audioWorklet = { addModule: async () => {} };
    }
    resume() {
      this.resumeCalls += 1;
      if (throwSync) {
        throw new Error('NotAllowedError: play() failed because the user did not interact');
      }
      this.state = 'running';
      return Promise.resolve();
    }
    suspend() {
      this.suspendCalls += 1;
      this.state = 'suspended';
      return Promise.resolve();
    }
    close() {
      this.closeCalls += 1;
      this.state = 'closed';
      return Promise.resolve();
    }
  }

  const { MockAudioWorkletNode } = createMockWebAudioEnv();
  const engine = new AudioEngine({
    sampleRate: 44100,
    AudioContextCtor: AutoplayBlockedContext,
    AudioWorkletNodeCtor: MockAudioWorkletNode,
  });

  // Calling resumeOnUserGesture without a browser window or with a hostile target never throws.
  const noopCleanup = engine.resumeOnUserGesture(null);
  assert.equal(typeof noopCleanup, 'function');
  noopCleanup();

  const hostileTarget = {
    addEventListener() {
      throw new Error('restricted host');
    },
    removeEventListener() {
      throw new Error('restricted host');
    },
  };
  const hostileCleanup = engine.resumeOnUserGesture(hostileTarget);
  assert.equal(typeof hostileCleanup, 'function');
  hostileCleanup();

  // Arm gesture before start(); firing gesture before start() safely re-arms without throwing.
  const target = createMockEventTarget();
  engine.resumeOnUserGesture(target);
  for (const evt of DEFAULT_GESTURE_EVENTS) {
    assert.equal(target.listenerCount(evt), 1);
  }
  target.dispatch('click');
  assert.equal(target.listenerCount('pointerdown'), 1);

  const startRes = await engine.start();
  assert.equal(startRes.ok, true);
  assert.equal(engine.contextState, 'suspended');
  assert.equal(engine.getAudioStats().contextState, 'suspended');
  assert.equal(engine.getAudioStats().sampleRate, 44100);

  // Sync-throwing resume() is caught cleanly and listeners re-arm for next gesture.
  target.dispatch('touchstart');
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(engine.context.resumeCalls, 1);
  assert.equal(engine.getAudioStats().contextState, 'suspended');
  assert.equal(engine.getAudioStats().resumeFailures, 1);
  assert.match(engine.getAudioStats().lastResumeError, /NotAllowedError/);
  assert.equal(target.listenerCount('click'), 1);

  // Next gesture succeeds -> context becomes running and all gesture listeners are removed.
  throwSync = false;
  target.dispatch('click');
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(engine.context.resumeCalls, 2);
  assert.equal(engine.contextState, 'running');
  assert.equal(engine.getAudioStats().contextState, 'running');
  assert.equal(engine.getAudioStats().resumeSuccesses, 1);
  assert.equal(engine.getAudioStats().lastResumeError, null);
  for (const evt of DEFAULT_GESTURE_EVENTS) {
    assert.equal(target.listenerCount(evt), 0);
  }

  // suspend() safely transitions running -> suspended; externally closed context does not throw on resume().
  assert.equal(await engine.suspend(), true);
  assert.equal(engine.getAudioStats().contextState, 'suspended');

  engine.context.state = 'closed';
  const callsBeforeClosedResume = engine.context.resumeCalls;
  assert.equal(await engine.resume(), false);
  assert.equal(engine.context.resumeCalls, callsBeforeClosedResume);

  await engine.close();

  // 2. autoResumeOnGesture option arms listeners automatically on suspended start().
  const autoEnv = createMockWebAudioEnv({ initialState: 'suspended' });
  const autoTarget = createMockEventTarget();
  const autoEngine = new AudioEngine({
    AudioContextCtor: autoEnv.MockAudioContext,
    AudioWorkletNodeCtor: autoEnv.MockAudioWorkletNode,
    autoResumeOnGesture: true,
    gestureTarget: autoTarget,
  });
  await autoEngine.start();
  assert.equal(autoTarget.listenerCount('pointerdown'), 1);
  assert.equal(autoTarget.listenerCount('touchstart'), 1);

  autoTarget.dispatch('touchstart');
  await Promise.resolve();
  assert.equal(autoEngine.getAudioStats().contextState, 'running');
  assert.equal(autoTarget.listenerCount('pointerdown'), 0);
  assert.equal(autoTarget.listenerCount('touchstart'), 0);
  await autoEngine.close();
});

test('W214: bounded PCM submitPcm/pullPcm tracks submittedFrames, droppedFrames, underruns, and overruns in getAudioStats()', async () => {
  const { MockAudioContext, MockAudioWorkletNode, events } = createMockWebAudioEnv({
    initialState: 'running',
    attachRealProcessor: true,
  });

  const engine = new AudioEngine({
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 256,
    declickFrames: 0,
    AudioContextCtor: MockAudioContext,
    AudioWorkletNodeCtor: MockAudioWorkletNode,
  });
  await engine.start();

  // Invalid inputs never throw and never corrupt counters.
  assert.equal(engine.submitPcm(null).ok, false);
  assert.equal(engine.submitPcm(undefined).ok, false);
  assert.equal(engine.submitPcm('bad').ok, false);
  assert.equal(engine.pullPcm('bad').ok, false);

  // Submit 200 valid frames (fits cleanly in 256-frame ring).
  const chunk1 = makeStereoInterleaved(0, 200);
  const r1 = engine.submitPcm(chunk1);
  assert.equal(r1.ok, true);
  assert.equal(r1.acceptedFrames, 200);
  assert.equal(r1.droppedFrames, 0);
  assert.equal(r1.overrun, false);
  assert.equal(r1.bufferedFrames, 200);
  assert.equal(r1.freeFrames, 56);

  // Submit 100 more frames into 56 free slots -> 56 accepted, 44 dropped, 1 overrun.
  const chunk2 = makeStereoInterleaved(200, 100);
  const r2 = engine.submitPcm(chunk2);
  assert.equal(r2.ok, true);
  assert.equal(r2.acceptedFrames, 56);
  assert.equal(r2.droppedFrames, 44);
  assert.equal(r2.overrun, true);
  assert.equal(r2.bufferedFrames, 256);
  assert.equal(r2.freeFrames, 0);

  let stats = engine.getAudioStats();
  assert.equal(stats.submittedFrames, 256);
  assert.equal(stats.droppedFrames, 44);
  assert.equal(stats.overruns, 1);
  assert.equal(stats.underruns, 0);
  assert.equal(stats.contextState, 'running');
  assert.equal(stats.sampleRate, 48000);
  assert.equal(stats.muted, false);
  assert.equal(stats.bufferedFrames, 256);

  // Drive the attached real AudioWorkletProcessor: 2 full 128-frame quanta consume exact samples,
  // 3rd quantum triggers an underrun.
  const proc = events.nodes[0]._processor;
  const outQ = makePlanarOutput(2, 128, 99.0);
  assert.equal(proc.process([], outQ), true);
  assert.equal(outQ[0][0][0], chunk1[0]);
  assert.equal(outQ[0][1][0], chunk1[1]);
  assert.equal(proc.process([], outQ), true);
  assert.equal(outQ[0][0][127], chunk2[55 * 2]);
  assert.equal(outQ[0][1][127], chunk2[55 * 2 + 1]);
  assert.equal(proc.process([], outQ), true); // full underrun (128 frames)
  assert.ok(outQ[0][0].every((v) => v === 0));

  stats = engine.getAudioStats();
  assert.equal(stats.underruns, 1);
  assert.equal(stats.underrunFrames, 128);
  assert.equal(stats.bufferedFrames, 0);

  // Submit PCM containing NaN / ±Infinity -> sanitized to 0.0; partial pullPcm zero-fills shortfall and records underrun.
  const dirty = new Float32Array([0.5, Number.NaN, Number.POSITIVE_INFINITY, -0.25, 0.75, Number.NEGATIVE_INFINITY]);
  const rDirty = engine.submitPcm(dirty);
  assert.equal(rDirty.acceptedFrames, 3);
  assert.equal(engine.getAudioStats().sanitizedSamples, 3);

  const pulled = new Float32Array(10 * 2);
  pulled.fill(42.0);
  const pullRes = engine.pullPcm(pulled);
  assert.equal(pullRes.ok, true);
  assert.equal(pullRes.framesRead, 3);
  assert.equal(pullRes.underrun, true);
  assert.equal(pullRes.underrunFrames, 7);
  assert.equal(pulled[0], 0.5);
  assert.equal(pulled[1], 0.0);
  assert.equal(pulled[2], 0.0);
  assert.equal(pulled[3], -0.25);
  assert.equal(pulled[4], 0.75);
  assert.equal(pulled[5], 0.0);
  for (let i = 6; i < pulled.length; i++) {
    assert.equal(pulled[i], 0.0, `underrun tail sample ${i} must be zero-filled`);
  }

  stats = engine.getAudioStats();
  assert.equal(stats.submittedFrames, 259);
  assert.equal(stats.underruns, 2);
  assert.equal(stats.underrunFrames, 128 + 7);
  assert.equal(stats.overruns, 1);
  assert.equal(stats.droppedFrames, 44);

  // Defensive snapshot: mutating returned object never mutates engine internals.
  stats.submittedFrames = -999;
  stats.overruns = 999;
  stats.muted = true;
  const freshStats = engine.getAudioStats();
  assert.equal(freshStats.submittedFrames, 259);
  assert.equal(freshStats.overruns, 1);
  assert.equal(freshStats.muted, false);

  // drop-oldest overflow policy keeps newest samples and counts overrun + droppedFrames.
  const newestChunk = makeStereoInterleaved(1000, 300);
  const dropOldRes = engine.submitPcm(newestChunk, { overflowPolicy: 'drop-oldest' });
  assert.equal(dropOldRes.ok, true);
  assert.equal(dropOldRes.acceptedFrames, 256);
  assert.equal(dropOldRes.droppedFrames, 44);
  assert.equal(dropOldRes.overrun, true);
  assert.equal(engine.getAudioStats().overruns, 2);
  assert.equal(engine.getAudioStats().droppedFrames, 88);

  const checkNewest = new Float32Array(4 * 2);
  engine.pullPcm(checkNewest);
  assert.equal(checkNewest[0], newestChunk[44 * 2]);
  assert.equal(checkNewest[1], newestChunk[44 * 2 + 1]);

  await engine.close();
});

test('W214: mute/unmute controls silence output, update worklet, and reflect in getAudioStats()', async () => {
  const { MockAudioContext, MockAudioWorkletNode, events } = createMockWebAudioEnv({
    initialState: 'running',
  });

  const engine = new AudioEngine({
    capacityFrames: 128,
    channels: 2,
    volume: 0.5,
    muted: false,
    AudioContextCtor: MockAudioContext,
    AudioWorkletNodeCtor: MockAudioWorkletNode,
  });
  await engine.start();

  assert.equal(engine.getAudioStats().muted, false);
  assert.equal(engine.getAudioStats().effectiveGain, 0.5);

  // Mute via mute() helper: silences pulled PCM while still consuming frames.
  assert.equal(engine.mute(), true);
  assert.equal(engine.muted, true);
  assert.equal(engine.getAudioStats().muted, true);
  assert.equal(engine.getAudioStats().effectiveGain, 0.0);

  const ones = new Float32Array(16 * 2).fill(1.0);
  engine.submitPcm(ones);
  const mutedOut = new Float32Array(16 * 2).fill(7.0);
  const mutedPull = engine.pullPcm(mutedOut);
  assert.equal(mutedPull.framesRead, 16);
  assert.equal(mutedPull.underrun, false);
  assert.ok(mutedOut.every((v) => v === 0.0));
  assert.equal(engine.availableFrames(), 0);

  // Unmute via unmute() helper: restores volume scaling on pulled PCM.
  assert.equal(engine.unmute(), false);
  assert.equal(engine.muted, false);
  assert.equal(engine.getAudioStats().muted, false);
  assert.equal(engine.getAudioStats().effectiveGain, 0.5);

  engine.submitPcm(ones);
  const unmutedOut = new Float32Array(16 * 2);
  engine.pullPcm(unmutedOut);
  assert.ok(unmutedOut.every((v) => v === 0.5));

  // toggleMute() flips state cleanly and posts expected messages to worklet.
  assert.equal(engine.toggleMute(), true);
  assert.equal(engine.toggleMute(), false);
  assert.deepEqual(events.nodes[0].postedMessages, [
    { type: 'setMuted', muted: true },
    { type: 'setMuted', muted: false },
    { type: 'setMuted', muted: true },
    { type: 'setMuted', muted: false },
  ]);

  await engine.close();
});

test('W214: reset() and close() clean up ring state, telemetry counters, gesture listeners, and reject post-close submissions', async () => {
  const { MockAudioContext, MockAudioWorkletNode, events } = createMockWebAudioEnv({
    initialState: 'suspended',
  });

  const engine = new AudioEngine({
    capacityFrames: 64,
    channels: 2,
    volume: 0.4,
    AudioContextCtor: MockAudioContext,
    AudioWorkletNodeCtor: MockAudioWorkletNode,
  });
  await engine.start();

  const target = createMockEventTarget();
  engine.resumeOnUserGesture(target);
  assert.equal(target.listenerCount('pointerdown'), 1);

  // Populate ring, force 1 overrun and 1 underrun.
  engine.submitPcm(makeStereoInterleaved(0, 80)); // 64 written, 16 dropped, 1 overrun
  engine.consumePcm(70); // 64 read, 6 underrunFrames, 1 underrun
  engine.submitPcm(makeStereoInterleaved(0, 20)); // leave 20 frames queued

  let stats = engine.getAudioStats();
  assert.equal(stats.submittedFrames, 84);
  assert.equal(stats.droppedFrames, 16);
  assert.equal(stats.overruns, 1);
  assert.equal(stats.underruns, 1);
  assert.equal(stats.bufferedFrames, 20);

  // reset({ preserveStats: true }) flushes queued frames while retaining counters.
  const flushedStats = engine.reset({ preserveStats: true });
  assert.equal(flushedStats.bufferedFrames, 0);
  assert.equal(flushedStats.freeFrames, 64);
  assert.equal(flushedStats.submittedFrames, 84);
  assert.equal(flushedStats.overruns, 1);
  assert.equal(flushedStats.underruns, 1);

  // Full reset() clears cursors and counters while keeping AudioContext open and usable.
  engine.submitPcm(makeStereoInterleaved(0, 12));
  const resetStats = engine.reset({ resetGain: true, clearGestureListeners: true });
  assert.equal(resetStats.submittedFrames, 0);
  assert.equal(resetStats.droppedFrames, 0);
  assert.equal(resetStats.underruns, 0);
  assert.equal(resetStats.overruns, 0);
  assert.equal(resetStats.bufferedFrames, 0);
  assert.equal(resetStats.volume, 1.0);
  assert.equal(resetStats.muted, false);
  assert.equal(resetStats.contextState, 'suspended');
  assert.equal(target.listenerCount('pointerdown'), 0);

  // Engine still accepts new PCM after reset().
  const postResetSubmit = engine.submitPcm(makeStereoInterleaved(0, 16));
  assert.equal(postResetSubmit.ok, true);
  assert.equal(postResetSubmit.acceptedFrames, 16);
  assert.equal(engine.getAudioStats().submittedFrames, 16);

  // Re-arm gesture listeners, then close() -> cleans up listeners, node, context, and is idempotent.
  engine.resumeOnUserGesture(target);
  assert.equal(target.listenerCount('pointerdown'), 1);

  await engine.close();
  assert.equal(engine.contextState, 'closed');
  assert.equal(engine.getAudioStats().contextState, 'closed');
  assert.equal(target.listenerCount('pointerdown'), 0);
  assert.equal(target.listenerCount('keydown'), 0);
  assert.equal(events.nodes[0].disconnected, true);
  assert.equal(events.nodes[0].portClosed, true);
  assert.equal(events.contexts[0].closeCalls, 1);

  // Post-close submitPcm is safely rejected without throwing or mutating submittedFrames/overruns.
  const postCloseSubmit = engine.submitPcm(makeStereoInterleaved(0, 128));
  assert.equal(postCloseSubmit.ok, false);
  assert.equal(postCloseSubmit.reason, 'closed');
  assert.equal(engine.getAudioStats().submittedFrames, 16);
  assert.equal(engine.getAudioStats().overruns, 0);

  // Second close() is an idempotent no-op.
  await engine.close();
  assert.equal(events.contexts[0].closeCalls, 1);
});

function createGainNodeMockWebAudioEnv(options = {}) {
  const base = createMockWebAudioEnv(options);
  const gainNodes = [];

  class MockGainNode {
    constructor(ctx) {
      this.context = ctx;
      this.connectedTo = null;
      this.disconnected = false;
      this.ops = [];
      const self = this;
      this.gain = {
        value: 1.0,
        cancelScheduledValues(time) {
          self.ops.push({ op: 'cancel', time });
        },
        setValueAtTime(value, time) {
          this.value = value;
          self.ops.push({ op: 'set', value, time });
        },
        linearRampToValueAtTime(value, endTime) {
          this.value = value;
          self.ops.push({ op: 'linearRamp', value, endTime });
        },
      };
      gainNodes.push(this);
    }

    connect(dest) {
      this.connectedTo = dest;
      return dest;
    }

    disconnect() {
      this.disconnected = true;
      this.connectedTo = null;
    }
  }

  class GainCapableAudioContext extends base.MockAudioContext {
    constructor(ctxOpts = {}) {
      super(ctxOpts);
      this.currentTime = options.initialCurrentTime ?? 1.5;
    }

    createGain() {
      return new MockGainNode(this);
    }
  }

  return {
    MockAudioContext: GainCapableAudioContext,
    MockAudioWorkletNode: base.MockAudioWorkletNode,
    events: { ...base.events, gainNodes },
  };
}

test('W238: attachGuestPcmRing / detachGuestPcmRing track state locally and forward cleanly to worker and bridge', async () => {
  // 1. Worker-style postMessage forwarding with #1027 Playgta6AudioRing layout.
  const workerMessages = [];
  const mockWorker = {
    postMessage(msg) {
      workerMessages.push(msg);
    },
  };

  const guestMem = new SharedArrayBuffer(131072 * 4 + 64);
  const engine = new AudioEngine({
    sampleRate: 48000,
    channels: 2,
    audioWorker: mockWorker,
  });

  assert.equal(engine.guestRingAttached, false);
  assert.equal(engine.guestPcmRing, null);
  assert.equal(engine.getStats().guestRingAttached, false);

  const attachRes = engine.attachGuestPcmRing({
    buffer: guestMem,
    base: 64,
    capacity: 131072,
    channels: 2,
    sample_rate: 48000,
    read_index_ptr: 0,
    write_index_ptr: 4,
  });

  assert.equal(attachRes.ok, true);
  assert.equal(attachRes.attached, true);
  assert.equal(attachRes.forwarded, true);
  assert.equal(attachRes.reason, null);
  assert.equal(attachRes.ring.base, 64);
  assert.equal(attachRes.ring.capacity, 131072);
  assert.equal(attachRes.ring.capacityFrames, 65536);
  assert.equal(attachRes.ring.channels, 2);
  assert.equal(attachRes.ring.sampleRate, 48000);
  assert.equal(attachRes.ring.readIndexPtr, 0);
  assert.equal(attachRes.ring.writeIndexPtr, 4);

  assert.equal(engine.guestRingAttached, true);
  assert.equal(engine.getStats().guestRingAttached, true);
  assert.equal(workerMessages.length, 1);
  assert.equal(workerMessages[0].type, 'attachGuestPcmRing');
  assert.equal(workerMessages[0].attachGuestPcmRing, true);
  assert.equal(workerMessages[0].ring.buffer, guestMem);
  assert.equal(workerMessages[0].ring.capacity, 131072);

  // Defensive clone on getter: mutating returned ring cannot mutate internal state.
  const ringSnap = engine.guestPcmRing;
  ringSnap.capacity = -999;
  assert.equal(engine.guestPcmRing.capacity, 131072);

  // Detach forwards cleanly and is idempotent.
  const detachRes = engine.detachGuestPcmRing();
  assert.equal(detachRes.ok, true);
  assert.equal(detachRes.attached, false);
  assert.equal(detachRes.wasAttached, true);
  assert.equal(detachRes.forwarded, true);
  assert.equal(engine.guestRingAttached, false);
  assert.equal(engine.guestPcmRing, null);
  assert.equal(engine.getStats().guestRingAttached, false);
  assert.equal(workerMessages.length, 2);
  assert.equal(workerMessages[1].type, 'detachGuestPcmRing');
  assert.equal(workerMessages[1].detachGuestPcmRing, true);

  const detachAgain = engine.detachGuestPcmRing();
  assert.equal(detachAgain.ok, true);
  assert.equal(detachAgain.wasAttached, false);

  // 2. Direct bridge method forwarding + Float32Array samples / Uint32Array index views.
  const bridgeCalls = [];
  const bridge = {
    attachGuestPcmRing(desc) {
      bridgeCalls.push({ method: 'attach', desc });
    },
    detachGuestPcmRing() {
      bridgeCalls.push({ method: 'detach' });
    },
  };
  engine.setAudioWorker(bridge);

  const sampleView = new Float32Array(new SharedArrayBuffer(2048 * 4));
  const readIdx = new Uint32Array(new SharedArrayBuffer(4));
  const writeIdx = new Uint32Array(new SharedArrayBuffer(4));
  const bridgeAttach = engine.attachGuestPcmRing({
    samples: sampleView,
    readIndex: readIdx,
    writeIndex: writeIdx,
    channels: 2,
    sampleRate: 48000,
  });
  assert.equal(bridgeAttach.ok, true);
  assert.equal(bridgeAttach.forwarded, true);
  assert.equal(bridgeAttach.ring.capacity, 2048);
  assert.equal(bridgeAttach.ring.capacityFrames, 1024);
  assert.equal(bridgeCalls.length, 1);
  assert.equal(bridgeCalls[0].method, 'attach');
  assert.equal(bridgeCalls[0].desc.samples, sampleView);

  // destroy() automatically detaches active guest ring on the bridge.
  await engine.destroy();
  assert.equal(engine.guestRingAttached, false);
  assert.equal(bridgeCalls.length, 2);
  assert.equal(bridgeCalls[1].method, 'detach');

  // Post-destroy attach is rejected cleanly without throwing.
  const postDestroyAttach = engine.attachGuestPcmRing({
    samples: sampleView,
    capacity: 2048,
  });
  assert.equal(postDestroyAttach.ok, false);
  assert.equal(postDestroyAttach.reason, 'closed');

  // 3. Invalid descriptors & no-worker standalone mode never throw.
  const errors = [];
  const standalone = new AudioEngine({ onError: (e) => errors.push(e) });
  assert.equal(standalone.attachGuestPcmRing(null).reason, 'invalid-descriptor');
  assert.equal(standalone.attachGuestPcmRing({}).reason, 'missing-buffer');
  assert.equal(
    standalone.attachGuestPcmRing({ base: 0, capacity: -10 }).reason,
    'invalid-capacity'
  );
  assert.equal(
    standalone.attachGuestPcmRing({ base: 0, capacity: Number.NaN }).reason,
    'invalid-capacity'
  );
  assert.equal(
    standalone.attachGuestPcmRing({ base: 0, capacity: 1024, channels: 0 }).reason,
    'invalid-channels'
  );
  assert.equal(
    standalone.attachGuestPcmRing({ base: 0, capacity: 1024, sampleRate: -48000 }).reason,
    'invalid-sample-rate'
  );
  assert.equal(
    standalone.attachGuestPcmRing({ base: 0, capacity: 1024, read_index_ptr: -4 }).reason,
    'invalid-index-ptr'
  );
  assert.equal(standalone.guestRingAttached, false);

  // Valid attach without worker tracks state locally with forwarded: false.
  const localOnly = standalone.attachGuestPcmRing({
    memory: { buffer: guestMem },
    base: 16,
    capacityFrames: 4096,
    channels: 2,
    sampleRate: 48000,
  });
  assert.equal(localOnly.ok, true);
  assert.equal(localOnly.attached, true);
  assert.equal(localOnly.forwarded, false);
  assert.equal(localOnly.ring.capacity, 8192);
  assert.equal(localOnly.ring.capacityFrames, 4096);

  // Throwing worker forwards error to onError without throwing out of attachGuestPcmRing.
  standalone.setAudioWorker({
    postMessage() {
      throw new Error('worker terminated');
    },
  });
  const throwingWorkerRes = standalone.attachGuestPcmRing({
    base: 0,
    capacity: 1024,
  });
  assert.equal(throwingWorkerRes.ok, true);
  assert.equal(throwingWorkerRes.forwarded, false);
  assert.equal(errors.length, 1);
  await standalone.destroy();
});

test('W238: setMasterGain and setMuted schedule smooth WebAudio GainNode ramps and fallback worklet messages', async () => {
  // 1. GainNode-capable AudioContext: worklet stays unity gain (no double attenuation),
  //    master GainNode performs smooth linear ramps and forwards to audioWorker.
  const workerMsgs = [];
  const env = createGainNodeMockWebAudioEnv({
    initialState: 'running',
    initialCurrentTime: 2.0,
  });

  const engine = new AudioEngine({
    sampleRate: 48000,
    masterGain: 0.7,
    muted: false,
    audioWorker: { postMessage: (m) => workerMsgs.push(m) },
    AudioContextCtor: env.MockAudioContext,
    AudioWorkletNodeCtor: env.MockAudioWorkletNode,
  });

  const startRes = await engine.start();
  assert.equal(startRes.ok, true);
  assert.equal(env.events.gainNodes.length, 1);
  const gainNode = env.events.gainNodes[0];
  assert.equal(env.events.nodes[0].connectedTo, gainNode);
  assert.equal(gainNode.connectedTo, env.events.contexts[0].destination);
  // Worklet runs at unity gain because masterGainNode owns gain/mute scaling.
  assert.equal(env.events.nodes[0].options.processorOptions.gain, 1.0);
  assert.equal(env.events.nodes[0].options.processorOptions.muted, false);
  assert.equal(gainNode.gain.value, 0.7);

  // Smooth master gain ramp over 20 ms -> linearRampToValueAtTime(0.4, 2.02).
  assert.equal(engine.setMasterGain(0.4, 20), 0.4);
  assert.equal(engine.masterGain, 0.4);
  assert.equal(engine.getStats().masterGain, 0.4);
  assert.equal(engine.getStats().effectiveGain, 0.4);
  const lastOps = gainNode.ops.slice(-3);
  assert.equal(lastOps[0].op, 'cancel');
  assert.equal(lastOps[0].time, 2.0);
  assert.equal(lastOps[1].op, 'set');
  assert.equal(lastOps[1].value, 0.7);
  assert.equal(lastOps[2].op, 'linearRamp');
  assert.equal(lastOps[2].value, 0.4);
  assert.ok(Math.abs(lastOps[2].endTime - 2.02) < 1e-9);

  // Immediate master gain change (rampMs = 0) uses setValueAtTime without linear ramp.
  env.events.contexts[0].currentTime = 3.0;
  assert.equal(engine.setMasterGain(0.9, 0), 0.9);
  assert.deepEqual(gainNode.ops.slice(-2), [
    { op: 'cancel', time: 3.0 },
    { op: 'set', value: 0.9, time: 3.0 },
  ]);

  // Smooth mute ramp over 25 ms -> linear ramp down to 0.0.
  env.events.contexts[0].currentTime = 4.0;
  assert.equal(engine.setMuted(true, 25), true);
  assert.equal(engine.getStats().muted, true);
  assert.equal(engine.getStats().effectiveGain, 0.0);
  const muteOps = gainNode.ops.slice(-3);
  assert.equal(muteOps[0].op, 'cancel');
  assert.equal(muteOps[1].op, 'set');
  assert.equal(muteOps[1].value, 0.9);
  assert.equal(muteOps[2].op, 'linearRamp');
  assert.equal(muteOps[2].value, 0.0);
  assert.ok(Math.abs(muteOps[2].endTime - 4.025) < 1e-9);

  // Smooth unmute ramp over 30 ms -> linear ramp back up to 0.9.
  env.events.contexts[0].currentTime = 5.0;
  assert.equal(engine.setMuted(false, 30), false);
  const unmuteOps = gainNode.ops.slice(-3);
  assert.equal(unmuteOps[1].value, 0.0);
  assert.equal(unmuteOps[2].op, 'linearRamp');
  assert.equal(unmuteOps[2].value, 0.9);
  assert.ok(Math.abs(unmuteOps[2].endTime - 5.03) < 1e-9);

  // Invalid gain inputs are rejected without mutating masterGain; extreme rampMs is clamped.
  const opsBeforeInvalid = gainNode.ops.length;
  assert.equal(engine.setMasterGain(Number.NaN, 10), 0.9);
  assert.equal(engine.setMasterGain(Number.POSITIVE_INFINITY, 10), 0.9);
  assert.equal(engine.setMasterGain('0.5', 10), 0.9);
  assert.equal(gainNode.ops.length, opsBeforeInvalid);
  assert.equal(engine.getStats().invalidGainInputs, 3);

  env.events.contexts[0].currentTime = 6.0;
  assert.equal(engine.setMasterGain(2.5, MAX_GAIN_RAMP_MS + 9999), 1.0);
  assert.equal(gainNode.ops[gainNode.ops.length - 1].value, 1.0);
  assert.ok(
    Math.abs(gainNode.ops[gainNode.ops.length - 1].endTime - (6.0 + MAX_GAIN_RAMP_MS / 1000)) < 1e-9
  );
  assert.equal(workerMsgs.some((m) => m.type === 'setMasterGain' && m.rampMs === 20), true);
  assert.equal(workerMsgs.some((m) => m.type === 'setMuted' && m.muted === true && m.rampMs === 25), true);

  await engine.destroy();
  assert.equal(gainNode.disconnected, true);

  // 2. Worklet-only fallback (no createGain): forwards rampMs / rampFrames metadata cleanly.
  const fallbackEnv = createMockWebAudioEnv({ initialState: 'running' });
  const fallbackEngine = new AudioEngine({
    sampleRate: 48000,
    AudioContextCtor: fallbackEnv.MockAudioContext,
    AudioWorkletNodeCtor: fallbackEnv.MockAudioWorkletNode,
  });
  await fallbackEngine.start();

  fallbackEngine.setMasterGain(0.5, 20);
  fallbackEngine.setMasterGain(0.8, 0);
  fallbackEngine.setMuted(true, 10);
  fallbackEngine.setMuted(false, 0);
  fallbackEngine.setMuted(true); // legacy 1-arg call keeps exact payload

  assert.deepEqual(fallbackEnv.events.nodes[0].postedMessages, [
    { type: 'setGain', gain: 0.5, rampMs: 20, rampFrames: 960 },
    { type: 'setGain', gain: 0.8, rampMs: 0, immediate: true },
    { type: 'setMuted', muted: true, rampMs: 10, rampFrames: 480 },
    { type: 'setMuted', muted: false, rampMs: 0, immediate: true },
    { type: 'setMuted', muted: true },
  ]);
  await fallbackEngine.destroy();
});

test('W238: unlockOnGesture resumes suspended AudioContext with click-free auto-fade-in and guards edge cases', async () => {
  // 1. GainNode-capable suspended AudioContext starts silent (0.0) and smoothly fades in on unlockOnGesture().
  const workerMsgs = [];
  const env = createGainNodeMockWebAudioEnv({
    initialState: 'suspended',
    initialCurrentTime: 10.0,
  });
  const engine = new AudioEngine({
    sampleRate: 48000,
    volume: 0.8,
    unlockFadeInMs: 40,
    audioWorker: { postMessage: (m) => workerMsgs.push(m) },
    AudioContextCtor: env.MockAudioContext,
    AudioWorkletNodeCtor: env.MockAudioWorkletNode,
  });

  // Calling unlockOnGesture() before start() fails gracefully without throwing.
  const preStart = await engine.unlockOnGesture();
  assert.equal(preStart.ok, false);
  assert.equal(preStart.unlocked, false);
  assert.equal(preStart.reason, 'not-started');
  assert.equal(engine.getStats().unlockedOnGesture, false);

  await engine.start();
  const gainNode = env.events.gainNodes[0];
  // Suspended context holds master gain at 0.0 to eliminate startup pop.
  assert.equal(gainNode.gain.value, 0.0);

  // Arm gesture listeners; unlockOnGesture() should also clean them up upon unlocking.
  const target = createMockEventTarget();
  engine.resumeOnUserGesture(target);
  assert.equal(target.listenerCount('pointerdown'), 1);

  const unlockRes = await engine.unlockOnGesture();
  assert.equal(unlockRes.ok, true);
  assert.equal(unlockRes.unlocked, true);
  assert.equal(unlockRes.alreadyRunning, false);
  assert.equal(unlockRes.state, 'running');
  assert.equal(unlockRes.fadeInMs, 40);
  assert.equal(unlockRes.effectiveGain, 0.8);
  assert.equal(engine.unlockedOnGesture, true);
  assert.equal(engine.getStats().unlockedOnGesture, true);
  assert.equal(target.listenerCount('pointerdown'), 0);

  // Verify fade-in ramp from 0.0 -> 0.8 over 40 ms (0.04 s).
  const rampOps = gainNode.ops.slice(-3);
  assert.equal(rampOps[0].op, 'cancel');
  assert.equal(rampOps[1].op, 'set');
  assert.equal(rampOps[1].value, 0.0);
  assert.equal(rampOps[2].op, 'linearRamp');
  assert.equal(rampOps[2].value, 0.8);
  assert.ok(Math.abs(rampOps[2].endTime - 10.04) < 1e-9);
  assert.equal(
    workerMsgs.some((m) => m.type === 'audio-context-state' && m.unlockedOnGesture === true),
    true
  );

  // Calling unlockOnGesture() when already running does not cause an audio dip.
  const opsCount = gainNode.ops.length;
  const repeatUnlock = await engine.unlockOnGesture({ rampMs: 50 });
  assert.equal(repeatUnlock.ok, true);
  assert.equal(repeatUnlock.alreadyRunning, true);
  assert.equal(repeatUnlock.fadeInMs, 0);
  assert.equal(gainNode.ops.length, opsCount);

  await engine.destroy();
  const closedUnlock = await engine.unlockOnGesture();
  assert.equal(closedUnlock.ok, false);
  assert.equal(closedUnlock.reason, 'closed');

  // 2. Worklet-only fallback (no createGain): anchors worklet gain to 0 then ramps to target.
  const fallbackEnv = createMockWebAudioEnv({ initialState: 'suspended' });
  const fallbackEngine = new AudioEngine({
    sampleRate: 48000,
    volume: 0.6,
    AudioContextCtor: fallbackEnv.MockAudioContext,
    AudioWorkletNodeCtor: fallbackEnv.MockAudioWorkletNode,
  });
  await fallbackEngine.start();
  const fbRes = await fallbackEngine.unlockOnGesture(DEFAULT_UNLOCK_FADE_IN_MS);
  assert.equal(fbRes.ok, true);
  assert.equal(fbRes.unlocked, true);
  assert.equal(fallbackEngine.getStats().unlockedOnGesture, true);
  assert.deepEqual(fallbackEnv.events.nodes[0].postedMessages, [
    { type: 'setGain', gain: 0.0, immediate: true },
    {
      type: 'setGain',
      gain: 0.6,
      rampMs: DEFAULT_UNLOCK_FADE_IN_MS,
      rampFrames: Math.round((DEFAULT_UNLOCK_FADE_IN_MS / 1000) * 48000),
    },
  ]);
  await fallbackEngine.destroy();

  // 3. Muted engine stays silent (effectiveGain === 0) after unlockOnGesture().
  const mutedEnv = createGainNodeMockWebAudioEnv({
    initialState: 'suspended',
    initialCurrentTime: 1.0,
  });
  const mutedEngine = new AudioEngine({
    volume: 0.75,
    muted: true,
    AudioContextCtor: mutedEnv.MockAudioContext,
    AudioWorkletNodeCtor: mutedEnv.MockAudioWorkletNode,
  });
  await mutedEngine.start();
  const mutedRes = await mutedEngine.unlockOnGesture({ rampMs: 20 });
  assert.equal(mutedRes.ok, true);
  assert.equal(mutedRes.effectiveGain, 0.0);
  assert.equal(mutedEngine.getStats().muted, true);
  assert.equal(mutedEngine.getStats().unlockedOnGesture, true);
  assert.equal(mutedEnv.events.gainNodes[0].gain.value, 0.0);
  await mutedEngine.destroy();

  // 4. Autoplay failure leaves unlockedOnGesture === false without throwing.
  const blockedEnv = createMockWebAudioEnv({
    initialState: 'suspended',
    resumeError: new Error('NotAllowedError: gesture required'),
  });
  const blockedEngine = new AudioEngine({
    AudioContextCtor: blockedEnv.MockAudioContext,
    AudioWorkletNodeCtor: blockedEnv.MockAudioWorkletNode,
  });
  await blockedEngine.start();
  const blockedRes = await blockedEngine.unlockOnGesture();
  assert.equal(blockedRes.ok, false);
  assert.equal(blockedRes.unlocked, false);
  assert.equal(blockedRes.reason, 'resume-failed');
  assert.equal(blockedEngine.getStats().unlockedOnGesture, false);
  await blockedEngine.destroy();
});

test('W238: updateGuestAudioTelemetry sanitizes guest PCM metrics and surfaces them via getStats()', async () => {
  assert.equal(DEFAULT_GAIN_RAMP_MS, 15);
  assert.equal(DEFAULT_UNLOCK_FADE_IN_MS, 25);

  const engine = new AudioEngine({
    masterGain: 0.65,
    muted: false,
  });

  // Initial sanitized defaults in getStats().
  let stats = engine.getStats();
  assert.equal(stats.guestRingAttached, false);
  assert.equal(stats.guestFramesDrained, 0);
  assert.equal(stats.guestUnderruns, 0);
  assert.equal(stats.guestOverruns, 0);
  assert.equal(stats.guestPeakLevel, 0);
  assert.equal(stats.unlockedOnGesture, false);
  assert.equal(stats.muted, false);
  assert.equal(stats.masterGain, 0.65);

  // Valid update + nested audio-stats shape + guestOverflows alias.
  const updated = engine.updateGuestAudioTelemetry({
    guestRingAttached: true,
    guestFramesDrained: 9600.9,
    guestUnderruns: 3,
    guestOverruns: 1,
    guestPeakLevel: 0.84,
  });
  assert.deepEqual(updated, {
    guestRingAttached: true,
    guestFramesDrained: 9600,
    guestUnderruns: 3,
    guestOverruns: 1,
    guestPeakLevel: 0.84,
    unlockedOnGesture: false,
    muted: false,
    masterGain: 0.65,
  });

  stats = engine.getStats();
  assert.equal(stats.guestRingAttached, true);
  assert.equal(stats.guestFramesDrained, 9600);
  assert.equal(stats.guestUnderruns, 3);
  assert.equal(stats.guestOverruns, 1);
  assert.equal(stats.guestPeakLevel, 0.84);

  // Nested `{ stats: { ... } }` envelope and peak clamping to [0, 1].
  engine.updateGuestAudioTelemetry({
    type: 'audio-stats',
    stats: {
      guestFramesDrained: 14400,
      guestOverflows: 4,
      guestPeakLevel: 1.75,
    },
  });
  stats = engine.getStats();
  assert.equal(stats.guestFramesDrained, 14400);
  assert.equal(stats.guestOverruns, 4);
  assert.equal(stats.guestPeakLevel, 1.0);

  // Non-finite, negative, invalid, and hostile throwing getters never corrupt telemetry.
  engine.updateGuestAudioTelemetry(null);
  engine.updateGuestAudioTelemetry({
    guestRingAttached: 'yes',
    guestFramesDrained: Number.NaN,
    guestUnderruns: -5,
    guestOverruns: Number.POSITIVE_INFINITY,
    guestPeakLevel: Number.NEGATIVE_INFINITY,
  });
  engine.updateGuestAudioTelemetry({
    get guestFramesDrained() {
      throw new Error('hostile getter');
    },
    get guestPeakLevel() {
      throw new Error('hostile getter');
    },
  });

  stats = engine.getStats();
  assert.equal(stats.guestRingAttached, true);
  assert.equal(stats.guestFramesDrained, 14400);
  assert.equal(stats.guestUnderruns, 3);
  assert.equal(stats.guestOverruns, 4);
  assert.equal(stats.guestPeakLevel, 1.0);

  // Negative peak level clamps cleanly to 0.0.
  engine.updateGuestAudioTelemetry({ guestPeakLevel: -0.25 });
  assert.equal(engine.getStats().guestPeakLevel, 0.0);

  // Mutating getStats() snapshot cannot mutate internal telemetry.
  stats.guestFramesDrained = -999;
  stats.guestPeakLevel = Number.NaN;
  stats.guestRingAttached = false;
  assert.equal(engine.getStats().guestFramesDrained, 14400);
  assert.equal(engine.getStats().guestPeakLevel, 0.0);
  assert.equal(engine.getStats().guestRingAttached, true);

  // reset({ preserveStats: true }) preserves guest counters; full reset() clears them.
  engine.reset({ preserveStats: true });
  assert.equal(engine.getStats().guestFramesDrained, 14400);
  assert.equal(engine.getStats().guestRingAttached, true);

  engine.reset();
  assert.equal(engine.getStats().guestFramesDrained, 0);
  assert.equal(engine.getStats().guestUnderruns, 0);
  assert.equal(engine.getStats().guestOverruns, 0);
  assert.equal(engine.getStats().guestPeakLevel, 0);
  assert.equal(engine.getStats().guestRingAttached, true);

  engine.reset({ detachGuestRing: true });
  assert.equal(engine.getStats().guestRingAttached, false);

  await engine.destroy();
});

test('W293: AudioContext statechange interruption arms gesture auto-recovery, avoids unhandled rejections, and cleans up in dispose/destroy/close', async () => {
  let failNextResume = false;
  const workerMsgs = [];

  class InterruptibleAudioContext {
    constructor(opts = {}) {
      this.sampleRate = opts.sampleRate ?? 48000;
      this.state = 'suspended';
      this.destination = { kind: 'mock-destination' };
      this.resumeCalls = 0;
      this.suspendCalls = 0;
      this.closeCalls = 0;
      this._listeners = new Map();
      this.audioWorklet = { addModule: async () => {} };
    }
    addEventListener(type, fn) {
      if (!this._listeners.has(type)) this._listeners.set(type, new Set());
      this._listeners.get(type).add(fn);
    }
    removeEventListener(type, fn) {
      const set = this._listeners.get(type);
      if (set) set.delete(fn);
    }
    listenerCount(type) {
      return this._listeners.has(type) ? this._listeners.get(type).size : 0;
    }
    _emit(type) {
      const set = this._listeners.get(type);
      if (!set) return;
      for (const fn of Array.from(set)) fn({ target: this });
    }
    async resume() {
      this.resumeCalls += 1;
      if (failNextResume) {
        failNextResume = false;
        throw new Error('NotAllowedError: gesture needed after interruption');
      }
      this.state = 'running';
      this._emit('statechange');
    }
    async suspend() {
      this.suspendCalls += 1;
      this.state = 'suspended';
      this._emit('statechange');
    }
    async close() {
      this.closeCalls += 1;
      this.state = 'closed';
      this._emit('statechange');
    }
  }

  const { MockAudioWorkletNode } = createMockWebAudioEnv();
  const target = createMockEventTarget();
  const engine = new AudioEngine({
    audioWorker: { postMessage: (m) => workerMsgs.push(m) },
    AudioContextCtor: InterruptibleAudioContext,
    AudioWorkletNodeCtor: MockAudioWorkletNode,
  });

  await engine.start();
  const ctx = engine.context;
  assert.equal(ctx.listenerCount('statechange'), 1);
  assert.equal(engine.getStats().interruptionCount, 0);
  assert.equal(engine.getStats().autoRecoveryCount, 0);
  assert.equal(engine.getStats().pendingAutoRecovery, false);

  // Initial user gesture unlocks the suspended context (not an interruption).
  engine.resumeOnUserGesture(target);
  target.dispatch('pointerdown');
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(engine.outputState, 'running');
  assert.equal(engine.interruptionCount, 0);
  assert.equal(engine.autoRecoveryCount, 0);
  assert.equal(target.listenerCount('pointerdown'), 0);

  // Unexpected OS audio interruption while playing -> increments interruptionCount once & re-arms gesture recovery.
  ctx.state = 'interrupted';
  ctx._emit('statechange');
  assert.equal(engine.interruptionCount, 1);
  assert.equal(engine.autoRecoveryCount, 0);
  assert.equal(engine.pendingAutoRecovery, true);
  assert.equal(engine.getStats().interruptionCount, 1);
  assert.equal(engine.getStats().autoRecoveryCount, 0);
  assert.equal(engine.getStats().pendingAutoRecovery, true);
  assert.equal(target.listenerCount('pointerdown'), 1);
  assert.equal(target.listenerCount('keydown'), 1);
  assert.equal(
    workerMsgs.some((m) => m.type === 'audio-context-state' && m.outputState === 'suspended' && m.interrupted === true),
    true
  );

  // Duplicate statechange or interrupted->suspended transition does not double-count interruptionCount.
  ctx._emit('statechange');
  ctx.state = 'suspended';
  ctx._emit('statechange');
  assert.equal(engine.interruptionCount, 1);

  // Next user gesture recovers automatically, updates autoRecoveryCount, and cleans up gesture listeners.
  target.dispatch('click');
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(engine.outputState, 'running');
  assert.equal(engine.interruptionCount, 1);
  assert.equal(engine.autoRecoveryCount, 1);
  assert.equal(engine.pendingAutoRecovery, false);
  assert.equal(target.listenerCount('pointerdown'), 0);
  assert.equal(target.listenerCount('click'), 0);
  assert.equal(
    workerMsgs.some(
      (m) => m.type === 'audio-context-state' && m.outputState === 'active' && m.recoveredFromInterruption === true
    ),
    true
  );

  // Unexpected drop directly to 'suspended' while running also triggers recovery; failed gesture retries cleanly.
  ctx.state = 'suspended';
  ctx._emit('statechange');
  assert.equal(engine.interruptionCount, 2);
  assert.equal(engine.pendingAutoRecovery, true);
  assert.equal(target.listenerCount('keydown'), 1);

  failNextResume = true;
  target.dispatch('keydown');
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(engine.autoRecoveryCount, 1);
  assert.equal(engine.pendingAutoRecovery, true);
  assert.equal(target.listenerCount('keydown'), 1);

  target.dispatch('keydown');
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(engine.outputState, 'running');
  assert.equal(engine.interruptionCount, 2);
  assert.equal(engine.autoRecoveryCount, 2);
  assert.equal(engine.pendingAutoRecovery, false);
  assert.equal(target.listenerCount('keydown'), 0);

  // Intentional engine.suspend() must NOT be miscounted as an unexpected interruption or arm gesture listeners.
  assert.equal(await engine.suspend(), true);
  assert.equal(engine.interruptionCount, 2);
  assert.equal(engine.pendingAutoRecovery, false);
  assert.equal(target.listenerCount('pointerdown'), 0);

  // Resume, then interrupt again and verify dispose() removes both statechange and armed gesture listeners.
  assert.equal(await engine.resume(), true);
  ctx.state = 'interrupted';
  ctx._emit('statechange');
  assert.equal(engine.interruptionCount, 3);
  assert.equal(target.listenerCount('pointerdown'), 1);

  await engine.dispose();
  assert.equal(engine.outputState, 'closed');
  assert.equal(ctx.listenerCount('statechange'), 0);
  assert.equal(target.listenerCount('pointerdown'), 0);
  assert.equal(target.listenerCount('keydown'), 0);

  // Emitting statechange on the closed context after dispose() is a safe no-op.
  ctx.state = 'interrupted';
  ctx._emit('statechange');
  assert.equal(engine.interruptionCount, 3);
});

test('W293: recoverFromInterruption handles explicit recovery, click-free GainNode fade-in, spontaneous OS resume, and error/reset edge cases', async () => {
  const env = createGainNodeMockWebAudioEnv({
    initialState: 'running',
    initialCurrentTime: 5.0,
  });
  const target = createMockEventTarget();
  const engine = new AudioEngine({
    volume: 0.75,
    gestureTarget: target,
    AudioContextCtor: env.MockAudioContext,
    AudioWorkletNodeCtor: env.MockAudioWorkletNode,
  });

  // Before start() -> fails gracefully.
  const preStart = await engine.recoverFromInterruption();
  assert.equal(preStart.ok, false);
  assert.equal(preStart.recovered, false);
  assert.equal(preStart.reason, 'not-started');

  await engine.start();
  const ctx = env.events.contexts[0];
  const gainNode = env.events.gainNodes[0];

  // Already running -> idempotent no-op.
  const already = await engine.recoverFromInterruption();
  assert.equal(already.ok, true);
  assert.equal(already.recovered, false);
  assert.equal(already.alreadyRunning, true);
  assert.equal(already.autoRecoveryCount, 0);

  // Interrupt running context -> explicit recoverFromInterruption() restores cleanly with GainNode ramp.
  ctx.state = 'interrupted';
  ctx._emit('statechange');
  assert.equal(engine.interruptionCount, 1);
  assert.equal(target.listenerCount('pointerdown'), 1);

  ctx.currentTime = 6.0;
  const recRes = await engine.recoverFromInterruption({ fadeInMs: 30 });
  assert.equal(recRes.ok, true);
  assert.equal(recRes.recovered, true);
  assert.equal(recRes.alreadyRunning, false);
  assert.equal(recRes.state, 'running');
  assert.equal(recRes.interruptionCount, 1);
  assert.equal(recRes.autoRecoveryCount, 1);
  assert.equal(recRes.reason, null);
  assert.equal(engine.pendingAutoRecovery, false);
  assert.equal(target.listenerCount('pointerdown'), 0);

  const rampOps = gainNode.ops.slice(-3);
  assert.equal(rampOps[0].op, 'cancel');
  assert.equal(rampOps[1].op, 'set');
  assert.equal(rampOps[1].value, 0.0);
  assert.equal(rampOps[2].op, 'linearRamp');
  assert.equal(rampOps[2].value, 0.75);
  assert.ok(Math.abs(rampOps[2].endTime - 6.03) < 1e-9);

  // Spontaneous OS/browser resume via statechange increments autoRecoveryCount once and disarms gesture listeners.
  ctx.state = 'interrupted';
  ctx._emit('statechange');
  assert.equal(engine.interruptionCount, 2);
  assert.equal(target.listenerCount('pointerdown'), 1);

  ctx.state = 'running';
  ctx._emit('statechange');
  assert.equal(engine.interruptionCount, 2);
  assert.equal(engine.autoRecoveryCount, 2);
  assert.equal(engine.pendingAutoRecovery, false);
  assert.equal(target.listenerCount('pointerdown'), 0);

  // reset({ preserveStats: true }) keeps interruption/recovery counts; full reset() clears them.
  engine.reset({ preserveStats: true });
  assert.equal(engine.getStats().interruptionCount, 2);
  assert.equal(engine.getStats().autoRecoveryCount, 2);

  engine.reset();
  assert.equal(engine.getStats().interruptionCount, 0);
  assert.equal(engine.getStats().autoRecoveryCount, 0);
  assert.equal(engine.getAudioStats().interruptionCount, 0);
  assert.equal(engine.getAudioStats().autoRecoveryCount, 0);

  // Failed explicit recovery returns structured error and arms gesture fallback without unhandled rejection.
  const failEnv = createMockWebAudioEnv({
    initialState: 'running',
    resumeError: new Error('NotAllowedError: blocked'),
  });
  const failTarget = createMockEventTarget();
  const failEngine = new AudioEngine({
    gestureTarget: failTarget,
    AudioContextCtor: failEnv.MockAudioContext,
    AudioWorkletNodeCtor: failEnv.MockAudioWorkletNode,
  });
  await failEngine.start();
  failEnv.events.contexts[0].state = 'interrupted';
  failEnv.events.contexts[0]._emit('statechange');

  const failRec = await failEngine.recoverFromInterruption();
  assert.equal(failRec.ok, false);
  assert.equal(failRec.recovered, false);
  assert.equal(failRec.reason, 'resume-failed');
  assert.equal(failRec.interruptionCount, 1);
  assert.equal(failRec.autoRecoveryCount, 0);
  assert.match(failRec.error, /NotAllowedError/);
  assert.equal(failTarget.listenerCount('pointerdown'), 1);

  await failEngine.close();
  assert.equal(failTarget.listenerCount('pointerdown'), 0);

  await engine.dispose();
  const closedRec = await engine.recoverFromInterruption();
  assert.equal(closedRec.ok, false);
  assert.equal(closedRec.reason, 'closed');
});

test('W293: setTargetLatencyMs clamps [10, 120] (default 30), forwards control message to audioWorker, and surfaces in getStats()', async () => {
  assert.equal(MIN_TARGET_LATENCY_MS, 10);
  assert.equal(MAX_TARGET_LATENCY_MS, 120);
  assert.equal(DEFAULT_TARGET_LATENCY_MS, 30);

  // Standalone clampTargetLatencyMs guardrails.
  assert.equal(clampTargetLatencyMs(30), 30);
  assert.equal(clampTargetLatencyMs(10), 10);
  assert.equal(clampTargetLatencyMs(120), 120);
  assert.equal(clampTargetLatencyMs(0), 10);
  assert.equal(clampTargetLatencyMs(-50), 10);
  assert.equal(clampTargetLatencyMs(-0), 10);
  assert.equal(clampTargetLatencyMs(250), 120);
  assert.equal(clampTargetLatencyMs(Number.NaN), 30);
  assert.equal(clampTargetLatencyMs(Number.POSITIVE_INFINITY), 30);
  assert.equal(clampTargetLatencyMs(Number.NEGATIVE_INFINITY), 30);
  assert.equal(clampTargetLatencyMs('45'), 30);
  assert.equal(clampTargetLatencyMs(Number.NaN, 50), 50);
  assert.equal(clampTargetLatencyMs(Number.NaN, 2), 10);
  assert.equal(clampTargetLatencyMs(Number.NaN, 999), 120);

  // Constructor default and clamping.
  assert.equal(new AudioEngine().targetLatencyMs, 30);
  assert.equal(new AudioEngine({ targetLatencyMs: 5 }).targetLatencyMs, 10);
  assert.equal(new AudioEngine({ targetLatencyMs: 300 }).targetLatencyMs, 120);
  assert.equal(new AudioEngine({ targetLatencyMs: Number.NaN }).targetLatencyMs, 30);
  assert.equal(new AudioEngine({ targetLatencyMs: 45 }).getStats().targetLatencyMs, 45);

  // Worker postMessage forwarding & non-finite rejection.
  const workerMsgs = [];
  const engine = new AudioEngine({
    sampleRate: 48000,
    audioWorker: { postMessage: (msg) => workerMsgs.push(msg) },
  });

  assert.equal(engine.targetLatencyMs, 30);
  assert.equal(engine.getStats().targetLatencyMs, 30);
  assert.equal(engine.getAudioStats().targetLatencyMs, 30);
  assert.equal(engine.getTelemetry().targetLatencyMs, 30);

  assert.equal(engine.setTargetLatencyMs(33.33), 33.33);
  assert.equal(engine.setTargetLatencyMs(4), 10);
  assert.equal(engine.setTargetLatencyMs(500), 120);

  // Non-finite / non-number inputs preserve active value and never forward.
  const msgCountBeforeInvalid = workerMsgs.length;
  assert.equal(engine.setTargetLatencyMs(Number.NaN), 120);
  assert.equal(engine.setTargetLatencyMs(Number.POSITIVE_INFINITY), 120);
  assert.equal(engine.setTargetLatencyMs(Number.NEGATIVE_INFINITY), 120);
  assert.equal(engine.setTargetLatencyMs('25'), 120);
  assert.equal(engine.setTargetLatencyMs(null), 120);
  assert.equal(engine.setTargetLatencyMs(undefined), 120);
  assert.equal(workerMsgs.length, msgCountBeforeInvalid);

  assert.deepEqual(workerMsgs, [
    {
      type: 'setTargetLatencyMs',
      setTargetLatencyMs: true,
      targetLatencyMs: 33.33,
      targetLatencyFrames: Math.round((33.33 / 1000) * 48000),
      sampleRate: 48000,
    },
    {
      type: 'setTargetLatencyMs',
      setTargetLatencyMs: true,
      targetLatencyMs: 10,
      targetLatencyFrames: 480,
      sampleRate: 48000,
    },
    {
      type: 'setTargetLatencyMs',
      setTargetLatencyMs: true,
      targetLatencyMs: 120,
      targetLatencyFrames: 5760,
      sampleRate: 48000,
    },
  ]);

  // Defensive getStats() snapshot cannot corrupt internal targetLatencyMs.
  const snap = engine.getStats();
  assert.equal(snap.targetLatencyMs, 120);
  snap.targetLatencyMs = -999;
  assert.equal(engine.getStats().targetLatencyMs, 120);

  // Direct bridge method forwarding.
  const bridgeCalls = [];
  engine.setAudioWorker({
    setTargetLatencyMs(ms) {
      bridgeCalls.push(ms);
    },
  });
  assert.equal(engine.setTargetLatencyMs(24), 24);
  assert.deepEqual(bridgeCalls, [24]);

  await engine.destroy();
});

test('W293: saveAudioPreferences / loadAudioPreferences persist sanitized { muted, volume } under playgta6:audio:prefs:v1 and handle storage errors gracefully', async () => {
  assert.equal(AUDIO_PREFS_STORAGE_KEY, 'playgta6:audio:prefs:v1');

  const storeMap = new Map();
  const mockStorage = {
    getItem(key) {
      return storeMap.has(key) ? storeMap.get(key) : null;
    },
    setItem(key, value) {
      storeMap.set(key, String(value));
    },
  };

  // Empty storage returns not-found without throwing or mutating engine state.
  const engine1 = new AudioEngine({ volume: 0.65, muted: false });
  const emptyLoad = engine1.loadAudioPreferences(mockStorage);
  assert.equal(emptyLoad.ok, true);
  assert.equal(emptyLoad.loaded, false);
  assert.equal(emptyLoad.applied, false);
  assert.equal(emptyLoad.prefs, null);
  assert.equal(emptyLoad.reason, 'not-found');
  assert.equal(engine1.volume, 0.65);
  assert.equal(engine1.muted, false);

  // Save engine preferences -> stored JSON contains strictly sanitized { muted, volume }.
  engine1.setVolume(0.42);
  engine1.setMuted(true);
  const saveRes = engine1.saveAudioPreferences(mockStorage);
  assert.equal(saveRes.ok, true);
  assert.equal(saveRes.saved, true);
  assert.equal(saveRes.key, 'playgta6:audio:prefs:v1');
  assert.deepEqual(saveRes.prefs, { muted: true, volume: 0.42 });

  const rawStored = storeMap.get('playgta6:audio:prefs:v1');
  assert.equal(typeof rawStored, 'string');
  const parsedStored = JSON.parse(rawStored);
  assert.deepEqual(Object.keys(parsedStored).sort(), ['muted', 'volume']);
  assert.deepEqual(parsedStored, { muted: true, volume: 0.42 });

  // Load into a fresh AudioEngine session and verify worklet/worker receive applied values.
  const workerMsgs = [];
  const { MockAudioContext, MockAudioWorkletNode, events } = createMockWebAudioEnv({
    initialState: 'running',
  });
  const engine2 = new AudioEngine({
    volume: 1.0,
    muted: false,
    audioWorker: { postMessage: (m) => workerMsgs.push(m) },
    AudioContextCtor: MockAudioContext,
    AudioWorkletNodeCtor: MockAudioWorkletNode,
  });
  await engine2.start();

  const inspectOnly = engine2.loadAudioPreferences(mockStorage, { apply: false });
  assert.equal(inspectOnly.ok, true);
  assert.equal(inspectOnly.loaded, true);
  assert.equal(inspectOnly.applied, false);
  assert.deepEqual(inspectOnly.prefs, { muted: true, volume: 0.42 });
  assert.equal(engine2.volume, 1.0);
  assert.equal(engine2.muted, false);

  const loadRes = engine2.loadAudioPreferences(mockStorage);
  assert.equal(loadRes.ok, true);
  assert.equal(loadRes.loaded, true);
  assert.equal(loadRes.applied, true);
  assert.deepEqual(loadRes.prefs, { muted: true, volume: 0.42 });
  assert.equal(engine2.volume, 0.42);
  assert.equal(engine2.muted, true);
  assert.equal(engine2.getStats().effectiveGain, 0.0);
  assert.deepEqual(events.nodes[0].postedMessages, [
    { type: 'setGain', gain: 0.42 },
    { type: 'setMuted', muted: true },
  ]);
  assert.equal(workerMsgs.some((m) => m.type === 'setMasterGain' && m.volume === 0.42), true);
  assert.equal(workerMsgs.some((m) => m.type === 'setMuted' && m.muted === true), true);

  // Corrupted JSON, non-object JSON, invalid-typed payloads, and extra fields are sanitized safely.
  storeMap.set('playgta6:audio:prefs:v1', '{not valid json');
  const badJson = engine2.loadAudioPreferences(mockStorage);
  assert.equal(badJson.ok, false);
  assert.equal(badJson.loaded, false);
  assert.equal(badJson.reason, 'invalid-data');
  assert.equal(engine2.volume, 0.42);
  assert.equal(engine2.muted, true);

  storeMap.set('playgta6:audio:prefs:v1', JSON.stringify([1, 2, 3]));
  assert.equal(engine2.loadAudioPreferences(mockStorage).reason, 'invalid-data');

  storeMap.set('playgta6:audio:prefs:v1', JSON.stringify({ muted: 'yes', volume: 'loud', secret: 'xyz' }));
  assert.equal(engine2.loadAudioPreferences(mockStorage).reason, 'invalid-data');
  assert.equal(engine2.volume, 0.42);
  assert.equal(engine2.muted, true);

  // Extra fields are stripped and out-of-range volume is clamped on load.
  storeMap.set(
    'playgta6:audio:prefs:v1',
    JSON.stringify({ muted: false, volume: 5.5, token: 'secret-value', extra: 123 })
  );
  const clampedLoad = engine2.loadAudioPreferences(mockStorage);
  assert.equal(clampedLoad.ok, true);
  assert.equal(clampedLoad.loaded, true);
  assert.deepEqual(clampedLoad.prefs, { muted: false, volume: 1.0 });
  assert.deepEqual(Object.keys(clampedLoad.prefs).sort(), ['muted', 'volume']);
  assert.equal(engine2.volume, 1.0);
  assert.equal(engine2.muted, false);

  // SecurityError / QuotaExceededError / unavailable storage are caught cleanly without throwing.
  const securityErrorStorage = {
    getItem() {
      const err = new Error('The operation is insecure');
      err.name = 'SecurityError';
      throw err;
    },
    setItem() {
      const err = new Error('The operation is insecure');
      err.name = 'SecurityError';
      throw err;
    },
  };
  const secSave = engine2.saveAudioPreferences(securityErrorStorage);
  assert.equal(secSave.ok, false);
  assert.equal(secSave.saved, false);
  assert.equal(secSave.reason, 'storage-error');
  assert.match(secSave.error, /SecurityError/);

  const secLoad = engine2.loadAudioPreferences(securityErrorStorage);
  assert.equal(secLoad.ok, false);
  assert.equal(secLoad.loaded, false);
  assert.equal(secLoad.reason, 'storage-error');
  assert.match(secLoad.error, /SecurityError/);

  // Explicit null storage -> storage-unavailable without throwing.
  assert.equal(engine2.saveAudioPreferences(null).reason, 'storage-unavailable');
  assert.equal(engine2.loadAudioPreferences(null).reason, 'storage-unavailable');

  // Standalone exported helpers strip extra fields and guard throwing globalThis.localStorage getter.
  const standaloneSave = saveAudioPreferences(
    { muted: 1, volume: -3.0, leakedPath: 'C:\\Users\\secret' },
    mockStorage
  );
  assert.equal(standaloneSave.ok, true);
  assert.deepEqual(standaloneSave.prefs, { muted: true, volume: 0.0 });
  assert.deepEqual(JSON.parse(storeMap.get('playgta6:audio:prefs:v1')), { muted: true, volume: 0.0 });

  const standaloneLoad = loadAudioPreferences(mockStorage);
  assert.equal(standaloneLoad.ok, true);
  assert.equal(standaloneLoad.loaded, true);
  assert.deepEqual(standaloneLoad.prefs, { muted: true, volume: 0.0 });

  const prevDesc = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  try {
    Object.defineProperty(globalThis, 'localStorage', {
      configurable: true,
      get() {
        const err = new Error('Storage access denied');
        err.name = 'SecurityError';
        throw err;
      },
    });
    assert.equal(engine2.saveAudioPreferences().reason, 'storage-unavailable');
    assert.equal(engine2.loadAudioPreferences().reason, 'storage-unavailable');
  } finally {
    if (prevDesc) {
      Object.defineProperty(globalThis, 'localStorage', prevDesc);
    } else {
      delete globalThis.localStorage;
    }
  }

  await engine1.destroy();
  await engine2.destroy();
});

test('O43: AudioEngine subscribes to BroadcastChannel("game-progress") audio-stats telemetry and closes cleanly on destroy()', async () => {
  const listeners = new Map();
  let openChannels = 0;

  class MockBroadcastChannel {
    constructor(name) {
      this.name = name;
      this.closed = false;
      this.onmessage = null;
      openChannels += 1;
      if (!listeners.has(name)) {
        listeners.set(name, new Set());
      }
      listeners.get(name).add(this);
    }
    postMessage(msg) {
      if (this.closed) return;
      const peers = listeners.get(this.name);
      if (!peers) return;
      for (const peer of peers) {
        if (peer !== this && !peer.closed && typeof peer.onmessage === 'function') {
          peer.onmessage({ data: msg });
        }
      }
    }
    close() {
      if (!this.closed) {
        this.closed = true;
        openChannels -= 1;
        const peers = listeners.get(this.name);
        if (peers) peers.delete(this);
      }
    }
  }

  const engine = new AudioEngine({
    BroadcastChannelCtor: MockBroadcastChannel,
    listenTelemetryBroadcast: true,
  });
  assert.equal(openChannels, 1);
  assert.equal(engine.getStats().guestFramesDrained, 0);
  assert.equal(engine.getStats().guestRingAttached, false);

  // Broadcast live audio-stats from a simulated audio_worker on 'game-progress'.
  const sender = new MockBroadcastChannel('game-progress');
  sender.postMessage({ label: 'WebGPU frame #1 presented' }); // unrelated progress message ignored
  assert.equal(engine.getStats().guestFramesDrained, 0);

  sender.postMessage({
    type: 'audio-stats',
    timestamp: 12345,
    stats: {
      outputState: 'headless-drain',
      framesDrained: 28800,
      guestRingAttached: true,
      guestFramesDrained: 28800,
      guestUnderruns: 2,
      guestOverruns: 0,
      guestPeakLevel: 0.625,
    },
  });

  const stats = engine.getStats();
  assert.equal(stats.guestRingAttached, true);
  assert.equal(stats.guestFramesDrained, 28800);
  assert.equal(stats.guestUnderruns, 2);
  assert.equal(stats.guestOverruns, 0);
  assert.equal(stats.guestPeakLevel, 0.625);

  const telem = engine.getTelemetry();
  assert.equal(telem.guestRingAttached, true);
  assert.equal(telem.guestFramesDrained, 28800);

  // Destroy closes telemetry channel and ignores subsequent messages.
  await engine.destroy();
  assert.equal(openChannels, 1); // only sender remains open
  sender.postMessage({
    type: 'audio-stats',
    stats: { guestFramesDrained: 99999 },
  });
  assert.equal(engine.getStats().guestFramesDrained, 28800);

  sender.close();
  assert.equal(openChannels, 0);
});

test('Jitter/stutter fix (#1028/#663): AudioEngine configures 341ms ring, click-free 160-frame ramps, 80ms hysteresis prebuffer, and 210ms drift trim', async () => {
  // 1. Default 48 kHz live config.
  {
    const env = createMockWebAudioEnv({
      initialState: 'running',
      actualSampleRate: 48000,
      attachRealProcessor: true,
    });
    const engine = new AudioEngine({
      AudioContextCtor: env.MockAudioContext,
      AudioWorkletNodeCtor: env.MockAudioWorkletNode,
      listenTelemetryBroadcast: false,
    });
    const res = await engine.start();
    assert.equal(res.ok, true);
    assert.equal(engine.getStats().capacityFrames, 16384);

    const opts = env.events.nodes[0].options.processorOptions;
    assert.equal(opts.declickFrames, 160);
    assert.equal(opts.fadeCurve, 'cosine');
    assert.equal(opts.prebufferFrames, 3840);
    assert.equal(opts.maxBufferedFrames, 10080);
    assert.equal(opts.targetWatermarkFrames, 4608);
    assert.equal(opts.trimCrossfadeFrames, 64);

    // Re-attaching the same SAB is a no-op and does not reset worklet de-click state.
    const postedBefore = env.events.nodes[0].postedMessages.length;
    assert.equal(engine.attachSab(engine.sab), engine.sab);
    assert.equal(env.events.nodes[0].postedMessages.length, postedBefore);

    await engine.destroy();
  }

  // 2. Host sample-rate scaling (44.1 kHz and 96 kHz).
  for (const hostRate of [44100, 96000]) {
    const env = createMockWebAudioEnv({
      initialState: 'running',
      actualSampleRate: hostRate,
      attachRealProcessor: true,
    });
    const engine = new AudioEngine({
      AudioContextCtor: env.MockAudioContext,
      AudioWorkletNodeCtor: env.MockAudioWorkletNode,
      listenTelemetryBroadcast: false,
    });
    await engine.start();
    const opts = env.events.nodes[0].options.processorOptions;
    assert.ok(opts.declickFrames >= 128 && opts.declickFrames <= 384);
    assert.ok(opts.prebufferFrames > 0 && opts.prebufferFrames <= 8192);
    assert.ok(opts.maxBufferedFrames > opts.targetWatermarkFrames);
    assert.ok(opts.targetWatermarkFrames >= opts.prebufferFrames);
    await engine.destroy();
  }

  // 3. End-to-end bursty/jittery Bink2/Wasm simulation (60-75ms hitches + 180ms stall):
  //    - 60-75ms hitches are absorbed cleanly with 0 underruns once prebuffered.
  //    - Extreme 180ms stall fades out click-free, holds silence during hysteresis refill
  //      (no stop-start oscillation), and fades back in click-free!
  {
    const { AudioMixerWorker } = require('../web/workers/audio_worker.js');
    let fakeNow = 1000;
    const timers = [];
    const worker = new AudioMixerWorker(() => {}, {
      now: () => fakeNow,
      autoDrain: true,
      declickFrames: 0,
      setIntervalFn: (fn, ms) => {
        const h = { fn, ms, cleared: false };
        timers.push(h);
        return h;
      },
      clearIntervalFn: (h) => {
        if (h) h.cleared = true;
      },
    });

    const env = createMockWebAudioEnv({
      initialState: 'running',
      actualSampleRate: 48000,
      attachRealProcessor: true,
    });
    const engine = new AudioEngine({
      audioWorker: worker,
      AudioContextCtor: env.MockAudioContext,
      AudioWorkletNodeCtor: env.MockAudioWorkletNode,
      listenTelemetryBroadcast: false,
    });
    await engine.start();
    const proc = env.events.nodes[0]._processor;

    worker.handleMessage({
      type: 'init',
      sampleRate: 48000,
      channels: 2,
      ringSab: engine.sab,
    });

    // Build simulated Playgta6AudioRing (65536 stereo frames).
    const capSamples = 131072;
    const guestSab = new SharedArrayBuffer(64 + capSamples * 4);
    const readCtrl = new Int32Array(guestSab, 0, 1);
    const writeCtrl = new Int32Array(guestSab, 4, 1);
    const guestData = new Float32Array(guestSab, 64, capSamples);
    const mask = capSamples - 1;
    let phaseFrame = 0;

    function pushGuestFrames(frameCount) {
      const write = Atomics.load(writeCtrl, 0) >>> 0;
      for (let f = 0; f < frameCount; f++) {
        const sample = 0.6;
        guestData[(write + f * 2) & mask] = sample;
        guestData[(write + f * 2 + 1) & mask] = -sample;
        phaseFrame++;
      }
      Atomics.store(writeCtrl, 0, ((write + frameCount * 2) >>> 0) | 0);
    }

    worker.handleMessage({
      type: 'attach-guest-pcm-ring',
      sab: guestSab,
      dataByteOffset: 64,
      capacitySamples: capSamples,
      channels: 2,
      sampleRate: 48000,
      readIndexByteOffset: 0,
      writeIndexByteOffset: 4,
      autoPump: true,
    });
    assert.equal(timers[timers.length - 1].ms, 8, 'active guest pump should run at 8ms cadence');

    // Initial startup: Chrome runs 2 empty quanta before Wasm writes -> arms hysteresis prebuffer.
    const out = makePlanarOutput(2, 128);
    proc.process([], out);
    proc.process([], out);
    assert.equal(proc.inRebuffer, true);

    // Wasm writes 90ms (4320 frames) -> single 8ms worker tick drains entire burst -> worklet exits prebuffer!
    pushGuestFrames(4320);
    fakeNow += 8;
    timers[timers.length - 1].fn();
    assert.equal(engine.availableFrames(), 4320);

    // Drain 4 quanta to finish the 160-frame fade-in and reach steady-state 0.6/-0.6.
    const underrunsBeforeHitches = proc.underruns;
    for (let q = 0; q < 4; q++) {
      proc.process([], out);
    }
    assert.equal(proc.inRebuffer, false);
    assert.equal(out[0][0][127], Math.fround(0.6));

    // Simulate 5 consecutive 64ms (3072-frame) Bink2/Wasm frame hitches followed by 64ms catch-up bursts!
    for (let hitch = 0; hitch < 5; hitch++) {
      // Worklet consumes 64ms = 24 quanta while Wasm is hitched (0 new frames produced).
      for (let q = 0; q < 24; q++) {
        proc.process([], out);
        assert.equal(out[0][0][0], Math.fround(0.6), `hitch ${hitch} q=${q} should stay uninterrupted`);
      }
      // Wasm unfreezes and bursts 64ms (3072 frames); worker pumps full burst on next 8ms tick.
      pushGuestFrames(3072);
      fakeNow += 64;
      timers[timers.length - 1].fn();
    }
    assert.equal(
      proc.underruns,
      underrunsBeforeHitches,
      '64ms Bink2/Wasm hitches must be completely absorbed with 0 underruns'
    );

    // Now simulate an extreme 150ms stall that drains the entire cushion -> must fade out click-free
    // and then hold silence cleanly during small trickles until 80ms (3840 frames) refills!
    let prevL = out[0][0][127];
    const maxAllowedStep = (Math.PI / (2 * 160)) * 0.6 + 1e-3;
    for (let q = 0; q < 40; q++) {
      proc.process([], out);
      for (let f = 0; f < 128; f++) {
        const curL = out[0][0][f];
        assert.ok(
          Math.abs(curL - prevL) <= maxAllowedStep,
          `Click detected during stall fade-out at q=${q} f=${f}: ${prevL} -> ${curL}`
        );
        prevL = curL;
      }
    }
    assert.equal(prevL, 0.0);
    assert.equal(proc.inUnderrun, true);
    assert.equal(proc.fadeOutCount, 1);

    // Feed 5 small 512-frame trickles (2560 frames < 3840 prebuffer): worklet MUST stay silent
    // (no stop-start oscillation!).
    const fadeInsBeforeTrickles = proc.fadeInCount;
    for (let t = 0; t < 5; t++) {
      pushGuestFrames(512);
      fakeNow += 8;
      timers[timers.length - 1].fn();
      proc.process([], out);
      assert.ok(out[0][0].every((v) => v === 0.0), `Trickle ${t} must stay held in hysteresis silence`);
      assert.equal(proc.inRebuffer, true);
    }
    assert.equal(proc.fadeInCount, fadeInsBeforeTrickles, 'Must not oscillate fade-in on tiny trickles');

    // Feed remaining 1280 frames -> total 3840 (>= 80ms prebuffer) -> resumes ONCE with click-free fade-in!
    pushGuestFrames(1280);
    fakeNow += 8;
    timers[timers.length - 1].fn();
    prevL = 0.0;
    for (let q = 0; q < 3; q++) {
      proc.process([], out);
      for (let f = 0; f < 128; f++) {
        const curL = out[0][0][f];
        assert.ok(
          Math.abs(curL - prevL) <= maxAllowedStep,
          `Click detected during recovery fade-in at q=${q} f=${f}: ${prevL} -> ${curL}`
        );
        prevL = curL;
      }
    }
    assert.equal(proc.inRebuffer, false);
    assert.equal(proc.fadeInCount, fadeInsBeforeTrickles + 1);
    assert.equal(out[0][0][127], Math.fround(0.6));

    worker.destroy();
    await engine.destroy();
  }
});

test('Orch-07 (#663/#1028): AudioEngine enables worklet softLimit, 4096-frame drift guard, smooth suspend/resume resync, and cursor normalization', async () => {
  // 1. Default processorOptions include softLimit: true & softLimitKnee: 0.9, and 4096-frame ring gets proportional drift guard.
  {
    const env = createMockWebAudioEnv({ initialState: 'running', actualSampleRate: 48000 });
    const engine = new AudioEngine({
      capacityFrames: 4096,
      AudioContextCtor: env.MockAudioContext,
      AudioWorkletNodeCtor: env.MockAudioWorkletNode,
      listenTelemetryBroadcast: false,
    });
    const res = await engine.start();
    assert.equal(res.ok, true);
    const opts = env.events.nodes[0].options.processorOptions;
    assert.equal(opts.softLimit, true);
    assert.equal(opts.softLimitKnee, 0.9);
    assert.ok(opts.maxBufferedFrames > 0 && opts.maxBufferedFrames < 4096);
    assert.ok(opts.targetWatermarkFrames > 0 && opts.targetWatermarkFrames < opts.maxBufferedFrames);
    await engine.destroy();
  }

  // 2. suspend() ramps masterGainNode to 0.0 and holds silent; resume() fades back in and forwards resyncBacklog: true.
  {
    const workerMessages = [];
    const ramps = [];
    const env = createMockWebAudioEnv({ initialState: 'running', actualSampleRate: 48000 });
    class GainMockAudioContext extends env.MockAudioContext {
      constructor(ctxOpts) {
        super(ctxOpts);
        this.currentTime = 1.0;
      }
      createGain() {
        const gainParam = {
          value: 1.0,
          cancelScheduledValues() {},
          setValueAtTime(v) {
            this.value = v;
          },
          linearRampToValueAtTime(v, t) {
            ramps.push({ target: v, endTime: t });
            this.value = v;
          },
        };
        return {
          gain: gainParam,
          connect() {},
          disconnect() {},
        };
      }
      async suspend() {
        this.state = 'suspended';
        this._emit('statechange');
      }
    }
    const engine = new AudioEngine({
      AudioContextCtor: GainMockAudioContext,
      AudioWorkletNodeCtor: env.MockAudioWorkletNode,
      audioWorker: {
        postMessage(msg) {
          workerMessages.push(msg);
        },
      },
      listenTelemetryBroadcast: false,
    });
    await engine.start();
    assert.equal(await engine.suspend(), true);
    assert.equal(engine._masterGainHeldSilent, true);
    assert.ok(ramps.some((r) => r.target === 0.0), 'suspend must ramp masterGainNode to 0.0');
    assert.equal(await engine.resume(), true);
    assert.equal(engine._masterGainHeldSilent, false);
    assert.ok(ramps.some((r) => r.target === 1.0), 'resume must ramp masterGainNode back to target volume');
    const activeMsg = workerMessages.find(
      (m) => m && m.setOutputState === true && m.outputState === 'active'
    );
    assert.ok(activeMsg, 'resume must forward active state to audio worker');
    assert.equal(activeMsg.resyncBacklog, true);
    await engine.destroy();
  }

  // 3. Corrupted out-of-range SAB cursors are normalized modulo 2*cap and unsigned telemetry counters > 2^31 stay non-negative.
  {
    const env = createMockWebAudioEnv({ initialState: 'running', actualSampleRate: 48000 });
    const engine = new AudioEngine({
      capacityFrames: 256,
      AudioContextCtor: env.MockAudioContext,
      AudioWorkletNodeCtor: env.MockAudioWorkletNode,
      listenTelemetryBroadcast: false,
    });
    await engine.start();
    const ctrl = new Int32Array(engine.sab, 0, 8);
    // Out-of-range write cursor (512 + 40 = 552) and negative read cursor (-502 => 10 mod 512) => 30 buffered frames.
    Atomics.store(ctrl, 0, 552);
    Atomics.store(ctrl, 1, -502);
    assert.equal(engine.availableFrames(), 30);

    const pulled = new Float32Array(10 * 2);
    const pullRes = engine.pullPcm(pulled);
    assert.equal(pullRes.ok, true);
    assert.equal(pullRes.framesRead, 10);
    assert.equal(pullRes.bufferedFrames, 20);

    // Unsigned Int32 counter wraparound (> 2^31) reads cleanly as positive integer.
    Atomics.store(ctrl, HDR_UNDERRUN_FRAMES, 0x80000100 | 0);
    Atomics.store(ctrl, HDR_DROPPED_FRAMES, 0x80000200 | 0);
    const stats = engine.getTelemetry();
    assert.equal(stats.underrunFrames, 0x80000100);
    assert.equal(stats.droppedFrames, 0x80000200);
    await engine.destroy();
  }
});
