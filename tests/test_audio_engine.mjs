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
