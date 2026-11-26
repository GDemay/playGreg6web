import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import {
  AudioEngine,
  DEFAULT_WORKLET_URL,
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
  createAudioRingSab,
} from '../web/audio/audio_engine.js';
import { bootstrapPlayGta6App } from '../web/app_bootstrap.js';

const require = createRequire(import.meta.url);
const audioWorkerMod = require('../web/workers/audio_worker.js');
const workletMod = require('../web/workers/audio_worklet_processor.js');

const {
  AudioRingBuffer,
  AudioMixerWorker,
  GuestPcmRingReader,
  OUTPUT_STATES,
} = audioWorkerMod;
const { PlayGta6AudioWorkletProcessor } = workletMod;

/**
 * Allocates a SharedArrayBuffer matching C++ `Playgta6AudioRing`
 * (`runtime/os/audio_downmix.h`): free-running uint32 float-sample cursors,
 * power-of-two float sample capacity, interleaved 48 kHz stereo float32 PCM.
 */
function createSimulatedGuestPcmRing({
  capacitySamples = 32768,
  sampleRate = 48000,
  channels = 2,
} = {}) {
  const readIndexByteOffset = 0;
  const writeIndexByteOffset = 4;
  const dataByteOffset = 64;
  const totalBytes = dataByteOffset + capacitySamples * Float32Array.BYTES_PER_ELEMENT;
  const sab = new SharedArrayBuffer(totalBytes);
  const readCtrl = new Int32Array(sab, readIndexByteOffset, 1);
  const writeCtrl = new Int32Array(sab, writeIndexByteOffset, 1);
  const data = new Float32Array(sab, dataByteOffset, capacitySamples);
  const mask = capacitySamples - 1;

  function producerPushStereo(interleavedSamples) {
    const read = Atomics.load(readCtrl, 0) >>> 0;
    const write = Atomics.load(writeCtrl, 0) >>> 0;
    const used = (write - read) >>> 0;
    const free = capacitySamples > used ? capacitySamples - used : 0;
    const count = Math.min(interleavedSamples.length, free);
    for (let i = 0; i < count; i++) {
      data[(write + i) & mask] = interleavedSamples[i];
    }
    Atomics.store(writeCtrl, 0, ((write + count) >>> 0) | 0);
    return count;
  }

  return {
    sab,
    readCtrl,
    writeCtrl,
    data,
    producerPushStereo,
    descriptor: {
      sab,
      buffer: sab,
      base: dataByteOffset,
      dataByteOffset,
      capacity: capacitySamples,
      capacitySamples,
      capacityFrames: capacitySamples >>> 1,
      channels,
      sampleRate,
      readIndexPtr: readIndexByteOffset,
      readIndexByteOffset,
      writeIndexPtr: writeIndexByteOffset,
      writeIndexByteOffset,
    },
  };
}

function makeStereoSineWave(frameCount, {
  sampleRate = 48000,
  freqL = 440,
  freqR = 880,
  ampL = 0.6,
  ampR = 0.4,
  startFrame = 0,
} = {}) {
  const out = new Float32Array(frameCount * 2);
  const twoPi = Math.PI * 2;
  for (let f = 0; f < frameCount; f++) {
    const t = (startFrame + f) / sampleRate;
    out[f * 2] = ampL * Math.sin(twoPi * freqL * t);
    out[f * 2 + 1] = ampR * Math.sin(twoPi * freqR * t);
  }
  return out;
}

function computeRms(samples) {
  let sumSq = 0;
  for (let i = 0; i < samples.length; i++) {
    sumSq += samples[i] * samples[i];
  }
  return samples.length > 0 ? Math.sqrt(sumSq / samples.length) : 0;
}

/**
 * Mock WebAudio host that instantiates the REAL `PlayGta6AudioWorkletProcessor`
 * and tracks the full `AudioWorkletNode -> GainNode -> AudioContext.destination` graph.
 */
function createRealWorkletWebAudioHost({
  initialState = 'suspended',
  actualSampleRate = 48000,
  failResumeCount = 0,
} = {}) {
  let remainingResumeFailures = failResumeCount;
  const hostState = {
    contexts: [],
    gainNodes: [],
    workletNodes: [],
    processors: [],
    addModuleUrls: [],
    connections: [],
  };

  class MockAudioContext {
    constructor(opts = {}) {
      this.requestedOptions = opts;
      this.sampleRate = actualSampleRate;
      this.state = initialState;
      this.currentTime = 0;
      this.destination = { kind: 'destination' };
      this._listeners = new Map();
      this.audioWorklet = {
        addModule: async (url) => {
          hostState.addModuleUrls.push(url);
        },
      };
      hostState.contexts.push(this);
    }

    addEventListener(type, fn) {
      if (!this._listeners.has(type)) this._listeners.set(type, new Set());
      this._listeners.get(type).add(fn);
    }

    removeEventListener(type, fn) {
      const set = this._listeners.get(type);
      if (set) set.delete(fn);
    }

    emitStateChange(nextState) {
      if (nextState !== undefined) {
        this.state = nextState;
      }
      const set = this._listeners.get('statechange');
      if (set) {
        for (const fn of [...set]) fn({ type: 'statechange' });
      }
      if (typeof this.onstatechange === 'function') {
        this.onstatechange({ type: 'statechange' });
      }
    }

    createGain() {
      const ops = [];
      const gainParam = {
        value: 1.0,
        cancelScheduledValues(t) {
          ops.push({ op: 'cancel', time: t });
        },
        setValueAtTime(v, t) {
          this.value = v;
          ops.push({ op: 'set', value: v, time: t });
        },
        linearRampToValueAtTime(v, endTime) {
          this.value = v;
          ops.push({ op: 'linearRamp', value: v, endTime });
        },
      };
      const node = {
        kind: 'gain',
        gain: gainParam,
        ops,
        connectedTo: [],
        connect(dest) {
          this.connectedTo.push(dest);
          hostState.connections.push({ from: this, to: dest });
          return dest;
        },
        disconnect() {
          this.connectedTo.length = 0;
        },
      };
      hostState.gainNodes.push(node);
      return node;
    }

    async resume() {
      if (remainingResumeFailures > 0) {
        remainingResumeFailures -= 1;
        throw new Error('NotAllowedError: user gesture required');
      }
      this.state = 'running';
      this.emitStateChange('running');
    }

    async suspend() {
      this.state = 'suspended';
      this.emitStateChange('suspended');
    }

    async close() {
      this.state = 'closed';
    }
  }

  class MockAudioWorkletNode {
    constructor(context, name, options = {}) {
      this.context = context;
      this.name = name;
      this.options = options;
      this.connectedTo = [];
      this.processor = new PlayGta6AudioWorkletProcessor(options);
      const nodePort = {
        onmessage: null,
        postMessage: (msg) => {
          if (this.processor.port && typeof this.processor.port.onmessage === 'function') {
            this.processor.port.onmessage({ data: msg });
          }
        },
        close() {},
      };
      this.processor.port.postMessage = (reply) => {
        if (typeof nodePort.onmessage === 'function') {
          nodePort.onmessage({ data: reply });
        }
      };
      this.port = nodePort;
      hostState.workletNodes.push(this);
      hostState.processors.push(this.processor);
    }

    connect(dest) {
      this.connectedTo.push(dest);
      hostState.connections.push({ from: this, to: dest });
      return dest;
    }

    disconnect() {
      this.connectedTo.length = 0;
    }
  }

  return { MockAudioContext, MockAudioWorkletNode, hostState };
}

function createFakeEventTarget(extra = {}) {
  const listeners = new Map();
  return {
    ...extra,
    addEventListener(type, fn, opts) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add({ fn, once: Boolean(opts && opts.once) });
    },
    removeEventListener(type, fn) {
      const set = listeners.get(type);
      if (!set) return;
      for (const entry of set) {
        if (entry.fn === fn) set.delete(entry);
      }
    },
    dispatch(type, event = {}) {
      const set = listeners.get(type);
      if (!set) return;
      for (const entry of [...set]) {
        if (entry.once) set.delete(entry);
        entry.fn({ type, ...event });
      }
    },
    listenerCount(type) {
      if (type) return listeners.get(type)?.size ?? 0;
      let total = 0;
      for (const set of listeners.values()) total += set.size;
      return total;
    },
  };
}

test('W370 SAB header offsets and module worklet URL align across AudioEngine, AudioRingBuffer, and PlayGta6AudioWorkletProcessor', () => {
  assert.equal(workletMod.HDR_WRITE_CURSOR, HDR_WRITE_CURSOR);
  assert.equal(workletMod.HDR_READ_CURSOR, HDR_READ_CURSOR);
  assert.equal(workletMod.HDR_CAPACITY_FRAMES, HDR_CAPACITY_FRAMES);
  assert.equal(workletMod.HDR_CHANNELS, HDR_CHANNELS);
  assert.equal(workletMod.HDR_UNDERRUNS, HDR_UNDERRUNS);
  assert.equal(workletMod.HDR_UNDERRUN_FRAMES, HDR_UNDERRUN_FRAMES);
  assert.equal(workletMod.HDR_OVERFLOWS, HDR_OVERFLOWS);
  assert.equal(workletMod.HDR_DROPPED_FRAMES, HDR_DROPPED_FRAMES);
  assert.equal(workletMod.RING_HEADER_INT32_COUNT, RING_HEADER_INT32_COUNT);
  assert.equal(workletMod.RING_HEADER_BYTES, RING_HEADER_BYTES);
  assert.equal(AudioRingBuffer.HEADER_BYTES, RING_HEADER_BYTES);
  assert.equal(AudioRingBuffer.HEADER_INT32_COUNT, RING_HEADER_INT32_COUNT);
  assert.ok(
    DEFAULT_WORKLET_URL.endsWith('/workers/audio_worklet_processor.js'),
    `expected worklet URL to resolve via import.meta.url, got ${DEFAULT_WORKLET_URL}`
  );

  // Verify createAudioRingSab -> AudioRingBuffer -> PlayGta6AudioWorkletProcessor compatibility.
  const sab = createAudioRingSab(512, 2);
  const ring = new AudioRingBuffer(sab);
  assert.equal(ring.capacityFrames, 512);
  assert.equal(ring.channels, 2);
  const proc = new PlayGta6AudioWorkletProcessor({ processorOptions: { sab } });
  assert.equal(proc.getStats().attached, true);
  assert.equal(proc.getStats().capacityFrames, 512);
  assert.equal(proc.getStats().channels, 2);
});

test('W370 end-to-end Playgta6AudioRing -> AudioMixerWorker -> AudioRingBuffer SAB -> PlayGta6AudioWorkletProcessor outputs exact non-zero stereo sine wave', async () => {
  let fakeNow = 1000;
  const timers = [];
  const workerReplies = [];

  const audioWorker = new AudioMixerWorker((msg) => workerReplies.push(msg), {
    now: () => fakeNow,
    autoDrain: true,
    drainIntervalMs: 20,
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

  const webAudio = createRealWorkletWebAudioHost({
    initialState: 'running',
    actualSampleRate: 48000,
  });

  const engine = new AudioEngine({
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 4096,
    volume: 1.0,
    muted: false,
    audioWorker,
    AudioContextCtor: webAudio.MockAudioContext,
    AudioWorkletNodeCtor: webAudio.MockAudioWorkletNode,
    listenTelemetryBroadcast: false,
  });

  const startRes = await engine.start();
  assert.equal(startRes.ok, true);
  assert.equal(startRes.state, 'running');
  assert.equal(webAudio.hostState.addModuleUrls[0], DEFAULT_WORKLET_URL);

  // Verify full WebAudio output graph: workletNode -> masterGainNode -> destination, gain > 0.
  const workletNode = webAudio.hostState.workletNodes[0];
  const gainNode = webAudio.hostState.gainNodes[0];
  const ctx = webAudio.hostState.contexts[0];
  const processor = webAudio.hostState.processors[0];
  assert.deepEqual(workletNode.connectedTo, [gainNode]);
  assert.deepEqual(gainNode.connectedTo, [ctx.destination]);
  assert.equal(gainNode.gain.value, 1.0);

  // Attach AudioEngine's SAB to AudioMixerWorker (simulating Dedicated Worker init with ringSab).
  audioWorker.handleMessage({
    type: 'init',
    sampleRate: 48000,
    channels: 2,
    ringSab: engine.sab,
  });
  assert.equal(audioWorker.outputState, OUTPUT_STATES.ACTIVE);
  assert.equal(workerReplies[0].ready, true);

  // Simulated Playgta6AudioRing with 2048 frames of distinct L (440 Hz @ 0.6) & R (880 Hz @ 0.4) sine waves.
  const guestRing = createSimulatedGuestPcmRing({ capacitySamples: 16384, sampleRate: 48000 });
  const totalFrames = 2048;
  const expectedSine = makeStereoSineWave(totalFrames, {
    sampleRate: 48000,
    freqL: 440,
    freqR: 880,
    ampL: 0.6,
    ampR: 0.4,
  });
  assert.equal(guestRing.producerPushStereo(expectedSine), totalFrames * 2);

  audioWorker.handleMessage({
    type: 'attach-guest-pcm-ring',
    ...guestRing.descriptor,
    autoPump: true,
    id: 1,
  });
  assert.equal(audioWorker.getStats().guestRingAttached, true);
  assert.equal(timers.length, 1);

  // Pump 3 ticks (60 ms -> 2880 frame budget) to transfer all 2048 frames from guest ring into AudioEngine SAB.
  for (let t = 1; t <= 3; t++) {
    fakeNow = 1000 + t * 20;
    timers[0].fn();
  }

  // Float-sample read cursor on Playgta6AudioRing must advance by exactly 2048 * 2 = 4096 floats.
  assert.equal(Atomics.load(guestRing.readCtrl, 0) >>> 0, totalFrames * 2);
  assert.equal(audioWorker.getStats().guestFramesDrained, totalFrames);
  assert.equal(engine.availableFrames(), totalFrames);

  // Pull 16 quanta x 128 frames = 2048 frames via REAL PlayGta6AudioWorkletProcessor.process().
  const outL = new Float32Array(totalFrames);
  const outR = new Float32Array(totalFrames);
  const qL = new Float32Array(128);
  const qR = new Float32Array(128);
  for (let q = 0; q < totalFrames / 128; q++) {
    const alive = processor.process([], [[qL, qR]]);
    assert.equal(alive, true);
    outL.set(qL, q * 128);
    outR.set(qR, q * 128);
  }

  // Exact sample match + channel separation + non-zero RMS verification.
  let maxErrL = 0;
  let maxErrR = 0;
  let maxLrDiff = 0;
  for (let f = 0; f < totalFrames; f++) {
    maxErrL = Math.max(maxErrL, Math.abs(outL[f] - expectedSine[f * 2]));
    maxErrR = Math.max(maxErrR, Math.abs(outR[f] - expectedSine[f * 2 + 1]));
    maxLrDiff = Math.max(maxLrDiff, Math.abs(outL[f] - outR[f]));
  }
  assert.ok(maxErrL < 1e-6, `L channel mismatch: ${maxErrL}`);
  assert.ok(maxErrR < 1e-6, `R channel mismatch: ${maxErrR}`);
  assert.ok(maxLrDiff > 0.5, `L and R channels should be distinct, maxLrDiff=${maxLrDiff}`);

  const rmsL = computeRms(outL);
  const rmsR = computeRms(outR);
  assert.ok(rmsL > 0.35 && rmsL < 0.5, `expected L sine RMS ~0.424, got ${rmsL}`);
  assert.ok(rmsR > 0.22 && rmsR < 0.35, `expected R sine RMS ~0.283, got ${rmsR}`);
  assert.equal(processor.getStats().underruns, 0);
  assert.equal(engine.getStats().overruns, 0);
  assert.equal(engine.availableFrames(), 0);

  audioWorker.destroy();
  await engine.destroy();
});

test('W370 attaching ringSab AFTER loader.js headless { init: true } + attach-guest-pcm-ring switches cleanly to live SAB output without losing guest ring', () => {
  let fakeNow = 2000;
  const timers = [];
  const replies = [];

  const audioWorker = new AudioMixerWorker((msg) => replies.push(msg), {
    now: () => fakeNow,
    autoDrain: true,
    drainIntervalMs: 20,
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

  // 1. Exact loader.js startup: { init: true } without ringSab -> enters headless-drain in Worker.
  audioWorker.handleMessage({ init: true });
  assert.equal(audioWorker.outputState, OUTPUT_STATES.HEADLESS_DRAIN);

  // 2. Guest ring attaches BEFORE AudioEngine ringSab arrives; headless drain drains 960 frames.
  const guestRing = createSimulatedGuestPcmRing({ capacitySamples: 16384, sampleRate: 48000 });
  audioWorker.handleMessage({
    type: 'attach-guest-pcm-ring',
    ...guestRing.descriptor,
    autoPump: true,
    id: 10,
  });
  assert.equal(guestRing.producerPushStereo(makeStereoSineWave(960)), 1920);
  fakeNow = 2020;
  timers[timers.length - 1].fn();
  assert.equal(audioWorker.getStats().guestFramesDrained, 960);
  assert.equal(audioWorker.getStats().framesDrained, 960);

  // 3. Now AudioEngine's ringSab arrives via attach-ring-sab: must switch from headless-drain
  //    to active AND preserve guest ring attachment + guest drain count!
  const workletSab = createAudioRingSab(4096, 2);
  audioWorker.handleMessage({
    type: 'attach-ring-sab',
    ringSab: workletSab,
    sampleRate: 48000,
    id: 11,
  });

  const attachReply = replies.find((r) => r.id === 11);
  assert.equal(attachReply.attached, true);
  assert.equal(attachReply.ringSabAttached, true);
  assert.equal(attachReply.guestRingAttached, true);
  assert.equal(attachReply.outputState, OUTPUT_STATES.ACTIVE);
  assert.equal(audioWorker.outputState, OUTPUT_STATES.ACTIVE);
  assert.equal(audioWorker.getStats().guestRingAttached, true);
  assert.equal(audioWorker.getStats().guestFramesDrained, 960);

  // 4. Subsequent guest frames flow straight into workletSab and play out of PlayGta6AudioWorkletProcessor!
  const nextWave = makeStereoSineWave(512, { startFrame: 960, freqL: 660, freqR: 330, ampL: 0.7, ampR: 0.5 });
  assert.equal(guestRing.producerPushStereo(nextWave), 1024);
  fakeNow = 2040;
  timers[timers.length - 1].fn();

  assert.equal(audioWorker.getStats().guestFramesDrained, 960 + 512);
  assert.equal(audioWorker.getStats().framesDrained, 960, 'active mode must NOT discard live frames');

  const proc = new PlayGta6AudioWorkletProcessor({ processorOptions: { sab: workletSab } });
  const qL = new Float32Array(128);
  const qR = new Float32Array(128);
  assert.equal(proc.process([], [[qL, qR]]), true);
  for (let f = 0; f < 128; f++) {
    assert.ok(Math.abs(qL[f] - nextWave[f * 2]) < 1e-6);
    assert.ok(Math.abs(qR[f] - nextWave[f * 2 + 1]) < 1e-6);
  }
  assert.ok(computeRms(qL) > 0.4);
  assert.ok(computeRms(qR) > 0.25);

  // Verify outputState aliases 'worklet' and 'live' map cleanly to 'active'.
  audioWorker.handleMessage({ type: 'audio-context-state', outputState: 'suspended', id: 12 });
  assert.equal(audioWorker.outputState, OUTPUT_STATES.SUSPENDED);
  audioWorker.handleMessage({ type: 'audio-context-state', outputState: 'worklet', id: 13 });
  assert.equal(audioWorker.outputState, OUTPUT_STATES.ACTIVE);
  audioWorker.handleMessage({ type: 'audio-context-state', outputState: 'suspended', id: 14 });
  audioWorker.handleMessage({ type: 'audio-context-state', outputState: 'live', id: 15 });
  assert.equal(audioWorker.outputState, OUTPUT_STATES.ACTIVE);

  audioWorker.destroy();
});

test('W370 autoplay suspension before user click never deadlocks GuestPcmRingReader and plays fresh audio immediately after unlock', async () => {
  // Case A: Suspended AudioContext during boot -> user click unlock -> immediate audible samples & gain > 0.
  {
    let fakeNow = 3000;
    const timers = [];
    const audioWorker = new AudioMixerWorker(() => {}, {
      now: () => fakeNow,
      autoDrain: true,
      drainIntervalMs: 20,
      declickFrames: 0,
      setIntervalFn: (fn, ms) => {
        const h = { fn, ms };
        timers.push(h);
        return h;
      },
      clearIntervalFn: () => {},
    });

    const webAudio = createRealWorkletWebAudioHost({
      initialState: 'suspended',
      actualSampleRate: 48000,
    });
    const engine = new AudioEngine({
      sampleRate: 48000,
      channels: 2,
      capacityFrames: 4096,
      volume: 0.9,
      unlockFadeInMs: 25,
      audioWorker,
      AudioContextCtor: webAudio.MockAudioContext,
      AudioWorkletNodeCtor: webAudio.MockAudioWorkletNode,
      listenTelemetryBroadcast: false,
    });

    await engine.start();
    const gainNode = webAudio.hostState.gainNodes[0];
    const processor = webAudio.hostState.processors[0];
    assert.equal(engine.state, 'suspended');
    assert.equal(gainNode.gain.value, 0.0, 'suspended context holds gain at 0 before unlock');

    // Worker attaches ringSab in suspended state while user hasn't clicked yet.
    audioWorker.handleMessage({
      type: 'attach-ring-sab',
      ringSab: engine.sab,
      audioContextState: 'suspended',
      sampleRate: 48000,
    });
    assert.equal(audioWorker.outputState, OUTPUT_STATES.SUSPENDED);

    const guestRing = createSimulatedGuestPcmRing({ capacitySamples: 16384, sampleRate: 48000 });
    audioWorker.handleMessage({
      type: 'attach-guest-pcm-ring',
      ...guestRing.descriptor,
      autoPump: true,
    });

    // Simulate 20 ticks (400 ms = 19,200 frames >> 4096 SAB capacity) while suspended before user click.
    for (let tick = 1; tick <= 20; tick++) {
      const batch = makeStereoSineWave(960, { startFrame: (tick - 1) * 960, ampL: 0.25, ampR: 0.25 });
      assert.equal(guestRing.producerPushStereo(batch), 1920);
      fakeNow = 3000 + tick * 20;
      timers[timers.length - 1].fn();
    }
    assert.equal(audioWorker.getStats().guestFramesDrained, 20 * 960);
    assert.equal(audioWorker.getStats().guestOverruns, 0);

    // User clicks! unlockOnGesture() resumes AudioContext, ramps gain up, and switches audioWorker to active!
    const unlockRes = await engine.unlockOnGesture();
    assert.equal(unlockRes.ok, true);
    assert.equal(unlockRes.state, 'running');
    assert.equal(unlockRes.effectiveGain, 0.9);
    assert.equal(gainNode.gain.value, 0.9, 'masterGainNode must ramp to target volume, not stay at 0');
    assert.equal(audioWorker.outputState, OUTPUT_STATES.ACTIVE);

    // Fresh post-click audio flows straight through ringSab -> PlayGta6AudioWorkletProcessor.
    const liveTone = makeStereoSineWave(960, { freqL: 1000, freqR: 500, ampL: 0.8, ampR: 0.6 });
    assert.equal(guestRing.producerPushStereo(liveTone), 1920);
    fakeNow += 20;
    timers[timers.length - 1].fn();

    const qL = new Float32Array(128);
    const qR = new Float32Array(128);
    assert.equal(processor.process([], [[qL, qR]]), true);
    for (let f = 0; f < 128; f++) {
      assert.ok(Math.abs(qL[f] - liveTone[f * 2]) < 1e-6);
      assert.ok(Math.abs(qR[f] - liveTone[f * 2 + 1]) < 1e-6);
    }
    assert.ok(computeRms(qL) > 0.5);

    audioWorker.destroy();
    await engine.destroy();
  }

  // Case B: Active-mode stall recovery guard if AudioWorklet hasn't started consuming yet (ring full while suspended).
  {
    let fakeNow = 5000;
    const timers = [];
    const sab = createAudioRingSab(1024, 2);
    const audioWorker = new AudioMixerWorker(() => {}, {
      now: () => fakeNow,
      autoDrain: true,
      drainIntervalMs: 20,
      declickFrames: 0,
      setIntervalFn: (fn, ms) => {
        const h = { fn, ms };
        timers.push(h);
        return h;
      },
      clearIntervalFn: () => {},
    });
    audioWorker.handleMessage({ type: 'init', ringSab: sab, sampleRate: 48000, channels: 2 });
    assert.equal(audioWorker.outputState, OUTPUT_STATES.ACTIVE);

    const guestRing = createSimulatedGuestPcmRing({ capacitySamples: 16384, sampleRate: 48000 });
    audioWorker.handleMessage({
      type: 'attach-guest-pcm-ring',
      ...guestRing.descriptor,
      autoPump: true,
    });

    // Push old 0.2 constant tone to fill the 1024-frame ring completely while consumer is suspended.
    const oldBatch = new Float32Array(1024 * 2).fill(0.2);
    guestRing.producerPushStereo(oldBatch);
    fakeNow = 5030;
    timers[0].fn();
    assert.equal(audioWorker.ring.freeFrames(), 0);

    // Keep feeding fresh 0.75 tone across multiple ticks while consumer stays suspended:
    // stall guard must keep GuestPcmRingReader draining and replace stale audio with fresh 0.75!
    const freshBatch = new Float32Array(960 * 2).fill(0.75);
    for (let i = 1; i <= 4; i++) {
      guestRing.producerPushStereo(freshBatch);
      fakeNow = 5030 + i * 20;
      timers[0].fn();
    }
    assert.ok(
      audioWorker.getStats().guestFramesDrained > 1024,
      'GuestPcmRingReader must NOT freeze when ring is full under suspended consumer'
    );

    // Consumer wakes up and reads from sab -> must hear fresh 0.75 samples, not wedged silence!
    const proc = new PlayGta6AudioWorkletProcessor({ processorOptions: { sab } });
    const outL = new Float32Array(1024);
    const outR = new Float32Array(1024);
    const qL = new Float32Array(128);
    const qR = new Float32Array(128);
    for (let q = 0; q < 8; q++) {
      assert.equal(proc.process([], [[qL, qR]]), true);
      outL.set(qL, q * 128);
      outR.set(qR, q * 128);
    }
    assert.ok(
      outL.some((v) => Math.abs(v - 0.75) < 1e-5),
      'woken worklet must receive fresh post-stall audio samples'
    );
    audioWorker.destroy();
  }

  // Case C: Spontaneous statechange to 'running' un-holds masterGainNode (not stuck at 0.0).
  {
    const webAudio = createRealWorkletWebAudioHost({ initialState: 'suspended' });
    const workerMsgs = [];
    const engine = new AudioEngine({
      volume: 0.75,
      audioWorker: { postMessage: (m) => workerMsgs.push(m) },
      AudioContextCtor: webAudio.MockAudioContext,
      AudioWorkletNodeCtor: webAudio.MockAudioWorkletNode,
      listenTelemetryBroadcast: false,
    });
    await engine.start();
    const gainNode = webAudio.hostState.gainNodes[0];
    assert.equal(gainNode.gain.value, 0.0);

    // Browser/OS spontaneously resumes AudioContext without _pendingAutoRecovery.
    webAudio.hostState.contexts[0].emitStateChange('running');
    assert.equal(gainNode.gain.value, 0.75, 'spontaneous resume must ramp masterGainNode out of 0.0');
    assert.ok(
      workerMsgs.some((m) => m.type === 'audio-context-state' && m.outputState === 'active'),
      'spontaneous resume must notify audioWorker'
    );
    await engine.destroy();
  }

  // Case D: Lazy start inside gesture where AudioContext starts already 'running' still notifies audioWorker on first unlockOnGesture().
  {
    const webAudio = createRealWorkletWebAudioHost({
      initialState: 'running',
      actualSampleRate: 44100,
    });
    const workerMsgs = [];
    const engine = new AudioEngine({
      sampleRate: 48000,
      audioWorker: { postMessage: (m) => workerMsgs.push(m) },
      AudioContextCtor: webAudio.MockAudioContext,
      AudioWorkletNodeCtor: webAudio.MockAudioWorkletNode,
      listenTelemetryBroadcast: false,
    });
    await engine.start();
    const res = await engine.unlockOnGesture();
    assert.equal(res.ok, true);
    assert.equal(res.alreadyRunning, true);
    assert.ok(
      workerMsgs.some(
        (m) =>
          m.type === 'audio-context-state' &&
          m.outputState === 'active' &&
          m.sampleRate === 44100 &&
          m.unlockedOnGesture === true
      ),
      'first unlockOnGesture() on already-running context must still forward active state + actual sampleRate'
    );
    await engine.destroy();
  }
});

test('W370 sample-rate mismatch (44100 Hz and 96000 Hz host AudioContext vs 48000 Hz guest PCM) syncs and resamples cleanly', async () => {
  for (const hostRate of [44100, 96000]) {
    let fakeNow = 7000;
    const timers = [];
    const audioWorker = new AudioMixerWorker(() => {}, {
      now: () => fakeNow,
      autoDrain: true,
      drainIntervalMs: 20,
      declickFrames: 0,
      setIntervalFn: (fn, ms) => {
        const h = { fn, ms };
        timers.push(h);
        return h;
      },
      clearIntervalFn: () => {},
    });

    const webAudio = createRealWorkletWebAudioHost({
      initialState: 'suspended',
      actualSampleRate: hostRate,
    });
    const engine = new AudioEngine({
      sampleRate: 48000,
      channels: 2,
      capacityFrames: 8192,
      audioWorker,
      AudioContextCtor: webAudio.MockAudioContext,
      AudioWorkletNodeCtor: webAudio.MockAudioWorkletNode,
      listenTelemetryBroadcast: false,
    });

    await engine.start();
    assert.equal(engine.sampleRate, hostRate);

    // Worker starts at default 48000 Hz, then receives host AudioContext unlock with actual hostRate!
    audioWorker.handleMessage({
      type: 'attach-ring-sab',
      ringSab: engine.sab,
      audioContextState: 'suspended',
    });
    const guestRing = createSimulatedGuestPcmRing({ capacitySamples: 16384, sampleRate: 48000 });
    audioWorker.handleMessage({
      type: 'attach-guest-pcm-ring',
      ...guestRing.descriptor,
      autoPump: true,
    });

    const unlockRes = await engine.unlockOnGesture();
    assert.equal(unlockRes.ok, true);
    assert.equal(audioWorker.outputState, OUTPUT_STATES.ACTIVE);
    assert.equal(audioWorker.sampleRate, hostRate, `worker must sync sampleRate to ${hostRate}`);

    // Push 20 ms (960 frames @ 48 kHz) stereo sine wave and pump 1 tick.
    const guestWave = makeStereoSineWave(960, { sampleRate: 48000, freqL: 440, freqR: 660, ampL: 0.6, ampR: 0.5 });
    assert.equal(guestRing.producerPushStereo(guestWave), 1920);
    fakeNow = 7020;
    timers[timers.length - 1].fn();

    const expectedOutFrames = Math.round(960 * (hostRate / 48000));
    const buffered = engine.availableFrames();
    assert.ok(
      Math.abs(buffered - expectedOutFrames) <= 4,
      `for hostRate=${hostRate}, expected ~${expectedOutFrames} resampled frames, got ${buffered}`
    );
    assert.equal(audioWorker.getStats().guestOverruns, 0);
    assert.ok(audioWorker.getStats().resampledFrames >= expectedOutFrames - 4);

    // Pull resampled frames via PlayGta6AudioWorkletProcessor and confirm non-zero finite L/R output.
    const processor = webAudio.hostState.processors[0];
    const qL = new Float32Array(128);
    const qR = new Float32Array(128);
    assert.equal(processor.process([], [[qL, qR]]), true);
    assert.ok(computeRms(qL) > 0.3, `resampled L RMS too low for ${hostRate}`);
    assert.ok(computeRms(qR) > 0.25, `resampled R RMS too low for ${hostRate}`);
    for (let i = 0; i < 128; i++) {
      assert.ok(Number.isFinite(qL[i]) && Math.abs(qL[i]) <= 1.0);
      assert.ok(Number.isFinite(qR[i]) && Math.abs(qR[i]) <= 1.0);
    }

    audioWorker.destroy();
    await engine.destroy();
  }
});

test('W370 app_bootstrap gesture unlock retries after failed first resume, supports click/keydown, and forwards state + ringSab to workers', async () => {
  const webAudio = createRealWorkletWebAudioHost({
    initialState: 'suspended',
    actualSampleRate: 48000,
    failResumeCount: 1, // First gesture fails (e.g. activation expired during addModule), second succeeds!
  });

  const windowObj = createFakeEventTarget({
    crossOriginIsolated: true,
    innerWidth: 1280,
    innerHeight: 720,
    devicePixelRatio: 1,
    location: { origin: 'http://127.0.0.1:8080', search: '' },
    AudioContext: webAudio.MockAudioContext,
    AudioWorkletNode: webAudio.MockAudioWorkletNode,
  });
  const documentObj = createFakeEventTarget({ pointerLockElement: null });
  const canvas = createFakeEventTarget({
    width: 1280,
    height: 720,
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 1280, height: 720 }),
    getContext: () => null,
  });

  const loaderMessages = [];
  const mockLoaderWorker = createFakeEventTarget({
    postMessage(payload) {
      loaderMessages.push(payload);
    },
    terminate() {},
  });

  const audioWorkerMessages = [];
  const mockAudioWorker = createFakeEventTarget({
    postMessage(payload) {
      audioWorkerMessages.push(payload);
    },
  });

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj: { gpu: {}, getGamepads: () => [] },
    fetchImpl: async () => ({
      ok: true,
      status: 200,
      async json() {
        return { ok: true, coopCoep: true, hasWasmBinary: true };
      },
    }),
    canvas,
    audioWorker: mockAudioWorker,
    createWasmWorker: () => mockLoaderWorker,
  });

  try {
    await app.start();
    assert.equal(loaderMessages.length, 1);
    assert.equal(loaderMessages[0].audioSab, app.audioEngine.sab);

    // 1. First gesture (pointerdown): lazy starts AudioEngine, resume fails once -> MUST re-arm listeners!
    windowObj.dispatch('pointerdown');
    await new Promise((resolve) => setTimeout(resolve, 10));
    assert.equal(app.audioEngine.state, 'suspended');
    assert.equal(app.audioEngine.getStats().unlockedOnGesture, false);
    assert.ok(windowObj.listenerCount('click') >= 1, 'gesture listeners must re-arm after failed unlock');
    assert.ok(windowObj.listenerCount('keydown') >= 1, 'keydown listener must remain armed after failed unlock');

    // 2. Second gesture (click): unlocks AudioContext synchronously, ramps gain to 1.0, notifies both workers!
    windowObj.dispatch('click');
    await new Promise((resolve) => setTimeout(resolve, 10));
    assert.equal(app.audioEngine.state, 'running');
    assert.equal(app.audioEngine.getStats().unlockedOnGesture, true);
    assert.equal(webAudio.hostState.gainNodes[0].gain.value, 1.0);
    assert.equal(windowObj.listenerCount('click'), 0);
    assert.equal(windowObj.listenerCount('pointerdown'), 0);

    assert.ok(
      audioWorkerMessages.some(
        (m) => m.type === 'audio-context-state' && m.outputState === 'active' && m.sampleRate === 48000
      ),
      'app_bootstrap must wire audioWorker to AudioEngine for active state forwarding'
    );
    assert.ok(
      loaderMessages.some(
        (m) =>
          m.type === 'audio-context-state' &&
          m.outputState === 'active' &&
          m.sampleRate === 48000 &&
          m.ringSab === app.audioEngine.sab
      ),
      'app_bootstrap must notify loaderWorker with active state, sampleRate, and ringSab on unlock'
    );
  } finally {
    await app.destroy();
  }

  // Also verify PlayGta6AudioWorkletProcessor honors explicit rampFrames on setGain fallback messages.
  const sab = createAudioRingSab(512, 2);
  const ring = new AudioRingBuffer(sab);
  ring.pushSamples(new Float32Array(256 * 2).fill(1.0));
  const proc = new PlayGta6AudioWorkletProcessor({
    processorOptions: { sab, gain: 0.0, smoothingFrames: 64 },
  });
  proc.port.onmessage({ data: { type: 'setGain', gain: 1.0, rampFrames: 256 } });
  const outL = new Float32Array(128);
  const outR = new Float32Array(128);
  proc.process([], [[outL, outR]]);
  // Halfway through 256-frame ramp at frame 127, gain should be ~0.5 (not already 1.0 as with 64 frames).
  assert.ok(outL[127] > 0.45 && outL[127] < 0.55, `expected ~0.5 mid-ramp with rampFrames=256, got ${outL[127]}`);
  proc.process([], [[outL, outR]]);
  assert.ok(Math.abs(outL[127] - 1.0) < 1e-5, `expected 1.0 at end of 256-frame ramp, got ${outL[127]}`);
});
