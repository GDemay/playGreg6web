import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

import {
  AudioEngine,
  createAudioRingSab,
} from '../web/audio/audio_engine.js';
import { bootstrapPlayGta6App } from '../web/app_bootstrap.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');
const loaderPath = path.join(repoRoot, 'web', 'loader.js');
const loaderSource = fs.readFileSync(loaderPath, 'utf8');

const require = createRequire(import.meta.url);
const loaderApi = require('../web/loader.js');
const audioWorkerMod = require('../web/workers/audio_worker.js');
const workletMod = require('../web/workers/audio_worklet_processor.js');

const {
  normalizeAudioContextStateValue,
  extractAudioRingSab,
  isAudioContextStateMessage,
  isAudioRingSabMessage,
  createAudioWorkerInitPayload,
  createAudioWorkerBridge,
} = loaderApi;

const {
  AudioMixerWorker,
  OUTPUT_STATES,
} = audioWorkerMod;

const { PlayGta6AudioWorkletProcessor } = workletMod;

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
    for (let i = 0; i < count; i += 1) {
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
} = {}) {
  const out = new Float32Array(frameCount * 2);
  const twoPi = Math.PI * 2;
  for (let f = 0; f < frameCount; f += 1) {
    const t = f / sampleRate;
    out[f * 2] = ampL * Math.sin(twoPi * freqL * t);
    out[f * 2 + 1] = ampR * Math.sin(twoPi * freqR * t);
  }
  return out;
}

function computeRms(samples) {
  let sumSq = 0;
  for (let i = 0; i < samples.length; i += 1) {
    sumSq += samples[i] * samples[i];
  }
  return samples.length > 0 ? Math.sqrt(sumSq / samples.length) : 0;
}

function makeMinimalWasmBytes() {
  return new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
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
  };
}

/**
 * Loads `web/loader.js` inside an isolated VM sandbox mimicking a DedicatedWorker
 * global scope. Optionally backs `./workers/audio_worker.js` with a REAL
 * `AudioMixerWorker` instance for full end-to-end speaker/worklet verification.
 */
function bootLoaderVm({
  realAudioWorker = false,
  autoReady = false,
  audioWorkerOpts = null,
} = {}) {
  const mainMessages = [];
  const spawnedWorkers = [];
  const workersByScript = new Map();
  const realAudioWorkers = [];
  const audioTimers = [];
  let fakeNow = 1000;

  class MockBroadcastChannel {
    postMessage() {}
    close() {}
  }

  class MockWorker {
    constructor(script, opts) {
      this.script = script;
      this.opts = opts || null;
      this.onmessage = null;
      this.onerror = null;
      this.posted = [];
      this.terminated = false;
      this.realMixer = null;
      spawnedWorkers.push(this);
      workersByScript.set(script, this);

      if (realAudioWorker && script === './workers/audio_worker.js') {
        const selfWorker = this;
        const mixer = new AudioMixerWorker(
          (reply) => {
            if (typeof selfWorker.onmessage === 'function') {
              selfWorker.onmessage({ data: reply });
            }
          },
          Object.assign(
            {
              now: () => fakeNow,
              autoDrain: true,
              drainIntervalMs: 20,
              declickFrames: 0,
              setIntervalFn: (fn, ms) => {
                const h = { fn, ms, cleared: false };
                audioTimers.push(h);
                return h;
              },
              clearIntervalFn: (h) => {
                if (h) h.cleared = true;
              },
            },
            audioWorkerOpts || null,
          ),
        );
        this.realMixer = mixer;
        realAudioWorkers.push(mixer);
      }
    }

    postMessage(msg, transfer) {
      if (this.terminated) {
        throw new Error('Cannot postMessage to terminated worker');
      }
      this.posted.push({ msg, transfer: transfer || [] });
      if (this.realMixer) {
        this.realMixer.handleMessage(msg);
        return;
      }
      if (autoReady && msg && msg.init) {
        if (typeof this.onmessage === 'function') {
          this.onmessage({ data: { ready: true, summary: 'ok', adapter: 'mock-gpu' } });
        }
      }
    }

    terminate() {
      this.terminated = true;
      if (this.realMixer) {
        this.realMixer.destroy();
      }
    }
  }

  const workerSelf = {
    name: '',
    location: { href: 'http://127.0.0.1:8080/loader.js' },
    Worker: MockWorker,
    onmessage: null,
    postMessage(msg) {
      mainMessages.push(msg);
    },
  };

  const sandbox = {
    self: workerSelf,
    globalThis: null,
    URL,
    BroadcastChannel: MockBroadcastChannel,
    Worker: MockWorker,
    WebAssembly,
    indexedDB: null,
    Uint8Array,
    Int32Array,
    Uint32Array,
    Float32Array,
    DataView,
    ArrayBuffer,
    SharedArrayBuffer,
    Atomics,
    BigInt,
    TextEncoder,
    TextDecoder,
    Map,
    Set,
    WeakMap,
    Promise,
    Error,
    RangeError,
    TypeError,
    Number,
    Math,
    Date,
    Object,
    Array,
    Proxy,
    Reflect,
    JSON,
    setTimeout,
    clearTimeout,
    queueMicrotask,
    fetch: async () => ({ ok: false, status: 404 }),
    importScripts(scriptPath) {
      const resolved = path.resolve(repoRoot, 'web', scriptPath);
      vm.runInContext(fs.readFileSync(resolved, 'utf8'), context, { filename: resolved });
    },
  };
  sandbox.globalThis = sandbox;
  const context = vm.createContext(sandbox);
  vm.runInContext(loaderSource, context, { filename: loaderPath });

  return {
    workerSelf,
    sandbox,
    context,
    api: context.PlayGta6Loader,
    mainMessages,
    spawnedWorkers,
    workersByScript,
    realAudioWorkers,
    audioTimers,
    advanceNow(ms) {
      fakeNow += ms;
      return fakeNow;
    },
    getAudioWorker() {
      return workersByScript.get('./workers/audio_worker.js') || null;
    },
    getGpuWorker() {
      return workersByScript.get('./workers/wgpu_worker.js') || null;
    },
    send(msg) {
      workerSelf.onmessage({ data: msg });
    },
  };
}

function createRealWorkletWebAudioHost({
  initialState = 'suspended',
  actualSampleRate = 48000,
} = {}) {
  const hostState = {
    contexts: [],
    gainNodes: [],
    workletNodes: [],
    processors: [],
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
        addModule: async () => {},
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
      const gainParam = {
        value: 1.0,
        cancelScheduledValues() {},
        setValueAtTime(v) {
          this.value = v;
        },
        linearRampToValueAtTime(v) {
          this.value = v;
        },
      };
      const node = {
        gain: gainParam,
        connect(dest) {
          return dest;
        },
        disconnect() {},
      };
      hostState.gainNodes.push(node);
      return node;
    }

    async resume() {
      this.state = 'running';
      this.emitStateChange('running');
    }

    async suspend() {
      this.state = 'suspended';
      this.emitStateChange('suspended');
    }

    async close() {
      this.state = 'closed';
      this.emitStateChange('closed');
    }
  }

  class MockAudioWorkletNode {
    constructor(context, name, options = {}) {
      this.context = context;
      this.name = name;
      this.options = options;
      this.processor = new PlayGta6AudioWorkletProcessor(options);
      this.port = {
        onmessage: null,
        postMessage: (msg) => {
          if (this.processor.port && typeof this.processor.port.onmessage === 'function') {
            this.processor.port.onmessage({ data: msg });
          }
        },
        close() {},
      };
      hostState.workletNodes.push(this);
      hostState.processors.push(this.processor);
    }

    connect(dest) {
      return dest;
    }

    disconnect() {}
  }

  return { MockAudioContext, MockAudioWorkletNode, hostState };
}

test('W377 pure helpers validate AudioContext states, SAB extraction, GPU-ring isolation, and init payloads', () => {
  assert.equal(normalizeAudioContextStateValue('running'), 'running');
  assert.equal(normalizeAudioContextStateValue('ACTIVE'), 'active');
  assert.equal(normalizeAudioContextStateValue(' suspended '), 'suspended');
  assert.equal(normalizeAudioContextStateValue('paused'), 'suspended');
  assert.equal(normalizeAudioContextStateValue('interrupted'), 'interrupted');
  assert.equal(normalizeAudioContextStateValue('closed'), 'closed');
  assert.equal(normalizeAudioContextStateValue('stopped'), 'closed');
  assert.equal(normalizeAudioContextStateValue('headless-drain'), 'headless-drain');
  assert.equal(normalizeAudioContextStateValue('invalid-state'), null);
  assert.equal(normalizeAudioContextStateValue(null), null);

  const sabA = createAudioRingSab(512, 2);
  const sabB = createAudioRingSab(512, 2);
  const nonSab = new ArrayBuffer(64);

  assert.equal(extractAudioRingSab({ ringSab: sabA, audioSab: sabB }), sabA);
  assert.equal(extractAudioRingSab({ audioSab: sabB }), sabB);
  assert.equal(extractAudioRingSab({ sab: sabA }), null);
  assert.equal(extractAudioRingSab({ sab: sabA }, { allowGenericSab: true }), sabA);
  assert.equal(extractAudioRingSab({ audioSab: nonSab }), null);

  // GPU command ring messages with `ringSab` must NEVER be misclassified as audio ring SAB messages.
  assert.equal(isAudioRingSabMessage({ type: 'attach-ring', ringSab: sabA }), false);
  assert.equal(isAudioRingSabMessage({ attachRing: true, ringSab: sabA }), false);
  assert.equal(isAudioRingSabMessage({ type: 'drain-ring', ringSab: sabA }), false);
  assert.equal(isAudioRingSabMessage({ type: 'attach-ring-sab', ringSab: sabA }), true);
  assert.equal(isAudioRingSabMessage({ audioSab: sabA }), true);
  assert.equal(isAudioContextStateMessage({ type: 'audio-context-state', outputState: 'running' }), true);
  assert.equal(isAudioContextStateMessage({ setOutputState: true, outputState: 'suspended' }), true);
  assert.equal(isAudioContextStateMessage({ type: 'attach-ring-sab', ringSab: sabA }), false);

  // Headless boot payload stays minimal `{ init: true }`.
  assert.deepEqual(createAudioWorkerInitPayload({ canvas: 'c', audioSab: null }), { init: true });

  // Headed boot payload forwards `ringSab` and optional audio init parameters.
  const headedInit = createAudioWorkerInitPayload({
    canvas: 'c',
    audioSab: sabA,
    audioSampleRate: 44100,
    audioChannels: 2,
    audioContextState: 'suspended',
  });
  assert.deepEqual(headedInit, {
    init: true,
    ringSab: sabA,
    sampleRate: 44100,
    channels: 2,
    outputState: 'suspended',
  });
});

test('W377 boot handoff forwards audioSab / ringSab to child audioWorker while preserving headless mode and W371 sharedModules', async () => {
  // 1. Headed boot with audioSab + W371 sharedModules.
  const sab = createAudioRingSab(1024, 2);
  const compiledMod = await WebAssembly.compile(makeMinimalWasmBytes());
  const vmHeaded = bootLoaderVm({ realAudioWorker: true });
  const bootMsg = {
    canvas: { id: 'offscreen-1' },
    width: 1280,
    height: 720,
    dataBase: '/data/',
    audioSab: sab,
    wasmBinary: makeMinimalWasmBytes().buffer,
    sharedModules: [compiledMod],
  };

  vmHeaded.send(bootMsg);
  const audioWorker = vmHeaded.getAudioWorker();
  assert.ok(audioWorker, 'audioWorker must be spawned on boot');
  assert.equal(audioWorker.posted.length, 1);
  assert.equal(audioWorker.posted[0].msg.init, true);
  assert.equal(audioWorker.posted[0].msg.ringSab, sab);
  assert.equal(audioWorker.realMixer.ring.sab, sab, 'AudioMixerWorker must attach the shared AudioEngine SAB');
  assert.equal(audioWorker.realMixer.outputState, OUTPUT_STATES.ACTIVE);
  assert.equal(bootMsg.wasmBinary, null, 'boot msg wasmBinary wire bytes are scrubbed after handoff');

  const status = vmHeaded.api.getLoaderStatus();
  assert.equal(status.audioRingSabAttached, true);
  assert.equal(status.audioContextState, 'active');
  assert.equal(status.sharedModuleCount, 1, 'W371 sharedModules behavior must remain intact');

  // 2. Headless boot (audioSab: null) preserves exact `{ init: true }` and headless-drain mode.
  const vmHeadless = bootLoaderVm({ realAudioWorker: true });
  vmHeadless.send({
    canvas: { id: 'offscreen-2' },
    width: 640,
    height: 360,
    dataBase: '/data/',
    audioSab: null,
  });
  const headlessAudio = vmHeadless.getAudioWorker();
  assert.deepEqual({ ...headlessAudio.posted[0].msg }, { init: true });
  assert.equal(headlessAudio.realMixer.outputState, OUTPUT_STATES.HEADLESS_DRAIN);
  assert.equal(vmHeadless.api.getLoaderStatus().audioRingSabAttached, false);

  // 3. Invalid non-SAB audioSab on boot must not break audioWorker initialization.
  const vmInvalid = bootLoaderVm({ realAudioWorker: true });
  vmInvalid.send({
    canvas: { id: 'offscreen-3' },
    width: 640,
    height: 360,
    dataBase: '/data/',
    audioSab: new ArrayBuffer(128),
  });
  const invalidAudio = vmInvalid.getAudioWorker();
  assert.deepEqual({ ...invalidAudio.posted[0].msg }, { init: true });
  assert.equal(invalidAudio.realMixer.initialized, true);
  assert.equal(invalidAudio.realMixer.outputState, OUTPUT_STATES.HEADLESS_DRAIN);

  vmHeaded.send({ type: 'destroy' });
  vmHeadless.send({ type: 'destroy' });
  vmInvalid.send({ type: 'destroy' });
});

test('W377 forwards standalone attach-ring-sab / audioSab messages with deduplication and GPU-ring isolation', () => {
  const vmEnv = bootLoaderVm({ realAudioWorker: true });
  const sab1 = createAudioRingSab(1024, 2);
  const sab2 = createAudioRingSab(2048, 2);
  const gpuRingSab = new SharedArrayBuffer(4096);

  // Pre-boot GPU attach-ring must NOT be captured by audio bridge.
  vmEnv.send({ type: 'attach-ring', ringSab: gpuRingSab });

  // Boot headless first (no audioSab yet).
  vmEnv.send({
    canvas: { id: 'offscreen' },
    width: 1280,
    height: 720,
    dataBase: '/data/',
  });
  const audioWorker = vmEnv.getAudioWorker();
  const gpuWorker = vmEnv.getGpuWorker();
  assert.deepEqual({ ...audioWorker.posted[0].msg }, { init: true });
  assert.equal(audioWorker.realMixer.outputState, OUTPUT_STATES.HEADLESS_DRAIN);

  // Post-boot GPU attach-ring goes ONLY to gpuWorker.
  vmEnv.send({ type: 'attach-ring', ringSab: gpuRingSab });
  assert.equal(
    gpuWorker.posted.some((p) => p.msg.type === 'attach-ring' && p.msg.ringSab === gpuRingSab),
    true,
  );
  assert.equal(audioWorker.posted.length, 1);

  // Guest PCM ring attaches while audioWorker is in headless-drain.
  const guestRing = createSimulatedGuestPcmRing({ capacitySamples: 8192, sampleRate: 48000 });
  audioWorker.realMixer.handleMessage({
    type: 'attach-guest-pcm-ring',
    ...guestRing.descriptor,
    autoPump: true,
  });

  // Late arrival of audioSab via bare `{ audioSab }` -> forwarded as `attach-ring-sab`!
  vmEnv.send({ audioSab: sab1 });
  assert.equal(audioWorker.posted.length, 2);
  assert.deepEqual({ ...audioWorker.posted[1].msg }, {
    type: 'attach-ring-sab',
    attachRingSab: true,
    ringSab: sab1,
  });
  assert.equal(audioWorker.realMixer.ring.sab, sab1);
  assert.equal(audioWorker.realMixer.outputState, OUTPUT_STATES.ACTIVE);
  assert.equal(audioWorker.realMixer.getStats().guestRingAttached, true, 'attaching ringSab must preserve guest PCM ring');

  // Duplicate identical audioSab without force/id/options is deduplicated.
  vmEnv.send({ type: 'attach-ring-sab', ringSab: sab1 });
  assert.equal(audioWorker.posted.length, 2, 'duplicate attach-ring-sab for same SAB should be deduplicated');

  // Explicit force or new SAB IS forwarded.
  vmEnv.send({ type: 'attach-ring-sab', ringSab: sab1, force: true, id: 7 });
  assert.equal(audioWorker.posted.length, 3);
  assert.equal(audioWorker.posted[2].msg.id, 7);

  vmEnv.send({ type: 'attach-ring-sab', ringSab: sab2, sampleRate: 44100, channels: 2 });
  assert.equal(audioWorker.posted.length, 4);
  assert.equal(audioWorker.realMixer.ring.sab, sab2);
  assert.equal(audioWorker.realMixer.sampleRate, 44100);

  // Invalid non-SAB attach-ring-sab is ignored safely without throwing.
  vmEnv.send({ type: 'attach-ring-sab', ringSab: new ArrayBuffer(64) });
  assert.equal(audioWorker.posted.length, 4);

  vmEnv.send({ type: 'destroy' });
});

test('W377 forwards main-thread audio-context-state transitions (running, suspended, interrupted, closed) to real AudioMixerWorker', () => {
  const vmEnv = bootLoaderVm({ realAudioWorker: true });
  const sab = createAudioRingSab(1024, 2);

  vmEnv.send({
    canvas: { id: 'offscreen' },
    width: 1280,
    height: 720,
    dataBase: '/data/',
    audioSab: sab,
    audioContextState: 'suspended',
  });
  const audioWorker = vmEnv.getAudioWorker();
  assert.equal(audioWorker.realMixer.outputState, OUTPUT_STATES.SUSPENDED);

  // 1. Unlock -> 'running' (with host sampleRate sync).
  vmEnv.send({
    type: 'audio-context-state',
    state: 'running',
    sampleRate: 44100,
    unlockedOnGesture: true,
    ringSab: sab,
  });
  assert.equal(audioWorker.realMixer.outputState, OUTPUT_STATES.ACTIVE);
  assert.equal(audioWorker.realMixer.sampleRate, 44100);
  const runningMsg = audioWorker.posted[audioWorker.posted.length - 1].msg;
  assert.equal(runningMsg.type, 'audio-context-state');
  assert.equal(runningMsg.setOutputState, true);
  assert.equal(runningMsg.outputState, 'running');
  assert.equal(runningMsg.unlockedOnGesture, true);
  assert.equal(runningMsg.sampleRate, 44100);
  assert.equal(runningMsg.ringSab, sab);

  // 2. OS interruption -> 'interrupted' transitions worker to null-drain ('suspended') so ring never stalls.
  vmEnv.send({
    type: 'audio-context-state',
    outputState: 'interrupted',
  });
  assert.equal(audioWorker.realMixer.outputState, OUTPUT_STATES.SUSPENDED);
  const interruptedMsg = audioWorker.posted[audioWorker.posted.length - 1].msg;
  assert.equal(interruptedMsg.outputState, 'interrupted');
  assert.equal(interruptedMsg.interrupted, true);

  // 3. Recovery -> 'active' with recoveredFromInterruption flag.
  vmEnv.send({
    type: 'audio-context-state',
    setOutputState: true,
    outputState: 'active',
    sampleRate: 48000,
    recoveredFromInterruption: true,
  });
  assert.equal(audioWorker.realMixer.outputState, OUTPUT_STATES.ACTIVE);
  assert.equal(audioWorker.realMixer.sampleRate, 48000);
  const recoveredMsg = audioWorker.posted[audioWorker.posted.length - 1].msg;
  assert.equal(recoveredMsg.recoveredFromInterruption, true);

  // 4. Explicit suspend -> 'suspended'.
  vmEnv.send({
    type: 'audio-context-state',
    outputState: 'suspended',
  });
  assert.equal(audioWorker.realMixer.outputState, OUTPUT_STATES.SUSPENDED);

  // 5. Close -> 'closed' transitions worker back to 'headless-drain' and clears retained SAB.
  vmEnv.send({
    type: 'audio-context-state',
    outputState: 'closed',
    ringSab: sab,
  });
  assert.equal(audioWorker.realMixer.outputState, OUTPUT_STATES.HEADLESS_DRAIN);
  const closedMsg = audioWorker.posted[audioWorker.posted.length - 1].msg;
  assert.equal(closedMsg.outputState, 'closed');
  assert.equal('ringSab' in closedMsg, false, 'closed state must not re-attach a closing SAB');
  assert.equal(vmEnv.api.getLoaderStatus().audioRingSabAttached, false);
  assert.equal(vmEnv.api.getLoaderStatus().audioContextState, 'closed');

  vmEnv.send({ type: 'destroy' });
});

test('W377 late-arrival ordering: pre-boot attach-ring-sab and audio-context-state coalesce and flush on boot or onWorkerReady', () => {
  // Case A: Pre-boot audio-context-state (suspended -> running) + attach-ring-sab arrive BEFORE msg.canvas.
  {
    const vmEnv = bootLoaderVm({ realAudioWorker: true });
    const sab = createAudioRingSab(1024, 2);

    vmEnv.send({ type: 'audio-context-state', outputState: 'suspended', sampleRate: 48000 });
    vmEnv.send({ type: 'attach-ring-sab', ringSab: sab });
    vmEnv.send({
      type: 'audio-context-state',
      outputState: 'running',
      sampleRate: 44100,
      unlockedOnGesture: true,
      ringSab: sab,
    });
    assert.equal(vmEnv.getAudioWorker(), null, 'no audioWorker exists before msg.canvas');

    // Boot WITHOUT audioSab on msg.canvas -> pre-boot ringSab and latest state flush cleanly!
    vmEnv.send({
      canvas: { id: 'offscreen' },
      width: 1280,
      height: 720,
      dataBase: '/data/',
    });
    const audioWorker = vmEnv.getAudioWorker();
    assert.ok(audioWorker);
    assert.equal(audioWorker.posted[0].msg.init, true);
    assert.equal(audioWorker.posted[0].msg.ringSab, sab);
    assert.equal(audioWorker.posted[0].msg.outputState, 'running');
    assert.equal(audioWorker.posted[0].msg.sampleRate, 44100);
    // Coalesced latest audio-context-state is flushed right after init (not the stale 'suspended').
    assert.equal(audioWorker.posted.length, 2);
    assert.equal(audioWorker.posted[1].msg.type, 'audio-context-state');
    assert.equal(audioWorker.posted[1].msg.outputState, 'running');
    assert.equal(audioWorker.posted[1].msg.unlockedOnGesture, true);
    assert.equal(audioWorker.realMixer.ring.sab, sab);
    assert.equal(audioWorker.realMixer.outputState, OUTPUT_STATES.ACTIVE);
    assert.equal(audioWorker.realMixer.sampleRate, 44100);

    vmEnv.send({ type: 'destroy' });
  }

  // Case B: Pre-boot audio-context-state carried sabOld, then boot msg.canvas brings fresher sabNew -> sabNew wins!
  {
    const vmEnv = bootLoaderVm({ realAudioWorker: true });
    const sabOld = createAudioRingSab(512, 2);
    const sabNew = createAudioRingSab(1024, 2);

    vmEnv.send({
      type: 'audio-context-state',
      outputState: 'running',
      ringSab: sabOld,
      unlockedOnGesture: true,
    });
    vmEnv.send({
      canvas: { id: 'offscreen' },
      width: 1280,
      height: 720,
      dataBase: '/data/',
      audioSab: sabNew,
    });
    const audioWorker = vmEnv.getAudioWorker();
    assert.equal(audioWorker.realMixer.ring.sab, sabNew, 'fresher boot audioSab must win over stale pre-boot SAB');
    assert.equal(audioWorker.realMixer.outputState, OUTPUT_STATES.ACTIVE);
    vmEnv.send({ type: 'destroy' });
  }

  // Case C: Pre-boot 'closed' state must NOT suppress a fresh boot with a new audioSab.
  {
    const vmEnv = bootLoaderVm({ realAudioWorker: true });
    const sabFresh = createAudioRingSab(1024, 2);

    vmEnv.send({ type: 'audio-context-state', outputState: 'closed' });
    vmEnv.send({
      canvas: { id: 'offscreen' },
      width: 1280,
      height: 720,
      dataBase: '/data/',
      audioSab: sabFresh,
    });
    const audioWorker = vmEnv.getAudioWorker();
    assert.equal(audioWorker.realMixer.ring.sab, sabFresh);
    assert.equal(audioWorker.realMixer.outputState, OUTPUT_STATES.ACTIVE);
    vmEnv.send({ type: 'destroy' });
  }

  // Case D: Direct createAudioWorkerBridge with setWorker(w, false) -> flushes strictly on onWorkerReady(w).
  {
    const posted = [];
    const mockWorker = { postMessage: (m) => posted.push(m) };
    const bridge = createAudioWorkerBridge();
    const sab = createAudioRingSab(512, 2);

    assert.equal(bridge.setWorker(mockWorker, false), true);
    const r1 = bridge.handleAttachRingSab({ type: 'attach-ring-sab', ringSab: sab });
    const r2 = bridge.handleAudioContextState({ type: 'audio-context-state', outputState: 'suspended' });
    const r3 = bridge.handleAudioContextState({
      type: 'audio-context-state',
      outputState: 'running',
      sampleRate: 48000,
    });
    assert.equal(r1.queued, true);
    assert.equal(r2.queued, true);
    assert.equal(r3.queued, true);
    assert.equal(posted.length, 0, 'must not post while workerReady=false');

    const flushed = bridge.onWorkerReady(mockWorker);
    assert.equal(flushed.flushedRingSab, true);
    assert.equal(flushed.flushedContextState, true);
    assert.equal(posted.length, 2);
    assert.equal(posted[0].type, 'attach-ring-sab');
    assert.equal(posted[0].ringSab, sab);
    assert.equal(posted[1].type, 'audio-context-state');
    assert.equal(posted[1].outputState, 'running');
    assert.equal(bridge.getState().ringSabAttached, true);
    assert.equal(bridge.getState().outputState, 'running');
  }
});

test('W377 teardown and re-init cleanup stops old audioWorker, clears retained SAB/state, and rejects post-destroy messages', () => {
  // 1. Re-init (second msg.canvas) tears down old audioWorker and prevents session #1 state bleed.
  {
    const vmEnv = bootLoaderVm({ realAudioWorker: true });
    const sab1 = createAudioRingSab(1024, 2);

    vmEnv.send({
      canvas: { id: 'offscreen-1' },
      width: 1280,
      height: 720,
      dataBase: '/data/',
      audioSab: sab1,
    });
    const firstAudioWorker = vmEnv.getAudioWorker();
    vmEnv.send({ type: 'audio-context-state', outputState: 'running', sampleRate: 44100 });
    assert.equal(vmEnv.api.getLoaderStatus().audioRingSabAttached, true);

    // Second boot WITHOUT audioSab -> must tear down firstAudioWorker and boot fresh headless session #2.
    vmEnv.send({
      canvas: { id: 'offscreen-2' },
      width: 1280,
      height: 720,
      dataBase: '/data/',
    });
    const secondAudioWorker = vmEnv.getAudioWorker();
    assert.notEqual(secondAudioWorker, firstAudioWorker);
    assert.equal(firstAudioWorker.terminated, true, 'previous audioWorker must be terminated on re-init');
    assert.equal(firstAudioWorker.onmessage, null, 'previous audioWorker.onmessage must be cleared');
    assert.equal(
      firstAudioWorker.posted.some((p) => p.msg.type === 'stop' && p.msg.stop === true),
      true,
      'previous audioWorker must receive stop before terminate',
    );
    assert.deepEqual({ ...secondAudioWorker.posted[0].msg }, { init: true });
    assert.equal(secondAudioWorker.realMixer.outputState, OUTPUT_STATES.HEADLESS_DRAIN);
    assert.equal(vmEnv.api.getLoaderStatus().audioRingSabAttached, false);
    assert.equal(vmEnv.api.getLoaderStatus().audioContextState, null);

    // Explicit audio-teardown message cleans up active audioWorker cleanly.
    vmEnv.send({ type: 'audio-teardown' });
    assert.equal(secondAudioWorker.terminated, true);
    assert.equal(secondAudioWorker.onmessage, null);
    assert.equal(vmEnv.api.getLoaderStatus().audioRingSabAttached, false);
  }

  // 2. Full `destroy` message tears down audioWorker and ignores subsequent messages without throwing.
  {
    const vmEnv = bootLoaderVm({ realAudioWorker: true });
    const sab = createAudioRingSab(1024, 2);

    vmEnv.send({
      canvas: { id: 'offscreen' },
      width: 1280,
      height: 720,
      dataBase: '/data/',
      audioSab: sab,
    });
    const audioWorker = vmEnv.getAudioWorker();
    const postedBeforeDestroy = audioWorker.posted.length;

    vmEnv.send({ type: 'destroy' });
    assert.equal(audioWorker.terminated, true);
    assert.equal(audioWorker.onmessage, null);
    assert.equal(audioWorker.posted.length, postedBeforeDestroy + 1);
    assert.deepEqual({ ...audioWorker.posted[postedBeforeDestroy].msg }, { type: 'stop', stop: true });

    // Late messages after destroy must NOT touch terminated worker or re-queue state.
    vmEnv.send({ type: 'audio-context-state', outputState: 'running', ringSab: sab });
    vmEnv.send({ type: 'attach-ring-sab', ringSab: sab });
    assert.equal(audioWorker.posted.length, postedBeforeDestroy + 1);
    assert.equal(vmEnv.api.getLoaderStatus().audioRingSabAttached, false);
    assert.equal(vmEnv.api.getLoaderStatus().audioContextState, null);
  }

  // 3. Direct createAudioWorkerBridge destroy({ sendStop: true }) invokes onResetWorker and blocks future calls.
  {
    const posted = [];
    const resetCalls = [];
    const mockWorker = { postMessage: (m) => posted.push(m) };
    const bridge = createAudioWorkerBridge({
      onResetWorker: (w) => resetCalls.push(w),
    });
    const sab = createAudioRingSab(512, 2);

    bridge.initWorker(mockWorker, { audioSab: sab });
    assert.equal(bridge.getState().ringSabAttached, true);
    bridge.destroy({ sendStop: true });

    assert.equal(resetCalls.length, 1);
    assert.equal(resetCalls[0], mockWorker);
    assert.deepEqual(posted[posted.length - 1], { type: 'stop', stop: true });
    assert.equal(bridge.getState().destroyed, true);
    assert.equal(bridge.getState().hasWorker, false);
    assert.equal(bridge.getState().ringSabAttached, false);
    assert.equal(bridge.handleAttachRingSab({ ringSab: sab }).reason, 'destroyed');
    assert.equal(bridge.handleAudioContextState({ outputState: 'running' }).reason, 'destroyed');
  }
});

test('W377 end-to-end app_bootstrap -> loader.js (VM) -> AudioMixerWorker -> AudioEngine.sab -> PlayGta6AudioWorkletProcessor plays non-zero stereo audio after gesture unlock', async () => {
  const vmEnv = bootLoaderVm({ realAudioWorker: true });
  const webAudio = createRealWorkletWebAudioHost({
    initialState: 'suspended',
    actualSampleRate: 48000,
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

  // Wire app_bootstrap's loader worker straight into the VM-loaded real `web/loader.js`!
  const vmLoaderWorkerBridge = createFakeEventTarget({
    postMessage(payload) {
      vmEnv.send(payload);
    },
    terminate() {
      vmEnv.send({ type: 'destroy' });
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
    createWasmWorker: () => vmLoaderWorkerBridge,
  });

  try {
    await app.start();
    const audioWorker = vmEnv.getAudioWorker();
    assert.ok(audioWorker && audioWorker.realMixer);
    assert.equal(audioWorker.realMixer.ring.sab, app.audioEngine.sab);

    // Attach simulated runtime Playgta6AudioRing and push 1024 frames of stereo sine wave.
    const guestRing = createSimulatedGuestPcmRing({ capacitySamples: 16384, sampleRate: 48000 });
    audioWorker.realMixer.handleMessage({
      type: 'attach-guest-pcm-ring',
      ...guestRing.descriptor,
      autoPump: true,
    });

    // User clicks to unlock headed WebAudio -> app_bootstrap notifies loader.js -> loader.js forwards to audioWorker!
    windowObj.dispatch('click');
    await new Promise((resolve) => setTimeout(resolve, 15));
    assert.equal(app.audioEngine.state, 'running');
    assert.equal(audioWorker.realMixer.outputState, OUTPUT_STATES.ACTIVE);

    // Guest produces 1024 stereo frames; pump AudioMixerWorker timer once.
    const sine = makeStereoSineWave(1024, { sampleRate: 48000, freqL: 440, freqR: 880, ampL: 0.6, ampR: 0.4 });
    assert.equal(guestRing.producerPushStereo(sine), 2048);
    vmEnv.advanceNow(25);
    const activeTimer = vmEnv.audioTimers.find((t) => !t.cleared);
    assert.ok(activeTimer, 'AudioMixerWorker guest auto-pump timer must be running');
    activeTimer.fn();

    // Real PlayGta6AudioWorkletProcessor pulls directly from AudioEngine.sab -> must hear non-zero stereo sine!
    const processor = webAudio.hostState.processors[0];
    assert.ok(processor);
    const outL = new Float32Array(512);
    const outR = new Float32Array(512);
    const qL = new Float32Array(128);
    const qR = new Float32Array(128);
    for (let q = 0; q < 4; q += 1) {
      assert.equal(processor.process([], [[qL, qR]]), true);
      outL.set(qL, q * 128);
      outR.set(qR, q * 128);
    }
    assert.ok(computeRms(outL) > 0.35, `expected audible left channel RMS > 0.35, got ${computeRms(outL)}`);
    assert.ok(computeRms(outR) > 0.2, `expected audible right channel RMS > 0.20, got ${computeRms(outR)}`);

    // OS suspends AudioContext -> app_bootstrap forwards 'suspended' -> loader.js -> AudioMixerWorker enters null-drain!
    webAudio.hostState.contexts[0].emitStateChange('suspended');
    assert.equal(audioWorker.realMixer.outputState, OUTPUT_STATES.SUSPENDED);
  } finally {
    await app.destroy();
  }
});
