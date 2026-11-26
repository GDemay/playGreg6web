// Unit & integration tests for W382 (Closes #1759, Refs #1028, #663):
// - Zero-alloc ITU-R BS.775 5.1/7.1/quad/stereo/mono downmix (`downmixToStereoInto` & `downmixToStereo(..., { out })`)
// - Zero-alloc SoftKneePeakLimiter (fast attack, smooth exponential release, bit-exact below knee, ceiling guard)
// - Smooth half-cosine underrun fade-out & deferred/resume fade-in ramps + telemetry
// - Headed Chrome AudioContext unlock (click/key/gamepad + active gamepad poll) & visibilitychange/focus recovery
// - Pointer Lock UX helpers in app_bootstrap.js & web/index.html (with Greg splash + live Elapsed timer preserved)

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

import { bootstrapPlayGta6App } from '../web/app_bootstrap.js';

const require = createRequire(import.meta.url);
const mixerMod = require('../web/workers/audio_mixer.js');

const {
  AudioMixer,
  SoftKneePeakLimiter,
  downmixToStereo,
  downmixToStereoInto,
  ITU_BS775_CENTER_GAIN,
  ITU_BS775_SURROUND_GAIN,
  DEFAULT_PEAK_LIMITER_KNEE,
  DEFAULT_PEAK_LIMITER_CEILING,
  DEFAULT_PEAK_LIMITER_ATTACK_MS,
  DEFAULT_PEAK_LIMITER_RELEASE_MS,
} = mixerMod;

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');
const INDEX_HTML_PATH = path.join(REPO_ROOT, 'web', 'index.html');
const SPLASH_OVERLAY_PATH = path.join(REPO_ROOT, 'web', 'ui', 'splash_overlay.js');

function createFakeEventTarget(extra = {}) {
  const listeners = new Map();
  return {
    ...extra,
    addEventListener(type, fn, opts) {
      if (!listeners.has(type)) {
        listeners.set(type, new Set());
      }
      listeners.get(type).add({ fn, once: Boolean(opts && opts.once) });
    },
    removeEventListener(type, fn) {
      const set = listeners.get(type);
      if (!set) return;
      for (const entry of set) {
        if (entry.fn === fn) {
          set.delete(entry);
        }
      }
    },
    dispatch(type, event = {}) {
      const set = listeners.get(type);
      if (!set) return;
      for (const entry of [...set]) {
        if (entry.once) {
          set.delete(entry);
        }
        entry.fn({ type, ...event });
      }
    },
    listenerCount(type) {
      if (type) {
        return listeners.get(type)?.size ?? 0;
      }
      let total = 0;
      for (const set of listeners.values()) {
        total += set.size;
      }
      return total;
    },
  };
}

function createFakeTimers() {
  let nextId = 1;
  const active = new Map();
  return {
    setInterval(fn, ms) {
      const id = nextId++;
      active.set(id, { fn, ms });
      return id;
    },
    clearInterval(id) {
      active.delete(id);
    },
    tickAll() {
      for (const entry of [...active.values()]) {
        entry.fn();
      }
    },
    activeCount() {
      return active.size;
    },
  };
}

function createMockWebAudioHost({ initialState = 'suspended', failResumeCount = 0 } = {}) {
  let remainingFailures = failResumeCount;
  const contexts = [];
  const gainNodes = [];

  class MockAudioContext {
    constructor() {
      this.state = initialState;
      this.sampleRate = 48000;
      this.currentTime = 0;
      this.destination = { kind: 'destination' };
      this._listeners = new Map();
      this.audioWorklet = {
        async addModule() {},
      };
      contexts.push(this);
    }
    addEventListener(type, fn) {
      if (!this._listeners.has(type)) this._listeners.set(type, new Set());
      this._listeners.get(type).add(fn);
    }
    removeEventListener(type, fn) {
      this._listeners.get(type)?.delete(fn);
    }
    emitStateChange(nextState) {
      this.state = nextState;
      for (const fn of [...(this._listeners.get('statechange') || [])]) {
        fn({ type: 'statechange' });
      }
    }
    createGain() {
      const node = {
        gain: {
          value: 1.0,
          cancelScheduledValues() {},
          setValueAtTime(v) {
            this.value = v;
          },
          linearRampToValueAtTime(v) {
            this.value = v;
          },
        },
        connect() {},
        disconnect() {},
      };
      gainNodes.push(node);
      return node;
    }
    async resume() {
      if (remainingFailures > 0) {
        remainingFailures -= 1;
        throw new Error('NotAllowedError: user activation required');
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
    constructor() {
      this.port = { postMessage() {}, close() {} };
    }
    connect() {}
    disconnect() {}
  }

  return {
    MockAudioContext,
    MockAudioWorkletNode,
    contexts,
    gainNodes,
    setFailResumeCount(n) {
      remainingFailures = n;
    },
  };
}

// ---------------------------------------------------------------------------
// 1. Zero-Alloc ITU 5.1/7.1/Quad/Stereo/Mono Downmix
// ---------------------------------------------------------------------------

test('W382: downmixToStereoInto and downmixToStereo({ out }) perform zero-alloc ITU-R BS.775 downmix across 1/2/4/6/8 channels', () => {
  assert.equal(typeof downmixToStereoInto, 'function');
  assert.equal(globalThis.PlayGta6AudioMixer.downmixToStereoInto, downmixToStereoInto);

  const frames = 128;
  const src51 = new Float32Array(frames * 6);
  for (let f = 0; f < frames; f++) {
    src51[f * 6 + 0] = 0.25; // FL
    src51[f * 6 + 1] = -0.2; // FR
    src51[f * 6 + 2] = 0.3; // FC
    src51[f * 6 + 3] = 0.9; // LFE (ignored by default)
    src51[f * 6 + 4] = 0.2; // BL
    src51[f * 6 + 5] = -0.1; // BR
  }
  // Inject NaN & +-Infinity on frame 0 without poisoning valid channels.
  src51[2] = Number.NaN;
  src51[3] = Number.POSITIVE_INFINITY;
  src51[4] = Number.NEGATIVE_INFINITY;

  const outStereo = new Float32Array(frames * 2);
  const reusableStats = {};
  const opts = { out: outStereo, stats: reusableStats };

  // Track Float32Array allocations across 32 consecutive downmix quanta.
  const origFloat32 = globalThis.Float32Array;
  let float32Allocs = 0;
  let tracking = false;
  globalThis.Float32Array = new Proxy(origFloat32, {
    construct(target, args, newTarget) {
      if (tracking) float32Allocs += 1;
      return Reflect.construct(target, args, newTarget);
    },
  });

  try {
    tracking = true;
    for (let q = 0; q < 16; q++) {
      const written = downmixToStereoInto(src51, 6, outStereo, frames, opts);
      assert.equal(written, frames);
      const returned = downmixToStereo(src51, 6, opts);
      assert.equal(returned, outStereo);
    }
    tracking = false;
  } finally {
    tracking = false;
    globalThis.Float32Array = origFloat32;
  }

  assert.equal(float32Allocs, 0, `Expected 0 Float32Array allocations, got ${float32Allocs}`);

  // Verify ITU-R BS.775 math & NaN isolation on frame 0 and steady frame 1.
  assert.ok(Math.abs(outStereo[0] - 0.25) < 1e-6);
  assert.ok(Math.abs(outStereo[1] - (-0.2 - 0.1 * ITU_BS775_SURROUND_GAIN)) < 1e-6);

  const expL = 0.25 + 0.3 * ITU_BS775_CENTER_GAIN + 0.2 * ITU_BS775_SURROUND_GAIN;
  const expR = -0.2 + 0.3 * ITU_BS775_CENTER_GAIN - 0.1 * ITU_BS775_SURROUND_GAIN;
  assert.ok(Math.abs(outStereo[2] - expL) < 1e-6);
  assert.ok(Math.abs(outStereo[3] - expR) < 1e-6);
  assert.equal(reusableStats.downmixedFrames, 32 * frames);
  assert.equal(reusableStats.nanSanitizedSamples, 32 * 3);
  assert.equal(reusableStats.peakLimiterGainReduction, 1.0);

  // Verify 7.1, quad, stereo, mono paths into pre-allocated buffer.
  const outSmall = new Float32Array(2);
  assert.equal(downmixToStereoInto(new Float32Array([0.1, 0.2, 0.2, 0.5, 0.1, 0.1, 0.1, 0.1]), 8, outSmall, 1), 1);
  assert.ok(Math.abs(outSmall[0] - (0.1 + 0.2 * ITU_BS775_CENTER_GAIN + 0.2 * ITU_BS775_SURROUND_GAIN)) < 1e-6);
  assert.equal(downmixToStereoInto(new Float32Array([0.3, -0.3, 0.2, -0.2]), 4, outSmall, 1), 1);
  assert.ok(Math.abs(outSmall[0] - (0.3 + 0.2 * ITU_BS775_SURROUND_GAIN)) < 1e-6);
  assert.equal(downmixToStereoInto(new Float32Array([0.4, -0.5]), 2, outSmall, 1), 1);
  assert.ok(Math.abs(outSmall[0] - 0.4) < 1e-6 && Math.abs(outSmall[1] - -0.5) < 1e-6);
  assert.equal(downmixToStereoInto(new Float32Array([0.6]), 1, outSmall, 1), 1);
  assert.ok(Math.abs(outSmall[0] - 0.6) < 1e-6 && Math.abs(outSmall[1] - 0.6) < 1e-6);

  // Validation guards.
  assert.throws(() => downmixToStereoInto(src51, 6, new Float32Array(4), frames), RangeError);
  assert.throws(() => downmixToStereoInto(src51, 6, [], frames), TypeError);
  assert.throws(() => downmixToStereo(src51, 6, { out: new Float32Array(2) }), RangeError);
});

// ---------------------------------------------------------------------------
// 2. SoftKneePeakLimiter & AudioMixer Integrated Peak Limiter + Underrun/Resume Ramps
// ---------------------------------------------------------------------------

test('W382: SoftKneePeakLimiter is bit-exact below knee, smoothly limits explosions without hard-clipping, and releases smoothly', () => {
  assert.equal(typeof SoftKneePeakLimiter, 'function');
  assert.equal(globalThis.PlayGta6AudioMixer.SoftKneePeakLimiter, SoftKneePeakLimiter);
  assert.equal(DEFAULT_PEAK_LIMITER_KNEE, 0.9);
  assert.equal(DEFAULT_PEAK_LIMITER_CEILING, 0.99);
  assert.ok(DEFAULT_PEAK_LIMITER_ATTACK_MS > 0);
  assert.ok(DEFAULT_PEAK_LIMITER_RELEASE_MS > 0);

  const limiter = new SoftKneePeakLimiter({
    sampleRate: 48000,
    knee: 0.88,
    ceiling: 0.99,
    attackMs: 0.5,
    releaseMs: 20,
  });

  // 1. Sub-knee audio is 100% bit-exact passthrough (gain reduction === 1.0).
  const normal = new Float32Array(128 * 2);
  for (let f = 0; f < 128; f++) {
    normal[f * 2] = 0.65 * Math.sin((2 * Math.PI * 440 * f) / 48000);
    normal[f * 2 + 1] = -0.8 * Math.cos((2 * Math.PI * 330 * f) / 48000);
  }
  const normalCopy = new Float32Array(normal);
  assert.equal(limiter.processInterleaved(normal, 128), 128);
  assert.deepEqual(Array.from(normal), Array.from(normalCopy));
  assert.equal(limiter.gainReduction, 1.0);
  assert.equal(limiter.engagedCount, 0);

  // 2. Massive simultaneous explosion + gunfire + radio sum (peak ~3.2) stays strictly <= ceiling (0.99),
  //    never propagates NaN, and preserves smooth envelope without flat square-wave hard clipping.
  const hot = new Float32Array(256 * 2);
  for (let f = 0; f < 256; f++) {
    const s = 3.2 * Math.sin((2 * Math.PI * 180 * f) / 48000);
    hot[f * 2] = s;
    hot[f * 2 + 1] = -0.5 * s;
  }
  hot[10] = Number.NaN;
  hot[11] = Number.POSITIVE_INFINITY;

  assert.equal(limiter.processInterleaved(hot, 256), 256);
  for (let i = 0; i < hot.length; i++) {
    assert.ok(Number.isFinite(hot[i]), `Sample ${i} must be finite`);
    assert.ok(
      Math.abs(hot[i]) <= 0.99 + 1e-6,
      `Sample ${i} exceeded ceiling 0.99: ${hot[i]}`
    );
  }
  assert.equal(hot[10], 0.0);
  assert.equal(hot[11], 0.0);
  assert.ok(limiter.gainReduction < 0.5, `Expected strong gain reduction on 3.2x peak, got ${limiter.gainReduction}`);
  assert.ok(limiter.engagedCount > 0);
  assert.equal(limiter.nonFiniteSanitized, 2);

  // 3. Smooth exponential release back to 1.0 over release window (no sudden jump).
  const gainAfterBurst = limiter.currentGain;
  assert.ok(gainAfterBurst < 0.9);
  const quiet = new Float32Array(256 * 2);
  for (let f = 0; f < 256; f++) {
    quiet[f * 2] = 0.25;
    quiet[f * 2 + 1] = -0.25;
  }
  let prevGain = gainAfterBurst;
  for (let q = 0; q < 60; q++) {
    quiet.fill(0.25);
    limiter.processInterleaved(quiet, 256);
    assert.ok(limiter.currentGain >= prevGain - 1e-7, 'Release gain must recover monotonically');
    prevGain = limiter.currentGain;
  }
  assert.equal(limiter.currentGain, 1.0, `Expected limiter gain to release cleanly back to 1.0, got ${limiter.currentGain}`);
});

test('W382: AudioMixer integrates SoftKneePeakLimiter zero-alloc in mix() + deferred cosine recovery & resumeFadeIns telemetry', () => {
  const mixer = new AudioMixer({
    sampleRate: 48000,
    quantumFrames: 128,
    reuseOutputBuffer: true,
    underrunRampFrames: 64,
    peakLimiter: {
      knee: 0.88,
      ceiling: 0.99,
      attackMs: 0.5,
      releaseMs: 30,
    },
  });

  // 1. Sub-knee voice passes bit-exact with peakLimiterGainReduction === 1.0.
  const radio = new Float32Array(256).fill(0.4);
  const vRadio = mixer.createVoice({ samples: radio, channels: 1, gain: 1.0, pan: -1.0, loop: true });
  const target = new Float32Array(128 * 2);
  assert.equal(mixer.mix(target, 128), 128);
  assert.equal(target[0], Math.fround(0.4));
  assert.equal(mixer.getStats().peakLimiterGainReduction, 1.0);
  assert.equal(mixer.getStats().peakLimiterEngagedCount, 0);
  assert.equal(mixer.getStats().peakLimiterEnabled, true);
  assert.equal(mixer.getStats().peakLimiterCeiling, 0.99);

  // 2. Add simultaneous loud explosions + 5.1 gunfire; verify zero allocations in hot mix() loop.
  const explosion = new Float32Array(256).fill(1.5);
  const gunfire51 = new Float32Array(256 * 6).fill(0.9);
  const vExp = mixer.createVoice({ samples: explosion, channels: 1, gain: 1.0, pan: -1.0, loop: true });
  const vGun = mixer.createVoice({ samples: gunfire51, channels: 6, gain: 1.0, pan: 0.0, loop: true });

  const origFloat32 = globalThis.Float32Array;
  let float32Allocs = 0;
  let tracking = false;
  globalThis.Float32Array = new Proxy(origFloat32, {
    construct(t, args, nt) {
      if (tracking) float32Allocs += 1;
      return Reflect.construct(t, args, nt);
    },
  });

  try {
    tracking = true;
    for (let q = 0; q < 16; q++) {
      assert.equal(mixer.mix(target, 128), 128);
    }
    tracking = false;
  } finally {
    tracking = false;
    globalThis.Float32Array = origFloat32;
  }

  assert.equal(float32Allocs, 0, `Expected 0 Float32Array allocations in peak-limited mix(), got ${float32Allocs}`);
  for (let i = 0; i < target.length; i++) {
    assert.ok(Number.isFinite(target[i]) && Math.abs(target[i]) <= 0.99 + 1e-6);
  }
  const loudStats = mixer.getStats();
  assert.ok(loudStats.peakLimiterGainReduction < 0.6);
  assert.ok(loudStats.peakLimiterEngagedCount >= 16 * 128);
  assert.equal(loudStats.scratchReallocations, 0);

  // 3. Underrun fade-out -> silent quantum (voices paused) -> resumed voice still gets smooth half-cosine fade-in!
  mixer.destroyVoice(vExp);
  mixer.destroyVoice(vGun);
  const fadeOutBuf = new Float32Array(64 * 2);
  assert.equal(mixer.renderUnderrunFadeOut(fadeOutBuf, 64), 64);
  assert.equal(fadeOutBuf[63 * 2], 0.0);
  assert.equal(mixer.getStats().underrunFadeOuts, 1);
  assert.equal(mixer.getStats().resumeFadeIns, 1);

  // Pause radio during silence window: recovery fade-in must stay armed (not wasted on silence).
  mixer.pauseVoice(vRadio);
  mixer.mix(target, 128);
  assert.equal(target[0], 0.0);
  assert.equal(target[127 * 2], 0.0);

  // Resume radio: first audible quantum gets the smooth half-cosine fade-in from 0.0 -> steady state!
  mixer.resumeVoice(vRadio);
  mixer.mix(target, 128);
  assert.equal(target[0], 0.0, 'First sample on recovery out of silence must start at 0.0');
  assert.ok(target[31 * 2] > 0.1 && target[31 * 2] < 0.3, `Midpoint of fade-in should be ~0.2, got ${target[31 * 2]}`);
  assert.ok(target[64 * 2] > 0.35, `Post-fade sample should recover cleanly, got ${target[64 * 2]}`);

  // 4. Explicit tab-switch notifyResume() arms smooth fade-in and increments resumeFadeIns.
  mixer.notifyResume(32);
  assert.equal(mixer.getStats().resumeFadeIns, 2);
  mixer.mix(target, 64);
  assert.equal(target[0], 0.0);
  assert.ok(target[31 * 2] > 0.35);

  // 5. resetStats() & reset() clear all W382 limiter & fade counters cleanly.
  mixer.reset();
  const afterReset = mixer.getStats();
  assert.equal(afterReset.peakLimiterGainReduction, 1.0);
  assert.equal(afterReset.peakLimiterCurrentGain, 1.0);
  assert.equal(afterReset.peakLimiterEngagedCount, 0);
  assert.equal(afterReset.underrunFadeOuts, 0);
  assert.equal(afterReset.resumeFadeIns, 0);
});

// ---------------------------------------------------------------------------
// 3. Headed AudioContext Unlock (Gamepad + visibilitychange) & Pointer Lock UX
// ---------------------------------------------------------------------------

test('W382: app_bootstrap unlocks AudioContext via gamepadconnected and pre-connected gamepad input polling', async () => {
  // Case A: gamepadconnected event unlocks suspended AudioContext and forwards active state.
  {
    const webAudio = createMockWebAudioHost({ initialState: 'suspended' });
    const timers = createFakeTimers();
    const windowObj = createFakeEventTarget({
      crossOriginIsolated: true,
      innerWidth: 1280,
      innerHeight: 720,
      devicePixelRatio: 1,
      location: { origin: 'http://127.0.0.1:8080', search: '' },
      AudioContext: webAudio.MockAudioContext,
      AudioWorkletNode: webAudio.MockAudioWorkletNode,
      setInterval: timers.setInterval,
      clearInterval: timers.clearInterval,
    });
    const documentObj = createFakeEventTarget({
      pointerLockElement: null,
      visibilityState: 'visible',
      hidden: false,
    });
    const canvas = createFakeEventTarget({
      width: 1280,
      height: 720,
      getContext: () => null,
    });
    const loaderMessages = [];
    const mockWorker = createFakeEventTarget({
      postMessage(msg) {
        loaderMessages.push(msg);
      },
      terminate() {},
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
      setInterval: timers.setInterval,
      clearInterval: timers.clearInterval,
      createWasmWorker: () => mockWorker,
    });

    try {
      await app.start();
      assert.equal(app.getAudioStatus().unlocked, false);

      windowObj.dispatch('gamepadconnected', { gamepad: { id: 'Xbox Wireless Controller', index: 0 } });
      await new Promise((r) => setTimeout(r, 10));

      assert.equal(app.audioEngine.state, 'running');
      assert.equal(app.getAudioStatus().unlocked, true);
      assert.ok(
        loaderMessages.some((m) => m.type === 'audio-context-state' && m.outputState === 'active'),
        'gamepadconnected unlock must forward active state to runtime worker'
      );
    } finally {
      await app.destroy();
    }
    assert.equal(windowObj.listenerCount(), 0);
    assert.equal(documentObj.listenerCount(), 0);
    assert.equal(canvas.listenerCount(), 0);
    assert.equal(timers.activeCount(), 0);
  }

  // Case B: Already-connected gamepad before start() polls until button/axis input unlocks AudioContext.
  {
    const webAudio = createMockWebAudioHost({ initialState: 'suspended' });
    const timers = createFakeTimers();
    const padState = {
      connected: true,
      id: 'Xbox Series Controller',
      buttons: [{ pressed: false, value: 0 }],
      axes: [0, 0, 0, 0],
    };

    const windowObj = createFakeEventTarget({
      crossOriginIsolated: true,
      innerWidth: 1280,
      innerHeight: 720,
      devicePixelRatio: 1,
      location: { origin: 'http://127.0.0.1:8080', search: '' },
      AudioContext: webAudio.MockAudioContext,
      AudioWorkletNode: webAudio.MockAudioWorkletNode,
      setInterval: timers.setInterval,
      clearInterval: timers.clearInterval,
    });
    const documentObj = createFakeEventTarget({
      pointerLockElement: null,
      visibilityState: 'visible',
      hidden: false,
    });
    const canvas = createFakeEventTarget({ width: 1280, height: 720, getContext: () => null });

    const app = bootstrapPlayGta6App({
      windowObj,
      documentObj,
      navigatorObj: { gpu: {}, getGamepads: () => [padState] },
      fetchImpl: async () => ({
        ok: true,
        status: 200,
        async json() {
          return { ok: true, coopCoep: true, hasWasmBinary: true };
        },
      }),
      canvas,
      setInterval: timers.setInterval,
      clearInterval: timers.clearInterval,
      createWasmWorker: () => createFakeEventTarget({ postMessage() {}, terminate() {} }),
    });

    try {
      await app.start();
      assert.equal(app.getAudioStatus().gamepadPollActive, true);

      // Neutral gamepad does not unlock yet.
      timers.tickAll();
      await new Promise((r) => setTimeout(r, 5));
      assert.equal(app.getAudioStatus().unlocked, false);

      // Player presses A button on gamepad -> triggers unlock and stops poll timer!
      padState.buttons[0] = { pressed: true, value: 1 };
      timers.tickAll();
      await new Promise((r) => setTimeout(r, 10));

      assert.equal(app.audioEngine.state, 'running');
      assert.equal(app.getAudioStatus().unlocked, true);
      assert.equal(app.getAudioStatus().gamepadPollActive, false);
    } finally {
      await app.destroy();
    }
    assert.equal(timers.activeCount(), 0);
    assert.equal(windowObj.listenerCount(), 0);
    assert.equal(documentObj.listenerCount(), 0);
  }
});

test('W382: visibilitychange recovers suspended AudioContext, re-arms gesture on autoplay rejection, and auto-releases Pointer Lock when hidden', async () => {
  const webAudio = createMockWebAudioHost({ initialState: 'suspended' });
  const timers = createFakeTimers();
  const loaderMessages = [];

  const windowObj = createFakeEventTarget({
    crossOriginIsolated: true,
    innerWidth: 1280,
    innerHeight: 720,
    devicePixelRatio: 1,
    location: { origin: 'http://127.0.0.1:8080', search: '' },
    AudioContext: webAudio.MockAudioContext,
    AudioWorkletNode: webAudio.MockAudioWorkletNode,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
  });

  let exitPointerLockCalls = 0;
  let requestPointerLockCalls = 0;
  let rejectNextPointerLock = false;

  const canvas = createFakeEventTarget({
    width: 1280,
    height: 720,
    getContext: () => null,
    requestPointerLock() {
      requestPointerLockCalls += 1;
      if (rejectNextPointerLock) {
        rejectNextPointerLock = false;
        return Promise.reject(new Error('SecurityError: user exited lock too quickly'));
      }
      documentObj.pointerLockElement = canvas;
      documentObj.dispatch('pointerlockchange');
      return Promise.resolve();
    },
  });

  const documentObj = createFakeEventTarget({
    pointerLockElement: null,
    visibilityState: 'visible',
    hidden: false,
    exitPointerLock() {
      exitPointerLockCalls += 1;
      documentObj.pointerLockElement = null;
      documentObj.dispatch('pointerlockchange');
    },
  });

  const hudLogs = [];
  const hudMetrics = [];
  const hud = {
    setStatus() {},
    appendLog(msg, level) {
      hudLogs.push({ msg, level });
    },
    updateMetrics(m) {
      hudMetrics.push({ ...m });
    },
  };

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
    hud,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker: () =>
      createFakeEventTarget({
        postMessage(msg) {
          loaderMessages.push(msg);
        },
        terminate() {},
      }),
  });

  try {
    await app.start();

    // 1. requestPointerLock() unlocks AudioContext on the same gesture and locks pointer.
    assert.equal(app.isPointerLocked(), false);
    const lockRes = await app.requestPointerLock();
    assert.equal(lockRes.ok, true);
    assert.equal(lockRes.locked, true);
    assert.equal(app.isPointerLocked(), true);
    assert.equal(requestPointerLockCalls, 1);

    await new Promise((r) => setTimeout(r, 10));
    assert.equal(app.audioEngine.state, 'running');

    // 2. Tab hidden -> auto-releases Pointer Lock; OS interrupts AudioContext.
    documentObj.visibilityState = 'hidden';
    documentObj.hidden = true;
    documentObj.dispatch('visibilitychange');
    assert.equal(exitPointerLockCalls, 1);
    assert.equal(app.isPointerLocked(), false);

    // Simulate OS/Chrome suspending AudioContext while tab was hidden.
    webAudio.contexts[0].emitStateChange('suspended');
    assert.equal(app.audioEngine.state, 'suspended');

    // 3. Tab becomes visible -> visibilitychange automatically recovers AudioContext!
    documentObj.visibilityState = 'visible';
    documentObj.hidden = false;
    documentObj.dispatch('visibilitychange');
    await new Promise((r) => setTimeout(r, 15));

    assert.equal(app.audioEngine.state, 'running');
    assert.equal(app.getAudioStatus().visibilityRecoveries, 1);
    assert.ok(
      loaderMessages.filter((m) => m.type === 'audio-context-state' && m.outputState === 'active')
        .length >= 2,
      'visibilitychange recovery must forward active state to runtime worker'
    );

    // 4. If autoplay policy rejects resume on visibilitychange, gesture listeners re-arm cleanly!
    webAudio.contexts[0].emitStateChange('suspended');
    webAudio.setFailResumeCount(1);
    documentObj.dispatch('visibilitychange');
    await new Promise((r) => setTimeout(r, 15));
    assert.equal(app.audioEngine.state, 'suspended');
    assert.equal(app.getAudioStatus().gestureArmed, true);

    // Next click gesture unlocks cleanly!
    windowObj.dispatch('click');
    await new Promise((r) => setTimeout(r, 15));
    assert.equal(app.audioEngine.state, 'running');

    // 5. Rejected requestPointerLock() Promise is caught cleanly without unhandled rejection.
    rejectNextPointerLock = true;
    const rejRes = await app.togglePointerLock();
    assert.equal(rejRes.ok, false);
    assert.equal(rejRes.reason, 'request-rejected');
    assert.equal(app.isPointerLocked(), false);
    assert.ok(hudLogs.some((l) => /Pointer lock request rejected/i.test(l.msg)));

    // 6. Toggle lock -> unlock works cleanly.
    const t1 = await app.togglePointerLock();
    assert.equal(t1.ok, true);
    assert.equal(t1.locked, true);
    const t2 = await app.togglePointerLock();
    assert.equal(t2.ok, true);
    assert.equal(t2.locked, false);
  } finally {
    await app.destroy();
  }

  assert.equal(windowObj.listenerCount(), 0);
  assert.equal(documentObj.listenerCount(), 0);
  assert.equal(canvas.listenerCount(), 0);
  assert.equal(timers.activeCount(), 0);
});

// ---------------------------------------------------------------------------
// 4. web/index.html Pointer Lock Button + UI Preservation Guard (#1665)
// ---------------------------------------------------------------------------

test('W382: web/index.html adds #btn-pointer-lock in collapsed Diagnostics & Tools while 100% preserving Greg splash & live Elapsed timer', () => {
  const indexHtml = fs.readFileSync(INDEX_HTML_PATH, 'utf8');
  const splashJs = fs.readFileSync(SPLASH_OVERLAY_PATH, 'utf8');

  // 1. Valid ESM module syntax in web/index.html.
  const match = /<script\s+type="module">([\s\S]*?)<\/script>/i.exec(indexHtml);
  assert.ok(match);
  const moduleScript = match[1];
  const check = spawnSync(process.execPath, ['--input-type=module', '--check'], {
    input: moduleScript,
    encoding: 'utf8',
  });
  assert.equal(check.status, 0, `Inline script syntax error:\n${check.stderr}`);

  // 2. #btn-pointer-lock exists inside collapsed Diagnostics & Tools, with no duplicate IDs.
  const diagStart = indexHtml.indexOf('<details id="hud-diagnostics-disclosure"');
  const diagEnd = indexHtml.indexOf('</details>', diagStart);
  assert.ok(diagStart !== -1 && diagEnd > diagStart);
  const diagHtml = indexHtml.slice(diagStart, diagEnd);
  assert.match(diagHtml, /<button\s+id="btn-pointer-lock"[^>]*data-locked="false"[^>]*>Lock Mouse<\/button>/);

  const ids = [...indexHtml.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]);
  assert.equal(new Set(ids).size, ids.length, 'web/index.html must have zero duplicate IDs');

  // 3. Inline script wires #btn-pointer-lock, togglePointerLockFromUi, and lock/unlock logs.
  assert.match(moduleScript, /getElementById\(\s*['"]btn-pointer-lock['"]\s*\)/);
  assert.match(moduleScript, /togglePointerLockFromUi/);
  assert.match(moduleScript, /Pointer lock active \(press Esc to release\)\./);
  assert.match(moduleScript, /Pointer lock released\./);
  assert.match(moduleScript, /togglePointerLock:\s*\(\)\s*=>\s*togglePointerLockFromUi\(\)/);

  // 4. 100% preservation of Greg Theft Auto VI splash artwork & live Elapsed MM:SS timer (#1665).
  assert.match(indexHtml, /<div id="splash-overlay-root"><\/div>/);
  assert.match(indexHtml, /<script src="\.\/ui\/splash_overlay\.js"><\/script>/);
  assert.match(splashJs, /assets\/greg\.png/);
  assert.match(splashJs, /Elapsed/);
});
