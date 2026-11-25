import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const require = createRequire(import.meta.url);
const audioMod = require('../web/workers/audio_worker.js');
const {
  AudioRingBuffer,
  AudioMixerWorker,
  GuestPcmRingReader,
  PcmScratchPool,
  OUTPUT_STATES,
  OVERFLOW_POLICIES,
  DEFAULT_MAX_PCM_BYTES,
  DEFAULT_DECLICK_FRAMES,
  DEFAULT_TARGET_LATENCY_MS,
  MIN_TARGET_LATENCY_MS,
  MAX_TARGET_LATENCY_MS,
  GUEST_PCM_RING_DEFAULT_SAMPLE_RATE,
  GUEST_PCM_RING_CHANNELS,
  GUEST_PCM_RING_DEFAULT_CAPACITY_SAMPLES,
  GUEST_PCM_RING_DEFAULT_QUANTUM_FRAMES,
  ITU_BS775_CENTER_GAIN,
  ITU_BS775_SURROUND_GAIN,
  clampTargetLatencyMs,
  deinterleavePcm,
  interleavePcm,
  extractSafePcmSamples,
  clonePcmForTransfer,
  sanitizeTransferList,
  softClipSample,
  applyFadeRamp,
  sanitizeAndSoftLimitBuffer,
  downmixToStereo,
  resampleLinear,
  buildAudioTelemetrySnapshot,
} = audioMod;

function makeStereoFrames(startIndex, frameCount) {
  const out = new Float32Array(frameCount * 2);
  for (let f = 0; f < frameCount; f++) {
    const idx = startIndex + f;
    out[f * 2] = (idx + 1) * 0.25; // Left channel
    out[f * 2 + 1] = -(idx + 1) * 0.125; // Right channel
  }
  return out;
}

function makeDetachedArrayBuffer(byteLength = 32) {
  const ab = new ArrayBuffer(byteLength);
  if (typeof ab.transfer === 'function') {
    ab.transfer();
  } else {
    structuredClone(ab, { transfer: [ab] });
  }
  return ab;
}

test('UMD exports expose AudioRingBuffer and AudioMixerWorker on module.exports and globalThis', () => {
  assert.equal(typeof AudioRingBuffer, 'function');
  assert.equal(typeof AudioMixerWorker, 'function');
  assert.equal(globalThis.PlayGta6Audio.AudioRingBuffer, AudioRingBuffer);
  assert.equal(globalThis.PlayGta6Audio.AudioMixerWorker, AudioMixerWorker);
  assert.deepEqual(OUTPUT_STATES, {
    ACTIVE: 'active',
    SUSPENDED: 'suspended',
    HEADLESS_DRAIN: 'headless-drain',
  });
});

test('AudioRingBuffer stereo frame push/pull exactness across shared SAB views', () => {
  const producer = AudioRingBuffer.create(16, 2);
  const consumer = new AudioRingBuffer(producer.sab);

  assert.equal(consumer.capacityFrames, 16);
  assert.equal(consumer.channels, 2);
  assert.equal(producer.availableFrames(), 0);
  assert.equal(consumer.availableFrames(), 0);

  const input = makeStereoFrames(0, 6);
  const written = producer.pushSamples(input);
  assert.equal(written, 6);
  assert.equal(producer.availableFrames(), 6);
  assert.equal(consumer.availableFrames(), 6);
  assert.equal(producer.freeFrames(), 10);

  const output = new Float32Array(6 * 2);
  const read = consumer.pullSamples(output);
  assert.equal(read, 6);
  assert.deepEqual(Array.from(output), Array.from(input));
  assert.equal(consumer.availableFrames(), 0);
  assert.equal(consumer.underruns, 0);
  assert.equal(producer.overflows, 0);
});

test('AudioRingBuffer wrap-around preserves exact sample order across many cycles', () => {
  const ring = AudioRingBuffer.create(8, 2);
  let nextFrameId = 0;

  // Run enough push/pull chunks of size 5 to cross both the 8-frame buffer boundary
  // and the 16-frame double-modulo cursor wrap multiple times.
  for (let round = 0; round < 10; round++) {
    const chunk = makeStereoFrames(nextFrameId, 5);
    assert.equal(ring.pushSamples(chunk), 5);
    assert.equal(ring.availableFrames(), 5);

    const out = new Float32Array(5 * 2);
    assert.equal(ring.pullSamples(out), 5);
    assert.deepEqual(Array.from(out), Array.from(chunk));
    assert.equal(ring.availableFrames(), 0);

    nextFrameId += 5;
  }

  // Also test filling to exact capacity while cursors are mid-buffer.
  const seed = makeStereoFrames(100, 3);
  assert.equal(ring.pushSamples(seed), 3);
  const discard = new Float32Array(3 * 2);
  assert.equal(ring.pullSamples(discard), 3);

  const full = makeStereoFrames(200, 8);
  assert.equal(ring.pushSamples(full), 8);
  assert.equal(ring.availableFrames(), 8);
  assert.equal(ring.freeFrames(), 0);

  const outFull = new Float32Array(8 * 2);
  assert.equal(ring.pullSamples(outFull), 8);
  assert.deepEqual(Array.from(outFull), Array.from(full));
  assert.equal(ring.underruns, 0);
  assert.equal(ring.overflows, 0);
});

test('AudioRingBuffer overflow backpressure drops excess frames without clobbering unread audio', () => {
  const ring = AudioRingBuffer.create(6, 2);

  const first = makeStereoFrames(0, 4);
  assert.equal(ring.pushSamples(first), 4);
  assert.equal(ring.overflows, 0);
  assert.equal(ring.droppedFrames, 0);

  // Only 2 frames free; pushing 5 should write 2 and drop 3.
  const second = makeStereoFrames(4, 5);
  const written = ring.pushSamples(second);
  assert.equal(written, 2);
  assert.equal(ring.availableFrames(), 6);
  assert.equal(ring.overflows, 1);
  assert.equal(ring.droppedFrames, 3);

  // Pushing when completely full writes 0 and increments counters again.
  assert.equal(ring.pushSamples(makeStereoFrames(20, 2)), 0);
  assert.equal(ring.overflows, 2);
  assert.equal(ring.droppedFrames, 5);

  // Verify the 6 readable frames are strictly the original 4 + first 2 of `second`.
  const pulled = new Float32Array(6 * 2);
  assert.equal(ring.pullSamples(pulled), 6);
  const expected = new Float32Array(6 * 2);
  expected.set(first, 0);
  expected.set(second.subarray(0, 4), 8);
  assert.deepEqual(Array.from(pulled), Array.from(expected));
});

test('AudioRingBuffer underrun zero-fills tail and increments underrun counters', () => {
  const ring = AudioRingBuffer.create(8, 2);
  const partial = makeStereoFrames(0, 3);
  assert.equal(ring.pushSamples(partial), 3);

  // Pre-fill destination with sentinel garbage to ensure zero-fill overwrites it.
  const out = new Float32Array(5 * 2);
  out.fill(99.0);

  const framesRead = ring.pullSamples(out);
  assert.equal(framesRead, 3);
  assert.equal(ring.availableFrames(), 0);
  assert.equal(ring.underruns, 1);
  assert.equal(ring.underrunFrames, 2);

  assert.deepEqual(Array.from(out.subarray(0, 6)), Array.from(partial));
  assert.deepEqual(Array.from(out.subarray(6, 10)), [0, 0, 0, 0]);

  // Pulling from an empty buffer zero-fills everything and bumps underrun counters again.
  const emptyOut = new Float32Array(4 * 2);
  emptyOut.fill(-7.5);
  assert.equal(ring.pullSamples(emptyOut), 0);
  assert.deepEqual(Array.from(emptyOut), [0, 0, 0, 0, 0, 0, 0, 0]);
  assert.equal(ring.underruns, 2);
  assert.equal(ring.underrunFrames, 6);
});

test('AudioMixerWorker handles init, pushPcm, setVolume/mute, and stats messages', () => {
  const replies = [];
  const worker = new AudioMixerWorker((msg) => replies.push(msg));

  const ringSab = new SharedArrayBuffer(AudioRingBuffer.byteLength(16, 2));
  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    ringSab,
  });

  assert.equal(replies.length, 1);
  assert.deepEqual(replies[0], {
    ready: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 16,
  });

  // Push PCM through worker message and read via shared ring view.
  const pcm = new Float32Array([1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.8, -0.8]);
  worker.handleMessage({ pushPcm: true, samples: pcm });

  // Set volume to 0.5 and verify scaled output.
  worker.handleMessage({ setVolume: true, volume: 0.5 });
  const outScaled = new Float32Array(2 * 2);
  assert.equal(worker.pullSamples(outScaled), 2);
  assert.deepEqual(Array.from(outScaled), [0.5, -0.5, 0.25, -0.25]);

  // Mute via setVolume message and verify silence while preserving volume.
  worker.handleMessage({ setVolume: true, muted: true });
  const outMuted = new Float32Array(2 * 2);
  outMuted.fill(1.0);
  assert.equal(worker.pullSamples(outMuted), 2);
  assert.deepEqual(Array.from(outMuted), [0, 0, 0, 0]);

  // Trigger an underrun via pullSamples.
  const outUnderrun = new Float32Array(3 * 2);
  assert.equal(worker.pullSamples(outUnderrun), 0);

  // Query stats with message id.
  worker.handleMessage({ stats: true, id: 77 });
  const statsReply = replies[replies.length - 1];
  assert.equal(statsReply.id, 77);
  assert.equal(statsReply.stats.initialized, true);
  assert.equal(statsReply.stats.sampleRate, 48000);
  assert.equal(statsReply.stats.channels, 2);
  assert.equal(statsReply.stats.capacityFrames, 16);
  assert.equal(statsReply.stats.availableFrames, 0);
  assert.equal(statsReply.stats.volume, 0.5);
  assert.equal(statsReply.stats.muted, true);
  assert.equal(statsReply.stats.effectiveGain, 0);
  assert.equal(statsReply.stats.underruns, 1);
  assert.equal(statsReply.stats.underrunFrames, 3);
  assert.equal(statsReply.stats.totalPushedFrames, 4);
  assert.equal(statsReply.stats.totalPulledFrames, 4);

  // Clamp volume bounds cleanly.
  worker.handleMessage({ setVolume: true, volume: 2.5, muted: false, id: 78 });
  assert.equal(replies[replies.length - 1].volume, 1.0);
  assert.equal(replies[replies.length - 1].muted, false);
  worker.handleMessage({ setVolume: true, volume: -0.4, id: 79 });
  assert.equal(replies[replies.length - 1].volume, 0.0);

  worker.destroy();
});

test('AudioMixerWorker default init (loader.js compatibility) works without ringSab', () => {
  const replies = [];
  const worker = new AudioMixerWorker((msg) => replies.push(msg));
  worker.handleMessage({ init: true });

  assert.equal(replies.length, 1);
  assert.equal(replies[0].ready, true);
  assert.equal(replies[0].sampleRate, 48000);
  assert.equal(replies[0].channels, 2);
  assert.ok(replies[0].capacityFrames > 0);
  worker.destroy();
});

// ---------------------------------------------------------------------------
// W84: Headless & Autoplay-Suspended Drain Mode
// ---------------------------------------------------------------------------

test('W84 headless-drain mode drains submitted PCM buffers cleanly without stalling and updates framesDrained / outputState', () => {
  const replies = [];
  let fakeNow = 1000;
  const scheduledTimers = [];
  let clearedCount = 0;

  const worker = new AudioMixerWorker((msg) => replies.push(msg), {
    now: () => fakeNow,
    drainIntervalMs: 20,
    setIntervalFn: (fn, ms) => {
      const handle = { fn, ms, unrefCalled: false, unref() { this.unrefCalled = true; } };
      scheduledTimers.push(handle);
      return handle;
    },
    clearIntervalFn: () => {
      clearedCount += 1;
    },
  });

  const ringSab = new SharedArrayBuffer(AudioRingBuffer.byteLength(512, 2));
  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    ringSab,
    headless: true,
  });

  assert.equal(worker.outputState, 'headless-drain');
  assert.equal(scheduledTimers.length, 1);
  assert.equal(scheduledTimers[0].ms, 20);
  assert.equal(scheduledTimers[0].unrefCalled, true);

  // Submit 8 consecutive 256-frame PCM packets (2048 frames total > 512-frame ring).
  // Headless pre-drain prevents ring overflow/deadlock while advancing framesDrained.
  for (let i = 0; i < 8; i++) {
    worker.handleMessage({
      pushPcm: true,
      id: i + 1,
      samples: makeStereoFrames(i * 256, 256),
    });
    const ack = replies[replies.length - 1];
    assert.equal(ack.id, i + 1);
    assert.equal(ack.pushedFrames, 256);
    assert.equal(ack.droppedFrames, 0);
  }

  // 2048 frames pushed into 512-frame ring -> 1536 pre-drained, 512 still buffered.
  assert.equal(worker.framesDrained, 1536);
  assert.equal(worker.ring.availableFrames(), 512);
  assert.equal(worker.getStats().overruns, 0);
  assert.equal(worker.getStats().underruns, 0);

  // Trigger manual drain message to drain remaining frames cleanly.
  worker.handleMessage({ drain: true, id: 99 });
  const drainAck = replies[replies.length - 1];
  assert.equal(drainAck.id, 99);
  assert.equal(drainAck.drainedFrames, 512);
  assert.equal(drainAck.framesDrained, 2048);
  assert.equal(drainAck.outputState, 'headless-drain');
  assert.equal(drainAck.availableFrames, 0);

  // Direct SAB producer (simulating C++ Xbox Audio HAL thread writing straight to SAB)
  // drained at bounded 48 kHz real-time rate via timer tick.
  const guestProducer = new AudioRingBuffer(ringSab);
  assert.equal(guestProducer.pushSamples(makeStereoFrames(0, 512)), 512);
  assert.equal(guestProducer.freeFrames(), 0);

  // Advance clock by 5ms -> at 48 kHz, budget is floor(48000 * 5 / 1000) = 240 frames.
  fakeNow = 1005;
  scheduledTimers[0].fn();
  assert.equal(guestProducer.availableFrames(), 512 - 240);
  assert.equal(guestProducer.freeFrames(), 240);
  assert.equal(worker.framesDrained, 2048 + 240);

  // Advance clock by 5000ms (simulated tab stall) -> clamped to MAX_DRAIN_ELAPSED_MS (100ms),
  // cleanly draining the remaining 272 frames without looping or throwing.
  fakeNow = 6005;
  scheduledTimers[0].fn();
  assert.equal(guestProducer.availableFrames(), 0);
  assert.equal(worker.framesDrained, 2048 + 512);

  // Transition to 'active' stops drain timer and prevents stealing AudioWorklet frames.
  worker.handleMessage({ setOutputState: true, outputState: 'active', id: 100 });
  assert.equal(replies[replies.length - 1].outputState, 'active');
  assert.equal(worker.outputState, 'active');
  assert.equal(clearedCount, 1);

  assert.equal(guestProducer.pushSamples(makeStereoFrames(0, 128)), 128);
  assert.equal(worker.drain(128), 0);
  assert.equal(guestProducer.availableFrames(), 128);

  // In 'active' mode, pushing beyond capacity records real overruns cleanly.
  worker.handleMessage({ pushPcm: true, id: 101, samples: makeStereoFrames(0, 500) });
  const activeOverflowAck = replies[replies.length - 1];
  assert.equal(activeOverflowAck.pushedFrames, 384);
  assert.equal(activeOverflowAck.droppedFrames, 116);
  assert.equal(worker.getStats().overruns, 1);
  assert.equal(worker.getStats().overflows, 1);

  // Transition to autoplay-locked 'suspended' restarts bounded draining.
  worker.handleMessage({ setOutputState: true, outputState: 'suspended', id: 102 });
  assert.equal(worker.outputState, 'suspended');
  assert.equal(scheduledTimers.length, 2);

  worker.handleMessage({ drain: true, maxFrames: 200, id: 103 });
  assert.equal(replies[replies.length - 1].drainedFrames, 200);
  assert.equal(replies[replies.length - 1].outputState, 'suspended');
  assert.equal(guestProducer.availableFrames(), 312);

  worker.handleMessage({ stop: true, id: 104 });
  assert.equal(replies[replies.length - 1].stopped, true);
  assert.equal(clearedCount, 2);
});

// ---------------------------------------------------------------------------
// W84: SAB-backed PCM Submission Safety
// ---------------------------------------------------------------------------

test('W84 SAB-backed PCM submissions copy cleanly with zero DataCloneError and isolate guest mutations', () => {
  const replies = [];
  const worker = new AudioMixerWorker((msg) => replies.push(msg), {
    outputState: 'active',
    autoDrain: false,
  });

  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 64,
    outputState: 'active',
  });

  // 1. Float32Array view at non-zero byteOffset inside a SharedArrayBuffer (like Emscripten HEAPF32).
  const wasmHeapSab = new SharedArrayBuffer(256);
  const sabFloatView = new Float32Array(wasmHeapSab, 32, 8); // 4 stereo frames at offset 32
  const expected1 = [0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0];
  sabFloatView.set(expected1);

  worker.handleMessage({ pushPcm: true, id: 10, samples: sabFloatView });
  assert.equal(replies[replies.length - 1].pushedFrames, 4);
  assert.equal(replies[replies.length - 1].droppedFrames, 0);

  // Mutate the source SAB immediately after pushPcm — ring data must remain uncorrupted.
  sabFloatView.fill(-0.99);

  const out1 = new Float32Array(8);
  assert.equal(worker.pullSamples(out1), 4);
  assert.deepEqual(Array.from(out1), expected1);

  // 2. Raw SharedArrayBuffer passed directly as msg.samples.
  const rawSab = new SharedArrayBuffer(4 * 2 * Float32Array.BYTES_PER_ELEMENT);
  const rawView = new Float32Array(rawSab);
  rawView.set([0.125, -0.125, 0.25, -0.25, 0.375, -0.375, 0.5, -0.5]);

  worker.handleMessage({ pushPcm: true, id: 11, samples: rawSab });
  assert.equal(replies[replies.length - 1].pushedFrames, 4);

  const out2 = new Float32Array(8);
  assert.equal(worker.pullSamples(out2), 4);
  assert.deepEqual(Array.from(out2), Array.from(rawView));

  // 3. Unaligned Uint8Array view over SharedArrayBuffer (byteOffset = 3, non-multiple of 4).
  const unalignedSab = new SharedArrayBuffer(64);
  const unalignedBytes = new Uint8Array(unalignedSab, 3, 16); // 2 stereo frames (4 float32s)
  const srcFloats = new Float32Array([0.25, 0.5, -0.25, -0.5]);
  unalignedBytes.set(new Uint8Array(srcFloats.buffer));

  worker.handleMessage({ pushPcm: true, id: 12, samples: unalignedBytes });
  assert.equal(replies[replies.length - 1].pushedFrames, 2);

  const out3 = new Float32Array(4);
  assert.equal(worker.pullSamples(out3), 2);
  assert.deepEqual(Array.from(out3), Array.from(srcFloats));

  // 4. Non-finite samples (NaN / +-Infinity) in SAB-backed view are sanitized to 0.0.
  const nanSab = new SharedArrayBuffer(16);
  const nanView = new Float32Array(nanSab);
  nanView.set([NaN, 0.5, Infinity, -Infinity]);

  worker.handleMessage({ pushPcm: true, id: 13, samples: nanView });
  assert.equal(replies[replies.length - 1].pushedFrames, 2);
  const outNan = new Float32Array(4);
  assert.equal(worker.pullSamples(outNan), 2);
  assert.deepEqual(Array.from(outNan), [0, 0.5, 0, 0]);

  const stats = worker.getStats();
  assert.equal(stats.sabCopiedPackets, 4);
  assert.equal(stats.sanitizedSamples, 3);

  // 5. clonePcmForTransfer produces a transferable non-SAB ArrayBuffer that survives structuredClone transfer!
  const cloned = clonePcmForTransfer(sabFloatView);
  assert.equal(cloned.ok, true);
  assert.equal(cloned.copiedFromSab, true);
  assert.ok(cloned.pcm.buffer instanceof ArrayBuffer);
  assert.equal(cloned.pcm.buffer instanceof SharedArrayBuffer, false);
  assert.equal(cloned.transfer.length, 1);
  assert.equal(cloned.transfer[0], cloned.pcm.buffer);

  // Real structuredClone with transfer list MUST succeed with zero DataCloneError.
  const transferred = structuredClone(cloned.pcm, { transfer: cloned.transfer });
  assert.equal(transferred.length, 8);
  assert.equal(cloned.pcm.buffer.detached, true);

  // 6. sanitizeTransferList strips SharedArrayBuffers, detached ArrayBuffers, and TypedArrays.
  const validAb = new ArrayBuffer(16);
  const detachedAb = makeDetachedArrayBuffer(16);
  const filtered = sanitizeTransferList([
    wasmHeapSab,
    detachedAb,
    new Float32Array(validAb),
    validAb,
    validAb,
    null,
  ]);
  assert.deepEqual(filtered, [validAb]);
});

// ---------------------------------------------------------------------------
// W84: Malformed / Oversized / Detached / Null Payloads Rejected Safely
// ---------------------------------------------------------------------------

test('W84 malformed, oversized, detached, and null payloads are rejected safely without breaking subsequent valid packets', () => {
  const replies = [];
  const worker = new AudioMixerWorker((msg) => replies.push(msg), {
    outputState: 'active',
    autoDrain: false,
    maxPcmBytes: 1024, // 128 stereo frames max for deterministic oversized check
  });

  // Pre-init pushPcm returns structured error without throwing.
  worker.handleMessage({ pushPcm: true, id: 1, samples: makeStereoFrames(0, 4) });
  assert.match(replies[replies.length - 1].error, /not initialized/i);

  // Invalid init payloads (non-SAB ringSab, too-small SAB, corrupt negative-capacity SAB) do not throw.
  worker.handleMessage({ init: true, id: 2, ringSab: new ArrayBuffer(128) });
  assert.equal(replies[replies.length - 1].ready, false);
  assert.match(replies[replies.length - 1].error, /SharedArrayBuffer/i);

  worker.handleMessage({ init: true, id: 3, ringSab: new SharedArrayBuffer(8) });
  assert.match(replies[replies.length - 1].error, /too small/i);

  const corruptSab = new SharedArrayBuffer(64);
  Atomics.store(new Int32Array(corruptSab, 0, 8), 2, -42);
  worker.handleMessage({ init: true, id: 4, ringSab: corruptSab });
  assert.ok(typeof replies[replies.length - 1].error === 'string');

  // Now perform a valid init.
  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 32,
    outputState: 'active',
  });
  assert.equal(replies[replies.length - 1].ready, true);

  // Throw every category of malformed / null / detached / oversized message at handleMessage.
  assert.doesNotThrow(() => {
    worker.handleMessage(null);
    worker.handleMessage(undefined);
    worker.handleMessage(42);
    worker.handleMessage('pushPcm');
    worker.handleMessage([1, 2, 3]);

    // Object with throwing getter
    const hostile = {};
    Object.defineProperty(hostile, 'pushPcm', {
      get() {
        throw new Error('hostile getter boom');
      },
    });
    worker.handleMessage(hostile);

    // Null / missing / wrong-type samples
    worker.handleMessage({ pushPcm: true, id: 10, samples: null });
    worker.handleMessage({ pushPcm: true, id: 11, samples: undefined });
    worker.handleMessage({ pushPcm: true, id: 12, samples: 'not-pcm' });
    worker.handleMessage({ pushPcm: true, id: 13, samples: { length: 8 } });
    worker.handleMessage({ pushPcm: true, id: 14, samples: [0.1, 'bad', 0.2, 0.3] });

    // Detached ArrayBuffer and TypedArray view over detached ArrayBuffer
    const detachedRaw = makeDetachedArrayBuffer(32);
    worker.handleMessage({ pushPcm: true, id: 15, samples: detachedRaw });

    const abForView = new ArrayBuffer(32);
    const detachedView = new Float32Array(abForView);
    if (typeof abForView.transfer === 'function') {
      abForView.transfer();
    } else {
      structuredClone(abForView, { transfer: [abForView] });
    }
    worker.handleMessage({ pushPcm: true, id: 16, samples: detachedView });

    // Oversized Float32Array (> 1024 bytes), oversized SAB, and oversized JS array
    worker.handleMessage({ pushPcm: true, id: 17, samples: new Float32Array(512) });
    worker.handleMessage({ pushPcm: true, id: 18, samples: new SharedArrayBuffer(2048) });
    worker.handleMessage({ pushPcm: true, id: 19, samples: new Array(512).fill(0) });

    // Default max limit check via extractSafePcmSamples
    const hugeCheck = extractSafePcmSamples(new Uint8Array(DEFAULT_MAX_PCM_BYTES + 16));
    assert.equal(hugeCheck.ok, false);
    assert.match(hugeCheck.error, /exceeds max/i);

    // Malformed setVolume / setMute values
    worker.handleMessage({ setVolume: true, volume: NaN, muted: 'not-bool', id: 20 });
  });

  // Verify error replies were returned for bad pushPcm ids 10..19.
  for (let id = 10; id <= 19; id++) {
    const reply = replies.find((r) => r.id === id);
    assert.ok(reply, `Expected structured error reply for malformed packet id=${id}`);
    assert.equal(typeof reply.error, 'string');
  }

  // Subsequent valid audio packet MUST succeed cleanly with bit-exact samples!
  const validPcm = makeStereoFrames(0, 6);
  worker.handleMessage({ pushPcm: true, id: 200, samples: validPcm });
  const validAck = replies[replies.length - 1];
  assert.equal(validAck.id, 200);
  assert.equal(validAck.pushedFrames, 6);
  assert.equal(validAck.droppedFrames, 0);

  const pulled = new Float32Array(6 * 2);
  assert.equal(worker.pullSamples(pulled), 6);
  assert.deepEqual(Array.from(pulled), Array.from(validPcm));
  assert.ok(worker.getStats().rejectedPackets >= 16);
});

// ---------------------------------------------------------------------------
// W84: Sanitized audio-stats Telemetry Round-Trip
// ---------------------------------------------------------------------------

test('W84 audio-stats telemetry request/response round-trip and strict sanitization', () => {
  const replies = [];
  let fakeNow = 1700000123456;
  const worker = new AudioMixerWorker((msg) => replies.push(msg), {
    now: () => fakeNow,
    autoDrain: false,
  });

  // Pre-init stats request works cleanly without throwing.
  worker.handleMessage({ stats: true, id: 'pre-init' });
  const preReply = replies[replies.length - 1];
  assert.equal(preReply.type, 'audio-stats');
  assert.equal(preReply.id, 'pre-init');
  assert.equal(preReply.timestamp, 1700000123456);
  assert.equal(preReply.stats.initialized, false);
  assert.equal(preReply.stats.outputState, 'headless-drain');
  assert.equal(preReply.stats.framesDrained, 0);
  assert.equal(preReply.stats.underruns, 0);
  assert.equal(preReply.stats.overruns, 0);

  // Initialize in headless-drain mode, submit + drain + trigger underrun & malformed rejection.
  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 16,
    outputState: 'headless-drain',
  });

  worker.handleMessage({ pushPcm: true, samples: makeStereoFrames(0, 10) });
  worker.handleMessage({ drain: true, maxFrames: 6 });
  // Pull remaining 4 + trigger 2-frame underrun.
  worker.pullSamples(new Float32Array(6 * 2));
  // Trigger a malformed rejection.
  worker.handleMessage({ pushPcm: true, samples: null });

  // Request stats via { stats: true, id } AND { type: 'request-audio-stats', id }.
  fakeNow = 1700000123999;
  worker.handleMessage({ stats: true, id: 505 });
  const s1 = replies[replies.length - 1];
  assert.equal(s1.type, 'audio-stats');
  assert.equal(s1.id, 505);
  assert.equal(s1.timestamp, 1700000123999);
  assert.equal(s1.stats.initialized, true);
  assert.equal(s1.stats.outputState, 'headless-drain');
  assert.equal(s1.stats.sampleRate, 48000);
  assert.equal(s1.stats.channels, 2);
  assert.equal(s1.stats.capacityFrames, 16);
  assert.equal(s1.stats.availableFrames, 0);
  assert.equal(s1.stats.freeFrames, 16);
  assert.equal(s1.stats.framesDrained, 6);
  assert.equal(s1.stats.underruns, 1);
  assert.equal(s1.stats.underrunFrames, 2);
  assert.equal(s1.stats.overruns, 0);
  assert.equal(s1.stats.totalPushedFrames, 10);
  assert.equal(s1.stats.totalPulledFrames, 4);
  assert.equal(s1.stats.rejectedPackets, 1);

  worker.handleMessage({ type: 'request-audio-stats', id: 'telemetry-req-2' });
  const s2 = replies[replies.length - 1];
  assert.equal(s2.type, 'audio-stats');
  assert.equal(s2.id, 'telemetry-req-2');
  assert.deepEqual(s2.stats, s1.stats);

  // Malicious non-primitive id (e.g. SharedArrayBuffer) must NOT leak back in audio-stats reply.
  worker.handleMessage({ stats: true, id: new SharedArrayBuffer(16) });
  const s3 = replies[replies.length - 1];
  assert.equal(s3.type, 'audio-stats');
  assert.equal(s3.id, undefined);

  // Synthetic dirty object passed to buildAudioTelemetrySnapshot must be strictly sanitized.
  const dirtySnap = buildAudioTelemetrySnapshot(
    {
      initialized: 1,
      outputState: 'C:\\Users\\secret\\game_scarlett.exe',
      sampleRate: Infinity,
      channels: -3,
      capacityFrames: NaN,
      availableFrames: -10,
      freeFrames: Infinity,
      volume: 42,
      muted: false,
      framesDrained: -5,
      underruns: NaN,
      overruns: -1,
      overflows: 3,
      hostPath: '/Users/secret/gta6.rpf',
      rawSab: new SharedArrayBuffer(32),
      pcmDump: new Float32Array([1, 2, 3]),
      stack: 'Error: leak',
    },
    { timestamp: 9001 }
  );

  assert.equal(dirtySnap.type, 'audio-stats');
  assert.equal(dirtySnap.timestamp, 9001);
  assert.equal(dirtySnap.stats.outputState, 'headless-drain');
  assert.equal(dirtySnap.stats.sampleRate, 48000);
  assert.equal(dirtySnap.stats.channels, 2);
  assert.equal(dirtySnap.stats.volume, 1.0);
  assert.equal(dirtySnap.stats.framesDrained, 0);
  assert.equal(dirtySnap.stats.underruns, 0);
  assert.equal(dirtySnap.stats.overruns, 3);

  const json = JSON.stringify(dirtySnap);
  assert.equal(json.includes('secret'), false);
  assert.equal(json.includes('hostPath'), false);
  assert.equal(json.includes('rawSab'), false);
  assert.equal(json.includes('pcmDump'), false);
  assert.equal(json.includes('stack'), false);
});

// ---------------------------------------------------------------------------
// W84: Classic Dedicated Web Worker (importScripts) Sandbox Compatibility
// ---------------------------------------------------------------------------

test('W84 audio_worker.js runs cleanly in a classic Dedicated Worker context without uncaught errors', () => {
  const posted = [];
  const workerSelf = {
    onmessage: null,
    postMessage(msg, transfer) {
      if (transfer && transfer.some((t) => t instanceof SharedArrayBuffer)) {
        const err = new Error('DataCloneError: SharedArrayBuffer cannot be transferred');
        err.name = 'DataCloneError';
        throw err;
      }
      posted.push(msg);
    },
  };

  const sandbox = {
    self: workerSelf,
    globalThis: null,
    SharedArrayBuffer,
    ArrayBuffer,
    Int32Array,
    Float32Array,
    Uint8Array,
    Atomics,
    Math,
    Number,
    Object,
    Array,
    Set,
    Error,
    TypeError,
    RangeError,
    Date,
    setInterval: () => 1,
    clearInterval: () => {},
  };
  sandbox.globalThis = sandbox;
  const ctx = vm.createContext(sandbox);

  const workerPath = path.resolve(__dirname, '../web/workers/audio_worker.js');
  vm.runInContext(fs.readFileSync(workerPath, 'utf8'), ctx, { filename: workerPath });

  assert.equal(typeof workerSelf.onmessage, 'function');
  assert.ok(workerSelf.playGta6AudioWorker);

  // Boot init, SAB push, malformed packet, drain, stats, stop inside classic worker.
  workerSelf.onmessage({ data: { init: true, capacityFrames: 32 } });
  assert.equal(posted[0].ready, true);

  const sab = new SharedArrayBuffer(16 * 2 * 4);
  new Float32Array(sab).set(makeStereoFrames(0, 16));
  workerSelf.onmessage({ data: { pushPcm: true, id: 1, samples: new Float32Array(sab) } });
  assert.equal(posted[1].pushedFrames, 16);

  // Malformed event & null event must never throw in self.onmessage.
  assert.doesNotThrow(() => {
    workerSelf.onmessage(null);
    workerSelf.onmessage({ data: null });
    workerSelf.onmessage({ data: { pushPcm: true, id: 2, samples: 'corrupt' } });
  });

  workerSelf.onmessage({ data: { drain: true, id: 3 } });
  assert.equal(posted[posted.length - 1].drainedFrames, 16);

  workerSelf.onmessage({ data: { stats: true, id: 4 } });
  const statsMsg = posted[posted.length - 1];
  assert.equal(statsMsg.type, 'audio-stats');
  assert.equal(statsMsg.id, 4);
  assert.equal(statsMsg.stats.outputState, 'headless-drain');
  assert.equal(statsMsg.stats.framesDrained, 16);
  assert.equal(statsMsg.stats.sabCopiedPackets, 1);

  workerSelf.onmessage({ data: { stop: true, id: 5 } });
  assert.equal(posted[posted.length - 1].stopped, true);
});

// ---------------------------------------------------------------------------
// W116: Underrun De-Click Fade-Out / Fade-In & Fill-Level Telemetry
// ---------------------------------------------------------------------------

function makeConstantStereoFrames(frameCount, value) {
  const out = new Float32Array(frameCount * 2);
  out.fill(value);
  return out;
}

test('W116 steady-state playback with full ring produces 100% bit-exact samples and zero fade counters', () => {
  const worker = new AudioMixerWorker(null, {
    outputState: 'active',
    autoDrain: false,
    declickFrames: DEFAULT_DECLICK_FRAMES,
  });
  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 512,
    outputState: 'active',
  });

  // Push and pull multiple steady-state blocks with non-trivial varying values.
  for (let round = 0; round < 4; round++) {
    const input = makeStereoFrames(round * 128, 128);
    worker.handleMessage({ pushPcm: true, samples: input });
    const output = new Float32Array(128 * 2);
    const pulled = worker.pullSamples(output);
    assert.equal(pulled, 128);
    assert.deepEqual(Array.from(output), Array.from(input));
  }

  const stats = worker.getStats();
  assert.equal(stats.underrunCount, 0);
  assert.equal(stats.underruns, 0);
  assert.equal(stats.fadeOutCount, 0);
  assert.equal(stats.fadeInCount, 0);
  assert.equal(stats.declickFrames, DEFAULT_DECLICK_FRAMES);
  worker.destroy();
});

test('W116 underrun on constant +0.8 waveform fades smoothly to 0.0 and recovery fades smoothly back to +0.8', () => {
  const fadeFrames = 32;
  const amp = Math.fround(0.8);
  const maxAllowedStep = amp / fadeFrames + 1e-5;

  const worker = new AudioMixerWorker(null, {
    outputState: 'active',
    autoDrain: false,
    declickFrames: fadeFrames,
  });
  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 512,
    outputState: 'active',
  });

  // 1. Steady-state pull: bit-exact +0.8 across all 128 stereo frames.
  worker.handleMessage({ pushPcm: true, samples: makeConstantStereoFrames(128, amp) });
  const steadyOut = new Float32Array(128 * 2);
  assert.equal(worker.pullSamples(steadyOut), 128);
  for (let i = 0; i < steadyOut.length; i++) {
    assert.equal(steadyOut[i], amp);
  }

  // 2. Partial underrun: only 64 frames available when 128 requested.
  worker.handleMessage({ pushPcm: true, samples: makeConstantStereoFrames(64, amp) });
  const partialUnderrunOut = new Float32Array(128 * 2);
  partialUnderrunOut.fill(99.0);
  assert.equal(worker.pullSamples(partialUnderrunOut), 64);

  // Leading frames remain at +0.8, tail ramps monotonically to 0.0 with bounded step.
  assert.equal(partialUnderrunOut[0], amp);
  assert.equal(partialUnderrunOut[(128 - 1) * 2], 0);
  assert.equal(partialUnderrunOut[(128 - 1) * 2 + 1], 0);
  let maxStepOut = 0;
  for (let f = 1; f < 128; f++) {
    for (let c = 0; c < 2; c++) {
      const prev = partialUnderrunOut[(f - 1) * 2 + c];
      const curr = partialUnderrunOut[f * 2 + c];
      assert.ok(curr <= prev + 1e-6, `Expected monotonic fade-out at frame ${f} ch ${c}: ${prev} -> ${curr}`);
      const step = Math.abs(curr - prev);
      if (step > maxStepOut) maxStepOut = step;
    }
  }
  assert.ok(
    maxStepOut <= maxAllowedStep,
    `Expected fade-out max adjacent step <= ${maxAllowedStep}, got ${maxStepOut}`
  );
  assert.ok(maxStepOut > 0, 'Expected non-zero smooth ramp steps during fade-out');

  let stats = worker.getStats();
  assert.equal(stats.underrunCount, 1);
  assert.equal(stats.underruns, 1);
  assert.equal(stats.fadeOutCount, 1);
  assert.equal(stats.fadeInCount, 0);

  // 3. Consecutive empty pull stays silent and does NOT double-count fadeOutCount.
  const silentOut = new Float32Array(128 * 2);
  silentOut.fill(42.0);
  assert.equal(worker.pullSamples(silentOut), 0);
  for (let i = 0; i < silentOut.length; i++) {
    assert.equal(silentOut[i], 0);
  }
  stats = worker.getStats();
  assert.equal(stats.underrunCount, 2);
  assert.equal(stats.fadeOutCount, 1);
  assert.equal(stats.fadeInCount, 0);

  // 4. Recovery after underrun: push 128 frames of +0.8 and verify smooth fade-in 0.0 -> +0.8.
  worker.handleMessage({ pushPcm: true, samples: makeConstantStereoFrames(128, amp) });
  const recoveryOut = new Float32Array(128 * 2);
  assert.equal(worker.pullSamples(recoveryOut), 128);

  assert.equal(recoveryOut[0], 0);
  assert.equal(recoveryOut[1], 0);
  assert.equal(recoveryOut[fadeFrames * 2], amp);
  assert.equal(recoveryOut[(128 - 1) * 2], amp);

  let maxStepIn = 0;
  for (let f = 1; f < 128; f++) {
    for (let c = 0; c < 2; c++) {
      const prev = recoveryOut[(f - 1) * 2 + c];
      const curr = recoveryOut[f * 2 + c];
      assert.ok(curr >= prev - 1e-6, `Expected monotonic fade-in at frame ${f} ch ${c}: ${prev} -> ${curr}`);
      const step = Math.abs(curr - prev);
      if (step > maxStepIn) maxStepIn = step;
    }
  }
  assert.ok(
    maxStepIn <= maxAllowedStep,
    `Expected fade-in max adjacent step <= ${maxAllowedStep}, got ${maxStepIn}`
  );
  assert.ok(maxStepIn > 0, 'Expected non-zero smooth ramp steps during fade-in');

  stats = worker.getStats();
  assert.equal(stats.underrunCount, 2);
  assert.equal(stats.fadeOutCount, 1);
  assert.equal(stats.fadeInCount, 1);

  // 5. Subsequent full steady-state pull returns to 100% bit-exact +0.8.
  worker.handleMessage({ pushPcm: true, samples: makeConstantStereoFrames(128, amp) });
  const postRecoveryOut = new Float32Array(128 * 2);
  assert.equal(worker.pullSamples(postRecoveryOut), 128);
  for (let i = 0; i < postRecoveryOut.length; i++) {
    assert.equal(postRecoveryOut[i], amp);
  }
  assert.equal(worker.getStats().fadeInCount, 1);
  worker.destroy();
});

test('W116 empty-ring underrun after steady-state and tight-tail underrun both ramp smoothly without abrupt steps', () => {
  const fadeFrames = 64;
  const amp = Math.fround(0.8);
  const maxAllowedStep = amp / fadeFrames + 1e-5;

  const ring = AudioRingBuffer.create(512, 2, { declickFrames: fadeFrames });

  // A. Steady-state pull followed immediately by a completely empty ring pull (0 available).
  assert.equal(ring.pushSamples(makeConstantStereoFrames(128, amp)), 128);
  const steady = new Float32Array(128 * 2);
  assert.equal(ring.pullSamples(steady), 128);
  assert.equal(steady[steady.length - 1], amp);

  const emptyUnderrun = new Float32Array(128 * 2);
  assert.equal(ring.pullSamples(emptyUnderrun), 0);

  // First sample steps smoothly from previous pull's +0.8 (NOT straight to 0.0).
  const boundaryStep = Math.abs(emptyUnderrun[0] - amp);
  assert.ok(boundaryStep <= maxAllowedStep, `Boundary step ${boundaryStep} exceeded ${maxAllowedStep}`);
  assert.ok(emptyUnderrun[0] > 0.7, `Expected smooth fade start near 0.8, got ${emptyUnderrun[0]}`);
  assert.equal(emptyUnderrun[(fadeFrames - 1) * 2], 0);
  assert.equal(emptyUnderrun[(128 - 1) * 2], 0);
  for (let f = 1; f < 128; f++) {
    const step = Math.abs(emptyUnderrun[f * 2] - emptyUnderrun[(f - 1) * 2]);
    assert.ok(step <= maxAllowedStep, `Empty underrun step ${step} at frame ${f} exceeded ${maxAllowedStep}`);
  }
  assert.equal(ring.underrunCount, 1);
  assert.equal(ring.fadeOutCount, 1);

  // B. Multi-chunk recovery (32 frames per pull < 64 declickFrames) ramps continuously across pulls.
  assert.equal(ring.pushSamples(makeConstantStereoFrames(96, amp)), 96);
  const rec1 = new Float32Array(32 * 2);
  const rec2 = new Float32Array(32 * 2);
  const rec3 = new Float32Array(32 * 2);
  assert.equal(ring.pullSamples(rec1), 32);
  assert.equal(ring.pullSamples(rec2), 32);
  assert.equal(ring.pullSamples(rec3), 32);
  assert.equal(ring.fadeInCount, 1);
  assert.equal(rec1[0], 0);
  assert.ok(Math.abs(rec2[0] - rec1[(32 - 1) * 2]) <= maxAllowedStep);
  assert.ok(Math.abs(rec3[0] - rec2[(32 - 1) * 2]) <= maxAllowedStep);
  assert.equal(rec3[0], amp);
  assert.equal(rec3[rec3.length - 1], amp);

  // C. Tight missing tail: 120 available / 128 requested (missing = 8 < fadeFrames = 64).
  assert.equal(ring.pushSamples(makeConstantStereoFrames(120, amp)), 120);
  const tightTail = new Float32Array(128 * 2);
  assert.equal(ring.pullSamples(tightTail), 120);
  assert.equal(tightTail[0], amp);
  assert.equal(tightTail[120 * 2], 0);
  assert.equal(tightTail[(128 - 1) * 2], 0);
  for (let f = 1; f < 128; f++) {
    const step = Math.abs(tightTail[f * 2] - tightTail[(f - 1) * 2]);
    assert.ok(step <= maxAllowedStep, `Tight tail step ${step} at frame ${f} exceeded ${maxAllowedStep}`);
  }
  assert.equal(ring.underrunCount, 2);
  assert.equal(ring.fadeOutCount, 2);
});

test('W116 fill-level telemetry tracks peakFillFrames, minFillFrames, avgFillRatio accurately and resets cleanly', () => {
  const replies = [];
  const worker = new AudioMixerWorker((msg) => replies.push(msg), {
    outputState: 'active',
    autoDrain: false,
    declickFrames: 16,
  });

  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 100,
    outputState: 'active',
  });

  // Initially all fill & underrun telemetry fields are zero.
  let s = worker.getStats();
  assert.equal(s.peakFillFrames, 0);
  assert.equal(s.minFillFrames, 0);
  assert.equal(s.avgFillRatio, 0);
  assert.equal(s.underrunCount, 0);
  assert.equal(s.fadeOutCount, 0);
  assert.equal(s.fadeInCount, 0);

  // Trajectory on 100-frame ring:
  // Push 80 -> fill = 80
  // Pull 60 -> fill = 20
  // Push 40 -> fill = 60
  // Pull 20 -> fill = 40
  // Observations = [80, 20, 60, 40] -> peak = 80, min = 20, avg = 50 / 100 = 0.5.
  worker.handleMessage({ pushPcm: true, samples: makeConstantStereoFrames(80, 0.5) });
  assert.equal(worker.pullSamples(new Float32Array(60 * 2)), 60);
  worker.handleMessage({ pushPcm: true, samples: makeConstantStereoFrames(40, 0.5) });
  assert.equal(worker.pullSamples(new Float32Array(20 * 2)), 20);

  s = worker.getStats();
  assert.equal(s.peakFillFrames, 80);
  assert.equal(s.minFillFrames, 20);
  assert.ok(Math.abs(s.avgFillRatio - 0.5) < 1e-9, `Expected avgFillRatio 0.5, got ${s.avgFillRatio}`);

  // Sanitized audio-stats message round-trip exposes all additive W116 fields.
  worker.handleMessage({ stats: true, id: 'w116-telemetry' });
  const snap = replies[replies.length - 1];
  assert.equal(snap.type, 'audio-stats');
  assert.equal(snap.id, 'w116-telemetry');
  assert.equal(snap.stats.peakFillFrames, 80);
  assert.equal(snap.stats.minFillFrames, 20);
  assert.ok(Math.abs(snap.stats.avgFillRatio - 0.5) < 1e-9);
  assert.equal(snap.stats.underrunCount, 0);
  assert.equal(snap.stats.fadeOutCount, 0);
  assert.equal(snap.stats.fadeInCount, 0);

  // Trigger an underrun + recovery so all counters are non-zero before reset checks.
  assert.equal(worker.pullSamples(new Float32Array(60 * 2)), 40);
  worker.handleMessage({ pushPcm: true, samples: makeConstantStereoFrames(32, 0.5) });
  assert.equal(worker.pullSamples(new Float32Array(32 * 2)), 32);

  s = worker.getStats();
  assert.equal(s.underrunCount, 1);
  assert.equal(s.fadeOutCount, 1);
  assert.equal(s.fadeInCount, 1);
  assert.equal(s.minFillFrames, 0);

  // 1. Clean reset via reset() / { reset: true } message.
  worker.handleMessage({ reset: true, id: 'w116-reset' });
  assert.equal(replies[replies.length - 1].reset, true);
  s = worker.getStats();
  assert.equal(s.underrunCount, 0);
  assert.equal(s.fadeOutCount, 0);
  assert.equal(s.fadeInCount, 0);
  assert.equal(s.peakFillFrames, 0);
  assert.equal(s.minFillFrames, 0);
  assert.equal(s.avgFillRatio, 0);

  // 2. Clean reset via stop().
  worker.handleMessage({ pushPcm: true, samples: makeConstantStereoFrames(50, 0.5) });
  assert.equal(worker.pullSamples(new Float32Array(80 * 2)), 50);
  assert.equal(worker.getStats().fadeOutCount, 1);
  worker.stop();
  s = worker.getStats();
  assert.equal(s.underrunCount, 0);
  assert.equal(s.fadeOutCount, 0);
  assert.equal(s.fadeInCount, 0);
  assert.equal(s.peakFillFrames, 0);
  assert.equal(s.minFillFrames, 0);
  assert.equal(s.avgFillRatio, 0);

  // 3. Clean reset via stream reconfigure (re-init with new capacity/declickFrames).
  worker.handleMessage({ pushPcm: true, samples: makeConstantStereoFrames(40, 0.5) });
  assert.equal(worker.pullSamples(new Float32Array(64 * 2)), 40);
  assert.equal(worker.getStats().underrunCount, 1);

  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 256,
    declickFrames: 64,
    outputState: 'active',
  });
  s = worker.getStats();
  assert.equal(s.capacityFrames, 256);
  assert.equal(s.declickFrames, 64);
  assert.equal(s.underrunCount, 0);
  assert.equal(s.fadeOutCount, 0);
  assert.equal(s.fadeInCount, 0);
  assert.equal(s.peakFillFrames, 0);
  assert.equal(s.minFillFrames, 0);
  assert.equal(s.avgFillRatio, 0);
  worker.destroy();
});

// ---------------------------------------------------------------------------
// W170: ITU-R BS.775 5.1 / 7.1 Downmix, 44.1<->48 kHz Resampling & Telemetry
// ---------------------------------------------------------------------------

test('W170 AudioMixerWorker downmixes 5.1 (6ch) and 7.1 (8ch) PCM to stereo with ITU-R BS.775 weights, soft-clip guard, and NaN isolation', () => {
  assert.equal(typeof softClipSample, 'function');
  assert.equal(typeof downmixToStereo, 'function');
  assert.equal(typeof resampleLinear, 'function');
  assert.ok(Math.abs(ITU_BS775_CENTER_GAIN - 0.70710678) < 1e-7);
  assert.ok(Math.abs(ITU_BS775_SURROUND_GAIN - 0.70710678) < 1e-7);

  const replies = [];
  const worker = new AudioMixerWorker((msg) => replies.push(msg), {
    outputState: 'active',
    autoDrain: false,
    declickFrames: 0,
  });

  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 256,
    outputState: 'active',
  });

  // 1. SAB-backed 5.1 packet (FL, FR, FC, LFE, BL, BR): 4 isolated frames.
  const sab51 = new SharedArrayBuffer(4 * 6 * Float32Array.BYTES_PER_ELEMENT);
  const view51 = new Float32Array(sab51);
  view51.set([
    0.5, 0.0, 0.0, 0.0, 0.0, 0.0, // FL
    0.0, 0.0, 0.8, 0.9, 0.0, 0.0, // FC + LFE (LFE omitted by default)
    0.0, 0.0, 0.0, 0.0, 0.6, 0.0, // BL
    0.0, 0.0, 0.0, 0.0, 0.0, 0.6, // BR
  ]);
  worker.handleMessage({ pushPcm: true, id: 1, channels: 6, samples: view51, activeVoices: 3 });
  assert.equal(replies[replies.length - 1].pushedFrames, 4);
  assert.equal(replies[replies.length - 1].droppedFrames, 0);

  const out51 = new Float32Array(4 * 2);
  assert.equal(worker.pullSamples(out51), 4);
  assert.ok(Math.abs(out51[0] - 0.5) < 1e-6 && out51[1] === 0.0);
  assert.ok(Math.abs(out51[2] - 0.8 * ITU_BS775_CENTER_GAIN) < 1e-6);
  assert.ok(Math.abs(out51[3] - 0.8 * ITU_BS775_CENTER_GAIN) < 1e-6);
  assert.ok(Math.abs(out51[4] - 0.6 * ITU_BS775_SURROUND_GAIN) < 1e-6 && out51[5] === 0.0);
  assert.ok(out51[6] === 0.0 && Math.abs(out51[7] - 0.6 * ITU_BS775_SURROUND_GAIN) < 1e-6);

  // 2. Full-scale 7.1 (8ch) packet triggers smooth soft-clip limiter instead of hard-clipping.
  const hot71 = new Float32Array([
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
  ]);
  worker.handleMessage({ pushPcm: true, id: 2, channels: 8, samples: hot71, activeVoices: 5 });
  assert.equal(replies[replies.length - 1].pushedFrames, 2);

  const out71 = new Float32Array(2 * 2);
  assert.equal(worker.pullSamples(out71), 2);
  assert.ok(out71[0] > 0.95 && out71[0] <= 1.0);
  assert.ok(out71[1] > 0.95 && out71[1] <= 1.0);
  assert.ok(out71[2] < -0.95 && out71[2] >= -1.0);
  assert.ok(out71[3] < -0.95 && out71[3] >= -1.0);

  // 3. NaN / +-Infinity in 5.1 center/surround channels does not poison FL/FR.
  const bad51 = new Float32Array([
    0.4, -0.3, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, 0.2,
  ]);
  worker.handleMessage({ pushPcm: true, id: 3, channels: 6, samples: bad51 });
  const outClean = new Float32Array(2);
  assert.equal(worker.pullSamples(outClean), 1);
  assert.ok(Math.abs(outClean[0] - 0.4) < 1e-6);
  assert.ok(Math.abs(outClean[1] - (-0.3 + 0.2 * ITU_BS775_SURROUND_GAIN)) < 1e-6);

  // 4. Unsupported channel counts are safely rejected without breaking subsequent packets.
  worker.handleMessage({ pushPcm: true, id: 4, channels: 3, samples: new Float32Array(9) });
  assert.match(replies[replies.length - 1].error, /Cannot convert/i);
  worker.handleMessage({ pushPcm: true, id: 5, channels: 9, samples: new Float32Array(18) });
  assert.match(replies[replies.length - 1].error, /Unsupported PCM channel count/i);

  const stats = worker.getStats();
  assert.equal(stats.downmixedFrames, 4 + 2 + 1);
  assert.equal(stats.clippedSamples, 4);
  assert.equal(stats.nanSanitizedSamples, 3);
  assert.equal(stats.sanitizedSamples, 3);
  assert.equal(stats.activeVoicePeak, 5);
  assert.equal(stats.sabCopiedPackets, 1);
  assert.equal(stats.rejectedPackets, 2);
  worker.destroy();
});

test('W170 AudioMixerWorker resamples 44.1 kHz <-> 48 kHz (including 5.1 Bink2 intro) smoothly and rejects invalid sample rates', () => {
  const replies = [];
  const worker = new AudioMixerWorker((msg) => replies.push(msg), {
    outputState: 'active',
    autoDrain: false,
    declickFrames: 0,
  });

  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 2048,
    outputState: 'active',
  });

  // 1. Combined Bink2 5.1 @ 44.1 kHz (441 frames = 10 ms) -> downmix + resample to 480 stereo frames @ 48 kHz.
  const bink51 = new Float32Array(441 * 6);
  for (let f = 0; f < 441; f++) {
    const tone = Math.sin((2 * Math.PI * 1000 * f) / 44100) * 0.5;
    bink51[f * 6 + 0] = tone; // FL
    bink51[f * 6 + 1] = -tone; // FR
    bink51[f * 6 + 2] = 0.0;
    bink51[f * 6 + 3] = 0.0;
    bink51[f * 6 + 4] = 0.0;
    bink51[f * 6 + 5] = 0.0;
  }
  worker.handleMessage({
    pushPcm: true,
    id: 10,
    channels: 6,
    sampleRate: 44100,
    samples: bink51,
  });
  const ack = replies[replies.length - 1];
  assert.equal(ack.id, 10);
  assert.equal(ack.pushedFrames, 480);
  assert.equal(ack.droppedFrames, 0);

  const pulled48 = new Float32Array(480 * 2);
  assert.equal(worker.pullSamples(pulled48), 480);
  let sumSqErr = 0;
  for (let f = 0; f < 478; f++) {
    const expected = Math.sin((2 * Math.PI * 1000 * f) / 48000) * 0.5;
    const dL = pulled48[f * 2] - expected;
    const dR = pulled48[f * 2 + 1] - -expected;
    sumSqErr += dL * dL + dR * dR;
  }
  const rms = Math.sqrt(sumSqErr / (478 * 2));
  assert.ok(rms < 0.005, `Combined 5.1@44.1k -> stereo@48k RMS error too high: ${rms}`);

  // 2. Reject invalid / out-of-range / extreme-ratio sampleRate safely.
  worker.handleMessage({ pushPcm: true, id: 11, sampleRate: 0, samples: new Float32Array(8) });
  assert.match(replies[replies.length - 1].error, /Invalid PCM sampleRate/i);
  worker.handleMessage({ pushPcm: true, id: 12, sampleRate: Number.NaN, samples: new Float32Array(8) });
  assert.match(replies[replies.length - 1].error, /Invalid PCM sampleRate/i);
  worker.handleMessage({ pushPcm: true, id: 13, sampleRate: 999999, samples: new Float32Array(8) });
  assert.match(replies[replies.length - 1].error, /Invalid PCM sampleRate/i);

  const stats48 = worker.getStats();
  assert.equal(stats48.downmixedFrames, 441);
  assert.equal(stats48.resampledFrames, 480);
  assert.equal(stats48.rejectedPackets, 3);

  // 3. Reverse direction: 48 kHz -> 44.1 kHz worker (480 frames -> exact 441 frames).
  worker.handleMessage({
    init: true,
    sampleRate: 44100,
    channels: 2,
    capacityFrames: 1024,
    outputState: 'active',
  });
  const src48 = new Float32Array(480 * 2);
  for (let f = 0; f < 480; f++) {
    src48[f * 2] = Math.sin((2 * Math.PI * 1000 * f) / 48000) * 0.4;
    src48[f * 2 + 1] = src48[f * 2];
  }
  worker.handleMessage({ pushPcm: true, id: 20, sampleRate: 48000, samples: src48 });
  assert.equal(replies[replies.length - 1].pushedFrames, 441);
  assert.equal(worker.getStats().resampledFrames, 441);
  worker.destroy();
});

test('W170 sanitized audio-stats telemetry exposes downmixedFrames, clippedSamples, nanSanitizedSamples, underrunCount, activeVoicePeak and resets cleanly', () => {
  const replies = [];
  const worker = new AudioMixerWorker((msg) => replies.push(msg), {
    outputState: 'active',
    autoDrain: false,
    declickFrames: 0,
  });

  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 64,
    outputState: 'active',
  });

  // Push a hot 5.1 frame with 1 NaN sample and activeVoices=6, then pull more than available to trigger 1 underrun.
  worker.handleMessage({
    pushPcm: true,
    channels: 6,
    samples: new Float32Array([1.0, 1.0, 1.0, Number.NaN, 1.0, 1.0]),
    activeVoices: 6,
  });
  const out = new Float32Array(4 * 2);
  assert.equal(worker.pullSamples(out), 1);

  worker.handleMessage({ stats: true, id: 'w170-stats' });
  const reply = replies[replies.length - 1];
  assert.equal(reply.type, 'audio-stats');
  assert.equal(reply.id, 'w170-stats');
  assert.equal(reply.stats.downmixedFrames, 1);
  assert.equal(reply.stats.clippedSamples, 2);
  assert.equal(reply.stats.nanSanitizedSamples, 1);
  assert.equal(reply.stats.underrunCount, 1);
  assert.equal(reply.stats.activeVoicePeak, 6);

  // reset() zeroes all W170 telemetry cleanly.
  worker.reset();
  const afterReset = worker.getSanitizedStats();
  assert.equal(afterReset.downmixedFrames, 0);
  assert.equal(afterReset.resampledFrames, 0);
  assert.equal(afterReset.clippedSamples, 0);
  assert.equal(afterReset.nanSanitizedSamples, 0);
  assert.equal(afterReset.underrunCount, 0);
  assert.equal(afterReset.activeVoicePeak, 0);

  // Dirty telemetry snapshot strictly sanitizes W170 fields and strips non-whitelisted fields.
  const dirty = buildAudioTelemetrySnapshot(
    {
      downmixedFrames: -5,
      resampledFrames: Infinity,
      clippedSamples: Number.NaN,
      sanitizedNonFiniteSamples: 7,
      underrunCount: 3,
      activeVoicePeak: 12.9,
      hostPath: 'C:\\secret\\bink2.bk2',
    },
    { timestamp: 1234 }
  );
  assert.equal(dirty.stats.downmixedFrames, 0);
  assert.equal(dirty.stats.resampledFrames, 0);
  assert.equal(dirty.stats.clippedSamples, 0);
  assert.equal(dirty.stats.nanSanitizedSamples, 7);
  assert.equal(dirty.stats.underrunCount, 3);
  assert.equal(dirty.stats.activeVoicePeak, 12);
  assert.equal(JSON.stringify(dirty).includes('bink2.bk2'), false);
  worker.destroy();
});

// ---------------------------------------------------------------------------
// W222: Zero-Alloc PCM Ring Forwarding, Overflow Drop Guards, Underrun Recovery & Telemetry
// ---------------------------------------------------------------------------

test('W222 interleaved<->planar conversion reuses scratch buffers with zero steady-state TypedArray allocations', () => {
  assert.equal(typeof PcmScratchPool, 'function');
  assert.equal(typeof deinterleavePcm, 'function');
  assert.equal(typeof interleavePcm, 'function');

  // 1. Standalone deinterleavePcm / interleavePcm round-trip + NaN sanitization across 1ch, 2ch, 6ch.
  const stereoIn = new Float32Array([0.25, -0.25, 0.5, Number.NaN, -0.75, Infinity, 1.0, -1.0]);
  const planar2 = deinterleavePcm(stereoIn, 2, 4);
  assert.equal(planar2.length, 2);
  assert.deepEqual(Array.from(planar2[0]), [0.25, 0.5, -0.75, 1.0]);
  assert.deepEqual(Array.from(planar2[1]), [-0.25, 0.0, 0.0, -1.0]);

  const reinterleaved = interleavePcm(planar2, 4);
  assert.deepEqual(Array.from(reinterleaved), [0.25, -0.25, 0.5, 0.0, -0.75, 0.0, 1.0, -1.0]);

  // 6-channel (5.1) deinterleave/interleave exactness.
  const pcm6 = new Float32Array(3 * 6);
  for (let i = 0; i < pcm6.length; i++) pcm6[i] = (i + 1) * 0.05;
  const planar6 = deinterleavePcm(pcm6, 6, 3);
  assert.equal(planar6.length, 6);
  for (let f = 0; f < 3; f++) {
    for (let c = 0; c < 6; c++) {
      assert.equal(planar6[c][f], pcm6[f * 6 + c]);
    }
  }
  assert.deepEqual(Array.from(interleavePcm(planar6, 3)), Array.from(pcm6));

  // 2. PcmScratchPool steady-state reuse: exact same Float32Array & Array instances, zero new allocations.
  const pool = new PcmScratchPool();
  const blockA = makeStereoFrames(0, 128);
  const blockB = makeStereoFrames(128, 128);

  const firstPlanar = pool.deinterleave(blockA, 2, 128);
  const allocsAfterFirst = pool.allocations;
  assert.equal(allocsAfterFirst, 2);
  assert.equal(pool.reuses, 0);

  for (let i = 0; i < 40; i++) {
    const nextPlanar = pool.deinterleave(i % 2 === 0 ? blockB : blockA, 2, 128);
    assert.equal(nextPlanar, firstPlanar);
    assert.equal(nextPlanar[0], firstPlanar[0]);
    assert.equal(nextPlanar[1], firstPlanar[1]);
  }
  assert.equal(pool.allocations, allocsAfterFirst, 'Steady-state deinterleave must perform zero allocations');
  assert.equal(pool.reuses, 40);

  // Smaller frame count reuses the exact same underlying ArrayBuffer with zero new allocations.
  const smallerPlanar = pool.deinterleave(makeStereoFrames(0, 64), 2, 64);
  assert.equal(smallerPlanar[0].length, 64);
  assert.equal(smallerPlanar[0].buffer, firstPlanar[0].buffer);
  assert.equal(smallerPlanar[1].buffer, firstPlanar[1].buffer);
  assert.equal(pool.allocations, allocsAfterFirst);

  // 3. AudioMixerWorker end-to-end steady-state SAB pushPcm + downmix + resample + pullPlanar zero-alloc check.
  const worker = new AudioMixerWorker(null, {
    outputState: 'active',
    autoDrain: false,
    declickFrames: 0,
  });
  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 1024,
    outputState: 'active',
  });

  const sab = new SharedArrayBuffer(128 * 2 * Float32Array.BYTES_PER_ELEMENT);
  const sabView = new Float32Array(sab);
  sabView.set(makeStereoFrames(0, 128));

  // Warmup packet #1 (allocates 'extract', 'pull', and 2 planar buffers).
  worker.handleMessage({ pushPcm: true, samples: sabView });
  const pull1 = worker.pullPlanar(128);
  assert.equal(pull1.framesRead, 128);
  const firstL = pull1.channels[0];
  const firstR = pull1.channels[1];
  assert.equal(firstL[0], sabView[0]);
  assert.equal(firstR[0], sabView[1]);
  const allocsAfterWarmup = worker.getScratchStats().allocations;
  assert.ok(allocsAfterWarmup > 0);

  // Steady-state packets #2..#31: ZERO additional scratch allocations!
  for (let round = 1; round <= 30; round++) {
    sabView.set(makeStereoFrames(round * 128, 128));
    worker.handleMessage({ pushPcm: true, samples: sabView });
    const pulled = worker.pullPlanar(128);
    assert.equal(pulled.framesRead, 128);
    assert.equal(pulled.channels[0], firstL);
    assert.equal(pulled.channels[1], firstR);
    assert.equal(pulled.channels[0][0], sabView[0]);
    assert.equal(pulled.channels[1][0], sabView[1]);
  }

  const steadyStats = worker.getStats();
  assert.equal(steadyStats.scratchAllocations, allocsAfterWarmup);
  assert.ok(steadyStats.scratchReuses >= 30 * 3, `Expected >= 90 scratch reuses, got ${steadyStats.scratchReuses}`);
  assert.equal(steadyStats.packetsReceived, 31);
  assert.equal(steadyStats.framesDecoded, 31 * 128);
  assert.equal(steadyStats.droppedPackets, 0);
  assert.equal(steadyStats.bufferFillPct, 0);

  // Planar pushPcm input + steady-state reuse.
  const chL = new Float32Array(64);
  const chR = new Float32Array(64);
  for (let f = 0; f < 64; f++) {
    chL[f] = (f + 1) * 0.01;
    chR[f] = -(f + 1) * 0.01;
  }
  worker.handleMessage({ pushPcm: true, planar: true, samples: [chL, chR] });
  const allocsAfterPlanarWarmup = worker.getScratchStats().allocations;
  for (let round = 0; round < 10; round++) {
    worker.handleMessage({ pushPcm: true, planar: true, channelData: [chL, chR] });
    const p = worker.pullPlanar(64);
    assert.equal(p.framesRead, 64);
    assert.ok(Math.abs(p.channels[0][0] - chL[0]) < 1e-6);
    assert.ok(Math.abs(p.channels[1][63] - chR[63]) < 1e-6);
  }
  assert.equal(worker.getScratchStats().allocations, allocsAfterPlanarWarmup);
  worker.destroy();
});

test('W222 bounded ring-overflow drop guards (drop-tail, drop-newest, drop-oldest, maxFillRatio) protect unread audio and track droppedPackets', () => {
  assert.deepEqual(OVERFLOW_POLICIES, {
    DROP_TAIL: 'drop-tail',
    DROP_NEWEST: 'drop-newest',
    DROP_OLDEST: 'drop-oldest',
  });

  // 1. drop-newest: atomic packet drop preserves all existing buffered frames intact.
  const repliesNewest = [];
  const wNewest = new AudioMixerWorker((m) => repliesNewest.push(m), {
    outputState: 'active',
    autoDrain: false,
    declickFrames: 0,
    overflowPolicy: OVERFLOW_POLICIES.DROP_NEWEST,
  });
  wNewest.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 16,
    outputState: 'active',
  });

  const p1 = makeStereoFrames(0, 12);
  wNewest.handleMessage({ pushPcm: true, id: 1, samples: p1 });
  assert.equal(repliesNewest[repliesNewest.length - 1].pushedFrames, 12);
  assert.equal(repliesNewest[repliesNewest.length - 1].droppedFrames, 0);
  assert.equal(wNewest.getStats().bufferFillPct, 75);

  // Only 4 frames free; pushing 8 under 'drop-newest' atomically drops all 8 frames.
  const p2 = makeStereoFrames(100, 8);
  wNewest.handleMessage({ pushPcm: true, id: 2, samples: p2 });
  assert.equal(repliesNewest[repliesNewest.length - 1].pushedFrames, 0);
  assert.equal(repliesNewest[repliesNewest.length - 1].droppedFrames, 8);
  assert.equal(repliesNewest[repliesNewest.length - 1].availableFrames, 12);

  let sNewest = wNewest.getStats();
  assert.equal(sNewest.packetsReceived, 2);
  assert.equal(sNewest.framesDecoded, 20);
  assert.equal(sNewest.droppedPackets, 1);
  assert.equal(sNewest.droppedFrames, 8);
  assert.equal(sNewest.overflows, 1);
  assert.equal(sNewest.bufferFillPct, 75);

  // Buffered 12 frames remain strictly p1.
  const outNewest = new Float32Array(12 * 2);
  assert.equal(wNewest.pullSamples(outNewest), 12);
  assert.deepEqual(Array.from(outNewest), Array.from(p1));
  wNewest.destroy();

  // 2. drop-oldest: evicts oldest unread frames to keep freshest low-latency audio bounded.
  const repliesOldest = [];
  const wOldest = new AudioMixerWorker((m) => repliesOldest.push(m), {
    outputState: 'active',
    autoDrain: false,
    declickFrames: 0,
    overflowPolicy: OVERFLOW_POLICIES.DROP_OLDEST,
  });
  wOldest.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 10,
    outputState: 'active',
  });

  const o1 = makeStereoFrames(0, 8); // frames 0..7
  const o2 = makeStereoFrames(8, 6); // frames 8..13 -> evicts oldest 4 (0..3), keeps 4..13
  wOldest.handleMessage({ pushPcm: true, id: 1, samples: o1 });
  wOldest.handleMessage({ pushPcm: true, id: 2, samples: o2 });
  assert.equal(repliesOldest[repliesOldest.length - 1].pushedFrames, 6);
  assert.equal(repliesOldest[repliesOldest.length - 1].droppedFrames, 4);
  assert.equal(repliesOldest[repliesOldest.length - 1].availableFrames, 10);

  const outOldest = new Float32Array(10 * 2);
  assert.equal(wOldest.pullSamples(outOldest), 10);
  assert.deepEqual(Array.from(outOldest), Array.from(makeStereoFrames(4, 10)));

  // Single oversized packet > capacity (14 frames into 10-frame ring) trims to newest 10 frames.
  const huge = makeStereoFrames(50, 14);
  wOldest.handleMessage({ pushPcm: true, id: 3, samples: huge });
  assert.equal(repliesOldest[repliesOldest.length - 1].pushedFrames, 10);
  assert.equal(repliesOldest[repliesOldest.length - 1].droppedFrames, 4);
  const outHuge = new Float32Array(10 * 2);
  assert.equal(wOldest.pullSamples(outHuge), 10);
  assert.deepEqual(Array.from(outHuge), Array.from(makeStereoFrames(54, 10)));

  const sOldest = wOldest.getStats();
  assert.equal(sOldest.packetsReceived, 3);
  assert.equal(sOldest.framesDecoded, 28);
  assert.equal(sOldest.droppedPackets, 2);
  assert.equal(sOldest.droppedFrames, 8);
  assert.equal(sOldest.overflows, 2);
  wOldest.destroy();

  // 3. maxFillRatio watermark caps latency at 50% of ring capacity.
  const repliesCap = [];
  const wCap = new AudioMixerWorker((m) => repliesCap.push(m), {
    outputState: 'active',
    autoDrain: false,
    declickFrames: 0,
    maxFillRatio: 0.5,
  });
  wCap.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 64,
    outputState: 'active',
  });
  wCap.handleMessage({ pushPcm: true, id: 1, samples: makeStereoFrames(0, 48) });
  assert.equal(repliesCap[repliesCap.length - 1].pushedFrames, 32);
  assert.equal(repliesCap[repliesCap.length - 1].droppedFrames, 16);
  assert.equal(wCap.getStats().bufferFillPct, 50);
  assert.equal(wCap.getStats().droppedPackets, 1);
  wCap.destroy();
});

test('W222 underrun recovery watermark holds silence until re-primed and telemetry snapshot exposes all W222 fields cleanly', () => {
  const replies = [];
  const worker = new AudioMixerWorker((m) => replies.push(m), {
    outputState: 'active',
    autoDrain: false,
    declickFrames: 8,
    underrunRecoveryFrames: 32,
  });

  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 128,
    outputState: 'active',
  });

  // 1. Trigger initial underrun on empty ring.
  const out0 = new Float32Array(16 * 2);
  out0.fill(5.0);
  assert.equal(worker.pullSamples(out0), 0);
  assert.ok(out0.every((v) => v === 0));
  assert.equal(worker.getStats().underruns, 1);
  assert.equal(worker.getStats().underrunRecoveries, 0);

  // 2. Trickle 16 frames (< 32 recovery watermark): pull holds silence without consuming trickle.
  worker.handleMessage({ pushPcm: true, samples: makeConstantStereoFrames(16, 0.5) });
  assert.equal(worker.ring.availableFrames(), 16);
  assert.equal(worker.getStats().bufferFillPct, 12.5);

  const holdOut = new Float32Array(16 * 2);
  holdOut.fill(3.0);
  assert.equal(worker.pullSamples(holdOut), 0);
  assert.ok(holdOut.every((v) => v === 0));
  assert.equal(worker.ring.availableFrames(), 16, 'Recovery hold must preserve trickled frames');
  assert.equal(worker.getStats().underruns, 2);
  assert.equal(worker.getStats().underrunRecoveries, 0);

  // 3. Push 16 more frames (total 32 == watermark): pull exits recovery cleanly with fade-in!
  worker.handleMessage({ pushPcm: true, samples: makeConstantStereoFrames(16, 0.5) });
  assert.equal(worker.getStats().bufferFillPct, 25);

  const recPlanar = worker.pullPlanar(32);
  assert.equal(recPlanar.framesRead, 32);
  assert.equal(recPlanar.channels[0][0], 0); // smooth fade-in starts at 0
  assert.equal(recPlanar.channels[0][8], 0.5); // full level after 8-frame declick ramp
  assert.equal(recPlanar.channels[1][31], 0.5);

  // 4. Request stats via message protocol and verify all W222 lightweight telemetry fields.
  worker.handleMessage({ stats: true, id: 'w222-stats' });
  const snap = replies[replies.length - 1];
  assert.equal(snap.type, 'audio-stats');
  assert.equal(snap.id, 'w222-stats');
  assert.equal(snap.stats.packetsReceived, 2);
  assert.equal(snap.stats.framesDecoded, 32);
  assert.equal(snap.stats.droppedPackets, 0);
  assert.equal(snap.stats.underruns, 2);
  assert.equal(snap.stats.underrunRecoveries, 1);
  assert.equal(snap.stats.bufferFillPct, 0);
  assert.ok(snap.stats.scratchAllocations >= 1);

  // 5. pullPlanar message protocol works cleanly.
  worker.handleMessage({ pushPcm: true, samples: makeStereoFrames(0, 64) });
  worker.handleMessage({ pullPlanar: true, frameCount: 32, id: 'w222-pull' });
  const pullReply = replies[replies.length - 1];
  assert.equal(pullReply.id, 'w222-pull');
  assert.equal(pullReply.framesRead, 32);
  assert.equal(pullReply.availableFrames, 32);
  assert.equal(pullReply.bufferFillPct, 25);

  // 6. Corrupt out-of-range SAB cursors are normalized defensively without crashing.
  const corruptRing = AudioRingBuffer.create(16, 2);
  const ctrl = new Int32Array(corruptRing.sab, 0, 8);
  Atomics.store(ctrl, 0, 999); // invalid write cursor >> 2 * capacity
  Atomics.store(ctrl, 1, -17); // invalid negative read cursor
  assert.doesNotThrow(() => {
    assert.ok(corruptRing.availableFrames() >= 0 && corruptRing.availableFrames() <= 16);
    corruptRing.pushSamples(makeStereoFrames(0, 4));
    corruptRing.pullSamples(new Float32Array(4 * 2));
  });

  // 7. Dirty telemetry sanitization clamps bufferFillPct and sanitizes W222 counters.
  const dirty = buildAudioTelemetrySnapshot(
    {
      packetsReceived: -5,
      framesDecoded: Infinity,
      droppedPackets: Number.NaN,
      underruns: 4,
      underrunRecoveries: 2.9,
      bufferFillPct: 999,
      overflowPolicy: 'evil-policy',
      maxFillRatio: -2,
    },
    { timestamp: 4321 }
  );
  assert.equal(dirty.stats.packetsReceived, 0);
  assert.equal(dirty.stats.framesDecoded, 0);
  assert.equal(dirty.stats.droppedPackets, 0);
  assert.equal(dirty.stats.underruns, 4);
  assert.equal(dirty.stats.underrunRecoveries, 2);
  assert.equal(dirty.stats.bufferFillPct, 100);
  assert.equal(dirty.stats.overflowPolicy, 'drop-tail');
  assert.equal(dirty.stats.maxFillRatio, 1.0);

  // 8. reset() clears all W222 counters cleanly.
  worker.reset();
  const afterReset = worker.getStats();
  assert.equal(afterReset.packetsReceived, 0);
  assert.equal(afterReset.framesDecoded, 0);
  assert.equal(afterReset.droppedPackets, 0);
  assert.equal(afterReset.underruns, 0);
  assert.equal(afterReset.underrunRecoveries, 0);
  assert.equal(afterReset.bufferFillPct, 0);
  worker.destroy();
});

// ---------------------------------------------------------------------------
// W235: R40 Host Stereo PCM Ring Reader & Pump (Playgta6AudioRing / #1027)
// ---------------------------------------------------------------------------

/**
 * Allocates a mock WASM linear-memory SharedArrayBuffer matching
 * `RingStorage` / `Playgta6AudioRing` in `runtime/os/audio_downmix.h`:
 *   [base .. base + capacitySamples*4)  float32 interleaved stereo samples
 *   [writeOffset .. writeOffset + 4)    uint32 producer write_index
 *   [readOffset  .. readOffset  + 4)    uint32 host read_index
 */
function createMockGuestPcmRing({
  capacitySamples = 64,
  baseOffset = 64,
  sampleRate = 48000,
  channels = 2,
} = {}) {
  const dataBytes = capacitySamples * Float32Array.BYTES_PER_ELEMENT;
  const writeIndexByteOffset = baseOffset + dataBytes;
  const readIndexByteOffset = writeIndexByteOffset + 4;
  const totalBytes = readIndexByteOffset + 64;
  const sab = new SharedArrayBuffer(totalBytes);
  const data = new Float32Array(sab, baseOffset, capacitySamples);
  const writeCtrl = new Uint32Array(sab, writeIndexByteOffset, 1);
  const readCtrl = new Uint32Array(sab, readIndexByteOffset, 1);
  const mask = capacitySamples - 1;

  function producerPush(samples) {
    const w = Atomics.load(writeCtrl, 0) >>> 0;
    const r = Atomics.load(readCtrl, 0) >>> 0;
    const freeSpace = capacitySamples - ((w - r) >>> 0);
    const n = Math.min(samples.length, freeSpace);
    for (let i = 0; i < n; i++) {
      data[(w + i) & mask] = samples[i];
    }
    Atomics.store(writeCtrl, 0, (w + n) >>> 0);
    return n;
  }

  function producerForcePush(samples) {
    const w = Atomics.load(writeCtrl, 0) >>> 0;
    for (let i = 0; i < samples.length; i++) {
      data[(w + i) & mask] = samples[i];
    }
    Atomics.store(writeCtrl, 0, (w + samples.length) >>> 0);
  }

  return {
    sab,
    data,
    writeCtrl,
    readCtrl,
    descriptor: {
      sab,
      base: baseOffset,
      capacity: capacitySamples,
      channels,
      sample_rate: sampleRate,
      read_index_ptr: readIndexByteOffset,
      write_index_ptr: writeIndexByteOffset,
    },
    producerPush,
    producerForcePush,
  };
}

test('W235 attachGuestPcmRing / drainGuestPcmRing / detachGuestPcmRing align with Playgta6AudioRing and pump exact stereo frames', () => {
  assert.equal(typeof GuestPcmRingReader, 'function');
  assert.equal(GUEST_PCM_RING_DEFAULT_SAMPLE_RATE, 48000);
  assert.equal(GUEST_PCM_RING_CHANNELS, 2);
  assert.equal(GUEST_PCM_RING_DEFAULT_CAPACITY_SAMPLES, 1 << 17);
  assert.equal(GUEST_PCM_RING_DEFAULT_QUANTUM_FRAMES, 480);

  const replies = [];
  const worker = new AudioMixerWorker((m) => replies.push(m), {
    outputState: 'active',
    autoDrain: false,
    declickFrames: 0,
  });

  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 256,
    outputState: 'active',
  });

  assert.equal(worker.getStats().guestRingAttached, false);

  // Use BigInt offsets (wasm64 uintptr_t style) via worker message contract.
  const mock = createMockGuestPcmRing({ capacitySamples: 64, baseOffset: 128 });
  worker.handleMessage({
    attachGuestPcmRing: true,
    id: 'attach-1',
    sab: mock.sab,
    base: BigInt(mock.descriptor.base),
    capacity: mock.descriptor.capacity,
    channels: 2,
    sample_rate: 48000,
    read_index_ptr: BigInt(mock.descriptor.read_index_ptr),
    write_index_ptr: BigInt(mock.descriptor.write_index_ptr),
    quantumFrames: 16,
  });

  const attachReply = replies[replies.length - 1];
  assert.equal(attachReply.id, 'attach-1');
  assert.equal(attachReply.attached, true);
  assert.equal(attachReply.guestRingAttached, true);
  assert.equal(attachReply.capacitySamples, 64);
  assert.equal(attachReply.capacityFrames, 32);
  assert.equal(attachReply.quantumFrames, 16);
  assert.equal(worker.getStats().guestRingAttached, true);

  // Push 16 stereo frames with amplitude inside [-0.95, 0.95] (bit-exact passthrough).
  const input16 = new Float32Array(16 * 2);
  for (let f = 0; f < 16; f++) {
    input16[f * 2] = Math.fround((f + 1) * 0.05);
    input16[f * 2 + 1] = Math.fround(-(f + 1) * 0.04);
  }
  assert.equal(mock.producerPush(input16), 32);

  // Drain via worker message protocol -> pumps into worker.ring.
  worker.handleMessage({ drainGuestPcmRing: true, id: 'drain-1', maxFrames: 16 });
  const drainReply = replies[replies.length - 1];
  assert.equal(drainReply.id, 'drain-1');
  assert.equal(drainReply.framesDrained, 16);
  assert.equal(drainReply.pushedFrames, 16);
  assert.equal(drainReply.droppedFrames, 0);
  assert.equal(drainReply.underrun, false);
  assert.equal(drainReply.overrun, false);
  assert.equal(drainReply.guestFramesDrained, 16);
  assert.equal(Atomics.load(mock.readCtrl, 0), 32);

  const pulled = new Float32Array(16 * 2);
  assert.equal(worker.pullSamples(pulled), 16);
  assert.deepEqual(Array.from(pulled), Array.from(input16));

  // Direct output buffer drain mode also works bit-exact.
  assert.equal(mock.producerPush(input16), 32);
  const directOut = new Float32Array(16 * 2);
  const directRes = worker.drainGuestPcmRing(directOut);
  assert.equal(directRes.framesDrained, 16);
  assert.equal(directRes.underrun, false);
  assert.deepEqual(Array.from(directOut), Array.from(input16));
  assert.equal(Atomics.load(mock.readCtrl, 0), 64);

  // Detach via message protocol (and verify idempotent second detach).
  worker.handleMessage({ detachGuestPcmRing: true, id: 'detach-1' });
  const detachReply = replies[replies.length - 1];
  assert.equal(detachReply.id, 'detach-1');
  assert.equal(detachReply.detached, true);
  assert.equal(detachReply.wasAttached, true);
  assert.equal(detachReply.guestRingAttached, false);
  assert.equal(worker.getStats().guestRingAttached, false);

  const detach2 = worker.detachGuestPcmRing();
  assert.equal(detach2.wasAttached, false);
  assert.equal(detach2.guestRingAttached, false);

  // Draining while detached is a safe no-op and zero-fills direct output.
  const detachedOut = new Float32Array(8 * 2);
  detachedOut.fill(5.0);
  const detachedRes = worker.drainGuestPcmRing(detachedOut);
  assert.equal(detachedRes.framesDrained, 0);
  assert.ok(detachedOut.every((v) => v === 0));

  worker.destroy();
});

test('W235 guest PCM ring wrap-around safety across ring boundary and uint32 2^32 cursor overflow with zero per-quantum heap allocation', () => {
  const worker = new AudioMixerWorker(null, {
    outputState: 'active',
    autoDrain: false,
    declickFrames: 0,
  });
  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 512,
    outputState: 'active',
  });

  // Small 16-sample (8-frame) ring to force heavy wrap-around.
  const mock = createMockGuestPcmRing({ capacitySamples: 16, baseOffset: 32 });
  worker.attachGuestPcmRing({
    ...mock.descriptor,
    quantumFrames: 6,
  });

  // 1. Cross the 2^32 unsigned 32-bit cursor boundary explicitly!
  // Start both cursors 4 samples (2 frames) before 2^32.
  const startCursor = 0xfffffffc; // 4294967292
  Atomics.store(mock.readCtrl, 0, startCursor);
  Atomics.store(mock.writeCtrl, 0, startCursor);

  const wrapIn = new Float32Array(6 * 2);
  for (let f = 0; f < 6; f++) {
    wrapIn[f * 2] = Math.fround((f + 1) * 0.1);
    wrapIn[f * 2 + 1] = Math.fround(-(f + 1) * 0.1);
  }
  assert.equal(mock.producerPush(wrapIn), 12);
  assert.equal(Atomics.load(mock.writeCtrl, 0), 8); // (0xFFFFFFFC + 12) >>> 0 === 8

  const wrapOut = new Float32Array(6 * 2);
  const wrapRes = worker.drainGuestPcmRing(wrapOut);
  assert.equal(wrapRes.framesDrained, 6);
  assert.equal(wrapRes.underrun, false);
  assert.equal(wrapRes.overrun, false);
  assert.equal(Atomics.load(mock.readCtrl, 0), 8);
  assert.deepEqual(Array.from(wrapOut), Array.from(wrapIn));

  // 2. Zero per-quantum heap allocation check across 50 wrapped pump + direct quanta.
  const allocsAfterAttach = worker.getScratchStats().allocations;
  const reusesBefore = worker.getScratchStats().reuses;
  const pullBuf = new Float32Array(6 * 2);

  for (let round = 0; round < 50; round++) {
    const chunk = new Float32Array(6 * 2);
    for (let f = 0; f < 6; f++) {
      const s = ((round * 6 + f) % 19) * 0.04;
      chunk[f * 2] = Math.fround(s);
      chunk[f * 2 + 1] = Math.fround(-s);
    }
    assert.equal(mock.producerPush(chunk), 12);

    const res = worker.drainGuestPcmRing(6);
    assert.equal(res.framesDrained, 6);
    assert.equal(res.pushedFrames, 6);
    assert.equal(res.underrun, false);
    assert.equal(res.overrun, false);

    assert.equal(worker.pullSamples(pullBuf), 6);
    assert.deepEqual(Array.from(pullBuf), Array.from(chunk));
  }

  // Steady-state pump quanta performed ZERO new scratch buffer allocations!
  assert.equal(
    worker.getScratchStats().allocations,
    allocsAfterAttach,
    'Steady-state drainGuestPcmRing must perform zero per-quantum scratch allocations'
  );
  assert.ok(worker.getScratchStats().reuses >= reusesBefore + 50);

  // Direct output mode also performs ZERO scratch allocations across 20 quanta.
  for (let round = 0; round < 20; round++) {
    assert.equal(mock.producerPush(wrapIn), 12);
    const res = worker.drainGuestPcmRing(wrapOut);
    assert.equal(res.framesDrained, 6);
    assert.deepEqual(Array.from(wrapOut), Array.from(wrapIn));
  }
  assert.equal(worker.getScratchStats().allocations, allocsAfterAttach);
  assert.equal(worker.getStats().guestFramesDrained, 6 + 50 * 6 + 20 * 6);
  worker.destroy();
});

test('W235 graceful underrun silence fill, soft-clipping [-1, 1], NaN sanitization, and overrun drop accounting', () => {
  const worker = new AudioMixerWorker(null, {
    outputState: 'active',
    autoDrain: false,
    declickFrames: 0,
  });
  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 8,
    outputState: 'active',
  });

  // 16-sample (8-frame) guest ring.
  const mock = createMockGuestPcmRing({ capacitySamples: 16, baseOffset: 64 });
  worker.attachGuestPcmRing({
    ...mock.descriptor,
    quantumFrames: 8,
    softClip: true,
    limiterKnee: 0.95,
  });

  // 1. Soft-clipping [-1, 1] + NaN/Infinity sanitization + partial underrun silence fill.
  const hotAndBad = new Float32Array([
    0.5, -0.5,                       // frame 0: inside knee -> bit-exact
    2.5, -3.0,                       // frame 1: hot -> soft-clipped in (0.95, 1.0] / [-1.0, -0.95)
    Number.NaN, Number.POSITIVE_INFINITY, // frame 2: non-finite -> 0.0, 0.0
    Number.NEGATIVE_INFINITY, 0.99,  // frame 3: -Inf -> 0.0, 0.99 -> soft-clipped
  ]);
  assert.equal(mock.producerPush(hotAndBad), 8);

  const out8 = new Float32Array(8 * 2);
  out8.fill(77.0); // sentinel to verify silence fill overwrites underrun tail
  const res1 = worker.drainGuestPcmRing(out8);
  assert.equal(res1.framesDrained, 4);
  assert.equal(res1.underrun, true);
  assert.equal(res1.underrunFrames, 4);
  assert.equal(out8[0], 0.5);
  assert.equal(out8[1], -0.5);
  assert.ok(out8[2] > 0.95 && out8[2] <= 1.0);
  assert.ok(out8[3] < -0.95 && out8[3] >= -1.0);
  assert.equal(out8[4], 0.0);
  assert.equal(out8[5], 0.0);
  assert.equal(out8[6], 0.0);
  assert.ok(out8[7] > 0.95 && out8[7] <= 1.0);
  // Underrun tail frames 4..7 must be strictly 0.0.
  for (let i = 8; i < 16; i++) {
    assert.equal(out8[i], 0.0);
  }
  assert.ok(res1.peakLevel > 0.95 && res1.peakLevel <= 1.0);
  assert.equal(worker.getStats().guestUnderruns, 1);
  assert.equal(worker.getStats().sanitizedSamples, 3);
  assert.equal(worker.getStats().clippedSamples, 3);

  // 2. Pump mode underrun with explicit fillSilence: true pushes silence frames cleanly.
  const emptyPump = worker.drainGuestPcmRing({ maxFrames: 4, fillSilence: true });
  assert.equal(emptyPump.framesDrained, 0);
  assert.equal(emptyPump.underrun, true);
  assert.equal(emptyPump.underrunFrames, 4);
  assert.equal(emptyPump.pushedFrames, 4);
  assert.equal(worker.getStats().guestUnderruns, 2);

  const silencePulled = new Float32Array(4 * 2);
  silencePulled.fill(1.0);
  assert.equal(worker.pullSamples(silencePulled), 4);
  assert.ok(silencePulled.every((v) => v === 0));

  // 3. Guest-ring producer lap overrun (producer advances write past read + capacity).
  // Push 12 frames (24 samples) into 8-frame (16-sample) ring without checking free space:
  // oldest 4 frames are overwritten, freshest 8 frames (frames 4..11) survive.
  const lapFrames = new Float32Array(12 * 2);
  for (let f = 0; f < 12; f++) {
    lapFrames[f * 2] = Math.fround((f + 1) * 0.05);
    lapFrames[f * 2 + 1] = Math.fround(-(f + 1) * 0.05);
  }
  mock.producerForcePush(lapFrames);

  const afterLap = new Float32Array(8 * 2);
  const lapRes = worker.drainGuestPcmRing(afterLap);
  assert.equal(lapRes.overrun, true);
  assert.equal(lapRes.overrunDroppedFrames, 4);
  assert.equal(lapRes.framesDrained, 8);
  assert.deepEqual(Array.from(afterLap), Array.from(lapFrames.subarray(4 * 2)));
  assert.equal(worker.getStats().guestOverruns, 1);

  // 4. Worker output ring backpressure overrun when pumping into a full active worker ring.
  // Fill worker.ring (capacity 8) with 6 frames first.
  worker.handleMessage({ pushPcm: true, samples: makeConstantStereoFrames(6, 0.25) });
  assert.equal(worker.ring.availableFrames(), 6);

  // Push 6 frames into guest ring and pump into worker.ring (only 2 frames free -> 4 dropped).
  assert.equal(mock.producerPush(makeConstantStereoFrames(6, 0.5)), 12);
  const backpressureRes = worker.drainGuestPcmRing(6);
  assert.equal(backpressureRes.framesDrained, 6);
  assert.equal(backpressureRes.pushedFrames, 2);
  assert.equal(backpressureRes.droppedFrames, 4);
  assert.equal(backpressureRes.overrun, true);
  assert.equal(worker.getStats().guestOverruns, 2);

  // 5. Backward/corrupt cursor anomaly (read > write by small delta -> unsigned diff > 2^31) resyncs safely.
  Atomics.store(mock.writeCtrl, 0, 10);
  Atomics.store(mock.readCtrl, 0, 20);
  const corruptOut = new Float32Array(4 * 2);
  corruptOut.fill(3.0);
  const corruptRes = worker.drainGuestPcmRing(corruptOut);
  assert.equal(corruptRes.overrun, true);
  assert.equal(corruptRes.framesDrained, 0);
  assert.equal(Atomics.load(mock.readCtrl, 0), 10); // resynced to write
  assert.ok(corruptOut.every((v) => v === 0));
  assert.equal(worker.getStats().guestOverruns, 3);

  worker.destroy();
});

test('W235 guest PCM ring resampling (44.1 kHz -> 48 kHz), validation, telemetry sanitization, headless autoPump, and reset', () => {
  const replies = [];
  let fakeNow = 2000;
  const worker = new AudioMixerWorker((m) => replies.push(m), {
    now: () => fakeNow,
    outputState: 'active',
    autoDrain: false,
    declickFrames: 0,
  });

  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 2048,
    outputState: 'active',
  });

  // 1. Invalid descriptors rejected cleanly via handleMessage (never throw) and throw via direct API.
  const goodSab = new SharedArrayBuffer(512);
  const invalidDescriptors = [
    { sab: new ArrayBuffer(512), base: 0, capacity: 16, read_index_ptr: 64, write_index_ptr: 68 }, // non-SAB
    { sab: goodSab, base: 0, capacity: 12, read_index_ptr: 64, write_index_ptr: 68 }, // non-power-of-two
    { sab: goodSab, base: 2, capacity: 16, read_index_ptr: 64, write_index_ptr: 68 }, // unaligned base
    { sab: goodSab, base: 0, capacity: 16, read_index_ptr: 64, write_index_ptr: 64 }, // aliased cursors
    { sab: goodSab, base: 0, capacity: 16, read_index_ptr: 8, write_index_ptr: 68 },  // cursor overlaps data
    { sab: goodSab, base: 0, capacity: 16, channels: 6, read_index_ptr: 64, write_index_ptr: 68 }, // non-stereo
    { sab: goodSab, base: 0, capacity: 16, sample_rate: 100, read_index_ptr: 64, write_index_ptr: 68 }, // bad rate
    { sab: goodSab, base: 480, capacity: 16, read_index_ptr: 0, write_index_ptr: 4 }, // data out of bounds
  ];

  for (let i = 0; i < invalidDescriptors.length; i++) {
    assert.throws(() => worker.attachGuestPcmRing(invalidDescriptors[i]));
    assert.doesNotThrow(() => {
      worker.handleMessage({
        type: 'attach-guest-pcm-ring',
        id: `bad-${i}`,
        descriptor: invalidDescriptors[i],
      });
    });
    const errReply = replies[replies.length - 1];
    assert.equal(errReply.id, `bad-${i}`);
    assert.equal(errReply.attached, false);
    assert.equal(typeof errReply.error, 'string');
  }
  assert.equal(worker.getStats().guestRingAttached, false);
  assert.equal(worker.getStats().rejectedPackets, invalidDescriptors.length);

  // 2. 44.1 kHz guest ring -> 48 kHz worker pump resampling.
  const mock441 = createMockGuestPcmRing({
    capacitySamples: 2048,
    baseOffset: 64,
    sampleRate: 44100,
  });
  worker.attachGuestPcmRing({ ...mock441.descriptor, quantumFrames: 441 });

  const tone441 = new Float32Array(441 * 2);
  for (let f = 0; f < 441; f++) {
    const v = Math.sin((2 * Math.PI * 1000 * f) / 44100) * 0.5;
    tone441[f * 2] = v;
    tone441[f * 2 + 1] = -v;
  }
  assert.equal(mock441.producerPush(tone441), 441 * 2);

  const rsDrain = worker.drainGuestPcmRing(441);
  assert.equal(rsDrain.framesDrained, 441);
  assert.equal(rsDrain.pushedFrames, 480);

  const pulled48 = new Float32Array(480 * 2);
  assert.equal(worker.pullSamples(pulled48), 480);
  let sumSq = 0;
  for (let f = 0; f < 478; f++) {
    const exp = Math.sin((2 * Math.PI * 1000 * f) / 48000) * 0.5;
    const dL = pulled48[f * 2] - exp;
    const dR = pulled48[f * 2 + 1] - -exp;
    sumSq += dL * dL + dR * dR;
  }
  assert.ok(Math.sqrt(sumSq / (478 * 2)) < 0.005);

  // 3. Sanitized audio-stats telemetry surfaces all 5 W235 counters and sanitizes dirty inputs.
  worker.handleMessage({ stats: true, id: 'w235-stats' });
  const statsMsg = replies[replies.length - 1];
  assert.equal(statsMsg.type, 'audio-stats');
  assert.equal(statsMsg.stats.guestRingAttached, true);
  assert.equal(statsMsg.stats.guestFramesDrained, 441);
  assert.equal(statsMsg.stats.guestUnderruns, 0);
  assert.equal(statsMsg.stats.guestOverruns, 0);
  assert.ok(statsMsg.stats.guestPeakLevel > 0.45 && statsMsg.stats.guestPeakLevel <= 0.55);

  const dirty = buildAudioTelemetrySnapshot(
    {
      guestRingAttached: 1,
      guestFramesDrained: -99,
      guestUnderruns: Number.NaN,
      guestOverruns: Infinity,
      guestPeakLevel: 42.5,
      secretPath: 'C:\\game\\audio.awc',
    },
    { timestamp: 5555 }
  );
  assert.equal(dirty.stats.guestRingAttached, true);
  assert.equal(dirty.stats.guestFramesDrained, 0);
  assert.equal(dirty.stats.guestUnderruns, 0);
  assert.equal(dirty.stats.guestOverruns, 0);
  assert.equal(dirty.stats.guestPeakLevel, 1.0);
  assert.equal(JSON.stringify(dirty).includes('audio.awc'), false);

  // 4. reset() zeroes guest counters while keeping the guest ring attached.
  worker.reset();
  const afterReset = worker.getStats();
  assert.equal(afterReset.guestRingAttached, true);
  assert.equal(afterReset.guestFramesDrained, 0);
  assert.equal(afterReset.guestUnderruns, 0);
  assert.equal(afterReset.guestOverruns, 0);
  assert.equal(afterReset.guestPeakLevel, 0);

  // 5. Headless drain mode with autoPump: true drains guest ring on drainTick().
  const mockHeadless = createMockGuestPcmRing({ capacitySamples: 1024, baseOffset: 64 });
  worker.setOutputState('headless-drain');
  worker.attachGuestPcmRing({
    ...mockHeadless.descriptor,
    quantumFrames: 240,
    autoPump: true,
  });
  assert.equal(mockHeadless.producerPush(makeConstantStereoFrames(240, 0.4)), 480);

  worker._lastDrainMs = 2000;
  fakeNow = 2005; // 5 ms @ 48 kHz = 240 frames
  const tickDrained = worker.drainTick(fakeNow);
  assert.equal(tickDrained, 240);
  assert.equal(worker.getStats().guestFramesDrained, 240);
  assert.equal(Atomics.load(mockHeadless.readCtrl, 0), 480);

  worker.destroy();
  assert.equal(worker.getStats().guestRingAttached, false);
});

// ---------------------------------------------------------------------------
// W299: W272 Fade Metrics, targetLatencyMs Jitter Control & Zero-Alloc #1028 Ring Drain
// ---------------------------------------------------------------------------

test('W299 targetLatencyMs clamping [10, 120] ms, control messages, drain cadence & watermark refill tuning', () => {
  assert.equal(DEFAULT_TARGET_LATENCY_MS, 30);
  assert.equal(MIN_TARGET_LATENCY_MS, 10);
  assert.equal(MAX_TARGET_LATENCY_MS, 120);
  assert.equal(clampTargetLatencyMs(undefined), 30);
  assert.equal(clampTargetLatencyMs(Number.NaN), 30);
  assert.equal(clampTargetLatencyMs(Number.POSITIVE_INFINITY), 30);
  assert.equal(clampTargetLatencyMs('60'), 30);
  assert.equal(clampTargetLatencyMs(2), 10);
  assert.equal(clampTargetLatencyMs(10), 10);
  assert.equal(clampTargetLatencyMs(45), 45);
  assert.equal(clampTargetLatencyMs(120), 120);
  assert.equal(clampTargetLatencyMs(500), 120);

  const replies = [];
  const scheduledTimers = [];
  let clearedCount = 0;
  const worker = new AudioMixerWorker((m) => replies.push(m), {
    outputState: 'headless-drain',
    autoDrain: true,
    setIntervalFn: (fn, ms) => {
      const handle = { fn, ms };
      scheduledTimers.push(handle);
      return handle;
    },
    clearIntervalFn: () => {
      clearedCount += 1;
    },
  });

  // Default targetLatencyMs is 30 ms.
  assert.equal(worker.targetLatencyMs, 30);
  assert.equal(worker.getStats().targetLatencyMs, 30);

  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 4096,
    outputState: 'headless-drain',
  });
  assert.equal(scheduledTimers.length, 1);

  // 1. `set-target-latency` message clamps low (< 10 ms -> 10 ms) & retunes timer cadence (5 ms).
  worker.handleMessage({ type: 'set-target-latency', targetLatencyMs: 3, id: 'lat-low' });
  const lowAck = replies[replies.length - 1];
  assert.equal(lowAck.id, 'lat-low');
  assert.equal(lowAck.type, 'target-latency');
  assert.equal(lowAck.targetLatencyMs, 10);
  assert.equal(lowAck.targetLatencyFrames, 480);
  assert.equal(lowAck.drainWatermarkFrames, 480);
  assert.equal(lowAck.drainIntervalMs, 5);
  assert.equal(worker.targetLatencyMs, 10);
  assert.equal(worker.drainIntervalMs, 5);
  assert.equal(clearedCount, 1);
  assert.equal(scheduledTimers[scheduledTimers.length - 1].ms, 5);

  // 2. W293 `setTargetLatencyMs` control message clamps high (> 120 ms -> 120 ms) & caps watermark to ring capacity.
  worker.handleMessage({
    type: 'setTargetLatencyMs',
    setTargetLatencyMs: true,
    targetLatencyMs: 250,
    id: 'lat-high',
  });
  const highAck = replies[replies.length - 1];
  assert.equal(highAck.id, 'lat-high');
  assert.equal(highAck.targetLatencyMs, 120);
  assert.equal(highAck.targetLatencyFrames, 5760);
  assert.equal(highAck.drainWatermarkFrames, 4096); // capped to ring capacityFrames
  assert.equal(highAck.drainIntervalMs, 60);

  // 3. Invalid/non-finite targetLatencyMs preserves current setting safely without throwing.
  worker.handleMessage({ type: 'set-target-latency', targetLatencyMs: Number.NaN, id: 'lat-nan' });
  assert.equal(replies[replies.length - 1].targetLatencyMs, 120);
  worker.handleMessage({ type: 'set-target-latency', targetLatencyMs: 'invalid', id: 'lat-str' });
  assert.equal(replies[replies.length - 1].targetLatencyMs, 120);

  // 4. Watermark-tuned refill cadence in active mode stops pumping once buffered >= targetLatencyFrames.
  worker.setOutputState('active');
  worker.handleMessage({ type: 'set-target-latency', targetLatencyMs: 20, id: 'lat-20' }); // 960 frames @ 48 kHz
  assert.equal(worker.targetLatencyFrames, 960);
  assert.equal(worker.drainWatermarkFrames, 960);

  const mock = createMockGuestPcmRing({ capacitySamples: 4096, baseOffset: 64 });
  worker.attachGuestPcmRing(mock.descriptor);
  assert.equal(mock.producerPush(makeConstantStereoFrames(1500, 0.25)), 3000);

  // First implicit drain fills exactly up to the 960-frame watermark (not all 1500 frames).
  const refill1 = worker.drainGuestPcmRing();
  assert.equal(refill1.framesDrained, 960);
  assert.equal(refill1.pushedFrames, 960);
  assert.equal(worker.ring.availableFrames(), 960);

  // Second implicit drain sees ring already at watermark -> pumps 0 frames (no over-buffering).
  const refill2 = worker.drainGuestPcmRing();
  assert.equal(refill2.framesDrained, 0);
  assert.equal(refill2.pushedFrames, 0);
  assert.equal(worker.ring.availableFrames(), 960);

  // Pull 320 frames -> deficit is 320 frames -> next implicit drain refills exactly 320 frames.
  assert.equal(worker.pullSamples(new Float32Array(320 * 2)), 320);
  const refill3 = worker.drainGuestPcmRing();
  assert.equal(refill3.framesDrained, 320);
  assert.equal(refill3.pushedFrames, 320);
  assert.equal(worker.ring.availableFrames(), 960);

  worker.destroy();
});

test('W299 zero-alloc reusable scratch drain across wrap-around stereo #1028 rings (Wasm64 & Wasm32) with variable sizes', () => {
  const worker = new AudioMixerWorker(null, {
    outputState: 'active',
    autoDrain: false,
    declickFrames: 0,
  });
  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 256,
    outputState: 'active',
  });

  // 32-sample (16-frame) stereo #1028 ring using Wasm64 BigInt byte offsets.
  const mock64 = createMockGuestPcmRing({ capacitySamples: 32, baseOffset: 64 });
  worker.attachGuestPcmRing({
    sab: mock64.sab,
    base: BigInt(mock64.descriptor.base),
    capacity: 32,
    channels: 2,
    sample_rate: 48000,
    read_index_ptr: BigInt(mock64.descriptor.read_index_ptr),
    write_index_ptr: BigInt(mock64.descriptor.write_index_ptr),
    quantumFrames: 16,
  });

  // Start near uint32 2^32 cursor wrap boundary to stress both slot wrap and uint32 wrap.
  Atomics.store(mock64.readCtrl, 0, 0xfffffff8);
  Atomics.store(mock64.writeCtrl, 0, 0xfffffff8);

  const allocsAfterAttach = worker.getScratchStats().allocations;
  const pullScratch = new Float32Array(16 * 2);

  // Spy on Float32Array.prototype.subarray to prove variable-size drains do not allocate even subarray views.
  const origSubarray = Float32Array.prototype.subarray;
  let subarrayCalls = 0;
  Float32Array.prototype.subarray = function (...args) {
    subarrayCalls += 1;
    return origSubarray.apply(this, args);
  };

  try {
    const variableSizes = [5, 11, 3, 13, 7, 9, 4, 12, 6, 10, 8, 15, 2, 14, 16, 1];
    let seq = 0;
    for (let round = 0; round < variableSizes.length; round++) {
      const frames = variableSizes[round];
      const input = new Float32Array(frames * 2);
      for (let f = 0; f < frames; f++) {
        const val = ((seq + f) % 17) * 0.04;
        input[f * 2] = Math.fround(val);
        input[f * 2 + 1] = Math.fround(-val);
      }
      seq += frames;
      assert.equal(mock64.producerPush(input), frames * 2);

      const drainRes = worker.drainGuestPcmRing(frames);
      assert.equal(drainRes.framesDrained, frames);
      assert.equal(drainRes.pushedFrames, frames);
      assert.equal(drainRes.zeroAlloc, true);

      const outView = origSubarray.call(pullScratch, 0, frames * 2);
      assert.equal(worker.pullSamples(outView), frames);
      for (let i = 0; i < frames * 2; i++) {
        assert.equal(outView[i], input[i]);
      }
    }
  } finally {
    Float32Array.prototype.subarray = origSubarray;
  }

  assert.equal(subarrayCalls, 0, 'Variable-size stereo #1028 ring drains must allocate zero Float32Array subarray views');
  assert.equal(worker.getScratchStats().allocations, allocsAfterAttach);
  assert.equal(worker.getStats().zeroAllocDrains, 16);

  // Verify Wasm32 numeric offsets attach & drain cleanly too.
  const mock32 = createMockGuestPcmRing({ capacitySamples: 16, baseOffset: 32 });
  worker.attachGuestPcmRing({
    ...mock32.descriptor,
    quantumFrames: 8,
  });
  const batch32 = makeConstantStereoFrames(6, 0.3);
  assert.equal(mock32.producerPush(batch32), 12);
  const res32 = worker.drainGuestPcmRing(6);
  assert.equal(res32.framesDrained, 6);
  assert.equal(res32.zeroAlloc, true);
  assert.equal(worker.getStats().zeroAllocDrains, 17);

  worker.destroy();
});

test('W299 NaN/Inf sanitization, soft-clipping [-1, 1], and W272 half-cosine underrun/recovery fade ramps', () => {
  // 1. Standalone W272 helpers: applyFadeRamp & sanitizeAndSoftLimitBuffer.
  assert.equal(typeof applyFadeRamp, 'function');
  assert.equal(typeof sanitizeAndSoftLimitBuffer, 'function');

  const rampBuf = makeConstantStereoFrames(16, 0.8);
  applyFadeRamp(rampBuf, 2, 16, 1.0, 0.0);
  assert.ok(Math.abs(rampBuf[0] - 0.8) < 1e-6);
  assert.equal(rampBuf[15 * 2], 0.0);
  assert.equal(rampBuf[15 * 2 + 1], 0.0);
  for (let f = 1; f < 16; f++) {
    assert.ok(rampBuf[f * 2] <= rampBuf[(f - 1) * 2] + 1e-6);
  }

  const sanitizeTarget = new Float32Array([0.5, -0.5, Number.NaN, Number.POSITIVE_INFINITY, 2.0, -3.5]);
  const reuseStats = { softClippedSamples: 0, nonFiniteSanitized: 0 };
  sanitizeAndSoftLimitBuffer(sanitizeTarget, sanitizeTarget.length, 0.95, reuseStats);
  assert.equal(sanitizeTarget[0], 0.5);
  assert.equal(sanitizeTarget[1], -0.5);
  assert.equal(sanitizeTarget[2], 0.0);
  assert.equal(sanitizeTarget[3], 0.0);
  assert.ok(sanitizeTarget[4] > 0.95 && sanitizeTarget[4] <= 1.0);
  assert.ok(sanitizeTarget[5] < -0.95 && sanitizeTarget[5] >= -1.0);
  assert.equal(reuseStats.nonFiniteSanitized, 2);
  assert.equal(reuseStats.softClippedSamples, 2);

  // 2. #1028 stereo ring drain with NaN/±Inf sanitization, soft-clipping, and half-cosine underrun/recovery fades.
  const worker = new AudioMixerWorker(null, {
    outputState: 'active',
    autoDrain: false,
    declickFrames: 16,
    fadeCurve: 'cosine',
  });
  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 256,
    outputState: 'active',
  });

  const mock = createMockGuestPcmRing({ capacitySamples: 128, baseOffset: 64 });
  worker.attachGuestPcmRing({
    ...mock.descriptor,
    quantumFrames: 32,
    softClip: true,
    limiterKnee: 0.95,
  });

  // Push 32 frames: first 2 frames carry NaN/±Inf and hot peaks, rest steady +0.8.
  const steadyAndDirty = makeConstantStereoFrames(32, 0.8);
  steadyAndDirty[0] = Number.NaN;
  steadyAndDirty[1] = Number.POSITIVE_INFINITY;
  steadyAndDirty[2] = Number.NEGATIVE_INFINITY;
  steadyAndDirty[3] = 2.5;
  steadyAndDirty[4] = -3.0;
  assert.equal(mock.producerPush(steadyAndDirty), 64);

  const out1 = new Float32Array(32 * 2);
  const d1 = worker.drainGuestPcmRing(out1);
  assert.equal(d1.framesDrained, 32);
  assert.equal(d1.underrun, false);
  assert.equal(d1.sanitizedNonFiniteSamples, 3);
  assert.equal(d1.softClippedSamples, 2);
  assert.equal(out1[0], 0.0);
  assert.equal(out1[1], 0.0);
  assert.equal(out1[2], 0.0);
  assert.ok(out1[3] > 0.95 && out1[3] <= 1.0);
  assert.ok(out1[4] < -0.95 && out1[4] >= -1.0);
  assert.equal(out1[31 * 2], Math.fround(0.8));

  // Trigger guest underrun (empty ring after active audio) -> smooth half-cosine fade-out from +0.8 to 0.0!
  const outUnderrun = new Float32Array(32 * 2);
  outUnderrun.fill(99.0);
  const d2 = worker.drainGuestPcmRing(outUnderrun);
  assert.equal(d2.framesDrained, 0);
  assert.equal(d2.underrun, true);
  assert.ok(outUnderrun[0] > 0.7 && outUnderrun[0] < 0.8, `Expected cosine fade-out start near 0.8, got ${outUnderrun[0]}`);
  assert.equal(outUnderrun[15 * 2], 0.0);
  assert.equal(outUnderrun[31 * 2], 0.0);
  for (let f = 1; f < 32; f++) {
    assert.ok(outUnderrun[f * 2] <= outUnderrun[(f - 1) * 2] + 1e-6);
  }

  // Consecutive empty drain stays silent and does NOT double-count underrunFadeCount.
  const outSilent = new Float32Array(32 * 2);
  outSilent.fill(5.0);
  worker.drainGuestPcmRing(outSilent);
  assert.ok(outSilent.every((v) => v === 0.0));
  assert.equal(worker.getStats().underrunFadeCount, 1);
  assert.equal(worker.getStats().recoveryFadeCount, 0);

  // Recovery drain with steady +0.8 -> smooth half-cosine fade-in from 0.0 to +0.8!
  assert.equal(mock.producerPush(makeConstantStereoFrames(32, 0.8)), 64);
  const outRecovery = new Float32Array(32 * 2);
  const d3 = worker.drainGuestPcmRing(outRecovery);
  assert.equal(d3.framesDrained, 32);
  assert.equal(d3.underrun, false);
  assert.equal(outRecovery[0], 0.0);
  assert.equal(outRecovery[16 * 2], Math.fround(0.8));
  assert.equal(outRecovery[31 * 2], Math.fround(0.8));
  for (let f = 1; f < 32; f++) {
    assert.ok(outRecovery[f * 2] >= outRecovery[(f - 1) * 2] - 1e-6);
  }

  // Verify AudioRingBuffer half-cosine fade curve & additive stats too.
  worker.handleMessage({ pushPcm: true, samples: makeConstantStereoFrames(16, 0.6) });
  const ringUnderrunOut = new Float32Array(32 * 2);
  assert.equal(worker.pullSamples(ringUnderrunOut), 16);
  assert.equal(ringUnderrunOut[31 * 2], 0.0);
  worker.handleMessage({ pushPcm: true, samples: makeConstantStereoFrames(32, 0.6) });
  const ringRecoveryOut = new Float32Array(32 * 2);
  assert.equal(worker.pullSamples(ringRecoveryOut), 32);
  assert.equal(ringRecoveryOut[0], 0.0);
  assert.equal(ringRecoveryOut[16 * 2], Math.fround(0.6));

  const stats = worker.getStats();
  assert.equal(stats.fadeCurve, 'cosine');
  assert.equal(stats.underrunFadeCount, 2); // 1 guest ring + 1 output ring
  assert.equal(stats.recoveryFadeCount, 2); // 1 guest ring + 1 output ring
  assert.equal(stats.sanitizedNonFiniteSamples, 3);
  assert.equal(stats.softClippedSamples, 2);
  assert.equal(stats.zeroAllocDrains, 4);

  worker.destroy();
});

test('W299 sanitized audio-stats telemetry surfaces all new fields, sanitizes dirty inputs, and resets cleanly', () => {
  const replies = [];
  const worker = new AudioMixerWorker((m) => replies.push(m), {
    outputState: 'active',
    autoDrain: false,
    declickFrames: 8,
    fadeCurve: 'cosine',
    targetLatencyMs: 42,
  });

  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 128,
    outputState: 'active',
  });

  const mock = createMockGuestPcmRing({ capacitySamples: 64, baseOffset: 64 });
  worker.attachGuestPcmRing({ ...mock.descriptor, quantumFrames: 16 });
  assert.equal(mock.producerPush(new Float32Array([Number.NaN, 2.2, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5])), 8);

  // Trigger partial underrun + fade-out + sanitization + soft-clip, then recovery + fade-in.
  const buf = new Float32Array(16 * 2);
  worker.drainGuestPcmRing(buf);
  assert.equal(mock.producerPush(makeConstantStereoFrames(16, 0.5)), 32);
  worker.drainGuestPcmRing(buf);

  worker.handleMessage({ type: 'request-audio-stats', id: 'w299-telemetry' });
  const reply = replies[replies.length - 1];
  assert.equal(reply.type, 'audio-stats');
  assert.equal(reply.id, 'w299-telemetry');
  assert.equal(reply.stats.targetLatencyMs, 42);
  assert.equal(reply.stats.zeroAllocDrains, 2);
  assert.equal(reply.stats.underrunFadeCount, 1);
  assert.equal(reply.stats.recoveryFadeCount, 1);
  assert.equal(reply.stats.sanitizedNonFiniteSamples, 1);
  assert.equal(reply.stats.softClippedSamples, 1);

  // Dirty snapshot sanitization clamps targetLatencyMs to [10, 120] and strips host paths / negatives / NaN.
  const dirty = buildAudioTelemetrySnapshot(
    {
      targetLatencyMs: 9999,
      zeroAllocDrains: -10,
      underrunFadeCount: Number.NaN,
      recoveryFadeCount: 3.9,
      sanitizedNonFiniteSamples: -4,
      softClippedSamples: Number.POSITIVE_INFINITY,
      hostPath: 'C:\\secret\\game.rpf',
    },
    { timestamp: 7777 }
  );
  assert.equal(dirty.stats.targetLatencyMs, 120);
  assert.equal(dirty.stats.zeroAllocDrains, 0);
  assert.equal(dirty.stats.underrunFadeCount, 0);
  assert.equal(dirty.stats.recoveryFadeCount, 3);
  assert.equal(dirty.stats.sanitizedNonFiniteSamples, 0);
  assert.equal(dirty.stats.softClippedSamples, 0);
  assert.equal(JSON.stringify(dirty).includes('game.rpf'), false);

  // reset() zeroes counters while preserving configured targetLatencyMs.
  worker.reset();
  const afterReset = worker.getStats();
  assert.equal(afterReset.targetLatencyMs, 42);
  assert.equal(afterReset.zeroAllocDrains, 0);
  assert.equal(afterReset.underrunFadeCount, 0);
  assert.equal(afterReset.recoveryFadeCount, 0);
  assert.equal(afterReset.sanitizedNonFiniteSamples, 0);
  assert.equal(afterReset.softClippedSamples, 0);

  worker.destroy();
});
