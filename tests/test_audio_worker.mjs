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
  OUTPUT_STATES,
  DEFAULT_MAX_PCM_BYTES,
  extractSafePcmSamples,
  clonePcmForTransfer,
  sanitizeTransferList,
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
