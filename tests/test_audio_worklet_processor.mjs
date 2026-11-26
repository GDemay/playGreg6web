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

const { AudioRingBuffer } = require('../web/workers/audio_worker.js');
const workletMod = require('../web/workers/audio_worklet_processor.js');
const {
  PlayGta6AudioWorkletProcessor,
  PROCESSOR_NAME,
  DEFAULT_DECLICK_FRAMES,
  DEFAULT_FADE_IN_FRAMES,
  MAX_DECLICK_FRAMES,
  DEFAULT_SAMPLE_CLAMP_LIMIT,
  DEFAULT_MAX_BUFFERED_FRAMES,
  DEFAULT_HIGH_WATERMARK_FRAMES,
  DEFAULT_TARGET_WATERMARK_FRAMES,
  DEFAULT_TRIM_CROSSFADE_FRAMES,
  MAX_TRIM_CROSSFADE_FRAMES,
  DEFAULT_STATS_INTERVAL_QUANTA,
  clampFillRatio,
  HDR_WRITE_CURSOR,
  HDR_READ_CURSOR,
  HDR_CAPACITY_FRAMES,
  HDR_CHANNELS,
  HDR_UNDERRUNS,
  HDR_UNDERRUN_FRAMES,
  HDR_OVERFLOWS,
  RING_HEADER_INT32_COUNT,
  RING_HEADER_BYTES,
} = workletMod;

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

function makeConstantStereo(frameCount, leftVal, rightVal = leftVal) {
  const out = new Float32Array(frameCount * 2);
  for (let f = 0; f < frameCount; f++) {
    out[f * 2] = leftVal;
    out[f * 2 + 1] = rightVal;
  }
  return out;
}

/**
 * Writes raw Float32 samples directly into the SAB ring buffer payload and
 * advances HDR_WRITE_CURSOR atomically, simulating a native/Wasm guest audio
 * producer writing unvalidated PCM (including NaN / +-Infinity) into shared memory.
 */
function pushRawSabFrames(sab, rawSamples) {
  const ctrl = new Int32Array(sab, 0, RING_HEADER_INT32_COUNT);
  const cap = Atomics.load(ctrl, HDR_CAPACITY_FRAMES);
  const channels = Atomics.load(ctrl, HDR_CHANNELS);
  const modulus = cap * 2;
  const payload = new Float32Array(sab, RING_HEADER_BYTES, cap * channels);

  const inputFrames = Math.floor(rawSamples.length / channels);
  const write = Atomics.load(ctrl, HDR_WRITE_CURSOR);
  const read = Atomics.load(ctrl, HDR_READ_CURSOR);
  let used = write - read;
  if (used < 0) used += modulus;
  const free = Math.max(0, cap - used);
  const toWrite = Math.min(inputFrames, free);

  if (toWrite > 0) {
    const writeSlot = write >= cap ? write - cap : write;
    const firstFrames = Math.min(toWrite, cap - writeSlot);
    const secondFrames = toWrite - firstFrames;
    const firstSamples = firstFrames * channels;

    for (let i = 0; i < firstSamples; i++) {
      payload[writeSlot * channels + i] = rawSamples[i];
    }
    for (let i = 0; i < secondFrames * channels; i++) {
      payload[i] = rawSamples[firstSamples + i];
    }

    let nextWrite = write + toWrite;
    if (nextWrite >= modulus) nextWrite -= modulus;
    Atomics.store(ctrl, HDR_WRITE_CURSOR, nextWrite);
  }

  return toWrite;
}

function attachMessageRecorder(processor) {
  const replies = [];
  processor.port.postMessage = (msg) => {
    replies.push(msg);
  };
  return replies;
}

test('W154 exports DEFAULT_DECLICK_FRAMES, MAX_DECLICK_FRAMES, and preserves AudioWorkletGlobalScope registration', () => {
  assert.equal(PROCESSOR_NAME, 'playgta6-audio-processor');
  assert.equal(DEFAULT_DECLICK_FRAMES, 32);
  assert.equal(MAX_DECLICK_FRAMES, 4096);
  assert.equal(globalThis.PlayGta6AudioWorklet.DEFAULT_DECLICK_FRAMES, DEFAULT_DECLICK_FRAMES);
  assert.equal(globalThis.PlayGta6AudioWorklet.MAX_DECLICK_FRAMES, MAX_DECLICK_FRAMES);

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

  const ring = AudioRingBuffer.create(256, 2);
  const proc = new registered[0].ctor({
    processorOptions: { sab: ring.sab, declickFrames: DEFAULT_DECLICK_FRAMES },
  });
  assert.equal(proc.declickFrames, DEFAULT_DECLICK_FRAMES);
  assert.equal(proc.framesProcessed, 0);
  assert.equal(proc.sanitizedSamples, 0);
  assert.equal(proc.underrunFrames, 0);
});

test('W154 sanitizes NaN and +-Infinity to 0.0 across unity, scalar, muted, and ramping gain paths including wrap-around', () => {
  // Capacity 192 forces wrap-around across 128-frame quanta.
  const ring = AudioRingBuffer.create(192, 2);
  const processor = new PlayGta6AudioWorkletProcessor({
    processorOptions: {
      sab: ring.sab,
      gain: 1.0,
      smoothingFrames: 16,
    },
  });

  // 1. Steady-state gain === 1.0 (first half + wrap-around on second quantum)
  const q1 = makeConstantStereo(128, 0.25, -0.5);
  q1[0] = NaN; // frame 0 L
  q1[3] = Infinity; // frame 1 R
  q1[254] = -Infinity; // frame 127 L
  assert.equal(pushRawSabFrames(ring.sab, q1), 128);

  const out1 = makePlanarOutput(2, 128, 99.0);
  assert.equal(processor.process([], out1), true);
  assert.equal(out1[0][0][0], 0.0);
  assert.equal(out1[0][1][0], -0.5);
  assert.equal(out1[0][0][1], 0.25);
  assert.equal(out1[0][1][1], 0.0);
  assert.equal(out1[0][0][127], 0.0);
  assert.equal(out1[0][1][127], -0.5);
  assert.ok(out1[0][0].every(Number.isFinite));
  assert.ok(out1[0][1].every(Number.isFinite));
  assert.equal(processor.sanitizedSamples, 3);
  assert.equal(processor.framesProcessed, 128);

  // 2. Wrap-around quantum at steady-state gain === 1.0 (readSlot=128 -> firstFrames=64, secondFrames=64)
  const q2 = makeConstantStereo(128, 0.5, -0.25);
  q2[10 * 2] = NaN; // inside firstFrames
  q2[80 * 2 + 1] = -Infinity; // inside secondFrames (wrapped)
  assert.equal(pushRawSabFrames(ring.sab, q2), 128);

  const out2 = makePlanarOutput(2, 128, -7.0);
  assert.equal(processor.process([], out2), true);
  assert.equal(out2[0][0][10], 0.0);
  assert.equal(out2[0][1][10], -0.25);
  assert.equal(out2[0][0][80], 0.5);
  assert.equal(out2[0][1][80], 0.0);
  assert.equal(processor.sanitizedSamples, 5);
  assert.equal(processor.framesProcessed, 256);

  // 3. Steady-state scalar gain === 0.5
  processor.port.onmessage({ data: { type: 'setGain', gain: 0.5, immediate: true } });
  const q3 = makeConstantStereo(128, 0.8, -0.6);
  q3[4 * 2] = Infinity;
  q3[100 * 2 + 1] = NaN;
  assert.equal(pushRawSabFrames(ring.sab, q3), 128);

  const out3 = makePlanarOutput(2, 128);
  assert.equal(processor.process([], out3), true);
  assert.equal(out3[0][0][4], 0.0);
  assert.equal(out3[0][1][4], Math.fround(-0.3));
  assert.equal(out3[0][0][100], Math.fround(0.4));
  assert.equal(out3[0][1][100], 0.0);
  assert.equal(processor.sanitizedSamples, 7);

  // 4. Ramping gain path (0.5 -> 0.0 over 16 frames) with non-finite samples during ramp AND after ramp hits 0.0
  processor.port.onmessage({ data: { type: 'setMuted', muted: true } });
  const q4 = makeConstantStereo(128, 0.4, -0.4);
  q4[2 * 2] = NaN; // during ramp (gain > 0)
  q4[20 * 2 + 1] = -Infinity; // after ramp hits 0.0
  q4[90 * 2] = Infinity; // in wrapped segment after ramp hits 0.0
  assert.equal(pushRawSabFrames(ring.sab, q4), 128);

  const out4 = makePlanarOutput(2, 128, 5.0);
  assert.equal(processor.process([], out4), true);
  assert.equal(out4[0][0][2], 0.0);
  assert.ok(out4[0][1][2] < 0); // finite right channel still ramped
  assert.equal(out4[0][1][20], 0.0);
  assert.equal(out4[0][0][90], 0.0);
  assert.ok(out4[0][0].every(Number.isFinite));
  assert.ok(out4[0][1].every(Number.isFinite));
  assert.equal(processor.sanitizedSamples, 10);

  // 5. Steady-state gain === 0.0 fast path still counts non-finite samples and outputs exact silence
  const q5 = makeConstantStereo(128, 0.2, 0.2);
  q5[0] = NaN;
  q5[1] = Infinity;
  q5[255] = -Infinity;
  assert.equal(pushRawSabFrames(ring.sab, q5), 128);

  const out5 = makePlanarOutput(2, 128, 12.0);
  assert.equal(processor.process([], out5), true);
  assert.ok(out5[0][0].every((v) => v === 0));
  assert.ok(out5[0][1].every((v) => v === 0));
  assert.equal(processor.sanitizedSamples, 13);
  assert.equal(processor.framesProcessed, 640);
});

test('W154 sanitizes non-finite samples for mono-to-stereo upmix and 4-channel planar output', () => {
  // Mono-to-stereo: 1 corrupted input sample must sanitize both L & R and count once.
  const ringMono = AudioRingBuffer.create(192, 1);
  const procMono = new PlayGta6AudioWorkletProcessor({
    processorOptions: { sab: ringMono.sab, gain: 1.0 },
  });

  const mono = new Float32Array(128);
  mono.fill(0.5);
  mono[5] = NaN;
  mono[70] = Infinity;
  assert.equal(pushRawSabFrames(ringMono.sab, mono), 128);

  const outStereo = makePlanarOutput(2, 128, 3.0);
  assert.equal(procMono.process([], outStereo), true);
  assert.equal(outStereo[0][0][5], 0.0);
  assert.equal(outStereo[0][1][5], 0.0);
  assert.equal(outStereo[0][0][70], 0.0);
  assert.equal(outStereo[0][1][70], 0.0);
  assert.equal(outStereo[0][0][0], 0.5);
  assert.equal(outStereo[0][1][0], 0.5);
  assert.equal(procMono.sanitizedSamples, 2);

  // 4-channel interleaved -> 4-channel planar at scalar gain 0.5
  const ring4 = AudioRingBuffer.create(192, 4);
  const proc4 = new PlayGta6AudioWorkletProcessor({
    processorOptions: { sab: ring4.sab, gain: 0.5 },
  });

  const pcm4 = new Float32Array(128 * 4);
  pcm4.fill(0.4);
  pcm4[3 * 4 + 2] = -Infinity;
  pcm4[100 * 4 + 3] = NaN;
  assert.equal(pushRawSabFrames(ring4.sab, pcm4), 128);

  const out4 = makePlanarOutput(4, 128, 9.0);
  assert.equal(proc4.process([], out4), true);
  assert.equal(out4[0][2][3], 0.0);
  assert.equal(out4[0][3][100], 0.0);
  assert.equal(out4[0][0][3], Math.fround(0.2));
  assert.equal(proc4.sanitizedSamples, 2);
});

test('W154 partial underrun applies smooth linear fade-out de-click ramp from last emitted sample to 0.0', () => {
  const fadeFrames = 32;
  const ampL = Math.fround(0.8);
  const ampR = Math.fround(-0.4);
  const maxStepL = Math.abs(ampL) / fadeFrames + 1e-5;
  const maxStepR = Math.abs(ampR) / fadeFrames + 1e-5;

  const ring = AudioRingBuffer.create(256, 2);
  const processor = new PlayGta6AudioWorkletProcessor({
    processorOptions: {
      sab: ring.sab,
      declickFrames: fadeFrames,
    },
  });

  // Push 64 frames, request 128 -> 64 valid + 32 fade-out + 32 silence.
  assert.equal(ring.pushSamples(makeConstantStereo(64, ampL, ampR)), 64);

  const out = makePlanarOutput(2, 128, 77.0);
  assert.equal(processor.process([], out), true);

  const [outL, outR] = out[0];

  // Valid region is bit-exact.
  for (let f = 0; f < 64; f++) {
    assert.equal(outL[f], ampL);
    assert.equal(outR[f], ampR);
  }

  // First fade frame steps smoothly from last emitted sample (not straight to 0).
  assert.ok(outL[64] > 0.7 && outL[64] < ampL);
  assert.ok(outR[64] < -0.35 && outR[64] > ampR);

  // Strictly monotonic linear ramp down to exact 0.0 at frame 64 + 31 = 95.
  for (let f = 64; f < 96; f++) {
    const stepL = Math.abs(outL[f] - outL[f - 1]);
    const stepR = Math.abs(outR[f] - outR[f - 1]);
    assert.ok(stepL <= maxStepL, `L step at frame ${f} (${stepL}) exceeded ${maxStepL}`);
    assert.ok(stepR <= maxStepR, `R step at frame ${f} (${stepR}) exceeded ${maxStepR}`);
    assert.ok(outL[f] <= outL[f - 1] + 1e-6);
    assert.ok(outR[f] >= outR[f - 1] - 1e-6);
  }

  // Final fade frame and remaining tail are exact 0.0.
  for (let f = 95; f < 128; f++) {
    assert.equal(outL[f], 0.0);
    assert.equal(outR[f], 0.0);
  }

  assert.equal(processor.framesProcessed, 64);
  assert.equal(processor.underrunFrames, 64);
  assert.equal(processor.underruns, 1);
  assert.equal(processor.fadeOutCount, 1);
});

test('W154 sudden full underrun after steady-state fades smoothly from prior quantum last sample and consecutive underruns stay silent', () => {
  const fadeFrames = 16;
  const amp = Math.fround(0.8);
  const maxStep = amp / fadeFrames + 1e-5;

  const ring = AudioRingBuffer.create(256, 2);
  const processor = new PlayGta6AudioWorkletProcessor({
    processorOptions: {
      sab: ring.sab,
      declickFrames: fadeFrames,
    },
  });

  // Quantum 1: full 128 frames of +0.8 / -0.8.
  assert.equal(ring.pushSamples(makeConstantStereo(128, amp, -amp)), 128);
  const q1Out = makePlanarOutput(2, 128);
  assert.equal(processor.process([], q1Out), true);
  assert.equal(q1Out[0][0][127], amp);
  assert.equal(q1Out[0][1][127], -amp);
  assert.equal(processor.fadeOutCount, 0);

  // Quantum 2: 0 frames available (sudden full underrun).
  const q2Out = makePlanarOutput(2, 128, 42.0);
  assert.equal(processor.process([], q2Out), true);

  // Boundary step from Quantum 1 frame 127 -> Quantum 2 frame 0 is bounded.
  assert.ok(Math.abs(q2Out[0][0][0] - amp) <= maxStep);
  assert.ok(Math.abs(q2Out[0][1][0] - -amp) <= maxStep);
  assert.ok(q2Out[0][0][0] > 0.7);
  assert.equal(q2Out[0][0][fadeFrames - 1], 0.0);
  assert.equal(q2Out[0][1][fadeFrames - 1], 0.0);

  for (let f = 1; f < 128; f++) {
    assert.ok(Math.abs(q2Out[0][0][f] - q2Out[0][0][f - 1]) <= maxStep);
    assert.ok(Math.abs(q2Out[0][1][f] - q2Out[0][1][f - 1]) <= maxStep);
  }
  for (let f = fadeFrames; f < 128; f++) {
    assert.equal(q2Out[0][0][f], 0.0);
    assert.equal(q2Out[0][1][f], 0.0);
  }
  assert.equal(processor.fadeOutCount, 1);
  assert.equal(processor.underruns, 1);
  assert.equal(processor.underrunFrames, 128);

  // Quantum 3: consecutive empty underrun stays silent and does NOT double-count fadeOutCount.
  const q3Out = makePlanarOutput(2, 128, -9.0);
  assert.equal(processor.process([], q3Out), true);
  assert.ok(q3Out[0][0].every((v) => v === 0.0));
  assert.ok(q3Out[0][1].every((v) => v === 0.0));
  assert.equal(processor.fadeOutCount, 1);
  assert.equal(processor.underruns, 2);
  assert.equal(processor.underrunFrames, 256);
});

test('W154 de-click fade-out spans seamlessly across quantum boundary when missing tail < declickFrames', () => {
  const fadeFrames = 32;
  const amp = Math.fround(0.8);
  const maxStep = amp / fadeFrames + 1e-5;

  const ring = AudioRingBuffer.create(256, 2);
  const processor = new PlayGta6AudioWorkletProcessor({
    processorOptions: {
      sab: ring.sab,
      declickFrames: fadeFrames,
    },
  });

  // Quantum 1: 112 valid frames -> 16 missing frames (< 32 fadeFrames).
  assert.equal(ring.pushSamples(makeConstantStereo(112, amp, -amp)), 112);
  const q1 = makePlanarOutput(2, 128);
  assert.equal(processor.process([], q1), true);
  assert.equal(q1[0][0][111], amp);
  // At frame 127 (i=15), scale = (32 - 1 - 15) / 32 = 16/32 = 0.5 -> 0.4.
  assert.ok(Math.abs(q1[0][0][127] - amp * 0.5) < 1e-5);
  assert.equal(processor.fadeOutCount, 1);

  // Quantum 2: 0 frames available -> continues remaining 16 fade frames smoothly to 0.0.
  const q2 = makePlanarOutput(2, 128, 11.0);
  assert.equal(processor.process([], q2), true);

  // Boundary between Q1[127] and Q2[0] has step exact amp / 32.
  const boundaryStep = Math.abs(q2[0][0][0] - q1[0][0][127]);
  assert.ok(boundaryStep <= maxStep, `Cross-quantum boundary step ${boundaryStep} exceeded ${maxStep}`);
  assert.ok(Math.abs(q2[0][0][0] - amp * (15 / 32)) < 1e-5);
  assert.equal(q2[0][0][15], 0.0);
  assert.equal(q2[0][1][15], 0.0);
  for (let f = 16; f < 128; f++) {
    assert.equal(q2[0][0][f], 0.0);
    assert.equal(q2[0][1][f], 0.0);
  }
  assert.equal(processor.fadeOutCount, 1);
  assert.equal(processor.framesProcessed, 112);
  assert.equal(processor.underrunFrames, 16 + 128);

  // Recovery: push 128 frames -> smooth linear fade-in back to (0.5, -0.5),
  // then next underrun triggers fadeOutCount === 2.
  assert.equal(ring.pushSamples(makeConstantStereo(128, 0.5, -0.5)), 128);
  const q3 = makePlanarOutput(2, 128);
  assert.equal(processor.process([], q3), true);
  assert.equal(q3[0][0][0], 0.0);
  assert.equal(q3[0][1][0], 0.0);
  assert.equal(q3[0][0][fadeFrames], 0.5);
  assert.equal(q3[0][1][127], -0.5);
  assert.equal(processor.fadeInCount, 1);

  const q4 = makePlanarOutput(2, 128);
  assert.equal(processor.process([], q4), true);
  assert.ok(q4[0][0][0] > 0.4 && q4[0][0][0] < 0.5);
  assert.equal(processor.fadeOutCount, 2);
});

test('W154 non-finite sample immediately before underrun is sanitized and never leaks NaN into fade-out ramp', () => {
  const ring = AudioRingBuffer.create(256, 2);
  const processor = new PlayGta6AudioWorkletProcessor({
    processorOptions: {
      sab: ring.sab,
      declickFrames: 16,
    },
  });

  // 40 frames where frame 39 (last emitted before underrun) is [NaN, -Infinity].
  const buf = makeConstantStereo(40, 0.5, -0.5);
  buf[39 * 2] = NaN;
  buf[39 * 2 + 1] = -Infinity;
  assert.equal(pushRawSabFrames(ring.sab, buf), 40);

  const out = makePlanarOutput(2, 128, 3.0);
  assert.equal(processor.process([], out), true);
  assert.ok(out[0][0].every(Number.isFinite));
  assert.ok(out[0][1].every(Number.isFinite));
  assert.equal(out[0][0][39], 0.0);
  assert.equal(out[0][1][39], 0.0);
  for (let f = 40; f < 128; f++) {
    assert.equal(out[0][0][f], 0.0);
    assert.equal(out[0][1][f], 0.0);
  }
  assert.equal(processor.sanitizedSamples, 2);
  assert.equal(processor.fadeOutCount, 0);
});

test('W154 stats message, getStats(), and setDeclickFrames control protocol report all telemetry counters accurately', () => {
  const ring = AudioRingBuffer.create(128, 2);
  const processor = new PlayGta6AudioWorkletProcessor({
    processorOptions: { sab: ring.sab },
  });
  const replies = attachMessageRecorder(processor);

  // Default declickFrames is 0 for backward compatibility.
  assert.equal(processor.declickFrames, 0);

  // Enable de-click via control message with id.
  processor.port.onmessage({ data: { type: 'setDeclickFrames', declickFrames: 24, id: 'dc-1' } });
  assert.deepEqual(replies[replies.length - 1], {
    type: 'declickFrames',
    declickFrames: 24,
    id: 'dc-1',
  });
  assert.equal(processor.declickFrames, 24);

  // Clamps excessive values and ignores negative/NaN values cleanly.
  assert.equal(processor.setDeclickFrames(999999), MAX_DECLICK_FRAMES);
  assert.equal(processor.setDeclickFrames(-5), MAX_DECLICK_FRAMES);
  assert.equal(processor.setDeclickFrames(NaN), MAX_DECLICK_FRAMES);
  assert.equal(processor.setDeclickFrames(16), 16);

  // Trigger an overflow on the ring producer side (push 150 into 128-frame ring -> 22 dropped).
  assert.equal(ring.pushSamples(makeConstantStereo(150, 0.5, -0.5)), 128);
  // Consume 128 frames, then inject 10 raw frames with 2 non-finite samples and trigger partial underrun.
  const q1 = makePlanarOutput(2, 128);
  assert.equal(processor.process([], q1), true);

  const bad10 = makeConstantStereo(10, 0.6, -0.6);
  bad10[0] = NaN;
  bad10[1] = Infinity;
  assert.equal(pushRawSabFrames(ring.sab, bad10), 10);
  const q2 = makePlanarOutput(2, 128);
  assert.equal(processor.process([], q2), true);

  processor.port.onmessage({ data: { type: 'stats', id: 42 } });
  const msg = replies[replies.length - 1];
  assert.equal(msg.type, 'stats');
  assert.equal(msg.id, 42);
  assert.equal(msg.attached, true);
  assert.equal(msg.processedQuanta, 2);
  assert.equal(msg.processedFrames, 138);
  assert.equal(msg.framesProcessed, 138);
  assert.equal(msg.underruns, 1);
  assert.equal(msg.underrunFrames, 118);
  assert.equal(msg.sanitizedSamples, 2);
  assert.equal(msg.overflows, 1);
  assert.equal(msg.overruns, 1);
  assert.equal(msg.droppedFrames, 22);
  assert.equal(msg.declickFrames, 16);
  assert.equal(msg.fadeOutCount, 1);

  // Re-attaching preserves exact attached reply shape and clears stale de-click state.
  const ring2 = AudioRingBuffer.create(256, 2);
  processor.port.onmessage({ data: { type: 'attach', sab: ring2.sab, id: 43 } });
  assert.deepEqual(replies[replies.length - 1], {
    type: 'attached',
    attached: true,
    capacityFrames: 256,
    channels: 2,
    id: 43,
  });

  // Empty ring2 must output pure silence (no stale lastSamples from ring1).
  const q3 = makePlanarOutput(2, 128, 5.0);
  assert.equal(processor.process([], q3), true);
  assert.ok(q3[0][0].every((v) => v === 0.0));
  assert.ok(q3[0][1].every((v) => v === 0.0));
});

test('W154 lock-free stress run maintains finite output and exact frame/sanitizer accounting without Atomics.wait', () => {
  const ring = AudioRingBuffer.create(192, 2);
  const processor = new PlayGta6AudioWorkletProcessor({
    processorOptions: {
      sab: ring.sab,
      declickFrames: DEFAULT_DECLICK_FRAMES,
      smoothingFrames: 32,
    },
  });

  const origWait = Atomics.wait;
  let waitCalls = 0;
  Atomics.wait = () => {
    waitCalls++;
    throw new Error('Atomics.wait must never be called on the AudioWorklet thread');
  };

  try {
    const out = makePlanarOutput(2, 128);
    let expectedSanitized = 0;
    const totalQuanta = 100;

    for (let q = 0; q < totalQuanta; q++) {
      if (q % 7 === 0) {
        processor.port.onmessage({ data: { type: 'setGain', gain: (q % 4) * 0.25 + 0.25 } });
      }
      if (q % 5 === 0) {
        // Partial underrun with injected non-finite samples.
        const chunk = makeConstantStereo(64, 0.5, -0.5);
        chunk[0] = NaN;
        chunk[13] = Infinity;
        chunk[40] = -Infinity;
        expectedSanitized += 3;
        assert.equal(pushRawSabFrames(ring.sab, chunk), 64);
      } else if (q % 5 !== 1) {
        // Full quantum with 1 corrupted sample.
        const chunk = makeConstantStereo(128, 0.3, -0.3);
        chunk[10] = NaN;
        expectedSanitized += 1;
        assert.equal(pushRawSabFrames(ring.sab, chunk), 128);
      }
      // q % 5 === 1 -> full empty underrun

      assert.equal(processor.process([], out), true);
      for (let f = 0; f < 128; f++) {
        assert.ok(Number.isFinite(out[0][0][f]), `non-finite L at q=${q} f=${f}`);
        assert.ok(Number.isFinite(out[0][1][f]), `non-finite R at q=${q} f=${f}`);
      }
    }

    const stats = processor.getStats();
    assert.equal(waitCalls, 0);
    assert.equal(stats.processedQuanta, totalQuanta);
    assert.equal(stats.framesProcessed + stats.underrunFrames, totalQuanta * 128);
    assert.equal(stats.sanitizedSamples, expectedSanitized);
    assert.ok(stats.fadeOutCount > 0);
  } finally {
    Atomics.wait = origWait;
  }
});

// ---------------------------------------------------------------------------
// W227: Zero-Alloc Render Quantum Hardening, Recovery Fade-In & Sample Clamping
// ---------------------------------------------------------------------------

test('W227 exports DEFAULT_FADE_IN_FRAMES, DEFAULT_SAMPLE_CLAMP_LIMIT, and control messages for fade-in & clamping', () => {
  assert.equal(DEFAULT_FADE_IN_FRAMES, DEFAULT_DECLICK_FRAMES);
  assert.equal(DEFAULT_SAMPLE_CLAMP_LIMIT, 1.0);
  assert.equal(globalThis.PlayGta6AudioWorklet.DEFAULT_FADE_IN_FRAMES, DEFAULT_FADE_IN_FRAMES);
  assert.equal(
    globalThis.PlayGta6AudioWorklet.DEFAULT_SAMPLE_CLAMP_LIMIT,
    DEFAULT_SAMPLE_CLAMP_LIMIT
  );

  const ring = AudioRingBuffer.create(256, 2);
  const processor = new PlayGta6AudioWorkletProcessor({
    processorOptions: { sab: ring.sab },
  });
  const replies = attachMessageRecorder(processor);

  // Defaults preserve full backward compatibility.
  assert.equal(processor.declickFrames, 0);
  assert.equal(processor.fadeInFrames, 0);
  assert.equal(processor.fadeInCount, 0);
  assert.equal(processor.inUnderrun, false);
  assert.equal(processor.clampSamples, false);
  assert.equal(processor.clampLimit, 0);
  assert.equal(processor.clampedSamples, 0);

  // Enabling declickFrames automatically configures matching fadeInFrames unless overridden.
  processor.setDeclickFrames(32);
  assert.equal(processor.fadeInFrames, 32);

  // Explicit fade-in override via control message.
  processor.port.onmessage({
    data: { type: 'setFadeInFrames', fadeInFrames: 16, id: 'fi-1' },
  });
  assert.deepEqual(replies[replies.length - 1], {
    type: 'fadeInFrames',
    fadeInFrames: 16,
    id: 'fi-1',
  });
  assert.equal(processor.fadeInFrames, 16);
  assert.equal(processor.declickFrames, 32);

  // Reset override back to tracking declickFrames.
  assert.equal(processor.setFadeInFrames(null), 32);
  assert.equal(processor.fadeInFrames, 32);

  // Enable and customize sample clamping via control message.
  processor.port.onmessage({
    data: { type: 'setClampSamples', clampSamples: true, id: 'cl-1' },
  });
  assert.deepEqual(replies[replies.length - 1], {
    type: 'clampSamples',
    clampSamples: true,
    clampLimit: 1.0,
    id: 'cl-1',
  });
  assert.equal(processor.clampSamples, true);
  assert.equal(processor.clampLimit, 1.0);

  processor.port.onmessage({
    data: { type: 'setSampleClamp', clampLimit: 0.85, id: 'cl-2' },
  });
  assert.deepEqual(replies[replies.length - 1], {
    type: 'clampSamples',
    clampSamples: true,
    clampLimit: 0.85,
    id: 'cl-2',
  });
  assert.equal(processor.clampLimit, 0.85);

  // Disable sample clamping cleanly.
  assert.equal(processor.setClampSamples(false), 0);
  assert.equal(processor.clampSamples, false);

  const stats = processor.getStats();
  assert.equal(stats.fadeInFrames, 32);
  assert.equal(stats.fadeInCount, 0);
  assert.equal(stats.inUnderrun, false);
  assert.equal(stats.clampSamples, false);
  assert.equal(stats.clampLimit, 0);
  assert.equal(stats.clampedSamples, 0);
});

test('W227 underrun fade-out and recovery fade-in produce strictly click-free bounded ramps and exact steady-state drain', () => {
  const fadeFrames = 32;
  const ampL = Math.fround(0.8);
  const ampR = Math.fround(-0.6);
  const maxStepL = Math.abs(ampL) / fadeFrames + 1e-5;
  const maxStepR = Math.abs(ampR) / fadeFrames + 1e-5;

  const ring = AudioRingBuffer.create(512, 2);
  const processor = new PlayGta6AudioWorkletProcessor({
    processorOptions: {
      sab: ring.sab,
      declickFrames: fadeFrames,
    },
  });

  // 1. Steady-state drain: 2 full quanta -> 100% bit-exact, zero underruns/fades.
  assert.equal(ring.pushSamples(makeConstantStereo(256, ampL, ampR)), 256);
  for (let q = 0; q < 2; q++) {
    const out = makePlanarOutput(2, 128, 99.0);
    assert.equal(processor.process([], out), true);
    for (let f = 0; f < 128; f++) {
      assert.equal(out[0][0][f], ampL);
      assert.equal(out[0][1][f], ampR);
    }
  }
  assert.equal(ring.availableFrames(), 0);
  assert.equal(processor.underruns, 0);
  assert.equal(processor.fadeOutCount, 0);
  assert.equal(processor.fadeInCount, 0);
  assert.equal(processor.inUnderrun, false);

  // 2. Partial underrun (64 frames available): smooth linear fade-out to 0.0.
  assert.equal(ring.pushSamples(makeConstantStereo(64, ampL, ampR)), 64);
  const qUnderrun = makePlanarOutput(2, 128, -5.0);
  assert.equal(processor.process([], qUnderrun), true);
  assert.equal(qUnderrun[0][0][63], ampL);
  assert.equal(qUnderrun[0][0][64 + fadeFrames - 1], 0.0);
  assert.equal(qUnderrun[0][1][64 + fadeFrames - 1], 0.0);
  for (let f = 64; f < 128; f++) {
    assert.ok(Math.abs(qUnderrun[0][0][f] - qUnderrun[0][0][f - 1]) <= maxStepL);
    assert.ok(Math.abs(qUnderrun[0][1][f] - qUnderrun[0][1][f - 1]) <= maxStepR);
  }
  assert.equal(processor.inUnderrun, true);
  assert.equal(processor.fadeOutCount, 1);
  assert.equal(processor.fadeInCount, 0);
  assert.equal(ring.underruns, 1);
  assert.equal(ring.underrunFrames, 64);

  // 3. Recovery: push 128 frames -> smooth monotonic fade-in 0.0 -> (ampL, ampR).
  assert.equal(ring.pushSamples(makeConstantStereo(128, ampL, ampR)), 128);
  const qRecover = makePlanarOutput(2, 128, 7.0);
  assert.equal(processor.process([], qRecover), true);

  assert.equal(qRecover[0][0][0], 0.0);
  assert.equal(qRecover[0][1][0], 0.0);
  for (let f = 1; f <= fadeFrames; f++) {
    const stepL = Math.abs(qRecover[0][0][f] - qRecover[0][0][f - 1]);
    const stepR = Math.abs(qRecover[0][1][f] - qRecover[0][1][f - 1]);
    assert.ok(stepL <= maxStepL, `Recovery L step at frame ${f} (${stepL}) exceeded ${maxStepL}`);
    assert.ok(stepR <= maxStepR, `Recovery R step at frame ${f} (${stepR}) exceeded ${maxStepR}`);
    assert.ok(qRecover[0][0][f] >= qRecover[0][0][f - 1] - 1e-6);
    assert.ok(qRecover[0][1][f] <= qRecover[0][1][f - 1] + 1e-6);
  }
  for (let f = fadeFrames; f < 128; f++) {
    assert.equal(qRecover[0][0][f], ampL);
    assert.equal(qRecover[0][1][f], ampR);
  }
  assert.equal(processor.inUnderrun, false);
  assert.equal(processor.fadeInCount, 1);

  // 4. Post-recovery steady-state quantum is 100% bit-exact without re-triggering fade-in.
  assert.equal(ring.pushSamples(makeConstantStereo(128, ampL, ampR)), 128);
  const qSteady = makePlanarOutput(2, 128);
  assert.equal(processor.process([], qSteady), true);
  for (let f = 0; f < 128; f++) {
    assert.equal(qSteady[0][0][f], ampL);
    assert.equal(qSteady[0][1][f], ampR);
  }
  assert.equal(processor.fadeInCount, 1);
  assert.equal(processor.underruns, 1);
});

test('W227 mid-fade-out recovery crossfades seamlessly without jumping to zero and handles re-underrun mid-fade-in', () => {
  const fadeFrames = 32;
  const amp = Math.fround(0.8);
  const maxStep = amp / fadeFrames + 1e-5;

  const ring = AudioRingBuffer.create(256, 2);
  const processor = new PlayGta6AudioWorkletProcessor({
    processorOptions: {
      sab: ring.sab,
      declickFrames: fadeFrames,
    },
  });

  // Quantum 1: 112 frames -> 16 missing (< 32 fadeFrames) -> ends mid-fade-out at 0.4.
  assert.equal(ring.pushSamples(makeConstantStereo(112, amp, -amp)), 112);
  const q1 = makePlanarOutput(2, 128);
  assert.equal(processor.process([], q1), true);
  assert.ok(Math.abs(q1[0][0][127] - 0.4) < 1e-5);
  assert.equal(processor.inUnderrun, true);
  assert.equal(processor.fadeOutCount, 1);

  // Quantum 2: Only 8 fresh frames arrive mid-fade-out, immediately followed by another underrun!
  // Must crossfade smoothly from Q1[127] (0.4) toward +0.8 for 8 frames, then fade out smoothly to 0.0.
  assert.equal(ring.pushSamples(makeConstantStereo(8, amp, -amp)), 8);
  const q2 = makePlanarOutput(2, 128, 3.0);
  assert.equal(processor.process([], q2), true);

  // Boundary between Q1[127] and Q2[0] has zero jump (starts at exact 0.4, NOT 0.0).
  assert.ok(
    Math.abs(q2[0][0][0] - q1[0][0][127]) <= maxStep,
    `Mid-fade-out recovery jumped from ${q1[0][0][127]} to ${q2[0][0][0]}`
  );
  assert.ok(q2[0][0][0] > 0.39 && q2[0][0][0] < 0.41);

  // Every adjacent sample across the entire quantum (crossfade-in + immediate re-fade-out) is bounded.
  for (let f = 1; f < 128; f++) {
    const stepL = Math.abs(q2[0][0][f] - q2[0][0][f - 1]);
    const stepR = Math.abs(q2[0][1][f] - q2[0][1][f - 1]);
    assert.ok(stepL <= maxStep, `L discontinuity at frame ${f}: step ${stepL} > ${maxStep}`);
    assert.ok(stepR <= maxStep, `R discontinuity at frame ${f}: step ${stepR} > ${maxStep}`);
  }
  // Fade-out completes cleanly to exact 0.0.
  assert.equal(q2[0][0][8 + fadeFrames - 1], 0.0);
  assert.equal(q2[0][0][127], 0.0);
  assert.equal(processor.fadeInCount, 1);
  assert.equal(processor.fadeOutCount, 2);
  assert.equal(processor.inUnderrun, true);

  // Multi-quantum recovery fade-in when fadeInFrames (192) > 128-frame quantum.
  processor.setFadeInFrames(192);
  assert.equal(ring.pushSamples(makeConstantStereo(256, amp, -amp)), 256);

  const q3 = makePlanarOutput(2, 128);
  const q4 = makePlanarOutput(2, 128);
  assert.equal(processor.process([], q3), true);
  assert.equal(processor.process([], q4), true);

  const maxStep192 = amp / 192 + 1e-5;
  assert.equal(q3[0][0][0], 0.0);
  assert.ok(Math.abs(q4[0][0][0] - q3[0][0][127]) <= maxStep192);
  assert.ok(q3[0][0][127] > 0.5 && q3[0][0][127] < 0.55);
  assert.equal(q4[0][0][64], amp);
  assert.equal(q4[0][1][127], -amp);
  assert.equal(processor.fadeInCount, 2);
});

test('W227 clamps finite out-of-range samples and sanitizes NaN/+-Infinity safely across all gain & channel paths', () => {
  const ring = AudioRingBuffer.create(192, 2);
  const processor = new PlayGta6AudioWorkletProcessor({
    processorOptions: {
      sab: ring.sab,
      gain: 1.0,
      smoothingFrames: 16,
      clampSamples: true,
    },
  });

  // 1. Unity gain + wrap-around: mix of NaN, +-Inf, out-of-range finite (+3.5, -2.5), and in-range samples.
  const q1 = makeConstantStereo(128, 0.5, -0.5);
  q1[0] = NaN;
  q1[1] = 3.5; // > +1.0 -> +1.0
  q1[2] = -2.5; // < -1.0 -> -1.0
  q1[3] = Infinity;
  q1[254] = -Infinity;
  q1[255] = 1e20; // > +1.0 -> +1.0
  assert.equal(pushRawSabFrames(ring.sab, q1), 128);

  const out1 = makePlanarOutput(2, 128);
  assert.equal(processor.process([], out1), true);
  assert.equal(out1[0][0][0], 0.0);
  assert.equal(out1[0][1][0], 1.0);
  assert.equal(out1[0][0][1], -1.0);
  assert.equal(out1[0][1][1], 0.0);
  assert.equal(out1[0][0][127], 0.0);
  assert.equal(out1[0][1][127], 1.0);
  assert.equal(out1[0][0][10], 0.5);
  assert.equal(out1[0][1][10], -0.5);
  assert.equal(processor.sanitizedSamples, 3);
  assert.equal(processor.clampedSamples, 3);

  // 2. Wrap-around quantum at scalar gain 0.5 with custom clampLimit 0.8.
  processor.setClampSamples(0.8);
  processor.port.onmessage({ data: { type: 'setGain', gain: 0.5, immediate: true } });
  const q2 = makeConstantStereo(128, 0.4, -0.4);
  q2[10 * 2] = 2.0; // clamped to 0.8 -> * 0.5 = 0.4
  q2[80 * 2 + 1] = -9.0; // wrapped segment, clamped to -0.8 -> * 0.5 = -0.4
  q2[81 * 2] = NaN;
  assert.equal(pushRawSabFrames(ring.sab, q2), 128);

  const out2 = makePlanarOutput(2, 128);
  assert.equal(processor.process([], out2), true);
  assert.equal(out2[0][0][10], Math.fround(0.4));
  assert.equal(out2[0][1][80], Math.fround(-0.4));
  assert.equal(out2[0][0][81], 0.0);
  assert.equal(processor.sanitizedSamples, 4);
  assert.equal(processor.clampedSamples, 5);

  // 3. Ramping gain path + muted steady-state path both clamp/count accurately.
  processor.setClampSamples(1.0);
  processor.port.onmessage({ data: { type: 'setMuted', muted: true } });
  const q3 = makeConstantStereo(128, 0.25, -0.25);
  q3[0] = 5.0; // clamped to 1.0 during ramp
  q3[20 * 2 + 1] = -5.0; // clamped during post-ramp gain===0
  assert.equal(pushRawSabFrames(ring.sab, q3), 128);
  const out3 = makePlanarOutput(2, 128);
  assert.equal(processor.process([], out3), true);
  assert.ok(out3[0][0].every((v) => Number.isFinite(v) && Math.abs(v) <= 1.0));
  assert.ok(out3[0][1].every((v) => Number.isFinite(v) && Math.abs(v) <= 1.0));
  assert.equal(processor.clampedSamples, 7);

  // Muted steady-state fast path still tracks out-of-range & non-finite samples.
  const q4 = makeConstantStereo(128, 0.2, -0.2);
  q4[0] = 4.0;
  q4[1] = -Infinity;
  assert.equal(pushRawSabFrames(ring.sab, q4), 128);
  const out4 = makePlanarOutput(2, 128, 9.0);
  assert.equal(processor.process([], out4), true);
  assert.ok(out4[0][0].every((v) => v === 0.0));
  assert.ok(out4[0][1].every((v) => v === 0.0));
  assert.equal(processor.sanitizedSamples, 5);
  assert.equal(processor.clampedSamples, 8);

  // 4. Mono-to-stereo & 4-channel clamping.
  const ringMono = AudioRingBuffer.create(128, 1);
  const procMono = new PlayGta6AudioWorkletProcessor({
    processorOptions: { sab: ringMono.sab, clampSamples: true },
  });
  const mono = new Float32Array(128);
  mono.fill(0.25);
  mono[3] = 2.5;
  mono[4] = -3.0;
  assert.equal(pushRawSabFrames(ringMono.sab, mono), 128);
  const outMono = makePlanarOutput(2, 128);
  assert.equal(procMono.process([], outMono), true);
  assert.equal(outMono[0][0][3], 1.0);
  assert.equal(outMono[0][1][3], 1.0);
  assert.equal(outMono[0][0][4], -1.0);
  assert.equal(outMono[0][1][4], -1.0);
  assert.equal(procMono.clampedSamples, 2);

  const ring4 = AudioRingBuffer.create(128, 4);
  const proc4 = new PlayGta6AudioWorkletProcessor({
    processorOptions: { sab: ring4.sab, sampleClampLimit: 1.0 },
  });
  const pcm4 = new Float32Array(128 * 4);
  pcm4.fill(0.1);
  pcm4[2 * 4 + 3] = -4.2;
  pcm4[5 * 4 + 1] = 8.8;
  assert.equal(pushRawSabFrames(ring4.sab, pcm4), 128);
  const out4ch = makePlanarOutput(4, 128);
  assert.equal(proc4.process([], out4ch), true);
  assert.equal(out4ch[0][3][2], -1.0);
  assert.equal(out4ch[0][1][5], 1.0);
  assert.equal(proc4.clampedSamples, 2);
});

test('W227 process() performs strictly zero heap allocations across 256 quanta of steady-state, underrun, recovery, and clamping', () => {
  const src = fs.readFileSync(
    path.resolve(__dirname, '../web/workers/audio_worklet_processor.js'),
    'utf8'
  );

  let allocationCount = 0;
  let trackingAllocations = false;
  let waitCalls = 0;

  function trackCtor(RealCtor) {
    return new Proxy(RealCtor, {
      construct(target, args, newTarget) {
        if (trackingAllocations) allocationCount++;
        return Reflect.construct(target, args, newTarget);
      },
      apply(target, thisArg, args) {
        if (trackingAllocations) allocationCount++;
        return Reflect.apply(target, thisArg, args);
      },
    });
  }

  const trackedAtomics = {
    load: Atomics.load,
    store: Atomics.store,
    add: Atomics.add,
    wait() {
      waitCalls++;
      throw new Error('Atomics.wait must never be called on AudioWorklet thread');
    },
  };

  let RegisteredProcessor = null;
  class MockBase {
    constructor() {
      this.port = { onmessage: null, postMessage() {} };
    }
  }

  const sandbox = {
    AudioWorkletProcessor: MockBase,
    registerProcessor(_name, ctor) {
      RegisteredProcessor = ctor;
    },
    Array: trackCtor(Array),
    Float32Array: trackCtor(Float32Array),
    Int32Array: trackCtor(Int32Array),
    Uint8Array: trackCtor(Uint8Array),
    ArrayBuffer: trackCtor(ArrayBuffer),
    SharedArrayBuffer: trackCtor(SharedArrayBuffer),
    Object: new Proxy(Object, {
      construct(target, args, newTarget) {
        if (trackingAllocations) allocationCount++;
        return Reflect.construct(target, args, newTarget);
      },
      apply(target, thisArg, args) {
        if (trackingAllocations) allocationCount++;
        return Reflect.apply(target, thisArg, args);
      },
      get(target, prop, receiver) {
        if (prop === 'assign') {
          return (...args) => {
            if (trackingAllocations) allocationCount++;
            return Reflect.apply(target.assign, target, args);
          };
        }
        if (prop === 'create') {
          return (...args) => {
            if (trackingAllocations) allocationCount++;
            return Reflect.apply(target.create, target, args);
          };
        }
        return Reflect.get(target, prop, receiver);
      },
    }),
    Atomics: trackedAtomics,
    Number,
    Math,
    Boolean,
    String,
    Error,
    TypeError,
    RangeError,
  };
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;

  vm.runInNewContext(src, sandbox, { filename: 'audio_worklet_processor.js' });
  assert.equal(typeof RegisteredProcessor, 'function');

  const ring = AudioRingBuffer.create(192, 2);
  const processor = new RegisteredProcessor({
    processorOptions: {
      sab: ring.sab,
      declickFrames: 32,
      smoothingFrames: 16,
      clampSamples: true,
    },
  });

  // Pre-allocate reusable producer & output buffers outside process().
  const out = makePlanarOutput(2, 128);
  const fullChunk = makeConstantStereo(128, 0.5, -0.5);
  fullChunk[4] = NaN;
  fullChunk[5] = 2.5;
  const partialChunk = makeConstantStereo(96, 0.6, -0.6);
  partialChunk[10] = -Infinity;
  partialChunk[11] = -3.0;

  let expectedFrames = 0;
  let expectedUnderruns = 0;
  let expectedUnderrunFrames = 0;

  for (let q = 0; q < 256; q++) {
    if (q % 8 === 3) {
      pushRawSabFrames(ring.sab, partialChunk);
      expectedFrames += 96;
      expectedUnderruns += 1;
      expectedUnderrunFrames += 32;
    } else if (q % 8 === 4) {
      // Full empty underrun.
      expectedUnderruns += 1;
      expectedUnderrunFrames += 128;
    } else {
      pushRawSabFrames(ring.sab, fullChunk);
      expectedFrames += 128;
    }

    trackingAllocations = true;
    const keepAlive = processor.process([], out);
    trackingAllocations = false;

    assert.equal(keepAlive, true);
    for (let f = 0; f < 128; f++) {
      const l = out[0][0][f];
      const r = out[0][1][f];
      assert.ok(Number.isFinite(l) && Math.abs(l) <= 1.0);
      assert.ok(Number.isFinite(r) && Math.abs(r) <= 1.0);
    }
  }

  assert.equal(allocationCount, 0, `Expected 0 allocations in process(), got ${allocationCount}`);
  assert.equal(waitCalls, 0);
  assert.equal(processor.framesProcessed, expectedFrames);
  assert.equal(processor.underruns, expectedUnderruns);
  assert.equal(processor.underrunFrames, expectedUnderrunFrames);
  assert.equal(ring.underruns, expectedUnderruns);
  assert.equal(ring.underrunFrames, expectedUnderrunFrames);
  assert.equal(ring.availableFrames(), 0);
  assert.ok(processor.fadeOutCount > 0);
  assert.ok(processor.fadeInCount > 0);
  assert.ok(processor.sanitizedSamples > 0);
  assert.ok(processor.clampedSamples > 0);
});

test('W227 defends against corrupted SAB cursors and degenerate output arrays without throwing', () => {
  const ring = AudioRingBuffer.create(128, 2);
  const ctrl = new Int32Array(ring.sab, 0, RING_HEADER_INT32_COUNT);
  const processor = new PlayGta6AudioWorkletProcessor({
    processorOptions: { sab: ring.sab, declickFrames: 16, clampSamples: true },
  });

  // Corrupt read/write cursors with negative and out-of-range values.
  Atomics.store(ctrl, HDR_READ_CURSOR, -17);
  Atomics.store(ctrl, HDR_WRITE_CURSOR, 99999);

  const out = makePlanarOutput(2, 128, 5.0);
  assert.doesNotThrow(() => {
    assert.equal(processor.process([], out), true);
  });
  assert.ok(out[0][0].every((v) => Number.isFinite(v) && Math.abs(v) <= 1.0));
  assert.ok(out[0][1].every((v) => Number.isFinite(v) && Math.abs(v) <= 1.0));
  assert.ok(processor.getStats().bufferedFrames >= 0);
  assert.ok(processor.getStats().bufferedFrames <= 128);

  // Degenerate outputs never throw.
  assert.equal(processor.process([], []), true);
  assert.equal(processor.process([], [[]]), true);
  assert.equal(processor.process([], [[null]]), true);
  assert.equal(processor.process([], [[new Float32Array(0)]]), true);
});

// ---------------------------------------------------------------------------
// W327: Lock-Free Watermark Telemetry, Overrun Drift Guard & Zero-Click Splice
// ---------------------------------------------------------------------------

test('W327 exports watermark constants, clampFillRatio, and control messages with safe defaults', () => {
  assert.equal(DEFAULT_MAX_BUFFERED_FRAMES, 0);
  assert.equal(DEFAULT_HIGH_WATERMARK_FRAMES, 4800);
  assert.equal(DEFAULT_TARGET_WATERMARK_FRAMES, 1440);
  assert.equal(DEFAULT_TRIM_CROSSFADE_FRAMES, 32);
  assert.equal(MAX_TRIM_CROSSFADE_FRAMES, 512);
  assert.equal(DEFAULT_STATS_INTERVAL_QUANTA, 0);
  assert.equal(typeof clampFillRatio, 'function');
  assert.equal(globalThis.PlayGta6AudioWorklet.clampFillRatio, clampFillRatio);
  assert.equal(
    globalThis.PlayGta6AudioWorklet.DEFAULT_HIGH_WATERMARK_FRAMES,
    DEFAULT_HIGH_WATERMARK_FRAMES
  );
  assert.equal(
    globalThis.PlayGta6AudioWorklet.DEFAULT_TRIM_CROSSFADE_FRAMES,
    DEFAULT_TRIM_CROSSFADE_FRAMES
  );

  // clampFillRatio strictly clamps to finite [0, 1].
  assert.equal(clampFillRatio(0), 0.0);
  assert.equal(Object.is(clampFillRatio(-0), 0), true);
  assert.equal(clampFillRatio(-0.25), 0.0);
  assert.equal(clampFillRatio(0.42), 0.42);
  assert.equal(clampFillRatio(1.0), 1.0);
  assert.equal(clampFillRatio(2.75), 1.0);
  assert.equal(clampFillRatio(Number.NaN), 0.0);
  assert.equal(clampFillRatio(Number.POSITIVE_INFINITY), 0.0);
  assert.equal(clampFillRatio(Number.NEGATIVE_INFINITY), 0.0);
  assert.equal(clampFillRatio(undefined), 0.0);
  assert.equal(clampFillRatio('0.5'), 0.0);

  // Default processor keeps overrun guard & periodic stats disabled for backward compatibility.
  const ring = AudioRingBuffer.create(512, 2);
  const proc = new PlayGta6AudioWorkletProcessor({ processorOptions: { sab: ring.sab } });
  assert.equal(proc.maxBufferedFrames, 0);
  assert.equal(proc.targetWatermarkFrames, 0);
  assert.equal(proc.trimCrossfadeFrames, DEFAULT_TRIM_CROSSFADE_FRAMES);
  assert.equal(proc.statsIntervalQuanta, 0);
  assert.equal(proc.fillRatio, 0.0);
  assert.equal(proc.peakFillRatio, 0.0);
  assert.equal(proc.overrunTrimmedFrames, 0);
  assert.equal(proc.overrunTrimCount, 0);
  assert.equal(proc.underrunQuanta, 0);
  assert.equal(proc.quantaProcessed, 0);

  // overrunGuard: true enables 48 kHz defaults (4800 high / 1440 target).
  const guarded = new PlayGta6AudioWorkletProcessor({
    processorOptions: { sab: ring.sab, overrunGuard: true },
  });
  assert.equal(guarded.maxBufferedFrames, DEFAULT_HIGH_WATERMARK_FRAMES);
  assert.equal(guarded.targetWatermarkFrames, DEFAULT_TARGET_WATERMARK_FRAMES);

  // Control messages configure watermark guard & stats interval and guard invalid values.
  const replies = attachMessageRecorder(proc);
  proc.port.onmessage({
    data: {
      type: 'setWatermarkConfig',
      maxBufferedFrames: 384,
      targetWatermarkFrames: 256,
      trimCrossfadeFrames: 48,
      id: 'wm-1',
    },
  });
  assert.deepEqual(replies[replies.length - 1], {
    type: 'watermarkConfig',
    maxBufferedFrames: 384,
    targetWatermarkFrames: 256,
    trimCrossfadeFrames: 48,
    id: 'wm-1',
  });
  assert.equal(proc.maxBufferedFrames, 384);
  assert.equal(proc.targetWatermarkFrames, 256);
  assert.equal(proc.trimCrossfadeFrames, 48);

  // Target watermark clamps to maxBufferedFrames and ignores NaN/negative inputs.
  assert.equal(proc.setTargetWatermarkFrames(9999), 384);
  assert.equal(proc.setTargetWatermarkFrames(-10), 384);
  assert.equal(proc.setTargetWatermarkFrames(null), 192); // auto half-of-max
  assert.equal(proc.setTrimCrossfadeFrames(9999), MAX_TRIM_CROSSFADE_FRAMES);
  assert.equal(proc.setTrimCrossfadeFrames(-5), MAX_TRIM_CROSSFADE_FRAMES);
  assert.equal(proc.setTrimCrossfadeFrames(Number.NaN), MAX_TRIM_CROSSFADE_FRAMES);
  assert.equal(proc.setMaxBufferedFrames(-100), 384);
  assert.equal(proc.setMaxBufferedFrames(Number.NaN), 384);

  proc.port.onmessage({
    data: { type: 'setStatsInterval', statsIntervalQuanta: 16, id: 'si-1' },
  });
  assert.deepEqual(replies[replies.length - 1], {
    type: 'statsInterval',
    statsIntervalQuanta: 16,
    id: 'si-1',
  });
  assert.equal(proc.statsIntervalQuanta, 16);
  assert.equal(proc.setStatsIntervalQuanta(0), 0);
});

test('W327 overrun drift guard trims excess buffered frames to target watermark and bounds latency across bursts', () => {
  const ring = AudioRingBuffer.create(2048, 2);
  const processor = new PlayGta6AudioWorkletProcessor({
    processorOptions: {
      sab: ring.sab,
      maxBufferedFrames: 512,
      targetWatermarkFrames: 256,
      trimCrossfadeFrames: 0, // isolate exact read-cursor jump in this test
    },
  });

  // Push 1024 sequentially tagged stereo frames: frame i -> (i / 2048, -i / 2048).
  const burst = new Float32Array(1024 * 2);
  for (let f = 0; f < 1024; f++) {
    burst[f * 2] = f / 2048;
    burst[f * 2 + 1] = -f / 2048;
  }
  assert.equal(ring.pushSamples(burst), 1024);
  assert.equal(processor.fillRatio, 0.5);

  // Used (1024) > maxBufferedFrames (512) -> trims 1024 - 256 = 768 frames, then plays 128 frames
  // starting at frame 768, leaving 128 frames buffered and 0 underruns.
  const out = makePlanarOutput(2, 128);
  assert.equal(processor.process([], out), true);
  assert.equal(processor.overrunTrimmedFrames, 768);
  assert.equal(processor.overrunTrimCount, 1);
  assert.equal(processor.underrunQuanta, 0);
  assert.equal(processor.quantaProcessed, 1);
  assert.equal(ring.availableFrames(), 128);
  assert.equal(processor.fillRatio, 128 / 2048);
  assert.equal(processor.peakFillRatio, 1024 / 2048);

  // Verify exact post-trim frame indices [768 .. 895].
  for (let f = 0; f < 128; f++) {
    const expected = Math.fround((768 + f) / 2048);
    assert.equal(out[0][0][f], expected);
    assert.equal(out[0][1][f], -expected);
  }

  // Sustained burst stress (simulating producer running ahead of 30 FPS video across wrap-around):
  // latency must remain strictly bounded <= maxBufferedFrames after every quantum with 0 underruns.
  const chunk384 = makeConstantStereo(384, 0.25, -0.25);
  for (let i = 0; i < 30; i++) {
    ring.pushSamples(chunk384);
    assert.equal(processor.process([], out), true);
    assert.ok(
      ring.availableFrames() <= 512,
      `Buffered frames (${ring.availableFrames()}) drifted above ceiling at step ${i}`
    );
  }
  assert.equal(processor.underrunQuanta, 0);
  assert.ok(processor.overrunTrimmedFrames > 768);
  assert.ok(processor.overrunTrimCount > 1);

  // Even when targetWatermarkFrames < quantumFrames, guard keeps at least 1 full quantum so
  // trimming never self-inflicts an artificial underrun.
  const smallTargetRing = AudioRingBuffer.create(1024, 2);
  const smallTargetProc = new PlayGta6AudioWorkletProcessor({
    processorOptions: {
      sab: smallTargetRing.sab,
      maxBufferedFrames: 256,
      targetWatermarkFrames: 32, // < 128-frame quantum
      trimCrossfadeFrames: 0,
    },
  });
  assert.equal(smallTargetRing.pushSamples(makeConstantStereo(400, 0.5, -0.5)), 400);
  const outSmall = makePlanarOutput(2, 128);
  assert.equal(smallTargetProc.process([], outSmall), true);
  assert.equal(smallTargetProc.overrunTrimmedFrames, 400 - 128);
  assert.equal(smallTargetProc.underrunQuanta, 0);
  assert.equal(smallTargetRing.availableFrames(), 0);
  assert.equal(outSmall[0][0][127], 0.5);
});

test('W327 overrun trim applies zero-click half-cosine crossfade across stereo, mono, 4ch, wrap-around, and NaN/peak old-head inputs', () => {
  const xfadeFrames = 32;
  const ampL = Math.fround(0.8);
  const ampR = Math.fround(-0.6);

  // 1. Stereo polarity reversal (+0.8/-0.6 -> -0.8/+0.6) across an overrun trim boundary.
  const ring = AudioRingBuffer.create(1024, 2);
  const processor = new PlayGta6AudioWorkletProcessor({
    processorOptions: {
      sab: ring.sab,
      maxBufferedFrames: 384,
      targetWatermarkFrames: 256,
      trimCrossfadeFrames: xfadeFrames,
    },
  });

  // Quantum 1: normal steady-state playback at (+0.8, -0.6).
  assert.equal(ring.pushSamples(makeConstantStereo(128, ampL, ampR)), 128);
  const q1 = makePlanarOutput(2, 128);
  assert.equal(processor.process([], q1), true);
  assert.equal(q1[0][0][127], ampL);
  assert.equal(q1[0][1][127], ampR);
  assert.equal(processor.overrunTrimmedFrames, 0);

  // Producer burst: 256 frames of old polarity (+0.8, -0.6) followed by 256 frames of inverted polarity (-0.8, +0.6).
  // Total = 512 > maxBufferedFrames (384) -> trims 256 frames so new read head lands on (-0.8, +0.6).
  assert.equal(ring.pushSamples(makeConstantStereo(256, ampL, ampR)), 256);
  assert.equal(ring.pushSamples(makeConstantStereo(256, -ampL, -ampR)), 256);

  const q2 = makePlanarOutput(2, 128);
  assert.equal(processor.process([], q2), true);
  assert.equal(processor.overrunTrimmedFrames, 256);
  assert.equal(processor.overrunTrimCount, 1);

  const [outL, outR] = q2[0];
  // Zero discontinuity at quantum boundary (Q1[127] -> Q2[0]).
  assert.equal(outL[0], ampL);
  assert.equal(outR[0], ampR);
  // Exact 50/50 midpoint at f = xfadeFrames / 2 = 16 -> 0.0.
  assert.ok(Math.abs(outL[16]) < 1e-5, `Expected midpoint ~0.0, got ${outL[16]}`);
  assert.ok(Math.abs(outR[16]) < 1e-5, `Expected midpoint ~0.0, got ${outR[16]}`);
  // Post-crossfade frames match new stream bit-exact.
  for (let f = xfadeFrames; f < 128; f++) {
    assert.equal(outL[f], -ampL);
    assert.equal(outR[f], -ampR);
  }
  // Every adjacent sample delta across the full quantum is bounded by half-cosine max slope.
  const maxStepL = (Math.PI / (2 * xfadeFrames)) * Math.abs(2 * ampL) + 1e-4;
  const maxStepR = (Math.PI / (2 * xfadeFrames)) * Math.abs(2 * ampR) + 1e-4;
  for (let f = 1; f < 128; f++) {
    const stepL = Math.abs(outL[f] - outL[f - 1]);
    const stepR = Math.abs(outR[f] - outR[f - 1]);
    assert.ok(stepL <= maxStepL, `L click at frame ${f}: step ${stepL} > ${maxStepL}`);
    assert.ok(stepR <= maxStepR, `R click at frame ${f}: step ${stepR} > ${maxStepR}`);
  }

  // 2. Wrap-around overrun trim with NaN/Infinity and hot peaks (> 1.0) in discarded old head.
  const wrapRing = AudioRingBuffer.create(512, 2);
  const wrapProc = new PlayGta6AudioWorkletProcessor({
    processorOptions: {
      sab: wrapRing.sab,
      maxBufferedFrames: 320,
      targetWatermarkFrames: 192,
      trimCrossfadeFrames: 32,
      softLimit: true,
    },
  });
  // Advance cursors cleanly near end of ring (slot 496) without trimming or underrunning
  // to force ring wrap-around right inside the old-head crossfade!
  const step124 = makeConstantStereo(124, 0.2, -0.2);
  const drain124 = makePlanarOutput(2, 124);
  for (let i = 0; i < 4; i++) {
    assert.equal(wrapRing.pushSamples(step124), 124);
    assert.equal(wrapProc.process([], drain124), true);
  }
  assert.equal(wrapProc.overrunTrimmedFrames, 0);
  assert.equal(wrapProc.inUnderrun, false);

  const dirtyOldHead = makeConstantStereo(200, 0.5, -0.5);
  dirtyOldHead[0] = Number.NaN;
  dirtyOldHead[1] = Number.POSITIVE_INFINITY;
  dirtyOldHead[20 * 2] = 3.5; // wrapped slot, hot peak
  dirtyOldHead[20 * 2 + 1] = -4.2;
  assert.equal(pushRawSabFrames(wrapRing.sab, dirtyOldHead), 200);
  assert.equal(pushRawSabFrames(wrapRing.sab, makeConstantStereo(200, -0.4, 0.4)), 200);

  const wrapOut = makePlanarOutput(2, 128);
  assert.equal(wrapProc.process([], wrapOut), true);
  assert.ok(wrapProc.overrunTrimmedFrames > 0);
  for (let f = 0; f < 128; f++) {
    assert.ok(Number.isFinite(wrapOut[0][0][f]) && Math.abs(wrapOut[0][0][f]) <= 1.0);
    assert.ok(Number.isFinite(wrapOut[0][1][f]) && Math.abs(wrapOut[0][1][f]) <= 1.0);
  }
  // Frame 0 had NaN/Inf in old head -> sanitized to 0.0 (alpha=0 -> output[0] === 0.0).
  assert.equal(wrapOut[0][0][0], 0.0);
  assert.equal(wrapOut[0][1][0], 0.0);

  // 3. Mono-to-stereo & 4-channel trim crossfades stay finite and smooth.
  const monoRing = AudioRingBuffer.create(512, 1);
  const monoProc = new PlayGta6AudioWorkletProcessor({
    processorOptions: {
      sab: monoRing.sab,
      maxBufferedFrames: 256,
      targetWatermarkFrames: 128,
      trimCrossfadeFrames: 16,
    },
  });
  const monoBurst = new Float32Array(384);
  monoBurst.fill(0.6, 0, 256);
  monoBurst.fill(-0.6, 256, 384);
  assert.equal(monoRing.pushSamples(monoBurst), 384);
  const monoOut = makePlanarOutput(2, 128);
  assert.equal(monoProc.process([], monoOut), true);
  assert.equal(monoProc.overrunTrimmedFrames, 256);
  assert.equal(monoOut[0][0][0], Math.fround(0.6));
  assert.equal(monoOut[0][1][0], Math.fround(0.6));
  assert.ok(Math.abs(monoOut[0][0][8]) < 1e-5);
  assert.equal(monoOut[0][0][16], Math.fround(-0.6));
  assert.equal(monoOut[0][1][16], Math.fround(-0.6));

  const ring4 = AudioRingBuffer.create(512, 4);
  const proc4 = new PlayGta6AudioWorkletProcessor({
    processorOptions: {
      sab: ring4.sab,
      maxBufferedFrames: 256,
      targetWatermarkFrames: 128,
      trimCrossfadeFrames: 16,
    },
  });
  const burst4 = new Float32Array(384 * 4);
  burst4.fill(0.5, 0, 256 * 4);
  burst4.fill(-0.5, 256 * 4, 384 * 4);
  assert.equal(ring4.pushSamples(burst4), 384);
  const out4 = makePlanarOutput(4, 128);
  assert.equal(proc4.process([], out4), true);
  assert.equal(proc4.overrunTrimmedFrames, 256);
  for (let c = 0; c < 4; c++) {
    assert.equal(out4[0][c][0], 0.5);
    assert.ok(Math.abs(out4[0][c][8]) < 1e-5);
    assert.equal(out4[0][c][16], -0.5);
  }
});

test('W327 fillRatio clamping [0, 1], peakFillRatio tracking, and periodic port stats telemetry', () => {
  // Detached processor reports exact 0.0 fillRatio and peakFillRatio.
  const detached = new PlayGta6AudioWorkletProcessor();
  assert.equal(detached.fillRatio, 0.0);
  assert.equal(detached.peakFillRatio, 0.0);
  assert.equal(detached.getStats().fillRatio, 0.0);
  assert.equal(detached.getStats().peakFillRatio, 0.0);

  const ring = AudioRingBuffer.create(512, 2);
  const processor = new PlayGta6AudioWorkletProcessor({
    processorOptions: {
      sab: ring.sab,
      maxBufferedFrames: 384,
      targetWatermarkFrames: 256,
      statsIntervalQuanta: 2,
    },
  });

  // Snapshot each posted message (mimicking MessagePort structured clone) while also verifying
  // periodic port stats reuse the same pre-allocated payload object (zero-alloc!).
  const postedRefs = [];
  const snapshots = [];
  processor.port.postMessage = (msg) => {
    postedRefs.push(msg);
    snapshots.push({ ...msg });
  };

  // Push 512 frames (100% full) -> fillRatio === 1.0.
  assert.equal(ring.pushSamples(makeConstantStereo(512, 0.4, -0.4)), 512);
  assert.equal(processor.fillRatio, 1.0);

  // Quantum 1: trims 256, consumes 128 -> remaining 128 (0.25), peakFillRatio = 1.0, no periodic stats yet.
  const out = makePlanarOutput(2, 128);
  assert.equal(processor.process([], out), true);
  assert.equal(snapshots.length, 0);
  assert.equal(processor.fillRatio, 0.25);
  assert.equal(processor.peakFillRatio, 1.0);

  // Quantum 2: consumes remaining 128 -> remaining 0 (0.0), periodic stats fires!
  assert.equal(processor.process([], out), true);
  assert.equal(snapshots.length, 1);
  assert.equal(snapshots[0].type, 'stats');
  assert.equal(snapshots[0].periodic, true);
  assert.equal(snapshots[0].quantaProcessed, 2);
  assert.equal(snapshots[0].processedQuanta, 2);
  assert.equal(snapshots[0].overrunTrimmedFrames, 256);
  assert.equal(snapshots[0].underrunQuanta, 0);
  assert.equal(snapshots[0].fillRatio, 0.0);
  assert.equal(snapshots[0].peakFillRatio, 1.0);

  // Quanta 3 & 4: empty underruns -> second periodic stats fires with underrunQuanta === 2.
  assert.equal(processor.process([], out), true);
  assert.equal(processor.process([], out), true);
  assert.equal(snapshots.length, 2);
  assert.equal(postedRefs[0], postedRefs[1], 'Periodic stats must reuse pre-allocated payload object');
  assert.equal(snapshots[1].quantaProcessed, 4);
  assert.equal(snapshots[1].underrunQuanta, 2);
  assert.equal(snapshots[1].fillRatio, 0.0);
  assert.equal(snapshots[1].peakFillRatio, 1.0);

  // Corrupted SAB cursors still clamp fillRatio strictly to [0, 1].
  const ctrl = new Int32Array(ring.sab, 0, RING_HEADER_INT32_COUNT);
  Atomics.store(ctrl, HDR_READ_CURSOR, -9999);
  Atomics.store(ctrl, HDR_WRITE_CURSOR, 999999);
  const corruptStats = processor.getStats();
  assert.ok(Number.isFinite(corruptStats.fillRatio));
  assert.ok(corruptStats.fillRatio >= 0.0 && corruptStats.fillRatio <= 1.0);
  assert.ok(corruptStats.peakFillRatio >= 0.0 && corruptStats.peakFillRatio <= 1.0);
});

test('W327 preserves W272 cosine underrun/recovery fade ramps and soft-limiter alongside overrun drift guard and resetStats()', () => {
  const ring = AudioRingBuffer.create(1024, 2);
  const processor = new PlayGta6AudioWorkletProcessor({
    processorOptions: {
      sab: ring.sab,
      cosineRamp: true,
      softLimit: true,
      maxBufferedFrames: 512,
      targetWatermarkFrames: 256,
      trimCrossfadeFrames: 32,
      statsIntervalQuanta: 4,
    },
  });

  // 1. Overrun burst with NaN/Inf and hot >0.9 peaks -> trims 384 frames with crossfade + soft-limit.
  const burst = makeConstantStereo(640, 0.8, -0.6);
  burst[384 * 2] = Number.NaN;
  burst[384 * 2 + 1] = 2.2;
  burst[(384 + 40) * 2] = 2.5;
  burst[(384 + 40) * 2 + 1] = Number.NEGATIVE_INFINITY;
  assert.equal(pushRawSabFrames(ring.sab, burst), 640);

  const q1 = makePlanarOutput(2, 128);
  assert.equal(processor.process([], q1), true);
  assert.equal(processor.overrunTrimmedFrames, 384);
  assert.equal(processor.overrunTrimCount, 1);
  assert.ok(q1[0][0][40] > 0.9 && q1[0][0][40] <= 1.0);
  assert.equal(q1[0][1][40], 0.0);
  // Bit-exact linear passthrough for normal |x| <= 0.9 after crossfade region.
  assert.equal(q1[0][0][60], Math.fround(0.8));
  assert.equal(q1[0][1][60], Math.fround(-0.6));
  assert.equal(processor.nonFiniteSanitized, 2);
  assert.equal(processor.softClippedSamples, 2);

  // Drain remaining 128 frames cleanly (no trim, no underrun).
  const q2 = makePlanarOutput(2, 128);
  assert.equal(processor.process([], q2), true);
  assert.equal(ring.availableFrames(), 0);
  assert.equal(q2[0][0][127], Math.fround(0.8));

  // 2. Sudden underrun -> W272 64-frame half-cosine fade-out preserved bit-exact.
  const q3 = makePlanarOutput(2, 128, 9.0);
  assert.equal(processor.process([], q3), true);
  const expectedCosFirst = 0.5 * (1.0 + Math.cos(Math.PI / 64));
  const expectedCosMid = 0.5 * (1.0 + Math.cos((32 * Math.PI) / 64));
  assert.ok(Math.abs(q3[0][0][0] - Math.fround(0.8) * expectedCosFirst) < 1e-5);
  assert.ok(Math.abs(q3[0][0][31] - Math.fround(0.8) * expectedCosMid) < 1e-5);
  assert.equal(q3[0][0][63], 0.0);
  assert.equal(q3[0][1][63], 0.0);
  assert.equal(processor.underrunFadeOuts, 1);
  assert.equal(processor.underrunQuanta, 1);

  // 3. Recovery -> W272 64-frame half-cosine fade-in preserved bit-exact.
  assert.equal(ring.pushSamples(makeConstantStereo(128, 0.7, -0.5)), 128);
  const q4 = makePlanarOutput(2, 128);
  assert.equal(processor.process([], q4), true);
  assert.equal(q4[0][0][0], 0.0);
  assert.equal(q4[0][1][0], 0.0);
  assert.ok(Math.abs(q4[0][0][32] - Math.fround(0.7) * 0.5) < 1e-5);
  assert.equal(q4[0][0][64], Math.fround(0.7));
  assert.equal(q4[0][1][64], Math.fround(-0.5));
  assert.equal(processor.recoveryFadeIns, 1);
  assert.equal(processor.quantaProcessed, 4);

  // 4. resetStats() clears all telemetry counters while preserving config & attachment.
  const replies = attachMessageRecorder(processor);
  processor.port.onmessage({ data: { type: 'resetStats', id: 'rs-1' } });
  assert.deepEqual(replies[replies.length - 1], { type: 'statsReset', id: 'rs-1' });

  const afterReset = processor.getStats();
  assert.equal(afterReset.attached, true);
  assert.equal(afterReset.quantaProcessed, 0);
  assert.equal(afterReset.processedQuanta, 0);
  assert.equal(afterReset.processedFrames, 0);
  assert.equal(afterReset.framesProcessed, 0);
  assert.equal(afterReset.underruns, 0);
  assert.equal(afterReset.underrunQuanta, 0);
  assert.equal(afterReset.underrunFrames, 0);
  assert.equal(afterReset.overrunTrimmedFrames, 0);
  assert.equal(afterReset.overrunTrimCount, 0);
  assert.equal(afterReset.peakFillRatio, 0.0);
  assert.equal(afterReset.sanitizedSamples, 0);
  assert.equal(afterReset.nonFiniteSanitized, 0);
  assert.equal(afterReset.clampedSamples, 0);
  assert.equal(afterReset.softClippedSamples, 0);
  assert.equal(afterReset.fadeOutCount, 0);
  assert.equal(afterReset.fadeInCount, 0);
  // Config stays intact.
  assert.equal(afterReset.fadeCurve, 'cosine');
  assert.equal(afterReset.softLimit, true);
  assert.equal(afterReset.maxBufferedFrames, 512);
  assert.equal(afterReset.targetWatermarkFrames, 256);
  assert.equal(afterReset.trimCrossfadeFrames, 32);
  assert.equal(afterReset.statsIntervalQuanta, 4);

  // Post-reset playback continues cleanly and tracks fresh stats from zero.
  assert.equal(ring.pushSamples(makeConstantStereo(128, 0.3, -0.3)), 128);
  const q5 = makePlanarOutput(2, 128);
  assert.equal(processor.process([], q5), true);
  assert.equal(q5[0][0][0], Math.fround(0.3));
  assert.equal(processor.quantaProcessed, 1);
  assert.equal(processor.framesProcessed, 128);
  assert.equal(processor.peakFillRatio, 128 / 1024);
});

test('W327 process() performs strictly zero allocations and zero Atomics.wait with overrun trim, crossfade, W272 cosine/softLimit, and periodic stats active', () => {
  const src = fs.readFileSync(
    path.resolve(__dirname, '../web/workers/audio_worklet_processor.js'),
    'utf8'
  );

  let allocationCount = 0;
  let trackingAllocations = false;
  let waitCalls = 0;
  let periodicMessages = 0;

  function trackCtor(RealCtor) {
    return new Proxy(RealCtor, {
      construct(target, args, newTarget) {
        if (trackingAllocations) allocationCount++;
        return Reflect.construct(target, args, newTarget);
      },
      apply(target, thisArg, args) {
        if (trackingAllocations) allocationCount++;
        return Reflect.apply(target, thisArg, args);
      },
    });
  }

  let RegisteredProcessor = null;
  class MockBase {
    constructor() {
      this.port = {
        onmessage: null,
        postMessage(msg) {
          if (msg && msg.type === 'stats' && msg.periodic) {
            periodicMessages++;
          }
        },
      };
    }
  }

  const sandbox = {
    AudioWorkletProcessor: MockBase,
    registerProcessor(_name, ctor) {
      RegisteredProcessor = ctor;
    },
    Array: trackCtor(Array),
    Float32Array: trackCtor(Float32Array),
    Int32Array: trackCtor(Int32Array),
    Uint8Array: trackCtor(Uint8Array),
    ArrayBuffer: trackCtor(ArrayBuffer),
    SharedArrayBuffer: trackCtor(SharedArrayBuffer),
    Object: new Proxy(Object, {
      construct(target, args, newTarget) {
        if (trackingAllocations) allocationCount++;
        return Reflect.construct(target, args, newTarget);
      },
      apply(target, thisArg, args) {
        if (trackingAllocations) allocationCount++;
        return Reflect.apply(target, thisArg, args);
      },
      get(target, prop, receiver) {
        if (prop === 'assign') {
          return (...args) => {
            if (trackingAllocations) allocationCount++;
            return Reflect.apply(target.assign, target, args);
          };
        }
        if (prop === 'create') {
          return (...args) => {
            if (trackingAllocations) allocationCount++;
            return Reflect.apply(target.create, target, args);
          };
        }
        return Reflect.get(target, prop, receiver);
      },
    }),
    Atomics: {
      load: Atomics.load,
      store: Atomics.store,
      add: Atomics.add,
      wait() {
        waitCalls++;
        throw new Error('Atomics.wait must never be called on AudioWorklet thread');
      },
    },
    Number,
    Math,
    Boolean,
    String,
    Error,
    TypeError,
    RangeError,
  };
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;

  vm.runInNewContext(src, sandbox, { filename: 'audio_worklet_processor.js' });
  assert.equal(typeof RegisteredProcessor, 'function');

  const ring = AudioRingBuffer.create(1024, 2);
  const processor = new RegisteredProcessor({
    processorOptions: {
      sab: ring.sab,
      cosineRamp: true,
      softLimit: true,
      smoothingFrames: 16,
      maxBufferedFrames: 384,
      targetWatermarkFrames: 256,
      trimCrossfadeFrames: 32,
      statsIntervalQuanta: 8,
    },
  });

  const out = makePlanarOutput(2, 128);
  const burstChunk = makeConstantStereo(512, 0.7, -0.7);
  burstChunk[10] = Number.NaN;
  burstChunk[11] = 2.4;
  const normalChunk = makeConstantStereo(128, -0.5, 0.5);

  const totalQuanta = 128;
  for (let q = 0; q < totalQuanta; q++) {
    if (q % 16 === 0) {
      // Trigger overrun trim + crossfade.
      pushRawSabFrames(ring.sab, burstChunk);
    } else if (q % 16 === 9 || q % 16 === 10) {
      // Let ring run dry to trigger W272 cosine fade-out & recovery fade-in.
    } else {
      pushRawSabFrames(ring.sab, normalChunk);
    }

    trackingAllocations = true;
    const keepAlive = processor.process([], out);
    trackingAllocations = false;

    assert.equal(keepAlive, true);
    for (let f = 0; f < 128; f++) {
      const l = out[0][0][f];
      const r = out[0][1][f];
      assert.ok(Number.isFinite(l) && Math.abs(l) <= 1.0);
      assert.ok(Number.isFinite(r) && Math.abs(r) <= 1.0);
    }
  }

  const stats = processor.getStats();
  assert.equal(allocationCount, 0, `Expected 0 allocations in process(), got ${allocationCount}`);
  assert.equal(waitCalls, 0);
  assert.equal(periodicMessages, totalQuanta / 8);
  assert.equal(stats.quantaProcessed, totalQuanta);
  assert.ok(stats.overrunTrimmedFrames > 0);
  assert.ok(stats.overrunTrimCount > 0);
  assert.ok(stats.underrunQuanta > 0);
  assert.ok(stats.underrunFadeOuts > 0);
  assert.ok(stats.recoveryFadeIns > 0);
  assert.ok(stats.peakFillRatio > 0 && stats.peakFillRatio <= 1.0);
  assert.ok(stats.fillRatio >= 0 && stats.fillRatio <= 1.0);
});

// ---------------------------------------------------------------------------
// Jitter/Stutter Fix (#1028/#663): Hysteresis Prebuffer, Cross-Quantum 160f
// Click-Free Micro-Ramps, High-Water Trim Safety, and Zero-Alloc Discipline
// ---------------------------------------------------------------------------

test('Jitter/stutter fix (#1028/#663): hysteresis prebuffer eliminates rapid stop-start oscillation and finishes cross-quantum 160-frame cosine ramps click-free', () => {
  assert.equal(workletMod.DEFAULT_PREBUFFER_FRAMES, 0);
  assert.equal(workletMod.DEFAULT_RECOVERY_PREBUFFER_FRAMES, 3840);
  assert.equal(workletMod.DEFAULT_LIVE_DECLICK_FRAMES, 160);
  assert.equal(workletMod.DEFAULT_LIVE_HIGH_WATERMARK_FRAMES, 10080);
  assert.equal(workletMod.DEFAULT_LIVE_TARGET_WATERMARK_FRAMES, 4608);

  const fadeFrames = 160; // ~3.33 ms @ 48 kHz (> 128-frame quantum -> spans 2 quanta)
  const prebufferFrames = 3840; // 80 ms @ 48 kHz
  const ampL = Math.fround(0.75);
  const ampR = Math.fround(-0.55);
  const maxStepL = (Math.PI / (2 * fadeFrames)) * Math.abs(ampL) + 1e-4;
  const maxStepR = (Math.PI / (2 * fadeFrames)) * Math.abs(ampR) + 1e-4;

  const ring = AudioRingBuffer.create(16384, 2);
  const processor = new PlayGta6AudioWorkletProcessor({
    processorOptions: {
      sab: ring.sab,
      declickFrames: fadeFrames,
      fadeCurve: 'cosine',
      prebufferFrames,
      maxBufferedFrames: 10080,
      targetWatermarkFrames: 4608,
      trimCrossfadeFrames: 64,
    },
  });

  // Control message protocol for prebufferFrames.
  const replies = attachMessageRecorder(processor);
  processor.port.onmessage({ data: { type: 'setPrebufferFrames', prebufferFrames: 3072, id: 'pb-1' } });
  assert.deepEqual(replies[replies.length - 1], {
    type: 'prebufferFrames',
    prebufferFrames: 3072,
    id: 'pb-1',
  });
  assert.equal(processor.prebufferFrames, 3072);
  assert.equal(processor.setPrebufferFrames(prebufferFrames), prebufferFrames);

  // 1. Steady-state playback followed by partial underrun (64 frames left):
  //    starts 160-frame fade-out across quantum boundary (64 frames faded in Q1, 96 frames in Q2).
  assert.equal(ring.pushSamples(makeConstantStereo(192, ampL, ampR)), 192);
  const out = makePlanarOutput(2, 128);
  assert.equal(processor.process([], out), true); // consumes 128 -> 64 left
  assert.equal(out[0][0][127], ampL);

  let prevL = out[0][0][127];
  let prevR = out[0][1][127];
  assert.equal(processor.process([], out), true); // 64 valid + 64 fade-out
  for (let f = 0; f < 128; f++) {
    assert.ok(Math.abs(out[0][0][f] - prevL) <= maxStepL, `Q1 fade-out L jump at f=${f}`);
    assert.ok(Math.abs(out[0][1][f] - prevR) <= maxStepR, `Q1 fade-out R jump at f=${f}`);
    prevL = out[0][0][f];
    prevR = out[0][1][f];
  }
  assert.equal(processor.inUnderrun, true);
  assert.equal(processor.fadeOutCount, 1);

  // 2. Immediately trickle 256 frames (< 3840 prebuffer): processor MUST enter hysteresis hold
  //    AND finish the remaining 96 frames of the 160-frame fade-out smoothly to exact 0.0!
  assert.equal(ring.pushSamples(makeConstantStereo(256, ampL, ampR)), 256);
  assert.equal(processor.process([], out), true);
  assert.equal(processor.inRebuffer, true);
  assert.equal(ring.availableFrames(), 256, 'trickle frames must remain buffered during hysteresis hold');
  for (let f = 0; f < 128; f++) {
    assert.ok(Math.abs(out[0][0][f] - prevL) <= maxStepL, `Q2 fade-out L jump at f=${f}`);
    assert.ok(Math.abs(out[0][1][f] - prevR) <= maxStepR, `Q2 fade-out R jump at f=${f}`);
    prevL = out[0][0][f];
    prevR = out[0][1][f];
  }
  assert.equal(out[0][0][95], 0.0);
  assert.equal(out[0][0][127], 0.0);
  assert.equal(out[0][1][127], 0.0);

  // 3. Push 12 more small 256-frame trickles (total = 13 * 256 = 3328 < 3840):
  //    must hold exact silence with ZERO premature resume / stop-start oscillation!
  for (let t = 0; t < 12; t++) {
    assert.equal(ring.pushSamples(makeConstantStereo(256, ampL, ampR)), 256);
    assert.equal(processor.process([], out), true);
    assert.equal(processor.inRebuffer, true);
    assert.ok(out[0][0].every((v) => v === 0.0));
    assert.ok(out[0][1].every((v) => v === 0.0));
  }
  assert.equal(processor.fadeInCount, 0);
  assert.equal(processor.fadeOutCount, 1);
  assert.equal(ring.availableFrames(), 3328);

  // 4. Push 512 more frames -> total 3840 (reaches 80ms hysteresis target!):
  //    exits hold ONCE and applies click-free 160-frame half-cosine fade-in across 2 quanta!
  assert.equal(ring.pushSamples(makeConstantStereo(512, ampL, ampR)), 512);
  prevL = 0.0;
  prevR = 0.0;
  for (let q = 0; q < 2; q++) {
    assert.equal(processor.process([], out), true);
    for (let f = 0; f < 128; f++) {
      assert.ok(Math.abs(out[0][0][f] - prevL) <= maxStepL, `Recovery fade-in L jump at q=${q} f=${f}`);
      assert.ok(Math.abs(out[0][1][f] - prevR) <= maxStepR, `Recovery fade-in R jump at q=${q} f=${f}`);
      prevL = out[0][0][f];
      prevR = out[0][1][f];
    }
  }
  assert.equal(processor.inRebuffer, false);
  assert.equal(processor.inUnderrun, false);
  assert.equal(processor.rebufferCount, 1);
  assert.equal(processor.rebufferQuanta, 13);
  assert.equal(processor.fadeInCount, 1);
  assert.equal(out[0][0][127], ampL);
  assert.equal(out[0][1][127], ampR);
  // Remaining jitter cushion after 2 quanta = 3840 - 256 = 3584 frames (~74.7 ms)!
  assert.equal(ring.availableFrames(), 3584);

  // 5. High-water drift trim (> 10080 frames = 210ms) trims gently down to 4608 (96ms >= 80ms prebuffer)
  //    with click-free crossfade and zero underruns!
  assert.equal(ring.pushSamples(makeConstantStereo(8000, -ampL, -ampR)), 8000);
  assert.equal(ring.availableFrames(), 11584); // > 10080 ceiling
  assert.equal(processor.process([], out), true);
  assert.equal(processor.overrunTrimCount, 1);
  assert.equal(processor.overrunTrimmedFrames, 11584 - 4608);
  assert.equal(ring.availableFrames(), 4608 - 128);
  assert.equal(processor.inRebuffer, false);
  assert.equal(processor.inUnderrun, false);

  // 6. Misconfigured prebufferFrames > maxBufferedFrames clamps safely and NEVER deadlocks!
  const smallRing = AudioRingBuffer.create(1024, 2);
  const clampedProc = new PlayGta6AudioWorkletProcessor({
    processorOptions: {
      sab: smallRing.sab,
      prebufferFrames: 9000,
      prebufferOnStart: true,
      maxBufferedFrames: 512,
      targetWatermarkFrames: 384,
      declickFrames: 32,
    },
  });
  assert.equal(clampedProc.inRebuffer, true);
  assert.equal(smallRing.pushSamples(makeConstantStereo(384, 0.5, -0.5)), 384);
  assert.equal(clampedProc.process([], out), true);
  assert.equal(clampedProc.inRebuffer, false, 'prebuffer must clamp to <= 75% of maxBufferedFrames (384)');
  assert.equal(clampedProc.rebufferCount, 1);
});

test('Jitter/stutter fix (#1028/#663): process() stays strictly zero-alloc with hysteresis prebuffer, 160-frame cosine ramps, drift trim, and periodic stats active', () => {
  const src = fs.readFileSync(
    path.resolve(__dirname, '../web/workers/audio_worklet_processor.js'),
    'utf8'
  );

  let allocationCount = 0;
  let trackingAllocations = false;
  function trackCtor(RealCtor) {
    return new Proxy(RealCtor, {
      construct(target, args, newTarget) {
        if (trackingAllocations) allocationCount++;
        return Reflect.construct(target, args, newTarget);
      },
      apply(target, thisArg, args) {
        if (trackingAllocations) allocationCount++;
        return Reflect.apply(target, thisArg, args);
      },
    });
  }

  let RegisteredProcessor = null;
  class MockBase {
    constructor() {
      this.port = { onmessage: null, postMessage() {} };
    }
  }
  const sandbox = {
    AudioWorkletProcessor: MockBase,
    registerProcessor(_name, ctor) {
      RegisteredProcessor = ctor;
    },
    Array: trackCtor(Array),
    Float32Array: trackCtor(Float32Array),
    Int32Array: trackCtor(Int32Array),
    Uint8Array: trackCtor(Uint8Array),
    ArrayBuffer: trackCtor(ArrayBuffer),
    SharedArrayBuffer: trackCtor(SharedArrayBuffer),
    Object: trackCtor(Object),
    Atomics: { load: Atomics.load, store: Atomics.store, add: Atomics.add },
    Number,
    Math,
    Boolean,
    String,
    Error,
    TypeError,
    RangeError,
  };
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;

  vm.runInNewContext(src, sandbox, { filename: 'audio_worklet_processor.js' });

  const ring = AudioRingBuffer.create(2048, 2);
  const proc = new RegisteredProcessor({
    processorOptions: {
      sab: ring.sab,
      declickFrames: 160,
      fadeCurve: 'cosine',
      prebufferFrames: 512,
      prebufferOnStart: true,
      maxBufferedFrames: 1280,
      targetWatermarkFrames: 640,
      trimCrossfadeFrames: 64,
      statsIntervalQuanta: 4,
    },
  });

  const out = makePlanarOutput(2, 128);
  const trickle = makeConstantStereo(128, 0.4, -0.4);
  const burst = makeConstantStereo(1536, -0.3, 0.3);

  for (let q = 0; q < 48; q++) {
    if (q < 6) {
      pushRawSabFrames(ring.sab, trickle);
    } else if (q === 20) {
      pushRawSabFrames(ring.sab, burst);
    } else if (q >= 32) {
      pushRawSabFrames(ring.sab, trickle);
    }
    trackingAllocations = true;
    assert.equal(proc.process([], out), true);
    trackingAllocations = false;
  }

  assert.equal(allocationCount, 0, `Expected 0 allocations in process(), got ${allocationCount}`);
  assert.ok(proc.rebufferCount >= 1);
  assert.ok(proc.overrunTrimCount >= 1);
});
