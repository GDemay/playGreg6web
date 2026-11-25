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
  MAX_DECLICK_FRAMES,
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

  // Recovery: push 128 frames -> exact playback, then next underrun triggers fadeOutCount === 2.
  assert.equal(ring.pushSamples(makeConstantStereo(128, 0.5, -0.5)), 128);
  const q3 = makePlanarOutput(2, 128);
  assert.equal(processor.process([], q3), true);
  assert.equal(q3[0][0][0], 0.5);
  assert.equal(q3[0][1][127], -0.5);

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
