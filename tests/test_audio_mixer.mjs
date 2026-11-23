import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const mixerMod = require('../web/workers/audio_mixer.js');
const audioWorkerMod = require('../web/workers/audio_worker.js');

const {
  AudioMixer,
  decodePcm16,
  decodeImaAdpcm,
  DEFAULT_SAMPLE_RATE,
  DEFAULT_CHANNELS,
  DEFAULT_MAX_VOICES,
  IMA_INDEX_TABLE,
  IMA_STEP_TABLE,
} = mixerMod;
const { AudioRingBuffer } = audioWorkerMod;

/**
 * Reference IMA-ADPCM encoder helper for testing mono & stereo waveforms
 * in both framed (WAV blockAlign) and raw headerless modes.
 */
function encodeImaNibble(sampleInt16, state) {
  const step = IMA_STEP_TABLE[state.stepIndex];
  let diff = sampleInt16 - state.predictor;
  let nibble = 0;
  if (diff < 0) {
    nibble = 8;
    diff = -diff;
  }

  let vpdiff = step >> 3;
  if (diff >= step) {
    nibble |= 4;
    diff -= step;
    vpdiff += step;
  }
  if (diff >= (step >> 1)) {
    nibble |= 2;
    diff -= step >> 1;
    vpdiff += step >> 1;
  }
  if (diff >= (step >> 2)) {
    nibble |= 1;
    vpdiff += step >> 2;
  }

  let predictor = (nibble & 8) ? state.predictor - vpdiff : state.predictor + vpdiff;
  if (predictor > 32767) predictor = 32767;
  else if (predictor < -32768) predictor = -32768;

  let stepIndex = state.stepIndex + IMA_INDEX_TABLE[nibble & 0x0f];
  if (stepIndex < 0) stepIndex = 0;
  else if (stepIndex > 88) stepIndex = 88;

  state.predictor = predictor;
  state.stepIndex = stepIndex;
  return nibble;
}

function encodeImaAdpcmBlocks(int16Samples, channels, blockAlign) {
  const headerBytes = 4 * channels;
  const payloadBytes = blockAlign - headerBytes;
  const samplesPerBlockPerCh =
    channels === 1 ? 1 + payloadBytes * 2 : 1 + Math.floor(payloadBytes / (4 * channels)) * 8;
  const totalFrames = Math.floor(int16Samples.length / channels);
  const blockCount = Math.floor(totalFrames / samplesPerBlockPerCh);
  const out = new Uint8Array(blockCount * blockAlign);
  const view = new DataView(out.buffer);

  const states = Array.from({ length: channels }, () => ({ predictor: 0, stepIndex: 0 }));

  for (let b = 0; b < blockCount; b++) {
    const blockStart = b * blockAlign;
    const frameBase = b * samplesPerBlockPerCh;

    for (let ch = 0; ch < channels; ch++) {
      const initialSample = int16Samples[frameBase * channels + ch];
      states[ch].predictor = initialSample;
      view.setInt16(blockStart + ch * 4, initialSample, true);
      out[blockStart + ch * 4 + 2] = states[ch].stepIndex;
      out[blockStart + ch * 4 + 3] = 0;
    }

    let pos = blockStart + headerBytes;
    if (channels === 1) {
      for (let f = 1; f < samplesPerBlockPerCh; f += 2) {
        const n0 = encodeImaNibble(int16Samples[frameBase + f], states[0]);
        const n1 = encodeImaNibble(int16Samples[frameBase + f + 1], states[0]);
        out[pos++] = (n0 & 0x0f) | ((n1 & 0x0f) << 4);
      }
    } else {
      for (let f = 1; f < samplesPerBlockPerCh; f += 8) {
        for (let ch = 0; ch < channels; ch++) {
          for (let k = 0; k < 8; k += 2) {
            const s0 = int16Samples[(frameBase + f + k) * channels + ch];
            const s1 = int16Samples[(frameBase + f + k + 1) * channels + ch];
            const n0 = encodeImaNibble(s0, states[ch]);
            const n1 = encodeImaNibble(s1, states[ch]);
            out[pos++] = (n0 & 0x0f) | ((n1 & 0x0f) << 4);
          }
        }
      }
    }
  }

  return { bytes: out, encodedFrames: blockCount * samplesPerBlockPerCh };
}

function encodeImaAdpcmRaw(int16Samples, channels) {
  const totalSamples = Math.floor(int16Samples.length / (channels * 2)) * channels * 2;
  const out = new Uint8Array(totalSamples / 2);
  const expectedFloat = new Float32Array(totalSamples);
  const states = Array.from({ length: channels }, () => ({ predictor: 0, stepIndex: 0 }));

  for (let i = 0; i < totalSamples; i += 2) {
    const ch0 = i % channels;
    const ch1 = (i + 1) % channels;
    const n0 = encodeImaNibble(int16Samples[i], states[ch0]);
    expectedFloat[i] = states[ch0].predictor / 32768.0;
    const n1 = encodeImaNibble(int16Samples[i + 1], states[ch1]);
    expectedFloat[i + 1] = states[ch1].predictor / 32768.0;
    out[i / 2] = (n0 & 0x0f) | ((n1 & 0x0f) << 4);
  }
  return { bytes: out, expectedFloat };
}

function computeRmsError(a, b, start, end) {
  let sumSq = 0;
  const count = end - start;
  for (let i = start; i < end; i++) {
    const d = a[i] - b[i];
    sumSq += d * d;
  }
  return Math.sqrt(sumSq / count);
}

test('UMD exports expose AudioMixer and decoders on module.exports and globalThis', () => {
  assert.equal(typeof AudioMixer, 'function');
  assert.equal(typeof decodePcm16, 'function');
  assert.equal(typeof decodeImaAdpcm, 'function');
  assert.equal(DEFAULT_SAMPLE_RATE, 48000);
  assert.equal(DEFAULT_CHANNELS, 2);
  assert.equal(DEFAULT_MAX_VOICES, 64);
  assert.equal(IMA_STEP_TABLE.length, 89);
  assert.equal(IMA_INDEX_TABLE.length, 16);
  assert.equal(globalThis.PlayGta6AudioMixer.AudioMixer, AudioMixer);
  assert.equal(globalThis.PlayGta6AudioMixer.decodePcm16, decodePcm16);
  assert.equal(globalThis.PlayGta6AudioMixer.decodeImaAdpcm, decodeImaAdpcm);
});

test('decodePcm16 accurately normalizes Int16Array and little-endian Uint8Array bytes to [-1, 1]', () => {
  const inputI16 = new Int16Array([-32768, -16384, 0, 16384, 32767, -8192]);
  const fromI16 = decodePcm16(inputI16, 2);
  assert.ok(fromI16 instanceof Float32Array);
  assert.equal(fromI16.length, 6);
  assert.equal(fromI16[0], -1.0);
  assert.equal(fromI16[1], -0.5);
  assert.equal(fromI16[2], 0.0);
  assert.equal(fromI16[3], 0.5);
  assert.ok(Math.abs(fromI16[4] - 32767 / 32768) < 1e-6);
  assert.equal(fromI16[5], -0.25);

  // Test unaligned Uint8Array slice with explicit little-endian encoding.
  const padded = new Uint8Array(1 + inputI16.byteLength);
  const dv = new DataView(padded.buffer, 1, inputI16.byteLength);
  for (let i = 0; i < inputI16.length; i++) {
    dv.setInt16(i * 2, inputI16[i], true);
  }
  const unalignedSlice = padded.subarray(1);
  const fromBytes = decodePcm16(unalignedSlice, 2);
  assert.deepEqual(Array.from(fromBytes), Array.from(fromI16));

  // Trailing incomplete stereo frame (5 samples) should truncate cleanly to 4 samples (2 frames).
  const oddStereo = decodePcm16(new Int16Array([16384, -16384, 8192, -8192, 32767]), 2);
  assert.equal(oddStereo.length, 4);
  assert.deepEqual(Array.from(oddStereo), [0.5, -0.5, 0.25, -0.25]);
});

test('decodeImaAdpcm faithfully reconstructs mono and stereo waveforms (block-framed and raw)', () => {
  // 1. Mono block-framed WAV IMA-ADPCM (blockAlign = 256 -> 505 samples/block).
  const monoBlockAlign = 256;
  const monoFrames = 1010; // exactly 2 blocks
  const monoOriginal = new Int16Array(monoFrames);
  const monoRefFloat = new Float32Array(monoFrames);
  for (let i = 0; i < monoFrames; i++) {
    const val = Math.sin((2 * Math.PI * 440 * i) / 48000) * 0.75;
    monoOriginal[i] = Math.round(val * 32767);
    monoRefFloat[i] = monoOriginal[i] / 32768.0;
  }

  const monoEncoded = encodeImaAdpcmBlocks(monoOriginal, 1, monoBlockAlign);
  assert.equal(monoEncoded.encodedFrames, monoFrames);
  const monoDecoded = decodeImaAdpcm(monoEncoded.bytes, 1, monoBlockAlign);
  assert.equal(monoDecoded.length, monoFrames);
  const monoRmsErr = computeRmsError(monoDecoded, monoRefFloat, 0, monoFrames);
  assert.ok(monoRmsErr < 0.015, `Mono IMA-ADPCM RMS error too high: ${monoRmsErr}`);

  // 2. Stereo block-framed WAV IMA-ADPCM (blockAlign = 512 -> 505 frames/block).
  const stereoBlockAlign = 512;
  const stereoFrames = 1010; // 2 stereo blocks
  const stereoOriginal = new Int16Array(stereoFrames * 2);
  const stereoRefFloat = new Float32Array(stereoFrames * 2);
  for (let f = 0; f < stereoFrames; f++) {
    const left = Math.sin((2 * Math.PI * 330 * f) / 48000) * 0.7;
    const right = Math.cos((2 * Math.PI * 660 * f) / 48000) * 0.6;
    stereoOriginal[f * 2] = Math.round(left * 32767);
    stereoOriginal[f * 2 + 1] = Math.round(right * 32767);
    stereoRefFloat[f * 2] = stereoOriginal[f * 2] / 32768.0;
    stereoRefFloat[f * 2 + 1] = stereoOriginal[f * 2 + 1] / 32768.0;
  }

  const stereoEncoded = encodeImaAdpcmBlocks(stereoOriginal, 2, stereoBlockAlign);
  assert.equal(stereoEncoded.encodedFrames, stereoFrames);
  const stereoDecoded = decodeImaAdpcm(stereoEncoded.bytes, 2, stereoBlockAlign);
  assert.equal(stereoDecoded.length, stereoFrames * 2);
  const stereoRmsErr = computeRmsError(stereoDecoded, stereoRefFloat, 0, stereoFrames * 2);
  assert.ok(stereoRmsErr < 0.02, `Stereo IMA-ADPCM RMS error too high: ${stereoRmsErr}`);

  // 3. Raw headerless mono and stereo IMA-ADPCM streams (bit-exact predictor match + steady-state RMS).
  const rawMono = encodeImaAdpcmRaw(monoOriginal.subarray(0, 512), 1);
  const rawMonoDecoded = decodeImaAdpcm(rawMono.bytes, 1);
  assert.equal(rawMonoDecoded.length, 512);
  assert.deepEqual(Array.from(rawMonoDecoded), Array.from(rawMono.expectedFloat));
  const rawMonoSteadyRms = computeRmsError(rawMonoDecoded, monoRefFloat, 16, 512);
  assert.ok(rawMonoSteadyRms < 0.015, `Raw mono IMA-ADPCM steady-state RMS error too high: ${rawMonoSteadyRms}`);

  const rawStereo = encodeImaAdpcmRaw(stereoOriginal.subarray(0, 512 * 2), 2);
  const rawStereoDecoded = decodeImaAdpcm(rawStereo.bytes, 2);
  assert.equal(rawStereoDecoded.length, 512 * 2);
  assert.deepEqual(Array.from(rawStereoDecoded), Array.from(rawStereo.expectedFloat));
  const rawStereoSteadyRms = computeRmsError(rawStereoDecoded, stereoRefFloat, 32, 512 * 2);
  assert.ok(rawStereoSteadyRms < 0.02, `Raw stereo IMA-ADPCM steady-state RMS error too high: ${rawStereoSteadyRms}`);
});

test('AudioMixer equal-power stereo panning (hard left, center, hard right)', () => {
  const mixer = new AudioMixer({ sampleRate: 48000 });
  const dc = new Float32Array(16).fill(0.5);

  // Hard left: pan = -1 => L = 0.5, R = 0.0
  const vLeft = mixer.createVoice({ samples: dc, channels: 1, gain: 1.0, pan: -1.0 });
  const outLeft = mixer.renderFrames(4);
  for (let f = 0; f < 4; f++) {
    assert.ok(Math.abs(outLeft[f * 2] - 0.5) < 1e-6);
    assert.equal(outLeft[f * 2 + 1], 0.0);
  }

  // Center: pan = 0 => L = R = 0.5 * sqrt(0.5), L^2 + R^2 === 0.5^2
  mixer.setVoicePan(vLeft, 0.0);
  const outCenter = mixer.renderFrames(4);
  const expectedCenter = 0.5 * Math.SQRT1_2;
  for (let f = 0; f < 4; f++) {
    const l = outCenter[f * 2];
    const r = outCenter[f * 2 + 1];
    assert.ok(Math.abs(l - expectedCenter) < 1e-6);
    assert.ok(Math.abs(r - expectedCenter) < 1e-6);
    assert.ok(Math.abs(l * l + r * r - 0.25) < 1e-6);
  }

  // Hard right: pan = +1 => L = 0.0, R = 0.5
  mixer.setVoicePan(vLeft, 1.0);
  const outRight = mixer.renderFrames(4);
  for (let f = 0; f < 4; f++) {
    assert.equal(outRight[f * 2], 0.0);
    assert.ok(Math.abs(outRight[f * 2 + 1] - 0.5) < 1e-6);
  }
});

test('AudioMixer linear resampling (24 kHz -> 48 kHz) and dynamic pitch shift', () => {
  const mixer = new AudioMixer({ sampleRate: 48000 });
  // 4-frame linear ramp at 24 kHz: [0.0, 0.2, 0.4, 0.6]
  const src = new Float32Array([0.0, 0.2, 0.4, 0.6]);
  const vId = mixer.createVoice({
    samples: src,
    sampleRate: 24000,
    channels: 1,
    gain: 1.0,
    pan: -1.0, // hard left for direct inspection
    loop: false,
    pitch: 1.0,
  });

  // 24 kHz -> 48 kHz has step = 0.5:
  // pos = 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5 -> 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6
  const rendered = mixer.renderFrames(8);
  const expectedL = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6];
  for (let f = 0; f < expectedL.length; f++) {
    assert.ok(
      Math.abs(rendered[f * 2] - expectedL[f]) < 1e-6,
      `Frame ${f}: expected ${expectedL[f]}, got ${rendered[f * 2]}`
    );
    assert.equal(rendered[f * 2 + 1], 0.0);
  }
  // One-shot voice reached the end and was automatically reaped.
  assert.equal(mixer.hasVoice(vId), false);

  // Pitch shift 2.0x on 48 kHz source -> step = 2.0 (skips every other sample).
  const src48 = new Float32Array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]);
  const vPitch = mixer.createVoice({
    samples: src48,
    sampleRate: 48000,
    channels: 1,
    pan: -1.0,
    pitch: 2.0,
  });
  const pitched = mixer.renderFrames(4);
  const expectedPitched = [0.0, 0.2, 0.4, 0.6];
  for (let f = 0; f < expectedPitched.length; f++) {
    assert.ok(Math.abs(pitched[f * 2] - expectedPitched[f]) < 1e-6);
  }
  assert.equal(mixer.hasVoice(vPitch), false);

  // Mid-playback setVoicePitch test: 1.0 -> 0.5 on a looping ramp.
  const vDyn = mixer.createVoice({
    samples: new Float32Array([0.0, 0.2, 0.4, 0.6]),
    sampleRate: 48000,
    channels: 1,
    pan: -1.0,
    loop: true,
    pitch: 1.0,
  });
  const first = mixer.renderFrames(1);
  assert.ok(Math.abs(first[0] - 0.0) < 1e-6);
  mixer.setVoicePitch(vDyn, 0.5);
  const next = mixer.renderFrames(3);
  // pos = 1.0 -> 0.2, pos = 1.5 -> 0.3, pos = 2.0 -> 0.4
  assert.ok(Math.abs(next[0] - 0.2) < 1e-6);
  assert.ok(Math.abs(next[2] - 0.3) < 1e-6);
  assert.ok(Math.abs(next[4] - 0.4) < 1e-6);
});

test('AudioMixer looping vs one-shot reaping, pause/resume/stop/destroy, and maxVoices limit', () => {
  const mixer = new AudioMixer({ sampleRate: 48000, maxVoices: 4 });

  const oneShot = mixer.createVoice({
    samples: new Float32Array([0.4, 0.4, 0.4]),
    channels: 1,
    pan: -1.0,
    loop: false,
  });
  const looper = mixer.createVoice({
    samples: new Float32Array([0.1, 0.2, 0.3]),
    channels: 1,
    pan: 1.0,
    loop: true,
  });

  assert.equal(mixer.getStats().activeVoices, 2);

  // Render 6 frames: oneShot produces 3 frames then silence and is reaped; looper wraps seamlessly.
  const out = mixer.renderFrames(6);
  assert.equal(mixer.hasVoice(oneShot), false);
  assert.equal(mixer.hasVoice(looper), true);
  assert.equal(mixer.getStats().activeVoices, 1);

  const expectedL = [0.4, 0.4, 0.4, 0.0, 0.0, 0.0];
  const expectedR = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3];
  for (let f = 0; f < 6; f++) {
    assert.ok(Math.abs(out[f * 2] - expectedL[f]) < 1e-6);
    assert.ok(Math.abs(out[f * 2 + 1] - expectedR[f]) < 1e-6);
  }

  // Pause & resume preserves cursor position and outputs silence while paused.
  assert.equal(mixer.pauseVoice(looper), true);
  const pausedOut = mixer.renderFrames(2);
  assert.deepEqual(Array.from(pausedOut), [0, 0, 0, 0]);
  assert.equal(mixer.resumeVoice(looper), true);
  mixer.setVoiceGain(looper, 2.0);
  const resumedOut = mixer.renderFrames(2);
  assert.ok(Math.abs(resumedOut[1] - 0.2) < 1e-6);
  assert.ok(Math.abs(resumedOut[3] - 0.4) < 1e-6);

  // Stop resets position to 0 and mutes until resumed.
  assert.equal(mixer.stopVoice(looper), true);
  assert.equal(mixer.getStats().activeVoices, 0);
  assert.deepEqual(Array.from(mixer.renderFrames(2)), [0, 0, 0, 0]);

  assert.equal(mixer.destroyVoice(looper), true);
  assert.equal(mixer.hasVoice(looper), false);

  // Enforce maxVoices = 4.
  for (let i = 0; i < 4; i++) {
    mixer.createVoice({ samples: new Float32Array([0.1]), loop: true });
  }
  assert.throws(
    () => mixer.createVoice({ samples: new Float32Array([0.1]), loop: true }),
    RangeError
  );
});

test('AudioMixer soft-clip Math.tanh limiter smoothly limits >1.0 multi-voice sums and tracks stats', () => {
  const mixer = new AudioMixer({ sampleRate: 48000 });

  // 4 hard-left voices summing to 4 * 0.6 = 2.4 (> 1.0) on left channel.
  // Also test negative sum on right channel (-2.4).
  for (let i = 0; i < 4; i++) {
    mixer.createVoice({
      samples: new Float32Array(16).fill(0.6),
      channels: 1,
      gain: 1.0,
      pan: -1.0,
      loop: true,
    });
    mixer.createVoice({
      samples: new Float32Array(16).fill(-0.6),
      channels: 1,
      gain: 1.0,
      pan: 1.0,
      loop: true,
    });
  }

  const out = mixer.renderFrames(8);
  for (let f = 0; f < 8; f++) {
    const l = out[f * 2];
    const r = out[f * 2 + 1];
    assert.ok(l > 0.9 && l <= 1.0, `Expected soft-clipped positive sample in (0.9, 1.0], got ${l}`);
    assert.ok(r < -0.9 && r >= -1.0, `Expected soft-clipped negative sample in [-1.0, -0.9), got ${r}`);
    assert.ok(Math.abs(l + r) < 1e-6, 'Limiter should be symmetric for positive and negative sums');
  }

  // Verify soft-clipping is monotonic and below hard-clipping for moderate overload (1.2 vs 2.4).
  const moderateMixer = new AudioMixer({ sampleRate: 48000 });
  moderateMixer.createVoice({ samples: new Float32Array(4).fill(0.6), channels: 1, pan: -1.0 });
  moderateMixer.createVoice({ samples: new Float32Array(4).fill(0.6), channels: 1, pan: -1.0 });
  const modOut = moderateMixer.renderFrames(4);
  assert.ok(modOut[0] > 0.9 && modOut[0] < 1.0, `Expected gentle tanh soft-clip < 1.0, got ${modOut[0]}`);
  assert.ok(modOut[0] < out[0], 'Soft-clip limiter must be strictly monotonic');

  const stats = mixer.getStats();
  assert.equal(stats.activeVoices, 8);
  assert.equal(stats.renderedFrames, 8);
  assert.ok(stats.peakLevel > 0.9 && stats.peakLevel <= 1.0);
  assert.equal(stats.clippedSamples, 16); // 8 frames * 2 channels exceeded 1.0 pre-limiter
});

test('AudioMixer.pumpToRing integrates directly with AudioRingBuffer without overflowing', () => {
  const ring = AudioRingBuffer.create(12, 2);
  const mixer = new AudioMixer({ sampleRate: 48000 });

  const wave = new Float32Array(32);
  for (let i = 0; i < wave.length; i++) {
    wave[i] = ((i % 8) - 4) * 0.1;
  }

  mixer.createVoice({
    samples: wave,
    sampleRate: 48000,
    channels: 1,
    gain: 1.0,
    pan: -1.0,
    loop: true,
  });

  // Pump 8 frames into 12-frame ring.
  const written1 = mixer.pumpToRing(ring, 8);
  assert.equal(written1, 8);
  assert.equal(ring.availableFrames(), 8);
  assert.equal(ring.freeFrames(), 4);
  assert.equal(ring.overflows, 0);

  // Requesting 10 frames when only 4 are free must pump exactly 4 and cause 0 overflows.
  const written2 = mixer.pumpToRing(ring, 10);
  assert.equal(written2, 4);
  assert.equal(ring.availableFrames(), 12);
  assert.equal(ring.freeFrames(), 0);
  assert.equal(ring.overflows, 0);

  // Pumping when ring is completely full writes 0 frames and does NOT advance voices.
  const renderedBeforeFullPump = mixer.getStats().renderedFrames;
  const written3 = mixer.pumpToRing(ring, 8);
  assert.equal(written3, 0);
  assert.equal(ring.overflows, 0);
  assert.equal(mixer.getStats().renderedFrames, renderedBeforeFullPump);

  // Pull all 12 frames from the SAB ring and verify exact continuity.
  const pulled = new Float32Array(12 * 2);
  assert.equal(ring.pullSamples(pulled), 12);
  for (let f = 0; f < 12; f++) {
    const expected = ((f % 8) - 4) * 0.1;
    assert.ok(
      Math.abs(pulled[f * 2] - expected) < 1e-6,
      `Frame ${f} L: expected ${expected}, got ${pulled[f * 2]}`
    );
    assert.equal(pulled[f * 2 + 1], 0.0);
  }

  // Pump again after draining: should seamlessly continue from frame 12 (no skipped frames!).
  const written4 = mixer.pumpToRing(ring, 4);
  assert.equal(written4, 4);
  const pulledNext = new Float32Array(4 * 2);
  assert.equal(ring.pullSamples(pulledNext), 4);
  for (let f = 0; f < 4; f++) {
    const expected = (((12 + f) % 8) - 4) * 0.1;
    assert.ok(Math.abs(pulledNext[f * 2] - expected) < 1e-6);
    assert.equal(pulledNext[f * 2 + 1], 0.0);
  }
  assert.equal(ring.underruns, 0);
  assert.equal(ring.overflows, 0);
});

test('AudioMixer stereo voice passthrough, panning, and masterGain scaling', () => {
  const mixer = new AudioMixer({ sampleRate: 48000, masterGain: 0.5 });
  const stereo = new Float32Array([0.4, -0.2, 0.8, -0.6]);
  const vId = mixer.createVoice({
    samples: stereo,
    channels: 2,
    gain: 1.0,
    pan: 0.0,
    loop: true,
  });

  // Center pan stereo passthrough scaled by masterGain = 0.5.
  const outCenter = mixer.renderFrames(2);
  assert.ok(Math.abs(outCenter[0] - 0.2) < 1e-6);
  assert.ok(Math.abs(outCenter[1] - -0.1) < 1e-6);
  assert.ok(Math.abs(outCenter[2] - 0.4) < 1e-6);
  assert.ok(Math.abs(outCenter[3] - -0.3) < 1e-6);

  // Hard right collapses L+R onto right channel; masterGain = 1.0.
  mixer.setMasterGain(1.0);
  mixer.setVoicePan(vId, 1.0);
  const outRight = mixer.renderFrames(1);
  assert.equal(outRight[0], 0.0);
  assert.ok(Math.abs(outRight[1] - 0.2) < 1e-6);
});

