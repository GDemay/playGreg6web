import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const mixerMod = require('../web/workers/audio_mixer.js');
const audioWorkerMod = require('../web/workers/audio_worker.js');
const workletMod = require('../web/workers/audio_worklet_processor.js');

const {
  AudioMixer,
  decodePcm16,
  decodeImaAdpcm,
  softClipSample,
  softLimitSample,
  sanitizeAndSoftLimitBuffer,
  applyFadeRamp,
  downmixToStereo,
  resampleLinear,
  DEFAULT_SAMPLE_RATE,
  MIN_SAMPLE_RATE,
  MAX_SAMPLE_RATE,
  MAX_RESAMPLE_RATIO,
  DEFAULT_CHANNELS,
  DEFAULT_MAX_VOICES,
  DEFAULT_QUANTUM_FRAMES,
  DEFAULT_INITIAL_BUFFER_FRAMES,
  DEFAULT_GAIN_RAMP_FRAMES,
  DEFAULT_UNDERRUN_RAMP_FRAMES,
  DEFAULT_SOFT_CLIP_KNEE,
  DEFAULT_SOFT_LIMIT_KNEE,
  ITU_BS775_CENTER_GAIN,
  ITU_BS775_SURROUND_GAIN,
  ITU_BS775_LFE_GAIN,
  SUPPORTED_VOICE_CHANNELS,
  IMA_INDEX_TABLE,
  IMA_STEP_TABLE,
} = mixerMod;
const { AudioRingBuffer } = audioWorkerMod;
const {
  PlayGta6AudioWorkletProcessor,
  applyFadeRamp: workletApplyFadeRamp,
  softLimitSample: workletSoftLimitSample,
  sanitizeAndSoftLimitBuffer: workletSanitizeAndSoftLimitBuffer,
  DEFAULT_UNDERRUN_RAMP_FRAMES: WORKLET_DEFAULT_UNDERRUN_RAMP_FRAMES,
  DEFAULT_SOFT_LIMIT_KNEE: WORKLET_DEFAULT_SOFT_LIMIT_KNEE,
  HDR_WRITE_CURSOR,
  HDR_READ_CURSOR,
  HDR_CAPACITY_FRAMES,
  HDR_CHANNELS,
  RING_HEADER_INT32_COUNT,
  RING_HEADER_BYTES,
} = workletMod;

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

test('W139: softClipSample is bit-exact for |x| <= knee, strictly monotonic & bounded in [-1, 1], and sanitizes non-finite inputs', () => {
  assert.equal(typeof softClipSample, 'function');
  assert.equal(DEFAULT_SOFT_CLIP_KNEE, 0.95);
  assert.equal(globalThis.PlayGta6AudioMixer.softClipSample, softClipSample);

  // Bit-exact passthrough for |x| <= 0.95 (including exact boundary and -0).
  const linearSamples = [-0.95, -0.9, -0.5, -0.123456789, -0.0, 0.0, 0.25, 0.75, 0.9499999, 0.95];
  for (const s of linearSamples) {
    assert.ok(
      Object.is(softClipSample(s), s),
      `Expected bit-exact passthrough for ${s}, got ${softClipSample(s)}`
    );
  }

  // Monotonic compression into (knee, 1.0] for |x| > 0.95 and exact odd symmetry.
  const overloaded = [0.955, 0.97, 1.0, 1.05, 1.15, 1.3, 1.6];
  const headroom = 1.0 - 0.95;
  let prev = 0.95;
  for (const x of overloaded) {
    const pos = softClipSample(x);
    const neg = softClipSample(-x);
    const expected = 0.95 + headroom * Math.tanh((x - 0.95) / headroom);
    assert.equal(pos, expected);
    assert.ok(pos > prev && pos <= 1.0, `Expected monotonic compression in (${prev}, 1.0], got ${pos}`);
    assert.equal(neg, -pos, 'softClipSample must be odd-symmetric');
    prev = pos;
  }

  // Extreme magnitudes stay strictly bounded in [-1.0, 1.0].
  assert.ok(softClipSample(1e6) <= 1.0 && softClipSample(1e6) > 0.9999);
  assert.ok(softClipSample(-1e6) >= -1.0 && softClipSample(-1e6) < -0.9999);

  // Custom knee support.
  assert.equal(softClipSample(0.8, 0.8), 0.8);
  const customCompressed = softClipSample(1.0, 0.8);
  assert.ok(customCompressed > 0.8 && customCompressed < 1.0);

  // Non-finite inputs sanitize to 0.0.
  assert.equal(softClipSample(Number.NaN), 0.0);
  assert.equal(softClipSample(Number.POSITIVE_INFINITY), 0.0);
  assert.equal(softClipSample(Number.NEGATIVE_INFINITY), 0.0);

  // Invalid knee values throw RangeError.
  assert.throws(() => softClipSample(0.5, -0.1), RangeError);
  assert.throws(() => softClipSample(0.5, 1.0), RangeError);
  assert.throws(() => softClipSample(0.5, Number.NaN), RangeError);
});

test('W139: AudioMixer sanitizes NaN and +/-Infinity voice inputs to 0.0 without poisoning concurrent voices', () => {
  const mixer = new AudioMixer({ sampleRate: 48000 });

  // Mono voice with NaN, +Infinity, -Infinity interleaved between valid samples.
  const badMono = new Float32Array([0.5, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, -0.25]);
  mixer.createVoice({ samples: badMono, channels: 1, gain: 1.0, pan: -1.0, loop: false });

  // Concurrent clean voice on the same left channel must NOT be poisoned by badMono.
  const cleanMono = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5]);
  mixer.createVoice({ samples: cleanMono, channels: 1, gain: 1.0, pan: -1.0, loop: false });

  const out = mixer.renderFrames(5);
  for (let i = 0; i < out.length; i++) {
    assert.ok(Number.isFinite(out[i]), `Output sample ${i} must be finite, got ${out[i]}`);
  }

  const expectedL = [0.6, 0.2, 0.3, 0.4, 0.25];
  for (let f = 0; f < expectedL.length; f++) {
    assert.ok(
      Math.abs(out[f * 2] - expectedL[f]) < 1e-6,
      `Frame ${f} L: expected ${expectedL[f]}, got ${out[f * 2]}`
    );
    assert.equal(out[f * 2 + 1], 0.0);
  }
  assert.equal(mixer.getStats().sanitizedNonFiniteSamples, 3);

  // Stereo voice: non-finite left sample sanitizes to 0.0 while finite right sample survives.
  const badStereo = new Float32Array([Number.NaN, 0.4, -0.3, Number.POSITIVE_INFINITY]);
  mixer.createVoice({ samples: badStereo, channels: 2, gain: 1.0, pan: 0.0, loop: false });
  const stereoOut = mixer.renderFrames(2);
  assert.ok(Math.abs(stereoOut[0]) < 1e-6);
  assert.ok(Math.abs(stereoOut[1] - 0.4) < 1e-6);
  assert.ok(Math.abs(stereoOut[2] - -0.3) < 1e-6);
  assert.equal(stereoOut[3], 0.0);
  assert.equal(mixer.getStats().sanitizedNonFiniteSamples, 5);

  // Fractional resampling across a NaN sample produces finite output and increments counter.
  const resampledMixer = new AudioMixer({ sampleRate: 48000 });
  resampledMixer.createVoice({
    samples: new Float32Array([0.2, Number.NaN, 0.4]),
    sampleRate: 24000,
    channels: 1,
    pan: -1.0,
    loop: false,
  });
  const resampledOut = resampledMixer.renderFrames(6);
  for (let i = 0; i < resampledOut.length; i++) {
    assert.ok(Number.isFinite(resampledOut[i]), `Resampled sample ${i} must be finite`);
  }
  assert.ok(Math.abs(resampledOut[0] - 0.2) < 1e-6); // pos=0.0 exact finite sample
  assert.equal(resampledOut[2], 0.0); // pos=0.5 interpolated with NaN -> sanitized
  assert.equal(resampledOut[4], 0.0); // pos=1.0 exact NaN -> sanitized
  assert.equal(resampledOut[6], 0.0); // pos=1.5 interpolated with NaN -> sanitized
  assert.ok(Math.abs(resampledOut[8] - 0.4) < 1e-6); // pos=2.0 exact finite sample
  assert.equal(resampledMixer.getStats().sanitizedNonFiniteSamples, 3);

  // Output-stage finite * huge masterGain overflow to Infinity is also sanitized cleanly.
  const overflowMixer = new AudioMixer({ sampleRate: 48000, masterGain: 1e308 });
  overflowMixer.createVoice({
    samples: new Float32Array([1e10]),
    channels: 1,
    pan: -1.0,
    loop: false,
  });
  const overflowOut = overflowMixer.renderFrames(1);
  assert.equal(overflowOut[0], 0.0);
  assert.equal(overflowOut[1], 0.0);
  assert.equal(overflowMixer.getStats().sanitizedNonFiniteSamples, 1);
});

test('W139: softClip: true preserves bit-exact |x| <= 0.95, smoothly compresses |x| > 0.95, and softClip: false hard-clips at +-1.0', () => {
  // 1. softClip: true — bit-exact for |x| <= 0.95, clippedSamples === 0.
  const softMixer = new AudioMixer({ sampleRate: 48000, softClip: true });
  const withinKnee = new Float32Array([-0.95, -0.5, 0.0, 0.5, 0.9, 0.95]);
  softMixer.createVoice({ samples: withinKnee, channels: 1, gain: 1.0, pan: -1.0, loop: false });
  const linearOut = softMixer.renderFrames(withinKnee.length);
  for (let f = 0; f < withinKnee.length; f++) {
    assert.equal(linearOut[f * 2], withinKnee[f], `Expected bit-exact sample at frame ${f}`);
    assert.equal(linearOut[f * 2 + 1], 0.0);
  }
  assert.equal(softMixer.getStats().clippedSamples, 0);

  // 2. softClip: true — |x| > 0.95 is strictly monotonic, bounded in [-1, 1], and increments clippedSamples.
  const overInputs = [0.96, 1.0, 1.1, 1.3, 2.0];
  softMixer.createVoice({
    samples: new Float32Array(overInputs),
    channels: 1,
    gain: 1.0,
    pan: -1.0,
    loop: false,
  });
  const softOut = softMixer.renderFrames(overInputs.length);
  let prev = 0.95;
  for (let f = 0; f < overInputs.length; f++) {
    const val = softOut[f * 2];
    assert.ok(val > prev && val <= 1.0, `Expected monotonic soft-clip at frame ${f}, got ${val} after ${prev}`);
    prev = val;
  }
  assert.equal(softMixer.getStats().clippedSamples, overInputs.length);

  // 3. softClip: false — exact passthrough up to +-1.0 and hard-clip beyond +-1.0.
  const hardMixer = new AudioMixer({ sampleRate: 48000, softClip: false });
  hardMixer.createVoice({
    samples: new Float32Array([0.98, -0.98, 1.0, 1.5, -2.0]),
    channels: 1,
    gain: 1.0,
    pan: -1.0,
    loop: false,
  });
  const hardOut = hardMixer.renderFrames(5);
  assert.equal(hardOut[0], Math.fround(0.98));
  assert.equal(hardOut[2], Math.fround(-0.98));
  assert.equal(hardOut[4], 1.0);
  assert.equal(hardOut[6], 1.0);
  assert.equal(hardOut[8], -1.0);
  assert.equal(hardMixer.getStats().clippedSamples, 2);
});

test('W139: getStats() tracks peakOutputLevel, activeVoicePeak, clippedSamples, sanitizedNonFiniteSamples and resets cleanly on reset()', () => {
  const mixer = new AudioMixer({ sampleRate: 48000, softClip: true });

  // Initial stats are zeroed.
  const initStats = mixer.getStats();
  assert.equal(initStats.clippedSamples, 0);
  assert.equal(initStats.sanitizedNonFiniteSamples, 0);
  assert.equal(initStats.peakOutputLevel, 0);
  assert.equal(initStats.peakLevel, 0);
  assert.equal(initStats.activeVoicePeak, 0);
  assert.equal(initStats.renderedFrames, 0);
  assert.equal(initStats.softClip, true);
  assert.equal(initStats.limiterKnee, 0.95);

  // Spawn 4 concurrent voices (2 short one-shots + 2 looping).
  mixer.createVoice({ samples: new Float32Array([0.4, 0.4]), channels: 1, pan: -1.0, loop: false });
  mixer.createVoice({ samples: new Float32Array([0.4, 0.4]), channels: 1, pan: -1.0, loop: false });
  mixer.createVoice({ samples: new Float32Array([0.3, Number.NaN]), channels: 1, pan: -1.0, loop: true });
  mixer.createVoice({ samples: new Float32Array([0.1, 0.1]), channels: 1, pan: 1.0, loop: true });

  assert.equal(mixer.getStats().activeVoices, 4);
  assert.equal(mixer.getStats().activeVoicePeak, 4);

  // Render 4 frames: frame 0 left sums to 0.4 + 0.4 + 0.3 = 1.1 (> 0.95, clips);
  // frame 1 & frame 3 have NaN from looping voice -> 2 sanitized samples;
  // the two 2-frame one-shots are reaped leaving 2 active voices while activeVoicePeak stays 4.
  const out = mixer.renderFrames(4);
  const stats = mixer.getStats();
  assert.equal(stats.activeVoices, 2);
  assert.equal(stats.activeVoicePeak, 4);
  assert.equal(stats.renderedFrames, 4);
  assert.equal(stats.clippedSamples, 1); // frame 0 left = 1.1 > 0.95
  assert.equal(stats.sanitizedNonFiniteSamples, 2); // frames 1 and 3 from looping voice
  const expectedPeak = softClipSample(1.1, 0.95);
  assert.ok(Math.abs(stats.peakOutputLevel - expectedPeak) < 1e-6);
  assert.equal(stats.peakLevel, stats.peakOutputLevel);
  assert.ok(Math.abs(out[0] - expectedPeak) < 1e-6);

  // reset() clears voices and zeroes all telemetry counters cleanly.
  mixer.reset();
  const afterReset = mixer.getStats();
  assert.equal(afterReset.activeVoices, 0);
  assert.equal(afterReset.totalVoices, 0);
  assert.equal(afterReset.renderedFrames, 0);
  assert.equal(afterReset.clippedSamples, 0);
  assert.equal(afterReset.sanitizedNonFiniteSamples, 0);
  assert.equal(afterReset.peakOutputLevel, 0);
  assert.equal(afterReset.peakLevel, 0);
  assert.equal(afterReset.activeVoicePeak, 0);

  // Mixer remains usable after reset() and tracks fresh post-reset telemetry.
  mixer.createVoice({ samples: new Float32Array([0.25, -0.5]), channels: 1, pan: -1.0, loop: false });
  mixer.renderFrames(2);
  const postResetStats = mixer.getStats();
  assert.equal(postResetStats.activeVoices, 0);
  assert.equal(postResetStats.activeVoicePeak, 1);
  assert.equal(postResetStats.renderedFrames, 2);
  assert.equal(postResetStats.clippedSamples, 0);
  assert.equal(postResetStats.sanitizedNonFiniteSamples, 0);
  assert.ok(Math.abs(postResetStats.peakOutputLevel - 0.5) < 1e-6);
});

// ---------------------------------------------------------------------------
// W170: ITU-R BS.775 5.1 / 7.1 Downmix, 44.1<->48 kHz Resampling & Telemetry
// ---------------------------------------------------------------------------

test('W170: downmixToStereo applies ITU-R BS.775 5.1 & 7.1 weights, soft-clip guard, and per-channel NaN sanitization', () => {
  assert.equal(typeof downmixToStereo, 'function');
  assert.equal(globalThis.PlayGta6AudioMixer.downmixToStereo, downmixToStereo);
  assert.ok(Math.abs(ITU_BS775_CENTER_GAIN - 0.70710678) < 1e-7);
  assert.ok(Math.abs(ITU_BS775_SURROUND_GAIN - 0.70710678) < 1e-7);
  assert.equal(ITU_BS775_LFE_GAIN, 0.0);
  assert.deepEqual(Array.from(SUPPORTED_VOICE_CHANNELS), [1, 2, 4, 6, 8]);

  // 6 isolated 5.1 frames: FL, FR, FC, LFE, BL, BR
  const isolated51 = new Float32Array([
    0.5, 0.0, 0.0, 0.0, 0.0, 0.0, // frame 0: FL
    0.0, 0.5, 0.0, 0.0, 0.0, 0.0, // frame 1: FR
    0.0, 0.0, 0.8, 0.0, 0.0, 0.0, // frame 2: FC -> 0.8 * 0.70710678 on L & R
    0.0, 0.0, 0.0, 0.9, 0.0, 0.0, // frame 3: LFE -> 0.0 by default
    0.0, 0.0, 0.0, 0.0, 0.6, 0.0, // frame 4: BL -> 0.6 * 0.70710678 on L
    0.0, 0.0, 0.0, 0.0, 0.0, 0.6, // frame 5: BR -> 0.6 * 0.70710678 on R
  ]);
  const stats51 = {};
  const out51 = downmixToStereo(isolated51, 6, { stats: stats51 });
  assert.equal(out51.length, 6 * 2);

  assert.ok(Math.abs(out51[0] - 0.5) < 1e-6 && out51[1] === 0.0);
  assert.ok(out51[2] === 0.0 && Math.abs(out51[3] - 0.5) < 1e-6);
  assert.ok(Math.abs(out51[4] - 0.8 * ITU_BS775_CENTER_GAIN) < 1e-6);
  assert.ok(Math.abs(out51[5] - 0.8 * ITU_BS775_CENTER_GAIN) < 1e-6);
  assert.equal(out51[6], 0.0);
  assert.equal(out51[7], 0.0);
  assert.ok(Math.abs(out51[8] - 0.6 * ITU_BS775_SURROUND_GAIN) < 1e-6 && out51[9] === 0.0);
  assert.ok(out51[10] === 0.0 && Math.abs(out51[11] - 0.6 * ITU_BS775_SURROUND_GAIN) < 1e-6);
  assert.equal(stats51.downmixedFrames, 6);
  assert.equal(stats51.clippedSamples, 0);
  assert.equal(stats51.nanSanitizedSamples, 0);

  // Explicit LFE gain override routes LFE cleanly into L & R.
  const out51Lfe = downmixToStereo(new Float32Array([0, 0, 0, 0.8, 0, 0]), 6, { lfeGain: 0.5 });
  assert.ok(Math.abs(out51Lfe[0] - 0.4) < 1e-6);
  assert.ok(Math.abs(out51Lfe[1] - 0.4) < 1e-6);

  // 7.1 (8ch: FL, FR, FC, LFE, BL, BR, SL, SR) routing + combined surround check.
  const input71 = new Float32Array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, // frame 0: SL only
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, // frame 1: SR only
    0.2, -0.2, 0.2, 0.9, 0.2, -0.2, 0.2, -0.2, // frame 2: combined
  ]);
  const stats71 = {};
  const out71 = downmixToStereo(input71, 8, { stats: stats71 });
  assert.equal(out71.length, 3 * 2);
  assert.ok(Math.abs(out71[0] - 0.6 * ITU_BS775_SURROUND_GAIN) < 1e-6 && out71[1] === 0.0);
  assert.ok(out71[2] === 0.0 && Math.abs(out71[3] - 0.6 * ITU_BS775_SURROUND_GAIN) < 1e-6);
  const expectedL = 0.2 + 0.2 * ITU_BS775_CENTER_GAIN + 0.4 * ITU_BS775_SURROUND_GAIN;
  const expectedR = -0.2 + 0.2 * ITU_BS775_CENTER_GAIN - 0.4 * ITU_BS775_SURROUND_GAIN;
  assert.ok(Math.abs(out71[4] - expectedL) < 1e-6);
  assert.ok(Math.abs(out71[5] - expectedR) < 1e-6);
  assert.equal(stats71.downmixedFrames, 3);

  // Full-scale 5.1 & 7.1 peaks soft-clip smoothly inside (0.95, 1.0] without hard-clipping.
  const hotStats = {};
  const hot51 = downmixToStereo(new Float32Array([1, 1, 1, 1, 1, 1]), 6, { stats: hotStats });
  const hot71 = downmixToStereo(new Float32Array([1, 1, 1, 1, 1, 1, 1, 1]), 8, { stats: hotStats });
  assert.ok(hot51[0] > 0.95 && hot51[0] <= 1.0);
  assert.ok(hot71[0] > 0.95 && hot71[0] <= 1.0);
  assert.equal(hotStats.clippedSamples, 4);
  assert.equal(hotStats.downmixedFrames, 2);

  // Per-channel NaN / +-Infinity isolation: bad FC/LFE/BL must NOT poison FL/FR/BR.
  const nanStats = {};
  const bad51 = new Float32Array([
    0.5, -0.4, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, 0.2,
  ]);
  const cleanOut = downmixToStereo(bad51, 6, { stats: nanStats });
  assert.ok(Math.abs(cleanOut[0] - 0.5) < 1e-6);
  assert.ok(Math.abs(cleanOut[1] - (-0.4 + 0.2 * ITU_BS775_SURROUND_GAIN)) < 1e-6);
  assert.equal(nanStats.nanSanitizedSamples, 3);

  // Unsupported channel counts throw RangeError.
  assert.throws(() => downmixToStereo(new Float32Array(9), 3), RangeError);
  assert.throws(() => downmixToStereo(new Float32Array(10), 5), RangeError);
  assert.throws(() => downmixToStereo(new Float32Array(18), 9), RangeError);
  assert.throws(() => downmixToStereo(new Float32Array(6), 0), RangeError);
});

test('W170: AudioMixer mixes 5.1 & 7.1 Bink2/XAudio2 voices with 44.1->48 kHz resampling, soft-clipping, and full telemetry', () => {
  const mixer = new AudioMixer({ sampleRate: 48000, softClip: true });

  // 1. Isolated 5.1 voice @ 48 kHz verifies exact ITU-R BS.775 rendering in mixer.
  const voice51 = new Float32Array([
    0.4, -0.2, 0.3, 0.9, 0.2, -0.1,
    0.4, -0.2, 0.3, 0.9, 0.2, -0.1,
  ]);
  mixer.createVoice({ samples: voice51, sampleRate: 48000, channels: 6, loop: false });
  const out51 = mixer.renderFrames(2);
  const expL = 0.4 + 0.3 * ITU_BS775_CENTER_GAIN + 0.2 * ITU_BS775_SURROUND_GAIN;
  const expR = -0.2 + 0.3 * ITU_BS775_CENTER_GAIN - 0.1 * ITU_BS775_SURROUND_GAIN;
  assert.ok(Math.abs(out51[0] - expL) < 1e-6);
  assert.ok(Math.abs(out51[1] - expR) < 1e-6);

  let stats = mixer.getStats();
  assert.equal(stats.downmixedFrames, 2);
  assert.equal(stats.resampledFrames, 0);
  assert.equal(stats.clippedSamples, 0);
  assert.equal(stats.nanSanitizedSamples, 0);
  assert.equal(stats.activeVoicePeak, 1);

  // 2. Concurrent 5.1 @ 44.1 kHz (Bink2 intro) + 7.1 @ 48 kHz (menu XAudio2) with hot peaks + NaN surround.
  const binkFrames = 16;
  const bink51 = new Float32Array(binkFrames * 6);
  for (let f = 0; f < binkFrames; f++) {
    bink51[f * 6 + 0] = 0.7; // FL
    bink51[f * 6 + 1] = 0.7; // FR
    bink51[f * 6 + 2] = f === 0 ? Number.NaN : 0.6; // FC (NaN on frame 0 must not poison FL/FR)
    bink51[f * 6 + 3] = 0.5; // LFE
    bink51[f * 6 + 4] = 0.5; // BL
    bink51[f * 6 + 5] = 0.5; // BR
  }
  const menu71 = new Float32Array(16 * 8);
  for (let f = 0; f < 16; f++) {
    menu71[f * 8 + 0] = 0.4;
    menu71[f * 8 + 1] = 0.4;
    menu71[f * 8 + 2] = 0.3;
    menu71[f * 8 + 4] = 0.2;
    menu71[f * 8 + 5] = 0.2;
    menu71[f * 8 + 6] = 0.2;
    menu71[f * 8 + 7] = 0.2;
  }

  mixer.createVoice({ samples: bink51, sampleRate: 44100, channels: 6, loop: true });
  mixer.createVoice({ samples: menu71, sampleRate: 48000, channels: 8, loop: true });

  const mixed = mixer.renderFrames(8);
  for (let i = 0; i < mixed.length; i++) {
    assert.ok(Number.isFinite(mixed[i]), `Mixed sample ${i} must be finite`);
    assert.ok(mixed[i] >= -1.0 && mixed[i] <= 1.0, `Mixed sample ${i} out of [-1, 1]: ${mixed[i]}`);
  }

  // Pump into an AudioRingBuffer that already experienced an underrun so underrunCount syncs cleanly.
  const ring = AudioRingBuffer.create(16, 2);
  ring.pullSamples(new Float32Array(4 * 2)); // triggers 1 underrun on ring
  assert.equal(ring.underrunCount, 1);
  const pumped = mixer.pumpToRing(ring, 8);
  assert.equal(pumped, 8);

  stats = mixer.getStats();
  assert.equal(stats.activeVoicePeak, 2);
  assert.ok(stats.downmixedFrames >= 2 + 16 + 16);
  assert.ok(stats.resampledFrames >= 16);
  assert.ok(stats.clippedSamples > 0, 'Expected soft-clip limiter to engage on summed 5.1+7.1 voices');
  assert.ok(stats.nanSanitizedSamples >= 1, 'Expected NaN in FC channel to be sanitized and counted');
  assert.equal(stats.nanSanitizedSamples, stats.sanitizedNonFiniteSamples);
  assert.equal(stats.underrunCount, 1);

  // reset() clears all W170 counters cleanly.
  mixer.reset();
  const afterReset = mixer.getStats();
  assert.equal(afterReset.downmixedFrames, 0);
  assert.equal(afterReset.resampledFrames, 0);
  assert.equal(afterReset.clippedSamples, 0);
  assert.equal(afterReset.nanSanitizedSamples, 0);
  assert.equal(afterReset.underrunCount, 0);
  assert.equal(afterReset.activeVoicePeak, 0);
});

test('W170: resampleLinear handles 44.1 kHz <-> 48 kHz accurately with streaming phase continuity and safeguards', () => {
  assert.equal(typeof resampleLinear, 'function');
  assert.equal(globalThis.PlayGta6AudioMixer.resampleLinear, resampleLinear);
  assert.equal(MIN_SAMPLE_RATE, 8000);
  assert.equal(MAX_SAMPLE_RATE, 384000);
  assert.equal(MAX_RESAMPLE_RATIO, 16);

  // 1. 44.1 kHz -> 48 kHz: 441 frames (10 ms) of 1 kHz stereo sine -> exact 480 frames.
  const freq = 1000;
  const src441 = new Float32Array(441 * 2);
  for (let f = 0; f < 441; f++) {
    src441[f * 2] = Math.sin((2 * Math.PI * freq * f) / 44100) * 0.7;
    src441[f * 2 + 1] = Math.cos((2 * Math.PI * freq * f) / 44100) * 0.5;
  }
  const rsUpStats = {};
  const up48 = resampleLinear(src441, 2, 44100, 48000, { stats: rsUpStats });
  assert.equal(up48.length, 480 * 2);
  assert.equal(rsUpStats.resampledFrames, 480);

  const ref48 = new Float32Array(480 * 2);
  for (let f = 0; f < 480; f++) {
    ref48[f * 2] = Math.sin((2 * Math.PI * freq * f) / 48000) * 0.7;
    ref48[f * 2 + 1] = Math.cos((2 * Math.PI * freq * f) / 48000) * 0.5;
  }
  const upRms = computeRmsError(up48, ref48, 0, (480 - 2) * 2);
  assert.ok(upRms < 0.005, `44.1 -> 48 kHz RMS error too high: ${upRms}`);

  // 2. 48 kHz -> 44.1 kHz: 480 frames (10 ms) -> exact 441 frames.
  const rsDownStats = {};
  const down441 = resampleLinear(ref48, 2, 48000, 44100, { stats: rsDownStats });
  assert.equal(down441.length, 441 * 2);
  assert.equal(rsDownStats.resampledFrames, 441);
  const downRms = computeRmsError(down441, src441, 0, (441 - 2) * 2);
  assert.ok(downRms < 0.005, `48 -> 44.1 kHz RMS error too high: ${downRms}`);

  // 3. Streaming chunk phase continuity: 20 chunks of 512 frames @ 44.1 kHz -> 48 kHz has zero drift.
  const streamState = { phase: 0.0 };
  let totalOutFrames = 0;
  for (let chunk = 0; chunk < 20; chunk++) {
    const buf = new Float32Array(512 * 2);
    for (let f = 0; f < 512; f++) {
      const globalF = chunk * 512 + f;
      buf[f * 2] = Math.sin((2 * Math.PI * 440 * globalF) / 44100) * 0.5;
      buf[f * 2 + 1] = -buf[f * 2];
    }
    const res = resampleLinear(buf, 2, 44100, 48000, { state: streamState });
    totalOutFrames += res.length / 2;
  }
  const expectedTotal = Math.round((20 * 512 * 48000) / 44100);
  assert.ok(
    Math.abs(totalOutFrames - expectedTotal) <= 1,
    `Expected drift-free total ~${expectedTotal}, got ${totalOutFrames}`
  );

  // 4. NaN / +-Infinity sanitization during resampling.
  const nanStats = {};
  const bad = new Float32Array([0.2, Number.NaN, 0.4, Number.POSITIVE_INFINITY, 0.6, -0.6]);
  const safeRes = resampleLinear(bad, 2, 44100, 48000, { stats: nanStats });
  for (let i = 0; i < safeRes.length; i++) {
    assert.ok(Number.isFinite(safeRes[i]), `Expected finite resampled value at ${i}`);
  }
  assert.ok(nanStats.nanSanitizedSamples >= 2);

  // 5. Safeguards reject out-of-range sample rates, >16x ratio, and excessive output allocations.
  assert.throws(() => resampleLinear(src441, 2, 0, 48000), RangeError);
  assert.throws(() => resampleLinear(src441, 2, 44100, Number.NaN), RangeError);
  assert.throws(() => resampleLinear(src441, 2, 4000, 48000), RangeError);
  assert.throws(() => resampleLinear(src441, 2, 44100, 500000), RangeError);
  assert.throws(() => resampleLinear(src441, 2, 8000, 384000), RangeError); // 48x > 16x guard
  assert.throws(() => resampleLinear(src441, 2, 44100, 48000, { maxOutputFrames: 100 }), RangeError);
  assert.throws(() => resampleLinear(src441, 0, 44100, 48000), RangeError);
  assert.throws(() => resampleLinear(src441, 9, 44100, 48000), RangeError);
});

// ---------------------------------------------------------------------------
// W217: Zero-Alloc Stereo Mixing, Per-Voice Gain Ramping, Soft-Clip & Telemetry
// ---------------------------------------------------------------------------

test('W217: multi-voice stereo mixing accuracy across 1:1 fast paths, panned mono/stereo, and resampled voices', () => {
  assert.equal(DEFAULT_QUANTUM_FRAMES, 128);
  assert.equal(DEFAULT_INITIAL_BUFFER_FRAMES, 1024);
  assert.equal(DEFAULT_GAIN_RAMP_FRAMES, 64);
  assert.equal(globalThis.PlayGta6AudioMixer.DEFAULT_QUANTUM_FRAMES, 128);
  assert.equal(globalThis.PlayGta6AudioMixer.DEFAULT_GAIN_RAMP_FRAMES, 64);

  const mixer = new AudioMixer({ sampleRate: 48000, masterGain: 1.0 });
  const frames = 128;

  // Voice 1: 1:1 center-panned stereo voice (hits zero-crosstalk center fast path).
  const stereoSrc = new Float32Array(frames * 2);
  for (let f = 0; f < frames; f++) {
    stereoSrc[f * 2] = 0.25 * Math.sin((2 * Math.PI * 220 * f) / 48000);
    stereoSrc[f * 2 + 1] = -0.2 * Math.cos((2 * Math.PI * 330 * f) / 48000);
  }
  mixer.createVoice({ samples: stereoSrc, sampleRate: 48000, channels: 2, gain: 0.8, pan: 0.0 });

  // Voice 2: 1:1 mono hard-left voice.
  const monoLeft = new Float32Array(frames);
  for (let f = 0; f < frames; f++) {
    monoLeft[f] = 0.15;
  }
  mixer.createVoice({ samples: monoLeft, sampleRate: 48000, channels: 1, gain: 1.0, pan: -1.0 });

  // Voice 3: 1:1 mono center-panned voice.
  const monoCenter = new Float32Array(frames).fill(0.2);
  mixer.createVoice({ samples: monoCenter, sampleRate: 48000, channels: 1, gain: 0.5, pan: 0.0 });

  // Voice 4: 24 kHz -> 48 kHz resampled mono hard-right voice.
  const monoRight24k = new Float32Array(frames / 2).fill(0.1);
  mixer.createVoice({ samples: monoRight24k, sampleRate: 24000, channels: 1, gain: 1.0, pan: 1.0 });

  const out = new Float32Array(frames * 2);
  const mixedFrames = mixer.mixInto(out, frames);
  assert.equal(mixedFrames, frames);

  for (let f = 0; f < frames; f++) {
    const expL = stereoSrc[f * 2] * 0.8 + 0.15 + 0.2 * 0.5 * Math.SQRT1_2;
    const expR = stereoSrc[f * 2 + 1] * 0.8 + 0.2 * 0.5 * Math.SQRT1_2 + 0.1;
    assert.ok(
      Math.abs(out[f * 2] - expL) < 1e-5,
      `Frame ${f} L mismatch: expected ${expL}, got ${out[f * 2]}`
    );
    assert.ok(
      Math.abs(out[f * 2 + 1] - expR) < 1e-5,
      `Frame ${f} R mismatch: expected ${expR}, got ${out[f * 2 + 1]}`
    );
  }

  // All 4 one-shot voices finished and were reaped cleanly without leaving stale _voiceList entries.
  const stats = mixer.getStats();
  assert.equal(stats.activeVoices, 0);
  assert.equal(stats.totalVoices, 0);
  assert.equal(stats.peakVoices, 4);
  assert.equal(stats.framesMixed, frames);
});

test('W217: per-voice and master gain ramping prevents clicks across quanta, mid-ramp retargeting, and multi-channel voices', () => {
  const mixer = new AudioMixer({ sampleRate: 48000, softClip: true });
  const dc = new Float32Array(128).fill(0.5);

  // 1. Click-free fade-in via createVoice({ fadeInFrames }) over 8 frames.
  const vId = mixer.createVoice({
    samples: dc,
    channels: 1,
    gain: 1.0,
    pan: -1.0,
    loop: true,
    fadeInFrames: 8,
  });
  assert.equal(mixer.getVoiceGain(vId), 0.0);

  const fadeOut = mixer.renderFrames(10);
  for (let f = 0; f < 8; f++) {
    const expectedGain = (f + 1) / 8;
    const expectedSample = 0.5 * expectedGain;
    assert.ok(
      Math.abs(fadeOut[f * 2] - expectedSample) < 1e-6,
      `Fade-in frame ${f}: expected ${expectedSample}, got ${fadeOut[f * 2]}`
    );
  }
  // Steady-state after ramp finishes hits exact target with zero drift.
  assert.ok(Math.abs(fadeOut[8 * 2] - 0.5) < 1e-6);
  assert.ok(Math.abs(fadeOut[9 * 2] - 0.5) < 1e-6);
  assert.equal(mixer.getVoiceGain(vId), 1.0);

  // 2. Multi-quantum gain ramp + mid-ramp retargeting (smooth bounded per-sample step, zero clicks).
  assert.equal(mixer.setVoiceGain(vId, 0.0, 16), true);
  const q1 = mixer.renderFrames(8); // ramps 1.0 -> 0.5 over first 8 frames
  assert.ok(Math.abs(mixer.getVoiceGain(vId) - 0.5) < 1e-6);

  // Retarget mid-ramp from currentGain (0.5) up to 0.9 over 8 frames -> step = +0.05/frame.
  assert.equal(mixer.rampVoiceGain(vId, 0.9, 8), true);
  const q2 = mixer.renderFrames(8);
  assert.ok(Math.abs(mixer.getVoiceGain(vId) - 0.9) < 1e-6);

  // Concatenate q1 + q2 left channel and verify maximum sample-to-sample jump is strictly bounded (no click).
  const combined = new Float32Array(16);
  for (let f = 0; f < 8; f++) combined[f] = q1[f * 2];
  for (let f = 0; f < 8; f++) combined[8 + f] = q2[f * 2];

  let prev = 0.5; // steady state before ramp began
  for (let f = 0; f < 16; f++) {
    const delta = Math.abs(combined[f] - prev);
    assert.ok(
      delta <= 0.5 / 16 + 1e-5,
      `Detected click at frame ${f}: delta=${delta} (prev=${prev}, cur=${combined[f]})`
    );
    prev = combined[f];
  }
  assert.ok(Math.abs(combined[15] - 0.5 * 0.9) < 1e-6);

  // 3. Stereo (2ch) and 5.1 (6ch) voices also ramp smoothly per sample, plus durationMs spec.
  const stereoMixer = new AudioMixer({ sampleRate: 48000, gainRampFrames: 4 });
  const stereoVoice = stereoMixer.createVoice({
    samples: new Float32Array([0.4, -0.4, 0.4, -0.4, 0.4, -0.4, 0.4, -0.4]),
    channels: 2,
    gain: 1.0,
    pan: 0.0,
    loop: true,
  });
  // Uses mixer default gainRampFrames = 4.
  stereoMixer.setVoiceGain(stereoVoice, 0.0);
  const stereoRamp = stereoMixer.renderFrames(4);
  const expectedGains = [0.75, 0.5, 0.25, 0.0];
  for (let f = 0; f < 4; f++) {
    assert.ok(Math.abs(stereoRamp[f * 2] - 0.4 * expectedGains[f]) < 1e-6);
    assert.ok(Math.abs(stereoRamp[f * 2 + 1] - -0.4 * expectedGains[f]) < 1e-6);
  }

  // Master gain ramping across frames.
  stereoMixer.setVoiceGain(stereoVoice, 1.0, 0); // explicit 0 snaps immediately
  stereoMixer.setMasterGain(0.2, 4); // ramps masterGain 1.0 -> 0.2 over 4 frames (0.8, 0.6, 0.4, 0.2)
  const masterRamp = stereoMixer.renderFrames(4);
  const expMaster = [0.8, 0.6, 0.4, 0.2];
  for (let f = 0; f < 4; f++) {
    assert.ok(Math.abs(masterRamp[f * 2] - 0.4 * expMaster[f]) < 1e-6);
    assert.ok(Math.abs(masterRamp[f * 2 + 1] - -0.4 * expMaster[f]) < 1e-6);
  }
  assert.ok(Math.abs(stereoMixer.masterGain - 0.2) < 1e-6);

  // 5.1 surround voice ramping via durationMs option (0.25 ms @ 48 kHz = 12 frames).
  const surroundMixer = new AudioMixer({ sampleRate: 48000 });
  const surrSrc = new Float32Array(12 * 6);
  for (let f = 0; f < 12; f++) {
    surrSrc[f * 6] = 0.5;     // FL
    surrSrc[f * 6 + 1] = 0.5; // FR
  }
  const surrId = surroundMixer.createVoice({
    samples: surrSrc,
    channels: 6,
    gain: 0.0,
    loop: true,
  });
  surroundMixer.setVoiceGain(surrId, 1.0, { durationMs: 0.25 });
  const surrOut = surroundMixer.renderFrames(12);
  for (let f = 0; f < 12; f++) {
    const expected = 0.5 * ((f + 1) / 12);
    assert.ok(Math.abs(surrOut[f * 2] - expected) < 1e-5);
    assert.ok(Math.abs(surrOut[f * 2 + 1] - expected) < 1e-5);
  }

  // Rejects invalid rampFrames.
  assert.throws(() => mixer.setVoiceGain(vId, 0.5, -1), RangeError);
  assert.throws(() => mixer.setVoiceGain(vId, 0.5, Number.NaN), RangeError);
});

test('W217: hot mix passes reuse pre-allocated Float32Array accumulation and output buffers with zero per-quantum allocations', () => {
  const mixer = new AudioMixer({
    sampleRate: 48000,
    quantumFrames: 128,
    reuseOutputBuffer: true,
  });

  // Pre-allocated Float32Array buffers exist immediately after construction.
  assert.ok(mixer._mixScratch instanceof Float32Array);
  assert.ok(mixer._outputScratch instanceof Float32Array);
  assert.ok(mixer._monoScratch instanceof Float32Array);
  assert.equal(mixer._mixScratch.length, 128 * 2);
  assert.equal(mixer.getStats().scratchCapacityFrames, 128);
  assert.equal(mixer.getStats().scratchReallocations, 0);

  const initialMixBuf = mixer._mixScratch.buffer;
  const initialOutBuf = mixer._outputScratch.buffer;
  const initialMonoBuf = mixer._monoScratch.buffer;

  // Populate multiple voices (mono + stereo) to exercise hot paths.
  const wave = new Float32Array(256);
  for (let i = 0; i < wave.length; i++) {
    wave[i] = 0.2 * Math.sin((2 * Math.PI * 440 * i) / 48000);
  }
  mixer.createVoice({ samples: wave, channels: 1, gain: 0.7, pan: -0.5, loop: true });
  mixer.createVoice({ samples: wave, channels: 2, gain: 0.6, pan: 0.0, loop: true });

  // 1. Repeated renderFrames(128) with reuseOutputBuffer: true returns the exact same Float32Array.
  const firstView = mixer.renderFrames(128);
  for (let q = 0; q < 64; q++) {
    const view = mixer.renderFrames(128);
    assert.equal(view, firstView, `Quantum ${q} must reuse cached output Float32Array view`);
    assert.equal(mixer._mixScratch.buffer, initialMixBuf);
    assert.equal(mixer._outputScratch.buffer, initialOutBuf);
  }
  assert.equal(mixer.getStats().scratchReallocations, 0);

  // 2. Caller-provided targetBuffer via renderFrames(128, target) and mixInto(target, 128) is zero-alloc.
  const callerBuf = new Float32Array(128 * 2);
  for (let q = 0; q < 32; q++) {
    const ret = mixer.renderFrames(128, callerBuf);
    assert.equal(ret, callerBuf);
    const count = mixer.mixInto(callerBuf, 128);
    assert.equal(count, 128);
    assert.equal(mixer._mixScratch.buffer, initialMixBuf);
  }
  assert.equal(mixer.getStats().scratchReallocations, 0);

  // 3. pumpToRing into stereo AND mono AudioRingBuffers reuses scratch buffers with zero reallocations.
  const stereoRing = AudioRingBuffer.create(256, 2);
  const drainStereo = new Float32Array(128 * 2);
  for (let q = 0; q < 32; q++) {
    const written = mixer.pumpToRing(stereoRing, 128);
    assert.equal(written, 128);
    assert.equal(stereoRing.pullSamples(drainStereo), 128);
    assert.equal(mixer._mixScratch.buffer, initialMixBuf);
    assert.equal(mixer._outputScratch.buffer, initialOutBuf);
  }

  const monoRing = AudioRingBuffer.create(256, 1);
  const drainMono = new Float32Array(128);
  for (let q = 0; q < 32; q++) {
    const written = mixer.pumpToRing(monoRing, 128);
    assert.equal(written, 128);
    assert.equal(monoRing.pullSamples(drainMono), 128);
    assert.equal(mixer._mixScratch.buffer, initialMixBuf);
    assert.equal(mixer._monoScratch.buffer, initialMonoBuf);
  }
  assert.equal(mixer.getStats().scratchReallocations, 0);

  // 4. Exceeding initial capacity grows geometrically once and stabilizes at zero further reallocations.
  const bigOut = mixer.renderFrames(256);
  assert.equal(bigOut.length, 256 * 2);
  assert.equal(mixer.getStats().scratchReallocations, 1);
  assert.equal(mixer.getStats().scratchCapacityFrames, 256);
  const grownMixBuf = mixer._mixScratch.buffer;
  for (let q = 0; q < 16; q++) {
    mixer.renderFrames(256);
    assert.equal(mixer._mixScratch.buffer, grownMixBuf);
  }
  assert.equal(mixer.getStats().scratchReallocations, 1);

  // 5. reset() retains pre-allocated buffers in place and zeroes reallocation counter.
  mixer.reset();
  assert.equal(mixer._mixScratch.buffer, grownMixBuf);
  assert.equal(mixer.getStats().scratchReallocations, 0);
});

test('W217: soft-clip bounds & telemetry expose framesMixed, activeVoices, peakVoices, clippedSamples, underrunQuanta', () => {
  const mixer = new AudioMixer({ sampleRate: 48000, softClip: true, limiterKnee: 0.9 });

  const init = mixer.getStats();
  assert.equal(init.framesMixed, 0);
  assert.equal(init.activeVoices, 0);
  assert.equal(init.peakVoices, 0);
  assert.equal(init.clippedSamples, 0);
  assert.equal(init.underrunQuanta, 0);

  // Create 6 hot voices summing well above +-1.0 (both positive on L and negative on R).
  for (let i = 0; i < 3; i++) {
    mixer.createVoice({
      samples: new Float32Array(64).fill(0.9),
      channels: 1,
      gain: 1.0,
      pan: -1.0,
      loop: false,
    });
    mixer.createVoice({
      samples: new Float32Array(64).fill(-0.9),
      channels: 1,
      gain: 1.0,
      pan: 1.0,
      loop: true,
    });
  }

  const out = new Float32Array(64 * 2);
  mixer.mixInto(out, 64);

  // Verify strict [-1, 1] bounds, bit-exact agreement with softClipSample, and odd symmetry.
  const expectedClipped = Math.fround(softClipSample(Math.fround(0.9) * 3, 0.9));
  for (let f = 0; f < 64; f++) {
    const l = out[f * 2];
    const r = out[f * 2 + 1];
    assert.ok(Number.isFinite(l) && l > 0.9 && l <= 1.0, `L out of soft-clip bounds: ${l}`);
    assert.ok(Number.isFinite(r) && r < -0.9 && r >= -1.0, `R out of soft-clip bounds: ${r}`);
    assert.ok(Math.abs(l - expectedClipped) < 1e-6);
    assert.ok(Math.abs(l + r) < 1e-6);
  }

  // Trigger 2 ring underruns and 1 explicit recordUnderrun(3).
  const ring = AudioRingBuffer.create(64, 2);
  ring.pullSamples(new Float32Array(16 * 2));
  ring.pullSamples(new Float32Array(16 * 2));
  assert.equal(ring.underrunCount, 2);
  mixer.pumpToRing(ring, 32);
  assert.equal(mixer.recordUnderrun(3), 5);

  const stats = mixer.getStats();
  assert.equal(stats.framesMixed, 64 + 32);
  assert.equal(stats.renderedFrames, stats.framesMixed);
  assert.equal(stats.activeVoices, 3); // 3 one-shots finished, 3 loopers active
  assert.equal(stats.peakVoices, 6);
  assert.equal(stats.activeVoicePeak, 6);
  // First 64 frames: L=+2.7 and R=-2.7 both clipped (128 samples); next 32 frames: only R=-2.7 clipped (32 samples).
  assert.equal(stats.clippedSamples, 128 + 32);
  assert.equal(stats.underrunQuanta, 5);
  assert.equal(stats.underrunCount, 5);
  assert.equal(stats.quantaMixed, 2);

  // reset() zeroes all W217 telemetry fields.
  mixer.reset();
  const after = mixer.getStats();
  assert.equal(after.framesMixed, 0);
  assert.equal(after.activeVoices, 0);
  assert.equal(after.peakVoices, 0);
  assert.equal(after.clippedSamples, 0);
  assert.equal(after.underrunQuanta, 0);
  assert.equal(after.quantaMixed, 0);

  // Invalid recordUnderrun arguments throw RangeError.
  assert.throws(() => mixer.recordUnderrun(-1), RangeError);
  assert.throws(() => mixer.recordUnderrun(1.5), RangeError);
});

// ---------------------------------------------------------------------------
// W272: Click-Free Underrun Cosine Ramps & NaN/Inf + Soft-Clip Limiter
// ---------------------------------------------------------------------------

function pushRawWorkletSabFrames(sab, rawSamples) {
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

test('W272: applyFadeRamp performs zero-alloc in-place half-cosine fade-out and fade-in ramps across channels', () => {
  assert.equal(typeof applyFadeRamp, 'function');
  assert.equal(typeof workletApplyFadeRamp, 'function');
  assert.equal(DEFAULT_UNDERRUN_RAMP_FRAMES, 64);
  assert.equal(WORKLET_DEFAULT_UNDERRUN_RAMP_FRAMES, 64);
  assert.equal(globalThis.PlayGta6AudioMixer.applyFadeRamp, applyFadeRamp);
  assert.equal(globalThis.PlayGta6AudioWorklet.applyFadeRamp, workletApplyFadeRamp);

  // 1. Stereo half-cosine fade-out 1.0 -> 0.0 over 64 frames (in-place, returns same buffer).
  const stereoOut = new Float32Array(64 * 2);
  for (let f = 0; f < 64; f++) {
    stereoOut[f * 2] = 0.8;
    stereoOut[f * 2 + 1] = -0.6;
  }
  const returned = applyFadeRamp(stereoOut, 2, 64, 1.0, 0.0);
  assert.equal(returned, stereoOut);
  assert.ok(Math.abs(stereoOut[0] - 0.8) < 1e-6);
  assert.ok(Math.abs(stereoOut[1] - -0.6) < 1e-6);
  assert.equal(stereoOut[63 * 2], 0.0);
  assert.equal(stereoOut[63 * 2 + 1], 0.0);

  // Verify half-cosine curvature (gentle slope at ends, steepest near midpoint ~0.5).
  const midExpectedGain = 0.5 * (1.0 + Math.cos((Math.PI * 31) / 63));
  assert.ok(Math.abs(stereoOut[31 * 2] - 0.8 * midExpectedGain) < 1e-5);
  assert.ok(Math.abs(stereoOut[31 * 2 + 1] - -0.6 * midExpectedGain) < 1e-5);
  const startSlope = Math.abs(stereoOut[0] - stereoOut[2]);
  const midSlope = Math.abs(stereoOut[31 * 2] - stereoOut[32 * 2]);
  const endSlope = Math.abs(stereoOut[62 * 2] - stereoOut[63 * 2]);
  assert.ok(startSlope < midSlope * 0.1, `Cosine start slope (${startSlope}) should be much gentler than midpoint (${midSlope})`);
  assert.ok(endSlope < midSlope * 0.1, `Cosine end slope (${endSlope}) should be much gentler than midpoint (${midSlope})`);

  // Strictly monotonic fade-out.
  for (let f = 1; f < 64; f++) {
    assert.ok(stereoOut[f * 2] <= stereoOut[(f - 1) * 2] + 1e-7);
    assert.ok(stereoOut[f * 2 + 1] >= stereoOut[(f - 1) * 2 + 1] - 1e-7);
  }

  // 2. Half-cosine fade-in 0.0 -> 1.0 over 64 frames via worklet helper matches mixer helper.
  const stereoIn = new Float32Array(64 * 2);
  for (let f = 0; f < 64; f++) {
    stereoIn[f * 2] = 0.5;
    stereoIn[f * 2 + 1] = -0.5;
  }
  workletApplyFadeRamp(stereoIn, 2, 64, 0.0, 1.0);
  assert.equal(stereoIn[0], 0.0);
  assert.equal(stereoIn[1], 0.0);
  assert.ok(Math.abs(stereoIn[63 * 2] - 0.5) < 1e-6);
  assert.ok(Math.abs(stereoIn[63 * 2 + 1] - -0.5) < 1e-6);
  for (let f = 1; f < 64; f++) {
    assert.ok(stereoIn[f * 2] >= stereoIn[(f - 1) * 2] - 1e-7);
    assert.ok(stereoIn[f * 2 + 1] <= stereoIn[(f - 1) * 2 + 1] + 1e-7);
  }

  // 3. Sanitizes NaN/Infinity inside ramped buffer so gain * Infinity never yields NaN.
  const dirtyMono = new Float32Array([Number.POSITIVE_INFINITY, 0.5, Number.NaN, 0.5]);
  applyFadeRamp(dirtyMono, 1, 4, 0.0, 1.0);
  assert.equal(dirtyMono[0], 0.0);
  assert.equal(dirtyMono[2], 0.0);
  assert.ok(dirtyMono.every(Number.isFinite));
});

test('W272: softLimitSample and sanitizeAndSoftLimitBuffer pass |x| <= 0.9 linearly, soft-clip |x| > 0.9, and sanitize NaN/Inf', () => {
  assert.equal(typeof softLimitSample, 'function');
  assert.equal(typeof sanitizeAndSoftLimitBuffer, 'function');
  assert.equal(typeof workletSoftLimitSample, 'function');
  assert.equal(typeof workletSanitizeAndSoftLimitBuffer, 'function');
  assert.equal(DEFAULT_SOFT_LIMIT_KNEE, 0.9);
  assert.equal(WORKLET_DEFAULT_SOFT_LIMIT_KNEE, 0.9);

  // 1. Bit-exact linear pass-through for |x| <= 0.9 (both mixer and worklet exports).
  const normalSamples = [-0.9, -0.75, -0.5, -0.123456, -0.0, 0.0, 0.25, 0.65, 0.8999999, 0.9];
  for (const x of normalSamples) {
    assert.ok(Object.is(softLimitSample(x), x), `mixer softLimitSample(${x}) must be bit-exact`);
    assert.ok(Object.is(workletSoftLimitSample(x), x), `worklet softLimitSample(${x}) must be bit-exact`);
  }

  // 2. Smooth monotonic soft-clipping for |x| > 0.9 and |x| > 1.0 strictly into [-1.0, 1.0].
  const hotInputs = [0.91, 0.95, 1.0, 1.05, 1.15, 1.3, 1.8];
  let prev = 0.9;
  for (const x of hotInputs) {
    const pos = softLimitSample(x);
    const neg = softLimitSample(-x);
    assert.equal(pos, workletSoftLimitSample(x));
    assert.ok(pos > prev && pos <= 1.0, `softLimitSample(${x})=${pos} must be monotonic in (${prev}, 1.0]`);
    assert.equal(neg, -pos, 'softLimitSample must be odd-symmetric');
    prev = pos;
  }
  assert.ok(softLimitSample(1.0) > 0.97 && softLimitSample(1.0) < 0.99);
  assert.ok(softLimitSample(3.0) <= 1.0 && softLimitSample(3.0) > 0.9999);
  assert.ok(softLimitSample(1e6) <= 1.0 && softLimitSample(-1e6) >= -1.0);

  // 3. Non-finite inputs (NaN / +-Infinity) sanitize to 0.0.
  for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
    assert.equal(softLimitSample(bad), 0.0);
    assert.equal(workletSoftLimitSample(bad), 0.0);
  }

  // 4. sanitizeAndSoftLimitBuffer mutates in place, preserves |x| <= 0.9, soft-clips peaks, sanitizes NaN/Inf.
  const buf = new Float32Array([
    -0.9,
    0.5,
    0.9,
    1.5,
    -2.5,
    Number.NaN,
    Number.POSITIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
    0.95,
  ]);
  const res = sanitizeAndSoftLimitBuffer(buf, buf.length);
  assert.equal(res.softClippedSamples, 3); // 1.5, -2.5, 0.95
  assert.equal(res.nonFiniteSanitized, 3); // NaN, +Inf, -Inf
  assert.equal(buf[0], Math.fround(-0.9));
  assert.equal(buf[1], 0.5);
  assert.equal(buf[2], Math.fround(0.9));
  assert.ok(buf[3] > 0.9 && buf[3] <= 1.0);
  assert.ok(buf[4] < -0.9 && buf[4] >= -1.0);
  assert.equal(buf[5], 0.0);
  assert.equal(buf[6], 0.0);
  assert.equal(buf[7], 0.0);
  assert.ok(buf[8] > 0.9 && buf[8] < 1.0);

  // Reusable outStats accumulator (zero-alloc call pattern).
  const reusableStats = { softClippedSamples: 0, nonFiniteSanitized: 0 };
  const buf2 = new Float32Array([0.4, 2.0, Number.NaN, -1.2]);
  const retStats = workletSanitizeAndSoftLimitBuffer(buf2, 4, DEFAULT_SOFT_LIMIT_KNEE, reusableStats);
  assert.equal(retStats, reusableStats);
  assert.equal(reusableStats.softClippedSamples, 2);
  assert.equal(reusableStats.nonFiniteSanitized, 1);
  assert.equal(buf2[0], Math.fround(0.4));
  assert.equal(buf2[2], 0.0);
  assert.ok(buf2[1] > 0.9 && buf2[1] <= 1.0);
  assert.ok(buf2[3] < -0.9 && buf2[3] >= -1.0);
});

test('W272: PlayGta6AudioWorkletProcessor applies cosine underrun fade-out, recovery fade-in, soft-limiter, and tracks stats', () => {
  const ring = AudioRingBuffer.create(512, 2);
  const processor = new PlayGta6AudioWorkletProcessor({
    processorOptions: {
      sab: ring.sab,
      cosineRamp: true,
      softLimit: true,
    },
  });

  assert.equal(processor.fadeCurve, 'cosine');
  assert.equal(processor.declickFrames, 64);
  assert.equal(processor.fadeInFrames, 64);
  assert.equal(processor.softLimit, true);
  assert.equal(processor.softLimitKnee, 0.9);

  // 1. Steady-state quantum with normal samples (|x| <= 0.9), hot peaks (> 1.0), and NaN/Infinity.
  const q1Pcm = new Float32Array(128 * 2);
  for (let f = 0; f < 128; f++) {
    q1Pcm[f * 2] = 0.8;
    q1Pcm[f * 2 + 1] = -0.6;
  }
  q1Pcm[0] = Number.NaN;
  q1Pcm[1] = Number.POSITIVE_INFINITY;
  q1Pcm[2] = 2.5; // soft-clipped into (0.9, 1.0]
  q1Pcm[3] = -1.8; // soft-clipped into [-1.0, -0.9)
  q1Pcm[4] = Number.NEGATIVE_INFINITY;
  assert.equal(pushRawWorkletSabFrames(ring.sab, q1Pcm), 128);

  const out1 = [[new Float32Array(128), new Float32Array(128)]];
  assert.equal(processor.process([], out1), true);
  assert.equal(out1[0][0][0], 0.0);
  assert.equal(out1[0][1][0], 0.0);
  assert.ok(out1[0][0][1] > 0.9 && out1[0][0][1] <= 1.0);
  assert.ok(out1[0][1][1] < -0.9 && out1[0][1][1] >= -1.0);
  assert.equal(out1[0][0][2], 0.0);
  // Normal audio |x| <= 0.9 passes bit-exact untouched.
  assert.equal(out1[0][0][10], Math.fround(0.8));
  assert.equal(out1[0][1][10], Math.fround(-0.6));
  assert.equal(processor.nonFiniteSanitized, 3);
  assert.equal(processor.softClippedSamples, 2);

  // 2. Full underrun after steady-state -> 64-frame half-cosine fade-out to exact 0.0.
  const out2 = [[new Float32Array(128).fill(9.0), new Float32Array(128).fill(-9.0)]];
  assert.equal(processor.process([], out2), true);
  const [fadeL, fadeR] = out2[0];
  const expectedFirstScale = 0.5 * (1.0 + Math.cos(Math.PI / 64));
  const expectedMidScale = 0.5 * (1.0 + Math.cos((32 * Math.PI) / 64)); // exact 0.5
  assert.ok(Math.abs(fadeL[0] - Math.fround(0.8) * expectedFirstScale) < 1e-5);
  assert.ok(Math.abs(fadeR[0] - Math.fround(-0.6) * expectedFirstScale) < 1e-5);
  assert.ok(Math.abs(fadeL[31] - Math.fround(0.8) * expectedMidScale) < 1e-5);
  assert.equal(fadeL[63], 0.0);
  assert.equal(fadeR[63], 0.0);
  for (let f = 64; f < 128; f++) {
    assert.equal(fadeL[f], 0.0);
    assert.equal(fadeR[f], 0.0);
  }
  assert.equal(processor.underrunFadeOuts, 1);
  assert.equal(processor.recoveryFadeIns, 0);

  // 3. Recovery after underrun -> 64-frame half-cosine fade-in from 0.0 -> full amplitude.
  const q3Pcm = new Float32Array(128 * 2);
  for (let f = 0; f < 128; f++) {
    q3Pcm[f * 2] = 0.7;
    q3Pcm[f * 2 + 1] = -0.5;
  }
  assert.equal(ring.pushSamples(q3Pcm), 128);
  const out3 = [[new Float32Array(128), new Float32Array(128)]];
  assert.equal(processor.process([], out3), true);
  const [recL, recR] = out3[0];
  assert.equal(recL[0], 0.0);
  assert.equal(recR[0], 0.0);
  // Midpoint (frame 32) is exact 0.5 * amplitude for cosine fade-in.
  assert.ok(Math.abs(recL[32] - Math.fround(0.7) * 0.5) < 1e-5);
  assert.ok(Math.abs(recR[32] - Math.fround(-0.5) * 0.5) < 1e-5);
  // Post-ramp frames (64..127) are bit-exact full amplitude.
  assert.equal(recL[64], Math.fround(0.7));
  assert.equal(recR[64], Math.fround(-0.5));
  assert.equal(recL[127], Math.fround(0.7));
  assert.equal(recR[127], Math.fround(-0.5));

  // Verify stats surface all W272 counters.
  const stats = processor.getStats();
  assert.equal(stats.fadeCurve, 'cosine');
  assert.equal(stats.underrunFadeOuts, 1);
  assert.equal(stats.recoveryFadeIns, 1);
  assert.equal(stats.softClippedSamples, 2);
  assert.equal(stats.nonFiniteSanitized, 3);
  assert.equal(stats.softLimit, true);
  assert.equal(stats.softLimitKnee, 0.9);

  // Verify control messages toggle fadeCurve and softLimit cleanly.
  const replies = [];
  processor.port.postMessage = (m) => replies.push(m);
  processor.port.onmessage({ data: { type: 'setFadeCurve', fadeCurve: 'linear', id: 'fc-1' } });
  assert.deepEqual(replies[replies.length - 1], { type: 'fadeCurve', fadeCurve: 'linear', id: 'fc-1' });
  processor.port.onmessage({ data: { type: 'setFadeCurve', fadeCurve: 'cosine', id: 'fc-2' } });
  assert.equal(processor.fadeCurve, 'cosine');

  processor.port.onmessage({ data: { type: 'setSoftLimit', softLimit: true, knee: 0.85, id: 'sl-1' } });
  assert.deepEqual(replies[replies.length - 1], {
    type: 'softLimit',
    softLimit: true,
    softLimitKnee: 0.85,
    id: 'sl-1',
  });
  assert.equal(processor.softLimitKnee, 0.85);
});

test('W272: PlayGta6AudioWorkletProcessor process() stays strictly zero-allocation with cosine ramps and softLimit active', () => {
  const ring = AudioRingBuffer.create(256, 2);
  const processor = new PlayGta6AudioWorkletProcessor({
    processorOptions: {
      sab: ring.sab,
      cosineRamp: true,
      softLimit: true,
      smoothingFrames: 16,
    },
  });

  const fullChunk = new Float32Array(128 * 2);
  for (let f = 0; f < 128; f++) {
    fullChunk[f * 2] = 0.7;
    fullChunk[f * 2 + 1] = -0.7;
  }
  fullChunk[2] = Number.NaN;
  fullChunk[3] = 2.4;

  const partialChunk = new Float32Array(96 * 2);
  for (let f = 0; f < 96; f++) {
    partialChunk[f * 2] = 0.6;
    partialChunk[f * 2 + 1] = -1.9;
  }
  partialChunk[4] = Number.NEGATIVE_INFINITY;

  const out = [[new Float32Array(128), new Float32Array(128)]];
  const origFloat32 = globalThis.Float32Array;
  let float32Allocs = 0;
  let tracking = false;
  globalThis.Float32Array = new Proxy(origFloat32, {
    construct(target, args, newTarget) {
      if (tracking) float32Allocs++;
      return Reflect.construct(target, args, newTarget);
    },
  });

  try {
    for (let q = 0; q < 64; q++) {
      if (q % 6 === 2) {
        pushRawWorkletSabFrames(ring.sab, partialChunk);
      } else if (q % 6 !== 3) {
        pushRawWorkletSabFrames(ring.sab, fullChunk);
      }
      tracking = true;
      const keepAlive = processor.process([], out);
      tracking = false;
      assert.equal(keepAlive, true);
      for (let f = 0; f < 128; f++) {
        assert.ok(Number.isFinite(out[0][0][f]) && Math.abs(out[0][0][f]) <= 1.0);
        assert.ok(Number.isFinite(out[0][1][f]) && Math.abs(out[0][1][f]) <= 1.0);
      }
    }
  } finally {
    tracking = false;
    globalThis.Float32Array = origFloat32;
  }

  assert.equal(float32Allocs, 0, `Expected 0 Float32Array allocations in process(), got ${float32Allocs}`);
  assert.ok(processor.underrunFadeOuts > 0);
  assert.ok(processor.recoveryFadeIns > 0);
  assert.ok(processor.softClippedSamples > 0);
  assert.ok(processor.nonFiniteSanitized > 0);
});

test('W272: AudioMixer tracks underrunFadeOuts, recoveryFadeIns, softClippedSamples, nonFiniteSanitized and performs zero-alloc cosine recovery', () => {
  const mixer = new AudioMixer({
    sampleRate: 48000,
    softClip: true,
    limiterKnee: 0.9,
    underrunRampFrames: 64,
    quantumFrames: 128,
    reuseOutputBuffer: true,
  });

  // 1. Render steady-state stereo voice + hot peak + NaN sample.
  const src = new Float32Array(256 * 2);
  for (let f = 0; f < 256; f++) {
    src[f * 2] = 0.6;
    src[f * 2 + 1] = -0.4;
  }
  src[0] = 1.6; // > 0.9 soft-clipped
  src[1] = -2.0; // < -0.9 soft-clipped
  src[2] = Number.NaN; // sanitized to 0.0
  src[3] = Number.POSITIVE_INFINITY; // sanitized to 0.0

  mixer.createVoice({ samples: src, channels: 2, gain: 1.0, pan: 0.0, loop: true });
  const q1 = mixer.renderFrames(64);
  assert.ok(q1[0] > 0.9 && q1[0] <= 1.0);
  assert.ok(q1[1] < -0.9 && q1[1] >= -1.0);
  assert.equal(q1[2], 0.0);
  assert.equal(q1[3], 0.0);
  assert.equal(q1[10], Math.fround(0.6)); // |0.6| <= 0.9 untouched

  // 2. Underrun fade-out writes smooth half-cosine ramp from last rendered stereo frame to 0.0.
  const fadeBuf = new Float32Array(96 * 2);
  assert.equal(mixer.renderUnderrunFadeOut(fadeBuf, 96), 96);
  assert.ok(Math.abs(fadeBuf[0] - Math.fround(0.6)) < 1e-6);
  assert.ok(Math.abs(fadeBuf[1] - Math.fround(-0.4)) < 1e-6);
  assert.equal(fadeBuf[63 * 2], 0.0);
  assert.equal(fadeBuf[63 * 2 + 1], 0.0);
  for (let f = 64; f < 96; f++) {
    assert.equal(fadeBuf[f * 2], 0.0);
    assert.equal(fadeBuf[f * 2 + 1], 0.0);
  }

  // Next renderFrames automatically applies armed half-cosine recovery fade-in (0.0 -> 1.0 over 64 frames).
  const recOut = new Float32Array(128 * 2);
  mixer.mixInto(recOut, 128);
  assert.equal(recOut[0], 0.0);
  assert.equal(recOut[1], 0.0);
  const midCos = 0.5 * (1.0 - Math.cos((31 * Math.PI) / 63));
  assert.ok(Math.abs(recOut[31 * 2] - Math.fround(0.6) * midCos) < 1e-5);
  assert.ok(Math.abs(recOut[63 * 2] - Math.fround(0.6)) < 1e-6);
  assert.ok(Math.abs(recOut[100 * 2] - Math.fround(0.6)) < 1e-6);

  // 3. pumpToRing detects AudioRingBuffer underrun and applies zero-alloc half-cosine recovery fade-in.
  const ring = AudioRingBuffer.create(256, 2);
  ring.pullSamples(new Float32Array(32 * 2)); // triggers ring underrun
  assert.equal(ring.underruns, 1);

  const allocsBefore = mixer.getStats().scratchReallocations;
  assert.equal(mixer.pumpToRing(ring, 128), 128);
  assert.equal(mixer.getStats().scratchReallocations, allocsBefore);

  const pulled = new Float32Array(128 * 2);
  assert.equal(ring.pullSamples(pulled), 128);
  assert.equal(pulled[0], 0.0);
  assert.equal(pulled[1], 0.0);
  assert.ok(pulled[32 * 2] > 0.25 && pulled[32 * 2] < 0.35);
  assert.ok(Math.abs(pulled[63 * 2] - Math.fround(0.6)) < 1e-6);
  assert.ok(Math.abs(pulled[127 * 2] - Math.fround(0.6)) < 1e-6);

  // 4. Stats counters reflect all W272 telemetry and reset cleanly on reset().
  const stats = mixer.getStats();
  assert.equal(stats.underrunRampFrames, 64);
  assert.equal(stats.underrunFadeOuts, 1);
  assert.equal(stats.recoveryFadeIns, 2); // 1 from renderUnderrunFadeOut + 1 from pumpToRing underrun
  assert.ok(stats.softClippedSamples >= 2);
  assert.ok(stats.nonFiniteSanitized >= 2);

  mixer.reset();
  const after = mixer.getStats();
  assert.equal(after.underrunFadeOuts, 0);
  assert.equal(after.recoveryFadeIns, 0);
  assert.equal(after.softClippedSamples, 0);
  assert.equal(after.nonFiniteSanitized, 0);
});

// ---------------------------------------------------------------------------
// W326: Zero-Alloc Stereo Bus Scratch Buffers, Volume/Mute Gain Ramping & Telemetry
// ---------------------------------------------------------------------------

test('W326: pre-allocated stereo bus scratch Float32Arrays achieve strict zero-allocation mixing across consecutive quanta', () => {
  const mixer = new AudioMixer({
    sampleRate: 48000,
    quantumFrames: 128,
    reuseOutputBuffer: true,
  });

  // Pre-allocated stereo bus scratch buffers share the _mixScratch backing store.
  assert.ok(mixer._busLeftScratch instanceof Float32Array);
  assert.ok(mixer._busRightScratch instanceof Float32Array);
  assert.equal(mixer._busLeftScratch.length, 128);
  assert.equal(mixer._busRightScratch.length, 128);
  assert.equal(mixer._busLeftScratch.buffer, mixer._mixScratch.buffer);
  assert.equal(mixer._busRightScratch.buffer, mixer._mixScratch.buffer);
  assert.equal(mixer.getStats().zeroAllocMixQuanta, 0);

  // Add mono, stereo, 5.1 surround, and 44.1 kHz resampled voices to exercise every mix loop.
  const waveMono = new Float32Array(256);
  const waveStereo = new Float32Array(256 * 2);
  const wave51 = new Float32Array(256 * 6);
  for (let f = 0; f < 256; f++) {
    waveMono[f] = 0.15 * Math.sin((2 * Math.PI * 440 * f) / 48000);
    waveStereo[f * 2] = 0.12 * Math.cos((2 * Math.PI * 220 * f) / 48000);
    waveStereo[f * 2 + 1] = -0.1 * Math.sin((2 * Math.PI * 330 * f) / 48000);
    wave51[f * 6] = 0.08;
    wave51[f * 6 + 1] = -0.06;
    wave51[f * 6 + 2] = 0.05;
  }
  mixer.createVoice({ samples: waveMono, sampleRate: 48000, channels: 1, gain: 0.8, pan: -0.4, loop: true });
  mixer.createVoice({ samples: waveStereo, sampleRate: 48000, channels: 2, gain: 0.7, pan: 0.0, loop: true });
  mixer.createVoice({ samples: wave51, sampleRate: 44100, channels: 6, gain: 0.5, pan: 0.25, loop: true });

  const callerInterleaved = new Float32Array(128 * 2);
  const planarLeft = new Float32Array(128);
  const planarRight = new Float32Array(128);
  const stereoRing = AudioRingBuffer.create(256, 2);
  const drainStereo = new Float32Array(128 * 2);
  const monoRing = AudioRingBuffer.create(256, 1);
  const drainMono = new Float32Array(128);

  const initialMixBuf = mixer._mixScratch.buffer;
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
    for (let q = 0; q < 32; q++) {
      mixer.renderFrames(128);
      mixer.mixInto(callerInterleaved, 128);
      mixer.mixIntoPlanar(planarLeft, planarRight, 128);
      assert.equal(mixer.pumpToRing(stereoRing, 128), 128);
      assert.equal(stereoRing.pullSamples(drainStereo), 128);
      assert.equal(mixer.pumpToRing(monoRing, 128), 128);
      assert.equal(monoRing.pullSamples(drainMono), 128);
    }
    tracking = false;
  } finally {
    tracking = false;
    globalThis.Float32Array = origFloat32;
  }

  assert.equal(float32Allocs, 0, `Expected 0 Float32Array allocations in hot mix loop, got ${float32Allocs}`);
  assert.equal(mixer._mixScratch.buffer, initialMixBuf);
  assert.equal(mixer._busLeftScratch.buffer, initialMixBuf);
  assert.equal(mixer._busRightScratch.buffer, initialMixBuf);

  let stats = mixer.getStats();
  assert.equal(stats.scratchReallocations, 0);
  assert.equal(stats.quantaMixed, 160);
  assert.equal(stats.zeroAllocMixQuanta, 160);

  // Verify planar output matches interleaved output bit-for-bit for identical voice state.
  const refInterleaved = new AudioMixer({ sampleRate: 48000, quantumFrames: 128 });
  const refPlanar = new AudioMixer({ sampleRate: 48000, quantumFrames: 128 });
  refInterleaved.createVoice({ samples: waveStereo, channels: 2, gain: 0.9, pan: 0.0, loop: true });
  refPlanar.createVoice({ samples: waveStereo, channels: 2, gain: 0.9, pan: 0.0, loop: true });
  const outInt = new Float32Array(128 * 2);
  const outL = new Float32Array(128);
  const outR = new Float32Array(128);
  refInterleaved.mixInto(outInt, 128);
  refPlanar.mixIntoPlanar(outL, outR, 128);
  for (let f = 0; f < 128; f++) {
    assert.equal(outL[f], outInt[f * 2]);
    assert.equal(outR[f], outInt[f * 2 + 1]);
  }

  // Growing beyond initial 128-frame capacity reallocates once and does NOT count that quantum as zero-alloc.
  mixer.renderFrames(256);
  stats = mixer.getStats();
  assert.equal(stats.scratchReallocations, 1);
  assert.equal(stats.quantaMixed, 161);
  assert.equal(stats.zeroAllocMixQuanta, 160);
  assert.equal(mixer._busLeftScratch.length, 256);
  assert.equal(mixer._busRightScratch.length, 256);

  // Subsequent 256-frame quanta run strictly zero-alloc again.
  const grownBuf = mixer._mixScratch.buffer;
  globalThis.Float32Array = new Proxy(origFloat32, {
    construct(target, args, newTarget) {
      if (tracking) float32Allocs += 1;
      return Reflect.construct(target, args, newTarget);
    },
  });
  try {
    tracking = true;
    for (let q = 0; q < 8; q++) {
      mixer.renderFrames(256);
    }
    tracking = false;
  } finally {
    tracking = false;
    globalThis.Float32Array = origFloat32;
  }
  assert.equal(float32Allocs, 0);
  assert.equal(mixer._mixScratch.buffer, grownBuf);
  stats = mixer.getStats();
  assert.equal(stats.scratchReallocations, 1);
  assert.equal(stats.quantaMixed, 169);
  assert.equal(stats.zeroAllocMixQuanta, 168);
});

test('W326: smooth linear per-sample gain ramping on volume and mute changes eliminates clicks and zipper noise', () => {
  const mixer = new AudioMixer({ sampleRate: 48000, softClip: true });
  const dcStereo = new Float32Array(256 * 2);
  for (let f = 0; f < 256; f++) {
    dcStereo[f * 2] = 0.8;
    dcStereo[f * 2 + 1] = -0.4;
  }
  const vId = mixer.createVoice({ samples: dcStereo, channels: 2, gain: 1.0, pan: 0.0, loop: true });

  // 1. setVolume(0.25) defaults to 64-frame smooth linear ramp from 1.0 -> 0.25.
  assert.equal(mixer.volume, 1.0);
  assert.equal(mixer.muted, false);
  assert.equal(mixer.setVolume(0.25), 0.25);
  assert.equal(mixer.getStats().gainRampActive, true);

  const volQ1 = mixer.renderFrames(32);
  assert.equal(mixer.getStats().gainRampActive, true);
  assert.ok(Math.abs(mixer.effectiveGain - 0.625) < 1e-6);

  const volQ2 = mixer.renderFrames(32);
  assert.equal(mixer.getStats().gainRampActive, false);
  assert.ok(Math.abs(mixer.effectiveGain - 0.25) < 1e-6);
  assert.equal(mixer.volume, 0.25);

  // Verify bounded sample-to-sample step across all 64 frames (no click / zipper jump).
  const expectedStepL = 0.8 * (0.75 / 64);
  let prevL = 0.8;
  let prevR = -0.4;
  for (let f = 0; f < 64; f++) {
    const l = f < 32 ? volQ1[f * 2] : volQ2[(f - 32) * 2];
    const r = f < 32 ? volQ1[f * 2 + 1] : volQ2[(f - 32) * 2 + 1];
    assert.ok(Math.abs(l - prevL) <= expectedStepL + 1e-5, `Volume ramp click on L frame ${f}`);
    assert.ok(Math.abs(r - prevR) <= expectedStepL * 0.5 + 1e-5, `Volume ramp click on R frame ${f}`);
    prevL = l;
    prevR = r;
  }
  assert.ok(Math.abs(prevL - 0.2) < 1e-6);
  assert.ok(Math.abs(prevR - -0.1) < 1e-6);

  // 2. setMuted(true) ramps smoothly from 0.25 -> 0.0 over 64 frames while preserving logical volume.
  assert.equal(mixer.setMuted(true), true);
  assert.equal(mixer.muted, true);
  assert.equal(mixer.volume, 0.25);
  const muteOut = mixer.renderFrames(64);
  let mutePrevL = 0.2;
  const maxMuteStepL = 0.2 / 64;
  for (let f = 0; f < 64; f++) {
    const l = muteOut[f * 2];
    assert.ok(Math.abs(l - mutePrevL) <= maxMuteStepL + 1e-5, `Mute ramp click at frame ${f}`);
    assert.ok(l <= mutePrevL + 1e-6);
    mutePrevL = l;
  }
  assert.equal(muteOut[63 * 2], 0.0);
  assert.equal(muteOut[63 * 2 + 1], 0.0);
  assert.equal(mixer.effectiveGain, 0.0);

  // Changing volume while muted stores new volume without unmuting or emitting audio.
  assert.equal(mixer.setVolume(0.8), 0.8);
  assert.equal(mixer.muted, true);
  const stillSilent = mixer.renderFrames(8);
  assert.ok(stillSilent.every((s) => s === 0.0));

  // 3. Unmuting ramps smoothly 0.0 -> 0.8, and mid-ramp mute reversal stays click-free.
  assert.equal(mixer.setMute(false, 32), false);
  const unmuteHalf = mixer.renderFrames(16); // reaches gain 0.4
  assert.ok(Math.abs(mixer.effectiveGain - 0.4) < 1e-6);
  // Reverse back to muted over 16 frames directly from instantaneous gain 0.4.
  assert.equal(mixer.setMuted(true, 16), true);
  const reverseHalf = mixer.renderFrames(16);
  let revPrev = 0.0;
  const maxUnmuteStep = (0.8 * 0.8) / 32;
  for (let f = 0; f < 16; f++) {
    const l = unmuteHalf[f * 2];
    assert.ok(Math.abs(l - revPrev) <= maxUnmuteStep + 1e-5);
    revPrev = l;
  }
  for (let f = 0; f < 16; f++) {
    const l = reverseHalf[f * 2];
    assert.ok(Math.abs(l - revPrev) <= maxUnmuteStep + 1e-5);
    revPrev = l;
  }
  assert.equal(reverseHalf[15 * 2], 0.0);

  // 4. Instant snap with explicit rampFrames = 0 and non-finite input safety.
  mixer.setMuted(false, 0);
  assert.ok(Math.abs(mixer.effectiveGain - 0.8) < 1e-6);
  mixer.setVolume(Number.NaN);
  assert.ok(Math.abs(mixer.volume - 0.8) < 1e-6);
  mixer.setVolume(-1.0, 0);
  assert.equal(mixer.volume, 0.0);
  assert.equal(mixer.effectiveGain, 0.0);
  mixer.rampMasterGain(1.0, 4);
  const rampMasterOut = mixer.renderFrames(4);
  assert.ok(Math.abs(rampMasterOut[3 * 2] - 0.8) < 1e-6);

  // 5. Per-voice smooth mute/unmute ramping preserves unmutedGain across mute.
  assert.equal(mixer.setVoiceMuted(vId, true, 8), true);
  const voiceMuteOut = mixer.renderFrames(8);
  assert.ok(voiceMuteOut[0] > 0.6 && voiceMuteOut[0] < 0.8);
  assert.equal(voiceMuteOut[7 * 2], 0.0);
  assert.equal(mixer.setVoiceGain(vId, 0.5, 0), true); // updates unmutedGain while muted
  assert.equal(mixer.renderFrames(2)[0], 0.0);
  assert.equal(mixer.setVoiceMuted(vId, false, 4), true);
  const voiceUnmuteOut = mixer.renderFrames(4);
  assert.ok(Math.abs(voiceUnmuteOut[3 * 2] - 0.4) < 1e-6);

  // 6. Constructor supports { volume, muted } options directly.
  const preMuted = new AudioMixer({ sampleRate: 48000, volume: 0.6, muted: true });
  assert.equal(preMuted.volume, 0.6);
  assert.equal(preMuted.muted, true);
  assert.equal(preMuted.effectiveGain, 0.0);
});

test('W326: sanitized per-bus peak/RMS/clip telemetry in getStats() and clean resetStats() preserving voices & scratch', () => {
  const mixer = new AudioMixer({
    sampleRate: 48000,
    softClip: true,
    limiterKnee: 0.95,
    quantumFrames: 128,
    reuseOutputBuffer: true,
  });

  // 1. Asymmetric stereo DC voices: L = 0.8, R = -0.3 -> distinct peakLeft/peakRight & rmsLeft/rmsRight.
  const vL = mixer.createVoice({
    samples: new Float32Array(128).fill(0.8),
    channels: 1,
    pan: -1.0,
    loop: true,
  });
  const vR = mixer.createVoice({
    samples: new Float32Array(128).fill(-0.3),
    channels: 1,
    pan: 1.0,
    loop: true,
  });

  mixer.renderFrames(128);
  let stats = mixer.getStats();
  assert.ok(Math.abs(stats.peakLeft - 0.8) < 1e-6);
  assert.ok(Math.abs(stats.peakRight - 0.3) < 1e-6);
  assert.ok(Math.abs(stats.rmsLeft - 0.8) < 1e-6);
  assert.ok(Math.abs(stats.rmsRight - 0.3) < 1e-6);
  assert.equal(stats.peakLevel, stats.peakLeft);
  assert.equal(stats.clippedLeft, 0);
  assert.equal(stats.clippedRight, 0);
  assert.equal(stats.clippedSamples, 0);
  assert.equal(stats.zeroAllocMixQuanta, 1);

  // 2. Asymmetric clipping on Left bus + NaN injection on Right bus.
  mixer.createVoice({
    samples: new Float32Array(128).fill(0.5),
    channels: 1,
    pan: -1.0,
    loop: false,
  });
  const badR = new Float32Array(128).fill(0.0);
  badR[0] = Number.NaN;
  badR[1] = Number.POSITIVE_INFINITY;
  mixer.createVoice({ samples: badR, channels: 1, pan: 1.0, loop: false });

  mixer.renderFrames(128);
  stats = mixer.getStats();
  const expectedClippedPeak = softClipSample(Math.fround(0.8) + Math.fround(0.5), 0.95);
  assert.ok(Math.abs(stats.peakLeft - expectedClippedPeak) < 1e-5);
  assert.ok(stats.peakLeft <= 1.0 && stats.peakRight <= 0.3 + 1e-6);
  assert.equal(stats.clippedLeft, 128);
  assert.equal(stats.clippedRight, 0);
  assert.equal(stats.clippedSamples, 128);
  assert.equal(stats.sanitizedNonFiniteSamples, 2);
  assert.equal(stats.zeroAllocMixQuanta, 2);

  // 3. Verify resetStats() clears all telemetry counters cleanly WITHOUT destroying voices, volume, or scratch buffers.
  const mixBufBefore = mixer._mixScratch.buffer;
  mixer.setVolume(0.5, 0);
  mixer.resetStats();

  const cleared = mixer.getStats();
  assert.equal(cleared.peakLeft, 0);
  assert.equal(cleared.peakRight, 0);
  assert.equal(cleared.rmsLeft, 0);
  assert.equal(cleared.rmsRight, 0);
  assert.equal(cleared.peakLevel, 0);
  assert.equal(cleared.peakOutputLevel, 0);
  assert.equal(cleared.clippedLeft, 0);
  assert.equal(cleared.clippedRight, 0);
  assert.equal(cleared.clippedSamples, 0);
  assert.equal(cleared.softClippedSamples, 0);
  assert.equal(cleared.sanitizedNonFiniteSamples, 0);
  assert.equal(cleared.framesMixed, 0);
  assert.equal(cleared.renderedFrames, 0);
  assert.equal(cleared.quantaMixed, 0);
  assert.equal(cleared.zeroAllocMixQuanta, 0);
  assert.equal(cleared.underrunQuanta, 0);
  assert.equal(cleared.underrunFadeOuts, 0);
  assert.equal(cleared.recoveryFadeIns, 0);
  assert.equal(cleared.scratchReallocations, 0);
  assert.equal(cleared.gainRampsStarted, 0);
  // Active voices, volume, and scratch buffers remain intact.
  assert.equal(mixer.hasVoice(vL), true);
  assert.equal(mixer.hasVoice(vR), true);
  assert.equal(cleared.activeVoices, 2);
  assert.equal(cleared.volume, 0.5);
  assert.equal(mixer._mixScratch.buffer, mixBufBefore);

  // 4. Verify exact sine RMS (amplitude / sqrt(2)) on fresh post-resetStats window.
  mixer.destroyVoice(vL);
  mixer.destroyVoice(vR);
  mixer.setVolume(1.0, 0);
  const sineFrames = 480; // 10 cycles of 1 kHz at 48 kHz
  const sineStereo = new Float32Array(sineFrames * 2);
  for (let f = 0; f < sineFrames; f++) {
    const theta = (2 * Math.PI * 1000 * f) / 48000;
    sineStereo[f * 2] = 0.6 * Math.sin(theta);
    sineStereo[f * 2 + 1] = 0.4 * Math.cos(theta);
  }
  mixer.createVoice({ samples: sineStereo, channels: 2, gain: 1.0, pan: 0.0, loop: false });
  mixer.resetStats();
  mixer.renderFrames(sineFrames);
  const sineStats = mixer.getStats();
  assert.ok(Math.abs(sineStats.rmsLeft - 0.6 * Math.SQRT1_2) < 1e-3);
  assert.ok(Math.abs(sineStats.rmsRight - 0.4 * Math.SQRT1_2) < 1e-3);
  assert.ok(Math.abs(sineStats.peakLeft - 0.6) < 1e-3);
  assert.ok(Math.abs(sineStats.peakRight - 0.4) < 1e-3);

  // 5. Telemetry sanitization guards against corrupted internal NaN/Infinity/negative values.
  mixer._peakLeft = Number.NaN;
  mixer._peakRight = Number.POSITIVE_INFINITY;
  mixer._rmsLeftSumSq = -5;
  mixer._clippedSamples = Number.NaN;
  mixer._zeroAllocMixQuanta = -10;
  const sanitizedStats = mixer.getStats();
  assert.equal(sanitizedStats.peakLeft, 0);
  assert.equal(sanitizedStats.peakRight, 0);
  assert.equal(sanitizedStats.rmsLeft, 0);
  assert.equal(sanitizedStats.clippedSamples, 0);
  assert.equal(sanitizedStats.zeroAllocMixQuanta, 0);

  // 6. resetStats() does not cause pumpToRing to falsely re-trigger underrun recovery for historical ring underruns.
  const ring = AudioRingBuffer.create(256, 2);
  ring.pullSamples(new Float32Array(32 * 2)); // 1 historical underrun
  mixer.createVoice({ samples: new Float32Array(128).fill(0.4), channels: 1, pan: -1.0, loop: true });
  mixer.pumpToRing(ring, 64);
  assert.equal(mixer.getStats().underrunQuanta, 1);
  assert.equal(mixer.getStats().recoveryFadeIns, 1);
  ring.pullSamples(new Float32Array(64 * 2));
  mixer.resetStats();
  mixer.pumpToRing(ring, 64);
  const postRingReset = mixer.getStats();
  assert.equal(postRingReset.underrunQuanta, 0);
  assert.equal(postRingReset.recoveryFadeIns, 0);
});

test('W326: W272 regression check — cosine underrun fade-out, planar recovery fade-in, NaN/Inf sanitization, and soft-clipping', () => {
  const mixer = new AudioMixer({
    sampleRate: 48000,
    softClip: true,
    limiterKnee: 0.9,
    underrunRampFrames: 64,
    quantumFrames: 128,
  });

  const src = new Float32Array(128 * 2);
  for (let f = 0; f < 128; f++) {
    src[f * 2] = 0.7;
    src[f * 2 + 1] = -0.5;
  }
  src[0] = 2.2; // soft-clipped on L
  src[1] = Number.NEGATIVE_INFINITY; // sanitized on R
  mixer.createVoice({ samples: src, channels: 2, gain: 1.0, pan: 0.0, loop: true });

  // Steady-state planar mix with NaN/Inf sanitization and soft-clipping.
  const outL = new Float32Array(128);
  const outR = new Float32Array(128);
  assert.equal(mixer.mixIntoPlanar(outL, outR, 128), 128);
  assert.ok(outL[0] > 0.9 && outL[0] <= 1.0);
  assert.equal(outR[0], 0.0);
  assert.equal(outL[10], Math.fround(0.7));
  assert.equal(outR[10], Math.fround(-0.5));

  // W272 cosine fade-out from last stereo sample to exact 0.0.
  const fadeOut = new Float32Array(64 * 2);
  assert.equal(mixer.renderUnderrunFadeOut(fadeOut, 64), 64);
  assert.ok(Math.abs(fadeOut[0] - Math.fround(0.7)) < 1e-6);
  assert.ok(Math.abs(fadeOut[1] - Math.fround(-0.5)) < 1e-6);
  assert.equal(fadeOut[63 * 2], 0.0);
  assert.equal(fadeOut[63 * 2 + 1], 0.0);

  // W272 cosine recovery fade-in on planar mixIntoPlanar (0.0 -> full amplitude over 64 frames).
  mixer.resetStats();
  assert.equal(mixer.mixIntoPlanar(outL, outR, 128), 128);
  assert.equal(outL[0], 0.0);
  assert.equal(outR[0], 0.0);
  const midCos = 0.5 * (1.0 - Math.cos((31 * Math.PI) / 63));
  assert.ok(Math.abs(outL[31] - Math.fround(0.7) * midCos) < 1e-5);
  assert.ok(Math.abs(outR[31] - Math.fround(-0.5) * midCos) < 1e-5);
  assert.equal(outL[63], Math.fround(0.7));
  assert.equal(outR[63], Math.fround(-0.5));

  const recStats = mixer.getStats();
  assert.equal(recStats.zeroAllocMixQuanta, 1);
  assert.ok(recStats.peakLeft <= 1.0 && recStats.peakLeft >= 0.69);
  assert.ok(recStats.peakRight <= 0.51 && recStats.peakRight >= 0.49);
  assert.ok(recStats.rmsLeft > 0 && recStats.rmsLeft < 0.7);
  assert.ok(recStats.rmsRight > 0 && recStats.rmsRight < 0.5);
});



