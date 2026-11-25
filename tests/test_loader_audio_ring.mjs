import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);

const {
  AUDIO_RING_STRUCT_BYTES_WASM64,
  AUDIO_RING_STRUCT_BYTES_WASM32,
  AUDIO_RING_CHANNELS,
  readPlaygta6AudioRingDescriptor,
  attachRuntimeAudioRing,
  resetAttachedRuntimeAudioRing,
  buildRuntimeModuleConfig,
} = require('../web/loader.js');

const { AudioMixerWorker } = require('../web/workers/audio_worker.js');

/**
 * Write a Playgta6AudioRing struct into `buffer` using the exact C ABI layout from
 * `runtime/os/audio_downmix.h`:
 *   Wasm64 (40 bytes): u64 base (+0), u32 capacity (+8), u32 channels (+12),
 *                      u32 sample_rate (+16), [4B pad (+20)], u64 read_index_ptr (+24),
 *                      u64 write_index_ptr (+32)
 *   Wasm32 (24 bytes): u32 base (+0), u32 capacity (+4), u32 channels (+8),
 *                      u32 sample_rate (+12), u32 read_index_ptr (+16),
 *                      u32 write_index_ptr (+20)
 */
function writeAudioRingStruct(buffer, structOffset, fields, { pointerSize = 8, paddingWord = 0xdeadbeef } = {}) {
  const view = new DataView(buffer);
  if (pointerSize === 8) {
    view.setBigUint64(structOffset + 0, BigInt(fields.base), true);
    view.setUint32(structOffset + 8, fields.capacity >>> 0, true);
    view.setUint32(structOffset + 12, fields.channels >>> 0, true);
    view.setUint32(structOffset + 16, fields.sampleRate >>> 0, true);
    // Deliberately populate struct padding bytes to verify the decoder skips +20..+23.
    view.setUint32(structOffset + 20, paddingWord >>> 0, true);
    view.setBigUint64(structOffset + 24, BigInt(fields.readIndexPtr), true);
    view.setBigUint64(structOffset + 32, BigInt(fields.writeIndexPtr), true);
  } else {
    view.setUint32(structOffset + 0, Number(fields.base) >>> 0, true);
    view.setUint32(structOffset + 4, fields.capacity >>> 0, true);
    view.setUint32(structOffset + 8, fields.channels >>> 0, true);
    view.setUint32(structOffset + 12, fields.sampleRate >>> 0, true);
    view.setUint32(structOffset + 16, Number(fields.readIndexPtr) >>> 0, true);
    view.setUint32(structOffset + 20, Number(fields.writeIndexPtr) >>> 0, true);
  }
}

function createValidLayout({ shared = true, pointerSize = 8, capacity = 64, sampleRate = 48000 } = {}) {
  const totalBytes = 4096;
  const buffer = shared ? new SharedArrayBuffer(totalBytes) : new ArrayBuffer(totalBytes);
  const structOffset = 64;
  const readIndexPtr = 128;
  const writeIndexPtr = 132;
  const base = 256;
  writeAudioRingStruct(
    buffer,
    structOffset,
    { base, capacity, channels: 2, sampleRate, readIndexPtr, writeIndexPtr },
    { pointerSize },
  );
  return { buffer, structOffset, base, capacity, channels: 2, sampleRate, readIndexPtr, writeIndexPtr };
}

test('W254 constants match Playgta6AudioRing C ABI sizes (Wasm64=40, Wasm32=24, stereo=2)', () => {
  assert.equal(AUDIO_RING_STRUCT_BYTES_WASM64, 40);
  assert.equal(AUDIO_RING_STRUCT_BYTES_WASM32, 24);
  assert.equal(AUDIO_RING_CHANNELS, 2);
});

test('W254 readPlaygta6AudioRingDescriptor decodes Wasm64 40-byte struct with BigInt pointer and +20 padding', () => {
  const layout = createValidLayout({ shared: true, pointerSize: 8, capacity: 128, sampleRate: 48000 });
  const mod = {
    _playgta6_audio_ring: () => BigInt(layout.structOffset),
  };

  const desc = readPlaygta6AudioRingDescriptor(mod, layout.buffer);
  assert.ok(desc, 'expected valid Wasm64 audio ring descriptor');
  assert.equal(desc.sab, layout.buffer);
  assert.equal(desc.buffer, layout.buffer);
  assert.equal(desc.base, layout.base);
  assert.equal(desc.dataByteOffset, layout.base);
  assert.equal(desc.capacity, 128);
  assert.equal(desc.capacitySamples, 128);
  assert.equal(desc.capacityFrames, 64);
  assert.equal(desc.channels, 2);
  assert.equal(desc.sampleRate, 48000);
  assert.equal(desc.readIndexPtr, layout.readIndexPtr);
  assert.equal(desc.readIndexByteOffset, layout.readIndexPtr);
  assert.equal(desc.writeIndexPtr, layout.writeIndexPtr);
  assert.equal(desc.writeIndexByteOffset, layout.writeIndexPtr);
  assert.equal(desc.structOffset, layout.structOffset);
  assert.equal(desc.pointerSize, 8);
  assert.equal(desc.isShared, true);
});

test('W254 readPlaygta6AudioRingDescriptor decodes Wasm32 24-byte struct with Number pointer on ArrayBuffer and SAB', () => {
  for (const shared of [false, true]) {
    const layout = createValidLayout({ shared, pointerSize: 4, capacity: 256, sampleRate: 44100 });
    const mod = {
      _playgta6_audio_ring: () => layout.structOffset,
      HEAPU8: new Uint8Array(layout.buffer),
    };

    const desc = readPlaygta6AudioRingDescriptor(mod);
    assert.ok(desc, `expected valid Wasm32 descriptor (shared=${shared})`);
    assert.equal(desc.buffer, layout.buffer);
    assert.equal(desc.dataByteOffset, layout.base);
    assert.equal(desc.capacitySamples, 256);
    assert.equal(desc.capacityFrames, 128);
    assert.equal(desc.channels, 2);
    assert.equal(desc.sampleRate, 44100);
    assert.equal(desc.readIndexByteOffset, layout.readIndexPtr);
    assert.equal(desc.writeIndexByteOffset, layout.writeIndexPtr);
    assert.equal(desc.pointerSize, 4);
    assert.equal(desc.isShared, shared);
  }
});

test('W254 readPlaygta6AudioRingDescriptor probes all Emscripten/wasm export shapes and explicit pointerSize override', () => {
  const layout = createValidLayout({ shared: true, pointerSize: 8, capacity: 64 });
  const exportVariants = [
    { _playgta6_audio_ring: () => BigInt(layout.structOffset), wasmMemory: { buffer: layout.buffer } },
    { playgta6_audio_ring: () => BigInt(layout.structOffset), memory: { buffer: layout.buffer } },
    { asm: { _playgta6_audio_ring: () => BigInt(layout.structOffset), memory: { buffer: layout.buffer } } },
    { asm: { playgta6_audio_ring: () => BigInt(layout.structOffset), memory: { buffer: layout.buffer } } },
    { wasmExports: { _playgta6_audio_ring: () => BigInt(layout.structOffset), memory: { buffer: layout.buffer } } },
    { wasmExports: { playgta6_audio_ring: () => BigInt(layout.structOffset), memory: { buffer: layout.buffer } } },
  ];

  for (const variant of exportVariants) {
    const desc = readPlaygta6AudioRingDescriptor(variant);
    assert.ok(desc);
    assert.equal(desc.dataByteOffset, layout.base);
    assert.equal(desc.pointerSize, 8);
  }

  // Number pointer with explicit pointerSize: 8 override decodes 40-byte Wasm64 layout.
  const overrideMod = { _playgta6_audio_ring: () => layout.structOffset };
  const overrideDesc = readPlaygta6AudioRingDescriptor(overrideMod, layout.buffer, { pointerSize: 8 });
  assert.ok(overrideDesc);
  assert.equal(overrideDesc.pointerSize, 8);
  assert.equal(overrideDesc.readIndexByteOffset, layout.readIndexPtr);
  assert.equal(overrideDesc.writeIndexByteOffset, layout.writeIndexPtr);
});

test('W254 readPlaygta6AudioRingDescriptor gracefully rejects null, missing, throwing, misaligned, non-pow2, and out-of-bounds descriptors', () => {
  const buf = new SharedArrayBuffer(512);

  // Missing / null module or export
  assert.equal(readPlaygta6AudioRingDescriptor(null, buf), null);
  assert.equal(readPlaygta6AudioRingDescriptor({}, buf), null);
  assert.equal(readPlaygta6AudioRingDescriptor({ _playgta6_audio_ring: () => 0 }, buf), null);
  assert.equal(readPlaygta6AudioRingDescriptor({ _playgta6_audio_ring: () => 0n }, buf), null);
  assert.equal(
    readPlaygta6AudioRingDescriptor(
      {
        _playgta6_audio_ring: () => {
          throw new Error('wasm trap');
        },
      },
      buf,
    ),
    null,
  );

  const checkInvalid = (structOffset, fields, pointerSize = 8) => {
    const sab = new SharedArrayBuffer(512);
    const structSize = pointerSize === 8 ? 40 : 24;
    if (structOffset >= 0 && structOffset + structSize <= sab.byteLength) {
      writeAudioRingStruct(sab, structOffset, fields, { pointerSize });
    }
    const ptr = pointerSize === 8 ? BigInt(structOffset) : structOffset;
    const desc = readPlaygta6AudioRingDescriptor({ _playgta6_audio_ring: () => ptr }, sab);
    assert.equal(desc, null);
  };

  const valid = { base: 128, capacity: 32, channels: 2, sampleRate: 48000, readIndexPtr: 96, writeIndexPtr: 100 };

  // Misaligned struct pointer (4-byte offset on Wasm64 requires 8-byte alignment; 2-byte offset on Wasm32)
  checkInvalid(68, valid, 8);
  checkInvalid(66, valid, 4);

  // Struct extends past buffer end
  checkInvalid(480, valid, 8); // 480 + 40 = 520 > 512
  checkInvalid(496, valid, 4); // 496 + 24 = 520 > 512

  // Non-power-of-two or too-small capacity
  checkInvalid(64, { ...valid, capacity: 0 }, 8);
  checkInvalid(64, { ...valid, capacity: 1 }, 8);
  checkInvalid(64, { ...valid, capacity: 3 }, 8);
  checkInvalid(64, { ...valid, capacity: 48 }, 8);

  // Non-stereo channels
  checkInvalid(64, { ...valid, channels: 0 }, 8);
  checkInvalid(64, { ...valid, channels: 1 }, 8);
  checkInvalid(64, { ...valid, channels: 6 }, 8);

  // Invalid sample rate
  checkInvalid(64, { ...valid, sampleRate: 0 }, 8);
  checkInvalid(64, { ...valid, sampleRate: 4000 }, 8);
  checkInvalid(64, { ...valid, sampleRate: 500000 }, 8);

  // Data buffer out of bounds, null, or misaligned
  checkInvalid(64, { ...valid, base: 0 }, 8);
  checkInvalid(64, { ...valid, base: 130 }, 8);
  checkInvalid(64, { ...valid, base: 448, capacity: 32 }, 8); // 448 + 128 = 576 > 512

  // Control index pointers null, misaligned, out of bounds, aliased, or overlapping sample data
  checkInvalid(64, { ...valid, readIndexPtr: 0 }, 8);
  checkInvalid(64, { ...valid, writeIndexPtr: 0 }, 8);
  checkInvalid(64, { ...valid, readIndexPtr: 98 }, 8);
  checkInvalid(64, { ...valid, writeIndexPtr: 512 }, 8);
  checkInvalid(64, { ...valid, readIndexPtr: 96, writeIndexPtr: 96 }, 8); // aliased
  checkInvalid(64, { ...valid, readIndexPtr: 140 }, 8); // inside [128, 256)
  checkInvalid(64, { ...valid, writeIndexPtr: 252 }, 8); // inside [128, 256)
});

test('W254 attachRuntimeAudioRing deduplicates same (buffer, baseOffset), handles missing/failing worker safely, and supports reset', () => {
  const layout = createValidLayout({ shared: true, pointerSize: 8, capacity: 64 });
  const mod = {
    _playgta6_audio_ring: () => BigInt(layout.structOffset),
    wasmMemory: { buffer: layout.buffer },
  };

  // Safe degradation with null audioWorker or missing export
  assert.equal(attachRuntimeAudioRing(mod, null).attached, false);
  assert.equal(attachRuntimeAudioRing({}, { postMessage() {} }).attached, false);

  // Safe degradation when postMessage throws
  const throwingWorker = {
    postMessage() {
      throw new Error('worker terminated');
    },
  };
  const failRes = attachRuntimeAudioRing(mod, throwingWorker);
  assert.equal(failRes.attached, false);
  assert.equal(failRes.reason, 'post-message-error');

  // Normal attach + deduplication
  const posted = [];
  const worker = { postMessage: (msg) => posted.push(msg) };

  const first = attachRuntimeAudioRing(mod, worker);
  assert.equal(first.attached, true);
  assert.equal(first.alreadyAttached, false);
  assert.equal(posted.length, 1);
  assert.equal(posted[0].type, 'attach-guest-pcm-ring');
  assert.equal(posted[0].sab, layout.buffer);
  assert.equal(posted[0].dataByteOffset, layout.base);
  assert.equal(posted[0].capacitySamples, 64);
  assert.equal(posted[0].channels, 2);
  assert.equal(posted[0].sampleRate, 48000);
  assert.equal(posted[0].readIndexByteOffset, layout.readIndexPtr);
  assert.equal(posted[0].writeIndexByteOffset, layout.writeIndexPtr);

  // Duplicate call for same (buffer, baseOffset) must NOT post again
  const second = attachRuntimeAudioRing(mod, worker);
  assert.equal(second.attached, true);
  assert.equal(second.alreadyAttached, true);
  assert.equal(posted.length, 1);

  // Reset allows re-attaching (e.g. after audio worker restart/reconnect)
  resetAttachedRuntimeAudioRing(worker);
  const third = attachRuntimeAudioRing(mod, worker);
  assert.equal(third.attached, true);
  assert.equal(third.alreadyAttached, false);
  assert.equal(posted.length, 2);

  // New baseOffset on same buffer triggers a fresh attach
  writeAudioRingStruct(
    layout.buffer,
    layout.structOffset,
    {
      base: 512,
      capacity: 64,
      channels: 2,
      sampleRate: 48000,
      readIndexPtr: layout.readIndexPtr,
      writeIndexPtr: layout.writeIndexPtr,
    },
    { pointerSize: 8 },
  );
  const fourth = attachRuntimeAudioRing(mod, worker);
  assert.equal(fourth.attached, true);
  assert.equal(fourth.alreadyAttached, false);
  assert.equal(fourth.descriptor.dataByteOffset, 512);
  assert.equal(posted.length, 3);
});

test('W254 buildRuntimeModuleConfig attaches whether audioWorker is ready before or after Wasm init and handles reconnect + exit', () => {
  const layout = createValidLayout({ shared: true, pointerSize: 8, capacity: 64 });
  const mod = {
    _playgta6_audio_ring: () => BigInt(layout.structOffset),
    wasmMemory: { buffer: layout.buffer },
  };

  // Case 1: audioWorker already ready before onRuntimeInitialized
  const postedA = [];
  const workerA = { postMessage: (msg) => postedA.push(msg) };
  const cfgA = buildRuntimeModuleConfig({
    audioWorker: workerA,
    instantiateWasm: false,
  });
  cfgA.onRuntimeInitialized.call(mod);
  assert.equal(postedA.length, 1);
  assert.equal(postedA[0].type, 'attach-guest-pcm-ring');
  assert.ok(cfgA.getAudioRingDescriptor());

  // Subsequent call after createRuntimeModule resolves deduplicates cleanly
  const dup = cfgA.attachAudioRing(mod);
  assert.equal(dup.attached, true);
  assert.equal(dup.alreadyAttached, true);
  assert.equal(postedA.length, 1);

  // Case 2: audioWorker becomes ready AFTER onRuntimeInitialized, then reconnects
  const postedB = [];
  let audioReady = false;
  let currentWorker = { postMessage: (msg) => postedB.push(msg) };
  const cfgB = buildRuntimeModuleConfig({
    getAudioWorker: () => currentWorker,
    isAudioWorkerReady: () => audioReady,
    instantiateWasm: false,
  });

  cfgB.onRuntimeInitialized.call(mod);
  assert.equal(postedB.length, 0, 'must defer attach until audioWorker reports ready');

  audioReady = true;
  const deferredRes = cfgB.notifyAudioWorkerReady(currentWorker);
  assert.equal(deferredRes.attached, true);
  assert.equal(deferredRes.alreadyAttached, false);
  assert.equal(postedB.length, 1);

  // Reconnect with a fresh worker instance
  const postedReconnect = [];
  currentWorker = { postMessage: (msg) => postedReconnect.push(msg) };
  const reconnectRes = cfgB.notifyAudioWorkerReady(currentWorker);
  assert.equal(reconnectRes.attached, true);
  assert.equal(reconnectRes.alreadyAttached, false);
  assert.equal(postedReconnect.length, 1);

  // onExit resets attachment state
  cfgB.onExit.call(mod);
  assert.equal(cfgB.getAudioRingDescriptor(), null);
});

test('W254 end-to-end integration with audio_worker.js AudioMixerWorker + GuestPcmRingReader drains exact stereo samples and advances read_index atomically', () => {
  // Small 16-float (8-frame) ring in SAB to exercise wrap-around too.
  const layout = createValidLayout({ shared: true, pointerSize: 8, capacity: 16, sampleRate: 48000 });
  const ringData = new Float32Array(layout.buffer, layout.base, layout.capacity);
  const readIdx = new Int32Array(layout.buffer, layout.readIndexPtr, 1);
  const writeIdx = new Int32Array(layout.buffer, layout.writeIndexPtr, 1);
  Atomics.store(readIdx, 0, 0);
  Atomics.store(writeIdx, 0, 0);

  const replies = [];
  const mixer = new AudioMixerWorker((reply) => replies.push(reply), {
    outputState: 'active',
    autoDrain: false,
    declickFrames: 0,
  });
  mixer.handleMessage({ init: true, sampleRate: 48000, channels: 2 });

  const bridgeWorker = {
    postMessage(msg) {
      mixer.handleMessage(msg);
    },
  };

  const mockWasmMod = {
    _playgta6_audio_ring: () => BigInt(layout.structOffset),
    wasmMemory: { buffer: layout.buffer },
  };

  // Attach through loader.js helper using the real audio_worker.js message protocol
  const attachRes = attachRuntimeAudioRing(mockWasmMod, bridgeWorker, {
    id: 'w254-e2e-attach',
    softClip: false,
    quantumFrames: 6,
  });
  assert.equal(attachRes.attached, true);
  assert.equal(attachRes.alreadyAttached, false);

  const attachReply = replies.find((r) => r && r.id === 'w254-e2e-attach');
  assert.ok(attachReply, 'expected attach reply from AudioMixerWorker');
  assert.equal(attachReply.attached, true);
  assert.equal(attachReply.guestRingAttached, true);
  assert.equal(attachReply.sampleRate, 48000);
  assert.equal(attachReply.channels, 2);
  assert.equal(attachReply.capacitySamples, 16);
  assert.equal(attachReply.capacityFrames, 8);

  // Producer helper matching runtime/os/audio_downmix.cpp ring_push semantics
  const mask = layout.capacity - 1;
  function producerPush(interleaved) {
    let w = Atomics.load(writeIdx, 0) >>> 0;
    for (let i = 0; i < interleaved.length; i += 1) {
      ringData[(w + i) & mask] = interleaved[i];
    }
    Atomics.store(writeIdx, 0, (w + interleaved.length) >>> 0);
  }

  // Batch 1: 6 stereo frames (12 floats), well within [-1, 1]
  const batch1 = new Float32Array([
    0.125, -0.125,
    0.25, -0.25,
    0.375, -0.375,
    0.5, -0.5,
    -0.625, 0.625,
    -0.75, 0.75,
  ]);
  producerPush(batch1);

  const out1 = new Float32Array(batch1.length);
  const drain1 = mixer.drainGuestPcmRing(out1);
  assert.equal(drain1.framesDrained, 6);
  assert.equal(drain1.underrun, false);
  for (let i = 0; i < batch1.length; i += 1) {
    assert.ok(
      Math.abs(out1[i] - batch1[i]) < 1e-6,
      `batch1 sample[${i}]: expected ${batch1[i]}, got ${out1[i]}`,
    );
  }
  assert.equal(Atomics.load(readIdx, 0) >>> 0, 12, 'read_index must advance atomically to 12 floats');

  // Batch 2: 5 stereo frames (10 floats) wrapping across the 16-float ring boundary (12 -> 22)
  const batch2 = new Float32Array([
    0.1, 0.2,
    0.3, 0.4,
    -0.1, -0.2,
    -0.3, -0.4,
    0.05, -0.05,
  ]);
  producerPush(batch2);

  const out2 = new Float32Array(batch2.length);
  const drain2 = mixer.drainGuestPcmRing(out2);
  assert.equal(drain2.framesDrained, 5);
  assert.equal(drain2.underrun, false);
  for (let i = 0; i < batch2.length; i += 1) {
    assert.ok(
      Math.abs(out2[i] - batch2[i]) < 1e-6,
      `batch2 wrap sample[${i}]: expected ${batch2[i]}, got ${out2[i]}`,
    );
  }
  assert.equal(Atomics.load(readIdx, 0) >>> 0, 22, 'read_index must advance atomically across wrap to 22 floats');
});
