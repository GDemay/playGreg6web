import test from 'node:test';
import assert from 'node:assert/strict';

import { SmokeDemoRunner, inspectCommandBuffer } from '../web/demo/run_smoke_demo.js';
import { GPU_OPCODES, UNIFORM_BYTE_SIZE } from '../web/demo/smoke_scene.js';
import { PerfMonitor } from '../web/telemetry/perf_monitor.js';

function createMockWebGpuEnv() {
  const calls = [];
  let nextHandle = 1;
  let swapchainViewCounter = 0;

  const createdBuffers = [];
  const createdTextures = [];
  const configureCalls = [];
  let unconfigureCount = 0;

  let resolveLost;
  const lost = new Promise((resolve) => {
    resolveLost = resolve;
  });

  const context = {
    configure(config) {
      configureCalls.push({ ...config });
      calls.push({ op: 'context.configure', config: { ...config } });
    },
    unconfigure() {
      unconfigureCount += 1;
      calls.push({ op: 'context.unconfigure' });
    },
    getCurrentTexture() {
      const id = `swapchain-view-${swapchainViewCounter++}`;
      const view = { kind: 'GPUTextureView', id };
      return {
        createView() {
          return view;
        },
      };
    },
  };

  const canvas = {
    width: 0,
    height: 0,
    getContext(kind) {
      return kind === 'webgpu' ? context : null;
    },
  };

  const device = {
    limits: { maxTextureDimension2D: 8192 },
    lost,
    triggerLost(info = { reason: 'unknown', message: 'simulated loss' }) {
      resolveLost(info);
    },
    createBuffer(desc) {
      const buffer = {
        kind: 'GPUBuffer',
        handle: nextHandle++,
        size: desc.size,
        usage: desc.usage,
        destroyed: false,
        writes: [],
        destroy() {
          this.destroyed = true;
          calls.push({ op: 'buffer.destroy', buffer });
        },
      };
      createdBuffers.push(buffer);
      calls.push({ op: 'createBuffer', desc: { ...desc }, result: buffer });
      return buffer;
    },
    createTexture(desc) {
      const handle = nextHandle++;
      const texture = {
        kind: 'GPUTexture',
        handle,
        desc: { ...desc, size: { ...desc.size } },
        destroyed: false,
        createView() {
          const view = { kind: 'GPUTextureView', textureHandle: handle, handle: nextHandle++ };
          calls.push({ op: 'texture.createView', texture, result: view });
          return view;
        },
        destroy() {
          this.destroyed = true;
          calls.push({ op: 'texture.destroy', texture });
        },
      };
      createdTextures.push(texture);
      calls.push({ op: 'createTexture', desc: { ...desc, size: { ...desc.size } }, result: texture });
      return texture;
    },
    createSampler(desc) {
      const sampler = { kind: 'GPUSampler', handle: nextHandle++, desc: { ...desc } };
      calls.push({ op: 'createSampler', desc: { ...desc }, result: sampler });
      return sampler;
    },
    createShaderModule(desc) {
      const module = { kind: 'GPUShaderModule', handle: nextHandle++, code: desc.code };
      calls.push({ op: 'createShaderModule', desc: { ...desc }, result: module });
      return module;
    },
    createRenderPipeline(desc) {
      const pipeline = {
        kind: 'GPURenderPipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayout(index) {
          const layout = { kind: 'GPUBindGroupLayout', pipelineKind: 'render', index };
          calls.push({ op: 'renderPipeline.getBindGroupLayout', index, result: layout });
          return layout;
        },
      };
      calls.push({ op: 'createRenderPipeline', desc, result: pipeline });
      return pipeline;
    },
    createComputePipeline(desc) {
      const pipeline = {
        kind: 'GPUComputePipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayout(index) {
          const layout = { kind: 'GPUBindGroupLayout', pipelineKind: 'compute', index };
          calls.push({ op: 'computePipeline.getBindGroupLayout', index, result: layout });
          return layout;
        },
      };
      calls.push({ op: 'createComputePipeline', desc, result: pipeline });
      return pipeline;
    },
    createBindGroup(desc) {
      const bindGroup = { kind: 'GPUBindGroup', handle: nextHandle++, desc };
      calls.push({ op: 'createBindGroup', desc, result: bindGroup });
      return bindGroup;
    },
    createCommandEncoder() {
      const commandBuffer = { kind: 'GPUCommandBuffer', handle: nextHandle++ };
      return {
        beginRenderPass(desc) {
          calls.push({ op: 'beginRenderPass', desc });
          return {
            setPipeline(pipeline) {
              calls.push({ op: 'renderPass.setPipeline', pipeline });
            },
            setBindGroup(index, bindGroup) {
              calls.push({ op: 'renderPass.setBindGroup', index, bindGroup });
            },
            setVertexBuffer(slot, buffer, offset) {
              calls.push({ op: 'renderPass.setVertexBuffer', slot, buffer, offset });
            },
            setIndexBuffer(buffer, indexFormat, offset) {
              calls.push({ op: 'renderPass.setIndexBuffer', buffer, indexFormat, offset });
            },
            draw(vertexCount, instanceCount, firstVertex, firstInstance) {
              calls.push({
                op: 'renderPass.draw',
                vertexCount,
                instanceCount,
                firstVertex,
                firstInstance,
              });
            },
            drawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance) {
              calls.push({
                op: 'renderPass.drawIndexed',
                indexCount,
                instanceCount,
                firstIndex,
                baseVertex,
                firstInstance,
              });
            },
            end() {
              calls.push({ op: 'renderPass.end' });
            },
          };
        },
        beginComputePass() {
          calls.push({ op: 'beginComputePass' });
          return {
            setPipeline(pipeline) {
              calls.push({ op: 'computePass.setPipeline', pipeline });
            },
            setBindGroup(index, bindGroup) {
              calls.push({ op: 'computePass.setBindGroup', index, bindGroup });
            },
            dispatchWorkgroups(x, y, z) {
              calls.push({ op: 'computePass.dispatchWorkgroups', x, y, z });
            },
            end() {
              calls.push({ op: 'computePass.end' });
            },
          };
        },
        finish() {
          calls.push({ op: 'encoder.finish', commandBuffer });
          return commandBuffer;
        },
      };
    },
    queue: {
      writeBuffer(buffer, dstOffset, data) {
        const copy = new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
        buffer.writes.push({ dstOffset, bytes: copy });
        calls.push({ op: 'queue.writeBuffer', buffer, dstOffset, data: copy });
      },
      writeTexture(destination, data, dataLayout, size) {
        const copy = new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
        calls.push({
          op: 'queue.writeTexture',
          destination: {
            texture: destination.texture,
            mipLevel: destination.mipLevel,
            origin: { ...destination.origin },
          },
          data: copy,
          dataLayout: { ...dataLayout },
          size: { ...size },
        });
      },
      submit(commandBuffers) {
        calls.push({ op: 'queue.submit', commandBuffers: [...commandBuffers] });
      },
    },
  };

  return {
    device,
    context,
    canvas,
    calls,
    createdBuffers,
    createdTextures,
    configureCalls,
    get unconfigureCount() {
      return unconfigureCount;
    },
  };
}

function createFakeRafController(initialNow = 1000) {
  let clockMs = initialNow;
  let nextId = 1;
  const scheduled = new Map();
  let rafScheduleCount = 0;
  let rafCancelCount = 0;

  return {
    now: () => clockMs,
    setNow(ms) {
      clockMs = ms;
    },
    raf(cb) {
      const id = nextId++;
      rafScheduleCount += 1;
      scheduled.set(id, cb);
      return id;
    },
    cancelRaf(id) {
      rafCancelCount += 1;
      scheduled.delete(id);
    },
    flushNext(timestampMs) {
      const first = scheduled.entries().next();
      if (first.done) {
        return false;
      }
      const [id, cb] = first.value;
      scheduled.delete(id);
      if (timestampMs !== undefined) {
        clockMs = timestampMs;
      }
      cb(clockMs);
      return true;
    },
    get pendingCount() {
      return scheduled.size;
    },
    get scheduleCount() {
      return rafScheduleCount;
    },
    get cancelCount() {
      return rafCancelCount;
    },
  };
}

function decodePackets(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const packets = [];
  let offset = 0;
  while (offset + 8 <= view.byteLength) {
    const opcode = view.getUint16(offset, true);
    const flags = view.getUint16(offset + 2, true);
    const payloadBytes = view.getUint32(offset + 4, true);
    assert.ok(offset + 8 + payloadBytes <= view.byteLength, `Packet at ${offset} is truncated`);
    packets.push({
      opcode,
      flags,
      payloadBytes,
      payload: new Uint8Array(bytes.buffer, bytes.byteOffset + offset + 8, payloadBytes),
    });
    offset += 8 + payloadBytes;
  }
  assert.equal(offset, view.byteLength, 'Trace byte stream must align cleanly to packet boundaries');
  return packets;
}

test('SmokeDemoRunner full lifecycle: init, stepFrame x5, resize, recording, PerfMonitor + HUD, idempotent destroy', () => {
  const env = createMockWebGpuEnv();
  const fakeRaf = createFakeRafController(1000);
  const perfMonitor = new PerfMonitor({ now: fakeRaf.now });
  const hudUpdates = [];
  const mockHud = {
    updateMetrics(metrics) {
      hudUpdates.push({ ...metrics });
    },
  };

  // 40-byte binary InputState SharedArrayBuffer matching runtime/abi/playgta6_abi.h
  const inputSab = new SharedArrayBuffer(40);
  const inputView = new DataView(inputSab);
  inputView.setUint32(0, 1, true);
  inputView.setUint32(4, 0x100, true); // buttons
  inputView.setFloat32(16, 0.5, true); // leftStickX
  inputView.setFloat32(20, -0.25, true); // leftStickY
  inputView.setFloat32(36, 0.75, true); // rightTrigger

  const runner = new SmokeDemoRunner({
    device: env.device,
    context: env.context,
    canvas: env.canvas,
    format: 'bgra8unorm',
    width: 640,
    height: 360,
    devicePixelRatio: 1,
    ringCapacityBytes: 256 * 1024,
    inputSab,
    perfMonitor,
    hud: mockHud,
    enableRecorder: false,
    raf: fakeRaf.raf,
    cancelRaf: fakeRaf.cancelRaf,
    now: fakeRaf.now,
  });

  // --- 1. init() and idempotency ---
  const init1 = runner.init();
  assert.equal(init1.initialized, true);
  assert.equal(init1.alreadyInitialized, false);
  assert.ok(init1.bytesSubmitted > 16384);
  assert.equal(init1.bytesDrained, init1.bytesSubmitted);
  assert.equal(env.canvas.width, 640);
  assert.equal(env.canvas.height, 360);
  assert.equal(env.configureCalls.length, 1);
  assert.equal(runner.processor.buffers.size, 4);
  assert.equal(runner.processor.textures.size, 1);
  assert.equal(runner.processor.samplers.size, 1);
  assert.equal(runner.processor.shaderModules.size, 3);
  assert.equal(runner.processor.pipelines.size, 1);
  assert.equal(runner.processor.computePipelines.size, 1);
  assert.equal(runner.processor.bindGroups.size, 2);

  const callCountAfterInit = env.calls.length;
  const init2 = runner.init();
  assert.equal(init2.initialized, true);
  assert.equal(init2.alreadyInitialized, true);
  assert.equal(env.calls.length, callCountAfterInit, 'Second init() must be a no-op');

  // --- 2. stepFrame() x5 with deterministic timestamps ---
  const timestamps = [1000, 1016, 1032, 1048, 1064];
  for (let i = 0; i < timestamps.length; i++) {
    fakeRaf.setNow(timestamps[i]);
    const res = runner.stepFrame(timestamps[i]);
    assert.equal(res.frameIndex, i);
    assert.equal(res.timestampMs, timestamps[i]);
    assert.equal(res.presents, 1);
    assert.equal(res.bytesDrained, res.bytesSubmitted);
    assert.equal(res.drawCalls, 1);
    assert.equal(res.computeDispatches, 1);
    assert.equal(res.packets, 12);
    assert.equal(runner.ring.availableBytes(), 0);
  }

  assert.equal(runner.frameIndex, 5);
  assert.equal(runner.processor.frameCount, 5);

  // Verify inputSab values reached uniform buffer write
  const uniformWrites = env.calls.filter(
    (c) => c.op === 'queue.writeBuffer' && c.data.byteLength === UNIFORM_BYTE_SIZE
  );
  assert.ok(uniformWrites.length >= 5);
  const lastUniform = uniformWrites[uniformWrites.length - 1].data;
  const f32 = new Float32Array(lastUniform.buffer, lastUniform.byteOffset, UNIFORM_BYTE_SIZE / 4);
  assert.ok(Math.abs(f32[4] - 0.5) < 1e-6, 'leftStickX from inputSab should populate uniform buffer');
  assert.ok(Math.abs(f32[5] - -0.25) < 1e-6, 'leftStickY from inputSab should populate uniform buffer');
  assert.ok(Math.abs(f32[7] - 0.75) < 1e-6, 'rightTrigger from inputSab should populate uniform buffer');
  assert.equal(f32[12], 0x100, 'buttons bitmask from inputSab should populate uniform buffer');

  // Verify PerfMonitor + HUD metrics after 5 frames (4 deltas of 16ms => 62.5 FPS)
  assert.equal(hudUpdates.length, 5);
  const lastHud = hudUpdates[hudUpdates.length - 1];
  assert.equal(lastHud.totalFrames, 5);
  assert.equal(lastHud.avgFrameTimeMs, 16);
  assert.equal(lastHud.fps, 62.5);
  assert.equal(lastHud.gpuPackets, 5 * 12);
  assert.equal(lastHud.drawCallsPerFrame, 1);

  const statsAfter5 = runner.getStats();
  assert.equal(statsAfter5.initialized, true);
  assert.equal(statsAfter5.running, false);
  assert.equal(statsAfter5.frameIndex, 5);
  assert.equal(statsAfter5.swapchain.totalFrames, 5);
  assert.equal(statsAfter5.swapchain.avgFrameTimeMs, 16);
  assert.equal(statsAfter5.swapchain.fps, 62.5);
  assert.equal(statsAfter5.perf.frames.totalFrames, 5);
  assert.equal(statsAfter5.perf.gpu.totalDrawCalls, 5);
  assert.equal(statsAfter5.perf.gpu.totalComputeDispatches, 5);
  assert.equal(statsAfter5.perf.gpu.totalBatches, 5);
  assert.equal(statsAfter5.recorder, null);

  // --- 3. resize() ---
  assert.equal(runner.resize(1280, 720, 2), true);
  assert.equal(env.canvas.width, 2560);
  assert.equal(env.canvas.height, 1440);
  assert.equal(runner.swapchain.getTelemetry().resizeCount, 1);
  assert.ok(Math.abs(runner.scene.aspect - 16 / 9) < 1e-6);

  // No-op resize with identical pixel dimensions
  assert.equal(runner.resize(1280, 720, 2), false);
  assert.equal(runner.swapchain.getTelemetry().resizeCount, 1);

  // --- 4. Live .pg6trace recording: startRecording() -> stepFrame() x2 -> exportTraceBytes() ---
  assert.equal(runner.exportTraceBytes().byteLength, 0);
  runner.startRecording();
  runner.stepFrame(1080);
  runner.stepFrame(1096);
  runner.stopRecording();

  // Frame after stopRecording() should not be captured
  runner.stepFrame(1112);

  const traceBytes = runner.exportTraceBytes();
  assert.ok(traceBytes instanceof Uint8Array);
  assert.ok(traceBytes.byteLength > 0, 'Recorded .pg6trace must be non-empty');

  const tracePackets = decodePackets(traceBytes);
  assert.equal(tracePackets.length, 24, '2 smoke frames should contain 2 * 12 = 24 packets');
  assert.equal(tracePackets[0].opcode, GPU_OPCODES.BEGIN_FRAME, 'Trace must begin with BeginFrame (1)');
  assert.equal(tracePackets[0].payloadBytes, 0);
  assert.equal(
    tracePackets[tracePackets.length - 1].opcode,
    GPU_OPCODES.PRESENT,
    'Trace must end with Present (7)'
  );
  assert.equal(tracePackets[tracePackets.length - 1].payloadBytes, 0);

  const meta = runner.exportTraceMetadata();
  assert.equal(meta.frameCount, 2);
  assert.equal(meta.recording, false);
  assert.equal(meta.totalBytes, traceBytes.byteLength);
  assert.equal(meta.opcodeCounts.BeginFrame, 2);
  assert.equal(meta.opcodeCounts.Present, 2);
  assert.equal(meta.opcodeCounts.DispatchCompute, 2);
  assert.equal(meta.opcodeCounts.DrawIndexed, 2);

  // --- 5. Idempotent destroy() ---
  runner.destroy();
  assert.equal(runner.destroyed, true);
  assert.equal(runner.initialized, false);
  assert.equal(runner.running, false);

  // All processor resources freed and GPU buffers/textures destroyed
  assert.equal(runner.processor.buffers.size, 0);
  assert.equal(runner.processor.textures.size, 0);
  assert.equal(runner.processor.samplers.size, 0);
  assert.equal(runner.processor.bindGroups.size, 0);
  assert.equal(runner.processor.shaderModules.size, 0);
  assert.equal(runner.processor.pipelines.size, 0);
  assert.equal(runner.processor.computePipelines.size, 0);

  for (const buf of env.createdBuffers) {
    assert.equal(buf.destroyed, true, 'All GPUBuffers must be destroyed');
  }
  for (const tex of env.createdTextures) {
    assert.equal(tex.destroyed, true, 'All GPUTextures (scene + depth) must be destroyed');
  }
  assert.equal(env.unconfigureCount, 1);

  const callsAfterDestroy = env.calls.length;
  runner.destroy();
  assert.equal(env.calls.length, callsAfterDestroy, 'Second destroy() must be idempotent');

  // Mutating calls after destroy must throw cleanly
  assert.throws(() => runner.init(), /destroyed/);
  assert.throws(() => runner.stepFrame(1200), /destroyed/);
  assert.throws(() => runner.start(), /destroyed/);
  assert.throws(() => runner.resize(800, 600), /destroyed/);
});

test('SmokeDemoRunner start()/stop() rAF loop handles double-start, double-stop, resume, and destroy while running', () => {
  const env = createMockWebGpuEnv();
  const fakeRaf = createFakeRafController(2000);
  const hudUpdates = [];

  const runner = new SmokeDemoRunner({
    device: env.device,
    context: env.context,
    canvas: env.canvas,
    hud: {
      updateMetrics(m) {
        hudUpdates.push(m);
      },
    },
    raf: fakeRaf.raf,
    cancelRaf: fakeRaf.cancelRaf,
    now: fakeRaf.now,
  });

  // First start auto-initializes and schedules exactly one rAF callback
  assert.equal(runner.start(), true);
  assert.equal(runner.initialized, true);
  assert.equal(runner.running, true);
  assert.equal(fakeRaf.pendingCount, 1);
  assert.equal(fakeRaf.scheduleCount, 1);

  // Double-start is a safe no-op
  assert.equal(runner.start(), false);
  assert.equal(fakeRaf.pendingCount, 1);
  assert.equal(fakeRaf.scheduleCount, 1);

  // Advance 3 rAF ticks
  assert.equal(fakeRaf.flushNext(2000), true);
  assert.equal(fakeRaf.flushNext(2016), true);
  assert.equal(fakeRaf.flushNext(2032), true);
  assert.equal(runner.frameIndex, 3);
  assert.equal(fakeRaf.pendingCount, 1);
  assert.equal(hudUpdates.length, 3);
  assert.equal(hudUpdates[2].fps, 62.5);
  assert.equal(hudUpdates[2].totalFrames, 3);

  // Stop cancels pending rAF and double-stop is safe
  assert.equal(runner.stop(), true);
  assert.equal(runner.running, false);
  assert.equal(fakeRaf.pendingCount, 0);
  assert.equal(fakeRaf.cancelCount, 1);

  assert.equal(runner.stop(), false);
  assert.equal(fakeRaf.cancelCount, 1);

  // Resume works cleanly
  assert.equal(runner.start(), true);
  assert.equal(fakeRaf.pendingCount, 1);
  assert.equal(fakeRaf.flushNext(2048), true);
  assert.equal(runner.frameIndex, 4);

  // Destroy while running cancels active rAF cleanly
  runner.destroy();
  assert.equal(runner.running, false);
  assert.equal(fakeRaf.pendingCount, 0);
});

test('SmokeDemoRunner enableRecorder=true captures frames immediately and inspectCommandBuffer counts opcodes accurately', () => {
  const env = createMockWebGpuEnv();
  const fakeRaf = createFakeRafController(500);

  const runner = new SmokeDemoRunner({
    device: env.device,
    context: env.context,
    canvas: env.canvas,
    enableRecorder: true,
    perfMonitor: true,
    raf: fakeRaf.raf,
    cancelRaf: fakeRaf.cancelRaf,
    now: fakeRaf.now,
  });

  runner.init();
  runner.stepFrame(500);
  runner.stepFrame(516);

  const trace = runner.exportTraceBytes();
  const summary = inspectCommandBuffer(trace);
  assert.equal(summary.packets, 24);
  assert.equal(summary.drawCalls, 2);
  assert.equal(summary.computeDispatches, 2);
  assert.equal(summary.bytes, trace.byteLength);

  const stats = runner.getStats();
  assert.equal(stats.recorder.frameCount, 2);
  assert.equal(stats.recorder.recording, true);
  assert.equal(stats.perf.frames.totalFrames, 2);
  assert.equal(stats.perf.gpu.totalDrawCalls, 2);

  runner.destroy();
  assert.equal(runner.exportTraceMetadata().recording, false);
});
