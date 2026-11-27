import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const wgpu = require('../web/workers/wgpu_worker.js');
const loader = require('../web/loader.js');
const { installRenderStatePackets } = require('../web/workers/render_state_packets.js');
const { bootstrapPlayGta6App } = await import('../web/app_bootstrap.js');

const {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  buildGpuTelemetrySnapshot,
  OPCODES,
  BUFFER_USAGE,
  TEXTURE_USAGE,
  REPLAY_LOG_MAX_ENTRIES,
  REPLAY_LOG_MAX_BYTES,
  REPLAY_LOG_MAX_BIND_GROUPS,
  SWAPCHAIN_PROBE_DEFAULT_TIMEOUT_MS,
  DEFAULT_GPU_STALL_WARN_MS,
} = wgpu;

const { createWorkerBootInitPayloads, forwardGpuWorkerMessage } = loader;

function makePacket(opcode, payloadBytes, writePayload) {
  const buf = new ArrayBuffer(8 + payloadBytes);
  const view = new DataView(buf);
  view.setUint16(0, opcode, true);
  view.setUint16(2, 0, true);
  view.setUint32(4, payloadBytes, true);
  if (writePayload) {
    writePayload(view, 8);
  }
  return new Uint8Array(buf);
}

function concatPackets(...parts) {
  const total = parts.reduce((acc, p) => acc + p.byteLength, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.byteLength;
  }
  return out;
}

function createBufferPacket(id, usage, size) {
  return makePacket(OPCODES.CREATE_BUFFER, 16, (v, o) => {
    v.setUint32(o, id, true);
    v.setUint32(o + 4, usage, true);
    v.setBigUint64(o + 8, BigInt(size), true);
  });
}

function writeBufferPacket(id, dstOffset, dataBytes) {
  const bytes = dataBytes instanceof Uint8Array ? dataBytes : new Uint8Array(dataBytes);
  return makePacket(OPCODES.WRITE_BUFFER, 16 + bytes.byteLength, (v, o) => {
    v.setUint32(o, id, true);
    v.setUint32(o + 4, dstOffset, true);
    v.setUint32(o + 8, bytes.byteLength, true);
    v.setUint32(o + 12, 0, true);
    new Uint8Array(v.buffer, v.byteOffset + o + 16, bytes.byteLength).set(bytes);
  });
}

function destroyResourcePacket(id) {
  return makePacket(OPCODES.DESTROY_RESOURCE, 4, (v, o) => {
    v.setUint32(o, id, true);
  });
}

function createShaderModulePacket(id, stage, code) {
  const enc = new TextEncoder().encode(code);
  return makePacket(OPCODES.CREATE_SHADER_MODULE, 16 + enc.byteLength, (v, o) => {
    v.setUint32(o, id, true);
    v.setUint32(o + 4, stage, true);
    v.setUint32(o + 8, enc.byteLength, true);
    v.setUint32(o + 12, 0, true);
    new Uint8Array(v.buffer, v.byteOffset + o + 16, enc.byteLength).set(enc);
  });
}

function createRenderPipelinePacket(id, vsId, fsId, stride = 0, topology = 0) {
  return makePacket(OPCODES.CREATE_RENDER_PIPELINE, 24, (v, o) => {
    v.setUint32(o, id, true);
    v.setUint32(o + 4, vsId, true);
    v.setUint32(o + 8, fsId, true);
    v.setUint32(o + 12, stride, true);
    v.setUint32(o + 16, topology, true);
    v.setUint32(o + 20, 0, true);
  });
}

function createBindGroupLayoutPacket(id, entries) {
  const payloadBytes = 24 + entries.length * 24;
  return makePacket(OPCODES.CREATE_BIND_GROUP_LAYOUT, payloadBytes, (v, o) => {
    v.setUint32(o, id, true);
    v.setUint32(o + 4, entries.length, true);
    for (let i = 0; i < entries.length; i += 1) {
      const e = entries[i];
      const base = o + 24 + i * 24;
      v.setUint32(base, e.binding, true);
      v.setUint32(base + 4, e.visibility ?? 3, true);
      v.setUint8(base + 8, e.resourceType ?? 0); // 0 = buffer
      v.setUint8(base + 9, e.subtype ?? 0); // 0 = uniform
      v.setUint8(base + 10, e.dimension ?? 0);
      v.setUint8(base + 11, e.flags ?? 0);
      v.setBigUint64(base + 16, BigInt(e.minBindingSize ?? 0), true);
    }
  });
}

function createBindGroup2Packet(id, layoutId, entries) {
  const payloadBytes = 16 + entries.length * 48;
  return makePacket(OPCODES.CREATE_BIND_GROUP2, payloadBytes, (v, o) => {
    v.setUint32(o, id, true);
    v.setUint32(o + 4, layoutId, true);
    v.setUint32(o + 8, entries.length, true);
    for (let i = 0; i < entries.length; i += 1) {
      const e = entries[i];
      const base = o + 16 + i * 48;
      v.setUint32(base, e.binding, true);
      v.setUint32(base + 4, e.resourceType ?? 0, true);
      v.setUint32(base + 8, e.resourceId, true);
      v.setBigUint64(base + 16, BigInt(e.offset ?? 0), true);
      v.setBigUint64(base + 24, BigInt(e.size ?? 0), true);
    }
  });
}

function setBindGroup2Packet(groupIndex, bindGroupId, dynamicOffsets = []) {
  const payloadBytes = 16 + dynamicOffsets.length * 4;
  return makePacket(OPCODES.SET_BIND_GROUP2, payloadBytes, (v, o) => {
    v.setUint32(o, groupIndex, true);
    v.setUint32(o + 4, bindGroupId, true);
    v.setUint32(o + 8, dynamicOffsets.length, true);
    for (let i = 0; i < dynamicOffsets.length; i += 1) {
      v.setUint32(o + 16 + i * 4, dynamicOffsets[i], true);
    }
  });
}

function setPipelinePacket(id) {
  return makePacket(OPCODES.SET_PIPELINE, 4, (v, o) => v.setUint32(o, id, true));
}

function drawPacket(vertexCount, instanceCount = 1) {
  return makePacket(OPCODES.DRAW, 16, (v, o) => {
    v.setUint32(o, vertexCount, true);
    v.setUint32(o + 4, instanceCount, true);
    v.setUint32(o + 8, 0, true);
    v.setUint32(o + 12, 0, true);
  });
}

function presentPacket() {
  return makePacket(OPCODES.PRESENT, 0);
}

function createMockDevice(opts = {}) {
  let resolveLost;
  const lostPromise = new Promise((r) => {
    resolveLost = r;
  });
  const writeBuffers = [];
  const submitted = [];
  const createdBindGroups = [];
  const createdBindGroupLayouts = [];
  const createdBuffers = [];
  const probeBuffers = [];

  const device = {
    label: opts.label || 'mock-device',
    lost: lostPromise,
    features: new Set(),
    limits: {},
    createBuffer(desc) {
      const isProbe = desc && desc.label === 'playgta6-swapchain-luma-probe';
      const buf = {
        ...desc,
        destroyed: false,
        unmappedCount: 0,
        destroy() {
          this.destroyed = true;
        },
        mapAsync(mode) {
          if (typeof opts.onProbeMapAsync === 'function' && isProbe) {
            return opts.onProbeMapAsync(buf, mode, probeBuffers.indexOf(buf));
          }
          return Promise.resolve();
        },
        getMappedRange() {
          if (typeof opts.onProbeGetMappedRange === 'function' && isProbe) {
            return opts.onProbeGetMappedRange(buf, probeBuffers.indexOf(buf));
          }
          return new ArrayBuffer(desc.size || 2048);
        },
        unmap() {
          this.unmappedCount += 1;
        },
      };
      createdBuffers.push(buf);
      if (isProbe) {
        probeBuffers.push(buf);
      }
      return buf;
    },
    createTexture(desc) {
      return {
        ...desc,
        destroyed: false,
        createView() {
          return { kind: 'view', desc };
        },
        destroy() {
          this.destroyed = true;
        },
      };
    },
    createSampler(desc) {
      return { ...desc };
    },
    createShaderModule(desc) {
      if (typeof opts.onCreateShaderModule === 'function') {
        opts.onCreateShaderModule(desc);
      }
      return { code: desc.code };
    },
    createBindGroupLayout(desc) {
      const bgl = { kind: 'bgl', desc, device };
      createdBindGroupLayouts.push(bgl);
      return bgl;
    },
    createPipelineLayout(desc) {
      return { kind: 'pl', desc, device };
    },
    createBindGroup(desc) {
      const bg = { kind: 'bg', desc, device };
      createdBindGroups.push(bg);
      return bg;
    },
    createRenderPipeline(desc) {
      if (typeof opts.onCreateRenderPipeline === 'function') {
        opts.onCreateRenderPipeline(desc);
      }
      return {
        kind: 'rp',
        desc,
        device,
        getBindGroupLayout() {
          return { kind: 'auto-bgl', device };
        },
      };
    },
    createComputePipeline(desc) {
      return {
        kind: 'cp',
        desc,
        device,
        getBindGroupLayout() {
          return { kind: 'auto-cbgl', device };
        },
      };
    },
    createCommandEncoder() {
      const passCalls = [];
      return {
        beginRenderPass() {
          return {
            setPipeline(p) {
              passCalls.push(['setPipeline', p]);
            },
            setBindGroup(idx, bg) {
              passCalls.push(['setBindGroup', idx, bg]);
            },
            setVertexBuffer() {},
            setIndexBuffer() {},
            setViewport() {},
            setScissorRect() {},
            draw(vc, ic) {
              passCalls.push(['draw', vc, ic]);
            },
            drawIndexed() {},
            end() {},
          };
        },
        beginComputePass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups() {},
            end() {},
          };
        },
        copyBufferToBuffer() {},
        copyBufferToTexture() {},
        copyTextureToBuffer() {},
        finish() {
          return { passCalls };
        },
      };
    },
    queue: {
      writeBuffer(buf, offset, data) {
        writeBuffers.push({ buf, offset, byteLength: data.byteLength });
      },
      writeTexture() {},
      submit(cbs) {
        if (typeof opts.onQueueSubmit === 'function') {
          opts.onQueueSubmit(cbs);
        }
        submitted.push(...cbs);
      },
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
    destroy() {},
    lose(info = { reason: 'unknown', message: 'GPU process crashed' }) {
      resolveLost(info);
    },
    _writeBuffers: writeBuffers,
    _submitted: submitted,
    _createdBindGroups: createdBindGroups,
    _createdBindGroupLayouts: createdBindGroupLayouts,
    _createdBuffers: createdBuffers,
    _probeBuffers: probeBuffers,
  };

  const ctx = {
    configure() {},
    getCurrentTexture() {
      return {
        width: 1280,
        height: 720,
        format: 'bgra8unorm',
        usage: TEXTURE_USAGE.RENDER_ATTACHMENT | TEXTURE_USAGE.COPY_SRC,
        createView() {
          return { kind: 'swapchain-view' };
        },
      };
    },
  };

  return { device, ctx };
}

test('Wave-5 #1: enableShadowBufferReplay=false skips WriteBuffer shadow copies while replaying creation packets & >512 bind groups across device.lost with 0 unknown bind group errors', async () => {
  assert.equal(REPLAY_LOG_MAX_ENTRIES, 262144);
  assert.equal(REPLAY_LOG_MAX_BYTES, 256 * 1024 * 1024);
  assert.equal(REPLAY_LOG_MAX_BIND_GROUPS, 65536);

  // 1a. Standalone WgpuCommandProcessor defaults to shadowBufferReplayEnabled = true for unit tests.
  const standaloneEnv = createMockDevice();
  const standaloneProc = new WgpuCommandProcessor(standaloneEnv.device, standaloneEnv.ctx, {
    enableShaderFallback: false,
  });
  assert.equal(standaloneProc.resourceReplayEnabled, true);
  assert.equal(standaloneProc.shadowBufferReplayEnabled, true);
  standaloneProc.executeCommandBuffer(
    concatPackets(
      createBufferPacket(1, BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST, 64),
      writeBufferPacket(1, 0, new Uint8Array([1, 2, 3, 4]))
    )
  );
  assert.equal(standaloneProc.getResourceReplayStats().shadowBufferCount, 1);
  standaloneProc.destroy();

  // 1b. Live config (enableResourceReplay: true, enableShadowBufferReplay: false) skips WriteBuffer
  // shadow-copying, retains >512 bind groups (e.g. 1200) without eviction, replays all layouts &
  // bind groups on device.lost recovery, and logs sanitized diagnostic warnings with replayed counts.
  const dev1Env = createMockDevice({ label: 'dev-1' });
  const dev2Env = createMockDevice({ label: 'dev-2' });
  const logs = [];
  const logger = {
    warn: (m) => logs.push({ level: 'warn', message: String(m) }),
    info: (m) => logs.push({ level: 'info', message: String(m) }),
    error: (m) => logs.push({ level: 'error', message: String(m) }),
  };

  const posted = [];
  const handler = createWgpuWorkerHandler({
    postMessage: (m) => posted.push(m),
    logger,
    deviceRecoveryMinIntervalMs: 0,
    initGpu: async (offscreen, width, height, initPayload) => {
      const p = new WgpuCommandProcessor(dev1Env.device, dev1Env.ctx, {
        width,
        height,
        format: 'bgra8unorm',
        logger,
        enableShaderFallback: false,
        enableResourceReplay: initPayload.enableResourceReplay,
        enableShadowBufferReplay: initPayload.enableShadowBufferReplay,
        captureValidationErrors: true,
      });
      installRenderStatePackets(p);
      return p;
    },
    recoverGpuDevice: async () => ({
      device: dev2Env.device,
      ctx: dev2Env.ctx,
      format: 'bgra8unorm',
    }),
  });

  // Use createWorkerBootInitPayloads defaults (enableResourceReplay: true, enableShadowBufferReplay: false).
  const bootPayloads = createWorkerBootInitPayloads({
    canvas: { width: 1280, height: 720 },
    width: 1280,
    height: 720,
    dataBase: '/data/',
  });
  assert.equal(bootPayloads.gpuInit.enableResourceReplay, true);
  assert.equal(bootPayloads.gpuInit.enableShadowBufferReplay, false);

  await handler.handleMessage(bootPayloads.gpuInit);
  const proc = handler.processor;
  assert.equal(proc.resourceReplayEnabled, true);
  assert.equal(proc.shadowBufferReplayEnabled, false);

  // Create buffer + write data + layout + 700 bind groups + destroy 10 + shader + pipeline.
  const packets = [
    createBufferPacket(10, BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST, 256),
    writeBufferPacket(10, 0, new Uint8Array(64).fill(0xab)),
    createBindGroupLayoutPacket(20, [{ binding: 0, visibility: 3, resourceType: 0 }]),
    createShaderModulePacket(30, 0, '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }'),
    createShaderModulePacket(31, 1, '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }'),
    createRenderPipelinePacket(40, 30, 31),
  ];
  for (let i = 0; i < 700; i += 1) {
    packets.push(
      createBindGroup2Packet(1000 + i, 20, [{ binding: 0, resourceType: 0, resourceId: 10, offset: 0, size: 256 }])
    );
  }
  packets.push(destroyResourcePacket(1000));
  handler.handleMessage({ cmdBuffer: concatPackets(...packets).buffer });

  const replayBefore = proc.getResourceReplayStats();
  assert.equal(replayBefore.shadowBufferCount, 0, 'WriteBuffer shadow copy must be skipped when enableShadowBufferReplay=false');
  assert.equal(replayBefore.shadowBufferBytes, 0);
  assert.equal(replayBefore.replayBindGroupEvictions, 0, '700 bind groups must fit well under 16384 cap');

  // Trigger device.lost and wait for async chunked replay recovery to finish.
  dev1Env.device.lose({ reason: 'unknown', message: 'NVIDIA Turing TDR reset' });
  for (let i = 0; i < 120; i += 1) {
    if (proc.getDeviceLossStats().deviceRecoveredCount === 1) break;
    await new Promise((r) => setTimeout(r, 5));
  }

  assert.equal(proc.getDeviceLossStats().deviceLostCount, 1);
  assert.equal(proc.getDeviceLossStats().deviceRecoveredCount, 1);
  assert.ok(proc.getDeviceLossStats().replayedResourceCount >= 700, 'all live resources + bind groups replayed');
  assert.equal(proc.getDeviceLossStats().replayedBufferWriteCount, 0, 'no shadow buffer writes replayed');
  assert.equal(dev2Env.device._writeBuffers.length, 0);

  // Subsequent CreateBindGroup2 (reusing layout 20) + SetBindGroup2 (using replayed bind group 1699) + Draw + Present
  // must succeed with ZERO validation errors!
  const errorsBefore = proc.validationErrorCount;
  handler.handleMessage({
    cmdBuffer: concatPackets(
      createBindGroup2Packet(9999, 20, [{ binding: 0, resourceType: 0, resourceId: 10, offset: 0, size: 256 }]),
      setPipelinePacket(40),
      setBindGroup2Packet(0, 1699),
      drawPacket(3, 1),
      presentPacket()
    ).buffer,
  });
  assert.equal(proc.validationErrorCount, errorsBefore, 'no unknown bind group or layout errors after recovery');

  // Verify sanitized diagnostic warnings on both device.lost and recovery completion (with replayed count).
  const lostWarn = logs.find((l) => l.level === 'warn' && /WebGPU device lost/.test(l.message));
  const recWarn = logs.find(
    (l) => l.level === 'warn' && /WebGPU device recovered/.test(l.message) && /replayed=\d+/.test(l.message)
  );
  assert.ok(lostWarn, 'expected sanitized warn log on device.lost');
  assert.ok(recWarn, 'expected sanitized warn log on recovery completion including replayed count');

  proc.destroy();
});

test('Wave-5 #2a: swapchain probe mapAsync timeout watchdog resets in-flight wedge, recreates buffer, and recovers cleanly', async () => {
  assert.equal(SWAPCHAIN_PROBE_DEFAULT_TIMEOUT_MS, 5000);

  let resolveHungMap = null;
  let probeIdx = 0;
  const env = createMockDevice({
    onProbeMapAsync(_buf) {
      probeIdx += 1;
      if (probeIdx === 1) {
        // First mapAsync hangs forever (simulating Chrome GPU process dropping callback).
        return new Promise((resolve) => {
          resolveHungMap = resolve;
        });
      }
      // Second probe resolves promptly.
      return Promise.resolve();
    },
    onProbeGetMappedRange() {
      // Bright non-black pixels so second probe detects visible content.
      const ab = new ArrayBuffer(2048);
      new Uint8Array(ab).fill(200);
      return ab;
    },
  });

  let fakeNow = 1000;
  const visibleEvents = [];
  const proc = new WgpuCommandProcessor(env.device, env.ctx, {
    width: 1280,
    height: 720,
    format: 'bgra8unorm',
    now: () => fakeNow,
    swapchainProbeIntervalMs: 100,
    swapchainProbeTimeoutMs: 35,
    requireNonBlackFrame: true,
    onFirstVisibleFrame: (ev) => visibleEvents.push(ev),
  });

  const drawFrame = concatPackets(
    createShaderModulePacket(1, 0, '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }'),
    createShaderModulePacket(2, 1, '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }'),
    createRenderPipelinePacket(3, 1, 2),
    setPipelinePacket(3),
    drawPacket(3, 1),
    presentPacket()
  );

  // Frame 1 starts probe #1 which hangs.
  proc.executeCommandBuffer(drawFrame);
  assert.equal(proc.swapchainProbeInFlight, true);
  assert.equal(proc.swapchainProbeCount, 1);
  assert.equal(proc.swapchainProbeTimeoutCount, 0);

  // Wait past the 35ms watchdog timeout.
  await new Promise((r) => setTimeout(r, 80));

  assert.equal(proc.swapchainProbeInFlight, false, 'watchdog must clear _swapchainProbeInFlight');
  assert.equal(proc.swapchainProbeTimeoutCount, 1);
  assert.equal(proc.swapchainProbeErrorCount, 1);
  assert.equal(env.device._probeBuffers[0].destroyed, true, 'stalled staging buffer must be destroyed for safe recreation');

  // Late resolution of the timed-out promise must be safely ignored.
  if (resolveHungMap) resolveHungMap();
  await Promise.resolve();
  assert.equal(proc.swapchainProbeInFlight, false);
  assert.equal(visibleEvents.length, 0);

  // Frame 2 recreates a fresh probe buffer and succeeds cleanly!
  fakeNow = 1500;
  proc.executeCommandBuffer(concatPackets(setPipelinePacket(3), drawPacket(3, 1), presentPacket()));
  await new Promise((r) => setTimeout(r, 20));

  assert.equal(env.device._probeBuffers.length, 2, 'fresh staging buffer allocated on next probe');
  assert.equal(proc.swapchainProbeInFlight, false);
  assert.equal(proc.firstVisibleFrame, true);
  assert.equal(visibleEvents.length, 1);
  assert.equal(proc.getStats().swapchainProbeTimeoutCount, 1);

  proc.destroy();
});

test('Wave-5 #2b & #2c: auto-enable ranged shader streaming for shaderPackUrl (>64 MiB) unless explicitly disabled, and forward Wave-4/Wave-5 stats through loader and app_bootstrap', async () => {
  // 2b-1: createWorkerBootInitPayloads auto-enables streamShaders + 64 MiB threshold when shaderPackUrl is present.
  const autoStream = createWorkerBootInitPayloads({
    canvas: 'c',
    width: 1280,
    height: 720,
    dataBase: '/data/',
    shaderPackUrl: '/shaders/splash.wgsl.pack',
  });
  assert.equal(autoStream.gpuInit.streamShaders, true);
  assert.equal(autoStream.gpuInit.streamThresholdBytes, 64 * 1024 * 1024);

  // 2b-2: Explicit streamShaders: false disables auto-streaming.
  const optOut = createWorkerBootInitPayloads({
    canvas: 'c',
    width: 1280,
    height: 720,
    dataBase: '/data/',
    shaderPackUrl: '/shaders/splash.wgsl.pack',
    streamShaders: false,
  });
  assert.equal(optOut.gpuInit.streamShaders, false);
  assert.equal('streamThresholdBytes' in optOut.gpuInit, false);

  // 2b-3: Explicit custom threshold is preserved.
  const customThresh = createWorkerBootInitPayloads({
    canvas: 'c',
    width: 1280,
    height: 720,
    dataBase: '/data/',
    shaderPackUrl: '/shaders/splash.wgsl.pack',
    streamShaders: true,
    streamThresholdBytes: 32 * 1024 * 1024,
  });
  assert.equal(customThresh.gpuInit.streamShaders, true);
  assert.equal(customThresh.gpuInit.streamThresholdBytes, 32 * 1024 * 1024);

  // 2c-1: loader.forwardGpuWorkerMessage forwards coalescedPresents, oversizedWgslFallbacks,
  // deviceLostCount, deviceRecoveredCount, swapchainProbeTimeoutCount, gpuStallWarnCount, and shaderPackStats.rangedPackActive.
  const forwarded = [];
  const loaderPayload = forwardGpuWorkerMessage(
    {
      type: 'gpu-telemetry',
      frameStats: { framesPresented: 42, droppedFrames: 0, drawCalls: 120 },
      fallbackStats: { packHits: 80, builtinHits: 5, totalLookups: 85 },
      missingShaders: [],
      coalescedPresents: 11,
      oversizedWgslFallbacks: 3,
      swapchainProbeTimeoutCount: 2,
      gpuStallWarnCount: 4,
      deviceStats: {
        deviceLostCount: 2,
        deviceRecoveredCount: 2,
        lastDeviceLostReason: 'TDR reset',
        deviceRecovering: false,
        replayedResourceCount: 1500,
        replayedBufferWriteCount: 0,
      },
      shaderPackStats: {
        mode: 'ranged',
        packCount: 1,
        shaderCount: 54000,
        rangeFetches: 80,
        cachedWgslBytes: 204800,
        rangedPackActive: true,
        rangedPackThresholdBytes: 64 * 1024 * 1024,
        asyncShaderPendingCount: 1,
        rangedShaderFetches: 81,
        rangedShaderReady: 80,
        rangedShaderErrors: 0,
        oversizedWgslFallbacks: 3,
      },
      timestamp: 9999,
    },
    (m) => forwarded.push(m)
  );
  assert.equal(forwarded.length, 1);
  assert.equal(loaderPayload.coalescedPresents, 11);
  assert.equal(loaderPayload.oversizedWgslFallbacks, 3);
  assert.equal(loaderPayload.deviceLostCount, 2);
  assert.equal(loaderPayload.deviceRecoveredCount, 2);
  assert.equal(loaderPayload.swapchainProbeTimeoutCount, 2);
  assert.equal(loaderPayload.gpuStallWarnCount, 4);
  assert.equal(loaderPayload.rangedPackActive, true);
  assert.equal(loaderPayload.shaderPackStats.rangedPackActive, true);
  assert.equal(loaderPayload.deviceStats.deviceRecoveredCount, 2);

  // 2c-2: app_bootstrap ingests the forwarded payload into getDiagnosticsSnapshot() & getBootTelemetry(),
  // and honors ?streamShaders=0 by forwarding streamShaders: false to loader.js.
  const postedToWorker = [];
  const workerListeners = [];
  const mockWorker = {
    addEventListener(type, fn) {
      if (type === 'message') workerListeners.push(fn);
    },
    removeEventListener(type, fn) {
      const idx = workerListeners.indexOf(fn);
      if (idx >= 0) workerListeners.splice(idx, 1);
    },
    postMessage(payload) {
      postedToWorker.push(payload);
    },
    terminate() {},
  };

  const mockCanvas = {
    width: 1280,
    height: 720,
    dataset: {},
    transferControlToOffscreen: () => ({ width: 1280, height: 720 }),
  };

  const mockAdapter = {
    features: new Set(),
    limits: {},
    async requestDevice() {
      return createMockDevice().device;
    },
  };
  const mockGpu = {
    async requestAdapter() {
      return mockAdapter;
    },
    getPreferredCanvasFormat() {
      return 'bgra8unorm';
    },
  };
  const app = bootstrapPlayGta6App({
    windowObj: {
      location: { origin: 'http://127.0.0.1:8080', search: '?streamShaders=0' },
      crossOriginIsolated: true,
      SharedArrayBuffer: globalThis.SharedArrayBuffer,
      addEventListener() {},
      removeEventListener() {},
      navigator: { gpu: mockGpu },
    },
    documentObj: {
      addEventListener() {},
      removeEventListener() {},
    },
    navigatorObj: { gpu: mockGpu },
    fetchImpl: async () => ({
      ok: true,
      status: 200,
      headers: new Headers({ 'content-type': 'application/json' }),
      async json() {
        return {
          ok: true,
          coopCoep: true,
          hasWasmBinary: true,
          preferredShaderPackUrl: '/shaders/splash.wgsl.pack',
        };
      },
    }),
    canvas: mockCanvas,
    createWasmWorker: () => mockWorker,
  });

  try {
    const startRes = await app.start();
    assert.equal(startRes.ok, true);
    assert.equal(postedToWorker.length, 1);
    assert.equal(postedToWorker[0].streamShaders, false, '?streamShaders=0 must propagate streamShaders: false');

    for (const fn of workerListeners) {
      fn({ data: loaderPayload });
    }

    const diag = app.getDiagnosticsSnapshot();
    const bootTel = app.getBootTelemetry();
    assert.equal(diag.coalescedPresents, 11);
    assert.equal(diag.oversizedWgslFallbacks, 3);
    assert.equal(diag.deviceLostCount, 2);
    assert.equal(diag.deviceRecoveredCount, 2);
    assert.equal(diag.swapchainProbeTimeoutCount, 2);
    assert.equal(diag.gpuStallWarnCount, 4);
    assert.equal(diag.rangedPackActive, true);
    assert.equal(bootTel.gpu.coalescedPresents, 11);
    assert.equal(bootTel.gpu.oversizedWgslFallbacks, 3);
    assert.equal(bootTel.gpu.deviceLostCount, 2);
    assert.equal(bootTel.gpu.deviceRecoveredCount, 2);
    assert.equal(bootTel.gpu.rangedPackActive, true);
    assert.equal(bootTel.coalescedPresents, 11);
    assert.equal(bootTel.oversizedWgslFallbacks, 3);
    assert.equal(bootTel.deviceLostCount, 2);
    assert.equal(bootTel.deviceRecoveredCount, 2);
    assert.equal(bootTel.rangedPackActive, true);
  } finally {
    await app.destroy();
  }
});

test('Wave-5 #3: GPU stall / hang attribution warnings fire for >gpuStallWarnMs on createShaderModule, createRenderPipeline, queue.submit, and executeCommandBuffer with rate limiting', () => {
  assert.equal(DEFAULT_GPU_STALL_WARN_MS, 1000);

  let fakeNow = 1000;
  let stallShaderMs = 0;
  let stallPipelineMs = 0;
  let stallSubmitMs = 0;
  const warnings = [];

  const env = createMockDevice({
    onCreateShaderModule() {
      fakeNow += stallShaderMs;
    },
    onCreateRenderPipeline() {
      fakeNow += stallPipelineMs;
    },
    onQueueSubmit() {
      fakeNow += stallSubmitMs;
    },
  });

  const proc = new WgpuCommandProcessor(env.device, env.ctx, {
    width: 1280,
    height: 720,
    format: 'bgra8unorm',
    now: () => fakeNow,
    gpuStallWarnMs: 1000,
    gpuStallWarnMinIntervalMs: 1000,
    enableShaderFallback: false,
    logger: {
      warn: (m) => warnings.push(String(m)),
      info() {},
      error() {},
    },
  });

  // 1. Slow createShaderModule (> 1000ms) also causes executeCommandBuffer (> 1000ms).
  stallShaderMs = 1250;
  proc.executeCommandBuffer(
    createShaderModulePacket(1, 0, '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }')
  );
  stallShaderMs = 0;

  assert.ok(
    warnings.some((w) => /op=createShaderModule/.test(w) && /durationMs=1250/.test(w) && /shader=/.test(w)),
    `expected createShaderModule stall warning, got: ${JSON.stringify(warnings)}`
  );
  assert.ok(
    warnings.some((w) => /op=executeCommandBuffer/.test(w) && /durationMs=1250/.test(w) && /draws=0/.test(w)),
    `expected executeCommandBuffer stall warning, got: ${JSON.stringify(warnings)}`
  );

  // 2. Rate-limiting: another createShaderModule stall 100ms later (< 1000ms min interval) is suppressed.
  const countBeforeRateLimit = warnings.length;
  fakeNow += 100;
  stallShaderMs = 1100;
  // Advance past min interval only for createShaderModule test: wait — fakeNow + 1100 >= 1000ms!
  // So test rate limiting with a 50ms threshold and 2000ms minInterval instead!
  stallShaderMs = 0;
  proc.destroy();

  // Test all 4 operations + strict rate limiting cleanly with a 50ms stall threshold and 5000ms rate limit.
  fakeNow = 10000;
  const warnings2 = [];
  const env2 = createMockDevice({
    onCreateShaderModule() {
      fakeNow += stallShaderMs;
    },
    onCreateRenderPipeline() {
      fakeNow += stallPipelineMs;
    },
    onQueueSubmit() {
      fakeNow += stallSubmitMs;
    },
  });
  const proc2 = new WgpuCommandProcessor(env2.device, env2.ctx, {
    width: 1280,
    height: 720,
    format: 'bgra8unorm',
    now: () => fakeNow,
    gpuStallWarnMs: 50,
    gpuStallWarnMinIntervalMs: 5000,
    enableShaderFallback: false,
    logger: {
      warn: (m) => warnings2.push(String(m)),
      info() {},
      error() {},
    },
  });

  // Trigger createShaderModule stall twice within 5000ms -> should emit ONCE for createShaderModule.
  stallShaderMs = 120;
  proc2.executeCommandBuffer(
    concatPackets(
      createShaderModulePacket(1, 0, '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }'),
      createShaderModulePacket(2, 1, '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }')
    )
  );
  stallShaderMs = 0;
  assert.equal(warnings2.filter((w) => /op=createShaderModule/.test(w)).length, 1, 'rate-limited per op');
  assert.equal(warnings2.filter((w) => /op=executeCommandBuffer/.test(w)).length, 1);

  // Trigger createRenderPipeline stall.
  stallPipelineMs = 200;
  proc2.executeCommandBuffer(createRenderPipelinePacket(10, 1, 2));
  stallPipelineMs = 0;
  assert.equal(warnings2.filter((w) => /op=createRenderPipeline/.test(w) && /shader=/.test(w)).length, 1);

  // Trigger queue.submit stall on Present with 1 draw.
  stallSubmitMs = 300;
  proc2.executeCommandBuffer(concatPackets(setPipelinePacket(10), drawPacket(3, 1), presentPacket()));
  stallSubmitMs = 0;
  const submitWarn = warnings2.find((w) => /op=queue\.submit/.test(w));
  assert.ok(submitWarn, 'expected queue.submit stall warning');
  assert.match(submitWarn, /durationMs=300/);
  assert.match(submitWarn, /draws=1/);
  assert.match(submitWarn, /dispatches=0/);

  assert.equal(proc2.gpuStallWarnCount, 4);
  assert.equal(proc2.getStats().gpuStallWarnCount, 4);
  assert.equal(buildGpuTelemetrySnapshot(proc2).gpuStallWarnCount, 4);

  // After advancing past 5000ms cooldown, createShaderModule can warn again.
  fakeNow += 6000;
  stallShaderMs = 90;
  proc2.executeCommandBuffer(
    createShaderModulePacket(3, 0, '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }')
  );
  stallShaderMs = 0;
  assert.equal(warnings2.filter((w) => /op=createShaderModule/.test(w)).length, 2);

  proc2.destroy();
  void countBeforeRateLimit;
});
