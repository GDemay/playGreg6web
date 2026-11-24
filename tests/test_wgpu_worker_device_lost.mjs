import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { WgpuCommandProcessor, createWgpuWorkerHandler, buildGpuTelemetrySnapshot, OPCODES } =
  require('../web/workers/wgpu_worker.js');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function instanceLostError() {
  const err = new Error('A valid external Instance reference no longer exists.');
  err.name = 'OperationError';
  return err;
}

function packet(opcode, payload = new Uint8Array(0)) {
  const buf = new Uint8Array(8 + payload.byteLength);
  const view = new DataView(buf.buffer);
  view.setUint16(0, opcode, true);
  view.setUint32(4, payload.byteLength, true);
  buf.set(payload, 8);
  return buf;
}

function createBufferPacket(id, usage, size) {
  const payload = new Uint8Array(16);
  const view = new DataView(payload.buffer);
  view.setUint32(0, id, true);
  view.setUint32(4, usage, true);
  view.setBigUint64(8, BigInt(size), true);
  return packet(OPCODES.CREATE_BUFFER, payload);
}

function setPipelinePacket(id) {
  const payload = new Uint8Array(4);
  new DataView(payload.buffer).setUint32(0, id, true);
  return packet(OPCODES.SET_PIPELINE, payload);
}

function drawPacket(vertexCount) {
  const payload = new Uint8Array(16);
  const view = new DataView(payload.buffer);
  view.setUint32(0, vertexCount, true);
  view.setUint32(4, 1, true);
  return packet(OPCODES.DRAW, payload);
}

function concat(...parts) {
  const total = parts.reduce((n, p) => n + p.byteLength, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const p of parts) {
    out.set(p, offset);
    offset += p.byteLength;
  }
  return out;
}

function framePackets(...middle) {
  return concat(packet(OPCODES.BEGIN_FRAME), ...middle, packet(OPCODES.PRESENT));
}

function createMockDevice(label = 'device') {
  const calls = [];
  let resolveLost;
  const lost = new Promise((resolve) => {
    resolveLost = resolve;
  });
  const pass = {
    end() {},
    setPipeline() {},
    setBindGroup() {},
    setVertexBuffer() {},
    setIndexBuffer() {},
    setViewport() {},
    setScissorRect() {},
    draw() {
      calls.push('draw');
    },
    drawIndexed() {},
    dispatchWorkgroups() {},
  };
  const device = {
    label,
    lost,
    broken: false,
    destroyed: false,
    features: new Set(),
    limits: {},
    queue: {
      submit() {
        if (device.broken) {
          throw instanceLostError();
        }
        calls.push('submit');
      },
      writeBuffer() {},
      writeTexture() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
    createBuffer(desc) {
      calls.push('createBuffer');
      return {
        kind: 'GPUBuffer',
        label,
        size: desc.size,
        destroyed: false,
        destroy() {
          this.destroyed = true;
        },
      };
    },
    createTexture(desc) {
      return {
        kind: 'GPUTexture',
        desc,
        createView() {
          return { kind: 'GPUTextureView' };
        },
        destroy() {},
      };
    },
    createSampler() {
      return { kind: 'GPUSampler' };
    },
    createShaderModule(desc) {
      return { kind: 'GPUShaderModule', code: desc.code };
    },
    createBindGroupLayout() {
      return { kind: 'GPUBindGroupLayout' };
    },
    createPipelineLayout() {
      return { kind: 'GPUPipelineLayout' };
    },
    createBindGroup() {
      return { kind: 'GPUBindGroup' };
    },
    createRenderPipeline() {
      return { kind: 'GPURenderPipeline', getBindGroupLayout: () => ({}) };
    },
    createComputePipeline() {
      return { kind: 'GPUComputePipeline', getBindGroupLayout: () => ({}) };
    },
    createCommandEncoder() {
      if (device.broken) {
        throw instanceLostError();
      }
      return {
        beginRenderPass() {
          return pass;
        },
        beginComputePass() {
          return pass;
        },
        copyBufferToBuffer() {},
        copyBufferToTexture() {},
        finish() {
          return { kind: 'GPUCommandBuffer' };
        },
      };
    },
    addEventListener() {},
    removeEventListener() {},
    destroy() {
      device.destroyed = true;
      resolveLost({ reason: 'destroyed', message: 'Device was destroyed.' });
    },
  };
  device.lose = (info = { reason: 'unknown', message: 'GPU process lost' }) => resolveLost(info);
  return { device, calls };
}

function createMockContext() {
  const configured = [];
  return {
    configured,
    getCurrentTexture() {
      return {
        createView() {
          return { kind: 'GPUTextureView', id: 'swapchain' };
        },
      };
    },
    configure(desc) {
      configured.push(desc);
    },
  };
}

function createLogger() {
  const lines = [];
  const logger = (level, message) => lines.push({ level, message });
  return { logger, lines };
}

async function settle(rounds = 6) {
  for (let i = 0; i < rounds; i += 1) {
    await new Promise((resolve) => setImmediate(resolve));
  }
}

async function withMockNavigatorGpu(gpu, fn) {
  const prevDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {
    value: { gpu },
    configurable: true,
    writable: true,
  });
  try {
    return await fn();
  } finally {
    if (prevDescriptor) {
      Object.defineProperty(globalThis, 'navigator', prevDescriptor);
    } else {
      delete globalThis.navigator;
    }
  }
}

async function collectUnhandledRejections(fn) {
  const rejections = [];
  const onUnhandled = (reason) => rejections.push(reason);
  process.on('unhandledRejection', onUnhandled);
  try {
    await fn();
    await settle(3);
  } finally {
    process.off('unhandledRejection', onUnhandled);
  }
  return rejections;
}

function newProcessor(device, ctx, extra = {}) {
  return new WgpuCommandProcessor(device, ctx, {
    format: 'bgra8unorm',
    enableShaderFallback: false,
    logger: null,
    ...extra,
  });
}

// ---------------------------------------------------------------------------
// Processor-level detection
// ---------------------------------------------------------------------------

test('W100 device.lost resolution is counted once, sanitized, and notifies the listener', async () => {
  const { device } = createMockDevice();
  const events = [];
  const { logger, lines } = createLogger();
  const processor = newProcessor(device, createMockContext(), {
    logger,
    onDeviceLost: (evt) => events.push(evt),
  });

  device.lose({ reason: 'unknown', message: 'GPU crashed\n see https://example.invalid/x' });
  await settle();

  const stats = processor.getDeviceLossStats();
  assert.equal(stats.deviceLostCount, 1);
  assert.equal(stats.deviceRecoveredCount, 0);
  assert.equal(stats.deviceRecovering, false);
  assert.equal(processor.isDeviceLost, true);
  assert.match(stats.lastDeviceLostReason, /^DeviceLost\(unknown\): GPU crashed see <url>$/);
  assert.equal(events.length, 1);
  assert.equal(events[0].source, 'device.lost');
  assert.equal(events[0].firstForGeneration, true);
  assert.equal(lines.filter((l) => l.level === 'warn' && /device lost/.test(l.message)).length, 1);
});

test('W100 intentional device.destroy() and processor.destroy() never signal device loss', async () => {
  const a = createMockDevice('a');
  const eventsA = [];
  const procA = newProcessor(a.device, createMockContext(), { onDeviceLost: (e) => eventsA.push(e) });
  a.device.destroy();
  await settle();
  assert.equal(procA.getDeviceLossStats().deviceLostCount, 0);
  assert.deepEqual(eventsA, []);

  const b = createMockDevice('b');
  const eventsB = [];
  const procB = newProcessor(b.device, createMockContext(), { onDeviceLost: (e) => eventsB.push(e) });
  procB.destroy();
  b.device.lose();
  await settle();
  assert.equal(procB.getDeviceLossStats().deviceLostCount, 0);
  assert.deepEqual(eventsB, []);
});

test('W100 present OperationError signals loss once per device generation; non-loss errors do not', () => {
  const { device } = createMockDevice();
  const events = [];
  const processor = newProcessor(device, createMockContext(), {
    onDeviceLost: (evt) => events.push(evt),
  });

  // An unrelated validation-style present failure is not a device loss.
  processor._notePresentSubmitFailure(new Error('Validation: bad attachment'));
  assert.equal(processor.getDeviceLossStats().deviceLostCount, 0);

  device.broken = true;
  assert.doesNotThrow(() => processor.execute(framePackets()));
  assert.doesNotThrow(() => processor.execute(framePackets()));
  assert.equal(processor.frameCount, 0);
  const stats = processor.getDeviceLossStats();
  assert.equal(stats.deviceLostCount, 1);
  assert.match(stats.lastDeviceLostReason, /OperationError: A valid external Instance reference/);
  // Listener is re-notified (handler decides on retry budget) but only the first counts.
  assert.equal(events.length, 2);
  assert.deepEqual(
    events.map((e) => [e.source, e.firstForGeneration]),
    [
      ['present', true],
      ['present', false],
    ]
  );
});

test('W100 work-done fence rejection with torn-down instance signals device loss', async () => {
  const { device } = createMockDevice();
  device.queue.onSubmittedWorkDone = () => Promise.reject(instanceLostError());
  const events = [];
  const processor = newProcessor(device, createMockContext(), {
    onDeviceLost: (evt) => events.push(evt),
  });
  const rejections = await collectUnhandledRejections(async () => {
    await processor._getWorkDoneFence();
  });
  assert.deepEqual(rejections, []);
  assert.equal(processor.getDeviceLossStats().deviceLostCount, 1);
  assert.equal(events[0].source, 'work-done-fence');
});

test('W100 submissions are dropped while recovering without throwing or bumping frameCount', () => {
  const { device, calls } = createMockDevice();
  const processor = newProcessor(device, createMockContext());
  processor.execute(framePackets());
  assert.equal(processor.frameCount, 1);

  assert.equal(processor.beginDeviceRecovery(), true);
  assert.equal(processor.isDeviceRecovering, true);
  const submitsBefore = calls.filter((c) => c === 'submit').length;
  let result;
  assert.doesNotThrow(() => {
    result = processor.execute(framePackets(createBufferPacket(1, 0x28, 64)));
  });
  assert.deepEqual(result, { frameCount: 1, presents: 0 });
  assert.equal(calls.filter((c) => c === 'submit').length, submitsBefore);
  assert.equal(processor.buffers.size, 0);
  assert.equal(processor.getDeviceLossStats().droppedWhileRecovering, 1);
  assert.equal(processor.getDeviceLossStats().deviceRecovering, true);

  processor.endDeviceRecovery(false);
  assert.equal(processor.isDeviceRecovering, false);
  assert.equal(processor.getDeviceLossStats().deviceRecoveryFailedCount, 1);
});

test('W100 rebindDevice clears stale handles, reconfigures context, re-hooks lost, keeps counters', async () => {
  const first = createMockDevice('first');
  const ctx = createMockContext();
  const events = [];
  const processor = newProcessor(first.device, ctx, { onDeviceLost: (e) => events.push(e) });

  processor.execute(framePackets(createBufferPacket(7, 0x28, 64)));
  assert.equal(processor.frameCount, 1);
  const staleBuffer = processor.buffers.get(7);
  assert.ok(staleBuffer);
  processor.extension = { handlePacket: () => true, destroyResource() {} };

  first.device.lose();
  await settle();
  assert.equal(events.length, 1);

  const second = createMockDevice('second');
  const ctx2 = createMockContext();
  processor.beginDeviceRecovery();
  processor.rebindDevice(second.device, ctx2, { format: 'rgba8unorm' });
  processor.endDeviceRecovery(true);

  assert.equal(processor.device, second.device);
  assert.equal(processor.ctx, ctx2);
  assert.equal(processor.format, 'rgba8unorm');
  assert.deepEqual(ctx2.configured, [
    { device: second.device, format: 'rgba8unorm', alphaMode: 'opaque' },
  ]);
  assert.equal(processor.buffers.size, 0);
  assert.equal(staleBuffer.destroyed, true);
  assert.equal(processor.extension, null);
  assert.equal(processor.frameCount, 1);
  assert.deepEqual(
    {
      lost: processor.getDeviceLossStats().deviceLostCount,
      recovered: processor.getDeviceLossStats().deviceRecoveredCount,
      isLost: processor.isDeviceLost,
    },
    { lost: 1, recovered: 1, isLost: false }
  );

  // Stale resource IDs referenced after reset never throw; the frame still presents.
  assert.doesNotThrow(() =>
    processor.execute(framePackets(setPipelinePacket(42), drawPacket(3)))
  );
  assert.equal(processor.frameCount, 2);

  // A late lost resolution of the old device is ignored; the new device is hooked.
  first.device.lose();
  await settle();
  assert.equal(processor.getDeviceLossStats().deviceLostCount, 1);
  second.device.lose();
  await settle();
  assert.equal(processor.getDeviceLossStats().deviceLostCount, 2);
});

// ---------------------------------------------------------------------------
// Telemetry snapshot
// ---------------------------------------------------------------------------

test('W100 buildGpuTelemetrySnapshot exposes deviceStats with safe defaults', () => {
  assert.equal('deviceStats' in buildGpuTelemetrySnapshot(null, { timestamp: 1 }), false);

  const bare = buildGpuTelemetrySnapshot({ getStats: () => ({ frameCount: 1 }) }, { timestamp: 1 });
  assert.deepEqual(bare.deviceStats, {
    deviceLostCount: 0,
    deviceRecoveredCount: 0,
    lastDeviceLostReason: null,
    deviceRecovering: false,
  });

  const throwing = buildGpuTelemetrySnapshot(
    {
      getDeviceLossStats() {
        throw new Error('boom');
      },
    },
    { timestamp: 1 }
  );
  assert.equal(throwing.deviceStats.deviceLostCount, 0);

  const custom = buildGpuTelemetrySnapshot(
    {
      getDeviceLossStats: () => ({
        deviceLostCount: 2.7,
        deviceRecoveredCount: -1,
        lastDeviceLostReason: 'lost at file:///secret/path now',
        deviceRecovering: 'yes',
      }),
    },
    { timestamp: 1 }
  );
  assert.deepEqual(custom.deviceStats, {
    deviceLostCount: 2,
    deviceRecoveredCount: 0,
    lastDeviceLostReason: 'lost at <url> now',
    deviceRecovering: false,
  });
});

// ---------------------------------------------------------------------------
// Handler-level recovery
// ---------------------------------------------------------------------------

function createMockGpu(devices) {
  const requestDeviceCalls = [];
  const adapterCalls = [];
  let index = 0;
  const gpu = {
    async requestAdapter(opts) {
      adapterCalls.push(opts);
      return {
        features: new Set(['float32-filterable']),
        limits: { maxSamplersPerShaderStage: 32 },
        info: { vendor: 'mock' },
        async requestDevice(desc) {
          requestDeviceCalls.push(desc);
          const next = devices[Math.min(index, devices.length - 1)];
          index += 1;
          return next;
        },
      };
    },
    getPreferredCanvasFormat() {
      return 'bgra8unorm';
    },
  };
  return { gpu, requestDeviceCalls, adapterCalls };
}

function createMockCanvas() {
  const ctx = createMockContext();
  const getContextCalls = [];
  return {
    ctx,
    getContextCalls,
    canvas: {
      width: 0,
      height: 0,
      getContext(kind) {
        getContextCalls.push(kind);
        return ctx;
      },
    },
  };
}

test('W100 handler recovers a torn-down device end-to-end and resumes presenting', async () => {
  const first = createMockDevice('first');
  const second = createMockDevice('second');
  const { gpu, requestDeviceCalls } = createMockGpu([first.device, second.device]);
  const { canvas, ctx, getContextCalls } = createMockCanvas();
  const posted = [];
  const { logger, lines } = createLogger();
  const handler = createWgpuWorkerHandler({
    postMessage: (m) => posted.push(m),
    enableSwapchain: false,
    logger,
  });

  const rejections = await collectUnhandledRejections(() =>
    withMockNavigatorGpu(gpu, async () => {
      await handler.handleMessage({ init: true, canvas, width: 320, height: 180 });
      const processor = handler.processor;
      assert.ok(processor);
      assert.equal(processor.device, first.device);

      await handler.handleMessage({ cmdBuffer: framePackets(createBufferPacket(3, 0x28, 64)) });
      assert.equal(processor.frameCount, 1);
      assert.equal(processor.buffers.size, 1);

      // Chrome tears down the instance: the next present throws OperationError.
      first.device.broken = true;
      await handler.handleMessage({ cmdBuffer: framePackets() });
      assert.equal(processor.frameCount, 1);
      assert.equal(processor.isDeviceRecovering, true);

      // While recovering: submissions are dropped silently.
      const postedBefore = posted.length;
      await handler.handleMessage({ cmdBuffer: framePackets() });
      assert.equal(processor.frameCount, 1);
      assert.equal(
        posted.slice(postedBefore).filter((m) => m && m.error).length,
        0,
        'no error posted for dropped frames'
      );

      // device.lost also resolves; must not double-trigger a recovery.
      first.device.lose({ reason: 'unknown', message: 'instance lost' });
      await settle();

      assert.equal(processor.isDeviceRecovering, false);
      assert.equal(processor.device, second.device);
      assert.equal(processor.ctx, ctx);
      // W104: the live buffer is replayed onto the recovered device (stale handle dropped).
      assert.equal(processor.buffers.size, 1);
      assert.equal(processor.buffers.get(3).label, 'second');
      assert.ok(processor.extension, 'packet ABI v2 extensions reinstalled');

      await handler.handleMessage({ cmdBuffer: framePackets() });
      assert.equal(processor.frameCount, 2);
    })
  );

  assert.deepEqual(rejections, []);
  // Same feature / elevated-limit negotiation for init and recovery.
  assert.equal(requestDeviceCalls.length, 2);
  assert.deepEqual(requestDeviceCalls[1], requestDeviceCalls[0]);
  assert.deepEqual(requestDeviceCalls[0].requiredFeatures, ['float32-filterable']);
  assert.equal(getContextCalls.length, 2);
  const lastConfigure = ctx.configured[ctx.configured.length - 1];
  assert.deepEqual(lastConfigure, {
    device: second.device,
    format: 'bgra8unorm',
    alphaMode: 'opaque',
  });

  const telemetry = posted.filter((m) => m && m.type === 'gpu-telemetry');
  const recoveredSnap = telemetry.find((m) => m.deviceStats && m.deviceStats.deviceRecoveredCount === 1);
  assert.ok(recoveredSnap, 'telemetry emitted after recovery');
  assert.equal(recoveredSnap.deviceStats.deviceLostCount, 1);
  assert.equal(recoveredSnap.deviceStats.deviceRecovering, false);
  assert.match(recoveredSnap.deviceStats.lastDeviceLostReason, /OperationError/);
  assert.equal(lines.filter((l) => /device lost/.test(l.message)).length, 1);
  assert.equal(lines.filter((l) => /device recovered/.test(l.message)).length, 1);
});

test('W100 handler recovery is bounded to maxDeviceRecoveries and never rejects', async () => {
  const first = createMockDevice('first');
  const { gpu } = createMockGpu([first.device]);
  const { canvas } = createMockCanvas();
  const { logger, lines } = createLogger();
  let recoverCalls = 0;
  const handler = createWgpuWorkerHandler({
    postMessage: () => {},
    enableSwapchain: false,
    logger,
    deviceRecoveryMinIntervalMs: 0,
    recoverGpuDevice: async () => {
      recoverCalls += 1;
      throw instanceLostError();
    },
  });

  const rejections = await collectUnhandledRejections(() =>
    withMockNavigatorGpu(gpu, async () => {
      await handler.handleMessage({ init: true, canvas, width: 64, height: 64 });
      first.device.broken = true;
      for (let i = 0; i < 8; i += 1) {
        await handler.handleMessage({ cmdBuffer: framePackets() });
        await settle(2);
      }
    })
  );

  assert.deepEqual(rejections, []);
  assert.equal(recoverCalls, 3);
  const stats = handler.processor.getDeviceLossStats();
  assert.equal(stats.deviceLostCount, 1);
  assert.equal(stats.deviceRecoveredCount, 0);
  assert.equal(stats.deviceRecoveryFailedCount, 3);
  assert.equal(stats.deviceRecovering, false);
  assert.equal(handler.processor.frameCount, 0);
  assert.equal(lines.filter((l) => /recovery failed/.test(l.message)).length, 3);
  assert.equal(lines.filter((l) => /budget exhausted/.test(l.message)).length, 1);
});

test('W100 handler rate-limits recovery attempts by deviceRecoveryMinIntervalMs', async () => {
  const first = createMockDevice('first');
  const { gpu } = createMockGpu([first.device]);
  const { canvas } = createMockCanvas();
  let nowMs = 10_000;
  let recoverCalls = 0;
  const handler = createWgpuWorkerHandler({
    postMessage: () => {},
    enableSwapchain: false,
    logger: null,
    now: () => nowMs,
    deviceRecoveryMinIntervalMs: 1000,
    recoverGpuDevice: async () => {
      recoverCalls += 1;
      throw new Error('adapter unavailable');
    },
  });

  await withMockNavigatorGpu(gpu, async () => {
    await handler.handleMessage({ init: true, canvas, width: 64, height: 64 });
    first.device.broken = true;
    await handler.handleMessage({ cmdBuffer: framePackets() });
    await settle(2);
    assert.equal(recoverCalls, 1);

    nowMs += 10;
    await handler.handleMessage({ cmdBuffer: framePackets() });
    await settle(2);
    assert.equal(recoverCalls, 1, 'second attempt suppressed inside the interval');

    nowMs += 2000;
    await handler.handleMessage({ cmdBuffer: framePackets() });
    await settle(2);
    assert.equal(recoverCalls, 2);
  });
});

test('W100 handler ignores device loss after an intentional host destroy', async () => {
  const first = createMockDevice('first');
  const { gpu } = createMockGpu([first.device]);
  const { canvas } = createMockCanvas();
  let recoverCalls = 0;
  const posted = [];
  const handler = createWgpuWorkerHandler({
    postMessage: (m) => posted.push(m),
    enableSwapchain: false,
    logger: null,
    recoverGpuDevice: async () => {
      recoverCalls += 1;
      return { device: createMockDevice('x').device, ctx: null, format: 'bgra8unorm' };
    },
  });

  await withMockNavigatorGpu(gpu, async () => {
    await handler.handleMessage({ init: true, canvas, width: 64, height: 64 });
    await handler.handleMessage({ destroy: true, id: 1 });
    first.device.lose({ reason: 'unknown', message: 'late loss' });
    await settle();
  });

  assert.equal(recoverCalls, 0);
  assert.ok(posted.some((m) => m && m.destroyed === true));
});

test('W100 splash frames are rejected cleanly while recovery is in flight', async () => {
  const first = createMockDevice('first');
  const { gpu } = createMockGpu([first.device]);
  const { canvas } = createMockCanvas();
  let releaseRecovery;
  const gate = new Promise((resolve) => {
    releaseRecovery = resolve;
  });
  const second = createMockDevice('second');
  const posted = [];
  const handler = createWgpuWorkerHandler({
    postMessage: (m) => posted.push(m),
    enableSwapchain: false,
    logger: null,
    recoverGpuDevice: async () => {
      await gate;
      return { device: second.device, ctx: null, format: 'bgra8unorm' };
    },
  });

  await withMockNavigatorGpu(gpu, async () => {
    await handler.handleMessage({ init: true, canvas, width: 64, height: 64 });
    first.device.lose();
    await settle();
    assert.equal(handler.processor.isDeviceRecovering, true);

    await handler.handleMessage({
      type: 'present-splash-frame',
      id: 9,
      frame: { frameIndex: 4, width: 2, height: 2, pixels: new Uint8Array(16) },
    });
    const reply = posted.find((m) => m && m.type === 'splash-frame-presented' && m.id === 9);
    assert.ok(reply);
    assert.equal(reply.ok, false);
    assert.equal(reply.reason, 'device-recovering');
    assert.equal(reply.frameIndex, 4);

    releaseRecovery();
    await settle();
    assert.equal(handler.processor.isDeviceRecovering, false);
    assert.equal(handler.processor.device, second.device);
    assert.equal(handler.processor.getDeviceLossStats().deviceRecoveredCount, 1);
  });
});
