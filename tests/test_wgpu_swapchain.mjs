import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const swapchainMod = require('../web/workers/wgpu_swapchain.js');
const {
  DEFAULT_DEPTH_FORMAT,
  DEFAULT_MAX_ERROR_HISTORY,
  MAX_VALIDATION_ERROR_MESSAGE_LENGTH,
  SUPPORTED_DEPTH_FORMATS,
  TEXTURE_USAGE_RENDER_ATTACHMENT,
  WgpuSwapchain,
} = swapchainMod;

function createMockWebGpuEnv({ limits, withUnconfigure = true } = {}) {
  const events = [];
  let nextTextureId = 1;
  let nextSwapViewId = 1;

  function makeDevice(label = 'device-0', deviceLimits = limits) {
    let resolveLost;
    const lost = new Promise((resolve) => {
      resolveLost = resolve;
    });

    const createdTextures = [];
    const device = {
      label,
      limits: deviceLimits,
      lost,
      createdTextures,
      triggerLost(info = { reason: 'unknown', message: 'simulated GPU reset' }) {
        resolveLost(info);
      },
      createTexture(desc) {
        const id = nextTextureId++;
        const view = { kind: 'GPUTextureView', id: `depth-view-${id}`, textureId: id };
        const texture = {
          kind: 'GPUTexture',
          id,
          deviceLabel: label,
          desc: {
            ...desc,
            size: { ...desc.size },
          },
          view,
          destroyed: false,
          destroyCalls: 0,
          createView() {
            events.push({ op: 'depthTexture.createView', id, deviceLabel: label });
            return view;
          },
          destroy() {
            this.destroyed = true;
            this.destroyCalls += 1;
            events.push({ op: 'depthTexture.destroy', id, deviceLabel: label });
          },
        };
        createdTextures.push(texture);
        events.push({ op: 'device.createTexture', id, deviceLabel: label, desc: texture.desc });
        return texture;
      },
    };
    return device;
  }

  const context = {
    configurations: [],
    unconfigureCalls: 0,
    configure(cfg) {
      this.configurations.push({ ...cfg });
      events.push({ op: 'context.configure', cfg: { ...cfg } });
    },
    getCurrentTexture() {
      const view = { kind: 'GPUTextureView', id: `color-view-${nextSwapViewId++}` };
      events.push({ op: 'context.getCurrentTexture', viewId: view.id });
      return {
        createView() {
          return view;
        },
      };
    },
  };

  if (withUnconfigure) {
    context.unconfigure = function unconfigure() {
      this.unconfigureCalls += 1;
      events.push({ op: 'context.unconfigure' });
    };
  }

  const canvas = {
    width: 0,
    height: 0,
    getContext(kind) {
      return kind === 'webgpu' ? context : null;
    },
  };

  const device = makeDevice('device-0', limits);

  return { canvas, context, device, events, makeDevice };
}

test('UMD export attaches PlayGta6WgpuSwapchain on globalThis and exposes depth constants', () => {
  assert.equal(globalThis.PlayGta6WgpuSwapchain.WgpuSwapchain, WgpuSwapchain);
  assert.equal(DEFAULT_DEPTH_FORMAT, 'depth24plus');
  assert.equal(globalThis.PlayGta6WgpuSwapchain.DEFAULT_DEPTH_FORMAT, 'depth24plus');
  assert.ok(SUPPORTED_DEPTH_FORMATS.includes('depth24plus'));
  assert.ok(SUPPORTED_DEPTH_FORMATS.includes('depth32float'));
  assert.ok(SUPPORTED_DEPTH_FORMATS.includes('depth24plus-stencil8'));
  assert.ok(Object.isFrozen(SUPPORTED_DEPTH_FORMATS));
});

test('initial configure applies DPR scaling, clamps to [1, maxDimension], and lazily creates depth texture', () => {
  const { canvas, context, device } = createMockWebGpuEnv();
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    depthFormat: 'depth32float',
    alphaMode: 'opaque',
    maxDimension: 4096,
  });

  // Constructor must not allocate a depth texture before configure().
  assert.equal(device.createdTextures.length, 0);
  assert.equal(swapchain.isConfigured, false);

  // DPR scaling within limits: 1280x720 @ 2x -> 2560x1440.
  const cfg = swapchain.configure({ width: 1280, height: 720, devicePixelRatio: 2 });
  assert.deepEqual(cfg, {
    width: 2560,
    height: 1440,
    format: 'bgra8unorm',
    depthFormat: 'depth32float',
    devicePixelRatio: 2,
  });
  assert.equal(canvas.width, 2560);
  assert.equal(canvas.height, 1440);
  assert.equal(context.configurations.length, 1);
  assert.deepEqual(context.configurations[0], {
    device,
    format: 'bgra8unorm',
    alphaMode: 'opaque',
  });
  assert.equal(device.createdTextures.length, 1);
  assert.deepEqual(device.createdTextures[0].desc, {
    label: 'playgta6-swapchain-depth',
    size: { width: 2560, height: 1440, depthOrArrayLayers: 1 },
    format: 'depth32float',
    usage: TEXTURE_USAGE_RENDER_ATTACHMENT,
  });
  assert.equal(swapchain.getTelemetry().resizeCount, 0);

  // Clamping: oversized width clamped to maxDimension (4096), zero/negative height clamped to 1.
  swapchain.configure({ width: 5000, height: -20, devicePixelRatio: 2 });
  assert.equal(canvas.width, 4096);
  assert.equal(canvas.height, 1);
  assert.equal(device.createdTextures[0].destroyed, true);
  assert.equal(device.createdTextures.length, 2);
  assert.deepEqual(device.createdTextures[1].desc.size, {
    width: 4096,
    height: 1,
    depthOrArrayLayers: 1,
  });
  assert.equal(swapchain.getTelemetry().resizeCount, 1);
});

test('resize no-ops when pixel dimensions are unchanged and destroys old depth texture before creating new one on change', () => {
  const { canvas, context, device, events } = createMockWebGpuEnv();
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
  });

  swapchain.configure({ width: 640, height: 360, devicePixelRatio: 2 });
  assert.equal(canvas.width, 1280);
  assert.equal(canvas.height, 720);
  assert.equal(device.createdTextures.length, 1);

  const firstDepthTex = device.createdTextures[0];

  // Exact same CSS size + DPR -> no-op.
  assert.equal(swapchain.resize(640, 360, 2), false);
  // Different CSS size + DPR resolving to the exact same pixel dimensions -> still a no-op.
  assert.equal(swapchain.resize(1280, 720, 1), false);
  assert.equal(context.configurations.length, 1);
  assert.equal(device.createdTextures.length, 1);
  assert.equal(firstDepthTex.destroyed, false);
  assert.equal(swapchain.getTelemetry().resizeCount, 0);

  // Clear recorded events to assert ordering on actual resize.
  events.length = 0;

  assert.equal(swapchain.resize(960, 540, 2), true);
  assert.equal(canvas.width, 1920);
  assert.equal(canvas.height, 1080);
  assert.equal(context.configurations.length, 2);
  assert.equal(device.createdTextures.length, 2);
  assert.equal(firstDepthTex.destroyed, true);
  assert.equal(firstDepthTex.destroyCalls, 1);
  assert.equal(swapchain.getTelemetry().resizeCount, 1);

  // Verify previous depth texture was destroyed strictly BEFORE creating the replacement.
  const destroyIdx = events.findIndex((e) => e.op === 'depthTexture.destroy');
  const createIdx = events.findIndex((e) => e.op === 'device.createTexture');
  assert.ok(destroyIdx !== -1, 'expected old depth texture destroy event');
  assert.ok(createIdx !== -1, 'expected new depth texture create event');
  assert.ok(
    destroyIdx < createIdx,
    `expected depthTexture.destroy (${destroyIdx}) before device.createTexture (${createIdx})`
  );
});

test('acquireRenderPassDescriptor builds color-only, color+depth, and color+depth-stencil descriptors', () => {
  // 1. Color-only mode (enableDepth: false)
  {
    const { canvas, context, device } = createMockWebGpuEnv();
    const swapchain = new WgpuSwapchain({
      canvas,
      device,
      context,
      format: 'bgra8unorm',
      enableDepth: false,
    });
    swapchain.configure({ width: 800, height: 600 });

    assert.equal(device.createdTextures.length, 0);
    assert.equal(swapchain.depthFormat, null);
    assert.equal(swapchain.getTelemetry().depthFormat, null);

    const desc = swapchain.acquireRenderPassDescriptor({
      clearColor: { r: 0.1, g: 0.2, b: 0.3, a: 1.0 },
      loadOp: 'clear',
      storeOp: 'store',
    });

    assert.equal(desc.colorAttachments.length, 1);
    assert.equal(desc.colorAttachments[0].view.kind, 'GPUTextureView');
    assert.equal(desc.colorAttachments[0].loadOp, 'clear');
    assert.equal(desc.colorAttachments[0].storeOp, 'store');
    assert.deepEqual(desc.colorAttachments[0].clearValue, { r: 0.1, g: 0.2, b: 0.3, a: 1.0 });
    assert.equal('depthStencilAttachment' in desc, false);
  }

  // 2. Color + depth24plus (default)
  {
    const { canvas, context, device } = createMockWebGpuEnv();
    const swapchain = new WgpuSwapchain({
      canvas,
      device,
      context,
      format: 'bgra8unorm',
      enableDepth: true,
    });
    swapchain.configure({ width: 800, height: 600 });

    const desc = swapchain.acquireRenderPassDescriptor({
      clearColor: { r: 0.25, g: 0.5, b: 0.75, a: 1.0 },
      depthClearValue: 0.0,
      depthLoadOp: 'clear',
      depthStoreOp: 'store',
    });

    assert.equal(desc.colorAttachments.length, 1);
    assert.deepEqual(desc.colorAttachments[0].clearValue, { r: 0.25, g: 0.5, b: 0.75, a: 1.0 });
    assert.deepEqual(desc.depthStencilAttachment, {
      view: device.createdTextures[0].view,
      depthLoadOp: 'clear',
      depthStoreOp: 'store',
      depthClearValue: 0.0,
    });
  }

  // 3. Color + depth24plus-stencil8 includes stencil load/store/clear fields
  {
    const { canvas, context, device } = createMockWebGpuEnv();
    const swapchain = new WgpuSwapchain({
      canvas,
      device,
      context,
      format: 'bgra8unorm',
      depthFormat: 'depth24plus-stencil8',
    });
    swapchain.configure({ width: 640, height: 480 });

    const desc = swapchain.acquireRenderPassDescriptor({
      loadOp: 'load',
      depthLoadOp: 'load',
      depthStoreOp: 'store',
      stencilClearValue: 7,
    });

    assert.equal('clearValue' in desc.colorAttachments[0], false);
    assert.deepEqual(desc.depthStencilAttachment, {
      view: device.createdTextures[0].view,
      depthLoadOp: 'load',
      depthStoreOp: 'store',
      stencilLoadOp: 'clear',
      stencilStoreOp: 'store',
      stencilClearValue: 7,
    });
  }
});

test('rolling FPS and frame-time telemetry uses injectable clock and evicts old samples beyond window', () => {
  const { canvas, context, device } = createMockWebGpuEnv();
  let clockMs = 1000;
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    now: () => clockMs,
  });
  swapchain.configure({ width: 1280, height: 720 });

  // Before any frames.
  let t = swapchain.getTelemetry();
  assert.deepEqual(t, {
    fps: 0,
    avgFrameTimeMs: 0,
    maxFrameTimeMs: 0,
    totalFrames: 0,
    resizeCount: 0,
    width: 1280,
    height: 720,
    format: 'bgra8unorm',
    depthFormat: 'depth24plus',
    isDeviceLost: false,
  });

  // First presented frame establishes baseline timestamp (no delta yet).
  swapchain.recordFramePresented();
  t = swapchain.getTelemetry();
  assert.equal(t.totalFrames, 1);
  assert.equal(t.avgFrameTimeMs, 0);
  assert.equal(t.maxFrameTimeMs, 0);
  assert.equal(t.fps, 0);

  // Frame 2: +20ms, Frame 3: +50ms spike, Frame 4: +20ms -> avg = 30ms, max = 50ms, fps = 1000/30.
  clockMs += 20;
  swapchain.recordFramePresented();
  clockMs += 50;
  swapchain.recordFramePresented();
  clockMs += 20;
  swapchain.recordFramePresented();

  t = swapchain.getTelemetry();
  assert.equal(t.totalFrames, 4);
  assert.equal(t.avgFrameTimeMs, 30);
  assert.equal(t.maxFrameTimeMs, 50);
  assert.ok(Math.abs(t.fps - 1000 / 30) < 1e-9);

  // Present 60 steady 16ms frames so the earlier 50ms spike rolls completely out of the 60-frame window.
  for (let i = 0; i < 60; i += 1) {
    clockMs += 16;
    swapchain.recordFramePresented();
  }

  t = swapchain.getTelemetry();
  assert.equal(t.totalFrames, 64);
  assert.equal(t.avgFrameTimeMs, 16);
  assert.equal(t.maxFrameTimeMs, 16);
  assert.equal(t.fps, 62.5);
});

test('device.lost handler destroys stale depth texture and recover(newDevice) restores swapchain cleanly', async () => {
  const { canvas, context, device, makeDevice } = createMockWebGpuEnv();
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    depthFormat: 'depth24plus',
  });
  swapchain.configure({ width: 1280, height: 720, devicePixelRatio: 1 });

  const oldDepthTex = device.createdTextures[0];
  const lostEvents = [];
  swapchain.attachDeviceLostHandler((info, sc) => {
    lostEvents.push({ info, sameInstance: sc === swapchain });
  });

  // Trigger device loss and await promise microtask flush.
  device.triggerLost({ reason: 'unknown', message: 'GPU driver crash' });
  await device.lost;
  await Promise.resolve();

  assert.equal(swapchain.isDeviceLost, true);
  assert.equal(swapchain.getTelemetry().isDeviceLost, true);
  assert.equal(oldDepthTex.destroyed, true);
  assert.equal(swapchain.depthTexture, null);
  assert.equal(lostEvents.length, 1);
  assert.deepEqual(lostEvents[0], {
    info: { reason: 'unknown', message: 'GPU driver crash' },
    sameInstance: true,
  });

  // Operations while lost should reject cleanly.
  assert.throws(() => swapchain.acquireRenderPassDescriptor(), /lost/i);
  assert.throws(() => swapchain.resize(800, 600), /lost/i);

  // Recover with a new GPUDevice and updated canvas format.
  const newDevice = makeDevice('device-1');
  const recovered = swapchain.recover(newDevice, 'rgba8unorm');

  assert.deepEqual(recovered, {
    width: 1280,
    height: 720,
    format: 'rgba8unorm',
    depthFormat: 'depth24plus',
    isDeviceLost: false,
  });
  assert.equal(swapchain.device, newDevice);
  assert.equal(swapchain.canvas, canvas);
  assert.equal(swapchain.context, context);
  assert.equal(swapchain.isDeviceLost, false);
  assert.equal(swapchain.getTelemetry().resizeCount, 0);

  // Context was reconfigured with the new device and format.
  const lastConfig = context.configurations[context.configurations.length - 1];
  assert.deepEqual(lastConfig, {
    device: newDevice,
    format: 'rgba8unorm',
    alphaMode: 'opaque',
  });

  // Fresh depth texture created on the new device and usable in render pass descriptors.
  assert.equal(newDevice.createdTextures.length, 1);
  const newDepthTex = newDevice.createdTextures[0];
  assert.deepEqual(newDepthTex.desc.size, { width: 1280, height: 720, depthOrArrayLayers: 1 });

  const passDesc = swapchain.acquireRenderPassDescriptor();
  assert.equal(passDesc.depthStencilAttachment.view, newDepthTex.view);

  // Registered onLost callback remains attached to the recovered device too.
  newDevice.triggerLost({ reason: 'destroyed', message: 'second loss' });
  await newDevice.lost;
  await Promise.resolve();
  assert.equal(lostEvents.length, 2);
  assert.equal(lostEvents[1].info.message, 'second loss');
  assert.equal(newDepthTex.destroyed, true);
});

test('destroy() destroys depth texture, unconfigures context, clears refs, and is idempotent', () => {
  const { canvas, context, device } = createMockWebGpuEnv({ withUnconfigure: true });
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
  });
  swapchain.configure({ width: 640, height: 480 });

  const depthTex = device.createdTextures[0];
  assert.equal(depthTex.destroyed, false);
  assert.equal(context.unconfigureCalls, 0);

  swapchain.destroy();
  assert.equal(swapchain.isDestroyed, true);
  assert.equal(swapchain.isConfigured, false);
  assert.equal(depthTex.destroyed, true);
  assert.equal(depthTex.destroyCalls, 1);
  assert.equal(context.unconfigureCalls, 1);
  assert.equal(swapchain.depthTexture, null);
  assert.equal(swapchain.device, null);
  assert.equal(swapchain.context, null);
  assert.equal(swapchain.canvas, null);

  // Second destroy() must be a safe no-op (idempotent).
  swapchain.destroy();
  assert.equal(depthTex.destroyCalls, 1);
  assert.equal(context.unconfigureCalls, 1);

  // Post-destroy operations throw cleanly while getTelemetry remains safe.
  assert.throws(() => swapchain.configure({ width: 320, height: 240 }), /destroyed/i);
  assert.throws(() => swapchain.resize(320, 240), /destroyed/i);
  assert.throws(() => swapchain.acquireRenderPassDescriptor(), /destroyed/i);
  assert.doesNotThrow(() => swapchain.getTelemetry());

  // Also works safely when context does not implement unconfigure().
  const noUnconfigEnv = createMockWebGpuEnv({ withUnconfigure: false });
  const swapchain2 = new WgpuSwapchain({
    canvas: noUnconfigEnv.canvas,
    device: noUnconfigEnv.device,
    context: noUnconfigEnv.context,
    format: 'bgra8unorm',
  });
  swapchain2.configure({ width: 320, height: 240 });
  assert.doesNotThrow(() => swapchain2.destroy());
  assert.doesNotThrow(() => swapchain2.destroy());
  assert.equal(noUnconfigEnv.device.createdTextures[0].destroyed, true);
});

test('constructor validates invalid options cleanly', () => {
  const { canvas, context, device } = createMockWebGpuEnv();

  assert.throws(() => new WgpuSwapchain({ device, context }), TypeError);
  assert.throws(() => new WgpuSwapchain({ canvas, context }), TypeError);
  assert.throws(
    () => new WgpuSwapchain({ canvas, device, context, depthFormat: 'invalid-depth' }),
    RangeError
  );
  assert.throws(
    () => new WgpuSwapchain({ canvas, device, context, alphaMode: 'bogus' }),
    RangeError
  );
  assert.throws(
    () => new WgpuSwapchain({ canvas, device, context, maxDimension: 0 }),
    RangeError
  );
  assert.throws(
    () => new WgpuSwapchain({ canvas, device, context, onValidationError: 'not-a-fn' }),
    TypeError
  );
  assert.throws(
    () => new WgpuSwapchain({ canvas, device, context, maxErrorHistory: 0 }),
    RangeError
  );
  assert.throws(
    () => new WgpuSwapchain({ canvas, device, context, maxErrorHistory: -5 }),
    RangeError
  );
  assert.throws(
    () => new WgpuSwapchain({ canvas, device, context, maxErrorHistory: 2.5 }),
    RangeError
  );
});

function attachMockEventTarget(device, { withRemove = true } = {}) {
  const listeners = new Map();
  const addCalls = [];
  const removeCalls = [];

  device.addEventListener = function addEventListener(type, fn) {
    addCalls.push({ type, fn });
    let set = listeners.get(type);
    if (!set) {
      set = new Set();
      listeners.set(type, set);
    }
    set.add(fn);
  };

  if (withRemove) {
    device.removeEventListener = function removeEventListener(type, fn) {
      removeCalls.push({ type, fn });
      const set = listeners.get(type);
      if (set) {
        set.delete(fn);
      }
    };
  }

  device.dispatchUncapturedError = function dispatchUncapturedError(error, extra = {}) {
    const event = { type: 'uncapturederror', error, ...extra };
    const set = listeners.get('uncapturederror');
    if (set) {
      for (const fn of Array.from(set)) {
        fn(event);
      }
    }
    return event;
  };

  device.listenerCount = function listenerCount(type = 'uncapturederror') {
    const set = listeners.get(type);
    return set ? set.size : 0;
  };

  device.addCalls = addCalls;
  device.removeCalls = removeCalls;
  return device;
}

test('G35 uncapturederror listener attaches, captures normalized errors, updates getStats(), and invokes onValidationError safely', () => {
  assert.equal(DEFAULT_MAX_ERROR_HISTORY, 32);
  assert.equal(MAX_VALIDATION_ERROR_MESSAGE_LENGTH, 512);

  const { canvas, context, device } = createMockWebGpuEnv();
  attachMockEventTarget(device);

  let clockMs = 5000;
  const callbackRecords = [];

  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    now: () => clockMs,
    onValidationError(record) {
      callbackRecords.push({ ...record });
      // Mutate received object to prove internal state is isolated.
      record.message = 'MUTATED_BY_CALLBACK';
      if (record.seq === 2) {
        throw new Error('user callback failure should be isolated');
      }
    },
  });

  assert.equal(device.listenerCount('uncapturederror'), 1);
  assert.equal(device.addCalls.length, 1);
  assert.equal(device.addCalls[0].type, 'uncapturederror');

  let stats = swapchain.getStats();
  assert.equal(stats.validationErrorCount, 0);
  assert.equal(stats.validationErrorHistorySize, 0);
  assert.equal(stats.maxErrorHistory, 32);
  assert.equal(stats.isDestroyed, false);
  assert.deepEqual(swapchain.getValidationErrors(), []);

  class GPUValidationErrorMock {
    constructor(message) {
      this.message = message;
    }
  }
  Object.defineProperty(GPUValidationErrorMock, 'name', { value: 'GPUValidationError' });

  class GPUOutOfMemoryErrorMock {
    constructor(message) {
      this.message = message;
    }
  }
  Object.defineProperty(GPUOutOfMemoryErrorMock, 'name', { value: 'GPUOutOfMemoryError' });

  // 1. Validation error
  clockMs = 5010;
  device.dispatchUncapturedError(
    new GPUValidationErrorMock('BindGroupLayout mismatch on pipeline 0x123456789abcdef0')
  );

  // 2. Out-of-memory error (callback throws, must not disrupt capture)
  clockMs = 5025;
  assert.doesNotThrow(() => {
    device.dispatchUncapturedError(
      new GPUOutOfMemoryErrorMock('Allocation of 256MB texture failed')
    );
  });

  // 3. Internal error via explicit type
  clockMs = 5040;
  device.dispatchUncapturedError({ type: 'internal', message: 'Dawn internal backend error' });

  // 4. Unknown error object
  clockMs = 5055;
  device.dispatchUncapturedError({ message: 'Unclassified GPU failure' });

  stats = swapchain.getStats();
  assert.equal(stats.validationErrorCount, 4);
  assert.equal(stats.validationErrorHistorySize, 4);

  const history = swapchain.getValidationErrors();
  assert.deepEqual(history, [
    {
      seq: 1,
      type: 'validation',
      message: 'BindGroupLayout mismatch on pipeline 0x123456789abcdef0',
      timestampMs: 5010,
    },
    {
      seq: 2,
      type: 'out-of-memory',
      message: 'Allocation of 256MB texture failed',
      timestampMs: 5025,
    },
    {
      seq: 3,
      type: 'internal',
      message: 'Dawn internal backend error',
      timestampMs: 5040,
    },
    {
      seq: 4,
      type: 'unknown',
      message: 'Unclassified GPU failure',
      timestampMs: 5055,
    },
  ]);

  assert.equal(callbackRecords.length, 4);
  assert.equal(callbackRecords[0].seq, 1);
  assert.equal(callbackRecords[1].type, 'out-of-memory');

  // Mutating returned shallow copies must not corrupt internal history.
  history[0].message = 'CORRUPTED';
  history.push({ seq: 999, type: 'validation', message: 'fake', timestampMs: 0 });
  const freshHistory = swapchain.getValidationErrors();
  assert.equal(freshHistory.length, 4);
  assert.equal(
    freshHistory[0].message,
    'BindGroupLayout mismatch on pipeline 0x123456789abcdef0'
  );
});

test('G35 ring evicts oldest entries at maxErrorHistory and clearValidationErrors() resets ring while keeping monotonic seq/count', () => {
  const { canvas, context, device } = createMockWebGpuEnv();
  attachMockEventTarget(device);

  let clockMs = 100;
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    now: () => clockMs,
    maxErrorHistory: 3,
  });

  for (let i = 1; i <= 5; i += 1) {
    clockMs = 100 + i * 10;
    device.dispatchUncapturedError({
      name: 'GPUValidationError',
      message: `validation error #${i}`,
    });
  }

  assert.equal(swapchain.getStats().validationErrorCount, 5);
  assert.equal(swapchain.getStats().validationErrorHistorySize, 3);
  assert.equal(swapchain.getStats().maxErrorHistory, 3);

  const retained = swapchain.getValidationErrors();
  assert.deepEqual(
    retained.map((r) => r.seq),
    [3, 4, 5]
  );
  assert.deepEqual(
    retained.map((r) => r.message),
    ['validation error #3', 'validation error #4', 'validation error #5']
  );

  // Clear history ring; cumulative count stays intact and next seq continues monotonically.
  swapchain.clearValidationErrors();
  assert.deepEqual(swapchain.getValidationErrors(), []);
  assert.equal(swapchain.getStats().validationErrorHistorySize, 0);
  assert.equal(swapchain.getStats().validationErrorCount, 5);

  clockMs = 200;
  device.dispatchUncapturedError({
    name: 'GPUValidationError',
    message: 'post-clear error',
  });
  const afterClear = swapchain.getValidationErrors();
  assert.equal(afterClear.length, 1);
  assert.deepEqual(afterClear[0], {
    seq: 6,
    type: 'validation',
    message: 'post-clear error',
    timestampMs: 200,
  });
  assert.equal(swapchain.getStats().validationErrorCount, 6);
});

test('G35 sanitizes host paths, long buffers, control characters, and caps message length to 512 chars', () => {
  const { canvas, context, device } = createMockWebGpuEnv();
  attachMockEventTarget(device);

  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    now: () => 42,
  });

  // 1. Host paths (Windows drive, quoted path with spaces, UNC, file://, Unix paths)
  device.dispatchUncapturedError({
    name: 'GPUValidationError',
    message:
      'Shader failed at C:\\Users\\alice\\local-data\\gta6\\shader.wgsl and "D:\\My Games\\GTA6\\x.rpf" and \\\\host01\\share\\dump.bin and file:///Users/alice/secret/a.wgsl and /home/alice/dev/pipeline.wgsl (format rgba8unorm/bgra8unorm)',
  });

  // 2. Long buffers (hex dump, byte array, base64 data URI, raw base64)
  const hexBlob = 'ab'.repeat(40);
  const spacedHex = Array.from({ length: 24 }, (_, i) => i.toString(16).padStart(2, '0')).join(' ');
  const byteArray = Array.from({ length: 24 }, (_, i) => String(i)).join(', ');
  const b64Blob = 'QUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFB';
  device.dispatchUncapturedError({
    name: 'GPUValidationError',
    message: `Buffer mismatch:\nhex=${hexBlob}\tbytes=[${byteArray}] dump=${spacedHex} uri=data:application/octet-stream;base64,${b64Blob} raw=${b64Blob}\x00\x07`,
  });

  // 3. Overly long message capped strictly to 512 characters
  const longText = `Validation failure: ${'texture_binding_mismatch '.repeat(40)}`;
  assert.ok(longText.length > 512);
  device.dispatchUncapturedError({
    name: 'GPUValidationError',
    message: longText,
  });

  const errors = swapchain.getValidationErrors();
  assert.equal(errors.length, 3);

  // Verify host paths stripped while preserving WebGPU format slash token.
  const pathMsg = errors[0].message;
  assert.doesNotMatch(pathMsg, /alice/i);
  assert.doesNotMatch(pathMsg, /C:\\/);
  assert.doesNotMatch(pathMsg, /D:\\/);
  assert.doesNotMatch(pathMsg, /\\\\host01/);
  assert.doesNotMatch(pathMsg, /file:\/\//);
  assert.doesNotMatch(pathMsg, /\/home\//);
  assert.match(pathMsg, /<path>/);
  assert.match(pathMsg, /rgba8unorm\/bgra8unorm/);

  // Verify long buffers & control chars stripped.
  const bufMsg = errors[1].message;
  assert.doesNotMatch(bufMsg, new RegExp(hexBlob));
  assert.doesNotMatch(bufMsg, new RegExp(b64Blob));
  assert.doesNotMatch(bufMsg, /[\x00-\x1F\x7F]/);
  assert.match(bufMsg, /<buffer>/);

  // Verify 512-char cap.
  const cappedMsg = errors[2].message;
  assert.equal(cappedMsg.length, MAX_VALIDATION_ERROR_MESSAGE_LENGTH);
  assert.ok(cappedMsg.endsWith('...'));
});

test('G35 removes uncapturederror listener cleanly on recover(newDevice) and destroy() with zero leaks', () => {
  const { canvas, context, device, makeDevice } = createMockWebGpuEnv();
  attachMockEventTarget(device);

  const seen = [];
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    onValidationError: (rec) => seen.push(rec),
  });
  swapchain.configure({ width: 640, height: 360 });

  assert.equal(device.listenerCount('uncapturederror'), 1);
  const firstRegisteredFn = device.addCalls[0].fn;

  // Re-bind to a new device via recover(): old listener must be removed, new one attached.
  const newDevice = attachMockEventTarget(makeDevice('device-1'));
  swapchain.recover(newDevice);

  assert.equal(device.listenerCount('uncapturederror'), 0);
  assert.equal(device.removeCalls.length, 1);
  assert.equal(device.removeCalls[0].type, 'uncapturederror');
  assert.equal(device.removeCalls[0].fn, firstRegisteredFn);

  assert.equal(newDevice.listenerCount('uncapturederror'), 1);
  const secondRegisteredFn = newDevice.addCalls[0].fn;

  // Dispatching on old device does nothing; dispatching on new device is captured.
  device.dispatchUncapturedError({ name: 'GPUValidationError', message: 'stale device error' });
  assert.equal(swapchain.getStats().validationErrorCount, 0);

  newDevice.dispatchUncapturedError({ name: 'GPUValidationError', message: 'recovered device error' });
  assert.equal(swapchain.getStats().validationErrorCount, 1);
  assert.equal(seen.length, 1);
  assert.equal(seen[0].message, 'recovered device error');

  // Destroy removes listener from active device and is idempotent.
  swapchain.destroy();
  assert.equal(newDevice.listenerCount('uncapturederror'), 0);
  assert.equal(newDevice.removeCalls.length, 1);
  assert.equal(newDevice.removeCalls[0].type, 'uncapturederror');
  assert.equal(newDevice.removeCalls[0].fn, secondRegisteredFn);
  assert.equal(swapchain.getStats().isDestroyed, true);

  // Calling destroy again does not double-remove.
  swapchain.destroy();
  assert.equal(newDevice.removeCalls.length, 1);

  // Events fired after destroy are ignored; previously captured history remains inspectable.
  newDevice.dispatchUncapturedError({ name: 'GPUValidationError', message: 'post-destroy error' });
  assert.equal(swapchain.getStats().validationErrorCount, 1);
  assert.equal(swapchain.getValidationErrors().length, 1);

  // Also test device without removeEventListener: stale handler still guards on destroyed/device mismatch.
  const env2 = createMockWebGpuEnv();
  attachMockEventTarget(env2.device, { withRemove: false });
  const sc2 = new WgpuSwapchain({
    canvas: env2.canvas,
    device: env2.device,
    context: env2.context,
    format: 'bgra8unorm',
  });
  sc2.destroy();
  env2.device.dispatchUncapturedError({ name: 'GPUValidationError', message: 'ignored after destroy' });
  assert.equal(sc2.getStats().validationErrorCount, 0);
});
