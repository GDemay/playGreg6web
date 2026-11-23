import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const swapchainMod = require('../web/workers/wgpu_swapchain.js');
const {
  DEFAULT_DEPTH_FORMAT,
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
});
