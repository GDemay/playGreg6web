import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const WORKERS_DIR = fileURLToPath(new URL('../web/workers/', import.meta.url));

const wgpuMod = require('../web/workers/wgpu_worker.js');
const binkMod = require('../web/workers/bink_presenter.js');
const bridgeMod = require('../web/workers/splash_frame_bridge.js');

const { WgpuCommandProcessor, createWgpuWorkerHandler } = wgpuMod;
const { BinkVideoPresenter, BINK_PIXEL_FORMAT } = binkMod;
const { SplashFrameBridge, SPLASH_FRAME_SOURCE } = bridgeMod;

function createMockWebGpuEnv(options = {}) {
  const {
    swapWidth = 1280,
    swapHeight = 720,
    supportCopyExternalImageToTexture = true,
  } = options;

  let nextId = 1;
  const createdTextures = [];
  const createdBuffers = [];
  const createdPipelines = [];
  const writeTextureCalls = [];
  const writeBufferCalls = [];
  const copyExternalImageCalls = [];
  const renderPasses = [];
  const submitCalls = [];

  const queue = {
    writeTexture(dest, data, dataLayout, size) {
      const bytes =
        data instanceof Uint8Array
          ? new Uint8Array(data)
          : new Uint8Array(data.buffer, data.byteOffset || 0, data.byteLength);
      writeTextureCalls.push({ dest, data: bytes, dataLayout: { ...dataLayout }, size: { ...size } });
    },
    writeBuffer(buffer, offset, data) {
      const bytes =
        data instanceof Uint8Array
          ? new Uint8Array(data)
          : ArrayBuffer.isView(data)
            ? new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength))
            : new Uint8Array(data.slice(0));
      writeBufferCalls.push({ buffer, offset, data: bytes });
    },
    submit(cmdBuffers) {
      submitCalls.push(cmdBuffers);
    },
    onSubmittedWorkDone() {
      return Promise.resolve();
    },
  };

  if (supportCopyExternalImageToTexture) {
    queue.copyExternalImageToTexture = (source, dest, size) => {
      copyExternalImageCalls.push({ source, dest, size });
    };
  }

  const device = {
    features: new Set(),
    queue,
    createTexture(desc) {
      const id = nextId++;
      const view = { kind: 'GPUTextureView', id: `view-${id}`, label: desc.label };
      const tex = {
        kind: 'GPUTexture',
        id,
        desc,
        width: desc.size.width,
        height: desc.size.height,
        destroyed: false,
        destroyCount: 0,
        createView() {
          return view;
        },
        destroy() {
          this.destroyed = true;
          this.destroyCount += 1;
        },
      };
      createdTextures.push(tex);
      return tex;
    },
    createBuffer(desc) {
      const id = nextId++;
      const buf = {
        kind: 'GPUBuffer',
        id,
        size: desc.size,
        usage: desc.usage,
        destroyed: false,
        destroyCount: 0,
        destroy() {
          this.destroyed = true;
          this.destroyCount += 1;
        },
      };
      createdBuffers.push(buf);
      return buf;
    },
    createShaderModule(desc) {
      return { kind: 'GPUShaderModule', id: nextId++, code: desc.code };
    },
    createBindGroupLayout(desc) {
      return { kind: 'GPUBindGroupLayout', id: nextId++, desc };
    },
    createPipelineLayout(desc) {
      return { kind: 'GPUPipelineLayout', id: nextId++, desc };
    },
    createRenderPipeline(desc) {
      const pipe = { kind: 'GPURenderPipeline', id: nextId++, desc };
      createdPipelines.push(pipe);
      return pipe;
    },
    createSampler(desc) {
      return { kind: 'GPUSampler', id: nextId++, desc };
    },
    createBindGroup(desc) {
      return { kind: 'GPUBindGroup', id: nextId++, desc };
    },
    createCommandEncoder(desc = {}) {
      return {
        desc,
        beginRenderPass(passDesc) {
          const rec = {
            passDesc,
            pipeline: null,
            viewport: null,
            scissor: null,
            bindGroups: [],
            draws: [],
            ended: false,
          };
          renderPasses.push(rec);
          return {
            setPipeline(p) {
              rec.pipeline = p;
            },
            setViewport(x, y, w, h, minD, maxD) {
              rec.viewport = { x, y, w, h, minD, maxD };
            },
            setScissorRect(x, y, w, h) {
              rec.scissor = { x, y, w, h };
            },
            setBindGroup(idx, bg) {
              rec.bindGroups.push({ idx, bg });
            },
            draw(vc, ic, fv, fi) {
              rec.draws.push({ vc, ic, fv, fi });
            },
            end() {
              rec.ended = true;
            },
          };
        },
        finish() {
          return { kind: 'GPUCommandBuffer', id: nextId++ };
        },
      };
    },
  };

  const ctx = {
    configured: null,
    canvas: { width: swapWidth, height: swapHeight },
    configure(opts) {
      this.configured = opts;
    },
    getCurrentTexture() {
      const view = { kind: 'GPUTextureView', id: `swap-${nextId++}` };
      return {
        width: swapWidth,
        height: swapHeight,
        createView() {
          return view;
        },
      };
    },
  };

  const canvas = {
    width: swapWidth,
    height: swapHeight,
    getContext(kind) {
      assert.equal(kind, 'webgpu');
      return ctx;
    },
  };

  const adapter = {
    info: { vendor: 'mock-gpu', architecture: 'bink-bridge-test' },
    async requestDevice() {
      return device;
    },
  };

  const navigator = {
    gpu: {
      async requestAdapter() {
        return adapter;
      },
      getPreferredCanvasFormat() {
        return 'bgra8unorm';
      },
    },
  };

  return {
    device,
    ctx,
    canvas,
    adapter,
    navigator,
    createdTextures,
    createdBuffers,
    createdPipelines,
    writeTextureCalls,
    writeBufferCalls,
    copyExternalImageCalls,
    renderPasses,
    submitCalls,
  };
}

function makeRgbaBytes(width, height, fill = 0x42) {
  const buf = new Uint8Array(width * height * 4);
  buf.fill(fill);
  for (let i = 3; i < buf.length; i += 4) {
    buf[i] = 255;
  }
  return buf;
}

function makeYuv420Planes(width, height, withAlpha = false) {
  const uvW = Math.ceil(width / 2);
  const uvH = Math.ceil(height / 2);
  const planes = {
    y: new Uint8Array(width * height).fill(200),
    u: new Uint8Array(uvW * uvH).fill(128),
    v: new Uint8Array(uvW * uvH).fill(128),
  };
  if (withAlpha) {
    planes.a = new Uint8Array(width * height).fill(255);
  }
  return planes;
}

// ---------------------------------------------------------------------------
// 1. WgpuCommandProcessor.presentSplashFrame: RGBA8, YUV420p, YUVA420p, ImageBitmap, diagnostic
// ---------------------------------------------------------------------------

test('W67 WgpuCommandProcessor.presentSplashFrame delegates RGBA8, YUV420p, YUVA420p, ImageBitmap, and diagnostic frames to SplashFrameBridge -> BinkVideoPresenter', () => {
  const mock = createMockWebGpuEnv({ swapWidth: 1280, swapHeight: 720 });
  const firstPresentEvents = [];
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    width: 1280,
    height: 720,
    onFirstPresent: (info) => firstPresentEvents.push(info),
  });

  // 1) RGBA8 frame (canonical planes shape) triggers Frame 1 + firstPresent callback once.
  const rgbaRes = processor.presentSplashFrame({
    width: 16,
    height: 9,
    format: 'rgba8',
    planes: { rgba: makeRgbaBytes(16, 9, 0x33) },
    frameIndex: 0,
  });

  assert.equal(rgbaRes.ok, true);
  assert.equal(rgbaRes.presented, true);
  assert.equal(rgbaRes.frameIndex, 0);
  assert.equal(rgbaRes.framesPresented, 1);
  assert.equal(rgbaRes.firstPresent, true);
  assert.equal(processor.frameCount, 1);
  assert.equal(processor.framesPresented, 1);
  assert.equal(firstPresentEvents.length, 1);
  assert.equal(mock.submitCalls.length, 1);
  assert.equal(mock.renderPasses.length, 1);
  assert.equal(mock.renderPasses[0].draws.length, 1);
  assert.deepEqual(mock.renderPasses[0].draws[0], { vc: 3, ic: 1, fv: 0, fi: 0 });
  assert.equal(processor.binkPresenter.pixelFormat, BINK_PIXEL_FORMAT.RGBA8);

  // 2) YUV420p frame (flat y/u/v shape) increments framesPresented cleanly, no duplicate firstPresent.
  const yuvPlanes = makeYuv420Planes(32, 18);
  const yuvRes = processor.presentSplashFrame({
    width: 32,
    height: 18,
    format: 'yuv420p',
    y: yuvPlanes.y,
    u: yuvPlanes.u,
    v: yuvPlanes.v,
    colorSpace: 'bt601',
    frameIndex: 1,
  });

  assert.equal(yuvRes.ok, true);
  assert.equal(yuvRes.presented, true);
  assert.equal(yuvRes.frameIndex, 1);
  assert.equal(yuvRes.framesPresented, 2);
  assert.equal(yuvRes.firstPresent, false);
  assert.equal(firstPresentEvents.length, 1);
  assert.equal(processor.binkPresenter.pixelFormat, BINK_PIXEL_FORMAT.YUV420P);
  assert.equal(processor.binkPresenter.colorSpace, 'bt601');

  // 3) YUVA420p frame (planes with alpha)
  const yuvaPlanes = makeYuv420Planes(16, 8, true);
  const yuvaRes = processor.presentSplashFrame({
    width: 16,
    height: 8,
    format: 'yuva420p',
    planes: yuvaPlanes,
    frameIndex: 2,
  });
  assert.equal(yuvaRes.ok, true);
  assert.equal(yuvaRes.framesPresented, 3);
  assert.equal(processor.binkPresenter.pixelFormat, BINK_PIXEL_FORMAT.YUVA420P);

  // 4) External ImageBitmap-like frame
  let bitmapClosed = false;
  const fakeBitmap = {
    width: 8,
    height: 8,
    data: makeRgbaBytes(8, 8, 0xaa),
    close() {
      bitmapClosed = true;
    },
  };
  const extRes = processor.presentSplashFrame({
    image: fakeBitmap,
    frameIndex: 3,
  });
  assert.equal(extRes.ok, true);
  assert.equal(extRes.framesPresented, 4);
  assert.equal(extRes.source, SPLASH_FRAME_SOURCE.EXTERNAL);
  assert.equal(mock.copyExternalImageCalls.length, 1);
  assert.equal(bitmapClosed, false); // extracted RGBA owns copy

  // 5) Diagnostic procedural frame
  const diagRes = processor.presentSplashFrame({
    diagnostic: true,
    width: 16,
    height: 12,
    frameIndex: 4,
  });
  assert.equal(diagRes.ok, true);
  assert.equal(diagRes.framesPresented, 5);
  assert.equal(diagRes.source, SPLASH_FRAME_SOURCE.DIAGNOSTIC);
  assert.equal(mock.submitCalls.length, 5);

  // Check getStats() exposes splashBridge + binkPresenter summary counters
  const stats = processor.getStats();
  assert.equal(stats.frameCount, 5);
  assert.equal(stats.framesPresented, 5);
  assert.equal(stats.splashFramesPresented, 5);
  assert.equal(stats.firstPresent, true);
  assert.ok(stats.splashBridge);
  assert.equal(stats.splashBridge.framesPresented, 5);
  assert.equal(stats.splashBridge.firstPresentSent, true);
  assert.equal(stats.splashBridge.lastSource, SPLASH_FRAME_SOURCE.DIAGNOSTIC);
  assert.ok(stats.binkPresenter);
  assert.equal(stats.binkPresenter.presentedFrames, 5);
  assert.equal(stats.binkPresenter.uploadedFrames, 5);
  assert.equal(stats.binkPresenter.destroyed, false);

  processor.destroy();
});

// ---------------------------------------------------------------------------
// 2. Constructor & instance injection ({ binkPresenter, splashFrameBridge, BinkVideoPresenter, SplashFrameBridge })
// ---------------------------------------------------------------------------

test('W67 supports injected binkPresenter, splashFrameBridge, and custom constructors', () => {
  // Case A: injected BinkVideoPresenter + SplashFrameBridge instances
  const mockA = createMockWebGpuEnv();
  const injectedPresenter = new BinkVideoPresenter({ device: mockA.device, targetFps: 60 });
  const injectedBridge = new SplashFrameBridge({ presenter: injectedPresenter });
  const procA = new WgpuCommandProcessor(mockA.device, mockA.ctx, {
    binkPresenter: injectedPresenter,
    splashFrameBridge: injectedBridge,
  });

  assert.equal(procA.binkPresenter, injectedPresenter);
  assert.equal(procA.splashFrameBridge, injectedBridge);

  const resA = procA.presentSplashFrame({
    width: 8,
    height: 8,
    rgba: makeRgbaBytes(8, 8, 0x77),
  });
  assert.equal(resA.ok, true);
  assert.equal(resA.framesPresented, 1);
  assert.equal(injectedPresenter.getStats().presentedFrames, 1);
  assert.equal(injectedBridge.getStats().framesPresented, 1);
  procA.destroy();

  // Case B: injected custom constructors with eager enableSplashBridge: true
  const mockB = createMockWebGpuEnv();
  let binkCtorArgs = null;
  let bridgeCtorArgs = null;
  class SpyBinkPresenter extends BinkVideoPresenter {
    constructor(opts) {
      super(opts);
      binkCtorArgs = opts;
    }
  }
  class SpySplashBridge extends SplashFrameBridge {
    constructor(opts) {
      super(opts);
      bridgeCtorArgs = opts;
    }
  }
  const procB = new WgpuCommandProcessor(mockB.device, mockB.ctx, {
    enableSplashBridge: true,
    BinkVideoPresenter: SpyBinkPresenter,
    SplashFrameBridge: SpySplashBridge,
  });

  assert.ok(procB.binkPresenter instanceof SpyBinkPresenter);
  assert.ok(procB.splashFrameBridge instanceof SpySplashBridge);
  assert.equal(binkCtorArgs.device, mockB.device);
  assert.equal(bridgeCtorArgs.presenter, procB.binkPresenter);
  procB.destroy();

  // Case C: direct binkPresenter fallback when SplashFrameBridge is explicitly disabled (null)
  const mockC = createMockWebGpuEnv();
  const directPresenter = new BinkVideoPresenter({ device: mockC.device });
  const procC = new WgpuCommandProcessor(mockC.device, mockC.ctx, {
    binkPresenter: directPresenter,
    SplashFrameBridge: null,
  });
  const resC = procC.presentSplashFrame({
    width: 8,
    height: 4,
    format: 'yuv420p',
    planes: makeYuv420Planes(8, 4),
    frameIndex: 7,
  });
  assert.equal(resC.ok, true);
  assert.equal(resC.frameIndex, 7);
  assert.equal(resC.framesPresented, 1);
  assert.equal(resC.firstPresent, true);
  assert.equal(procC.splashFrameBridge, null);
  assert.equal(procC.getStats().binkPresenter.presentedFrames, 1);
  procC.destroy();
});

// ---------------------------------------------------------------------------
// 3. Worker handler message routing: { type: 'present-splash-frame' | 'bink-frame' } + Frame-1 telemetry
// ---------------------------------------------------------------------------

test('W67 createWgpuWorkerHandler routes present-splash-frame & bink-frame and emits Frame-1 gpu-telemetry + first-present', async () => {
  const mock = createMockWebGpuEnv();
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, { width: 1280, height: 720 });
  const posted = [];
  const firstPresentCallbacks = [];
  let fakeNow = 1000;

  const handler = createWgpuWorkerHandler({
    processor,
    postMessage: (msg) => posted.push(msg),
    now: () => fakeNow,
    onFirstPresent: (info) => firstPresentCallbacks.push(info),
  });

  // Frame 1 via { type: 'present-splash-frame', frame }
  await handler.handleMessage({
    type: 'present-splash-frame',
    id: 'splash-1',
    frame: {
      width: 16,
      height: 8,
      format: 'rgba8',
      planes: { rgba: makeRgbaBytes(16, 8, 0x55) },
      frameIndex: 0,
    },
  });

  const presented1 = posted.find((m) => m.type === 'splash-frame-presented' && m.id === 'splash-1');
  assert.deepEqual(presented1, {
    type: 'splash-frame-presented',
    id: 'splash-1',
    frameIndex: 0,
    framesPresented: 1,
    ok: true,
    firstPresent: true,
  });

  const firstPresentMsgs = posted.filter((m) => m.type === 'first-present');
  assert.equal(firstPresentMsgs.length, 1);
  assert.equal(firstPresentMsgs[0].firstPresent, true);
  assert.equal(firstPresentMsgs[0].framesPresented, 1);
  assert.equal(firstPresentMsgs[0].frameIndex, 0);
  assert.equal(firstPresentCallbacks.length, 1);

  const telemetryAfterFrame1 = posted.filter((m) => m.type === 'gpu-telemetry');
  assert.equal(telemetryAfterFrame1.length, 1);
  assert.equal(telemetryAfterFrame1[0].frameStats.framesPresented, 1);
  assert.equal(telemetryAfterFrame1[0].timestamp, 1000);

  // Frame 2 via { type: 'bink-frame', frame } within throttle window (1016ms)
  fakeNow = 1016;
  await handler.handleMessage({
    type: 'bink-frame',
    id: 'bink-2',
    frame: {
      width: 16,
      height: 8,
      format: 'yuv420p',
      planes: makeYuv420Planes(16, 8),
      frameIndex: 1,
    },
  });

  const presented2 = posted.find((m) => m.type === 'splash-frame-presented' && m.id === 'bink-2');
  assert.deepEqual(presented2, {
    type: 'splash-frame-presented',
    id: 'bink-2',
    frameIndex: 1,
    framesPresented: 2,
    ok: true,
  });
  // No duplicate first-present; telemetry throttled within 250ms window.
  assert.equal(posted.filter((m) => m.type === 'first-present').length, 1);
  assert.equal(posted.filter((m) => m.type === 'gpu-telemetry').length, 1);

  // Frame 3 via flat { type: 'bink-frame', width, height, y, u, v } after throttle window (1300ms)
  fakeNow = 1300;
  const flatYuv = makeYuv420Planes(16, 8);
  await handler.handleMessage({
    type: 'bink-frame',
    id: 'bink-3',
    width: 16,
    height: 8,
    format: 'yuv420p',
    y: flatYuv.y,
    u: flatYuv.u,
    v: flatYuv.v,
    frameIndex: 2,
  });

  const presented3 = posted.find((m) => m.type === 'splash-frame-presented' && m.id === 'bink-3');
  assert.equal(presented3.ok, true);
  assert.equal(presented3.frameIndex, 2);
  assert.equal(presented3.framesPresented, 3);

  const telemetryAfterFrame3 = posted.filter((m) => m.type === 'gpu-telemetry');
  assert.equal(telemetryAfterFrame3.length, 2);
  assert.equal(telemetryAfterFrame3[1].frameStats.framesPresented, 3);
  assert.equal(telemetryAfterFrame3[1].timestamp, 1300);

  processor.destroy();
});

// ---------------------------------------------------------------------------
// 4. Graceful error handling when device/canvas not ready or presenter is absent
// ---------------------------------------------------------------------------

test('W67 gracefully reports errors without throwing when device/canvas is not ready or presenter is absent', async () => {
  // A) Worker handler before init (no processor/device/canvas)
  const postedPreInit = [];
  const preInitHandler = createWgpuWorkerHandler({
    postMessage: (m) => postedPreInit.push(m),
  });

  await assert.doesNotReject(() =>
    preInitHandler.handleMessage({
      type: 'present-splash-frame',
      id: 'pre-1',
      frame: { width: 8, height: 8, format: 'rgba8', planes: { rgba: makeRgbaBytes(8, 8) }, frameIndex: 4 },
    })
  );
  await assert.doesNotReject(() =>
    preInitHandler.handleMessage({
      type: 'bink-frame',
      id: 'pre-2',
      frame: { width: 8, height: 8, format: 'yuv420p', planes: makeYuv420Planes(8, 8) },
    })
  );

  assert.equal(postedPreInit.length, 2);
  assert.equal(postedPreInit[0].type, 'splash-frame-presented');
  assert.equal(postedPreInit[0].id, 'pre-1');
  assert.equal(postedPreInit[0].frameIndex, 4);
  assert.equal(postedPreInit[0].framesPresented, 0);
  assert.equal(postedPreInit[0].ok, false);
  assert.match(postedPreInit[0].error, /not initialized/i);
  assert.equal(postedPreInit[1].type, 'splash-frame-presented');
  assert.equal(postedPreInit[1].ok, false);
  assert.equal(postedPreInit.filter((m) => m.type === 'gpu-telemetry').length, 0);
  assert.equal(postedPreInit.filter((m) => m.type === 'first-present').length, 0);

  // B) Processor with null device or null ctx
  const noDeviceProc = new WgpuCommandProcessor(null, null);
  const noDevRes = noDeviceProc.presentSplashFrame({
    width: 8,
    height: 8,
    rgba: makeRgbaBytes(8, 8),
  });
  assert.equal(noDevRes.ok, false);
  assert.equal(noDevRes.presented, false);
  assert.equal(noDevRes.reason, 'not-ready');
  assert.equal(noDevRes.framesPresented, 0);

  // C) Presenter absent (BinkVideoPresenter: null, SplashFrameBridge: null)
  const mock = createMockWebGpuEnv();
  const absentProc = new WgpuCommandProcessor(mock.device, mock.ctx, {
    BinkVideoPresenter: null,
    SplashFrameBridge: null,
  });
  const absentRes = absentProc.presentSplashFrame({
    width: 8,
    height: 8,
    format: 'rgba8',
    planes: { rgba: makeRgbaBytes(8, 8) },
    frameIndex: 0,
  });
  assert.equal(absentRes.ok, false);
  assert.equal(absentRes.presented, false);
  assert.equal(absentRes.reason, 'presenter-unavailable');
  assert.equal(absentRes.framesPresented, 0);
  assert.equal(absentProc.getStats().splashBridge, null);
  assert.equal(absentProc.getStats().binkPresenter, null);

  // D) Worker handler routing when enableSplashBridge: false
  const disabledProc = new WgpuCommandProcessor(mock.device, mock.ctx, {
    enableSplashBridge: false,
  });
  const postedDisabled = [];
  const disabledHandler = createWgpuWorkerHandler({
    processor: disabledProc,
    postMessage: (m) => postedDisabled.push(m),
  });
  await disabledHandler.handleMessage({
    type: 'present-splash-frame',
    id: 'disabled-1',
    frame: { width: 8, height: 8, rgba: makeRgbaBytes(8, 8) },
  });
  assert.equal(postedDisabled.length, 1);
  assert.equal(postedDisabled[0].type, 'splash-frame-presented');
  assert.equal(postedDisabled[0].ok, false);
  assert.equal(postedDisabled[0].reason, 'presenter-unavailable');
  assert.equal(postedDisabled[0].framesPresented, 0);

  // E) Invalid frame payload returns error cleanly without throwing
  const validProc = new WgpuCommandProcessor(mock.device, mock.ctx);
  const badRes = validProc.presentSplashFrame(null);
  assert.equal(badRes.ok, false);
  assert.equal(badRes.reason, 'invalid-frame');
  const badPlanesRes = validProc.presentSplashFrame({
    width: 8,
    height: 8,
    format: 'rgba8',
    planes: {},
  });
  assert.equal(badPlanesRes.ok, false);
  assert.equal(badPlanesRes.reason, 'error');
  validProc.destroy();
});

// ---------------------------------------------------------------------------
// 5. Destroy cleanup releases SplashFrameBridge & BinkVideoPresenter GPU resources idempotently
// ---------------------------------------------------------------------------

test('W67 destroy() cleans up SplashFrameBridge and BinkVideoPresenter GPU textures & buffers idempotently', () => {
  const mock = createMockWebGpuEnv();
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx);

  const res = processor.presentSplashFrame({
    width: 16,
    height: 8,
    format: 'yuv420p',
    planes: makeYuv420Planes(16, 8),
  });
  assert.equal(res.ok, true);
  assert.ok(mock.createdTextures.length >= 4); // Y, U, V + dummy 1x1
  assert.ok(mock.createdBuffers.length >= 1); // uniform buffer

  processor.destroy();

  assert.equal(processor.destroyed, true);
  assert.equal(processor.splashFrameBridge.isDestroyed, true);
  assert.equal(processor.binkPresenter.isDestroyed, true);
  for (const tex of mock.createdTextures) {
    assert.equal(tex.destroyed, true, `expected texture ${tex.desc.label} to be destroyed`);
  }
  for (const buf of mock.createdBuffers) {
    assert.equal(buf.destroyed, true, `expected buffer ${buf.id} to be destroyed`);
  }

  const statsAfter = processor.getStats();
  assert.equal(statsAfter.destroyed, true);
  assert.equal(statsAfter.splashBridge.destroyed, true);
  assert.equal(statsAfter.binkPresenter.destroyed, true);

  // Idempotent second destroy + graceful rejection on subsequent presentSplashFrame
  assert.doesNotThrow(() => processor.destroy());
  const afterDestroy = processor.presentSplashFrame({
    width: 8,
    height: 8,
    rgba: makeRgbaBytes(8, 8),
  });
  assert.equal(afterDestroy.ok, false);
  assert.equal(afterDestroy.reason, 'destroyed');
});

// ---------------------------------------------------------------------------
// 6. Classic WorkerGlobalScope importScripts resolution with zero top-level const collisions
// ---------------------------------------------------------------------------

test('W67 classic WorkerGlobalScope resolves bink_presenter.js & splash_frame_bridge.js via importScripts with zero const collisions', async () => {
  const mock = createMockWebGpuEnv({ swapWidth: 640, swapHeight: 360 });
  const importedScripts = [];
  const postedMessages = [];

  const sandbox = {
    console,
    TextEncoder,
    TextDecoder,
    URL,
    performance,
    setTimeout,
    clearTimeout,
    navigator: mock.navigator,
    postMessage(msg) {
      postedMessages.push(msg);
    },
  };
  const vmCtx = vm.createContext(sandbox);
  vmCtx.self = vmCtx;
  vmCtx.importScripts = (...urls) => {
    for (const rel of urls) {
      const fullPath = path.resolve(WORKERS_DIR, rel);
      const source = readFileSync(fullPath, 'utf8');
      importedScripts.push(rel);
      vm.runInContext(source, vmCtx, { filename: fullPath });
    }
  };

  // Load wgpu_worker.js with NO CommonJS module/require (forces browser importScripts path).
  assert.equal(typeof vmCtx.require, 'undefined');
  assert.equal(typeof vmCtx.module, 'undefined');
  vm.runInContext(
    readFileSync(path.resolve(WORKERS_DIR, './wgpu_worker.js'), 'utf8'),
    vmCtx,
    { filename: path.resolve(WORKERS_DIR, './wgpu_worker.js') }
  );

  await vmCtx.self.onmessage({
    data: {
      init: true,
      canvas: mock.canvas,
      width: 640,
      height: 360,
      enableSplashBridge: true,
    },
  });

  assert.ok(
    importedScripts.includes('./bink_presenter.js'),
    `expected ./bink_presenter.js to be loaded via importScripts; got ${JSON.stringify(importedScripts)}`
  );
  assert.ok(
    importedScripts.includes('./splash_frame_bridge.js'),
    `expected ./splash_frame_bridge.js to be loaded via importScripts; got ${JSON.stringify(importedScripts)}`
  );
  assert.ok(vmCtx.PlayGta6BinkPresenter && typeof vmCtx.PlayGta6BinkPresenter.BinkVideoPresenter === 'function');
  assert.ok(vmCtx.PlayGta6SplashFrameBridge && typeof vmCtx.PlayGta6SplashFrameBridge.SplashFrameBridge === 'function');

  // Re-importing both scripts a second time in the same global scope must throw zero SyntaxErrors.
  assert.doesNotThrow(() => {
    vmCtx.importScripts('./bink_presenter.js', './splash_frame_bridge.js');
  });

  // Present an RGBA8 splash frame and a YUV420p Bink frame inside the classic worker realm.
  const vmRgba = vm.runInContext('new Uint8Array(8 * 4 * 4).fill(200)', vmCtx);
  await vmCtx.self.onmessage({
    data: {
      type: 'present-splash-frame',
      id: 'vm-f1',
      frame: {
        width: 8,
        height: 4,
        format: 'rgba8',
        planes: { rgba: vmRgba },
        frameIndex: 0,
      },
    },
  });

  const vmY = vm.runInContext('new Uint8Array(8 * 4).fill(180)', vmCtx);
  const vmU = vm.runInContext('new Uint8Array(4 * 2).fill(128)', vmCtx);
  const vmV = vm.runInContext('new Uint8Array(4 * 2).fill(128)', vmCtx);
  await vmCtx.self.onmessage({
    data: {
      type: 'bink-frame',
      id: 'vm-f2',
      frame: {
        width: 8,
        height: 4,
        format: 'yuv420p',
        planes: { y: vmY, u: vmU, v: vmV },
        frameIndex: 1,
      },
    },
  });

  const errors = postedMessages.filter((m) => m && m.error);
  assert.deepEqual(errors, [], `unexpected worker errors: ${JSON.stringify(errors)}`);

  const r1 = postedMessages.find((m) => m.type === 'splash-frame-presented' && m.id === 'vm-f1');
  const r2 = postedMessages.find((m) => m.type === 'splash-frame-presented' && m.id === 'vm-f2');
  assert.equal(r1.ok, true);
  assert.equal(r1.framesPresented, 1);
  assert.equal(r1.firstPresent, true);
  assert.equal(r2.ok, true);
  assert.equal(r2.framesPresented, 2);

  const telemetry = postedMessages.filter((m) => m.type === 'gpu-telemetry');
  assert.ok(telemetry.length >= 1, 'expected Frame-1 gpu-telemetry');
  assert.equal(telemetry[0].frameStats.framesPresented, 1);

  const firstPresents = postedMessages.filter((m) => m.type === 'first-present');
  assert.equal(firstPresents.length, 1);
});
