import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const bridgeMod = require('../web/workers/splash_frame_bridge.js');
const binkMod = require('../web/workers/bink_presenter.js');

const {
  DEFAULT_MAX_QUEUE_DEPTH,
  SPLASH_FRAME_SOURCE,
  PIXEL_FORMAT,
  PIXEL_FORMAT_ID,
  ALLOWED_FRAME_FORMATS,
  COLOR_SPACE,
  COLOR_RANGE,
  CHROMA_ORDER,
  LOAD_OP,
  sanitizeFrameFormat,
  sanitizeChromaOrder,
  sanitizeLoadOp,
  generateDiagnosticSplashFrame,
  SplashFrameBridge,
} = bridgeMod;

const {
  BinkVideoPresenter,
  BINK_PIXEL_FORMAT,
  BINK_CHROMA_ORDER,
  BINK_LOAD_OP,
  UNIFORM_FLAG_SWAP_CHROMA,
} = binkMod;

function createMockWebGpuDevice(options = {}) {
  let nextTextureId = 1;
  let nextBufferId = 1;
  let nextBindGroupId = 1;
  let nextPipelineId = 1;
  let nextShaderId = 1;
  let nextSamplerId = 1;
  let nextEncoderId = 1;

  const createdTextures = [];
  const createdBuffers = [];
  const createdShaderModules = [];
  const createdBindGroupLayouts = [];
  const createdPipelineLayouts = [];
  const createdRenderPipelines = [];
  const createdSamplers = [];
  const createdBindGroups = [];
  const createdEncoders = [];
  const writeTextureCalls = [];
  const writeBufferCalls = [];
  const copyExternalImageToTextureCalls = [];
  const submitCalls = [];

  const queue = {
    writeTexture(dest, data, dataLayout, size) {
      const bytes =
        data instanceof Uint8Array
          ? new Uint8Array(data)
          : new Uint8Array(data.buffer, data.byteOffset || 0, data.byteLength);
      writeTextureCalls.push({
        dest: { ...dest, origin: dest.origin ? { ...dest.origin } : undefined },
        data: bytes,
        rawData: data,
        dataLayout: { ...dataLayout },
        size: { ...size },
      });
    },
    writeBuffer(buffer, bufferOffset, data) {
      const bytes =
        data instanceof Uint8Array
          ? new Uint8Array(data)
          : ArrayBuffer.isView(data)
            ? new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength))
            : new Uint8Array(data.slice(0));
      if (buffer._backing) {
        buffer._backing.set(bytes, bufferOffset);
      }
      writeBufferCalls.push({ buffer, bufferOffset, data: bytes });
    },
    submit(commandBuffers) {
      submitCalls.push(commandBuffers);
    },
    onSubmittedWorkDone() {
      return Promise.resolve();
    },
  };

  if (options.supportCopyExternalImageToTexture !== false) {
    queue.copyExternalImageToTexture = (source, destination, copySize) => {
      copyExternalImageToTextureCalls.push({
        source: { ...source },
        destination: { ...destination },
        copySize: { ...copySize },
      });
    };
  }

  const device = {
    queue,
    createdTextures,
    createdBuffers,
    createdShaderModules,
    createdBindGroupLayouts,
    createdPipelineLayouts,
    createdRenderPipelines,
    createdSamplers,
    createdBindGroups,
    createdEncoders,
    writeTextureCalls,
    writeBufferCalls,
    copyExternalImageToTextureCalls,
    submitCalls,

    createTexture(desc) {
      const id = nextTextureId++;
      const view = { kind: 'GPUTextureView', id: `view-${id}`, textureId: id, label: desc.label };
      const texture = {
        kind: 'GPUTexture',
        id,
        label: desc.label,
        desc: { ...desc, size: { ...desc.size } },
        view,
        destroyed: false,
        destroyCalls: 0,
        createView() {
          return view;
        },
        destroy() {
          this.destroyed = true;
          this.destroyCalls += 1;
        },
      };
      createdTextures.push(texture);
      return texture;
    },

    createBuffer(desc) {
      const id = nextBufferId++;
      const backing = new Uint8Array(desc.size);
      let mapped = Boolean(desc.mappedAtCreation);
      const buffer = {
        kind: 'GPUBuffer',
        id,
        label: desc.label,
        desc: { ...desc },
        size: desc.size,
        _backing: backing,
        destroyed: false,
        destroyCalls: 0,
        getMappedRange() {
          if (!mapped) {
            throw new Error('Buffer is not mapped');
          }
          return backing.buffer;
        },
        unmap() {
          mapped = false;
        },
        mapAsync() {
          mapped = true;
          return Promise.resolve();
        },
        destroy() {
          this.destroyed = true;
          this.destroyCalls += 1;
        },
      };
      createdBuffers.push(buffer);
      return buffer;
    },

    createShaderModule(desc) {
      const mod = { kind: 'GPUShaderModule', id: nextShaderId++, desc: { ...desc } };
      createdShaderModules.push(mod);
      return mod;
    },

    createBindGroupLayout(desc) {
      const layout = {
        kind: 'GPUBindGroupLayout',
        id: createdBindGroupLayouts.length + 1,
        desc: { ...desc, entries: desc.entries.map((e) => ({ ...e })) },
      };
      createdBindGroupLayouts.push(layout);
      return layout;
    },

    createPipelineLayout(desc) {
      const layout = { kind: 'GPUPipelineLayout', id: createdPipelineLayouts.length + 1, desc };
      createdPipelineLayouts.push(layout);
      return layout;
    },

    createRenderPipeline(desc) {
      const pipeline = { kind: 'GPURenderPipeline', id: nextPipelineId++, desc };
      createdRenderPipelines.push(pipeline);
      return pipeline;
    },

    createSampler(desc) {
      const sampler = { kind: 'GPUSampler', id: nextSamplerId++, desc: { ...desc } };
      createdSamplers.push(sampler);
      return sampler;
    },

    createBindGroup(desc) {
      const bg = {
        kind: 'GPUBindGroup',
        id: nextBindGroupId++,
        desc: { ...desc, entries: desc.entries.map((e) => ({ ...e })) },
      };
      createdBindGroups.push(bg);
      return bg;
    },

    createCommandEncoder(desc = {}) {
      const ops = [];
      const renderPasses = [];
      const encoder = {
        id: nextEncoderId++,
        desc,
        ops,
        renderPasses,
        finished: false,
        beginRenderPass(descriptor) {
          const passRecord = {
            descriptor,
            pipeline: null,
            viewport: null,
            scissorRect: null,
            bindGroups: [],
            draws: [],
            ended: false,
          };
          renderPasses.push(passRecord);
          ops.push({ op: 'beginRenderPass', descriptor });
          return {
            setPipeline(pipeline) {
              passRecord.pipeline = pipeline;
              ops.push({ op: 'setPipeline', pipeline });
            },
            setViewport(x, y, width, height, minDepth, maxDepth) {
              passRecord.viewport = { x, y, width, height, minDepth, maxDepth };
              ops.push({ op: 'setViewport', viewport: passRecord.viewport });
            },
            setScissorRect(x, y, width, height) {
              passRecord.scissorRect = { x, y, width, height };
              ops.push({ op: 'setScissorRect', scissorRect: passRecord.scissorRect });
            },
            setBindGroup(index, bindGroup) {
              passRecord.bindGroups.push({ index, bindGroup });
              ops.push({ op: 'setBindGroup', index, bindGroup });
            },
            draw(vertexCount, instanceCount, firstVertex, firstInstance) {
              passRecord.draws.push({ vertexCount, instanceCount, firstVertex, firstInstance });
              ops.push({ op: 'draw', vertexCount, instanceCount, firstVertex, firstInstance });
            },
            end() {
              passRecord.ended = true;
              ops.push({ op: 'endRenderPass' });
            },
          };
        },
        finish() {
          this.finished = true;
          return { kind: 'GPUCommandBuffer', encoderId: this.id };
        },
      };
      createdEncoders.push(encoder);
      return encoder;
    },
  };

  return device;
}

function makeRgbaFrame(width, height, fill = 0x80) {
  const rgba = new Uint8Array(width * height * 4);
  rgba.fill(fill);
  for (let i = 3; i < rgba.length; i += 4) {
    rgba[i] = 255;
  }
  return rgba;
}

function makeYuv420Planes(width, height) {
  const uvW = Math.ceil(width / 2);
  const uvH = Math.ceil(height / 2);
  return {
    y: new Uint8Array(width * height).fill(180),
    u: new Uint8Array(uvW * uvH).fill(128),
    v: new Uint8Array(uvW * uvH).fill(128),
  };
}

test('UMD export attaches PlayGta6SplashFrameBridge and freezes public constants', () => {
  assert.equal(globalThis.PlayGta6SplashFrameBridge, bridgeMod);
  assert.equal(globalThis.PlayGta6SplashFrameBridge.SplashFrameBridge, SplashFrameBridge);
  assert.ok(Object.isFrozen(bridgeMod));
  assert.ok(Object.isFrozen(SPLASH_FRAME_SOURCE));
  assert.equal(SPLASH_FRAME_SOURCE.RAW, 'raw');
  assert.equal(SPLASH_FRAME_SOURCE.EXTERNAL, 'external');
  assert.equal(SPLASH_FRAME_SOURCE.DIAGNOSTIC, 'diagnostic');
  assert.equal(PIXEL_FORMAT.RGBA8, BINK_PIXEL_FORMAT.RGBA8);
  assert.equal(DEFAULT_MAX_QUEUE_DEPTH, 3);
});

test('generateDiagnosticSplashFrame returns original RGBA8 color-bar/gradient pattern', () => {
  const frame = generateDiagnosticSplashFrame(32, 24, 7);
  assert.equal(frame.width, 32);
  assert.equal(frame.height, 24);
  assert.equal(frame.format, PIXEL_FORMAT.RGBA8);
  assert.equal(frame.source, SPLASH_FRAME_SOURCE.DIAGNOSTIC);
  assert.equal(frame.frameIndex, 7);
  assert.equal(frame.planes.rgba.byteLength, 32 * 24 * 4);

  // Leftmost top pixel should be near-white bar; alpha always opaque.
  assert.equal(frame.planes.rgba[0], 235);
  assert.equal(frame.planes.rgba[1], 235);
  assert.equal(frame.planes.rgba[2], 235);
  assert.equal(frame.planes.rgba[3], 255);

  // Bottom gradient row differs from top color bars.
  const bottomOffset = (23 * 32 + 16) * 4;
  assert.notEqual(frame.planes.rgba[bottomOffset], 235);

  // Animation phase changes pixels across frameIndex.
  const frame0 = generateDiagnosticSplashFrame(16, 12, 0);
  const frame5 = generateDiagnosticSplashFrame(16, 12, 5);
  assert.notDeepEqual(frame0.planes.rgba, frame5.planes.rgba);

  assert.throws(() => generateDiagnosticSplashFrame(0, 10, 0), RangeError);
  assert.throws(() => generateDiagnosticSplashFrame(10, -1, 0), RangeError);
});

test('pushRawFrame accepts RGBA8 buffers and clones plane ownership', () => {
  const device = createMockWebGpuDevice();
  const bridge = new SplashFrameBridge({ device, maxQueueDepth: 4, targetFps: 60 });

  const rgba = makeRgbaFrame(8, 4, 0x11);
  const pushed = bridge.pushRawFrame({
    width: 8,
    height: 4,
    format: PIXEL_FORMAT.RGBA8,
    planes: { rgba },
    timestampMs: 0,
  });

  assert.equal(pushed.queueDepth, 1);
  assert.equal(pushed.droppedFrames, 0);
  assert.equal(bridge.getStats().queueDepth, 1);

  // Mutating the caller buffer must not alter the queued copy.
  rgba.fill(0);
  const targetView = { kind: 'GPUTextureView', id: 'swap-0' };
  const result = bridge.renderFrame(device, targetView, 640, 360, 0);
  assert.equal(result.presented, true);
  assert.equal(result.source, SPLASH_FRAME_SOURCE.RAW);
  assert.ok(device.writeTextureCalls.length >= 1);
  const rgbaUpload = device.writeTextureCalls.find((call) => call.data[0] === 0x11);
  assert.ok(rgbaUpload, 'expected cloned RGBA upload to preserve source pixels');
  assert.equal(rgbaUpload.data[3], 255);
});

test('pushRawFrame accepts YUV420P planar frames and presents through BinkVideoPresenter', () => {
  const device = createMockWebGpuDevice();
  const presenter = new BinkVideoPresenter({ device, targetFps: 120 });
  const bridge = new SplashFrameBridge({ presenter, maxQueueDepth: 2 });

  const planes = makeYuv420Planes(16, 8);
  bridge.pushRawFrame({
    width: 16,
    height: 8,
    format: PIXEL_FORMAT.YUV420P,
    planes,
    timestampMs: 0,
    colorSpace: 'bt601',
  });

  const result = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 320, 180, 0);
  assert.equal(result.presented, true);
  assert.equal(result.width, 16);
  assert.equal(result.height, 8);
  assert.equal(presenter.pixelFormat, BINK_PIXEL_FORMAT.YUV420P);
  assert.equal(presenter.colorSpace, 'bt601');
  // Y + U + V plane uploads (+ optional 1x1 dummy texture write during present)
  assert.ok(device.writeTextureCalls.length >= 3);
  assert.equal(bridge.getStats().lastSource, SPLASH_FRAME_SOURCE.RAW);
});

test('bounded queue drops oldest frames and increments droppedFrames under overload', () => {
  const device = createMockWebGpuDevice();
  const bridge = new SplashFrameBridge({ device, maxQueueDepth: 2, targetFps: 1000 });

  bridge.pushRawFrame({
    width: 4,
    height: 4,
    format: PIXEL_FORMAT.RGBA8,
    planes: { rgba: makeRgbaFrame(4, 4, 1) },
    frameIndex: 0,
    timestampMs: 0,
  });
  bridge.pushRawFrame({
    width: 4,
    height: 4,
    format: PIXEL_FORMAT.RGBA8,
    planes: { rgba: makeRgbaFrame(4, 4, 2) },
    frameIndex: 1,
    timestampMs: 1,
  });
  const overflow = bridge.pushRawFrame({
    width: 4,
    height: 4,
    format: PIXEL_FORMAT.RGBA8,
    planes: { rgba: makeRgbaFrame(4, 4, 3) },
    frameIndex: 2,
    timestampMs: 2,
  });

  assert.equal(overflow.queueDepth, 2);
  assert.equal(overflow.droppedFrames, 1);
  assert.equal(bridge.getStats().droppedFrames, 1);
  assert.equal(bridge.getStats().queueDepth, 2);

  // Oldest (frameIndex 0) was dropped by overflow. At nowMs=1 only frameIndex 1 is due.
  const first = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 128, 72, 1);
  assert.equal(first.presented, true);
  assert.equal(first.frameIndex, 1);

  const second = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 128, 72, 50);
  assert.equal(second.presented, true);
  assert.equal(second.frameIndex, 2);
  assert.equal(bridge.getStats().queueDepth, 0);
});

test('pushDiagnosticFrame enqueues generateDiagnosticSplashFrame output', () => {
  const device = createMockWebGpuDevice();
  const bridge = new SplashFrameBridge({ device, targetFps: 60 });

  const pushed = bridge.pushDiagnosticFrame(20, 10, 3, { timestampMs: 0 });
  assert.equal(pushed.queueDepth, 1);
  assert.equal(pushed.frameIndex, 3);

  const result = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 400, 300, 0);
  assert.equal(result.presented, true);
  assert.equal(result.source, SPLASH_FRAME_SOURCE.DIAGNOSTIC);
  assert.equal(result.frameIndex, 3);
  assert.equal(result.width, 20);
  assert.equal(result.height, 10);
  assert.equal(bridge.getStats().lastSource, SPLASH_FRAME_SOURCE.DIAGNOSTIC);
  assert.ok(device.writeTextureCalls.length >= 1);
});

test('pushExternalImage uses copyExternalImageToTexture and RGBA extraction fallback', () => {
  const device = createMockWebGpuDevice();
  const bridge = new SplashFrameBridge({ device, targetFps: 90 });

  const rgba = makeRgbaFrame(8, 8, 0x44);
  const imageBitmap = {
    width: 8,
    height: 8,
    data: rgba,
    closeCalls: 0,
    close() {
      this.closeCalls += 1;
    },
  };

  bridge.pushExternalImage(imageBitmap, { width: 8, height: 8, timestampMs: 0 });
  const result = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 256, 256, 0);

  assert.equal(result.presented, true);
  assert.equal(result.source, SPLASH_FRAME_SOURCE.EXTERNAL);
  assert.equal(device.copyExternalImageToTextureCalls.length, 1);
  assert.equal(device.copyExternalImageToTextureCalls[0].source.source, imageBitmap);
  assert.ok(device.writeTextureCalls.length >= 1);
  const rgbaUpload = device.writeTextureCalls.find((call) => call.data[0] === 0x44);
  assert.ok(rgbaUpload, 'expected RGBA upload containing extracted external pixels');
});

test('pushExternalImage works when only copyExternalImageToTexture is available (no pixel data)', () => {
  const device = createMockWebGpuDevice();
  const bridge = new SplashFrameBridge({ device, targetFps: 90 });

  const imageBitmap = {
    width: 4,
    height: 4,
    closeCalls: 0,
    close() {
      this.closeCalls += 1;
    },
  };

  bridge.pushExternalImage(imageBitmap, { timestampMs: 0 });
  const result = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 128, 128, 0);
  assert.equal(result.presented, true);
  assert.equal(result.source, SPLASH_FRAME_SOURCE.EXTERNAL);
  assert.equal(device.copyExternalImageToTextureCalls.length, 1);
  // Fallback solid RGBA upload still feeds the presenter.
  assert.ok(device.writeTextureCalls.length >= 1);
});

test('onFirstPresent fires exactly once across multiple presented frames', () => {
  const device = createMockWebGpuDevice();
  const events = [];
  const bridge = new SplashFrameBridge({
    device,
    targetFps: 1000,
    onFirstPresent(info) {
      events.push(info);
    },
  });

  bridge.pushDiagnosticFrame(16, 9, 0, { timestampMs: 0 });
  const r0 = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 640, 360, 0);
  assert.equal(r0.presented, true);

  bridge.pushDiagnosticFrame(16, 9, 1, { timestampMs: 10 });
  const r1 = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 640, 360, 10);
  assert.equal(r1.presented, true);

  bridge.pushDiagnosticFrame(16, 9, 2, { timestampMs: 20 });
  const r2 = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 640, 360, 20);
  assert.equal(r2.presented, true);

  assert.equal(events.length, 1);
  assert.equal(events[0].frameIndex, 0);
  assert.equal(events[0].width, 16);
  assert.equal(events[0].height, 9);
  assert.equal(events[0].source, SPLASH_FRAME_SOURCE.DIAGNOSTIC);
  assert.equal(bridge.getStats().firstPresentSent, true);
  assert.equal(bridge.getStats().framesPresented, 3);
});

test('renderFrame respects presenter.shouldAdvance pacing and skips when queue empty', () => {
  const device = createMockWebGpuDevice();
  const bridge = new SplashFrameBridge({ device, targetFps: 30 });

  const empty = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 100, 100, 0);
  assert.equal(empty.presented, false);
  assert.equal(empty.reason, 'empty');

  bridge.pushRawFrame({
    width: 4,
    height: 4,
    format: PIXEL_FORMAT.RGBA8,
    planes: { rgba: makeRgbaFrame(4, 4) },
    timestampMs: 0,
  });
  bridge.pushRawFrame({
    width: 4,
    height: 4,
    format: PIXEL_FORMAT.RGBA8,
    planes: { rgba: makeRgbaFrame(4, 4, 9) },
    timestampMs: 1,
  });

  const first = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 100, 100, 0);
  assert.equal(first.presented, true);

  const paced = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 100, 100, 10);
  assert.equal(paced.presented, false);
  assert.equal(paced.reason, 'pacing');
  assert.equal(bridge.getStats().queueDepth, 1);

  const second = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 100, 100, 40);
  assert.equal(second.presented, true);
  assert.equal(bridge.getStats().queueDepth, 0);
});

test('renderFrame dequeues the latest due timestamped frame and drops stale ones', () => {
  const device = createMockWebGpuDevice();
  const bridge = new SplashFrameBridge({ device, maxQueueDepth: 8, targetFps: 1000 });

  for (let i = 0; i < 4; i += 1) {
    bridge.pushRawFrame({
      width: 2,
      height: 2,
      format: PIXEL_FORMAT.RGBA8,
      planes: { rgba: makeRgbaFrame(2, 2, i + 1) },
      frameIndex: i,
      timestampMs: i * 10,
    });
  }

  const result = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 64, 64, 25);
  assert.equal(result.presented, true);
  assert.equal(result.frameIndex, 2);
  // Frames 0 and 1 were stale relative to the selected due frame.
  assert.equal(bridge.getStats().droppedFrames, 2);
  assert.equal(bridge.getStats().queueDepth, 1);
});

test('getStats reports queue, present, drop, and nested presenter telemetry', () => {
  const device = createMockWebGpuDevice();
  const bridge = new SplashFrameBridge({ device, maxQueueDepth: 2, targetFps: 60 });

  let stats = bridge.getStats();
  assert.deepEqual(
    {
      framesPresented: stats.framesPresented,
      droppedFrames: stats.droppedFrames,
      queueDepth: stats.queueDepth,
      firstPresentSent: stats.firstPresentSent,
      lastSource: stats.lastSource,
      maxQueueDepth: stats.maxQueueDepth,
      destroyed: stats.destroyed,
    },
    {
      framesPresented: 0,
      droppedFrames: 0,
      queueDepth: 0,
      firstPresentSent: false,
      lastSource: null,
      maxQueueDepth: 2,
      destroyed: false,
    }
  );

  bridge.pushDiagnosticFrame(8, 8, 0, { timestampMs: 0 });
  bridge.pushDiagnosticFrame(8, 8, 1, { timestampMs: 1 });
  bridge.pushDiagnosticFrame(8, 8, 2, { timestampMs: 2 });
  const presented = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 200, 100, 2);
  assert.equal(presented.presented, true);

  stats = bridge.getStats();
  assert.equal(stats.framesPresented, 1);
  // One overflow drop (maxQueueDepth=2) plus one stale due-frame drop (frame 1).
  assert.equal(stats.droppedFrames, 2);
  assert.equal(stats.queueDepth, 0);
  assert.equal(stats.firstPresentSent, true);
  assert.equal(stats.lastSource, SPLASH_FRAME_SOURCE.DIAGNOSTIC);
  assert.ok(stats.presenterStats);
  assert.equal(stats.presenterStats.uploadedFrames, 1);
  assert.equal(stats.presenterStats.presentedFrames, 1);
  assert.equal(stats.presenterStats.hasFrame, true);
});

test('reset clears queue/telemetry and allows onFirstPresent to fire again', () => {
  const device = createMockWebGpuDevice();
  const events = [];
  const bridge = new SplashFrameBridge({
    device,
    targetFps: 1000,
    onFirstPresent(info) {
      events.push(info.frameIndex);
    },
  });

  bridge.pushDiagnosticFrame(8, 4, 0, { timestampMs: 0 });
  bridge.renderFrame(device, { kind: 'GPUTextureView' }, 100, 50, 0);
  assert.deepEqual(events, [0]);
  assert.equal(bridge.getStats().firstPresentSent, true);

  bridge.pushDiagnosticFrame(8, 4, 1, { timestampMs: 1 });
  bridge.reset();

  assert.equal(bridge.getStats().framesPresented, 0);
  assert.equal(bridge.getStats().droppedFrames, 0);
  assert.equal(bridge.getStats().queueDepth, 0);
  assert.equal(bridge.getStats().firstPresentSent, false);
  assert.equal(bridge.getStats().lastSource, null);

  bridge.pushDiagnosticFrame(8, 4, 9, { timestampMs: 0 });
  bridge.renderFrame(device, { kind: 'GPUTextureView' }, 100, 50, 0);
  assert.deepEqual(events, [0, 9]);
  assert.equal(bridge.getStats().firstPresentSent, true);
});

test('destroy is idempotent, destroys owned presenter, and blocks further use', () => {
  const device = createMockWebGpuDevice();
  const bridge = new SplashFrameBridge({ device, targetFps: 60 });

  bridge.pushDiagnosticFrame(4, 4, 0, { timestampMs: 0 });
  bridge.renderFrame(device, { kind: 'GPUTextureView' }, 64, 64, 0);

  const presenter = bridge.presenter;
  assert.ok(presenter);
  assert.equal(presenter.isDestroyed, false);

  bridge.destroy();
  assert.equal(bridge.isDestroyed, true);
  assert.equal(presenter.isDestroyed, true);
  assert.equal(bridge.getStats().destroyed, true);

  bridge.destroy();
  assert.equal(presenter.isDestroyed, true);

  assert.throws(
    () =>
      bridge.pushRawFrame({
        width: 2,
        height: 2,
        format: PIXEL_FORMAT.RGBA8,
        planes: { rgba: makeRgbaFrame(2, 2) },
      }),
    /destroyed/i
  );
  assert.throws(
    () => bridge.renderFrame(device, { kind: 'GPUTextureView' }, 64, 64, 0),
    /destroyed/i
  );
  assert.throws(() => bridge.reset(), /destroyed/i);
});

test('injected presenter is reused and not destroyed by bridge.destroy()', () => {
  const device = createMockWebGpuDevice();
  const presenter = new BinkVideoPresenter({ device, targetFps: 120 });
  const bridge = new SplashFrameBridge({ presenter, maxQueueDepth: 3 });

  bridge.pushRawFrame({
    width: 4,
    height: 4,
    format: PIXEL_FORMAT.RGBA8,
    planes: { rgba: makeRgbaFrame(4, 4) },
    timestampMs: 0,
  });
  const result = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 128, 72, 0);
  assert.equal(result.presented, true);
  assert.equal(bridge.presenter, presenter);

  bridge.destroy();
  assert.equal(bridge.isDestroyed, true);
  assert.equal(presenter.isDestroyed, false);
  presenter.destroy();
  assert.equal(presenter.isDestroyed, true);
});

function makeNv12Planes(width, height, options = {}) {
  const uvW = Math.ceil(width / 2);
  const uvH = Math.ceil(height / 2);
  const yStride = options.yStride !== undefined ? options.yStride : width;
  const uvStride = options.uvStride !== undefined ? options.uvStride : uvW * 2;
  const yFill = options.yFill !== undefined ? options.yFill : 0xa5;
  const uFill = options.uFill !== undefined ? options.uFill : 0x60;
  const vFill = options.vFill !== undefined ? options.vFill : 0x90;

  const yPlane = new Uint8Array(yStride * height);
  for (let row = 0; row < height; row += 1) {
    yPlane.fill(yFill, row * yStride, row * yStride + width);
  }

  const uvPlane = new Uint8Array(uvStride * uvH);
  for (let row = 0; row < uvH; row += 1) {
    for (let col = 0; col < uvW; col += 1) {
      const idx = row * uvStride + col * 2;
      uvPlane[idx] = uFill;
      uvPlane[idx + 1] = vFill;
    }
  }

  return { yPlane, uvPlane, yStride, uvStride };
}

test('pushRawFrame forwards NV12 payloads (yPlane/uvPlane, strides, colorSpace, colorRange) to BinkVideoPresenter.uploadNv12Frame', () => {
  const device = createMockWebGpuDevice();
  const presenter = new BinkVideoPresenter({ device, targetFps: 120 });
  const bridge = new SplashFrameBridge({ presenter, maxQueueDepth: 3 });

  assert.equal(PIXEL_FORMAT.NV12, BINK_PIXEL_FORMAT.NV12);
  assert.equal(PIXEL_FORMAT_ID[3], PIXEL_FORMAT.NV12);
  assert.equal(COLOR_SPACE.BT601, 'bt601');
  assert.equal(COLOR_RANGE.FULL, 'full');

  const { yPlane, uvPlane, yStride, uvStride } = makeNv12Planes(16, 8, {
    yStride: 24,
    uvStride: 20,
    yFill: 0xb2,
    uFill: 0x44,
    vFill: 0xcc,
  });

  bridge.pushRawFrame({
    width: 16,
    height: 8,
    format: 'nv12',
    yPlane,
    uvPlane,
    yStride,
    uvStride,
    colorSpace: 'bt601',
    colorRange: 'full',
    timestampMs: 0,
  });

  // Mutating caller buffers after enqueue must not affect uploaded plane bytes.
  yPlane.fill(0);
  uvPlane.fill(0);

  const result = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 640, 360, 0);
  assert.equal(result.presented, true);
  assert.equal(result.format, ALLOWED_FRAME_FORMATS.NV12);
  assert.equal(result.colorSpace, 'bt601');
  assert.equal(result.colorRange, 'full');
  assert.equal(presenter.pixelFormat, BINK_PIXEL_FORMAT.NV12);
  assert.equal(presenter.colorSpace, 'bt601');
  assert.equal(presenter.colorRange, 'full');

  const yUpload = device.writeTextureCalls.find(
    (call) => call.size.width === 16 && call.size.height === 8 && call.data[0] === 0xb2
  );
  const uvUpload = device.writeTextureCalls.find(
    (call) =>
      call.size.width === 8 &&
      call.size.height === 4 &&
      call.data[0] === 0x44 &&
      call.data[1] === 0xcc
  );
  assert.ok(yUpload, 'expected cloned NV12 Y plane upload');
  assert.ok(uvUpload, 'expected cloned NV12 interleaved UV plane upload');

  const stats = bridge.getStats();
  assert.equal(stats.nv12FramesForwarded, 1);
  assert.equal(stats.lastFrameFormat, 'nv12');
  assert.equal(stats.framesBeforeFirstPresent, 1);
});

test('NV12 numeric format 3 and uploadFrame fallback forward cleanly without counting dropped frames', () => {
  const device = createMockWebGpuDevice();
  const uploadedFrames = [];
  const mockPresenter = {
    uploadFrame(frame) {
      uploadedFrames.push(frame);
      return { format: frame.format };
    },
    presentFrame() {
      return {
        presented: true,
        viewport: { x: 0, y: 0, width: 320, height: 180 },
        colorSpace: 'bt709',
        colorRange: 'limited',
      };
    },
  };

  const bridge = new SplashFrameBridge({ presenter: mockPresenter, maxQueueDepth: 1 });
  const droppedNv12 = makeNv12Planes(8, 4, { yFill: 0x10 });
  const keptNv12 = makeNv12Planes(8, 4, { yStride: 12, uvStride: 10, yFill: 0x77 });

  // First NV12 frame will be evicted by maxQueueDepth=1 before forwarding.
  bridge.pushRawFrame({
    width: 8,
    height: 4,
    format: 3,
    planes: { y: droppedNv12.yPlane, uv: droppedNv12.uvPlane },
    frameIndex: 0,
    timestampMs: 0,
  });
  bridge.pushNv12Frame({
    width: 8,
    height: 4,
    yPlane: keptNv12.yPlane,
    uvPlane: keptNv12.uvPlane,
    yStride: 12,
    uvStride: 10,
    colorSpace: 'bt709',
    colorRange: 'limited',
    frameIndex: 1,
    timestampMs: 1,
  });

  const result = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 320, 180, 1);
  assert.equal(result.presented, true);
  assert.equal(result.frameIndex, 1);
  assert.equal(uploadedFrames.length, 1);
  assert.equal(uploadedFrames[0].format, 'nv12');
  assert.equal(uploadedFrames[0].y[0], 0x77);
  assert.equal(uploadedFrames[0].yStride, 12);
  assert.equal(uploadedFrames[0].uvStride, 10);
  assert.equal(uploadedFrames[0].colorSpace, 'bt709');
  assert.equal(uploadedFrames[0].colorRange, 'limited');

  const stats = bridge.getStats();
  assert.equal(stats.droppedFrames, 1);
  assert.equal(stats.nv12FramesForwarded, 1);
  assert.equal(stats.lastFrameFormat, 'nv12');
});

test('YUV420P and YUVA420P propagate colorSpace, colorRange, and optional strides into BinkVideoPresenter', () => {
  const device = createMockWebGpuDevice();
  const bridge = new SplashFrameBridge({
    device,
    targetFps: 1000,
    colorSpace: 'bt709',
    colorRange: 'full',
  });

  const planes = makeYuv420Planes(16, 8);
  bridge.pushRawFrame({
    width: 16,
    height: 8,
    format: 1,
    planes,
    colorSpace: 'bt601',
    colorRange: 'full',
    timestampMs: 0,
  });

  const r0 = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 320, 180, 0);
  assert.equal(r0.presented, true);
  assert.equal(r0.colorSpace, 'bt601');
  assert.equal(r0.colorRange, 'full');
  assert.equal(bridge.presenter.colorSpace, 'bt601');
  assert.equal(bridge.presenter.colorRange, 'full');
  assert.equal(bridge.getStats().lastFrameFormat, 'yuv420p');

  const alpha = new Uint8Array(16 * 8).fill(200);
  bridge.pushRawFrame({
    width: 16,
    height: 8,
    format: PIXEL_FORMAT.YUVA420P,
    planes: { ...planes, a: alpha },
    colorSpace: 'bt709',
    colorRange: 'limited',
    timestampMs: 10,
  });

  const r1 = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 320, 180, 10);
  assert.equal(r1.presented, true);
  assert.equal(bridge.presenter.pixelFormat, BINK_PIXEL_FORMAT.YUVA420P);
  assert.equal(bridge.presenter.colorSpace, 'bt709');
  assert.equal(bridge.presenter.colorRange, 'limited');
  // YUVA420P maps to allowlisted 'yuv420p' token in telemetry.
  assert.equal(bridge.getStats().lastFrameFormat, 'yuv420p');

  assert.throws(
    () =>
      bridge.pushRawFrame({
        width: 8,
        height: 4,
        format: 'nv12',
        yPlane: new Uint8Array(32),
        uvPlane: new Uint8Array(16),
        colorRange: 'invalid-range',
      }),
    RangeError
  );
  assert.throws(
    () =>
      bridge.pushRawFrame({
        width: 8,
        height: 4,
        format: 'nv12',
        yPlane: new Uint8Array(32),
      }),
    TypeError
  );
});

test('lastFrameFormat is strictly sanitized to allowlisted tokens across all formats and invalid inputs', () => {
  assert.equal(sanitizeFrameFormat('rgba8'), 'rgba8');
  assert.equal(sanitizeFrameFormat('RGBA8UNORM'), 'rgba8');
  assert.equal(sanitizeFrameFormat(0), 'rgba8');
  assert.equal(sanitizeFrameFormat('yuv420p'), 'yuv420p');
  assert.equal(sanitizeFrameFormat('YUVA420P'), 'yuv420p');
  assert.equal(sanitizeFrameFormat(1), 'yuv420p');
  assert.equal(sanitizeFrameFormat(2), 'yuv420p');
  assert.equal(sanitizeFrameFormat('NV12'), 'nv12');
  assert.equal(sanitizeFrameFormat(3), 'nv12');
  assert.equal(sanitizeFrameFormat('ImageBitmap'), 'imagebitmap');
  assert.equal(sanitizeFrameFormat('image_bitmap'), 'imagebitmap');

  // Non-allowlisted / adversarial values (including host paths & URLs) must sanitize to null.
  assert.equal(sanitizeFrameFormat(null), null);
  assert.equal(sanitizeFrameFormat(undefined), null);
  assert.equal(sanitizeFrameFormat(99), null);
  assert.equal(sanitizeFrameFormat({}), null);
  assert.equal(sanitizeFrameFormat('C:\\Users\\player\\Movies\\intro.bk2'), null);
  assert.equal(sanitizeFrameFormat('/home/user/dump.nv12'), null);
  assert.equal(sanitizeFrameFormat('https://example.invalid/frame.raw'), null);
  assert.equal(sanitizeFrameFormat('bgra10_xr'), null);

  const device = createMockWebGpuDevice();
  const bridge = new SplashFrameBridge({ device, targetFps: 1000 });
  assert.equal(bridge.getStats().lastFrameFormat, null);

  bridge.pushDiagnosticFrame(8, 4, 0, { timestampMs: 0 });
  bridge.renderFrame(device, { kind: 'GPUTextureView' }, 64, 32, 0);
  assert.equal(bridge.getStats().lastFrameFormat, 'rgba8');

  bridge.pushRawFrame({
    width: 8,
    height: 4,
    format: PIXEL_FORMAT.YUV420P,
    planes: makeYuv420Planes(8, 4),
    timestampMs: 10,
  });
  bridge.renderFrame(device, { kind: 'GPUTextureView' }, 64, 32, 10);
  assert.equal(bridge.getStats().lastFrameFormat, 'yuv420p');

  const nv12 = makeNv12Planes(8, 4);
  bridge.pushRawFrame({
    width: 8,
    height: 4,
    format: 'nv12',
    yPlane: nv12.yPlane,
    uvPlane: nv12.uvPlane,
    timestampMs: 20,
  });
  bridge.renderFrame(device, { kind: 'GPUTextureView' }, 64, 32, 20);
  assert.equal(bridge.getStats().lastFrameFormat, 'nv12');

  bridge.pushExternalImage(
    { width: 4, height: 4, data: makeRgbaFrame(4, 4, 0x55) },
    { timestampMs: 30 }
  );
  bridge.renderFrame(device, { kind: 'GPUTextureView' }, 64, 32, 30);
  assert.equal(bridge.getStats().lastFrameFormat, 'imagebitmap');

  // Even if internal state were corrupted with an arbitrary path, getStats() sanitizes it.
  bridge._lastFrameFormat = 'C:\\secret\\host\\path.bin';
  assert.equal(bridge.getStats().lastFrameFormat, null);
});

test('framesBeforeFirstPresent and lateSplashFramesIgnored track splash-to-first-present handoff and suppression', () => {
  const device = createMockWebGpuDevice();
  const bridge = new SplashFrameBridge({ device, targetFps: 1000, maxQueueDepth: 4 });

  // Present 2 splash frames before live guest FIRST_PRESENT handoff.
  bridge.pushDiagnosticFrame(8, 4, 0, { timestampMs: 0 });
  bridge.renderFrame(device, { kind: 'GPUTextureView' }, 128, 72, 0);

  const nv12 = makeNv12Planes(8, 4);
  bridge.pushRawFrame({
    width: 8,
    height: 4,
    format: 'nv12',
    yPlane: nv12.yPlane,
    uvPlane: nv12.uvPlane,
    timestampMs: 10,
  });
  bridge.renderFrame(device, { kind: 'GPUTextureView' }, 128, 72, 10);

  // Queue one more splash frame right before handoff.
  bridge.pushDiagnosticFrame(8, 4, 2, { timestampMs: 20 });

  let stats = bridge.getStats();
  assert.equal(stats.framesPresented, 2);
  assert.equal(stats.framesBeforeFirstPresent, 2);
  assert.equal(stats.lateSplashFramesIgnored, 0);
  assert.equal(stats.nv12FramesForwarded, 1);
  assert.equal(stats.guestFirstPresent, false);
  assert.equal(stats.splashSuppressed, false);

  // Signal live guest FIRST_PRESENT handoff (default splash suppression enabled).
  const handoff = bridge.notifyGuestFirstPresent();
  assert.equal(handoff.guestFirstPresent, true);
  assert.equal(handoff.splashSuppressed, true);
  assert.equal(handoff.framesBeforeFirstPresent, 2);
  assert.equal(handoff.drainedFrames, 1);
  assert.equal(handoff.lateSplashFramesIgnored, 1);

  const encodersBefore = device.createdEncoders.length;
  const submitsBefore = device.submitCalls.length;

  // Late splash frames pushed after handoff are ignored without uploading or presenting.
  const lateRaw = bridge.pushRawFrame({
    width: 8,
    height: 4,
    format: 3,
    yPlane: nv12.yPlane,
    uvPlane: nv12.uvPlane,
    timestampMs: 30,
  });
  assert.equal(lateRaw.ignored, true);
  assert.equal(lateRaw.reason, 'suppressed');

  const lateDiag = bridge.pushDiagnosticFrame(8, 4, 4, { timestampMs: 40 });
  assert.equal(lateDiag.ignored, true);
  assert.equal(lateDiag.reason, 'suppressed');

  const lateExt = bridge.pushExternalImage(
    { width: 4, height: 4, data: makeRgbaFrame(4, 4) },
    { timestampMs: 50 }
  );
  assert.equal(lateExt.ignored, true);
  assert.equal(lateExt.reason, 'suppressed');

  const suppressedRender = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 128, 72, 60);
  assert.equal(suppressedRender.presented, false);
  assert.equal(suppressedRender.reason, 'suppressed');
  assert.equal(device.createdEncoders.length, encodersBefore);
  assert.equal(device.submitCalls.length, submitsBefore);

  stats = bridge.getStats();
  assert.equal(stats.framesPresented, 2);
  assert.equal(stats.framesBeforeFirstPresent, 2);
  assert.equal(stats.lateSplashFramesIgnored, 4);
  assert.equal(stats.nv12FramesForwarded, 1);
  assert.equal(stats.lastFrameFormat, 'nv12');
  assert.equal(stats.queueDepth, 0);

  // Reset clears handoff, suppression, and format telemetry.
  bridge.reset();
  stats = bridge.getStats();
  assert.equal(stats.framesBeforeFirstPresent, 0);
  assert.equal(stats.lateSplashFramesIgnored, 0);
  assert.equal(stats.nv12FramesForwarded, 0);
  assert.equal(stats.lastFrameFormat, null);
  assert.equal(stats.guestFirstPresent, false);
  assert.equal(stats.splashSuppressed, false);
});

test('notifyGuestFirstPresent without suppression freezes framesBeforeFirstPresent until suppression is enabled', () => {
  const device = createMockWebGpuDevice();
  const bridge = new SplashFrameBridge({
    device,
    targetFps: 1000,
    suppressSplashAfterFirstPresent: false,
  });

  bridge.pushDiagnosticFrame(8, 4, 0, { timestampMs: 0 });
  bridge.renderFrame(device, { kind: 'GPUTextureView' }, 100, 50, 0);
  assert.equal(bridge.getStats().framesBeforeFirstPresent, 1);

  const handoff = bridge.notifyGuestFirstPresent();
  assert.equal(handoff.guestFirstPresent, true);
  assert.equal(handoff.splashSuppressed, false);

  // Without suppression active, splash presentation can continue while framesBeforeFirstPresent stays frozen.
  bridge.pushDiagnosticFrame(8, 4, 1, { timestampMs: 10 });
  const r1 = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 100, 50, 10);
  assert.equal(r1.presented, true);
  assert.equal(bridge.getStats().framesPresented, 2);
  assert.equal(bridge.getStats().framesBeforeFirstPresent, 1);
  assert.equal(bridge.getStats().lateSplashFramesIgnored, 0);

  // Enabling suppression later ignores subsequent splash frames.
  bridge.setSplashSuppressed(true);
  const ignored = bridge.pushDiagnosticFrame(8, 4, 2, { timestampMs: 20 });
  assert.equal(ignored.ignored, true);
  assert.equal(bridge.getStats().lateSplashFramesIgnored, 1);
});

test('W189: CHROMA_ORDER and LOAD_OP constants, defaults, sanitization, and validation preserve invariants', () => {
  assert.ok(Object.isFrozen(CHROMA_ORDER));
  assert.ok(Object.isFrozen(LOAD_OP));
  assert.equal(CHROMA_ORDER.YUV, BINK_CHROMA_ORDER.YUV);
  assert.equal(CHROMA_ORDER.YVU, BINK_CHROMA_ORDER.YVU);
  assert.equal(LOAD_OP.CLEAR, BINK_LOAD_OP.CLEAR);
  assert.equal(LOAD_OP.LOAD, BINK_LOAD_OP.LOAD);

  assert.equal(sanitizeChromaOrder('yuv'), 'yuv');
  assert.equal(sanitizeChromaOrder(' YVU '), 'yvu');
  assert.equal(sanitizeChromaOrder('ycrcb'), null);
  assert.equal(sanitizeChromaOrder('C:\\video\\intro.bk2'), null);
  assert.equal(sanitizeChromaOrder(null), null);
  assert.equal(sanitizeChromaOrder(42), null);

  assert.equal(sanitizeLoadOp('clear'), 'clear');
  assert.equal(sanitizeLoadOp(' LOAD '), 'load');
  assert.equal(sanitizeLoadOp('discard'), null);
  assert.equal(sanitizeLoadOp('/tmp/frame.raw'), null);
  assert.equal(sanitizeLoadOp(undefined), null);
  assert.equal(sanitizeLoadOp({}), null);

  const device = createMockWebGpuDevice();
  const bridge = new SplashFrameBridge({ device, targetFps: 1000 });
  assert.equal(bridge.chromaOrder, 'yuv');
  assert.equal(bridge.loadOp, 'clear');
  assert.equal(bridge.zeroCopy, false);

  const initialStats = bridge.getStats();
  const initialState = bridge.getState();
  assert.equal(initialStats.chromaOrder, 'yuv');
  assert.equal(initialStats.loadOp, 'clear');
  assert.equal(initialStats.lastChromaOrder, null);
  assert.equal(initialStats.lastLoadOp, null);
  assert.equal(initialStats.yvuFramesForwarded, 0);
  assert.equal(initialStats.underlayLoadFramesPresented, 0);
  assert.equal(initialStats.zeroCopyFramesForwarded, 0);
  assert.equal(initialState.chromaOrder, 'yuv');
  assert.equal(initialState.loadOp, 'clear');
  assert.equal(initialState.lastChromaOrder, null);
  assert.equal(initialState.lastLoadOp, null);
  assert.equal(initialState.zeroCopy, false);

  // Invalid constructor / setter / frame / render options throw RangeError without mutating state.
  assert.throws(() => new SplashFrameBridge({ device, chromaOrder: 'ycrcb' }), RangeError);
  assert.throws(() => new SplashFrameBridge({ device, chromaOrder: 123 }), RangeError);
  assert.throws(() => new SplashFrameBridge({ device, loadOp: 'discard' }), RangeError);
  assert.throws(() => new SplashFrameBridge({ device, loadOp: 0 }), RangeError);

  assert.throws(() => bridge.setChromaOrder('nv21'), RangeError);
  assert.throws(() => bridge.setChromaOrder(''), RangeError);
  assert.throws(() => bridge.setChromaOrder(null), RangeError);
  assert.throws(() => bridge.setLoadOp('preserve'), RangeError);
  assert.throws(() => bridge.setLoadOp(''), RangeError);
  assert.throws(() => bridge.setLoadOp(undefined), RangeError);
  assert.equal(bridge.chromaOrder, 'yuv');
  assert.equal(bridge.loadOp, 'clear');

  const planes = makeYuv420Planes(8, 4);
  assert.throws(
    () =>
      bridge.pushRawFrame({
        width: 8,
        height: 4,
        format: PIXEL_FORMAT.YUV420P,
        planes,
        chromaOrder: 'invalid',
      }),
    RangeError
  );
  assert.throws(
    () =>
      bridge.pushRawFrame({
        width: 8,
        height: 4,
        format: PIXEL_FORMAT.YUV420P,
        planes,
        loadOp: 'keep',
      }),
    RangeError
  );
  assert.throws(
    () =>
      bridge.pushExternalImage(
        { width: 4, height: 4, data: makeRgbaFrame(4, 4) },
        { loadOp: 'bad' }
      ),
    RangeError
  );
  assert.throws(() => bridge.pushDiagnosticFrame(4, 4, 0, { loadOp: 'bad' }), RangeError);
  assert.equal(bridge.getStats().queueDepth, 0);

  bridge.pushDiagnosticFrame(4, 4, 0, { timestampMs: 0 });
  assert.throws(
    () => bridge.renderFrame(device, { kind: 'GPUTextureView' }, 64, 64, 0, { loadOp: 'bad' }),
    RangeError
  );
  assert.throws(
    () => bridge.renderFrame(device, { kind: 'GPUTextureView' }, 64, 64, 0, 'invalid-opts'),
    TypeError
  );
  // Queue was not consumed by invalid render options.
  assert.equal(bridge.getStats().queueDepth, 1);

  // Corrupted internal metadata is defensively sanitized in getStats() / getState().
  bridge._chromaOrder = 'C:\\secret\\bad';
  bridge._loadOp = '/etc/passwd';
  bridge._lastChromaOrder = 'https://invalid';
  bridge._lastLoadOp = 'discard';
  assert.equal(bridge.getStats().chromaOrder, 'yuv');
  assert.equal(bridge.getStats().loadOp, 'clear');
  assert.equal(bridge.getStats().lastChromaOrder, null);
  assert.equal(bridge.getStats().lastLoadOp, null);
  assert.equal(bridge.getState().chromaOrder, 'yuv');
  assert.equal(bridge.getState().loadOp, 'clear');
  assert.equal(bridge.getState().lastChromaOrder, null);
  assert.equal(bridge.getState().lastLoadOp, null);

  bridge.destroy();
});

test('W189: forwards chromaOrder (yuv/yvu) and underlay loadOp (clear/load) through BinkVideoPresenter', () => {
  const device = createMockWebGpuDevice();
  const firstPresentEvents = [];
  const bridge = new SplashFrameBridge({
    device,
    targetFps: 1000,
    chromaOrder: CHROMA_ORDER.YVU,
    loadOp: LOAD_OP.LOAD,
    onFirstPresent(info) {
      firstPresentEvents.push(info);
    },
  });

  const yuvPlanes = makeYuv420Planes(16, 8);
  const enq0 = bridge.pushRawFrame({
    width: 16,
    height: 8,
    format: PIXEL_FORMAT.YUV420P,
    planes: yuvPlanes,
    timestampMs: 0,
  });
  assert.equal(enq0.chromaOrder, 'yvu');
  assert.equal(enq0.loadOp, 'load');

  const r0 = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 320, 180, 0);
  assert.equal(r0.presented, true);
  assert.equal(r0.chromaOrder, 'yvu');
  assert.equal(r0.loadOp, 'load');
  assert.equal(r0.yvuFramesForwarded, 1);
  assert.equal(r0.underlayLoadFramesPresented, 1);
  assert.equal(bridge.presenter.chromaOrder, 'yvu');
  assert.equal(bridge.presenter.loadOp, 'load');

  // Uniform buffer carries SWAP_CHROMA flag (bit 1 = 2) for YVU.
  const u0 = device.writeBufferCalls[device.writeBufferCalls.length - 1].data;
  const u0View = new Uint32Array(u0.buffer, u0.byteOffset, u0.byteLength / 4);
  assert.equal(u0View[3] & UNIFORM_FLAG_SWAP_CHROMA, UNIFORM_FLAG_SWAP_CHROMA);

  // Underlay loadOp: 'load' omits clearValue on the render pass attachment.
  const pass0Attachment =
    device.createdEncoders[device.createdEncoders.length - 1].renderPasses[0].descriptor
      .colorAttachments[0];
  assert.equal(pass0Attachment.loadOp, 'load');
  assert.equal('clearValue' in pass0Attachment, false);

  // First-present telemetry includes chromaOrder and loadOp.
  assert.equal(firstPresentEvents.length, 1);
  assert.equal(firstPresentEvents[0].chromaOrder, 'yvu');
  assert.equal(firstPresentEvents[0].loadOp, 'load');

  // Per-frame override switches back to yuv + clear without altering bridge defaults.
  bridge.pushRawFrame({
    width: 16,
    height: 8,
    format: PIXEL_FORMAT.YUV420P,
    planes: yuvPlanes,
    chromaOrder: 'YUV',
    loadOp: 'CLEAR',
    timestampMs: 10,
  });
  const r1 = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 320, 180, 10);
  assert.equal(r1.presented, true);
  assert.equal(r1.chromaOrder, 'yuv');
  assert.equal(r1.loadOp, 'clear');
  assert.equal(bridge.presenter.chromaOrder, 'yuv');

  const u1 = device.writeBufferCalls[device.writeBufferCalls.length - 1].data;
  const u1View = new Uint32Array(u1.buffer, u1.byteOffset, u1.byteLength / 4);
  assert.equal(u1View[3] & UNIFORM_FLAG_SWAP_CHROMA, 0);

  const pass1Attachment =
    device.createdEncoders[device.createdEncoders.length - 1].renderPasses[0].descriptor
      .colorAttachments[0];
  assert.equal(pass1Attachment.loadOp, 'clear');
  assert.ok('clearValue' in pass1Attachment);

  // Subsequent frame omitting chromaOrder/loadOp restores bridge defaults (yvu + load),
  // and renderFrame options override loadOp & opacity per call.
  const nv12 = makeNv12Planes(16, 8);
  bridge.pushNv12Frame({
    width: 16,
    height: 8,
    yPlane: nv12.yPlane,
    uvPlane: nv12.uvPlane,
    timestampMs: 20,
  });
  const r2 = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 320, 180, {
    nowMs: 20,
    loadOp: 'clear',
    opacity: 0.5,
  });
  assert.equal(r2.presented, true);
  assert.equal(r2.chromaOrder, 'yvu');
  assert.equal(r2.loadOp, 'clear');
  assert.equal(bridge.presenter.chromaOrder, 'yvu');

  const u2 = device.writeBufferCalls[device.writeBufferCalls.length - 1].data;
  const u2u32 = new Uint32Array(u2.buffer, u2.byteOffset, u2.byteLength / 4);
  const u2f32 = new Float32Array(u2.buffer, u2.byteOffset, u2.byteLength / 4);
  assert.equal(u2u32[3] & UNIFORM_FLAG_SWAP_CHROMA, UNIFORM_FLAG_SWAP_CHROMA);
  assert.ok(Math.abs(u2f32[2] - 0.5) < 1e-6);

  // Setters synchronize with presenter and snapshots.
  bridge.setChromaOrder(CHROMA_ORDER.YUV);
  bridge.setLoadOp(LOAD_OP.CLEAR);
  assert.equal(bridge.chromaOrder, 'yuv');
  assert.equal(bridge.loadOp, 'clear');
  assert.equal(bridge.presenter.chromaOrder, 'yuv');
  assert.equal(bridge.presenter.loadOp, 'clear');

  const handoff = bridge.notifyGuestFirstPresent();
  assert.equal(handoff.chromaOrder, 'yuv');
  assert.equal(handoff.loadOp, 'clear');
  assert.equal(handoff.lastChromaOrder, 'yvu');
  assert.equal(handoff.lastLoadOp, 'clear');

  const stats = bridge.getStats();
  const state = bridge.getState();
  assert.equal(stats.yvuFramesForwarded, 2);
  assert.equal(stats.underlayLoadFramesPresented, 1);
  assert.equal(stats.lastChromaOrder, 'yvu');
  assert.equal(stats.lastLoadOp, 'clear');
  assert.equal(state.guestFirstPresent, true);
  assert.equal(state.splashSuppressed, true);
  assert.equal(state.lastChromaOrder, 'yvu');
  assert.equal(state.lastLoadOp, 'clear');

  // Reset clears per-session last* and counters while preserving configured defaults.
  bridge.reset();
  assert.equal(bridge.getStats().yvuFramesForwarded, 0);
  assert.equal(bridge.getStats().underlayLoadFramesPresented, 0);
  assert.equal(bridge.getStats().lastChromaOrder, null);
  assert.equal(bridge.getStats().lastLoadOp, null);
  assert.equal(bridge.getState().chromaOrder, 'yuv');
  assert.equal(bridge.getState().loadOp, 'clear');

  bridge.destroy();
});

test('W189: injected presenter chromaOrder and loadOp defaults are inherited when omitted', () => {
  const device = createMockWebGpuDevice();
  const presenter = new BinkVideoPresenter({
    device,
    targetFps: 1000,
    chromaOrder: BINK_CHROMA_ORDER.YVU,
    loadOp: BINK_LOAD_OP.LOAD,
  });
  const bridge = new SplashFrameBridge({ presenter });

  assert.equal(bridge.chromaOrder, 'yvu');
  assert.equal(bridge.loadOp, 'load');
  assert.equal(bridge.getState().chromaOrder, 'yvu');
  assert.equal(bridge.getState().loadOp, 'load');

  bridge.pushDiagnosticFrame(8, 4, 0, { timestampMs: 0 });
  const r0 = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 128, 72, 0);
  assert.equal(r0.presented, true);
  assert.equal(r0.loadOp, 'load');
  const att =
    device.createdEncoders[0].renderPasses[0].descriptor.colorAttachments[0];
  assert.equal(att.loadOp, 'load');
  assert.equal('clearValue' in att, false);

  bridge.destroy();
  presenter.destroy();
});

test('W189: 4K (3840x2160 Y + 1920x1080 Cr/Cb) Bink2Scarlett yvu preserves plane order and zero-copy typed-array views', () => {
  const width = 3840;
  const height = 2160;
  const uvW = 1920;
  const uvH = 1080;
  const yBytes = width * height;
  const chromaBytes = uvW * uvH;

  // Single contiguous 4K Y/Cr/Cb decode slab with distinct typed-array subviews.
  const slab = new ArrayBuffer(yBytes + chromaBytes * 2);
  const yView = new Uint8Array(slab, 0, yBytes);
  const crView = new Uint8Array(slab, yBytes, chromaBytes);
  const cbView = new Uint8Array(slab, yBytes + chromaBytes, chromaBytes);

  yView[0] = 0x11;
  yView[width - 1] = 0x22;
  yView[(height - 1) * width + width - 1] = 0x33;

  crView[0] = 0xa1;
  crView[uvW - 1] = 0xa2;
  crView[uvW] = 0xa3;
  crView[(uvH - 1) * uvW + uvW - 1] = 0xa4;

  cbView[0] = 0xb1;
  cbView[uvW - 1] = 0xb2;
  cbView[uvW] = 0xb3;
  cbView[(uvH - 1) * uvW + uvW - 1] = 0xb4;

  // 1) Verify exact typed-array view preservation (zero payload copies in bridge) via spy presenter.
  const uploadedPayloads = [];
  const presentCalls = [];
  const spyPresenter = {
    uploadFrame(frame) {
      return this.uploadYuv420Frame(frame);
    },
    uploadYuv420Frame(frame) {
      uploadedPayloads.push(frame);
      return { format: frame.format, chromaOrder: frame.chromaOrder };
    },
    presentFrame(_enc, _view, _w, _h, opts) {
      presentCalls.push(opts);
      return {
        presented: true,
        viewport: { x: 0, y: 0, width: 3840, height: 2160 },
        chromaOrder: 'yvu',
        loadOp: opts.loadOp,
      };
    },
  };
  const spyDevice = createMockWebGpuDevice();
  const spyBridge = new SplashFrameBridge({
    presenter: spyPresenter,
    zeroCopy: true,
    maxQueueDepth: 2,
  });

  // Push via named cr/cb planes with chromaOrder: 'yvu' and underlay loadOp: 'load'.
  spyBridge.pushRawFrame({
    width,
    height,
    format: PIXEL_FORMAT.YUV420P,
    planes: { y: yView, cr: crView, cb: cbView },
    chromaOrder: CHROMA_ORDER.YVU,
    loadOp: LOAD_OP.LOAD,
    timestampMs: 0,
  });

  const spyRes = spyBridge.renderFrame(spyDevice, { kind: 'GPUTextureView' }, 3840, 2160, 0);
  assert.equal(spyRes.presented, true);
  assert.equal(spyRes.chromaOrder, 'yvu');
  assert.equal(spyRes.loadOp, 'load');
  assert.equal(spyRes.zeroCopy, true);
  assert.equal(uploadedPayloads.length, 1);
  assert.equal(uploadedPayloads[0].chromaOrder, 'yvu');
  // Exact Uint8Array view references preserved (no copy, same backing ArrayBuffer & byteOffset).
  assert.equal(uploadedPayloads[0].y, yView);
  assert.equal(uploadedPayloads[0].u, crView);
  assert.equal(uploadedPayloads[0].v, cbView);
  assert.equal(uploadedPayloads[0].y.buffer, slab);
  assert.equal(uploadedPayloads[0].u.byteOffset, yBytes);
  assert.equal(uploadedPayloads[0].v.byteOffset, yBytes + chromaBytes);
  assert.equal(presentCalls[0].loadOp, 'load');

  // Also verify non-Uint8Array ArrayBufferView (Uint8ClampedArray) keeps zero-copy buffer view.
  const clampedCr = new Uint8ClampedArray(slab, yBytes, chromaBytes);
  const clampedCb = new Uint8ClampedArray(slab, yBytes + chromaBytes, chromaBytes);
  spyBridge.pushRawFrame({
    width,
    height,
    format: PIXEL_FORMAT.YUV420P,
    planes: { y: yView, u: clampedCr, v: clampedCb },
    chromaOrder: CHROMA_ORDER.YVU,
    timestampMs: 10,
  });
  spyBridge.renderFrame(spyDevice, { kind: 'GPUTextureView' }, 3840, 2160, 10);
  assert.equal(uploadedPayloads[1].u.buffer, slab);
  assert.equal(uploadedPayloads[1].u.byteOffset, yBytes);
  assert.equal(uploadedPayloads[1].v.buffer, slab);
  assert.equal(uploadedPayloads[1].v.byteOffset, yBytes + chromaBytes);
  spyBridge.destroy();

  // 2) End-to-end 4K Bink2Scarlett YVU through real BinkVideoPresenter (tight 1920 chroma).
  const device = createMockWebGpuDevice();
  const bridge = new SplashFrameBridge({
    device,
    targetFps: 1000,
    zeroCopy: true,
    chromaOrder: CHROMA_ORDER.YVU,
    loadOp: LOAD_OP.LOAD,
  });

  bridge.pushRawFrame({
    width,
    height,
    format: PIXEL_FORMAT.YUV420P,
    planes: { y: yView, cr: crView, cb: cbView },
    timestampMs: 0,
  });
  const res4k = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 3840, 2160, 0);
  assert.equal(res4k.presented, true);
  assert.equal(res4k.width, 3840);
  assert.equal(res4k.height, 2160);
  assert.equal(res4k.chromaOrder, 'yvu');
  assert.equal(res4k.loadOp, 'load');
  assert.equal(res4k.zeroCopy, true);

  const planeCalls = device.writeTextureCalls.filter(
    (c) => c.dest.texture.label !== 'playgta6-bink-presenter-dummy'
  );
  assert.equal(planeCalls.length, 3);
  const [yCall, crCall, cbCall] = planeCalls;

  // Y plane: exact caller Uint8Array passed zero-copy with 3840 bytesPerRow.
  assert.equal(yCall.rawData, yView);
  assert.equal(yCall.dataLayout.bytesPerRow, 3840);
  assert.equal(yCall.size.width, 3840);
  assert.equal(yCall.size.height, 2160);

  // Cr/Cb planes: plane order preserved (slot.u = Cr, slot.v = Cb) and 2048-aligned.
  assert.equal(crCall.dataLayout.bytesPerRow, 2048);
  assert.equal(cbCall.dataLayout.bytesPerRow, 2048);
  assert.equal(crCall.data[0], 0xa1);
  assert.equal(crCall.data[uvW - 1], 0xa2);
  assert.equal(crCall.data[2048], 0xa3);
  assert.equal(crCall.data[(uvH - 1) * 2048 + uvW - 1], 0xa4);
  assert.equal(cbCall.data[0], 0xb1);
  assert.equal(cbCall.data[uvW - 1], 0xb2);
  assert.equal(cbCall.data[2048], 0xb3);
  assert.equal(cbCall.data[(uvH - 1) * 2048 + uvW - 1], 0xb4);

  // Render pass uses underlay loadOp: 'load' without clearValue.
  const att4k = device.createdEncoders[0].renderPasses[0].descriptor.colorAttachments[0];
  assert.equal(att4k.loadOp, 'load');
  assert.equal('clearValue' in att4k, false);

  const stats4k = bridge.getStats();
  assert.equal(stats4k.lastFrameWidth, 3840);
  assert.equal(stats4k.lastFrameHeight, 2160);
  assert.equal(stats4k.chromaOrder, 'yvu');
  assert.equal(stats4k.lastChromaOrder, 'yvu');
  assert.equal(stats4k.loadOp, 'load');
  assert.equal(stats4k.lastLoadOp, 'load');
  assert.equal(stats4k.zeroCopyFramesForwarded, 1);
  assert.equal(stats4k.presenterStats.directZeroCopyUploads, 1);
  assert.equal(stats4k.presenterStats.chromaRepackUploads, 2);
  assert.equal(stats4k.presenterStats.scratchAllocations, 1);
  assert.equal(stats4k.presenterStats.chromaOrder, 'yvu');
  bridge.destroy();

  // 3) Pre-aligned 2048-byte chroma stride: all 3 4K planes pass end-to-end zero-copy with 0 scratch allocs.
  const alignedDevice = createMockWebGpuDevice();
  const alignedBridge = new SplashFrameBridge({
    device: alignedDevice,
    targetFps: 1000,
    chromaOrder: CHROMA_ORDER.YVU,
  });
  const crAligned = new Uint8Array(2048 * uvH);
  const cbAligned = new Uint8Array(2048 * uvH);
  crAligned[0] = 0xc1;
  cbAligned[0] = 0xd1;

  alignedBridge.pushRawFrame({
    width,
    height,
    format: PIXEL_FORMAT.YUV420P,
    planes: { y: yView, cr: crAligned, cb: cbAligned },
    yStride: 3840,
    crStride: 2048,
    cbStride: 2048,
    chromaOrder: CHROMA_ORDER.YVU,
    zeroCopy: true,
    timestampMs: 0,
  });
  const resAligned = alignedBridge.renderFrame(
    alignedDevice,
    { kind: 'GPUTextureView' },
    3840,
    2160,
    0
  );
  assert.equal(resAligned.presented, true);

  const alignedCalls = alignedDevice.writeTextureCalls.filter(
    (c) => c.dest.texture.label !== 'playgta6-bink-presenter-dummy'
  );
  assert.equal(alignedCalls.length, 3);
  assert.equal(alignedCalls[0].rawData, yView);
  assert.equal(alignedCalls[1].rawData, crAligned);
  assert.equal(alignedCalls[2].rawData, cbAligned);
  assert.equal(alignedBridge.getStats().presenterStats.directZeroCopyUploads, 3);
  assert.equal(alignedBridge.getStats().presenterStats.chromaRepackUploads, 0);
  assert.equal(alignedBridge.getStats().presenterStats.scratchAllocations, 0);
  alignedBridge.destroy();
});

function createMockImageBitmap(width, height, options = {}) {
  const bmp = {
    width,
    height,
    closeCalls: 0,
    closed: false,
    close() {
      this.closeCalls += 1;
      if (options.throwOnClose) {
        throw new Error(typeof options.throwOnClose === 'string' ? options.throwOnClose : 'ImageBitmap already closed');
      }
      this.closed = true;
    },
  };
  if (options.withData) {
    bmp.data = makeRgbaFrame(width, height, options.fill !== undefined ? options.fill : 0x66);
  }
  return bmp;
}

test('W279: pushExternalImage + renderFrame explicitly closes owned ImageBitmaps once, preserves extracted-copy caller ownership, and tracks framesBridged/bitmapsClosed', () => {
  const device = createMockWebGpuDevice();
  const bridge = new SplashFrameBridge({ device, targetFps: 1000 });

  // 1) ImageBitmap with RGBA data (extracted RGBA owns copy; caller retains bitmap unless closeOnRelease: true).
  const bmpWithData = createMockImageBitmap(8, 8, { withData: true, fill: 0x33 });
  bridge.pushExternalImage(bmpWithData, { timestampMs: 0 });
  assert.equal(bmpWithData.closeCalls, 0);
  assert.equal(bridge.getStats().retainedBitmaps, 1);

  const r0 = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 128, 128, 0);
  assert.equal(r0.presented, true);
  assert.equal(r0.framesBridged, 1);
  assert.equal(r0.bitmapsClosed, 0);
  assert.equal(bmpWithData.closeCalls, 0);
  assert.equal(bridge.getStats().retainedBitmaps, 0);

  const firstScratchTex = device.createdTextures.find((t) =>
    t.label && t.label.startsWith('playgta6-splash-external-')
  );
  assert.ok(firstScratchTex);
  assert.equal(firstScratchTex.destroyed, false);

  // 2) Copy-only ImageBitmap (real browser ImageBitmap with no .data; must close exactly once after renderFrame).
  const bmpCopyOnly = createMockImageBitmap(4, 4, { withData: false });
  bridge.pushExternalImage(bmpCopyOnly, { timestampMs: 10 });
  assert.equal(bridge.getStats().retainedBitmaps, 1);

  const r1 = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 128, 128, 10);
  assert.equal(r1.presented, true);
  assert.equal(r1.framesBridged, 2);
  assert.equal(r1.bitmapsClosed, 1);
  assert.equal(bmpCopyOnly.closeCalls, 1);
  assert.equal(bmpCopyOnly.closed, true);
  // Prior scratch texture was destroyed when replaced by the newer external frame.
  assert.equal(firstScratchTex.destroyed, true);

  // 3) ImageBitmap with data + explicit closeOnRelease: true closes after renderFrame.
  const bmpOwned = createMockImageBitmap(4, 4, { withData: true, fill: 0x55 });
  bridge.pushExternalImage(bmpOwned, { timestampMs: 20, closeOnRelease: true });
  const r2 = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 128, 128, 20);
  assert.equal(r2.presented, true);
  assert.equal(r2.framesBridged, 3);
  assert.equal(r2.bitmapsClosed, 2);
  assert.equal(bmpOwned.closeCalls, 1);
  assert.equal(bmpOwned.closed, true);

  const stats = bridge.getStats();
  assert.equal(stats.framesBridged, 3);
  assert.equal(stats.framesPresented, 3);
  assert.equal(stats.bitmapsClosed, 2);
  assert.equal(stats.retainedBitmaps, 0);
  assert.equal(stats.droppedStaleFrames, 0);

  bridge.dispose();
});

test('W279: replacing older frames via queue overflow or stale timestamps closes ImageBitmaps, nulls buffers, and increments droppedStaleFrames', () => {
  const device = createMockWebGpuDevice();
  const bridge = new SplashFrameBridge({ device, maxQueueDepth: 2, targetFps: 1000 });

  const bmp0 = createMockImageBitmap(4, 4, { withData: true, fill: 0x10 });
  const bmp1 = createMockImageBitmap(4, 4, { withData: false });
  const bmp2 = createMockImageBitmap(4, 4, { withData: false });

  bridge.pushExternalImage(bmp0, { frameIndex: 0, timestampMs: 0 });
  const entry0 = bridge._queue[0];
  bridge.pushExternalImage(bmp1, { frameIndex: 1, timestampMs: 10 });
  const entry1 = bridge._queue[1];

  // Overflow: evicts frame 0 immediately, closing bmp0 and nulling entry0 references.
  const enq2 = bridge.pushExternalImage(bmp2, { frameIndex: 2, timestampMs: 20 });
  assert.equal(enq2.droppedStaleFrames, 1);
  assert.equal(bmp0.closeCalls, 1);
  assert.equal(entry0.image, null);
  assert.equal(entry0.externalSource, null);
  assert.equal(entry0.planes, null);
  assert.equal(bmp1.closeCalls, 0);
  assert.equal(bmp2.closeCalls, 0);
  assert.equal(bridge.getStats().droppedStaleFrames, 1);
  assert.equal(bridge.getStats().bitmapsClosed, 1);
  assert.equal(bridge.getStats().retainedBitmaps, 2);

  // Stale due-frame skip: at nowMs=25, frame 1 is skipped as stale and frame 2 is presented.
  const res = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 64, 64, 25);
  assert.equal(res.presented, true);
  assert.equal(res.frameIndex, 2);
  assert.equal(bmp1.closeCalls, 1);
  assert.equal(bmp2.closeCalls, 1);
  assert.equal(entry1.image, null);
  assert.equal(entry1.externalSource, null);
  assert.equal(entry1.planes, null);

  const stats = bridge.getStats();
  assert.equal(stats.framesBridged, 1);
  assert.equal(stats.droppedFrames, 2);
  assert.equal(stats.droppedStaleFrames, 2);
  assert.equal(stats.bitmapsClosed, 3);
  assert.equal(stats.retainedBitmaps, 0);

  bridge.dispose();
});

test('W279: releaseSplashResources() releases queued bitmaps, pixel buffers, and scratch textures idempotently without double-close', () => {
  const device = createMockWebGpuDevice();
  const bridge = new SplashFrameBridge({ device, maxQueueDepth: 4, targetFps: 1000 });

  // Present one copy-only external frame so a scratch texture is allocated and bitmap is closed.
  const presentedBmp = createMockImageBitmap(4, 4, { withData: false });
  bridge.pushExternalImage(presentedBmp, { timestampMs: 0 });
  bridge.renderFrame(device, { kind: 'GPUTextureView' }, 64, 64, 0);
  assert.equal(presentedBmp.closeCalls, 1);

  const scratchTex = device.createdTextures.find(
    (t) => t.label && t.label.startsWith('playgta6-splash-external-')
  );
  assert.ok(scratchTex);
  assert.equal(scratchTex.destroyed, false);

  // Queue an ImageBitmap and a raw YUV frame without rendering them.
  const queuedBmp = createMockImageBitmap(8, 8, { withData: true });
  bridge.pushExternalImage(queuedBmp, { timestampMs: 10 });
  bridge.pushRawFrame({
    width: 8,
    height: 4,
    format: PIXEL_FORMAT.YUV420P,
    planes: makeYuv420Planes(8, 4),
    timestampMs: 20,
  });
  const queuedExtEntry = bridge._queue[0];
  const queuedYuvEntry = bridge._queue[1];

  const rel1 = bridge.releaseSplashResources();
  assert.equal(rel1.released, true);
  assert.equal(rel1.framesReleased, 2);
  assert.equal(rel1.bitmapsClosed, 1);
  assert.equal(rel1.buffersReleased, 4); // 1 rgba + 3 y/u/v planes
  assert.equal(rel1.texturesDestroyed, 1);
  assert.equal(rel1.releaseCount, 1);
  assert.equal(rel1.queueDepth, 0);

  assert.equal(queuedBmp.closeCalls, 1);
  assert.equal(scratchTex.destroyed, true);
  assert.equal(scratchTex.destroyCalls, 1);
  assert.equal(queuedExtEntry.image, null);
  assert.equal(queuedExtEntry.externalSource, null);
  assert.equal(queuedExtEntry.planes, null);
  assert.equal(queuedYuvEntry.planes, null);
  assert.equal(bridge.getStats().retainedBitmaps, 0);
  assert.equal(bridge.getStats().releaseCount, 1);
  assert.equal(bridge.getStats().bitmapsClosed, 2);

  // Repeated releaseSplashResources() is idempotent: no double-close, no throw, releaseCount unchanged.
  const rel2 = bridge.releaseSplashResources();
  assert.equal(rel2.released, false);
  assert.equal(rel2.framesReleased, 0);
  assert.equal(rel2.bitmapsClosed, 0);
  assert.equal(rel2.texturesDestroyed, 0);
  assert.equal(rel2.releaseCount, 1);
  assert.equal(queuedBmp.closeCalls, 1);
  assert.equal(presentedBmp.closeCalls, 1);
  assert.equal(scratchTex.destroyCalls, 1);

  // Pushing a new frame re-arms releaseSplashResources() for a second release cycle.
  const secondBmp = createMockImageBitmap(4, 4, { withData: false });
  bridge.pushExternalImage(secondBmp, { timestampMs: 30 });
  const rel3 = bridge.releaseSplashResources();
  assert.equal(rel3.released, true);
  assert.equal(rel3.framesReleased, 1);
  assert.equal(rel3.bitmapsClosed, 1);
  assert.equal(rel3.releaseCount, 2);
  assert.equal(secondBmp.closeCalls, 1);
  assert.equal(bridge.getStats().releaseCount, 2);
  assert.equal(bridge.getStats().bitmapsClosed, 3);

  bridge.dispose();
  // Calling releaseSplashResources() after dispose() never throws or double-closes.
  assert.doesNotThrow(() => bridge.releaseSplashResources());
  assert.equal(secondBmp.closeCalls, 1);
});

test('W279: notifyGuestFirstPresent handoff closes queued bitmaps, destroys scratch textures, and closes late post-handoff bitmaps', () => {
  const device = createMockWebGpuDevice();
  const bridge = new SplashFrameBridge({ device, maxQueueDepth: 4, targetFps: 1000 });

  const firstBmp = createMockImageBitmap(8, 8, { withData: false });
  bridge.pushExternalImage(firstBmp, { timestampMs: 0 });
  bridge.renderFrame(device, { kind: 'GPUTextureView' }, 128, 72, 0);
  assert.equal(firstBmp.closeCalls, 1);

  const scratchTex = device.createdTextures.find(
    (t) => t.label && t.label.startsWith('playgta6-splash-external-')
  );
  assert.ok(scratchTex);
  assert.equal(scratchTex.destroyed, false);

  const pendingBmp = createMockImageBitmap(8, 8, { withData: false });
  bridge.pushExternalImage(pendingBmp, { timestampMs: 10 });
  const pendingEntry = bridge._queue[0];

  const handoff = bridge.notifyGuestFirstPresent();
  assert.equal(handoff.guestFirstPresent, true);
  assert.equal(handoff.splashSuppressed, true);
  assert.equal(handoff.drainedFrames, 1);
  assert.equal(handoff.bitmapsClosed, 2);
  assert.equal(handoff.releaseCount, 1);
  assert.equal(pendingBmp.closeCalls, 1);
  assert.equal(pendingEntry.image, null);
  assert.equal(pendingEntry.externalSource, null);
  assert.equal(scratchTex.destroyed, true);
  assert.equal(bridge.getStats().retainedBitmaps, 0);

  // Late post-handoff ImageBitmap is ignored AND closed immediately to prevent leaks.
  const lateBmp = createMockImageBitmap(4, 4, { withData: true });
  const lateRes = bridge.pushExternalImage(lateBmp, { timestampMs: 20 });
  assert.equal(lateRes.ignored, true);
  assert.equal(lateRes.reason, 'suppressed');
  assert.equal(lateBmp.closeCalls, 1);
  assert.equal(bridge.getStats().bitmapsClosed, 3);
  assert.equal(bridge.getStats().retainedBitmaps, 0);

  // Idempotent second handoff & dispose never double-close.
  const handoff2 = bridge.notifyGuestFirstPresent();
  assert.equal(handoff2.drainedFrames, 0);
  assert.equal(handoff2.releaseCount, 1);
  bridge.dispose();
  bridge.dispose();
  assert.equal(firstBmp.closeCalls, 1);
  assert.equal(pendingBmp.closeCalls, 1);
  assert.equal(lateBmp.closeCalls, 1);
});

test('W279: dispose() is idempotent, catches already-closed bitmap errors safely, avoids double-close, and cleans up on renderFrame error', () => {
  const device = createMockWebGpuDevice();
  const bridge = new SplashFrameBridge({ device, maxQueueDepth: 6, targetFps: 1000 });

  const throwingBmp = createMockImageBitmap(4, 4, { withData: true, throwOnClose: 'already closed' });
  const sharedBmp = createMockImageBitmap(4, 4, { withData: false });
  const plainExternal = { width: 4, height: 4, data: makeRgbaFrame(4, 4) };
  const validBmp = createMockImageBitmap(4, 4, { withData: true });

  bridge.pushExternalImage(throwingBmp, { timestampMs: 0 });
  bridge.pushExternalImage(sharedBmp, { timestampMs: 10 });
  bridge.pushExternalImage(sharedBmp, { timestampMs: 20 });
  bridge.pushExternalImage(plainExternal, { timestampMs: 30 });
  bridge.pushExternalImage(validBmp, { timestampMs: 40 });

  assert.doesNotThrow(() => bridge.dispose());
  assert.doesNotThrow(() => bridge.dispose());
  assert.doesNotThrow(() => bridge.destroy());
  assert.doesNotThrow(() => bridge.releaseSplashResources());

  assert.equal(bridge.isDestroyed, true);
  assert.equal(throwingBmp.closeCalls, 1);
  assert.equal(sharedBmp.closeCalls, 1); // deduplicated via WeakSet
  assert.equal(validBmp.closeCalls, 1);
  const stats = bridge.getStats();
  assert.equal(stats.bitmapsClosed, 2); // sharedBmp + validBmp (throwingBmp failed close safely)
  assert.equal(stats.releaseCount, 1);
  assert.equal(stats.retainedBitmaps, 0);
  assert.equal(stats.queueDepth, 0);

  // Error during renderFrame upload still releases dequeued owned ImageBitmap via finally.
  const failingPresenter = {
    uploadRgbaFrame() {
      throw new Error('simulated GPU upload failure');
    },
    presentFrame() {
      return { presented: true };
    },
  };
  const errBridge = new SplashFrameBridge({ presenter: failingPresenter, maxQueueDepth: 2 });
  const errBmp = createMockImageBitmap(4, 4, { withData: false });
  errBridge.pushExternalImage(errBmp, { timestampMs: 0 });
  assert.throws(
    () => errBridge.renderFrame(device, { kind: 'GPUTextureView' }, 64, 64, 0),
    /simulated GPU upload failure/
  );
  assert.equal(errBmp.closeCalls, 1);
  assert.equal(errBridge.getStats().bitmapsClosed, 1);
  assert.equal(errBridge.getStats().framesBridged, 0);
  assert.equal(errBridge.getStats().retainedBitmaps, 0);
  errBridge.dispose();
});

test('W279: resetStats() cleanly resets lifecycle counters without dropping queue and getStats()/getState() sanitize corrupted counters', () => {
  const device = createMockWebGpuDevice();
  const bridge = new SplashFrameBridge({ device, maxQueueDepth: 2, targetFps: 1000 });

  const bmp0 = createMockImageBitmap(4, 4, { withData: false });
  const bmp1 = createMockImageBitmap(4, 4, { withData: true });
  const bmp2 = createMockImageBitmap(4, 4, { withData: true });

  bridge.pushExternalImage(bmp0, { frameIndex: 0, timestampMs: 0 });
  bridge.renderFrame(device, { kind: 'GPUTextureView' }, 64, 64, 0);
  bridge.releaseSplashResources();

  bridge.pushExternalImage(bmp1, { frameIndex: 1, timestampMs: 10 });
  bridge.pushExternalImage(bmp2, { frameIndex: 2, timestampMs: 20 });
  // Push one more to force overflow drop while leaving 2 frames queued.
  bridge.pushDiagnosticFrame(4, 4, 3, { timestampMs: 30 });

  let stats = bridge.getStats();
  assert.equal(stats.framesBridged, 1);
  assert.equal(stats.bitmapsClosed, 2); // bmp0 (rendered) + bmp1 (evicted)
  assert.equal(stats.releaseCount, 1);
  assert.equal(stats.droppedStaleFrames, 1);
  assert.equal(stats.queueDepth, 2);

  // resetStats() zeroes all counters while keeping queued frames and presenter intact.
  const afterReset = bridge.resetStats();
  assert.equal(afterReset.framesBridged, 0);
  assert.equal(afterReset.framesPresented, 0);
  assert.equal(afterReset.bitmapsClosed, 0);
  assert.equal(afterReset.releaseCount, 0);
  assert.equal(afterReset.droppedStaleFrames, 0);
  assert.equal(afterReset.droppedFrames, 0);
  assert.equal(afterReset.queueDepth, 2);
  assert.equal(afterReset.retainedBitmaps, 1); // bmp2 still queued
  assert.equal(afterReset.firstPresentSent, true); // session handoff flag preserved
  assert.ok(bridge.presenter);

  // Corrupted internal counters / source tokens are defensively sanitized.
  bridge._framesBridged = -42;
  bridge._bitmapsClosed = Number.NaN;
  bridge._releaseCount = 'C:\\Users\\secret\\splash.raw';
  bridge._droppedStaleFrames = Infinity;
  bridge._lastSource = '/etc/passwd';

  stats = bridge.getStats();
  const state = bridge.getState();
  assert.equal(stats.framesBridged, 0);
  assert.equal(stats.bitmapsClosed, 0);
  assert.equal(stats.releaseCount, 0);
  assert.equal(stats.droppedStaleFrames, 0);
  assert.equal(stats.lastSource, null);
  assert.equal(state.framesBridged, 0);
  assert.equal(state.bitmapsClosed, 0);
  assert.equal(state.releaseCount, 0);
  assert.equal(state.droppedStaleFrames, 0);
  assert.equal(state.lastSource, null);

  bridge.dispose();
  assert.equal(bmp2.closeCalls, 1);
  assert.throws(() => bridge.resetStats(), /destroyed/i);
});
