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
  generateDiagnosticSplashFrame,
  SplashFrameBridge,
} = bridgeMod;

const { BinkVideoPresenter, BINK_PIXEL_FORMAT } = binkMod;

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
