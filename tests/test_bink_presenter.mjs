import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const binkMod = require('../web/workers/bink_presenter.js');
const stagingMod = require('../web/workers/staging_pool.js');

const {
  WEBGPU_BYTES_PER_ROW_ALIGNMENT,
  UNIFORM_BUFFER_SIZE,
  UNIFORM_FLAG_FULL_RANGE,
  UNIFORM_FLAG_SWAP_CHROMA,
  TARGET_FPS_30_INTERVAL_MS,
  DEFAULT_BIND_GROUP_CACHE_CAPACITY,
  BINK_PIXEL_FORMAT,
  BINK_CHROMA_ORDER,
  BINK_LOAD_OP,
  YUV_COLOR_SPACE,
  YUV_COLOR_RANGE,
  YUV_MATRICES,
  YUV_FULL_RANGE_MATRICES,
  YUV_RANGE_MATRICES,
  TEXTURE_USAGE,
  BUFFER_USAGE,
  BINK_PRESENTER_WGSL,
  getYuvMatrix,
  prepareAlignedPlaneData,
  computeLetterboxViewport,
  packPresenterUniforms,
  BinkVideoPresenter,
} = binkMod;

const { GpuStagingPool } = stagingMod;

function createMockWebGpuDevice() {
  let nextTextureId = 1;
  let nextBufferId = 1;
  let nextBindGroupId = 1;
  let nextPipelineId = 1;
  let nextShaderId = 1;
  let nextSamplerId = 1;

  const createdTextures = [];
  const createdViews = [];
  const createdBuffers = [];
  const createdShaderModules = [];
  const createdBindGroupLayouts = [];
  const createdPipelineLayouts = [];
  const createdRenderPipelines = [];
  const createdSamplers = [];
  const createdBindGroups = [];
  const writeTextureCalls = [];
  const writeBufferCalls = [];

  const queue = {
    writeTexture(dest, data, dataLayout, size) {
      const bytes =
        data instanceof Uint8Array
          ? new Uint8Array(data)
          : new Uint8Array(data.buffer, data.byteOffset || 0, data.byteLength);
      writeTextureCalls.push({
        dest: { ...dest, origin: dest.origin ? { ...dest.origin } : undefined },
        rawDest: dest,
        rawOrigin: dest ? dest.origin : undefined,
        data: bytes,
        rawData: data,
        dataLayout: { ...dataLayout },
        rawDataLayout: dataLayout,
        size: { ...size },
        rawSize: size,
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
      writeBufferCalls.push({
        buffer,
        bufferOffset,
        data: bytes,
        rawData: data,
      });
    },
    onSubmittedWorkDone() {
      return Promise.resolve();
    },
  };

  const device = {
    queue,
    createdTextures,
    createdViews,
    createdBuffers,
    createdShaderModules,
    createdBindGroupLayouts,
    createdPipelineLayouts,
    createdRenderPipelines,
    createdSamplers,
    createdBindGroups,
    writeTextureCalls,
    writeBufferCalls,

    createTexture(desc) {
      const id = nextTextureId++;
      const view = { kind: 'GPUTextureView', id: `view-${id}`, textureId: id, label: desc.label };
      const texture = {
        kind: 'GPUTexture',
        id,
        label: desc.label,
        desc: { ...desc, size: { ...desc.size } },
        view,
        createViewCalls: 0,
        destroyed: false,
        destroyCalls: 0,
        createView() {
          this.createViewCalls += 1;
          createdViews.push(view);
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
  };

  return device;
}

function createMockCommandEncoder() {
  const ops = [];
  const renderPasses = [];
  const copyBufferToTextureCalls = [];

  return {
    ops,
    renderPasses,
    copyBufferToTextureCalls,
    copyBufferToTexture(source, destination, copySize) {
      const rec = {
        source: { ...source },
        destination: { ...destination },
        copySize: { ...copySize },
      };
      copyBufferToTextureCalls.push(rec);
      ops.push({ op: 'copyBufferToTexture', ...rec });
    },
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
  };
}

function assertNear(actual, expected, eps = 1e-6, message = '') {
  assert.ok(
    Math.abs(actual - expected) <= eps,
    `${message ? `${message}: ` : ''}expected ${actual} to be within ${eps} of ${expected}`
  );
}

test('UMD export attaches PlayGta6BinkPresenter on globalThis and freezes constants', () => {
  assert.equal(globalThis.PlayGta6BinkPresenter, binkMod);
  assert.equal(globalThis.PlayGta6BinkPresenter.BinkVideoPresenter, BinkVideoPresenter);
  assert.ok(Object.isFrozen(binkMod));
  assert.ok(Object.isFrozen(BINK_PIXEL_FORMAT));
  assert.ok(Object.isFrozen(YUV_COLOR_SPACE));
  assert.equal(BINK_PIXEL_FORMAT.RGBA8, 'rgba8');
  assert.equal(BINK_PIXEL_FORMAT.YUV420P, 'yuv420p');
  assert.equal(BINK_PIXEL_FORMAT.YUVA420P, 'yuva420p');
  assert.equal(YUV_COLOR_SPACE.BT709, 'bt709');
  assert.equal(YUV_COLOR_SPACE.BT601, 'bt601');
  assert.equal(WEBGPU_BYTES_PER_ROW_ALIGNMENT, 256);
  assert.equal(UNIFORM_BUFFER_SIZE, 96);
});

test('computeLetterboxViewport preserves exact aspect ratios across 16:9, 4:3, and ultrawide targets', () => {
  // 16:9 -> 16:9 (exact fill)
  const same = computeLetterboxViewport(1920, 1080, 1280, 720);
  assert.deepEqual(same, {
    x: 0,
    y: 0,
    width: 1280,
    height: 720,
    scaleX: 1,
    scaleY: 1,
  });

  // 16:9 -> 4:3 (horizontal fit, top & bottom black bars)
  const lb = computeLetterboxViewport(1920, 1080, 800, 600);
  assertNear(lb.x, 0);
  assertNear(lb.y, 75);
  assertNear(lb.width, 800);
  assertNear(lb.height, 450);
  assertNear(lb.scaleX, 1);
  assertNear(lb.scaleY, 0.75);
  assertNear(lb.width / lb.height, 16 / 9);

  // 4:3 -> 16:9 (vertical fit, left & right pillarboxes)
  const pb = computeLetterboxViewport(800, 600, 1920, 1080);
  assertNear(pb.x, 240);
  assertNear(pb.y, 0);
  assertNear(pb.width, 1440);
  assertNear(pb.height, 1080);
  assertNear(pb.scaleX, 0.75);
  assertNear(pb.scaleY, 1);
  assertNear(pb.width / pb.height, 4 / 3);

  // Ultrawide 21:9 (2560x1080) -> 16:9 (1920x1080): top/bottom letterbox
  const uwTo169 = computeLetterboxViewport(2560, 1080, 1920, 1080);
  assertNear(uwTo169.x, 0);
  assertNear(uwTo169.y, 135);
  assertNear(uwTo169.width, 1920);
  assertNear(uwTo169.height, 810);
  assertNear(uwTo169.scaleX, 1);
  assertNear(uwTo169.scaleY, 0.75);
  assertNear(uwTo169.width / uwTo169.height, 2560 / 1080);

  // 16:9 (1920x1080) -> Ultrawide 21:9 (2560x1080): left/right pillarbox
  const w169ToUw = computeLetterboxViewport(1920, 1080, 2560, 1080);
  assertNear(w169ToUw.x, 320);
  assertNear(w169ToUw.y, 0);
  assertNear(w169ToUw.width, 1920);
  assertNear(w169ToUw.height, 1080);
  assertNear(w169ToUw.scaleX, 0.75);
  assertNear(w169ToUw.scaleY, 1);
  assertNear(w169ToUw.width / w169ToUw.height, 16 / 9);

  // Invalid dimensions reject
  assert.throws(() => computeLetterboxViewport(0, 1080, 1920, 1080), RangeError);
  assert.throws(() => computeLetterboxViewport(1920, -1, 1920, 1080), RangeError);
  assert.throws(() => computeLetterboxViewport(1920, 1080, NaN, 1080), RangeError);
  assert.throws(() => computeLetterboxViewport(1920, 1080, 1920, Infinity), RangeError);
});

test('uploadRgbaFrame pads non-256-aligned widths, honors custom strideBytes, and double-buffers cleanly', () => {
  const device = createMockWebGpuDevice();
  const presenter = new BinkVideoPresenter({ device });

  // 10x3 RGBA8 -> tight row = 40 bytes -> padded to 256 bytes/row -> 768 total bytes.
  const width = 10;
  const height = 3;
  const rgba = new Uint8Array(width * height * 4);
  for (let row = 0; row < height; row += 1) {
    for (let x = 0; x < width * 4; x += 1) {
      rgba[row * width * 4 + x] = (row * 50 + x + 1) & 0xff;
    }
  }

  const r1 = presenter.uploadRgbaFrame(rgba, width, height);
  assert.equal(r1.format, BINK_PIXEL_FORMAT.RGBA8);
  assert.equal(r1.bytesPerRow, 256);
  assert.equal(r1.bytesUploaded, 256 * 3);
  assert.equal(r1.slotIndex, 0);
  assert.equal(r1.uploadedFrames, 1);

  assert.equal(device.createdTextures.length, 1);
  assert.equal(device.createdTextures[0].desc.format, 'rgba8unorm');
  assert.deepEqual(device.createdTextures[0].desc.size, {
    width: 10,
    height: 3,
    depthOrArrayLayers: 1,
  });
  assert.equal(
    device.createdTextures[0].desc.usage,
    TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST
  );

  // Verify row padding and zero-filled padding bytes.
  assert.equal(device.writeTextureCalls.length, 1);
  const upload1 = device.writeTextureCalls[0];
  assert.equal(upload1.dataLayout.bytesPerRow, 256);
  assert.equal(upload1.dataLayout.rowsPerImage, 3);
  assert.equal(upload1.data.byteLength, 768);
  for (let row = 0; row < height; row += 1) {
    const srcStart = row * 40;
    const dstStart = row * 256;
    assert.deepEqual(
      upload1.data.subarray(dstStart, dstStart + 40),
      rgba.subarray(srcStart, srcStart + 40)
    );
    for (let p = 40; p < 256; p += 1) {
      assert.equal(upload1.data[dstStart + p], 0);
    }
  }

  // Second frame with custom strideBytes=48 allocates slot 1 (double-buffered).
  const strided = new Uint8Array(48 * height);
  strided.fill(0xee);
  for (let row = 0; row < height; row += 1) {
    strided.set(rgba.subarray(row * 40, row * 40 + 40), row * 48);
  }
  const r2 = presenter.uploadRgbaFrame(strided, width, height, { strideBytes: 48 });
  assert.equal(r2.slotIndex, 1);
  assert.equal(r2.uploadedFrames, 2);
  assert.equal(device.createdTextures.length, 2);
  // Ensure stride sentinel bytes (0xee) did not bleed into the 256-byte padded row tail.
  assert.equal(device.writeTextureCalls[1].data[40], 0);
  assert.equal(device.writeTextureCalls[1].data[256 + 39], rgba[40 + 39]);

  // Third frame reuses slot 0 with zero new texture allocations.
  const r3 = presenter.uploadRgbaFrame(rgba, width, height);
  assert.equal(r3.slotIndex, 0);
  assert.equal(r3.uploadedFrames, 3);
  assert.equal(device.createdTextures.length, 2);
});

test('uploadYuv420Frame allocates r8unorm Y/U/V planes with 256-byte row padding and odd dimensions', () => {
  const device = createMockWebGpuDevice();
  const presenter = new BinkVideoPresenter({ device, colorSpace: YUV_COLOR_SPACE.BT709 });

  // Non-256-aligned & odd dimensions: 15x9 -> Y=15x9, U=8x5, V=8x5
  const width = 15;
  const height = 9;
  const uvW = 8;
  const uvH = 5;

  const y = new Uint8Array(width * height).map((_, i) => (i + 16) & 0xff);
  const u = new Uint8Array(uvW * uvH).map((_, i) => (i + 100) & 0xff);
  const v = new Uint8Array(uvW * uvH).map((_, i) => (i + 150) & 0xff);

  const res = presenter.uploadYuv420Frame({
    y,
    u,
    v,
    width,
    height,
    colorSpace: YUV_COLOR_SPACE.BT601,
  });

  assert.equal(res.format, BINK_PIXEL_FORMAT.YUV420P);
  assert.equal(res.colorSpace, YUV_COLOR_SPACE.BT601);
  assert.equal(res.uvWidth, 8);
  assert.equal(res.uvHeight, 5);
  assert.equal(res.yBytesPerRow, 256);
  assert.equal(res.uvBytesPerRow, 256);
  assert.equal(res.aBytesPerRow, null);
  assert.equal(res.bytesUploaded, 256 * 9 + 256 * 5 + 256 * 5);

  assert.equal(device.createdTextures.length, 3);
  assert.deepEqual(
    device.createdTextures.map((t) => ({
      format: t.desc.format,
      w: t.desc.size.width,
      h: t.desc.size.height,
    })),
    [
      { format: 'r8unorm', w: 15, h: 9 },
      { format: 'r8unorm', w: 8, h: 5 },
      { format: 'r8unorm', w: 8, h: 5 },
    ]
  );

  // Verify Y, U, V writeTexture row alignment and content
  assert.equal(device.writeTextureCalls.length, 3);
  const [yCall, uCall, vCall] = device.writeTextureCalls;
  assert.equal(yCall.dataLayout.bytesPerRow, 256);
  assert.deepEqual(yCall.data.subarray(256, 256 + width), y.subarray(width, width * 2));
  assert.equal(yCall.data[256 + width], 0);
  assert.equal(uCall.dataLayout.bytesPerRow, 256);
  assert.deepEqual(uCall.data.subarray(256, 256 + uvW), u.subarray(uvW, uvW * 2));
  assert.equal(vCall.dataLayout.bytesPerRow, 256);
  assert.deepEqual(vCall.data.subarray(256 * 4, 256 * 4 + uvW), v.subarray(uvW * 4, uvW * 5));
});

test('uploadYuv420Frame with alpha plane switches to YUVA420P and binds alpha texture', () => {
  const device = createMockWebGpuDevice();
  const presenter = new BinkVideoPresenter({ device });

  const width = 32;
  const height = 16;
  const y = new Uint8Array(width * height).fill(200);
  const u = new Uint8Array((width / 2) * (height / 2)).fill(128);
  const v = new Uint8Array((width / 2) * (height / 2)).fill(128);
  const a = new Uint8Array(width * height).map((_, i) => i & 0xff);

  const res = presenter.uploadYuv420Frame({
    y,
    u,
    v,
    a,
    width,
    height,
    aStride: width,
  });

  assert.equal(res.format, BINK_PIXEL_FORMAT.YUVA420P);
  assert.equal(res.aBytesPerRow, 256);
  assert.equal(device.createdTextures.length, 4);
  assert.equal(device.createdTextures[3].desc.format, 'r8unorm');
  assert.deepEqual(device.createdTextures[3].desc.size, {
    width: 32,
    height: 16,
    depthOrArrayLayers: 1,
  });

  const encoder = createMockCommandEncoder();
  const targetView = { kind: 'GPUTextureView', id: 'swap-0' };
  presenter.presentFrame(encoder, targetView, 640, 320);

  // Binding 5 must reference the alpha plane texture view (not the dummy texture view).
  const bg = device.createdBindGroups[0];
  const entry5 = bg.desc.entries.find((e) => e.binding === 5);
  assert.equal(entry5.resource, device.createdTextures[3].view);

  // Uniform buffer mode lane must be 2 (YUVA420P).
  const uniformWrite = device.writeBufferCalls[device.writeBufferCalls.length - 1];
  const u32 = new Uint32Array(
    uniformWrite.data.buffer,
    uniformWrite.data.byteOffset,
    uniformWrite.data.byteLength / 4
  );
  assert.equal(u32[0], 2);
});

test('packPresenterUniforms accurately packs BT.709 and BT.601 matrices and mode metadata', () => {
  const bt709Bytes = packPresenterUniforms({
    pixelFormat: BINK_PIXEL_FORMAT.YUV420P,
    colorSpace: YUV_COLOR_SPACE.BT709,
    opacity: 0.75,
    uvTransform: [0.5, 0.5, 0.25, 0.25],
  });
  assert.equal(bt709Bytes.byteLength, UNIFORM_BUFFER_SIZE);

  const u32_709 = new Uint32Array(bt709Bytes.buffer);
  const f32_709 = new Float32Array(bt709Bytes.buffer);
  assert.equal(u32_709[0], 1); // YUV420P
  assert.equal(u32_709[1], 0); // BT709
  assertNear(f32_709[2], 0.75);
  assertNear(f32_709[4], 0.5);
  assertNear(f32_709[5], 0.5);
  assertNear(f32_709[6], 0.25);
  assertNear(f32_709[7], 0.25);
  assertNear(f32_709[8], YUV_MATRICES.bt709.row0[0], 1e-5);
  assertNear(f32_709[10], YUV_MATRICES.bt709.row0[2], 1e-5);
  assertNear(f32_709[13], YUV_MATRICES.bt709.row1[1], 1e-5);
  assertNear(f32_709[17], YUV_MATRICES.bt709.row2[1], 1e-5);
  assertNear(f32_709[20], 16 / 255, 1e-5);
  assertNear(f32_709[21], 128 / 255, 1e-5);

  const bt601Bytes = packPresenterUniforms({
    pixelFormat: BINK_PIXEL_FORMAT.YUVA420P,
    colorSpace: YUV_COLOR_SPACE.BT601,
    opacity: 1.0,
  });
  const u32_601 = new Uint32Array(bt601Bytes.buffer);
  const f32_601 = new Float32Array(bt601Bytes.buffer);
  assert.equal(u32_601[0], 2); // YUVA420P
  assert.equal(u32_601[1], 1); // BT601
  assertNear(f32_601[10], YUV_MATRICES.bt601.row0[2], 1e-5);
  assertNear(f32_601[13], YUV_MATRICES.bt601.row1[1], 1e-5);
  assert.notEqual(f32_709[10], f32_601[10]);
});

test('stagingPool integration stages uploads via GpuStagingPool both deferred and with explicit encoder', () => {
  const device = createMockWebGpuDevice();
  const stagingPool = new GpuStagingPool({
    device,
    chunkSize: 64 * 1024,
    maxPoolBytes: 256 * 1024,
    mode: 'writeBuffer',
  });
  const presenter = new BinkVideoPresenter({ device, stagingPool });

  stagingPool.beginFrame(0);
  const rgba = new Uint8Array(20 * 4 * 4).fill(0xab);

  // Deferred staging (no commandEncoder passed to uploadRgbaFrame -> flushed in presentFrame)
  presenter.uploadRgbaFrame(rgba, 20, 4);
  assert.equal(presenter.getStats().pendingUploads, 1);
  assert.equal(presenter.getStats().usesStagingPool, true);

  const encoder0 = createMockCommandEncoder();
  const targetView = { kind: 'GPUTextureView', id: 'swap-0' };
  presenter.presentFrame(encoder0, targetView, 1280, 720);

  assert.equal(presenter.getStats().pendingUploads, 0);
  assert.equal(encoder0.copyBufferToTextureCalls.length, 1);
  assert.equal(encoder0.copyBufferToTextureCalls[0].source.bytesPerRow, 256);
  // Copy MUST be recorded before beginRenderPass.
  const copyOpIdx = encoder0.ops.findIndex((o) => o.op === 'copyBufferToTexture');
  const passOpIdx = encoder0.ops.findIndex((o) => o.op === 'beginRenderPass');
  assert.ok(copyOpIdx >= 0 && copyOpIdx < passOpIdx);
  stagingPool.endFrame();

  // Explicit commandEncoder passed directly to uploadYuv420Frame
  stagingPool.beginFrame(1);
  const encoder1 = createMockCommandEncoder();
  const y = new Uint8Array(16 * 8).fill(180);
  const u = new Uint8Array(8 * 4).fill(120);
  const v = new Uint8Array(8 * 4).fill(130);
  presenter.uploadYuv420Frame({ y, u, v, width: 16, height: 8, commandEncoder: encoder1 });

  assert.equal(presenter.getStats().pendingUploads, 0);
  assert.equal(encoder1.copyBufferToTextureCalls.length, 3);
  assert.ok(encoder1.copyBufferToTextureCalls.every((c) => c.source.bytesPerRow === 256));
  stagingPool.endFrame();

  presenter.destroy();
  stagingPool.destroy();
});

test('dynamic resolution resize and format change destroy all old slot textures cleanly', () => {
  const device = createMockWebGpuDevice();
  const presenter = new BinkVideoPresenter({ device });

  // Populate both double-buffer slots at 64x32 RGBA8
  const frameSmall = new Uint8Array(64 * 32 * 4);
  presenter.uploadRgbaFrame(frameSmall, 64, 32);
  presenter.uploadRgbaFrame(frameSmall, 64, 32);
  assert.equal(device.createdTextures.length, 2);
  const oldSlot0 = device.createdTextures[0];
  const oldSlot1 = device.createdTextures[1];
  assert.equal(oldSlot0.destroyed, false);
  assert.equal(oldSlot1.destroyed, false);

  // Resize to 128x64 RGBA8 -> both old slot textures must be destroyed!
  const frameLarge = new Uint8Array(128 * 64 * 4);
  const rLarge = presenter.uploadRgbaFrame(frameLarge, 128, 64);
  assert.equal(rLarge.slotIndex, 0);
  assert.equal(oldSlot0.destroyed, true);
  assert.equal(oldSlot1.destroyed, true);
  assert.equal(presenter.getStats().resizeCount, 1);
  assert.equal(device.createdTextures.length, 3);
  const newRgbaTex = device.createdTextures[2];
  assert.equal(newRgbaTex.destroyed, false);
  assert.deepEqual(newRgbaTex.desc.size, { width: 128, height: 64, depthOrArrayLayers: 1 });

  // Switch format to YUV420P at same 128x64 -> destroys old RGBA texture, allocates 3 r8unorm planes
  presenter.uploadYuv420Frame({
    y: new Uint8Array(128 * 64),
    u: new Uint8Array(64 * 32),
    v: new Uint8Array(64 * 32),
    width: 128,
    height: 64,
  });
  assert.equal(newRgbaTex.destroyed, true);
  assert.equal(presenter.getStats().resizeCount, 2);
  assert.equal(device.createdTextures.length, 6);
});

test('presentFrame lazily builds pipeline once, encodes letterbox viewport and draw(3, 1, 0, 0), and caches bind groups', () => {
  const device = createMockWebGpuDevice();
  const presenter = new BinkVideoPresenter({
    device,
    presentationFormat: 'rgba8unorm',
  });

  // Constructor must not build pipeline resources eagerly.
  assert.equal(device.createdRenderPipelines.length, 0);
  assert.equal(device.createdShaderModules.length, 0);
  assert.equal(presenter.getStats().pipelineReady, false);

  // Presenting before any frame clears target safely without issuing draw().
  const emptyEnc = createMockCommandEncoder();
  const targetView = { kind: 'GPUTextureView', id: 'swap-view-0' };
  const emptyRes = presenter.presentFrame(emptyEnc, targetView, 800, 600);
  assert.equal(emptyRes.presented, false);
  assert.equal(emptyEnc.renderPasses.length, 1);
  assert.equal(emptyEnc.renderPasses[0].draws.length, 0);
  assert.equal(emptyEnc.renderPasses[0].ended, true);
  assert.equal(device.createdRenderPipelines.length, 1);
  assert.equal(
    device.createdRenderPipelines[0].desc.fragment.targets[0].format,
    'rgba8unorm'
  );

  // Upload 16:9 frame (160x90) and present onto 4:3 target (800x600) -> letterboxed 800x450 at y=75.
  const rgba = new Uint8Array(160 * 90 * 4).fill(255);
  presenter.uploadRgbaFrame(rgba, 160, 90);
  const enc1 = createMockCommandEncoder();
  const res1 = presenter.presentFrame(enc1, targetView, 800, 600, {
    clearColor: { r: 0.02, g: 0.03, b: 0.04, a: 1 },
  });

  assert.equal(res1.presented, true);
  assert.equal(res1.presentedFrames, 1);
  assert.equal(enc1.renderPasses.length, 1);
  const pass1 = enc1.renderPasses[0];
  assert.deepEqual(pass1.descriptor.colorAttachments[0].clearValue, {
    r: 0.02,
    g: 0.03,
    b: 0.04,
    a: 1,
  });
  assert.equal(pass1.pipeline, device.createdRenderPipelines[0]);
  assertNear(pass1.viewport.x, 0);
  assertNear(pass1.viewport.y, 75);
  assertNear(pass1.viewport.width, 800);
  assertNear(pass1.viewport.height, 450);
  assert.equal(pass1.viewport.minDepth, 0);
  assert.equal(pass1.viewport.maxDepth, 1);
  assert.equal(pass1.bindGroups.length, 1);
  assert.equal(pass1.bindGroups[0].index, 0);
  assert.deepEqual(pass1.draws, [
    { vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 },
  ]);
  assert.equal(pass1.ended, true);

  // Present slot 1, then slot 0 again -> only 2 bind groups created total, 1 pipeline total.
  presenter.uploadRgbaFrame(rgba, 160, 90);
  presenter.presentFrame(createMockCommandEncoder(), targetView, 800, 600);
  presenter.uploadRgbaFrame(rgba, 160, 90);
  presenter.presentFrame(createMockCommandEncoder(), targetView, 800, 600);

  assert.equal(device.createdRenderPipelines.length, 1);
  assert.equal(device.createdShaderModules.length, 1);
  assert.equal(device.createdBindGroupLayouts.length, 1);
  assert.equal(device.createdSamplers.length, 1);
  assert.equal(device.createdBindGroups.length, 2);
  assert.equal(presenter.getStats().presentedFrames, 3);
});

test('shouldAdvance enforces FPS pacing, pause/resume, looping, and skip state', () => {
  const device = createMockWebGpuDevice();
  const presenter = new BinkVideoPresenter({ device, targetFps: 30 });

  assert.equal(presenter.isPaused, false);
  assert.equal(presenter.isLooping, false);
  assert.equal(presenter.isSkipped, false);

  // 30 fps -> interval = 33.3333... ms
  assert.equal(presenter.shouldAdvance(0), true);
  assert.equal(presenter.shouldAdvance(16), false);
  assert.equal(presenter.shouldAdvance(33), false);
  assert.equal(presenter.shouldAdvance(34), true);
  assert.equal(presenter.shouldAdvance(50), false);
  assert.equal(presenter.shouldAdvance(67), true);
  assert.equal(presenter.shouldAdvance(100), true);

  // Large hitch (> 2 intervals) advances once and snaps anchor without catch-up burst
  assert.equal(presenter.shouldAdvance(500), true);
  assert.equal(presenter.shouldAdvance(510), false);
  assert.equal(presenter.shouldAdvance(534), true);

  // Pause prevents advancement; resume allows immediate next frame
  presenter.setPaused(true);
  assert.equal(presenter.isPaused, true);
  assert.equal(presenter.shouldAdvance(600), false);
  presenter.setPaused(false);
  assert.equal(presenter.isPaused, false);
  assert.equal(presenter.shouldAdvance(600), true);

  // Change FPS to 60 fps (16.6667ms interval)
  presenter.setTargetFps(60);
  assert.equal(presenter.shouldAdvance(610), false);
  assert.equal(presenter.shouldAdvance(617), true);

  // Looping flag updates stats
  presenter.setLooping(true);
  assert.equal(presenter.isLooping, true);
  assert.equal(presenter.getStats().looping, true);

  // Skip stops advancement
  presenter.skip();
  assert.equal(presenter.isSkipped, true);
  assert.equal(presenter.shouldAdvance(1000), false);
});

test('BINK_PRESENTER_WGSL passes structural WGSL syntax and binding sanity checks', () => {
  assert.equal(typeof BINK_PRESENTER_WGSL, 'string');
  assert.ok(BINK_PRESENTER_WGSL.length > 200);

  // Required entry points & structs
  assert.match(BINK_PRESENTER_WGSL, /struct\s+PresenterUniforms\s*\{/);
  assert.match(BINK_PRESENTER_WGSL, /@vertex\s+fn\s+vs_main\s*\(/);
  assert.match(BINK_PRESENTER_WGSL, /@fragment\s+fn\s+fs_main\s*\(/);
  assert.match(BINK_PRESENTER_WGSL, /@builtin\(vertex_index\)/);
  assert.match(BINK_PRESENTER_WGSL, /@builtin\(position\)/);

  // Required bindings 0..5
  for (let b = 0; b <= 5; b += 1) {
    assert.match(
      BINK_PRESENTER_WGSL,
      new RegExp(`@group\\(0\\)\\s*@binding\\(${b}\\)`),
      `missing @group(0) @binding(${b})`
    );
  }

  // Supports RGBA8 (0u), YUV420P (1u), YUVA420P (2u), and uv_transform
  assert.match(BINK_PRESENTER_WGSL, /uniforms\.mode\s*==\s*0u/);
  assert.match(BINK_PRESENTER_WGSL, /uniforms\.mode\s*==\s*2u/);
  assert.match(BINK_PRESENTER_WGSL, /uniforms\.uv_transform/);
  assert.match(BINK_PRESENTER_WGSL, /uniforms\.yuv_row0/);
  assert.match(BINK_PRESENTER_WGSL, /uniforms\.yuv_row1/);
  assert.match(BINK_PRESENTER_WGSL, /uniforms\.yuv_row2/);
  assert.match(BINK_PRESENTER_WGSL, /uniforms\.yuv_offset/);

  // Balanced braces and parentheses
  let braceDepth = 0;
  let parenDepth = 0;
  for (const ch of BINK_PRESENTER_WGSL) {
    if (ch === '{') braceDepth += 1;
    else if (ch === '}') braceDepth -= 1;
    else if (ch === '(') parenDepth += 1;
    else if (ch === ')') parenDepth -= 1;
    assert.ok(braceDepth >= 0, 'unexpected closing brace in WGSL');
    assert.ok(parenDepth >= 0, 'unexpected closing paren in WGSL');
  }
  assert.equal(braceDepth, 0, 'unclosed brace in WGSL');
  assert.equal(parenDepth, 0, 'unclosed paren in WGSL');
});

test('destroy() destroys all textures and uniform buffers idempotently and blocks further use', () => {
  const device = createMockWebGpuDevice();
  const presenter = new BinkVideoPresenter({ device });

  const width = 32;
  const height = 16;
  const y = new Uint8Array(width * height);
  const u = new Uint8Array((width / 2) * (height / 2));
  const v = new Uint8Array((width / 2) * (height / 2));
  const a = new Uint8Array(width * height);

  // Fill both slots (8 plane textures) + present (1 dummy texture + 1 uniform buffer)
  presenter.uploadYuv420Frame({ y, u, v, a, width, height });
  presenter.uploadYuv420Frame({ y, u, v, a, width, height });
  presenter.presentFrame(createMockCommandEncoder(), { kind: 'GPUTextureView' }, 640, 360);

  assert.equal(device.createdTextures.length, 9);
  assert.equal(device.createdBuffers.length, 1);
  assert.equal(device.createdBuffers[0].desc.usage, BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST);

  presenter.destroy();
  assert.equal(presenter.isDestroyed, true);
  assert.equal(presenter.getStats().destroyed, true);
  assert.equal(presenter.shouldAdvance(100), false);

  for (const tex of device.createdTextures) {
    assert.equal(tex.destroyed, true);
    assert.equal(tex.destroyCalls, 1);
  }
  assert.equal(device.createdBuffers[0].destroyed, true);
  assert.equal(device.createdBuffers[0].destroyCalls, 1);

  // Second destroy() is a clean idempotent no-op
  presenter.destroy();
  for (const tex of device.createdTextures) {
    assert.equal(tex.destroyCalls, 1);
  }
  assert.equal(device.createdBuffers[0].destroyCalls, 1);

  // Operations after destroy throw clearly
  assert.throws(() => presenter.uploadRgbaFrame(new Uint8Array(16), 2, 2), /destroyed/i);
  assert.throws(
    () => presenter.uploadYuv420Frame({ y, u, v, width, height }),
    /destroyed/i
  );
  assert.throws(
    () => presenter.presentFrame(createMockCommandEncoder(), { kind: 'GPUTextureView' }, 640, 360),
    /destroyed/i
  );
});

test('W115: 5 consecutive YUV420p frames reuse plane textures, bind groups, and scratch buffer', () => {
  // Single-slot mode: literally 3 plane textures + 1 dummy texture + 1 bind group across 5 frames.
  const deviceSingle = createMockWebGpuDevice();
  const presenterSingle = new BinkVideoPresenter({
    device: deviceSingle,
    doubleBuffer: false,
  });

  const width = 64;
  const height = 36;
  const uvW = width / 2;
  const uvH = height / 2;
  const targetView = { kind: 'GPUTextureView', id: 'swap-yuv-single' };

  for (let i = 0; i < 5; i += 1) {
    const y = new Uint8Array(width * height).fill((40 + i * 20) & 0xff);
    const u = new Uint8Array(uvW * uvH).fill(128);
    const v = new Uint8Array(uvW * uvH).fill(128);
    presenterSingle.uploadYuv420Frame({ y, u, v, width, height });
    const res = presenterSingle.presentFrame(createMockCommandEncoder(), targetView, 1280, 720);
    assert.equal(res.presented, true);
    assert.equal(res.framesPresented, i + 1);
  }

  const singleStats = presenterSingle.getStats();
  assert.equal(singleStats.framesPresented, 5);
  assert.equal(singleStats.presentedFrames, 5);
  assert.equal(singleStats.planeReallocations, 1);
  assert.equal(singleStats.resizeCount, 0);
  assert.equal(singleStats.scratchAllocations, 1);
  assert.equal(singleStats.droppedFrames, 0);
  assert.equal(singleStats.lastFrameWidth, width);
  assert.equal(singleStats.lastFrameHeight, height);
  // 3 plane textures (Y/U/V) + 1 neutral dummy texture = 4 total textures, 1 bind group.
  assert.equal(deviceSingle.createdTextures.length, 4);
  assert.equal(deviceSingle.createdBindGroups.length, 1);
  assert.equal(deviceSingle.createdRenderPipelines.length, 1);

  // Default double-buffered mode: 2 slots warmed once (6 plane + 1 dummy = 7), 2 bind groups, planeReallocations === 1.
  const deviceDouble = createMockWebGpuDevice();
  const presenterDouble = new BinkVideoPresenter({ device: deviceDouble });

  for (let i = 0; i < 5; i += 1) {
    const y = new Uint8Array(width * height).fill((50 + i * 10) & 0xff);
    const u = new Uint8Array(uvW * uvH).fill(120);
    const v = new Uint8Array(uvW * uvH).fill(136);
    presenterDouble.uploadYuv420Frame({ y, u, v, width, height });
    presenterDouble.presentFrame(createMockCommandEncoder(), targetView, 1280, 720);
  }

  const doubleStats = presenterDouble.getStats();
  assert.equal(doubleStats.framesPresented, 5);
  assert.equal(doubleStats.planeReallocations, 1);
  assert.equal(doubleStats.scratchAllocations, 1);
  assert.equal(doubleStats.droppedFrames, 0);
  assert.equal(doubleStats.lastFrameWidth, width);
  assert.equal(doubleStats.lastFrameHeight, height);
  assert.equal(deviceDouble.createdTextures.length, 7);
  assert.equal(deviceDouble.createdBindGroups.length, 2);

  presenterSingle.destroy();
  presenterDouble.destroy();
});

test('W115: 5 consecutive NV12 frames reuse r8unorm Y + rg8unorm UV planes and bind groups, and reallocate on resolution change', () => {
  const device = createMockWebGpuDevice();
  const presenter = new BinkVideoPresenter({ device, doubleBuffer: false });
  const targetView = { kind: 'GPUTextureView', id: 'swap-nv12' };

  assert.equal(BINK_PIXEL_FORMAT.NV12, 'nv12');
  assert.match(BINK_PRESENTER_WGSL, /uniforms\.mode\s*==\s*3u/);

  const w1 = 48;
  const h1 = 24;
  const uvW1 = 24;
  const uvH1 = 12;

  for (let i = 0; i < 5; i += 1) {
    const y = new Uint8Array(w1 * h1).fill((30 + i * 15) & 0xff);
    const uv = new Uint8Array(uvW1 * uvH1 * 2).fill((100 + i) & 0xff);
    const up =
      i === 0
        ? presenter.uploadNv12Frame({ y, uv, width: w1, height: h1 })
        : presenter.uploadFrame({ format: BINK_PIXEL_FORMAT.NV12, y, uv, width: w1, height: h1 });
    assert.equal(up.format, BINK_PIXEL_FORMAT.NV12);
    assert.equal(up.uvWidth, uvW1);
    assert.equal(up.uvHeight, uvH1);
    assert.equal(up.yBytesPerRow, 256);
    assert.equal(up.uvBytesPerRow, 256);

    const res = presenter.presentFrame(createMockCommandEncoder(), targetView, 960, 540);
    assert.equal(res.presented, true);
    assert.equal(res.format, BINK_PIXEL_FORMAT.NV12);
    assert.equal(res.framesPresented, i + 1);
  }

  // Created once: Y (r8unorm), UV (rg8unorm), plus 1 dummy texture = 3 textures total, 1 bind group.
  assert.equal(device.createdTextures.length, 3);
  const oldY = device.createdTextures[0];
  const oldUv = device.createdTextures[1];
  const dummyTex = device.createdTextures[2];
  assert.equal(oldY.desc.format, 'r8unorm');
  assert.deepEqual(oldY.desc.size, { width: 48, height: 24, depthOrArrayLayers: 1 });
  assert.equal(oldUv.desc.format, 'rg8unorm');
  assert.deepEqual(oldUv.desc.size, { width: 24, height: 12, depthOrArrayLayers: 1 });

  // Bindings: 2 = Y, 3 = UV, 4 = dummy, 5 = dummy; uniform mode = 3 (NV12).
  assert.equal(device.createdBindGroups.length, 1);
  const bg0Entries = device.createdBindGroups[0].desc.entries;
  assert.equal(bg0Entries.find((e) => e.binding === 2).resource, oldY.view);
  assert.equal(bg0Entries.find((e) => e.binding === 3).resource, oldUv.view);
  assert.equal(bg0Entries.find((e) => e.binding === 4).resource, dummyTex.view);
  assert.equal(bg0Entries.find((e) => e.binding === 5).resource, dummyTex.view);

  const uniformWrite0 = device.writeBufferCalls[device.writeBufferCalls.length - 1];
  const u32Mode = new Uint32Array(
    uniformWrite0.data.buffer,
    uniformWrite0.data.byteOffset,
    uniformWrite0.data.byteLength / 4
  );
  assert.equal(u32Mode[0], 3);

  let stats = presenter.getStats();
  assert.equal(stats.framesPresented, 5);
  assert.equal(stats.planeReallocations, 1);
  assert.equal(stats.scratchAllocations, 1);
  assert.equal(stats.lastFrameWidth, w1);
  assert.equal(stats.lastFrameHeight, h1);

  // Change resolution -> destroys old Y & UV planes, recreates new planes, planeReallocations === 2.
  const w2 = 96;
  const h2 = 54;
  presenter.uploadNv12Frame({
    y: new Uint8Array(w2 * h2).fill(180),
    uv: new Uint8Array((w2 / 2) * (h2 / 2) * 2).fill(128),
    width: w2,
    height: h2,
  });
  presenter.presentFrame(createMockCommandEncoder(), targetView, 960, 540);

  assert.equal(oldY.destroyed, true);
  assert.equal(oldY.destroyCalls, 1);
  assert.equal(oldUv.destroyed, true);
  assert.equal(oldUv.destroyCalls, 1);
  assert.equal(dummyTex.destroyed, false);
  assert.equal(device.createdTextures.length, 5);
  assert.equal(device.createdBindGroups.length, 2);

  const newY = device.createdTextures[3];
  const newUv = device.createdTextures[4];
  assert.equal(newY.desc.format, 'r8unorm');
  assert.deepEqual(newY.desc.size, { width: 96, height: 54, depthOrArrayLayers: 1 });
  assert.equal(newUv.desc.format, 'rg8unorm');
  assert.deepEqual(newUv.desc.size, { width: 48, height: 27, depthOrArrayLayers: 1 });

  stats = presenter.getStats();
  assert.equal(stats.framesPresented, 6);
  assert.equal(stats.planeReallocations, 2);
  assert.equal(stats.resizeCount, 1);
  assert.equal(stats.lastFrameWidth, w2);
  assert.equal(stats.lastFrameHeight, h2);

  // dispose() alias destroys remaining cached planes, dummy texture, and uniform buffer idempotently.
  presenter.dispose();
  assert.equal(presenter.isDestroyed, true);
  assert.equal(newY.destroyed, true);
  assert.equal(newY.destroyCalls, 1);
  assert.equal(newUv.destroyed, true);
  assert.equal(newUv.destroyCalls, 1);
  assert.equal(dummyTex.destroyed, true);
  assert.equal(device.createdBuffers[0].destroyed, true);
  presenter.dispose();
  assert.equal(newY.destroyCalls, 1);
  assert.throws(
    () =>
      presenter.uploadNv12Frame({
        y: new Uint8Array(16),
        uv: new Uint8Array(8),
        width: 4,
        height: 4,
      }),
    /destroyed/i
  );
});

test('W115: BT.709 vs BT.601 and limited vs full color range selection updates conversion uniforms cleanly', () => {
  assert.equal(YUV_COLOR_RANGE.LIMITED, 'limited');
  assert.equal(YUV_COLOR_RANGE.FULL, 'full');
  assert.ok(Object.isFrozen(YUV_COLOR_RANGE));
  assert.ok(Object.isFrozen(YUV_FULL_RANGE_MATRICES));
  assert.ok(Object.isFrozen(YUV_RANGE_MATRICES));

  // Verify getYuvMatrix and packPresenterUniforms for all 4 combinations.
  const bt709Lim = getYuvMatrix(YUV_COLOR_SPACE.BT709, YUV_COLOR_RANGE.LIMITED);
  const bt709Full = getYuvMatrix(YUV_COLOR_SPACE.BT709, YUV_COLOR_RANGE.FULL);
  const bt601Lim = getYuvMatrix(YUV_COLOR_SPACE.BT601, YUV_COLOR_RANGE.LIMITED);
  const bt601Full = getYuvMatrix(YUV_COLOR_SPACE.BT601, YUV_COLOR_RANGE.FULL);

  assertNear(bt709Lim.row0[0], 1.16438356, 1e-6);
  assertNear(bt709Full.row0[0], 1.0, 1e-6);
  assertNear(bt709Full.row0[2], 1.5748, 1e-6);
  assertNear(bt601Full.row0[2], 1.402, 1e-6);
  assertNear(bt709Lim.offset[0], 16 / 255, 1e-6);
  assertNear(bt709Full.offset[0], 0, 1e-6);
  assertNear(bt601Full.offset[1], 128 / 255, 1e-6);

  const packedFull = packPresenterUniforms({
    pixelFormat: BINK_PIXEL_FORMAT.NV12,
    colorSpace: YUV_COLOR_SPACE.BT601,
    colorRange: YUV_COLOR_RANGE.FULL,
  });
  const u32Full = new Uint32Array(packedFull.buffer);
  const f32Full = new Float32Array(packedFull.buffer);
  assert.equal(u32Full[0], 3); // NV12
  assert.equal(u32Full[1], 1); // BT601
  assert.equal(u32Full[3] & UNIFORM_FLAG_FULL_RANGE, UNIFORM_FLAG_FULL_RANGE);
  assertNear(f32Full[8], 1.0, 1e-5);
  assertNear(f32Full[10], 1.402, 1e-5);
  assertNear(f32Full[20], 0, 1e-5);

  // Verify presenter defaults, constructor options, and setColorSpace(space, range).
  const device = createMockWebGpuDevice();
  const presenter = new BinkVideoPresenter({
    device,
    colorSpace: YUV_COLOR_SPACE.BT709,
    colorRange: YUV_COLOR_RANGE.LIMITED,
  });
  assert.equal(presenter.colorSpace, 'bt709');
  assert.equal(presenter.colorRange, 'limited');
  assert.equal(presenter.getStats().colorSpace, 'bt709');
  assert.equal(presenter.getStats().colorRange, 'limited');

  const y = new Uint8Array(16 * 8).fill(200);
  const uv = new Uint8Array(8 * 4 * 2).fill(128);
  const targetView = { kind: 'GPUTextureView', id: 'swap-cs' };

  presenter.uploadNv12Frame({ y, uv, width: 16, height: 8 });
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360);
  const writesBefore = device.writeBufferCalls.length;

  // Switching to BT.601 + full range via setColorSpace(space, range) marks uniforms dirty and updates GPU buffer.
  presenter.setColorSpace(YUV_COLOR_SPACE.BT601, YUV_COLOR_RANGE.FULL);
  assert.equal(presenter.colorSpace, 'bt601');
  assert.equal(presenter.colorRange, 'full');
  assert.equal(presenter.getStats().colorSpace, 'bt601');
  assert.equal(presenter.getStats().colorRange, 'full');

  const presRes = presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360);
  assert.equal(presRes.colorSpace, 'bt601');
  assert.equal(presRes.colorRange, 'full');
  assert.equal(device.writeBufferCalls.length, writesBefore + 1);

  const latestUniform = device.writeBufferCalls[device.writeBufferCalls.length - 1].data;
  const latestU32 = new Uint32Array(
    latestUniform.buffer,
    latestUniform.byteOffset,
    latestUniform.byteLength / 4
  );
  const latestF32 = new Float32Array(
    latestUniform.buffer,
    latestUniform.byteOffset,
    latestUniform.byteLength / 4
  );
  assert.equal(latestU32[1], 1); // BT601
  assert.equal(latestU32[3] & UNIFORM_FLAG_FULL_RANGE, UNIFORM_FLAG_FULL_RANGE);
  assertNear(latestF32[8], bt601Full.row0[0], 1e-5);
  assertNear(latestF32[10], bt601Full.row0[2], 1e-5);
  assertNear(latestF32[20], 0, 1e-5);

  // Switch back to limited range via setColorRange('limited') and BT.709 via setColorSpace('bt709').
  presenter.setColorRange(YUV_COLOR_RANGE.LIMITED);
  presenter.setColorSpace(YUV_COLOR_SPACE.BT709);
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360);
  const backUniform = device.writeBufferCalls[device.writeBufferCalls.length - 1].data;
  const backU32 = new Uint32Array(
    backUniform.buffer,
    backUniform.byteOffset,
    backUniform.byteLength / 4
  );
  const backF32 = new Float32Array(
    backUniform.buffer,
    backUniform.byteOffset,
    backUniform.byteLength / 4
  );
  assert.equal(backU32[1], 0); // BT709
  assert.equal(backU32[3] & UNIFORM_FLAG_FULL_RANGE, 0);
  assertNear(backF32[8], bt709Lim.row0[0], 1e-5);
  assertNear(backF32[20], 16 / 255, 1e-5);

  // Invalid colorSpace / colorRange reject clearly.
  assert.throws(() => presenter.setColorSpace('bt2020'), RangeError);
  assert.throws(() => presenter.setColorSpace('bt709', 'studio'), RangeError);
  assert.throws(() => presenter.setColorRange('narrow'), RangeError);
  assert.throws(() => new BinkVideoPresenter({ device, colorRange: 'invalid' }), RangeError);

  presenter.destroy();
});

test('W115: odd / non-256-aligned widths pad rows cleanly with reusable scratch buffer and zero tail bleed', () => {
  // Direct helper test with pre-dirtied scratch buffer.
  const dirtyScratch = new Uint8Array(256 * 10).fill(0xfe);
  const oddSrc = new Uint8Array(19 * 5).map((_, i) => ((i * 7 + 3) & 0xff) || 1);
  const aligned = prepareAlignedPlaneData(oddSrc, 19, 5, 19, 'odd-plane', dirtyScratch);
  assert.equal(aligned.bytesPerRow, 256);
  assert.equal(aligned.totalBytes, 256 * 5);
  assert.equal(aligned.padded, true);
  assert.equal(aligned.usesScratch, true);
  assert.equal(aligned.data.buffer, dirtyScratch.buffer);

  for (let row = 0; row < 5; row += 1) {
    const dstStart = row * 256;
    assert.deepEqual(
      aligned.data.subarray(dstStart, dstStart + 19),
      oddSrc.subarray(row * 19, row * 19 + 19)
    );
    // Every padding byte must be cleanly zeroed even though dirtyScratch was filled with 0xfe.
    for (let p = 19; p < 256; p += 1) {
      assert.equal(aligned.data[dstStart + p], 0, `row ${row} pad byte ${p} not zeroed`);
    }
  }

  // End-to-end NV12 odd dimensions: 19x7 -> Y=19x7 (stride 23), UV=10x4 (tightRow=20, stride 24).
  const device = createMockWebGpuDevice();
  const presenter = new BinkVideoPresenter({ device, doubleBuffer: false });

  const width = 19;
  const height = 7;
  const uvW = 10;
  const uvH = 4;
  const yStride = 23;
  const uvStride = 24;

  const y = new Uint8Array(yStride * height).fill(0xdd);
  for (let row = 0; row < height; row += 1) {
    for (let col = 0; col < width; col += 1) {
      y[row * yStride + col] = ((row * 31 + col + 1) & 0xff) || 1;
    }
  }

  const uv = new Uint8Array(uvStride * uvH).fill(0xcc);
  for (let row = 0; row < uvH; row += 1) {
    for (let col = 0; col < uvW * 2; col += 1) {
      uv[row * uvStride + col] = ((row * 41 + col + 5) & 0xff) || 2;
    }
  }

  presenter.uploadNv12Frame({ y, uv, width, height, yStride, uvStride });
  presenter.uploadNv12Frame({ y, uv, width, height, yStride, uvStride });

  // Scratch buffer allocated ONCE across both Y & UV planes and both frames!
  assert.equal(presenter.getStats().scratchAllocations, 1);
  assert.equal(device.writeTextureCalls.length, 4);

  const yCall = device.writeTextureCalls[2];
  const uvCall = device.writeTextureCalls[3];
  assert.equal(yCall.dataLayout.bytesPerRow, 256);
  assert.equal(uvCall.dataLayout.bytesPerRow, 256);

  for (let row = 0; row < height; row += 1) {
    assert.deepEqual(
      yCall.data.subarray(row * 256, row * 256 + width),
      y.subarray(row * yStride, row * yStride + width)
    );
    assert.equal(yCall.data[row * 256 + width], 0);
  }
  for (let row = 0; row < uvH; row += 1) {
    assert.deepEqual(
      uvCall.data.subarray(row * 256, row * 256 + uvW * 2),
      uv.subarray(row * uvStride, row * uvStride + uvW * 2)
    );
    assert.equal(uvCall.data[row * 256 + uvW * 2], 0);
  }

  presenter.destroy();
});

test('W115: droppedFrames tracks frame-pacing hitches and overwritten unpresented frames', () => {
  const device = createMockWebGpuDevice();
  const presenter = new BinkVideoPresenter({ device, targetFps: 30 });
  const targetView = { kind: 'GPUTextureView', id: 'swap-drop' };

  assert.equal(presenter.getStats().droppedFrames, 0);

  // Normal pacing does not drop frames.
  assert.equal(presenter.shouldAdvance(0), true);
  assert.equal(presenter.shouldAdvance(34), true);
  assert.equal(presenter.getStats().droppedFrames, 0);

  // 100ms jump at 30fps (33.33ms interval) misses 2 frame intervals -> droppedFrames += 2.
  assert.equal(presenter.shouldAdvance(134), true);
  assert.equal(presenter.getStats().droppedFrames, 2);

  // Uploading two frames without presenting the first supersedes 1 frame -> droppedFrames += 1.
  const rgba = new Uint8Array(8 * 4 * 4).fill(255);
  presenter.uploadRgbaFrame(rgba, 8, 4);
  assert.equal(presenter.getStats().droppedFrames, 2);
  presenter.uploadRgbaFrame(rgba, 8, 4);
  assert.equal(presenter.getStats().droppedFrames, 3);

  // Presenting clears the unpresented flag, so the next upload does not drop.
  presenter.presentFrame(createMockCommandEncoder(), targetView, 320, 180);
  presenter.uploadRgbaFrame(rgba, 8, 4);
  assert.equal(presenter.getStats().droppedFrames, 3);

  // Explicit driver drop recording also increments droppedFrames.
  presenter.recordDroppedFrame(2);
  assert.equal(presenter.getStats().droppedFrames, 5);

  presenter.destroy();
});

test('W165: BINK_CHROMA_ORDER, uniform flags, and WGSL chroma swap support yuv vs yvu without red<->blue inversion', () => {
  assert.ok(Object.isFrozen(BINK_CHROMA_ORDER));
  assert.equal(BINK_CHROMA_ORDER.YUV, 'yuv');
  assert.equal(BINK_CHROMA_ORDER.YVU, 'yvu');
  assert.equal(UNIFORM_FLAG_SWAP_CHROMA, 2);

  // WGSL must inspect bit 1 (2u) of uniforms.flags and swap chroma_u/chroma_v via select().
  assert.match(BINK_PRESENTER_WGSL, /\(uniforms\.flags\s*&\s*2u\)\s*!=\s*0u/);
  assert.match(BINK_PRESENTER_WGSL, /select\s*\(\s*raw_u\s*,\s*raw_v\s*,\s*swap_chroma\s*\)/);
  assert.match(BINK_PRESENTER_WGSL, /select\s*\(\s*raw_v\s*,\s*raw_u\s*,\s*swap_chroma\s*\)/);

  // Default packPresenterUniforms uses 'yuv' (swap flag cleared).
  const packedYuv = packPresenterUniforms({
    pixelFormat: BINK_PIXEL_FORMAT.YUV420P,
    colorSpace: YUV_COLOR_SPACE.BT709,
    colorRange: YUV_COLOR_RANGE.LIMITED,
  });
  const u32Yuv = new Uint32Array(packedYuv.buffer);
  assert.equal(u32Yuv[3] & UNIFORM_FLAG_SWAP_CHROMA, 0);
  assert.equal(u32Yuv[3] & UNIFORM_FLAG_FULL_RANGE, 0);

  // 'yvu' sets UNIFORM_FLAG_SWAP_CHROMA and composes cleanly with UNIFORM_FLAG_FULL_RANGE.
  const packedYvuFull = packPresenterUniforms({
    pixelFormat: BINK_PIXEL_FORMAT.YUV420P,
    colorSpace: YUV_COLOR_SPACE.BT709,
    colorRange: YUV_COLOR_RANGE.FULL,
    chromaOrder: BINK_CHROMA_ORDER.YVU,
  });
  const u32YvuFull = new Uint32Array(packedYvuFull.buffer);
  assert.equal(u32YvuFull[3] & UNIFORM_FLAG_FULL_RANGE, UNIFORM_FLAG_FULL_RANGE);
  assert.equal(u32YvuFull[3] & UNIFORM_FLAG_SWAP_CHROMA, UNIFORM_FLAG_SWAP_CHROMA);

  // Presenter constructor, setter, and per-frame chromaOrder transitions update GPU uniforms cleanly.
  const device = createMockWebGpuDevice();
  const presenter = new BinkVideoPresenter({
    device,
    chromaOrder: BINK_CHROMA_ORDER.YVU,
  });
  assert.equal(presenter.chromaOrder, 'yvu');
  assert.equal(presenter.getStats().chromaOrder, 'yvu');

  const y = new Uint8Array(16 * 8).fill(180);
  const cr = new Uint8Array(8 * 4).fill(220);
  const cb = new Uint8Array(8 * 4).fill(90);
  const targetView = { kind: 'GPUTextureView', id: 'swap-chroma' };

  const up1 = presenter.uploadYuv420Frame({ y, u: cr, v: cb, width: 16, height: 8 });
  assert.equal(up1.chromaOrder, 'yvu');
  const pres1 = presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360);
  assert.equal(pres1.chromaOrder, 'yvu');

  const latestFlags = () => {
    const buf = device.writeBufferCalls[device.writeBufferCalls.length - 1].data;
    return new Uint32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4)[3];
  };
  assert.equal(latestFlags() & UNIFORM_FLAG_SWAP_CHROMA, UNIFORM_FLAG_SWAP_CHROMA);

  const writesAfterYvu = device.writeBufferCalls.length;
  // Per-frame switch back to 'yuv' marks uniforms dirty and clears swap flag.
  const up2 = presenter.uploadYuv420Frame({
    y,
    u: cb,
    v: cr,
    width: 16,
    height: 8,
    chromaOrder: BINK_CHROMA_ORDER.YUV,
  });
  assert.equal(up2.chromaOrder, 'yuv');
  assert.equal(presenter.chromaOrder, 'yuv');
  assert.equal(presenter.getStats().chromaOrder, 'yuv');
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360);
  assert.equal(device.writeBufferCalls.length, writesAfterYvu + 1);
  assert.equal(latestFlags() & UNIFORM_FLAG_SWAP_CHROMA, 0);

  // setChromaOrder('yvu') marks uniforms dirty and re-enables swap flag.
  presenter.setChromaOrder(BINK_CHROMA_ORDER.YVU);
  assert.equal(presenter.chromaOrder, 'yvu');
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360);
  assert.equal(latestFlags() & UNIFORM_FLAG_SWAP_CHROMA, UNIFORM_FLAG_SWAP_CHROMA);

  // uploadNv12Frame also supports per-frame chromaOrder ('yuv' vs 'yvu' / NV21).
  const uv = new Uint8Array(8 * 4 * 2).fill(128);
  const nvUp = presenter.uploadNv12Frame({
    y,
    uv,
    width: 16,
    height: 8,
    chromaOrder: BINK_CHROMA_ORDER.YUV,
  });
  assert.equal(nvUp.chromaOrder, 'yuv');
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360);
  assert.equal(latestFlags() & UNIFORM_FLAG_SWAP_CHROMA, 0);

  presenter.destroy();
});

test('W165: 4K (3840x2160 Y + 1920x1080 Cr/Cb) Bink2Scarlett YUV420P uses zero-copy Y upload and reusable chroma scratch buffer', () => {
  const device = createMockWebGpuDevice();
  const presenter = new BinkVideoPresenter({
    device,
    chromaOrder: BINK_CHROMA_ORDER.YVU,
  });
  const targetView = { kind: 'GPUTextureView', id: 'swap-4k' };

  const width = 3840;
  const height = 2160;
  const uvW = 1920;
  const uvH = 1080;

  const y = new Uint8Array(width * height);
  const cr = new Uint8Array(uvW * uvH);
  const cb = new Uint8Array(uvW * uvH);

  // Stamp deterministic sentinels on first, middle, and last rows of each plane.
  y[0] = 0x11;
  y[width - 1] = 0x22;
  y[(height - 1) * width] = 0x33;
  y[height * width - 1] = 0x44;

  cr[0] = 0xa1;
  cr[uvW - 1] = 0xa2;
  cr[uvW] = 0xa3;
  cr[(uvH - 1) * uvW + uvW - 1] = 0xa4;

  cb[0] = 0xb1;
  cb[uvW - 1] = 0xb2;
  cb[uvW] = 0xb3;
  cb[(uvH - 1) * uvW + uvW - 1] = 0xb4;

  for (let frameIdx = 0; frameIdx < 3; frameIdx += 1) {
    const up = presenter.uploadYuv420Frame({
      y,
      u: cr,
      v: cb,
      width,
      height,
      chromaOrder: BINK_CHROMA_ORDER.YVU,
    });
    assert.equal(up.format, BINK_PIXEL_FORMAT.YUV420P);
    assert.equal(up.chromaOrder, 'yvu');
    assert.equal(up.yBytesPerRow, 3840);
    assert.equal(up.uvBytesPerRow, 2048);
    assert.equal(up.yZeroCopy, true);
    assert.equal(up.chromaZeroCopy, false);
    assert.equal(up.bytesUploaded, 3840 * 2160 + 2048 * 1080 * 2);

    const pres = presenter.presentFrame(createMockCommandEncoder(), targetView, 3840, 2160);
    assert.equal(pres.presented, true);
    assert.equal(pres.chromaOrder, 'yvu');
  }

  // 3 frames * 3 planes = 9 plane writeTexture calls (+ 1 lazy 1x1 neutral dummy texture upload).
  const planeCalls = device.writeTextureCalls.filter(
    (c) => c.dest.texture.label !== 'playgta6-bink-presenter-dummy'
  );
  assert.equal(planeCalls.length, 9);

  // Frame 0 & Frame 2 calls: Y zero-copy, Cr/Cb scratch-repacked into the same reusable 2048x1080 buffer.
  const yCall0 = planeCalls[0];
  const crCall0 = planeCalls[1];
  const cbCall0 = planeCalls[2];
  const yCall2 = planeCalls[6];
  const crCall2 = planeCalls[7];

  // Zero-copy Y: exact Uint8Array reference passed directly to writeTexture with 3840 bytesPerRow.
  assert.equal(yCall0.rawData, y);
  assert.equal(yCall2.rawData, y);
  assert.equal(yCall0.dataLayout.bytesPerRow, 3840);
  assert.equal(yCall0.dataLayout.rowsPerImage, 2160);

  // Reusable chroma scratch buffer: allocated once (2048 * 1080), reused across Cr, Cb, and all frames.
  assert.equal(crCall0.dataLayout.bytesPerRow, 2048);
  assert.equal(cbCall0.dataLayout.bytesPerRow, 2048);
  assert.equal(crCall0.rawData.byteLength, 2048 * 1080);
  assert.equal(crCall0.rawData.buffer, cbCall0.rawData.buffer);
  assert.equal(crCall0.rawData.buffer, crCall2.rawData.buffer);

  // Verify repacked row bytes and zeroed padding tail [1920..2048) on Cr and Cb.
  assert.equal(crCall0.data[0], 0xa1);
  assert.equal(crCall0.data[uvW - 1], 0xa2);
  assert.equal(crCall0.data[uvW], 0);
  assert.equal(crCall0.data[2047], 0);
  assert.equal(crCall0.data[2048], 0xa3);
  assert.equal(crCall0.data[(uvH - 1) * 2048 + uvW - 1], 0xa4);
  assert.equal(crCall0.data[(uvH - 1) * 2048 + 2047], 0);

  assert.equal(cbCall0.data[0], 0xb1);
  assert.equal(cbCall0.data[uvW - 1], 0xb2);
  assert.equal(cbCall0.data[uvW], 0);
  assert.equal(cbCall0.data[2048], 0xb3);
  assert.equal(cbCall0.data[(uvH - 1) * 2048 + uvW - 1], 0xb4);

  // Verify 4K upload telemetry counters.
  const stats = presenter.getStats();
  assert.equal(stats.directZeroCopyUploads, 3);
  assert.equal(stats.chromaRepackUploads, 6);
  assert.equal(stats.scratchAllocations, 1);
  assert.equal(stats.chromaOrder, 'yvu');
  assert.equal(stats.lastFrameWidth, 3840);
  assert.equal(stats.lastFrameHeight, 2160);
  assert.equal(stats.planeReallocations, 1);
  assert.equal(stats.framesPresented, 3);

  // Verify uniform buffer carries YUV420P mode (1) and SWAP_CHROMA flag (2).
  const latestUniform = device.writeBufferCalls[device.writeBufferCalls.length - 1].data;
  const u32 = new Uint32Array(
    latestUniform.buffer,
    latestUniform.byteOffset,
    latestUniform.byteLength / 4
  );
  assert.equal(u32[0], 1);
  assert.equal(u32[3] & UNIFORM_FLAG_SWAP_CHROMA, UNIFORM_FLAG_SWAP_CHROMA);

  presenter.destroy();
});

test('W165: pre-aligned 256-byte chroma strides and 4K NV12 take zero-copy fast path with zero scratch allocations', () => {
  // Direct helper checks for 256-byte-multiple strides (2048 and 2304) vs truncated last row.
  const aligned2304 = new Uint8Array(2304 * 4);
  const direct2304 = prepareAlignedPlaneData(aligned2304, 1920, 4, 2304, 'Cr-2304');
  assert.equal(direct2304.padded, false);
  assert.equal(direct2304.usesScratch, false);
  assert.equal(direct2304.bytesPerRow, 2304);
  assert.equal(direct2304.data, aligned2304);

  // Truncated last row (2048 * 3 + 1920 < 2048 * 4) must safely fall back to scratch repack.
  const truncated2048 = new Uint8Array(2048 * 3 + 1920);
  const scratch = new Uint8Array(2048 * 4);
  const repacked = prepareAlignedPlaneData(truncated2048, 1920, 4, 2048, 'Cr-trunc', scratch);
  assert.equal(repacked.padded, true);
  assert.equal(repacked.usesScratch, true);
  assert.equal(repacked.bytesPerRow, 2048);

  // End-to-end 4K YUV420P with pre-aligned uvStride=2048: all 3 planes zero-copy, 0 scratch allocs.
  const device = createMockWebGpuDevice();
  const presenter = new BinkVideoPresenter({ device, doubleBuffer: false });

  const y4k = new Uint8Array(3840 * 2160);
  const crAligned = new Uint8Array(2048 * 1080);
  const cbAligned = new Uint8Array(2048 * 1080);

  const up = presenter.uploadYuv420Frame({
    y: y4k,
    u: crAligned,
    v: cbAligned,
    width: 3840,
    height: 2160,
    yStride: 3840,
    uvStride: 2048,
    chromaOrder: BINK_CHROMA_ORDER.YVU,
  });
  assert.equal(up.yZeroCopy, true);
  assert.equal(up.chromaZeroCopy, true);
  assert.equal(up.yBytesPerRow, 3840);
  assert.equal(up.uvBytesPerRow, 2048);

  let stats = presenter.getStats();
  assert.equal(stats.directZeroCopyUploads, 3);
  assert.equal(stats.chromaRepackUploads, 0);
  assert.equal(stats.scratchAllocations, 0);

  // 4K NV12 (3840 Y + 1920*2 = 3840 UV) also uploads both planes zero-copy with 0 scratch allocs.
  const uv4k = new Uint8Array(3840 * 1080);
  const nvUp = presenter.uploadNv12Frame({
    y: y4k,
    uv: uv4k,
    width: 3840,
    height: 2160,
  });
  assert.equal(nvUp.yZeroCopy, true);
  assert.equal(nvUp.chromaZeroCopy, true);
  assert.equal(nvUp.yBytesPerRow, 3840);
  assert.equal(nvUp.uvBytesPerRow, 3840);

  stats = presenter.getStats();
  assert.equal(stats.directZeroCopyUploads, 5);
  assert.equal(stats.chromaRepackUploads, 0);
  assert.equal(stats.scratchAllocations, 0);

  presenter.destroy();
});

test('W165: loadOp supports clear (default) and load for menu/UI swapchain underlay and overlay composition', () => {
  assert.ok(Object.isFrozen(BINK_LOAD_OP));
  assert.equal(BINK_LOAD_OP.CLEAR, 'clear');
  assert.equal(BINK_LOAD_OP.LOAD, 'load');

  const device = createMockWebGpuDevice();
  const targetView = { kind: 'GPUTextureView', id: 'swap-menu' };

  // Default presenter uses loadOp: 'clear' with clearValue.
  const defaultPresenter = new BinkVideoPresenter({ device });
  assert.equal(defaultPresenter.loadOp, 'clear');
  assert.equal(defaultPresenter.getStats().loadOp, 'clear');

  const rgba = new Uint8Array(16 * 8 * 4).fill(255);
  defaultPresenter.uploadRgbaFrame(rgba, 16, 8);

  const encClear = createMockCommandEncoder();
  const resClear = defaultPresenter.presentFrame(encClear, targetView, 640, 360);
  assert.equal(resClear.loadOp, 'clear');
  assert.equal(encClear.renderPasses[0].descriptor.colorAttachments[0].loadOp, 'clear');
  assert.ok('clearValue' in encClear.renderPasses[0].descriptor.colorAttachments[0]);

  // Per-frame loadOp: 'load' omits clearValue so existing menu/UI pixels stay intact.
  const encLoad = createMockCommandEncoder();
  const resLoad = defaultPresenter.presentFrame(encLoad, targetView, 640, 360, {
    loadOp: BINK_LOAD_OP.LOAD,
    opacity: 0.5,
  });
  assert.equal(resLoad.presented, true);
  assert.equal(resLoad.loadOp, 'load');
  const loadAttachment = encLoad.renderPasses[0].descriptor.colorAttachments[0];
  assert.equal(loadAttachment.loadOp, 'load');
  assert.equal('clearValue' in loadAttachment, false);

  // Opacity 0.5 was written into uniform buffer lane 2.
  const latestUniform = device.writeBufferCalls[device.writeBufferCalls.length - 1].data;
  const f32 = new Float32Array(
    latestUniform.buffer,
    latestUniform.byteOffset,
    latestUniform.byteLength / 4
  );
  assertNear(f32[2], 0.5);
  defaultPresenter.destroy();

  // Constructor default loadOp: 'load' preserves existing swapchain even when !hasFrame.
  const overlayPresenter = new BinkVideoPresenter({
    device,
    loadOp: BINK_LOAD_OP.LOAD,
  });
  assert.equal(overlayPresenter.loadOp, 'load');
  assert.equal(overlayPresenter.getStats().loadOp, 'load');

  const encEmpty = createMockCommandEncoder();
  const emptyRes = overlayPresenter.presentFrame(encEmpty, targetView, 640, 360);
  assert.equal(emptyRes.presented, false);
  assert.equal(emptyRes.loadOp, 'load');
  assert.equal(encEmpty.renderPasses[0].descriptor.colorAttachments[0].loadOp, 'load');
  assert.equal('clearValue' in encEmpty.renderPasses[0].descriptor.colorAttachments[0], false);
  assert.equal(encEmpty.renderPasses[0].draws.length, 0);

  // setLoadOp('clear') switches the default back to 'clear'.
  overlayPresenter.setLoadOp(BINK_LOAD_OP.CLEAR);
  assert.equal(overlayPresenter.loadOp, 'clear');
  const encAfterSet = createMockCommandEncoder();
  overlayPresenter.presentFrame(encAfterSet, targetView, 640, 360);
  assert.equal(encAfterSet.renderPasses[0].descriptor.colorAttachments[0].loadOp, 'clear');
  assert.ok('clearValue' in encAfterSet.renderPasses[0].descriptor.colorAttachments[0]);

  overlayPresenter.destroy();
});

test('W165: option validation rejects invalid chromaOrder, loadOp, storeOp, and opacity without mutating state', () => {
  const device = createMockWebGpuDevice();
  const targetView = { kind: 'GPUTextureView', id: 'swap-val' };

  assert.throws(() => new BinkVideoPresenter({ device, chromaOrder: 'rgb' }), RangeError);
  assert.throws(() => new BinkVideoPresenter({ device, chromaOrder: 'YVU' }), RangeError);
  assert.throws(() => new BinkVideoPresenter({ device, loadOp: 'discard' }), RangeError);
  assert.throws(() => packPresenterUniforms({ chromaOrder: 'invalid' }), RangeError);
  assert.throws(() => packPresenterUniforms({ opacity: -0.1 }), RangeError);
  assert.throws(() => packPresenterUniforms({ opacity: 1.5 }), RangeError);
  assert.throws(() => packPresenterUniforms({ opacity: NaN }), RangeError);

  const presenter = new BinkVideoPresenter({ device });
  assert.throws(() => presenter.setChromaOrder('ycrcb'), RangeError);
  assert.throws(() => presenter.setLoadOp('preserve'), RangeError);
  assert.equal(presenter.chromaOrder, 'yuv');
  assert.equal(presenter.loadOp, 'clear');

  const y = new Uint8Array(16 * 8);
  const u = new Uint8Array(8 * 4);
  const v = new Uint8Array(8 * 4);
  const uv = new Uint8Array(8 * 4 * 2);

  assert.throws(
    () => presenter.uploadYuv420Frame({ y, u, v, width: 16, height: 8, chromaOrder: 'bad' }),
    RangeError
  );
  assert.throws(
    () => presenter.uploadNv12Frame({ y, uv, width: 16, height: 8, chromaOrder: 'nv21' }),
    RangeError
  );
  assert.equal(presenter.getStats().uploadedFrames, 0);
  assert.equal(presenter.chromaOrder, 'yuv');

  assert.throws(
    () => presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360, 'bad'),
    TypeError
  );
  assert.throws(
    () =>
      presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360, {
        loadOp: 'invalid',
      }),
    RangeError
  );
  assert.throws(
    () =>
      presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360, {
        storeOp: 'keep',
      }),
    RangeError
  );
  assert.throws(
    () =>
      presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360, {
        opacity: 2,
      }),
    RangeError
  );

  presenter.destroy();
});

test('W192: steady-state 4K (3840x2160) YUV420P frames perform 0 createTexture, 0 createView, 0 createBindGroup, and 0 uniform writeBuffer calls', () => {
  const width = 3840;
  const height = 2160;
  const uvW = 1920;
  const uvH = 1080;

  const y = new Uint8Array(width * height);
  const cr = new Uint8Array(uvW * uvH);
  const cb = new Uint8Array(uvW * uvH);
  const targetView = { kind: 'GPUTextureView', id: 'swap-4k-w192' };

  // 1) Single-buffered steady-state over 8 consecutive 4K frames.
  const singleDevice = createMockWebGpuDevice();
  const singlePresenter = new BinkVideoPresenter({
    device: singleDevice,
    doubleBuffer: false,
    chromaOrder: BINK_CHROMA_ORDER.YVU,
  });

  // Frame 0 warms Y/U/V + dummy textures, views, bind group, and initial uniform write.
  singlePresenter.uploadYuv420Frame({ y, u: cr, v: cb, width, height, chromaOrder: 'yvu' });
  singlePresenter.presentFrame(createMockCommandEncoder(), targetView, 3840, 2160);

  const warmTexSingle = singleDevice.createdTextures.length;
  const warmViewSingle = singleDevice.createdViews.length;
  const warmBgSingle = singleDevice.createdBindGroups.length;
  const warmUniformSingle = singleDevice.writeBufferCalls.length;
  assert.equal(warmTexSingle, 4); // Y, U, V + 1x1 dummy
  assert.equal(warmViewSingle, 4);
  assert.equal(warmBgSingle, 1);
  assert.equal(warmUniformSingle, 1);
  assert.equal(singlePresenter.bindGroupCacheHits, 0);
  assert.equal(singlePresenter.bindGroupCacheMisses, 1);
  assert.equal(singlePresenter.uniformWritesSkipped, 0);
  assert.equal(singlePresenter.framesPresented, 1);

  // Frames 1..7: strictly 0 texture, 0 view, 0 bindGroup, and 0 uniform writeBuffer calls.
  for (let i = 1; i < 8; i += 1) {
    singlePresenter.uploadYuv420Frame({ y, u: cr, v: cb, width, height, chromaOrder: 'yvu' });
    const res = singlePresenter.presentFrame(createMockCommandEncoder(), targetView, 3840, 2160);
    assert.equal(res.presented, true);
    assert.equal(res.framesPresented, i + 1);
  }

  assert.equal(singleDevice.createdTextures.length, warmTexSingle);
  assert.equal(singleDevice.createdViews.length, warmViewSingle);
  assert.equal(singleDevice.createdBindGroups.length, warmBgSingle);
  assert.equal(singleDevice.writeBufferCalls.length, warmUniformSingle);
  for (const tex of singleDevice.createdTextures) {
    assert.equal(tex.createViewCalls, 1);
  }

  const singleStats = singlePresenter.getStats();
  assert.equal(singleStats.framesPresented, 8);
  assert.equal(singleStats.bindGroupCacheHits, 7);
  assert.equal(singleStats.bindGroupCacheMisses, 1);
  assert.equal(singleStats.uniformWritesSkipped, 7);
  assert.equal(singlePresenter.framesPresented, 8);
  assert.equal(singlePresenter.bindGroupCacheHits, 7);
  assert.equal(singlePresenter.bindGroupCacheMisses, 1);
  assert.equal(singlePresenter.uniformWritesSkipped, 7);
  singlePresenter.destroy();

  // 2) Double-buffered steady-state over 10 consecutive 4K frames.
  const doubleDevice = createMockWebGpuDevice();
  const doublePresenter = new BinkVideoPresenter({
    device: doubleDevice,
    doubleBuffer: true,
    chromaOrder: BINK_CHROMA_ORDER.YVU,
  });

  // Frames 0 & 1 warm slot 0 and slot 1.
  for (let i = 0; i < 2; i += 1) {
    doublePresenter.uploadYuv420Frame({ y, u: cr, v: cb, width, height, chromaOrder: 'yvu' });
    doublePresenter.presentFrame(createMockCommandEncoder(), targetView, 3840, 2160);
  }

  const warmTexDouble = doubleDevice.createdTextures.length;
  const warmViewDouble = doubleDevice.createdViews.length;
  const warmBgDouble = doubleDevice.createdBindGroups.length;
  const warmUniformDouble = doubleDevice.writeBufferCalls.length;
  assert.equal(warmTexDouble, 7); // 2 slots * 3 planes + 1x1 dummy
  assert.equal(warmViewDouble, 7);
  assert.equal(warmBgDouble, 2);
  assert.equal(warmUniformDouble, 1);
  assert.equal(doublePresenter.bindGroupCacheHits, 0);
  assert.equal(doublePresenter.bindGroupCacheMisses, 2);
  assert.equal(doublePresenter.uniformWritesSkipped, 1);

  // Frames 2..9: strictly 0 texture, 0 view, 0 bindGroup, and 0 uniform writeBuffer calls.
  for (let i = 2; i < 10; i += 1) {
    doublePresenter.uploadYuv420Frame({ y, u: cr, v: cb, width, height, chromaOrder: 'yvu' });
    doublePresenter.presentFrame(createMockCommandEncoder(), targetView, 3840, 2160);
  }

  assert.equal(doubleDevice.createdTextures.length, warmTexDouble);
  assert.equal(doubleDevice.createdViews.length, warmViewDouble);
  assert.equal(doubleDevice.createdBindGroups.length, warmBgDouble);
  assert.equal(doubleDevice.writeBufferCalls.length, warmUniformDouble);

  const doubleStats = doublePresenter.getStats();
  assert.equal(doubleStats.framesPresented, 10);
  assert.equal(doubleStats.bindGroupCacheHits, 8);
  assert.equal(doubleStats.bindGroupCacheMisses, 2);
  assert.equal(doubleStats.uniformWritesSkipped, 9);
  assert.equal(doublePresenter.bindGroupCacheHits, 8);
  assert.equal(doublePresenter.bindGroupCacheMisses, 2);
  assert.equal(doublePresenter.uniformWritesSkipped, 9);
  doublePresenter.destroy();
});

test('W192: bindGroup cache invalidates cleanly on chromaOrder and dimension changes while reusing cached texture views', () => {
  const device = createMockWebGpuDevice();
  const presenter = new BinkVideoPresenter({
    device,
    doubleBuffer: false,
    chromaOrder: BINK_CHROMA_ORDER.YVU,
  });
  const targetView = { kind: 'GPUTextureView', id: 'swap-invalidate' };

  const y4k = new Uint8Array(3840 * 2160);
  const c4k = new Uint8Array(1920 * 1080);

  // Warm 4K YVU and verify 2 steady-state hits.
  for (let i = 0; i < 3; i += 1) {
    presenter.uploadYuv420Frame({ y: y4k, u: c4k, v: c4k, width: 3840, height: 2160 });
    presenter.presentFrame(createMockCommandEncoder(), targetView, 3840, 2160);
  }
  assert.equal(device.createdTextures.length, 4);
  assert.equal(device.createdViews.length, 4);
  assert.equal(device.createdBindGroups.length, 1);
  assert.equal(device.writeBufferCalls.length, 1);
  assert.equal(presenter.bindGroupCacheHits, 2);
  assert.equal(presenter.bindGroupCacheMisses, 1);
  assert.equal(presenter.uniformWritesSkipped, 2);

  // Switch chromaOrder from 'yvu' -> 'yuv' via per-frame option:
  // 0 createTexture, 0 createView (reuses cached Y/U/V views), 1 new bindGroup, 1 uniform write.
  presenter.uploadYuv420Frame({
    y: y4k,
    u: c4k,
    v: c4k,
    width: 3840,
    height: 2160,
    chromaOrder: BINK_CHROMA_ORDER.YUV,
  });
  presenter.presentFrame(createMockCommandEncoder(), targetView, 3840, 2160);
  assert.equal(device.createdTextures.length, 4);
  assert.equal(device.createdViews.length, 4);
  assert.equal(device.createdBindGroups.length, 2);
  assert.equal(device.writeBufferCalls.length, 2);
  assert.equal(presenter.bindGroupCacheMisses, 2);
  assert.equal(presenter.bindGroupCacheHits, 2);

  // Verify the recreated bindGroup still binds the cached 4K Y/U/V views and cleared swap flag.
  const bgAfterChroma = device.createdBindGroups[1].desc.entries;
  assert.equal(bgAfterChroma.find((e) => e.binding === 2).resource, device.createdTextures[0].view);
  assert.equal(bgAfterChroma.find((e) => e.binding === 3).resource, device.createdTextures[1].view);
  assert.equal(bgAfterChroma.find((e) => e.binding === 4).resource, device.createdTextures[2].view);
  const u32Yuv = new Uint32Array(
    device.writeBufferCalls[1].data.buffer,
    device.writeBufferCalls[1].data.byteOffset,
    device.writeBufferCalls[1].data.byteLength / 4
  );
  assert.equal(u32Yuv[3] & UNIFORM_FLAG_SWAP_CHROMA, 0);

  // Subsequent 'yuv' frame hits cache again (0 allocs, 0 uniform write).
  presenter.uploadYuv420Frame({ y: y4k, u: c4k, v: c4k, width: 3840, height: 2160 });
  presenter.presentFrame(createMockCommandEncoder(), targetView, 3840, 2160);
  assert.equal(device.createdTextures.length, 4);
  assert.equal(device.createdViews.length, 4);
  assert.equal(device.createdBindGroups.length, 2);
  assert.equal(device.writeBufferCalls.length, 2);
  assert.equal(presenter.bindGroupCacheHits, 3);
  assert.equal(presenter.uniformWritesSkipped, 3);

  // Switch chromaOrder via setChromaOrder('yvu') -> invalidates bindGroup cache without touching textures/views.
  presenter.setChromaOrder(BINK_CHROMA_ORDER.YVU);
  presenter.presentFrame(createMockCommandEncoder(), targetView, 3840, 2160);
  assert.equal(device.createdTextures.length, 4);
  assert.equal(device.createdViews.length, 4);
  assert.equal(device.createdBindGroups.length, 3);
  assert.equal(device.writeBufferCalls.length, 3);
  assert.equal(presenter.bindGroupCacheMisses, 3);
  assert.equal(presenter.bindGroupCacheHits, 3);

  // Change dimensions 3840x2160 -> 1920x1080 with unchanged format & chromaOrder:
  // Old 4K textures destroyed, 3 new textures + 3 new views + 1 new bindGroup, uniform write STILL skipped!
  const oldY = device.createdTextures[0];
  const oldU = device.createdTextures[1];
  const oldV = device.createdTextures[2];
  const y1080 = new Uint8Array(1920 * 1080);
  const c1080 = new Uint8Array(960 * 540);

  presenter.uploadYuv420Frame({ y: y1080, u: c1080, v: c1080, width: 1920, height: 1080 });
  presenter.presentFrame(createMockCommandEncoder(), targetView, 3840, 2160);

  assert.equal(oldY.destroyed, true);
  assert.equal(oldU.destroyed, true);
  assert.equal(oldV.destroyed, true);
  assert.equal(device.createdTextures.length, 7);
  assert.equal(device.createdViews.length, 7);
  assert.equal(device.createdBindGroups.length, 4);
  assert.equal(device.writeBufferCalls.length, 3); // uniform params unchanged -> write skipped!
  assert.equal(presenter.bindGroupCacheMisses, 4);
  assert.equal(presenter.uniformWritesSkipped, 4);

  const bg1080 = device.createdBindGroups[3].desc.entries;
  assert.equal(bg1080.find((e) => e.binding === 2).resource, device.createdTextures[4].view);
  assert.equal(bg1080.find((e) => e.binding === 3).resource, device.createdTextures[5].view);
  assert.equal(bg1080.find((e) => e.binding === 4).resource, device.createdTextures[6].view);

  // Steady-state at 1920x1080 hits cache with 0 allocations and 0 uniform writes.
  presenter.uploadYuv420Frame({ y: y1080, u: c1080, v: c1080, width: 1920, height: 1080 });
  presenter.presentFrame(createMockCommandEncoder(), targetView, 3840, 2160);
  assert.equal(device.createdTextures.length, 7);
  assert.equal(device.createdViews.length, 7);
  assert.equal(device.createdBindGroups.length, 4);
  assert.equal(device.writeBufferCalls.length, 3);
  assert.equal(presenter.bindGroupCacheHits, 4);
  assert.equal(presenter.uniformWritesSkipped, 5);
  assert.equal(presenter.framesPresented, 8);

  const stats = presenter.getStats();
  assert.equal(stats.bindGroupCacheHits, 4);
  assert.equal(stats.bindGroupCacheMisses, 4);
  assert.equal(stats.uniformWritesSkipped, 5);
  assert.equal(stats.framesPresented, 8);

  presenter.destroy();
});

test('W192: uniform scratch Float32Array is pre-allocated and reused across updates while redundant writes are skipped', () => {
  const device = createMockWebGpuDevice();
  const presenter = new BinkVideoPresenter({ device, doubleBuffer: false });
  const targetView = { kind: 'GPUTextureView', id: 'swap-uniform-scratch' };

  const y = new Uint8Array(16 * 8).fill(180);
  const u = new Uint8Array(8 * 4).fill(128);
  const v = new Uint8Array(8 * 4).fill(128);

  // Frame 0: writes initial uniforms using pre-allocated Float32Array(24).
  presenter.uploadYuv420Frame({ y, u, v, width: 16, height: 8 });
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360);
  assert.equal(device.writeBufferCalls.length, 1);
  const scratchRef = device.writeBufferCalls[0].rawData;
  assert.ok(scratchRef instanceof Float32Array);
  assert.equal(scratchRef.length, UNIFORM_BUFFER_SIZE / 4);
  assert.equal(scratchRef.byteLength, UNIFORM_BUFFER_SIZE);

  // No-op setUvTransform(1, 1, 0, 0) + unchanged present -> skipped.
  presenter.setUvTransform(1, 1, 0, 0);
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360);
  assert.equal(device.writeBufferCalls.length, 1);
  assert.equal(presenter.uniformWritesSkipped, 1);

  // Toggle chromaOrder away and back before presenting -> bitwise match skips writeBuffer!
  presenter.setChromaOrder(BINK_CHROMA_ORDER.YVU);
  presenter.setChromaOrder(BINK_CHROMA_ORDER.YUV);
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360);
  assert.equal(device.writeBufferCalls.length, 1);
  assert.equal(presenter.uniformWritesSkipped, 2);

  // Real parameter updates (opacity, colorSpace, colorRange, uvTransform) reuse the exact same Float32Array scratch instance.
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360, { opacity: 0.75 });
  presenter.setColorSpace(YUV_COLOR_SPACE.BT601, YUV_COLOR_RANGE.FULL);
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360, { opacity: 0.75 });
  presenter.setUvTransform(0.5, 0.5, 0.25, 0.25);
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360, { opacity: 0.75 });

  assert.equal(device.writeBufferCalls.length, 4);
  for (let i = 1; i < device.writeBufferCalls.length; i += 1) {
    assert.equal(
      device.writeBufferCalls[i].rawData,
      scratchRef,
      `writeBuffer call ${i} should reuse the exact pre-allocated Float32Array scratch buffer`
    );
  }

  // Subsequent frame with same opacity (0.75) skips writeBuffer again.
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360, { opacity: 0.75 });
  assert.equal(device.writeBufferCalls.length, 4);
  assert.equal(presenter.uniformWritesSkipped, 3);
  assert.equal(presenter.getStats().uniformWritesSkipped, 3);
  assert.equal(presenter.getStats().framesPresented, 7);

  presenter.destroy();
});

test('W241: 4K Bink2 pacing telemetry tracks avgUploadMs, maxUploadMs, lastFrameIntervalMs, droppedFrames, and targetFps accurately', () => {
  const device = createMockWebGpuDevice();
  let clockMs = 1000;
  let perPlaneUploadMs = 0;

  const origWriteTexture = device.queue.writeTexture.bind(device.queue);
  device.queue.writeTexture = (dest, data, dataLayout, size) => {
    origWriteTexture(dest, data, dataLayout, size);
    if (dest.texture && dest.texture.label !== 'playgta6-bink-presenter-dummy') {
      clockMs += perPlaneUploadMs;
    }
  };

  assert.throws(() => new BinkVideoPresenter({ device, now: 'not-a-fn' }), TypeError);

  const presenter = new BinkVideoPresenter({
    device,
    targetFps: 60,
    doubleBuffer: false,
    chromaOrder: BINK_CHROMA_ORDER.YVU,
    now: () => clockMs,
  });

  // Initial pacing stats are sanitized zeros while reflecting configured targetFps.
  const initialPacing = presenter.getPacingStats();
  assert.equal(initialPacing.targetFps, 60);
  assertNear(initialPacing.targetFrameIntervalMs, 1000 / 60, 1e-6);
  assert.equal(initialPacing.presentedFrames, 0);
  assert.equal(initialPacing.droppedFrames, 0);
  assert.equal(initialPacing.dropRate, 0);
  assert.equal(initialPacing.uploadSamples, 0);
  assert.equal(initialPacing.avgUploadMs, 0);
  assert.equal(initialPacing.maxUploadMs, 0);
  assert.equal(initialPacing.lastUploadMs, 0);
  assert.equal(initialPacing.lastFrameIntervalMs, 0);

  const y = new Uint8Array(3840 * 2160);
  const cr = new Uint8Array(1920 * 1080);
  const cb = new Uint8Array(1920 * 1080);
  const targetView = { kind: 'GPUTextureView', id: 'swap-w241-pacing' };

  // Frame 0: 3 planes * 1.5ms = 4.5ms upload, presented at t=1004.5 -> first frame interval stays 0.
  perPlaneUploadMs = 1.5;
  const up0 = presenter.uploadYuv420Frame({ y, u: cr, v: cb, width: 3840, height: 2160 });
  assertNear(up0.uploadMs, 4.5, 1e-6);
  presenter.presentFrame(createMockCommandEncoder(), targetView, 3840, 2160);

  let pacing = presenter.getPacingStats();
  assert.equal(pacing.presentedFrames, 1);
  assert.equal(pacing.droppedFrames, 0);
  assert.equal(pacing.uploadSamples, 1);
  assertNear(pacing.avgUploadMs, 4.5, 1e-6);
  assertNear(pacing.maxUploadMs, 4.5, 1e-6);
  assertNear(pacing.lastUploadMs, 4.5, 1e-6);
  assert.equal(pacing.lastFrameIntervalMs, 0);

  // Frame 1: advance clock to t=1020.0, upload takes 3 * 0.5ms = 1.5ms -> present at t=1021.5 -> interval = 17.0ms.
  clockMs = 1020.0;
  perPlaneUploadMs = 0.5;
  const up1 = presenter.uploadYuv420Frame({ y, u: cr, v: cb, width: 3840, height: 2160 });
  assertNear(up1.uploadMs, 1.5, 1e-6);
  presenter.presentFrame(createMockCommandEncoder(), targetView, 3840, 2160);

  pacing = presenter.getPacingStats();
  assert.equal(pacing.presentedFrames, 2);
  assert.equal(pacing.uploadSamples, 2);
  assertNear(pacing.avgUploadMs, 3.0, 1e-6); // (4.5 + 1.5) / 2
  assertNear(pacing.maxUploadMs, 4.5, 1e-6);
  assertNear(pacing.lastUploadMs, 1.5, 1e-6);
  assertNear(pacing.lastFrameIntervalMs, 17.0, 1e-6);

  // Frame 2: advance clock to t=1050.0, upload takes 3 * 2.0ms = 6.0ms, present with explicit nowMs=1055.0 -> interval = 33.5ms.
  clockMs = 1050.0;
  perPlaneUploadMs = 2.0;
  presenter.uploadYuv420Frame({ y, u: cr, v: cb, width: 3840, height: 2160 });
  presenter.presentFrame(createMockCommandEncoder(), targetView, 3840, 2160, { nowMs: 1055.0 });
  presenter.recordDroppedFrame(1);

  pacing = presenter.getPacingStats();
  const fullStats = presenter.getStats();
  assert.equal(pacing.presentedFrames, 3);
  assert.equal(pacing.droppedFrames, 1);
  assertNear(pacing.dropRate, 0.25, 1e-6);
  assert.equal(pacing.uploadSamples, 3);
  assertNear(pacing.avgUploadMs, 4.0, 1e-6); // (4.5 + 1.5 + 6.0) / 3
  assertNear(pacing.maxUploadMs, 6.0, 1e-6);
  assertNear(pacing.lastUploadMs, 6.0, 1e-6);
  assertNear(pacing.lastFrameIntervalMs, 33.5, 1e-6);

  // Verify getStats() and convenience getters match getPacingStats().
  assert.equal(fullStats.targetFps, pacing.targetFps);
  assertNear(fullStats.targetFrameIntervalMs, pacing.targetFrameIntervalMs, 1e-6);
  assert.equal(fullStats.presentedFrames, pacing.presentedFrames);
  assert.equal(fullStats.droppedFrames, pacing.droppedFrames);
  assertNear(fullStats.dropRate, pacing.dropRate, 1e-6);
  assertNear(fullStats.avgUploadMs, pacing.avgUploadMs, 1e-6);
  assertNear(fullStats.maxUploadMs, pacing.maxUploadMs, 1e-6);
  assertNear(fullStats.lastUploadMs, pacing.lastUploadMs, 1e-6);
  assertNear(fullStats.lastFrameIntervalMs, pacing.lastFrameIntervalMs, 1e-6);
  assert.equal(presenter.presentedFrames, 3);
  assert.equal(presenter.droppedFrames, 1);
  assertNear(presenter.avgUploadMs, 4.0, 1e-6);
  assertNear(presenter.maxUploadMs, 6.0, 1e-6);
  assertNear(presenter.lastFrameIntervalMs, 33.5, 1e-6);

  // Mutating returned snapshot objects does not mutate presenter internals.
  pacing.presentedFrames = 999;
  pacing.avgUploadMs = -42;
  assert.equal(presenter.getPacingStats().presentedFrames, 3);
  assertNear(presenter.getPacingStats().avgUploadMs, 4.0, 1e-6);

  // Backwards clock / non-finite explicit nowMs handling.
  assert.throws(
    () => presenter.presentFrame(createMockCommandEncoder(), targetView, 3840, 2160, { nowMs: NaN }),
    RangeError
  );
  assert.throws(
    () =>
      presenter.presentFrame(createMockCommandEncoder(), targetView, 3840, 2160, {
        nowMs: Infinity,
      }),
    RangeError
  );
  // Clock moving backwards sanitizes lastFrameIntervalMs to 0 (never negative).
  presenter.presentFrame(createMockCommandEncoder(), targetView, 3840, 2160, { nowMs: 1000.0 });
  assert.equal(presenter.getPacingStats().lastFrameIntervalMs, 0);

  presenter.destroy();
});

test('W241: writeTexture destination/origin/dataLayout/size and presentFrame descriptors reuse scratch objects with zero steady-state allocations', () => {
  const device = createMockWebGpuDevice();
  const presenter = new BinkVideoPresenter({
    device,
    doubleBuffer: false,
    chromaOrder: BINK_CHROMA_ORDER.YVU,
  });
  const targetView = { kind: 'GPUTextureView', id: 'swap-w241-zero-alloc' };

  const y4k = new Uint8Array(3840 * 2160);
  const cr4k = new Uint8Array(1920 * 1080);
  const cb4k = new Uint8Array(1920 * 1080);

  // Run 4 consecutive 4K YUV420P frames (12 plane writeTexture calls + 1 dummy texture writeTexture).
  const clearEncoders = [];
  const clearResults = [];
  for (let i = 0; i < 4; i += 1) {
    presenter.uploadYuv420Frame({ y: y4k, u: cr4k, v: cb4k, width: 3840, height: 2160 });
    const enc = createMockCommandEncoder();
    const res = presenter.presentFrame(enc, targetView, 3840, 2160, {
      clearColor: { r: 0.01 * i, g: 0.02, b: 0.03, a: 1 },
    });
    clearEncoders.push(enc);
    clearResults.push(res);
  }

  // Also upload 4K NV12 and RGBA8 frames to confirm scratch descriptor reuse across all formats.
  presenter.uploadNv12Frame({
    y: y4k,
    uv: new Uint8Array(3840 * 1080),
    width: 3840,
    height: 2160,
  });
  presenter.uploadRgbaFrame(new Uint8Array(64 * 32 * 4), 64, 32);

  // Every single writeTexture call must reuse the exact same destination, origin, dataLayout, and size objects.
  assert.ok(device.writeTextureCalls.length >= 16);
  const firstCall = device.writeTextureCalls[0];
  for (let i = 1; i < device.writeTextureCalls.length; i += 1) {
    const call = device.writeTextureCalls[i];
    assert.equal(call.rawDest, firstCall.rawDest, `writeTexture[${i}] must reuse destination scratch`);
    assert.equal(call.rawOrigin, firstCall.rawOrigin, `writeTexture[${i}] must reuse origin scratch`);
    assert.equal(
      call.rawDataLayout,
      firstCall.rawDataLayout,
      `writeTexture[${i}] must reuse dataLayout scratch`
    );
    assert.equal(call.rawSize, firstCall.rawSize, `writeTexture[${i}] must reuse size scratch`);
  }

  // Verify the recorded snapshot values were still distinct and accurate per plane.
  const yCall0 = device.writeTextureCalls[0];
  const crCall0 = device.writeTextureCalls[1];
  assert.equal(yCall0.size.width, 3840);
  assert.equal(yCall0.size.height, 2160);
  assert.equal(yCall0.dataLayout.bytesPerRow, 3840);
  assert.equal(crCall0.size.width, 1920);
  assert.equal(crCall0.size.height, 1080);
  assert.equal(crCall0.dataLayout.bytesPerRow, 2048);

  // Every clear-mode presentFrame() call reuses the exact same renderPassDescriptor, colorAttachments,
  // colorAttachment, clearValue, result object, and viewport object.
  const pass0Desc = clearEncoders[0].renderPasses[0].descriptor;
  for (let i = 1; i < clearEncoders.length; i += 1) {
    const passDesc = clearEncoders[i].renderPasses[0].descriptor;
    assert.equal(passDesc, pass0Desc, `presentFrame[${i}] must reuse clear renderPassDescriptor`);
    assert.equal(
      passDesc.colorAttachments,
      pass0Desc.colorAttachments,
      `presentFrame[${i}] must reuse colorAttachments array`
    );
    assert.equal(
      passDesc.colorAttachments[0],
      pass0Desc.colorAttachments[0],
      `presentFrame[${i}] must reuse colorAttachment scratch`
    );
    assert.equal(
      passDesc.colorAttachments[0].clearValue,
      pass0Desc.colorAttachments[0].clearValue,
      `presentFrame[${i}] must reuse clearValue scratch`
    );
    assert.equal(clearResults[i], clearResults[0], `presentFrame[${i}] must reuse result scratch`);
    assert.equal(
      clearResults[i].viewport,
      clearResults[0].viewport,
      `presentFrame[${i}] must reuse viewport scratch`
    );
  }

  // Load-mode presentFrame() reuses its own dedicated descriptor without clearValue key.
  const loadEnc0 = createMockCommandEncoder();
  const loadEnc1 = createMockCommandEncoder();
  presenter.presentFrame(loadEnc0, targetView, 1920, 1080, { loadOp: BINK_LOAD_OP.LOAD });
  presenter.presentFrame(loadEnc1, targetView, 1920, 1080, { loadOp: BINK_LOAD_OP.LOAD });
  const loadDesc0 = loadEnc0.renderPasses[0].descriptor;
  const loadDesc1 = loadEnc1.renderPasses[0].descriptor;
  assert.equal(loadDesc1, loadDesc0);
  assert.equal(loadDesc1.colorAttachments, loadDesc0.colorAttachments);
  assert.equal(loadDesc1.colorAttachments[0], loadDesc0.colorAttachments[0]);
  assert.equal('clearValue' in loadDesc0.colorAttachments[0], false);
  assert.notEqual(loadDesc0, pass0Desc);

  // Staging pool immediate-encoder uploads reuse layout scratch; deferred uploads keep distinct layouts.
  const stagedLayouts = [];
  const fakeStagingPool = {
    stageTextureUpload(_encoder, _texture, _data, layout) {
      stagedLayouts.push({ raw: layout, snapshot: { ...layout } });
    },
  };
  const stagedPresenter = new BinkVideoPresenter({
    device,
    stagingPool: fakeStagingPool,
    doubleBuffer: false,
  });
  const syncEnc = createMockCommandEncoder();
  stagedPresenter.uploadYuv420Frame({
    y: new Uint8Array(16 * 8),
    u: new Uint8Array(8 * 4),
    v: new Uint8Array(8 * 4),
    width: 16,
    height: 8,
    commandEncoder: syncEnc,
  });
  assert.equal(stagedLayouts.length, 3);
  assert.equal(stagedLayouts[1].raw, stagedLayouts[0].raw);
  assert.equal(stagedLayouts[2].raw, stagedLayouts[0].raw);
  assert.equal(stagedLayouts[0].snapshot.width, 16);
  assert.equal(stagedLayouts[1].snapshot.width, 8);

  // Deferred staging (no commandEncoder) keeps independent layouts until flushPendingUploads().
  stagedLayouts.length = 0;
  stagedPresenter.uploadYuv420Frame({
    y: new Uint8Array(16 * 8),
    u: new Uint8Array(8 * 4),
    v: new Uint8Array(8 * 4),
    width: 16,
    height: 8,
  });
  stagedPresenter.flushPendingUploads(createMockCommandEncoder());
  assert.equal(stagedLayouts.length, 3);
  assert.notEqual(stagedLayouts[1].raw, stagedLayouts[0].raw);
  assert.equal(stagedLayouts[0].snapshot.width, 16);
  assert.equal(stagedLayouts[1].snapshot.width, 8);
  assert.equal(stagedLayouts[2].snapshot.width, 8);

  stagedPresenter.destroy();
  presenter.destroy();
});

test('W241: resetStats() clears pacing and upload telemetry cleanly without invalidating cached 4K GPU resources', () => {
  const device = createMockWebGpuDevice();
  let clockMs = 500;
  let uploadStepMs = 2.0;

  const origWriteTexture = device.queue.writeTexture.bind(device.queue);
  device.queue.writeTexture = (dest, data, dataLayout, size) => {
    origWriteTexture(dest, data, dataLayout, size);
    if (dest.texture && dest.texture.label !== 'playgta6-bink-presenter-dummy') {
      clockMs += uploadStepMs;
    }
  };

  const presenter = new BinkVideoPresenter({
    device,
    targetFps: 30,
    doubleBuffer: false,
    chromaOrder: BINK_CHROMA_ORDER.YVU,
    now: () => clockMs,
  });
  const targetView = { kind: 'GPUTextureView', id: 'swap-w241-reset' };
  const y = new Uint8Array(3840 * 2160);
  const cr = new Uint8Array(1920 * 1080);
  const cb = new Uint8Array(1920 * 1080);

  // Present 2 frames and record 2 dropped frames.
  presenter.uploadYuv420Frame({ y, u: cr, v: cb, width: 3840, height: 2160 });
  presenter.presentFrame(createMockCommandEncoder(), targetView, 3840, 2160);
  clockMs += 30;
  presenter.uploadYuv420Frame({ y, u: cr, v: cb, width: 3840, height: 2160 });
  presenter.presentFrame(createMockCommandEncoder(), targetView, 3840, 2160);
  presenter.recordDroppedFrame(2);

  const texCountBefore = device.createdTextures.length;
  const viewCountBefore = device.createdViews.length;
  const bgCountBefore = device.createdBindGroups.length;
  const uniformWritesBefore = device.writeBufferCalls.length;

  assert.equal(presenter.getPacingStats().presentedFrames, 2);
  assert.equal(presenter.getPacingStats().droppedFrames, 2);
  assert.ok(presenter.getPacingStats().avgUploadMs > 0);
  assert.ok(presenter.getPacingStats().maxUploadMs > 0);
  assert.ok(presenter.getPacingStats().lastFrameIntervalMs > 0);

  // Reset telemetry counters & timing accumulators.
  presenter.resetStats();

  const pacingAfterReset = presenter.getPacingStats();
  assert.deepEqual(pacingAfterReset, {
    targetFps: 30,
    targetFrameIntervalMs: 1000 / 30,
    presentedFrames: 0,
    droppedFrames: 0,
    dropRate: 0,
    uploadSamples: 0,
    avgUploadMs: 0,
    maxUploadMs: 0,
    lastUploadMs: 0,
    lastFrameIntervalMs: 0,
  });

  const statsAfterReset = presenter.getStats();
  assert.equal(statsAfterReset.uploadedFrames, 0);
  assert.equal(statsAfterReset.presentedFrames, 0);
  assert.equal(statsAfterReset.droppedFrames, 0);
  assert.equal(statsAfterReset.bytesUploaded, 0);
  assert.equal(statsAfterReset.bindGroupCacheHits, 0);
  assert.equal(statsAfterReset.bindGroupCacheMisses, 0);
  assert.equal(statsAfterReset.uniformWritesSkipped, 0);
  assert.equal(statsAfterReset.resizeCount, 0);
  assert.equal(statsAfterReset.planeReallocations, 0);
  assert.equal(statsAfterReset.scratchAllocations, 0);
  // Configuration and cached GPU state remain intact.
  assert.equal(statsAfterReset.targetFps, 30);
  assert.equal(statsAfterReset.chromaOrder, 'yvu');
  assert.equal(statsAfterReset.frameWidth, 3840);
  assert.equal(statsAfterReset.frameHeight, 2160);
  assert.equal(statsAfterReset.hasFrame, true);
  assert.equal(statsAfterReset.pipelineReady, true);

  // Subsequent 4K frames reuse all GPU resources with 0 allocations and track fresh pacing metrics.
  clockMs = 2000;
  uploadStepMs = 1.0; // 3 planes = 3.0ms
  presenter.uploadYuv420Frame({ y, u: cr, v: cb, width: 3840, height: 2160 });
  presenter.presentFrame(createMockCommandEncoder(), targetView, 3840, 2160);

  assert.equal(device.createdTextures.length, texCountBefore);
  assert.equal(device.createdViews.length, viewCountBefore);
  assert.equal(device.createdBindGroups.length, bgCountBefore);
  assert.equal(device.writeBufferCalls.length, uniformWritesBefore);
  assert.equal(presenter.getPacingStats().presentedFrames, 1);
  assert.equal(presenter.getPacingStats().lastFrameIntervalMs, 0); // fresh baseline after resetStats()
  assertNear(presenter.getPacingStats().avgUploadMs, 3.0, 1e-6);
  assertNear(presenter.getPacingStats().maxUploadMs, 3.0, 1e-6);
  assert.equal(presenter.getStats().bindGroupCacheHits, 1);
  assert.equal(presenter.getStats().uniformWritesSkipped, 1);

  clockMs = 2035.0; // upload takes 3.0ms -> present at 2038.0 vs previous 2003.0 -> 35.0ms interval
  presenter.uploadYuv420Frame({ y, u: cr, v: cb, width: 3840, height: 2160 });
  presenter.presentFrame(createMockCommandEncoder(), targetView, 3840, 2160);
  assert.equal(presenter.getPacingStats().presentedFrames, 2);
  assertNear(presenter.getPacingStats().lastFrameIntervalMs, 35.0, 1e-6);

  presenter.destroy();
  assert.throws(() => presenter.resetStats(), /destroyed/i);
});

test('W280: bounded LRU GPUBindGroup cache hits across repeated 4K frames, evicts LRU at capacity, and reuses upload/present scratch descriptors', () => {
  assert.equal(DEFAULT_BIND_GROUP_CACHE_CAPACITY, 16);
  assertNear(TARGET_FPS_30_INTERVAL_MS, 1000 / 30, 1e-9);

  const device = createMockWebGpuDevice();
  assert.throws(
    () => new BinkVideoPresenter({ device, bindGroupCacheCapacity: 0 }),
    RangeError
  );
  assert.throws(
    () => new BinkVideoPresenter({ device, bindGroupCacheCapacity: -4 }),
    RangeError
  );
  assert.throws(
    () => new BinkVideoPresenter({ device, bindGroupCacheCapacity: 2.5 }),
    RangeError
  );

  // 1) Double-buffered 4K sustained playback: 2 warm misses + 18 cache hits = 0.90 hit rate,
  //    zero extra bind groups created, and upload/present result & descriptor objects reused.
  const presenter = new BinkVideoPresenter({
    device,
    doubleBuffer: true,
    chromaOrder: BINK_CHROMA_ORDER.YVU,
  });
  const targetView = { kind: 'GPUTextureView', id: 'swap-w280-4k' };
  const y = new Uint8Array(3840 * 2160);
  const cr = new Uint8Array(1920 * 1080);
  const cb = new Uint8Array(1920 * 1080);

  let firstUploadRes = null;
  let firstPresentRes = null;
  let firstPassDesc = null;
  for (let i = 0; i < 20; i += 1) {
    const upRes = presenter.uploadYuv420Frame({ y, u: cr, v: cb, width: 3840, height: 2160 });
    const enc = createMockCommandEncoder();
    const presRes = presenter.present(enc, targetView, 3840, 2160, { nowMs: 1000 + i * 33 });
    if (i === 0) {
      firstUploadRes = upRes;
      firstPresentRes = presRes;
      firstPassDesc = enc.renderPasses[0].descriptor;
    } else {
      assert.equal(upRes, firstUploadRes, `frame ${i} uploadYuv420Frame must reuse result scratch`);
      assert.equal(presRes, firstPresentRes, `frame ${i} present must reuse result scratch`);
      assert.equal(
        enc.renderPasses[0].descriptor,
        firstPassDesc,
        `frame ${i} present must reuse renderPassDescriptor scratch`
      );
    }
  }

  const stats = presenter.getStats();
  assert.equal(device.createdBindGroups.length, 2);
  assert.equal(stats.bindGroupCacheMisses, 2);
  assert.equal(stats.bindGroupCacheHits, 18);
  assertNear(stats.bindGroupHitRate, 18 / 20, 1e-9);
  assertNear(presenter.bindGroupHitRate, 18 / 20, 1e-9);
  assert.equal(stats.bindGroupCacheSize, 2);
  assert.equal(stats.bindGroupCacheCapacity, 16);
  assert.ok(stats.bindGroupHitRate >= 0 && stats.bindGroupHitRate <= 1);
  assertNear(stats.targetFpsHitRate30, 1.0, 1e-9);
  presenter.destroy();

  // 2) Bounded LRU eviction at capacity (3 entries across 4 slots).
  const lruDevice = createMockWebGpuDevice();
  const lruPresenter = new BinkVideoPresenter({
    device: lruDevice,
    slotCount: 4,
    bindGroupCacheCapacity: 3,
  });
  const smallY = new Uint8Array(16 * 8);
  const smallC = new Uint8Array(8 * 4);
  const smallTarget = { kind: 'GPUTextureView', id: 'swap-w280-lru' };

  // Fill slots 0, 1, 2 -> 3 misses, cache order MRU->LRU is [2, 1, 0].
  for (let i = 0; i < 3; i += 1) {
    lruPresenter.uploadYuv420Frame({ y: smallY, u: smallC, v: smallC, width: 16, height: 8 });
    lruPresenter.presentFrame(createMockCommandEncoder(), smallTarget, 320, 180);
  }
  assert.equal(lruDevice.createdBindGroups.length, 3);
  assert.equal(lruPresenter.bindGroupCacheSize, 3);
  assert.equal(lruPresenter.bindGroupCacheMisses, 3);
  assert.equal(lruPresenter.bindGroupCacheHits, 0);

  // Present slot 3 -> evicts LRU (slot 0), cache stays bounded at 3 entries [3, 2, 1].
  lruPresenter.uploadYuv420Frame({ y: smallY, u: smallC, v: smallC, width: 16, height: 8 });
  lruPresenter.presentFrame(createMockCommandEncoder(), smallTarget, 320, 180);
  assert.equal(lruDevice.createdBindGroups.length, 4);
  assert.equal(lruPresenter.bindGroupCacheSize, 3);
  assert.equal(lruPresenter.bindGroupCacheMisses, 4);

  // Cycling back to slot 0 misses because slot 0 was evicted as LRU; cache stays bounded at 3.
  lruPresenter.uploadYuv420Frame({ y: smallY, u: smallC, v: smallC, width: 16, height: 8 });
  lruPresenter.presentFrame(createMockCommandEncoder(), smallTarget, 320, 180);
  assert.equal(lruDevice.createdBindGroups.length, 5);
  assert.equal(lruPresenter.bindGroupCacheSize, 3);
  assert.equal(lruPresenter.bindGroupCacheMisses, 5);

  // Presenting the active slot (slot 0) again hits MRU immediately.
  lruPresenter.presentFrame(createMockCommandEncoder(), smallTarget, 320, 180);
  assert.equal(lruDevice.createdBindGroups.length, 5);
  assert.equal(lruPresenter.bindGroupCacheHits, 1);
  assertNear(lruPresenter.bindGroupHitRate, 1 / 6, 1e-9);
  lruPresenter.destroy();

  // 3) Default 16-entry LRU bound with 17 distinct slots.
  const cap16Device = createMockWebGpuDevice();
  const cap16Presenter = new BinkVideoPresenter({
    device: cap16Device,
    slotCount: 17,
  });
  for (let i = 0; i < 17; i += 1) {
    cap16Presenter.uploadYuv420Frame({ y: smallY, u: smallC, v: smallC, width: 16, height: 8 });
    cap16Presenter.presentFrame(createMockCommandEncoder(), smallTarget, 320, 180);
  }
  assert.equal(cap16Presenter.bindGroupCacheCapacity, 16);
  assert.equal(cap16Presenter.bindGroupCacheSize, 16);
  assert.equal(cap16Device.createdBindGroups.length, 17);
  cap16Presenter.destroy();
});

test('W280: bindGroup cache invalidates automatically on plane view/sampler/uniformBuffer/device changes and on resize(), reset(), destroy()', () => {
  const device1 = createMockWebGpuDevice();
  const presenter = new BinkVideoPresenter({
    device: device1,
    doubleBuffer: false,
  });
  const targetView = { kind: 'GPUTextureView', id: 'swap-w280-inval' };
  const y = new Uint8Array(32 * 16).fill(180);
  const u = new Uint8Array(16 * 8).fill(128);
  const v = new Uint8Array(16 * 8).fill(128);

  // Warm frame + repeated present -> 1 miss, 1 hit.
  presenter.uploadYuv420Frame({ y, u, v, width: 32, height: 16 });
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360);
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360);
  assert.equal(device1.createdBindGroups.length, 1);
  assert.equal(presenter.bindGroupCacheMisses, 1);
  assert.equal(presenter.bindGroupCacheHits, 1);
  assertNear(presenter.bindGroupHitRate, 0.5, 1e-9);

  // Replace Y plane view identity without bumping slot generation -> automatic cache miss & rebind!
  const replacedYView = { kind: 'GPUTextureView', id: 'replaced-y-view' };
  presenter._slots[0].y.view = replacedYView;
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360);
  assert.equal(device1.createdBindGroups.length, 2);
  assert.equal(presenter.bindGroupCacheMisses, 2);
  assert.equal(
    device1.createdBindGroups[1].desc.entries.find((e) => e.binding === 2).resource,
    replacedYView
  );
  // Repeating with replacedYView hits cache cleanly.
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360);
  assert.equal(device1.createdBindGroups.length, 2);
  assert.equal(presenter.bindGroupCacheHits, 2);

  // Replace U plane view and V plane view -> each triggers an automatic cache miss!
  const replacedUView = { kind: 'GPUTextureView', id: 'replaced-u-view' };
  presenter._slots[0].u.view = replacedUView;
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360);
  assert.equal(device1.createdBindGroups.length, 3);
  assert.equal(
    device1.createdBindGroups[2].desc.entries.find((e) => e.binding === 3).resource,
    replacedUView
  );

  const replacedVView = { kind: 'GPUTextureView', id: 'replaced-v-view' };
  presenter._slots[0].v.view = replacedVView;
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360);
  assert.equal(device1.createdBindGroups.length, 4);
  assert.equal(
    device1.createdBindGroups[3].desc.entries.find((e) => e.binding === 4).resource,
    replacedVView
  );

  // Replace sampler identity -> automatic cache miss & rebind!
  const replacedSampler = { kind: 'GPUSampler', id: 'replaced-sampler' };
  presenter._sampler = replacedSampler;
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360);
  assert.equal(device1.createdBindGroups.length, 5);
  assert.equal(
    device1.createdBindGroups[4].desc.entries.find((e) => e.binding === 1).resource,
    replacedSampler
  );

  // Replace uniformBuffer identity -> automatic cache miss & rebind!
  const replacedUniformBuf = device1.createBuffer({
    label: 'replaced-uniforms',
    size: UNIFORM_BUFFER_SIZE,
    usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST,
  });
  presenter._uniformBuffer = replacedUniformBuf;
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360);
  assert.equal(device1.createdBindGroups.length, 6);
  assert.equal(
    device1.createdBindGroups[5].desc.entries.find((e) => e.binding === 0).resource.buffer,
    replacedUniformBuf
  );

  // resize() clears bindGroup cache cleanly while keeping active frame renderable.
  assert.throws(() => presenter.resize(0, 360), RangeError);
  assert.throws(() => presenter.resize(640, -1), RangeError);
  presenter.resize(1280, 720);
  assert.equal(presenter.bindGroupCacheSize, 0);
  const resAfterResize = presenter.presentFrame(createMockCommandEncoder(), targetView, 1280, 720);
  assert.equal(resAfterResize.presented, true);
  assert.equal(device1.createdBindGroups.length, 7);
  assert.equal(presenter.bindGroupCacheSize, 1);

  // reset() destroys old slot textures and clears bindGroup cache while keeping same-device pipeline.
  const texturesBeforeReset = device1.createdTextures.slice(0, 3);
  presenter.reset();
  assert.equal(presenter.bindGroupCacheSize, 0);
  assert.equal(presenter.getStats().hasFrame, false);
  assert.equal(presenter.getStats().pipelineReady, true);
  for (const tex of texturesBeforeReset) {
    assert.equal(tex.destroyed, true);
  }
  presenter.uploadYuv420Frame({ y, u, v, width: 32, height: 16 });
  presenter.presentFrame(createMockCommandEncoder(), targetView, 1280, 720);
  assert.equal(device1.createdBindGroups.length, 8);
  assert.equal(presenter.bindGroupCacheSize, 1);

  // Device change via setDevice(device2) releases old device resources and invalidates cache.
  const device2 = createMockWebGpuDevice();
  assert.throws(() => presenter.setDevice(null), TypeError);
  assert.throws(() => presenter.setDevice({}), TypeError);
  presenter.setDevice(device2);
  assert.equal(presenter.bindGroupCacheSize, 0);
  assert.equal(presenter.getStats().pipelineReady, false);
  assert.equal(presenter.getStats().hasFrame, false);

  presenter.uploadYuv420Frame({ y, u, v, width: 32, height: 16 });
  presenter.presentFrame(createMockCommandEncoder(), targetView, 1280, 720);
  assert.equal(device2.createdRenderPipelines.length, 1);
  assert.equal(device2.createdBindGroups.length, 1);
  assert.equal(presenter.bindGroupCacheSize, 1);
  presenter.presentFrame(createMockCommandEncoder(), targetView, 1280, 720);
  assert.equal(device2.createdBindGroups.length, 1);

  // Device change via reset(device3) also rebuilds cleanly on device3.
  const device3 = createMockWebGpuDevice();
  assert.throws(() => presenter.reset({}), TypeError);
  presenter.reset(device3);
  assert.equal(presenter.bindGroupCacheSize, 0);
  assert.equal(presenter.getStats().pipelineReady, false);
  presenter.uploadYuv420Frame({ y, u, v, width: 32, height: 16 });
  presenter.presentFrame(createMockCommandEncoder(), targetView, 1280, 720);
  assert.equal(device3.createdBindGroups.length, 1);
  assert.equal(presenter.bindGroupCacheSize, 1);

  // destroy() clears bindGroup cache and blocks subsequent resize()/reset()/setDevice()/present().
  presenter.destroy();
  assert.equal(presenter.bindGroupCacheSize, 0);
  assert.equal(presenter.getStats().bindGroupCacheSize, 0);
  assert.throws(() => presenter.resize(640, 360), /destroyed/i);
  assert.throws(() => presenter.reset(), /destroyed/i);
  assert.throws(() => presenter.setDevice(device1), /destroyed/i);
  assert.throws(
    () => presenter.present(createMockCommandEncoder(), targetView, 640, 360),
    /destroyed/i
  );
});

test('W280: targetFpsHitRate30 tracks <= 33.333ms intervals, clamps to [0, 1], and resets cleanly in resetStats()', () => {
  const device = createMockWebGpuDevice();
  const presenter = new BinkVideoPresenter({
    device,
    doubleBuffer: false,
    targetFps: 30,
  });
  const targetView = { kind: 'GPUTextureView', id: 'swap-w280-fps30' };
  const y = new Uint8Array(16 * 8).fill(180);
  const u = new Uint8Array(8 * 4).fill(128);
  const v = new Uint8Array(8 * 4).fill(128);

  // Before any frames: both hit rates are 0.
  let stats = presenter.getStats();
  assert.equal(stats.bindGroupHitRate, 0);
  assert.equal(stats.targetFpsHitRate30, 0);
  assert.equal(presenter.bindGroupHitRate, 0);
  assert.equal(presenter.targetFpsHitRate30, 0);

  presenter.uploadYuv420Frame({ y, u, v, width: 16, height: 8 });

  // Frame 0 at t=1000.0 (initial baseline, 0 measured intervals -> targetFpsHitRate30 === 0).
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360, { nowMs: 1000.0 });
  assert.equal(presenter.getStats().targetFpsHitRate30, 0);
  assert.equal(presenter.getStats().bindGroupHitRate, 0);

  // Frame 1 at t=1016.666 (delta=16.666ms <= 33.333ms -> hit 1/1 = 1.0).
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360, { nowMs: 1016.666 });
  assertNear(presenter.getStats().targetFpsHitRate30, 1.0, 1e-9);
  assertNear(presenter.getStats().bindGroupHitRate, 0.5, 1e-9);

  // Frame 2 at t=1049.999 (delta=33.333ms <= 33.3333...ms -> hit 2/2 = 1.0).
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360, { nowMs: 1049.999 });
  assertNear(presenter.getStats().targetFpsHitRate30, 1.0, 1e-9);

  // Frame 3 at t=1083.333 (delta=33.334ms > 1000/30 -> miss 2/3).
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360, { nowMs: 1083.333 });
  assertNear(presenter.getStats().targetFpsHitRate30, 2 / 3, 1e-9);

  // Frame 4 at t=1133.333 (delta=50.0ms > 33.333ms -> miss 2/4 = 0.5).
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360, { nowMs: 1133.333 });
  assertNear(presenter.getStats().targetFpsHitRate30, 0.5, 1e-9);

  // Backwards clock (delta < 0) and zero-delta do not corrupt measured frame interval count.
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360, { nowMs: 1100.0 });
  assertNear(presenter.getStats().targetFpsHitRate30, 0.5, 1e-9);
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360, { nowMs: 1100.0 });
  assertNear(presenter.getStats().targetFpsHitRate30, 0.5, 1e-9);

  // Frame 7 at t=1130.0 (delta=30.0ms <= 33.333ms -> hit 3/5 = 0.6).
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360, { nowMs: 1130.0 });
  stats = presenter.getStats();
  assertNear(stats.targetFpsHitRate30, 0.6, 1e-9);
  assertNear(presenter.targetFpsHitRate30, 0.6, 1e-9);
  assert.ok(stats.targetFpsHitRate30 >= 0 && stats.targetFpsHitRate30 <= 1);
  assert.ok(stats.bindGroupHitRate >= 0 && stats.bindGroupHitRate <= 1);

  // resetStats() clears bindGroupHitRate and targetFpsHitRate30 while preserving cached bind groups.
  const bgCountBeforeReset = device.createdBindGroups.length;
  presenter.resetStats();
  stats = presenter.getStats();
  assert.equal(stats.bindGroupCacheHits, 0);
  assert.equal(stats.bindGroupCacheMisses, 0);
  assert.equal(stats.bindGroupHitRate, 0);
  assert.equal(stats.targetFpsHitRate30, 0);
  assert.equal(stats.bindGroupCacheSize, 1);

  // Fresh timing baseline + 100% cache hit after resetStats().
  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360, { nowMs: 2000.0 });
  assert.equal(device.createdBindGroups.length, bgCountBeforeReset);
  assert.equal(presenter.getStats().bindGroupHitRate, 1);
  assert.equal(presenter.getStats().targetFpsHitRate30, 0);

  presenter.presentFrame(createMockCommandEncoder(), targetView, 640, 360, { nowMs: 2032.0 });
  assert.equal(presenter.getStats().bindGroupHitRate, 1);
  assert.equal(presenter.getStats().targetFpsHitRate30, 1);

  presenter.destroy();
});
