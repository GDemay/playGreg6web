import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const binkMod = require('../web/workers/bink_presenter.js');
const stagingMod = require('../web/workers/staging_pool.js');

const {
  WEBGPU_BYTES_PER_ROW_ALIGNMENT,
  UNIFORM_BUFFER_SIZE,
  BINK_PIXEL_FORMAT,
  YUV_COLOR_SPACE,
  YUV_MATRICES,
  TEXTURE_USAGE,
  BUFFER_USAGE,
  BINK_PRESENTER_WGSL,
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
      writeBufferCalls.push({
        buffer,
        bufferOffset,
        data: bytes,
      });
    },
    onSubmittedWorkDone() {
      return Promise.resolve();
    },
  };

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
