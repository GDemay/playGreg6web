import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const binkMod = require('../web/workers/bink_presenter.js');
const bridgeMod = require('../web/workers/splash_frame_bridge.js');

const {
  BinkVideoPresenter,
  BINK_PIXEL_FORMAT,
  YUV_COLOR_SPACE,
  YUV_COLOR_RANGE,
  UNIFORM_FLAG_OUTPUT_LINEAR,
  BINK_PRESENTER_WGSL,
  isLinearPresentationFormat,
  packPresenterUniforms,
} = binkMod;

const { SplashFrameBridge, PIXEL_FORMAT } = bridgeMod;

function createMockDevice() {
  const createdTextures = [];
  const createdPipelines = [];
  const createdShaderModules = [];
  const createdUniformBuffers = [];
  const writeTextureCalls = [];
  const writeBufferCalls = [];
  const copyExternalImageCalls = [];
  const scissorCalls = [];
  let nextId = 1;

  const device = {
    createdTextures,
    createdPipelines,
    createdShaderModules,
    createdUniformBuffers,
    writeTextureCalls,
    writeBufferCalls,
    copyExternalImageCalls,
    scissorCalls,
    queue: {
      writeTexture(dest, data, layout, size) {
        writeTextureCalls.push({
          texture: dest.texture,
          data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice(),
          layout: { ...layout },
          size: { ...size },
        });
      },
      writeBuffer(buffer, offset, data) {
        const view = new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
        writeBufferCalls.push({ buffer, offset, data: view });
      },
      copyExternalImageToTexture(source, dest, size) {
        // Enforce WebGPU spec: copyExternalImageToTexture requires RENDER_ATTACHMENT (0x10).
        if (((dest.texture.usage || 0) & 0x10) === 0) {
          throw new Error(
            `WebGPU validation: copyExternalImageToTexture destination texture lacks RENDER_ATTACHMENT (usage=0x${(dest.texture.usage || 0).toString(16)})`
          );
        }
        copyExternalImageCalls.push({
          source: { ...source },
          texture: dest.texture,
          size: { ...size },
        });
      },
      submit() {},
    },
    createTexture(desc) {
      const tex = {
        id: nextId++,
        label: desc.label,
        format: desc.format,
        usage: desc.usage,
        size: { ...desc.size },
        destroyed: false,
        createView() {
          return { kind: 'GPUTextureView', textureId: tex.id, format: tex.format };
        },
        destroy() {
          this.destroyed = true;
        },
      };
      createdTextures.push(tex);
      return tex;
    },
    createShaderModule(desc) {
      const mod = { id: nextId++, ...desc };
      createdShaderModules.push(mod);
      return mod;
    },
    createBindGroupLayout(desc) {
      return { id: nextId++, ...desc };
    },
    createPipelineLayout(desc) {
      return { id: nextId++, ...desc };
    },
    createRenderPipeline(desc) {
      const pipe = {
        id: nextId++,
        targetFormat: desc.fragment.targets[0].format,
        desc,
      };
      createdPipelines.push(pipe);
      return pipe;
    },
    createSampler(desc) {
      return { id: nextId++, ...desc };
    },
    createBuffer(desc) {
      const buf = { id: nextId++, ...desc, destroy() {} };
      if (desc.label && desc.label.includes('uniform')) {
        createdUniformBuffers.push(buf);
      }
      return buf;
    },
    createBindGroup(desc) {
      return { id: nextId++, ...desc };
    },
    createCommandEncoder() {
      return {
        beginRenderPass() {
          return {
            setPipeline() {},
            setViewport() {},
            setScissorRect(x, y, w, h) {
              scissorCalls.push({ x, y, w, h });
            },
            setBindGroup() {},
            draw() {},
            end() {},
          };
        },
        finish() {
          return { kind: 'GPUCommandBuffer' };
        },
      };
    },
  };
  return device;
}

test('sRGB / linear output conversion prevents double-gamma washout and updates across setPresentationFormat()', () => {
  assert.equal(isLinearPresentationFormat('bgra8unorm'), false);
  assert.equal(isLinearPresentationFormat('rgba8unorm'), false);
  assert.equal(isLinearPresentationFormat('bgra8unorm-srgb'), true);
  assert.equal(isLinearPresentationFormat('rgba8unorm-srgb'), true);
  assert.equal(isLinearPresentationFormat('rgba16float'), true);

  // Uniform flag bit 2 (UNIFORM_FLAG_OUTPUT_LINEAR = 4) follows presentationFormat.
  const uUnorm = new Uint32Array(
    packPresenterUniforms({
      pixelFormat: BINK_PIXEL_FORMAT.YUV420P,
      presentationFormat: 'bgra8unorm',
    }).buffer
  );
  assert.equal((uUnorm[3] & UNIFORM_FLAG_OUTPUT_LINEAR) !== 0, false);

  const uSrgb = new Uint32Array(
    packPresenterUniforms({
      pixelFormat: BINK_PIXEL_FORMAT.YUV420P,
      presentationFormat: 'bgra8unorm-srgb',
    }).buffer
  );
  assert.equal((uSrgb[3] & UNIFORM_FLAG_OUTPUT_LINEAR) !== 0, true);

  const uHdr = new Uint32Array(
    packPresenterUniforms({
      pixelFormat: BINK_PIXEL_FORMAT.RGBA8,
      presentationFormat: 'rgba16float',
    }).buffer
  );
  assert.equal((uHdr[3] & UNIFORM_FLAG_OUTPUT_LINEAR) !== 0, true);

  // WGSL shader implements IEC 61966-2-1 srgb_to_linear and gates it on flags & 4u.
  assert.match(BINK_PRESENTER_WGSL, /fn\s+srgb_to_linear\s*\(/);
  assert.match(BINK_PRESENTER_WGSL, /12\.92/);
  assert.match(BINK_PRESENTER_WGSL, /0\.04045/);
  assert.match(BINK_PRESENTER_WGSL, /uniforms\.flags\s*&\s*4u/);

  // Switching presentationFormat lazily rebuilds ONLY the render pipeline (reusing shader & uniform buffer)
  // and toggles UNIFORM_FLAG_OUTPUT_LINEAR in the uploaded uniform buffer.
  const device = createMockDevice();
  const presenter = new BinkVideoPresenter({ device, presentationFormat: 'bgra8unorm' });
  const rgba = new Uint8Array(4 * 4 * 4).fill(128);
  presenter.uploadRgbaFrame(rgba, 4, 4);

  const enc1 = device.createCommandEncoder();
  presenter.presentFrame(enc1, { kind: 'GPUTextureView' }, 64, 64);
  assert.equal(device.createdPipelines.length, 1);
  assert.equal(device.createdPipelines[0].targetFormat, 'bgra8unorm');
  assert.equal(device.createdShaderModules.length, 1);
  assert.equal(device.createdUniformBuffers.length, 1);
  const firstFlags = new Uint32Array(device.writeBufferCalls.at(-1).data.buffer)[3];
  assert.equal((firstFlags & UNIFORM_FLAG_OUTPUT_LINEAR) !== 0, false);

  presenter.setPresentationFormat('bgra8unorm-srgb');
  const enc2 = device.createCommandEncoder();
  presenter.presentFrame(enc2, { kind: 'GPUTextureView' }, 64, 64);
  assert.equal(device.createdPipelines.length, 2);
  assert.equal(device.createdPipelines[1].targetFormat, 'bgra8unorm-srgb');
  assert.equal(device.createdShaderModules.length, 1, 'shader module should be reused');
  assert.equal(device.createdUniformBuffers.length, 1, 'uniform buffer should be reused');
  const secondFlags = new Uint32Array(device.writeBufferCalls.at(-1).data.buffer)[3];
  assert.equal((secondFlags & UNIFORM_FLAG_OUTPUT_LINEAR) !== 0, true);

  presenter.destroy();
});

test('WGSL corrects odd-dimension chroma scale and crop-edge half-texel clamping', () => {
  assert.match(BINK_PRESENTER_WGSL, /textureDimensions\(plane_primary\)/);
  assert.match(BINK_PRESENTER_WGSL, /textureDimensions\(plane_u\)/);
  assert.match(BINK_PRESENTER_WGSL, /chroma_scale\s*=\s*luma_dim\s*\/\s*max\(chroma_dim\s*\*\s*2\.0/);
  assert.match(BINK_PRESENTER_WGSL, /luma_half\s*=\s*vec2<f32>\(0\.5\)/);
  assert.match(BINK_PRESENTER_WGSL, /chroma_half\s*=\s*vec2<f32>\(0\.5\)/);
});

test('BGRA8 frames use native bgra8unorm plane texture without R/B swap and scissor avoids 1px fractional clip', () => {
  const device = createMockDevice();
  const bridge = new SplashFrameBridge({ device, targetFps: 120 });

  // 16:9 frame presented into 100x100 yields fractional letterbox y = 21.875, height = 56.25
  // -> y + height = 78.125 -> sy = 21, ceil(78.125) - 21 = 58 (old ceil(56.25) = 57 clipped 1px!).
  const bgra = new Uint8Array(16 * 9 * 4);
  for (let i = 0; i < bgra.length; i += 4) {
    bgra[i] = 10; // B
    bgra[i + 1] = 20; // G
    bgra[i + 2] = 240; // R
    bgra[i + 3] = 255; // A
  }
  bridge.pushRawFrame({
    width: 16,
    height: 9,
    format: PIXEL_FORMAT.BGRA8,
    planes: { bgra },
    timestampMs: 0,
  });

  const res = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 100, 100, 0);
  assert.equal(res.presented, true);
  assert.equal(bridge.presenter.pixelFormat, BINK_PIXEL_FORMAT.BGRA8);
  const bgraTex = device.createdTextures.find((t) => t.format === 'bgra8unorm');
  assert.ok(bgraTex, 'BGRA8 frame must create a bgra8unorm plane texture so R/B channels are not swapped');

  const scissor = device.scissorCalls.at(-1);
  assert.equal(scissor.y, 21);
  assert.equal(scissor.h, 58, 'fractional letterbox scissor height must cover ceil(y + height) - floor(y)');

  bridge.destroy();
});

test('SplashFrameBridge external ImageBitmap satisfies RENDER_ATTACHMENT and presents GPU copy directly when CPU data is absent', () => {
  const device = createMockDevice();
  const bridge = new SplashFrameBridge({ device, targetFps: 120 });

  const gpuOnlyBitmap = {
    width: 8,
    height: 8,
    closeCalls: 0,
    close() {
      this.closeCalls += 1;
    },
  };

  bridge.pushExternalImage(gpuOnlyBitmap, { timestampMs: 0 });
  const res = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 128, 128, 0);
  assert.equal(res.presented, true);
  assert.equal(device.copyExternalImageCalls.length, 1);
  assert.equal(device.copyExternalImageCalls[0].source.source, gpuOnlyBitmap);
  assert.notEqual(
    device.copyExternalImageCalls[0].texture.usage & 0x10,
    0,
    'copyExternalImageToTexture destination must include RENDER_ATTACHMENT'
  );
  // Must NOT have uploaded a synthetic solid-white 8x8 RGBA buffer via queue.writeTexture!
  const solidWhiteUpload = device.writeTextureCalls.find(
    (c) => c.size.width === 8 && c.size.height === 8
  );
  assert.equal(solidWhiteUpload, undefined, 'GPU-only ImageBitmap must not be overwritten with solid white');

  bridge.destroy();
});

test('SplashFrameBridge does not leak per-frame colorSpace/colorRange overrides into subsequent default frames', () => {
  const device = createMockDevice();
  const bridge = new SplashFrameBridge({
    device,
    colorSpace: YUV_COLOR_SPACE.BT709,
    colorRange: YUV_COLOR_RANGE.LIMITED,
    targetFps: 120,
  });

  const makeYuv = (w, h) => ({
    y: new Uint8Array(w * h).fill(128),
    u: new Uint8Array((w / 2) * (h / 2)).fill(128),
    v: new Uint8Array((w / 2) * (h / 2)).fill(128),
  });

  // Frame 0 explicitly overrides to bt601 + full range.
  bridge.pushRawFrame({
    width: 16,
    height: 8,
    format: PIXEL_FORMAT.YUV420P,
    planes: makeYuv(16, 8),
    colorSpace: YUV_COLOR_SPACE.BT601,
    colorRange: YUV_COLOR_RANGE.FULL,
    timestampMs: 0,
  });
  const r0 = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 160, 80, 0);
  assert.equal(r0.colorSpace, YUV_COLOR_SPACE.BT601);
  assert.equal(r0.colorRange, YUV_COLOR_RANGE.FULL);
  assert.equal(bridge.presenter.colorSpace, YUV_COLOR_SPACE.BT601);
  assert.equal(bridge.presenter.colorRange, YUV_COLOR_RANGE.FULL);

  // Frame 1 omits colorSpace/colorRange and must restore bridge defaults (bt709 + limited).
  bridge.pushRawFrame({
    width: 16,
    height: 8,
    format: PIXEL_FORMAT.YUV420P,
    planes: makeYuv(16, 8),
    timestampMs: 20,
  });
  const r1 = bridge.renderFrame(device, { kind: 'GPUTextureView' }, 160, 80, 20);
  assert.equal(r1.colorSpace, YUV_COLOR_SPACE.BT709);
  assert.equal(r1.colorRange, YUV_COLOR_RANGE.LIMITED);
  assert.equal(bridge.presenter.colorSpace, YUV_COLOR_SPACE.BT709);
  assert.equal(bridge.presenter.colorRange, YUV_COLOR_RANGE.LIMITED);

  // setPresentationFormat propagates through SplashFrameBridge to presenter.
  bridge.setPresentationFormat('rgba16float');
  assert.equal(bridge.presenter.presentationFormat, 'rgba16float');

  bridge.destroy();
});
