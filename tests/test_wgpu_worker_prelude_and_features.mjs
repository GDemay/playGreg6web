import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

import {
  RT_BVH_PRELUDE_VERSION,
  RT_BVH_PRELUDE_WGSL,
  applyRtBvhPrelude,
  needsRtBvhPrelude,
} from '../web/workers/rt_bvh_prelude.js';

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WORKERS_DIR = path.resolve(__dirname, '..', 'web', 'workers');

const {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  OPTIONAL_ADAPTER_FEATURES,
  selectSupportedAdapterFeatures,
  INLINE_RT_BVH_PRELUDE_VERSION,
  OPCODES,
  SHADER_STAGE,
} = require('../web/workers/wgpu_worker.js');
const { WgpuSwapchain } = require('../web/workers/wgpu_swapchain.js');
const { BinkVideoPresenter } = require('../web/workers/bink_presenter.js');
const { SplashFrameBridge } = require('../web/workers/splash_frame_bridge.js');
const { WebGpuPipelineCache } = require('../web/workers/pipeline_cache.js');

const PACKET_HEADER_SIZE = 8;

function align4(n) {
  return (n + 3) & ~3;
}

function buildCreateShaderModulePacket(id, stage, wgsl) {
  const wgslBytes = new TextEncoder().encode(wgsl);
  const payloadBytes = 16 + wgslBytes.byteLength;
  const buf = new ArrayBuffer(PACKET_HEADER_SIZE + payloadBytes);
  const dv = new DataView(buf);
  dv.setUint16(0, OPCODES.CREATE_SHADER_MODULE, true);
  dv.setUint16(2, 0, true);
  dv.setUint32(4, payloadBytes, true);
  dv.setUint32(8, id, true);
  dv.setUint32(12, stage, true);
  dv.setUint32(16, wgslBytes.byteLength, true);
  dv.setUint32(20, 0, true);
  new Uint8Array(buf, 24).set(wgslBytes);
  return new Uint8Array(buf);
}

function createMockGpuDevice(featuresList = []) {
  const createdModules = [];
  const createdRenderPipelines = [];
  const createdTextures = [];
  const createdBuffers = [];
  const configured = [];

  const device = {
    features: new Set(featuresList),
    limits: {},
    queue: {
      submit() {},
      writeBuffer() {},
      writeTexture() {},
      onSubmittedWorkDone: () => Promise.resolve(),
    },
    createShaderModule(desc) {
      const mod = { __desc: desc, code: desc.code, label: desc.label };
      createdModules.push(mod);
      return mod;
    },
    createBuffer(desc) {
      const buf = {
        size: desc.size,
        usage: desc.usage,
        destroyed: false,
        destroy() {
          this.destroyed = true;
        },
        getMappedRange() {
          return new ArrayBuffer(desc.size);
        },
        unmap() {},
      };
      createdBuffers.push(buf);
      return buf;
    },
    createTexture(desc) {
      const tex = {
        width: desc.size?.width ?? desc.size?.[0] ?? 1,
        height: desc.size?.height ?? desc.size?.[1] ?? 1,
        format: desc.format,
        usage: desc.usage,
        destroyed: false,
        createView(viewDesc) {
          return { __texture: tex, __viewDesc: viewDesc || {} };
        },
        destroy() {
          this.destroyed = true;
        },
      };
      createdTextures.push(tex);
      return tex;
    },
    createSampler(desc) {
      return { __desc: desc };
    },
    createBindGroupLayout(desc) {
      return { __desc: desc };
    },
    createPipelineLayout(desc) {
      return { __desc: desc };
    },
    createBindGroup(desc) {
      return { __desc: desc };
    },
    createRenderPipeline(desc) {
      const pipe = { __desc: desc, getBindGroupLayout: () => ({}) };
      createdRenderPipelines.push(pipe);
      return pipe;
    },
    createComputePipeline(desc) {
      return { __desc: desc, getBindGroupLayout: () => ({}) };
    },
    createCommandEncoder() {
      return {
        beginRenderPass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            setVertexBuffer() {},
            setIndexBuffer() {},
            setViewport() {},
            setScissorRect() {},
            draw() {},
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
        copyTextureToTexture() {},
        finish() {
          return {};
        },
      };
    },
    pushErrorScope() {},
    popErrorScope: () => Promise.resolve(null),
    addEventListener() {},
    removeEventListener() {},
    lost: new Promise(() => {}),
  };

  const ctx = {
    configure(desc) {
      configured.push(desc);
    },
    unconfigure() {},
    getCurrentTexture() {
      const fmt = configured.length > 0 ? configured[configured.length - 1].format : 'bgra8unorm';
      return {
        width: 1280,
        height: 720,
        format: fmt,
        createView(viewDesc) {
          return { __swapchainView: true, format: viewDesc?.format || fmt };
        },
      };
    },
  };

  return { device, ctx, createdModules, createdRenderPipelines, createdTextures, createdBuffers, configured };
}

function createClassicWorkerVm({ mockGpu = null } = {}) {
  const posted = [];
  const context = {
    console,
    TextEncoder,
    TextDecoder,
    URL,
    Blob,
    AbortController,
    performance,
    setTimeout,
    clearTimeout,
    queueMicrotask,
    ArrayBuffer,
    SharedArrayBuffer,
    Uint8Array,
    Uint16Array,
    Uint32Array,
    Int32Array,
    Float32Array,
    BigUint64Array,
    DataView,
    Map,
    Set,
    WeakMap,
    WeakSet,
    Promise,
    Error,
    TypeError,
    RangeError,
    Object,
    Array,
    Number,
    String,
    Boolean,
    BigInt,
    Math,
    Date,
    JSON,
    RegExp,
    Symbol,
    GPUBufferUsage: {
      MAP_READ: 0x0001,
      MAP_WRITE: 0x0002,
      COPY_SRC: 0x0004,
      COPY_DST: 0x0008,
      INDEX: 0x0010,
      VERTEX: 0x0020,
      UNIFORM: 0x0040,
      STORAGE: 0x0080,
      INDIRECT: 0x0100,
      QUERY_RESOLVE: 0x0200,
    },
    GPUTextureUsage: {
      COPY_SRC: 0x01,
      COPY_DST: 0x02,
      TEXTURE_BINDING: 0x04,
      STORAGE_BINDING: 0x08,
      RENDER_ATTACHMENT: 0x10,
    },
    GPUShaderStage: { VERTEX: 0x1, FRAGMENT: 0x2, COMPUTE: 0x4 },
    GPUMapMode: { READ: 0x1, WRITE: 0x2 },
    navigator: mockGpu ? { gpu: mockGpu } : {},
    postMessage(data, transfer) {
      posted.push({ data, transfer });
    },
    importScripts(...scripts) {
      for (const rel of scripts) {
        const clean = String(rel).replace(/^\.\//, '');
        const fullPath = path.join(WORKERS_DIR, clean);
        const src = fs.readFileSync(fullPath, 'utf8');
        vm.runInContext(src, context, { filename: fullPath });
      }
    },
  };
  context.self = context;
  context.globalThis = context;
  vm.createContext(context);

  const workerPath = path.join(WORKERS_DIR, 'wgpu_worker.js');
  const workerSource = fs.readFileSync(workerPath, 'utf8');
  // Intentionally no require, no module, no importModuleDynamically -> simulates classic worker
  // before or without dynamic ESM import('./rt_bvh_prelude.js') resolving.
  vm.runInContext(workerSource, context, { filename: workerPath });

  return { context, posted };
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

test('Orch-35 P0 #5: classic worker synchronous RT BVH prelude fallback matches rt_bvh_prelude.js byte-for-byte with zero race window', async () => {
  assert.equal(INLINE_RT_BVH_PRELUDE_VERSION, RT_BVH_PRELUDE_VERSION);

  const { context } = createClassicWorkerVm();
  const vmApi = context.PlayGta6Wgpu;
  assert.equal(typeof vmApi.applyRtBvhPreludeIfNeeded, 'function');
  assert.equal(typeof vmApi.ensureRtBvhPreludeLoaded, 'function');

  // Verify the classic worker VM has neither require nor globalThis.PlayGta6RtBvhPrelude.
  assert.equal(context.require, undefined);
  assert.equal(context.PlayGta6RtBvhPrelude, undefined);

  const samples = [
    // Plain RT compute shader calling pg6_bvh_intersect
    `@compute @workgroup_size(64)
fn main() {
  let hit = pg6_bvh_intersect(vec4<u32>(0u), vec2<u32>(0u), 100.0, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0));
  _ = hit;
}`,
    // Directives + nested block comments + line comment + BOM calling pg6_load32
    `\uFEFF/* outer /* nested */ comment */
enable f16; // keep on same line
diagnostic(off, derivative_uniformity);
requires readonly_and_readwrite_storage_textures;
@compute @workgroup_size(1)
fn main() {
  let w = pg6_load32(vec2<u32>(0u, 0u));
  _ = w;
}`,
    // Non-RT shader (must remain untouched)
    `@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }`,
    // Substring-only identifiers (must remain untouched)
    `fn pg6_load32_stub() -> u32 { return 0u; }
fn my_pg6_bvh_intersect() -> u32 { return pg6_load32_stub(); }`,
  ];

  for (const wgsl of samples) {
    const expected = applyRtBvhPrelude(wgsl);
    const actual = vmApi.applyRtBvhPreludeIfNeeded(wgsl);
    assert.equal(actual, expected);
    // Strict idempotency check
    assert.equal(vmApi.applyRtBvhPreludeIfNeeded(actual), expected);
    if (needsRtBvhPrelude(wgsl)) {
      assert.ok(actual.includes(RT_BVH_PRELUDE_WGSL));
    }
  }

  // Synchronous WgpuCommandProcessor CreateShaderModule execution without ever calling initGpu
  const { device, ctx, createdModules } = createMockGpuDevice();
  const proc = new vmApi.WgpuCommandProcessor(device, ctx, {
    lazyShaderCompile: false,
    enableShaderFallback: false,
  });
  const rtWgsl = samples[1].replace(/^\uFEFF/, '');
  proc.executeCommandBuffer(buildCreateShaderModulePacket(77, SHADER_STAGE.COMPUTE, rtWgsl));

  assert.equal(createdModules.length, 1);
  assert.equal(createdModules[0].code, applyRtBvhPrelude(rtWgsl));
  const enableIdx = createdModules[0].code.indexOf('enable f16;');
  const preludeIdx = createdModules[0].code.indexOf('fn pg6_load32(');
  assert.ok(enableIdx >= 0 && preludeIdx > enableIdx, 'prelude must follow leading WGSL directives');

  // ensureRtBvhPreludeLoaded() resolves to a valid API even when dynamic import() is unavailable
  const loadedApi = await vmApi.ensureRtBvhPreludeLoaded();
  assert.ok(loadedApi && typeof loadedApi.applyRtBvhPrelude === 'function');
  assert.equal(loadedApi.applyRtBvhPrelude(rtWgsl), applyRtBvhPrelude(rtWgsl));

  proc.destroy();
});

test('Orch-35 P0 #5: classic worker self.onmessage init + immediate CreateShaderModule never drops RT prelude', async () => {
  const { device, ctx, createdModules } = createMockGpuDevice();
  const mockGpu = {
    async requestAdapter() {
      return {
        info: { vendor: 'mock-rdna2', architecture: 'rdna2' },
        features: new Set(OPTIONAL_ADAPTER_FEATURES),
        async requestDevice() {
          return device;
        },
      };
    },
    getPreferredCanvasFormat() {
      return 'bgra8unorm';
    },
  };

  const { context, posted } = createClassicWorkerVm({ mockGpu });
  const offscreenCanvas = {
    width: 0,
    height: 0,
    getContext(kind) {
      assert.equal(kind, 'webgpu');
      return ctx;
    },
  };

  await context.self.onmessage({
    data: {
      init: true,
      canvas: offscreenCanvas,
      width: 640,
      height: 360,
      enableSwapchain: false,
      lazyShaderCompile: false,
      enableShaderFallback: false,
    },
  });

  const ready = posted.find((m) => m.data && m.data.ready === true);
  assert.ok(ready, 'expected ready message from classic worker');

  const rtWgsl = `enable f16;
@compute @workgroup_size(64)
fn cs_rt() {
  let h = pg6_bvh_intersect(vec4<u32>(1u), vec2<u32>(2u), 50.0, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0));
  _ = h;
}`;
  const cmd = buildCreateShaderModulePacket(501, SHADER_STAGE.COMPUTE, rtWgsl);
  await context.self.onmessage({ data: { cmdBuffer: cmd.buffer } });

  assert.equal(createdModules.length, 1);
  assert.equal(createdModules[0].code, applyRtBvhPrelude(rtWgsl));
  assert.ok(createdModules[0].code.includes('fn pg6_bvh_intersect('));
  assert.ok(createdModules[0].code.includes('fn pg6_load32('));
});

test('Orch-35 P2 #10d: negotiates console-fidelity optional WebGPU features gracefully without failing on minimal adapters', async () => {
  const expectedConsoleFeatures = [
    'texture-compression-bc',
    'depth32float-stencil8',
    'float32-filterable',
    'bgra8unorm-storage',
    'rg11b10ufloat-renderable',
    'texture-formats-tier1',
    'float32-blendable',
    'dual-source-blending',
    'depth-clip-control',
  ];

  for (const feat of expectedConsoleFeatures) {
    assert.ok(
      OPTIONAL_ADAPTER_FEATURES.includes(feat),
      `OPTIONAL_ADAPTER_FEATURES must include ${feat}`
    );
  }

  // 1. Adapter supporting all console features -> requests all 9 and unlocks pipeline_cache features
  {
    const { device, ctx } = createMockGpuDevice(expectedConsoleFeatures);
    const requestDeviceCalls = [];
    const mockGpu = {
      async requestAdapter() {
        return {
          info: { vendor: 'rdna2-full', architecture: 'rdna2' },
          features: new Set(expectedConsoleFeatures),
          async requestDevice(desc) {
            requestDeviceCalls.push(desc);
            return device;
          },
        };
      },
      getPreferredCanvasFormat() {
        return 'bgra8unorm';
      },
    };
    const canvas = {
      width: 0,
      height: 0,
      getContext: () => ctx,
    };
    const posted = [];
    const handler = createWgpuWorkerHandler({
      postMessage: (m) => posted.push(m),
      enableSwapchain: false,
    });

    await withMockNavigatorGpu(mockGpu, async () => {
      await handler.handleMessage({ init: true, canvas, width: 640, height: 360 });
    });

    assert.equal(requestDeviceCalls.length, 1);
    assert.deepEqual(requestDeviceCalls[0].requiredFeatures, expectedConsoleFeatures);
    const ready = posted.find((m) => m && m.ready === true);
    assert.ok(ready);
    assert.deepEqual(ready.features, expectedConsoleFeatures);

    // Downstream WebGpuPipelineCache honors float32-blendable, dual-source-blending, depth-clip-control
    const cache = new WebGpuPipelineCache({ device });
    const vsMod = device.createShaderModule({ code: '@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }' });
    const fsMod = device.createShaderModule({ code: '@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }' });
    const pipe = cache.getOrCreateRenderPipeline({
      vertex: { module: vsMod, entryPoint: 'vs' },
      fragment: {
        module: fsMod,
        entryPoint: 'fs',
        targets: [
          {
            format: 'rgba32float',
            blend: {
              color: { srcFactor: 'src1', dstFactor: 'one-minus-src1', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'zero', operation: 'add' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list', unclippedDepth: true },
    });
    assert.ok(pipe.__desc.fragment.targets[0].blend, 'float32 blend must be preserved when float32-blendable is enabled');
    assert.equal(
      pipe.__desc.fragment.targets[0].blend.color.srcFactor,
      'src1',
      'dual-source blend factor must be preserved when dual-source-blending is enabled'
    );
    assert.equal(
      pipe.__desc.primitive.unclippedDepth,
      true,
      'unclippedDepth must be preserved when depth-clip-control is enabled'
    );
    handler.processor.destroy();
  }

  // 2. Partial adapter lacking float32-blendable / dual-source-blending / depth-clip-control
  {
    const partial = ['texture-compression-bc', 'float32-filterable', 'rg11b10ufloat-renderable'];
    assert.deepEqual(selectSupportedAdapterFeatures({ features: new Set(partial) }), partial);

    const { device, ctx } = createMockGpuDevice(partial);
    const requestDeviceCalls = [];
    const mockGpu = {
      async requestAdapter() {
        return {
          info: { vendor: 'partial-gpu', architecture: '' },
          features: new Set(partial),
          async requestDevice(desc) {
            requestDeviceCalls.push(desc);
            return device;
          },
        };
      },
      getPreferredCanvasFormat() {
        return 'bgra8unorm';
      },
    };
    const canvas = { width: 0, height: 0, getContext: () => ctx };
    const posted = [];
    const handler = createWgpuWorkerHandler({
      postMessage: (m) => posted.push(m),
      enableSwapchain: false,
    });

    await withMockNavigatorGpu(mockGpu, async () => {
      await handler.handleMessage({ init: true, canvas, width: 320, height: 180 });
    });

    assert.equal(requestDeviceCalls.length, 1);
    assert.deepEqual(requestDeviceCalls[0], { requiredFeatures: partial });
    const ready = posted.find((m) => m && m.ready === true);
    assert.deepEqual(ready.features, partial);
    handler.processor.destroy();
  }
});

test('Orch-35 Bink2 + SplashFrameBridge stay synced across setSwapchainFormat, extended HDR, legacy mock rebuild, and bindDevice', () => {
  // 1. Real WgpuSwapchain + real BinkVideoPresenter + real SplashFrameBridge: in-place format sync
  {
    const { device, ctx } = createMockGpuDevice([
      'rg11b10ufloat-renderable',
      'float32-filterable',
    ]);
    const canvas = {
      width: 1280,
      height: 720,
      getContext: () => ctx,
    };
    const swapchain = new WgpuSwapchain({
      canvas,
      device,
      context: ctx,
      format: 'bgra8unorm',
    });
    swapchain.configure({ width: 1280, height: 720, devicePixelRatio: 1 });

    const processor = new WgpuCommandProcessor(device, ctx, {
      format: 'bgra8unorm',
      swapchain,
      BinkVideoPresenter,
      SplashFrameBridge,
      binkPresenterOptions: { presentationFormat: 'rgba8unorm' }, // stale option must NOT win over processor.format
    });

    const bridge = processor._ensureSplashBridge();
    assert.ok(bridge, 'SplashFrameBridge should initialize');
    const presenter = processor.binkPresenter;
    assert.ok(presenter, 'BinkVideoPresenter should initialize');
    assert.equal(presenter.presentationFormat, 'bgra8unorm');
    assert.equal(bridge._presentationFormat, 'bgra8unorm');

    // Switch to extended HDR rgba16float -> both bridge and presenter update in place without destroy
    processor.setSwapchainToneMapOptions({ hdrMode: 'extended', mode: 'none', applySrgbEncode: null });
    assert.equal(processor.setSwapchainFormat('rgba16float'), true);
    assert.equal(processor.format, 'rgba16float');
    assert.equal(processor.binkPresenter, presenter, 'presenter should update in place, not be destroyed');
    assert.equal(processor.splashFrameBridge, bridge, 'bridge should update in place, not be destroyed');
    assert.equal(presenter.presentationFormat, 'rgba16float');
    assert.equal(bridge._presentationFormat, 'rgba16float');

    // Switch back to SDR bgra8unorm -> both update back to bgra8unorm
    assert.equal(processor.setSwapchainFormat('bgra8unorm'), true);
    assert.equal(processor.format, 'bgra8unorm');
    assert.equal(presenter.presentationFormat, 'bgra8unorm');
    assert.equal(bridge._presentationFormat, 'bgra8unorm');

    // bindDevice with a new format also keeps bridge & presenter synced
    const { device: device2 } = createMockGpuDevice();
    processor.bindDevice(device2, { format: 'rgba8unorm', recoverSwapchain: false });
    assert.equal(processor.format, 'rgba8unorm');
    assert.equal(presenter.presentationFormat, 'rgba8unorm');
    assert.equal(bridge._presentationFormat, 'rgba8unorm');

    processor.destroy();
    swapchain.destroy();
  }

  // 2. Real SplashFrameBridge wrapping a legacy/mock presenter WITHOUT setPresentationFormat():
  //    reconciles cleanly even when splashFrameBridge is non-null (fixes previously dead code path).
  {
    const { device, ctx } = createMockGpuDevice();
    const createdPresenters = [];
    class LegacyMockPresenter {
      constructor(opts) {
        this.presentationFormat = opts.presentationFormat;
        this.isDestroyed = false;
        createdPresenters.push(this);
      }
      uploadRgbaFrame() {}
      uploadFrame() {}
      presentFrame() {}
      destroy() {
        this.isDestroyed = true;
      }
    }

    const processor = new WgpuCommandProcessor(device, ctx, {
      format: 'bgra8unorm',
      BinkVideoPresenter: LegacyMockPresenter,
      SplashFrameBridge,
    });

    const firstBridge = processor._ensureSplashBridge();
    assert.ok(firstBridge);
    assert.equal(createdPresenters.length, 1);
    assert.equal(createdPresenters[0].presentationFormat, 'bgra8unorm');

    assert.equal(processor.setSwapchainFormat('rgba8unorm'), true);
    assert.equal(createdPresenters[0].isDestroyed, true, 'legacy owned presenter should be torn down on format mismatch');
    assert.equal(firstBridge.isDestroyed, true, 'owned bridge wrapping legacy presenter should also tear down');

    const secondBridge = processor._ensureSplashBridge();
    assert.ok(secondBridge && secondBridge !== firstBridge);
    assert.equal(createdPresenters.length, 2);
    assert.equal(createdPresenters[1].presentationFormat, 'rgba8unorm');
    assert.equal(secondBridge._presentationFormat, 'rgba8unorm');

    processor.destroy();
  }
});
