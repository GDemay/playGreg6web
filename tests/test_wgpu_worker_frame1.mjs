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
const swapchainMod = require('../web/workers/wgpu_swapchain.js');
const fallbackMod = require('../web/workers/shader_fallback_router.js');
const { default: RuntimeSessionController } = await import('../web/runtime/session_controller.js');

const {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  selectFallbackShaderProfile,
  sanitizeValidationErrorMessage,
  FALLBACK_PROFILE,
  OPCODES,
  SHADER_STAGE,
  PRIMITIVE_TOPOLOGY,
  BUFFER_USAGE,
  TEXTURE_USAGE,
  TEXTURE_FORMAT,
  TEXTURE_DIMENSION,
  ADDRESS_MODE,
  FILTER_MODE,
  BIND_RESOURCE_TYPE,
} = wgpuMod;

const { WgpuSwapchain } = swapchainMod;
const { ShaderFallbackRouter, BUILTIN_FALLBACK_PROFILES } = fallbackMod;

class PacketBuilder {
  constructor() {
    this.chunks = [];
  }

  _push(opcode, payloadBytes) {
    const buf = new ArrayBuffer(8 + payloadBytes.byteLength);
    const view = new DataView(buf);
    view.setUint16(0, opcode, true);
    view.setUint16(2, 0, true);
    view.setUint32(4, payloadBytes.byteLength, true);
    new Uint8Array(buf, 8).set(payloadBytes);
    this.chunks.push(new Uint8Array(buf));
  }

  beginFrame() {
    this._push(OPCODES.BEGIN_FRAME, new Uint8Array(0));
    return this;
  }

  createBuffer(id, usageFlags, byteSize) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, usageFlags, true);
    view.setBigUint64(8, BigInt(byteSize), true);
    this._push(OPCODES.CREATE_BUFFER, new Uint8Array(buf));
    return this;
  }

  createTexture(
    id,
    width,
    height,
    format = TEXTURE_FORMAT.RGBA8_UNORM,
    usage = TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST
  ) {
    const buf = new ArrayBuffer(32);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, width, true);
    view.setUint32(8, height, true);
    view.setUint32(12, 1, true);
    view.setUint32(16, 1, true);
    view.setUint32(20, format, true);
    view.setUint32(24, usage, true);
    view.setUint32(28, TEXTURE_DIMENSION.DIM_2D, true);
    this._push(OPCODES.CREATE_TEXTURE, new Uint8Array(buf));
    return this;
  }

  createSampler(id) {
    const buf = new ArrayBuffer(32);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, ADDRESS_MODE.CLAMP_TO_EDGE, true);
    view.setUint32(8, ADDRESS_MODE.CLAMP_TO_EDGE, true);
    view.setUint32(12, ADDRESS_MODE.CLAMP_TO_EDGE, true);
    view.setUint32(16, FILTER_MODE.LINEAR, true);
    view.setUint32(20, FILTER_MODE.LINEAR, true);
    view.setUint32(24, FILTER_MODE.NEAREST, true);
    view.setUint32(28, 1, true);
    this._push(OPCODES.CREATE_SAMPLER, new Uint8Array(buf));
    return this;
  }

  createShaderModule(id, stage, wgslOrHash) {
    const encoded = new TextEncoder().encode(wgslOrHash);
    const buf = new ArrayBuffer(16 + encoded.byteLength);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, stage, true);
    view.setUint32(8, encoded.byteLength, true);
    view.setUint32(12, 0, true);
    new Uint8Array(buf, 16).set(encoded);
    this._push(OPCODES.CREATE_SHADER_MODULE, new Uint8Array(buf));
    return this;
  }

  createRenderPipeline(id, vsId, fsId, vertexStride = 0, topology = PRIMITIVE_TOPOLOGY.TRIANGLE_LIST) {
    const buf = new ArrayBuffer(24);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, vsId, true);
    view.setUint32(8, fsId, true);
    view.setUint32(12, vertexStride, true);
    view.setUint32(16, topology, true);
    view.setUint32(20, 0, true);
    this._push(OPCODES.CREATE_RENDER_PIPELINE, new Uint8Array(buf));
    return this;
  }

  createBindGroup(id, pipelineId, groupIndex, entries = []) {
    const buf = new ArrayBuffer(16 + entries.length * 32);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, pipelineId, true);
    view.setUint32(8, groupIndex, true);
    view.setUint32(12, entries.length, true);
    for (let i = 0; i < entries.length; i += 1) {
      const e = entries[i];
      const off = 16 + i * 32;
      view.setUint32(off, e.binding, true);
      view.setUint32(off + 4, e.resourceType, true);
      view.setUint32(off + 8, e.resourceId, true);
      view.setUint32(off + 12, 0, true);
      view.setBigUint64(off + 16, BigInt(e.offset || 0), true);
      view.setBigUint64(off + 24, BigInt(e.size || 0), true);
    }
    this._push(OPCODES.CREATE_BIND_GROUP, new Uint8Array(buf));
    return this;
  }

  setPipeline(pipelineId) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, pipelineId, true);
    this._push(OPCODES.SET_PIPELINE, new Uint8Array(buf));
    return this;
  }

  setVertexBuffer(slot, bufferId, offset = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, slot, true);
    view.setUint32(4, bufferId, true);
    view.setBigUint64(8, BigInt(offset), true);
    this._push(OPCODES.SET_VERTEX_BUFFER, new Uint8Array(buf));
    return this;
  }

  setBindGroup(groupIndex, bindGroupId) {
    const buf = new ArrayBuffer(8);
    const view = new DataView(buf);
    view.setUint32(0, groupIndex, true);
    view.setUint32(4, bindGroupId, true);
    this._push(OPCODES.SET_BIND_GROUP, new Uint8Array(buf));
    return this;
  }

  draw(vertexCount, instanceCount = 1, firstVertex = 0, firstInstance = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, vertexCount, true);
    view.setUint32(4, instanceCount, true);
    view.setUint32(8, firstVertex, true);
    view.setUint32(12, firstInstance, true);
    this._push(OPCODES.DRAW, new Uint8Array(buf));
    return this;
  }

  present() {
    this._push(OPCODES.PRESENT, new Uint8Array(0));
    return this;
  }

  finish() {
    const total = this.chunks.reduce((sum, c) => sum + c.byteLength, 0);
    const out = new Uint8Array(total);
    let offset = 0;
    for (const chunk of this.chunks) {
      out.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return out.buffer;
  }
}

function createMockWebGpuEnv(options = {}) {
  const { strictAutoLayoutForMissingGroup0 = false } = options;
  let nextHandle = 1;
  const calls = {
    createBuffer: [],
    createTexture: [],
    createSampler: [],
    createShaderModule: [],
    createRenderPipeline: [],
    createBindGroup: [],
    passOps: [],
    submit: 0,
    configure: [],
  };

  const listeners = new Map();
  const addEventListenerCalls = [];
  const removeEventListenerCalls = [];

  const device = {
    features: new Set(),
    limits: { maxTextureDimension2D: 8192 },
    lost: new Promise(() => {}),
    addEventListener(type, fn) {
      addEventListenerCalls.push({ type, fn });
      if (!listeners.has(type)) {
        listeners.set(type, new Set());
      }
      listeners.get(type).add(fn);
    },
    removeEventListener(type, fn) {
      removeEventListenerCalls.push({ type, fn });
      if (listeners.has(type)) {
        listeners.get(type).delete(fn);
      }
    },
    dispatchUncapturedError(errorOrEvent) {
      const event =
        errorOrEvent && typeof errorOrEvent === 'object' && 'error' in errorOrEvent
          ? errorOrEvent
          : { error: errorOrEvent };
      const set = listeners.get('uncapturederror');
      if (!set) return 0;
      let invoked = 0;
      for (const fn of Array.from(set)) {
        fn(event);
        invoked += 1;
      }
      return invoked;
    },
    getUncapturedErrorListenerCount() {
      const set = listeners.get('uncapturederror');
      return set ? set.size : 0;
    },
    createBuffer(desc) {
      const buf = {
        kind: 'GPUBuffer',
        handle: nextHandle++,
        size: desc.size,
        usage: desc.usage,
        destroyed: false,
        destroy() {
          this.destroyed = true;
        },
      };
      calls.createBuffer.push({ desc, result: buf });
      return buf;
    },
    createTexture(desc) {
      const handle = nextHandle++;
      const tex = {
        kind: 'GPUTexture',
        handle,
        desc,
        destroyed: false,
        createView() {
          return { kind: 'GPUTextureView', textureHandle: handle, handle: nextHandle++ };
        },
        destroy() {
          this.destroyed = true;
        },
      };
      calls.createTexture.push({ desc, result: tex });
      return tex;
    },
    createSampler(desc) {
      const sampler = { kind: 'GPUSampler', handle: nextHandle++, desc };
      calls.createSampler.push({ desc, result: sampler });
      return sampler;
    },
    createShaderModule(desc) {
      const mod = { kind: 'GPUShaderModule', handle: nextHandle++, code: desc.code };
      calls.createShaderModule.push({ desc, result: mod });
      return mod;
    },
    createRenderPipeline(desc) {
      const vsCode = desc && desc.vertex && desc.vertex.module ? desc.vertex.module.code || '' : '';
      const fsCode =
        desc && desc.fragment && desc.fragment.module ? desc.fragment.module.code || '' : '';
      const hasGroup0 = /@group\(0\)/.test(vsCode) || /@group\(0\)/.test(fsCode);
      const pipe = {
        kind: 'GPURenderPipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayout(index) {
          if (strictAutoLayoutForMissingGroup0 && index === 0 && !hasGroup0) {
            throw new Error('GPURenderPipeline auto layout has no @group(0)');
          }
          return { kind: 'GPUBindGroupLayout', pipelineHandle: this.handle, index };
        },
      };
      calls.createRenderPipeline.push({ desc, result: pipe });
      return pipe;
    },
    createBindGroup(desc) {
      const bg = { kind: 'GPUBindGroup', handle: nextHandle++, desc };
      calls.createBindGroup.push({ desc, result: bg });
      return bg;
    },
    createCommandEncoder() {
      return {
        beginRenderPass() {
          return {
            setPipeline(pipeline) {
              calls.passOps.push({ op: 'setPipeline', pipeline });
            },
            setVertexBuffer(slot, buffer, offset) {
              calls.passOps.push({ op: 'setVertexBuffer', slot, buffer, offset });
            },
            setIndexBuffer(buffer, format, offset) {
              calls.passOps.push({ op: 'setIndexBuffer', buffer, format, offset });
            },
            setBindGroup(index, bindGroup) {
              calls.passOps.push({ op: 'setBindGroup', index, bindGroup });
            },
            draw(vertexCount, instanceCount, firstVertex, firstInstance) {
              calls.passOps.push({
                op: 'draw',
                vertexCount,
                instanceCount,
                firstVertex,
                firstInstance,
              });
            },
            drawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance) {
              calls.passOps.push({
                op: 'drawIndexed',
                indexCount,
                instanceCount,
                firstIndex,
                baseVertex,
                firstInstance,
              });
            },
            end() {
              calls.passOps.push({ op: 'end' });
            },
          };
        },
        finish() {
          return { kind: 'GPUCommandBuffer', handle: nextHandle++ };
        },
      };
    },
    queue: {
      writeBuffer() {},
      writeTexture() {},
      submit() {
        calls.submit += 1;
      },
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
  };

  const ctx = {
    configure(opts) {
      calls.configure.push(opts);
    },
    unconfigure() {},
    getCurrentTexture() {
      return {
        createView() {
          return { kind: 'GPUTextureView', id: `swap-${nextHandle++}` };
        },
      };
    },
  };

  const canvas = {
    width: 1280,
    height: 720,
    getContext(kind) {
      assert.equal(kind, 'webgpu');
      return ctx;
    },
  };

  const adapter = {
    info: { vendor: 'mock-gpu', architecture: 'g44-frame1' },
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
    calls,
    addEventListenerCalls,
    removeEventListenerCalls,
  };
}

// ---------------------------------------------------------------------------
// 1. Live WebGPU validation errors via WgpuSwapchain -> gpu-validation-error + SessionController
// ---------------------------------------------------------------------------

test('G44 defaultInitGpu wires WgpuSwapchain onValidationError -> sanitized gpu-validation-error, stats counter, and W72 SessionController compatibility', async () => {
  const mock = createMockWebGpuEnv();
  const posted = [];
  let fakeNow = 5000;

  const prevNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: mock.navigator,
  });

  let handler;
  try {
    handler = createWgpuWorkerHandler({
      postMessage: (m) => posted.push(m),
      now: () => fakeNow,
      WgpuSwapchain,
    });

    await handler.handleMessage({
      init: true,
      canvas: mock.canvas,
      width: 1280,
      height: 720,
    });
  } finally {
    if (prevNavigator) {
      Object.defineProperty(globalThis, 'navigator', prevNavigator);
    } else {
      delete globalThis.navigator;
    }
  }

  assert.ok(handler.processor, 'processor must be initialized');
  assert.ok(handler.swapchain instanceof WgpuSwapchain, 'WgpuSwapchain must be created during defaultInitGpu');
  assert.equal(
    mock.device.getUncapturedErrorListenerCount(),
    1,
    'exactly one uncapturederror listener should be registered (no double-hooking)'
  );
  assert.equal(handler.processor.getStats().validationErrorCount, 0);

  // Dispatch a validation error containing Windows/Unix paths & a long hex dump.
  fakeNow = 5123;
  mock.device.dispatchUncapturedError({
    error: {
      name: 'GPUValidationError',
      message:
        'Pipeline layout mismatch at C:\\Users\\secret\\game\\shaders.bin and /Users/alice/private/dump.wgsl bytes: ' +
        'deadbeefcafebabe0123456789abcdefdeadbeefcafebabe0123456789abcdef',
    },
  });

  // Dispatch an OOM error.
  fakeNow = 5456;
  mock.device.dispatchUncapturedError({
    error: {
      name: 'GPUOutOfMemoryError',
      message: 'Out of VRAM allocating 1024MB texture from file:///C:/secret/game.rpf',
    },
  });

  const validationMsgs = posted.filter((m) => m && m.type === 'gpu-validation-error');
  assert.equal(validationMsgs.length, 2, 'must post one gpu-validation-error per uncapturederror');

  assert.equal(validationMsgs[0].totalCount, 1);
  assert.equal(validationMsgs[0].error.seq, 1);
  assert.equal(validationMsgs[0].error.type, 'validation');
  assert.equal(validationMsgs[0].error.timestampMs, 5123);
  assert.equal(validationMsgs[0].error.message.includes('secret'), false);
  assert.equal(validationMsgs[0].error.message.includes('alice'), false);
  assert.equal(validationMsgs[0].error.message.includes('deadbeefcafebabe'), false);
  assert.match(validationMsgs[0].error.message, /<path>/);
  assert.match(validationMsgs[0].error.message, /<buffer>/);

  assert.equal(validationMsgs[1].totalCount, 2);
  assert.equal(validationMsgs[1].error.seq, 2);
  assert.equal(validationMsgs[1].error.type, 'out-of-memory');
  assert.equal(validationMsgs[1].error.timestampMs, 5456);
  assert.equal(validationMsgs[1].error.message.includes('game.rpf'), false);
  assert.match(validationMsgs[1].error.message, /<path>/);

  // Processor stats & history match.
  const stats = handler.processor.getStats();
  assert.equal(stats.validationErrorCount, 2);
  assert.equal(stats.validationErrorHistorySize, 2);
  assert.equal(handler.processor.getValidationErrors().length, 2);

  // End-to-end check: feed emitted gpu-validation-error messages directly into W72 RuntimeSessionController.
  const controller = new RuntimeSessionController({ now: () => fakeNow });
  const r1 = controller.handleWorkerMessage(validationMsgs[0]);
  const r2 = controller.handleWorkerMessage(validationMsgs[1]);
  assert.equal(r1.handled, true);
  assert.equal(r1.type, 'gpu-validation-error');
  assert.equal(r1.validationErrorCount, 1);
  assert.equal(r2.handled, true);
  assert.equal(r2.validationErrorCount, 2);
  assert.equal(controller.getSnapshot().validationErrorCount, 2);

  // Destroy cleans up listener cleanly; subsequent events do not post.
  await handler.handleMessage({ id: 'd1', destroy: true });
  assert.equal(mock.device.getUncapturedErrorListenerCount(), 0);
  mock.device.dispatchUncapturedError({
    error: { name: 'GPUValidationError', message: 'should be ignored after destroy' },
  });
  assert.equal(posted.filter((m) => m && m.type === 'gpu-validation-error').length, 2);
});

// ---------------------------------------------------------------------------
// 2. Fallback device.addEventListener('uncapturederror') when swapchain is disabled/absent
// ---------------------------------------------------------------------------

test('G44 fallback device.addEventListener(uncapturederror) captures, sanitizes, counts, and posts when swapchain is not used', async () => {
  const mock = createMockWebGpuEnv();
  const posted = [];
  let fakeNow = 1000;

  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    now: () => fakeNow,
    maxValidationErrorHistory: 2,
  });
  assert.equal(processor.swapchain, null);
  assert.equal(mock.device.getUncapturedErrorListenerCount(), 0);

  const handler = createWgpuWorkerHandler({
    processor,
    postMessage: (m) => posted.push(m),
    now: () => fakeNow,
  });
  assert.equal(mock.device.getUncapturedErrorListenerCount(), 1);

  // Fire 3 errors to verify bounded history (2) vs monotonic totalCount (3).
  fakeNow = 1010;
  mock.device.dispatchUncapturedError({
    error: { type: 'validation', message: 'Error 1 at /home/user/secret/file.wgsl' },
  });
  fakeNow = 1020;
  mock.device.dispatchUncapturedError({
    error: { type: 'internal', message: 'Error 2 ' + 'mismatch '.repeat(80) },
  });
  fakeNow = 1030;
  mock.device.dispatchUncapturedError({
    error: { type: 'out-of-memory', message: 'Error 3 data:application/octet-stream;base64,' + 'A'.repeat(80) },
  });

  const msgs = posted.filter((m) => m && m.type === 'gpu-validation-error');
  assert.equal(msgs.length, 3);
  assert.deepEqual(
    msgs.map((m) => [m.error.seq, m.error.type, m.totalCount]),
    [
      [1, 'validation', 1],
      [2, 'internal', 2],
      [3, 'out-of-memory', 3],
    ]
  );
  assert.equal(msgs[0].error.message.includes('secret'), false);
  assert.ok(msgs[1].error.message.length <= 512);
  assert.ok(msgs[1].error.message.endsWith('...'));
  assert.match(msgs[2].error.message, /<buffer>/);

  // History bounded to 2 (seq 2 and 3 retained), totalCount stays 3.
  const stats = processor.getStats();
  assert.equal(stats.validationErrorCount, 3);
  assert.equal(stats.validationErrorHistorySize, 2);
  assert.deepEqual(
    processor.getValidationErrors().map((e) => e.seq),
    [2, 3]
  );

  // Worker stats message also reports validationErrorCount.
  await handler.handleMessage({ id: 'stats-1', stats: true });
  const statsReply = posted.find((m) => m && m.id === 'stats-1');
  assert.equal(statsReply.stats.validationErrorCount, 3);

  // bindDevice switches listener cleanly to a new GPUDevice.
  const mock2 = createMockWebGpuEnv();
  processor.bindDevice(mock2.device);
  assert.equal(mock.device.getUncapturedErrorListenerCount(), 0);
  assert.equal(mock2.device.getUncapturedErrorListenerCount(), 1);

  fakeNow = 2000;
  mock.device.dispatchUncapturedError({ error: { type: 'validation', message: 'stale device' } });
  assert.equal(processor.getStats().validationErrorCount, 3);

  mock2.device.dispatchUncapturedError({ error: { type: 'validation', message: 'new device ok' } });
  assert.equal(processor.getStats().validationErrorCount, 4);
  assert.equal(posted.filter((m) => m && m.type === 'gpu-validation-error').length, 4);

  processor.destroy();
  assert.equal(mock2.device.getUncapturedErrorListenerCount(), 0);
});

test('G44 externally injected WgpuSwapchain preserves its existing onValidationError callback and avoids double-counting', () => {
  const mock = createMockWebGpuEnv();
  const userCaptured = [];
  const procCaptured = [];

  const swapchain = new WgpuSwapchain({
    canvas: mock.canvas,
    device: mock.device,
    context: mock.ctx,
    format: 'bgra8unorm',
    enableDepth: false,
    onValidationError: (rec) => userCaptured.push(rec),
  });
  swapchain.configure({ width: 1280, height: 720 });

  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    swapchain,
    onValidationError: (rec, total) => procCaptured.push({ rec, total }),
  });

  assert.equal(mock.device.getUncapturedErrorListenerCount(), 1);

  mock.device.dispatchUncapturedError({
    error: { name: 'GPUValidationError', message: 'Bind group slot 0 unbound' },
  });

  assert.equal(userCaptured.length, 1);
  assert.equal(procCaptured.length, 1);
  assert.equal(procCaptured[0].total, 1);
  assert.equal(processor.getStats().validationErrorCount, 1);

  // Destroying processor unhooks wrapper without destroying the externally-owned swapchain.
  processor.destroy();
  assert.equal(swapchain.isDestroyed, false);

  mock.device.dispatchUncapturedError({
    error: { name: 'GPUValidationError', message: 'After processor destroy' },
  });
  assert.equal(userCaptured.length, 2, 'external swapchain callback must remain intact');
  assert.equal(procCaptured.length, 1, 'destroyed processor must not receive further events');
  swapchain.destroy();
});

// ---------------------------------------------------------------------------
// 3. Frame-1 fallback shader profile selection (G34 / G38)
// ---------------------------------------------------------------------------

test('G44 selectFallbackShaderProfile accurately detects fullscreen_triangle_cbv, textured_quad_24b, and default patterns', () => {
  assert.equal(
    selectFallbackShaderProfile({ vertexCount: 3, hasBoundCbv: true, hasBoundTexture: false }),
    FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV
  );
  assert.equal(
    selectFallbackShaderProfile({
      vertexCount: 3,
      bindGroups: [{ hasCbv: true, hasTexture: false }],
    }),
    FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV
  );
  assert.equal(
    selectFallbackShaderProfile({
      vertexCount: 6,
      vertexStride: 24,
      bindGroups: [{ hasTexture: true, hasSampler: true }],
    }),
    FALLBACK_PROFILE.TEXTURED_QUAD_24B
  );
  // vertexCount === 3 with BOTH texture and 24B stride selects textured_quad_24b (not fullscreen_triangle_cbv).
  assert.equal(
    selectFallbackShaderProfile({
      vertexCount: 3,
      vertexStride: 24,
      hasBoundCbv: true,
      hasBoundTexture: true,
    }),
    FALLBACK_PROFILE.TEXTURED_QUAD_24B
  );
  // Missing CBV on 3-vertex draw stays default.
  assert.equal(
    selectFallbackShaderProfile({ vertexCount: 3, hasBoundCbv: false, hasBoundTexture: false }),
    FALLBACK_PROFILE.DEFAULT
  );
  // 24B stride without texture stays default.
  assert.equal(
    selectFallbackShaderProfile({ vertexCount: 6, vertexStride: 24, hasBoundTexture: false }),
    FALLBACK_PROFILE.DEFAULT
  );
  assert.equal(selectFallbackShaderProfile(null), FALLBACK_PROFILE.DEFAULT);
});

test('G44 single-buffer Frame-1 DrawInstanced(3,1) + CBV routes missing shaders to fullscreen_triangle_cbv profile', () => {
  const mock = createMockWebGpuEnv({ strictAutoLayoutForMissingGroup0: true });
  const resolveCalls = [];
  const realRouter = new ShaderFallbackRouter();
  const spyRouter = {
    resolve(hash, stage, opts) {
      resolveCalls.push({ hash, stage, opts, argCount: arguments.length });
      return realRouter.resolve(hash, stage, opts);
    },
    getStats: () => realRouter.getStats(),
    getMissingReport: (o) => realRouter.getMissingReport(o),
  };

  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    shaderFallbackRouter: spyRouter,
  });

  const stream = new PacketBuilder()
    .beginFrame()
    .createBuffer(1, BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST, 256)
    .createShaderModule(2, SHADER_STAGE.VERTEX, '0xfa1e000000000001')
    .createShaderModule(3, SHADER_STAGE.FRAGMENT, '0xfa1e000000000002')
    .createRenderPipeline(4, 2, 3, 0)
    .createBindGroup(5, 4, 0, [
      { binding: 0, resourceType: BIND_RESOURCE_TYPE.BUFFER, resourceId: 1, offset: 0, size: 256 },
    ])
    .setPipeline(4)
    .setBindGroup(0, 5)
    .draw(3, 1, 0, 0)
    .present()
    .finish();

  const res = processor.executeCommandBuffer(stream);
  assert.deepEqual(res, { frameCount: 1, presents: 1 });

  // Verify { profile: 'fullscreen_triangle_cbv' } was passed to router.resolve for both VS and FS.
  assert.equal(resolveCalls.length, 2);
  assert.deepEqual(resolveCalls[0].opts, { profile: 'fullscreen_triangle_cbv' });
  assert.deepEqual(resolveCalls[1].opts, { profile: 'fullscreen_triangle_cbv' });

  // Verify actual compiled WGSL matches G34 FULLSCREEN_TRIANGLE_CBV template.
  assert.equal(
    processor.shaderModules.get(2).code,
    BUILTIN_FALLBACK_PROFILES.fullscreen_triangle_cbv.vertex
  );
  assert.equal(
    processor.shaderModules.get(3).code,
    BUILTIN_FALLBACK_PROFILES.fullscreen_triangle_cbv.fragment
  );
  assert.equal(processor.shaderModules.get(2).fallbackProfile, 'fullscreen_triangle_cbv');
  assert.equal(processor.shaderModules.get(3).fallbackProfile, 'fullscreen_triangle_cbv');

  // Strict auto layout succeeded for group(0) CBV bind group!
  assert.equal(mock.calls.createBindGroup.length, 1);
  assert.equal(mock.calls.createRenderPipeline.length, 1);
  assert.deepEqual(mock.calls.createRenderPipeline[0].desc.vertex.buffers, []);

  const stats = processor.getStats();
  assert.equal(stats.fallbackProfileCounts.fullscreen_triangle_cbv, 2);
  assert.equal(stats.lastFallbackProfile, 'fullscreen_triangle_cbv');
});

test('G44 single-buffer 24B-stride VB + texture routes missing shaders to textured_quad_24b profile with 24B vertex layout', () => {
  const mock = createMockWebGpuEnv({ strictAutoLayoutForMissingGroup0: true });
  const resolveCalls = [];
  const realRouter = new ShaderFallbackRouter();
  const spyRouter = {
    resolve(hash, stage, opts) {
      resolveCalls.push({ hash, stage, opts });
      return realRouter.resolve(hash, stage, opts);
    },
    getStats: () => realRouter.getStats(),
    getMissingReport: (o) => realRouter.getMissingReport(o),
  };

  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    shaderFallbackRouter: spyRouter,
  });

  const stream = new PacketBuilder()
    .beginFrame()
    .createBuffer(1, BUFFER_USAGE.VERTEX | BUFFER_USAGE.COPY_DST, 24 * 6)
    .createTexture(2, 64, 64)
    .createSampler(3)
    .createShaderModule(4, SHADER_STAGE.VERTEX, '0x0bad000000000001')
    .createShaderModule(5, SHADER_STAGE.FRAGMENT, '0x0bad000000000002')
    .createRenderPipeline(6, 4, 5, 24)
    .createBindGroup(7, 6, 0, [
      { binding: 0, resourceType: BIND_RESOURCE_TYPE.TEXTURE_VIEW, resourceId: 2 },
      { binding: 1, resourceType: BIND_RESOURCE_TYPE.SAMPLER, resourceId: 3 },
    ])
    .setPipeline(6)
    .setVertexBuffer(0, 1, 0)
    .setBindGroup(0, 7)
    .draw(6, 1, 0, 0)
    .present()
    .finish();

  const res = processor.executeCommandBuffer(stream);
  assert.deepEqual(res, { frameCount: 1, presents: 1 });

  assert.equal(resolveCalls.length, 2);
  assert.deepEqual(resolveCalls[0].opts, { profile: 'textured_quad_24b' });
  assert.deepEqual(resolveCalls[1].opts, { profile: 'textured_quad_24b' });

  assert.equal(
    processor.shaderModules.get(4).code,
    BUILTIN_FALLBACK_PROFILES.textured_quad_24b.vertex
  );
  assert.equal(
    processor.shaderModules.get(5).code,
    BUILTIN_FALLBACK_PROFILES.textured_quad_24b.fragment
  );

  // Verify 24B 2-attribute vertex buffer layout (vec4<f32> pos @ 0 + vec2<f32> uv @ 16).
  assert.equal(mock.calls.createRenderPipeline.length, 1);
  assert.deepEqual(mock.calls.createRenderPipeline[0].desc.vertex.buffers, [
    {
      arrayStride: 24,
      stepMode: 'vertex',
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x4' },
        { shaderLocation: 1, offset: 16, format: 'float32x2' },
      ],
    },
  ]);
  assert.equal(processor.getStats().fallbackProfileCounts.textured_quad_24b, 2);
});

test('G44 cross-buffer Frame-1 draw specializes default fallback pipeline on draw and caches it across frames', () => {
  const mock = createMockWebGpuEnv({ strictAutoLayoutForMissingGroup0: true });
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx);

  // Batch 1: shaders + render pipeline created before any draw pattern is known -> 'default'
  const setupBatch = new PacketBuilder()
    .createBuffer(1, BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST, 256)
    .createShaderModule(2, SHADER_STAGE.VERTEX, '0xcafe000000000001')
    .createShaderModule(3, SHADER_STAGE.FRAGMENT, '0xcafe000000000002')
    .createRenderPipeline(4, 2, 3, 0)
    .finish();
  processor.executeCommandBuffer(setupBatch);

  assert.equal(processor.shaderModules.get(2).fallbackProfile, 'default');
  assert.equal(mock.calls.createRenderPipeline.length, 1);

  // Batch 2: Frame 1 creates CBV bind group (initially no @group(0) on default auto layout) and issues DRAW(3, 1)
  const frame1Batch = new PacketBuilder()
    .beginFrame()
    .createBindGroup(5, 4, 0, [
      { binding: 0, resourceType: BIND_RESOURCE_TYPE.BUFFER, resourceId: 1, offset: 0, size: 256 },
    ])
    .setPipeline(4)
    .setBindGroup(0, 5)
    .draw(3, 1, 0, 0)
    .present()
    .finish();
  processor.executeCommandBuffer(frame1Batch);

  // Specialized fullscreen_triangle_cbv pipeline and matching bind group were built and bound!
  assert.equal(mock.calls.createRenderPipeline.length, 2);
  const specPipe = mock.calls.createRenderPipeline[1].result;
  assert.equal(
    specPipe.desc.vertex.module.code,
    BUILTIN_FALLBACK_PROFILES.fullscreen_triangle_cbv.vertex
  );
  assert.equal(
    specPipe.desc.fragment.module.code,
    BUILTIN_FALLBACK_PROFILES.fullscreen_triangle_cbv.fragment
  );
  assert.equal(mock.calls.createBindGroup.length, 1);

  const drawIdx = mock.calls.passOps.findIndex((op) => op.op === 'draw');
  assert.ok(drawIdx >= 2);
  assert.equal(mock.calls.passOps[drawIdx - 2].op, 'setPipeline');
  assert.equal(mock.calls.passOps[drawIdx - 2].pipeline, specPipe);
  assert.equal(mock.calls.passOps[drawIdx - 1].op, 'setBindGroup');

  // Batch 3: Frame 2 reuses cached specialized pipeline & bind group (no extra compilations).
  const frame2Batch = new PacketBuilder()
    .beginFrame()
    .setPipeline(4)
    .setBindGroup(0, 5)
    .draw(3, 1, 0, 0)
    .present()
    .finish();
  processor.executeCommandBuffer(frame2Batch);

  assert.equal(mock.calls.createRenderPipeline.length, 2, 'specialized pipeline must be cached');
  assert.equal(mock.calls.createBindGroup.length, 1, 'specialized bind group must be cached');
  assert.equal(processor.frameCount, 2);
});

// ---------------------------------------------------------------------------
// 4. Legacy 2-arg ShaderFallbackRouter compatibility
// ---------------------------------------------------------------------------

test('G44 remains 100% compatible with legacy 2-arg ShaderFallbackRouter implementations (including strict routers)', () => {
  const legacyVs = '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }';
  const legacyFs = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }';

  // Case A: strict 2-arg router that throws if a 3rd options object is passed.
  const mockA = createMockWebGpuEnv();
  const strictCalls = [];
  const strictLegacyRouter = {
    resolve(hash, stage) {
      strictCalls.push(arguments.length);
      if (arguments.length > 2) {
        throw new Error('Legacy 2-arg router does not accept options');
      }
      return {
        wgsl: stage === 'vertex' ? legacyVs : legacyFs,
        source: 'builtin',
        requestedHash: hash,
        resolvedHash: null,
        stage,
        entryPoint: stage === 'vertex' ? 'vs_main' : 'fs_main',
      };
    },
  };

  const procA = new WgpuCommandProcessor(mockA.device, mockA.ctx, {
    shaderFallbackRouter: strictLegacyRouter,
  });

  const stream = new PacketBuilder()
    .beginFrame()
    .createBuffer(1, BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST, 256)
    .createShaderModule(2, SHADER_STAGE.VERTEX, '0x1e6a000000000001')
    .createShaderModule(3, SHADER_STAGE.FRAGMENT, '0x1e6a000000000002')
    .createRenderPipeline(4, 2, 3, 0)
    .createBindGroup(5, 4, 0, [
      { binding: 0, resourceType: BIND_RESOURCE_TYPE.BUFFER, resourceId: 1, offset: 0, size: 256 },
    ])
    .setPipeline(4)
    .setBindGroup(0, 5)
    .draw(3, 1, 0, 0)
    .present()
    .finish();

  assert.doesNotThrow(() => procA.executeCommandBuffer(stream));
  assert.equal(procA.frameCount, 1);
  assert.equal(procA.shaderModules.get(2).code, legacyVs);
  assert.equal(procA.shaderModules.get(3).code, legacyFs);
  // First tried 3 args, caught, and cleanly retried with 2 args for each shader.
  assert.ok(strictCalls.includes(3));
  assert.ok(strictCalls.includes(2));

  // Case B: bare-string returning 2-arg router + explicit resolveFallbackShader helper.
  const mockB = createMockWebGpuEnv();
  const stringRouter = {
    resolve(_hash, stage) {
      return stage === 'vertex' ? legacyVs : legacyFs;
    },
  };
  const procB = new WgpuCommandProcessor(mockB.device, mockB.ctx, {
    shaderFallbackRouter: stringRouter,
  });
  const res = procB.resolveFallbackShader('0x1234', SHADER_STAGE.VERTEX, {
    vertexCount: 3,
    hasBoundCbv: true,
    hasBoundTexture: false,
  });
  assert.equal(res.wgsl, legacyVs);
  assert.equal(res.source, 'builtin');
});

// ---------------------------------------------------------------------------
// 5. Classic Web Worker importScripts lexical isolation (G31) + live Frame-1 & error posting
// ---------------------------------------------------------------------------

test('G44 classic WorkerGlobalScope loads wgpu_swapchain.js & shader_fallback_router.js via importScripts with zero collisions and runs live Frame-1 + gpu-validation-error', async () => {
  const mock = createMockWebGpuEnv({ strictAutoLayoutForMissingGroup0: true });
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
      width: 1280,
      height: 720,
      enableShaderFallback: true,
    },
  });

  assert.ok(
    importedScripts.includes('./wgpu_swapchain.js'),
    `expected ./wgpu_swapchain.js via importScripts; got ${JSON.stringify(importedScripts)}`
  );
  assert.ok(
    importedScripts.includes('./shader_fallback_router.js'),
    `expected ./shader_fallback_router.js via importScripts; got ${JSON.stringify(importedScripts)}`
  );
  assert.ok(postedMessages.some((m) => m && m.ready === true));

  // Re-importing helper scripts a second time in the same WorkerGlobalScope must throw zero SyntaxErrors.
  assert.doesNotThrow(() => {
    vmCtx.importScripts('./wgpu_swapchain.js', './shader_fallback_router.js');
  });

  // Fire a live uncaptured WebGPU validation error inside the classic worker realm.
  mock.device.dispatchUncapturedError({
    error: {
      name: 'GPUValidationError',
      message: 'Classic worker validation error at C:\\Users\\dev\\secret.wgsl',
    },
  });

  const valMsgs = postedMessages.filter((m) => m && m.type === 'gpu-validation-error');
  assert.equal(valMsgs.length, 1);
  assert.equal(valMsgs[0].totalCount, 1);
  assert.equal(valMsgs[0].error.seq, 1);
  assert.equal(valMsgs[0].error.type, 'validation');
  assert.equal(valMsgs[0].error.message.includes('dev'), false);
  assert.match(valMsgs[0].error.message, /<path>/);

  // Execute a Frame-1 DrawInstanced(3, 1) + CBV packet buffer inside the VM realm.
  const hostBuf = new Uint8Array(
    new PacketBuilder()
      .beginFrame()
      .createBuffer(1, BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST, 256)
      .createShaderModule(2, SHADER_STAGE.VERTEX, '0xab12000000000001')
      .createShaderModule(3, SHADER_STAGE.FRAGMENT, '0xab12000000000002')
      .createRenderPipeline(4, 2, 3, 0)
      .createBindGroup(5, 4, 0, [
        { binding: 0, resourceType: BIND_RESOURCE_TYPE.BUFFER, resourceId: 1, offset: 0, size: 256 },
      ])
      .setPipeline(4)
      .setBindGroup(0, 5)
      .draw(3, 1, 0, 0)
      .present()
      .finish()
  );
  const vmBytes = vm.runInContext(`new Uint8Array(${hostBuf.byteLength})`, vmCtx);
  vmBytes.set(hostBuf);

  await vmCtx.self.onmessage({ data: { cmdBuffer: vmBytes } });

  assert.equal(mock.calls.createShaderModule.length, 2);
  assert.equal(
    mock.calls.createShaderModule[0].desc.code,
    BUILTIN_FALLBACK_PROFILES.fullscreen_triangle_cbv.vertex
  );
  assert.equal(
    mock.calls.createShaderModule[1].desc.code,
    BUILTIN_FALLBACK_PROFILES.fullscreen_triangle_cbv.fragment
  );

  // Check exported sanitization helper works cleanly too.
  assert.equal(sanitizeValidationErrorMessage('ok'), 'ok');
});
