import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import {
  SHADER_STAGE as WGPK_SHADER_STAGE,
  buildWgslPackBuffer,
  fnv1a64,
  formatHashHex,
} from '../web/workers/wgsl_pack_loader.js';

const require = createRequire(import.meta.url);
const wgpuMod = require('../web/workers/wgpu_worker.js');
const swapchainMod = require('../web/workers/wgpu_swapchain.js');
const drawExecutorMod = require('../web/workers/draw_executor.js');

const {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  extractSanitizedGpuAdapter,
  MAX_GPU_ADAPTER_FIELD_LENGTH,
  MAX_DIRECT_WRITE_BUFFER_CHUNK_BYTES,
  OPCODES,
  SHADER_STAGE,
  PRIMITIVE_TOPOLOGY,
  BUFFER_USAGE,
} = wgpuMod;

const { WgpuSwapchain } = swapchainMod;
const { DrawExecutor } = drawExecutorMod;

const PACK_VS_WGSL = [
  '@vertex',
  'fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {',
  '    return vec4<f32>(0.0, 0.0, 0.0, 1.0);',
  '}',
  '',
].join('\n');

const PACK_FS_WGSL = [
  '@fragment',
  'fn fs_main() -> @location(0) vec4<f32> {',
  '    return vec4<f32>(0.2, 0.6, 0.9, 1.0);',
  '}',
  '',
].join('\n');

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

  clearColor(r, g, b, a) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setFloat32(0, r, true);
    view.setFloat32(4, g, true);
    view.setFloat32(8, b, true);
    view.setFloat32(12, a, true);
    this._push(OPCODES.CLEAR_COLOR, new Uint8Array(buf));
    return this;
  }

  createBuffer(id, usage, byteSize) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, usage, true);
    view.setBigUint64(8, BigInt(byteSize), true);
    this._push(OPCODES.CREATE_BUFFER, new Uint8Array(buf));
    return this;
  }

  writeBuffer(id, dstOffset, dataBytes) {
    const bytes = dataBytes instanceof Uint8Array ? dataBytes : new Uint8Array(dataBytes);
    const buf = new ArrayBuffer(16 + bytes.byteLength);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, dstOffset, true);
    view.setUint32(8, bytes.byteLength, true);
    view.setUint32(12, 0, true);
    new Uint8Array(buf, 16).set(bytes);
    this._push(OPCODES.WRITE_BUFFER, new Uint8Array(buf));
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

  setPipeline(pipelineId) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, pipelineId, true);
    this._push(OPCODES.SET_PIPELINE, new Uint8Array(buf));
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

function createMockWebGpu() {
  let nextHandle = 1;
  const writeBufferCalls = [];
  const canvas = {
    width: 1280,
    height: 720,
    getContext() {
      return ctx;
    },
  };
  const ctx = {
    canvas,
    configure() {},
    unconfigure() {},
    getCurrentTexture() {
      const view = { kind: 'GPUTextureView', id: `swap-${nextHandle++}` };
      return {
        width: canvas.width,
        height: canvas.height,
        createView() {
          return view;
        },
      };
    },
  };

  const device = {
    lost: new Promise(() => {}),
    addEventListener() {},
    removeEventListener() {},
    createBuffer(desc) {
      return {
        kind: 'GPUBuffer',
        handle: nextHandle++,
        size: desc.size,
        usage: desc.usage,
        destroy() {},
      };
    },
    createTexture(desc) {
      return {
        kind: 'GPUTexture',
        handle: nextHandle++,
        desc,
        createView() {
          return { kind: 'GPUTextureView', id: `tex-${nextHandle++}` };
        },
        destroy() {},
      };
    },
    createSampler(desc) {
      return { kind: 'GPUSampler', handle: nextHandle++, desc };
    },
    createShaderModule(desc) {
      return { kind: 'GPUShaderModule', handle: nextHandle++, code: desc.code };
    },
    createRenderPipeline(desc) {
      return {
        kind: 'GPURenderPipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayout(idx) {
          return { kind: 'GPUBindGroupLayout', idx };
        },
      };
    },
    createBindGroup(desc) {
      return { kind: 'GPUBindGroup', handle: nextHandle++, desc };
    },
    createCommandEncoder() {
      const cmdBuf = { kind: 'GPUCommandBuffer', handle: nextHandle++ };
      return {
        beginRenderPass() {
          return {
            setPipeline() {},
            setVertexBuffer() {},
            setIndexBuffer() {},
            setBindGroup() {},
            setViewport() {},
            setScissorRect() {},
            draw() {},
            drawIndexed() {},
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
        finish() {
          return cmdBuf;
        },
      };
    },
    queue: {
      writeBuffer(buffer, dstOffset, data) {
        writeBufferCalls.push({
          buffer,
          dstOffset,
          byteLength: data.byteLength,
          data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice(),
        });
      },
      writeTexture() {},
      submit() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
  };

  return { canvas, device, ctx, writeBufferCalls };
}

test('W257 clear-only frames do NOT trigger first-visible-frame, while first non-clear real-pack draw triggers once with blackLoadingMs', async () => {
  const { canvas, device, ctx } = createMockWebGpu();
  let fakeNow = 1000;
  const nowFn = () => fakeNow;

  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context: ctx,
    format: 'bgra8unorm',
    enableDepth: false,
    now: nowFn,
  });
  swapchain.configure({ width: 1280, height: 720, devicePixelRatio: 1 });

  const vsHash = fnv1a64(PACK_VS_WGSL);
  const fsHash = fnv1a64(PACK_FS_WGSL);
  const packBuffer = buildWgslPackBuffer([
    { hash: vsHash, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: PACK_VS_WGSL },
    { hash: fsHash, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: PACK_FS_WGSL },
  ]);

  const processorCallbacks = [];
  const handlerCallbacks = [];
  const posted = [];

  const processor = new WgpuCommandProcessor(device, ctx, {
    now: nowFn,
    swapchain,
    wgslPackBuffer: packBuffer,
    gpuAdapter: {
      vendor: 'NVIDIA',
      architecture: 'Ada',
      device: 'RTX 4090',
      description: 'NVIDIA GeForce RTX 4090',
    },
    onFirstVisibleFrame: (info) => processorCallbacks.push(info),
  });

  const handler = createWgpuWorkerHandler({
    processor,
    postMessage: (msg) => posted.push(msg),
    now: nowFn,
    onFirstVisibleFrame: (info) => handlerCallbacks.push(info),
  });

  // Frame 1 (clear-only): must trigger first-present, but NOT first-visible-frame.
  const clearOnlyBuf = new PacketBuilder()
    .beginFrame()
    .clearColor(0.0, 0.0, 0.0, 1.0)
    .present()
    .finish();
  await handler.handleMessage({ cmdBuffer: clearOnlyBuf });

  assert.equal(processor.frameCount, 1);
  assert.equal(processor.firstVisibleFrame, false);
  assert.equal(processor.firstVisibleFrameMs, null);
  assert.equal(processor.firstPresentMs, 1000);
  assert.equal(processor.blackLoadingMs, null);
  assert.equal(processorCallbacks.length, 0);
  assert.equal(handlerCallbacks.length, 0);
  assert.equal(posted.filter((m) => m.type === 'first-visible-frame').length, 0);
  assert.equal(swapchain.hasVisibleContent, false);
  assert.equal(swapchain.blackPresentsTotal, 1);

  const firstPresentMsgs = posted.filter((m) => m.type === 'first-present');
  assert.equal(firstPresentMsgs.length, 1);
  assert.equal(firstPresentMsgs[0].gpuAdapter.vendor, 'nvidia');

  // Frame 2 (still clear-only at t=1120): still no first-visible-frame.
  fakeNow = 1120;
  await handler.handleMessage({ cmdBuffer: clearOnlyBuf });
  assert.equal(processor.firstVisibleFrame, false);
  assert.equal(posted.filter((m) => m.type === 'first-visible-frame').length, 0);
  assert.equal(swapchain.blackPresentsTotal, 2);

  // Frame 3 (real-pack non-clear draw at t=1450): triggers first-visible-frame ONCE.
  fakeNow = 1450;
  const drawFrameBuf = new PacketBuilder()
    .beginFrame()
    .createShaderModule(1, SHADER_STAGE.VERTEX, formatHashHex(vsHash))
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, formatHashHex(fsHash))
    .createRenderPipeline(10, 1, 2)
    .setPipeline(10)
    .draw(3, 1, 0, 0)
    .present()
    .finish();
  await handler.handleMessage({ cmdBuffer: drawFrameBuf });

  assert.equal(processor.frameCount, 3);
  assert.equal(processor.firstVisibleFrame, true);
  assert.equal(processor.firstPresentMs, 1000);
  assert.equal(processor.firstVisibleFrameMs, 1450);
  assert.equal(processor.blackLoadingMs, 450);
  assert.equal(processorCallbacks.length, 1);
  assert.equal(handlerCallbacks.length, 1);
  assert.equal(swapchain.hasVisibleContent, true);
  assert.equal(swapchain.firstVisiblePresentMs, 1450);

  const visibleMsgs = posted.filter((m) => m.type === 'first-visible-frame');
  assert.equal(visibleMsgs.length, 1);
  assert.equal(visibleMsgs[0].firstVisibleFrame, true);
  assert.equal(visibleMsgs[0].firstPresentMs, 1000);
  assert.equal(visibleMsgs[0].firstVisibleFrameMs, 1450);
  assert.equal(visibleMsgs[0].blackLoadingMs, 450);
  assert.equal(visibleMsgs[0].presentCount, 3);
  assert.equal(visibleMsgs[0].realPackDrawsThisFrame, 1);
  assert.equal(visibleMsgs[0].fallbackDrawsThisFrame, 0);
  assert.equal(visibleMsgs[0].drawsThisFrame, 1);
  assert.equal(visibleMsgs[0].gpuAdapter.vendor, 'nvidia');
  assert.equal(visibleMsgs[0].gpuAdapter.architecture, 'ada');

  // Frame 4 (another draw at t=1466): must NOT fire first-visible-frame again.
  fakeNow = 1466;
  const drawFrame2Buf = new PacketBuilder()
    .beginFrame()
    .setPipeline(10)
    .draw(3, 1, 0, 0)
    .present()
    .finish();
  await handler.handleMessage({ cmdBuffer: drawFrame2Buf });

  assert.equal(processorCallbacks.length, 1);
  assert.equal(handlerCallbacks.length, 1);
  assert.equal(posted.filter((m) => m.type === 'first-visible-frame').length, 1);

  // Verify getStats() and request-gpu-stats expose first-visible-frame + gpuAdapter fields.
  const stats = processor.getStats();
  assert.equal(stats.firstVisibleFrame, true);
  assert.equal(stats.firstVisibleFrameMs, 1450);
  assert.equal(stats.blackLoadingMs, 450);
  assert.equal(stats.gpuAdapter.vendor, 'nvidia');

  await handler.handleMessage({ id: 'stats-w257', type: 'request-gpu-stats' });
  const statsReply = posted.find((m) => m.id === 'stats-w257' && m.type === 'gpu-stats');
  assert.ok(statsReply);
  assert.equal(statsReply.firstVisibleFrame, true);
  assert.equal(statsReply.firstVisibleFrameMs, 1450);
  assert.equal(statsReply.blackLoadingMs, 450);
  assert.equal(statsReply.gpuAdapter.vendor, 'nvidia');
});

test('W257 wires DrawExecutor.beginFrame() + getFrameDrawSummary() and fallback draws trigger first-visible-frame once', () => {
  const { canvas, device, ctx } = createMockWebGpu();
  let fakeNow = 200;
  const nowFn = () => fakeNow;

  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context: ctx,
    format: 'bgra8unorm',
    enableDepth: false,
    now: nowFn,
  });
  swapchain.configure({ width: 1280, height: 720, devicePixelRatio: 1 });

  const drawExecutor = new DrawExecutor({ device });
  let beginFrameCalls = 0;
  const origBeginFrame = drawExecutor.beginFrame.bind(drawExecutor);
  drawExecutor.beginFrame = () => {
    beginFrameCalls += 1;
    return origBeginFrame();
  };

  const visibleEvents = [];
  const processor = new WgpuCommandProcessor(device, ctx, {
    now: nowFn,
    swapchain,
    drawExecutor,
    onFirstVisibleFrame: (info) => visibleEvents.push(info),
  });

  // Simulate DrawExecutor clear-only frame: beginFrame resets counters and clear draw does not trigger visible frame.
  processor.beginFrame(0);
  assert.equal(beginFrameCalls, 1);
  const clearPipe = { kind: 'GPURenderPipeline', isClear: true };
  const mockPass = {
    setPipeline() {},
    setVertexBuffer() {},
    setIndexBuffer() {},
    setBindGroup() {},
    draw() {},
  };
  drawExecutor.setPipeline(clearPipe);
  drawExecutor.draw(mockPass, 3, 1, 0, 0);
  assert.equal(drawExecutor.getFrameDrawSummary().hasNonClearDraw, false);

  processor.executeCommandBuffer(new PacketBuilder().present().finish());
  assert.equal(visibleEvents.length, 0);
  assert.equal(processor.firstVisibleFrame, false);
  assert.equal(swapchain.hasVisibleContent, false);

  // Next frame: BEGIN_FRAME packet calls drawExecutor.beginFrame(), then a fallback draw executes via command buffer.
  fakeNow = 500;
  const fallbackDrawBuf = new PacketBuilder()
    .beginFrame()
    .createShaderModule(1, SHADER_STAGE.VERTEX, '0x1111111111111111')
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, '0x2222222222222222')
    .createRenderPipeline(20, 1, 2)
    .setPipeline(20)
    .draw(3, 1, 0, 0)
    .present()
    .finish();
  processor.executeCommandBuffer(fallbackDrawBuf);

  assert.equal(beginFrameCalls, 2);
  assert.equal(visibleEvents.length, 1);
  assert.equal(visibleEvents[0].firstVisibleFrame, true);
  assert.equal(visibleEvents[0].firstPresentMs, 200);
  assert.equal(visibleEvents[0].firstVisibleFrameMs, 500);
  assert.equal(visibleEvents[0].blackLoadingMs, 300);
  assert.ok(visibleEvents[0].fallbackDrawsThisFrame >= 1);
  assert.equal(visibleEvents[0].drawsThisFrame, 1);
  assert.equal(swapchain.hasVisibleContent, true);
});

test('W257 extractSanitizedGpuAdapter handles adapter.info, async requestAdapterInfo(), rejections, missing info, and sanitizes paths/controls', async () => {
  // 1. Synchronous adapter.info with uppercase, control chars, host paths, and over-long field.
  const longDesc = `NVIDIA GeForce RTX 4090 C:\\Users\\alice\\drivers\\nv.dll /Users/bob/libgpu.so file:///etc/passwd \x00\x07\n\t${'X'.repeat(120)}`;
  const fromProp = await extractSanitizedGpuAdapter({
    isFallbackAdapter: false,
    info: {
      vendor: '  NVIDIA\r\n ',
      architecture: 'Ada-Lovelace',
      device: 'C:\\Windows\\System32\\gpu.sys 0x2684',
      description: longDesc,
    },
  });
  assert.equal(fromProp.vendor, 'nvidia');
  assert.equal(fromProp.architecture, 'ada-lovelace');
  assert.equal(fromProp.device, '0x2684');
  assert.equal(fromProp.isFallbackAdapter, false);
  assert.ok(fromProp.description.length <= MAX_GPU_ADAPTER_FIELD_LENGTH);
  assert.ok(fromProp.label.length <= MAX_GPU_ADAPTER_FIELD_LENGTH);
  assert.equal(fromProp.description.includes('users'), false);
  assert.equal(fromProp.description.includes('alice'), false);
  assert.equal(fromProp.description.includes('bob'), false);
  assert.equal(fromProp.description.includes('passwd'), false);
  assert.equal(/[\x00-\x1F\x7F]/.test(JSON.stringify(fromProp)), false);

  // 2. Async requestAdapterInfo() fallback when adapter.info is missing.
  const fromAsync = await extractSanitizedGpuAdapter({
    isFallbackAdapter: true,
    async requestAdapterInfo() {
      return {
        vendor: 'Google',
        architecture: 'SwiftShader',
        device: 'CPU',
        description: 'SwiftShader Device',
      };
    },
  });
  assert.equal(fromAsync.vendor, 'google');
  assert.equal(fromAsync.architecture, 'swiftshader');
  assert.equal(fromAsync.device, 'cpu');
  assert.equal(fromAsync.description, 'swiftshader device');
  assert.equal(fromAsync.isFallbackAdapter, true);

  // 3. Rejecting requestAdapterInfo() never throws and returns safe defaults.
  const fromRejected = await extractSanitizedGpuAdapter({
    requestAdapterInfo() {
      return Promise.reject(new Error('adapter info blocked'));
    },
  });
  assert.deepEqual(fromRejected, {
    vendor: '',
    architecture: '',
    device: '',
    description: '',
    label: 'unknown',
    isFallbackAdapter: false,
  });

  // 4. Missing/null adapter returns safe defaults.
  const fromNull = await extractSanitizedGpuAdapter(null);
  assert.deepEqual(fromNull, {
    vendor: '',
    architecture: '',
    device: '',
    description: '',
    label: 'unknown',
    isFallbackAdapter: false,
  });
});

test('W257 worker init emits sanitized gpuAdapter on ready, stats, and first-present messages', async () => {
  const { canvas, device } = createMockWebGpu();
  const prevNavigator = globalThis.navigator;
  const posted = [];
  let fakeNow = 500;

  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    writable: true,
    value: {
      gpu: {
        getPreferredCanvasFormat() {
          return 'bgra8unorm';
        },
        async requestAdapter() {
          return {
            features: new Set(),
            limits: {},
            isFallbackAdapter: false,
            async requestAdapterInfo() {
              return {
                vendor: 'AMD',
                architecture: 'RDNA3',
                device: 'RX 7900 XTX',
                description: 'AMD Radeon /home/dev/driver.so',
              };
            },
            async requestDevice() {
              return device;
            },
          };
        },
      },
    },
  });

  try {
    const handler = createWgpuWorkerHandler({
      postMessage: (m) => posted.push(m),
      now: () => fakeNow,
      enableSwapchain: false,
      enableAsyncPipelines: false,
    });
    await handler.handleMessage({ init: true, canvas, width: 1280, height: 720 });

    const readyMsg = posted.find((m) => m && m.ready === true);
    assert.ok(readyMsg, 'worker init must emit ready message');
    assert.ok(readyMsg.gpuAdapter, 'ready message must include sanitized gpuAdapter');
    assert.equal(readyMsg.gpuAdapter.vendor, 'amd');
    assert.equal(readyMsg.gpuAdapter.architecture, 'rdna3');
    assert.equal(readyMsg.gpuAdapter.device, 'rx 7900 xtx');
    assert.equal(readyMsg.gpuAdapter.description.includes('/home/dev'), false);

    await handler.handleMessage({ id: 's1', stats: true });
    const statsMsg = posted.find((m) => m && m.id === 's1');
    assert.ok(statsMsg);
    assert.equal(statsMsg.gpuAdapter.vendor, 'amd');
    assert.equal(statsMsg.stats.gpuAdapter.vendor, 'amd');
  } finally {
    if (prevNavigator === undefined) {
      delete globalThis.navigator;
    } else {
      Object.defineProperty(globalThis, 'navigator', {
        configurable: true,
        writable: true,
        value: prevNavigator,
      });
    }
  }
});

test('W257 _handleWriteBuffer keeps <= 4 MiB writes as 1 call and chunks 10 MiB write into 3 4-byte-aligned calls (#1077)', () => {
  assert.equal(MAX_DIRECT_WRITE_BUFFER_CHUNK_BYTES, 4 * 1024 * 1024);
  assert.equal(MAX_DIRECT_WRITE_BUFFER_CHUNK_BYTES % 4, 0);

  const { device, ctx, writeBufferCalls } = createMockWebGpu();
  const processor = new WgpuCommandProcessor(device, ctx, {
    enableStagingPool: false,
    enableResourceReplay: false,
  });

  const ONE_MIB = 1024 * 1024;
  const FOUR_MIB = 4 * ONE_MIB;
  const TEN_MIB = 10 * ONE_MIB;

  // Create a 16 MiB destination buffer.
  processor.executeCommandBuffer(
    new PacketBuilder()
      .createBuffer(1, BUFFER_USAGE.COPY_DST | BUFFER_USAGE.VERTEX, 16 * ONE_MIB)
      .finish()
  );

  // 1. Exact 4 MiB write -> single queue.writeBuffer call, chunkedWriteBufferUploads stays 0.
  const exact4Mib = new Uint8Array(FOUR_MIB);
  exact4Mib[0] = 0xaa;
  exact4Mib[FOUR_MIB - 1] = 0x55;
  processor.executeCommandBuffer(new PacketBuilder().writeBuffer(1, 0, exact4Mib).finish());

  assert.equal(writeBufferCalls.length, 1);
  assert.equal(writeBufferCalls[0].dstOffset, 0);
  assert.equal(writeBufferCalls[0].byteLength, FOUR_MIB);
  assert.equal(writeBufferCalls[0].data[0], 0xaa);
  assert.equal(writeBufferCalls[0].data[FOUR_MIB - 1], 0x55);
  assert.equal(processor.getStats().chunkedWriteBufferUploads, 0);

  writeBufferCalls.length = 0;

  // 2. Oversized 10 MiB write at 4-byte aligned offset 64 -> 3 chunks: 4 MiB, 4 MiB, 2 MiB.
  const payload10Mib = new Uint8Array(TEN_MIB);
  for (let i = 0; i < payload10Mib.length; i += 4096) {
    payload10Mib[i] = (i >>> 12) & 0xff;
  }
  payload10Mib[0] = 0x11;
  payload10Mib[FOUR_MIB - 1] = 0x22;
  payload10Mib[FOUR_MIB] = 0x33;
  payload10Mib[2 * FOUR_MIB - 1] = 0x44;
  payload10Mib[2 * FOUR_MIB] = 0x55;
  payload10Mib[TEN_MIB - 1] = 0x66;

  processor.executeCommandBuffer(new PacketBuilder().writeBuffer(1, 64, payload10Mib).finish());

  assert.equal(writeBufferCalls.length, 3);
  assert.equal(writeBufferCalls[0].dstOffset, 64);
  assert.equal(writeBufferCalls[0].byteLength, FOUR_MIB);
  assert.equal(writeBufferCalls[0].dstOffset % 4, 0);
  assert.equal(writeBufferCalls[0].byteLength % 4, 0);
  assert.equal(writeBufferCalls[0].data[0], 0x11);
  assert.equal(writeBufferCalls[0].data[FOUR_MIB - 1], 0x22);

  assert.equal(writeBufferCalls[1].dstOffset, 64 + FOUR_MIB);
  assert.equal(writeBufferCalls[1].byteLength, FOUR_MIB);
  assert.equal(writeBufferCalls[1].dstOffset % 4, 0);
  assert.equal(writeBufferCalls[1].byteLength % 4, 0);
  assert.equal(writeBufferCalls[1].data[0], 0x33);
  assert.equal(writeBufferCalls[1].data[FOUR_MIB - 1], 0x44);

  assert.equal(writeBufferCalls[2].dstOffset, 64 + 2 * FOUR_MIB);
  assert.equal(writeBufferCalls[2].byteLength, 2 * ONE_MIB);
  assert.equal(writeBufferCalls[2].dstOffset % 4, 0);
  assert.equal(writeBufferCalls[2].byteLength % 4, 0);
  assert.equal(writeBufferCalls[2].data[0], 0x55);
  assert.equal(writeBufferCalls[2].data[2 * ONE_MIB - 1], 0x66);

  // Verify exact byte-for-byte reconstruction across all 3 chunks.
  let totalBytes = 0;
  for (const call of writeBufferCalls) {
    const start = call.dstOffset - 64;
    for (let i = 0; i < call.byteLength; i += 4096) {
      assert.equal(call.data[i], payload10Mib[start + i]);
    }
    totalBytes += call.byteLength;
  }
  assert.equal(totalBytes, TEN_MIB);
  assert.equal(processor.getStats().chunkedWriteBufferUploads, 1);
});
