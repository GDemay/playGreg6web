import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import {
  SHADER_STAGE as WGPK_SHADER_STAGE,
  buildWgslPackBuffer,
  fnv1a64,
  formatHashHex,
} from '../web/workers/wgsl_pack_loader.js';
import { bootstrapPlayGta6App } from '../web/app_bootstrap.js';

const require = createRequire(import.meta.url);
const wgpuMod = require('../web/workers/wgpu_worker.js');
const swapchainMod = require('../web/workers/wgpu_swapchain.js');
const { SplashOverlayController, DISMISS_REASONS } = require('../web/ui/splash_overlay.js');
const { BootOrchestrator } = require('../web/runtime/boot_orchestrator.js');

const {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  buildGpuTelemetrySnapshot,
  SWAPCHAIN_PROBE_DEFAULT_LUMA_THRESHOLD,
  SWAPCHAIN_PROBE_MAX_CONSECUTIVE_ERRORS,
  SWAPCHAIN_PROBE_STRIPS,
  SWAPCHAIN_PROBE_BYTES_PER_ROW,
  SWAPCHAIN_PROBE_BUFFER_SIZE,
  OPCODES,
  SHADER_STAGE,
  PRIMITIVE_TOPOLOGY,
  BUFFER_USAGE,
  TEXTURE_USAGE,
} = wgpuMod;

const { WgpuSwapchain, TEXTURE_USAGE_COPY_SRC, TEXTURE_USAGE_RENDER_ATTACHMENT } = swapchainMod;

class BinaryCommandBuilder {
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

function createPackBuffer() {
  const vsWgsl = '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }';
  const fsWgsl = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.2, 0.3, 0.4, 1.0); }';
  const vsHash = fnv1a64(vsWgsl);
  const fsHash = fnv1a64(fsWgsl);
  const packBuffer = buildWgslPackBuffer([
    { hash: vsHash, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: vsWgsl },
    { hash: fsHash, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: fsWgsl },
  ]);
  return { vsHash, fsHash, packBuffer };
}

function buildDrawStream(vsHash, fsHash, { includeSetup = true } = {}) {
  const b = new BinaryCommandBuilder().beginFrame();
  if (includeSetup) {
    b.createShaderModule(1, SHADER_STAGE.VERTEX, formatHashHex(vsHash))
      .createShaderModule(2, SHADER_STAGE.FRAGMENT, formatHashHex(fsHash))
      .createRenderPipeline(3, 1, 2, 0);
  }
  return b.setPipeline(3).draw(3, 1, 0, 0).present().finish();
}

async function flushMicrotasks(turns = 6) {
  for (let i = 0; i < turns; i += 1) {
    await new Promise((resolve) => setImmediate(resolve));
  }
}

function createProbeEnv({
  width = 1280,
  height = 720,
  format = 'bgra8unorm',
  textureUsage = TEXTURE_USAGE.RENDER_ATTACHMENT | TEXTURE_USAGE.COPY_SRC,
  rejectMapAsync = null,
  throwCopyTextureToBuffer = null,
} = {}) {
  let nextHandle = 1;
  const pixels = new Uint8Array(width * height * 4);
  for (let i = 3; i < pixels.length; i += 4) {
    pixels[i] = 255;
  }

  const copyCalls = [];
  const probeBuffers = [];
  const configured = [];
  const encoderOps = [];
  let activeConfiguredUsage = textureUsage;
  let activeCanvasFormat = format;

  const setPixel = (x, y, c0, c1, c2, a = 255) => {
    const off = (y * width + x) * 4;
    pixels[off] = c0;
    pixels[off + 1] = c1;
    pixels[off + 2] = c2;
    pixels[off + 3] = a;
  };

  const clearPixels = (c0 = 0, c1 = 0, c2 = 0, a = 255) => {
    for (let i = 0; i < pixels.length; i += 4) {
      pixels[i] = c0;
      pixels[i + 1] = c1;
      pixels[i + 2] = c2;
      pixels[i + 3] = a;
    }
  };

  const canvas = {
    width,
    height,
    getContext(kind) {
      assert.equal(kind, 'webgpu');
      return ctx;
    },
  };

  const ctx = {
    canvas,
    configured,
    configure(desc) {
      configured.push({ ...desc });
      activeConfiguredUsage =
        typeof desc.usage === 'number' ? desc.usage : TEXTURE_USAGE.RENDER_ATTACHMENT;
      if (typeof desc.format === 'string' && desc.format.length > 0) {
        activeCanvasFormat = desc.format;
      }
    },
    unconfigure() {},
    getCurrentTexture() {
      const tex = {
        kind: 'GPUTexture',
        label: 'canvas-swapchain-texture',
        id: nextHandle++,
        width: canvas.width || width,
        height: canvas.height || height,
        format: activeCanvasFormat,
        usage: activeConfiguredUsage,
        createView() {
          return { kind: 'GPUTextureView', id: `canvas-view-${tex.id}`, texture: tex };
        },
      };
      return tex;
    },
  };

  const device = {
    lost: new Promise(() => {}),
    addEventListener() {},
    removeEventListener() {},
    createBuffer(desc) {
      const backing = new Uint8Array(desc.size);
      let mapped = false;
      const buf = {
        kind: 'GPUBuffer',
        handle: nextHandle++,
        label: desc.label || '',
        size: desc.size,
        usage: desc.usage,
        destroyed: false,
        _backing: backing,
        mapCalls: 0,
        mapAsync() {
          buf.mapCalls += 1;
          if (buf.destroyed) {
            return Promise.reject(new Error('cannot map destroyed buffer'));
          }
          if (typeof rejectMapAsync === 'function') {
            const rejection = rejectMapAsync(buf.mapCalls);
            if (rejection) {
              return Promise.reject(
                rejection instanceof Error ? rejection : new Error(String(rejection))
              );
            }
          }
          return Promise.resolve().then(() => {
            if (buf.destroyed) {
              throw new Error('buffer destroyed while mapping');
            }
            mapped = true;
          });
        },
        getMappedRange(offset = 0, size = backing.byteLength - offset) {
          if (!mapped || buf.destroyed) {
            throw new Error('buffer not mapped');
          }
          return backing.buffer.slice(offset, offset + size);
        },
        unmap() {
          mapped = false;
        },
        destroy() {
          mapped = false;
          buf.destroyed = true;
        },
      };
      if (desc && desc.label === 'playgta6-swapchain-luma-probe') {
        probeBuffers.push(buf);
      }
      return buf;
    },
    createTexture(desc) {
      const tex = {
        kind: 'GPUTexture',
        handle: nextHandle++,
        label: desc.label || '',
        width: desc.size && desc.size.width ? desc.size.width : width,
        height: desc.size && desc.size.height ? desc.size.height : height,
        format: desc.format,
        usage: desc.usage,
        destroyed: false,
        createView() {
          return { kind: 'GPUTextureView', id: `tex-view-${tex.handle}`, texture: tex };
        },
        destroy() {
          tex.destroyed = true;
        },
      };
      return tex;
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
        beginRenderPass(desc) {
          encoderOps.push({ op: 'beginRenderPass', label: (desc && desc.label) || '' });
          return {
            setPipeline() {},
            setVertexBuffer() {},
            setIndexBuffer() {},
            setBindGroup() {},
            setViewport() {},
            setScissorRect() {},
            draw() {},
            drawIndexed() {},
            end() {
              encoderOps.push({ op: 'endRenderPass', label: (desc && desc.label) || '' });
            },
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
        copyTextureToBuffer(source, destination, copySize) {
          if (typeof throwCopyTextureToBuffer === 'function') {
            const err = throwCopyTextureToBuffer(copyCalls.length);
            if (err) {
              throw err instanceof Error ? err : new Error(String(err));
            }
          }
          encoderOps.push({
            op: 'copyTextureToBuffer',
            textureLabel: source && source.texture ? source.texture.label : '',
          });
          copyCalls.push({
            source: {
              ...source,
              origin: source.origin ? { ...source.origin } : { x: 0, y: 0, z: 0 },
            },
            destination: { ...destination },
            copySize: { ...copySize },
          });
          const dstBacking = destination.buffer && destination.buffer._backing;
          if (dstBacking) {
            const ox = (source.origin && source.origin.x) || 0;
            const oy = (source.origin && source.origin.y) || 0;
            const cw = copySize.width || 0;
            const ch = copySize.height || 1;
            const dstOff = destination.offset || 0;
            const bpr = destination.bytesPerRow || cw * 4;
            for (let row = 0; row < ch; row += 1) {
              const srcStart = ((oy + row) * width + ox) * 4;
              const dstStart = dstOff + row * bpr;
              dstBacking.set(pixels.subarray(srcStart, srcStart + cw * 4), dstStart);
            }
          }
        },
        finish() {
          return cmdBuf;
        },
      };
    },
    queue: {
      writeBuffer() {},
      writeTexture() {},
      submit() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
  };

  return {
    canvas,
    ctx,
    device,
    pixels,
    setPixel,
    clearPixels,
    copyCalls,
    probeBuffers,
    configured,
    encoderOps,
  };
}

test('W350 upgrades swapchain probe to 8-strip grid (2048 bytes) with threshold 8 while holding pure black', async () => {
  assert.equal(SWAPCHAIN_PROBE_STRIPS.length, 8);
  assert.equal(SWAPCHAIN_PROBE_BYTES_PER_ROW, 256);
  assert.equal(SWAPCHAIN_PROBE_BUFFER_SIZE, 2048);
  assert.equal(SWAPCHAIN_PROBE_DEFAULT_LUMA_THRESHOLD, 8);
  assert.equal(SWAPCHAIN_PROBE_MAX_CONSECUTIVE_ERRORS, 3);
  assert.deepEqual(
    SWAPCHAIN_PROBE_STRIPS.map((s) => s.yFraction),
    [0.15, 0.35, 0.5, 0.65, 0.8, 0.89, 0.92, 0.95]
  );

  const { vsHash, fsHash, packBuffer } = createPackBuffer();

  // 1. Pure black (0,0,0,255) and near-black (8,8,8,255) MUST stay held; dark slate (15,18,24) triggers.
  {
    const env = createProbeEnv({ width: 1280, height: 720 });
    let fakeNow = 1000;
    const processor = new WgpuCommandProcessor(env.device, env.ctx, {
      format: 'bgra8unorm',
      width: 1280,
      height: 720,
      now: () => fakeNow,
      wgslPackBuffer: packBuffer,
    });

    processor.execute(buildDrawStream(vsHash, fsHash, { includeSetup: true }));
    await flushMicrotasks();
    assert.equal(processor.firstVisibleFrame, false);
    assert.equal(processor.swapchainProbeCount, 1);
    assert.equal(processor.swapchainProbeSupported, true);
    assert.equal(processor.swapchainProbeErrorCount, 0);
    assert.equal(processor.suppressedBlackFrames, 1);
    assert.equal(processor.lastSampledMaxLuma, 0);
    assert.equal(env.probeBuffers[0].size, 2048);

    // Near-black <= 8 stays suppressed.
    fakeNow = 1300;
    env.clearPixels(8, 8, 8, 255);
    processor.execute(buildDrawStream(vsHash, fsHash, { includeSetup: false }));
    await flushMicrotasks();
    assert.equal(processor.firstVisibleFrame, false);
    assert.equal(processor.suppressedBlackFrames, 2);
    assert.equal(processor.lastSampledMaxLuma, 8);

    // Dark slate Start Screen background rgb(15, 18, 24) (> 8) triggers promptly!
    fakeNow = 1600;
    env.clearPixels(24, 18, 15, 255);
    processor.execute(buildDrawStream(vsHash, fsHash, { includeSetup: false }));
    await flushMicrotasks();
    assert.equal(processor.firstVisibleFrame, true);
    assert.equal(processor.lastSampledMaxLuma, 24);
    assert.equal(env.probeBuffers[0].destroyed, true);
    processor.destroy();
  }

  // 2. Verify every new sparse strip region (y=0.35 left, y=0.65 right, y=0.80 center, y=0.89 & y=0.95 bottom-right) detects UI!
  const sparseTargets = [
    { label: 'upper-left menu y=0.35', yFraction: 0.35, xFraction: 0.25 },
    { label: 'lower-right menu y=0.65', yFraction: 0.65, xFraction: 0.75 },
    { label: 'lower-center prompt y=0.80', yFraction: 0.8, xFraction: 0.5 },
    { label: 'bottom-right bar top y=0.89', yFraction: 0.89, xFraction: 0.9 },
    { label: 'bottom-right bar bottom y=0.95', yFraction: 0.95, xFraction: 0.9 },
  ];

  for (const target of sparseTargets) {
    const env = createProbeEnv({ width: 1280, height: 720 });
    const processor = new WgpuCommandProcessor(env.device, env.ctx, {
      format: 'bgra8unorm',
      width: 1280,
      height: 720,
      now: () => 2000,
      wgslPackBuffer: packBuffer,
    });
    const originX = Math.floor((1280 - 64) * target.xFraction);
    const originY = Math.floor(720 * target.yFraction);
    // Low-luma anti-aliased UI edge (luma=9, strictly > 8)
    env.setPixel(originX + 10, originY, 9, 0, 0, 255);
    processor.execute(buildDrawStream(vsHash, fsHash, { includeSetup: true }));
    await flushMicrotasks();
    assert.equal(processor.firstVisibleFrame, true, `${target.label} must fire first-visible-frame`);
    assert.equal(processor.lastSampledMaxLuma, 9, `${target.label} max luma`);
    processor.destroy();
  }
});

test('W350 falls back cleanly after 3 consecutive mapAsync rejections so splash never hangs permanently', async () => {
  const { vsHash, fsHash, packBuffer } = createPackBuffer();
  const warnings = [];
  const env = createProbeEnv({
    width: 1280,
    height: 720,
    rejectMapAsync: () => new Error('OperationError: mapAsync failed at C:\\Users\\dev\\secret\\gpu.dll'),
  });
  let fakeNow = 1000;
  const posted = [];
  const swapchain = new WgpuSwapchain({
    canvas: env.canvas,
    device: env.device,
    context: env.ctx,
    format: 'bgra8unorm',
    enableDepth: false,
    preserveCopySrcUntilVisible: true,
    now: () => fakeNow,
  });
  swapchain.configure({ width: 1280, height: 720 });

  const processor = new WgpuCommandProcessor(env.device, env.ctx, {
    format: 'bgra8unorm',
    width: 1280,
    height: 720,
    swapchain,
    now: () => fakeNow,
    wgslPackBuffer: packBuffer,
    logger: {
      warn(msg) {
        warnings.push(String(msg));
      },
      info() {},
      error() {},
    },
  });
  const handler = createWgpuWorkerHandler({
    processor,
    postMessage: (m) => posted.push(m),
    now: () => fakeNow,
  });

  // Error #1 & #2: probe stays supported, splash stays held, no fallback warning yet.
  await handler.handleMessage({ cmdBuffer: buildDrawStream(vsHash, fsHash, { includeSetup: true }) });
  await flushMicrotasks();
  assert.equal(processor.swapchainProbeErrorCount, 1);
  assert.equal(processor.swapchainProbeConsecutiveErrors, 1);
  assert.equal(processor.swapchainProbeSupported, true);
  assert.equal(processor.firstVisibleFrame, false);
  assert.equal(warnings.length, 0);

  fakeNow = 1300;
  await handler.handleMessage({ cmdBuffer: buildDrawStream(vsHash, fsHash, { includeSetup: false }) });
  await flushMicrotasks();
  assert.equal(processor.swapchainProbeErrorCount, 2);
  assert.equal(processor.swapchainProbeConsecutiveErrors, 2);
  assert.equal(processor.swapchainProbeSupported, true);
  assert.equal(processor.firstVisibleFrame, false);
  assert.equal(warnings.length, 0);

  // Error #3: logs sanitized warning ONCE, disables probe, destroys buffer, and fires fallback first-visible-frame!
  fakeNow = 1600;
  await handler.handleMessage({ cmdBuffer: buildDrawStream(vsHash, fsHash, { includeSetup: false }) });
  await flushMicrotasks();
  assert.equal(processor.swapchainProbeErrorCount, 3);
  assert.equal(processor.swapchainProbeSupported, false);
  assert.equal(processor.swapchainProbeInFlight, false);
  assert.equal(env.probeBuffers[0].destroyed, true);
  assert.equal(processor.firstVisibleFrame, true);
  assert.equal(swapchain.hasVisibleContent, true);
  assert.equal(warnings.length, 1);
  assert.equal(warnings[0].includes('C:\\Users'), false, 'warning must sanitize host paths');

  const visMsgs = posted.filter((m) => m.type === 'first-visible-frame');
  assert.equal(visMsgs.length, 1);
  assert.equal(visMsgs[0].source, 'swapchain-probe-fallback');
  assert.equal(visMsgs[0].swapchainProbeErrorCount, 3);

  // Verify gpu-telemetry surfaces probe failure & fallback state cleanly.
  const snap = buildGpuTelemetrySnapshot(processor, { timestamp: 1650 });
  assert.equal(snap.firstVisibleFrame, true);
  assert.equal(snap.swapchainProbeCount, 3);
  assert.equal(snap.swapchainProbeInFlight, false);
  assert.equal(snap.swapchainProbeSupported, false);
  assert.equal(snap.swapchainProbeErrorCount, 3);
  assert.equal(snap.lastSampledMaxLuma, 0);
  assert.equal(snap.suppressedBlackFrames, 0);

  processor.destroy();
  swapchain.destroy();
});

test('W350 falls back cleanly after 3 consecutive copyTextureToBuffer throws and resets error streak on success', async () => {
  const { vsHash, fsHash, packBuffer } = createPackBuffer();

  // 1. Transient errors (< 3) followed by a successful black readback reset consecutive error streak.
  {
    let failCount = 0;
    const env = createProbeEnv({
      width: 1280,
      height: 720,
      rejectMapAsync: () => {
        if (failCount < 2) {
          failCount += 1;
          return new Error('transient mapAsync failure');
        }
        return null;
      },
    });
    let fakeNow = 1000;
    const processor = new WgpuCommandProcessor(env.device, env.ctx, {
      format: 'bgra8unorm',
      width: 1280,
      height: 720,
      now: () => fakeNow,
      wgslPackBuffer: packBuffer,
    });

    processor.execute(buildDrawStream(vsHash, fsHash, { includeSetup: true }));
    await flushMicrotasks();
    fakeNow = 1300;
    processor.execute(buildDrawStream(vsHash, fsHash, { includeSetup: false }));
    await flushMicrotasks();
    assert.equal(processor.swapchainProbeConsecutiveErrors, 2);
    assert.equal(processor.swapchainProbeSupported, true);

    // 3rd probe succeeds as black -> resets consecutive error streak to 0 while keeping splash held!
    fakeNow = 1600;
    processor.execute(buildDrawStream(vsHash, fsHash, { includeSetup: false }));
    await flushMicrotasks();
    assert.equal(processor.swapchainProbeConsecutiveErrors, 0);
    assert.equal(processor.swapchainProbeErrorCount, 2);
    assert.equal(processor.swapchainProbeSupported, true);
    assert.equal(processor.suppressedBlackFrames, 1);
    assert.equal(processor.firstVisibleFrame, false);
    processor.destroy();
  }

  // 2. 3 consecutive copyTextureToBuffer throws disable probe and allow frame #3 to emit first-visible-frame.
  {
    const warnings = [];
    const env = createProbeEnv({
      width: 1280,
      height: 720,
      throwCopyTextureToBuffer: () => new Error('copyTextureToBuffer rejected on OffscreenCanvas'),
    });
    let fakeNow = 1000;
    const processor = new WgpuCommandProcessor(env.device, env.ctx, {
      format: 'bgra8unorm',
      width: 1280,
      height: 720,
      now: () => fakeNow,
      wgslPackBuffer: packBuffer,
      logger: {
        warn(msg) {
          warnings.push(String(msg));
        },
        info() {},
        error() {},
      },
    });

    processor.execute(buildDrawStream(vsHash, fsHash, { includeSetup: true }));
    assert.equal(processor.swapchainProbeErrorCount, 1);
    assert.equal(processor.firstVisibleFrame, false);

    fakeNow = 1300;
    processor.execute(buildDrawStream(vsHash, fsHash, { includeSetup: false }));
    assert.equal(processor.swapchainProbeErrorCount, 2);
    assert.equal(processor.firstVisibleFrame, false);

    fakeNow = 1600;
    processor.execute(buildDrawStream(vsHash, fsHash, { includeSetup: false }));
    assert.equal(processor.swapchainProbeErrorCount, 3);
    assert.equal(processor.swapchainProbeSupported, false);
    assert.equal(env.probeBuffers[0].destroyed, true);
    assert.equal(processor.firstVisibleFrame, true);
    assert.equal(warnings.length, 1);
    processor.destroy();
  }
});

test('W350 preserves COPY_SRC across WgpuSwapchain reconfigures until visible and probes presented canvasTexture after blit fallback', async () => {
  const { vsHash, fsHash, packBuffer } = createPackBuffer();
  const env = createProbeEnv({ width: 1280, height: 720, format: 'bgra8unorm' });
  let fakeNow = 1000;

  const swapchain = new WgpuSwapchain({
    canvas: env.canvas,
    device: env.device,
    context: env.ctx,
    format: 'bgra8unorm',
    enableDepth: false,
    preserveCopySrcUntilVisible: true,
    now: () => fakeNow,
  });

  // Even without explicit usage, preserveCopySrcUntilVisible adds COPY_SRC and keeps it across resize/setFormat/blit-fallback.
  swapchain.configure({ width: 1280, height: 720 });
  assert.equal(swapchain.hasCopySrcUsage, true);
  assert.equal(
    swapchain.usage,
    TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_COPY_SRC
  );

  swapchain.resize(1920, 1080, 1);
  assert.equal(swapchain.hasCopySrcUsage, true);
  swapchain.resize(1280, 720, 1);
  assert.equal(swapchain.hasCopySrcUsage, true);

  // Switch to a non-canvas stream format (rgb10a2unorm) -> activates blit fallback while keeping canvas COPY_SRC!
  swapchain.configureForStreamFormat('rgb10a2unorm');
  assert.equal(swapchain.usesBlitFallback, true);
  assert.equal(swapchain.hasCopySrcUsage, true);
  assert.equal(swapchain.canvasFormat, 'bgra8unorm');

  const processor = new WgpuCommandProcessor(env.device, env.ctx, {
    format: 'bgra8unorm',
    width: 1280,
    height: 720,
    swapchain,
    now: () => fakeNow,
    wgslPackBuffer: packBuffer,
  });

  // Frame 1 (black): blit pass runs BEFORE copyTextureToBuffer, and probe copies from canvasTexture (not intermediate).
  processor.execute(buildDrawStream(vsHash, fsHash, { includeSetup: true }));
  await flushMicrotasks();
  assert.equal(swapchain.blitCount, 1);
  assert.equal(processor.firstVisibleFrame, false);
  assert.equal(processor.suppressedBlackFrames, 1);

  const blitPassIdx = env.encoderOps.findIndex(
    (op) => op.op === 'endRenderPass' && op.label === 'playgta6-swapchain-blit-pass'
  );
  const firstCopyIdx = env.encoderOps.findIndex((op) => op.op === 'copyTextureToBuffer');
  assert.ok(blitPassIdx >= 0, 'blit render pass must have executed');
  assert.ok(firstCopyIdx > blitPassIdx, 'swapchain probe must encode copyTextureToBuffer AFTER blitToCanvasIfNeeded');
  assert.equal(
    env.encoderOps[firstCopyIdx].textureLabel,
    'canvas-swapchain-texture',
    'probe must sample the actual presented canvasTexture'
  );

  // Frame 2 (non-black on presented canvasTexture): fires first-visible-frame!
  fakeNow = 1350;
  env.setPixel(640, Math.floor(720 * 0.5), 0, 160, 0, 255);
  processor.execute(buildDrawStream(vsHash, fsHash, { includeSetup: false }));
  await flushMicrotasks();
  assert.equal(processor.firstVisibleFrame, true);
  assert.equal(swapchain.hasVisibleContent, true);
  assert.equal(processor.lastSampledMaxLuma, 160);

  processor.destroy();
  swapchain.destroy();
});

test('W350 preserves swapchain probe telemetry fields in app_bootstrap boot_state.gpu and getDiagnosticsSnapshot()', async () => {
  function makeMockNode(tag = 'div') {
    const attrs = new Map();
    const styles = new Map();
    const node = {
      tagName: String(tag).toUpperCase(),
      className: '',
      hidden: false,
      textContent: '',
      innerHTML: '',
      parentNode: null,
      children: [],
      style: {
        setProperty(k, v) {
          styles.set(k, String(v));
        },
        getPropertyValue(k) {
          return styles.get(k) || '';
        },
      },
      classList: {
        add(c) {
          const set = new Set(String(node.className || '').split(/\s+/).filter(Boolean));
          set.add(c);
          node.className = [...set].join(' ');
        },
        remove(c) {
          const set = new Set(String(node.className || '').split(/\s+/).filter(Boolean));
          set.delete(c);
          node.className = [...set].join(' ');
        },
        contains(c) {
          return String(node.className || '').split(/\s+/).includes(c);
        },
      },
      setAttribute(k, v) {
        attrs.set(k, String(v));
      },
      getAttribute(k) {
        return attrs.has(k) ? attrs.get(k) : null;
      },
      removeAttribute(k) {
        attrs.delete(k);
      },
      appendChild(child) {
        child.parentNode = node;
        node.children.push(child);
        return child;
      },
      removeChild(child) {
        const idx = node.children.indexOf(child);
        if (idx >= 0) node.children.splice(idx, 1);
        child.parentNode = null;
        return child;
      },
      addEventListener() {},
      removeEventListener() {},
      querySelector() {
        return null;
      },
    };
    return node;
  }

  let nowMs = 1000;
  const clockNow = () => nowMs;
  const listeners = new Map();
  const mockWorker = {
    addEventListener(type, fn) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add(fn);
    },
    removeEventListener(type, fn) {
      if (listeners.has(type)) listeners.get(type).delete(fn);
    },
    dispatch(type, ev) {
      for (const fn of listeners.get(type) || []) fn(ev);
    },
    postMessage() {},
    terminate() {},
  };

  const broadcastInstances = [];
  class MockBroadcastChannel {
    constructor(name) {
      this.name = name;
      this.onmessage = null;
      broadcastInstances.push(this);
    }
    close() {}
  }

  const windowObj = {
    location: { origin: 'http://127.0.0.1:8080', search: '' },
    BroadcastChannel: MockBroadcastChannel,
    addEventListener() {},
    removeEventListener() {},
    requestAnimationFrame() {
      return 1;
    },
    cancelAnimationFrame() {},
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    crossOriginIsolated: true,
    devicePixelRatio: 1,
  };
  const documentObj = {
    head: makeMockNode('head'),
    body: makeMockNode('body'),
    createElement: (tag) => makeMockNode(tag),
    querySelector: () => null,
    addEventListener() {},
    removeEventListener() {},
  };
  const navigatorObj = {
    gpu: {
      getPreferredCanvasFormat() {
        return 'bgra8unorm';
      },
      async requestAdapter() {
        return {
          info: { vendor: 'nvidia', architecture: 'turing' },
          async requestDevice() {
            return { lost: new Promise(() => {}), addEventListener() {}, removeEventListener() {} };
          },
        };
      },
    },
  };

  const fetchImpl = async (url) => {
    if (String(url).includes('status')) {
      return {
        ok: true,
        status: 200,
        headers: new Map([
          ['cross-origin-opener-policy', 'same-origin'],
          ['cross-origin-embedder-policy', 'require-corp'],
        ]),
        async json() {
          return {
            ok: true,
            coopCoep: true,
            hasWasmBinary: true,
            wasmUrl: '/boot/guest.wasm',
            pgimUrl: '/boot/guest.pgim',
          };
        },
      };
    }
    return {
      ok: true,
      status: 200,
      headers: new Map(),
      async arrayBuffer() {
        return new ArrayBuffer(4);
      },
      async json() {
        return {};
      },
    };
  };

  const canvas = {
    width: 1280,
    height: 720,
    getBoundingClientRect: () => ({ width: 1280, height: 720 }),
    transferControlToOffscreen: () => ({ width: 1280, height: 720 }),
    addEventListener() {},
    removeEventListener() {},
  };

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl,
    canvas,
    splashContainer: makeMockNode('div'),
    SplashOverlayCtor: SplashOverlayController,
    BootOrchestratorCtor: BootOrchestrator,
    pgimPreloader: {
      async preloadBootBundle() {
        return { pgim: new Uint8Array(4), bytesLoaded: 4 };
      },
    },
    splashOverlayOptions: { fadeMs: 0, visibleFrameFadeMs: 0, nowFn: clockNow },
    defaultHoldUntilVisible: true,
    now: clockNow,
    setInterval: () => 1,
    clearInterval: () => {},
    createWasmWorker: () => mockWorker,
  });

  try {
    const started = await app.start();
    assert.equal(started.ok, true);

    nowMs = 1250;
    broadcastInstances[0].onmessage({ data: { label: 'WebGPU frame #1 presented' } });
    assert.equal(app.splashOverlay.getState().holdingForVisibleFrame, true);

    // Worker sends gpu-telemetry while probing black frames.
    nowMs = 1750;
    mockWorker.dispatch('message', {
      data: {
        type: 'gpu-telemetry',
        firstVisibleFrame: false,
        firstVisibleFrameMs: null,
        blackLoadingMs: null,
        swapchainProbeCount: 7,
        swapchainProbeInFlight: true,
        swapchainProbeSupported: true,
        swapchainProbeErrorCount: 1,
        lastSampledMaxLuma: 6,
        suppressedBlackFrames: 5,
        frameStats: {
          framesPresented: 30,
          droppedFrames: 0,
          drawCalls: 90,
          realPackDrawCount: 90,
        },
      },
    });

    const blackBootState = app.getBootTelemetry();
    const blackDiag = app.getDiagnosticsSnapshot();
    assert.equal(blackBootState.gpu.hasVisibleContent, false);
    assert.equal(blackBootState.gpu.swapchainProbeCount, 7);
    assert.equal(blackBootState.gpu.swapchainProbeInFlight, true);
    assert.equal(blackBootState.gpu.swapchainProbeSupported, true);
    assert.equal(blackBootState.gpu.swapchainProbeErrorCount, 1);
    assert.equal(blackBootState.gpu.lastSampledMaxLuma, 6);
    assert.equal(blackBootState.gpu.suppressedBlackFrames, 5);

    assert.equal(blackDiag.swapchainProbeCount, 7);
    assert.equal(blackDiag.swapchainProbeInFlight, true);
    assert.equal(blackDiag.swapchainProbeSupported, true);
    assert.equal(blackDiag.swapchainProbeErrorCount, 1);
    assert.equal(blackDiag.lastSampledMaxLuma, 6);
    assert.equal(blackDiag.suppressedBlackFrames, 5);
    assert.equal(app.splashOverlay.getState().dismissed, false);

    // Explicit first-visible-frame dismisses splash and updates luma/suppressed counters.
    nowMs = 2250;
    mockWorker.dispatch('message', {
      data: {
        type: 'first-visible-frame',
        firstVisibleFrame: true,
        firstVisibleFrameMs: 2250,
        firstPresentMs: 1250,
        blackLoadingMs: 1000,
        presentCount: 45,
        maxLuma: 210,
        suppressedBlackFrames: 6,
        source: 'swapchain-probe',
      },
    });

    const visibleBootState = app.getBootTelemetry();
    assert.equal(visibleBootState.gpu.hasVisibleContent, true);
    assert.equal(visibleBootState.gpu.lastSampledMaxLuma, 210);
    assert.equal(visibleBootState.gpu.suppressedBlackFrames, 6);
    assert.equal(app.splashOverlay.getState().dismissed, true);
    assert.equal(app.splashOverlay.getState().dismissReason, DISMISS_REASONS.FIRST_VISIBLE_FRAME);
  } finally {
    await app.destroy();
  }
});

test('Zero swapchain probe GPU stalls (no copyTextureToBuffer, mapAsync, or ensureCopySrcUsage) once swapchain.hasVisibleContent is latched', async () => {
  const { vsHash, fsHash, packBuffer } = createPackBuffer();
  const env = createProbeEnv({ width: 1280, height: 720, format: 'bgra8unorm' });
  let fakeNow = 500;
  let ensureCopySrcCalls = 0;

  const swapchain = new WgpuSwapchain({
    canvas: env.canvas,
    device: env.device,
    context: env.ctx,
    format: 'bgra8unorm',
    enableDepth: false,
    preserveCopySrcUntilVisible: true,
    now: () => fakeNow,
  });
  swapchain.configure({ width: 1280, height: 720 });

  const origEnsureCopySrc = swapchain.ensureCopySrcUsage.bind(swapchain);
  swapchain.ensureCopySrcUsage = () => {
    ensureCopySrcCalls += 1;
    return origEnsureCopySrc();
  };

  // Pre-latch visible content on swapchain (e.g. from Bink2 intro frame or prior visible frame).
  swapchain.confirmVisibleFrame({ timestampMs: 500, realPackDraws: 1 });
  assert.equal(swapchain.hasVisibleContent, true);

  const processor = new WgpuCommandProcessor(env.device, env.ctx, {
    format: 'bgra8unorm',
    width: 1280,
    height: 720,
    swapchain,
    wgslPackBuffer: packBuffer,
    now: () => fakeNow,
    swapchainProbeIntervalMs: 250,
    enableResourceReplay: false,
    enableRtPool: false,
  });

  for (let i = 0; i < 6; i += 1) {
    fakeNow += 300;
    processor.execute(buildDrawStream(vsHash, fsHash, { includeSetup: i === 0 }));
    await flushMicrotasks();
  }

  assert.equal(env.copyCalls.length, 0, 'must emit 0 copyTextureToBuffer calls when already visible');
  assert.equal(env.probeBuffers.length, 0, 'must allocate 0 probe buffers when already visible');
  assert.equal(ensureCopySrcCalls, 0, 'must not call ensureCopySrcUsage when already visible');
  assert.equal(processor.swapchainProbeCount, 0);
  assert.equal(processor._swapchainProbeBuffer, null);

  processor.destroy();
  swapchain.destroy();
});

test('defaultInitGpu keeps WgpuSwapchain preserveCopySrcUntilVisible droppable instead of locking COPY_SRC permanently', async () => {
  const env = createProbeEnv({ width: 1280, height: 720, format: 'bgra8unorm' });
  const posted = [];
  const prevNavDesc = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  const fakeGpu = {
    getPreferredCanvasFormat: () => 'bgra8unorm',
    async requestAdapter() {
      return {
        info: { vendor: 'amd', architecture: 'rdna2', device: 'test', description: 'test' },
        features: new Set(),
        limits: {},
        async requestDevice() {
          return env.device;
        },
      };
    },
  };
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    writable: true,
    value: { ...(globalThis.navigator || {}), gpu: fakeGpu },
  });

  let proc = null;
  try {
    const handler = createWgpuWorkerHandler({
      postMessage: (msg) => posted.push(msg),
      WgpuSwapchain,
    });

    await handler.handleMessage({
      init: true,
      canvas: env.canvas,
      width: 1280,
      height: 720,
      enableResourceReplay: false,
    });

    proc = handler.processor;
    assert.ok(proc && proc.swapchain);
    assert.equal(proc.swapchain.hasCopySrcUsage, true, 'COPY_SRC active before first-visible-frame');

    // Once first-visible-frame fires, swapchain is synced and subsequent resize drops COPY_SRC.
    proc._notifyFirstVisibleFrame({ firstVisibleFrame: true, firstVisibleFrameMs: 1000 });
    assert.equal(proc.swapchain.hasVisibleContent, true);
    assert.equal(proc.swapchain.ensureCopySrcUsage(), false);
    proc.swapchain.resize(1920, 1080);
    assert.equal(
      proc.swapchain.hasCopySrcUsage,
      false,
      'COPY_SRC must drop on resize after first-visible-frame for direct scanout'
    );
  } finally {
    if (proc) proc.destroy();
    if (prevNavDesc) {
      Object.defineProperty(globalThis, 'navigator', prevNavDesc);
    } else {
      delete globalThis.navigator;
    }
  }
});
