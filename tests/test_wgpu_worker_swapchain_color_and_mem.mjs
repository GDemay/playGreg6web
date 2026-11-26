import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const {
  OPCODES,
  WgpuCommandProcessor,
  buildGpuTelemetrySnapshot,
  createWgpuWorkerHandler,
} = require('../web/workers/wgpu_worker.js');
const { WgpuSwapchain } = require('../web/workers/wgpu_swapchain.js');
const {
  installRenderStatePackets,
  SWAPCHAIN_TEXTURE_ID,
  TEXTURE_FORMAT_NAMES: RS_TEXTURE_FORMAT_NAMES,
} = require('../web/workers/render_state_packets.js');
const { installTexturePackets } = require('../web/workers/texture_packets.js');

function formatCodeOf(name) {
  const idx = RS_TEXTURE_FORMAT_NAMES.indexOf(name);
  assert.ok(idx > 0, `unknown WebGPU format ${name}`);
  return idx;
}

class BinaryCommandBuilder {
  constructor() {
    this._chunks = [];
    this._totalBytes = 0;
  }
  _pushPacket(opcode, payloadBytes) {
    const payload = payloadBytes instanceof Uint8Array ? payloadBytes : new Uint8Array(0);
    const pkt = new Uint8Array(8 + payload.byteLength);
    const dv = new DataView(pkt.buffer);
    dv.setUint32(0, opcode >>> 0, true);
    dv.setUint32(4, payload.byteLength >>> 0, true);
    pkt.set(payload, 8);
    this._chunks.push(pkt);
    this._totalBytes += pkt.byteLength;
    return this;
  }
  beginFrame() {
    return this._pushPacket(OPCODES.BEGIN_FRAME, new Uint8Array(0));
  }
  present() {
    return this._pushPacket(OPCODES.PRESENT, new Uint8Array(0));
  }
  createBuffer(id, usage, byteSize) {
    const p = new Uint8Array(16);
    const dv = new DataView(p.buffer);
    dv.setUint32(0, id >>> 0, true);
    dv.setUint32(4, usage >>> 0, true);
    dv.setBigUint64(8, BigInt(byteSize), true);
    return this._pushPacket(OPCODES.CREATE_BUFFER, p);
  }
  createTexture(id, width, height, depthOrArrayLayers, mipLevelCount, formatCode, usage, dimensionCode) {
    const p = new Uint8Array(32);
    const dv = new DataView(p.buffer);
    dv.setUint32(0, id >>> 0, true);
    dv.setUint32(4, width >>> 0, true);
    dv.setUint32(8, height >>> 0, true);
    dv.setUint32(12, depthOrArrayLayers >>> 0, true);
    dv.setUint32(16, mipLevelCount >>> 0, true);
    dv.setUint32(20, formatCode >>> 0, true);
    dv.setUint32(24, usage >>> 0, true);
    dv.setUint32(28, dimensionCode >>> 0, true);
    return this._pushPacket(OPCODES.CREATE_TEXTURE, p);
  }
  destroyResource(id) {
    const p = new Uint8Array(4);
    new DataView(p.buffer).setUint32(0, id >>> 0, true);
    return this._pushPacket(OPCODES.DESTROY_RESOURCE, p);
  }
  finish() {
    const out = new Uint8Array(this._totalBytes);
    let off = 0;
    for (const c of this._chunks) {
      out.set(c, off);
      off += c.byteLength;
    }
    return out;
  }
}

const GPU_TEXTURE_USAGE_RENDER_ATTACHMENT = 0x10;
const GPU_TEXTURE_USAGE_TEXTURE_BINDING = 0x04;
const GPU_TEXTURE_USAGE_COPY_DST = 0x02;

function createFullMockWebGpu(opts = {}) {
  const calls = [];
  const createdTextures = [];
  const createdBuffers = [];
  const writeBuffers = [];
  const canvasViews = [];
  const configured = [];
  const supportedFeatures = new Set(
    opts.features || ['rg11b10ufloat-renderable', 'float32-filterable', 'texture-compression-bc']
  );
  const rejectViewFormats = Boolean(opts.rejectViewFormats);

  const canvas = { width: 1280, height: 720 };
  const ctx = {
    canvas,
    configure(cfg) {
      if (rejectViewFormats && Array.isArray(cfg.viewFormats) && cfg.viewFormats.length > 0) {
        throw new Error('viewFormats unsupported on mock canvas');
      }
      configured.push({ ...cfg });
      calls.push({ op: 'ctx.configure', cfg: { ...cfg } });
    },
    getCurrentTexture() {
      const tex = {
        kind: 'canvasTexture',
        width: canvas.width,
        height: canvas.height,
        createView(desc) {
          const v = {
            kind: 'canvasView',
            index: canvasViews.length,
            format: desc && desc.format ? desc.format : null,
            desc: desc ? { ...desc } : undefined,
          };
          canvasViews.push(v);
          return v;
        },
      };
      return tex;
    },
  };

  const device = {
    features: {
      has: (name) => supportedFeatures.has(name),
    },
    createShaderModule(desc) {
      calls.push({ op: 'createShaderModule', label: desc && desc.label });
      return { kind: 'shaderModule', label: desc && desc.label, code: desc && desc.code };
    },
    createBuffer(desc) {
      const buf = {
        kind: 'buffer',
        size: desc.size,
        usage: desc.usage,
        label: desc.label,
        destroyed: false,
        destroy() {
          this.destroyed = true;
        },
      };
      createdBuffers.push(buf);
      calls.push({ op: 'createBuffer', desc: { ...desc } });
      return buf;
    },
    createTexture(desc) {
      const tex = {
        kind: 'texture',
        desc: { ...desc, size: desc.size ? { ...desc.size } : undefined },
        width: desc.size ? desc.size.width : 1,
        height: desc.size ? desc.size.height : 1,
        depthOrArrayLayers: desc.size ? desc.size.depthOrArrayLayers || 1 : 1,
        mipLevelCount: desc.mipLevelCount || 1,
        sampleCount: desc.sampleCount || 1,
        format: desc.format,
        dimension: desc.dimension || '2d',
        destroyed: false,
        createView(viewDesc) {
          return { kind: 'textureView', texture: tex, desc: viewDesc ? { ...viewDesc } : undefined };
        },
        destroy() {
          this.destroyed = true;
        },
      };
      createdTextures.push(tex);
      calls.push({ op: 'createTexture', desc: tex.desc });
      return tex;
    },
    createSampler(desc) {
      return { kind: 'sampler', desc: desc ? { ...desc } : undefined };
    },
    createBindGroupLayout(desc) {
      return { kind: 'bgl', entries: desc && desc.entries ? desc.entries : [] };
    },
    createPipelineLayout(desc) {
      return { kind: 'pipelineLayout', desc };
    },
    createBindGroup(desc) {
      return { kind: 'bindGroup', desc };
    },
    createRenderPipeline(desc) {
      const pipe = {
        kind: 'renderPipeline',
        label: desc && desc.label,
        desc,
        getBindGroupLayout(idx) {
          return { kind: 'autoBgl', idx, label: desc && desc.label };
        },
      };
      calls.push({ op: 'createRenderPipeline', label: desc && desc.label, desc });
      return pipe;
    },
    createComputePipeline(desc) {
      return { kind: 'computePipeline', desc };
    },
    createCommandEncoder() {
      return {
        beginRenderPass(desc) {
          calls.push({ op: 'beginRenderPass', desc });
          return {
            setPipeline(p) {
              calls.push({ op: 'renderPass.setPipeline', label: p && p.label });
            },
            setBindGroup(idx, bg) {
              calls.push({ op: 'renderPass.setBindGroup', idx, bg });
            },
            setVertexBuffer() {},
            setIndexBuffer() {},
            setViewport() {},
            setScissorRect() {},
            draw(vertexCount, instanceCount, firstVertex, firstInstance) {
              calls.push({
                op: 'renderPass.draw',
                vertexCount,
                instanceCount,
                firstVertex,
                firstInstance,
              });
            },
            drawIndexed() {},
            end() {
              calls.push({ op: 'renderPass.end' });
            },
          };
        },
        beginComputePass() {
          return { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
        },
        finish() {
          calls.push({ op: 'encoder.finish' });
          return { kind: 'commandBuffer' };
        },
      };
    },
    queue: {
      writeBuffer(buffer, offset, data) {
        const bytes =
          data instanceof Uint8Array
            ? new Uint8Array(data)
            : ArrayBuffer.isView(data)
              ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice()
              : new Uint8Array(data).slice();
        writeBuffers.push({ buffer, offset, bytes });
      },
      writeTexture() {},
      submit(cbs) {
        calls.push({ op: 'queue.submit', count: cbs.length });
      },
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
  };

  return {
    canvas,
    ctx,
    device,
    calls,
    createdTextures,
    createdBuffers,
    writeBuffers,
    canvasViews,
    configured,
  };
}

function buildSwapchainBeginRenderPassPayload(formatName) {
  const formatCode = formatCodeOf(formatName);
  const buf = new ArrayBuffer(368);
  const view = new DataView(buf);
  view.setUint32(0, 1, true); // colorCount = 1
  const o = 8;
  view.setUint32(o, SWAPCHAIN_TEXTURE_ID, true);
  view.setUint8(o + 4, 1); // loadOp = 'clear'
  view.setUint8(o + 5, 1); // storeOp = 'store'
  view.setUint8(o + 8, formatCode); // view.format
  view.setFloat32(o + 24, 0.5, true);
  view.setFloat32(o + 28, 0.5, true);
  view.setFloat32(o + 32, 0.5, true);
  view.setFloat32(o + 36, 1.0, true);
  return new Uint8Array(buf);
}

function buildCreateTexture2Packet({
  id,
  width,
  height,
  depth = 1,
  mipLevelCount = 1,
  sampleCount = 1,
  usage = GPU_TEXTURE_USAGE_TEXTURE_BINDING | GPU_TEXTURE_USAGE_COPY_DST,
  dimensionCode = 1, // 0=1d, 1=2d, 2=3d
  formatName = 'rgba8unorm',
}) {
  const formatCode = formatCodeOf(formatName);
  const payload = new Uint8Array(40);
  const view = new DataView(payload.buffer);
  view.setUint32(0, id >>> 0, true);
  view.setUint32(4, width >>> 0, true);
  view.setUint32(8, height >>> 0, true);
  view.setUint32(12, depth >>> 0, true);
  view.setUint32(16, mipLevelCount >>> 0, true);
  view.setUint32(20, sampleCount >>> 0, true);
  view.setUint32(24, usage >>> 0, true);
  view.setUint8(28, dimensionCode);
  view.setUint8(29, formatCode);
  view.setUint8(30, 0); // viewFormatCount
  const builder = new BinaryCommandBuilder();
  builder._pushPacket(OPCODES.CREATE_TEXTURE2, payload);
  return builder.finish();
}

function decodeToneMapUniform(writeRecord) {
  assert.ok(writeRecord && writeRecord.bytes.byteLength >= 16);
  const view = new DataView(
    writeRecord.bytes.buffer,
    writeRecord.bytes.byteOffset,
    writeRecord.bytes.byteLength
  );
  return {
    mode: view.getUint32(0, true),
    exposure: view.getFloat32(4, true),
    applySrgb: view.getUint32(8, true),
  };
}

test('Orch-20 sRGB swapchain formats (bgra8unorm-srgb & rgba8unorm-srgb) enable linear->sRGB blit encoding so midtones are never double-darkened', () => {
  const mock = createFullMockWebGpu();
  const swapchain = new WgpuSwapchain({
    canvas: mock.canvas,
    device: mock.device,
    context: mock.ctx,
    format: 'bgra8unorm',
    enableDepth: false,
  });
  swapchain.configure({ width: 1280, height: 720 });

  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
    swapchain,
  });
  installRenderStatePackets(processor);
  installTexturePackets(processor);

  for (const srgbFormat of ['bgra8unorm-srgb', 'rgba8unorm-srgb']) {
    const writesBefore = mock.writeBuffers.length;
    const builder = new BinaryCommandBuilder();
    builder.beginFrame();
    builder._pushPacket(OPCODES.BEGIN_RENDER_PASS, buildSwapchainBeginRenderPassPayload(srgbFormat));
    builder._pushPacket(OPCODES.END_RENDER_PASS, new Uint8Array(0));
    builder.present();
    processor.executeCommandBuffer(builder.finish());

    assert.equal(swapchain.usesBlitFallback, true);
    assert.equal(swapchain.streamFormat, srgbFormat);
    assert.equal(swapchain.format, 'bgra8unorm');

    const toneMapWrites = mock.writeBuffers
      .slice(writesBefore)
      .filter((w) => w.buffer && w.buffer === swapchain._toneMapUniformBuffer);
    assert.equal(
      toneMapWrites.length,
      1,
      `${srgbFormat} must invoke BLIT_TONEMAP_WGSL to re-encode linear->sRGB`
    );
    const params = decodeToneMapUniform(toneMapWrites[0]);
    assert.equal(params.mode, 0, 'sRGB blit must not apply HDR tonemap compression');
    assert.equal(params.exposure, 1.0);
    assert.equal(params.applySrgb, 1, `${srgbFormat} blit must set apply_srgb=1`);

    const stats = processor.getStats();
    assert.equal(stats.swapchainFormat, srgbFormat);
    assert.equal(stats.toneMapMode, 'none');
    assert.equal(stats.applySrgbEncode, true);
  }

  // Switching back to non-sRGB SDR bgra8unorm disables blit & resets applySrgbEncode.
  assert.equal(processor.setSwapchainFormat('bgra8unorm'), true);
  assert.equal(swapchain.usesBlitFallback, false);
  assert.equal(processor.getStats().applySrgbEncode, false);

  processor.destroy();
  swapchain.destroy();
});

test('Orch-20 directSrgbView uses zero-copy sRGB canvas view when supported, survives rebindDevice, and falls back to applySrgbEncode blit when viewFormats are rejected', () => {
  // 1. Supported viewFormats -> zero-copy direct sRGB view (blitCount === 0)
  const mock = createFullMockWebGpu();
  const swapchain = new WgpuSwapchain({
    canvas: mock.canvas,
    device: mock.device,
    context: mock.ctx,
    format: 'bgra8unorm',
    enableDepth: false,
  });
  swapchain.configure({ width: 1280, height: 720 });

  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
    swapchain,
    directSrgbView: true,
  });

  assert.equal(processor.setSwapchainFormat('bgra8unorm-srgb'), true);
  assert.equal(swapchain.usesBlitFallback, false);
  assert.equal(processor._swapchainViewFormat, 'bgra8unorm-srgb');
  assert.equal(processor._getRenderTargetFormat(), 'bgra8unorm-srgb');

  processor.executeCommandBuffer(new BinaryCommandBuilder().beginFrame().present().finish());
  assert.equal(swapchain.blitCount, 0);
  assert.equal(mock.canvasViews[mock.canvasViews.length - 1].format, 'bgra8unorm-srgb');

  const snap = buildGpuTelemetrySnapshot(processor, { timestamp: 100 });
  assert.deepEqual(snap.swapchainStats, {
    swapchainFormat: 'bgra8unorm-srgb',
    canvasFormat: 'bgra8unorm',
    usesBlitFallback: false,
    blitCount: 0,
  });

  // Device loss + recovery preserves zero-copy direct sRGB view (does NOT flip usesBlitFallback).
  const mock2 = createFullMockWebGpu();
  processor.rebindDevice(mock2.device, mock.ctx);
  assert.equal(swapchain.usesBlitFallback, false);
  assert.equal(processor._swapchainViewFormat, 'bgra8unorm-srgb');
  assert.equal(processor._getRenderTargetFormat(), 'bgra8unorm-srgb');

  processor.destroy();
  swapchain.destroy();

  // 2. Rejected viewFormats -> automatic fallback to sRGB intermediate + applySrgbEncode: true blit
  const mockReject = createFullMockWebGpu({ rejectViewFormats: true });
  const swapReject = new WgpuSwapchain({
    canvas: mockReject.canvas,
    device: mockReject.device,
    context: mockReject.ctx,
    format: 'bgra8unorm',
    enableDepth: false,
  });
  swapReject.configure({ width: 1280, height: 720 });
  const procReject = new WgpuCommandProcessor(mockReject.device, mockReject.ctx, {
    format: 'bgra8unorm',
    swapchain: swapReject,
    directSrgbView: true,
  });

  assert.equal(procReject.setSwapchainFormat('bgra8unorm-srgb'), true);
  assert.equal(swapReject.usesBlitFallback, true);
  assert.equal(procReject._swapchainViewFormat, null);
  assert.equal(procReject.getStats().applySrgbEncode, true);

  procReject.destroy();
  swapReject.destroy();
});

test('Orch-20 HDR linear (rgba16float, rg11b10ufloat, rgba32float) tonemaps via ACES+sRGB onto SDR canvas, guards missing features, and keeps rgb10a2unorm SDR-safe by default', () => {
  // Missing rg11b10ufloat-renderable and float32-filterable -> must safely substitute rgba16float
  const mock = createFullMockWebGpu({ features: [] });
  const swapchain = new WgpuSwapchain({
    canvas: mock.canvas,
    device: mock.device,
    context: mock.ctx,
    format: 'bgra8unorm',
    enableDepth: false,
  });
  swapchain.configure({ width: 1280, height: 720 });

  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
    swapchain,
    hdrMode: 'sdr-tonemap',
  });

  // 1. rgba16float in sdr-tonemap mode -> keeps SDR canvas + ACES + applySrgbEncode=true
  assert.equal(processor.setSwapchainFormat('rgba16float'), true);
  assert.equal(swapchain.usesBlitFallback, true);
  assert.equal(swapchain.streamFormat, 'rgba16float');
  assert.equal(swapchain.format, 'bgra8unorm');

  processor.executeCommandBuffer(new BinaryCommandBuilder().beginFrame().present().finish());
  const hdrWrites = mock.writeBuffers.filter(
    (w) => w.buffer && w.buffer === swapchain._toneMapUniformBuffer
  );
  assert.equal(hdrWrites.length, 1);
  assert.deepEqual(decodeToneMapUniform(hdrWrites[0]), {
    mode: 2, // ACES
    exposure: 1.0,
    applySrgb: 1,
  });

  // 2. Runtime tone-map update to Reinhard + exposure 1.25
  const updated = processor.setSwapchainToneMapOptions({ mode: 'reinhard', exposure: 1.25 });
  assert.equal(updated.toneMapMode, 'reinhard');
  assert.equal(updated.exposure, 1.25);
  assert.equal(updated.applySrgbEncode, true);
  processor.executeCommandBuffer(new BinaryCommandBuilder().beginFrame().present().finish());
  const latestWrite = mock.writeBuffers
    .filter((w) => w.buffer && w.buffer === swapchain._toneMapUniformBuffer)
    .pop();
  assert.deepEqual(decodeToneMapUniform(latestWrite), {
    mode: 1, // Reinhard
    exposure: 1.25,
    applySrgb: 1,
  });

  // 3. rg11b10ufloat & rgba32float without optional features safely fall back to rgba16float
  assert.equal(processor.setSwapchainFormat('rg11b10ufloat'), true);
  assert.equal(swapchain.streamFormat, 'rgba16float');
  assert.equal(processor.setSwapchainFormat('rgba32float'), true);
  assert.equal(swapchain.streamFormat, 'rgba16float');

  // 4. rgb10a2unorm defaults to SDR passthrough (no double-gamma) unless linear transfer requested
  processor.setSwapchainToneMapOptions({ mode: 'auto', exposure: 1.0, applySrgbEncode: null });
  assert.equal(processor.setSwapchainFormat('rgb10a2unorm'), true);
  assert.equal(swapchain.usesBlitFallback, true);
  assert.equal(swapchain.streamFormat, 'rgb10a2unorm');
  assert.equal(processor.getStats().toneMapMode, 'none');
  assert.equal(processor.getStats().applySrgbEncode, false);

  // Explicit linear transfer on rgb10a2unorm activates ACES + sRGB encode
  processor.setSwapchainToneMapOptions({ rgb10a2Transfer: 'linear' });
  assert.equal(processor.getStats().toneMapMode, 'aces');
  assert.equal(processor.getStats().applySrgbEncode, true);

  // 5. Extended HDR opt-in switches rgba16float to direct canvas output
  processor.setSwapchainToneMapOptions({ hdrMode: 'extended', mode: 'none', applySrgbEncode: null });
  assert.equal(processor.setSwapchainFormat('rgba16float'), true);
  assert.equal(swapchain.usesBlitFallback, false);
  assert.equal(swapchain.format, 'rgba16float');

  processor.destroy();
  swapchain.destroy();
});

test('Orch-20 preserves alphaMode on raw ctx paths and reconciles owned BinkVideoPresenter on canvas format changes', () => {
  const mock = createFullMockWebGpu();
  const createdPresenters = [];
  class MockBinkPresenter {
    constructor(opts) {
      this.presentationFormat = opts.presentationFormat;
      this.destroyed = false;
      createdPresenters.push(this);
    }
    destroy() {
      this.destroyed = true;
    }
  }

  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
    alphaMode: 'premultiplied',
    BinkVideoPresenter: MockBinkPresenter,
    SplashFrameBridge: null,
  });

  // Create presenter for bgra8unorm
  processor._ensureSplashBridge();
  assert.equal(createdPresenters.length, 1);
  assert.equal(createdPresenters[0].presentationFormat, 'bgra8unorm');

  // Switch raw ctx to rgba8unorm -> preserves premultiplied alphaMode & recreates presenter
  assert.equal(processor.setSwapchainFormat('rgba8unorm'), true);
  assert.equal(mock.configured[mock.configured.length - 1].alphaMode, 'premultiplied');
  assert.equal(mock.configured[mock.configured.length - 1].format, 'rgba8unorm');

  processor._ensureSplashBridge();
  assert.equal(createdPresenters[0].destroyed, true);
  assert.equal(createdPresenters.length, 2);
  assert.equal(createdPresenters[1].presentationFormat, 'rgba8unorm');

  // rebindDevice also preserves premultiplied alphaMode when opts.alphaMode is omitted
  const mock2 = createFullMockWebGpu();
  processor.rebindDevice(mock2.device, mock.ctx);
  assert.equal(mock.configured[mock.configured.length - 1].alphaMode, 'premultiplied');

  processor.destroy();
});

test('Orch-20 #1077 tracks exact live/peak buffer & texture bytes across TexturePackets/RenderStatePackets, BCn, mips, MSAA, overwrites, and gpu-stats replies', async () => {
  const mock = createFullMockWebGpu();
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
  });
  installRenderStatePackets(processor);
  installTexturePackets(processor);

  // 1. Create two buffers (64 KiB + 128 KiB = 192 KiB)
  const b1 = new BinaryCommandBuilder()
    .createBuffer(1, 0x28, 65536)
    .createBuffer(2, 0x28, 131072)
    .finish();
  processor.executeCommandBuffer(b1);
  assert.equal(processor.getStats().liveBufferBytes, 196608);
  assert.equal(processor.getStats().peakBufferBytes, 196608);

  // Inject an untracked buffer and destroy it -> must NOT corrupt _liveBufferBytes
  processor.buffers.set(999, { size: 999999, destroy() {} });
  processor._destroyBufferIfPresent(999);
  assert.equal(processor.getStats().liveBufferBytes, 196608);

  // 2. Create v1 texture (via TexturePackets hook): 64x64 rgba8unorm, 2 mips = 64*64*4 + 32*32*4 = 20,480 B
  const t1 = new BinaryCommandBuilder()
    .createTexture(10, 64, 64, 1, 2, 0, GPU_TEXTURE_USAGE_TEXTURE_BINDING | GPU_TEXTURE_USAGE_COPY_DST, 1)
    .finish();
  processor.executeCommandBuffer(t1);
  const expectedT1 = 64 * 64 * 4 + 32 * 32 * 4;
  assert.equal(processor.getStats().liveTextureBytes, expectedT1);
  assert.equal(processor.getStats().peakTextureBytes, expectedT1);

  // 3. Create v2 BC1 texture: 16x16 bc1-rgba-unorm, 1 mip = 4*4*8 = 128 B
  processor.executeCommandBuffer(
    buildCreateTexture2Packet({ id: 11, width: 16, height: 16, formatName: 'bc1-rgba-unorm' })
  );
  const expectedT2 = 4 * 4 * 8;

  // 4. Create v2 BC7-sRGB texture: 16x16 bc7-rgba-unorm-srgb, 2 mips = (4*4 + 2*2)*16 = 320 B
  processor.executeCommandBuffer(
    buildCreateTexture2Packet({
      id: 12,
      width: 16,
      height: 16,
      mipLevelCount: 2,
      formatName: 'bc7-rgba-unorm-srgb',
    })
  );
  const expectedT3 = (4 * 4 + 2 * 2) * 16;

  // 5. Create v2 4x MSAA rgba16float render target: 32x32, 8 Bpp, 4 samples = 32*32*8*4 = 32,768 B
  processor.executeCommandBuffer(
    buildCreateTexture2Packet({
      id: 13,
      width: 32,
      height: 32,
      sampleCount: 4,
      usage: GPU_TEXTURE_USAGE_RENDER_ATTACHMENT,
      formatName: 'rgba16float',
    })
  );
  const expectedT4 = 32 * 32 * 8 * 4;

  // 6. Create v2 r8unorm texture (1 Bpp): 64x32 = 2,048 B
  processor.executeCommandBuffer(
    buildCreateTexture2Packet({ id: 14, width: 64, height: 32, formatName: 'r8unorm' })
  );
  const expectedT5 = 64 * 32 * 1;

  const expectedTotalTex = expectedT1 + expectedT2 + expectedT3 + expectedT4 + expectedT5;
  const statsMid = processor.getStats();
  assert.equal(statsMid.liveTextureBytes, expectedTotalTex);
  assert.equal(statsMid.peakTextureBytes, expectedTotalTex);
  assert.equal(statsMid.gpuMemory.liveTextureBytes, expectedTotalTex);
  assert.equal(statsMid.gpuMemory.peakTextureBytes, expectedTotalTex);
  assert.equal(statsMid.gpuMemory.liveBufferBytes, 196608);
  assert.equal(statsMid.gpuMemory.peakBufferBytes, 196608);

  // Verify buildGpuTelemetrySnapshot surfaces gpuMemory
  const snap = buildGpuTelemetrySnapshot(processor, { timestamp: 42 });
  assert.equal(snap.gpuMemory.liveBufferBytes, 196608);
  assert.equal(snap.gpuMemory.peakBufferBytes, 196608);
  assert.equal(snap.gpuMemory.liveTextureBytes, expectedTotalTex);
  assert.equal(snap.gpuMemory.peakTextureBytes, expectedTotalTex);

  // Verify worker `type: 'gpu-stats'` reply promotes gpuMemory + top-level byte fields
  const posted = [];
  const handler = createWgpuWorkerHandler({
    postMessage: (msg) => posted.push(msg),
    initGpu: async () => processor,
  });
  await handler.handleMessage({ init: true, canvas: mock.canvas, width: 1280, height: 720 });
  await handler.handleMessage({ type: 'gpu-stats', id: 77 });
  const statsReply = posted.find((m) => m && m.type === 'gpu-stats' && m.id === 77);
  assert.ok(statsReply, 'expected gpu-stats reply');
  assert.equal(statsReply.liveBufferBytes, 196608);
  assert.equal(statsReply.peakBufferBytes, 196608);
  assert.equal(statsReply.liveTextureBytes, expectedTotalTex);
  assert.equal(statsReply.peakTextureBytes, expectedTotalTex);
  assert.equal(statsReply.gpuMemory.liveTextureBytes, expectedTotalTex);
  assert.equal(statsReply.gpuMemory.peakTextureBytes, expectedTotalTex);

  // 7. Overwrite texture 13 with smaller 8x8 rgba8unorm (256 B) and destroy all -> live returns to 0, peak preserved
  processor.executeCommandBuffer(
    buildCreateTexture2Packet({ id: 13, width: 8, height: 8, formatName: 'rgba8unorm' })
  );
  assert.equal(processor.getStats().liveTextureBytes, expectedTotalTex - expectedT4 + 256);
  assert.equal(processor.getStats().peakTextureBytes, expectedTotalTex);

  const destroyAll = new BinaryCommandBuilder()
    .destroyResource(1)
    .destroyResource(2)
    .destroyResource(10)
    .destroyResource(11)
    .destroyResource(12)
    .destroyResource(13)
    .destroyResource(14)
    .finish();
  processor.executeCommandBuffer(destroyAll);

  const finalStats = processor.getStats();
  assert.equal(finalStats.liveBufferBytes, 0);
  assert.equal(finalStats.liveTextureBytes, 0);
  assert.equal(finalStats.peakBufferBytes, 196608);
  assert.equal(finalStats.peakTextureBytes, expectedTotalTex);

  processor.destroy();
});

test('Orch-20 loadShaderPackUrl refuses full-buffer GET for multi-GB (>256 MiB) pack with Accept-Ranges: none', async () => {
  const mock = createFullMockWebGpu();
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
  });

  let getCalled = false;
  const fakeFetch = async (_url, init) => {
    const method = (init && init.method ? init.method : 'GET').toUpperCase();
    if (method === 'HEAD') {
      return {
        ok: true,
        status: 200,
        headers: new Map([
          ['content-length', String(8_063_800_000)], // 7.51 GiB
          ['accept-ranges', 'none'],
        ]),
      };
    }
    getCalled = true;
    return { ok: true, status: 200, arrayBuffer: async () => new ArrayBuffer(0) };
  };

  await assert.rejects(
    () => processor.loadShaderPackUrl('/shaders/huge.wgsl.pack', { fetch: fakeFetch }),
    /refusing full-buffer download/i
  );
  assert.equal(getCalled, false, 'must never issue full-buffer GET for 7.51 GB Accept-Ranges: none pack');
  processor.destroy();
});
