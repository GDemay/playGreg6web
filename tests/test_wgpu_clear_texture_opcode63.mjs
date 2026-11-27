import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const wgpuMod = require('../web/workers/wgpu_worker.js');
const rsMod = require('../web/workers/render_state_packets.js');

const {
  WgpuCommandProcessor,
  OPCODES,
  OPCODE_CLEAR_TEXTURE,
  CLEAR_TEXTURE_PAYLOAD_BYTES,
  TEXTURE_USAGE,
  TEXTURE_FORMAT,
  TEXTURE_DIMENSION,
  BUFFER_USAGE,
  planClearTextureCopies,
} = wgpuMod;

const {
  installRenderStatePackets,
  V2_OPCODES,
  OPCODE_CLEAR_TEXTURE: RS_OPCODE_CLEAR_TEXTURE,
  TEXTURE_FORMAT_NAMES: V2_TEXTURE_FORMAT_NAMES,
} = rsMod;

function v2FormatCode(name) {
  const idx = V2_TEXTURE_FORMAT_NAMES.indexOf(name);
  assert.ok(idx > 0, `unknown v2 format ${name}`);
  return idx;
}

class StreamBuilder {
  constructor() {
    this.chunks = [];
  }

  packet(opcode, payload = new Uint8Array(0)) {
    const bytes = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
    const out = new Uint8Array(8 + bytes.byteLength);
    const view = new DataView(out.buffer);
    view.setUint16(0, opcode, true);
    view.setUint16(2, 0, true);
    view.setUint32(4, bytes.byteLength, true);
    out.set(bytes, 8);
    this.chunks.push(out);
    return this;
  }

  beginFrame() {
    return this.packet(OPCODES.BEGIN_FRAME);
  }

  clearColor(r, g, b, a) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setFloat32(0, r, true);
    view.setFloat32(4, g, true);
    view.setFloat32(8, b, true);
    view.setFloat32(12, a, true);
    return this.packet(OPCODES.CLEAR_COLOR, new Uint8Array(buf));
  }

  present() {
    return this.packet(OPCODES.PRESENT);
  }

  destroyResource(id) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, id, true);
    return this.packet(OPCODES.DESTROY_RESOURCE, new Uint8Array(buf));
  }

  createTexture(
    id,
    width,
    height,
    depthOrLayers = 1,
    mipLevels = 1,
    format = TEXTURE_FORMAT.RGBA8_UNORM,
    usage = TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
    dimension = TEXTURE_DIMENSION.DIM_2D
  ) {
    const buf = new ArrayBuffer(32);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, width, true);
    view.setUint32(8, height, true);
    view.setUint32(12, depthOrLayers, true);
    view.setUint32(16, mipLevels, true);
    view.setUint32(20, format, true);
    view.setUint32(24, usage, true);
    view.setUint32(28, dimension, true);
    return this.packet(OPCODES.CREATE_TEXTURE, new Uint8Array(buf));
  }

  createTexture2({
    id,
    width,
    height,
    depthOrLayers = 1,
    mipLevels = 1,
    sampleCount = 1,
    usage = TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
    dimension = TEXTURE_DIMENSION.DIM_2D,
    format = 'rgba8unorm',
    viewFormats = [],
  }) {
    const buf = new ArrayBuffer(40);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, width, true);
    view.setUint32(8, height, true);
    view.setUint32(12, depthOrLayers, true);
    view.setUint32(16, mipLevels, true);
    view.setUint32(20, sampleCount, true);
    view.setUint32(24, usage, true);
    view.setUint8(28, dimension);
    view.setUint8(29, typeof format === 'number' ? format : v2FormatCode(format));
    view.setUint8(30, viewFormats.length);
    view.setUint8(31, 0);
    for (let i = 0; i < Math.min(8, viewFormats.length); i += 1) {
      view.setUint8(32 + i, typeof viewFormats[i] === 'number' ? viewFormats[i] : v2FormatCode(viewFormats[i]));
    }
    return this.packet(OPCODES.CREATE_TEXTURE2, new Uint8Array(buf));
  }

  writeTexture({
    textureId,
    mipLevel = 0,
    originX = 0,
    originY = 0,
    originZ = 0,
    width,
    height,
    depthOrLayers = 1,
    bytesPerRow = 0,
    rowsPerImage = 0,
    bytes,
  }) {
    const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    const buf = new ArrayBuffer(48 + data.byteLength);
    const view = new DataView(buf);
    view.setUint32(0, textureId, true);
    view.setUint32(4, mipLevel, true);
    view.setUint32(8, originX, true);
    view.setUint32(12, originY, true);
    view.setUint32(16, originZ, true);
    view.setUint32(20, width, true);
    view.setUint32(24, height, true);
    view.setUint32(28, depthOrLayers, true);
    view.setUint32(32, bytesPerRow, true);
    view.setUint32(36, rowsPerImage, true);
    view.setUint32(40, data.byteLength, true);
    view.setUint32(44, 0, true);
    new Uint8Array(buf, 48).set(data);
    return this.packet(OPCODES.WRITE_TEXTURE, new Uint8Array(buf));
  }

  clearTexture(textureId, reserved = 0) {
    const buf = new ArrayBuffer(8);
    const view = new DataView(buf);
    view.setUint32(0, textureId, true);
    view.setUint32(4, reserved, true);
    return this.packet(OPCODE_CLEAR_TEXTURE, new Uint8Array(buf));
  }

  finish() {
    const total = this.chunks.reduce((sum, c) => sum + c.byteLength, 0);
    const out = new Uint8Array(total);
    let off = 0;
    for (const c of this.chunks) {
      out.set(c, off);
      off += c.byteLength;
    }
    return out.buffer;
  }
}

function createMockWebGpu({ features = [], supportCopyBufferToTexture = true } = {}) {
  const events = [];
  const createdBuffers = [];
  const renderPasses = [];
  const copyBufferToTextures = [];
  const writeTextures = [];
  const submits = [];
  let nextHandle = 1;
  const swapchainView = { kind: 'GPUTextureView', id: 'swapchain-view' };

  const ctx = {
    getCurrentTexture() {
      return {
        createView() {
          return swapchainView;
        },
      };
    },
  };

  const device = {
    features: new Set(features),
    createBuffer(desc) {
      const buf = {
        kind: 'GPUBuffer',
        handle: nextHandle++,
        size: desc.size,
        usage: desc.usage,
        destroyed: false,
        destroy() {
          this.destroyed = true;
          events.push({ op: 'buffer.destroy', handle: this.handle });
        },
      };
      createdBuffers.push(buf);
      events.push({ op: 'createBuffer', size: desc.size, usage: desc.usage, handle: buf.handle });
      return buf;
    },
    createTexture(desc) {
      const size = desc.size || {};
      const tex = {
        kind: 'GPUTexture',
        handle: nextHandle++,
        desc: { ...desc, size: { ...size } },
        width: size.width || 1,
        height: size.height || 1,
        depthOrArrayLayers: size.depthOrArrayLayers || 1,
        mipLevelCount: desc.mipLevelCount || 1,
        sampleCount: desc.sampleCount || 1,
        format: desc.format,
        usage: desc.usage,
        dimension: desc.dimension || '2d',
        destroyed: false,
        views: [],
        createView(viewDesc) {
          const view = {
            kind: 'GPUTextureView',
            handle: nextHandle++,
            textureHandle: tex.handle,
            viewDesc: viewDesc ? { ...viewDesc } : null,
          };
          tex.views.push(view);
          return view;
        },
        destroy() {
          this.destroyed = true;
          events.push({ op: 'texture.destroy', handle: this.handle });
        },
      };
      events.push({ op: 'createTexture', handle: tex.handle, desc: tex.desc });
      return tex;
    },
    createCommandEncoder() {
      const encOps = [];
      const encoder = {
        beginRenderPass(desc) {
          const snapshot = {
            colorAttachments: (desc.colorAttachments || []).map((a) => ({
              view: a.view,
              loadOp: a.loadOp,
              storeOp: a.storeOp,
              clearValue: a.clearValue ? { ...a.clearValue } : null,
            })),
          };
          renderPasses.push(snapshot);
          encOps.push({ op: 'beginRenderPass', desc: snapshot });
          return {
            setPipeline() {},
            setBindGroup() {},
            setVertexBuffer() {},
            setIndexBuffer() {},
            draw() {},
            drawIndexed() {},
            end() {
              encOps.push({ op: 'renderPass.end' });
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
        finish() {
          const cb = { kind: 'GPUCommandBuffer', handle: nextHandle++, ops: encOps };
          events.push({ op: 'encoder.finish', handle: cb.handle, opCount: encOps.length });
          return cb;
        },
      };
      if (supportCopyBufferToTexture) {
        encoder.copyBufferToTexture = (src, dst, copySize) => {
          const rec = {
            src: {
              buffer: src.buffer,
              offset: src.offset || 0,
              bytesPerRow: src.bytesPerRow,
              rowsPerImage: src.rowsPerImage,
            },
            dst: {
              texture: dst.texture,
              mipLevel: dst.mipLevel,
              origin: { ...dst.origin },
              aspect: dst.aspect,
            },
            copySize: { ...copySize },
          };
          copyBufferToTextures.push(rec);
          encOps.push({ op: 'copyBufferToTexture', ...rec });
        };
      }
      return encoder;
    },
    queue: {
      writeBuffer() {},
      writeTexture(dst, data, dataLayout, size) {
        const bytes = new Uint8Array(
          data.buffer,
          data.byteOffset,
          data.byteLength
        ).slice();
        const rec = {
          dst: {
            texture: dst.texture,
            mipLevel: dst.mipLevel,
            origin: dst.origin ? { ...dst.origin } : { x: 0, y: 0, z: 0 },
            aspect: dst.aspect,
          },
          data: bytes,
          dataLayout: { ...dataLayout },
          size: { ...size },
        };
        writeTextures.push(rec);
        events.push({
          op: 'queue.writeTexture',
          textureHandle: dst.texture.handle,
          mipLevel: dst.mipLevel,
          firstByte: bytes.length > 0 ? bytes[0] : -1,
          byteLength: bytes.byteLength,
        });
      },
      submit(buffers) {
        submits.push(buffers);
        events.push({
          op: 'queue.submit',
          handles: buffers.map((b) => b.handle),
          ops: buffers.flatMap((b) => b.ops.map((o) => o.op)),
        });
      },
    },
  };

  return {
    device,
    ctx,
    events,
    createdBuffers,
    renderPasses,
    copyBufferToTextures,
    writeTextures,
    submits,
  };
}

test('ClearTexture opcode 63 matches playgta6_abi.h and keeps OPCODES/V2_OPCODES parity intact', () => {
  assert.equal(OPCODE_CLEAR_TEXTURE, 63);
  assert.equal(RS_OPCODE_CLEAR_TEXTURE, 63);
  assert.equal(CLEAR_TEXTURE_PAYLOAD_BYTES, 8);
  assert.equal('CLEAR_TEXTURE' in OPCODES, false, 'must stay outside OPCODES for repo_health GpuOpcode parity');
  assert.equal('CLEAR_TEXTURE' in V2_OPCODES, false, 'must stay outside V2_OPCODES for enum parity');

  const abi = readFileSync(new URL('../runtime/abi/playgta6_abi.h', import.meta.url), 'utf8');
  assert.match(abi, /kGpuOpcodeClearTexture\s*=\s*static_cast<GpuOpcode>\(63\)/);
  assert.match(abi, /static_assert\(sizeof\(GpuClearTexturePayload\)\s*==\s*8/);
});

test('ClearTexture clears 2D RGBA8 RENDER_ATTACHMENT texture across all mips and array layers via render passes', () => {
  const mock = createMockWebGpu();
  const proc = new WgpuCommandProcessor(mock.device, mock.ctx);
  const rs = installRenderStatePackets(proc);

  const stream = new StreamBuilder()
    .createTexture2({
      id: 10,
      width: 16,
      height: 8,
      depthOrLayers: 2,
      mipLevels: 3,
      format: 'rgba8unorm',
      usage: TEXTURE_USAGE.RENDER_ATTACHMENT | TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
    })
    .clearTexture(10)
    .finish();

  proc.executeCommandBuffer(stream);

  // 3 mips * 2 array layers = 6 clearing render passes in a single submit.
  assert.equal(mock.renderPasses.length, 6);
  assert.equal(mock.submits.length, 1);
  assert.equal(mock.copyBufferToTextures.length, 0);
  assert.equal(mock.writeTextures.length, 0);
  assert.equal(proc.getValidationErrors().length, 0);
  assert.equal(rs.telemetry.textureClears, 1);
  assert.equal(proc.getStats().clearTextureStats.renderPassClears, 1);

  const seen = new Set();
  for (const pass of mock.renderPasses) {
    assert.equal(pass.colorAttachments.length, 1);
    const att = pass.colorAttachments[0];
    assert.equal(att.loadOp, 'clear');
    assert.equal(att.storeOp, 'store');
    assert.deepEqual(att.clearValue, { r: 0, g: 0, b: 0, a: 0 });
    assert.equal(att.view.viewDesc.dimension, '2d');
    assert.equal(att.view.viewDesc.mipLevelCount, 1);
    assert.equal(att.view.viewDesc.arrayLayerCount, 1);
    seen.add(`${att.view.viewDesc.baseMipLevel}:${att.view.viewDesc.baseArrayLayer}`);
  }
  assert.deepEqual(
    [...seen].sort(),
    ['0:0', '0:1', '1:0', '1:1', '2:0', '2:1']
  );
});

test('ClearTexture clears non-RENDER_ATTACHMENT 2D RGBA8 with mips + array layers via copyBufferToTexture and writeTexture fallback', () => {
  // 1) copyBufferToTexture path with reusable zero GPUBuffer.
  {
    const mock = createMockWebGpu({ supportCopyBufferToTexture: true });
    const proc = new WgpuCommandProcessor(mock.device, mock.ctx);

    const stream = new StreamBuilder()
      .createTexture(
        20,
        16,
        8,
        2,
        3,
        TEXTURE_FORMAT.RGBA8_UNORM,
        TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
        TEXTURE_DIMENSION.DIM_2D
      )
      .clearTexture(20)
      .finish();

    proc.executeCommandBuffer(stream);

    assert.equal(mock.renderPasses.length, 0);
    assert.equal(mock.submits.length, 1);
    assert.equal(mock.createdBuffers.length, 1);
    assert.equal(mock.createdBuffers[0].usage, BUFFER_USAGE.COPY_SRC);
    assert.equal(mock.copyBufferToTextures.length, 3);

    const [m0, m1, m2] = mock.copyBufferToTextures;
    assert.deepEqual(m0.copySize, { width: 16, height: 8, depthOrArrayLayers: 2 });
    assert.equal(m0.dst.mipLevel, 0);
    assert.equal(m0.src.bytesPerRow, 256);
    assert.equal(m0.src.rowsPerImage, 8);

    assert.deepEqual(m1.copySize, { width: 8, height: 4, depthOrArrayLayers: 2 });
    assert.equal(m1.dst.mipLevel, 1);
    assert.equal(m1.src.bytesPerRow, 256);
    assert.equal(m1.src.rowsPerImage, 4);

    assert.deepEqual(m2.copySize, { width: 4, height: 2, depthOrArrayLayers: 2 });
    assert.equal(m2.dst.mipLevel, 2);
    assert.equal(m2.src.bytesPerRow, 256);
    assert.equal(m2.src.rowsPerImage, 2);
    assert.equal(proc.getValidationErrors().length, 0);
  }

  // 2) queue.writeTexture fallback when encoder.copyBufferToTexture is unavailable.
  {
    const mock = createMockWebGpu({ supportCopyBufferToTexture: false });
    const proc = new WgpuCommandProcessor(mock.device, mock.ctx);

    const stream = new StreamBuilder()
      .createTexture(
        21,
        16,
        8,
        2,
        3,
        TEXTURE_FORMAT.RGBA8_UNORM,
        TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
        TEXTURE_DIMENSION.DIM_2D
      )
      .clearTexture(21)
      .finish();

    proc.executeCommandBuffer(stream);

    assert.equal(mock.writeTextures.length, 3);
    assert.equal(proc.getStats().clearTextureStats.writeTextureClears, 1);
    assert.deepEqual(mock.writeTextures[0].size, { width: 16, height: 8, depthOrArrayLayers: 2 });
    assert.equal(mock.writeTextures[0].dataLayout.bytesPerRow, 256);
    assert.equal(mock.writeTextures[0].dataLayout.rowsPerImage, 8);
    assert.equal(mock.writeTextures[0].data.byteLength, 256 * 8 * 2);
    assert.ok(mock.writeTextures[0].data.every((b) => b === 0));
    assert.deepEqual(mock.writeTextures[2].size, { width: 4, height: 2, depthOrArrayLayers: 2 });
    assert.ok(mock.writeTextures[2].data.every((b) => b === 0));
    assert.equal(proc.getValidationErrors().length, 0);
  }
});

test('ClearTexture clears 3D texture with per-mip halving of width, height, and depth', () => {
  const mock = createMockWebGpu({ supportCopyBufferToTexture: true });
  const proc = new WgpuCommandProcessor(mock.device, mock.ctx);
  installRenderStatePackets(proc);

  const stream = new StreamBuilder()
    .createTexture2({
      id: 30,
      width: 8,
      height: 8,
      depthOrLayers: 4,
      mipLevels: 3,
      dimension: TEXTURE_DIMENSION.DIM_3D,
      format: 'rgba8unorm',
      usage: TEXTURE_USAGE.STORAGE_BINDING | TEXTURE_USAGE.COPY_DST,
    })
    .clearTexture(30)
    .finish();

  proc.executeCommandBuffer(stream);

  assert.equal(mock.renderPasses.length, 0);
  assert.equal(mock.copyBufferToTextures.length, 3);
  assert.deepEqual(mock.copyBufferToTextures[0].copySize, { width: 8, height: 8, depthOrArrayLayers: 4 });
  assert.deepEqual(mock.copyBufferToTextures[1].copySize, { width: 4, height: 4, depthOrArrayLayers: 2 });
  assert.deepEqual(mock.copyBufferToTextures[2].copySize, { width: 2, height: 2, depthOrArrayLayers: 1 });
  for (const copy of mock.copyBufferToTextures) {
    assert.equal(copy.src.bytesPerRow % 256, 0);
    assert.equal(copy.src.rowsPerImage, copy.copySize.height);
  }
  assert.equal(proc.getValidationErrors().length, 0);
});

test('ClearTexture clears BC7 block-compressed texture with 4x4 block rounding and 256-byte row alignment', () => {
  const mock = createMockWebGpu({
    features: ['texture-compression-bc'],
    supportCopyBufferToTexture: true,
  });
  const proc = new WgpuCommandProcessor(mock.device, mock.ctx);
  installRenderStatePackets(proc);

  const stream = new StreamBuilder()
    // 12x12 with 4 mips -> virtual 12x12, 6x6, 3x3, 1x1 -> physical 12x12, 8x8, 4x4, 4x4.
    .createTexture2({
      id: 40,
      width: 12,
      height: 12,
      depthOrLayers: 1,
      mipLevels: 4,
      dimension: TEXTURE_DIMENSION.DIM_2D,
      format: 'bc7-rgba-unorm',
      usage: TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
    })
    // 132x8 BC7 -> 33 blocks wide * 16B = 528B tight row -> 768B (256-byte aligned).
    .createTexture2({
      id: 41,
      width: 132,
      height: 8,
      depthOrLayers: 1,
      mipLevels: 1,
      dimension: TEXTURE_DIMENSION.DIM_2D,
      format: 'bc7-rgba-unorm',
      usage: TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
    })
    .clearTexture(40)
    .clearTexture(41)
    .finish();

  proc.executeCommandBuffer(stream);

  assert.equal(mock.renderPasses.length, 0);
  assert.equal(mock.copyBufferToTextures.length, 5);
  assert.equal(proc.getValidationErrors().length, 0);

  // Texture 40: 4 mips
  const expectedMips = [
    { mip: 0, width: 12, height: 12, rowsPerImage: 3, bytesPerRow: 256 },
    { mip: 1, width: 8, height: 8, rowsPerImage: 2, bytesPerRow: 256 },
    { mip: 2, width: 4, height: 4, rowsPerImage: 1, bytesPerRow: 256 },
    { mip: 3, width: 4, height: 4, rowsPerImage: 1, bytesPerRow: 256 },
  ];
  for (let i = 0; i < expectedMips.length; i += 1) {
    const c = mock.copyBufferToTextures[i];
    const exp = expectedMips[i];
    assert.equal(c.dst.mipLevel, exp.mip);
    assert.deepEqual(c.copySize, { width: exp.width, height: exp.height, depthOrArrayLayers: 1 });
    assert.equal(c.src.rowsPerImage, exp.rowsPerImage);
    assert.equal(c.src.bytesPerRow, exp.bytesPerRow);
  }

  // Texture 41: 132x8 BC7 -> 33 * 16 = 528 -> 768 bytesPerRow, 2 block rows.
  const wide = mock.copyBufferToTextures[4];
  assert.deepEqual(wide.copySize, { width: 132, height: 8, depthOrArrayLayers: 1 });
  assert.equal(wide.src.bytesPerRow, 768);
  assert.equal(wide.src.rowsPerImage, 2);

  // Zero GPUBuffer was reused across both textures (only 1 buffer created).
  assert.equal(mock.createdBuffers.length, 1);
});

test('ClearTexture gracefully no-ops on unknown or destroyed texture ids without throwing or recording errors', () => {
  const mock = createMockWebGpu();
  const proc = new WgpuCommandProcessor(mock.device, mock.ctx);
  const rs = installRenderStatePackets(proc);

  const stream = new StreamBuilder()
    .clearTexture(9999) // unknown
    .createTexture(50, 4, 4)
    .destroyResource(50)
    .clearTexture(50) // destroyed via DestroyResource
    .createTexture(51, 4, 4)
    .finish();

  proc.executeCommandBuffer(stream);
  // Mark live texture object as destroyed directly.
  proc.textures.get(51).destroyed = true;
  proc.executeCommandBuffer(new StreamBuilder().clearTexture(51).finish());

  assert.equal(mock.submits.length, 0);
  assert.equal(mock.writeTextures.length, 0);
  assert.equal(proc.getValidationErrors().length, 0);
  assert.equal(rs.stats.errors, 0);
  assert.equal(rs.telemetry.textureClearsMissing, 3);
  assert.equal(proc.getStats().clearTextureStats.missing, 3);
});

test('ClearTexture rejects depth/stencil, multisampled, non-COPY_DST, and truncated payloads gracefully without throwing', () => {
  const mock = createMockWebGpu();
  const proc = new WgpuCommandProcessor(mock.device, mock.ctx);
  installRenderStatePackets(proc);

  const stream = new StreamBuilder()
    .createTexture2({
      id: 60,
      width: 4,
      height: 4,
      format: 'depth32float',
      usage: TEXTURE_USAGE.COPY_DST | TEXTURE_USAGE.TEXTURE_BINDING,
    })
    .createTexture2({
      id: 61,
      width: 4,
      height: 4,
      sampleCount: 4,
      format: 'rgba8unorm',
      usage: TEXTURE_USAGE.RENDER_ATTACHMENT,
    })
    .createTexture2({
      id: 62,
      width: 4,
      height: 4,
      format: 'rgba8unorm',
      usage: TEXTURE_USAGE.TEXTURE_BINDING,
    })
    .clearTexture(60)
    .clearTexture(61)
    .clearTexture(62)
    .packet(OPCODE_CLEAR_TEXTURE, new Uint8Array([1, 2, 3, 4])) // truncated 4B < 8B
    .finish();

  proc.executeCommandBuffer(stream);

  assert.equal(mock.submits.length, 0);
  assert.equal(mock.writeTextures.length, 0);
  assert.equal(proc.getValidationErrors().length, 4);
  assert.equal(proc.getStats().clearTextureStats.invalid, 3);
});

test('Multi-packet stream continues cleanly after opcode 63 and preserves WriteTexture -> ClearTexture -> WriteTexture -> Present ordering', () => {
  const mock = createMockWebGpu({ supportCopyBufferToTexture: true });
  const proc = new WgpuCommandProcessor(mock.device, mock.ctx);
  installRenderStatePackets(proc);

  const beforeBytes = new Uint8Array(256 * 4).fill(0x55);
  const afterBytes = new Uint8Array(256 * 1).fill(0x07);

  const stream = new StreamBuilder()
    .beginFrame()
    .clearColor(0.1, 0.2, 0.3, 1.0)
    .createTexture(70, 4, 4, 1, 2, TEXTURE_FORMAT.RGBA8_UNORM, TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST)
    .writeTexture({
      textureId: 70,
      mipLevel: 0,
      width: 4,
      height: 4,
      depthOrLayers: 1,
      bytesPerRow: 256,
      rowsPerImage: 4,
      bytes: beforeBytes,
    })
    .clearTexture(70)
    .writeTexture({
      textureId: 70,
      mipLevel: 0,
      width: 4,
      height: 1,
      depthOrLayers: 1,
      bytesPerRow: 256,
      rowsPerImage: 1,
      bytes: afterBytes,
    })
    .present()
    .finish();

  const res = proc.executeCommandBuffer(stream);
  assert.equal(res.presents, 1);
  assert.equal(res.frameCount, 1);
  assert.equal(proc.getValidationErrors().length, 0);

  // Verify exact queue timeline order:
  // 1) writeTexture(0x55) -> 2) submit(copyBufferToTexture clear) -> 3) writeTexture(0x07) -> 4) submit(Present)
  const timeline = mock.events.filter(
    (e) => e.op === 'queue.writeTexture' || e.op === 'queue.submit'
  );
  assert.equal(timeline.length, 4);
  assert.equal(timeline[0].op, 'queue.writeTexture');
  assert.equal(timeline[0].firstByte, 0x55);
  assert.equal(timeline[1].op, 'queue.submit');
  assert.deepEqual(timeline[1].ops, ['copyBufferToTexture', 'copyBufferToTexture']);
  assert.equal(timeline[2].op, 'queue.writeTexture');
  assert.equal(timeline[2].firstByte, 0x07);
  assert.equal(timeline[3].op, 'queue.submit');
  assert.ok(timeline[3].ops.includes('beginRenderPass'));

  // ClearTexture must not add extra entries to the device-loss replay log (only CreateTexture id 70).
  assert.equal(proc.getResourceReplayStats().recordedResourceCount, 1);
});

test('ClearTexture chunks large textures into bounded <= 4 MiB slices and reuses zero buffer without leaking', () => {
  const plan = planClearTextureCopies(
    {
      width: 2048,
      height: 2048,
      depthOrArrayLayers: 1,
      mipLevelCount: 1,
      dimension: '2d',
      blockW: 1,
      blockH: 1,
      blockBytes: 16, // rgba32float -> 32768 B/row -> 64 MiB slice
    },
    4 * 1024 * 1024
  );

  assert.equal(plan.copies.length, 16);
  assert.equal(plan.maxRequiredBytes, 4 * 1024 * 1024);
  for (let i = 0; i < plan.copies.length; i += 1) {
    assert.equal(plan.copies[i].y, i * 128);
    assert.equal(plan.copies[i].height, 128);
    assert.equal(plan.copies[i].bytesPerRow, 32768);
    assert.equal(plan.copies[i].rowsPerImage, 128);
    assert.equal(plan.copies[i].copyBytes, 4 * 1024 * 1024);
  }

  // Verify destroy() cleans up the cached zero GPUBuffer and scratch Uint8Array.
  const mock = createMockWebGpu({ supportCopyBufferToTexture: true });
  const proc = new WgpuCommandProcessor(mock.device, mock.ctx);
  proc.executeCommandBuffer(
    new StreamBuilder()
      .createTexture(80, 16, 16, 1, 1, TEXTURE_FORMAT.RGBA8_UNORM, TEXTURE_USAGE.COPY_DST)
      .clearTexture(80)
      .clearTexture(80)
      .clearTexture(80)
      .finish()
  );
  assert.equal(mock.createdBuffers.length, 1);
  assert.equal(mock.createdBuffers[0].destroyed, false);
  proc.destroy();
  assert.equal(mock.createdBuffers[0].destroyed, true);
  assert.equal(proc._clearZeroBuffer, null);
  assert.equal(proc._clearZeroScratch, null);
});

test('ClearTexture flushes staged pending uploads before clearing and handles non-promoted 1D textures', () => {
  const mock = createMockWebGpu({ supportCopyBufferToTexture: true });
  let stagedUploads = 0;
  const mockStagingPool = {
    _destroyed: false,
    stageTextureUpload(encoder, texture, data, layout) {
      stagedUploads += 1;
      encoder.copyBufferToTexture(
        { buffer: { kind: 'StagingBuffer' }, offset: 0, bytesPerRow: 256, rowsPerImage: layout.height },
        { texture, mipLevel: layout.mipLevel, origin: { x: 0, y: 0, z: 0 }, aspect: 'all' },
        { width: layout.width, height: layout.height, depthOrArrayLayers: layout.depthOrArrayLayers }
      );
    },
    prepareSubmit() {},
    endFrame() {},
  };

  const proc = new WgpuCommandProcessor(mock.device, mock.ctx, {
    stagingPool: mockStagingPool,
    promote1DTo2D: false,
  });

  const stream = new StreamBuilder()
    .createTexture(90, 8, 1, 1, 1, TEXTURE_FORMAT.RGBA8_UNORM, TEXTURE_USAGE.COPY_DST, TEXTURE_DIMENSION.DIM_1D)
    .writeTexture({
      textureId: 90,
      width: 8,
      height: 1,
      depthOrLayers: 1,
      bytesPerRow: 256,
      rowsPerImage: 1,
      bytes: new Uint8Array(256).fill(0xaa),
    })
    .clearTexture(90)
    .finish();

  proc.executeCommandBuffer(stream);

  assert.equal(stagedUploads, 1);
  // First submit = flushed staged WriteTexture; second submit = ClearTexture copyBufferToTexture.
  assert.equal(mock.submits.length, 2);
  assert.equal(mock.copyBufferToTextures.length, 2);
  assert.equal(mock.copyBufferToTextures[0].src.buffer.kind, 'StagingBuffer');
  assert.equal(mock.copyBufferToTextures[1].src.buffer.kind, 'GPUBuffer');
  assert.deepEqual(mock.copyBufferToTextures[1].copySize, { width: 8, height: 1, depthOrArrayLayers: 1 });
  assert.equal(mock.copyBufferToTextures[1].src.bytesPerRow, 256);
  assert.equal(proc.getValidationErrors().length, 0);
});
