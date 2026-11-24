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
const {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  buildGpuTelemetrySnapshot,
  selectSupportedAdapterFeatures,
  OPTIONAL_ADAPTER_FEATURES,
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
  INDEX_FORMAT,
} = wgpuMod;

class BinaryCommandBuilder {
  constructor() {
    this.chunks = [];
  }

  _pushPacket(opcode, payloadBytes) {
    const buf = new ArrayBuffer(8 + payloadBytes.byteLength);
    const view = new DataView(buf);
    view.setUint16(0, opcode, true);
    view.setUint16(2, 0, true);
    view.setUint32(4, payloadBytes.byteLength, true);
    new Uint8Array(buf, 8).set(payloadBytes);
    this.chunks.push(new Uint8Array(buf));
  }

  nop() {
    this._pushPacket(OPCODES.NOP, new Uint8Array(0));
    return this;
  }

  beginFrame() {
    this._pushPacket(OPCODES.BEGIN_FRAME, new Uint8Array(0));
    return this;
  }

  clearColor(r, g, b, a) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setFloat32(0, r, true);
    view.setFloat32(4, g, true);
    view.setFloat32(8, b, true);
    view.setFloat32(12, a, true);
    this._pushPacket(OPCODES.CLEAR_COLOR, new Uint8Array(buf));
    return this;
  }

  createBuffer(id, usageFlags, byteSize) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, usageFlags, true);
    view.setBigUint64(8, BigInt(byteSize), true);
    this._pushPacket(OPCODES.CREATE_BUFFER, new Uint8Array(buf));
    return this;
  }

  writeBuffer(id, dstOffset, bytes) {
    const src = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    const buf = new ArrayBuffer(16 + src.byteLength);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, dstOffset, true);
    view.setUint32(8, src.byteLength, true);
    view.setUint32(12, 0, true);
    new Uint8Array(buf, 16).set(src);
    this._pushPacket(OPCODES.WRITE_BUFFER, new Uint8Array(buf));
    return this;
  }

  createShaderModule(id, stage, wgslCode) {
    const encoded = new TextEncoder().encode(wgslCode);
    const buf = new ArrayBuffer(16 + encoded.byteLength);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, stage, true);
    view.setUint32(8, encoded.byteLength, true);
    view.setUint32(12, 0, true);
    new Uint8Array(buf, 16).set(encoded);
    this._pushPacket(OPCODES.CREATE_SHADER_MODULE, new Uint8Array(buf));
    return this;
  }

  createRenderPipeline(id, vsShaderId, fsShaderId, vertexStride, topology = PRIMITIVE_TOPOLOGY.TRIANGLE_LIST, flags = 0) {
    const buf = new ArrayBuffer(24);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, vsShaderId, true);
    view.setUint32(8, fsShaderId, true);
    view.setUint32(12, vertexStride, true);
    view.setUint32(16, topology, true);
    view.setUint32(20, flags, true);
    this._pushPacket(OPCODES.CREATE_RENDER_PIPELINE, new Uint8Array(buf));
    return this;
  }

  setPipeline(pipelineId) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, pipelineId, true);
    this._pushPacket(OPCODES.SET_PIPELINE, new Uint8Array(buf));
    return this;
  }

  setVertexBuffer(slot, bufferId, offset = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, slot, true);
    view.setUint32(4, bufferId, true);
    view.setBigUint64(8, BigInt(offset), true);
    this._pushPacket(OPCODES.SET_VERTEX_BUFFER, new Uint8Array(buf));
    return this;
  }

  draw(vertexCount, instanceCount = 1, firstVertex = 0, firstInstance = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, vertexCount, true);
    view.setUint32(4, instanceCount, true);
    view.setUint32(8, firstVertex, true);
    view.setUint32(12, firstInstance, true);
    this._pushPacket(OPCODES.DRAW, new Uint8Array(buf));
    return this;
  }

  present() {
    this._pushPacket(OPCODES.PRESENT, new Uint8Array(0));
    return this;
  }

  destroyResource(id) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, id, true);
    this._pushPacket(OPCODES.DESTROY_RESOURCE, new Uint8Array(buf));
    return this;
  }

  createTexture(
    id,
    width,
    height,
    depthOrLayers = 1,
    mipLevels = 1,
    format = TEXTURE_FORMAT.RGBA8_UNORM,
    usageFlags = TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
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
    view.setUint32(24, usageFlags, true);
    view.setUint32(28, dimension, true);
    this._pushPacket(OPCODES.CREATE_TEXTURE, new Uint8Array(buf));
    return this;
  }

  writeTexture(
    textureId,
    mipLevel,
    originX,
    originY,
    originZ,
    width,
    height,
    depthOrLayers,
    bytesPerRow,
    rowsPerImage,
    bytes
  ) {
    const src = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    const buf = new ArrayBuffer(48 + src.byteLength);
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
    view.setUint32(40, src.byteLength, true);
    view.setUint32(44, 0, true);
    new Uint8Array(buf, 48).set(src);
    this._pushPacket(OPCODES.WRITE_TEXTURE, new Uint8Array(buf));
    return this;
  }

  createSampler(
    id,
    addressU = ADDRESS_MODE.CLAMP_TO_EDGE,
    addressV = ADDRESS_MODE.CLAMP_TO_EDGE,
    addressW = ADDRESS_MODE.CLAMP_TO_EDGE,
    magFilter = FILTER_MODE.LINEAR,
    minFilter = FILTER_MODE.LINEAR,
    mipmapFilter = FILTER_MODE.LINEAR,
    maxAnisotropy = 1
  ) {
    const buf = new ArrayBuffer(32);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, addressU, true);
    view.setUint32(8, addressV, true);
    view.setUint32(12, addressW, true);
    view.setUint32(16, magFilter, true);
    view.setUint32(20, minFilter, true);
    view.setUint32(24, mipmapFilter, true);
    view.setUint32(28, maxAnisotropy, true);
    this._pushPacket(OPCODES.CREATE_SAMPLER, new Uint8Array(buf));
    return this;
  }

  createBindGroup(id, pipelineId, groupIndex, entries = []) {
    const buf = new ArrayBuffer(16 + entries.length * 32);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, pipelineId, true);
    view.setUint32(8, groupIndex, true);
    view.setUint32(12, entries.length, true);
    for (let i = 0; i < entries.length; i++) {
      const e = entries[i];
      const off = 16 + i * 32;
      view.setUint32(off, e.binding, true);
      view.setUint32(off + 4, e.resourceType, true);
      view.setUint32(off + 8, e.resourceId, true);
      view.setUint32(off + 12, 0, true);
      view.setBigUint64(off + 16, BigInt(e.offset || 0), true);
      view.setBigUint64(off + 24, BigInt(e.size || 0), true);
    }
    this._pushPacket(OPCODES.CREATE_BIND_GROUP, new Uint8Array(buf));
    return this;
  }

  setBindGroup(groupIndex, bindGroupId) {
    const buf = new ArrayBuffer(8);
    const view = new DataView(buf);
    view.setUint32(0, groupIndex, true);
    view.setUint32(4, bindGroupId, true);
    this._pushPacket(OPCODES.SET_BIND_GROUP, new Uint8Array(buf));
    return this;
  }

  setIndexBuffer(bufferId, indexFormat = INDEX_FORMAT.UINT16, offset = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, bufferId, true);
    view.setUint32(4, indexFormat, true);
    view.setBigUint64(8, BigInt(offset), true);
    this._pushPacket(OPCODES.SET_INDEX_BUFFER, new Uint8Array(buf));
    return this;
  }

  drawIndexed(indexCount, instanceCount = 1, firstIndex = 0, baseVertex = 0, firstInstance = 0) {
    const buf = new ArrayBuffer(20);
    const view = new DataView(buf);
    view.setUint32(0, indexCount, true);
    view.setUint32(4, instanceCount, true);
    view.setUint32(8, firstIndex, true);
    view.setInt32(12, baseVertex, true);
    view.setUint32(16, firstInstance, true);
    this._pushPacket(OPCODES.DRAW_INDEXED, new Uint8Array(buf));
    return this;
  }

  createComputePipeline(id, csShaderId, flags = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, csShaderId, true);
    view.setUint32(8, flags, true);
    view.setUint32(12, 0, true);
    this._pushPacket(OPCODES.CREATE_COMPUTE_PIPELINE, new Uint8Array(buf));
    return this;
  }

  setComputePipeline(pipelineId) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, pipelineId, true);
    this._pushPacket(OPCODES.SET_COMPUTE_PIPELINE, new Uint8Array(buf));
    return this;
  }

  dispatchCompute(workgroupX, workgroupY = 1, workgroupZ = 1) {
    const buf = new ArrayBuffer(12);
    const view = new DataView(buf);
    view.setUint32(0, workgroupX, true);
    view.setUint32(4, workgroupY, true);
    view.setUint32(8, workgroupZ, true);
    this._pushPacket(OPCODES.DISPATCH_COMPUTE, new Uint8Array(buf));
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
  const calls = [];
  let nextHandle = 1;
  const mockTextureView = { kind: 'GPUTextureView', id: 'swapchain-view-0' };

  const ctx = {
    getCurrentTexture() {
      return {
        createView() {
          return mockTextureView;
        },
      };
    },
  };

  const device = {
    createBuffer(desc) {
      const buffer = {
        kind: 'GPUBuffer',
        handle: nextHandle++,
        size: desc.size,
        usage: desc.usage,
        destroyed: false,
        destroy() {
          this.destroyed = true;
          calls.push({ op: 'buffer.destroy', buffer });
        },
      };
      calls.push({ op: 'createBuffer', desc: { ...desc }, result: buffer });
      return buffer;
    },

    createShaderModule(desc) {
      const module = {
        kind: 'GPUShaderModule',
        handle: nextHandle++,
        code: desc.code,
      };
      calls.push({ op: 'createShaderModule', desc: { ...desc }, result: module });
      return module;
    },

    createRenderPipeline(desc) {
      const pipeline = {
        kind: 'GPURenderPipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayout(index) {
          const layout = { kind: 'GPUBindGroupLayout', pipelineHandle: this.handle, index };
          calls.push({ op: 'pipeline.getBindGroupLayout', pipeline: this, index, result: layout });
          return layout;
        },
      };
      calls.push({ op: 'createRenderPipeline', desc, result: pipeline });
      return pipeline;
    },

    createComputePipeline(desc) {
      const pipeline = {
        kind: 'GPUComputePipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayout(index) {
          const layout = { kind: 'GPUBindGroupLayout', pipelineHandle: this.handle, index };
          calls.push({ op: 'computePipeline.getBindGroupLayout', pipeline: this, index, result: layout });
          return layout;
        },
      };
      calls.push({ op: 'createComputePipeline', desc, result: pipeline });
      return pipeline;
    },

    createTexture(desc) {
      const textureHandle = nextHandle++;
      const texture = {
        kind: 'GPUTexture',
        handle: textureHandle,
        desc: { ...desc, size: { ...desc.size } },
        destroyed: false,
        createView() {
          const view = { kind: 'GPUTextureView', textureHandle, handle: nextHandle++ };
          calls.push({ op: 'texture.createView', texture, result: view });
          return view;
        },
        destroy() {
          this.destroyed = true;
          calls.push({ op: 'texture.destroy', texture });
        },
      };
      calls.push({ op: 'createTexture', desc: { ...desc, size: { ...desc.size } }, result: texture });
      return texture;
    },

    createSampler(desc) {
      const sampler = {
        kind: 'GPUSampler',
        handle: nextHandle++,
        desc: { ...desc },
      };
      calls.push({ op: 'createSampler', desc: { ...desc }, result: sampler });
      return sampler;
    },

    createBindGroup(desc) {
      const bindGroup = {
        kind: 'GPUBindGroup',
        handle: nextHandle++,
        desc,
      };
      calls.push({ op: 'createBindGroup', desc, result: bindGroup });
      return bindGroup;
    },

    createCommandEncoder() {
      const commandBuffer = { kind: 'GPUCommandBuffer', handle: nextHandle++ };
      return {
        beginRenderPass(desc) {
          calls.push({ op: 'beginRenderPass', desc });
          return {
            setPipeline(pipeline) {
              calls.push({ op: 'setPipeline', pipeline });
            },
            setBindGroup(index, bindGroup) {
              calls.push({ op: 'renderPass.setBindGroup', index, bindGroup });
            },
            setVertexBuffer(slot, buffer, offset) {
              calls.push({ op: 'setVertexBuffer', slot, buffer, offset });
            },
            setIndexBuffer(buffer, indexFormat, offset) {
              calls.push({ op: 'setIndexBuffer', buffer, indexFormat, offset });
            },
            draw(vertexCount, instanceCount, firstVertex, firstInstance) {
              calls.push({
                op: 'draw',
                vertexCount,
                instanceCount,
                firstVertex,
                firstInstance,
              });
            },
            drawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance) {
              calls.push({
                op: 'drawIndexed',
                indexCount,
                instanceCount,
                firstIndex,
                baseVertex,
                firstInstance,
              });
            },
            end() {
              calls.push({ op: 'renderPass.end' });
            },
          };
        },
        beginComputePass() {
          calls.push({ op: 'beginComputePass' });
          return {
            setPipeline(pipeline) {
              calls.push({ op: 'computePass.setPipeline', pipeline });
            },
            setBindGroup(index, bindGroup) {
              calls.push({ op: 'computePass.setBindGroup', index, bindGroup });
            },
            dispatchWorkgroups(x, y, z) {
              calls.push({ op: 'dispatchWorkgroups', x, y, z });
            },
            end() {
              calls.push({ op: 'computePass.end' });
            },
          };
        },
        finish() {
          calls.push({ op: 'encoder.finish', result: commandBuffer });
          return commandBuffer;
        },
      };
    },

    queue: {
      writeBuffer(buffer, dstOffset, data) {
        calls.push({
          op: 'writeBuffer',
          buffer,
          dstOffset,
          data: new Uint8Array(data),
        });
      },
      writeTexture(destination, data, dataLayout, size) {
        calls.push({
          op: 'writeTexture',
          destination: { ...destination, origin: { ...destination.origin } },
          data: new Uint8Array(data),
          dataLayout: { ...dataLayout },
          size: { ...size },
        });
      },
      submit(commandBuffers) {
        calls.push({ op: 'submit', commandBuffers });
      },
    },
  };

  return { device, ctx, calls, mockTextureView };
}

test('UMD export attaches PlayGta6Wgpu on globalThis', () => {
  assert.equal(globalThis.PlayGta6Wgpu.WgpuCommandProcessor, WgpuCommandProcessor);
  assert.equal(globalThis.PlayGta6Wgpu.OPCODES.CREATE_RENDER_PIPELINE, 8);
  assert.equal(globalThis.PlayGta6Wgpu.OPCODES.DESTROY_RESOURCE, 11);
});

test('full binary command stream drives every WebGPU device, render-pass, and destroy call', () => {
  const { device, ctx, calls, mockTextureView } = createMockWebGpu();
  const processor = new WgpuCommandProcessor(device, ctx, { format: 'bgra8unorm' });

  const vertexFloats = new Float32Array([
    0.0, 0.5, 0.0,
    -0.5, -0.5, 0.0,
    0.5, -0.5, 0.0,
  ]);
  const vertexBytes = new Uint8Array(vertexFloats.buffer);
  const vsWgsl = '@vertex fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(pos, 1.0); }';
  const fsWgsl = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.5, 0.0, 1.0); }';

  const stream = new BinaryCommandBuilder()
    .beginFrame()
    .createBuffer(1, BUFFER_USAGE.VERTEX | BUFFER_USAGE.COPY_DST, vertexBytes.byteLength)
    .writeBuffer(1, 0, vertexBytes)
    .createShaderModule(2, SHADER_STAGE.VERTEX, vsWgsl)
    .createShaderModule(3, SHADER_STAGE.FRAGMENT, fsWgsl)
    .createRenderPipeline(4, 2, 3, 12, PRIMITIVE_TOPOLOGY.TRIANGLE_LIST, 0)
    .clearColor(0.25, 0.5, 0.75, 1.0)
    .setPipeline(4)
    .setVertexBuffer(0, 1, 0)
    .draw(3, 1, 0, 0)
    .present()
    .destroyResource(1)
    .finish();

  const result = processor.execute(stream);
  assert.deepEqual(result, { frameCount: 1, presents: 1 });
  assert.equal(processor.frameCount, 1);

  assert.deepEqual(
    calls.map((c) => c.op),
    [
      'createBuffer',
      'writeBuffer',
      'createShaderModule',
      'createShaderModule',
      'createRenderPipeline',
      'beginRenderPass',
      'setPipeline',
      'setVertexBuffer',
      'draw',
      'renderPass.end',
      'encoder.finish',
      'submit',
      'buffer.destroy',
    ],
  );

  // 1. createBuffer
  const createBufCall = calls[0];
  assert.deepEqual(createBufCall.desc, {
    size: vertexBytes.byteLength,
    usage: BUFFER_USAGE.VERTEX | BUFFER_USAGE.COPY_DST,
  });
  const gpuBuffer = createBufCall.result;

  // 2. writeBuffer
  const writeBufCall = calls[1];
  assert.equal(writeBufCall.buffer, gpuBuffer);
  assert.equal(writeBufCall.dstOffset, 0);
  assert.deepEqual(writeBufCall.data, vertexBytes);

  // 3 & 4. createShaderModule (VS & FS)
  const vsCall = calls[2];
  const fsCall = calls[3];
  assert.equal(vsCall.desc.code, vsWgsl);
  assert.equal(fsCall.desc.code, fsWgsl);

  // 5. createRenderPipeline
  const pipelineCall = calls[4];
  assert.equal(pipelineCall.desc.layout, 'auto');
  assert.equal(pipelineCall.desc.vertex.module, vsCall.result);
  assert.equal(pipelineCall.desc.vertex.entryPoint, 'vs_main');
  assert.deepEqual(pipelineCall.desc.vertex.buffers, [
    {
      arrayStride: 12,
      attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
    },
  ]);
  assert.equal(pipelineCall.desc.fragment.module, fsCall.result);
  assert.equal(pipelineCall.desc.fragment.entryPoint, 'fs_main');
  assert.deepEqual(pipelineCall.desc.fragment.targets, [{ format: 'bgra8unorm' }]);
  assert.deepEqual(pipelineCall.desc.primitive, { topology: 'triangle-list' });
  const gpuPipeline = pipelineCall.result;

  // 6. beginRenderPass
  const passCall = calls[5];
  assert.deepEqual(passCall.desc, {
    colorAttachments: [
      {
        view: mockTextureView,
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0.25, g: 0.5, b: 0.75, a: 1.0 },
      },
    ],
  });

  // 7. setPipeline
  assert.equal(calls[6].pipeline, gpuPipeline);

  // 8. setVertexBuffer
  assert.equal(calls[7].slot, 0);
  assert.equal(calls[7].buffer, gpuBuffer);
  assert.equal(calls[7].offset, 0);

  // 9. draw
  assert.deepEqual(calls[8], {
    op: 'draw',
    vertexCount: 3,
    instanceCount: 1,
    firstVertex: 0,
    firstInstance: 0,
  });

  // 11. submit
  assert.deepEqual(calls[11].commandBuffers, [calls[10].result]);

  // 12. destroyResource after present
  assert.equal(calls[12].buffer, gpuBuffer);
  assert.equal(gpuBuffer.destroyed, true);
  assert.equal(processor.buffers.has(1), false);
  assert.equal(processor.shaderModules.has(2), true);
  assert.equal(processor.pipelines.has(4), true);
});

test('DestroyResource safely handles shaders, pipelines, missing IDs, and buffers without destroy()', () => {
  const { device, ctx, calls } = createMockWebGpu();
  const processor = new WgpuCommandProcessor(device, ctx);

  // Inject a buffer object without a .destroy method to confirm safe cleanup.
  processor.buffers.set(99, { kind: 'GPUBufferNoDestroy' });

  const stream = new BinaryCommandBuilder()
    .nop()
    .createShaderModule(10, SHADER_STAGE.VERTEX, 'fn vs_main() {}')
    .createRenderPipeline(11, 10, 0, 0, PRIMITIVE_TOPOLOGY.LINE_STRIP, 0)
    .destroyResource(99)
    .destroyResource(10)
    .destroyResource(11)
    .destroyResource(9999)
    .finish();

  const res = processor.execute(stream);
  assert.deepEqual(res, { frameCount: 0, presents: 0 });
  assert.equal(processor.buffers.size, 0);
  assert.equal(processor.shaderModules.size, 0);
  assert.equal(processor.pipelines.size, 0);
  assert.equal(calls[1].desc.primitive.topology, 'line-strip');
  assert.deepEqual(calls[1].desc.vertex.buffers, []);
  assert.equal(calls[1].desc.fragment, undefined);
});

test('G5 opcodes drive textures, samplers, bind groups, indexed draw, and compute dispatch', () => {
  const { device, ctx, calls, mockTextureView } = createMockWebGpu();
  const processor = new WgpuCommandProcessor(device, ctx, { format: 'bgra8unorm' });

  assert.equal(OPCODES.CREATE_TEXTURE, 12);
  assert.equal(OPCODES.WRITE_TEXTURE, 13);
  assert.equal(OPCODES.CREATE_SAMPLER, 14);
  assert.equal(OPCODES.CREATE_BIND_GROUP, 15);
  assert.equal(OPCODES.SET_BIND_GROUP, 16);
  assert.equal(OPCODES.SET_INDEX_BUFFER, 17);
  assert.equal(OPCODES.DRAW_INDEXED, 18);
  assert.equal(OPCODES.CREATE_COMPUTE_PIPELINE, 19);
  assert.equal(OPCODES.SET_COMPUTE_PIPELINE, 20);
  assert.equal(OPCODES.DISPATCH_COMPUTE, 21);

  const texels = new Uint8Array([
    255, 0, 0, 255,   0, 255, 0, 255,
    0, 0, 255, 255,   255, 255, 255, 255,
  ]);
  const indices = new Uint16Array([0, 1, 2, 2, 3, 0]);
  const indexBytes = new Uint8Array(indices.buffer);

  const stream = new BinaryCommandBuilder()
    .beginFrame()
    .createBuffer(1, BUFFER_USAGE.STORAGE | BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST, 256)
    .createBuffer(2, BUFFER_USAGE.VERTEX | BUFFER_USAGE.COPY_DST, 64)
    .createBuffer(3, BUFFER_USAGE.INDEX | BUFFER_USAGE.COPY_DST, indexBytes.byteLength)
    .writeBuffer(3, 0, indexBytes)
    .createTexture(
      4,
      2,
      2,
      1,
      1,
      TEXTURE_FORMAT.RGBA8_UNORM,
      TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
      TEXTURE_DIMENSION.DIM_2D
    )
    .writeTexture(4, 0, 0, 0, 0, 2, 2, 1, 8, 2, texels)
    .createSampler(
      5,
      ADDRESS_MODE.REPEAT,
      ADDRESS_MODE.MIRROR_REPEAT,
      ADDRESS_MODE.CLAMP_TO_EDGE,
      FILTER_MODE.LINEAR,
      FILTER_MODE.LINEAR,
      FILTER_MODE.NEAREST,
      8
    )
    .createShaderModule(6, SHADER_STAGE.COMPUTE, '@compute @workgroup_size(8) fn cs_main() {}')
    .createShaderModule(7, SHADER_STAGE.VERTEX, '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }')
    .createShaderModule(8, SHADER_STAGE.FRAGMENT, '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }')
    .createComputePipeline(9, 6, 0)
    .createRenderPipeline(10, 7, 8, 16, PRIMITIVE_TOPOLOGY.TRIANGLE_LIST, 0)
    .createBindGroup(11, 9, 0, [
      { binding: 0, resourceType: BIND_RESOURCE_TYPE.BUFFER, resourceId: 1, offset: 64, size: 128 },
    ])
    .createBindGroup(12, 10, 0, [
      { binding: 0, resourceType: BIND_RESOURCE_TYPE.BUFFER, resourceId: 1, offset: 0, size: 0 },
      { binding: 1, resourceType: BIND_RESOURCE_TYPE.TEXTURE_VIEW, resourceId: 4 },
      { binding: 2, resourceType: BIND_RESOURCE_TYPE.SAMPLER, resourceId: 5 },
    ])
    .clearColor(0.25, 0.5, 0.75, 1.0)
    .setComputePipeline(9)
    .setBindGroup(0, 11)
    .dispatchCompute(4, 2, 1)
    .setPipeline(10)
    .setBindGroup(0, 12)
    .setVertexBuffer(0, 2, 0)
    .setIndexBuffer(3, INDEX_FORMAT.UINT16, 0)
    .drawIndexed(6, 2, 0, -4, 1)
    .present()
    .finish();

  const res = processor.execute(stream);
  assert.deepEqual(res, { frameCount: 1, presents: 1 });

  assert.deepEqual(
    calls.map((c) => c.op),
    [
      'createBuffer',
      'createBuffer',
      'createBuffer',
      'writeBuffer',
      'createTexture',
      'texture.createView',
      'writeTexture',
      'createSampler',
      'createShaderModule',
      'createShaderModule',
      'createShaderModule',
      'createComputePipeline',
      'createRenderPipeline',
      'computePipeline.getBindGroupLayout',
      'createBindGroup',
      'pipeline.getBindGroupLayout',
      'createBindGroup',
      'beginComputePass',
      'computePass.setPipeline',
      'computePass.setBindGroup',
      'dispatchWorkgroups',
      'computePass.end',
      'beginRenderPass',
      'setPipeline',
      'renderPass.setBindGroup',
      'setVertexBuffer',
      'setIndexBuffer',
      'drawIndexed',
      'renderPass.end',
      'encoder.finish',
      'submit',
    ]
  );

  // Verify createTexture & writeTexture
  const createTexCall = calls.find((c) => c.op === 'createTexture');
  assert.deepEqual(createTexCall.desc, {
    size: { width: 2, height: 2, depthOrArrayLayers: 1 },
    mipLevelCount: 1,
    format: 'rgba8unorm',
    usage: TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
    dimension: '2d',
  });
  const writeTexCall = calls.find((c) => c.op === 'writeTexture');
  assert.equal(writeTexCall.destination.texture, createTexCall.result);
  assert.equal(writeTexCall.destination.mipLevel, 0);
  assert.deepEqual(writeTexCall.destination.origin, { x: 0, y: 0, z: 0 });
  assert.deepEqual(writeTexCall.data, texels);
  assert.deepEqual(writeTexCall.dataLayout, { offset: 0, bytesPerRow: 8, rowsPerImage: 2 });
  assert.deepEqual(writeTexCall.size, { width: 2, height: 2, depthOrArrayLayers: 1 });

  // Verify createSampler
  const samplerCall = calls.find((c) => c.op === 'createSampler');
  assert.deepEqual(samplerCall.desc, {
    addressModeU: 'repeat',
    addressModeV: 'mirror-repeat',
    addressModeW: 'clamp-to-edge',
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'nearest',
    maxAnisotropy: 8,
  });

  // Verify createBindGroup entries (size > 0 included; size === 0 omitted)
  const bgCalls = calls.filter((c) => c.op === 'createBindGroup');
  assert.equal(bgCalls.length, 2);
  assert.deepEqual(bgCalls[0].desc.entries, [
    { binding: 0, resource: { buffer: processor.buffers.get(1), offset: 64, size: 128 } },
  ]);
  assert.deepEqual(bgCalls[1].desc.entries, [
    { binding: 0, resource: { buffer: processor.buffers.get(1), offset: 0 } },
    { binding: 1, resource: processor.textureViews.get(4) },
    { binding: 2, resource: processor.samplers.get(5) },
  ]);

  // Verify compute dispatch & indexed draw (including negative baseVertex)
  const dispatchCall = calls.find((c) => c.op === 'dispatchWorkgroups');
  assert.deepEqual(dispatchCall, { op: 'dispatchWorkgroups', x: 4, y: 2, z: 1 });

  const renderPassCall = calls.find((c) => c.op === 'beginRenderPass');
  assert.deepEqual(renderPassCall.desc, {
    colorAttachments: [
      {
        view: mockTextureView,
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0.25, g: 0.5, b: 0.75, a: 1.0 },
      },
    ],
  });

  const setIbCall = calls.find((c) => c.op === 'setIndexBuffer');
  assert.equal(setIbCall.buffer, processor.buffers.get(3));
  assert.equal(setIbCall.indexFormat, 'uint16');
  assert.equal(setIbCall.offset, 0);

  const drawIdxCall = calls.find((c) => c.op === 'drawIndexed');
  assert.deepEqual(drawIdxCall, {
    op: 'drawIndexed',
    indexCount: 6,
    instanceCount: 2,
    firstIndex: 0,
    baseVertex: -4,
    firstInstance: 1,
  });
});

test('Pass switching automatically ends render->compute->render passes and preserves prior render contents', () => {
  const { device, ctx, calls } = createMockWebGpu();
  const processor = new WgpuCommandProcessor(device, ctx, { format: 'bgra8unorm' });

  const stream = new BinaryCommandBuilder()
    .beginFrame()
    .createShaderModule(1, SHADER_STAGE.VERTEX, 'fn vs_main() {}')
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, 'fn fs_main() {}')
    .createShaderModule(3, SHADER_STAGE.COMPUTE, 'fn cs_main() {}')
    .createRenderPipeline(4, 1, 2, 0)
    .createComputePipeline(5, 3)
    .createBuffer(6, BUFFER_USAGE.INDEX, 16)
    .clearColor(0.1, 0.2, 0.3, 1.0)
    // Pass 1: Render (clear)
    .setPipeline(4)
    .draw(3, 1, 0, 0)
    // Pass 2: Compute (auto-ends render pass 1)
    .setComputePipeline(5)
    .dispatchCompute(2, 1, 1)
    // Pass 3: Render resumed (auto-ends compute pass, uses loadOp: 'load')
    .setPipeline(4)
    .setIndexBuffer(6, INDEX_FORMAT.UINT32, 4)
    .drawIndexed(3, 1, 0, 0, 0)
    .present()
    // Frame 2: Compute-only before Present (auto-ends compute pass, still clears/presents swapchain)
    .beginFrame()
    .clearColor(0.875, 0.75, 0.625, 1.0)
    .setComputePipeline(5)
    .dispatchCompute(1, 1, 1)
    .present()
    .finish();

  const res = processor.execute(stream);
  assert.deepEqual(res, { frameCount: 2, presents: 2 });

  const passOps = calls
    .filter((c) =>
      [
        'beginRenderPass',
        'renderPass.end',
        'beginComputePass',
        'computePass.end',
        'draw',
        'dispatchWorkgroups',
        'setIndexBuffer',
        'drawIndexed',
        'submit',
      ].includes(c.op)
    )
    .map((c) => c.op);

  assert.deepEqual(passOps, [
    // Frame 1
    'beginRenderPass',
    'draw',
    'renderPass.end',
    'beginComputePass',
    'dispatchWorkgroups',
    'computePass.end',
    'beginRenderPass',
    'setIndexBuffer',
    'drawIndexed',
    'renderPass.end',
    'submit',
    // Frame 2
    'beginComputePass',
    'dispatchWorkgroups',
    'computePass.end',
    'beginRenderPass',
    'renderPass.end',
    'submit',
  ]);

  const renderPasses = calls.filter((c) => c.op === 'beginRenderPass');
  assert.equal(renderPasses.length, 3);
  assert.equal(renderPasses[0].desc.colorAttachments[0].loadOp, 'clear');
  assert.equal(renderPasses[1].desc.colorAttachments[0].loadOp, 'load');
  assert.equal(renderPasses[1].desc.colorAttachments[0].clearValue, undefined);
  assert.equal(renderPasses[2].desc.colorAttachments[0].loadOp, 'clear');
  assert.deepEqual(renderPasses[2].desc.colorAttachments[0].clearValue, {
    r: 0.875,
    g: 0.75,
    b: 0.625,
    a: 1.0,
  });

  const setIbCall = calls.find((c) => c.op === 'setIndexBuffer');
  assert.equal(setIbCall.indexFormat, 'uint32');
  assert.equal(setIbCall.offset, 4);
});

test('DestroyResource cleans up textures, textureViews, samplers, bindGroups, computePipelines, and replaced textures', () => {
  const { device, ctx, calls } = createMockWebGpu();
  const processor = new WgpuCommandProcessor(device, ctx);

  const stream = new BinaryCommandBuilder()
    .createTexture(100, 4, 4, 1, 1, TEXTURE_FORMAT.BC7_RGBA_UNORM, TEXTURE_USAGE.TEXTURE_BINDING)
    // Recreating with same ID must destroy previous texture first.
    .createTexture(100, 8, 8, 1, 2, TEXTURE_FORMAT.RGBA16_FLOAT, TEXTURE_USAGE.TEXTURE_BINDING)
    .createSampler(101)
    .createShaderModule(102, SHADER_STAGE.COMPUTE, 'fn cs_main() {}')
    .createComputePipeline(103, 102)
    .createBindGroup(104, 103, 0, [
      { binding: 0, resourceType: BIND_RESOURCE_TYPE.TEXTURE_VIEW, resourceId: 100 },
      { binding: 1, resourceType: BIND_RESOURCE_TYPE.SAMPLER, resourceId: 101 },
    ])
    .destroyResource(100)
    .destroyResource(101)
    .destroyResource(102)
    .destroyResource(103)
    .destroyResource(104)
    .finish();

  processor.execute(stream);

  const createdTextures = calls.filter((c) => c.op === 'createTexture').map((c) => c.result);
  assert.equal(createdTextures.length, 2);
  assert.equal(createdTextures[0].desc.format, 'bc7-rgba-unorm');
  assert.equal(createdTextures[1].desc.format, 'rgba16float');
  assert.equal(createdTextures[0].destroyed, true);
  assert.equal(createdTextures[1].destroyed, true);

  assert.equal(processor.textures.size, 0);
  assert.equal(processor.textureViews.size, 0);
  assert.equal(processor.samplers.size, 0);
  assert.equal(processor.bindGroups.size, 0);
  assert.equal(processor.shaderModules.size, 0);
  assert.equal(processor.computePipelines.size, 0);
});

test('W92 selectSupportedAdapterFeatures picks only supported optional features and handles missing adapter.features', () => {
  assert.equal(Object.isFrozen(OPTIONAL_ADAPTER_FEATURES), true);
  assert.deepEqual(OPTIONAL_ADAPTER_FEATURES, [
    'texture-compression-bc',
    'depth32float-stencil8',
    'float32-filterable',
    'bgra8unorm-storage',
    'rg11b10ufloat-renderable',
    'texture-formats-tier1',
  ]);
  assert.equal(globalThis.PlayGta6Wgpu.OPTIONAL_ADAPTER_FEATURES, OPTIONAL_ADAPTER_FEATURES);
  assert.equal(
    globalThis.PlayGta6Wgpu.selectSupportedAdapterFeatures,
    selectSupportedAdapterFeatures
  );

  assert.deepEqual(selectSupportedAdapterFeatures(null), []);
  assert.deepEqual(selectSupportedAdapterFeatures(undefined), []);
  assert.deepEqual(selectSupportedAdapterFeatures({}), []);
  assert.deepEqual(selectSupportedAdapterFeatures({ features: null }), []);
  assert.deepEqual(selectSupportedAdapterFeatures({ features: {} }), []);
  assert.deepEqual(
    selectSupportedAdapterFeatures({
      features: {
        has() {
          throw new Error('probe error');
        },
      },
    }),
    []
  );

  const partialAdapter = {
    features: new Set([
      'texture-compression-bc',
      'depth32float-stencil8',
      'rg11b10ufloat-renderable',
      'indirect-first-instance',
      'timestamp-query',
    ]),
  };
  assert.deepEqual(selectSupportedAdapterFeatures(partialAdapter), [
    'texture-compression-bc',
    'depth32float-stencil8',
    'rg11b10ufloat-renderable',
  ]);

  const fullAdapter = {
    features: new Set(OPTIONAL_ADAPTER_FEATURES),
  };
  assert.deepEqual(selectSupportedAdapterFeatures(fullAdapter), [...OPTIONAL_ADAPTER_FEATURES]);
});

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

function createMockOffscreenCanvas(ctx) {
  const configured = [];
  const webgpuCtx = {
    ...ctx,
    configure(desc) {
      configured.push(desc);
    },
  };
  return {
    canvas: {
      width: 0,
      height: 0,
      getContext(kind) {
        assert.equal(kind, 'webgpu');
        return webgpuCtx;
      },
    },
    configured,
  };
}

test('W92 defaultInitGpu requests supported optional adapter features and falls back cleanly if requestDevice rejects', async () => {
  // 1. Adapter supports optional features -> passes requiredFeatures to requestDevice.
  {
    const { device, ctx } = createMockWebGpu();
    const supported = [
      'texture-compression-bc',
      'depth32float-stencil8',
      'float32-filterable',
    ];
    device.features = new Set(supported);
    const requestDeviceCalls = [];

    const mockGpu = {
      async requestAdapter() {
        return {
          info: { vendor: 'mock-gpu', architecture: 'rdna2' },
          features: new Set([...supported, 'timestamp-query']),
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

    const { canvas } = createMockOffscreenCanvas(ctx);
    const posted = [];
    const handler = createWgpuWorkerHandler({
      postMessage: (m) => posted.push(m),
      enableSwapchain: false,
    });

    await withMockNavigatorGpu(mockGpu, async () => {
      await handler.handleMessage({ init: true, canvas, width: 640, height: 360 });
    });

    assert.equal(requestDeviceCalls.length, 1);
    assert.deepEqual(requestDeviceCalls[0], { requiredFeatures: supported });
    const readyMsg = posted.find((m) => m && m.ready === true);
    assert.ok(readyMsg, 'expected ready postMessage');
    assert.equal(readyMsg.adapter, 'mock-gpu rdna2');
    assert.deepEqual(readyMsg.features, supported);
  }

  // 2. requestDevice({ requiredFeatures }) rejects -> falls back to requestDevice() with no args.
  {
    const { device, ctx } = createMockWebGpu();
    device.features = new Set();
    const requestDeviceCalls = [];

    const mockGpu = {
      async requestAdapter() {
        return {
          info: { vendor: 'fallback-gpu', architecture: '' },
          features: new Set(['texture-compression-bc', 'texture-formats-tier1']),
          async requestDevice(desc) {
            requestDeviceCalls.push(desc);
            if (desc && desc.requiredFeatures) {
              throw new Error('requested feature rejected by driver');
            }
            return device;
          },
        };
      },
      getPreferredCanvasFormat() {
        return 'bgra8unorm';
      },
    };

    const { canvas } = createMockOffscreenCanvas(ctx);
    const posted = [];
    const handler = createWgpuWorkerHandler({
      postMessage: (m) => posted.push(m),
      enableSwapchain: false,
    });

    await withMockNavigatorGpu(mockGpu, async () => {
      await handler.handleMessage({ init: true, canvas, width: 640, height: 360 });
    });

    assert.equal(requestDeviceCalls.length, 2);
    assert.deepEqual(requestDeviceCalls[0], {
      requiredFeatures: ['texture-compression-bc', 'texture-formats-tier1'],
    });
    assert.equal(requestDeviceCalls[1], undefined);
    const readyMsg = posted.find((m) => m && m.ready === true);
    assert.ok(readyMsg, 'expected ready postMessage after fallback');
    assert.equal(readyMsg.adapter, 'fallback-gpu');
    assert.deepEqual(readyMsg.features, []);
  }

  // 3. Adapter without features -> calls requestDevice() once with no arguments.
  {
    const { device, ctx } = createMockWebGpu();
    const requestDeviceCalls = [];

    const mockGpu = {
      async requestAdapter() {
        return {
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

    const { canvas } = createMockOffscreenCanvas(ctx);
    const posted = [];
    const handler = createWgpuWorkerHandler({
      postMessage: (m) => posted.push(m),
      enableSwapchain: false,
    });

    await withMockNavigatorGpu(mockGpu, async () => {
      await handler.handleMessage({ init: true, canvas, width: 320, height: 180 });
    });

    assert.equal(requestDeviceCalls.length, 1);
    assert.equal(requestDeviceCalls[0], undefined);
    const readyMsg = posted.find((m) => m && m.ready === true);
    assert.ok(readyMsg);
    assert.deepEqual(readyMsg.features, []);
  }
});

test('W92 buildGpuTelemetrySnapshot exposes realPackDrawCount and fallbackDrawCount in frameStats', () => {
  const { device, ctx } = createMockWebGpu();
  const vsWgsl =
    '@vertex fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }';
  const fsWgsl =
    '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.2, 0.7, 0.4, 1.0); }';
  const fallbackFsWgsl =
    '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.9, 0.1, 0.5, 1.0); }';

  const vsHash = formatHashHex(fnv1a64(vsWgsl));
  const fsHash = formatHashHex(fnv1a64(fsWgsl));
  const fallbackFsHash = '0x000000000000f001';
  const missingFsHash = '0xdeadbeef00009201';

  const packBuf = buildWgslPackBuffer([
    { hash: vsHash, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: vsWgsl },
    { hash: fsHash, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: fsWgsl },
    { hash: fallbackFsHash, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: fallbackFsWgsl },
  ]);

  const processor = new WgpuCommandProcessor(device, ctx, {
    format: 'bgra8unorm',
    wgslPackBuffer: packBuf,
    fallbackShaderHashes: { fragment: fallbackFsHash },
  });

  const stream = new BinaryCommandBuilder()
    .beginFrame()
    .createShaderModule(1, SHADER_STAGE.VERTEX, vsHash)
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, fsHash)
    .createShaderModule(3, SHADER_STAGE.FRAGMENT, missingFsHash)
    .createRenderPipeline(10, 1, 2, 0)
    .createRenderPipeline(11, 1, 3, 0)
    .setPipeline(10)
    .draw(3, 1, 0, 0)
    .draw(3, 1, 0, 0)
    .setPipeline(11)
    .draw(3, 1, 0, 0)
    .present()
    .finish();

  processor.execute(stream);

  const snapshot = buildGpuTelemetrySnapshot(processor, { timestamp: 12345 });
  assert.equal(snapshot.frameStats.framesPresented, 1);
  assert.equal(snapshot.frameStats.drawCalls, 3);
  assert.equal(snapshot.frameStats.realPackDrawCount, 2);
  assert.equal(snapshot.frameStats.fallbackDrawCount, 1);

  // Fallback to direct processor properties and non-negative integer coercion.
  const directPropSnap = buildGpuTelemetrySnapshot(
    {
      getStats: () => ({ frameCount: 2, drawCalls: 5 }),
      realPackDrawCount: 4,
      fallbackDrawCount: 1,
    },
    { timestamp: 200 }
  );
  assert.equal(directPropSnap.frameStats.realPackDrawCount, 4);
  assert.equal(directPropSnap.frameStats.fallbackDrawCount, 1);

  const coercedSnap = buildGpuTelemetrySnapshot(
    {
      getStats: () => ({
        frameCount: 1,
        realPackDrawCount: -5,
        fallbackDrawCount: Number.NaN,
      }),
    },
    { timestamp: 300 }
  );
  assert.equal(coercedSnap.frameStats.realPackDrawCount, 0);
  assert.equal(coercedSnap.frameStats.fallbackDrawCount, 0);
});
