import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import {
  WgslPackCache,
  SHADER_STAGE as WGPK_SHADER_STAGE,
  buildWgslPackBuffer,
  fnv1a64 as wgpkFnv1a64,
  formatHashHex as wgpkFormatHashHex,
} from '../web/workers/wgsl_pack_loader.js';

const require = createRequire(import.meta.url);

const wgpuMod = require('../web/workers/wgpu_worker.js');
const stagingMod = require('../web/workers/staging_pool.js');
const uniformMod = require('../web/workers/uniform_ring.js');
const timerMod = require('../web/workers/gpu_timer.js');
const reflectionMod = require('../web/workers/reflection_loader.js');
const registryMod = require('../web/workers/resource_registry.js');
const rtPoolMod = require('../web/workers/rt_pool.js');
const pipelineCacheMod = require('../web/workers/pipeline_cache.js');
const swapchainMod = require('../web/workers/wgpu_swapchain.js');
const renderStateMod = require('../web/workers/render_state_packets.js');
const texturePacketsMod = require('../web/workers/texture_packets.js');

const {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  buildGpuTelemetrySnapshot,
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

const { WgpuSwapchain } = swapchainMod;
const {
  installRenderStatePackets,
  SWAPCHAIN_TEXTURE_ID,
  TEXTURE_FORMAT_NAMES: RS_TEXTURE_FORMAT_NAMES,
} = renderStateMod;
const { installTexturePackets, CREATE_TEXTURE2 } = texturePacketsMod;

const { GpuStagingPool } = stagingMod;
const { DynamicUniformRing } = uniformMod;
const { GpuPassTimer } = timerMod;
const {
  ReflectionRegistry,
  SHADER_STAGE_BITS,
  fnv1a64,
  formatShaderHash,
  computeBindGroupLayoutHash,
  computePipelineLayoutHash,
} = reflectionMod;
const { GpuResourceRegistry } = registryMod;
const { RenderTargetPool } = rtPoolMod;
const { WebGpuPipelineCache } = pipelineCacheMod;

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

  createRenderPipeline(
    id,
    vsShaderId,
    fsShaderId,
    vertexStride,
    topology = PRIMITIVE_TOPOLOGY.TRIANGLE_LIST,
    flags = 0
  ) {
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
    this._pushPacket(OPCODES.CREATE_BIND_GROUP, new Uint8Array(buf));
    return this;
  }

  setPipeline(pipelineId) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, pipelineId, true);
    this._pushPacket(OPCODES.SET_PIPELINE, new Uint8Array(buf));
    return this;
  }

  setComputePipeline(pipelineId) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, pipelineId, true);
    this._pushPacket(OPCODES.SET_COMPUTE_PIPELINE, new Uint8Array(buf));
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

  setVertexBuffer(slot, bufferId, offset = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, slot, true);
    view.setUint32(4, bufferId, true);
    view.setBigUint64(8, BigInt(offset), true);
    this._pushPacket(OPCODES.SET_VERTEX_BUFFER, new Uint8Array(buf));
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

  dispatchCompute(workgroupX, workgroupY = 1, workgroupZ = 1) {
    const buf = new ArrayBuffer(12);
    const view = new DataView(buf);
    view.setUint32(0, workgroupX, true);
    view.setUint32(4, workgroupY, true);
    view.setUint32(8, workgroupZ, true);
    this._pushPacket(OPCODES.DISPATCH_COMPUTE, new Uint8Array(buf));
    return this;
  }

  destroyResource(id) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, id, true);
    this._pushPacket(OPCODES.DESTROY_RESOURCE, new Uint8Array(buf));
    return this;
  }

  present() {
    this._pushPacket(OPCODES.PRESENT, new Uint8Array(0));
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

/**
 * Full-featured WebGPU mock device supporting:
 * - Buffers with backing ArrayBuffer, mapAsync/getMappedRange/unmap, destroy tracking
 * - Textures + views + destroy tracking
 * - Samplers, ShaderModules, BindGroupLayouts, PipelineLayouts, Pipelines, BindGroups
 * - Timestamp GPUQuerySet + resolveQuerySet + simulated nanosecond pass durations
 * - copyBufferToBuffer (copies real bytes!) + copyBufferToTexture
 * - Controllable queue.onSubmittedWorkDone fences
 */
function createFullMockWebGpu(options = {}) {
  const {
    timestampQuery = true,
    passDurationNs = [1_500_000n, 3_250_000n, 750_000n, 2_000_000n],
    manualFences = false,
  } = options;

  const calls = [];
  const createdBuffers = [];
  const createdTextures = [];
  const createdQuerySets = [];
  const pendingFences = [];
  let nextHandle = 1;
  let swapchainCounter = 0;

  const ctx = {
    getCurrentTexture() {
      const view = { kind: 'GPUTextureView', id: `swapchain-view-${swapchainCounter++}` };
      return {
        createView() {
          return view;
        },
      };
    },
  };

  const features = new Set(timestampQuery ? ['timestamp-query'] : []);

  const device = {
    features,
    limits: {
      minUniformBufferOffsetAlignment: 256,
      maxTextureDimension2D: 8192,
    },

    createQuerySet(desc) {
      const qs = {
        kind: 'GPUQuerySet',
        handle: nextHandle++,
        type: desc.type,
        count: desc.count,
        label: desc.label || '',
        timestamps: new BigUint64Array(desc.count),
        destroyed: false,
        destroyCount: 0,
        destroy() {
          this.destroyed = true;
          this.destroyCount += 1;
          calls.push({ op: 'querySet.destroy', querySet: this });
        },
      };
      createdQuerySets.push(qs);
      calls.push({ op: 'createQuerySet', desc: { ...desc }, result: qs });
      return qs;
    },

    createBuffer(desc) {
      const backing = new ArrayBuffer(desc.size);
      const buffer = {
        kind: 'GPUBuffer',
        handle: nextHandle++,
        label: desc.label || '',
        size: desc.size,
        usage: desc.usage,
        _backing: backing,
        _bytes: new Uint8Array(backing),
        mapState: desc.mappedAtCreation ? 'mapped' : 'unmapped',
        destroyed: false,
        destroyCount: 0,
        unmapCount: 0,
        getMappedRange(offset = 0, size = undefined) {
          const len = size === undefined ? this.size - offset : size;
          return new Uint8Array(this._backing, offset, len).buffer.slice(offset, offset + len);
        },
        // Override for mappedAtCreation writes to mutate backing directly
        _writableRange() {
          return this._backing;
        },
        unmap() {
          this.mapState = 'unmapped';
          this.unmapCount += 1;
          calls.push({ op: 'buffer.unmap', buffer: this });
        },
        async mapAsync() {
          this.mapState = 'mapped';
          calls.push({ op: 'buffer.mapAsync', buffer: this });
        },
        destroy() {
          this.destroyed = true;
          this.destroyCount += 1;
          calls.push({ op: 'buffer.destroy', buffer: this });
        },
      };
      // Return direct backing buffer from getMappedRange so mapped-mode staging writes land in _bytes
      buffer.getMappedRange = function getMappedRange(offset = 0, size = undefined) {
        if (offset === 0 && (size === undefined || size === this.size)) {
          return this._backing;
        }
        const len = size === undefined ? this.size - offset : size;
        return new Uint8Array(this._backing, offset, len);
      };
      createdBuffers.push(buffer);
      calls.push({ op: 'createBuffer', desc: { ...desc }, result: buffer });
      return buffer;
    },

    createTexture(desc) {
      const handle = nextHandle++;
      const texture = {
        kind: 'GPUTexture',
        handle,
        label: desc.label || '',
        desc: { ...desc, size: { ...desc.size } },
        destroyed: false,
        destroyCount: 0,
        uploads: [],
        createView() {
          const view = { kind: 'GPUTextureView', textureHandle: handle, handle: nextHandle++ };
          calls.push({ op: 'texture.createView', texture, result: view });
          return view;
        },
        destroy() {
          this.destroyed = true;
          this.destroyCount += 1;
          calls.push({ op: 'texture.destroy', texture });
        },
      };
      createdTextures.push(texture);
      calls.push({ op: 'createTexture', desc: { ...desc, size: { ...desc.size } }, result: texture });
      return texture;
    },

    createSampler(desc) {
      const sampler = { kind: 'GPUSampler', handle: nextHandle++, desc: { ...desc } };
      calls.push({ op: 'createSampler', desc: { ...desc }, result: sampler });
      return sampler;
    },

    createShaderModule(desc) {
      const mod = { kind: 'GPUShaderModule', handle: nextHandle++, code: desc.code };
      calls.push({ op: 'createShaderModule', desc: { ...desc }, result: mod });
      return mod;
    },

    createBindGroupLayout(desc) {
      const bgl = { kind: 'GPUBindGroupLayout', handle: nextHandle++, desc };
      calls.push({ op: 'createBindGroupLayout', desc, result: bgl });
      return bgl;
    },

    createPipelineLayout(desc) {
      const pl = { kind: 'GPUPipelineLayout', handle: nextHandle++, desc };
      calls.push({ op: 'createPipelineLayout', desc, result: pl });
      return pl;
    },

    createRenderPipeline(desc) {
      const pipeline = {
        kind: 'GPURenderPipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayoutCalls: 0,
        getBindGroupLayout(index) {
          this.getBindGroupLayoutCalls += 1;
          const layout = { kind: 'GPUBindGroupLayout', pipelineHandle: this.handle, index };
          calls.push({ op: 'renderPipeline.getBindGroupLayout', index, result: layout });
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
        getBindGroupLayoutCalls: 0,
        getBindGroupLayout(index) {
          this.getBindGroupLayoutCalls += 1;
          const layout = { kind: 'GPUBindGroupLayout', pipelineHandle: this.handle, index };
          calls.push({ op: 'computePipeline.getBindGroupLayout', index, result: layout });
          return layout;
        },
      };
      calls.push({ op: 'createComputePipeline', desc, result: pipeline });
      return pipeline;
    },

    createBindGroup(desc) {
      const bg = { kind: 'GPUBindGroup', handle: nextHandle++, desc };
      calls.push({ op: 'createBindGroup', desc, result: bg });
      return bg;
    },

    createCommandEncoder() {
      const commandBuffer = { kind: 'GPUCommandBuffer', handle: nextHandle++ };
      let passCursor = 0;
      let clockNs = 10_000_000n;

      const recordTimestampWrites = (tsWrites) => {
        if (!tsWrites || !tsWrites.querySet) return;
        const dur = passDurationNs[passCursor % passDurationNs.length];
        passCursor += 1;
        const beginNs = clockNs;
        const endNs = clockNs + dur;
        clockNs = endNs + 100_000n;
        tsWrites.querySet.timestamps[tsWrites.beginningOfPassWriteIndex] = beginNs;
        tsWrites.querySet.timestamps[tsWrites.endOfPassWriteIndex] = endNs;
      };

      return {
        copyBufferToBuffer(src, srcOffset, dst, dstOffset, size) {
          if (src._bytes && dst._bytes) {
            dst._bytes.set(src._bytes.subarray(srcOffset, srcOffset + size), dstOffset);
          }
          calls.push({ op: 'copyBufferToBuffer', src, srcOffset, dst, dstOffset, size });
        },

        copyBufferToTexture(source, destination, copySize) {
          const sliceBytes = source.buffer._bytes
            ? source.buffer._bytes.slice(
                source.offset,
                source.offset + source.bytesPerRow * source.rowsPerImage * (copySize.depthOrArrayLayers || 1)
              )
            : new Uint8Array(0);
          if (destination.texture && destination.texture.uploads) {
            destination.texture.uploads.push({
              source: { ...source },
              destination: { ...destination, origin: { ...destination.origin } },
              copySize: { ...copySize },
              bytes: sliceBytes,
            });
          }
          calls.push({
            op: 'copyBufferToTexture',
            source: { ...source },
            destination: { ...destination, origin: { ...destination.origin } },
            copySize: { ...copySize },
          });
        },

        resolveQuerySet(querySet, firstQuery, queryCount, destination, destinationOffset) {
          const byteLen = queryCount * 8;
          const srcBytes = new Uint8Array(
            querySet.timestamps.buffer,
            querySet.timestamps.byteOffset + firstQuery * 8,
            byteLen
          );
          destination._bytes.set(srcBytes, destinationOffset);
          calls.push({
            op: 'resolveQuerySet',
            querySet,
            firstQuery,
            queryCount,
            destination,
            destinationOffset,
          });
        },

        beginRenderPass(desc) {
          recordTimestampWrites(desc && desc.timestampWrites);
          calls.push({ op: 'beginRenderPass', desc });
          return {
            setPipeline(pipeline) {
              calls.push({ op: 'renderPass.setPipeline', pipeline });
            },
            setBindGroup(index, bindGroup) {
              calls.push({ op: 'renderPass.setBindGroup', index, bindGroup });
            },
            setVertexBuffer(slot, buffer, offset) {
              calls.push({ op: 'renderPass.setVertexBuffer', slot, buffer, offset });
            },
            setIndexBuffer(buffer, indexFormat, offset) {
              calls.push({ op: 'renderPass.setIndexBuffer', buffer, indexFormat, offset });
            },
            draw(vertexCount, instanceCount, firstVertex, firstInstance) {
              calls.push({
                op: 'renderPass.draw',
                vertexCount,
                instanceCount,
                firstVertex,
                firstInstance,
              });
            },
            drawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance) {
              calls.push({
                op: 'renderPass.drawIndexed',
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

        beginComputePass(desc) {
          recordTimestampWrites(desc && desc.timestampWrites);
          calls.push({ op: 'beginComputePass', desc });
          return {
            setPipeline(pipeline) {
              calls.push({ op: 'computePass.setPipeline', pipeline });
            },
            setBindGroup(index, bindGroup) {
              calls.push({ op: 'computePass.setBindGroup', index, bindGroup });
            },
            dispatchWorkgroups(x, y, z) {
              calls.push({ op: 'computePass.dispatchWorkgroups', x, y, z });
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
        const bytes =
          data instanceof Uint8Array
            ? data
            : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        if (buffer._bytes) {
          buffer._bytes.set(bytes, dstOffset);
        }
        calls.push({
          op: 'queue.writeBuffer',
          buffer,
          dstOffset,
          data: bytes.slice(),
        });
      },
      writeTexture(destination, data, dataLayout, size) {
        const bytes =
          data instanceof Uint8Array
            ? data
            : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        calls.push({
          op: 'queue.writeTexture',
          destination: { ...destination, origin: { ...destination.origin } },
          data: bytes.slice(),
          dataLayout: { ...dataLayout },
          size: { ...size },
        });
      },
      submit(commandBuffers) {
        calls.push({ op: 'queue.submit', commandBuffers: [...commandBuffers] });
      },
      onSubmittedWorkDone() {
        if (manualFences) {
          let resolve;
          const promise = new Promise((res) => {
            resolve = res;
          });
          pendingFences.push({ promise, resolve });
          return promise;
        }
        return Promise.resolve();
      },
    },
  };

  return {
    device,
    ctx,
    calls,
    createdBuffers,
    createdTextures,
    createdQuerySets,
    resolveAllFences() {
      while (pendingFences.length > 0) {
        const f = pendingFences.shift();
        f.resolve();
      }
    },
  };
}

function buildS6ReflectionManifest(vsWgsl, fsWgsl, csWgsl) {
  const vsHash = formatShaderHash(fnv1a64(vsWgsl));
  const fsHash = formatShaderHash(fnv1a64(fsWgsl));
  const csHash = formatShaderHash(fnv1a64(csWgsl));

  const vsBglEntries = [
    {
      binding: 0,
      visibility: SHADER_STAGE_BITS.VERTEX,
      buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: 64 },
    },
  ];
  const fsBglEntries = [
    {
      binding: 0,
      visibility: SHADER_STAGE_BITS.FRAGMENT,
      buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: 64 },
    },
    {
      binding: 1,
      visibility: SHADER_STAGE_BITS.FRAGMENT,
      texture: { sampleType: 'float', viewDimension: '2d', multisampled: false },
    },
    {
      binding: 2,
      visibility: SHADER_STAGE_BITS.FRAGMENT,
      sampler: { type: 'filtering' },
    },
  ];
  const csBglEntries = [
    {
      binding: 0,
      visibility: SHADER_STAGE_BITS.COMPUTE,
      buffer: { type: 'storage', hasDynamicOffset: false, minBindingSize: 16 },
    },
  ];

  const vsBglId = computeBindGroupLayoutHash({ entries: vsBglEntries });
  const fsBglId = computeBindGroupLayoutHash({ entries: fsBglEntries });
  const csBglId = computeBindGroupLayoutHash({ entries: csBglEntries });

  const vsPlId = computePipelineLayoutHash([vsBglId]);
  const fsPlId = computePipelineLayoutHash([fsBglId]);
  const csPlId = computePipelineLayoutHash([csBglId]);

  return {
    schema_version: 1,
    register_convention: { source: 'dxc-spirv', shifts: { b: 0, t: 0, u: 0, s: 0 } },
    bind_group_layouts: {
      [vsBglId]: { entries: vsBglEntries },
      [fsBglId]: { entries: fsBglEntries },
      [csBglId]: { entries: csBglEntries },
    },
    pipeline_layouts: {
      [vsPlId]: { bindGroupLayouts: [vsBglId] },
      [fsPlId]: { bindGroupLayouts: [fsBglId] },
      [csPlId]: { bindGroupLayouts: [csBglId] },
    },
    shaders: {
      [vsHash]: {
        stage: 'vertex',
        entry_point: 'vs_main',
        name: 'g23_vs',
        pipeline_layout: vsPlId,
        bind_group_layouts: [vsBglId],
        layout_complete: true,
        vertex_inputs: [
          { location: 0, type: 'vec3<f32>', format: 'float32x3' },
          { location: 1, type: 'vec2<f32>', format: 'float32x2' },
        ],
        bindings: [
          { group: 0, binding: 0, name: 'uFrame', resource: 'buffer', in_layout: true },
        ],
      },
      [fsHash]: {
        stage: 'fragment',
        entry_point: 'fs_main',
        name: 'g23_fs',
        pipeline_layout: fsPlId,
        bind_group_layouts: [fsBglId],
        layout_complete: true,
        fragment_outputs: [{ location: 0, type: 'vec4<f32>' }],
        bindings: [
          { group: 0, binding: 0, name: 'uFrame', resource: 'buffer', in_layout: true },
          { group: 0, binding: 1, name: 'tAlbedo', resource: 'texture', in_layout: true },
          { group: 0, binding: 2, name: 'sLinear', resource: 'sampler', in_layout: true },
        ],
      },
      [csHash]: {
        stage: 'compute',
        entry_point: 'cs_main',
        name: 'g23_cs',
        pipeline_layout: csPlId,
        bind_group_layouts: [csBglId],
        layout_complete: true,
        workgroup_size: [8, 8, 1],
        bindings: [
          { group: 0, binding: 0, name: 'rwSim', resource: 'buffer', in_layout: true },
        ],
      },
    },
  };
}

// ---------------------------------------------------------------------------
// 1. Full 6-subsystem multi-frame render + compute integration test
// ---------------------------------------------------------------------------

test('G23 full 6-subsystem integration across multi-frame render + compute streams', async () => {
  const mock = createFullMockWebGpu({
    timestampQuery: true,
    passDurationNs: [1_250_000n, 2_500_000n], // compute = 1.25ms, render = 2.50ms
    manualFences: true,
  });

  const stagingPool = new GpuStagingPool({
    device: mock.device,
    chunkSize: 4096,
    maxPoolBytes: 64 * 1024,
    framesInFlight: 2,
    mode: 'writeBuffer',
  });
  const uniformRing = new DynamicUniformRing({
    device: mock.device,
    byteSize: 4096,
    framesInFlight: 3,
  });
  const passTimer = new GpuPassTimer({
    device: mock.device,
    maxPassesPerFrame: 8,
    readbackSlots: 3,
  });
  const reflectionRegistry = new ReflectionRegistry();
  const resourceRegistry = new GpuResourceRegistry({
    framesInFlight: 2,
    softBudgetBytes: 1024 * 1024,
  });
  const rtPool = new RenderTargetPool({
    device: mock.device,
    maxPoolBytes: 16 * 1024 * 1024,
    idleFrameTtl: 3,
  });
  const pipelineCache = new WebGpuPipelineCache({ device: mock.device });

  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
    stagingPool,
    uniformRing,
    passTimer,
    reflectionRegistry,
    resourceRegistry,
    rtPool,
    pipelineCache,
  });

  const vsWgsl = '@vertex fn vs_main(@location(0) pos: vec3<f32>, @location(1) uv: vec2<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(pos, 1.0); }';
  const fsWgsl = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.2, 0.6, 0.9, 1.0); }';
  const csWgsl = '@compute @workgroup_size(8, 8, 1) fn cs_main() {}';

  const manifest = buildS6ReflectionManifest(vsWgsl, fsWgsl, csWgsl);
  processor.loadReflectionManifest(JSON.stringify(manifest));

  // Verify initial reflection shader count
  assert.equal(processor.getStats().reflectionShaderCount, 3);

  // Frame 0: resource creation, pooled buffer + texture uploads, reflected pipelines + bind groups,
  // dynamic uniform batching, transient RT aliasing, compute + render passes, Present.
  const vertexPayload = new Float32Array([
    0.0, 0.5, 0.0, 0.5, 0.0,
    -0.5, -0.5, 0.0, 0.0, 1.0,
    0.5, -0.5, 0.0, 1.0, 1.0,
  ]);
  const vertexBytes = new Uint8Array(vertexPayload.buffer);
  const indexBytes = new Uint8Array(new Uint16Array([0, 1, 2]).buffer); // 6 bytes -> padded to 8
  const texels = new Uint8Array(4 * 4 * 4); // 4x4 RGBA8 (16 bytes/row -> padded to 256 by stagingPool)
  for (let i = 0; i < texels.length; i += 1) {
    texels[i] = (i * 13) & 0xff;
  }

  processor.beginFrame(0);

  // Push 3 per-draw constant payloads into DynamicUniformRing before frame submission.
  const draw0Const = processor.pushDrawConstants(new Float32Array([1, 2, 3, 4]));
  const draw1Const = processor.pushDrawConstants(new Float32Array([5, 6, 7, 8]));
  const draw2Const = processor.pushDrawConstants(new Float32Array([9, 10, 11, 12]));
  assert.equal(draw0Const.dynamicOffset, 0);
  assert.equal(draw1Const.dynamicOffset, 256);
  assert.equal(draw2Const.dynamicOffset, 512);
  assert.equal(draw0Const.buffer, draw1Const.buffer);

  // Acquire & alias transient render targets within Frame 0.
  const gbufA = processor.acquireRenderTarget(
    { width: 128, height: 128, format: 'rgba8unorm' },
    'gbuffer-albedo'
  );
  assert.equal(gbufA.reused, false);
  assert.equal(processor.releaseRenderTarget(gbufA), true);
  const postFx = processor.acquireRenderTarget(
    { width: 128, height: 128, format: 'rgba8unorm' },
    'postfx-temp'
  );
  assert.equal(postFx.reused, true, 'Same-frame released RT should alias immediately');
  assert.equal(postFx.texture, gbufA.texture);
  assert.equal(processor.releaseRenderTarget(postFx), true);

  const frame0Stream = new BinaryCommandBuilder()
    .beginFrame()
    .createBuffer(1, BUFFER_USAGE.UNIFORM | BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST, 256)
    .createBuffer(2, BUFFER_USAGE.VERTEX | BUFFER_USAGE.COPY_DST, vertexBytes.byteLength)
    .writeBuffer(2, 0, vertexBytes)
    .createBuffer(3, BUFFER_USAGE.INDEX | BUFFER_USAGE.COPY_DST, 8)
    .writeBuffer(3, 0, indexBytes)
    .createTexture(
      4,
      4,
      4,
      1,
      2,
      TEXTURE_FORMAT.RGBA8_UNORM,
      TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
      TEXTURE_DIMENSION.DIM_2D
    )
    .writeTexture(4, 0, 0, 0, 0, 4, 4, 1, 16, 4, texels)
    .createSampler(5)
    .createShaderModule(6, SHADER_STAGE.COMPUTE, csWgsl)
    .createShaderModule(7, SHADER_STAGE.VERTEX, vsWgsl)
    .createShaderModule(8, SHADER_STAGE.FRAGMENT, fsWgsl)
    .createComputePipeline(9, 6)
    .createRenderPipeline(10, 7, 8, 20, PRIMITIVE_TOPOLOGY.TRIANGLE_LIST, 0)
    .createBindGroup(11, 9, 0, [
      { binding: 0, resourceType: BIND_RESOURCE_TYPE.BUFFER, resourceId: 1, offset: 0, size: 128 },
    ])
    .createBindGroup(12, 10, 0, [
      { binding: 0, resourceType: BIND_RESOURCE_TYPE.BUFFER, resourceId: 1, offset: 0, size: 64 },
      { binding: 1, resourceType: BIND_RESOURCE_TYPE.TEXTURE_VIEW, resourceId: 4 },
      { binding: 2, resourceType: BIND_RESOURCE_TYPE.SAMPLER, resourceId: 5 },
    ])
    .clearColor(0.1, 0.2, 0.3, 1.0)
    .setComputePipeline(9)
    .setBindGroup(0, 11)
    .dispatchCompute(2, 2, 1)
    .setPipeline(10)
    .setBindGroup(0, 12)
    .setVertexBuffer(0, 2, 0)
    .setIndexBuffer(3, INDEX_FORMAT.UINT16, 0)
    .drawIndexed(3, 1, 0, 0, 0)
    .present()
    .finish();

  const res0 = processor.executeCommandBuffer(frame0Stream);
  assert.deepEqual(res0, { frameCount: 1, presents: 1 });

  // Verify G14 StagingPool routed WriteBuffer & WriteTexture via copyBufferToBuffer / copyBufferToTexture
  const c2bCalls = mock.calls.filter((c) => c.op === 'copyBufferToBuffer');
  const c2tCalls = mock.calls.filter((c) => c.op === 'copyBufferToTexture');
  const directTexWrites = mock.calls.filter((c) => c.op === 'queue.writeTexture');
  assert.equal(directTexWrites.length, 0, 'WriteTexture must be staged through GpuStagingPool');
  assert.ok(c2bCalls.length >= 2, 'Expected staged copyBufferToBuffer for VB + IB');
  assert.equal(c2tCalls.length, 1, 'Expected staged copyBufferToTexture for 4x4 texture');
  assert.equal(c2tCalls[0].source.bytesPerRow, 256, 'StagingPool must pad bytesPerRow to 256 bytes');

  // Verify actual vertex buffer bytes arrived intact via staging copyBufferToBuffer
  const vb = processor.buffers.get(2);
  assert.deepEqual(vb._bytes.subarray(0, vertexBytes.byteLength), vertexBytes);

  // Verify G18 DynamicUniformRing flushed all 3 draws in a SINGLE queue.writeBuffer call (768 bytes)
  const uniformFlushes = mock.calls.filter(
    (c) => c.op === 'queue.writeBuffer' && c.buffer === draw0Const.buffer
  );
  assert.equal(uniformFlushes.length, 1, '3 draws must batch into 1 uniform ring writeBuffer');
  assert.equal(uniformFlushes[0].data.byteLength, 768);
  assert.equal(uniformRing.getStats().flushCount, 1);

  // Verify G17 ReflectionRegistry built explicit GPUPipelineLayout + merged visibility + reflected vertex inputs
  const renderPipeCall = mock.calls.find((c) => c.op === 'createRenderPipeline');
  const computePipeCall = mock.calls.find((c) => c.op === 'createComputePipeline');
  assert.equal(renderPipeCall.desc.layout.kind, 'GPUPipelineLayout');
  assert.equal(computePipeCall.desc.layout.kind, 'GPUPipelineLayout');
  assert.equal(renderPipeCall.desc.vertex.buffers.length, 1);
  assert.equal(renderPipeCall.desc.vertex.buffers[0].arrayStride, 20);
  assert.deepEqual(renderPipeCall.desc.vertex.buffers[0].attributes, [
    { shaderLocation: 0, offset: 0, format: 'float32x3' },
    { shaderLocation: 1, offset: 12, format: 'float32x2' },
  ]);

  // Verify merged group 0 binding 0 has VERTEX | FRAGMENT visibility
  const bglCalls = mock.calls.filter((c) => c.op === 'createBindGroupLayout');
  const renderBglCall = bglCalls.find((c) => c.desc.entries.length === 3);
  assert.ok(renderBglCall, 'Expected merged 3-entry render BindGroupLayout');
  assert.equal(
    renderBglCall.desc.entries[0].visibility,
    SHADER_STAGE_BITS.VERTEX | SHADER_STAGE_BITS.FRAGMENT
  );

  // Verify CreateBindGroup used the reflected GPUBindGroupLayout directly without pipeline.getBindGroupLayout()
  const renderPipeline = processor.pipelines.get(10);
  const computePipeline = processor.computePipelines.get(9);
  assert.equal(renderPipeline.getBindGroupLayoutCalls, 0);
  assert.equal(computePipeline.getBindGroupLayoutCalls, 0);
  assert.equal(processor.getBindGroupLayout(10, 0), renderBglCall.result);

  // Verify G16 GpuPassTimer attached timestampWrites and resolved per-pass GPU timings
  const computePassCall = mock.calls.find((c) => c.op === 'beginComputePass');
  const renderPassCall = mock.calls.find((c) => c.op === 'beginRenderPass');
  assert.ok(computePassCall.desc && computePassCall.desc.timestampWrites);
  assert.ok(renderPassCall.desc && renderPassCall.desc.timestampWrites);
  const timings0 = await processor.collectPassTimings();
  assert.ok(timings0, 'Expected resolved pass timings after frame 0');
  assert.equal(timings0.passes.length, 2);
  assert.equal(timings0.passes[0].name, 'compute_0');
  assert.equal(timings0.passes[0].durationMs, 1.25);
  assert.equal(timings0.passes[1].name, 'render_0');
  assert.equal(timings0.passes[1].durationMs, 2.5);
  assert.equal(timings0.totalGpuMs, 3.75);
  assert.deepEqual(processor.getLastPassTimings(), timings0);

  // Verify G21 GpuResourceRegistry VRAM byte accounting (buffers: 256 + 60 + 8 = 324;
  // texture 4x4 rgba8 mip0=64 + mip1=16 = 80; shader modules = byte lengths of 3 WGSL strings)
  const expectedShaderBytes =
    new TextEncoder().encode(csWgsl).byteLength +
    new TextEncoder().encode(vsWgsl).byteLength +
    new TextEncoder().encode(fsWgsl).byteLength;
  const regStats0 = resourceRegistry.getStats();
  assert.equal(regStats0.countsByKind.buffer, 3);
  assert.equal(regStats0.countsByKind.texture, 1);
  assert.equal(regStats0.countsByKind.textureView, 1);
  assert.equal(regStats0.countsByKind.sampler, 1);
  assert.equal(regStats0.countsByKind.shaderModule, 3);
  assert.equal(regStats0.countsByKind.renderPipeline, 1);
  assert.equal(regStats0.countsByKind.computePipeline, 1);
  assert.equal(regStats0.countsByKind.bindGroup, 2);
  assert.equal(regStats0.activeBytes, 324 + 80 + expectedShaderBytes);

  // Frame 1: retire buffer #1 via DestroyResource (deferred GC - NOT destroyed immediately!),
  // push uniform constants on rotated slot 1, and upload a small buffer to test staging chunk lifecycle.
  const retiredBuf = processor.buffers.get(1);
  const frame1Stream = new BinaryCommandBuilder()
    .beginFrame()
    .destroyResource(1)
    .writeBuffer(2, 0, vertexBytes)
    .setPipeline(10)
    .draw(3, 1, 0, 0)
    .present()
    .finish();

  processor.beginFrame(1);
  const frame1Const = processor.pushDrawConstants(new Float32Array([42, 43, 44, 45]));
  assert.notEqual(frame1Const.buffer, draw0Const.buffer, 'Frame 1 must rotate to uniform ring slot 1');
  assert.equal(frame1Const.dynamicOffset, 0);

  processor.executeCommandBuffer(frame1Stream);
  assert.equal(processor.buffers.has(1), false);
  assert.equal(retiredBuf.destroyed, false, 'Retired buffer must NOT be destroyed while fence is pending');
  assert.equal(resourceRegistry.getStats().pendingRetireCount, 1);
  assert.equal(resourceRegistry.getStats().pendingRetireBytes, 256);

  // Resolve GPU work-done fences -> Frame 2 beginFrame collects retired buffer AND recycles staging chunks!
  mock.resolveAllFences();
  await Promise.resolve();
  await Promise.resolve();

  const frame2Stream = new BinaryCommandBuilder()
    .beginFrame()
    .writeBuffer(2, 0, vertexBytes)
    .setPipeline(10)
    .draw(3, 1, 0, 0)
    .present()
    .finish();
  processor.executeCommandBuffer(frame2Stream);

  assert.equal(retiredBuf.destroyed, true, 'Retired buffer must be destroyed after GPU fence resolves');
  assert.equal(resourceRegistry.getStats().pendingRetireCount, 0);
  assert.equal(resourceRegistry.getStats().pendingRetireBytes, 0);
  assert.ok(
    stagingPool.getStats().bufferReuseHits >= 1,
    'Staging pool must recycle chunks across frames without churning'
  );

  // Verify RenderTargetPool resize invalidation
  const rtBeforeResize = rtPool.getStats();
  assert.equal(rtBeforeResize.idleCount, 1);
  const invalidated = processor.onResize(1920, 1080);
  assert.equal(invalidated, 1);
  assert.equal(rtPool.getStats().idleCount, 0);

  // Verify aggregated getStats()
  const stats = processor.getStats();
  assert.equal(stats.frameCount, 3);
  assert.ok(stats.packetCount > 20);
  assert.equal(stats.drawCalls, 3);
  assert.equal(stats.computeDispatches, 1);
  assert.equal(stats.destroyed, false);
  assert.equal(stats.reflectionShaderCount, 3);
  assert.equal(stats.vramBytes, resourceRegistry.getStats().totalBytes);
  assert.equal(stats.rtPoolHitRate, 0.5);
  assert.ok(stats.stagingPool.bytesUploaded > 0);
  assert.equal(stats.uniformRing.totalDrawCount, 4);
  assert.equal(stats.uniformRing.flushCount, 2);
  assert.equal(stats.passTimer.supported, true);
  assert.ok(stats.passTimer.resolvedFrames >= 1);

  // Verify idempotent destroy() cleans up all subsystems and resources without double-destroy
  processor.destroy();
  assert.equal(processor.destroyed, true);
  assert.equal(processor.buffers.size, 0);
  assert.equal(processor.textures.size, 0);
  assert.equal(rtPool.isDestroyed, true);
  assert.equal(passTimer.isDestroyed, true);
  assert.equal(uniformRing.getStats().destroyed, true);
  assert.equal(resourceRegistry.getStats().activeCount, 0);

  for (const buf of mock.createdBuffers) {
    assert.equal(buf.destroyed, true, `Buffer ${buf.label || buf.handle} must be destroyed`);
    assert.equal(buf.destroyCount, 1, `Buffer ${buf.label || buf.handle} must not be double-destroyed`);
  }
  for (const tex of mock.createdTextures) {
    assert.equal(tex.destroyed, true, `Texture ${tex.label || tex.handle} must be destroyed`);
    assert.equal(tex.destroyCount, 1, `Texture ${tex.label || tex.handle} must not be double-destroyed`);
  }
  for (const qs of mock.createdQuerySets) {
    assert.equal(qs.destroyed, true);
    assert.equal(qs.destroyCount, 1);
  }

  // Second destroy() is a clean no-op; post-destroy executeCommandBuffer is a safe no-op
  const callCountAfterDestroy = mock.calls.length;
  processor.destroy();
  assert.equal(mock.calls.length, callCountAfterDestroy);
  assert.deepEqual(processor.executeCommandBuffer(frame2Stream), { frameCount: 3, presents: 0 });
});

// ---------------------------------------------------------------------------
// 2. Boolean enable* auto-construction & mapped-mode staging pool + init flush
// ---------------------------------------------------------------------------

test('G23 enable* options auto-construct all 6 subsystems and handle mapped-mode staging + init flush', async () => {
  const mock = createFullMockWebGpu({ timestampQuery: true });

  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
    enableStagingPool: true,
    stagingPoolOptions: { chunkSize: 2048, framesInFlight: 2, mode: 'mapped' },
    enableUniformRing: true,
    uniformRingOptions: { byteSize: 2048, framesInFlight: 2 },
    enablePassTimer: true,
    enableReflectionRegistry: true,
    enableResourceRegistry: true,
    resourceRegistryOptions: { framesInFlight: 2 },
    enableRtPool: true,
    rtPoolOptions: { idleFrameTtl: 2 },
  });

  assert.ok(processor.stagingPool instanceof GpuStagingPool);
  assert.ok(processor.uniformRing instanceof DynamicUniformRing);
  assert.ok(processor.passTimer instanceof GpuPassTimer);
  assert.ok(processor.reflectionRegistry instanceof ReflectionRegistry);
  assert.ok(processor.resourceRegistry instanceof GpuResourceRegistry);
  assert.ok(processor.rtPool instanceof RenderTargetPool);

  // Out-of-frame init batch (no Present): writes a buffer through mapped staging pool.
  const initBytes = new Uint8Array([10, 20, 30, 40, 50, 60, 70, 80]);
  const initStream = new BinaryCommandBuilder()
    .createBuffer(1, BUFFER_USAGE.VERTEX | BUFFER_USAGE.COPY_DST, 16)
    .writeBuffer(1, 0, initBytes)
    .finish();
  processor.executeCommandBuffer(initStream);

  // Frame 0 BeginFrame must flush the pending init staging encoder (unmapping first!) before rotating epochs.
  const frame0 = new BinaryCommandBuilder()
    .beginFrame()
    .clearColor(0.5, 0.5, 0.5, 1.0)
    .present()
    .finish();
  processor.executeCommandBuffer(frame0);

  const dstBuf = processor.buffers.get(1);
  assert.deepEqual(dstBuf._bytes.subarray(0, 8), initBytes);
  const unmapCalls = mock.calls.filter((c) => c.op === 'buffer.unmap');
  assert.ok(unmapCalls.length >= 1, 'Mapped staging chunks must be unmapped before queue.submit');

  // Verify RT pool TTL eviction across frames (idleFrameTtl = 2)
  const rt = processor.acquireRenderTarget({ width: 64, height: 64, format: 'rgba16float' }, 'hdr');
  processor.releaseRenderTarget(rt);
  for (let f = 0; f < 4; f += 1) {
    processor.executeCommandBuffer(new BinaryCommandBuilder().beginFrame().present().finish());
  }
  assert.equal(processor.rtPool.getStats().evictions, 1);
  assert.equal(rt.texture.destroyed, true);

  processor.destroy();
  assert.equal(processor.getStats().destroyed, true);
});

// ---------------------------------------------------------------------------
// 3. Graceful degradation when timestamp-query is absent & lazy reflection/RT init
// ---------------------------------------------------------------------------

test('G23 degrades cleanly without timestamp-query and lazily initializes ReflectionRegistry & RenderTargetPool', async () => {
  const mock = createFullMockWebGpu({ timestampQuery: false });
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    enablePassTimer: true,
    enableResourceRegistry: true,
  });

  // PassTimer should degrade to zero-overhead no-op without creating any QuerySet
  assert.equal(processor.passTimer.supported, false);
  assert.equal(mock.createdQuerySets.length, 0);

  // Lazily create ReflectionRegistry via loadReflectionManifest()
  const vsWgsl = '@vertex fn custom_vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }';
  const csWgsl = '@compute @workgroup_size(4, 1, 1) fn custom_cs() {}';
  const manifest = buildS6ReflectionManifest(vsWgsl, '@fragment fn fs_main() {}', csWgsl);
  // Adjust entry_point on VS/CS to verify reflected entryPoint propagation
  const vsHash = formatShaderHash(fnv1a64(vsWgsl));
  const csHash = formatShaderHash(fnv1a64(csWgsl));
  manifest.shaders[vsHash].entry_point = 'custom_vs';
  manifest.shaders[csHash].entry_point = 'custom_cs';
  processor.loadReflectionManifest(manifest);

  // Lazily create RenderTargetPool via acquireRenderTarget()
  const rt = processor.acquireRenderTarget({ width: 32, height: 32, format: 'depth24plus' });
  assert.equal(rt.reused, false);
  assert.equal(processor.releaseRenderTarget(rt), true);

  const stream = new BinaryCommandBuilder()
    .beginFrame()
    .createShaderModule(1, SHADER_STAGE.VERTEX, vsWgsl)
    .createShaderModule(2, SHADER_STAGE.COMPUTE, csWgsl)
    .createRenderPipeline(3, 1, 0, 0)
    .createComputePipeline(4, 2)
    .setComputePipeline(4)
    .dispatchCompute(1, 1, 1)
    .setPipeline(3)
    .draw(3, 1, 0, 0)
    .present()
    .finish();

  processor.executeCommandBuffer(stream);

  const rpCall = mock.calls.find((c) => c.op === 'createRenderPipeline');
  const cpCall = mock.calls.find((c) => c.op === 'createComputePipeline');
  assert.equal(rpCall.desc.vertex.entryPoint, 'custom_vs');
  assert.equal(cpCall.desc.compute.entryPoint, 'custom_cs');

  // Verify no timestampWrites were attached on passes
  const beginRender = mock.calls.find((c) => c.op === 'beginRenderPass');
  const beginCompute = mock.calls.find((c) => c.op === 'beginComputePass');
  assert.equal(beginRender.desc.timestampWrites, undefined);
  assert.equal(beginCompute.desc, undefined);
  assert.equal(await processor.collectPassTimings(), null);

  // Verify resource overwrite retires previous handle cleanly without immediate destroy
  const shaderBytes =
    new TextEncoder().encode(vsWgsl).byteLength + new TextEncoder().encode(csWgsl).byteLength;
  const bufOverwriteStream = new BinaryCommandBuilder()
    .createBuffer(50, BUFFER_USAGE.UNIFORM, 64)
    .createBuffer(50, BUFFER_USAGE.UNIFORM, 128)
    .finish();
  processor.executeCommandBuffer(bufOverwriteStream);
  const regStats = processor.resourceRegistry.getStats();
  assert.equal(regStats.countsByKind.buffer, 1);
  assert.equal(regStats.activeBytes, 128 + shaderBytes);
  assert.equal(regStats.pendingRetireCount, 1);
  assert.equal(regStats.pendingRetireBytes, 64);

  processor.destroy();
});

// ---------------------------------------------------------------------------
// 4. Worker message handler integration (loadReflectionManifest, resize, stats, destroy)
// ---------------------------------------------------------------------------

test('G23 createWgpuWorkerHandler wires loadReflectionManifest, resize -> rtPool.invalidateAll, stats, and destroy', async () => {
  const mock = createFullMockWebGpu({ timestampQuery: true });
  const posted = [];
  const canvas = { width: 640, height: 360 };

  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    enableStagingPool: true,
    enableUniformRing: true,
    enableResourceRegistry: true,
    enableRtPool: true,
  });

  const handler = createWgpuWorkerHandler({
    canvas,
    processor,
    postMessage: (m) => posted.push(m),
  });

  const vsWgsl = '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }';
  const fsWgsl = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }';
  const csWgsl = '@compute @workgroup_size(1, 1, 1) fn cs_main() {}';
  const manifest = buildS6ReflectionManifest(vsWgsl, fsWgsl, csWgsl);

  await handler.handleMessage({ id: 1, loadReflectionManifest: manifest });
  assert.equal(posted[0].id, 1);
  assert.equal(posted[0].reflectionLoaded, true);
  assert.equal(posted[0].shaderCount, 3);

  // Acquire and release an RT, then send a resize message -> should invalidate idle RTs
  const rt = processor.acquireRenderTarget({ width: 640, height: 360, format: 'rgba8unorm' });
  processor.releaseRenderTarget(rt);
  assert.equal(processor.rtPool.getStats().idleCount, 1);

  await handler.handleMessage({ resize: true, width: 1280, height: 720 });
  assert.equal(canvas.width, 1280);
  assert.equal(canvas.height, 720);
  assert.equal(processor.rtPool.getStats().idleCount, 0);
  assert.equal(rt.texture.destroyed, true);

  // Execute a frame and query stats
  const frameBuf = new BinaryCommandBuilder().beginFrame().present().finish();
  await handler.handleMessage({ cmdBuffer: frameBuf });
  await handler.handleMessage({ id: 2, stats: true });
  const statsMsg = posted.find((m) => m.id === 2);
  assert.ok(statsMsg && statsMsg.stats);
  assert.equal(statsMsg.stats.frameCount, 1);
  assert.equal(statsMsg.stats.reflectionShaderCount, 3);
  assert.ok(statsMsg.stats.stagingPool);
  assert.ok(statsMsg.stats.uniformRing);
  assert.ok(statsMsg.stats.resourceRegistry);
  assert.ok(statsMsg.stats.rtPool);

  // Destroy via worker message
  await handler.handleMessage({ id: 3, destroy: true });
  const destroyMsg = posted.find((m) => m.id === 3);
  assert.equal(destroyMsg.destroyed, true);
  assert.equal(processor.destroyed, true);
});

// ---------------------------------------------------------------------------
// 5. W121: W112 swapchain offscreen blit fallback & W109 ranged pack threshold
// ---------------------------------------------------------------------------

function buildSwapchainBeginRenderPassPayload(formatName, clearRgba = [0.1, 0.2, 0.3, 1.0]) {
  const formatCode = RS_TEXTURE_FORMAT_NAMES.indexOf(formatName);
  assert.ok(formatCode > 0, `Unknown render_state_packets format: ${formatName}`);
  const buf = new ArrayBuffer(368);
  const view = new DataView(buf);
  view.setUint32(0, 1, true); // colorCount = 1
  const o = 8;
  view.setUint32(o, SWAPCHAIN_TEXTURE_ID, true);
  view.setUint8(o + 4, 1); // loadOp = 'clear'
  view.setUint8(o + 5, 1); // storeOp = 'store'
  view.setUint8(o + 8, formatCode); // view.format
  view.setFloat32(o + 24, clearRgba[0], true);
  view.setFloat32(o + 28, clearRgba[1], true);
  view.setFloat32(o + 32, clearRgba[2], true);
  view.setFloat32(o + 36, clearRgba[3], true);
  return new Uint8Array(buf);
}

function createBlitAwareMockWebGpu() {
  const base = createFullMockWebGpu({ timestampQuery: false });
  const configuredCalls = [];
  const canvasViews = [];
  let canvasCounter = 0;

  const ctx = {
    configure(cfg) {
      configuredCalls.push({ ...cfg });
    },
    unconfigure() {},
    getCurrentTexture() {
      const view = { kind: 'GPUTextureView', id: `canvas-view-${canvasCounter++}` };
      canvasViews.push(view);
      return {
        width: 1280,
        height: 720,
        createView() {
          return view;
        },
      };
    },
  };

  const canvas = {
    width: 1280,
    height: 720,
    getContext(kind) {
      return kind === 'webgpu' ? ctx : null;
    },
  };

  return {
    ...base,
    ctx,
    canvas,
    configuredCalls,
    canvasViews,
  };
}

test('W121 non-canvas swapchain format (rgb10a2unorm) renders into intermediate view and blits on Present', () => {
  const mock = createBlitAwareMockWebGpu();
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

  const vsWgsl = '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }';
  const fsWgsl = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }';

  const builder = new BinaryCommandBuilder();
  builder
    .beginFrame()
    .createShaderModule(1, SHADER_STAGE.VERTEX, vsWgsl)
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, fsWgsl)
    .createRenderPipeline(3, 1, 2, 0);
  builder._pushPacket(OPCODES.BEGIN_RENDER_PASS, buildSwapchainBeginRenderPassPayload('rgb10a2unorm'));
  builder.setPipeline(3).draw(6, 1, 0, 0);
  builder._pushPacket(OPCODES.END_RENDER_PASS, new Uint8Array(0));
  builder.present();

  const res = processor.executeCommandBuffer(builder.finish());
  assert.deepEqual(res, { frameCount: 1, presents: 1 });

  // Swapchain switched into W112 blit fallback while keeping canvas configured in a valid format.
  assert.equal(swapchain.usesBlitFallback, true);
  assert.equal(swapchain.streamFormat, 'rgb10a2unorm');
  assert.equal(swapchain.format, 'bgra8unorm');
  assert.equal(swapchain.blitCount, 1);
  assert.ok(swapchain.intermediateTexture, 'Expected offscreen intermediate texture');
  assert.equal(swapchain.intermediateTexture.desc.format, 'rgb10a2unorm');

  // Verify scene render pass bound the intermediate texture view and Present ran the blit pass (draw(3)) into the canvas view.
  const passCalls = mock.calls.filter((c) => c.op === 'beginRenderPass');
  assert.equal(passCalls.length, 2, 'Expected 1 scene render pass + 1 blit pass');
  assert.equal(
    passCalls[0].desc.colorAttachments[0].view,
    swapchain.intermediateTextureView,
    'SWAPCHAIN_TEXTURE_ID pass must target intermediate texture view'
  );
  assert.equal(passCalls[1].desc.label, 'playgta6-swapchain-blit-pass');
  assert.equal(
    passCalls[1].desc.colorAttachments[0].view,
    mock.canvasViews[0],
    'Blit pass must target the acquired canvas view'
  );

  const drawCalls = mock.calls.filter((c) => c.op === 'renderPass.draw');
  assert.equal(drawCalls.length, 2);
  assert.equal(drawCalls[0].vertexCount, 6);
  assert.equal(drawCalls[1].vertexCount, 3, 'Fullscreen blit pass must issue draw(3)');

  const finishIdx = mock.calls.findIndex((c) => c.op === 'encoder.finish');
  const submitIdx = mock.calls.findIndex((c) => c.op === 'queue.submit');
  const blitPassIdx = mock.calls.findIndex(
    (c) => c.op === 'beginRenderPass' && c.desc && c.desc.label === 'playgta6-swapchain-blit-pass'
  );
  assert.ok(blitPassIdx > 0 && blitPassIdx < finishIdx && finishIdx < submitIdx);

  // Telemetry & stats surface swapchainFormat / usesBlitFallback / blitCount cleanly.
  const stats = processor.getStats();
  assert.equal(stats.swapchainFormat, 'rgb10a2unorm');
  assert.equal(stats.usesBlitFallback, true);
  assert.equal(stats.blitCount, 1);

  const snap = buildGpuTelemetrySnapshot(processor, { timestamp: 12345 });
  assert.deepEqual(snap.swapchainStats, {
    swapchainFormat: 'rgb10a2unorm',
    canvasFormat: 'bgra8unorm',
    usesBlitFallback: true,
    blitCount: 1,
  });
  assert.deepEqual(snap.textureStats, {
    bc3dFallbacks: 0,
    bc3dFallbackUploadsSkipped: 0,
  });

  processor.destroy();
  swapchain.destroy();
});

test('W121 direct canvas format (rgba8unorm) stays direct with zero blit overhead (blitCount === 0)', () => {
  const mock = createBlitAwareMockWebGpu();
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

  const builder = new BinaryCommandBuilder();
  builder.beginFrame();
  builder._pushPacket(OPCODES.BEGIN_RENDER_PASS, buildSwapchainBeginRenderPassPayload('rgba8unorm'));
  builder._pushPacket(OPCODES.END_RENDER_PASS, new Uint8Array(0));
  builder.present();

  const res = processor.executeCommandBuffer(builder.finish());
  assert.deepEqual(res, { frameCount: 1, presents: 1 });

  assert.equal(swapchain.usesBlitFallback, false);
  assert.equal(swapchain.streamFormat, 'rgba8unorm');
  assert.equal(swapchain.format, 'rgba8unorm');
  assert.equal(swapchain.blitCount, 0);
  assert.equal(swapchain.intermediateTexture, null);

  const passCalls = mock.calls.filter((c) => c.op === 'beginRenderPass');
  assert.equal(passCalls.length, 1, 'Direct canvas format must not record a blit pass');
  assert.equal(passCalls[0].desc.colorAttachments[0].view, mock.canvasViews[0]);

  const snap = buildGpuTelemetrySnapshot(processor, { timestamp: 22222 });
  assert.deepEqual(snap.swapchainStats, {
    swapchainFormat: 'rgba8unorm',
    canvasFormat: 'rgba8unorm',
    usesBlitFallback: false,
    blitCount: 0,
  });

  processor.destroy();
  swapchain.destroy();
});

test('W121 worker init and loadShaderPack forward streamThresholdBytes to WgslPackCache.loadPackUrl', async () => {
  const vsWgsl = '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }\n';
  const fsWgsl = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.4, 0.7, 1.0, 1.0); }\n';
  const vsHash = wgpkFormatHashHex(wgpkFnv1a64(vsWgsl));
  const fsHash = wgpkFormatHashHex(wgpkFnv1a64(fsWgsl));
  const packBuffer = buildWgslPackBuffer([
    { hash: vsHash, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: vsWgsl },
    { hash: fsHash, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: fsWgsl },
  ]);
  const packBytes = new Uint8Array(packBuffer);

  // 1. Spy cache verifies exact options forwarding from init + loadShaderPack messages.
  const recordedCalls = [];
  const spyCache = {
    async loadPackUrl(url, opts) {
      recordedCalls.push({ url, opts });
      return { size: 2 };
    },
    getStats() {
      return { mode: 'ranged', packCount: 1, shaderCount: 2, rangeFetches: 1, cachedWgslBytes: 0 };
    },
  };
  const mockA = createFullMockWebGpu({ timestampQuery: false });
  const procA = new WgpuCommandProcessor(mockA.device, mockA.ctx, { wgslPackCache: spyCache });
  const dummyFetch = async () => ({ ok: true, status: 200, arrayBuffer: async () => packBuffer.slice(0) });
  const postedA = [];
  const handlerA = createWgpuWorkerHandler({
    processor: procA,
    fetch: dummyFetch,
    postMessage: (m) => postedA.push(m),
  });

  await handlerA.handleMessage({
    init: true,
    shaderPackUrl: '/shaders/game.wgsl.pack',
    streamThresholdBytes: 128,
  });
  assert.equal(recordedCalls.length, 1);
  assert.equal(recordedCalls[0].url, '/shaders/game.wgsl.pack');
  assert.equal(recordedCalls[0].opts.streamThresholdBytes, 128);
  assert.equal(recordedCalls[0].opts.fetch, dummyFetch);

  // loadShaderPack message with streamShaders: true forwards default threshold (64 MiB).
  await handlerA.handleMessage({
    id: 99,
    type: 'loadShaderPack',
    url: '/shaders/extra.wgsl.pack',
    streamShaders: true,
  });
  assert.equal(recordedCalls.length, 2);
  assert.equal(recordedCalls[1].url, '/shaders/extra.wgsl.pack');
  assert.equal(recordedCalls[1].opts.streamThresholdBytes, 64 * 1024 * 1024);
  const loadReply = postedA.find((m) => m.id === 99);
  assert.equal(loadReply.shaderPackLoaded, true);
  assert.equal(loadReply.mode, 'ranged');

  // 2. Real WgslPackCache + HTTP Range mock proves ranged mode activates when pack > streamThresholdBytes
  //    and stays in full-buffer mode when streamThresholdBytes is omitted.
  const createRangeFetch = () => {
    const requests = [];
    const fetchImpl = async (url, init = {}) => {
      const method = (init && init.method ? String(init.method) : 'GET').toUpperCase();
      const headers = init && init.headers ? init.headers : {};
      const rangeHeader = headers.Range || headers.range || null;
      requests.push({ url, method, rangeHeader });

      if (method === 'HEAD') {
        return {
          ok: true,
          status: 200,
          headers: new Map([
            ['content-length', String(packBytes.byteLength)],
            ['accept-ranges', 'bytes'],
          ]),
        };
      }
      if (rangeHeader) {
        const m = /^bytes=(\d+)-(\d+)$/.exec(rangeHeader);
        assert.ok(m, `Unexpected Range header: ${rangeHeader}`);
        const start = Number(m[1]);
        const end = Math.min(Number(m[2]), packBytes.byteLength - 1);
        const slice = packBytes.slice(start, end + 1);
        return {
          ok: true,
          status: 206,
          headers: new Map([
            ['content-range', `bytes ${start}-${end}/${packBytes.byteLength}`],
            ['content-length', String(slice.byteLength)],
          ]),
          async arrayBuffer() {
            return slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength);
          },
        };
      }
      return {
        ok: true,
        status: 200,
        headers: new Map([['content-length', String(packBytes.byteLength)]]),
        async arrayBuffer() {
          return packBuffer.slice(0);
        },
      };
    };
    return { requests, fetchImpl };
  };

  // Ranged mode when pack exceeds streamThresholdBytes.
  const rangedNet = createRangeFetch();
  const realRangedCache = new WgslPackCache();
  const mockRanged = createFullMockWebGpu({ timestampQuery: false });
  const procRanged = new WgpuCommandProcessor(mockRanged.device, mockRanged.ctx, {
    wgslPackCache: realRangedCache,
  });
  const handlerRanged = createWgpuWorkerHandler({
    processor: procRanged,
    fetch: rangedNet.fetchImpl,
  });
  await handlerRanged.handleMessage({
    init: true,
    shaderPackUrl: '/shaders/ranged.wgsl.pack',
    streamThresholdBytes: 32,
  });
  const rangedSnap = buildGpuTelemetrySnapshot(procRanged, { timestamp: 33333 });
  assert.equal(rangedSnap.shaderPackStats.mode, 'ranged');
  assert.equal(rangedSnap.shaderPackStats.shaderCount, 2);
  assert.ok(rangedSnap.shaderPackStats.rangeFetches >= 1);
  assert.ok(
    rangedNet.requests.some((r) => r.rangeHeader !== null),
    'Expected HTTP Range request when pack size exceeds streamThresholdBytes'
  );

  // Default full-buffer mode unchanged when streamThresholdBytes is omitted.
  const fullNet = createRangeFetch();
  const realFullCache = new WgslPackCache();
  const mockFull = createFullMockWebGpu({ timestampQuery: false });
  const procFull = new WgpuCommandProcessor(mockFull.device, mockFull.ctx, {
    wgslPackCache: realFullCache,
  });
  const handlerFull = createWgpuWorkerHandler({
    processor: procFull,
    fetch: fullNet.fetchImpl,
  });
  await handlerFull.handleMessage({
    init: true,
    shaderPackUrl: '/shaders/full.wgsl.pack',
  });
  const fullSnap = buildGpuTelemetrySnapshot(procFull, { timestamp: 44444 });
  assert.equal(fullSnap.shaderPackStats.mode, 'buffer');
  assert.equal(fullSnap.shaderPackStats.shaderCount, 2);
  assert.equal(fullSnap.shaderPackStats.rangeFetches, 0);
  assert.equal(
    fullNet.requests.filter((r) => r.rangeHeader !== null).length,
    0,
    'Omitting streamThresholdBytes must preserve default full-buffer GET'
  );

  procA.destroy();
  procRanged.destroy();
  procFull.destroy();
});

test('W121 telemetry surfaces TexturePackets bc3dFallbacks and bc3dFallbackUploadsSkipped', () => {
  const mock = createFullMockWebGpu({ timestampQuery: false });
  // Device has BC compression but NOT texture-compression-bc-sliced-3d.
  mock.device.features = new Set(['texture-compression-bc']);

  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, { format: 'bgra8unorm' });
  installRenderStatePackets(processor);
  installTexturePackets(processor);

  // Build CREATE_TEXTURE2 payload for a 3D BC7 texture (4x4x4) -> triggers W108 fallback.
  const tex2Buf = new ArrayBuffer(40);
  const tex2View = new DataView(tex2Buf);
  tex2View.setUint32(0, 77, true); // id
  tex2View.setUint32(4, 4, true); // width
  tex2View.setUint32(8, 4, true); // height
  tex2View.setUint32(12, 4, true); // depthOrArrayLayers
  tex2View.setUint32(16, 1, true); // mipLevelCount
  tex2View.setUint32(20, 1, true); // sampleCount
  tex2View.setUint32(24, TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST, true); // usage
  tex2View.setUint8(28, 2); // dimension = '3d'
  tex2View.setUint8(29, RS_TEXTURE_FORMAT_NAMES.indexOf('bc7-rgba-unorm')); // format
  tex2View.setUint8(30, 0); // viewFormatCount

  const builder = new BinaryCommandBuilder();
  builder._pushPacket(CREATE_TEXTURE2, new Uint8Array(tex2Buf));
  builder.writeTexture(77, 0, 0, 0, 0, 4, 4, 4, 16, 4, new Uint8Array(64));

  processor.executeCommandBuffer(builder.finish());

  const stats = processor.getStats();
  assert.equal(stats.bc3dFallbacks, 1);
  assert.equal(stats.bc3dFallbackUploadsSkipped, 1);

  const snap = buildGpuTelemetrySnapshot(processor, { timestamp: 55555 });
  assert.deepEqual(snap.textureStats, {
    bc3dFallbacks: 1,
    bc3dFallbackUploadsSkipped: 1,
  });

  processor.destroy();
});

// ---------------------------------------------------------------------------
// 6. W136: On-demand ranged WGSL pack shader resolution via getWgslAsync(hash)
// ---------------------------------------------------------------------------

function createControllableRangeFetch(packBuffer, options = {}) {
  const packBytes = new Uint8Array(packBuffer);
  const requests = [];
  const deferredRanges = [];
  const { deferDataRanges = false, failDataRangeStatus = null, dataStartOffset = 0 } = options;

  const fetchImpl = async (url, init = {}) => {
    const method = (init && init.method ? String(init.method) : 'GET').toUpperCase();
    const headers = init && init.headers ? init.headers : {};
    const rangeHeader = headers.Range || headers.range || null;
    const req = { url, method, rangeHeader, start: null, end: null };
    requests.push(req);

    if (method === 'HEAD') {
      return {
        ok: true,
        status: 200,
        headers: new Map([
          ['content-length', String(packBytes.byteLength)],
          ['accept-ranges', 'bytes'],
        ]),
      };
    }

    if (rangeHeader) {
      const m = /^bytes=(\d+)-(\d+)$/.exec(rangeHeader);
      assert.ok(m, `Unexpected Range header: ${rangeHeader}`);
      const start = Number(m[1]);
      const end = Math.min(Number(m[2]), packBytes.byteLength - 1);
      req.start = start;
      req.end = end;

      const isDataRange = dataStartOffset > 0 && start >= dataStartOffset;
      if (isDataRange && deferDataRanges) {
        await new Promise((resolve, reject) => {
          deferredRanges.push({ req, start, end, resolve, reject });
        });
      }

      if (isDataRange && failDataRangeStatus !== null) {
        return {
          ok: false,
          status: failDataRangeStatus,
          statusText: 'Internal Server Error',
          headers: new Map(),
          async arrayBuffer() {
            return new ArrayBuffer(0);
          },
        };
      }

      const slice = packBytes.slice(start, end + 1);
      return {
        ok: true,
        status: 206,
        headers: new Map([
          ['content-range', `bytes ${start}-${end}/${packBytes.byteLength}`],
          ['content-length', String(slice.byteLength)],
        ]),
        async arrayBuffer() {
          return slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength);
        },
      };
    }

    return {
      ok: true,
      status: 200,
      headers: new Map([['content-length', String(packBytes.byteLength)]]),
      async arrayBuffer() {
        return packBuffer.slice(0);
      },
    };
  };

  return {
    requests,
    deferredRanges,
    fetchImpl,
    resolveNextDeferred() {
      const next = deferredRanges.shift();
      assert.ok(next, 'Expected a deferred range fetch to resolve');
      next.resolve();
      return next;
    },
    resolveAllDeferred() {
      while (deferredRanges.length > 0) {
        deferredRanges.shift().resolve();
      }
    },
  };
}

test('W136 ranged .wgsl.pack resolves shader on demand via HTTP 206 Range, swaps real GPUShaderModule, deduplicates in-flight fetches, and caches LRU hits', async () => {
  const vsWgsl = '@vertex fn vs_ranged() -> @builtin(position) vec4<f32> { return vec4<f32>(0.25, 0.5, 0.0, 1.0); }\n';
  const fsWgsl = '@fragment fn fs_ranged() -> @location(0) vec4<f32> { return vec4<f32>(0.2, 0.8, 0.4, 1.0); }\n';
  const csWgsl = '@compute @workgroup_size(8, 1, 1) fn cs_unused() {}\n';
  const vsHash = wgpkFormatHashHex(wgpkFnv1a64(vsWgsl));
  const fsHash = wgpkFormatHashHex(wgpkFnv1a64(fsWgsl));
  const csHash = wgpkFormatHashHex(wgpkFnv1a64(csWgsl));
  const unknownHash = '0x0123456789abcdef';

  const packBuffer = buildWgslPackBuffer([
    { hash: vsHash, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: vsWgsl },
    { hash: fsHash, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: fsWgsl },
    { hash: csHash, stage: WGPK_SHADER_STAGE.COMPUTE, wgsl: csWgsl },
  ]);
  const dataStart = 16 + 3 * 20; // WGPK header (16B) + 3 entries (20B each)
  const vsByteLen = new TextEncoder().encode(vsWgsl).byteLength;

  const net = createControllableRangeFetch(packBuffer, {
    deferDataRanges: true,
    dataStartOffset: dataStart,
  });
  const packCache = new WgslPackCache();
  const mock = createFullMockWebGpu({ timestampQuery: false });
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
    wgslPackCache: packCache,
    enableResourceRegistry: true,
  });

  // Load pack in ranged index-only mode (`streamThresholdBytes: 1`, `initialRangeBytes: dataStart` so only header+table are fetched).
  await processor.loadShaderPackUrl('/shaders/ranged-w136.wgsl.pack', {
    fetch: net.fetchImpl,
    streamThresholdBytes: 1,
    initialRangeBytes: dataStart,
  });
  assert.equal(packCache.getStats().mode, 'ranged');
  assert.equal(net.requests.filter((r) => r.method === 'GET' && r.rangeHeader === null).length, 0);
  const requestsAfterIndex = net.requests.length;

  // Issue CreateShaderModule(1, VERTEX, vsHash) twice while in-flight to verify deduplication for (shaderId, hash).
  const stream = new BinaryCommandBuilder()
    .createShaderModule(1, SHADER_STAGE.VERTEX, vsHash)
    .createShaderModule(1, SHADER_STAGE.VERTEX, vsHash)
    .createShaderModule(99, SHADER_STAGE.FRAGMENT, unknownHash)
    .finish();
  processor.executeCommandBuffer(stream);

  // Synchronously installed fallback module while range fetch is in flight; unknown hash did NOT trigger ranged fetch.
  assert.equal(processor.shaderModules.get(1).source, 'builtin');
  assert.equal(processor.getStats().rangedShaderFetches, 1);
  assert.equal(processor.getStats().rangedShaderReady, 0);
  assert.equal(processor.getStats().rangedShaderErrors, 0);
  assert.equal(net.deferredRanges.length, 1);

  // Verify the HTTP 206 Range request targets ONLY vsWgsl's exact byte range.
  const sliceReq = net.deferredRanges[0];
  assert.equal(sliceReq.start, dataStart);
  assert.equal(sliceReq.end, dataStart + vsByteLen - 1);

  // Release the deferred HTTP 206 Range fetch and wait for async compilation + swap.
  net.resolveNextDeferred();
  await processor.flushPendingRangedShaders();

  const updated = processor.shaderModules.get(1);
  assert.ok(updated && updated.module, 'Expected real compiled GPUShaderModule swapped into shaderModules');
  assert.equal(updated.source, 'pack');
  assert.equal(updated.code, vsWgsl);
  assert.equal(updated.module.code, vsWgsl);
  assert.equal(updated.resolvedHash, vsHash);
  assert.equal(updated.entryPoint, 'vs_ranged');
  assert.equal(updated.fallbackProfile, null);

  // Telemetry counters on both getStats() and buildGpuTelemetrySnapshot().shaderPackStats.
  const stats = processor.getStats();
  assert.equal(stats.rangedShaderFetches, 1);
  assert.equal(stats.rangedShaderReady, 1);
  assert.equal(stats.rangedShaderErrors, 0);
  const snap = buildGpuTelemetrySnapshot(processor, { timestamp: 60001 });
  assert.equal(snap.shaderPackStats.mode, 'ranged');
  assert.equal(snap.shaderPackStats.rangedShaderFetches, 1);
  assert.equal(snap.shaderPackStats.rangedShaderReady, 1);
  assert.equal(snap.shaderPackStats.rangedShaderErrors, 0);

  // Only 1 data range request occurred (never fetched fsWgsl or csWgsl, never full 200 GET).
  assert.equal(net.requests.length, requestsAfterIndex + 1);
  assert.equal(net.requests.filter((r) => r.method === 'GET' && r.rangeHeader === null).length, 0);

  // Subsequent pipeline creation uses the real compiled GPUShaderModule (not fallback).
  const inlineFs = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }\n';
  const pipeStream = new BinaryCommandBuilder()
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, inlineFs)
    .createRenderPipeline(10, 1, 2, 0)
    .finish();
  processor.executeCommandBuffer(pipeStream);
  const rpCall = mock.calls.find((c) => c.op === 'createRenderPipeline');
  assert.ok(rpCall);
  assert.equal(rpCall.desc.vertex.module, updated.module);
  assert.equal(rpCall.desc.vertex.module.code, vsWgsl);

  // Subsequent CreateShaderModule for the same hash hits the synchronous LRU fast path (zero new fetches).
  const cachedStream = new BinaryCommandBuilder()
    .createShaderModule(10, SHADER_STAGE.VERTEX, vsHash)
    .finish();
  processor.executeCommandBuffer(cachedStream);
  assert.equal(processor.shaderModules.get(10).source, 'pack');
  assert.equal(processor.shaderModules.get(10).code, vsWgsl);
  assert.equal(processor.shaderModules.get(10).module.code, vsWgsl);
  assert.equal(processor.getStats().rangedShaderFetches, 1);
  assert.equal(processor.getStats().rangedShaderReady, 1);
  assert.equal(net.requests.length, requestsAfterIndex + 1);

  processor.destroy();
});

test('W136 destroying or recreating shaderId before async range fetch resolves prevents stale module overwrite', async () => {
  const wgslA = '@vertex fn vs_a() -> @builtin(position) vec4<f32> { return vec4<f32>(1.0, 0.0, 0.0, 1.0); }\n';
  const wgslB = '@vertex fn vs_b() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 1.0, 0.0, 1.0); }\n';
  const wgslC = '@vertex fn vs_c() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 1.0, 1.0); }\n';
  const hashA = wgpkFormatHashHex(wgpkFnv1a64(wgslA));
  const hashB = wgpkFormatHashHex(wgpkFnv1a64(wgslB));
  const hashC = wgpkFormatHashHex(wgpkFnv1a64(wgslC));

  const packBuffer = buildWgslPackBuffer([
    { hash: hashA, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: wgslA },
    { hash: hashB, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: wgslB },
    { hash: hashC, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: wgslC },
  ]);
  const dataStart = 16 + 3 * 20;

  const net = createControllableRangeFetch(packBuffer, {
    deferDataRanges: true,
    dataStartOffset: dataStart,
  });
  const packCache = new WgslPackCache();
  const mock = createFullMockWebGpu({ timestampQuery: false });
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
    wgslPackCache: packCache,
  });

  await processor.loadShaderPackUrl('/shaders/ranged-stale.wgsl.pack', {
    fetch: net.fetchImpl,
    streamThresholdBytes: 1,
    initialRangeBytes: dataStart,
  });

  // 1. Destroy shader 20 before its range fetch resolves -> stale module must NOT be installed.
  processor.executeCommandBuffer(
    new BinaryCommandBuilder()
      .createShaderModule(20, SHADER_STAGE.VERTEX, hashA)
      .destroyResource(20)
      .finish()
  );
  assert.equal(processor.shaderModules.has(20), false);
  assert.equal(processor.getStats().rangedShaderFetches, 1);

  net.resolveNextDeferred();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(processor.shaderModules.has(20), false);
  assert.equal(processor.getStats().rangedShaderReady, 0);
  assert.equal(processor.getStats().rangedShaderErrors, 0);

  // 2. Recreate shader 21 with inline WGSL before its range fetch resolves -> inline module preserved.
  const inlineOverride = '@vertex fn vs_inline_override() -> @builtin(position) vec4<f32> { return vec4<f32>(0.5); }\n';
  processor.executeCommandBuffer(
    new BinaryCommandBuilder()
      .createShaderModule(21, SHADER_STAGE.VERTEX, hashB)
      .createShaderModule(21, SHADER_STAGE.VERTEX, inlineOverride)
      .finish()
  );
  assert.equal(processor.shaderModules.get(21).source, 'inline');
  assert.equal(processor.shaderModules.get(21).code, inlineOverride);
  assert.equal(processor.getStats().rangedShaderFetches, 2);

  net.resolveNextDeferred();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(processor.shaderModules.get(21).source, 'inline');
  assert.equal(processor.shaderModules.get(21).code, inlineOverride);
  assert.equal(processor.getStats().rangedShaderReady, 0);
  assert.equal(processor.getStats().rangedShaderErrors, 0);

  // 3. Recreate shader 30 with a different ranged hash (hashC) while hashA-fresh is in flight ->
  //    even if the newer fetch resolves first and the older fetch resolves later, newer module wins.
  const packBuffer2 = buildWgslPackBuffer([
    { hash: hashA, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: wgslA },
    { hash: hashC, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: wgslC },
  ]);
  const dataStart2 = 16 + 2 * 20;
  const net2 = createControllableRangeFetch(packBuffer2, {
    deferDataRanges: true,
    dataStartOffset: dataStart2,
  });
  const packCache2 = new WgslPackCache();
  const mock2 = createFullMockWebGpu({ timestampQuery: false });
  const proc2 = new WgpuCommandProcessor(mock2.device, mock2.ctx, {
    format: 'bgra8unorm',
    wgslPackCache: packCache2,
  });
  await proc2.loadShaderPackUrl('/shaders/ranged-recreate.wgsl.pack', {
    fetch: net2.fetchImpl,
    streamThresholdBytes: 1,
    initialRangeBytes: dataStart2,
  });

  proc2.executeCommandBuffer(
    new BinaryCommandBuilder()
      .createShaderModule(30, SHADER_STAGE.VERTEX, hashA)
      .createShaderModule(30, SHADER_STAGE.VERTEX, hashC)
      .finish()
  );
  assert.equal(proc2.getStats().rangedShaderFetches, 2);
  assert.equal(net2.deferredRanges.length, 2);

  // Resolve the SECOND fetch (hashC) first, then resolve the FIRST stale fetch (hashA) afterwards.
  const staleFetch = net2.deferredRanges.shift();
  const newerFetch = net2.deferredRanges.shift();
  newerFetch.resolve();
  await proc2.flushPendingRangedShaders();
  assert.equal(proc2.shaderModules.get(30).code, wgslC);
  assert.equal(proc2.getStats().rangedShaderReady, 1);

  staleFetch.resolve();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(proc2.shaderModules.get(30).code, wgslC, 'Late stale resolution must not overwrite newer shader');
  assert.equal(proc2.getStats().rangedShaderReady, 1);
  assert.equal(proc2.getStats().rangedShaderErrors, 0);

  processor.destroy();
  proc2.destroy();
});

test('W136 ranged fetch error and compile error increment rangedShaderErrors cleanly without crashing', async () => {
  const vsWgsl = '@vertex fn vs_err() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }\n';
  const fsWgsl = '@fragment fn fs_compile_fail() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }\n';
  const vsHash = wgpkFormatHashHex(wgpkFnv1a64(vsWgsl));
  const fsHash = wgpkFormatHashHex(wgpkFnv1a64(fsWgsl));

  const packBuffer = buildWgslPackBuffer([
    { hash: vsHash, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: vsWgsl },
    { hash: fsHash, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: fsWgsl },
  ]);
  const dataStart = 16 + 2 * 20;

  // 1. HTTP 500 on data range slice increments rangedShaderErrors without crashing.
  const failNet = createControllableRangeFetch(packBuffer, {
    failDataRangeStatus: 500,
    dataStartOffset: dataStart,
  });
  const failCache = new WgslPackCache();
  const mockFail = createFullMockWebGpu({ timestampQuery: false });
  const procFail = new WgpuCommandProcessor(mockFail.device, mockFail.ctx, {
    format: 'bgra8unorm',
    wgslPackCache: failCache,
  });

  await procFail.loadShaderPackUrl('/shaders/ranged-err.wgsl.pack', {
    fetch: failNet.fetchImpl,
    streamThresholdBytes: 1,
    initialRangeBytes: dataStart,
  });

  procFail.executeCommandBuffer(
    new BinaryCommandBuilder()
      .createShaderModule(40, SHADER_STAGE.VERTEX, vsHash)
      .finish()
  );
  await procFail.flushPendingRangedShaders();

  assert.equal(procFail.getStats().rangedShaderFetches, 1);
  assert.equal(procFail.getStats().rangedShaderReady, 0);
  assert.equal(procFail.getStats().rangedShaderErrors, 1);
  assert.equal(procFail.shaderModules.get(40).source, 'builtin', 'Fallback shader stays intact on fetch error');

  const failSnap = buildGpuTelemetrySnapshot(procFail, { timestamp: 60002 });
  assert.equal(failSnap.shaderPackStats.rangedShaderFetches, 1);
  assert.equal(failSnap.shaderPackStats.rangedShaderReady, 0);
  assert.equal(failSnap.shaderPackStats.rangedShaderErrors, 1);

  // Worker remains healthy and continues rendering frames after a ranged fetch error.
  const frameRes = procFail.executeCommandBuffer(
    new BinaryCommandBuilder().beginFrame().clearColor(0.1, 0.2, 0.3, 1.0).present().finish()
  );
  assert.deepEqual(frameRes, { frameCount: 1, presents: 1 });

  // 2. Compile error during async GPUShaderModule creation increments rangedShaderErrors cleanly.
  const compileNet = createControllableRangeFetch(packBuffer, { dataStartOffset: dataStart });
  const compileCache = new WgslPackCache();
  const mockCompile = createFullMockWebGpu({ timestampQuery: false });
  const origCreateShaderModule = mockCompile.device.createShaderModule.bind(mockCompile.device);
  mockCompile.device.createShaderModule = (desc) => {
    if (desc && desc.code === fsWgsl) {
      throw new Error('Synthetic WGSL compile failure');
    }
    return origCreateShaderModule(desc);
  };
  const procCompile = new WgpuCommandProcessor(mockCompile.device, mockCompile.ctx, {
    format: 'bgra8unorm',
    wgslPackCache: compileCache,
  });
  await procCompile.loadShaderPackUrl('/shaders/ranged-compile-err.wgsl.pack', {
    fetch: compileNet.fetchImpl,
    streamThresholdBytes: 1,
    initialRangeBytes: dataStart,
  });

  procCompile.executeCommandBuffer(
    new BinaryCommandBuilder()
      .createShaderModule(41, SHADER_STAGE.FRAGMENT, fsHash)
      .finish()
  );
  await procCompile.flushPendingRangedShaders();

  assert.equal(procCompile.getStats().rangedShaderFetches, 1);
  assert.equal(procCompile.getStats().rangedShaderReady, 0);
  assert.equal(procCompile.getStats().rangedShaderErrors, 1);
  assert.equal(procCompile.shaderModules.get(41).source, 'builtin');

  procFail.destroy();
  procCompile.destroy();
});

test('W136 ranged pack resolves on demand cleanly when enableShaderFallback is false', async () => {
  const vsWgsl = '@vertex fn vs_no_fallback() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }\n';
  const vsHash = wgpkFormatHashHex(wgpkFnv1a64(vsWgsl));
  const packBuffer = buildWgslPackBuffer([
    { hash: vsHash, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: vsWgsl },
  ]);
  const dataStart = 16 + 20;

  const net = createControllableRangeFetch(packBuffer, {
    deferDataRanges: true,
    dataStartOffset: dataStart,
  });
  const packCache = new WgslPackCache();
  const mock = createFullMockWebGpu({ timestampQuery: false });
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
    enableShaderFallback: false,
    wgslPackCache: packCache,
  });

  await processor.loadShaderPackUrl('/shaders/ranged-nofb.wgsl.pack', {
    fetch: net.fetchImpl,
    streamThresholdBytes: 1,
    initialRangeBytes: dataStart,
  });

  processor.executeCommandBuffer(
    new BinaryCommandBuilder()
      .createShaderModule(50, SHADER_STAGE.VERTEX, vsHash)
      .finish()
  );

  // Must NOT compile the raw hash string as WGSL before range fetch completes.
  const shaderCallsBefore = mock.calls.filter((c) => c.op === 'createShaderModule');
  assert.equal(shaderCallsBefore.length, 0);
  assert.equal(processor.ensureShaderModule(50), null);
  assert.equal(processor.getStats().rangedShaderFetches, 1);

  net.resolveNextDeferred();
  await processor.flushPendingRangedShaders();

  const resolved = processor.shaderModules.get(50);
  assert.ok(resolved && resolved.module);
  assert.equal(resolved.source, 'pack');
  assert.equal(resolved.code, vsWgsl);
  assert.equal(resolved.module.code, vsWgsl);
  assert.equal(processor.getStats().rangedShaderFetches, 1);
  assert.equal(processor.getStats().rangedShaderReady, 1);
  assert.equal(processor.getStats().rangedShaderErrors, 0);

  processor.destroy();
});
