import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import {
  GPU_OPCODES,
  GPU_BUFFER_USAGE,
  GPU_TEXTURE_USAGE,
  GPU_TEXTURE_FORMAT,
  GPU_TEXTURE_DIMENSION,
  GPU_ADDRESS_MODE,
  GPU_FILTER_MODE,
  GPU_BIND_RESOURCE_TYPE,
  GPU_INDEX_FORMAT,
  GPU_SHADER_STAGE,
  GPU_PRIMITIVE_TOPOLOGY,
  DEFAULT_RESOURCE_IDS,
  UNIFORM_BYTE_SIZE,
  STORAGE_BYTE_SIZE,
  VERTEX_STRIDE_BYTES,
  DEFAULT_TARGET_FPS,
  DEFAULT_FRAME_BUDGET_MS,
  DEFAULT_NON_BLACK_THRESHOLD,
  SMOKE_VERTEX_WGSL,
  SMOKE_FRAGMENT_WGSL,
  SMOKE_COMPUTE_WGSL,
  SmokeCommandBuilder,
  SmokeSceneController,
  generateProceduralTexture,
  normalizeInputState,
  submitToRing,
  evaluateFramePacing,
  hasNonBlackRgbaPixels,
  summarizeDrawBatch,
} from '../web/demo/smoke_scene.js';

const require = createRequire(import.meta.url);
const { GpuRingBuffer } = require('../web/workers/gpu_ring.js');
const { WgpuCommandProcessor, OPCODES } = require('../web/workers/wgpu_worker.js');

function createMockWebGpu() {
  const calls = [];
  let nextHandle = 1;
  let swapchainCounter = 0;

  const ctx = {
    getCurrentTexture() {
      const id = `swapchain-view-${swapchainCounter++}`;
      const view = { kind: 'GPUTextureView', id };
      return {
        createView() {
          return view;
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
        writes: [],
        destroy() {
          this.destroyed = true;
          calls.push({ op: 'buffer.destroy', buffer });
        },
      };
      calls.push({ op: 'createBuffer', desc: { ...desc }, result: buffer });
      return buffer;
    },

    createTexture(desc) {
      const handle = nextHandle++;
      const texture = {
        kind: 'GPUTexture',
        handle,
        desc: { ...desc, size: { ...desc.size } },
        destroyed: false,
        createView() {
          const view = { kind: 'GPUTextureView', textureHandle: handle, handle: nextHandle++ };
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
          const layout = { kind: 'GPUBindGroupLayout', pipelineKind: 'render', pipelineHandle: this.handle, index };
          calls.push({ op: 'renderPipeline.getBindGroupLayout', pipeline: this, index, result: layout });
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
          const layout = { kind: 'GPUBindGroupLayout', pipelineKind: 'compute', pipelineHandle: this.handle, index };
          calls.push({ op: 'computePipeline.getBindGroupLayout', pipeline: this, index, result: layout });
          return layout;
        },
      };
      calls.push({ op: 'createComputePipeline', desc, result: pipeline });
      return pipeline;
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
              calls.push({ op: 'renderPass.draw', vertexCount, instanceCount, firstVertex, firstInstance });
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
              calls.push({ op: 'computePass.dispatchWorkgroups', x, y, z });
            },
            end() {
              calls.push({ op: 'computePass.end' });
            },
          };
        },
        finish() {
          calls.push({ op: 'encoder.finish', commandBuffer });
          return commandBuffer;
        },
      };
    },

    queue: {
      writeBuffer(buffer, dstOffset, data) {
        const copy = new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
        buffer.writes.push({ dstOffset, bytes: copy });
        calls.push({ op: 'queue.writeBuffer', buffer, dstOffset, data: copy });
      },
      writeTexture(destination, data, dataLayout, size) {
        const copy = new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
        calls.push({
          op: 'queue.writeTexture',
          destination: {
            texture: destination.texture,
            mipLevel: destination.mipLevel,
            origin: { ...destination.origin },
          },
          data: copy,
          dataLayout: { ...dataLayout },
          size: { ...size },
        });
      },
      submit(commandBuffers) {
        calls.push({ op: 'queue.submit', commandBuffers: [...commandBuffers] });
      },
    },
  };

  return { device, ctx, calls };
}

function decodePackets(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const packets = [];
  let offset = 0;
  while (offset + 8 <= view.byteLength) {
    const opcode = view.getUint16(offset, true);
    const flags = view.getUint16(offset + 2, true);
    const payloadBytes = view.getUint32(offset + 4, true);
    assert.ok(offset + 8 + payloadBytes <= view.byteLength, `Packet at ${offset} truncated`);
    packets.push({
      opcode,
      flags,
      payloadBytes,
      payload: new Uint8Array(bytes.buffer, bytes.byteOffset + offset + 8, payloadBytes),
    });
    offset += 8 + payloadBytes;
  }
  assert.equal(offset, view.byteLength, 'Stream must end cleanly on a packet boundary');
  return packets;
}

test('GPU_OPCODES match WgpuCommandProcessor OPCODES 0..21 verbatim', () => {
  assert.deepEqual(GPU_OPCODES, OPCODES);
});

test('SmokeCommandBuilder encodes all opcodes 0..21 with exact ABI payload sizes', () => {
  const builder = new SmokeCommandBuilder();
  builder
    .nop()
    .beginFrame()
    .clearColor(0.25, 0.5, 0.75, 1.0)
    .createBuffer(10, GPU_BUFFER_USAGE.VERTEX | GPU_BUFFER_USAGE.COPY_DST, 64)
    .writeBuffer(10, 4, new Uint8Array([1, 2, 3, 4]))
    .createShaderModule(11, GPU_SHADER_STAGE.VERTEX, SMOKE_VERTEX_WGSL)
    .draw(3, 2, 1, 4)
    .present()
    .createRenderPipeline(12, 11, 13, 12, GPU_PRIMITIVE_TOPOLOGY.TRIANGLE_STRIP, 7)
    .setPipeline(12)
    .setVertexBuffer(0, 10, 16)
    .destroyResource(10)
    .createTexture(
      14,
      8,
      8,
      1,
      1,
      GPU_TEXTURE_FORMAT.RGBA8_UNORM,
      GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.COPY_DST,
      GPU_TEXTURE_DIMENSION.DIM_2D
    )
    .writeTexture(14, 0, 0, 0, 0, 2, 2, 1, 8, 2, new Uint8Array(16))
    .createSampler(
      15,
      GPU_ADDRESS_MODE.REPEAT,
      GPU_ADDRESS_MODE.MIRROR_REPEAT,
      GPU_ADDRESS_MODE.CLAMP_TO_EDGE,
      GPU_FILTER_MODE.LINEAR,
      GPU_FILTER_MODE.NEAREST,
      GPU_FILTER_MODE.LINEAR,
      4
    )
    .createBindGroup(16, 12, 0, [
      { binding: 0, resourceType: GPU_BIND_RESOURCE_TYPE.BUFFER, resourceId: 10, offset: 0, size: 64 },
      { binding: 1, resourceType: GPU_BIND_RESOURCE_TYPE.TEXTURE_VIEW, resourceId: 14 },
      { binding: 2, resourceType: GPU_BIND_RESOURCE_TYPE.SAMPLER, resourceId: 15 },
    ])
    .setBindGroup(0, 16)
    .setIndexBuffer(10, GPU_INDEX_FORMAT.UINT32, 8)
    .drawIndexed(6, 2, 3, -4, 5)
    .createComputePipeline(17, 18, 9)
    .setComputePipeline(17)
    .dispatchCompute(2, 3, 4);

  assert.equal(builder.packetCount, 22);
  const bytes = builder.finish();
  assert.equal(bytes.byteLength, builder.byteLength);

  const packets = decodePackets(bytes);
  assert.equal(packets.length, 22);

  const expected = [
    [GPU_OPCODES.NOP, 0],
    [GPU_OPCODES.BEGIN_FRAME, 0],
    [GPU_OPCODES.CLEAR_COLOR, 16],
    [GPU_OPCODES.CREATE_BUFFER, 16],
    [GPU_OPCODES.WRITE_BUFFER, 16 + 4],
    [GPU_OPCODES.CREATE_SHADER_MODULE, 16 + new TextEncoder().encode(SMOKE_VERTEX_WGSL).byteLength],
    [GPU_OPCODES.DRAW, 16],
    [GPU_OPCODES.PRESENT, 0],
    [GPU_OPCODES.CREATE_RENDER_PIPELINE, 24],
    [GPU_OPCODES.SET_PIPELINE, 4],
    [GPU_OPCODES.SET_VERTEX_BUFFER, 16],
    [GPU_OPCODES.DESTROY_RESOURCE, 4],
    [GPU_OPCODES.CREATE_TEXTURE, 32],
    [GPU_OPCODES.WRITE_TEXTURE, 48 + 16],
    [GPU_OPCODES.CREATE_SAMPLER, 32],
    [GPU_OPCODES.CREATE_BIND_GROUP, 16 + 3 * 32],
    [GPU_OPCODES.SET_BIND_GROUP, 8],
    [GPU_OPCODES.SET_INDEX_BUFFER, 16],
    [GPU_OPCODES.DRAW_INDEXED, 20],
    [GPU_OPCODES.CREATE_COMPUTE_PIPELINE, 16],
    [GPU_OPCODES.SET_COMPUTE_PIPELINE, 4],
    [GPU_OPCODES.DISPATCH_COMPUTE, 12],
  ];

  for (let i = 0; i < expected.length; i++) {
    assert.equal(packets[i].opcode, expected[i][0], `Packet ${i} opcode mismatch`);
    assert.equal(packets[i].payloadBytes, expected[i][1], `Packet ${i} payload length mismatch`);
  }

  // Verify signed negative baseVertex in DrawIndexed payload (offset 12, i32)
  const drawIndexedView = new DataView(
    packets[18].payload.buffer,
    packets[18].payload.byteOffset,
    packets[18].payload.byteLength
  );
  assert.equal(drawIndexedView.getUint32(0, true), 6);
  assert.equal(drawIndexedView.getUint32(4, true), 2);
  assert.equal(drawIndexedView.getUint32(8, true), 3);
  assert.equal(drawIndexedView.getInt32(12, true), -4);
  assert.equal(drawIndexedView.getUint32(16, true), 5);
});

test('WGSL shaders contain valid entry points and matching bindings', () => {
  assert.match(SMOKE_VERTEX_WGSL, /@vertex[\s\S]*fn vs_main\(/);
  assert.match(SMOKE_VERTEX_WGSL, /@location\(0\)\s+position\s*:\s*vec3<f32>/);
  assert.match(SMOKE_VERTEX_WGSL, /@group\(0\)\s+@binding\(0\)\s+var<uniform>/);
  assert.match(SMOKE_VERTEX_WGSL, /@group\(0\)\s+@binding\(1\)\s+var<storage,\s*read>/);

  assert.match(SMOKE_FRAGMENT_WGSL, /@fragment[\s\S]*fn fs_main\(/);
  assert.match(SMOKE_FRAGMENT_WGSL, /@group\(0\)\s+@binding\(2\)\s+var\s+smoke_sampler\s*:\s*sampler/);
  assert.match(SMOKE_FRAGMENT_WGSL, /@group\(0\)\s+@binding\(3\)\s+var\s+smoke_texture\s*:\s*texture_2d<f32>/);
  assert.match(SMOKE_FRAGMENT_WGSL, /textureSample\(/);

  assert.match(SMOKE_COMPUTE_WGSL, /@compute\s+@workgroup_size\(1,\s*1,\s*1\)[\s\S]*fn cs_main\(/);
  assert.match(SMOKE_COMPUTE_WGSL, /@group\(0\)\s+@binding\(0\)\s+var<uniform>/);
  assert.match(SMOKE_COMPUTE_WGSL, /@group\(0\)\s+@binding\(1\)\s+var<storage,\s*read_write>/);
});

test('SmokeSceneController full lifecycle: init + 3 animated frames + teardown through real GpuRingBuffer -> WgpuCommandProcessor', () => {
  const { device, ctx, calls } = createMockWebGpu();
  const ring = GpuRingBuffer.create(65536);
  const processor = new WgpuCommandProcessor(device, ctx, { format: 'bgra8unorm' });
  processor.attachRing(ring);

  const controller = new SmokeSceneController({ textureSize: 64, aspect: 16 / 9 });
  const ids = controller.resourceIds;

  // --- 1. Init command buffer ---
  const initBytes = controller.buildInitCommandBuffer();
  assert.equal(controller.initialized, true);
  assert.ok(initBytes.byteLength > 16384, 'Init command buffer should contain 64x64 RGBA8 texture upload + resources');

  assert.equal(controller.submitToRing(ring, initBytes), true);
  assert.equal(ring.droppedPackets, 0);

  const initDrain = processor.drainRing();
  assert.equal(initDrain.bytesDrained, initBytes.byteLength);
  assert.equal(initDrain.presents, 0);
  assert.equal(ring.availableBytes(), 0);

  // Verify every GPU resource was created in WgpuCommandProcessor
  assert.equal(processor.textures.size, 1);
  assert.ok(processor.textures.has(ids.TEXTURE));
  assert.ok(processor.textureViews.has(ids.TEXTURE));
  assert.equal(processor.samplers.size, 1);
  assert.ok(processor.samplers.has(ids.SAMPLER));
  assert.equal(processor.buffers.size, 4);
  assert.ok(processor.buffers.has(ids.VERTEX_BUFFER));
  assert.ok(processor.buffers.has(ids.INDEX_BUFFER));
  assert.ok(processor.buffers.has(ids.UNIFORM_BUFFER));
  assert.ok(processor.buffers.has(ids.STORAGE_BUFFER));
  assert.equal(processor.shaderModules.size, 3);
  assert.ok(processor.shaderModules.has(ids.VERTEX_SHADER));
  assert.ok(processor.shaderModules.has(ids.FRAGMENT_SHADER));
  assert.ok(processor.shaderModules.has(ids.COMPUTE_SHADER));
  assert.equal(processor.pipelines.size, 1);
  assert.ok(processor.pipelines.has(ids.RENDER_PIPELINE));
  assert.equal(processor.computePipelines.size, 1);
  assert.ok(processor.computePipelines.has(ids.COMPUTE_PIPELINE));
  assert.equal(processor.bindGroups.size, 2);
  assert.ok(processor.bindGroups.has(ids.COMPUTE_BIND_GROUP));
  assert.ok(processor.bindGroups.has(ids.RENDER_BIND_GROUP));

  // Verify procedural texture upload parameters
  const writeTexCalls = calls.filter((c) => c.op === 'queue.writeTexture');
  assert.equal(writeTexCalls.length, 1);
  assert.equal(writeTexCalls[0].size.width, 64);
  assert.equal(writeTexCalls[0].size.height, 64);
  assert.equal(writeTexCalls[0].dataLayout.bytesPerRow, 256);
  assert.equal(writeTexCalls[0].dataLayout.rowsPerImage, 64);
  assert.equal(writeTexCalls[0].data.byteLength, 64 * 64 * 4);

  // Verify render pipeline vertex stride matches float32x3
  const renderPipeCall = calls.find((c) => c.op === 'createRenderPipeline');
  assert.equal(renderPipeCall.desc.vertex.entryPoint, 'vs_main');
  assert.equal(renderPipeCall.desc.fragment.entryPoint, 'fs_main');
  assert.equal(renderPipeCall.desc.vertex.buffers[0].arrayStride, VERTEX_STRIDE_BYTES);
  assert.equal(renderPipeCall.desc.vertex.buffers[0].attributes[0].format, 'float32x3');

  // Verify compute pipeline entryPoint
  const computePipeCall = calls.find((c) => c.op === 'createComputePipeline');
  assert.equal(computePipeCall.desc.compute.entryPoint, 'cs_main');

  // Verify bind group entry counts (compute=2, render=4)
  const bindGroupCalls = calls.filter((c) => c.op === 'createBindGroup');
  assert.equal(bindGroupCalls.length, 2);
  assert.equal(bindGroupCalls[0].desc.entries.length, 2);
  assert.equal(bindGroupCalls[1].desc.entries.length, 4);

  // --- 2. Execute 3 animated frames with varying inputState ---
  const frameInputs = [
    null,
    { leftStickX: 0.5, leftStickY: -0.25, rightStickX: 0.75, rightTrigger: 0.8, buttons: 0x100 },
    (() => {
      // Binary 40-byte InputState SharedArrayBuffer snapshot
      const sab = new SharedArrayBuffer(40);
      const dv = new DataView(sab);
      dv.setUint32(0, 2, true); // seq
      dv.setUint32(4, 0x10, true); // buttons (JUMP)
      dv.setInt32(8, 12, true); // mouse_dx
      dv.setInt32(12, -8, true); // mouse_dy
      dv.setFloat32(16, -0.4, true); // left_stick_x
      dv.setFloat32(20, 0.6, true); // left_stick_y
      dv.setFloat32(24, -0.5, true); // right_stick_x
      dv.setFloat32(28, 0.1, true); // right_stick_y
      dv.setFloat32(32, 0.2, true); // left_trigger
      dv.setFloat32(36, 0.9, true); // right_trigger
      return sab;
    })(),
  ];

  const frameTimes = [0.0, 0.5, 1.25];
  const uniformSnapshots = [];

  for (let i = 0; i < 3; i++) {
    const startIdx = calls.length;
    const frameBytes = controller.buildFrameCommandBuffer(i, frameTimes[i], frameInputs[i]);
    assert.equal(submitToRing(ring, frameBytes), true);
    const drained = processor.drainRing();
    assert.equal(drained.bytesDrained, frameBytes.byteLength);
    assert.equal(drained.presents, 1);
    assert.equal(drained.frameCount, i + 1);
    assert.equal(ring.availableBytes(), 0);

    const frameCalls = calls.slice(startIdx);
    const ops = frameCalls.map((c) => c.op);

    // Confirm exact pass order: uniform write -> compute pass -> render pass -> submit
    assert.deepEqual(ops, [
      'queue.writeBuffer',
      'beginComputePass',
      'computePass.setPipeline',
      'computePass.setBindGroup',
      'computePass.dispatchWorkgroups',
      'computePass.end',
      'beginRenderPass',
      'renderPass.setPipeline',
      'renderPass.setBindGroup',
      'renderPass.setVertexBuffer',
      'renderPass.setIndexBuffer',
      'renderPass.drawIndexed',
      'renderPass.end',
      'encoder.finish',
      'queue.submit',
    ]);

    const dispatchCall = frameCalls.find((c) => c.op === 'computePass.dispatchWorkgroups');
    assert.deepEqual({ x: dispatchCall.x, y: dispatchCall.y, z: dispatchCall.z }, { x: 1, y: 1, z: 1 });

    const drawCall = frameCalls.find((c) => c.op === 'renderPass.drawIndexed');
    assert.equal(drawCall.indexCount, 6);
    assert.equal(drawCall.instanceCount, 1);
    assert.equal(drawCall.firstIndex, 0);
    assert.equal(drawCall.baseVertex, 0);
    assert.equal(drawCall.firstInstance, 0);

    const idxCall = frameCalls.find((c) => c.op === 'renderPass.setIndexBuffer');
    assert.equal(idxCall.indexFormat, 'uint16');
    assert.equal(idxCall.offset, 0);

    const ubWrite = frameCalls.find((c) => c.op === 'queue.writeBuffer');
    assert.equal(ubWrite.data.byteLength, UNIFORM_BYTE_SIZE);
    const f32 = new Float32Array(ubWrite.data.buffer, ubWrite.data.byteOffset, UNIFORM_BYTE_SIZE / 4);
    uniformSnapshots.push(Array.from(f32));
  }

  assert.equal(controller.frameCount, 3);
  assert.equal(processor.frameCount, 3);
  assert.equal(ring.droppedPackets, 0);

  // Verify uniform updates reflect frameIndex, timeSec, and inputState changes
  assert.equal(uniformSnapshots[0][0], 0.0);
  assert.equal(uniformSnapshots[0][1], 0);
  assert.equal(uniformSnapshots[0][4], 0);

  assert.equal(uniformSnapshots[1][0], 0.5);
  assert.equal(uniformSnapshots[1][1], 1);
  assert.equal(uniformSnapshots[1][4], 0.5); // leftStickX
  assert.equal(uniformSnapshots[1][5], -0.25); // leftStickY
  assert.equal(uniformSnapshots[1][6], 0.75); // rightStickX
  assert.ok(Math.abs(uniformSnapshots[1][7] - 0.8) < 1e-6); // rightTrigger - leftTrigger

  assert.equal(uniformSnapshots[2][0], 1.25);
  assert.equal(uniformSnapshots[2][1], 2);
  assert.ok(Math.abs(uniformSnapshots[2][4] - -0.4) < 1e-5);
  assert.ok(Math.abs(uniformSnapshots[2][5] - 0.6) < 1e-5);
  assert.ok(Math.abs(uniformSnapshots[2][7] - 0.7) < 1e-5); // 0.9 - 0.2
  assert.equal(uniformSnapshots[2][12], 0x10);
  assert.equal(uniformSnapshots[2][13], 12);
  assert.equal(uniformSnapshots[2][14], -8);

  // --- 3. Teardown command buffer ---
  const createdBuffers = Array.from(processor.buffers.values());
  const createdTextures = Array.from(processor.textures.values());
  assert.equal(createdBuffers.length, 4);
  assert.equal(createdTextures.length, 1);

  const teardownBytes = controller.buildTeardownCommandBuffer();
  assert.equal(controller.initialized, false);
  const teardownPackets = decodePackets(teardownBytes);
  assert.equal(teardownPackets.length, controller.allocatedIds.length);
  for (const pkt of teardownPackets) {
    assert.equal(pkt.opcode, GPU_OPCODES.DESTROY_RESOURCE);
    assert.equal(pkt.payloadBytes, 4);
  }

  assert.equal(controller.submitToRing(ring, teardownBytes), true);
  const teardownDrain = processor.drainRing();
  assert.equal(teardownDrain.bytesDrained, teardownBytes.byteLength);
  assert.equal(teardownDrain.presents, 0);

  // Confirm all resource maps are completely empty and GPU resources had .destroy() called
  assert.equal(processor.buffers.size, 0);
  assert.equal(processor.textures.size, 0);
  assert.equal(processor.textureViews.size, 0);
  assert.equal(processor.samplers.size, 0);
  assert.equal(processor.bindGroups.size, 0);
  assert.equal(processor.shaderModules.size, 0);
  assert.equal(processor.pipelines.size, 0);
  assert.equal(processor.computePipelines.size, 0);

  for (const buf of createdBuffers) {
    assert.equal(buf.destroyed, true, 'Every GPUBuffer must be destroyed on teardown');
  }
  for (const tex of createdTextures) {
    assert.equal(tex.destroyed, true, 'Every GPUTexture must be destroyed on teardown');
  }
});

test('submitToRing handles backpressure gracefully when ring is full and validates inputs', () => {
  const smallRing = GpuRingBuffer.create(256);
  const bigBuffer = new Uint8Array(512);
  assert.equal(submitToRing(smallRing, bigBuffer), false);
  assert.equal(smallRing.droppedPackets, 1);

  assert.equal(submitToRing(smallRing, new Uint8Array(0)), true);
  assert.throws(() => submitToRing(null, new Uint8Array(4)), TypeError);
});

test('generateProceduralTexture and normalizeInputState validate edge cases', () => {
  const tex = generateProceduralTexture(16, 16);
  assert.equal(tex.byteLength, 16 * 16 * 4);
  // Every pixel alpha must be 255
  for (let i = 3; i < tex.byteLength; i += 4) {
    assert.equal(tex[i], 255);
  }
  assert.throws(() => generateProceduralTexture(0, 16), RangeError);

  const empty = normalizeInputState(null);
  assert.equal(empty.buttons, 0);
  assert.equal(empty.leftStickX, 0);

  const snake = normalizeInputState({
    buttons: 5,
    left_stick_x: 0.25,
    right_trigger: 0.5,
  });
  assert.equal(snake.buttons, 5);
  assert.equal(snake.leftStickX, 0.25);
  assert.equal(snake.rightTrigger, 0.5);

  assert.equal(Object.keys(DEFAULT_RESOURCE_IDS).length, 13);
  assert.equal(STORAGE_BYTE_SIZE, 48);
  assert.equal(DEFAULT_TARGET_FPS, 30);
  assert.equal(DEFAULT_FRAME_BUDGET_MS, 33.333);
  assert.equal(DEFAULT_NON_BLACK_THRESHOLD, 4);
});

test('evaluateFramePacing computes deterministic 30 FPS pacing metrics and guards invalid inputs', () => {
  const input = [16, 20, 24, 28, 32, 16, 20, 24, 28, 32];
  const snapshot = input.slice();
  const steady = evaluateFramePacing(input);

  // Input array must not be mutated
  assert.deepEqual(input, snapshot);
  assert.equal(steady.sampleCount, 10);
  assert.equal(steady.droppedSamples, 0);
  assert.equal(steady.budgetMs, DEFAULT_FRAME_BUDGET_MS);
  assert.ok(Math.abs(steady.targetFps - 1000 / 33.333) < 1e-9);
  assert.equal(steady.avgMs, 24);
  assert.equal(steady.minMs, 16);
  assert.equal(steady.maxMs, 32);
  assert.equal(steady.p50Ms, 24);
  assert.equal(steady.p95Ms, 32);
  assert.equal(steady.p99Ms, 32);
  assert.ok(Math.abs(steady.avgFps - 1000 / 24) < 1e-9);
  assert.ok(Math.abs(steady.jitterMs - 48 / 9) < 1e-9);
  assert.equal(steady.overBudgetCount, 0);
  assert.equal(steady.overBudgetRatio, 0);
  assert.equal(steady.withinBudgetRatio, 1);
  assert.equal(steady.withinBudget, true);
  assert.equal(steady.meetsTarget, true);

  // 19 frames at 20ms + 1 spike at 50ms => p95 stays within 33.333ms while withinBudget is false
  const withSingleSpike = Array(19).fill(20).concat([50]);
  const spikeMetrics = evaluateFramePacing(withSingleSpike);
  assert.equal(spikeMetrics.sampleCount, 20);
  assert.equal(spikeMetrics.overBudgetCount, 1);
  assert.equal(spikeMetrics.overBudgetRatio, 0.05);
  assert.equal(spikeMetrics.withinBudgetRatio, 0.95);
  assert.equal(spikeMetrics.p95Ms, 20);
  assert.equal(spikeMetrics.p99Ms, 50);
  assert.equal(spikeMetrics.withinBudget, false);
  assert.equal(spikeMetrics.meetsTarget, true);

  // Sustained slow frames miss the 30 FPS target
  const slow = evaluateFramePacing([38, 40, 42, 44, 46]);
  assert.equal(slow.overBudgetCount, 5);
  assert.equal(slow.withinBudget, false);
  assert.equal(slow.meetsTarget, false);

  // Custom 60 FPS budget + TypedArray + invalid sample filtering
  const mixed = evaluateFramePacing(new Float32Array([16, -2, Number.NaN, 17, Infinity, 16]), 16.667);
  assert.equal(mixed.sampleCount, 3);
  assert.equal(mixed.droppedSamples, 3);
  assert.equal(mixed.overBudgetCount, 1);
  assert.equal(mixed.withinBudget, false);

  // Generator iterable support
  function* gen() {
    yield 33;
    yield 33;
  }
  const fromIter = evaluateFramePacing(gen());
  assert.equal(fromIter.sampleCount, 2);
  assert.equal(fromIter.jitterMs, 0);
  assert.equal(fromIter.withinBudget, true);

  // Empty / null / undefined guards
  for (const emptyInput of [null, undefined, [], [NaN, -5]]) {
    const res = evaluateFramePacing(emptyInput);
    assert.equal(res.sampleCount, 0);
    assert.equal(res.avgMs, 0);
    assert.equal(res.avgFps, 0);
    assert.equal(res.withinBudget, false);
    assert.equal(res.meetsTarget, false);
  }

  // Invalid budgetMs throws RangeError
  assert.throws(() => evaluateFramePacing([16], 0), RangeError);
  assert.throws(() => evaluateFramePacing([16], -10), RangeError);
  assert.throws(() => evaluateFramePacing([16], Number.NaN), RangeError);
});

test('hasNonBlackRgbaPixels validates all-black, alpha-only, single bright pixel, and empty/null guards', () => {
  // Empty / null / malformed guards
  assert.equal(hasNonBlackRgbaPixels(null), false);
  assert.equal(hasNonBlackRgbaPixels(undefined), false);
  assert.equal(hasNonBlackRgbaPixels([]), false);
  assert.equal(hasNonBlackRgbaPixels(new Uint8Array(0)), false);
  assert.equal(hasNonBlackRgbaPixels(new Uint8Array(3)), false);
  assert.equal(hasNonBlackRgbaPixels({}), false);

  // All-black buffer
  assert.equal(hasNonBlackRgbaPixels(new Uint8Array(64)), false);

  // Alpha-only buffer (R=G=B=0, A=255) must remain false
  const alphaOnly = new Uint8Array(16);
  for (let i = 3; i < alphaOnly.length; i += 4) {
    alphaOnly[i] = 255;
  }
  assert.equal(hasNonBlackRgbaPixels(alphaOnly), false);
  assert.equal(hasNonBlackRgbaPixels(alphaOnly, 0), false);

  // Threshold boundary checks
  assert.equal(hasNonBlackRgbaPixels(new Uint8Array([4, 4, 4, 255]), 4), false);
  assert.equal(hasNonBlackRgbaPixels(new Uint8Array([5, 0, 0, 255]), 4), true);
  assert.equal(hasNonBlackRgbaPixels(new Uint8Array([1, 0, 0, 255]), 0), true);

  // Single bright pixel in R, G, or B among alpha-only black pixels
  for (const channel of [0, 1, 2]) {
    const buf = new Uint8Array(32);
    for (let i = 3; i < buf.length; i += 4) {
      buf[i] = 255;
    }
    buf[4 * 5 + channel] = 200;
    assert.equal(hasNonBlackRgbaPixels(buf), true, `Bright channel ${channel} should pass`);
  }

  // Supports Uint8ClampedArray, ArrayBuffer, SharedArrayBuffer, and plain arrays
  assert.equal(hasNonBlackRgbaPixels(new Uint8ClampedArray([0, 0, 0, 255, 0, 0, 128, 255])), true);
  const ab = new Uint8Array([0, 0, 0, 255, 10, 0, 0, 255]).buffer;
  assert.equal(hasNonBlackRgbaPixels(ab), true);
  const sab = new SharedArrayBuffer(8);
  new Uint8Array(sab).set([0, 0, 0, 255, 0, 20, 0, 255]);
  assert.equal(hasNonBlackRgbaPixels(sab), true);
  assert.equal(hasNonBlackRgbaPixels([0, 0, 0, 255, 0, 0, 255, 255]), true);

  // Trailing incomplete pixel bytes (< 4 bytes remainder) are ignored safely
  assert.equal(hasNonBlackRgbaPixels(new Uint8Array([0, 0, 0, 255, 255, 255])), false);

  // Procedural smoke texture is guaranteed non-black
  assert.equal(hasNonBlackRgbaPixels(generateProceduralTexture(16, 16)), true);
});

test('summarizeDrawBatch and SmokeSceneController batch metadata preserve default rendering behavior', () => {
  const controller = new SmokeSceneController({ textureSize: 16, aspect: 16 / 9 });
  assert.equal(controller.getLastBatchSummary(), null);

  // Default buildFrameCommandBuffer stays unchanged (12 packets, 1 drawIndexed, 6 indices)
  const defaultFrame = controller.buildFrameCommandBuffer(0, 0.25, null);
  const defaultSummary = summarizeDrawBatch(defaultFrame);
  assert.equal(defaultSummary.byteLength, defaultFrame.byteLength);
  assert.equal(defaultSummary.packetCount, 12);
  assert.equal(defaultSummary.truncated, false);
  assert.equal(defaultSummary.beginFrameCount, 1);
  assert.equal(defaultSummary.presentCount, 1);
  assert.equal(defaultSummary.clearColorCount, 1);
  assert.equal(defaultSummary.computeDispatchCount, 1);
  assert.equal(defaultSummary.drawCallCount, 1);
  assert.equal(defaultSummary.drawCount, 0);
  assert.equal(defaultSummary.drawIndexedCount, 1);
  assert.equal(defaultSummary.totalIndices, 6);
  assert.equal(defaultSummary.totalInstances, 1);
  assert.equal(defaultSummary.hasRenderWork, true);
  assert.equal(defaultSummary.opcodeCounts.DRAW_INDEXED, 1);
  assert.equal(defaultSummary.opcodeCounts.PRESENT, 1);

  const lastDefault = controller.getLastBatchSummary();
  assert.equal(lastDefault.frameIndex, 0);
  assert.equal(lastDefault.packetCount, 12);
  assert.equal(lastDefault.drawBatchCount, 1);
  assert.equal(lastDefault.instanceCount, 1);
  assert.equal(lastDefault.totalIndices, 6);
  // Returned summary is a defensive copy
  lastDefault.drawBatchCount = 999;
  assert.equal(controller.getLastBatchSummary().drawBatchCount, 1);

  // Optional multi-draw batching updates summary accurately and runs cleanly in WgpuCommandProcessor
  const batchedFrame = controller.buildFrameCommandBuffer(1, 0.5, null, {
    drawBatchCount: 4,
    instanceCount: 3,
  });
  const batchedSummary = summarizeDrawBatch(batchedFrame);
  assert.equal(batchedSummary.packetCount, 15);
  assert.equal(batchedSummary.drawCallCount, 4);
  assert.equal(batchedSummary.drawIndexedCount, 4);
  assert.equal(batchedSummary.totalIndices, 4 * 6 * 3);
  assert.equal(batchedSummary.totalInstances, 12);
  assert.equal(batchedSummary.hasRenderWork, true);

  const lastBatched = controller.getLastBatchSummary();
  assert.equal(lastBatched.drawBatchCount, 4);
  assert.equal(lastBatched.instanceCount, 3);
  assert.equal(lastBatched.totalIndices, 72);
  assert.equal(lastBatched.totalInstances, 12);

  // Invalid batch options throw RangeError
  assert.throws(() => controller.buildFrameCommandBuffer(2, 0.75, null, { drawBatchCount: 0 }), RangeError);
  assert.throws(() => controller.buildFrameCommandBuffer(2, 0.75, null, { instanceCount: 1.5 }), RangeError);

  // Mixed DRAW + DRAW_INDEXED stream and empty/truncated guards
  const mixedBuilder = new SmokeCommandBuilder()
    .beginFrame()
    .draw(3, 2, 0, 0)
    .drawIndexed(6, 4, 0, 0, 0)
    .present();
  const mixedSummary = summarizeDrawBatch(mixedBuilder.finish());
  assert.equal(mixedSummary.drawCallCount, 2);
  assert.equal(mixedSummary.drawCount, 1);
  assert.equal(mixedSummary.drawIndexedCount, 1);
  assert.equal(mixedSummary.totalVertices, 6);
  assert.equal(mixedSummary.totalIndices, 24);
  assert.equal(mixedSummary.totalInstances, 6);
  assert.equal(mixedSummary.hasRenderWork, true);

  const emptySummary = summarizeDrawBatch(null);
  assert.equal(emptySummary.packetCount, 0);
  assert.equal(emptySummary.truncated, false);
  assert.equal(emptySummary.hasRenderWork, false);

  const truncatedHeader = summarizeDrawBatch(new Uint8Array([1, 0, 0, 0]));
  assert.equal(truncatedHeader.truncated, true);
  assert.equal(truncatedHeader.packetCount, 0);

  const truncatedPayload = summarizeDrawBatch(defaultFrame.subarray(0, defaultFrame.byteLength - 4));
  assert.equal(truncatedPayload.truncated, true);
  assert.ok(truncatedPayload.packetCount < defaultSummary.packetCount);
});
