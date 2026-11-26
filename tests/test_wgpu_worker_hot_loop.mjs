import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  OPCODES,
  SHADER_STAGE,
  BUFFER_USAGE,
  TEXTURE_USAGE,
  TEXTURE_FORMAT,
  TEXTURE_DIMENSION,
  BIND_RESOURCE_TYPE,
  INDEX_FORMAT,
  SWAPCHAIN_PROBE_STRIPS,
  SWAPCHAIN_PROBE_BUFFER_SIZE,
} = require('../web/workers/wgpu_worker.js');
const { GpuRingBuffer } = require('../web/workers/gpu_ring.js');
const { installRenderStatePackets } = require('../web/workers/render_state_packets.js');

const enc = new TextEncoder();

class PacketBuilder {
  constructor() {
    this.chunks = [];
    this.totalBytes = 0;
  }

  _push(opcode, payloadBuf, flags = 0) {
    const payload = payloadBuf ? new Uint8Array(payloadBuf) : new Uint8Array(0);
    const pkt = new Uint8Array(8 + payload.byteLength);
    const view = new DataView(pkt.buffer);
    view.setUint16(0, opcode, true);
    view.setUint16(2, flags, true);
    view.setUint32(4, payload.byteLength, true);
    if (payload.byteLength > 0) pkt.set(payload, 8);
    this.chunks.push(pkt);
    this.totalBytes += pkt.byteLength;
    return this;
  }

  beginFrame() {
    return this._push(OPCODES.BEGIN_FRAME);
  }

  clearColor(r, g, b, a) {
    const buf = new ArrayBuffer(16);
    const v = new DataView(buf);
    v.setFloat32(0, r, true);
    v.setFloat32(4, g, true);
    v.setFloat32(8, b, true);
    v.setFloat32(12, a, true);
    return this._push(OPCODES.CLEAR_COLOR, buf);
  }

  createBuffer(id, usage, byteSize) {
    const buf = new ArrayBuffer(16);
    const v = new DataView(buf);
    v.setUint32(0, id, true);
    v.setUint32(4, usage, true);
    v.setBigUint64(8, BigInt(byteSize), true);
    return this._push(OPCODES.CREATE_BUFFER, buf);
  }

  createTexture(id, width, height, format = TEXTURE_FORMAT.RGBA8_UNORM, usage = TEXTURE_USAGE.TEXTURE_BINDING) {
    const buf = new ArrayBuffer(32);
    const v = new DataView(buf);
    v.setUint32(0, id, true);
    v.setUint32(4, width, true);
    v.setUint32(8, height, true);
    v.setUint32(12, 1, true);
    v.setUint32(16, 1, true);
    v.setUint32(20, format, true);
    v.setUint32(24, usage, true);
    v.setUint32(28, TEXTURE_DIMENSION.DIM_2D, true);
    return this._push(OPCODES.CREATE_TEXTURE, buf);
  }

  createShaderModule(id, stage, wgsl) {
    const src = enc.encode(wgsl);
    const buf = new ArrayBuffer(16 + src.byteLength);
    const v = new DataView(buf);
    v.setUint32(0, id, true);
    v.setUint32(4, stage, true);
    v.setUint32(8, src.byteLength, true);
    v.setUint32(12, 0, true);
    new Uint8Array(buf, 16).set(src);
    return this._push(OPCODES.CREATE_SHADER_MODULE, buf);
  }

  createRenderPipeline(id, vsId, fsId, vertexStride = 0, topology = 0) {
    const buf = new ArrayBuffer(24);
    const v = new DataView(buf);
    v.setUint32(0, id, true);
    v.setUint32(4, vsId, true);
    v.setUint32(8, fsId, true);
    v.setUint32(12, vertexStride, true);
    v.setUint32(16, topology, true);
    v.setUint32(20, 0, true);
    return this._push(OPCODES.CREATE_RENDER_PIPELINE, buf);
  }

  createComputePipeline(id, csId) {
    const buf = new ArrayBuffer(16);
    const v = new DataView(buf);
    v.setUint32(0, id, true);
    v.setUint32(4, csId, true);
    v.setUint32(8, 0, true);
    v.setUint32(12, 0, true);
    return this._push(OPCODES.CREATE_COMPUTE_PIPELINE, buf);
  }

  createBindGroup(id, pipelineId, groupIndex, entries) {
    const buf = new ArrayBuffer(16 + entries.length * 32);
    const v = new DataView(buf);
    v.setUint32(0, id, true);
    v.setUint32(4, pipelineId, true);
    v.setUint32(8, groupIndex, true);
    v.setUint32(12, entries.length, true);
    for (let i = 0; i < entries.length; i += 1) {
      const e = entries[i];
      const off = 16 + i * 32;
      v.setUint32(off, e.binding, true);
      v.setUint32(off + 4, e.resourceType, true);
      v.setUint32(off + 8, e.resourceId, true);
      v.setUint32(off + 12, 0, true);
      v.setBigUint64(off + 16, BigInt(e.offset || 0), true);
      v.setBigUint64(off + 24, BigInt(e.size || 0), true);
    }
    return this._push(OPCODES.CREATE_BIND_GROUP, buf);
  }

  setPipeline(pipelineId) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, pipelineId, true);
    return this._push(OPCODES.SET_PIPELINE, buf);
  }

  setVertexBuffer(slot, bufferId, offset = 0) {
    const buf = new ArrayBuffer(16);
    const v = new DataView(buf);
    v.setUint32(0, slot, true);
    v.setUint32(4, bufferId, true);
    v.setBigUint64(8, BigInt(offset), true);
    return this._push(OPCODES.SET_VERTEX_BUFFER, buf);
  }

  setIndexBuffer(bufferId, indexFormat = INDEX_FORMAT.UINT16, offset = 0) {
    const buf = new ArrayBuffer(16);
    const v = new DataView(buf);
    v.setUint32(0, bufferId, true);
    v.setUint32(4, indexFormat, true);
    v.setBigUint64(8, BigInt(offset), true);
    return this._push(OPCODES.SET_INDEX_BUFFER, buf);
  }

  setBindGroup(groupIndex, bindGroupId) {
    const buf = new ArrayBuffer(8);
    const v = new DataView(buf);
    v.setUint32(0, groupIndex, true);
    v.setUint32(4, bindGroupId, true);
    return this._push(OPCODES.SET_BIND_GROUP, buf);
  }

  draw(vertexCount, instanceCount = 1, firstVertex = 0, firstInstance = 0) {
    const buf = new ArrayBuffer(16);
    const v = new DataView(buf);
    v.setUint32(0, vertexCount, true);
    v.setUint32(4, instanceCount, true);
    v.setUint32(8, firstVertex, true);
    v.setUint32(12, firstInstance, true);
    return this._push(OPCODES.DRAW, buf);
  }

  drawIndexed(indexCount, instanceCount = 1, firstIndex = 0, baseVertex = 0, firstInstance = 0) {
    const buf = new ArrayBuffer(20);
    const v = new DataView(buf);
    v.setUint32(0, indexCount, true);
    v.setUint32(4, instanceCount, true);
    v.setUint32(8, firstIndex, true);
    v.setInt32(12, baseVertex, true);
    v.setUint32(16, firstInstance, true);
    return this._push(OPCODES.DRAW_INDEXED, buf);
  }

  setComputePipeline(pipelineId) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, pipelineId, true);
    return this._push(OPCODES.SET_COMPUTE_PIPELINE, buf);
  }

  dispatchCompute(x, y, z) {
    const buf = new ArrayBuffer(12);
    const v = new DataView(buf);
    v.setUint32(0, x, true);
    v.setUint32(4, y, true);
    v.setUint32(8, z, true);
    return this._push(OPCODES.DISPATCH_COMPUTE, buf);
  }

  destroyResource(id) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, id, true);
    return this._push(OPCODES.DESTROY_RESOURCE, buf);
  }

  present() {
    return this._push(OPCODES.PRESENT);
  }

  beginRenderPassSwapchain(clearRgba = [0, 0, 0, 1]) {
    // ABI v2 BeginRenderPass: 1 color attachment targeting SWAPCHAIN_TEXTURE_ID (0xFFFFFFFF)
    const buf = new ArrayBuffer(4 + 32 + 32);
    const v = new DataView(buf);
    v.setUint32(0, 1, true); // colorCount = 1, hasDepth = 0
    v.setUint32(4, 0xffffffff, true); // SWAPCHAIN_TEXTURE_ID
    v.setUint32(8, 2, true); // loadOp = clear
    v.setUint32(12, 1, true); // storeOp = store
    v.setFloat32(16, clearRgba[0], true);
    v.setFloat32(20, clearRgba[1], true);
    v.setFloat32(24, clearRgba[2], true);
    v.setFloat32(28, clearRgba[3], true);
    return this._push(OPCODES.BEGIN_RENDER_PASS, buf);
  }

  endRenderPass() {
    return this._push(OPCODES.END_RENDER_PASS);
  }

  setBindGroup2(groupIndex, bindGroupId) {
    const buf = new ArrayBuffer(16);
    const v = new DataView(buf);
    v.setUint32(0, groupIndex, true);
    v.setUint32(4, bindGroupId, true);
    v.setUint32(8, 0, true);
    v.setUint32(12, 0, true);
    return this._push(OPCODES.SET_BIND_GROUP2, buf);
  }

  finish() {
    const out = new Uint8Array(this.totalBytes);
    let off = 0;
    for (const c of this.chunks) {
      out.set(c, off);
      off += c.byteLength;
    }
    return out;
  }
}

function createFastMockWebGpu(opts = {}) {
  const counts = {
    beginRenderPass: 0,
    beginComputePass: 0,
    setPipeline: 0,
    setComputePipeline: 0,
    setVertexBuffer: 0,
    setIndexBuffer: 0,
    setBindGroup: 0,
    draw: 0,
    drawIndexed: 0,
    dispatchWorkgroups: 0,
    submits: 0,
    copyTextureToBuffer: 0,
  };
  const clearValues = [];
  const destroyedResources = [];
  const pendingFences = [];
  let nextResId = 1;

  const renderPass = {
    setPipeline() { counts.setPipeline += 1; },
    setVertexBuffer() { counts.setVertexBuffer += 1; },
    setIndexBuffer() { counts.setIndexBuffer += 1; },
    setBindGroup() { counts.setBindGroup += 1; },
    draw() { counts.draw += 1; },
    drawIndexed() { counts.drawIndexed += 1; },
    end() {},
  };

  const computePass = {
    setPipeline() { counts.setComputePipeline += 1; },
    setBindGroup() { counts.setBindGroup += 1; },
    dispatchWorkgroups() { counts.dispatchWorkgroups += 1; },
    end() {},
  };

  const device = {
    createBuffer(desc) {
      const id = nextResId++;
      const buf = {
        _mockId: id,
        size: desc.size,
        usage: desc.usage,
        destroyed: false,
        destroy() {
          this.destroyed = true;
          destroyedResources.push({ kind: 'buffer', id });
        },
        mapAsync() {
          return Promise.resolve();
        },
        getMappedRange() {
          const out = new Uint8Array(desc.size);
          if (opts.probeVisibleAfterCopies && counts.copyTextureToBuffer >= opts.probeVisibleAfterCopies) {
            out[0] = 200;
            out[1] = 200;
            out[2] = 200;
            out[3] = 255;
          }
          return out.buffer;
        },
        unmap() {},
      };
      return buf;
    },
    createTexture(desc) {
      const id = nextResId++;
      return {
        _mockId: id,
        width: desc.size.width,
        height: desc.size.height,
        format: desc.format,
        usage: desc.usage,
        destroyed: false,
        createView() {
          return { _viewFor: id };
        },
        destroy() {
          this.destroyed = true;
          destroyedResources.push({ kind: 'texture', id });
        },
      };
    },
    createShaderModule(desc) {
      return { code: desc.code };
    },
    createRenderPipeline(desc) {
      return {
        desc,
        getBindGroupLayout(idx) {
          return { _bglIdx: idx };
        },
      };
    },
    createComputePipeline(desc) {
      return {
        desc,
        getBindGroupLayout(idx) {
          return { _bglIdx: idx };
        },
      };
    },
    createBindGroup(desc) {
      return { desc };
    },
    createCommandEncoder() {
      return {
        beginRenderPass(desc) {
          counts.beginRenderPass += 1;
          if (opts.recordClearValues && desc.colorAttachments && desc.colorAttachments[0]) {
            clearValues.push(desc.colorAttachments[0].clearValue);
          }
          return renderPass;
        },
        beginComputePass() {
          counts.beginComputePass += 1;
          return computePass;
        },
        copyTextureToBuffer() {
          counts.copyTextureToBuffer += 1;
        },
        finish() {
          return { _cmdBuf: true };
        },
      };
    },
    queue: {
      writeBuffer() {},
      writeTexture() {},
      submit() {
        counts.submits += 1;
      },
      onSubmittedWorkDone: opts.manualFence
        ? () =>
            new Promise((resolve) => {
              pendingFences.push(resolve);
            })
        : undefined,
    },
  };

  const swapchainTexture = {
    width: 1280,
    height: 720,
    usage: 0x11, // RENDER_ATTACHMENT | COPY_SRC
    createView() {
      return { _swapView: true };
    },
  };

  const ctx = {
    getCurrentTexture() {
      return swapchainTexture;
    },
  };

  return { device, ctx, counts, clearValues, destroyedResources, pendingFences };
}

test('W359: 120 frames x 50 draws reuses pooled pass commands, binding maps/slots, immutable clearColor, frozen gpuAdapter, and preserves W346 fast-path', () => {
  const mock = createFastMockWebGpu({ recordClearValues: true });
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    width: 1280,
    height: 720,
    enableResourceReplay: false,
    requireNonBlackFrame: false,
    gpuAdapter: {
      vendor: 'amd',
      architecture: 'rdna2',
      device: 'series-x',
      description: 'mock-rdna2',
    },
  });

  // Frozen cached gpuAdapter snapshot returns identical object reference with zero spread churn.
  const adapterSnap1 = processor.gpuAdapter;
  const adapterSnap2 = processor.gpuAdapter;
  assert.equal(adapterSnap1, adapterSnap2);
  assert.equal(Object.isFrozen(adapterSnap1), true);
  assert.equal(adapterSnap1.vendor, 'amd');

  // Setup resources once before steady-state loop.
  const setup = new PacketBuilder()
    .createBuffer(1, BUFFER_USAGE.VERTEX, 4096)
    .createBuffer(2, BUFFER_USAGE.INDEX, 2048)
    .createBuffer(3, BUFFER_USAGE.UNIFORM, 256)
    .createShaderModule(10, SHADER_STAGE.VERTEX, '@vertex fn vs_main() -> @builtin(position) vec4f { return vec4f(0.0); }')
    .createShaderModule(11, SHADER_STAGE.FRAGMENT, '@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1.0); }')
    .createShaderModule(12, SHADER_STAGE.COMPUTE, '@compute @workgroup_size(1) fn cs_main() {}')
    .createRenderPipeline(20, 10, 11, 24, 0)
    .createComputePipeline(21, 12)
    .createBindGroup(30, 20, 0, [
      { binding: 0, resourceType: BIND_RESOURCE_TYPE.BUFFER, resourceId: 3, offset: 0, size: 256 },
    ])
    .finish();
  processor.executeCommandBuffer(setup);

  // Build one steady-state frame with 50 draws (25 non-indexed + 25 indexed) + 1 compute dispatch.
  const buildFrame = (r, g, b, a) => {
    const fb = new PacketBuilder().beginFrame().clearColor(r, g, b, a);
    fb.setPipeline(20).setBindGroup(0, 30).setVertexBuffer(0, 1, 0).setIndexBuffer(2, INDEX_FORMAT.UINT32, 0);
    for (let d = 0; d < 25; d += 1) {
      fb.setVertexBuffer(0, 1, d * 24);
      fb.draw(3, 1, 0, d);
      fb.drawIndexed(6, 1, 0, d, 0);
    }
    fb.setComputePipeline(21).dispatchCompute(4, 1, 1);
    fb.present();
    return fb.finish();
  };

  const steadyFrameBytes = buildFrame(0.125, 0.25, 0.5, 1.0);
  const altColorFrameBytes = buildFrame(0.75, 0.5, 0.25, 1.0);

  const ring = GpuRingBuffer.create(64 * 1024);
  processor.attachRing(ring);

  const initialPassCommandsRef = processor._passCommands;
  const initialBoundBgMapRef = processor._boundBindGroups;
  const initialBoundVbMapRef = processor._boundVertexBuffers;

  let poolSizeAfterFrame1 = 0;
  let firstBgSlotRef = null;
  let firstVbSlotRef = null;

  const t0 = performance.now();
  const FRAMES = 120;
  for (let f = 0; f < FRAMES; f += 1) {
    const frameBytes = f === 60 ? altColorFrameBytes : steadyFrameBytes;
    assert.equal(ring.writeBytes(frameBytes), true);
    const res = processor.drainRing({ reuseResult: true });
    assert.equal(res.presents, 1);
    assert.equal(res.frameCount, f + 1);

    if (f === 0) {
      poolSizeAfterFrame1 = processor._passCommandPool.length;
      firstBgSlotRef = processor._boundBindGroupSlots.get(0);
      firstVbSlotRef = processor._boundVertexBufferSlots.get(0);
      assert.ok(poolSizeAfterFrame1 >= 80, 'pool should accommodate all pass commands of frame 1');
      assert.ok(firstBgSlotRef && firstVbSlotRef);
    } else {
      assert.equal(
        processor._passCommandPool.length,
        poolSizeAfterFrame1,
        `pool must not allocate new command slots on steady-state frame ${f + 1}`
      );
      assert.equal(processor._passCommands, initialPassCommandsRef);
      assert.equal(processor._boundBindGroups, initialBoundBgMapRef);
      assert.equal(processor._boundVertexBuffers, initialBoundVbMapRef);
      assert.equal(processor._boundBindGroupSlots.get(0), firstBgSlotRef);
      assert.equal(processor._boundVertexBufferSlots.get(0), firstVbSlotRef);
    }
  }
  const elapsedMs = performance.now() - t0;

  assert.equal(processor.frameCount, FRAMES);
  assert.equal(mock.counts.draw, FRAMES * 25);
  assert.equal(mock.counts.drawIndexed, FRAMES * 25);
  assert.equal(mock.counts.dispatchWorkgroups, FRAMES);
  assert.equal(mock.counts.setPipeline, FRAMES);
  assert.equal(mock.counts.setVertexBuffer, FRAMES * 26);
  assert.equal(mock.counts.setIndexBuffer, FRAMES);
  assert.equal(mock.counts.setBindGroup, FRAMES);
  assert.equal(mock.counts.submits, FRAMES);

  // Verify immutable clearColor snapshots: frames 0..59 share one frozen object, frame 60 has alt color,
  // and frame 60's clearValue was NOT mutated by subsequent frames or _resetFrameState().
  assert.equal(mock.clearValues.length, FRAMES);
  assert.equal(mock.clearValues[0], mock.clearValues[59]);
  assert.equal(Object.isFrozen(mock.clearValues[0]), true);
  assert.deepEqual(mock.clearValues[60], { r: 0.75, g: 0.5, b: 0.25, a: 1.0 });
  assert.deepEqual(mock.clearValues[61], { r: 0.125, g: 0.25, b: 0.5, a: 1.0 });

  // Verify pooled slots release GPU object references after frame reset (no retained leaks).
  assert.equal(processor._passCommands.length, 0);
  assert.equal(processor._passCommandPoolUsed, 0);
  for (const slot of processor._passCommandPool) {
    assert.equal(slot.pipeline, null);
    assert.equal(slot.bindGroup, null);
    assert.equal(slot._wgpuMeta, null);
  }
  assert.equal(processor._boundBindGroups.size, 0);
  assert.equal(processor._boundVertexBuffers.size, 0);
  assert.equal(firstBgSlotRef.bindGroup, null);
  assert.equal(firstVbSlotRef.buffer, null);

  // W346 invariants intact: zero fallback lookahead scans and zero replay entries when disabled.
  assert.equal(processor._fallbackProfileScanCount, 0);
  assert.equal(processor.resourceReplayEnabled, false);
  assert.equal(processor.getResourceReplayStats().recordedResourceCount, 0);

  // Throughput sanity check: 120 frames x 50 draws should execute well under 2 seconds (60 FPS budget).
  assert.ok(elapsedMs < 2000, `120x50 draw hot loop took ${elapsedMs.toFixed(1)}ms (> 2000ms)`);
});

test('W359: GpuRingBuffer wrap-around drains reuse persistent wrapScratch + cached DataView with directView zero-subarray mode', () => {
  // Small 1024B ring guarantees wrap-around on nearly every frame.
  const producer = GpuRingBuffer.create(1024);
  const consumer = GpuRingBuffer.attach(producer.sab);

  const initialWrapBuf = consumer._wrapScratch.buffer;
  const initialWrapView = consumer.wrapScratchView;
  const initialDataView = consumer.dataView;
  assert.ok(initialWrapView instanceof DataView);
  assert.ok(initialDataView instanceof DataView);

  const frameStream = new PacketBuilder()
    .beginFrame()
    .clearColor(0.2, 0.4, 0.6, 1.0)
    .draw(3, 1, 0, 0)
    .draw(6, 1, 0, 1)
    .present()
    .finish();

  let wrapCount = 0;
  let nonWrapCount = 0;
  let firstResultRef = null;
  let firstSpanRef = null;

  for (let i = 0; i < 64; i += 1) {
    assert.equal(producer.writeBytes(frameStream), true);
    const res = consumer.drainBatchZeroAlloc(
      (bytesView, byteOffset, byteLength, span) => {
        if (firstSpanRef === null) {
          firstSpanRef = span;
        } else {
          assert.equal(span, firstSpanRef, 'span descriptor must be reused');
        }
        assert.equal(byteLength, frameStream.byteLength);
        if (span.wrapped) {
          wrapCount += 1;
          assert.equal(bytesView, consumer._wrapScratch, 'directView wrap must pass _wrapScratch directly');
          assert.equal(bytesView.buffer, initialWrapBuf);
          assert.equal(byteOffset, 0);
          assert.equal(span.dataView, initialWrapView, 'wrap span must reuse cached wrapScratchView');
          assert.equal(span.viewOffset, 0);
        } else {
          nonWrapCount += 1;
          assert.equal(bytesView, consumer._data, 'directView non-wrap must pass _data directly');
          assert.equal(bytesView.buffer, producer.sab);
          assert.equal(span.dataView, initialDataView, 'non-wrap span must reuse cached dataView');
          assert.equal(span.viewOffset, byteOffset);
        }
        // Verify exact binary integrity of first and last opcodes via span.dataView.
        assert.equal(span.dataView.getUint16(span.viewOffset, true), OPCODES.BEGIN_FRAME);
        assert.equal(
          span.dataView.getUint16(span.viewOffset + byteLength - 8, true),
          OPCODES.PRESENT
        );
      },
      { reuseResult: true, directView: true }
    );
    if (firstResultRef === null) {
      firstResultRef = res;
    } else {
      assert.equal(res, firstResultRef, 'reuseResult must return same result object');
    }
    assert.equal(res.packets, 5);
    assert.equal(res.bytes, frameStream.byteLength);
  }

  assert.ok(wrapCount >= 5, `expected repeated wrap-arounds, got ${wrapCount}`);
  assert.ok(nonWrapCount >= 5, `expected non-wrap drains, got ${nonWrapCount}`);
  assert.equal(consumer._wrapScratch.buffer, initialWrapBuf);
  assert.equal(consumer.wrapScratchView, initialWrapView);

  // Verify drainRing({ reuseResult: true }) reuses result object across empty & non-empty polls
  // while default drainRing() returns distinct objects.
  const mock = createFastMockWebGpu();
  const proc = new WgpuCommandProcessor(mock.device, mock.ctx, { requireNonBlackFrame: false });
  proc.attachRing(consumer);

  const empty1 = proc.drainRing({ reuseResult: true });
  const empty2 = proc.drainRing({ reuseResult: true });
  assert.equal(empty1, empty2);
  assert.equal(empty1.bytesDrained, 0);

  const fresh1 = proc.drainRing();
  const fresh2 = proc.drainRing();
  assert.notEqual(fresh1, fresh2);

  // Stream 40 wrapped frames end-to-end through proc.drainRing({ reuseResult: true }).
  for (let i = 0; i < 40; i += 1) {
    assert.equal(producer.writeBytes(frameStream), true);
    const r = proc.drainRing({ reuseResult: true });
    assert.equal(r, empty1);
    assert.equal(r.presents, 1);
    assert.equal(r.frameCount, i + 1);
  }
  assert.equal(mock.counts.draw, 80);
  assert.equal(consumer._wrapScratch.buffer, initialWrapBuf);
  assert.equal(consumer.wrapScratchView, initialWrapView);

  // Verify monkey-patched executeCommandBuffer (e.g. GpuFrameRecorder.createInterceptor)
  // still intercepts ring-drained spans with exact byteLength, then returns to fast path on detach.
  const interceptedLengths = [];
  const origExec = proc.executeCommandBuffer;
  proc.executeCommandBuffer = function wrappedExec(bytes, opts) {
    interceptedLengths.push(bytes.byteLength);
    return origExec.call(this, bytes, opts);
  };
  assert.equal(producer.writeBytes(frameStream), true);
  proc.drainRing({ reuseResult: true });
  assert.deepEqual(interceptedLengths, [frameStream.byteLength]);
  delete proc.executeCommandBuffer;
  assert.equal(producer.writeBytes(frameStream), true);
  proc.drainRing({ reuseResult: true });
  assert.deepEqual(interceptedLengths, [frameStream.byteLength]);
});

test('W359: W342 mid-frame deferred destroys remain intact across hot-loop pooled frames', async () => {
  const mock = createFastMockWebGpu({ manualFence: true });
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    requireNonBlackFrame: false,
    enableResourceReplay: false,
  });

  // Create buffer + texture on frame 1, then destroy both mid-frame on frame 2 before PRESENT.
  const frame1 = new PacketBuilder()
    .beginFrame()
    .createBuffer(101, BUFFER_USAGE.VERTEX, 256)
    .createTexture(102, 64, 64)
    .present()
    .finish();
  processor.executeCommandBuffer(frame1);
  assert.equal(mock.destroyedResources.length, 0);

  const frame2 = new PacketBuilder()
    .beginFrame()
    .destroyResource(101)
    .destroyResource(102)
    .present()
    .finish();
  processor.executeCommandBuffer(frame2);

  // Resources must NOT be destroyed yet while the submitted fence is pending.
  assert.equal(mock.destroyedResources.length, 0);
  assert.equal(mock.pendingFences.length, 2);

  // Resolve fences and yield microtasks -> deferred destroys execute exactly once.
  for (const resolve of mock.pendingFences) resolve();
  await new Promise((r) => setImmediate(r));

  assert.equal(mock.destroyedResources.length, 2);
  assert.equal(processor._deferredDestroyCount, 2);
});

test('W359: worker handler throttles telemetry posts and preserves W350 8-strip swapchain probe + render_state_packets compatibility', async () => {
  assert.equal(SWAPCHAIN_PROBE_STRIPS.length, 8);
  assert.equal(SWAPCHAIN_PROBE_BUFFER_SIZE, 2048);

  const mock = createFastMockWebGpu({ probeVisibleAfterCopies: 8 });
  let fakeNow = 1000;
  const posted = [];
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    width: 1280,
    height: 720,
    now: () => fakeNow,
    swapchainProbeIntervalMs: 250,
    enableResourceReplay: false,
    gpuAdapter: { vendor: 'amd', architecture: 'rdna2', device: 'test', description: 'test' },
  });
  installRenderStatePackets(processor);

  const ring = GpuRingBuffer.create(32 * 1024);
  processor.attachRing(ring);

  const handler = createWgpuWorkerHandler({
    processor,
    postMessage: (m) => posted.push(m),
    telemetryIntervalMs: 250,
    now: () => fakeNow,
  });

  // Setup shader + pipeline + bind group so draws count as non-clear candidates for W350 probe.
  const setup = new PacketBuilder()
    .createBuffer(1, BUFFER_USAGE.UNIFORM, 64)
    .createShaderModule(10, SHADER_STAGE.VERTEX, '@vertex fn vs_main() -> @builtin(position) vec4f { return vec4f(0.0); }')
    .createShaderModule(11, SHADER_STAGE.FRAGMENT, '@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1.0); }')
    .createRenderPipeline(20, 10, 11, 0, 0)
    .createBindGroup(30, 20, 0, [
      { binding: 0, resourceType: BIND_RESOURCE_TYPE.BUFFER, resourceId: 1, offset: 0, size: 64 },
    ])
    .finish();
  processor.executeCommandBuffer(setup);

  const frameWithExt = new PacketBuilder()
    .beginFrame()
    .beginRenderPassSwapchain([0.1, 0.2, 0.3, 1.0])
    .setPipeline(20)
    .setBindGroup2(0, 30)
    .draw(3, 1, 0, 0)
    .endRenderPass()
    .present()
    .finish();

  // Run 30 frames at 16ms (~60 Hz) = 480ms total -> telemetry should fire on frame 1 + at ~250ms only (<= 3 times, NOT 30!).
  for (let f = 0; f < 30; f += 1) {
    fakeNow = 1000 + f * 16;
    assert.equal(ring.writeBytes(frameWithExt), true);
    await handler.handleMessage({ drainRing: true });
    if (f === 0) {
      // Allow W350 8-strip probe async readback to settle after frame 1.
      await new Promise((r) => setImmediate(r));
    }
  }

  const frameMsgs = posted.filter((m) => typeof m.frame === 'number');
  const telemetryMsgs = posted.filter((m) => m.type === 'gpu-telemetry');
  const visibleMsgs = posted.filter((m) => m.type === 'first-visible-frame');

  assert.equal(frameMsgs.length, 30);
  assert.ok(
    telemetryMsgs.length >= 2 && telemetryMsgs.length <= 3,
    `expected throttled telemetry (2-3 posts over 480ms), got ${telemetryMsgs.length}`
  );
  assert.equal(visibleMsgs.length, 1, 'W350 8-strip swapchain probe must fire first-visible-frame once');
  assert.equal(mock.counts.copyTextureToBuffer, 8, 'W350 probe must encode 8 strips then stop once visible');
  assert.equal(processor._setBindGroup2MetaPool.length, 1, 'setBindGroup2 _wgpuMeta pool must reuse single slot');
});
