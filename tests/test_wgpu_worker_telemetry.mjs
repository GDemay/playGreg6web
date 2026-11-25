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
const gpuRingMod = require('../web/workers/gpu_ring.js');
const panelMod = require('../web/ui/gpu_boot_panel.js');

const {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  buildGpuTelemetrySnapshot,
  DEFAULT_TELEMETRY_INTERVAL_MS,
  DEFAULT_TELEMETRY_MISSING_LIMIT,
  OPCODES,
  SHADER_STAGE,
  PRIMITIVE_TOPOLOGY,
} = wgpuMod;

const { GpuRingBuffer } = gpuRingMod;
const { GpuBootPanelController } = panelMod;

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

const S9_FALLBACK_FS_WGSL = [
  '@fragment',
  'fn fs_main() -> @location(0) vec4<f32> {',
  '    return vec4<f32>(0.9, 0.1, 0.5, 1.0);',
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

  createComputePipeline(id, csId) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, csId, true);
    view.setUint32(8, 0, true);
    view.setUint32(12, 0, true);
    this._push(OPCODES.CREATE_COMPUTE_PIPELINE, new Uint8Array(buf));
    return this;
  }

  setPipeline(pipelineId) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, pipelineId, true);
    this._push(OPCODES.SET_PIPELINE, new Uint8Array(buf));
    return this;
  }

  setComputePipeline(pipelineId) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, pipelineId, true);
    this._push(OPCODES.SET_COMPUTE_PIPELINE, new Uint8Array(buf));
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

  dispatchCompute(x, y = 1, z = 1) {
    const buf = new ArrayBuffer(12);
    const view = new DataView(buf);
    view.setUint32(0, x, true);
    view.setUint32(4, y, true);
    view.setUint32(8, z, true);
    this._push(OPCODES.DISPATCH_COMPUTE, new Uint8Array(buf));
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
  const ctx = {
    getCurrentTexture() {
      const view = { kind: 'GPUTextureView', id: `swap-${nextHandle++}` };
      return {
        createView() {
          return view;
        },
      };
    },
  };

  const device = {
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
    createComputePipeline(desc) {
      return {
        kind: 'GPUComputePipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayout(idx) {
          return { kind: 'GPUBindGroupLayout', idx };
        },
      };
    },
    createCommandEncoder() {
      const cmdBuf = { kind: 'GPUCommandBuffer', handle: nextHandle++ };
      return {
        beginRenderPass() {
          return {
            setPipeline() {},
            draw() {},
            end() {},
          };
        },
        beginComputePass() {
          return {
            setPipeline() {},
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
      submit() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
  };

  return { device, ctx };
}

class MockElement {
  constructor(tagName) {
    this.tagName = String(tagName).toUpperCase();
    this.children = [];
    this.parentNode = null;
    this.attributes = new Map();
    this.style = { display: '', width: '' };
    this.hidden = false;
    this.textContent = '';
    this.className = '';
  }

  setAttribute(name, value) {
    this.attributes.set(String(name), String(value));
    if (name === 'class') this.className = String(value);
  }

  getAttribute(name) {
    return this.attributes.has(String(name)) ? this.attributes.get(String(name)) : null;
  }

  appendChild(child) {
    if (child.parentNode && child.parentNode !== this) {
      child.parentNode.removeChild(child);
    }
    child.parentNode = this;
    this.children.push(child);
    return child;
  }

  removeChild(child) {
    const idx = this.children.indexOf(child);
    if (idx >= 0) {
      this.children.splice(idx, 1);
      child.parentNode = null;
    }
    return child;
  }

  querySelectorAll(selector) {
    const out = [];
    const match = (el) => {
      if (selector.startsWith('.')) {
        return String(el.className || '')
          .split(/\s+/)
          .includes(selector.slice(1));
      }
      return el.tagName === selector.toUpperCase();
    };
    const walk = (nodes) => {
      for (const child of nodes) {
        if (match(child)) out.push(child);
        walk(child.children);
      }
    };
    walk(this.children);
    return out;
  }

  querySelector(selector) {
    const found = this.querySelectorAll(selector);
    return found.length > 0 ? found[0] : null;
  }
}

function createMockDocument() {
  const head = new MockElement('head');
  const body = new MockElement('body');
  const documentElement = new MockElement('html');
  documentElement.appendChild(head);
  documentElement.appendChild(body);
  return {
    head,
    body,
    documentElement,
    createElement(tag) {
      return new MockElement(tag);
    },
    querySelector(selector) {
      return head.querySelector(selector) || body.querySelector(selector);
    },
  };
}

// ---------------------------------------------------------------------------
// 1. buildGpuTelemetrySnapshot shape matches GpuBootPanelController.update(...)
// ---------------------------------------------------------------------------

test('G30 buildGpuTelemetrySnapshot returns exact shape compatible with GpuBootPanelController.update()', () => {
  assert.equal(DEFAULT_TELEMETRY_INTERVAL_MS, 250);
  assert.equal(DEFAULT_TELEMETRY_MISSING_LIMIT, 64);

  const mock = createMockWebGpu();
  const vsHash = formatHashHex(fnv1a64(PACK_VS_WGSL));
  const fallbackFsHash = '0x000000000000f001';
  const missingFsHash = '0xdeadbeef00000002';
  const missingCsHash = '0xfeedface00000003';

  const packBuf = buildWgslPackBuffer([
    { hash: vsHash, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: PACK_VS_WGSL },
    { hash: fallbackFsHash, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: S9_FALLBACK_FS_WGSL },
  ]);

  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    wgslPackBuffer: packBuf,
    fallbackShaderHashes: { fragment: fallbackFsHash },
  });

  const frameStream = new PacketBuilder()
    .beginFrame()
    .createShaderModule(1, SHADER_STAGE.VERTEX, vsHash)
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, missingFsHash)
    .createShaderModule(3, SHADER_STAGE.FRAGMENT, missingFsHash)
    .createShaderModule(4, SHADER_STAGE.COMPUTE, missingCsHash)
    .createRenderPipeline(10, 1, 2, 0)
    .createComputePipeline(11, 4)
    .setComputePipeline(11)
    .dispatchCompute(4, 1, 1)
    .setPipeline(10)
    .draw(3, 1, 0, 0)
    .present()
    .finish();

  processor.executeCommandBuffer(frameStream);

  const snapshot = buildGpuTelemetrySnapshot(processor, { now: () => 1700000001234 });

  assert.equal(snapshot.type, 'gpu-telemetry');
  assert.equal(snapshot.timestamp, 1700000001234);

  assert.deepEqual(snapshot.frameStats, {
    framesPresented: 1,
    droppedFrames: 0,
    drawCalls: 1,
    realPackDrawCount: 0,
    fallbackDrawCount: 1,
    computeDispatches: 1,
    packetsProcessed: 12,
  });

  assert.deepEqual(snapshot.fallbackStats, {
    totalLookups: 4,
    totalResolves: 4,
    packHits: 1,
    fallbackHits: 2,
    builtinHits: 1,
    hitRatePct: 25,
    uniqueMissing: 2,
    missingCount: 2,
    overflowMissingCount: 0,
  });

  assert.equal(snapshot.missingShaders.length, 2);
  assert.equal(snapshot.missingShaders[0].hash, missingFsHash);
  assert.equal(snapshot.missingShaders[0].stage, 'fragment');
  assert.equal(snapshot.missingShaders[0].hits, 2);
  assert.equal(snapshot.missingShaders[0].hitCount, 2);
  assert.equal(snapshot.missingShaders[1].hash, missingCsHash);
  assert.equal(snapshot.missingShaders[1].stage, 'compute');
  assert.equal(snapshot.missingShaders[1].hits, 1);
  assert.equal(snapshot.missingShaders[1].hitCount, 1);

  // Feed telemetry snapshot directly into GpuBootPanelController.update(...)
  const doc = createMockDocument();
  const panel = new GpuBootPanelController({ documentObj: doc, now: () => 1700000001234 });
  const state = panel.update(snapshot);

  assert.equal(state.frameStats.framesPresented, 1);
  assert.equal(state.frameStats.droppedFrames, 0);
  assert.equal(state.fallbackStats.packHits, 1);
  assert.equal(state.fallbackStats.fallbackHits, 2);
  assert.equal(state.fallbackStats.builtinHits, 1);
  assert.equal(state.fallbackStats.totalLookups, 4);
  assert.equal(state.fallbackStats.hitRatePct, 25);
  assert.equal(state.hitRateText, '25.0%');
  assert.equal(state.missingShaders.length, 2);
  assert.equal(state.missingShaders[0].hash, missingFsHash);
  assert.equal(state.missingShaders[0].hits, 2);
  assert.equal(state.missingShaders[1].hash, missingCsHash);
  assert.equal(state.missingShaders[1].hits, 1);

  const exported = JSON.parse(panel.exportMissingShadersJson());
  assert.equal(exported.gpu.framesPresented, 1);
  assert.equal(exported.gpu.packHits, 1);
  assert.equal(exported.gpu.fallbackHits, 2);
  assert.equal(exported.gpu.builtinHits, 1);
  assert.equal(exported.missingShaderCount, 2);

  panel.destroy();
});

// ---------------------------------------------------------------------------
// 2. buildGpuTelemetrySnapshot sanitization, limit, and null/disabled processor
// ---------------------------------------------------------------------------

test('G30 buildGpuTelemetrySnapshot sanitizes missingShaders, enforces limit, and handles null/disabled processor', () => {
  const nullSnap = buildGpuTelemetrySnapshot(null, { timestamp: 42 });
  assert.deepEqual(nullSnap, {
    type: 'gpu-telemetry',
    frameStats: {
      framesPresented: 0,
      droppedFrames: 0,
      drawCalls: 0,
      realPackDrawCount: 0,
      fallbackDrawCount: 0,
      computeDispatches: 0,
      packetsProcessed: 0,
    },
    fallbackStats: {
      totalLookups: 0,
      totalResolves: 0,
      packHits: 0,
      fallbackHits: 0,
      builtinHits: 0,
      hitRatePct: 0,
      uniqueMissing: 0,
      missingCount: 0,
      overflowMissingCount: 0,
    },
    missingShaders: [],
    timestamp: 42,
  });

  // Disabled ShaderFallbackRouter returns clean zeroed fallbackStats.
  const mock = createMockWebGpu();
  const disabledProc = new WgpuCommandProcessor(mock.device, mock.ctx, {
    enableShaderFallback: false,
  });
  const disabledSnap = buildGpuTelemetrySnapshot(disabledProc, { timestamp: 99 });
  assert.equal(disabledSnap.fallbackStats.totalResolves, 0);
  assert.equal(disabledSnap.fallbackStats.missingCount, 0);
  assert.deepEqual(disabledSnap.missingShaders, []);

  // Synthetic dirty processor must have host paths, raw buffers, and invalid hashes stripped.
  const dirtyProcessor = {
    getStats() {
      return {
        frameCount: 3,
        droppedFrames: 1,
        drawCalls: 10,
        computeDispatches: 2,
        packetCount: 45,
      };
    },
    getShaderFallbackStats() {
      return {
        totalLookups: 5,
        packHits: 2,
        fallbackHits: 1,
        builtinHits: 2,
        uniqueMissing: 3,
        overflowMissingCount: 0,
        hitRatePct: 40,
      };
    },
    getMissingShaders({ limit }) {
      assert.equal(limit, 2);
      return [
        {
          hash: 'C:\\Users\\secret\\game_scarlett_master.exe',
          stage: 'vertex',
          hitCount: 99,
        },
        {
          hash: '0x1234',
          stage: 'PS',
          hitCount: 7,
          firstSeenMs: 10,
          lastSeenMs: 20,
          hostPath: '/Users/secret/gta6.rpf',
          rawBytes: new Uint8Array([0xde, 0xad, 0xbe, 0xef]),
        },
        {
          hash: '0xabcdef0123456789',
          stage: 'compute',
          hits: 3,
          wgsl: '@compute fn leak() {}',
        },
        {
          hash: '0x00000000000000ff',
          stage: 'vertex',
          hitCount: 1,
        },
      ];
    },
  };

  const cleanSnap = buildGpuTelemetrySnapshot(dirtyProcessor, { limit: 2, timestamp: 500 });
  assert.equal(cleanSnap.frameStats.framesPresented, 3);
  assert.equal(cleanSnap.frameStats.droppedFrames, 1);
  assert.equal(cleanSnap.missingShaders.length, 2);
  assert.deepEqual(cleanSnap.missingShaders[0], {
    hash: '0x0000000000001234',
    stage: 'fragment',
    hits: 7,
    hitCount: 7,
    firstSeenMs: 10,
    lastSeenMs: 20,
  });
  assert.deepEqual(cleanSnap.missingShaders[1], {
    hash: '0xabcdef0123456789',
    stage: 'compute',
    hits: 3,
    hitCount: 3,
  });

  const serialized = JSON.stringify(cleanSnap);
  assert.equal(serialized.includes('secret'), false);
  assert.equal(serialized.includes('rawBytes'), false);
  assert.equal(serialized.includes('hostPath'), false);
  assert.equal(serialized.includes('leak'), false);
});

// ---------------------------------------------------------------------------
// 3. Frame-1 immediate emission, throttling of rapid frames, and request-gpu-telemetry
// ---------------------------------------------------------------------------

test('G30 createWgpuWorkerHandler emits immediately on Frame 1 Present, throttles rapid frames, and responds to request-gpu-telemetry', async () => {
  const mock = createMockWebGpu();
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx);
  const posted = [];
  let fakeNow = 1000;

  const handler = createWgpuWorkerHandler({
    processor,
    postMessage: (msg) => posted.push(msg),
    now: () => fakeNow,
  });

  // Batch without Present must NOT emit gpu-telemetry.
  const noPresentBuf = new PacketBuilder().beginFrame().clearColor(0.1, 0.2, 0.3, 1.0).finish();
  await handler.handleMessage({ cmdBuffer: noPresentBuf });
  assert.equal(posted.filter((m) => m.type === 'gpu-telemetry').length, 0);

  // Frame 1 Present (opcode 7) emits immediately with zero delay.
  const frameBuf = () => new PacketBuilder().beginFrame().clearColor(0.2, 0.4, 0.6, 1.0).present().finish();
  await handler.handleMessage({ cmdBuffer: frameBuf() });

  const telemetryAfterFrame1 = posted.filter((m) => m.type === 'gpu-telemetry');
  assert.equal(telemetryAfterFrame1.length, 1);
  assert.equal(telemetryAfterFrame1[0].frameStats.framesPresented, 1);
  assert.equal(telemetryAfterFrame1[0].timestamp, 1000);
  assert.ok(posted.some((m) => m.frame === 1));

  // Rapid subsequent frames within default 250ms throttle window do NOT spam gpu-telemetry.
  fakeNow = 1016;
  await handler.handleMessage({ cmdBuffer: frameBuf() });
  fakeNow = 1033;
  await handler.handleMessage({ cmdBuffer: frameBuf() });
  fakeNow = 1249;
  await handler.handleMessage({ cmdBuffer: frameBuf() });

  assert.equal(processor.frameCount, 4);
  assert.equal(posted.filter((m) => m.type === 'gpu-telemetry').length, 1);

  // On-demand { type: 'request-gpu-telemetry' } responds immediately even while throttled!
  fakeNow = 1249;
  await handler.handleMessage({ type: 'request-gpu-telemetry' });
  const afterRequest = posted.filter((m) => m.type === 'gpu-telemetry');
  assert.equal(afterRequest.length, 2);
  assert.equal(afterRequest[1].frameStats.framesPresented, 4);
  assert.equal(afterRequest[1].timestamp, 1249);

  // Once >= 250ms elapsed since Frame 1 (1000 -> 1250), next Present emits telemetry again.
  fakeNow = 1250;
  await handler.handleMessage({ cmdBuffer: frameBuf() });
  const afterThrottleWindow = posted.filter((m) => m.type === 'gpu-telemetry');
  assert.equal(afterThrottleWindow.length, 3);
  assert.equal(afterThrottleWindow[2].frameStats.framesPresented, 5);
  assert.equal(afterThrottleWindow[2].timestamp, 1250);

  // Subsequent frame 10ms later is throttled again.
  fakeNow = 1260;
  await handler.handleMessage({ cmdBuffer: frameBuf() });
  assert.equal(posted.filter((m) => m.type === 'gpu-telemetry').length, 3);
});

// ---------------------------------------------------------------------------
// 4. Configurable telemetryIntervalMs, drainRing, submit/executeBatch, and pre-init request
// ---------------------------------------------------------------------------

test('G30 createWgpuWorkerHandler supports custom telemetryIntervalMs, drainRing, submit/executeBatch, and pre-init telemetry request', async () => {
  const postedPreInit = [];
  const preInitHandler = createWgpuWorkerHandler({
    postMessage: (m) => postedPreInit.push(m),
    now: () => 500,
  });
  await preInitHandler.handleMessage({ type: 'request-gpu-telemetry', id: 9 });
  assert.equal(postedPreInit.length, 1);
  assert.equal(postedPreInit[0].type, 'gpu-telemetry');
  assert.equal(postedPreInit[0].id, 9);
  assert.equal(postedPreInit[0].frameStats.framesPresented, 0);
  assert.deepEqual(postedPreInit[0].missingShaders, []);

  // Custom telemetryIntervalMs = 50 via drainRing + submit/executeBatch
  const mock = createMockWebGpu();
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx);
  const producer = GpuRingBuffer.create(512);
  processor.attachRing(producer.sab, { GpuRingBuffer });

  const posted = [];
  let fakeNow = 2000;
  const handler = createWgpuWorkerHandler({
    processor,
    postMessage: (m) => posted.push(m),
    telemetryIntervalMs: 50,
    now: () => fakeNow,
  });

  const frameStream = new PacketBuilder().beginFrame().present().finish();

  // Frame 1 via drainRing -> emits immediately.
  assert.equal(producer.writeBytes(new Uint8Array(frameStream)), true);
  await handler.handleMessage({ drainRing: true });
  assert.equal(posted.filter((m) => m.type === 'gpu-telemetry').length, 1);
  assert.equal(posted.filter((m) => m.type === 'gpu-telemetry')[0].frameStats.framesPresented, 1);

  // Frame 2 via { type: 'submit', buffer } 20ms later -> throttled.
  fakeNow = 2020;
  await handler.handleMessage({ type: 'submit', buffer: frameStream });
  assert.equal(processor.frameCount, 2);
  assert.equal(posted.filter((m) => m.type === 'gpu-telemetry').length, 1);

  // Frame 3 via { type: 'executeBatch', buffer } 50ms after Frame 1 -> emits!
  fakeNow = 2050;
  await handler.handleMessage({ type: 'executeBatch', buffer: frameStream });
  const telemetries = posted.filter((m) => m.type === 'gpu-telemetry');
  assert.equal(telemetries.length, 2);
  assert.equal(telemetries[1].frameStats.framesPresented, 3);
  assert.equal(telemetries[1].timestamp, 2050);
});

// ---------------------------------------------------------------------------
// 5. Missing-shader records from hash-only CreateShaderModule packets appear in telemetry
// ---------------------------------------------------------------------------

test('G30 hash-only CreateShaderModule misses populate missingShaders and fallbackStats in emitted gpu-telemetry', async () => {
  const mock = createMockWebGpu();
  const packVsHash = formatHashHex(fnv1a64(PACK_VS_WGSL));
  const packFsHash = formatHashHex(fnv1a64(PACK_FS_WGSL));
  const missVsHash = '0xaaaaaaaaaaaaaaaa';
  const missFsHash = '0x00000000deadbeef';
  const missCsHash = '0x123456789abcdef0';

  const packBuf = buildWgslPackBuffer([
    { hash: packVsHash, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: PACK_VS_WGSL },
    { hash: packFsHash, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: PACK_FS_WGSL },
  ]);

  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    wgslPackBuffer: packBuf,
  });

  const posted = [];
  let fakeNow = 5000;
  const handler = createWgpuWorkerHandler({
    processor,
    postMessage: (msg) => posted.push(msg),
    now: () => fakeNow,
  });

  const frame1 = new PacketBuilder()
    .beginFrame()
    .createShaderModule(1, SHADER_STAGE.VERTEX, packVsHash)
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, `hash:${packFsHash}`)
    .createShaderModule(3, SHADER_STAGE.VERTEX, missVsHash)
    .createShaderModule(4, SHADER_STAGE.FRAGMENT, 'hash:0xDEADBEEF')
    .createShaderModule(5, SHADER_STAGE.FRAGMENT, 'wgpk:00000000deadbeef')
    .createShaderModule(6, SHADER_STAGE.COMPUTE, '123456789abcdef0')
    .createRenderPipeline(10, 1, 4, 0)
    .setPipeline(10)
    .draw(3, 1, 0, 0)
    .present()
    .finish();

  await handler.handleMessage({ cmdBuffer: frame1 });

  const telemetryMsgs = posted.filter((m) => m.type === 'gpu-telemetry');
  assert.equal(telemetryMsgs.length, 1);
  const t1 = telemetryMsgs[0];

  assert.equal(t1.frameStats.framesPresented, 1);
  assert.equal(t1.frameStats.drawCalls, 1);
  assert.equal(t1.fallbackStats.packHits, 2);
  assert.equal(t1.fallbackStats.fallbackHits, 0);
  assert.equal(t1.fallbackStats.builtinHits, 4);
  assert.equal(t1.fallbackStats.totalResolves, 6);
  assert.equal(t1.fallbackStats.missingCount, 3);
  assert.equal(t1.fallbackStats.hitRatePct, 33.33);

  assert.equal(t1.missingShaders.length, 3);
  // Sorted by hitCount descending: missFsHash (2) first, then missCsHash & missVsHash (1 each).
  assert.equal(t1.missingShaders[0].hash, missFsHash);
  assert.equal(t1.missingShaders[0].stage, 'fragment');
  assert.equal(t1.missingShaders[0].hits, 2);
  assert.equal(t1.missingShaders[0].hitCount, 2);

  const byHash = new Map(t1.missingShaders.map((r) => [r.hash, r]));
  assert.equal(byHash.get(missVsHash).stage, 'vertex');
  assert.equal(byHash.get(missVsHash).hits, 1);
  assert.equal(byHash.get(missCsHash).stage, 'compute');
  assert.equal(byHash.get(missCsHash).hits, 1);
});

// ---------------------------------------------------------------------------
// 6. Issue #659: bindGroupKindMismatches surfaces in GPU telemetry snapshots
// ---------------------------------------------------------------------------

test('#659 bindGroupKindMismatches propagates through buildGpuTelemetrySnapshot and createWgpuWorkerHandler', async () => {
  const { installRenderStatePackets, V2_OPCODES } = require('../web/workers/render_state_packets.js');
  const mock = createMockWebGpu();
  let nextObj = 100;
  mock.device.createBuffer = (desc) => ({
    kind: 'GPUBuffer',
    handle: nextObj++,
    size: desc.size,
    usage: desc.usage,
    desc,
  });
  mock.device.createTexture = (desc) => {
    const handle = nextObj++;
    return {
      kind: 'GPUTexture',
      handle,
      format: desc.format,
      usage: desc.usage,
      dimension: desc.dimension || '2d',
      desc,
      createView(viewDesc = {}) {
        return { kind: 'GPUTextureView', handle: nextObj++, textureHandle: handle, desc: viewDesc };
      },
    };
  };
  mock.device.createSampler = (desc = {}) => ({ kind: 'GPUSampler', handle: nextObj++, desc });
  mock.device.createBindGroupLayout = (desc) => ({
    kind: 'GPUBindGroupLayout',
    handle: nextObj++,
    desc,
  });
  mock.device.createBindGroup = (desc) => ({ kind: 'GPUBindGroup', handle: nextObj++, desc });

  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, { format: 'rgba8unorm' });
  const rsHandler = installRenderStatePackets(processor);

  // Clean processor reports bindGroupKindMismatches === 0.
  assert.equal(
    buildGpuTelemetrySnapshot(processor, { timestamp: 100 }).renderStateStats.bindGroupKindMismatches,
    0
  );

  // Helpers to build v2 CreateBindGroupLayout + CreateBindGroup2 packets.
  const bglPayload = new ArrayBuffer(24 + 3 * 24);
  const bglView = new DataView(bglPayload);
  bglView.setUint32(0, 20, true); // id
  bglView.setUint32(4, 3, true); // count
  // Entry 0: uniform buffer @0
  bglView.setUint32(24 + 0, 0, true);
  bglView.setUint32(24 + 4, 4, true); // COMPUTE
  bglView.setUint8(24 + 8, 0); // buffer
  // Entry 1: texture @1
  bglView.setUint32(24 + 24 + 0, 1, true);
  bglView.setUint32(24 + 24 + 4, 4, true);
  bglView.setUint8(24 + 24 + 8, 2); // texture
  bglView.setUint8(24 + 24 + 10, 2); // 2d
  // Entry 2: sampler @2
  bglView.setUint32(24 + 48 + 0, 2, true);
  bglView.setUint32(24 + 48 + 4, 4, true);
  bglView.setUint8(24 + 48 + 8, 1); // sampler

  const bgPayload = new ArrayBuffer(16 + 3 * 48);
  const bgView = new DataView(bgPayload);
  bgView.setUint32(0, 30, true); // id
  bgView.setUint32(4, 20, true); // layoutId
  bgView.setUint32(8, 3, true); // count
  // Binding 0 (expects buffer) <- sampler 301
  bgView.setUint32(16 + 0, 0, true);
  bgView.setUint32(16 + 4, 2, true); // SAMPLER
  bgView.setUint32(16 + 8, 301, true);
  // Binding 1 (expects texture) <- buffer 101
  bgView.setUint32(16 + 48 + 0, 1, true);
  bgView.setUint32(16 + 48 + 4, 0, true); // BUFFER
  bgView.setUint32(16 + 48 + 8, 101, true);
  bgView.setBigUint64(16 + 48 + 24, 256n, true);
  // Binding 2 (expects sampler) <- texture 201
  bgView.setUint32(16 + 96 + 0, 2, true);
  bgView.setUint32(16 + 96 + 4, 1, true); // TEXTURE_VIEW
  bgView.setUint32(16 + 96 + 8, 201, true);

  // Create buffer 101, texture 201, sampler 301 via processor maps & execute packets + Present.
  processor.buffers.set(101, mock.device.createBuffer({ size: 256, usage: 0x0040 | 0x0080 | 0x0008 }));
  processor.textures.set(
    201,
    mock.device.createTexture({ size: { width: 4, height: 4, depthOrArrayLayers: 1 }, format: 'rgba8unorm', usage: 0x04 })
  );
  processor.samplers.set(301, mock.device.createSampler({}));

  const builder = new PacketBuilder();
  builder._push(V2_OPCODES.CREATE_BIND_GROUP_LAYOUT, new Uint8Array(bglPayload));
  builder._push(V2_OPCODES.CREATE_BIND_GROUP2, new Uint8Array(bgPayload));
  builder.beginFrame().present();

  const posted = [];
  const workerHandler = createWgpuWorkerHandler({
    processor,
    postMessage: (msg) => posted.push(msg),
    now: () => 9000,
  });

  await workerHandler.handleMessage({ cmdBuffer: builder.finish() });
  assert.equal(rsHandler.stats.errors, 0, rsHandler.lastError);
  assert.equal(rsHandler.telemetry.bindGroupKindMismatches, 3);

  const emitted = posted.filter((m) => m.type === 'gpu-telemetry');
  assert.equal(emitted.length, 1);
  assert.equal(emitted[0].renderStateStats.bindGroupKindMismatches, 3);

  // On-demand request-gpu-telemetry also reports bindGroupKindMismatches.
  await workerHandler.handleMessage({ type: 'request-gpu-telemetry', id: 77 });
  const requested = posted.filter((m) => m.type === 'gpu-telemetry' && m.id === 77);
  assert.equal(requested.length, 1);
  assert.equal(requested[0].renderStateStats.bindGroupKindMismatches, 3);

  // A telemetry source exposing only bindGroupKindMismatches is recognized by RENDER_STATE_TELEMETRY_KEYS.
  const onlyMismatches = buildGpuTelemetrySnapshot(
    {
      getStats: () => ({}),
      renderStateProcessor: { getTelemetry: () => ({ bindGroupKindMismatches: 6.9 }) },
    },
    { timestamp: 9001 }
  );
  assert.equal(onlyMismatches.renderStateStats.bindGroupKindMismatches, 6);
  assert.equal(onlyMismatches.renderStateStats.autoLayoutFallbacks, 0);
});


// ---------------------------------------------------------------------------
// 7. W295: W284 drainBatchZeroAlloc hot path + legacy drainTo fallback
// ---------------------------------------------------------------------------

test('W295 drainRing prefers GpuRingBuffer.drainBatchZeroAlloc and surfaces zeroAllocDrains with clean drainTo fallback', async () => {
  const mock = createMockWebGpu();
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx);
  const producer = GpuRingBuffer.create(4096);
  processor.attachRing(producer.sab, { GpuRingBuffer });

  let zeroAllocCalls = 0;
  const origZeroAlloc = processor.ring.drainBatchZeroAlloc.bind(processor.ring);
  processor.ring.drainBatchZeroAlloc = (...args) => {
    zeroAllocCalls += 1;
    return origZeroAlloc(...args);
  };
  processor.ring.drainTo = () => {
    throw new Error('drainTo must not be called when drainBatchZeroAlloc is available');
  };

  const posted = [];
  let fakeNow = 1000;
  const handler = createWgpuWorkerHandler({
    processor,
    postMessage: (m) => posted.push(m),
    telemetryIntervalMs: 50,
    now: () => fakeNow,
  });

  const frameStream = new PacketBuilder().beginFrame().clearColor(0.1, 0.2, 0.3, 1.0).present().finish();
  assert.equal(producer.writeBytes(new Uint8Array(frameStream)), true);
  await handler.handleMessage({ drainRing: true });

  assert.equal(zeroAllocCalls, 1);
  assert.equal(processor.frameCount, 1);
  assert.equal(processor.ring.zeroAllocDrains, 1);
  assert.equal(processor.getStats().zeroAllocDrains, 1);

  const telemetry = posted.filter((m) => m.type === 'gpu-telemetry');
  assert.equal(telemetry.length, 1);
  assert.equal(telemetry[0].zeroAllocDrains, 1);

  // Second drain via worker message also increments zeroAllocDrains and surfaces in request-gpu-stats.
  fakeNow = 1060;
  assert.equal(producer.writeBytes(new Uint8Array(frameStream)), true);
  await handler.handleMessage({ type: 'drain-ring' });
  await handler.handleMessage({ type: 'request-gpu-stats', id: 'w295-ring' });

  const statsReply = posted.find((m) => m.type === 'gpu-stats' && m.id === 'w295-ring');
  assert.ok(statsReply);
  assert.equal(statsReply.zeroAllocDrains, 2);

  // Graceful fallback: legacy mock ring exposing only drainTo (no drainBatchZeroAlloc).
  const legacyMock = createMockWebGpu();
  const legacyProc = new WgpuCommandProcessor(legacyMock.device, legacyMock.ctx);
  let legacyDrainToCalls = 0;
  const legacyRing = {
    capacityBytes: 1024,
    drainTo(cb) {
      legacyDrainToCalls += 1;
      cb(new Uint8Array(frameStream));
      return frameStream.byteLength;
    },
  };
  legacyProc.attachRing(legacyRing);
  const res = legacyProc.drainRing();
  assert.equal(legacyDrainToCalls, 1);
  assert.deepEqual(res, {
    frameCount: 1,
    presents: 1,
    bytesDrained: frameStream.byteLength,
  });
  assert.equal(legacyProc.getStats().zeroAllocDrains, 0);
  assert.equal(buildGpuTelemetrySnapshot(legacyProc, { timestamp: 2000 }).zeroAllocDrains, 0);
});

// ---------------------------------------------------------------------------
// 8. W295: W282 DrawExecutor reuseDrawResult: true & zero-alloc getFrameDrawSummary
// ---------------------------------------------------------------------------

test('W295 configures DrawExecutor with reuseDrawResult: true, uses zero-alloc getFrameDrawSummary({ reuse: true }), and tolerates legacy/frozen mocks', () => {
  const { DrawExecutor } = require('../web/workers/draw_executor.js');
  const mock = createMockWebGpu();

  // 1. Owned DrawExecutor created via enableDrawExecutor: true gets reuseDrawResult: true.
  const ownedProc = new WgpuCommandProcessor(mock.device, mock.ctx, {
    enableDrawExecutor: true,
    DrawExecutor,
  });
  assert.ok(ownedProc.drawExecutor instanceof DrawExecutor);
  assert.equal(ownedProc.drawExecutor._reuseDrawResult, true);

  // 2. Injected DrawExecutor is configured for reuseDrawResult: true and queried with { reuse: true }.
  const injectedExec = new DrawExecutor({ device: mock.device });
  assert.equal(injectedExec._reuseDrawResult, false);

  const summaryArgs = [];
  const origSummary = injectedExec.getFrameDrawSummary.bind(injectedExec);
  injectedExec.getFrameDrawSummary = (arg) => {
    summaryArgs.push(arg);
    return origSummary(arg);
  };

  const visibleEvents = [];
  const proc = new WgpuCommandProcessor(mock.device, mock.ctx, {
    drawExecutor: injectedExec,
    onFirstVisibleFrame: (e) => visibleEvents.push(e),
  });
  assert.equal(injectedExec._reuseDrawResult, true);

  // Execute a non-clear draw via DrawExecutor, then present via command buffer.
  proc.beginFrame(0);
  const mockPass = {
    setPipeline() {},
    setVertexBuffer() {},
    setIndexBuffer() {},
    setBindGroup() {},
    draw() {},
  };
  injectedExec.setPipeline({ kind: 'GPURenderPipeline', isRealPack: true });
  const r1 = injectedExec.draw(mockPass, 3, 1, 0, 0);
  const r2 = injectedExec.draw(mockPass, 3, 1, 0, 0);
  assert.equal(r1, r2, 'reuseDrawResult: true should return the same reusable draw result object');

  proc.executeCommandBuffer(new PacketBuilder().present().finish());
  assert.equal(summaryArgs.length, 1);
  assert.deepEqual(summaryArgs[0], { reuse: true });
  assert.equal(visibleEvents.length, 1);
  assert.equal(visibleEvents[0].firstVisibleFrame, true);
  assert.equal(visibleEvents[0].drawsThisFrame, 2);

  // 3. Frozen strict mock DrawExecutor whose getFrameDrawSummary rejects args falls back cleanly.
  let zeroArgSummaryCalls = 0;
  const strictMockExec = Object.freeze({
    beginFrame() {},
    getFrameDrawSummary(...args) {
      if (args.length > 0) {
        throw new TypeError('legacy mock does not accept options');
      }
      zeroArgSummaryCalls += 1;
      return {
        hasNonClearDraw: true,
        realPackDrawsThisFrame: 1,
        fallbackDrawsThisFrame: 0,
        drawsThisFrame: 1,
      };
    },
  });
  const strictProc = new WgpuCommandProcessor(mock.device, mock.ctx, {
    drawExecutor: strictMockExec,
  });
  strictProc.executeCommandBuffer(new PacketBuilder().beginFrame().present().finish());
  assert.equal(zeroArgSummaryCalls, 1);
  assert.equal(strictProc.firstVisibleFrame, true);
});

// ---------------------------------------------------------------------------
// 9. W295: W285 rtPool.reclaimOversizedIdle(2) on scene transitions & idle frames + VRAM telemetry
// ---------------------------------------------------------------------------

test('W295 triggers rtPool.reclaimOversizedIdle(2) on scene transitions and periodic idle frames, surfacing vramBudgetUtilization and reclaimedOversizedBytes', async () => {
  const { RenderTargetPool } = require('../web/workers/rt_pool.js');
  const mock = createMockWebGpu();
  let texHandle = 1;
  let destroyedTextures = 0;
  mock.device.createTexture = (desc) => {
    const handle = texHandle++;
    return {
      kind: 'GPUTexture',
      handle,
      desc,
      createView() {
        return { kind: 'GPUTextureView', handle: texHandle++, textureHandle: handle };
      },
      destroy() {
        destroyedTextures += 1;
      },
    };
  };

  const rtPool = new RenderTargetPool({
    device: mock.device,
    maxPoolBytes: 128 * 1024 * 1024,
    idleFrameTtl: 16,
  });
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    rtPool,
  });

  const posted = [];
  let fakeNow = 3000;
  const handler = createWgpuWorkerHandler({
    processor,
    postMessage: (m) => posted.push(m),
    telemetryIntervalMs: 50,
    now: () => fakeNow,
  });

  // Acquire and release a 4K Bink2-sized RT (3840x2160 rgba8unorm = 33,177,600 bytes >= 24 MiB).
  const expected4kBytes = 3840 * 2160 * 4;
  const rt4k = processor.acquireRenderTarget(
    { width: 3840, height: 2160, format: 'rgba8unorm' },
    'bink2-4k'
  );
  processor.releaseRenderTarget(rt4k);

  const beforeSnap = buildGpuTelemetrySnapshot(processor, { timestamp: 3000 });
  assert.ok(beforeSnap.vramBudgetUtilization > 0.24 && beforeSnap.vramBudgetUtilization < 0.26);
  assert.equal(beforeSnap.reclaimedOversizedBytes, 0);

  // Frame 1 (currentFrame -> 1): age is 1 < 2, so 4K RT stays pooled.
  const idleFrame = new PacketBuilder().beginFrame().present().finish();
  await handler.handleMessage({ cmdBuffer: idleFrame });
  assert.equal(destroyedTextures, 0);
  assert.equal(processor.getStats().reclaimedOversizedBytes, 0);

  // Frame 2 (currentFrame -> 2): periodic idle reclaim invokes reclaimOversizedIdle(2) -> 4K RT reclaimed!
  fakeNow = 3100;
  await handler.handleMessage({ cmdBuffer: idleFrame });
  assert.equal(destroyedTextures, 1);

  const afterStats = processor.getStats();
  assert.equal(afterStats.reclaimedOversizedBytes, expected4kBytes);
  assert.equal(afterStats.vramBudgetUtilization, 0);

  const afterTelemetry = posted.filter((m) => m.type === 'gpu-telemetry');
  assert.ok(afterTelemetry.length >= 2);
  const latest = afterTelemetry[afterTelemetry.length - 1];
  assert.equal(latest.reclaimedOversizedBytes, expected4kBytes);
  assert.equal(latest.vramBudgetUtilization, 0);

  // Explicit scene-transition message also triggers reclaimOversizedIdle(2).
  const rt4k2 = processor.acquireRenderTarget(
    { width: 3840, height: 2160, format: 'rgba8unorm' },
    'bink2-4k-second'
  );
  processor.releaseRenderTarget(rt4k2);
  rtPool.endFrame();
  rtPool.endFrame();
  await handler.handleMessage({ type: 'scene-transition', id: 'st-1' });
  const stReply = posted.find((m) => m.type === 'scene-transition' && m.id === 'st-1');
  assert.ok(stReply);
  assert.equal(stReply.reclaimedCount, 1);
  assert.equal(stReply.reclaimedBytes, expected4kBytes);
  assert.equal(processor.getStats().reclaimedOversizedBytes, expected4kBytes * 2);

  // Graceful fallback: mock rtPool without reclaimOversizedIdle must never throw.
  const mockNoReclaim = {
    isDestroyed: false,
    endFrame() {},
    getStats() {
      return { hitRate: 0.5, totalBytes: 16, maxPoolBytes: 64 };
    },
  };
  const fallbackProc = new WgpuCommandProcessor(mock.device, mock.ctx, {
    rtPool: mockNoReclaim,
  });
  fallbackProc.executeCommandBuffer(idleFrame);
  fallbackProc.executeCommandBuffer(idleFrame);
  const fbSnap = buildGpuTelemetrySnapshot(fallbackProc, { timestamp: 3200 });
  assert.equal(fbSnap.vramBudgetUtilization, 0.25);
  assert.equal(fbSnap.reclaimedOversizedBytes, 0);
});

// ---------------------------------------------------------------------------
// 10. W295: W281 fast-path PSO lookup + W268 negative compile-failure cache telemetry
// ---------------------------------------------------------------------------

test('W295 wires W281 getFastRenderPipeline/getFastComputePipeline and surfaces W268 negative compile-failure cache metrics', async () => {
  const { WebGpuPipelineCache } = require('../web/workers/pipeline_cache.js');
  const mock = createMockWebGpu();

  let renderCompileCalls = 0;
  let computeCompileCalls = 0;
  let failNextRender = false;
  const origCreateRender = mock.device.createRenderPipeline.bind(mock.device);
  const origCreateCompute = mock.device.createComputePipeline.bind(mock.device);
  mock.device.createRenderPipeline = (desc) => {
    renderCompileCalls += 1;
    if (failNextRender) {
      throw new Error('simulated invalid render pipeline C:\\Users\\secret\\shader.wgsl');
    }
    return origCreateRender(desc);
  };
  mock.device.createComputePipeline = (desc) => {
    computeCompileCalls += 1;
    return origCreateCompute(desc);
  };

  const pipelineCache = new WebGpuPipelineCache({
    device: mock.device,
    quarantineFailedKeys: true,
  });

  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    pipelineCache,
    captureValidationErrors: true,
  });

  const csWgsl = '@compute @workgroup_size(1) fn cs_main() {}\n';
  const badFsWgsl = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.0, 0.0, 1.0); }\n';

  // Two render pipelines sharing the same VS+FS and two compute pipelines sharing the same CS.
  const warmupStream = new PacketBuilder()
    .beginFrame()
    .createShaderModule(1, SHADER_STAGE.VERTEX, PACK_VS_WGSL)
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, PACK_FS_WGSL)
    .createShaderModule(3, SHADER_STAGE.COMPUTE, csWgsl)
    .createRenderPipeline(10, 1, 2, 0)
    .createRenderPipeline(11, 1, 2, 0)
    .createComputePipeline(20, 3)
    .createComputePipeline(21, 3)
    .present()
    .finish();

  processor.executeCommandBuffer(warmupStream);

  assert.equal(renderCompileCalls, 1, 'second identical render pipeline should hit W281 fast lookup');
  assert.equal(computeCompileCalls, 1, 'second identical compute pipeline should hit W281 fast lookup');
  assert.equal(processor.pipelines.get(10), processor.pipelines.get(11));
  assert.equal(processor.computePipelines.get(20), processor.computePipelines.get(21));

  // Trigger a render pipeline compile failure -> quarantined in W268 negative cache.
  failNextRender = true;
  const failStream = new PacketBuilder()
    .beginFrame()
    .createShaderModule(4, SHADER_STAGE.FRAGMENT, badFsWgsl)
    .createRenderPipeline(30, 1, 4, 0)
    .createRenderPipeline(31, 1, 4, 0)
    .present()
    .finish();
  processor.executeCommandBuffer(failStream);

  // Only 1 additional compile attempt for the failed shader; second attempt short-circuited by negative cache!
  assert.equal(renderCompileCalls, 2);
  assert.equal(processor.pipelines.has(30), false);
  assert.equal(processor.pipelines.has(31), false);

  const stats = processor.getStats();
  assert.ok(stats.fastLookupHits >= 2);
  assert.ok(stats.fastLookupHitRate > 0);
  assert.equal(stats.negativeCacheHits, 1);
  assert.equal(stats.negativeCacheSize, 1);
  assert.equal(stats.negativeCacheEvictions, 0);

  const snap = buildGpuTelemetrySnapshot(processor, { timestamp: 4000 });
  assert.equal(snap.pipelineCacheStats.fastLookupHits, stats.fastLookupHits);
  assert.equal(snap.pipelineCacheStats.negativeCacheHits, 1);
  assert.equal(snap.pipelineCacheStats.negativeCacheSize, 1);
  assert.equal(snap.pipelineCacheStats.negativeCacheEvictions, 0);

  // Verify no host path leaked into telemetry JSON.
  assert.equal(JSON.stringify(snap).includes('secret'), false);
});

// ---------------------------------------------------------------------------
// 11. W295: Full 30 FPS telemetry (W286 swapchain + W267 ResourceTable) & adversarial sanitization
// ---------------------------------------------------------------------------

test('W295 surfaces W286 swapchain pacing & W267 bindGroupHitRate in periodic telemetry and sanitizes dirty/partial mocks', async () => {
  const { WgpuSwapchain } = require('../web/workers/wgpu_swapchain.js');
  const { ResourceTable } = require('../web/workers/resource_table.js');
  const mock = createMockWebGpu();
  let nextObj = 50;
  mock.device.createBuffer = (desc) => ({ kind: 'GPUBuffer', handle: nextObj++, size: desc.size, usage: desc.usage });
  mock.device.createBindGroup = (desc) => ({ kind: 'GPUBindGroup', handle: nextObj++, desc });
  mock.device.addEventListener = () => {};
  mock.device.removeEventListener = () => {};
  mock.device.lost = new Promise(() => {});

  const canvas = { width: 1280, height: 720, getContext: () => mock.ctx };
  mock.ctx.configure = () => {};
  mock.ctx.unconfigure = () => {};

  let fakeNow = 1000;
  const nowFn = () => fakeNow;

  const swapchain = new WgpuSwapchain({
    canvas,
    device: mock.device,
    context: mock.ctx,
    format: 'bgra8unorm',
    enableDepth: false,
    now: nowFn,
  });
  swapchain.configure({ width: 1280, height: 720, devicePixelRatio: 1 });

  const resourceTable = new ResourceTable(mock.device);
  const layout = { kind: 'GPUBindGroupLayout', handle: 1 };
  const buf = mock.device.createBuffer({ size: 256, usage: 0x40 });
  resourceTable.setSlot(0, buf);
  // 1 miss + 3 hits => bindGroupHitRate === 0.75
  resourceTable.getOrCreateBindGroup('bg-0', { layout, slots: [0] });
  resourceTable.getOrCreateBindGroup('bg-0', { layout, slots: [0] });
  resourceTable.getOrCreateBindGroup('bg-0', { layout, slots: [0] });
  resourceTable.getOrCreateBindGroup('bg-0', { layout, slots: [0] });
  assert.equal(resourceTable.getStats().bindGroupHitRate, 0.75);

  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    now: nowFn,
    swapchain,
    resourceTable,
    gpuAdapter: { vendor: 'amd', architecture: 'rdna2', device: 'xbox', description: 'test' },
  });

  const posted = [];
  const handler = createWgpuWorkerHandler({
    processor,
    postMessage: (m) => posted.push(m),
    telemetryIntervalMs: 50,
    now: nowFn,
  });

  const drawFrame = new PacketBuilder()
    .beginFrame()
    .createShaderModule(1, SHADER_STAGE.VERTEX, PACK_VS_WGSL)
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, PACK_FS_WGSL)
    .createRenderPipeline(10, 1, 2, 0)
    .setPipeline(10)
    .draw(3, 1, 0, 0)
    .present()
    .finish();

  // Frame 1 at t=1000, Frame 2 at t=1030 (30ms, <=33.3ms), Frame 3 at t=1060 (30ms), Frame 4 at t=1140 (80ms stall).
  await handler.handleMessage({ cmdBuffer: drawFrame });
  fakeNow = 1030;
  await handler.handleMessage({ cmdBuffer: drawFrame });
  fakeNow = 1060;
  await handler.handleMessage({ cmdBuffer: drawFrame });
  fakeNow = 1140;
  await handler.handleMessage({ cmdBuffer: drawFrame });

  // Existing message types untouched: first-visible-frame + gpu-telemetry + request-gpu-stats.
  assert.equal(posted.filter((m) => m.type === 'first-visible-frame').length, 1);

  await handler.handleMessage({ type: 'request-gpu-telemetry', id: 'tel-w295' });
  const tel = posted.find((m) => m.type === 'gpu-telemetry' && m.id === 'tel-w295');
  assert.ok(tel);
  assert.ok(Math.abs(tel.fps30ComplianceRate - 2 / 3) < 1e-6);
  assert.ok(tel.frameJitterMs > 0);
  assert.equal(tel.longestStallMs, 80);
  assert.equal(tel.bindGroupHitRate, 0.75);
  assert.equal(tel.vramBudgetUtilization, 0);
  assert.equal(tel.zeroAllocDrains, 0);
  assert.equal(tel.reclaimedOversizedBytes, 0);
  assert.equal(tel.gpuAdapter.vendor, 'amd');

  await handler.handleMessage({ type: 'request-gpu-stats', id: 'stats-w295' });
  const stReply = posted.find((m) => m.type === 'gpu-stats' && m.id === 'stats-w295');
  assert.ok(stReply);
  assert.ok(Math.abs(stReply.fps30ComplianceRate - 2 / 3) < 1e-6);
  assert.ok(stReply.frameJitterMs > 0);
  assert.equal(stReply.longestStallMs, 80);
  assert.equal(stReply.bindGroupHitRate, 0.75);

  // Dirty / out-of-range / throwing mocks must clamp safely and never throw.
  const dirtyMockProcessor = {
    getStats() {
      return {
        frameCount: 2,
        fps30ComplianceRate: 99,
        frameJitterMs: -5,
        longestStallMs: Number.POSITIVE_INFINITY,
        vramBudgetUtilization: -2,
        bindGroupHitRate: Number.NaN,
        zeroAllocDrains: 4.9,
        reclaimedOversizedBytes: -100,
      };
    },
    swapchain: {
      getStats() {
        throw new Error('broken swapchain mock');
      },
    },
    rtPool: {
      getStats() {
        return { vramBudgetUtilization: 4.5, reclaimedOversizedBytes: 1024.7 };
      },
    },
    resourceTable: {
      getStats() {
        return { bindGroupHitRate: 2.5 };
      },
    },
    pipelineCache: {
      getStats() {
        return {
          fastLookupHits: 10,
          fastLookupMisses: -3,
          fastLookupHitRate: 1.5,
          negativeCacheHits: 3.8,
          negativeCacheSize: 2,
          negativeCacheEvictions: -1,
        };
      },
    },
  };

  const sanitized = buildGpuTelemetrySnapshot(dirtyMockProcessor, { timestamp: 7777 });
  assert.equal(sanitized.fps30ComplianceRate, 1);
  assert.equal(sanitized.frameJitterMs, 0);
  assert.equal(sanitized.longestStallMs, 0);
  assert.equal(sanitized.vramBudgetUtilization, 1);
  assert.equal(sanitized.reclaimedOversizedBytes, 1024);
  assert.equal(sanitized.bindGroupHitRate, 1);
  assert.equal(sanitized.zeroAllocDrains, 4);
  assert.deepEqual(sanitized.pipelineCacheStats, {
    fastLookupHits: 10,
    fastLookupMisses: 0,
    fastLookupHitRate: 1,
    negativeCacheHits: 3,
    negativeCacheSize: 2,
    negativeCacheEvictions: 0,
  });
});
