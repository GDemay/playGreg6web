import test from 'node:test';
import assert from 'node:assert/strict';

import bootstrapDefault, {
  GUEST_RELAUNCH_MESSAGE_TYPE,
  MAX_GUEST_RELAUNCHES,
  resetGuestRelaunchCount,
  shouldRelaunchGuest,
  DEFAULT_DUMP_ENDPOINT,
  DEFAULT_LOADER_URL,
  DEFAULT_MAX_VALIDATION_ERRORS,
  DEFAULT_SHADER_PACK_URL,
  DEFAULT_SPLASH_FADE_OUT_MS,
  DEFAULT_STREAM_SHADERS_THRESHOLD_BYTES,
  DEFAULT_STREAM_SHADERS_THRESHOLD_MB,
  MAX_STREAM_THRESHOLD_BYTES,
  MAX_STREAM_THRESHOLD_MB,
  PlayGta6AppHandle,
  bootstrapPlayGta6App,
  detectWasm64,
  extractBootAssetUrls,
  extractGpuAdapterFromLogLine,
  parseBootUrlOptions,
  parseSplashHoldQuery,
  parseWebGpuFramePresentedLabel,
  resolveDefaultGameBase,
  sanitizeInputScriptPreset,
  sanitizeSideModuleUrls,
  shouldLogWebGpuPresentedFrame,
} from '../web/app_bootstrap.js';
import { SESSION_MODE, SESSION_STATE } from '../web/runtime/session_controller.js';
import { SmokeDemoRunner } from '../web/demo/run_smoke_demo.js';
import { INPUT_BUTTONS } from '../web/input/input_bridge.js';
import { InputProfileManager } from '../web/input/input_profiles.js';

function createClock(startMs = 1000) {
  let current = startMs;
  return {
    now: () => current,
    advance(deltaMs) {
      current += deltaMs;
      return current;
    },
  };
}

function createFakeEventTarget(extra = {}) {
  const listeners = new Map();
  return {
    ...extra,
    addEventListener(type, fn, opts) {
      if (!listeners.has(type)) {
        listeners.set(type, new Set());
      }
      listeners.get(type).add({ fn, once: Boolean(opts && opts.once) });
    },
    removeEventListener(type, fn) {
      const set = listeners.get(type);
      if (!set) return;
      for (const entry of set) {
        if (entry.fn === fn) {
          set.delete(entry);
        }
      }
    },
    dispatch(type, event = {}) {
      const set = listeners.get(type);
      if (!set) return;
      for (const entry of [...set]) {
        if (entry.once) {
          set.delete(entry);
        }
        entry.fn({ type, ...event });
      }
    },
    listenerCount(type) {
      if (type) {
        return listeners.get(type)?.size ?? 0;
      }
      let total = 0;
      for (const set of listeners.values()) {
        total += set.size;
      }
      return total;
    },
  };
}

function createMemoryStorage(initial = {}) {
  const map = new Map(Object.entries(initial));
  return {
    getItem(key) {
      return map.has(key) ? map.get(key) : null;
    },
    setItem(key, value) {
      map.set(key, String(value));
    },
    removeItem(key) {
      map.delete(key);
    },
    clear() {
      map.clear();
    },
  };
}

function createFakeRaf(clock) {
  let nextId = 1;
  const scheduled = new Map();
  return {
    raf(cb) {
      const id = nextId++;
      scheduled.set(id, cb);
      return id;
    },
    cancelRaf(id) {
      scheduled.delete(id);
    },
    flushFrame(deltaMs = 16) {
      if (scheduled.size === 0) return 0;
      const batch = [...scheduled.entries()];
      scheduled.clear();
      const ts = clock.advance(deltaMs);
      for (const [, cb] of batch) {
        cb(ts);
      }
      return batch.length;
    },
    pendingCount() {
      return scheduled.size;
    },
  };
}

function createFakeTimers() {
  let nextId = 1;
  const active = new Map();
  return {
    setInterval(fn, ms) {
      const id = nextId++;
      active.set(id, { fn, ms });
      return id;
    },
    clearInterval(id) {
      active.delete(id);
    },
    tickAll() {
      for (const entry of [...active.values()]) {
        entry.fn();
      }
    },
    activeCount() {
      return active.size;
    },
  };
}

function createMockHud() {
  const statuses = [];
  const logs = [];
  const metricsUpdates = [];
  return {
    statuses,
    logs,
    metricsUpdates,
    setStatus(text, level = 'info') {
      statuses.push({ text, level });
    },
    appendLog(message, level = 'info') {
      logs.push({ message, level });
    },
    updateMetrics(metrics) {
      metricsUpdates.push({ ...metrics });
    },
  };
}

function createMockFetch(payload, { ok = true, status = 200, throwError = null } = {}) {
  const calls = [];
  const fetchImpl = async (url, init) => {
    calls.push({ url, init });
    if (throwError) {
      throw throwError;
    }
    return {
      ok,
      status,
      async json() {
        return typeof payload === 'function' ? payload() : payload;
      },
    };
  };
  fetchImpl.calls = calls;
  return fetchImpl;
}

function createMockWebGpuEnv() {
  let nextHandle = 1;
  let swapchainViewCounter = 0;
  const configureCalls = [];
  let unconfigureCount = 0;
  let deviceDestroyed = false;

  let resolveLost;
  const lost = new Promise((resolve) => {
    resolveLost = resolve;
  });

  const context = {
    configure(config) {
      configureCalls.push({ ...config });
    },
    unconfigure() {
      unconfigureCount += 1;
    },
    getCurrentTexture() {
      const id = `swapchain-view-${swapchainViewCounter++}`;
      const view = { kind: 'GPUTextureView', id };
      return {
        createView() {
          return view;
        },
      };
    },
  };

  const canvas = createFakeEventTarget({
    width: 640,
    height: 360,
    clientWidth: 640,
    clientHeight: 360,
    getContext(kind) {
      return kind === 'webgpu' ? context : null;
    },
  });

  const device = createFakeEventTarget({
    limits: { maxTextureDimension2D: 8192 },
    lost,
    triggerLost(info = { reason: 'unknown', message: 'simulated GPU device loss' }) {
      resolveLost(info);
    },
    destroy() {
      deviceDestroyed = true;
    },
    createBuffer(desc) {
      return {
        kind: 'GPUBuffer',
        handle: nextHandle++,
        size: desc.size,
        usage: desc.usage,
        destroyed: false,
        writes: [],
        destroy() {
          this.destroyed = true;
        },
      };
    },
    createTexture(desc) {
      const handle = nextHandle++;
      return {
        kind: 'GPUTexture',
        handle,
        desc: { ...desc, size: { ...desc.size } },
        destroyed: false,
        createView() {
          return { kind: 'GPUTextureView', textureHandle: handle, handle: nextHandle++ };
        },
        destroy() {
          this.destroyed = true;
        },
      };
    },
    createSampler(desc) {
      return { kind: 'GPUSampler', handle: nextHandle++, desc: { ...desc } };
    },
    createShaderModule(desc) {
      return { kind: 'GPUShaderModule', handle: nextHandle++, code: desc.code };
    },
    createRenderPipeline(desc) {
      return {
        kind: 'GPURenderPipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayout(index) {
          return { kind: 'GPUBindGroupLayout', pipelineKind: 'render', index };
        },
      };
    },
    createComputePipeline(desc) {
      return {
        kind: 'GPUComputePipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayout(index) {
          return { kind: 'GPUBindGroupLayout', pipelineKind: 'compute', index };
        },
      };
    },
    createBindGroup(desc) {
      return { kind: 'GPUBindGroup', handle: nextHandle++, desc };
    },
    createCommandEncoder() {
      const commandBuffer = { kind: 'GPUCommandBuffer', handle: nextHandle++ };
      return {
        beginRenderPass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            setVertexBuffer() {},
            setIndexBuffer() {},
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
          return commandBuffer;
        },
      };
    },
    queue: {
      writeBuffer(buffer, dstOffset, data) {
        buffer.writes.push({
          dstOffset,
          bytes: new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice(),
        });
      },
      writeTexture() {},
      submit() {},
    },
  });

  let requestAdapterCalls = 0;
  let requestDeviceCalls = 0;

  const adapter = {
    async requestDevice() {
      requestDeviceCalls += 1;
      return device;
    },
  };

  const gpu = {
    getPreferredCanvasFormat() {
      return 'bgra8unorm';
    },
    async requestAdapter() {
      requestAdapterCalls += 1;
      return adapter;
    },
  };

  return {
    gpu,
    adapter,
    device,
    context,
    canvas,
    configureCalls,
    get unconfigureCount() {
      return unconfigureCount;
    },
    get deviceDestroyed() {
      return deviceDestroyed;
    },
    get requestAdapterCalls() {
      return requestAdapterCalls;
    },
    get requestDeviceCalls() {
      return requestDeviceCalls;
    },
  };
}

function createMockBrowserHost({ clock, raf, timers, gpu, localStorage }) {
  const windowObj = createFakeEventTarget({
    crossOriginIsolated: true,
    innerWidth: 800,
    innerHeight: 600,
    devicePixelRatio: 1,
    location: { origin: 'https://playgta6.local' },
    localStorage: localStorage || createMemoryStorage(),
    requestAnimationFrame: raf.raf,
    cancelAnimationFrame: raf.cancelRaf,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
  });

  const documentObj = createFakeEventTarget({
    pointerLockElement: null,
    hidden: false,
    visibilityState: 'visible',
  });

  const navigatorObj = {
    gpu: gpu ?? null,
    getGamepads() {
      return [];
    },
    storage: {
      async estimate() {
        return { usage: 1024, quota: 1024 * 1024 };
      },
      async persisted() {
        return false;
      },
      async getDirectory() {
        return {
          async getDirectoryHandle() {
            const err = new Error('Not found');
            err.name = 'NotFoundError';
            throw err;
          },
        };
      },
    },
  };

  return { windowObj, documentObj, navigatorObj, clock };
}

test('exports and argument validation', () => {
  assert.equal(bootstrapDefault, bootstrapPlayGta6App);
  assert.equal(typeof DEFAULT_LOADER_URL, 'string');
  assert.ok(DEFAULT_LOADER_URL.endsWith('/loader.js'));

  assert.throws(() => bootstrapPlayGta6App(null), TypeError);
  assert.throws(() => bootstrapPlayGta6App({ webgpuAdapterFactory: 'bad' }), TypeError);
  assert.throws(() => bootstrapPlayGta6App({ createWasmWorker: 123 }), TypeError);
});

test('end-to-end smoke demo fallback when /api/status reports hasWasmBinary: false', async () => {
  const clock = createClock(2000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();

  // Persist a custom active input profile ('lefty-kb') in mocked localStorage beforehand.
  const storage = createMemoryStorage();
  const seedProfiles = new InputProfileManager({ storage });
  seedProfiles.setActiveProfile('lefty-kb');
  seedProfiles.save();

  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
    localStorage: storage,
  });

  const fetchImpl = createMockFetch({
    ok: true,
    coopCoep: true,
    dataDirConfigured: false,
    hasManifest: false,
    hasBootset: false,
    hasWasmBinary: false,
    shaderPackCount: 0,
  });

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl,
    canvas: webgpu.canvas,
    hud,
    now: clock.now,
    raf: raf.raf,
    cancelRaf: raf.cancelRaf,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
  });

  assert.ok(app instanceof PlayGta6AppHandle);
  assert.equal(app.destroyed, false);
  assert.equal(app.getSmokeRunner(), null);
  assert.equal(app.getWasmWorker(), null);
  assert.equal(app.crashReporter.isGlobalHandlersAttached, true);

  // Saved input profile was loaded and applied to InputBridge.
  assert.equal(app.inputProfiles.getActiveProfile().id, 'lefty-kb');
  windowObj.dispatch('keydown', { code: 'KeyI' });
  const inputSnap = app.inputBridge.getSnapshot();
  assert.ok(inputSnap);
  assert.notEqual(inputSnap.buttons & INPUT_BUTTONS.MOVE_FORWARD, 0);
  windowObj.dispatch('keyup', { code: 'KeyI' });

  // Start session -> auto-starts SmokeDemoRunner because hasWasmBinary is false.
  const startResult = await app.start();
  assert.equal(startResult.ok, true);
  assert.equal(startResult.mode, SESSION_MODE.SMOKE);
  assert.equal(startResult.state, SESSION_STATE.RUNNING_SMOKE);
  assert.equal(webgpu.requestAdapterCalls, 1);
  assert.equal(webgpu.requestDeviceCalls, 1);

  const smokeRunner = app.getSmokeRunner();
  assert.ok(smokeRunner instanceof SmokeDemoRunner);
  assert.equal(smokeRunner.initialized, true);
  assert.equal(smokeRunner.running, true);
  assert.equal(smokeRunner.inputSab, app.inputBridge.sab);

  // Step frames via RAF loop and verify PerfMonitor + HUD receive live updates.
  const metricsBefore = hud.metricsUpdates.length;
  assert.ok(raf.flushFrame(16) >= 1);
  assert.ok(raf.flushFrame(16) >= 1);
  assert.ok(raf.flushFrame(16) >= 1);

  assert.equal(smokeRunner.frameIndex, 3);
  const perfSnap = app.perfMonitor.getSnapshot();
  assert.ok(perfSnap.frames.totalFrames >= 3);
  assert.ok(perfSnap.gpu.totalGpuPackets > 0);
  assert.ok(perfSnap.gpu.totalDrawCalls > 0);
  assert.ok(hud.metricsUpdates.length > metricsBefore);

  // Periodic PerfMonitor HUD timer tick also works.
  const countBeforeTick = hud.metricsUpdates.length;
  timers.tickAll();
  assert.equal(hud.metricsUpdates.length, countBeforeTick + 1);

  // Programmatic resize() reconfigures the smoke demo swapchain.
  const resized = app.resize(1280, 720, 2);
  assert.equal(resized, true);
  assert.equal(webgpu.canvas.width, 2560);
  assert.equal(webgpu.canvas.height, 1440);

  // Window resize event also routes through handle.resize().
  webgpu.canvas.clientWidth = 1024;
  webgpu.canvas.clientHeight = 576;
  windowObj.devicePixelRatio = 1;
  windowObj.dispatch('resize');
  assert.equal(webgpu.canvas.width, 1024);
  assert.equal(webgpu.canvas.height, 576);

  // Global error, unhandledrejection, and WebGPU validation/device-lost errors record into CrashReporter.
  windowObj.dispatch('error', {
    error: new Error('Boom at C:\\Users\\alice\\secret\\game.js'),
    filename: 'file:///C:/Users/alice/app.js',
    lineno: 42,
    colno: 7,
  });
  windowObj.dispatch('unhandledrejection', {
    reason: new Error('Rejected promise from /home/alice/data'),
  });
  webgpu.device.dispatch('uncapturederror', {
    error: { name: 'GPUValidationError', message: 'Invalid bind group layout' },
  });
  webgpu.device.triggerLost({ reason: 'unknown', message: 'Adapter reset' });
  await Promise.resolve();

  const errors = app.crashReporter.getErrors();
  assert.equal(errors.length, 4);
  assert.ok(!errors[0].message.includes('alice'));
  assert.equal(errors[1].category, 'unhandled_rejection');
  assert.ok(!errors[1].message.includes('/home/alice'));
  assert.equal(errors[2].category, 'webgpu_validation');
  assert.equal(errors[3].category, 'webgpu_device_lost');

  // Destroy cleanly tears down every subsystem and leaks zero listeners or timers.
  await app.destroy();
  assert.equal(app.destroyed, true);
  assert.equal(app.getSmokeRunner(), null);
  assert.equal(smokeRunner.destroyed, true);
  assert.equal(app.session.getState(), SESSION_STATE.STOPPED);
  assert.equal(app.inputBridge.disposed, true);
  assert.equal(app.perfMonitor.isHudAttached, false);
  assert.equal(app.crashReporter.isGlobalHandlersAttached, false);
  assert.equal(app.audioEngine.state, 'closed');
  assert.equal(timers.activeCount(), 0);
  assert.equal(windowObj.listenerCount(), 0);
  assert.equal(documentObj.listenerCount(), 0);
  assert.equal(webgpu.canvas.listenerCount(), 0);
  assert.equal(webgpu.device.listenerCount(), 0);

  // Destroy is idempotent and start() after destroy returns cleanly.
  await app.destroy();
  const afterDestroy = await app.start();
  assert.equal(afterDestroy.ok, false);
  assert.equal(afterDestroy.destroyed, true);
});

test('Wasm64 runtime path when /api/status reports hasWasmBinary: true invokes createWasmWorker', async () => {
  const clock = createClock(5000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();

  const offscreenCanvas = { kind: 'MockOffscreenCanvas' };
  let transferCalls = 0;
  webgpu.canvas.transferControlToOffscreen = () => {
    transferCalls += 1;
    return offscreenCanvas;
  };

  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });

  const workerCalls = [];
  const postedMessages = [];
  let terminatedCount = 0;

  const mockWorker = createFakeEventTarget({
    postMessage(payload, transfer) {
      postedMessages.push({ payload, transfer });
    },
    terminate() {
      terminatedCount += 1;
    },
  });

  const fetchImpl = createMockFetch({
    ok: true,
    coopCoep: true,
    dataDirConfigured: true,
    hasManifest: true,
    hasBootset: true,
    hasWasmBinary: true,
    shaderPackCount: 3,
  });

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl,
    canvas: webgpu.canvas,
    hud,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker(args) {
      workerCalls.push(args);
      return mockWorker;
    },
  });

  const startResult = await app.start();
  assert.equal(startResult.ok, true);
  assert.equal(startResult.mode, SESSION_MODE.WASM);
  assert.equal(startResult.state, SESSION_STATE.RUNNING_WASM);
  assert.equal(app.getSmokeRunner(), null);
  assert.equal(app.getWasmWorker(), mockWorker);
  assert.equal(webgpu.requestAdapterCalls, 0);

  assert.equal(workerCalls.length, 1);
  assert.equal(workerCalls[0].url, DEFAULT_LOADER_URL);
  assert.equal(workerCalls[0].inputSab, app.inputBridge.sab);
  assert.equal(workerCalls[0].audioSab, app.audioEngine.sab);
  assert.equal(workerCalls[0].dataBase, 'https://playgta6.local/data/');
  assert.equal(transferCalls, 1);

  assert.equal(postedMessages.length, 1);
  assert.equal(postedMessages[0].payload.canvas, offscreenCanvas);
  assert.equal(postedMessages[0].payload.width, 640);
  assert.equal(postedMessages[0].payload.height, 360);
  assert.equal(postedMessages[0].payload.inputSab, app.inputBridge.sab);
  assert.equal(postedMessages[0].payload.audioSab, app.audioEngine.sab);
  assert.deepEqual(postedMessages[0].transfer, [offscreenCanvas]);

  // resize() forwards pixel dimensions to the Wasm64 worker.
  const resized = app.resize(1920, 1080, 1.5);
  assert.equal(resized, true);
  assert.equal(postedMessages.length, 2);
  assert.deepEqual(postedMessages[1].payload, {
    resize: true,
    width: 2880,
    height: 1620,
  });

  // Worker error events are captured by CrashReporter and reported to HUD.
  mockWorker.dispatch('error', { message: 'wasm trap in worker' });
  const errs = app.crashReporter.getErrors();
  assert.equal(errs.length, 1);
  assert.equal(errs[0].category, 'worker_error');
  assert.match(errs[0].message, /wasm trap in worker/);

  // Stopping terminates worker and detaches its listeners.
  const stopResult = await app.stop();
  assert.equal(stopResult.ok, true);
  assert.equal(terminatedCount, 1);
  assert.equal(app.getWasmWorker(), null);
  assert.equal(mockWorker.listenerCount(), 0);

  await app.destroy();
  assert.equal(windowObj.listenerCount(), 0);
  assert.equal(documentObj.listenerCount(), 0);
  assert.equal(webgpu.canvas.listenerCount(), 0);
});

test('preferSmoke: true selects SmokeDemoRunner via injectable webgpuAdapterFactory even when hasWasmBinary: true', async () => {
  const clock = createClock(1000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();

  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: null, // Verified that injectable webgpuAdapterFactory works even without navigator.gpu
  });

  let wasmCalled = 0;
  let factoryCalled = 0;

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch({ ok: true, hasWasmBinary: true }),
    canvas: webgpu.canvas,
    hud,
    now: clock.now,
    raf: raf.raf,
    cancelRaf: raf.cancelRaf,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    webgpuAdapterFactory: async () => {
      factoryCalled += 1;
      return webgpu.adapter;
    },
    createWasmWorker: () => {
      wasmCalled += 1;
      return { postMessage() {}, terminate() {} };
    },
  });

  const res = await app.start({ preferSmoke: true });
  assert.equal(res.ok, true);
  assert.equal(res.mode, SESSION_MODE.SMOKE);
  assert.equal(wasmCalled, 0);
  assert.equal(factoryCalled, 1);
  assert.ok(app.getSmokeRunner() instanceof SmokeDemoRunner);

  // Stop and restart smoke demo cleanly.
  const firstRunner = app.getSmokeRunner();
  await app.stop();
  assert.equal(firstRunner.destroyed, true);
  assert.equal(app.getSmokeRunner(), null);

  const res2 = await app.start({ preferSmoke: true });
  assert.equal(res2.ok, true);
  assert.equal(res2.mode, SESSION_MODE.SMOKE);
  assert.notEqual(app.getSmokeRunner(), firstRunner);

  await app.destroy();
  assert.equal(windowObj.listenerCount(), 0);
  assert.equal(documentObj.listenerCount(), 0);
  assert.equal(webgpu.canvas.listenerCount(), 0);
});

test('graceful handling of corrupt persisted input profiles and session probe/launcher failures', async () => {
  const clock = createClock(1000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();

  // Corrupt JSON in localStorage should be recorded in CrashReporter, not crash bootstrap.
  const corruptStorage = createMemoryStorage({
    'playgta6.inputProfiles.v1': '{not valid json',
  });

  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: {
      async requestAdapter() {
        return null; // Simulate missing GPU adapter
      },
    },
    localStorage: corruptStorage,
  });

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch({ ok: true, hasWasmBinary: false }),
    canvas: webgpu.canvas,
    hud,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
  });

  // Default profile active despite corrupt storage; load error captured.
  assert.equal(app.inputProfiles.getActiveProfile().id, 'default');
  assert.equal(app.crashReporter.errorCount, 1);
  assert.equal(app.crashReporter.getErrors()[0].category, 'io_error');

  // Starting smoke demo fails cleanly when GPU adapter is null and records the error.
  const failed = await app.start();
  assert.equal(failed.ok, false);
  assert.equal(failed.state, SESSION_STATE.ERROR);
  assert.equal(app.getSmokeRunner(), null);
  assert.equal(app.crashReporter.errorCount, 2);
  assert.match(app.crashReporter.getErrors()[1].message, /GPUAdapter/i);

  await app.destroy();
  assert.equal(windowObj.listenerCount(), 0);
  assert.equal(documentObj.listenerCount(), 0);
  assert.equal(webgpu.canvas.listenerCount(), 0);
});

test('W45 wires BootTracer, SplashOverlayController, and BootOrchestrator through /api/status discovery and FIRST_PRESENT dismissal', async () => {
  const { createRequire } = await import('node:module');
  const require = createRequire(import.meta.url);
  const { SplashOverlayController } = require('../web/ui/splash_overlay.js');
  const { BootOrchestrator } = require('../web/runtime/boot_orchestrator.js');

  const clock = createClock(1000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();

  function makeMockNode(tag = 'div') {
    const attrs = new Map();
    const styles = new Map();
    const listeners = new Map();
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
      addEventListener(type, fn) {
        if (!listeners.has(type)) listeners.set(type, new Set());
        listeners.get(type).add(fn);
      },
      removeEventListener(type, fn) {
        listeners.get(type)?.delete(fn);
      },
      querySelector() {
        return null;
      },
    };
    return node;
  }

  const splashContainer = makeMockNode('div');
  const head = makeMockNode('head');
  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
  });
  documentObj.head = head;
  documentObj.createElement = (tag) => makeMockNode(tag);
  documentObj.querySelector = () => null;

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch({
      ok: true,
      hasWasmBinary: false,
      splashImage: '/game/xbs_splash.png',
    }),
    canvas: webgpu.canvas,
    splashContainer,
    SplashOverlayCtor: SplashOverlayController,
    BootOrchestratorCtor: BootOrchestrator,
    splashOverlayOptions: { fadeMs: 0 },
    hud,
    webgpuAdapterFactory: async () => webgpu.adapter,
    now: clock.now,
    raf: raf.raf,
    cancelRaf: raf.cancelRaf,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
  });

  assert.ok(app.bootTracer, 'bootTracer must be exposed');
  assert.ok(app.splashOverlay, 'splashOverlay must be mounted when splashContainer is provided');
  assert.ok(app.bootOrchestrator, 'bootOrchestrator must be wired when splashOverlay is active');
  assert.equal(app.splashOverlay.getState().visible, true);

  const started = await app.start();
  assert.equal(started.ok, true);
  assert.equal(started.mode, SESSION_MODE.SMOKE);

  // Splash image discovered from /api/status must be applied and dismissed on smoke FIRST_PRESENT.
  const splashState = app.splashOverlay.getState();
  assert.equal(splashState.splashImageUrl, '/game/xbs_splash.png');
  assert.equal(splashState.dismissed, true);
  assert.equal(splashState.visible, false);
  assert.equal(app.bootTracer.getSummary().readyForSplash, true);

  await app.destroy();
  assert.equal(app.splashOverlay.getState().destroyed, true);
  assert.equal(app.bootOrchestrator.getState().destroyed, true);
});

test('W49 startWasmRuntime forwards bootBase, shaderPackUrl, reflectionManifestUrl, and pgimUrl from /api/status', async () => {
  const { createRequire } = await import('node:module');
  const require = createRequire(import.meta.url);
  const { BootOrchestrator } = require('../web/runtime/boot_orchestrator.js');
  const { createWorkerBootInitPayloads } = require('../web/loader.js');

  const clock = createClock(7000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();

  const offscreenCanvas = { kind: 'MockOffscreenCanvas' };
  webgpu.canvas.transferControlToOffscreen = () => offscreenCanvas;

  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });

  const workerCalls = [];
  const postedMessages = [];
  const mockWorker = createFakeEventTarget({
    postMessage(payload, transfer) {
      postedMessages.push({ payload, transfer });
    },
    terminate() {},
  });

  const statusPayload = {
    ok: true,
    coopCoep: true,
    dataDirConfigured: true,
    hasManifest: true,
    hasBootset: true,
    hasWasmBinary: true,
    shaderPackCount: 1,
    shaderPackUrl: '/shaders/fallback.wgsl.pack',
    reflectionManifestUrl: '/shaders/reflection-manifest.json',
    pgimUrl: '/boot/guest.pgim',
  };

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch(statusPayload),
    canvas: webgpu.canvas,
    hud,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker(args) {
      workerCalls.push(args);
      return mockWorker;
    },
  });

  const started = await app.start();
  assert.equal(started.ok, true);
  assert.equal(started.mode, SESSION_MODE.WASM);
  assert.equal(workerCalls.length, 1);
  assert.equal(postedMessages.length, 1);

  assert.equal(workerCalls[0].bootBase, 'https://playgta6.local/boot/');
  assert.equal(workerCalls[0].shaderPackUrl, '/shaders/fallback.wgsl.pack');
  assert.equal(workerCalls[0].reflectionManifestUrl, '/shaders/reflection-manifest.json');
  assert.equal(workerCalls[0].pgimUrl, '/boot/guest.pgim');

  const initPayload = postedMessages[0].payload;
  assert.equal(initPayload.bootBase, 'https://playgta6.local/boot/');
  assert.equal(initPayload.shaderPackUrl, '/shaders/fallback.wgsl.pack');
  assert.equal(initPayload.reflectionManifestUrl, '/shaders/reflection-manifest.json');
  assert.equal(initPayload.pgimUrl, '/boot/guest.pgim');

  // Verify W47 loader init payload forwarding consumes the exact fields posted by app_bootstrap.
  const workerInits = createWorkerBootInitPayloads(initPayload);
  assert.equal(workerInits.ioInit.bootBase, 'https://playgta6.local/boot/');
  assert.equal(workerInits.ioInit.pgimUrl, '/boot/guest.pgim');
  assert.equal(workerInits.gpuInit.shaderPackUrl, '/shaders/fallback.wgsl.pack');
  assert.equal(workerInits.gpuInit.reflectionManifestUrl, '/shaders/reflection-manifest.json');

  await app.destroy();

  // Also verify BootOrchestrator discovery path and default '/boot/' fallback when location.origin is absent.
  const orchWorkerCalls = [];
  const orchPostedMessages = [];
  const orchWorker = createFakeEventTarget({
    postMessage(payload, transfer) {
      orchPostedMessages.push({ payload, transfer });
    },
    terminate() {},
  });
  const hostNoOrigin = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });
  hostNoOrigin.windowObj.location = { origin: 'null' };

  const orchApp = bootstrapPlayGta6App({
    windowObj: hostNoOrigin.windowObj,
    documentObj: hostNoOrigin.documentObj,
    navigatorObj: hostNoOrigin.navigatorObj,
    fetchImpl: createMockFetch(statusPayload),
    canvas: webgpu.canvas,
    hud,
    enableBootOrchestrator: true,
    BootOrchestratorCtor: BootOrchestrator,
    pgimPreloader: {
      async preloadBootBundle() {
        return { pgim: new Uint8Array(4), shaderPackBytes: null, reflectionManifest: null, bytesLoaded: 4 };
      },
    },
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker(args) {
      orchWorkerCalls.push(args);
      return orchWorker;
    },
  });

  const orchStarted = await orchApp.start();
  assert.equal(orchStarted.ok, true);
  assert.equal(orchStarted.mode, SESSION_MODE.WASM);
  assert.equal(orchWorkerCalls.length, 1);
  assert.equal(orchPostedMessages.length, 1);

  assert.equal(orchWorkerCalls[0].bootBase, '/boot/');
  assert.equal(orchWorkerCalls[0].shaderPackUrl, '/shaders/fallback.wgsl.pack');
  assert.equal(orchWorkerCalls[0].reflectionManifestUrl, '/shaders/reflection-manifest.json');
  assert.equal(orchWorkerCalls[0].pgimUrl, '/boot/guest.pgim');

  assert.equal(orchPostedMessages[0].payload.bootBase, '/boot/');
  assert.equal(orchPostedMessages[0].payload.shaderPackUrl, '/shaders/fallback.wgsl.pack');
  assert.equal(orchPostedMessages[0].payload.reflectionManifestUrl, '/shaders/reflection-manifest.json');
  assert.equal(orchPostedMessages[0].payload.pgimUrl, '/boot/guest.pgim');

  await orchApp.destroy();
});

test('W54 resolveDefaultGameBase and extractBootAssetUrls discover gameBase and programManifestUrl', () => {
  assert.equal(
    resolveDefaultGameBase({ gameBase: 'https://cdn.local/custom-game/' }, { location: { origin: 'https://playgta6.local' } }),
    'https://cdn.local/custom-game/'
  );
  assert.equal(
    resolveDefaultGameBase({}, { location: { origin: 'https://playgta6.local' } }),
    'https://playgta6.local/game/'
  );
  assert.equal(resolveDefaultGameBase({}, { location: { origin: 'null' } }), '/game/');
  assert.equal(resolveDefaultGameBase({}, null), '/game/');

  assert.deepEqual(
    extractBootAssetUrls({
      bootBase: '/boot/',
      gameBase: '/game-alt/',
      shaderPackUrl: '/shaders/pack.wgsl',
      reflectionManifestUrl: '/shaders/reflection.json',
      pgimUrl: '/boot/guest.pgim',
      programManifestUrl: '/boot/program-manifest.json',
    }),
    {
      bootBase: '/boot/',
      gameBase: '/game-alt/',
      shaderPackUrl: '/shaders/pack.wgsl',
      reflectionManifestUrl: '/shaders/reflection.json',
      pgimUrl: '/boot/guest.pgim',
      programManifestUrl: '/boot/program-manifest.json',
    }
  );

  assert.deepEqual(
    extractBootAssetUrls({
      programManifest: { url: '/boot/alt-program-manifest.json' },
    }),
    {
      programManifestUrl: '/boot/alt-program-manifest.json',
    }
  );
});

test('W54 startWasmRuntime forwards gameBase and programManifestUrl into createWasmWorker and worker.postMessage', async () => {
  const clock = createClock(8000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();

  const offscreenCanvas = { kind: 'MockOffscreenCanvas' };
  webgpu.canvas.transferControlToOffscreen = () => offscreenCanvas;

  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });

  const workerCalls = [];
  const postedMessages = [];
  const mockWorker = createFakeEventTarget({
    postMessage(payload, transfer) {
      postedMessages.push({ payload, transfer });
    },
    terminate() {},
  });

  const statusPayload = {
    ok: true,
    coopCoep: true,
    dataDirConfigured: true,
    hasManifest: true,
    hasBootset: true,
    hasWasmBinary: true,
    shaderPackCount: 2,
    shaderPackUrl: '/shaders/fallback.wgsl.pack',
    reflectionManifestUrl: '/shaders/reflection-manifest.json',
    pgimUrl: '/boot/guest.pgim',
    programManifestUrl: '/boot/program-manifest.json',
  };

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch(statusPayload),
    canvas: webgpu.canvas,
    hud,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker(args) {
      workerCalls.push(args);
      return mockWorker;
    },
  });

  const started = await app.start();
  assert.equal(started.ok, true);
  assert.equal(started.mode, SESSION_MODE.WASM);
  assert.equal(workerCalls.length, 1);
  assert.equal(postedMessages.length, 1);

  assert.equal(workerCalls[0].bootBase, 'https://playgta6.local/boot/');
  assert.equal(workerCalls[0].gameBase, 'https://playgta6.local/game/');
  assert.equal(workerCalls[0].shaderPackUrl, '/shaders/fallback.wgsl.pack');
  assert.equal(workerCalls[0].reflectionManifestUrl, '/shaders/reflection-manifest.json');
  assert.equal(workerCalls[0].pgimUrl, '/boot/guest.pgim');
  assert.equal(workerCalls[0].programManifestUrl, '/boot/program-manifest.json');

  const initPayload = postedMessages[0].payload;
  assert.equal(initPayload.bootBase, 'https://playgta6.local/boot/');
  assert.equal(initPayload.gameBase, 'https://playgta6.local/game/');
  assert.equal(initPayload.shaderPackUrl, '/shaders/fallback.wgsl.pack');
  assert.equal(initPayload.reflectionManifestUrl, '/shaders/reflection-manifest.json');
  assert.equal(initPayload.pgimUrl, '/boot/guest.pgim');
  assert.equal(initPayload.programManifestUrl, '/boot/program-manifest.json');

  await app.destroy();

  // Verify discovered gameBase + programManifest alias and '/game/' fallback when origin is 'null'.
  const altWorkerCalls = [];
  const altPostedMessages = [];
  const altWorker = createFakeEventTarget({
    postMessage(payload, transfer) {
      altPostedMessages.push({ payload, transfer });
    },
    terminate() {},
  });
  const hostNoOrigin = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });
  hostNoOrigin.windowObj.location = { origin: 'null' };

  const altApp = bootstrapPlayGta6App({
    windowObj: hostNoOrigin.windowObj,
    documentObj: hostNoOrigin.documentObj,
    navigatorObj: hostNoOrigin.navigatorObj,
    fetchImpl: createMockFetch({
      ok: true,
      hasWasmBinary: true,
      gameBase: '/custom-game-root/',
      programManifest: '/boot/alias-program-manifest.json',
      pgimUrl: '/boot/guest.pgim',
    }),
    canvas: webgpu.canvas,
    hud,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker(args) {
      altWorkerCalls.push(args);
      return altWorker;
    },
  });

  const altStarted = await altApp.start();
  assert.equal(altStarted.ok, true);
  assert.equal(altWorkerCalls[0].gameBase, '/custom-game-root/');
  assert.equal(altWorkerCalls[0].programManifestUrl, '/boot/alias-program-manifest.json');
  assert.equal(altPostedMessages[0].payload.gameBase, '/custom-game-root/');
  assert.equal(altPostedMessages[0].payload.programManifestUrl, '/boot/alias-program-manifest.json');

  await altApp.destroy();

  // Verify '/game/' default fallback when neither options.gameBase, status.gameBase, nor valid origin is set.
  const fallbackWorkerCalls = [];
  const fallbackPostedMessages = [];
  const fallbackWorker = createFakeEventTarget({
    postMessage(payload, transfer) {
      fallbackPostedMessages.push({ payload, transfer });
    },
    terminate() {},
  });

  const fallbackApp = bootstrapPlayGta6App({
    windowObj: hostNoOrigin.windowObj,
    documentObj: hostNoOrigin.documentObj,
    navigatorObj: hostNoOrigin.navigatorObj,
    fetchImpl: createMockFetch({ ok: true, hasWasmBinary: true }),
    canvas: webgpu.canvas,
    hud,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker(args) {
      fallbackWorkerCalls.push(args);
      return fallbackWorker;
    },
  });

  const fallbackStarted = await fallbackApp.start();
  assert.equal(fallbackStarted.ok, true);
  assert.equal(fallbackWorkerCalls[0].gameBase, '/game/');
  assert.equal(fallbackPostedMessages[0].payload.gameBase, '/game/');
  assert.equal(fallbackWorkerCalls[0].programManifestUrl, undefined);
  assert.equal(fallbackPostedMessages[0].payload.programManifestUrl, undefined);

  await fallbackApp.destroy();
});

test('W54 connects BootTracer updates and worker gpu-telemetry events to injected gpuBootPanel and cleans up on destroy', async () => {
  const clock = createClock(9000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();

  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });

  const mockWorker = createFakeEventTarget({
    postMessage() {},
    terminate() {},
  });

  const panelUpdates = [];
  let panelDestroyCalls = 0;
  const mockGpuBootPanel = {
    update(snapshot) {
      panelUpdates.push(snapshot);
    },
    destroy() {
      panelDestroyCalls += 1;
    },
  };

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch({ ok: true, hasWasmBinary: true }),
    canvas: webgpu.canvas,
    hud,
    gpuBootPanel: mockGpuBootPanel,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker() {
      return mockWorker;
    },
  });

  assert.equal(app.gpuBootPanel, mockGpuBootPanel);
  assert.equal(panelUpdates.length, 0);

  const started = await app.start();
  assert.equal(started.ok, true);
  assert.equal(started.mode, SESSION_MODE.WASM);
  assert.equal(mockWorker.listenerCount('message'), 1);

  // BootTracer milestone updates flow into gpuBootPanel.update({ bootSummary }).
  app.bootTracer.markMilestone('PGIM_LOAD');
  assert.equal(panelUpdates.length, 1);
  assert.equal(panelUpdates[0].bootSummary.stage, 'PGIM_LOAD');
  assert.equal(panelUpdates[0].bootSummary.milestonesReached, 1);

  // BootTracer ingestLogLine updates flow once (without duplicate updates).
  app.bootTracer.ingestLogLine('[boot] D3D12_CREATE_DEVICE');
  assert.equal(panelUpdates.length, 2);
  assert.equal(panelUpdates[1].bootSummary.stage, 'D3D12_CREATE_DEVICE');

  // Unrelated log lines do not emit spurious panel updates.
  app.bootTracer.ingestLogLine('ordinary non-milestone message');
  assert.equal(panelUpdates.length, 2);

  // Worker gpu-telemetry messages forward bootSummary, frameStats, fallbackStats, missingShaders.
  const frameStats = { framesPresented: 4, droppedFrames: 1 };
  const fallbackStats = { totalLookups: 10, packHits: 8, fallbackHits: 2, builtinHits: 0 };
  const missingShaders = [{ hash: '0x00000000deadbeef', stage: 'fragment', hits: 2 }];

  mockWorker.dispatch('message', {
    data: {
      type: 'gpu-telemetry',
      frameStats,
      fallbackStats,
      missingShaders,
    },
  });

  assert.equal(panelUpdates.length, 3);
  assert.equal(panelUpdates[2].bootSummary.stage, 'D3D12_CREATE_DEVICE');
  assert.deepEqual(panelUpdates[2].frameStats, frameStats);
  assert.deepEqual(panelUpdates[2].fallbackStats, fallbackStats);
  assert.deepEqual(panelUpdates[2].missingShaders, missingShaders);

  // Non-telemetry worker messages are ignored.
  mockWorker.dispatch('message', { data: { type: 'other-event', ok: true } });
  assert.equal(panelUpdates.length, 3);

  // destroy() removes worker and tracer hooks, and does NOT destroy an externally-owned gpuBootPanel.
  await app.destroy();
  assert.equal(mockWorker.listenerCount('message'), 0);
  assert.equal(mockWorker.listenerCount(), 0);
  assert.equal(panelDestroyCalls, 0);

  // Subsequent worker telemetry or BootTracer calls after destroy() must not invoke gpuBootPanel.update().
  mockWorker.dispatch('message', {
    data: { type: 'gpu-telemetry', frameStats: { framesPresented: 99, droppedFrames: 0 } },
  });
  app.bootTracer.markMilestone('FIRST_PRESENT');
  assert.equal(panelUpdates.length, 3);
});

test('W54 constructs owned GpuBootPanelController from gpuBootPanelRoot and globalThis auto-discovery and destroys on teardown', async () => {
  const { createRequire } = await import('node:module');
  const require = createRequire(import.meta.url);
  const gpuBootPanelApi = require('../web/ui/gpu_boot_panel.js');
  const { GpuBootPanelController } = gpuBootPanelApi;

  const clock = createClock(10000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();

  function makeMockDomNode(tag = 'div') {
    const attrs = new Map();
    const listeners = new Map();
    const node = {
      tagName: String(tag).toUpperCase(),
      className: '',
      hidden: false,
      textContent: '',
      parentNode: null,
      children: [],
      style: {},
      setAttribute(k, v) {
        attrs.set(k, String(v));
      },
      getAttribute(k) {
        return attrs.has(k) ? attrs.get(k) : null;
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
      addEventListener(type, fn) {
        if (!listeners.has(type)) listeners.set(type, new Set());
        listeners.get(type).add(fn);
      },
      removeEventListener(type, fn) {
        listeners.get(type)?.delete(fn);
      },
      querySelector() {
        return null;
      },
    };
    return node;
  }

  const panelRoot = makeMockDomNode('div');
  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });
  documentObj.head = makeMockDomNode('head');
  documentObj.createElement = (tag) => makeMockDomNode(tag);
  documentObj.querySelector = () => null;

  const mockWorker = createFakeEventTarget({
    postMessage() {},
    terminate() {},
  });

  // Auto-discover via globalThis.PlayGta6GpuBootPanel.GpuBootPanelController (no GpuBootPanelCtor passed).
  const prevGlobal = globalThis.PlayGta6GpuBootPanel;
  globalThis.PlayGta6GpuBootPanel = gpuBootPanelApi;
  let app = null;
  try {
    app = bootstrapPlayGta6App({
      windowObj,
      documentObj,
      navigatorObj,
      fetchImpl: createMockFetch({ ok: true, hasWasmBinary: true }),
      canvas: webgpu.canvas,
      hud,
      gpuBootPanelRoot: panelRoot,
      now: clock.now,
      setInterval: timers.setInterval,
      clearInterval: timers.clearInterval,
      createWasmWorker() {
        return mockWorker;
      },
    });

    assert.ok(app.gpuBootPanel instanceof GpuBootPanelController);
    assert.equal(panelRoot.children.length, 1);

    const started = await app.start();
    assert.equal(started.ok, true);

    app.bootTracer.markMilestone('FIRST_PRESENT');
    mockWorker.dispatch('message', {
      data: {
        type: 'gpu-telemetry',
        frameStats: { framesPresented: 7, droppedFrames: 0 },
        fallbackStats: { totalLookups: 20, packHits: 18, fallbackHits: 2, builtinHits: 0 },
        missingShaders: [{ hash: '0x1234', stage: 'vertex', hits: 2 }],
      },
    });

    const state = app.gpuBootPanel.getState();
    assert.equal(state.boot.stage, 'FIRST_PRESENT');
    assert.equal(state.boot.readyForSplash, true);
    assert.equal(state.frameStats.framesPresented, 7);
    assert.equal(state.fallbackStats.packHits, 18);
    assert.equal(state.totalMissingShaders, 1);
    assert.equal(state.missingShaders[0].hash, '0x0000000000001234');

    await app.destroy();
    assert.equal(app.gpuBootPanel.getState().destroyed, true);
    assert.equal(panelRoot.children.length, 0);
    assert.equal(mockWorker.listenerCount(), 0);
  } finally {
    if (app && !app.destroyed) {
      await app.destroy();
    }
    if (prevGlobal === undefined) {
      delete globalThis.PlayGta6GpuBootPanel;
    } else {
      globalThis.PlayGta6GpuBootPanel = prevGlobal;
    }
  }
});

test('W58 detectWasm64 supports spec-standard address: i64, legacy index: i64, and rejects non-Wasm64 engines', () => {
  // Simulates WebIDL ToNumber coercion on i32 memory limits, which throws TypeError on BigInt.
  function assertI32Limit(value) {
    return +value;
  }

  // 1. Modern Chrome/V8/Firefox/Node engine accepting only `{ address: 'i64' }` with BigInt limits.
  const modernCalls = [];
  const modernWasmApi = {
    Memory: class MockModernMemory {
      constructor(desc) {
        modernCalls.push(desc);
        if (desc.address === 'i64') {
          if (typeof desc.initial !== 'bigint' || typeof desc.maximum !== 'bigint') {
            throw new TypeError('Expected BigInt limits for i64 memory');
          }
          return;
        }
        assertI32Limit(desc.initial);
      }
    },
  };
  assert.equal(detectWasm64(modernWasmApi), true);
  assert.equal(modernCalls.length, 1);
  assert.equal(modernCalls[0].address, 'i64');
  assert.equal(modernCalls[0].initial, 1n);
  assert.equal(modernCalls[0].maximum, 2n);

  // 2. Older memory64 proposal engine accepting only legacy `{ index: 'i64' }`.
  const legacyCalls = [];
  const legacyWasmApi = {
    Memory: class MockLegacyMemory {
      constructor(desc) {
        legacyCalls.push(desc);
        if (desc.index === 'i64') {
          if (typeof desc.initial !== 'bigint' || typeof desc.maximum !== 'bigint') {
            throw new TypeError('Expected BigInt limits for i64 memory');
          }
          return;
        }
        // Unknown `address` key is ignored -> treated as i32 -> BigInt throws TypeError.
        assertI32Limit(desc.initial);
      }
    },
  };
  assert.equal(detectWasm64(legacyWasmApi), true);
  assert.equal(legacyCalls.length, 2);
  assert.equal(legacyCalls[0].address, 'i64');
  assert.equal(legacyCalls[1].index, 'i64');

  // 3. Non-Wasm64 engine that ignores both keys and throws on BigInt initial/maximum.
  const wasm32Calls = [];
  const wasm32OnlyApi = {
    Memory: class MockWasm32Memory {
      constructor(desc) {
        wasm32Calls.push(desc);
        assertI32Limit(desc.initial);
      }
    },
  };
  assert.equal(detectWasm64(wasm32OnlyApi), false);
  assert.equal(wasm32Calls.length, 2);
  assert.equal(wasm32Calls[0].address, 'i64');
  assert.equal(wasm32Calls[1].index, 'i64');

  // 4. Defensive edge cases: missing API or mock whose type reflection reports i32.
  assert.equal(detectWasm64(null), false);
  assert.equal(detectWasm64(undefined), false);
  assert.equal(detectWasm64({}), false);
  assert.equal(
    detectWasm64({
      Memory: class MockI32ReflectingMemory {
        type() {
          return { minimum: 1, maximum: 2, address: 'i32' };
        }
      },
    }),
    false
  );
});

test('W58 bootstrapPlayGta6App capability probe reports wasm64 for address: i64, legacy index: i64, and non-Wasm64 engines', async () => {
  async function probeWithWebAssembly(mockWebAssembly) {
    const clock = createClock(11000);
    const raf = createFakeRaf(clock);
    const timers = createFakeTimers();
    const webgpu = createMockWebGpuEnv();
    const hud = createMockHud();
    const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
      clock,
      raf,
      timers,
      gpu: webgpu.gpu,
    });
    windowObj.WebAssembly = mockWebAssembly;

    const app = bootstrapPlayGta6App({
      windowObj,
      documentObj,
      navigatorObj,
      fetchImpl: createMockFetch({ ok: true, hasWasmBinary: false }),
      canvas: webgpu.canvas,
      hud,
      now: clock.now,
      raf: raf.raf,
      cancelRaf: raf.cancelRaf,
      setInterval: timers.setInterval,
      clearInterval: timers.clearInterval,
    });

    try {
      const env = await app.session.probeEnvironment();
      return env.capabilities.wasm64;
    } finally {
      await app.destroy();
    }
  }

  // Modern engine accepting only `{ address: 'i64' }`.
  const modernResult = await probeWithWebAssembly({
    Memory: class MockModernMemory {
      constructor(desc) {
        if (desc.address !== 'i64') {
          return +desc.initial;
        }
      }
    },
  });
  assert.equal(modernResult, true);

  // Legacy engine accepting only `{ index: 'i64' }`.
  const legacyResult = await probeWithWebAssembly({
    Memory: class MockLegacyMemory {
      constructor(desc) {
        if (desc.index !== 'i64') {
          return +desc.initial;
        }
      }
    },
  });
  assert.equal(legacyResult, true);

  // Non-Wasm64 engine rejecting BigInt/i64 memory.
  const wasm32Result = await probeWithWebAssembly({
    Memory: class MockWasm32Memory {
      constructor(desc) {
        return +desc.initial;
      }
    },
  });
  assert.equal(wasm32Result, false);

  // Explicitly absent WebAssembly API on windowObj.
  const absentResult = await probeWithWebAssembly(null);
  assert.equal(absentResult, false);
});

test('W73 sanitizeSideModuleUrls and extractBootAssetUrls filter unsafe paths and preserve clean split-module metadata', () => {
  assert.equal(typeof DEFAULT_MAX_VALIDATION_ERRORS, 'number');
  assert.ok(DEFAULT_MAX_VALIDATION_ERRORS >= 1);

  assert.deepEqual(sanitizeSideModuleUrls(null), []);
  assert.deepEqual(sanitizeSideModuleUrls('not-an-array'), []);
  assert.deepEqual(
    sanitizeSideModuleUrls([
      '/dist/guest_part_0.wasm',
      ' /dist/guest_part_1.wasm?token=secret ',
      '/dist/guest_part_0.wasm', // duplicate
      'guest_part_2.wasm',
      'https://playgta6.local/dist/guest_part_3.wasm?auth=123',
      'https://user:pass@playgta6.local/dist/cred.wasm', // credentials rejected
      'file:///C:/Users/alice/guest_part_4.wasm', // file URL rejected
      'C:\\Users\\alice\\guest_part_5.wasm', // Windows host path rejected
      '/home/alice/work/guest_part_6.wasm', // POSIX host path rejected
      '/Users/bob/guest_part_7.wasm', // macOS host path rejected
      '../escape/guest_part_8.wasm', // traversal rejected
      '/dist/runtime.js', // non-wasm rejected
      '',
      42,
    ]),
    [
      '/dist/guest_part_0.wasm',
      '/dist/guest_part_1.wasm',
      'guest_part_2.wasm',
      'https://playgta6.local/dist/guest_part_3.wasm',
    ]
  );

  // Backwards compatibility: objects without sideModule fields do not gain sideModule keys.
  assert.deepEqual(extractBootAssetUrls({ bootBase: '/boot/' }), { bootBase: '/boot/' });

  // Derives sideModuleCount when only sideModuleUrls is present.
  assert.deepEqual(
    extractBootAssetUrls({
      sideModuleUrls: ['/dist/guest_part_0.wasm', '/dist/guest_part_1.wasm', 'C:\\bad\\x.wasm'],
    }),
    {
      sideModuleUrls: ['/dist/guest_part_0.wasm', '/dist/guest_part_1.wasm'],
      sideModuleCount: 2,
    }
  );

  // Preserves explicit sideModuleCount when sideModuleUrls is absent.
  assert.deepEqual(extractBootAssetUrls({ sideModuleCount: 4 }), { sideModuleCount: 4 });

  // Handles explicit empty sideModuleUrls with count 0.
  assert.deepEqual(extractBootAssetUrls({ sideModuleUrls: [], sideModuleCount: 0 }), {
    sideModuleUrls: [],
    sideModuleCount: 0,
  });
});

test('W73 forwards /api/status sideModuleUrls and sideModuleCount into loader boot payload and optional session/HUD hooks', async () => {
  const clock = createClock(12000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();

  const offscreenCanvas = { kind: 'MockOffscreenCanvas' };
  webgpu.canvas.transferControlToOffscreen = () => offscreenCanvas;

  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });

  const workerCalls = [];
  const postedMessages = [];
  const mockWorker = createFakeEventTarget({
    postMessage(payload, transfer) {
      postedMessages.push({ payload, transfer });
    },
    terminate() {},
  });

  const hudControllerSideModules = [];
  const hudController = {
    updateSideModules(info) {
      hudControllerSideModules.push(info);
    },
  };

  const sessionSideModules = [];

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch({
      ok: true,
      coopCoep: true,
      dataDirConfigured: true,
      hasManifest: true,
      hasBootset: true,
      hasWasmBinary: true,
      shaderPackCount: 1,
      sideModuleCount: 2,
      sideModuleUrls: [
        '/dist/guest_part_0.wasm',
        '/dist/guest_part_1.wasm',
        '/home/alice/leak.wasm',
      ],
    }),
    canvas: webgpu.canvas,
    hud,
    hudController,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker(args) {
      workerCalls.push(args);
      return mockWorker;
    },
  });

  // Attach optional session hook (simulating W72 SessionController extension).
  app.session.setSideModules = (info) => {
    sessionSideModules.push(info);
  };

  // Attach a throwing post-bootstrap window.playgta6HudController to verify error isolation.
  windowObj.playgta6HudController = {
    setSideModules() {
      throw new Error('simulated HUD controller failure');
    },
  };

  const started = await app.start();
  assert.equal(started.ok, true);
  assert.equal(started.mode, SESSION_MODE.WASM);
  assert.equal(workerCalls.length, 1);
  assert.equal(postedMessages.length, 1);

  assert.deepEqual(workerCalls[0].sideModuleUrls, [
    '/dist/guest_part_0.wasm',
    '/dist/guest_part_1.wasm',
  ]);
  assert.equal(workerCalls[0].sideModuleCount, 2);

  const initPayload = postedMessages[0].payload;
  assert.deepEqual(initPayload.sideModuleUrls, [
    '/dist/guest_part_0.wasm',
    '/dist/guest_part_1.wasm',
  ]);
  assert.equal(initPayload.sideModuleCount, 2);

  assert.equal(sessionSideModules.length, 1);
  assert.equal(sessionSideModules[0].sideModuleCount, 2);
  assert.deepEqual(sessionSideModules[0].sideModuleUrls, [
    '/dist/guest_part_0.wasm',
    '/dist/guest_part_1.wasm',
  ]);

  assert.equal(hudControllerSideModules.length, 1);
  assert.equal(hudControllerSideModules[0].sideModuleCount, 2);
  assert.deepEqual(hudControllerSideModules[0].sideModuleUrls, [
    '/dist/guest_part_0.wasm',
    '/dist/guest_part_1.wasm',
  ]);

  const sideMetric = hud.metricsUpdates.find((m) => m.sideModuleCount === 2);
  assert.ok(sideMetric, 'hud.updateMetrics must receive discovered sideModuleCount');
  assert.deepEqual(sideMetric.sideModuleUrls, [
    '/dist/guest_part_0.wasm',
    '/dist/guest_part_1.wasm',
  ]);

  const snap = app.getDiagnosticsSnapshot();
  assert.equal(snap.bootMode, 'real-wasm64');
  assert.equal(snap.sideModuleCount, 2);

  await app.destroy();
  assert.equal(windowObj.__playgta6, undefined);
});

test('W73 forwards wgpu_worker gpu-validation-error and splash-frame-presented to optional HUD/session hooks safely', async () => {
  const clock = createClock(15000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();

  const broadcastInstances = [];
  class MockBroadcastChannel {
    constructor(name) {
      this.name = name;
      this.onmessage = null;
      this.closed = false;
      broadcastInstances.push(this);
    }
    close() {
      this.closed = true;
    }
  }

  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });
  windowObj.BroadcastChannel = MockBroadcastChannel;

  const mockWorker = createFakeEventTarget({
    postMessage() {},
    terminate() {},
  });

  const panelUpdates = [];
  const mockGpuBootPanel = {
    update(snapshot) {
      panelUpdates.push(snapshot);
    },
  };

  const sessionValidationErrors = [];
  const sessionSplashFrames = [];
  const hudValidationErrors = [];
  const hudSplashFrames = [];

  const hudController = {
    onGpuValidationError(ev) {
      hudValidationErrors.push(ev);
    },
    onSplashFramePresented(ev) {
      hudSplashFrames.push(ev);
    },
  };

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch({
      ok: true,
      coopCoep: true,
      hasWasmBinary: true,
      sideModuleUrls: ['/dist/guest_part_0.wasm'],
      sideModuleCount: 1,
    }),
    canvas: webgpu.canvas,
    hud,
    hudController,
    gpuBootPanel: mockGpuBootPanel,
    maxValidationErrors: 3,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker() {
      return mockWorker;
    },
  });

  app.session.onGpuValidationError = (ev) => {
    sessionValidationErrors.push(ev);
  };
  app.session.onSplashFramePresented = (ev) => {
    sessionSplashFrames.push(ev);
  };

  // Post-bootstrap window.playgta6HudController that throws on every hook must be isolated.
  windowObj.playgta6HudController = {
    recordGpuValidationError() {
      throw new Error('throwing validation hook');
    },
    recordSplashFramePresented() {
      throw new Error('throwing splash hook');
    },
  };

  const started = await app.start();
  assert.equal(started.ok, true);
  assert.equal(started.mode, SESSION_MODE.WASM);
  assert.equal(broadcastInstances.length, 1);

  // 1. Dispatch gpu-validation-error over worker message with sensitive host path & long hex dump.
  mockWorker.dispatch('message', {
    data: {
      type: 'gpu-validation-error',
      errorType: 'validation',
      message:
        'Shader error at C:\\Users\\alice\\shader.wgsl with dump 0xdeadbeefcafebabe0123456789abcdef0123456789abcdef01234567',
      timestampMs: 15100,
    },
  });

  // 2. Dispatch out-of-memory error via BroadcastChannel('game-progress') with POSIX host path.
  broadcastInstances[0].onmessage({
    data: {
      type: 'gpu-validation-error',
      error: { name: 'GPUOutOfMemoryError', message: 'Allocation failed in /home/bob/vram.bin' },
      timestampMs: 15200,
    },
  });

  // 3. Dispatch 3 more errors to verify bounded recentValidationErrors cap (maxValidationErrors: 3).
  for (let i = 3; i <= 5; i += 1) {
    mockWorker.dispatch('message', {
      data: {
        type: 'gpu-validation-error',
        errorType: i === 5 ? 'internal' : 'validation',
        message: `Validation error #${i}`,
        timestampMs: 15000 + i * 100,
      },
    });
  }

  assert.equal(sessionValidationErrors.length, 5);
  assert.equal(hudValidationErrors.length, 5);
  assert.equal(sessionValidationErrors[0].errorType, 'validation');
  assert.ok(!sessionValidationErrors[0].message.includes('alice'));
  assert.ok(!sessionValidationErrors[0].message.includes('deadbeefcafebabe'));
  assert.equal(sessionValidationErrors[1].errorType, 'out-of-memory');
  assert.ok(!sessionValidationErrors[1].message.includes('/home/bob'));
  assert.equal(sessionValidationErrors[4].errorType, 'internal');

  // Primary hud.appendLog also surfaced validation errors because hud had no dedicated hook.
  assert.ok(hud.logs.some((entry) => entry.level === 'error' && entry.message.includes('[WebGPU validation]')));

  // 4. Dispatch splash-frame-presented messages (first present, second frame, and non-fatal failure).
  mockWorker.dispatch('message', {
    data: {
      type: 'splash-frame-presented',
      frameIndex: 0,
      framesPresented: 1,
      ok: true,
      firstPresent: true,
    },
  });
  mockWorker.dispatch('message', {
    data: {
      type: 'splash-frame-presented',
      frameIndex: 1,
      framesPresented: 2,
      ok: true,
    },
  });
  mockWorker.dispatch('message', {
    data: {
      type: 'splash-frame-presented',
      frameIndex: 2,
      framesPresented: 2,
      ok: false,
      error: 'Context lost at /Users/alice/frame.png',
      reason: 'device-not-ready',
    },
  });

  assert.equal(sessionSplashFrames.length, 3);
  assert.equal(hudSplashFrames.length, 3);
  assert.equal(sessionSplashFrames[0].ok, true);
  assert.equal(sessionSplashFrames[0].firstPresent, true);
  assert.equal(sessionSplashFrames[0].framesPresented, 1);
  assert.equal(sessionSplashFrames[1].framesPresented, 2);
  assert.equal(sessionSplashFrames[2].ok, false);
  assert.ok(!sessionSplashFrames[2].error.includes('/Users/alice'));

  // FIRST_PRESENT milestone was marked in BootTracer and panel received frameStats.
  assert.equal(app.bootTracer.getSummary().stage, 'FIRST_PRESENT');
  assert.ok(panelUpdates.some((u) => u.frameStats && u.frameStats.framesPresented === 2));

  // Verify bounded diagnostics snapshot on both app handle and globalScope.__playgta6.
  const snap = app.getDiagnosticsSnapshot();
  assert.ok(windowObj.__playgta6);
  assert.deepEqual(Object.keys(windowObj.__playgta6), ['getDiagnosticsSnapshot']);
  assert.equal(Object.isFrozen(windowObj.__playgta6), true);

  const globalSnap = windowObj.__playgta6.getDiagnosticsSnapshot();
  assert.deepEqual(globalSnap, snap);
  assert.equal(Object.isFrozen(snap), true);
  assert.equal(Object.isFrozen(snap.recentValidationErrors), true);
  assert.equal(Object.isFrozen(snap.recentValidationErrors[0]), true);

  assert.equal(snap.bootMode, 'real-wasm64');
  assert.equal(snap.milestone, 'FIRST_PRESENT');
  assert.equal(snap.sideModuleCount, 1);
  assert.equal(snap.validationErrorCount, 5);
  assert.equal(snap.recentValidationErrors.length, 3);
  assert.deepEqual(
    snap.recentValidationErrors.map((e) => e.message),
    ['Validation error #3', 'Validation error #4', 'Validation error #5']
  );
  assert.equal(snap.framesPresented, 2);
  assert.equal(snap.splashFramesPresented, 2);
  assert.equal(snap.destroyed, false);

  await app.destroy();
  assert.equal(windowObj.__playgta6, undefined);
  assert.equal(app.getDiagnosticsSnapshot().destroyed, true);
});

test('W73 globalScope.__playgta6 supports custom globalScope, attachGlobalHook opt-out, smoke diagnostics, and safe teardown', async () => {
  const clock = createClock(20000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();

  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });

  // 1. Custom globalScope receives sanitized __playgta6 hook.
  const customGlobal = {};
  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    globalScope: customGlobal,
    fetchImpl: createMockFetch({ ok: true, hasWasmBinary: false }),
    canvas: webgpu.canvas,
    hud,
    now: clock.now,
    raf: raf.raf,
    cancelRaf: raf.cancelRaf,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
  });

  assert.equal(windowObj.__playgta6, undefined);
  assert.ok(customGlobal.__playgta6);
  assert.equal(customGlobal.__playgta6.getDiagnosticsSnapshot().bootMode, 'idle');
  assert.equal(customGlobal.__playgta6.getDiagnosticsSnapshot().milestone, 'NOT_STARTED');

  await app.start();
  raf.flushFrame(16);
  raf.flushFrame(16);

  // Smoke device uncapturederror updates validationErrorCount without leaking paths.
  webgpu.device.dispatch('uncapturederror', {
    error: { name: 'GPUValidationError', message: 'Bad layout in C:\\Users\\secret\\file.wgsl' },
  });

  const smokeSnap = customGlobal.__playgta6.getDiagnosticsSnapshot();
  assert.equal(smokeSnap.bootMode, 'smoke');
  assert.equal(smokeSnap.milestone, 'FIRST_PRESENT');
  assert.equal(smokeSnap.sideModuleCount, 0);
  assert.ok(smokeSnap.framesPresented >= 2);
  assert.equal(smokeSnap.validationErrorCount, 1);
  assert.equal(smokeSnap.recentValidationErrors.length, 1);
  assert.ok(!smokeSnap.recentValidationErrors[0].message.includes('secret'));

  // Optional session.getBootMode() override is honored if present.
  app.session.getBootMode = () => 'real-wasm64';
  assert.equal(app.getDiagnosticsSnapshot().bootMode, 'real-wasm64');

  // If another instance replaces customGlobal.__playgta6, destroy() must not clobber it.
  const sentinelHook = Object.freeze({ getDiagnosticsSnapshot: () => ({ sentinel: true }) });
  customGlobal.__playgta6 = sentinelHook;
  await app.destroy();
  assert.equal(customGlobal.__playgta6, sentinelHook);

  // 2. attachGlobalHook: false skips attaching __playgta6 even when windowObj is provided.
  const optOutGlobal = {};
  const optOutApp = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    globalScope: optOutGlobal,
    attachGlobalHook: false,
    fetchImpl: createMockFetch({ ok: true, hasWasmBinary: false }),
    canvas: webgpu.canvas,
    hud,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
  });
  assert.equal(optOutGlobal.__playgta6, undefined);
  assert.equal(optOutApp.getDiagnosticsSnapshot().bootMode, 'idle');
  await optOutApp.destroy();

  // 3. Frozen globalScope never throws during bootstrap or destroy.
  const frozenGlobal = Object.freeze({});
  const frozenApp = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    globalScope: frozenGlobal,
    fetchImpl: createMockFetch({ ok: true, hasWasmBinary: false }),
    canvas: webgpu.canvas,
    hud,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
  });
  assert.equal(frozenGlobal.__playgta6, undefined);
  assert.equal(frozenApp.getDiagnosticsSnapshot().bootMode, 'idle');
  await frozenApp.destroy();
});

test('W79 forwards preferredShaderPackUrl from /api/status, honors explicit overrides, and falls back to /shaders/fallback.wgsl.pack', async () => {
  const { createRequire } = await import('node:module');
  const require = createRequire(import.meta.url);
  const { BootOrchestrator } = require('../web/runtime/boot_orchestrator.js');
  const { createWorkerBootInitPayloads } = require('../web/loader.js');

  assert.equal(DEFAULT_SHADER_PACK_URL, '/shaders/fallback.wgsl.pack');

  // extractBootAssetUrls prefers preferredShaderPackUrl over shaderPackUrl and legacy aliases.
  assert.deepEqual(
    extractBootAssetUrls({
      preferredShaderPackUrl: '/shaders/splash.wgsl.pack',
      shaderPackUrl: '/shaders/fallback.wgsl.pack',
    }),
    { shaderPackUrl: '/shaders/splash.wgsl.pack' }
  );
  assert.deepEqual(
    extractBootAssetUrls({
      preferredShaderPackUrl: null,
      shaderPackUrl: '/shaders/fallback.wgsl.pack',
    }),
    { shaderPackUrl: '/shaders/fallback.wgsl.pack' }
  );
  assert.deepEqual(
    extractBootAssetUrls({
      preferredShaderPackUrl: null,
      shaderPackUrl: null,
    }),
    {}
  );

  async function runWasmLaunch({ statusPayload, bootstrapOptions = {}, startOpts = undefined }) {
    const clock = createClock(25000);
    const raf = createFakeRaf(clock);
    const timers = createFakeTimers();
    const webgpu = createMockWebGpuEnv();
    const hud = createMockHud();

    const offscreenCanvas = { kind: 'MockOffscreenCanvas' };
    webgpu.canvas.transferControlToOffscreen = () => offscreenCanvas;

    const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
      clock,
      raf,
      timers,
      gpu: webgpu.gpu,
    });

    const workerCalls = [];
    const postedMessages = [];
    const mockWorker = createFakeEventTarget({
      postMessage(payload, transfer) {
        postedMessages.push({ payload, transfer });
      },
      terminate() {},
    });

    const app = bootstrapPlayGta6App({
      windowObj,
      documentObj,
      navigatorObj,
      fetchImpl: createMockFetch(statusPayload),
      canvas: webgpu.canvas,
      hud,
      now: clock.now,
      setInterval: timers.setInterval,
      clearInterval: timers.clearInterval,
      createWasmWorker(args) {
        workerCalls.push(args);
        return mockWorker;
      },
      ...bootstrapOptions,
    });

    try {
      const started = await app.start(startOpts);
      assert.equal(started.ok, true);
      assert.equal(started.mode, SESSION_MODE.WASM);
      assert.equal(workerCalls.length, 1);
      assert.equal(postedMessages.length, 1);
      const initPayload = postedMessages[0].payload;
      const workerInits = createWorkerBootInitPayloads(initPayload);
      return {
        workerCall: workerCalls[0],
        initPayload,
        workerInits,
      };
    } finally {
      await app.destroy();
    }
  }

  // 1. /api/status reports preferredShaderPackUrl -> forwarded to loader.js and wgpu_worker.js.
  const splashLaunch = await runWasmLaunch({
    statusPayload: {
      ok: true,
      coopCoep: true,
      hasWasmBinary: true,
      shaderPackCount: 2,
      preferredShaderPackUrl: '/shaders/splash.wgsl.pack',
      shaderPackUrl: '/shaders/fallback.wgsl.pack',
    },
  });
  assert.equal(splashLaunch.workerCall.shaderPackUrl, '/shaders/splash.wgsl.pack');
  assert.equal(splashLaunch.initPayload.shaderPackUrl, '/shaders/splash.wgsl.pack');
  assert.equal(splashLaunch.workerInits.gpuInit.shaderPackUrl, '/shaders/splash.wgsl.pack');

  // 2. BootOrchestrator path also preserves status.preferredShaderPackUrl over normalized assets.
  const orchSplashLaunch = await runWasmLaunch({
    statusPayload: {
      ok: true,
      coopCoep: true,
      hasWasmBinary: true,
      pgimUrl: '/boot/guest.pgim',
      preferredShaderPackUrl: '/shaders/splash.wgsl.pack',
      shaderPackUrl: '/shaders/fallback.wgsl.pack',
    },
    bootstrapOptions: {
      enableBootOrchestrator: true,
      BootOrchestratorCtor: BootOrchestrator,
      pgimPreloader: {
        async preloadBootBundle() {
          return { pgim: new Uint8Array(4), shaderPackBytes: null, reflectionManifest: null, bytesLoaded: 4 };
        },
      },
    },
  });
  assert.equal(orchSplashLaunch.workerCall.shaderPackUrl, '/shaders/splash.wgsl.pack');
  assert.equal(orchSplashLaunch.initPayload.shaderPackUrl, '/shaders/splash.wgsl.pack');
  assert.equal(orchSplashLaunch.workerInits.gpuInit.shaderPackUrl, '/shaders/splash.wgsl.pack');

  // 3. Explicit bootstrap options.shaderPackUrl override wins over /api/status.
  const explicitOptionLaunch = await runWasmLaunch({
    statusPayload: {
      ok: true,
      coopCoep: true,
      hasWasmBinary: true,
      preferredShaderPackUrl: '/shaders/splash.wgsl.pack',
      shaderPackUrl: '/shaders/splash.wgsl.pack',
    },
    bootstrapOptions: {
      shaderPackUrl: '/shaders/custom-override.wgsl.pack',
    },
  });
  assert.equal(explicitOptionLaunch.workerCall.shaderPackUrl, '/shaders/custom-override.wgsl.pack');
  assert.equal(explicitOptionLaunch.initPayload.shaderPackUrl, '/shaders/custom-override.wgsl.pack');
  assert.equal(
    explicitOptionLaunch.workerInits.gpuInit.shaderPackUrl,
    '/shaders/custom-override.wgsl.pack'
  );

  // 4. Explicit start({ shaderPackUrl }) override wins over both bootstrap option and /api/status.
  const explicitStartLaunch = await runWasmLaunch({
    statusPayload: {
      ok: true,
      coopCoep: true,
      hasWasmBinary: true,
      preferredShaderPackUrl: '/shaders/splash.wgsl.pack',
      shaderPackUrl: '/shaders/splash.wgsl.pack',
    },
    bootstrapOptions: {
      shaderPackUrl: '/shaders/custom-override.wgsl.pack',
    },
    startOpts: {
      shaderPackUrl: '/shaders/per-start.wgsl.pack',
    },
  });
  assert.equal(explicitStartLaunch.workerCall.shaderPackUrl, '/shaders/per-start.wgsl.pack');
  assert.equal(explicitStartLaunch.initPayload.shaderPackUrl, '/shaders/per-start.wgsl.pack');
  assert.equal(
    explicitStartLaunch.workerInits.gpuInit.shaderPackUrl,
    '/shaders/per-start.wgsl.pack'
  );

  // 5. When /api/status omits or nulls shader pack URLs, degrades safely to /shaders/fallback.wgsl.pack.
  const fallbackOmittedLaunch = await runWasmLaunch({
    statusPayload: {
      ok: true,
      coopCoep: true,
      hasWasmBinary: true,
      shaderPackCount: 0,
    },
  });
  assert.equal(fallbackOmittedLaunch.workerCall.shaderPackUrl, '/shaders/fallback.wgsl.pack');
  assert.equal(fallbackOmittedLaunch.initPayload.shaderPackUrl, '/shaders/fallback.wgsl.pack');
  assert.equal(
    fallbackOmittedLaunch.workerInits.gpuInit.shaderPackUrl,
    '/shaders/fallback.wgsl.pack'
  );

  const fallbackNullLaunch = await runWasmLaunch({
    statusPayload: {
      ok: true,
      coopCoep: true,
      hasWasmBinary: true,
      preferredShaderPackUrl: null,
      shaderPackUrl: null,
    },
  });
  assert.equal(fallbackNullLaunch.workerCall.shaderPackUrl, '/shaders/fallback.wgsl.pack');
  assert.equal(fallbackNullLaunch.initPayload.shaderPackUrl, '/shaders/fallback.wgsl.pack');
  assert.equal(
    fallbackNullLaunch.workerInits.gpuInit.shaderPackUrl,
    '/shaders/fallback.wgsl.pack'
  );
});

test('W117 parseBootUrlOptions and sanitizeInputScriptPreset parse valid options and reject unsafe inputs', () => {
  assert.equal(DEFAULT_STREAM_SHADERS_THRESHOLD_MB, 64);
  assert.equal(DEFAULT_STREAM_SHADERS_THRESHOLD_BYTES, 64 * 1024 * 1024);
  assert.equal(MAX_STREAM_THRESHOLD_MB, 4096);
  assert.equal(MAX_STREAM_THRESHOLD_BYTES, 4096 * 1024 * 1024);

  // Empty / missing inputs return frozen defaults.
  for (const empty of [undefined, null, '', '?', '?foo=bar', {}, { search: '' }]) {
    const parsed = parseBootUrlOptions(empty);
    assert.equal(Object.isFrozen(parsed), true);
    assert.deepEqual(parsed, {
      inputScript: null,
      streamShaders: false,
      streamThresholdMb: null,
      streamThresholdBytes: null,
      dumpPipelines: false,
      allowDumpWrites: false,
      dumpEndpoint: null,
    });
  }

  // Valid inputScript preset (exact, uppercase, mixed-case, whitespace, URL, URLSearchParams).
  assert.equal(sanitizeInputScriptPreset('start-screen'), 'start-screen');
  assert.equal(sanitizeInputScriptPreset('START-SCREEN'), 'start-screen');
  assert.equal(sanitizeInputScriptPreset('  Start-Screen  '), 'start-screen');
  assert.equal(parseBootUrlOptions('?inputScript=start-screen').inputScript, 'start-screen');
  assert.equal(parseBootUrlOptions('?inputscript=START-SCREEN').inputScript, 'start-screen');
  assert.equal(
    parseBootUrlOptions('https://playgta6.local/?inputScript=Start-Screen#top').inputScript,
    'start-screen'
  );
  assert.equal(
    parseBootUrlOptions(new URLSearchParams('inputScript=start-screen')).inputScript,
    'start-screen'
  );
  assert.equal(
    parseBootUrlOptions({ search: '?inputScript=start-screen' }).inputScript,
    'start-screen'
  );

  // Invalid / unsafe inputScript values are strictly rejected.
  const unsafePresets = [
    '../../evil',
    '../start-screen',
    'start-screen/../../evil',
    'start-screen\\evil',
    '<script>alert(1)</script>',
    '__proto__',
    'constructor',
    'toString',
    'valueOf',
    'unknown-preset',
    'start screen',
    'a\0b',
    '',
    '   ',
    'a'.repeat(128),
  ];
  for (const bad of unsafePresets) {
    assert.equal(sanitizeInputScriptPreset(bad), null, `Expected reject for ${JSON.stringify(bad)}`);
    assert.equal(
      parseBootUrlOptions(`?inputScript=${encodeURIComponent(bad)}`).inputScript,
      null,
      `Expected URL reject for ${JSON.stringify(bad)}`
    );
  }
  assert.equal(sanitizeInputScriptPreset(null), null);
  assert.equal(sanitizeInputScriptPreset(123), null);

  // streamShaders=1 / true enables default 64 MiB threshold.
  for (const q of ['?streamShaders=1', '?streamShaders=true', '?streamShaders=TRUE', '?streamshaders=yes']) {
    assert.deepEqual(parseBootUrlOptions(q), {
      inputScript: null,
      streamShaders: true,
      streamThresholdMb: 64,
      streamThresholdBytes: 64 * 1024 * 1024,
      dumpPipelines: false,
      allowDumpWrites: false,
      dumpEndpoint: null,
    });
  }

  // streamThresholdMb=32 resolves to exact 32 MiB bytes (both with and without streamShaders=1).
  assert.deepEqual(parseBootUrlOptions('?streamShaders=1&streamThresholdMb=32'), {
    inputScript: null,
    streamShaders: true,
    streamThresholdMb: 32,
    streamThresholdBytes: 32 * 1024 * 1024,
    dumpPipelines: false,
    allowDumpWrites: false,
    dumpEndpoint: null,
  });
  assert.deepEqual(parseBootUrlOptions('?streamThresholdMb=32'), {
    inputScript: null,
    streamShaders: true,
    streamThresholdMb: 32,
    streamThresholdBytes: 32 * 1024 * 1024,
    dumpPipelines: false,
    allowDumpWrites: false,
    dumpEndpoint: null,
  });

  // Fractional MB and boundary clamping.
  assert.deepEqual(parseBootUrlOptions('?streamThresholdMb=0.5'), {
    inputScript: null,
    streamShaders: true,
    streamThresholdMb: 0.5,
    streamThresholdBytes: 512 * 1024,
    dumpPipelines: false,
    allowDumpWrites: false,
    dumpEndpoint: null,
  });
  assert.deepEqual(parseBootUrlOptions('?streamThresholdMb=999999'), {
    inputScript: null,
    streamShaders: true,
    streamThresholdMb: MAX_STREAM_THRESHOLD_MB,
    streamThresholdBytes: MAX_STREAM_THRESHOLD_BYTES,
    dumpPipelines: false,
    allowDumpWrites: false,
    dumpEndpoint: null,
  });
  const tiny = parseBootUrlOptions('?streamThresholdMb=0.00000001');
  assert.equal(tiny.streamShaders, true);
  assert.equal(tiny.streamThresholdBytes, 1);

  // Explicit streamShaders=0 / false disables streaming even when streamThresholdMb is set.
  assert.deepEqual(parseBootUrlOptions('?streamShaders=0&streamThresholdMb=32'), {
    inputScript: null,
    streamShaders: false,
    streamThresholdMb: null,
    streamThresholdBytes: null,
    dumpPipelines: false,
    allowDumpWrites: false,
    dumpEndpoint: null,
  });
  assert.deepEqual(parseBootUrlOptions('?streamShaders=false'), {
    inputScript: null,
    streamShaders: false,
    streamThresholdMb: null,
    streamThresholdBytes: null,
    dumpPipelines: false,
    allowDumpWrites: false,
    dumpEndpoint: null,
  });

  // Invalid streamThresholdMb values are ignored without crashing.
  for (const badMb of ['0', '-5', 'NaN', 'Infinity', '-Infinity', 'abc', '0x10', '1e309', '']) {
    assert.deepEqual(parseBootUrlOptions(`?streamThresholdMb=${encodeURIComponent(badMb)}`), {
      inputScript: null,
      streamShaders: false,
      streamThresholdMb: null,
      streamThresholdBytes: null,
      dumpPipelines: false,
      allowDumpWrites: false,
      dumpEndpoint: null,
    });
  }
  // Invalid streamThresholdMb with valid streamShaders=1 falls back to default 64 MiB.
  assert.deepEqual(parseBootUrlOptions('?streamShaders=1&streamThresholdMb=-10'), {
    inputScript: null,
    streamShaders: true,
    streamThresholdMb: 64,
    streamThresholdBytes: 64 * 1024 * 1024,
    dumpPipelines: false,
    allowDumpWrites: false,
    dumpEndpoint: null,
  });

  // Combined inputScript + streamShaders + streamThresholdMb.
  assert.deepEqual(parseBootUrlOptions('?inputScript=START-SCREEN&streamShaders=1&streamThresholdMb=32'), {
    inputScript: 'start-screen',
    streamShaders: true,
    streamThresholdMb: 32,
    streamThresholdBytes: 32 * 1024 * 1024,
    dumpPipelines: false,
    allowDumpWrites: false,
    dumpEndpoint: null,
  });
});

test('W117 ?inputScript=start-screen auto-starts inputBridge.startScript on boot and stops on stop() and destroy()', async () => {
  const clock = createClock(30000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();

  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });
  windowObj.location = {
    origin: 'https://playgta6.local',
    search: '?inputScript=START-SCREEN',
  };

  const mockWorker = createFakeEventTarget({
    postMessage() {},
    terminate() {},
  });

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch({ ok: true, coopCoep: true, hasWasmBinary: true }),
    canvas: webgpu.canvas,
    hud,
    now: clock.now,
    raf: raf.raf,
    cancelRaf: raf.cancelRaf,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker() {
      return mockWorker;
    },
  });

  assert.equal(app.bootUrlOptions.inputScript, 'start-screen');

  const startScriptCalls = [];
  let stopScriptCalls = 0;
  const origStartScript = app.inputBridge.startScript.bind(app.inputBridge);
  const origStopScript = app.inputBridge.stopScript.bind(app.inputBridge);
  app.inputBridge.startScript = (...args) => {
    startScriptCalls.push(args);
    return origStartScript(...args);
  };
  app.inputBridge.stopScript = (...args) => {
    stopScriptCalls += 1;
    return origStopScript(...args);
  };

  // Not started before app.start().
  assert.equal(startScriptCalls.length, 0);
  assert.equal(app.inputBridge.getScriptState().active, false);

  const started = await app.start();
  assert.equal(started.ok, true);
  assert.equal(started.mode, SESSION_MODE.WASM);
  assert.equal(startScriptCalls.length, 1);
  assert.deepEqual(startScriptCalls[0], ['start-screen']);
  assert.equal(app.inputBridge.getScriptState().active, true);
  assert.equal(app.inputBridge.getScriptState().preset, 'start-screen');

  // stop() cleanly invokes stopScript() and halts active playback.
  const stopped = await app.stop();
  assert.equal(stopped.ok, true);
  assert.ok(stopScriptCalls >= 1);
  assert.equal(app.inputBridge.getScriptState().active, false);

  // Restarting starts the preset again; destroy() invokes stopScript() directly on teardown.
  const stopCountBeforeRestart = stopScriptCalls;
  const restarted = await app.start();
  assert.equal(restarted.ok, true);
  assert.equal(startScriptCalls.length, 2);
  assert.deepEqual(startScriptCalls[1], ['start-screen']);
  assert.equal(app.inputBridge.getScriptState().active, true);

  await app.destroy();
  assert.ok(stopScriptCalls > stopCountBeforeRestart);
  assert.equal(app.inputBridge.getScriptState().active, false);
  assert.equal(app.inputBridge.disposed, true);
});

test('W117 invalid/unsafe ?inputScript=../../evil is rejected and never passed to startScript', async () => {
  const clock = createClock(31000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();

  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });
  windowObj.location = {
    origin: 'https://playgta6.local',
    search: '?inputScript=../../evil',
  };

  const mockWorker = createFakeEventTarget({
    postMessage() {},
    terminate() {},
  });

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch({ ok: true, coopCoep: true, hasWasmBinary: true }),
    canvas: webgpu.canvas,
    hud,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker() {
      return mockWorker;
    },
  });

  assert.equal(app.bootUrlOptions.inputScript, null);

  const startScriptCalls = [];
  app.inputBridge.startScript = (...args) => {
    startScriptCalls.push(args);
  };

  const started = await app.start();
  assert.equal(started.ok, true);
  assert.equal(startScriptCalls.length, 0);

  // Unsafe per-start override is also rejected and never reaches startScript.
  await app.stop();
  const restarted = await app.start({ inputScript: '../../evil' });
  assert.equal(restarted.ok, true);
  assert.equal(startScriptCalls.length, 0);

  await app.destroy();
});

test('W117 ?streamShaders=1 and ?streamThresholdMb=32 forward streamThresholdBytes into boot options while omitted params keep defaults untouched', async () => {
  async function launchWithUrl({ search, bootstrapOptions = {}, startOpts = undefined } = {}) {
    const clock = createClock(32000);
    const raf = createFakeRaf(clock);
    const timers = createFakeTimers();
    const webgpu = createMockWebGpuEnv();
    const hud = createMockHud();

    const offscreenCanvas = { kind: 'MockOffscreenCanvas' };
    webgpu.canvas.transferControlToOffscreen = () => offscreenCanvas;

    const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
      clock,
      raf,
      timers,
      gpu: webgpu.gpu,
    });
    if (search !== undefined) {
      windowObj.location = { origin: 'https://playgta6.local', search };
    }

    const workerCalls = [];
    const postedMessages = [];
    const mockWorker = createFakeEventTarget({
      postMessage(payload, transfer) {
        postedMessages.push({ payload, transfer });
      },
      terminate() {},
    });

    const app = bootstrapPlayGta6App({
      windowObj,
      documentObj,
      navigatorObj,
      fetchImpl: createMockFetch({
        ok: true,
        coopCoep: true,
        hasWasmBinary: true,
        preferredShaderPackUrl: '/shaders/splash.wgsl.pack',
      }),
      canvas: webgpu.canvas,
      hud,
      now: clock.now,
      setInterval: timers.setInterval,
      clearInterval: timers.clearInterval,
      createWasmWorker(args) {
        workerCalls.push(args);
        return mockWorker;
      },
      ...bootstrapOptions,
    });

    try {
      const started = await app.start(startOpts);
      assert.equal(started.ok, true);
      assert.equal(started.mode, SESSION_MODE.WASM);
      assert.equal(workerCalls.length, 1);
      assert.equal(postedMessages.length, 1);
      return {
        bootUrlOptions: app.bootUrlOptions,
        workerCall: workerCalls[0],
        initPayload: postedMessages[0].payload,
      };
    } finally {
      await app.destroy();
    }
  }

  // 1. Omitted params leave default boot options completely untouched.
  const defaultLaunch = await launchWithUrl({ search: '' });
  assert.equal(defaultLaunch.bootUrlOptions.streamShaders, false);
  assert.equal(defaultLaunch.bootUrlOptions.streamThresholdBytes, null);
  assert.equal('streamShaders' in defaultLaunch.workerCall, false);
  assert.equal('streamThresholdBytes' in defaultLaunch.workerCall, false);
  assert.equal('streamShaders' in defaultLaunch.initPayload, false);
  assert.equal('streamThresholdBytes' in defaultLaunch.initPayload, false);

  // 2. ?streamShaders=1 forwards streamShaders: true and default 64 MiB streamThresholdBytes.
  const defaultThresholdLaunch = await launchWithUrl({ search: '?streamShaders=1' });
  assert.equal(defaultThresholdLaunch.bootUrlOptions.streamShaders, true);
  assert.equal(defaultThresholdLaunch.bootUrlOptions.streamThresholdBytes, 64 * 1024 * 1024);
  assert.equal(defaultThresholdLaunch.workerCall.streamShaders, true);
  assert.equal(defaultThresholdLaunch.workerCall.streamThresholdBytes, 64 * 1024 * 1024);
  assert.equal(defaultThresholdLaunch.initPayload.streamShaders, true);
  assert.equal(defaultThresholdLaunch.initPayload.streamThresholdBytes, 64 * 1024 * 1024);

  // 3. ?streamShaders=1&streamThresholdMb=32 forwards exact 32 MiB streamThresholdBytes.
  const customThresholdLaunch = await launchWithUrl({
    search: '?streamShaders=1&streamThresholdMb=32',
  });
  assert.equal(customThresholdLaunch.bootUrlOptions.streamShaders, true);
  assert.equal(customThresholdLaunch.bootUrlOptions.streamThresholdMb, 32);
  assert.equal(customThresholdLaunch.bootUrlOptions.streamThresholdBytes, 32 * 1024 * 1024);
  assert.equal(customThresholdLaunch.workerCall.streamShaders, true);
  assert.equal(customThresholdLaunch.workerCall.streamThresholdBytes, 32 * 1024 * 1024);
  assert.equal(customThresholdLaunch.initPayload.streamShaders, true);
  assert.equal(customThresholdLaunch.initPayload.streamThresholdBytes, 32 * 1024 * 1024);

  // 4. ?streamThresholdMb=32 alone also enables streaming with 32 MiB threshold.
  const thresholdOnlyLaunch = await launchWithUrl({ search: '?streamThresholdMb=32' });
  assert.equal(thresholdOnlyLaunch.workerCall.streamShaders, true);
  assert.equal(thresholdOnlyLaunch.workerCall.streamThresholdBytes, 32 * 1024 * 1024);
  assert.equal(thresholdOnlyLaunch.initPayload.streamShaders, true);
  assert.equal(thresholdOnlyLaunch.initPayload.streamThresholdBytes, 32 * 1024 * 1024);

  // 5. Explicit bootstrap option streamShaders: false overrides URL query params.
  const disabledOverrideLaunch = await launchWithUrl({
    search: '?streamShaders=1&streamThresholdMb=32',
    bootstrapOptions: { streamShaders: false },
  });
  assert.equal(disabledOverrideLaunch.bootUrlOptions.streamShaders, false);
  assert.equal('streamShaders' in disabledOverrideLaunch.initPayload, false);
  assert.equal('streamThresholdBytes' in disabledOverrideLaunch.initPayload, false);

  // 6. Per-start override updates streamThresholdBytes cleanly.
  const perStartLaunch = await launchWithUrl({
    search: '?streamShaders=1&streamThresholdMb=32',
    startOpts: { streamThresholdMb: 16 },
  });
  assert.equal(perStartLaunch.workerCall.streamShaders, true);
  assert.equal(perStartLaunch.workerCall.streamThresholdBytes, 16 * 1024 * 1024);
  assert.equal(perStartLaunch.initPayload.streamShaders, true);
  assert.equal(perStartLaunch.initPayload.streamThresholdBytes, 16 * 1024 * 1024);
});

test('W184 parseBootUrlOptions and sanitizeInputScriptPreset support menu-flow, calibration-skip, intro-watch, dumpPipelines, and allowDumpWrites', () => {
  assert.equal(DEFAULT_DUMP_ENDPOINT, '/__pg6_dump/');

  // All W174 presets plus start-screen are accepted case-insensitively and canonicalized.
  const presets = ['start-screen', 'menu-flow', 'calibration-skip', 'intro-watch'];
  for (const preset of presets) {
    assert.equal(sanitizeInputScriptPreset(preset), preset);
    assert.equal(sanitizeInputScriptPreset(preset.toUpperCase()), preset);
    assert.equal(sanitizeInputScriptPreset(`  ${preset}  `), preset);
    assert.equal(parseBootUrlOptions(`?inputScript=${preset}`).inputScript, preset);
    assert.equal(
      parseBootUrlOptions(`https://playgta6.local/?inputscript=${preset.toUpperCase()}#boot`).inputScript,
      preset
    );
    assert.equal(
      parseBootUrlOptions(new URLSearchParams(`inputScript=${preset}`)).inputScript,
      preset
    );
  }

  // dumpPipelines=1 / true / yes / on enables dumpPipelines without turning on allowDumpWrites.
  for (const q of ['?dumpPipelines=1', '?dumpPipelines=true', '?dumpPipelines=TRUE', '?dumppipelines=yes', '?dumpPipelines=on']) {
    const parsed = parseBootUrlOptions(q);
    assert.equal(Object.isFrozen(parsed), true);
    assert.deepEqual(parsed, {
      inputScript: null,
      streamShaders: false,
      streamThresholdMb: null,
      streamThresholdBytes: null,
      dumpPipelines: true,
      allowDumpWrites: false,
      dumpEndpoint: null,
    });
  }

  // dumpPipelines=0 / false / invalid stays false.
  for (const q of ['?dumpPipelines=0', '?dumpPipelines=false', '?dumpPipelines=no', '?dumpPipelines=off', '?dumpPipelines=maybe', '?dumpPipelines=']) {
    assert.equal(parseBootUrlOptions(q).dumpPipelines, false);
    assert.equal(parseBootUrlOptions(q).allowDumpWrites, false);
    assert.equal(parseBootUrlOptions(q).dumpEndpoint, null);
  }

  // allowDumpWrites=1 / true / yes / on enables allowDumpWrites and sets default '/__pg6_dump/' endpoint.
  for (const q of ['?allowDumpWrites=1', '?allowDumpWrites=true', '?allowDumpWrites=TRUE', '?allowdumpwrites=yes', '?allowDumpWrites=on']) {
    const parsed = parseBootUrlOptions(q);
    assert.equal(Object.isFrozen(parsed), true);
    assert.deepEqual(parsed, {
      inputScript: null,
      streamShaders: false,
      streamThresholdMb: null,
      streamThresholdBytes: null,
      dumpPipelines: false,
      allowDumpWrites: true,
      dumpEndpoint: '/__pg6_dump/',
    });
  }

  // allowDumpWrites=0 / false / invalid stays false with null endpoint.
  for (const q of ['?allowDumpWrites=0', '?allowDumpWrites=false', '?allowDumpWrites=no', '?allowDumpWrites=off', '?allowDumpWrites=nope', '?allowDumpWrites=']) {
    assert.equal(parseBootUrlOptions(q).allowDumpWrites, false);
    assert.equal(parseBootUrlOptions(q).dumpEndpoint, null);
  }

  // Combined query with all options active.
  assert.deepEqual(
    parseBootUrlOptions(
      '?inputScript=Menu-Flow&dumpPipelines=1&allowDumpWrites=true&streamShaders=1&streamThresholdMb=16'
    ),
    {
      inputScript: 'menu-flow',
      streamShaders: true,
      streamThresholdMb: 16,
      streamThresholdBytes: 16 * 1024 * 1024,
      dumpPipelines: true,
      allowDumpWrites: true,
      dumpEndpoint: '/__pg6_dump/',
    }
  );
});

test('W184 forwards dumpPipelines to loader.js and allowDumpWrites/dumpEndpoint to io_worker.js init config only when enabled', async () => {
  const { createRequire } = await import('node:module');
  const require = createRequire(import.meta.url);
  const { resolveDumpConfig } = require('../web/workers/io_worker.js');

  async function launchWithDumpUrl({ search, bootstrapOptions = {}, startOpts = undefined } = {}) {
    const clock = createClock(35000);
    const raf = createFakeRaf(clock);
    const timers = createFakeTimers();
    const webgpu = createMockWebGpuEnv();
    const hud = createMockHud();

    const offscreenCanvas = { kind: 'MockOffscreenCanvas' };
    webgpu.canvas.transferControlToOffscreen = () => offscreenCanvas;

    const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
      clock,
      raf,
      timers,
      gpu: webgpu.gpu,
    });
    if (search !== undefined) {
      windowObj.location = { origin: 'https://playgta6.local', search };
    }

    const workerCalls = [];
    const postedMessages = [];
    const mockWorker = createFakeEventTarget({
      postMessage(payload, transfer) {
        postedMessages.push({ payload, transfer });
      },
      terminate() {},
    });

    const app = bootstrapPlayGta6App({
      windowObj,
      documentObj,
      navigatorObj,
      fetchImpl: createMockFetch({
        ok: true,
        coopCoep: true,
        hasWasmBinary: true,
      }),
      canvas: webgpu.canvas,
      hud,
      now: clock.now,
      setInterval: timers.setInterval,
      clearInterval: timers.clearInterval,
      createWasmWorker(args) {
        workerCalls.push(args);
        return mockWorker;
      },
      ...bootstrapOptions,
    });

    try {
      const started = await app.start(startOpts);
      assert.equal(started.ok, true);
      assert.equal(started.mode, SESSION_MODE.WASM);
      assert.equal(workerCalls.length, 1);
      assert.equal(postedMessages.length, 1);
      return {
        bootUrlOptions: app.bootUrlOptions,
        workerCall: workerCalls[0],
        initPayload: postedMessages[0].payload,
        diagnostics: app.getDiagnosticsSnapshot(),
        telemetry: app.getBootTelemetry(),
        globalDiagnostics: windowObj.__playgta6.getDiagnosticsSnapshot(),
        globalBootState: windowObj.__playgta6BootState,
      };
    } finally {
      await app.destroy();
    }
  }

  // 1. Default (omitted): dumpPipelines/allowDumpWrites/dumpEndpoint absent from launch payloads.
  const defaultLaunch = await launchWithDumpUrl({ search: '' });
  assert.equal(defaultLaunch.bootUrlOptions.dumpPipelines, false);
  assert.equal(defaultLaunch.bootUrlOptions.allowDumpWrites, false);
  assert.equal(defaultLaunch.bootUrlOptions.dumpEndpoint, null);
  assert.equal('dumpPipelines' in defaultLaunch.workerCall, false);
  assert.equal('allowDumpWrites' in defaultLaunch.workerCall, false);
  assert.equal('dumpEndpoint' in defaultLaunch.workerCall, false);
  assert.equal('dumpPipelines' in defaultLaunch.initPayload, false);
  assert.equal('allowDumpWrites' in defaultLaunch.initPayload, false);
  assert.equal('dumpEndpoint' in defaultLaunch.initPayload, false);
  assert.deepEqual(resolveDumpConfig(defaultLaunch.initPayload), {
    allowDumpWrites: false,
    dumpEndpoint: null,
  });
  assert.equal(defaultLaunch.diagnostics.dumpPipelines, false);
  assert.equal(defaultLaunch.diagnostics.allowDumpWrites, false);
  assert.equal(defaultLaunch.diagnostics.inputScript, null);
  assert.equal(defaultLaunch.telemetry.dumpPipelines, false);
  assert.equal(defaultLaunch.telemetry.allowDumpWrites, false);
  assert.equal(defaultLaunch.telemetry.inputScript, null);

  // 2. ?dumpPipelines=1 alone forwards dumpPipelines: true without enabling allowDumpWrites.
  const dumpOnlyLaunch = await launchWithDumpUrl({ search: '?dumpPipelines=1' });
  assert.equal(dumpOnlyLaunch.bootUrlOptions.dumpPipelines, true);
  assert.equal(dumpOnlyLaunch.bootUrlOptions.allowDumpWrites, false);
  assert.equal(dumpOnlyLaunch.workerCall.dumpPipelines, true);
  assert.equal('allowDumpWrites' in dumpOnlyLaunch.workerCall, false);
  assert.equal('dumpEndpoint' in dumpOnlyLaunch.workerCall, false);
  assert.equal(dumpOnlyLaunch.initPayload.dumpPipelines, true);
  assert.equal('allowDumpWrites' in dumpOnlyLaunch.initPayload, false);
  assert.equal('dumpEndpoint' in dumpOnlyLaunch.initPayload, false);
  assert.deepEqual(resolveDumpConfig(dumpOnlyLaunch.initPayload), {
    allowDumpWrites: false,
    dumpEndpoint: null,
  });
  assert.equal(dumpOnlyLaunch.diagnostics.dumpPipelines, true);
  assert.equal(dumpOnlyLaunch.diagnostics.allowDumpWrites, false);
  assert.equal(dumpOnlyLaunch.telemetry.dumpPipelines, true);
  assert.equal(dumpOnlyLaunch.globalDiagnostics.dumpPipelines, true);
  assert.equal(dumpOnlyLaunch.globalBootState.dumpPipelines, true);

  // 3. ?allowDumpWrites=1 alone forwards allowDumpWrites: true + dumpEndpoint: '/__pg6_dump/' to io_worker config.
  const writeOnlyLaunch = await launchWithDumpUrl({ search: '?allowDumpWrites=1' });
  assert.equal(writeOnlyLaunch.bootUrlOptions.dumpPipelines, false);
  assert.equal(writeOnlyLaunch.bootUrlOptions.allowDumpWrites, true);
  assert.equal(writeOnlyLaunch.bootUrlOptions.dumpEndpoint, '/__pg6_dump/');
  assert.equal('dumpPipelines' in writeOnlyLaunch.workerCall, false);
  assert.equal(writeOnlyLaunch.workerCall.allowDumpWrites, true);
  assert.equal(writeOnlyLaunch.workerCall.dumpEndpoint, '/__pg6_dump/');
  assert.equal('dumpPipelines' in writeOnlyLaunch.initPayload, false);
  assert.equal(writeOnlyLaunch.initPayload.allowDumpWrites, true);
  assert.equal(writeOnlyLaunch.initPayload.dumpEndpoint, '/__pg6_dump/');
  assert.deepEqual(resolveDumpConfig(writeOnlyLaunch.initPayload), {
    allowDumpWrites: true,
    dumpEndpoint: '/__pg6_dump/',
  });
  assert.equal(writeOnlyLaunch.diagnostics.dumpPipelines, false);
  assert.equal(writeOnlyLaunch.diagnostics.allowDumpWrites, true);

  // 4. Both ?dumpPipelines=true&allowDumpWrites=true enabled together.
  const bothLaunch = await launchWithDumpUrl({
    search: '?dumpPipelines=true&allowDumpWrites=true&inputScript=intro-watch',
  });
  assert.equal(bothLaunch.workerCall.dumpPipelines, true);
  assert.equal(bothLaunch.workerCall.allowDumpWrites, true);
  assert.equal(bothLaunch.workerCall.dumpEndpoint, '/__pg6_dump/');
  assert.equal(bothLaunch.initPayload.dumpPipelines, true);
  assert.equal(bothLaunch.initPayload.allowDumpWrites, true);
  assert.equal(bothLaunch.initPayload.dumpEndpoint, '/__pg6_dump/');
  assert.deepEqual(resolveDumpConfig(bothLaunch.initPayload), {
    allowDumpWrites: true,
    dumpEndpoint: '/__pg6_dump/',
  });
  assert.equal(bothLaunch.diagnostics.dumpPipelines, true);
  assert.equal(bothLaunch.diagnostics.allowDumpWrites, true);
  assert.equal(bothLaunch.diagnostics.inputScript, 'intro-watch');
  assert.equal(bothLaunch.telemetry.dumpPipelines, true);
  assert.equal(bothLaunch.telemetry.allowDumpWrites, true);
  assert.equal(bothLaunch.telemetry.inputScript, 'intro-watch');

  // 5. Explicit programmatic overrides: false disables URL flags, unsafe dumpEndpoint falls back safely.
  const disabledLaunch = await launchWithDumpUrl({
    search: '?dumpPipelines=1&allowDumpWrites=1',
    bootstrapOptions: { dumpPipelines: false, allowDumpWrites: false },
  });
  assert.equal(disabledLaunch.bootUrlOptions.dumpPipelines, false);
  assert.equal(disabledLaunch.bootUrlOptions.allowDumpWrites, false);
  assert.equal('dumpPipelines' in disabledLaunch.initPayload, false);
  assert.equal('allowDumpWrites' in disabledLaunch.initPayload, false);
  assert.equal('dumpEndpoint' in disabledLaunch.initPayload, false);

  const sanitizedEndpointLaunch = await launchWithDumpUrl({
    search: '',
    startOpts: {
      dumpPipelines: true,
      allowDumpWrites: true,
      dumpEndpoint: 'C:\\Users\\alice\\secret_dump',
    },
  });
  assert.equal(sanitizedEndpointLaunch.initPayload.dumpPipelines, true);
  assert.equal(sanitizedEndpointLaunch.initPayload.allowDumpWrites, true);
  assert.equal(sanitizedEndpointLaunch.initPayload.dumpEndpoint, '/__pg6_dump/');
  assert.ok(!JSON.stringify(sanitizedEndpointLaunch.diagnostics).includes('alice'));
  assert.ok(!JSON.stringify(sanitizedEndpointLaunch.telemetry).includes('alice'));
});

test('W184 ?inputScript=menu-flow, calibration-skip, and intro-watch auto-start InputBridge presets and expose sanitized diagnostics', async () => {
  for (const preset of ['menu-flow', 'calibration-skip', 'intro-watch']) {
    const clock = createClock(40000);
    const raf = createFakeRaf(clock);
    const timers = createFakeTimers();
    const webgpu = createMockWebGpuEnv();
    const hud = createMockHud();

    const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
      clock,
      raf,
      timers,
      gpu: webgpu.gpu,
    });
    windowObj.location = {
      origin: 'https://playgta6.local',
      search: `?inputScript=${preset.toUpperCase()}&dumpPipelines=1`,
    };

    const mockWorker = createFakeEventTarget({
      postMessage() {},
      terminate() {},
    });

    const app = bootstrapPlayGta6App({
      windowObj,
      documentObj,
      navigatorObj,
      fetchImpl: createMockFetch({ ok: true, coopCoep: true, hasWasmBinary: true }),
      canvas: webgpu.canvas,
      hud,
      now: clock.now,
      raf: raf.raf,
      cancelRaf: raf.cancelRaf,
      setInterval: timers.setInterval,
      clearInterval: timers.clearInterval,
      createWasmWorker() {
        return mockWorker;
      },
    });

    try {
      assert.equal(app.bootUrlOptions.inputScript, preset);
      assert.equal(app.bootUrlOptions.dumpPipelines, true);
      assert.equal(app.getDiagnosticsSnapshot().inputScript, preset);
      assert.equal(app.getDiagnosticsSnapshot().dumpPipelines, true);
      assert.equal(app.getBootTelemetry().inputScript, preset);
      assert.equal(app.getBootTelemetry().dumpPipelines, true);

      const started = await app.start();
      assert.equal(started.ok, true);
      assert.equal(app.inputBridge.getScriptState().active, true);
      assert.equal(app.inputBridge.getScriptState().preset, preset);
      assert.equal(app.getDiagnosticsSnapshot().inputScript, preset);
      assert.equal(windowObj.__playgta6.getDiagnosticsSnapshot().inputScript, preset);
      assert.equal(windowObj.__playgta6BootState.inputScript, preset);
      assert.equal(globalThis.__playgta6Telemetry().inputScript, preset);

      // Per-start override cleanly switches preset and updates diagnostics.
      await app.stop();
      assert.equal(app.inputBridge.getScriptState().active, false);

      const switched = await app.start({ inputScript: 'start-screen', dumpPipelines: false });
      assert.equal(switched.ok, true);
      assert.equal(app.inputBridge.getScriptState().active, true);
      assert.equal(app.inputBridge.getScriptState().preset, 'start-screen');
      assert.equal(app.getDiagnosticsSnapshot().inputScript, 'start-screen');
      assert.equal(app.getDiagnosticsSnapshot().dumpPipelines, false);
    } finally {
      await app.destroy();
    }
    assert.equal(app.inputBridge.getScriptState().active, false);
  }

  // Hostile host paths in query params never leak into diagnostics or telemetry.
  const clock = createClock(41000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();
  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });
  windowObj.location = {
    origin: 'https://playgta6.local',
    search: '?inputScript=C%3A%5CUsers%5Calice%5Cmenu-flow&dumpPipelines=%2Fhome%2Falice%2Fdump&allowDumpWrites=C%3A%5Csecret',
  };

  const hostileApp = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch({ ok: true, coopCoep: true, hasWasmBinary: true }),
    canvas: webgpu.canvas,
    hud,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker() {
      return createFakeEventTarget({ postMessage() {}, terminate() {} });
    },
  });

  try {
    const diag = hostileApp.getDiagnosticsSnapshot();
    const telem = hostileApp.getBootTelemetry();
    assert.equal(diag.inputScript, null);
    assert.equal(diag.dumpPipelines, false);
    assert.equal(diag.allowDumpWrites, false);
    assert.equal(telem.inputScript, null);
    assert.equal(telem.dumpPipelines, false);
    assert.equal(telem.allowDumpWrites, false);
    assert.ok(!JSON.stringify(diag).includes('alice'));
    assert.ok(!JSON.stringify(telem).includes('alice'));
    assert.ok(!JSON.stringify(hostileApp.bootUrlOptions).includes('alice'));
  } finally {
    await hostileApp.destroy();
  }
});

test('W253 parseSplashHoldQuery and extractGpuAdapterFromLogLine parse and sanitize inputs safely', () => {
  for (const q of [
    '?splashHold=1',
    '?splashHold=true',
    '?SPLASHHOLD=YES',
    '?splashhold=on',
    'https://playgta6.local/?foo=1&splashHold=1#boot',
    new URLSearchParams('splashHold=1'),
    { search: '?splashHold=true' },
    '?splashHold=0&splashHold=1',
  ]) {
    assert.equal(parseSplashHoldQuery(q), true, `Expected true for ${String(q)}`);
  }

  for (const q of [
    '?splashHold=0',
    '?splashHold=false',
    '?SPLASHHOLD=OFF',
    '?splashhold=no',
    'https://playgta6.local/?splashHold=0#boot',
    new URLSearchParams('splashHold=0'),
    { search: '?splashHold=false' },
    '?splashHold=1&splashHold=0',
  ]) {
    assert.equal(parseSplashHoldQuery(q), false, `Expected false for ${String(q)}`);
  }

  for (const q of [undefined, null, '', '?', '?foo=bar', '?splashHold=maybe', '?splashHold=C:\\Users\\alice']) {
    assert.equal(parseSplashHoldQuery(q), null, `Expected null for ${String(q)}`);
  }

  assert.equal(
    extractGpuAdapterFromLogLine('[wgpu_worker] WebGPU adapter: NVIDIA / Turing'),
    'nvidia / turing'
  );
  assert.equal(
    extractGpuAdapterFromLogLine(
      '[wgpu_worker] WebGPU adapter: Intel / Gen-12LP (C:\\Users\\alice\\igd10iumd64.dll)'
    ),
    'intel / gen-12lp'
  );
  assert.equal(
    extractGpuAdapterFromLogLine(
      'WebGPU adapter: <script>alert(1)</script> AMD / RDNA2 https://evil.example/path'
    ),
    'amd / rdna2'
  );
  assert.equal(extractGpuAdapterFromLogLine('[wgpu_worker] no WebGPU adapter available'), null);
  assert.equal(extractGpuAdapterFromLogLine(null), null);
});

test('W253 smooth splash hold through FIRST_PRESENT -> black loading progress -> FIRST_VISIBLE_FRAME cross-fade', async () => {
  const { createRequire } = await import('node:module');
  const require = createRequire(import.meta.url);
  const { SplashOverlayController, DISMISS_REASONS } = require('../web/ui/splash_overlay.js');
  const { BootOrchestrator } = require('../web/runtime/boot_orchestrator.js');

  function makeMockNode(tag = 'div') {
    const attrs = new Map();
    const styles = new Map();
    const listeners = new Map();
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
      addEventListener(type, fn) {
        if (!listeners.has(type)) listeners.set(type, new Set());
        listeners.get(type).add(fn);
      },
      removeEventListener(type, fn) {
        listeners.get(type)?.delete(fn);
      },
      querySelector() {
        return null;
      },
    };
    return node;
  }

  const clock = createClock(1000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();

  let nextTimeoutId = 1;
  const pendingTimeouts = new Map();
  const fakeSetTimeout = (fn, ms) => {
    const id = nextTimeoutId++;
    pendingTimeouts.set(id, { fn, ms });
    return id;
  };
  const fakeClearTimeout = (id) => {
    pendingTimeouts.delete(id);
  };
  const flushFadeTimeouts = (expectedMs) => {
    for (const [id, entry] of [...pendingTimeouts.entries()]) {
      if (expectedMs === undefined || entry.ms === expectedMs) {
        pendingTimeouts.delete(id);
        entry.fn();
      }
    }
  };

  const splashContainer = makeMockNode('div');
  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });
  windowObj.location = { origin: 'https://playgta6.local', search: '?splashHold=1' };
  documentObj.head = makeMockNode('head');
  documentObj.createElement = (tag) => makeMockNode(tag);
  documentObj.querySelector = () => null;

  const mockWorker = createFakeEventTarget({
    postMessage() {},
    terminate() {},
  });

  const hudGpuUpdates = [];
  const hudController = {
    updateGpuStats(stats) {
      hudGpuUpdates.push(stats);
    },
  };

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch({
      ok: true,
      coopCoep: true,
      hasWasmBinary: true,
      pgimUrl: '/boot/guest.pgim',
      splashImage: '/game/xbs_splash.png',
    }),
    canvas: webgpu.canvas,
    hud,
    hudController,
    splashContainer,
    SplashOverlayCtor: SplashOverlayController,
    BootOrchestratorCtor: BootOrchestrator,
    pgimPreloader: {
      async preloadBootBundle() {
        return { pgim: new Uint8Array(4), shaderPackBytes: null, reflectionManifest: null, bytesLoaded: 4 };
      },
    },
    splashOverlayOptions: {
      visibleFrameFadeMs: 450,
      setTimeout: fakeSetTimeout,
      clearTimeout: fakeClearTimeout,
      nowFn: clock.now,
    },
    now: clock.now,
    raf: raf.raf,
    cancelRaf: raf.cancelRaf,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker() {
      return mockWorker;
    },
  });

  try {
    assert.equal(app.bootUrlOptions.splashHold, true);
    assert.equal(app.bootUrlOptions.holdUntilVisible, true);
    assert.equal(app.splashOverlay.getState().holdUntilVisible, true);
    assert.equal(app.splashOverlay.getState().visible, true);

    const started = await app.start();
    assert.equal(started.ok, true);
    assert.equal(started.mode, SESSION_MODE.WASM);
    assert.equal(app.splashOverlay.getState().splashImageUrl, '/game/xbs_splash.png');

    // 1. FIRST_PRESENT arrives during black archive-loading phase -> splash MUST stay visible & held!
    clock.advance(500);
    mockWorker.dispatch('message', {
      data: {
        type: 'splash-frame-presented',
        ok: true,
        frameIndex: 0,
        framesPresented: 1,
        firstPresent: true,
      },
    });

    let splashState = app.splashOverlay.getState();
    assert.equal(splashState.visible, true);
    assert.equal(splashState.dismissed, false);
    assert.equal(splashState.holdingForVisibleFrame, true);
    assert.equal(splashState.heldForVisibleFrame, true);
    assert.equal(app.bootTracer.getSummary().readyForSplash, true);
    assert.equal(app.bootTracer.getSummary().firstPresentMs, 500);
    assert.equal(app.bootTracer.getSummary().firstVisibleFrameMs, null);
    assert.equal(app.getBootTelemetry().loadingAssets, true);
    assert.equal(app.getDiagnosticsSnapshot().loadingAssets, true);

    // 2. Black loading frames update setLoadingAssetsProgress({ presents, realPackDraws }) without dismissing.
    clock.advance(1000);
    mockWorker.dispatch('message', {
      data: {
        type: 'gpu-stats',
        presentCount: 14,
        realPackDrawCount: 0,
        fallbackDrawCount: 0,
      },
    });
    splashState = app.splashOverlay.getState();
    assert.equal(splashState.visible, true);
    assert.equal(splashState.dismissed, false);
    assert.equal(splashState.holdingForVisibleFrame, true);
    assert.equal(splashState.blackPresentsWhileHeld, 14);
    assert.match(splashState.statusText, /Loading game assets\.\.\. \(14 frames\)/);

    // Explicit hasVisibleContent: false keeps holding even if realPackDrawCount > 0.
    clock.advance(500);
    mockWorker.dispatch('message', {
      data: {
        type: 'gpu-stats',
        presentCount: 20,
        realPackDrawCount: 3,
        hasVisibleContent: false,
      },
    });
    splashState = app.splashOverlay.getState();
    assert.equal(splashState.holdingForVisibleFrame, true);
    assert.equal(splashState.dismissed, false);
    assert.equal(splashState.blackPresentsWhileHeld, 20);
    assert.equal(splashState.loadingRealPackDraws, 3);
    assert.match(splashState.statusText, /3 draws, 20 frames/);

    // 3. First real visible frame arrives -> BootTracer records FIRST_VISIBLE_FRAME & splash smoothly cross-fades!
    clock.advance(2000);
    mockWorker.dispatch('message', {
      data: {
        type: 'gpu-stats',
        presentCount: 25,
        realPackDrawCount: 18,
        hasVisibleContent: true,
      },
    });

    const tracerSummary = app.bootTracer.getSummary();
    assert.equal(tracerSummary.stage, 'FIRST_VISIBLE_FRAME');
    assert.equal(tracerSummary.firstPresentMs, 500);
    assert.equal(tracerSummary.firstVisibleFrameMs, 4000);
    assert.equal(tracerSummary.blackLoadingMs, 3500);

    splashState = app.splashOverlay.getState();
    assert.equal(splashState.holdingForVisibleFrame, false);
    assert.equal(splashState.heldForVisibleFrame, true);
    assert.equal(splashState.dismissed, true);
    assert.equal(splashState.fading, true);

    // Complete the 450 ms smooth cross-fade.
    clock.advance(450);
    flushFadeTimeouts(450);
    splashState = app.splashOverlay.getState();
    assert.equal(splashState.visible, false);
    assert.equal(splashState.fading, false);
    assert.equal(splashState.dismissReason, DISMISS_REASONS.FIRST_VISIBLE_FRAME);

    const telem = app.getBootTelemetry();
    assert.equal(telem.hasVisibleContent, true);
    assert.equal(telem.loadingAssets, false);
    assert.equal(telem.firstVisibleFrameMs, 4000);
    assert.equal(telem.blackLoadingMs, 3500);
    assert.equal(telem.gpu.firstVisibleFrameMs, 4000);
    assert.equal(telem.gpu.blackLoadingMs, 3500);

    const lastHudGpu = hudGpuUpdates[hudGpuUpdates.length - 1];
    assert.equal(lastHudGpu.hasVisibleContent, true);
    assert.equal(lastHudGpu.loadingAssets, false);
    assert.equal(lastHudGpu.firstVisibleFrameMs, 4000);
    assert.equal(lastHudGpu.blackLoadingMs, 3500);

    // Subsequent notifyFirstVisibleFrame calls are idempotent.
    const dup = app.notifyFirstVisibleFrame({ source: 'duplicate' });
    assert.equal(dup.ok, true);
    assert.equal(dup.alreadyVisible, true);
  } finally {
    await app.destroy();
  }
});

test('W253 ?splashHold=0 opt-out and explicit holdUntilVisible overrides work cleanly', async () => {
  const { createRequire } = await import('node:module');
  const require = createRequire(import.meta.url);
  const { SplashOverlayController, DISMISS_REASONS } = require('../web/ui/splash_overlay.js');
  const { BootOrchestrator } = require('../web/runtime/boot_orchestrator.js');

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

  async function bootWithHoldConfig({ search, holdUntilVisible }) {
    const clock = createClock(5000);
    const raf = createFakeRaf(clock);
    const timers = createFakeTimers();
    const webgpu = createMockWebGpuEnv();
    const hud = createMockHud();
    const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
      clock,
      raf,
      timers,
      gpu: webgpu.gpu,
    });
    windowObj.location = { origin: 'https://playgta6.local', search };
    documentObj.head = makeMockNode('head');
    documentObj.createElement = (tag) => makeMockNode(tag);
    documentObj.querySelector = () => null;

    const mockWorker = createFakeEventTarget({
      postMessage() {},
      terminate() {},
    });

    const app = bootstrapPlayGta6App({
      windowObj,
      documentObj,
      navigatorObj,
      fetchImpl: createMockFetch({
        ok: true,
        coopCoep: true,
        hasWasmBinary: true,
        pgimUrl: '/boot/guest.pgim',
      }),
      canvas: webgpu.canvas,
      hud,
      splashContainer: makeMockNode('div'),
      SplashOverlayCtor: SplashOverlayController,
      BootOrchestratorCtor: BootOrchestrator,
      pgimPreloader: {
        async preloadBootBundle() {
          return { pgim: new Uint8Array(4), shaderPackBytes: null, reflectionManifest: null, bytesLoaded: 4 };
        },
      },
      splashOverlayOptions: { fadeMs: 0, visibleFrameFadeMs: 0, nowFn: clock.now },
      ...(holdUntilVisible !== undefined ? { holdUntilVisible } : {}),
      now: clock.now,
      setInterval: timers.setInterval,
      clearInterval: timers.clearInterval,
      createWasmWorker() {
        return mockWorker;
      },
    });

    await app.start();
    clock.advance(250);
    mockWorker.dispatch('message', {
      data: {
        type: 'first-present',
        framesPresented: 1,
      },
    });
    return { app, mockWorker, clock };
  }

  // 1. ?splashHold=0 opts out -> immediate dismiss on FIRST_PRESENT.
  const optOut = await bootWithHoldConfig({ search: '?splashHold=0' });
  try {
    assert.equal(optOut.app.bootUrlOptions.splashHold, false);
    assert.equal(optOut.app.bootUrlOptions.holdUntilVisible, false);
    const st = optOut.app.splashOverlay.getState();
    assert.equal(st.holdingForVisibleFrame, false);
    assert.equal(st.heldForVisibleFrame, false);
    assert.equal(st.dismissed, true);
    assert.equal(st.visible, false);
    assert.equal(st.dismissReason, DISMISS_REASONS.FIRST_PRESENT);

    // Explicit first-visible-frame worker message still records BootTracer timing cleanly.
    optOut.clock.advance(600);
    optOut.mockWorker.dispatch('message', {
      data: { type: 'first-visible-frame', framesPresented: 5 },
    });
    assert.equal(optOut.app.bootTracer.getSummary().firstVisibleFrameMs, 850);
    assert.equal(optOut.app.bootTracer.getSummary().blackLoadingMs, 600);
  } finally {
    await optOut.app.destroy();
  }

  // 2. Explicit holdUntilVisible: false overrides ?splashHold=1.
  const forcedOff = await bootWithHoldConfig({ search: '?splashHold=1', holdUntilVisible: false });
  try {
    assert.equal(forcedOff.app.bootUrlOptions.holdUntilVisible, false);
    assert.equal(forcedOff.app.splashOverlay.getState().dismissed, true);
    assert.equal(forcedOff.app.splashOverlay.getState().holdingForVisibleFrame, false);
  } finally {
    await forcedOff.app.destroy();
  }

  // 3. Explicit holdUntilVisible: true overrides ?splashHold=0 and holds until realPackDrawCount > 0.
  const forcedOn = await bootWithHoldConfig({ search: '?splashHold=0', holdUntilVisible: true });
  try {
    assert.equal(forcedOn.app.bootUrlOptions.holdUntilVisible, true);
    assert.equal(forcedOn.app.splashOverlay.getState().holdingForVisibleFrame, true);
    assert.equal(forcedOn.app.splashOverlay.getState().visible, true);

    forcedOn.clock.advance(750);
    forcedOn.mockWorker.dispatch('message', {
      data: { type: 'gpu-stats', presentCount: 9, realPackDrawCount: 6 },
    });
    assert.equal(forcedOn.app.splashOverlay.getState().dismissed, true);
    assert.equal(forcedOn.app.splashOverlay.getState().visible, false);
    assert.equal(forcedOn.app.splashOverlay.getState().dismissReason, DISMISS_REASONS.FIRST_VISIBLE_FRAME);
    assert.equal(forcedOn.app.bootTracer.getSummary().blackLoadingMs, 750);
  } finally {
    await forcedOn.app.destroy();
  }
});

test('W253 forwards sanitized WebGPU adapter labels from logs, telemetry, and smoke adapter to BootTracer and HudController', async () => {
  const { HudController } = await import('../web/ui/hud_controller.js');

  const clock = createClock(20000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();

  const broadcastInstances = [];
  class MockBroadcastChannel {
    constructor(name) {
      this.name = name;
      this.onmessage = null;
      broadcastInstances.push(this);
    }
    close() {}
  }

  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });
  windowObj.BroadcastChannel = MockBroadcastChannel;

  const mockWorker = createFakeEventTarget({
    postMessage() {},
    terminate() {},
  });

  const adapterSpyCalls = [];
  const hudController = new HudController({
    hud,
    now: clock.now,
  });
  const origSetGpuAdapter = hudController.setGpuAdapter.bind(hudController);
  hudController.setGpuAdapter = (label, opts) => {
    adapterSpyCalls.push(label);
    return origSetGpuAdapter(label, opts);
  };

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch({ ok: true, coopCoep: true, hasWasmBinary: true }),
    canvas: webgpu.canvas,
    hud,
    hudController,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker() {
      return mockWorker;
    },
  });

  try {
    await app.start();
    assert.equal(broadcastInstances.length, 1);

    // 1. Adapter reported via [wgpu_worker] log line over BroadcastChannel with host path noise.
    broadcastInstances[0].onmessage({
      data: {
        label: '[wgpu_worker] WebGPU adapter: NVIDIA / Turing (C:\\Users\\alice\\nvwgf2umx.dll)',
      },
    });

    assert.equal(app.bootTracer.gpuAdapter, 'nvidia / turing');
    assert.equal(app.bootTracer.getSummary().gpuAdapter, 'nvidia / turing');
    assert.equal(hudController.getGpuAdapter(), 'nvidia / turing');
    assert.deepEqual(adapterSpyCalls, ['nvidia / turing']);
    assert.equal(app.getBootTelemetry().gpuAdapter, 'nvidia / turing');
    assert.equal(app.getBootTelemetry().gpu.gpuAdapter, 'nvidia / turing');
    assert.equal(app.getDiagnosticsSnapshot().gpuAdapter, 'nvidia / turing');
    assert.ok(!JSON.stringify(app.getBootTelemetry()).includes('alice'));
    assert.ok(!JSON.stringify(app.bootTracer.getSummary()).includes('alice'));

    // Duplicate identical adapter line does not spam setGpuAdapter.
    broadcastInstances[0].onmessage({
      data: {
        label: '[wgpu_worker] WebGPU adapter: nvidia / turing',
      },
    });
    assert.equal(adapterSpyCalls.length, 1);

    // 2. Adapter updated via worker gpu-stats object payload `{ vendor, architecture }`.
    mockWorker.dispatch('message', {
      data: {
        type: 'gpu-stats',
        presentCount: 2,
        realPackDrawCount: 0,
        adapterInfo: { vendor: 'Intel', architecture: 'Gen-12LP' },
      },
    });
    assert.equal(app.bootTracer.gpuAdapter, 'intel / gen-12lp');
    assert.equal(hudController.getGpuAdapter(), 'intel / gen-12lp');
    assert.deepEqual(adapterSpyCalls, ['nvidia / turing', 'intel / gen-12lp']);

    // 3. Explicit [boot] FIRST_VISIBLE_FRAME log line records FIRST_VISIBLE_FRAME even without gpu-stats draws.
    clock.advance(1200);
    broadcastInstances[0].onmessage({
      data: { label: '[boot] FIRST_VISIBLE_FRAME rendered' },
    });
    assert.equal(app.bootTracer.getSummary().stage, 'FIRST_VISIBLE_FRAME');
    assert.equal(app.getBootTelemetry().hasVisibleContent, true);
  } finally {
    hudController.destroy();
    await app.destroy();
  }

  // 4. Smoke demo adapter.info is also forwarded cleanly.
  const smokeClock = createClock(30000);
  const smokeRaf = createFakeRaf(smokeClock);
  const smokeTimers = createFakeTimers();
  const smokeWebgpu = createMockWebGpuEnv();
  smokeWebgpu.adapter.info = { vendor: 'Apple', architecture: 'Metal-3' };
  const smokeHost = createMockBrowserHost({
    clock: smokeClock,
    raf: smokeRaf,
    timers: smokeTimers,
    gpu: smokeWebgpu.gpu,
  });
  const smokeAdapterCalls = [];
  const smokeApp = bootstrapPlayGta6App({
    windowObj: smokeHost.windowObj,
    documentObj: smokeHost.documentObj,
    navigatorObj: smokeHost.navigatorObj,
    fetchImpl: createMockFetch({ ok: true, hasWasmBinary: false }),
    canvas: smokeWebgpu.canvas,
    hud: createMockHud(),
    hudController: {
      setGpuAdapter(label) {
        smokeAdapterCalls.push(label);
      },
    },
    now: smokeClock.now,
    raf: smokeRaf.raf,
    cancelRaf: smokeRaf.cancelRaf,
    setInterval: smokeTimers.setInterval,
    clearInterval: smokeTimers.clearInterval,
  });
  try {
    const res = await smokeApp.start();
    assert.equal(res.ok, true);
    assert.equal(smokeApp.bootTracer.gpuAdapter, 'apple / metal-3');
    assert.deepEqual(smokeAdapterCalls, ['apple / metal-3']);
  } finally {
    await smokeApp.destroy();
  }
});

test('W261 defaultHoldUntilVisible holds splash until W257 first-visible-frame feeds blackLoadingMs + gpuAdapter, and ?splashHold=0 overrides it cleanly', async () => {
  const fs = await import('node:fs');
  const { createRequire } = await import('node:module');
  const require = createRequire(import.meta.url);
  const { SplashOverlayController, DISMISS_REASONS } = require('../web/ui/splash_overlay.js');
  const { BootOrchestrator } = require('../web/runtime/boot_orchestrator.js');
  const { HudController } = await import('../web/ui/hud_controller.js');

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

  // Verify web/index.html enables smooth Astryx splash hold by default in real browser boots.
  const indexHtmlPath = new URL('../web/index.html', import.meta.url);
  const indexHtml = fs.readFileSync(indexHtmlPath, 'utf8');
  assert.match(indexHtml, /bootstrapPlayGta6App\(\{[\s\S]*defaultHoldUntilVisible:\s*true/);

  // 1. defaultHoldUntilVisible: true holds splash on FIRST_PRESENT until W257 first-visible-frame arrives.
  const clock = createClock(1000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();
  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });
  windowObj.location = { origin: 'http://127.0.0.1:8080', search: '' };
  documentObj.head = makeMockNode('head');
  documentObj.createElement = (tag) => makeMockNode(tag);
  documentObj.querySelector = () => null;

  const mockWorker = createFakeEventTarget({
    postMessage() {},
    terminate() {},
  });

  const hudController = new HudController({ hud, now: clock.now });
  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch({
      ok: true,
      coopCoep: true,
      hasWasmBinary: true,
      pgimUrl: '/boot/guest.pgim',
      splashImage: '/game/xbs_splash.png',
    }),
    canvas: webgpu.canvas,
    hud,
    hudController,
    splashContainer: makeMockNode('div'),
    SplashOverlayCtor: SplashOverlayController,
    BootOrchestratorCtor: BootOrchestrator,
    pgimPreloader: {
      async preloadBootBundle() {
        return { pgim: new Uint8Array(4), shaderPackBytes: null, reflectionManifest: null, bytesLoaded: 4 };
      },
    },
    splashOverlayOptions: { fadeMs: 0, visibleFrameFadeMs: 0, nowFn: clock.now },
    defaultHoldUntilVisible: true,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker() {
      return mockWorker;
    },
  });

  try {
    assert.equal(app.bootUrlOptions.splashHold, null);
    assert.equal(app.bootUrlOptions.holdUntilVisible, true);
    assert.equal(app.splashOverlay.getState().holdUntilVisible, true);

    const started = await app.start();
    assert.equal(started.ok, true);

    // FIRST_PRESENT arrives -> splash stays visible & held waiting for first-visible-frame.
    clock.advance(500);
    mockWorker.dispatch('message', {
      data: { type: 'first-present', framesPresented: 1 },
    });
    assert.equal(app.splashOverlay.getState().holdingForVisibleFrame, true);
    assert.equal(app.splashOverlay.getState().visible, true);
    assert.equal(app.splashOverlay.getState().dismissed, false);

    // W257 first-visible-frame arrives with blackLoadingMs & raw gpuAdapter -> updates all 3 targets.
    mockWorker.dispatch('message', {
      data: {
        type: 'first-visible-frame',
        firstVisibleFrame: true,
        firstVisibleFrameMs: 4200,
        firstPresentMs: 500,
        blackLoadingMs: 3700,
        gpuAdapter: 'NVIDIA / Turing (C:\\Users\\alice\\nvwgf2umx.dll)',
        framesPresented: 18,
        realPackDrawCount: 4,
      },
    });

    const summary = app.bootTracer.getSummary();
    assert.equal(summary.stage, 'FIRST_VISIBLE_FRAME');
    assert.equal(summary.firstPresentMs, 500);
    assert.equal(summary.firstVisibleFrameMs, 4200);
    assert.equal(summary.blackLoadingMs, 3700);
    assert.equal(summary.gpuAdapter, 'nvidia / turing');

    const splashState = app.splashOverlay.getState();
    assert.equal(splashState.holdingForVisibleFrame, false);
    assert.equal(splashState.dismissed, true);
    assert.equal(splashState.visible, false);
    assert.equal(splashState.dismissReason, DISMISS_REASONS.FIRST_VISIBLE_FRAME);

    assert.equal(hudController.getGpuAdapter(), 'nvidia / turing');
    assert.equal(hudController.getTelemetry().gpu.blackLoadingMs, 3700);
    assert.equal(app.getBootTelemetry().blackLoadingMs, 3700);
    assert.equal(app.getBootTelemetry().gpuAdapter, 'nvidia / turing');
    assert.ok(!JSON.stringify(app.getBootTelemetry()).includes('alice'));
  } finally {
    hudController.destroy();
    await app.destroy();
  }

  // 2. ?splashHold=0 cleanly overrides defaultHoldUntilVisible: true.
  const optOutClock = createClock(2000);
  const optOutRaf = createFakeRaf(optOutClock);
  const optOutTimers = createFakeTimers();
  const optOutWebgpu = createMockWebGpuEnv();
  const optOutHost = createMockBrowserHost({
    clock: optOutClock,
    raf: optOutRaf,
    timers: optOutTimers,
    gpu: optOutWebgpu.gpu,
  });
  optOutHost.windowObj.location = { origin: 'http://127.0.0.1:8080', search: '?splashHold=0' };
  optOutHost.documentObj.head = makeMockNode('head');
  optOutHost.documentObj.createElement = (tag) => makeMockNode(tag);
  optOutHost.documentObj.querySelector = () => null;
  const optOutWorker = createFakeEventTarget({ postMessage() {}, terminate() {} });

  const optOutApp = bootstrapPlayGta6App({
    windowObj: optOutHost.windowObj,
    documentObj: optOutHost.documentObj,
    navigatorObj: optOutHost.navigatorObj,
    fetchImpl: createMockFetch({
      ok: true,
      coopCoep: true,
      hasWasmBinary: true,
      pgimUrl: '/boot/guest.pgim',
    }),
    canvas: optOutWebgpu.canvas,
    hud: createMockHud(),
    splashContainer: makeMockNode('div'),
    SplashOverlayCtor: SplashOverlayController,
    BootOrchestratorCtor: BootOrchestrator,
    pgimPreloader: {
      async preloadBootBundle() {
        return { pgim: new Uint8Array(4), shaderPackBytes: null, reflectionManifest: null, bytesLoaded: 4 };
      },
    },
    splashOverlayOptions: { fadeMs: 0, visibleFrameFadeMs: 0, nowFn: optOutClock.now },
    defaultHoldUntilVisible: true,
    now: optOutClock.now,
    setInterval: optOutTimers.setInterval,
    clearInterval: optOutTimers.clearInterval,
    createWasmWorker() {
      return optOutWorker;
    },
  });

  try {
    assert.equal(optOutApp.bootUrlOptions.splashHold, false);
    assert.equal(optOutApp.bootUrlOptions.holdUntilVisible, false);
    await optOutApp.start();
    optOutClock.advance(200);
    optOutWorker.dispatch('message', { data: { type: 'first-present', framesPresented: 1 } });
    const st = optOutApp.splashOverlay.getState();
    assert.equal(st.holdingForVisibleFrame, false);
    assert.equal(st.dismissed, true);
    assert.equal(st.visible, false);
    assert.equal(st.dismissReason, DISMISS_REASONS.FIRST_PRESENT);
  } finally {
    await optOutApp.destroy();
  }

  // 3. Smoke demo under defaultHoldUntilVisible: true transitions past hold cleanly without hanging.
  const smokeClock = createClock(3000);
  const smokeRaf = createFakeRaf(smokeClock);
  const smokeTimers = createFakeTimers();
  const smokeWebgpu = createMockWebGpuEnv();
  const smokeHost = createMockBrowserHost({
    clock: smokeClock,
    raf: smokeRaf,
    timers: smokeTimers,
    gpu: smokeWebgpu.gpu,
  });
  smokeHost.windowObj.location = { origin: 'http://127.0.0.1:8080', search: '' };
  smokeHost.documentObj.head = makeMockNode('head');
  smokeHost.documentObj.createElement = (tag) => makeMockNode(tag);
  smokeHost.documentObj.querySelector = () => null;

  const smokeApp = bootstrapPlayGta6App({
    windowObj: smokeHost.windowObj,
    documentObj: smokeHost.documentObj,
    navigatorObj: smokeHost.navigatorObj,
    fetchImpl: createMockFetch({ ok: true, coopCoep: true, hasWasmBinary: false }),
    canvas: smokeWebgpu.canvas,
    hud: createMockHud(),
    splashContainer: makeMockNode('div'),
    SplashOverlayCtor: SplashOverlayController,
    BootOrchestratorCtor: BootOrchestrator,
    splashOverlayOptions: { fadeMs: 0, visibleFrameFadeMs: 0, nowFn: smokeClock.now },
    defaultHoldUntilVisible: true,
    now: smokeClock.now,
    raf: smokeRaf.raf,
    cancelRaf: smokeRaf.cancelRaf,
    setInterval: smokeTimers.setInterval,
    clearInterval: smokeTimers.clearInterval,
  });

  try {
    const res = await smokeApp.start();
    assert.equal(res.ok, true);
    assert.equal(smokeApp.splashOverlay.getState().holdingForVisibleFrame, false);
    assert.equal(smokeApp.splashOverlay.getState().dismissed, true);
    assert.equal(smokeApp.splashOverlay.getState().visible, false);
  } finally {
    await smokeApp.destroy();
  }
});

test('W261 first-user-gesture (pointerdown / keydown) unlocks audioEngine once and removes listeners cleanly on destroy()', async () => {
  function createTrackingTarget(extra = {}) {
    const listeners = new Map();
    const registrations = [];
    return {
      ...extra,
      registrations,
      addEventListener(type, fn, opts) {
        registrations.push({ type, opts: opts ? { ...opts } : null });
        if (!listeners.has(type)) {
          listeners.set(type, new Set());
        }
        listeners.get(type).add({ fn, once: Boolean(opts && opts.once) });
      },
      removeEventListener(type, fn) {
        const set = listeners.get(type);
        if (!set) return;
        for (const entry of set) {
          if (entry.fn === fn) {
            set.delete(entry);
          }
        }
      },
      dispatch(type, event = {}) {
        const set = listeners.get(type);
        if (!set) return;
        for (const entry of [...set]) {
          if (entry.once) {
            set.delete(entry);
          }
          entry.fn({ type, ...event });
        }
      },
      listenerCount(type) {
        if (type) {
          return listeners.get(type)?.size ?? 0;
        }
        let total = 0;
        for (const set of listeners.values()) {
          total += set.size;
        }
        return total;
      },
    };
  }

  // 1. pointerdown calls unlockOnUserGesture() once with { passive: true, once: true } and disarms keydown too.
  {
    const clock = createClock(1000);
    const raf = createFakeRaf(clock);
    const timers = createFakeTimers();
    const webgpu = createMockWebGpuEnv();
    const windowObj = createTrackingTarget({
      crossOriginIsolated: true,
      innerWidth: 800,
      innerHeight: 600,
      devicePixelRatio: 1,
      location: { origin: 'http://127.0.0.1:8080', search: '' },
      localStorage: createMemoryStorage(),
      requestAnimationFrame: raf.raf,
      cancelAnimationFrame: raf.cancelRaf,
      setInterval: timers.setInterval,
      clearInterval: timers.clearInterval,
    });
    const documentObj = createTrackingTarget({ pointerLockElement: null });
    let unlockCalls = 0;
    let resumeCalls = 0;
    const mockAudioEngine = {
      sab: null,
      state: 'suspended',
      async unlockOnUserGesture() {
        unlockCalls += 1;
        this.state = 'running';
        return true;
      },
      async resume() {
        resumeCalls += 1;
        return true;
      },
      async destroy() {
        this.state = 'closed';
      },
    };

    const app = bootstrapPlayGta6App({
      windowObj,
      documentObj,
      navigatorObj: { gpu: webgpu.gpu, getGamepads: () => [] },
      fetchImpl: createMockFetch({ ok: true, hasWasmBinary: false }),
      canvas: webgpu.canvas,
      hud: createMockHud(),
      audioEngine: mockAudioEngine,
      now: clock.now,
      setInterval: timers.setInterval,
      clearInterval: timers.clearInterval,
    });

    try {
      const pointerReg = windowObj.registrations.find(
        (r) => r.type === 'pointerdown' && r.opts && r.opts.once === true
      );
      const keyReg = windowObj.registrations.find(
        (r) => r.type === 'keydown' && r.opts && r.opts.once === true
      );
      assert.deepEqual(pointerReg?.opts, { passive: true, once: true });
      assert.deepEqual(keyReg?.opts, { passive: true, once: true });

      const keydownBefore = windowObj.listenerCount('keydown');
      assert.equal(windowObj.listenerCount('pointerdown'), 1);

      windowObj.dispatch('pointerdown');
      assert.equal(unlockCalls, 1);
      assert.equal(resumeCalls, 0);
      assert.equal(windowObj.listenerCount('pointerdown'), 0);
      assert.equal(windowObj.listenerCount('keydown'), keydownBefore - 1);

      // Subsequent gestures must not re-invoke unlock.
      windowObj.dispatch('pointerdown');
      windowObj.dispatch('keydown', { code: 'KeyW' });
      assert.equal(unlockCalls, 1);
      assert.equal(resumeCalls, 0);
    } finally {
      await app.destroy();
    }
    assert.equal(windowObj.listenerCount(), 0);
    assert.equal(documentObj.listenerCount(), 0);
  }

  // 2. keydown falls back to audioEngine.resume() once when unlockOnUserGesture/unlockOnGesture are absent.
  {
    const clock = createClock(2000);
    const raf = createFakeRaf(clock);
    const timers = createFakeTimers();
    const webgpu = createMockWebGpuEnv();
    const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
      clock,
      raf,
      timers,
      gpu: webgpu.gpu,
    });
    let resumeCalls = 0;
    const resumeOnlyAudio = {
      sab: null,
      state: 'suspended',
      async resume() {
        resumeCalls += 1;
        this.state = 'running';
        return true;
      },
      async destroy() {
        this.state = 'closed';
      },
    };

    const app = bootstrapPlayGta6App({
      windowObj,
      documentObj,
      navigatorObj,
      fetchImpl: createMockFetch({ ok: true, hasWasmBinary: false }),
      canvas: webgpu.canvas,
      hud: createMockHud(),
      audioEngine: resumeOnlyAudio,
      now: clock.now,
      setInterval: timers.setInterval,
      clearInterval: timers.clearInterval,
    });

    try {
      assert.equal(windowObj.listenerCount('pointerdown'), 1);
      windowObj.dispatch('keydown', { code: 'Space' });
      assert.equal(resumeCalls, 1);
      assert.equal(windowObj.listenerCount('pointerdown'), 0);

      // Second gesture is a no-op.
      windowObj.dispatch('keydown', { code: 'KeyE' });
      windowObj.dispatch('pointerdown');
      assert.equal(resumeCalls, 1);
    } finally {
      await app.destroy();
    }
    assert.equal(windowObj.listenerCount(), 0);
    assert.equal(documentObj.listenerCount(), 0);
  }

  // 3. Real AudioEngine with mocked AudioContext + AudioWorkletNode lazily starts and unlocks on first pointerdown.
  {
    const clock = createClock(3000);
    const raf = createFakeRaf(clock);
    const timers = createFakeTimers();
    const webgpu = createMockWebGpuEnv();
    let resumeCount = 0;
    let addModuleCount = 0;
    class MockAudioContext {
      constructor() {
        this.state = 'suspended';
        this.sampleRate = 48000;
        this.currentTime = 0;
        this.destination = { kind: 'destination' };
        this.audioWorklet = {
          async addModule() {
            addModuleCount += 1;
          },
        };
      }
      createGain() {
        return {
          gain: {
            value: 1,
            setValueAtTime(v) {
              this.value = v;
            },
            linearRampToValueAtTime(v) {
              this.value = v;
            },
            cancelScheduledValues() {},
          },
          connect() {},
          disconnect() {},
        };
      }
      async resume() {
        resumeCount += 1;
        this.state = 'running';
      }
      async close() {
        this.state = 'closed';
      }
    }
    class MockAudioWorkletNode {
      constructor() {
        this.port = { postMessage() {}, close() {} };
      }
      connect() {}
      disconnect() {}
    }

    const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
      clock,
      raf,
      timers,
      gpu: webgpu.gpu,
    });
    windowObj.AudioContext = MockAudioContext;
    windowObj.AudioWorkletNode = MockAudioWorkletNode;

    const app = bootstrapPlayGta6App({
      windowObj,
      documentObj,
      navigatorObj,
      fetchImpl: createMockFetch({ ok: true, hasWasmBinary: false }),
      canvas: webgpu.canvas,
      hud: createMockHud(),
      now: clock.now,
      setInterval: timers.setInterval,
      clearInterval: timers.clearInterval,
    });

    try {
      assert.equal(app.audioEngine.state, 'idle');
      windowObj.dispatch('pointerdown');
      // Let lazy AudioEngine.start() + unlockOnGesture() promises settle cleanly.
      await new Promise((resolve) => setTimeout(resolve, 0));
      assert.equal(addModuleCount, 1);
      assert.equal(resumeCount, 1);
      assert.equal(app.audioEngine.state, 'running');
      assert.equal(app.audioEngine.getStats().unlockedOnGesture, true);
    } finally {
      await app.destroy();
    }
    assert.equal(windowObj.listenerCount(), 0);
    assert.equal(documentObj.listenerCount(), 0);
  }

  // 4. destroy() before any user gesture removes pointerdown/keydown listeners cleanly and prevents late unlock.
  {
    const clock = createClock(4000);
    const raf = createFakeRaf(clock);
    const timers = createFakeTimers();
    const webgpu = createMockWebGpuEnv();
    const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
      clock,
      raf,
      timers,
      gpu: webgpu.gpu,
    });
    let unlockCalls = 0;
    const mockAudioEngine = {
      sab: null,
      state: 'suspended',
      async unlockOnUserGesture() {
        unlockCalls += 1;
        return true;
      },
      async destroy() {
        this.state = 'closed';
      },
    };

    const app = bootstrapPlayGta6App({
      windowObj,
      documentObj,
      navigatorObj,
      fetchImpl: createMockFetch({ ok: true, hasWasmBinary: false }),
      canvas: webgpu.canvas,
      hud: createMockHud(),
      audioEngine: mockAudioEngine,
      now: clock.now,
      setInterval: timers.setInterval,
      clearInterval: timers.clearInterval,
    });

    assert.equal(windowObj.listenerCount('pointerdown'), 1);
    await app.destroy();
    assert.equal(windowObj.listenerCount(), 0);
    assert.equal(documentObj.listenerCount(), 0);

    windowObj.dispatch('pointerdown');
    windowObj.dispatch('keydown', { code: 'Enter' });
    assert.equal(unlockCalls, 0);
  }
});

test('W333 W308 lazy _playgta6_audio_ring() auto-retry with audioWorker exposes audioRingAttached and audioRingAttachAttempts', async () => {
  const { createRequire } = await import('node:module');
  const require = createRequire(import.meta.url);
  const loaderApi = require('../web/loader.js');

  function createFakeTimeoutScheduler() {
    let nextId = 1;
    const pending = new Map();
    return {
      setTimeout(fn, ms) {
        const id = nextId++;
        pending.set(id, { fn, ms });
        return id;
      },
      clearTimeout(id) {
        pending.delete(id);
      },
      fireNext() {
        const first = pending.entries().next();
        if (first.done) return false;
        const [id, entry] = first.value;
        pending.delete(id);
        entry.fn();
        return true;
      },
      pendingCount() {
        return pending.size;
      },
    };
  }

  // Build a valid Wasm64 Playgta6AudioRing struct inside a SharedArrayBuffer.
  function makeValidWasm64AudioRingSab() {
    const sab = new SharedArrayBuffer(65536);
    const view = new DataView(sab);
    const structOffset = 64;
    const baseOffset = 4096;
    const readIdxOffset = 128;
    const writeIdxOffset = 132;
    view.setBigUint64(structOffset + 0, BigInt(baseOffset), true);
    view.setUint32(structOffset + 8, 2048, true); // capacity (power of 2)
    view.setUint32(structOffset + 12, 2, true); // channels (stereo)
    view.setUint32(structOffset + 16, 48000, true); // sampleRate
    view.setBigUint64(structOffset + 24, BigInt(readIdxOffset), true);
    view.setBigUint64(structOffset + 32, BigInt(writeIdxOffset), true);
    return { sab, structOffset };
  }

  // Case 1: Real PlayGta6Loader.scheduleAudioRingRetry retries null_ptr twice, then attaches on 3rd attempt.
  {
    const clock = createClock(1000);
    const raf = createFakeRaf(clock);
    const timers = createFakeTimers();
    const timeouts = createFakeTimeoutScheduler();
    const webgpu = createMockWebGpuEnv();
    const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
      clock,
      raf,
      timers,
      gpu: webgpu.gpu,
    });

    const { sab, structOffset } = makeValidWasm64AudioRingSab();
    let ptrCalls = 0;
    const mockWasmModule = {
      wasmMemory: { buffer: sab },
      _playgta6_audio_ring() {
        ptrCalls += 1;
        return ptrCalls >= 3 ? BigInt(structOffset) : 0n;
      },
    };

    const audioPosted = [];
    const audioWorker = createFakeEventTarget({
      postMessage(msg) {
        audioPosted.push(msg);
      },
    });
    const loaderPosted = [];
    const mockWorker = createFakeEventTarget({
      postMessage(msg) {
        loaderPosted.push(msg);
      },
      terminate() {},
    });

    const app = bootstrapPlayGta6App({
      windowObj,
      documentObj,
      navigatorObj,
      fetchImpl: createMockFetch({ ok: true, coopCoep: true, hasWasmBinary: true }),
      canvas: webgpu.canvas,
      hud: createMockHud(),
      audioWorker,
      wasmModule: mockWasmModule,
      scheduleAudioRingRetry: loaderApi.scheduleAudioRingRetry,
      audioRingRetryIntervalMs: 125,
      audioRingMaxRetries: 10,
      audioRingAutoRetry: true,
      setTimeout: timeouts.setTimeout,
      clearTimeout: timeouts.clearTimeout,
      now: clock.now,
      setInterval: timers.setInterval,
      clearInterval: timers.clearInterval,
      createWasmWorker() {
        return mockWorker;
      },
    });

    try {
      assert.equal(app.getBootTelemetry().audioRingAttached, false);
      assert.equal(app.getBootTelemetry().audioRingAttachAttempts, 0);

      const started = await app.start();
      assert.equal(started.ok, true);

      // Explicit W308 retry options are forwarded in initPayload.
      assert.equal(loaderPosted[0].audioRingRetryIntervalMs, 125);
      assert.equal(loaderPosted[0].audioRingMaxRetries, 10);
      assert.equal(loaderPosted[0].audioRingAutoRetry, true);

      // Attempt 1 returned 0n -> retry timer scheduled.
      assert.equal(app.getBootTelemetry().audioRingAttached, false);
      assert.equal(app.getBootTelemetry().audioRingAttachAttempts, 1);
      assert.equal(timeouts.pendingCount(), 1);

      // Attempt 2 returned 0n -> still retrying.
      assert.equal(timeouts.fireNext(), true);
      assert.equal(app.getBootTelemetry().audioRingAttached, false);
      assert.equal(app.getBootTelemetry().audioRingAttachAttempts, 2);
      assert.equal(timeouts.pendingCount(), 1);

      // Attempt 3 returns valid Wasm64 struct -> attaches to audioWorker & clears timer.
      assert.equal(timeouts.fireNext(), true);
      assert.equal(audioPosted.length, 1);
      assert.equal(audioPosted[0].type, 'attach-guest-pcm-ring');
      assert.equal(audioPosted[0].sampleRate, 48000);
      assert.equal(audioPosted[0].channels, 2);
      assert.equal(timeouts.pendingCount(), 0);

      const snap = app.getBootTelemetry();
      assert.equal(snap.audioRingAttached, true);
      assert.equal(snap.audioRingAttachAttempts, 3);
      assert.deepEqual(Object.keys(snap.audio).sort(), [
        'framesDrained',
        'outputState',
        'overruns',
        'underruns',
      ]);

      const diag = app.getDiagnosticsSnapshot();
      assert.equal(diag.audioRingAttached, true);
      assert.equal(diag.audioRingAttachAttempts, 3);
    } finally {
      await app.destroy();
    }
    assert.equal(timeouts.pendingCount(), 0);
    assert.equal(audioWorker.listenerCount(), 0);
  }

  // Case 2: Fallback built-in retry + deferred audioWorker ready + loader-status / log ingestion.
  {
    const clock = createClock(2000);
    const raf = createFakeRaf(clock);
    const timers = createFakeTimers();
    const timeouts = createFakeTimeoutScheduler();
    const webgpu = createMockWebGpuEnv();
    const broadcastInstances = [];
    class MockBroadcastChannel {
      constructor(name) {
        this.name = name;
        this.onmessage = null;
        broadcastInstances.push(this);
      }
      close() {}
    }
    const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
      clock,
      raf,
      timers,
      gpu: webgpu.gpu,
    });
    windowObj.BroadcastChannel = MockBroadcastChannel;

    const audioPosted = [];
    const audioWorker = createFakeEventTarget({
      postMessage(msg) {
        audioPosted.push(msg);
      },
    });
    const mockWorker = createFakeEventTarget({
      postMessage() {},
      terminate() {},
    });

    const app = bootstrapPlayGta6App({
      windowObj,
      documentObj,
      navigatorObj,
      fetchImpl: createMockFetch({ ok: true, coopCoep: true, hasWasmBinary: true }),
      canvas: webgpu.canvas,
      hud: createMockHud(),
      audioWorker,
      audioWorkerReady: false,
      scheduleAudioRingRetry: null,
      attachRuntimeAudioRing: null,
      wasmModule: {
        _playgta6_audio_ring() {
          return 0x1000n;
        },
      },
      setTimeout: timeouts.setTimeout,
      clearTimeout: timeouts.clearTimeout,
      now: clock.now,
      setInterval: timers.setInterval,
      clearInterval: timers.clearInterval,
      createWasmWorker() {
        return mockWorker;
      },
    });

    try {
      await app.start();
      // Initial probe saw audioWorkerReady=false -> scheduled retry.
      assert.equal(app.getBootTelemetry().audioRingAttached, false);
      assert.equal(app.getBootTelemetry().audioRingAttachAttempts, 1);

      // audioWorker emits { ready: true } -> immediately probes and attaches!
      audioWorker.dispatch('message', { data: { ready: true } });
      assert.equal(audioPosted.length, 1);
      assert.equal(audioPosted[0].attachGuestPcmRing, true);
      assert.equal(app.getBootTelemetry().audioRingAttached, true);
      assert.equal(app.getBootTelemetry().audioRingAttachAttempts, 2);

      // Also verify loader-status & BroadcastChannel [audio] log ingestion preserve higher attempt counts.
      mockWorker.dispatch('message', {
        data: {
          type: 'loader-status',
          audioRingAttached: true,
          audioRingRetryCount: 4,
          audioRingLastReason: 'ok',
        },
      });
      assert.equal(app.getBootTelemetry().audioRingAttachAttempts, 5);

      broadcastInstances[0].onmessage({
        data: {
          label: '[audio] audio_ring_attached=true sampleRate=48000 channels=2 capacity=131072 retries=6',
        },
      });
      assert.equal(app.getBootTelemetry().audioRingAttachAttempts, 7);
    } finally {
      await app.destroy();
    }
    assert.equal(timeouts.pendingCount(), 0);
  }
});

test('W333 passes through W309 BootOrchestrator concurrent preload options, W310 PgimPreloader OPFS cache options, and W311 SplashOverlay monotonic phase progress + 220ms fadeOutMs', async () => {
  const { createRequire } = await import('node:module');
  const require = createRequire(import.meta.url);
  const { BootOrchestrator } = require('../web/runtime/boot_orchestrator.js');
  const { PgimPreloader } = require('../web/runtime/pgim_preloader.js');
  const { SplashOverlayController } = require('../web/ui/splash_overlay.js');

  assert.equal(DEFAULT_SPLASH_FADE_OUT_MS, 220);

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

  // Build a minimal valid 64-byte PGIM v1 container for real PgimPreloader OPFS warm-boot testing.
  function buildMinimalPgimBytes() {
    const buf = new ArrayBuffer(64);
    const bytes = new Uint8Array(buf);
    const view = new DataView(buf);
    bytes[0] = 0x50; // P
    bytes[1] = 0x47; // G
    bytes[2] = 0x49; // I
    bytes[3] = 0x4d; // M
    view.setUint32(4, 1, true); // version
    view.setUint32(8, 64, true); // header_bytes
    view.setUint32(12, 64, true); // section_entry_bytes
    view.setUint32(16, 0, true); // flags
    view.setUint32(20, 4096, true); // page_size
    view.setBigUint64(24, 0x140000000n, true); // image_base
    view.setUint32(32, 0x1000, true); // size_of_image
    view.setUint32(36, 0x1000, true); // section_alignment
    view.setUint32(40, 0, true); // entry_point_rva
    view.setUint32(44, 0, true); // section_count
    view.setUint32(48, 0, true); // dir64_count
    view.setUint32(52, 0, true); // highlow_count
    view.setBigUint64(56, 64n, true); // file_size
    return bytes;
  }

  const clock = createClock(10000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });
  documentObj.head = makeMockNode('head');
  documentObj.createElement = (tag) => makeMockNode(tag);
  documentObj.querySelector = () => null;

  const pgimBytes = buildMinimalPgimBytes();
  const opfsStoreMap = new Map();
  let opfsReads = 0;
  let opfsWrites = 0;
  const mockOpfsCache = {
    async read(key) {
      opfsReads += 1;
      return opfsStoreMap.has(key) ? opfsStoreMap.get(key) : null;
    },
    async write(key, bytes) {
      opfsWrites += 1;
      opfsStoreMap.set(key, new Uint8Array(bytes));
    },
    async delete(key) {
      opfsStoreMap.delete(key);
    },
  };

  let inFlightPreloads = 0;
  let maxConcurrentPreloads = 0;
  let runtimeWasmCalls = 0;
  let opfsProbeCalls = 0;
  let customTaskCalls = 0;
  const splashProgressSamples = [];
  let fadeScheduledMs = null;
  let fadeCallback = null;

  const fetchImpl = async (url, init = {}) => {
    if (String(url).includes('/api/status')) {
      return {
        ok: true,
        status: 200,
        async json() {
          return {
            ok: true,
            coopCoep: true,
            hasWasmBinary: true,
            pgimUrl: '/boot/guest.pgim',
            runtimeWasm: '/dist/runtime.wasm',
          };
        },
      };
    }
    if (String(url).includes('/boot/guest.pgim')) {
      const headers = {
        get(name) {
          const lower = String(name).toLowerCase();
          if (lower === 'content-length') return String(pgimBytes.byteLength);
          if (lower === 'accept-ranges') return 'bytes';
          if (lower === 'etag') return '"pgim-v1"';
          return null;
        },
      };
      if (init.method === 'HEAD') {
        return { ok: true, status: 200, headers };
      }
      return {
        ok: true,
        status: 200,
        headers,
        async arrayBuffer() {
          return pgimBytes.slice().buffer;
        },
      };
    }
    return { ok: false, status: 404, headers: { get: () => null } };
  };

  const mockWorker = createFakeEventTarget({
    postMessage() {},
    terminate() {},
  });

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl,
    canvas: webgpu.canvas,
    hud: createMockHud(),
    splashContainer: makeMockNode('div'),
    SplashOverlayCtor: SplashOverlayController,
    PgimPreloaderCtor: PgimPreloader,
    BootOrchestratorCtor: BootOrchestrator,
    // W310 OPFS warm-boot cache options pass-through
    opfsCache: mockOpfsCache,
    opfsCacheVersion: 2,
    opfsCacheDirName: 'custom-pgim-cache',
    opfsValidateWithHead: true,
    // W309 BootOrchestrator concurrent preload options pass-through
    parallelPreload: true,
    preloadConcurrency: 4,
    releasePreloadBuffers: true,
    preloadRuntimeWasm: async () => {
      runtimeWasmCalls += 1;
      inFlightPreloads += 1;
      maxConcurrentPreloads = Math.max(maxConcurrentPreloads, inFlightPreloads);
      clock.advance(40);
      await Promise.resolve();
      inFlightPreloads -= 1;
      return { compiled: true, bytesLoaded: 1024 };
    },
    opfsProbe: async () => {
      opfsProbeCalls += 1;
      inFlightPreloads += 1;
      maxConcurrentPreloads = Math.max(maxConcurrentPreloads, inFlightPreloads);
      clock.advance(30);
      await Promise.resolve();
      inFlightPreloads -= 1;
      return { warmBootReady: true };
    },
    bootOrchestratorOptions: {
      preloadTasks: [
        {
          name: 'customSidecar',
          optional: true,
          async run() {
            customTaskCalls += 1;
            return { ok: true };
          },
        },
      ],
      onPhaseChange() {
        if (app && app.splashOverlay) {
          splashProgressSamples.push(app.splashOverlay.getState().progressPct);
        }
      },
    },
    // W311 SplashOverlay default 220ms fadeOutMs (no explicit fadeMs/fadeOutMs)
    splashOverlayOptions: {
      setTimeout(fn, ms) {
        fadeScheduledMs = ms;
        fadeCallback = fn;
        return 1;
      },
      clearTimeout() {
        fadeCallback = null;
      },
    },
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker() {
      return mockWorker;
    },
  });

  try {
    // W310 options were passed cleanly to PgimPreloader.
    assert.ok(app.pgimPreloader instanceof PgimPreloader);
    assert.equal(app.pgimPreloader.opfsCacheVersion, 2);
    assert.equal(app.pgimPreloader.opfsCacheDirName, 'custom-pgim-cache');
    assert.equal(app.pgimPreloader.opfsValidateWithHead, true);

    // W311 default fadeOutMs is 220ms.
    assert.equal(app.splashOverlay.getState().fadeOutMs, 220);
    assert.equal(app.splashOverlay.getState().fadeMs, 220);

    const started = await app.start();
    assert.equal(started.ok, true);
    await app.pgimPreloader.flushOpfsWrites();

    // W309 concurrent preload tasks ran through BootOrchestrator.
    assert.equal(runtimeWasmCalls, 1);
    assert.equal(opfsProbeCalls, 1);
    assert.equal(customTaskCalls, 1);
    assert.ok(maxConcurrentPreloads >= 2);

    // W310 OPFS cache was probed and written through.
    assert.ok(opfsReads >= 1);
    assert.ok(opfsWrites >= 1);

    // W311 splash progress stayed strictly monotonic across orchestrator phases.
    assert.ok(splashProgressSamples.length >= 2);
    for (let i = 1; i < splashProgressSamples.length; i += 1) {
      assert.ok(
        splashProgressSamples[i] >= splashProgressSamples[i - 1],
        `Expected monotonic splash progress: ${splashProgressSamples.join(', ')}`
      );
    }

    // Out-of-order earlier phase update cannot regress splash progress.
    const beforeStale = app.splashOverlay.getState().progressPct;
    app.splashOverlay.updateBootState({ stage: 'PGIM_LOAD', phaseProgress: 0.1 });
    assert.ok(app.splashOverlay.getState().progressPct >= beforeStale);

    // FIRST_PRESENT dismisses with default 220ms fadeOutMs.
    clock.advance(50);
    mockWorker.dispatch('message', { data: { type: 'first-present', framesPresented: 1 } });
    assert.equal(fadeScheduledMs, 220);
    assert.equal(app.splashOverlay.getState().fading, true);
    clock.advance(220);
    fadeCallback();
    assert.equal(app.splashOverlay.getState().fadeOutCompleted, true);
    assert.equal(app.splashOverlay.getState().visible, false);

    // Telemetry snapshots surface W309/W310 preloadTelemetry and W311 splashTelemetry cleanly.
    const snap = app.getBootTelemetry();
    assert.equal(Object.isFrozen(snap.preloadTelemetry), true);
    assert.equal(snap.preloadTelemetry.parallelPreload, true);
    assert.equal(snap.preloadTelemetry.preloadConcurrency, 4);
    assert.ok(snap.preloadTelemetry.parallelStagesCount >= 2);
    assert.ok(snap.preloadTelemetry.opfsWrites >= 1);
    assert.equal(Object.isFrozen(snap.splashTelemetry), true);
    assert.equal(snap.splashTelemetry.fadeOutMs, 220);
    assert.equal(snap.splashTelemetry.fadeOutCompleted, true);
    assert.equal(snap.splashTelemetry.progressPct, 100);
  } finally {
    await app.destroy();
  }
});

test('W333 wires W321 RuntimeSessionController phase durations, getPhaseTelemetry(), and reclaimSoftMemory() / memory-pressure callbacks', async () => {
  const clock = createClock(5000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });

  const mockWorker = createFakeEventTarget({
    postMessage() {},
    terminate() {},
  });

  let releasePreloadCalls = 0;
  let evictionCalls = 0;
  const memoryReclaimEvents = [];
  const memoryPressureEvents = [];

  const mockOrchestrator = {
    async start() {
      return { ok: true, bootMode: 'wasm-guest' };
    },
    releasePreloadBuffers() {
      releasePreloadCalls += 1;
      return { releasedFields: ['pgim.bytes'], releasedBytes: 1024 };
    },
    getStats() {
      return { parallelPreload: true, preloadConcurrency: 8, totalBootMs: 120 };
    },
  };

  const mockStorageManager = {
    async getFastBootPlan() {
      return {
        warmCacheReady: true,
        recommendedSlabMaxBytes: 64 * 1024 * 1024,
        recommendedOpfsBudgetBytes: 256 * 1024 * 1024,
        quotaHeadroomBytes: 1024 * 1024 * 1024,
        pressureLevel: 'low',
        shouldEvictColdChunks: false,
      };
    },
    async runQuotaPressureEviction() {
      evictionCalls += 1;
      return { ok: true, evictedBytes: 8192, evictedEntries: 2 };
    },
  };

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch({ ok: true, coopCoep: true, hasWasmBinary: true }),
    canvas: webgpu.canvas,
    hud: createMockHud(),
    bootOrchestrator: mockOrchestrator,
    storageManager: mockStorageManager,
    onMemoryReclaim(ctx) {
      memoryReclaimEvents.push(ctx);
    },
    onMemoryPressure(ctx) {
      memoryPressureEvents.push(ctx);
    },
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker() {
      return mockWorker;
    },
  });

  try {
    const started = await app.start();
    assert.equal(started.ok, true);
    assert.equal(app.getPhaseTelemetry().currentPhase, 'boot');

    // Advance clock through boot -> start_screen -> main_menu -> in_game via milestones & worker message.
    clock.advance(400);
    app.recordFastBootMilestone('START_SCREEN');
    clock.advance(300);
    mockWorker.dispatch('message', {
      data: { type: 'session-phase', phase: 'main_menu' },
    });
    clock.advance(500);
    app.recordFastBootMilestone('IN_GAME');
    clock.advance(250);

    const phaseTel = app.getPhaseTelemetry();
    assert.equal(Object.isFrozen(phaseTel), true);
    assert.equal(Object.isFrozen(phaseTel.phaseDurationsMs), true);
    assert.equal(phaseTel.currentPhase, 'in_game');
    assert.equal(phaseTel.phaseDurationsMs.boot, 400);
    assert.equal(phaseTel.phaseDurationsMs.start_screen, 300);
    assert.equal(phaseTel.phaseDurationsMs.main_menu, 500);
    assert.equal(phaseTel.phaseDurationsMs.in_game, 250);
    assert.ok(phaseTel.phaseTransitionCount >= 4);

    // Register an extra dynamic reclaim hook + a throwing hook to verify error isolation.
    let dynamicHookCount = 0;
    const unregisterDynamic = app.registerReclaimHook(() => {
      dynamicHookCount += 1;
    });
    const unregisterFailing = app.registerReclaimHook(() => {
      throw new Error('non-fatal reclaim warning');
    });

    const reclaimRes = await app.reclaimSoftMemory('vram-high');
    assert.equal(reclaimRes.reason, 'vram-high');
    assert.equal(reclaimRes.reclaimRuns, 1);
    assert.equal(reclaimRes.reclaimFailures, 1);
    assert.equal(releasePreloadCalls, 1);
    assert.equal(evictionCalls, 1);
    assert.equal(memoryReclaimEvents.length, 1);
    assert.equal(memoryPressureEvents.length, 1);
    assert.equal(dynamicHookCount, 1);

    // Unregister failing hook, then trigger memory-pressure worker message.
    unregisterFailing();
    mockWorker.dispatch('message', {
      data: { type: 'memory-pressure', reason: 'heap-spike' },
    });
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.equal(app.getPhaseTelemetry().reclaimRuns, 2);
    assert.equal(app.getPhaseTelemetry().reclaimFailures, 1);
    assert.equal(app.getPhaseTelemetry().lastReclaimReason, 'heap-spike');
    assert.equal(memoryReclaimEvents.length, 2);
    assert.equal(dynamicHookCount, 2);
    unregisterDynamic();

    // Verify telemetry snapshots include W321 phase & reclaim fields.
    const snap = app.getBootTelemetry();
    assert.equal(snap.sessionPhase, 'in_game');
    assert.equal(snap.phaseDurationsMs.boot, 400);
    assert.equal(snap.phaseDurationsMs.main_menu, 500);
    assert.equal(snap.reclaimRuns, 2);
    assert.equal(snap.reclaimFailures, 1);
    assert.equal(snap.phaseTelemetry.currentPhase, 'in_game');

    const diag = app.getDiagnosticsSnapshot();
    assert.equal(diag.sessionPhase, 'in_game');
    assert.equal(diag.phaseDurationsMs.boot, 400);
    assert.equal(diag.reclaimRuns, 2);
    assert.equal(diag.reclaimFailures, 1);
  } finally {
    await app.destroy();
  }

  // Reclaim after destroy returns cleanly without invoking hooks.
  const afterDestroy = await app.reclaimSoftMemory('late-pressure');
  assert.equal(afterDestroy.ok, false);
  assert.equal(afterDestroy.destroyed, true);
  assert.equal(memoryReclaimEvents.length, 2);
});

test('W343 throttles per-frame WebGPU frame presented logs while updating perfMonitor, HUD metrics, and defaultHoldUntilVisible splash hold', async () => {
  const { createRequire } = await import('node:module');
  const require = createRequire(import.meta.url);
  const { SplashOverlayController, DISMISS_REASONS } = require('../web/ui/splash_overlay.js');
  const { BootOrchestrator } = require('../web/runtime/boot_orchestrator.js');

  // Helper unit checks.
  assert.equal(parseWebGpuFramePresentedLabel('WebGPU frame #1 presented'), 1);
  assert.equal(parseWebGpuFramePresentedLabel('WebGPU frame #3014 presented'), 3014);
  assert.equal(parseWebGpuFramePresentedLabel('wgpu_worker ready (nvidia turing)'), null);
  assert.equal(parseWebGpuFramePresentedLabel(null), null);

  assert.equal(shouldLogWebGpuPresentedFrame(1), true);
  assert.equal(shouldLogWebGpuPresentedFrame(2, { lastLoggedFrame: 1, lastLoggedAtMs: 0, nowMs: 16 }), false);
  assert.equal(shouldLogWebGpuPresentedFrame(10, { lastLoggedFrame: 1, lastLoggedAtMs: 0, nowMs: 160 }), true);
  assert.equal(shouldLogWebGpuPresentedFrame(11, { lastLoggedFrame: 10, lastLoggedAtMs: 160, nowMs: 176 }), false);
  assert.equal(shouldLogWebGpuPresentedFrame(60, { lastLoggedFrame: 10, lastLoggedAtMs: 160, nowMs: 1000 }), true);
  assert.equal(shouldLogWebGpuPresentedFrame(61, { lastLoggedFrame: 60, lastLoggedAtMs: 1000, nowMs: 1016 }), false);
  assert.equal(shouldLogWebGpuPresentedFrame(360, { lastLoggedFrame: 60, lastLoggedAtMs: 1000, nowMs: 6000 }), true);
  assert.equal(shouldLogWebGpuPresentedFrame(65, { lastLoggedFrame: 60, lastLoggedAtMs: 1000, nowMs: 11000 }), true);

  // wgpu_worker ready (<adapter>) log extraction.
  assert.equal(extractGpuAdapterFromLogLine('wgpu_worker ready (nvidia turing)'), 'nvidia turing');
  assert.equal(extractGpuAdapterFromLogLine('wgpu_worker ready (WebGPU)'), null);

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

  const clock = createClock(1000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();
  const broadcastInstances = [];
  class MockBroadcastChannel {
    constructor(name) {
      this.name = name;
      this.onmessage = null;
      broadcastInstances.push(this);
    }
    close() {}
  }

  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });
  windowObj.location = { origin: 'http://127.0.0.1:8080', search: '' };
  windowObj.BroadcastChannel = MockBroadcastChannel;
  documentObj.head = makeMockNode('head');
  documentObj.createElement = (tag) => makeMockNode(tag);
  documentObj.querySelector = () => null;

  const mockWorker = createFakeEventTarget({
    postMessage() {},
    terminate() {},
  });

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch({
      ok: true,
      coopCoep: true,
      hasWasmBinary: true,
      pgimUrl: '/boot/guest.pgim',
      splashImage: '/game/xbs_splash.png',
    }),
    canvas: webgpu.canvas,
    hud,
    splashContainer: makeMockNode('div'),
    SplashOverlayCtor: SplashOverlayController,
    BootOrchestratorCtor: BootOrchestrator,
    pgimPreloader: {
      async preloadBootBundle() {
        return { pgim: new Uint8Array(4), shaderPackBytes: null, reflectionManifest: null, bytesLoaded: 4 };
      },
    },
    splashOverlayOptions: { fadeMs: 0, visibleFrameFadeMs: 0, nowFn: clock.now },
    defaultHoldUntilVisible: true,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker() {
      return mockWorker;
    },
  });

  try {
    assert.equal(app.bootUrlOptions.holdUntilVisible, true);
    const started = await app.start();
    assert.equal(started.ok, true);
    assert.equal(broadcastInstances.length, 1);

    // Boot log & adapter ready log must never be evicted by per-frame spam.
    broadcastInstances[0].onmessage({ data: { label: 'wgpu_worker ready (nvidia turing)' } });
    broadcastInstances[0].onmessage({ data: { label: '[boot] D3D12_CREATE_DEVICE' } });

    // Simulate 400 real-Wasm64 frames at ~60 FPS arriving via BOTH BroadcastChannel label
    // AND worker gpu-telemetry (just like web/loader.js in real Chrome).
    for (let frame = 1; frame <= 400; frame += 1) {
      clock.advance(16);
      broadcastInstances[0].onmessage({
        data: { label: `WebGPU frame #${frame} presented` },
      });
      mockWorker.dispatch('message', {
        data: {
          type: 'gpu-telemetry',
          frameStats: {
            framesPresented: frame,
            droppedFrames: 0,
            drawCalls: frame * 3,
            packetsProcessed: frame * 12,
            computeDispatches: frame,
          },
        },
      });
    }

    // 1. Splash stays held on black frames while defaultHoldUntilVisible: true is active!
    assert.equal(app.splashOverlay.getState().holdingForVisibleFrame, true);
    assert.equal(app.splashOverlay.getState().dismissed, false);
    assert.equal(app.splashOverlay.getState().blackPresentsWhileHeld, 400);

    // 2. Per-frame log spam is stopped: only milestones (#1, #10, #60, #360) logged, not 400!
    const crashFrameLogs = app.crashReporter
      .getLogs()
      .filter((entry) => /^WebGPU frame #\d+ presented$/.test(entry.message))
      .map((entry) => entry.message);
    assert.deepEqual(crashFrameLogs, [
      'WebGPU frame #1 presented',
      'WebGPU frame #10 presented',
      'WebGPU frame #60 presented',
      'WebGPU frame #360 presented',
    ]);
    const hudFrameLogs = hud.logs
      .filter((entry) => /^WebGPU frame #\d+ presented$/.test(entry.message))
      .map((entry) => entry.message);
    assert.deepEqual(hudFrameLogs, crashFrameLogs);

    // Early boot log is still preserved in CrashReporter recentLogs (not evicted!).
    assert.ok(
      app.crashReporter.getLogs().some((entry) => entry.message === 'wgpu_worker ready (nvidia turing)')
    );

    // 3. HUD Frames / FPS metrics still updated smoothly.
    const lastFrameMetric = [...hud.metricsUpdates]
      .reverse()
      .find((m) => m.framesPresented >= 390 || m.totalFrames >= 390);
    assert.ok(lastFrameMetric, 'Expected HUD metrics to receive live frame updates');
    assert.ok(
      hud.metricsUpdates.some((m) => typeof m.fps === 'number' && m.fps > 30),
      'Expected HUD metrics to receive live positive FPS updates'
    );

    // 4. PerfMonitor has exact live real-Wasm64 counts with ZERO double-counting!
    const perfSnap = app.perfMonitor.getSnapshot();
    assert.equal(perfSnap.frames.totalFrames, 400);
    assert.ok(perfSnap.frames.fps > 55 && perfSnap.frames.fps < 70, `Expected ~62.5 FPS, got ${perfSnap.frames.fps}`);
    assert.equal(perfSnap.gpu.totalDrawCalls, 1200);
    assert.equal(perfSnap.gpu.totalGpuPackets, 4800);
    assert.equal(perfSnap.gpu.totalComputeDispatches, 400);

    // 5. W338 first-visible-frame dismisses splash smoothly.
    clock.advance(50);
    mockWorker.dispatch('message', {
      data: {
        type: 'first-visible-frame',
        firstVisibleFrame: true,
        framesPresented: 401,
      },
    });
    assert.equal(app.splashOverlay.getState().dismissed, true);
    assert.equal(app.splashOverlay.getState().dismissReason, DISMISS_REASONS.FIRST_VISIBLE_FRAME);
  } finally {
    await app.destroy();
  }
});

test('W343 Export Report populates capabilities.wasm64 (Chrome 133+ address: i64), live perf, and bootDiagnostics in real-Wasm64 sessions', async () => {
  const { HudController } = await import('../web/ui/hud_controller.js');

  function assertI32Limit(value) {
    return +value;
  }

  // Strict Chrome 133+ V8 WebAssembly.Memory mock: accepts ONLY `{ address: 'i64' }` with BigInts,
  // and throws TypeError on legacy `{ index: 'i64' }` (which previously broke CrashReporter.collectCapabilities).
  const chrome133WasmApi = {
    Memory: class Chrome133Wasm64Memory {
      constructor(desc) {
        if (desc.address === 'i64') {
          if (typeof desc.initial !== 'bigint' || typeof desc.maximum !== 'bigint') {
            throw new TypeError('Expected BigInt limits for i64 memory');
          }
          return;
        }
        assertI32Limit(desc.initial);
      }
    },
  };

  const clock = createClock(2000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();
  const broadcastInstances = [];
  class MockBroadcastChannel {
    constructor(name) {
      this.name = name;
      this.onmessage = null;
      broadcastInstances.push(this);
    }
    close() {}
  }

  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });
  windowObj.WebAssembly = chrome133WasmApi;
  windowObj.BroadcastChannel = MockBroadcastChannel;
  windowObj.location = { origin: 'http://127.0.0.1:8080', search: '' };

  const mockWorker = createFakeEventTarget({
    postMessage() {},
    terminate() {},
  });

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch({ ok: true, coopCoep: true, hasWasmBinary: true }),
    canvas: webgpu.canvas,
    hud,
    defaultHoldUntilVisible: true,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker() {
      return mockWorker;
    },
  });

  const downloads = [];
  const hudController = new HudController({
    app,
    sessionController: app.session,
    storageManager: app.storageManager,
    crashReporter: app.crashReporter,
    perfMonitor: app.perfMonitor,
    inputProfiles: app.inputProfiles,
    hud,
    downloadBlob(filename, content) {
      downloads.push({ filename, content });
      return true;
    },
  });

  try {
    const started = await app.start();
    assert.equal(started.ok, true);

    // Emit boot adapter, 30 frames, gpu-telemetry, and first-visible-frame.
    broadcastInstances[0].onmessage({ data: { label: 'wgpu_worker ready (NVIDIA / Turing)' } });
    for (let i = 1; i <= 30; i += 1) {
      clock.advance(20);
      broadcastInstances[0].onmessage({ data: { label: `WebGPU frame #${i} presented` } });
    }
    mockWorker.dispatch('message', {
      data: {
        type: 'gpu-telemetry',
        hasVisibleContent: false,
        frameStats: {
          framesPresented: 30,
          droppedFrames: 0,
          drawCalls: 420,
          realPackDrawCount: 380,
          fallbackDrawCount: 40,
          packetsProcessed: 1500,
          computeDispatches: 15,
        },
        fallbackStats: {
          totalLookups: 420,
          packHits: 380,
          fallbackHits: 40,
          builtinHits: 0,
        },
        missingShaders: [{ hash: '0x00000000cafe1234', stage: 'fragment', hits: 40 }],
      },
    });
    clock.advance(100);
    mockWorker.dispatch('message', {
      data: {
        type: 'first-visible-frame',
        firstVisibleFrame: true,
        firstVisibleFrameMs: 1200,
        blackLoadingMs: 1180,
        framesPresented: 31,
        gpuAdapter: 'NVIDIA / Turing (C:\\Users\\alice\\nvwgf2umx.dll)',
      },
    });

    const exportResult = await hudController.exportCrashReport();
    assert.equal(exportResult.ok, true);
    assert.equal(downloads.length, 1);

    const report = JSON.parse(exportResult.json);

    // 1. capabilities.wasm64 is true on Chrome 133+ `{ address: 'i64' }`!
    assert.equal(report.capabilities.wasm64, true);

    // 2. Live perf stats populated (NOT 0).
    assert.equal(report.perf.frames.totalFrames, 31);
    assert.ok(report.perf.frames.fps > 0);
    assert.equal(report.perf.gpu.totalDrawCalls, 420);
    assert.equal(report.perf.gpu.totalGpuPackets, 1500);
    assert.equal(report.perf.gpu.totalComputeDispatches, 15);

    // 3. bootDiagnostics populated (NOT null) and strictly sanitized.
    assert.equal(report.bootDiagnostics.gpuAdapter, 'nvidia / turing');
    assert.equal(report.bootDiagnostics.firstVisibleFrameMs, 1200);
    assert.equal(report.bootDiagnostics.blackLoadingMs, 1180);
    assert.ok(
      typeof report.bootDiagnostics.presentFpsAvg === 'number' &&
        report.bootDiagnostics.presentFpsAvg > 0
    );
    assert.ok(report.bootDiagnostics.bootSnapshot);
    assert.equal(report.bootDiagnostics.bootSnapshot.stage, 'FIRST_VISIBLE_FRAME');
    assert.ok(report.bootDiagnostics.gpuTelemetry);
    assert.equal(report.bootDiagnostics.gpuTelemetry.framesPresented, 31);
    assert.equal(report.bootDiagnostics.gpuTelemetry.drawCalls, 420);
    assert.equal(report.bootDiagnostics.gpuTelemetry.fallbackStats.packHits, 380);
    assert.ok(!exportResult.json.includes('alice'));

    // 4. recentLogs not flooded by per-frame spam.
    const frameLogEntries = report.recentLogs.filter((l) => /^WebGPU frame #\d+ presented$/.test(l.message));
    assert.deepEqual(
      frameLogEntries.map((l) => l.message),
      ['WebGPU frame #1 presented', 'WebGPU frame #10 presented']
    );
  } finally {
    hudController.destroy();
    await app.destroy();
  }

  // Non-Wasm64 engine still reports capabilities.wasm64: false in CrashReporter.
  const wasm32Host = createMockBrowserHost({ clock, raf, timers, gpu: webgpu.gpu });
  wasm32Host.windowObj.WebAssembly = {
    Memory: class Wasm32OnlyMemory {
      constructor(desc) {
        assertI32Limit(desc.initial);
      }
    },
  };
  const wasm32App = bootstrapPlayGta6App({
    windowObj: wasm32Host.windowObj,
    documentObj: wasm32Host.documentObj,
    navigatorObj: wasm32Host.navigatorObj,
    fetchImpl: createMockFetch({ ok: true, coopCoep: true, hasWasmBinary: false }),
    canvas: webgpu.canvas,
    hud: createMockHud(),
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
  });
  try {
    assert.equal(wasm32App.crashReporter.collectCapabilities().wasm64, false);
    assert.equal(wasm32App.crashReporter.buildReport().capabilities.wasm64, false);
  } finally {
    await wasm32App.destroy();
  }
});

test('W349 gpu-telemetry with firstVisibleFrame: false holds splash despite realPackDrawCount > 0 until first-visible-frame arrives', async () => {
  const { createRequire } = await import('node:module');
  const require = createRequire(import.meta.url);
  const { SplashOverlayController, DISMISS_REASONS } = require('../web/ui/splash_overlay.js');
  const { BootOrchestrator } = require('../web/runtime/boot_orchestrator.js');

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

  function createHarness() {
    const clock = createClock(1000);
    const raf = createFakeRaf(clock);
    const timers = createFakeTimers();
    const webgpu = createMockWebGpuEnv();
    const hud = createMockHud();
    const broadcastInstances = [];
    class MockBroadcastChannel {
      constructor(name) {
        this.name = name;
        this.onmessage = null;
        broadcastInstances.push(this);
      }
      close() {}
    }

    const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
      clock,
      raf,
      timers,
      gpu: webgpu.gpu,
    });
    windowObj.location = { origin: 'http://127.0.0.1:8080', search: '' };
    windowObj.BroadcastChannel = MockBroadcastChannel;
    documentObj.head = makeMockNode('head');
    documentObj.createElement = (tag) => makeMockNode(tag);
    documentObj.querySelector = () => null;

    const mockWorker = createFakeEventTarget({
      postMessage() {},
      terminate() {},
    });

    const app = bootstrapPlayGta6App({
      windowObj,
      documentObj,
      navigatorObj,
      fetchImpl: createMockFetch({
        ok: true,
        coopCoep: true,
        hasWasmBinary: true,
        pgimUrl: '/boot/guest.pgim',
        splashImage: '/game/xbs_splash.png',
      }),
      canvas: webgpu.canvas,
      hud,
      splashContainer: makeMockNode('div'),
      SplashOverlayCtor: SplashOverlayController,
      BootOrchestratorCtor: BootOrchestrator,
      pgimPreloader: {
        async preloadBootBundle() {
          return {
            pgim: new Uint8Array(4),
            shaderPackBytes: null,
            reflectionManifest: null,
            bytesLoaded: 4,
          };
        },
      },
      splashOverlayOptions: { fadeMs: 0, visibleFrameFadeMs: 0, nowFn: clock.now },
      defaultHoldUntilVisible: true,
      now: clock.now,
      setInterval: timers.setInterval,
      clearInterval: timers.clearInterval,
      createWasmWorker() {
        return mockWorker;
      },
    });

    return { app, clock, mockWorker, broadcastInstances };
  }

  // Scenario 1: Active non-black pixel probe reports firstVisibleFrame: false with realPackDrawCount: 42.
  const gated = createHarness();
  try {
    const started = await gated.app.start();
    assert.equal(started.ok, true);

    // FIRST_PRESENT arrives -> splash enters hold-until-visible state.
    gated.clock.advance(250);
    gated.broadcastInstances[0].onmessage({ data: { label: 'WebGPU frame #1 presented' } });
    assert.equal(gated.app.splashOverlay.getState().holdingForVisibleFrame, true);
    assert.equal(gated.app.splashOverlay.getState().dismissed, false);

    // gpu-telemetry reports realPackDrawCount: 42 while firstVisibleFrame: false -> MUST stay held!
    gated.clock.advance(500);
    gated.mockWorker.dispatch('message', {
      data: {
        type: 'gpu-telemetry',
        firstVisibleFrame: false,
        firstVisibleFrameMs: null,
        blackLoadingMs: null,
        maxLuma: 0,
        suppressedBlackFrames: 12,
        frameStats: {
          framesPresented: 12,
          droppedFrames: 0,
          drawCalls: 64,
          realPackDrawCount: 42,
        },
      },
    });
    assert.equal(gated.app.splashOverlay.getState().holdingForVisibleFrame, true);
    assert.equal(gated.app.splashOverlay.getState().dismissed, false);
    assert.equal(gated.app.bootTracer.getSummary().firstVisibleFrameMs, null);

    // Even if hasVisibleContent is null alongside firstVisibleFrame: false, or a follow-up partial
    // gpu-telemetry omits firstVisibleFrame while the probe gate is active, splash MUST stay held!
    gated.clock.advance(500);
    gated.mockWorker.dispatch('message', {
      data: {
        type: 'gpu-telemetry',
        hasVisibleContent: null,
        firstVisibleFrame: false,
        frameStats: {
          framesPresented: 24,
          realPackDrawCount: 84,
        },
      },
    });
    gated.mockWorker.dispatch('message', {
      data: {
        type: 'gpu-telemetry',
        frameStats: {
          framesPresented: 30,
          realPackDrawCount: 99,
        },
      },
    });
    assert.equal(gated.app.splashOverlay.getState().holdingForVisibleFrame, true);
    assert.equal(gated.app.splashOverlay.getState().dismissed, false);
    assert.equal(gated.app.bootTracer.getSummary().firstVisibleFrameMs, null);

    // Explicit first-visible-frame with non-black pixels (maxLuma: 235) dismisses splash immediately!
    gated.clock.advance(1000);
    gated.mockWorker.dispatch('message', {
      data: {
        type: 'first-visible-frame',
        firstVisibleFrame: true,
        firstVisibleFrameMs: 2250,
        firstPresentMs: 250,
        blackLoadingMs: 2000,
        presentCount: 31,
        maxLuma: 235,
        suppressedBlackFrames: 30,
        source: 'swapchain_probe',
        gpuAdapter: { vendor: 'nvidia', architecture: 'turing', label: 'nvidia turing' },
      },
    });
    assert.equal(gated.app.splashOverlay.getState().dismissed, true);
    assert.equal(
      gated.app.splashOverlay.getState().dismissReason,
      DISMISS_REASONS.FIRST_VISIBLE_FRAME,
    );
    assert.equal(gated.app.bootTracer.getSummary().stage, 'FIRST_VISIBLE_FRAME');
    assert.equal(gated.app.getBootTelemetry().hasVisibleContent, true);
  } finally {
    await gated.app.destroy();
  }

  // Scenario 2: Backward compatibility — legacy mock worker omitting firstVisibleFrame still dismisses on realPackDrawCount > 0.
  const legacy = createHarness();
  try {
    const started = await legacy.app.start();
    assert.equal(started.ok, true);

    legacy.clock.advance(200);
    legacy.broadcastInstances[0].onmessage({ data: { label: 'WebGPU frame #1 presented' } });
    assert.equal(legacy.app.splashOverlay.getState().holdingForVisibleFrame, true);
    assert.equal(legacy.app.splashOverlay.getState().dismissed, false);

    legacy.clock.advance(300);
    legacy.mockWorker.dispatch('message', {
      data: {
        type: 'gpu-telemetry',
        frameStats: {
          framesPresented: 3,
          realPackDrawCount: 6,
        },
      },
    });
    assert.equal(legacy.app.splashOverlay.getState().dismissed, true);
    assert.equal(
      legacy.app.splashOverlay.getState().dismissReason,
      DISMISS_REASONS.FIRST_VISIBLE_FRAME,
    );
  } finally {
    await legacy.app.destroy();
  }
});

test('shouldRelaunchGuest caps self-relaunches per session and resets after a present', () => {
  const store = new Map();
  const win = {
    sessionStorage: {
      getItem: (k) => (store.has(k) ? store.get(k) : null),
      setItem: (k, v) => store.set(k, String(v)),
      removeItem: (k) => store.delete(k),
    },
  };
  assert.equal(GUEST_RELAUNCH_MESSAGE_TYPE, 'guest-relaunch');
  for (let i = 0; i < MAX_GUEST_RELAUNCHES; i += 1) {
    assert.equal(shouldRelaunchGuest(win), true);
  }
  assert.equal(shouldRelaunchGuest(win), false);
  resetGuestRelaunchCount(win);
  assert.equal(shouldRelaunchGuest(win), true);
  assert.equal(shouldRelaunchGuest({}), true);
  assert.equal(shouldRelaunchGuest(null), true);
});

test('Orch-13 InputBridge visual boot sync (notifyFirstPresent & notifyFirstVisibleFrame) and canvas dataset.bootState lifecycle', async () => {
  const { createRequire } = await import('node:module');
  const require = createRequire(import.meta.url);
  const { SplashOverlayController } = require('../web/ui/splash_overlay.js');
  const { BootOrchestrator } = require('../web/runtime/boot_orchestrator.js');

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

  function createFakeTimeouts() {
    let nextId = 1;
    const active = new Map();
    return {
      setTimeout(fn, ms) {
        const id = nextId++;
        active.set(id, { fn, ms });
        return id;
      },
      clearTimeout(id) {
        active.delete(id);
      },
      pendingCount() {
        return active.size;
      },
    };
  }

  // Scenario A: Worker first-present / frame-presented / first-visible-frame messages sync
  // InputBridge (even when BootOrchestrator handles notifyFirstPresent early), activate
  // startOn: 'FIRST_VISIBLE_FRAME' scripts, enforce idempotency, and update canvas.dataset.bootState.
  const clockA = createClock(1000);
  const rafA = createFakeRaf(clockA);
  const timersA = createFakeTimers();
  const timeoutsA = createFakeTimeouts();
  const webgpuA = createMockWebGpuEnv();
  webgpuA.canvas.dataset = {};
  const { windowObj: winA, documentObj: docA, navigatorObj: navA } = createMockBrowserHost({
    clock: clockA,
    raf: rafA,
    timers: timersA,
    gpu: webgpuA.gpu,
  });
  winA.location = { origin: 'http://127.0.0.1:8080', search: '' };
  docA.head = makeMockNode('head');
  docA.createElement = (tag) => makeMockNode(tag);
  docA.querySelector = () => null;

  const mockWorkerA = createFakeEventTarget({
    postMessage() {},
    terminate() {},
  });

  const appA = bootstrapPlayGta6App({
    windowObj: winA,
    documentObj: docA,
    navigatorObj: navA,
    fetchImpl: createMockFetch({
      ok: true,
      coopCoep: true,
      hasWasmBinary: true,
      pgimUrl: '/boot/guest.pgim',
      splashImage: '/game/xbs_splash.png',
    }),
    canvas: webgpuA.canvas,
    hud: createMockHud(),
    splashContainer: makeMockNode('div'),
    SplashOverlayCtor: SplashOverlayController,
    BootOrchestratorCtor: BootOrchestrator,
    pgimPreloader: {
      async preloadBootBundle() {
        return { pgim: new Uint8Array(4), shaderPackBytes: null, reflectionManifest: null, bytesLoaded: 4 };
      },
    },
    splashOverlayOptions: { fadeMs: 0, visibleFrameFadeMs: 0, nowFn: clockA.now },
    defaultHoldUntilVisible: true,
    now: clockA.now,
    setInterval: timersA.setInterval,
    clearInterval: timersA.clearInterval,
    createWasmWorker: () => mockWorkerA,
  });

  let firstPresentCalls = 0;
  let firstVisibleCalls = 0;
  const origNotifyPresent = appA.inputBridge.notifyFirstPresent.bind(appA.inputBridge);
  const origNotifyVisible = appA.inputBridge.notifyFirstVisibleFrame.bind(appA.inputBridge);
  appA.inputBridge.notifyFirstPresent = (...args) => {
    firstPresentCalls += 1;
    return origNotifyPresent(...args);
  };
  appA.inputBridge.notifyFirstVisibleFrame = (...args) => {
    firstVisibleCalls += 1;
    return origNotifyVisible(...args);
  };

  try {
    assert.equal(webgpuA.canvas.dataset.bootState, 'loading');
    const started = await appA.start();
    assert.equal(started.ok, true);
    assert.ok(appA.bootOrchestrator, 'BootOrchestrator must be active');

    // Arm a milestone-gated script waiting for FIRST_VISIBLE_FRAME.
    appA.inputBridge.startScript('start-screen', {
      startOn: 'FIRST_VISIBLE_FRAME',
      visibleFrameFallbackMs: 60_000,
      nowFn: clockA.now,
      setTimeoutFn: timeoutsA.setTimeout,
      clearTimeoutFn: timeoutsA.clearTimeout,
      setIntervalFn: timersA.setInterval,
      clearIntervalFn: timersA.clearInterval,
    });
    assert.equal(appA.inputBridge.getStats().scriptPending, true);
    assert.equal(appA.inputBridge.getStats().scriptActive, false);

    // Worker first-present -> BootOrchestrator handles early, InputBridge still notified once, canvas -> presenting.
    mockWorkerA.dispatch('message', {
      data: { type: 'first-present', presentCount: 1 },
    });
    assert.equal(firstPresentCalls, 1);
    assert.equal(appA.inputBridge.getStats().firstPresentNotified, true);
    assert.equal(webgpuA.canvas.dataset.bootState, 'presenting');

    // Subsequent frame-presented and explicit app.notifyFirstPresent() are idempotent.
    mockWorkerA.dispatch('message', {
      data: { type: 'frame-presented', presentCount: 2 },
    });
    appA.notifyFirstPresent();
    assert.equal(firstPresentCalls, 1);
    assert.equal(webgpuA.canvas.dataset.bootState, 'presenting');

    // Worker first-visible-frame -> InputBridge notified once, script activates, canvas -> visible.
    mockWorkerA.dispatch('message', {
      data: {
        type: 'first-visible-frame',
        firstVisibleFrame: true,
        firstVisibleFrameMs: 1500,
        presentCount: 3,
        maxLuma: 220,
      },
    });
    assert.equal(firstVisibleCalls, 1);
    assert.equal(appA.inputBridge.getStats().firstVisibleFrameNotified, true);
    assert.equal(appA.inputBridge.getStats().scriptPending, false);
    assert.equal(appA.inputBridge.getStats().scriptActive, true);
    assert.equal(appA.inputBridge.getStats().scriptStartedOn, 'FIRST_VISIBLE_FRAME');
    assert.equal(webgpuA.canvas.dataset.bootState, 'visible');

    // Explicit repeat app.notifyFirstVisibleFrame() is idempotent.
    appA.notifyFirstVisibleFrame();
    assert.equal(firstVisibleCalls, 1);

    // stop() resets canvas.dataset.bootState back to 'loading'.
    await appA.stop();
    assert.equal(webgpuA.canvas.dataset.bootState, 'loading');
  } finally {
    await appA.destroy();
  }

  // Scenario B: Direct app.notifyFirstPresent() / app.notifyFirstVisibleFrame() re-anchors
  // an already-running anchorable menu preset (chrome-menu-flow-4) and resets state on stop().
  const clockB = createClock(2000);
  const rafB = createFakeRaf(clockB);
  const timersB = createFakeTimers();
  const webgpuB = createMockWebGpuEnv();
  webgpuB.canvas.dataset = {};
  const { windowObj: winB, documentObj: docB, navigatorObj: navB } = createMockBrowserHost({
    clock: clockB,
    raf: rafB,
    timers: timersB,
    gpu: webgpuB.gpu,
  });

  const appB = bootstrapPlayGta6App({
    windowObj: winB,
    documentObj: docB,
    navigatorObj: navB,
    fetchImpl: createMockFetch({ ok: true, coopCoep: true, hasWasmBinary: true }),
    canvas: webgpuB.canvas,
    hud: createMockHud(),
    now: clockB.now,
    setInterval: timersB.setInterval,
    clearInterval: timersB.clearInterval,
    createWasmWorker: () => createFakeEventTarget({ postMessage() {}, terminate() {} }),
  });

  try {
    assert.equal(webgpuB.canvas.dataset.bootState, 'loading');
    await appB.start();
    appB.inputBridge.startScript('chrome-menu-flow-4', {
      nowFn: clockB.now,
      setIntervalFn: timersB.setInterval,
      clearIntervalFn: timersB.clearInterval,
    });
    assert.equal(appB.inputBridge.getStats().scriptStartedOn, 'immediate');
    assert.equal(appB.inputBridge.getStats().scriptReanchoredCount, 0);

    appB.notifyFirstPresent();
    assert.equal(webgpuB.canvas.dataset.bootState, 'presenting');
    assert.equal(appB.inputBridge.getStats().firstPresentNotified, true);

    appB.notifyFirstVisibleFrame();
    assert.equal(webgpuB.canvas.dataset.bootState, 'visible');
    assert.equal(appB.inputBridge.getStats().scriptStartedOn, 'FIRST_VISIBLE_FRAME');
    assert.equal(appB.inputBridge.getStats().scriptReanchoredCount, 1);

    await appB.stop();
    assert.equal(webgpuB.canvas.dataset.bootState, 'loading');
  } finally {
    await appB.destroy();
  }
});

test('Orch-13 smooth splash defaults (fadeOutMs 220 + visibleFrameFadeMs 450), sidecar progress filtering, visiblePresentsTotal: 0 gate, and BootOrchestrator dismiss without hide', async () => {
  const { createRequire } = await import('node:module');
  const require = createRequire(import.meta.url);
  const { SplashOverlayController, DISMISS_REASONS } = require('../web/ui/splash_overlay.js');
  const { BootOrchestrator } = require('../web/runtime/boot_orchestrator.js');

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

  // Part 1: BootOrchestrator.notifyFirstPresent() calls dismiss() without snapping via hide() when dismiss exists.
  {
    const calls = [];
    const orchestrator = new BootOrchestrator({
      splashOverlay: {
        dismiss(meta) {
          calls.push({ fn: 'dismiss', meta });
        },
        hide(meta) {
          calls.push({ fn: 'hide', meta });
        },
      },
    });
    const res = orchestrator.notifyFirstPresent({ source: 'unit-test' });
    assert.equal(res.ok, true);
    assert.deepEqual(
      calls.map((c) => c.fn),
      ['dismiss']
    );

    // Fallback: when overlay only implements hide(), hide() is still invoked.
    const hideOnlyCalls = [];
    const fallbackOrch = new BootOrchestrator({
      splashOverlay: {
        hide() {
          hideOnlyCalls.push('hide');
        },
      },
    });
    fallbackOrch.notifyFirstPresent({ source: 'unit-test' });
    assert.deepEqual(hideOnlyCalls, ['hide']);
  }

  // Part 2: Default SplashOverlayController keeps fadeOutMs: 220 and visibleFrameFadeMs: 450,
  // sidecar progress (opfsWarmBoot, reflectionSidecar) is ignored for PGIM_LOAD phaseProgress,
  // and visiblePresentsTotal: 0 latches the active pixel-probe gate until visiblePresentsTotal > 0.
  const clock = createClock(5000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const broadcastInstances = [];
  class MockBroadcastChannel {
    constructor(name) {
      this.name = name;
      this.onmessage = null;
      broadcastInstances.push(this);
    }
    close() {}
  }

  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });
  windowObj.location = { origin: 'http://127.0.0.1:8080', search: '' };
  windowObj.BroadcastChannel = MockBroadcastChannel;
  documentObj.head = makeMockNode('head');
  documentObj.createElement = (tag) => makeMockNode(tag);
  documentObj.querySelector = () => null;

  const mockWorker = createFakeEventTarget({
    postMessage() {},
    terminate() {},
  });

  const bootStateUpdates = [];
  let releasePgimPreload = null;
  let pgimOnProgress = null;
  const splashTimeouts = [];
  let nextSplashTimerId = 1;
  const splashTimerMap = new Map();

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch({
      ok: true,
      coopCoep: true,
      hasWasmBinary: true,
      pgimUrl: '/boot/guest.pgim',
      shaderPackUrl: '/shaders/splash.wgsl.pack',
      splashImage: '/game/xbs_splash.png',
    }),
    canvas: webgpu.canvas,
    hud: createMockHud(),
    splashContainer: makeMockNode('div'),
    SplashOverlayCtor: SplashOverlayController,
    BootOrchestratorCtor: BootOrchestrator,
    bootOrchestratorOptions: {
      parallelPreload: true,
      opfsProbe: async ({ onProgress }) => {
        onProgress({ loadedBytes: 1024, totalBytes: 1024, pct: 100 });
        return { warmed: true };
      },
      reflectionSidecarLoader: async (_url, { onProgress }) => {
        onProgress({ loadedBytes: 2048, totalBytes: 2048, pct: 100 });
        return { entries: 4 };
      },
    },
    pgimPreloader: {
      async preloadBootBundle({ onProgress }) {
        pgimOnProgress = onProgress;
        await new Promise((resolve) => {
          releasePgimPreload = resolve;
        });
        return {
          pgim: new Uint8Array(4),
          shaderPackBytes: null,
          reflectionManifest: null,
          bytesLoaded: 4,
        };
      },
    },
    // Intentionally omit fadeMs / fadeOutMs / visibleFrameFadeMs to verify defaults!
    splashOverlayOptions: {
      nowFn: clock.now,
      setTimeout: (fn, ms) => {
        const id = nextSplashTimerId++;
        const entry = { id, fn, ms };
        splashTimeouts.push(entry);
        splashTimerMap.set(id, entry);
        return id;
      },
      clearTimeout: (id) => {
        splashTimerMap.delete(id);
      },
    },
    defaultHoldUntilVisible: true,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker: () => mockWorker,
  });

  try {
    // Default fadeOutMs stays 220 while preserving smooth 450ms visible-frame cross-fade.
    assert.equal(app.splashOverlay.getState().fadeOutMs, 220);
    assert.equal(app.splashOverlay.getState().visibleFrameFadeMs, 450);

    // Spy on splashOverlay.updateBootState to verify sidecar progress does not inject phaseProgress.
    const origUpdateBootState = app.splashOverlay.updateBootState.bind(app.splashOverlay);
    app.splashOverlay.updateBootState = (patch) => {
      bootStateUpdates.push({ ...patch });
      return origUpdateBootState(patch);
    };

    const startPromise = app.start();
    // Wait a tick for parallel opfsWarmBoot + reflectionSidecar tasks to report 100% progress.
    await new Promise((r) => setTimeout(r, 10));

    const sidecarPatches = bootStateUpdates.filter(
      (p) =>
        p.streamingProgress &&
        (p.streamingProgress.asset === 'opfsWarmBoot' ||
          p.streamingProgress.asset === 'reflectionSidecar')
    );
    assert.ok(sidecarPatches.length >= 2, 'expected opfsWarmBoot and reflectionSidecar progress updates');
    for (const patch of sidecarPatches) {
      assert.equal(
        patch.phaseProgress,
        undefined,
        `non-primary sidecar ${patch.streamingProgress.asset} must not set phaseProgress`
      );
    }
    assert.ok(
      app.splashOverlay.getState().progressPct < 20,
      `tiny sidecar progress must not slam splash progressPct to 20% (got ${app.splashOverlay.getState().progressPct})`
    );

    // Primary pgim progress DOES set stage: 'PGIM_LOAD' and phaseProgress.
    assert.equal(typeof pgimOnProgress, 'function');
    pgimOnProgress({ asset: 'pgim', loadedBytes: 50, totalBytes: 100, pct: 50 });
    const pgimPatch = bootStateUpdates.find(
      (p) => p.streamingProgress && p.streamingProgress.asset === 'pgim' && p.phaseProgress === 0.5
    );
    assert.ok(pgimPatch, 'primary pgim preload progress must set phaseProgress: 0.5');

    releasePgimPreload();
    const started = await startPromise;
    assert.equal(started.ok, true);

    // FIRST_PRESENT enters splash hold.
    clock.advance(200);
    broadcastInstances[0].onmessage({ data: { label: 'WebGPU frame #1 presented' } });
    assert.equal(app.splashOverlay.getState().holdingForVisibleFrame, true);
    assert.equal(app.splashOverlay.getState().dismissed, false);

    // Explicit visiblePresentsTotal: 0 latches the pixel-probe gate and holds splash despite realPackDrawCount > 0.
    clock.advance(300);
    mockWorker.dispatch('message', {
      data: {
        type: 'gpu-telemetry',
        visiblePresentsTotal: 0,
        frameStats: { framesPresented: 8, realPackDrawCount: 35 },
      },
    });
    assert.equal(app.splashOverlay.getState().holdingForVisibleFrame, true);
    assert.equal(app.splashOverlay.getState().dismissed, false);

    // Subsequent partial telemetry omitting visiblePresentsTotal stays held while latched.
    mockWorker.dispatch('message', {
      data: {
        type: 'gpu-telemetry',
        frameStats: { framesPresented: 12, realPackDrawCount: 64 },
      },
    });
    assert.equal(app.splashOverlay.getState().holdingForVisibleFrame, true);
    assert.equal(app.splashOverlay.getState().dismissed, false);

    // Explicit visiblePresentsTotal > 0 releases the gate and starts the 450ms visible-frame fade.
    clock.advance(400);
    mockWorker.dispatch('message', {
      data: {
        type: 'gpu-telemetry',
        visiblePresentsTotal: 1,
        frameStats: { framesPresented: 13, realPackDrawCount: 68 },
      },
    });
    assert.equal(app.splashOverlay.getState().dismissed, true);
    assert.equal(app.splashOverlay.getState().fading, true);
    const fadeTimer = [...splashTimerMap.values()].find((entry) => entry.ms === 450);
    assert.ok(fadeTimer, 'expected smooth 450ms visible-frame fade timer');
    fadeTimer.fn();
    assert.equal(app.splashOverlay.getState().fadeOutCompleted, true);
    assert.equal(app.splashOverlay.getState().dismissReason, DISMISS_REASONS.FIRST_VISIBLE_FRAME);
  } finally {
    await app.destroy();
  }
});

test('Orch-13 viewport metrics (300x150 fallback + DPR clamp), initPayload devicePixelRatio/swapchainOptions, ResizeObserver, no-op resize dedupe, and restart canvas safety', async () => {
  // Part 1: 300x150 default canvas fallback to innerWidth/innerHeight, DPR clamping to 2,
  // initPayload devicePixelRatio: 1 + swapchainOptions, no-op resize dedupe, ResizeObserver,
  // and parentNode.replaceChild(cloneNode(false)) on Wasm64 restart.
  const clock = createClock(1000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();

  let canvasSeq = 0;
  let replaceChildCalls = 0;
  let cloneNodeCalls = 0;
  const parentNode = {
    child: null,
    replaceChild(nextNode, oldNode) {
      replaceChildCalls += 1;
      assert.equal(oldNode, this.child);
      this.child = nextNode;
      nextNode.parentNode = this;
      return oldNode;
    },
  };

  function makeTransferableCanvas() {
    const id = ++canvasSeq;
    let transferred = false;
    const offscreen = { kind: 'MockOffscreenCanvas', canvasId: id };
    const c = createFakeEventTarget({
      id: `canvas-${id}`,
      width: 300,
      height: 150,
      clientWidth: 0,
      clientHeight: 0,
      dataset: {},
      parentNode,
      transferControlToOffscreen() {
        if (transferred) {
          throw new Error('InvalidStateError: Cannot transfer control from a canvas more than once');
        }
        transferred = true;
        return offscreen;
      },
      cloneNode(deep) {
        cloneNodeCalls += 1;
        assert.equal(deep, false);
        return makeTransferableCanvas();
      },
    });
    c._offscreen = offscreen;
    return c;
  }

  const initialCanvas = makeTransferableCanvas();
  parentNode.child = initialCanvas;

  const observerInstances = [];
  class MockResizeObserver {
    constructor(cb) {
      this.cb = cb;
      this.observed = [];
      this.disconnected = 0;
      observerInstances.push(this);
    }
    observe(target) {
      this.observed.push(target);
    }
    disconnect() {
      this.disconnected += 1;
    }
    trigger() {
      this.cb([]);
    }
  }

  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });
  windowObj.innerWidth = 1440;
  windowObj.innerHeight = 900;
  windowObj.devicePixelRatio = 3; // Exceeds default maxDevicePixelRatio (2)
  windowObj.ResizeObserver = MockResizeObserver;

  const workers = [];
  const createWasmWorker = () => {
    const posted = [];
    const worker = createFakeEventTarget({
      posted,
      postMessage(payload, transfer) {
        posted.push({ payload, transfer });
      },
      terminate() {},
    });
    workers.push(worker);
    return worker;
  };

  const swapchainOptions = { alphaMode: 'opaque', powerPreference: 'high-performance' };
  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch({ ok: true, coopCoep: true, hasWasmBinary: true }),
    canvas: initialCanvas,
    hud: createMockHud(),
    swapchainOptions,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker,
  });

  try {
    // ResizeObserver observed the canvas on bootstrap.
    assert.equal(observerInstances.length, 1);
    assert.deepEqual(observerInstances[0].observed, [initialCanvas]);

    // 1. Start Wasm64: prefers innerWidth/innerHeight (1440x900) over HTML default 300x150,
    //    clamps implicit DPR 3 -> 2 (2880x1800), sends devicePixelRatio: 1, forwards swapchainOptions.
    const r1 = await app.start();
    assert.equal(r1.ok, true);
    assert.equal(workers.length, 1);
    const init1 = workers[0].posted[0];
    assert.equal(init1.payload.canvas, initialCanvas._offscreen);
    assert.equal(init1.payload.width, 2880);
    assert.equal(init1.payload.height, 1800);
    assert.equal(init1.payload.devicePixelRatio, 1);
    assert.deepEqual(init1.payload.swapchainOptions, swapchainOptions);
    assert.deepEqual(init1.transfer, [initialCanvas._offscreen]);

    // 2. Duplicate no-op resize with identical pixel dimensions is skipped.
    assert.equal(app.resize(1440, 900), false);
    assert.equal(workers[0].posted.length, 1);

    // Real resize posts new pixel dimensions (DPR 3 clamped to 2 -> 2560x1440), then skips repeat.
    assert.equal(app.resize(1280, 720), true);
    assert.equal(workers[0].posted.length, 2);
    assert.deepEqual(workers[0].posted[1].payload, { resize: true, width: 2560, height: 1440 });
    assert.equal(app.resize(1280, 720, 2), false);
    assert.equal(workers[0].posted.length, 2);

    // 3. ResizeObserver callback triggers resize with clamped DPR.
    initialCanvas.clientWidth = 1000;
    initialCanvas.clientHeight = 500;
    observerInstances[0].trigger();
    assert.equal(workers[0].posted.length, 3);
    assert.deepEqual(workers[0].posted[2].payload, { resize: true, width: 2000, height: 1000 });

    // 4. Restart Wasm64 after transferControlToOffscreen(): clones & replaces canvas in parentNode
    //    instead of crashing with a second transfer on the same canvas.
    await app.stop();
    const r2 = await app.start();
    assert.equal(r2.ok, true);
    assert.equal(cloneNodeCalls, 1);
    assert.equal(replaceChildCalls, 1);
    assert.equal(workers.length, 2);
    assert.notEqual(parentNode.child, initialCanvas);
    assert.equal(workers[1].posted[0].payload.canvas, parentNode.child._offscreen);
    assert.deepEqual(workers[1].posted[0].transfer, [parentNode.child._offscreen]);
  } finally {
    await app.destroy();
  }

  // ResizeObserver disconnects cleanly on destroy().
  assert.equal(observerInstances[0].disconnected, 1);

  // Part 2: Restart without parentNode / cloneNode reuses transferredOffscreenCanvas safely without re-transferring.
  let detachedTransfers = 0;
  const detachedOffscreen = { kind: 'DetachedOffscreen' };
  const detachedCanvas = createFakeEventTarget({
    width: 640,
    height: 360,
    clientWidth: 640,
    clientHeight: 360,
    transferControlToOffscreen() {
      detachedTransfers += 1;
      if (detachedTransfers > 1) {
        throw new Error('InvalidStateError: second transfer');
      }
      return detachedOffscreen;
    },
  });
  const detachedWorkers = [];
  const detachedHost = createMockBrowserHost({
    clock: createClock(3000),
    raf: createFakeRaf(createClock(3000)),
    timers: createFakeTimers(),
    gpu: webgpu.gpu,
  });
  const detachedApp = bootstrapPlayGta6App({
    windowObj: detachedHost.windowObj,
    documentObj: detachedHost.documentObj,
    navigatorObj: detachedHost.navigatorObj,
    fetchImpl: createMockFetch({ ok: true, coopCoep: true, hasWasmBinary: true }),
    canvas: detachedCanvas,
    hud: createMockHud(),
    createWasmWorker: () => {
      const posted = [];
      const w = createFakeEventTarget({
        posted,
        postMessage(payload, transfer) {
          posted.push({ payload, transfer });
        },
        terminate() {},
      });
      detachedWorkers.push(w);
      return w;
    },
  });
  try {
    assert.equal((await detachedApp.start()).ok, true);
    await detachedApp.stop();
    assert.equal((await detachedApp.start()).ok, true);
    assert.equal(detachedTransfers, 1);
    assert.equal(detachedWorkers.length, 2);
    assert.equal(detachedWorkers[1].posted[0].payload.canvas, detachedOffscreen);
    assert.equal(detachedWorkers[1].posted[0].transfer, undefined);
  } finally {
    await detachedApp.destroy();
  }
});

test('Orch-23 FIRST_VISIBLE_FRAME anchor presets pass startOn FIRST_VISIBLE_FRAME and activate on notifyFirstVisibleFrame()', async () => {
  for (const preset of ['chrome-to-ingame', 'chrome-menu-flow-6', 'menu-to-ingame']) {
    const clock = createClock(50000);
    const raf = createFakeRaf(clock);
    const timers = createFakeTimers();
    const webgpu = createMockWebGpuEnv();
    const hud = createMockHud();

    const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
      clock,
      raf,
      timers,
      gpu: webgpu.gpu,
    });
    windowObj.location = {
      origin: 'https://playgta6.local',
      search: `?inputScript=${preset}`,
    };

    const mockWorker = createFakeEventTarget({
      postMessage() {},
      terminate() {},
    });

    const app = bootstrapPlayGta6App({
      windowObj,
      documentObj,
      navigatorObj,
      fetchImpl: createMockFetch({ ok: true, coopCoep: true, hasWasmBinary: true }),
      canvas: webgpu.canvas,
      hud,
      now: clock.now,
      raf: raf.raf,
      cancelRaf: raf.cancelRaf,
      setInterval: timers.setInterval,
      clearInterval: timers.clearInterval,
      inputScriptOptions: {
        nowFn: clock.now,
        setIntervalFn: timers.setInterval,
        clearIntervalFn: timers.clearInterval,
      },
      createWasmWorker() {
        return mockWorker;
      },
    });

    try {
      const startScriptCalls = [];
      const origStartScript = app.inputBridge.startScript.bind(app.inputBridge);
      app.inputBridge.startScript = (...args) => {
        startScriptCalls.push(args);
        return origStartScript(...args);
      };

      const started = await app.start();
      assert.equal(started.ok, true);
      assert.equal(startScriptCalls.length, 1);
      assert.equal(startScriptCalls[0][0], preset);
      assert.equal(
        startScriptCalls[0][1]?.startOn,
        'FIRST_VISIBLE_FRAME',
        `${preset} must pass startOn: 'FIRST_VISIBLE_FRAME'`
      );

      // Before FIRST_VISIBLE_FRAME, script is armed/pending and does NOT fire inputs yet.
      assert.equal(app.inputBridge.getScriptState().active, false);
      assert.equal(app.inputBridge.getStats().scriptPending, true);
      assert.equal(app.getDiagnosticsSnapshot().inputScript, preset);
      clock.advance(1000);
      timers.tickAll();
      assert.equal(app.inputBridge.getScriptState().scriptedPressesSent, 0);

      // Triggering FIRST_VISIBLE_FRAME activates the anchored script reliably.
      app.notifyFirstVisibleFrame({ source: 'test-visible-frame' });
      assert.equal(app.inputBridge.getScriptState().active, true);
      assert.equal(app.inputBridge.getStats().scriptPending, false);
      assert.equal(app.inputBridge.getStats().scriptStartedOn, 'FIRST_VISIBLE_FRAME');

      // Advancing past initial +500ms FIRST_VISIBLE_FRAME A press emits scripted inputs.
      clock.advance(600);
      timers.tickAll();
      assert.ok(
        app.inputBridge.getScriptState().scriptedPressesSent > 0,
        `${preset} must send scripted inputs after notifyFirstVisibleFrame()`
      );
    } finally {
      await app.destroy();
    }
  }
});

test('Orch-23 explicit inputScriptOptions.startOn override is respected and stop() cancels pending anchored scripts', async () => {
  const clock = createClock(60000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();

  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });
  windowObj.location = {
    origin: 'https://playgta6.local',
    search: '?inputScript=chrome-to-ingame',
  };

  // 1. Explicit startOn: 'immediate' override is preserved.
  const appImmediate = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch({ ok: true, coopCoep: true, hasWasmBinary: true }),
    canvas: webgpu.canvas,
    hud: createMockHud(),
    now: clock.now,
    raf: raf.raf,
    cancelRaf: raf.cancelRaf,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    inputScriptOptions: { startOn: 'immediate' },
    createWasmWorker: () => createFakeEventTarget({ postMessage() {}, terminate() {} }),
  });

  try {
    assert.equal((await appImmediate.start()).ok, true);
    assert.equal(appImmediate.inputBridge.getScriptState().active, true);
    assert.equal(appImmediate.inputBridge.getStats().scriptPending, false);
  } finally {
    await appImmediate.destroy();
  }

  // 2. Pending anchored script is cleanly stopped by app.stop().
  const appPending = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch({ ok: true, coopCoep: true, hasWasmBinary: true }),
    canvas: webgpu.canvas,
    hud: createMockHud(),
    now: clock.now,
    raf: raf.raf,
    cancelRaf: raf.cancelRaf,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker: () => createFakeEventTarget({ postMessage() {}, terminate() {} }),
  });

  try {
    assert.equal((await appPending.start()).ok, true);
    assert.equal(appPending.inputBridge.getStats().scriptPending, true);
    assert.equal((await appPending.stop()).ok, true);
    assert.equal(appPending.inputBridge.getStats().scriptPending, false);
    assert.equal(appPending.inputBridge.getScriptState().active, false);
  } finally {
    await appPending.destroy();
  }
});

test('Orch-23 bootstrapPlayGta6App wires stream config into BootOrchestrator so 7.51 GB shader pack is never GET-downloaded', async () => {
  const { createRequire } = await import('node:module');
  const require = createRequire(import.meta.url);
  const { BootOrchestrator } = require('../web/runtime/boot_orchestrator.js');
  const { PgimPreloader } = require('../web/runtime/pgim_preloader.js');
  const SEVEN_POINT_FIVE_GB = 8_063_800_000;

  for (const search of ['', '?streamShaders=1']) {
    const clock = createClock(70000);
    const raf = createFakeRaf(clock);
    const timers = createFakeTimers();
    const webgpu = createMockWebGpuEnv();
    const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
      clock,
      raf,
      timers,
      gpu: webgpu.gpu,
    });
    windowObj.location = {
      origin: 'https://playgta6.local',
      search,
    };

    const shaderMethods = [];
    const fetchImpl = async (url, init = {}) => {
      const method = (init.method || 'GET').toUpperCase();
      const u = String(url);
      if (u === '/api/status') {
        return {
          ok: true,
          status: 200,
          headers: new Headers({ 'content-type': 'application/json' }),
          async json() {
            return {
              ok: true,
              coopCoep: true,
              hasWasmBinary: true,
              pgimUrl: '/data/boot/GTA6.pgim',
              shaderPackUrl: '/data/shaders/splash.wgsl.pack',
              runtimeWasm: '/dist/runtime.wasm',
            };
          },
        };
      }
      if (u === '/data/shaders/splash.wgsl.pack') {
        shaderMethods.push(method);
        if (method === 'HEAD') {
          return {
            ok: true,
            status: 200,
            headers: new Headers({
              'content-length': String(SEVEN_POINT_FIVE_GB),
              'accept-ranges': 'bytes',
            }),
          };
        }
        throw new Error(`Unexpected ${method} on 7.51 GB shader pack for search="${search}"`);
      }
      return { ok: false, status: 404, headers: new Headers() };
    };

    const mockPreloader = {
      calls: [],
      async preloadBootBundle(args) {
        this.calls.push(args);
        return {
          pgim: { header: { magic: 'PGIM' }, bytesLoaded: 64 },
          shaderPackBytes: null,
          reflectionManifest: null,
          bytesLoaded: 64,
        };
      },
    };

    const app = bootstrapPlayGta6App({
      windowObj,
      documentObj,
      navigatorObj,
      fetchImpl,
      canvas: webgpu.canvas,
      hud: createMockHud(),
      now: clock.now,
      raf: raf.raf,
      cancelRaf: raf.cancelRaf,
      setInterval: timers.setInterval,
      clearInterval: timers.clearInterval,
      enableBootOrchestrator: true,
      BootOrchestratorCtor: BootOrchestrator,
      PgimPreloaderCtor: PgimPreloader,
      pgimPreloader: mockPreloader,
      createWasmWorker: () => createFakeEventTarget({ postMessage() {}, terminate() {} }),
    });

    try {
      const res = await app.start();
      assert.equal(res.ok, true);
      assert.equal(mockPreloader.calls.length, 1);
      assert.equal(
        mockPreloader.calls[0].shaderPackUrl,
        null,
        `search="${search}" must omit 7.51 GB shaderPackUrl from preloadBootBundle`
      );
      if (search === '?streamShaders=1') {
        assert.deepEqual(shaderMethods, [], '?streamShaders=1 should skip without even HEAD');
      } else {
        assert.deepEqual(shaderMethods, ['HEAD'], 'plain URL should probe HEAD and skip GET');
      }
    } finally {
      await app.destroy();
    }
  }
});
