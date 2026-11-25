import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import { BootTracer, BOOT_MILESTONES } from '../web/telemetry/boot_tracer.js';

const require = createRequire(import.meta.url);
const orchestratorMod = require('../web/runtime/boot_orchestrator.js');
const {
  BOOT_PHASE,
  BOOT_MODE,
  BootOrchestrator,
  DEFAULT_STATUS_URL,
  DEFAULT_STAGE_TIMEOUT_MS,
  DEFAULT_PARALLEL_PRELOAD,
  DEFAULT_PRELOAD_CONCURRENCY,
  DEFAULT_RELEASE_PRELOAD_BUFFERS,
  discoverBootAssets,
  isRealGuestBoot,
  releaseBootBundleBytes,
  runConcurrentPreloads,
} = orchestratorMod;

function createMockFetch(payload, { ok = true, status = 200, throwError = null, delayMs = 0 } = {}) {
  const calls = [];
  const fetchImpl = async (url, init) => {
    calls.push({ url, init });
    if (init && init.signal) {
      if (init.signal.aborted) {
        const err = new Error('The operation was aborted.');
        err.name = 'AbortError';
        throw err;
      }
    }
    if (delayMs > 0) {
      await new Promise((resolve, reject) => {
        const timer = setTimeout(resolve, delayMs);
        if (init && init.signal) {
          init.signal.addEventListener(
            'abort',
            () => {
              clearTimeout(timer);
              const err = new Error('The operation was aborted.');
              err.name = 'AbortError';
              reject(err);
            },
            { once: true }
          );
        }
      });
    }
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

function readyStatus(overrides = {}) {
  return {
    ok: true,
    coopCoep: true,
    dataDirConfigured: true,
    hasManifest: true,
    hasBootset: true,
    hasWasmBinary: true,
    shaderPackCount: 1,
    splashImage: '/game/xbs_splash.png',
    pgimUrl: '/data/boot/GTA6.pgim',
    shaderPackUrl: '/data/shaders/boot.wgsl.pack',
    reflectionManifestUrl: '/data/shaders/reflection-manifest.json',
    runtimeWasm: '/dist/runtime.wasm',
    ...overrides,
  };
}

function createOverlay() {
  const calls = [];
  return {
    calls,
    inits: [],
    updates: [],
    progresses: [],
    tracers: [],
    dismissals: [],
    destroys: 0,
    init(payload) {
      this.inits.push(payload);
      calls.push(['init', payload]);
    },
    update(payload) {
      this.updates.push(payload);
      calls.push(['update', payload]);
    },
    updateProgress(event) {
      this.progresses.push(event);
      calls.push(['updateProgress', event]);
    },
    updateTracer(summary) {
      this.tracers.push(summary);
      calls.push(['updateTracer', summary]);
    },
    setSplashImage(url) {
      calls.push(['setSplashImage', url]);
    },
    dismiss(meta) {
      this.dismissals.push(meta);
      calls.push(['dismiss', meta]);
    },
    destroy() {
      this.destroys += 1;
      calls.push(['destroy']);
    },
  };
}

function createPreloader({ progressEvents = null, bundle = null, delayMs = 0 } = {}) {
  const calls = [];
  const defaultBundle = {
    pgim: { header: { magic: 'PGIM' }, bytesLoaded: 64 },
    shaderPackBytes: new Uint8Array([1, 2, 3]),
    reflectionManifest: { schema_version: 1 },
    bytesLoaded: 64,
  };
  const preloadBootBundle = async ({ pgimUrl, shaderPackUrl, reflectionUrl, onProgress, signal }) => {
    calls.push({ pgimUrl, shaderPackUrl, reflectionUrl });
    if (signal && signal.aborted) {
      const err = new Error('The operation was aborted.');
      err.name = 'AbortError';
      throw err;
    }
    const events = progressEvents || [
      { phase: 'probe', asset: 'pgim', loadedBytes: 0, totalBytes: 64, pct: 0 },
      { phase: 'download', asset: 'pgim', loadedBytes: 32, totalBytes: 64, pct: 50 },
      { phase: 'complete', asset: 'pgim', loadedBytes: 64, totalBytes: 64, pct: 100 },
      { phase: 'complete', asset: 'bundle', loadedBytes: 64, totalBytes: 64, pct: 100 },
    ];
    for (const event of events) {
      if (typeof onProgress === 'function') onProgress(event);
    }
    if (delayMs > 0) {
      await new Promise((resolve, reject) => {
        const timer = setTimeout(resolve, delayMs);
        if (signal) {
          signal.addEventListener(
            'abort',
            () => {
              clearTimeout(timer);
              const err = new Error('The operation was aborted.');
              err.name = 'AbortError';
              reject(err);
            },
            { once: true }
          );
        }
      });
    }
    return bundle || { ...defaultBundle, pgimUrl, shaderPackUrl, reflectionUrl };
  };
  return { preloadBootBundle, calls };
}

function createOrchestrator(overrides = {}) {
  const phases = [];
  const overlay = overrides.splashOverlay !== undefined ? overrides.splashOverlay : createOverlay();
  const bootTracer = overrides.bootTracer !== undefined ? overrides.bootTracer : new BootTracer();
  const launches = [];
  const fallbacks = [];
  const firstPresents = [];
  const orchestrator = new BootOrchestrator({
    fetchImpl: overrides.fetchImpl || createMockFetch(readyStatus()),
    statusUrl: overrides.statusUrl,
    splashOverlay: overlay,
    preloader: overrides.preloader !== undefined ? overrides.preloader : createPreloader(),
    bootTracer,
    onLaunchRuntime:
      overrides.onLaunchRuntime !== undefined
        ? overrides.onLaunchRuntime
        : (payload) => {
            launches.push(payload);
          },
    onFallbackSmokeDemo:
      overrides.onFallbackSmokeDemo !== undefined
        ? overrides.onFallbackSmokeDemo
        : (payload) => {
            fallbacks.push(payload);
          },
    onPhaseChange:
      overrides.onPhaseChange !== undefined
        ? overrides.onPhaseChange
        : (event) => {
            phases.push(event.phase);
          },
    onFirstPresent:
      overrides.onFirstPresent !== undefined
        ? overrides.onFirstPresent
        : (meta) => {
            firstPresents.push(meta);
          },
    onStageStall: overrides.onStageStall,
    nowFn: overrides.nowFn,
    stageTimeoutMs: overrides.stageTimeoutMs,
    parallelPreload: overrides.parallelPreload,
    preloadConcurrency: overrides.preloadConcurrency,
    releasePreloadBuffers: overrides.releasePreloadBuffers,
    preloadRuntimeWasm: overrides.preloadRuntimeWasm,
    opfsProbe: overrides.opfsProbe,
    preloadShaderPackHeader: overrides.preloadShaderPackHeader,
    preloadReflectionSidecar: overrides.preloadReflectionSidecar,
    preloadTasks: overrides.preloadTasks,
  });
  return { orchestrator, overlay, bootTracer, phases, launches, fallbacks, firstPresents };
}

test('UMD exports PlayGta6BootOrchestrator, BOOT_PHASE, and BootOrchestrator', () => {
  assert.equal(globalThis.PlayGta6BootOrchestrator, orchestratorMod);
  assert.equal(typeof BootOrchestrator, 'function');
  assert.equal(orchestratorMod.default, BootOrchestrator);
  assert.equal(DEFAULT_STATUS_URL, '/api/status');
  assert.deepEqual(BOOT_PHASE, {
    IDLE: 'IDLE',
    DISCOVERING: 'DISCOVERING',
    PRELOADING: 'PRELOADING',
    LAUNCHING_RUNTIME: 'LAUNCHING_RUNTIME',
    FALLBACK_SMOKE_DEMO: 'FALLBACK_SMOKE_DEMO',
    PRESENTING: 'PRESENTING',
    ERROR: 'ERROR',
    CANCELLED: 'CANCELLED',
    DESTROYED: 'DESTROYED',
  });
  assert.ok(Object.isFrozen(BOOT_PHASE));
});

test('constructor validates options', () => {
  assert.throws(() => new BootOrchestrator(null), TypeError);
  assert.throws(() => new BootOrchestrator({ fetchImpl: 1 }), TypeError);
  assert.throws(() => new BootOrchestrator({ statusUrl: '' }), TypeError);
  assert.throws(() => new BootOrchestrator({ onLaunchRuntime: 'nope' }), TypeError);
  assert.throws(() => new BootOrchestrator({ splashOverlay: 'overlay' }), TypeError);
  assert.equal(new BootOrchestrator({ fetchImpl: async () => ({}) }).getState().phase, BOOT_PHASE.IDLE);
});

test('discoverBootAssets maps /api/status artifact URLs and hasWasmBinary fallback', () => {
  const rich = discoverBootAssets(readyStatus());
  assert.equal(rich.splashImage, '/game/xbs_splash.png');
  assert.equal(rich.pgimUrl, '/data/boot/GTA6.pgim');
  assert.equal(rich.shaderPackUrl, '/data/shaders/boot.wgsl.pack');
  assert.equal(rich.reflectionManifestUrl, '/data/shaders/reflection-manifest.json');
  assert.equal(rich.runtimeWasmAvailable, true);
  assert.equal(rich.runtimeWasmUrl, '/dist/runtime.wasm');

  const legacy = discoverBootAssets({
    ok: true,
    hasWasmBinary: true,
    shaderPackCount: 0,
  });
  assert.equal(legacy.runtimeWasmAvailable, true);
  assert.equal(legacy.runtimeWasmUrl, '/dist/runtime.wasm');
  assert.equal(legacy.pgimUrl, null);

  const objectWasm = discoverBootAssets({
    runtimeWasm: { present: true, url: '/dist/runtime.wasm' },
    pgimUrl: '/boot.pgim',
  });
  assert.equal(objectWasm.runtimeWasmAvailable, true);
  assert.equal(objectWasm.runtimeWasmUrl, '/dist/runtime.wasm');
});

test('happy-path boot discovers artifacts, preloads, launches runtime, and reaches FIRST_PRESENT', async () => {
  const { orchestrator, overlay, bootTracer, phases, launches, firstPresents } = createOrchestrator();
  const result = await orchestrator.start();

  assert.equal(result.ok, true);
  assert.equal(result.phase, BOOT_PHASE.LAUNCHING_RUNTIME);
  assert.equal(launches.length, 1);
  assert.equal(launches[0].status.pgimUrl, '/data/boot/GTA6.pgim');
  assert.equal(launches[0].bundle.pgim.header.magic, 'PGIM');
  assert.equal(launches[0].bootTracer, bootTracer);
  assert.deepEqual(phases, [BOOT_PHASE.DISCOVERING, BOOT_PHASE.PRELOADING, BOOT_PHASE.LAUNCHING_RUNTIME]);
  assert.equal(overlay.inits[0].splashImage, '/game/xbs_splash.png');
  assert.ok(overlay.calls.some((call) => call[0] === 'setSplashImage' && call[1] === '/game/xbs_splash.png'));

  orchestrator.feedRuntimeLog('[boot] PGIM_LOAD mapped');
  orchestrator.feedRuntimeLog('[boot] TLS_INIT ready');
  orchestrator.feedRuntimeLog('[boot] CRT_INITTERM');
  orchestrator.feedRuntimeLog('[boot] calling guest entry');
  orchestrator.feedRuntimeLog('[boot] XMEM_INIT');
  orchestrator.feedRuntimeLog('[boot] XGameRuntimeInitialize');
  orchestrator.feedRuntimeLog('[boot] D3D12CreateDevice');
  orchestrator.feedRuntimeLog('[boot] BinkOpen');
  orchestrator.feedRuntimeLog('[boot] FIRST_PRESENT');

  assert.equal(orchestrator.getState().phase, BOOT_PHASE.PRESENTING);
  assert.equal(orchestrator.getState().firstPresent, true);
  assert.equal(bootTracer.getSummary().readyForSplash, true);
  assert.equal(bootTracer.getSummary().stage, 'FIRST_PRESENT');
  assert.equal(overlay.dismissals.length, 1);
  assert.equal(firstPresents.length, 1);
  assert.equal(phases.at(-1), BOOT_PHASE.PRESENTING);
});

test('falls back to smoke demo when runtimeWasm is absent', async () => {
  const fetchImpl = createMockFetch(
    readyStatus({
      runtimeWasm: null,
      hasWasmBinary: false,
    })
  );
  const { orchestrator, fallbacks, launches, phases } = createOrchestrator({ fetchImpl });
  const result = await orchestrator.start();

  assert.equal(result.ok, true);
  assert.equal(result.fallback, true);
  assert.equal(result.phase, BOOT_PHASE.FALLBACK_SMOKE_DEMO);
  assert.equal(launches.length, 0);
  assert.equal(fallbacks.length, 1);
  assert.match(fallbacks[0].reason, /runtimeWasm|hasWasmBinary|Wasm64/i);
  assert.equal(fallbacks[0].status.hasWasmBinary, false);
  assert.ok(phases.includes(BOOT_PHASE.PRELOADING));
  assert.equal(phases.at(-1), BOOT_PHASE.FALLBACK_SMOKE_DEMO);
});

test('falls back to smoke demo when pgimUrl is absent even if runtimeWasm is present', async () => {
  const fetchImpl = createMockFetch(
    readyStatus({
      pgimUrl: null,
      shaderPackUrl: null,
    })
  );
  const preloader = createPreloader();
  const { orchestrator, fallbacks, launches, phases } = createOrchestrator({ fetchImpl, preloader });
  const result = await orchestrator.start();

  assert.equal(result.ok, true);
  assert.equal(result.phase, BOOT_PHASE.FALLBACK_SMOKE_DEMO);
  assert.equal(launches.length, 0);
  assert.equal(fallbacks.length, 1);
  assert.match(fallbacks[0].reason, /pgimUrl/i);
  assert.equal(preloader.calls.length, 0);
  assert.deepEqual(phases, [BOOT_PHASE.DISCOVERING, BOOT_PHASE.FALLBACK_SMOKE_DEMO]);
});

test('streaming preload progress is forwarded to BootTracer and splash overlay', async () => {
  const progressEvents = [
    { phase: 'download', asset: 'pgim', loadedBytes: 10, totalBytes: 40, pct: 25 },
    { phase: 'download', asset: 'pgim', loadedBytes: 20, totalBytes: 40, pct: 50 },
    { phase: 'complete', asset: 'pgim', loadedBytes: 40, totalBytes: 40, pct: 100 },
  ];
  const recorded = [];
  const bootTracer = new BootTracer();
  bootTracer.recordProgress = (event) => {
    recorded.push(event);
  };
  const preloader = createPreloader({ progressEvents });
  const { orchestrator, overlay } = createOrchestrator({ preloader, bootTracer });

  await orchestrator.start();

  assert.ok(recorded.length >= 3);
  assert.equal(recorded[0].loadedBytes, 10);
  assert.equal(recorded[1].pct, 50);
  assert.ok(overlay.progresses.some((event) => event.pct === 50));
  assert.equal(overlay.progresses.at(-1).phase, 'complete');
  assert.equal(bootTracer.getSummary().milestones.find((m) => m.milestone === 'PGIM_LOAD').reached, true);
  assert.equal(orchestrator.getState().progress.pct, 100);
});

test('runtime log lines propagate BootTracer milestones and overlay snapshots', async () => {
  const { orchestrator, overlay, bootTracer } = createOrchestrator();
  await orchestrator.start();

  const ingested = orchestrator.feedRuntimeLog('[boot] D3D12CreateDevice');
  assert.equal(ingested.type, 'milestone');
  assert.ok(ingested.milestones.some((entry) => entry.milestone === 'D3D12_CREATE_DEVICE'));
  assert.equal(bootTracer.getSummary().stage, 'D3D12_CREATE_DEVICE');
  assert.ok(overlay.tracers.some((summary) => summary.stage === 'D3D12_CREATE_DEVICE'));
  assert.equal(orchestrator.getState().firstPresent, false);
});

test('runtime trap log lines are recorded by BootTracer and do not dismiss splash', async () => {
  const { orchestrator, overlay, bootTracer, firstPresents } = createOrchestrator();
  await orchestrator.start();

  const ingested = orchestrator.feedRuntimeLog('[trap] UNRESOLVED_IMPORT kernel32.dll!ExitProcess va=0x140001000');
  assert.equal(ingested.type, 'trap');
  assert.equal(ingested.trap.kind, 'UNRESOLVED_IMPORT');
  assert.equal(bootTracer.getSummary().trapCount, 1);
  assert.equal(bootTracer.getSummary().lastTrap.kind, 'UNRESOLVED_IMPORT');
  assert.equal(bootTracer.getSummary().readyForSplash, false);
  assert.equal(overlay.dismissals.length, 0);
  assert.equal(firstPresents.length, 0);
  assert.ok(overlay.tracers.some((summary) => summary.trapCount === 1));
});

test('AbortSignal cancellation during discovery returns CANCELLED without launching', async () => {
  const controller = new AbortController();
  const fetchImpl = createMockFetch(readyStatus(), { delayMs: 40 });
  const { orchestrator, launches, fallbacks, phases } = createOrchestrator({ fetchImpl });

  const pending = orchestrator.start({ signal: controller.signal });
  controller.abort();
  const result = await pending;

  assert.equal(result.ok, false);
  assert.equal(result.aborted, true);
  assert.equal(result.phase, BOOT_PHASE.CANCELLED);
  assert.equal(launches.length, 0);
  assert.equal(fallbacks.length, 0);
  assert.equal(orchestrator.getState().aborted, true);
  assert.ok(phases.includes(BOOT_PHASE.CANCELLED));
});

test('network failure during /api/status is resilient and surfaces ERROR state', async () => {
  const fetchImpl = createMockFetch(null, { throwError: new TypeError('fetch failed') });
  const { orchestrator, launches } = createOrchestrator({ fetchImpl });
  const result = await orchestrator.start();

  assert.equal(result.ok, false);
  assert.equal(result.phase, BOOT_PHASE.ERROR);
  assert.match(result.error.message, /fetch failed/);
  assert.equal(orchestrator.getState().error, result.error.message);
  assert.equal(launches.length, 0);
  assert.doesNotThrow(() => orchestrator.feedRuntimeLog('[boot] FIRST_PRESENT'));
});

test('HTTP error from /api/status does not reject and does not launch', async () => {
  const fetchImpl = createMockFetch({ ok: false }, { ok: false, status: 503 });
  const { orchestrator, fallbacks, launches } = createOrchestrator({ fetchImpl });
  const result = await orchestrator.start();
  assert.equal(result.ok, false);
  assert.equal(result.phase, BOOT_PHASE.ERROR);
  assert.match(result.error.message, /HTTP 503/);
  assert.equal(launches.length, 0);
  assert.equal(fallbacks.length, 0);
});

test('overlay and callback exceptions never escape start, feedRuntimeLog, or notifyFirstPresent', async () => {
  const overlay = {
    init() {
      throw new Error('overlay init boom');
    },
    update() {
      throw new Error('overlay update boom');
    },
    updateProgress() {
      throw new Error('overlay progress boom');
    },
    dismiss() {
      throw new Error('overlay dismiss boom');
    },
    destroy() {
      throw new Error('overlay destroy boom');
    },
  };
  const bootTracer = new BootTracer();
  const { orchestrator } = createOrchestrator({
    splashOverlay: overlay,
    bootTracer,
    onPhaseChange() {
      throw new Error('phase boom');
    },
    onLaunchRuntime() {
      throw new Error('launch boom');
    },
    onFirstPresent() {
      throw new Error('present boom');
    },
  });

  const result = await orchestrator.start();
  assert.equal(result.ok, true);
  assert.equal(result.phase, BOOT_PHASE.LAUNCHING_RUNTIME);
  assert.doesNotThrow(() => orchestrator.feedRuntimeLog('[boot] FIRST_PRESENT'));
  assert.equal(orchestrator.getState().phase, BOOT_PHASE.PRESENTING);
  assert.doesNotThrow(() => orchestrator.notifyFirstPresent({ extra: true }));
  assert.doesNotThrow(() => orchestrator.destroy());
});

test('notifyFirstPresent is idempotent and dismisses splash once', async () => {
  const { orchestrator, overlay, firstPresents } = createOrchestrator();
  await orchestrator.start();

  const first = orchestrator.notifyFirstPresent({ source: 'bink' });
  const second = orchestrator.notifyFirstPresent({ source: 'duplicate' });

  assert.equal(first.ok, true);
  assert.equal(first.alreadyPresented, false);
  assert.equal(second.alreadyPresented, true);
  assert.equal(overlay.dismissals.length, 1);
  assert.equal(firstPresents.length, 1);
  assert.equal(orchestrator.getState().phase, BOOT_PHASE.PRESENTING);
  assert.equal(orchestrator.getState().firstPresent, true);
});

test('destroy is idempotent, aborts in-flight start, and tears down the overlay', async () => {
  const fetchImpl = createMockFetch(readyStatus(), { delayMs: 50 });
  const overlay = createOverlay();
  const { orchestrator, launches } = createOrchestrator({ fetchImpl, splashOverlay: overlay });

  const pending = orchestrator.start();
  const firstDestroy = orchestrator.destroy();
  const result = await pending;
  const secondDestroy = orchestrator.destroy();

  assert.equal(firstDestroy.alreadyDestroyed, false);
  assert.equal(secondDestroy.alreadyDestroyed, true);
  assert.equal(overlay.destroys, 1);
  assert.equal(orchestrator.getState().destroyed, true);
  assert.equal(orchestrator.getState().phase, BOOT_PHASE.DESTROYED);
  assert.equal(launches.length, 0);
  assert.equal(result.ok, false);
  const afterDestroy = await orchestrator.start();
  assert.equal(afterDestroy.destroyed, true);
});

test('preloadBootBundle receives status URLs including reflectionManifestUrl as reflectionUrl', async () => {
  const preloader = createPreloader();
  const fetchImpl = createMockFetch(readyStatus());
  const { orchestrator } = createOrchestrator({ fetchImpl, preloader });
  await orchestrator.start();
  assert.equal(preloader.calls.length, 1);
  assert.deepEqual(preloader.calls[0], {
    pgimUrl: '/data/boot/GTA6.pgim',
    shaderPackUrl: '/data/shaders/boot.wgsl.pack',
    reflectionUrl: '/data/shaders/reflection-manifest.json',
  });
  assert.equal(fetchImpl.calls[0].url, '/api/status');
  assert.equal(fetchImpl.calls[0].init.cache, 'no-store');
});

test('getState reports structured boot snapshot including tracer milestones', async () => {
  const { orchestrator, bootTracer } = createOrchestrator();
  assert.equal(orchestrator.getState().phase, BOOT_PHASE.IDLE);
  await orchestrator.start();
  orchestrator.feedRuntimeLog('[boot] TLS_INIT ready');
  const state = orchestrator.getState();
  assert.equal(state.phase, BOOT_PHASE.LAUNCHING_RUNTIME);
  assert.equal(state.assets.pgimUrl, '/data/boot/GTA6.pgim');
  assert.equal(state.status.ok, true);
  assert.equal(state.destroyed, false);
  assert.equal(state.tracer.stage, 'TLS_INIT');
  assert.equal(bootTracer.getSummary().milestonesReached >= 1, true);
  assert.ok(BOOT_MILESTONES.includes(state.tracer.stage));
});

test('W56 discovers programManifestUrl, gameBase, and bootMode from /api/status', async () => {
  assert.deepEqual(BOOT_MODE, {
    PENDING: 'pending',
    WASM_GUEST: 'wasm-guest',
    SMOKE_DEMO: 'smoke-demo',
  });
  assert.ok(Object.isFrozen(BOOT_MODE));

  const discovered = discoverBootAssets(
    readyStatus({
      programManifestUrl: '/boot/program-manifest.json',
      gameBase: '/game/',
    })
  );
  assert.equal(discovered.programManifestUrl, '/boot/program-manifest.json');
  assert.equal(discovered.gameBase, '/game/');
  assert.equal(discovered.hasWasmBinary, true);
  assert.equal(discovered.bootMode, 'wasm-guest');
  assert.equal(isRealGuestBoot(discovered), true);

  const objectManifest = discoverBootAssets({
    ok: true,
    hasWasmBinary: true,
    programManifest: { url: '/boot/program-manifest.json' },
    gameBase: '  /game/  ',
  });
  assert.equal(objectManifest.programManifestUrl, '/boot/program-manifest.json');
  assert.equal(objectManifest.gameBase, '/game/');
  assert.equal(objectManifest.bootMode, 'wasm-guest');

  const absent = discoverBootAssets({ ok: true, hasWasmBinary: false });
  assert.equal(absent.programManifestUrl, null);
  assert.equal(absent.gameBase, null);
  assert.equal(absent.hasWasmBinary, false);
  assert.equal(absent.bootMode, 'smoke-demo');
  assert.equal(isRealGuestBoot(absent), false);

  const phaseEvents = [];
  const fetchImpl = createMockFetch(
    readyStatus({
      programManifestUrl: '/boot/program-manifest.json',
      gameBase: '/game/',
    })
  );
  const { orchestrator, launches } = createOrchestrator({
    fetchImpl,
    onPhaseChange: (event) => phaseEvents.push(event),
  });

  // Before discovery completes, bootMode is pending and discovered URLs are null.
  const initialState = orchestrator.getState();
  assert.equal(initialState.bootMode, 'pending');
  assert.equal(initialState.programManifestUrl, null);
  assert.equal(initialState.gameBase, null);
  assert.equal(initialState.firstPresentSource, null);
  assert.equal(orchestrator.isRealGuestBoot(), false);
  assert.equal(isRealGuestBoot(orchestrator), false);

  const result = await orchestrator.start();
  assert.equal(result.ok, true);
  assert.equal(result.bootMode, 'wasm-guest');
  assert.equal(result.programManifestUrl, '/boot/program-manifest.json');
  assert.equal(result.gameBase, '/game/');

  assert.equal(launches.length, 1);
  assert.equal(launches[0].bootMode, 'wasm-guest');
  assert.equal(launches[0].programManifestUrl, '/boot/program-manifest.json');
  assert.equal(launches[0].gameBase, '/game/');
  assert.equal(launches[0].assets.programManifestUrl, '/boot/program-manifest.json');
  assert.equal(launches[0].assets.gameBase, '/game/');
  assert.equal(launches[0].status.programManifestUrl, '/boot/program-manifest.json');
  assert.equal(launches[0].status.gameBase, '/game/');

  const state = orchestrator.getState();
  assert.equal(state.bootMode, 'wasm-guest');
  assert.equal(state.programManifestUrl, '/boot/program-manifest.json');
  assert.equal(state.gameBase, '/game/');
  assert.equal(state.assets.programManifestUrl, '/boot/program-manifest.json');
  assert.equal(state.assets.gameBase, '/game/');
  assert.equal(state.assets.bootMode, 'wasm-guest');
  assert.equal(state.status.programManifestUrl, '/boot/program-manifest.json');
  assert.equal(state.status.gameBase, '/game/');
  assert.equal(state.status.bootMode, 'wasm-guest');
  assert.equal(orchestrator.isRealGuestBoot(), true);
  assert.equal(isRealGuestBoot(orchestrator), true);
  assert.equal(isRealGuestBoot(state), true);

  const launchPhaseEvent = phaseEvents.find((ev) => ev.phase === BOOT_PHASE.LAUNCHING_RUNTIME);
  assert.ok(launchPhaseEvent);
  assert.equal(launchPhaseEvent.bootMode, 'wasm-guest');
  assert.equal(launchPhaseEvent.programManifestUrl, '/boot/program-manifest.json');
  assert.equal(launchPhaseEvent.gameBase, '/game/');
});

test('W56 launches real wasm-guest mode when programManifestUrl is present without single pgimUrl', async () => {
  const fetchImpl = createMockFetch({
    ok: true,
    hasWasmBinary: true,
    splashImage: '/game/xbs_splash.png',
    programManifestUrl: '/boot/program-manifest.json',
    gameBase: '/game/',
    pgimUrl: null,
    shaderPackUrl: null,
  });
  const preloader = createPreloader();
  const { orchestrator, launches, fallbacks } = createOrchestrator({ fetchImpl, preloader });

  const result = await orchestrator.start();
  assert.equal(result.ok, true);
  assert.equal(result.phase, BOOT_PHASE.LAUNCHING_RUNTIME);
  assert.equal(result.bootMode, 'wasm-guest');
  assert.equal(orchestrator.isRealGuestBoot(), true);
  assert.equal(launches.length, 1);
  assert.equal(fallbacks.length, 0);
  assert.equal(preloader.calls.length, 0);
  assert.equal(orchestrator.getState().programManifestUrl, '/boot/program-manifest.json');
  assert.equal(orchestrator.getState().gameBase, '/game/');
});

test('W56 tracks bootMode pending -> wasm-guest vs smoke-demo and tags firstPresentSource accurately', async () => {
  // 1. Real Wasm guest boot via FIRST_PRESENT runtime log
  let resolveFetch;
  const deferredFetch = async () =>
    new Promise((resolve) => {
      resolveFetch = () =>
        resolve({
          ok: true,
          status: 200,
          async json() {
            return readyStatus({
              programManifestUrl: '/boot/program-manifest.json',
              gameBase: '/game/',
            });
          },
        });
    });

  const guest = createOrchestrator({ fetchImpl: deferredFetch });
  const pendingStart = guest.orchestrator.start();
  // While discovery is in flight, bootMode remains pending and isRealGuestBoot() is false.
  assert.equal(guest.orchestrator.getState().phase, BOOT_PHASE.DISCOVERING);
  assert.equal(guest.orchestrator.getState().bootMode, 'pending');
  assert.equal(guest.orchestrator.isRealGuestBoot(), false);

  resolveFetch();
  await pendingStart;
  assert.equal(guest.orchestrator.getState().bootMode, 'wasm-guest');
  assert.equal(guest.orchestrator.isRealGuestBoot(), true);

  guest.orchestrator.feedRuntimeLog('[boot] FIRST_PRESENT');
  assert.equal(guest.orchestrator.getState().phase, BOOT_PHASE.PRESENTING);
  assert.equal(guest.orchestrator.getState().firstPresent, true);
  assert.equal(guest.orchestrator.getState().firstPresentSource, 'wasm-guest');
  assert.equal(guest.orchestrator.isRealGuestBoot(), true);
  assert.equal(guest.firstPresents.length, 1);
  assert.equal(guest.firstPresents[0].firstPresentSource, 'wasm-guest');
  assert.equal(guest.firstPresents[0].bootMode, 'wasm-guest');
  assert.equal(guest.firstPresents[0].isRealGuestBoot, true);
  assert.equal(guest.overlay.dismissals.length, 1);
  assert.equal(guest.overlay.dismissals[0].firstPresentSource, 'wasm-guest');

  // 2. Real Wasm guest boot via markFirstPresent()
  const guestMark = createOrchestrator();
  await guestMark.orchestrator.start();
  const markRes = guestMark.orchestrator.markFirstPresent({ frameIndex: 1 });
  assert.equal(markRes.ok, true);
  assert.equal(markRes.alreadyPresented, false);
  assert.equal(markRes.bootMode, 'wasm-guest');
  assert.equal(markRes.firstPresentSource, 'wasm-guest');
  assert.equal(guestMark.orchestrator.getState().firstPresentSource, 'wasm-guest');
  assert.equal(guestMark.bootTracer.getSummary().readyForSplash, true);
  // Duplicate call is idempotent and preserves firstPresentSource.
  const dupRes = guestMark.orchestrator.markFirstPresent({ source: 'smoke_demo' });
  assert.equal(dupRes.alreadyPresented, true);
  assert.equal(dupRes.firstPresentSource, 'wasm-guest');
  assert.equal(guestMark.orchestrator.getState().firstPresentSource, 'wasm-guest');

  // 3. Fallback to Smoke Demo (no runtime.wasm) -> bootMode === 'smoke-demo'
  const smokeFetch = createMockFetch(
    readyStatus({
      hasWasmBinary: false,
      runtimeWasm: null,
      programManifestUrl: null,
      gameBase: '/game/',
    })
  );
  const smoke = createOrchestrator({ fetchImpl: smokeFetch });
  const smokeResult = await smoke.orchestrator.start();
  assert.equal(smokeResult.ok, true);
  assert.equal(smokeResult.fallback, true);
  assert.equal(smokeResult.bootMode, 'smoke-demo');
  assert.equal(smoke.orchestrator.getState().bootMode, 'smoke-demo');
  assert.equal(smoke.orchestrator.getState().gameBase, '/game/');
  assert.equal(smoke.orchestrator.getState().programManifestUrl, null);
  assert.equal(smoke.orchestrator.isRealGuestBoot(), false);
  assert.equal(isRealGuestBoot(smoke.orchestrator), false);
  assert.equal(smoke.fallbacks.length, 1);
  assert.equal(smoke.fallbacks[0].bootMode, 'smoke-demo');

  const smokePresent = smoke.orchestrator.markFirstPresent({ source: 'smoke_demo' });
  assert.equal(smokePresent.ok, true);
  assert.equal(smokePresent.alreadyPresented, false);
  assert.equal(smokePresent.bootMode, 'smoke-demo');
  assert.equal(smokePresent.firstPresentSource, 'smoke-demo');
  assert.equal(smoke.orchestrator.getState().firstPresent, true);
  assert.equal(smoke.orchestrator.getState().firstPresentSource, 'smoke-demo');
  assert.equal(smoke.orchestrator.isRealGuestBoot(), false);
  assert.equal(smoke.firstPresents.length, 1);
  assert.equal(smoke.firstPresents[0].firstPresentSource, 'smoke-demo');
  assert.equal(smoke.firstPresents[0].isRealGuestBoot, false);

  // 4. Discovery failure keeps bootMode === 'pending' and isRealGuestBoot() === false.
  const errFetch = createMockFetch(null, { throwError: new TypeError('offline') });
  const failed = createOrchestrator({ fetchImpl: errFetch });
  const failRes = await failed.orchestrator.start();
  assert.equal(failRes.ok, false);
  assert.equal(failRes.bootMode, 'pending');
  assert.equal(failed.orchestrator.getState().bootMode, 'pending');
  assert.equal(failed.orchestrator.isRealGuestBoot(), false);
});

test('W146 validates nowFn, stageTimeoutMs, and onStageStall options and exports DEFAULT_STAGE_TIMEOUT_MS', () => {
  assert.equal(DEFAULT_STAGE_TIMEOUT_MS, 0);
  assert.throws(() => new BootOrchestrator({ nowFn: 123 }), TypeError);
  assert.throws(() => new BootOrchestrator({ nowFn: 'clock' }), TypeError);
  assert.throws(() => new BootOrchestrator({ onStageStall: {} }), TypeError);
  assert.throws(() => new BootOrchestrator({ stageTimeoutMs: '500' }), TypeError);
  assert.throws(() => new BootOrchestrator({ stageTimeoutMs: -1 }), RangeError);
  assert.throws(() => new BootOrchestrator({ stageTimeoutMs: Number.NaN }), RangeError);
  assert.throws(() => new BootOrchestrator({ stageTimeoutMs: Number.POSITIVE_INFINITY }), RangeError);
  assert.throws(() => new BootOrchestrator({ stageTimeoutMs: Number.NEGATIVE_INFINITY }), RangeError);

  const defaultOrch = new BootOrchestrator({ fetchImpl: async () => ({}) });
  const initialState = defaultOrch.getState();
  assert.deepEqual(initialState.stageDurationsMs, {});
  assert.equal(initialState.currentStageElapsedMs, 0);
  assert.equal(initialState.totalStageDurationMs, 0);
  assert.equal(initialState.stageTimeoutMs, 0);
  assert.equal(initialState.isStalled, false);
  assert.equal(initialState.stalledStage, null);
  assert.equal(initialState.stalledMs, 0);
  assert.equal(initialState.stallCount, 0);

  const nullOptsOrch = new BootOrchestrator({
    fetchImpl: async () => ({}),
    nowFn: null,
    stageTimeoutMs: null,
    onStageStall: null,
  });
  assert.equal(nullOptsOrch.getState().stageTimeoutMs, 0);
});

test('W146 tracks per-stage duration timing breakdown across transitions with injectable nowFn', async () => {
  let now = 1000;
  const nowFn = () => now;

  let resolveFetch;
  const deferredFetch = async () =>
    new Promise((resolve) => {
      resolveFetch = () =>
        resolve({
          ok: true,
          status: 200,
          async json() {
            return readyStatus();
          },
        });
    });

  let resolvePreload;
  const deferredPreloader = {
    preloadBootBundle: async () =>
      new Promise((resolve) => {
        resolvePreload = () =>
          resolve({
            pgim: { header: { magic: 'PGIM' }, bytesLoaded: 64 },
            shaderPackBytes: new Uint8Array([1, 2, 3]),
            reflectionManifest: { schema_version: 1 },
            bytesLoaded: 64,
          });
      }),
  };

  const phaseSnapshots = [];
  const { orchestrator } = createOrchestrator({
    fetchImpl: deferredFetch,
    preloader: deferredPreloader,
    nowFn,
    onPhaseChange: (event) => phaseSnapshots.push({ phase: event.phase, durations: event.stageDurationsMs }),
  });

  // Idle time before start() does not pollute boot stage durations.
  now = 2500;
  assert.deepEqual(orchestrator.getState().stageDurationsMs, {});
  assert.equal(orchestrator.getState().currentStageElapsedMs, 0);
  assert.equal(orchestrator.getState().totalStageDurationMs, 0);

  // Enter DISCOVERING at t=2500.
  const startPromise = orchestrator.start();
  assert.equal(orchestrator.getState().phase, BOOT_PHASE.DISCOVERING);

  now = 2540;
  assert.deepEqual(orchestrator.getState().stageDurationsMs, {});
  assert.equal(orchestrator.getState().currentStageElapsedMs, 40);
  assert.equal(orchestrator.getState().totalStageDurationMs, 40);

  // Finish DISCOVERING at t=2620 (120ms) -> enters PRELOADING.
  now = 2620;
  resolveFetch();
  // Wait for microtasks to transition into PRELOADING.
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(orchestrator.getState().phase, BOOT_PHASE.PRELOADING);
  assert.deepEqual(orchestrator.getState().stageDurationsMs, { DISCOVERING: 120 });
  assert.equal(orchestrator.getState().currentStageElapsedMs, 0);

  // Mid-PRELOADING at t=2820 (200ms elapsed).
  now = 2820;
  assert.equal(orchestrator.getState().currentStageElapsedMs, 200);
  assert.equal(orchestrator.getState().totalStageDurationMs, 320);

  // Finish PRELOADING at t=2970 (350ms) -> enters LAUNCHING_RUNTIME.
  now = 2970;
  resolvePreload();
  const result = await startPromise;
  assert.equal(result.ok, true);
  assert.equal(orchestrator.getState().phase, BOOT_PHASE.LAUNCHING_RUNTIME);
  assert.deepEqual(orchestrator.getState().stageDurationsMs, {
    DISCOVERING: 120,
    PRELOADING: 350,
  });
  assert.equal(orchestrator.getState().currentStageElapsedMs, 0);

  // Reach FIRST_PRESENT at t=3470 (500ms in LAUNCHING_RUNTIME) -> enters PRESENTING.
  now = 3470;
  orchestrator.notifyFirstPresent({ frameIndex: 1 });
  assert.equal(orchestrator.getState().phase, BOOT_PHASE.PRESENTING);
  const presentedState = orchestrator.getState();
  assert.deepEqual(presentedState.stageDurationsMs, {
    DISCOVERING: 120,
    PRELOADING: 350,
    LAUNCHING_RUNTIME: 500,
  });
  assert.equal(presentedState.currentStageElapsedMs, 0);
  assert.equal(presentedState.totalStageDurationMs, 970);

  // Defensive clone: mutating returned stageDurationsMs does not corrupt internal state.
  presentedState.stageDurationsMs.DISCOVERING = 999999;
  assert.equal(orchestrator.getState().stageDurationsMs.DISCOVERING, 120);

  // Advancing clock while PRESENTING does not accumulate or mark stalled.
  now = 99999;
  assert.deepEqual(orchestrator.getState().stageDurationsMs, {
    DISCOVERING: 120,
    PRELOADING: 350,
    LAUNCHING_RUNTIME: 500,
  });
  assert.equal(orchestrator.getState().currentStageElapsedMs, 0);
  assert.equal(orchestrator.getState().totalStageDurationMs, 970);

  // Phase change events carry stageDurationsMs snapshots.
  const presentingEvent = phaseSnapshots.find((entry) => entry.phase === BOOT_PHASE.PRESENTING);
  assert.ok(presentingEvent);
  assert.deepEqual(presentingEvent.durations, {
    DISCOVERING: 120,
    PRELOADING: 350,
    LAUNCHING_RUNTIME: 500,
  });
});

test('W146 tracks smoke-demo fallback, error, abort, and destroy stage durations cleanly', async () => {
  // 1. Smoke-demo fallback path.
  let now = 100;
  const smokeFetch = async () => {
    now += 45;
    return {
      ok: true,
      status: 200,
      async json() {
        return readyStatus({ hasWasmBinary: false, runtimeWasm: null, pgimUrl: null, shaderPackUrl: null });
      },
    };
  };
  const smoke = createOrchestrator({ fetchImpl: smokeFetch, nowFn: () => now, stageTimeoutMs: 500 });
  await smoke.orchestrator.start();
  assert.equal(smoke.orchestrator.getState().phase, BOOT_PHASE.FALLBACK_SMOKE_DEMO);
  assert.deepEqual(smoke.orchestrator.getState().stageDurationsMs, { DISCOVERING: 45 });
  now += 80;
  smoke.orchestrator.markFirstPresent({ source: 'smoke-demo' });
  assert.deepEqual(smoke.orchestrator.getState().stageDurationsMs, {
    DISCOVERING: 45,
    FALLBACK_SMOKE_DEMO: 80,
  });
  assert.equal(smoke.orchestrator.getState().totalStageDurationMs, 125);

  // 2. Network error during DISCOVERING.
  let errNow = 500;
  const errFetch = async () => {
    errNow += 75;
    throw new TypeError('network down');
  };
  const failed = createOrchestrator({ fetchImpl: errFetch, nowFn: () => errNow, stageTimeoutMs: 50 });
  await failed.orchestrator.start();
  assert.equal(failed.orchestrator.getState().phase, BOOT_PHASE.ERROR);
  assert.deepEqual(failed.orchestrator.getState().stageDurationsMs, { DISCOVERING: 75 });
  errNow += 10000;
  assert.equal(failed.orchestrator.getState().isStalled, false);
  assert.equal(failed.orchestrator.getState().stalledStage, null);
  assert.equal(failed.orchestrator.getState().stalledMs, 0);

  // 3. Abort during DISCOVERING.
  let abortNow = 200;
  const controller = new AbortController();
  const abortFetch = async (_url, { signal }) =>
    new Promise((_resolve, reject) => {
      signal.addEventListener(
        'abort',
        () => {
          const err = new Error('The operation was aborted.');
          err.name = 'AbortError';
          reject(err);
        },
        { once: true }
      );
    });
  const cancelled = createOrchestrator({ fetchImpl: abortFetch, nowFn: () => abortNow, stageTimeoutMs: 100 });
  const pendingCancel = cancelled.orchestrator.start({ signal: controller.signal });
  abortNow += 60;
  controller.abort();
  await pendingCancel;
  assert.equal(cancelled.orchestrator.getState().phase, BOOT_PHASE.CANCELLED);
  assert.deepEqual(cancelled.orchestrator.getState().stageDurationsMs, { DISCOVERING: 60 });
  abortNow += 5000;
  assert.equal(cancelled.orchestrator.getState().isStalled, false);

  // 4. Destroy during DISCOVERING.
  let destroyNow = 300;
  const destroyed = createOrchestrator({ fetchImpl: abortFetch, nowFn: () => destroyNow, stageTimeoutMs: 100 });
  const pendingDestroy = destroyed.orchestrator.start();
  destroyNow += 30;
  destroyed.orchestrator.destroy();
  await pendingDestroy;
  assert.equal(destroyed.orchestrator.getState().phase, BOOT_PHASE.DESTROYED);
  assert.deepEqual(destroyed.orchestrator.getState().stageDurationsMs, { DISCOVERING: 30 });
  destroyNow += 5000;
  assert.equal(destroyed.orchestrator.getState().isStalled, false);
});

test('W146 stalled-stage watchdog stays disabled when stageTimeoutMs=0 and detects/recovers cleanly when enabled', async () => {
  // 1. Disabled watchdog (stageTimeoutMs = 0).
  let disabledNow = 0;
  let resolveDisabledFetch;
  const disabledFetch = async () =>
    new Promise((resolve) => {
      resolveDisabledFetch = () =>
        resolve({
          ok: true,
          status: 200,
          async json() {
            return readyStatus();
          },
        });
    });
  const disabled = createOrchestrator({ fetchImpl: disabledFetch, nowFn: () => disabledNow, stageTimeoutMs: 0 });
  const disabledStart = disabled.orchestrator.start();
  disabledNow = 600_000;
  const disabledCheck = disabled.orchestrator.checkStageStall();
  assert.equal(disabledCheck.isStalled, false);
  assert.equal(disabledCheck.stalledStage, null);
  assert.equal(disabledCheck.stalledMs, 0);
  assert.equal(disabledCheck.currentStageElapsedMs, 600_000);
  assert.equal(disabled.orchestrator.getState().isStalled, false);
  resolveDisabledFetch();
  await disabledStart;

  // 2. Enabled watchdog (stageTimeoutMs = 500) with non-destructive recovery and onStageStall callback.
  let now = 0;
  let resolveFetch;
  const deferredFetch = async () =>
    new Promise((resolve) => {
      resolveFetch = () =>
        resolve({
          ok: true,
          status: 200,
          async json() {
            return readyStatus();
          },
        });
    });
  let resolvePreload;
  const deferredPreloader = {
    preloadBootBundle: async () =>
      new Promise((resolve) => {
        resolvePreload = () =>
          resolve({
            pgim: { header: { magic: 'PGIM' }, bytesLoaded: 64 },
            shaderPackBytes: new Uint8Array([1, 2, 3]),
            reflectionManifest: { schema_version: 1 },
            bytesLoaded: 64,
          });
      }),
  };

  const stallEvents = [];
  const { orchestrator } = createOrchestrator({
    fetchImpl: deferredFetch,
    preloader: deferredPreloader,
    nowFn: () => now,
    stageTimeoutMs: 500,
    onStageStall: (event) => {
      stallEvents.push(event);
      // Verify throwing inside onStageStall never escapes checkStageStall().
      if (stallEvents.length === 1) {
        throw new Error('listener boom');
      }
    },
  });

  // Sitting in IDLE past stageTimeoutMs is never a stall.
  now = 5000;
  const idleCheck = orchestrator.checkStageStall();
  assert.equal(idleCheck.isStalled, false);
  assert.equal(idleCheck.stalledStage, null);
  assert.equal(idleCheck.stalledMs, 0);
  assert.equal(stallEvents.length, 0);

  // Start DISCOVERING at t=5000.
  const startPromise = orchestrator.start();

  // Just below threshold (499ms).
  now = 5499;
  assert.equal(orchestrator.getState().isStalled, false);
  assert.equal(orchestrator.checkStageStall().isStalled, false);
  assert.equal(stallEvents.length, 0);

  // At threshold (500ms) -> stalled!
  now = 5500;
  const liveState = orchestrator.getState();
  assert.equal(liveState.isStalled, true);
  assert.equal(liveState.stalledStage, BOOT_PHASE.DISCOVERING);
  assert.equal(liveState.stalledMs, 500);
  assert.equal(liveState.stageTimeoutMs, 500);

  const stall1 = orchestrator.checkStageStall();
  assert.equal(stall1.isStalled, true);
  assert.equal(stall1.stalledStage, BOOT_PHASE.DISCOVERING);
  assert.equal(stall1.stalledMs, 500);
  assert.equal(stallEvents.length, 1);
  assert.equal(orchestrator.getState().stallCount, 1);

  // Repeated checkStageStall() while still in DISCOVERING updates stalledMs without duplicate callback.
  now = 5750;
  const stall1Again = orchestrator.checkStageStall();
  assert.equal(stall1Again.isStalled, true);
  assert.equal(stall1Again.stalledMs, 750);
  assert.equal(stallEvents.length, 1);
  assert.equal(orchestrator.getState().stallCount, 1);

  // Stage recovers cleanly and transitions to PRELOADING at t=5800 (800ms total in DISCOVERING).
  now = 5800;
  resolveFetch();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(orchestrator.getState().phase, BOOT_PHASE.PRELOADING);
  assert.equal(orchestrator.getState().isStalled, false);
  assert.equal(orchestrator.getState().stalledStage, null);
  assert.equal(orchestrator.getState().stalledMs, 0);
  assert.deepEqual(orchestrator.getState().stageDurationsMs, { DISCOVERING: 800 });

  // Stall again in PRELOADING at t=6400 (600ms elapsed).
  now = 6400;
  const stall2 = orchestrator.checkStageStall();
  assert.equal(stall2.isStalled, true);
  assert.equal(stall2.stalledStage, BOOT_PHASE.PRELOADING);
  assert.equal(stall2.stalledMs, 600);
  assert.equal(stallEvents.length, 2);
  assert.equal(orchestrator.getState().stallCount, 2);

  // Complete preload and present -> clears stall and preserves full timing breakdown.
  now = 6450;
  resolvePreload();
  await startPromise;
  now = 6550;
  orchestrator.notifyFirstPresent();
  const finalState = orchestrator.getState();
  assert.equal(finalState.phase, BOOT_PHASE.PRESENTING);
  assert.equal(finalState.isStalled, false);
  assert.equal(finalState.stalledStage, null);
  assert.equal(finalState.stalledMs, 0);
  assert.deepEqual(finalState.stageDurationsMs, {
    DISCOVERING: 800,
    PRELOADING: 650,
    LAUNCHING_RUNTIME: 100,
  });
  assert.equal(finalState.stallCount, 2);
});

test('W146 tolerates backwards, non-finite, and throwing nowFn readings without producing NaN or negative ms', async () => {
  const readings = [1000, 1000, 400, Number.NaN, Number.POSITIVE_INFINITY, 'bad', () => { throw new Error('clock fail'); }, 1250];
  let idx = 0;
  const nowFn = () => {
    const entry = readings[Math.min(idx++, readings.length - 1)];
    return typeof entry === 'function' ? entry() : entry;
  };

  let resolveFetch;
  const deferredFetch = async () =>
    new Promise((resolve) => {
      resolveFetch = () =>
        resolve({
          ok: true,
          status: 200,
          async json() {
            return readyStatus({ hasWasmBinary: false, runtimeWasm: null, pgimUrl: null, shaderPackUrl: null });
          },
        });
    });

  const { orchestrator } = createOrchestrator({ fetchImpl: deferredFetch, nowFn, stageTimeoutMs: 100 });
  const pending = orchestrator.start();

  // Backwards reading (400 < 1000), NaN, Infinity, non-number, and throwing clock all stay finite >= 0.
  for (let i = 0; i < 5; i += 1) {
    const state = orchestrator.getState();
    assert.equal(Number.isFinite(state.currentStageElapsedMs), true);
    assert.equal(state.currentStageElapsedMs >= 0, true);
    assert.equal(Object.is(state.currentStageElapsedMs, -0), false);
    assert.equal(Number.isFinite(state.stalledMs), true);
    assert.equal(state.stalledMs >= 0, true);
  }

  resolveFetch();
  await pending;
  const after = orchestrator.getState();
  assert.equal(Number.isFinite(after.stageDurationsMs.DISCOVERING), true);
  assert.equal(after.stageDurationsMs.DISCOVERING >= 0, true);
});

test('W146 reset() clears stage durations, watchdog state, and in-flight work cleanly', async () => {
  let now = 100;
  let resolveSlowFetch;
  let fetchCount = 0;
  const fetchImpl = async (_url, { signal }) => {
    fetchCount += 1;
    if (fetchCount === 1) {
      return new Promise((resolve, reject) => {
        resolveSlowFetch = () =>
          resolve({
            ok: true,
            status: 200,
            async json() {
              return readyStatus();
            },
          });
        signal.addEventListener(
          'abort',
          () => {
            const err = new Error('The operation was aborted.');
            err.name = 'AbortError';
            reject(err);
          },
          { once: true }
        );
      });
    }
    now += 25;
    return {
      ok: true,
      status: 200,
      async json() {
        return readyStatus({ pgimUrl: null, shaderPackUrl: null, programManifestUrl: '/boot/program-manifest.json' });
      },
    };
  };

  const { orchestrator } = createOrchestrator({ fetchImpl, nowFn: () => now, stageTimeoutMs: 200 });

  // 1. Reset during an in-flight stalled DISCOVERING stage.
  const firstRun = orchestrator.start();
  now = 450; // 350ms >= 200ms -> stalled
  assert.equal(orchestrator.checkStageStall().isStalled, true);
  assert.equal(orchestrator.getState().stallCount, 1);

  const resetRes = orchestrator.reset();
  assert.equal(resetRes.ok, true);
  assert.equal(resetRes.phase, BOOT_PHASE.IDLE);
  assert.equal(resetRes.bootMode, BOOT_MODE.PENDING);

  // Superseded firstRun settles without clobbering reset IDLE state.
  const firstRunRes = await firstRun;
  assert.equal(firstRunRes.ok, false);
  assert.equal(firstRunRes.aborted, true);

  const afterReset = orchestrator.getState();
  assert.equal(afterReset.phase, BOOT_PHASE.IDLE);
  assert.equal(afterReset.bootMode, BOOT_MODE.PENDING);
  assert.equal(afterReset.aborted, false);
  assert.equal(afterReset.firstPresent, false);
  assert.equal(afterReset.error, null);
  assert.deepEqual(afterReset.stageDurationsMs, {});
  assert.equal(afterReset.currentStageElapsedMs, 0);
  assert.equal(afterReset.totalStageDurationMs, 0);
  assert.equal(afterReset.isStalled, false);
  assert.equal(afterReset.stalledStage, null);
  assert.equal(afterReset.stalledMs, 0);
  assert.equal(afterReset.stallCount, 0);

  // 2. Can start cleanly again after reset().
  const secondRun = await orchestrator.start();
  assert.equal(secondRun.ok, true);
  assert.equal(secondRun.phase, BOOT_PHASE.LAUNCHING_RUNTIME);
  assert.deepEqual(orchestrator.getState().stageDurationsMs, { DISCOVERING: 25 });
  assert.equal(orchestrator.getState().isStalled, false);

  // 3. Reset after PRESENTING returns cleanly to IDLE.
  now += 40;
  orchestrator.notifyFirstPresent();
  assert.deepEqual(orchestrator.getState().stageDurationsMs, {
    DISCOVERING: 25,
    LAUNCHING_RUNTIME: 40,
  });
  orchestrator.reset();
  assert.equal(orchestrator.getState().phase, BOOT_PHASE.IDLE);
  assert.deepEqual(orchestrator.getState().stageDurationsMs, {});
  assert.equal(orchestrator.getState().firstPresent, false);

  // 4. Reset after destroy() is a no-op.
  orchestrator.destroy();
  const postDestroyReset = orchestrator.reset();
  assert.equal(postDestroyReset.ok, false);
  assert.equal(postDestroyReset.destroyed, true);
  assert.equal(orchestrator.getState().phase, BOOT_PHASE.DESTROYED);
  void resolveSlowFetch;
});

test('W146 getDiagnosticSnapshot exports JSON-safe sanitized telemetry without leaking paths or secrets', async () => {
  let now = 1000;
  const sensitiveError = new Error(
    'Failed reading C:\\Users\\alice\\gta6\\game.bin and /Users/alice/secret.key from https://user:pass@example.com/api?token=supersecret (Bearer tok_12345 api_key=mykey)'
  );
  const errFetch = async () => {
    now += 90;
    throw sensitiveError;
  };
  const { orchestrator } = createOrchestrator({
    fetchImpl: errFetch,
    nowFn: () => now,
    stageTimeoutMs: 500,
  });

  await orchestrator.start();
  const snapshot = orchestrator.getDiagnosticSnapshot();

  assert.equal(snapshot.phase, BOOT_PHASE.ERROR);
  assert.equal(snapshot.bootMode, BOOT_MODE.PENDING);
  assert.deepEqual(snapshot.stageDurationsMs, { DISCOVERING: 90 });
  assert.equal(snapshot.totalStageDurationMs, 90);
  assert.equal(snapshot.stageTimeoutMs, 500);
  assert.equal(snapshot.isStalled, false);
  assert.equal(snapshot.stalledStage, null);
  assert.equal(snapshot.stalledMs, 0);
  assert.equal('bundle' in snapshot, false);
  assert.equal('status' in snapshot, false);

  // Verify sanitization removed host paths, credentials, query tokens, and auth secrets.
  assert.equal(snapshot.error.includes('alice'), false);
  assert.equal(snapshot.error.includes('supersecret'), false);
  assert.equal(snapshot.error.includes('tok_12345'), false);
  assert.equal(snapshot.error.includes('mykey'), false);
  assert.equal(snapshot.error.includes('user:pass'), false);
  assert.match(snapshot.error, /\[REDACTED_PATH\]/);
  assert.match(snapshot.error, /\[REDACTED\]/);

  // Must serialize cleanly to JSON without throwing.
  const roundTripped = JSON.parse(JSON.stringify(snapshot));
  assert.deepEqual(roundTripped.stageDurationsMs, { DISCOVERING: 90 });
});

// ---------------------------------------------------------------------------
// W225: Parallel preload coordination & phase timing telemetry
// ---------------------------------------------------------------------------

test('W225 validates parallelPreload and preloadTasks options and discovers sidecar URLs', () => {
  assert.equal(DEFAULT_PARALLEL_PRELOAD, true);

  assert.throws(() => new BootOrchestrator({ parallelPreload: 'yes' }), TypeError);
  assert.throws(() => new BootOrchestrator({ parallelPreload: 1 }), TypeError);
  assert.throws(() => new BootOrchestrator({ preloadTasks: 'nope' }), TypeError);
  assert.throws(() => new BootOrchestrator({ preloadTasks: [null] }), TypeError);
  assert.throws(() => new BootOrchestrator({ preloadTasks: [{ name: '', run: () => {} }] }), TypeError);
  assert.throws(() => new BootOrchestrator({ preloadTasks: [{ name: '__proto__', run: () => {} }] }), TypeError);
  assert.throws(() => new BootOrchestrator({ preloadTasks: [{ name: 'a', run: 'bad' }] }), TypeError);
  assert.throws(() => new BootOrchestrator({ preloadTasks: [{ name: 'a', run: () => {}, optional: 'yes' }] }), TypeError);
  assert.throws(
    () =>
      new BootOrchestrator({
        preloadTasks: [
          { name: 'dup', run: () => {} },
          { name: 'dup', run: () => {} },
        ],
      }),
    TypeError
  );

  const discovered = discoverBootAssets(
    readyStatus({
      shaderPackHeaderUrl: '/data/shaders/boot.wgsl.idx',
      stageReflectionSidecarUrl: '/data/shaders/stage-reflection.json',
    })
  );
  assert.equal(discovered.shaderPackIndexUrl, '/data/shaders/boot.wgsl.idx');
  assert.equal(discovered.reflectionSidecarUrl, '/data/shaders/stage-reflection.json');

  const withoutSidecars = discoverBootAssets(readyStatus());
  assert.equal(withoutSidecars.shaderPackIndexUrl, null);
  assert.equal(withoutSidecars.reflectionSidecarUrl, null);

  const orch = new BootOrchestrator({ fetchImpl: async () => ({}) });
  const stats = orch.getTimingStats();
  assert.equal(stats.parallelPreloadEnabled, true);
  assert.equal(stats.parallelPreloadSavedMs, 0);
  assert.equal(stats.totalBootMs, 0);
  assert.deepEqual(stats.phaseTimingsMs, {});
  assert.deepEqual(stats.preloadTaskTimingsMs, {});
});

test('W225 overlaps independent pgim, shader pack, header/index, and reflection sidecar preloads with deterministic phases and savings stats', async () => {
  let now = 1000;
  const nowFn = () => now;

  const events = [];
  let inFlight = 0;
  let peakInFlight = 0;
  const resolvers = {};

  function makeDeferredTask(name, resultFn) {
    return (url, { onProgress, signal } = {}) => {
      events.push(`start:${name}`);
      inFlight += 1;
      peakInFlight = Math.max(peakInFlight, inFlight);
      return new Promise((resolve, reject) => {
        const onAbort = () => {
          inFlight -= 1;
          const err = new Error(`Aborted ${name}`);
          err.name = 'AbortError';
          reject(err);
        };
        if (signal) {
          if (signal.aborted) {
            onAbort();
            return;
          }
          signal.addEventListener('abort', onAbort, { once: true });
        }
        resolvers[name] = () => {
          if (signal) signal.removeEventListener('abort', onAbort);
          inFlight -= 1;
          events.push(`end:${name}`);
          if (typeof onProgress === 'function') {
            onProgress({ phase: 'complete', loadedBytes: 16, totalBytes: 16, pct: 100 });
          }
          resolve(resultFn(url));
        };
      });
    };
  }

  const granularPreloader = {
    preloadPgim: makeDeferredTask('pgim', (url) => ({ header: { magic: 'PGIM' }, url, bytesLoaded: 64 })),
    preloadShaderPack: makeDeferredTask('shaderPack', (url) => ({ bytes: new Uint8Array([1, 2, 3]), url, bytesLoaded: 32 })),
    preloadShaderPackHeader: makeDeferredTask('shaderPackIndex', (url) => ({ entryCount: 12, url, bytesLoaded: 16 })),
    preloadReflectionManifest: makeDeferredTask('reflection', (url) => ({ schema_version: 1, url, bytesLoaded: 24 })),
    preloadReflectionSidecar: makeDeferredTask('reflectionSidecar', (url) => ({ stages: 4, url, bytesLoaded: 20 })),
  };

  const fetchImpl = async () => {
    now += 50; // DISCOVERING = 50ms (1000 -> 1050)
    return {
      ok: true,
      status: 200,
      async json() {
        return readyStatus({
          shaderPackIndexUrl: '/data/shaders/boot.wgsl.idx',
          reflectionSidecarUrl: '/data/shaders/stage-reflection.json',
        });
      },
    };
  };

  const phaseEvents = [];
  const { orchestrator, phases, launches, bootTracer } = createOrchestrator({
    fetchImpl,
    preloader: granularPreloader,
    nowFn,
    parallelPreload: true,
    onPhaseChange: (ev) => {
      phases.push(ev.phase);
      phaseEvents.push(ev);
    },
  });

  const pendingStart = orchestrator.start();
  await new Promise((r) => setTimeout(r, 0));

  // All 5 independent preload tasks must be in flight concurrently!
  assert.equal(orchestrator.getState().phase, BOOT_PHASE.PRELOADING);
  assert.equal(peakInFlight, 5);
  assert.equal(inFlight, 5);
  assert.equal(launches.length, 0);

  // Resolve out-of-order at distinct deterministic timestamps, flushing microtasks between settlements.
  const flush = () => new Promise((r) => setTimeout(r, 0));
  now = 1090; // reflection: 40ms
  resolvers.reflection();
  await flush();
  now = 1110; // reflectionSidecar: 60ms
  resolvers.reflectionSidecar();
  await flush();
  now = 1130; // shaderPackIndex: 80ms
  resolvers.shaderPackIndex();
  await flush();
  now = 1150; // shaderPack: 100ms
  resolvers.shaderPack();
  await flush();
  now = 1200; // pgim: 150ms (wallTimeMs = 150ms, sequential sum = 430ms, saved = 280ms)
  resolvers.pgim();

  const result = await pendingStart;
  assert.equal(result.ok, true);
  assert.equal(result.phase, BOOT_PHASE.LAUNCHING_RUNTIME);
  assert.deepEqual(phases, [BOOT_PHASE.DISCOVERING, BOOT_PHASE.PRELOADING, BOOT_PHASE.LAUNCHING_RUNTIME]);
  assert.equal(launches.length, 1);
  assert.equal(launches[0].bundle.pgim.header.magic, 'PGIM');
  assert.equal(launches[0].bundle.shaderPackIndex.entryCount, 12);
  assert.equal(launches[0].bundle.reflectionManifest.schema_version, 1);
  assert.equal(launches[0].bundle.reflectionSidecar.stages, 4);
  assert.equal(launches[0].bundle.bytesLoaded, 64 + 32 + 16 + 24 + 20);
  assert.equal(bootTracer.getSummary().milestones.find((m) => m.milestone === 'PGIM_LOAD').reached, true);

  // Verify timing stats on start() return, getState(), and getTimingStats().
  assert.deepEqual(result.phaseTimingsMs, { DISCOVERING: 50, PRELOADING: 150 });
  assert.equal(result.parallelPreloadSavedMs, 280);
  assert.equal(result.totalBootMs, 200);

  // Advance clock into LAUNCHING_RUNTIME and reach FIRST_PRESENT.
  now = 1300; // LAUNCHING_RUNTIME = 100ms
  orchestrator.notifyFirstPresent({ frameIndex: 1 });
  assert.deepEqual(phases, [
    BOOT_PHASE.DISCOVERING,
    BOOT_PHASE.PRELOADING,
    BOOT_PHASE.LAUNCHING_RUNTIME,
    BOOT_PHASE.PRESENTING,
  ]);

  const stats = orchestrator.getTimingStats();
  assert.deepEqual(stats.phaseTimingsMs, {
    DISCOVERING: 50,
    PRELOADING: 150,
    LAUNCHING_RUNTIME: 100,
  });
  assert.equal(stats.totalBootMs, 300);
  assert.equal(stats.parallelPreloadEnabled, true);
  assert.equal(stats.parallelPreloadSavedMs, 280);
  assert.deepEqual(stats.preloadTaskTimingsMs, {
    pgim: 150,
    shaderPack: 100,
    shaderPackIndex: 80,
    reflection: 40,
    reflectionSidecar: 60,
  });
  assert.equal(stats.preloadStats.maxConcurrency, 5);
  assert.equal(stats.preloadStats.taskCount, 5);
  assert.equal(stats.preloadStats.completedCount, 5);
  assert.equal(stats.preloadStats.optionalMissingCount, 0);
  assert.equal(stats.preloadStats.wallTimeMs, 150);
  assert.equal(stats.preloadStats.sequentialTimeMs, 430);

  const presentingEvent = phaseEvents.find((ev) => ev.phase === BOOT_PHASE.PRESENTING);
  assert.deepEqual(presentingEvent.phaseTimingsMs, {
    DISCOVERING: 50,
    PRELOADING: 150,
    LAUNCHING_RUNTIME: 100,
  });
  assert.equal(presentingEvent.parallelPreloadSavedMs, 280);
  assert.equal(presentingEvent.totalBootMs, 300);
});

test('W225 parallelPreload=false runs preload tasks strictly sequentially with parallelPreloadSavedMs=0', async () => {
  let now = 2000;
  let inFlight = 0;
  let peakInFlight = 0;
  const order = [];

  function makeSequentialTask(name, durationMs, result) {
    return async () => {
      order.push(`start:${name}`);
      inFlight += 1;
      peakInFlight = Math.max(peakInFlight, inFlight);
      await Promise.resolve();
      now += durationMs;
      inFlight -= 1;
      order.push(`end:${name}`);
      return result;
    };
  }

  const preloader = {
    preloadPgim: makeSequentialTask('pgim', 150, { header: { magic: 'PGIM' }, bytesLoaded: 64 }),
    preloadShaderPack: makeSequentialTask('shaderPack', 100, { bytes: new Uint8Array([1]), bytesLoaded: 16 }),
    preloadReflectionManifest: makeSequentialTask('reflection', 40, { schema_version: 1, bytesLoaded: 12 }),
  };

  const fetchImpl = async () => {
    now += 50;
    return {
      ok: true,
      status: 200,
      async json() {
        return readyStatus();
      },
    };
  };

  const { orchestrator } = createOrchestrator({
    fetchImpl,
    preloader,
    nowFn: () => now,
    parallelPreload: false,
  });

  const result = await orchestrator.start();
  assert.equal(result.ok, true);
  assert.equal(peakInFlight, 1);
  assert.deepEqual(order, [
    'start:pgim',
    'end:pgim',
    'start:shaderPack',
    'end:shaderPack',
    'start:reflection',
    'end:reflection',
  ]);
  assert.deepEqual(result.phaseTimingsMs, { DISCOVERING: 50, PRELOADING: 290 });
  assert.equal(result.parallelPreloadSavedMs, 0);
  assert.equal(result.totalBootMs, 340);
  assert.equal(orchestrator.getTimingStats().preloadStats.maxConcurrency, 1);
  assert.equal(orchestrator.getTimingStats().parallelPreloadEnabled, false);
});

test('W225 overlaps preloadBootBundle with extra sidecars and custom preloadTasks, and aggregates preloadBootBundle timingMs', async () => {
  let now = 500;
  let active = 0;
  let peak = 0;
  let resolveBundle;
  let resolveIndex;
  let resolveCustom;

  const preloader = {
    preloadBootBundle: async () => {
      active += 1;
      peak = Math.max(peak, active);
      return new Promise((resolve) => {
        resolveBundle = () => {
          active -= 1;
          resolve({
            pgim: { header: { magic: 'PGIM' }, bytesLoaded: 100 },
            shaderPackBytes: new Uint8Array([1, 2]),
            reflectionManifest: { schema_version: 1 },
            bytesLoaded: 100,
            timingMs: { pgim: 120, shaderPack: 70, reflection: 30, total: 120 },
          });
        };
      });
    },
    preloadShaderPackIndex: async () => {
      active += 1;
      peak = Math.max(peak, active);
      return new Promise((resolve) => {
        resolveIndex = () => {
          active -= 1;
          resolve({ indexEntries: 8, bytesLoaded: 16 });
        };
      });
    },
  };

  const customTask = {
    name: 'shaderCacheWarmup',
    optional: true,
    run: async () => {
      active += 1;
      peak = Math.max(peak, active);
      return new Promise((resolve) => {
        resolveCustom = () => {
          active -= 1;
          resolve({ warmed: true, bytesLoaded: 8 });
        };
      });
    },
  };

  const fetchImpl = async () => {
    now += 20;
    return {
      ok: true,
      status: 200,
      async json() {
        return readyStatus({ shaderPackIndexUrl: '/data/shaders/boot.wgsl.idx' });
      },
    };
  };

  const { orchestrator, launches } = createOrchestrator({
    fetchImpl,
    preloader,
    nowFn: () => now,
    parallelPreload: true,
    preloadTasks: [customTask],
  });

  const pending = orchestrator.start();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(peak, 3);

  const flush = () => new Promise((r) => setTimeout(r, 0));
  now = 570; // 50ms into preload
  resolveCustom();
  await flush();
  now = 590; // 70ms into preload
  resolveIndex();
  await flush();
  now = 640; // 120ms into preload
  resolveBundle();

  const res = await pending;
  assert.equal(res.ok, true);
  assert.equal(launches[0].bundle.shaderPackIndex.indexEntries, 8);
  assert.equal(launches[0].bundle.sidecars.shaderCacheWarmup.warmed, true);
  assert.equal(launches[0].bundle.bytesLoaded, 124);

  const stats = orchestrator.getTimingStats();
  // Subtasks: pgim (120) + shaderPack (70) + reflection (30) + shaderPackIndex (70) + shaderCacheWarmup (50) = 340
  // Wall time: 120ms -> saved = 220ms
  assert.equal(stats.preloadStats.sequentialTimeMs, 340);
  assert.equal(stats.preloadStats.wallTimeMs, 120);
  assert.equal(stats.parallelPreloadSavedMs, 220);
  assert.deepEqual(stats.phaseTimingsMs, { DISCOVERING: 20, PRELOADING: 120 });
  assert.equal(stats.totalBootMs, 140);
});

test('W225 propagates AbortSignal cancellation across all concurrent preload tasks including non-cooperative tasks', async () => {
  let now = 100;
  const abortedTasks = [];
  const signals = [];

  function makeAbortableTask(name) {
    return (_url, { signal } = {}) => {
      signals.push({ name, signal });
      return new Promise((_resolve, reject) => {
        signal.addEventListener(
          'abort',
          () => {
            abortedTasks.push(name);
            const err = new Error(`aborted ${name}`);
            err.name = 'AbortError';
            reject(err);
          },
          { once: true }
        );
      });
    };
  }

  const preloader = {
    preloadPgim: makeAbortableTask('pgim'),
    preloadShaderPack: makeAbortableTask('shaderPack'),
    preloadReflectionManifest: makeAbortableTask('reflection'),
  };

  // Also include a custom task that ignores signal to verify raceAbort still unblocks cancellation.
  const hangingTask = {
    name: 'slowSidecar',
    optional: true,
    run: ({ signal }) => {
      signals.push({ name: 'slowSidecar', signal });
      return new Promise(() => {});
    },
  };

  const controller = new AbortController();
  const { orchestrator, launches, fallbacks, phases } = createOrchestrator({
    fetchImpl: async () => {
      now += 25;
      return {
        ok: true,
        status: 200,
        async json() {
          return readyStatus();
        },
      };
    },
    preloader,
    nowFn: () => now,
    parallelPreload: true,
    preloadTasks: [hangingTask],
  });

  const pending = orchestrator.start({ signal: controller.signal });
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(orchestrator.getState().phase, BOOT_PHASE.PRELOADING);
  assert.equal(signals.length, 4);

  now += 40;
  controller.abort();
  const result = await pending;

  assert.equal(result.ok, false);
  assert.equal(result.aborted, true);
  assert.equal(result.phase, BOOT_PHASE.CANCELLED);
  assert.equal(launches.length, 0);
  assert.equal(fallbacks.length, 0);
  assert.deepEqual(phases, [BOOT_PHASE.DISCOVERING, BOOT_PHASE.PRELOADING, BOOT_PHASE.CANCELLED]);
  assert.equal(signals.every((entry) => entry.signal.aborted === true), true);
  assert.deepEqual(abortedTasks.sort(), ['pgim', 'reflection', 'shaderPack']);
  assert.equal(result.parallelPreloadSavedMs, 0);
  assert.deepEqual(result.phaseTimingsMs, { DISCOVERING: 25, PRELOADING: 40 });
  assert.equal(result.totalBootMs, 65);
  assert.equal(orchestrator.getTimingStats().preloadStats.abortedCount, 4);
});

test('W225 required preload failure aborts in-flight sibling tasks and surfaces the root fatal error', async () => {
  let now = 300;
  const siblingAborts = [];
  let failPgim;

  const preloader = {
    preloadPgim: () =>
      new Promise((_resolve, reject) => {
        failPgim = () => reject(new Error('PGIM HTTP 500 server failure'));
      }),
    preloadShaderPack: (_url, { signal }) =>
      new Promise((_resolve, reject) => {
        signal.addEventListener(
          'abort',
          () => {
            siblingAborts.push('shaderPack');
            const err = new Error('aborted');
            err.name = 'AbortError';
            reject(err);
          },
          { once: true }
        );
      }),
    preloadReflectionManifest: (_url, { signal }) =>
      new Promise((_resolve, reject) => {
        signal.addEventListener(
          'abort',
          () => {
            siblingAborts.push('reflection');
            const err = new Error('aborted');
            err.name = 'AbortError';
            reject(err);
          },
          { once: true }
        );
      }),
  };

  const { orchestrator, launches } = createOrchestrator({
    fetchImpl: createMockFetch(readyStatus()),
    preloader,
    nowFn: () => now,
    parallelPreload: true,
  });

  const pending = orchestrator.start();
  await new Promise((r) => setTimeout(r, 0));

  now += 35;
  failPgim();
  const result = await pending;

  assert.equal(result.ok, false);
  assert.equal(result.phase, BOOT_PHASE.ERROR);
  assert.match(result.error.message, /PGIM HTTP 500/);
  assert.equal(launches.length, 0);
  assert.deepEqual(siblingAborts.sort(), ['reflection', 'shaderPack']);
  assert.equal(orchestrator.getTimingStats().preloadStats.failedCount, 1);
  assert.equal(orchestrator.getTimingStats().preloadStats.abortedCount, 2);
  assert.equal(orchestrator.getTimingStats().parallelPreloadSavedMs, 0);
});

test('W225 falls back cleanly when optional reflection/index sidecars are missing while rejecting corrupt sidecars', async () => {
  let now = 1000;
  const progressEvents = [];

  const notFoundErr = new Error('Reflection sidecar not found at C:\\Users\\bob\\secret\\sidecar.json?token=xyz (HTTP 404)');
  notFoundErr.status = 404;
  notFoundErr.code = 'NOT_FOUND';

  const indexMissingErr = new Error('Shader pack index missing');
  indexMissingErr.code = 'ENOENT';

  const preloader = {
    preloadPgim: async () => {
      await Promise.resolve();
      now += 60;
      return { header: { magic: 'PGIM' }, bytesLoaded: 128 };
    },
    preloadShaderPack: async () => {
      await Promise.resolve();
      return { bytes: new Uint8Array([9, 8, 7]), bytesLoaded: 3 };
    },
    preloadShaderPackHeader: async () => {
      await Promise.resolve();
      throw indexMissingErr;
    },
    preloadReflectionManifest: async () => {
      await Promise.resolve();
      throw notFoundErr;
    },
  };

  const bootTracer = new BootTracer();
  bootTracer.recordProgress = (ev) => progressEvents.push(ev);

  const { orchestrator, launches } = createOrchestrator({
    fetchImpl: createMockFetch(
      readyStatus({
        shaderPackIndexUrl: '/data/shaders/boot.wgsl.idx',
      })
    ),
    preloader,
    bootTracer,
    nowFn: () => now,
    parallelPreload: true,
    preloadTasks: [
      {
        name: 'optionalAudioSidecar',
        optional: true,
        run: async () => {
          throw new Error('optional sidecar unavailable');
        },
      },
    ],
  });

  const result = await orchestrator.start();
  assert.equal(result.ok, true);
  assert.equal(result.phase, BOOT_PHASE.LAUNCHING_RUNTIME);
  assert.equal(launches.length, 1);
  assert.equal(launches[0].bundle.pgim.header.magic, 'PGIM');
  assert.equal(launches[0].bundle.shaderPackBytes.bytesLoaded, 3);
  assert.equal(launches[0].bundle.shaderPackIndex, null);
  assert.equal(launches[0].bundle.reflectionManifest, null);
  assert.equal(launches[0].bundle.sidecars.optionalAudioSidecar, null);

  assert.ok(progressEvents.some((ev) => ev.phase === 'optional-missing' && ev.asset === 'reflection'));
  assert.ok(progressEvents.some((ev) => ev.phase === 'optional-missing' && ev.asset === 'shaderPackIndex'));
  assert.ok(progressEvents.some((ev) => ev.phase === 'optional-missing' && ev.asset === 'optionalAudioSidecar'));

  const snapshot = orchestrator.getDiagnosticSnapshot();
  assert.equal(snapshot.preloadStats.optionalMissingCount, 3);
  assert.equal(snapshot.preloadStats.tasks.reflection.status, 'missing');
  assert.equal(snapshot.preloadStats.tasks.shaderPackIndex.status, 'missing');
  assert.equal(snapshot.preloadStats.tasks.optionalAudioSidecar.status, 'fallback');
  // Verify sensitive path and query token in optional sidecar error are redacted in diagnostics.
  assert.equal(snapshot.preloadStats.tasks.reflection.error.includes('bob'), false);
  assert.equal(snapshot.preloadStats.tasks.reflection.error.includes('token=xyz'), false);
  assert.match(snapshot.preloadStats.tasks.reflection.error, /\[REDACTED_PATH\]/);
  assert.equal('bundle' in snapshot, false);
  assert.equal('status' in snapshot, false);
  assert.doesNotThrow(() => JSON.parse(JSON.stringify(snapshot)));

  // Corrupt sidecar (BAD_* validation error) MUST fail boot rather than silently falling back.
  const corruptErr = new Error('corrupt reflection JSON');
  corruptErr.code = 'BAD_REFLECTION_MANIFEST';
  const corruptOrch = createOrchestrator({
    fetchImpl: createMockFetch(readyStatus()),
    preloader: {
      preloadPgim: async () => ({ header: { magic: 'PGIM' }, bytesLoaded: 64 }),
      preloadReflectionManifest: async () => {
        throw corruptErr;
      },
    },
    parallelPreload: true,
  });
  const corruptRes = await corruptOrch.orchestrator.start();
  assert.equal(corruptRes.ok, false);
  assert.equal(corruptRes.phase, BOOT_PHASE.ERROR);
  assert.match(corruptRes.error.message, /corrupt reflection JSON/);
});

test('W225 overlaps non-PGIM shaderPack + reflection preload and records positive real-clock parallelPreloadSavedMs', async () => {
  let inFlight = 0;
  let peakInFlight = 0;

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const preloader = {
    preloadShaderPack: async () => {
      inFlight += 1;
      peakInFlight = Math.max(peakInFlight, inFlight);
      await sleep(35);
      inFlight -= 1;
      return { bytes: new Uint8Array([1, 2, 3]), bytesLoaded: 3 };
    },
    preloadReflectionManifest: async () => {
      inFlight += 1;
      peakInFlight = Math.max(peakInFlight, inFlight);
      await sleep(35);
      inFlight -= 1;
      return { schema_version: 1, bytesLoaded: 10 };
    },
  };

  const { orchestrator, fallbacks } = createOrchestrator({
    fetchImpl: createMockFetch(
      readyStatus({
        pgimUrl: null,
        runtimeWasm: null,
        hasWasmBinary: false,
      })
    ),
    preloader,
    parallelPreload: true,
  });

  const result = await orchestrator.start();
  assert.equal(result.ok, true);
  assert.equal(result.fallback, true);
  assert.equal(result.phase, BOOT_PHASE.FALLBACK_SMOKE_DEMO);
  assert.equal(fallbacks.length, 1);
  assert.equal(peakInFlight, 2);
  assert.equal(result.parallelPreloadSavedMs > 0, true);
  assert.equal(result.totalBootMs >= result.phaseTimingsMs.PRELOADING, true);
  assert.equal(orchestrator.getState().parallelPreloadSavedMs > 0, true);

  // reset() clears all W225 preload and timing stats cleanly.
  orchestrator.reset();
  const resetStats = orchestrator.getTimingStats();
  assert.equal(resetStats.parallelPreloadSavedMs, 0);
  assert.equal(resetStats.totalBootMs, 0);
  assert.deepEqual(resetStats.phaseTimingsMs, {});
  assert.deepEqual(resetStats.preloadTaskTimingsMs, {});
  assert.equal(resetStats.preloadStats.taskCount, 0);
});

// ---------------------------------------------------------------------------
// W258: Release large preload ArrayBuffer / Uint8Array references after handoff
// ---------------------------------------------------------------------------

test('W258 releaseBootBundleBytes nulls binary payloads without double-counting, preserves metadata, and is defensive/idempotent', () => {
  assert.equal(typeof releaseBootBundleBytes, 'function');
  assert.equal(DEFAULT_RELEASE_PRELOAD_BUFFERS, true);

  // Defensive on null, undefined, primitives, and frozen objects.
  for (const input of [null, undefined, 0, 42, 'bundle', true, Symbol('x')]) {
    assert.deepEqual(releaseBootBundleBytes(input), { releasedFields: [], releasedBytes: 0 });
  }
  const frozenBundle = Object.freeze({
    shaderPackBytes: new Uint8Array(64),
    pgim: Object.freeze({ bytes: new Uint8Array(32), bytesLoaded: 32 }),
    bytesLoaded: 96,
  });
  assert.deepEqual(releaseBootBundleBytes(frozenBundle), { releasedFields: [], releasedBytes: 0 });
  assert.equal(frozenBundle.shaderPackBytes.byteLength, 64);

  // Full boot bundle with aliased pgim backing buffer + shaderPackBytes + reflectionManifest + reflectionSidecar + sidecars.
  const pgimBuf = new ArrayBuffer(128);
  const pgimBytes = new Uint8Array(pgimBuf);
  const shaderPackBytes = new Uint8Array(256);
  const reflBytes = new Uint8Array(48);
  const sidecarBuf = new ArrayBuffer(64);
  const sidecarBytes = new Uint8Array(sidecarBuf);
  const customBuf = new ArrayBuffer(32);
  const customBytes = new Uint8Array(customBuf);
  const customDirect = new Uint8Array(16);

  const bundle = {
    buffer: pgimBuf,
    pgimBytes,
    pgim: {
      bytes: pgimBytes,
      buffer: pgimBuf,
      header: { magic: 'PGIM', entryPointRva: 0x1000 },
      bytesLoaded: 128,
      totalBytes: 128,
      transport: 'range',
      timingMs: 12,
      durationMs: 12,
    },
    shaderPackBytes,
    shaderPackIndex: {
      entryCount: 42,
      bytes: new Uint8Array(24),
      bytesLoaded: 24,
    },
    reflectionManifest: {
      manifest: { schema_version: 1, shaders: { a: {} } },
      bytes: reflBytes,
      bytesLoaded: 48,
      totalBytes: 48,
      durationMs: 4,
    },
    reflectionSidecar: {
      manifest: { stages: 4 },
      bytes: sidecarBytes,
      buffer: sidecarBuf,
      bytesLoaded: 64,
    },
    sidecars: {
      extraObj: { bytes: customBytes, buffer: customBuf, bytesLoaded: 32, tag: 'ok' },
      extraRaw: customDirect,
    },
    bytesLoaded: 568,
    totalBytes: 568,
    timingMs: { pgim: 12, shaderPack: 20, reflection: 4, total: 20 },
    durationMs: 20,
  };

  const expectedUniqueBytes = 128 + 256 + 24 + 48 + 64 + 32 + 16;
  const first = releaseBootBundleBytes(bundle);
  assert.equal(first.releasedBytes, expectedUniqueBytes);
  assert.ok(first.releasedFields.includes('buffer'));
  assert.ok(first.releasedFields.includes('pgimBytes'));
  assert.ok(first.releasedFields.includes('shaderPackBytes'));
  assert.ok(first.releasedFields.includes('pgim.bytes'));
  assert.ok(first.releasedFields.includes('pgim.buffer'));
  assert.ok(first.releasedFields.includes('shaderPackIndex.bytes'));
  assert.ok(first.releasedFields.includes('reflectionManifest.bytes'));
  assert.ok(first.releasedFields.includes('reflectionSidecar.bytes'));
  assert.ok(first.releasedFields.includes('reflectionSidecar.buffer'));
  assert.ok(first.releasedFields.includes('sidecars.extraObj.bytes'));
  assert.ok(first.releasedFields.includes('sidecars.extraObj.buffer'));
  assert.ok(first.releasedFields.includes('sidecars.extraRaw'));

  // All binary payloads must be nulled.
  assert.equal(bundle.buffer, null);
  assert.equal(bundle.pgimBytes, null);
  assert.equal(bundle.shaderPackBytes, null);
  assert.equal(bundle.pgim.bytes, null);
  assert.equal(bundle.pgim.buffer, null);
  assert.equal(bundle.shaderPackIndex.bytes, null);
  assert.equal(bundle.reflectionManifest.bytes, null);
  assert.equal(bundle.reflectionSidecar.bytes, null);
  assert.equal(bundle.reflectionSidecar.buffer, null);
  assert.equal(bundle.sidecars.extraObj.bytes, null);
  assert.equal(bundle.sidecars.extraObj.buffer, null);
  assert.equal(bundle.sidecars.extraRaw, null);

  // Lightweight metadata must remain intact.
  assert.equal(bundle.bytesLoaded, 568);
  assert.equal(bundle.totalBytes, 568);
  assert.deepEqual(bundle.timingMs, { pgim: 12, shaderPack: 20, reflection: 4, total: 20 });
  assert.equal(bundle.durationMs, 20);
  assert.equal(bundle.pgim.header.magic, 'PGIM');
  assert.equal(bundle.pgim.bytesLoaded, 128);
  assert.equal(bundle.pgim.transport, 'range');
  assert.equal(bundle.shaderPackIndex.entryCount, 42);
  assert.deepEqual(bundle.reflectionManifest.manifest, { schema_version: 1, shaders: { a: {} } });
  assert.equal(bundle.reflectionManifest.bytesLoaded, 48);
  assert.deepEqual(bundle.reflectionSidecar.manifest, { stages: 4 });
  assert.equal(bundle.sidecars.extraObj.bytesLoaded, 32);
  assert.equal(bundle.sidecars.extraObj.tag, 'ok');

  // Calling twice is idempotent and reports 0 released bytes.
  const second = releaseBootBundleBytes(bundle);
  assert.deepEqual(second, { releasedFields: [], releasedBytes: 0 });
});

test('W258 BootOrchestrator hands intact buffers to onLaunchRuntime and releases them after handoff by default', async () => {
  assert.throws(() => new BootOrchestrator({ releasePreloadBuffers: 'yes' }), TypeError);
  assert.throws(() => new BootOrchestrator({ releasePreloadBuffers: 1 }), TypeError);

  const pgimBuf = new ArrayBuffer(256);
  const shaderBuf = new ArrayBuffer(512);
  const reflBytes = new Uint8Array(64);
  const sidecarBytes = new Uint8Array(96);

  const granularPreloader = {
    preloadPgim: async () => ({
      header: { magic: 'PGIM' },
      bytes: new Uint8Array(pgimBuf),
      buffer: pgimBuf,
      bytesLoaded: 256,
      totalBytes: 256,
      transport: 'range',
    }),
    preloadShaderPack: async () => ({
      bytes: new Uint8Array(shaderBuf),
      buffer: shaderBuf,
      bytesLoaded: 512,
      totalBytes: 512,
      transport: 'range',
    }),
    preloadReflectionManifest: async () => ({
      manifest: { schema_version: 1 },
      bytes: reflBytes,
      bytesLoaded: 64,
    }),
    preloadReflectionSidecar: async () => ({
      manifest: { stageCount: 2 },
      bytes: sidecarBytes,
      bytesLoaded: 96,
    }),
  };

  let observedDuringLaunch = null;
  const { orchestrator, launches } = createOrchestrator({
    fetchImpl: createMockFetch(
      readyStatus({
        reflectionSidecarUrl: '/data/shaders/stage-reflection.json',
      })
    ),
    preloader: granularPreloader,
    onLaunchRuntime: async (payload) => {
      launches.push(payload);
      observedDuringLaunch = {
        pgimLen: payload.bundle.pgim.bytes.byteLength,
        pgimBufLen: payload.bundle.pgim.buffer.byteLength,
        shaderLen: payload.bundle.shaderPackBytes.bytes.byteLength,
        shaderBufLen: payload.bundle.shaderPackBytes.buffer.byteLength,
        reflLen: payload.bundle.reflectionManifest.bytes.byteLength,
        sidecarLen: payload.bundle.reflectionSidecar.bytes.byteLength,
      };
    },
  });

  const result = await orchestrator.start();
  assert.equal(result.ok, true);

  // Inside onLaunchRuntime, all binary payloads were intact for worker handoff.
  assert.deepEqual(observedDuringLaunch, {
    pgimLen: 256,
    pgimBufLen: 256,
    shaderLen: 512,
    shaderBufLen: 512,
    reflLen: 64,
    sidecarLen: 96,
  });

  // After onLaunchRuntime resolves, internal bundle reference is cleared and binary fields are nulled.
  assert.equal(orchestrator.getState().bundle, null);
  assert.equal(launches[0].bundle.pgim.bytes, null);
  assert.equal(launches[0].bundle.pgim.buffer, null);
  assert.equal(launches[0].bundle.shaderPackBytes.bytes, null);
  assert.equal(launches[0].bundle.shaderPackBytes.buffer, null);
  assert.equal(launches[0].bundle.reflectionManifest.bytes, null);
  assert.equal(launches[0].bundle.reflectionSidecar.bytes, null);

  // Metadata, getPreloadStats(), getState(), and assets remain intact.
  assert.equal(launches[0].bundle.pgim.header.magic, 'PGIM');
  assert.equal(launches[0].bundle.pgim.bytesLoaded, 256);
  assert.equal(launches[0].bundle.shaderPackBytes.bytesLoaded, 512);
  assert.deepEqual(launches[0].bundle.reflectionManifest.manifest, { schema_version: 1 });
  assert.deepEqual(launches[0].bundle.reflectionSidecar.manifest, { stageCount: 2 });
  assert.equal(orchestrator.getPreloadStats().completedCount, 4);
  assert.equal(orchestrator.getState().assets.pgimUrl, '/data/boot/GTA6.pgim');

  // Calling releasePreloadBuffers() again is a safe idempotent no-op.
  assert.deepEqual(orchestrator.releasePreloadBuffers(), { releasedFields: [], releasedBytes: 0 });
});

test('W258 BootOrchestrator.releasePreloadBuffers() and dispose() release retained references while preserving getPreloadStats()', async () => {
  // 1. Explicit releasePreloadBuffers() when auto-release is opted out (`releasePreloadBuffers: false`).
  {
    const retainedBundle = {
      pgim: {
        header: { magic: 'PGIM' },
        bytes: new Uint8Array(128),
        buffer: null,
        bytesLoaded: 128,
      },
      shaderPackBytes: new Uint8Array(1024),
      reflectionManifest: {
        manifest: { schema_version: 1 },
        bytes: new Uint8Array(40),
        bytesLoaded: 40,
      },
      reflectionSidecar: {
        manifest: { ok: true },
        bytes: new Uint8Array(32),
        bytesLoaded: 32,
      },
      bytesLoaded: 1224,
    };
    retainedBundle.pgim.buffer = retainedBundle.pgim.bytes.buffer;

    const { orchestrator } = createOrchestrator({
      preloader: createPreloader({ bundle: retainedBundle }),
      releasePreloadBuffers: false,
    });

    await orchestrator.start();
    assert.equal(orchestrator.getState().bundle, retainedBundle);
    assert.equal(retainedBundle.shaderPackBytes.byteLength, 1024);

    const released = orchestrator.releasePreloadBuffers();
    assert.equal(released.releasedBytes, 128 + 1024 + 40 + 32);
    assert.equal(orchestrator.getState().bundle, null);
    assert.equal(retainedBundle.shaderPackBytes, null);
    assert.equal(retainedBundle.pgim.bytes, null);
    assert.equal(retainedBundle.pgim.buffer, null);
    assert.equal(retainedBundle.reflectionManifest.bytes, null);
    assert.equal(retainedBundle.reflectionSidecar.bytes, null);

    // Lightweight metadata, getPreloadStats(), getState(), and assets remain intact.
    assert.equal(retainedBundle.bytesLoaded, 1224);
    assert.deepEqual(retainedBundle.reflectionManifest.manifest, { schema_version: 1 });
    assert.equal(orchestrator.getPreloadStats().completedCount, 1);
    assert.equal(orchestrator.getState().phase, BOOT_PHASE.LAUNCHING_RUNTIME);
    assert.equal(orchestrator.getState().assets.pgimUrl, '/data/boot/GTA6.pgim');

    // Second release is idempotent.
    assert.deepEqual(orchestrator.releasePreloadBuffers(), { releasedFields: [], releasedBytes: 0 });
  }

  // 2. dispose() releases retained preload buffers while preserving getPreloadStats() and assets.
  {
    const retainedBundle = {
      pgim: {
        header: { magic: 'PGIM' },
        bytes: new Uint8Array(200),
        bytesLoaded: 200,
      },
      shaderPackBytes: new Uint8Array(800),
      reflectionManifest: {
        manifest: { schema_version: 1 },
        bytes: new Uint8Array(50),
        bytesLoaded: 50,
      },
      bytesLoaded: 1050,
    };

    const { orchestrator, overlay } = createOrchestrator({
      preloader: createPreloader({ bundle: retainedBundle }),
      releasePreloadBuffers: false,
    });

    await orchestrator.start();
    assert.equal(orchestrator.getState().bundle, retainedBundle);

    const disposeRes = orchestrator.dispose();
    assert.equal(disposeRes.ok, true);
    assert.equal(disposeRes.alreadyDestroyed, false);
    assert.equal(disposeRes.phase, BOOT_PHASE.DESTROYED);
    assert.equal(overlay.destroys, 1);

    assert.equal(orchestrator.getState().bundle, null);
    assert.equal(retainedBundle.shaderPackBytes, null);
    assert.equal(retainedBundle.pgim.bytes, null);
    assert.equal(retainedBundle.reflectionManifest.bytes, null);

    assert.equal(orchestrator.getPreloadStats().completedCount, 1);
    assert.equal(orchestrator.getState().assets.pgimUrl, '/data/boot/GTA6.pgim');
    assert.equal(orchestrator.getState().destroyed, true);
    assert.equal(orchestrator.dispose().alreadyDestroyed, true);
  }

  // 3. Without onLaunchRuntime, buffers stay available on start() return until notifyFirstPresent() or releasePreloadBuffers().
  {
    const retainedBundle = {
      pgim: { header: { magic: 'PGIM' }, bytes: new Uint8Array(64), bytesLoaded: 64 },
      shaderPackBytes: new Uint8Array(128),
      bytesLoaded: 192,
    };
    const { orchestrator } = createOrchestrator({
      preloader: createPreloader({ bundle: retainedBundle }),
      onLaunchRuntime: null,
    });

    const result = await orchestrator.start();
    assert.equal(result.ok, true);
    assert.equal(result.bundle.shaderPackBytes.byteLength, 128);
    assert.equal(orchestrator.getState().bundle, retainedBundle);

    orchestrator.notifyFirstPresent({ frameIndex: 1 });
    assert.equal(orchestrator.getState().bundle, null);
    assert.equal(retainedBundle.shaderPackBytes, null);
    assert.equal(retainedBundle.pgim.bytes, null);
    assert.equal(orchestrator.getPreloadStats().completedCount, 1);
  }
});

// ---------------------------------------------------------------------------
// W309: Concurrent preload overlap scheduling, bounded concurrency & telemetry
// ---------------------------------------------------------------------------

test('W309 exports runConcurrentPreloads and DEFAULT_PRELOAD_CONCURRENCY and validates new options', () => {
  assert.equal(typeof runConcurrentPreloads, 'function');
  assert.equal(Number.isInteger(DEFAULT_PRELOAD_CONCURRENCY) && DEFAULT_PRELOAD_CONCURRENCY >= 4, true);

  assert.throws(() => new BootOrchestrator({ preloadConcurrency: '4' }), TypeError);
  assert.throws(() => new BootOrchestrator({ preloadConcurrency: 0 }), RangeError);
  assert.throws(() => new BootOrchestrator({ preloadConcurrency: -2 }), RangeError);
  assert.throws(() => new BootOrchestrator({ preloadConcurrency: 2.5 }), RangeError);
  assert.throws(() => new BootOrchestrator({ preloadConcurrency: Number.NaN }), RangeError);
  assert.throws(() => new BootOrchestrator({ preloadRuntimeWasm: 'nope' }), TypeError);
  assert.throws(() => new BootOrchestrator({ opfsProbe: 42 }), TypeError);
  assert.throws(() => new BootOrchestrator({ preloadShaderPackHeader: {} }), TypeError);
  assert.throws(() => new BootOrchestrator({ preloadReflectionSidecar: true }), TypeError);

  const aliased = discoverBootAssets({
    ok: true,
    hasWasmBinary: true,
    pgimUrl: '/data/boot/GTA6.pgim',
    splashShaderPackUrl: '/data/shaders/splash.wgsl.pack',
    wgpkHeaderUrl: '/data/shaders/splash.wgsl.pack.hdr',
    reflectJsonUrl: '/data/shaders/splash.wgsl.pack.reflect.json',
  });
  assert.equal(aliased.shaderPackUrl, '/data/shaders/splash.wgsl.pack');
  assert.equal(aliased.shaderPackIndexUrl, '/data/shaders/splash.wgsl.pack.hdr');
  assert.equal(aliased.reflectionSidecarUrl, '/data/shaders/splash.wgsl.pack.reflect.json');

  const orch = new BootOrchestrator({ fetchImpl: async () => ({}) });
  const stats = orch.getStats();
  assert.equal(stats.serialEstimateMs, 0);
  assert.equal(stats.parallelWallMs, 0);
  assert.equal(stats.overlapSavedMs, 0);
  assert.equal(stats.parallelStagesCount, 0);
  assert.equal(stats.preloadConcurrency, DEFAULT_PRELOAD_CONCURRENCY);
  assert.equal(stats.parallelPreload, true);
  assert.doesNotThrow(() => JSON.parse(JSON.stringify(stats)));
});

test('W309 runConcurrentPreloads enforces bounded concurrency, computes overlap savings, and propagates deterministic fatal errors', async () => {
  await assert.rejects(() => runConcurrentPreloads('not-array'), TypeError);
  await assert.rejects(() => runConcurrentPreloads([], null), TypeError);
  await assert.rejects(() => runConcurrentPreloads([{ name: 'a', run: () => {} }], { maxConcurrency: 0 }), RangeError);

  const emptyRes = await runConcurrentPreloads([]);
  assert.equal(emptyRes.ok, true);
  assert.equal(emptyRes.stats.parallelStagesCount, 0);
  assert.equal(emptyRes.stats.overlapSavedMs, 0);

  // Bounded concurrency: maxConcurrency = 2 across 4 overlapping async tasks.
  let active = 0;
  let peak = 0;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const boundedTasks = [
    {
      name: 'pgim',
      optional: false,
      run: async () => {
        active += 1;
        peak = Math.max(peak, active);
        await sleep(30);
        active -= 1;
        return { tag: 'pgim' };
      },
    },
    {
      name: 'wgpkHeader',
      optional: true,
      run: async () => {
        active += 1;
        peak = Math.max(peak, active);
        await sleep(30);
        active -= 1;
        return { tag: 'hdr' };
      },
    },
    {
      name: 'reflectSidecar',
      optional: true,
      run: async () => {
        active += 1;
        peak = Math.max(peak, active);
        await sleep(30);
        active -= 1;
        return { tag: 'reflect' };
      },
    },
    {
      name: 'runtimeWasm',
      optional: true,
      run: async () => {
        active += 1;
        peak = Math.max(peak, active);
        await sleep(30);
        active -= 1;
        return { tag: 'wasm' };
      },
    },
  ];

  const boundedOut = await runConcurrentPreloads(boundedTasks, { parallel: true, maxConcurrency: 2 });
  assert.equal(boundedOut.ok, true);
  assert.equal(peak, 2);
  assert.equal(boundedOut.stats.maxConcurrency, 2);
  assert.equal(boundedOut.stats.concurrencyLimit, 2);
  assert.equal(boundedOut.stats.parallelStagesCount, 4);
  assert.equal(boundedOut.stats.completedCount, 4);
  assert.equal(boundedOut.stats.serialEstimateMs > boundedOut.stats.parallelWallMs, true);
  assert.equal(boundedOut.stats.overlapSavedMs > 0, true);
  assert.equal(boundedOut.results.pgim.tag, 'pgim');
  assert.equal(boundedOut.results.runtimeWasm.tag, 'wasm');

  // Serial mode (parallel: false) reports parallelStagesCount = 0 and overlapSavedMs = 0.
  let serialActive = 0;
  let serialPeak = 0;
  const serialOut = await runConcurrentPreloads(
    [
      {
        name: 's1',
        run: async () => {
          serialActive += 1;
          serialPeak = Math.max(serialPeak, serialActive);
          await sleep(10);
          serialActive -= 1;
          return 1;
        },
      },
      {
        name: 's2',
        run: async () => {
          serialActive += 1;
          serialPeak = Math.max(serialPeak, serialActive);
          await sleep(10);
          serialActive -= 1;
          return 2;
        },
      },
    ],
    { parallel: false }
  );
  assert.equal(serialPeak, 1);
  assert.equal(serialOut.stats.parallelStagesCount, 0);
  assert.equal(serialOut.stats.overlapSavedMs, 0);

  // Deterministic error propagation:
  // (a) When two required tasks fail concurrently with distinct non-abort errors, lowest task index wins.
  let queuedStarted = false;
  const concurrentErr = await runConcurrentPreloads(
    [
      {
        name: 'firstRequired',
        optional: false,
        run: () => Promise.reject(new Error('deterministic root error #0')),
      },
      {
        name: 'secondRequired',
        optional: false,
        run: () => Promise.reject(new Error('later-indexed error #1')),
      },
      {
        name: 'unstartedQueued',
        optional: true,
        run: async () => {
          queuedStarted = true;
          return { ok: true };
        },
      },
    ],
    { parallel: true, maxConcurrency: 2, throwOnError: false }
  );
  assert.equal(concurrentErr.ok, false);
  assert.match(concurrentErr.error.message, /deterministic root error #0/);
  assert.equal(concurrentErr.stats.overlapSavedMs, 0);
  assert.equal(queuedStarted, false);

  // (b) When a later-indexed required task fails while an earlier task is still pending,
  //     fail-fast abort cancels the pending sibling and propagates the fatal error.
  let siblingAborted = false;
  await assert.rejects(
    () =>
      runConcurrentPreloads(
        [
          {
            name: 'pendingSibling',
            optional: false,
            run: (sig) =>
              new Promise((_resolve, reject) => {
                sig.addEventListener(
                  'abort',
                  () => {
                    siblingAborted = true;
                    const err = new Error('aborted');
                    err.name = 'AbortError';
                    reject(err);
                  },
                  { once: true }
                );
              }),
          },
          {
            name: 'failingRequired',
            optional: false,
            run: async () => {
              throw new Error('fatal sibling boom');
            },
          },
        ],
        { parallel: true, maxConcurrency: 2 }
      ),
    /fatal sibling boom/
  );
  assert.equal(siblingAborted, true);
});

test('W309 overlaps .pgim, splash.wgsl.pack header, auto-derived .reflect.json sidecar, runtime.wasm, and OPFS probe with overlapSavedMs > 0', async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  let inFlight = 0;
  let peakInFlight = 0;
  const observedUrls = {};

  function timedStage(key, delayMs, makeValue) {
    return async (arg1) => {
      inFlight += 1;
      peakInFlight = Math.max(peakInFlight, inFlight);
      if (typeof arg1 === 'string') {
        observedUrls[key] = arg1;
      }
      await sleep(delayMs);
      inFlight -= 1;
      return makeValue(arg1);
    };
  }

  const preloader = {
    preloadPgim: timedStage('pgim', 45, (url) => ({
      header: { magic: 'PGIM' },
      url,
      bytesLoaded: 1024,
    })),
    preloadShaderPackHeader: timedStage('shaderPackIndex', 35, (url) => ({
      magic: 'WGPK',
      entryCount: 32,
      url,
      bytesLoaded: 128,
    })),
    preloadReflectionSidecar: timedStage('reflectionSidecar', 30, (url) => ({
      schema: 'playgta6-stage-reflection',
      stageCount: 6,
      url,
      bytesLoaded: 96,
    })),
  };

  const preloadRuntimeWasm = timedStage('runtimeWasm', 40, (url) => ({
    module: { compiled: true, kind: 'WebAssembly.Module' },
    url,
    bytesLoaded: 2048,
  }));

  const opfsProbe = async () => {
    inFlight += 1;
    peakInFlight = Math.max(peakInFlight, inFlight);
    await sleep(25);
    inFlight -= 1;
    return { warm: true, cacheHit: true, entryCount: 14 };
  };

  const fetchImpl = createMockFetch(
    readyStatus({
      shaderPackUrl: '/data/shaders/splash.wgsl.pack',
      reflectionManifestUrl: null,
      runtimeWasm: '/dist/runtime.wasm',
    })
  );

  const phaseEvents = [];
  const { orchestrator, launches } = createOrchestrator({
    fetchImpl,
    preloader,
    preloadRuntimeWasm,
    opfsProbe,
    parallelPreload: true,
    onPhaseChange: (ev) => phaseEvents.push(ev),
  });

  const result = await orchestrator.start();
  assert.equal(result.ok, true);
  assert.equal(result.phase, BOOT_PHASE.LAUNCHING_RUNTIME);
  assert.equal(launches.length, 1);

  // All 5 stages overlapped concurrently in real wall time!
  assert.equal(peakInFlight, 5);
  assert.equal(observedUrls.pgim, '/data/boot/GTA6.pgim');
  assert.equal(observedUrls.shaderPackIndex, '/data/shaders/splash.wgsl.pack');
  assert.equal(observedUrls.reflectionSidecar, '/data/shaders/splash.wgsl.pack.reflect.json');
  assert.equal(observedUrls.runtimeWasm, '/dist/runtime.wasm');

  assert.equal(launches[0].bundle.pgim.header.magic, 'PGIM');
  assert.equal(launches[0].bundle.shaderPackIndex.magic, 'WGPK');
  assert.equal(launches[0].bundle.reflectionSidecar.stageCount, 6);
  assert.equal(launches[0].bundle.runtimeWasm.module.compiled, true);
  assert.equal(launches[0].bundle.opfsWarmBoot.warm, true);

  // Verify timing savings telemetry on start() result, getStats(), getTimingStats(), getState(), and getDiagnosticSnapshot().
  assert.equal(result.parallelStagesCount, 5);
  assert.equal(result.serialEstimateMs > result.parallelWallMs, true);
  assert.equal(result.overlapSavedMs > 0, true);
  assert.equal(result.overlapSavedMs, result.parallelPreloadSavedMs);

  const stats = orchestrator.getStats();
  assert.equal(stats.parallelStagesCount, 5);
  assert.equal(stats.serialEstimateMs, result.serialEstimateMs);
  assert.equal(stats.parallelWallMs, result.parallelWallMs);
  assert.equal(stats.overlapSavedMs, result.overlapSavedMs);
  assert.equal(stats.preloadStats.maxConcurrency, 5);
  assert.equal(stats.preloadStats.completedCount, 5);
  assert.equal('bundle' in stats, false);

  const timingStats = orchestrator.getTimingStats();
  assert.equal(timingStats.serialEstimateMs, stats.serialEstimateMs);
  assert.equal(timingStats.parallelWallMs, stats.parallelWallMs);
  assert.equal(timingStats.overlapSavedMs, stats.overlapSavedMs);
  assert.equal(timingStats.parallelStagesCount, 5);

  const state = orchestrator.getState();
  assert.equal(state.serialEstimateMs, stats.serialEstimateMs);
  assert.equal(state.parallelWallMs, stats.parallelWallMs);
  assert.equal(state.overlapSavedMs, stats.overlapSavedMs);
  assert.equal(state.parallelStagesCount, 5);

  const diag = orchestrator.getDiagnosticSnapshot();
  assert.equal(diag.overlapSavedMs, stats.overlapSavedMs);
  assert.equal(diag.parallelStagesCount, 5);
  assert.doesNotThrow(() => JSON.parse(JSON.stringify(stats)));
  assert.doesNotThrow(() => JSON.parse(JSON.stringify(diag)));

  const launchEvent = phaseEvents.find((ev) => ev.phase === BOOT_PHASE.LAUNCHING_RUNTIME);
  assert.ok(launchEvent);
  assert.equal(launchEvent.overlapSavedMs, stats.overlapSavedMs);
  assert.equal(launchEvent.parallelStagesCount, 5);
});

test('W309 tolerates optional sidecar, OPFS probe, and runtimeWasm prefetch failures with sanitized diagnostics while failing on BAD_* validation errors', async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const progressEvents = [];
  const bootTracer = new BootTracer();
  bootTracer.recordProgress = (ev) => progressEvents.push(ev);

  const preloader = {
    preloadPgim: async () => {
      await sleep(30);
      return { header: { magic: 'PGIM' }, bytesLoaded: 256 };
    },
    preloadShaderPackHeader: async () => {
      await sleep(15);
      const err = new Error('Missing header at C:\\Users\\dev\\game\\splash.wgsl.pack?token=topsecret (404)');
      err.status = 404;
      throw err;
    },
    preloadReflectionSidecar: async () => {
      await sleep(15);
      const err = new Error('Sidecar not found at /Users/dev/splash.wgsl.pack.reflect.json');
      err.code = 'NOT_FOUND';
      throw err;
    },
  };

  const { orchestrator, launches } = createOrchestrator({
    fetchImpl: createMockFetch(
      readyStatus({
        shaderPackUrl: '/data/shaders/splash.wgsl.pack',
        reflectionManifestUrl: null,
      })
    ),
    preloader,
    bootTracer,
    preloadRuntimeWasm: async () => {
      await sleep(15);
      throw new Error('Transient wasm prefetch error Bearer secret_tok_99');
    },
    opfsProbe: async () => {
      await sleep(15);
      throw new Error('OPFS unavailable in private mode');
    },
    parallelPreload: true,
  });

  const result = await orchestrator.start();
  assert.equal(result.ok, true);
  assert.equal(result.phase, BOOT_PHASE.LAUNCHING_RUNTIME);
  assert.equal(launches.length, 1);
  assert.equal(launches[0].bundle.pgim.header.magic, 'PGIM');
  assert.equal(launches[0].bundle.shaderPackIndex, null);
  assert.equal(launches[0].bundle.reflectionSidecar, null);
  assert.equal(launches[0].bundle.runtimeWasm, null);
  assert.equal(launches[0].bundle.opfsWarmBoot, null);

  const stats = orchestrator.getStats();
  assert.equal(stats.preloadStats.completedCount, 1);
  assert.equal(stats.preloadStats.optionalMissingCount, 4);
  assert.equal(stats.preloadStats.failedCount, 0);
  assert.equal(stats.parallelStagesCount, 5);
  assert.equal(stats.overlapSavedMs > 0, true);

  // Sanitization check: no host paths, query tokens, or bearer tokens leak in getStats().
  const serialized = JSON.stringify(stats);
  assert.equal(serialized.includes('topsecret'), false);
  assert.equal(serialized.includes('secret_tok_99'), false);
  assert.equal(serialized.includes('C:\\\\Users'), false);
  assert.equal(serialized.includes('/Users/dev'), false);
  assert.ok(progressEvents.some((ev) => ev.phase === 'optional-missing' && ev.asset === 'opfsWarmBoot'));
  assert.ok(progressEvents.some((ev) => ev.phase === 'optional-missing' && ev.asset === 'runtimeWasm'));

  // Corrupt validation error (BAD_*) on runtimeWasm MUST fail boot deterministically.
  const corruptWasmErr = new Error('Corrupt runtime.wasm magic header');
  corruptWasmErr.code = 'BAD_WASM_BINARY';
  const badWasm = createOrchestrator({
    fetchImpl: createMockFetch(readyStatus()),
    preloader: {
      preloadPgim: async () => ({ header: { magic: 'PGIM' }, bytesLoaded: 64 }),
    },
    preloadRuntimeWasm: async () => {
      throw corruptWasmErr;
    },
    parallelPreload: true,
  });
  const badRes = await badWasm.orchestrator.start();
  assert.equal(badRes.ok, false);
  assert.equal(badRes.phase, BOOT_PHASE.ERROR);
  assert.match(badRes.error.message, /Corrupt runtime\.wasm/);
  assert.equal(badWasm.orchestrator.getStats().overlapSavedMs, 0);
});

test('W309 preserves W258 buffer release across all concurrent preload stages and resets telemetry cleanly on reset()', async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const pgimBuf = new ArrayBuffer(512);
  const hdrBuf = new ArrayBuffer(128);
  const sidecarBuf = new ArrayBuffer(96);
  const wasmBuf = new ArrayBuffer(1024);
  const opfsBuf = new ArrayBuffer(64);
  const compiledModule = { kind: 'WebAssembly.Module', valid: true };

  const preloader = {
    preloadPgim: async () => {
      await sleep(25);
      return {
        header: { magic: 'PGIM' },
        bytes: new Uint8Array(pgimBuf),
        buffer: pgimBuf,
        bytesLoaded: 512,
      };
    },
    preloadShaderPackHeader: async () => {
      await sleep(20);
      return {
        magic: 'WGPK',
        entryCount: 16,
        bytes: new Uint8Array(hdrBuf),
        bytesLoaded: 128,
      };
    },
    preloadReflectionSidecar: async () => {
      await sleep(20);
      return {
        manifest: { stages: 4 },
        bytes: new Uint8Array(sidecarBuf),
        bytesLoaded: 96,
      };
    },
  };

  let observedDuringLaunch = null;
  const { orchestrator, launches } = createOrchestrator({
    fetchImpl: createMockFetch(
      readyStatus({
        shaderPackUrl: '/data/shaders/splash.wgsl.pack',
        reflectionManifestUrl: null,
      })
    ),
    preloader,
    preloadRuntimeWasm: async () => {
      await sleep(25);
      return {
        module: compiledModule,
        bytes: new Uint8Array(wasmBuf),
        buffer: wasmBuf,
        bytesLoaded: 1024,
        url: '/dist/runtime.wasm',
      };
    },
    opfsProbe: async () => {
      await sleep(15);
      return {
        warm: true,
        bytes: new Uint8Array(opfsBuf),
        bytesLoaded: 64,
      };
    },
    parallelPreload: true,
    releasePreloadBuffers: true,
    onLaunchRuntime: async (payload) => {
      launches.push(payload);
      observedDuringLaunch = {
        pgimLen: payload.bundle.pgim.bytes.byteLength,
        hdrLen: payload.bundle.shaderPackIndex.bytes.byteLength,
        sidecarLen: payload.bundle.reflectionSidecar.bytes.byteLength,
        wasmLen: payload.bundle.runtimeWasm.bytes.byteLength,
        wasmBufLen: payload.bundle.runtimeWasm.buffer.byteLength,
        opfsLen: payload.bundle.opfsWarmBoot.bytes.byteLength,
      };
    },
  });

  const result = await orchestrator.start();
  assert.equal(result.ok, true);
  assert.deepEqual(observedDuringLaunch, {
    pgimLen: 512,
    hdrLen: 128,
    sidecarLen: 96,
    wasmLen: 1024,
    wasmBufLen: 1024,
    opfsLen: 64,
  });

  // After worker handoff, W258 buffer release nulled every large binary payload while preserving metadata & compiled module.
  assert.equal(orchestrator.getState().bundle, null);
  assert.equal(launches[0].bundle.pgim.bytes, null);
  assert.equal(launches[0].bundle.pgim.buffer, null);
  assert.equal(launches[0].bundle.shaderPackIndex.bytes, null);
  assert.equal(launches[0].bundle.reflectionSidecar.bytes, null);
  assert.equal(launches[0].bundle.runtimeWasm.bytes, null);
  assert.equal(launches[0].bundle.runtimeWasm.buffer, null);
  assert.equal(launches[0].bundle.opfsWarmBoot.bytes, null);

  assert.equal(launches[0].bundle.runtimeWasm.module, compiledModule);
  assert.equal(launches[0].bundle.runtimeWasm.bytesLoaded, 1024);
  assert.equal(launches[0].bundle.opfsWarmBoot.warm, true);

  // Telemetry stats remain intact after W258 release.
  const preResetStats = orchestrator.getStats();
  assert.equal(preResetStats.parallelStagesCount, 5);
  assert.equal(preResetStats.overlapSavedMs > 0, true);
  assert.equal(preResetStats.serialEstimateMs > preResetStats.parallelWallMs, true);

  // Direct releaseBootBundleBytes also counts runtimeWasm & opfsWarmBoot without double-counting aliased buffers.
  const standaloneWasmBuf = new ArrayBuffer(2048);
  const standaloneRelease = releaseBootBundleBytes({
    runtimeWasm: {
      bytes: new Uint8Array(standaloneWasmBuf),
      buffer: standaloneWasmBuf,
      module: compiledModule,
    },
    opfsWarmBoot: new Uint8Array(32),
  });
  assert.equal(standaloneRelease.releasedBytes, 2048 + 32);

  // reset() clears W309 overlap telemetry cleanly and allows a fresh subsequent boot.
  const resetRes = orchestrator.reset();
  assert.equal(resetRes.ok, true);
  assert.equal(resetRes.phase, BOOT_PHASE.IDLE);

  const afterReset = orchestrator.getStats();
  assert.equal(afterReset.phase, BOOT_PHASE.IDLE);
  assert.equal(afterReset.serialEstimateMs, 0);
  assert.equal(afterReset.parallelWallMs, 0);
  assert.equal(afterReset.overlapSavedMs, 0);
  assert.equal(afterReset.parallelPreloadSavedMs, 0);
  assert.equal(afterReset.parallelStagesCount, 0);
  assert.equal(afterReset.preloadStats.taskCount, 0);
  assert.deepEqual(afterReset.preloadTaskTimingsMs, {});
});
