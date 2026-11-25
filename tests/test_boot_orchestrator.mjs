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
  discoverBootAssets,
  isRealGuestBoot,
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
