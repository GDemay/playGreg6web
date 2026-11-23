import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import { BootTracer, BOOT_MILESTONES } from '../web/telemetry/boot_tracer.js';

const require = createRequire(import.meta.url);
const orchestratorMod = require('../web/runtime/boot_orchestrator.js');
const { BOOT_PHASE, BootOrchestrator, DEFAULT_STATUS_URL, discoverBootAssets } = orchestratorMod;

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
