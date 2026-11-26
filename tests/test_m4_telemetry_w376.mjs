import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import {
  BOOT_MILESTONES,
  FAST_BOOT_MILESTONES,
  PLAYABILITY_STATE,
  BootTracer,
  parseLogLine,
} from '../web/telemetry/boot_tracer.js';
import {
  HudController,
  HIDDEN_HUD_MIN_RENDER_INTERVAL_MS,
  PLAYABILITY_STATE as HUD_PLAYABILITY_STATE,
  sanitizePercentage,
  formatIoMailboxLanes,
  formatOpfsWarmBootStatus,
  formatAudioFrameCount,
  formatAudioDrainedDurationSec,
  formatAudioStatus,
  formatAudioRingSummary,
  formatGuestAudioStatus,
  formatDiagnosticsSummary,
} from '../web/ui/hud_controller.js';

const require = createRequire(import.meta.url);
const {
  BOOT_STAGE_ORDER,
  POST_PRESENT_BOOT_STAGES,
  ALL_BOOT_STAGE_ORDER,
  PLAYABILITY_STATE: PANEL_PLAYABILITY_STATE,
  sanitizePercentage: panelSanitizePercentage,
  sanitizeAudioOutputState,
  formatMailboxLanes,
  formatAudioFrameCount: panelFormatAudioFrameCount,
  formatOpfsHitRate,
  formatOpfsBytes,
  formatSummary,
  buildMissingShaderExport,
  GpuBootPanelController,
} = require('../web/ui/gpu_boot_panel.js');

// --- Shared lightweight DOM mocks ---

function createSpyDisplayElement(initial = {}) {
  let _textContent = initial.textContent !== undefined ? initial.textContent : '';
  let _title = initial.title !== undefined ? initial.title : '';
  let _hidden = initial.hidden !== undefined ? initial.hidden : false;
  const attributes = new Map();
  const counts = {
    textWrites: 0,
    titleWrites: 0,
    hiddenWrites: 0,
    attrWrites: 0,
  };

  return {
    counts,
    dataset: { ...(initial.dataset || {}) },
    style: { ...(initial.style || {}) },
    get textContent() {
      return _textContent;
    },
    set textContent(v) {
      counts.textWrites += 1;
      _textContent = v;
    },
    get title() {
      return _title;
    },
    set title(v) {
      counts.titleWrites += 1;
      _title = v;
    },
    get hidden() {
      return _hidden;
    },
    set hidden(v) {
      counts.hiddenWrites += 1;
      _hidden = v;
    },
    setAttribute(name, value) {
      counts.attrWrites += 1;
      attributes.set(name, String(value));
    },
    getAttribute(name) {
      return attributes.has(name) ? attributes.get(name) : null;
    },
  };
}

class MockPanelElement {
  constructor(tagName, ownerDocument) {
    this.tagName = String(tagName).toUpperCase();
    this.ownerDocument = ownerDocument;
    this.children = [];
    this.parentNode = null;
    this.attributes = new Map();
    this.style = { display: '', width: '' };
    this._hidden = false;
    this._textContent = '';
    this.textContentWrites = 0;
    this.childMutations = 0;
    this.className = '';
    this._listeners = new Map();
    this.onclick = null;
  }

  get hidden() {
    return this._hidden;
  }

  set hidden(v) {
    this._hidden = Boolean(v);
  }

  get textContent() {
    return this._textContent;
  }

  set textContent(v) {
    this.textContentWrites += 1;
    this._textContent = String(v ?? '');
  }

  setAttribute(name, value) {
    this.attributes.set(String(name), String(value));
    if (name === 'class') this.className = String(value);
    if (name === 'hidden') this._hidden = true;
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
    this.childMutations += 1;
    return child;
  }

  removeChild(child) {
    const idx = this.children.indexOf(child);
    if (idx >= 0) {
      this.children.splice(idx, 1);
      child.parentNode = null;
      this.childMutations += 1;
    }
    return child;
  }

  addEventListener(type, fn) {
    if (!this._listeners.has(type)) this._listeners.set(type, new Set());
    this._listeners.get(type).add(fn);
  }

  removeEventListener(type, fn) {
    const set = this._listeners.get(type);
    if (set) set.delete(fn);
  }

  querySelector(selector) {
    const matches = this.querySelectorAll(selector);
    return matches.length > 0 ? matches[0] : null;
  }

  querySelectorAll(selector) {
    const out = [];
    const match = (el) => {
      if (selector.startsWith('.')) {
        const cls = selector.slice(1);
        return String(el.className || '')
          .split(/\s+/)
          .includes(cls);
      }
      if (selector.startsWith('style[')) {
        const attr = selector.slice('style['.length, -1);
        return el.tagName === 'STYLE' && el.getAttribute(attr) !== null;
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
}

class MockPanelDocument {
  constructor() {
    this.head = new MockPanelElement('head', this);
    this.body = new MockPanelElement('body', this);
    this.documentElement = new MockPanelElement('html', this);
    this.documentElement.appendChild(this.head);
    this.documentElement.appendChild(this.body);
  }

  createElement(tag) {
    return new MockPanelElement(tag, this);
  }

  querySelector(selector) {
    return this.head.querySelector(selector) || this.body.querySelector(selector);
  }
}

// =============================================================================
// 1. BootTracer M4 milestones, log heuristics, m4Phase/playabilityState & liveTelemetry
// =============================================================================

test('W376 BootTracer: enhanced M4 log heuristics recognize MAIN_MENU and IN_GAME patterns without false positives on failure lines', () => {
  // Positive M4 patterns
  assert.deepEqual(parseLogLine('[boot] story mode menu ready')?.milestones, ['MAIN_MENU']);
  assert.deepEqual(parseLogLine('[wasm] main menu reached after legal notices')?.milestones, ['MAIN_MENU']);
  assert.deepEqual(parseLogLine('[boot] title screen loaded')?.milestones, ['MAIN_MENU']);
  assert.deepEqual(parseLogLine('[boot] press A to start')?.milestones, ['START_SCREEN']);
  assert.deepEqual(parseLogLine('[wasm] world loaded - entering Vice City')?.milestones, ['IN_GAME']);
  assert.deepEqual(parseLogLine('[boot] gameplay active (60 fps)')?.milestones, ['IN_GAME']);
  assert.deepEqual(parseLogLine('[wasm] playable state reached')?.milestones, ['IN_GAME']);
  assert.deepEqual(parseLogLine('[boot] player is in-game')?.milestones, ['IN_GAME']);

  // Failure / error / timeout / not-reached lines MUST NOT trigger false positives!
  const falsePositives = [
    '[boot] main menu failed to load',
    '[wasm] story mode error: missing pak',
    '[boot] title screen timeout after 30s',
    '[wasm] world loaded timed out',
    '[boot] gameplay active exception thrown',
    '[boot] MAIN_MENU not reached before deadline',
    '[wasm] IN_GAME never reached',
    '[boot] crashed during world loaded transition',
    '[boot] abort while entering story mode',
  ];
  for (const line of falsePositives) {
    assert.equal(parseLogLine(line), null, `Must not match failure line: ${line}`);
  }
});

test('W376 BootTracer: tracks m4Phase / playabilityState and sanitized live M4 telemetry across summary, snapshot, Chrome trace, and reset', () => {
  assert.deepEqual(PLAYABILITY_STATE, {
    BOOTING: 'BOOTING',
    FIRST_PRESENT: 'FIRST_PRESENT',
    START_SCREEN: 'START_SCREEN',
    INTRO_VIDEO: 'INTRO_VIDEO',
    MAIN_MENU: 'MAIN_MENU',
    IN_GAME: 'IN_GAME',
  });
  assert.equal(BOOT_MILESTONES.length, 9);
  assert.deepEqual(FAST_BOOT_MILESTONES, ['START_SCREEN', 'INTRO_VIDEO', 'MAIN_MENU', 'IN_GAME']);

  let nowMs = 1000;
  const tracer = new BootTracer({ now: () => nowMs });

  const s0 = tracer.getSummary();
  assert.equal(s0.m4Phase, 'BOOTING');
  assert.equal(s0.playabilityState, 'BOOTING');
  assert.equal(s0.progressPct, 0);
  assert.equal(s0.liveTelemetry, null);

  nowMs += 500;
  tracer.markMilestone('PGIM_LOAD');
  assert.equal(tracer.getSummary().playabilityState, 'BOOTING');
  assert.equal(tracer.getSummary().progressPct, 11);

  nowMs += 1500;
  tracer.markMilestone('FIRST_PRESENT');
  assert.equal(tracer.getSummary().playabilityState, 'FIRST_PRESENT');
  assert.equal(tracer.getSummary().m4Phase, 'FIRST_PRESENT');
  assert.equal(tracer.getSummary().progressPct, 100);
  assert.equal(tracer.getSummary().readyForSplash, true);

  nowMs += 1000;
  tracer.recordStartScreen();
  assert.equal(tracer.getSummary().playabilityState, 'START_SCREEN');
  assert.equal(tracer.getSummary().bootToStartScreenMs, 3000);

  nowMs += 1000;
  tracer.recordIntroVideo();
  assert.equal(tracer.getSummary().playabilityState, 'INTRO_VIDEO');
  assert.equal(tracer.getSummary().bootToIntroVideoMs, 4000);

  nowMs += 2000;
  tracer.recordMainMenu();
  assert.equal(tracer.getSummary().playabilityState, 'MAIN_MENU');
  assert.equal(tracer.getSummary().bootToMainMenuMs, 6000);

  nowMs += 4000;
  tracer.recordInGame();
  const sInGame = tracer.getSummary();
  assert.equal(sInGame.stage, 'IN_GAME');
  assert.equal(sInGame.m4Phase, 'IN_GAME');
  assert.equal(sInGame.playabilityState, 'IN_GAME');
  assert.equal(sInGame.progressPct, 100);
  assert.equal(sInGame.bootToInGameMs, 10000);

  // Attach sanitized live M4 telemetry (8-lane HTTPFS, audio ring, OPFS warm-boot).
  const live = tracer.recordLiveTelemetry({
    io: {
      mailboxLanes: 8,
      activeMailboxLanes: 6,
      httpfsPeakConcurrency: 8,
      rangeCacheHitRatePct: 94.2,
    },
    audio: {
      outputState: 'active',
      guestRingAttached: true,
      guestFramesDrained: 12_600_000,
      framesDrained: 12_600_000,
      guestUnderruns: 0,
      overruns: 0,
    },
    preloadTelemetry: {
      opfsHits: 18,
      opfsMisses: 2,
      opfsWrites: 2,
      opfsInvalidations: 0,
      bytesFromOpfs: 128 * 1024 * 1024,
      bytesFromNetwork: 16 * 1024 * 1024,
    },
  });

  assert.equal(live.httpfs.mailboxLanes, 8);
  assert.equal(live.httpfs.activeMailboxLanes, 6);
  assert.equal(live.httpfs.peakMailboxLanes, 8);
  assert.equal(live.httpfs.rangeCacheHitRatePct, 94.2);
  assert.equal(live.audio.outputState, 'active');
  assert.equal(live.audio.guestRingAttached, true);
  assert.equal(live.audio.guestFramesDrained, 12_600_000);
  assert.equal(live.opfs.opfsHits, 18);
  assert.equal(live.opfs.opfsMisses, 2);
  assert.equal(live.opfs.opfsHitRatePct, 90);

  const summaryWithLive = tracer.getSummary();
  assert.deepEqual(summaryWithLive.liveTelemetry, live);
  const trace = tracer.toChromeTraceEvents();
  assert.equal(trace.metadata.m4Phase, 'IN_GAME');
  assert.equal(trace.metadata.playabilityState, 'IN_GAME');
  assert.equal(trace.metadata.liveTelemetry.httpfs.peakMailboxLanes, 8);
  assert.equal(trace.metadata.liveTelemetry.opfs.opfsHitRatePct, 90);

  // Hostile inputs sanitize cleanly with zero leaks.
  tracer.recordLiveTelemetry({
    mailboxLanes: -5,
    peakMailboxLanes: 'C:\\Users\\alice\\secret.rpf',
    outputState: '/home/bob/audio.awc',
    guestRingAttached: 'yes-fake',
    opfsHits: Number.NaN,
    opfsHitRatePct: 999,
  });
  const afterHostile = tracer.getLiveTelemetry();
  assert.equal(afterHostile.httpfs.mailboxLanes, null);
  assert.equal(afterHostile.httpfs.peakMailboxLanes, null);
  assert.equal(afterHostile.audio.outputState, null);
  assert.equal(afterHostile.audio.guestRingAttached, false);
  assert.equal(afterHostile.opfs.opfsHitRatePct, 100);
  assert.doesNotMatch(tracer.toJson(), /alice|bob|secret\.rpf|audio\.awc/i);

  // reset() clears playabilityState and liveTelemetry cleanly.
  tracer.reset();
  assert.equal(tracer.getSummary().playabilityState, 'BOOTING');
  assert.equal(tracer.getSummary().liveTelemetry, null);
  assert.equal(tracer.getLiveTelemetry(), null);
});

// =============================================================================
// 2. HudController M4 milestones, 8-lane HTTPFS, audio ring, OPFS & W364 zero-DOM-churn
// =============================================================================

test('W376 HudController: recognizes post-present M4 milestones without breaking core 0-100% progress or getState() contract', () => {
  assert.deepEqual(HUD_PLAYABILITY_STATE, {
    BOOTING: 'BOOTING',
    FIRST_PRESENT: 'FIRST_PRESENT',
    START_SCREEN: 'START_SCREEN',
    INTRO_VIDEO: 'INTRO_VIDEO',
    MAIN_MENU: 'MAIN_MENU',
    IN_GAME: 'IN_GAME',
  });

  const bootStageEl = createSpyDisplayElement();
  const bootProgressFillEl = createSpyDisplayElement();
  const controller = new HudController({
    bootMode: 'real-wasm64',
    displayElements: { bootStageEl, bootProgressFillEl },
  });

  // Core milestone progress stays 100% unchanged (PGIM_LOAD=11%, FIRST_PRESENT=100%).
  controller.updateMetrics({ bootSummary: { stage: 'PGIM_LOAD' } });
  assert.equal(controller.getBootStatus().progressPct, 11);
  assert.equal(controller.getBootStatus().playabilityState, 'BOOTING');
  assert.equal(controller.getBootStatus().readyForSplash, false);

  controller.updateMetrics({
    bootSummary: { stage: 'FIRST_VISIBLE_FRAME', firstVisibleFrameMs: 1200 },
  });
  assert.equal(controller.getBootStatus().bootStage, 'FIRST_VISIBLE_FRAME');
  assert.equal(controller.getBootStatus().progressPct, 100);
  assert.equal(controller.getBootStatus().readyForSplash, true);
  assert.equal(controller.getBootStatus().playabilityState, 'FIRST_PRESENT');

  controller.ingestProgressLog('[boot] Boot 100%: START_SCREEN');
  assert.equal(controller.getBootStatus().bootStage, 'START_SCREEN');
  assert.equal(controller.getBootStatus().playabilityState, 'START_SCREEN');

  controller.updateMetrics({
    bootSummary: { stage: 'MAIN_MENU', bootToMainMenuMs: 4200, elapsedMs: 4200 },
  });
  assert.equal(controller.getBootStatus().bootStage, 'MAIN_MENU');
  assert.equal(controller.getBootStatus().playabilityState, 'MAIN_MENU');
  assert.equal(controller.getBootStatus().bootMilestoneTimings.bootToMainMenuMs, 4200);
  assert.equal(bootStageEl.textContent, 'MAIN_MENU');
  assert.equal(bootStageEl.dataset.state, 'ok');

  controller.updateTelemetry({
    bootSummary: { stage: 'IN_GAME', bootToInGameMs: 8500, elapsedMs: 8500 },
  });
  assert.equal(controller.getBootStatus().bootStage, 'IN_GAME');
  assert.equal(controller.getBootStatus().playabilityState, 'IN_GAME');
  assert.equal(controller.getBootStatus().bootMilestoneTimings.bootToInGameMs, 8500);
  assert.equal(controller.getBootStatus().progressPct, 100);
  assert.equal(bootStageEl.textContent, 'IN_GAME');
  assert.equal(bootProgressFillEl.style.width, '100.0%');

  // Exact 7-key getState() contract MUST remain untouched.
  assert.deepEqual(Object.keys(controller.getState()).sort(), [
    'activeProfileId',
    'boundControlCount',
    'destroyed',
    'isRecording',
    'lastAction',
    'sessionMode',
    'sessionState',
  ]);

  controller.destroy();
});

test('W376 HudController: 8-lane HTTPFS concurrency, live audio ring frames drained & OPFS warm-boot hit rate format, render, sanitize, and reset cleanly', () => {
  // Pure sanitizer & formatter checks (keeping backward-compatible contracts intact!)
  assert.equal(sanitizePercentage(94.24), 94.2);
  assert.equal(sanitizePercentage(125), 100);
  assert.equal(sanitizePercentage(0.9, true), 90);
  assert.equal(sanitizePercentage(-1), null);
  assert.equal(sanitizePercentage('C:\\Users\\alice\\90'), null);

  assert.equal(formatIoMailboxLanes(8), 'I/O lanes: 8');
  assert.equal(formatIoMailboxLanes({ mailboxLanes: 8 }), 'I/O lanes: 8');
  assert.equal(
    formatIoMailboxLanes({
      mailboxLanes: 8,
      activeMailboxLanes: 8,
      peakMailboxLanes: 8,
      rangeCacheHitRatePct: 94.2,
    }),
    'I/O: 8/8 lanes (peak 8, cache 94.2%)'
  );
  assert.equal(
    formatIoMailboxLanes({ mailboxLanes: 8, httpfsPeakConcurrency: 8 }),
    'I/O lanes: 8 (peak 8)'
  );

  // Audio formatters: preserve existing formatGuestAudioStatus while adding human-readable ring summary
  assert.equal(
    formatGuestAudioStatus({
      guestRingAttached: true,
      guestFramesDrained: 4800,
      guestUnderruns: 1,
      guestPeakLevel: 0.42,
    }),
    'Guest PCM: attached (4800 drained, 1 underrun, peak 42%)'
  );
  assert.equal(formatAudioFrameCount(12_600_000), '12.6M frames');
  assert.equal(formatAudioFrameCount(480_000), '480.0K frames');
  assert.equal(formatAudioFrameCount(4800), '4800 frames');
  assert.equal(formatAudioDrainedDurationSec(12_600_000), 262.5);
  assert.equal(
    formatAudioRingSummary({
      outputState: 'active',
      guestRingAttached: true,
      guestFramesDrained: 12_600_000,
      guestUnderruns: 0,
      overruns: 0,
    }),
    'Audio: ring attached (12.6M frames / 262.5s, active)'
  );
  assert.equal(
    formatAudioStatus({
      outputState: 'active',
      framesDrained: 9600,
      guestRingAttached: true,
      guestFramesDrained: 4800,
      guestUnderruns: 1,
      guestPeakLevel: 0.42,
    }),
    'Audio: active (9600 drained) · Guest PCM: attached (4800 drained, 1 underrun, peak 42%)'
  );

  // OPFS warm-boot formatter
  assert.equal(
    formatOpfsWarmBootStatus({
      opfsHits: 18,
      opfsMisses: 2,
      bytesFromOpfs: 128 * 1024 * 1024,
    }),
    'OPFS warm-boot: 90% (18/20 hits, 128 MiB cached)'
  );
  assert.equal(formatOpfsWarmBootStatus({}), '');
  assert.equal(formatOpfsWarmBootStatus(null), '');

  const ioMailboxLanesEl = createSpyDisplayElement();
  const opfsWarmBootEl = createSpyDisplayElement();
  const audioRingEl = createSpyDisplayElement();
  const diagnosticsSummaryEl = createSpyDisplayElement();

  const controller = new HudController({
    bootMode: 'real-wasm64',
    displayElements: {
      ioMailboxLanesEl,
      opfsWarmBootEl,
      audioRingEl,
      diagnosticsSummaryEl,
    },
  });

  // Ingest via updateMetrics (including nested preloadTelemetry & 8-lane HTTPFS & audio ring)
  controller.updateMetrics({
    mailboxLanes: 8,
    activeMailboxLanes: 6,
    httpfsPeakConcurrency: 8,
    slabHitRatePct: 91.5,
    audio: {
      outputState: 'active',
      framesDrained: 12_600_000,
      guestRingAttached: true,
      guestFramesDrained: 12_600_000,
      guestUnderruns: 0,
      overruns: 0,
    },
    preloadTelemetry: {
      opfsHits: 18,
      opfsMisses: 2,
      opfsWrites: 2,
      opfsInvalidations: 1,
      bytesFromOpfs: 128 * 1024 * 1024,
      bytesFromNetwork: 16 * 1024 * 1024,
    },
  });

  const diag = controller.getDiagnosticsSnapshot();
  assert.equal(diag.mailboxLanes, 8);
  assert.equal(diag.activeMailboxLanes, 6);
  assert.equal(diag.peakMailboxLanes, 8);
  assert.equal(diag.rangeCacheHitRatePct, 91.5);
  assert.equal(diag.opfsHits, 18);
  assert.equal(diag.opfsMisses, 2);
  assert.equal(diag.opfsWrites, 2);
  assert.equal(diag.opfsInvalidations, 1);
  assert.equal(diag.bytesFromOpfs, 128 * 1024 * 1024);
  assert.equal(diag.bytesFromNetwork, 16 * 1024 * 1024);
  assert.equal(diag.opfsHitRatePct, 90);
  assert.equal(diag.audioRingSummaryText, 'Audio: ring attached (12.6M frames / 262.5s, active)');

  assert.equal(ioMailboxLanesEl.textContent, 'I/O: 6/8 lanes (peak 8, cache 91.5%)');
  assert.equal(ioMailboxLanesEl.dataset.peakMailboxLanes, '8');
  assert.equal(opfsWarmBootEl.textContent, 'OPFS warm-boot: 90% (18/20 hits, 128 MiB cached)');
  assert.equal(opfsWarmBootEl.dataset.opfsHitRatePct, '90');
  assert.equal(opfsWarmBootEl.dataset.state, 'ok');
  assert.equal(audioRingEl.textContent, 'Audio: ring attached (12.6M frames / 262.5s, active)');
  assert.equal(audioRingEl.dataset.state, 'ok');
  assert.match(diagnosticsSummaryEl.textContent, /I\/O: 6\/8 lanes \(peak 8, cache 91\.5%\)/);
  assert.match(diagnosticsSummaryEl.textContent, /OPFS warm-boot: 90%/);
  assert.match(diagnosticsSummaryEl.textContent, /Audio: ring attached \(12\.6M frames \/ 262\.5s, active\)/);

  // Verify formatDiagnosticsSummary standalone with all M4 fields
  assert.match(
    formatDiagnosticsSummary({
      io: { mailboxLanes: 8, activeMailboxLanes: 8, peakMailboxLanes: 8 },
      opfs: { opfsHits: 9, opfsMisses: 1 },
    }),
    /I\/O: 8\/8 lanes \(peak 8\) · OPFS warm-boot: 90% \(9\/10 hits\)/
  );

  // Hostile inputs sanitize safely with zero host-path or XSS leaks.
  controller.setOpfsWarmBootStats({
    opfsHits: 'C:\\Users\\alice\\opfs.bin',
    opfsMisses: -4,
    opfsHitRatePct: '<script>alert(1)</script>',
  });
  const afterHostile = controller.getDiagnosticsSnapshot();
  assert.equal(afterHostile.opfsHits, 0);
  assert.equal(afterHostile.opfsMisses, 0);
  assert.equal(afterHostile.opfsHitRatePct, null);
  assert.doesNotMatch(JSON.stringify(afterHostile), /alice|opfs\.bin|<script>/i);

  // resetTelemetry clears OPFS and IO concurrency cleanly.
  controller.setOpfsWarmBootStats({ opfsHits: 5, opfsMisses: 0 });
  controller.resetTelemetry();
  assert.equal(controller.getDiagnosticsSnapshot().opfsHitRatePct, null);
  assert.equal(controller.getDiagnosticsSnapshot().opfsHits, 0);
  assert.equal(controller.getDiagnosticsSnapshot().peakMailboxLanes, null);
  assert.equal(opfsWarmBootEl.textContent, '--');

  controller.destroy();
});

test('W376 HudController: preserves W364 hidden-HUD zero-DOM-churn invariant for all new M4 telemetry elements', () => {
  assert.equal(HIDDEN_HUD_MIN_RENDER_INTERVAL_MS, 500);

  let nowMs = 2000;
  const hudRootEl = createSpyDisplayElement();
  const framesEl = createSpyDisplayElement();
  const bootStageEl = createSpyDisplayElement();
  const opfsWarmBootEl = createSpyDisplayElement();
  const audioRingEl = createSpyDisplayElement();
  const ioMailboxLanesEl = createSpyDisplayElement();
  const diagnosticsSummaryEl = createSpyDisplayElement();

  const controller = new HudController({
    bootMode: 'real-wasm64',
    visible: false,
    now: () => nowMs,
    displayElements: {
      hudRootEl,
      framesEl,
      bootStageEl,
      opfsWarmBootEl,
      audioRingEl,
      ioMailboxLanesEl,
      diagnosticsSummaryEl,
    },
  });

  // 1. Initial M4 update while hidden updates core bootStageEl/framesEl, but NEVER writes visual-only M4 elements.
  controller.updateMetrics({
    bootSummary: { stage: 'MAIN_MENU', bootToMainMenuMs: 3200 },
    framesPresented: 60,
    mailboxLanes: 8,
    activeMailboxLanes: 8,
    peakMailboxLanes: 8,
    opfsHits: 10,
    opfsMisses: 0,
    audio: { outputState: 'active', guestRingAttached: true, guestFramesDrained: 960_000 },
  });

  assert.equal(bootStageEl.textContent, 'MAIN_MENU');
  assert.equal(framesEl.textContent, '60');
  assert.equal(opfsWarmBootEl.counts.textWrites, 0, 'opfsWarmBootEl must not be written while HUD hidden');
  assert.equal(audioRingEl.counts.textWrites, 0, 'audioRingEl must not be written while HUD hidden');
  assert.equal(ioMailboxLanesEl.counts.textWrites, 0, 'ioMailboxLanesEl must not be written while HUD hidden');
  assert.equal(diagnosticsSummaryEl.counts.textWrites, 0, 'diagnosticsSummaryEl must not be written while HUD hidden');

  // 2. Rapid updates inside the 500ms hidden throttle window incur ZERO DOM writes total.
  const domWritesBeforeBurst = controller.domWrites;
  nowMs = 2100;
  controller.updateMetrics({ framesPresented: 65, opfsHits: 12, opfsMisses: 0 });
  nowMs = 2250;
  controller.updateMetrics({ framesPresented: 70, opfsHits: 15, opfsMisses: 0 });
  nowMs = 2499;
  controller.updateMetrics({
    bootSummary: { stage: 'IN_GAME', bootToInGameMs: 5000 },
    framesPresented: 75,
    audio: { outputState: 'active', guestRingAttached: true, guestFramesDrained: 2_400_000 },
  });
  assert.equal(controller.domWrites, domWritesBeforeBurst, 'zero DOM writes during hidden 500ms throttle window');
  assert.equal(controller.getRenderStats().throttledRenders, 3);

  // 3. Even after >= 500ms while hidden, visual-only M4 elements STILL incur zero writes.
  nowMs = 2600;
  controller.updateMetrics({ framesPresented: 90, opfsHits: 19, opfsMisses: 1 });
  assert.equal(framesEl.textContent, '90');
  assert.equal(bootStageEl.textContent, 'IN_GAME');
  assert.equal(opfsWarmBootEl.counts.textWrites, 0);
  assert.equal(audioRingEl.counts.textWrites, 0);
  assert.equal(ioMailboxLanesEl.counts.textWrites, 0);
  assert.equal(diagnosticsSummaryEl.counts.textWrites, 0);

  // 4. Showing the HUD immediately flushes all deferred M4 visual elements.
  controller.setVisible(true);
  assert.equal(opfsWarmBootEl.textContent, 'OPFS warm-boot: 95% (19/20 hits)');
  assert.equal(audioRingEl.textContent, 'Audio: ring attached (2.4M frames / 50.0s, active)');
  assert.equal(ioMailboxLanesEl.textContent, 'I/O: 8/8 lanes (peak 8)');
  assert.match(diagnosticsSummaryEl.textContent, /OPFS warm-boot: 95%/);

  controller.destroy();
});

// =============================================================================
// 3. GpuBootPanelController M4 milestones, live telemetry, sanitization & collapsed/hidden zero-DOM-churn
// =============================================================================

test('W376 GpuBootPanelController: recognizes M4 post-present milestones, renders 8-lane HTTPFS / audio ring / OPFS cards, exports sanitized JSON, and resets cleanly', () => {
  assert.equal(BOOT_STAGE_ORDER.length, 9);
  assert.deepEqual(POST_PRESENT_BOOT_STAGES, [
    'FIRST_VISIBLE_FRAME',
    'START_SCREEN',
    'INTRO_VIDEO',
    'MAIN_MENU',
    'IN_GAME',
  ]);
  assert.equal(ALL_BOOT_STAGE_ORDER.length, 14);
  assert.equal(PANEL_PLAYABILITY_STATE.IN_GAME, 'IN_GAME');

  // Pure helper checks (including backward-compatible formatMailboxLanes)
  assert.equal(formatMailboxLanes(8), '8');
  assert.equal(formatMailboxLanes(8, { peakMailboxLanes: 8 }), '8 lanes (peak 8)');
  assert.equal(
    formatMailboxLanes({ mailboxLanes: 8, activeMailboxLanes: 6, peakMailboxLanes: 8 }),
    '6/8 lanes (peak 8)'
  );
  assert.equal(panelSanitizePercentage(95.46), 95.5);
  assert.equal(panelSanitizePercentage('C:\\secret'), null);
  assert.equal(sanitizeAudioOutputState('active'), 'active');
  assert.equal(sanitizeAudioOutputState('headless_drain'), 'headless-drain');
  assert.equal(sanitizeAudioOutputState('<script>alert(1)</script>'), null);
  assert.equal(panelFormatAudioFrameCount(12_600_000), '12.6M');
  assert.equal(formatOpfsHitRate(90), '90.0%');
  assert.equal(formatOpfsBytes(128 * 1024 * 1024), '128.0 MiB');

  const doc = new MockPanelDocument();
  const controller = new GpuBootPanelController({
    container: doc.body,
    documentObj: doc,
    now: () => Date.UTC(2026, 10, 26, 12, 0, 0),
  });
  const root = controller.getSnapshot().root;

  // Post-present M4 milestones set readyForSplash: true and progressPct: 100
  for (const stage of POST_PRESENT_BOOT_STAGES) {
    controller.update({ bootSummary: { stage } });
    const snap = controller.getSnapshot();
    assert.equal(snap.boot.stage, stage);
    assert.equal(snap.boot.readyForSplash, true);
    assert.equal(snap.boot.progressPct, 100);
  }
  assert.equal(controller.getSnapshot().boot.playabilityState, 'IN_GAME');
  assert.equal(root.querySelector('.pg6-gpu-panel__boot-stage').textContent, 'IN_GAME');
  assert.equal(root.querySelector('.pg6-gpu-panel__boot-ready').textContent, 'Ready');

  // Update with 8-lane HTTPFS concurrency, live audio ring, and OPFS warm-boot hit rate
  controller.update({
    mailboxLanes: 8,
    activeMailboxLanes: 6,
    peakMailboxLanes: 8,
    rangeCacheHitRatePct: 92.5,
    audioStats: {
      outputState: 'active',
      guestRingAttached: true,
      guestFramesDrained: 12_600_000,
      framesDrained: 12_600_000,
      underruns: 0,
      guestUnderruns: 0,
      overruns: 0,
    },
    opfsTelemetry: {
      opfsHits: 18,
      opfsMisses: 2,
      opfsWrites: 2,
      opfsInvalidations: 0,
      bytesFromOpfs: 128 * 1024 * 1024,
      bytesFromNetwork: 16 * 1024 * 1024,
    },
  });

  const snap = controller.getSnapshot();
  assert.equal(snap.mailboxLanes, 8);
  assert.equal(snap.activeMailboxLanes, 6);
  assert.equal(snap.peakMailboxLanes, 8);
  assert.equal(snap.rangeCacheHitRatePct, 92.5);
  assert.equal(snap.guestRingAttached, true);
  assert.equal(snap.guestFramesDrained, 12_600_000);
  assert.equal(snap.audioOutputState, 'active');
  assert.equal(snap.opfsHits, 18);
  assert.equal(snap.opfsMisses, 2);
  assert.equal(snap.opfsHitRatePct, 90);

  // Verify rendered metric cards
  assert.equal(root.querySelector('.pg6-gpu-panel__mailbox-lanes').textContent, '6/8 lanes (peak 8)');
  assert.equal(root.querySelector('.pg6-gpu-panel__audio-ring').textContent, 'Attached');
  assert.equal(root.querySelector('.pg6-gpu-panel__audio-output-state').textContent, 'active');
  assert.equal(root.querySelector('.pg6-gpu-panel__audio-frames-drained').textContent, '12.6M');
  assert.equal(root.querySelector('.pg6-gpu-panel__opfs-hit-rate').textContent, '90.0%');
  assert.equal(root.querySelector('.pg6-gpu-panel__opfs-hits-misses').textContent, '18 / 2');
  assert.equal(root.querySelector('.pg6-gpu-panel__opfs-bytes').textContent, '128.0 MiB');

  // Verify formatSummary() and exportDiagnosticsJson()
  const summaryText = controller.formatSummary();
  assert.match(summaryText, /IN_GAME \(100%\)/);
  assert.match(summaryText, /mailboxLanes=6\/8\(peak=8\)/);
  assert.match(summaryText, /ioCacheHit=92\.5%/);
  assert.match(summaryText, /audio=attached\(12\.6M\)/);
  assert.match(summaryText, /opfsHit=90\.0%/);

  const exportedJson = controller.exportDiagnosticsJson();
  const parsed = JSON.parse(exportedJson);
  assert.equal(parsed.boot.stage, 'IN_GAME');
  assert.equal(parsed.boot.playabilityState, 'IN_GAME');
  assert.deepEqual(parsed.httpfsConcurrency, {
    mailboxLanes: 8,
    activeMailboxLanes: 6,
    peakMailboxLanes: 8,
    rangeCacheHitRatePct: 92.5,
  });
  assert.equal(parsed.audio.outputState, 'active');
  assert.equal(parsed.audio.guestRingAttached, true);
  assert.equal(parsed.audio.guestFramesDrained, 12_600_000);
  assert.equal(parsed.opfs.opfsHits, 18);
  assert.equal(parsed.opfs.opfsMisses, 2);
  assert.equal(parsed.opfs.opfsHitRatePct, 90);

  // Hostile inputs sanitize cleanly across DOM, snapshot, formatSummary, and buildMissingShaderExport
  controller.update({
    mailboxLanes: 'C:\\Users\\alice\\lanes',
    activeMailboxLanes: -1,
    peakMailboxLanes: Number.NaN,
    rangeCacheHitRatePct: '<img src=x onerror=alert(1)>',
    audioStats: {
      outputState: '/home/bob/audio.bin',
      guestRingAttached: 'yes',
      guestFramesDrained: -50,
      rawDump: new Uint8Array([1, 2, 3, 4]),
    },
    opfsTelemetry: {
      opfsHits: -10,
      opfsMisses: 'file:///Users/carol/cache',
      opfsHitRatePct: Infinity,
    },
  });
  const hostileExport = controller.exportDiagnosticsJson();
  assert.doesNotMatch(hostileExport, /alice|bob|carol|audio\.bin|rawDump|<img|file:\/\//i);
  const directExport = buildMissingShaderExport({
    bootSummary: { stage: 'MAIN_MENU' },
    audioStats: { outputState: 'C:\\bad\\state', guestFramesDrained: -9 },
    opfsStats: { opfsHits: 4, opfsMisses: 1, hostPath: '/Users/alice/secret' },
  });
  assert.equal(directExport.audio.outputState, null);
  assert.equal(directExport.opfs.opfsHitRatePct, 80);
  assert.doesNotMatch(JSON.stringify(directExport), /alice|secret|bad/i);
  assert.equal(formatSummary(directExport).includes('opfsHit=80.0%'), true);

  // reset() restores defaults cleanly
  const afterReset = controller.reset();
  assert.equal(afterReset.peakMailboxLanes, null);
  assert.equal(afterReset.guestRingAttached, false);
  assert.equal(afterReset.guestFramesDrained, 0);
  assert.equal(afterReset.opfsHitRatePct, null);
  assert.equal(root.querySelector('.pg6-gpu-panel__audio-ring').textContent, 'Detached');
  assert.equal(root.querySelector('.pg6-gpu-panel__opfs-hit-rate').textContent, '—');

  controller.destroy();
});

test('W376 GpuBootPanelController: skips body DOM writes when collapsed and all DOM writes when hidden, syncing immediately on expand/show', () => {
  const doc = new MockPanelDocument();
  const controller = new GpuBootPanelController({
    container: doc.body,
    documentObj: doc,
    collapsed: true,
    visible: true,
  });
  const root = controller.getSnapshot().root;

  const badgeEl = root.querySelector('.pg6-gpu-panel__summary-badge');
  const stageEl = root.querySelector('.pg6-gpu-panel__boot-stage');
  const framesEl = root.querySelector('.pg6-gpu-panel__frames-presented');
  const lanesEl = root.querySelector('.pg6-gpu-panel__mailbox-lanes');
  const audioRingEl = root.querySelector('.pg6-gpu-panel__audio-ring');
  const audioDrainedEl = root.querySelector('.pg6-gpu-panel__audio-frames-drained');
  const opfsHitEl = root.querySelector('.pg6-gpu-panel__opfs-hit-rate');
  const tbodyEl = root.querySelector('.pg6-gpu-panel__table-body');

  const writesWhileCollapsedBefore = {
    stage: stageEl.textContentWrites,
    frames: framesEl.textContentWrites,
    lanes: lanesEl.textContentWrites,
    audioRing: audioRingEl.textContentWrites,
    audioDrained: audioDrainedEl.textContentWrites,
    opfsHit: opfsHitEl.textContentWrites,
    tbody: tbodyEl.childMutations,
  };

  // Multiple live updates while collapsed: header badge updates, body receives ZERO DOM writes!
  controller.update({
    bootSummary: { stage: 'MAIN_MENU' },
    frameStats: { framesPresented: 300, droppedFrames: 1 },
    mailboxLanes: 8,
    peakMailboxLanes: 8,
    audioStats: { outputState: 'active', guestRingAttached: true, guestFramesDrained: 4_800_000 },
    opfsTelemetry: { opfsHits: 9, opfsMisses: 1 },
    missingShaders: [{ hash: '0x1234', stage: 'fragment', hits: 5 }],
  });

  assert.equal(badgeEl.textContent, 'MAIN_MENU · 100%');
  assert.equal(stageEl.textContentWrites, writesWhileCollapsedBefore.stage);
  assert.equal(framesEl.textContentWrites, writesWhileCollapsedBefore.frames);
  assert.equal(lanesEl.textContentWrites, writesWhileCollapsedBefore.lanes);
  assert.equal(audioRingEl.textContentWrites, writesWhileCollapsedBefore.audioRing);
  assert.equal(audioDrainedEl.textContentWrites, writesWhileCollapsedBefore.audioDrained);
  assert.equal(opfsHitEl.textContentWrites, writesWhileCollapsedBefore.opfsHit);
  assert.equal(tbodyEl.childMutations, writesWhileCollapsedBefore.tbody);

  // Expanding immediately syncs all deferred body elements!
  controller.toggle(false);
  assert.equal(stageEl.textContent, 'MAIN_MENU');
  assert.equal(framesEl.textContent, '300');
  assert.equal(lanesEl.textContent, '8 lanes (peak 8)');
  assert.equal(audioRingEl.textContent, 'Attached');
  assert.equal(audioDrainedEl.textContent, '4.8M');
  assert.equal(opfsHitEl.textContent, '90.0%');
  assert.equal(root.querySelectorAll('.pg6-gpu-panel__shader-row').length, 1);

  // Hiding panel skips ALL DOM writes (both header and body)!
  controller.hide();
  const badgeWritesBeforeHidden = badgeEl.textContentWrites;
  const stageWritesBeforeHidden = stageEl.textContentWrites;
  const opfsWritesBeforeHidden = opfsHitEl.textContentWrites;

  controller.update({
    bootSummary: { stage: 'IN_GAME' },
    opfsTelemetry: { opfsHits: 20, opfsMisses: 0 },
  });
  assert.equal(badgeEl.textContentWrites, badgeWritesBeforeHidden);
  assert.equal(stageEl.textContentWrites, stageWritesBeforeHidden);
  assert.equal(opfsHitEl.textContentWrites, opfsWritesBeforeHidden);

  // Showing panel immediately syncs header AND body!
  controller.show();
  assert.equal(badgeEl.textContent, 'IN_GAME · 100%');
  assert.equal(stageEl.textContent, 'IN_GAME');
  assert.equal(opfsHitEl.textContent, '100.0%');

  controller.destroy();
});
