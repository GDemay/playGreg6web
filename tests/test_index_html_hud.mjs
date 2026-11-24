// Unit & integration tests for W76 Astryx-styled boot diagnostics overlay:
// - Astryx Neutral dark theme self-contained CSS (web/ui/overlay.css) with zero external hotlinks
// - Honest real-vs-smoke boot-mode badge (#boot-mode-badge), boot progress bar/milestones/elapsed,
//   split-module status (#side-module-status), Frame & GPU strip + sanitized validation alert
// - Compact 4-chip capability row (red only when missing), fault-highlighted runtime log,
//   focused action controls, and collapsed "More" disclosure preserving all legacy IDs.

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import { HudController, BOOT_MODE } from '../web/ui/hud_controller.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');
const INDEX_HTML_PATH = path.join(REPO_ROOT, 'web', 'index.html');
const OVERLAY_CSS_PATH = path.join(REPO_ROOT, 'web', 'ui', 'overlay.css');

const indexHtml = fs.readFileSync(INDEX_HTML_PATH, 'utf8');
const overlayCss = fs.readFileSync(OVERLAY_CSS_PATH, 'utf8');

function extractModuleScript(html) {
  const match = /<script\s+type="module">([\s\S]*?)<\/script>/i.exec(html);
  assert.ok(match, 'web/index.html must contain a <script type="module"> block');
  return match[1];
}

function createMockDisplayElement(initial = {}) {
  const attributes = new Map();
  return {
    textContent: initial.textContent ?? '',
    title: initial.title ?? '',
    hidden: initial.hidden ?? false,
    dataset: { ...(initial.dataset || {}) },
    style: { ...(initial.style || {}) },
    children: [],
    setAttribute(name, val) {
      attributes.set(name, String(val));
    },
    getAttribute(name) {
      return attributes.has(name) ? attributes.get(name) : null;
    },
  };
}

test('W76 overlay.css implements self-contained Astryx Neutral dark tokens with COOP/COEP safety', () => {
  assert.match(indexHtml, /<link\s+rel="stylesheet"\s+href="\.\/ui\/overlay\.css"\s*\/?>/);
  assert.match(indexHtml, /data-astryx-theme="neutral"/);

  // Credits Astryx and defines Astryx CSS layers & custom properties.
  assert.match(overlayCss, /Astryx/i);
  assert.match(overlayCss, /astryx\.atmeta\.com/);
  assert.match(overlayCss, /@layer\s+astryx-tokens/);
  assert.match(overlayCss, /\[data-astryx-theme="neutral"\]/);
  assert.match(overlayCss, /--astryx-color-surface:/);
  assert.match(overlayCss, /--astryx-color-text-primary:/);
  assert.match(overlayCss, /--astryx-color-negative:/);
  assert.match(overlayCss, /--astryx-font-sans:/);
  assert.match(overlayCss, /--astryx-font-mono:/);

  // Zero external stylesheet/font hotlinks (COOP: same-origin + COEP: require-corp safe).
  assert.doesNotMatch(overlayCss, /@import\s+url\(/i);
  assert.doesNotMatch(overlayCss, /url\(\s*['"]?https?:/i);
  assert.doesNotMatch(indexHtml, /<link[^>]+href="https?:/i);

  // Accessibility: visible keyboard focus rings and reduced-motion support.
  assert.match(overlayCss, /:focus-visible/);
  assert.match(overlayCss, /prefers-reduced-motion:\s*reduce/);

  // Compact panel stays left-docked (never full-width center-blocking) and hides 0/0 split modules.
  assert.match(overlayCss, /width:\s*min\(3\d\dpx/);
  assert.match(overlayCss, /#side-module-status\[data-total="0"\]\[data-loaded="0"\]\s*\{\s*display:\s*none;/);

  // Fault log highlighting and red-only-when-missing capability chips.
  assert.match(overlayCss, /\.log-entry\[data-fault="true"\]/);
  assert.match(overlayCss, /\.badge\[data-state="err"\]/);
});

test('W76 web/index.html defines focused real-boot sections, accessibility attributes, and collapsed More disclosure', () => {
  const newIds = [
    'boot-mode-badge',
    'boot-stage',
    'boot-milestones',
    'boot-elapsed',
    'boot-progress-bar',
    'boot-progress-fill',
    'side-module-status',
    'metric-frames-presented',
    'gpu-validation-errors',
    'gpu-validation-banner',
    'gpu-validation-latest',
    'metric-missing-shaders',
    'hud-more',
    'hud-more-toggle',
  ];

  for (const id of newIds) {
    assert.match(
      indexHtml,
      new RegExp(`id="${id}"`),
      `web/index.html must define element with id="${id}"`
    );
  }

  // Semantic headings & live regions for accessibility.
  assert.match(indexHtml, /<h1\s+class="hud-brand">playgta6web<\/h1>/);
  assert.match(indexHtml, /id="boot-mode-badge"[^>]*role="status"[^>]*aria-live="polite"/);
  assert.match(indexHtml, /id="hud-status"[^>]*role="status"[^>]*aria-live="polite"/);
  assert.match(indexHtml, /id="boot-progress-bar"[^>]*role="progressbar"/);
  assert.match(indexHtml, /id="gpu-validation-banner"[^>]*role="alert"[^>]*\bhidden\b/);
  assert.match(indexHtml, /id="hud-log"[^>]*role="log"[^>]*aria-live="polite"/);
  assert.match(indexHtml, /id="hud-toggle"[^>]*aria-expanded="true"[^>]*aria-controls="hud-body"/);

  // Compact capability chip row has exactly the 4 core chips (COI/SAB, WebGPU, Wasm64, OPFS).
  const capRowMatch = /<div\s+class="cap-row">([\s\S]*?)<\/div>/i.exec(indexHtml);
  assert.ok(capRowMatch, 'Expected .cap-row for compact capability chips');
  const capRowIds = [...capRowMatch[1].matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]);
  assert.deepEqual(capRowIds, ['cap-coi', 'cap-webgpu', 'cap-wasm64', 'cap-opfs']);

  // Secondary/rarely-needed controls live inside collapsed <details id="hud-more"> (not open by default).
  const moreMatch = /<details\b([^>]*)id="hud-more"([^>]*)>([\s\S]*?)<\/details>/i.exec(indexHtml);
  assert.ok(moreMatch, 'Expected collapsed <details id="hud-more"> section');
  const detailsAttrs = `${moreMatch[1]} ${moreMatch[2]}`;
  assert.doesNotMatch(detailsAttrs, /\bopen\b/i, '#hud-more must be collapsed by default');

  const moreBody = moreMatch[3];
  for (const secondaryId of [
    'hud-refresh-status',
    'srv-manifest',
    'srv-bootset',
    'srv-wasm',
    'srv-shaders',
    'cap-gamepad',
    'metric-gpu-packets',
    'metric-opfs-hits',
    'metric-pointer-lock',
    'metric-gamepad',
    'select-input-profile',
    'btn-clear-cache',
    'btn-export-saves',
    'btn-export-missing-shaders',
    'gpu-boot-panel-root',
  ]) {
    assert.match(
      moreBody,
      new RegExp(`id="${secondaryId}"`),
      `Secondary control #${secondaryId} should be tucked inside collapsed #hud-more`
    );
  }

  // Primary useful actions remain visible outside #hud-more.
  const beforeMore = indexHtml.slice(0, moreMatch.index);
  for (const primaryActionId of [
    'btn-restart-session',
    'btn-export-crash',
    'btn-record-trace',
    'btn-export-trace',
    'btn-smoke-demo',
  ]) {
    assert.match(
      beforeMore,
      new RegExp(`id="${primaryActionId}"`),
      `Primary action #${primaryActionId} must stay outside collapsed #hud-more`
    );
  }
  assert.match(
    beforeMore,
    /id="btn-smoke-demo"[^>]*hud-btn--secondary/,
    'Smoke Demo button must be styled as clearly secondary'
  );
});

test('W76 inline script highlights runtime faults/traps, wires all displayElements, and passes syntax check', () => {
  const moduleScript = extractModuleScript(indexHtml);

  assert.match(moduleScript, /FAULT_LOG_RE/);
  assert.match(moduleScript, /\\\[trap\\\]/);
  assert.match(moduleScript, /guest boot:/);
  assert.match(moduleScript, /entry\.dataset\.fault\s*=\s*['"]true['"]/);
  assert.match(moduleScript, /displayElements:\s*\{[\s\S]*bootModeBadge:\s*bootModeBadgeEl[\s\S]*bootStageEl[\s\S]*framesPresentedEl[\s\S]*missingShadersEl[\s\S]*\}/);
  assert.match(moduleScript, /setBootMode:\s*updateBootModeDisplay/);
  assert.match(moduleScript, /onGpuValidationError:\s*recordGpuValidationErrorDisplay/);
  assert.match(moduleScript, /onSplashFramePresented\s*\(/);

  // Zero hardcoded local host paths or copyrighted game files.
  assert.doesNotMatch(indexHtml, /C:\\\\Users\\\\|\/Users\/|\/home\/|\.rpf|\.xvd|\.xvc/i);
  assert.doesNotMatch(overlayCss, /C:\\\\Users\\\\|\/Users\/|\/home\/|\.rpf|\.xvd|\.xvc/i);

  const check = spawnSync(process.execPath, ['--input-type=module', '--check'], {
    input: moduleScript,
    encoding: 'utf8',
  });
  assert.equal(check.status, 0, `Inline module script failed syntax check:\n${check.stderr}`);
});

test('W76 integration: HudController drives all overlay display elements across real boot, smoke demo, split modules, and GPU errors', () => {
  const displayElements = {
    bootModeBadge: createMockDisplayElement({ textContent: 'Idle', dataset: { bootMode: 'idle', state: 'info' } }),
    sideModuleStatusEl: createMockDisplayElement({ textContent: 'No split modules', dataset: { loaded: '0', total: '0' } }),
    validationCountEl: createMockDisplayElement({ textContent: '0', dataset: { count: '0', state: 'ok' } }),
    validationLatestEl: createMockDisplayElement({ textContent: 'None' }),
    validationBannerEl: createMockDisplayElement({ hidden: true }),
    bootStageEl: createMockDisplayElement({ textContent: 'NOT_STARTED' }),
    bootMilestonesEl: createMockDisplayElement({ textContent: '0/9' }),
    bootProgressBarEl: createMockDisplayElement(),
    bootProgressFillEl: createMockDisplayElement(),
    bootElapsedEl: createMockDisplayElement({ textContent: '0.0s' }),
    framesPresentedEl: createMockDisplayElement({ textContent: '0' }),
    missingShadersEl: createMockDisplayElement({ textContent: '0' }),
  };

  const controller = new HudController({
    displayElements,
    now: () => 1700000000000,
  });

  // 1. Initial idle state.
  assert.equal(displayElements.bootModeBadge.textContent, 'Idle');
  assert.equal(displayElements.bootStageEl.textContent, 'NOT_STARTED');
  assert.equal(displayElements.bootMilestonesEl.textContent, '0/9');
  assert.equal(displayElements.bootElapsedEl.textContent, '0.0s');
  assert.equal(displayElements.framesPresentedEl.textContent, '0');
  assert.equal(displayElements.missingShadersEl.textContent, '0');
  assert.equal(displayElements.validationBannerEl.hidden, true);

  // 2. Real Wasm64 boot progress + split side modules + missing shaders + splash frame.
  controller.setBootMode(BOOT_MODE.REAL_WASM64);
  controller.setSideModules({ sideModuleCount: 4, loadedCount: 2 });
  controller.updateGpuBootTelemetry({
    bootSummary: {
      stage: 'D3D12_CREATE_DEVICE',
      milestonesReached: 7,
      totalMilestones: 9,
      progressPct: 78,
      elapsedMs: 4320,
    },
    frameStats: { framesPresented: 3 },
    missingShaders: [
      { hash: '0x1111', stage: 'vertex', hits: 2 },
      { hash: '0x2222', stage: 'fragment', hits: 5 },
    ],
  });

  assert.equal(displayElements.bootModeBadge.textContent, 'Real Wasm64');
  assert.equal(displayElements.bootModeBadge.dataset.bootMode, 'real-wasm64');
  assert.equal(displayElements.bootStageEl.textContent, 'D3D12_CREATE_DEVICE');
  assert.equal(displayElements.bootMilestonesEl.textContent, '7/9');
  assert.equal(displayElements.bootElapsedEl.textContent, '4.3s');
  assert.equal(displayElements.bootProgressBarEl.getAttribute('aria-valuenow'), '78');
  assert.equal(displayElements.bootProgressFillEl.style.width, '78%');
  assert.equal(displayElements.sideModuleStatusEl.textContent, 'Side modules: 2/4 (50%)');
  assert.equal(displayElements.sideModuleStatusEl.dataset.loaded, '2');
  assert.equal(displayElements.sideModuleStatusEl.dataset.total, '4');
  assert.equal(displayElements.framesPresentedEl.textContent, '3');
  assert.equal(displayElements.missingShadersEl.textContent, '2');
  assert.equal(displayElements.missingShadersEl.dataset.state, 'warn');

  // 3. Splash frame event increments presented frames cleanly.
  controller.onSplashFramePresented({ ok: true, framesPresented: 4 });
  assert.equal(displayElements.framesPresentedEl.textContent, '4');

  // 4. Live WebGPU validation error sanitizes path & deduplicates by seq across hooks.
  controller.onGpuValidationError({
    seq: 7,
    source: 'wgpu_worker',
    type: 'validation',
    message: 'Invalid bind group layout at C:\\Users\\dev\\secret\\shader.wgsl',
  });
  controller.onGpuValidationError({
    seq: 7,
    source: 'wgpu_worker',
    type: 'validation',
    message: 'Invalid bind group layout at C:\\Users\\dev\\secret\\shader.wgsl',
  });
  assert.equal(displayElements.validationCountEl.textContent, '1');
  assert.equal(displayElements.validationBannerEl.hidden, false);
  assert.doesNotMatch(displayElements.validationLatestEl.textContent, /secret|dev/i);
  assert.match(displayElements.validationLatestEl.textContent, /\[REDACTED_PATH\]/);

  // 5. Smoke demo is never labeled as real.
  controller.setBootMode(BOOT_MODE.SMOKE);
  assert.equal(displayElements.bootModeBadge.textContent, 'Smoke Demo');
  assert.equal(displayElements.bootModeBadge.dataset.bootMode, 'smoke');
  assert.equal(displayElements.bootModeBadge.dataset.state, 'warn');

  controller.destroy();
});
